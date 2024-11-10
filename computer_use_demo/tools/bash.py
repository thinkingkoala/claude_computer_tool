import asyncio
import os
import platform
import subprocess
from typing import ClassVar, Literal

from anthropic.types.beta import BetaToolBash20241022Param

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult


class _CommandSession:
    """A session of a command shell (cmd.exe on Windows, bash on Unix)."""

    _started: bool
    _process: asyncio.subprocess.Process

    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False
        # Use appropriate shell based on platform
        self.command = "cmd.exe" if platform.system() == "Windows" else "/bin/bash"
        # Different echo syntax for Windows
        self._sentinel_command = f"echo {self._sentinel}" if platform.system() == "Windows" else f"echo '{self._sentinel}'"

    async def start(self):
        if self._started:
            return

        # Windows-specific process creation
        if platform.system() == "Windows":
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Remove Unix-specific option
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                preexec_fn=os.setsid,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self._started = True

    def stop(self):
        """Terminate the command shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        if platform.system() == "Windows":
            import signal
            os.kill(self._process.pid, signal.CTRL_BREAK_EVENT)
        else:
            self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            shell_name = "cmd.exe" if platform.system() == "Windows" else "bash"
            return ToolResult(
                system="tool must be restarted",
                error=f"{shell_name} has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: command has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # Add appropriate line endings for Windows
        line_ending = "\r\n" if platform.system() == "Windows" else "\n"
        
        # send command to the process
        full_command = f"{command}{line_ending}{self._sentinel_command}{line_ending}"
        self._process.stdin.write(full_command.encode())
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: command has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        # Normalize line endings
        output = output.replace('\r\n', '\n')
        if output.endswith("\n"):
            output = output[:-1]

        error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        error = error.replace('\r\n', '\n')
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class CommandTool(BaseAnthropicTool):
    """
    A tool that allows the agent to run shell commands (cmd.exe on Windows, bash on Unix).
    The tool parameters are defined by Anthropic and are not editable.
    """

    _session: _CommandSession | None
    name: ClassVar[Literal["bash"]] = "bash"  # Keep the name "bash" for compatibility
    api_type: ClassVar[Literal["bash_20241022"]] = "bash_20241022"

    def __init__(self):
        self._session = None
        super().__init__()

    async def __call__(
        self, command: str | None = None, restart: bool = False, **kwargs
    ):
        if restart:
            if self._session:
                self._session.stop()
            self._session = _CommandSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _CommandSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")

    def to_params(self) -> BetaToolBash20241022Param:
        return {
            "type": self.api_type,
            "name": self.name,
        }