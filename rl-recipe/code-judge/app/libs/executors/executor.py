import subprocess
from dataclasses import dataclass, field
import time
from contextlib import contextmanager
from typing import Any, Generator, Protocol


class ExecuteResult(Protocol):
    success: bool
    cost: float # in seconds


class Executor:
    def execute(self, config: dict[str, Any], stdin: str | None = None, timeout: float | None = None) -> ExecuteResult:
        ...


@dataclass
class ProcessExecuteResult:
    stdout: str
    stderr: str
    exit_code: int
    cost: float # in seconds
    success: bool = field(init=False)

    def __post_init__(self):
        self.success = self.exit_code == 0


class CompileError(Exception):
    pass


TIMEOUT_EXIT_CODE = -101
COMPILE_ERROR_EXIT_CODE = -102


class ProcessExecutor:
    def execute(self, config: dict[str, Any], stdin: str | None = None, timeout: float | None = None) -> ProcessExecuteResult:
        time_start = time.perf_counter()
        try:
            args = config['args']
            std_input = stdin.encode() if stdin else None
            result = subprocess.run(args, shell=False, check=False, capture_output=True, timeout=timeout, input=std_input)
            stdout = result.stdout.decode()
            stderr = result.stderr.decode()
            exit_code = result.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode()
            stderr = e.stderr.decode()
            exit_code = TIMEOUT_EXIT_CODE

        time_end = time.perf_counter()

        return ProcessExecuteResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            cost=time_end - time_start
        )


class ScriptExecutor(ProcessExecutor):
    @contextmanager
    def setup_command(self, script: str) -> Generator[list[str], Any, None]:
        """
        Prepare the command to execute the script
        """
        raise NotImplementedError

    def process_result(self, result: ProcessExecuteResult) -> ProcessExecuteResult:
        return result

    def execute_script(self, script: str, stdin: str | None = None, timeout: float | None = None) -> ProcessExecuteResult:
        with self.setup_command(script) as command:
            return self.process_result(self.execute({'args': command}, stdin=stdin, timeout=timeout))
