from contextlib import contextmanager
import tempfile
from typing import Any, Generator
from app.libs.executors.executor import (
    COMPILE_ERROR_EXIT_CODE, TIMEOUT_EXIT_CODE,
    ProcessExecuteResult, ScriptExecutor, CompileError
)


RESOURCE_LIMIT_TEMPLATE = """
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

static void handler(int sig) {{
    printf("Time limit exceeded\\n");
    kill(getpid(), SIGKILL);
    exit({TIMEOUT_EXIT_CODE});
}}

class ResourceLimit {{
public:
    ResourceLimit(int timeout, int memory_limit) {{
        struct rlimit rlim;
        if (timeout > 0) {{
            getrlimit(RLIMIT_CPU, &rlim);
            rlim.rlim_cur = timeout;
            setrlimit(RLIMIT_CPU, &rlim);
        }}
        if (memory_limit > 0) {{
            getrlimit(RLIMIT_AS, &rlim);
            rlim.rlim_cur = memory_limit;
            setrlimit(RLIMIT_AS, &rlim);
        }}
        getrlimit(RLIMIT_CORE, &rlim);
        rlim.rlim_cur = 0;
        setrlimit(RLIMIT_CORE, &rlim);

        alarm(timeout);
        signal(SIGALRM, handler);
    }}
}};

ResourceLimit resource_limit = ResourceLimit({timeout}, {memory_limit});
""".strip()


class CppExecutor(ScriptExecutor):
    def __init__(self, compiler_path: str, timeout: int = None, memory_limit: int = None):
        self.compiler_path = compiler_path
        self.timeout = timeout
        self.memory_limit = memory_limit

    @contextmanager
    def setup_command(self, script: str) -> Generator[list[str], Any, None]:
        with tempfile.TemporaryDirectory() as tmp_path:
            source_path = f"{tmp_path}/source.cpp"
            resource_limit_path = f"{tmp_path}/resource_limit.h"
            exec_path = f"{tmp_path}/run"
            with open(resource_limit_path, "w") as f:
                f.write(RESOURCE_LIMIT_TEMPLATE.format(
                    timeout=self.timeout or 0,
                    memory_limit=self.memory_limit or 0,
                    TIMEOUT_EXIT_CODE=TIMEOUT_EXIT_CODE)
                )
            with open(source_path, "w") as f:
                f.write('#include "resource_limit.h"\n')
                f.write(script)
            result = self.execute(
                {'args': [self.compiler_path,  "-O2", source_path,  "-o", exec_path]},
                timeout=self.timeout or None
            )
            if not result.success:
                raise CompileError(result.stderr)
            yield [exec_path]

    def execute_script(self, script, stdin=None, timeout=None):
        try:
            return super().execute_script(script, stdin, timeout)
        except CompileError as e:
            return ProcessExecuteResult(stdout='', stderr=str(e), exit_code=COMPILE_ERROR_EXIT_CODE, cost=0)
