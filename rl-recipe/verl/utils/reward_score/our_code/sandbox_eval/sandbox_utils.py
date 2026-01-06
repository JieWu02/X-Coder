# Copyright (2025) critic-rl Authors 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.
import asyncio
import logging
from typing import Union

from sandbox_fusion import (
    EvalResult,
    RunCodeRequest,
    RunCodeResponse,
    SubmitRequest,
    TestConfig,
    run_code_async,
    submit_async,
)
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .eval_utils import desanitize

logger = logging.getLogger(__name__)
TIMEOUT = 320
RUN_TIMEOUT = 30
NUM_RETRIES = 5


def get_submit_fn(dataset_name: str):
    if dataset_name == "code_contests":

        def submit_fn(example, response, test_field):
            if example[test_field].startswith("assert"):
                return RunCodeRequest(
                    **{
                        "code": desanitize(response).strip()
                        + "\n"
                        + "\n".join(example[test_field]),
                        "language": "python",
                        "run_timeout": RUN_TIMEOUT,
                    }
                )

            provided_data = {
                "test": example[test_field],
            }

            req = SubmitRequest(
                dataset="code_contests",
                id=0,
                config=TestConfig(
                    language="python",
                    dataset_type="CommonOJDataset",
                    provided_data=provided_data,
                    extra={"run_all_cases": True},
                    run_timeout=RUN_TIMEOUT,
                ),
                completion=response,
            )

            return req

    elif dataset_name == "livecodebench":

        def submit_fn(example, response, test_field):
            provided_data = {
                k: example[k] for k in ["id", "content", "labels", test_field]
            }
            req = SubmitRequest(
                dataset="dataset_id",
                id=0,
                config=TestConfig(
                    dataset_type="LiveCodeBenchDataset",
                    provided_data=provided_data,
                    run_timeout=RUN_TIMEOUT,
                ),
                completion=response,
            )

            return req

    elif dataset_name == "mbppplus":

        def submit_fn(example, response, test_field):
            if isinstance(example[test_field], list):
                ut = "\n".join(example[test_field])
            else:
                ut = example[test_field]
            req = RunCodeRequest(
                **{
                    "code": desanitize(response).strip() + "\n" + ut,
                    "language": "python",
                    "run_timeout": RUN_TIMEOUT,
                }
            )

            return req

    else:
        raise NotImplementedError(f"dataset {dataset_name} not supported")
    return submit_fn


def on_retry_error(s):
    e = s.outcome.exception()
    logger.error(f"give up requesting sandbox. error: {e}")
    raise e


def before_retry_sleep(s):
    logger.warning(
        f"error requesting sandbox for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}"
    )


@retry(
    wait=wait_exponential_jitter(),
    stop=stop_after_attempt(NUM_RETRIES),
    before_sleep=before_retry_sleep,
    retry_error_callback=on_retry_error,
)
async def submit_to_sandbox(
    request: Union[RunCodeRequest, SubmitRequest], semaphore: asyncio.Semaphore
) -> Union[RunCodeResponse, EvalResult]:
    fn = run_code_async if isinstance(request, RunCodeRequest) else submit_async
    async with semaphore:
        resp = await fn(request, client_timeout=TIMEOUT)
    return resp
