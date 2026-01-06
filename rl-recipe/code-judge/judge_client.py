import requests
import math
from typing import Literal
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor


def chunkify(iterable, size):
    """Yield successive chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


@dataclass
class Submission:
    type: Literal['python', 'cpp']
    solution: str
    input: str | None = None
    expected_output: str | None = None


@dataclass
class SubmissionResult:
    sub_id: str
    success: bool
    run_success: bool
    cost: float
    stdout: str | None = None
    stderr: str | None = None
    reason: str = ''


@dataclass
class BatchSubmission:
    type: Literal['batch']
    submissions: list[Submission]


@dataclass
class BatchSubmissionResult:
    sub_id: str
    results: list[SubmissionResult]

    @classmethod
    def from_response(cls, response: dict):
        return cls(
            sub_id=response['sub_id'],
            results=[SubmissionResult(**result) for result in response['results']]
        )


@dataclass
class ServerStatus:
    queue: int
    num_workers: int


def _judge_batch(url: str, submissions: list[Submission], timeout: int = 3600) -> list[SubmissionResult]:
    if not submissions:
        return []

    batch_submission = BatchSubmission(submissions=submissions, type='batch')
    response = requests.post(
        f'{url}/judge/long-batch',
        json=asdict(batch_submission),
        timeout=timeout,
    )
    response.raise_for_status()
    result = BatchSubmissionResult.from_response(response.json())
    return result.results


class JudgeClient:
    def __init__(self, url, *, max_batch_size=1000, max_workers=4):
        self.url = url
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def get_status(self, timeout: int = 10) -> ServerStatus:
        response = requests.get(
            f'{self.url}/status',
            timeout=timeout,
        )
        response.raise_for_status()
        return ServerStatus(**response.json())

    def judge(self, submissions: list[Submission]) -> list[SubmissionResult]:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            return self._judge(executor, submissions)

    def _judge(
            self,
            executor: ProcessPoolExecutor,
            submissions: list[Submission],
    ) -> list[SubmissionResult]:
        if not submissions:
            return []

        n_sumissions = len(submissions)
        sub_ids = list(range(len(submissions)))
        results = {}

        while submissions:
            num_batches = max(math.ceil(len(submissions) / self.max_batch_size), self.max_workers)
            batch_size = math.ceil(len(submissions) / num_batches)

            print(f'Judging {len(submissions)} submissions in {num_batches} batches of {batch_size}.')

            pending_chunks = list(
                chunkify(
                    [(sub, id) for sub, id in zip(submissions, sub_ids)],
                    batch_size)
            )
            futures = [
                executor.submit(_judge_batch, self.url, [c[0] for c in chunk])
                for chunk in pending_chunks
            ]
            queue_timeouts = []
            for i, future in enumerate(futures):
                pending_chunk = pending_chunks[i]
                result = future.result()
                for (sub, sub_id), sub_result in zip(pending_chunk, result):
                    if sub_result.reason == 'queue_timeout':
                        # Retry the submission later
                        queue_timeouts.append((sub_id, sub))
                    else:
                        results[sub_id] = sub_result
                print(f'Processed {len(results)} submissions, Got {len(queue_timeouts)} timeouts in total.')

            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]

        return [results[i] for i in range(n_sumissions)]
