#!/usr/bin/env python3
"""
Concurrent answer generation using direct SGLang chat completions.

This is a lightweight alternative to the batch API workflow. It focuses on
robust retries and simple concurrency while staying model-agnostic.
"""

import argparse
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import requests


_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def sort_jsonl_file(file_path: str) -> None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        lines.sort(key=lambda x: x.get("question_index", float("inf")))
        with open(file_path, "w", encoding="utf-8") as f:
            for record in lines:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(f"Sorted output file: {file_path} ({len(lines)} records)")
    except FileNotFoundError:
        print(f"Output file not found: {file_path}")
    except Exception as exc:
        print(f"Failed to sort output file: {exc}")


def load_questions_from_jsonl(file_path: str) -> List[Dict]:
    questions: List[Dict] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Warning: JSON parse error at line {line_num}: {exc}")
                    continue

                question_content = None
                question_index = record.get("question_index", line_num - 1)
                for field in ("question", "problem_statement", "task", "content"):
                    if field in record:
                        question_content = record[field]
                        break
                if question_content is None:
                    print(f"Warning: missing question content at line {line_num}")
                    continue
                for idx_field in ("question_index", "config_index", "index", "id"):
                    if idx_field in record:
                        question_index = record[idx_field]
                        break

                questions.append(
                    {
                        "question_index": question_index,
                        "question_content": question_content,
                        "original_record": record,
                    }
                )
        print(f"Loaded {len(questions)} questions")
    except FileNotFoundError:
        print(f"Input file not found: {file_path}")
    except Exception as exc:
        print(f"Failed to read input file: {exc}")
    return questions


def generate_answer_prompt(question_content: str) -> str:
    return (
        "Now that you are a code expert, I have provided you with the QUESTION. "
        "Complete the problem with awesome code logic and give a richly commented "
        "analysis in the code of your answer. Include the necessary packages. Give "
        "out code implementation. Enclose the python code with ```python and ```\n"
        f"- QUESTION {question_content} -\n"
    )


def sglang_call(
    prompt: str,
    worker_ip: str,
    port: int,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    request_timeout: float,
) -> Dict:
    url = f"http://{worker_ip}:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional competitive programming expert and code implementer.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer None"}
    session = _get_session()
    response = session.post(url, headers=headers, json=payload, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def extract_python_code(content: str) -> str:
    if "```python" not in content:
        return ""
    start_idx = content.find("```python")
    end_idx = content.find("```", start_idx + 9)
    if end_idx == -1:
        return ""
    return content[start_idx + 9 : end_idx].strip()


def process_question_once(
    question_data: Dict,
    worker_ip: str,
    port: int,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    request_timeout: float,
    verbose: bool = False,
) -> Dict:
    question_index = question_data["question_index"]
    question_content = question_data["question_content"]
    prompt = generate_answer_prompt(question_content)
    if verbose:
        print(f"[Q{question_index}] prompt length: {len(prompt)}")
    response_json = sglang_call(
        prompt,
        worker_ip,
        port,
        model_name,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        request_timeout,
    )
    choices = response_json.get("choices")
    if not choices:
        raise ValueError(f"Missing choices in response: {response_json}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    if not content:
        raise ValueError(f"Missing message.content in response: {response_json}")
    python_code = extract_python_code(content)
    return {
        "question_index": question_index,
        "question_content": question_content,
        "generated_answer": content,
        "extracted_code": python_code,
        "original_record": question_data["original_record"],
        "generation_metadata": {
            "model_name": model_name,
            "worker_ip": worker_ip,
            "timestamp": time.time(),
        },
    }


def process_question_with_retry(
    question_data: Dict,
    worker_ip: str,
    port: int,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    request_timeout: float,
    max_retries: int,
    retry_delay: float,
    verbose: bool,
) -> Tuple[Dict, bool, int, float]:
    attempts = 0
    start_time = time.time()
    last_error: Optional[str] = None
    while attempts <= max_retries:
        attempts += 1
        try:
            result = process_question_once(
                question_data,
                worker_ip,
                port,
                model_name,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                request_timeout,
                verbose,
            )
            latency = time.time() - start_time
            return result, True, attempts, latency
        except Exception as exc:
            last_error = str(exc)
            if verbose:
                print(
                    f"[Q{question_data['question_index']}] request failed "
                    f"({worker_ip}:{port}) attempt {attempts}: {exc}"
                )
            if attempts > max_retries:
                break
            time.sleep(retry_delay * attempts)

    latency = time.time() - start_time
    error_result = {
        "question_index": question_data["question_index"],
        "error": last_error or "unknown error",
        "question_content": question_data["question_content"],
        "original_record": question_data["original_record"],
        "generation_metadata": {
            "model_name": model_name,
            "worker_ip": worker_ip,
            "timestamp": time.time(),
            "failed_attempts": attempts,
        },
    }
    return error_result, False, attempts, latency


def select_questions(questions: List[Dict], begin_idx: int, end_idx: int) -> List[Dict]:
    questions.sort(key=lambda q: q["question_index"])
    if end_idx == -1:
        selected = [q for q in questions if q["question_index"] >= begin_idx]
    else:
        selected = [q for q in questions if begin_idx <= q["question_index"] <= end_idx]
    if not selected:
        total = len(questions)
        if end_idx == -1:
            selected = questions[begin_idx:] if begin_idx < total else []
        else:
            slice_end = min(end_idx + 1, total)
            if begin_idx < slice_end:
                selected = questions[begin_idx:slice_end]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent answer generation via SGLang")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--worker-ips", nargs="+", default=["localhost"], help="Worker IP list")
    parser.add_argument("--worker-port", type=int, default=30001, help="SGLang port")
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528",
        help="Model name",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32768, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p")
    parser.add_argument("--begin-idx", type=int, default=0, help="Start question_index")
    parser.add_argument("--end-idx", type=int, default=-1, help="End question_index")
    parser.add_argument("--concurrency", type=int, default=16, help="Thread count")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Base retry delay (s)")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Request timeout (s)")
    parser.add_argument("--log-every", type=int, default=10, help="Progress log frequency")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    questions = load_questions_from_jsonl(args.input_file)
    if not questions:
        print("No questions loaded. Exiting.")
        return

    selected_questions = select_questions(questions, args.begin_idx, args.end_idx)
    if not selected_questions:
        print("No questions in the specified range.")
        return

    first_idx = selected_questions[0]["question_index"]
    last_idx = selected_questions[-1]["question_index"]
    print(f"Processing {len(selected_questions)} questions ({first_idx} - {last_idx})")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8"):
        pass

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(selected_questions)
    success_count = 0
    failure_count = 0
    start_time = time.time()

    worker_ips = list(args.worker_ips)
    if not worker_ips:
        print("No worker IPs provided.")
        return

    def select_worker(i: int) -> str:
        return worker_ips[i % len(worker_ips)]

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor, open(
        args.output_file, "a", encoding="utf-8"
    ) as output_file:
        future_map = {}
        for idx, question_data in enumerate(selected_questions):
            worker_ip = select_worker(idx)
            future = executor.submit(
                process_question_with_retry,
                question_data,
                worker_ip,
                args.worker_port,
                args.model_name,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.min_p,
                args.request_timeout,
                args.max_retries,
                args.retry_delay,
                args.verbose,
            )
            future_map[future] = question_data["question_index"]

        for idx, future in enumerate(as_completed(future_map), 1):
            question_index = future_map[future]
            try:
                result, success, attempts, latency = future.result()
            except Exception as exc:
                failure_count += 1
                result = {
                    "question_index": question_index,
                    "error": str(exc),
                    "generation_metadata": {
                        "model_name": args.model_name,
                        "worker_ip": "unknown",
                        "timestamp": time.time(),
                    },
                }
                success = False
                attempts = args.max_retries + 1
                latency = float("nan")
            json.dump(result, output_file, ensure_ascii=False)
            output_file.write("\n")
            output_file.flush()

            if success:
                success_count += 1
            else:
                failure_count += 1

            if idx % args.log_every == 0 or idx == total:
                elapsed = time.time() - start_time
                throughput = success_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {idx}/{total} | ok {success_count} | fail {failure_count} | "
                    f"last {question_index} | attempts {attempts} | latency {latency:.1f}s | "
                    f"throughput {throughput:.2f} req/s"
                )

    print(
        f"Done: ok {success_count} | fail {failure_count} | "
        f"elapsed {time.time() - start_time:.1f}s"
    )
    sort_jsonl_file(args.output_file)


if __name__ == "__main__":
    main()
