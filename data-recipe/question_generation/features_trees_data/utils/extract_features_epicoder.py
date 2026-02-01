#!/usr/bin/env python3
"""Extract competitive-programming feature annotations from (problem, solution) pairs.

This is the "extract" stage used to build the feature tree behind `feature_config_sampled.jsonl`.
It mirrors the logic in EpiCoder `extract/extract_features.py`, but is packaged inside
X-Coder for reproducibility.

Input:
- JSONL/JSON records with at least a problem field (default: `question`)
- Optional solution field (default: `solutions`) which can be:
  - a Python-list string (TACO-style), or
  - a plain code string

Output:
- JSONL with fields: idx, problem, original_code, features
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from aoai_client import chat_completion


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    if path.suffix.lower() == ".json":
        obj = json.load(path.open("r", encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        raise ValueError("Expected a JSON array at root.")
    raise ValueError(f"Unsupported input format: {path} (expected .jsonl or .json)")


def _extract_code(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        # Common case: TACO stores solutions as a stringified Python list.
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                return parsed[0]
        except Exception:
            pass
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    return ""


def _parse_json_between_tags(text: str) -> Dict[str, Any]:
    start = text.find("<begin>")
    end = text.find("<end>")
    if start == -1 or end == -1 or end <= start:
        return {"parse_error_str": text}
    json_str = text[start + len("<begin>") : end].strip()
    json_str = json_str.replace("\n", "").replace("\r", "")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"parse_error_str": json_str}


def _load_prompt(prompt_file: Path) -> str:
    return prompt_file.read_text(encoding="utf-8")


def _load_existing_indices(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    seen: Set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("idx"), int):
                seen.add(int(obj["idx"]))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Input dataset (.jsonl or .json).")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Prompt template file. Defaults to feature_tree_prompts/extract_prompt13.txt",
    )
    parser.add_argument("--problem-field", type=str, default="question", help="Field name for problem text.")
    parser.add_argument("--solution-field", type=str, default="solutions", help="Field name for solution/code text.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls.")
    parser.add_argument("--resume", action="store_true", help="Skip indices already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and exit (no API call).")

    # AOAI params
    parser.add_argument("--deployment", type=str, default=None, help="Override AZURE_OPENAI_DEPLOYMENT.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=None)

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    qg_root = script_dir.parents[1]  # .../question_generation
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser()
    else:
        prompt_path = qg_root / "feature_tree_prompts" / "extract_prompt13.txt"
    prompt_template = _load_prompt(prompt_path)

    data = _load_json_or_jsonl(input_path)
    start = max(0, args.start)
    end = len(data) if args.end is None else max(start, min(args.end, len(data)))

    seen = _load_existing_indices(output_path) if args.resume else set()

    for idx in range(start, end):
        if idx in seen:
            continue
        item = data[idx]
        problem = str(item.get(args.problem_field, ""))
        code = _extract_code(item.get(args.solution_field))

        prompt = prompt_template.replace("{problem}", problem).replace("{source_code}", code)

        if args.dry_run:
            print(prompt)
            return

        content = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            deployment=args.deployment,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )
        features = _parse_json_between_tags(content)

        rec = {
            "idx": idx,
            "problem": problem,
            "original_code": code,
            "features": features,
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. Wrote to {output_path}")


if __name__ == "__main__":
    main()
