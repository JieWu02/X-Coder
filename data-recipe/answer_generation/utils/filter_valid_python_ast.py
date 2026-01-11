#!/usr/bin/env python3
"""
Filter JSONL records to keep only those with exactly one complete Python code block.
Validates the code block via ast.parse and optional token count thresholds.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PY_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def count_tokens(code: str) -> int:
    return len(code.split())


def extract_python_block(text: str) -> Tuple[Optional[str], int]:
    matches = PY_BLOCK_RE.findall(text)
    if not matches:
        return None, 0
    if len(matches) != 1:
        return None, len(matches)
    return matches[0].strip(), 1


def get_text_from_record(record: Dict, field_name: Optional[str], fields: List[str]) -> Optional[str]:
    if field_name:
        value = record.get(field_name)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and "generated_answer" in value:
            return value.get("generated_answer")
        return None

    for key in fields:
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and "generated_answer" in value:
            return value.get("generated_answer")
    return None


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except ValueError:
        return False


def filter_records(
    input_file: Path,
    output_file: Path,
    field_name: Optional[str],
    min_tokens: Optional[int],
    max_tokens: Optional[int],
) -> int:
    fields = [
        "generated_answer",
        "answer",
        "response",
        "content",
        "message",
        "output",
    ]

    stats = {
        "total": 0,
        "kept": 0,
        "missing_text": 0,
        "invalid_blocks": 0,
        "invalid_ast": 0,
        "token_filtered": 0,
    }

    with input_file.open("r", encoding="utf-8") as infile, output_file.open("w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["total"] += 1

            text = get_text_from_record(record, field_name, fields)
            if not text:
                # Fall back to extracted_code if present
                extracted = record.get("extracted_code")
                if isinstance(extracted, str) and extracted.strip():
                    code = extracted.strip()
                    python_blocks = 1
                else:
                    stats["missing_text"] += 1
                    continue
            else:
                code, python_blocks = extract_python_block(text)
                if code is None or python_blocks != 1:
                    stats["invalid_blocks"] += 1
                    continue

            if not is_valid_python(code):
                stats["invalid_ast"] += 1
                continue

            token_count = count_tokens(code)
            if min_tokens is not None and token_count < min_tokens:
                stats["token_filtered"] += 1
                continue
            if max_tokens is not None and token_count > max_tokens:
                stats["token_filtered"] += 1
                continue

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            if line_num % 1000 == 0:
                print(f"Processed {line_num} records...")

    print("Filtering complete")
    print(f"Total: {stats['total']}")
    print(f"Kept: {stats['kept']}")
    print(f"Missing text: {stats['missing_text']}")
    print(f"Invalid blocks: {stats['invalid_blocks']}")
    print(f"Invalid AST: {stats['invalid_ast']}")
    print(f"Token filtered: {stats['token_filtered']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter JSONL records with exactly one valid Python code block (AST-checked)."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--field",
        default=None,
        help="Field to inspect for code blocks (default: auto)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Minimum token count (whitespace-based) for the code block",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token count (whitespace-based) for the code block",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1
    output_path = Path(args.output)
    return filter_records(input_path, output_path, args.field, args.min_tokens, args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
