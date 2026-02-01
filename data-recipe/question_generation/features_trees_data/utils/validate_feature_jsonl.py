#!/usr/bin/env python3
"""Validate a feature JSONL for X-Coder question generation.

This is a lightweight offline sanity-check:
- JSONL parses
- Each record contains `features` (dict)
- `leaf_count` matches recomputed count (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union


JSONValue = Union[Dict[str, "JSONValue"], List["JSONValue"], str, int, float, bool, None]


def count_leaf_features(node: JSONValue) -> int:
    if isinstance(node, dict):
        total = 0
        for v in node.values():
            if isinstance(v, list):
                total += sum(1 for x in v if isinstance(x, str))
            elif isinstance(v, dict):
                total += count_leaf_features(v)
        return total
    if isinstance(node, list):
        return sum(1 for x in node if isinstance(x, str))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Path to feature JSONL.")
    parser.add_argument("--min-leaf-count", type=int, default=0, help="Fail if any record has leaf_count < this value.")
    parser.add_argument(
        "--check-leaf-count-field",
        action="store_true",
        help="If set, validate that each record's `leaf_count` equals the recomputed value.",
    )
    args = parser.parse_args()

    path = Path(args.input).expanduser()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    leaf_counts: List[int] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[FAIL] line {line_num}: JSON decode error: {exc}")
                bad += 1
                continue
            if not isinstance(rec, dict):
                print(f"[FAIL] line {line_num}: record is not an object")
                bad += 1
                continue
            features = rec.get("features")
            if not isinstance(features, dict):
                print(f"[FAIL] line {line_num}: missing/invalid `features` (expected object)")
                bad += 1
                continue
            leaf = count_leaf_features(features)
            leaf_counts.append(leaf)

            if leaf < args.min_leaf_count:
                print(f"[FAIL] line {line_num}: leaf_count {leaf} < {args.min_leaf_count}")
                bad += 1
                continue

            if args.check_leaf_count_field:
                if "leaf_count" not in rec:
                    print(f"[FAIL] line {line_num}: missing `leaf_count` field")
                    bad += 1
                    continue
                if rec["leaf_count"] != leaf:
                    print(f"[FAIL] line {line_num}: leaf_count field {rec['leaf_count']} != recomputed {leaf}")
                    bad += 1
                    continue

    if not leaf_counts:
        raise SystemExit("No valid records found.")

    leaf_sorted = sorted(leaf_counts)
    p50 = leaf_sorted[len(leaf_sorted) // 2]
    print(
        f"OK: {len(leaf_counts)} records | "
        f"leaf_count min={leaf_sorted[0]} p50={p50} max={leaf_sorted[-1]} | "
        f"failures={bad}"
    )
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

