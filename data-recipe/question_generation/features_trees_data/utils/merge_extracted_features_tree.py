#!/usr/bin/env python3
"""Merge extracted feature JSONLs into a single feature tree JSON.

This is part of the "extract -> evol -> merge -> sample" pipeline for building
`question_generation/features_trees_data/feature_config_sampled.jsonl`.

Input:
- JSONL produced by `extract_features_epicoder.py`

Output:
- A merged feature tree JSON (dict at root).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


JSONValue = Union[Dict[str, "JSONValue"], List["JSONValue"], str, int, float, bool, None]


_SKIP_KEYS = {
    # Free-form fields that aren't useful for sampling a feature tree.
    "summary",
    "purpose",
    "issues & bugs",
    "potential scenario",
    "parse_error_str",
}


def _dedup_strings_preserve_order(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if not isinstance(x, str):
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _list_to_dict(lst: List[Any]) -> Dict[str, JSONValue]:
    # Convert leaf list -> dict leafs with empty children. This matches the mixed
    # representation used in EpiCoder merged trees, and makes merging stable.
    out: Dict[str, JSONValue] = {}
    for s in _dedup_strings_preserve_order(lst):
        out[s] = []
    return out


def _merge(a: JSONValue, b: JSONValue) -> JSONValue:
    if a is None:
        return b
    if b is None:
        return a

    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    if isinstance(a, list) and isinstance(b, list):
        return _dedup_strings_preserve_order(list(a) + list(b))

    if isinstance(a, dict) and isinstance(b, dict):
        out: Dict[str, JSONValue] = dict(a)
        for k, v in b.items():
            if k in _SKIP_KEYS:
                continue
            if k not in out:
                out[k] = v
            else:
                out[k] = _merge(out[k], v)
        return out

    # Mixed types: normalize lists into dicts so we don't lose structure.
    if isinstance(a, dict) and isinstance(b, list):
        return _merge(a, _list_to_dict(b))
    if isinstance(a, list) and isinstance(b, dict):
        return _merge(_list_to_dict(a), b)

    # Fallback: keep the more structured side if possible.
    if isinstance(a, dict) and not isinstance(b, dict):
        return a
    if isinstance(b, dict) and not isinstance(a, dict):
        return b
    if isinstance(a, list) and not isinstance(b, list):
        return a
    if isinstance(b, list) and not isinstance(a, list):
        return b
    return a


def _recursive_dedup(node: JSONValue) -> JSONValue:
    if isinstance(node, dict):
        return {k: _recursive_dedup(v) for k, v in node.items() if k not in _SKIP_KEYS}
    if isinstance(node, list):
        return _dedup_strings_preserve_order(node)
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Extracted feature JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Output merged tree JSON path.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, JSONValue] = {}
    n = 0
    n_skipped = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            features = obj.get("features")
            if not isinstance(features, dict) or "parse_error_str" in features:
                n_skipped += 1
                continue
            # Drop non-tree-ish fields early.
            features = {k: v for k, v in features.items() if k not in _SKIP_KEYS}
            merged = _merge(merged, features)  # type: ignore[arg-type]
            n += 1

    merged = _recursive_dedup(merged)  # type: ignore[assignment]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {n} records (skipped {n_skipped}) -> {output_path}")


if __name__ == "__main__":
    main()
