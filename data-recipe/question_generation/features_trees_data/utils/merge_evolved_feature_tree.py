#!/usr/bin/env python3
"""Merge evolved feature expansions into a single feature tree JSON.

Inputs:
- --base-tree: base tree JSON (typically from `merge_extracted_features_tree.py`)
- --evol-jsonl: JSONL from `evolve_feature_tree_epicoder.py`

Output:
- merged tree JSON (base + all expanded features), with recursive list de-duplication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union


JSONValue = Union[Dict[str, "JSONValue"], List["JSONValue"], str, int, float, bool, None]


def _dedup_strings_preserve_order(items: List[Any]) -> List[str]:
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


def _recursive_dedup(node: JSONValue) -> JSONValue:
    if isinstance(node, dict):
        return {k: _recursive_dedup(v) for k, v in node.items()}
    if isinstance(node, list):
        # Lists are leaf features.
        return _dedup_strings_preserve_order(node)
    return node


def _list_to_dict(lst: List[Any]) -> Dict[str, JSONValue]:
    out: Dict[str, JSONValue] = {}
    for s in _dedup_strings_preserve_order(lst):
        out[s] = []
    return out


def merge_dicts(a: JSONValue, b: JSONValue) -> JSONValue:
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
            if k not in out:
                out[k] = v
            else:
                out[k] = merge_dicts(out[k], v)
        return out

    if isinstance(a, dict) and isinstance(b, list):
        return merge_dicts(a, _list_to_dict(b))
    if isinstance(a, list) and isinstance(b, dict):
        return merge_dicts(_list_to_dict(a), b)

    # Fallback: prefer dict over list, list over scalar.
    if isinstance(a, dict) and not isinstance(b, dict):
        return a
    if isinstance(b, dict) and not isinstance(a, dict):
        return b
    if isinstance(a, list) and not isinstance(b, list):
        return a
    if isinstance(b, list) and not isinstance(a, list):
        return b
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tree", type=str, required=True, help="Base feature tree JSON path.")
    parser.add_argument("--evol-jsonl", type=str, required=True, help="Evolution results JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Output merged tree JSON path.")
    args = parser.parse_args()

    base_path = Path(args.base_tree).expanduser()
    evol_path = Path(args.evol_jsonl).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged: JSONValue = json.load(base_path.open("r", encoding="utf-8"))
    if not isinstance(merged, dict):
        raise SystemExit(f"Expected dict base tree at {base_path}, got {type(merged)}")

    n_used = 0
    n_skipped = 0
    with evol_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expanded = obj.get("expanded_feature")
            if not isinstance(expanded, dict) or "parse_error_str" in expanded:
                n_skipped += 1
                continue
            merged = merge_dicts(merged, expanded)
            n_used += 1

    merged = _recursive_dedup(merged)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged evol expansions: used={n_used} skipped={n_skipped} -> {out_path}")


if __name__ == "__main__":
    main()

