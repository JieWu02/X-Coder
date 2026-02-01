#!/usr/bin/env python3
"""
Sample X-Coder question-generation feature configs (JSONL) from a merged feature tree.

Why this exists:
- In this repo, `features_trees_data/feature_config_sampled.jsonl` is tracked by Git LFS. Without `git-lfs`,
  you will only have a small pointer file, not the real data.
- This script provides a reproducible way to generate a compatible feature JSONL locally.

Notes:
- This script is intentionally offline (no LLM/API calls). It only samples subtrees from a
  pre-built merged feature tree produced by EpiCoder (extract + evol + merge).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


JSONValue = Union[Dict[str, "JSONValue"], List["JSONValue"], str, int, float, bool, None]


_SKIP_KEYS = {
    # Remove language tags so the downstream prompt focuses on algorithmic concepts.
    "programming language",
    "programming_language",
}


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            head = [next(f, "") for _ in range(3)]
    except OSError:
        return False
    joined = "".join(head)
    return "git-lfs.github.com/spec/v1" in joined and "oid sha256:" in joined


def _find_candidate_epicoder_tree(epicoder_root: Path) -> Optional[Path]:
    """Pick a reasonable default merged feature tree from an EpiCoder checkout.

    We prefer `*__merged_fea.json` when present (this is the artifact used in our large-scale run),
    and fall back to other merged tree variants.
    """
    search_root = epicoder_root / "output" / "feature_evol"
    if not search_root.exists():
        return None

    patterns = [
        "*__merged_fea.json",
        "*_merged_fea.json",
        "*__merged_fea_ori.json",
        "*_merged_fea_ori.json",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(search_root.rglob(pat))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    # Prefer specific suffixes, then break ties by size.
    def pref_key(p: Path) -> tuple:
        name = p.name
        if name.endswith("__merged_fea.json"):
            rank = 0
        elif name.endswith("_merged_fea.json"):
            rank = 1
        elif name.endswith("__merged_fea_ori.json"):
            rank = 2
        elif name.endswith("_merged_fea_ori.json"):
            rank = 3
        else:
            rank = 9
        return (rank, -p.stat().st_size, str(p))

    candidates.sort(key=pref_key)
    return candidates[0]

def _find_candidate_xcoder_tree(data_recipe_root: Path) -> Optional[Path]:
    """Pick a default merged tree shipped with X-Coder (if present).

    This repo may vendor a precomputed merged feature tree under
    `question_generation/features_trees_data/` to avoid requiring an external EpiCoder checkout.
    """
    search_root = data_recipe_root / "question_generation" / "features_trees_data"
    if not search_root.exists():
        return None

    patterns = [
        # Current vendored filename (preferred). Note: this is a JSON object (not JSONL),
        # but we keep the filename short and descriptive.
        "feature_evoled.json",
        # Back-compat: earlier filename used in this repo.
        "feature_evoled.jsonl",
        # Back-compat: older naming conventions copied from EpiCoder outputs.
        "*__merged_fea.json",
        "*_merged_fea.json",
        "*__merged_fea_ori.json",
        "*_merged_fea_ori.json",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(search_root.rglob(pat))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    def pref_key(p: Path) -> tuple:
        name = p.name
        if name == "feature_evoled.json":
            rank = 0
        elif name == "feature_evoled.jsonl":
            rank = 1
        elif name.endswith("__merged_fea.json"):
            rank = 2
        elif name.endswith("_merged_fea.json"):
            rank = 3
        elif name.endswith("__merged_fea_ori.json"):
            rank = 4
        elif name.endswith("_merged_fea_ori.json"):
            rank = 5
        else:
            rank = 9
        return (rank, -p.stat().st_size, str(p))

    candidates.sort(key=pref_key)
    return candidates[0]


def _parse_shapes(values: Sequence[str]) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for raw in values:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid --shapes entry: {raw!r}. Expected 'a,b,c'.")
        a, b, c = (int(parts[0]), int(parts[1]), int(parts[2]))
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError(f"Invalid --shapes entry: {raw!r}. All values must be > 0.")
        shapes.append((a, b, c))
    if not shapes:
        raise ValueError("No shapes provided.")
    return shapes


def _dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _prune_empty(node: JSONValue) -> JSONValue:
    """Drop empty dict/list branches to keep leaf_count meaningful and prompts compact."""
    if isinstance(node, dict):
        out: Dict[str, JSONValue] = {}
        for k, v in node.items():
            if k in _SKIP_KEYS:
                continue
            pv = _prune_empty(v)
            if isinstance(pv, dict) and not pv:
                continue
            if isinstance(pv, list) and len(pv) == 0:
                continue
            out[k] = pv
        return out
    if isinstance(node, list):
        filtered = [x for x in node if isinstance(x, str)]
        return _dedup_preserve_order(filtered)
    return node


def _count_leaf_features(node: JSONValue) -> int:
    """Count leaf features (strings) in the sampled tree.

    This is aligned with X-Coder `generate_questions.py` leaf counting:
    - When a dict value is a list, each list item is a leaf.
    - When a dict value is a dict, recurse.
    """
    if isinstance(node, dict):
        total = 0
        for v in node.values():
            if isinstance(v, list):
                total += sum(1 for x in v if isinstance(x, str))
            elif isinstance(v, dict):
                total += _count_leaf_features(v)
        return total
    if isinstance(node, list):
        return sum(1 for x in node if isinstance(x, str))
    return 0


def _sample_list(items: Sequence[str], k: int, rng: random.Random) -> List[str]:
    if k <= 0:
        return []
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    return rng.sample(list(items), k)


def _sample_tree(
    node: JSONValue,
    *,
    rng: random.Random,
    root_k: int,
    inner_k: int,
    leaf_k: int,
    max_depth: int,
    depth: int = 0,
) -> JSONValue:
    # Depth is only counted over dict nodes; lists are treated as leaves.
    if isinstance(node, dict):
        if depth >= max_depth:
            return {}
        keys = [k for k in node.keys() if k not in _SKIP_KEYS]
        if not keys:
            return {}
        keys = sorted(keys)  # stable order for reproducibility across JSON dumps / Python versions
        k = root_k if depth == 0 else inner_k
        chosen_keys = _sample_list(keys, min(k, len(keys)), rng)
        out: Dict[str, JSONValue] = {}
        for ck in chosen_keys:
            child = _sample_tree(
                node[ck],
                rng=rng,
                root_k=root_k,
                inner_k=inner_k,
                leaf_k=leaf_k,
                max_depth=max_depth,
                depth=depth + 1,
            )
            # Prune empty children early to avoid large but useless structures.
            if isinstance(child, dict) and not child:
                continue
            if isinstance(child, list) and len(child) == 0:
                continue
            out[ck] = child
        return out

    if isinstance(node, list):
        # Keep only string leaves.
        items = [x for x in node if isinstance(x, str)]
        items = _dedup_preserve_order(items)
        # Sample from a sorted pool so identical seeds produce identical results.
        items = sorted(items)
        return _sample_list(items, min(leaf_k, len(items)), rng)

    return node


def _canonical_key(node: JSONValue) -> str:
    # Sort keys to dedup identical structures regardless of insertion order.
    return json.dumps(node, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_one(
    tree: Dict[str, JSONValue],
    *,
    rng: random.Random,
    shapes: Sequence[Tuple[int, int, int]],
    min_leaf_count: int,
    max_leaf_count: Optional[int],
    max_depth: int,
    max_attempts: int,
) -> Tuple[Dict[str, JSONValue], int, Tuple[int, int, int]]:
    last_err = None
    for _ in range(max_attempts):
        shape = rng.choice(list(shapes))
        root_k, inner_k, leaf_k = shape
        sampled = _sample_tree(
            tree,
            rng=rng,
            root_k=root_k,
            inner_k=inner_k,
            leaf_k=leaf_k,
            max_depth=max_depth,
        )
        sampled = _prune_empty(sampled)
        leaf_count = _count_leaf_features(sampled)
        if leaf_count < min_leaf_count:
            last_err = f"leaf_count {leaf_count} < min_leaf_count {min_leaf_count}"
            continue
        if max_leaf_count is not None and leaf_count > max_leaf_count:
            last_err = f"leaf_count {leaf_count} > max_leaf_count {max_leaf_count}"
            continue
        if not sampled:
            last_err = "empty sampled tree after pruning"
            continue
        return sampled, leaf_count, shape
    raise RuntimeError(f"Failed to sample a valid feature tree after {max_attempts} attempts. last_err={last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-tree",
        type=str,
        default=None,
        help=(
            "Path to merged feature tree JSON (e.g. 'feature_evoled.json' or '*_merged_fea.json'). "
            "If omitted, we try to auto-detect from a sibling '../EpiCoder' checkout."
        ),
    )
    parser.add_argument(
        "--epicoder-root",
        type=str,
        default=None,
        help="Optional path to an EpiCoder checkout to auto-detect the merged feature tree.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path (one JSON object per line).",
    )
    parser.add_argument("--num", type=int, default=100, help="Number of feature configs to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument("--min-leaf-count", type=int, default=25, help="Minimum leaf_count per sampled config.")
    parser.add_argument("--max-leaf-count", type=int, default=None, help="Optional maximum leaf_count per config.")
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=["3,3,1", "3,3,2", "4,3,2"],
        help="Sampling shapes. Each entry is 'root_k,inner_k,leaf_k'. You can pass multiple.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum dict depth to descend when sampling (lists are leaves).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=500,
        help="Max rejection-sampling attempts per config.",
    )
    args = parser.parse_args()

    shapes = _parse_shapes(args.shapes)

    # Resolve input tree.
    input_tree_path: Optional[Path] = Path(args.input_tree).expanduser() if args.input_tree else None
    if input_tree_path is None:
        # Prefer an explicitly provided EpiCoder root.
        script_dir = Path(__file__).resolve().parent
        # script_dir = .../X-Coder/data-recipe/question_generation/features_trees_data/utils
        data_recipe_root = script_dir.parents[2]

        if args.epicoder_root:
            epicoder_root = Path(args.epicoder_root).expanduser()
            input_tree_path = _find_candidate_epicoder_tree(epicoder_root)
        else:
            # Use a vendored artifact if available; otherwise fall back to a sibling ../EpiCoder checkout.
            input_tree_path = _find_candidate_xcoder_tree(data_recipe_root)
            if input_tree_path is None:
                epicoder_root = data_recipe_root.parent / "EpiCoder"
                input_tree_path = _find_candidate_epicoder_tree(epicoder_root)

    if input_tree_path is None or not input_tree_path.exists():
        raise SystemExit(
            "Could not find EpiCoder merged feature tree.\n"
            "Provide --input-tree, or set --epicoder-root to an EpiCoder checkout that contains "
            "`output/feature_evol/**/_merged_fea.json` (or `_merged_fea_ori.json`)."
        )

    print(f"Using merged feature tree: {input_tree_path}")

    with input_tree_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)
    if not isinstance(tree, dict):
        raise SystemExit(f"Expected a JSON object at root in {input_tree_path}, got {type(tree)}")

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng_global = random.Random(args.seed)
    seen = set()

    records: List[Dict[str, Any]] = []
    for idx in range(args.num):
        # Per-item RNG keeps each item stable even if other items change count/attempts later.
        rng = random.Random(rng_global.randint(0, 2**31 - 1) ^ (args.seed + idx))
        sampled, leaf_count, shape = _build_one(
            tree,
            rng=rng,
            shapes=shapes,
            min_leaf_count=args.min_leaf_count,
            max_leaf_count=args.max_leaf_count,
            max_depth=args.max_depth,
            max_attempts=args.max_attempts,
        )
        key = _canonical_key(sampled)
        if key in seen:
            # Extremely unlikely, but keep deterministic: resample with a new rng stream.
            sampled, leaf_count, shape = _build_one(
                tree,
                rng=random.Random((args.seed + 1_000_000) ^ idx),
                shapes=shapes,
                min_leaf_count=args.min_leaf_count,
                max_leaf_count=args.max_leaf_count,
                max_depth=args.max_depth,
                max_attempts=args.max_attempts,
            )
            key = _canonical_key(sampled)
        seen.add(key)

        records.append(
            {
                "features": sampled,
                "mandatory_features": [],
                "idx": idx,
                "leaf_count": leaf_count,
                "shape": list(shape),
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} configs -> {out_path}")
    if _is_git_lfs_pointer(out_path):
        print(
            "WARNING: output looks like a Git LFS pointer file. "
            "Did you accidentally write to a path tracked by LFS without git-lfs installed?"
        )


if __name__ == "__main__":
    main()
