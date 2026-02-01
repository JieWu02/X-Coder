#!/usr/bin/env python3
"""Evolve (expand) a feature tree via LLM, Epicoder-style.

This is the "evol" stage used to expand the initial extracted feature tree.

Workflow:
1) Sample a small subtree from a base feature tree JSON.
2) Ask the model to expand it in breadth+depth.
3) Save the expansion results to JSONL for later merging.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set

from aoai_client import chat_completion


JSONValue = Union[Dict[str, "JSONValue"], List["JSONValue"], str, int, float, bool, None]


FEATURE_EXPLANATIONS = {
    "problem_domain": "The category/field of the competitive programming problem.",
    "algorithmic_paradigm": "High-level paradigms used to solve the problem (e.g., DP, greedy).",
    "algorithms": "Specific algorithms used, grouped by area (graph/DP/string/etc).",
    "mathematics": "Math concepts and techniques used (number theory, combinatorics, etc).",
    "problem_solving_strategies": "General problem solving tactics (case analysis, modeling, etc).",
    "data_structures": "Data structures involved (arrays, trees, heaps, DSU, etc).",
    "coding_techniques": "Implementation patterns (two pointers, recursion, prefix sums, etc).",
    "complexity_analysis": "Time/space complexity considerations.",
    "optimization_techniques": "Performance improvements beyond baseline algorithm choice.",
}


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


def _parse_int_list(csv: str) -> List[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    out = [int(p) for p in parts]
    if not out or any(x <= 0 for x in out):
        raise ValueError(f"Invalid conditions: {csv!r} (expected positive ints)")
    return out


def _dedup_strings(items: Sequence[Any]) -> List[str]:
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


def sample_feature_tree(feature_tree: JSONValue, conditions: List[int], rng: random.Random) -> JSONValue:
    """Randomly sample a subtree by limiting branching factor per depth."""

    def sample_subtree(node: JSONValue, depth: int) -> JSONValue:
        if not isinstance(node, (dict, list)) or depth > len(conditions):
            return node
        if isinstance(node, dict):
            keys = list(node.keys())
            if not keys:
                return {}
            k = min(conditions[depth - 1], len(keys))
            chosen = rng.sample(keys, k)
            out: Dict[str, JSONValue] = {}
            for ck in chosen:
                out[ck] = sample_subtree(node[ck], depth + 1)
            return out
        # list
        items = _dedup_strings(node)
        if not items:
            return []
        k = min(conditions[depth - 1], len(items))
        return rng.sample(items, k)

    return sample_subtree(feature_tree, 1)


def form_feature_explanations(keys: Sequence[str]) -> str:
    lines: List[str] = []
    for k in keys:
        desc = FEATURE_EXPLANATIONS.get(k)
        if desc:
            lines.append(f"{k}: {desc}")
    return "\n".join(lines)


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
    parser.add_argument("--input-tree", type=str, required=True, help="Base feature tree JSON path.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of evolution calls.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--conditions",
        type=str,
        default="2,1,2,1",
        help="Branching limits per depth, e.g. '2,1,2,1'.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Prompt template file. Defaults to feature_tree_prompts/feature_evol_prompt10.txt",
    )
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

    in_path = Path(args.input_tree).expanduser()
    tree = json.load(in_path.open("r", encoding="utf-8"))
    if not isinstance(tree, dict):
        raise SystemExit(f"Expected a JSON object at root in {in_path}, got {type(tree)}")

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    qg_root = script_dir.parents[1]  # .../question_generation
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser()
    else:
        prompt_path = qg_root / "feature_tree_prompts" / "feature_evol_prompt10.txt"
    prompt_template = _load_prompt(prompt_path)

    rng = random.Random(args.seed)
    conditions = _parse_int_list(args.conditions)

    seen = _load_existing_indices(out_path) if args.resume else set()

    for i in range(args.num_steps):
        if i in seen:
            continue

        sampled = sample_feature_tree(tree, conditions, rng)
        if not isinstance(sampled, dict) or not sampled:
            continue

        explanations = form_feature_explanations(list(sampled.keys()))
        features_json = json.dumps(sampled, ensure_ascii=False, indent=2)
        prompt = prompt_template.replace("{explanations}", explanations).replace("{features}", features_json)

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
        expanded = _parse_json_between_tags(content)

        rec = {
            "idx": i,
            "features": sampled,
            "expanded_feature": expanded,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. Wrote to {out_path}")


if __name__ == "__main__":
    main()
