#!/usr/bin/env python3
"""
Two-stage approach for generating competitive programming problems with random question generation styles.
This version uses an OpenAI-compatible API endpoint without any SGLang dependency.
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

from openai import OpenAI

_SCRIPT_DIR = Path(__file__).resolve().parent
_TEMPLATE_DIR = _SCRIPT_DIR.parent / "question_gen_template"
if str(_TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(_TEMPLATE_DIR))

from question_gen_template import ATCODER_TEMPLATE, CODEFORCES_TEMPLATE, LEETCODE_TEMPLATE
from question_gen_template.select_feature import STAGE1_PROMPT_TEMPLATE

# Question generation templates with their names
QUESTION_TEMPLATES = {
    "codeforces": CODEFORCES_TEMPLATE,
    "leetcode": LEETCODE_TEMPLATE,
    "atcoder": ATCODER_TEMPLATE,
}


def load_json_file(file_path: str) -> List[Dict]:
    """Load a JSON array file into memory."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array")
        print(f"📁 Loaded {len(data)} feature samples from {file_path}")
        return data
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Error loading JSON features file {file_path}: {exc}")
        return []


def load_jsonl_file(file_path: str) -> List[Dict]:
    """Load a JSONL file into memory."""
    records: List[Dict] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"⚠️ JSONL parse error at line {line_num}: {exc}")
                    continue
                if isinstance(record, dict):
                    records.append(record)
                else:
                    print(f"⚠️ Skipping non-object record at line {line_num}")
        print(f"📁 Loaded {len(records)} feature samples from {file_path}")
        return records
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Error loading JSONL features file {file_path}: {exc}")
        return []


def load_features_from_file(features_file_override: Optional[str] = None) -> Tuple[bool, List[Dict]]:
    """Load feature configurations with JSON/JSONL fallback."""

    def _load_file(path: Path) -> List[Dict]:
        if path.suffix.lower() == ".jsonl":
            return load_jsonl_file(str(path))
        return load_json_file(str(path))

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent.parent

    if features_file_override:
        override_path = Path(features_file_override)
        if not override_path.is_absolute():
            override_path = repo_root / override_path
        if override_path.exists():
            print(f"📁 Using custom features file: {override_path}")
            return False, _load_file(override_path)
        print(f"⚠️ Custom features file not found: {override_path}")

    candidates = [
        repo_root / "question_generation" / "features_trees_data" / "feature_all.jsonl",
        base_dir / "feature_set" / "features_set_with_leaf_nodes_12_to_15_no_overlap_dedup.json",
        base_dir / "feature_set" / "features_with_leaf_node_greater_than20.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            print(f"📁 Using features file: {candidate}")
            return "dedup" in candidate.name, _load_file(candidate)

    print("💡 Tip: place features in question_generation/features_trees_data/feature_all.jsonl or pass --features-file.")
    return False, []


def generate_varied_features(index: int, features_data: List[Dict]) -> Dict:
    """Fetch a feature configuration by index with wrap-around support."""
    if not features_data:
        raise ValueError("No features data loaded from file")
    if index < len(features_data):
        return features_data[index]["features"]
    return features_data[index % len(features_data)]["features"]


@dataclass
class PromptRequest:
    custom_id: str
    prompt: str


class APIClient:
    """Utility for issuing chat completion requests against an OpenAI-compatible API."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model_name: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        request_timeout: int,
    ) -> None:
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.request_timeout = request_timeout
        self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _extract_content(self, response_body: Dict) -> Optional[str]:
        choices = response_body.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        if isinstance(choices, dict):
            message = choices.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        message = response_body.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        return None

    def call_single(self, prompt: str) -> Dict:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(prompt),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        try:
            response = self.client.chat.completions.create(
                timeout=self.request_timeout,
                **payload,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                return {"success": False, "error": "Missing choices in response"}
            content = getattr(choices[0].message, "content", None)
            if not content:
                return {"success": False, "error": "Missing content in response"}
            return {"success": True, "content": content}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Request failed: {exc}"}

    def run_batch(
        self,
        requests_to_run: List[PromptRequest],
        stage_tag: str,
        max_wait_time: int,
    ) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        if not requests_to_run:
            return {}, {"submitted": 0, "succeeded": 0, "failed": 0, "mode": "none"}

        stats = {"submitted": len(requests_to_run), "succeeded": 0, "failed": 0, "mode": "single"}
        responses: Dict[str, Dict] = {}

        for request in requests_to_run:
            single_result = self.call_single(request.prompt)
            if single_result.get("success"):
                responses[request.custom_id] = {
                    "success": True,
                    "content": single_result["content"],
                    "stage": stage_tag,
                }
                stats["succeeded"] += 1
            else:
                responses[request.custom_id] = {
                    "success": False,
                    "error": single_result.get("error", "Unknown error"),
                    "stage": stage_tag,
                }
        stats["failed"] = stats["submitted"] - stats["succeeded"]
        return responses, stats


class MultiStyleTwoStageGenerator:
    """Implements the two-stage generation workflow using an API backend."""

    def __init__(
        self,
        batch_client: APIClient,
        template_weights: Optional[List[float]] = None,
    ) -> None:
        self.batch_client = batch_client
        self.template_weights = template_weights or [0.7, 0.15, 0.15]

    # --- JSON cleaning helpers (ported from the original Azure version) ---
    def clean_json_response(self, response_text: str) -> Optional[str]:
        if not response_text:
            return None
        cleaned = re.sub(r"```json\s*", "", response_text)
        cleaned = re.sub(r"```\s*", "", cleaned)
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return self.fix_json_format(json_str)
        return response_text.strip()

    def fix_json_format(self, json_str: str) -> str:
        if len(json_str) > 100_000:
            print(f"⚠️ JSON string too large ({len(json_str)} chars), truncating to 100k")
            json_str = json_str[:100_000]
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as exc:
            print(f"🔧 Attempting to fix JSON format error: {exc}")
            original = json_str
            json_str = json_str.strip()
            start_brace = json_str.find("{")
            if start_brace > 0:
                json_str = json_str[start_brace:]
            last_brace = json_str.rfind("}")
            if last_brace > 0:
                json_str = json_str[: last_brace + 1]
            for attempt in range(5):
                if attempt == 0:
                    json_str = re.sub(r"\\n", " ", json_str)
                    json_str = re.sub(r"\\t", " ", json_str)
                    json_str = re.sub(r"\\r", " ", json_str)
                    json_str = re.sub(r"\s+", " ", json_str)
                elif attempt == 1:
                    json_str = re.sub(r"\\\\\"", '"', json_str)
                    json_str = re.sub(r"(?<!\\)\"(?=[^:,}\]\s]*[^:,}\]\s])", r"\\\"", json_str)
                elif attempt == 2:
                    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                    json_str = re.sub(r'(}[\s]*)\n\s*("[^"]*":)', r'\1,\n\2', json_str)
                elif attempt == 3:
                    json_str = re.sub(r"(\})(\{)", r"\1,\2", json_str)
                    json_str = re.sub(r"(\])(\[)", r"\1,\2", json_str)
                    json_str = re.sub(r'("[^"]*")(\{)', r"\1: \2", json_str)
                elif attempt == 4:
                    json_str = re.sub(r'"([^"]*?)\n([^"]*?)"', lambda m: f'"{m.group(1).strip()} {m.group(2).strip()}"', json_str)
                    json_str = re.sub(r":\s*([^\"\{\[\d\-][^,}\]]*?)([,}\]])", r': "\1"\2', json_str)
                try:
                    json.loads(json_str)
                    print(f"✅ JSON format fixed on attempt {attempt + 1}")
                    return json_str
                except json.JSONDecodeError as err:
                    print(f"❌ Attempt {attempt + 1} failed: {err}")
            print("❌ All JSON fix attempts failed")
            return original

    def parse_stage1_with_fallback(self, cleaned_result: str) -> Optional[Dict]:
        try:
            return json.loads(cleaned_result)
        except json.JSONDecodeError as exc:
            print(f"❌ Stage 1 JSON parsing failed: {exc}")
        try:
            print("🔄 Using minimal Stage 1 fallback structure")
            return {
                "feature_roles_tree": {"core": "algorithms", "auxiliary": "implementation"},
                "selected_features_tree": {"data_structures": "arrays", "algorithms": "sorting"},
                "integration_strategy": "Create a problem that tests fundamental algorithmic thinking and implementation skills",
            }
        except Exception as exc:
            print(f"❌ Minimal Stage 1 fallback failed: {exc}")
            return None

    def parse_stage2_with_fallback(self, cleaned_result: str) -> Optional[Dict]:
        try:
            parsed = json.loads(cleaned_result)
            if "question" in parsed:
                return parsed
            print("❌ Stage 2 JSON parsed but missing 'question'")
        except json.JSONDecodeError as exc:
            print(f"❌ Stage 2 JSON parsing failed: {exc}")
        try:
            print("🔄 Attempting regex extraction for Stage 2 question")
            patterns = [
                r'"?question"?\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                r'"?question"?\s*:\s*"([\s\S]*?)"(?=\s*[,}])',
                r'"?question"?\s*:\s*"([\s\S]*?)"\s*}?\s*$',
            ]
            for pattern in patterns:
                match = re.search(pattern, cleaned_result, re.DOTALL | re.IGNORECASE)
                if match:
                    question_content = match.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
                    if question_content:
                        print("✅ Stage 2 question extracted via regex")
                        return {"question": question_content}
        except Exception as exc:
            print(f"❌ Stage 2 regex extraction failed: {exc}")
        try:
            print("🔄 Falling back to minimal Stage 2 stub")
            text_parts = re.findall(r'[A-Za-z][^{}"\[\]]*[.!?]', cleaned_result)
            fallback = max(text_parts, key=len).strip() if text_parts else "Generated competitive programming problem based on selected features."
            return {"question": fallback}
        except Exception as exc:
            print(f"❌ Stage 2 minimal fallback failed: {exc}")
            return None

    # --- Feature utilities ---
    def filter_programming_language_features(self, features_tree):
        if isinstance(features_tree, dict):
            filtered = {}
            for key, value in features_tree.items():
                if key == "programming language":
                    continue
                filtered_value = self.filter_programming_language_features(value)
                if filtered_value:
                    filtered[key] = filtered_value
            return filtered
        if isinstance(features_tree, list):
            filtered_list = []
            for item in features_tree:
                filtered_item = self.filter_programming_language_features(item)
                if filtered_item:
                    filtered_list.append(filtered_item)
            return filtered_list
        return features_tree

    def count_leaf_nodes(self, features_tree) -> int:
        leaf_count = 0

        def traverse(node):
            nonlocal leaf_count
            if isinstance(node, dict):
                for value in node.values():
                    if isinstance(value, list):
                        leaf_count += len(value)
                    elif isinstance(value, dict):
                        if "feature" in value:
                            leaf_count += 1
                        else:
                            traverse(value)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(features_tree)
        return leaf_count

    # --- Stage preparation & parsing ---
    def prepare_stage1_request(self, features_tree, use_pre_filtered: bool, custom_id: str) -> Tuple[Optional[PromptRequest], Optional[Dict]]:
        if use_pre_filtered:
            filtered_tree = features_tree
        else:
            filtered_tree = self.filter_programming_language_features(features_tree)
            leaf_count = self.count_leaf_nodes(filtered_tree)
            print(f"🔢 Feature tree has {leaf_count} leaf nodes")
            if leaf_count <= 50:
                return None, {
                    "error": f"Feature tree has insufficient leaf nodes: {leaf_count} (required > 50)",
                    "api_call_successful": False,
                    "stage": "stage1",
                }
        features_json = json.dumps(filtered_tree, ensure_ascii=False, indent=2)
        prompt_text = STAGE1_PROMPT_TEMPLATE.format(features_json=features_json)
        return PromptRequest(custom_id=custom_id, prompt=prompt_text), {"filtered_tree": filtered_tree}

    def parse_stage1_response(self, response_text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        cleaned = self.clean_json_response(response_text)
        if not cleaned:
            return False, None, "Empty Stage 1 response"
        parsed = self.parse_stage1_with_fallback(cleaned)
        if parsed is None:
            return False, None, "Failed to parse Stage 1 response"
        for field in ["feature_roles_tree", "selected_features_tree", "integration_strategy"]:
            if field not in parsed:
                return False, None, f"Stage 1 missing required field: {field}"
        parsed["api_call_successful"] = True
        return True, parsed, None

    def prepare_stage2_request(
        self,
        stage1_result: Dict,
        custom_id: str,
        template_style: Optional[str] = None,
    ) -> PromptRequest:
        if template_style is None:
            templates = list(QUESTION_TEMPLATES.keys())
            template_style = random.choices(templates, weights=self.template_weights, k=1)[0]
        template = QUESTION_TEMPLATES[template_style]
        selected_features_tree = stage1_result.get("selected_features_tree", {})
        selected_features_info = json.dumps(selected_features_tree, ensure_ascii=False, indent=2)
        integration_strategy = stage1_result.get(
            "integration_strategy",
            "Create a comprehensive algorithmic problem combining the selected features",
        )
        prompt_text = template.format(
            selected_features_info=selected_features_info,
            integration_strategy=integration_strategy,
        )
        return PromptRequest(custom_id=custom_id, prompt=prompt_text), template_style

    def parse_stage2_response(self, response_text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        cleaned = self.clean_json_response(response_text)
        if not cleaned:
            return False, None, "Empty Stage 2 response"
        parsed = self.parse_stage2_with_fallback(cleaned)
        if parsed is None or "question" not in parsed:
            return False, None, "Failed to extract Stage 2 question"
        parsed["api_call_successful"] = True
        return True, parsed, None


def append_results(new_results: List[Dict], output_file: str) -> int:
    existing = load_existing_results(output_file)
    combined = existing + new_results
    save_results(combined, output_file)
    return len(combined)


def load_existing_results(output_file: str) -> List[Dict]:
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📂 Loaded {len(data)} existing results from {output_file}")
            return data
        except Exception as exc:
            print(f"⚠️ Could not load existing results: {exc}")
    else:
        print(f"📁 No existing results file found at {output_file}, starting fresh")
    return []


def save_results(results: List[Dict], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Results saved to: {output_file}")


def process_batch_generation(
    generator: MultiStyleTwoStageGenerator,
    features_batch: List[Dict],
    indices_batch: List[int],
    use_pre_filtered: bool,
    batch_client: APIClient,
    max_wait_time: int,
    template_override: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    stage1_requests: List[PromptRequest] = []
    stage1_context: Dict[str, Dict] = {}
    batch_results: List[Dict] = []

    for features_tree, index in zip(features_batch, indices_batch):
        custom_id = f"stage1_{index}"
        request, error_info = generator.prepare_stage1_request(features_tree, use_pre_filtered, custom_id)
        if request is None:
            error_result = {
                "config_index": index,
                "error": error_info["error"],
                "stage": "stage1",
            }
            batch_results.append(error_result)
            continue
        stage1_requests.append(request)
        stage1_context[custom_id] = {"index": index, "features": features_tree}

    stage1_stats = {"submitted": 0, "succeeded": 0, "failed": 0, "mode": "none"}
    if stage1_requests:
        stage1_responses, stage1_stats = batch_client.run_batch(stage1_requests, "stage1", max_wait_time)
    else:
        stage1_responses = {}

    stage2_requests: List[PromptRequest] = []
    stage2_context: Dict[str, Dict] = {}

    for custom_id, context in stage1_context.items():
        response = stage1_responses.get(custom_id)
        index = context["index"]
        if not response:
            batch_results.append({
                "config_index": index,
                "error": "Stage 1 response missing",
                "stage": "stage1",
                "api_call_failed": True,
            })
            continue
        if not response.get("success"):
            batch_results.append({
                "config_index": index,
                "error": response.get("error", "Stage 1 call failed"),
                "stage": "stage1",
                "api_call_failed": True,
            })
            continue
        ok, stage1_result, error_message = generator.parse_stage1_response(response.get("content", ""))
        if not ok or stage1_result is None:
            batch_results.append({
                "config_index": index,
                "error": error_message or "Failed to parse Stage 1 response",
                "stage": "stage1",
                "api_call_successful": True,
            })
            continue
        stage1_result["config_index"] = index
        stage1_result["stage"] = "stage1"
        context["stage1_result"] = stage1_result
        stage2_request, style = generator.prepare_stage2_request(
            stage1_result,
            custom_id=f"stage2_{index}",
            template_style=template_override,
        )
        stage2_requests.append(stage2_request)
        stage2_context[stage2_request.custom_id] = {
            "index": index,
            "stage1_result": stage1_result,
            "template_style": style,
        }

    stage2_stats = {"submitted": 0, "succeeded": 0, "failed": 0, "mode": "none"}
    if stage2_requests:
        stage2_responses, stage2_stats = batch_client.run_batch(stage2_requests, "stage2", max_wait_time)
    else:
        stage2_responses = {}

    for custom_id, context in stage2_context.items():
        index = context["index"]
        response = stage2_responses.get(custom_id)
        if not response:
            batch_results.append({
                "config_index": index,
                "error": "Stage 2 response missing",
                "stage": "stage2",
                "stage1_analysis": context["stage1_result"],
                "api_call_failed": True,
            })
            continue
        if not response.get("success"):
            batch_results.append({
                "config_index": index,
                "error": response.get("error", "Stage 2 call failed"),
                "stage": "stage2",
                "stage1_analysis": context["stage1_result"],
                "api_call_failed": True,
            })
            continue
        ok, stage2_result, error_message = generator.parse_stage2_response(response.get("content", ""))
        if not ok or stage2_result is None:
            batch_results.append({
                "config_index": index,
                "error": error_message or "Failed to parse Stage 2 response",
                "stage": "stage2",
                "stage1_analysis": context["stage1_result"],
                "api_call_successful": True,
            })
            continue
        template_style = context["template_style"]
        final_result = {
            "config_index": index,
            "question": stage2_result.get("question", ""),
            "style": template_style,
            "stage1_analysis": context["stage1_result"],
            "stage2_output": stage2_result,
            "multi_style_two_stage_approach": True,
            "api_call_successful": True,
        }
        batch_results.append(final_result)

    batch_stats = {
        "api_success": stage1_stats.get("succeeded", 0) + stage2_stats.get("succeeded", 0),
        "api_failed": stage1_stats.get("failed", 0) + stage2_stats.get("failed", 0),
        "stage1_mode": stage1_stats.get("mode", "none"),
        "stage2_mode": stage2_stats.get("mode", "none"),
        "stage1_submitted": stage1_stats.get("submitted", 0),
        "stage2_submitted": stage2_stats.get("submitted", 0),
    }

    return batch_results, batch_stats


def summarize_batch(batch_stats: Dict[str, int]) -> str:
    return (
        f"Stage1: {batch_stats['stage1_submitted']} ({batch_stats['stage1_mode']}), "
        f"Stage2: {batch_stats['stage2_submitted']} ({batch_stats['stage2_mode']})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-style competitive programming questions using an OpenAI-compatible API",
    )
    parser.add_argument("--start", type=int, default=10000, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=20000, help="End index (exclusive)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of samples per batch")
    parser.add_argument("--api-base", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model-name", type=str, default=os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-R1-0528"))
    parser.add_argument("--system-prompt", type=str, default="You are a professional competitive programming problem setter.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=3600, help="Maximum wait time for batch completion")
    parser.add_argument("--target", type=int, default=16000, help="Target number of successful generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for template sampling")
    parser.add_argument("--template-style", choices=list(QUESTION_TEMPLATES.keys()), default=None,
                        help="Force a specific template style instead of random sampling")
    parser.add_argument("--features-file", type=str, default=None,
                        help="Optional path to feature JSON file (relative paths resolved from script directory)")

    args = parser.parse_args()

    random.seed(args.seed)

    print(
        "🎯 Generating questions with API two-stage pipeline | "
        f"base={args.api_base}, model={args.model_name}"
    )

    use_pre_filtered, features_data = load_features_from_file(args.features_file)
    if not features_data:
        print("❌ No feature data available. Abort.")
        return

    total_available = len(features_data)
    start_index = max(0, min(args.start, total_available))
    end_index = max(start_index, min(args.end, total_available))
    range_size = end_index - start_index
    print(f"📊 Using indices [{start_index}, {end_index}) -> {range_size} entries")

    target_successful_samples = min(args.target, range_size if range_size > 0 else total_available)

    output_file = args.output
    if not output_file:
        base_path = (
            "/mnt/wujie/Epicoder2_Data_Syn/problem_synthesis/question_generation/"
            "2stage_generation/multi-style-hard-question-12to15"
        )
        if range_size:
            output_file = f"{base_path}_{start_index}_{end_index}.json"
        else:
            output_file = f"{base_path}.json"

    existing_results = load_existing_results(output_file)
    existing_count = len(existing_results)

    print(f"🔄 Existing successful results: {existing_count}")
    if existing_count >= target_successful_samples:
        print(f"✅ Target already met ({existing_count}/{target_successful_samples}). Nothing to do.")
        return

    batch_client = APIClient(
        api_base_url=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        max_tokens=32_768,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        request_timeout=args.request_timeout,
    )
    generator = MultiStyleTwoStageGenerator(batch_client=batch_client)

    results_buffer: List[Dict] = []
    successful = 0
    failed = 0
    skipped = 0
    api_calls_successful = 0
    api_calls_failed = 0
    template_usage = {key: 0 for key in QUESTION_TEMPLATES}
    save_interval = 500

    indices = list(range(start_index, end_index)) if range_size else list(range(total_available))
    batch_size = max(1, args.batch_size)

    start_time = time.time()

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start: batch_start + batch_size]
        features_batch = []
        for idx in batch_indices:
            try:
                features_batch.append(generate_varied_features(idx, features_data))
            except Exception as exc:
                results_buffer.append({
                    "config_index": idx,
                    "error": f"Failed to load features: {exc}",
                    "stage": "load",
                })
        if not features_batch:
            continue

        batch_results, batch_stats = process_batch_generation(
            generator,
            features_batch,
            batch_indices,
            use_pre_filtered,
            batch_client,
            args.max_wait_seconds,
            template_override=args.template_style,
        )

        results_buffer.extend(batch_results)
        api_calls_successful += batch_stats.get("api_success", 0)
        api_calls_failed += batch_stats.get("api_failed", 0)

        success_this_batch = sum(
            1 for result in batch_results if result.get("question") and not result.get("error")
        )

        def is_skipped(item: Dict) -> bool:
            return item.get("stage") == "stage1" and "insufficient leaf nodes" in item.get("error", "")

        skipped_this_batch = sum(1 for result in batch_results if is_skipped(result))
        failure_this_batch = sum(
            1 for result in batch_results if result.get("error") and not is_skipped(result)
        )

        successful += success_this_batch
        failed += failure_this_batch
        skipped += skipped_this_batch

        for result in batch_results:
            if result.get("question") and not result.get("error"):
                style = result.get("style", "unknown")
                if style in template_usage:
                    template_usage[style] += 1

        total_generated = existing_count + successful
        print(
            f"✅ [{total_generated:,}/{target_successful_samples:,}] +{success_this_batch} in batch | "
            f"Failures this batch: {failure_this_batch}, skipped: {skipped_this_batch} | {summarize_batch(batch_stats)}"
        )

        if successful > 0 and successful % save_interval == 0:
            total_saved = append_results(results_buffer, output_file)
            results_buffer = []
            print(
                f"💾 Intermediate save at {successful} successes (total saved: {total_saved}) | "
                f"API calls success/fail: {api_calls_successful}/{api_calls_failed}"
            )

        if total_generated >= target_successful_samples:
            print("🎯 Target reached, stopping further batches")
            break

    if results_buffer:
        total_saved = append_results(results_buffer, output_file)
        print(f"🎉 Final save completed. Total records: {total_saved}")
    else:
        print("📄 No new results to save.")

    end_time = time.time()
    elapsed = end_time - start_time
    total_success = existing_count + successful

    print("\n📊 Final Summary")
    print(f"   ✅ Successful this session: {successful}")
    print(f"   📊 Total successful (including previous): {total_success}/{target_successful_samples}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏭️ Skipped (insufficient features): {skipped}")
    print(f"   📞 API calls: success={api_calls_successful}, failed={api_calls_failed}")
    print(f"   ⏱️ Elapsed time: {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print(f"   🎲 Template usage: {template_usage}")
    if successful:
        print(f"   ⏱️ Avg time per success: {elapsed / successful:.1f}s")


if __name__ == "__main__":
    main()
