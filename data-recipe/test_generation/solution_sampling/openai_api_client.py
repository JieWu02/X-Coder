#!/usr/bin/env python3
"""
Minimal OpenAI-compatible API client with retry/backoff and concurrency control.
"""

import os
import re
import time
import threading
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI

# ----------------- Global concurrency & monitoring -----------------

# Limit concurrent API calls (same逻辑：100个并发)
API_SEMAPHORE = threading.Semaphore(100)

# Track recent success/fail for monitoring
API_SUCCESS_WINDOW = []
API_SUCCESS_LOCK = threading.Lock()


def record_api_result(success: bool) -> None:
    """Record API call result for monitoring (keeps last 50 results)."""
    global API_SUCCESS_WINDOW
    with API_SUCCESS_LOCK:
        API_SUCCESS_WINDOW.append(success)
        if len(API_SUCCESS_WINDOW) > 50:
            API_SUCCESS_WINDOW.pop(0)


# ----------------- GPT Client 封装 -----------------

class GPTClient:
    """Thin wrapper around an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "deepseek-ai/DeepSeek-R1-0528",
    ) -> None:
        """
        Args:
            base_url: OpenAI-compatible base URL (default: OPENAI_BASE_URL).
            api_key:  API key (default: OPENAI_API_KEY).
            model_name: Model name.
        """
        self.model_name = model_name
        self.client = self._setup_client(base_url, api_key)

    @staticmethod
    def _setup_client(
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> OpenAI:
        """Setup OpenAI-compatible client."""
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        return OpenAI(base_url=base_url, api_key=api_key)

    def call_chat(
        self,
        prompt: str,
        max_completion_tokens: int = 32768,
        timeout: int = 300,
        reasoning_effort: str = "high",
        max_retries: int = 5,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Call OpenAI-compatible chat completions with retry & rate-limit handling.

        Args:
            prompt:              User prompt.
            max_completion_tokens: Max completion tokens.
            timeout:            Request timeout (seconds).
            reasoning_effort:   "low" | "medium" | "high".
            max_retries:        Max retry attempts.

        Returns:
            (content, usage) where:
              - content: str | None  (model输出文本)
              - usage:   dict | None (OpenAI usage，对应 response.usage)
        """
        for attempt in range(max_retries):
            # 控制并发
            with API_SEMAPHORE:
                try:
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_completion_tokens,
                    }
                    # Keep compatibility without enforcing reasoning_effort.
                    response = self.client.chat.completions.create(
                        timeout=timeout,
                        **payload,
                    )

                    if response and response.choices:
                        record_api_result(True)
                        print("✅ API call successful")
                        return response.choices[0].message.content, response.usage
                    else:
                        record_api_result(False)
                        print("❌ API call returned no response")
                        return None, None

                except Exception as e:
                    error_msg = str(e)

                    # ----- 429 / rate limit -----
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        record_api_result(False)
                        # 从报错里抽取 "Try again in X seconds"
                        wait_match = re.search(r"Try again in (\d+) seconds", error_msg)
                        if wait_match:
                            wait_time = float(wait_match.group(1))
                        else:
                            # 0.5, 1, 2, 4... 最多 10s（和原脚本一致）
                            wait_time = min(0.5 * (2 ** attempt), 10)

                        if attempt < max_retries - 1:
                            print(
                                f"⏳ Rate limit, retry in {wait_time:.1f}s "
                                f"(attempt {attempt + 2}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            print(
                                f"❌ API call failed after {max_retries} attempts: {e}"
                            )
                            return None, None

                    # ----- timeout -----
                    elif "timeout" in error_msg.lower():
                        if attempt < max_retries - 1:
                            print(
                                f"⏰ Timeout, retrying {attempt + 2}/{max_retries}..."
                            )
                            time.sleep(2)
                            continue
                        else:
                            print(
                                f"⏰ API call timed out after {max_retries} attempts"
                            )
                            return None, None

                    # ----- 其它错误 -----
                    else:
                        print(f"❌ API call failed: {e}")
                        return None, None

        # 理论上不会走到这里
        return None, None


# 你在其它脚本里可以这样用：
#
# from openai_api_client import GPTClient
#
# gpt = GPTClient()
# content, usage = gpt.call_chat("Your prompt here")
# print(content)
