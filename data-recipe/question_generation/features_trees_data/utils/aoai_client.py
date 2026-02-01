#!/usr/bin/env python3
"""Azure OpenAI chat client (TRAPI-style) used by feature extract/evol scripts.

This follows the style of `/home/v-jiewu5/call_api.py`:
- AzureOpenAI client from the `openai` package
- AAD token provider via `azure.identity`

Configuration (env vars):
- AZURE_OPENAI_ENDPOINT: e.g. "https://trapi.research.microsoft.com/gcr/shared"
- AZURE_OPENAI_API_VERSION: e.g. "2024-12-01-preview"
- AZURE_OPENAI_DEPLOYMENT: e.g. "gpt-5_2025-08-07"
- AZURE_OPENAI_TOKEN_SCOPE: defaults to "api://trapi/.default" (TRAPI)

If you're not using TRAPI, you may need to set:
- AZURE_OPENAI_TOKEN_SCOPE="https://cognitiveservices.azure.com/.default"
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _build_token_provider():
    try:
        from azure.identity import (
            ChainedTokenCredential,
            AzureCliCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "Missing dependency `azure-identity`.\n"
            "Install with: pip install azure-identity\n"
            f"Original error: {exc}"
        ) from exc

    scope = os.getenv("AZURE_OPENAI_TOKEN_SCOPE", "api://trapi/.default")
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )
    return credential


def _get_client():
    try:
        from openai import AzureOpenAI
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "Missing dependency `openai`.\n"
            "Install with: pip install openai\n"
            f"Original error: {exc}"
        ) from exc

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        # Convenience: match `/home/v-jiewu5/call_api.py` shape.
        instance = os.getenv("TRAPI_INSTANCE")
        if instance:
            endpoint = f"https://trapi.research.microsoft.com/{instance}"
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", os.getenv("TRAPI_API_VERSION", "2024-12-01-preview"))
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is required.\n"
            'Example: export AZURE_OPENAI_ENDPOINT="https://trapi.research.microsoft.com/gcr/shared"\n'
            'Or (TRAPI-style): export TRAPI_INSTANCE="gcr/shared"'
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=_build_token_provider(),
        api_version=api_version,
    )


def chat_completion(
    *,
    messages: List[Dict[str, Any]],
    deployment: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: Optional[float] = None,
) -> str:
    """Call Azure OpenAI chat completion and return the first message content."""
    deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("TRAPI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_DEPLOYMENT is required.\n"
            'Example: export AZURE_OPENAI_DEPLOYMENT="gpt-5_2025-08-07"'
        )

    client = _get_client()
    # Azure model deployments may differ in which params they support:
    # - Many chat models support `max_tokens`, `temperature`, `top_p`
    # - Some newer reasoning models (e.g. gpt-5) may require `max_completion_tokens`
    #   and only support default sampling params (so we omit non-defaults).
    kwargs: Dict[str, Any] = {
        "model": deployment,
        "messages": messages,
        "timeout": timeout,
    }
    is_gpt5 = "gpt-5" in deployment
    if not is_gpt5:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        # Keep compatibility with models that only accept defaults by omitting
        # non-default sampling params.
        if abs(temperature - 1.0) < 1e-9:
            kwargs["temperature"] = temperature
        if abs(top_p - 1.0) < 1e-9:
            kwargs["top_p"] = top_p
    if is_gpt5:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # pragma: no cover - backend dependent
        msg = str(exc)
        if "Unsupported parameter" in msg and "'max_tokens'" in msg:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
        elif "Unsupported parameter" in msg and "'max_completion_tokens'" in msg:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content or ""
