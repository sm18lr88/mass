# JSON extraction and response/usage parsing helpers for the JSON LLM backend.
"""Pure parsing helpers shared by ``llm_backend`` (no network, no client)."""

from __future__ import annotations

import json
import re
from typing import Any

EXCERPT_CHARS = 4000


class LLMJsonError(RuntimeError):
    """The model output could not be turned into a JSON object.

    Subclasses RuntimeError so callers that caught the historical error type
    keep working. ``excerpt`` holds the (truncated) raw text for diagnosis.
    """

    def __init__(self, message: str, *, excerpt: str = "") -> None:
        super().__init__(message)
        self.excerpt = excerpt


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise LLMJsonError("LLM returned an empty response.")
    excerpt = text[:EXCERPT_CHARS]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Fallback candidates, tried in order; accept the first that parses.
        # The outermost-brace slice comes first: a fenced ``` block can also
        # appear ESCAPED inside a JSON string value (e.g. a design text that
        # itself contains a code fence), in which case the fenced regex
        # captures an unparseable fragment.
        candidates = []
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        if not candidates:
            raise LLMJsonError(
                f"LLM returned non-JSON output:\n{excerpt}", excerpt=excerpt
            ) from None
        value = None
        last_exc: json.JSONDecodeError | None = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_exc = exc
        if value is None:
            raise LLMJsonError(
                f"LLM returned invalid JSON: {last_exc}\n---\n{excerpt}",
                excerpt=excerpt,
            ) from last_exc
    if not isinstance(value, dict):
        raise LLMJsonError(
            f"LLM returned JSON {type(value).__name__}; expected an object.",
            excerpt=excerpt,
        )
    return value


def responses_text(response: Any) -> str:
    raw = getattr(response, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                text = getattr(block, "text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return ""


def chat_usage_metrics(response: Any, *, model: str) -> dict[str, Any]:
    usage_obj = getattr(response, "usage", None)
    usage_raw: dict[str, Any] = {}
    if usage_obj is not None and hasattr(usage_obj, "model_dump"):
        dumped = usage_obj.model_dump()
        if isinstance(dumped, dict):
            usage_raw = dumped
    prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
    completion_tokens = int(usage_raw.get("completion_tokens") or 0)
    total_tokens = int(
        usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens)
    )
    usage: dict[str, Any] = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    prompt_details = usage_raw.get("prompt_tokens_details")
    completion_details = usage_raw.get("completion_tokens_details")
    if isinstance(prompt_details, dict):
        usage["input_tokens_details"] = prompt_details
    if isinstance(completion_details, dict):
        usage["output_tokens_details"] = completion_details
    return {
        "usage": usage,
        "estimated_cost_usd": 0.0,
        "cost_estimate_note": "Locally hosted vLLM inference; API cost recorded as $0.",
        "response_id": str(getattr(response, "id", "") or "") or None,
        "served_model": model,
    }
