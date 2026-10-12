# Provider-agnostic parsing of judge responses: output text, JSON candidates, and usage dicts.
from __future__ import annotations

import re
from typing import Any


def _text_from_response(response: Any) -> str:
    """Output text of an OpenAI Responses API result (``output_text`` or first message block)."""
    raw = getattr(response, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                t = getattr(block, "text", "")
                if isinstance(t, str) and t.strip():
                    return t.strip()
    return ""


def _text_from_anthropic_message(message: Any) -> str:
    """Join the non-empty text blocks of an Anthropic Messages API result."""
    chunks: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", "")
            if isinstance(t, str) and t.strip():
                chunks.append(t.strip())
    return "\n".join(chunks).strip()


def _extract_json_candidate(raw: str) -> str:
    """Best-effort JSON object substring: fenced ```json block, else outermost braces, else input."""
    s = raw.strip()
    if not s:
        return s
    # Accept markdown code-fenced JSON often returned by non-strict output routes.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()
    return s


def _usage_dict_from_anthropic_message(message: Any) -> dict[str, Any]:
    """Normalize Anthropic usage into the Responses-style usage dict (with cache details)."""
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    out_dict: dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
    }
    if cache_create:
        out_dict["input_tokens_details"] = {"cache_creation_input_tokens": cache_create}
    if cache_read:
        details = out_dict.setdefault("input_tokens_details", {})
        details["cache_read_input_tokens"] = cache_read
    return out_dict


def _usage_dict_from_google_response(response: Any) -> dict[str, Any]:
    """Normalize Gemini ``usage_metadata`` into the Responses-style usage dict."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    inp = int(getattr(meta, "prompt_token_count", 0) or 0)
    out = int(getattr(meta, "candidates_token_count", 0) or 0)
    tot = int(getattr(meta, "total_token_count", 0) or (inp + out))
    out_dict: dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": tot,
    }
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
    cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
    if thoughts:
        out_dict["output_tokens_details"] = {"thoughts_token_count": thoughts}
    if cached:
        details = out_dict.setdefault("input_tokens_details", {})
        details["cached_content_token_count"] = cached
    return out_dict
