# Provider adapters (OpenAI Responses, Anthropic Messages, Google Gemini) for one pairwise judge call.
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from evaluation.openai_responses import responses_create_adaptive
from evaluation.responses_metrics import ResponsesCallMetrics, estimate_cost_usd
from evaluation_claudecodex.pairwise_parsing import (
    _extract_json_candidate,
    _text_from_anthropic_message,
    _text_from_response,
    _usage_dict_from_anthropic_message,
    _usage_dict_from_google_response,
)
from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore[assignment]
try:
    from google.genai import types as google_types
except ImportError:  # pragma: no cover - optional dependency
    google_types = None  # type: ignore[assignment]

JudgeOutcome = tuple[PairwiseJudgeResult, ResponsesCallMetrics]
DEFAULT_JUDGE_MAX_TOKENS = 4096


def _normalized_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    effort = reasoning_effort.strip().lower()
    return effort or None


def _anthropic_budget_tokens(reasoning_effort: str | None) -> int | None:
    effort = _normalized_effort(reasoning_effort)
    if effort is None:
        return None
    return {"low": 1024, "medium": 4096, "high": 8192}.get(effort, 4096)


def _anthropic_output_effort(reasoning_effort: str | None) -> str | None:
    """Anthropic ``output_config.effort``: passed through normalized (the API validates the value)."""
    return _normalized_effort(reasoning_effort)


def _google_thinking_level(reasoning_effort: str | None) -> str | None:
    effort = _normalized_effort(reasoning_effort)
    if effort is None:
        return None
    return {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(effort, "MEDIUM")


def judge_openai(
    client: Any, *, model: str, instructions: str, user: str, temperature: float | None, reasoning_effort: str | None
) -> JudgeOutcome:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user,
        "text": {"format": {"type": "json_object"}},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None and reasoning_effort != "":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    response = responses_create_adaptive(client, **kwargs)
    raw = _text_from_response(response)
    if not raw:
        raise RuntimeError("Empty pairwise judge response from Responses API")
    try:
        result = PairwiseJudgeResult.model_validate_json(raw)
    except ValidationError as e:
        raise RuntimeError(f"Pairwise judge returned invalid JSON: {e}\n---\n{raw[:4000]}") from e
    return result, ResponsesCallMetrics.from_response(response, model=model)


def judge_anthropic(
    client: Any,
    *,
    model: str,
    instructions: str,
    user: str,
    temperature: float | None,
    reasoning_effort: str | None,
    judge_max_tokens: int | None,
) -> JudgeOutcome:
    if Anthropic is None:
        raise RuntimeError("anthropic package is not installed. Please install dependencies first.")
    effort = _anthropic_output_effort(reasoning_effort)
    kwargs: dict[str, Any] = {
        "model": model,
        "system": instructions,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": judge_max_tokens if judge_max_tokens is not None else DEFAULT_JUDGE_MAX_TOKENS,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if effort is not None:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}
    # The Anthropic SDK requires streaming for requests whose configured
    # output ceiling could take more than ten minutes. We still consume the
    # stream into one final Message, so parsing and persisted usage remain
    # identical to the non-streaming route.
    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    raw = _text_from_anthropic_message(response)
    usage = _usage_dict_from_anthropic_message(response)
    context = f"(stop_reason={getattr(response, 'stop_reason', None)!r}, usage={usage!r})"
    if not raw:
        raise RuntimeError(f"Empty pairwise judge response from Anthropic Messages API {context}")
    try:
        result = PairwiseJudgeResult.model_validate_json(_extract_json_candidate(raw))
    except ValidationError as e:
        raise RuntimeError(f"Pairwise judge returned invalid JSON: {e} {context}\n---\n{raw[:4000]}") from e
    est, note = estimate_cost_usd(model, usage)
    metrics = ResponsesCallMetrics(
        usage=usage,
        estimated_cost_usd=est,
        cost_estimate_note=note,
        response_id=str(getattr(response, "id", "")) or None,
    )
    return result, metrics


def judge_google(
    client: Any,
    *,
    model: str,
    instructions: str,
    user: str,
    temperature: float | None,
    reasoning_effort: str | None,
    judge_max_tokens: int | None,
) -> JudgeOutcome:
    if google_types is None:
        raise RuntimeError("google-genai types are not available. Please install/update google-genai.")
    thinking_level = _google_thinking_level(reasoning_effort)
    max_output_tokens = judge_max_tokens if judge_max_tokens is not None else DEFAULT_JUDGE_MAX_TOKENS

    def config(with_thinking: bool) -> Any:
        extra: dict[str, Any] = {}
        if with_thinking:
            extra["thinking_config"] = (
                # Keep internal thoughts hidden to reduce leakage/noise in JSON judge output.
                google_types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=False) if thinking_level is not None else None
            )
        return google_types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **extra,
        )

    try:
        response = client.models.generate_content(model=model, contents=user, config=config(True))
    except Exception as thinking_err:  # SDK/model error types vary across google-genai versions.
        # Fallback for SDK/model versions where thinking config is rejected.
        try:
            response = client.models.generate_content(model=model, contents=user, config=config(False))
        except Exception as retry_err:
            raise retry_err from thinking_err
    raw = (getattr(response, "text", None) or "").strip()
    if not raw:
        raise RuntimeError("Empty pairwise judge response from Google Gemini API")
    try:
        result = PairwiseJudgeResult.model_validate_json(_extract_json_candidate(raw))
    except ValidationError as e:
        raise RuntimeError(f"Pairwise judge returned invalid JSON: {e}\n---\n{raw[:4000]}") from e
    usage = _usage_dict_from_google_response(response)
    est, note = estimate_cost_usd(model, usage)
    metrics = ResponsesCallMetrics(
        usage=usage,
        estimated_cost_usd=est,
        cost_estimate_note=note,
        response_id=str(getattr(response, "response_id", "")) or None,
    )
    return result, metrics
