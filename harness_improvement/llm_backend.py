# JSON-mode LLM client (OpenAI Responses or local vLLM chat) for RHI and judging.
"""JSON-only LLM backends for recursive harness improvement.

The historical RHI path used OpenAI's Responses API.  The local path uses the
OpenAI-compatible Chat Completions endpoint exposed by vLLM.  Keeping both
behind one small interface makes the evaluator and harness optimizer use the
same explicitly selected backend.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from evaluation.openai_responses import responses_create_adaptive
from evaluation.responses_metrics import ResponsesCallMetrics
from harness_improvement.llm_json import (
    LLMJsonError,
    chat_usage_metrics as _chat_usage_metrics,
    extract_json_object as _extract_json_object,
    responses_text as _responses_text,
)

__all__ = [
    "BACKEND_CHOICES",
    "BACKEND_LOCAL_VLLM",
    "BACKEND_OPENAI_RESPONSES",
    "BackgroundResponseTimeout",
    "JsonCallResult",
    "JsonLLMBackend",
    "LLMJsonError",
]

BACKEND_OPENAI_RESPONSES = "openai-responses"
BACKEND_LOCAL_VLLM = "local-vllm"
BACKEND_CHOICES = (BACKEND_OPENAI_RESPONSES, BACKEND_LOCAL_VLLM)

# Read/write/pool timeout is the caller's ``timeout_seconds`` (default 1 hour):
# long non-streaming reasoning generations legitimately produce no bytes for
# many minutes. Connecting is fast or never, so it gets a short bound.
DEFAULT_TIMEOUT_SECONDS = 3600.0
CONNECT_TIMEOUT_SECONDS = 30.0
MAX_CONSECUTIVE_POLL_ERRORS = 5


class BackgroundResponseTimeout(TimeoutError):
    """A background Responses job did not reach a terminal status in time."""


def _await_background_response(
    client: OpenAI,
    response: Any,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 30.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Poll a background Responses job until it reaches a terminal status.

    Long non-streaming Responses calls can exceed NAT/conntrack idle windows
    (observed on GCP: the connection stays ESTABLISHED locally but the reply
    can never arrive, so the client waits until its own timeout). Background
    mode sidesteps that: the create returns immediately and each poll is a
    short, fresh HTTPS request.

    Polling is bounded by ``timeout_seconds`` (raises BackgroundResponseTimeout)
    and by consecutive retrieve failures (the last error is re-raised).
    """
    deadline = clock() + timeout_seconds
    consecutive_poll_errors = 0
    while getattr(response, "status", None) in ("queued", "in_progress"):
        if clock() > deadline:
            raise BackgroundResponseTimeout(
                f"Background response {response.id} still "
                f"{response.status!r} after {timeout_seconds:.0f}s"
            )
        sleep(poll_interval_seconds)
        try:
            response = client.responses.retrieve(response.id)
            consecutive_poll_errors = 0
        except Exception:
            consecutive_poll_errors += 1
            if consecutive_poll_errors > MAX_CONSECUTIVE_POLL_ERRORS:
                raise
    if getattr(response, "status", None) != "completed":
        raise RuntimeError(
            f"Background response {response.id} ended with status "
            f"{getattr(response, 'status', None)!r}: "
            f"error={getattr(response, 'error', None)!r} "
            f"incomplete_details={getattr(response, 'incomplete_details', None)!r}"
        )
    return response


@dataclass
class JsonCallResult:
    payload: dict[str, Any]
    raw_text: str
    metrics: dict[str, Any]


class JsonLLMBackend:
    """Call either OpenAI Responses or a local vLLM chat endpoint for JSON."""

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        enable_thinking: bool = True,
        seed: int | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        background: bool = False,
    ) -> None:
        if backend not in BACKEND_CHOICES:
            raise ValueError(f"Unsupported backend: {backend}")
        if backend == BACKEND_LOCAL_VLLM and not base_url:
            raise ValueError("local-vllm requires base_url")
        self.backend = backend
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.background = background and backend == BACKEND_OPENAI_RESPONSES
        # max_retries=0: a retry could duplicate a judgment/generation.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(
                timeout_seconds,
                connect=min(CONNECT_TIMEOUT_SECONDS, timeout_seconds),
            ),
            max_retries=0,
        )

    def verify_model(self) -> None:
        """Fail early when the selected model is not served by local vLLM."""
        if self.backend != BACKEND_LOCAL_VLLM:
            return
        models = self.client.models.list()
        ids = {str(item.id) for item in models.data}
        if self.model not in ids:
            raise RuntimeError(
                f"Local endpoint is reachable but does not serve {self.model!r}; "
                f"available models: {sorted(ids)}"
            )

    def _responses_kwargs(
        self, *, system_prompt: str, user_prompt: str, json_mode: bool
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.background:
            kwargs["background"] = True
        return kwargs

    def _responses_call(self, kwargs: dict[str, Any]) -> Any:
        response = responses_create_adaptive(self.client, **kwargs)
        if self.background:
            response = _await_background_response(
                self.client, response, timeout_seconds=self.timeout_seconds
            )
        return response

    def _local_text(self, response: Any, *, what: str, hint: str = "") -> str:
        if not response.choices:
            raise RuntimeError("Local vLLM returned no completion choices.")
        message = response.choices[0].message
        raw_content = message.content
        raw = raw_content.strip() if isinstance(raw_content, str) else ""
        if not raw:
            reasoning = getattr(message, "reasoning_content", None)
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(
                f"Local vLLM returned no final {what} "
                f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars})."
                f"{hint}"
            )
        return raw

    def call_json(self, *, system_prompt: str, user_prompt: str) -> JsonCallResult:
        if self.backend == BACKEND_OPENAI_RESPONSES:
            response = self._responses_call(
                self._responses_kwargs(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
            )
            raw = _responses_text(response)
            metrics = ResponsesCallMetrics.from_response(
                response, model=self.model
            ).to_json_dict()
            return JsonCallResult(
                payload=_extract_json_object(raw),
                raw_text=raw,
                metrics=metrics,
            )

        response = self._local_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )
        raw = self._local_text(
            response,
            what="JSON content",
            hint=" This usually means the output ceiling was consumed by reasoning.",
        )
        return JsonCallResult(
            payload=_extract_json_object(raw),
            raw_text=raw,
            metrics=_chat_usage_metrics(response, model=self.model),
        )

    def call_text(self, *, system_prompt: str, user_prompt: str) -> JsonCallResult:
        """Return unwrapped text while retaining the same metrics container."""
        if self.backend == BACKEND_OPENAI_RESPONSES:
            response = self._responses_call(
                self._responses_kwargs(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=False,
                )
            )
            raw = _responses_text(response)
            if not raw:
                raise RuntimeError("OpenAI Responses returned no final text.")
            metrics = ResponsesCallMetrics.from_response(
                response, model=self.model
            ).to_json_dict()
            return JsonCallResult(payload={}, raw_text=raw, metrics=metrics)

        response = self._local_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=False,
        )
        raw = self._local_text(response, what="design text")
        return JsonCallResult(
            payload={},
            raw_text=raw,
            metrics=_chat_usage_metrics(response, model=self.model),
        )

    def _local_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                **({"top_k": self.top_k} if self.top_k is not None else {}),
            },
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed

        return self.client.chat.completions.create(**kwargs)
