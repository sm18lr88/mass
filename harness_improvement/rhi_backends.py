# Builds the optimizer backend and the (optionally separate) in-loop judge backend for RHI.
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from harness_improvement.llm_backend import BACKEND_LOCAL_VLLM, JsonLLMBackend


@dataclass
class Backends:
    """Optimizer backend, judge backend, and the judge provenance recorded in history rows."""

    backend: JsonLLMBackend | None
    judge_backend: JsonLLMBackend | None
    judge_backend_name: str
    judge_model: str
    judge_reasoning_effort: str | None
    judge_temperature: float | None
    judge_base_url: str | None


def build_backends(
    args: argparse.Namespace,
    model: str,
    temperature_kw: float | None,
    backend_cls: type[JsonLLMBackend],
) -> Backends:
    """Create and verify backends (none in --dry-run); ``backend_cls`` is injected by the entry module."""
    api_key_env = args.api_key_env or (
        "VLLM_API_KEY"
        if args.llm_backend == BACKEND_LOCAL_VLLM
        else "OPENAI_API_KEY"
    )
    api_key = os.getenv(api_key_env)
    if args.llm_backend == BACKEND_LOCAL_VLLM:
        api_key = api_key or "EMPTY"
    elif not api_key and not args.dry_run:
        raise SystemExit(f"{api_key_env} is not set.")

    backend: JsonLLMBackend | None = None
    if not args.dry_run:
        assert api_key is not None
        backend = backend_cls(
            backend=args.llm_backend,
            model=model,
            api_key=api_key,
            base_url=args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None,
            reasoning_effort=args.reasoning_effort,
            temperature=temperature_kw,
            top_p=args.top_p,
            top_k=args.top_k,
            max_output_tokens=args.max_output_tokens,
            enable_thinking=args.local_thinking,
            seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            background=args.openai_background,
        )
        backend.verify_model()

    # In-loop judge backend. Default (no --judge-llm-backend): the judge IS
    # the main backend object — identical behavior to every earlier run. With
    # the override, judge calls go to a second backend that deliberately does
    # NOT inherit the main sampling knobs (claude-line judge parity: model +
    # reasoning effort only, temperature omitted).
    judge_backend = backend
    judge_backend_name = args.llm_backend
    judge_model = model
    judge_reasoning_effort = args.reasoning_effort
    judge_temperature = temperature_kw
    judge_base_url = args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None
    if args.judge_llm_backend:
        judge_backend_name = args.judge_llm_backend
        judge_model = args.judge_model or (
            "Qwen/Qwen3.5-9B"
            if args.judge_llm_backend == BACKEND_LOCAL_VLLM
            else "gpt-5.5"
        )
        judge_reasoning_effort = args.judge_reasoning_effort
        judge_temperature = None
        judge_base_url = (
            args.judge_base_url
            if args.judge_llm_backend == BACKEND_LOCAL_VLLM
            else None
        )
        if args.judge_llm_backend == BACKEND_LOCAL_VLLM and not args.judge_base_url:
            raise SystemExit("--judge-llm-backend local-vllm requires --judge-base-url")
        judge_api_key_env = (
            "VLLM_API_KEY"
            if args.judge_llm_backend == BACKEND_LOCAL_VLLM
            else "OPENAI_API_KEY"
        )
        judge_api_key = os.getenv(judge_api_key_env)
        if args.judge_llm_backend == BACKEND_LOCAL_VLLM:
            judge_api_key = judge_api_key or "EMPTY"
        elif not judge_api_key and not args.dry_run:
            raise SystemExit(
                f"{judge_api_key_env} is not set (required for --judge-llm-backend)."
            )
        if not args.dry_run:
            assert judge_api_key is not None
            judge_backend = backend_cls(
                backend=args.judge_llm_backend,
                model=judge_model,
                api_key=judge_api_key,
                base_url=judge_base_url,
                reasoning_effort=judge_reasoning_effort,
                temperature=None,
                timeout_seconds=args.timeout_seconds,
                background=args.openai_background,
            )
            judge_backend.verify_model()
    return Backends(
        backend=backend,
        judge_backend=judge_backend,
        judge_backend_name=judge_backend_name,
        judge_model=judge_model,
        judge_reasoning_effort=judge_reasoning_effort,
        judge_temperature=judge_temperature,
        judge_base_url=judge_base_url,
    )
