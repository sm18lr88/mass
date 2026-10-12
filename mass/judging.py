# Pairwise trajectory judging: self-judge (current model) for ranking, external judges for reporting.
"""Self-evaluation uses the current generation; reporting uses external judges."""

import os
import random
from collections.abc import Collection, Iterator
from pathlib import Path

# Explicit per-request timeout for the external provider clients (the SDKs' default read timeout).
EXTERNAL_TIMEOUT_S = 600.0


def bundles(task: int, workspace_a: Path, workspace_b: Path) -> list:
    from evaluation.bundle import build_judge_bundle

    from .io import ROOT

    task_text = (ROOT / f"tasks/bare/query{task}.txt").read_text(encoding="utf-8")
    task_text = task_text.split("Use uv for all Python workflows")[0].replace("\\n", "\n").strip()
    row = {"query_id": f"query_{task}", "query": task_text}
    return [
        build_judge_bundle(
            memory_root=p, task_row=row, log_path=None, report_max_tokens=None, json_max_tokens=50000, code_max_tokens=10000, token_count_model="gpt-5.5"
        )
        for p in (workspace_a, workspace_b)
    ]


def self_judge(config: dict, task: int, workspace_a: Path, workspace_b: Path, seed: int) -> dict:
    from evaluation_claudecodex.pairwise_judge import build_pairwise_user_message
    from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num
    from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult
    from harness_improvement.llm_backend import JsonLLMBackend

    a, b = bundles(task, workspace_a, workspace_b)
    backend = JsonLLMBackend(
        backend="local-vllm",
        model=config["model_id"],
        base_url=config["endpoint"],
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_output_tokens=49152,
        enable_thinking=True,
        seed=seed,
    )
    backend.verify_model()
    result = backend.call_json(
        system_prompt=pairwise_judge_system_for_query_num(task), user_prompt=build_pairwise_user_message(a, b, label_a="candidate-a", label_b="candidate-b")
    )
    verdict = PairwiseJudgeResult.model_validate(result.payload)
    return {"winner": verdict.winner, "rationale": verdict.rationale, "model": config["model_id"], "seed": seed, "metrics": result.metrics}


def external_judgments(task: int, workspace_a: Path, workspace_b: Path, skip: Collection[tuple[str, int]] = ()) -> Iterator[dict]:
    from anthropic import Anthropic
    from openai import OpenAI

    from evaluation_claudecodex.pairwise_judge import JudgeProvider, run_pairwise_judge

    a, b = bundles(task, workspace_a, workspace_b)
    judges: list[tuple[JudgeProvider, str, str, OpenAI | Anthropic]] = [
        ("openai", "gpt-5.5", "xhigh", OpenAI(max_retries=0, timeout=EXTERNAL_TIMEOUT_S)),
        ("anthropic", "claude-opus-4-8", "max", Anthropic(max_retries=0, timeout=EXTERNAL_TIMEOUT_S)),
    ]
    for provider, model, effort, client in judges:
        for seed in [42, 43, 44]:
            if (provider, seed) in skip:
                continue
            # Provider APIs do not expose identical deterministic seed controls.
            # The seed controls presentation; each call is an independent judgment.
            swapped = random.Random(seed).choice([False, True])
            left, right = (b, a) if swapped else (a, b)
            verdict, metrics = run_pairwise_judge(
                client=client,
                provider=provider,
                model=model,
                bundle_a=left,
                bundle_b=right,
                label_a="candidate-a",
                label_b="candidate-b",
                reasoning_effort=effort,
                judge_max_tokens=32768,
            )
            winner = {"A": "B", "B": "A"}.get(verdict.winner, verdict.winner) if swapped else verdict.winner
            yield {
                "provider": provider,
                "model": model,
                "seed": seed,
                "swapped": swapped,
                "winner": winner,
                "rationale": verdict.rationale,
                "metrics": metrics.to_json_dict(),
            }
