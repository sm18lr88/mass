# Pairwise judge entry points: build the A-vs-B user message and dispatch to a provider adapter.
from __future__ import annotations

from typing import Any, Literal

from evaluation.bundle import JudgeBundle, bundle_workspace_evidence
from evaluation.queries_loader import numeric_query_id
from evaluation.responses_metrics import ResponsesCallMetrics
from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num, pairwise_user_suffix
from evaluation_claudecodex.pairwise_providers import judge_anthropic, judge_google, judge_openai
from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult

JudgeProvider = Literal["openai", "anthropic", "google"]


def build_pairwise_user_message(
    bundle_a: JudgeBundle,
    bundle_b: JudgeBundle,
    *,
    label_a: str,
    label_b: str,
) -> str:
    """Render the task, run-type labels, both workspaces, and the rubric as one user message."""
    if bundle_a.task_text != bundle_b.task_text:
        raise ValueError("Pairwise bundles must share the same task text")
    wa = bundle_workspace_evidence(
        bundle_a,
        workspace_section_title="# Workspace for submission **A** (evidence only)",
    )
    wb = bundle_workspace_evidence(
        bundle_b,
        workspace_section_title="# Workspace for submission **B** (evidence only)",
    )
    body = "\n\n".join(
        [
            "# Task",
            "",
            bundle_a.task_text,
            "",
            "# Labels (for orientation only)",
            f"- Submission **A** corresponds to run type: `{label_a}`",
            f"- Submission **B** corresponds to run type: `{label_b}`",
            "",
            wa,
            "",
            wb,
            "",
            pairwise_user_suffix().strip(),
        ]
    )
    return body.strip()


def run_pairwise_judge(
    *,
    client: Any,
    provider: JudgeProvider = "openai",
    model: str,
    bundle_a: JudgeBundle,
    bundle_b: JudgeBundle,
    label_a: str,
    label_b: str,
    temperature: float | None = None,
    reasoning_effort: str | None = "medium",
    judge_max_tokens: int | None = None,
) -> tuple[PairwiseJudgeResult, ResponsesCallMetrics]:
    """Judge submission A vs B for the same task with the given provider client."""
    qn = numeric_query_id(bundle_a.task_query_id)
    if bundle_b.task_query_id != bundle_a.task_query_id:
        raise ValueError("Pairwise bundles must share the same task_query_id")
    instructions = pairwise_judge_system_for_query_num(qn)
    user = build_pairwise_user_message(bundle_a, bundle_b, label_a=label_a, label_b=label_b)
    common: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "user": user,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    provider_norm = provider.strip().lower()
    if provider_norm == "openai":
        return judge_openai(client, **common)
    if provider_norm == "anthropic":
        return judge_anthropic(client, judge_max_tokens=judge_max_tokens, **common)
    if provider_norm == "google":
        return judge_google(client, judge_max_tokens=judge_max_tokens, **common)
    raise ValueError(f"Unsupported provider {provider!r}; expected 'openai', 'anthropic', or 'google'.")
