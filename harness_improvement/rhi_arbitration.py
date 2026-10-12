# Champion arbitration: one judge call between the reigning champion and the current version.
from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.artifacts import scan_deliverables
from evaluation.bundle import build_judge_bundle
from evaluation_claudecodex.pairwise_judge import (
    build_pairwise_user_message,
)
from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num
from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult
from harness_improvement.rhi_backends import Backends
from harness_improvement.rhi_context import RunContext
from harness_improvement.rhi_io import (
    _append_history_row,
    _utc_now_iso,
    _write_full_prompt_dump,
    _write_json_once,
)


def arbitrate_champion(
    ctx: RunContext,
    jb: Backends,
    champ: int,
    champion_path: Path,
    champion_rows: list[dict[str, Any]],
) -> int:
    """Judge champion vs current, append the row to the champion ledger, and return the new champion."""
    args = ctx.args
    model = ctx.model
    template = ctx.template
    comparisons_dir = ctx.comparisons_dir
    current_repo = ctx.current_repo
    run_cost_estimates = ctx.run_cost_estimates
    judge_backend = jb.judge_backend
    judge_backend_name = jb.judge_backend_name
    judge_model = jb.judge_model
    judge_reasoning_effort = jb.judge_reasoning_effort
    judge_temperature = jb.judge_temperature
    judge_base_url = jb.judge_base_url
    assert judge_backend is not None
    if champ == 0:
        champ_repo = (
            args.champion_repo_v0_root.expanduser().resolve()
            / template.query_name
        )
    else:
        champ_repo = Path(
            args.champion_repo_root_pattern.format(version=champ)
        ).expanduser().resolve() / template.query_name
    if not champ_repo.is_dir():
        raise SystemExit(
            f"Champion arbitration repo not found: {champ_repo}"
        )
    arb_key = f"champion_v{champ}_vs_v{args.current_version}"
    arb_task_row = {
        "query_id": f"query_{template.query_num}",
        "query": template.task_text_for_eval,
    }
    arb_bundle_a = build_judge_bundle(
        memory_root=champ_repo,
        task_row=arb_task_row,
        log_path=None,
        scan=scan_deliverables(champ_repo, template.task_text_for_eval),
        report_max_tokens=args.report_max_tokens,
        json_max_tokens=args.json_max_tokens,
        code_max_tokens=args.code_max_tokens,
        token_count_model=model,
    )
    arb_bundle_b = build_judge_bundle(
        memory_root=current_repo,
        task_row=arb_task_row,
        log_path=None,
        scan=scan_deliverables(current_repo, template.task_text_for_eval),
        report_max_tokens=args.report_max_tokens,
        json_max_tokens=args.json_max_tokens,
        code_max_tokens=args.code_max_tokens,
        token_count_model=model,
    )
    arb_system = pairwise_judge_system_for_query_num(template.query_num)
    arb_user = build_pairwise_user_message(
        bundle_a=arb_bundle_a,
        bundle_b=arb_bundle_b,
        label_a=f"multi-agent-design-v{champ}",
        label_b=f"multi-agent-design-v{args.current_version}",
    )
    arb_dir = comparisons_dir / arb_key
    _write_full_prompt_dump(
        arb_dir / "full_prompt.txt",
        system_prompt=arb_system,
        user_prompt=arb_user,
    )
    arb_call = judge_backend.call_json(
        system_prompt=arb_system, user_prompt=arb_user
    )
    arb_verdict = PairwiseJudgeResult.model_validate(arb_call.payload)
    ctx.run_api_calls_total += 1
    arb_record = {
        "key": arb_key,
        "comparison": (
            f"multi-agent-design-v{champ} vs "
            f"multi-agent-design-v{args.current_version}"
        ),
        "kind": "champion_arbitration",
        "version_a": champ,
        "version_b": args.current_version,
        "repo_a": str(champ_repo),
        "repo_b": str(current_repo),
        "winner": arb_verdict.winner,
        "rationale": arb_verdict.rationale,
        "judged_at_utc": _utc_now_iso(),
        "llm_backend": judge_backend_name,
        "model": judge_model,
        "base_url": judge_base_url,
        "reasoning_effort": judge_reasoning_effort,
        "temperature": judge_temperature,
        "call_metrics": arb_call.metrics,
    }
    arb_cost = arb_call.metrics.get("estimated_cost_usd")
    if isinstance(arb_cost, (int, float)):
        run_cost_estimates.append(float(arb_cost))
    _append_history_row(champion_path, arb_record)
    _write_json_once(arb_dir / "result.json", arb_record)
    champion_rows.append(arb_record)
    if arb_verdict.winner == "B":
        champ = args.current_version
    return champ
