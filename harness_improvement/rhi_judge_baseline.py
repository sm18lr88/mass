# Baseline-anchored judging (--judge-vs-baseline): v0 seed and current version vs the no-harness repo.
from __future__ import annotations

from pathlib import Path

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
    _enforce_context_budget,
    _utc_now_iso,
    _write_full_prompt_dump,
    _write_json_once,
)


def judge_against_baseline(ctx: RunContext, jb: Backends) -> None:
    """Judge v0 (once) and the current version against the fixed no-harness baseline repo."""
    args = ctx.args
    model = ctx.model
    template = ctx.template
    comparisons_dir = ctx.comparisons_dir
    current_repo = ctx.current_repo
    history_path = ctx.history_path
    history_rows = ctx.history_rows
    run_cost_estimates = ctx.run_cost_estimates
    judge_backend = jb.judge_backend
    judge_backend_name = jb.judge_backend_name
    judge_model = jb.judge_model
    judge_reasoning_effort = jb.judge_reasoning_effort
    judge_temperature = jb.judge_temperature
    judge_base_url = jb.judge_base_url
    baseline_repo = (
        args.baseline_repo_root.expanduser().resolve()
        / f"query{template.query_num}"
    )
    if not baseline_repo.is_dir():
        raise SystemExit(f"Baseline repo not found: {baseline_repo}")

    def _judge_version_vs_baseline(version: int, version_repo: Path) -> None:
        key = f"v{version}_vs_no_harness"
        matching = [r for r in history_rows if r.get("key") == key]
        comp_dir = comparisons_dir / key
        result_path = comp_dir / "result.json"
        if matching:
            _write_json_once(result_path, matching[0])
            return
        if result_path.exists():
            raise SystemExit(
                "Found an orphan baseline comparison result without its "
                f"history row; refusing another judge call: {result_path}"
            )
        if not version_repo.is_dir():
            raise SystemExit(
                f"Repo for baseline judging not found: {version_repo}"
            )
        task_row = {
            "query_id": f"query_{template.query_num}",
            "query": template.task_text_for_eval,
        }
        bundle_a = build_judge_bundle(
            memory_root=baseline_repo,
            task_row=task_row,
            log_path=None,
            scan=scan_deliverables(baseline_repo, template.task_text_for_eval),
            report_max_tokens=args.report_max_tokens,
            json_max_tokens=args.json_max_tokens,
            code_max_tokens=args.code_max_tokens,
            token_count_model=model,
        )
        bundle_b = build_judge_bundle(
            memory_root=version_repo,
            task_row=task_row,
            log_path=None,
            scan=scan_deliverables(version_repo, template.task_text_for_eval),
            report_max_tokens=args.report_max_tokens,
            json_max_tokens=args.json_max_tokens,
            code_max_tokens=args.code_max_tokens,
            token_count_model=model,
        )
        system_prompt = pairwise_judge_system_for_query_num(template.query_num)
        user_prompt = build_pairwise_user_message(
            bundle_a=bundle_a,
            bundle_b=bundle_b,
            label_a="no-harness-single-agent-baseline",
            label_b=f"multi-agent-design-v{version}",
        )
        _write_full_prompt_dump(
            comp_dir / "full_prompt.txt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        _enforce_context_budget(
            stage="baseline judge",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=args.max_output_tokens,
            context_window_tokens=args.context_window_tokens,
            context_safety_tokens=args.context_safety_tokens,
        )
        if args.dry_run:
            return
        assert judge_backend is not None
        judge_call = judge_backend.call_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        verdict = PairwiseJudgeResult.model_validate(judge_call.payload)
        nonlocal_counters["api_calls"] += 1
        record = {
            "key": key,
            "comparison": (
                "no-harness-single-agent-baseline vs "
                f"multi-agent-design-v{version}"
            ),
            "version_b": version,
            "baseline": "no_harness",
            "repo_a": str(baseline_repo),
            "repo_b": str(version_repo),
            "winner": verdict.winner,
            "rationale": verdict.rationale,
            "judged_at_utc": _utc_now_iso(),
            "llm_backend": judge_backend_name,
            "model": judge_model,
            "base_url": judge_base_url,
            "reasoning_effort": judge_reasoning_effort,
            "temperature": judge_temperature,
            "call_metrics": judge_call.metrics,
        }
        cost = judge_call.metrics.get("estimated_cost_usd")
        if isinstance(cost, (int, float)):
            run_cost_estimates.append(float(cost))
        _append_history_row(history_path, record)
        _write_json_once(result_path, record)
        history_rows.append(record)

    nonlocal_counters = {"api_calls": 0}
    seed_repo = (
        args.champion_repo_v0_root.expanduser().resolve()
        / template.query_name
    )
    _judge_version_vs_baseline(0, seed_repo)
    _judge_version_vs_baseline(args.current_version, current_repo)
    ctx.run_api_calls_total += nonlocal_counters["api_calls"]
