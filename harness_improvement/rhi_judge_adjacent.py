# Adjacent v(i-1)-vs-v(i) pairwise judging with immutable history and strict provenance checks.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation.artifacts import scan_deliverables
from evaluation.bundle import build_judge_bundle
from evaluation_claudecodex.pairwise_judge import (
    build_pairwise_user_message,
)
from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num
from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult
from harness_improvement.llm_backend import BACKEND_LOCAL_VLLM
from harness_improvement.rhi_backends import Backends
from harness_improvement.rhi_context import RunContext
from harness_improvement.rhi_history import _history_key
from harness_improvement.rhi_io import (
    _append_history_row,
    _enforce_context_budget,
    _utc_now_iso,
    _write_full_prompt_dump,
    _write_json_once,
)
from harness_improvement.rhi_template import (
    _infer_versioned_runs_root,
    _resolve_iteration_repo,
)


def _adjacent_history_provenance_errors(
    row: dict[str, Any],
    *,
    comparison_key: str,
    prev_version: int,
    current_version: int,
    prev_repo: Path,
    current_repo: Path,
    llm_backend: str,
    model: str,
    base_url: str | None,
) -> list[str]:
    """Validate that reused pairwise history belongs to this exact RHI step."""
    errors: list[str] = []
    expected_scalars = {
        "key": comparison_key,
        "version_a": prev_version,
        "version_b": current_version,
        "llm_backend": llm_backend,
        "model": model,
    }
    for field, expected in expected_scalars.items():
        if row.get(field) != expected:
            errors.append(
                f"{field}={row.get(field)!r}, expected {expected!r}"
            )

    expected_repos = {
        "repo_a": prev_repo,
        "repo_b": current_repo,
    }
    for field, expected in expected_repos.items():
        raw = row.get(field)
        if not isinstance(raw, str) or Path(raw).expanduser().resolve() != expected:
            errors.append(f"{field}={raw!r}, expected {str(expected)!r}")

    if llm_backend == BACKEND_LOCAL_VLLM:
        actual_url = str(row.get("base_url") or "").rstrip("/")
        expected_url = str(base_url or "").rstrip("/")
        if actual_url != expected_url:
            errors.append(
                f"base_url={row.get('base_url')!r}, expected {base_url!r}"
            )
        metrics = row.get("call_metrics")
        cost = metrics.get("estimated_cost_usd") if isinstance(metrics, dict) else None
        if cost != 0.0:
            errors.append(
                f"call_metrics.estimated_cost_usd={cost!r}, expected 0.0"
            )
    return errors


def resolve_prev_runs_root(args: argparse.Namespace, runs_root: Path | None) -> Path | None:
    """Resolve the previous version's runs root (explicit, inferred _v(N-1) sibling, or --runs-root)."""
    prev_runs_root: Path | None = None
    if args.current_version >= 1:
        prev_version = args.current_version - 1
        if args.previous_repo is not None:
            prev_runs_root = None
        elif args.prev_runs_root is not None:
            prev_runs_root = args.prev_runs_root.expanduser().resolve()
        else:
            assert runs_root is not None
            prev_runs_root = _infer_versioned_runs_root(runs_root, prev_version)
            if prev_runs_root is None:
                prev_runs_root = runs_root

        if prev_runs_root is not None and not prev_runs_root.is_dir():
            raise SystemExit(f"prev runs root not found: {prev_runs_root}")
    return prev_runs_root


def judge_adjacent(ctx: RunContext, jb: Backends, prev_runs_root: Path | None) -> None:
    """Record (or reuse) the v(i-1) vs v(i) verdict in the comparison history."""
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
    prev_version = args.current_version - 1
    comparison_key = _history_key(prev_version, args.current_version)
    prev_repo = (
        args.previous_repo.expanduser().resolve()
        if args.previous_repo is not None
        else _resolve_iteration_repo(
            prev_runs_root, template.query_name, prev_version
        )
    )
    if not prev_repo.is_dir():
        raise SystemExit(
            "Previous iteration repo required for adjacent self-comparison "
            f"was not found: {prev_repo}"
        )

    matching_history = [
        row for row in history_rows if row.get("key") == comparison_key
    ]
    if len(matching_history) > 1 and args.strict_adjacent_history_provenance:
        raise SystemExit(
            f"Expected exactly one {comparison_key} history row, found "
            f"{len(matching_history)} in {history_path}"
        )
    if matching_history and args.strict_adjacent_history_provenance:
        provenance_errors = _adjacent_history_provenance_errors(
            matching_history[0],
            comparison_key=comparison_key,
            prev_version=prev_version,
            current_version=args.current_version,
            prev_repo=prev_repo,
            current_repo=current_repo,
            llm_backend=judge_backend_name,
            model=judge_model,
            base_url=judge_base_url,
        )
        if provenance_errors:
            details = "\n- ".join(provenance_errors)
            raise SystemExit(
                "Existing adjacent history provenance check failed:\n"
                f"- {details}"
            )

    comp_dir = comparisons_dir / comparison_key
    comparison_result_path = comp_dir / "result.json"
    if matching_history:
        # Recover the immutable per-comparison result if a process stopped
        # after appending history but before mirroring that row to result.json.
        _write_json_once(comparison_result_path, matching_history[0])
    elif comparison_result_path.exists():
        raise SystemExit(
            "Found an orphan comparison result without its history row; "
            "refusing another judge call until provenance is repaired: "
            f"{comparison_result_path}"
        )

    if not matching_history:
        task_row = {
            "query_id": f"query_{template.query_num}",
            "query": template.task_text_for_eval,
        }
        scan_a = scan_deliverables(prev_repo, template.task_text_for_eval)
        scan_b = scan_deliverables(current_repo, template.task_text_for_eval)
        bundle_a = build_judge_bundle(
            memory_root=prev_repo,
            task_row=task_row,
            log_path=None,
            scan=scan_a,
            report_max_tokens=args.report_max_tokens,
            json_max_tokens=args.json_max_tokens,
            code_max_tokens=args.code_max_tokens,
            token_count_model=model,
        )
        bundle_b = build_judge_bundle(
            memory_root=current_repo,
            task_row=task_row,
            log_path=None,
            scan=scan_b,
            report_max_tokens=args.report_max_tokens,
            json_max_tokens=args.json_max_tokens,
            code_max_tokens=args.code_max_tokens,
            token_count_model=model,
        )
        label_a = f"multi-agent-design-v{prev_version}"
        label_b = f"multi-agent-design-v{args.current_version}"
        system_prompt = pairwise_judge_system_for_query_num(template.query_num)
        user_prompt = build_pairwise_user_message(
            bundle_a=bundle_a,
            bundle_b=bundle_b,
            label_a=label_a,
            label_b=label_b,
        )
        _write_full_prompt_dump(
            comp_dir / "full_prompt.txt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        _enforce_context_budget(
            stage="pairwise judge",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=args.max_output_tokens,
            context_window_tokens=args.context_window_tokens,
            context_safety_tokens=args.context_safety_tokens,
        )
        if not args.dry_run:
            assert judge_backend is not None
            judge_call = judge_backend.call_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            verdict = PairwiseJudgeResult.model_validate(judge_call.payload)
            ctx.run_api_calls_total += 1
            record = {
                "key": comparison_key,
                "comparison": f"multi-agent-design-v{prev_version} vs multi-agent-design-v{args.current_version}",
                "version_a": prev_version,
                "version_b": args.current_version,
                "repo_a": str(prev_repo),
                "repo_b": str(current_repo),
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
            estimated_cost = judge_call.metrics.get("estimated_cost_usd")
            if isinstance(estimated_cost, (int, float)):
                run_cost_estimates.append(float(estimated_cost))
            _append_history_row(history_path, record)
            _write_json_once(comparison_result_path, record)
            history_rows.append(record)
            matching_history = [record]

    if args.strict_adjacent_history_provenance and not args.dry_run:
        if len(matching_history) != 1:
            raise SystemExit(
                f"Expected exactly one {comparison_key} history row after "
                f"self-comparison, found {len(matching_history)}"
            )
        provenance_errors = _adjacent_history_provenance_errors(
            matching_history[0],
            comparison_key=comparison_key,
            prev_version=prev_version,
            current_version=args.current_version,
            prev_repo=prev_repo,
            current_repo=current_repo,
            llm_backend=judge_backend_name,
            model=judge_model,
            base_url=judge_base_url,
        )
        if provenance_errors:
            details = "\n- ".join(provenance_errors)
            raise SystemExit(
                "Adjacent history provenance check failed:\n"
                f"- {details}"
            )
