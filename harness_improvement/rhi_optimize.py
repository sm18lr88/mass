# Calls the optimizer, validates/normalizes the generated design, and writes v(i+1) artifacts and summaries.
from __future__ import annotations

import json
from pathlib import Path

from harness_improvement.llm_backend import BACKEND_LOCAL_VLLM
from harness_improvement.rhi_backends import Backends
from harness_improvement.rhi_context import RunContext
from harness_improvement.rhi_design import (
    _fixed_agent_limit_matches,
    _normalize_generated_design_block,
    _render_query_with_design,
)
from harness_improvement.rhi_improve import ImprovementPlan
from harness_improvement.rhi_io import (
    _utc_now_iso,
    _write_json_once,
)


def run_optimizer(
    ctx: RunContext,
    jb: Backends,
    plan: ImprovementPlan,
    prev_runs_root: Path | None,
    current_run_status_file: Path | None,
) -> None:
    """Propose v(i+1); rejected designs are recorded and abort the run via SystemExit."""
    args = ctx.args
    query_file = ctx.query_file
    runs_root = ctx.runs_root
    model = ctx.model
    template = ctx.template
    out_dir = ctx.out_dir
    current_repo = ctx.current_repo
    history_path = ctx.history_path
    temperature_kw = ctx.temperature_kw
    run_cost_estimates = ctx.run_cost_estimates
    backend = jb.backend
    improve_system = plan.improve_system
    improve_user = plan.improve_user
    improve_dir = plan.improve_dir
    next_version = plan.next_version
    next_design_path = plan.next_design_path
    rendered_query_path = plan.rendered_query_path
    improve_result_path = plan.improve_result_path
    estimated_input_tokens = plan.estimated_input_tokens
    assert backend is not None
    if args.improvement_output_mode == "json":
        improve_call = backend.call_json(
            system_prompt=improve_system,
            user_prompt=improve_user,
        )
        improve_payload = improve_call.payload
        improved_design_raw = str(
            improve_payload.get("improved_multi_agent_design", "")
        ).strip()
    else:
        improve_call = backend.call_text(
            system_prompt=improve_system,
            user_prompt=improve_user,
        )
        improved_design_raw = improve_call.raw_text.strip()
        improve_payload = {
            "improved_multi_agent_design": improved_design_raw,
            "changes_from_previous": [],
            "why_it_should_improve": [],
            "evidence_used": [],
            "expected_impact": [],
            "verification_checks": [],
            "metadata_note": (
                "design-text fallback: explanatory arrays were intentionally omitted "
                "to avoid JSON-escaping/truncation failures."
            ),
        }
    ctx.run_api_calls_total += 1
    estimated_cost = improve_call.metrics.get("estimated_cost_usd")
    if isinstance(estimated_cost, (int, float)):
        run_cost_estimates.append(float(estimated_cost))
    improved_design = _normalize_generated_design_block(
        improved_design_raw,
        start_marker=args.start_marker,
        end_marker=args.end_marker,
        after_design=template.after_design,
    )
    fixed_limit_matches = (
        _fixed_agent_limit_matches(improved_design)
        if args.forbid_fixed_agent_limits
        else []
    )
    if fixed_limit_matches:
        rejected_result = {
            "rejected_at_utc": _utc_now_iso(),
            "reason": "Generated harness violates the no-fixed-agent-limits policy.",
            "matches": fixed_limit_matches,
            "llm_backend": args.llm_backend,
            "model": model,
            "seed": args.seed,
            "call_metrics": improve_call.metrics,
            "response_json": improve_payload,
        }
        rejected_result_path = (
            improve_dir / f"rejected_result_seed_{args.seed or 'none'}.json"
        )
        _write_json_once(rejected_result_path, rejected_result)
        raise SystemExit(
            "Rejected generated harness because it introduced fixed agent limits: "
            + ", ".join(fixed_limit_matches)
        )
    if args.max_design_chars is not None and len(improved_design) > args.max_design_chars:
        rejected_result = {
            "rejected_at_utc": _utc_now_iso(),
            "reason": "Generated harness exceeds the configured artifact-size limit.",
            "design_characters": len(improved_design),
            "max_design_characters": args.max_design_chars,
            "llm_backend": args.llm_backend,
            "model": model,
            "seed": args.seed,
            "call_metrics": improve_call.metrics,
            "response_json": improve_payload,
        }
        rejected_result_path = (
            improve_dir / f"rejected_result_seed_{args.seed or 'none'}.json"
        )
        _write_json_once(rejected_result_path, rejected_result)
        raise SystemExit(
            "Rejected generated harness because it is too large: "
            f"{len(improved_design)} characters > {args.max_design_chars}"
        )

    next_design_path.write_text(improved_design + "\n", encoding="utf-8")

    rendered_query = _render_query_with_design(template, improved_design)
    rendered_query_path.write_text(rendered_query, encoding="utf-8")

    improve_result = {
        "improvement": f"multi-agent-design-v{args.current_version} -> multi-agent-design-v{next_version}",
        "current_repo": str(current_repo),
        "query_file_source": str(query_file),
        "rendered_query_file": str(rendered_query_path),
        "design_file": str(next_design_path),
        "created_at_utc": _utc_now_iso(),
        "llm_backend": args.llm_backend,
        "model": model,
        "base_url": (
            args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None
        ),
        "reasoning_effort": args.reasoning_effort,
        "temperature": temperature_kw,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
        "local_thinking": (
            args.local_thinking
            if args.llm_backend == BACKEND_LOCAL_VLLM
            else None
        ),
        "seed": args.seed,
        "submission_label": args.submission_label,
        "improvement_output_mode": args.improvement_output_mode,
        "current_run_status_file": (
            str(current_run_status_file)
            if current_run_status_file is not None
            else None
        ),
        "estimated_input_tokens_chars_div_4": estimated_input_tokens,
        "call_metrics": improve_call.metrics,
        "response_json": improve_payload,
    }
    improve_result_path.write_text(
        json.dumps(improve_result, indent=2), encoding="utf-8"
    )

    run_summary = {
        "query_name": template.query_name,
        "query_num": template.query_num,
        "runs_root": str(runs_root) if runs_root is not None else None,
        "prev_runs_root": str(prev_runs_root) if prev_runs_root is not None else None,
        "current_version": args.current_version,
        "next_version": next_version,
        "current_repo": str(current_repo),
        "history_path": str(history_path),
        "next_design_path": str(next_design_path),
        "rendered_query_path": str(rendered_query_path),
        "out_dir": str(out_dir),
        "llm_backend": args.llm_backend,
        "model": model,
        "base_url": (
            args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None
        ),
        "estimated_cost_usd_this_run": (round(sum(run_cost_estimates), 8) if run_cost_estimates else None),
        "api_calls_count_this_run": ctx.run_api_calls_total,
        "latest_improvement_call": improve_call.metrics,
    }
    run_summary_path = (
        out_dir / "last_run_summary.json"
        if args.current_version == 0
        else out_dir
        / f"last_run_summary_v{args.current_version}_to_v{next_version}.json"
    )
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(run_summary, indent=2))
