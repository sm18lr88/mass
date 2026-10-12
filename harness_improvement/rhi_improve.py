# Assembles the optimizer prompt for the next version, guards overwrites/context, and writes dry-run summaries.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from harness_improvement.rhi_best import BestDesign
from harness_improvement.rhi_champion import _baseline_scoreboard
from harness_improvement.rhi_context import RunContext
from harness_improvement.rhi_history import _design_diffs_for_history
from harness_improvement.rhi_io import (
    _enforce_context_budget,
    _write_full_prompt_dump,
)
from harness_improvement.rhi_prompt import _build_single_submission_improvement_prompt


@dataclass
class ImprovementPlan:
    """The optimizer prompt plus every output path for the v(i) -> v(i+1) step."""

    improve_system: str
    improve_user: str
    improve_dir: Path
    next_version: int
    next_design_path: Path
    rendered_query_path: Path
    improve_result_path: Path
    estimated_input_tokens: int
    estimated_total_with_output: int


def prepare_improvement(
    ctx: RunContext,
    best: BestDesign,
    workspace_evidence: str,
    execution_metadata: str | None,
) -> ImprovementPlan:
    """Build and dump the optimizer prompt, refusing to overwrite existing results (SystemExit)."""
    args = ctx.args
    template = ctx.template
    versions_dir = ctx.versions_dir
    rendered_queries_dir = ctx.rendered_queries_dir
    prompts_dir = ctx.prompts_dir
    current_design = ctx.current_design
    history_rows = ctx.history_rows
    best_design_text = best.best_design_text
    best_design_version = best.best_design_version
    best_design_rationale = best.best_design_rationale
    best_baseline_case = best.best_baseline_case
    champ_has_baseline_win = best.champ_has_baseline_win
    scoreboard_champion = best.scoreboard_champion
    history_design_diffs: dict[str, str] | None = None
    if args.history_design_diffs:
        history_design_diffs = _design_diffs_for_history(
            history_rows,
            versions_dir,
            baseline_mode=args.judge_vs_baseline,
        )

    baseline_scoreboard_text: str | None = None
    if args.judge_vs_baseline and args.current_version >= 1:
        baseline_scoreboard_text = _baseline_scoreboard(
            history_rows,
            args.current_version,
            scoreboard_champion,
            champ_has_baseline_win,
        )

    # v0 root anchor: skipped at step 0 (v0 IS the current design) and when
    # v0 is already in the prompt as the champion (dedup).
    v0_design_text: str | None = None
    if (
        args.include_v0_design
        and args.current_version >= 1
        and best_design_version != 0
    ):
        v0_root_path = versions_dir / "multi_agent_design_v0.txt"
        if not v0_root_path.is_file():
            raise SystemExit(f"v0 design file missing: {v0_root_path}")
        v0_design_text = v0_root_path.read_text(encoding="utf-8").strip()

    improve_system, improve_user = _build_single_submission_improvement_prompt(
        template=template,
        current_version=args.current_version,
        current_design=current_design,
        workspace_evidence=workspace_evidence,
        history_rows=history_rows,
        submission_label=args.submission_label,
        execution_metadata=execution_metadata,
        output_mode=args.improvement_output_mode,
        max_design_chars=args.max_design_chars,
        target_design_chars=args.target_design_chars,
        forbid_fixed_agent_limits=args.forbid_fixed_agent_limits,
        compact_era_prompt=args.compact_era_prompt,
        best_design=best_design_text,
        best_version=best_design_version,
        best_rationale=best_design_rationale,
        design_diffs=history_design_diffs,
        v0_design=v0_design_text,
        baseline_mode=args.judge_vs_baseline,
        baseline_scoreboard=baseline_scoreboard_text,
        best_baseline_case=best_baseline_case,
    )
    improve_key = f"v{args.current_version}_to_v{args.current_version + 1}"
    improve_dir = prompts_dir / improve_key
    next_version = args.current_version + 1
    next_design_path = versions_dir / f"multi_agent_design_v{next_version}.txt"
    rendered_query_path = (
        rendered_queries_dir / f"{template.query_name}-v{next_version}.txt"
    )
    improve_result_path = improve_dir / "result.json"
    if not args.dry_run:
        existing_outputs = [
            path
            for path in (next_design_path, rendered_query_path, improve_result_path)
            if path.exists()
        ]
        if existing_outputs:
            joined = "\n- ".join(str(path) for path in existing_outputs)
            raise SystemExit(
                "Refusing to overwrite an existing RHI result. Existing paths:\n"
                f"- {joined}"
            )
    _write_full_prompt_dump(
        improve_dir / "full_prompt.txt",
        system_prompt=improve_system,
        user_prompt=improve_user,
    )
    estimated_input_tokens, estimated_total_with_output = (
        _enforce_context_budget(
            stage="harness optimizer",
            system_prompt=improve_system,
            user_prompt=improve_user,
            max_output_tokens=args.max_output_tokens,
            context_window_tokens=args.context_window_tokens,
            context_safety_tokens=args.context_safety_tokens,
        )
    )
    return ImprovementPlan(
        improve_system=improve_system,
        improve_user=improve_user,
        improve_dir=improve_dir,
        next_version=next_version,
        next_design_path=next_design_path,
        rendered_query_path=rendered_query_path,
        improve_result_path=improve_result_path,
        estimated_input_tokens=estimated_input_tokens,
        estimated_total_with_output=estimated_total_with_output,
    )


def write_dry_run_summary(
    ctx: RunContext, plan: ImprovementPlan, current_run_status_file: Path | None
) -> None:
    """Write and print the dry-run summary (no API call was made)."""
    args = ctx.args
    model = ctx.model
    template = ctx.template
    out_dir = ctx.out_dir
    current_repo = ctx.current_repo
    improve_system = plan.improve_system
    improve_user = plan.improve_user
    improve_dir = plan.improve_dir
    next_version = plan.next_version
    estimated_input_tokens = plan.estimated_input_tokens
    estimated_total_with_output = plan.estimated_total_with_output
    dry_summary = {
        "query_name": template.query_name,
        "query_num": template.query_num,
        "dry_run": True,
        "llm_backend": args.llm_backend,
        "model": model,
        "current_version": args.current_version,
        "current_repo": str(current_repo),
        "current_run_status_file": (
            str(current_run_status_file)
            if current_run_status_file is not None
            else None
        ),
        "full_prompt_path": str(improve_dir / "full_prompt.txt"),
        "prompt_characters": len(improve_system) + len(improve_user),
        "estimated_input_tokens_chars_div_4": estimated_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "context_window_tokens": args.context_window_tokens,
        "context_safety_tokens": args.context_safety_tokens,
        "estimated_total_with_output": estimated_total_with_output,
    }
    dry_summary_path = (
        out_dir / "last_dry_run_summary.json"
        if args.current_version == 0
        else out_dir
        / f"last_dry_run_summary_v{args.current_version}_to_v{next_version}.json"
    )
    dry_summary_path.write_text(
        json.dumps(dry_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(dry_summary, indent=2))
