# Per-run RHI state: resolved output dirs, query template, current design/repo, history, counters.
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness_improvement.llm_backend import BACKEND_LOCAL_VLLM
from harness_improvement.rhi_io import _load_history
from harness_improvement.rhi_template import (
    QueryTemplate,
    _read_query_template,
    _resolve_iteration_repo,
)


@dataclass
class RunContext:
    """Resolved inputs plus the mutable cost/call counters shared by the RHI step functions."""

    args: argparse.Namespace
    query_file: Path
    runs_root: Path | None
    model: str
    template: QueryTemplate
    out_dir: Path
    versions_dir: Path
    rendered_queries_dir: Path
    prompts_dir: Path
    comparisons_dir: Path
    records_dir: Path
    current_design: str
    current_repo: Path
    history_path: Path
    history_rows: list[dict[str, Any]]
    temperature_kw: float | None
    run_cost_estimates: list[float] = field(default_factory=list)
    run_api_calls_total: int = 0


def prepare_run_context(
    args: argparse.Namespace, query_file: Path, runs_root: Path | None
) -> RunContext:
    """Create output dirs, seed v0, and load the current design, repo, and history (no API calls)."""
    model = args.model or (
        "Qwen/Qwen3.5-9B"
        if args.llm_backend == BACKEND_LOCAL_VLLM
        else "gpt-5.5"
    )

    template = _read_query_template(
        query_file,
        start_marker=args.start_marker,
        end_marker=args.end_marker,
    )

    if args.out_dir is None:
        assert runs_root is not None
        out_dir = (
            query_file.parent.parent.parent
            / "harness_improvement"
            / "runs"
            / runs_root.name
            / template.query_name
        ).resolve()
    else:
        out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    versions_dir = out_dir / "versions"
    rendered_queries_dir = out_dir / "updated_queries"
    prompts_dir = out_dir / "evaluation_prompts"
    comparisons_dir = out_dir / "comparisons"
    records_dir = out_dir / "records"
    for d in (versions_dir, rendered_queries_dir, prompts_dir, comparisons_dir, records_dir):
        d.mkdir(parents=True, exist_ok=True)

    v0_path = versions_dir / "multi_agent_design_v0.txt"
    if not v0_path.is_file():
        v0_path.write_text(template.design_block.strip() + "\n", encoding="utf-8")

    current_design_path = versions_dir / f"multi_agent_design_v{args.current_version}.txt"
    if not current_design_path.is_file() and args.current_version > 0:
        # Recovery path: if user has rendered/edited query for this version, bootstrap design from it.
        candidate_query = rendered_queries_dir / f"{template.query_name}-v{args.current_version}.txt"
        if candidate_query.is_file():
            recovered = _read_query_template(
                candidate_query,
                start_marker=args.start_marker,
                end_marker=args.end_marker,
            ).design_block
            current_design_path.write_text(recovered.strip() + "\n", encoding="utf-8")
    if not current_design_path.is_file():
        raise SystemExit(
            f"Missing current design file: {current_design_path}. "
            f"Expected existing history up to v{args.current_version}."
        )
    current_design = current_design_path.read_text(encoding="utf-8").strip()

    current_repo = (
        args.current_repo.expanduser().resolve()
        if args.current_repo is not None
        else _resolve_iteration_repo(runs_root, template.query_name, args.current_version)
    )
    if not current_repo.is_dir():
        raise SystemExit(f"Current iteration repo not found: {current_repo}")

    history_path = records_dir / "comparison_history.jsonl"
    history_rows = _load_history(history_path)

    temperature_kw: float | None = None if args.omit_temperature else args.temperature
    return RunContext(
        args=args,
        query_file=query_file,
        runs_root=runs_root,
        model=model,
        template=template,
        out_dir=out_dir,
        versions_dir=versions_dir,
        rendered_queries_dir=rendered_queries_dir,
        prompts_dir=prompts_dir,
        comparisons_dir=comparisons_dir,
        records_dir=records_dir,
        current_design=current_design,
        current_repo=current_repo,
        history_path=history_path,
        history_rows=history_rows,
        temperature_kw=temperature_kw,
    )
