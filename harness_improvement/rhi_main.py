# Orchestrates one RHI step: validate, judge, resolve champion, build evidence and prompt, call optimizer.
from __future__ import annotations

import argparse

from harness_improvement.llm_backend import JsonLLMBackend
from harness_improvement.rhi_backends import build_backends
from harness_improvement.rhi_best import resolve_best_design
from harness_improvement.rhi_context import prepare_run_context
from harness_improvement.rhi_evidence import build_workspace_evidence, load_execution_metadata
from harness_improvement.rhi_improve import prepare_improvement, write_dry_run_summary
from harness_improvement.rhi_judge_adjacent import judge_adjacent, resolve_prev_runs_root
from harness_improvement.rhi_judge_baseline import judge_against_baseline
from harness_improvement.rhi_optimize import run_optimizer
from harness_improvement.rhi_validate import validate_args


def run(args: argparse.Namespace, *, backend_cls: type[JsonLLMBackend]) -> int:
    """Run one RHI step in the original order of side effects; return the process exit code."""
    query_file, runs_root = validate_args(args)
    ctx = prepare_run_context(args, query_file, runs_root)
    jb = build_backends(args, ctx.model, ctx.temperature_kw, backend_cls)
    prev_runs_root = resolve_prev_runs_root(args, runs_root)

    # Step A: in-loop pairwise judging (baseline-anchored or adjacent).
    if args.current_version >= 1 and args.judge_vs_baseline:
        judge_against_baseline(ctx, jb)
    if args.current_version >= 1 and not args.judge_vs_baseline:
        judge_adjacent(ctx, jb, prev_runs_root)

    # Step B: single-submission evaluation prompt to improve v(i) -> v(i+1).
    workspace_evidence = build_workspace_evidence(ctx)
    execution_metadata, current_run_status_file = load_execution_metadata(args)
    best = resolve_best_design(ctx, jb)
    if best is None:  # --evaluate-only finished after champion resolution
        return 0
    plan = prepare_improvement(ctx, best, workspace_evidence, execution_metadata)
    if args.dry_run:
        write_dry_run_summary(ctx, plan, current_run_status_file)
        return 0
    run_optimizer(ctx, jb, plan, prev_runs_root, current_run_status_file)
    return 0
