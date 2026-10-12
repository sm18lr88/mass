# Optimizer evidence: current workspace bundle, results/*.json appendix, and run-status failure notices.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.artifacts import scan_deliverables
from evaluation.bundle import build_judge_bundle, bundle_workspace_evidence
from harness_improvement.rhi_context import RunContext
from harness_improvement.rhi_results_json import _build_all_results_json_evidence


def build_workspace_evidence(ctx: RunContext) -> str:
    """Return the capped workspace evidence for the current version (plus the results JSON appendix)."""
    args = ctx.args
    model = ctx.model
    template = ctx.template
    current_repo = ctx.current_repo
    task_row_current = {
        "query_id": f"query_{template.query_num}",
        "query": template.task_text_for_eval,
    }
    current_scan = scan_deliverables(current_repo, template.task_text_for_eval)
    current_bundle = build_judge_bundle(
        memory_root=current_repo,
        task_row=task_row_current,
        log_path=None,
        scan=current_scan,
        report_max_tokens=args.report_max_tokens,
        json_max_tokens=args.json_max_tokens,
        code_max_tokens=args.code_max_tokens,
        token_count_model=model,
    )
    workspace_evidence = bundle_workspace_evidence(
        current_bundle,
        workspace_section_title=f"# Workspace for current iteration (multi-agent-design-v{args.current_version})",
    )
    if not args.omit_full_results_json:
        results_json_evidence = _build_all_results_json_evidence(
            current_repo,
            template.query_name,
            per_file_max_chars=args.results_json_max_chars,
            total_max_chars=args.results_json_total_max_chars,
        )
        workspace_evidence = (
            f"{workspace_evidence}\n\n"
            "# Full JSON artifacts under results/\n"
            f"{results_json_evidence}"
        )
    return workspace_evidence


def load_execution_metadata(args: argparse.Namespace) -> tuple[str | None, Path | None]:
    """Return (execution metadata text with failure notices, resolved run-status path)."""
    execution_metadata: str | None = None
    current_run_status_file: Path | None = None
    if args.current_run_status_file is not None:
        current_run_status_file = args.current_run_status_file.expanduser().resolve()
        if not current_run_status_file.is_file():
            raise SystemExit(
                f"current run status file not found: {current_run_status_file}"
            )
        try:
            metadata_obj = json.loads(
                current_run_status_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON in current run status file: {current_run_status_file}: {exc}"
            ) from exc
        execution_metadata = json.dumps(metadata_obj, indent=2, ensure_ascii=False)
        run_failed = isinstance(metadata_obj, dict) and (
            metadata_obj.get("exit_code") not in (0, None)
            or metadata_obj.get("trace_has_success_result") is False
        )
        if run_failed:
            execution_metadata += (
                "\n\n"
                "FAILURE NOTICE: this submission's execution ended in a force quit "
                "before a successful final result (see `exit_code` and "
                "`trace_has_success_result` above; when present, `failure.message` "
                "records the halt reason and "
                "`failure.last_tool_calls_most_recent_first` shows what the agent "
                "was doing when it was halted). Treat this failure as first-order "
                "evidence and pursue BOTH goals together when revising the design:\n"
                "(1) Make the next run complete without a force quit, targeting "
                "the specific stall pattern shown in `failure`: tighten each "
                "subagent's 'Output to orchestrator' contract with explicit "
                "completion and acceptance criteria so agents return a definitive "
                "result instead of repeating equivalent tool calls; add "
                "orchestrator progress checks with recovery/re-delegation hops "
                "that redirect a stalled subagent to a narrower follow-up scope; "
                "state in shared protocols how agents must detect and break "
                "repeated tool-call patterns (e.g., after re-reading the same file "
                "or re-running the same command without new information, change "
                "strategy or report the blocker to the orchestrator).\n"
                "(2) Still improve the agentic workflow (hops, contracts) and "
                "agent design (roles, instructions) for deliverable quality as "
                "instructed below.\n"
                "Do not achieve (1) with numerical caps on agents, subagents, "
                "hops, tool calls, turns, or duration, and do not collapse the "
                "design toward single-agent execution; resolve stalls through "
                "acceptance gates, explicit fallback instructions, and "
                "re-delegation."
            )
            if metadata_obj.get("prior_failures"):
                execution_metadata += (
                    "\nRECURRENT FAILURE: `prior_failures` above shows that "
                    "earlier harness version(s) of this task also ended in a "
                    "force quit. Countermeasures written into the current "
                    "design did not change execution behavior. Compare the "
                    "failure fingerprints across versions to diagnose why, and "
                    "revise the approach materially — restructure the workflow "
                    "so the stall pattern cannot arise (e.g., size "
                    "long-running operations to complete within a single tool "
                    "call, or require an explicit strategy change after any "
                    "repeated result) — rather than restating similar "
                    "anti-repetition rules."
                )
        elif isinstance(metadata_obj, dict) and metadata_obj.get("prior_failures"):
            execution_metadata += (
                "\n\n"
                "RESOLVED FAILURE HISTORY: `prior_failures` above shows that "
                "earlier harness version(s) of this task ended in a force "
                "quit, while the current version completed successfully. "
                "Compare the current design with the prior failure "
                "fingerprints to identify which provisions plausibly fixed "
                "the failure, and PRESERVE them in the revised design — do "
                "not remove or weaken anti-stall protections that are now "
                "working. Otherwise continue improving the agentic workflow "
                "(hops, contracts) and agent design (roles, instructions) "
                "for deliverable quality as instructed below."
            )
    return execution_metadata, current_run_status_file
