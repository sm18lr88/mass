# BEST-H champion resolution: derive the champion, arbitrate, and pick the previous-best design.
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from harness_improvement.rhi_arbitration import arbitrate_champion
from harness_improvement.rhi_backends import Backends
from harness_improvement.rhi_champion import _derive_champion, _derive_champion_baseline
from harness_improvement.rhi_context import RunContext
from harness_improvement.rhi_history import _history_key
from harness_improvement.rhi_io import _load_history


@dataclass
class BestDesign:
    """Previous-best design shown to the optimizer and the champion state for the scoreboard."""

    best_design_text: str | None
    best_design_version: int | None
    best_design_rationale: str | None
    best_baseline_case: str | None
    champ_has_baseline_win: bool
    scoreboard_champion: int | None


def resolve_best_design(ctx: RunContext, jb: Backends) -> BestDesign | None:
    """Resolve the champion (default off); return None after --evaluate-only wrote retained_workflow.json."""
    args = ctx.args
    model = ctx.model
    out_dir = ctx.out_dir
    versions_dir = ctx.versions_dir
    records_dir = ctx.records_dir
    history_rows = ctx.history_rows
    best_design_text: str | None = None
    best_design_version: int | None = None
    best_design_rationale: str | None = None
    best_baseline_case: str | None = None
    champ_has_baseline_win = True
    scoreboard_champion: int | None = None
    if args.include_best_design and args.current_version >= 1:
        champion_path = records_dir / "champion_history.jsonl"
        champion_rows = _load_history(champion_path)
        if args.judge_vs_baseline:
            champ, champ_has_baseline_win, needs_arbitration = (
                _derive_champion_baseline(
                    history_rows, champion_rows, args.current_version
                )
            )
        else:
            champ, needs_arbitration = _derive_champion(
                history_rows, champion_rows, args.current_version
            )
        if needs_arbitration and not args.dry_run:
            champ = arbitrate_champion(ctx, jb, champ, champion_path, champion_rows)
        scoreboard_champion = (
            champ
            if (not args.judge_vs_baseline or champ_has_baseline_win)
            else None
        )
        if args.evaluate_only:
            result = {"version": scoreboard_champion,
                      "beats_reference": scoreboard_champion is not None,
                      "model": model, "last_evaluated_version": args.current_version}
            (out_dir / "retained_workflow.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result))
            return None
        # Best-section renders only for a PROVEN champion: always true in
        # adjacent mode (the chain crowned it); in baseline mode it requires
        # an actual baseline win (a by-default v0 champion is suppressed —
        # no verdict supports a "previous best" claim there).
        if champ != args.current_version and (
            not args.judge_vs_baseline or champ_has_baseline_win
        ):
            best_design_version = champ
            champ_design_path = versions_dir / f"multi_agent_design_v{champ}.txt"
            if not champ_design_path.is_file():
                raise SystemExit(
                    f"Champion design file missing: {champ_design_path}"
                )
            best_design_text = champ_design_path.read_text(encoding="utf-8").strip()
            # Rationale from the verdict where the CURRENT design was directly
            # beaten: the champion-arbitration row if one exists for this pair,
            # else (adjacent mode) the adjacent row it just lost, or (baseline
            # mode) the current version's baseline row (why the baseline won).
            rationale_row: dict[str, Any] | None = None
            arb_lookup_key = f"champion_v{champ}_vs_v{args.current_version}"
            for row in reversed(champion_rows):
                if row.get("key") == arb_lookup_key:
                    rationale_row = row
                    break
            if args.judge_vs_baseline:
                best_baseline_case = (
                    "both_win_arbitration"
                    if rationale_row is not None
                    else "champion_only_win"
                )
            if rationale_row is None:
                if args.judge_vs_baseline:
                    nh_lookup_key = f"v{args.current_version}_vs_no_harness"
                    for row in reversed(history_rows):
                        if row.get("key") == nh_lookup_key:
                            rationale_row = row
                            break
                else:
                    adj_lookup_key = _history_key(
                        args.current_version - 1, args.current_version
                    )
                    for row in reversed(history_rows):
                        if row.get("key") == adj_lookup_key:
                            rationale_row = row
                            break
            if rationale_row is not None and rationale_row.get("rationale"):
                best_design_rationale = str(rationale_row["rationale"])
    return BestDesign(
        best_design_text=best_design_text,
        best_design_version=best_design_version,
        best_design_rationale=best_design_rationale,
        best_baseline_case=best_baseline_case,
        champ_has_baseline_win=champ_has_baseline_win,
        scoreboard_champion=scoreboard_champion,
    )
