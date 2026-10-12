# Builds the optimizer (system, user) prompt that asks for the next multi-agent design block.
from __future__ import annotations

from typing import Any

from harness_improvement.rhi_history import (
    _history_delta_checklist_text,
    _history_summary_text,
    _shorten,
)
from harness_improvement.rhi_template import DEFAULT_START_MARKER, QueryTemplate


def _build_single_submission_improvement_prompt(
    *,
    template: QueryTemplate,
    current_version: int,
    current_design: str,
    workspace_evidence: str,
    history_rows: list[dict[str, Any]],
    submission_label: str,
    execution_metadata: str | None,
    output_mode: str,
    max_design_chars: int | None,
    target_design_chars: int | None,
    forbid_fixed_agent_limits: bool = False,
    compact_era_prompt: bool = False,
    best_design: str | None = None,
    best_version: int | None = None,
    best_rationale: str | None = None,
    design_diffs: dict[str, str] | None = None,
    v0_design: str | None = None,
    baseline_mode: bool = False,
    baseline_scoreboard: str | None = None,
    best_baseline_case: str | None = None,
) -> tuple[str, str]:
    if output_mode == "json":
        response_rule = "Return valid JSON only."
        output_contract = (
            "Output schema:\n"
            "{{\n"
            '  "improved_multi_agent_design": "<full replacement text for the design block only>",\n'
            '  "changes_from_previous": ["<specific change 1>", "..."],\n'
            '  "why_it_should_improve": ["<mechanistic rationale 1>", "..."],\n'
            '  "evidence_used": ["<file/metric/history citation 1>", "..."],\n'
            '  "expected_impact": ["<expected gain in quality/reliability/reproducibility 1>", "..."],\n'
            '  "verification_checks": ["<how to validate this change worked in next iteration>", "..."]\n'
            "}}\n"
        )
    elif output_mode == "design-text":
        response_rule = (
            "Return only the full replacement multi-agent design block as plain text. "
            "Do not wrap it in JSON, Markdown fences, or explanatory prose."
        )
        output_contract = (
            "Output contract: return only the full replacement design block as plain "
            "text. Its first characters must be the required design marker. Do not "
            "include JSON metadata, Markdown fences, the task text, or the post-design "
            "prompt suffix.\n"
        )
    else:
        raise ValueError(f"Unsupported improvement output mode: {output_mode}")

    if target_design_chars is not None:
        if max_design_chars is None:
            raise ValueError(
                "target_design_chars requires a finite max_design_chars value"
            )
        if target_design_chars > max_design_chars:
            raise ValueError(
                "target_design_chars cannot exceed max_design_chars"
            )
        compact_system_rule = (
            "target improved_multi_agent_design at no more than "
            f"{target_design_chars:,} characters and never exceed "
            f"{max_design_chars:,} characters, and keep each explanatory "
        )
        compact_user_rule = (
            f"Target at most {target_design_chars:,} characters and never exceed "
            f"{max_design_chars:,} characters. Prefer shared "
        )
    elif max_design_chars is not None:
        compact_system_rule = (
            "keep improved_multi_agent_design under "
            f"{max_design_chars:,} characters, and keep each explanatory "
        )
        compact_user_rule = (
            f"Keep the replacement design under {max_design_chars:,} characters. "
            "Prefer shared "
        )
    else:
        # No explicit size flags: no compactness directive at all (claude-era
        # behavior; the qwen35-era generic "keep it compact" wording is only
        # emitted when a size flag is set).
        compact_system_rule = None
        compact_user_rule = None

    # --compact-era-prompt: byte-exact restoration of the compact-lineage
    # optimizer prompt (the generic compact wording that predated the
    # size-flag variants). Used to CONTINUE the archived compact lineage;
    # never set for the free lineage.
    if compact_era_prompt and compact_system_rule is None:
        compact_system_rule = (
            "keep improved_multi_agent_design compact, and keep each explanatory "
        )
        compact_user_rule = "Keep the replacement design compact. Prefer shared "

    if compact_system_rule is not None:
        size_rule_system = (
            "Keep the replacement focused and compact: preserve useful existing instructions, "
            "make evidence-targeted edits, avoid repeating the same protocol in every agent, "
            f"{compact_system_rule}"
            "JSON list to at most 8 concise items. This artifact-size constraint is not an "
            "execution or agent-count constraint.\n"
        )
        size_rule_user = (
            f"{compact_user_rule}"
            "protocols plus compact agent-specific deltas over duplicated schemas.\n"
        )
    else:
        size_rule_system = ""
        size_rule_user = ""

    # The no-numeric-caps directive is tied to the flag that also validates
    # the output for such caps; without the flag the prompt matches the
    # claude-era wording. --compact-era-prompt emits the SENTENCES without
    # the validator (the compact lineage had the sentences unconditionally
    # but never ran the validator).
    if forbid_fixed_agent_limits or compact_era_prompt:
        caps_rule_system = (
            "Do not introduce arbitrary numerical caps on agent or subagent invocations, "
            "recall/re-delegation loops, orchestration hops, tool calls, execution turns, "
            "or wall-clock duration. Terminate through acceptance gates or a genuinely "
            "unrecoverable documented blocker.\n"
        )
        caps_rule_user = (
            "Do not impose a fixed maximum number of recalls, agent/subagent calls, "
            "hops, turns, tools, or duration. Use acceptance-based convergence.\n"
        )
    else:
        caps_rule_system = ""
        caps_rule_user = ""

    system = (
        "You are a principal prompt engineer for autonomous coding agents.\n"
        f"The following query was executed by {submission_label}.\n"
        "Your task is to improve only the 'multi agent design' block of the query prompt to improve the quality of the query's deliverables (see 'Current multi agent design' below).\n"
        "Prioritize harnesses that multi-agent systems can do well (or have) and single-agent systems cannot (or does not have), "
        "such as parallel specialist decomposition, cross-agent critique, agent to agent communication, and iterative reconciliation.\n"
        "Prioritize two improvements: (1) stronger agent-to-agent communication contracts "
        "for each subagent 'Output to orchestrator' schema, and "
        "(2) more orchestrator-subagent hops with explicit iterative recall/re-delegation "
        "instead of one-pass unidirectional execution.\n"
        f"{caps_rule_system}"
        f"{size_rule_system}"
        "You must ground changes in concrete evidence from submission artifacts and comparison history.\n"
        f"{response_rule}"
    )
    save_rules= """
    Use uv for all Python workflows—run code with uv run, install dependencies with uv add, use uvx for tools. write your history log: write your plan, execution (or tool-execution), reflection during the reasoning logs in a logs.txt file. In logs.txt, Document all created and spawned agents, describe the workflow between agents (e.g., as a structured outline or diagram), track which agents were executed and their roles in the process. 
    """
    # BEST-H anchoring (default off): when the in-loop judge chain says an
    # OLDER design is still the best, show it next to the current design so
    # the optimizer can ratchet on the champion instead of a regressed base.
    best_sections: list[str] = []
    if best_design is not None and best_version is not None:
        rationale_text = ""
        if best_rationale:
            rationale_text = _shorten(best_rationale.strip(), max_chars=2000)
            if rationale_text and rationale_text[-1] not in ".!?\"'":
                rationale_text += "."
        if baseline_mode and best_baseline_case == "champion_only_win":
            rationale_sentence = (
                (
                    " The judge's stated reasons for the baseline's win over "
                    f"v{current_version}: {rationale_text}"
                )
                if rationale_text
                else ""
            )
            note_text = (
                f"NOTE: this earlier design v{best_version} is shown because it "
                f"BEAT the no-harness baseline while the current design "
                f"v{current_version} did not (see the record in section 3)."
                f"{rationale_sentence} Refer to BOTH v{best_version} and "
                f"v{current_version} when writing v{current_version + 1}: keep "
                f"what made v{best_version} strong, and merge in only the "
                f"genuinely useful changes attempted since."
            )
        elif baseline_mode and best_baseline_case == "both_win_arbitration":
            rationale_sentence = (
                f" The judge's stated reasons: {rationale_text}"
                if rationale_text
                else ""
            )
            note_text = (
                f"NOTE: both this earlier design v{best_version} and the current "
                f"design v{current_version} beat the no-harness baseline; in a "
                f"direct comparison the judge found v{best_version}'s "
                f"deliverables better.{rationale_sentence} Refer to BOTH "
                f"v{best_version} and v{current_version} when writing "
                f"v{current_version + 1}: keep what made v{best_version} strong, "
                f"and merge in only the genuinely useful changes attempted since."
            )
        else:
            rationale_sentence = (
                f" The judge's stated reasons for that verdict: {rationale_text}"
                if rationale_text
                else ""
            )
            note_text = (
                f"NOTE: this earlier design v{best_version} is shown because the in-loop "
                f"pairwise judge determined it produced BETTER deliverables than the "
                f"current design v{current_version} (v{best_version} > v{current_version} "
                f"in the comparison history).{rationale_sentence} Refer to BOTH "
                f"v{best_version} and "
                f"v{current_version} when writing v{current_version + 1}: keep what made "
                f"v{best_version} strong, and merge in only the genuinely useful changes "
                f"attempted since."
            )
        best_sections = [
            f"# Previous best multi agent design (v{best_version})",
            note_text,
            best_design,
        ]
        if baseline_mode:
            improve_lead = (
                "Improve the multi-agent design to v{n} by combining the "
                f"strengths of the previous best design v{best_version} with "
                "the current design v{v}, with the explicit objective of "
                "producing deliverables that decisively OUTPERFORM the "
                "no-harness baseline (see section 3), not merely improving "
                "on v{v}.\n"
            )
        else:
            improve_lead = (
                "Improve the multi-agent design to v{n} by combining the strengths of the "
                f"previous best design v{best_version} with the current design v{{v}}, "
                "to enhance the quality of the query's deliverables by creating genuine "
                "multi-agent advantages over single-agent execution.\n"
            )
    elif baseline_mode:
        improve_lead = (
            "Improve the multi-agent design from v{v} to v{n} to enhance the "
            "quality of the query's deliverables by creating genuine "
            "multi-agent advantages over single-agent execution, with the "
            "explicit objective of producing deliverables that decisively "
            "OUTPERFORM the no-harness baseline (see section 3), not merely "
            "improving on v{v}.\n"
        )
    else:
        improve_lead = (
            "Improve the multi-agent design from v{v} to v{n} to enhance the quality of the query's deliverables by creating genuine multi-agent advantages over single-agent execution.\n"
        )

    v0_sections: list[str] = []
    if v0_design is not None:
        v0_sections = [
            "# Original multi agent design (v0 - the initial starting point)",
            (
                "NOTE: this is the original starting design, shown as the ROOT "
                "of the version track whose per-step changes appear in the "
                "pairwise history below. Use it to follow how the design "
                "evolved and to notice drift from the original intent. The "
                "current and best designs above supersede it - do not revert "
                "to it wholesale."
            ),
            v0_design,
        ]

    history_header = "3. Pairwise history summary"
    history_body = _history_summary_text(history_rows, design_diffs=design_diffs)
    if baseline_mode:
        history_header = "3. Pairwise history vs the no-harness baseline"
        framing_lines: list[str] = []
        if baseline_scoreboard:
            framing_lines.append(baseline_scoreboard)
            if "★" in baseline_scoreboard:
                framing_lines.append(
                    "★ = best design so far: among the versions that beat the "
                    "baseline, direct pairwise arbitration judged this one's "
                    "deliverables best."
                )
        framing_lines.append(
            "Each entry below compares that version's outputs against the "
            "FIXED no-harness baseline - the same task executed by a bare "
            "single agent with no multi-agent design. The diff attached to an "
            "entry shows what changed in the design before that verdict. Each "
            "verdict is a single judgment; treat trends across rounds as more "
            "informative than any single win/loss flip."
        )
        history_body = "\n".join(framing_lines) + "\n\n" + history_body

    user = "\n\n".join(
        [
            f"1. Query executed by {submission_label}",
            "--------------------------------",
            "# Task",
            template.task_text_for_eval,
            f"# Current multi agent design (v{current_version})",
            current_design,
            *best_sections,
            *v0_sections,
            "# Save rules",
            save_rules,
            "--------------------------------",
            f"2. Current submission code repo from {submission_label} (evidence only)",
            workspace_evidence,
            "2b. Execution metadata",
            execution_metadata or "- No execution metadata was supplied.",
            history_header,
            history_body,
            "3b. Pairwise history delta checklist (recurring issues to fix)",
            _history_delta_checklist_text(history_rows),
            "4. Instructions",
            (
                improve_lead +
                "Preserve the original intent and required deliverables, but address weaknesses revealed by evidence.\n"
                "Explicitly strengthen 'Output to orchestrator' contracts so downstream agents can consume structured outputs.\n"
                "Explicitly increase orchestrator-subagent feedback loops: allow orchestrator recall of previously called subagents "
                "with narrower follow-up scopes and updated acceptance criteria.\n"
                f"{caps_rule_user}"
                f"{size_rule_user}"
                "Most importantly, prioritize improvements that create genuine multi-agent advantages over single-agent execution "
                "(e.g., specialist parallelism, cross-agent validation, agent to agent communication, and conflict resolution loops, etc).\n"
                "Feedback quality requirements:\n"
                "- Be specific and evidence-grounded. Reference concrete files/metrics/history signals from the provided evidence.\n"
                "- Avoid generic claims (e.g., 'better coordination'); explain mechanism and expected effect.\n"
                "Do not change non-design parts of the query.\n"
                f"{output_contract}"
                "The `improved_multi_agent_design` should start with "
                f"'{DEFAULT_START_MARKER}' and contain the full block."
            ).format(v=current_version, n=current_version + 1),
        ]
    )
    return system, user
