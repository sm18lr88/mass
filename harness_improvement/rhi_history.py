# Pairwise-history rendering for the optimizer prompt: summaries, recurring-issue checklist, design diffs.
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any


def _history_key(version_a: int, version_b: int) -> str:
    return f"v{version_a}_vs_v{version_b}"


def _history_action_items(rationale: str, *, max_items: int = 6) -> list[str]:
    cues = (
        "fail",
        "fails",
        "failed",
        "missing",
        "inconsistent",
        "mismatch",
        "not ",
        "lack",
        "weak",
        "nan",
        "null",
        "error",
        "defect",
        "reproducibility",
    )
    cleaned = " ".join(rationale.split())
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    picks: list[str] = []
    seen = set()
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        low = s.lower()
        if not any(cue in low for cue in cues):
            continue
        if s in seen:
            continue
        seen.add(s)
        picks.append(s)
        if len(picks) >= max_items:
            break
    return picks


def _shorten(text: str, *, max_chars: int) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _design_diffs_for_history(
    history_rows: list[dict[str, Any]],
    versions_dir: Path,
    *,
    recent_full: int = 3,
    recent_cap: int = 1500,
    old_cap: int = 400,
    baseline_mode: bool = False,
) -> dict[str, str]:
    """Compact unified diffs of the design blocks behind each history entry.

    Deterministic and API-free: reads versions/multi_agent_design_v{k}.txt.
    The most recent `recent_full` pairs get `recent_cap` chars each; older
    pairs get `old_cap` — recency carries the credit-assignment signal while
    keeping total prompt mass bounded for the local optimizer. In
    baseline_mode the entries are v{k}_vs_no_harness rows and each carries
    the diff v{k-1} -> v{k} (the v0 row has no diff).
    """
    adjacent: list[tuple[int, str]] = []
    for row in history_rows:
        key = str(row.get("key") or "")
        if baseline_mode:
            m = re.fullmatch(r"v(\d+)_vs_no_harness", key)
            if m and int(m.group(1)) >= 1:
                adjacent.append((int(m.group(1)), key))
            continue
        m = re.fullmatch(r"v(\d+)_vs_v(\d+)", key)
        if m and int(m.group(2)) == int(m.group(1)) + 1:
            adjacent.append((int(m.group(2)), key))
    adjacent = sorted(set(adjacent))
    diffs: dict[str, str] = {}
    for rank, (newer, key) in enumerate(reversed(adjacent)):
        older = newer - 1
        pa = versions_dir / f"multi_agent_design_v{older}.txt"
        pb = versions_dir / f"multi_agent_design_v{newer}.txt"
        if not (pa.is_file() and pb.is_file()):
            continue
        diff_lines = difflib.unified_diff(
            pa.read_text(encoding="utf-8").splitlines(),
            pb.read_text(encoding="utf-8").splitlines(),
            fromfile=f"multi_agent_design_v{older}",
            tofile=f"multi_agent_design_v{newer}",
            lineterm="",
            n=1,
        )
        text = "\n".join(diff_lines).strip()
        if not text:
            text = "(designs are identical)"
        cap = recent_cap if rank < recent_full else old_cap
        diffs[key] = _shorten(text, max_chars=cap)
    return diffs


def _history_summary_text(
    history_rows: list[dict[str, Any]],
    design_diffs: dict[str, str] | None = None,
) -> str:
    if not history_rows:
        return "- No prior pairwise history yet."
    lines: list[str] = []
    for row in history_rows:
        key = str(row.get("key", ""))
        comp = str(row.get("comparison", ""))
        winner = row.get("winner", "unknown")
        rationale = str(row.get("rationale", "")).strip()
        judged_at = str(row.get("judged_at_utc", ""))
        model = str(row.get("model", ""))
        repo_a = str(row.get("repo_a", ""))
        repo_b = str(row.get("repo_b", ""))
        action_items = _history_action_items(rationale)

        lines.append(f"- Comparison: {comp}")
        if key:
            lines.append(f"  - Key: {key}")
        lines.append(f"  - Winner: {winner}")
        if model:
            lines.append(f"  - Judge model: {model}")
        if judged_at:
            lines.append(f"  - Judged at (UTC): {judged_at}")
        if repo_a:
            lines.append(f"  - Repo A: {repo_a}")
        if repo_b:
            lines.append(f"  - Repo B: {repo_b}")
        if rationale:
            lines.append(f"  - Rationale (verbatim excerpt): {_shorten(rationale, max_chars=2400)}")
        if action_items:
            lines.append("  - Actionable takeaways for next version:")
            for item in action_items:
                lines.append(f"    - {item}")
        if design_diffs and key in design_diffs:
            lines.append(
                "  - Design change behind this verdict (unified diff of the "
                "design block; '-' lines were removed, '+' lines added):"
            )
            for diff_line in design_diffs[key].splitlines():
                lines.append(f"    {diff_line}")
        lines.append("  - Use this history as constraints: preserve winning traits, fix listed defects.")
    return "\n".join(lines)


def _history_delta_checklist_text(history_rows: list[dict[str, Any]]) -> str:
    if not history_rows:
        return "- No recurring issues yet."

    # Heuristic issue buckets for recurring weak signals in pairwise rationales.
    patterns: list[tuple[str, str]] = [
        ("report-metrics inconsistency", r"(inconsistent|mismatch|contradict|does not match|conflict)"),
        ("missing deliverables or sections", r"(missing|absent|not found|not present)"),
        ("stability/robustness gap", r"(stability|robustness|subperiod|pre_2020|post_2020)"),
        ("null/nan quality defect", r"(nan|null|none|non-standard json)"),
        ("reproducibility gap", r"(reproducibility|not reproducible|cannot reproduce|requirements|pin)"),
        ("validation/test weakness", r"(validation|test|insufficient evidence|not proven|unclear)"),
    ]

    counts = {name: 0 for name, _ in patterns}
    examples: dict[str, str] = {}
    for row in history_rows:
        rationale = str(row.get("rationale", "")).strip()
        if not rationale:
            continue
        low = rationale.lower()
        for name, pat in patterns:
            if re.search(pat, low):
                counts[name] += 1
                if name not in examples:
                    items = _history_action_items(rationale, max_items=1)
                    if items:
                        examples[name] = items[0]

    ranked = sorted(((name, c) for name, c in counts.items() if c > 0), key=lambda x: x[1], reverse=True)
    if not ranked:
        return "- No repeated issue patterns detected yet."

    lines = [
        "- Prioritize fixes for recurring issues below (higher count = more repeated in history):"
    ]
    for name, c in ranked:
        ex = examples.get(name)
        if ex:
            lines.append(f"- [{c}x] {name}: e.g., {ex}")
        else:
            lines.append(f"- [{c}x] {name}")
    return "\n".join(lines)
