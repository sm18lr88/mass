# Heuristic deliverable checklist: expected plot paths from task text vs files in the workspace.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeliverableScan:
    """Heuristic checklist derived from the task text + filesystem."""

    png_mentions_in_query: list[str] = field(default_factory=list)
    png_found: list[str] = field(default_factory=list)
    png_missing: list[str] = field(default_factory=list)
    core_paths: dict[str, bool] = field(default_factory=dict)


def extract_png_paths_from_query(query_text: str) -> list[str]:
    """Pull likely plot paths from the Deliverables section (best-effort)."""
    # deliverables/plots/foo.png or results/... not png usually; focus on .png
    paths = set(m.group(1) for m in re.finditer(r"([\w./\\-]+deliverables/plots/[\w.-]+\.png)", query_text, re.IGNORECASE))
    # Also bare filenames under plots/
    for m in re.finditer(r"(?:deliverables/plots/|\b)([a-z0-9_-]+\.png)\b", query_text, re.IGNORECASE):
        name = m.group(1)
        paths.add(f"deliverables/plots/{name}")
    return sorted(paths)


def scan_deliverables(memory_root: Path, query_text: str) -> DeliverableScan:
    """Check plot paths mentioned in ``query_text`` and core report/metrics files under ``memory_root``."""
    mentions = extract_png_paths_from_query(query_text)
    found: list[str] = []
    missing: list[str] = []
    for rel in mentions:
        p = memory_root / rel
        if p.is_file():
            found.append(rel)
        else:
            missing.append(rel)
    core = {
        "research_report.md": (memory_root / "research_report.md").is_file(),
        "deliverables/index.json": (memory_root / "deliverables" / "index.json").is_file(),
        "results/metrics.json": (memory_root / "results" / "metrics.json").is_file(),
        "requirements.txt": (memory_root / "requirements.txt").is_file(),
    }
    return DeliverableScan(
        png_mentions_in_query=mentions,
        png_found=found,
        png_missing=missing,
        core_paths=core,
    )


def format_scan_for_prompt(scan: DeliverableScan) -> str:
    """Render the preflight scan as the judge-prompt markdown section."""
    lines = [
        "## Programmatic preflight (heuristic evidence only)",
        "Derived from the **task text** (expected paths) vs this workspace; not a substitute for reading the Deliverables block.",
        "",
        "### Core files",
    ]
    for k, ok in sorted(scan.core_paths.items()):
        lines.append(f"- `{k}`: {'FOUND' if ok else 'MISSING'}")
    lines.append("")
    lines.append("### Plot paths extracted from task text (substring match)")
    if not scan.png_mentions_in_query:
        lines.append("- (none detected — task may use different wording)")
    else:
        lines.append(f"- mentioned: {len(scan.png_mentions_in_query)}")
        for p in scan.png_mentions_in_query[:40]:
            status = "ok" if p in scan.png_found else "missing"
            lines.append(f"  - [{status}] `{p}`")
        if len(scan.png_mentions_in_query) > 40:
            lines.append("  - ...")
    return "\n".join(lines)
