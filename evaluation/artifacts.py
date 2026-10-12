# Size/token-capped reads of high-signal workspace files and logs for judge bundles.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Re-exported for compatibility: these lived in this module before the split.
from evaluation.deliverables import DeliverableScan, extract_png_paths_from_query, format_scan_for_prompt, scan_deliverables
from evaluation.file_tree import SKIP_DIR_NAMES, SKIP_SUFFIXES, build_file_tree
from evaluation.token_budget import _token_counter_for_model, _truncate_text_to_token_budget, estimate_tokens

__all__ = [
    "SKIP_DIR_NAMES",
    "SKIP_SUFFIXES",
    "DeliverableScan",
    "FileRead",
    "build_file_tree",
    "bundle_supporting_reads",
    "discover_entry_scripts",
    "extract_png_paths_from_query",
    "format_scan_for_prompt",
    "read_log_excerpt",
    "read_text_file",
    "scan_deliverables",
    "_token_counter_for_model",
    "_truncate_text_to_token_budget",
]


@dataclass
class FileRead:
    """One file excerpt for the judge bundle."""

    path: str  # relative to memory root
    truncated: bool
    content: str


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def read_text_file(
    memory_root: Path,
    rel: str,
    *,
    max_chars: int | None,
    max_tokens: int | None = None,
    token_count: Callable[[str], int] | None = None,
) -> FileRead | None:
    """Read ``memory_root / rel`` capped by tokens then chars; ``None`` if missing or unreadable."""
    path = memory_root / rel
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Unreadable evidence is treated like absent evidence (the tree still lists it).
        return None
    truncated = False
    if max_tokens is not None:
        token_fn = token_count if token_count is not None else estimate_tokens
        text, truncated = _truncate_text_to_token_budget(text, max_tokens=max_tokens, token_count=token_fn)
    if max_chars is not None and len(text) > max_chars:
        truncated = True
        half = max_chars // 2
        text = text[:half] + "\n\n[... middle omitted ...]\n\n" + text[-half:]
    return FileRead(path=rel.replace("\\", "/"), truncated=truncated, content=text)


def read_log_excerpt(
    log_path: Path,
    *,
    head_chars: int,
    tail_chars: int,
    full: bool = False,
) -> str | None:
    """Head/tail excerpt of a workflow log; ``None`` if missing or unreadable."""
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if full:
        return text
    if len(text) <= head_chars + tail_chars + 200:
        return text
    return text[:head_chars] + "\n\n[... log middle omitted ...]\n\n" + text[-tail_chars:]


def discover_entry_scripts(memory_root: Path) -> list[str]:
    """Likely orchestration scripts at repo root of the memory folder."""
    names: list[str] = []
    for pattern in ("run_*.py", "main.py", "pipeline.py"):
        for p in memory_root.glob(pattern):
            if p.is_file():
                names.append(_rel(p, memory_root))
    return sorted(set(names))


def bundle_supporting_reads(
    memory_root: Path,
    *,
    report_max_chars: int | None = 28_000,
    json_max_chars: int | None = 24_000,
    code_max_chars: int | None = 12_000,
    report_max_tokens: int | None = None,
    json_max_tokens: int | None = None,
    code_max_tokens: int | None = None,
    token_count_model: str | None = None,
    max_code_files: int = 4,
) -> list[FileRead]:
    """Load high-signal text artifacts (size-capped)."""
    out: list[FileRead] = []
    token_count, _token_mode = _token_counter_for_model(token_count_model)
    priority_rel = [
        "research_report.md",
        "deliverables/index.json",
        "results/metrics.json",
        "results/model_comparison.json",
        "requirements.txt",
    ]
    for rel in priority_rel:
        is_json = rel.endswith(".json")
        fr = read_text_file(
            memory_root,
            rel,
            max_chars=json_max_chars if is_json else report_max_chars,
            max_tokens=json_max_tokens if is_json else report_max_tokens,
            token_count=token_count,
        )
        if fr:
            out.append(fr)

    scripts = discover_entry_scripts(memory_root)
    for rel in scripts[:max_code_files]:
        fr = read_text_file(memory_root, rel, max_chars=code_max_chars, max_tokens=code_max_tokens, token_count=token_count)
        if fr:
            out.append(fr)

    # One pytest / test file if present
    tests_dir = memory_root / "tests"
    if tests_dir.is_dir():
        for p in sorted(tests_dir.glob("test_*.py")):
            test_cap = None if code_max_chars is None else min(8000, code_max_chars)
            test_token_cap = None if code_max_tokens is None else min(8000, code_max_tokens)
            fr = read_text_file(memory_root, _rel(p, memory_root), max_chars=test_cap, max_tokens=test_token_cap, token_count=token_count)
            if fr:
                out.append(fr)
            break

    return out
