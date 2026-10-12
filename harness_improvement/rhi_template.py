# Query template parsing (design-block markers) and per-version repo resolution for RHI.
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_START_MARKER = "Create an agent team with following agent candidates to solve this problem:"
DEFAULT_END_MARKER = "Use uv for all Python workflows"


@dataclass
class QueryTemplate:
    query_num: int
    query_name: str
    raw_text: str
    before_design: str
    design_block: str
    after_design: str
    task_text_for_eval: str


def _read_query_template(
    query_file: Path,
    *,
    start_marker: str,
    end_marker: str,
) -> QueryTemplate:
    raw = query_file.read_text(encoding="utf-8")
    m_query = re.search(r"query(\d+)_ourTeam", query_file.stem)
    if not m_query:
        raise ValueError(f"Cannot parse query number from filename: {query_file.name}")
    query_num = int(m_query.group(1))
    query_name = f"query{query_num}_ourTeam"

    start_idx = raw.find(start_marker)
    if start_idx < 0:
        raise ValueError(f"Start marker not found in {query_file}: {start_marker!r}")
    # Use the last end marker in the file because the phrase can appear
    # inside the design block as a normal instruction line.
    end_idx = raw.rfind(end_marker)
    if end_idx < 0:
        raise ValueError(f"End marker not found in {query_file}: {end_marker!r}")
    if end_idx <= start_idx:
        raise ValueError(
            f"End marker occurs before start marker in {query_file}: {end_marker!r}"
        )

    before = raw[:start_idx]
    design = raw[start_idx:end_idx].strip()
    after = raw[end_idx:]

    # Some query files may store task text with escaped newlines.
    task_text = before.rstrip()
    if "\\n" in task_text:
        task_text = task_text.replace("\\n", "\n")
    task_text = task_text.strip()

    return QueryTemplate(
        query_num=query_num,
        query_name=query_name,
        raw_text=raw,
        before_design=before,
        design_block=design,
        after_design=after,
        task_text_for_eval=task_text,
    )


def _resolve_iteration_repo(runs_root: Path, query_name: str, version: int) -> Path:
    if version < 0:
        raise ValueError("version must be >= 0")
    if version == 0:
        return runs_root / query_name
    versioned = runs_root / f"{query_name}-v{version}"
    if versioned.is_dir():
        return versioned
    # Backward-compatible fallback: some runs roots store improved versions
    # in plain query folders without the -v{n} suffix.
    plain = runs_root / query_name
    if plain.is_dir():
        return plain
    return versioned


def _infer_versioned_runs_root(base_runs_root: Path, version: int) -> Path | None:
    """Infer sibling runs root from a trailing _v{n} naming scheme.

    Supports both conventions:
    - v0 as explicit suffix:   name_v0 -> name_v1 -> name_v2
    - v0 as base name:         name -> name_v1 -> name_v2
      (for version == 0, prefer stripping the _v{n} suffix).
    """
    m = re.search(r"_v\d+$", base_runs_root.name)
    if not m:
        return None

    # If caller asks for v0, first try the unsuffixed base name.
    if version == 0:
        unsuffixed_name = re.sub(r"_v\d+$", "", base_runs_root.name)
        unsuffixed = base_runs_root.with_name(unsuffixed_name)
        if unsuffixed.is_dir():
            return unsuffixed

    inferred_name = re.sub(r"_v\d+$", f"_v{version}", base_runs_root.name)
    inferred = base_runs_root.with_name(inferred_name)
    if inferred.is_dir():
        return inferred
    return None
