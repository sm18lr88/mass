# Immutable artifact writes, history JSONL IO, prompt dumps, and the context-budget guard for RHI.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_full_prompt_dump(path: Path, *, system_prompt: str, user_prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "=== SYSTEM PROMPT ===\n\n"
        f"{system_prompt.strip()}\n\n"
        "=== USER PROMPT ===\n\n"
        f"{user_prompt.strip()}\n"
    )
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == body:
            return
        raise RuntimeError(
            "Refusing to overwrite an existing prompt with different content: "
            f"{path}"
        )
    path.write_text(body, encoding="utf-8")


def _enforce_context_budget(
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int | None,
    context_window_tokens: int | None,
    context_safety_tokens: int,
) -> tuple[int, int]:
    """Apply the same conservative context guard to every local model call."""
    estimated_input_tokens = max(
        1, (len(system_prompt) + len(user_prompt) + 3) // 4
    )
    estimated_total_with_output = estimated_input_tokens + int(
        max_output_tokens or 0
    )
    if context_window_tokens is not None:
        allowed = context_window_tokens - context_safety_tokens
        if estimated_total_with_output > allowed:
            raise SystemExit(
                f"Conservative context guard failed for {stage}: "
                f"estimated_input_tokens={estimated_input_tokens}, "
                f"max_output_tokens={max_output_tokens or 0}, "
                f"total={estimated_total_with_output}, allowed={allowed}. "
                "Reduce evidence caps or output tokens."
            )
    return estimated_input_tokens, estimated_total_with_output


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Write an immutable JSON artifact, accepting an identical existing value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Refusing to overwrite invalid existing JSON: {path}: {exc}"
            ) from exc
        if existing == payload:
            return
        raise RuntimeError(
            f"Refusing to overwrite an existing JSON artifact: {path}"
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_history(history_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not history_path.is_file():
        return rows
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_history_row(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
