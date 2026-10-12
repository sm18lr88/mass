"""UTF-8 JSON/JSONL file helpers and repo-relative path resolution; no model or accelerator imports."""

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | os.PathLike, value: Any) -> None:
    """Atomically replace `path`: write a sibling temp file, fsync it, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def records(path: str | os.PathLike) -> list[Any]:
    """Read a JSONL file; a missing file is empty, an undecodable line is an error."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as err:
            # Typically a record truncated by a crash mid-append; never skip it silently.
            raise ValueError(f"{path}: line {number} is not valid JSON ({err.msg}); repair or archive the file before resuming") from err
    return rows


def append_record(path: str | os.PathLike, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def package_path(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
