# Pair two rollout manifests by task and replicate and write the joined workspace list without scoring.
"""Pair two generation rollout manifests on task and replicate, without judging."""
import argparse
import json
from pathlib import Path
from typing import Any


def keyed(path: Path) -> dict[tuple[Any, Any], Any]:
    """Map (task, replicate) to workspace for one rollout manifest, rejecting duplicates."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    try:
        mapped = {(r["task"], r["replicate"]): r["workspace"] for r in rows}
    except (KeyError, TypeError) as err:
        raise ValueError(f"Rollout rows need task, replicate, and workspace fields: {path}") from err
    if len(rows) != len(mapped):
        raise ValueError(f"Duplicate task/replicate in {path}")
    return mapped


def pair(a_path: Path, b_path: Path, out: Path) -> list[dict[str, Any]]:
    """Write the sorted task/replicate pairing of two manifests to out and return the rows."""
    a, b = keyed(a_path), keyed(b_path)
    if not a or set(a) != set(b):
        raise ValueError("Rollout manifests must cover the same nonempty task/replicate set")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing pairing: {out}")
    rows = [{"task": task, "a": a[(task, rep)], "b": b[(task, rep)]} for task, rep in sorted(a)]
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    pair(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
