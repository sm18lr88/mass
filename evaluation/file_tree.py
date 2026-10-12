# Bounded, representative directory-tree rendering of a submission workspace.
from __future__ import annotations

from pathlib import Path

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
}

# Large or binary artifacts we only mention in the tree, not inline.
SKIP_SUFFIXES = (
    ".parquet",
    ".pq",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".h5",
    ".hdf5",
    ".feather",
    ".arrow",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
)


def build_file_tree(
    root: Path,
    *,
    max_files: int = 400,
    max_depth: int = 6,
    max_children_per_dir: int = 100,
) -> str:
    """Return a bounded, representative workspace tree.

    ``max_files`` is retained as the public argument name for compatibility,
    but it now limits all displayed filesystem entries, including
    directories.  Previously, directory-only dataset trees could grow
    without bound because only files consumed the budget.

    Directories with more than ``max_children_per_dir`` immediate children
    are represented by one explicit omission marker instead of enumerating
    raw dataset/cache contents.  The directory itself remains visible as
    provenance evidence.
    """
    lines: list[str] = []
    remaining_entries = max(0, max_files)
    tree_truncated = False

    def walk(cur: Path, depth: int) -> None:
        nonlocal remaining_entries, tree_truncated
        if remaining_entries <= 0:
            tree_truncated = True
            return
        if depth > max_depth:
            return
        if not cur.is_dir():
            return
        if cur.name in SKIP_DIR_NAMES:
            return
        try:
            subs = sorted(
                (path for path in cur.iterdir() if not (path.is_dir() and path.name in SKIP_DIR_NAMES)),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            # Unreadable directory: show it (already listed by the parent) without children.
            return
        if max_children_per_dir > 0 and len(subs) > max_children_per_dir:
            lines.append(f"{'  ' * depth}... [contents omitted: {len(subs)} entries exceed per-directory limit {max_children_per_dir}]")
            return
        for p in subs:
            if remaining_entries <= 0:
                tree_truncated = True
                return
            if p.is_dir():
                lines.append(f"{'  ' * depth}{p.name}/")
                remaining_entries -= 1
                walk(p, depth + 1)
            else:
                suf = p.suffix.lower()
                if suf in SKIP_SUFFIXES:
                    lines.append(f"{'  ' * depth}{p.name}  [{suf} binary/large]")
                else:
                    lines.append(f"{'  ' * depth}{p.name}")
                remaining_entries -= 1

    lines.append(f"{root.name}/")
    walk(root, 1)
    if tree_truncated:
        lines.append("... (tree truncated)")
    return "\n".join(lines)
