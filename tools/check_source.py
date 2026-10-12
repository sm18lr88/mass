# Verify reviewed source hashes and scan manifest files for personal paths and credentials offline.
"""Check reviewed source integrity and common personal-path/credential leaks offline."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from collections.abc import Iterable

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "personal absolute path": re.compile(r"/(?:home|Users)/[\w.-]+"),
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "possible private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "possible API credential": re.compile(r"\b(?:sk|ghp)[-_][A-Za-z0-9_-]{25,}"),
}
REVIEWED_MEDIA_ASSETS = {
    "assets/mass_logo.png": ((b"\x89PNG\r\n\x1a\n",), 2_000_000),
    "assets/mass_teaser_figure.png": ((b"\x89PNG\r\n\x1a\n",), 2_000_000),
    "assets/mass_teaser_minimal.gif": ((b"GIF87a", b"GIF89a"), 10_000_000),
    "assets/mass_teaser_minimal.mp4": ((b"\x00\x00\x00\x20ftyp",), 4_000_000),
}


def load_manifest(root: Path) -> dict[str, str]:
    """Return the manifest mapping of relative file path to expected SHA-256 hex digest."""
    path = root / "SOURCE_MANIFEST.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ValueError(f"Cannot read source manifest {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Source manifest is not valid JSON: {path}: {err}") from err
    if not isinstance(manifest, dict) or not all(isinstance(value, str) for value in manifest.values()):
        raise ValueError(f"Source manifest must map file paths to SHA-256 strings: {path}")
    return manifest


def is_unsafe(name: str) -> bool:
    """Return True for absolute, drive-qualified, or parent-traversing manifest paths on any platform."""
    windows, posix = PureWindowsPath(name), PurePosixPath(name)
    parts = re.split(r"[\\/]", name)
    return windows.is_absolute() or bool(windows.drive) or posix.is_absolute() or ".." in parts


def has_symlink(root: Path, rel: Path) -> bool:
    """Return True if any component of root/rel below root is a symbolic link."""
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def check(root: Path | str = ROOT, deny_terms: Iterable[str] = ()) -> tuple[list[str], int]:
    """Return (problems, manifest_count) for the source tree rooted at root."""
    root = Path(root)
    manifest = load_manifest(root)
    problems: list[str] = []
    for name, expected in manifest.items():
        if is_unsafe(name):
            problems.append(f"Unsafe manifest path: {name}")
            continue
        rel = Path(name)
        if has_symlink(root, rel):
            problems.append(f"Symlink: {name}")
            continue
        path = root / rel
        if not path.is_file():
            problems.append(f"Missing file: {name}")
            continue
        data = path.read_bytes()
        signatures, size_limit = REVIEWED_MEDIA_ASSETS.get(name, ((), 2_000_000))
        if len(data) > size_limit:
            problems.append(f"Unexpectedly large source file: {name}")
        if hashlib.sha256(data).hexdigest() != expected:
            problems.append(f"Source changed after review: {name}")
        if signatures:
            if not data.startswith(signatures):
                problems.append(f"Invalid media asset: {name}")
            text = ""
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                problems.append(f"Unexpected binary source: {name}")
                continue
        for kind, pattern in PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{kind}: {name}")
        for term in deny_terms:
            if term.casefold() in (name + "\n" + text).casefold():
                problems.append(f"Forbidden identity term: {name}")
    return problems, len(manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deny-term", action="append", default=[])
    a = p.parse_args()
    errors, n = check(deny_terms=a.deny_term)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {n} manifest files; {len(errors)} problems.")
    raise SystemExit(bool(errors))
