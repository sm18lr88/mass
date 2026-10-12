# Build the reviewed-source ZIP with fixed timestamps once the source check passes.
"""Create a ZIP from the reviewed source manifest, excluding all run outputs."""
import argparse
from pathlib import Path
import zipfile

from check_source import ROOT, check, load_manifest

FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def build_archive(output: Path, root: Path = ROOT) -> list[str]:
    """Write the manifest files under mass/ into a new ZIP at output and return the stored names."""
    errors, _ = check(root)
    if errors:
        raise ValueError("Source check failed:\n" + "\n".join(errors))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output}")
    names = sorted(load_manifest(root)) + ["SOURCE_MANIFEST.json"]
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            item = zipfile.ZipInfo(f"mass/{name}", FIXED_TIME)
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, (root / name).read_bytes())
    return names


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        names = build_archive(a.output)
    except (ValueError, FileExistsError) as err:
        raise SystemExit(str(err)) from err
    print(f"Wrote {a.output}: {a.output.stat().st_size} bytes, {len(names)} files")


if __name__ == "__main__":
    main()
