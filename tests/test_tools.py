# Offline tests for tools/: source manifest checks, archive building, and rollout pairing.
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name: str) -> ModuleType:
    """Load tools/<name>.py by path; tools/ is on sys.path so sibling imports resolve."""
    spec = importlib.util.spec_from_file_location(f"tools_{name}", TOOLS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
    return module


check_source = load_tool("check_source")
make_source_archive = load_tool("make_source_archive")
pair_rollouts = load_tool("pair_rollouts")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TempRepoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def write_repo(self, files: dict[str, bytes], extra: dict[str, str] | None = None) -> Path:
        """Write files and a manifest hashing them; extra manifest entries are not written to disk."""
        root = self.tmp / "repo"
        manifest = {}
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest[name] = sha(data)
        manifest.update(extra or {})
        root.mkdir(parents=True, exist_ok=True)
        (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root


class CheckSourceTests(TempRepoTestCase):
    def test_clean_repo_has_no_problems(self) -> None:
        root = self.write_repo({"a.txt": b"plain text\n"})
        self.assertEqual(check_source.check(root), ([], 1))

    def test_changed_hash_is_reported(self) -> None:
        root = self.write_repo({"a.txt": b"reviewed\n"})
        (root / "a.txt").write_bytes(b"edited\n")
        self.assertEqual(check_source.check(root)[0], ["Source changed after review: a.txt"])

    def test_missing_file_is_reported(self) -> None:
        root = self.write_repo({"b.txt": b"gone soon\n"})
        (root / "b.txt").unlink()
        self.assertEqual(check_source.check(root)[0], ["Missing file: b.txt"])

    def test_unsafe_paths_are_reported(self) -> None:
        digest = sha(b"x")
        root = self.write_repo({}, extra={"../escape.txt": digest, "/abs/file.txt": digest, "C:/drive.txt": digest})
        self.assertCountEqual(
            check_source.check(root)[0],
            ["Unsafe manifest path: ../escape.txt", "Unsafe manifest path: /abs/file.txt", "Unsafe manifest path: C:/drive.txt"],
        )

    def test_symlink_is_reported(self) -> None:
        root = self.write_repo({"real.txt": b"target\n"})
        try:
            os.symlink(root / "real.txt", root / "link.txt")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not permitted on this platform")
        manifest_path = root / "SOURCE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["link.txt"] = sha(b"target\n")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(check_source.check(root)[0], ["Symlink: link.txt"])

    def test_personal_path_is_reported(self) -> None:
        root = self.write_repo({"f.txt": b"see /" + b"Users/alice/project\n"})
        self.assertEqual(check_source.check(root)[0], ["personal absolute path: f.txt"])

    def test_email_is_reported(self) -> None:
        root = self.write_repo({"f.txt": b"contact alice" + b"@" + b"example.com\n"})
        self.assertEqual(check_source.check(root)[0], ["email address: f.txt"])

    def test_api_key_is_reported(self) -> None:
        root = self.write_repo({"f.txt": b"token sk-" + b"a" * 30 + b"\n"})
        self.assertEqual(check_source.check(root)[0], ["possible API credential: f.txt"])

    def test_deny_term_is_matched_case_insensitively(self) -> None:
        root = self.write_repo({"f.txt": b"The PROJECT CODENAME is here\n"})
        self.assertEqual(check_source.check(root, deny_terms=["project codename"])[0], ["Forbidden identity term: f.txt"])

    def test_malformed_manifest_raises_clear_error(self) -> None:
        root = self.write_repo({"a.txt": b"x"})
        (root / "SOURCE_MANIFEST.json").write_text("[1, 2]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must map file paths to SHA-256 strings"):
            check_source.check(root)


class MakeSourceArchiveTests(TempRepoTestCase):
    def test_archive_stores_files_under_mass_with_fixed_timestamps(self) -> None:
        root = self.write_repo({"src/a.py": b"print(1)\n"})
        output = self.tmp / "out.zip"
        names = make_source_archive.build_archive(output, root)
        self.assertEqual(names, ["src/a.py", "SOURCE_MANIFEST.json"])
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.namelist(), ["mass/src/a.py", "mass/SOURCE_MANIFEST.json"])
            for info in archive.infolist():
                self.assertEqual(info.date_time, (2026, 1, 1, 0, 0, 0))
            self.assertEqual(archive.read("mass/src/a.py"), b"print(1)\n")

    def test_existing_output_is_refused_and_untouched(self) -> None:
        root = self.write_repo({"a.txt": b"x"})
        output = self.tmp / "out.zip"
        output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            make_source_archive.build_archive(output, root)
        self.assertEqual(output.read_bytes(), b"keep")

    def test_failed_source_check_writes_no_archive(self) -> None:
        root = self.write_repo({"a.txt": b"reviewed"})
        (root / "a.txt").write_bytes(b"edited")
        output = self.tmp / "out.zip"
        with self.assertRaisesRegex(ValueError, "Source changed after review: a.txt"):
            make_source_archive.build_archive(output, root)
        self.assertFalse(output.exists())


class PairRolloutsTests(TempRepoTestCase):
    def write_rows(self, name: str, rows: list[dict[str, object]]) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_pairs_sorted_by_task_and_replicate(self) -> None:
        a = self.write_rows("a.json", [{"task": "t2", "replicate": 0, "workspace": "wa2"}, {"task": "t1", "replicate": 0, "workspace": "wa1"}])
        b = self.write_rows("b.json", [{"task": "t1", "replicate": 0, "workspace": "wb1"}, {"task": "t2", "replicate": 0, "workspace": "wb2"}])
        out = self.tmp / "pairs.json"
        pair_rollouts.pair(a, b, out)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8")), [
            {"task": "t1", "a": "wa1", "b": "wb1"},
            {"task": "t2", "a": "wa2", "b": "wb2"},
        ])

    def test_mismatched_sets_are_rejected(self) -> None:
        a = self.write_rows("a.json", [{"task": "t1", "replicate": 0, "workspace": "wa"}])
        b = self.write_rows("b.json", [{"task": "t9", "replicate": 0, "workspace": "wb"}])
        with self.assertRaisesRegex(ValueError, "same nonempty task/replicate set"):
            pair_rollouts.pair(a, b, self.tmp / "pairs.json")

    def test_duplicate_rows_are_rejected(self) -> None:
        row = {"task": "t1", "replicate": 0, "workspace": "w"}
        a = self.write_rows("a.json", [row, row])
        b = self.write_rows("b.json", [row])
        with self.assertRaisesRegex(ValueError, "Duplicate task/replicate"):
            pair_rollouts.pair(a, b, self.tmp / "pairs.json")

    def test_missing_field_raises_clear_error(self) -> None:
        a = self.write_rows("a.json", [{"task": "t1", "replicate": 0}])
        b = self.write_rows("b.json", [{"task": "t1", "replicate": 0, "workspace": "w"}])
        with self.assertRaisesRegex(ValueError, "need task, replicate, and workspace fields"):
            pair_rollouts.pair(a, b, self.tmp / "pairs.json")

    def test_existing_output_is_refused(self) -> None:
        row = {"task": "t1", "replicate": 0, "workspace": "w"}
        a = self.write_rows("a.json", [row])
        b = self.write_rows("b.json", [row])
        out = self.tmp / "pairs.json"
        out.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            pair_rollouts.pair(a, b, out)
        self.assertEqual(out.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
