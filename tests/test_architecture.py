# Fitness checks for package dependency direction and repository source conventions.
import ast
import re
import sys
import unittest

from mass.io import ROOT

LAYERS = ["evaluation", "evaluation_claudecodex", "harness_improvement", "mass"]
INDEPENDENT = ["runtime", "training", "tools"]
REPO_PACKAGES = set(LAYERS + INDEPENDENT + ["benchmarks", "tests"])
SKIP_DIRS = {"vendor", "work", "__pycache__"}


def python_files(*packages):
    for package in packages:
        for path in sorted((ROOT / package).rglob("*.py")):
            if not SKIP_DIRS & set(path.relative_to(ROOT).parts):
                yield path


def imports(path, top_level_only=False):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body if top_level_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield "." * node.level + (node.module or "")


def repo_package(module):
    top = module.split(".")[0]
    return top if top in REPO_PACKAGES else None


def rel(path):
    return path.relative_to(ROOT).as_posix()


class ArchitectureTests(unittest.TestCase):
    maxDiff = None

    def test_layers_import_only_themselves_and_lower_layers(self):
        violations = []
        for rank, package in enumerate(LAYERS):
            allowed = set(LAYERS[: rank + 1])
            for path in python_files(package):
                violations += [f"{rel(path)} imports {m}" for m in imports(path) if repo_package(m) not in allowed | {None}]
        self.assertEqual(violations, [])

    def test_independent_packages_import_no_other_repository_package(self):
        violations = []
        for package in INDEPENDENT:
            for path in python_files(package):
                violations += [f"{rel(path)} imports {m}" for m in imports(path) if repo_package(m) not in {package, None}]
        self.assertEqual(violations, [])

    def test_benchmarks_stay_outside_the_mass_loop(self):
        # `evaluation` here is ScienceAgentBench's vendored package, not this repository's.
        forbidden = {"mass", "harness_improvement", "training", "runtime", "evaluation_claudecodex"}
        violations = [f"{rel(path)} imports {m}" for path in python_files("benchmarks") for m in imports(path) if m.split(".")[0] in forbidden]
        self.assertEqual(violations, [])

    def test_driver_imports_heavy_dependencies_lazily(self):
        violations = [
            f"{rel(path)} imports {m} at module level"
            for path in python_files("mass")
            for m in imports(path, top_level_only=True)
            if not m.startswith(".") and m.split(".")[0] not in sys.stdlib_module_names
        ]
        self.assertEqual(violations, [])

    def test_every_python_file_starts_with_a_responsibility_comment(self):
        missing = []
        for path in python_files(*REPO_PACKAGES):
            lines = path.read_text(encoding="utf-8").splitlines()
            while lines and (lines[0].startswith("#!") or lines[0].startswith("# -*-")):
                lines.pop(0)
            if lines and not re.match(r"# \S", lines[0]):
                missing.append(rel(path))
        self.assertEqual(missing, [])

    def test_logging_proxy_copies_are_identical(self):
        self.assertEqual((ROOT / "runtime/api_proxy.py").read_bytes(), (ROOT / "benchmarks/scripts/bench_api_proxy.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
