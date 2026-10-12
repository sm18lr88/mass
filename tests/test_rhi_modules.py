# Offline unit tests for the pure helpers and the optimizer path extracted from the RHI entry module.
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_improvement.llm_backend import BACKEND_LOCAL_VLLM, BACKEND_OPENAI_RESPONSES
from harness_improvement.rhi_champion import (
    _baseline_scoreboard,
    _derive_champion,
    _derive_champion_baseline,
)
from harness_improvement.rhi_design import (
    _fixed_agent_limit_matches,
    _normalize_generated_design_block,
)
from harness_improvement.rhi_history import _history_key
from harness_improvement.rhi_io import _write_json_once
from harness_improvement.rhi_judge_adjacent import _adjacent_history_provenance_errors
from harness_improvement.rhi_template import DEFAULT_END_MARKER, DEFAULT_START_MARKER
from harness_improvement.rhi_validate import _is_loopback_base_url
from mass.io import ROOT


def adj(a, b, winner):
    return {"key": _history_key(a, b), "winner": winner}


def base(v, winner):
    return {"key": f"v{v}_vs_no_harness", "winner": winner}


def arb(champ, v, winner):
    return {"key": f"champion_v{champ}_vs_v{v}", "winner": winner}


class HistoryAndProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.prev, self.cur = Path("/runs/v0").resolve(), Path("/runs/v1").resolve()
        self.row = {"key": "v0_vs_v1", "version_a": 0, "version_b": 1, "llm_backend": BACKEND_LOCAL_VLLM,
                    "model": "m", "repo_a": str(self.prev), "repo_b": str(self.cur),
                    "base_url": "http://127.0.0.1:8001/v1/", "call_metrics": {"estimated_cost_usd": 0.0}}

    def errors(self, row, **overrides):
        kwargs = dict(comparison_key="v0_vs_v1", prev_version=0, current_version=1, prev_repo=self.prev,
                      current_repo=self.cur, llm_backend=BACKEND_LOCAL_VLLM, model="m",
                      base_url="http://127.0.0.1:8001/v1")
        kwargs.update(overrides)
        return _adjacent_history_provenance_errors(row, **kwargs)

    def test_history_key(self):
        self.assertEqual(_history_key(3, 4), "v3_vs_v4")

    def test_matching_row_has_no_errors(self):
        self.assertEqual(self.errors(self.row), [])

    def test_mismatches_are_reported(self):
        row = dict(self.row, model="other", repo_b="/elsewhere", call_metrics={"estimated_cost_usd": 1.5})
        errors = self.errors(row)
        self.assertEqual(len(errors), 3)
        self.assertTrue(errors[0].startswith("model='other'"))

    def test_external_backend_skips_local_url_and_cost_checks(self):
        row = dict(self.row, llm_backend=BACKEND_OPENAI_RESPONSES, base_url=None, call_metrics={})
        self.assertEqual(self.errors(row, llm_backend=BACKEND_OPENAI_RESPONSES), [])


class ChampionTests(unittest.TestCase):
    def test_adjacent_chain_advances_on_direct_wins(self):
        self.assertEqual(_derive_champion([adj(0, 1, "B"), adj(1, 2, "B")], [], 2), (2, False))

    def test_adjacent_win_over_non_champion_needs_arbitration(self):
        rows = [adj(0, 1, "B"), adj(1, 2, "A"), adj(2, 3, "B")]
        self.assertEqual(_derive_champion(rows, [], 3), (1, True))
        self.assertEqual(_derive_champion(rows, [arb(1, 3, "B")], 3), (3, False))
        self.assertEqual(_derive_champion(rows, [arb(1, 3, "A")], 3), (1, False))

    def test_baseline_champion(self):
        self.assertEqual(_derive_champion_baseline([base(0, "A"), base(1, "A")], [], 1), (0, False, False))
        self.assertEqual(_derive_champion_baseline([base(0, "A"), base(1, "B")], [], 1), (1, True, False))
        rows = [base(0, "B"), base(1, "B")]
        self.assertEqual(_derive_champion_baseline(rows, [], 1), (0, True, True))
        self.assertEqual(_derive_champion_baseline(rows, [arb(0, 1, "B")], 1), (1, True, False))

    def test_scoreboard_marks_proven_champion(self):
        rows = [base(0, "A"), base(1, "B"), base(2, "tie")]
        self.assertEqual(_baseline_scoreboard(rows, 3, 1, True),
                         "Record vs no_harness: v0 L · v1 W★ · v2 T · v3 ?")
        self.assertNotIn("★", _baseline_scoreboard(rows, 3, 0, False))


class DesignAndUrlTests(unittest.TestCase):
    def test_normalize_trims_copied_tail_and_late_end_marker(self):
        design = f"{DEFAULT_START_MARKER}\nAgent A."
        after = f"{DEFAULT_END_MARKER} and log everything."
        self.assertEqual(_normalize_generated_design_block(f"{design}\n\n{after}\nmore", start_marker=DEFAULT_START_MARKER,
                                                           end_marker=DEFAULT_END_MARKER, after_design=after), design)
        late = f"{design}\n{DEFAULT_END_MARKER}"
        self.assertEqual(_normalize_generated_design_block(late, start_marker=DEFAULT_START_MARKER,
                                                           end_marker=DEFAULT_END_MARKER, after_design=""), design)

    def test_normalize_requires_start_marker(self):
        with self.assertRaises(RuntimeError):
            _normalize_generated_design_block("no marker", start_marker=DEFAULT_START_MARKER,
                                              end_marker=DEFAULT_END_MARKER, after_design="")

    def test_fixed_agent_limit_matches(self):
        self.assertEqual(_fixed_agent_limit_matches("Use at most 3 recalls. Tool calls are limited to 5."),
                         ["at most 3 recalls", "Tool calls are limited to 5"])
        self.assertEqual(_fixed_agent_limit_matches("Iterate until acceptance gates pass."), [])

    def test_loopback_base_url(self):
        for url in ("http://127.0.0.1:8001/v1", "http://localhost:8000/v1", "https://[::1]:9/v1"):
            self.assertTrue(_is_loopback_base_url(url), url)
        for url in ("http://10.0.0.5:8000/v1", "ftp://127.0.0.1/", "http://u:p@127.0.0.1/", "http://example.com", ""):
            self.assertFalse(_is_loopback_base_url(url), url)

    def test_write_json_once_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a" / "r.json"
            _write_json_once(path, {"x": 1})
            _write_json_once(path, {"x": 1})
            with self.assertRaises(RuntimeError):
                _write_json_once(path, {"x": 2})


@unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("pydantic"), "requires the core dependencies (uv sync)")
class OptimizerPathTests(unittest.TestCase):
    def test_adjacent_judge_then_json_proposal_writes_next_version(self):
        from harness_improvement import iterate_multi_agent_prompt as rhi
        calls = []

        class FakeBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def verify_model(self):
                pass

            def call_json(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    payload = {"winner": "B", "rationale": "v1 is missing nothing."}
                else:
                    payload = {"improved_multi_agent_design": f"{rhi.DEFAULT_START_MARKER}\nNew design v2."}
                return SimpleNamespace(payload=payload, raw_text=json.dumps(payload), metrics={"estimated_cost_usd": 0.0})

        prompt = ROOT / "tasks/initial/query51_ourTeam.txt"
        template = rhi._read_query_template(prompt, start_marker=rhi.DEFAULT_START_MARKER, end_marker=rhi.DEFAULT_END_MARKER)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "search"
            (out / "versions").mkdir(parents=True)
            for v in (0, 1):
                (out / f"versions/multi_agent_design_v{v}.txt").write_text(template.design_block, encoding="utf-8")
                (root / f"v{v}/query51_ourTeam").mkdir(parents=True)
            argv = ["rhi", "--query-file", str(prompt), "--current-version", "1", "--out-dir", str(out),
                    "--current-repo", str(root / "v1/query51_ourTeam"), "--previous-repo", str(root / "v0/query51_ourTeam"),
                    "--llm-backend", "local-vllm", "--require-local-vllm", "--model", "fixture-model",
                    "--strict-adjacent-history-provenance", "--forbid-fixed-agent-limits"]
            with patch.object(sys, "argv", argv), patch.object(rhi, "JsonLLMBackend", FakeBackend), \
                    patch("builtins.print"):
                self.assertEqual(rhi.main(), 0)
            self.assertEqual(len(calls), 2)
            next_design = (out / "versions/multi_agent_design_v2.txt").read_text(encoding="utf-8")
            self.assertEqual(next_design, f"{rhi.DEFAULT_START_MARKER}\nNew design v2.\n")
            rendered = (out / "updated_queries/query51_ourTeam-v2.txt").read_text(encoding="utf-8")
            self.assertIn("New design v2.", rendered)
            summary = json.loads((out / "last_run_summary_v1_to_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["api_calls_count_this_run"], 2)
            self.assertEqual(summary["estimated_cost_usd_this_run"], 0.0)
            history = (out / "records/comparison_history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(history[0])["key"], "v0_vs_v1")


if __name__ == "__main__":
    unittest.main()
