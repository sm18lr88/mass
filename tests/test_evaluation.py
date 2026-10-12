# Offline tests for evaluation evidence bundling and pairwise judging (protocol goldens included).
import hashlib
import json
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from evaluation import token_budget
from evaluation.artifacts import build_file_tree, read_text_file, scan_deliverables
from evaluation.bundle import build_judge_bundle, bundle_to_user_message
from evaluation.token_budget import TRUNCATION_MARKER, _truncate_text_to_token_budget, estimate_tokens
from evaluation_claudecodex import pairwise_providers
from evaluation_claudecodex.pairwise_judge import build_pairwise_user_message, run_pairwise_judge
from evaluation_claudecodex.pairwise_parsing import _extract_json_candidate
from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num

TASK = {
    "query_id": "query_51",
    "query": "Build a strategy.\n\nDeliverables:\n(1) deliverables/plots/equity_curve.png\n(2) drawdown.png\n(3) research_report.md — résumé\n",
}

# SHA-256 of protocol outputs rendered by the pre-refactor code (HEAD fb40358); workspace root replaced by <ROOT>.
GOLDEN_SHA256 = {
    "bundle_a": "704d76bed769d714fcb3c16dda2ce2560dcfa6bfc568b412371bb1346a535512",
    "bundle_tokens": "16081863e5b61fdd4190f3f72a86ba3160ee6c51480b414e91a64f15b48600b1",
    "pairwise": "55276a6c7eddf03aaf49380d233dda1fec71a6450aa50f493f7ac1c8a838751d",
    "system_51": "2d008009d920bbb1e4b87381a2719698cabcb7243f39144ea8de257313a14215",
    "system_60": "2d008009d920bbb1e4b87381a2719698cabcb7243f39144ea8de257313a14215",
    "system_202": "522f61f177a26585f02987dca3a38398d9652d7bd88024862d0049c1f7e6dfa6",
    "system_305": "48466b1a30e27d67ceba29e02ad16cbdfea31e1468548c7e6f118a5ceedcf882",
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def make_workspace(root: Path, variant: str) -> None:
    write(root / "research_report.md", f"# Report {variant}\n" + ("Résumé line αβγ " * 20 + "\n") * 60)
    write(root / "deliverables" / "index.json", json.dumps({"variant": variant}))
    write(root / "results" / "metrics.json", json.dumps({"sharpe": 1.23}))
    write(root / "requirements.txt", "numpy\n")
    write(root / "run_pipeline.py", "print('hi')\n" * 2000)
    write(root / "tests" / "test_x.py", "def test_x():\n    assert True\n")
    (root / "deliverables" / "plots").mkdir(parents=True, exist_ok=True)
    (root / "deliverables" / "plots" / "equity_curve.png").write_bytes(b"\x89PNG")
    write(root / ".git" / "HEAD", "ref\n")
    write(root / "node_modules" / "x.js", "x\n")
    for i in range(105):
        write(root / "data" / f"f{i:03d}.csv", "a,b\n")
    write(root / "workflow.log", "LOG START\n" + "step ok\n" * 4000 + "LOG END\n")


def render_protocol_outputs(base: Path) -> dict[str, str]:
    ra, rb = base / "subA", base / "subB"
    make_workspace(ra, "A")
    make_workspace(rb, "B")
    ba = build_judge_bundle(memory_root=ra, task_row=TASK, log_path=ra / "workflow.log")
    bb = build_judge_bundle(memory_root=rb, task_row=TASK, log_path=None)
    bt = build_judge_bundle(
        memory_root=ra,
        task_row=TASK,
        log_path=ra / "workflow.log",
        report_max_tokens=300,
        json_max_tokens=5,
        code_max_tokens=50,
        report_max_chars=None,
        tree_max_files=12,
        tree_max_depth=2,
        log_head=100,
        log_tail=100,
    )
    out = {
        "bundle_a": bundle_to_user_message(ba),
        "bundle_tokens": bundle_to_user_message(bt),
        "pairwise": build_pairwise_user_message(ba, bb, label_a="bare", label_b="ourTeam"),
    }
    out = {k: v.replace(str(base.resolve()), "<ROOT>") for k, v in out.items()}
    for q in (51, 60, 202, 305):
        out[f"system_{q}"] = pairwise_judge_system_for_query_num(q)
    return out


def sha256s(outputs: dict[str, str]) -> dict[str, str]:
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in outputs.items()}


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class ProtocolGoldenTests(TempDirCase):
    def test_rendered_protocol_matches_pre_refactor_goldens(self) -> None:
        self.assertEqual(sha256s(render_protocol_outputs(self.root)), GOLDEN_SHA256)


class DeliverableScanTests(TempDirCase):
    def test_plot_mentions_split_into_found_and_missing(self) -> None:
        make_workspace(self.root, "A")
        scan = scan_deliverables(self.root, TASK["query"])
        self.assertEqual(scan.png_mentions_in_query, ["deliverables/plots/drawdown.png", "deliverables/plots/equity_curve.png"])
        self.assertEqual(scan.png_found, ["deliverables/plots/equity_curve.png"])
        self.assertEqual(scan.png_missing, ["deliverables/plots/drawdown.png"])
        self.assertEqual(
            scan.core_paths,
            {"research_report.md": True, "deliverables/index.json": True, "results/metrics.json": True, "requirements.txt": True},
        )

    def test_empty_workspace_reports_core_files_missing(self) -> None:
        scan = scan_deliverables(self.root, "no plots here")
        self.assertEqual(scan.png_mentions_in_query, [])
        self.assertFalse(any(scan.core_paths.values()))


class FileTreeTests(TempDirCase):
    def test_ignored_dirs_hidden_and_binaries_marked(self) -> None:
        make_workspace(self.root, "A")
        tree = build_file_tree(self.root, max_children_per_dir=200)
        self.assertNotIn(".git", tree)
        self.assertNotIn("node_modules", tree)
        self.assertIn("    equity_curve.png  [.png binary/large]", tree)
        self.assertTrue(tree.startswith(f"{self.root.name}/\n"))
        self.assertNotIn("(tree truncated)", tree)

    def test_large_directory_collapsed_to_omission_marker(self) -> None:
        make_workspace(self.root, "A")
        tree = build_file_tree(self.root)
        self.assertIn("  data/\n    ... [contents omitted: 105 entries exceed per-directory limit 100]", tree)
        self.assertNotIn("f000.csv", tree)

    def test_entry_budget_truncates_tree(self) -> None:
        make_workspace(self.root, "A")
        lines = build_file_tree(self.root, max_files=3).splitlines()
        # Three entries consume the budget; the per-directory omission marker does not.
        self.assertEqual(
            lines,
            [
                f"{self.root.name}/",
                "  data/",
                "    ... [contents omitted: 105 entries exceed per-directory limit 100]",
                "  deliverables/",
                "    plots/",
                "... (tree truncated)",
            ],
        )


class TokenBudgetTests(unittest.TestCase):
    def test_text_within_budget_is_unchanged(self) -> None:
        self.assertEqual(_truncate_text_to_token_budget("abc", max_tokens=10, token_count=estimate_tokens), ("abc", False))
        self.assertEqual(_truncate_text_to_token_budget("x" * 999, max_tokens=0, token_count=estimate_tokens), ("x" * 999, False))

    def test_long_text_keeps_head_and_tail_within_budget(self) -> None:
        text = "HEAD" + "m" * 10_000 + "TAIL"
        out, truncated = _truncate_text_to_token_budget(text, max_tokens=200, token_count=estimate_tokens)
        self.assertTrue(truncated)
        self.assertLessEqual(estimate_tokens(out), 200)
        self.assertTrue(out.startswith("HEAD") and out.endswith("TAIL"))
        self.assertIn(TRUNCATION_MARKER, out)

    def test_budget_smaller_than_marker_returns_marker(self) -> None:
        self.assertEqual(_truncate_text_to_token_budget("y" * 500, max_tokens=5, token_count=estimate_tokens), (TRUNCATION_MARKER, True))

    def test_read_text_file_char_cap_marks_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp) / "a.md", "é" * 100)
            fr = read_text_file(Path(tmp), "a.md", max_chars=10)
            assert fr is not None
            self.assertTrue(fr.truncated)
            self.assertEqual(fr.content, "é" * 5 + "\n\n[... middle omitted ...]\n\n" + "é" * 5)
            self.assertIsNone(read_text_file(Path(tmp), "missing.md", max_chars=10))

    def test_tokenizer_failure_warns_and_falls_back_to_estimate(self) -> None:
        fake = types.SimpleNamespace(get_encoding=mock.Mock(side_effect=OSError("offline")))
        with mock.patch.object(token_budget, "tiktoken", fake), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            count, mode = token_budget._token_counter_for_model("gpt-5.5")
        self.assertEqual(mode, "estimate:chars_div_4")
        self.assertEqual(count("abcdefgh"), 2)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))


class JsonCandidateTests(unittest.TestCase):
    def test_extracts_fenced_embedded_or_passthrough(self) -> None:
        self.assertEqual(_extract_json_candidate('```json\n{"winner": "A"}\n```'), '{"winner": "A"}')
        self.assertEqual(_extract_json_candidate('Verdict: {"winner": "B", "x": {"y": 1}} done'), '{"winner": "B", "x": {"y": 1}}')
        self.assertEqual(_extract_json_candidate("  no json  "), "no json")
        self.assertEqual(_extract_json_candidate("   "), "")


def _bundles(root: Path):
    make_workspace(root / "a", "A")
    make_workspace(root / "b", "B")
    return (build_judge_bundle(memory_root=root / "a", task_row=TASK, log_path=None), build_judge_bundle(memory_root=root / "b", task_row=TASK, log_path=None))


class RunPairwiseJudgeTests(TempDirCase):
    def _openai_client(self, text: str) -> mock.Mock:
        client = mock.Mock()
        client.responses.create.return_value = types.SimpleNamespace(output_text=text, usage=None, id="resp_1")
        return client

    def test_openai_route_parses_verdict_and_sends_protocol(self) -> None:
        ba, bb = _bundles(self.root)
        client = self._openai_client('{"winner": "tie", "rationale": "comparable"}')
        result, metrics = run_pairwise_judge(client=client, model="gpt-5.5", bundle_a=ba, bundle_b=bb, label_a="x", label_b="y")
        self.assertEqual((result.winner, result.rationale, metrics.response_id), ("tie", "comparable", "resp_1"))
        kwargs = client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["instructions"], pairwise_judge_system_for_query_num(51))
        self.assertEqual(kwargs["input"], build_pairwise_user_message(ba, bb, label_a="x", label_b="y"))
        self.assertEqual(kwargs["reasoning"], {"effort": "medium"})

    def test_invalid_json_raises_chained_runtime_error(self) -> None:
        ba, bb = _bundles(self.root)
        with self.assertRaises(RuntimeError) as ctx:
            run_pairwise_judge(client=self._openai_client("not json"), model="m", bundle_a=ba, bundle_b=bb, label_a="x", label_b="y")
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_unknown_provider_rejected(self) -> None:
        ba, bb = _bundles(self.root)
        with self.assertRaises(ValueError):
            run_pairwise_judge(client=None, provider="bogus", model="m", bundle_a=ba, bundle_b=bb, label_a="x", label_b="y")  # type: ignore[arg-type]

    def test_anthropic_route_extracts_fenced_json(self) -> None:
        ba, bb = _bundles(self.root)
        message = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text='```json\n{"winner": "B", "rationale": "r"}\n```')],
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
            id="msg_1",
            stop_reason="end_turn",
        )
        stream = mock.MagicMock()
        stream.__enter__.return_value.get_final_message.return_value = message
        client = mock.Mock()
        client.messages.stream.return_value = stream
        result, metrics = run_pairwise_judge(client=client, provider="anthropic", model="claude-opus-4-7", bundle_a=ba, bundle_b=bb, label_a="x", label_b="y")
        self.assertEqual(result.winner, "B")
        self.assertEqual(metrics.usage, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        self.assertEqual(client.messages.stream.call_args.kwargs["output_config"], {"effort": "medium"})

    def test_google_retries_without_thinking_and_chains_errors(self) -> None:
        ba, bb = _bundles(self.root)
        fake_types = types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw, ThinkingConfig=lambda **kw: kw)
        ok = types.SimpleNamespace(text='{"winner": "A", "rationale": "r"}', usage_metadata=None, response_id="g1")
        first, second = RuntimeError("thinking rejected"), RuntimeError("still failing")
        client = mock.Mock()
        with mock.patch.object(pairwise_providers, "google_types", fake_types):
            client.models.generate_content.side_effect = [first, ok]
            result, _ = run_pairwise_judge(client=client, provider="google", model="gemini-3.1-pro", bundle_a=ba, bundle_b=bb, label_a="x", label_b="y")
            self.assertEqual(result.winner, "A")
            self.assertNotIn("thinking_config", client.models.generate_content.call_args.kwargs["config"])
            client.models.generate_content.side_effect = [first, second]
            with self.assertRaises(RuntimeError) as ctx:
                run_pairwise_judge(client=client, provider="google", model="gemini-3.1-pro", bundle_a=ba, bundle_b=bb, label_a="x", label_b="y")
        self.assertIs(ctx.exception, second)
        self.assertIs(ctx.exception.__cause__, first)


if __name__ == "__main__":
    unittest.main()
