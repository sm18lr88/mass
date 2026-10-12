# Driver tests: explicit stage dispatch, JSONL integrity, UTF-8 episode reuse, and plan output.
import contextlib
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from mass import collection, episodes, pipeline, process, reporting, selection, sft
from mass.io import ROOT, append_record, read_json, records, write_json

BASELINE = ROOT / ".omo/evidence/baseline"


class StageTableTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        c = read_json(ROOT / "configs/paper.json")
        c["run_dir"] = str(self.base / "run")
        self.config = self.base / "config.json"
        write_json(self.config, c)
        self.c = c

    def main(self, *argv):
        pipeline.main([*argv, "--config", str(self.config)])

    def test_each_stage_dispatches_to_its_callable(self):
        train, all_tasks = self.c["tasks"]["train"], self.c["tasks"]["train_candidates"] + self.c["tasks"]["test"]
        cases = {
            "search": (collection, "search", lambda c: ((c, all_tasks),)),
            "collect": (collection, "collect", lambda c: ((c, train),)),
            "rank": (selection, "rank", lambda c: ((c, train),)),
            "select": (selection, "select", lambda c: ((c,),)),
            "render": (sft, "render", lambda c: ((c,),)),
            "train": (sft, "train", lambda c: ((c,),)),
            "export": (sft, "export", lambda c: ((c,),)),
            "rollout": (reporting, "rollout", lambda c: ((c, all_tasks, 10),)),
            "report": (reporting, "report", lambda c: ((c, Path("pairs.json")),)),
        }
        self.assertEqual(set(pipeline.STAGES), set(cases) | {"plan"})
        for stage, (module, name, expected) in cases.items():
            with self.subTest(stage=stage), patch.object(module, name) as fake:
                extra = ["--pairs", "pairs.json"] if stage == "report" else []
                self.main(stage, *extra)
                fake.assert_called_once()
                self.assertEqual(fake.call_args.args, expected(self.c)[0])

    def test_plan_dispatch_does_not_freeze_config(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.main("plan")
        self.assertFalse((self.base / "run/config.json").exists())

    def test_train_refuses_existing_log_without_launching(self):
        log = self.base / "run/training/train_log.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text("", encoding="utf-8")
        with patch.object(process, "run") as fake, self.assertRaises(FileExistsError):
            self.main("train")
        fake.assert_not_called()

    def test_train_launches_through_the_process_seam(self):
        with patch.object(process, "run") as fake:
            self.main("train")
        self.assertEqual(fake.call_args.args[0], sft.training_command(self.c))


class RecordsTests(unittest.TestCase):
    def test_truncated_last_line_raises_with_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.jsonl"
            append_record(path, {"a": "x", "winner": "A"})
            append_record(path, {"a": "y", "winner": "B"})
            with path.open("a", encoding="utf-8") as f:
                f.write('{"a": "z", "win')
            with self.assertRaises(ValueError) as caught:
                records(path)
            self.assertIn(str(path), str(caught.exception))
            self.assertIn("line 3", str(caught.exception))

    def test_complete_file_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            self.assertEqual(records(path), [])
            append_record(path, {"k": "é"})
            self.assertEqual(records(path), [{"k": "é"}])


class EpisodeReuseTests(unittest.TestCase):
    def test_non_ascii_prompt_reuses_completed_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, logs = base / "prompt.txt", base / "logs"
            text = "Analyse the données — résumé, 数据, ✓\n"
            prompt.write_text(text, encoding="utf-8")
            calls = []

            def fake_run(cmd, env=None):
                calls.append(cmd)
                logs.mkdir()
                (logs / "prompt.txt").write_text(prompt.read_text(encoding="utf-8"), encoding="utf-8")
                write_json(logs / "run_status.json", {"exit_code": 0, "trace_has_success_result": True})

            c = read_json(ROOT / "configs/paper.json")
            with patch.object(process, "run", fake_run):
                episodes.episode(c, prompt, base / "ws", logs, 1, "t1")
                episodes.episode(c, prompt, base / "ws", logs, 1, "t1")
            self.assertEqual(len(calls), 1)
            self.assertEqual((logs / "prompt.txt").read_bytes(), prompt.read_bytes())
            prompt.write_text(text + "changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                episodes.episode(c, prompt, base / "ws", logs, 1, "t1")


def normalized_plan(text: str) -> str:
    # Only the interpreter token on the training-command line may differ between environments.
    return re.sub(r"^(Training command \(not executed\): )('[^']*'|\S+)", r"\1<python>", text, flags=re.M)


class PlanOutputTests(unittest.TestCase):
    def test_plan_output_matches_baseline(self):
        for name in ("paper", "cycle2"):
            baseline = BASELINE / f"plan-{name}.stdout"
            if not baseline.is_file():
                self.skipTest(f"baseline not present: {baseline}")
            with self.subTest(config=name):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    pipeline.main(["plan", "--config", str(ROOT / f"configs/{name}.json")])
                self.assertEqual(normalized_plan(buffer.getvalue()),
                                 normalized_plan(baseline.read_text(encoding="utf-8")))
                self.assertIn(shlex.quote(sys.executable), buffer.getvalue())

    def test_plan_prints_validated_config_json(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            pipeline.main(["plan", "--config", str(ROOT / "configs/paper.json")])
        document = buffer.getvalue().split("\nL^(")[0]
        self.assertEqual(json.loads(document), read_json(ROOT / "configs/paper.json"))


if __name__ == "__main__":
    unittest.main()
