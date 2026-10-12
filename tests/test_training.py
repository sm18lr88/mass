# Offline checks for the torch-free SFT helpers in training/ (rendering, prompt matching, windows, sampling).
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

from mass.io import ROOT
from training import sft_prompts
from training.sft_prompts import prompt_match, substitute_prompt, task_of_cond, thread_role
from training.sft_requests import group_threads, load_requests, response_to_replay_message, vllm_postprocess
from training.sft_sampling import TaskRoleSampler, WeightedSampler, load_windows, role_group, split_val, task_of
from training.sft_windows import IGNORE, Renderer, make_windows

SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


class FakeTokenizer:
    """Char-level tokenizer with single-id special tokens and a minimal Qwen-like chat template."""

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for k, s in enumerate(SPECIAL):
                if text.startswith(s, i):
                    ids.append(1000 + k)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return 1000 + SPECIAL.index(token)

    def decode(self, ids):
        return "".join(SPECIAL[t - 1000] if t >= 1000 else chr(t) for t in ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, enable_thinking=True):
        out = ""
        for m in messages:
            if m["role"] == "assistant":
                out += f"<|im_start|>assistant\n<think>\n{m.get('reasoning_content', '')}\n</think>\n\n{m['content']}<|im_end|>\n"
            elif m["role"] == "tool":
                out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
            else:
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return out


def block(kind, ids, loss):
    return {"kind": kind, "ids": ids, "mask": [loss] * len(ids)}


def chat(seq, messages):
    return {"seq": seq, "request": {"messages": messages}}


class WindowTests(unittest.TestCase):
    def test_windows_mask_context_and_train_each_assistant_token_once(self):
        blocks = [
            block("system", [1, 2], False),
            block("user", [3, 4], False),
            block("assistant", [10, 11, 12], True),
            block("tool", [20, 21], False),
            block("assistant", [30, 31, 32], True),
        ]
        windows, skipped = make_windows(blocks, window=10, overlap=3)
        self.assertFalse(skipped)
        self.assertGreater(len(windows), 1)
        self.assertEqual([x for w in windows for x in w["labels"] if x != -100], [10, 11, 12, 30, 31, 32])
        self.assertTrue(all(len(w["input_ids"]) <= 10 for w in windows))

    def test_oversized_block_is_skipped_and_oversized_header_rejected(self):
        blocks = [block("system", [1], False), block("user", [2], False), block("assistant", list(range(20)), True)]
        windows, skipped = make_windows(blocks, window=10, overlap=3)
        self.assertEqual((windows, skipped), ([], [{"block": 2, "tokens": 20}]))
        with self.assertRaises(ValueError):
            make_windows([block("system", [1] * 10, False), block("user", [2], False)], window=10, overlap=3)

    def test_renderer_masks_only_generated_assistant_tokens(self):
        tok = FakeTokenizer()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "ok", "reasoning_content": "hmm", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]},
            {"role": "tool", "content": "res"},
            {"role": "assistant", "content": "done"},
        ]
        segs = Renderer(tok).segments(msgs, None)
        self.assertEqual(len(segs), 1)
        self.assertEqual([b["kind"] for b in segs[0]], ["system", "user", "assistant", "tool", "assistant"])
        trained = [tok.decode([t for t, m in zip(b["ids"], b["mask"], strict=True) if m]) for b in segs[0]]
        self.assertEqual(trained, ["", "", "hmm\n</think>\n\nok<|im_end|>", "", "\n</think>\n\ndone<|im_end|>"])
        self.assertEqual(Renderer.block_plan(msgs + [{"role": "tool", "content": "x"}, {"role": "tool", "content": "y"}])[-1], (5, 7, "tool"))
        with self.assertRaises(ValueError):
            Renderer(tok).segments(msgs[1:], None)
        windows, _ = make_windows(segs[0], window=200, overlap=50)
        self.assertEqual(sum(w["n_loss_tokens"] for w in windows), sum(sum(b["mask"]) for b in segs[0]))
        self.assertTrue(all(len(w["labels"]) == len(w["input_ids"]) and IGNORE in w["labels"] for w in windows))


class RequestTests(unittest.TestCase):
    def test_group_threads_splits_non_prefix_requests_and_orders_by_seq(self):
        s, u = {"role": "system", "content": "S"}, {"role": "user", "content": "U"}
        a1, a2 = {"role": "assistant", "content": "1"}, {"role": "assistant", "content": "2"}
        other = [{"role": "system", "content": "S"}, {"role": "user", "content": "V"}]
        chats = [chat(2, [s, u, a1, u]), chat(1, [s, u]), chat(3, [s, u, a2]), chat(4, other)]
        convs = group_threads(chats)
        self.assertEqual([[r["seq"] for r in seg] for _, seg in convs], [[1, 2], [3], [4]])
        self.assertEqual(convs[0][0], convs[1][0])
        self.assertNotEqual(convs[0][0], convs[2][0])

    def test_load_requests_prefers_gz_and_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_requests(d), (None, None))
            p = os.path.join(d, "api_requests.jsonl.gz")
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write(json.dumps({"seq": 1, "x": "\u00e9"}) + "\n\n{truncated\n")
            recs, path = load_requests(d)
        self.assertEqual((recs, path), ([{"seq": 1, "x": "\u00e9"}], p))

    def test_replay_message_and_vllm_argument_parsing(self):
        resp = {"content": None, "reasoning_content": "r", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}]}
        m = response_to_replay_message(resp)
        self.assertEqual(
            m,
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "r",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}],
            },
        )
        bad = dict(m, tool_calls=[{"function": {"name": "g", "arguments": "not json"}}])
        out = vllm_postprocess([m, bad])
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], "not json")
        self.assertEqual(m["tool_calls"][0]["function"]["arguments"], '{"a": 1}')  # input not mutated


class PromptTests(unittest.TestCase):
    def test_prompt_match_modes(self):
        self.assertEqual(prompt_match("x TASK y", "TASK"), ("TASK", "exact"))
        self.assertEqual(prompt_match("x TASK y", "  TASK\n"), ("TASK", "strip"))
        self.assertEqual(prompt_match([{"type": "text", "text": "use precision @k now"}], "use precision@k now\n"), ("use precision @k now", "qwen_metric_at"))
        self.assertIsNone(prompt_match("nothing", "TASK"))
        self.assertIsNone(prompt_match("TASK", None))
        with self.assertRaises(ValueError):
            prompt_match("TASK TASK", "TASK")

    def test_substitute_prompt_normalizes_bare_prompt_like_qwen_code(self):
        self.assertEqual(substitute_prompt("<ctx>TEACH</ctx>", "TEACH", "BARE"), "<ctx>BARE</ctx>")
        content = [{"type": "text", "text": "pre"}, {"type": "text", "text": "a precision @k b"}]
        out = substitute_prompt(content, "a precision@k b", "bare recall@k \n")
        assert isinstance(out, list)
        self.assertEqual(out[1]["text"], "bare recall @k")
        self.assertEqual(content[1]["text"], "a precision @k b")  # input not mutated
        with self.assertRaises(ValueError):
            substitute_prompt("unrelated", "TEACH", "BARE")

    def test_sent_index_enables_qwen_sent_substitution(self):
        self.addCleanup(sft_prompts.SENT.clear)
        with tempfile.TemporaryDirectory() as d:
            import hashlib

            entries = {}
            for name, orig, sent in (("t.txt", "teach @a", "teach SENT\u00e9"), ("b.txt", "bare @b", "bare SENT")):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(sent)
                entries[hashlib.md5(orig.encode("utf-8")).hexdigest()] = {"sent_file": name}
            with open(os.path.join(d, "INDEX.json"), "w", encoding="utf-8") as f:
                json.dump(entries, f)
            self.assertEqual(sft_prompts.load_sent_index(os.path.join(d, "INDEX.json")), 2)
        self.assertEqual(prompt_match("x teach SENT\u00e9 y", "teach @a"), ("teach SENT\u00e9", "qwen_sent"))
        self.assertEqual(substitute_prompt("x teach SENT\u00e9 y", "teach @a", "bare @b"), "x bare SENT y")

    def test_thread_role_and_task_of_cond(self):
        self.assertEqual(thread_role("anything", True), "orchestrator")
        self.assertEqual(thread_role([{"text": "You are Qwen Code, ..."}], False), "orchestrator")
        self.assertEqual(thread_role("You are a general-purpose subagent", False), "worker")
        self.assertEqual(thread_role(None, False), "auxiliary")
        self.assertEqual((task_of_cond("t202_v3"), task_of_cond("bare"), task_of_cond(None)), (202, None, None))


class SamplingTests(unittest.TestCase):
    ROWS = [
        {"condition": "t51_a", "episode": "t51_a_s1", "_task": 51, "_rg": "orch"},
        {"condition": "t51_a", "episode": "t51_a_s2", "_task": 51, "_rg": "nonmain"},
        {"condition": "t53_b", "episode": "t53_b_s1", "_task": 53, "_rg": "nonmain"},
    ]

    def test_role_group_and_task_of(self):
        self.assertEqual([role_group(r) for r in ({"role": "orchestrator"}, {"role": "worker"}, {"is_main": True}, {})], ["orch", "nonmain", "orch", "nonmain"])
        self.assertEqual([task_of(r) for r in ({"task": "7"}, {"condition": "t302_x"}, {"condition": "bare"})], [7, 302, -1])

    def test_split_val_by_episode_regex(self):
        self.assertEqual(split_val(self.ROWS, None), (self.ROWS, []))
        tr, va = split_val(self.ROWS, r"_s2$")
        self.assertEqual(([r["episode"] for r in tr], [r["episode"] for r in va]), (["t51_a_s1", "t53_b_s1"], ["t51_a_s2"]))

    def test_load_windows_filters_and_crops(self):
        rows = [
            {"condition": "t51_a", "is_main": True, "input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
            {"condition": "t51_a", "role": "worker", "input_ids": [1, 2], "labels": [2, -100]},
            {"condition": "skip", "input_ids": [1, 2], "labels": [-100, 2]},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            out = load_windows(p, argparse.Namespace(exclude_condition=["skip"], max_len=3))
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["input_ids"], out[0]["labels"], out[0]["_task"], out[0]["_rg"]), ([2, 3, 4], [-100, 3, 4], 51, "orch"))

    def test_samplers_are_seed_deterministic_and_respect_weights(self):
        draws = lambda s: [s.draw(r, k) for r in range(2) for k in range(50)]  # noqa: E731
        trs = TaskRoleSampler(self.ROWS, 0.5, seed=7)
        self.assertEqual(draws(trs), draws(TaskRoleSampler(self.ROWS, 0.5, seed=7)))
        self.assertNotEqual(draws(trs), draws(TaskRoleSampler(self.ROWS, 0.5, seed=8)))
        self.assertEqual(set(draws(trs)), {0, 1, 2})
        always_orch = draws(TaskRoleSampler(self.ROWS, 1.0, seed=7))
        self.assertNotIn(1, always_orch)
        ws = WeightedSampler(self.ROWS, {"t53_b": 0.0}, seed=3)
        self.assertEqual(draws(ws), draws(WeightedSampler(self.ROWS, {"t53_b": 0.0}, seed=3)))
        self.assertEqual(set(draws(ws)), {0, 1})
        self.assertIn("2 conditions", ws.describe())


class ScriptTests(unittest.TestCase):
    def test_prepare_sft_data_runs_by_path_without_transformers(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "training/prepare_sft_data.py"), "--help"], capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=ROOT
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--sent-index", p.stdout)


if __name__ == "__main__":
    unittest.main()
