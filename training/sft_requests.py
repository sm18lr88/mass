"""Torch-free proxy request-log loading, vLLM message post-processing, and grouping of chat requests into conversation threads.

Imported as ``training.sft_requests``: the training scripts append the repository root to
``sys.path`` before importing (see ``training/__init__.py``), so the same module object is used
when a script runs by path and when tests import it.
"""

import copy
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import IO, Any

Message = dict[str, Any]


def open_maybe_gz(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_requests(logs_dir: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Read api_requests.jsonl[.gz]; malformed lines (e.g. a truncated tail) are skipped with a warning."""
    for cand in ("api_requests.jsonl.gz", "api_requests.jsonl"):
        p = os.path.join(logs_dir, cand)
        if os.path.exists(p):
            recs, bad = [], 0
            with open_maybe_gz(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            bad += 1
            if bad:
                print(f"WARN {p}: skipped {bad} malformed JSON line(s)", file=sys.stderr)
            return recs, p
    return None, None


def vllm_postprocess(messages: list[Message]) -> list[Message]:
    """Mimic vLLM's request post-processing: tool_call.function.arguments str -> object."""
    out = []
    for m in messages:
        m = copy.deepcopy(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a)
                    except json.JSONDecodeError:
                        pass  # vLLM leaves non-JSON arguments as the raw string; so do we
        out.append(m)
    return out


def response_to_replay_message(resp_msg: Message) -> Message:
    """Convert a proxy-reassembled response message to the replay (request) form."""
    m = {"role": "assistant", "content": resp_msg.get("content") or ""}
    if resp_msg.get("reasoning_content"):
        m["reasoning_content"] = resp_msg["reasoning_content"]
    tcs = resp_msg.get("tool_calls") or []
    if tcs:
        m["tool_calls"] = [
            {"id": t.get("id"), "type": t.get("type") or "function", "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
            for t in tcs
        ]
    return m


def thread_key(msgs: list[Message]) -> str:
    u = msgs[1]["content"] if len(msgs) > 1 else ""
    return hashlib.md5((msgs[0]["content"] + "\x00" + json.dumps(u, ensure_ascii=False)).encode()).hexdigest()[:10]


def messages_prefix(a: list[Message], b: list[Message]) -> bool:
    """True if message list a is a prefix of b (exact JSON equality per message)."""
    if len(a) > len(b):
        return False
    return all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b, strict=False))


def group_threads(chats: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group chat requests into (thread_key, segment) conversations."""
    by_key = defaultdict(list)
    for r in chats:
        by_key[thread_key(r["request"]["messages"])].append(r)
    convs = []
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r["seq"])
        seg = [rs[0]]
        for r in rs[1:]:
            if messages_prefix(seg[-1]["request"]["messages"], r["request"]["messages"]):
                seg.append(r)
            else:
                convs.append((key, seg))
                seg = [r]
        convs.append((key, seg))
    return convs
