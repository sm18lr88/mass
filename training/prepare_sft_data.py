#!/usr/bin/env python3
# CLI: renders recorded qwen-code episodes into masked SFT windows JSONL (+ optional stats/strict audit).
"""Render recorded qwen-code conversations into masked SFT windows."""

import argparse
import copy
import glob
import json
import os
import sys
from collections import Counter
from typing import Any

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:  # run by path: sys.path[0] is training/; helpers are imported as training.<module>
    sys.path.append(_REPO)

from training.sft_prompts import (  # noqa: E402
    load_sent_index,
    prompt_match,
    substitute_prompt,
    task_of_cond,
    thread_role,
)
from training.sft_requests import group_threads, load_requests, response_to_replay_message  # noqa: E402
from training.sft_windows import Renderer, make_windows  # noqa: E402


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def process_episode(
    logs_dir: str, renderer: Renderer, args: argparse.Namespace, teacher_prompts: dict[str, str], bare: Any
) -> tuple[list[dict[str, Any]] | None, Any]:
    bare_map, bare_default = bare if isinstance(bare, tuple) else ({}, bare)
    status_p = os.path.join(logs_dir, "run_status.json")
    status = _read_json(status_p) if os.path.exists(status_p) else None
    recs, _log_path = load_requests(logs_dir)
    if recs is None:
        return None, "no api log"
    chats = [
        r for r in recs if r.get("path") == "/v1/chat/completions" and r.get("status") == 200 and r.get("response") and (r["response"].get("choices") or [])
    ]
    if not chats:
        return None, "no chat requests"
    cond = (status or {}).get("hiw", {}).get("condition") or os.path.basename(logs_dir).rsplit("_s", 1)[0]
    teacher_prompt = teacher_prompts.get(cond)
    bare_prompt = bare_map.get(cond, bare_default)
    if bare_prompt is None:
        return None, f"no bare prompt for condition {cond}"
    convs = group_threads(chats)
    ep_name = os.path.basename(logs_dir.rstrip("/"))
    samples, info = (
        [],
        {
            "episode": ep_name,
            "condition": cond,
            "seed": (status or {}).get("seed"),
            "success": (status or {}).get("trace_has_success_result"),
            "n_requests": len(chats),
            "threads": [],
            "skipped_blocks": [],
        },
    )
    for key, seg in convs:
        last = seg[-1]
        req = last["request"]
        msgs = copy.deepcopy(req["messages"])
        choice = last["response"]["choices"][0]
        final = response_to_replay_message(choice["message"])
        truncated = choice.get("finish_reason") == "length"
        if not (truncated and args.drop_truncated_final):
            msgs.append(final)
        hit = prompt_match(msgs[1]["content"], teacher_prompt) if teacher_prompt else prompt_match(msgs[1]["content"], bare_prompt)
        is_main = hit is not None
        role = thread_role(msgs[0]["content"], is_main)
        if is_main and teacher_prompt and args.substitute:
            msgs[1]["content"] = substitute_prompt(msgs[1]["content"], teacher_prompt, bare_prompt)
        try:
            seg_blocks = renderer.segments(msgs, req.get("tools"))
        except ValueError as e:
            info["threads"].append({"key": key, "error": str(e)})
            continue
        n_tok = sum(len(b["ids"]) for b in seg_blocks[-1])
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        wins, skipped = [], []
        try:
            for s_i, blocks in enumerate(seg_blocks):
                w_s, sk_s = make_windows(blocks, args.window, args.context_overlap)
                for w in w_s:
                    w["segment"] = s_i
                wins += w_s
                skipped += sk_s
        except ValueError as e:  # e.g. a sub-agent thread whose header alone nearly fills the window
            info["threads"].append({"key": key, "is_main": is_main, "error": str(e)})
            continue
        for w_i, w in enumerate(wins):
            w.update({
                "episode": ep_name,
                "condition": cond,
                "seed": info["seed"],
                "thread": key,
                "task": task_of_cond(cond),
                "is_main": is_main,
                "role": role,
                "match_mode": (hit[1] if hit else None),
                "window": w_i,
                "n_windows": len(wins),
                "success": info["success"],
            })
            samples.append(w)
        info["threads"].append({
            "key": key,
            "is_main": is_main,
            "role": role,
            "match_mode": (hit[1] if hit else None),
            "n_requests": len(seg),
            "n_messages": len(msgs),
            "n_segments": len(seg_blocks),
            "n_tokens": n_tok,
            "n_assistant_blocks": n_asst,
            "n_windows": len(wins),
            "final_truncated": truncated,
            "skipped": skipped,
        })
        info["skipped_blocks"] += skipped
    return samples, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", default=[], help="runlog dirs or globs")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--bare-prompt", default=None, help="student prompt (single-task); or use --bare-prompt-map")
    ap.add_argument("--bare-prompt-map", action="append", default=[], help="condition=path (multi-task): bare prompt per condition")
    ap.add_argument("--episode-list", default=None, help="JSON list of {logs_dir, condition?, weight?} (from select_episodes.py); overrides --episodes")
    ap.add_argument("--teacher-prompt", action="append", default=[], help="condition=path, e.g. teacher_v10=/path/query51_ourTeam-v10.txt")
    ap.add_argument("--window", type=int, default=32768)
    ap.add_argument("--context-overlap", type=int, default=8192)
    ap.add_argument("--require-success", action="store_true")
    ap.add_argument("--no-substitute", dest="substitute", action="store_false")
    ap.add_argument("--keep-truncated-final", dest="drop_truncated_final", action="store_false")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument(
        "--strict", action="store_true", help="exit 1 if any used episode has a thread error, a skipped block, or a main thread without a prompt match"
    )
    ap.add_argument("--sent-index", default=None, help="sent/INDEX.json (qwen-code sent forms of the prompts)")
    args = ap.parse_args()
    if args.sent_index:
        print(f"sent-form index: {load_sent_index(args.sent_index)} prompts", file=sys.stderr)

    from transformers import AutoTokenizer  # heavy: imported only when actually rendering

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Renderer(tok)
    bare_default = _read_text(args.bare_prompt) if args.bare_prompt else None
    bare_map = {}
    for spec in args.bare_prompt_map:
        c, p = spec.split("=", 1)
        bare_map[c] = _read_text(p)
    teacher_prompts = {}
    for spec in args.teacher_prompt:
        c, p = spec.split("=", 1)
        teacher_prompts[c] = _read_text(p)

    weights = {}
    if args.episode_list:
        items = _read_json(args.episode_list)
        dirs = [it["logs_dir"] for it in items]
        weights = {os.path.basename(it["logs_dir"].rstrip("/")): it.get("weight", 1.0) for it in items}
    else:
        dirs = []
        for pat in args.episodes:
            dirs += sorted(d for d in glob.glob(os.path.expanduser(pat)) if os.path.isdir(d))
    all_info, n_samples = [], 0
    agg = Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for d in dirs:
            samples, info = process_episode(d, renderer, args, teacher_prompts, (bare_map, bare_default))
            for smp in samples or []:
                smp["weight"] = weights.get(smp["episode"], 1.0)
            if samples is None:
                print(f"SKIP {d}: {info}", file=sys.stderr)
                continue
            if args.require_success and not info["success"]:
                print(f"SKIP {info['episode']}: not successful", file=sys.stderr)
                all_info.append({**info, "used": False})
                continue
            for s in samples:
                out.write(json.dumps(s) + "\n")
            n_samples += len(samples)
            agg["episodes"] += 1
            agg["windows"] += len(samples)
            agg["loss_tokens"] += sum(s["n_loss_tokens"] for s in samples)
            agg["total_tokens"] += sum(s["n_tokens"] for s in samples)
            agg["threads"] += len(info["threads"])
            agg["skipped_blocks"] += len(info["skipped_blocks"])
            all_info.append({**info, "used": True})
            print(
                f"{info['episode']}: cond={info['condition']} success={info['success']} threads={len(info['threads'])} "
                f"windows={len(samples)} loss_tokens={sum(s['n_loss_tokens'] for s in samples)} "
                f"skipped_blocks={len(info['skipped_blocks'])}",
                file=sys.stderr,
            )
    agg["empty_reasoning_turns"] = renderer.n_empty_reasoning
    print(f"TOTAL: {dict(agg)} -> {args.out}", file=sys.stderr)
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "aggregate": dict(agg), "episodes": all_info}, f, indent=1)
    if args.strict:
        bad = []
        for e in all_info:
            if not e.get("used"):
                continue
            errs = [t for t in e["threads"] if "error" in t]
            no_main = not any(t.get("is_main") for t in e["threads"])
            if errs or e["skipped_blocks"] or no_main:
                bad.append({
                    "episode": e["episode"],
                    "thread_errors": [t.get("error") for t in errs],
                    "skipped": e["skipped_blocks"],
                    "no_main_thread": no_main,
                })
        if bad:
            print("STRICT AUDIT FAILED:\n" + json.dumps(bad, indent=1), file=sys.stderr)
            sys.exit(1)
        print("strict audit passed: 0 thread errors, 0 skipped blocks, every episode has a matched main thread", file=sys.stderr)


if __name__ == "__main__":
    main()
