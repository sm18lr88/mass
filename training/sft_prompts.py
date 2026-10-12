"""Torch-free matching and swapping of teacher/bare task prompts in user messages (incl. qwen-code normalizations), plus thread role and task parsing."""

import copy
import hashlib
import json
import os
import re
from typing import Any

SENT: dict[str, str] = {}  # md5(original prompt text) -> qwen-code sent form (loaded by load_sent_index)

UserContent = str | list[Any]


def load_sent_index(index_path: str) -> int:
    """Load sent/INDEX.json: {md5(original): {"sent_file": ...}} with the sent texts next to it."""
    d = os.path.dirname(index_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    for h, v in index.items():
        with open(os.path.join(d, v["sent_file"]), encoding="utf-8") as f:
            SENT[h] = f.read()
    return len(SENT)


def sent_form(prompt: str) -> str | None:
    return SENT.get(hashlib.md5(prompt.encode("utf-8")).hexdigest())


def task_of_cond(cond: str | None) -> int | None:
    m = re.match(r"t(\d+)_", cond or "")
    return int(m.group(1)) if m else None


def user_text_items(content: UserContent) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [it.get("text", "") for it in content if isinstance(it, dict)]


def qwen_metric_at_prompt(prompt: str) -> str:
    """qwen-code 0.20.0's atCommandProcessor rewrites unresolved @-paths ('precision@k' -> 'precision @k') and trims the query.
    Only the two forms present in the frozen prompts are handled (task 202); anything else must match exactly."""
    if "precision@k" not in prompt and "recall@k" not in prompt:
        return prompt
    return prompt.replace("precision@k", "precision @k").replace("recall@k", "recall @k").strip()


def prompt_variants(prompt: str) -> list[tuple[str, str]]:
    out = [(prompt, "exact")]
    sf = sent_form(prompt)
    if sf and sf != prompt:
        out.append((sf, "qwen_sent"))
    for cand, mode in (
        (prompt.strip(), "strip"),
        (qwen_metric_at_prompt(prompt), "qwen_metric_at"),
        (qwen_metric_at_prompt(prompt).strip(), "qwen_metric_at_strip"),
    ):
        if cand and all(cand != c for c, _ in out):
            out.append((cand, mode))
    return out


def prompt_match(user_content: UserContent, prompt: str | None) -> tuple[str, str] | None:
    """Return (matched_text, mode) if exactly one variant of `prompt` occurs in the user message, else None."""
    if not prompt:
        return None
    texts = user_text_items(user_content)
    for cand, mode in prompt_variants(prompt):
        hits = sum(t.count(cand) for t in texts)
        if hits == 1:
            return cand, mode
        if hits > 1:
            raise ValueError(f"ambiguous prompt match: {hits} occurrences ({mode})")
    return None


def substitute_prompt(user_content: UserContent, teacher_prompt: str, bare_prompt: str) -> UserContent:
    """Replace the (possibly qwen-normalized) teacher prompt with the bare prompt, normalized the way qwen-code will present it."""
    hit = prompt_match(user_content, teacher_prompt)
    if hit is None:
        raise ValueError("teacher prompt not found in user message")
    matched, mode = hit
    if mode == "qwen_sent":
        bare = sent_form(bare_prompt) or bare_prompt  # what qwen-code presents for the bare prompt (identical unless it has @-tokens)
    else:
        bare = qwen_metric_at_prompt(bare_prompt) if mode.startswith("qwen_metric_at") else bare_prompt
    if isinstance(user_content, str):
        return user_content.replace(matched, bare, 1)
    out = copy.deepcopy(user_content)
    hits = 0
    for it in out:
        if isinstance(it, dict) and it.get("type") == "text" and matched in it.get("text", ""):
            it["text"] = it["text"].replace(matched, bare, 1)
            hits += 1
    if hits != 1:
        raise ValueError(f"expected exactly one prompt item, found {hits}")
    return out


def thread_role(system_content: UserContent | None, is_main: bool) -> str:
    s = system_content if isinstance(system_content, str) else " ".join(x.get("text", "") for x in (system_content or []) if isinstance(x, dict))
    if is_main or s.startswith("You are Qwen Code"):
        return "orchestrator"
    if s.startswith("You are a general-purpose subagent"):
        return "worker"
    return "auxiliary"
