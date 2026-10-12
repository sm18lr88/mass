# Loads SFT windows, splits validation episodes, and draws deterministic task/role-balanced samples (torch-free).
"""Window loading and seeded samplers used by train_sft_lora.py."""
import argparse
import json
import random
import re
from collections import defaultdict
from typing import Any

from training.sft_windows import IGNORE

Row = dict[str, Any]


def role_group(r: Row) -> str:
    role = r.get("role")
    if role is None:
        role = "orchestrator" if r.get("is_main") else "worker"
    return "orch" if role == "orchestrator" else "nonmain"


def task_of(r: Row) -> int:
    if r.get("task") is not None:
        return int(r["task"])
    m = re.match(r"t(\d+)_", r.get("condition", ""))
    return int(m.group(1)) if m else -1


def load_windows(path: str, args: argparse.Namespace) -> list[Row]:
    """Read windows JSONL; uses args.exclude_condition and args.max_len; drops rows without loss tokens."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in args.exclude_condition:
                continue
            if args.max_len:  # debug crop: keep the TAIL so label tokens survive
                r["input_ids"] = r["input_ids"][-args.max_len:]
                r["labels"] = r["labels"][-args.max_len:]
            if not any(l != IGNORE for l in r["labels"][1:]):
                continue
            r["_task"] = task_of(r); r["_rg"] = role_group(r)
            rows.append(r)
    return rows


def split_val(rows: list[Row], regex: str | None) -> tuple[list[Row], list[Row]]:
    if not regex:
        return rows, []
    pat = re.compile(regex)
    tr = [r for r in rows if not pat.search(r["episode"])]
    va = [r for r in rows if pat.search(r["episode"])]
    return tr, va


class WeightedSampler:
    """draw condition ~ weights, then uniform window within the condition."""

    def __init__(self, rows: list[Row], cond_weights: dict[str, float], seed: int) -> None:
        self.by_cond = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_cond[r["condition"]].append(i)
        self.conds = sorted(self.by_cond)
        w = [cond_weights.get(c, 1.0) for c in self.conds]
        s = sum(w)
        self.probs = [x / s for x in w]
        self.seed = seed

    def describe(self) -> str:
        return f"condition-uniform over {len(self.conds)} conditions"

    def draw(self, rank: int, step: int) -> int:
        rng = random.Random(f"{self.seed}-{rank}-{step}")
        c = rng.choices(self.conds, weights=self.probs, k=1)[0]
        return rng.choice(self.by_cond[c])


class TaskRoleSampler:
    """draw a task uniformly, then a role (orchestrator with p_orch when both exist), then a window uniformly within (task, role)."""

    def __init__(self, rows: list[Row], p_orch: float, seed: int) -> None:
        self.by = defaultdict(lambda: {"orch": [], "nonmain": []})
        for i, r in enumerate(rows):
            self.by[r["_task"]][r["_rg"]].append(i)
        self.tasks = sorted(self.by)
        self.p_orch = p_orch
        self.seed = seed

    def describe(self) -> str:
        return "task-uniform over %s; role P(orch)=%.4f; per task orch/nonmain windows: %s" % (
            self.tasks, self.p_orch, {t: (len(self.by[t]["orch"]), len(self.by[t]["nonmain"])) for t in self.tasks})

    def draw(self, rank: int, step: int) -> int:
        rng = random.Random(f"{self.seed}-{rank}-{step}")
        t = rng.choice(self.tasks)
        have = [g for g in ("orch", "nonmain") if self.by[t][g]]
        if len(have) == 2:
            g = "orch" if rng.random() < self.p_orch else "nonmain"
        else:
            g = have[0]
        return rng.choice(self.by[t][g])
