# Within-task self-ranking tournaments and train/validation trajectory selection.
import random
from pathlib import Path

from .config import query_name, run_root
from .episodes import completed
from .io import append_record, read_json, records, write_json
from .ranking import bradley_terry


def tournament_pairs(names, opponents: int, seed: int) -> list[tuple[str, str]]:
    """Connected balanced ring schedule; unique unordered pairs, seeded order."""
    if opponents < 2 or opponents % 2:
        raise ValueError("Use an even opponents_per_episode >= 2")
    names = sorted(names)
    random.Random(seed).shuffle(names)
    pairs = {tuple(sorted((name, names[(i + d) % len(names)]))) for i, name in enumerate(names) for d in range(1, min(opponents // 2, len(names) - 1) + 1)}
    pairs = sorted(pairs)
    rng = random.Random(seed + 1)
    return [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]


def rank(c: dict, tasks: list[int]) -> None:
    # Looked up at call time so tests can patch mass.judging.self_judge.
    from .judging import self_judge

    base = run_root(c)
    pool = read_json(base / "pool.json")
    for q in tasks:
        retained = read_json(base / f"search/{query_name(q)}/retained_workflow.json")
        if not retained["beats_reference"]:
            print(f"Task {q}: no eligible retained workflow; no teacher ranking.")
            continue
        rows = {r["episode"]: r for r in pool if r["task"] == q and completed(Path(r["logs_dir"]))}
        if len(rows) < c["collection"]["validation_rank"]:
            raise ValueError(f"Task {q}: need at least {c['collection']['validation_rank']} successful candidates, found {len(rows)}")
        pairs = tournament_pairs(rows, c["ranking"]["opponents_per_episode"], c["ranking"]["seed"] + q)
        path = base / f"selection/task{q}_matches.jsonl"
        done = {(r["a"], r["b"]): r for r in records(path)}
        for i, (a, b) in enumerate(pairs):
            if (a, b) not in done:
                verdict = self_judge(c, q, Path(rows[a]["workspace"]), Path(rows[b]["workspace"]), c["ranking"]["seed"] + q * 1000 + i)
                done[(a, b)] = {"a": a, "b": b, **verdict}
                append_record(path, done[(a, b)])
        matches = {(a, b): {"self": a if done[(a, b)]["winner"] == "A" else b if done[(a, b)]["winner"] == "B" else "tie"} for a, b in pairs}
        scores = bradley_terry(sorted(rows), matches)
        ordered = sorted(rows, key=lambda name: (-scores[name], name))
        write_json(base / f"selection/task{q}_ranked.json", [{**rows[n], "score": scores[n], "rank": i + 1} for i, n in enumerate(ordered)])


def select(c: dict) -> None:
    base = run_root(c)
    train, val = [], []
    for q in c["tasks"]["train"]:
        retained = read_json(base / f"search/{query_name(q)}/retained_workflow.json")
        if not retained["beats_reference"]:
            continue
        rows = read_json(base / f"selection/task{q}_ranked.json")
        if len(rows) < c["collection"]["validation_rank"]:
            raise ValueError(f"Missing validation rank for task {q}")
        train.extend(rows[: c["collection"]["train_ranks"]])
        val.append(rows[c["collection"]["validation_rank"] - 1])
    if not train or not val:
        raise ValueError("No eligible training/validation trajectories")
    if {r["episode"] for r in train} & {r["episode"] for r in val}:
        raise ValueError("Training and validation episodes overlap")
    write_json(base / "selection/train.json", train)
    write_json(base / "selection/val.json", val)
    print(f"Selected {len(train)} training and {len(val)} validation trajectories.")
