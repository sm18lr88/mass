# Bare-task evaluation rollouts and resumable external-judge reporting.
import hashlib
import json
from pathlib import Path

from .config import run_root
from .episodes import episode
from .io import ROOT, append_record, read_json, records, write_json


def rollout(c: dict, tasks: list[int], count: int) -> None:
    base = run_root(c)
    rows = []
    for q in tasks:
        for i in range(count):
            workspace = base / f"evaluation/workspaces/query{q}_r{i}"
            episode(c, ROOT / f"tasks/bare/query{q}.txt", workspace, base / f"evaluation/logs/query{q}_r{i}", 910000 + q * 100 + i, f"t{q}_bare")
            rows.append({"task": q, "replicate": i, "workspace": str(workspace)})
    write_json(base / "evaluation/rollouts.json", rows)


def report(c: dict, pairs_file: Path) -> None:
    # Looked up at call time: judging pulls in provider SDKs lazily.
    from .judging import external_judgments

    path = run_root(c) / "evaluation/external.jsonl"
    done = {(r["pair_id"], r["provider"], r["seed"]) for r in records(path)}
    for pair in read_json(pairs_file):
        # Hash input paths to keep distinct workspace pairs distinct on resume.
        pair_id = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
        if all((pair_id, p, s) in done for p in ["openai", "anthropic"] for s in [42, 43, 44]):
            continue
        skip = {(provider, seed) for pid, provider, seed in done if pid == pair_id}
        for row in external_judgments(pair["task"], Path(pair["a"]).resolve(), Path(pair["b"]).resolve(), skip=skip):
            key = (pair_id, row["provider"], row["seed"])
            if key not in done:
                append_record(path, {"pair_id": pair_id, "pair": pair, **row})
                done.add(key)
    rows = records(path)
    wins = sum(r["winner"] == "A" for r in rows)
    losses = sum(r["winner"] == "B" for r in rows)
    write_json(
        path.with_name("summary.json"),
        {
            "wins_a": wins,
            "wins_b": losses,
            "ties": len(rows) - wins - losses,
            "win_rate_a": wins / (wins + losses) if wins + losses else None,
            "judgments": len(rows),
        },
    )
