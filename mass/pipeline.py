# Stage CLI: argument parsing, task-set resolution, and the explicit stage dispatch table.
"""Portable orchestration of RHI -> post-training -> the next MASS cycle."""
import argparse
from collections.abc import Callable
import json
from pathlib import Path
import shlex

from . import collection, reporting, selection, sft
from .config import freeze_config, query_name, run_root, validate, workflow_prompt
from .episodes import completed, episode
from .io import ROOT, read_json
from .selection import tournament_pairs
from .sft import best_checkpoint, training_command

# Re-exported stage functions keep `pipeline.<stage>` call sites working.
from .collection import collect, search
from .reporting import report, rollout
from .selection import rank, select
from .sft import export, render, train

__all__ = ["STAGES", "best_checkpoint", "collect", "completed", "episode", "export", "freeze_config",
           "main", "query_name", "rank", "render", "report", "rollout", "run_root", "search", "select",
           "tournament_pairs", "train", "training_command", "validate", "workflow_prompt"]


def plan(c: dict) -> None:
    print(json.dumps(c, indent=2))
    print(f"L^({c['generation']}) -> RHI -> teacher collection -> SFT -> L^({c['generation'] + 1})")
    print("Training command (not executed): " + shlex.join(list(map(str, training_command(c)))))


def _rollout(c: dict, tasks: list[int], args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.count < 1:
        parser.error("--count must be positive")
    reporting.rollout(c, tasks, args.count)


def _report(c: dict, tasks: list[int], args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.pairs:
        parser.error("report requires --pairs")
    reporting.report(c, args.pairs)


# Stage name -> handler(c, tasks, args, parser). Handlers resolve stage functions
# through their modules at call time so tests can patch e.g. mass.collection.search.
Handler = Callable[[dict, list[int], argparse.Namespace, argparse.ArgumentParser], None]
STAGES: dict[str, Handler] = {
    "plan": lambda c, tasks, args, parser: plan(c),
    "search": lambda c, tasks, args, parser: collection.search(c, tasks),
    "collect": lambda c, tasks, args, parser: collection.collect(c, tasks),
    "rank": lambda c, tasks, args, parser: selection.rank(c, tasks),
    "select": lambda c, tasks, args, parser: selection.select(c),
    "render": lambda c, tasks, args, parser: sft.render(c),
    "train": lambda c, tasks, args, parser: sft.train(c),
    "export": lambda c, tasks, args, parser: sft.export(c),
    "rollout": _rollout,
    "report": _report,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--config", default=str(ROOT / "configs/paper.json"))
    parser.add_argument("--tasks", type=int, nargs="+", help="Subset for search, collect, rank, or rollout")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--count", type=int, default=10, help="Bare-task evaluation rollouts per task")
    parser.add_argument("--pairs", type=Path, help="External reporting JSON list of {task, a, b}; a and b are workspace paths")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    c = validate(read_json(args.config))
    if args.stage == "plan":
        # Planning is read-only: it neither freezes the config nor resolves tasks.
        STAGES["plan"](c, [], args, parser)
        return
    freeze_config(c)
    tasks = args.tasks or (c["tasks"]["train"] if args.stage in ("collect", "rank") else
                          c["tasks"][args.split] if args.split != "all" else
                          c["tasks"]["train_candidates"] + c["tasks"]["test"])
    if not set(tasks) <= set(c["tasks"]["train_candidates"] + c["tasks"]["test"]):
        parser.error("Unknown task ID")
    STAGES[args.stage](c, tasks, args, parser)
