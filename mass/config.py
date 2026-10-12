# Config validation, run-dir location/freezing, and per-task path naming.
from pathlib import Path
from urllib.parse import urlparse

from .io import ROOT, package_path, read_json, write_json


def validate(c: dict) -> dict:
    tasks = c["tasks"]
    train, test, excluded = map(set, (tasks["train"], tasks["test"], tasks["excluded"]))
    if train & test or train & excluded or test & excluded:
        raise ValueError("Train, test, and excluded tasks must be disjoint")
    if train | excluded != set(tasks["train_candidates"]):
        raise ValueError("Training candidates must equal train + excluded tasks")
    if len(train) != len(tasks["train"]) or len(test) != len(tasks["test"]):
        raise ValueError("Duplicate task IDs")
    p = urlparse(c["endpoint"])
    if p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost") or p.path != "/v1" or not p.port:
        raise ValueError("Episode runner requires http://127.0.0.1:PORT/v1")
    if p.username or p.password or p.query or p.fragment or p.port == c["proxy_port"]:
        raise ValueError("Endpoint must have no credentials/query and must differ from the proxy port")
    if c["search"]["iterations"] < 1:
        raise ValueError("At least one RHI update is required")
    if c["collection"]["validation_rank"] != c["collection"]["train_ranks"] + 1:
        raise ValueError("Validation must immediately follow the selected training ranks")
    if c["collection"]["candidates"] < c["collection"]["validation_rank"]:
        raise ValueError("Too few collection candidates")
    if c["training"]["replicas"] * c["training"]["grad_accum"] != 3:
        raise ValueError("The paper recipe requires three windows per optimizer step")
    for q in train | test | excluded:
        for path in [ROOT / f"tasks/bare/query{q}.txt", ROOT / f"tasks/initial/query{q}_ourTeam.txt"]:
            if not path.is_file():
                raise FileNotFoundError(path)
    return c


def run_root(c: dict) -> Path:
    return package_path(c["run_dir"])


def freeze_config(c: dict) -> None:
    path = run_root(c) / "config.json"
    if path.exists() and read_json(path) != c:
        raise ValueError("Configuration differs from this run's saved config. Use a new run_dir.")
    if not path.exists():
        write_json(path, c)


def query_name(q: int) -> str:
    return f"query{q}_ourTeam"


def workflow_prompt(c: dict, q: int, version: int) -> Path:
    if version == 0:
        return ROOT / f"tasks/initial/{query_name(q)}.txt"
    return run_root(c) / f"search/{query_name(q)}/updated_queries/{query_name(q)}-v{version}.txt"
