# RHI workflow search and teacher trajectory collection stages.
import sys

from . import process
from .config import query_name, run_root, workflow_prompt
from .episodes import episode
from .io import ROOT, read_json, write_json


def search(c: dict, tasks: list[int]) -> None:
    base = run_root(c)
    last = c["search"]["iterations"]
    for q in tasks:
        episode(
            c, ROOT / f"tasks/bare/query{q}.txt", base / f"reference/query{q}", base / f"logs/reference/query{q}", c["search"]["seed_base"] + q, f"t{q}_bare"
        )
        output = base / f"search/{query_name(q)}"
        for k in range(last + 1):
            workspace = base / f"workspaces/v{k}/{query_name(q)}"
            logs = base / f"logs/search/v{k}/{query_name(q)}"
            episode(c, workflow_prompt(c, q, k), workspace, logs, c["search"]["seed_base"] + q + k * 1000, f"t{q}_v{k}")
            if k < last and workflow_prompt(c, q, k + 1).exists():
                continue
            if k == last and (output / "retained_workflow.json").exists():
                continue
            cmd = [
                sys.executable,
                "-m",
                "harness_improvement.iterate_multi_agent_prompt",
                "--query-file",
                workflow_prompt(c, q, k),
                "--current-version",
                k,
                "--current-repo",
                workspace,
                "--runs-root",
                workspace.parent,
                "--current-run-status-file",
                logs / "run_status.json",
                "--out-dir",
                output,
                "--include-best-design",
                "--history-design-diffs",
                "--include-v0-design",
                "--champion-repo-root-pattern",
                str(base / "workspaces/v{version}"),
                "--champion-repo-v0-root",
                base / "workspaces/v0",
                "--llm-backend",
                "local-vllm",
                "--require-local-vllm",
                "--model",
                c["model_id"],
                "--base-url",
                c["endpoint"],
                "--temperature",
                "0.6",
                "--top-p",
                "0.95",
                "--top-k",
                "20",
                "--max-output-tokens",
                "49152",
                "--local-thinking",
                "--seed",
                c["search"]["seed_base"] + q,
                "--context-window-tokens",
                "262144",
                "--context-safety-tokens",
                "8192",
                "--results-json-max-chars",
                "24000",
                "--results-json-total-max-chars",
                "120000",
            ]
            if k > 0:
                cmd += ["--previous-repo", base / f"workspaces/v{k - 1}/{query_name(q)}", "--judge-vs-baseline", "--baseline-repo-root", base / "reference"]
            if k == last:
                cmd += ["--evaluate-only"]
            process.run(cmd)


def collect(c: dict, tasks: list[int]) -> None:
    base = run_root(c)
    manifest_path = base / "pool.json"
    rows = {r["episode"]: r for r in read_json(manifest_path)} if manifest_path.exists() else {}
    for q in tasks:
        if q not in c["tasks"]["train"]:
            raise ValueError(f"Task {q} is excluded from SFT collection")
        selected = read_json(base / f"search/{query_name(q)}/retained_workflow.json")
        if not selected["beats_reference"]:
            print(f"Task {q}: no retained workflow beats the reference; excluded from collection.")
            continue
        teacher = base / f"teacher_prompts/query{q}.txt"
        teacher.parent.mkdir(parents=True, exist_ok=True)
        text = (
            workflow_prompt(c, q, selected["version"]).read_text(encoding="utf-8").rstrip()
            + "\n\n"
            + (ROOT / "tasks/teacher_directive.txt").read_text(encoding="utf-8")
        )
        if teacher.exists() and teacher.read_text(encoding="utf-8") != text:
            raise ValueError(f"Teacher prompt changed: {teacher}")
        teacher.write_text(text, encoding="utf-8")
        for i in range(c["collection"]["candidates"]):
            seed = c["collection"]["seed_base"] + 100 * q + i
            condition = f"t{q}_L{c['generation']}"
            name = f"{condition}_s{seed}"
            workspace, logs = base / f"pool/workspaces/{name}", base / f"pool/runlogs/{name}"
            episode(c, teacher, workspace, logs, seed, condition)
            rows[name] = {
                "episode": name,
                "task": q,
                "condition": condition,
                "seed": seed,
                "workspace": str(workspace),
                "logs_dir": str(logs),
                "teacher_prompt": str(teacher),
            }
            write_json(manifest_path, sorted(rows.values(), key=lambda r: r["episode"]))
