# One qwen-code episode via runtime/run_episode.sh, reused only when it completed successfully.
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from . import process
from .io import ROOT, read_json


def completed(logs: Path) -> bool:
    p = logs / "run_status.json"
    if not p.exists():
        return False
    st = read_json(p)
    return st.get("exit_code") == 0 and st.get("trace_has_success_result") is True


def episode(c: dict, prompt: Path, workspace: Path, logs: Path, seed: int, condition: str) -> None:
    if completed(logs):
        if (logs / "prompt.txt").read_text(encoding="utf-8") != prompt.read_text(encoding="utf-8"):
            raise ValueError(f"Completed episode has a different prompt: {logs}")
        return
    # The shell runner refuses to overwrite partial/failed episodes. Keep the
    # evidence and use a new run directory after correcting an infrastructure failure.
    env = dict(os.environ, MODEL_ID=c["model_id"], PROXY_PORT=str(c["proxy_port"]),
               PROXY_PY=sys.executable, HIW_CONDITION=condition)
    process.run(["bash", ROOT / "runtime/run_episode.sh", prompt, workspace, logs,
                 str(urlparse(c["endpoint"]).port), str(seed)], env)
