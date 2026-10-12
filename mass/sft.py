# SFT data rendering, LoRA training launch, checkpoint selection, and merge/FP8 export.
import json
import math
import sys
from pathlib import Path

from . import process
from .config import run_root
from .io import ROOT, package_path, read_json, records, write_json


def render(c: dict) -> None:
    base = run_root(c)
    (base / "data").mkdir(exist_ok=True)
    for split in ("train", "val"):
        manifest = base / f"selection/{split}.json"
        rows = read_json(manifest)
        cmd = [
            sys.executable,
            ROOT / "training/prepare_sft_data.py",
            "--tokenizer",
            package_path(c["parent_bf16"]),
            "--episode-list",
            manifest,
            "--window",
            c["training"]["window"],
            "--context-overlap",
            c["training"]["context_overlap"],
            "--require-success",
            "--strict",
            "--out",
            base / f"data/{split}.jsonl",
            "--stats",
            base / f"data/{split}_stats.json",
        ]
        for condition, row in {r["condition"]: r for r in rows}.items():
            cmd += ["--bare-prompt-map", f"{condition}={ROOT}/tasks/bare/query{row['task']}.txt", "--teacher-prompt", f"{condition}={row['teacher_prompt']}"]
        process.run(cmd)
        # Prevent the renderer's logged skips from silently shrinking a split.
        found = set()
        with (base / f"data/{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                found.add(json.loads(line)["episode"])
        if found != {r["episode"] for r in rows}:
            raise ValueError(f"{split}: selected episodes did not all yield windows; inspect rendering stats")


def training_command(c: dict) -> list:
    t, base = c["training"], run_root(c)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={t['replicas']}",
        ROOT / "training/train_sft_lora.py",
        "--parallel",
        "pp_ddp",
        "--gpus-per-rank",
        t["gpus_per_rank"],
        "--model",
        package_path(c["parent_bf16"]),
        "--data",
        base / "data/train.jsonl",
        "--val-data",
        base / "data/val.jsonl",
        "--out",
        base / "training",
        "--sampler",
        "task_role",
        "--p-orch",
        str(1 / 3),
        "--val-at-start",
        "--autocast-bf16",
    ]
    for key in [
        "steps",
        "lr",
        "warmup_steps",
        "min_lr_ratio",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "save_every",
        "val_every",
        "grad_accum",
        "ce_chunk",
        "seed",
    ]:
        cmd += ["--" + key.replace("_", "-"), t[key]]
    return cmd


def train(c: dict) -> None:
    if sys.platform == "win32":
        raise RuntimeError(
            "The train stage launches NCCL pp_ddp ranks, which need Linux (use WSL on Windows). "
            "For single-GPU training on Windows run training/train_sft_lora.py --parallel pp directly."
        )
    if (run_root(c) / "training/train_log.jsonl").exists():
        raise FileExistsError("Training log exists; use train_sft_lora.py --resume explicitly")
    process.run(training_command(c))


def best_checkpoint(directory: Path) -> dict:
    candidates = []
    for row in records(directory / "train_log.jsonl"):
        if "val_loss" not in row:
            continue
        if not math.isfinite(row["val_loss"]) or row.get("val_tokens", 0) <= 0:
            raise ValueError("Invalid validation loss or zero validation tokens")
        step = row["step"]
        if isinstance(step, int) and step > 0:
            checkpoint = directory / f"step_{step:05d}"
            if (checkpoint / "lora_state_dict.safetensors").is_file():
                candidates.append((row["val_loss"], step, checkpoint))
    if not candidates:
        raise ValueError("No saved checkpoint has a finite validation loss")
    loss, step, path = min(candidates)
    return {"step": step, "val_loss": loss, "checkpoint": str(path), "criterion": "minimum validation loss"}


def export(c: dict) -> None:
    base = run_root(c)
    selected = best_checkpoint(base / "training")
    write_json(base / "selected_checkpoint.json", selected)
    for key in ("next_bf16", "next_fp8"):
        p = package_path(c[key])
        if p.exists() and any(p.iterdir()):
            raise FileExistsError(f"Refusing to overwrite model output: {p}")
    process.run([
        sys.executable,
        ROOT / "training/merge_lora.py",
        "--base",
        package_path(c["parent_bf16"]),
        "--lora",
        Path(selected["checkpoint"]) / "lora_state_dict.safetensors",
        "--r",
        c["training"]["lora_r"],
        "--alpha",
        c["training"]["lora_alpha"],
        "--out",
        package_path(c["next_bf16"]),
    ])
    process.run([
        sys.executable,
        ROOT / "training/quantize_fp8_blockwise.py",
        "--src",
        package_path(c["next_bf16"]),
        "--ref",
        package_path(c["fp8_reference"]),
        "--out",
        package_path(c["next_fp8"]),
    ])
