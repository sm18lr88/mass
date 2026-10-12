# PROJECT KNOWLEDGE BASE

**Generated:** 2026-10-12
**Commit:** fb40358
**Branch:** main

## OVERVIEW

MASS: research code for recursive self-improvement. RHI (workflow search with frozen weights) -> teacher trajectory collection -> self-ranking -> LoRA SFT -> next model generation (L0 -> L1 -> L2). Python 3.14 default (3.12+ supported), stdlib-heavy driver; Qwen3.6-27B served by vLLM, qwen-code 0.20.0 as agent runtime.

## STRUCTURE

```
mass/
├── mass/                   # stage driver: python -m mass <stage>; pipeline.py = CLI + STAGES table; config/episodes/collection/selection/sft/reporting modules; process.run = the one subprocess seam
├── harness_improvement/    # RHI optimizer; iterate_multi_agent_prompt.py is a thin entry over rhi_*.py, invoked as a subprocess
├── runtime/                # run_episode.sh (one qwen-code episode) + api_proxy.py (request logger for SFT)
├── training/               # prepare_sft_data (render+mask), train_sft_lora (FSDP2/pp_ddp), merge_lora, FP8 quantize
├── evaluation/             # workspace evidence bundling for judges
├── evaluation_claudecodex/ # pairwise judge prompts/schema (OpenAI, Anthropic, local vLLM)
├── benchmarks/             # ScienceAgentBench + MLR-Bench adapters (Docker/Harbor); own venv, see benchmarks/AGENTS.md
├── tasks/                  # 12 task prompts: bare/query<ID>.txt, initial/query<ID>_ourTeam.txt, splits.json
├── configs/                # paper.json (cycle 1), cycle2.json, models.json (pinned HF revisions)
├── tools/                  # check_source, make_source_archive, fetch_models, pair_rollouts
└── SOURCE_MANIFEST.json    # reviewed release file list + SHA-256; CI-checked
```

## WHERE TO LOOK

| Task                               | Location                                                                  | Notes                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Add/change a pipeline stage        | `mass/pipeline.py` `STAGES` + the stage module                            | handlers resolve stage functions via their module so tests can patch them      |
| Config invariants                  | `mass/config.py` `validate`                                               | split disjointness, loopback endpoint, 3 windows/step                          |
| RHI proposal/judging logic         | `harness_improvement/rhi_*.py` (prompts in `rhi_prompt.py`)               | prompt strings are protocol; keep byte-identical                               |
| Dependency direction / conventions | `tests/test_architecture.py`                                              | layer rule, lazy heavy imports in mass/, responsibility comments, proxy copies |
| Local/remote LLM JSON calls        | `harness_improvement/llm_backend.py` `JsonLLMBackend`                     | used by RHI and `mass.judging.self_judge`                                      |
| Self vs external judging           | `mass/judging.py`                                                         | self = current model; external = gpt-5.5 + claude-opus-4-8, seeds 42-44        |
| Ranking                            | `mass/ranking.py` `bradley_terry`, `mass/selection.py` `tournament_pairs` |                                                                                |
| Loss masks / window rendering      | `training/sft_windows.py` `Renderer`, `make_windows` (torch-free)         | orchestrator prompt swapped for bare task                                      |
| Paper notation -> code             | `docs/paper_to_code.md`                                                   |                                                                                |
| Known deviations from paper        | `docs/reproduction.md`, `docs/experiment_coverage.md`                     |                                                                                |

## CODE MAP

| Symbol                               | Type     | Location                                  | Role                                                                    |
| ------------------------------------ | -------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| `main` / `STAGES`                    | fn       | mass/pipeline.py                          | CLI: plan/search/collect/rank/select/render/train/export/rollout/report |
| `validate` / `freeze_config`         | fn       | mass/config.py                            | config checks; run_dir config is frozen on first stage                  |
| `episode` / `completed`              | fn       | mass/episodes.py                          | runs `runtime/run_episode.sh`; reuse only if exit 0 + success trace     |
| `run`                                | fn       | mass/process.py                           | every external command; call as `process.run(...)`                      |
| `search` / `collect`                 | fn       | mass/collection.py                        | bare reference + v0..vN workflows, shells out to RHI                    |
| `best_checkpoint`                    | fn       | mass/sft.py                               | min finite val_loss among saved `step_NNNNN` dirs                       |
| `ROOT`, `package_path`, `write_json` | fn       | mass/io.py                                | repo-relative paths; atomic JSON writes                                 |
| `JsonLLMBackend`                     | class    | harness_improvement/llm_backend.py        | JSON-mode LLM client                                                    |
| `build_judge_bundle`                 | fn       | evaluation/bundle.py                      | workspace -> judge evidence                                             |
| `PairwiseJudgeResult`                | pydantic | evaluation_claudecodex/pairwise_schema.py | judge verdict schema                                                    |

## CONVENTIONS

- Generations are named `L0`/`L1`/`L2` everywhere (configs, run dirs, model dirs). Keep the paper notation.
- Paths in configs are repo-relative, resolved through `package_path`; commands run from the repo root, no editable install.
- `mass/io.py` must stay free of model/accelerator imports; heavy deps are imported lazily inside functions (`judging.py`, `rank`, `report`).
- Stages are resumable and idempotent: completed episodes/comparisons are reused, JSONL records appended, never rewritten.
- Task IDs: train 51 53 56 202 205 302 307 324; excluded 221 (train candidate, not SFT); test 60 207 305. `validate` enforces disjointness.
- Episode seeds are derived from config `seed_base` + task/version/candidate; do not reorder loops that compute them.

## ANTI-PATTERNS (THIS PROJECT)

- Never reuse a `run_dir` after changing its config; `freeze_config` raises. Make a new `configs/local-*.json` (gitignored) and new `run_dir`.
- Never overwrite or delete partial episode dirs to retry; `run_episode.sh` refuses (exit 73). Archive explicitly or use a new run_dir.
- Never let test tasks (60, 207, 305) reach collect/select/render or checkpoint selection.
- Never put API keys in source; `.env` is not auto-loaded. `tools/check_source.py` flags emails, `/home|/Users/<name>` paths, `sk-`/`ghp_` tokens in manifest files.
- Tests must not need GPUs, model weights, network, or API keys; mock backends/fixtures only.
- Do not silently drop selected episodes during rendering; `render` asserts every selected episode yields windows.
- `train` refuses when `training/train_log.jsonl` exists; resume only via `train_sft_lora.py --resume`. `export` refuses populated model dirs.

## UNIQUE STYLES

- Terse functional style in `mass/`: short functions, config dict `c`, task id `q`, no classes.
- `runtime/api_proxy.py` and `benchmarks/scripts/bench_api_proxy.py` are byte-identical copies; edit both.
- `run_episode.sh` qwen-code settings are byte-identical to the original experiment episodes; changing them changes the protocol.

## COMMANDS

```bash
# Offline checks (what CI runs, Python 3.12 + 3.13 + 3.14); deps in pyproject.toml, locked in uv.lock
uv sync --locked
uv run ruff check
uv run python -m unittest discover -s tests -v
uv run python -m mass plan --config configs/paper.json
uv run python -m mass plan --config configs/cycle2.json
uv run python tools/check_source.py

# Changing a pin: edit pyproject.toml, `uv lock`, commit both. Groups: dev (default), train, serve, bench.

# Real runs (Linux, GPUs, vLLM on 127.0.0.1:8001): see docs/running.md
python -m mass search --config configs/paper.json --tasks 51
```

## NOTES

- After editing any file listed in `SOURCE_MANIFEST.json`, refresh its hash (snippet in `docs/development.md`) or CI's `check_source.py` fails. Only list files intended for release.
- Episodes need Linux + bash, `jq`, `curl`, `qwen` on PATH; qwen-code runs with `--approval-mode yolo` on the host with no sandbox.
- Training and serving need separate venvs (`.venv-train`, `.venv-serve`): vLLM pins an older torch than the train group. openai>=3 and anthropic>=1 run on `httpx2` (pass `httpx2` objects, never `httpx`). `pyproject.toml` declares the `train` and `serve` groups as conflicting; create each with `UV_PROJECT_ENVIRONMENT=<venv> uv sync --locked ...` (docs/running.md).
- Windows: `uv sync --locked --group train` gives CUDA torch (pytorch cu132 index) and `triton-windows`, so `train_sft_lora.py --parallel pp` runs natively with the fla kernels; `python -m mass train` (NCCL pp_ddp) refuses on Windows. vLLM, NCCL training, and `run_episode.sh` run in WSL against the same checkout; pass absolute paths to `run_episode.sh`. vLLM needs a C compiler (Triton) and `VLLM_USE_FLASHINFER_SAMPLER=0` without a CUDA toolkit (docs/running.md). The GPU may be shared with other jobs: size `--gpu-memory-utilization` to free VRAM.
- `rollout` overwrites `evaluation/rollouts.json`; `report` aggregates every judgment in the run dir. One run dir per evaluation pool.
- RHI here is fixed-reference + champion arbitration, not the paper's Algorithm 1; document protocol changes in `docs/reproduction.md`.
