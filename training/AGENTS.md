# training/

## OVERVIEW
SFT data rendering, LoRA training, merge, and FP8 export. Own file: separate venv (`.venv-train`), platform-specific parallel modes, and loss-mask rules that define what the next generation learns (score 8, distinct domain).

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Loss masks / chat rendering | `sft_windows.py` `Renderer` | torch-free |
| Window packing | `sft_windows.py` `make_windows` | overlap context labelled `IGNORE` |
| Proxy logs -> threads | `sft_requests.py` | `load_requests`, `group_threads`, `vllm_postprocess` |
| Teacher -> bare prompt swap | `sft_prompts.py` | `prompt_match`, `substitute_prompt`, `thread_role` |
| Train sampling | `sft_sampling.py` | `TaskRoleSampler` (default), `WeightedSampler`, `split_val` |
| Render CLI | `prepare_sft_data.py` | `transformers` imported lazily in `main` |
| Training | `train_sft_lora.py` (504 LOC; `main` ~265 lines) | `--parallel` fsdp2 (default) / pp / pp_ddp |
| Merge | `merge_lora.py` | CPU, shard-wise `W += (alpha/r) B@A`, safetensors only |
| FP8 | `quantize_fp8_blockwise.py` | 128x128 e4m3 blocks; quantize iff reference tensor has `_scale_inv` |

Callers: only `mass/sft.py`, by path via `process.run` (render, train via `torch.distributed.run`, export). `mass` never imports `training`; `training` is an independent layer (imports only itself).

## CONVENTIONS
- Torch-free split: `sft_windows`, `sft_requests`, `sft_prompts`, `sft_sampling` must not import torch/transformers (tests import them in the core venv). Only `train_sft_lora.py` and `quantize_fp8_blockwise.py` import torch at top level.
- Scripts run by path and append the repo root to `sys.path` so helpers resolve as `training.<module>` (one module object).
- `train_sft_lora.py` slices `CUDA_VISIBLE_DEVICES` per rank BEFORE importing torch when `pp_ddp` is in argv (`--gpus-per-rank`, default 2). Keep that block above the torch import.
- Parallel modes: `fsdp2` and `pp_ddp` need NCCL (Linux/WSL); `pp` is single-process `device_map="auto"` and runs natively on Windows.
- Checkpoints: `<out>/step_NNNNN/` = `lora_state_dict.safetensors` (fp32), `adapter/`, `optimizer.pt`, `train_state.json`. Run files: `args.json`, `train_log.jsonl`, `exposure_log.jsonl`, `resume_log.jsonl`.
- `--resume DIR` restores LoRA, step, and optimizer (prints `RESET` if `optimizer.pt` is missing) and ignores `--init-lora`; start-of-run validation only when `start_step == 0`.
- Loss: only assistant blocks of the current segment; starts after the `<think>` generation prefix (at `</think>` for empty reasoning), ends at the last `<|im_end|>`. Earlier segments render with reasoning stripped; every segment must re-tokenize identically.
- Prompt swap: orchestrator (main) thread only, with `--substitute` (default on); replacement text is `tasks/bare/query<ID>.txt`.
- Sampler draws are seeded `f"{seed}-{rank}-{step}"`: deterministic per rank and step.

## ANTI-PATTERNS
- Never import torch in the torch-free modules; it breaks `tests/test_training.py` and the core-venv `prepare_sft_data.py --help` check.
- `mass/sft.py` always passes `--strict`: thread errors, skipped blocks, or no main thread exit 1. Do not relax it to get render through.
- Adapter export failures are only printed (`train_sft_lora.py:200`); `best_checkpoint` keys on `lora_state_dict.safetensors`, so a step can lack `adapter/`.
- Bare `assert`s guard shapes (`merge_lora.py`, `quantize_fp8_blockwise.py`, `sft_windows.make_windows`); never run these with `python -O`.

## COMMANDS
```bash
uv run python -m unittest tests.test_training tests.test_driver -v
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --locked --group train   # torch scripts run from this venv; see docs/running.md
```
