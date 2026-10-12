# benchmarks/

## OVERVIEW
Public-benchmark adapters (ScienceAgentBench 102 tasks, MLR-Bench 12-task subset) that run a served L0/L1/L2 through qwen-code inside Docker via a pinned Harbor checkout. Separate domain from the MASS loop: own venv, own deps, nothing here feeds training. User-facing guide: `README.md` in this directory.

## STRUCTURE
```
benchmarks/
├── run.py                  # one trial: run.py {sab|mlr} {L0|L1|L2} <trial> --model-id <served name> [--dry-run]
├── fetch_sources.py        # clones pinned repos (configs/sources.json) into vendor/, applies patches/
├── prepare_sab_base.py     # SAB base image prep
├── summarize.py            # SAB: mean +/- sample SD of trial means from official evaluator JSONL
├── summarize_mlr.py        # MLR: judge reviews -> per-generation scores
├── configs/                # sources.json (pinned commits), mlr_tasks.txt, compose_host_gateway.yaml; seeds_*.json generated
├── images/                 # qwen.Dockerfile (portable qwen-code bundle), mlr.Dockerfile
├── patches/                # sab_codebert.patch (CodeBERTScore failure must not zero exec scores)
└── scripts/                # Harbor agent, logging proxy, task builders, judge, seed map
```
Generated and gitignored: `vendor/`, `work/` (override with `MASS_BENCH_ROOT`), `runtime/qwen-code/`.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Harbor agent / retries | `scripts/bench_qwen_code_agent.py` `BenchQwenCode` | `BenchInfraApiError` = infra failure, retried by Harbor (`--retry-include`), never a final score |
| Request logging | `scripts/bench_api_proxy.py` | byte-identical copy of `runtime/api_proxy.py`; keep in sync |
| Seeds | `scripts/bench_seeds.py` | `3100000 + bench*10000 + trial*1000 + task_index`; shared across generations so trials are paired |
| SAB tasks / predictions | `scripts/bench_sab_make_tasks.py`, `bench_sab_collect_preds.py` | evaluator paths must be absolute (it changes cwd; tested) |
| MLR tasks / judging | `scripts/bench_mlr_make_tasks.py`, `bench_mlr_judge.py` | imports official `OVERALL_RUBRIC`; judges gpt-5.5 + claude-opus-4-8 |

## CONVENTIONS
- Setup uses `.venv-bench` from the root `pyproject.toml` `bench` group (`uv sync --locked --no-dev --group bench --inexact`), then `uv pip install -e benchmarks/vendor/harbor`; run `fetch_sources.py` first. Scripts import `harbor`, so they are not importable from the core venv (tests load only harbor-free modules by path).
- `run.py` sets `PYTHONPATH=scripts` so Harbor resolves `bench_qwen_code_agent:BenchQwenCode`; it hard-checks 102 SAB / 12 MLR task dirs before launching.
- SAB's own top-level package is also named `evaluation`. `prepare_sab_base.py` runs it in a subprocess with `cwd`/`PYTHONPATH` = `vendor/ScienceAgentBench`; never import SAB evaluator code in a process where the repo root is on `sys.path`.
- Each concurrent Harbor job needs its own proxy port block (`--proxy-base`, 200 ports, range 9100-9899). Proxy binds the Docker bridge (`--proxy-host`, default 172.17.0.1).

## ANTI-PATTERNS
- Never report the Harbor reward for SAB: it only checks that a program file exists. Use the official evaluator JSONL.
- Never pad or fill incomplete trials: `summarize.py` requires all 102 rows per trial; paper aggregates are not test expectations.
- MLR: missing or <200-byte paper scores 0, but a missing judge response is an error, not a zero (tested).
- Do not expose the logging proxy on an untrusted network.
- Do not redistribute the SAB verified benchmark archive; it is a separate upstream download into `work/sab/benchmark/`.
- Record the built MLR image digest: its scientific packages are unpinned, so rebuilds drift.

## COMMANDS
```bash
python benchmarks/run.py sab L0 1 --model-id Qwen/Qwen3.6-27B-FP8 --dry-run   # prints Harbor command only
python -m unittest tests.test_benchmarks -v                                  # offline; no Docker/vLLM/scorers
```
