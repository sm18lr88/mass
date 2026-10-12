# MASS: Recursive Self-Improvement through Multi-Agent Self-Supervision

MASS is a method that lets a language model improve itself in cycles. This
repository is a research implementation for running and extending the MASS
pipeline. Read the [paper](https://arxiv.org/abs/2610.12176) for the method
and the full analysis.

## How MASS works

Each MASS cycle has two steps:

1. **Workflow search (RHI).** The model searches for better multi-agent
   workflows. The model weights do not change in this step.
2. **Post-training.** The model trains on the work (trajectories) that the
   selected workflows produce. This step updates the model weights.

The next cycle starts from the updated model:

```text
L0 ── RHI → post-training ──> L1 ── RHI → post-training ──> L2
```

`L0` is the base model. `L1` and `L2` are the models after one and two cycles.
The paper writes these generations as $\mathcal{L}^{(0)}$, $\mathcal{L}^{(1)}$,
and $\mathcal{L}^{(2)}$. The configuration files and output directories use
`L0`, `L1`, and `L2`.

In each cycle, the current model does three jobs: it executes the tasks, it
optimizes the workflows, and it evaluates the results in the loop. The base
model is Qwen3.6-27B. Each generation uses qwen-code 0.20.0 as its
coding-agent runtime.

![MASS overview: improve a team's workflow, learn from its trajectories, and repeat.](assets/mass_teaser_minimal.gif)

## Results

### Summary

After two MASS cycles, the model (`L2`) is better than the base model (`L0`)
at scientific coding and at long research tasks. Its scores on data science,
terminal, and software-engineering benchmarks stay approximately the same or
go down slightly.

**Improvements from `L0` to `L2`:**

- **Scientific coding (ScienceAgentBench):** the model solves 31.7% of tasks,
  up from 28.8%. This is approximately 3 more solved tasks out of every 100.
- **Writing machine-learning research papers (MLR-Bench):** the model delivers
  a paper for 76% of tasks, up from 57%. The paper score goes up from 1.71 to
  2.25.
- **End-to-end research reports (AstaBench, E2E-Bench-Hard):** the model
  delivers a report for 76% of tasks, up from 50%. The rubric score goes up
  from 0.053 to 0.067. But the mean score of the delivered reports goes down
  from 0.105 to 0.088 (see the
  [AstaBench breakdown](benchmarks/README.md#astabench-e2e-bench-hard-results)).

**No improvement from `L0` to `L2`:**

| Benchmark (what it measures)                  | L0            | L1            | L2            | Change, L0 to L2 |
| --------------------------------------------- | ------------: | ------------: | ------------: | ---------------: |
| DSBench (data science score)                  | 0.473 ± 0.029 | 0.465 ± 0.011 | 0.482 ± 0.025 |           +0.009 |
| Terminal-Bench 2.0 (terminal tasks passed)    | 37.5% ± 2.2   | 35.3% ± 2.1   | 35.2% ± 1.7   |  -2.3 points     |
| SWE-bench Verified (software bugs fixed)      | 62.6% ± 1.5   | 61.8% ± 1.9   | 61.1% ± 0.6   |  -1.5 points     |

- **DSBench:** the score is approximately the same. The change (+0.009) is
  smaller than the variation between trials (± 0.011 to ± 0.029).
- **Terminal-Bench 2.0:** the pass rate goes down by 2.3 percentage points.
  This decrease is approximately as large as the variation between trials
  (± 1.7 to ± 2.2).
- **SWE-bench Verified:** the pass rate goes down by 1.5 percentage points.
  This decrease is approximately as large as the variation between trials
  (± 0.6 to ± 1.9).

Each model ran every task three times (three trials). A "±" value is the
standard deviation across the three trials. A large "±" value means that the
result changed much from one trial to the next.

### Full results

Task and trial counts are per model. ScienceAgentBench and delivery rates are
percentages. The other scores use the scale of their benchmark.

| Benchmark                                                                                                       | Tasks × trials |            L0 |            L1 |            L2 |
| --------------------------------------------------------------------------------------------------------------- | :------------: | ------------: | ------------: | ------------: |
| [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) (%)                                     |    102 × 3     |    28.8 ± 1.5 |    30.4 ± 2.0 |    31.7 ± 2.0 |
| [MLR-Bench](https://github.com/chchenhui/mlrbench)                                                              |    107 × 3     |   1.71 ± 0.08 |   1.77 ± 0.09 |   2.25 ± 0.06 |
| [MLR-Bench](https://github.com/chchenhui/mlrbench) papers delivered                                             |    107 × 3     |           57% |           59% |           76% |
| [AstaBench (E2E-Bench-Hard)](https://github.com/allenai/asta-bench#discovery-tasks-discovery)                   |     40 × 3     | 0.053 ± 0.002 | 0.055 ± 0.004 | 0.067 ± 0.017 |
| [AstaBench (E2E-Bench-Hard)](https://github.com/allenai/asta-bench#discovery-tasks-discovery) reports delivered |     40 × 3     |           50% |           58% |           76% |
| [DSBench](https://github.com/LiqiangJing/DSBench)                                                               |     74 × 3     | 0.473 ± 0.029 | 0.465 ± 0.011 | 0.482 ± 0.025 |
| [Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench-2)                                      |     89 × 3     | 0.375 ± 0.022 | 0.353 ± 0.021 | 0.352 ± 0.017 |
| [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)                          |    500 × 3     | 0.626 ± 0.015 | 0.618 ± 0.019 | 0.611 ± 0.006 |

Notes on the table:

- Terminal-Bench 2.0 uses the official time limits.
- For Terminal-Bench 2.0 and SWE-bench Verified, the score is the mean pass
  rate ± the sample standard deviation across three trials.
- For these two benchmarks, the task count is the number of tasks attempted in
  each trial. The scores use only the tasks that have results for all three
  models in that trial: 86/88/87 tasks for Terminal-Bench and 499/499/496 tasks
  for SWE-bench.
- The delivery rates for MLR-Bench and AstaBench are separate from their
  scores.
- The included MLR-Bench scorer uses the overall rating of two AI reviewers on
  a 1–10 scale. A task with no paper gets 0 (see the
  [public benchmark guide](benchmarks/README.md)).
- The [AstaBench breakdown](benchmarks/README.md#astabench-e2e-bench-hard-results)
  gives the score of each trial, the report counts, and the scores of the
  delivered reports.
- The [benchmark resources](benchmarks/README.md#official-benchmark-resources)
  list the official papers, code, and data.

This repository does not include all of these benchmark runs:

- The packaged MLR-Bench configuration covers an earlier subset of 12 tasks.
- The 107-task MLR-Bench configuration is not included.
- The evaluation adapters for AstaBench (E2E-Bench-Hard), DSBench,
  Terminal-Bench 2.0, and SWE-bench Verified are not included.

See [experiment support](docs/experiment_coverage.md) for the available
runners.

## What this repository contains

- **Workflow search:** RHI, trajectory collection, and ranking within each
  task.
- **Post-training:** conversation rendering, loss masks for assistant tokens
  only, LoRA, checkpoint selection by validation loss, and BF16/FP8 model
  export.
- **12 synthetic tasks:** every task prompt and initial workflow, with the
  nine-task training split and the three-task test split.
- **Evaluation:** pairwise judging for the synthetic tasks, and adapters for
  ScienceAgentBench and MLR-Bench.
- **Two MASS cycles:** a separate configuration for each of the two model
  updates.

The code runs without the original research repository. The pipeline
downloads or generates model weights, task datasets, trajectories, and
checkpoints when it needs them.

## Quick start

### 1. Preview the pipeline

These commands do not need a GPU or a model server. Run them from your cloned
`mass` directory. They install the locked core dependencies with
[uv](https://docs.astral.sh/uv/), preview the first cycle, and run the offline
tests:

```bash
uv sync --locked
uv run python -m mass plan --config configs/paper.json
uv run python -m unittest discover -s tests -v
```

`plan` prints the configuration and the training command. It does not execute
them. The tests use small fixtures and do not call a model. The RHI
integration test needs the core dependencies in
[pyproject.toml](pyproject.toml). If these dependencies are not available, the
test is skipped.

### 2. Run your first search

For a real experiment, follow the [setup guide](docs/running.md). It covers
Python environments, model downloads, vLLM serving, GPU allocation, and
Windows workstations. When the model server runs, you can start with one task:

```bash
python -m mass search --config configs/paper.json --tasks 51
```

This command uses the configured search budget. The
[single-task example](docs/running.md#try-one-task) shows how to do a shorter
search in a separate run directory.

## Run a MASS cycle

Each stage is a separate command. Thus you can examine the intermediate
outputs and resume completed work.

Run a stage with `python -m mass <stage> --config <config>`.

| Command   | Output                                                                            |
| --------- | --------------------------------------------------------------------------------- |
| `search`  | Proposed workflows, execution workspaces, and one retained workflow for each task |
| `collect` | Candidate trajectories from the retained workflows                                |
| `rank`    | Comparisons within each task and trajectory rankings                              |
| `select`  | Manifests of training and validation trajectories                                 |
| `render`  | Tokenized conversations and loss masks for assistant tokens only                  |
| `train`   | LoRA checkpoints and validation losses                                            |
| `export`  | The selected checkpoint, merged into BF16 and FP8 models                          |

| Cycle  | Model update | Configuration                              |
| ------ | ------------ | ------------------------------------------ |
| First  | `L0` → `L1`  | [configs/paper.json](configs/paper.json)   |
| Second | `L1` → `L2`  | [configs/cycle2.json](configs/cycle2.json) |

The second cycle trains a new LoRA adapter on the merged first-cycle model.
The [run guide](docs/running.md) gives the complete sequence of commands.

For each eligible training task, the trajectories with ranks 1–15 go into
training, and the trajectory with rank 16 goes into validation. The pipeline
selects the saved checkpoint with the lowest validation loss. External judges
are only for reporting. They are not part of this training loop.

## Tasks and evaluation

The synthetic tasks cover finance, pharmacy, and robotics:

| Split    | Task IDs                                 |
| -------- | ---------------------------------------- |
| Training | 51, 53, 56, 202, 205, 221, 302, 307, 324 |
| Test     | 60, 207, 305                             |

The repository includes task 221. But the reported experiment did not use task
221 for teacher collection. The other eight training tasks gave the SFT data.
Test tasks are never used for SFT or for checkpoint selection. See the
[task index](tasks/README.md) for every prompt and initial workflow. See
[tasks/splits.json](tasks/splits.json) for the split in machine-readable form.

Use `rollout` and `report` for
[synthetic-task evaluation](docs/running.md#synthetic-task-evaluation). The
[public benchmark guide](benchmarks/README.md) describes how to evaluate
`L0`, `L1`, or `L2` + qwen-code on ScienceAgentBench and MLR-Bench.

## Repository layout

| Path                                                                           | Contents                                                       |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| [mass/](mass/)                                                                 | Pipeline commands and trajectory selection                     |
| [configs/](configs/)                                                           | Cycle settings and base-model revisions                        |
| [tasks/](tasks/)                                                               | Task prompts, initial workflows, and teacher directive         |
| [harness_improvement/](harness_improvement/)                                   | RHI feedback, workflow history, and updates                    |
| [runtime/](runtime/)                                                           | Episode runner and request logging proxy                       |
| [training/](training/)                                                         | Conversation rendering, LoRA training, and model export        |
| [evaluation/](evaluation/), [evaluation_claudecodex/](evaluation_claudecodex/) | Workspace evidence and pairwise judging                        |
| [benchmarks/](benchmarks/)                                                     | Public benchmark setup, execution, and scoring                 |
| [tests/](tests/)                                                               | Offline tests                                                  |
| [tools/](tools/)                                                               | Model downloads, rollout pairing, and source archive utilities |
| [assets/](assets/)                                                             | Teaser figure and animated overview                            |

## Documentation

| To do this                                       | Read                                                    |
| ------------------------------------------------ | ------------------------------------------------------- |
| Set up an experiment                             | [Installation and running](docs/running.md)             |
| Examine the synthetic tasks                      | [Task prompts and splits](tasks/README.md)              |
| Run ScienceAgentBench or MLR-Bench               | [Public benchmarks](benchmarks/README.md)               |
| Connect the paper to the code                    | [Notation and training settings](docs/paper_to_code.md) |
| Find which experiments are included              | [Experiment support](docs/experiment_coverage.md)       |
| Understand differences from the paper's protocol | [Implementation notes](docs/reproduction.md)            |
| Run tests or build a source ZIP                  | [Development guide](docs/development.md)                |

## Limitations

- This repository supports the pipeline and benchmarks in this README. It does
  not include every ablation or analysis from the paper.
- The current RHI driver uses comparisons with a fixed reference and champion
  arbitration. Algorithm 1 in the paper compares directly with the best
  retained output. Thus the RHI driver is different from Algorithm 1.
- [Experiment support](docs/experiment_coverage.md) and the
  [implementation notes](docs/reproduction.md) describe these differences and
  the current limitations of the evaluation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes, bug reports, and
experiment extensions. [NOTICE.md](NOTICE.md) covers third-party software and
data.
