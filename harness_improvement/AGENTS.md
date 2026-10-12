# harness_improvement/

## OVERVIEW
RHI optimizer: one invocation = one step (judge v(i-1) vs v(i), resolve champion, gather evidence, ask the optimizer LLM for design v(i+1)). Own file because it is the largest code area (24 modules, ~3.5k LOC, score 11) and owns the prompt protocol.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Entry / launch | `iterate_multi_agent_prompt.py` `main()` -> `rhi_main.run` | launched only as `python -m harness_improvement.iterate_multi_agent_prompt` from `mass/collection.py` (flag list ~L29-79) |
| CLI contract | `rhi_cli.py`, `rhi_cli_options.py` | flag order, defaults, help text are the contract |
| Arg checks | `rhi_validate.py` | `--require-local-vllm` demands loopback `--base-url` / `--judge-base-url` |
| Per-run state | `rhi_context.py` `RunContext` | threaded through nearly every module |
| Design block markers | `rhi_template.py` | `DEFAULT_START_MARKER` / `DEFAULT_END_MARKER` |
| Adjacent judging | `rhi_judge_adjacent.py` | v(i-1) vs v(i), strict provenance checks |
| Baseline judging | `rhi_judge_baseline.py` | `--judge-vs-baseline`: vs no-harness `reference` repo |
| Champion | `rhi_best.py` -> `rhi_champion.py`, `rhi_arbitration.py` | champion (A) vs current (B) |
| Optimizer prompt | `rhi_prompt.py` (337 LOC, most flag branches) | evidence: `rhi_evidence.py`, `rhi_results_json.py`, `rhi_history.py` |
| Optimizer call + outputs | `rhi_improve.py`, `rhi_optimize.py`, `rhi_design.py` | writes v(i+1) artifacts |
| LLM clients | `rhi_backends.py` `Backends` -> `llm_backend.py` `JsonLLMBackend` -> `llm_json.py` | `llm_json` = parsing only, no network |

## CONVENTIONS
- Defaults are byte-identical to the recorded runs: every new behavior goes behind a flag that defaults off (`rhi_cli.py:126`, `rhi_cli_options.py:22,54`). `--compact-era-prompt` replays older wording; never set it for the free lineage (`rhi_prompt.py:101`).
- `--improvement-output-mode json|design-text` (default `json`): `json` requires `improved_multi_agent_design` plus the explanatory arrays; `design-text` is a recorded fallback for local models, and `rhi_optimize.py` fills the arrays empty with a `metadata_note`.
- Artifacts are write-once: `rhi_io._write_json_once` and prompt dumps accept identical content, raise on different content. History is append-only JSONL.
- Champion advances only on a direct win (arbitration winner `B`); in baseline mode v0 keeps the crown until a version proves a win.
- Judge backend = optimizer backend unless `--judge-llm-backend`; a separate judge gets `temperature=None` and inherits no sampling knobs (`rhi_backends.py:73-109`).
- Context guard: tokens estimated as chars/4 (`rhi_io.py:44`); over budget -> `SystemExit`, never silent truncation.
- `main()` resolves `JsonLLMBackend` from module globals at call time; tests patch `iterate_multi_agent_prompt.JsonLLMBackend`. Keep that name importable there.
- Layer rule: may import `evaluation`, `evaluation_claudecodex`; never `mass`.

## ANTI-PATTERNS
- No automatic retries: `max_retries=0` (`llm_backend.py:142`) because a retry can duplicate a judgment/generation. Background polling tolerates only `MAX_CONSECUTIVE_POLL_ERRORS = 5` retrieve errors.
- Protocol rejections (fixed agent limit, size) write `rejected_result_seed_<seed>.json` and `SystemExit` (`rhi_optimize.py:104-127`). Do not wrap them in a retry loop.
- Do not edit prompt text in `rhi_prompt.py` / `rhi_history.py` on the default path; add a flag.
- `_load_history` skips unparsable JSONL lines silently (`rhi_io.py:88-92`): missing verdicts may mean a corrupt `history.jsonl`, not an absent judgment.
- `assert backend is not None` guards exist for `--dry-run` paths; do not remove them to "fix" a dry run.

## COMMANDS
```bash
uv run python -m unittest tests.test_rhi_core tests.test_rhi_modules tests.test_llm_backend tests.test_sdk_contracts -v
```
