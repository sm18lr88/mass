# evaluation/ (+ evaluation_claudecodex/)

## OVERVIEW
Judge evidence and pairwise judging. `evaluation/` turns a submission workspace into bounded prompt text; `evaluation_claudecodex/` runs one A-vs-B judge call per provider. Own file: distinct domain, the lowest layers that RHI, `mass.judging`, and tests all build on (score 9, 21 inbound imports). Covers both packages.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Evidence bundle | `evaluation/bundle.py` `build_judge_bundle`, `JudgeBundle` | caps: tree 400 files/depth 6, report 28k, json 24k, code 12k chars, 4 code files, log head 12k/tail 16k |
| Bundle -> text | `bundle_workspace_evidence`, `bundle_to_user_message` | pairwise reuses the workspace section with A/B titles |
| Capped reads | `evaluation/artifacts.py` | binaries omitted, `[truncated]` flag in headers |
| Expected deliverables | `evaluation/deliverables.py` `scan_deliverables` | plot paths from task text vs files on disk |
| Token budgets | `evaluation/token_budget.py` | head/tail truncation with `TRUNCATION_MARKER` |
| OpenAI quirks | `evaluation/openai_responses.py` `responses_create_adaptive` | drops `temperature`/`reasoning` on 400/TypeError and retries (max 8) |
| Usage / cost | `evaluation/responses_metrics.py` | built-in $/1M table; estimates only |
| Pairwise entry | `evaluation_claudecodex/pairwise_judge.py` `run_pairwise_judge` | providers: `openai`, `anthropic`, `google` |
| Provider adapters | `evaluation_claudecodex/pairwise_providers.py` | Anthropic is streamed, then consumed into one Message |
| Judge protocol text | `evaluation_claudecodex/pairwise_prompts.py` | system prompt chosen per numeric query id |
| Verdict schema | `evaluation_claudecodex/pairwise_schema.py` `PairwiseJudgeResult` | pydantic |

## CONVENTIONS
- Layer order (enforced by `tests/test_architecture.py`): `evaluation` imports no repo package; `evaluation_claudecodex` imports only `evaluation`. Classes/dataclasses/pydantic are normal here; the no-classes style is `mass/` only.
- Local vLLM self-judging does NOT go through these adapters: `mass/judging.py` uses `harness_improvement.llm_backend.JsonLLMBackend` (`local-vllm`) plus `build_pairwise_user_message` and the same prompts/schema.
- Pairwise bundles must share `task_text` and `task_query_id` (raises `ValueError`).
- Persist raw `usage` with every judgment; `estimated_cost_usd` is a linear estimate and `None` for models missing from the table.
- Optional SDKs (`anthropic`, `google.genai`) are imported in try/except; a missing SDK raises only when that provider is called.

## ANTI-PATTERNS
- Empty or schema-invalid judge output raises `RuntimeError` with the raw head; never default it to a tie or a winner.
- Do not edit `pairwise_prompts.py` text or bundle layout/caps casually: they are the judge protocol shared by RHI, self-judging, and external judging; changing them changes recorded comparisons.

## COMMANDS
```bash
uv run python -m unittest tests.test_evaluation tests.test_sdk_contracts -v
```
