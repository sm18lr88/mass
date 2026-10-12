# argparse flags for optional RHI protocol features, sampling, evidence caps, and output validation.
from __future__ import annotations

import argparse
from pathlib import Path

from harness_improvement.rhi_template import DEFAULT_END_MARKER, DEFAULT_START_MARKER


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-best-design",
        action="store_true",
        help=(
            "BEST-H anchoring: track the best-so-far design (champion) from the "
            "in-loop pairwise verdicts and, when the champion is OLDER than the "
            "current design, show it to the optimizer alongside the current "
            "design with adapted instructions. When the current version wins its "
            "adjacent comparison while the champion is older, ONE extra judge "
            "call (champion vs current) arbitrates; those rows live in a "
            "separate records/champion_history.jsonl ledger so the standard "
            "history pipeline is untouched. Default off (byte-identical prompts)."
        ),
    )
    parser.add_argument(
        "--champion-repo-root-pattern",
        type=str,
        default=None,
        help=(
            "Runs-root pattern for champion arbitration repos, with a "
            "'{version}' placeholder (e.g. 'runs/L0/workspaces/v{version}'). "
            "Required with --include-best-design at --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--champion-repo-v0-root",
        type=Path,
        default=None,
        help=(
            "Runs root holding the v0 repos for champion arbitration (the fork "
            "point may live outside the pattern's namespace). Required with "
            "--include-best-design at --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--history-design-diffs",
        action="store_true",
        help=(
            "DIFF-HISTORY: render each adjacent pairwise-history entry with a "
            "compact unified diff of the design blocks behind it, so the "
            "optimizer sees WHAT change produced WHAT judgment (credit "
            "assignment). Deterministic and API-free (reads versions/*.txt); "
            "recent pairs get fuller diffs than old ones. Default off "
            "(byte-identical history sections)."
        ),
    )
    parser.add_argument(
        "--judge-vs-baseline",
        action="store_true",
        help=(
            "BASELINE-ANCHORED JUDGING: replace the adjacent v{k-1}-vs-v{k} "
            "in-loop comparison with v{k} vs the FIXED no-harness baseline "
            "repo (plus a one-time v0-vs-baseline seed at the first judged "
            "step). History entries, scoreboard, champion rule, and the "
            "improvement instructions all become baseline-anchored: the "
            "stated objective is to decisively outperform the no-harness "
            "baseline, not merely the previous round. Champion advances only "
            "on direct wins (a baseline win when no champion has one, else "
            "one arbitration call). Incompatible with "
            "--strict-adjacent-history-provenance. Default off."
        ),
    )
    parser.add_argument(
        "--baseline-repo-root",
        type=Path,
        default=None,
        help=(
            "Root holding the no-harness baseline workspaces, resolved as "
            "<root>/query{N} (no _ourTeam suffix). Required with "
            "--judge-vs-baseline at --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--include-v0-design",
        action="store_true",
        help=(
            "Show the ORIGINAL design v0 (the human-written starting point) in "
            "the improvement prompt as the root of the version track, so the "
            "history diffs are groundable and drift from the original intent "
            "is visible. Skipped at step 0 (v0 IS the current design) and when "
            "v0 is already shown as the champion. Default off."
        ),
    )


def _add_sampling_and_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature; omitted by default")
    parser.add_argument("--omit-temperature", action="store_true", help="Force omitting temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Optional local-vLLM top-p.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional local-vLLM top-k.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. The Qwen local recipe uses 32768.",
    )
    parser.add_argument(
        "--local-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Qwen thinking through chat_template_kwargs (local-vllm only).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional local-vLLM sampling seed.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3600.0,
        help="Per-call API timeout.",
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=None,
        help="Optional context ceiling used for a conservative pre-call guard.",
    )
    parser.add_argument(
        "--context-safety-tokens",
        type=int,
        default=8192,
        help="Reserved context headroom for the conservative local context guard.",
    )
    parser.add_argument(
        "--submission-label",
        type=str,
        default="the coding agent",
        help="Neutral provenance label used inside evaluator/optimizer prompts.",
    )
    parser.add_argument(
        "--compact-era-prompt",
        action="store_true",
        help=(
            "Byte-exact restoration of the compact-lineage optimizer prompt: "
            "emits the two generic keep-it-compact sentences and the two "
            "no-numeric-caps sentences WITHOUT enabling the fixed-agent-limit "
            "output validator. For continuing the archived compact lineage "
            "only; never set for the free lineage."
        ),
    )
    parser.add_argument(
        "--results-json-max-chars",
        type=int,
        default=None,
        help=(
            "Per-file character cap for the full results/*.json appendix "
            "(head+tail kept, middle omitted). Default: no cap, matching the "
            "original claude-era behavior."
        ),
    )
    parser.add_argument(
        "--results-json-total-max-chars",
        type=int,
        default=None,
        help=(
            "Total character budget for the results/*.json appendix; files "
            "beyond it are listed as omitted. Default: no cap."
        ),
    )
    parser.add_argument(
        "--omit-full-results-json",
        action="store_true",
        help=(
            "Do not append every results/*.json file after the already capped evidence "
            "bundle. Recommended for local models to avoid duplicate evidence."
        ),
    )
    parser.add_argument(
        "--forbid-fixed-agent-limits",
        action="store_true",
        help=(
            "Reject generated designs that numerically cap agents, subagents, recalls, "
            "hops, turns, tools, or duration."
        ),
    )
    parser.add_argument(
        "--max-design-chars",
        type=int,
        default=None,
        help="Reject a generated design longer than this many characters.",
    )
    parser.add_argument(
        "--target-design-chars",
        type=int,
        default=None,
        help=(
            "Optional soft size target included in the optimizer prompt. Requires "
            "--max-design-chars and must not exceed it."
        ),
    )
    parser.add_argument(
        "--improvement-output-mode",
        choices=("json", "design-text"),
        default="json",
        help=(
            "Optimizer response format. design-text is a recorded fallback for local "
            "models that truncate while JSON-escaping a long harness."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save prompt artifacts, enforce the context guard, but make no API call.",
    )
    parser.add_argument(
        "--require-local-vllm",
        action="store_true",
        help=(
            "Refuse every non-local backend or non-loopback base URL. Use this for "
            "self-RHI trajectories that must not contact an external API."
        ),
    )
    parser.add_argument(
        "--strict-adjacent-history-provenance",
        action="store_true",
        help=(
            "For version 1 and later, require exactly one adjacent pairwise-history "
            "row matching this backend, model, base URL, and normalized repo pair."
        ),
    )
    parser.add_argument("--start-marker", type=str, default=DEFAULT_START_MARKER)
    parser.add_argument("--end-marker", type=str, default=DEFAULT_END_MARKER)
    parser.add_argument("--report-max-tokens", type=int, default=None)
    parser.add_argument("--json-max-tokens", type=int, default=50000)
    parser.add_argument("--code-max-tokens", type=int, default=10000)
