# argparse definition of the RHI command line; flag order, defaults, and help text are the CLI contract.
from __future__ import annotations

import argparse
from pathlib import Path

from harness_improvement.llm_backend import BACKEND_CHOICES, BACKEND_OPENAI_RESPONSES
from harness_improvement.rhi_cli_options import (
    _add_feature_arguments,
    _add_sampling_and_prompt_arguments,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the RHI parser; arguments are added in the original --help order."""
    parser = argparse.ArgumentParser(description="Iterative multi-agent design prompt improver")
    _add_run_and_backend_arguments(parser)
    _add_feature_arguments(parser)
    _add_sampling_and_prompt_arguments(parser)
    return parser


def _add_run_and_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Complete the final comparison and champion arbitration without proposing another workflow.")
    parser.add_argument("--query-file", type=Path, required=True, help="Path to query*_ourTeam.txt")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root directory containing query*_ourTeam and query*_ourTeam-v* output repos",
    )
    parser.add_argument(
        "--current-repo",
        type=Path,
        default=None,
        help=(
            "Explicit normalized submission root for the current version. "
            "When provided, it takes precedence over --runs-root resolution."
        ),
    )
    parser.add_argument(
        "--previous-repo",
        type=Path,
        default=None,
        help=(
            "Explicit normalized submission root for the previous version's pairwise "
            "comparison. Used only when --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--current-run-status-file",
        type=Path,
        default=None,
        help="Optional run_status.json included as provenance in the improvement prompt.",
    )
    parser.add_argument(
        "--prev-runs-root",
        type=Path,
        default=None,
        help=(
            "Optional previous-version runs root for comparison step. "
            "If omitted and --runs-root ends with _vN, the script tries to infer _v(N-1)."
        ),
    )
    parser.add_argument(
        "--current-version",
        type=int,
        required=True,
        help="Current available version index. 0 means base query*_ourTeam repo exists.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root for this harness. Default: ./harness_improvement/runs/<query_name>",
    )
    parser.add_argument(
        "--llm-backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_OPENAI_RESPONSES,
        help=(
            "Backend used for both pairwise evaluation and harness optimization. "
            "External GPT is never selected implicitly when local-vllm is requested."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID. Defaults to gpt-5.5 externally or Qwen/Qwen3.5-9B locally.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL for --llm-backend local-vllm.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help=(
            "Environment variable containing the API key. Defaults to OPENAI_API_KEY "
            "externally; local vLLM uses VLLM_API_KEY when set, otherwise EMPTY."
        ),
    )
    parser.add_argument("--reasoning-effort", type=str, default="xhigh", help="Reasoning effort for OpenAI Responses API")
    parser.add_argument(
        "--openai-background",
        action="store_true",
        help=(
            "Submit OpenAI Responses calls with background=True and poll for "
            "completion (short requests each poll). Use when long non-streaming "
            "calls die on NAT/conntrack idle timeouts (observed on GCP hosts). "
            "openai-responses backend only; default off (identical call path)."
        ),
    )
    parser.add_argument(
        "--judge-llm-backend",
        choices=list(BACKEND_CHOICES),
        default=None,
        help=(
            "Optional separate backend for the in-loop pairwise judge ONLY. "
            "Default (unset): the judge shares the main --llm-backend backend "
            "object unchanged (byte-identical call path). When set, a second "
            "backend is built for judge calls (e.g. an external gpt-5.5 judge "
            "steering a local-model optimizer). The judge backend does NOT "
            "inherit the main sampling knobs: it uses --judge-model / "
            "--judge-reasoning-effort with temperature omitted (claude-line "
            "judge parity) and honors --openai-background."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Model for --judge-llm-backend. Defaults to gpt-5.5 for "
            "openai-responses; required in spirit for local-vllm."
        ),
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort for the judge backend (e.g. xhigh). Omitted when unset.",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Base URL for --judge-llm-backend local-vllm (required in that case).",
    )
