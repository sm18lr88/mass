# Argument validation for the RHI CLI, including the loopback-only endpoint guard.
from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

from harness_improvement.llm_backend import BACKEND_LOCAL_VLLM


def _is_loopback_base_url(value: str) -> bool:
    """Return whether an API base URL is explicitly local to this host."""
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if hostname.lower() == "localhost":
            return True
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Resolve --query-file/--runs-root and reject invalid flag combinations via SystemExit."""
    query_file = args.query_file.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve() if args.runs_root else None
    if not query_file.is_file():
        raise SystemExit(f"query file not found: {query_file}")
    if runs_root is not None and not runs_root.is_dir():
        raise SystemExit(f"runs root not found: {runs_root}")
    if runs_root is None and args.current_repo is None:
        raise SystemExit("Provide either --runs-root or --current-repo.")
    if args.current_version < 0:
        raise SystemExit("--current-version must be >= 0")
    if args.context_safety_tokens < 0:
        raise SystemExit("--context-safety-tokens must be >= 0")
    if args.require_local_vllm:
        if args.llm_backend != BACKEND_LOCAL_VLLM:
            raise SystemExit(
                "--require-local-vllm refuses every backend except local-vllm"
            )
        if not _is_loopback_base_url(args.base_url):
            raise SystemExit(
                "--require-local-vllm requires a loopback base URL; "
                f"received {args.base_url!r}"
            )
        if args.judge_llm_backend and args.judge_llm_backend != BACKEND_LOCAL_VLLM:
            raise SystemExit(
                "--require-local-vllm refuses a non-local --judge-llm-backend"
            )
        if args.judge_llm_backend == BACKEND_LOCAL_VLLM and not _is_loopback_base_url(
            args.judge_base_url
        ):
            raise SystemExit(
                "--require-local-vllm requires a loopback --judge-base-url; "
                f"received {args.judge_base_url!r}"
            )
    if args.strict_adjacent_history_provenance and args.current_version < 1:
        raise SystemExit(
            "--strict-adjacent-history-provenance requires --current-version >= 1"
        )
    if args.judge_vs_baseline and args.strict_adjacent_history_provenance:
        raise SystemExit(
            "--judge-vs-baseline is incompatible with "
            "--strict-adjacent-history-provenance (no adjacent rows exist)"
        )
    if args.judge_vs_baseline and args.current_version >= 1:
        if args.baseline_repo_root is None:
            raise SystemExit(
                "--judge-vs-baseline at --current-version >= 1 requires "
                "--baseline-repo-root"
            )
        if args.champion_repo_v0_root is None:
            raise SystemExit(
                "--judge-vs-baseline at --current-version >= 1 requires "
                "--champion-repo-v0-root (for the one-time v0 seed judging)"
            )
    if args.include_best_design and args.current_version >= 1:
        if not args.champion_repo_root_pattern or "{version}" not in args.champion_repo_root_pattern:
            raise SystemExit(
                "--include-best-design at --current-version >= 1 requires "
                "--champion-repo-root-pattern containing '{version}'"
            )
        if args.champion_repo_v0_root is None:
            raise SystemExit(
                "--include-best-design at --current-version >= 1 requires "
                "--champion-repo-v0-root"
            )
    return query_file, runs_root
