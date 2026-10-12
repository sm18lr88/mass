"""RHI: execute a workflow, compare artifacts, and propose the next workflow.

CLI entry point (``python -m harness_improvement.iterate_multi_agent_prompt``); the logic lives in ``rhi_*.py``.
Use ``python -m mass search`` for the paper configuration.
"""

from __future__ import annotations

from harness_improvement.llm_backend import JsonLLMBackend
from harness_improvement.rhi_cli import build_parser
from harness_improvement.rhi_main import run
from harness_improvement.rhi_template import (
    DEFAULT_END_MARKER,
    DEFAULT_START_MARKER,
    QueryTemplate,
    _read_query_template,
)

__all__ = [
    "DEFAULT_END_MARKER",
    "DEFAULT_START_MARKER",
    "JsonLLMBackend",
    "QueryTemplate",
    "_read_query_template",
    "main",
]


def main() -> int:
    """Parse the CLI and run one RHI step.

    ``JsonLLMBackend`` is resolved from this module's globals at call time so
    tests can patch ``iterate_multi_agent_prompt.JsonLLMBackend``.
    """
    return run(build_parser().parse_args(), backend_cls=JsonLLMBackend)


if __name__ == "__main__":
    raise SystemExit(main())
