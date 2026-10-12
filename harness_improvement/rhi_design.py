# Post-processing of generated designs: block normalization, fixed-limit detection, query rendering.
from __future__ import annotations

import re

from harness_improvement.rhi_template import QueryTemplate


def _render_query_with_design(template: QueryTemplate, design_text: str) -> str:
    return f"{template.before_design}{design_text.strip()}\n\n{template.after_design.lstrip()}"


def _normalize_generated_design_block(
    design_text: str,
    *,
    start_marker: str,
    end_marker: str,
    after_design: str,
) -> str:
    """Normalize model output so only the design block body is kept.

    The model occasionally returns text that continues past the intended
    design block and includes the end marker or trailing prompt sections.
    Keep only the design block content (start marker .. before end marker).
    """
    text = design_text.strip()
    if not text.startswith(start_marker):
        raise RuntimeError(
            "LLM output missing required design block prefix. "
            f"Expected to start with: {start_marker!r}"
        )

    # Prefer trimming only if the model copied the exact post-design tail.
    after_prefix = after_design.lstrip()
    if after_prefix:
        probe = after_prefix[:200]
        tail_idx = text.find(probe)
        if tail_idx >= 0:
            text = text[:tail_idx].rstrip()
            return text

    # Fallback: trim on end_marker only if it appears near the end.
    # This avoids false truncation when the phrase appears naturally in
    # the middle of a generated instruction sentence.
    end_idx = text.find(end_marker)
    if end_idx >= 0 and end_idx >= int(0.7 * len(text)):
        text = text[:end_idx].rstrip()
    return text


def _fixed_agent_limit_matches(design_text: str) -> list[str]:
    """Find numerical caps that contradict the uncapped experiment policy."""
    patterns = [
        re.compile(
            r"\b(?:maximum|max(?:imum)?\s+of|at\s+most|up\s+to)\s+\d+\s+"
            r"(?:agent|subagent|recall|re-delegation|redelegation|hop|turn|tool|"
            r"iteration|round|minute|hour)s?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:agent|subagent|recall|re-delegation|redelegation|hop|turn|tool)"
            r"(?:\s+\w+){0,3}\s+(?:is|are)?\s*(?:limited|capped)\s+to\s+\d+\b",
            flags=re.IGNORECASE,
        ),
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(design_text):
            excerpt = " ".join(match.group(0).split())
            if excerpt not in matches:
                matches.append(excerpt)
    return matches
