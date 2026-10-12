# Token counting and head/tail token-budget truncation for judge evidence excerpts.
from __future__ import annotations

import math
import warnings
from typing import Callable

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]

TRUNCATION_MARKER = "\n\n[... middle omitted (token-capped) ...]\n\n"


def estimate_tokens(text: str) -> int:
    """Conservative chars->tokens estimate (4 chars per token, minimum 1)."""
    return max(1, math.ceil(len(text) / 4))


def _token_counter_for_model(model_name: str | None) -> tuple[Callable[[str], int], str]:
    """
    Return a token counting function and a short mode label.

    - For OpenAI GPT models, use ``tiktoken`` (exact tokenizer for this use-case).
    - Fallback to a conservative chars->tokens estimate when tokenizer is unavailable.
    """
    model = (model_name or "").strip().lower()
    if model.startswith("gpt-") and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except (OSError, ValueError) as err:
            # get_encoding may need to download the BPE file; fall back but say so.
            warnings.warn(f"tiktoken o200k_base unavailable ({err!r}); using chars/4 token estimate", RuntimeWarning, stacklevel=2)
        else:
            return (lambda s: len(enc.encode(s))), "tiktoken:o200k_base"
    # Conservative fallback for non-OpenAI models or missing tokenizer.
    return estimate_tokens, "estimate:chars_div_4"


def _truncate_text_to_token_budget(
    text: str,
    *,
    max_tokens: int,
    token_count: Callable[[str], int],
) -> tuple[str, bool]:
    """Keep a head and tail of ``text`` joined by a marker so the excerpt fits ``max_tokens``.

    The fit check counts the full candidate (head + marker + tail) against ``max_tokens``,
    so the marker cost is already included; no separate head/tail budget is needed.
    """
    if max_tokens <= 0:
        return text, False
    if token_count(text) <= max_tokens:
        return text, False

    marker = TRUNCATION_MARKER
    if token_count(marker) >= max_tokens:
        return marker, True

    head_chars = max(1, len(text) // 4)
    tail_chars = max(1, len(text) // 4)
    # Shrink head/tail windows until the excerpt fits token budget.
    for _ in range(24):
        candidate = text[:head_chars] + marker + text[-tail_chars:]
        if token_count(candidate) <= max_tokens:
            return candidate, True
        head_chars = max(1, int(head_chars * 0.7))
        tail_chars = max(1, int(tail_chars * 0.7))

    # Last-resort tiny excerpt.
    tiny = text[:256] + marker + text[-256:]
    return tiny, True
