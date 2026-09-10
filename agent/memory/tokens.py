"""Approximate token counting -- what agent/memory/queue.py budgets X and Y
against.

Mercury's own tokenizer isn't public (Inception's docs -- checked directly,
2026-09-10 -- list only chat/FIM/edit/model endpoints, nothing exposing a
tokenizer or vocabulary). tiktoken's cl100k_base encoding stands in as a
reasonable, already-available approximation: it won't match Mercury's real
token count exactly, but being off by some percentage doesn't matter much
against a 40%-of-260K budget (agent/memory/queue.py's X_BUDGET/Y_BUDGET)
with real headroom on both sides of that line.

cl100k_base's own vocabulary file is fetched over the network on first use
(tiktoken's own doc/behavior, not this file's choice) and cached locally
after that. If that first fetch can't reach the network -- exactly the
situation this was built and tested under, a sandboxed dev environment
with a narrow egress allowlist -- count_tokens() falls back to a plain
chars-per-token heuristic rather than raising. A live `otto chat`/`otto
tui` run, with ordinary internet access, hits the real encoding after its
first (one-time, then cached) download.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: ~4 characters per token is a commonly cited rough average for English
#: text -- good enough for a fallback whose only job is "don't let the
#: budget check silently stop working," not to be a precise count.
_FALLBACK_CHARS_PER_TOKEN = 4

_encoding = None
_encoding_load_attempted = False


def _get_encoding():
    global _encoding, _encoding_load_attempted
    if not _encoding_load_attempted:
        _encoding_load_attempted = True
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # network, cache, or import failure
            logger.warning(
                "tiktoken's cl100k_base encoding is unavailable (%s) -- "
                "falling back to a chars/%d heuristic for token counting",
                exc, _FALLBACK_CHARS_PER_TOKEN,
            )
            _encoding = None
    return _encoding


def count_tokens(text: str) -> int:
    """Approximate token count for `text`. 0 for empty/None-ish input."""
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
