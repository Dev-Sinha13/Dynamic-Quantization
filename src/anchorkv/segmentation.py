"""Sentence-level segmentation helpers for visible reasoning traces."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .types import TokenSpan

_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")


def sentence_character_spans(text: str) -> list[tuple[int, int]]:
    """Return non-empty half-open character spans for sentence-like steps.

    Newlines are treated as boundaries because reasoning models commonly emit
    one logical step per line even when punctuation is omitted.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        start, end = _trimmed_bounds(text, cursor, match.start())
        if start < end:
            spans.append((start, end))
        cursor = match.end()

    start, end = _trimmed_bounds(text, cursor, len(text))
    if start < end:
        spans.append((start, end))
    return spans


def token_spans_from_offsets(
    text: str,
    token_offsets: Sequence[tuple[int, int]],
) -> list[TokenSpan]:
    """Map sentence character spans to tokenizer offset mappings.

    Tokens with empty offsets, such as special tokens, are ignored. A token is
    assigned to a sentence when its character interval overlaps the sentence.
    """

    result: list[TokenSpan] = []
    for char_start, char_end in sentence_character_spans(text):
        token_ids = [
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if token_end > token_start
            and token_end > char_start
            and token_start < char_end
        ]
        if not token_ids:
            continue
        result.append(
            TokenSpan(
                start=token_ids[0],
                end=token_ids[-1] + 1,
                text=text[char_start:char_end],
            )
        )
    return result


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end

