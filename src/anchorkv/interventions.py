"""Causal attention-mask interventions for thought-anchor validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .types import TokenSpan


@dataclass(frozen=True, slots=True)
class ControlSelection:
    """Anchor and deterministic baseline sentence indices for one trace."""

    anchor_index: int
    recency_index: int
    length_matched_index: int


def causal_attention_mask(
    sequence_length: int,
    *,
    blocked_span: TokenSpan | None = None,
    query_end: int | None = None,
) -> NDArray[np.float32]:
    """Build a 4D additive causal mask, optionally suppressing one sentence.

    The returned shape is ``[1, 1, query, key]``. Zeros retain attention and
    negative infinity removes it. When a span is supplied, only queries after
    that sentence are prevented from attending to its key positions.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    effective_query_end = sequence_length if query_end is None else query_end
    if not 0 < effective_query_end <= sequence_length:
        raise ValueError("query_end must be within the sequence")
    if blocked_span is not None and blocked_span.end > effective_query_end:
        raise ValueError("blocked_span must end no later than query_end")

    mask = np.zeros((sequence_length, sequence_length), dtype=np.float32)
    mask[np.triu_indices(sequence_length, k=1)] = -np.inf
    if blocked_span is not None and blocked_span.end < effective_query_end:
        mask[
            blocked_span.end : effective_query_end,
            blocked_span.start : blocked_span.end,
        ] = -np.inf
    return mask[np.newaxis, np.newaxis, :, :]


def select_control_indices(
    spans: Sequence[TokenSpan],
    *,
    anchor_index: int,
    seed: int,
) -> ControlSelection:
    """Choose recency and length-matched random controls for an anchor.

    The random control is sampled deterministically from spans with the nearest
    token length to the anchor, excluding the anchor and recency controls when
    enough alternatives exist.
    """

    if not 0 <= anchor_index < len(spans):
        raise ValueError("anchor_index is out of range")
    eligible = [index for index in range(len(spans)) if index != anchor_index]
    if len(eligible) < 2:
        raise ValueError("at least three spans are required for two controls")

    recency_index = eligible[-1]
    random_pool = [index for index in eligible if index != recency_index]
    anchor_length = spans[anchor_index].length
    smallest_delta = min(abs(spans[index].length - anchor_length) for index in random_pool)
    matched = [
        index
        for index in random_pool
        if abs(spans[index].length - anchor_length) == smallest_delta
    ]
    generator = np.random.default_rng(seed)
    length_matched_index = int(generator.choice(matched))
    return ControlSelection(
        anchor_index=anchor_index,
        recency_index=recency_index,
        length_matched_index=length_matched_index,
    )


def downstream_window(
    spans: Sequence[TokenSpan],
    selection: ControlSelection,
    *,
    query_end: int,
) -> tuple[int, int]:
    """Return a common downstream token window for comparable interventions."""

    selected = (
        spans[selection.anchor_index],
        spans[selection.recency_index],
        spans[selection.length_matched_index],
    )
    start = max(span.end for span in selected)
    if start >= query_end:
        raise ValueError("selected spans leave no shared downstream evaluation window")
    return start, query_end
