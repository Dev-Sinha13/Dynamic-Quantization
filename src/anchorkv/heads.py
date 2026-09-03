"""Receiver-head discovery from token-level causal attention."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .types import HeadScore, TokenSpan

FloatArray = NDArray[np.floating]


def sentence_vertical_scores(
    attention: FloatArray,
    spans: Sequence[TokenSpan],
    *,
    normalize_target_length: bool = True,
) -> NDArray[np.float64]:
    """Aggregate causal token attention into per-sentence vertical scores.

    Args:
        attention: Array shaped ``[layers, query_heads, query_tokens, key_tokens]``.
        spans: Ordered, non-overlapping sentence token spans in the same sequence.
        normalize_target_length: Divide each score by the target sentence length.

    Returns:
        Array shaped ``[layers, query_heads, sentences]``. Each value is the
        average attention paid to a sentence by all tokens generated after it.
    """

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "attention must have shape [layers, query_heads, query_tokens, key_tokens]"
        )
    query_tokens, key_tokens = values.shape[-2:]
    _validate_spans(spans, min(query_tokens, key_tokens))

    scores = np.zeros((*values.shape[:2], len(spans)), dtype=np.float64)
    for sentence_index, span in enumerate(spans):
        future_start = span.end
        if future_start >= query_tokens:
            continue
        future_attention = values[:, :, future_start:, span.start : span.end]
        # Sum the mass received by the sentence, then average across future queries.
        received = future_attention.sum(axis=-1).mean(axis=-1)
        if normalize_target_length:
            received = received / span.length
        scores[:, :, sentence_index] = received
    return scores


def discover_receiver_heads(
    traces: Sequence[FloatArray],
    *,
    top_k: int = 16,
) -> list[HeadScore]:
    """Rank heads by concentrated, repeatable sentence-level attention.

    Each trace is a ``[layers, query_heads, sentences]`` vertical-score array.
    Pearson kurtosis measures attentional concentration. Because raw kurtosis
    is sensitive to the number and distribution of sentences in a trace, heads
    are selected by their within-trace percentile ranks. Stability penalizes
    heads whose percentile rank changes substantially across traces.
    """

    if not traces:
        raise ValueError("at least one trace is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    shape = np.asarray(traces[0]).shape[:2]
    if len(shape) != 2:
        raise ValueError("trace scores must have shape [layers, query_heads, sentences]")

    per_trace: list[NDArray[np.float64]] = []
    for trace in traces:
        values = np.asarray(trace, dtype=np.float64)
        if values.ndim != 3 or values.shape[:2] != shape:
            raise ValueError("all traces must share [layers, query_heads] dimensions")
        if values.shape[-1] < 4:
            raise ValueError("each trace needs at least four sentences for kurtosis")
        per_trace.append(_pearson_kurtosis(values, axis=-1))

    stacked = np.stack(per_trace, axis=0)
    means = stacked.mean(axis=0)
    percentile_ranks = np.stack([_percentile_ranks(values) for values in per_trace])
    mean_percentiles = percentile_ranks.mean(axis=0)
    percentile_deviations = percentile_ranks.std(axis=0)
    stability = 1.0 / (
        1.0
        + percentile_deviations / np.maximum(np.abs(mean_percentiles), 1e-12)
    )

    ranked = [
        HeadScore(
            layer=layer,
            query_head=head,
            mean_kurtosis=float(means[layer, head]),
            stability=float(stability[layer, head]),
            mean_percentile=float(mean_percentiles[layer, head]),
        )
        for layer in range(shape[0])
        for head in range(shape[1])
    ]
    ranked.sort(key=lambda item: item.ranking_score, reverse=True)
    return ranked[: min(top_k, len(ranked))]


def query_head_to_kv_head(
    query_head: int,
    *,
    num_query_heads: int,
    num_kv_heads: int,
) -> int:
    """Map a query head to its shared grouped-query-attention KV head."""

    if num_query_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if not 0 <= query_head < num_query_heads:
        raise ValueError("query_head is out of range")
    return query_head // (num_query_heads // num_kv_heads)


def aggregate_receiver_scores_by_kv_head(
    scores: Sequence[HeadScore],
    *,
    num_query_heads: int,
    num_kv_heads: int,
    reduction: str = "max",
) -> dict[tuple[int, int], float]:
    """Aggregate query-side receiver scores onto physically stored KV heads."""

    grouped: dict[tuple[int, int], list[float]] = {}
    for score in scores:
        kv_head = query_head_to_kv_head(
            score.query_head,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        )
        grouped.setdefault((score.layer, kv_head), []).append(score.ranking_score)

    reducers = {
        "max": np.max,
        "mean": np.mean,
        "sum": np.sum,
    }
    if reduction not in reducers:
        raise ValueError("reduction must be one of: max, mean, sum")
    reducer = reducers[reduction]
    return {key: float(reducer(values)) for key, values in grouped.items()}


def _pearson_kurtosis(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    centered = values - values.mean(axis=axis, keepdims=True)
    second = np.mean(centered**2, axis=axis)
    fourth = np.mean(centered**4, axis=axis)
    return np.divide(
        fourth,
        second**2,
        out=np.zeros_like(fourth),
        where=second > 1e-12,
    )


def _percentile_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return tie-aware within-array percentile ranks in the closed interval [0, 1]."""

    flattened = np.asarray(values, dtype=np.float64).ravel()
    if flattened.size == 1:
        return np.ones_like(values, dtype=np.float64)
    _, inverse, counts = np.unique(flattened, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    midranks = starts + (counts - 1) / 2
    return (midranks[inverse] / (flattened.size - 1)).reshape(values.shape)


def _validate_spans(spans: Sequence[TokenSpan], sequence_length: int) -> None:
    previous_end = 0
    for span in spans:
        if span.start < previous_end:
            raise ValueError("spans must be ordered and non-overlapping")
        if span.end > sequence_length:
            raise ValueError("span exceeds attention sequence length")
        previous_end = span.end
