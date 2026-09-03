"""Cross-trace receiver-head discovery and causal-effect metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .artifacts import AttentionTrace, load_trace
from .heads import aggregate_receiver_scores_by_kv_head, discover_receiver_heads


@dataclass(frozen=True, slots=True)
class CausalEffect:
    segment_id: int
    mean_kl: float
    max_kl: float


@dataclass(frozen=True, slots=True)
class RankingDiagnostics:
    """Agreement between a proxy ranking and measured causal importance."""

    spearman: float
    top_k: int
    top_k_overlap: float
    top_k_regret: float
    top_1_regret: float
    proxy_top_indices: tuple[int, ...]
    causal_top_indices: tuple[int, ...]


def receiver_head_manifest(
    traces: Sequence[AttentionTrace],
    *,
    top_k: int = 16,
) -> dict[str, object]:
    """Discover receiver heads from compatible sentence-level traces."""

    if not traces:
        raise ValueError("at least one attention trace is required")
    reference = traces[0].metadata
    compatibility = (
        reference.model_id,
        reference.model_revision,
        reference.layers,
        reference.query_heads,
        reference.kv_heads,
        reference.head_dim,
    )
    for trace in traces[1:]:
        candidate = trace.metadata
        if (
            candidate.model_id,
            candidate.model_revision,
            candidate.layers,
            candidate.query_heads,
            candidate.kv_heads,
            candidate.head_dim,
        ) != compatibility:
            raise ValueError("receiver-head discovery requires traces from one model revision")

    heads = discover_receiver_heads(
        [trace.vertical_scores for trace in traces],
        top_k=top_k,
    )
    kv_scores = aggregate_receiver_scores_by_kv_head(
        heads,
        num_query_heads=reference.query_heads,
        num_kv_heads=reference.kv_heads,
        reduction="max",
    )
    return {
        "schema_version": 1,
        "model_id": reference.model_id,
        "model_revision": reference.model_revision,
        "source_samples": [trace.metadata.sample_id for trace in traces],
        "receiver_heads": [asdict(head) | {"ranking_score": head.ranking_score} for head in heads],
        "kv_heads": [
            {"layer": layer, "kv_head": kv_head, "score": score}
            for (layer, kv_head), score in sorted(
                kv_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def discover_from_paths(
    paths: Sequence[str | Path],
    *,
    top_k: int = 16,
) -> dict[str, object]:
    return receiver_head_manifest([load_trace(path) for path in paths], top_k=top_k)


def save_manifest(manifest: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def kl_divergence_from_logits(
    reference_logits: NDArray[np.floating],
    intervention_logits: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return KL(reference || intervention) over the final vocabulary axis."""

    reference = np.asarray(reference_logits, dtype=np.float64)
    intervention = np.asarray(intervention_logits, dtype=np.float64)
    if reference.shape != intervention.shape or reference.ndim < 1:
        raise ValueError("reference and intervention logits must have the same non-scalar shape")
    reference_log_probs = _log_softmax(reference)
    intervention_log_probs = _log_softmax(intervention)
    reference_probs = np.exp(reference_log_probs)
    return np.sum(
        reference_probs * (reference_log_probs - intervention_log_probs),
        axis=-1,
    )


def score_causal_interventions(
    reference_logits: NDArray[np.floating],
    interventions: Mapping[int, NDArray[np.floating]],
) -> list[CausalEffect]:
    """Rank sentence interventions by their downstream logit divergence."""

    effects: list[CausalEffect] = []
    for segment_id, logits in interventions.items():
        divergences = kl_divergence_from_logits(reference_logits, logits)
        effects.append(
            CausalEffect(
                segment_id=segment_id,
                mean_kl=float(np.mean(divergences)),
                max_kl=float(np.max(divergences)),
            )
        )
    effects.sort(key=lambda effect: effect.mean_kl, reverse=True)
    return effects


def ranking_diagnostics(
    proxy_scores: NDArray[np.floating],
    causal_scores: NDArray[np.floating],
    *,
    top_k: int = 3,
) -> RankingDiagnostics:
    """Measure whether a cheap proxy recovers causally important sentences.

    Regret is measured in causal-score units. ``top_k_regret`` compares the
    mean causal effect of the oracle top-k set with the mean effect of the set
    selected by the proxy. Ties use deterministic, lower-index-first ordering.
    """

    proxy = np.asarray(proxy_scores, dtype=np.float64)
    causal = np.asarray(causal_scores, dtype=np.float64)
    if proxy.ndim != 1 or causal.ndim != 1 or proxy.shape != causal.shape:
        raise ValueError("proxy_scores and causal_scores must be equal-length vectors")
    if proxy.size < 2:
        raise ValueError("ranking diagnostics require at least two candidates")
    if not np.isfinite(proxy).all() or not np.isfinite(causal).all():
        raise ValueError("ranking scores must be finite")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    effective_k = min(top_k, proxy.size)
    proxy_order = np.lexsort((np.arange(proxy.size), -proxy))
    causal_order = np.lexsort((np.arange(causal.size), -causal))
    proxy_top = tuple(int(index) for index in proxy_order[:effective_k])
    causal_top = tuple(int(index) for index in causal_order[:effective_k])
    overlap = len(set(proxy_top).intersection(causal_top)) / effective_k
    top_k_regret = float(
        np.mean(causal[list(causal_top)]) - np.mean(causal[list(proxy_top)])
    )
    top_1_regret = float(causal[causal_top[0]] - causal[proxy_top[0]])

    proxy_ranks = _midranks(proxy)
    causal_ranks = _midranks(causal)
    proxy_centered = proxy_ranks - proxy_ranks.mean()
    causal_centered = causal_ranks - causal_ranks.mean()
    denominator = np.sqrt(
        np.sum(proxy_centered**2) * np.sum(causal_centered**2)
    )
    spearman = (
        0.0
        if denominator <= 1e-12
        else float(np.sum(proxy_centered * causal_centered) / denominator)
    )
    return RankingDiagnostics(
        spearman=spearman,
        top_k=effective_k,
        top_k_overlap=float(overlap),
        top_k_regret=top_k_regret,
        top_1_regret=top_1_regret,
        proxy_top_indices=proxy_top,
        causal_top_indices=causal_top,
    )


def _log_softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _midranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2)[inverse].astype(np.float64)
