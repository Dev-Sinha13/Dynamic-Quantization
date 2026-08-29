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


def _log_softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

