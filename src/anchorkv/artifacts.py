"""Portable, pickle-free artifacts for sentence-level attention traces."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .types import TokenSpan

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    model_id: str
    sample_id: str
    prompt_sha256: str
    seed: int
    dtype: str
    sequence_length: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    model_revision: str = "main"
    created_at_utc: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_id or not self.sample_id:
            raise ValueError("model_id and sample_id are required")
        if len(self.prompt_sha256) != 64:
            raise ValueError("prompt_sha256 must be a hexadecimal SHA-256 digest")
        try:
            int(self.prompt_sha256, 16)
        except ValueError as error:
            raise ValueError("prompt_sha256 must be hexadecimal") from error
        for name in ("sequence_length", "layers", "query_heads", "kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version {self.schema_version}")

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        sample_id: str,
        prompt: str,
        seed: int,
        dtype: str,
        sequence_length: int,
        layers: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        model_revision: str = "main",
    ) -> TraceMetadata:
        return cls(
            model_id=model_id,
            sample_id=sample_id,
            prompt_sha256=hash_text(prompt),
            seed=seed,
            dtype=dtype,
            sequence_length=sequence_length,
            layers=layers,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            model_revision=model_revision,
            created_at_utc=datetime.now(UTC).isoformat(),
        )


@dataclass(frozen=True, slots=True)
class AttentionTrace:
    """Compact output of one model trace after token-to-sentence reduction."""

    metadata: TraceMetadata
    spans: tuple[TokenSpan, ...]
    vertical_scores: NDArray[np.float32]

    def __post_init__(self) -> None:
        scores = np.asarray(self.vertical_scores)
        expected = (self.metadata.layers, self.metadata.query_heads, len(self.spans))
        if scores.shape != expected:
            raise ValueError(f"vertical_scores has shape {scores.shape}; expected {expected}")
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("vertical_scores must be finite and non-negative")
        previous_end = 0
        for span in self.spans:
            if span.start < previous_end or span.end > self.metadata.sequence_length:
                raise ValueError("spans must be ordered, non-overlapping, and within the sequence")
            previous_end = span.end


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_trace(trace: AttentionTrace, path: str | Path) -> Path:
    """Atomically save a compressed artifact that can be loaded without pickle."""

    destination = Path(path)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")

    metadata_json = json.dumps(asdict(trace.metadata), sort_keys=True)
    np.savez_compressed(
        temporary,
        vertical_scores=np.asarray(trace.vertical_scores, dtype=np.float32),
        span_starts=np.asarray([span.start for span in trace.spans], dtype=np.int32),
        span_ends=np.asarray([span.end for span in trace.spans], dtype=np.int32),
        span_text=np.asarray([span.text for span in trace.spans], dtype=np.str_),
        metadata_json=np.asarray(metadata_json, dtype=np.str_),
    )
    os.replace(temporary, destination)
    return destination


def load_trace(path: str | Path) -> AttentionTrace:
    source = Path(path)
    with np.load(source, allow_pickle=False) as artifact:
        metadata = TraceMetadata(**json.loads(str(artifact["metadata_json"].item())))
        starts = artifact["span_starts"].astype(np.int64)
        ends = artifact["span_ends"].astype(np.int64)
        texts = artifact["span_text"].astype(np.str_)
        if not (len(starts) == len(ends) == len(texts)):
            raise ValueError("trace span arrays have inconsistent lengths")
        spans = tuple(
            TokenSpan(int(start), int(end), str(text))
            for start, end, text in zip(starts, ends, texts, strict=True)
        )
        scores = artifact["vertical_scores"].astype(np.float32)
    return AttentionTrace(metadata=metadata, spans=spans, vertical_scores=scores)


def trace_summary(trace: AttentionTrace) -> dict[str, object]:
    score_min = float(trace.vertical_scores.min()) if trace.vertical_scores.size else 0.0
    score_max = float(trace.vertical_scores.max()) if trace.vertical_scores.size else 0.0
    return {
        "model_id": trace.metadata.model_id,
        "model_revision": trace.metadata.model_revision,
        "sample_id": trace.metadata.sample_id,
        "sequence_length": trace.metadata.sequence_length,
        "sentences": len(trace.spans),
        "layers": trace.metadata.layers,
        "query_heads": trace.metadata.query_heads,
        "kv_heads": trace.metadata.kv_heads,
        "score_min": score_min,
        "score_max": score_max,
    }
