"""AnchorKV: thought-anchor-guided KV-cache compression research tools."""

from .artifacts import AttentionTrace, TraceMetadata, hash_text, load_trace, save_trace
from .capacity import CaptureEstimate, ModelGeometry, estimate_eager_capture
from .heads import (
    aggregate_receiver_scores_by_kv_head,
    discover_receiver_heads,
    query_head_to_kv_head,
    sentence_vertical_scores,
)
from .policy import (
    AnchorDecision,
    AnchorState,
    CacheBudgetError,
    CacheMode,
    CachePlan,
    DelayedAnchorTracker,
    KVGeometry,
    PlannedSegment,
    plan_cache,
)
from .segmentation import sentence_character_spans, token_spans_from_offsets
from .types import HeadScore, TokenSpan

__all__ = [
    "HeadScore",
    "AnchorDecision",
    "AnchorState",
    "AttentionTrace",
    "CacheBudgetError",
    "CacheMode",
    "CachePlan",
    "CaptureEstimate",
    "DelayedAnchorTracker",
    "KVGeometry",
    "ModelGeometry",
    "PlannedSegment",
    "TokenSpan",
    "TraceMetadata",
    "aggregate_receiver_scores_by_kv_head",
    "discover_receiver_heads",
    "estimate_eager_capture",
    "hash_text",
    "load_trace",
    "query_head_to_kv_head",
    "plan_cache",
    "sentence_character_spans",
    "sentence_vertical_scores",
    "save_trace",
    "token_spans_from_offsets",
]

__version__ = "0.1.0"
