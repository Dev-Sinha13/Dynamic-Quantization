"""AnchorKV: thought-anchor-guided KV-cache compression research tools."""

from .artifacts import AttentionTrace, TraceMetadata, hash_text, load_trace, save_trace
from .backend import (
    AttentionMode,
    BackendUnavailableError,
    CacheDirective,
    DeclarativeAttentionParser,
    DeclarativeKVCache,
    DirectiveKind,
    KVBlock,
    LossyPromotionError,
    MaterializedCache,
    PackedTensor,
    SegmentRecord,
    SegmentRole,
    quantize_tensor,
)
from .analysis import (
    CausalEffect,
    RankingDiagnostics,
    discover_from_paths,
    kl_divergence_from_logits,
    ranking_diagnostics,
    receiver_head_manifest,
    score_causal_interventions,
)
from .capacity import CaptureEstimate, ModelGeometry, estimate_eager_capture
from .heads import (
    aggregate_receiver_scores_by_kv_head,
    discover_receiver_heads,
    query_head_to_kv_head,
    sentence_vertical_scores,
)
from .interventions import (
    ControlSelection,
    causal_attention_mask,
    downstream_window,
    select_control_indices,
)
from .hf import HFTraceConfig, HFTraceResult, decoded_token_offsets, reduce_attention_layers
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
    "HFTraceConfig",
    "HFTraceResult",
    "AnchorDecision",
    "AnchorState",
    "AttentionMode",
    "AttentionTrace",
    "BackendUnavailableError",
    "CacheBudgetError",
    "CacheMode",
    "CachePlan",
    "CacheDirective",
    "CaptureEstimate",
    "CausalEffect",
    "ControlSelection",
    "DeclarativeAttentionParser",
    "DeclarativeKVCache",
    "DelayedAnchorTracker",
    "DirectiveKind",
    "KVGeometry",
    "KVBlock",
    "LossyPromotionError",
    "MaterializedCache",
    "ModelGeometry",
    "PlannedSegment",
    "PackedTensor",
    "RankingDiagnostics",
    "SegmentRecord",
    "SegmentRole",
    "TokenSpan",
    "TraceMetadata",
    "aggregate_receiver_scores_by_kv_head",
    "causal_attention_mask",
    "discover_receiver_heads",
    "discover_from_paths",
    "decoded_token_offsets",
    "downstream_window",
    "estimate_eager_capture",
    "hash_text",
    "load_trace",
    "kl_divergence_from_logits",
    "ranking_diagnostics",
    "query_head_to_kv_head",
    "quantize_tensor",
    "reduce_attention_layers",
    "receiver_head_manifest",
    "plan_cache",
    "sentence_character_spans",
    "sentence_vertical_scores",
    "save_trace",
    "score_causal_interventions",
    "select_control_indices",
    "token_spans_from_offsets",
]

__version__ = "0.1.0"
