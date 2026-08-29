"""AnchorKV: thought-anchor-guided KV-cache compression research tools."""

from .heads import (
    aggregate_receiver_scores_by_kv_head,
    discover_receiver_heads,
    query_head_to_kv_head,
    sentence_vertical_scores,
)
from .segmentation import sentence_character_spans, token_spans_from_offsets
from .types import HeadScore, TokenSpan

__all__ = [
    "HeadScore",
    "TokenSpan",
    "aggregate_receiver_scores_by_kv_head",
    "discover_receiver_heads",
    "query_head_to_kv_head",
    "sentence_character_spans",
    "sentence_vertical_scores",
    "token_spans_from_offsets",
]

__version__ = "0.1.0"
