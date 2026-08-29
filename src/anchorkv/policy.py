"""Delayed thought-anchor detection and byte-budgeted cache planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .types import TokenSpan


class CacheMode(str, Enum):
    """Representation modes supported by the research simulator."""

    EVICTED = "evicted"
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"


@dataclass(frozen=True, slots=True)
class KVGeometry:
    """Model dimensions needed to estimate KV-cache storage."""

    layers: int
    kv_heads: int
    head_dim: int
    quant_group_size: int = 64
    scale_bytes: int = 2

    def __post_init__(self) -> None:
        for name in ("layers", "kv_heads", "head_dim", "quant_group_size", "scale_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def bytes_for(self, token_count: int, mode: CacheMode) -> int:
        """Estimate stored K and V bytes, including group quantization scales."""

        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        if mode is CacheMode.EVICTED or token_count == 0:
            return 0

        elements = token_count * self.layers * self.kv_heads * self.head_dim * 2
        if mode is CacheMode.FP16:
            return elements * 2

        packed = elements if mode is CacheMode.INT8 else math.ceil(elements / 2)
        scale_groups = math.ceil(elements / self.quant_group_size)
        return packed + scale_groups * self.scale_bytes


@dataclass(slots=True)
class AnchorState:
    """Online evidence accumulated for one reasoning sentence."""

    segment_id: int
    span: TokenSpan
    emitted_step: int
    ema_score: float = 0.0
    observations: int = 0

    @property
    def token_count(self) -> int:
        return self.span.length


@dataclass(frozen=True, slots=True)
class AnchorDecision:
    segment_id: int
    score: float
    provisional: bool
    anchor: bool


@dataclass(slots=True)
class DelayedAnchorTracker:
    """Keep new steps provisional until later reasoning can attend to them."""

    ema_decay: float = 0.8
    min_observations: int = 2
    provisional_steps: int = 2
    anchor_threshold: float = 0.05
    _states: dict[int, AnchorState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.min_observations < 0 or self.provisional_steps < 0:
            raise ValueError("observation and provisional counts must be non-negative")
        if self.anchor_threshold < 0:
            raise ValueError("anchor_threshold must be non-negative")

    def register(self, segment_id: int, span: TokenSpan, *, emitted_step: int) -> None:
        if segment_id in self._states:
            raise ValueError(f"segment {segment_id} is already registered")
        self._states[segment_id] = AnchorState(segment_id, span, emitted_step)

    def observe(self, segment_id: int, receiver_attention: float) -> None:
        if receiver_attention < 0 or not math.isfinite(receiver_attention):
            raise ValueError("receiver_attention must be finite and non-negative")
        state = self._get(segment_id)
        if state.observations == 0:
            state.ema_score = receiver_attention
        else:
            state.ema_score = (
                self.ema_decay * state.ema_score
                + (1.0 - self.ema_decay) * receiver_attention
            )
        state.observations += 1

    def decisions(self, *, current_step: int) -> list[AnchorDecision]:
        decisions: list[AnchorDecision] = []
        for state in sorted(self._states.values(), key=lambda item: item.segment_id):
            provisional = (
                current_step - state.emitted_step < self.provisional_steps
                or state.observations < self.min_observations
            )
            decisions.append(
                AnchorDecision(
                    segment_id=state.segment_id,
                    score=state.ema_score,
                    provisional=provisional,
                    anchor=not provisional and state.ema_score >= self.anchor_threshold,
                )
            )
        return decisions

    def states(self) -> list[AnchorState]:
        return sorted(self._states.values(), key=lambda item: item.segment_id)

    def _get(self, segment_id: int) -> AnchorState:
        try:
            return self._states[segment_id]
        except KeyError as error:
            raise KeyError(f"unknown segment {segment_id}") from error


@dataclass(frozen=True, slots=True)
class PlannedSegment:
    segment_id: int
    mode: CacheMode
    bytes_used: int
    importance: float
    reason: str


@dataclass(frozen=True, slots=True)
class CachePlan:
    segments: tuple[PlannedSegment, ...]
    budget_bytes: int
    used_bytes: int
    full_cache_bytes: int

    @property
    def compression_ratio(self) -> float:
        """Return full-cache bytes divided by planned bytes."""

        if self.used_bytes == 0:
            return math.inf if self.full_cache_bytes else 1.0
        return self.full_cache_bytes / self.used_bytes

    @property
    def budget_utilization(self) -> float:
        return self.used_bytes / self.budget_bytes if self.budget_bytes else 0.0


class CacheBudgetError(ValueError):
    """Raised when mandatory protected segments cannot fit in the budget."""


def plan_cache(
    states: list[AnchorState],
    decisions: list[AnchorDecision],
    *,
    geometry: KVGeometry,
    budget_bytes: int,
    current_step: int,
    recent_window: int = 1,
    recency_weight: float = 0.02,
    recency_half_life: float = 4.0,
) -> CachePlan:
    """Allocate mixed precision with protected anchors and greedy upgrades.

    Provisional, detected-anchor, and recent segments are retained in FP16.
    Remaining segments compete for INT4, INT8, and FP16 upgrades according to
    marginal retained utility per added byte.
    """

    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    if recent_window < 0 or recency_half_life <= 0:
        raise ValueError("recent_window must be non-negative and half-life positive")

    decision_by_id = {decision.segment_id: decision for decision in decisions}
    if {state.segment_id for state in states} != set(decision_by_id):
        raise ValueError("states and decisions must contain the same segment IDs")

    modes = {state.segment_id: CacheMode.EVICTED for state in states}
    reasons = {state.segment_id: "budgeted" for state in states}
    importance: dict[int, float] = {}
    used_bytes = 0

    for state in states:
        decision = decision_by_id[state.segment_id]
        age = max(0, current_step - state.emitted_step)
        recency = recency_weight * math.exp(-math.log(2) * age / recency_half_life)
        importance[state.segment_id] = max(0.0, decision.score) + recency

        protected_reason = ""
        if decision.provisional:
            protected_reason = "provisional"
        elif decision.anchor:
            protected_reason = "anchor"
        elif age < recent_window:
            protected_reason = "recent"

        if protected_reason:
            modes[state.segment_id] = CacheMode.FP16
            reasons[state.segment_id] = protected_reason
            used_bytes += geometry.bytes_for(state.token_count, CacheMode.FP16)

    if used_bytes > budget_bytes:
        raise CacheBudgetError(
            f"protected segments require {used_bytes} bytes, exceeding {budget_bytes}-byte budget"
        )

    retention = {
        CacheMode.EVICTED: 0.0,
        CacheMode.INT4: 0.72,
        CacheMode.INT8: 0.92,
        CacheMode.FP16: 1.0,
    }
    next_mode = {
        CacheMode.EVICTED: CacheMode.INT4,
        CacheMode.INT4: CacheMode.INT8,
        CacheMode.INT8: CacheMode.FP16,
    }

    while True:
        candidates: list[tuple[float, int, CacheMode, int]] = []
        for state in states:
            current_mode = modes[state.segment_id]
            if current_mode is CacheMode.FP16:
                continue
            upgraded_mode = next_mode[current_mode]
            current_bytes = geometry.bytes_for(state.token_count, current_mode)
            upgraded_bytes = geometry.bytes_for(state.token_count, upgraded_mode)
            byte_delta = upgraded_bytes - current_bytes
            if used_bytes + byte_delta > budget_bytes:
                continue
            utility_delta = importance[state.segment_id] * (
                retention[upgraded_mode] - retention[current_mode]
            )
            if utility_delta <= 0:
                continue
            candidates.append(
                (utility_delta / max(byte_delta, 1), state.segment_id, upgraded_mode, byte_delta)
            )

        if not candidates:
            break
        _, segment_id, upgraded_mode, byte_delta = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        modes[segment_id] = upgraded_mode
        used_bytes += byte_delta

    planned = tuple(
        PlannedSegment(
            segment_id=state.segment_id,
            mode=modes[state.segment_id],
            bytes_used=geometry.bytes_for(state.token_count, modes[state.segment_id]),
            importance=importance[state.segment_id],
            reason=reasons[state.segment_id],
        )
        for state in sorted(states, key=lambda item: item.segment_id)
    )
    full_cache_bytes = sum(
        geometry.bytes_for(state.token_count, CacheMode.FP16) for state in states
    )
    return CachePlan(planned, budget_bytes, used_bytes, full_cache_bytes)
