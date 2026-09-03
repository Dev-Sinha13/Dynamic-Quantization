"""Shared value objects used by AnchorKV's analysis and cache policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A half-open token span representing one reasoning step."""

    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class HeadScore:
    """Offline receiver-head discovery result."""

    layer: int
    query_head: int
    mean_kurtosis: float
    stability: float
    mean_percentile: float | None = None

    @property
    def ranking_score(self) -> float:
        """Favor heads that rank highly and consistently within each trace."""

        concentration = (
            self.mean_kurtosis if self.mean_percentile is None else self.mean_percentile
        )
        return concentration * self.stability
