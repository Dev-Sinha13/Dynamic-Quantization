"""Conservative memory estimates for eager attention capture experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import CacheMode, KVGeometry

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    parameters: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name in ("parameters", "layers", "query_heads", "kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")


@dataclass(frozen=True, slots=True)
class CaptureEstimate:
    model_weight_bytes: int
    attention_bytes: int
    kv_cache_bytes: int
    lower_bound_bytes: int
    available_bytes: int
    safety_fraction: float

    @property
    def safe_limit_bytes(self) -> int:
        return int(self.available_bytes * self.safety_fraction)

    @property
    def fits_conservative_limit(self) -> bool:
        return self.lower_bound_bytes <= self.safe_limit_bytes

    def as_gib(self) -> dict[str, float | bool]:
        return {
            "model_weights_gib": round(self.model_weight_bytes / GIB, 4),
            "returned_attentions_gib": round(self.attention_bytes / GIB, 4),
            "kv_cache_gib": round(self.kv_cache_bytes / GIB, 4),
            "lower_bound_gib": round(self.lower_bound_bytes / GIB, 4),
            "safe_limit_gib": round(self.safe_limit_bytes / GIB, 4),
            "fits_conservative_limit": self.fits_conservative_limit,
        }


def estimate_eager_capture(
    geometry: ModelGeometry,
    *,
    sequence_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
    available_gib: float = 16.0,
    safety_fraction: float = 0.8,
) -> CaptureEstimate:
    """Estimate the unavoidable tensors retained by ``output_attentions=True``.

    The result is intentionally a lower bound: it excludes hidden states, logits,
    temporary softmax buffers, CUDA workspaces, allocator fragmentation, and
    framework overhead. Keeping the bound below 80% of device capacity provides
    useful headroom but is not a guarantee that a run will fit.
    """

    if sequence_length <= 0 or batch_size <= 0 or dtype_bytes <= 0:
        raise ValueError("sequence length, batch size, and dtype bytes must be positive")
    if available_gib <= 0 or not 0 < safety_fraction <= 1:
        raise ValueError("available_gib must be positive and safety_fraction in (0, 1]")

    weight_bytes = geometry.parameters * dtype_bytes
    attention_bytes = (
        batch_size
        * geometry.layers
        * geometry.query_heads
        * sequence_length
        * sequence_length
        * dtype_bytes
    )
    kv_geometry = KVGeometry(
        layers=geometry.layers,
        kv_heads=geometry.kv_heads,
        head_dim=geometry.head_dim,
    )
    kv_bytes = batch_size * kv_geometry.bytes_for(sequence_length, CacheMode.FP16)
    lower_bound = weight_bytes + attention_bytes + kv_bytes
    return CaptureEstimate(
        model_weight_bytes=weight_bytes,
        attention_bytes=attention_bytes,
        kv_cache_bytes=kv_bytes,
        lower_bound_bytes=lower_bound,
        available_bytes=int(available_gib * GIB),
        safety_fraction=safety_fraction,
    )

