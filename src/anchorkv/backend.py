"""Reference backend for blockwise mixed-precision KV-cache storage.

This module implements real tensor storage and bit packing. It deliberately
does not patch a serving engine: ``materialize`` reconstructs dense tensors for
correctness experiments, while a future vLLM adapter can map the same segment
and block metadata onto paged-attention block tables and kernels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .policy import CacheMode, CachePlan

try:
    import torch
except ImportError:  # pragma: no cover - exercised by lightweight installations
    torch = None


class BackendUnavailableError(RuntimeError):
    """Raised when the optional PyTorch backend dependency is unavailable."""


class LossyPromotionError(RuntimeError):
    """Raised when code attempts to restore precision that was already lost."""


class SegmentRole(str, Enum):
    SINK = "sink"
    SCAFFOLD = "scaffold"
    CONTEXT = "context"
    RESPONSE = "response"


class AttentionMode(str, Enum):
    GLOBAL = "global"
    FOCUS = "focus"
    LOCAL = "local"


class DirectiveKind(str, Enum):
    MODE = "mode"
    ANCHOR = "anchor"
    ARCHIVE = "archive"


@dataclass(frozen=True, slots=True)
class CacheDirective:
    kind: DirectiveKind
    mode: AttentionMode | None = None
    segment_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class DeclarativeState:
    mode: AttentionMode = AttentionMode.GLOBAL
    focus_segment_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class DeclarativeAttentionParser:
    """Incrementally parse DA-compatible attention and precision tags."""

    state: DeclarativeState = field(default_factory=DeclarativeState)
    _buffer: str = field(default="", init=False, repr=False)

    _segment_attribute = re.compile(
        r'(?:segments|magic_chunks)\s*=\s*["\']([0-9,\s]+)["\']',
        re.IGNORECASE,
    )

    def feed(self, text: str) -> tuple[CacheDirective, ...]:
        self._buffer += text
        events: list[CacheDirective] = []
        while True:
            start = self._buffer.find("<")
            if start < 0:
                self._buffer = ""
                break
            end = self._buffer.find(">", start + 1)
            if end < 0:
                self._buffer = self._buffer[start:]
                break
            tag = self._buffer[start + 1 : end].strip()
            self._buffer = self._buffer[end + 1 :]
            event = self._parse_tag(tag)
            if event is not None:
                events.append(event)
        return tuple(events)

    def _parse_tag(self, tag: str) -> CacheDirective | None:
        lowered = tag.lower()
        if lowered == "global" or lowered in {"/focus", "/local"}:
            self.state.mode = AttentionMode.GLOBAL
            self.state.focus_segment_ids = ()
            return CacheDirective(DirectiveKind.MODE, mode=AttentionMode.GLOBAL)
        if lowered == "local":
            self.state.mode = AttentionMode.LOCAL
            self.state.focus_segment_ids = ()
            return CacheDirective(DirectiveKind.MODE, mode=AttentionMode.LOCAL)
        if lowered.startswith("focus"):
            segment_ids = self._parse_segment_ids(tag)
            self.state.mode = AttentionMode.FOCUS
            self.state.focus_segment_ids = segment_ids
            return CacheDirective(
                DirectiveKind.MODE,
                mode=AttentionMode.FOCUS,
                segment_ids=segment_ids,
            )
        if lowered.startswith("anchor"):
            return CacheDirective(
                DirectiveKind.ANCHOR,
                segment_ids=self._parse_segment_ids(tag),
            )
        if lowered.startswith("archive"):
            return CacheDirective(
                DirectiveKind.ARCHIVE,
                segment_ids=self._parse_segment_ids(tag),
            )
        return None

    def _parse_segment_ids(self, tag: str) -> tuple[int, ...]:
        match = self._segment_attribute.search(tag)
        if match is None:
            raise ValueError(f"declarative tag is missing a segment list: <{tag}>")
        segment_ids = tuple(
            dict.fromkeys(int(value.strip()) for value in match.group(1).split(","))
        )
        if not segment_ids or any(segment_id < 0 for segment_id in segment_ids):
            raise ValueError("segment IDs must be non-negative")
        return segment_ids


@dataclass(frozen=True, slots=True)
class PackedTensor:
    """A tensor stored as FP16, symmetric INT8, or packed symmetric INT4."""

    mode: CacheMode
    shape: tuple[int, ...]
    original_dtype: Any
    payload: Any | None
    scales: Any | None
    group_size: int
    original_numel: int

    @property
    def stored_bytes(self) -> int:
        return _tensor_bytes(self.payload) + _tensor_bytes(self.scales)

    def dequantize(self, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        _require_torch()
        if self.mode is CacheMode.EVICTED or self.payload is None:
            raise RuntimeError("evicted tensors cannot be materialized")
        output_dtype = self.original_dtype if dtype is None else dtype
        payload = self.payload if device is None else self.payload.to(device)
        if self.mode is CacheMode.FP16:
            return payload.to(dtype=output_dtype).reshape(self.shape)

        if self.scales is None:
            raise RuntimeError("quantized tensor is missing scales")
        scales = self.scales if device is None else self.scales.to(device)
        if self.mode is CacheMode.INT8:
            quantized = payload.to(torch.float32)
        elif self.mode is CacheMode.INT4:
            low = torch.bitwise_and(payload, 0x0F)
            high = torch.bitwise_right_shift(payload, 4)
            quantized = torch.stack((low, high), dim=-1).reshape(-1).to(torch.int16)
            quantized = torch.where(quantized >= 8, quantized - 16, quantized)
            quantized = quantized.to(torch.float32)
        else:  # pragma: no cover - CacheMode is exhaustive
            raise AssertionError(f"unsupported cache mode: {self.mode}")

        padded_numel = int(scales.numel()) * self.group_size
        values = quantized[:padded_numel].reshape(-1, self.group_size)
        values = values * scales.to(torch.float32).reshape(-1, 1)
        return values.reshape(-1)[: self.original_numel].reshape(self.shape).to(output_dtype)


def quantize_tensor(
    tensor: Any,
    mode: CacheMode,
    *,
    group_size: int = 64,
) -> PackedTensor:
    """Store a floating tensor in a concrete cache representation."""

    _require_torch()
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError("tensor must be a floating-point torch.Tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if mode is CacheMode.INT4 and group_size % 2:
        raise ValueError("INT4 group_size must be even")

    source = tensor.detach().contiguous()
    shape = tuple(int(value) for value in source.shape)
    original_numel = int(source.numel())
    if original_numel == 0:
        raise ValueError("tensor must not be empty")
    if mode is CacheMode.EVICTED:
        return PackedTensor(mode, shape, source.dtype, None, None, group_size, original_numel)
    if mode is CacheMode.FP16:
        payload = source.to(torch.float16).contiguous()
        return PackedTensor(mode, shape, source.dtype, payload, None, group_size, original_numel)

    flat = source.to(torch.float32).reshape(-1)
    padded_numel = ((original_numel + group_size - 1) // group_size) * group_size
    if padded_numel > original_numel:
        flat = torch.nn.functional.pad(flat, (0, padded_numel - original_numel))
    groups = flat.reshape(-1, group_size)
    qmax = 127 if mode is CacheMode.INT8 else 7
    maxima = groups.abs().amax(dim=1)
    scales = torch.where(maxima > 0, maxima / qmax, torch.ones_like(maxima))
    quantized = torch.round(groups / scales[:, None]).clamp(-qmax, qmax).to(torch.int8)
    scales = scales.to(torch.float16).contiguous()
    if mode is CacheMode.INT8:
        payload = quantized.reshape(-1).contiguous()
    else:
        encoded = torch.bitwise_and(quantized.to(torch.int16), 0x0F).to(torch.uint8).reshape(-1)
        payload = torch.bitwise_or(encoded[0::2], torch.bitwise_left_shift(encoded[1::2], 4))
        payload = payload.contiguous()
    return PackedTensor(mode, shape, source.dtype, payload, scales, group_size, original_numel)


def stack_legacy_cache(past_key_values: Any) -> tuple[Any, Any]:
    """Stack a Hugging Face cache into layer-first key and value tensors."""

    _require_torch()
    legacy = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    layers = tuple(legacy)
    if not layers:
        raise ValueError("past_key_values must contain at least one layer")
    if any(len(layer) != 2 for layer in layers):
        raise ValueError("each legacy cache layer must contain one key and one value tensor")
    keys = [layer[0] for layer in layers]
    values = [layer[1] for layer in layers]
    reference_shape = keys[0].shape
    if any(key.shape != reference_shape for key in keys) or any(
        value.shape != reference_shape for value in values
    ):
        raise ValueError("all legacy cache tensors must share one shape")
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


@dataclass(frozen=True, slots=True)
class KVBlock:
    block_id: int
    segment_id: int
    token_start: int
    token_end: int
    key: PackedTensor
    value: PackedTensor

    @property
    def mode(self) -> CacheMode:
        if self.key.mode is not self.value.mode:
            raise RuntimeError("key and value cache modes diverged")
        return self.key.mode

    @property
    def stored_bytes(self) -> int:
        return self.key.stored_bytes + self.value.stored_bytes

    def materialize(
        self,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> tuple[Any, Any]:
        return (
            self.key.dequantize(dtype=dtype, device=device),
            self.value.dequantize(dtype=dtype, device=device),
        )


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    segment_id: int
    role: SegmentRole
    token_start: int
    token_end: int
    protected: bool = False


@dataclass(frozen=True, slots=True)
class MaterializedCache:
    key: Any
    value: Any
    positions: Any
    block_ids: tuple[int, ...]
    segment_ids: tuple[int, ...]

    def to_legacy_cache(self) -> tuple[tuple[Any, Any], ...]:
        """Convert stacked layer tensors to Hugging Face's legacy cache tuple."""

        if self.key.ndim != 5 or self.value.ndim != 5:
            raise ValueError(
                "legacy conversion requires [layers, batch, heads, tokens, head_dim] tensors"
            )
        return tuple(
            (self.key[layer], self.value[layer])
            for layer in range(int(self.key.shape[0]))
        )


@dataclass(slots=True)
class DeclarativeKVCache:
    """Own quantized KV blocks and apply declarative visibility/precision state."""

    block_size: int = 16
    group_size: int = 64
    archive_mode: CacheMode = CacheMode.INT4
    parser: DeclarativeAttentionParser = field(default_factory=DeclarativeAttentionParser)
    _blocks: dict[int, KVBlock] = field(default_factory=dict, init=False, repr=False)
    _segments: dict[int, SegmentRecord] = field(default_factory=dict, init=False, repr=False)
    _segment_blocks: dict[int, list[int]] = field(default_factory=dict, init=False, repr=False)
    _next_block_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.group_size <= 0:
            raise ValueError("block_size and group_size must be positive")
        if self.archive_mode not in {CacheMode.INT4, CacheMode.INT8}:
            raise ValueError("archive_mode must be INT4 or INT8")

    @property
    def resident_bytes(self) -> int:
        return sum(block.stored_bytes for block in self._blocks.values())

    @property
    def full_precision_bytes(self) -> int:
        return sum(
            (block.key.original_numel + block.value.original_numel) * 2
            for block in self._blocks.values()
        )

    @property
    def compression_ratio(self) -> float:
        return self.full_precision_bytes / max(self.resident_bytes, 1)

    def add_segment(
        self,
        segment_id: int,
        key: Any,
        value: Any,
        *,
        token_start: int,
        role: SegmentRole = SegmentRole.CONTEXT,
        protected: bool = False,
    ) -> SegmentRecord:
        _require_torch()
        if segment_id in self._segments:
            raise ValueError(f"segment {segment_id} is already registered")
        if token_start < 0:
            raise ValueError("token_start must be non-negative")
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("key and value must be torch tensors")
        if key.shape != value.shape or key.ndim not in {4, 5}:
            raise ValueError(
                "key and value must share [batch, heads, tokens, head_dim] or "
                "[layers, batch, heads, tokens, head_dim] shape"
            )
        token_count = int(key.shape[-2])
        if token_count <= 0:
            raise ValueError("segments must contain at least one token")
        token_end = token_start + token_count
        for existing in self._segments.values():
            if token_start < existing.token_end and existing.token_start < token_end:
                raise ValueError("segment token ranges must not overlap")

        record = SegmentRecord(segment_id, role, token_start, token_end, protected)
        block_ids: list[int] = []
        for offset in range(0, token_count, self.block_size):
            end = min(offset + self.block_size, token_count)
            key_slice = key[..., offset:end, :]
            value_slice = value[..., offset:end, :]
            block_id = self._next_block_id
            self._next_block_id += 1
            self._blocks[block_id] = KVBlock(
                block_id=block_id,
                segment_id=segment_id,
                token_start=token_start + offset,
                token_end=token_start + end,
                key=quantize_tensor(key_slice, CacheMode.FP16, group_size=self.group_size),
                value=quantize_tensor(value_slice, CacheMode.FP16, group_size=self.group_size),
            )
            block_ids.append(block_id)
        self._segments[segment_id] = record
        self._segment_blocks[segment_id] = block_ids
        return record

    def segment_mode(self, segment_id: int) -> CacheMode:
        blocks = self._blocks_for_segment(segment_id)
        modes = {block.mode for block in blocks}
        if len(modes) != 1:
            raise RuntimeError(f"segment {segment_id} has mixed block modes")
        return next(iter(modes))

    def storage_report(self) -> dict[str, object]:
        """Return JSON-safe physical storage and state-machine statistics."""

        return {
            "full_precision_bytes": self.full_precision_bytes,
            "resident_bytes": self.resident_bytes,
            "compression_ratio": self.compression_ratio,
            "attention_mode": self.parser.state.mode.value,
            "focus_segment_ids": list(self.parser.state.focus_segment_ids),
            "visible_segment_ids": list(self.visible_segment_ids()),
            "segments": [
                {
                    "segment_id": segment_id,
                    "role": record.role.value,
                    "token_start": record.token_start,
                    "token_end": record.token_end,
                    "tokens": record.token_end - record.token_start,
                    "blocks": len(self._segment_blocks[segment_id]),
                    "mode": self.segment_mode(segment_id).value,
                    "protected": record.protected,
                    "stored_bytes": sum(
                        block.stored_bytes for block in self._blocks_for_segment(segment_id)
                    ),
                }
                for segment_id, record in sorted(
                    self._segments.items(),
                    key=lambda item: item[1].token_start,
                )
            ],
        }

    def protect_segment(self, segment_id: int) -> None:
        record = self._get_segment(segment_id)
        if self.segment_mode(segment_id) is not CacheMode.FP16:
            raise LossyPromotionError(
                "a quantized segment cannot be restored to lossless FP16; "
                "protect it before archiving"
            )
        self._segments[segment_id] = SegmentRecord(
            record.segment_id,
            record.role,
            record.token_start,
            record.token_end,
            True,
        )

    def requantize_segment(self, segment_id: int, mode: CacheMode) -> None:
        record = self._get_segment(segment_id)
        current_mode = self.segment_mode(segment_id)
        precision = {
            CacheMode.EVICTED: 0,
            CacheMode.INT4: 1,
            CacheMode.INT8: 2,
            CacheMode.FP16: 3,
        }
        if record.protected and mode is not CacheMode.FP16:
            raise ValueError(f"segment {segment_id} is protected")
        if precision[mode] > precision[current_mode]:
            raise LossyPromotionError(
                f"cannot restore {current_mode.value} segment {segment_id} to {mode.value}"
            )
        if mode is current_mode:
            return

        replacements: dict[int, KVBlock] = {}
        for block in self._blocks_for_segment(segment_id):
            if current_mode is CacheMode.EVICTED:
                raise RuntimeError("evicted segments cannot be requantized")
            key, value = block.materialize()
            replacements[block.block_id] = KVBlock(
                block.block_id,
                block.segment_id,
                block.token_start,
                block.token_end,
                quantize_tensor(key, mode, group_size=self.group_size),
                quantize_tensor(value, mode, group_size=self.group_size),
            )
        self._blocks.update(replacements)

    def apply_directive(self, directive: CacheDirective) -> None:
        if directive.kind is DirectiveKind.MODE:
            for segment_id in directive.segment_ids:
                self._get_segment(segment_id)
            return
        if directive.kind is DirectiveKind.ANCHOR:
            for segment_id in directive.segment_ids:
                self.protect_segment(segment_id)
            return
        if directive.kind is DirectiveKind.ARCHIVE:
            for segment_id in directive.segment_ids:
                self.requantize_segment(segment_id, self.archive_mode)
            return
        raise AssertionError(f"unsupported directive: {directive.kind}")

    def apply_plan(self, plan: CachePlan) -> None:
        """Apply a simulator plan to physical storage without lossy promotion."""

        targets = {segment.segment_id: segment.mode for segment in plan.segments}
        if set(targets) != set(self._segments):
            raise ValueError("cache plan and backend must contain the same segment IDs")
        precision = {
            CacheMode.EVICTED: 0,
            CacheMode.INT4: 1,
            CacheMode.INT8: 2,
            CacheMode.FP16: 3,
        }
        for segment_id, target in targets.items():
            current = self.segment_mode(segment_id)
            record = self._segments[segment_id]
            if record.protected and target is not CacheMode.FP16:
                raise ValueError(f"segment {segment_id} is protected")
            if precision[target] > precision[current]:
                raise LossyPromotionError(
                    f"plan cannot restore {current.value} segment {segment_id} to {target.value}"
                )
        for segment_id, target in targets.items():
            self.requantize_segment(segment_id, target)

    def feed_declarations(self, text: str) -> tuple[CacheDirective, ...]:
        directives = self.parser.feed(text)
        for directive in directives:
            self.apply_directive(directive)
        return directives

    def visible_segment_ids(self) -> tuple[int, ...]:
        state = self.parser.state
        always_visible = {
            segment_id
            for segment_id, record in self._segments.items()
            if record.role in {SegmentRole.SINK, SegmentRole.SCAFFOLD, SegmentRole.RESPONSE}
        }
        if state.mode is AttentionMode.GLOBAL:
            selected = set(self._segments)
        elif state.mode is AttentionMode.FOCUS:
            selected = always_visible.union(state.focus_segment_ids)
        else:
            selected = always_visible
        return tuple(
            segment_id
            for segment_id in sorted(selected, key=lambda item: self._segments[item].token_start)
            if self.segment_mode(segment_id) is not CacheMode.EVICTED
        )

    def materialize_visible(
        self,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> MaterializedCache:
        return self.materialize_segments(self.visible_segment_ids(), dtype=dtype, device=device)

    def materialize_segments(
        self,
        segment_ids: Iterable[int],
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> MaterializedCache:
        _require_torch()
        requested = tuple(dict.fromkeys(int(segment_id) for segment_id in segment_ids))
        blocks = sorted(
            (
                block
                for segment_id in requested
                for block in self._blocks_for_segment(segment_id)
                if block.mode is not CacheMode.EVICTED
            ),
            key=lambda block: block.token_start,
        )
        if not blocks:
            raise RuntimeError("no resident KV blocks were selected")
        materialized = [block.materialize(dtype=dtype, device=device) for block in blocks]
        keys, values = zip(*materialized, strict=True)
        positions = torch.cat(
            [
                torch.arange(block.token_start, block.token_end, device=keys[0].device)
                for block in blocks
            ]
        )
        return MaterializedCache(
            key=torch.cat(keys, dim=-2),
            value=torch.cat(values, dim=-2),
            positions=positions,
            block_ids=tuple(block.block_id for block in blocks),
            segment_ids=tuple(block.segment_id for block in blocks),
        )

    def _get_segment(self, segment_id: int) -> SegmentRecord:
        try:
            return self._segments[segment_id]
        except KeyError as error:
            raise KeyError(f"unknown segment {segment_id}") from error

    def _blocks_for_segment(self, segment_id: int) -> list[KVBlock]:
        self._get_segment(segment_id)
        return [self._blocks[block_id] for block_id in self._segment_blocks[segment_id]]


def _tensor_bytes(tensor: Any | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _require_torch() -> None:
    if torch is None:
        raise BackendUnavailableError(
            "the requantization backend requires the 'research' extra: "
            "pip install anchorkv[research]"
        )
