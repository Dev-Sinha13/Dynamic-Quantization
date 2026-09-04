import unittest

try:
    import torch
except ImportError:  # pragma: no cover - base installs intentionally omit torch
    torch = None

from anchorkv.backend import (
    AttentionMode,
    DeclarativeAttentionParser,
    DeclarativeKVCache,
    DirectiveKind,
    LossyPromotionError,
    SegmentRole,
    quantize_tensor,
)
from anchorkv.policy import CacheMode, CachePlan, PlannedSegment


@unittest.skipIf(torch is None, "PyTorch research dependency is not installed")
class QuantizedTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.tensor = torch.randn(1, 2, 10, 8, dtype=torch.float32)

    def test_int8_round_trip_uses_real_quantized_storage(self) -> None:
        packed = quantize_tensor(self.tensor, CacheMode.INT8, group_size=16)
        restored = packed.dequantize()

        self.assertEqual(packed.payload.dtype, torch.int8)
        self.assertEqual(packed.scales.dtype, torch.float16)
        self.assertEqual(restored.shape, self.tensor.shape)
        self.assertLess(float((restored - self.tensor).abs().max()), 0.02)
        self.assertLess(packed.stored_bytes, self.tensor.numel() * 2)

    def test_int4_is_nibble_packed_and_reconstructs_shape(self) -> None:
        int8 = quantize_tensor(self.tensor, CacheMode.INT8, group_size=16)
        int4 = quantize_tensor(self.tensor, CacheMode.INT4, group_size=16)
        restored = int4.dequantize()

        self.assertEqual(int4.payload.dtype, torch.uint8)
        self.assertEqual(int4.payload.numel(), 80)
        self.assertEqual(restored.shape, self.tensor.shape)
        self.assertLess(float((restored - self.tensor).abs().max()), 0.3)
        self.assertLess(int4.stored_bytes, int8.stored_bytes)

    def test_zero_groups_round_trip_exactly(self) -> None:
        zeros = torch.zeros(1, 1, 4, 8)

        restored = quantize_tensor(zeros, CacheMode.INT4, group_size=16).dequantize()

        torch.testing.assert_close(restored, zeros)

    def test_evicted_tensor_cannot_be_materialized(self) -> None:
        packed = quantize_tensor(self.tensor, CacheMode.EVICTED)

        self.assertEqual(packed.stored_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "evicted"):
            packed.dequantize()


class DeclarativeParserTests(unittest.TestCase):
    def test_parses_tags_split_across_decode_chunks(self) -> None:
        parser = DeclarativeAttentionParser()

        self.assertEqual(parser.feed("reason <fo"), ())
        focus = parser.feed('cus magic_chunks="2, 3">inspect')
        closing = parser.feed("</focus><local>")
        anchor = parser.feed("<anchor segments='4, 4'>")

        self.assertEqual(focus[0].mode, AttentionMode.FOCUS)
        self.assertEqual(focus[0].segment_ids, (2, 3))
        self.assertEqual(closing[0].mode, AttentionMode.GLOBAL)
        self.assertEqual(closing[1].mode, AttentionMode.LOCAL)
        self.assertEqual(anchor[0].kind, DirectiveKind.ANCHOR)
        self.assertEqual(anchor[0].segment_ids, (4,))
        self.assertEqual(parser.state.mode, AttentionMode.LOCAL)

    def test_rejects_recognized_tag_without_segment_ids(self) -> None:
        parser = DeclarativeAttentionParser()

        with self.assertRaisesRegex(ValueError, "missing a segment list"):
            parser.feed("<focus>")


@unittest.skipIf(torch is None, "PyTorch research dependency is not installed")
class DeclarativeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.cache = DeclarativeKVCache(block_size=4, group_size=16)
        self.tensors = {}
        specifications = [
            (0, 0, 4, SegmentRole.SINK),
            (1, 4, 6, SegmentRole.CONTEXT),
            (2, 10, 6, SegmentRole.CONTEXT),
            (3, 16, 4, SegmentRole.RESPONSE),
        ]
        for segment_id, token_start, token_count, role in specifications:
            key = torch.randn(1, 2, token_count, 8)
            value = torch.randn(1, 2, token_count, 8)
            self.tensors[segment_id] = (key, value)
            self.cache.add_segment(
                segment_id,
                key,
                value,
                token_start=token_start,
                role=role,
            )

    def test_archive_reduces_physical_storage(self) -> None:
        before = self.cache.resident_bytes

        self.cache.requantize_segment(1, CacheMode.INT4)

        self.assertEqual(self.cache.segment_mode(1), CacheMode.INT4)
        self.assertLess(self.cache.resident_bytes, before)
        self.assertGreater(self.cache.compression_ratio, 1.0)

    def test_precision_loss_is_not_reported_as_reversible(self) -> None:
        self.cache.requantize_segment(1, CacheMode.INT4)

        with self.assertRaises(LossyPromotionError):
            self.cache.requantize_segment(1, CacheMode.FP16)
        with self.assertRaises(LossyPromotionError):
            self.cache.protect_segment(1)

    def test_protected_anchor_cannot_be_archived(self) -> None:
        self.cache.feed_declarations('<anchor segments="2">')

        with self.assertRaisesRegex(ValueError, "protected"):
            self.cache.feed_declarations('<archive segments="2">')

    def test_focus_and_local_modes_control_visible_blocks(self) -> None:
        self.cache.requantize_segment(1, CacheMode.INT4)
        self.cache.feed_declarations('<focus segments="1">')

        focused = self.cache.materialize_visible(dtype=torch.float32)

        self.assertEqual(self.cache.visible_segment_ids(), (0, 1, 3))
        self.assertEqual(focused.key.shape[-2], 14)
        self.assertEqual(focused.positions.tolist(), list(range(0, 10)) + list(range(16, 20)))
        original_key = self.tensors[1][0]
        restored_context = focused.key[..., 4:10, :]
        self.assertLess(float((restored_context - original_key).abs().max()), 0.4)

        self.cache.feed_declarations("</focus><local>")
        self.assertEqual(self.cache.visible_segment_ids(), (0, 3))

    def test_evict_removes_segment_from_materialization(self) -> None:
        self.cache.requantize_segment(1, CacheMode.EVICTED)
        self.cache.feed_declarations('<focus segments="1">')

        self.assertEqual(self.cache.visible_segment_ids(), (0, 3))

    def test_applies_cache_plan_to_physical_blocks(self) -> None:
        modes = [CacheMode.FP16, CacheMode.INT4, CacheMode.INT8, CacheMode.FP16]
        segments = tuple(
            PlannedSegment(index, mode, 0, 0.0, "test")
            for index, mode in enumerate(modes)
        )
        plan = CachePlan(segments, budget_bytes=1, used_bytes=0, full_cache_bytes=0)

        self.cache.apply_plan(plan)

        self.assertEqual(
            [self.cache.segment_mode(index) for index in range(4)],
            modes,
        )


if __name__ == "__main__":
    unittest.main()
