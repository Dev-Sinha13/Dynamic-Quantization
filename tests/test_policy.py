import unittest

from anchorkv.policy import (
    CacheBudgetError,
    CacheMode,
    DelayedAnchorTracker,
    KVGeometry,
    plan_cache,
)
from anchorkv.types import TokenSpan


class CachePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = KVGeometry(layers=2, kv_heads=2, head_dim=4)

    def test_quantized_modes_use_less_storage(self) -> None:
        int4 = self.geometry.bytes_for(16, CacheMode.INT4)
        int8 = self.geometry.bytes_for(16, CacheMode.INT8)
        bf16 = self.geometry.bytes_for(16, CacheMode.BF16)

        self.assertLess(int4, int8)
        self.assertLess(int8, bf16)

    def test_anchor_decision_is_delayed(self) -> None:
        tracker = DelayedAnchorTracker(
            min_observations=2,
            provisional_steps=2,
            anchor_threshold=0.05,
        )
        tracker.register(0, TokenSpan(0, 4, "plan"), emitted_step=0)
        tracker.observe(0, 0.1)

        early = tracker.decisions(current_step=1)[0]
        tracker.observe(0, 0.1)
        mature = tracker.decisions(current_step=2)[0]

        self.assertTrue(early.provisional)
        self.assertFalse(early.anchor)
        self.assertFalse(mature.provisional)
        self.assertTrue(mature.anchor)

    def test_plan_respects_budget_and_protects_anchor(self) -> None:
        tracker = DelayedAnchorTracker(
            min_observations=1,
            provisional_steps=0,
            anchor_threshold=0.05,
        )
        tracker.register(0, TokenSpan(0, 4, "plan"), emitted_step=0)
        tracker.register(1, TokenSpan(4, 8, "detour"), emitted_step=1)
        tracker.observe(0, 0.2)
        tracker.observe(1, 0.01)
        anchor_bytes = self.geometry.bytes_for(4, CacheMode.BF16)

        plan = plan_cache(
            tracker.states(),
            tracker.decisions(current_step=4),
            geometry=self.geometry,
            budget_bytes=anchor_bytes,
            current_step=4,
            recent_window=0,
            recency_weight=0.0,
        )

        modes = {segment.segment_id: segment.mode for segment in plan.segments}
        self.assertEqual(modes[0], CacheMode.BF16)
        self.assertEqual(modes[1], CacheMode.EVICTED)
        self.assertLessEqual(plan.used_bytes, plan.budget_bytes)

    def test_reports_impossible_protected_budget(self) -> None:
        tracker = DelayedAnchorTracker(min_observations=2, provisional_steps=2)
        tracker.register(0, TokenSpan(0, 4, "new step"), emitted_step=0)

        with self.assertRaises(CacheBudgetError):
            plan_cache(
                tracker.states(),
                tracker.decisions(current_step=0),
                geometry=self.geometry,
                budget_bytes=1,
                current_step=0,
            )


if __name__ == "__main__":
    unittest.main()

