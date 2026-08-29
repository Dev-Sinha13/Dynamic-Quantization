import unittest

from anchorkv.capacity import ModelGeometry, estimate_eager_capture


class CapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qwen_06b = ModelGeometry(
            parameters=600_000_000,
            layers=28,
            query_heads=16,
            kv_heads=8,
            head_dim=64,
        )

    def test_attention_capture_scales_quadratically(self) -> None:
        one_k = estimate_eager_capture(self.qwen_06b, sequence_length=1024)
        two_k = estimate_eager_capture(self.qwen_06b, sequence_length=2048)

        self.assertEqual(two_k.attention_bytes, one_k.attention_bytes * 4)

    def test_one_k_qwen_capture_fits_conservative_t4_limit(self) -> None:
        estimate = estimate_eager_capture(self.qwen_06b, sequence_length=1024)

        self.assertTrue(estimate.fits_conservative_limit)
        self.assertLess(estimate.lower_bound_bytes, estimate.available_bytes)

    def test_four_k_qwen_capture_exceeds_t4_limit(self) -> None:
        estimate = estimate_eager_capture(self.qwen_06b, sequence_length=4096)

        self.assertFalse(estimate.fits_conservative_limit)


if __name__ == "__main__":
    unittest.main()

