import unittest

from anchorkv.cli import run_requantization_demo, run_synthetic_demo
from anchorkv.policy import CacheMode

try:
    import torch
except ImportError:  # pragma: no cover - base installs intentionally omit torch
    torch = None


class SyntheticDemoTests(unittest.TestCase):
    def test_demo_identifies_injected_receiver_head_and_respects_budget(self) -> None:
        report = run_synthetic_demo(0.6, seed=7)

        strongest_head = report["receiver_heads"][0]
        cache = report["cache"]

        self.assertEqual((strongest_head["layer"], strongest_head["query_head"]), (1, 2))
        self.assertLessEqual(cache["used_bytes"], cache["budget_bytes"])
        self.assertGreater(cache["compression_ratio"], 1.0)

    @unittest.skipIf(torch is None, "PyTorch research dependency is not installed")
    def test_requantization_demo_reports_real_storage_and_error(self) -> None:
        report = run_requantization_demo(archive_mode=CacheMode.INT4, seed=7)

        self.assertEqual(report["archive_mode"], "int4")
        self.assertEqual(report["attention_mode"], "focus")
        self.assertEqual(report["visible_segment_ids"], [0, 1, 3, 4])
        self.assertLess(report["resident_bytes"], report["full_precision_bytes"])
        self.assertGreater(report["compression_ratio"], 1.0)
        self.assertLess(report["max_abs_error"], 0.5)


if __name__ == "__main__":
    unittest.main()
