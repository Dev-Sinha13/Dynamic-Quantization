import unittest

from anchorkv.cli import run_synthetic_demo


class SyntheticDemoTests(unittest.TestCase):
    def test_demo_identifies_injected_receiver_head_and_respects_budget(self) -> None:
        report = run_synthetic_demo(0.6, seed=7)

        strongest_head = report["receiver_heads"][0]
        cache = report["cache"]

        self.assertEqual((strongest_head["layer"], strongest_head["query_head"]), (1, 2))
        self.assertLessEqual(cache["used_bytes"], cache["budget_bytes"])
        self.assertGreater(cache["compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

