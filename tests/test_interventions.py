import unittest

import numpy as np

from anchorkv.interventions import (
    causal_attention_mask,
    downstream_window,
    select_control_indices,
)
from anchorkv.types import TokenSpan


class InterventionTests(unittest.TestCase):
    def test_builds_causal_mask_with_future_sentence_suppression(self) -> None:
        span = TokenSpan(1, 3, "anchor")

        mask = causal_attention_mask(6, blocked_span=span, query_end=5)[0, 0]

        self.assertTrue(np.isneginf(mask[0, 1]))
        self.assertEqual(mask[2, 1], 0.0)
        self.assertTrue(np.isneginf(mask[3, 1]))
        self.assertTrue(np.isneginf(mask[4, 2]))
        self.assertEqual(mask[5, 1], 0.0)

    def test_control_selection_is_distinct_and_deterministic(self) -> None:
        spans = [
            TokenSpan(0, 2),
            TokenSpan(2, 5),
            TokenSpan(5, 7),
            TokenSpan(7, 11),
        ]

        first = select_control_indices(spans, anchor_index=0, seed=7)
        second = select_control_indices(spans, anchor_index=0, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len({first.anchor_index, first.recency_index, first.length_matched_index}), 3)
        self.assertEqual(first.length_matched_index, 2)

    def test_common_window_starts_after_every_selected_span(self) -> None:
        spans = [TokenSpan(0, 2), TokenSpan(2, 5), TokenSpan(5, 7), TokenSpan(7, 9)]
        selection = select_control_indices(spans, anchor_index=1, seed=3)

        start, end = downstream_window(spans, selection, query_end=12)

        self.assertEqual(start, 9)
        self.assertEqual(end, 12)


if __name__ == "__main__":
    unittest.main()
