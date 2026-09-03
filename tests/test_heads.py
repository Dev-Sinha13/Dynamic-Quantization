import unittest

import numpy as np

from anchorkv.heads import (
    _percentile_ranks,
    aggregate_receiver_scores_by_kv_head,
    discover_receiver_heads,
    query_head_to_kv_head,
    sentence_vertical_scores,
)
from anchorkv.types import HeadScore, TokenSpan


class HeadAnalysisTests(unittest.TestCase):
    def test_vertical_score_uses_only_future_queries(self) -> None:
        attention = np.zeros((1, 1, 6, 6), dtype=np.float64)
        attention[0, 0, 2:, 0:2] = 0.5
        spans = [TokenSpan(0, 2, "plan"), TokenSpan(2, 4, "work"), TokenSpan(4, 6, "answer")]

        scores = sentence_vertical_scores(attention, spans)

        self.assertAlmostEqual(scores[0, 0, 0], 0.5)
        self.assertEqual(scores[0, 0, 2], 0.0)

    def test_discovers_concentrated_head(self) -> None:
        traces = []
        for peak in (9.0, 10.0, 11.0):
            trace = np.zeros((1, 2, 6), dtype=np.float64)
            trace[0, 0] = [0, 0, peak, 0, 0, 0]
            trace[0, 1] = [1, 2, 1, 2, 1, 2]
            traces.append(trace)

        head = discover_receiver_heads(traces, top_k=1)[0]

        self.assertEqual((head.layer, head.query_head), (0, 0))
        self.assertEqual(head.mean_percentile, 1.0)
        self.assertGreater(head.stability, 0.9)

    def test_percentile_ranks_are_tie_aware(self) -> None:
        ranks = _percentile_ranks(np.array([[1.0, 2.0, 2.0, 4.0]]))

        np.testing.assert_allclose(ranks, [[0.0, 0.5, 0.5, 1.0]])

    def test_prefers_consistently_high_within_trace_rank(self) -> None:
        traces = []
        patterns = (
            ([0, 0, 12, 0, 0, 0], [0, 8, 0, 0, 0, 0], [1, 2, 1, 2, 1, 2]),
            ([0, 0, 11, 0, 0, 0], [1, 2, 1, 2, 1, 2], [0, 9, 0, 0, 0, 0]),
            ([0, 0, 10, 0, 0, 0], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]),
        )
        for head_zero, head_one, head_two in patterns:
            traces.append(np.array([[head_zero, head_one, head_two]], dtype=np.float64))

        head = discover_receiver_heads(traces, top_k=1)[0]

        self.assertEqual(head.query_head, 0)
        self.assertGreater(head.mean_percentile, 0.8)

    def test_maps_query_heads_to_gqa_kv_heads(self) -> None:
        mapping = [
            query_head_to_kv_head(head, num_query_heads=8, num_kv_heads=2)
            for head in range(8)
        ]

        self.assertEqual(mapping, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_rejects_degenerate_short_kurtosis_trace(self) -> None:
        trace = np.ones((1, 2, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "at least four sentences"):
            discover_receiver_heads([trace])

    def test_aggregates_receiver_scores_on_shared_kv_head(self) -> None:
        scores = [
            HeadScore(1, 0, 4.0, 0.5),
            HeadScore(1, 1, 3.0, 1.0),
            HeadScore(1, 4, 2.0, 1.0),
        ]

        aggregated = aggregate_receiver_scores_by_kv_head(
            scores,
            num_query_heads=8,
            num_kv_heads=2,
        )

        self.assertEqual(aggregated[(1, 0)], 3.0)
        self.assertEqual(aggregated[(1, 1)], 2.0)


if __name__ == "__main__":
    unittest.main()
