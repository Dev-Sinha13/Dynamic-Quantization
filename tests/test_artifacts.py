import tempfile
import unittest
from pathlib import Path

import numpy as np

from anchorkv.artifacts import AttentionTrace, TraceMetadata, load_trace, save_trace, trace_summary
from anchorkv.types import TokenSpan


class ArtifactTests(unittest.TestCase):
    def make_trace(self) -> AttentionTrace:
        metadata = TraceMetadata.create(
            model_id="Qwen/Qwen3-0.6B",
            sample_id="sample-001",
            prompt="Solve 2 + 2.",
            seed=7,
            dtype="float16",
            sequence_length=6,
            layers=2,
            query_heads=4,
            kv_heads=2,
            head_dim=8,
            model_revision="test-revision",
        )
        spans = (TokenSpan(0, 3, "Plan."), TokenSpan(3, 6, "Answer."))
        scores = np.arange(16, dtype=np.float32).reshape(2, 4, 2) / 100
        return AttentionTrace(metadata, spans, scores)

    def test_round_trip_without_pickle(self) -> None:
        original = self.make_trace()
        with tempfile.TemporaryDirectory() as directory:
            path = save_trace(original, Path(directory) / "trace")
            restored = load_trace(path)

        self.assertEqual(restored.metadata, original.metadata)
        self.assertEqual(restored.spans, original.spans)
        np.testing.assert_array_equal(restored.vertical_scores, original.vertical_scores)

    def test_summary_exposes_reproducibility_fields(self) -> None:
        summary = trace_summary(self.make_trace())

        self.assertEqual(summary["model_id"], "Qwen/Qwen3-0.6B")
        self.assertEqual(summary["model_revision"], "test-revision")
        self.assertEqual(summary["sentences"], 2)

    def test_rejects_mismatched_score_shape(self) -> None:
        trace = self.make_trace()

        with self.assertRaises(ValueError):
            AttentionTrace(trace.metadata, trace.spans, np.zeros((1, 1, 1), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

