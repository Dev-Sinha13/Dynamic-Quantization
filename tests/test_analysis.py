import unittest

import numpy as np

from anchorkv.analysis import (
    kl_divergence_from_logits,
    ranking_diagnostics,
    receiver_head_manifest,
    score_causal_interventions,
)
from anchorkv.artifacts import AttentionTrace, TraceMetadata
from anchorkv.types import TokenSpan


class AnalysisTests(unittest.TestCase):
    def make_trace(self, sample_id: str, peak: float) -> AttentionTrace:
        metadata = TraceMetadata.create(
            model_id="test/model",
            sample_id=sample_id,
            prompt=sample_id,
            seed=7,
            dtype="float16",
            sequence_length=8,
            layers=1,
            query_heads=2,
            kv_heads=1,
            head_dim=4,
            model_revision="abc123",
        )
        spans = tuple(TokenSpan(index * 2, index * 2 + 2, str(index)) for index in range(4))
        scores = np.zeros((1, 2, 4), dtype=np.float32)
        scores[0, 0] = [0, peak, 0, 0]
        scores[0, 1] = [1, 2, 1, 2]
        return AttentionTrace(metadata, spans, scores)

    def test_builds_receiver_head_manifest(self) -> None:
        manifest = receiver_head_manifest(
            [self.make_trace("a", 8), self.make_trace("b", 9)],
            top_k=1,
        )

        strongest = manifest["receiver_heads"][0]
        self.assertEqual((strongest["layer"], strongest["query_head"]), (0, 0))
        self.assertEqual(strongest["mean_percentile"], 1.0)
        self.assertEqual(manifest["source_samples"], ["a", "b"])
        self.assertEqual(manifest["kv_heads"][0]["kv_head"], 0)

    def test_identical_logits_have_zero_kl(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0]])

        divergence = kl_divergence_from_logits(logits, logits)

        np.testing.assert_allclose(divergence, 0.0, atol=1e-12)

    def test_ranks_larger_causal_intervention_first(self) -> None:
        reference = np.array([[5.0, 0.0]])
        interventions = {
            0: np.array([[4.0, 1.0]]),
            1: np.array([[0.0, 5.0]]),
        }

        effects = score_causal_interventions(reference, interventions)

        self.assertEqual([effect.segment_id for effect in effects], [1, 0])
        self.assertGreater(effects[0].mean_kl, effects[1].mean_kl)

    def test_ranking_diagnostics_reports_agreement_and_regret(self) -> None:
        diagnostics = ranking_diagnostics(
            np.array([0.9, 0.8, 0.1, 0.0]),
            np.array([0.1, 0.2, 0.9, 0.0]),
            top_k=2,
        )

        self.assertAlmostEqual(diagnostics.spearman, 0.2)
        self.assertEqual(diagnostics.proxy_top_indices, (0, 1))
        self.assertEqual(diagnostics.causal_top_indices, (2, 1))
        self.assertEqual(diagnostics.top_k_overlap, 0.5)
        self.assertAlmostEqual(diagnostics.top_k_regret, 0.4)
        self.assertAlmostEqual(diagnostics.top_1_regret, 0.8)

    def test_ranking_diagnostics_handles_ties_and_constant_proxy(self) -> None:
        diagnostics = ranking_diagnostics(
            np.ones(3),
            np.array([0.0, 2.0, 1.0]),
            top_k=1,
        )

        self.assertEqual(diagnostics.spearman, 0.0)
        self.assertEqual(diagnostics.proxy_top_indices, (0,))
        self.assertEqual(diagnostics.causal_top_indices, (1,))
        self.assertEqual(diagnostics.top_1_regret, 2.0)


if __name__ == "__main__":
    unittest.main()
