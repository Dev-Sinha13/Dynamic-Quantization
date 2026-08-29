import unittest

import numpy as np

from anchorkv.hf import decoded_token_offsets, reduce_attention_layers
from anchorkv.types import TokenSpan


class FakeTokenizer:
    pieces = {1: "Plan", 2: ". ", 3: "Solve", 4: "."}

    def decode(self, token_ids, **_kwargs):
        return self.pieces[token_ids[0]]


class HuggingFaceHelperTests(unittest.TestCase):
    def test_builds_offsets_from_decoded_token_pieces(self) -> None:
        text, offsets = decoded_token_offsets(FakeTokenizer(), [1, 2, 3, 4])

        self.assertEqual(text, "Plan. Solve.")
        self.assertEqual(offsets, [(0, 4), (4, 6), (6, 11), (11, 12)])

    def test_reduces_attention_without_stacking_token_matrices(self) -> None:
        layer_zero = np.zeros((1, 2, 6, 6), dtype=np.float32)
        layer_one = np.zeros((1, 2, 6, 6), dtype=np.float32)
        layer_zero[0, 0, 2:, 0:2] = 0.5
        layer_one[0, 1, 4:, 2:4] = 0.25
        spans = [TokenSpan(0, 2), TokenSpan(2, 4), TokenSpan(4, 6)]

        reduced = reduce_attention_layers([layer_zero, layer_one], spans)

        self.assertEqual(reduced.shape, (2, 2, 3))
        self.assertAlmostEqual(float(reduced[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(reduced[1, 1, 1]), 0.25)
        self.assertEqual(float(reduced[0, 0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()

