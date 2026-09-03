import unittest

import numpy as np

from anchorkv.hf import (
    _ends_with_eos,
    _render_chat_prompt,
    decoded_token_offsets,
    reduce_attention_layers,
)
from anchorkv.types import TokenSpan


class FakeTokenizer:
    pieces = {1: "Plan", 2: ". ", 3: "Solve", 4: "."}

    def decode(self, token_ids, **_kwargs):
        return self.pieces[token_ids[0]]


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.enable_thinking = None

    def apply_chat_template(self, _messages, **kwargs):
        self.enable_thinking = kwargs["enable_thinking"]
        return "rendered"


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

    def test_disables_unbounded_thinking_for_bounded_capture(self) -> None:
        tokenizer = FakeChatTokenizer()

        rendered = _render_chat_prompt(tokenizer, "problem", enable_thinking=False)

        self.assertEqual(rendered, "rendered")
        self.assertFalse(tokenizer.enable_thinking)

    def test_detects_eos_for_scalar_and_multiple_stop_tokens(self) -> None:
        self.assertTrue(_ends_with_eos([10, 20], 20))
        self.assertTrue(_ends_with_eos([10, 21], [20, 21]))
        self.assertFalse(_ends_with_eos([10, 22], [20, 21]))
        self.assertFalse(_ends_with_eos([], 20))

    def test_requires_enough_spans_for_kurtosis(self) -> None:
        from anchorkv.hf import HFTraceConfig

        with self.assertRaisesRegex(ValueError, "at least four"):
            HFTraceConfig(min_reasoning_spans=3)


if __name__ == "__main__":
    unittest.main()
