import unittest

from anchorkv.segmentation import sentence_character_spans, token_spans_from_offsets


class SegmentationTests(unittest.TestCase):
    def test_splits_punctuation_and_newline_steps(self) -> None:
        text = "Plan the solution.\nCompute x\nTherefore, x is 3."

        spans = sentence_character_spans(text)
        sentences = [text[start:end] for start, end in spans]

        self.assertEqual(
            sentences,
            ["Plan the solution.", "Compute x", "Therefore, x is 3."],
        )

    def test_maps_character_spans_to_tokens(self) -> None:
        text = "Plan. Solve."
        offsets = [(0, 4), (4, 5), (6, 11), (11, 12)]

        spans = token_spans_from_offsets(text, offsets)

        self.assertEqual([(span.start, span.end) for span in spans], [(0, 2), (2, 4)])
        self.assertEqual([span.text for span in spans], ["Plan.", "Solve."])


if __name__ == "__main__":
    unittest.main()

