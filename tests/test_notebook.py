import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_colab_notebook_is_clean_and_t4_bounded(self) -> None:
        path = Path(__file__).parents[1] / "notebooks" / "AnchorKV_T4_Trace_Collection.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        all_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("Qwen/Qwen3-0.6B", all_source)
        self.assertIn("max_sequence_length=768", all_source)
        self.assertIn("max_new_tokens=320", all_source)
        self.assertIn("enable_thinking=True", all_source)
        self.assertIn("min_reasoning_spans=6", all_source)
        self.assertIn("EXPECTED_ANSWER_TERMS", all_source)
        self.assertIn("head_dim=128", all_source)
        self.assertIn("model_info(MODEL_ID).sha", all_source)
        self.assertIn("run-summary.json", all_source)
        self.assertIn("'peak_gpu_gib'", all_source)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_standalone_colab_notebook_needs_no_repository_access(self) -> None:
        path = Path(__file__).parents[1] / "notebooks" / "AnchorKV_T4_Standalone.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        all_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertNotIn("git clone", all_source)
        self.assertNotIn("from anchorkv", all_source)
        self.assertIn("Qwen/Qwen3-0.6B", all_source)
        self.assertIn("MAX_SEQUENCE_LENGTH = 768", all_source)
        self.assertIn("MAX_NEW_TOKENS = 320", all_source)
        self.assertIn("enable_thinking=True", all_source)
        self.assertIn("MIN_REASONING_SPANS = 6", all_source)
        self.assertIn("EXPECTED_ANSWER_TERMS", all_source)
        self.assertIn("head_dim = 600_000_000, 28, 16, 8, 128", all_source)
        self.assertIn("do not use this truncated trace", all_source)
        self.assertIn("run-summary.json", all_source)
        self.assertIn("'peak_gpu_gib'", all_source)
        self.assertIn("files.download(archive)", all_source)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
