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
        self.assertIn("model_info(MODEL_ID).sha", all_source)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()

