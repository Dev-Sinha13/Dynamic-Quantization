import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_requantization_notebook_is_standalone_and_compiles(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "AnchorKV_T4_Requantization.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        all_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertNotIn("git clone", all_source)
        self.assertNotIn("from anchorkv", all_source)
        self.assertIn("Qwen/Qwen3-0.6B", all_source)
        self.assertIn("MAX_PROMPT_TOKENS = 640", all_source)
        self.assertIn("MAX_NEW_TOKENS = 32", all_source)
        self.assertIn("TEACHER_FORCED_STEPS = 24", all_source)
        self.assertIn("torch.bitwise_left_shift", all_source)
        self.assertIn("DynamicCache(cache_data)", all_source)
        self.assertIn("<anchor segments=", all_source)
        self.assertIn("<archive segments=", all_source)
        self.assertIn("teacher_mean_kl", all_source)
        self.assertIn("peak_gpu_gib_dense_reference", all_source)
        self.assertIn("requantization-results.json", all_source)
        self.assertIn("model_info(MODEL_ID).sha", all_source)

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            if source.startswith("%pip "):
                source = "# " + source
            compile(source, f"{path}:cell-{index}", "exec")

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
        self.assertIn("max_new_tokens=512", all_source)
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
        self.assertIn("MAX_NEW_TOKENS = 512", all_source)
        self.assertIn("enable_thinking=True", all_source)
        self.assertIn("MIN_REASONING_SPANS = 6", all_source)
        self.assertIn("MIN_FUTURE_TOKENS = 32", all_source)
        self.assertIn("EXPECTED_ANSWER_TERMS", all_source)
        self.assertIn("return_offsets_mapping=True", all_source)
        self.assertIn("did not round-trip to the generated token IDs", all_source)
        self.assertIn("query_end=query_end", all_source)
        self.assertIn(
            "end + prompt_tokens <= query_end - min_future_tokens",
            all_source,
        )
        self.assertIn("head_dim = 600_000_000, 28, 16, 8, 128", all_source)
        self.assertIn("do not use this truncated trace", all_source)
        self.assertIn("run-summary.json", all_source)
        self.assertIn("'peak_gpu_gib'", all_source)
        self.assertIn("causal-results.json", all_source)
        self.assertIn("all_candidate_teacher_forced_all_head_sentence_suppression", all_source)
        self.assertIn("attention_mask=mask", all_source)
        self.assertIn("ranking_diagnostics", all_source)
        self.assertIn("cross_head_normalized", all_source)
        self.assertIn("top_k_regret", all_source)
        self.assertIn("evaluation_start = int(np.max(ends))", all_source)
        self.assertIn("attention-causal-scatter.png", all_source)
        self.assertIn("mask_sanity_max_abs_logit", all_source)
        self.assertIn("files.download(archive)", all_source)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
