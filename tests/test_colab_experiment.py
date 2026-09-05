import gc
import unittest
import weakref

try:
    import torch
    from anchorkv.colab_experiment import (
        Settings, answer_check, attention_route, cache_cpu, forward_token, make_live_cache,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch dependency is unavailable')
class ExperimentTests(unittest.TestCase):
    def test_answers_require_exact_text_and_track_completion(self):
        self.assertTrue(answer_check(' 7319\n', '7319', True)['completed_correct'])
        self.assertFalse(answer_check('7319\nAnswer: 7319', '7319', True)['answer_correct'])
        self.assertFalse(answer_check('7319', '7319', False)['completed_correct'])
        self.assertTrue(answer_check('7319', '7319', False)['truncated'])

    def test_real_tiny_qwen_routes_cache_and_releases_registry_reference(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest('optional transformers dependency is unavailable')
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=100, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=64)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).half().eval()
        with torch.inference_mode():
            ids = torch.randint(0, 100, (1, 53))
            prefill = model(input_ids=ids, use_cache=True)
            source = cache_cpu(prefill.past_key_values)
            original = forward_token(model, prefill.past_key_values, 11).logits
            live = make_live_cache(source, Settings(), bits=16, device='cpu')
            with attention_route(model, live, 'dense'):
                actual = forward_token(model, live, 11, packed=True).logits
            torch.testing.assert_close(actual, original, atol=0.001, rtol=0.001)
            self.assertEqual(model.config._attn_implementation, 'sdpa')
            cache_ref = weakref.ref(live)
            del live
            gc.collect()
            self.assertIsNone(cache_ref(), 'the global attention registry retained a GPU-cache closure')
            mixed = make_live_cache(source, Settings(recent_pages=1), bits=4, device='cpu')
            with attention_route(model, mixed, 'dense'):
                quantized = forward_token(model, mixed, 11, packed=True).logits
            self.assertTrue(torch.isfinite(quantized).all())
            self.assertGreater(mixed.report()['demotions'], 0)


if __name__ == '__main__':
    unittest.main()
