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
    def test_expanded_prompts_with_cached_pinned_qwen_tokenizer(self):
        from pathlib import Path
        root = Path(__file__).parents[1]
        snapshot = root / '.cache/hf/models--Qwen--Qwen3-0.6B/snapshots' / Settings().revision
        if not snapshot.is_dir():
            self.skipTest('optional pinned tokenizer cache is unavailable')
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('optional transformers dependency is unavailable')
        from anchorkv.benchmark_suite import SuiteSettings, build_quality_suite
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        cases = build_quality_suite(tokenizer, Settings(), SuiteSettings())
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({c['family_id'] for c in cases}), 6)
        for case in cases:
            self.assertLessEqual(len(case['ids']), case['target_length'])
            self.assertGreater(len(case['ids']), case['target_length'] - 100)
            self.assertEqual(tokenizer(case['prompt'], add_special_tokens=False).input_ids, case['ids'])
            self.assertTrue(case['evidence_pages'])
            self.assertLess(max(case['evidence_pages']), (len(case['ids']) + 15) // 16)
            gold = tokenizer(case['answer'], add_special_tokens=False).input_ids
            self.assertEqual(tokenizer.decode(gold), case['answer'])
            self.assertLess(len(gold) + 1, Settings().max_new_tokens)

    def test_controlled_decode_uses_exact_history_and_ages_cache(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest('optional transformers dependency is unavailable')
        from anchorkv.benchmark_suite import controlled_decode
        torch.set_num_threads(2)
        config = Qwen3Config(vocab_size=100, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=64)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).half().eval()
        with torch.inference_mode():
            output = model(input_ids=torch.randint(0, 100, (1, 53)), use_cache=True)
            source = cache_cpu(output.past_key_values)
            del output
        inputs = list(range(48))
        seen = []
        def capture(module, args, kwargs):
            seen.extend(kwargs['input_ids'][0].tolist())
        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            rows = [controlled_decode(model, source, Settings(recent_pages=1), policy,
                                      inputs, 40, warmup_tokens=8, protected={0}, backend='dense')
                    for policy in ('native_fp16', 'paged_fp16', 'int4')]
        finally:
            handle.remove()
        self.assertEqual(seen, inputs * 3)
        for row in rows:
            self.assertEqual(row['final_cache']['tokens'], 101)
            self.assertEqual(row['measurement_start_cache']['tokens'], 61)
            self.assertGreater(row['steady_tokens_per_second'], 0)
            self.assertIsNone(row['measured_cuda_timeline_seconds'])
            self.assertNotIn('answer_correct', row)
        self.assertGreater(rows[-1]['new_measured_demotions'], 0)
        self.assertLess(rows[-1]['final_cache']['resident_bytes'], rows[0]['final_cache']['resident_bytes'])
        self.assertEqual(len({r['input_sha256'] for r in rows}), 1)
        self.assertEqual(model.config._attn_implementation, 'sdpa')
        with self.assertRaises(ValueError):
            controlled_decode(model, source, Settings(), 'int4', [1], 128)

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
