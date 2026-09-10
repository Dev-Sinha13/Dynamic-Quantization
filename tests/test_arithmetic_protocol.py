import unittest
from anchorkv.arithmetic_protocol import build_protocol, development_gate, render_case


class ArithmeticProtocolTests(unittest.TestCase):
    def test_splits_gold_and_matched_contexts(self):
        splits = build_protocol()
        self.assertEqual(splits, build_protocol())
        self.assertEqual(len(splits['development']), 12)
        self.assertEqual(len(splits['heldout']), 48)
        keys = {}
        for split, cases in splits.items():
            keys[split] = set()
            for case in cases:
                a, b, c = case['operands']
                expected = {'addition': a + b, 'subtraction': a - b, 'multiply_subtract': a * b - c}[case['operation']]
                self.assertEqual(int(case['answer']), expected)
                for fact in case['evidence_text']:
                    self.assertEqual(case['messages'][1]['content'].count(fact), 1)
                keys[split].add((case['operation'], a, b, c))
            self.assertEqual(len({c['case_id'] for c in cases}), len(cases))
        self.assertFalse(keys['development'] & keys['heldout'])

    def test_readiness_gate_rejects_incomplete_duplicate_or_filtered_data(self):
        cases = build_protocol()['development']
        rows = [{'case_id': c['case_id'], 'policy': 'native_fp16', 'messages_sha256': c['messages_sha256'],
                 'text': c['answer'], 'ended_eos': True, 'truncated': False} for c in cases]
        self.assertEqual(development_gate(cases, rows)['status'], 'passed')
        self.assertEqual(development_gate(cases, rows[:-1])['status'], 'incomplete')
        rows[0]['text'] = 'invalid'
        self.assertEqual(development_gate(cases, rows)['status'], 'passed')
        rows[1]['text'] = 'invalid'
        self.assertEqual(development_gate(cases, rows)['status'], 'failed')
        with self.assertRaises(ValueError):
            development_gate(cases[2:], rows[2:])
        with self.assertRaises(ValueError):
            development_gate(cases, rows + [rows[0]])

    def test_rendered_token_bounds_using_cached_tokenizer(self):
        from pathlib import Path
        from anchorkv.arithmetic_protocol import MODEL_REVISION
        path = Path(__file__).parents[1] / '.cache/hf/models--Qwen--Qwen3-0.6B/snapshots' / MODEL_REVISION
        if not path.exists():
            self.skipTest('optional local tokenizer absent')
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('optional transformers absent')
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        for cases in build_protocol().values():
            for case in cases:
                rendered = render_case(case, tokenizer)
                self.assertLessEqual(len(rendered['ids']), 1536)
                self.assertEqual(tokenizer(rendered['prompt'], add_special_tokens=False).input_ids, rendered['ids'])
        with self.assertRaises(ValueError):
            render_case(build_protocol()['development'][0], tokenizer, max_prompt_tokens=1)


if __name__ == '__main__':
    unittest.main()
