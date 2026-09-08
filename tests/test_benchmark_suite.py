import unittest

from anchorkv.benchmark_suite import (
    SuiteSettings, build_quality_case, paired_family_differences, task_spec,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages) + '\nASSISTANT:'

    def __call__(self, text, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=list(map(ord, text)),
                               offset_mapping=[(i, i + 1) for i in range(len(text))])


class QualitySuiteTests(unittest.TestCase):
    def test_complete_deterministic_prompts_and_matched_families(self):
        from types import SimpleNamespace
        settings = SimpleNamespace(max_prompt_tokens=2000, block=16)
        for task in SuiteSettings().tasks:
            variants = [build_quality_case(CharacterTokenizer(), settings, task, 7, 1400, pos)
                        for pos in ('early', 'middle', 'late')]
            self.assertEqual(len({c['answer'] for c in variants}), 1)
            self.assertEqual(len({c['family_id'] for c in variants}), 1)
            self.assertEqual(len({c['prompt_sha256'] for c in variants}), 3)
            for case in variants:
                self.assertLessEqual(len(case['ids']), 1400)
                self.assertTrue(case['prompt'].endswith('ASSISTANT:'))
                self.assertTrue(case['evidence_pages'])
                for fact, (a, b) in zip(case['evidence_text'], case['evidence_spans']):
                    self.assertEqual(case['prompt'][a:b], fact)
            self.assertEqual(variants[0], build_quality_case(
                CharacterTokenizer(), settings, task, 7, 1400, 'early'))
        with self.assertRaises(ValueError):
            build_quality_case(CharacterTokenizer(), settings, 'arithmetic', 7, 20, 'early')

    def test_arithmetic_gold_derived_from_all_three_facts(self):
        import re
        for seed in (7, 19, 43):
            evidence, _, answer = task_spec('arithmetic', seed)
            a, b, c = [int(re.search(r'\d+', fact).group()) for fact in evidence]
            self.assertEqual(int(answer), a * b - c)

    def test_repeated_draws_and_matched_positions_are_not_independent_samples(self):
        rows = []
        for case in ('early', 'late'):
            for policy, values in [('automatic', [4]), ('random', [1, 3, 5])]:
                for value in values:
                    rows.append(dict(family_id='a', case_id=case, policy=policy,
                                     budget=.25, mean_kl=value))
        self.assertEqual(paired_family_differences(rows, 'random', 'mean_kl', .25), {'a': 1})
        with self.assertRaises(ValueError):
            paired_family_differences(rows[:-3], 'random', 'mean_kl', .25)

    def test_all_selector_draws_have_equal_page_budget(self):
        try:
            from anchorkv.benchmark_suite import quality_plans
            plans = quality_plans(list(range(1, 31)), {i: i / 100 for i in range(1, 31)},
                                  [5, 6], SuiteSettings())
        except ImportError:
            self.skipTest('optional torch dependency is unavailable')
        self.assertEqual(len(plans), 16)
        for budget in (.10, .25):
            matched = [p for p in plans if p['budget'] == budget]
            self.assertEqual(len(matched), 6)
            self.assertEqual({len(p['protected']) for p in matched}, {1 + int(30 * budget)})
            self.assertTrue(all(0 in p['protected'] for p in matched))


if __name__ == '__main__':
    unittest.main()
