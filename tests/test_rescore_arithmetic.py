import unittest
from anchorkv.rescore_arithmetic import extract_final_integer


class FinalAnswerTests(unittest.TestCase):
    def test_archive_scoring_preserves_strict_scores_and_requires_completion(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from zipfile import ZipFile
        from anchorkv.rescore_arithmetic import rescore
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'input.zip'
            rows = [dict(case_id='a', variant=f'random-{i}', policy='random', random_seed=i,
                         text='Explanation\n202', answer_correct=False, completed_correct=False,
                         ended_eos=i == 0, truncated=i != 0) for i in range(2)]
            with ZipFile(archive, 'w') as bundle:
                bundle.writestr('expanded-prompts.json', json.dumps([dict(case_id='a', task='arithmetic', answer='202')]))
                bundle.writestr('expanded-quality.json', json.dumps(rows))
                bundle.writestr('environment.json', '{}')
            before = archive.read_bytes()
            output = Path(directory) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                result = rescore(archive, output)
            self.assertEqual(archive.read_bytes(), before)
            self.assertEqual((output / 'original-results.zip').read_bytes(), before)
            self.assertEqual(result['summary'][0]['strict_correct'], 0)
            self.assertEqual(result['summary'][0]['final_correct'], 1)
            self.assertEqual(result['records'][1]['review_status'], 'incomplete')
            with self.assertRaises(ValueError):
                rescore(archive, output)

    def test_explicit_final_answers(self):
        for text, expected in [
            ('Work: 493 - 10 = 483\n\n483 units remain in the Zephyr shipment.', '483'),
            ('Wrong earlier value 221.\nReturn only the integer: 202.', '202'),
            ('The integer is: 221.', '221'),
            ('Explanation\n483', '483'),
            ('493 - 10000 = 493.', '493'),
            ('Answer: -17.', '-17'),
        ]:
            self.assertEqual(extract_final_integer(text)[0], expected)

    def test_does_not_search_for_gold_or_take_arbitrary_last_number(self):
        for text in ('', 'The gold might be 202 but I cannot answer.',
                     '17 units were removed from the Zephyr shipment.',
                     '221 units. 22  221 221.', '493 - 100 493.', '483 or 202', '483.5'):
            self.assertIsNone(extract_final_integer(text)[0], text)


if __name__ == '__main__':
    unittest.main()
