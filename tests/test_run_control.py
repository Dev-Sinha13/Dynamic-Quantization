import json
from pathlib import Path
import tempfile
import unittest

from anchorkv.run_control import RunControl


class RunControlTests(unittest.TestCase):
    def test_atomic_resume_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RunControl(directory, {'seed': 7})
            controller.save('rows.json', [{'done': 1}])
            resumed = RunControl(directory, {'seed': 7})
            self.assertEqual(resumed.load('rows.json'), [{'done': 1}])
            with self.assertRaises(ValueError):
                RunControl(directory, {'seed': 19})
            self.assertEqual(resumed.load('rows.json'), [{'done': 1}])
            with self.assertRaises(ValueError):
                resumed.save('rows.json', [float('nan')])
            self.assertEqual(resumed.load('rows.json'), [{'done': 1}])

    def test_time_budget_is_cooperative_and_resets_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0]
            controller = RunControl(directory, {}, clock=lambda: now[0])
            controller.start('quality', 1)
            self.assertTrue(controller.allow())
            now[0] = 60
            self.assertFalse(controller.allow())
            self.assertEqual(controller.load('quality-status.json')['status'], 'paused_time_budget')
            controller.start('quality', 1)
            self.assertTrue(controller.allow())

    def test_quick_notebook_defaults_and_legacy_guard(self):
        root = Path(__file__).parents[1]
        notebook = json.loads((root / 'notebooks/AnchorKV_T4_All_In_One.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn("PROFILE = 'quick'", code)
        self.assertIn('RUN_LEGACY_PILOT = False', code)
        self.assertIn('if RUN_LEGACY_PILOT:\n    raw_rows = []', code)
        self.assertIn('settings.max_new_tokens = 16', code)
        self.assertIn('QUALITY_MINUTES = 10', code)


if __name__ == '__main__':
    unittest.main()
