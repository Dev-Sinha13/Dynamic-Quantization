"""Exercise notebook orchestration/export with fake GPU work, not GPU validation."""

import contextlib
import io
from pathlib import Path
import random
import statistics
import tempfile
import time
from types import SimpleNamespace
import unittest


class ExpandedNotebookFlowTests(unittest.TestCase):
    def test_quality_decode_and_report_cells_export_complete_bundle(self):
        try:
            import pandas as pd
            import torch
            from anchorkv.colab_experiment import Settings, bootstrap_interval, save_json
            from anchorkv.packed_decode import kl_divergence
            from anchorkv.benchmark_suite import SuiteSettings, quality_plans, paired_family_differences
            from anchorkv.run_control import RunControl
        except ImportError:
            self.skipTest('optional torch/pandas dependencies are unavailable')
        suite = SuiteSettings(decode_repeats=1)
        cases = [dict(case_id=f'{seed}-{position}', family_id=f'retrieval-{seed}',
                      task='retrieval', content_seed=seed, position=position,
                      ids=[2] * 128, answer='1234', evidence_pages=[2], target_length=128)
                 for seed in (7, 19) for position in ('early', 'late')]

        class Tokenizer:
            eos_token_id = 1
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(input_ids=[2, 3])

        def run_policy(model, tokenizer, source, case, settings, policy, protected=(),
                       backend='packed', forced=None):
            row = dict(policy=policy, case_id=case['case_id'], answer_correct=True,
                       completed_correct=True, resident_bytes=100 + len(protected),
                       initial_resident_bytes=100 + len(protected))
            return row, torch.zeros((len(forced) if forced else 3, 8))

        def controlled_decode(model, source, settings, policy, inputs, length, warm, protected, backend):
            return dict(policy=policy, measured_tokens=length, new_measured_demotions=1,
                        initial_cache={'resident_bytes': 100 + len(protected)},
                        final_cache={'resident_bytes': 200 + len(protected)},
                        steady_tokens_per_second=20, setup_seconds=.1, warmup_seconds=.2,
                        measured_wall_seconds=length / 20,
                        setup_warmup_decode_seconds=.3 + length / 20,
                        measured_peak_allocated_bytes=10000)

        with tempfile.TemporaryDirectory() as directory:
            env = dict(pd=pd, torch=torch, time=time, statistics=statistics, random=random,
                       Settings=Settings, SuiteSettings=SuiteSettings, settings=Settings(), suite=suite,
                       expanded_cases=cases, OUTPUT=Path(directory), BACKEND='packed',
                       checkpoint=RunControl(directory, {'test': 1}), QUALITY_MINUTES=10, DECODE_MINUTES=10,
                       gates={'packed_status': 'passed'}, RUN_EXPANDED_QUALITY=True,
                       RUN_CONTROLLED_DECODE=True, model=None, tokenizer=Tokenizer(),
                       quality_plans=quality_plans, paired_family_differences=paired_family_differences,
                       bootstrap_interval=bootstrap_interval, save_json=save_json,
                       kl_divergence=kl_divergence, run_policy=run_policy,
                       controlled_decode=controlled_decode, clean=lambda: None, sync=lambda: None,
                       prefill_case=lambda *args: [], display=lambda *args: None,
                       automatic_scores=lambda model, source, case, candidates, settings:
                           ({p: float(p) for p in candidates}, None, .01))
            source = (Path(__file__).parents[1] / 'notebooks/expanded_cells.py').read_text()
            code_cells = [part.split('\n', 1)[1] for part in source.split('# %%')[1:]
                          if not part.startswith(' [markdown]')]
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in code_cells:
                    exec(compile(cell, 'expanded-notebook-cell', 'exec'), env)
            self.assertEqual(len(env['expanded_rows']), 64)
            self.assertEqual(len(env['decode_rows']), 18)
            self.assertEqual(len(env['decode_warmups']), 18)
            self.assertEqual(len(env['comparisons']), 12)
            self.assertTrue(all(row['family_count'] == 2 for row in env['comparisons']))
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in code_cells:
                    exec(compile(cell, 'resumed-notebook-cell', 'exec'), env)
            self.assertEqual(len(env['expanded_rows']), 64, 'resume duplicated quality rows')
            self.assertEqual(len(env['decode_rows']), 18, 'resume duplicated decode rows')
            for name in ('expanded-report.md', 'expanded-breakdown.csv',
                         'expanded-paired-comparisons.json', 'controlled-decode-summary.csv',
                         'expanded-failures.json', 'decode-workload.json'):
                self.assertTrue((Path(directory) / name).is_file(), name)
            # Gate failures must skip timing, not silently benchmark dense fallback.
            env['BACKEND'] = 'dense'
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code_cells[1], env)
            self.assertEqual(len(env['decode_rows']), 18, 'existing results must be preserved')
            self.assertIn('packed gate did not pass',
                          (Path(directory) / 'controlled-decode-status.json').read_text())
            # Interrupt between variants, export a partial report, then finish without duplication.
            partial = Path(directory) / 'partial'
            clock = [0]
            env.update(OUTPUT=partial, BACKEND='packed', QUALITY_MINUTES=1,
                       checkpoint=RunControl(partial, {'test': 2}, clock=lambda: clock[0]))
            def slow_policy(*args, **kwargs):
                clock[0] += 25
                return run_policy(*args, **kwargs)
            env['run_policy'] = slow_policy
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in code_cells:
                    exec(cell, env)
            self.assertEqual(len(env['expanded_rows']), 1)
            self.assertFalse(env['comparisons'], 'partial cases must not enter paired comparisons')
            self.assertEqual(env['checkpoint'].load('expanded-quality-status.json')['status'], 'paused_time_budget')
            self.assertTrue((partial / 'expanded-report.md').exists())
            env.update(run_policy=run_policy, QUALITY_MINUTES=10)
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in code_cells:
                    exec(cell, env)
            self.assertEqual(len(env['expanded_rows']), 64)
            self.assertEqual(env['checkpoint'].load('expanded-quality-status.json')['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
