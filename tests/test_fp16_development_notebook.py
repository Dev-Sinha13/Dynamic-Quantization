import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from anchorkv.arithmetic_protocol import build_protocol, canonical_hash, development_gate
from anchorkv.run_control import RunControl

ROOT = Path(__file__).parents[1]


class DevelopmentNotebookTests(unittest.TestCase):
    def test_frozen_manifest_and_embedded_development_inputs(self):
        bundle = ROOT / 'docs/experiments/data/arithmetic-protocol-v1'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        for split, cases in build_protocol().items():
            self.assertEqual(json.loads((bundle / f'{split}.json').read_text()), cases)
            self.assertEqual(manifest['splits'][split]['sha256'], canonical_hash(cases))
        for name, digest in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / 'src/anchorkv' / name).read_text(encoding='utf-8').encode()).hexdigest(), digest)
        nb = json.loads((ROOT / 'notebooks/AnchorKV_T4_FP16_Development.ipynb').read_text())
        embedded = {}
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            self.assertIsNone(cell['execution_count'])
            self.assertFalse(cell['outputs'])
            code = ''.join(cell['source'])
            compile(code, 'baseline-cell', 'exec')
            for node in ast.parse(code).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in ('SOURCES', 'DEVELOPMENT_CASES'):
                        embedded[name] = ast.literal_eval(node.value)
        self.assertEqual(embedded['DEVELOPMENT_CASES'], build_protocol()['development'])
        self.assertTrue(all(c['split'] == 'development' for c in embedded['DEVELOPMENT_CASES']))
        for name, source in embedded['SOURCES'].items():
            if name != '__init__.py':
                self.assertEqual(source, (ROOT / 'src/anchorkv' / name).read_text(encoding='utf-8'))

    def test_generation_cell_pauses_resumes_and_skips_completed_cases(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional CPU torch absent')
        if not hasattr(torch, 'tensor'):
            self.skipTest('CPU torch inaccessible')
        import time
        development = build_protocol()['development']
        calls, clock = [], [0]
        class FakeModel:
            device = 'cpu'
            generation_config = SimpleNamespace(eos_token_id=99)
            def generate(self, input_ids, **kwargs):
                calls.append(kwargs)
                clock[0] += 30
                index = int(input_ids[0, 0])
                return torch.cat([input_ids, torch.tensor([[1000 + index, 99]])], dim=1)
        class FakeTokenizer:
            eos_token_id = 99
            def decode(self, tokens, **kwargs):
                return development[tokens[0] - 1000]['answer']
        source = (ROOT / 'notebooks/baseline_cells.py').read_text()
        parts = [s.split('\n', 1)[1] for s in source.split('# %%')[1:] if not s.startswith(' [markdown]')]
        generation = next(s for s in parts if s.startswith('rows = checkpoint.load'))
        report = next(s for s in parts if s.startswith('gate = development_gate'))
        with tempfile.TemporaryDirectory() as directory:
            env = {'torch': torch, 'time': time, 'json': json, 'OUTPUT': Path(directory),
                   'checkpoint': RunControl(directory, {}, clock=lambda: clock[0]),
                   'cases': [{**c, 'ids': [i], 'prompt_sha256': str(i)} for i, c in enumerate(development)],
                   'DEVELOPMENT_CASES': development, 'development_gate': development_gate,
                   'model': FakeModel(), 'tokenizer': FakeTokenizer(), 'PHASE_MINUTES': 1}
            with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'empty_cache'), contextlib.redirect_stdout(io.StringIO()):
                exec(generation, env)
                exec(report, env)
                self.assertEqual(env['gate']['status'], 'incomplete')
                self.assertEqual(len(calls), 2)
                env['PHASE_MINUTES'] = 8
                exec(generation, env)
                exec(report, env)
                self.assertEqual(env['gate']['status'], 'passed')
                exec(generation, env)
            self.assertEqual(len(calls), 12)
            self.assertTrue(all(c['max_new_tokens'] == 64 and not c['do_sample'] for c in calls))


if __name__ == '__main__':
    unittest.main()
