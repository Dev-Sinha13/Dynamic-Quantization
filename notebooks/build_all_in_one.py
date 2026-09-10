"""Mechanically generate the standalone notebook from reviewed Python sources."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build(arithmetic_only=False):
    sources = {'__init__.py': ''}
    for name in ('packed_decode.py', 'triton_decode.py', 'colab_experiment.py', 'benchmark_suite.py', 'run_control.py'):
        sources[name] = (ROOT / 'src' / 'anchorkv' / name).read_text(encoding='utf-8')
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    bootstrap = (
        '# Embedded runtime: generated from the repository, no network checkout required.\n'
        f'EMBEDDED_SOURCES = {sources!r}\n'
        f'EMBEDDED_SOURCE_SHA256 = {digest!r}\n'
        "runtime_root = Path('/content/anchorkv-embedded-runtime')\n"
        "runtime_package = runtime_root / 'anchorkv_notebook'\n"
        'runtime_package.mkdir(parents=True, exist_ok=True)\n'
        'for filename, source in EMBEDDED_SOURCES.items():\n'
        "    (runtime_package / filename).write_text(source, encoding='utf-8')\n"
        'sys.path.insert(0, str(runtime_root))\n'
    )
    script = (ROOT / 'notebooks' / 'all_in_one_cells.py').read_text(encoding='utf-8')
    if arithmetic_only:
        script = script.replace("PROFILE = 'quick'", "PROFILE = 'arithmetic'", 1)
    script = script.replace('# EXPANDED_CELLS',
                            (ROOT / 'notebooks' / 'expanded_cells.py').read_text(encoding='utf-8'))
    bootstrap += f'WORKFLOW_SHA256 = {hashlib.sha256(script.encode()).hexdigest()!r}\n'
    cells = []
    for section in script.split('# %%')[1:]:
        header, body = section.split('\n', 1)
        markdown = '[markdown]' in header
        if markdown:
            body = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line
                             for line in body.rstrip().splitlines()) + '\n'
        else:
            body = body.replace('# EMBED_RUNTIME', bootstrap).strip() + '\n'
            # Keep the historical pilot available without charging every new run for it.
            if body.startswith(('raw_rows = []', 'stress_case = cases[0]', 'raw = pd.DataFrame(raw_rows)')):
                body = 'if RUN_LEGACY_PILOT:\n' + ''.join('    ' + line + '\n' for line in body.splitlines())
        cell = {'cell_type': 'markdown' if markdown else 'code',
                'id': f'anchorkv-{len(cells):02}', 'metadata': {}, 'source': body.splitlines(keepends=True)}
        if not markdown:
            cell.update(execution_count=None, outputs=[])
            compile('\n'.join('# ' + line if line.startswith('%') else line for line in body.splitlines()),
                    f'cell-{len(cells)}', 'exec')
        cells.append(cell)
    if arithmetic_only:
        cells[0]['source'] = [
            '# AnchorKV: arithmetic-only follow-up\n',
            '\nUpload to a fresh Colab T4 runtime and Run all. No edits or old archive required.\n',
            '\nReruns only the two original arithmetic prompts (seeds 7 and 19), with the\n',
            'same nine policy variants and a **64-token answer cap** instead of 16.\n',
            'That is 18 variants and 38 quality generation/replay calls, plus numerical gates.\n',
            'The historical pilot and throughput experiments are disabled.\n',
            '\nThe quality phase retains its 10-minute soft limit and resumable checkpoints.\n',
            'Setup/gates are outside the limit; in-flight work may overrun it.\n',
            'If paused, rerun the quality cell, then the report and download cells.\n',
            '\nScoring remains strict: explanation text is not silently accepted as a final answer.\n',
            'The larger cap may reveal a prompt-format failure even if arithmetic is correct.\n',
            'Use a new output folder; the old 16-token checkpoint is intentionally incompatible.\n',
            'Return the final results zip. These two cases are diagnostic, not broad validation.\n',
        ]
    notebook = {'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }, 'nbformat': 4, 'nbformat_minor': 5}
    target = ROOT / 'notebooks' / ('AnchorKV_T4_Arithmetic_Followup.ipynb' if arithmetic_only
                                  else 'AnchorKV_T4_All_In_One.ipynb')
    target.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    print(f'Built {target.name}: {len(cells)} cells, source {digest}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arithmetic-only', action='store_true')
    build(arithmetic_only=parser.parse_args().arithmetic_only)
