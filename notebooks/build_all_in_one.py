"""Mechanically generate the standalone notebook from reviewed Python sources."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build():
    sources = {'__init__.py': ''}
    for name in ('packed_decode.py', 'triton_decode.py', 'colab_experiment.py'):
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
    cells = []
    for section in script.split('# %%')[1:]:
        header, body = section.split('\n', 1)
        markdown = '[markdown]' in header
        if markdown:
            body = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line
                             for line in body.rstrip().splitlines()) + '\n'
        else:
            body = body.replace('# EMBED_RUNTIME', bootstrap).strip() + '\n'
        cell = {'cell_type': 'markdown' if markdown else 'code',
                'id': f'anchorkv-{len(cells):02}', 'metadata': {}, 'source': body.splitlines(keepends=True)}
        if not markdown:
            cell.update(execution_count=None, outputs=[])
            compile('\n'.join('# ' + line if line.startswith('%') else line for line in body.splitlines()),
                    f'cell-{len(cells)}', 'exec')
        cells.append(cell)
    notebook = {'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }, 'nbformat': 4, 'nbformat_minor': 5}
    target = ROOT / 'notebooks' / 'AnchorKV_T4_All_In_One.ipynb'
    target.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    print(f'Built {target.name}: {len(cells)} cells, source {digest}')


if __name__ == '__main__':
    build()
