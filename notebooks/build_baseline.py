"""Build a standalone development-only FP16 notebook from frozen inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    sources = {'__init__.py': ''}
    for name in ('arithmetic_protocol.py', 'arithmetic_scoring.py', 'run_control.py'):
        sources[name] = (ROOT / 'src/anchorkv' / name).read_text(encoding='utf-8')
    development = json.loads((ROOT / 'docs/experiments/data/arithmetic-protocol-v1/development.json').read_text())
    script = (ROOT / 'notebooks/baseline_cells.py').read_text(encoding='utf-8')
    bootstrap = (
        f'SOURCES = {sources!r}\n'
        f'DEVELOPMENT_CASES = {development!r}\n'
        f'SOURCE_SHA256 = {hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()!r}\n'
        f'WORKFLOW_SHA256 = {hashlib.sha256(script.encode()).hexdigest()!r}\n'
        "runtime = Path('/content/anchorkv-baseline-runtime')\n"
        "package = runtime / 'anchorkv_baseline'\n"
        'package.mkdir(parents=True, exist_ok=True)\n'
        'for name, source in SOURCES.items():\n'
        "    (package / name).write_text(source, encoding='utf-8')\n"
        'sys.path.insert(0, str(runtime))\n'
    )
    cells = []
    for section in script.split('# %%')[1:]:
        header, body = section.split('\n', 1)
        markdown = '[markdown]' in header
        if markdown:
            body = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line for line in body.splitlines())
        else:
            body = body.replace('# EMBED_BASELINE', bootstrap)
            compile(body, f'baseline-cell-{len(cells)}', 'exec')
        cell = {'cell_type': 'markdown' if markdown else 'code', 'id': f'baseline-{len(cells)}',
                'metadata': {}, 'source': body.strip().splitlines(keepends=True)}
        if not markdown:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    target = ROOT / 'notebooks/AnchorKV_T4_FP16_Development.ipynb'
    target.write_text(json.dumps(nb, indent=1) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
