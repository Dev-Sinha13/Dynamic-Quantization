"""CPU-only, post-hoc arithmetic scoring. Never imports or executes archived code."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile


def extract_final_integer(text):
    """Conservative terminal answer parser; gold labels never enter extraction.

    No last-number/contains-gold fallback: malformed or unsupported responses
    remain unparsed. This recognizes final claims, not reasoning correctness.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None, 'empty'
    final = lines[-1]
    integer = r'([+-]?\d+)'
    rules = [
        ('bare_final_integer', rf'{integer}[.]?'),
        ('explicit_integer_label', rf'(?:the integer is|return only the integer|final answer|answer)\s*:\s*{integer}[.]?'),
        ('remaining_units', rf'{integer} units remain(?: in the Zephyr shipment)?[.]?'),
    ]
    for rule, pattern in rules:
        match = re.fullmatch(pattern, final, re.IGNORECASE)
        if match:
            return str(int(match.group(1))), rule
    # Only an unambiguous terminal equality, not arbitrary trailing numbers.
    match = re.search(r'=\s*([+-]?\d+)\s*\.?$', final)
    if match:
        return str(int(match.group(1))), 'terminal_equality'
    return None, 'unparsed_requires_review'


def rescore(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError('Output directory already exists; choose a new directory to preserve earlier results')
    with ZipFile(archive) as bundle:
        def read(name):
            entries = [e for e in bundle.infolist() if e.filename == name]
            if len(entries) != 1 or entries[0].file_size > 5_000_000:
                raise ValueError(f'Missing, duplicate or oversized entry: {name}')
            return json.loads(bundle.read(entries[0]))
        prompts = read('expanded-prompts.json')
        original = read('expanded-quality.json')
        environment = read('environment.json')
    cases = {p['case_id']: p for p in prompts}
    if len(cases) != len(prompts):
        raise ValueError('Duplicate prompt IDs')
    records, keys = [], set()
    for row in original:
        case = cases[row['case_id']]
        if case['task'] != 'arithmetic':
            raise ValueError('This scorer is scoped to arithmetic only')
        key = row['case_id'], row['variant']
        if key in keys:
            raise ValueError('Duplicate case/variant measurement')
        keys.add(key)
        extracted, rule = extract_final_integer(row['text'])
        completed = row['ended_eos'] and not row['truncated']
        correct = completed and extracted is not None and extracted == case['answer']
        records.append({
            'case_id': row['case_id'], 'policy': row['policy'], 'variant': row['variant'],
            'random_seed': row['random_seed'], 'gold': case['answer'], 'text': row['text'],
            'original_answer_correct': row['answer_correct'],
            'original_completed_correct': row['completed_correct'],
            'ended_eos': row['ended_eos'], 'truncated': row['truncated'],
            'extracted_final_integer': extracted, 'extraction_rule': rule,
            'supplemental_completed_final_correct': correct,
            'review_status': ('incomplete' if not completed else 'unparsed' if extracted is None
                              else 'correct_final_claim' if correct else 'incorrect_final_claim'),
        })
    grouped = defaultdict(list)
    for row in records:
        grouped[row['policy']].append(row)
    summary = []
    for policy, rows in sorted(grouped.items()):
        # Equal-weight cases; repeated random draws are not independent questions.
        per_case = defaultdict(list)
        for row in rows:
            per_case[row['case_id']].append(int(row['supplemental_completed_final_correct']))
        summary.append({'policy': policy, 'runs': len(rows),
                        'strict_correct': sum(r['original_completed_correct'] for r in rows),
                        'final_correct': sum(r['supplemental_completed_final_correct'] for r in rows),
                        'parsed': sum(r['extracted_final_integer'] is not None for r in rows),
                        'case_averaged_final_accuracy': sum(sum(v) / len(v) for v in per_case.values()) / len(per_case)})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    shutil.copyfile(archive, output / 'original-results.zip')
    result = {'schema_version': 1, 'analysis': 'post-hoc supplemental final-answer metric',
              'archive_sha256': digest, 'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment': environment, 'summary': summary, 'records': records}
    (output / 'rescore.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    report = ['# Arithmetic follow-up: local rescoring', '',
              'Post-hoc analysis of saved responses; no GPU rerun and no original scores changed.',
              f'Original archive SHA-256: `{digest}`.', '',
              '| Policy | Strict correct | Final-answer correct | Parsed | Case-averaged final accuracy |',
              '|---|---:|---:|---:|---:|']
    for row in summary:
        n = row['runs']
        report.append(f"| {row['policy']} | {row['strict_correct']}/{n} | {row['final_correct']}/{n} | {row['parsed']}/{n} | {row['case_averaged_final_accuracy']:.0%} |")
    report += ['', '## Interpretation limits', '',
               '- Strict exact-format scoring remains unchanged. This supplemental metric ignores extra explanation.',
               '- Extraction uses only the response, not the gold answer. It accepts explicit terminal integer claims or equalities; no contains-gold or last-number fallback.',
               '- Unparsed answers count as no demonstrated success in the denominator, but are not labeled proven arithmetic errors. Inspect them in rescore.json.',
               '- Final-answer correctness does not validate every reasoning step.',
               '- The parser was introduced after inspecting these outputs: this is exploratory post-hoc analysis, not a pre-registered result.',
               '- Only two distinct questions were tested. Random has two draws per question; its rate is averaged within case before across cases.',
               '- Native FP16 and automatic each have one correct final answer. This does not establish general equivalence or superiority.',
               '- Next validate prompts/scoring on an independent development set, freeze the scorer, then compare policies on held-out cases.', '']
    (output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'summary': summary}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('output')
    args = parser.parse_args()
    rescore(args.archive, args.output)
