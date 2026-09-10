"""Deterministic development/held-out arithmetic protocol. No model or GPU needed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .arithmetic_scoring import SCORER_VERSION, score_response

PROTOCOL_VERSION = 'arithmetic-protocol-v1'
MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
OPERATIONS = ('addition', 'subtraction', 'multiply_subtract')
SYSTEM = ('Solve the requested inventory calculation using only the Zephyr records. '
          'Other projects are unrelated. Return only the final integer, without units or explanation.')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def build_protocol():
    splits = {'development': [], 'heldout': []}
    used = set()
    for split, per_operation in (('development', 2), ('heldout', 4)):
        for operation in OPERATIONS:
            for index in range(per_operation):
                family = f'{split}-{operation}-{index:02}'
                rng = random.Random(f'{PROTOCOL_VERSION}:{family}')
                while True:
                    a, b, c = rng.randint(40, 90), rng.randint(2, 30), 0
                    if operation == 'multiply_subtract':
                        a, b, c = rng.randint(3, 12), rng.randint(4, 15), rng.randint(1, 9)
                    key = operation, a, b, c
                    if key not in used:
                        used.add(key)
                        break
                if operation == 'addition':
                    facts = [f'Zephyr opening inventory: {a} units.', f'Zephyr received: {b} units.']
                    query = 'What is Zephyr inventory after receiving the delivery?'
                    gold = a + b
                elif operation == 'subtraction':
                    facts = [f'Zephyr opening inventory: {a} units.', f'Zephyr shipped: {b} units.']
                    query = 'What is Zephyr inventory after shipping?'
                    gold = a - b
                else:
                    facts = [f'Zephyr received: {a} crates.', f'Zephyr units per crate: {b}.',
                             f'Zephyr shipped after receiving: {c} units.']
                    query = 'How many Zephyr units remain after shipping?'
                    gold = a * b - c
                distractors = [f'Cedar{i} inventory: {rng.randint(100, 999)} units; '
                               f'Cedar{i} shipment: {rng.randint(10, 99)} units.' for i in range(24)]
                positions = ('clean', 'middle') if split == 'development' else ('clean', 'early', 'middle', 'late')
                for position in positions:
                    if position == 'clean':
                        records = facts[:]
                    else:
                        records = distractors[:]
                        where = {'early': 0, 'middle': len(records) // 2, 'late': len(records)}[position]
                        records[where:where] = facts
                    user = '\n'.join(records) + '\n\n' + query + '\nReturn only the integer.'
                    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
                    splits[split].append({'case_id': f'{family}-{position}', 'family_id': family,
                        'split': split, 'operation': operation, 'position': position,
                        'operands': [a, b, c], 'answer': str(gold), 'evidence_text': facts,
                        'messages': messages, 'messages_sha256': canonical_hash(messages)})
    return splits


def render_case(case, tokenizer, max_prompt_tokens=1536):
    text = tokenizer.apply_chat_template(case['messages'], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > max_prompt_tokens:
        raise ValueError(f"{case['case_id']}: {len(ids)} tokens exceeds {max_prompt_tokens}; do not truncate evidence")
    return {**case, 'prompt': text, 'ids': ids,
            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}


def development_gate(cases, results):
    """Predeclared baseline gate, never case selection or a held-out filter."""
    if cases != build_protocol()['development']:
        raise ValueError('Gate requires the complete frozen development set')
    expected = {c['case_id']: c for c in cases}
    by_id = {}
    scores = []
    for row in results:
        key = row['case_id']
        if key not in expected or key in by_id:
            raise ValueError('Unknown or duplicate development result')
        if row['policy'] != 'native_fp16':
            raise ValueError('Development gate accepts only native FP16 results')
        case = expected[key]
        if row['messages_sha256'] != case['messages_sha256']:
            raise ValueError('Result prompt identity mismatch')
        scored = score_response(row['text'], case['answer'], ended_eos=row['ended_eos'], truncated=row['truncated'])
        by_id[key] = scored
        scores.append({**scored, 'case_id': key, 'operation': case['operation'], 'position': case['position']})
    correct = sum(s['final_completed_correct'] for s in scores)
    operation_counts = {op: sum(s['final_completed_correct'] for s in scores if s['operation'] == op) for op in OPERATIONS}
    position_counts = {p: sum(s['final_completed_correct'] for s in scores if s['position'] == p) for p in ('clean', 'middle')}
    complete = len(results) == len(cases)
    passed = complete and correct >= 11 and min(operation_counts.values()) >= 3 and min(position_counts.values()) >= 5
    return {'protocol_version': PROTOCOL_VERSION, 'scorer_version': SCORER_VERSION,
            'status': 'incomplete' if not complete else 'passed' if passed else 'failed',
            'correct': correct, 'received': len(results), 'expected': len(cases),
            'correct_by_operation': operation_counts, 'correct_by_position': position_counts,
            'scores': scores, 'warning': 'Development readiness only; no held-out or compression claim. Never filter held-out cases using this gate.'}


def export_protocol(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Choose a new directory; frozen artifacts are not overwritten')
    splits = build_protocol()
    manifest = {'protocol_version': PROTOCOL_VERSION, 'scorer_version': SCORER_VERSION,
        'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'generation': {'max_new_tokens': 64, 'do_sample': False, 'enable_thinking': False},
        'max_prompt_tokens': 1536, 'development_gate': {
            'minimum_correct': 11, 'total': 12, 'minimum_correct_per_operation': 3,
            'minimum_correct_per_context': 5},
        'splits': {name: {'cases': len(cases), 'families': len({c['family_id'] for c in cases}),
                          'sha256': canonical_hash(cases)} for name, cases in splits.items()},
        'source_hash_encoding': 'UTF-8 with universal newlines',
        'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_text(encoding='utf-8').encode()).hexdigest()
                          for name in ('arithmetic_protocol.py', 'arithmetic_scoring.py')},
        'baseline_status': 'not_run', 'heldout_status': 'not_evaluated'}
    directory.mkdir(parents=True)
    for name, value in {**splits, 'manifest': manifest}.items():
        (directory / f'{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory')
    export_protocol(parser.parse_args().output_directory)
