"""Frozen prospective arithmetic scoring v1; legacy post-hoc scoring is untouched."""
import re

SCORER_VERSION = 'arithmetic-final-v1'
MAX_RESPONSE_CHARS = 20000
INTEGER = r'[+-]?[0-9]{1,12}'


def extract_answer(text):
    if not isinstance(text, str):
        raise TypeError('response must be text')
    if len(text) > MAX_RESPONSE_CHARS:
        return None, 'oversized'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, 'empty'
    final = lines[-1]
    # Entire final line only. Never search preceding reasoning for the gold value.
    rules = (
        ('bare_integer', rf'({INTEGER})[.]?'),
        ('answer_label', rf'(?:final answer|answer|the integer is|return only the integer)\s*:\s*({INTEGER})[.]?'),
        ('remaining_units', rf'({INTEGER}) units remain(?: in the Zephyr shipment)?[.]?'),
        # A syntactically complete integer expression; !=, <=, decimals, prose,
        # alternatives, multiple equalities and unfinished arithmetic are rejected.
        ('terminal_equation', rf'{INTEGER}(?:\s*[-+*/×]\s*{INTEGER})+\s*=\s*({INTEGER})[.]?'),
    )
    for rule, pattern in rules:
        match = re.fullmatch(pattern, final, flags=re.IGNORECASE | re.ASCII)
        if match:
            return str(int(match.group(1))), rule
    return None, 'unparsed'


def score_response(text, gold, *, ended_eos, truncated):
    if not isinstance(gold, str) or not re.fullmatch(INTEGER, gold, re.ASCII):
        raise ValueError('gold must be an ASCII integer string')
    if type(ended_eos) is not bool or type(truncated) is not bool:
        raise TypeError('completion flags must be booleans')
    extracted, rule = extract_answer(text)
    complete = ended_eos and not truncated
    matched = extracted == str(int(gold)) if extracted is not None else False
    return {
        'scorer_version': SCORER_VERSION,
        'strict_text_correct': text.strip() == gold,
        'strict_completed_correct': complete and text.strip() == gold,
        'extracted_answer': extracted, 'extraction_rule': rule,
        'parsed': extracted is not None, 'complete': complete,
        'final_completed_correct': complete and matched,
        'status': ('incomplete' if not complete else 'unparsed' if extracted is None
                   else 'correct_final_claim' if matched else 'incorrect_final_claim'),
    }
