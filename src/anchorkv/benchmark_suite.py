"""Expanded synthetic quality suite and controlled decode workloads.

Prompt families are the sampling unit; positions/lengths are matched variants.
This suite is deliberately not advertised as a standard reasoning benchmark.
"""

from dataclasses import dataclass
import hashlib
import json
import random
import time


@dataclass
class SuiteSettings:
    content_seeds: tuple = (7, 19)
    tasks: tuple = ('retrieval', 'multi_fact', 'arithmetic')
    lengths: tuple = (768, 1536)
    positions: tuple = ('early', 'middle', 'late')
    budgets: tuple = (0.10, 0.25)
    random_seeds: tuple = (7, 19, 43)
    decode_lengths: tuple = (128, 256, 512)
    decode_warmup_tokens: int = 32
    decode_repeats: int = 3


def task_spec(task, seed):
    rng = random.Random(seed)
    code_a, code_b = rng.sample(range(1000, 9999), 2)
    crates, each, removed = rng.randint(11, 29), rng.randint(12, 24), rng.randint(5, 19)
    if task == 'retrieval':
        return ([f'Current authoritative record: Project Zephyr has access code {code_a}.'],
                'What is the current access code for Project Zephyr? Return only the four-digit code.',
                str(code_a))
    if task == 'multi_fact':
        return ([f'Current authoritative record: Project Zephyr has access code {code_a}.',
                 f'Current authoritative record: Project Juniper has access code {code_b}.'],
                'Return the current codes for Zephyr then Juniper, separated by one comma and no spaces.',
                f'{code_a},{code_b}')
    if task == 'arithmetic':
        return ([f'Current authoritative record: Zephyr received {crates} crates.',
                 f'Current authoritative record: each Zephyr crate contains {each} units.',
                 f'Current authoritative record: {removed} units were removed from the Zephyr shipment.'],
                'How many units remain in the Zephyr shipment? Multiply crates by units per crate, '
                'then subtract removed units. Return only the integer, with no explanation.',
                str(crates * each - removed))
    raise ValueError(f'unknown task: {task}')


def build_quality_case(tokenizer, settings, task, seed, target, position):
    if position not in ('early', 'middle', 'late'):
        raise ValueError('invalid evidence position')
    evidence, query, answer = task_spec(task, seed)
    rng = random.Random(seed + 100000)
    # Precompute once: fitting a token bound must not change the underlying records.
    decoys = [f'Retired unrelated Cedar record {i}: code {rng.randrange(1000, 9999)}; '
              f'{rng.randrange(10, 40)} crates. This is not a current Zephyr or Juniper record.'
              for i in range(max(128, target))]
    bound = min(target, settings.max_prompt_tokens)

    def render(n):
        records = decoys[:n]
        # Separate required facts with distractors instead of putting them in one page.
        bundle = []
        for j, fact in enumerate(evidence):
            bundle.append(fact)
            if j + 1 < len(evidence):
                bundle.append(decoys[-1 - j])
        where = {'early': 0, 'middle': n // 2, 'late': n}[position]
        records[where:where] = bundle
        text = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'Use the current authoritative records, ignoring retired records. '
             'Follow the requested answer format exactly.'},
            {'role': 'user', 'content': '\n'.join(records) + '\n\n' + query}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text, tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    best = None
    for n in range(len(decoys) - len(evidence)):
        text, encoded = render(n)
        if len(encoded.input_ids) > bound:
            break
        best = text, encoded
    if best is None:
        raise ValueError(f'{task}: token cap {bound} cannot fit complete evidence and query')
    text, encoded = best
    spans = [(text.index(fact), text.index(fact) + len(fact)) for fact in evidence]
    pages = {i // settings.block for i, (a, b) in enumerate(encoded.offset_mapping)
             if any(b > start and a < end for start, end in spans)}
    family = f'{task}-seed{seed}'
    return {'case_id': f'{family}-n{target}-{position}', 'family_id': family,
            'task': task, 'content_seed': seed, 'target_length': target,
            'position': position, 'answer': answer, 'prompt': text,
            'ids': encoded.input_ids, 'evidence_pages': sorted(pages),
            'evidence_spans': spans, 'evidence_text': evidence,
            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}


def build_quality_suite(tokenizer, settings, suite):
    cases = [build_quality_case(tokenizer, settings, task, seed, length, position)
             for task in suite.tasks for seed in suite.content_seeds
             for length in suite.lengths for position in suite.positions]
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('duplicate task/seed/length/position settings')
    return cases


def quality_plans(candidates, scores, evidence, suite):
    from .packed_decode import choose_pages
    plans = []
    for policy in ('native_fp16', 'paged_fp16', 'int8', 'int4'):
        plans.append({'variant': policy, 'policy': policy, 'budget': None,
                      'random_seed': None, 'extra_fp16_pages': 0,
                      'protected': [] if policy.endswith('fp16') else [0]})
    if not candidates or len(set(suite.budgets)) != len(suite.budgets):
        raise ValueError('need eligible pages and unique budgets')
    for budget in suite.budgets:
        if not 0 <= budget <= 1:
            raise ValueError('budget fraction must be in [0, 1]')
        count = int(len(candidates) * budget)
        for policy in ('recent', 'automatic', 'oracle', 'random'):
            seeds = suite.random_seeds if policy == 'random' else (None,)
            for seed in seeds:
                pages = choose_pages(policy, candidates, count, scores=scores,
                                     evidence=evidence, seed=seed or 0) | {0}
                plans.append({'variant': f'{policy}-b{budget}-s{seed}', 'policy': policy,
                              'budget': budget, 'random_seed': seed,
                              'extra_fp16_pages': count, 'protected': sorted(pages)})
    if len({p['variant'] for p in plans}) != len(plans):
        raise ValueError('duplicate selector seeds')
    return plans


def paired_family_differences(rows, control, metric, budget):
    """Average random draws, then paired variants, then return one value per family."""
    from statistics import mean
    by_case = {}
    for row in rows:
        if row['budget'] == budget and row['policy'] in ('automatic', control):
            key = (row['family_id'], row['case_id'])
            by_case.setdefault(key, {}).setdefault(row['policy'], []).append(row[metric])
    families = {}
    for (family, _), policies in by_case.items():
        if set(policies) != {'automatic', control}:
            raise ValueError('incomplete paired comparison')
        difference = mean(policies['automatic']) - mean(policies[control])
        families.setdefault(family, []).append(difference)
    return {family: mean(values) for family, values in families.items()}


def controlled_decode(model, source, settings, policy, input_tokens, measured_tokens,
                      warmup_tokens=32, protected=(), backend='packed'):
    """Fixed input history: warm-up and measured tokens are distinct cache appends.

    No sampling, stopping, accuracy scoring, or per-token logit transfers. CUDA
    event duration is device timeline elapsed time (including launch gaps), NOT
    a sum of kernel execution times. CPU support exists for adapter unit tests.
    """
    import gc
    import torch
    from .colab_experiment import attention_route, make_live_cache, stock_cache

    total = warmup_tokens + measured_tokens
    if warmup_tokens < 0 or measured_tokens <= 0 or len(input_tokens) != total:
        raise ValueError('provide exactly warmup_tokens + measured_tokens input tokens')
    if policy not in ('native_fp16', 'paged_fp16', 'int8', 'int4',
                      'automatic', 'recent', 'random', 'oracle'):
        raise ValueError('unknown policy')
    device = model.device
    gpu = device.type == 'cuda'
    if not gpu and backend == 'packed' and policy != 'native_fp16':
        raise ValueError('packed throughput requires CUDA')

    def synchronize():
        if gpu:
            torch.cuda.synchronize(device)

    def report(cache):
        if not native:
            return cache.report()
        size = sum(t.numel() * t.element_size() for layer in cache.layers
                   for t in (layer.keys, layer.values))
        return {'tokens': cache.get_seq_length(), 'resident_bytes': size,
                'payload_bytes': size, 'demotions': 0}

    gc.collect()
    if gpu:
        torch.cuda.empty_cache()
    prefix = source[0][0].shape[-2]
    # Materialize the common workload outside setup and decode timing.
    ids = torch.tensor([input_tokens], device=device, dtype=torch.long)
    positions = torch.arange(prefix, prefix + total, device=device)
    synchronize()
    base_allocated = torch.cuda.memory_allocated(device) if gpu else None
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    native = policy == 'native_fp16'
    bits = 16 if policy == 'paged_fp16' else 8 if policy == 'int8' else 4
    started = time.perf_counter()
    with torch.inference_mode():
        cache = (stock_cache(source, device=device) if native else
                 make_live_cache(source, settings, protected, bits,
                                 max_tokens=prefix + total, device=device))
        synchronize()
        setup_seconds = time.perf_counter() - started
        initial = report(cache)

        def step(index):
            output = model(input_ids=ids[:, index:index + 1], past_key_values=cache,
                           cache_position=positions[index:index + 1], use_cache=True,
                           attention_mask=None if native else {'full_attention': None},
                           logits_to_keep=1)
            del output

        with attention_route(model, None if native else cache, backend):
            warm_started = time.perf_counter()
            for index in range(warmup_tokens):
                step(index)
            synchronize()
            warmup_seconds = time.perf_counter() - warm_started
            before = report(cache)
            setup_warmup_peak = torch.cuda.max_memory_allocated(device) if gpu else None
            if gpu:
                torch.cuda.reset_peak_memory_stats(device)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            measured_started = time.perf_counter()
            if gpu:
                begin.record()
            for index in range(warmup_tokens, total):
                step(index)
            if gpu:
                end.record()
            synchronize()
            wall_seconds = time.perf_counter() - measured_started
            cuda_seconds = begin.elapsed_time(end) / 1000 if gpu else None
        final = report(cache)
        peak = torch.cuda.max_memory_allocated(device) if gpu else None
        end_allocated = torch.cuda.memory_allocated(device) if gpu else None
    assert final['tokens'] == prefix + total, 'decode did not append the complete fixed workload'
    row = {'policy': policy, 'backend': 'stock_sdpa' if native else backend,
           'scope': 'fixed-history throughput and cache lifecycle; not answer accuracy',
           'prefix_tokens': prefix, 'warmup_tokens': warmup_tokens,
           'measured_tokens': measured_tokens, 'setup_seconds': setup_seconds,
           'warmup_seconds': warmup_seconds, 'measured_wall_seconds': wall_seconds,
           'measured_cuda_timeline_seconds': cuda_seconds,
           'steady_tokens_per_second': measured_tokens / wall_seconds,
           'setup_warmup_decode_seconds': setup_seconds + warmup_seconds + wall_seconds,
           'initial_cache': initial, 'measurement_start_cache': before, 'final_cache': final,
           'new_measured_demotions': final['demotions'] - before['demotions'],
           'setup_warmup_peak_allocated_bytes': setup_warmup_peak,
           'measured_peak_allocated_bytes': peak,
           'measured_peak_incremental_bytes': peak - base_allocated if gpu else None,
           'end_incremental_bytes': end_allocated - base_allocated if gpu else None,
           'input_sha256': hashlib.sha256(json.dumps(input_tokens).encode()).hexdigest()}
    del cache, ids, positions
    gc.collect()
    if gpu:
        torch.cuda.empty_cache()
    return row
