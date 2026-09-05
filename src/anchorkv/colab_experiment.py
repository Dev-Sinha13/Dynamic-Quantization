"""Reproducible experiment helpers embedded in the all-in-one Colab notebook."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

import torch

from .packed_decode import PagedLayer, choose_pages, dense_attention, kl_divergence, pack_page


@dataclass
class Settings:
    model_id: str = 'Qwen/Qwen3-0.6B'
    revision: str = 'c1899de289a04d12100db370d81485cdf75e47ca'
    seed: int = 7
    block: int = 16
    recent_pages: int = 2
    keep_fraction: float = 0.15
    max_new_tokens: int = 48
    max_prompt_tokens: int = 1536
    repeats: int = 3
    score_layers: int = 4
    sensitivity_pages: int = 8
    lengths: tuple = (384, 896)
    positions: tuple = ('early', 'middle', 'late')


def sync():
    torch.cuda.synchronize()


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def build_case(tokenizer, settings, index, target_length, position):
    rng = random.Random(settings.seed + index)
    answer = str(rng.randrange(1000, 9999))
    project = ['Zephyr', 'Juniper', 'Cobalt', 'Orion', 'Maple', 'Atlas'][index % 6]
    evidence = f'Authoritative record: the access code for Project {project} is {answer}.'
    instruction = 'Read the records and return only the requested four-digit code. Do not explain or repeat it.'
    query = f'What is the access code for Project {project}?'
    def render(repetitions):
        decoys = [f'Retired record {i}: Project Cedar{i} used code {rng.randrange(1000, 9999)}. '
                  'This expired record is unrelated to the requested project.' for i in range(repetitions)]
        location = {'early': 0, 'middle': len(decoys) // 2, 'late': len(decoys)}[position]
        decoys.insert(location, evidence)
        text = '\n'.join(decoys) + '\n\n' + query
        return tokenizer.apply_chat_template(
            [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    # Fit complete records before loading weights; never truncate the query/answer marker.
    best = None
    for repetitions in range(1, 65):
        text = render(repetitions)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded.input_ids) > min(target_length, settings.max_prompt_tokens):
            break
        best = text, encoded
    if best is None:
        raise ValueError('context bound is too small for the chat scaffold and evidence')
    text, encoded = best
    start = text.index(evidence)
    end = start + len(evidence)
    evidence_tokens = [i for i, (a, b) in enumerate(encoded.offset_mapping) if b > start and a < end]
    return {'case_id': f'case-{index:03}', 'answer': answer, 'position': position,
            'target_length': target_length, 'prompt': text, 'ids': encoded.input_ids,
            'evidence_pages': sorted({i // settings.block for i in evidence_tokens}),
            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}


def stock_cache(source, device='cuda'):
    from transformers import DynamicCache
    return DynamicCache([(k.to(device), v.to(device)) for k, v in source])


def cache_cpu(cache):
    return [(layer.keys.detach().cpu(), layer.values.detach().cpu()) for layer in cache.layers]


def make_live_cache(source, settings, protected=(), bits=4, max_tokens=None, device='cuda'):
    from transformers import DynamicCache
    class LiveCache(DynamicCache):
        def __init__(self):
            super().__init__()
            self.stores = []
            for key, value in source:
                store = PagedLayer(key.shape[1], key.shape[-1], max_tokens,
                                   block=settings.block, archive_bits=bits,
                                   protected=protected, recent_pages=settings.recent_pages, device=device)
                store.append(key.to(device), value.to(device))
                self.stores.append(store)

        def get_seq_length(self, layer_idx=0):
            return self.stores[layer_idx].length

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            self.stores[layer_idx].append(key_states, value_states)
            # These are ignored by our attention interface, which reads the stores.
            return key_states, value_states

        def report(self):
            return {'resident_bytes': sum(s.resident_bytes for s in self.stores),
                    'payload_bytes': sum(s.payload_bytes for s in self.stores),
                    'demotions': sum(s.demotions for s in self.stores),
                    'tokens': self.get_seq_length()}
    if max_tokens is None:
        max_tokens = source[0][0].shape[-2] + settings.max_new_tokens + 32
    return LiveCache()


@contextmanager
def attention_route(model, cache=None, backend='dense', capture=None):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = model.config._attn_implementation
    dense_interface = ALL_ATTENTION_FUNCTIONS['sdpa']
    def attention(module, query, key, value, attention_mask, scaling, **kwargs):
        if capture is not None:
            capture[module.layer_idx] = query.detach().cpu()
        if cache is None:
            return dense_interface(module, query, key, value, attention_mask,
                                   scaling=scaling, **kwargs)
        store = cache.stores[module.layer_idx]
        if backend == 'packed':
            from .triton_decode import packed_attention
            result = packed_attention(query, store, scaling)
        else:
            result = dense_attention(query, store, scaling)
        return result.transpose(1, 2).contiguous(), None
    ALL_ATTENTION_FUNCTIONS.register('anchorkv_experiment', attention)
    model.config._attn_implementation = 'anchorkv_experiment'
    try:
        yield
    finally:
        model.config._attn_implementation = original
        # The global registry must not retain a closure owning the previous GPU cache.
        ALL_ATTENTION_FUNCTIONS.register('anchorkv_experiment', dense_interface)


def forward_token(model, cache, token, packed=False):
    position = cache.get_seq_length()
    return model(input_ids=torch.tensor([[token]], device=model.device), past_key_values=cache,
                 cache_position=torch.tensor([position], device=model.device),
                 attention_mask={'full_attention': None} if packed else None,
                 use_cache=True, logits_to_keep=1)


@torch.inference_mode()
def prefill_case(model, case):
    # Leave the final prompt token out so every policy computes its own FIRST answer logit.
    ids = torch.tensor([case['ids'][:-1]], device='cuda')
    output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    source = cache_cpu(output.past_key_values)
    del output, ids
    clean()
    return source


@torch.inference_mode()
def automatic_scores(model, source, case, candidates, settings):
    started = time.perf_counter()
    captured = {}
    cache = stock_cache(source)
    with attention_route(model, capture=captured):
        output = forward_token(model, cache, case['ids'][-1])
    reference_logits = output.logits[0, -1].float().cpu()
    del cache, output
    # Query tensors from the last prompt token only. No gold answer/evidence labels enter this score.
    scores = {page: 0.0 for page in candidates}
    layers = sorted(captured)[-settings.score_layers:]
    for idx in layers:
        q = captured[idx].float()
        key, value = source[idx]
        repeats = q.shape[1] // key.shape[1]
        probabilities = (q @ key.float().repeat_interleave(repeats, 1).transpose(-2, -1))
        probabilities = (probabilities / math.sqrt(key.shape[-1])).softmax(-1).mean((0, 1, 2))
        for page in candidates:
            start, end = page * settings.block, (page + 1) * settings.block
            k = key[0, :, start:end].contiguous()
            v = value[0, :, start:end].contiguous()
            k_error = (pack_page(k, 4).dense().float() - k.float()).square().mean()
            v_error = (pack_page(v, 4).dense().float() - v.float()).square().mean()
            scores[page] += float(probabilities[start:end].sum() * (k_error + v_error)) / len(layers)
    sync()
    return scores, reference_logits, time.perf_counter() - started


def eos_ids(model, tokenizer):
    value = model.generation_config.eos_token_id
    ids = set(value if isinstance(value, list) else [value])
    ids.add(tokenizer.eos_token_id)
    return ids - {None}


def answer_check(text, answer, ended):
    return {'answer_correct': text.strip() == answer,
            'completed_correct': ended and text.strip() == answer,
            'ended_eos': ended, 'truncated': not ended}


@torch.inference_mode()
def run_policy(model, tokenizer, source, case, settings, policy, protected=(),
               backend='packed', forced=None):
    clean()
    sync()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    started = time.perf_counter()
    native = policy == 'native_fp16'
    bits = 16 if policy == 'paged_fp16' else 8 if policy == 'int8' else 4
    cache = stock_cache(source) if native else make_live_cache(source, settings, protected, bits)
    sync()
    setup_seconds = time.perf_counter() - started
    payload_before = cache.report() if not native else {'tokens': cache.get_seq_length()}
    decode_started = time.perf_counter()
    outputs, logits = [], []
    token = case['ids'][-1]
    stopped = False
    steps = len(forced) if forced is not None else settings.max_new_tokens
    route_cache = None if native else cache
    with attention_route(model, cache=route_cache, backend=backend):
        for step in range(steps):
            output = forward_token(model, cache, token, packed=not native)
            current = output.logits[0, -1].float()
            logits.append(current.detach().cpu())
            # Gold tokens condition the diagnostic replay only, never the free generation.
            token = int(forced[step]) if forced is not None else int(current.argmax())
            outputs.append(token)
            if forced is None and token in eos_ids(model, tokenizer):
                stopped = True
                break
    sync()
    decode_seconds = time.perf_counter() - decode_started
    # Starts before cache transfer, packing, table creation, materialization and generation.
    total_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    live_alloc = torch.cuda.memory_allocated()
    if native:
        physical = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
        report = {'resident_bytes': physical, 'payload_bytes': physical,
                  'demotions': 0, 'tokens': cache.get_seq_length()}
    else:
        report = cache.report()
    text = tokenizer.decode(outputs, skip_special_tokens=True)
    row = {'policy': policy, 'backend': 'stock_sdpa' if native else backend,
           'case_id': case['case_id'], 'text': text, 'generated_tokens': len(outputs),
           **answer_check(text, case['answer'], stopped), **report,
           'initial_resident_bytes': payload_before.get('resident_bytes'),
           'setup_seconds': setup_seconds, 'decode_seconds': decode_seconds,
           'decode_tokens_per_second': len(outputs) / decode_seconds,
           'cache_plus_decode_seconds': total_seconds,
           'peak_allocated_gib': peak / 2**30,
           'peak_incremental_mib': (peak - base_alloc) / 2**20,
           'end_incremental_mib': (live_alloc - base_alloc) / 2**20}
    del cache, output, current
    clean()
    return row, torch.stack(logits)


@torch.inference_mode()
def kernel_gate():
    from .triton_decode import packed_attention
    checks = []
    for dim in (64, 128):
        for length in (1, 15, 16, 17, 33, 65, 97):
            for bits in (4, 8, 16):
                layer = PagedLayer(2, dim, 128, archive_bits=bits, protected=(0,), recent_pages=1)
                k = torch.randn(1, 2, length, dim, device='cuda', dtype=torch.float16)
                v = torch.randn_like(k)
                # Append in uneven increments to exercise partial blocks and online demotion.
                for begin in range(0, length, 7):
                    layer.append(k[:, :, begin:begin + 7], v[:, :, begin:begin + 7])
                q = torch.randn(1, 4, 1, dim, device='cuda', dtype=torch.float16)
                reference = dense_attention(q, layer, dim ** -0.5)
                actual = packed_attention(q, layer, dim ** -0.5)
                torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.003)
                checks.append({'dim': dim, 'tokens': length, 'bits': bits,
                               'max_error': float((actual - reference).abs().max()),
                               'demotions': layer.demotions})
    return checks


@torch.inference_mode()
def model_gate(model, tokenizer, source, case, settings):
    forced = tokenizer(case['answer'], add_special_tokens=False).input_ids
    _, stock = run_policy(model, tokenizer, source, case, settings, 'native_fp16', forced=forced)
    checks = []
    for policy, protected in [('paged_fp16', ()), ('int8', ()), ('int4', ()), ('oracle', (0, 2))]:
        _, dense = run_policy(model, tokenizer, source, case, settings, policy,
                              protected, backend='dense', forced=forced)
        _, packed = run_policy(model, tokenizer, source, case, settings, policy,
                               protected, backend='packed', forced=forced)
        discrepancy = float(kl_divergence(dense, packed).mean())
        if discrepancy > 0.002 or not torch.isfinite(packed).all():
            raise AssertionError(f'kernel/model mismatch for {policy}: KL={discrepancy}')
        if policy == 'paged_fp16':
            torch.testing.assert_close(dense, stock, atol=0.05, rtol=0.01)
            if float(kl_divergence(stock, packed).mean()) > 0.002:
                raise AssertionError('FP16 packed route does not match the stock model')
        checks.append({'policy': policy, 'mean_kl_dense_vs_packed': discrepancy,
                       'max_logit_error': float((dense - packed).abs().max())})
    return checks


@torch.inference_mode()
def sensitivity_audit(model, source, case, settings, scores, reference):
    """Diagnostic only: never feed these measured labels back into the selector."""
    from .packed_decode import rank_correlation
    candidates = sorted(scores)
    if len(candidates) > settings.sensitivity_pages:
        indices = torch.linspace(0, len(candidates) - 1, settings.sensitivity_pages).round().int().tolist()
        candidates = [candidates[i] for i in indices]
    rows = []
    for page in candidates:
        cache = make_live_cache(source, settings, bits=16)
        for layer in cache.stores:
            layer.demote(page, 4)
        with attention_route(model, cache, backend='dense'):
            output = forward_token(model, cache, case['ids'][-1], packed=True)
        measured = float(kl_divergence(reference, output.logits[0, -1].float().cpu()))
        rows.append({'page': page, 'automatic_score': scores[page], 'measured_kl': measured})
        del cache, output
    correlation = rank_correlation([r['automatic_score'] for r in rows], [r['measured_kl'] for r in rows])
    return {'case_id': case['case_id'], 'spearman': correlation, 'pages': rows,
            'scope': 'single-page INT4 intervention; first answer-token distribution only'}


def bootstrap_interval(values, seed=7, iterations=2000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return [means[int(iterations * 0.025)], means[int(iterations * 0.975)]]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
