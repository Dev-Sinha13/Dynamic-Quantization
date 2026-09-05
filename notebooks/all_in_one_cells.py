# %% [markdown]
# # AnchorKV: complete T4 research notebook
#
# Upload this one notebook to a **fresh Colab T4 runtime**, then run all cells.
# No GitHub checkout, previous notebook, HF token, or results zip is required.
# It includes: validated chat prompts and answers; equal-byte-budget selectors;
# automatic query/error scoring and a quantization-sensitivity audit; continuously
# demoted KV pages; experimental packed Triton attention; repeated measurements;
# and a downloadable report with machine-readable results and source snapshots.
#
# Default pilot: six prompts, two context lengths, three evidence positions,
# three timing repetitions. Allow tens of minutes; Python page allocation and
# reference fallbacks can be slow. A larger suite is configurable below.
# The automatic score is a new hypothesis, not an already validated thought-anchor
# detector. The oracle uses labeled evidence solely as a comparison. Packed
# attention must pass numerical tests on your GPU before its results are accepted.
#
# Scope: Qwen3-0.6B, FP16 weights, batch one, full attention, single-token decode.
# This is an experimental kernel and benchmark, not a vLLM integration or a full
# reproduction of model-generated Declarative Attention tags.

# %%
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'transformers==4.57.6', 'huggingface_hub>=0.34,<1',
                       'accelerate>=1,<2', 'pandas', 'matplotlib'])

# %%
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time
import traceback

import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers

assert transformers.__version__ == '4.57.6', 'Restart the Colab session after installing, then run from the top.'
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__, '| Transformers:', transformers.__version__)

# %%
# EMBED_RUNTIME

# %%
from anchorkv_notebook.colab_experiment import (
    Settings, asdict, build_case, prefill_case, automatic_scores, choose_pages,
    run_policy, kernel_gate, model_gate, sensitivity_audit, bootstrap_interval,
    save_json, clean, sync, eos_ids,
)
from anchorkv_notebook.packed_decode import kl_divergence
from transformers import AutoTokenizer, AutoModelForCausalLM

settings = Settings()
# Optional expanded run (slower):
# settings.lengths = (384, 896, 1408)
# settings.repeats = 5
# settings.sensitivity_pages = 16
# Re-run with different settings.seed values for broader evidence.
REQUEST_PACKED_KERNEL = True
RUN_SENSITIVITY_AUDIT = True
OUTPUT = Path('/content/anchorkv-complete-results') / time.strftime('%Y%m%d-%H%M%S')
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
random.seed(settings.seed)
torch.set_num_threads(2)

tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
cases = []
for target in settings.lengths:
    for position in settings.positions:
        cases.append(build_case(tokenizer, settings, len(cases), target, position))
assert all(len(case['ids']) <= settings.max_prompt_tokens for case in cases)
display(pd.DataFrame([{'case': c['case_id'], 'tokens': len(c['ids']),
                      'evidence_position': c['position'], 'answer': c['answer']} for c in cases]))
save_json(OUTPUT / 'prompts.json', cases)
save_json(OUTPUT / 'settings.json', asdict(settings))

# %% [markdown]
# ## Load weights after validating every prompt
#
# Prompt length is checked before GPU model allocation. Each prompt uses Qwen's
# chat template with thinking disabled for the retrieval benchmark. Generated
# answers must equal the four-digit code; EOS completion is scored separately.

# %%
model = AutoModelForCausalLM.from_pretrained(
    settings.model_id, revision=settings.revision, torch_dtype=torch.float16,
    attn_implementation='sdpa', low_cpu_mem_usage=True,
).to('cuda').eval()
assert not getattr(model.model, 'has_sliding_layers', False), 'Only full-attention Qwen3 is supported.'
assert model.config.head_dim == 128
try:
    triton_version = importlib.metadata.version('triton')
except importlib.metadata.PackageNotFoundError:
    triton_version = None
environment = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'transformers': transformers.__version__, 'triton': triton_version,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
    'compute_capability': list(torch.cuda.get_device_capability(0)),
    'model_id': settings.model_id, 'model_revision': settings.revision,
    'source_sha256': EMBEDDED_SOURCE_SHA256,
}
save_json(OUTPUT / 'environment.json', environment)
print(environment)

# %% [markdown]
# ## Numerical gates for the packed kernel
#
# The first test compares 42 combinations of length, dtype, and head dimension
# against dense attention, including partial pages, GQA, online demotion, and
# empty split partitions. The second compares full-model logits across stock
# FP16, reconstructed-cache attention, and packed attention. A failure is saved
# as a failure and the remaining quality experiment uses the dense reference.
# Kernel compilation and these warm-ups are excluded from timed measurements.

# %%
gates = {'requested_packed': REQUEST_PACKED_KERNEL, 'packed_status': 'not_requested'}
BACKEND = 'dense'
if REQUEST_PACKED_KERNEL:
    try:
        gates['attention_checks'] = kernel_gate()
        source = prefill_case(model, cases[0])
        gates['model_checks'] = model_gate(model, tokenizer, source, cases[0], settings)
        del source
        gates['packed_status'] = 'passed'
        BACKEND = 'packed'
    except Exception:
        gates['packed_status'] = 'failed'
        gates['error'] = traceback.format_exc()
        print(gates['error'])
        print('Continuing with dense reconstruction. This run cannot validate packed-kernel performance.')
save_json(OUTPUT / 'correctness-gates.json', gates)
print('Selected backend:', BACKEND, '| gate:', gates['packed_status'])
clean()

# %% [markdown]
# ## Automatic selection and equal-budget evaluation
#
# Recent, random, automatic, and evidence-oracle selectors preserve exactly the
# same number of full FP16 pages. They also share one sink page and the rolling
# recent window. All other eligible pages use INT4. Initial resident bytes,
# including scales and pointer tables, must be identical or the experiment stops.
# Residual INT4, residual INT8, stock FP16, and paged FP16 are additional baselines.
#
# Automatic scores combine attention from the final prompt query with measured
# K/V reconstruction error, averaged over the final four layers. Selection never
# sees the gold answer or labeled evidence. Selector preparation time is reported
# separately and included in its accounting. This selection is fixed per prompt;
# age-based precision changes continue throughout decoding.
#
# Every policy recomputes the final prompt token from the intervened prefix cache,
# so the first generated answer token is independently evaluated. The gold answer
# is used only for the separate teacher-forced diagnostic, including its EOS.

# %%
raw_rows = []
diagnostics = []
selection_records = []
audits = []
for case in cases:
    print('\nStarting', case['case_id'], case['position'], 'tokens=', len(case['ids']), flush=True)
    sync()
    prefill_started = time.perf_counter()
    source = prefill_case(model, case)
    sync()
    prefill_seconds = time.perf_counter() - prefill_started
    prefix_length = len(case['ids']) - 1
    candidates = list(range(1, prefix_length // settings.block - settings.recent_pages))
    assert candidates, 'Increase context length to leave eligible archive pages.'
    count = max(1, int(len(candidates) * settings.keep_fraction))
    scores, reference_first, score_seconds = automatic_scores(model, source, case, candidates, settings)
    plans = {'native_fp16': frozenset(), 'paged_fp16': frozenset(),
             'int8': frozenset({0}), 'int4': frozenset({0})}
    for name in ('recent', 'random', 'automatic', 'oracle'):
        plans[name] = choose_pages(name, candidates, count, scores=scores,
                                   evidence=case['evidence_pages'], seed=settings.seed + len(selection_records)) | {0}
    selection_records.append({'case_id': case['case_id'], 'eligible_pages': candidates,
                              'extra_fp16_pages': count, 'automatic_scores': scores,
                              'selection_seconds': score_seconds,
                              'plans': {name: sorted(pages) for name, pages in plans.items()}})
    if RUN_SENSITIVITY_AUDIT:
        audits.append(sensitivity_audit(model, source, case, settings, scores, reference_first))
        save_json(OUTPUT / 'sensitivity-audit.json', audits)

    # Same gold token history, including EOS, for all policies.
    forced = tokenizer(case['answer'], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    _, reference = run_policy(model, tokenizer, source, case, settings,
                              'native_fp16', forced=forced, backend=BACKEND)
    initial_bytes = {}
    for policy, pages in plans.items():
        row, logits = run_policy(model, tokenizer, source, case, settings,
                                 policy, pages, backend=BACKEND, forced=forced)
        if policy in ('recent', 'random', 'automatic', 'oracle'):
            initial_bytes[policy] = row['initial_resident_bytes']
        divergence = kl_divergence(reference, logits)
        nll = -logits.log_softmax(-1).gather(-1, torch.tensor(forced).unsqueeze(-1)).mean()
        diagnostics.append({'case_id': case['case_id'], 'policy': policy,
                            'mean_kl': float(divergence.mean()), 'max_kl': float(divergence.max()),
                            'top1_agreement': float((reference.argmax(-1) == logits.argmax(-1)).float().mean()),
                            'gold_nll': float(nll), 'first_answer_kl': float(divergence[0]),
                            'initial_resident_bytes': row['initial_resident_bytes']})
        # Full free-generation warm-up, so the same allocation and decode paths are exercised.
        run_policy(model, tokenizer, source, case, settings, policy, pages, backend=BACKEND)
    assert len(set(initial_bytes.values())) == 1, f'Budget mismatch: {initial_bytes}'

    for repetition in range(settings.repeats):
        order = list(plans)
        random.Random(settings.seed + repetition + len(raw_rows)).shuffle(order)
        for policy in order:
            row, _ = run_policy(model, tokenizer, source, case, settings,
                                 policy, plans[policy], backend=BACKEND)
            selector_cost = score_seconds if policy == 'automatic' else 0.0
            row.update({'repeat': repetition, 'prompt_tokens': len(case['ids']),
                        'evidence_position': case['position'], 'prefill_seconds': prefill_seconds,
                        'selector_seconds': selector_cost,
                        'accounted_total_seconds': prefill_seconds + selector_cost + row['cache_plus_decode_seconds']})
            raw_rows.append(row)
        # Persist progress after each repetition, even if the session later disconnects.
        save_json(OUTPUT / 'raw-results.json', raw_rows)
        save_json(OUTPUT / 'teacher-forced.json', diagnostics)
        save_json(OUTPUT / 'selection.json', selection_records)
        print('Finished repetition', repetition + 1, '/', settings.repeats, flush=True)
    del source
    clean()

# %% [markdown]
# ## Continuous-generation stress check
#
# Retrieval answers are short, so they do not sufficiently exercise aging of
# newly generated pages. This separate test forces 96 continuation tokens solely
# to check cache lifecycle behavior. It is not an answer-accuracy measurement.
# More archived pages must exist at the end than in the prefix-only cache.

# %%
stress_case = cases[0]
stress_source = prefill_case(model, stress_case)
stress_settings = Settings(**asdict(settings))
stress_settings.max_new_tokens = 128
forced_stress = (tokenizer(' diagnostic continuation', add_special_tokens=False).input_ids * 96)[:96]
stress_row, _ = run_policy(model, tokenizer, stress_source, stress_case, stress_settings,
                           'int4', {0}, backend=BACKEND, forced=forced_stress)
initial_demotions_per_layer = max(0, (len(stress_case['ids']) - 1) // settings.block - settings.recent_pages - 1)
assert stress_row['demotions'] > initial_demotions_per_layer * model.config.num_hidden_layers
stress_row['scope'] = 'forced continuation lifecycle test; exclude from answer-accuracy statistics'
save_json(OUTPUT / 'continuous-cache-check.json', stress_row)
print('Continuous cache check passed:', stress_row['tokens'], 'tokens;', stress_row['demotions'], 'layer-page demotions')
del stress_source
clean()

# %% [markdown]
# ## Per-case results, uncertainty, and report
#
# Repetitions are timing replicates, not independent questions. Confidence
# intervals resample cases after reducing repetitions. Six pilot questions give
# weak generalization evidence; expand prompts/seeds before making a broad claim.
# Prefill is measured once per case and amortized equally. Accounted total time
# includes CPU snapshot/transfer, automatic selection where used, cache setup,
# and decode. Separate measured cache-plus-decode times are also retained.
# Prefill's peak GPU allocation is outside the decode-stage memory window.

# %%
raw = pd.DataFrame(raw_rows)
quality = pd.DataFrame(diagnostics)
per_case = raw.groupby(['case_id', 'policy'], as_index=False).agg(
    answer_correct=('answer_correct', 'mean'), completed_correct=('completed_correct', 'mean'),
    peak_allocated_gib=('peak_allocated_gib', 'median'),
    peak_incremental_mib=('peak_incremental_mib', 'median'),
    end_incremental_mib=('end_incremental_mib', 'median'),
    resident_bytes=('resident_bytes', 'median'),
    cache_plus_decode_seconds=('cache_plus_decode_seconds', 'median'),
    accounted_total_seconds=('accounted_total_seconds', 'median'),
)
per_case = per_case.merge(quality, on=['case_id', 'policy'])
summary = per_case.groupby('policy', as_index=False).mean(numeric_only=True)
summary['resident_mib'] = summary.resident_bytes / 2**20
display(summary[['policy', 'completed_correct', 'resident_mib', 'mean_kl',
                 'peak_incremental_mib', 'accounted_total_seconds']].round(5))

paired = []
for control in ('recent', 'random', 'oracle', 'int4'):
    a = per_case[per_case.policy == 'automatic'].set_index('case_id')
    b = per_case[per_case.policy == control].set_index('case_id')
    for metric in ('mean_kl', 'completed_correct', 'accounted_total_seconds'):
        differences = (a[metric] - b[metric]).tolist()
        paired.append({'comparison': 'automatic minus ' + control, 'metric': metric,
                       'mean_difference': statistics.mean(differences),
                       'case_bootstrap_95pct': bootstrap_interval(differences, settings.seed)})
save_json(OUTPUT / 'paired-comparisons.json', paired)
raw.to_csv(OUTPUT / 'raw-results.csv', index=False)
per_case.to_csv(OUTPUT / 'per-case.csv', index=False)
summary.to_csv(OUTPUT / 'summary.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].bar(summary.policy, summary.resident_mib)
axes[0].set_ylabel('End-of-run KV payload + metadata (MiB)')
axes[1].bar(summary.policy, summary.completed_correct)
axes[1].set_ylabel('Completed exact-answer fraction')
axes[1].set_ylim(0, 1.05)
axes[2].scatter(summary.resident_mib, summary.mean_kl)
for row in summary.itertuples():
    axes[2].annotate(row.policy, (row.resident_mib, row.mean_kl), fontsize=8)
axes[2].set_xlabel('Resident cache (MiB)')
axes[2].set_ylabel('Mean teacher-forced KL vs stock FP16')
for ax in axes[:2]:
    ax.tick_params(axis='x', rotation=65)
fig.tight_layout()
fig.savefig(OUTPUT / 'summary.png', dpi=160, bbox_inches='tight')
plt.show()

report = [
    '# AnchorKV T4 experiment report', '',
    f'GPU: {environment["gpu"]}; model: {settings.model_id}@{settings.revision}.',
    f'Backend: {BACKEND}; packed correctness gate: {gates["packed_status"]}.',
    f'Independent cases: {len(cases)}; timing repetitions: {settings.repeats}.', '',
    '## Results', '', '```', summary.to_string(index=False), '```', '',
    '## Interpretation boundaries', '',
    '- Equal budgets apply to recent/random/automatic/oracle initial prefix caches.',
    '- INT4 and INT8 baselines keep the same FP16 sink and rolling recent window.',
    '- Automatic scores use only prompt-query attention and KV quantization error.',
    '- Oracle selection uses labeled evidence; it is not an automatic result.',
    '- Cache allocation/packing and CPU transfer are included in cache-plus-decode time.',
    '- Automatic selection and shared prefill are included in accounted total time.',
    '- Decode-stage peak memory excludes prefill; payload bytes include scales and tables.',
    '- Final cache sizes may differ because policies can generate different token counts.',
    '- The pilot is too small for broad accuracy claims. Repeated timings are not extra cases.',
    '- Production batching, serving integration, and model-generated directive control remain future work.',
]
(OUTPUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
print('Report:', OUTPUT / 'report.md')

# %% [markdown]
# ## Download the complete evidence bundle
#
# Return this zip for analysis. It includes raw generations, exact prompts,
# settings, package/model/source versions, selector choices, numerical gates,
# lifecycle tests, per-case tables, confidence intervals, and the report.
# Passing the notebook establishes results for this configuration. Remaining
# research is replication across larger prompt sets, seeds, models, and longer
# reasoning tasks. Production use additionally needs batching, an allocator and
# serving-engine integration, and performance tuning against strong baselines.

# %%
source_dir = OUTPUT / 'runtime-source'
source_dir.mkdir(exist_ok=True)
for filename, source in EMBEDDED_SOURCES.items():
    (source_dir / filename).write_text(source, encoding='utf-8')
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('Results:', archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print('Download the zip from the path above.')
