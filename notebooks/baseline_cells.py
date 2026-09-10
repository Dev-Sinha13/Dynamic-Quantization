# %% [markdown]
# # AnchorKV: FP16 arithmetic development gate
#
# Fresh Colab T4, Run all. **Only 12 FP16 generations**, no compression sweep,
# Triton compilation, throughput benchmark or held-out model evaluation.
# This checks readiness; the baseline is NOT already validated.
# Six underlying arithmetic problems each have a clean and distractor context.
# Cap: 64 generated tokens. Soft phase limit: 8 minutes after loading weights.
# Checkpoints resume by rerunning the generation cell in the same live runtime.
# Downloads/setup and in-flight generation are outside the soft limit.
# Preserve the output folder on Drive if you need to survive runtime loss.
# Return the final zip whether the gate passes or fails; do not drop hard cases.

# %%
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'transformers==4.57.6', 'huggingface_hub>=0.34,<1', 'accelerate>=1,<2'])

# %%
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
assert transformers.__version__ == '4.57.6', 'Restart the runtime after installing.'
assert torch.cuda.is_available(), 'Select a T4 GPU runtime.'
# EMBED_BASELINE

# %%
from anchorkv_baseline.arithmetic_protocol import (
    MODEL_ID, MODEL_REVISION, SCORER_VERSION, PROTOCOL_VERSION,
    canonical_hash, render_case, development_gate,
)
from anchorkv_baseline.run_control import RunControl
RESUME_DIRECTORY = None
PHASE_MINUTES = 8
OUTPUT = Path(RESUME_DIRECTORY) if RESUME_DIRECTORY else Path('/content/anchorkv-fp16-development') / time.strftime('%Y%m%d-%H%M%S')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
cases = [render_case(c, tokenizer) for c in DEVELOPMENT_CASES]
environment = {'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'torch': str(torch.__version__),
               'transformers': transformers.__version__, 'python': platform.python_version(),
               'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
if RESUME_DIRECTORY and not (OUTPUT / 'resume-manifest.json').exists():
    raise ValueError('Resume directory lacks a compatible manifest; start a new run.')
checkpoint = RunControl(OUTPUT, {'environment': environment, 'source_sha256': SOURCE_SHA256,
    'workflow_sha256': WORKFLOW_SHA256, 'cases_sha256': canonical_hash(DEVELOPMENT_CASES),
    'max_new_tokens': 64, 'do_sample': False, 'protocol_version': PROTOCOL_VERSION,
    'scorer_version': SCORER_VERSION, 'prompt_hashes': [c['prompt_sha256'] for c in cases]})
checkpoint.save('prompts.json', cases)
checkpoint.save('environment.json', environment)
print('12 native FP16 generations; no held-out cases or compressed policies.')
print('Output/resume folder:', OUTPUT)
for case in cases:
    print(case['case_id'], len(case['ids']), 'prompt tokens')

# %%
torch.manual_seed(20260910)
torch.cuda.manual_seed_all(20260910)
torch.set_num_threads(2)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
    torch_dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True).to('cuda').eval()

# %%
rows = checkpoint.load('development-results.json')
# Validate loaded identities/completion records before trusting them as completed work.
development_gate(DEVELOPMENT_CASES, rows)
done = {r['case_id'] for r in rows}
checkpoint.start('development', PHASE_MINUTES)
stop_ids = model.generation_config.eos_token_id
stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids]) | {tokenizer.eos_token_id}
stop_ids.discard(None)
for case in cases:
    if case['case_id'] in done:
        continue
    if not checkpoint.allow():
        break
    checkpoint.log(f"Generating {case['case_id']} ({len(done)}/12 saved)")
    input_ids = torch.tensor([case['ids']], dtype=torch.long, device=model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=64, do_sample=False, use_cache=True,
            eos_token_id=sorted(stop_ids), pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = output[0, input_ids.shape[1]:].tolist()
    ended = bool(tokens and tokens[-1] in stop_ids)
    rows.append({'case_id': case['case_id'], 'policy': 'native_fp16',
        'messages_sha256': case['messages_sha256'], 'prompt_sha256': case['prompt_sha256'],
        'text': tokenizer.decode(tokens, skip_special_tokens=True), 'generated_ids': tokens,
        'ended_eos': ended, 'truncated': not ended, 'generation_seconds': seconds})
    checkpoint.save('development-results.json', rows)
    done.add(case['case_id'])
    checkpoint.log(f'Saved {len(done)}/12')
    del input_ids, output
    torch.cuda.empty_cache()

# %%
gate = development_gate(DEVELOPMENT_CASES, checkpoint.load('development-results.json'))
checkpoint.save('development-gate.json', gate)
print('Gate:', gate['status'], '| correct final answers:', gate['correct'], '/', gate['expected'])
print('By operation:', gate['correct_by_operation'])
print('By context:', gate['correct_by_position'])
print('Pass requires at least 11/12 overall, 3/4 per operation, and 5/6 per context.')
if gate['status'] == 'incomplete':
    print('Rerun generation, then this report and download cell. Completed cases are skipped.')
elif gate['status'] == 'failed':
    print('Do not run the held-out compression sweep yet. Return the failures for development review.')
else:
    print('Development readiness passed. Held-out performance and compression benefit remain untested.')
report = ['# FP16 development readiness', '', json.dumps(gate, indent=2), '',
          'Strict and final-answer scores are separate. Unparsed is not a proven arithmetic error.',
          'Matched contexts are not independent questions. This is six underlying problems.',
          'No held-out inference or compression comparison was performed.']
(OUTPUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')

# %%
source_dir = OUTPUT / 'runtime-source'
source_dir.mkdir(exist_ok=True)
for name, source in SOURCES.items():
    (source_dir / name).write_text(source, encoding='utf-8')
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('Results:', archive)
from google.colab import files
files.download(archive)
