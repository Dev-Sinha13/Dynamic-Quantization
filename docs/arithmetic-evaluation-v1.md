# Frozen arithmetic evaluation v1

Status: **local validation complete; FP16 development readiness not yet measured**.
No model was run on the held-out set. This protocol does not establish accuracy
or compression benefit. The earlier two questions and their post-hoc report
remain unchanged and are not members of this dataset.

## What is frozen

- Scorer: `arithmetic-final-v1`, implemented in `arithmetic_scoring.py`.
- Protocol: `arithmetic-protocol-v1`, pinned Qwen3-0.6B revision, FP16 weights,
  SDPA, greedy generation, thinking disabled, 64 new tokens, 1536 prompt cap.
- [Manifest and source hashes](experiments/data/arithmetic-protocol-v1/manifest.json).
- [Development data](experiments/data/arithmetic-protocol-v1/development.json):
  six numerical problems, each in clean and middle-distractor contexts: 12 prompts.
- [Held-out data](experiments/data/arithmetic-protocol-v1/heldout.json): 12 fresh
  problems with clean/early/middle/late contexts: 48 prompts. Operand tuples and
  family IDs are disjoint from development. Answer values need not be unique.

The operations are addition, subtraction, and multiplication followed by
subtraction. Evidence is explicit inventory data, not ambiguous retired records.
These are synthetic problems, not a standardized reasoning benchmark. Clean and
distractor variants isolate arithmetic ability from retrieval difficulty; they
are matched views of the same problem, not independent samples. The tokenizer
checks every complete prompt without truncating records or query text.

## Scoring rules

Preserve both strict text match and strict completed match. Separately report
parsed final answer, parse coverage, completion and completed final correctness.
Final extraction sees only text, never gold. It accepts a bare terminal integer,
an explicit answer label, supported remaining-units phrasing, or a syntactically
complete integer equation on the final nonempty line. It rejects ambiguous
alternatives, non-ASCII numbers, decimals, inequalities, incomplete arithmetic,
oversized responses and unsupported formats. No contains-gold or last-number
fallback. A correct-looking truncated output is not a completed success.

An extracted final value does not certify the reasoning. An unparsed response
counts as no demonstrated success, but is not labeled a proven arithmetic error.
All rejected responses must remain available for audit. Version 1 was designed
after inspecting the old pilot, but before any model outputs from these new
splits. Do not describe the old rescoring as prospective or change its results.

## Next model run: development only

Upload [AnchorKV_T4_FP16_Development.ipynb](../notebooks/AnchorKV_T4_FP16_Development.ipynb)
to a fresh T4 runtime and Run all. It performs only **12 FP16 generations**, with
an eight-minute soft generation-phase limit, checkpoints and zip export. Setup
is outside the limit. In-flight generation can overrun it. If paused, rerun the
generation, report and download cells. Preserve the directory to resume; use a
persistent output location if the runtime may be lost.

Predeclared readiness gate: all 12 results must exist, at least **11/12** must
have correct completed final answers, at least **3/4 per operation**, and at
least **5/6 per context type**. This is an engineering threshold, not a
statistical claim. Report strict-format and parsing failures separately.

If the gate fails, review development failures only. Do not drop difficult
examples or run compression policies to pick favorable prompts. Any protocol,
prompt, model-setting or scorer revision must get a new version and manifest.
Do not edit v1 in place or reuse its checkpoint under changed settings.

## After readiness, not before

Freeze a held-out run manifest including equal byte budgets, random seeds,
generation settings and timing scope before collecting outputs. Evaluate every
policy, including fresh FP16, on every held-out case. Do not filter held-out data
to FP16-correct examples as the primary result. Report aggregate accuracy and
paired losses/gains relative to FP16; an FP16-correct subgroup may be secondary
and must be labeled. Average random draws within case and contexts within family
before estimating uncertainty. These 48 prompts represent only 12 families.

Avoid tuning the scorer on held-out outputs. Any later changes require a new
untouched evaluation set. Multi-model, standard-benchmark and production-serving
validation remain outside this milestone.

## Reproduce local artifacts

```powershell
$env:PYTHONPATH = 'src'
python -m anchorkv.arithmetic_protocol path/to/new-output-directory
python notebooks/build_baseline.py
python -m unittest discover -s tests -p 'test_arithmetic*.py' -v
```

The export refuses to overwrite a frozen directory. Tests verify source/data
hashes, exact answers, split separation, malformed-output handling, gate failure
conditions, token limits with the optional cached Qwen tokenizer, and mocked
notebook pause/resume behavior. CPU mocks do not validate model accuracy.
