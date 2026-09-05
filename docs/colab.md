# T4 Colab handoff

**Current complete workflow:** upload
[`AnchorKV_T4_All_In_One.ipynb`](../notebooks/AnchorKV_T4_All_In_One.ipynb)
to a fresh T4 runtime. It combines quality evaluation, equal budgets, automatic
selection diagnostics, online cache aging, experimental packed attention, and
report export. See [the complete guide](all-in-one-colab.md). The earlier
notebooks described below remain available to reproduce previous experiments.

The [first T4 smoke-run postmortem](experiments/2026-09-01-t4-smoke.md)
documents why the initial truncated artifacts were rejected and what changed
before the next run.
The [non-thinking rerun](experiments/2026-09-03-t4-nonthinking.md) completed but
was rejected because two answers were wrong and its 2–3 sentence traces make
kurtosis degenerate.
The [320-token thinking-mode run](experiments/2026-09-03-t4-thinking-320.md)
validated one sample, then correctly stopped when the second sample did not
reach EOS within its generation allowance.
The [successful 512-token T4 run](experiments/2026-09-03-t4-thinking-512.md)
validated execution and trace quality; its initial head manifest was rejected
after exposing a terminal-answer/EOS attention confound.

The repository provides bounded GPU workflows for collecting sentence-level
attention traces and testing physical KV requantization on Qwen3-0.6B.

There are now two standalone T4 workflows:

- `AnchorKV_T4_Standalone.ipynb` reproduces attention-head discovery and causal
  sentence suppression.
- `AnchorKV_T4_Requantization.ipynb` captures Qwen3-0.6B's real KV tensors,
  physically packs semantic segments, reconstructs a live `DynamicCache`, and
  measures storage, controlled logit drift, greedy drift, latency, and peak CUDA
  allocation.

## Start the notebook

Open [the AnchorKV T4 notebook](../notebooks/AnchorKV_T4_Trace_Collection.ipynb)
or use the Colab badge in the main README. In Colab, select **Runtime → Change
runtime type → T4 GPU** before running any cells.

For a private repository, download
[`AnchorKV_T4_Standalone.ipynb`](../notebooks/AnchorKV_T4_Standalone.ipynb),
then choose **File → Upload notebook** in Colab. It embeds the required helpers
and does not require GitHub access.

To test the new requantization backend instead, upload
[`AnchorKV_T4_Requantization.ipynb`](../notebooks/AnchorKV_T4_Requantization.ipynb).
It is also standalone and uses a 640-token prompt cap, 32 greedy decode tokens,
and 24 controlled teacher-forced steps.

## Physical requantization workflow

The requantization notebook compares four policies over the same captured
prompt cache:

1. FP16 reference storage.
2. Uniform groupwise INT8.
3. Uniform packed INT4.
4. Declarative INT4, where `<anchor>` keeps the instruction, relevant evidence,
   and query in FP16 while `<archive>` demotes distractor segments.

It saves `requantization-results.json`, a CSV table, and a storage–fidelity plot.
The JSON pins the resolved model revision and records exact packed payload and
scale bytes. Teacher-forced KL and top-1 agreement use identical token histories
across policies; greedy common-prefix length exposes compounding decode changes.

The notebook intentionally reports CUDA memory as
`peak_gpu_gib_dense_reference`. Hugging Face's stock attention path consumes a
dense materialized cache, so that number must not be presented as a packed-cache
GPU saving. The physical `resident_bytes` result is valid; a fused or paged
mixed-precision attention kernel is still required to realize that saving during
attention.

## Trace collection workflow

The notebook will:

1. Clone and install the current GitHub repository.
2. Verify CUDA and print the assigned GPU.
3. Estimate the lower-bound eager-attention memory requirement.
4. Resolve and record the exact Hugging Face model commit.
5. Generate three short reasoning traces with SDPA.
6. Replay each completed trace with eager attention.
7. Immediately reduce full attention to `[layers, heads, sentences]` scores.
8. Save pickle-free compressed `.npz` artifacts.
9. Discover receiver heads and save a JSON manifest.
10. Download a zip archive of the run.

## Safe first-run configuration

```text
Model: Qwen/Qwen3-0.6B
Precision: FP16
Batch size: 1
Maximum total sequence: 768 tokens
Maximum generated tokens: 512
Thinking mode: enabled, with EOS and answer validation
Minimum reasoning spans: 6
Minimum downstream horizon per candidate: 32 tokens
Samples: 3
```

Do not increase all dimensions at once. After a successful run, increase the
sequence limit to 1,024 while keeping batch size one. Record
`torch.cuda.max_memory_allocated()` for every sample.

## Expected artifacts

```text
artifacts/colab/
  math-000.npz
  math-001.npz
  math-002.npz
  run-summary.json
  receiver-heads.json
  receiver-heads.png
  causal-results.json
  selector-causal-kl.png
  attention-causal-scatter.png
```

Each `.npz` contains:

- Sentence-level vertical attention scores
- Token spans and decoded sentence text
- Model ID and exact model revision
- Prompt hash
- Seed and FP16 dtype
- Sequence and head geometry

The prompt itself is deliberately represented by a hash in the portable trace.
The notebook source remains the record of the public benchmark prompts.
`run-summary.json` records the GPU, package versions, generation configuration,
completed text, token counts, and peak allocated GPU memory for each sample.
`causal-results.json` records teacher-forced downstream KL and NLL changes for
every eligible sentence over one shared evaluation window. It reports Spearman
rank correlation, top-k overlap, and causal regret for raw receiver-head,
cross-head-normalized, and all-head attention. Recency, random selection, and a
causal oracle are included as selector baselines. The sentence mask broadcasts
across all heads, so this evaluates sentence-level causality rather than
head-specific causality.

## If the notebook runs out of memory

1. Restart the runtime to clear all CUDA state.
2. Reduce `max_sequence_length` from 768 to 512.
3. Reduce `max_new_tokens` from 512 to 384 only if generations still reach EOS.
4. Keep batch size one.
5. Do not request hidden states or scores from generation.

An out-of-memory run is still an experiment record, but do not report simulated
cache bytes as measured GPU savings.

## Boundary after this notebook

The notebook discovers correlational receiver heads and measures causal effects
for every eligible sentence. A larger prompt set is required before deciding
whether any attention proxy is predictive enough to drive cache allocation.
Head-specific suppression and a live cache policy should only follow a positive,
confidence-bounded ranking result.
