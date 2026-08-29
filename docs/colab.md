# T4 Colab handoff

The repository is ready for its first real GPU-only step: collecting bounded
sentence-level attention traces from Qwen3-0.6B.

## Start the notebook

Open [the AnchorKV T4 notebook](../notebooks/AnchorKV_T4_Trace_Collection.ipynb)
or use the Colab badge in the main README. In Colab, select **Runtime → Change
runtime type → T4 GPU** before running any cells.

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
Maximum generated tokens: 128
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
  receiver-heads.json
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

## If the notebook runs out of memory

1. Restart the runtime to clear all CUDA state.
2. Reduce `max_sequence_length` from 768 to 512.
3. Reduce `max_new_tokens` from 128 to 64.
4. Keep batch size one.
5. Do not request hidden states or scores from generation.

An out-of-memory run is still an experiment record, but do not report simulated
cache bytes as measured GPU savings.

## Boundary after this notebook

The notebook discovers correlational receiver heads. The next research milestone
is causal attention suppression: mask access to one candidate sentence at a time,
save downstream logits, and rank the interventions with
`score_causal_interventions`. That step should only begin after the trace format
and memory envelope have been validated on the assigned GPU.

