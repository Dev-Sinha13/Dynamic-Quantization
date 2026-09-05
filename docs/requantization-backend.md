# Requantization backend

This page describes the original block-store reference API. A separate
experimental online implementation now lives in `packed_decode.py`, with a
Triton single-token attention kernel in `triton_decode.py`. The
[all-in-one notebook](all-in-one-colab.md) embeds both, numerical gates, and the
benchmark. Its GPU results are pending; the older T4 archive only tests dense
cache reconstruction.

## Status

AnchorKV has a functional PyTorch reference backend for physical KV-cache
storage. It performs real quantization and bit packing; it no longer represents
INT8 and INT4 solely as byte estimates.

Implemented:

- FP16 payload storage.
- Symmetric groupwise INT8 with FP16 scales.
- Symmetric groupwise INT4 packed two signed nibbles per byte.
- Fixed-size logical KV blocks and semantic segment metadata.
- Irreversible FP16 → INT8 → INT4 → eviction lifecycle checks.
- Application of the existing byte-budgeted `CachePlan` to physical blocks.
- Incremental declarative-attention and precision tag parsing.
- Global, focus, and local visibility sets.
- Dense materialization for correctness experiments.
- Layer-first conversion to and from Hugging Face legacy cache tuples.
- Actual payload-plus-scale byte reporting and reconstruction-error tests.

Not implemented:

- Replacement of a live Hugging Face `CacheLayerMixin` during generation.
- A vLLM paged-cache allocator or attention-metadata hook.
- A kernel that directly consumes mixed FP16/INT8/INT4 blocks.
- Asynchronous GPU requantization or CPU/NVMe offload.
- Measured end-to-end T4 latency or throughput gains.

## Quick demonstration

```bash
python -m pip install -e ".[research]"
anchorkv requantize-demo --archive-mode int4
```

The seeded demonstration creates a synthetic multi-region KV cache with the
standard `[batch, KV heads, tokens, head dimension]` layout. It protects the
response as an anchor, archives two context segments, enters focus mode for one
of them, materializes the visible cache, and reports physical bytes and error.

Current deterministic INT4 output:

| Metric | Value |
|---|---:|
| Full FP16 storage | 12,288 bytes |
| Mixed resident storage | 6,272 bytes |
| Compression ratio | 1.959× |
| Maximum absolute error | 0.2782 |
| Mean absolute error | 0.0915 |

The ratio includes FP16 sink, scaffold, and protected-response regions. It is
therefore lower than the isolated INT4 tensor ratio, as it should be.

## Storage representation

`quantize_tensor` flattens each block into fixed-size groups. For each group it
stores an FP16 scale based on the maximum absolute value.

- INT8 maps values to `[-127, 127]` and stores one signed byte per value.
- INT4 maps values to `[-7, 7]`, encodes each signed value as a four-bit two's
  complement nibble, and packs two nibbles into one `torch.uint8` value.
- Zero-valued groups use a scale of one and reconstruct exactly.
- Padding exists only inside the final quantization group and is removed during
  materialization.

`PackedTensor.stored_bytes` counts the actual PyTorch payload and scale tensor
storage. `DeclarativeKVCache.full_precision_bytes` uses the equivalent FP16 K+V
size as its comparison baseline.

## Declarative protocol

The incremental parser accepts tags even when they are split across generated
tokens:

```text
<global>
<focus segments="2,3">
<local>
<anchor segments="7">
<archive segments="2,3">
```

`magic_chunks="..."` is accepted as an alias for compatibility with
[Declarative Attention](https://arxiv.org/html/2609.02737v1). Sink, scaffold,
and generated-response regions remain visible in focus and local modes. Focus
adds only the declared context segments; global exposes every resident segment.

`<anchor>` protects a still-FP16 segment. `<archive>` converts an unprotected
segment to the configured INT4 or INT8 archive representation. The backend
refuses to label a quantized segment FP16 later: dequantization creates an FP16
execution tensor but cannot restore information lost during quantization.

## Hugging Face boundary

Hugging Face cache tensors use one
`[batch, heads, sequence, head_dimension]` pair per layer. `stack_legacy_cache`
converts those pairs to the backend's optional layer-first layout:

```python
from anchorkv import DeclarativeKVCache, SegmentRole, stack_legacy_cache

keys, values = stack_legacy_cache(outputs.past_key_values)
cache = DeclarativeKVCache(block_size=16, group_size=64)
cache.add_segment(
    0,
    keys,
    values,
    token_start=0,
    role=SegmentRole.CONTEXT,
)
legacy_cache = cache.materialize_visible().to_legacy_cache()
```

This conversion is a correctness bridge, not a fast path. Hugging Face already
offers whole-cache `QuantizedCache` implementations, but its documented cache
matrix does not provide per-semantic-segment mixed precision. The AnchorKV
backend exists to evaluate that heterogeneous policy rather than replace those
well-tested homogeneous implementations.

## Why vLLM integration comes next

Declarative Attention demonstrates the appropriate systems boundary: it rounds
semantic spans to 16–32-token physical blocks and edits the request block table
through attention-metadata hooks, allowing existing attention kernels to skip
unselected blocks. AnchorKV can reuse that control-plane idea, but heterogeneous
storage adds a data-plane requirement: the selected kernel must understand the
block's dtype and scales, or the runtime must stage dequantized blocks into a
temporary execution buffer.

The standalone
[`AnchorKV_T4_Requantization.ipynb`](../notebooks/AnchorKV_T4_Requantization.ipynb)
now completes the first correctness milestone: it captures Qwen3-0.6B's real
prompt cache, packs it under homogeneous and declarative mixed policies,
reconstructs `DynamicCache`, and measures controlled logit and answer deltas.
Its dense execution allocation is labeled separately from physical packed
storage.

The next implementation milestones are therefore:

1. Run the standalone notebook on a T4 and record the pinned result artifact.
2. Benchmark CUDA-native quantization, materialization, peak allocation, and
   decode latency on a T4.
3. Choose either separate dtype-specific block pools or one dequantization
   staging pool for the vLLM prototype.
4. Align semantic spans outward to vLLM physical block boundaries and resolve
   conflicts by retaining the highest required precision.
5. Add an attention-metadata hook for visibility and a compatible mixed-cache
   attention path before claiming throughput improvements.

References:

- [Declarative Attention](https://arxiv.org/html/2609.02737v1)
- [Hugging Face cache strategies](https://huggingface.co/docs/transformers/main/kv_cache)
- [vLLM attention backend API](https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backend/)
