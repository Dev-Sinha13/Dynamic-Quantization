"""Experimental mixed-page single-query GQA attention, FP16/INT8/packed INT4.

Each split streams pages, unpacks only the current tile in registers, and
maintains online softmax statistics. A second kernel merges split statistics.
No sequence-length dense K/V tensor is created. GPU gates in the notebook
must pass before this backend contributes benchmark results.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _page_values(address, scale_address, mode, head, rows, cols,
                 BLOCK: tl.constexpr, DIM: tl.constexpr, GROUP: tl.constexpr):
    offsets = (head * BLOCK + rows[:, None]) * DIM + cols[None, :]
    if mode == 16:
        values = tl.load(address.to(tl.pointer_type(tl.float16)) + offsets)
    else:
        if mode == 8:
            integers = tl.load(address.to(tl.pointer_type(tl.int8)) + offsets).to(tl.float32)
        else:
            packed_offsets = (head * BLOCK + rows[:, None]) * (DIM // 2) + cols[None, :] // 2
            packed = tl.load(address.to(tl.pointer_type(tl.uint8)) + packed_offsets).to(tl.int32)
            nibble = (packed >> ((cols[None, :] % 2) * 4)) & 15
            integers = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
        scale_offsets = (head * BLOCK + rows[:, None]) * (DIM // GROUP) + cols[None, :] // GROUP
        scales = tl.load(scale_address.to(tl.pointer_type(tl.float16)) + scale_offsets)
        # Match the FP16 execution values of the dense reference exactly.
        values = (integers * scales.to(tl.float32)).to(tl.float16)
    return values.to(tl.float32)


@triton.jit
def _decode_parts(Q, TABLE, PART, n_tokens, scale,
                  DIM: tl.constexpr, BLOCK: tl.constexpr,
                  GROUP: tl.constexpr, GQA: tl.constexpr, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // GQA
    cols = tl.arange(0, DIM)
    rows = tl.arange(0, BLOCK)
    query = tl.load(Q + head * DIM + cols).to(tl.float32)
    maximum = tl.full((), float('-inf'), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((DIM,), 0.0, tl.float32)
    for page in range(split, tl.cdiv(n_tokens, BLOCK), SPLITS):
        kptr = tl.load(TABLE + page * 5)
        vptr = tl.load(TABLE + page * 5 + 1)
        kscale = tl.load(TABLE + page * 5 + 2)
        vscale = tl.load(TABLE + page * 5 + 3)
        mode = tl.load(TABLE + page * 5 + 4)
        keys = _page_values(kptr, kscale, mode, kv_head, rows, cols, BLOCK, DIM, GROUP)
        values = _page_values(vptr, vscale, mode, kv_head, rows, cols, BLOCK, DIM, GROUP)
        scores = tl.sum(keys * query[None, :], 1) * scale
        scores = tl.where(page * BLOCK + rows < n_tokens, scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 0))
        rescale = tl.exp(maximum - new_maximum)
        probabilities = tl.exp(scores - new_maximum)
        numerator = numerator * rescale + tl.sum(probabilities[:, None] * values, 0)
        denominator = denominator * rescale + tl.sum(probabilities, 0)
        maximum = new_maximum
    base = (head * SPLITS + split) * (DIM + 2)
    tl.store(PART + base + cols, numerator)
    tl.store(PART + base + DIM, maximum)
    tl.store(PART + base + DIM + 1, denominator)


@triton.jit
def _merge_parts(PART, OUT, DIM: tl.constexpr, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    cols = tl.arange(0, DIM)
    base = (head * SPLITS + splits) * (DIM + 2)
    maxima = tl.load(PART + base + DIM)
    denominators = tl.load(PART + base + DIM + 1)
    weights = tl.exp(maxima - tl.max(maxima, 0))
    numerators = tl.load(PART + base[:, None] + cols[None, :])
    denominator = tl.sum(weights * denominators, 0)
    output = tl.sum(numerators * weights[:, None], 0) / denominator
    tl.store(OUT + head * DIM + cols, output)


def packed_attention(query, layer, scale, splits=4):
    if query.shape[0] != 1 or query.shape[2] != 1 or query.shape[-1] != layer.dim:
        raise ValueError('packed kernel supports batch=1, query length=1 only')
    if query.dtype != torch.float16 or not query.is_cuda or layer.length == 0:
        raise ValueError('packed kernel requires a nonempty CUDA FP16 cache')
    if layer.dim not in (64, 128) or layer.block != 16 or layer.group != 64:
        raise ValueError('supported geometry: block=16, group=64, head dim=64/128')
    heads = query.shape[1]
    if heads % layer.heads:
        raise ValueError('query heads must be divisible by KV heads')
    query = query.contiguous()
    partial = torch.empty((heads, splits, layer.dim + 2), device=query.device, dtype=torch.float32)
    output = torch.empty_like(query)
    _decode_parts[(heads, splits)](
        query, layer.table, partial, layer.length, scale,
        DIM=layer.dim, BLOCK=layer.block, GROUP=layer.group,
        GQA=heads // layer.heads, SPLITS=splits, num_warps=4,
    )
    _merge_parts[(heads,)](partial, output, DIM=layer.dim, SPLITS=splits, num_warps=4)
    return output
