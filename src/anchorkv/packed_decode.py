"""Experimental append-only, batch-one KV pages for the standalone Colab suite.

Payloads remain on their input device. Demoted pages release their FP16 owners;
the pointer table addresses independent allocations rather than a padded FP16
arena. This implementation favors inspectability over allocation throughput.
"""

from dataclasses import dataclass
import math
import random

import torch
import torch.nn.functional as F


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


@dataclass
class QuantizedPage:
    bits: int
    data: torch.Tensor
    scales: torch.Tensor | None
    shape: tuple
    group: int

    @property
    def nbytes(self):
        return tensor_bytes(self.data) + (
            tensor_bytes(self.scales) if self.scales is not None else 0
        )

    def dense(self):
        if self.bits == 16:
            return self.data
        if self.bits == 8:
            integers = self.data.float()
        else:
            lo = self.data & 15
            hi = self.data >> 4
            values = torch.stack((lo, hi), -1).reshape(self.shape).short()
            integers = torch.where(values >= 8, values - 16, values).float()
        groups = integers.reshape(*self.shape[:-1], -1, self.group)
        return (groups * self.scales.float().unsqueeze(-1)).reshape(self.shape).half()


def pack_page(tensor, bits=4, group=64):
    if bits not in (4, 8, 16):
        raise ValueError('bits must be 4, 8, or 16')
    if tensor.ndim != 3 or tensor.shape[-1] % group or group % 2:
        raise ValueError('expected [KV heads, page tokens, head dim], with even groups')
    source = tensor.detach().half().contiguous()
    if bits == 16:
        # Own storage: do not retain a view of the original full prompt cache.
        return QuantizedPage(bits, source.clone(), None, tuple(source.shape), group)
    groups = source.float().reshape(*source.shape[:-1], -1, group)
    maxima = groups.abs().amax(-1)
    qmax = 7 if bits == 4 else 127
    # Clamp before FP16 conversion: tiny nonzero groups must not get scale zero.
    scales = (maxima / qmax).clamp_min(torch.finfo(torch.float16).tiny).half()
    integers = torch.round(groups / scales.float().unsqueeze(-1))
    integers = integers.clamp(-qmax, qmax).to(torch.int8).reshape(source.shape)
    if bits == 4:
        nibble = (integers.short() & 15).byte()
        integers = (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).contiguous()
    return QuantizedPage(bits, integers, scales, tuple(source.shape), group)


def choose_pages(name, candidates, count, *, scores=None, evidence=(), seed=0):
    """Choose exactly the same number of full pages for every mixed policy."""
    candidates = sorted(set(candidates))
    if not 0 <= count <= len(candidates):
        raise ValueError('invalid high-precision page allowance')
    if name == 'random':
        order = random.Random(seed).sample(candidates, len(candidates))
    elif name == 'recent':
        order = sorted(candidates, reverse=True)
    elif name == 'automatic':
        if scores is None or any(page not in scores for page in candidates):
            raise ValueError('automatic selection requires every candidate score')
        order = sorted(candidates, key=lambda page: (-scores[page], page))
    elif name == 'oracle':
        evidence = set(evidence)
        order = sorted(candidates, key=lambda page: (page not in evidence, -page))
    else:
        raise ValueError(f'unknown selector: {name}')
    return frozenset(order[:count])


def rank_correlation(x, y):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = torch.empty(len(values), dtype=torch.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            out[order[start:end]] = (start + end - 1) / 2
            start = end
        return out - out.mean()
    a, b = ranks(x), ranks(y)
    denominator = a.norm() * b.norm()
    return float(a.dot(b) / denominator) if denominator > 0 else None


class PagedLayer:
    def __init__(self, heads, dim, max_tokens, *, block=16, group=64,
                 archive_bits=4, protected=(), recent_pages=2, device='cuda'):
        if dim % group or archive_bits not in (4, 8, 16):
            raise ValueError('invalid quantization geometry')
        self.heads, self.dim, self.block, self.group = heads, dim, block, group
        self.archive_bits = archive_bits
        self.protected = frozenset(protected)
        self.recent_pages = recent_pages
        self.capacity = math.ceil(max_tokens / block)
        self.device = torch.device(device)
        self.pages = []
        self.length = 0
        self.demotions = 0
        self.table = torch.zeros((self.capacity, 5), dtype=torch.int64, device=device)
        self.dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
        self.archived_through = -1

    def _publish(self, page):
        key, value = self.pages[page]
        ks = self.dummy_scale if key.scales is None else key.scales
        vs = self.dummy_scale if value.scales is None else value.scales
        self.table[page] = torch.tensor(
            [key.data.data_ptr(), value.data.data_ptr(), ks.data_ptr(), vs.data_ptr(), key.bits],
            dtype=torch.int64, device=self.device,
        )

    def append(self, key, value):
        if key.shape != value.shape or key.ndim != 4 or key.shape[:2] != (1, self.heads):
            raise ValueError('only batch-one matching K/V tensors are supported')
        if key.shape[-1] != self.dim or self.length + key.shape[-2] > self.capacity * self.block:
            raise ValueError('cache geometry or capacity exceeded')
        offset = 0
        while offset < key.shape[-2]:
            page, position = divmod(self.length, self.block)
            if position == 0:
                empty = torch.zeros((self.heads, self.block, self.dim),
                                    dtype=torch.float16, device=self.device)
                self.pages.append((pack_page(empty, 16, self.group), pack_page(empty, 16, self.group)))
                self._publish(page)
            take = min(self.block - position, key.shape[-2] - offset)
            kpage, vpage = self.pages[page]
            kpage.data[:, position:position + take].copy_(key[0, :, offset:offset + take])
            vpage.data[:, position:position + take].copy_(value[0, :, offset:offset + take])
            self.length += take
            offset += take
            self._archive_old_pages()

    def _archive_old_pages(self):
        # Complete pages only; the partial append page always stays FP16.
        last_eligible = self.length // self.block - self.recent_pages - 1
        for page in range(self.archived_through + 1, last_eligible + 1):
            if page not in self.protected and self.archive_bits != 16:
                self.demote(page, self.archive_bits)
        self.archived_through = max(self.archived_through, last_eligible)

    def demote(self, page, bits):
        key, value = self.pages[page]
        if bits > key.bits:
            raise ValueError('lossy promotion cannot restore FP16 values')
        if page in self.protected and bits != 16:
            raise ValueError('cannot demote a protected page')
        if bits != key.bits:
            self.pages[page] = (pack_page(key.dense(), bits, self.group),
                                pack_page(value.dense(), bits, self.group))
            self._publish(page)
            self.demotions += 1

    def protect(self, page):
        if page < len(self.pages) and self.pages[page][0].bits != 16:
            raise ValueError('anchor must be declared before demotion')
        self.protected = self.protected | {page}

    @property
    def payload_bytes(self):
        return sum(k.nbytes + v.nbytes for k, v in self.pages)

    @property
    def resident_bytes(self):
        return self.payload_bytes + tensor_bytes(self.table) + tensor_bytes(self.dummy_scale)

    def dense(self):
        if not self.pages:
            raise ValueError('cache is empty')
        k = torch.cat([pair[0].dense() for pair in self.pages], dim=1)[:, :self.length]
        v = torch.cat([pair[1].dense() for pair in self.pages], dim=1)[:, :self.length]
        return k.unsqueeze(0), v.unsqueeze(0)


def dense_attention(query, layer, scale):
    key, value = layer.dense()
    repeats = query.shape[1] // key.shape[1]
    return F.scaled_dot_product_attention(
        query, key.repeat_interleave(repeats, 1), value.repeat_interleave(repeats, 1),
        is_causal=False, scale=scale,
    )


def kl_divergence(reference, candidate):
    p = reference.float().log_softmax(-1)
    q = candidate.float().log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1).clamp_min(0)
