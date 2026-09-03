# T4 non-thinking rerun: 2026-09-03

## Outcome

This rerun confirmed that the revised notebook prevents truncation and records
runtime provenance. It is still **not an accepted receiver-head discovery
result** because two of three answers are wrong and the 2–3 sentence traces are
too short for a meaningful kurtosis statistic.

## Provenance

- Archive: `anchorkv-colab-traces (1).zip`
- Archive SHA-256:
  `6A511E274D90F8DBE3C616270FEE37A40DD3BB5DE1CBD4FC5F8C24F72CC0AA07`
- GPU: Tesla T4, 14.56 GiB
- Model: `Qwen/Qwen3-0.6B`
- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- PyTorch: `2.11.0+cu128`
- Transformers: `5.16.1`
- Thinking mode: disabled
- Maximum new tokens: 192

The ZIP passed its integrity check. All NPZ artifacts used the expected
pickle-free schema and `[28, 16, sentences]` float32 score tensors.

## Task results

| Sample | Tokens | Spans | Peak GPU | Result |
|---|---:|---:|---:|---|
| `math-000` | 89 | 2 | 1.151 GiB | Incorrect: `$21`; expected `$28` |
| `math-001` | 129 | 3 | 1.170 GiB | Correct: 300 miles |
| `math-002` | 101 | 2 | 1.157 GiB | Incorrect: 18/18; expected Maria 24, Lee 12 |

The measured peak allocation stayed below the notebook's 1.69 GiB tensor
lower-bound estimate because that estimate includes a conservative full
768-token attention matrix while actual sequences were only 89–129 tokens.

## Why the head manifest is invalid

For any two unequal observations, Pearson kurtosis is exactly 1. For three
nonconstant observations, it is exactly 1.5. Consequently, each head in this
run received the same theoretical per-trace kurtosis except for constant or
floating-point edge cases. The manifest's repeated mean-kurtosis value of
`1.1667` is the average of two 1.0 values and one 1.5 value.

The reported top heads are therefore driven by floating-point tie behavior,
not meaningful differences in attention concentration. Neither their ranking
scores nor the generated plot should be used as evidence.

## Corrections made before the next run

1. Re-enable Qwen thinking mode for explicit multi-step reasoning.
2. Increase the generation cap to 320 tokens while retaining EOS validation.
3. Prompt for at least six concise reasoning sentences.
4. Require at least six extracted spans before saving a usable trace.
5. Reject head discovery on any trace with fewer than four sentences.
6. Validate the known numerical answer terms before building the manifest.

These safeguards landed in commit `db79e06`. The next run is accepted only if
all three answers are correct, all three generations reach EOS, and every trace
contains at least six sentence spans.
