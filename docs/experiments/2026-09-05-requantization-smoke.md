# T4 physical-cache requantization smoke run

The user-submitted [original archive](data/requantization-t4-smoke.zip) completed
the first real-model cache-intervention experiment on a Tesla T4. Archive SHA-256:
`e4b114b6c19ad13ee15b915f75962ec0d377dac89dc80a8ba39b2f38679e8fcc`.

Model: Qwen/Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`; Transformers 5.16.1;
PyTorch 2.11.0+cu128; Python 3.13.15. The manually increased prompt cap permitted
the original 664-token prompt. Generation was capped at 32 tokens; the shared
teacher-forced window had 24 steps. Quantization group size was 64.

| Policy | Packed MiB | FP16/packed ratio | Mean KL vs FP16 |
|---|---:|---:|---:|
| FP16 | 72.625 | 1.000 | 0 |
| INT8 | 37.447 | 1.939 | 0.00142046 |
| INT4 | 19.291 | 3.765 | 0.05473408 |
| Declarative INT4 | 25.315 | 2.869 | 0.00875759 |

The mixed policy protected 75 tokens and archived 589. Its packed storage was
65.14% below FP16. Mean KL was 84.00% below uniform INT4, at the cost of 31.23%
more storage. Every policy had identical 32-token greedy output and 100% top-1
agreement over the diagnostic window. Thus no answer-quality advantage over
uniform INT4 was established.

## Limitations exposed by this run

- All policies repeated `7319` and `Answer:` until the token cap; these were not
  clean completed answers. The prompt did not use Qwen's chat template.
- Every policy received its first generated token from unchanged FP16 prefill
  logits. Agreement on that token was guaranteed.
- Protection was chosen manually, and initial bytes differed across policies.
- `<focus>` appeared in the recorded program but the embedded precision parser
  did not enforce attention visibility.
- Packing happened once, on CPU; generated cache entries remained dense.
- All policies peaked at 1.2647037506 GiB of allocated CUDA memory. Attention
  rematerialized the dense cache, so the physical storage reduction did not
  translate into an inference GPU-memory improvement.
- Single decode measurements excluded packing/materialization and lacked
  balanced warm-ups. They do not support a speedup claim.

These observations motivated the
[all-in-one experiment](../all-in-one-colab.md): equal initial byte budgets,
independent first-answer prediction, chat-template/EOS validation, automatic
selection diagnostics, online aging, and an experimental packed-attention path.
This archive does **not** validate that new kernel or notebook.
