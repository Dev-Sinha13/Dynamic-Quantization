# Packed-attention T4 pilot: measured results and limits

Original user submission: `20260905-225302.zip`, preserved unchanged as
[packed-t4-pilot.zip](data/packed-t4-pilot.zip). Archive SHA-256:
`d7bd456fda9b97d6e90e38d0ec2e093bed3346e1679845b9a4215c4e6c6705c9`.
Embedded source SHA-256:
`9528b35c89eff4b2c4c13f39c29d2a3224b5f9d944e88c4e0feaeb4f6db78b30`.

Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`;
Tesla T4 (7.5), Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8,
Transformers 4.57.6, Triton 3.6.0. This is distinct from the earlier
[dense-reconstruction smoke run](2026-09-05-requantization-smoke.md).

## Results

All 42 packed-attention configurations passed (largest absolute attention-output
error 0.0009765625). Four full-model packed/reconstructed comparisons passed;
largest mean KL was approximately 1.13e-6. The forced 96-token lifecycle check
finished with 460 cache tokens and 700 layer-page demotions, including 168 after
prefix initialization. This is a lifecycle check, not a reasoning score.

All eight policies answered all six retrieval questions correctly with EOS,
generating only five tokens each. There were 144 free-generation measurements
(six questions, eight policies, three timing repeats) and 48 teacher-forced
measurements. Timing repeats are not independent accuracy samples.

| Policy | Resident KV MiB | Mean gold-history KL | Accounted seconds |
|---|---:|---:|---:|
| Stock FP16 | 68.615 | 0 | 1.182 |
| Paged FP16 | 69.464 | 0.000000557 | 1.324 |
| Residual INT8 | 39.231 | 0.000003404 | 1.724 |
| Residual INT4 | 23.627 | 0.019216 | 1.861 |
| Recent | 29.410 | 0.015002 | 1.809 |
| Random | 29.410 | 0.003344 | 1.854 |
| Automatic | 29.410 | 0.004243 | 2.398 |
| Evidence oracle | 29.410 | 0.015394 | 1.845 |

Timing values are means of per-case medians. Accounted time includes shared
prefill/snapshot, selector preparation where used, cache setup, and decode.
Quality-path decode also copies logits to CPU; these short-answer timings are
not clean steady-state throughput measurements. Mixed selectors have identical
initial resident bytes within each case.

## What this supports

- Physical resident-cache savings: 65.6% for INT4 and 57.1% for automatic vs stock
  FP16. These percentages are not whole-model GPU-memory savings.
- Experimental packed attention works on the tested T4 configuration.
- Automatic retention lowers mean KL vs recent by 71.7% on this small set.

## What it does not support

- A speedup: automatic accounted time is 2.03x stock FP16.
- Superiority to random: automatic KL is 26.9% higher; paired automatic-minus-
  random mean KL is 0.000899, with case-bootstrap 95% interval
  [-0.001339, 0.004334]. This does not establish an advantage.
- Broad accuracy preservation: every policy passes six easy questions. One
  late-evidence case dominates the KL differences. Random selection has only
  one draw per question, and the questions are not a standard benchmark.
- A causal upper bound from the evidence oracle: protecting semantic evidence
  pages is not guaranteed to protect the pages most sensitive to quantization.
- Production readiness, thought-anchor validity, or long-generation throughput.

Next: broaden tasks and lengths, repeat random selectors at multiple matched
budgets, and isolate controlled 128/256/512-token decode timing from quality
diagnostics before choosing an optimization target.
