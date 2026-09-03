# T4 controlled causal-suppression pilot: 2026-09-03

## Outcome

The horizon- and position-controlled rerun does **not** support the current
AnchorKV sentence-selection hypothesis. Suppressing the most recent eligible
sentence changed downstream logits much more than suppressing the sentence
selected by the leading receiver head.

This reverses the apparent positive result of the first pilot. The reversal is
scientifically useful: it shows that the earlier effect was dominated by answer
proximity and that raw vertical-attention concentration is not yet a reliable
causal-importance score.

## Provenance

- Archive: `anchorkv-colab-traces (4).zip`
- Archive SHA-256:
  `9176083CE3C61FEAF38E1A7C1E34671C6E7873172384EAEF118A16708C050ED2`
- GPU: Tesla T4, 14.56 GiB
- Peak causal-stage allocation: 1.524 GiB
- Model: `Qwen/Qwen3-0.6B`
- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Receiver-head candidate: layer 26, query head 10
- Minimum future horizon: 32 tokens
- Mask sanity maximum absolute logit difference: `0.0`
- All three generated-answer checks passed

The complete submitted archive is preserved under
[`data/causal-controlled/`](data/causal-controlled/), including the three NPZ
traces, JSON outputs, and plots.

## Aggregate results

| Suppressed sentence | Mean downstream KL | Mean NLL change |
|---|---:|---:|
| Head-selected anchor | 0.0066 | +0.0142 |
| Recency control | **0.1006** | **+0.0985** |
| Token-length-matched control | 0.0005 | +0.0009 |
| Position-matched control | 0.0022 | +0.0062 |

The recency effect is about 15.2 times the head-selected effect. The selected
sentence still exceeds the length- and position-matched controls on this tiny
sample, but it loses decisively to the strongest baseline.

![Controlled causal sentence suppression](data/causal-controlled/causal-kl.png)

## Per-sample interpretation

| Sample | Selected KL | Recency KL | Length KL | Position KL |
|---|---:|---:|---:|---:|
| `math-000` | 0.0147 | 0.0399 | 0.0011 | 0.0049 |
| `math-001` | 0.0041 | 0.2414 | 0.0002 | 0.0012 |
| `math-002` | 0.0010 | 0.0206 | 0.0001 | 0.0006 |

Recency wins on every sample. The selected text was span zero in every trace:

- `math-000`: “Okay, let's see.”
- `math-001`: “Okay, so there's a train that goes 180 miles in 3 hours.”
- `math-002`: “Okay, let's see.”

For the selected head, span zero's raw vertical-attention score was the maximum
in all three traces and was roughly 1.6 to 10.5 times the next-highest score.
This is a boundary-attention warning, not evidence that those generic phrases
are reasoning-critical.

## Interpretation and next experiment

Strong attention to early tokens can be a non-semantic *attention sink*, as
documented by [StreamingLLM](https://arxiv.org/abs/2309.17453) and subsequent
[attention-sink analysis](https://arxiv.org/abs/2410.10781). That literature
specifically cautions that high attention weights can behave like key biases
without a corresponding informative value contribution.

The next experiment should stop treating one attention argmax as ground truth.
For every eligible sentence, it should measure both attention salience and the
causal KL from suppression, then report:

1. Spearman correlation between attention score and causal effect.
2. Precision and regret for top-k attention-selected sentences against a
   causal oracle.
3. Separate metrics with boundary spans included and excluded.
4. Raw attention, cross-head-normalized attention, recency, and random
   baselines over the exact same downstream window.
5. Bootstrap confidence intervals over a larger prompt set before implementing
   a live KV-cache policy.

This turns the failed binary pilot into a direct test of whether attention is a
useful ranking signal for dynamic quantization.
