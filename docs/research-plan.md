# AnchorKV research plan

## Scope

The first empirical version targets one open-weight reasoning model and visible
reasoning traces. It evaluates algorithmic quality and estimated cache capacity
before attempting fused kernels or a serving-engine integration.

## Hypotheses

### H1: Receiver-head concentration

A small, repeatable subset of attention heads has higher sentence-level vertical
attention concentration than the median head.

Evidence:

- Head-wise concentration distribution
- Within-trace percentile ranking to avoid comparing raw kurtosis across
  different sentence counts
- Exclusion of the terminal answer span and trailing EOS query from receiver
  scoring
- Split-half or bootstrap rank stability
- Cross-seed stability

### H2: Causal anchor validity

Suppressing access to receiver-head-selected sentences causes a larger change in
future logits and answer accuracy than suppressing random or equally recent
sentences.

Evidence:

- Next-token KL divergence
- Final-answer accuracy delta
- Counterfactual sentence replacement

The first intervention uses an eager-attention 4D additive mask. It blocks all
future queries from attending to one selected sentence and compares the
teacher-forced downstream logit KL against recency and token-length-matched
controls over one common evaluation window. This validates sentence causality;
head-specific masking is a subsequent refinement.

### H3: Cache allocation value

At an equal KV-byte budget, receiver-head anchor retention preserves more full
cache accuracy than recency, random, and all-head attention policies.

Evidence:

- Accuracy versus retained-byte curves
- Area under the quality-memory curve
- Regret relative to a causal oracle

### H4: Transfer limits

Receiver heads and anchor thresholds discovered on one task do not transfer
uniformly across reasoning domains.

Evidence:

- Math-to-logic and math-to-code head-rank correlation
- In-domain versus transferred-policy accuracy

## Initial model and datasets

Start with one model that fits the available hardware and exposes visible
reasoning tokens. Candidate development tasks:

- GSM8K for fast iteration
- MATH-500 for the main mathematical evaluation
- A small AIME subset for difficult long-form reasoning
- One logic or code task for transfer analysis

Model and dataset revisions, prompts, seeds, and hardware must be recorded with
every result.

## Baselines

1. Full FP16 KV cache
2. Recent window with pinned initial tokens
3. Random sentence retention
4. Accumulated attention across every head
5. Receiver-head attention without causal validation
6. Causally validated receiver-head attention

Published implementations can be added after this minimum comparison is
reproducible.

## Experimental controls

- Equal byte budget, not equal number of tokens
- Identical model, prompt, decoding parameters, and random seed
- GQA query-to-KV head mapping reported explicitly
- Separate instrumentation memory from simulated cache memory
- Report failure rates and out-of-memory runs
- Do not report simulated byte savings as measured GPU savings

## Success criteria

The initial result is portfolio-ready when it provides at least one of:

- At least 2x measured KV-cache compression with no more than one absolute point
  of answer-accuracy loss, or
- A statistically supported improvement over the strongest baseline at equal
  memory, or
- A clear negative result identifying when receiver attention fails to predict
  causal importance.

The systems claim additionally requires a compatible cache implementation and
measured peak GPU allocation, decode throughput, and policy overhead.

## Milestones

### M0: Model-agnostic core — complete

- Segmentation
- Receiver-head ranking
- GQA mapping
- Delayed anchor tracking
- Byte-budget planning
- Synthetic end-to-end test

### M1: Attention trace extraction — implemented, GPU validation pending

- One supported Hugging Face model family
- Capture and immediately aggregate selected attention statistics
- Persist compact trace artifacts with model metadata

### M2: Causal validation

- Attention-suppression intervention
- Random and recency controls
- KL-divergence and answer-accuracy reports

### M3: Cache simulation benchmark

- Baseline policy interface
- Multiple memory budgets
- Quality-memory plots and reproducible result files

### M4: Physical cache integration

- FP16 and one quantized representation
- Measured GPU allocations
- Decode latency and throughput
- Correctness tests against the full-cache model

### Stretch: serving integration

- Selected-head statistics without materializing full attention
- Concurrency-safe cache transitions
- vLLM or another block-based serving backend
