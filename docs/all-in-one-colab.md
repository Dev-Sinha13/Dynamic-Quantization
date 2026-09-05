# Complete T4 notebook

Upload [`AnchorKV_T4_All_In_One.ipynb`](../notebooks/AnchorKV_T4_All_In_One.ipynb)
to a **fresh** Colab T4 runtime. The file embeds its Python and Triton sources;
it does not clone the repository or import a separately installed AnchorKV.
Use Runtime > Run all. Expect tens of minutes, depending on allocator overhead
and whether the experimental packed path passes its gates.

The notebook deliberately pins Transformers 4.57.6 to constrain its attention
interface. Start a fresh runtime rather than reusing the earlier notebook's
Transformers 5.16.1 imports. The model remains Qwen3-0.6B at the exact revision
used in the submitted T4 smoke run. Colab's CUDA PyTorch/Triton installation is
recorded rather than replaced. If Triton is missing or fails a correctness gate,
the notebook records that failure and runs quality measurements using dense
reconstruction. Those measurements are explicitly labeled `dense`.

## Included experiments

1. Build six chat-template retrieval prompts spanning two context lengths and
   three evidence positions. Validate token bounds before loading model weights.
2. Run packed-attention numerical gates across page boundaries, FP16/INT8/INT4,
   GQA, empty split partitions, and two head dimensions; compare full-model
   packed logits against reconstructed attention and stock FP16.
3. Allocate equal initial cache bytes to recent, random, automatic, and oracle
   page selection. Sink and recent-window rules are shared. Residual INT4/INT8,
   paged FP16, and stock FP16 provide additional reference points.
4. Select automatic anchors using the last prompt query's attention and K/V
   quantization error. The score does not receive answer labels or evidence
   locations. Audit it against single-page quantization effects separately.
5. Independently recompute the first answer distribution under each policy,
   check exact completed answers, stop at EOS, and measure gold-history KL/NLL.
6. Repeatedly benchmark in shuffled order after warming each policy. Retain
   setup, decode, selection, shared prefill, and accounted total times; distinguish
   actual allocated CUDA memory from payload-plus-scale-and-table bytes.
7. Force a separate 96-token continuation to verify that newly generated pages
   age into compressed storage. Do not score this diagnostic as answer accuracy.
8. Export per-case data, bootstrap comparisons, raw generations, selected pages,
   numerical gates, environment/source hashes, plots, and a Markdown report.

## Packed attention implementation

The cache owns independent 16-token pages. An append writes to the current FP16
page; eligible complete pages are demoted as the recent window advances. A
protected page cannot be demoted, and an archived page cannot be promoted as if
its lost information were recovered. Each tensor's scales are included in its
byte count; the GPU pointer table and its spare capacity are also counted.

The experimental Triton kernel reads these page pointers directly. Four splits
per query head stream subsets of pages and compute softmax statistics while
unpacking a tile in registers. A second kernel combines the partial results.
There is no full-length dense KV tensor in the packed attention path. The query
head maps to its shared KV head explicitly.

This is a simple single-token decoder kernel, not a production kernel. It does
not use tensor cores, eliminate Python allocation overhead, handle arbitrary
masks, or support batches, sliding windows, beam search, or multi-token decode.
The adapter uses the stock Qwen attention projections, normalization, and RoPE
through a registered attention function. Its global registry entry is reset
after each run so it cannot retain an earlier GPU cache.

## Interpretation boundaries

- The automatic score is a proposed heuristic, not a validated receiver-head
  detector. Its evidence must come from equal-budget comparisons on new data.
- The semantic oracle receives the evidence location. Its result is an upper
  reference for the supplied evidence, not learned selection performance.
- Precision changes continuously, but the protected set is chosen once per
  prompt. Generated `<focus>`/`<local>` tags are not parsed or used by this suite.
- Equal budgets apply to initial mixed-policy caches. Different generated
  lengths can make final cache sizes differ; raw token counts are preserved.
- Cache-plus-decode time includes packing, transfer, allocation, and attention.
  Accounted total time adds measured automatic-selector cost and the shared
  prefill/snapshot cost. The latter is measured once per case, not once per repeat.
- CUDA peak allocation is scoped to cache setup and decode, not the entire
  model-loading/prefill process. Source snapshots are on CPU and their memory is
  outside that GPU number. CPU analysis remains an experimental overhead.
- Timing repetitions are collapsed per case before bootstrapping. Six questions
  are a smoke/pilot set; even a positive interval would need broader replication.
- Kernel numerical tests and integrated GPU measurements remain pending until
  the exported notebook is run on a T4. CPU tests do not validate CUDA code.

## Remaining work after a successful run

A complete passing run supplies a reproducible research prototype and measured
evidence for its chosen configuration. It does not finish production deployment.
Further work is: replicate on more seeds, reasoning workloads and models; tune
or reject the selector based on the results; optimize the kernel and allocator
against established quantized-cache baselines; add batching and a serving-engine
integration such as vLLM; and benchmark long-context throughput under load.

## Maintaining the standalone file

Edit `src/anchorkv/packed_decode.py`, `triton_decode.py`,
`colab_experiment.py`, and `notebooks/all_in_one_cells.py`, then regenerate with:

```bash
python notebooks/build_all_in_one.py
```

The notebook test compares embedded sources with their repository counterparts
and parses every code cell, preventing drift between the delivered notebook and
the tested implementation.

References for the integration interfaces:

- [Pinned Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)
- [Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
