# Complete T4 notebook

**Next validation step:** use the
[FP16-only development notebook](../notebooks/AnchorKV_T4_FP16_Development.ipynb)
and [frozen evaluation protocol](arithmetic-evaluation-v1.md). It runs 12 new
development prompts before any held-out compression sweep. Baseline readiness
is not yet measured; existing pilot scores remain unchanged.

The completed 64-token arithmetic follow-up has a
[CPU-only rescoring report](experiments/data/arithmetic-followup-rescore/report.md).
It preserves strict scores and adds a post-hoc, conservative final-integer
metric with per-response extraction reasons in `rescore.json`. No GPU rerun
was used. Reproduce into a new directory with:

```powershell
$env:PYTHONPATH = 'src'
python -m anchorkv.rescore_arithmetic path/to/results.zip path/to/new-output
```

For the arithmetic follow-up to the completed quick run, upload
[`AnchorKV_T4_Arithmetic_Followup.ipynb`](../notebooks/AnchorKV_T4_Arithmetic_Followup.ipynb)
to a fresh T4 runtime and Run all. It reruns the same two arithmetic prompts and
nine policy variants with a 64-token cap, while skipping retrieval evaluation,
the historical pilot and throughput. Numerical gates still run. It retains the
10-minute soft quality budget, checkpoints, strict exact-answer scoring and zip
export. Use a new folder, not the incompatible 16-token run's checkpoint.
If paused, rerun the quality cell followed by report/download cells. Regenerate
this companion with `python notebooks/build_all_in_one.py --arithmetic-only`.

Upload [`AnchorKV_T4_All_In_One.ipynb`](../notebooks/AnchorKV_T4_All_In_One.ipynb)
to a **fresh** Colab T4 runtime. The file embeds its Python and Triton sources;
it does not clone the repository or import a separately installed AnchorKV.
Use Runtime > Run all. **The default is now `PROFILE = 'quick'`.** It runs six
prompts (three tasks, two content seeds, one length and position), one matched
budget and two random draws: 54 policy variants / 114 quality generation-replay
calls, rather than the old 576 variants / 1,188 calls. Answers are capped at 16
tokens; missing EOS is still reported as incomplete, not counted as a success.
Throughput uses 128 measured tokens and two repetitions (18 workloads including
excluded warm-ups). The historical six-question pilot is disabled by default.

Quality and throughput each have a **10-minute soft limit**. Limits are checked
between work units; they cannot interrupt an in-flight GPU operation, and do
not cover installation, downloads, model loading, numerical gates or export.
This is not a promise of a 20-minute total runtime. Progress prints before each
replay, greedy answer, warm-up and timing run, plus saved counts and elapsed time.
At the limit, the phase checkpoints and the remaining cells produce partial
reports and a downloadable zip. The full matrix remains opt-in with
`PROFILE = 'full'`; increase the phase budgets intentionally if desired.

## Resume without repeating completed measurements

- In the same live runtime, rerun the quality or throughput cell. Completed
  variant/run keys are skipped, and each invocation gets a fresh phase budget.
- To rerun setup, set `RESUME_DIRECTORY` to the printed output directory first.
  Configuration, prompts, source hash, environment and backend must match.
- To survive runtime loss, mount Google Drive yourself and use a persistent
  output folder there. `/content` alone is temporary. For a first persistent
  run, set `OUTPUT` to a new Drive folder and keep `RESUME_DIRECTORY = None`;
  on later runs point `RESUME_DIRECTORY` to that folder.
- Checkpoints use atomic replacement, and incomplete matched cases are excluded
  from paired comparisons. Partial summary tables are descriptive only.
- Correctness gates and necessary setup/warm-ups run again; measured rows do not.
- Older output bundles have no resume manifest and cannot be silently imported.
  Keep those results separately and start the new quick notebook in a fresh run.

The notebook deliberately pins Transformers 4.57.6 to constrain its attention
interface. Start a fresh runtime rather than reusing the earlier notebook's
Transformers 5.16.1 imports. The model remains Qwen3-0.6B at the exact revision
used in the submitted T4 smoke run. Colab's CUDA PyTorch/Triton installation is
recorded rather than replaced. If Triton is missing or fails a correctness gate,
the notebook records that failure and runs quality measurements using dense
reconstruction. Those measurements are explicitly labeled `dense`.

## Available experiments (historical pilot and full matrix are opt-in)

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
6. In the optional historical pilot, benchmark in shuffled order after warming each policy. Retain
   setup, decode, selection, shared prefill, and accounted total times; distinguish
   actual allocated CUDA memory from payload-plus-scale-and-table bytes.
7. Force a separate 96-token continuation to verify that newly generated pages
   age into compressed storage. Do not score this diagnostic as answer accuracy.
8. Export per-case data, bootstrap comparisons, raw generations, selected pages,
   numerical gates, environment/source hashes, plots, and a Markdown report.
9. In the full profile, run 36 prompts, comprising three synthetic tasks
   (retrieval, two-fact retrieval, and three-fact arithmetic), two content seeds,
   target lengths 768/1536, and early/middle/late evidence. Complete records and
   queries are fitted with the pinned tokenizer before loading model weights.
10. Compare two extra-FP16-page fractions (10% and 25% of eligible full pages)
    and three random seeds at each budget. Physical initial bytes must match.
    Four baselines plus twelve mixed variants give 16 variants per prompt;
    each gets one greedy answer and one gold-history diagnostic. These 576
    variant measurements are not 576 independent questions.
11. In the full profile, run decode with 32 warm-up inputs plus 128/256/512 measured inputs,
    six policies, and three timing repetitions after an excluded full-workload
    warm-up for every policy/length. All policies receive identical GPU-resident
    inputs; no sampling, EOS stopping, or per-token CPU-logit copies. This is a
    synthetic throughput workload using one longest prefix, not task accuracy.
    The phase is skipped if the packed gate fails. New workload results still
    require a Colab run; the original pilot does not validate their performance.

## Expanded outputs and interpretation

`expanded-quality.json` and `expanded-selection.json` checkpoint each variant
and case. `expanded-per-case.csv` averages random draws before comparisons;
`expanded-breakdown.csv` separates tasks, lengths and positions.
`expanded-failures.json` retains wrong answers and missing EOS completions.
`expanded-paired-comparisons.json` bootstraps task/content-seed family means,
not correlated position/length variants or random draws. With just six default
synthetic families, these intervals remain exploratory. Arithmetic is evaluated
with thinking disabled; it is not evidence about internal reasoning anchors.

`decode-workload.json` exports exact inputs, selected pages, and separate
prefill/snapshot and scoring costs. `decode-warmups.json` contains excluded full
workload warm-ups. `controlled-decode.json` checkpoints each measured run and
reports setup, in-cache warm-up, steady-state wall time, CUDA timeline elapsed
time, cache growth/demotions, and scoped GPU allocations. CUDA-event duration
includes launch gaps and is not a pure kernel-time sum. The primary tokens/s
uses synchronized wall time. Allocation baselines exclude already-resident
weights and prepared inputs; absolute peaks are also retained. Prefill is outside
the setup/decode peak window. `accounted_workload_seconds` adds shared prefill,
automatic scoring where used, per-plan assignment, setup, warm-up and measured
decode, not just the fast portion.

The final zip contains `expanded-report.md` (new quality and throughput), plus
`report.md` when the optional original pilot runs. Checkpoints survive cell
interruptions while the runtime remains alive, but `/content` is not persistent
storage. You can run the final download cell early to retrieve partial evidence;
resume requires retaining the folder and using the same manifest-compatible configuration.

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
- The semantic oracle receives the evidence location. It is a labeled control,
  not a causal upper bound or learned selection performance.
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
- The [submitted packed T4 pilot](experiments/2026-09-05-packed-t4-pilot.md)
  passed its numerical gates. This validates that configuration, not every
  future workload. CPU tests do not validate CUDA code.

## Remaining work after a successful run

A complete passing run supplies a reproducible research prototype and measured
evidence for its chosen configuration. It does not finish production deployment.
Further work is: replicate on more seeds, reasoning workloads and models; tune
or reject the selector based on the results; optimize the kernel and allocator
against established quantized-cache baselines; add batching and a serving-engine
integration such as vLLM; and benchmark long-context throughput under load.

## Maintaining the standalone file

Edit `src/anchorkv/packed_decode.py`, `triton_decode.py`,
`colab_experiment.py`, `benchmark_suite.py`, `notebooks/all_in_one_cells.py`, and
`notebooks/expanded_cells.py`, then regenerate with:

```bash
python notebooks/build_all_in_one.py
```

The notebook test compares embedded sources with their repository counterparts
and parses every code cell, preventing drift between the delivered notebook and
the tested implementation.

References for the integration interfaces:

- [Pinned Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)
- [Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
