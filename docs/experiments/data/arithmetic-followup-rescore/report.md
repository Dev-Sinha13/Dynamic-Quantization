# Arithmetic follow-up: local rescoring

Post-hoc analysis of saved responses; no GPU rerun and no original scores changed.
Original archive SHA-256: `71752d13083297626d7f4da87cfe36eced94599c279d847183d29dd35fa60b65`.

| Policy | Strict correct | Final-answer correct | Parsed | Case-averaged final accuracy |
|---|---:|---:|---:|---:|
| automatic | 0/2 | 1/2 | 1/2 | 50% |
| int4 | 0/2 | 0/2 | 0/2 | 0% |
| int8 | 0/2 | 1/2 | 2/2 | 50% |
| native_fp16 | 0/2 | 1/2 | 2/2 | 50% |
| oracle | 0/2 | 0/2 | 0/2 | 0% |
| paged_fp16 | 0/2 | 1/2 | 2/2 | 50% |
| random | 0/4 | 1/4 | 1/4 | 25% |
| recent | 0/2 | 0/2 | 1/2 | 0% |

## Interpretation limits

- Strict exact-format scoring remains unchanged. This supplemental metric ignores extra explanation.
- Extraction uses only the response, not the gold answer. It accepts explicit terminal integer claims or equalities; no contains-gold or last-number fallback.
- Unparsed answers count as no demonstrated success in the denominator, but are not labeled proven arithmetic errors. Inspect them in rescore.json.
- Final-answer correctness does not validate every reasoning step.
- The parser was introduced after inspecting these outputs: this is exploratory post-hoc analysis, not a pre-registered result.
- Only two distinct questions were tested. Random has two draws per question; its rate is averaged within case before across cases.
- Native FP16 and automatic each have one correct final answer. This does not establish general equivalence or superiority.
- Next validate prompts/scoring on an independent development set, freeze the scorer, then compare policies on held-out cases.
