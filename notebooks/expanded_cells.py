# %% [markdown]
# ## Expanded quality: tasks, matched budgets, and random seeds
#
# These are synthetic retrieval, two-fact retrieval, and three-fact arithmetic
# questions, not a standardized reasoning benchmark or thought-anchor test.
# Thinking is disabled and answers are scored strictly (including comma format).
# The same facts appear at multiple lengths and positions within each family.
# Default: 36 prompts, six families, 16 policy variants per prompt. Each variant
# gets one greedy answer and one gold-history diagnostic, not timing repeats.
# Two budget fractions specify extra protected pages among eligible full pages;
# exact bytes and realized page counts are retained. Three random selector draws
# are averaged within each case before comparison. No claim is based on repeated
# timings being additional questions. Automatic scoring is computed once per
# prompt, never fitted using answers. Quality timing includes instrumentation and
# is not the throughput result. Raw results checkpoint after every variant.

# %%
expanded_rows = []
expanded_selections = []
if RUN_EXPANDED_QUALITY:
    for case in expanded_cases:
        print('Expanded quality:', case['case_id'], flush=True)
        source = prefill_case(model, case)
        candidates = list(range(1, (len(case['ids']) - 1) // settings.block - settings.recent_pages))
        scores, _, score_seconds = automatic_scores(model, source, case, candidates, settings)
        planning_started = time.perf_counter()
        plans = quality_plans(candidates, scores, case['evidence_pages'], suite)
        planning_seconds = time.perf_counter() - planning_started
        expanded_selections.append({'case_id': case['case_id'], 'scores': scores,
                                    'score_seconds': score_seconds, 'plans': plans,
                                    'all_plan_assignment_seconds': planning_seconds})
        save_json(OUTPUT / 'expanded-selection.json', expanded_selections)
        forced = tokenizer(case['answer'], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        _, reference = run_policy(model, tokenizer, source, case, settings,
                                  'native_fp16', forced=forced, backend=BACKEND)
        initial_by_budget = {}
        random.Random(case['content_seed'] + len(expanded_rows)).shuffle(plans)
        for plan in plans:
            diagnostic, logits = run_policy(model, tokenizer, source, case, settings,
                plan['policy'], plan['protected'], backend=BACKEND, forced=forced)
            if plan['budget'] is not None:
                previous = initial_by_budget.setdefault(plan['budget'], diagnostic['initial_resident_bytes'])
                assert previous == diagnostic['initial_resident_bytes'], 'Unequal physical initial budget'
            divergence = kl_divergence(reference, logits)
            gold_nll = -logits.log_softmax(-1).gather(-1, torch.tensor(forced).unsqueeze(-1)).mean()
            free, _ = run_policy(model, tokenizer, source, case, settings,
                plan['policy'], plan['protected'], backend=BACKEND)
            expanded_rows.append({**free, **plan,
                'family_id': case['family_id'], 'task': case['task'],
                'content_seed': case['content_seed'], 'position': case['position'],
                'target_length': case['target_length'], 'prompt_tokens': len(case['ids']),
                'mean_kl': float(divergence.mean()), 'first_answer_kl': float(divergence[0]),
                'gold_nll': float(gold_nll),
                'top1_agreement': float((reference.argmax(-1) == logits.argmax(-1)).float().mean()),
                'score_seconds': score_seconds if plan['policy'] == 'automatic' else 0,
                'timing_scope': 'instrumented quality path, not steady-state throughput'})
            save_json(OUTPUT / 'expanded-quality.json', expanded_rows)
        del source, reference, logits
        clean()
else:
    save_json(OUTPUT / 'expanded-quality-status.json', {'status': 'disabled'})

# %% [markdown]
# ## Controlled long decode: no per-token CPU logits or sampling
#
# For each policy and measured length, run one full untimed workload to warm the
# exact path, then three shuffled timing repetitions. Each fresh cache receives
# 32 warm-up inputs followed by exactly 128, 256, or 512 measured inputs. The
# inputs are identical across policies, preallocated on GPU, and exported; EOS
# does not stop this synthetic workload. It is NOT an accuracy/reasoning test.
# This measures the model plus Python cache adapter, online packing, and attention,
# not just a kernel. CUDA-event elapsed time includes GPU timeline launch gaps;
# synchronized wall time is the primary throughput denominator. No `.cpu()` or
# `.item()` is added inside the measured decode loop. Full-model prefill and
# selector work are excluded from steady-state time and exported separately.
# Setup, warm-up, steady-state, memory, and cache growth have separate fields.
# Packed-gate failure skips this phase rather than reporting dense fallback as
# packed throughput. This phase is new and needs validation on your T4.

# %%
decode_rows = []
decode_warmups = []
if RUN_CONTROLLED_DECODE and BACKEND == 'packed':
    # One representative longest prefix isolates runtime scaling, not generality.
    decode_case = max(expanded_cases, key=lambda c: len(c['ids']))
    sync()
    started = time.perf_counter()
    source = prefill_case(model, decode_case)
    sync()
    decode_prefill_seconds = time.perf_counter() - started
    candidates = list(range(1, (len(decode_case['ids']) - 1) // settings.block - settings.recent_pages))
    scores, _, decode_score_seconds = automatic_scores(model, source, decode_case, candidates, settings)
    performance_suite = SuiteSettings(budgets=(suite.budgets[0],), random_seeds=(suite.random_seeds[0],))
    plans = [p for p in quality_plans(candidates, scores, decode_case['evidence_pages'], performance_suite)
             if p['policy'] in ('native_fp16', 'paged_fp16', 'int4', 'automatic', 'recent', 'random')]
    max_steps = suite.decode_warmup_tokens + max(suite.decode_lengths)
    pattern = tokenizer(' Diagnostic fixed continuation for cache aging and throughput.',
                        add_special_tokens=False).input_ids
    workload = [decode_case['ids'][-1]] + (pattern * (max_steps // len(pattern) + 1))[:max_steps - 1]
    save_json(OUTPUT / 'decode-workload.json', {
        'case_id': decode_case['case_id'], 'input_tokens': workload,
        'measured_lengths': suite.decode_lengths, 'warmup_tokens': suite.decode_warmup_tokens,
        'prefill_snapshot_seconds': decode_prefill_seconds,
        'automatic_score_seconds': decode_score_seconds, 'plans': plans,
        'scope': 'synthetic fixed-input replay; never score as task accuracy',
    })
    for length in suite.decode_lengths:
        inputs = workload[:suite.decode_warmup_tokens + length]
        warm_order = list(plans)
        random.Random(settings.seed + length).shuffle(warm_order)
        for plan in warm_order:
            row = controlled_decode(model, source, settings, plan['policy'], inputs, length,
                suite.decode_warmup_tokens, plan['protected'], BACKEND)
            decode_warmups.append({**row, **plan, 'phase': 'excluded_full_workload_warmup'})
            save_json(OUTPUT / 'decode-warmups.json', decode_warmups)
        expected_initial = None
        for repetition in range(suite.decode_repeats):
            order = list(plans)
            random.Random(settings.seed + length + repetition).shuffle(order)
            for plan in order:
                row = controlled_decode(model, source, settings, plan['policy'], inputs, length,
                    suite.decode_warmup_tokens, plan['protected'], BACKEND)
                if plan['budget'] is not None:
                    if expected_initial is None:
                        expected_initial = row['initial_cache']['resident_bytes']
                    assert row['initial_cache']['resident_bytes'] == expected_initial
                if plan['policy'] not in ('native_fp16', 'paged_fp16'):
                    assert row['new_measured_demotions'] > 0, 'Generated pages did not age into compressed storage'
                row.update({**plan, 'repeat': repetition, 'case_id': decode_case['case_id'],
                            'prefill_snapshot_seconds': decode_prefill_seconds,
                            'selector_seconds': plan['assignment_seconds'] +
                                (decode_score_seconds if plan['policy'] == 'automatic' else 0)})
                row['accounted_workload_seconds'] = (row['setup_warmup_decode_seconds'] +
                    row['prefill_snapshot_seconds'] + row['selector_seconds'])
                decode_rows.append(row)
                save_json(OUTPUT / 'controlled-decode.json', decode_rows)
                print('Decode:', length, 'tokens;', plan['variant'], 'repeat', repetition + 1,
                      round(row['steady_tokens_per_second'], 2), 'tokens/s', flush=True)
    del source
    clean()
else:
    save_json(OUTPUT / 'controlled-decode-status.json', {
        'status': 'skipped', 'reason': 'disabled' if not RUN_CONTROLLED_DECODE else 'packed gate did not pass'})

# %% [markdown]
# ## Expanded report and decision evidence
#
# Random draws are first averaged per case/policy/budget. Paired differences are
# then averaged within each task/content-seed family across positions and lengths.
# Only those family means enter the bootstrap. Six default families still give
# weak uncertainty estimates; inspect task/length/position breakdowns and failures.
# Evidence-oracle retention is a semantic control, not a causal upper bound.

# %%
expanded_report = ['# Expanded benchmark report', '',
    f'Backend: {BACKEND}; packed gate: {gates["packed_status"]}.',
    f'Quality prompts: {len(expanded_cases)}; families: {len({c["family_id"] for c in expanded_cases})}.',
    'Synthetic tasks only; arithmetic uses thinking-disabled exact final answers.', '',
    'Quality timings contain CPU-logit instrumentation. Use controlled decode for throughput.',
    'Controlled decode uses one prefix and synthetic inputs, not naturally generated reasoning.',
    'CUDA timeline elapsed time is not pure kernel execution time.', '',
]
if expanded_rows:
    frame = pd.DataFrame(expanded_rows)
    frame['budget'] = frame['budget'].fillna(-1)  # -1 denotes non-mixed baselines in CSV only.
    keys = ['case_id', 'family_id', 'task', 'content_seed', 'target_length', 'position', 'policy', 'budget']
    metrics = ['completed_correct', 'answer_correct', 'mean_kl', 'first_answer_kl', 'gold_nll',
               'top1_agreement', 'resident_bytes', 'initial_resident_bytes']
    collapsed = frame.groupby(keys, as_index=False)[metrics].mean()
    quality_summary = collapsed.groupby(['policy', 'budget'], as_index=False)[metrics].mean()
    breakdown = collapsed.groupby(['task', 'target_length', 'position', 'policy', 'budget'], as_index=False)[metrics].mean()
    frame.to_csv(OUTPUT / 'expanded-quality.csv', index=False)
    collapsed.to_csv(OUTPUT / 'expanded-per-case.csv', index=False)
    quality_summary.to_csv(OUTPUT / 'expanded-summary.csv', index=False)
    breakdown.to_csv(OUTPUT / 'expanded-breakdown.csv', index=False)
    failures = [r for r in expanded_rows if not r['completed_correct']]
    save_json(OUTPUT / 'expanded-failures.json', failures)
    comparisons = []
    for budget in suite.budgets:
        for control in ('random', 'recent', 'oracle'):
            for metric in ('mean_kl', 'completed_correct'):
                differences = paired_family_differences(expanded_rows, control, metric, budget)
                values = list(differences.values())
                comparisons.append({'control': control, 'budget': budget, 'metric': metric,
                    'direction': 'automatic minus control', 'family_differences': differences,
                    'family_count': len(values), 'mean_difference': statistics.mean(values),
                    'family_bootstrap_95pct': bootstrap_interval(values, settings.seed) if len(values) > 1 else None,
                    'warning': 'Few synthetic families; exploratory, not broad validation'})
    save_json(OUTPUT / 'expanded-paired-comparisons.json', comparisons)
    display(quality_summary.round(6))
    expanded_report += ['## Quality (random draws averaged per case)', '',
                        '```', quality_summary.to_string(index=False), '```', '',
                        f'Incomplete or incorrect runs: {len(failures)} / {len(expanded_rows)}.', '']
if decode_rows:
    decode_frame = pd.json_normalize(decode_rows)
    decode_frame.to_csv(OUTPUT / 'controlled-decode.csv', index=False)
    decode_summary = decode_frame.groupby(['policy', 'measured_tokens'], as_index=False).agg(
        median_tokens_per_second=('steady_tokens_per_second', 'median'),
        min_tokens_per_second=('steady_tokens_per_second', 'min'),
        max_tokens_per_second=('steady_tokens_per_second', 'max'),
        setup_seconds=('setup_seconds', 'median'), warmup_seconds=('warmup_seconds', 'median'),
        measured_wall_seconds=('measured_wall_seconds', 'median'),
        accounted_workload_seconds=('accounted_workload_seconds', 'median'),
        final_resident_bytes=('final_cache.resident_bytes', 'median'),
        measured_peak_allocated_bytes=('measured_peak_allocated_bytes', 'median'))
    decode_summary.to_csv(OUTPUT / 'controlled-decode-summary.csv', index=False)
    display(decode_summary.round(4))
    expanded_report += ['## Controlled throughput (one prefix; three repeats by default)', '',
                        '```', decode_summary.to_string(index=False), '```', '']
expanded_report += ['## What to decide next', '',
    '- Does automatic retention improve completed accuracy or KL over multiple random draws at equal bytes?',
    '- Is any effect consistent across tasks, positions, lengths, and content seeds?',
    '- Does steady-state decoding improve while setup/selection still dominate total cost?',
    '- If no reliable selector benefit appears, keep the negative result and simplify or revise the score.',
    '- Optimize packing/allocation or kernel execution only after identifying the measured bottleneck.',
    '- Production serving, batching, standard benchmarks, other models, and true reasoning traces remain future work.']
(OUTPUT / 'expanded-report.md').write_text('\n'.join(expanded_report), encoding='utf-8')
print('Expanded report:', OUTPUT / 'expanded-report.md')
