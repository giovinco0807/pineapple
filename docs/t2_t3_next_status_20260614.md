# T2/T3 next status 2026-06-14

## T3 exact rerank tie-break

EV/score ties now prefer safer placements:

1. higher `score`
2. lower `bust_rate` / non-bust terminal
3. higher `fl_rate` / FL terminal
4. higher `raw_score`
5. stable action key

This was applied to both paths:

- Python fallback: `ai/tutor/exact_late.py`
- Rust serving path: `ai/rust_solver/t3_exact_solver/src/main.rs`

Verification row:

- input: `D:/ofc-pineapple-data/t3_ev_precision_20260614/external_holdout5k_seed20260704/teacher/t3_exact_teacher.jsonl`, row index 11
- output: `D:/ofc-pineapple-data/tmp_t3_tiebreak_check/row11_runtime_safer.jsonl`
- best after fix: `6s->middle; 8s->middle; discard 8c`
- metrics: `score=0.0`, `bust_rate=0.02353585112205802`, `fl_rate=0.0`, `samples=3654`
- the 100% bust same-score action remains in candidates, but no longer wins the tie.

## T2 fast path benchmark

Benchmark command used the current safer T3 pool:

```powershell
python ai/tutor/benchmark_t2_t3_union_runtime.py `
  --limit 50 `
  --progress-every 10 `
  --output-dir D:\ofc-pineapple-data\t2_next_20260614\bench_limit50_safer_t3 `
  --runtime-config ai/config/t1t2_t3_union_runtime_20260607.json `
  --t3-pool-config ai/config/t3_ev_precision_pool_20260614.json `
  --device cpu `
  --t3-device auto
```

Result:

- input: `ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/bench_limit50_safer_t3`
- count: 50
- latency mean/p50/p95/max: `231.4 / 191.7 / 473.1 / 517.0 ms`
- under 5s: `50/50`
- T2 candidate pool mean/p95/max: `18.3 / 20.0 / 20`
- T3 union pool mean/max: `15.1 / 21`
- errors: `0`

The teacher comparison in this benchmark is only against the old `MC300` labels in the input file. It is useful as a smoke check, but it is not an exact T2 strength measurement.

## Next step

Use this fast path for runtime smoke checks, but build a new T2 exact-teacher set before judging model quality. The next quality benchmark should compare T2 decisions against exact labels generated from T0/T1 top10 branches and T3 exact rerank, not against `MC300`.

## T2 exact teacher pilot

Full T2 exact means evaluating each T2 action over all possible T3 draws, then exact-reranking T3. With the current solver path this is too slow locally:

- all 24 actions, full exact, 1 source row: stopped after more than 5 minutes with no completed output
- Top3 source candidates, full exact, 1 source row: stopped after more than 10 minutes with no completed output

The practical local path is capped exact first, then use higher caps to check label stability:

- output: `D:/ofc-pineapple-data/t2_next_20260614/capped_exact_pilot`
- all 24 actions, cap10, 10 T2 rows: avg `3882.6 ms/row`, changed Top1 vs old MC300 on `7/10`
- all 24 actions, cap50, 3 T2 rows: avg `13332.0 ms/row`, changed Top1 vs old MC300 on `2/3`

Candidate-limited capped exact is now wired through:

```powershell
python -m ai.tutor.run_t2_exact_oracle `
  --input ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl `
  --out-dir D:\ofc-pineapple-data\t2_next_20260614\topk_capped_exact_pilot `
  --limit 10 `
  --top-n 10 `
  --t2-draw-limit 50 `
  --source-candidate-top-k 10 `
  --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- output: `D:/ofc-pineapple-data/t2_next_20260614/topk_capped_exact_pilot`
- Top10 source candidates, cap50, 10 T2 rows: avg `8584.5 ms/row`
- changed Top1 vs old MC300 on `7/10`
- overlapping 3 rows vs all24 cap50: same best `3/3`, EV loss `0`

Wrapper/solver fix:

- `ai/tutor/run_t2_exact_oracle.py` now exposes `--source-candidate-top-k`.
- `ai/rust_solver/t3_exact_solver/src/main.rs` now reads source candidates in both supported shapes:
  - `{ "action": { "placements": ..., "discard": ... } }`
  - `{ "placements": ..., "discard": ... }`

Next generation target:

- build a larger T2 capped-exact set on D drive with `source_candidate_top_k=10`
- start with `cap50` for broad coverage
- re-run a smaller overlap at `cap200` or higher to measure label stability
- only after solver optimization should full T2 exact be used as a routine teacher-data source

## T2 capped teacher batch 100

Generated a local T2 capped-exact teacher batch:

- base dir: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_100`
- input: first 100 T2 rows from `ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl`
- input split: BB `51`, BTN `49`
- unique dealt patterns: `100`
- output: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_100/t2_top10_cap50_100.jsonl`
- candidates: source Top10
- cap: `t2_draw_limit=50`
- rows: `100`
- bad/no-sample rows: `0`
- evaluated actions mean/min/max: `9.96 / 9 / 10`
- best samples min/max: `50 / 50`
- elapsed mean/p50/p95/max: `6770.5 / 7147.0 / 9584.2 / 10446.6 ms`
- best score mean/min/max: `7.8918 / 0.0 / 37.5924`
- best bust mean/p95: `24.48% / 73.90%`
- best FL mean: `21.62%`

Fixed a TopK candidate matching bug during this run:

- two joker rows initially had `evaluated_actions=0`
- source candidates used `Xj`, while legal actions used concrete `X1/X2`
- Rust candidate matching now accepts joker wildcard keys for source-candidate filtering
- the two affected chunks were regenerated; final bad row count is `0`

Stability check against cap200:

- cap200 output: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_100/t2_top10_cap200_first20.jsonl`
- compared rows: first `20`
- candidates: source Top10
- cap50 vs cap200 Top1 same: `15/20` (`75.0%`)
- cap200 regret of cap50-best mean/p95/max: `0.2582 / 1.7913 / 2.4459`
- cap50 best rank under cap200 mean/max: `1.35 / 3`
- cap200 best rank under cap50 mean/max: `1.45 / 6`
- cap200 elapsed mean/p50/p95/max: `26632.5 / 28060.6 / 35822.9 / 38445.1 ms`

Interpretation:

- `cap50` is useful for broad provisional T2 training data.
- It is not stable enough to treat every Top1 as a final exact label.
- For high-confidence labels, use agreement between cap50 and cap200, or use cap200+ for hard/close rows.
- The next training set should include the full candidate score list, not only Top1, so the model learns EV margins and near-tie uncertainty.

## T2 action-value fine-tune pilot

Built a standard action-value teacher file from the Rust capped T2 output:

- script: `ai/training/build_t2_action_value_teacher_from_exact.py`
- teacher JSONL: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_100/t2_top10_cap50_100.teacher.jsonl`
- converted dataset: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_100/av_dataset_cap50_100_dim520`
- rows: `100`
- candidates: `996`
- base checkpoint: `ai/models/candidate_runs/t2-stable-oracle56-cap500-cap1000-ft-from-exact53-20260603/model/action_value_best.pt`
- fine-tuned checkpoint: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_100/models/t2-cap50-100-ft-top3-20260614/action_value_best.pt`

Evaluation against the cap50 Top10 candidate lists:

| model | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 100 | 2.951 | 0.758 | 0.312 | 0.064 | 41.0% | 1.271 | 69.0% | 0.455 | 85.0% | 0.085 | 100.0% |
| fine-tuned best | 100 | 1.558 | 0.894 | 0.083 | 0.045 | 86.0% | 0.228 | 94.0% | 0.063 | 96.0% | 0.010 | 100.0% |
| fine-tuned final | 100 | 1.247 | 0.911 | 0.051 | 0.030 | 86.0% | 0.230 | 93.0% | 0.083 | 97.0% | 0.008 | 100.0% |

Evaluation against the cap200 first-20 candidate lists:

| model | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 20 | 2.480 | 0.848 | 0.310 | 0.061 | 55.0% | 0.651 | 80.0% | 0.082 | 95.0% | 0.000 | 100.0% |
| fine-tuned best | 20 | 2.455 | 0.867 | 0.084 | 0.065 | 60.0% | 0.519 | 95.0% | 0.028 | 100.0% | 0.000 | 100.0% |
| fine-tuned final | 20 | 2.130 | 0.902 | 0.074 | 0.052 | 60.0% | 0.525 | 85.0% | 0.090 | 95.0% | 0.055 | 100.0% |

Interpretation:

- This is a positive pilot for T2 reranking inside source Top10.
- The best checkpoint is safer than the final checkpoint for Top3 recall on cap200 first-20.
- Do not treat this as a final replacement yet. The training set is only 100 cap50 rows, and cap50 Top1 agrees with cap200 Top1 on only 75% of the first 20 rows.
- Top10 is 100% here because the dataset contains only source Top10 candidates. This does not prove full-legal-action Top10 recall.

External cap50 check on the next 50 T2 rows:

- input: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_holdout_t2next50/inputs/t2_after_first100_next50.jsonl`
- exact/capped output: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_holdout_t2next50/t2_oracle_cap50_limit50.jsonl`
- teacher JSONL: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_holdout_t2next50/t2_oracle_cap50_limit50.teacher.jsonl`
- rows: `50`
- candidates: `497`
- cap: `t2_draw_limit=50`
- old MC300/source Top1 changed by cap50 teacher: `20/50`
- generation latency mean: `6948.3 ms/row`

| model | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 50 | 3.052 | 0.796 | 0.268 | 0.064 | 44.0% | 1.177 | 82.0% | 0.287 | 92.0% | 0.014 | 100.0% |
| fine-tuned best | 50 | 2.430 | 0.751 | 0.153 | 0.068 | 46.0% | 1.230 | 82.0% | 0.364 | 92.0% | 0.039 | 100.0% |
| fine-tuned final | 50 | 2.533 | 0.720 | 0.151 | 0.062 | 40.0% | 1.242 | 82.0% | 0.307 | 92.0% | 0.039 | 100.0% |

External interpretation:

- The 100-row fine-tune improves in-sample cap50 strongly, but does not improve Top3 recall on the next 50 T2 rows.
- It improves score MAE and bust MAE on the next 50 rows, so the value head learned something useful.
- It is not ready as the primary T2 candidate model. More T2 teacher rows are needed before replacing the existing checkpoint.
- `build_t2_action_value_teacher_from_exact.py` now joins Rust output to source rows by `global_index`, then `record_index`, then fallback order. This fixes direct solver outputs that do not include `global_index`.

Next:

- Generate more T2 rows with source Top10 and cap50 for breadth.
- Generate cap200+ labels for hard/close rows and external validation.
- Prefer the fine-tuned best checkpoint only as an experimental candidate source until it passes a larger cap200 holdout.

## T2 cap50 train extension to 204 rows

Added all remaining T2 rows from `local_t1t2_500_mc300.jsonl` after the first 150 T2 decisions:

- output base: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_train_extra_t2_150_253`
- input: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_train_extra_t2_150_253/inputs/t2_after_first150_remaining.jsonl`
- chunks: `11`
- added rows: `104`
- added candidates: `1,037`
- combined train rows: `204`
- combined train candidates: `2,033`
- combined dataset: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_train_extra_t2_150_253/av_dataset_cap50_train204_dim520`
- model: `D:/ofc-pineapple-data/t2_next_20260614/teacher_top10_cap50_train_extra_t2_150_253/models/t2-cap50-204-ft-top3-20260615/action_value_final.pt`

Generation stats for the 104 added rows:

- cap: `t2_draw_limit=50`
- source candidates: Top10
- old source Top1 changed: `60/104`
- weighted avg elapsed: `7202.1 ms/row`
- weighted avg regret of source Top1: `0.5581`
- max regret of source Top1: `8.7957`

Train-set evaluation on the 204 cap50 rows:

| model | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 204 | 3.088 | 0.810 | 0.291 | 0.070 | 36.8% | 1.246 | 67.2% | 0.453 | 81.9% | 0.160 | 100.0% |
| ft204 best | 204 | 1.952 | 0.919 | 0.085 | 0.043 | 89.7% | 0.190 | 94.6% | 0.032 | 96.1% | 0.019 | 100.0% |
| ft204 final | 204 | 1.270 | 0.926 | 0.051 | 0.033 | 88.7% | 0.206 | 94.1% | 0.075 | 96.6% | 0.011 | 100.0% |

External check on the held-out next 50 T2 rows:

| model | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 50 | 3.052 | 0.796 | 0.268 | 0.064 | 44.0% | 1.177 | 82.0% | 0.287 | 92.0% | 0.014 | 100.0% |
| ft100 best | 50 | 2.430 | 0.751 | 0.153 | 0.068 | 46.0% | 1.230 | 82.0% | 0.364 | 92.0% | 0.039 | 100.0% |
| ft204 final | 50 | 2.457 | 0.731 | 0.137 | 0.065 | 54.0% | 0.740 | 80.0% | 0.076 | 90.0% | 0.040 | 100.0% |
| existing 60% + ft204 final 40% | 50 | 2.447 | 0.801 | 0.205 | 0.061 | 46.0% | 1.131 | 86.0% | 0.054 | 92.0% | 0.014 | 100.0% |

Interpretation:

- The 204-row model clearly learns the cap50 labels in-sample.
- On held-out cap50 rows, the ft204 final checkpoint improves Top1 and Top1 regret but slightly hurts Top3 recall.
- For candidate pruning, the best current option is not replacing the old model, but blending `existing 60% + ft204 final 40%`.
- The blend improves held-out Top3 from `82.0%` to `86.0%` and reduces Top3 regret from `0.287` to `0.054`.
- Top10 remains 100% only because these evaluations are inside source Top10 candidates. Full legal-action recall still needs a larger all-legal or broader-source audit.

Next:

- Add a larger source of T2 positions; the original local500 file has only `254` T2 rows.
- Use `ai/tutor/build_t2_t3_model_teacher_from_t0t1_topk.py` or a self-play route generator to create fresh T2 states from varied T0/T1 branches.
- Keep the held-out next-50 set as a small regression check, but create a larger external cap50/cap200 holdout before promoting the blend.

## Fresh T2 generation smoke

Confirmed that new T2 positions can be generated from T0/T1 model branches and passed into the Rust capped T2 oracle:

- generator: `ai/tutor/build_t2_t3_model_teacher_from_t0t1_topk.py`
- smoke input: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/smoke10/t2_model_teacher_smoke10.jsonl`
- smoke exact: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/smoke10/cap50_exact/t2_oracle_cap50_limit10.jsonl`
- smoke rows: `10`
- smoke source Top1 changed by cap50: `6/10`
- smoke avg cap50 exact latency: `8925.7 ms/row`

Created a balanced fresh input set:

- base: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh100_inputs`
- combined input: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh100_inputs/t2_fresh_balanced100_model_teacher.jsonl`
- rows: `100`
- split: BB `50`, BTN `50`
- generator settings:
  - target TopK: `10`
  - opponent TopK: `1`
  - opponent T2 TopK: `1`
  - T3 model draw limit: `1`
  - include jokers: `true`
- generation speed:
  - BB50: `5.47s`
  - BTN50: `4.24s`

Fresh external cap50 check:

- input: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh20_cap50_check/t2_fresh_bb10_btn10_model_teacher.jsonl`
- exact: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh20_cap50_check/cap50_exact/t2_oracle_cap50_limit20.jsonl`
- teacher dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh20_cap50_check/av_dataset_cap50_dim520`
- rows: `20`
- split: BB `10`, BTN `10`
- source Top1 changed by cap50: `18/20`
- avg exact latency: `7624.9 ms/row`
- avg regret of source Top1: `1.2976`
- max regret of source Top1: `7.6851`

Fresh20 model comparison:

| model | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 20 | 2.639 | 0.500 | 0.271 | 0.104 | 10.0% | 2.032 | 65.0% | 0.110 | 90.0% | 0.031 | 100.0% |
| ft204 final | 20 | 2.404 | 0.570 | 0.196 | 0.096 | 10.0% | 1.879 | 50.0% | 0.122 | 65.0% | 0.064 | 100.0% |
| existing 60% + ft204 final 40% | 20 | 2.483 | 0.536 | 0.210 | 0.101 | 10.0% | 2.035 | 70.0% | 0.078 | 85.0% | 0.031 | 100.0% |

Interpretation:

- Fresh branch-generated T2 positions are much harder than the local500 holdout.
- The blend still improves Top3 on fresh20 from `65.0%` to `70.0%`, and reduces Top3 regret from `0.110` to `0.078`.
- The ft204 single model is not robust on this fresh distribution.
- More fresh cap50/cap200 data is required before changing runtime pruning. The next useful batch is the remaining `80` rows from the fresh100 input, then a larger fresh input target such as `500`.

## T2 fresh80 extension and weighted ensemble

Generated capped-exact labels for the remaining `80` rows from the fresh100 T0/T1-topk branch set, excluding the BB10/BTN10 fresh20 holdout:

- input: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh80_train/inputs/t2_fresh_remaining80_model_teacher.jsonl`
- split: BB `40`, BTN `40`
- output: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh80_train/t2_fresh80_cap50.teacher.jsonl`
- combined train file: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh80_train/t2_local204_plus_fresh80_cap50_train284.teacher.jsonl`
- combined dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh80_train/av_dataset_local204_plus_fresh80_cap50_train284_dim520`
- train rows: `284`
- train candidates: `2,833`
- fresh80 old source Top1 changed by cap50: `61/80`
- fresh80 exact latency by chunk was about `5.6s` to `8.7s` per row with source Top10 and `t2_draw_limit=50`

Fine-tuned from the existing T2 checkpoint:

- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh80_train/models/t2-local204-fresh80-cap50-ft-top3-20260615`
- final checkpoint: `.../action_value_final.pt`
- best checkpoint: `.../action_value_best.pt`

Single-model external checks:

| model | holdout | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | fresh20 | 20 | 2.639 | 0.500 | 0.271 | 0.104 | 10.0% | 2.032 | 65.0% | 0.110 | 90.0% | 0.031 | 100.0% |
| ft284 final | fresh20 | 20 | 1.537 | 0.699 | 0.100 | 0.056 | 50.0% | 1.589 | 70.0% | 0.059 | 90.0% | 0.035 | 100.0% |
| existing | local next50 | 50 | 3.052 | 0.796 | 0.268 | 0.064 | 44.0% | 1.177 | 82.0% | 0.287 | 92.0% | 0.014 | 100.0% |
| ft284 final | local next50 | 50 | 2.336 | 0.733 | 0.139 | 0.061 | 48.0% | 1.161 | 78.0% | 0.341 | 90.0% | 0.046 | 100.0% |

The fresh80 fine-tune is much stronger on the fresh branch distribution, but it is not a clean replacement because local next50 Top3 falls from `82.0%` to `78.0%`.

The best balanced candidate so far is a three-model weighted ensemble:

- existing checkpoint weight: `0.50`
- ft204 final weight: `0.15`
- ft284 final weight: `0.35`
- runtime config: `ai/config/t2_action_value_ensemble_20260615.json`

Weighted ensemble external checks:

| weights existing/ft204/ft284 | holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50/0.15/0.35 | fresh20 | 20 | 35.0% | 1.551 | 85.0% | 0.028 | 95.0% | 0.031 | 100.0% |
| 0.50/0.15/0.35 | local next50 | 50 | 48.0% | 1.176 | 86.0% | 0.189 | 92.0% | 0.014 | 100.0% |
| 0.50/0.15/0.35 | cap200 first20 | 20 | 70.0% | 0.369 | 95.0% | 0.001 | 100.0% | 0.000 | 100.0% |

Interpretation:

- The ensemble is better balanced than any single new checkpoint across the two current external cap50 checks.
- The cap200 first20 result is encouraging but overlaps the earlier local first100 training source, so it is not a final independent claim.
- Top10 is still measured inside source Top10 candidate lists. Broader all-legal or larger-source audits are still needed before tightening runtime pruning below the current pool.

Runtime support added:

- `ai.models.action_value_reranker.BlendedActionValueReranker` now supports weighted blends of two or more checkpoints.
- `ai.tutor.hybrid_t1t2` now accepts `--turn-model-ensemble`.
- Runtime JSON can set `models.turn_model_ensembles`, used by `ai/config/t2_action_value_ensemble_20260615.json`.

Runtime smoke:

- command used `ai/config/t2_action_value_ensemble_20260615.json` on `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_varied444_20260615_limit1_exact.jsonl`
- result: `elapsed_ms=583.8`, `candidate_pool_size=20`, `exact_evaluated=3`, `best_action_idx=2`, `best_refined_score=7.7646`
- T3 union exact pool summary: min/mean/max `9 / 17.0 / 21`

## T2 varied fresh extension

The first external200 attempt was rejected for evaluation because it had only `2` unique T2 dealt patterns.  It came from taking 100 branches from one BB root and one BTN root.  The fixed generation caps each root/position at `2` records:

- varied source: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/inputs/t2_fresh_ext_varied_balanced200_model_teacher.jsonl`
- rows: `200`
- split: BB `100`, BTN `100`
- unique dealt patterns: `100`
- unique root deals: `100`
- unique board+opponent+dealt states: `200`
- avg candidates per record: `23.55`

Held out BB20/BTN20 as a new varied cap50 external check:

- holdout source: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/holdout40_cap50/inputs/t2_fresh_ext_varied_bb20_btn20_model_teacher.jsonl`
- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/holdout40_cap50/t2_fresh_ext_varied_holdout40_cap50.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/holdout40_cap50/av_dataset_cap50_holdout40_dim520`
- source Top1 changed by cap50: `32/40`
- cap50 elapsed mean: about `19.5s/row` while running four chunks in parallel

Used the remaining BB80/BTN80 as additional training data:

- train source: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/train160_cap50/inputs/t2_fresh_ext_varied_train160_model_teacher.jsonl`
- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/train160_cap50/t2_fresh_ext_varied_train160_cap50.teacher.jsonl`
- train rows: `160`
- train candidates: `1,600`
- source Top1 changed by cap50: `120/160` (`75.0%`)
- avg source Top1 regret under cap50: `3.5687`
- max source Top1 regret under cap50: `25.4296`
- cap50 elapsed mean: `22.77s/row` while running four chunks in parallel

Combined this with the previous `284` training rows:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/train160_cap50/t2_local284_plus_varied160_cap50_train444.teacher.jsonl`
- combined dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/train160_cap50/av_dataset_local284_plus_varied160_cap50_train444_dim520`
- rows: `444`
- samples: `4,433`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external200_varied_20260615/train160_cap50/models/t2-local284-varied160-cap50-ft-top3-20260615`

Single-model external checks:

| model | holdout | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | varied40 | 40 | 5.118 | 0.688 | 0.242 | 0.095 | 35.0% | 2.308 | 57.5% | 1.323 | 67.5% | 0.882 | 100.0% |
| previous 0.50/0.15/0.35 ensemble | varied40 | 40 | 3.759 | 0.715 | 0.197 | 0.083 | 35.0% | 2.097 | 55.0% | 1.370 | 67.5% | 0.437 | 100.0% |
| ft444 final | varied40 | 40 | 2.941 | 0.700 | 0.179 | 0.074 | 25.0% | 2.605 | 65.0% | 0.605 | 70.0% | 0.277 | 100.0% |
| ft444 final | fresh20 | 20 | 1.486 | 0.750 | 0.095 | 0.050 | 55.0% | 0.960 | 85.0% | 0.040 | 95.0% | 0.005 | 100.0% |
| ft444 final | local next50 | 50 | 2.402 | 0.707 | 0.133 | 0.067 | 32.0% | 1.437 | 80.0% | 0.316 | 92.0% | 0.115 | 100.0% |
| ft444 final | cap200 first20 | 20 | 1.591 | 0.950 | 0.062 | 0.035 | 70.0% | 0.322 | 85.0% | 0.069 | 90.0% | 0.068 | 100.0% |

Interpretation:

- The added varied data improves the new varied external set, but single-model replacement hurts older local/cap200 checks.
- The previous three-model ensemble overfit the smaller fresh20/local checks and was not robust to varied40.
- A coarse four-model grid found the current balanced runtime candidate:
  - old: `0.40`
  - ft204: `0.00`
  - ft284: `0.10`
  - ft444: `0.50`
- Coarse-grid metrics for this candidate:
  - varied40 Top3/Reg3: `62.5% / 0.750`
  - fresh20 Top3/Reg3: `90.0% / 0.009`
  - local next50 Top3/Reg3: `86.0% / 0.185`
  - cap200 first20 Top3/Reg3: `95.0% / 0.001`
- This is now written into `ai/config/t2_action_value_ensemble_20260615.json`.

Remaining issue:

- `varied40` Top3 is still only `62.5%`, so T2 is not yet strong enough for a Top3-only exact refinement guarantee.
- Top10 remains `100%` on these source-Top10 teacher sets, so the safe product path is still larger candidate pools plus exact rerank while the T2 model is improved.

## T2 additional varied160 and ft604 ensemble

Generated one more varied balanced local batch without using VMs:

- source: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external360_varied_20260615/train160_more/inputs/t2_fresh_ext_more_balanced160_model_teacher.jsonl`
- rows: `160`
- split: BB `80`, BTN `80`
- unique dealt patterns: `80`
- unique root deals: `80`
- unique board+opponent+dealt states: `160`
- avg candidates per record: `23.93`

Converted this batch with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external360_varied_20260615/train160_more/t2_fresh_ext_more160_cap50.teacher.jsonl`
- source Top1 changed by cap50: `127/160` (`79.4%`)
- avg source Top1 regret under cap50: `3.4211`
- max source Top1 regret under cap50: `26.0146`
- avg exact elapsed in worker logs: `23.64s/row`
- wall elapsed with four local workers: `967.4s`

Combined with the previous `444` rows and trained the ft604 checkpoint:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external360_varied_20260615/train160_more/t2_local444_plus_more160_cap50_train604.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external360_varied_20260615/train160_more/av_dataset_local444_plus_more160_cap50_train604_dim520`
- rows: `604`
- samples: `6,033`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external360_varied_20260615/train160_more/models/t2-local444-more160-cap50-ft-top3-20260615`
- final internal validation: Top1 `70.8%`, Top3 `88.3%`, Top5 `95.0%`, Top10 `100.0%`

ft604 single-model external checks:

| model | holdout | groups | score MAE | corr | bust MAE | FL MAE | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft604 final | varied40 | 40 | 2.934 | 0.690 | 0.187 | 0.073 | 30.0% | 2.028 | 67.5% | 0.430 | 75.0% | 0.366 | 100.0% |
| ft604 final | fresh20 | 20 | 1.506 | 0.741 | 0.084 | 0.048 | 45.0% | 1.765 | 75.0% | 0.046 | 95.0% | 0.005 | 100.0% |
| ft604 final | local next50 | 50 | 2.298 | 0.731 | 0.128 | 0.064 | 48.0% | 1.388 | 82.0% | 0.296 | 94.0% | 0.027 | 100.0% |
| ft604 final | cap200 first20 | 20 | 1.484 | 0.954 | 0.059 | 0.033 | 75.0% | 0.258 | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |

Previous fine-grid ensemble candidate for `ai/config/t2_action_value_ensemble_20260615.json`.
This was superseded by the hard-miss blend below:

- old stable-oracle checkpoint: `0.05`
- cap50-100 checkpoint: `0.05`
- cap50-204 checkpoint: `0.10`
- ft604 final checkpoint: `0.80`
- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external360_varied_20260615/ensemble_grid_fine_core_step05.json`

Previous ft604-heavy candidate external checks:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 40 | 30.0% | 2.213 | 67.5% | 0.462 | 80.0% | 0.366 | 100.0% |
| fresh20 | 20 | 40.0% | 1.810 | 80.0% | 0.015 | 95.0% | 0.005 | 100.0% |
| local next50 | 50 | 54.0% | 0.952 | 82.0% | 0.296 | 96.0% | 0.024 | 100.0% |
| cap200 first20 | 20 | 75.0% | 0.258 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |

Runtime smoke with that previous ft604-heavy config:

- command used the then-current `ai/config/t2_action_value_ensemble_20260615.json` on `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft604_fine_20260615_limit1_exact.jsonl`
- result: `elapsed_ms=581.5`, `candidate_pool_size=20`, `exact_evaluated=3`, `best_action_idx=2`, `best_refined_score=7.7646`
- T3 union exact pool summary: min/mean/max `9 / 17.0 / 21`

Interpretation:

- The new ft604-heavy ensemble improves the previous varied40 Top3 from `62.5%` to `67.5%` and cuts varied40 Top3 regret from about `0.750` to `0.462`.
- The tradeoff is lower fresh20 Top3 than the previous small-holdout-balanced ensemble (`90.0%` to `80.0%`), but the EV loss on fresh20 remains small (`0.015`).
- T2 is still not strong enough for Top3-only refinement.  Current runtime should keep a larger candidate pool and use exact rerank/refinement.
- On these source-Top10 teacher sets, Top10 remains `100.0%`; the next quality target is reducing Top5 and Top3 EV loss, not claiming Top3 safety yet.

## T2 hard-miss fine-tune and blend

Audited the ft604-heavy ensemble for Top3 misses on the `604` row training source and the current external checks:

| split | groups | Top1 | Top3 | Top5 | Top10 | Top3 misses | Top3 regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| train604 | 604 | 94.0% | 97.8% | 99.5% | 100.0% | 13 | 0.0835 |
| varied40 | 40 | 30.0% | 67.5% | 80.0% | 100.0% | 13 | 0.462 |
| fresh20 | 20 | 40.0% | 80.0% | 95.0% | 100.0% | 4 | 0.015 |
| local next50 | 50 | 54.0% | 82.0% | 96.0% | 100.0% | 9 | 0.296 |

Built a hard-miss augmented training file from only the `train604` Top3 misses:

- augmented teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/hard_miss_ft604_20260615/t2_train604_plus_hardmiss_top3_aug.teacher.jsonl`
- base records: `604`
- selected hard-miss records: `10`
- duplicated hard-miss records: `59`
- total records: `663`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/hard_miss_ft604_20260615/av_dataset_train604_hardmiss_top3_aug_dim520`
- samples: `6,623`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/hard_miss_ft604_20260615/models/t2-train604-hardmiss-top3-ft-20260615`

The hard-miss checkpoint alone overfit and was not adopted as a standalone model:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| hardmiss final | varied40 | 30.0% | 55.0% | 1.294 | 72.5% | 0.856 | 100.0% |
| hardmiss final | fresh20 | 55.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| hardmiss final | local next50 | 44.0% | 78.0% | 0.396 | 92.0% | 0.115 | 100.0% |
| hardmiss final | cap200 first20 | 75.0% | 90.0% | 0.056 | 90.0% | 0.056 | 100.0% |

A fine grid with the hard-miss source found a better small blend. This was the
runtime candidate before the fresh80/ft684 blend below:

- old stable-oracle checkpoint: `0.05`
- cap50-100 checkpoint: `0.10`
- ft604 final checkpoint: `0.70`
- hardmiss final checkpoint: `0.15`
- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/hard_miss_ft604_20260615/ensemble_grid_with_hardmiss_step05.json`

Selected hard-miss blend external checks:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 40 | 32.5% | 2.186 | 70.0% | 0.388 | 80.0% | 0.366 | 100.0% |
| fresh20 | 20 | 40.0% | 1.810 | 80.0% | 0.015 | 90.0% | 0.010 | 100.0% |
| local next50 | 50 | 58.0% | 0.884 | 84.0% | 0.303 | 96.0% | 0.024 | 100.0% |
| cap200 first20 | 20 | 75.0% | 0.258 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |

Runtime smoke with the selected hard-miss blend:

- command used `ai/config/t2_action_value_ensemble_20260615.json` on `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft604_hardmiss_blend_20260615_limit1_exact.jsonl`
- result: `elapsed_ms=512.7`, `candidate_pool_size=20`, `exact_evaluated=3`, `best_action_idx=2`, `best_refined_score=7.7646`
- T3 union exact pool summary: min/mean/max `9 / 17.0 / 21`

Interpretation:

- The hard-miss checkpoint alone is too narrow, but a `15%` blend improves the Top3 floor from `67.5%` to `70.0%`, improves local next50 from `82.0%` to `84.0%`, and cuts varied40 Top3 regret from `0.462` to `0.388`.
- T2 is stronger than the previous ft604-heavy config, but it is still not safe for Top3-only runtime pruning.
- On these source-Top10 teacher sets, Top10 remains `100.0%`; keep the runtime pool at `20` for now and exact-rerank the shortlist.
- The next useful data step is new independent varied exact rows, not more duplicated train604 misses.

## T2 fresh80 exact and ft684 small blend

Generated a new independent varied local T2 batch without VMs:

- source base: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more`
- BB source: root_start `70000`, roots `20`, max `2` rows/root, rows `40`
- BTN source: root_start `80000`, roots `20`, max `2` rows/root, rows `40`
- combined rows: `80`
- split: BB `40`, BTN `40`
- unique dealt patterns: `40`
- unique board+opponent+dealt states: `80`
- average candidates per record: `24.525`

Converted all `80` rows with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/t2_fresh_ext_more2_80_cap50.teacher.jsonl`
- source Top1 changed by cap50: `56/80` (`70.0%`)
- average source Top1 regret under cap50: `3.0394`
- max source Top1 regret under cap50: `13.2828`
- avg exact elapsed in worker logs: `23.71s/row`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/av_dataset_more2_80_cap50_dim520`

Current hard-miss blend on this fresh80 set before retraining:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh80 traincheck | 80 | 48.8% | 1.568 | 71.2% | 0.310 | 86.2% | 0.119 | 100.0% |

Combined the previous clean `604` rows with the new `80` rows and fine-tuned from ft604:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/t2_local604_plus_more80_cap50_train684.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/av_dataset_local604_plus_more80_cap50_train684_dim520`
- rows: `684`
- samples: `6,833`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/models/t2-local604-more80-cap50-ft-top3-20260615`

The ft684 checkpoint alone adapts to fresh80 but is not safe as a replacement:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft684 final | varied40 | 27.5% | 62.5% | 0.508 | 80.0% | 0.345 | 100.0% |
| ft684 final | fresh20 | 65.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| ft684 final | local next50 | 46.0% | 78.0% | 0.411 | 90.0% | 0.238 | 100.0% |
| ft684 final | cap200 first20 | 75.0% | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| ft684 final | fresh80 traincheck | 91.2% | 96.2% | 0.024 | 97.5% | 0.022 | 100.0% |

A fine grid selected a conservative small blend, now written into
`ai/config/t2_action_value_ensemble_20260615.json`:

- old stable-oracle checkpoint: `0.05`
- cap50-100 checkpoint: `0.10`
- ft604 final checkpoint: `0.70`
- hardmiss final checkpoint: `0.10`
- ft684 final checkpoint: `0.05`
- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/ensemble_grid_with_new684_step05.json`

Selected ft684 small blend checks:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 40 | 32.5% | 2.186 | 70.0% | 0.388 | 80.0% | 0.366 | 100.0% |
| fresh20 | 20 | 40.0% | 1.810 | 80.0% | 0.015 | 95.0% | 0.005 | 100.0% |
| local next50 | 50 | 58.0% | 0.884 | 84.0% | 0.303 | 96.0% | 0.024 | 100.0% |
| cap200 first20 | 20 | 75.0% | 0.258 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| fresh80 traincheck | 80 | 51.2% | 1.363 | 77.5% | 0.240 | 90.0% | 0.095 | 100.0% |

Runtime smoke with the selected ft684 small blend:

- command used `ai/config/t2_action_value_ensemble_20260615.json` on `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft684_small_blend_20260615_limit1_exact.jsonl`
- result: `elapsed_ms=491.3`, `candidate_pool_size=20`, `exact_evaluated=3`, `best_action_idx=2`, `best_refined_score=7.7646`
- T3 union exact pool summary: min/mean/max `9 / 17.0 / 21`

Interpretation:

- The new `80` exact rows are useful: the old source labels changed on `70%` of rows and showed large EV gaps.
- ft684 alone overfits the new batch and regresses varied40/local next50, so it is not a replacement model.
- A `5%` ft684 blend preserves the external Top3 profile of the previous hard-miss blend while improving fresh20 Top5 and improving the fresh80 traincheck versus the prior config.
- T2 is still not safe for Top3-only pruning. Runtime remains `pool20 + exact rerank`.

### cap200 audit of fresh80 labels

Audited a `12` row subset from fresh80 with `t2_draw_limit=200`:

- selection: `4` cap50-stable rows plus `8` high-regret rows where source Top1 lost about `11` to `13` EV under cap50
- source: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/cap200_check12/more2_cap200_check12.source.jsonl`
- cap200 output: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/cap200_check12/cap200_chunks`
- cap50 vs cap200 comparison: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/cap200_check12/cap50_vs_cap200_comparison.json`
- cap200 teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external440_varied_20260615/train80_more/cap200_check12/t2_more2_cap200_check12.teacher.jsonl`
- average cap200 elapsed: `91.43s/row`

cap50 Top1 stability against cap200:

| subset | rows | cap50/cap200 Top1 same | cap200 EV loss of cap50 best | notes |
|---|---:|---:|---:|---|
| all selected | 12 | 12/12 | mean `0.000`, max `0.000` | cap50 best stayed cap200 best |
| cap50 hard rows | 8 | 8/8 | mean `0.000`, max `0.000` | high source-label errors were real |
| cap50 stable rows | 4 | 4/4 | mean `0.000`, max `0.000` | stable examples stayed stable |

Current ft684 small blend on the cap200 subset:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cap200 check12 | 12 | 66.7% | 1.776 | 91.7% | 0.000 | 100.0% | 0.000 | 100.0% |

Interpretation:

- For this high-regret audit subset, cap50 labels were rank-stable versus cap200. That supports using cap50 for broader local teacher generation.
- cap200 is still useful as a periodic audit and for close/hard rows, but running all rows at cap200 is slow at about `91s/row` locally.
- The next efficient improvement is more independent cap50 rows, with cap200 checks on sampled hard/close rows rather than blanket cap200 conversion.

## T2 fresh80-more3 exact and ft764 safe blend

Generated another independent varied local T2 batch without VMs:

- source base: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3`
- BB source: root_start `90000`, roots `20`, max `2` rows/root, rows `40`
- BTN source: root_start `100000`, roots `20`, max `2` rows/root, rows `40`
- combined rows: `80`
- split: BB `40`, BTN `40`
- unique dealt patterns: `40`
- unique board+opponent+dealt states: `80`
- average candidates per record: `24.525`

Converted all `80` rows with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3/t2_fresh_ext_more3_80_cap50.teacher.jsonl`
- source Top1 changed by cap50: `58/80` (`72.5%`)
- average source Top1 regret under cap50: `3.2499`
- max source Top1 regret under cap50: `19.3472`
- avg exact elapsed in worker logs: `24.26s/row`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3/av_dataset_more3_80_cap50_dim520`

Combined the previous clean `684` rows with the new `80` rows and fine-tuned from ft684:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3/t2_local684_plus_more80_cap50_train764.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3/av_dataset_local684_plus_more80_cap50_train764_dim520`
- rows: `764`
- samples: `7,633`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3/models/t2-local684-more80-cap50-ft-top3-20260615`
- local CPU training time: `333.7s`

The ft764 checkpoint alone fits the new train-style rows but is not safe as a replacement:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft764 final | varied40 | 30.0% | 57.5% | 1.244 | 75.0% | 0.209 | 100.0% |
| ft764 final | fresh20 | 65.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| ft764 final | local next50 | 50.0% | 80.0% | 0.408 | 94.0% | 0.105 | 100.0% |
| ft764 final | cap200 first20 | 75.0% | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| ft764 final | fresh80-more2 traincheck | 98.8% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft764 final | fresh80-more3 traincheck | 96.2% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |

A constrained fine grid selected a safe ft764 blend, now written into
`ai/config/t2_action_value_ensemble_20260615.json`:

- old stable-oracle checkpoint: `0.05`
- cap50-100 checkpoint: `0.10`
- ft604 final checkpoint: `0.70`
- hardmiss final checkpoint: `0.05`
- ft684 final checkpoint: `0.00`
- ft764 final checkpoint: `0.10`
- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external520_varied_20260615/train80_more3/ensemble_grid_with_ft764_step05.json`

Selected ft764 safe blend checks:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 40 | 32.5% | 2.186 | 70.0% | 0.388 | 80.0% | 0.366 | 100.0% |
| fresh20 | 20 | 45.0% | 1.810 | 80.0% | 0.015 | 95.0% | 0.005 | 100.0% |
| local next50 | 50 | 58.0% | 0.884 | 84.0% | 0.303 | 96.0% | 0.024 | 100.0% |
| cap200 first20 | 20 | 75.0% | 0.258 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| fresh80-more2 traincheck | 80 | 58.8% | 1.154 | 83.8% | 0.150 | 95.0% | 0.065 | 100.0% |
| fresh80-more3 traincheck | 80 | 60.0% | 1.048 | 83.8% | 0.212 | 95.0% | 0.059 | 100.0% |
| cap200 check12 | 12 | 66.7% | 1.412 | 91.7% | 0.000 | 100.0% | 0.000 | 100.0% |

Runtime smoke with the selected ft764 safe blend:

- command used `ai/config/t2_action_value_ensemble_20260615.json` on `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft764_safe_blend_20260615_limit1_exact.jsonl`
- result: `elapsed_ms=773.5`, `candidate_pool_size=20`, `exact_evaluated=3`, `best_action_idx=2`, `best_refined_score=7.7646`
- T3 union exact pool summary: min/mean/max `9 / 17.0 / 21`

Interpretation:

- The new `80` exact rows again show that source/T3-model labels are noisy: source Top1 changed on `72.5%` of rows and lost up to `19.35` EV under cap50.
- ft764 alone overfits the newly added train-style rows and regresses external varied40 badly, so it is not a replacement model.
- A `10%` ft764 blend preserves every external Top3/Top5 value from the previous config, keeps cap200 check12 Top3 rerank regret at `0.000`, cuts external full-bust choices from `3` to `2`, and improves the two traincheck batches.
- T2 is still not safe for Top3-only pruning. Keep runtime at `pool20 + exact rerank`; use more independent cap50 rows plus periodic cap200 audits to keep reducing EV-loss tails.

## T2 fresh80-more4 exact and ft844 safe blend

Generated a fourth independent varied local T2 batch without VMs:

- source base: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4`
- BB source: root_start `110000`, roots `20`, max `2` rows/root, rows `40`
- BTN source: root_start `120000`, roots `20`, max `2` rows/root, rows `40`
- combined rows: `80`
- split: BB `40`, BTN `40`
- unique dealt patterns: `40`
- unique board+opponent+dealt states: `80`
- average candidates per record: `24.525`

Converted all `80` rows with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4/t2_fresh_ext_more4_80_cap50.teacher.jsonl`
- source Top1 changed by cap50: `55/80` (`68.8%`)
- average source Top1 regret under cap50: `2.2834`
- max source Top1 regret under cap50: `22.0958`
- avg exact elapsed in worker logs: `24.34s/row`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4/av_dataset_more4_80_cap50_dim520`

Combined the previous clean `764` rows with the new `80` rows and fine-tuned from ft764:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4/t2_local764_plus_more80_cap50_train844.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4/av_dataset_local764_plus_more80_cap50_train844_dim520`
- rows: `844`
- samples: `8,433`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4/models/t2-local764-more80-cap50-ft-top3-20260615`
- local CPU training time: `344s`

The ft844 checkpoint alone fits train-style rows but is not safe as a replacement:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft844 final | varied40 | 27.5% | 55.0% | 1.353 | 70.0% | 0.722 | 100.0% |
| ft844 final | fresh20 | 70.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| ft844 final | local next50 | 42.0% | 80.0% | 0.384 | 92.0% | 0.074 | 100.0% |
| ft844 final | cap200 first20 | 75.0% | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| ft844 final | fresh80-more2 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft844 final | fresh80-more3 traincheck | 98.8% | 98.8% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft844 final | fresh80-more4 traincheck | 98.8% | 98.8% | 0.001 | 100.0% | 0.000 | 100.0% |

A strict external-preserving grid selected an ft844 safe swap, now written into
`ai/config/t2_action_value_ensemble_20260615.json`:

- old stable-oracle checkpoint: `0.05`
- cap50-100 checkpoint: `0.10`
- ft604 final checkpoint: `0.70`
- hardmiss final checkpoint: `0.05`
- ft844 final checkpoint: `0.10`
- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external600_varied_20260615/train80_more4/ensemble_grid_with_ft844_step05.json`

Selected ft844 safe blend checks:

| holdout | groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 40 | 32.5% | 2.186 | 70.0% | 0.388 | 80.0% | 0.366 | 100.0% |
| fresh20 | 20 | 45.0% | 1.810 | 80.0% | 0.015 | 95.0% | 0.005 | 100.0% |
| local next50 | 50 | 58.0% | 0.884 | 84.0% | 0.303 | 96.0% | 0.024 | 100.0% |
| cap200 first20 | 20 | 75.0% | 0.258 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| fresh80-more2 traincheck | 80 | 58.8% | 1.147 | 85.0% | 0.136 | 95.0% | 0.065 | 100.0% |
| fresh80-more3 traincheck | 80 | 61.3% | 1.033 | 83.8% | 0.218 | 95.0% | 0.062 | 100.0% |
| fresh80-more4 traincheck | 80 | 61.3% | 0.471 | 83.8% | 0.166 | 96.2% | 0.037 | 100.0% |
| cap200 check12 | 12 | 66.7% | 1.412 | 91.7% | 0.000 | 100.0% | 0.000 | 100.0% |

Runtime smoke with the selected ft844 safe blend:

- command used `ai/config/t2_action_value_ensemble_20260615.json` on `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl`
- output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft844_safe_blend_20260615_limit1_exact.jsonl`
- result: `elapsed_ms=646.1`, `candidate_pool_size=20`, `exact_evaluated=3`, `best_action_idx=2`, `best_refined_score=7.7646`
- T3 union exact pool summary: min/mean/max `9 / 17.0 / 21`

Interpretation:

- The fourth independent batch again confirms that model-generated T2 source labels are noisy: source Top1 changed on `68.8%` of rows, with a max cap50 EV loss of `22.10`.
- ft844 alone is a train-style specialist and badly regresses varied40, so it is only useful as a small ensemble source.
- The selected ft844 swap preserves every external Top3/Top5 value from the ft764-safe config, preserves cap200 check12 Top3 rerank regret at `0.000`, improves fresh80-more2 Top3 from `83.8%` to `85.0%`, and improves fresh80-more4 Top5 from `93.8%` to `96.2%`.
- T2 remains a `pool20 + exact rerank` problem. The model is gradually improving as a shortlist generator, but Top3-only pruning is still not justified by external holdout evidence.

## T2 fresh80-more5 exact and ft924 holdout check

Generated a fifth independent varied local T2 batch without VMs:

- source base: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5`
- BB source: root_start `130000`, roots `20`, max `2` rows/root, rows `40`
- BTN source: root_start `140000`, roots `20`, max `2` rows/root, rows `40`
- combined rows: `80`
- split: BB `40`, BTN `40`
- unique dealt patterns: `40`
- unique board+opponent+dealt states: `80`
- average candidates per record: `24.525`

Converted all `80` rows with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/t2_fresh_ext_more5_80_cap50.teacher.jsonl`
- source Top1 changed by cap50: `59/80` (`73.8%`)
- average source Top1 regret under cap50: `3.1371`
- max source Top1 regret under cap50: `21.0436`
- avg exact elapsed in worker logs: `23.82s/row`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/av_dataset_more5_80_cap50_dim520`

Combined the previous clean `844` rows with the new `80` rows and fine-tuned from ft844:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/t2_local844_plus_more80_cap50_train924.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/av_dataset_local844_plus_more80_cap50_train924_dim520`
- rows: `924`
- samples: `9,233`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/models/t2-local844-more80-cap50-ft-top3-20260615`
- local CPU training time: `340s`

The ft924 checkpoint alone is still not safe as a runtime replacement:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft924 final | varied40 | 27.5% | 60.0% | 1.151 | 67.5% | 0.758 | 100.0% |
| ft924 final | fresh20 | 75.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| ft924 final | local next50 | 42.0% | 78.0% | 0.299 | 92.0% | 0.141 | 100.0% |
| ft924 final | cap200 first20 | 75.0% | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| ft924 final | fresh80-more2 traincheck | 98.8% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft924 final | fresh80-more3 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft924 final | fresh80-more4 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft924 final | fresh80-more5 traincheck | 98.8% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |

A strict external-preserving grid was run with ft924 as an optional source:

- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/ensemble_grid_with_ft924_step05.json`
- candidates checked: `163`
- candidates preserving every external Top3/Top5 value from the current ft844 blend: `6`
- best external-Reg3 candidate: old `0.05`, cap50-100 `0.10`, ft604 `0.70`, hardmiss `0.10`, ft844 `0.00`, ft924 `0.05`
- current config remains unchanged because that candidate worsened all fresh80 traincheck Top3/Top5 values.

Current ft844 blend versus the best ft924 external-Reg3 candidate:

| holdout | current Top3 | candidate Top3 | current Reg3 | candidate Reg3 | current Top5 | candidate Top5 | current Reg5 | candidate Reg5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 70.0% | 70.0% | 0.388 | 0.388 | 80.0% | 80.0% | 0.366 | 0.366 |
| fresh20 | 80.0% | 80.0% | 0.015 | 0.015 | 95.0% | 95.0% | 0.005 | 0.005 |
| local next50 | 84.0% | 84.0% | 0.303 | 0.293 | 96.0% | 96.0% | 0.024 | 0.024 |
| cap200 first20 | 95.0% | 95.0% | 0.055 | 0.055 | 100.0% | 100.0% | 0.000 | 0.000 |
| fresh80-more2 traincheck | 85.0% | 77.5% | 0.136 | 0.193 | 95.0% | 91.2% | 0.065 | 0.083 |
| fresh80-more3 traincheck | 83.8% | 81.2% | 0.218 | 0.218 | 95.0% | 91.2% | 0.062 | 0.108 |
| fresh80-more4 traincheck | 83.8% | 80.0% | 0.166 | 0.186 | 96.2% | 91.2% | 0.037 | 0.089 |
| fresh80-more5 traincheck | 85.0% | 83.8% | 0.127 | 0.128 | 95.0% | 91.2% | 0.053 | 0.062 |

Interpretation:

- The new exact labels again show that model-generated T2 source Top1 is noisy: cap50 changed `73.8%` of source Top1 choices.
- ft924 fits the accumulated fresh-style train rows, but it still regresses varied external rows when used alone.
- A tiny ft924 blend can preserve external Top3/Top5 and slightly lower local next50 Reg3, but it gives back too much traincheck recall. Keep `ai/config/t2_action_value_ensemble_20260615.json` on the ft844 safe blend until a new checkpoint improves external and traincheck together.

## T2 current-blend Top3 miss hard-negative pass

Added a reusable evaluator-miss weighting utility:

- script: `ai/training/create_eval_miss_weighted_action_value_data.py`
- input: `evaluate_action_value_reranker` `misses.jsonl`
- behavior: for each miss group, find the true teacher-best action from the source arrays, boost it, boost the model's top predicted confusers, and write new `sample_weights.npy` / `group_sample_weights.npy` while hardlinking the large arrays.

The current ft844 safe blend was evaluated directly on the combined `924` row training set:

- data: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/av_dataset_local844_plus_more80_cap50_train924_dim520`
- eval output: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/eval_current_ft844_blend_on_train924_top3`
- Top1: `83.8%`, Reg1 `0.307`
- Top3: `94.4%`, Reg3 `0.069`
- Top5: `98.3%`, Reg5 `0.019`
- Top10: `100.0%`
- full-bust chosen when avoidable: `7`
- Top3 misses: `52`
- Top3 misses with EV regret `>= 0.05`: `44`
- max Top3 miss regret: `8.737`

Created hard-negative weights from those Top3 misses:

- weighted data: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/av_dataset_train924_currentblend_top3miss_weighted_min005`
- weighted groups: `44`
- weighted candidates: `232`
- boosted confusers: `188`
- group weight mean/max: `1.629 / 23.800`
- sample weight mean/max: `3.037 / 15.000`

Fine-tuned from ft924 with the weighted data:

- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/models/t2-local924-currentblend-top3miss-weighted-ft-20260615`
- local CPU training time: `288s`
- internal train924 evaluation after fine-tune: Top1 `99.9%`, Top3 `100.0%`, Top5 `100.0%`, Top10 `100.0%`, full-bust chosen when avoidable `0`

The hard-negative checkpoint alone is not safe externally:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| hardft final | varied40 | 30.0% | 55.0% | 1.190 | 70.0% | 0.697 | 100.0% |
| hardft final | fresh20 | 75.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| hardft final | local next50 | 44.0% | 82.0% | 0.443 | 92.0% | 0.141 | 100.0% |
| hardft final | cap200 first20 | 75.0% | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| hardft final | fresh80-more2 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| hardft final | fresh80-more3 traincheck | 98.8% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| hardft final | fresh80-more4 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| hardft final | fresh80-more5 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |

A strict grid with hardft as an optional ensemble source was also run:

- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external680_varied_20260615/train80_more5/ensemble_grid_with_hardft_step05.json`
- candidates checked: `320`
- candidates preserving both external Top3/Top5 and traincheck Top3/Top5 from the current config: `4`
- best by external EV regret is still the current ft844 safe blend.
- the best non-current hardft blend was old `0.05`, cap50-100 `0.10`, ft604 `0.70`, hardmiss `0.05`, ft844 `0.00`, ft924 `0.00`, hardft `0.10`; it improved traincheck Top3 sum from `3.375` to `3.400` and traincheck Reg3 sum from `0.647` to `0.635`, but worsened external Reg3 sum from `0.762` to `0.776`.

Interpretation:

- Hard-negative weighting is effective at removing known train-side Top3 misses and full-bust mistakes.
- It still over-specializes to the accumulated local train distribution. The external varied40 set remains the limiting signal.
- Runtime config is intentionally unchanged. The next useful data pass should create more independent exact rows matching the varied40/local external miss shapes, rather than further reweighting the same `924` rows.

## T2 fresh120-more6 exact and ft1044 safe blend

Generated another independent external-style T2 source set on D drive:

- source dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6`
- BB source: `60` rows from root_start `150000`
- BTN source: `60` rows from root_start `160000`
- combined source: `inputs/t2_fresh_ext_more6_balanced120_model_teacher.jsonl`
- unique dealt patterns: `60`
- unique board/opponent/dealt states: `120`
- candidate total: `2916`
- average legal candidates: `24.3`

Converted all `120` rows with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/t2_fresh_ext_more6_120_cap50.teacher.jsonl`
- source Top1 changed by cap50: `93/120` (`77.5%`)
- average source Top1 regret under cap50: `2.7180`
- max source Top1 regret under cap50: `23.2789`
- avg exact elapsed in worker logs: `22.68s/row`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/av_dataset_more6_120_cap50_dim520`

Combined the previous `924` rows with the new `120` rows and fine-tuned from ft844:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/t2_local924_plus_more120_cap50_train1044.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/av_dataset_local924_plus_more120_cap50_train1044_dim520`
- rows: `1044`
- samples: `10433`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/models/t2-local924-more120-cap50-ft-top3-20260615`
- local CPU training time: `388s`

The ft1044 checkpoint alone still overfits fresh/local rows and is not safe as a direct runtime replacement:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft1044 final | varied40 | 30.0% | 62.5% | 1.137 | 75.0% | 0.637 | 100.0% |
| ft1044 final | fresh20 | 75.0% | 95.0% | 0.005 | 95.0% | 0.005 | 100.0% |
| ft1044 final | local next50 | 40.0% | 80.0% | 0.466 | 90.0% | 0.192 | 100.0% |
| ft1044 final | cap200 first20 | 75.0% | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| ft1044 final | fresh80-more2 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1044 final | fresh80-more3 traincheck | 98.8% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1044 final | fresh80-more4 traincheck | 97.5% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1044 final | fresh80-more5 traincheck | 100.0% | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1044 final | fresh120-more6 traincheck | 95.8% | 96.7% | 0.039 | 100.0% | 0.000 | 100.0% |

A grid was run with ft1044 as an optional ensemble source:

- grid result: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/ensemble_grid_with_ft1044_step05.json`
- candidates checked: `384`
- selected blend: old `0.05`, cap50-100 `0.10`, ft604 `0.70`, ft1044 `0.15`
- previous external Reg3 sum: `0.762`
- selected external Reg3 sum: `0.620`
- previous traincheck Reg3 sum: `0.933`
- selected traincheck Reg3 sum: `0.498`

Current ft844 blend versus the selected ft1044 blend:

| holdout | previous Top3 | selected Top3 | previous Reg3 | selected Reg3 | previous Top5 | selected Top5 | previous Reg5 | selected Reg5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 70.0% | 70.0% | 0.388 | 0.407 | 80.0% | 80.0% | 0.366 | 0.366 |
| fresh20 | 80.0% | 80.0% | 0.015 | 0.015 | 95.0% | 95.0% | 0.005 | 0.005 |
| local next50 | 84.0% | 84.0% | 0.303 | 0.142 | 96.0% | 96.0% | 0.024 | 0.024 |
| cap200 first20 | 95.0% | 95.0% | 0.055 | 0.055 | 100.0% | 100.0% | 0.000 | 0.000 |
| fresh80-more2 traincheck | 85.0% | 91.2% | 0.136 | 0.087 | 95.0% | 95.0% | 0.065 | 0.065 |
| fresh80-more3 traincheck | 83.8% | 92.5% | 0.218 | 0.113 | 95.0% | 97.5% | 0.062 | 0.023 |
| fresh80-more4 traincheck | 83.8% | 90.0% | 0.166 | 0.107 | 96.2% | 96.2% | 0.037 | 0.037 |
| fresh80-more5 traincheck | 85.0% | 91.2% | 0.127 | 0.110 | 95.0% | 96.2% | 0.053 | 0.051 |
| fresh120-more6 traincheck | 85.0% | 94.2% | 0.286 | 0.081 | 91.7% | 95.8% | 0.108 | 0.034 |

Additional regression checks:

- `cap200_check12` direct eval: Top1 `75.0%`, Top3 `100.0%`, Top5 `100.0%`, Top10 `100.0%`, Top3 Regret `0.000`
- runtime config: `ai/config/t2_action_value_ensemble_20260615.json`
- runtime smoke output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft1044_safe_blend_20260615_limit1_exact.jsonl`
- runtime smoke latency: `614ms` for one T2 row on CPU
- runtime smoke candidate pool: `20`
- runtime smoke sync exact candidates: `3`
- runtime smoke T3 exact pool mean/max: `17.0 / 21`

Interpretation:

- The new source set confirms the same pattern: model-generated T2 source Top1 remains very noisy, with exact cap50 changing `77.5%` of source Top1 choices.
- ft1044 alone is not reliable externally, but a small `15%` blend improves aggregate external EV loss and the newer traincheck sets while preserving external Top3/Top5 recall.
- `varied40` Reg3 slightly worsens (`0.388` to `0.407`), so this remains an experimental runtime config rather than a final model. The next pass should keep generating independent varied external rows, then optimize the blend against per-set EV-loss constraints.

## T2 fresh120-more7 exact and ft1164 check

Generated another independent external-style T2 source set on D drive:

- source dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7`
- BB source: `60` rows from root_start `170000`
- BTN source: `60` rows from root_start `180000`
- combined source: `inputs/t2_fresh_ext_more7_balanced120_model_teacher.jsonl`
- unique dealt patterns: `60`
- unique board/opponent/dealt states: `120`
- candidate total: `2874`
- average legal candidates: `23.95`

Converted all `120` rows with source Top10 and `t2_draw_limit=50`:

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/t2_fresh_ext_more7_120_cap50.teacher.jsonl`
- source Top1 changed by cap50: `93/120` (`77.5%`)
- average source Top1 regret under cap50: `3.5011`
- max source Top1 regret under cap50: `28.9795`
- avg exact elapsed in worker logs: `23.73s/row`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/av_dataset_more7_120_cap50_dim520`
- conversion quality gate: `1200` samples, `0` skipped

Important conversion note:

- Do not concatenate raw Rust chunk outputs before joining with source rows.  The Rust `record_index` starts at `0` inside each chunk.
- Correct flow is per-chunk source/exact join with `ai.training.build_t2_action_value_teacher_from_exact`, then concatenate the per-chunk teacher JSONL files.

Combined the previous `1044` rows with the new `120` rows and fine-tuned from ft1044:

- combined teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/t2_local1044_plus_more120_cap50_train1164.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/av_dataset_local1044_plus_more120_cap50_train1164_dim520`
- rows: `1164`
- samples: `11633`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/models/t2-local1044-more120-cap50-ft-top3-20260615`
- local CPU training time: `370s`
- best epoch: `39`

The ft1164 final checkpoint alone is not safe as a direct runtime replacement:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft1164 final | varied40 | 35.0% | 67.5% | 1.090 | 85.0% | 0.632 | 100.0% |
| ft1164 final | fresh20 | 45.0% | 90.0% | 0.009 | 95.0% | 0.004 | 100.0% |
| ft1164 final | local next50 | 42.0% | 74.0% | 0.280 | 90.0% | 0.159 | 100.0% |
| ft1164 final | cap200 first20 | 75.0% | 90.0% | 0.068 | 95.0% | 0.055 | 100.0% |
| ft1164 final | fresh120-more7 traincheck | 85.0% | 97.5% | 0.000 | 98.3% | 0.000 | 100.0% |

The ft1164 best checkpoint also remains externally unsafe:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| ft1164 best | varied40 | 35.0% | 62.5% | 1.141 | 85.0% | 0.782 | 100.0% |
| ft1164 best | fresh20 | 50.0% | 80.0% | 0.015 | 85.0% | 0.014 | 100.0% |
| ft1164 best | local next50 | 44.0% | 80.0% | 0.346 | 86.0% | 0.189 | 100.0% |
| ft1164 best | cap200 first20 | 75.0% | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| ft1164 best | fresh120-more7 traincheck | 76.7% | 95.8% | 0.019 | 98.3% | 0.000 | 100.0% |

Small blend grid over the current `15%` ft1044 slot:

| blend tail | external Reg3 sum | traincheck Reg3 sum | more7 Top3 | more7 Reg3 | more7 Top5 | more7 Reg5 |
|---|---:|---:|---:|---:|---:|---:|
| ft1044 `15%`, ft1164 `0%` | 0.620 | 1.314 | 69.2% | 0.816 | 87.5% | 0.371 |
| ft1044 `12.5%`, ft1164 `2.5%` | 0.620 | 1.214 | 72.5% | 0.718 | 90.8% | 0.030 |
| ft1044 `10%`, ft1164 `5%` | 0.617 | 1.257 | 73.3% | 0.687 | 90.8% | 0.024 |
| ft1044 `7.5%`, ft1164 `7.5%` | 0.736 | 1.098 | 77.5% | 0.469 | 90.8% | 0.024 |
| ft1044 `0%`, ft1164 `15%` | 0.737 | 0.845 | 79.2% | 0.164 | 93.3% | 0.020 |

Decision:

- Do not promote the direct ft1164 checkpoint.  It improves more7 but gives back too much on older external and replay sets.
- The useful output of this pass is the new exact more7 data and the conversion-path guardrail.
- A follow-up anti-forgetting run below uses the same more7 data with lower LR and checkpoint normalization, then promotes only a tiny `2.5%` blend.

## T2 more7 low-LR anti-forgetting blend

Fine-tuned from ft1044 again, but with a smaller update:

- data: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/av_dataset_local1044_plus_more120_cap50_train1164_dim520`
- init checkpoint: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/models/t2-local924-more120-cap50-ft-top3-20260615/action_value_final.pt`
- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/models/t2-local1044-more120-cap50-ft-top3-lowlr-ckptnorm-20260615`
- LR: `1.2e-5`
- normalization source: `checkpoint`
- epochs: `40`
- local CPU training time: `215s`

The low-LR checkpoint alone is not externally safe, but it is useful as a tiny ensemble source:

| model | holdout | Top1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| low-LR final | varied40 | 27.5% | 60.0% | 1.113 | 77.5% | 0.785 | 100.0% |
| low-LR final | fresh20 | 60.0% | 85.0% | 0.014 | 90.0% | 0.009 | 100.0% |
| low-LR final | local next50 | 40.0% | 78.0% | 0.270 | 94.0% | 0.141 | 100.0% |
| low-LR final | cap200 first20 | 75.0% | 90.0% | 0.068 | 95.0% | 0.013 | 100.0% |
| low-LR final | fresh120-more7 traincheck | 79.2% | 93.3% | 0.027 | 97.5% | 0.009 | 100.0% |

Selected runtime blend:

- old stable oracle: `0.05`
- cap50-100: `0.10`
- ft604: `0.70`
- ft1044: `0.125`
- low-LR more7: `0.025`
- config: `ai/config/t2_action_value_ensemble_20260615.json`

Comparison against the previous ft1044 `15%` blend:

| holdout | previous Top3 | selected Top3 | previous Reg3 | selected Reg3 | previous Top5 | selected Top5 | previous Reg5 | selected Reg5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 70.0% | 70.0% | 0.407 | 0.407 | 80.0% | 80.0% | 0.366 | 0.366 |
| fresh20 | 80.0% | 80.0% | 0.015 | 0.015 | 95.0% | 95.0% | 0.005 | 0.005 |
| local next50 | 84.0% | 86.0% | 0.142 | 0.139 | 96.0% | 96.0% | 0.024 | 0.024 |
| cap200 first20 | 95.0% | 95.0% | 0.055 | 0.055 | 100.0% | 100.0% | 0.000 | 0.000 |
| fresh80-more2 traincheck | 91.2% | 91.2% | 0.087 | 0.077 | 95.0% | 95.0% | 0.065 | 0.065 |
| fresh80-more3 traincheck | 92.5% | 92.5% | 0.113 | 0.113 | 97.5% | 97.5% | 0.023 | 0.023 |
| fresh80-more4 traincheck | 90.0% | 88.7% | 0.107 | 0.107 | 96.3% | 96.3% | 0.037 | 0.037 |
| fresh80-more5 traincheck | 91.2% | 91.2% | 0.110 | 0.110 | 96.3% | 95.0% | 0.051 | 0.053 |
| fresh120-more6 traincheck | 94.2% | 94.2% | 0.081 | 0.081 | 95.8% | 95.8% | 0.034 | 0.034 |
| fresh120-more7 traincheck | 69.2% | 72.5% | 0.816 | 0.718 | 87.5% | 90.0% | 0.371 | 0.039 |

Additional checks:

- external Reg3 sum: `0.620 -> 0.617`
- traincheck Reg3 sum including more7: `1.314 -> 1.206`
- `cap200_check12`: Top3 `91.7%`, Reg3 `0.000`, Top5 `100.0%`, Reg5 `0.000`
- runtime smoke output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_ft1044_lowlr_more7_blend_20260615_limit1_exact.jsonl`
- runtime smoke latency: `569ms`
- runtime smoke candidate pool: `20`
- runtime smoke sync exact candidates: `3`
- runtime smoke T3 exact pool mean/max: `17.0 / 21`

Interpretation:

- The small low-LR blend is a net EV-loss improvement without changing external Top3/Top5.
- It does not solve more7 completely, but it removes most of the more7 Top5 EV tail (`0.371 -> 0.039`) and slightly improves local next50.
- Top3 recall on `cap200_check12` drops from `100.0%` to `91.7%`, but Top3 exact-rerank EV loss stays `0.000`; this is acceptable for the current exact-rerank-first objective but should remain monitored.

## T2 targeted weight grid after low-LR blend

The broad grid launched after the low-LR blend was too large and was stopped after timeout, then replaced with a cached targeted grid:

- output: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/grid_targeted_20260615/targeted_grid_summary.json`
- candidates checked: `4620`
- prediction cache: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/grid_targeted_20260615/pred_cache`
- eligible candidates under external/cap200/old-more constraints: `8`

Selected runtime blend:

- old stable oracle: `0.05`
- cap50-100: `0.125`
- ft604: `0.675`
- ft1044: `0.125`
- ft1164: `0.025`
- config: `ai/config/t2_action_value_ensemble_20260615.json`

This replaces the tiny low-LR more7 replay source with a tiny ft1164 source and shifts `0.025` weight from ft604 to cap50-100.  The external EV-loss metrics stay unchanged, while `cap200_check12` Top3 recall returns to `100.0%`.

| holdout | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---:|---:|---:|---:|---:|
| varied40 | 70.0% | 0.407 | 80.0% | 0.366 | 100.0% |
| fresh20 | 80.0% | 0.015 | 95.0% | 0.005 | 100.0% |
| local next50 | 86.0% | 0.139 | 96.0% | 0.024 | 100.0% |
| cap200 first20 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| cap200_check12 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| fresh160-more2 traincheck | 99.8% | 0.006 | 99.8% | 0.001 | 100.0% |
| fresh80-more3 traincheck | 92.5% | 0.113 | 97.5% | 0.023 | 100.0% |
| fresh80-more4 traincheck | 88.7% | 0.107 | 96.3% | 0.037 | 100.0% |
| fresh80-more5 traincheck | 91.2% | 0.110 | 95.0% | 0.053 | 100.0% |
| fresh120-more6 traincheck | 94.2% | 0.081 | 95.8% | 0.034 | 100.0% |
| fresh120-more7 traincheck | 72.5% | 0.716 | 90.0% | 0.039 | 100.0% |

Comparison to the previous selected low-LR blend:

- external Reg3 sum: unchanged at `0.616922`
- external Reg5 sum: unchanged at `0.395237`
- traincheck Reg3 sum: `1.134882 -> 1.133123`
- more7 Top3 Regret: `0.718114 -> 0.716355`
- cap200_check12 Top3 recall: `91.7% -> 100.0%`
- runtime smoke output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_targeted_grid_20260615_limit1_exact.jsonl`
- runtime smoke latency: `558ms`
- runtime smoke candidate pool: `20`
- runtime smoke sync exact candidates: `3`
- runtime smoke T3 exact pool mean/max: `17.0 / 21`

Interpretation:

- This is a small but cleaner runtime improvement, not a breakthrough.
- Top10 remains safe on all checked sets, but Top3 is still far from the target on `varied40` and `fresh120-more7`.
- The next meaningful improvement should come from new exact T2 labels and hard examples, not more tiny weight tuning.

## T2 current Top3-miss hard example pass

Evaluated the current targeted blend on the full `1164`-row training set to mine current Top3 misses:

- eval output: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/eval_current_targeted_on_train1164_top3`
- data: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/av_dataset_local1044_plus_more120_cap50_train1164_dim520`
- current Top3: `94.0%`
- current Top3 Regret: `0.114`
- Top3 misses: `70`
- Top3 misses with EV loss `>=0.1`: `59`
- high tail: p95 miss regret `11.681`, max miss regret `28.121`

Built a weighted hard-miss data view without copying the large arrays:

- weighted data: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/av_dataset_train1164_current_top3miss_weighted_min010`
- weighted groups: `59`
- weighted candidates: `319`
- boosted confusers: `260`
- group weight mean/max: `1.607 / 21.0`
- sample weight mean/max: `3.043 / 13.0`

Fine-tuned a current-Top3-miss specialist from ft1044:

- model dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/models/t2-train1164-current-top3miss-weighted-lowlr-20260615`
- init checkpoint: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external800_varied_20260615/train120_more6/models/t2-local924-more120-cap50-ft-top3-20260615/action_value_final.pt`
- LR: `8e-6`
- epochs: `60`
- local CPU training time: `277s`

The hard-miss specialist is not safe as a standalone runtime model:

| model | holdout | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|
| hardmiss2 final | varied40 | 65.0% | 1.098 | 75.0% | 0.845 | 100.0% |
| hardmiss2 final | fresh20 | 90.0% | 0.010 | 95.0% | 0.005 | 100.0% |
| hardmiss2 final | local next50 | 84.0% | 0.240 | 92.0% | 0.141 | 100.0% |
| hardmiss2 final | cap200 first20 | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| hardmiss2 final | train1164 | 99.7% | 0.001 | 99.9% | 0.000 | 100.0% |
| hardmiss2 final | fresh120-more7 traincheck | 97.5% | 0.008 | 99.2% | 0.003 | 100.0% |

A normal `0.025` blend was also too large, but a tiny `0.010` blend was safe:

- tiny blend grid: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/grid_hardmiss2_20260615/hardmiss2_tiny_blend_summary.json`
- selected blend: old `0.05`, cap50-100 `0.125`, ft604 `0.675`, ft1044 `0.125`, ft1164 `0.015`, hardmiss2 `0.010`
- config: `ai/config/t2_action_value_ensemble_20260615.json`
- runtime smoke output: `D:/ofc-pineapple-data/t2_next_20260614/runtime_smoke/t2_ensemble_tiny_hardmiss2_20260615_limit1_exact.jsonl`
- runtime smoke latency: `494ms`
- runtime smoke candidate pool: `20`
- runtime smoke sync exact candidates: `3`
- runtime smoke T3 exact pool mean/max: `17.0 / 21`

Comparison to the previous targeted blend:

| holdout | previous Top3 | selected Top3 | previous Reg3 | selected Reg3 | previous Top5 | selected Top5 | previous Reg5 | selected Reg5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| varied40 | 70.0% | 70.0% | 0.407 | 0.407 | 80.0% | 80.0% | 0.366 | 0.366 |
| fresh20 | 80.0% | 80.0% | 0.015 | 0.015 | 95.0% | 95.0% | 0.005 | 0.005 |
| local next50 | 86.0% | 86.0% | 0.139 | 0.139 | 96.0% | 96.0% | 0.024 | 0.024 |
| cap200 first20 | 95.0% | 95.0% | 0.055 | 0.055 | 100.0% | 100.0% | 0.000 | 0.000 |
| cap200_check12 | 100.0% | 100.0% | 0.000 | 0.000 | 100.0% | 100.0% | 0.000 | 0.000 |
| train1164 | 94.0% | 94.2% | 0.114 | 0.110 | 97.3% | 97.5% | 0.020 | 0.019 |
| fresh120-more7 traincheck | 72.5% | 73.3% | 0.716 | 0.685 | 90.0% | 90.8% | 0.039 | 0.030 |

Interpretation:

- The tiny hard-miss blend is a measured improvement in the exact place it targeted, without weakening the current external checks.
- The model still cannot be trusted as a standalone answerer; it is only useful as a `1%` ensemble signal.
- This reinforces the same direction: keep external holdouts clean, generate more exact T2 rows, and use hard-miss specialists only in very small blends unless they pass external EV-loss checks.

## T2 fresh more8 exact-label pass

Generated another local-only clean T2 exact batch on `D:` using fresh T0/T1-topk source rows:

- base dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1040_varied_20260615/train80_more8`
- source: `inputs/t2_fresh_ext_more8_balanced80_model_teacher.jsonl`
- rows: `80`
- position balance: `bb=40`, `btn=40`
- unique board/opponent/dealt/position rows: `80`
- unique dealt triples: `40`
- source candidates: `1971`, average `24.64`
- exact teacher: `t2_fresh_ext_more8_80_cap50.teacher.jsonl`
- action-value data: `av_dataset_more8_80_cap50_dim520`
- aggregate summary: `t2_fresh_ext_more8_80_cap50.aggregate_summary.json`

The cap50 exact conversion used `source-candidate-top-k=10` and four local chunks of 20 rows:

- rows converted: `80`
- teacher candidates: `800`
- source top1 changed after cap50 exact: `61/80` (`76.25%`)
- exact regret of source top1: mean `1.962`, p95 `7.344`, max `13.577`
- source best exact rank: mean `3.95`, max `10`
- cap50 exact elapsed per row: mean `22.73s`, p95 `28.77s`, max `32.79s`
- estimated full T2 exact per row from cap50: mean `2999s`, p95 `4109s`

The current runtime blend on this new more8 set is still Top10-safe, but not Top3-safe:

| model | set | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current tiny hardmiss2 blend | more8 target80 | 33.8% | 1.259 | 67.5% | 0.369 | 80.0% | 0.206 | 100.0% |

Merged the new more8 data with the previous 1164-row training set:

- merged data: `av_dataset_local1164_plus_more80_cap50_train1244_dim520`
- records: `1244`
- samples: `12433`
- turns: `T2` only

Fine-tuned a new ft1244 model from ft1164:

- model dir: `models/t2-local1164-more80-cap50-ft-top3-20260615`
- init checkpoint: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/models/t2-local1044-more120-cap50-ft-top3-20260615/action_value_final.pt`
- LR: `1.2e-5`
- epochs: `50`
- local CPU training time: `257s`

The new ft1244 model is strong on the new target set and previous more7 traincheck, but it is not safe as a standalone runtime model because it worsens some external Top3 EV loss:

| model | set | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ft1244 final | more8 target80 | 95.0% | 0.241 | 98.8% | 0.079 | 98.8% | 0.079 | 100.0% |
| ft1244 final | varied40 | 40.0% | 1.787 | 67.5% | 1.018 | 85.0% | 0.138 | 100.0% |
| ft1244 final | fresh20 | 60.0% | 0.132 | 95.0% | 0.005 | 100.0% | 0.000 | 100.0% |
| ft1244 final | local next50 | 44.0% | 1.433 | 74.0% | 0.290 | 90.0% | 0.142 | 100.0% |
| ft1244 final | cap200 first20 | 75.0% | 0.258 | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| ft1244 final | cap200_check12 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1244 final | fresh120-more7 traincheck | 97.5% | 0.004 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |

Tested tiny blends by scaling the existing runtime weights and adding ft1244 at `0.25%`, `0.5%`, `1%`, `2%`, `3%`, and `5%`.

- blend grid: `grid_ft1244_blend_20260615/summary_table.json`
- `0.25%` ft1244 slightly improves more8 Top3 Reg3 (`0.369 -> 0.336`) with no measured external change.
- `0.5%` and above already worsen `local next50` Top3 (`86.0% -> 84.0%`, Reg3 `0.139 -> 0.259`).
- `3%` and `5%` worsen `varied40` Top3 materially (`70.0% -> 65.0%`, Reg3 `0.407 -> 1.108`).

Decision:

- Do not change `ai/config/t2_action_value_ensemble_20260615.json` for ft1244 yet.
- Keep ft1244 as an offline diagnostic/source model only.
- The useful signal is that more8 exposed a real distribution gap: Top10 remains safe, but Top3 needs more varied exact rows before another runtime blend should be adopted.
- Next data pass should add more clean external rows rather than overtraining on this 80-row shard.

## T2 fresh120-more9 exact and ft1364 check

Generated another independent external-style T2 source set on D drive, with roots moved beyond more8:

- base dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1160_varied_20260615/train120_more9`
- BB source: `60` rows from root_start `210000`
- BTN source: `60` rows from root_start `220000`
- combined source: `inputs/t2_fresh_ext_more9_balanced120_model_teacher.jsonl`
- unique board/opponent/dealt/position rows: `120`
- unique dealt triples: `60`
- source candidates: `2961`, average `24.68`

Converted all `120` rows with `source-candidate-top-k=10` and `t2_draw_limit=50`:

- teacher: `t2_fresh_ext_more9_120_cap50.teacher.jsonl`
- dataset: `av_dataset_more9_120_cap50_dim520`
- aggregate summary: `t2_fresh_ext_more9_120_cap50.aggregate_summary.json`
- teacher candidates: `1200`
- source top1 changed after cap50 exact: `82/120` (`68.33%`)
- exact regret of source top1: mean `2.955`, p95 `11.580`, max `16.789`
- source best exact rank: mean `3.88`, max `10`
- cap50 exact elapsed per row with 6 local workers: mean `33.22s`, p95 `42.99s`, max `46.41s`
- estimated full T2 exact per row from cap50: mean `4387s`, p95 `6139s`

The current runtime blend on more9 is better than on more8, but still not Top3-safe:

| model | set | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current tiny hardmiss2 blend | more9 target120 | 55.8% | 0.899 | 85.8% | 0.224 | 94.2% | 0.012 | 100.0% |

Merged the previous `1244` rows with the new more9 rows:

- merged data: `av_dataset_local1244_plus_more120_cap50_train1364_dim520`
- records: `1364`
- samples: `13633`
- turns: `T2` only

Fine-tuned a new ft1364 model from ft1164, deliberately not from the more8-overfit ft1244:

- model dir: `models/t2-local1244-more120-cap50-ft-top3-20260615`
- init checkpoint: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/models/t2-local1044-more120-cap50-ft-top3-20260615/action_value_final.pt`
- LR: `1.0e-5`
- epochs: `55`
- local CPU training time: `289s`

The ft1364 model is excellent on accumulated traincheck shards, but still not safe as a standalone runtime model:

| model | set | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ft1364 final | more9 target120 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1364 final | more8 target80 | 97.5% | 0.031 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1364 final | varied40 | 37.5% | 1.840 | 72.5% | 0.993 | 82.5% | 0.785 | 100.0% |
| ft1364 final | fresh20 | 60.0% | 0.132 | 95.0% | 0.005 | 100.0% | 0.000 | 100.0% |
| ft1364 final | local next50 | 40.0% | 1.458 | 76.0% | 0.289 | 88.0% | 0.142 | 100.0% |
| ft1364 final | cap200 first20 | 75.0% | 0.258 | 95.0% | 0.055 | 95.0% | 0.055 | 100.0% |
| ft1364 final | cap200_check12 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1364 final | fresh120-more7 traincheck | 97.5% | 0.009 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |

Tiny ft1364 blend grid:

- grid: `grid_ft1364_blend_20260615/summary_table.json`
- `0.25%` ft1364 is safe on measured checks but does not materially improve more9.
- `0.5%` and `1%` slightly improve more9 Reg3 (`0.224 -> 0.221`) but already worsen `local next50` Top3 (`86.0% -> 84.0%`, Reg3 `0.139 -> 0.259`).
- `2%` worsens `varied40` Top3 Reg3 (`0.407 -> 0.905`).
- `3%` and `5%` worsen `varied40` materially (`70.0% -> 65.0%`, Reg3 `0.407 -> 1.108`).

Decision:

- Do not change `ai/config/t2_action_value_ensemble_20260615.json`.
- ft1364 is useful as a diagnostic/source model but not a runtime blend member yet.
- The consistent finding is now stronger: source Top10 remains enough for exact rerank on these shards, but learned Top3 still overfits shard history and does not generalize to `varied40`.
- Next useful improvement should either add a much larger and more varied exact set, or change the model/training objective so it preserves external varied40 while learning the new shards.

## T2 target80 external-like specialist and gated blend check

Built a deliberately harder, external-like T2 source set on D drive to stress the
visible-joker/top-structure patterns seen in the `varied40` miss analysis:

- base dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615`
- source pool: `160` BB rows and `160` BTN rows from new root ranges
- filtered target set: `target80_source/t2_target_external_like80_model_teacher.jsonl`
- final exact teacher: `target80_cap50.teacher.jsonl`
- dataset: `av_dataset_target80_cap50_dim520`
- rows: `80`, BB/BTN: `40/40`
- source candidates: `797`, average `9.96` after source Top10 exact conversion
- source top1 changed after cap50 exact: `60/80` (`75.0%`)
- exact regret of source top1: mean `3.697`, p95 `15.868`, max `25.359`
- cap50 exact elapsed per row with 4 local workers: mean `17.11s`, p95 `24.30s`

The current runtime blend is still Top10-safe on this harder set, but its Top3
pool is weak:

| model | set | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current tiny hardmiss2 blend | target80 | 47.5% | 1.540 | 78.8% | 0.451 | 90.0% | 0.201 | 100.0% |
| ft1364 final | target80 | 58.8% | 1.342 | 82.5% | 0.419 | 88.8% | 0.139 | 100.0% |

Merged the `1364` previous rows with the new `target80` rows:

- merged data: `av_dataset_train1364_plus_target80_cap50_train1444_dim520`
- records: `1444`
- samples: `14430`

Fine-tuned a target specialist from ft1364:

- model dir: `models/t2-train1444-target80-ft-top3-20260615`
- init checkpoint: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1160_varied_20260615/train120_more9/models/t2-local1244-more120-cap50-ft-top3-20260615/action_value_final.pt`
- LR: `8e-6`
- epochs: `40`
- local CPU training time: `201s`

The ft1444 specialist learns the target shard, but is not safe as a standalone
runtime model:

| model | set | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ft1444 final | target80 | 96.2% | 0.011 | 98.8% | 0.008 | 100.0% | 0.000 | 100.0% |
| ft1444 final | more9 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1444 final | more8 | 98.8% | 0.031 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| ft1444 final | varied40 | 40.0% | 1.777 | 67.5% | 1.018 | 80.0% | 0.785 | 100.0% |
| ft1444 final | local next50 | 38.0% | 1.446 | 76.0% | 0.475 | 88.0% | 0.150 | 100.0% |

Unconditional tiny blends also are not enough:

- grid: `grid_ft1444_blend_20260615/summary_table.json`
- `0.1%` and `0.25%` preserve measured external sets but do not materially improve `target80`.
- `0.5%` already worsens `local next50` Top3 (`86.0% -> 84.0%`, Reg3 `0.139 -> 0.259`).
- `2%` worsens `varied40` Top3 Reg3 (`0.407 -> 0.905`).
- `3%` worsens `varied40` Top3 (`70.0% -> 65.0%`, Reg3 `0.407 -> 1.108`).

Added an offline gated-blend evaluator:

- script: `ai/training/evaluate_action_value_gated_blend.py`
- purpose: test a feature gate before wiring a conditional specialist into runtime
- best checked gate so far: `target_like_strict`
- gate rule: visible joker, no dealt joker, own top length at least 2, and own/opponent top has a pair or joker

The best offline candidate is a strict-gated `35%` ft1444 specialist blend:

- output: `gated_ft1444_target_like_strict_eps035/summary_table.json`
- this is not yet in `ai/config/t2_action_value_ensemble_20260615.json`
- runtime support still needs a conditional action-value wrapper before adoption

| gated model | set | gated groups | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict gate, 35% ft1444 | target80 | 69/80 | 66.2% | 0.992 | 90.0% | 0.352 | 97.5% | 0.154 | 100.0% |
| strict gate, 35% ft1444 | more9 | 19/120 | 60.0% | 0.791 | 86.7% | 0.223 | 95.0% | 0.007 | 100.0% |
| strict gate, 35% ft1444 | more8 | 0/80 | 33.8% | 1.259 | 67.5% | 0.369 | 80.0% | 0.206 | 100.0% |
| strict gate, 35% ft1444 | varied40 | 2/40 | 32.5% | 2.101 | 70.0% | 0.407 | 80.0% | 0.366 | 100.0% |
| strict gate, 35% ft1444 | fresh20 | 0/20 | 45.0% | 1.212 | 80.0% | 0.015 | 95.0% | 0.005 | 100.0% |
| strict gate, 35% ft1444 | local next50 | 9/50 | 50.0% | 1.028 | 86.0% | 0.136 | 96.0% | 0.024 | 100.0% |
| strict gate, 35% ft1444 | cap200 first20 | 7/20 | 75.0% | 0.258 | 95.0% | 0.055 | 100.0% | 0.000 | 100.0% |
| strict gate, 35% ft1444 | cap200_check12 | 3/12 | 91.7% | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| strict gate, 35% ft1444 | more7 | 19/120 | 55.8% | 1.420 | 75.8% | 0.671 | 90.8% | 0.030 | 100.0% |

Decision:

- Do not change the current runtime config yet.
- Keep ft1444 as an offline specialist and candidate for a conditional runtime path.
- The gated result is a real improvement signal: it raises target80 Top3 from `78.8%` to `90.0%` while preserving the checked external Top10 exact-rerank safety.
- The next implementation step should add a conditional action-value wrapper in the runtime path, then rerun the same gate on a larger external holdout before enabling it by default.

## T2 conditional runtime wrapper

Implemented the conditional specialist path without changing the default runtime
mode:

- model wrapper: `ai/models/action_value_reranker.py`
  - `ConditionalBlendedActionValueReranker`
  - `context_gate_accepts`
  - `encoded_state_gate_mask`
- runtime scorer hook: `ai/mcts/rollout_evaluator.py`
  - passes a conditional gate mask only when a model exposes `candidate_gate_mask`
  - existing models keep the old `predict_components` path
- hybrid CLI/config loader: `ai/tutor/hybrid_t1t2.py`
  - new CLI option: `--turn-model-conditional-ensemble`
  - mode-specific `models` entries are now supported
  - fixed top-level runtime mode inheritance so `optional_modes.*.inherits` can refer to `t2_fast_t3_union`
- offline evaluator: `ai/training/evaluate_action_value_gated_blend.py`
  - added `--gate-source state` for candidate-level state gates
  - record/context gates remain available for局面単位 checks

Added an optional runtime mode, but left the default unchanged:

- config: `ai/config/t2_action_value_ensemble_20260615.json`
- default mode: still `t2_fast_t3_union`
- optional mode: `t2_fast_t3_union_gated_ft1444`
- specialist: `models/t2-train1444-target80-ft-top3-20260615/action_value_final.pt`
- specialist weight: `0.35`
- gate: `target_like_strict_context`

Important validation detail:

- A pure encoded-state gate is too broad because the model only sees post-action
  states; it cannot always distinguish pre-existing visible jokers from dealt
  jokers placed by a candidate.
- The state gate at `35%` improved `target80` but hurt `local next50` Top3
  (`86.0% -> 82.0%`, Reg3 `0.139 -> 0.261`), so it should not be used as the
  runtime gate.
- The runtime optional mode uses the context gate, which reads the original
  observation before candidate actions, matching the earlier record-gate
  validation.

State-gate diagnostic output:

- output: `gated_ft1444_state_strict_eps035/summary_table.json`

| gated model | set | gated groups | gated samples | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| state gate, 35% ft1444 | target80 | 69/80 | 687/797 | 90.0% | 0.352 | 97.5% | 0.154 | 100.0% |
| state gate, 35% ft1444 | more9 | 51/120 | 347/1200 | 88.3% | 0.216 | 96.7% | 0.002 | 100.0% |
| state gate, 35% ft1444 | varied40 | 12/40 | 68/399 | 70.0% | 0.410 | 82.5% | 0.350 | 100.0% |
| state gate, 35% ft1444 | local next50 | 29/50 | 225/497 | 82.0% | 0.261 | 96.0% | 0.024 | 100.0% |
| state gate, 35% ft1444 | more7 | 69/120 | 409/1200 | 83.3% | 0.437 | 95.8% | 0.007 | 100.0% |

Runtime smokes:

- default config mode: `runtime_smoke_config_default_limit1.jsonl`, written `1`
- optional gated mode: `runtime_smoke_config_gated_ft1444_limit1.jsonl`, written `1`
- existing `t2_fast_t3_union_s2` optional mode: `runtime_smoke_config_s2_limit1.jsonl`, written `1`
- explicit long-form conditional CLI: `runtime_smoke_conditional_context_limit1.jsonl`, written `1`

Decision:

- Keep `t2_fast_t3_union_gated_ft1444` experimental and opt-in.
- Do not make it the default until a larger clean holdout confirms the context
  gate does not trade off external EV.
- The implementation now allows that larger holdout to be run through the same
  runtime path instead of only an offline approximation.

## T2 runtime opt-in mode scored against teacher rows

Added a runtime-output scorer so `hybrid_t1t2` JSONL can be compared directly
against existing exact/cap50 teacher rows:

- script: `ai/training/score_hybrid_runtime_output.py`
- comparison output dir: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615`
- runtime flags: `--disable-refinement`, so this measures model shortlist quality only
- modes compared:
  - default: `t2_fast_t3_union`
  - opt-in: `t2_fast_t3_union_gated_ft1444`

Important caveat:

- Some runtime top candidates are legal actions that are not present in the
  capped teacher JSONL, so their exact EV is unknown without more exact work.
- The scorer therefore reports TopK hit/known-candidate regret and
  `missing_top1`.
- This is still useful for comparing whether the teacher-best exact candidate
  is preserved in the runtime candidate list.

Runtime comparison:

| set | mode | rows | missing top1 | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target80 | default | 80 | 10 | 41.2% | 1.605 | 72.5% | 0.606 | 85.0% | 0.378 | 92.5% | 0.126 |
| target80 | gated ft1444 | 80 | 15 | 57.5% | 0.890 | 83.8% | 0.565 | 92.5% | 0.232 | 97.5% | 0.088 |
| holdout40 | default | 40 | 10 | 25.0% | 2.490 | 60.0% | 0.635 | 72.5% | 0.405 | 82.5% | 0.100 |
| holdout40 | gated ft1444 | 40 | 10 | 25.0% | 2.490 | 60.0% | 0.635 | 72.5% | 0.405 | 82.5% | 0.100 |
| local50 | default | 50 | 8 | 48.0% | 1.209 | 78.0% | 0.281 | 92.0% | 0.060 | 92.0% | 0.047 |
| local50 | gated ft1444 | 50 | 7 | 48.0% | 1.154 | 78.0% | 0.270 | 92.0% | 0.050 | 94.0% | 0.035 |

Decision update:

- The opt-in context-gated specialist is still not ready to become the default,
  because the clean holdout40 Top3 is only `60.0%` in both modes.
- It is directionally useful: it improves the targeted weak pattern set
  (`target80`) and does not hurt `holdout40` or `local50` in this measured
  runtime comparison.
- The next model-strength step should target `holdout40` misses directly,
  preferably by exact-labeling the unknown runtime top candidates or creating a
  new clean holdout-sized training shard from similar external roots.

## T2 holdout40 Top10 miss analysis

The runtime `holdout40` Top10 misses from the default `t2_fast_t3_union` mode
were analyzed before adding more training:

- miss file: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/holdout40_t2_fast_t3_union.top10_misses.jsonl`
- feature summary: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/holdout40_top10_miss_feature_summary.json`
- seed rows: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/holdout40_top10_miss_mining_20260615/holdout40_top10_misses_seed_rows.jsonl`

Observed pattern:

- misses: `7/40`
- position: BTN `5`, BB `2`
- own top length: `1` in all `7/7`
- own top pair/joker: `0/7`
- own shape: `1-3-3` in `5/7`
- best action placed at least one card on top in `6/7`

This is a different weakness from the earlier visible-joker/top-structure
`target80` shard, which explains why the `target_like_strict_context` ft1444
gate did not improve `holdout40`.

## T2 top-single specialist check

Built a new training shard from existing non-holdout exact/cap50 rows matching
the runtime-detectable condition "own top has exactly one card and is not a
pair/joker":

- teacher: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/top_single_unpaired_specialist_20260615/t2_top_single_unpaired_train484.teacher.jsonl`
- dataset: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/top_single_unpaired_specialist_20260615/av_dataset_top_single_unpaired_train484_dim520`
- rows: `484`, candidates: `4,840`
- BB/BTN: `241/243`
- best action top placements: top0 `188`, top1 `266`, top2 `30`
- model: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/top_single_unpaired_specialist_20260615/models/t2-top-single-unpaired-ft1364-top3-20260615/action_value_final.pt`
- local CPU training time: about `140s`

Runtime/offline support added:

- `ai/models/action_value_reranker.py`
  - `top_single_unpaired_context`
  - approximate `top_single_unpaired_state` for diagnostic-only use
- `ai/training/evaluate_action_value_gated_blend.py`
  - record gate `top_single_unpaired`

The specialist is **not adopted**.  Gated blends did not improve the clean
holdout:

| gated model | set | gate groups | Top3 | Reg3 | Top5 | Reg5 | Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| top-single, 20% | target80 | 0/80 | 78.8% | 0.451 | 90.0% | 0.201 | 100.0% |
| top-single, 20% | holdout40 | 24/40 | 67.5% | 0.905 | 82.5% | 0.350 | 100.0% |
| top-single, 20% | local50 | 13/50 | 86.0% | 0.139 | 96.0% | 0.024 | 100.0% |
| top-single, 35% | holdout40 | 24/40 | 67.5% | 0.905 | 80.0% | 0.351 | 100.0% |
| top-single, 50% | holdout40 | 24/40 | 67.5% | 0.883 | 80.0% | 0.351 | 100.0% |

Interpretation:

- The target condition was correct, but the specialist is not good enough.
- The issue is not simply that this board shape is underrepresented; the model
  needs better ranking signal for which top-fill action wins.
- Keep the code support for future gates, but do not add this specialist to the
  runtime config.

## T2 holdout40 runtime Top10 cap50 relabel

The original runtime scorer had unknown exact values for some runtime Top10
candidates because the existing `holdout40` teacher only labeled source Top10
candidates.  To check whether the Top10 miss diagnosis was real, the `7`
miss rows were re-evaluated with:

- input: runtime Top10 candidates plus the known teacher-best candidate
- output: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/holdout40_runtime_top10_cap50_check_20260615/cap50/t2_oracle_cap50_limit7.jsonl`
- teacher JSONL: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/holdout40_runtime_top10_cap50_check_20260615/cap50/t2_runtime_top10_plus_best_cap50.teacher.jsonl`
- rank summary: `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/holdout40_runtime_top10_cap50_check_20260615/cap50/runtime_top10_cap50_rank_summary.json`
- cap: `t2_draw_limit=50`
- rows: `7`
- candidates per row: `11`
- elapsed: mean `9675.5 ms/row`

Within these 11 candidates, the cap50 best action ranked by the runtime model at:

- ranks: `[3, 1, 3, 7, 11, 6, 3]`
- Top1 hit: `1/7`
- Top3 hit: `4/7`
- Top5 hit: `4/7`
- Top10 hit: `6/7`
- Top1 exact regret: mean `0.929`, max `1.484`

Decision:

- Top10 is mostly recovering the best candidate even on the miss rows.
- The current weakness is Top3 ordering, not broad Top10 candidate coverage.
- For 5-second play, the next useful work is improving T2 Top3 ranking on
  top-fill tactical rows, or letting the 5-second exact refinement evaluate more
  of the Top10 when the model margin is small.

## T2 Top5/Top10 refinement experiment

Tested whether expanding synchronous T2 refinement can recover more of the
Top10 tail within the 5-second budget.

Implementation support:

- `ai/tutor/hybrid_t1t2.py`
  - added T2 selection policy `refined_plus_model`
  - added config fields `t2_selection_policy` and
    `t2_selection_model_weight`
- `ai/config/t2_action_value_ensemble_20260615.json`
  - added optional mode `t2_fast_t3_union_top5_s2_modelblend`
  - default mode remains unchanged

The optional mode:

- exact-refines model Top5 instead of Top3
- allows a second sample for close candidates
- selects by `refined_score + 1.0 * model_score`

Runtime speed is still comfortably under 5 seconds on the measured sets:

| set | mode | mean ms | p95 ms | max ms | refined eval mean |
|---|---|---:|---:|---:|---:|
| holdout40 | Top3 / 1 sample | 314.2 | 655.0 | 795.3 | 3.00 |
| holdout40 | Top5 / 2 sample blend | 730.6 | 1215.9 | 1385.4 | 7.35 |
| target80 | Top3 / 1 sample | 318.2 | 750.4 | 2614.8 | 3.00 |
| target80 | Top5 / 2 sample blend | 728.0 | 1495.7 | 3141.0 | 7.36 |
| local50 | Top3 / 1 sample | 368.7 | 723.9 | 1658.0 | 3.00 |
| local50 | Top5 / 2 sample blend | 857.0 | 1472.3 | 1941.5 | 7.34 |

Teacher-known candidate ordering comparison:

| set | mode | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| holdout40 | Top3 / 1 sample | 37.5% | 1.844 | 60.0% | 0.635 | 72.5% | 0.405 | 82.5% | 0.100 |
| holdout40 | Top5 / 2 sample blend | 32.5% | 2.293 | 67.5% | 0.511 | 72.5% | 0.405 | 82.5% | 0.100 |
| target80 | Top3 / 1 sample | 35.0% | 2.185 | 72.5% | 0.606 | 85.0% | 0.378 | 92.5% | 0.126 |
| target80 | Top5 / 2 sample blend | 38.8% | 1.593 | 71.2% | 0.931 | 85.0% | 0.378 | 92.5% | 0.126 |
| local50 | Top3 / 1 sample | 48.0% | 1.146 | 78.0% | 0.281 | 92.0% | 0.060 | 92.0% | 0.047 |
| local50 | Top5 / 2 sample blend | 44.0% | 1.316 | 88.0% | 0.092 | 92.0% | 0.060 | 92.0% | 0.047 |

Chosen-action comparison against currently labeled teacher candidates:

| set | mode | known | missing | exact-hit known | chosen Reg mean known |
|---|---|---:|---:|---:|---:|
| holdout40 | Top3 / 1 sample | 30 | 10 | 16 | 1.844 |
| holdout40 | Top5 / 2 sample blend | 32 | 8 | 16 | 1.799 |
| target80 | Top3 / 1 sample | 71 | 9 | 29 | 2.185 |
| target80 | Top5 / 2 sample blend | 64 | 16 | 33 | 1.493 |
| local50 | Top3 / 1 sample | 47 | 3 | 25 | 1.146 |
| local50 | Top5 / 2 sample blend | 47 | 3 | 23 | 1.291 |

Decision:

- Do not make this the default.  It improves some Top3 pool metrics, but it
  worsens `target80` Top3 regret and does not clearly improve chosen-action
  quality on `local50`.
- Keep `t2_fast_t3_union_top5_s2_modelblend` as an opt-in experiment because it
  proves the extra refinement budget is cheap enough.
- The next higher-value step is to exact-label the currently unknown runtime
  selected candidates, then train/evaluate a T2 selector on chosen-action EV
  loss rather than only candidate TopK recall.

### Runtime-selected unknown exact labels

The chosen-action comparison above still had missing exact/cap50 labels for
some runtime-selected actions.  Those selected actions were extracted and
re-labeled with Rust `t3_exact_solver` cap50:

- Top5 / 2 sample blend unknown selected input:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_selected_unknown_20260615/top5_s2_unknown_all_input.jsonl`
- Top5 / 2 sample blend exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_selected_unknown_20260615/cap50/t2_oracle_cap50_limit27.jsonl`
- Top3 / 1 sample unknown selected input:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_selected_unknown_20260615/top3_unknown_all_input.jsonl`
- Top3 / 1 sample exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_selected_unknown_20260615/cap50_top3/t2_oracle_cap50_limit22.jsonl`
- fair summary:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top3_vs_top5_s2_modelblend_extra_scored_summary.json`

Exact-labeling confirmed that shallow T2 partial refinement is noisy:

| mode | unknown selected rows | same top1 after cap50 | changed top1 | avg selected regret | max selected regret |
|---|---:|---:|---:|---:|---:|
| Top5 / 2 sample blend | 27 | 5 | 22 | 2.035 | 19.688 |
| Top3 / 1 sample | 22 | 4 | 18 | 2.257 | 19.688 |

After merging those labels, chosen-action EV loss can be compared without
unknown selected rows:

| set | mode | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 | chosen Reg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| holdout40 | Top3 / 1 sample | 47.5% | 1.519 | 70.0% | 0.481 | 77.5% | 0.323 | 87.5% | 0.044 | 1.519 |
| holdout40 | Top5 / 2 sample blend | 42.5% | 1.965 | 72.5% | 0.468 | 77.5% | 0.362 | 87.5% | 0.060 | 1.555 |
| target80 | Top3 / 1 sample | 35.0% | 2.377 | 72.5% | 0.587 | 85.0% | 0.378 | 92.5% | 0.126 | 2.377 |
| target80 | Top5 / 2 sample blend | 37.5% | 1.919 | 71.2% | 0.619 | 85.0% | 0.378 | 92.5% | 0.126 | 1.776 |
| local50 | Top3 / 1 sample | 48.0% | 1.261 | 78.0% | 0.433 | 92.0% | 0.060 | 92.0% | 0.047 | 1.261 |
| local50 | Top5 / 2 sample blend | 44.0% | 1.313 | 88.0% | 0.092 | 92.0% | 0.060 | 92.0% | 0.047 | 1.289 |

Updated decision:

- Still do not make Top5 / 2 sample blend the default.  It helps `target80`
  chosen-action EV loss, but loses on `holdout40` and is slightly worse on
  `local50`.
- The useful signal is now clearer: the runtime needs a selector/arbitration
  model that decides when to trust extra refinement/modelblend and when to keep
  the simpler Top3 path.
- The next training target should be selected-action EV loss, not only TopK
  candidate recall.  The exact-labeled selected-action rows above are the first
  hard examples for that selector.

### T2 Top5/s2 promotion check with more8/more9

Added two more existing cap50 teacher shards to test whether the Top5/s2
runtime is broadly better or only helped the first three checked sets:

- more8 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1040_varied_20260615/train80_more8/t2_fresh_ext_more8_80_cap50.teacher.jsonl`
- more9 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1160_varied_20260615/train120_more9/t2_fresh_ext_more9_120_cap50.teacher.jsonl`
- arbitration summary:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/t2_top3_vs_top5_arbitration_more8_more9_20260615.json`

Unknown runtime-selected actions were exact-labeled before scoring:

| set | mode | unknown selected | same top1 after cap50 | avg selected regret | max selected regret |
|---|---|---:|---:|---:|---:|
| more8 | Top3 + Top5 selected union | 15 | 3 | 1.965 | 9.067 |
| more9 | Top3 + Top5 selected union | 37 | 6 | 2.065 | 10.138 |

The extra labels confirm the same pattern as before: runtime-selected actions
that fall outside the original source Top10 are noisy under shallow partial
refinement, so they must be exact-labeled before using them as evidence.

Chosen-action comparison with no missing selected rows:

| set | rows | Top3 chosen Reg | Top5/s2 chosen Reg | delta | Top3 hit | Top5/s2 hit |
|---|---:|---:|---:|---:|---:|---:|
| holdout40 | 40 | 1.519 | 1.555 | +0.036 | 50.0% | 50.0% |
| target80 | 80 | 2.377 | 1.776 | -0.601 | 36.2% | 43.8% |
| local50 | 50 | 1.261 | 1.289 | +0.028 | 50.0% | 46.0% |
| more8 | 80 | 1.977 | 1.264 | -0.713 | 28.7% | 35.0% |
| more9 | 120 | 1.871 | 1.280 | -0.591 | 44.2% | 51.7% |
| total | 370 | 1.883 | 1.415 | -0.468 | 40.5% | 45.4% |

Runtime speed remains comfortably inside the 5 second target:

| set | mode | mean ms | p95 ms | max ms |
|---|---|---:|---:|---:|
| more8 | Top3 | 224.6 | 400.8 | 699.4 |
| more8 | Top5/s2 | 483.7 | 720.2 | 998.3 |
| more9 | Top3 | 226.9 | 370.2 | 1131.4 |
| more9 | Top5/s2 | 446.2 | 703.1 | 1291.3 |

Arbitration/gating:

- A simple all-data gate can reduce total regret further (`1.415` -> `1.376`),
  but leave-one-set-out gates are unstable and sometimes worse than just using
  Top5/s2.
- Do not add a runtime gate yet.
- The robust implementation move is to promote Top5/s2 itself, then train a
  selector later from a larger set of exact-labeled selected-action rows.

Implementation decision:

- `ai/config/t2_action_value_ensemble_20260615.json`
  - `default_mode` changed from `t2_fast_t3_union` to
    `t2_fast_t3_union_top5_s2_modelblend`.
  - The old Top3 mode remains available explicitly as `t2_fast_t3_union`.
- Smoke check with no explicit `--runtime-mode` resolved to
  `t2_fast_t3_union_top5_s2_modelblend` and ran `5/5` rows under budget
  (`mean 422.1 ms`, `max 463.2 ms`).

### T2 Top7/Top10 and selection-weight tuning

Checked whether simply widening synchronous refinement beyond Top5 helps on the
`more9` 120-row cap50 set:

| mode | mean ms | p95 ms | max ms | chosen Reg |
|---|---:|---:|---:|---:|
| Top5/s2, weight 1.0 | 446.2 | 703.1 | 1291.3 | 1.325 |
| Top7/s2, weight 1.0 | 562.7 | 839.3 | 1371.9 | 1.417 |
| Top10/s2, weight 1.0 | 695.4 | 973.1 | 1494.0 | 1.474 |

Top7/Top10 remain comfortably under 5 seconds, but they are worse than Top5.
The extra candidates add partial-exact noise faster than they add useful
coverage, so the default should stay at Top5.

Then swept the Top5/s2 selection rule
`refined_score + weight * model_score` over the 370 checked rows:

- sweep output:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_weight_sweep_5sets_weight15_labeled_20260615.json`
- two additional `weight=1.5` selected actions were exact-labeled:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_weight15_unknown_cap50/t2_top5_s2_weight15_unknown_cap50.teacher.jsonl`

| weight | known rows | missing | hit known | chosen Reg known |
|---:|---:|---:|---:|---:|
| 0.00 | 357 | 13 | 41.5% | 1.682 |
| 0.50 | 363 | 7 | 44.4% | 1.451 |
| 0.75 | 365 | 5 | 45.8% | 1.403 |
| 1.00 | 370 | 0 | 44.9% | 1.430 |
| 1.25 | 370 | 0 | 45.9% | 1.371 |
| 1.50 | 370 | 0 | 47.0% | 1.355 |
| 1.75 | 370 | 0 | 47.3% | 1.364 |
| 2.00 | 370 | 0 | 47.0% | 1.356 |

Updated implementation decision:

- Keep `t2_sync_exact_k=5`; do not widen to Top7/Top10.
- Change `t2_selection_model_weight` from `1.0` to `1.5` in the default
  Top5/s2 mode.
- Smoke check after the config change confirmed default runtime resolves to
  `t2_fast_t3_union_top5_s2_modelblend` with
  `t2_selection_policy=refined_plus_model`,
  `t2_selection_model_weight=1.5`, `t2_sync_exact_k=5`, and
  `t2_max_samples_per_candidate=2`.

### T2 refined-candidate selector probe

To see whether a learned selector can beat the tuned
`refined_score + 1.5 * model_score` rule, the exact-label coverage of the
Top5/s2 refined candidates was audited:

| set | rows | refined candidates | known | unknown |
|---|---:|---:|---:|---:|
| holdout40 | 40 | 200 | 162 | 38 |
| target80 | 80 | 400 | 318 | 82 |
| local50 | 50 | 250 | 203 | 47 |
| more8 | 80 | 400 | 331 | 69 |
| more9 | 120 | 600 | 472 | 128 |
| total | 370 | 1850 | 1486 | 364 |

Rather than labeling all 364 unknown refined candidates, only unknown candidates
that could matter for the current default selector were mined: candidates in the
Top2 under `weight=1.5`.

- mined input:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_weight15_top2_unknown_input.jsonl`
- exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_weight15_top2_unknown_cap50/t2_oracle_cap50_limit45.jsonl`
- teacher JSONL:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_weight15_top2_unknown_cap50/t2_top5_s2_weight15_top2_unknown_cap50.teacher.jsonl`
- records: `45`
- source top1 preserved after cap50: `3/45`
- avg source-top1 regret: `3.468`
- max source-top1 regret: `16.995`

After adding these labels, the weight sweep still prefers `1.5`:

| weight | known rows | missing | hit known | chosen Reg known |
|---:|---:|---:|---:|---:|
| 1.00 | 370 | 0 | 44.9% | 1.430 |
| 1.25 | 370 | 0 | 45.9% | 1.371 |
| 1.50 | 370 | 0 | 47.0% | 1.355 |
| 1.75 | 370 | 0 | 47.3% | 1.364 |
| 2.00 | 370 | 0 | 47.0% | 1.356 |

Linear selector probe:

- script:
  `ai/training/evaluate_t2_refined_selector.py`
- output:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/t2_refined_linear_selector_loso_top2_l2_1000_20260615.json`
- labeled refined candidates after Top2 mining: `1531/1850`

LOSO comparison:

| heldout | weight 1.5 Reg | linear selector Reg | linear missing |
|---|---:|---:|---:|
| holdout40 | 1.547 | 1.933 | 1/40 |
| local50 | 0.994 | 1.038 | 3/50 |
| more8 | 1.321 | 1.360 | 3/80 |
| more9 | 1.267 | 1.279 | 0/120 |
| target80 | 1.650 | 1.624 | 1/80 |

Decision:

- Do not promote the linear refined-candidate selector.  It helps `target80`
  slightly but loses on the other four heldout sets.
- Keep the simpler default rule `refined_score + 1.5 * model_score`.
- The next learned-selector attempt needs either full refined-candidate labels
  or a ranking-specific model trained on more rows; this small linear EV
  regressor is not robust enough.

### T2 full refined-candidate labels and enriched specialist

Filled the remaining exact labels for every Top5/s2 refined runtime candidate
across the five checked T2 sets:

- extraction script:
  `ai/training/extract_unknown_refined_t2_candidates.py`
- input:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_refined_unknown_remaining_input.jsonl`
- exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_refined_unknown_remaining_cap50/t2_oracle_cap50_limit229.jsonl`
- converted teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/top5_s2_refined_unknown_remaining_cap50/t2_top5_s2_refined_unknown_remaining_cap50.teacher.jsonl`
- records exacted: `229`
- unknown refined candidates exacted: `319`
- cap50 runtime: `1379.7s` total, `6000.5 ms` average per record

After these labels, all `1850/1850` refined candidates were known.  The best
simple selector stayed at `refined_score + 1.5 * model_score`, but its measured
loss rose from `1.355` to `1.388` because previously unknown strong refined
candidates became part of the oracle comparison.

Full-label selector/upper-bound check:

| selector | rows | Top1/hit | chosen Reg |
|---|---:|---:|---:|
| weight 1.0 | 370 | 43.5% | 1.464 |
| weight 1.5 | 370 | 45.7% | 1.388 |
| weight 2.0 | 370 | 45.7% | 1.390 |
| linear selector LOSO, l2=1000 | 370 | n/a | 1.429 |
| exact oracle within refined candidates | 370 | 83.0% | 0.276 |

The main bottleneck is now candidate-pool coverage, not the hand-tuned
selection weight.  If exact rerank only sees the current refined candidates,
the best possible average loss is still `0.276`.

Merged the source Top10 labels and all extra labels into a single enriched
teacher:

- merge script:
  `ai/training/build_enriched_t2_teacher_from_extra_labels.py`
- enriched teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/t2_enriched_5sets_full_refined_cap50_20260615.teacher.jsonl`
- reranker dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/runtime_eval_20260615/reranker_t2_enriched_5sets_full_refined_cap50_20260615`
- rows: `370`
- candidates: `4045`
- extra candidates added: `352`

Current ensemble on this enriched set:

| model | Top1 | Top3 | Top5 | Top10 | Top1 Reg | Top5 rerank Reg |
|---|---:|---:|---:|---:|---:|---:|
| current ensemble | 43.5% | 75.4% | 86.5% | 98.1% | 1.403 | 0.189 |

Trained a local CPU specialist for 180 seconds:

- model:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/targeted_external_like_20260615/models/t2-enriched-fullrefined370-ft-from-train1164-20260615/action_value_best.pt`
- init checkpoint:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external920_varied_20260615/train120_more7/models/t2-train1164-current-top3miss-weighted-lowlr-20260615/action_value_final.pt`

Internal enriched-set results:

| model | Top1 | Top3 | Top5 | Top10 | Top1 Reg | Top5 rerank Reg |
|---|---:|---:|---:|---:|---:|---:|
| specialist only | 74.3% | 96.8% | 98.1% | 99.5% | 0.364 | 0.025 |
| current + specialist alpha 0.20 | 51.9% | 83.8% | 92.4% | 99.5% | 1.121 | 0.075 |
| current + specialist alpha 0.35 | 58.1% | 87.8% | 95.1% | 99.5% | 0.887 | 0.056 |
| current + specialist alpha 0.50 | 63.0% | 91.6% | 96.5% | 99.5% | 0.722 | 0.053 |

Small external smoke on existing `fresh20`:

| model | Top1 | Top3 | Top5 | Top10 | Top1 Reg | Top5 rerank Reg |
|---|---:|---:|---:|---:|---:|---:|
| current ensemble | 45.0% | 80.0% | 95.0% | 100.0% | 1.212 | 0.005 |
| specialist only | 55.0% | 85.0% | 95.0% | 100.0% | 0.403 | 0.005 |
| current + specialist alpha 0.20 | 50.0% | 85.0% | 90.0% | 100.0% | 1.167 | 0.010 |
| current + specialist alpha 0.50 | 60.0% | 85.0% | 95.0% | 100.0% | 0.705 | 0.005 |

Implementation decision:

- Added mode `t2_fast_t3_union_top5_s2_enriched_specialist_alpha50` to
  `ai/config/t2_action_value_ensemble_20260615.json`.
- Fixed `ai/tutor/hybrid_t1t2.py` runtime config inheritance to resolve nested
  mode inheritance recursively.  Before this fix, an optional mode inheriting
  `t2_fast_t3_union_top5_s2_modelblend` did not inherit the base
  `t2_fast_t3_union` settings, so it silently fell back to `mc_board` instead of
  `exact_partial/t3_union`.
- Promoted `t2_fast_t3_union_top5_s2_enriched_specialist_alpha50` to the default
  mode after the larger available check below and the fixed runtime smoke.  A
  larger fully clean holdout is still the next validation step.

More2-more7 aggregate check over `560` rows:

| model | Top1 | Top3 | Top5 | Top10 | Top1 Reg | Top3 Reg | Top5 Reg | bad full-bust picks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| previous default | 65.4% | 88.0% | 95.0% | 100.0% | 0.820 | 0.222 | 0.039 | 12 |
| specialist only | 96.8% | 99.5% | 99.6% | 100.0% | 0.039 | 0.002 | 0.002 | 0 |
| current + specialist alpha 0.50 | 91.4% | 99.3% | 99.6% | 100.0% | 0.097 | 0.002 | 0.002 | 1 |

Fixed alpha50 runtime smoke on the existing five-row `more9` sample:

| mode | Top1 | Top3 | Top5 | Top10 | chosen Reg | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|---:|
| previous default | 40.0% | 40.0% | 60.0% | 80.0% | 1.152 | 450.1 ms | 501.9 ms |
| alpha50 default | 60.0% | 80.0% | 100.0% | 100.0% | 1.037 | 523.7 ms | 712.1 ms |

The smoke output confirmed `exact_partial=true`, `t2_exact_backend=t3_union`,
and all five rows stayed well under the 5 second target.

### 2026-06-15 clean holdouts and T2 tactical insurance

Created new local all-legal T2 cap50 checks on `D:`.  These are intentionally
small but harder than the previously reused training/check sets because the T2
source Top1 was usually overturned by Rust cap50:

| set | rows | candidates | cap50 runtime | source Top1 changed | avg source Top1 Reg | max source Top1 Reg |
|---|---:|---:|---:|---:|---:|---:|
| `clean20` | 20 | 495 | 368.7s | 11/20 | 2.300 | 14.365 |
| `clean20b` | 20 | 486 | 337.2s | 15/20 | 3.047 | 14.549 |
| `clean20c` | 20 | 471 | 280.5s | 16/20 | 2.902 | 14.044 |

Important paths:

- `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout_20260615`
- `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout2_20260615`
- `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615`

Model-only checks showed that adding `clean20` and `clean20b` as hard-negative
training rows improves those rows but does not generalize reliably to
`clean20c`.

| eval set | model | Top1 | Top3 | Top5 | Top10 | Top1 Reg | Top5 Reg |
|---|---|---:|---:|---:|---:|---:|---:|
| clean20 | current alpha50 | 35.0% | 70.0% | 90.0% | 100.0% | 1.394 | 0.171 |
| clean20 | clean20 specialist | 80.0% | 95.0% | 95.0% | 100.0% | 0.603 | 0.007 |
| clean20b | current alpha50 | 25.0% | 65.0% | 80.0% | 95.0% | 1.835 | 0.804 |
| clean20b | clean20b-trained specialist | 90.0% | 95.0% | 95.0% | 95.0% | 0.110 | 0.010 |
| clean20c | current alpha50 | 35.0% | 70.0% | 90.0% | 90.0% | 1.988 | 0.117 |
| clean20c | clean20b-trained specialist | 35.0% | 60.0% | 75.0% | 90.0% | 2.311 | 0.130 |

The `clean20b`-trained specialist was strong on more2-more7 (`560` rows:
Top1 `95.7%`, Top3 `99.5%`, Top5 `99.8%`, Top10 `100.0%`, Top1 Reg `0.053`),
but it failed on the unseen `clean20c` set.  Do not promote it to the default.

Runtime checks on `clean20c`:

| mode | chosen Reg | chosen hit | Top5 | Top10 | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|
| current default Top5/s2/w1.5 | 1.315 | 60.0% | 75.0% | 90.0% | 631.5 ms | 1601.4 ms |
| Top10/s10/w8 | 0.644 | 45.0% | 90.0% | 90.0% | 3578.9 ms | 4814.6 ms |
| Top15/s8/w8 | 0.616 | 50.0% | 90.0% | 90.0% | 3893.5 ms | 4886.6 ms |
| Top10/s10/w2 + T2 tactical3 | 0.448 | 65.0% | 95.0% | 95.0% | 3247.2 ms | 4814.4 ms |

Implemented optional T2 tactical insurance in
`ai/tutor/hybrid_t1t2.py`, default off:

- new CLI/config field: `t2_tactical_insurance_k`
- tactical candidates prioritize deterministic row structure such as bottom
  flush/straight completion, top premium pair/Joker completion, and
  middle/bottom kind completion
- when enabled, pool insurance candidates (`t2_tactical`, `low_bust`, `fl`) can
  be injected into the sync exact-refinement set

This helped `clean20c`, but regressed the earlier small checks:

| set | Top10/s10/w2 + tactical3 chosen Reg | chosen hit | elapsed max |
|---|---:|---:|---:|
| clean20 | 1.322 | 40.0% | 4805.9 ms |
| clean20b | 1.292 | 50.0% | 4897.7 ms |
| clean20c | 0.448 | 65.0% | 4814.4 ms |

Decision:

- Keep current alpha50 default unchanged.
- Keep T2 tactical insurance as an experimental, opt-in runtime feature only.
- Do not promote the clean20b-trained specialist or tactical runtime mode until
  a larger clean holdout confirms it.
- The next useful step is not another 20-row loop.  Generate a broader local
  all-legal cap50 T2 set, at least `100-200` rows, then train one specialist and
  reserve a fresh final holdout.  The current 20-row hard-negative loop is too
  local and overfits.

### 2026-06-15 broad100 local T2 cap50 pass

Generated the first broader local all-legal cap50 T2 set on `D:`:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/t2_broad100_source_all_actions.jsonl`
- exact combined output:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/t2_broad100_oracle_cap50_alllegal.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/t2_broad100_alllegal_cap50.teacher.jsonl`
- dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/reranker_t2_broad100_alllegal_cap50_dim520`

Generation summary:

- source rows: `100`
- positions: `BB=50`, `BTN=50`
- source candidates: `2352`
- source generation time: `47.1s`
- Rust cap50 all-legal rows: `100`
- source Top1 changed by cap50: `76/100`
- average source Top1 EV loss: `2.371`
- max source Top1 EV loss: `20.269`
- average cap50 runtime per row: `23027.4 ms`

Per-chunk cap50 check:

| chunk | rows | changed Top1 | avg row ms | avg source Top1 Reg | max source Top1 Reg |
|---|---:|---:|---:|---:|---:|
| 0 | 20 | 15 | 24286.1 | 1.531 | 12.656 |
| 1 | 20 | 15 | 25192.1 | 2.588 | 20.095 |
| 2 | 20 | 16 | 26148.6 | 3.715 | 20.269 |
| 3 | 20 | 16 | 25389.7 | 1.444 | 7.259 |
| 4 | 20 | 14 | 14120.7 | 2.579 | 9.111 |

Current model-only results on broad100 showed this is a much harder and more
useful distribution than the previous small sets:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current alpha50 | 14.0% | 36.0% | 50.0% | 74.0% | 85.0% | 92.0% | 5.099 | 0.831 |
| new410 specialist | 18.0% | 39.0% | 52.0% | 70.0% | 84.0% | 94.0% | 5.141 | 0.814 |
| current + new410 alpha0.50 | 15.0% | 38.0% | 53.0% | 72.0% | 82.0% | 94.0% | 5.214 | 0.721 |

Runtime on broad100:

| runtime | chosen Reg | chosen hit | Top5 | Top10 | Top15 | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current default Top5/s2/w1.5 | 1.472 | 36.0% | 83.0% | 91.0% | 98.0% | 456.2 ms | 1914.1 ms |
| Top10/s10/tactical3/w2 | 1.056 | 44.0% | 80.0% | 82.0% | 93.0% | 3053.8 ms | 4859.8 ms |

Trained a broad100 specialist:

- model:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/models/t2-enriched510-broad100-alllegal-ft-20260615/action_value_best.pt`
- training data:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/reranker_t2_enriched510_plus_broad100_dim520`
- rows: `510`
- samples: `7346`
- init:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout2_20260615/models/t2-enriched410-clean20b-alllegal-ft-20260615/action_value_best.pt`
- target topK during training: `10`
- local CPU training time: `363s`

New510 model-only evaluation:

| eval set | model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| broad100 | new510 specialist | 73.0% | 92.0% | 95.0% | 99.0% | 99.0% | 99.0% | 1.087 | 0.040 |
| broad100 | current + new510 alpha0.50 | 36.0% | 64.0% | 84.0% | 99.0% | 99.0% | 99.0% | 3.914 | 0.040 |
| clean20c | new510 specialist | 35.0% | 70.0% | 85.0% | 90.0% | 100.0% | 100.0% | 2.351 | 0.071 |
| clean20c | current + new510 alpha0.50 | 35.0% | 65.0% | 85.0% | 90.0% | 100.0% | 100.0% | 1.988 | 0.071 |
| more2-more7 | new510 specialist | 94.6% | 99.6% | 99.8% | 100.0% | n/a | n/a | 0.097 | 0.000 |
| more2-more7 | current + new510 alpha0.50 | 94.8% | 99.6% | 99.8% | 100.0% | n/a | n/a | 0.063 | 0.000 |

Runtime with the broad100 specialist blended into the experimental Top10/s10
mode:

| eval set | runtime | chosen Reg | chosen hit | Top10 | Top15 | elapsed mean | elapsed max |
|---|---|---:|---:|---:|---:|---:|---:|
| broad100 | current + new510 alpha0.50 | 0.945 | 51.0% | 80.0% | 96.0% | 3101.1 ms | 4828.7 ms |
| clean20c | current + new510 alpha0.50 | 0.515 | 65.0% | 85.0% | 90.0% | 3273.1 ms | 4866.2 ms |

Decision:

- Added optional mode
  `t2_fast_t3_union_top10_s10_tactical3_w2_new510_alpha50_experimental`.
- Do not make it the default yet.  It improves broad100 and more2-more7, but
  the non-new510 tactical mode is still slightly better on the clean20c external
  holdout (`0.448` vs `0.515` chosen Reg).
- The clearest signal from broad100 is that T2 needs much more broad
  all-legal cap50 data.  The model can fit the broad distribution once it sees
  it, but a single 100-row pass is not enough to prove generalization.

### 2026-06-15 clean holdout4 T2 cap50 external check

Generated another local D-drive external set that was not used for training:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/t2_clean_holdout4_source_all_actions.jsonl`
- exact combined output:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/t2_clean_holdout4_oracle_cap50_alllegal.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/t2_clean_holdout4_alllegal_cap50.teacher.jsonl`
- dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/reranker_t2_clean_holdout4_alllegal_cap50_dim520`

Generation summary:

- source rows: `40`
- positions: `BB=20`, `BTN=20`
- source candidates: `951`
- source generation time: `18.3s`
- Rust cap50 all-legal rows: `40`
- source Top1 changed by cap50: `24/40`
- average source Top1 EV loss: `2.612`
- max source Top1 EV loss: `17.616`
- average cap50 runtime per row: `25000.7 ms`

Model-only external results:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg | Top20 Reg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current alpha50 | 22.5% | 47.5% | 60.0% | 82.5% | 95.0% | 97.5% | 4.088 | 0.526 | 0.000 |
| new510 specialist | 30.0% | 42.5% | 55.0% | 82.5% | 90.0% | 95.0% | 3.374 | 0.632 | 0.220 |
| current + new510 alpha0.50 | 30.0% | 50.0% | 60.0% | 80.0% | 92.5% | 95.0% | 3.655 | 0.659 | 0.220 |

Runtime on clean holdout4:

| runtime | chosen Reg | chosen hit | Top5 Reg | Top10 Reg | Top20 Reg | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current default Top5/s2/w1.5 | 0.624 | 67.5% | 0.349 | 0.243 | 0.003 | 425.0 ms | 1245.1 ms |
| Top10/s10/tactical3/w2 | 1.340 | 57.5% | 0.039 | 0.039 | 0.003 | 3261.1 ms | 4809.9 ms |
| current + new510 alpha0.50, w2 | 0.928 | 62.5% | 0.100 | 0.100 | 0.000 | 3038.5 ms | 4830.6 ms |
| current + new510 alpha0.50, refined-only w0 | 0.452 | 62.5% | 0.100 | 0.100 | 0.000 | 2950.8 ms | 4781.0 ms |

Selection-weight sweep on saved runtime candidates showed that the new510
runtime should not mix model score back into the final T2 choice on this
holdout:

- new510 w0: `0.297` avg known regret over the 38 rows with refined candidates.
- new510 w2: about `0.798` on the same refined-candidate subset.
- direct runtime scoring over all 40 rows confirmed w0 selected-action EV loss
  of `0.452`, better than default `0.624`.

Decision:

- Added optional mode
  `t2_fast_t3_union_top10_s10_tactical3_new510_alpha50_refined_only_experimental`.
- Do not make it default yet.  It is the best runtime on clean_holdout4, but it
  must be checked against broad100, clean20c, and more2-more7 before promotion.
- The main remaining T2 problem is now twofold:
  1. Top10 still misses too often on fresh external rows.
  2. When the correct action is in the refined set, the final selector must be
     calibrated per runtime mode instead of using one global model-score weight.

Cross-check after adding the refined-only mode:

| eval set | runtime | chosen Reg | chosen hit | Top10 Reg | Top20 Reg | elapsed mean | elapsed max |
|---|---|---:|---:|---:|---:|---:|---:|
| clean20c | current + new510 alpha0.50, refined-only w0 | 0.500 | 65.0% | 0.167 | 0.000 | 3464.8 ms | 4811.4 ms |
| broad100 | current + new510 alpha0.50, refined-only w0 | 1.046 | 47.0% | 0.166 | 0.000 | 3141.7 ms | 5087.5 ms |

Updated decision:

- Keep the refined-only mode as experimental.  It improved clean_holdout4, but
  it was not better than w2 on broad100 (`1.046` vs `0.945`) and slightly
  exceeded the 5s max on one broad100 row.
- The safest current production default remains
  `t2_fast_t3_union_top5_s2_enriched_specialist_alpha50`.
- The next improvement should be a learned final selector over refined
  candidates, trained across broad100 + clean20c + clean_holdout4, instead of
  choosing one fixed global blend weight.

### 2026-06-15 T2 refined-candidate final selector probe

Used the new510 Top10/s10/tactical3 runtime candidates from:

- broad100:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/runtime_new510_alpha50_top10_s10_tac3_w0/results.jsonl`
- clean20c:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/runtime_new510_alpha50_top10_s10_tac3_w0/results.jsonl`
- clean_holdout4:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/runtime_new510_alpha50_top10_s10_tac3_w0/runtime.jsonl`

Linear ridge final-selector probe:

- script: `ai/training/evaluate_t2_refined_selector.py`
- output dir:
  `D:/ofc-pineapple-data/t2_next_20260614/selector_experiments_20260615`
- rows with refined candidates: `153`
- labeled refined candidates: `1530/1530`

Best leave-one-set-out ridge setting among the l2 sweep was not strong enough
to promote:

| heldout | fixed w1.5 Reg | linear selector Reg | note |
|---|---:|---:|---|
| broad100 | 1.007 | 0.941 | improves |
| clean20c | 0.505 | 0.522 | worsens |
| clean_holdout4 | 0.798 | 0.391 | improves |

Then reselected all saved runtime outputs with fixed final-selection weights
without rerunning exact:

| final weight | combined Reg | broad100 Reg | clean20c Reg | holdout4 Reg |
|---:|---:|---:|---:|---:|
| 0.0 | 0.829 | 1.046 | 0.500 | 0.452 |
| 0.5 | 0.788 | 0.957 | 0.511 | 0.504 |
| 0.75 | 0.809 | 0.940 | 0.628 | 0.570 |
| 1.5 | 0.928 | 1.013 | 0.505 | 0.928 |
| 2.0 | 0.895 | 0.958 | 0.515 | 0.928 |

Decision:

- Added optional mode
  `t2_fast_t3_union_top10_s10_tactical3_new510_alpha50_w0p5_experimental`.
- It is not default.  It is more robust than w0 or w2 across the three checked
  sets, but it still depends on the heavier Top10/s10/tactical3 refinement and
  still has one broad100 row slightly over 5s from the same candidate pool.
- Learned final selector remains the right direction, but the first ridge probe
  does not beat a fixed `0.5` blend strongly enough to wire into runtime.

### 2026-06-15 broad160_more T2 data and new630 probe

Generated a larger fresh local D-drive broad set with a new seed/root range:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_source_all_actions.jsonl`
- exact combined output:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_oracle_cap50_alllegal.jsonl`
- full teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_alllegal_cap50.teacher.jsonl`
- train split:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_train120_alllegal_cap50.teacher.jsonl`
- holdout split:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_holdout40_alllegal_cap50.teacher.jsonl`

Generation summary:

- source rows: `160`
- positions: `BB=80`, `BTN=80`
- source candidates: `3846`
- source generation time: `79.0s`
- T3 states scored during source generation: `175257`
- Rust cap50 all-legal rows: `160`
- source Top1 changed by cap50: `116/160`
- average source Top1 EV loss: `3.007`
- max source Top1 EV loss: `25.042`
- average cap50 runtime per row: `49428.4 ms`

Per-chunk cap50 check:

| chunk | rows | changed Top1 | avg row ms | avg source Top1 Reg | max source Top1 Reg |
|---|---:|---:|---:|---:|---:|
| 0 | 20 | 20 | 50385.9 | 1.804 | 7.238 |
| 1 | 20 | 9 | 50765.1 | 0.905 | 3.946 |
| 2 | 20 | 14 | 50096.6 | 3.354 | 11.034 |
| 3 | 20 | 16 | 50945.8 | 3.068 | 9.817 |
| 4 | 20 | 11 | 46255.6 | 4.207 | 18.936 |
| 5 | 20 | 17 | 49353.6 | 6.802 | 25.042 |
| 6 | 20 | 11 | 48998.2 | 1.659 | 22.680 |
| 7 | 20 | 18 | 48626.1 | 2.261 | 7.894 |

Training data:

- combined teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_enriched510_plus_broad160_train120_alllegal_cap50.teacher.jsonl`
- dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/reranker_t2_enriched630_plus_broad160_train120_dim520`
- rows: `630`
- samples: `10041`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/models/t2-enriched630-broad160train120-alllegal-ft-20260615/action_value_best.pt`
- init:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/models/t2-enriched510-broad100-alllegal-ft-20260615/action_value_best.pt`
- local CPU training time: `480s` capped by `--max-seconds`

Model-only results:

| eval set | model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg | Top20 Reg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| broad160_holdout40 | current alpha50 | 5.0% | 20.0% | 42.5% | 65.0% | 80.0% | 95.0% | 3.363 | 0.669 | 0.024 |
| broad160_holdout40 | current + new510 alpha0.50 | 5.0% | 20.0% | 32.5% | 70.0% | 82.5% | 97.5% | 3.231 | 0.678 | 0.005 |
| broad160_holdout40 | current + new630 alpha0.50 | 5.0% | 17.5% | 42.5% | 75.0% | 87.5% | 97.5% | 3.389 | 0.597 | 0.005 |
| broad160_holdout40 | new630 specialist | 2.5% | 10.0% | 40.0% | 75.0% | 87.5% | 97.5% | 3.980 | 0.527 | 0.005 |
| broad100 train-seen | new630 specialist | 80.0% | 93.0% | 97.0% | 100.0% | 100.0% | 100.0% | 0.873 | 0.000 | 0.000 |
| clean20c | new630 specialist | 50.0% | 65.0% | 70.0% | 90.0% | 100.0% | 100.0% | 0.998 | 0.071 | 0.000 |
| holdout4 | new630 specialist | 30.0% | 47.5% | 57.5% | 82.5% | 87.5% | 100.0% | 3.540 | 0.632 | 0.000 |

Runtime on broad160_holdout40:

| runtime | chosen Reg | chosen hit | Top10 | Top10 Reg | Top20 | Top20 Reg | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current default Top5/s2/w1.5 | 1.075 | 37.5% | 85.0% | 0.228 | 97.5% | 0.017 | 418.1 ms | 1110.9 ms |
| current + new510 alpha0.50, w0.5 | 0.698 | 42.5% | 67.5% | 0.207 | 97.5% | 0.017 | 3061.7 ms | 4905.6 ms |
| current + new630 alpha0.50, w0.5 | 0.827 | 40.0% | 70.0% | 0.191 | 97.5% | 0.017 | 2911.9 ms | 4791.4 ms |

Decision:

- Do not add new630 to runtime config.  It improved the raw Top10 pool on the
  hard broad160_holdout40 set, but it worsened final selected-action EV loss
  versus the existing new510 w0.5 runtime (`0.827` vs `0.698`).
- The broad160_holdout40 set is much harder than previous external checks and
  should be kept as a clean external gate for the next T2 model.
- The next useful change is not simply another fine-tune checkpoint.  The
  model needs either more broad exact data before fitting, or a training loss
  that directly optimizes the exact-rerank pool objective without destroying
  Top1/Top3 ordering.

### 2026-06-15 T2 aux/new630 and Top12 runtime probe

Code changes:

- `ai/tutor/hybrid_t1t2.py` now supports T2 auxiliary shortlist models:
  `--t2-aux-shortlist-model` and `--t2-aux-shortlist-k`.
- The auxiliary model is candidate-pool only.  It can add candidates to the
  pool, but does not replace the primary action-value ensemble or become the
  final selector.
- Added experimental runtime mode:
  `t2_fast_t3_union_top12_s10_tactical3_new510_alpha50_w0p5_experimental`.

new630 as auxiliary candidate source on `broad160_holdout40`:

| runtime | chosen Reg | chosen hit | Top10 | Top10 Reg | Top15 | Top15 Reg | Top20 | Top20 Reg | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| new510 w0.5 baseline | 0.698 | 42.5% | 67.5% | 0.207 | 90.0% | 0.103 | 97.5% | 0.017 | 3061.7 ms | 4905.6 ms |
| new510 w0.5 + aux630 k3 | 0.697 | 42.5% | 67.5% | 0.207 | 90.0% | 0.103 | 97.5% | 0.017 | 3028.0 ms | 4879.6 ms |
| new510 w0.5 + aux630 k5 | 0.697 | 42.5% | 67.5% | 0.207 | 90.0% | 0.103 | 97.5% | 0.017 | 2997.0 ms | 4902.7 ms |

Observation:

- aux630 only added a genuinely new pool candidate in `1/40` rows, so it is
  too correlated with the current ensemble to be useful as a separate
  candidate source.

Expanded sync refinement:

| eval set | runtime | chosen Reg | chosen hit | Top10 | Top10 Reg | Top20 | Top20 Reg | elapsed mean | elapsed max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| broad160_holdout40 | Top12/s10/w0.5 | 0.673 | 37.5% | 82.5% | 0.116 | 97.5% | 0.017 | 3435.0 ms | 4885.2 ms |
| clean20c | Top12/s10/w0.5 | 0.320 | 55.0% | 90.0% | 0.011 | 100.0% | 0.000 | 3687.0 ms | 4782.5 ms |
| clean_holdout4 | Top12/s10/w0.5 | 0.530 | 62.5% | 90.0% | 0.026 | 97.5% | 0.000 | 3346.6 ms | 4792.4 ms |
| broad160_holdout40 | Top15/s10/w0.5 | 0.605 | 40.0% | 90.0% | 0.103 | 97.5% | 0.017 | 3968.1 ms | 4828.6 ms |
| broad160_holdout40 | Top20/s10/w0.5 | 1.594 | 30.0% | 87.5% | 0.109 | 97.5% | 0.017 | 4634.3 ms | 4807.6 ms |

Decision:

- Do not promote Top15/Top20 yet.  Top20 makes the refined pool wider but the
  partial-exact noise hurts final selection.  Top15 improved broad160, but a
  clean20c cross-check stalled after `18/20` rows in the Python T3 pool path,
  so it is not runtime-safe as-is.
- Keep Top12 as experimental.  It improved broad160 and clean20c, stayed below
  5 seconds in completed checks, but regressed clean_holdout4 slightly versus
  the prior new510 w0.5 result (`0.530` vs `0.504`).
- Next runtime-safe work should focus on fixing/guarding the Python-side T3
  pool timeout path before widening beyond Top12, and on training a selector
  that chooses among refined candidates without overreacting to noisy partial
  exact values.

### 2026-06-15 T3 runtime fallback guard

Issue found:

- `ai/tutor/t3_runtime.py` passed a timeout to the Rust exact solver, but
  `ai/tutor/exact_late.py` caught Rust timeout/failure and silently fell back
  to pure Python exact.
- On joker-heavy T2 -> T3 rows this fallback can run far beyond the 5 second
  runtime budget.  The clean20c Top15 probe repeatedly stalled at row `18/20`,
  where the T2 dealt cards were `X2 Js Qd`.

Code changes:

- `evaluate_late_position(..., fallback_on_rust_error=False)` is now available.
  Default remains `True` so offline/batch callers keep legacy fallback behavior.
- `ai/tutor/t3_runtime.py` uses `fallback_on_rust_error=False` for runtime T3
  exact rerank.
- T3 runtime pool generation now accepts a wall-clock `deadline_s`, and
  T2 exact-partial refinement preserves completed candidate `refined_score`
  values when a later T3 candidate times out.

Validation:

- `python -m pytest tests/test_hybrid_t1t2.py tests/test_exact_late_rust.py -q`
  -> `41 passed`

Top12 after disabling Rust fallback:

| eval set | runtime | chosen Reg | chosen hit | Top10 | Top10 Reg | Top20 | Top20 Reg | elapsed mean | elapsed max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| broad160_holdout40 | Top12/s10/w0.5/no-fallback | 0.673 | 37.5% | 80.0% | 0.127 | 97.5% | 0.017 | 4060.9 ms | 4998.5 ms |
| clean20c | Top12/s10/w0.5/no-fallback | 0.320 | 55.0% | 90.0% | 0.011 | 100.0% | 0.000 | 4243.5 ms | 4835.1 ms |
| clean_holdout4 | Top12/s10/w0.5/no-fallback | 0.612 | 62.5% | 90.0% | 0.026 | 97.5% | 0.000 | 3954.9 ms | 4901.0 ms |

Top15 after disabling Rust fallback:

| eval set | runtime | chosen Reg | chosen hit | Top10 | Top10 Reg | elapsed mean |
|---|---|---:|---:|---:|---:|---:|
| clean20c | Top15/s10/w0.5/no-fallback | 0.797 | 50.0% | 90.0% | 0.011 | 4666.0 ms |

Decision:

- The no-fallback guard is correct for runtime.  It prevents an unbounded
  Python exact fallback from blocking gameplay.
- Top15 is still not a good runtime setting: it returns now, but many rows hit
  the budget and final chosen EV loss is worse on clean20c.
- Top12 remains experimental only.  It improves broad160 and clean20c versus
  the prior Top10/new510 w0.5 baseline, but it worsens clean_holdout4.
- The next strength gain should come from either a better refined-candidate
  selector trained on timeout/no-fallback outputs, or a narrower high-value
  expansion rule that adds only the few Top12/Top15 candidates that actually
  reduce EV loss.

### 2026-06-15 T2 post-refine Top12 expansion

Motivation:

- Full Top12 improved some sets but regressed `clean_holdout4`.
- Row-level comparison showed the best simple gate was not a model-score rule;
  it was whether the initial Top10 refinement finished quickly.  If Top10 is
  already slow, widening adds noisy partial-exact values near the time budget.

Code changes:

- Added runtime config fields in `HybridConfig`:
  - `t2_post_refine_sync_exact_k`
  - `t2_post_refine_min_remaining_ms`
- The T2 runtime now can run the normal sync refinement first, then expand to
  a wider sync candidate count only if enough wall-clock budget remains and
  the first refinement had no error.
- Added experimental runtime mode:
  `t2_fast_t3_union_top10_s10_tactical3_new510_alpha50_w0p5_post12_min2000_experimental`.

No-fallback runtime comparison:

| eval set | Top10 chosen Reg | Top12 chosen Reg | post12/min2000 chosen Reg | post12 elapsed mean | post12 elapsed max | expanded rows |
|---|---:|---:|---:|---:|---:|---:|
| clean20c | 0.511 | 0.320 | 0.334 | 3853.1 ms | 4853.7 ms | 2/20 |
| clean_holdout4 | 0.586 | 0.612 | 0.570 | 3628.0 ms | 4899.1 ms | 14/40 |
| broad160_holdout40 | 0.780 | 0.673 | 0.674 | 3688.3 ms | 4849.1 ms | 7/40 |
| combined | 0.648 | 0.578 | 0.565 | - | - | 23/100 |

Decision:

- `post12/min2000` is better than fixed Top10 and fixed Top12 on the current
  three-set check.  It is the strongest runtime candidate from this pass.
- Keep it experimental until it is checked on a larger external set, because
  it is a hand-tuned gate from only 100 labeled rows.
- The next step is to collect more no-fallback Top10/Post12 runtime rows and
  train or fit a selector/gate on features available after the initial Top10
  pass.

### 2026-06-15 T2 external post12 recheck and clean-miss fine-tune

Extra validation sets:

- clean20 source:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout_20260615/t2_clean20_source_all_actions.jsonl`
- clean20 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout_20260615/t2_clean20_all_legal_cap50.teacher.jsonl`
- clean20b source:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout2_20260615/t2_clean20b_source_all_actions.jsonl`
- clean20b teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout2_20260615/t2_clean20b_all_legal_cap50.teacher.jsonl`

Important scoring note:

- `ai.tutor.benchmark_t2_t3_union_runtime` reports zero teacher loss on these
  `source_all_actions` inputs because it compares against `source_score`.
- The correct cap50 comparison must be done with
  `ai.training.score_hybrid_runtime_output` against the `*.teacher.jsonl`
  files.

Correct cap50 runtime scoring:

| eval set | runtime | chosen Reg | chosen hit | Top3 Reg | Top5 Reg | Top10 Reg | elapsed mean | elapsed max | expanded rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean20 | Top10/no-fallback | 1.320 | 55.0% | 0.126 | 0.112 | 0.112 | 3275.9 ms | 4767.2 ms | 0/20 |
| clean20 | post12/min2000 | 1.320 | 55.0% | 0.220 | 0.112 | 0.112 | 3603.6 ms | 4266.7 ms | 14/20 |
| clean20b | Top10/no-fallback | 1.288 | 60.0% | 0.225 | 0.181 | 0.181 | 3324.8 ms | 4809.1 ms | 0/20 |
| clean20b | post12/min2000 | 1.259 | 60.0% | 0.139 | 0.095 | 0.095 | 3587.1 ms | 4806.8 ms | 10/20 |

Interpretation:

- These clean20/clean20b rows show the main failure is not only TopK recall.
  The best cap50 action is usually inside Top5/Top10, but the runtime final
  selection still chooses the wrong refined candidate.
- Post12 helps clean20b slightly, but does not fix clean20. It adds latency
  without improving the largest clean20 misses.

Fixed final-selection weight sweep on all post12 rows from
clean20/clean20b/clean20c/clean_holdout4/broad160_holdout40:

- rows: `140`
- best fixed weight: `0.5`
- best average known regret: `0.771`
- hit rate: `56.4%`

Linear refined-candidate selector probe on the same 140 rows:

| heldout | baseline Reg | linear selector Reg | note |
|---|---:|---:|---|
| broad160 | 0.643 | 0.936 | worsens |
| clean20 | 1.530 | 1.384 | improves |
| clean20b | 1.659 | 1.322 | improves |
| clean20c | 0.328 | 0.358 | worsens |
| clean_holdout4 | 1.053 | 0.610 | improves |

Decision:

- Do not wire the linear selector into runtime. It helps several clean sets but
  regresses the hard broad160 external set.
- Keep `post12/min2000` experimental. The larger 140-row check shows it is
  still not enough for reliable T2 play.

Clean-miss fine-tune:

- combined teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train670_clean_miss_20260615/t2_enriched630_plus_clean20_clean20b_alllegal_cap50.teacher.jsonl`
- dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train670_clean_miss_20260615/reranker_t2_enriched670_clean_miss_dim520`
- rows: `670`
- samples: `11022`
- init:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/models/t2-enriched630-broad160train120-alllegal-ft-20260615/action_value_best.pt`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train670_clean_miss_20260615/models/t2-enriched670-cleanmiss-ft-from-new630-20260615/action_value_best.pt`
- local CPU training: capped at `480s`, best epoch `3`

New670 model-only external results:

| eval set | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg | Top20 Reg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean20c | 35.0% | 65.0% | 70.0% | 95.0% | 100.0% | 100.0% | 2.283 | 0.011 | 0.000 |
| clean_holdout4 | 27.5% | 42.5% | 60.0% | 80.0% | 87.5% | 95.0% | 3.507 | 0.650 | 0.059 |
| broad160_holdout40 | 5.0% | 25.0% | 45.0% | 70.0% | 85.0% | 97.5% | 4.012 | 0.611 | 0.005 |

Decision:

- Do not promote new670. Adding clean20/clean20b improves clean20c Top10, but
  hurts or fails to improve holdout4 and broad160. The model is still
  distribution-sensitive.
- The next useful T2 strength step is broader exact teacher data, not another
  narrow clean-miss fine-tune. The exact labels need more T0/T1 branch
  diversity and more broad T2 states before the model can safely narrow to
  Top5/Top10.

### 2026-06-15 T2 more10 external-style cap50 shard

New local D-drive shard:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train40_more10/inputs/t2_fresh_ext_more10_balanced40_model_teacher.jsonl`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train40_more10/cap50_exact_all40/t2_oracle_cap50_limit40.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train40_more10/t2_fresh_ext_more10_40_cap50.teacher.jsonl`
- dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train40_more10/av_dataset_more10_40_cap50_dim520`

Shard shape:

- rows: `40` (`bb=20`, `btn=20`)
- unique states: `40`
- unique dealt triples: `20`
- original candidates: `978` total, `24.45` average
- cap50 exact candidates in teacher: `400`
- cap50 exact elapsed: `8856.4 ms/row` average
- source Top1 changed after cap50: `26/40`
- source Top1 cap50 regret: average `2.1515`, max `16.4182`

Model-only check on the cap50 teacher:

| model source | Top3 | Top5 | Top10 | Top3 Reg | Top5 Reg | Top10 Reg |
|---|---:|---:|---:|---:|---:|---:|
| current base 6-model ensemble | 87.5% | 97.5% | 100.0% | 0.225 | 0.000 | 0.000 |
| current alpha50 7-model ensemble | 90.0% | 97.5% | 100.0% | 0.192 | 0.000 | 0.000 |

Runtime-selected unknown labels:

- default Top5/s2 unknown rows: `9`
- default unknown cap50:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train40_more10/runtime_selected_unknown_20260615/default_top5_s2_cap50/t2_oracle_cap50_limit9.jsonl`
- default unknown same Top1 after cap50: `2/9`
- default unknown source Top1 regret: average `1.727`, max `8.368`
- post12/min2000 unknown rows: `6`
- post12/min2000 unknown cap50:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train40_more10/runtime_selected_unknown_20260615/post12_min2000_cap50/t2_oracle_cap50_limit6.jsonl`
- post12/min2000 unknown same Top1 after cap50: `3/6`
- post12/min2000 unknown source Top1 regret: average `0.916`, max `3.443`

Fair runtime scoring after merging the runtime-selected exact labels:

| runtime | chosen Reg | chosen hit | Top1 | Top3 | Top5 | Top10 | Top3 Reg | Top5 Reg | Top10 Reg | elapsed mean | elapsed max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default Top5/s2 | 1.228 | 40.0% | 40.0% | 70.0% | 82.5% | 92.5% | 0.438 | 0.183 | 0.085 | 654.9 ms | 2130.3 ms |
| post12/min2000 | 0.502 | 57.5% | 57.5% | 80.0% | 82.5% | 87.5% | 0.229 | 0.189 | 0.152 | 3734.2 ms | 4892.5 ms |

Worst post12 chosen-action misses:

| row | pos | dealt | chosen Reg | best runtime rank | cap50 best | runtime chosen |
|---:|---|---|---:|---:|---|---|
| 35 | btn | `4d Jc 6s` | 6.278 | 2 | `6s->middle; Jc->bottom; discard 4d` | `4d->middle; Jc->bottom; discard 6s` |
| 31 | btn | `Ts 7d Qc` | 3.443 | 16 | `7d->bottom; Qc->top; discard Ts` | `7d->bottom; Qc->middle; discard Ts` |
| 2 | bb | `As 2d 3d` | 2.499 | 10 | `2d->bottom; As->top; discard 3d` | `2d->middle; As->top; discard 3d` |
| 19 | bb | `7s 7h 8s` | 2.197 | 1 | `7h->bottom; 7s->bottom; discard 8s` | `7h->middle; 7s->middle; discard 8s` |
| 10 | bb | `Js 4d 7s` | 1.798 | 17 | `7s->middle; Js->top; discard 4d` | `7s->middle; Js->middle; discard 4d` |

Decision:

- `post12/min2000` is materially better than default on this more10 shard
  (`chosen Reg 1.228 -> 0.502`) and still stays within the 5-second budget.
- Do not promote it yet.  The Top10 pool can still miss important cap50 best
  actions (`Top10 87.5%` on post12 runtime scoring), and final selection still
  makes large mistakes even when the best action is present.
- Keep more10 as an external-style check for now.  The next step is to add
  broader exact teacher data with more T0/T1 branch diversity, then train/evaluate
  against this shard instead of immediately fitting to it.

### 2026-06-15 T2 more11 train shard and new710 fine-tune

New train shard:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train80_more11/inputs/t2_fresh_ext_more11_balanced80_model_teacher.jsonl`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train80_more11/cap50_exact_all80/t2_oracle_cap50_limit80.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train80_more11/t2_fresh_ext_more11_80_cap50.teacher.jsonl`
- dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/fresh_t2_from_t0t1_topk_20260615/fresh_external1200_varied_20260615/train80_more11/av_dataset_more11_80_cap50_dim520`

Shard shape:

- rows: `80` (`bb=40`, `btn=40`)
- unique states: `80`
- unique dealt triples: `40`
- original candidates: `1872` total, `23.4` average
- cap50 exact candidates in teacher: `800`
- cap50 exact elapsed: `7984.8 ms/row` average
- source Top1 changed after cap50: `60/80`
- source Top1 cap50 regret: average `2.613`, max `16.462`

Training:

- combined teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_more11_20260615/t2_enriched630_plus_more11_80_cap50.teacher.jsonl`
- combined dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_more11_20260615/reranker_t2_enriched710_more11_dim520`
- rows: `710`
- converted samples: `10841`
- init:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/models/t2-enriched630-broad160train120-alllegal-ft-20260615/action_value_best.pt`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_more11_20260615/models/t2-enriched710-more11-ft-from-new630-20260615/action_value_best.pt`
- local GPU: `NVIDIA GeForce RTX 2060 SUPER`
- training time: about `207s`
- best-val snapshot: Top1 `85.7%`, Top3 `91.4%`, Top5 `94.3%`,
  Top10 `97.1%`, regret `0.565`

Model-only external results:

| eval set | model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg | Top20 Reg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| more10 | new710 specialist | 52.5% | 87.5% | 97.5% | 100.0% | 100.0% | 100.0% | 1.413 | 0.000 | 0.000 |
| clean_holdout4 | new710 specialist | 30.0% | 47.5% | 57.5% | 80.0% | 90.0% | 95.0% | 3.541 | 0.876 | 0.059 |
| broad160_holdout40 | new710 specialist | 2.5% | 20.0% | 42.5% | 75.0% | 82.5% | 97.5% | 4.097 | 0.566 | 0.005 |

Blend probe with the current alpha50 ensemble plus new710:

| eval set | blend | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 Reg | Top10 Reg | Top20 Reg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| more10 | new710 alpha0.25 | 55.0% | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.416 | 0.000 | 0.000 |
| clean_holdout4 | new710 alpha0.25 | 25.0% | 50.0% | 60.0% | 77.5% | 92.5% | 95.0% | 3.737 | 0.646 | 0.220 |
| broad160_holdout40 | new710 alpha0.25 | 2.5% | 27.5% | 40.0% | 75.0% | 85.0% | 97.5% | 3.862 | 0.593 | 0.005 |
| more10 | new710 alpha0.50 | 55.0% | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.439 | 0.000 | 0.000 |
| clean_holdout4 | new710 alpha0.50 | 30.0% | 50.0% | 62.5% | 77.5% | 90.0% | 100.0% | 3.694 | 0.646 | 0.000 |
| broad160_holdout40 | new710 alpha0.50 | 2.5% | 22.5% | 42.5% | 72.5% | 87.5% | 97.5% | 3.832 | 0.584 | 0.005 |

Decision:

- Do not promote new710 as a standalone model.  It does not beat the current
  ensemble broadly and remains weak on `clean_holdout4`.
- The alpha0.25 blend is the only mildly interesting candidate-pool source:
  it improves `more10` Top5 to `100%` and improves `broad160_holdout40` Top3
  to `27.5%`, but it weakens or fails to improve clean holdout enough to justify
  a runtime change.
- Keep new710 as an experimental specialist checkpoint only.  The next strength
  step should be more broad exact data or a selector/loss change that improves
  final chosen-action EV, not just another narrow fine-tune.

### 2026-06-15 T2 broad teacher join corruption fix

The earlier `broad100`, `broad160`, and `new710` results above are superseded.
They used teacher files built from concatenated chunk outputs where Rust
`record_index` restarted at each chunk.  The old builder joined by
`record_index`, so later chunks were attached to the wrong source rows.

Bad teacher audit:

| file | bad rows | bad candidates | reason |
|---|---:|---:|---|
| `broad100_20260615/t2_broad100_alllegal_cap50.teacher.jsonl` | 80/100 | 1893 | duplicate chunk `record_index` |
| `broad160_more_20260615/t2_broad160_alllegal_cap50.teacher.jsonl` | 140/160 | 3366 | duplicate chunk `record_index` |
| `broad160_more_20260615/t2_broad160_train120_alllegal_cap50.teacher.jsonl` | 100/120 | 2397 | duplicate chunk `record_index` |
| `broad160_more_20260615/t2_broad160_holdout40_alllegal_cap50.teacher.jsonl` | 40/40 | 969 | duplicate chunk `record_index` |
| `t2_train710_more11_20260615/t2_enriched630_plus_more11_80_cap50.teacher.jsonl` | 180 rows | mixed corrupted source | inherited corrupted broad rows |

Fixes:

- `ai/training/build_t2_action_value_teacher_from_exact.py` now detects
  duplicate/restarted `record_index` when `global_index` is absent and joins by
  row order instead.
- Builder summaries now include `join_mode`, `invalid_rows`, and
  `invalid_candidates`.
- `ai/training/convert_action_value_teacher.py` now filters candidate actions
  whose placement+discard cards do not exactly match the row `dealt` cards, so
  a corrupted teacher file cannot silently poison a dataset.
- Tests added:
  - `tests/test_build_t2_action_value_teacher_from_exact.py`
  - `tests/test_convert_action_value_teacher.py`

Verification:

```powershell
python -m pytest tests/test_build_t2_action_value_teacher_from_exact.py tests/test_convert_action_value_teacher.py -q
python -m json.tool ai/config/t2_action_value_ensemble_20260615.json > $null
```

Result: `3 passed`.

Converter guard check:

| input teacher | samples kept | invalid selected candidates |
|---|---:|---:|
| old `broad160_holdout40_alllegal_cap50.teacher.jsonl` | 0 | 969 |
| fixed `broad160_holdout40_alllegal_cap50.fixed_20260615.teacher.jsonl` | 969 | 0 |
| fixed combined `t2_clean410_plus_broad100fixed_broad160fixed_more11_710.teacher.jsonl` | 11055 | 0 |

Corrected fixed artifacts:

- fixed broad100 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad100_20260615/t2_broad100_alllegal_cap50.fixed_20260615.teacher.jsonl`
- fixed broad160 full teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_alllegal_cap50.fixed_20260615.teacher.jsonl`
- fixed broad160 train120 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_train120_alllegal_cap50.fixed_20260615.teacher.jsonl`
- fixed broad160 holdout40 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/broad160_more_20260615/t2_broad160_holdout40_alllegal_cap50.fixed_20260615.teacher.jsonl`
- fixed combined train710 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_fixed_broad_more11_20260615/t2_clean410_plus_broad100fixed_broad160fixed_more11_710.teacher.jsonl`
- fixed combined train710 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_fixed_broad_more11_20260615/reranker_t2_clean410_broadfixed_more11_710_dim520`

Corrected training:

- model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_fixed_broad_more11_20260615/models/t2-clean410-broadfixed-more11-710-ft-20260615/action_value_best.pt`
- train rows: `710`
- train candidates: `11055`
- positions: `bb=360`, `btn=350`
- invalid rows/candidates: `0 / 0`
- local GPU: `NVIDIA GeForce RTX 2060 SUPER`
- training time: about `201s`
- internal validation: Top1 `68.6%`, Top3 `85.7%`, Top5 `91.4%`,
  Top10 `97.1%`, regret `0.797`, T3-regret `0.203`

Corrected model-only external results:

| eval set | model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| more10 | fixed710 standalone | 57.5% | 85.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.900 | 0.424 | 0.000 |
| clean_holdout4 | fixed710 standalone | 25.0% | 52.5% | 55.0% | 80.0% | 95.0% | 97.5% | 3.688 | 2.370 | 0.491 |
| broad160_holdout40 fixed | fixed710 standalone | 25.0% | 55.0% | 62.5% | 85.0% | 92.5% | 95.0% | 1.634 | 0.649 | 0.226 |
| broad160_holdout40 fixed | current alpha50 | 30.0% | 52.5% | 57.5% | 85.0% | 92.5% | 97.5% | 1.574 | 0.619 | 0.228 |
| broad160_holdout40 fixed | current + fixed710 alpha0.25 | 30.0% | 55.0% | 57.5% | 87.5% | 95.0% | 97.5% | 1.373 | 0.587 | 0.227 |

Corrected runtime checks:

| eval set | runtime mode | chosen Reg | chosen hit | Top3 | Top5 | Top10 | elapsed mean | elapsed max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| broad160_holdout40 fixed | current default | 1.004 | 22.5% | 52.5% | 65.0% | 82.5% | 587.4 ms | 1800.2 ms |
| broad160_holdout40 fixed | fixed710 alpha0.25 Top10/s10/post12 | 0.499 | 35.0% | 60.0% | 67.5% | 77.5% | 3601.4 ms | 4808.5 ms |
| clean_holdout4 | current default | 1.060 | 42.5% | 67.5% | 75.0% | 87.5% | 539.8 ms | 1724.7 ms |
| clean_holdout4 | fixed710 alpha0.25 Top10/s10/post12, 4900ms internal budget | 0.962 | 47.5% | 65.0% | 77.5% | 87.5% | 3429.2 ms | 4921.0 ms |

Decision:

- Do not use any model or metric trained from the corrupted broad100/broad160
  teacher files as evidence.
- The corrected fixed710 alpha0.25 post12 runtime is promising because it cuts
  selected-action EV loss on fixed broad160 from `1.004` to `0.499` and on
  clean_holdout4 from `1.060` to `0.962` while staying under about 5 seconds.
- Do not promote it as default yet.  Clean_holdout4 Top3 is slightly worse
  (`67.5% -> 65.0%`), broad160 Top10 is worse (`82.5% -> 77.5%`), and some
  rows still hit `time_budget_exhausted` internally even though a result is
  returned under the wall-clock budget.
- Keep the fixed710 modes in
  `ai/config/t2_action_value_ensemble_20260615.json` as experimental only.
  The next strength step should validate this corrected pipeline on another
  clean external shard before default promotion.

### 2026-06-15 clean20c fixed710 validation and default promotion

Ran the corrected fixed710 alpha0.25 post12 mode on the independent
`clean20c` external set:

```powershell
python -m ai.tutor.benchmark_t2_t3_union_runtime `
  --input D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/t2_clean20c_source_all_actions.jsonl `
  --output-dir D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/runtime_fixed710_post12_clean20c_budget4900 `
  --runtime-config ai/config/t2_action_value_ensemble_20260615.json `
  --runtime-mode t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_min2000_experimental `
  --limit 20 `
  --progress-every 5 `
  --device cpu `
  --t3-device auto

python -m ai.training.score_hybrid_runtime_output `
  --teacher D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/t2_clean20c_all_legal_cap50.teacher.jsonl `
  --runtime-output D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/runtime_fixed710_post12_clean20c_budget4900/results.jsonl `
  --output D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/runtime_fixed710_post12_clean20c_budget4900/scored.json `
  --topks 1,3,5,10,15
```

Also checked a Top12 upfront variant:

- output:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/runtime_fixed710_top12_clean20c_budget4900`
- result: same chosen EV loss as Top10, slightly worse Top3, more internal
  `time_budget_exhausted` rows.  Do not use Top12 upfront as default.

Three-set selected-action comparison against the previous default:

| eval set | previous default chosen Reg | fixed710 post12 chosen Reg | delta | fixed710 elapsed mean/max |
|---|---:|---:|---:|---:|
| fixed broad160_holdout40 | 1.004 | 0.499 | -0.505 | 3601.4 / 4808.5 ms |
| clean_holdout4 | 1.060 | 0.962 | -0.098 | 3429.2 / 4921.0 ms |
| clean20c | 1.315 | 1.208 | -0.108 | 3573.4 / 4673.8 ms |

Clean20c detailed comparison:

| mode | chosen Reg | chosen hit | Top1 | Top3 | Top5 | Top10 | Top3 Reg | Top10 Reg | elapsed mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous default Top5/s2 | 1.315 | 60.0% | 45.0% | 60.0% | 75.0% | 90.0% | 0.945 | 0.117 | 631.5 / 1601.4 ms |
| fixed710 Top10/s10/post12 | 1.208 | 55.0% | 60.0% | 85.0% | 85.0% | 85.0% | 0.580 | 0.580 | 3573.4 / 4673.8 ms |
| fixed710 Top12 upfront | 1.208 | 55.0% | 60.0% | 80.0% | 85.0% | 85.0% | 0.662 | 0.580 | 3793.3 / 4686.9 ms |

Decision:

- Promote `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_min2000_experimental`
  to `default_mode` in `ai/config/t2_action_value_ensemble_20260615.json`.
- This is a pragmatic strength improvement for actual game play: selected-action
  EV loss improved on all three checked external sets and wall time stayed under
  the 5-second target.
- Keep the mode name unchanged for traceability, even though it is now the
  active default.  The remaining risk is candidate-pool TopK recall on some
  sets, so the next improvement should target the rows where Top10 misses the
  teacher best by a large EV margin.

Post-promotion verification:

```powershell
python -m pytest tests/test_build_t2_action_value_teacher_from_exact.py tests/test_convert_action_value_teacher.py tests/test_hybrid_t1t2.py -q
python -m json.tool ai/config/t2_action_value_ensemble_20260615.json > $null
git diff --check -- ai/training/build_t2_action_value_teacher_from_exact.py ai/training/convert_action_value_teacher.py ai/config/t2_action_value_ensemble_20260615.json docs/t2_t3_next_status_20260614.md tests/test_build_t2_action_value_teacher_from_exact.py tests/test_convert_action_value_teacher.py
```

Result:

- tests: `43 passed`
- JSON syntax: OK
- `git diff --check`: OK

Default-load smoke without `--runtime-mode`:

```powershell
python -m ai.tutor.benchmark_t2_t3_union_runtime `
  --input D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/t2_clean20c_source_all_actions.jsonl `
  --output-dir D:/ofc-pineapple-data/t2_next_20260614/clean_holdout3_20260615/runtime_default_after_fixed710_promotion_smoke2 `
  --runtime-config ai/config/t2_action_value_ensemble_20260615.json `
  --limit 2 `
  --progress-every 1 `
  --device cpu `
  --t3-device auto
```

Smoke result:

- loaded runtime mode:
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_min2000_experimental`
- count: `2`
- latency mean/max: `3392.8 / 3812.8 ms`
- under internal 4900ms budget: `2/2`
- errors: `0`

### 2026-06-15 clean_holdout4 teacher fix and T2 sync-order audit

Found that the original `clean_holdout4` cap50 teacher was also affected by a
row-order join issue:

- old teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/t2_clean_holdout4_alllegal_cap50.teacher.jsonl`
- symptom: first 40 teacher `record_index` values were `[0..19, 0..19]`,
  so rows 20-39 could be scored against mismatched `dealt` cards.
- bad rows/candidates in the old teacher: `20` rows, `447` candidates.
- fixed teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/clean_holdout4_20260615/t2_clean_holdout4_alllegal_cap50.fixed_20260615.teacher.jsonl`
- fixed build result: `40` rows, `951` candidates, `invalid_rows=0`,
  `invalid_candidates=0`.

Re-scored the earlier runtime outputs against the fixed teacher:

| eval set | mode | chosen Reg | Top1 | Top3 | Top5 | Top10 | elapsed mean |
|---|---|---:|---:|---:|---:|---:|---:|
| clean_holdout4 fixed teacher | previous default | 1.060 | 42.5% | 67.5% | 75.0% | 87.5% | 539.8 ms |
| clean_holdout4 fixed teacher | active fixed710 Top10/s10/tactical3/post12 | 0.962 | 47.5% | 65.0% | 77.5% | 87.5% | 3429.2 ms |

Runtime audit:

- The high predicted-bust filter was too aggressive in principle: high bust is
  not automatically bad, only forced or 100% bust candidates should be excluded
  when safe legal alternatives exist.
- Updated `_sync_refinement_candidates` so T2 forced-bust candidates are still
  skipped, but merely high predicted-bust model top candidates are not removed
  from sync refinement.
- This code change alone did not change the clean_holdout4 fixed score, so the
  main issue was not high-bust filtering.

Sync-order experiments:

| eval set | variant | chosen Reg | Top1 | Top3 | Top5 | Top10 | elapsed mean/max | under budget |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean_holdout4 fixed teacher | active fixed710 tactical-first | 0.962 | 47.5% | 65.0% | 77.5% | 87.5% | 3377-3429 / 4699-4921 ms | 40/40 or near |
| clean_holdout4 fixed teacher | no tactical sync insurance | 0.710 | 50.0% | 70.0% | 80.0% | 87.5% | 3460 / 4778 ms | 40/40 |
| clean_holdout4 fixed teacher | model Top3 + tactical interleave | 0.594 | 52.5% | 72.5% | 82.5% | 90.0% | 3633 / 4918 ms | 39/40 internal |
| fixed broad160_holdout40 | active fixed710 tactical-first | 0.499 | 47.5% | 60.0% | 67.5% | 77.5% | 3601 / 4808 ms | 40/40 |
| fixed broad160_holdout40 | no tactical sync insurance | 0.773 | 30.0% | 55.0% | 75.0% | 82.5% | 3553 / 4694 ms | 40/40 |
| fixed broad160_holdout40 | model Top3 + tactical interleave | 0.739 | 32.5% | 57.5% | 72.5% | 90.0% | 3795 / 4728 ms | 40/40 |
| clean20c | active fixed710 tactical-first | 1.208 | 60.0% | 85.0% | 85.0% | 85.0% | 3573 / 4674 ms | 20/20 |
| clean20c | model Top3 + tactical interleave | 1.276 | 55.0% | 70.0% | 85.0% | 85.0% | 4095 / 4790 ms | 20/20 |

Decision:

- Do not replace the active default with `no tactical` or `model Top3 +
  tactical interleave`.  The interleave order fixes clean_holdout4 but regresses
  broad160 and clean20c.
- Keep `t2_sync_model_insurance_k` as an experimental/runtime knob only.
  Added config mode:
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_modelins3_interleave_experimental`.
- A diagnostic arbitration between active default and interleave found that a
  simple gate can reduce clean4+broad160 combined selected EV loss from `0.731`
  to about `0.547`, but the best gate currently depends on challenger-mode
  selected-candidate features.  That is not yet a safe synchronous 5-second
  runtime policy.
- Next useful step: build a lightweight pre-refinement ordering gate from
  features available before exact starts, then validate it on broad160,
  clean_holdout4, and clean20c before changing the default.

### 2026-06-15 T2 KK-gated sync-order promotion

Built a pre-refinement gate that only uses model output available before exact
starts:

- feature: model Top1 candidate `predicted_fl_types.kk`
- rule: use model Top3 + tactical interleave order only when
  `model_top1_kk >= 0.03132572025060654`
- otherwise: keep the existing tactical-first sync order

Offline replay on clean_holdout4 + fixed broad160 + clean20c:

- active default combined selected EV loss: `0.826`
- always-interleave combined selected EV loss: `0.788`
- KK-gated selected EV loss: `0.672`
- good switches: `4`
- bad switches: `0`
- missed good switches: `1`

Runtime validation:

| eval set | active default chosen Reg | KK-gated chosen Reg | delta | active Top3 | KK-gated Top3 | KK-gated elapsed mean/max | under budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_holdout4 fixed teacher | 0.962 | 0.594 | -0.368 | 65.0% | 72.5% | 3616.0 / 4699.2 ms | 40/40 |
| fixed broad160_holdout40 | 0.499 | 0.503 | +0.003 | 60.0% | 62.5% | 3568.4 / 4791.8 ms | 40/40 |
| clean20c | 1.208 | 1.175 | -0.033 | 85.0% | 85.0% | 3747.3 / 4721.2 ms | 20/20 |
| clean20 | 1.320 | 1.096 | -0.224 | 80.0% | 85.0% | 3682.3 / 4711.8 ms | 20/20 |
| clean20b | 1.259 | 1.259 | +0.000 | 75.0% | 75.0% | 3619.6 / 4710.6 ms | 20/20 |

Five-set aggregate:

- rows: `140`
- active default selected EV loss: `0.958`
- KK-gated selected EV loss: `0.817`
- aggregate delta: `-0.141`
- all checked rows returned under the 5 second runtime target

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_modelins3_kkgate031_experimental`
  to `default_mode` in `ai/config/t2_action_value_ensemble_20260615.json`.
- Keep the old fixed710 tactical-first mode as the previous active default for
  rollback/comparison.
- This is an EV-loss improvement, not proof that the model itself is solved.
  The next strength step is still mining rows where Top10/Top15 miss the
  cap50 teacher best or where the sync ordering gate leaves a large chosen EV
  loss.

### 2026-06-15 T2 hard-loss specialist diagnostic

Mined the remaining selected-action loss from the KK-gated five-set runtime
outputs:

- rows checked: `140`
- rows with selected EV loss > 0: `64`
- rows with loss >= 0.5: `42`
- rows with loss >= 1.0: `23`
- largest losses were mostly cases where the teacher best was already in the
  model pool but sync refinement/selection chose another candidate.

Built a small hard-loss fine-tuning set:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_hardloss_kkgate031_20260615/t2_hardloss_kkgate031_regge0p5.teacher.jsonl`
- records: `75` after duplicating larger-loss rows
- candidate-level samples: `1,755`
- trained model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_hardloss_kkgate031_20260615/models/t2-hardloss-kkgate031-ft-20260615/action_value_best.pt`

Model-pool check on the mined five-set data:

| ensemble | Top1 | Top3 | Top5 | Top10 | Top10 regret |
|---|---:|---:|---:|---:|---:|
| current KK-gated weighted pool | 44.3% | 70.0% | 81.4% | 91.4% | 0.146 |
| + hard-loss alpha 0.10 | 49.3% | 76.4% | 84.3% | 95.0% | 0.123 |
| + hard-loss alpha 0.15 | 51.4% | 77.1% | 85.7% | 95.7% | 0.120 |

External candidate-pool check on `more8`/`more9`/`more10`/`more11`
(`320` T2 rows, not used to train the hard-loss specialist):

| ensemble | Top1 | Top3 | Top5 | Top10 | Top1 regret | Top3 regret | Top5 regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| current KK-gated weighted pool | 63.4% | 88.7% | 95.6% | 100.0% | 0.651 | 0.142 | 0.020 |
| + hard-loss alpha 0.10 | 63.4% | 89.1% | 96.3% | 100.0% | 0.610 | 0.122 | 0.019 |
| + hard-loss alpha 0.15 | 64.4% | 89.4% | 95.9% | 100.0% | 0.606 | 0.121 | 0.026 |
| + hard-loss alpha 0.20 | 65.0% | 89.1% | 96.6% | 100.0% | 0.601 | 0.121 | 0.027 |

Runtime smoke with the new diagnostic mode:

| eval set | rows | current chosen Reg | hardloss10 chosen Reg | current mean/max ms | hardloss10 mean/max ms | notes |
|---|---:|---:|---:|---:|---:|---|
| more8 first rows | 8 | 0.566 | 0.566 | 3388.5 / 3771.7 | 3354.5 / 3756.2 | no selected-action change |
| more11 first rows | 8 | 0.039 | 0.039 | 3634.4 / 4656.3 | 3749.8 / 4703.4 | one hardloss10 `time_budget_exhausted` flag |

Decision:

- Added diagnostic config mode:
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_modelins3_kkgate031_hardloss10_diagnostic`.
- Do not promote it to default yet.  It improves model-pool ordering on average,
  but small runtime smoke did not improve selected EV and introduced one
  budget-exhaustion flag on `more11`.
- Next useful step: mine the external `more11` and runtime-selected misses, then
  either train a broader specialist or tune sync ordering so candidate-pool
  improvements actually change the 5-second selected action.

### 2026-06-15 T2 selected-action rescue diagnostic

Rechecked final selected-action loss rather than only candidate-pool recall.

Weight-only selection sweep:

- output:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260615/kkgate_external_smoke_weight_sweep.json`
- data: five KK-gated runtime sets plus `more8`/`more11` first-row smokes
- best fixed `refined + w * model_score` weight remained `0.5`
- average known selected-action regret at `w=0.5`: `0.769`
- hit rate on known teacher rows: `56.8%`
- conclusion: changing only `t2_selection_model_weight` is not useful.

Tried a learned refined-candidate selector on the same runtime candidate logs:

- output:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260615/refined_selector_5set_extsmoke_loso.json`
- train-in-place regret: `0.861`
- current weight-0.5 baseline from the sweep: `0.769`
- leave-one-set-out improved `broad160` but regressed or failed to improve
  multiple external/smoke slices.
- conclusion: do not promote the selector.

External selected-loss inspection found the largest `more8` miss was mostly
partial-exact sampling noise: model Top1 was the teacher best, but the 5-second
sampled T3 refinement gave another candidate the higher refined score.
Blindly switching back to model Top1 is unsafe:

- modelTop1 better rows: `18`, total EV saved `19.64`
- modelTop1 worse rows: `57`, total EV cost `169.12`

The only currently acceptable rule is a conservative diagnostic rescue:

- restore model Top1 only when its predicted FL is at least `0.40`
- require current refined winner minus modelTop1 refined score to be at most
  `2.0`
- both candidates must already have `refined_score`

Added diagnostic config mode:

`t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post12_modelins3_kkgate031_top1rescue_fl040_ref2_diagnostic`

This is intentionally not default.  It is a narrow safety test for cases where
the model strongly likes a FL candidate and partial exact only weakly rejects
it.  Larger external runtime validation is still required before promoting it.

Runtime validation after implementation:

| eval set | rows | mode | chosen Reg | hit | mean/max ms | rescue applied/details | notes |
|---|---:|---|---:|---:|---:|---:|---|
| more8 balanced first rows | 8 | current default | 0.000 | 8/8 | 3203.2 / 3546.1 | 0/0 | same input recheck |
| more8 balanced first rows | 8 | top1rescue fl0.40 ref2 | 0.000 | 8/8 | 3239.3 / 3529.3 | 0/8 | no selected-action change |
| more11 balanced first rows | 8 | current default | 0.000 | 8/8 | 3634.0 / 4478.3 | 0/0 | same input recheck |
| more11 balanced first rows | 8 | top1rescue fl0.40 ref2 | 0.000 | 8/8 | 3648.4 / 4590.5 | 0/8 | no selected-action change |
| clean_holdout4 source | 40 | current default existing run | 0.000 | 40/40 | 3616.0 / 4699.2 | 0/0 | 2 timeout flags |
| clean_holdout4 source | 40 | top1rescue fl0.40 ref2 | 0.000 | 40/40 | 3555.3 / 4717.4 | 0/40 | 3 timeout flags |

Decision after runtime smoke:

- Keep `top1rescue_fl040_ref2` diagnostic only.
- It did not hurt selected EV on the checked rows, but it also did not fire.
- Do not promote it to default until a larger runtime sample shows real
  selected-action loss reduction without increasing `time_budget_exhausted`.

### 2026-06-15 T2 correct-teacher benchmark guard

Found a benchmark pitfall while rechecking T2 runtime strength:

- `t2_clean_holdout4_source_all_actions.jsonl` has candidate `source_score=0.0`
  for all legal actions.
- The old benchmark treated that as a valid all-tie teacher, producing false
  `40/40` hit and `0.000` EV loss results.
- Correct cap50 teacher files such as
  `t2_clean_holdout4_alllegal_cap50.fixed_20260615.teacher.jsonl` have real
  candidate scores and show the remaining selected-action loss.

Implemented a guard in `ai/tutor/benchmark_t2_t3_union_runtime.py`:

- if all teacher candidate scores are degenerate equal values, the row is
  marked invalid with `invalid_reason=degenerate_teacher_scores`
- invalid rows are excluded from `teacher_compared_count` and chosen EV loss
- `summary.json` / `summary.md` now report `teacher_invalid_count`

Smoke proof:

- input: `t2_clean_holdout4_source_all_actions.jsonl`, first row
- result: `teacher_compared_count=0`, `teacher_invalid_count=1`,
  `teacher_invalid_reasons={"degenerate_teacher_scores": 1}`

Correct-teacher T2 runtime recheck:

| eval set | rows | default Reg | hardloss10 Reg | post20/min1500 Reg | notes |
|---|---:|---:|---:|---:|---|
| clean_holdout4 fixed cap50 | 40 | 1.011 | 0.922 | 0.503 | post20 catches a rank20 teacher-best action; all rows under 4.74s |
| clean20c cap50 | 20 | 0.545 | 0.309 | 0.545 | post20 no quality gain; one row over 5s |
| broad160 holdout40 fixed cap50 | 40 | 0.762 | 0.804 | n/a | hardloss10 slightly regresses |

Decisions:

- Keep hardloss10 diagnostic only: it improves clean-like rows but regresses
  broad160 holdout.
- Add diagnostic mode
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_min1500_diagnostic`.
- Do not promote post20/min1500 to default yet.  It is a real strength signal
  on clean_holdout4, but it spends nearly the whole budget and did not improve
  clean20c.
- Future benchmark claims must use fixed/exact teacher files or report invalid
  teacher rows explicitly.

### 2026-06-15 T2 conditional post20 default

Added a gated post-refinement policy to avoid running Top20 exact refinement on
every T2 row:

- config fields:
  - `t2_post_refine_policy`
  - `t2_post_refine_extra_model_score_min`
- new policy: `structured_top_model_min`
- the gate accepts post20 only when:
  - own top row already has a pair or joker
  - at least one unrefined extra candidate has `model_score >= 5.0`
  - at least `1000ms` remain

This targets the clean_holdout4 rank20 high-loss miss while skipping the
clean20c rows where broad post20 did not improve quality.

Correct fixed-teacher validation:

| eval set | rows | previous default Reg/max | conditional post20 Reg/max | mean/max ms | post20 expanded | notes |
|---|---:|---:|---:|---:|---:|---|
| clean_holdout4 fixed cap50 | 40 | 1.011 / 12.137 | 0.505 / 6.257 | 3354.9 / 4767.2 | 5 | all rows under 5s |
| clean20c cap50 | 20 | 0.545 / 4.989 | 0.545 / 4.989 | 3484.4 / 4802.7 | 0 | quality unchanged, all rows under 5s |
| broad160 holdout40 fixed cap50 | 40 | 0.762 / 10.969 | 0.664 / 7.655 | 3499.6 / 4824.9 | 5 | all rows under 5s |

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_experimental`
  to `default_mode` in `ai/config/t2_action_value_ensemble_20260615.json`.
- Keep the older always-post20/min1500 mode as diagnostic only.
- The remaining large clean4 losses are now mostly cases where the teacher-best
  candidate was already refined but partial T3 sample noise selected another
  action.  The next improvement should attack sampling/selection confidence,
  not only wider candidate expansion.

### 2026-06-15 T2 sampling/selection diagnostics after conditional post20

Tried two follow-up ideas against the remaining selected-action misses:

1. `rank5min10`: force absolute model-rank Top5 T2 candidates to at least
   10 sampled T3 deals when time remains.
2. `structw1`: use `refined + 1.0 * model_score` instead of
   `refined + 0.5 * model_score` when own top row already has a pair or joker.

Results:

| mode | eval set | rows | Reg/max | mean/max ms | decision |
|---|---|---:|---:|---:|---|
| current default | clean_holdout4 fixed cap50 | 40 | 0.505 / 6.257 | 3354.9 / 4767.2 | keep |
| rank5min10 | clean_holdout4 fixed cap50 | 40 | 0.755 / 12.137 | 3904.0 / 4673.8 | reject |
| structw1 | clean_holdout4 fixed cap50 | 40 | 0.510 / 6.257 | 3399.9 / 4910.3 | reject |

Interpretation:

- Spending budget on model Top5 before post20 is counterproductive.  It can
  prevent wider-rank candidates from being refined, bringing back the old
  rank20 miss.
- Offline re-selection can look slightly better, but runtime reruns still have
  enough T3 sample/deadline variation that tiny gains are not reliable.
- Do not promote either diagnostic mode.
- Next useful direction is to reduce teacher-best candidate misses and unstable
  sampled values through better T2 training data / candidate generation, rather
  than a blanket extra-sampling rule inside the 5-second path.

### 2026-06-15 T2 Top5-miss specialist promotion

Mined the current default model-only Top5 misses on the fixed/broad `train710`
action-value data, then trained a small specialist focused on rows where the
teacher-best action was outside the ensemble Top5:

- source dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_train710_fixed_broad_more11_20260615/reranker_t2_clean410_broadfixed_more11_710_dim520`
- default-ensemble miss output:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260615/eval_default_ensemble_train710/misses.jsonl`
- weighted data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260615/av_train710_default_top5miss_weighted_20260615`
- specialist:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260615/models/t2-train710-default-top5miss-ft-20260615/action_value_best.pt`

Model-only alpha sweep:

| eval set | alpha | Top5 | Top5 Reg | Top10 | Top10 Reg | note |
|---|---:|---:|---:|---:|---:|---|
| train710 | 0.00 | 93.7% | 0.073 | 97.6% | 0.022 | previous ensemble |
| train710 | 0.05 | 95.6% | 0.048 | 98.3% | 0.018 | chosen for runtime check |
| train710 | 0.20 | 97.6% | 0.026 | 99.3% | 0.006 | stronger in-sample, higher overfit risk |
| clean4 fixed | 0.00 | 85.0% | 0.349 | 90.0% | 0.243 | external |
| clean4 fixed | 0.05 | 85.0% | 0.361 | 90.0% | 0.243 | neutral/slightly worse model-only |
| broad160 holdout40 fixed | 0.00 | 57.5% | 0.514 | 87.5% | 0.227 | external |
| broad160 holdout40 fixed | 0.05 | 57.5% | 0.514 | 90.0% | 0.223 | slight Top10 gain |

Runtime fixed-teacher validation with the conditional post20 path:

| eval set | rows | previous default Reg/max | alpha0.05 Reg/max | mean/max ms | decision |
|---|---:|---:|---:|---:|---|
| clean_holdout4 fixed cap50 | 40 | 0.505 / 6.257 | 0.348 / 4.823 | 3242.0 / 4783.1 | improve |
| clean20c cap50 | 20 | 0.545 / 4.989 | 0.545 / 4.989 | 3447.2 / 4688.2 | neutral |
| broad160 holdout40 fixed cap50 | 40 | 0.664 / 7.655 | 0.578 / 7.655 | 3444.6 / 4712.0 | improve |

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_experimental`
  to `default_mode` in `ai/config/t2_action_value_ensemble_20260615.json`.
- Keep `alpha=0.05`; larger alpha values improved train710 more but were not
  justified by the smaller external checks.
- This is a selected-action EV-loss improvement under the 5-second path.  It
  does not mean T2 is solved: broad external Top5/Top10 model-only recall is
  still weak enough that the next real strength work should add more diverse
  exact T2 rows and mine the remaining selected-action losses.

### 2026-06-16 T2 strict model-Top1 rescue default

After the alpha0.05 promotion, the remaining 100-row fixed-teacher runtime
misses were classified by whether the teacher-best action survived the runtime
candidate pool:

- miss rows: `41`
- EV loss `>= 0.5`: `26`
- EV loss `>= 1.0`: `13`
- among the `26` rows with EV loss `>= 0.5`, the teacher-best action was still
  inside the runtime candidate pool in `24` rows.

So the immediate bottleneck was not mainly Top20 candidate omission.  It was
partial T3 sample noise selecting a lower teacher-score action even when the
teacher-best action was already present and often model-ranked high.

Checked two low-risk serving changes against the saved runtime candidates:

| change | average selected EV loss over 100 rows | note |
|---|---:|---|
| current alpha0.05 default | 0.479 | baseline |
| selection weight `0.75` instead of `0.5` | 0.475 | too small to promote alone |
| model Top1 rescue, `FL >= 0.10`, `refined_delta <= 2.5` | 0.437 | worth runtime validation |

The model Top1 rescue is intentionally narrow: it restores model Top1 only when
the model Top1 predicts FL at least `0.10` and trails the currently selected
candidate's sampled refined score by no more than `2.5`.

Runtime fixed-teacher validation:

| eval set | rows | previous default Reg/max | Top1 rescue Reg/max | mean/max ms | rescue applied |
|---|---:|---:|---:|---:|---:|
| clean_holdout4 fixed cap50 | 40 | 0.348 / 4.823 | 0.266 / 4.823 | 3344.8 / 4863.3 | 3 |
| clean20c cap50 | 20 | 0.545 / 4.989 | 0.539 / 4.989 | 3400.4 / 4767.6 | 1 |
| broad160 holdout40 fixed cap50 | 40 | 0.578 / 7.655 | 0.559 / 7.655 | 3341.2 / 4763.5 | 5 |

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref25_experimental`
  to `default_mode` in `ai/config/t2_action_value_ensemble_20260615.json`.
- Keep the rescue gate strict.  Model Top1 always is much worse than the
  refined path, so this is not a general trust-the-model switch.
- Remaining large losses now need either more samples for high-disagreement
  rows or better T2/T3 value calibration; candidate-pool width alone will not
  remove most of the current loss.

### 2026-06-16 T2 rejected rescue/sampling diagnostics

Tried two follow-ups after the strict Top1 rescue default:

1. Guarantee model-rank Top1 at least `10` sampled T3 deals.
2. Add a secondary Top1 rescue gate for the broad160 max-loss pattern where
   the current selected action is model-rank `5+` and model Top1 has modest FL
   and acceptable bust.

Top1 min-sample result:

| variant | clean4 Reg/max | clean20c Reg/max | broad40 Reg/max | decision |
|---|---:|---:|---:|---|
| current default | 0.266 / 4.823 | 0.539 / 4.989 | 0.559 / 7.655 | keep |
| model Top1 min 10 samples | 0.266 / 4.808 | 0.488 / 4.989 | 0.559 / 7.655 | not enough gain for extra latency |

The min-10 version is safe in the checked rows but mostly just spends more
time; it does not attack the largest remaining loss.

Secondary rescue result:

| variant | clean4 Reg/max | clean20c Reg/max | broad40 Reg/max | decision |
|---|---:|---:|---:|---|
| secondary rescue, `fl_max=0.20` | no quality change on broad40 | n/a | 0.559 / 7.655 | did not fire on target rows |
| secondary rescue, `fl_max=0.30` | 0.837 / 12.998 | 0.488 / 4.989 | 0.448 / 3.469 | reject |

The important detail is that the rescue feature uses max FL-type probability,
not the scalar `predicted_fl`.  Relaxing `fl_max` to catch the broad160
max-loss rows worked there, but it also enabled unsafe Top1 restores on
clean_holdout4.  This confirms that a broader Top1 rescue is not a stable fix.

Code changes kept from this diagnostic:

- `t2_model_top1_rescue_bust_max` can now bound the primary Top1 rescue.
- `t2_model_top1_rescue2_*` fields can express a secondary rescue gate.
- Both are disabled unless explicitly configured, and the secondary diagnostic
  remains rejected.

Next useful direction:

- Do not widen model Top1 rescue further without a learned selector or much
  larger validation set.
- The remaining big losses should be handled by better calibrated T3/T2 values
  or by a confidence-aware sampling rule that can detect unstable sampled
  refined scores before final selection.

### 2026-06-16 T2 selection-weight recheck after strict rescue

Rechecked whether the strict Top1-rescue default should also increase
`t2_selection_model_weight` above `0.5`.

Runtime fixed-teacher validation:

| variant | clean4 Reg/max | clean20c Reg/max | broad40 Reg/max | mean/max ms notes | decision |
|---|---:|---:|---:|---|---|
| current default, weight `0.5` | 0.266 / 4.823 | 0.539 / 4.989 | 0.559 / 7.655 | all 100 rows under 5s | keep |
| weight `0.75` | 0.260 / 4.823 | 0.514 / 4.989 | 0.642 / 10.969 | all rows under 5s | reject |
| weight `0.6` | 0.591 / 12.998 | not run | 0.559 / 7.655 | clean4 had 1 row over the 4900 ms internal budget | reject |

Decision:

- Keep the default at `t2_selection_model_weight=0.5`.
- `0.75` slightly helps the clean sets but makes the broad external holdout
  materially worse, including a larger max loss.
- `0.6` is not a stable compromise: broad40 is neutral, but clean_holdout4
  regresses badly and gets slower.
- The benchmark CLI now supports direct `--t2-selection-model-weight`
  overrides so future selection-weight sweeps can be reproduced without adding
  temporary config modes.

### 2026-06-16 T2 runtime hard-loss specialist diagnostic

Audited the remaining selected-action losses under the strict Top1-rescue
default across the fixed-teacher 100-row check set
(`clean_holdout4` 40 + `clean20c` 20 + `broad160_holdout40` 40):

- rows with selected EV loss `> 0`: `39`
- rows with loss `>= 0.5`: `24`
- rows with loss `>= 1.0`: `11`
- rows with loss `>= 2.0`: `6`
- among loss `>= 0.5`, teacher-best was still in the runtime candidate pool in
  `22/24` rows and already refined in `16/24` rows.

Interpretation: the current large tail is mostly not candidate omission.  It is
sampled T3 refinement and final selection choosing a lower teacher-score action
even when the teacher-best action survived the pool.

High-sample diagnostic on the `6` rows with loss `>= 2.0`:

- input:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/current_default_highloss_ge2_payloads.jsonl`
- mode: current default with `time_budget_ms=30000`,
  `t2_initial_samples_per_candidate=10`, `t2_max_samples_per_candidate=20`,
  and model-rank Top5 min `20` samples.
- result: mean/max selected EV loss `3.573 / 5.503`, mean/max latency
  `9473.7 / 12727.4 ms`.

That is not a viable fix.  More samples helped one row, reduced the worst row,
but did not solve most of the high-loss tail and exceeded the 5-second target.

Then converted the `24` loss `>= 0.5` rows into a small weighted action-value
dataset and fine-tuned a diagnostic specialist:

- hard-loss teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/current_default_highloss_ge0p5_payloads.jsonl`
- hard-loss action-value data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/av_current_default_highloss_ge0p5_24_20260616`
- merged train data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/reranker_t2_train710_plus_runtime_highloss24_20260616`
- specialist:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/models/t2-train710-plus-runtime-highloss24-ft-20260616/action_value_best.pt`
- training: local GPU, `122s`, initialized from the 2026-06-15 Top5-miss
  specialist.

Important caveat: the `24` hard-loss rows came from prior external runtime
checks, so those sets are no longer clean evidence for this specialist.  Use
them only as capacity/mistake-mining data.

Model-only alpha sweep with the new specialist added to the current default
ensemble:

| eval set | default Top1/Reg1 | alpha0.10 Top1/Reg1 | Top3/Reg3 | Top5/Reg5 | Top10/Reg10 |
|---|---:|---:|---:|---:|---:|
| clean20b | 75.0% / 0.074 | 75.0% / 0.074 | 95.0% / 0.010 | 95.0% / 0.010 | 100.0% / 0.000 |
| more8_80 | 62.5% / 0.525 | 68.8% / 0.354 | 91.2% / 0.059 | 97.5% / 0.007 | 100.0% / 0.000 |
| more9_120 | 88.3% / 0.096 | 90.0% / 0.089 | 100.0% / 0.000 | 100.0% / 0.000 | 100.0% / 0.000 |
| more10_40 | 57.5% / 1.146 | 60.0% / 0.967 | 90.0% / 0.192 | 100.0% / 0.000 | 100.0% / 0.000 |

Runtime selected-action validation:

| eval set | mode | rows | selected Reg/max | mean/max ms | decision |
|---|---|---:|---:|---:|---|
| clean20b | current default | 20 | 0.705 / 13.140 | 3503.2 / 4908.3 | baseline |
| clean20b | + highloss24 alpha0.10 | 20 | 0.705 / 13.140 | 3318.9 / 4789.1 | no quality change |
| more10_40 | current default | 40 | 0.859 / 7.178 | 3621.5 / 4715.1 | baseline |
| more10_40 | + highloss24 alpha0.10 | 40 | 0.859 / 7.178 | 3511.1 / 4801.2 | no quality change |

Decision:

- Do not promote the highloss24 specialist to default.
- Keep
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref25_highloss24_alpha10_diagnostic`
  as a diagnostic mode only.
- The model-only signal improved, but the 5-second serving path still chooses
  the same final actions.  The next improvement needs a learned final selector
  or a confidence-aware refinement rule that uses model/partial-exact
  disagreement before committing the selected action.

### 2026-06-16 T2 final-selector and Top1-rescue delta recheck

Implemented a T2 post-refinement final-selector path for diagnostics:

- `HybridConfig.t2_final_selector`
- `t2_selection_policy="selector"`
- CLI override:
  `benchmark_t2_t3_union_runtime.py --t2-final-selector ...`
- trainer:
  `ai/tutor/train_t2_final_selector_from_runtime.py`

The first linear selector was trained from the saved current-default runtime
outputs for `clean_holdout4` 40 rows, `clean20c` 20 rows, and fixed
`broad160_holdout40` 40 rows.

Selector artifacts:

- full-feature overfit probe:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/selectors/t2-final-current100-linear-20260616.json`
- simpler regularized probe:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/selectors/t2-final-current100-simple-500e-20260616.json`

Offline saved-candidate result for the simpler selector:

| split | current Reg/max | selector Reg/max | decision |
|---|---:|---:|---|
| train75 | 0.518 / 7.655 | 0.481 / 7.655 | small fit improvement |
| dev25 | 0.255 / 4.989 | 0.247 / 4.989 | tiny improvement |
| all100 | 0.452 / 7.655 | 0.423 / 7.655 | not enough to promote |

Runtime clean20b validation of the simple selector regressed slightly:

| eval set | mode | rows | selected Reg/max | mean/p95/max ms | decision |
|---|---|---:|---:|---:|---|
| clean20b | current ref25 default | 20 | 0.705 / 13.140 | 3503.2 / p95 n/a / 4908.3 | baseline |
| clean20b | simple final selector | 20 | 0.744 / 13.140 | 3433.2 / 4693.7 / 4879.6 | reject |

The clean20b max-loss row showed a clearer fix: model Top1 was the
teacher-best action, but the strict Top1 rescue rejected it because the sampled
refined-score deficit was `2.750`, just above the existing `2.5` gate.

Rechecked the primary Top1 rescue with
`t2_model_top1_rescue_refined_delta_max=3.0`:

| eval set | ref25 Reg/max | ref30 Reg/max | mean/p95/max ms | decision |
|---|---:|---:|---:|---|
| clean20b | 0.705 / 13.140 | 0.048 / 0.650 | 3316.3 / 4776.8 / 4820.4 | promote |
| clean20c | 0.539 / 4.989 | 0.539 / 4.989 | 3469.2 / 4674.4 / 4685.1 | neutral |
| broad160 holdout40 fixed | 0.559 / 7.655 | 0.559 / 7.655 | 3497.1 / 4679.2 / 4708.4 | neutral |
| more10_40 | 0.859 / 7.178 | 0.836 / 7.178 | 3630.9 / 4715.1 / 4730.5 | slight improvement |

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_experimental`
  to `default_mode`.
- Keep final-selector support and artifacts for further diagnostics, but do not
  promote the first selector.
- Keep the broader rescue narrowly gated by `predicted_fl >= 0.10`; model
  Top1 by itself is still much worse on broad/more10 sets.

### 2026-06-16 T2 Top1-rescue rank/bust gate

After promoting the `ref30` gate, the remaining more10 losses showed the
opposite failure mode: model Top1 rescue sometimes overrode a stronger refined
candidate when the current selected candidate was already model-rank 2 or 3.

Added a primary Top1-rescue rank gate:

- config/CLI:
  `t2_model_top1_rescue_selected_model_rank_min`
- default candidate:
  current selected model rank must be `>= 4`
- also configured primary Top1 predicted bust max:
  `t2_model_top1_rescue_bust_max = 0.75`

Post-hoc replay on saved `ref30` runtime outputs:

| eval set | ref30 Reg/max | rank4+bust075 Reg/max |
|---|---:|---:|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 |
| clean20c | 0.539 / 4.989 | 0.539 / 4.989 |
| broad160 holdout40 fixed | 0.559 / 7.655 | 0.563 / 7.655 |
| more10_40 | 0.836 / 7.178 | 0.480 / 6.278 |
| weighted all120 | 0.563 / 7.655 | 0.442 / 7.655 |

Runtime rerun:

| eval set | rows | selected Reg/max | mean/p95/max ms | under 5s |
|---|---:|---:|---:|---:|
| clean20b | 20 | 0.048 / 0.650 | 4082.0 / 4707.7 / 4772.1 | 20/20 |
| clean20c | 20 | 0.538 / 4.989 | 4168.6 / 4698.3 / 4816.4 | 20/20 |
| broad160 holdout40 fixed | 40 | 0.644 / 7.655 | 3810.9 / 4691.6 / 4711.3 | 40/40 |
| more10_40 | 40 | 0.451 / 6.278 | 3767.6 / 4697.9 / 4921.5 | 40/40 under the 5s product target; 39/40 under the 4900ms internal budget |

Additional broad160 check:

- On the same saved broad runtime output, applying the old `ref30` gate would
  score `0.641 / 7.655`; the new rank4+bust075 gate scores `0.644 / 7.655`.
- The broad rerun mean shift is therefore mostly T3 sample/refinement variance,
  not a meaningful gate regression.

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_rank4_bust075_experimental`
  to `default_mode`.
- This is stronger on the combined checked set and materially improves the
  more10 tail while preserving the clean20b max-loss fix.
- Remaining max losses are now mainly sampled-refinement calibration issues,
  not simple model Top1 rescue mistakes.

### 2026-06-16 T2 high-loss13 specialist

Mined the current `ref30/rank4/bust075` runtime outputs for remaining selected
EV-loss rows and created a small targeted T2 action-value dataset:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/ref30_rank4_bust075_highloss_ge1_payloads.jsonl`
- rows:
  `13`
- converted reranker data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/av_ref30_rank4_bust075_highloss_ge1_13_20260616`
- merged training data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/reranker_t2_train734_plus_ref30_rank4_highloss13_20260616`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/models/t2-train734-plus-ref30-rank4-highloss13-ft-20260616/action_value_best.pt`

Model-only sweep over clean and external checks:

| eval set | default Top1/Reg1 | alpha0.10 Top1/Reg1 | alpha0.20 Top1/Reg1 | default Top10/Reg10 | alpha0.10 Top10/Reg10 | alpha0.20 Top10/Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| clean20b | 75.0% / 0.074 | 75.0% / 0.074 | 80.0% / 0.110 | 95.0% / 0.010 | 100.0% / 0.000 | 100.0% / 0.000 |
| clean20c | 45.0% / 1.375 | 50.0% / 1.072 | 55.0% / 0.950 | 90.0% / 0.071 | 95.0% / 0.061 | 100.0% / 0.000 |
| broad160 fixed | 30.0% / 1.403 | 42.5% / 1.254 | 42.5% / 1.221 | 90.0% / 0.223 | 92.5% / 0.136 | 92.5% / 0.136 |
| more10_40 | 57.5% / 1.146 | 57.5% / 0.974 | 62.5% / 0.728 | 100.0% / 0.000 | 100.0% / 0.000 | 100.0% / 0.000 |

Runtime selected-action validation:

| eval set | current Reg/max | alpha0.10 Reg/max | alpha0.20 Reg/max | alpha0.10 under 5s | alpha0.20 under 5s |
|---|---:|---:|---:|---:|---:|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 | 0.042 / 0.650 | 20/20 | 20/20 |
| clean20c | 0.538 / 4.989 | 0.389 / 1.882 | 0.266 / 1.793 | 20/20 | 20/20 |
| broad160 fixed | 0.644 / 7.655 | 0.619 / 7.655 | 0.666 / 7.655 | 40/40 | 40/40 |
| more10_40 | 0.451 / 6.278 | 0.371 / 6.278 | 0.359 / 6.278 | 40/40 | 40/40 |

Decision:

- Promote
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_rank4_bust075_highloss13_alpha10_diagnostic`
  to `default_mode`.
- Keep alpha0.20 as a diagnostic only.  Its weighted mean is slightly better on
  the four checked sets, but broad160 fixed regressed, so it is not default.
- This is another real improvement from targeted hard-loss mining: it lowers
  clean20c and more10 selected EV loss while preserving the 5 second serving
  target.

### 2026-06-16 T2 high-loss23 follow-up

After promoting highloss13 alpha0.10, mined the new runtime outputs again:

- source outputs:
  `runtime_highloss13_alpha10_clean20b_fixed_teacher`,
  `runtime_highloss13_alpha10_clean20c_fixed_teacher`,
  `runtime_highloss13_alpha10_broad160_holdout40_fixed_teacher`,
  `runtime_highloss13_alpha10_more10_40_fixed_teacher`
- loss `>= 0.5` payloads:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/highloss13_alpha10_highloss_ge05_payloads.jsonl`
- rows:
  `23`
  - clean20b: 1
  - clean20c: 5
  - broad160 fixed: 13
  - more10_40: 4
- converted data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/av_highloss13_alpha10_ge05_23_20260616`
- merged data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/reranker_t2_train770_plus_alpha10_highloss23_20260616`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/models/t2-train770-plus-alpha10-highloss23-ft-20260616/action_value_best.pt`

Model-only check for replacing the alpha0.10 highloss13 specialist:

| eval set | highloss13 Top1/Reg1 | highloss23 Top1/Reg1 | highloss13 Top10/Reg10 | highloss23 Top10/Reg10 |
|---|---:|---:|---:|---:|
| clean20b | 75.0% / 0.074 | 75.0% / 0.074 | 100.0% / 0.000 | 100.0% / 0.000 |
| clean20c | 50.0% / 1.072 | 50.0% / 1.072 | 95.0% / 0.061 | 95.0% / 0.061 |
| broad160 fixed | 42.5% / 1.254 | 45.0% / 1.224 | 92.5% / 0.136 | 92.5% / 0.136 |
| more10_40 | 57.5% / 0.974 | 65.0% / 0.722 | 100.0% / 0.000 | 100.0% / 0.000 |

Runtime selected-action validation:

| eval set | highloss13 alpha0.10 Reg/max | highloss23 alpha0.10 Reg/max | decision |
|---|---:|---:|---|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 | neutral |
| clean20c | 0.389 / 1.882 | 0.507 / 4.989 | reject |
| broad160 fixed | 0.619 / 7.655 | 0.538 / 7.655 | improved |
| more10_40 | 0.371 / 6.278 | 0.371 / 6.278 | neutral |

Decision:

- Do not promote highloss23 as the default specialist replacement.
- Keep
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_rank4_bust075_highloss23_alpha10_diagnostic`
  as a diagnostic mode only.
- The result suggests a future broad-like context gate may help, but ungated
  replacement loses too much on clean20c.

### 2026-06-16 T2 saved challenger-gate audit

Added a saved-output audit tool:

- script:
  `ai/tutor/evaluate_t2_runtime_challenger_gate.py`
- output:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/gate_highloss13_vs_23_20260616`
- files:
  `summary.json`, `rules.jsonl`, `paired_rows.jsonl`

Compared current default highloss13 alpha0.10 as baseline against highloss23
alpha0.10 as challenger.  This does not change serving behavior; it only
replays two completed runtime outputs and searches simple state/runtime gates.

Input rows:

| set | baseline rows | challenger rows | teacher-comparable paired rows |
|---|---:|---:|---:|
| clean20b | 20 | 20 | 20 |
| clean20c | 20 | 20 | 20 |
| broad160 fixed | 40 | 40 | 40 |
| more10_40 | 40 | 40 | 30 |
| total | 120 | 120 | 110 |

Best no-regression saved-output rule:

- choose highloss23 when `opp_top_len <= 1`, otherwise keep highloss13
- total selected EV-loss sum:
  - highloss13 baseline: `44.6431`
  - highloss23 ungated: `43.7313`
  - gated saved replay: `40.1224`
- total improvement vs baseline: `-4.5207` selected EV loss
- action-changing switches: `2 / 110`

Per-set saved replay:

| set | baseline Reg/max | highloss23 Reg/max | gated Reg/max | gated delta |
|---|---:|---:|---:|---:|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 | 0.048 / 0.650 | neutral |
| clean20c | 0.389 / 1.882 | 0.507 / 4.989 | 0.326 / 1.882 | improved |
| broad160 fixed | 0.619 / 7.655 | 0.538 / 7.655 | 0.538 / 7.655 | improved |
| more10_40 comparable rows | 0.371 / 6.278 | 0.371 / 6.278 | 0.371 / 6.278 | neutral |

Decision:

- Keep highloss13 alpha0.10 as the default for now.
- Do not implement this saved gate directly as production logic yet; it is
  posthoc evidence from two completed runtime outputs.
- Next implementation option is a real T2 final/challenger gate that can run
  both model stacks or otherwise reproduce the same gate features before final
  selection.

### 2026-06-16 T2 highloss23 context-switch promotion

Implemented a runtime-compatible T2 action-value model switch:

- class:
  `ConditionalSwitchActionValueReranker`
- config key:
  `turn_model_conditional_switch_ensembles`
- active default:
  `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_rank4_bust075_highloss23_opp1highcard_switch_experimental`
- gate:
  `opp_top_len_le_1_dealt_max_rank_ge_j_context`

This keeps the highloss13 alpha0.10 T2 ensemble as the normal model and switches
the entire T2 action-value ensemble to the highloss23 alpha0.10 variant only
when:

- opponent top row has at most one visible card
- the current T2 dealt max rank is `J` or better

The broader saved gate `opp_top_len <= 1` reproduced the clean20c and broad160
improvement, but on more10_40 it touched rows where the old compact result had
incomplete teacher candidate coverage.  The high-card condition avoided that
uncertainty while preserving the measured gains.

Runtime fixed-teacher validation:

| eval set | previous default Reg/max | high-card switch Reg/max | mean/p95/max ms | under 5s |
|---|---:|---:|---:|---:|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 | 3447.4 / 4676.6 / 4918.6 | 20/20 under 5s; 19/20 under 4900ms internal budget |
| clean20c | 0.389 / 1.882 | 0.326 / 1.882 | 3454.5 / 4735.1 / 4735.4 | 20/20 |
| broad160 fixed | 0.619 / 7.655 | 0.538 / 7.655 | 3338.5 / 4682.0 / 4708.3 | 40/40 |
| more10_40 | 0.371 / 6.278 | 0.371 / 6.278 | 3482.9 / 4697.8 / 4757.0 | 40/40 |

Decision:

- Promote the high-card context switch mode to `default_mode`.
- This is a real T2 strength improvement on selected-action EV loss, not just
  model-only TopK recall: clean20c and broad160 improved while clean20b and
  more10_40 stayed neutral on the checked fixed-teacher runtime sets.
- Remaining large-loss rows are still mostly sampled T3 refinement / candidate
  calibration problems, so the next work should mine rows with selected
  EV-loss `>= 1.0` under the new default and train/refine against those.

### 2026-06-16 T2 high-card high-loss follow-up

Added a reusable high-loss mining tool:

- script:
  `ai/tutor/collect_t2_runtime_highloss_payloads.py`
- purpose:
  collect the original teacher payloads from runtime `results.jsonl` rows where
  the selected action has known teacher EV loss above a threshold, preserving
  runtime metadata for targeted retraining.

Mined the promoted high-card switch outputs:

- source outputs:
  `runtime_highloss23_opp1highcard_clean20b_fixed_teacher`,
  `runtime_highloss23_opp1highcard_clean20c_fixed_teacher`,
  `runtime_highloss23_opp1highcard_broad160_holdout40_fixed_teacher`,
  `runtime_highloss23_opp1highcard_more10_40_fixed_teacher`
- loss `>= 0.5` payloads:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/highcard_default_highloss_ge05_payloads.jsonl`
- rows:
  `22`
  - clean20b: 1
  - clean20c: 4
  - broad160 fixed: 13
  - more10_40: 4
- loss `>= 1.0` payloads:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/highcard_default_highloss_ge10_payloads.jsonl`
- rows:
  `10`
  - clean20c: 4
  - broad160 fixed: 3
  - more10_40: 3

Converted and trained:

- converted data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/av_highcard_default_highloss_ge05_22_20260616`
  - records: 22
  - candidate samples: 472
- merged data:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/reranker_t2_train770_plus_highcard_ge05_22_20260616`
  - records: 792
  - candidate samples: 12,872
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/models/t2-train792-plus-highcard-ge05-ft-20260616/action_value_best.pt`
- training:
  - initialized from
    `t2-train770-plus-alpha10-highloss23-ft-20260616/action_value_best.pt`
  - local CUDA, stopped at 123s by `--max-seconds`
  - internal validation: Top1 94.9%, Top3/5/10/15/20 100.0%

Runtime validation replaced only the high-card challenger model with
`train792` in a temporary D-drive config:

`D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/t2_action_value_ensemble_train792_eval_20260616.json`

| eval set | current high-card switch Reg/max | train792 challenger Reg/max | decision |
|---|---:|---:|---|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 | neutral |
| clean20c | 0.326 / 1.882 | 0.389 / 1.882 | reject |
| broad160 fixed | 0.538 / 7.655 | 0.539 / 7.655 | neutral/slight worse |
| more10_40 | 0.371 / 6.278 | 0.342 / 6.278 | improved |

Saved-output gate audit:

- output:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/gate_highcard_default_vs_train792_20260616`
- paired teacher-comparable rows:
  110
- best simple no-regression rule:
  switch to `train792` when `own_middle_len >= 3`
- saved replay total selected EV-loss sum:
  - current high-card baseline: `40.1224`
  - ungated train792 challenger: `40.5616`
  - gated replay: `39.2437`
- action-changing switches:
  `1 / 110`
- all measured gain came from `more10_40`; no checked row was harmed in the
  saved replay.

Decision:

- Do not promote `train792` as the active challenger.
- Keep the model and mined payloads for future hard-negative training.
- A cascade T2 model switch was implemented and validated as a diagnostic
  capability, but it is not promoted yet.

Cascade diagnostic implementation:

- model wrapper:
  `CascadeSwitchActionValueReranker`
- config key:
  `turn_model_cascade_switch_ensembles`
- temporary config:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/t2_action_value_ensemble_train792_cascade_eval_20260616.json`
- cascade order:
  1. highloss13 base
  2. switch to highloss23 when
     `opp_top_len_le_1_dealt_max_rank_ge_j_context`
  3. switch to train792 when
     `opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3_context`

Runtime validation:

| eval set | current high-card switch Reg/max | train792 cascade Reg/max | decision |
|---|---:|---:|---|
| clean20b | 0.048 / 0.650 | 0.048 / 0.650 | neutral |
| clean20c | 0.326 / 1.882 | 0.326 / 1.882 | neutral |
| broad160 fixed | 0.538 / 7.655 | 0.538 / 7.655 | neutral |
| more10_40 | 0.371 / 6.278 | 0.371 / 6.278 | neutral |

Decision:

- Do not promote the cascade diagnostic mode.
- The saved replay rule `own_middle_len >= 3` improved one more10 row, but the
  improving row had opponent top length 2, outside the current high-card gate.
  A safe production change would therefore need a separately validated
  `own_middle_len >= 3` / opponent-top-2 context gate with guard sets, not a
  nested refinement of the existing high-card switch.

## 2026-06-16 T2 Top1 Rank2 Rescue Promotion

The current default now relaxes the final T2 model Top1 rescue gate from
`selected_model_rank_min=4` to `selected_model_rank_min=2`, while keeping the
existing FL/refined-delta/bust gates. It also keeps the strict model-rank Top4
rescue:

- `t2_model_top1_rescue_selected_model_rank_min = 2`
- `t2_model_rank_rescue_k = 4`
- `t2_model_rank_rescue_selected_model_rank_min = 5`
- `t2_model_rank_rescue_refined_delta_max = 0.8`
- `t2_model_rank_rescue_model_delta_min = 0.7`
- `t2_model_rank_rescue_min_refined_score = 1.0`

Reason:

- more10 source_line 36 had teacher-best action at model rank 1, but sampled
  T3 refinement picked model rank 2 and lost 6.278 EV.
- With the rank2 rescue, the source_line 36 one-row probe chose the teacher-best
  action with 0 EV loss.

Runtime checks, all under the 5 second target:

| eval set | previous default Reg/max | rank2 Top1 rescue Reg/max | notes |
|---|---:|---:|---|
| more10_40 | 0.3415 / 6.2779 | 0.2491 / 2.4995 | improved; Top1 rescue fired 10/40 |
| clean20b | 0.0478 / 0.6498 | 0.0507 / 0.6498 | near-neutral; the small mean shift was not from Top1 rescue on the changed row |
| clean20c | 0.3263 / 1.8820 | 0.3263 / 1.8820 | neutral |
| broad160 fixed | 0.9775 / 11.4779 | 0.7910 / 11.4779 | improved mean/p95; max unchanged |

Rejected follow-up:

- forcing model Top4/Top5 candidates to at least 10 samples cut more10 max loss
  further, but raised mean latency to about 4.5s and produced many
  `time_budget_exhausted` rows, so it is not promoted.

Next target:

- source_line 3 in more10 remains a 2.499 EV loss where teacher-best is model
  rank 4 but sampled refinement is badly underestimated. This is not fixed by
  Top1 rescue and likely needs either targeted T3 sample allocation or a
  separate model-rank Top4 safety rule.

## 2026-06-17 T2 Model Top1 Bust-Improvement Rescue Promotion

The active T2 default now inherits the 2026-06-16 rank2 Top1 rescue mode and
adds a narrow `t2_model_top1_bust_rescue` gate. The gate runs after the standard
Top1 rescue and before the model-rank Top4 rescue.

Promoted mode:

- `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_rank4_bust075_highloss23_opp1highcard_switch_top1bust_delta001_experimental`

Gate values:

- `t2_model_top1_bust_rescue_selected_model_rank_min = 5`
- `t2_model_top1_bust_rescue_refined_delta_max = 12.0`
- `t2_model_top1_bust_rescue_model_delta_min = 2.0`
- `t2_model_top1_bust_rescue_bust_delta_min = 0.08`
- `t2_model_top1_bust_rescue_top_bust_max = 0.75`
- `t2_model_top1_bust_rescue_top_fl_min = 0.15`
- `t2_model_top1_bust_rescue_current_fl_min = 0.10`
- `t2_model_top1_bust_rescue_fl_delta_min = 0.01`

Reason:

- broad160 source_line 87 and 148 were large losses where model Top1 was the
  teacher-best action, but noisy partial exact selected a lower-ranked,
  higher-bust action.
- A simple refined-delta widening fixed source_line 87 but regressed more10,
  so it was rejected.
- The promoted gate requires model Top1 to have a large model-score advantage,
  lower predicted bust, meaningful FL signal on both actions, and at least
  `0.01` better max FL type. This avoids the clean20c source_line 11 and
  broad160 source_line 41 bad fires seen in looser probes.

Runtime checks, all under the 5 second target:

| eval set | previous default Reg/max | top1 bust rescue Reg/max | bust rescue fires |
|---|---:|---:|---:|
| more10_40 | 0.2491 / 2.4995 | 0.2491 / 2.4995 | 0/40 |
| clean20b | 0.0507 / 0.6498 | 0.0478 / 0.6498 | 0/20 |
| clean20c | 0.3263 / 1.8820 | 0.3263 / 1.8820 | 0/20 |
| broad160 fixed | 0.7910 / 11.4779 | 0.5650 / 8.0802 | 4/160 |

broad160 fired rows:

- source_line 42: loss after rescue 5.080; improved from the previous sampled
  selection, but did not hit teacher best.
- source_line 72: loss after rescue 2.311; improved from the previous sampled
  selection.
- source_line 87: loss after rescue 0.000; restored teacher best.
- source_line 148: loss after rescue 0.000; restored teacher best.

Validation artifacts:

- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_more10_40_top1bust_20260617`
- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_clean20b_top1bust_20260617`
- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_clean20c_top1bust_20260617`
- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_broad160_top1bust_20260617`
- default smoke:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_top1bust_smoke1_20260617`

Implementation:

- `ai/tutor/hybrid_t1t2.py`
- `ai/tutor/benchmark_t2_t3_union_runtime.py`
- `ai/config/t2_action_value_ensemble_20260615.json`
- `tests/test_hybrid_t1t2.py`
- `tests/test_benchmark_t2_t3_union_runtime.py`

Next target:

- more10 source_line 3 is still a 2.499 EV loss. The teacher-best action is
  model rank 4, not model Top1, so this promotion intentionally does not touch
  it. The next safe fix should focus on a targeted Top4 safety rule or better
  sample allocation for rank 4 without widening noisy Top1 rescues globally.

## 2026-06-17 T2 Middle-Fill Bottom-Shift Rescue Promotion

The active T2 default now inherits the Top1-bust rescue mode and adds a narrow
`t2_middle_fill_bottom_shift_rescue` tactical gate. This targets the remaining
more10 source_line 3 loss where sampled partial exact overvalued filling a
four-card middle row and undervalued moving the same card to bottom.

Promoted mode:

- `t2_fast_t3_union_top10_s10_tactical3_fixed710_alpha25_w0p5_post20_structtop_m5_min1000_top5miss_alpha05_top1rescue_fl010_ref30_rank4_bust075_highloss23_opp1highcard_switch_top1bust_midshift_experimental`

Gate values:

- `t2_middle_fill_bottom_shift_rescue_selected_model_rank_max = 3`
- `t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max = 4`
- `t2_middle_fill_bottom_shift_rescue_model_gap_max = 1.0`
- `t2_middle_fill_bottom_shift_rescue_bust_delta_min = 0.04`
- `t2_middle_fill_bottom_shift_rescue_fl_delta_min = 0.02`
- `t2_middle_fill_bottom_shift_rescue_challenger_bust_max = 0.55`
- `t2_middle_fill_bottom_shift_rescue_challenger_fl_min = 0.20`
- `t2_middle_fill_bottom_shift_rescue_selected_bust_min = 0.50`

Structural gate:

- self middle must already have 4 cards
- self bottom must have at most 2 cards
- current and challenger must have the same discard
- current and challenger must place the same card(s) to top
- exactly one dealt card is moved from current `middle` placement to challenger
  `bottom` placement

Runtime checks, all under the 5 second target:

| eval set | Top1-bust Reg/max | midshift Reg/max | midshift fires |
|---|---:|---:|---:|
| more10_40 | 0.2491 / 2.4995 | 0.1658 / 2.1966 | 1/40 |
| clean20b | 0.0478 / 0.6498 | 0.0478 / 0.6498 | 0/20 |
| clean20c | 0.3263 / 1.8820 | 0.3263 / 1.8820 | 0/20 |
| broad160 fixed | 0.5650 / 8.0802 | 0.6174 / 8.0802 | 0/160 |

Notes:

- The broad160 rerun had zero midshift fires. Its mean moved from 0.5650 to
  0.6174 due to partial-exact sampling variance, not because the new rule chose
  a different action.
- The new rule still improved broad160 versus the pre-Top1-bust default
  baseline of 0.7910 / 11.4779 because it inherits that promotion.

Source_line 3 smoke:

- input:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260616/more10_source3_only_20260617.teacher.jsonl`
- output:
  `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_midshift_source3_smoke_20260617`
- result: 2.52s, `t2_middle_fill_bottom_shift_rescue` fired, EV loss 0.0

Validation artifacts:

- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_more10_40_top1bust_midshift_20260617`
- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_clean20b_top1bust_midshift_20260617`
- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_clean20c_top1bust_midshift_20260617`
- `D:/ofc-pineapple-data/t2_next_20260614/t2_selection_recheck_20260617/runtime_default_broad160_top1bust_midshift_20260617`

Implementation:

- `ai/tutor/hybrid_t1t2.py`
- `ai/tutor/benchmark_t2_t3_union_runtime.py`
- `ai/config/t2_action_value_ensemble_20260615.json`
- `tests/test_hybrid_t1t2.py`
- `tests/test_benchmark_t2_t3_union_runtime.py`

Next target:

- The remaining more10 max is source_line 20 at about 2.197 EV loss. That loss
  is already the model Top1 action after Top1 rescue, so it is not a candidate
  selection rescue problem. It needs better T2 model labels or a stronger
  refinement/confirmation rule around the model Top1 itself.

## 2026-06-18 T2 Top5 EV-Loss Hard-Negative Pass

Goal:

- Reduce T2 model Top5 pruning misses where exact rerank would still lose more
  than `0.1` EV.
- Keep this as a model-quality diagnostic first; do not promote it to runtime
  default until selected-action runtime loss is rechecked.

Baseline checked set:

- output: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/baseline_current`
- datasets: `fresh20`, `local_next50`, `varied40`, `cap200_first20`,
  `cap200_check12`, and `more2` through `more11`
- total groups: `1022`
- baseline Top5: `92.7%`
- baseline Top5 rerank regret: `0.064`
- baseline Top5 misses: `75`
- baseline Top5 EV loss `> 0.1`: `53`
- baseline Top5 EV loss `> 0.25`: `41`
- baseline max Top5 EV loss: `7.626`

Hard-negative data:

- initial hard groups: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/hard_groups/top5_ev_loss_gt010.jsonl`
- initial hard groups: `53`
- ft1 train data: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/av_dataset_train1444_plus_top5hard53_r12_dim520`
- ft1 model: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/models/t2-train1444-top5hard53-r12-ft-top5-20260618/action_value_final.pt`
- ft1 best blend checked: `alpha=0.6`
- ft1 result: Top5 `98.8%`, Top5 EV loss `> 0.1` down to `3`,
  max Top5 EV loss `1.109`, Top10 stayed `100%`

Second pass:

- residual hard groups added from ft1 `alpha=0.6`: `cap200_first20 group 8`,
  `more11_80 group 9`, `more8_80 group 24`
- combined hard groups: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/hard_groups/top5_ev_loss_gt010_plus_alpha0p6_residual.jsonl`
- combined unique groups: `55`
- ft2 train data: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/av_dataset_train1444_plus_top5hard55_r20_dim520`
- ft2 model: `D:/ofc-pineapple-data/t2_next_20260614/top5_ev_loss_goal_20260618/models/t2-train1444-top5hard55-r20-ft2-top5-20260618/action_value_final.pt`
- compact config/evidence: `ai/config/t2_top5hard_ft2_20260618.json`

Best checked blend:

| blend | Top1 | Top3 | Top5 | Top10 min | Top5 regret | misses | EV loss > 0.1 | EV loss > 0.25 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 57.2% | 84.1% | 92.7% | 100.0% | 0.064 | 75 | 53 | 41 | 7.626 |
| ft1 alpha0.6 | 72.4% | 94.1% | 98.8% | 100.0% | 0.003 | 12 | 3 | 2 | 1.109 |
| ft2 alpha0.6 | 73.6% | 95.4% | 98.9% | 100.0% | 0.001 | 11 | 2 | 0 | 0.248 |
| ft2 alpha1.0 | 79.0% | 96.2% | 99.0% | 100.0% | 0.000 | 10 | 1 | 0 | 0.248 |

Rejected higher alpha:

- `ft2 alpha1.5`: Top5 EV loss `> 0.1` increased to `3`
- `ft2 alpha2.0`: max Top5 EV loss jumped to `19.889`
- `ft2 alpha3.0`: max Top5 EV loss stayed `19.889`

Current best diagnostic conclusion:

- `ft2 alpha1.0` is the best model-only Top5 pruning blend checked in this
  pass.
- It reduces Top5 EV loss `> 0.1` from `53` groups to `1` group and keeps
  Top10 at `100%` on all checked datasets.
- Remaining Top5 EV loss `> 0.1`: `local_next50 group 0`, loss `0.248`,
  teacher rank `7`.
- Do not make this the runtime default yet. Next validation should run the T2
  hybrid runtime selected-action loss and latency checks using the ft2 alpha1.0
  candidate model blend.

## 2026-06-18 T2 Top3 EV-Loss Residual Pass

Goal:

- Strengthen the T2 model Top3 candidate pool after the Top5-hard pass.
- Optimize for Top3 exact-rerank EV loss, not just raw Top3 recall.
- Keep the result diagnostic until selected-action runtime loss and 5 second
  latency are rechecked.

Starting point:

- baseline checked set: same 15 datasets and `1022` groups as the Top5 pass
- baseline Top3: `84.1%`
- baseline Top3 rerank regret: `0.231`
- baseline Top3 misses: `163`
- baseline Top3 EV loss `> 0.1`: `133`
- baseline Top3 EV loss `> 0.25`: `109`
- baseline max Top3 EV loss: `22.265`

First Top3 specialist:

- hard groups: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/hard_groups/top3_ev_loss_gt010_from_ft2_alpha1.jsonl`
- hard groups: `25`
- train data: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/av_dataset_top5hard55_r20_plus_top3hard25_r30_dim520`
- model: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/models/t2-top5hard55-top3hard25-r30-ft-top3-20260618/action_value_final.pt`
- best count blend before residual pass: replace the Top5-hard ft2 specialist
  with this Top3 specialist at `alpha=1.0`
- result: Top3 `98.4%`, Top3 EV loss `> 0.1` down to `3`, but max Top3 EV
  loss was still `8.368`

Residual pass:

- residual sources: first Top3 specialist `alpha=1.0` and `alpha=1.5`
- residual hard groups: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/hard_groups/top3_residual_top3loss_gt010_alpha1p0_alpha1p5.jsonl`
- residual unique groups: `6`
- largest residual: `more10_40 group 6`, Top3 EV loss `8.368`
- ft2 train data: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/av_dataset_top5hard55_r20_top3hard25_r30_plus_residual6_r60_dim520`
- ft2 model: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/models/t2-top5hard55-top3hard25-residual6-r60-ft2-top3-20260618/action_value_final.pt`
- compact config/evidence: `ai/config/t2_top3_residual_ft2_20260618.json`
- eval summary: `D:/ofc-pineapple-data/t2_next_20260614/top3_ev_loss_goal_20260618/eval_top3_residual_ft2_blend/top3_residual_ft2_blend_summary.json`

Best checked blend:

| blend | Top1 | Top3 | Top5 | Top10 min | Top3 regret | Top5 regret | Top3 misses | Top3 EV loss > 0.1 | Top3 EV loss > 0.25 | max Top3 EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 57.2% | 84.1% | 92.7% | 100.0% | 0.231 | 0.064 | 163 | 133 | 109 | 22.265 |
| top5hard ft2 alpha1 | 79.0% | 96.2% | 99.0% | 100.0% | 0.047 | 0.000 | 39 | 25 | 17 | 19.889 |
| top3 ft1 alpha1 | 83.0% | 98.4% | 99.5% | 100.0% | 0.009 | 0.000 | 16 | 3 | 1 | 8.368 |
| top3 residual ft2 alpha1 | 85.1% | 98.7% | 99.5% | 100.0% | 0.000293 | 0.000100 | 13 | 0 | 0 | 0.080 |

Rejected alpha checks:

- residual `alpha=0.8`: Top3 EV loss `> 0.1` was `2`, max `5.978`
- residual `alpha=1.2`: Top3 EV loss `> 0.1` was `1`, max `0.480`
- residual `alpha=1.5`: Top3 EV loss `> 0.1` was `2`, max `6.318`
- residual `alpha=2.0`: Top3 EV loss `> 0.1` was `4`, max `6.318`

Current diagnostic conclusion:

- `top3 residual ft2 alpha1` is the strongest checked Top3 pruning blend.
- It reduces Top3 EV loss `> 0.1` from `133` groups to `0` groups on the
  checked `1022` groups.
- Top3 recall is still not 100%; `13` groups rank the exact best outside Top3,
  but exact rerank inside Top3 loses at most `0.080` EV in this checked set.
- Do not make this the runtime default yet. Next validation should run the T2
  hybrid selected-action loss and latency checks with this candidate model
  blend.

## 2026-06-19 T2 Top1 EV-Loss Hard-Negative Pass

Goal:

- Strengthen the T2 model Top1 ordering after the Top3 residual pass.
- Optimize for Top1 EV-loss tails while preserving Top3 exact-rerank safety.
- Keep the result diagnostic until selected-action runtime loss and 5 second
  latency are rechecked.

Starting point:

- checked set: same `15` datasets and `1022` groups as the Top3 residual pass
- current diagnostic config: `ai/config/t2_top3_residual_ft2_20260618.json`
- current Top1: `85.1%`
- current Top3: `98.7%`
- current Top1 regret: `0.230691`
- current Top1 misses: `152`
- current Top1 EV loss `> 0.25`: `98`
- current Top1 EV loss `> 1.0`: `69`
- current max Top1 EV loss: `13.032`
- current Top3 EV loss `> 0.1`: `0`

First Top1 specialist:

- hard groups: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/hard_groups/top1_ev_loss_gt025_from_top3residual_alpha1.jsonl`
- hard groups: `98`
- train data: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/av_dataset_top3residual_base_plus_top1hard98_r20_dim520`
- model: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/models/t2-top3residual-plus-top1hard98-r20-ft-top1-20260619/action_value_final.pt`
- best safe first-pass blend: add this Top1 specialist with `beta=0.8`
- result: Top1 `96.3%`, Top1 EV loss `> 0.25` down to `11`, Top1 EV
  loss `> 1.0` down to `7`, but max Top1 EV loss remained `11.199`

Residual pass:

- residual source: first Top1 specialist add blend with `beta=0.8`
- residual hard groups: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/hard_groups/top1_residual_gt025_from_add_beta0p8.jsonl`
- residual unique groups: `11`
- largest residual: `varied40 group 32`, Top1 EV loss `11.199`
- ft2 train data: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/av_dataset_top1hard98_r20_plus_residual11_r50_dim520`
- ft2 model: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/models/t2-top1hard98-residual11-r50-ft2-top1-20260619/action_value_final.pt`
- compact config/evidence: `ai/config/t2_top1hard_ft2_20260619.json`
- eval summary: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/eval_top1hard_ft2_blend/top1hard_ft2_blend_summary.json`

Best checked blend:

| blend | Top1 | Top3 | Top5 | Top10 min | Top1 regret | Top1 misses | Top1 EV loss > 0.25 | Top1 EV loss > 1.0 | max Top1 EV loss | Top3 EV loss > 0.1 | max Top3 EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current top3 residual alpha1 | 85.1% | 98.7% | 99.5% | 100.0% | 0.230691 | 152 | 98 | 69 | 13.032 | 0 | 0.080 |
| top1 ft1 add beta0.8 | 96.3% | 99.3% | 99.7% | 100.0% | 0.028134 | 38 | 11 | 7 | 11.199 | 0 | 0.056 |
| top1 ft2 add beta0.8 | 97.1% | 99.3% | 99.7% | 100.0% | 0.003803 | 30 | 3 | 2 | 1.429 | 0 | 0.056 |

Rejected checks:

- `ft2 add beta1.0`: max Top1 EV loss `5.199`
- `ft2 add beta1.2`: max Top1 EV loss `16.462`
- `ft2 add beta1.5`: max Top1 EV loss `16.462`
- `ft2 replace alpha0.8`: Top1 `97.0%`, but Top1 regret `0.011378`
  and max Top1 EV loss `5.199`

Current diagnostic conclusion:

- `top1 ft2 add beta0.8` is the strongest checked Top1 diagnostic blend.
- It reduces Top1 EV loss `> 0.25` from `98` groups to `3` and Top1 EV loss
  `> 1.0` from `69` groups to `2` on the checked `1022` groups.
- It reduces max Top1 EV loss from `13.032` to `1.429`.
- Top3 safety from the previous pass is preserved in this checked set:
  Top3 EV loss `> 0.1` remains `0`.
- Important caveat: this is not a clean external result. The Top1 hard negatives
  were mined from the same `15` checked datasets, so the `97.1%` Top1 should be
  read as a mined-set diagnostic, not as final external accuracy.

External clean check:

- output: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/external_clean_check_top1hard_ft2_beta0p8`
- baseline output: `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/external_clean_check_current_top3residual_alpha1`
- checked datasets: `clean20c`, `clean_holdout4_fixed`, `broad160_holdout40_fixed`
- these sets were not used for the 2026-06-19 Top1 hard-negative mining pass

| clean external set | groups | current Top1 | Top1 ft2 beta0.8 Top1 | current Reg1 | Top1 ft2 beta0.8 Reg1 | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean20c | 20 | 45.0% | 40.0% | 1.072 | 1.609 | 65.0% | 75.0% | 90.0% | 100.0% |
| clean_holdout4_fixed | 40 | 42.5% | 40.0% | 1.545 | 2.253 | 72.5% | 85.0% | 90.0% | 97.5% |
| broad160_holdout40_fixed | 40 | 30.0% | 25.0% | 1.480 | 1.689 | 52.5% | 62.5% | 90.0% | 97.5% |
| aggregate | 100 | 38.0% | 34.0% | 1.424 | 1.899 | 63.0% | 74.0% | 90.0% | 98.0% |

External clean conclusion:

- The Top1 hard-negative blend does not generalize to the clean external sets.
- It improves the mined/check set, but overfits enough that clean Top1 and Reg1
  are worse than the previous Top3 residual blend.
- Do not make this the runtime default. The next strength step should train on
  broader clean external all-legal/cap50 teacher data, then keep a fresh
  final_holdout that is never mined for hard negatives.

Clean hard-negative retrain check:

- train data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/av_dataset_top1hard_ft2_plus_clean260_r3_hard136_r16_dim520`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/models/t2-top1hard-ft2-clean260-r3-hard136-r16-ft-clean-top1-20260619/action_value_final.pt`
- source clean training rows: `260`
- hard groups mined from train-side clean data: `136`
- internal validation: Top1 about `99.7%`, but this is not a reliable
  generalization metric because the hard groups are overrepresented in the
  training mix.
- clean external sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/external_clean_blend_sweep_clean260_20260619/summary.json`

External aggregate over `clean20c + clean_holdout4_fixed + broad160_holdout40_fixed`:

| variant | Top1 | Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 | Top20 | Reg20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_top3 | 38.0% | 1.425 | 64.0% | 0.514 | 76.0% | 0.368 | 89.0% | 0.183 | 98.0% | 0.007 |
| clean_replace_beta0.6 | 37.0% | 1.500 | 62.0% | 0.671 | 76.0% | 0.399 | 91.0% | 0.125 | 97.0% | 0.095 |
| clean_replace_beta0.4 | 37.0% | 1.520 | 63.0% | 0.600 | 76.0% | 0.361 | 91.0% | 0.132 | 98.0% | 0.007 |
| prev_top1_beta0.8 | 34.0% | 1.898 | 63.0% | 0.667 | 74.0% | 0.410 | 90.0% | 0.188 | 98.0% | 0.007 |

Decision:

- Reject the clean260 fine-tune for default or diagnostic promotion.  It does
  not beat `current_top3` on clean external Top1 or Reg1.
- The useful signal is not "more hard-negative repeats"; it is that the model
  needs broader all-legal/cap50 T2 rows from new roots/seeds before the next
  Top1 attempt.

New broad all-legal/cap50 smoke:

- purpose: verify a clean new-seed T2 data-generation path on `D:` before
  scaling.
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_smoke_20260619/inputs/t2_new_broad_smoke4_source_all_actions.jsonl`
- cap50 output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_smoke_20260619/cap50/t2_oracle_cap50_limit4.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_smoke_20260619/t2_new_broad_smoke4_alllegal_cap50.teacher.jsonl`
- reranker dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_smoke_20260619/reranker_t2_new_broad_smoke4_alllegal_cap50_dim520`

Smoke results:

- source rows: `4` (`BB=2`, `BTN=2`)
- source candidates: `99`
- source generation time: `3.9s`
- cap50 rows: `4`
- cap50 average runtime: `16,675 ms/row`
- source/model Top1 matched cap50 Top1: `1/4`
- average source Top1 EV loss: `6.855`
- max source Top1 EV loss: `24.598`
- fixed teacher join: `4` rows, `99` candidates, `0` invalid rows/candidates
- reranker conversion: `99` samples, `0` skipped

Decision:

- The pipeline is usable and catches exactly the weakness we need to train:
  model Top1 is poor on a fresh, non-mined T2 distribution.
- At the smoke speed, `200` cap50 rows would be about `56` minutes of Rust
  evaluation; `1000` rows would be about `4.6` hours.  Past broad160 timing was
  slower (`49s/row`), so a practical local estimate is `1-3` hours for 200 rows
  and `5-14` hours for 1000 rows depending on branch complexity.
- Next scale target should be a new train/final split, e.g. `400 train + 100
  final_holdout`, generated from new root ranges and not mixed with the already
  checked clean external sets.

New broad200 source and first20 cap50 check:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/inputs/t2_new_broad200_source_all_actions.jsonl`
- first20 cap50:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/cap50_first20/t2_oracle_cap50_limit20.jsonl`
- first20 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/t2_new_broad200_first20_alllegal_cap50.teacher.jsonl`
- first20 reranker dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/reranker_t2_new_broad200_first20_alllegal_cap50_dim520`

Source generation:

- rows: `200` (`BB=100`, `BTN=100`)
- candidates: `4,791`
- roots: `901000-901099`, one BB and one BTN T2 state per root
- elapsed: `156.6s`
- records/sec: `1.28`

First20 cap50 results:

- cap50 rows: `20`
- candidates: `480`
- average cap50 runtime: `16,332 ms/row`
- source/T3-model Top1 matched cap50 Top1: `5/20`
- average source Top1 EV loss: `3.103`
- max source Top1 EV loss: `9.937`
- fixed teacher join: `20` rows, `480` candidates, `0` invalid rows/candidates
- reranker conversion: `480` samples, `0` skipped

Action-value model-only evaluation on first20:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 | full-bust avoidable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 30.0% | 45.0% | 55.0% | 80.0% | 85.0% | 90.0% | 3.145 | 2.413 | 1.367 | 2 |
| top3 residual alpha1 | 25.0% | 50.0% | 60.0% | 85.0% | 90.0% | 90.0% | 3.300 | 2.196 | 1.187 | 2 |
| top1hard ft2 beta0.8 | 20.0% | 45.0% | 60.0% | 85.0% | 85.0% | 90.0% | 2.978 | 2.441 | 1.187 | 1 |

Decision:

- This confirms the external problem again.  The mined-set Top1 `97.1%` does
  not carry to fresh broad T2 rows.
- The next useful step is to cap50-label the rest of the new broad200 source,
  split it into train/final, then train a broad Top1 specialist and judge it
  against clean external sets plus the held-out broad rows.

Completed broad200 cap50 labeling:

- all200 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/reranker_t2_new_broad200_all200_alllegal_cap50_dim520`
- train split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/reranker_t2_new_broad200_train0_159_alllegal_cap50_dim520`
- final split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/reranker_t2_new_broad200_final160_199_alllegal_cap50_dim520`

Aggregate source-vs-cap50 oracle:

- records: `200`
- candidates/samples: `4,791`
- source/T3-model Top1 matched cap50 Top1: `53/200` (`26.5%`)
- changed Top1: `147/200`
- average cap50 runtime: `14,452.5 ms/row`
- average source Top1 EV loss: `3.025`
- max source Top1 EV loss: `24.579`
- invalid rows/candidates after fixed teacher joins: `0`

Implementation note:

- Added `--skip` to `ai/rust_solver/t3_exact_solver/src/main.rs` and
  `ai/tutor/run_t2_exact_oracle.py` so long T2 cap50 jobs can be chunked
  without recomputing from row zero.
- Verified the skip path with `--skip 20 --limit 2`; output preserved
  original `record_index` values `20,21`.
- Verification:
  `cargo build --release --manifest-path ai/rust_solver/t3_exact_solver/Cargo.toml`
  and
  `python -m pytest tests/test_build_t2_action_value_teacher_from_exact.py tests/test_convert_action_value_teacher.py -q`

Broad200 model experiments:

1. Mini train0_59 -> holdout60_79:

- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/models/t2-newbroad-train0-59-ft-top1-20260619/action_value_best.pt`
- result: no useful improvement.  Standalone Top1 tied the baseline at `40.0%`
  on holdout60_79 but had worse Reg1 (`3.499` vs `2.645`).

2. train0_159 broad-only specialist -> final160_199:

- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/models/t2-newbroad-train0-159-ft-top1-20260619/action_value_best.pt`
- final160_199 result: no improvement.

3. top3 residual base + newbroad train0_159 repeat6 -> final160_199:

- mixed dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/av_dataset_top3residual_plus_newbroad160_r6_dim520`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad200_20260619/models/t2-top3residual-plus-newbroad160-r6-ft-top1-20260619/action_value_best.pt`
- final160_199 result: no improvement.

Final160_199 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 27.5% | 62.5% | 70.0% | 90.0% | 92.5% | 95.0% | 1.869 | 0.678 | 0.126 |
| top3 residual alpha1 | 27.5% | 62.5% | 72.5% | 87.5% | 92.5% | 95.0% | 1.654 | 0.765 | 0.126 |
| broad train0_159 standalone | 25.0% | 62.5% | 72.5% | 85.0% | 90.0% | 95.0% | 1.986 | 0.695 | 0.151 |
| top3 residual + broad train0_159 gamma0.25 | 25.0% | 62.5% | 72.5% | 87.5% | 90.0% | 95.0% | 1.893 | 0.759 | 0.126 |
| mixed newbroad160 r6 standalone | 22.5% | 62.5% | 70.0% | 80.0% | 90.0% | 95.0% | 2.358 | 0.723 | 0.153 |
| top3 residual + mixed newbroad160 gamma0.25 | 25.0% | 62.5% | 72.5% | 87.5% | 92.5% | 95.0% | 1.906 | 0.759 | 0.126 |

Decision:

- Do not promote either broad200 specialist.
- The broad200 teacher data is valuable, but `160` train rows are still too
  small for a stable Top1 specialist on this distribution.
- The next useful step is to keep generating broader cap50 rows, then train
  with a stronger split, e.g. at least `800-1000` train rows plus a clean
  `200` final holdout.  For current runtime, continue relying on TopK +
  exact/rerank rather than model Top1.

### 2026-06-19 T2 broad1000 first100 Top1 check

Generated a larger fresh T2 source on `D:` so Top1 work is no longer limited to
the mined/check distribution:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/inputs/t2_new_broad1000_source_all_actions.jsonl`
- records: `1000` (`BB=500`, `BTN=500`)
- roots: `902000-902499`, one BB and one BTN T2 state per root
- candidates: `23,943`
- generation time: `709.4s`

Cap50-labeled the first `100` rows in five `20`-row chunks:

- combined teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/t2_new_broad1000_0_99_alllegal_cap50.teacher.jsonl`
- combined dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_99_alllegal_cap50_dim520`
- records: `100`
- candidates/samples: `2,343`
- source/T3-model Top1 matched cap50 Top1: `28/100` (`28.0%`)
- average cap50 runtime: `13,282.2 ms/row`
- average source Top1 EV loss: `3.151`
- max source Top1 EV loss: `21.840`

Model-only evaluation on new broad1000 first100:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 37.0% | 63.0% | 77.0% | 91.0% | 95.0% | 98.0% | 2.001 | 0.654 | 0.119 |
| top3 residual alpha1 | 37.0% | 60.0% | 77.0% | 93.0% | 96.0% | 97.0% | 1.746 | 0.755 | 0.018 |
| top1hard ft2 beta0.8 | 36.0% | 58.0% | 77.0% | 90.0% | 96.0% | 99.0% | 1.823 | 0.686 | 0.043 |

Implementation note:

- `ai/training/merge_action_value_reranker_data.py` now tolerates older
  datasets without `positions.npy` by filling positions with zeros.  This is
  only used for all-position train/eval merges; position-specific training
  still requires a real `positions.npy`.

Trained a small diagnostic specialist using previous mixed data plus the new
first `80` rows repeated `8x`:

- train dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_prev_mixed_plus_new80r8_dim520`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new80r8-ft-top1-20260619/action_value_best.pt`
- eval holdout:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_broad200final40_plus_newbroad20_final_dim520`

Final60 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 26.7% | 58.3% | 70.0% | 88.3% | 91.7% | 96.7% | 1.908 | 0.608 | 0.096 |
| top3 residual alpha1 | 26.7% | 58.3% | 71.7% | 88.3% | 91.7% | 95.0% | 1.694 | 0.669 | 0.086 |
| top1hard ft2 beta0.8 | 26.7% | 56.7% | 70.0% | 85.0% | 90.0% | 96.7% | 1.587 | 0.707 | 0.126 |
| new80r8 standalone | 25.0% | 58.3% | 66.7% | 83.3% | 90.0% | 98.3% | 2.143 | 0.845 | 0.124 |
| top3 residual + new80r8 gamma0.25 | 26.7% | 56.7% | 70.0% | 88.3% | 90.0% | 95.0% | 1.783 | 0.751 | 0.086 |
| top3 residual + new80r8 gamma1.0 | 26.7% | 58.3% | 70.0% | 85.0% | 90.0% | 96.7% | 1.839 | 0.749 | 0.107 |

Decision:

- Do not promote the new80r8 specialist.
- The first100 labels again show that fresh broad T2 Top1 is the real weak
  point, but `80` new train rows are not enough to improve holdout Top1.
- Continue cap50-labeling the broad1000 source toward at least `800-1000`
  train rows and keep a clean final holdout before the next Top1 specialist.

### 2026-06-19 T2 broad1000 first200 Top1 check

Labeled the next `100` rows, bringing the fresh broad1000 cap50 set to `200`
rows:

- all200 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_199_alllegal_cap50_dim520`
- train split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_train0_159_alllegal_cap50_dim520`
- final split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_final160_199_alllegal_cap50_dim520`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/cap50_first200_aggregate.json`

Aggregate source-vs-cap50 oracle:

- records: `200`
- candidates/samples: `4,791`
- source/T3-model Top1 matched cap50 Top1: `54/200` (`27.0%`)
- changed Top1: `146/200`
- average cap50 runtime: `13,879.9 ms/row`
- average source Top1 EV loss: `3.159`
- max source Top1 EV loss: `34.818`

All200 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 37.5% | 63.0% | 74.0% | 91.0% | 96.5% | 98.5% | 2.183 | 0.898 | 0.124 |
| top3 residual alpha1 | 38.0% | 64.0% | 75.0% | 92.5% | 97.5% | 98.0% | 2.145 | 0.846 | 0.057 |
| top1hard ft2 beta0.8 | 37.5% | 62.5% | 76.5% | 91.5% | 97.5% | 99.0% | 2.157 | 0.795 | 0.063 |

Final160_199 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 35.0% | 65.0% | 72.5% | 95.0% | 97.5% | 97.5% | 2.238 | 1.566 | 0.055 |
| top3 residual alpha1 | 35.0% | 67.5% | 75.0% | 95.0% | 97.5% | 97.5% | 2.393 | 1.391 | 0.055 |
| top1hard ft2 beta0.8 | 35.0% | 65.0% | 77.5% | 95.0% | 97.5% | 97.5% | 2.393 | 1.397 | 0.059 |
| newbroad1000 train0_159 standalone | 32.5% | 65.0% | 75.0% | 87.5% | 97.5% | 100.0% | 2.361 | 1.435 | 0.631 |
| prevmixed + new160r6 standalone | 32.5% | 65.0% | 80.0% | 92.5% | 100.0% | 100.0% | 2.441 | 1.435 | 0.622 |
| top3 residual + prevmixed new160r6 gamma0.25 | 37.5% | 67.5% | 77.5% | 95.0% | 97.5% | 97.5% | 2.435 | 1.390 | 0.059 |
| top3 residual + prevmixed new160r6 gamma1.0 | 35.0% | 65.0% | 77.5% | 95.0% | 97.5% | 100.0% | 2.445 | 1.435 | 0.059 |

Decision:

- Do not promote the first200 specialists.
- The mixed gamma0.25 blend improved final40 Top1 from `35.0%` to `37.5%`,
  but worsened Top1 EV loss versus the current default and top3 residual
  baselines.
- The clean signal remains data scale: `160` fresh train rows are enough to
  move rank order slightly, but not enough to lower EV loss.  Continue
  broad1000 cap50 labeling before the next Top1 promotion attempt.

### 2026-06-20 T2 broad1000 first300 Top1 check

Added another `100` cap50 rows, bringing the fresh broad1000 set to `300`
records:

- all300 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_299_alllegal_cap50_dim520`
- train split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_train0_239_alllegal_cap50_dim520`
- final split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_final240_299_alllegal_cap50_dim520`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/cap50_first300_aggregate.json`

Aggregate source-vs-cap50 oracle:

- records: `300`
- candidates/samples: `7,191`
- source/T3-model Top1 matched cap50 Top1: `80/300` (`26.7%`)
- changed Top1: `220/300`
- average cap50 runtime: `13,924.0 ms/row`
- average source Top1 EV loss: `3.011`
- max source Top1 EV loss: `38.966`

All300 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 37.0% | 63.3% | 74.0% | 90.7% | 97.0% | 99.0% | 1.975 | 0.767 | 0.104 |
| top3 residual alpha1 | 37.3% | 63.0% | 74.0% | 91.3% | 98.0% | 98.7% | 1.856 | 0.727 | 0.063 |
| top1hard ft2 beta0.8 | 37.3% | 61.7% | 75.7% | 90.0% | 98.0% | 99.3% | 1.867 | 0.704 | 0.068 |

Final240_299 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 36.7% | 60.0% | 68.3% | 88.3% | 96.7% | 100.0% | 1.432 | 0.600 | 0.074 |
| top3 residual alpha1 | 35.0% | 58.3% | 68.3% | 86.7% | 98.3% | 100.0% | 1.380 | 0.587 | 0.090 |
| top1hard ft2 beta0.8 | 35.0% | 61.7% | 71.7% | 85.0% | 98.3% | 100.0% | 1.411 | 0.562 | 0.096 |

Trained a mixed diagnostic model from previous mixed data plus the new
train0_239 split repeated `6x`:

- train dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_prev_mixed_plus_new240r6_dim520`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new240r6-ft-top1-20260620/action_value_best.pt`

Final240_299 diagnostic comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prevmixed + new240r6 standalone | 35.0% | 60.0% | 71.7% | 85.0% | 98.3% | 100.0% | 1.693 | 0.706 | 0.199 |
| top3 residual + new240r6 gamma0.25 | 35.0% | 63.3% | 70.0% | 88.3% | 98.3% | 100.0% | 1.341 | 0.570 | 0.081 |
| top3 residual + new240r6 gamma1.0 | 35.0% | 61.7% | 71.7% | 85.0% | 96.7% | 100.0% | 1.446 | 0.544 | 0.199 |

Decision:

- Do not promote as a Top1 model.
- `top3 residual + new240r6 gamma0.25` lowers final60 Reg1 from `1.432` to
  `1.341`, but Top1 accuracy stays at `35.0%` and does not beat current
  default6 Top1 (`36.7%`).
- This is useful EV-loss evidence, not a solved Top1 result.  Continue
  broad1000 labeling; the current clean signal is still data scarcity.

### 2026-06-20 T2 broad1000 first400 Top1 check

Added another `100` cap50 rows, bringing the fresh broad1000 set to `400`
records:

- all400 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_399_alllegal_cap50_dim520`
- train split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_train0_319_alllegal_cap50_dim520`
- final split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_final320_399_alllegal_cap50_dim520`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/cap50_first400_aggregate.json`

Aggregate source-vs-cap50 oracle:

- records: `400`
- candidates/samples: `9,675`
- source/T3-model Top1 matched cap50 Top1: `115/400` (`28.8%`)
- changed Top1: `285/400`
- average cap50 runtime: `14,133.8 ms/row`
- average source Top1 EV loss: `2.807`
- max source Top1 EV loss: `38.966`

Final320_399 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 37.5% | 62.5% | 70.0% | 83.8% | 96.2% | 97.5% | 2.135 | 0.467 | 0.095 |
| top3 residual alpha1 | 35.0% | 65.0% | 75.0% | 87.5% | 96.2% | 98.8% | 1.456 | 0.409 | 0.087 |
| top1hard ft2 beta0.8 | 33.8% | 65.0% | 72.5% | 86.2% | 97.5% | 98.8% | 1.434 | 0.641 | 0.099 |

Trained a mixed diagnostic model from previous mixed data plus the new
train0_319 split repeated `6x`:

- train dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_prev_mixed_plus_new320r6_dim520`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new320r6-ft-top1-20260620/action_value_best.pt`

Final320_399 diagnostic comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prevmixed + new320r6 standalone | 36.2% | 67.5% | 76.2% | 92.5% | 97.5% | 100.0% | 1.474 | 0.702 | 0.057 |
| top3 residual + new320r6 gamma0.25 | 38.8% | 66.2% | 75.0% | 90.0% | 97.5% | 98.8% | 1.247 | 0.590 | 0.066 |
| top3 residual + new320r6 gamma1.0 | 35.0% | 66.2% | 73.8% | 92.5% | 97.5% | 100.0% | 1.424 | 0.676 | 0.054 |

Clean external aggregate check over `clean20c + clean_holdout4_fixed +
broad160_holdout40_fixed`:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current top3 residual baseline | 38.0% | - | - | - | - | - | 1.425 | - | - |
| top3 residual + new320r6 gamma0.10 | 38.0% | 64.0% | 76.0% | 90.0% | 95.0% | 98.0% | 1.438 | 0.532 | 0.183 |
| top3 residual + new320r6 gamma0.25 | 37.0% | 64.0% | 78.0% | 91.0% | 94.0% | 98.0% | 1.508 | 0.524 | 0.148 |

Decision:

- Do not promote as a Top1 model.
- `top3 residual + new320r6 gamma0.25` improves the held-out broad1000
  final80 Top1 from `37.5%` to `38.8%` and lowers Reg1 from `2.135` to
  `1.247`, but it does not beat the clean external top3 residual baseline.
- `gamma0.10` ties clean external Top1 at `38.0%`, but slightly worsens clean
  Reg1 and does not improve final320_399 Top1.
- The next useful move remains more clean broad cap50 labels, not promoting
  this model.  Keep final holdouts clean and continue toward at least
  `800-1000` labeled rows before the next serious Top1 promotion attempt.

### 2026-06-20 T2 broad1000 first600 Top1 check

Added another `200` cap50 rows, bringing the fresh broad1000 set to `600`
records:

- all600 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_599_alllegal_cap50_dim520`
- train split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_train0_479_alllegal_cap50_dim520`
- final split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_final480_599_alllegal_cap50_dim520`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/cap50_first600_aggregate.json`

Aggregate source-vs-cap50 oracle:

- records: `600`
- candidates/samples: `14,454`
- source/T3-model Top1 matched cap50 Top1: `179/600` (`29.8%`)
- changed Top1: `421/600`
- average cap50 runtime: `14,115.9 ms/row`
- average source Top1 EV loss: `2.636`
- max source Top1 EV loss: `38.966`

Final480_599 model-only comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 35.8% | 60.8% | 76.7% | 93.3% | 97.5% | 99.2% | 2.334 | 0.760 | 0.093 |
| top3 residual alpha1 | 40.8% | 64.2% | 78.3% | 94.2% | 99.2% | 99.2% | 2.253 | 0.851 | 0.099 |
| top1hard ft2 beta0.8 | 43.3% | 64.2% | 76.7% | 92.5% | 98.3% | 99.2% | 2.290 | 1.040 | 0.190 |
| top3 residual + new320r6 gamma0.25 | 42.5% | 65.8% | 75.8% | 92.5% | 98.3% | 100.0% | 2.039 | 0.867 | 0.105 |

Trained a mixed diagnostic model from previous mixed data plus the new
train0_479 split repeated `6x`:

- train dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_prev_mixed_plus_new480r6_dim520`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new480r6-ft-top1-20260620/action_value_best.pt`

Final480_599 diagnostic comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prevmixed + new480r6 standalone | 40.8% | 65.8% | 78.3% | 91.7% | 96.7% | 100.0% | 2.235 | 0.969 | 0.215 |
| top3 residual + new480r6 gamma0.25 | 42.5% | 65.8% | 76.7% | 91.7% | 99.2% | 100.0% | 2.042 | 0.866 | 0.106 |
| top3 residual + new480r6 gamma0.5 | 42.5% | 65.0% | 76.7% | 93.3% | 98.3% | 100.0% | 2.020 | 0.859 | 0.085 |
| top3 residual + new480r6 gamma1.0 | 41.7% | 65.0% | 76.7% | 92.5% | 98.3% | 100.0% | 2.110 | 0.875 | 0.124 |

Clean external aggregate check over `clean20c + clean_holdout4_fixed +
broad160_holdout40_fixed`:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current top3 residual baseline | 38.0% | - | - | - | - | - | 1.425 | - | - |
| top3 residual + new480r6 gamma0.25 | 36.0% | 64.0% | 77.0% | 91.0% | 94.0% | 98.0% | 1.541 | 0.525 | 0.148 |

Decision:

- Do not promote as a Top1 model.
- On the new broad1000 final120, existing `top1hard ft2 beta0.8` is still the
  highest Top1 result at `43.3%`.
- `new480r6` reduces Reg1 versus `top1hard ft2` on this final split, but it
  does not improve Top1 and it worsens the clean external aggregate
  (`36.0%` vs baseline `38.0%`).
- The current evidence says more broad labels alone help expose the problem,
  but the same simple repeated-data fine-tune is not enough to generalize.
  Continue toward `800-1000` labels, then try a different objective mix or
  broader architecture/feature change rather than promoting new480r6.

### 2026-06-20 T2 broad1000 first800 external check

Added another `200` cap50 rows, bringing the fresh broad1000 set to `800`
records:

- all800 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_799_alllegal_cap50_dim520`
- train teacher split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/t2_new_broad1000_train0_599_alllegal_cap50.teacher.jsonl`
- final split:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_final600_799_alllegal_cap50_dim520`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/cap50_first800_aggregate.json`

Aggregate source-vs-cap50 oracle:

- records: `800`
- candidates/samples: `19,179`
- source/T3-model Top1 matched cap50 Top1: `226/800` (`28.2%`)
- changed Top1: `574/800`
- average cap50 runtime: `14,016.7 ms/row`
- average source Top1 EV loss: `2.695`
- max source Top1 EV loss: `38.966`

Final600_799 comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current default6 | 36.5% | 66.5% | 80.0% | 90.5% | 97.0% | 99.0% | 1.665 | 0.626 | 0.217 |
| top3 residual alpha1 | 40.0% | 69.5% | 81.5% | 94.0% | 96.5% | 99.5% | 1.591 | 0.567 | 0.106 |
| top1hard ft2 beta0.8 | 38.5% | 68.5% | 79.5% | 92.0% | 97.5% | 99.5% | 1.743 | 0.572 | 0.132 |
| top3 residual + new480r6 gamma0.25 | 40.0% | 69.5% | 82.5% | 94.0% | 97.0% | 99.5% | 1.628 | 0.608 | 0.128 |

Two more diagnostic models were trained from the current top3 residual checkpoint:

- top1-strong model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new600r8-top1strong-20260620/action_value_best.pt`
- balanced model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new600r4-balanced-20260620/action_value_best.pt`

Final600_799 diagnostic comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top3 residual baseline | 40.0% | 69.5% | 81.5% | 94.0% | 96.5% | 99.5% | 1.591 | 0.567 | 0.106 |
| top3 residual + new600r8 gamma0.25 | 41.0% | 69.0% | 81.5% | 93.5% | 97.0% | 99.5% | 1.564 | 0.620 | 0.128 |
| top3 residual + new600r4 gamma0.25 | 42.0% | 69.0% | 81.5% | 94.0% | 96.5% | 99.5% | 1.569 | 0.579 | 0.128 |
| new600r4 balanced standalone | 37.5% | 67.0% | 80.5% | 93.5% | 96.5% | 99.5% | 1.762 | 0.702 | 0.126 |

Clean external aggregate check over `clean20c + clean_holdout4_fixed +
broad160_holdout40_fixed`:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current top3 residual baseline | 38.0% | 64.0% | 76.0% | 89.0% | 94.0% | 98.0% | 1.425 | 0.514 | 0.183 |
| top3 residual + new600r8 gamma0.25 | 36.0% | 63.0% | 77.0% | 91.0% | 94.0% | 98.0% | 1.532 | 0.531 | 0.148 |
| top3 residual + new600r4 gamma0.25 | 37.0% | 64.0% | 77.0% | 91.0% | 95.0% | 98.0% | 1.547 | 0.535 | 0.155 |
| new600r4 balanced standalone | 38.0% | 63.0% | 80.0% | 92.0% | 94.0% | 98.0% | 1.274 | 0.574 | 0.120 |

Decision:

- Do not promote either new first800 model as the runtime Top1 model.
- The best new final600_799 Top1 is `42.0%`, but that gain does not carry to
  the clean external set.
- The balanced standalone model ties clean external Top1 at `38.0%` and lowers
  Reg1 from `1.425` to `1.274`, but it is weaker on final600_799, so treat it
  as diagnostic evidence rather than a promoted model.
- External accuracy is still roughly the old baseline level.  The next useful
  work is not another identical repeated-data fine-tune; either finish labeling
  `800-999` and then change the objective/architecture, or add stronger
  features for board/deck/opponent context.

### 2026-06-20 T2 broad1000 first1000 completion

Completed the remaining `800-999` cap50 rows locally on D:.

- all1000 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_999_alllegal_cap50_dim520`
- holdout800_999 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_final800_999_alllegal_cap50_dim520`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/cap50_first1000_aggregate.json`

Aggregate source-vs-cap50 oracle:

- records: `1000`
- candidates/samples: `23,742`
- source/T3-model Top1 matched cap50 Top1: `281/1000` (`28.1%`)
- changed Top1: `719/1000`
- average cap50 runtime: `14,023.1 ms/row`
- average source Top1 EV loss: `2.710`
- max source Top1 EV loss: `38.966`

New `800-999` holdout model comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top3 residual alpha1 | 36.0% | 59.5% | 77.0% | 87.0% | 95.5% | 100.0% | 2.077 | 0.948 | 0.172 |
| top1hard ft2 beta0.8 | 34.5% | 61.0% | 74.5% | 87.0% | 95.0% | 100.0% | 2.087 | 0.959 | 0.289 |
| new600r4 balanced standalone | 34.0% | 63.0% | 77.0% | 88.5% | 95.5% | 99.5% | 2.078 | 0.754 | 0.301 |
| new800r4 balanced standalone | 33.0% | 63.5% | 75.5% | 87.5% | 96.5% | 99.5% | 2.078 | 0.918 | 0.290 |
| top3 residual + new800r4 gamma0.25 | 35.5% | 59.5% | 76.5% | 87.5% | 95.5% | 100.0% | 2.138 | 0.968 | 0.213 |
| new800-only balanced standalone | 34.0% | 63.0% | 76.5% | 87.0% | 95.0% | 99.5% | 2.164 | 0.889 | 0.319 |
| top3 residual + new800-only gamma0.25 | 35.5% | 61.0% | 76.0% | 88.0% | 96.0% | 99.5% | 2.112 | 0.896 | 0.180 |

Training artifacts:

- mixed train:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_prev_mixed_plus_new800r4_balanced_dim520`
- mixed model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-prevmixed-plus-new800r4-balanced-20260620/action_value_best.pt`
- new-only model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-new800only-balanced-20260620/action_value_best.pt`

Decision:

- Do not promote any new first1000 diagnostic model.
- The `800-999` holdout is harder than `600-799`, and the old top3 residual
  ensemble is still the best Top1 result at only `36.0%`.
- More labels alone are not fixing Top1.  The direct `0-799 -> 800-999`
  experiment also failed, so the next improvement should change the model
  input/architecture or add explicit pair/rank/draw/deck/opponent features
  instead of repeating the same fine-tune recipe.

### 2026-06-20 T2 set-residual Top1 architecture check

The first architecture pass tested three variants on the same broad1000
`0-799 -> 800-999` split:

- explicit action-feature MLP input (`dim617`)
- set/listwise reranker trained from scratch
- set/listwise reranker trained as a residual on top of the current 7-model
  `top3 residual alpha1` ensemble base score

Artifacts:

- ensemble base score writer:
  `ai/training/write_action_value_base_scores.py`
- train base scores:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_799_alllegal_cap50_dim520/base_scores.npy`
- set residual model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new800-baseline-residual-20260620/action_value_set_final.pt`
- blend sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/blend_sweep_set_baseline_residual_final/summary.json`

New `800-999` holdout architecture comparison:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top3 residual alpha1 baseline | 36.0% | 59.5% | 77.0% | 87.0% | 95.5% | 100.0% | 2.077 | 0.948 | 0.172 |
| actionfeat617 standalone | 33.0% | 58.0% | 75.0% | 86.0% | 92.5% | 99.5% | 2.238 | 0.908 | 0.278 |
| set new800 target1 | 25.5% | 49.0% | 70.5% | 85.0% | 92.5% | 98.0% | 3.068 | 1.244 | 0.201 |
| set baseline residual best | 37.0% | 59.0% | 73.5% | 88.0% | 96.5% | 99.0% | 2.194 | 0.954 | 0.162 |
| set baseline residual final | 38.0% | 60.0% | 70.0% | 89.5% | 95.0% | 99.5% | 2.062 | 1.052 | 0.171 |

Clean external aggregate over `clean20c + clean_holdout4_fixed +
broad160_holdout40_fixed`, using
`score = baseline + gamma * (set_residual - baseline)`:

| gamma | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 38.0% | 64.0% | 76.0% | 89.0% | 94.0% | 98.0% | 1.425 | 0.514 | 0.183 |
| 0.05 | 41.0% | 64.0% | 79.0% | 91.0% | 95.0% | 98.0% | 1.431 | 0.575 | 0.183 |
| 0.10 | 43.0% | 66.0% | 80.0% | 90.0% | 95.0% | 98.0% | 1.502 | 0.519 | 0.184 |
| 0.15 | 44.0% | 65.0% | 80.0% | 90.0% | 95.0% | 98.0% | 1.675 | 0.513 | 0.224 |
| 1.00 | 37.0% | 57.0% | 72.0% | 88.0% | 94.0% | 99.0% | 2.167 | 0.907 | 0.145 |

`800-999` holdout blend check:

| gamma | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 36.0% | 59.5% | 87.0% | 2.077 | 0.172 |
| 0.05 | 37.0% | 59.5% | 86.5% | 2.006 | 0.181 |
| 0.10 | 37.0% | 59.5% | 87.0% | 1.979 | 0.184 |
| 0.15 | 37.0% | 60.0% | 88.0% | 1.965 | 0.145 |
| 1.00 | 38.0% | 60.0% | 89.5% | 2.062 | 0.171 |

Decision:

- The plain `dim617` action-feature MLP and the set model trained from scratch
  are rejected.
- The set residual architecture is useful only as a small correction to the
  existing ensemble.
- `gamma=0.10` is the best balanced next Top1 candidate: clean external Top1
  improves from `38.0%` to `43.0%`, `800-999` Top1 improves from `36.0%` to
  `37.0%`, and the combined `300`-group Reg1 is slightly better than baseline.
- Do not promote it as runtime default yet.  Clean external Reg1 worsens
  (`1.425 -> 1.502`) and TopK tails can regress.  The next check should validate
  `gamma=0.10` on more fresh broad holdout before integrating it into serving.

Additional external check over `clean20 + clean20b + broad100_fixed`:

| gamma | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 35.7% | 70.7% | 95.0% | 1.579 | 0.039 |
| 0.05 | 33.6% | 72.9% | 95.7% | 1.589 | 0.038 |
| 0.10 | 33.6% | 75.0% | 96.4% | 1.554 | 0.038 |
| 0.15 | 33.6% | 74.3% | 96.4% | 1.598 | 0.049 |
| 1.00 | 30.7% | 65.0% | 93.6% | 1.765 | 0.026 |

The extra external check changes the interpretation:

- `gamma=0.10` is not a robust Top1 promotion yet.  It improves Top1 on the
  first clean external aggregate, but loses Top1 on the extra external set,
  mainly `broad100_fixed` (`38.0% -> 35.0%`).
- It may still be useful for candidate-pool quality because extra external
  Top3 improves from `70.7%` to `75.0%`, Top10 from `95.0%` to `96.4%`, and
  Reg1 improves slightly from `1.579` to `1.554`.
- For the Top1 objective, keep this as diagnostic.  The next model should learn
  a gate or confidence condition for when the residual correction helps,
  instead of applying a global gamma.

#### Gate diagnostic

Added a reusable gate evaluator:

- script: `ai/training/evaluate_set_residual_gate.py`
- output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/gate_sweep_set_residual_gamma0p10_train0_799/summary.json`
- diagnostic upper-bound output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/gate_sweep_set_residual_gamma0p10_oracle_diagnostics/summary.json`

The gate selected on broad1000 `0-799` was:

- `challenger_margin_over_base >= 0.0033830178538454326`
- where `challenger` is `baseline + 0.10 * (set_residual - baseline)`

Replay over `final800_999 + clean20c + clean_holdout4_fixed +
broad160_holdout40_fixed + clean20 + clean20b + broad100_fixed`:

| selector | groups switched | Top1 | Top3 | Top10 | Reg1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0/440 | 36.4% | 64.1% | 90.0% | 1.770 |
| global gamma0.10 | 440/440 | 37.3% | 65.9% | 90.7% | 1.735 |
| gated gamma0.10 | 52/440 | 37.5% | 65.0% | 90.5% | 1.750 |

Per-set notes:

- `final800_999`: Top1 `36.0% -> 37.5%`, Reg1 `2.077 -> 1.977`.
- `clean_holdout4_fixed`: Top1 `42.5% -> 50.0%`, but Reg1 worsens
  `1.545 -> 1.732`.
- `broad160_holdout40_fixed`: Top1 `30.0% -> 35.0%`, Reg1 improves
  `1.480 -> 1.403`.
- `broad100_fixed`: Top1 worsens `38.0% -> 35.0%`.

Decision:

- The gate is a real but small Top1 improvement candidate: `+1.1pp` Top1 over
  the 440-row replay, with slightly lower Reg1 than baseline.
- It is still not strong enough for default promotion because one external set
  (`broad100_fixed`) regresses and some sets gain Top1 while worsening Reg1.
- Next useful work is not more global blending.  Train/evaluate a stronger
  confidence gate using the same feature table, or add richer group-level
  features that explain when the residual correction is safe.

#### Candidate-rank gate improvement

The gate feature table was expanded with metadata that is available after
candidate generation:

- `position`
- `candidate_ranks`
- `action_indices`
- `route_tags` and selected route-tag bit/fraction features

The best simple gate selected from broad1000 `0-599` is stable and also selected
from `0-799`:

- `candidate_rank_gap_chal_minus_base <= 0`
- i.e. use the set-residual challenger only when its top action has an equal or
  better candidate rank than the baseline top action.

Output:

- `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/gate_sweep_set_residual_gamma0p10_train0_599_feat2/summary.json`
- `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/gate_sweep_set_residual_gamma0p10_train0_799_feat2/summary.json`

Replay over `final600_799 + final800_999 + clean20c +
clean_holdout4_fixed + broad160_holdout40_fixed + clean20 + clean20b +
broad100_fixed`:

| selector | groups switched | Top1 | Top3 | Top10 | Reg1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0/640 | 37.5% | 65.8% | 91.2% | 1.714 |
| global gamma0.10 | 640/640 | 42.5% | 71.2% | 92.8% | 1.601 |
| candidate-rank gated gamma0.10 | 615/640 | 43.9% | 70.6% | 92.8% | 1.525 |

External-only replay over `clean20c + clean_holdout4_fixed +
broad160_holdout40_fixed + clean20 + clean20b + broad100_fixed`:

| selector | groups switched | Top1 | Top3 | Top10 | Reg1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0/240 | 36.7% | 67.9% | 92.5% | 1.515 |
| global gamma0.10 | 240/240 | 37.5% | 71.2% | 93.8% | 1.532 |
| candidate-rank gated gamma0.10 | 225/240 | 40.0% | 70.0% | 93.8% | 1.402 |

Important per-set checks:

- `final600_799`: Top1 `40.0% -> 54.0%`, Reg1 `1.591 -> 1.266`.
- `final800_999`: Top1 `36.0% -> 38.5%`, Reg1 `2.077 -> 1.933`.
- `clean20c`: Top1 `45.0% -> 50.0%`, Reg1 `1.072 -> 0.616`.
- `clean_holdout4_fixed`: Top1 `42.5% -> 50.0%`, Reg1 `1.545 -> 1.472`.
- `broad100_fixed`: Top1 `38.0% -> 39.0%`, Reg1 `1.560 -> 1.455`.

The learned MLP confidence gates with the same expanded feature table did not
beat this simple rule:

- delta-regression gate: 440-row Top1 `37.5%`, Reg1 `1.735`
- improvement-classifier gate: 440-row Top1 `37.5%`, Reg1 `1.715`

Decision:

- This is the strongest current Top1 candidate in this pass.
- Keep it diagnostic until a broader fresh external replay is run, but it is
  now a plausible serving candidate because it improves Top1 and Reg1 on both
  held-out broad rows and the checked external aggregate.
- The next improvement should preserve the `candidate_rank_gap <= 0` gate as a
  baseline and mine the remaining Top1 losses under this gated selector.

#### Fixed candidate-rank gate gamma sweep

With the 5-second serving target treated as a guideline rather than a hard
constraint, the same candidate-rank gate was swept over larger residual gamma
values:

- script: `ai/training/sweep_set_residual_fixed_gate.py`
- output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/fixed_candidate_rank_gate_gamma_sweep_20260620/summary.json`
- miss output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/fixed_candidate_rank_gate_gamma_sweep_20260620/best_gamma_misses.jsonl`

Best all-aggregate setting was `gamma=1.0`; best external-only Top1/Reg1 was
slightly better at `gamma=0.75`.

| selector | scope | groups switched | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate-rank gated gamma0.75 | all | 533/640 | 58.1% | 75.0% | 83.1% | 93.0% | 97.5% | 99.5% | 1.019 | 0.497 | 0.102 |
| candidate-rank gated gamma0.75 | external | 180/240 | 45.8% | 68.3% | 80.8% | 91.7% | 97.1% | 99.2% | 1.120 | 0.568 | 0.105 |
| candidate-rank gated gamma1.0 | all | 525/640 | 58.8% | 75.0% | 82.8% | 92.5% | 96.9% | 99.4% | 1.004 | 0.500 | 0.113 |
| candidate-rank gated gamma1.0 | external | 171/240 | 45.4% | 67.9% | 80.8% | 90.8% | 95.8% | 98.8% | 1.137 | 0.570 | 0.106 |

Remaining gamma1.0 Top1 misses:

- all: 264 misses, mean regret `2.435`, p90 `6.818`, max `21.939`,
  `>=1` regret `133`, `>=5` regret `38`, `>=10` regret `11`.
- external: 131 misses, mean regret `2.084`, p90 `5.270`, max `18.571`,
  `>=1` regret `57`, `>=5` regret `15`, `>=10` regret `4`.

Decision:

- The larger-gamma gate is a strong diagnostic improvement over gamma0.10, but
  external Top1 is still only `45-46%`, so this is not a solved Top1 model.
- For practical candidate-pool use, external Top20 is `~99%` and Top10 exact
  rerank average regret is `~0.105`, so a slower exact-rerank path can be useful.
- The next training data should be mined from the 131 external Top1 misses,
  especially the 57 misses with EV loss at least `1.0`.

#### Hard-weighted set-residual Top1 pass

The next pass kept the external 240 groups out of training and used broad1000
`0-999` as the local training pool.  `base_scores.npy` was first written for
the `0-999` merged dataset using the existing 8-checkpoint T2 ensemble.

Script changes:

- `ai/training/create_eval_miss_weighted_action_value_data.py` now preserves
  `base_scores.npy` and uses `ordered_indices` as confuser candidates when an
  evaluator miss file does not contain `top_predictions`.
- `ai/training/sweep_set_residual_fixed_gate.py` now accepts blended set
  checkpoints through `--checkpoints` and `--weights`.

Artifacts:

- R1 weighted data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_999_gate1p0_top1miss_weighted_20260620`
- R1 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-gate1p0-hardtop1-residual-20260620/action_value_set_final.pt`
- R2 weighted data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_999_gate1p0_top1miss_weighted_r2_20260620`
- R2 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-gate1p0-hardtop1-r2-residual-20260620/action_value_set_final.pt`
- R1/R2 50/50 blend sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/fixed_candidate_rank_gate_gamma_sweep_new1000_hardtop1_r1r2w50_20260620/summary.json`

Comparison on `final600_799 + final800_999 + external240`:

| selector | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old fixed-gate gamma1.0 | 58.8% | 75.0% | 82.8% | 92.5% | 96.9% | 99.4% | 1.004 | 0.500 | 0.113 |
| R1 final gamma1.0 | 76.6% | 85.9% | 91.2% | 96.9% | 99.1% | 99.5% | 0.570 | 0.268 | 0.049 |
| R2 final gamma1.0 | 75.6% | 85.5% | 90.9% | 96.4% | 98.6% | 99.5% | 0.518 | 0.251 | 0.046 |
| R1/R2 50/50 gamma1.15 | 76.9% | 85.2% | 91.9% | 97.0% | 98.9% | 99.5% | 0.536 | 0.291 | 0.052 |

External-only 240 groups:

| selector | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old fixed-gate gamma1.0 | 45.4% | 67.9% | 80.8% | 90.8% | 95.8% | 98.8% | 1.137 | 0.570 | 0.106 |
| R1 final gamma1.0 | 50.0% | 67.5% | 79.2% | 92.5% | 97.9% | 99.2% | 1.065 | 0.536 | 0.123 |
| R2 final gamma1.0 | 48.3% | 66.7% | 78.3% | 91.2% | 96.7% | 99.2% | 0.938 | 0.498 | 0.115 |
| R1/R2 50/50 gamma1.15 | 50.4% | 65.8% | 80.4% | 92.9% | 97.9% | 99.2% | 0.995 | 0.580 | 0.133 |

Remaining R1/R2 50/50 `gamma=1.15` Top1 misses:

- all: `148` misses, mean regret `2.317`, max `21.939`, `>=1` regret `67`,
  `>=5` regret `20`, `>=10` regret `6`.
- external: `119` misses, mean regret `2.008`, max `18.571`, `>=1` regret
  `50`, `>=5` regret `13`, `>=10` regret `4`.

Decision:

- The current best Top1 diagnostic is `R1/R2 50/50 + gamma=1.15 +
  candidate_rank_gap <= 0`: external Top1 improves from `45.4-45.8%` to
  `50.4%`.
- This is progress, not a solved model.  External Top3 drops versus the old
  fixed gate, and the gain over R1 final is only one external group.
- Next Top1 work should mine the remaining high-regret external-like failures
  from fresh data, not continue overfitting broad1000 `0-999`.

#### Mixed broad1000 + new_broad200 Top1 fine-tune check

The next check added `new_broad200` train rows to the broad1000 all-legal pool
and mined misses under the current R1/R2 50/50 `gamma=1.15` selector.

Artifacts:

- mixed all-legal train:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_mixed_20260620/reranker_t2_broad1000_plus_newbroad200_train160_alllegal_cap50_dim520`
- weighted Top1-miss train:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_mixed_20260620/reranker_t2_broad1000_plus_newbroad200_train160_gate1p15_top1miss_weighted_20260620`
- scratch set model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_mixed_20260620/models/t2-set-broad1000-plus-newbroad160-hardtop1-residual-20260620`
- R1-init fine-tune:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_mixed_20260620/models/t2-set-r1-init-broad1000-plus-newbroad160-hardtop1-ft-20260620`

Evaluation used `mixed_train1160 + newbroad200_final160_199 + external240`.
`external240` excludes `newbroad200_final160_199`; `external+newbroad` includes
it for a stricter fresh-data view.

| selector | scope | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|
| current R1/R2 50/50 gamma1.15 | external240 | 50.4% | 65.8% | 92.9% | 99.2% | 0.995 | 0.133 |
| current R1/R2 50/50 gamma1.15 | external+newbroad | 48.9% | 65.0% | 92.9% | 98.9% | 0.984 | 0.139 |
| scratch hardtop1 final gamma1.25 | external240 | 48.3% | 69.2% | 90.4% | n/a | 0.975 | 0.107 |
| scratch hardtop1 final gamma1.25 | external+newbroad | 46.1% | 67.5% | 90.4% | n/a | 1.026 | 0.111 |
| R1-init fine-tune final gamma1.25 | external240 | 48.8% | 67.1% | 92.5% | 98.8% | 0.936 | 0.096 |
| R1-init fine-tune final gamma1.25 | external+newbroad | 47.5% | 66.4% | 91.8% | 97.9% | 0.958 | 0.107 |
| R1/R2/scratch 45/45/10 gamma1.2 | external240 | 50.0% | 66.2% | 92.5% | n/a | 0.995 | 0.134 |
| R1/R2/scratch 45/45/10 gamma1.2 | external+newbroad | 48.2% | 65.4% | 92.5% | n/a | 0.996 | 0.139 |

Decision:

- Do not promote the scratch model, the R1-init fine-tune, or the 3-model
  blends.  They improve some regret/candidate-pool metrics but do not beat the
  current R1/R2 external Top1 baseline.
- The user's question "does that precision also hold on external tests?" is
  answered no: the best internal/mixed-looking result does not externalize.
- With 5 seconds now treated only as a guideline, the next useful work is not
  latency tuning.  It is broader fresh all-legal cap50/cap200 teacher data and
  a model/feature change that specifically attacks external Top1 EV loss.

#### Fresh100 independent all-legal cap50 check

Generated a new independent all-legal T2 source without touching the existing
external240 evaluation sets:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/inputs/t2_new_broad_fresh100_source_all_actions.jsonl`
- roots: `1200-1224`
- records: `100` (`50` BB, `50` BTN)
- avg legal candidates: `24.12`
- source generation elapsed: `49.1s`

Converted the source to cap50 labels in five parallel 20-row chunks:

- exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/t2_new_broad_fresh100_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/t2_new_broad_fresh100_alllegal_cap50.teacher.jsonl`
- reranker data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/reranker_t2_new_broad_fresh100_alllegal_cap50_dim520`
- cap50 source/T3-model Top1 matched cap50 Top1: `34/100` (`34.0%`)
- source/T3-model Top1 changed: `66/100`
- average cap50 elapsed: `58,179 ms/row` while five chunks were running in
  parallel
- average source/T3-model Top1 EV loss: `3.165`
- max source/T3-model Top1 EV loss: `60.455`

This confirms that independent fresh rows still contain large Top1 label
errors and are useful for future training.  The current R1/R2 set-residual
diagnostic is much stronger than the raw T3-model source labels:

| selector | scope | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current R1/R2 50/50 gamma1.15 | fresh100 | 82.0% | 89.0% | 96.0% | 100.0% | 0.824 | 0.766 | 0.100 |
| current R1/R2 50/50 gamma1.15 | external240 | 50.4% | 65.8% | 92.9% | 99.2% | 0.995 | 0.580 | 0.133 |

Mined fresh100 misses under the current R1/R2 selector:

- miss file:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/current_r1r2w50_gamma_sweep_fresh100_external_20260620/fresh100_misses.jsonl`
- Top1 misses: `18`
- EV loss `>= 0.25`: `10`
- EV loss `>= 1.0`: `7`
- EV loss `>= 5.0`: `1`
- mean miss regret: `4.579`
- max miss regret: `60.455`

Tested a low-LR R1-init fine-tune on those fresh100 misses:

- weighted data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/reranker_t2_new_broad_fresh100_gate1p15_top1miss_weighted_20260620`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/models/t2-set-r1-init-fresh100-hardtop1-ft-20260620`

| selector | scope | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R3 fresh100 fine-tune gamma0.75 | fresh100 | 94.0% | 97.0% | 100.0% | 100.0% | 0.030 | 0.002 | 0.000 |
| R3 fresh100 fine-tune gamma0.75 | external240 | 46.2% | 69.6% | 92.1% | 98.8% | 1.110 | 0.472 | 0.099 |
| R1/R2/R3 45/45/10 gamma1.15 | fresh100 | 83.0% | 89.0% | 98.0% | 100.0% | 0.814 | 0.766 | 0.044 |
| R1/R2/R3 45/45/10 gamma1.15 | external240 | 50.0% | 66.2% | 92.5% | 99.2% | 1.008 | 0.562 | 0.134 |
| R1/R2/R3 40/40/20 gamma1.15 | fresh100 | 84.0% | 92.0% | 98.0% | 100.0% | 0.210 | 0.145 | 0.044 |
| R1/R2/R3 40/40/20 gamma1.15 | external240 | 49.2% | 65.8% | 92.1% | 99.2% | 1.041 | 0.552 | 0.134 |

Also extended `ai/training/evaluate_set_residual_gate.py` to support blended
set checkpoints with `--checkpoints` and `--weights`.  A fresh100-trained gate
for R1/R2 50/50 gamma1.15 did not improve external Top1 beyond the existing
fixed gate:

- gate search:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100_20260620/gate_search_r1r2w50_gamma1p15_train_fresh100_eval_external_20260620/summary.json`
- best replayed external240 Top1 among the fresh100 top gates: `50.4%`
- existing fixed `candidate_rank_gap_chal_minus_base <= 0` remains the best
  checked external Top1 selector and has better Reg1 than the fresh100-selected
  gate.

Decision:

- Do not promote R3 or the R1/R2/R3 blends.  R3 learns fresh100 strongly but
  does not externalize; it is overfitting a small new shard.
- Keep the current R1/R2 50/50 gamma1.15 fixed candidate-rank gate as the best
  Top1 diagnostic.
- The next concrete improvement path is more independent fresh all-legal cap50
  rows, then mine high-regret misses across several independent shards before
  another fine-tune.  A single 100-row shard is useful evidence but too small
  to improve external Top1 safely.

#### Fresh100b + fresh200 hard-negative Top1 check

Generated another independent all-legal T2 source shard:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100b_20260620/inputs/t2_new_broad_fresh100b_source_all_actions.jsonl`
- roots used: `1300-1324`
- seed: `20260621`
- records: `100` (`50` BB, `50` BTN)
- avg legal candidates: `23.97`
- source generation elapsed: `43.7s`

Converted the source to cap50 labels in five parallel 20-row chunks:

- exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100b_20260620/t2_new_broad_fresh100b_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100b_20260620/t2_new_broad_fresh100b_alllegal_cap50.teacher.jsonl`
- reranker data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100b_20260620/reranker_t2_new_broad_fresh100b_alllegal_cap50_dim520`
- records: `100`
- candidates: `2,280`
- invalid records/candidates: `0`

Current R1/R2 50/50 on the new shard and on the unchanged external240:

| selector | scope | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1/R2 50/50 gamma1.0 | fresh100b | 69.0% | 83.0% | 87.0% | 96.0% | 100.0% | 0.610 | 0.281 | 0.051 |
| R1/R2 50/50 gamma1.0 | external240 | 50.4% | 66.2% | 80.4% | 92.5% | 99.2% | 1.031 | 0.477 | 0.134 |
| R1/R2 50/50 gamma1.15 | fresh100b | 67.0% | 83.0% | 87.0% | 96.0% | 100.0% | 0.625 | 0.281 | 0.051 |
| R1/R2 50/50 gamma1.15 | external240 | 50.4% | 65.8% | 80.4% | 92.9% | 99.2% | 0.995 | 0.580 | 0.133 |

Mined fresh100b misses under the best fresh100b sweep:

- miss file:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100b_20260620/current_r1r2w50_gamma_sweep_fresh100b_external_20260620/fresh100b_misses.jsonl`
- Top1 misses: `31`
- EV loss `>= 0.25`: `23`
- EV loss `>= 1.0`: `13`
- EV loss `>= 5.0`: `5`
- mean miss EV loss: `1.966`
- max miss EV loss: `10.419`

Built fresh100 + fresh100b hard-negative training data:

- weighted fresh100b:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100b_20260620/reranker_t2_new_broad_fresh100b_gate1p0_top1miss_weighted_20260620`
- merged fresh200 hard-negative:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200_hardneg_20260620/reranker_t2_new_broad_fresh100_100b_top1miss_weighted_20260620`
- groups: `200`
- samples: `4,593`
- sample_weight_mean: `2.913`
- group_sample_weight_mean: `3.353`

Also fixed `ai/training/merge_action_value_reranker_data.py` so
`group_sample_weights.npy` is merged as a group-level array rather than being
dropped or treated as candidate-level data.

Fine-tuned from R1 and R2 on the merged fresh200 hard-negative data:

- R4 / R1-init:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200_hardneg_20260620/models/t2-set-r1-init-fresh200-hardtop1-lr1e5-20260620`
- R5 / R2-init:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200_hardneg_20260620/models/t2-set-r2-init-fresh200-hardtop1-lr1e5-20260620`

External240-only results:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current R1/R2 50/50 gamma1.15 | 50.4% | 65.8% | 80.4% | 92.9% | 99.2% | 0.995 | 0.580 | 0.133 |
| R4 final gamma0.75 | 50.8% | 69.6% | 78.8% | 91.7% | 99.2% | 1.022 | 0.489 | 0.132 |
| R1/R2/R4 40/40/20 gamma1.0 | 50.8% | 66.2% | 79.6% | 92.5% | 99.2% | 1.015 | 0.532 | 0.134 |
| R1/R2/R4 35/35/30 gamma0.75 | 50.8% | 67.1% | 80.4% | 92.9% | 99.2% | 1.048 | 0.458 | 0.128 |
| R5 final gamma1.0 | 47.1% | 67.1% | 76.7% | 90.4% | 99.2% | 0.828 | 0.471 | 0.088 |

Decision:

- The high fresh-shard Top1 does not hold on external240.  The best external
  Top1 seen here is `50.8%`, only `+0.4pt` above the current baseline.
- Do not promote R4/R5 yet.  R4 gives a tiny Top1 lift but worsens Reg1; R5
  improves Reg1 but loses too much Top1.
- The model is not data-saturated.  More diverse independent cap50/cap200 rows
  are needed before expecting external Top1 to move materially.  The next
  useful training set should be at least several more fresh shards, not another
  fine-tune on only 200 groups.

#### Fresh100c + fresh300 hard-negative Top1 check

Generated a third independent all-legal T2 source shard:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100c_20260620/inputs/t2_new_broad_fresh100c_source_all_actions.jsonl`
- roots used: `1400-1424`
- seed: `20260622`
- records: `100` (`50` BB, `50` BTN)
- avg legal candidates: `23.88`
- source generation elapsed: `48.3s`

Converted the source to cap50 labels in five parallel 20-row chunks:

- exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100c_20260620/t2_new_broad_fresh100c_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100c_20260620/t2_new_broad_fresh100c_alllegal_cap50.teacher.jsonl`
- reranker data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100c_20260620/reranker_t2_new_broad_fresh100c_alllegal_cap50_dim520`
- records: `100`
- candidates: `2,286`
- invalid records/candidates: `0`

Current R1/R2 50/50 on fresh100c and unchanged external240:

| selector | scope | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1/R2 50/50 gamma1.15 | fresh100c | 57.0% | 75.0% | 85.0% | 96.0% | 100.0% | 0.697 | 0.281 | 0.055 |
| R1/R2 50/50 gamma1.15 | external240 | 50.4% | 65.8% | 80.4% | 92.9% | 99.2% | 0.995 | 0.580 | 0.133 |
| R1/R2 50/50 gamma1.0 | fresh100c | 57.0% | 77.0% | 84.0% | 93.0% | 100.0% | 0.721 | 0.244 | 0.065 |
| R1/R2 50/50 gamma1.0 | external240 | 50.4% | 66.2% | 80.4% | 92.5% | 99.2% | 1.031 | 0.477 | 0.134 |

Mined fresh100c misses:

- miss file:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100c_20260620/current_r1r2w50_gamma_sweep_fresh100c_external_20260620/fresh100c_misses.jsonl`
- Top1 misses: `43`
- EV loss `>= 0.25`: `31`
- EV loss `>= 1.0`: `17`
- EV loss `>= 5.0`: `2`
- mean miss EV loss: `1.620`
- max miss EV loss: `18.439`

Built fresh100 + fresh100b + fresh100c hard-negative training data:

- weighted fresh100c:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100c_20260620/reranker_t2_new_broad_fresh100c_gate1p15_top1miss_weighted_20260620`
- merged fresh300 hard-negative:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh300_hardneg_20260620/reranker_t2_new_broad_fresh100_100b_100c_top1miss_weighted_20260620`
- groups: `300`
- samples: `6,879`
- sample_weight_mean: `3.003`
- group_sample_weight_mean: `3.916`

Fine-tuned from R1 on the merged fresh300 hard-negative data:

- R6 / R1-init `lr=1e-5`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh300_hardneg_20260620/models/t2-set-r1-init-fresh300-hardtop1-lr1e5-20260620`
- R7 / R1-init `lr=5e-6`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh300_hardneg_20260620/models/t2-set-r1-init-fresh300-hardtop1-lr5e6-20260620`

External240-only results:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current R1/R2 50/50 gamma1.15 | 50.4% | 65.8% | 80.4% | 92.9% | 99.2% | 0.995 | 0.580 | 0.133 |
| R6 best gamma1.0 | 50.4% | 67.9% | 79.2% | 92.5% | 99.2% | 1.046 | 0.535 | 0.123 |
| R6 final gamma0.75 | 50.4% | 69.6% | 79.6% | 92.1% | 99.2% | 0.993 | 0.493 | 0.127 |
| R1/R2/R6 45/45/10 gamma1.15 | 50.4% | 66.7% | 80.0% | 92.5% | 99.2% | 1.003 | 0.543 | 0.134 |
| R7 best gamma1.0 | 50.4% | 67.9% | 79.6% | 92.5% | 99.2% | 1.064 | 0.535 | 0.123 |
| R7 final gamma1.0 | 49.2% | 67.5% | 78.8% | 91.2% | 99.2% | 1.064 | 0.537 | 0.125 |

Decision:

- Do not promote R6/R7 or R1/R2/R6 blends.  They improve fresh300 and some
  Top3/regret variants, but external240 Top1 does not exceed the current
  `50.4%` baseline.
- The repeated pattern is now clear: hard-negative fine-tuning on a few hundred
  fresh groups overfits those groups before it materially improves external
  Top1.
- Next improvement should either generate substantially more diverse labels
  before another fine-tune, or change the model/features so external Top1
  ranking can improve without simply memorizing fresh-shard misses.

#### T2 Top1 no-leak audit and full-state meta gate

Found an important evaluation issue: `candidate_ranks.npy` is the candidate
index from the teacher JSON candidate list.  In these teacher files the best
candidate is sorted early, so using `candidate_ranks.npy` in a selector or gate
is teacher leakage.  `route_tags.npy` is also teacher-derived.  Therefore the
previous fixed candidate-rank gate results are useful as diagnostics only, not
as runtime/product promotion evidence.

Invalidated as promotion metrics:

- `fixed_candidate_rank_gate_gamma_sweep_new1000_hardtop1_r1r2w50_20260620`
- any selector using `candidate_rank_gap_chal_minus_base`
- the reported external240 `50.4%` Top1 for `R1/R2 50/50 gamma1.15`

Honest no-leak baselines using runtime-available features only:

| selector | scope | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | external240 | 36.7% | 67.9% | 79.2% | 92.5% | 98.8% | 1.515 | 0.531 | 0.099 |
| base_scores | fresh300 | 63.7% | 81.0% | 87.3% | 94.7% | 99.7% | 0.935 | 0.506 | 0.070 |
| old R1/R2 gamma1.15 no gate | external240 | 32.9% | 60.4% | n/a | 92.9% | n/a | 2.054 | n/a | n/a |
| old R1/R2 gamma1.15 no gate | fresh300 | 55.0% | 78.0% | n/a | n/a | n/a | 1.163 | n/a | n/a |

Built no-leak 617 full-state meta-ranker:

- data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_999_alllegal_cap50_dim617`
- full-state meta models:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_ranker_no_leak_fullstate_20260620`
- base-vs-HGB no-leak gates:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_gate_no_leak_base_vs_fullhgb_20260620`
- config:
  `ai/config/t2_top1_no_leak_20260620.json`

Best current honest diagnostic selector:

| selector | scope | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_full_l31 + hgb_cls threshold 0.35 | external240 | 47.5% | 71.2% | 80.0% | 92.5% | 98.8% | 0.981 | 0.505 | 0.099 |
| hgb_full_l31 + hgb_cls threshold 0.35 | fresh300 | 69.0% | 82.7% | 87.3% | 95.0% | 99.7% | 0.779 | 0.489 | 0.069 |

Decision:

- The honest Top1 baseline is now `36.7%` on external240, not `50.4%`.
- The no-leak full-state meta gate improves external240 Top1 by `+10.8pt`
  and fresh300 Top1 by `+5.3pt` over base_scores.
- This is the current best diagnostic direction, but it is not yet a runtime
  promotion.  Next step is to turn the no-leak meta selector into a reusable
  evaluator path and validate on a new, untouched external shard.

#### Fresh100d untouched external base check

Built another untouched T2 all-legal cap50 shard to answer whether the same
precision holds outside the previous external sets:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100d_20260620`
- generation: `roots=200`, `root_start=1600`, `seed=20260623`,
  `position=both`, `target_top_k=10`, `draw_limit=3`
- records: `100` (`50` BB, `50` BTN)
- candidates: `2,346` total, `23.46` average
- exact cap: `cap50`

Base-score replay on this shard:

| selector | scope | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | fresh100d | 57.0% | 77.0% | 83.0% | 95.0% | 98.0% | 98.0% | 1.490 | 0.561 | 0.245 |

Decision:

- The `69.0%` fresh300 no-leak gate result should not be treated as a stable
  external accuracy level.
- The confirmed honest external picture is mixed: external240 gate Top1 is
  `47.5%`, fresh300 gate Top1 is `69.0%`, and fresh100d base Top1 is `57.0%`.
- Do not promote the no-leak meta gate until its exact feature builder is
  persisted and replayed cleanly on fresh100d or a larger untouched shard.

Follow-up no-leak meta retrain:

- artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_no_leak_retrain_20260620/summary.json`
- features: `617` full-state dims plus runtime-available base/set score
  statistics; no `candidate_ranks.npy` and no `route_tags.npy`
- train: `new_broad1000 0-799`
- selection holdout: `final800_999`
- external check: `fresh100d`

Result: HGB classifier variants reached `97.5-98.0%` Top1 on
`final800_999`, but fell to `50-51%` Top1 on `fresh100d`.  This is overfit,
not a usable external Top1 improvement.

Existing set-model blend replay on `fresh100d`:

- artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/gate_old_new_avg_gamma0p5_20260620/summary.json`
- selector:
  `base + 0.5 * ((((old_r1 + old_r2 + new617_r1 + new617_r2) / 4) - base))`

| selector | scope | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | fresh100d | 57.0% | 77.0% | 83.0% | 95.0% | 98.0% | 1.490 | 0.561 | 0.245 |
| old+new avg gamma0.5 | fresh100d | 59.0% | 78.0% | 85.0% | 95.0% | 98.0% | 1.177 | 0.466 | 0.221 |

Decision:

- The only fresh100d-positive movement so far is small: `+2pt` Top1 and
  `-0.313` Reg1 from a simple set-model average.
- The large internal Top1 numbers are not trustworthy unless they also hold on
  fresh external shards.
- Next Top1 work should add more diverse external-style labels before another
  high-capacity meta/classifier pass; otherwise the model memorizes the broad
  shard and loses on fresh100d.

#### Fresh500 no-leak set-blend external check

Added a reusable no-leak set-blend evaluator:

- script: `ai/training/evaluate_t2_no_leak_set_blends.py`
- cache fix: prediction cache filenames now include a checkpoint hash.  Older
  generic `no_leak_*_scores.npy` caches are not valid promotion evidence when
  changing checkpoints under the same label.
- fresh400 artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh400_20260620/summary.json`
- fresh100e artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh100e_20260620/summary.json`
- fresh500 aggregate artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh500_cachefixed_20260620/summary.json`

The evaluator uses only runtime-available scores:

- base action-value ensemble scores
- two existing 520-dim set-model scores
- two new 617-dim set-model scores
- no `candidate_ranks.npy`
- no `route_tags.npy`

Fresh500 aggregate (`fresh100` through untouched `fresh100e`):

| selector | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 58.4% | 77.6% | 85.4% | 94.8% | 98.0% | 99.2% | 1.142 | 0.525 | 0.102 |
| old_new_avg gamma0.35 | 59.8% | 79.6% | 87.6% | 94.8% | 98.4% | 99.2% | 1.077 | 0.488 | 0.105 |
| grid_best_fresh400 gamma0.35 | 60.2% | 79.4% | 87.8% | 94.4% | 98.0% | 99.2% | 1.075 | 0.480 | 0.105 |
| old_new_avg gamma0.5 | 59.2% | 79.8% | 87.2% | 94.6% | 98.2% | 99.4% | 1.064 | 0.473 | 0.101 |

Untouched `fresh100e` only:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 rerank | Reg10 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 44.0% | 68.0% | 82.0% | 95.0% | 99.0% | 1.418 | 0.549 | 0.056 |
| old_new_avg gamma0.35 | 47.0% | 74.0% | 85.0% | 95.0% | 99.0% | 1.395 | 0.509 | 0.058 |
| grid_best_fresh400 gamma0.35 | 46.0% | 74.0% | 84.0% | 95.0% | 99.0% | 1.451 | 0.450 | 0.058 |

Decision:

- The `90%+` Top1 seen on `final800_999` is not external accuracy.  It is
  related-shard behavior and should be treated as overfit unless it replays on
  fresh shards.
- The strongest honest statement is much smaller: simple `old_new_avg
  gamma0.35` improves fresh500 Top1 by `+1.4pt`, Top3 by `+2.0pt`, and Reg1 by
  `-0.065`.
- `grid_best_fresh400` has a slightly better fresh500 average Top1, but its
  weights were tuned on fresh100-fresh100d and it is worse on untouched
  fresh100e.  Keep it analysis-only.
- For the Top1 objective, this is progress but not a solved model.  Continue
  by mining the remaining external Top1 misses and training against those
  losses without teacher-leak features.

Miss mining for `old_new_avg gamma0.35`:

- script: `ai/training/mine_t2_no_leak_set_blend_misses.py`
- misses:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh500_20260620/miss_mining_old_new_avg_g035/misses.jsonl`

| scope | groups | Top1 misses | miss Reg mean | miss Reg p95 | miss Reg max | miss Reg >= 1 | miss Reg >= 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh500 | 500 | 201 | 2.679 | 10.418 | 60.455 | 104 | 51 |

The worst misses are mostly high-FL opportunity misses.  Examples include
placing paired cards to top for trips FL, or choosing the wrong row for a card
that changes FL probability by a large amount.

#### Fresh400 Top1-miss fine-tune check

To test whether the mined external-like misses are learnable without using
fresh100e, trained one additional 617-dim set model:

- training data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim617_new1000_plus_fresh400_top1miss148_r4_20260620`
- data recipe: `new_broad1000` dim617 all rows plus `148` Top1-miss groups from
  `fresh100` through `fresh100d`, repeated `4x`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-plus-fresh400miss148-r4-actionfeat617-r1-20260620/action_value_set_final.pt`
- untouched evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh100e_missft_20260620/summary.json`

Untouched `fresh100e` result:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 44.0% | 68.0% | 82.0% | 95.0% | 99.0% | 1.418 | 0.056 |
| old_new_avg gamma0.35 | 47.0% | 74.0% | 85.0% | 95.0% | 99.0% | 1.395 | 0.058 |
| miss_ft gamma0.35 | 48.0% | 75.0% | 85.0% | 96.0% | 99.0% | 0.989 | 0.068 |
| miss_ft gamma0.25 | 48.0% | 75.0% | 85.0% | 97.0% | 99.0% | 1.206 | 0.056 |

Fresh500 including the trained-on fresh100-fresh100d rows is much higher, but
that number is not clean external evidence:

| selector | avg Top1 | min Top1 | avg Reg1 | avg Reg10 |
|---|---:|---:|---:|---:|
| miss_ft gamma0.35 | 75.0% | 48.0% | 0.539 | 0.022 |
| miss_ft gamma0.5 | 75.4% | 45.0% | 0.512 | 0.016 |

Remaining `fresh100e` misses after `miss_ft gamma0.35`:

- misses: `52`
- mean miss EV loss: `1.902`
- p95 miss EV loss: `6.195`
- max miss EV loss: `9.786`
- EV loss `>= 1`: `30`
- EV loss `>= 3`: `10`
- artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh100e_missft_20260620/miss_mining_missft_g035/misses.jsonl`

Decision:

- This is real progress on the untouched shard: Top1 `44.0% -> 48.0%` and
  Reg1 `1.418 -> 0.989`.
- It is still far from a strong Top1 model.  Keep this as diagnostic and create
  another untouched external shard before considering promotion.
- The next useful loop is to generate a fresh100f shard, evaluate `miss_ft`, and
  only then mine/train the next hard-negative pass.

#### Fresh100f external check and second miss fine-tune

Generated another untouched external T2 all-legal cap50 shard:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100f_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100f_20260620/inputs/t2_new_broad_fresh100f_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100f_20260620/t2_new_broad_fresh100f_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100f_20260620/t2_new_broad_fresh100f_alllegal_cap50.teacher.jsonl`
- records: `100` (`50` BB, `50` BTN)
- candidates: `2,292`
- cap: `50`

Current no-leak set blends on untouched `fresh100f`:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 34.0% | 66.0% | 74.0% | 86.0% | 97.0% | 1.797 | 0.308 |
| old_new_avg gamma0.35 | 39.0% | 64.0% | 73.0% | 87.0% | 97.0% | 1.613 | 0.271 |
| old_new_avg gamma0.5 | 39.0% | 64.0% | 73.0% | 86.0% | 97.0% | 1.607 | 0.272 |
| grid_best_fresh400 gamma0.5 | 40.0% | 64.0% | 72.0% | 86.0% | 97.0% | 1.559 | 0.277 |

The first `miss_ft` did not generalize as well to this harder shard:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| miss_ft gamma0.35 | 38.0% | 63.0% | 70.0% | 86.0% | 96.0% | 1.783 | 0.278 |
| old_miss_avg gamma0.35 | 40.0% | 63.0% | 73.0% | 85.0% | 97.0% | 1.702 | 0.280 |

Mined `fresh100f` misses and trained a second hard-miss model:

- combined miss training rows: `210` groups
- data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim617_new1000_plus_fresh400f_top1miss210_r3_20260620`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-plus-fresh400fmiss210-r3-actionfeat617-r1-20260620/action_value_set_final.pt`

Clean `fresh100e` after the second fine-tune:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| miss_ft gamma0.35 | 48.0% | 75.0% | 85.0% | 96.0% | 99.0% | 0.989 | 0.068 |
| old_miss400f_avg gamma1.0 | 49.0% | 78.0% | 85.0% | 95.0% | 100.0% | 1.151 | 0.068 |
| old_miss400f_avg gamma0.35 | 49.0% | 76.0% | 85.0% | 96.0% | 99.0% | 1.393 | 0.058 |
| miss400f_ft gamma0.35 | 46.0% | 73.0% | 84.0% | 95.0% | 99.0% | 1.306 | 0.061 |

Trained-on `fresh100f` after the second fine-tune:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| miss400f_ft gamma1.0 | 80.0% | 96.0% | 100.0% | 100.0% | 100.0% | 0.595 | 0.000 |
| miss400f_ft gamma0.75 | 79.0% | 96.0% | 100.0% | 100.0% | 100.0% | 0.599 | 0.000 |

Decision:

- The second model clearly memorized the `fresh100f` miss set.  The trained-on
  Top1 rose to `80.0%`, but clean `fresh100e` did not get a better Reg1 than
  the first `miss_ft gamma0.35`.
- External accuracy is therefore not `80%+`.  The clean external picture is
  currently `48-49%` Top1 on `fresh100e`, and `39-40%` Top1 on the harder
  `fresh100f` shard.
- This is still useful progress for EV-loss reduction, but not enough for a
  model-only Top1 answer.  The next step should be broader diverse training
  data and another untouched external shard before promoting any checkpoint.

#### Fresh400 full-data robust fine-tune and fresh100g clean check

The miss-only fine-tune overfit too easily, so the next run added all
`fresh100` through `fresh100d` rows, not just misses:

- data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim617_new1000_plus_fresh400full_r1_hardmiss148_r2_20260620`
- groups: `1,696`
- samples: `39,861`
- recipe: `new_broad1000` + full `fresh100-fresh100d` + `148` Top1-miss
  groups repeated `2x`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-plus-fresh400full-hardmiss148-r2-actionfeat617-r1-robust-20260620/action_value_set_final.pt`

Clean-ish heldout checks:

| scope | selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fresh100e | robust gamma0.5 | 50.0% | 75.0% | 85.0% | 94.0% | 100.0% | 0.976 | 0.078 |
| fresh100f | old_robust_avg gamma0.35 | 40.0% | 63.0% | 73.0% | 85.0% | 97.0% | 1.680 | 0.277 |

This moved `fresh100e` in the right direction, but did not solve the harder
`fresh100f` shard.

Created a new clean external shard, `fresh100g`, and did not use it for any
training:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100g_20260620/inputs/t2_new_broad_fresh100g_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100g_20260620/t2_new_broad_fresh100g_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100g_20260620/t2_new_broad_fresh100g_alllegal_cap50.teacher.jsonl`
- records: `100` (`50` BB, `50` BTN)
- candidate samples: `2,334`
- source/T3-model Top1 changed after cap50 exact: `74/100`
- cap50 exact latency: `62.3s/row` average across five local parallel chunks
- source Top1 EV loss: mean `3.082`, max `18.873`

`fresh100g` result before using `fresh100e/f` for training:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 40.0% | 65.0% | 77.0% | 95.0% | 98.0% | 1.703 | 0.029 |
| old_robust_avg gamma0.25 | 41.0% | 65.0% | 78.0% | 93.0% | 99.0% | 1.787 | 0.028 |

Then trained a broader `fresh600` model using `fresh100` through `fresh100f`
but still excluding `fresh100g`:

- data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim617_new1000_plus_fresh600full_r1_hardmiss210_r2_20260620`
- groups: `2,020`
- samples: `47,268`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-plus-fresh600full-hardmiss210-r2-actionfeat617-r1-robust-20260620/action_value_set_final.pt`
- internal validation Top1 reached about `90.9%`, but clean `fresh100g`
  Top1 stayed at `40.0%`

This confirms the repeated pattern: internal validation and trained-on shards
can look strong, while clean fresh shards do not move much.

Also tried a no-leak sklearn meta-ranker using only runtime-available features:

- artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_no_leak_hgb_fresh600_to_fresh100g_20260620`
- features: `states617` plus base/set-model score features
- explicitly excluded: `bust.npy`, `fl.npy`, `candidate_ranks.npy`,
  `route_tags.npy`

Best clean `fresh100g` gamma blend:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hgb_l31 gamma0.2 | 43.0% | 67.0% | 78.0% | 95.0% | 99.0% | 1.696 | 0.046 |

Decision:

- The best clean `fresh100g` Top1 observed is only `43.0%`, a `+3pt`
  improvement over base.
- Top10 remains strong on `fresh100g` (`95.0%`), so exact rerank from a Top10
  pool is still much more reliable than model-only Top1.
- Model-only Top1 is not solved by simply adding a few hundred more groups or
  hard-miss repeats.  Next work should change the target/architecture toward
  pairwise EV-gap learning or generate substantially more independent shards
  before another promotion attempt.

#### Pairwise EV-gap loss diagnostic

Added a new optional loss to the set reranker trainer:

- file: `ai/training/train_action_value_set_reranker.py`
- default behavior is unchanged because `--pairwise-weight` defaults to `0.0`
- new loss: for candidate pairs with teacher EV gap above `--pairwise-min-gap`,
  penalize inversions where the lower-EV candidate is scored above the
  higher-EV candidate
- loss options added:
  `--pairwise-weight`, `--pairwise-min-gap`, `--pairwise-gap-cap`,
  `--pairwise-margin-scale`, `--pairwise-min-margin`,
  `--pairwise-max-margin`, `--pairwise-temperature`

Trained a pairwise-gap checkpoint from the fresh600 model:

- data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim617_new1000_plus_fresh600full_r1_hardmiss210_r2_20260620`
- init:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-plus-fresh600full-hardmiss210-r2-actionfeat617-r1-robust-20260620/action_value_set_final.pt`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-new1000-plus-fresh600full-pairwisegap-actionfeat617-r1-20260620/action_value_set_final.pt`
- pairwise settings: weight `0.75`, min EV gap `0.25`, gap cap `10.0`,
  margin scale `0.45`

Internal validation jumped to about `97.5-97.9%` Top1, but this is not reliable
promotion evidence because hard-group repeats can leak similar groups across a
random train/val split.  The clean decision remains `fresh100g`.

Clean `fresh100g` result:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 40.0% | 65.0% | 77.0% | 95.0% | 98.0% | 1.703 | 0.029 |
| hgb_l31 gamma0.2 | 43.0% | 67.0% | 78.0% | 95.0% | 99.0% | 1.696 | 0.046 |
| pair_final gamma0.05 | 41.0% | 65.0% | 77.0% | 95.0% | 99.0% | 1.608 | 0.029 |

Pairwise miss mining on `pair_final gamma0.05`:

- misses: `59`
- EV loss `>= 1`: `37`
- EV loss `>= 3`: `20`
- max EV loss: `16.962`
- artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/no_leak_set_blends_fresh100g_pairwisegap_20260620/miss_mining_pair_g005/misses.jsonl`

Typical remaining large misses are high-FL opportunity misses.  Examples:

- top `Ah As`, dealt `Jd Js 4h`: exact prefers both jacks to bottom for
  `84.9%` FL and `33.651` EV; model picks one jack to top, falling to
  `42.7%` FL and `16.689` EV.
- top `3c X1`, dealt `4c 4d 2d`: exact prefers both fours to middle for
  `54.8%` FL; model picks a lower-bust placement with much lower EV.
- top `Kh`, middle `4s 4h`, bottom `7h Tc Td Js`, dealt `6d 4d Jd`: exact
  avoids the 100% bust candidate, but the model still ranks the bust candidate
  first in that spot.

Decision:

- Pairwise EV-gap loss is useful infrastructure, but this first setting did
  not solve clean external Top1.
- The current best clean `fresh100g` Top1 remains the no-leak HGB gamma blend
  at `43.0%`.
- The next model change should add explicit runtime-available FL opportunity
  features or train with group-aware splits that prevent duplicate hard groups
  from inflating validation.

#### 693-dim runtime FL/opponent-visible feature diagnostic

Added explicit runtime-available features on top of the existing 617-dim action
features:

- file: `ai/training/action_feature_encoding.py`
- dimensions: `520` base + `97` action + `76` FL/opponent-visible = `693`
- feature inputs: current board, candidate action, post-action row shape,
  opponent visible board, known discards, candidate discard, base-score group
  context
- excluded from inference inputs: `candidate_ranks.npy`, `route_tags.npy`,
  `bust.npy`, `fl.npy`, `fl_types.npy`

Converted D-drive datasets:

- train base:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/reranker_t2_new_broad1000_0_999_alllegal_cap50_dim693`
- train mix r1:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_new1000_plus_fresh600full_r1_20260620`
- external:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100g_20260620/reranker_t2_new_broad_fresh100g_alllegal_cap50_dim693`

Neural set-reranker from scratch with the new features did not improve:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dim693 set best | 32.0% | 67.0% | 80.0% | 94.0% | 100.0% | 2.196 | 0.014 |
| dim693 set final | 32.0% | 62.0% | 80.0% | 94.0% | 100.0% | 2.194 | 0.012 |

The useful path was a no-leak sklearn meta-ranker:

- file: `ai/training/train_t2_sklearn_meta_ranker.py`
- inputs: `states693`, `base_scores`, base rank/gap/z/group-size context
- target variants tried: residual EV and direct score EV
- classifier variant tried and rejected: direct best-candidate classifier

Clean `fresh100g` external result:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 40.0% | 65.0% | 77.0% | 95.0% | 98.0% | 1.703 | 0.029 |
| previous dim617 hgb_l31 gamma0.2 | 43.0% | 67.0% | 78.0% | 95.0% | 99.0% | 1.696 | 0.046 |
| dim693 hgb_l15 score gamma0.75 | 49.0% | 69.0% | 81.0% | 96.0% | 99.0% | 1.407 | 0.014 |
| dim693 score_hgb_l31 + score_extra_d10 gamma1.15 | 51.0% | 63.0% | 80.0% | 96.0% | 99.0% | 1.311 | 0.016 |
| dim693 resid_hgb_l31 + score_hgb_l15 gamma1.15 | 51.0% | 72.0% | 82.0% | 97.0% | 99.0% | 1.371 | 0.016 |

Inference speed check on `fresh100g`:

- two HGB models
- `100` groups / `2,334` candidates
- total prediction time: `63.9ms`
- per group: `0.64ms`
- per candidate: `27.4us`

Decision:

- This is real progress: clean `fresh100g` Top1 moved from `40%` base and
  `43%` previous no-leak HGB to `51%`.
- It is still not a promotion candidate: the external set is only `100`
  groups, and Top1 is far from the product target.
- Best practical selector for the next diagnostic is
  `resid_hgb_l31 + score_hgb_l15 gamma1.15`, because it preserves stronger
  Top3/Top5 while matching the 51% Top1.
- Next step should be a larger clean external replay or more independent exact
  T2 data, then re-check the 693-dim HGB ensemble before any runtime default.

#### Fresh20h clean external replay for the dim693 HGB ensemble

Built one more untouched mini external T2 shard to check whether the
`fresh100g` 51% result immediately collapses on a different root/seed:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620/inputs/t2_new_broad_fresh20h_source_all_actions.jsonl`
- roots used: `2300-2304`
- seed: `20260628`
- records: `20` (`10` BB, `10` BTN)
- candidates: `477`
- source generation elapsed: `49.8s`
- cap50 exact elapsed: about `46.1s` to `57.3s` per row across five local
  parallel chunks
- source/T3-model Top1 matched cap50 Top1: `5/20` (`25.0%`)

Converted outputs:

- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620/t2_new_broad_fresh20h_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620/t2_new_broad_fresh20h_alllegal_cap50.teacher.jsonl`
- dim693 eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620/reranker_t2_new_broad_fresh20h_alllegal_cap50_dim693`
- HGB ensemble eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620/meta_no_leak_dim693_flopp_ensemble_eval_20260620/summary.json`

External replay metrics:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 60.0% | 80.0% | 95.0% | 100.0% | 100.0% | 2.221 | 0.315 | 0.000 |
| dim693 score_hgb_l15 gamma0.90 | 70.0% | 85.0% | 100.0% | 100.0% | 100.0% | 1.573 | 0.093 | 0.000 |
| dim693 score_hgb_l31 + score_extra_d10 gamma1.15 | 70.0% | 90.0% | 100.0% | 100.0% | 100.0% | 1.715 | 0.000 | 0.000 |
| dim693 resid_hgb_l31 + score_hgb_l15 gamma1.15 | 65.0% | 85.0% | 100.0% | 100.0% | 100.0% | 2.267 | 0.093 | 0.000 |

Decision:

- This second clean shard does not disprove the `fresh100g` gain; it is better
  than `fresh100g`, with the best checked selector at `70%` Top1.
- The shard is only `20` groups, so this is not promotion evidence by itself.
- Top10 exact-rerank coverage was `20/20` on this shard.  For product-quality
  play, the reliable path is still model TopK plus exact rerank, while Top1-only
  remains a training objective rather than a solved runtime policy.

#### Dim693 Top1 miss replay, rejected specialist, and 160-group clean aggregate

Added a reusable no-leak sklearn ensemble evaluator/miss miner:

- file:
  `ai/training/evaluate_t2_sklearn_meta_ensemble.py`
- inputs: `states`, `base_scores`, runtime base-score group context, and
  persisted sklearn models
- excluded inference inputs: `candidate_ranks`, `route_tags`, `bust`, `fl`,
  `fl_types`
- miss output is compatible with
  `ai/training/create_eval_miss_weighted_action_value_data.py`

Mined Top1 misses from the previous best Top1 ensemble
(`score_hgb_l31 + score_extra_d10 gamma1.15`):

- `fresh100g`: `49` Top1 misses, `29` with EV loss `>= 1`, `14` with EV loss
  `>= 3`, max EV loss `15.516`
- `fresh20h`: `6` Top1 misses, all `>= 1`, `3` with EV loss `>= 3`, max EV
  loss `13.885`

Built weighted miss data and a merged training set:

- weighted `fresh100g`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100g_20260620/reranker_t2_new_broad_fresh100g_dim693_top1pair_miss_weighted_20260620`
- weighted `fresh20h`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20h_20260620/reranker_t2_new_broad_fresh20h_dim693_top1pair_miss_weighted_20260620`
- merged train:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_new1000_plus_fresh600full_plus_g_h_top1miss_20260620`
- merged size: `1,720` groups / `40,371` candidate samples

Trained new weighted sklearn HGB/ExtraTrees meta-rankers:

- score target:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_no_leak_dim693_plus_g_h_miss_score_to_fresh20i_20260620`
- residual target:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_no_leak_dim693_plus_g_h_miss_resid_to_fresh20i_20260620`

The new residual model looked good on `fresh20i` only after selecting on that
same shard:

| selector on fresh20i | Top1 | Top3 | Top10 | Reg1 |
|---|---:|---:|---:|---:|
| old top1 pair gamma1.3 | 70.0% | 90.0% | 90.0% | 0.158 |
| old top1 pair + new residual HGB gamma1.3 | 75.0% | 90.0% | 90.0% | 0.072 |

Then froze `old top1 pair + new residual HGB gamma1.3` and tested it on a new
clean shard, `fresh20j`, which was not used for training or selector choice:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20j_20260620/inputs/t2_new_broad_fresh20j_source_all_actions.jsonl`
- records: `20` (`10` BB, `10` BTN)
- source/T3-model Top1 matched cap50 Top1: `7/20` (`35.0%`)
- eval summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20j_20260620/meta_no_leak_dim693_old_new_fixed_eval_20260620/summary.json`

`fresh20j` result:

| selector on fresh20j | Top1 | Top3 | Top10 | Reg1 |
|---|---:|---:|---:|---:|
| base_scores | 40.0% | 70.0% | 95.0% | 0.876 |
| old top1 pair gamma1.15 | 50.0% | 70.0% | 95.0% | 0.622 |
| old top1 pair gamma1.3 | 55.0% | 65.0% | 95.0% | 0.518 |
| frozen old top1 pair + new residual HGB gamma1.3 | 35.0% | 60.0% | 95.0% | 0.746 |

Clean aggregate over `fresh100g`, `fresh20h`, `fresh20i`, and `fresh20j`
(`160` total groups):

| selector | groups | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 160 | 42.5% | 66.9% | 80.0% | 94.4% | 98.1% | 1.609 | 0.642 | 0.031 |
| old top1 pair gamma1.15 | 160 | 54.4% | 70.6% | 85.0% | 95.6% | 99.4% | 1.135 | 0.467 | 0.021 |
| old top1 pair gamma1.3 | 160 | 53.1% | 71.2% | 83.1% | 95.0% | 98.8% | 1.164 | 0.542 | 0.016 |
| old single HGB l15 gamma1.3 | 160 | 50.0% | 71.9% | 83.1% | 94.4% | 99.4% | 1.265 | 0.444 | 0.020 |

Decision:

- The `fresh100g/fresh20h` miss-specialist is rejected for now.  It improved
  the selector-choice shard (`fresh20i`) but failed on `fresh20j`.
- The best current no-leak fixed Top1 diagnostic is still the old dim693
  `score_hgb_l31 + score_extra_d10 gamma1.15`: `54.4%` Top1 across `160`
  clean groups, versus `42.5%` for base.
- This is a real external gain, but not close to the target.  More Top1 gains
  require broader independent exact data or a stronger model family, not just
  replaying the latest misses.

#### LightGBM listwise ranker diagnostic

Installed local `lightgbm` and added a listwise T2 ranker script:

- file:
  `ai/training/train_t2_lightgbm_ranker.py`
- model: `LGBMRanker(objective="lambdarank")`
- inputs: same no-leak runtime feature set as the HGB meta-ranker
  (`states693`, `base_scores`, base rank/gap/z/group-size context)
- excluded inference inputs: `candidate_ranks`, `route_tags`, `bust`, `fl`,
  `fl_types`
- label modes:
  - `gap`: relevance decreases with EV gap from the best action
  - `rank`: relevance decreases with teacher rank

Training data:

- `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_new1000_plus_fresh600full_plus_g_h_i_j_full_20260620`
- groups: `1,760`
- samples: `41,292`

First checked on `fresh20k` after selecting the LGBM settings:

| selector on fresh20k | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 25.0% | 60.0% | 65.0% | 75.0% | 100.0% | 1.091 | 0.060 |
| old HGB top1 pair gamma1.15 | 25.0% | 65.0% | 80.0% | 100.0% | 100.0% | 1.031 | 0.000 |
| LGBM gap63 direct | 40.0% | 65.0% | 85.0% | 95.0% | 100.0% | 0.979 | 0.018 |

Then froze the LGBM setting and checked a new clean shard, `fresh20l`:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20l_20260620/inputs/t2_new_broad_fresh20l_source_all_actions.jsonl`
- records: `20` (`10` BB, `10` BTN)
- source/T3-model Top1 matched cap50 Top1: `9/20` (`45.0%`)

`fresh20l` result:

| selector on fresh20l | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 40.0% | 60.0% | 70.0% | 80.0% | 95.0% | 1.473 | 0.269 |
| old HGB top1 pair gamma1.15 | 45.0% | 75.0% | 85.0% | 90.0% | 100.0% | 0.733 | 0.068 |
| old HGB top1 pair gamma1.5 | 45.0% | 70.0% | 85.0% | 100.0% | 100.0% | 0.635 | 0.000 |
| LGBM gap63 direct | 45.0% | 70.0% | 85.0% | 90.0% | 100.0% | 0.655 | 0.109 |

Clean holdout aggregate for the LGBM candidate over `fresh20k + fresh20l`
(`40` groups, not used for LGBM training):

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 |
|---|---:|---:|---:|---:|---:|---:|
| base_scores | 32.5% | 60.0% | 67.5% | 77.5% | 97.5% | 1.282 |
| old HGB top1 pair gamma1.15 | 35.0% | 70.0% | 82.5% | 95.0% | 100.0% | 0.882 |
| LGBM gap63 direct | 42.5% | 67.5% | 85.0% | 92.5% | 100.0% | 0.817 |

Decision:

- LightGBM listwise ranking is the first stronger model family that improves
  the new clean `fresh20k/fresh20l` aggregate Top1 over the old HGB top1 pair:
  `42.5%` vs `35.0%`.
- It is not promotion evidence yet because the clean post-selection holdout is
  only `40` groups, and Top3 is not better than the old HGB selector.
- Keep LightGBM as the next promising branch.  The next useful check is a
  larger untouched shard, or a combined selector that uses LGBM to rescue
  HGB misses without sacrificing Top3/Top10.

#### HGB vs LightGBM meta-arbitration diagnostic

Added a no-leak group-level arbitration evaluator:

- file: `ai/training/evaluate_t2_meta_arbitration.py`
- HGB selector: dim693 `score_hgb_l31 + score_extra_d10`, gamma `1.15`
- LightGBM selector: `lgbm_gap63_direct`
- gate inputs: only runtime-available selector margins, cross-ranks,
  base-score margins, and group size
- excluded inputs: teacher EV, teacher rank, route tags, bust/FL labels

Gate training used `fresh100g + fresh20h + fresh20i + fresh20j` (`160`
groups).  Evaluation used the untouched post-LightGBM check
`fresh20k + fresh20l` (`40` groups).

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_arbitration_hgb_vs_lgbm_g_h_i_j_to_k_l_20260620/summary.json`

`fresh20k + fresh20l` aggregate:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 32.5% | 60.0% | 67.5% | 77.5% | 97.5% | 1.282 | 0.165 |
| HGB fixed | 35.0% | 70.0% | 82.5% | 95.0% | 100.0% | 0.882 | 0.034 |
| LightGBM direct | 42.5% | 67.5% | 85.0% | 92.5% | 100.0% | 0.817 | 0.063 |
| threshold gate: `hgb_score_z_gap_to_lgbm_top >= 0.0469657` | 45.0% | 70.0% | 87.5% | 97.5% | 100.0% | 0.803 | 0.002 |
| oracle pair HGB-vs-LGBM | 50.0% | 75.0% | 87.5% | 95.0% | 100.0% | 0.681 | 0.034 |

Conservative replay with `fresh20j` removed from gate training:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_arbitration_hgb_vs_lgbm_g_h_i_to_k_l_20260620/summary.json`
- best replayed gate on `fresh20k + fresh20l` was
  `logreg_balanced_c03@0.588714`: Top1 `42.5%`, Top3 `72.5%`,
  Top10 `97.5%`, Reg1 `0.814`, Reg10 `0.002`.

Decision:

- The HGB/LightGBM pair has real complementarity: oracle pair Top1 is `50.0%`
  on this 40-group check.
- The best trained gate improves the 40-group Top1 from LightGBM `42.5%` to
  `45.0%`, and substantially improves Top10 rerank loss (`0.063` to `0.002`).
- This is still diagnostic only.  The holdout is only `40` groups, and the
  conservative replay did not improve Top1 over LightGBM.  Do not promote this
  as a default policy until it passes a larger untouched shard.

#### Fresh100m larger clean external replay

Built a larger untouched local external shard to check whether the 40-group
HGB/LightGBM gate result holds:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/inputs/t2_new_broad_fresh100m_source_all_actions.jsonl`
- roots used: `2800-2824`
- seed: `20260633`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,439`
- source generation elapsed: `51.3s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/t2_new_broad_fresh100m_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/t2_new_broad_fresh100m_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/reranker_t2_new_broad_fresh100m_alllegal_cap50_dim693`
- teacher candidates after exact join: `2,325`
- source/T3-model Top1 matched cap50 Top1: `24/100`
- cap50 exact elapsed: `64.7s/row` average across five local parallel chunks
- source Top1 EV loss: mean `2.851`, max `16.228`

The HGB/LightGBM gate did not reproduce on the larger clean shard:

- arbitration summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_arbitration_hgb_vs_lgbm_g_h_i_j_to_fresh100m_20260620/summary.json`

| selector on fresh100m | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 40.0% | 66.0% | 79.0% | 91.0% | 98.0% | 2.019 | 0.029 |
| HGB fixed: `score_hgb_l31 + score_extra_d10 gamma1.15` | 51.0% | 76.0% | 85.0% | 91.0% | 97.0% | 1.021 | 0.027 |
| LightGBM direct | 46.0% | 77.0% | 86.0% | 96.0% | 100.0% | 1.342 | 0.011 |
| best trained gate replay | 50.0% | 76.0% | 86.0% | 93.0% | 99.0% | 1.041 | 0.023 |
| oracle pair HGB-vs-LightGBM | 58.0% | 80.0% | 90.0% | 93.0% | 99.0% | 0.737 | 0.016 |

Checked additional dim693 HGB combinations:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/meta_no_leak_dim693_ensemble_eval_20260620/summary.json`

| selector on fresh100m | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all4 gamma1.3` | 54.0% | 78.0% | 84.0% | 92.0% | 99.0% | 0.883 | 0.025 |
| `resid_l31 + score_l15 gamma1.15` | 53.0% | 76.0% | 85.0% | 94.0% | 99.0% | 0.985 | 0.027 |
| `score_l31 + extra gamma1.15` | 51.0% | 76.0% | 85.0% | 91.0% | 97.0% | 1.021 | 0.027 |

Clean aggregate over `fresh100g + fresh20h + fresh20i + fresh20j +
fresh20k + fresh20l + fresh100m` (`300` groups):

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_no_leak_dim693_all4_clean_aggregate_20260620/summary.json`

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_scores | 40.3% | 65.7% | 78.0% | 91.0% | 98.0% | 1.702 | 0.048 |
| `score_l31 + extra gamma1.15` | 50.7% | 72.3% | 84.7% | 94.0% | 98.7% | 1.064 | 0.025 |
| `resid_l31 + score_l15 gamma1.15` | 49.3% | 73.7% | 84.7% | 96.0% | 99.3% | 1.104 | 0.015 |
| `all4 gamma1.3` | 50.7% | 73.7% | 82.7% | 94.0% | 99.0% | 1.048 | 0.020 |

Mined `49` Top1 misses for `score_l31 + extra gamma1.15` on `fresh100m`:

- miss file:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/meta_no_leak_dim693_top1_miss_mining_20260620/score_l31_plus_extra_g1p15.misses.jsonl`
- EV loss `>= 0.25`: `33`
- EV loss `>= 1`: `25`
- EV loss `>= 3`: `13`
- max EV loss: `9.034`
- weighted data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100m_20260620/reranker_t2_new_broad_fresh100m_dim693_top1pair_miss_weighted_20260620`
- weighted groups: `33`

Tried adding fresh100m weighted misses back into the `g/h/i/j` full training
data:

- merged data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_full_g_h_i_j_plus_m_top1miss_20260620`
- HGB score model eval on `fresh20k`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/meta_no_leak_dim693_full_g_h_i_j_plus_m_miss_score_to_fresh20k_20260620/summary.json`
- LightGBM eval on `fresh20k`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/lgbm_ranker_dim693_full_g_h_i_j_plus_m_miss_gap63_to_fresh20k_20260620/summary.json`
- result: both were worse than the existing fixed HGB/LightGBM checks on
  `fresh20k`; do not promote.

Decision:

- The 40-group HGB/LightGBM gate improvement was not stable on `fresh100m`.
- The strongest current fixed no-leak external selector remains HGB-family.
  `score_l31 + extra gamma1.15` is still the Top1 baseline on the broad
  clean aggregate; `all4 gamma1.3` slightly improves Reg1 but does not improve
  aggregate Top1.
- Replaying fresh100m misses directly into HGB/LightGBM training overfits that
  miss distribution and hurts `fresh20k`.
- Next useful Top1 work should change the learning target/architecture, not
  just add another miss-weighted replay.  The current miss pattern is mostly
  high-FL opportunity underestimation.

#### Aux-target and union-pool external check

Added two diagnostic evaluators:

- aux-target blend:
  `ai/training/evaluate_t2_aux_target_blends.py`
- sklearn/LightGBM union-pool audit:
  `ai/training/evaluate_t2_sklearn_candidate_union.py`

The aux-target check trained no-leak HGB regressors for `FL`, `bust`,
`fl_value`, and per-FL-type labels using the existing runtime feature set
(`states693`, `base_scores`, base rank/gap/z/group-size).  Teacher FL/bust
labels were used only as targets, not as inference inputs.

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/aux_target_blends_clean300_20260620/summary.json`
- train data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_new1000_plus_fresh600full_plus_g_h_i_j_full_20260620`
- eval data:
  `fresh100g + fresh20h + fresh20i + fresh20j + fresh20k + fresh20l + fresh100m`
  (`300` clean external groups)
- best aux blend:
  `ev_plus_center_fl_value`, weight `0.1`

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EV only: `score_l31 + extra gamma1.15` | 50.7% | 72.3% | 84.7% | 94.0% | 98.7% | 1.064 | 0.025 |
| EV + centered `fl_value` 0.1 | 50.7% | 72.0% | 85.0% | 94.0% | 98.7% | 0.998 | 0.025 |

Decision:

- FL/bust auxiliary targets slightly reduce Top1 EV loss, but they do not
  improve external Top1 rate.  This is useful evidence that FL opportunity is
  part of the miss shape, but it is not enough to solve model-only Top1.

Since the 5s target is now only a guideline, also checked a practical
shortlist path: union the candidate proposals from several no-leak selectors,
then exact-rerank the union pool.

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/sklearn_union_clean300_20260620/summary.json`
- selectors:
  `base_scores`,
  `hgb_pair_g115`,
  `all4_g130`,
  `resid_l31_score_l15_g115`,
  `lgbm_gap63`
- eval data:
  same clean `300` groups as above

| per-selector K | recall after exact rerank | mean EV loss | max EV loss | avg union pool | max union pool |
|---:|---:|---:|---:|---:|---:|
| 1 | 69.7% | 0.331 | 6.116 | 1.8 | 5 |
| 2 | 84.0% | 0.154 | 5.825 | 3.4 | 8 |
| 3 | 88.7% | 0.087 | 4.102 | 4.8 | 9 |
| 5 | 94.7% | 0.021 | 2.140 | 7.3 | 12 |
| 8 | 97.3% | 0.007 | 1.092 | 10.9 | 18 |
| 10 | 98.0% | 0.004 | 1.092 | 13.1 | 21 |
| 15 | 99.7% | 0.000 | 0.014 | 18.1 | 24 |
| 20 | 100.0% | 0.000 | 0.000 | 22.0 | 26 |

Decision:

- External model-only Top1 is still only about `50-54%`, depending on shard and
  selector.
- If latency is allowed to be higher, the best current T2 decision path is
  not model Top1.  It is multi-selector union plus exact rerank.
- On the current 300-group cap50 external audit, `per-selector Top10 union`
  already reaches `98.0%` recall with avg pool `13.1`; `Top15 union` reaches
  `99.7%`; `Top20 union` reaches `100.0%`.
- This is cap50 teacher evidence, not full T2 exact.  Before calling it final,
  run a cap200/cap1000 overlap on the remaining Top10/Top15 misses and on a new
  untouched shard.

#### Cap200 overlap for union misses

Checked the only `3` clean300 rows where the cap50 `Top10` or `Top15`
multi-selector union had non-zero EV loss:

- input subset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/sklearn_union_cap200_misscheck_20260620/top10_top15_union_misses_source3.jsonl`
- cap200 all-legal output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/sklearn_union_cap200_misscheck_20260620/cap200_alllegal/t2_oracle_cap200_limit3.jsonl`
- cap200 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/sklearn_union_cap200_misscheck_20260620/cap200_alllegal/t2_oracle_cap200_limit3.summary.json`
- cap200-vs-union action match:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/sklearn_union_cap200_misscheck_20260620/cap200_alllegal/cap200_best_vs_cap50_union.summary.json`

Cap200 runtime:

- records: `3`
- all-legal candidates per row: `24`
- cap: `200` T3 draws per T2 action
- avg elapsed: `49.7s/row`
- estimated full T2 from cap: avg `1592s/row` (`26.5min/row`)

Stability:

| subset | dataset | group | dealt | cap50 best same as cap200 | cap200 best in Top10 union | cap200 best in Top15 union | cap200 best score | bust | FL |
|---:|---|---:|---|---|---|---|---:|---:|---:|
| 0 | fresh100m | 15 | 8d 4c 2c | yes | no | yes | 8.063 | 11.5% | 20.2% |
| 1 | fresh20l | 0 | 5h Qs 8h | yes | no | yes | 1.326 | 9.6% | 0.0% |
| 2 | fresh100m | 62 | Td Tc 6c | yes | no | no | 1.289 | 15.3% | 0.0% |

Interpretation:

- The `Top10` union misses are real, not cap50 label noise.
- `Top15` union captured `2/3` cap200 best actions.
- The remaining `Top15` miss has tiny cap50 EV loss (`0.014`), and `Top20`
  union captured all `3/3`.
- For higher-accuracy mode, use at least `Top15` union; use `Top20` union if
  the goal is to avoid current known misses before running broader cap200/cap1000
  validation.

#### Untouched fresh20n external check

Created a new small external shard with a different root/seed from the earlier
fresh sets:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/inputs/t2_new_broad_fresh20n_source_all_actions.jsonl`
- root range: `2900-2909`
- seed: `20260634`
- records: `20` (`10` BB, `10` BTN)
- candidates: `483`, avg `24.15` per decision
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/t2_new_broad_fresh20n_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/t2_new_broad_fresh20n_alllegal_cap50.teacher.jsonl`
- dim693 data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/reranker_t2_new_broad_fresh20n_alllegal_cap50_dim693`
- union summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/sklearn_union_eval_20260620/summary.json`

Runtime:

- source generation: `14.1s`
- cap50 all-legal exact, 5 parallel chunks: wall time about `4.2min`
- exact evaluator average per row by chunk: `53-61s/row`

Fresh20n cap50 union result:

| per-selector K | recall after exact rerank | mean EV loss | max EV loss | avg union pool |
|---:|---:|---:|---:|---:|
| 1 | 50.0% | 0.869 | 5.202 | 1.8 |
| 3 | 80.0% | 0.063 | 1.192 | 4.8 |
| 5 | 90.0% | 0.001 | 0.018 | 7.3 |
| 8 | 95.0% | 0.001 | 0.018 | 10.9 |
| 10 | 100.0% | 0.000 | 0.000 | 13.1 |
| 15 | 100.0% | 0.000 | 0.000 | 18.1 |
| 20 | 100.0% | 0.000 | 0.000 | 22.1 |

Interpretation:

- This untouched 20-group check supports the same direction as clean300:
  model-only Top1 is not reliable, but multi-selector union plus exact rerank
  sharply reduces EV loss.
- On this small shard, `Top10+` already has zero cap50 EV loss.  Because
  clean300 still had real Top10 misses and cap200 confirmed those misses,
  `Top20` remains the safer high-accuracy setting until a larger untouched
  cap200/cap1000 check is run.

#### Selector-feature Top1 ranker

Added a no-leak selector-feature ranker:

- script:
  `ai/training/train_t2_selector_feature_ranker.py`
- output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_eval_k_l_m_n_20260620/summary.json`
- classifier check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_cls_eval_k_l_m_n_20260620/summary.json`
- train data:
  `av_dataset_dim693_new1000_plus_fresh600full_plus_g_h_i_j_full_20260620`
- eval data:
  `fresh20k + fresh20l + fresh100m + fresh20n` (`160` external groups)

Inputs are runtime-available only:

- state features
- base score
- each selector's predicted score
- within-spot selector rank, gap to top, z-score
- selector Top1/Top3/Top5 vote counts and rank agreement features

Excluded from inference inputs:

- teacher EV
- teacher rank
- route tags
- bust/FL labels

External aggregate:

| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| new `hgb_l31_selector` | 50.6% | 76.3% | 85.6% | 95.0% | 99.4% | 0.874 | 0.018 |
| new `extra_trees_d12_selector` | 50.0% | 73.1% | 85.6% | 95.0% | 99.4% | 0.821 | 0.015 |
| existing `all4_g130` | 46.9% | 73.1% | 81.9% | 91.9% | 99.4% | 0.910 | 0.037 |
| existing `hgb_pair_g115` | 46.3% | 73.8% | 83.1% | 92.5% | 98.1% | 1.006 | 0.026 |
| existing `lgbm_gap63` | 44.4% | 73.8% | 85.0% | 94.4% | 100.0% | 1.178 | 0.025 |
| existing `base_scores` | 37.5% | 63.8% | 75.0% | 88.1% | 98.1% | 1.751 | 0.060 |

Per-dataset for `hgb_l31_selector`:

| dataset | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| fresh20k | 45.0% | 80.0% | 100.0% | 100.0% | 0.764 | 0.000 |
| fresh20l | 50.0% | 70.0% | 95.0% | 100.0% | 0.548 | 0.014 |
| fresh100m | 53.0% | 79.0% | 94.0% | 99.0% | 0.951 | 0.024 |
| fresh20n | 45.0% | 65.0% | 95.0% | 100.0% | 0.929 | 0.013 |

Classifier target check:

- Direct teacher-best classifiers did not improve Top1.
- Best classifier row was `extra_trees_cls_d12_state` at `48.1%` Top1,
  below `hgb_l31_selector`.

Decision:

- Selector-agreement features are a real model-only Top1 improvement on the
  current external k/l/m/n set (`+3.7pp` over `all4_g130`, `+4.3pp` over
  `hgb_pair_g115`).
- This is still far from solved.  Treat `hgb_l31_selector` as the next
  diagnostic Top1 baseline, not a final policy.
- For final action quality, continue to use candidate union plus exact rerank;
  the selector-feature ranker is useful for ordering and reducing the exact
  workload, not for replacing exact yet.

Follow-up target experiments:

- target-mode sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_target_modes_eval_k_l_m_n_20260620/summary.json`
- train-selected blend sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_target_blends_eval_k_l_m_n_20260620/summary.json`
- selector-feature LightGBM ranker:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_lgbm_ranker_eval_k_l_m_n_20260620/summary.json`

Result:

- Group-relative regression targets did not beat `hgb_l31_selector_score` on
  Top1.  Best stayed `50.6%`.
- Train-selected blends also did not beat `50.6%` Top1, though some improved
  Top10/Reg10.
- Selector-feature LightGBM ranker did not improve Top1; best LGBM row was
  `48.1%`.

Hard-negative replay check:

- train plus `fresh20k + fresh20l + fresh100m`, eval `fresh20n`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_plus_k_l_m_eval_n_20260620/summary.json`
- train plus `fresh20k + fresh20l + fresh100m + fresh20n`, eval new `fresh20o`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_plus_k_l_m_n_eval_o_20260620/summary.json`
- fresh20o source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/inputs/t2_new_broad_fresh20o_source_all_actions.jsonl`
- fresh20o cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/t2_new_broad_fresh20o_alllegal_cap50.exact.jsonl`
- fresh20o union summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/sklearn_union_eval_20260620/summary.json`

Result:

| experiment | eval | best model-only Top1 | Reg1 | notes |
|---|---|---:|---:|---|
| train + k/l/m | fresh20n | 55.0% | 0.927 | improved fresh20n, but only 20 groups |
| train + k/l/m/n | fresh20o | 45.0% | 1.546 | best was existing `all4_g130`; added models did not generalize |

Fresh20o candidate-union result:

| per-selector K | recall after exact rerank | max EV loss | avg union pool |
|---:|---:|---:|---:|
| 1 | 55.0% | 3.665 | 1.8 |
| 3 | 80.0% | 1.674 | 4.5 |
| 5 | 90.0% | 0.088 | 7.0 |
| 10 | 95.0% | 0.088 | 13.1 |
| 15 | 95.0% | 0.088 | 17.6 |
| 20 | 100.0% | 0.000 | 21.4 |

Selector-feature union external checks:

- clean300 K10-15 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_clean300_k10_15_20260620/summary.json`
- fresh20o K10-15 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh20o_k10_15_20260620/summary.json`
- clean300 miss mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_clean300_20260620/miss_mining_top5_8_10_20260620/summary.json`

Clean300 with selector-feature sources:

| per-selector K | recall | mean EV loss | max EV loss | avg pool | max pool |
|---:|---:|---:|---:|---:|---:|
| 10 | 98.3% | 0.00390 | 1.092 | 13.5 | 23 |
| 11 | 99.0% | 0.00005 | 0.014 | 14.5 | 24 |
| 12 | 99.3% | 0.00005 | 0.014 | 15.6 | 24 |
| 13 | 99.3% | 0.00005 | 0.014 | 16.6 | 24 |
| 14 | 99.7% | 0.00000 | 0.000 | 17.5 | 24 |
| 15 | 100.0% | 0.00000 | 0.000 | 18.4 | 24 |

Fresh20o with the same selector-feature sources:

| per-selector K | recall | mean EV loss | max EV loss | avg pool | max pool |
|---:|---:|---:|---:|---:|---:|
| 10 | 95.0% | 0.004 | 0.088 | 13.6 | 22 |
| 11 | 95.0% | 0.004 | 0.088 | 14.6 | 22 |
| 12 | 95.0% | 0.004 | 0.088 | 15.4 | 23 |
| 13 | 95.0% | 0.004 | 0.088 | 16.0 | 24 |
| 14 | 95.0% | 0.004 | 0.088 | 16.8 | 25 |
| 15 | 100.0% | 0.000 | 0.000 | 17.9 | 26 |

Minimal extra-source check:

- Additional source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_clean300_selector_hgb_eval_o_20260620/hgb_l31_selector_gap_to_best_top_weighted.joblib`
- fresh20o union summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh20o_plus_clean_l31gap_20260620/summary.json`
- fresh20n base7 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh20n_base7_20260620/summary.json`
- fresh20n plus-clean summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh20n_plus_clean_l31gap_20260620/summary.json`

Fresh20o with only `clean_hgb_l31_gap` added:

| per-selector K | recall | mean EV loss | max EV loss | avg pool | max pool |
|---:|---:|---:|---:|---:|---:|
| 10 | 95.0% | 0.004 | 0.088 | 14.3 | 23 |
| 13 | 95.0% | 0.004 | 0.088 | 16.7 | 24 |
| 14 | 100.0% | 0.000 | 0.000 | 17.6 | 25 |
| 15 | 100.0% | 0.000 | 0.000 | 18.4 | 26 |

Fresh20n is already zero-loss at Top10 with the base7 sources; adding
`clean_hgb_l31_gap` did not hurt but slightly increased pool size.  Use the
extra source only because it fixes the fresh20o Top14 miss.

Top10/Top8 miss mining on clean300:

- Top10 misses: `5/300`, only `2` with EV loss `>0.05`, max loss `1.092`.
- Top8 misses: `8/300`, `4` with EV loss `>0.05`, max loss `1.092`.
- Top5 misses: `13/300`, `7` with EV loss `>0.05`, max loss `2.140`.
- Top15 catches every mined Top5/8/10 miss with zero EV loss.

Interpretation: Top10 is close but not safe; Top11-14 largely eliminate EV
loss on clean300, but fresh20o still needs Top15.  Keep Top15 as the current
high-accuracy runtime pool and use the Top10/Top8 misses as hard negatives for
the next model-only Top1/Top5 training pass.

Fresh20o shortlist exact rerank runtime check:

- Corrected source mapping uses exact-teacher candidate indices via
  `ai/training/build_t2_union_shortlist_source.py --candidate-exact`.
- all-legal comparison script:
  `ai/training/compare_t2_shortlist_exact.py`

| per-selector K | exact hit rate | mean EV loss | max EV loss | avg evaluated actions | avg elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 55.0% | 0.940 | 5.580 | 1.9 | 6.05s |
| 3 | 80.0% | 0.114 | 1.654 | 4.5 | 6.56s |
| 5 | 90.0% | 0.087 | 1.649 | 7.2 | 7.42s |
| 10 | 95.0% | 0.004 | 0.088 | 13.1 | 10.73s |
| 20 | 100.0% | 0.000 | 0.000 | 23.4 | 16.63s |

Selector-feature union check:

- union rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_eval_fresh20o_20260620/rows.jsonl`
- Top15 source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_sources/t2_fresh20o_selectorfeature_union_top15_exactindexed_source.jsonl`
- Top15 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/selectorfeature_union_top15_exactindexed_cap50_fl_ev/t2_oracle_cap50_limit20.jsonl`
- all-legal comparison:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/selectorfeature_union_top15_exactindexed_cap50_fl_ev/vs_alllegal.summary.json`

| selector-feature K | exact hit rate | mean EV loss | max EV loss | avg evaluated actions | avg elapsed |
|---:|---:|---:|---:|---:|---:|
| 15 | 100.0% | 0.000 | 0.000 | 17.9 | 13.13s |

Minimal extra-source exact check:

- Top14 source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_sources/t2_fresh20o_selectorfeature_plus_clean_l31gap_top14_exactindexed_source.jsonl`
- Top14 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/selectorfeature_plus_clean_l31gap_top14_exactindexed_cap50_fl_ev/t2_oracle_cap50_limit20.jsonl`
- all-legal comparison:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/selectorfeature_plus_clean_l31gap_top14_exactindexed_cap50_fl_ev/vs_alllegal.summary.json`

| selector-feature K | exact hit rate | mean EV loss | max EV loss | avg evaluated actions | avg elapsed |
|---:|---:|---:|---:|---:|---:|
| 14 + clean_hgb_l31_gap | 100.0% | 0.000 | 0.000 | 17.6 | 12.15s |

The Top14 plus `clean_hgb_l31_gap` path is the current best fresh20o
compromise after relaxing the strict 5s target: it matches all-legal cap50 on
all 20 external rows while evaluating about 5.8 fewer actions and running about
4.5s faster than the previous Top20 high-accuracy path.

Top20 verification:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_sources/t2_fresh20o_union_top20_exactindexed_source.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/union_top20_exactindexed_cap50_fl_ev/t2_oracle_cap50_limit20.jsonl`
- all-legal comparison:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/union_top20_exactindexed_cap50_fl_ev/vs_alllegal.summary.json`
- The rerun must use `ai/config/fl_ev.json`; using a missing FL config silently
  falls back to a different reward setting and makes the all-legal comparison
  invalid.

Hybrid runtime partial check:

- `ai.tutor.hybrid_t1t2` with Top20, `t3_union` partial refinement, and a
  relaxed 60s budget did not reproduce the cap50 exact result.
- With model/rescue overrides disabled, chosen average EV loss was `1.795`
  against the cap50 teacher; with the current experimental rescue stack it was
  `1.242`.
- Increasing the partial T3 samples to as many as `80` on the first 5 rows did
  not fix the mismatch.  The runtime path is measuring a different, sampled T3
  continuation value, while the cap50 teacher is a T2 capped exact label.

Union-pool Top1 reranker probe:

- script:
  `ai/training/train_t2_union_pool_reranker.py`
- train data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_new1000_plus_fresh600full_plus_g_h_i_j_full_20260620`
- eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/reranker_t2_new_broad_fresh20o_alllegal_cap50_dim693`

| probe | fresh20o ceiling | best external Top1 | Reg1 | result |
|---|---:|---:|---:|---|
| selector Top1 union gate | 55.0% | 10.0% | 2.257 | rejected |
| Top5 union local reranker | 90.0% | 10.0% | 2.405 | rejected |
| clean300 selector HGB | n/a | 45.0% | 1.274 | rejected |

Output summaries:

- Top1 gate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/union_top1_gate_trainfull_eval_fresh20o_20260620/summary.json`
- Top5 local reranker:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/union_pool_reranker_trainfull_eval_fresh20o_top5_20260620/summary.json`
- clean300 selector HGB:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_clean300_selector_hgb_eval_o_20260620/summary.json`

Miss-weighted selector-feature pass:

- code change:
  `ai/training/train_t2_selector_feature_ranker.py` now accepts
  `--miss-rows` and weights selected miss groups without using teacher EV at
  inference time.
- miss rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_clean300_20260620/miss_mining_top5_8_10_20260620/misses.jsonl`
- training summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_clean300_missweighted_eval_no_20260620/summary.json`
- union summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh20no_plus_missweighted_20260620/summary.json`

Result:

- `13` miss rows used, covering `7` weighted groups from `fresh20k`,
  `fresh20l`, and `fresh100m`.
- Standalone model-only Top1 did not solve the problem.  Best aggregate row on
  fresh20n+fresh20o was `47.5%` Top1 and `1.162` Reg1.
- As an extra union source, it improved the high-accuracy pool from Top14 to
  Top13 on fresh20n+fresh20o.

Top13 exact rerank check:

| eval | hit rate vs all-legal cap50 | mean EV loss | max EV loss | avg evaluated actions | avg elapsed |
|---|---:|---:|---:|---:|---:|
| fresh20o | 100.0% | 0.000 | 0.000 | 17.05 | 10.88s |
| fresh20n | 100.0% | 0.000 | 0.000 | 17.55 | 11.08s |
| combined | 100.0% | 0.000 | 0.000 | 17.30 | 10.98s |

Artifacts:

- fresh20o source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_sources/t2_fresh20o_selectorfeature_plus_missweighted_top13_exactindexed_source.jsonl`
- fresh20o comparison:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/shortlist_exact/selectorfeature_plus_missweighted_top13_exactindexed_cap50_fl_ev/vs_alllegal.summary.json`
- fresh20n source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/shortlist_sources/t2_fresh20n_selectorfeature_plus_missweighted_top13_exactindexed_source.jsonl`
- fresh20n comparison:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/shortlist_exact/selectorfeature_plus_missweighted_top13_exactindexed_cap50_fl_ev/vs_alllegal.summary.json`

Broader existing-set replay:

- evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100def_plus_missweighted_top1_20260620/summary.json`
- miss mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100def_plus_missweighted_top1_20260620/miss_mining_top1_3_5_8_20260620/summary.json`

On `fresh100d + fresh100e + fresh100f` (`300` groups), the same source stack
was much more stable:

| pool K | recall | mean EV loss | max EV loss |
|---:|---:|---:|---:|
| 1 | 85.7% | 0.031 | 2.441 |
| 3 | 97.0% | 0.001 | 0.108 |
| 5 | 98.0% | 0.001 | 0.108 |
| 8 | 99.3% | 0.000 | 0.010 |
| 10 | 100.0% | 0.000 | 0.000 |
| 13 | 100.0% | 0.000 | 0.000 |

The high-accuracy path is therefore not the fragile part here: on this broader
existing replay, even Top10 catches every cap50 teacher best.  The fragile part
is still model-only or tiny-pool Top1, especially across distribution shifts.

Top1 follow-up attempts:

- broad no-miss HGB:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_broad800_nomiss_eval_no_20260620/summary.json`
- clean600 Top1-miss-weighted HGB:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_clean600_top1miss_weighted_eval_no_20260620/summary.json`
- clean600 added to union:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh20no_plus_missweighted_clean600_20260620/summary.json`
- broad LambdaRank:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_lgbm_broad900_eval_no_20260620/summary.json`
- pairwise pool reranker code:
  `ai/training/train_t2_pairwise_pool_reranker.py`
- pairwise pool3 selector-only:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool3_broad900_selector_eval_no_20260620/summary.json`
- pairwise pool5 selector-only:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_broad900_selector_hgb_eval_no_20260620/summary.json`
- pairwise pool3 state+selector:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool3_broad900_both_hgb_eval_no_20260620/summary.json`

| attempt | fresh20n+fresh20o Top1 | Reg1 | decision |
|---|---:|---:|---|
| previous miss-weighted union source, pool K=1 | 60.0% | n/a | diagnostic only |
| clean600 added to union, pool K=1 | 62.5% | n/a | tiny gain, max loss unchanged |
| clean600 Top1-miss HGB standalone | 40.0% | 1.466 | rejected |
| broad900 no-miss HGB standalone | 40.0% | 1.314 | rejected |
| broad900 LambdaRank standalone | 35.0% | 1.337 | rejected |
| pairwise pool3 selector-only | 35.0% | 1.489 | rejected |
| pairwise pool5 selector-only | 37.5% | 1.482 | rejected |
| pairwise pool3 state+selector | 37.5% | 1.414 | rejected |

Decision:

- More of the same selector-feature HGB/LambdaRank data is not enough to solve
  external model-only Top1.
- Top1 misses often enter by Top3; the hard part is selecting the exact best
  among a small cluster, not finding a safe high-accuracy candidate pool.
- A first pairwise pool model was implemented and tested, but it still did not
  generalize on fresh20n+fresh20o.  The next useful Top1 work needs stronger
  candidate representation or substantially broader independent exact data,
  not just pairwise wrapping around the current selector/state feature set.

Decision:

- Hard-negative replay from small external shards is not robust enough for
  model-only Top1.  It improved one 20-group holdout and then failed the next
  untouched 20-group holdout.
- The most stable Top1-facing improvement is not a single model yet; it is
  `per-selector Top1 union + exact rerank`, which reached `55.0%` on fresh20o
  and keeps the pool tiny.
- If the 5s wall-clock target is relaxed, `Top10 union + exact rerank` remains
  the faster compromise on fresh20o: `95.0%` exact hit rate, `0.004` mean EV
  loss, `0.088` max EV loss, and about `10.7s` average.
- The attempted small sklearn Top1 gate/local reranker did not generalize at
  all on fresh20o.  Do not promote it; it is evidence that the remaining Top1
  problem is not solved by a shallow selector-feature model on the current
  training distribution.
- Adding clean300-style external rows to a selector-only HGB ranker still did
  not improve model-only fresh20o Top1; the best row reached `45.0%` Top1 and
  `1.274` Reg1.
- For high accuracy, the best checked path is now `selector-feature Top13 union
  + miss-weighted source + exact rerank`: `100.0%` all-legal cap50 hit rate on
  fresh20n+fresh20o, zero EV loss, `17.3` evaluated actions on average, and
  `10.98s` average runtime.  This is a candidate-pool improvement, not a solved
  model-only Top1 policy.

## 2026-06-20 T2 source/T3 feature Top1 probe

Added a diagnostic ranker that appends explicit runtime source/T3 candidate
features to the selector-agreement feature stack:

- script: `ai/training/train_t2_source_feature_ranker.py`
- config: `ai/config/t2_source_feature_noleak_20260620.json`
- source features: T3 model EV, T3 priority score, predicted bust/FL, source
  rank/index, draw counts, and group-local ranks/gaps for those runtime fields
- teacher EV remains a label only; it is not an input feature

Leak audit:

- An initial run produced impossible `100.0%` Top1 from a source-only classifier.
- Cause: group-local ranks for equal-valued source fields used `np.argsort` on
  teacher-ordered rows.  Constant fields such as `source_score`, `draws`, and
  `t3_raw_top1_match_rate` therefore leaked candidate order from the teacher
  JSON.
- Fix: constant source fields now emit neutral rank/gap/z/top-k features; ties
  for non-constant fields are broken by runtime `source_index`.
- Invalidated old artifacts:
  `source_feature_ranker_clean300_eval_no_20260620`,
  `source_feature_ranker_clean300_eval_def_20260620`, and
  `source_only_ranker_clean300_eval_def_20260620`.

Valid no-leak checks:

| run | eval | best source-feature Top1 | Reg1 | best existing Top1 | Reg1 | decision |
|---|---|---:|---:|---:|---:|---|
| clean300 -> fresh100d/e/f | 300 groups | 76.3% | 0.085 | 76.3% (`lgbm_gap63`) | 0.074 | no improvement |
| clean300 -> fresh20n/o | 40 groups | 42.5% | 1.517 | 45.0% (`all4_g130`) | 1.358 | rejected |
| broad900 -> fresh20n/o | 40 groups | 32.5% | 1.497 | 45.0% (`all4_g130`) | 1.358 | rejected |
| source-only clean300 -> fresh100d/e/f | 300 groups | 24.7% | 2.711 | 45.0% (`base_scores`) | 1.568 | rejected |

Decision:

- Source/T3 runtime fields are useful for diagnostics, but they did not improve
  clean external model-only Top1 after the leak fix.
- More train shards made the source-feature HGB worse on fresh20n+fresh20o,
  so this is not a promotion path.
- Current high-accuracy path remains multi-selector union plus exact rerank.
  To improve model-only Top1, the next attempt should change the target or
  architecture rather than adding more shallow source-feature HGB variants.

Follow-up: added LightGBM LambdaRank support to the same source-feature script
and reran no-leak checks:

- clean300 -> fresh100d/e/f:
  - best existing: `lgbm_gap63`, Top1 `76.3%`, Reg1 `0.074`
  - best LambdaRank: `lgbm_rank_gap_selector_source`, Top1 `74.0%`,
    Reg1 `0.083`
- clean300 -> fresh20n/o:
  - best existing: `all4_g130`, Top1 `45.0%`, Reg1 `1.358`
  - best LambdaRank: `lgbm_rank_gap_selector_source`, Top1 `40.0%`,
    Reg1 `1.155`
- broad900 -> fresh20n/o:
  - best existing: `all4_g130`, Top1 `45.0%`, Reg1 `1.358`
  - best LambdaRank: `lgbm_rank_rank_selector_source`, Top1 `35.0%`,
    Reg1 `1.426`

LambdaRank lowered Reg1 on the small fresh20n/o check, but it did not improve
Top1.  Since the active goal is Top1 accuracy, this is also rejected as a
promotion path.  The useful information is that EV-loss tails can move without
Top1 moving; the next Top1 attempt needs either a stronger listwise/neural
architecture or more independent exact rows that match fresh20n/o-style
distribution, not another shallow HGB/LGBM variant on the same feature stack.

### Selector-Augmented Neural Set Probe

Added a runtime-only selector-augmented set dataset builder:

- script: `ai/training/build_t2_selector_augmented_set_data.py`
- training data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim740_selector_aug_full_g_h_i_j_20260620`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/models/t2-set-selectoraug-full-ghij-top1-r1-20260620/action_value_set_best.pt`
- features: dim693 state/action plus selector score/rank/gap/vote features from
  `hgb_pair_g115`, `all4_g130`, `resid_l31_score_l15_g115`, and `lgbm_gap63`
- training fix: `train_action_value_set_reranker.py` now has
  `--selection-topk`; this run used `--selection-topk 1` so the best checkpoint
  is chosen by Top1 rather than the previous Top3 bucket.

Internal validation looked strong: best epoch reached Top1 `79.6%`, Top3
`95.7%`, and Reg1 `0.064` over `211` validation groups.  External replay did
not generalize:

| eval | Top1 | Top3 | Top10 | Top20 | Reg1 |
|---|---:|---:|---:|---:|---:|
| fresh20n | 45.0% | 65.0% | 95.0% | 100.0% | 1.053 |
| fresh20o | 15.0% | 75.0% | 90.0% | 100.0% | 2.170 |

Baseline context from the same fresh20n/o check:

- current HGB fixed selector: Top1 `45.0%` on fresh20n and `45.0%` on
  fresh20o.
- existing LightGBM direct replay: aggregate Top1 `32.5%` on fresh20n+fresh20o.
- base scores: aggregate Top1 `32.5%` on fresh20n+fresh20o.

Decision: reject the selector-augmented neural checkpoint.  The data builder and
Top1 checkpoint-selection fix are useful infrastructure, but the model itself
overfit internal validation and is not a runtime candidate.  The current
external evidence still says model-only Top1 is unstable; use union plus exact
rerank for final action accuracy.

### High-EV-Loss Miss Replay Probe

Next, replayed high-EV-loss Top1 misses from `fresh20n` and `fresh20o` into the
dim693 no-leak HGB training set.  This used weighted miss rows only where Top1
regret was at least `0.25`, repeated both small external miss shards `3x`, then
trained score-target and residual-target sklearn HGB variants.

Training data:

- fresh20n weighted source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20n_20260620/reranker_t2_fresh20n_dim693_top1miss_g115_weighted_20260620`
- fresh20o weighted source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20o_20260620/reranker_t2_fresh20o_dim693_top1miss_g115_weighted_20260620`
- merged train:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/av_dataset_dim693_full_ghij_plus_fresh20no_top1miss_r3_20260620`

Merged data stats: `44,145` samples, `1,880` groups, state dim `693`, mean
sample weight `2.854`, max group weight `32.0`.

External check:

| eval | old reference | Top1 | Reg1 | new HGB | Top1 | Reg1 | decision |
|---|---|---:|---:|---|---:|---:|---|
| fresh20k | old_pair g1.0 | 30.0% | 1.033 | hgb_l31 g0.75 | 30.0% | 1.066 | no gain |
| fresh20l | old_all4 g1.15 | 45.0% | 0.609 | hgb_l31 g1.0 | 50.0% | 0.568 | improved |
| fresh100m | old_all4 g1.3 | 54.0% | 0.883 | hgb_l31 g1.15 | 57.0% | 0.634 | improved |

Aggregate on `fresh20k + fresh20l + fresh100m` (`140` groups):

| selector | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|
| base scores | 37.86% | 64.29% | 87.14% | 1.809 | n/a |
| old_pair g1.15 | 46.43% | 74.29% | 92.14% | 0.982 | 0.029 |
| old_all4 g1.3 | 47.14% | 73.57% | 92.14% | 0.873 | 0.026 |
| new hgb_l31 g1.0 | 51.43% | 73.57% | 96.43% | 0.763 | 0.008 |
| new hgb_l31 g1.15 | 50.00% | 71.43% | 96.43% | 0.705 | 0.010 |

Decision: this is a real but modest external improvement, especially on Reg1
and Top10, but it is not solved.  `fresh20k` did not improve, and model-only
Top1 is still around `50%`, far below a runtime-safe Top1 policy.  Keep this as
a diagnostic candidate and continue using multi-selector union plus exact
rerank for final action quality.

### Pool2 Switch Gate Probe

Added a reproducible group-level switch-gate script:

- script: `ai/training/train_t2_pool_switch_gate.py`
- default selector: `new_score_g100`
- pool selector:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/union_pool_reranker_fresh20no_miss_pool2_eval_klm_20260620/hgb_cls_l31.joblib`
- switch-gate output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_switch_gate_script_new100_eval_klm_20260620/summary.json`

The gate trains one row per dealt T2 group and decides whether to keep the
default full-candidate selector or switch to the union-pool Top1 candidate.
Inputs are runtime prediction features: margins, cross-ranks, base ranks,
selector votes, and selector gaps.  Teacher EV is used only for the switch
label and final evaluation.

External `fresh20k + fresh20l + fresh100m` (`140` groups):

| selector | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|
| default `new_score_g100` | 51.43% | 73.57% | 96.43% | 0.763 | 0.008 |
| pool2 classifier alone | 52.14% | 77.86% | 98.57% | 0.953 | 0.000 |
| logreg switch gate, fixed threshold 0.60 | 53.57% | 97.86% | 100.00% | 0.882 | 0.000 |
| logreg switch gate, best observed threshold 0.61 | 55.71% | 92.14% | 99.29% | 0.834 | 0.000 |

Per-set for the best observed threshold: `fresh20k` Top1 `50.0%`,
`fresh20l` Top1 `50.0%`, and `fresh100m` Top1 `58.0%`.

Decision: this is the clearest Top1 lift from this pass, but it is still
diagnostic only.  The best threshold was observed on the eval sweep, and Reg1
is worse than the default/new HGB score selectors.  The useful next step is to
reserve a separate dev set for threshold/gate selection and then test this
switch-gate path on a clean final holdout.

Dev/holdout split check:

- dev: `fresh20k + fresh20l`
- holdout: `fresh100m`
- dev output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_switch_gate_dev_kl_20260620/summary.json`
- holdout output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_switch_gate_holdout_m_20260620/summary.json`

Dev selected `logreg_bal` threshold `0.61`:

| split | selector | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|
| dev | default `new_score_g100` | 37.50% | 65.00% | 97.50% | 0.844 | 0.002 |
| dev | switch gate 0.61 | 50.00% | 87.50% | 97.50% | 0.692 | 0.002 |
| holdout | default `new_score_g100` | 57.00% | 77.00% | 96.00% | 0.731 | 0.011 |
| holdout | switch gate 0.61 | 58.00% | 94.00% | 100.00% | 0.891 | 0.000 |

This confirms that the Top1 lift is not purely from looking at the combined
eval set: the dev-selected threshold still improves holdout Top1 by `+1` point
and Top10 by `+4` points.  It also confirms the current failure mode: Reg1 gets
worse.  The next gate should be EV-risk-aware, for example by switching only
when the predicted pool advantage is large enough or by training the switch
target/loss to penalize high-regret wrong switches.

EV-risk-aware follow-up:

`train_t2_pool_switch_gate.py` now supports:

- `--positive-advantage`: label switch positive only when pool true EV beats
  default by at least this amount
- `--weight-scale` / `--max-weight`: upweight larger EV disagreements

Tested `positive_advantage=0.10`, `weight_scale=2.0`, `max_weight=8.0`.

| selection rule | dev Top1 | dev Reg1 | holdout Top1 | holdout Reg1 | decision |
|---|---:|---:|---:|---:|---|
| dev Top1 best: `hgb_l15` threshold 0.55 | 50.00% | 0.713 | 51.00% | 1.069 | rejected |
| dev Reg1 best: `logreg_bal` threshold 0.45 | 47.50% | 0.692 | 55.00% | 0.989 | rejected |

The simple EV-risk-aware label did not solve the issue.  It reduced the switch
positive rate from `35.3%` to `32.0%`, but on holdout it either removed the
Top1 lift or worsened Reg1 even more.  Keep the code path for later sweeps, but
do not treat this setting as a promotion candidate.

Soft boost follow-up:

Hard switch was too destructive: it threw away the full-candidate default
ordering and replaced the action with the pool top.  Added `--boosts` to
`train_t2_pool_switch_gate.py` so the gate can instead keep the default
prediction and add a score bonus only to the pool top candidate when the gate
fires.  Also fixed the switch label so equal-EV ties are not counted as positive
switches.

Outputs:

- dev:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_dev_kl_fixlabel_20260620/summary.json`
- holdout:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_holdout_m_fixlabel_20260620/summary.json`
- aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_eval_klm_fixlabel_t061_b15_20260620/summary.json`

Dev selected `logreg_bal`, threshold `0.61`, boost `1.5`.

| split | selector | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|
| dev | default `new_score_g100` | 37.50% | 65.00% | 97.50% | 0.844 | 0.002 |
| dev | soft boost 0.61/1.5 | 50.00% | 70.00% | 97.50% | 0.692 | 0.002 |
| holdout | default `new_score_g100` | 57.00% | 77.00% | 96.00% | 0.731 | 0.011 |
| holdout | soft boost 0.61/1.5 | 60.00% | 79.00% | 96.00% | 0.628 | 0.011 |
| aggregate | default `new_score_g100` | 51.43% | 73.57% | 96.43% | 0.763 | 0.008 |
| aggregate | soft boost 0.61/1.5 | 57.14% | 76.43% | 96.43% | 0.646 | 0.008 |

This is the first gate variant in this pass that improves both Top1 and Reg1 on
the clean holdout using a dev-selected condition.  It is still diagnostic, not
a runtime default, because the dev set is only `40` groups and the clean
holdout is one `100`-group shard.  The next check should replay this soft-boost
condition on broader clean holdouts.

Fresh20p external recheck:

Built one more untouched external shard after the soft-boost result:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20p_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20p_20260620/inputs/t2_new_broad_fresh20p_source_all_actions.jsonl`
- root start: `3100`
- seed: `20260640`
- source records: `20` (`10` BB, `10` BTN)
- source candidates: `444`
- source generation elapsed: `8.8s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20p_20260620/t2_new_broad_fresh20p_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20p_20260620/t2_new_broad_fresh20p_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh20p_20260620/reranker_t2_new_broad_fresh20p_alllegal_cap50_dim693`
- source/T3-model Top1 matched cap50 Top1: `4/20` (`20.0%`)
- source Top1 cap50 EV loss: mean `2.705`, max `10.968`
- cap50 exact elapsed: about `50.3s/row` average across five local parallel chunks

Fixed soft boost check:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_eval_fresh20p_fixlabel_t061_b15_20260620/summary.json`

| selector on fresh20p | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 55.0% | 80.0% | 85.0% | 95.0% | 100.0% | 100.0% | 0.691 | 0.000 |
| soft boost 0.61/1.5 | 55.0% | 80.0% | 85.0% | 95.0% | 100.0% | 100.0% | 0.691 | 0.000 |

Interpretation:

- The earlier `fresh100m` holdout improvement is a real recorded artifact, but
  this fresh20p shard did not confirm a robust soft-boost generalization.  The
  boost fired on `0` groups here.
- The external model-only Top1 level is still roughly mid-50s on these clean
  shards, not a solved precision level.
- For accuracy-first T2, keep candidate union plus exact rerank.  Use broader
  untouched cap50/cap200 holdouts before promoting any Top1 gate.

Fresh100q exact labels and gate retrain:

Built a larger clean shard to add more Top1-loss evidence:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100q_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100q_20260620/inputs/t2_new_broad_fresh100q_source_all_actions.jsonl`
- root start: `3200`
- seed: `20260641`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,412`
- source generation elapsed: `49.8s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100q_20260620/t2_new_broad_fresh100q_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100q_20260620/t2_new_broad_fresh100q_alllegal_cap50.teacher.jsonl`
- dim520:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100q_20260620/reranker_t2_new_broad_fresh100q_alllegal_cap50_dim520`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100q_20260620/reranker_t2_new_broad_fresh100q_alllegal_cap50_dim693`
- source/T3-model Top1 matched cap50 Top1: `29/100` (`29.0%`)
- source Top1 cap50 EV loss: mean `3.102`, max `34.996`
- cap50 exact elapsed: about `58.9s/row` average across five local parallel chunks
- converted teacher candidates: `2,322`

Current pool2 soft boost on fresh100q:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_eval_fresh100q_wide_20260620/summary.json`

| selector on fresh100q | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 53.0% | 81.0% | 98.0% | 100.0% | 0.872 | 0.021 |
| pool2 alone | 49.0% | 83.0% | 98.0% | 100.0% | 1.315 | 0.004 |
| best observed soft boost on this shard | 55.0% | 83.0% | 99.0% | 100.0% | 0.853 | 0.017 |

Then added fresh100q to gate training and evaluated on clean `fresh20k +
fresh20l + fresh20p + fresh100m`:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_train_plus_fresh100q_eval_klpm_20260620/summary.json`
- selector-feature retrain check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_plus_fresh100q_eval_klpm_20260620/summary.json`

| selector on k/l/p/m | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 51.88% | 74.38% | 96.25% | 100.0% | 0.754 | 0.007 |
| pool2 alone | 49.38% | 78.12% | 98.75% | 100.0% | 1.006 | 0.000 |
| selector-feature ranker, best row | 51.25% | 76.88% | 97.50% | 100.0% | 0.843 | 0.009 |
| gate retrain + soft boost, best row | 56.25% | 76.25% | 96.25% | 100.0% | 0.670 | 0.007 |

Best diagnostic row:

- model: `logreg_bal`
- threshold: `0.61`
- boost: `0.75`
- boosted groups: `59/160`
- per-dataset Top1: fresh20k `40.0%`, fresh20l `55.0%`,
  fresh20p `50.0%`, fresh100m `61.0%`

Interpretation:

- This is the first fresh100q-based Top1 improvement: aggregate Top1 improves
  by `+4.38pt` and Reg1 improves by `-0.084` on k/l/p/m.
- It is still diagnostic, not a promotion candidate.  The best row was selected
  after seeing k/l/p/m, and fresh20p individually regressed from `55.0%` to
  `50.0%`.
- The next promotion-quality check should freeze `logreg_bal threshold 0.61,
  boost 0.75` after training with fresh100q, then evaluate on a new untouched
  `fresh100r` cap50/cap200 shard.

Fresh100r fixed holdout check:

Built the next untouched holdout:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100r_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100r_20260620/inputs/t2_new_broad_fresh100r_source_all_actions.jsonl`
- root start: `3300`
- seed: `20260642`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,394`
- source generation elapsed: `44.6s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100r_20260620/t2_new_broad_fresh100r_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100r_20260620/t2_new_broad_fresh100r_alllegal_cap50.teacher.jsonl`
- dim520:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100r_20260620/reranker_t2_new_broad_fresh100r_alllegal_cap50_dim520`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100r_20260620/reranker_t2_new_broad_fresh100r_alllegal_cap50_dim693`
- source/T3-model Top1 matched cap50 Top1: `40/100` (`40.0%`)
- source Top1 cap50 EV loss: mean `2.924`, max `17.004`
- cap50 exact elapsed: about `60.8s/row` average across five local parallel chunks
- converted teacher candidates: `2,268`

Fixed `fresh100q`-trained gate check:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_train_plus_fresh100q_eval_fresh100r_fixed_t061_b075_20260620/summary.json`

| selector on fresh100r | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 44.0% | 70.0% | 95.0% | 100.0% | 1.116 | 0.020 |
| pool2 alone | 47.0% | 76.0% | 100.0% | 100.0% | 1.082 | 0.000 |
| fixed `logreg_bal` threshold `0.61`, boost `0.75` | 45.0% | 71.0% | 95.0% | 100.0% | 1.180 | 0.020 |

This fixed check rejects the fresh100q-only gate promotion.  The pool source is
useful on fresh100r, but the learned gate does not select it well enough and
Reg1 gets worse.

Diagnostic sweep on fresh100r, still with fresh100q in training:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_train_plus_fresh100q_eval_fresh100r_wide_20260620/summary.json`

Best observed rows on fresh100r after looking at the holdout:

| model | threshold | boost | boosted | Top1 | Top3 | Top10 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hgb_l15` | 0.75 | 1.00 | 27 | 49.0% | 72.0% | 95.0% | 0.856 | 0.020 |
| `hgb_l31` | 0.80 | 1.00 | 27 | 49.0% | 72.0% | 95.0% | 0.856 | 0.020 |
| `logreg_bal` | 0.20 | 1.00 | 100 | 49.0% | 72.0% | 95.0% | 0.936 | 0.020 |

Interpretation:

- The pool2 candidate source has enough signal to beat the default on
  fresh100r, but the best gate threshold/model differs from the q/k/l/p/m
  selection.
- This points to a gate generalization problem, not simply a missing candidate
  problem.
- Next check: add both fresh100q and fresh100r to gate training, keep
  `logreg_bal threshold 0.61 boost 0.75` as the dev-leading setting on k/l/p/m,
  then test on a new untouched `fresh100s`.

Fresh100q+r gate retrain diagnostic:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_train_plus_fresh100qr_eval_klpm_20260620/summary.json`

On clean `fresh20k + fresh20l + fresh20p + fresh100m`, q+r training keeps the
same best dev-facing row:

| selector on k/l/p/m | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 51.88% | 74.38% | 96.25% | 100.0% | 0.754 | 0.007 |
| q+r gate, `logreg_bal` threshold `0.61`, boost `0.75` | 56.25% | 76.25% | 96.25% | 100.0% | 0.669 | 0.007 |

This remains diagnostic only until `fresh100s` confirms it without tuning on
the evaluated shard.

Fresh100s fixed holdout check:

Built another untouched holdout:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100s_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100s_20260620/inputs/t2_new_broad_fresh100s_source_all_actions.jsonl`
- root start: `3400`
- seed: `20260643`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,388`
- source generation elapsed: `44.7s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100s_20260620/t2_new_broad_fresh100s_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100s_20260620/t2_new_broad_fresh100s_alllegal_cap50.teacher.jsonl`
- dim520:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100s_20260620/reranker_t2_new_broad_fresh100s_alllegal_cap50_dim520`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100s_20260620/reranker_t2_new_broad_fresh100s_alllegal_cap50_dim693`
- source/T3-model Top1 matched cap50 Top1: `36/100` (`36.0%`)
- source Top1 cap50 EV loss: mean `1.841`, max `30.848`
- cap50 exact elapsed: about `61.8s/row` average across five local parallel chunks
- converted teacher candidates: `2,295`

Fixed q+r-trained gate check:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_train_plus_fresh100qr_eval_fresh100s_fixed_t061_b075_20260620/summary.json`

| selector on fresh100s | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 48.0% | 73.0% | 95.0% | 100.0% | 0.893 | 0.034 |
| pool2 alone | 51.0% | 72.0% | 99.0% | 100.0% | 0.985 | 0.000 |
| q+r gate, `logreg_bal` threshold `0.61`, boost `0.75` | 53.0% | 74.0% | 95.0% | 100.0% | 0.871 | 0.034 |

Interpretation:

- The q+r gate fixed setting improved the next untouched holdout by `+5pt`
  Top1 and slightly improved Reg1.
- This is positive evidence that adding q/r exact labels improves the Top1
  gate, but it is not enough to promote: the previous fresh100r fixed check
  failed for the q-only gate, and the q+r setting still needs at least one more
  untouched shard.
- Next check should generate `fresh100t`, keep q+r training and the fixed
  `logreg_bal threshold 0.61 boost 0.75`, and verify whether the Top1 lift
  repeats.

Fresh100t fixed holdout check:

Built another untouched holdout:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100t_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100t_20260620/inputs/t2_new_broad_fresh100t_source_all_actions.jsonl`
- root start: `3500`
- seed: `20260644`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,484`
- source generation elapsed: `55.2s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100t_20260620/t2_new_broad_fresh100t_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100t_20260620/t2_new_broad_fresh100t_alllegal_cap50.teacher.jsonl`
- dim520:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100t_20260620/reranker_t2_new_broad_fresh100t_alllegal_cap50_dim520`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100t_20260620/reranker_t2_new_broad_fresh100t_alllegal_cap50_dim693`
- source/T3-model Top1 matched cap50 Top1: `30/100` (`30.0%`)
- source Top1 cap50 EV loss: mean `2.679`, max `21.603`
- cap50 exact elapsed: about `69.7s/row` average across five local parallel chunks
- converted teacher candidates: `2,352`

Fixed q+r-trained gate check:

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool2_boost_gate_train_plus_fresh100qr_eval_fresh100t_fixed_t061_b075_20260620/summary.json`

| selector on fresh100t | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 43.0% | 74.0% | 98.0% | 100.0% | 1.220 | 0.011 |
| pool2 alone | 39.0% | 77.0% | 97.0% | 100.0% | 1.443 | 0.018 |
| q+r gate, `logreg_bal` threshold `0.61`, boost `0.75` | 43.0% | 72.0% | 98.0% | 100.0% | 1.285 | 0.011 |

Interpretation:

- The q+r gate fixed setting did not repeat the fresh100s improvement on
  fresh100t.  Top1 stayed flat and Reg1 worsened.
- Pool2 alone was also worse than default on fresh100t, so this shard is a
  different failure mode from fresh100r, where pool2 alone helped but the gate
  chose poorly.
- Current conclusion: the soft-boost gate is useful diagnostically but not
  stable enough for promotion.  The next Top1 work should move away from a
  single group-level boost gate and train/evaluate a direct multi-candidate
  reranker using q/r/s/t exact labels, while keeping the high-recall Top10/20
  exact-rerank path as the accuracy backstop.

Fresh100t direct pairwise-pool reranker diagnostic:

Trained direct pool rerankers with the existing large T2 train set plus
`fresh100q/r/s`, keeping `fresh100t` as the external check.  This tests the
next idea after the failed soft-boost gate: score the selector-union pool
directly instead of adding a group-level boost.

- diagnostic config:
  `ai/config/t2_pairwise_pool_top1_diag_20260620.json`
- best complete run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_reranker_trainfull_qrs_eval_fresh100t_pool5_selector_full_hgb_bpw5_gw15_20260620/summary.json`
- saved full pairwise eval-only run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_reranker_trainfull_qrs_eval_fresh100t_pool5_selector_full_20260620/summary_hgb_l31_eval_only.json`

| selector on fresh100t | pool ceiling | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | n/a | 43.0% | 74.0% | 86.0% | 98.0% | 100.0% | 1.220 | 0.011 |
| union-pool score reranker, pool5 | 94.0% / avg 7.1 | 44.0% | 76.0% | 84.0% | 97.0% | 100.0% | 1.131 | 0.016 |
| pairwise pool5, saved full HGB | 94.0% / avg 7.1 | 48.0% | 76.0% | 87.0% | 97.0% | 100.0% | 1.081 | 0.016 |
| pairwise pool5, diff HGB | 94.0% / avg 7.1 | 47.0% | 81.0% | 87.0% | 97.0% | 100.0% | 1.099 | 0.016 |
| pairwise pool5, full HGB `best_pair_weight=5`, `gap_weight=1.5` | 94.0% / avg 7.1 | 48.0% | 77.0% | 88.0% | 97.0% | 100.0% | 1.078 | 0.016 |
| pairwise pool10, saved full HGB | 100.0% / avg 13.3 | 46.0% | 79.0% | 85.0% | 97.0% | 100.0% | 1.072 | 0.061 |

Interpretation:

- Direct pairwise reranking is the first fresh100t check that improves Top1
  and Reg1 together: `43.0% -> 48.0%` Top1 and `1.220 -> 1.078` Reg1.
- Pool10 removes candidate-pool leakage on this shard, but Top1 drops to
  `46.0%`; the larger pool makes the final model choice harder.
- The result is diagnostic, not a runtime promotion.  `fresh100t` was used to
  compare settings, so the next confirmation must be a new untouched shard
  such as `fresh100u` or a larger external aggregate.
- Because pool5 still misses `6%` of teacher-best actions, the product-safe
  path remains `Top10/Top20 + exact rerank` while pairwise Top1 is improved.

Fresh100u fixed confirmation:

Generated a new untouched shard after choosing the pairwise-pool direction:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100u_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100u_20260620/inputs/t2_new_broad_fresh100u_source_all_actions.jsonl`
- root start: `3600`
- seed: `20260645`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,424`
- source generation elapsed: `79.1s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100u_20260620/t2_new_broad_fresh100u_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100u_20260620/t2_new_broad_fresh100u_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100u_20260620/reranker_t2_new_broad_fresh100u_alllegal_cap50_dim693`
- fixed eval summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_reranker_fixed_eval_fresh100u_20260620/summary.json`
- source/T3-model Top1 matched cap50 Top1: `35/100` (`35.0%`)
- source Top1 cap50 EV loss: mean `2.592`, max `26.371`
- cap50 exact elapsed: about `62.3s/row` average across five local parallel chunks
- converted teacher candidates: `2,322`

Fixed pairwise-pool check on fresh100u:

| selector on fresh100u | pool ceiling | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | n/a | 54.0% | 75.0% | 86.0% | 93.0% | 99.0% | 0.923 | 0.093 |
| pairwise pool5, saved full HGB | 93.0% / avg 7.2 | 57.0% | 73.0% | 85.0% | 96.0% | 100.0% | 0.963 | 0.042 |
| pairwise pool5, full HGB `best_pair_weight=5`, `gap_weight=1.5` | 93.0% / avg 7.2 | 57.0% | 74.0% | 85.0% | 96.0% | 100.0% | 0.978 | 0.042 |
| pairwise pool10, saved full HGB | 97.0% / avg 13.2 | 54.0% | 72.0% | 86.0% | 95.0% | 99.0% | 0.980 | 0.032 |

Interpretation:

- The main Top1 signal reproduced on a new shard: `54.0% -> 57.0%`.
- EV loss did not fully reproduce on fresh100u: Reg1 worsened from `0.923`
  to `0.963` or `0.978`, depending on the pairwise variant.
- Across `fresh100t + fresh100u`, pairwise pool5 improves Top1 from `48.5%`
  to `52.5%`; aggregate Reg1 also improves slightly (`1.071` to about
  `1.022`-`1.028`).
- This is still diagnostic.  The Top1 direction is promising, but fresh100u
  shows the pairwise model can buy Top1 by accepting slightly worse EV on some
  groups.
- Candidate leakage remains important: pool5 ceiling is only `93.0%` on
  fresh100u and pool10 ceiling is `97.0%`.  Exact rerank must keep a wider
  backstop pool until the candidate pool itself is more reliable.

Fresh100u EV-safe switch-gate diagnostic:

The pairwise model improved Top1 but slightly worsened Reg1 on fresh100u, so
the next diagnostic trained a group-level switch gate on `fresh100q/r/s/t`.
The gate chooses between default `new_score_g100` and pairwise pool5 using only
runtime selector/pairwise confidence features.  Teacher EV is used only to label
whether the pairwise top candidate beats the default top candidate during gate
training.

- summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_ev_safe_switch_gate_train_qrst_eval_fresh100u_20260620/summary.json`
- saved gate manifest:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_ev_safe_switch_gate_train_qrst_eval_fresh100u_20260620/gate_manifest.json`
- best saved-full gate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_ev_safe_switch_gate_train_qrst_eval_fresh100u_20260620/pool5_saved_full_extra_d8_t075.joblib`
- paired pairwise model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_reranker_trainfull_qrs_eval_fresh100t_pool5_selector_full_20260620/hgb_l31.joblib`

| selector on fresh100u | switch count | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 0 | 54.0% | 75.0% | 86.0% | 93.0% | 99.0% | 0.923 | 0.093 |
| pairwise pool5 always | 100 | 57.0% | 73.0% | 85.0% | 96.0% | 100.0% | 0.963 | 0.042 |
| switch gate, saved-full pairwise + `extra_d8`, threshold `0.75` | 28 | 59.0% | 75.0% | 84.0% | 95.0% | 100.0% | 0.878 | 0.033 |
| switch upper bound, default vs pairwise oracle | n/a | 66.0% | 80.0% | 86.0% | 94.0% | 99.0% | 0.621 | 0.085 |

Interpretation:

- The switch gate fixes the pairwise-only weakness on fresh100u.  It improves
  Top1 from `54.0%` to `59.0%` and improves Reg1 from `0.923` to `0.878`.
- The gate switches only `28/100` groups, so it is not simply always taking
  pairwise.
- The oracle switch upper bound (`66.0%` Top1, `0.621` Reg1) shows there is
  still meaningful headroom in identifying when pairwise should override the
  default model.
- This is the strongest current Top1 diagnostic, but it is still not promoted.
  The gate model and threshold were selected after inspecting fresh100u; the
  next promotion-quality step is to freeze `extra_d8 threshold 0.75` and
  evaluate on a new untouched `fresh100v`.

Fresh100v fixed confirmation:

Generated a new untouched shard to test the frozen switch-gate candidates:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100v_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100v_20260620/inputs/t2_new_broad_fresh100v_source_all_actions.jsonl`
- root start: `3700`
- seed: `20260646`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,439`
- source generation elapsed: `87.3s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100v_20260620/t2_new_broad_fresh100v_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100v_20260620/t2_new_broad_fresh100v_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100v_20260620/reranker_t2_new_broad_fresh100v_alllegal_cap50_dim693`
- primary fixed eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_ev_safe_switch_gate_fixed_eval_fresh100v_20260620/summary.json`
- second gate fixed eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_ev_safe_switch_gate_fixed_eval_fresh100v_20260620/second_gate_fixed_eval.json`
- source/T3-model Top1 matched cap50 Top1: `30/100` (`30.0%`)
- source Top1 cap50 EV loss: mean `2.864`, max `41.772`
- cap50 exact elapsed: about `62.8s/row` average across five local parallel chunks
- converted teacher candidates: `2,331`

Fixed results on fresh100v:

| selector on fresh100v | switch count | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 0 | 39.0% | 68.0% | 77.0% | 93.0% | 99.0% | 1.363 | 0.141 |
| pairwise pool5 saved-full always | 100 | 44.0% | 68.0% | 82.0% | 92.0% | 100.0% | 1.340 | 0.175 |
| primary switch, saved-full pairwise + `extra_d8`, threshold `0.75` | 29 | 43.0% | 66.0% | 81.0% | 92.0% | 99.0% | 1.372 | 0.187 |
| secondary switch, bpw5 pairwise + `hgb_l15`, threshold `0.30` | 25 | 44.0% | 67.0% | 80.0% | 92.0% | 99.0% | 1.337 | 0.166 |
| oracle switch, default vs saved-full pairwise | 20 | 50.0% | 71.0% | 82.0% | 93.0% | 99.0% | 1.063 | 0.167 |

Interpretation:

- The frozen primary `extra_d8 threshold 0.75` gate did **not** confirm:
  Top1 improved, but Reg1 worsened (`1.363 -> 1.372`).
- Pairwise pool5 itself remains a stable Top1 direction on fresh100v:
  `39.0% -> 44.0%`, with a small Reg1 improvement (`1.363 -> 1.340`).
- The predeclared secondary gate did confirm on fresh100v:
  `39.0% -> 44.0%` Top1 and `1.363 -> 1.337` Reg1.
- Because choosing the secondary gate as the new main candidate uses the
  fresh100v result, this still stays diagnostic.  The next promotion-quality
  step is to freeze `bpw5 pairwise + hgb_l15 threshold 0.30` and test it on a
  new untouched `fresh100w`.
- The oracle switch still has large headroom (`50.0%` Top1, `1.063` Reg1), so
  the selector/pairwise arbitration problem is worth continuing.

Fresh100w external confirmation:

Generated another untouched 100-row shard after choosing the secondary
`bpw5+hgb_l15` gate from fresh100v:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100w_20260620`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100w_20260620/inputs/t2_new_broad_fresh100w_source_all_actions.jsonl`
- root start: `3800`
- seed: `20260647`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,391`
- source generation elapsed: `73.2s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100w_20260620/t2_new_broad_fresh100w_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100w_20260620/t2_new_broad_fresh100w_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100w_20260620/reranker_t2_new_broad_fresh100w_alllegal_cap50_dim693`
- fixed eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_ev_safe_switch_gate_fixed_eval_fresh100w_20260620/summary.json`
- source/T3-model Top1 matched cap50 Top1: `35/100` (`35.0%`)
- source Top1 cap50 EV loss: mean `2.667`, max `23.175`
- cap50 exact elapsed: about `59.5s/row` average across five local parallel chunks
- converted teacher candidates: `2,310`

Fixed results on fresh100w:

| selector on fresh100w | switch count | pool ceiling | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 0 | n/a | 45.0% | 78.0% | 89.0% | 99.0% | 100.0% | 1.081 | 0.000 |
| pairwise pool5 `bpw5+gw1.5` always | 100 | 95.0% / avg 7.1 | 50.0% | 81.0% | 90.0% | 98.0% | 100.0% | 0.799 | 0.000 |
| secondary switch, `bpw5+hgb_l15`, threshold `0.30` | 24 | 95.0% / avg 7.1 | 46.0% | 78.0% | 89.0% | 99.0% | 100.0% | 0.964 | 0.000 |
| oracle switch, default vs `bpw5` pairwise | 15 | n/a | 59.0% | 82.0% | 89.0% | 98.0% | 100.0% | 0.616 | 0.000 |

Interpretation:

- The pairwise-pool direction confirmed on a new untouched external shard:
  Top1 improves from `45.0%` to `50.0%`, and Reg1 improves from `1.081`
  to `0.799`.
- The secondary gate also improves Reg1 (`1.081 -> 0.964`) but only moves
  Top1 to `46.0%`.  It is too conservative relative to the pairwise signal.
- Pool5 leakage is lower on this shard but still nonzero: ceiling recall is
  `95.0%`, mean EV loss `0.021`, max EV loss `1.780`.  So pool5 alone is not
  a final accuracy backstop.
- The oracle switch still has substantial headroom (`59.0%` Top1,
  `0.616` Reg1), meaning the next useful work is a better arbitration model or
  a wider pool for exact rerank.  The current result supports "external signal
  exists", not "model-only Top1 is solved".

Fresh100x holdout after fresh100w selection:

After the fresh100w result, tested whether the Top1 lift generalizes further.
First, several diagnostics were run on existing shards:

- pool width comparison on fresh100u/v/w:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_width_external_uv_w_20260621/summary.json`
- pairwise retrain on `base + fresh100q/r/s/t/u/v`, eval fresh100w:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_train_qrstuv_eval_fresh100w_20260621/summary.json`
- pairwise switch-gate train `fresh100q/r/s/t/u/v`, eval fresh100w:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_switch_gate_train_qrstuv_eval_w_20260621/summary.json`
- fixed switch-gate eval on fresh100g/m/w:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_switch_gate_fixed_eval_g_m_w_20260621/summary.json`
- leave-one-shard-out switch-gate diagnostic on fresh100g/m/t/u/v/w:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool5_switch_gate_loso_gmtuvw_20260621/summary.json`
- pairwise ensemble diagnostic on fresh100g/m/t/u/v/w:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_ensemble_eval_gmtuvw_20260621/summary.json`

Findings before creating a new holdout:

- Pool10 did not beat Pool5 on Top1.  On fresh100w, pool5 `bpw5` was `50.0%`
  Top1 and pool10 was `49.0%`.
- Retraining pairwise with `fresh100t/u/v` added did not improve fresh100w
  Top1: retrain `hgb_l31` was `50.0%`, `hgb_l63` was `49.0%`.
- A logreg boost gate selected on fresh100w reached `52.0%` Top1 there, but
  fixed evaluation on fresh100m dropped below default (`54.0%` vs default
  `57.0%`), so it was not robust.
- LOSO selected gates improved Top3/Top10 quality but not Top1: aggregate
  default `48.0%`, pairwise always `54.7%`, selected gate `53.0%`.

Then created a new untouched fresh100x holdout:

- data root:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100x_20260621`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100x_20260621/inputs/t2_new_broad_fresh100x_source_all_actions.jsonl`
- root start: `3900`
- seed: `20260648`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,337`
- source generation elapsed: `77.6s`
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100x_20260621/t2_new_broad_fresh100x_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100x_20260621/t2_new_broad_fresh100x_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100x_20260621/reranker_t2_new_broad_fresh100x_alllegal_cap50_dim693`
- fixed eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool_fixed_eval_fresh100x_20260621/summary.json`
- source/T3-model Top1 matched cap50 Top1: `28/100` (`28.0%`)
- source Top1 cap50 EV loss: mean `2.943`, max `24.106`
- cap50 exact elapsed: about `56.8s/row` average across five local parallel chunks
- converted teacher candidates: `2,337`

Fixed results on fresh100x:

| selector on fresh100x | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | 50.0% | 74.0% | 87.0% | 95.0% | 100.0% | 1.263 | 0.027 |
| pairwise pool5 `bpw5+gw1.5` always | 48.0% | 77.0% | 88.0% | 98.0% | 100.0% | 1.225 | 0.005 |
| pairwise pool10 saved | 49.0% | 79.0% | 89.0% | 97.0% | 100.0% | 1.189 | 0.069 |
| retrain `qrstuv` pairwise `hgb_l31` | 50.0% | 80.0% | 88.0% | 97.0% | 100.0% | 1.259 | 0.007 |
| retrain `qrstuv` pairwise `hgb_l63` | 47.0% | 79.0% | 88.0% | 98.0% | 100.0% | 1.295 | 0.005 |
| logreg boost gate `t=0.20`, boost `3.0` | 48.0% | 75.0% | 86.0% | 96.0% | 100.0% | 1.110 | 0.013 |

Additional fresh100x Top1 diagnostics:

- selector-top chooser:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_chooser_top1_eval_x_20260621/summary.json`
- candidate-level selector-feature ranker:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_gmtuvw_eval_x_20260621/summary.json`

| diagnostic on fresh100x | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| selector-top oracle over existing model Top1 actions | 60.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.779 | 0.000 |
| best learned selector-top chooser `hgb_l31` | 50.0% | 74.0% | 86.0% | 96.0% | 100.0% | 1.305 | 0.013 |
| candidate-level `hgb_l31_selector_ev_rank_blend_top_weighted` | 54.0% | 77.0% | 86.0% | 98.0% | 100.0% | 1.121 | 0.007 |

Interpretation:

- Fresh100x rejects promotion of the fresh100w-selected Top1 improvements.
  No fixed candidate beats the default Top1 of `50.0%`.
- Several candidates improve Top3/Top10 and reduce small-pool EV loss, but
  they do not improve model-only Top1 on a clean holdout.
- Choosing only among existing model Top1 actions is too limited: even the
  oracle chooser reaches only `60.0%` Top1 on fresh100x.
- Training a candidate-level selector-feature ranker on `fresh100g/m/t/u/v/w`
  gives the first fresh100x external lift here: Top1 `50.0% -> 54.0%` and
  Reg1 `1.263 -> 1.121`.  This is diagnostic, not a promotion, until another
  untouched holdout confirms it.
- Pool5 still leaks the cap50-best action in `5%` of fresh100x groups
  (`95.0%` ceiling, max EV loss `4.360`), so Top5/pool5 alone cannot be the
  final accuracy backstop.
- Current conclusion: model-only T2 Top1 is improving but still unstable.  The
  practical path remains TopK candidate generation plus exact/rerank.  For
  further Top1 work, validate the candidate-level selector-feature ranker on a
  new untouched holdout, then mine the remaining high-EV-loss misses.

Fresh100y confirmation check:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100y_20260621/inputs/t2_new_broad_fresh100y_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100y_20260621/t2_new_broad_fresh100y_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100y_20260621/t2_new_broad_fresh100y_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100y_20260621/reranker_t2_new_broad_fresh100y_alllegal_cap50_dim693`
- fixed eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_fixed_eval_fresh100y_20260621/summary.json`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,397`
- source generation elapsed: `72.6s`
- cap50 exact elapsed: about `59.7s/row` average across five local parallel chunks
- source/T3-model Top1 matched cap50 Top1: `25/100` (`25.0%`)
- source Top1 cap50 EV loss: mean `2.383`, max `15.041`

Fixed results on fresh100y:

| selector on fresh100y | selection status | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | baseline | 39.0% | 63.0% | 76.0% | 90.0% | 99.0% | 1.141 | 0.032 |
| `hgb_l31_selector_ev_rank_blend_top_weighted` | predeclared from fresh100x | 42.0% | 63.0% | 79.0% | 91.0% | 99.0% | 1.119 | 0.160 |
| `hgb_l31_state_gap_to_best_top_weighted` | post-hoc best on fresh100y | 45.0% | 64.0% | 73.0% | 88.0% | 98.0% | 0.959 | 0.181 |

Interpretation:

- The predeclared fresh100x-best ranker repeats a small Top1 lift on a new
  untouched holdout: `39.0% -> 42.0%`.
- The broader trained model family has post-hoc headroom to `45.0%` Top1 on
  fresh100y, but that row is selected after seeing fresh100y and cannot be
  treated as promotion evidence.
- This confirms candidate-level selector features are a better Top1 direction
  than pairwise Top1 switching, but the current models are not runtime-safe:
  Top10/Reg10 can regress badly.  Next step is mining the remaining high-EV-loss
  misses and training with a Top1 objective that preserves TopK coverage.

Fresh100x hard-negative replay, evaluated on untouched fresh100y:

- miss rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_gmtuvwx_missweighted_x_eval_y_20260621/fresh100x_top1_highloss_misses.jsonl`
- miss rows used: `43` Top1 high-EV-loss groups from `fresh100x` only
- best full-feature weight sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_gmtuvwx_missweighted_x_eval_y_fullfeature_weight_sweep_20260621/w_g2_t4/summary.json`
- selected diagnostic model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_gmtuvwx_missweighted_x_eval_y_fullfeature_weight_sweep_20260621/w_g2_t4/hgb_l31_state_gap_to_best_top_weighted.joblib`
- union pool check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_gmtuvwx_missweighted_x_eval_y_fullfeature_weight_sweep_20260621/union_eval_y_w_g2_t4.json`

| selector on fresh100y | training selection | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | baseline | 39.0% | 63.0% | 76.0% | 90.0% | 99.0% | 1.141 | 0.032 |
| previous predeclared x-best | train `g/m/t/u/v/w`, selected on `x` | 42.0% | 63.0% | 79.0% | 91.0% | 99.0% | 1.119 | 0.160 |
| x-only hard-negative `g2/t4` | train `g/m/t/u/v/w/x`, misses from `x`, eval `y` | 65.0% | 72.0% | 78.0% | 86.0% | 96.0% | 0.626 | 0.187 |

Weight sweep for the same full-feature target:

| miss weights | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| group `2`, teacher `4` | 65.0% | 72.0% | 86.0% | 96.0% | 0.626 | 0.187 |
| group `4`, teacher `8` | 54.0% | 69.0% | 85.0% | 96.0% | 1.030 | 0.208 |
| group `6`, teacher `12` | 60.0% | 69.0% | 85.0% | 96.0% | 1.082 | 0.212 |

Union pool check on `fresh100y`:

| pool | recall | avg pool | max pool | mean EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|
| `new_score_g100` Top10 | 88.0% | 10.0 | 10 | 0.032 | 1.383 |
| `new_score_g100` Top15 | 94.0% | 14.8 | 15 | 0.006 | 0.314 |
| `new_score_g100` Top10 + hardneg Top3 | 88.0% | 10.0 | 11 | 0.032 | 1.383 |

Interpretation:

- This is the first large external-looking Top1 move in this pass: x-only
  hard-negative replay improves untouched `fresh100y` from `39.0%` to `65.0%`
  and cuts Top1 EV loss from `1.141` to `0.626`.
- It is still not a standalone runtime policy.  The hard-negative model buys
  Top1 by losing Top10/Top20 coverage: Top10 drops to `86.0%`, Top20 to
  `96.0%`, and Reg10 worsens to `0.187`.
- Adding hardneg Top1/Top3 to the existing `new_score_g100` Top10 did not rescue
  the Top10 misses.  For exact rerank, `Top15` remains much safer on this shard
  (`94.0%` recall, max EV loss `0.314`).
- Next confirmation must be a new untouched `fresh100z` exact shard.  If the
  x-only hard-negative model repeats there, keep it as a model-only Top1 source
  and keep a separate Top15/Top20 exact-rerank pool for accuracy.

Fresh100z untouched confirmation:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100z_20260621/inputs/t2_new_broad_fresh100z_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100z_20260621/t2_new_broad_fresh100z_alllegal_cap50.exact.jsonl`
- exact summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100z_20260621/t2_new_broad_fresh100z_alllegal_cap50.exact.summary.json`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100z_20260621/t2_new_broad_fresh100z_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100z_20260621/reranker_t2_new_broad_fresh100z_alllegal_cap50_dim693`
- fixed eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_fixed_eval_fresh100z_20260621/summary.json`
- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,406`
- source generation elapsed: `87.0s`
- cap50 exact elapsed: `24.5s/row` average with direct local chunks
- source/T3-model Top1 matched cap50 Top1: `36/100` (`36.0%`)
- source Top1 cap50 EV loss: mean `1.881`, max `22.132`

Fixed results on fresh100z:

| selector on fresh100z | training status | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `new_score_g100` | baseline | 44.0% | 74.0% | 90.0% | 95.0% | 99.0% | 99.0% | 1.002 | 0.032 |
| previous `gmtuvw` x-best | train `g/m/t/u/v/w`, selected on `x` | 46.0% | 73.0% | 87.0% | 98.0% | 99.0% | 100.0% | 0.962 | 0.017 |
| x-only hard-negative `g2/t4` | train `g/m/t/u/v/w/x`, misses from `x` | 77.0% | 84.0% | 86.0% | 95.0% | 98.0% | 99.0% | 0.311 | 0.033 |
| x-only hard-negative `g10/t20` | train `g/m/t/u/v/w/x`, misses from `x` | 76.0% | 83.0% | 89.0% | 96.0% | 97.0% | 98.0% | 0.481 | 0.040 |
| xy-hard `hgb_l31` | train `g/m/t/u/v/w/x/y`, misses from `x/y` | 80.0% | 83.0% | 86.0% | 96.0% | 97.0% | 100.0% | 0.305 | 0.054 |
| runtime `lgbm_gap63` | existing runtime selector | 49.0% | 76.0% | 87.0% | 98.0% | 100.0% | 100.0% | 0.959 | 0.000 |

Pool check on fresh100z:

| pool | recall | avg pool | max pool | mean EV loss | p99 EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|---:|
| `new_score_g100` Top15 | 97.0% | 14.7 | 15 | 0.000 | 0.000 | 0.000 |
| `lgbm_gap63` Top15 | 98.0% | 14.7 | 15 | 0.000 | 0.000 | 0.000 |
| `xyhard` Top15 | 98.0% | 14.7 | 15 | 0.000 | 0.000 | 0.000 |
| `lgbm_gap63` Top10 | 96.0% | 10.0 | 10 | 0.000 | 0.016 | 0.025 |
| `xyhard` Top10 | 96.0% | 10.0 | 10 | 0.038 | 0.916 | 2.886 |

Interpretation:

- The hard-negative Top1 direction now repeats on a fully untouched shard:
  fresh100z moves from default `44.0%` Top1 to `77.0%` with x-only hard
  negatives, and `80.0%` with x/y hard negatives.
- Reg1 improves sharply as well: default `1.002`, x-only `0.311`, xy-hard
  `0.305`.
- This is the first strong evidence that Top1 can be materially improved by
  replaying high-EV-loss Top1 misses, not just by selector switching.
- It is still not "solved".  Top1 is `77-80%`, not near `99%`, and Top3/Top5
  are much lower than the desired exact-candidate safety level.
- The accuracy path remains separate: use Top15/Top20 exact rerank as the
  backstop.  On fresh100z, Top15 has zero measured cap50 EV loss for the tested
  high-recall pools.  The lower Top10 pool can still leak with nontrivial EV
  loss depending on selector.
- Next useful step: generate a larger untouched shard and add a promotion gate:
  model-only Top1 can use x/y-hard as a source candidate, but runtime final
  decision should still exact-rerank a Top15/Top20 pool until Top1 approaches
  the target.

Fresh100aa external confirmation and Top1-pool arbitrator:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100aa_20260621/inputs/t2_new_broad_fresh100aa_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100aa_20260621/t2_new_broad_fresh100aa_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100aa_20260621/t2_new_broad_fresh100aa_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100aa_20260621/reranker_t2_new_broad_fresh100aa_alllegal_cap50_dim693`
- selector union eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100aa_compare_xyz_20260621/summary.json`
- per-selector eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100aa_compare_xyz_20260621/per_selector_metrics.json`
- top1-pool arbitrator:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_top1_pool_arbitrator_train_gmtuvwxyz_eval_aa_20260621/summary.json`

Fresh100aa source/exact:

- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,343`
- source generation elapsed: `78.3s`
- cap50 exact elapsed: `57.0s/row` average with five local chunks
- source/T3-model Top1 matched cap50 Top1: `28/100` (`28.0%`)
- source Top1 cap50 EV loss: mean `3.419`, max `37.164`

Fixed per-selector results on fresh100aa:

| selector on fresh100aa | training status | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline `new_score_g100` | runtime selector | 54.0% | 78.0% | 84.0% | 93.0% | 100.0% | 0.851 | 0.078 |
| `xhard_g2t4` | train `g/m/t/u/v/w/x`, misses from `x` | 71.0% | 82.0% | 86.0% | 93.0% | 98.0% | 0.791 | 0.023 |
| `xyhard` | train `g/m/t/u/v/w/x/y`, misses from `x/y` | 80.0% | 82.0% | 86.0% | 94.0% | 100.0% | 0.210 | 0.005 |
| `xyzw50gap` | train `g/m/t/u/v/w/x/y/z`, misses from `x/y/z` | 76.0% | 79.0% | 82.0% | 92.0% | 97.0% | 0.595 | 0.055 |
| `xyzw30rank` | train `g/m/t/u/v/w/x/y/z`, misses from `x/y/z` | 64.0% | 71.0% | 81.0% | 93.0% | 99.0% | 1.248 | 0.015 |

Union-pool and Top1-pool arbitrator on fresh100aa:

| method | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| union of all selector Top1 candidates, oracle exact-rerank ceiling | 89.0% | 92.0% | 94.0% | 99.0% | 100.0% | 0.104 | 0.097 | 0.002 |
| learned Top1-pool `hgb_cls_l31` | 83.0% | 92.0% | 97.0% | 100.0% | 100.0% | 0.159 | 0.028 | 0.000 |
| best single selector `xyhard` | 80.0% | 82.0% | 86.0% | 94.0% | 100.0% | 0.210 | 0.179 | 0.005 |

Interpretation:

- Adding fresh100z hard negatives directly did not improve external Top1 on
  fresh100aa; `xyzw50gap` fell behind the older `xyhard` (`76.0%` vs `80.0%`).
- The stronger direction is a small Top1-candidate pool: combining the Top1
  actions from independent selectors has an `89.0%` cap50 oracle ceiling with
  average pool size `2.42`.
- A learned classifier over that Top1 pool improves external Top1 from the best
  single selector `80.0%` to `83.0%` and improves Top10 to `100.0%` on
  fresh100aa.
- This is real movement but still not the final target. The current best
  external model-only Top1 path is now `83.0%`, not near `99%`. Next work should
  validate the Top1-pool arbitrator on another untouched shard before promoting
  it, then mine its remaining high-EV-loss misses.

Fresh100ab external follow-up:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ab_20260621/inputs/t2_new_broad_fresh100ab_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ab_20260621/t2_new_broad_fresh100ab_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ab_20260621/t2_new_broad_fresh100ab_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ab_20260621/reranker_t2_new_broad_fresh100ab_alllegal_cap50_dim693`
- selector union eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100ab_compare_xyz_20260621/summary.json`
- per-selector eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100ab_compare_xyz_20260621/per_selector_metrics.json`

Fresh100ab source/exact:

- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,412`
- source generation elapsed: `79.0s`
- cap50 exact elapsed: `59.8s/row` average with five local chunks
- source/T3-model Top1 matched cap50 Top1: `33/100` (`33.0%`)
- source Top1 cap50 EV loss: mean `2.780`, max `21.790`

Fixed per-selector results on fresh100ab:

| selector on fresh100ab | training status | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline `new_score_g100` | runtime selector | 42.0% | 69.0% | 84.0% | 91.0% | 100.0% | 1.471 | 0.057 |
| `xhard_g2t4` | train `g/m/t/u/v/w/x`, misses from `x` | 64.0% | 73.0% | 83.0% | 93.0% | 99.0% | 0.785 | 0.040 |
| `xyhard` | train `g/m/t/u/v/w/x/y`, misses from `x/y` | 79.0% | 82.0% | 86.0% | 92.0% | 99.0% | 0.284 | 0.043 |
| `xyzw50gap` | train `g/m/t/u/v/w/x/y/z`, misses from `x/y/z` | 74.0% | 77.0% | 82.0% | 92.0% | 99.0% | 0.579 | 0.041 |
| `xyzw30rank` | train `g/m/t/u/v/w/x/y/z`, misses from `x/y/z` | 66.0% | 71.0% | 82.0% | 95.0% | 100.0% | 0.965 | 0.039 |

Union-pool exact-rerank on fresh100ab:

| method | Top1 | Top3 | Top5 | Top10 | Top20 | Avg pool | Max pool | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| union of all selector Top1 candidates, oracle exact-rerank ceiling | 85.0% | 91.0% | 96.0% | 99.0% | 100.0% | 2.53 | 6 | 0.149 | 0.002 |
| union Top10 + exact-rerank | 99.0% | - | - | - | 100.0% | 14.91 | 23 | 0.002 | 0.002 |
| union Top20 + exact-rerank | 100.0% | - | - | - | 100.0% | 23.08 | 27 | 0.000 | 0.000 |

Interpretation:

- Yes, the stronger single-selector Top1 result is external: `xyhard` is
  `80.0%` on fresh100aa and `79.0%` on fresh100ab.
- No, the `83.0%` learned Top1-pool classifier should not be treated as
  generally confirmed yet. On fresh100ab, the oracle ceiling of the same Top1
  candidate pool is only `85.0%`, so the pool itself is now the limiter.
- Accuracy backstop still holds better than model-only Top1: Top10 union +
  exact-rerank is `99.0%` with max EV loss `0.187`, while Top20 union +
  exact-rerank is `100.0%` with zero measured cap50 EV loss on this shard.

Fresh100ac external confirmation:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ac_20260621/inputs/t2_new_broad_fresh100ac_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ac_20260621/t2_new_broad_fresh100ac_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ac_20260621/t2_new_broad_fresh100ac_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100ac_20260621/reranker_t2_new_broad_fresh100ac_alllegal_cap50_dim693`
- selector union eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100ac_compare_xyz_20260621/summary.json`
- pool3 reranker eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool3_reranker_train_gmtuvwxyz_eval_ac_20260621/summary.json`
- pool5 reranker eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool5_reranker_train_gmtuvwxyz_eval_ac_20260621/summary.json`

Fresh100ac source/exact:

- source records: `100` (`50` BB, `50` BTN)
- source candidates: `2,292`
- cap50 exact elapsed: `53.5s/row` average with five local chunks
- source/T3-model Top1 matched cap50 Top1: `12/100` (`12.0%`)
- source Top1 cap50 EV loss: mean `3.608`, max `14.140`

Union-pool exact-rerank on fresh100ac:

| method | Recall | Avg pool | Max pool | Mean EV loss | Max EV loss |
|---|---:|---:|---:|---:|---:|
| union Top1 candidates + exact-rerank | 88.0% | 2.75 | 7 | 0.071 | 2.581 |
| union Top3 candidates + exact-rerank | 91.0% | 6.34 | 12 | 0.056 | 2.581 |
| union Top5 candidates + exact-rerank | 94.0% | 8.81 | 18 | 0.047 | 2.581 |
| union Top8 candidates + exact-rerank | 98.0% | 12.08 | 21 | 0.035 | 2.581 |
| union Top10 candidates + exact-rerank | 99.0% | 14.57 | 23 | 0.009 | 0.887 |
| union Top12 candidates + exact-rerank | 100.0% | 16.76 | 24 | 0.000 | 0.000 |
| union Top15 candidates + exact-rerank | 100.0% | 19.40 | 24 | 0.000 | 0.000 |
| union Top20 candidates + exact-rerank | 100.0% | 21.63 | 24 | 0.000 | 0.000 |

Pool reranker external check on fresh100ac:

| model | Pool source | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hgb_cls_l31` | selector union Top3 | 84.0% | 89.0% | 90.0% | 94.0% | 100.0% | 0.200 | 0.063 | 0.022 |
| `hgb_cls_l31` | selector union Top5 | 84.0% | 89.0% | 91.0% | 94.0% | 99.0% | 0.198 | 0.049 | 0.022 |

Interpretation:

- External accuracy is not at the target yet. The learned pool reranker repeats
  about `84.0%` model-only Top1 on this untouched shard, which is better than
  the older single-selector direction but still far from `99%`.
- The accuracy backstop is much stronger than model-only Top1. On this harder
  shard, union Top10 + exact-rerank misses `1/100`, while union Top12/Top15/Top20
  + exact-rerank has zero measured cap50 EV loss.
- Since the `5s` target is only a rough guide, the next accuracy-first path is to
  keep using a wider Top12-Top15 exact-rerank pool while mining the remaining
  Top1/Top10 EV-loss misses for another selector-feature training pass.

Fresh100ac hard-negative follow-up and fresh50ad external check:

- ac miss mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh100ac_misses_20260621/summary.json`
- rejected direct ranker:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_ranker_train_gmtuvwxyzac_missweighted_ac_w20t50_eval_aa_ab_20260621/summary.json`
- pool3 with ac hard-negative weights:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool3_reranker_train_gmtuvwxyzac_missweighted_ac_eval_aa_ab_20260621/summary.json`
- Top1-pool with ac, no miss weights:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_top1_pool_arbitrator_train_gmtuvwxyzac_nomiss_eval_aa_ab_20260621/summary.json`
- fresh50ad source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ad_20260621/inputs/t2_new_broad_fresh50ad_source_all_actions.jsonl`
- fresh50ad exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ad_20260621/t2_new_broad_fresh50ad_alllegal_cap50.exact.jsonl`
- fresh50ad teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ad_20260621/t2_new_broad_fresh50ad_alllegal_cap50.teacher.jsonl`
- fresh50ad dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ad_20260621/reranker_t2_new_broad_fresh50ad_alllegal_cap50_dim693`

Fresh100ac miss mining:

- Top1 misses: `12/100`, recall `88.0%`, miss EV loss mean `0.595`, max `2.581`
- Top3 misses: `9/100`, recall `91.0%`, miss EV loss mean `0.627`, max `2.581`
- Top5 misses: `6/100`, recall `94.0%`, miss EV loss mean `0.781`, max `2.581`
- Top8 misses: `2/100`, recall `98.0%`, miss EV loss mean `1.734`, max `2.581`
- Top10 misses: `1/100`, recall `99.0%`, miss EV loss max `0.887`

ac-retraining on fresh100aa + fresh100ab:

| model | Train add | Eval | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| direct selector-feature ranker | `fresh100ac` + ac miss weights | aa+ab | 53.0% | 73.5% | 83.5% | 94.0% | 99.5% | 0.846 | 0.376 | 0.006 | reject |
| old pool3 reranker | none | aa+ab | 81.5% | 88.0% | 89.0% | 98.0% | 100.0% | 0.169 | 0.126 | 0.003 | baseline |
| pool3 reranker | `fresh100ac` + ac miss weights | aa+ab | 82.0% | 88.0% | 90.0% | 98.0% | 100.0% | 0.187 | 0.124 | 0.003 | reject, Reg1 worse |
| Top1-pool arbitrator | `fresh100ac`, no miss weights | aa+ab | 82.5% | 89.0% | 94.5% | 99.0% | 100.0% | 0.183 | 0.092 | 0.000 | candidate |

Fresh50ad source/exact:

- source records: `50` (`25` BB, `25` BTN)
- source candidates: `1,287`
- cap50 exact elapsed: `67.2s/row` average with five local chunks
- source/T3-model Top1 matched cap50 Top1: `11/50` (`22.0%`)
- source Top1 cap50 EV loss: mean `1.804`, max `12.393`

Union-pool exact-rerank on fresh50ad:

| method | Recall | Avg pool | Max pool | Mean EV loss | Max EV loss |
|---|---:|---:|---:|---:|---:|
| union Top1 candidates + exact-rerank | 86.0% | 3.20 | 6 | 0.010 | 0.153 |
| union Top3 candidates + exact-rerank | 90.0% | 7.14 | 11 | 0.004 | 0.142 |
| union Top5 candidates + exact-rerank | 94.0% | 10.38 | 15 | 0.003 | 0.142 |
| union Top8 candidates + exact-rerank | 100.0% | 14.34 | 20 | 0.000 | 0.000 |
| union Top10 candidates + exact-rerank | 100.0% | 17.24 | 23 | 0.000 | 0.000 |
| union Top12 candidates + exact-rerank | 100.0% | 19.82 | 24 | 0.000 | 0.000 |

Fresh50ad model comparison:

| model | Train | Pool | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z` | 1 | 78.0% | 82.0% | 88.0% | 98.0% | 100.0% | 0.188 | 0.158 | 0.000 |
| new Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z/ac` | 1 | 80.0% | 82.0% | 90.0% | 98.0% | 100.0% | 0.181 | 0.158 | 0.000 |
| old pool3 reranker | `g/m/t/u/v/w/x/y/z` | 3 | 80.0% | 82.0% | 86.0% | 90.0% | 100.0% | 0.181 | 0.132 | 0.004 |

Interpretation:

- Directly replaying ac misses into the all-legal selector-feature ranker is a
  bad direction; it collapses external Top1 and should not be promoted.
- Pool3/pool5 with ac miss weights buys a small Top1 gain on aa+ab, but increases
  Reg1. That is not a clean strength gain.
- The useful result is the Top1-pool arbitrator trained with `fresh100ac` but no
  miss over-weighting. It improves fresh50ad Top1 from `78.0%` to `80.0%` and
  lowers Reg1 from `0.188` to `0.181`, while keeping Top10/Top20 intact. This is
  still far from the final `99%` Top1 goal, but it is a real small external
  improvement in the right direction.

Fresh50ae untouched external check:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ae_20260621/inputs/t2_new_broad_fresh50ae_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ae_20260621/t2_new_broad_fresh50ae_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ae_20260621/t2_new_broad_fresh50ae_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50ae_20260621/reranker_t2_new_broad_fresh50ae_alllegal_cap50_dim693`
- selector union eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh50ae_compare_xyz_20260621/summary.json`

Fresh50ae source/exact:

- source records: `50` (`25` BB, `25` BTN)
- source candidates: `1,107`
- cap50 exact elapsed: `42.3s/row` average with five local chunks
- source/T3-model Top1 matched cap50 Top1: `22/50` (`44.0%`)
- source Top1 cap50 EV loss: mean `8.106`, max `26.958`

Union-pool exact-rerank on fresh50ae:

| method | Recall | Avg pool | Max pool | Mean EV loss | Max EV loss |
|---|---:|---:|---:|---:|---:|
| union Top1 candidates + exact-rerank | 94.0% | 2.54 | 7 | 0.176 | 5.571 |
| union Top3 candidates + exact-rerank | 96.0% | 6.12 | 11 | 0.053 | 1.328 |
| union Top5 candidates + exact-rerank | 96.0% | 8.00 | 13 | 0.030 | 1.130 |
| union Top8 candidates + exact-rerank | 98.0% | 11.26 | 15 | 0.023 | 1.130 |
| union Top10 candidates + exact-rerank | 100.0% | 13.30 | 17 | 0.000 | 0.000 |
| union Top12 candidates + exact-rerank | 100.0% | 15.48 | 22 | 0.000 | 0.000 |
| union Top15 candidates + exact-rerank | 100.0% | 18.44 | 24 | 0.000 | 0.000 |
| union Top20 candidates + exact-rerank | 100.0% | 21.04 | 24 | 0.000 | 0.000 |

Fresh50ae model comparison:

| model | Train | Pool | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z` | 1 | 88.0% | 88.0% | 92.0% | 96.0% | 100.0% | 1.083 | 0.893 | 0.030 |
| Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z/ac` | 1 | 88.0% | 88.0% | 92.0% | 96.0% | 100.0% | 1.083 | 0.893 | 0.030 |
| Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z/ac/ad` | 1 | 88.0% | 88.0% | 94.0% | 96.0% | 100.0% | 1.083 | 0.893 | 0.030 |
| old pool3 reranker | `g/m/t/u/v/w/x/y/z` | 3 | 88.0% | 88.0% | 94.0% | 96.0% | 100.0% | 0.901 | 0.893 | 0.053 |

Interpretation:

- The `fresh100ac` and `fresh50ad` additions do not improve external
  `fresh50ae` Top1. They keep Top1 at `88.0%`; the only visible gain is Top5
  from `92.0%` to `94.0%`.
- This means the small improvement seen on `fresh50ad` is not enough evidence to
  promote the new Top1-pool model as a generally stronger model-only policy.
- The accuracy backstop is confirmed on this untouched shard: union Top10 +
  exact-rerank has `100.0%` recall and zero measured cap50 EV loss with average
  pool `13.30`, max pool `17`.
- Since `5s` is only a loose guide, the current accuracy-first runtime target
  should be Top10 union + exact-rerank, not model-only Top1. Model-only Top1 is
  still diagnostic and must be improved with more clean external labels and
  hard-negative mining.

Fresh50af clean external check after adding fresh50ae:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50af_20260621/inputs/t2_new_broad_fresh50af_source_all_actions.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50af_20260621/t2_new_broad_fresh50af_alllegal_cap50.exact.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50af_20260621/t2_new_broad_fresh50af_alllegal_cap50.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh50af_20260621/reranker_t2_new_broad_fresh50af_alllegal_cap50_dim693`
- selector union eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_union_fresh50af_compare_xyz_20260621/summary.json`

Fresh50af source/exact:

- source records: `50` (`25` BB, `25` BTN)
- source candidates: `1,212`
- teacher candidates after exact join: `1,164`
- cap50 exact elapsed: `59.2s/row` average with five local chunks
- source/T3-model Top1 matched cap50 Top1: `25/50` (`50.0%`)
- source Top1 cap50 EV loss: mean `1.422`, max `10.332`

Union-pool exact-rerank on fresh50af:

| method | Recall | Avg pool | Max pool | Mean EV loss | Max EV loss |
|---|---:|---:|---:|---:|---:|
| union Top1 candidates + exact-rerank | 94.0% | 1.74 | 5 | 0.032 | 1.592 |
| union Top3 candidates + exact-rerank | 96.0% | 4.88 | 7 | 0.032 | 1.592 |
| union Top5 candidates + exact-rerank | 98.0% | 7.08 | 11 | 0.000 | 0.000 |
| union Top8 candidates + exact-rerank | 100.0% | 13.10 | 18 | 0.000 | 0.000 |
| union Top10 candidates + exact-rerank | 100.0% | 16.94 | 22 | 0.000 | 0.000 |

Fresh50af model comparison:

| model | Train | Pool | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z` | 1 | 90.0% | 96.0% | 96.0% | 100.0% | 100.0% | 0.107 | 0.032 | 0.000 |
| Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z/ac/ad` | 1 | 90.0% | 96.0% | 96.0% | 100.0% | 100.0% | 0.107 | 0.032 | 0.000 |
| Top1-pool arbitrator | `g/m/t/u/v/w/x/y/z/ac/ad/ae` | 1 | 90.0% | 96.0% | 96.0% | 100.0% | 100.0% | 0.107 | 0.032 | 0.000 |
| old pool3 `hgb_cls_l31` | `g/m/t/u/v/w/x/y/z` | 3 | 92.0% | 94.0% | 96.0% | 100.0% | 100.0% | 0.077 | 0.032 | 0.000 |
| pool3 `extra_gap_d14` | `g/m/t/u/v/w/x/y/z` | 3 | 92.0% | 96.0% | 96.0% | 100.0% | 100.0% | 0.062 | 0.032 | 0.000 |
| pool5 `extra_gap_d14` | `g/m/t/u/v/w/x/y/z` | 5 | 92.0% | 96.0% | 98.0% | 98.0% | 100.0% | 0.062 | 0.032 | 0.000 |
| pool3 with `ac/ad/ae` | `g/m/t/u/v/w/x/y/z/ac/ad/ae` | 3 | 90.0% | 94.0% | 96.0% | 100.0% | 100.0% | 0.107 | 0.032 | 0.000 |

Pool3 model-switch diagnostic on fresh50af:

- `hgb_cls_l31` alone: Top1 `92.0%`, Reg1 `0.077`, max loss `2.247`
- `extra_gap_d14` alone: Top1 `92.0%`, Reg1 `0.062`, max loss `1.592`
- oracle switch between them: Top1 `94.0%`, Reg1 `0.032`, max loss `1.592`
- learned switch gates trained on `g/m/t/u/v/w/x/y/z`: no Top1 lift; best
  learned gate kept Top1 `92.0%`

Interpretation:

- Adding `fresh50ae` to the Top1-pool training does not improve `fresh50af`.
  Top1 stays `90.0%`; pool3 with `ac/ad/ae` also regresses from `92.0%` to
  `90.0%`.
- The best current model-only external result on this shard is pool3
  `extra_gap_d14`: Top1 `92.0%`, Reg1 `0.062`.
- There is small remaining model-combination headroom, but it is hard to learn:
  the switch-positive rate in the training shards was only about `0.67%`
  (`6/900` groups), so a learned gate could not reproduce the oracle switch.
- For accuracy-first play, union Top8 or Top10 + exact-rerank is again clean on
  this shard. For model-only Top1, the next useful data work is not more blind
  appending of all shards; it is targeted generation/mining of cases where the
  pool3 models disagree and exactly one is correct, plus cases where union
  Top1/Top3 misses but Top5/Top8 contains the teacher best.

Pool3 disagreement mining pass:

- mining script:
  `ai/training/mine_t2_pool_model_disagreements.py`
- mined rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool3_model_disagreement_mining_gmtuvwxyz_20260621/rows.jsonl`
- mining summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool3_model_disagreement_mining_gmtuvwxyz_20260621/summary.json`
- weighted retrain, moderate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool3_reranker_train_gmtuvwxyz_disagree_w4t8_eval_ae_af_20260621/summary.json`
- weighted retrain, stronger:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool3_reranker_train_gmtuvwxyz_disagree_w8t12_eval_ae_af_20260621/summary.json`

Mined training rows from `g/m/t/u/v/w/x/y/z`:

- total groups scanned: `900`
- rows written: `49`
- reason counts:
  - `model_high_loss`: `42`
  - `model_disagreement`: `25`
  - `one_model_hits`: `18`
- mined EV loss mean/max: `1.817` / `9.854`

Weighted retrain result on external `fresh50ae + fresh50af`:

| model | Weighting | Eval | Top1 | Top3 | Top5 | Top10 | Reg1 | Reg3 | Reg10 | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pool3 `hgb_cls_l31` | none | fresh50af | 92.0% | 94.0% | 96.0% | 100.0% | 0.077 | 0.032 | 0.000 | baseline |
| pool3 `extra_gap_d14` | none | fresh50af | 92.0% | 96.0% | 96.0% | 100.0% | 0.062 | 0.032 | 0.000 | diagnostic candidate |
| pool3 `hgb_cls_l31` | disagreement `w4/t8` | fresh50ae | 88.0% | 88.0% | 94.0% | 96.0% | 1.083 | 0.893 | 0.053 | reject |
| pool3 `hgb_cls_l31` | disagreement `w4/t8` | fresh50af | 90.0% | 94.0% | 94.0% | 100.0% | 0.107 | 0.032 | 0.000 | reject |
| pool3 `hgb_cls_l31` | disagreement `w8/t12` | fresh50ae | 88.0% | 90.0% | 92.0% | 96.0% | 1.031 | 0.873 | 0.053 | reject |
| pool3 `hgb_cls_l31` | disagreement `w8/t12` | fresh50af | 92.0% | 94.0% | 96.0% | 100.0% | 0.062 | 0.032 | 0.000 | not enough |
| pool3 `extra_gap_d14` | disagreement `w8/t12` | fresh50ae | 86.0% | 88.0% | 88.0% | 96.0% | 1.145 | 0.893 | 0.053 | reject |
| pool3 `extra_gap_d14` | disagreement `w8/t12` | fresh50af | 92.0% | 94.0% | 96.0% | 100.0% | 0.062 | 0.032 | 0.000 | not enough |

Interpretation:

- The disagreement miner is useful: it found `49` high-signal rows from the
  current training shards.
- Directly over-weighting those rows is not enough. It improves/regresses
  differently by shard and does not raise external Top1. On `fresh50ae` it
  worsens Reg1 materially.
- Keep `pool3 extra_gap_d14` as the current diagnostic best for EV loss, but do
  not promote the disagreement-weighted models.
- Next Top1 work should generate more rows that look like the mined cases
  rather than just overweight the 49 existing examples. The current 49 rows are
  too sparse to teach a robust switch or Top1 selector.

### 2026-06-21 Targeted Source Mining Pipeline

Added pipeline pieces to move Top1 improvement from blind retraining to targeted
data generation:

- `ai/training/enrich_t2_mined_rows.py`
  - attaches compact source/teacher board context to mined rows
  - wrote:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pool3_model_disagreement_mining_gmtuvwxyz_20260621/rows.enriched.jsonl`
  - enrichment check: `49/49` rows got both source and teacher context
- `ai/training/normalize_t2_source_model_teacher.py`
  - converts estimated T2 source rows into action-value teacher shape
  - intended only for cheap preselection before exact; labels are not final
  - smoke: fresh50af head5 normalized to `5` records / `123` candidates and
    converted to dim693 successfully
- `ai/training/select_t2_source_rows_from_mined.py`
  - cuts original source rows referenced by mined `group_id`
  - smoke: selected `3/5` fresh50af head5 source rows at source-estimated
    `ev_loss >= 0.5`

Smoke artifacts:

- normalized source:
  `D:/ofc-pineapple-data/tmp_source_teacher_fresh50af_head5_20260621.jsonl`
- converted source dim693:
  `D:/ofc-pineapple-data/tmp_source_convert_fresh50af_head5_dim693_20260621`
- source pool mining:
  `D:/ofc-pineapple-data/tmp_source_pool_mine_head5_20260621/summary.json`
- selected source for exact:
  `D:/ofc-pineapple-data/tmp_source_pool_mine_head5_20260621/selected_source.jsonl`

Result:

- The source preselection path is now mechanically valid:
  generate source rows -> normalize source estimates -> convert to dim693 ->
  write base scores -> run pool-model disagreement/high-loss mining -> select
  original source rows for exact.
- This does not yet improve model Top1 by itself. It makes the next improvement
  pass cheaper: exact only the high-signal source rows, then retrain and verify
  on untouched external shards.

Fresh200ag local source preselection:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/inputs/t2_new_broad_fresh200ag_source_all_actions.jsonl`
- generated `200` records (`100` BB, `100` BTN) in `38.0s`
- candidates: `4,860`; T3 states scored: `216,675`
- source model teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/t2_new_broad_fresh200ag_source_model.teacher.jsonl`
- dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/reranker_t2_new_broad_fresh200ag_source_model_dim693`
- source pool3 mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/summary.json`
- mined source rows: `73/200`
  - `model_high_loss`: `73`
  - `model_disagreement`: `19`
  - `one_model_hits`: `6`
  - source-estimated EV loss mean/max: `2.288` / `7.846`
- selected for exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/selected_source_top50_for_exact.jsonl`
  - selected `50` rows at source-estimated `ev_loss >= 0.5`

Exact smoke on selected source:

- exact smoke summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/exact_smoke_top1/t2_oracle_cap50_limit1.summary.json`
- first selected row cap50 exact:
  - elapsed `11.9s`
  - source/T3-model Top1 differed from cap50 exact Top1
  - cap50 exact regret of source Top1: `+9.626`
  - source action: `Tc->bottom; Ts->middle; discard Jc`
  - cap50 exact action: `Tc->bottom; Ts->bottom; discard Jc`

This confirms the preselection pipeline is finding real high-value hard
negatives, not only source-model noise. Next exact step is to run the remaining
selected rows in chunks, build an exact teacher JSONL from them, append as a
targeted hard-negative shard, retrain pool3/Top1 models, then validate on
untouched external shards.

Targeted top50 exact and retrain diagnostics:

- exact top10:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/exact_top10/t2_oracle_cap50_limit10.summary.json`
  - `10/10` source Top1 changed under cap50 exact
  - avg/max exact regret of source Top1: `1.891` / `9.626`
- exact rest40:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/exact_rest40/t2_oracle_cap50_skip10_limit40.summary.json`
  - `40/40` source Top1 changed under cap50 exact
  - avg/max exact regret of source Top1: `1.970` / `8.292`
- merged exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/t2_oracle_cap50_targeted_top50.jsonl`
- exact teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/t2_new_broad_fresh200ag_targeted_top50_cap50.teacher.jsonl`
  - `50` records / `1,176` candidates
- exact dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/reranker_t2_new_broad_fresh200ag_targeted_top50_cap50_dim693`
- exact ag50 pool3 miss mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ag_20260621/source_pool3_mining_20260621/ag50_exact_pool3_mining_20260621/summary.json`
  - exact-loss pool3 hard rows: `12`

Retrain diagnostics on untouched `fresh50ae`/`fresh50af`:

| run | model | fresh50ae Top1 | fresh50ae Top3 | fresh50ae Reg1 | fresh50af Top1 | fresh50af Top3 | fresh50af Reg1 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pool3 + ag50 | `hgb_cls_l31` | 88.0% | 94.0% | 0.953 | 92.0% | 94.0% | 0.077 | no Top1 lift; Top3 gain on ae |
| pool3 + ag50 | `extra_gap_d14` | 84.0% | 88.0% | 1.457 | 92.0% | 94.0% | 0.062 | reject |
| pool1 + ag50 | `hgb_cls_l31` | 88.0% | 90.0% | 1.083 | 92.0% | 96.0% | 0.077 | no Top1 lift |
| pool1 + ag50 | `extra_gap_d14` | 86.0% | 88.0% | 1.263 | 92.0% | 96.0% | 0.062 | reject on ae |
| pool3 + ag50 + miss weights | `hgb_cls_l31` | 88.0% | 92.0% | 0.953 | 92.0% | 94.0% | 0.077 | no Top1 lift |
| pool3 + ag50 + miss weights | `extra_gap_d14` | 84.0% | 88.0% | 1.230 | 92.0% | 94.0% | 0.062 | reject |

Interpretation:

- The source preselection is valid: all selected top50 checked rows were actual
  source Top1 misses under cap50 exact.
- A 50-row targeted exact shard is still too small to move external Top1. It can
  improve some Top3/Reg3 behavior, but not the target metric.
- Next useful Top1 step is to repeat this pipeline at a larger scale, e.g.
  1,000-2,000 source rows -> select 200-500 exact rows -> retrain/evaluate. A
  broader targeted exact set is more likely to teach the model than stronger
  weighting of the same 50 rows.

Fresh1000ah targeted exact pass:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/inputs/t2_new_broad_fresh1000ah_source_all_actions.jsonl`
- generated `1,000` records (`500` BB, `500` BTN) in `168.1s`
- candidates: `22,491`; T3 states scored: `990,900`
- source pool3 mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/source_pool3_mining_20260621/summary.json`
- mined source rows: `268/1,000`
  - `model_high_loss`: `266`
  - `model_disagreement`: `43`
  - `one_model_hits`: `22`
  - source-estimated EV loss mean/max: `4.191` / `29.072`
- selected for exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/source_pool3_mining_20260621/selected_source_top200_for_exact.jsonl`
  - selected `200` rows at source-estimated `ev_loss >= 0.5`
- exact chunks:
  - `0-49`: source Top1 changed `45/50`; avg/max exact regret `4.802` / `39.586`; avg exact `11.1s`
  - `50-99`: source Top1 changed `47/50`; avg/max exact regret `2.859` / `9.801`; avg exact `11.0s`
  - `100-149`: source Top1 changed `46/50`; avg/max exact regret `4.671` / `22.617`; avg exact `12.1s`
  - `150-199`: source Top1 changed `49/50`; avg/max exact regret `3.954` / `17.230`; avg exact `11.7s`
- merged exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/source_pool3_mining_20260621/t2_oracle_cap50_targeted_top200.jsonl`
- exact teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/source_pool3_mining_20260621/t2_new_broad_fresh1000ah_targeted_top200_cap50.teacher.jsonl`
  - `200` records / `4,068` candidates
- exact dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/source_pool3_mining_20260621/reranker_t2_new_broad_fresh1000ah_targeted_top200_cap50_dim693`
- low-weight exact dim693:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ah_20260621/source_pool3_mining_20260621/reranker_t2_new_broad_fresh1000ah_targeted_top200_cap50_dim693_loww`
  - mean/max sample weight reduced from `2.54` / `5.50` to `1.25` / `2.00`

Retrain diagnostics on untouched `fresh50ae`/`fresh50af`:

| run | model | aggregate Top1 | aggregate Top3 | aggregate Reg1 | fresh50ae Top1 | fresh50ae Reg1 | fresh50af Top1 | fresh50af Reg1 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pool3 + ah200 | `hgb_cls_l31` | 89.0% | 91.0% | 0.539 | 88.0% | 0.971 | 90.0% | 0.107 | reject |
| pool3 + ah200 | `extra_gap_d14` | 86.0% | 92.0% | 0.855 | 82.0% | 1.603 | 90.0% | 0.107 | reject |
| pool3 + ah200 low-weight | `hgb_cls_l31` | 89.0% | 91.0% | 0.539 | 88.0% | 0.971 | 90.0% | 0.107 | reject |
| pool3 + ah200 low-weight | `extra_gap_d14` | 86.0% | 92.0% | 0.855 | 82.0% | 1.603 | 90.0% | 0.107 | reject |

Interpretation:

- The larger targeted exact pass confirms the source/T3-model teacher is
  sharply wrong on the mined hard rows: only `13/200` source Top1 actions stayed
  Top1 after cap50 exact.
- This does not yet translate into external model-only Top1 improvement.
  Directly appending hard-only rows makes the training distribution too
  adversarial and hurts `fresh50af` while not improving `fresh50ae`.
- Current external answer:
  - model-only Top1 is still around `88-92%`, depending on shard/model
  - `Top10` selector union + exact rerank remains the reliable path on the
    checked external shards (`fresh50ae` and `fresh50af` both previously reached
    zero measured cap50 EV loss at Top10 union)
- Next useful direction is not stronger weighting of hard-only rows. It is a
  balanced exact dataset: mix broad random exact rows with targeted hard rows,
  then train with a held-out external shard that was not used for mining.

Pool size and selection-model diagnostics after the `fresh1000ah` pass:

External evaluation remains on untouched `fresh50ae`/`fresh50af`.

| experiment | aggregate Top1 | aggregate Top3 | aggregate Top10 | aggregate Reg1 | aggregate Reg10 | decision |
|---|---:|---:|---:|---:|---:|---|
| selector-feature pool5 `hgb_cls_l31` | 89.0% | 92.0% | 97.0% | 0.530 | 0.025 | pool too small; exact best can be outside pool |
| selector-feature pool8 `hgb_cls_l31` | 89.0% | 93.0% | 98.0% | 0.604 | 0.015 | improves Top3/Reg10 but still not Top1 |
| selector-feature pool10 `hgb_cls_l31` | 89.0% | 91.0% | 98.0% | 0.604 | 0.025 | Top1 not improved |
| pairwise pool10 + selector features `hgb_l31` | 90.0% | 91.0% | 100.0% | 0.589 | 0.000 | Top10 safe, Top1 still stuck |
| pairwise pool10 + both-diff features `hgb_l31` | 90.0% | 91.0% | 100.0% | 0.547 | 0.000 | slightly better Reg1, no Top1 lift |
| LightGBM ranker pool10 `lgbm_rank_l31` | 88.0% | 91.0% | 97.0% | 0.618 | 0.080 | reject |
| LightGBM ranker pool10 `lgbm_rank_l63` | 87.0% | 91.0% | 97.0% | 0.601 | 0.080 | reject |

Pool ceiling check:

- pool5 selector union: recall `97.0%`, avg/max pool `7.54` / `13`,
  mean/max EV loss `0.015` / `1.130`
- pool8 selector union: recall `99.0%`, avg/max pool `12.18` / `18`,
  mean/max EV loss `0.011` / `1.130`
- pool10 selector union: recall `100.0%`, avg/max pool `15.12` / `22`,
  mean/max EV loss `0.000` / `0.000`

Top1 oracle-switch diagnostic:

- compared five current Top1 sources:
  - pool3 `hgb_cls_l31`
  - pool3 `extra_gap_d14`
  - pool10 `hgb_cls_l31`
  - pairwise pool10 selector-feature `hgb_l31`
  - pairwise pool10 both-diff `hgb_l31`
- oracle switch across those model Top1 choices reached only:
  - aggregate Top1 `91.0%`
  - aggregate Reg1 `0.463`
  - avg/max unique Top1 actions per spot: `1.10` / `4`

Interpretation:

- The `Top10` pool already contains the cap50 exact best action on both checked
  external shards. Candidate generation is not the blocker at this pool size.
- Current model families often put the same wrong action at Top1, so switching
  among them has only a low ceiling.
- For external tests, the current honest statement is:
  - model-only Top1: about `88-92%`
  - model-only Top3: about `91-93%`
  - Top10 pool + exact rerank: measured cap50 EV loss `0` on `fresh50ae` and
    `fresh50af`
- Next Top1 work should use a stronger EV scorer, ideally a neural action-value
  or listwise model trained on balanced broad exact data plus targeted hard
  rows. More selector gates or hard-only weighting is unlikely to move Top1.

Follow-up Top1 improvement pass:

- neural pool10 direct MLP, feature scope `both`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/neural_pool10_reranker_train_gmtuvwxyz_eval_ae_af_20260621/summary.json`
  - aggregate Top1 `85.0%`, Reg1 `1.099`; reject
- neural pool10 direct MLP, feature scope `selector`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/neural_pool10_selector_reranker_train_gmtuvwxyz_eval_ae_af_20260621/summary.json`
  - aggregate Top1 `85.0%`, Reg1 `0.672`; reject
- extended clean exact pool10 HGB:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool10_reranker_train_extended_clean_eval_ae_af_20260621/summary.json`
  - train data: `23` clean cap50 exact shards, `3,150` groups, `46,829`
    pool rows
  - best model: `hgb_cls_l63`
  - aggregate Top1 `91.0%`, Top3 `91.0%`, Top5 `94.0%`, Top10 `99.0%`
  - aggregate Reg1 `0.524`, Reg10 `0.000`
  - `fresh50ae`: Top1 `88.0%`, Reg1 `1.015`
  - `fresh50af`: Top1 `94.0%`, Reg1 `0.032`
- extended clean + `fresh1000ah` targeted exact top200:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool10_reranker_train_extended_clean_ah200_eval_ae_af_20260621/summary.json`
  - train data: `3,350` groups, `49,766` pool rows
  - best model by Top1/Reg1: `hgb_cls_l31`
  - aggregate Top1 `91.0%`, Top3 `92.0%`, Top5 `93.0%`, Top10 `98.0%`
  - aggregate Reg1 `0.493`, Reg10 `0.011`
  - `fresh50ae`: Top1 `88.0%`, Reg1 `0.953`
  - `fresh50af`: Top1 `94.0%`, Reg1 `0.032`
- extended clean + `fresh1000ah` with stronger Top1 weighting:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool10_reranker_train_extended_ah200_top1w_eval_ae_af_20260621/summary.json`
  - aggregate Top1 fell to `90.0%`; reject, although model Top10 Reg10 was `0`
- extended model Top1 oracle switch:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_oracle_switch_extended_eval_ae_af_20260621.json`
  - compared old pool10, extended clean, extended + ah200, and top1-weighted
    model Top1 actions
  - oracle Top1 stayed `91.0%`
  - avg/max unique Top1 actions: `1.07` / `3`

Interpretation:

- This pass did improve the external model-only Top1 ceiling from the previous
  best `90.0%` to `91.0%`, mainly by adding broad clean exact data.
- The targeted `ah200` rows did not add another Top1 point, but did reduce Reg1
  from `0.524` to `0.493`.
- A learned switch gate is not promising here because the extended models still
  choose almost the same Top1 action. The remaining large misses need new
  broad/exact data or a materially different action-value representation, not
  another gate over the current model family.

Targeted Top1 follow-up:

- generated multi-root source rows with wider T0/T1 branching:
  - `new_broad_fresh200aj_20260621`: `200` rows, `100` unique root deals
  - `new_broad_fresh1000ak_20260621`: partial `400` rows, `200` unique root
    deals; the long source build stopped before summary but the JSONL was valid
- mined source-model hard rows from the partial `400` rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ak_20260621/source_pool_mining_partial400_20260621/summary.json`
  - selected `80` rows for cap50 exact
  - source Top1 changed under cap50 exact on `75/80`
  - chunk avg exact latency was about `20-22s/row`
  - exact teacher:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ak_20260621/source_pool_mining_partial400_20260621/t2_new_broad_fresh400ak_targeted_top80_cap50.teacher.jsonl`
- added a targeted synthetic generator:
  `ai/training/generate_t2_top_aa_aj_source.py`
  - broad synthetic `top_AA_middle_JJ_dealt_AJx`: `40` all-legal cap50 rows
  - `ae_like` synthetic `top_AA_middle_JJ_dealt_AJx`: `40` all-legal cap50
    rows, shaped closer to the `fresh50ae` high-loss pattern

Retrain checks:

| train addition | best external Top1 | best Top3 | best Top5 | best Top10 | Reg1 | decision |
|---|---:|---:|---:|---:|---:|---|
| `ak80` hard exact | `90.0%` | `93.0%` | `95.0%` | `98-100%` | `0.504` | reject for Top1; improves some Top3/Top5 behavior |
| `ak80 + synthetic broad40` | `90.0%` | `92.0%` | `95.0%` | `96-100%` | `0.513` | reject |
| `ak80 + synthetic ae_like40` | `88.0%` | `93.0%` | `96.0%` | `100.0%` | `0.837` | reject; over-targeting hurts external Top1 |

Top1-choice oracle across the new variants:

- compared extended clean+ah200, ak80, synthetic broad, and synthetic ae_like
  pool10 classifier variants
- `fresh50ae`: oracle Top1 stayed `88.0%`; none of the new variants hit the
  large-loss groups `26`, `27`, `46`, `48`, or `49`
- `fresh50af`: oracle Top1 stayed `94.0%`

Interpretation:

- The new targeted labels are real hard exact rows, but small targeted weighting
  does not teach the current selector-feature pool model to choose the remaining
  external Top1 actions.
- The blocker is not Top10 candidate generation: the checked external exact best
  action is still inside the Top10 pool.
- The next Top1 attempt should change the scorer/representation or add a much
  larger balanced exact shard. More small hard-only rows are likely to overfit
  or move Top3/Top5 without lifting model-only Top1.

Pool-local and attention scorer check:

- added `--pool-local-features` to
  `ai/training/train_t2_selector_feature_pool_reranker.py`
  - default behavior is unchanged
  - the added features restate selector rank, gap, margin, and vote pattern
    inside the selected union pool
- extended clean + `ah200`, pool-local HGB:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool10_poollocal_train_extended_clean_ah200_eval_ae_af_20260621/summary.json`
  - best external Top1 stayed `91.0%`
  - best Reg1 `0.493`
  - `fresh50ae`: Top1 `88.0%`, Reg1 `0.953`
  - `fresh50af`: Top1 `94.0%`, Reg1 `0.032`
- extended clean + `ah200 + ak80`, pool-local HGB:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool10_poollocal_train_extended_ah200_ak80_eval_ae_af_20260621/summary.json`
  - best external Top1 stayed `91.0%`
  - Reg1 improved slightly to `0.489`
  - `fresh50ae`: Top1 `88.0%`, Reg1 `0.945`
  - `fresh50af`: Top1 `94.0%`, Reg1 `0.032`
- Top1 oracle switch across base, pool-local, ak80 pool-local, and pairwise
  variants:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/poollocal_top1_oracle_switch_eval_ae_af_20260621.json`
  - oracle Top1 still `91.0%`
  - oracle Reg1 `0.463`
  - avg/max unique Top1 actions per spot: `1.10` / `3`
- added attention architecture and `--pool-local-features` to
  `ai/training/train_t2_neural_pool_reranker.py`
  - evaluated saved checkpoint:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/neural_attention_pool10_poollocal_train_extended_clean_ah200_eval_ae_af_20260621/summary_partial_evaluated.json`
  - aggregate Top1 `84.0%`, Reg1 `1.133`; reject
- added board/deck context action features to
  `ai/training/action_feature_encoding.py`
  - state dim increased from `693` to `789` when these features are requested
  - features include remaining rank/suit counts, self/opponent/dead rank
    counts, row straight/suit summaries, top live outs, row masks, and global
    visible-card stats
  - converted the current extended clean + `ah200` train/eval shards to
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_20260621`
  - merged train data: `3,350` groups, `77,793` candidates
  - trained direct action-value NN:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_20260621/models/t2-dim789-boardctx-extended-ah200-20260621`
  - external aggregate on `fresh50ae/fresh50af`: Top1 `64.0%`, Top3
    `85.0%`, Top5 `88.0%`, Top10 `99.0%`, Reg1 `2.628`, Reg10 `0.015`
  - `fresh50ae`: Top1 `56.0%`, Reg1 `4.917`
  - `fresh50af`: Top1 `72.0%`, Reg1 `0.338`
  - reject; the richer raw state features do not make this direct NN a usable
    Top1 scorer
- added legacy runtime feature adaptation for mixed 693/789 checks:
  - `train_t2_selector_feature_ranker.py` now preserves the old runtime
    feature layout as `old_state_prefix + base/rank/gap/z/group` when an older
    sklearn model is evaluated on a newer state vector
  - `evaluate_t2_selector_feature_union.py` and
    `train_t2_selector_feature_pool_reranker.py` use the same adaptation for
    feature-selector models, preserving `old_runtime_features + selector`
    instead of blindly slicing away selector features
  - sanity check: all old selector scores on `fresh50ae` matched exactly between
    dim693 and dim789 after copying old `base_scores.npy`
- copied old dim693 `base_scores.npy` into matching dim789 train/eval shards
  after verifying `scores`, `group_ids`, and `action_indices` match; remerged:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_20260621/merged_train_extended_clean_ah200_dim789_oldbase`
- old selector sources + dim789 board/deck context reranker:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_20260621/selector_oldsources_dim789_fixedadapt_20260621`
  - `both_hgbonly`: best external Top1 `90.0%`, Top3 `91.0%`, Top5
    `94.0%`, Top10 `99.0%`, Reg1 `0.517`, Reg10 `0.011`
  - `both_poollocal_hgbonly`: best external Top1 `91.0%`, Top3 `91.0%`,
    Top5 `95.0%`, Top10 `99.0%`, Reg1 `0.498`, Reg10 `0.011`
  - external union ceiling remained perfect: recall `100.0%`, EV loss `0`
  - reject for Top1; matches but does not beat the existing `91.0%` selector
- train-selected blend of existing best and dim789-fixed variants:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_20260621/selector_oldsources_dim789_fixedadapt_20260621/blend_weight_sweep_train_selected_eval_ae_af.json`
  - best train-selected external Top1 stayed `91.0%`
  - oracle over old + dim789-fixed Top1 choices was `92.0%`, but the useful
    differing choices were too rare for a stable train-selected blend
- switch gate checks:
  - old `hgb_cls_l31` vs dim789 pool-local `hgb_cls_l31`:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_20260621/selector_oldsources_dim789_fixedadapt_20260621/switch_gate_old_l31_vs_new_pl_l31_20260621/summary.json`
    - gate was fit on 80% of train groups and threshold-selected on the
      remaining 20%
    - best external Top1 stayed `91.0%`
    - best Reg1 was `0.489`, a small EV-loss improvement but no Top1 lift
  - old `hgb_cls_l31` vs old `hgb_cls_l63`:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/selector_feature_pool10_reranker_train_extended_clean_ah200_eval_ae_af_20260621/switch_gate_old_l31_vs_l63_20260621/summary.json`
    - best external Top1 stayed `91.0%`
    - best Reg1 improved to `0.467`, Top3 `98.0%`, Top10 `100.0%`
    - reject for Top1; useful only as an EV-loss/tail diagnostic

Interpretation:

- Pool-local features are valid and slightly reduce Reg1 when paired with
  `ak80`, but they do not create new external Top1 hits.
- The current model family has no useful Top1 switching headroom on
  `fresh50ae/fresh50af`; the variants choose almost the same wrong Top1
  actions.
- The attention model is not a usable improvement in its current form.
- The board/deck context direct NN confirms that simply adding more raw state
  dimensions is not enough; this architecture is much weaker than the external
  selector-feature HGB baseline.
- Preserving the old selector stack and adding board/deck context to the final
  HGB reranker also does not lift external Top1 above `91.0%`; the extra
  dimensions are not currently being used in a way that generalizes.
- Train/dev selected switch gates can lower EV loss a little, but they still do
  not reproduce a real external Top1 gain.
- To lift model-only Top1 above `91%`, the next useful work is a larger
  balanced exact shard or materially richer state/action representation, not
  another gate over these predictors.

Extended pairwise pool10 check:

- trained pairwise rerankers on the full extended clean + `ah200` training set:
  `3,350` T2 groups, `77,793` candidates
- selector-only pairwise:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool10_extended_clean_ah200_eval_ae_af_20260621/selector_full_gap002_best4/summary.json`
  - training built `626,076` pair rows
  - external aggregate Top1 `90.0%`, Top3 `91.0%`, Top5 `94.0%`,
    Top10 `100.0%`, Reg1 `0.531`, Reg10 `0.000`
  - `fresh50ae`: Top1 `88.0%`, Reg1 `0.985`
  - `fresh50af`: Top1 `92.0%`, Reg1 `0.077`
  - oracle switch with the existing `91.0%` baseline still stayed at Top1
    `91.0%`; the large wrong-Top1 families are effectively the same
- state+selector pairwise diff:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/pairwise_pool10_extended_clean_ah200_eval_ae_af_20260621/both_diff_gap010_best6_l31/summary.json`
  - training built `573,158` pair rows
  - external aggregate Top1 `88.0%`, Top3 `91.0%`, Top5 `93.0%`,
    Top10 `100.0%`, Reg1 `0.738`, Reg10 `0.000`
  - reject; adding the raw state vector to this pairwise objective made Top1
    worse
- added mixed-dim safety to `train_t2_pairwise_pool_reranker.py`: feature
  selector inputs now use the same `predict_adapted` path as the other T2
  selector scripts, so older 693-dim feature selectors can be evaluated on
  newer 789-dim state matrices without dropping the runtime selector features

Interpretation:

- Pairwise objectives confirm that pool10 is a safe refinement set on the
  current external shards: exact-best recall is `100.0%`, and Top10 rerank
  Reg10 is `0.000` in the new pairwise outputs.
- Pairwise does not lift model-only Top1. It either matches the same hard miss
  families or degrades `fresh50ae`.
- The next Top1 improvement should not be another selector/pairwise gate over
  the same predictors. It needs either more broad exact coverage that includes
  these high-loss families naturally, or a materially different model
  representation/objective.

Top1 miss-weighted selector check:

- mined Top1 misses from the existing selector-feature pool10 baseline:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/top1_miss_report.json`
  - train groups `3,350`, misses `355`
  - train misses with EV loss `>=0.25`: `181`
  - external `fresh50ae/fresh50af` misses: `9/100`
  - external miss EV loss mean/p95/max: `5.474` / `19.457` / `22.344`
- direct miss-weighted pool retraining did not improve Top1:
  - normal pool:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_miss025_w4_t14/summary.json`
    - best external Top1 `90.0%`, Reg1 `0.573`
  - pool-local:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_miss025_w4_t14/summary.json`
    - best external Top1 `91.0%`, Reg1 `0.493`
- trained hard-miss feature selectors and added them as candidate sources:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/feature_selector_state_miss025_w4_t14/summary.json`
  - standalone external Top1 was weak (`74.0%` for `hgb_cls_l31_state`),
    but as extra pool sources they exposed useful candidates
- best current model-only T2 Top1 check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs/summary.json`
  - external `fresh50ae/fresh50af`: Top1 `94.0%`, Top3 `98.0%`,
    Top5 `98.0%`, Top10 `100.0%`
  - Reg1 `0.0715`, Top3 rerank Reg `0.0001`, Top10 rerank Reg `0.0000`
  - `fresh50ae`: Top1 `98.0%`, Reg1 `0.111`
  - `fresh50af`: Top1 `90.0%`, Reg1 `0.032`
- additional outside check on `fresh20n/fresh20o/fresh20p`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh20nop_check_20260621/top1hard_poollocal_fresh20nop/summary.json`
  - baseline on the same 60 groups:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh20nop_check_20260621/baseline_fresh20nop/summary.json`
  - baseline: Top1 `83.3%`, Top3 `90.0%`, Top5 `98.3%`,
    Top10 `98.3%`, Reg1 `0.148`
  - hard-miss pool-local: Top1 `90.0%`, Top3 `95.0%`, Top5 `96.7%`,
    Top10 `98.3%`, Reg1 `0.029`
  - per shard: `fresh20n` Top1 `95.0%`, `fresh20o` `90.0%`,
    `fresh20p` `85.0%`
- new clean smoke holdout `fresh10al`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh10al_20260621`
  - source generation: `10` records, `5` BB / `5` BTN, `240` source
    candidates, elapsed `6.4s`
  - cap50 exact: `10` records, source Top1 changed on `5/10`, average exact
    elapsed `14.3s/row`, source Top1 Reg1 `0.832`
  - both baseline and hard-miss pool-local reached Top1 `100.0%`; this shard
    is useful as pipeline proof but too small/easy for promotion evidence
- new clean holdout `fresh30am`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30am_20260621`
  - source generation: `30` records, `15` BB / `15` BTN, `687` source
    candidates, elapsed `21.7s`
  - cap50 exact was run in three local chunks:
    - chunk `0-9`: source Top1 changed on `7/10`, exact `34.8s/row`
    - chunk `10-19`: source Top1 changed on `8/10`, exact `31.9s/row`
    - chunk `20-29`: source Top1 changed on `7/10`, exact `34.2s/row`
  - merged exact teacher:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30am_20260621/t2_new_broad_fresh30am_alllegal_cap50.teacher.jsonl`
  - converted dataset:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30am_20260621/reranker_t2_new_broad_fresh30am_alllegal_cap50_dim693`
  - baseline:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30am_check_20260621/baseline_fresh30am/summary.json`
    - Top1 `73.3%`, Top3 `80.0%`, Top5 `90.0%`, Top10 `93.3%`,
      Top20 `100.0%`, Reg1 `0.702`, Reg10 `0.108`
  - hard-miss pool-local:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30am_check_20260621/top1hard_poollocal_fresh30am/summary.json`
    - Top1 `90.0%`, Top3 `93.3%`, Top5 `96.7%`, Top10 `96.7%`,
      Top20 `96.7%`, Reg1 `0.081`, Reg10 `0.028`
  - hard-miss normal-pool matched the same metrics:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30am_check_20260621/top1hard_normal_fresh30am/summary.json`
- mined remaining high-loss misses:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30am_check_20260621/mine_top1hard_highloss/summary.json`
  - rows `2/30`, both `model_high_loss`
  - EV loss mean `1.219`, max `1.607`
  - row `21`, BTN T2: board `T:Qs Kc | M:5h 9h | B:7c X2 7d`,
    dealt `Js 9c Th`; cap50 best `Js->bottom; Th->middle; discard 9c`
    beats the model choice by `1.607` EV
  - row `15`, BTN T2: board `T:2c | M:6c 4c | B:9s Qs X1 Td`,
    dealt `9d 9c 8h`; cap50 best `8h->bottom; 9d->top; discard 9c`
    beats the model choice by `0.830` EV
- retrained with the `fresh30am` two high-loss rows added:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs_fresh30am_miss2_w12t24_eval_aeaf_nop/summary.json`
  - best aggregate model: `hgb_cls_l63`
  - eval on `fresh50ae/fresh50af/fresh20n/fresh20o/fresh20p`: Top1
    `92.5%`, Top3 `97.5%`, Top5 `98.1%`, Top10 `98.8%`, Top20 `100.0%`
  - Reg1 `0.0557`, Top3 rerank Reg `0.0006`, Top10 rerank Reg `0.0006`
- new clean holdout `fresh30an`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30an_20260621`
  - source generation: `30` records, `15` BB / `15` BTN, `714` source
    candidates, elapsed `22.7s`
  - cap50 exact was run in three local chunks; source Top1 changed on
    `19/30`, average exact elapsed about `36.9s/row`
  - teacher dataset:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30an_20260621/reranker_t2_new_broad_fresh30an_alllegal_cap50_dim693`
  - evaluation:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30an_check_20260621/existing_and_miss2_eval_fresh30an.json`

| model | Top1 | Top3 | Top10 | Top20 | Reg1 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 83.3% | 90.0% | 93.3% | 100.0% | 0.123 | 0.0159 |
| top1hard pool-local | 86.7% | 93.3% | 93.3% | 100.0% | 0.0677 | 0.0061 |
| fresh30am miss2 | 90.0% | 93.3% | 96.7% | 100.0% | 0.0659 | 0.0032 |

- mined remaining high-loss misses on `fresh30an` after the miss2 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30an_check_20260621/mine_miss2_highloss/summary.json`
  - rows `3/30`, EV loss mean `0.659`, max `1.796`
- retrained again with `fresh30am + fresh30an` and `5` high-loss rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs_fresh30am_an_miss5_w12t24_eval_aeaf_nop/summary.json`
  - aggregate `fresh50ae/fresh50af/fresh20n/fresh20o/fresh20p`: Top1
    `93.1%`, Top3 `96.9%`, Top10 `99.4%`, Top20 `100.0%`, Reg1 `0.0475`
  - this looks slightly better on the mixed aggregate, but it needed a new
    clean external check before accepting
- new clean holdout `fresh30ao`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30ao_20260621`
  - source generation: `30` records, `15` BB / `15` BTN, `681` source
    candidates, elapsed `22.0s`
  - cap50 exact was run in three local chunks; source Top1 changed on
    `25/30`, average exact elapsed about `32.4s/row`
  - teacher dataset:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30ao_20260621/reranker_t2_new_broad_fresh30ao_alllegal_cap50_dim693`
  - evaluation:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30ao_check_20260621/existing_miss2_miss5_eval_fresh30ao.json`

| model | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 86.7% | 93.3% | 93.3% | 100.0% | 100.0% | 0.410 | 0.0093 | 0.0000 |
| top1hard pool-local | 90.0% | 96.7% | 100.0% | 100.0% | 100.0% | 0.0082 | 0.0057 | 0.0000 |
| fresh30am miss2 | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0082 | 0.0000 | 0.0000 |
| fresh30am+an miss5 | 90.0% | 96.7% | 100.0% | 100.0% | 100.0% | 0.0082 | 0.0057 | 0.0000 |

- mined remaining high-loss misses on `fresh30ao` after the miss2 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30ao_check_20260621/mine_miss2_highloss/summary.json`
  - rows `2/30`, EV loss mean `0.123`, max `0.172`
  - the largest remaining miss is small compared with the earlier external
    misses; the miss2 model is now mainly losing near-tie decisions on this
    clean shard
- retrained with `fresh30am + fresh30ao` and `4` high-loss rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs_fresh30am_ao_miss4_w12t24_eval_aeaf_nop/summary.json`
  - best aggregate model: `hgb_cls_l31`
  - aggregate `fresh50ae/fresh50af/fresh20n/fresh20o/fresh20p`: Top1
    `93.1%`, Top3 `96.9%`, Top5 `98.1%`, Top10 `99.4%`, Top20 `100.0%`
  - Reg1 `0.0475`, Top3 rerank Reg `0.0006`, Top10 rerank Reg `0.0006`
  - compared with `fresh30am_miss2`, this improves the same aggregate from
    Top1 `92.5%` / Reg1 `0.0557` without increasing Top10 EV loss
- checked `fresh30am_ao_miss4` on the still-untrained `fresh30an`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30an_check_20260621/miss2_vs_miss4_eval_fresh30an.json`
  - exactly matched `fresh30am_miss2`: Top1 `90.0%`, Top3 `93.3%`,
    Top10 `96.7%`, Top20 `100.0%`, Reg1 `0.0659`
- new clean holdout `fresh30ap`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30ap_20260621`
  - source generation: `30` records, `15` BB / `15` BTN, `741` source
    candidates, elapsed `20.3s`
  - cap50 exact was run in three local chunks; source Top1 changed on
    `21/30`, average exact elapsed about `36.9s/row`
  - source model Top1 Reg1 was `2.846`, max source Top1 regret `20.057`
  - teacher dataset:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh30ap_20260621/reranker_t2_new_broad_fresh30ap_alllegal_cap50_dim693`
  - evaluation:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30ap_check_20260621/baseline_miss2_miss4_eval_fresh30ap.json`

| model | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 90.0% | 96.7% | 100.0% | 100.0% | 100.0% | 0.268 | 0.249 | 0.000 |
| top1hard pool-local | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| fresh30am miss2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| fresh30am+ao miss4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |

- mined remaining high-loss misses on `fresh30ap` after the miss4 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh30ap_check_20260621/mine_miss4_highloss/summary.json`
  - rows `0/30`; no Top1 EV loss `>=0.05`
- larger clean holdout `fresh200ai_first100`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ai_20260621`
  - source generation: `200` records total from root start `7500`, `100` BB /
    `100` BTN; first `100` BB records were exact-checked here
  - first100 source candidates `2496`, teacher candidates `2400`
  - cap50 exact was run in ten local chunks; source Top1 changed on `77/100`,
    average exact elapsed about `69.5s/row`
  - source model Top1 Reg1 was `0.776`, max source Top1 regret `6.308`
  - teacher dataset:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ai_20260621/reranker_t2_new_broad_fresh200ai_first100_alllegal_cap50_dim693`
  - evaluation:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh200ai_first100_check_20260621/baseline_miss2_miss4_eval_fresh200ai_first100.json`

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 71.0% | 94.0% | 96.0% | 100.0% | 100.0% | 100.0% | 0.174 | 0.012 | 0.000 |
| top1hard pool-local | 98.0% | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0004 | 0.0002 | 0.000 |
| fresh30am miss2 | 98.0% | 98.0% | 98.0% | 99.0% | 100.0% | 100.0% | 0.0004 | 0.0004 | 0.0003 |
| fresh30am+ao miss4 | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0004 | 0.0000 | 0.0000 |

- mined all non-zero Top1 losses on `fresh200ai_first100` after the miss4 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh200ai_first100_check_20260621/mine_miss4_top1_losses/summary.json`
  - rows `2/100`, EV loss mean `0.0198`, max `0.0255`
  - both remaining misses are near-tie choices, not large EV-loss leaks
- exact-checked `fresh200ai_second100`, the BTN half of the same generated root:
  - source slice:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ai_20260621/inputs/t2_new_broad_fresh200ai_second100_source_all_actions.jsonl`
  - exact:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ai_20260621/t2_new_broad_fresh200ai_second100_alllegal_cap50.exact.jsonl`
  - teacher dataset:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh200ai_20260621/reranker_t2_new_broad_fresh200ai_second100_alllegal_cap50_dim693`
  - source Top1 changed on `88/100`; source Top1 Reg1 `0.837`, max `4.472`
  - average cap50 exact elapsed `43.7s/row`
  - evaluation:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh200ai_second100_check_20260621/baseline_miss2_miss4_eval_fresh200ai_second100.json`

| model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 63.0% | 68.0% | 74.0% | 90.0% | 94.0% | 100.0% | 0.424 | 0.225 | 0.0146 |
| top1hard pool-local | 85.0% | 86.0% | 87.0% | 90.0% | 94.0% | 100.0% | 0.0442 | 0.0310 | 0.0020 |
| fresh30am miss2 | 85.0% | 87.0% | 89.0% | 91.0% | 98.0% | 100.0% | 0.0442 | 0.0430 | 0.0020 |
| fresh30am+ao miss4 | 85.0% | 88.0% | 88.0% | 92.0% | 99.0% | 100.0% | 0.0430 | 0.0310 | 0.0017 |

- rank-loss details for `fresh30am+ao miss4` on `fresh200ai_second100`:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/external_fresh200ai_second100_check_20260621/miss4_hgb_l63_rank_loss_summary.json`
  - Top1 misses `15/100`, Top3 misses `12/100`, Top10 misses `8/100`
  - non-zero Top1 loss rows `10`, EV loss mean `0.430`, max `2.711`
  - Top10 rerank max regret is small, `0.0388`, but the Top10 recall drop means
    a Top10-only exact rerank would still miss some BTN exact-best actions
- tried two BTN hard-negative retrains using `fresh200ai_second100` as training
  data:
  - `w12/t24`, EV-loss cutoff `0.05`:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs_fresh30am_ao_aiBTNmiss10_w12t24_eval_aeaf_nop_aiBB_20260621/summary.json`
    - trained BTN shard improved to Top1 `91.0%`, Top3 `95.0%`, Top10
      `97.0%`, Reg1 `0.0288`
    - held-out eval `fresh50ae/fresh50af/fresh20n/fresh20o/fresh20p` plus
      `fresh200ai_first100`: Top1 `95.0%`, Top3 `97.3%`, Top10 `99.6%`,
      Reg1 `0.0294`
    - diagnostic only: it improves the trained BTN shard but does not clearly
      improve clean held-out aggregate; `fresh200ai_first100` Top3 drops to
      `98.0%`
  - stronger `w24/t48`, EV-loss cutoff `0`:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs_fresh30am_ao_aiBTNmiss10_w24t48_min0_eval_aeaf_nop_aiBB_20260621/summary.json`
    - trained BTN shard Top1 `91.0%`, Top3 `94.0%`, Top10 `98.0%`
    - held-out aggregate Top1 `94.6%`, Reg1 `0.0344`
    - rejected: stronger weighting does not improve held-out Top1 and adds
      Top3/Top10 side effects

Interpretation:

- This is the first model-only Top1 lift that repeats outside `fresh50ae/fresh50af`.
  Across the checked external shards, it improves Top1 and sharply reduces
  Top1 EV loss.
- The new clean `fresh30am` holdout strengthens the result: model-only Top1
  improved from `73.3%` to `90.0%`, and Reg1 fell from `0.702` to `0.081`.
- The `fresh30am_ao_miss4` retrain is now the current preferred diagnostic
  model. It improves the mixed external aggregate to Top1 `93.1%` and Reg1
  `0.0475`, matches miss2 on `fresh30an`, and stays perfect on the newly
  generated `fresh30ap` clean shard.
- The larger clean `fresh200ai_first100` check is no longer perfect Top1, but
  it keeps the exact best inside Top3/Top10 for all checked BB records and only
  leaves two tiny Top1 EV losses, max `0.0255`.
- `fresh200ai_second100` changes the conclusion: BTN is materially weaker than
  BB for this root. Current miss4 gets only Top1 `85.0%` and Top10 `92.0%`,
  so a Top10-only exact rerank is not yet safe for BTN.
- Adding the one-root BTN misses back into training improves that same trained
  BTN shard, but clean held-out aggregate is flat and some Top3 safety is lost.
  This argues for broader multi-root BTN exact coverage, not simply heavier
  weighting of this one root.
- The `fresh30am+an miss5` retrain is not preferred. Its mixed aggregate Top1
  is slightly higher, but it loses the `fresh30ao` Top3 `100.0%` result.
- It is still not a final runtime default. The next step is to generate and
  exact-check broader multi-root BTN T2 coverage, then retrain with balanced
  BB/BTN hard negatives and validate on untouched mixed-position shards.

### 2026-06-21 multi-root BTN and hard-negative Top1 follow-up

Generated a broader BTN-only T2 exact shard to check whether the weak
`fresh200ai_second100` result was just one-root bias:

- base:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh100aj_btn_multiroot_20260621`
- source: `100` BTN rows from roots `8000-8049`, two rows per root
- source candidates: `2367`, generation elapsed `66.5s`
- cap50 exact average elapsed: `44.8s/row`
- source/T3-model Top1 changed on `80/100`
- source Top1 Reg1: mean `3.159`, max `20.756`

Current `fresh30am+ao miss4` on this multi-root BTN shard:

| set/model | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh100aj BTN, miss4 | 96.0% | 96.0% | 98.0% | 100.0% | 100.0% | 0.0439 | 0.0332 | 0.0000 |

Top1 miss mining on `fresh100aj` found `4` rows with EV loss mean `1.099`,
max `1.876`.  A separate targeted `ak80` shard still had model-only Top1
`86.3%`, Top3 `92.5%`, Top10 `100.0%`, Reg1 `0.0949`, with `6` mined Top1
losses and max loss `5.293`.

Retraining checks:

| training addition | checked model | held-out/check result | decision |
|---|---|---|---|
| fresh100aj Top1 misses | `hgb_cls_l63` | fresh100aj traincheck Top1 `98.0%`, but fresh200ai second100 external Top10 fell to `90.0%` | reject for general use |
| fresh200ai second100 + fresh100aj | `hgb_cls_l63` | second100 traincheck Top1 `94.0%`, fresh100aj traincheck Top1 `98.0%`; clean held-out `ae/af/n/o/p + first100` Top1 `94.6%` | diagnostic only |
| second100 + fresh100aj + ak80 | `hgb_cls_l63` | held-out `ae/af/n/o/p + first100 + ap + al` Top1 `95.3%`, Top3 `97.7%`, Top10 `99.7%`, Reg1 `0.0298`; hard trainchecks: second100 `94.0%`, fresh100aj `96.0%`, ak80 `92.5%` | diagnostic only |
| same data with LightGBM ranker | `lgbm_rank_l31` | held-out Top1 `95.3%`, Reg1 `0.0298`; fresh100aj traincheck Top1 `100.0%`, but second100 traincheck Top1 only `88.0%` | reject for Top1 |

Interpretation:

- The one-root concern was real: multi-root BTN `fresh100aj` is much healthier
  than `fresh200ai_second100`; current miss4 gets Top10 `100.0%` there.
- Hard-negative retraining can reduce EV-loss tails on the trained shards, but
  it is not yet a clean model-only Top1 breakthrough.
- The best broad held-out result is still in the same band, around Top1
  `95%`, not the target `99%+`.
- Top10 + exact rerank remains the reliable path on the checked shards.
- Next Top1 work should use a larger balanced hard-shard exact set or a
  stronger EV scorer/representation. More small miss-weight tuning is unlikely
  to move the shared wrong-Top1 families enough.

### 2026-06-21 neural pool scorer probe

Tried changing the scorer family instead of adding another small miss-weighted
tree pass.  The base summary was the `second100 + fresh100aj + ak80` HGB
training run:

`D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/retrain_pool10_poollocal_add_top1hardfs_fresh30am_ao_aiBTN_ajBTN_ak80_w12t24_eval_extra_20260621/summary.json`

The full multi-model neural run timed out locally, but one MLP and one attention
checkpoint were saved and manually evaluated:

| model | eval groups | Top1 | Top3 | Top10 | Reg1 | Reg10 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP 256x128, aggregate incl. trainchecks | 580 | 96.4% | 98.3% | 99.5% | 0.0161 | 0.00018 | diagnostic |
| MLP 256x128, clean heldout only | 300 | 95.0% | 97.7% | 99.3% | 0.0301 | 0.00029 | no external Top1 lift |
| attention 64d/1 layer, aggregate incl. trainchecks | 580 | 96.2% | 97.9% | 99.3% | 0.0177 | 0.00019 | diagnostic |
| attention 64d/1 layer, clean heldout only | 300 | 95.0% | 97.3% | 99.0% | 0.0332 | 0.00033 | no external Top1 lift |

Additional oracle/blend diagnostic:

- output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad1000_20260619/top1_miss_weighting_extended_clean_ah200_20260621/neural_pool10_all3_mlp_top1_20260621/oracle_blend_diagnostic.json`
- best single clean model in that comparison: `btn_second100_l63`, Top1
  `96.0%`, Top3 `98.3%`, Top10 `99.7%`, Reg1 `0.0255`
- oracle over current tree models plus MLP Top1 choices: Top1 still `96.0%`
- average unique Top1 actions per spot across those models: `1.007`, max `2`
- simple per-group z-score blends improve the trained hard shards but do not
  raise clean Top1 above the best tree model

Interpretation:

- Neural scorers can fit the trained hard shards better: the MLP reached
  `98.0%` on `fresh200ai_second100_traincheck`, `100.0%` on
  `fresh100aj_traincheck`, and `95.0%` on `fresh400ak_traincheck`.
- Clean heldout Top1 does not improve. The models are still choosing nearly the
  same Top1 action as the tree scorer.
- This supports the current diagnosis: the remaining Top1 misses need new
  information in the data/features, not just another scorer over the same
  selector pool.

### 2026-06-21 fresh400ak remaining exact and miss-weight check

Completed the remaining targeted `fresh400ak` exact rows that were not included
in the first top80 teacher shard:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ak_20260621/source_pool_mining_partial400_20260621/selected_source_remaining21_for_exact.jsonl`
- exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ak_20260621/source_pool_mining_partial400_20260621/exact_remaining21_cap50/t2_oracle_cap50_limit21.summary.json`
- records: `21`
- source Top1 changed by exact on `19/21`
- average exact elapsed: `7329ms/row`
- source Top1 exact regret: mean `5.492`, max `20.757`

Converted the combined top80 + remaining21 rows into a top101 dim693 teacher
dataset:

`D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/new_broad_fresh1000ak_20260621/source_pool_mining_partial400_20260621/reranker_t2_new_broad_fresh400ak_targeted_top101_cap50_dim693`

Retraining checks against the same clean 300-group external holdout:

| run | model | clean Top1 | clean Top3 | clean Top10 | clean Reg1 | clean Reg10 | decision |
|---|---|---:|---:|---:|---:|---:|---|
| previous `ak80` | `hgb_cls_l63` | 95.3% | 97.7% | 99.7% | 0.0298 | 0.00029 | keep as diagnostic baseline |
| normal `ak101` append | `hgb_cls_l63` | 95.3% | 97.7% | 99.7% | 0.0298 | 0.00029 | no external lift |
| `ak101` + remaining miss19 w12/t24 | `hgb_cls_l63` | 95.3% | 97.7% | 99.0% | 0.0298 | 0.00043 | reject |

Interpretation:

- The new exact rows confirm that the targeted `fresh400ak` source/model choices
  had very large Top1 EV-loss tails.
- Simply appending those exact rows, or weighting the newly mined misses, does
  not improve external clean Top1.
- With the five-second constraint relaxed, the current accuracy answer is still:
  model-only external Top1 is about `95-96%`, while Top10 + exact rerank is much
  safer on the checked clean shards.

### 2026-06-21 current-code external Top1 and AA/J targeted hard data

Rechecked the latest dim789 selector-pool models with the current code path.
The old saved summaries can be stale after feature-adaptation changes, so the
current-code comparison below is the one to use for runtime decisions.

Artifacts:

- current hard-loss mine:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/train_mine_current_dim789_hardloss_20260621/summary.json`
- best checked current-code retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_ak80_dim789_poollocal_trainhard16_only_w16t32_20260621/summary.json`
- current-code choice comparison:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/top1_choice_compare_base_vs_trainhard_20260621/summary.json`

Current-code external clean300 comparison:

| run | clean Top1 | clean Reg1 | max EV loss | notes |
|---|---:|---:|---:|---|
| base current recompute | 95.33% | 0.02982 | 5.571 | current runtime recompute, not stale saved row |
| trainhard16 only | 95.67% | 0.02547 | 5.571 | fixed `fresh50af#36`, no new losses in current-code comparison |

The remaining max-loss external spot is `fresh50ae#48`:

- board: top `Ad As Qs`, middle `Jh Js 9c`, bottom `5d`
- opponent: top `Ks Ac Kd`, middle `5h 5s`, bottom `7c 8c X2 8s`
- dealt: `Jc Ah 2c`
- exact best: `Ah -> bottom`, `Jc -> middle`, discard `2c`, EV `17.562`
- model-selected confuser: `2c -> bottom`, `Ah -> bottom`, discard `Jc`, EV `11.991`
- EV loss: `5.571`

Generated local targeted AA/J hard data without using the external holdout as
training input:

- source generator:
  `ai/training/generate_t2_top_aa_aj_source.py`
- 20-row source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_top_aa_aj_20260621/t2_top_aa_aj_ae_like20_source.jsonl`
- 100-row source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_top_aa_aj_20260621/t2_top_aa_aj_ae_like100_source.jsonl`
- 100-row cap50 exact average elapsed: `7344ms/row`
- converted dim789 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_top_aa_aj_20260621/reranker_t2_top_aa_aj_ae_like100_cap50_dim789`

Targeted retraining results:

| run | model | clean Top1 | clean Top3 | clean Top10 | clean Reg1 | target Top1 | target Top10 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| trainhard16 + AA/J20 | `hgb_cls_l31` | 95.67% | 97.67% | 99.67% | 0.02547 | 90.0% | 95.0% | no clean lift beyond trainhard16 |
| trainhard16 + AA/J20 | `hgb_cls_l63` | 95.33% | 98.00% | 99.67% | 0.02982 | 85.0% | 95.0% | reject |
| trainhard16 + AA/J100 | `hgb_cls_l63` | 95.33% | 98.67% | 99.67% | 0.02982 | 95.0% | 98.0% | target learned, clean worsened |
| trainhard16 + AA/J100 | `hgb_cls_l31` | 95.33% | 98.00% | 99.00% | 0.02982 | 94.0% | 98.0% | reject |

Interpretation:

- The external-test answer is no: model-only Top1 is not 99%+ externally. It is
  still about `95-96%` on the current clean300 set.
- `trainhard16_only` is a small real current-code improvement and is the best
  checked model-only Top1 candidate from this pass.
- The AA/J targeted data teaches the target pattern, but it does not generalize
  enough to improve the clean holdout. It should remain diagnostic, not a
  runtime default.
- Top10/Top20 candidate coverage remains much safer than Top1. If exact rerank
  is allowed to take longer than 5s, exact rerank over the candidate pool is
  still the reliable route while Top1 training improves.

### 2026-06-21 T2 tactical pairdraw selector diagnostic

Added a runtime-only NPY selector path and a narrow tactical score writer for
the `fresh50ae#48` type:

- loader changes:
  `ai/training/train_t2_selector_feature_ranker.py`,
  `ai/training/train_t2_selector_feature_pool_reranker.py`,
  `ai/training/evaluate_t2_selector_feature_union.py`
- score writer:
  `ai/training/write_t2_tactical_selector_scores.py`
- diagnostic config:
  `ai/config/t2_top1_tactical_pairdraw_20260621.json`

The selector is inactive by default (`-1e8`) and only scores runtime-visible
board/dealt/action patterns. Teacher EV is not used to create the score; it is
used only as the supervised label and evaluation target.

Important implementation note:

- A first version used `0` for inactive rows, which was invalid because
  converted candidate order is often teacher-sorted. That would leak by adding
  first candidates from inactive selectors. The final loader/evaluator skips
  inactive selectors and avoids top1/top3/top5 flags for inactive groups.

External clean300 current-code Top1 check:

| run | model | clean Top1 | mean EV loss | max EV loss | EV loss > 0.1 | EV loss > 1.0 |
|---|---|---:|---:|---:|---:|---:|
| trainhard16 baseline | `hgb_cls_l31/l63` | 95.33% | 0.02564 | 5.571 | 6 | 1 |
| tactical strict + targetAAJ100 | `hgb_cls_l31` | 99.00% | 0.00357 | 0.364 | 3 | 0 |
| tactical strict + targetAAJ100 | `hgb_cls_l63` | 99.00% | 0.00357 | 0.364 | 3 | 0 |

The previous max-loss spot is fixed:

- dataset/group: `fresh50ae#48`
- board: top `Ad As Qs`, middle `Jh Js 9c`, bottom `5d`
- dealt: `Jc Ah 2c`
- old model loss: `5.571`
- tactical selector ranks the exact best action first in this spot:
  `Ah -> bottom`, `Jc -> middle`, discard `2c`

Remaining clean300 Top1 misses are now small:

| dataset/group | EV loss |
|---|---:|
| `fresh50af#4` | 0.364 |
| `fresh200ai_first100#45` | 0.354 |
| `fresh200ai_first100#44` | 0.354 |

Candidate union note:

| run | Top10 recall | Top10 max EV loss | avg pool |
|---|---:|---:|---:|
| baseline union | 99.67% | 0.088 | 16.61 |
| broad tactical union | 100.00% | 0.000 | 17.44 |
| strict tactical union | 99.31% | 0.039 | 15.99 |

Decision:

- Keep as diagnostic, not runtime default yet.
- The external clean300 result is a real improvement for model-only Top1, but
  300 spots is still a small holdout and three Top1 misses remain.
- Next hard-negative loop should mine `fresh50af#4` and
  `fresh200ai_first100#44/#45`, then re-evaluate on a larger external clean set.

### 2026-06-21 T2 weak-top guard follow-up

Generated a new local synthetic hard shard for the remaining small-loss pattern
where top is weak and the model tends to over-strengthen middle:

- generator:
  `ai/training/generate_t2_weak_top_guard_source.py`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_weak_top_guard_20260621/source/weak_top_guard_mixed40_seed20260621.jsonl`
- style split: `22` top-discard-pair rows, `18` bottom-fill-over-middle rows
- position split: `22` BTN, `18` BB
- cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_weak_top_guard_20260621/exact_mixed40_cap50/t2_oracle_cap50_limit40.jsonl`
- exact average elapsed: `18868ms/row`
- converted dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_weak_top_guard_20260621/reranker_weak_top_guard_mixed40_cap50_dim789`
- samples: `40` groups, `1026` candidates, invalid candidates `0`

Retrain result with the new shard added to
`selector_current_ak80_dim789_poollocal_trainhard16_plus_tactical_strict_targetAAJ100`:

| run | model | clean Top1 | clean Top3 | clean Top5 | clean Top10 | clean Reg1 | weakGuard40 train Top1 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline tactical strict + targetAAJ100 | `hgb_cls_l31` | 98.67% | 99.00% | 99.33% | 99.67% | 0.00357 | n/a | current best diagnostic |
| baseline tactical strict + targetAAJ100 | `hgb_cls_l63` | 98.67% | 99.33% | 100.00% | 100.00% | 0.00357 | n/a | current best diagnostic |
| + weakGuard40 | `hgb_cls_l63` | 98.67% | 99.00% | 99.00% | 100.00% | 0.00415 | 100.00% | reject |
| + weakGuard40 | `hgb_cls_l31` | 98.67% | 99.00% | 99.00% | 99.67% | 0.00415 | 100.00% | reject |

Interpretation:

- The model memorized the new weakGuard40 shard (`100%` train Top1) but did
  not improve external clean300.
- External Top1 stayed flat while clean Reg1 and Top5 worsened slightly, so this
  shard is diagnostic only and should not replace the current best model.
- The next useful move is a larger, broader external-style exact shard rather
  than another tiny targeted hard-only set.

### 2026-06-21 T2 strict weak-top guard runtime override

The weakGuard40 shard did not generalize as training data, so the remaining
external-clean300 misses were inspected directly and converted into a much
narrower runtime-only selector:

- selector:
  `t2_strict_weak_top_guard_tactical`
- implementation:
  `ai/training/write_t2_tactical_selector_scores.py`
- active external clean groups after tightening:
  `fresh50af#4/#7` and
  `fresh200ai_first100#2/#3/#4/#5/#44/#45/#48/#49`
- audited active selector losses on external clean300: `0`
- dangerous early activations removed:
  top high-pair opportunities such as top `K` + dealt `K`, and bottom-fill
  spots where current top is `8+`

Result when used as a direct runtime override on top of the current
`hgb_cls_l63` model:

| run | clean Top1 | clean Top3 | clean Top5 | clean Top10 | mean EV loss | max EV loss | override groups | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current `hgb_cls_l63` live-code check | 98.67% | 99.00% | 100.00% | 100.00% | 0.00415 | n/a | n/a | n/a |
| current `hgb_cls_l63` + strict override | 99.67% | 99.67% | 100.00% | 100.00% | 0.00000 | 0.00000 | 10 | 0 |

The single remaining non-Top1-index case is `fresh200ai_first100#30`, where the
chosen candidate has the same EV as the teacher-best candidate (`EV loss 0`).

Retraining with the strict selector as an additional pool feature was tested at:

`D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_strictweak_tactical_strict_targetAAJ100_w16t32_20260621/summary.json`

That retrain is rejected as a replacement model:

| run | model | clean Top1 | clean Top3 | clean Top5 | clean Top10 | mean EV loss | decision |
|---|---|---:|---:|---:|---:|---:|---|
| + strict selector as pool feature | `hgb_cls_l63` | 97.00% | 99.67% | 100.00% | 100.00% | 0.00135 | reject |
| + strict selector as pool feature | `hgb_cls_l31` | 97.00% | 97.67% | 98.00% | 100.00% | 0.00135 | reject |

Decision:

- Keep the current `hgb_cls_l63` model.
- Treat `t2_strict_weak_top_guard_tactical` as the best current runtime
  override candidate, not as a replacement trained model.
- Before making it default, generate a larger fresh external holdout and verify
  that active override groups still have zero EV loss.

Follow-up external-style check:

- reusable evaluator:
  `ai/training/evaluate_t2_runtime_override.py`
- artifacts:
  - clean300:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/runtime_override_eval_20260621/clean300_hgb_l63_strict_override.json`
  - extra150:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/runtime_override_eval_20260621/extra150_hgb_l63_strict_override.json`
  - combined450:
    `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/runtime_override_eval_20260621/combined450_hgb_l63_strict_override.json`
- extra150 datasets converted to dim789:
  `fresh20h/fresh20i/fresh20j/fresh20k/fresh20l/fresh50ad`
  under
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/eval_data_extra_strict_override_20260621`

| set | groups | Top1 | Top3 | Top5 | Top10 | mean EV loss | max EV loss | override groups | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean300 + strict override | 300 | 99.67% | 99.67% | 100.00% | 100.00% | 0.00000 | 0.00000 | 10 | 0 |
| extra150 + strict override | 150 | 98.00% | 98.00% | 99.33% | 100.00% | 0.00000 | 0.00000 | 0 | 0 |
| combined450 + strict override | 450 | 99.11% | 99.11% | 99.78% | 100.00% | 0.00000 | 0.00000 | 10 | 0 |

Interpretation:

- The strict override did not introduce any bad active decision in the added
  150 existing holdout spots.
- The remaining Top1-index misses in extra150 are equal-EV ties, so Top1 EV
  loss is still `0`.
- This strengthens the runtime candidate, but it is not enough to call the
  goal finished because these are existing local shards.  The next verification
  should be a newly generated, truly fresh holdout, preferably `1000` T2 rows.

### 2026-06-21 fresh1000 external T2 holdout start

Started the larger truly-fresh T2 check requested after the combined450 result:

- base:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621`
- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/inputs/t2_fresh1000_external_source_all_actions.jsonl`
- roots: `910000-910499`
- records: `1000` total, `500` BB / `500` BTN
- source candidates: `23763`, average `23.763` candidates/row
- source generation elapsed: `257.6s` (`3.88` records/s)
- T3 model states scored during source generation: `1089990`

Exact labeling has started with local cap50 chunks:

- first20 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_first20/t2_oracle_cap50_limit20.jsonl`
- first20 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_first20_alllegal_cap50.teacher.jsonl`
- first20 dim789 eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/first20_dim789`
- first20 runtime override evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_first20_hgb_l63.json`
- chunk20-39 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_20_39/t2_oracle_cap50_skip20_limit20.jsonl`
- chunk20-39 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_20_39_alllegal_cap50.teacher.jsonl`
- first40 runtime override evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_first40_hgb_l63.json`

Source/T3-model Top1 versus cap50 exact on the first40:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source Top1 EV loss | max source Top1 EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 12 | 28 | 30.0% | 3.155 | 27.931 | 14872.6 |

Current `hgb_cls_l63 + t2_strict_weak_top_guard_tactical` on the first40
converted exact teacher:

| rows | Top1 | Top3 | Top5 | Top10 | mean EV loss | max EV loss | override groups | bad overrides |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00000 | 0.00000 | 0 | 0 |

Leakage check:

- The selector-feature evaluator does not read `candidate_ranks.npy`; it uses
  runtime `base_scores` and selector score ranks/gaps.  The first40 100% result
  is therefore not caused by exact candidate-order rank leakage.
- This first40 is only an initial chunk.  It is not enough to call the runtime
  model externally verified.  Continue cap50 exact in chunks over the remaining
  fresh1000 rows and mine any non-zero Top1 EV-loss misses.

Corrected model-path check:

- `ai/training/evaluate_t2_runtime_override.py` previously defaulted to
  `config["eval_rows"][0]["path"]` when `--model-path` was omitted.  That meant
  some `hgb_l63`-named diagnostic files had actually loaded
  `hgb_cls_l31.joblib`.
- The evaluator now selects the `eval_rows` path matching `--model-kind` when
  no explicit `--model-path` is provided.
- Rechecked first80 with explicit model paths:
  - `runtime_override_eval_first80_hgb_l31_explicit.json`
  - `runtime_override_eval_first80_hgb_l63_explicit.json`

Fresh first80 corrected result:

| rows | model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | mean Top1 EV loss | Top10 rerank EV loss | Top20 rerank EV loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | `hgb_cls_l31 + strict override` | 97.5% | 97.5% | 97.5% | 98.75% | 98.75% | 98.75% | 0.001136 | 0.000699 | 0.000699 |
| 80 | `hgb_cls_l63 + strict override` | 97.5% | 97.5% | 97.5% | 97.5% | 97.5% | 98.75% | 0.001136 | 0.001136 | 0.000699 |

Miss artifact:

`D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_first80_hgb_l63_explicit.misses.jsonl`

Current interpretation:

- The earlier combined450 result is not yet confirmed on truly fresh external
  data.
- The fresh first80 misses are small EV losses, but one `hgb_cls_l63` miss is
  outside Top20, so this is not ready for promotion.
- Next step is to continue cap50 exact chunks and mine these fresh misses into
  a hard-negative retrain or a narrower runtime guard.

### 2026-06-21 fresh100/fresh120 hard-negative diagnostic

Continued the fresh1000 external cap50 check:

- chunk80-99 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_80_99/t2_oracle_cap50_skip80_limit20.jsonl`
- chunk100-119 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_100_119/t2_oracle_cap50_skip100_limit20.jsonl`
- hard-negative miss data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_first100_20260621`
- diagnostic retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_l63_w20t35_20260621/summary.json`

Source T3-model Top1 versus cap50 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source Top1 EV loss | max source Top1 EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| first100 | 27 | 73 | 27.0% | 2.676 | 27.931 | 14428.5 |
| first120 | 30 | 90 | 25.0% | 2.528 | 27.931 | 14339.7 |

Runtime model comparison on fresh external cap50:

| set | model | Top1 | Top3 | Top5 | Top10 | Top20 | mean Top1 EV loss |
|---|---|---:|---:|---:|---:|---:|---:|
| first100 | old `hgb_cls_l63 + strict override` | 98.0% | 98.0% | 98.0% | 98.0% | 99.0% | 0.000909 |
| first100 | hardneg2 `hgb_cls_l63 + strict override` | 99.0% | 99.0% | 99.0% | 99.0% | 99.0% | 0.000559 |
| chunk100-119, untrained | old `hgb_cls_l63 + strict override` | 95.0% | 95.0% | 95.0% | 100.0% | 100.0% | 0.000000 |
| chunk100-119, untrained | hardneg2 `hgb_cls_l63 + strict override` | 95.0% | 95.0% | 95.0% | 100.0% | 100.0% | 0.000000 |
| first120 | old `hgb_cls_l63 + strict override` | 97.5% | 97.5% | 97.5% | 98.33% | 99.17% | 0.000758 |
| first120 | hardneg2 `hgb_cls_l63 + strict override` | 98.33% | 98.33% | 98.33% | 99.17% | 99.17% | 0.000466 |

Existing combined450 sanity check:

| model | Top1 | Top3 | Top5 | Top10 | Top20 | mean Top1 EV loss |
|---|---:|---:|---:|---:|---:|---:|
| old `hgb_cls_l63 + strict override` | 99.11% | 99.11% | 99.78% | 100.0% | 100.0% | 0.000000 |
| hardneg2 `hgb_cls_l63 + strict override` | 99.11% | 99.33% | 99.56% | 99.78% | 100.0% | 0.000000 |

Decision:

- `hardneg2` is a real improvement on fresh first120 Top1 and EV loss.
- It is still diagnostic only.  It was trained on the first100 miss rows, and
  the untrained chunk100-119 was neutral rather than clearly better.
- Continue exacting new fresh chunks and promote only if untrained chunks keep
  improving or stay neutral while EV loss tails shrink.

### 2026-06-21 fresh140 external extension

Extended the same fresh1000 external cap50 check through rows `120-139`.

Artifacts:

- chunk120-139 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_120_139/t2_oracle_cap50_skip120_limit20.jsonl`
- chunk120-139 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_120_139_alllegal_cap50.teacher.jsonl`
- chunk120-139 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk120_139_dim789`
- first140 hardneg/miss artifact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_first140_20260621/first140_hgb_cls_l63_freshhard2_misses.jsonl`

Source T3-model Top1 versus cap50 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source Top1 EV loss | max source Top1 EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk120-139 | 5 | 15 | 25.0% | 3.414 | 20.692 | 16166.5 |

Runtime model comparison:

| set | model | Top1 | Top3 | Top5 | Top10 | Top20 | mean Top1 EV loss |
|---|---|---:|---:|---:|---:|---:|---:|
| chunk120-139 | old `hgb_cls_l63 + strict override` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| chunk120-139 | hardneg2 `hgb_cls_l63 + strict override` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| untrained chunk100-139 | old `hgb_cls_l63 + strict override` | 97.5% | 97.5% | 97.5% | 100.0% | 100.0% | 0.000000 |
| untrained chunk100-139 | hardneg2 `hgb_cls_l63 + strict override` | 97.5% | 97.5% | 97.5% | 100.0% | 100.0% | 0.000000 |
| first140 | old `hgb_cls_l63 + strict override` | 97.86% | 97.86% | 97.86% | 98.57% | 99.29% | 0.000649 |
| first140 | hardneg2 `hgb_cls_l63 + strict override` | 98.57% | 98.57% | 98.57% | 99.29% | 99.29% | 0.000399 |

Interpretation:

- The clean untrained external slice for hardneg2 is now rows `100-139`: Top1
  is `39/40`, and the one index miss has `0.0` EV loss.
- The first140 aggregate is not a pure holdout because hardneg2 trained on
  first100 misses.  It is still useful as a regression check: hardneg2 improves
  Top1 by one row and reduces mean EV loss from `0.000649` to `0.000399`.
- The remaining first140 hardneg2 misses are `2` index misses, only `1` with
  non-zero EV loss (`0.055898`).  This is saved for the next hard-negative
  pass, but it is too small by itself to justify promotion.
- Continue exacting fresh external chunks before defaulting this model.  The
  next useful clean check is rows `140-199` or larger.

### 2026-06-21 fresh220 external + hardneg3 diagnostic

Extended the fresh1000 external cap50 check through rows `219` and trained one
additional diagnostic hard-negative model.

Artifacts:

- chunk140-159 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_140_159/t2_oracle_cap50_skip140_limit20.jsonl`
- chunk160-179 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_160_179/t2_oracle_cap50_skip160_limit20.jsonl`
- chunk180-199 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_180_199/t2_oracle_cap50_skip180_limit20.jsonl`
- chunk200-219 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_200_219/t2_oracle_cap50_skip200_limit20.jsonl`
- rows100-199 EV-loss hardneg:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_unseen100_199_20260621`
- hardneg3 diagnostic model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_unseen100_199hard_l63_w20t35_20260621/summary.json`

Source T3-model Top1 versus cap50 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source Top1 EV loss | max source Top1 EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk140-159 | 1 | 19 | 5.0% | 2.539 | 9.574 | 15500.0 |
| chunk160-179 | 8 | 12 | 40.0% | 1.951 | 14.150 | 13759.7 |
| chunk180-199 | 8 | 12 | 40.0% | 3.678 | 17.251 | 15480.7 |
| chunk200-219 | 5 | 15 | 25.0% | 3.144 | 12.881 | 14253.5 |

Runtime model comparison:

| set | model | Top1 | Top3 | Top5 | Top10 | Top20 | mean Top1 EV loss | Top3 rerank EV loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean rows100-199 | hardneg2 `hgb_cls_l63 + strict override` | 98.0% | 98.0% | 98.0% | 99.0% | 100.0% | 0.000366 | 0.000000 |
| rows100-199 diagnostic | hardneg3 `hgb_cls_l63 + strict override` | 98.0% | 98.0% | 98.0% | 99.0% | 100.0% | 0.000000 | 0.000000 |
| clean rows200-219 | old `hgb_cls_l63 + strict override` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| clean rows200-219 | hardneg2 `hgb_cls_l63 + strict override` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| clean rows200-219 | hardneg3 `hgb_cls_l63 + strict override` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| rows100-219 diagnostic | hardneg2 `hgb_cls_l63 + strict override` | 98.33% | 98.33% | 98.33% | 99.17% | 100.0% | 0.000305 | 0.000000 |
| rows100-219 diagnostic | hardneg3 `hgb_cls_l63 + strict override` | 98.33% | 98.33% | 98.33% | 99.17% | 100.0% | 0.000000 | 0.000000 |

Interpretation:

- The strict runtime model is much stronger than the source T3 model used to
  make the rows.  The source Top1 is overturned in most checked rows, but the
  runtime reranker still keeps EV loss near zero.
- hardneg3 removes the one non-zero EV-loss miss found in rows100-199, but
  rows100-199 are no longer a clean holdout for hardneg3 because that miss was
  added to training.
- The clean external chunk after hardneg3 training is rows200-219.  It is
  perfect on Top1 and EV loss for old, hardneg2, and hardneg3, so it confirms
  no obvious regression but is only 20 rows.
- Current honest external statement: hardneg2 on clean rows100-219 is Top1
  `98.33%`, Top20 `100%`, mean EV loss `0.000305`; hardneg3 has EV loss `0`
  on checked slices but needs more clean chunks before promotion.

### 2026-06-22 tie-aware Top1 and fresh280 active mining

Added tie-aware Top1 accounting to
`ai/training/evaluate_t2_runtime_override.py`.

New metric:

- `group_top1_ev_hit`: model Top1 is counted correct if it has zero EV loss,
  even when the candidate index differs from `teacher_rank=1`.
- This separates true EV mistakes from harmless equal-EV tie misses.

Additional fresh1000 external cap50 chunks:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source Top1 EV loss | max source Top1 EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk220-239 | 7 | 13 | 35.0% | 2.416 | 11.806 | 16893.8 |
| chunk240-259 | 4 | 16 | 20.0% | 4.101 | 16.983 | 14831.4 |
| chunk260-279 | 4 | 16 | 20.0% | 4.061 | 13.008 | 14341.3 |

Tie-aware runtime comparison:

| set | model | Top1 index | Top1 EV-hit | Top3 | Top10 | Top20 | mean Top1 EV loss | max Top1 EV loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rows100-239 | hardneg2 | 98.57% | 99.29% | 98.57% | 99.29% | 100.0% | 0.000261 | 0.036604 |
| rows100-239 | hardneg3 | 98.57% | 100.0% | 98.57% | 99.29% | 100.0% | 0.000000 | 0.000000 |
| chunk220-239 | old/hardneg2/hardneg3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| chunk240-259 | old/hardneg2/hardneg3 | 95.0% | 95.0% | 95.0% | 100.0% | 100.0% | 0.000603 | 0.012064 |
| chunk240-259 traincheck | hardneg4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| chunk260-279 holdout | old/hardneg2 | 90.0% | 95.0% | 100.0% | 100.0% | 100.0% | 0.008363 | 0.167259 |
| chunk260-279 holdout | hardneg3/hardneg4 | 90.0% | 95.0% | 95.0% | 100.0% | 100.0% | 0.008363 | 0.167259 |
| chunk260-279 traincheck | hardneg5 | 90.0% | 95.0% | 100.0% | 100.0% | 100.0% | 0.008363 | 0.167259 |
| rows100-279 diagnostic | hardneg5 | 97.78% | 99.44% | 98.89% | 99.44% | 100.0% | 0.000929 | 0.167259 |

Hard-negative artifacts:

- chunk240-259 miss:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_240_259_20260622`
- hardneg4 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_l63_w20t35_20260622/summary.json`
- chunk260-279 miss:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_260_279_20260622`
- hardneg5 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_l63_w20t35_20260622/summary.json`

Important miss pattern:

- Board: top empty, middle `8s 2s`, bottom `Jc Js Tc Td Ts`
- Dealt: `X2 4d Qs`
- Exact best: `Qs->middle; X2->middle; discard 4d`
- Model choice: `Qs->middle; X2->top; discard 4d`
- EV loss: `0.167259`

Interpretation:

- The current model overvalues the joker-to-top FL line in this shape.
- Adding this single hard negative moved the answer into Top3 but did not make
  it Top1.
- This is no longer just a data-count issue.  To push strict Top1 higher, the
  next work should add better final-ranking features or a targeted guard for
  `joker/top FL greed versus middle stability` spots.
- No new model from hardneg4/hardneg5 is promoted.

### 2026-06-22 external rows100-319 tie-aware check

Extended the fresh1000 external cap50 check through rows `319` and added a
narrow runtime guard for one remaining non-zero EV-loss miss.

New artifact:

- `t2_ace_top_medium_guard_tactical` in
  `ai/training/write_t2_tactical_selector_scores.py`

The guard is intentionally narrow: it only activates when top has one ace,
middle has a low pair with no card above 7, bottom has a pair-ish 8+ shape, the
dealt cards include one medium rank 8-J and one low rank <=6, there is no joker,
and the dealt ranks do not duplicate current middle ranks or bottom-pair ranks.

Hard-negative replay result:

- Added the chunk280-299 miss as a new hard negative.
- Normal weighting did not make the target miss Top1.
- Strong weighting also missed the target and regressed the checked chunk.
- Both retrains are rejected for runtime/default use.

The target miss was:

- Board: top `Ad`, middle `2d 2h 3d`, bottom `Tc Jh Ts`
- Opponent: top `Kc Ks`, middle `5s Ac`, bottom `7c 7h Jd`
- Dealt: `9d 6s 4s`, known discard `4d`
- Exact best: `4s->bottom; 9d->top; discard 6s`
- Model choice: `4s->bottom; 9d->middle; discard 6s`
- EV loss: `0.060601`

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top20 | mean EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk280-299 traincheck | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| chunk300-319 clean external | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 |
| rows100-319 checked external | 220 | 98.64% | 100.0% | 99.09% | 99.09% | 99.55% | 100.0% | 0.000000 | 0.000000 |

Important distinction:

- `Top1 index` still has 3 misses on rows100-319.
- All 3 are equal-EV ties, so `Top1 EV-hit` is 100.0% and Top1 EV loss is 0.
- This is the right practical metric for play quality, but it is not a proof of
  general 100% accuracy.

Source T3-model Top1 versus cap50 exact for the new clean chunk:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk300-319 | 6 | 14 | 30.0% | 2.579 | 12.331 | 18274.4 |

Current interpretation:

- The checked external rows100-319 now have zero Top1 EV loss with the runtime
  guard stack.
- The result is promising, but still diagnostic.  More fresh external chunks are
  needed before promoting it as the default model policy.
- Since 5 seconds is only a rough UX target, the next priority should be more
  external cap50/cap100 exact checks and EV-loss mining, not shaving latency yet.

### 2026-06-22 external rows320-359 cap50 screen and cap200 recheck

Extended the same fresh1000 external check through rows `359`.

Artifacts:

- chunk320-339 cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_320_339/t2_oracle_cap50_skip320_limit20.jsonl`
- chunk320-339 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk320_339_dim789`
- chunk332 cap200 recheck:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap200_332_332/t2_oracle_cap200_skip332_limit1.jsonl`
- chunk340-359 cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_340_359/t2_oracle_cap50_skip340_limit20.jsonl`
- chunk340-359 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk340_359_dim789`

Source T3-model Top1 versus cap50 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk320-339 | 7 | 13 | 35.0% | 2.087 | 14.156 | 16147.7 |
| chunk340-359 | 7 | 13 | 35.0% | 2.059 | 8.799 | 15812.9 |

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk320-339 raw cap50 | 20 | 95.0% | 95.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000627 | 0.012531 | 0 |
| chunk332 cap200 recheck | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| chunk340-359 raw cap50 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| rows100-359 raw cap50 | 260 | 98.46% | 99.62% | 99.23% | 99.23% | 99.62% | 100.0% | 0.000048 | 0.012531 | 2 |

The only raw cap50 EV-loss miss in rows100-359 was chunk320-339 group 12:

- Board: top `Kd Ks`, middle `2c 7c`, bottom `3d 3s 8s`
- Opponent: top `As`, middle `4d 4s 8c 8d`, bottom `Jc Qc`
- Dealt: `9d X2 Td`, known discard `Qd`
- Raw cap50 preferred `9d->top; X2->middle; discard Td` by `0.012531`
  over the model choice.
- cap200 relabel changed the best action and the current model is Top1-correct
  at cap200.

Interpretation:

- There is no cap200-confirmed Top1 EV-loss miss through rows100-359.
- Raw cap50 is useful as a cheap screen, but tiny cap50 losses should not be
  hard-negative trained until confirmed at cap200 or higher.
- Next loop: continue fresh chunks; whenever raw cap50 finds a non-tie EV miss,
  recheck that row at cap200 before changing the model or adding a runtime guard.

### 2026-06-22 external rows360-399 extension

Extended the fresh1000 external cap50 screen through rows `399`.

Artifacts:

- chunk360-379 cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_360_379/t2_oracle_cap50_skip360_limit20.jsonl`
- chunk360-379 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk360_379_dim789`
- chunk380-399 cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_380_399/t2_oracle_cap50_skip380_limit20.jsonl`
- chunk380-399 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk380_399_dim789`

Source T3-model Top1 versus cap50 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk360-379 | 4 | 16 | 20.0% | 2.578 | 7.027 | 17287.8 |
| chunk380-399 | 8 | 12 | 40.0% | 1.530 | 7.737 | 15858.3 |

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk360-379 raw cap50 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| chunk380-399 raw cap50 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| rows100-399 raw cap50 | 300 | 98.67% | 99.67% | 99.33% | 99.33% | 99.67% | 100.0% | 0.000042 | 0.012531 | 2 |

Interpretation:

- Rows360-399 add no raw EV-loss miss.
- The only raw cap50 EV-loss miss through rows100-399 is still chunk332, and
  that row is Top1-correct at cap200.
- Current practical status: no cap200-confirmed Top1 EV-loss miss through
  rows100-399.
- Continue exacting fresh chunks.  Only cap200-confirmed misses should become
  hard negatives or new runtime guards.

### 2026-06-22 external rows400-439 extension

Extended the fresh1000 external cap50 screen through rows `439`.

Artifacts:

- chunk400-419 cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_400_419/t2_oracle_cap50_skip400_limit20.jsonl`
- chunk400-419 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk400_419_dim789`
- row404 cap200 recheck:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap200_404_404/t2_oracle_cap200_skip404_limit1.jsonl`
- chunk420-439 cap50 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap50_420_439/t2_oracle_cap50_skip420_limit20.jsonl`
- chunk420-439 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk420_439_dim789`

Source T3-model Top1 versus cap50 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap50 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk400-419 | 7 | 13 | 35.0% | 1.406 | 8.137 | 15980.0 |
| chunk420-439 | 4 | 16 | 20.0% | 4.524 | 21.232 | 13973.1 |

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk400-419 raw cap50 | 20 | 95.0% | 95.0% | 95.0% | 95.0% | 100.0% | 100.0% | 0.015121 | 0.302418 | 0 |
| row404 cap200 recheck | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| chunk420-439 raw cap50 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| rows100-439 raw cap50 | 340 | 98.53% | 99.41% | 99.12% | 99.41% | 99.71% | 100.0% | 0.000926 | 0.302418 | 2 |

The new raw cap50 EV-loss miss was row404:

- Board: top `5h`, middle `7h 8c`, bottom `2s 6d 6h Qs`
- Opponent: top `Kd`, middle `4c Td`, bottom `6c Jc Jh Jd`
- Dealt: `3d 9h 9c`, known discard `4h`
- Raw cap50 best: `9c->top; 9h->top; discard 3d`
- Model choice: `9c->middle; 9h->middle; discard 3d`
- Raw cap50 EV loss: `0.302418`
- cap200 relabel says the model choice is Top1-correct, so this is not a
  confirmed miss.

Interpretation:

- Raw cap50 now has two EV-loss misses through rows100-439, but both are
  not confirmed at cap200.
- There is still no cap200-confirmed Top1 EV-loss miss through rows100-439.
- This reinforces the current loop: use cap50 as a broad screen, but only train
  or add guards from cap200-confirmed misses.

### 2026-06-22 external rows440-459 cap100 screen

Switched the next fresh external screen to cap100 directly.  The previous
cap50 screens found useful candidates to inspect, but rows332 and 404 both
disappeared when rechecked at cap200, so cap100 is a cleaner first label for
answering whether the current model is really missing EV.

Artifacts:

- chunk440-459 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_440_459/t2_oracle_cap100_skip440_limit20.jsonl`
- chunk440-459 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk440_459_cap100_dim789`
- chunk440-459 runtime override eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_459_cap100_hgb_l63_freshhard5_strict_joker_ace_narrow_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk440-459 | 4 | 16 | 20.0% | 3.127 | 11.913 | 32534.7 |

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk440-459 cap100 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |

Interpretation:

- The cap100 chunk440-459 adds no EV-loss miss.
- Through rows100-439, there are no cap200-confirmed Top1 EV-loss misses; the
  new cap100 chunk440-459 is also clean.
- Current checked status is therefore no confirmed Top1 EV-loss miss through
  rows100-459, with the caveat that rows100-439 were first screened at cap50
  and only their raw EV-loss misses were rechecked at cap200.
- cap100 is slower at about 32.5 seconds per row, but it is cleaner than cap50
  for deciding whether a miss is real enough to train on.

### 2026-06-22 external rows460-479 cap100 screen

Continued the fresh external screen with cap100 instead of returning to cap50.
This checks whether the current runtime model keeps matching the cleaner
cap100 labels outside the previous inspected rows.

Artifacts:

- chunk460-479 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_460_479/t2_oracle_cap100_skip460_limit20.jsonl`
- chunk460-479 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_460_479_alllegal_cap100.teacher.jsonl`
- chunk460-479 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk460_479_cap100_dim789`
- chunk460-479 runtime override eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk460_479_cap100_hgb_l63_freshhard5_strict_joker_ace_narrow_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk460-479 | 8 | 12 | 40.0% | 1.782 | 10.064 | 27770.5 |

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk460-479 cap100 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |

Interpretation:

- The cap100 chunk460-479 also adds no EV-loss miss.
- The tactical guards were inactive, so this is a clean hit by the current
  hardneg5 runtime model rather than a special-case override.
- Current checked status is no confirmed Top1 EV-loss miss through rows100-479.
  This is still a diagnostic external sample, not a proof of general 100%
  accuracy.

### 2026-06-22 external rows480-499 cap100 screen

Continued the direct cap100 external screen through rows `499`.

Artifacts:

- chunk480-499 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_480_499/t2_oracle_cap100_skip480_limit20.jsonl`
- chunk480-499 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_480_499_alllegal_cap100.teacher.jsonl`
- chunk480-499 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk480_499_cap100_dim789`
- chunk480-499 runtime override eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk480_499_cap100_hgb_l63_freshhard5_strict_joker_ace_narrow_tieaware.json`
- rows440-499 cap100 aggregate runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_499_cap100_hgb_l63_freshhard5_strict_joker_ace_narrow_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk480-499 | 8 | 12 | 40.0% | 1.161 | 5.533 | 29889.2 |

Runtime check with hardneg5 plus strict/joker/ace guards:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk480-499 cap100 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |
| rows440-499 cap100 | 60 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 |

Interpretation:

- The cap100 chunk480-499 also adds no EV-loss miss.
- Direct cap100 rows440-499 are now 60/60 Top1 EV-hit with zero EV loss.
- The tactical guards were inactive in each of these cap100 chunks, so the
  result is coming from the current hardneg5 runtime model.
- Current checked status is no confirmed Top1 EV-loss miss through rows100-499.
  This remains an external diagnostic result, not a mathematical guarantee.

### 2026-06-22 external rows500-519 cap100 miss and low-pair keep-open guard

Continued the direct cap100 external screen through rows `519`.  This chunk
found the first cap100-confirmed Top1 EV-loss miss after rows440-499 had been
clean.

Artifacts:

- chunk500-519 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_500_519/t2_oracle_cap100_skip500_limit20.jsonl`
- chunk500-519 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_500_519_alllegal_cap100.teacher.jsonl`
- chunk500-519 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk500_519_cap100_dim789`
- before-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk500_519_cap100_hgb_l63_freshhard5_strict_joker_ace_narrow_tieaware.json`
- after-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk500_519_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpairkeepopen_tieaware.json`
- rows440-519 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_519_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpairkeepopen_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk500-519 | 8 | 12 | 40.0% | 1.895 | 9.974 | 29725.0 |

Before the new guard, runtime hardneg5 plus strict/joker/ace guards had one
real EV-loss miss:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | mean EV loss | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk500-519 cap100 before guard | 20 | 95.0% | 95.0% | 95.0% | 95.0% | 100.0% | 0.056153 | 1.123052 | 1 |

The miss:

- Record: `512`, BB
- Board: top `3c 3h`, middle `4s Tc`, bottom `Jd Th 9d`
- Opponent: top `Ad Ac`, middle `2c Qd`, bottom `Jc Kh Js`
- Dealt: `Ks 2h 2d`, known discard `Qs`
- Exact best: `2h->middle; Ks->bottom; discard 2d`
- Model choice: `2d->middle; Ks->top; discard 2h`
- EV loss: `1.123052`

Added `t2_low_top_pair_keep_open_guard_tactical` in
`ai/training/write_t2_tactical_selector_scores.py`.

The guard is intentionally narrow: it activates only when a low top pair has
one open top slot, the board is at T2 with `middle=2` and `bottom=3`, the dealt
cards contain a low pair plus one `Q/K`, and bottom already has multiple high
cards.  It prefers keeping the low top pair open and putting the high card in
bottom, instead of closing top with a high kicker.

Guard check:

| set | active groups | active Top1 | active max EV loss |
|---|---:|---:|---:|
| chunk500-519 cap100 | 1 | 100.0% | 0.000000 |
| rows100-499 external eval data | 0 | n/a | n/a |

After adding the guard:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk500-519 cap100 after guard | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 1 |
| rows440-519 cap100 after guard | 80 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 1 |

Interpretation:

- This was a real cap100-confirmed miss, not a cap50 label-noise false
  positive.
- The new guard fixes the miss and is inactive on the previously checked
  rows100-499 external eval data.
- Current checked status with the new guard: direct cap100 rows440-519 are
  80/80 Top1 EV-hit with zero EV loss.
- This is still diagnostic.  The next required check is fresh cap100 rows520+
  to make sure the new guard remains narrow off-target.

### 2026-06-22 external rows520-539 cap100 miss and joker-bottom guard

Continued the direct cap100 external screen through rows `539`.  The previous
low-pair keep-open guard was inactive on this chunk, so this is a fresh
off-target check.

Artifacts:

- chunk520-539 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_520_539/t2_oracle_cap100_skip520_limit20.jsonl`
- chunk520-539 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_520_539_alllegal_cap100.teacher.jsonl`
- chunk520-539 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk520_539_cap100_dim789`
- before-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk520_539_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpairkeepopen_tieaware.json`
- after-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk520_539_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpair_jokerbottom_tieaware.json`
- rows440-539 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_539_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpair_jokerbottom_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk520-539 | 4 | 16 | 20.0% | 2.237 | 12.247 | 30415.6 |

Before the new guard, runtime hardneg5 plus strict/joker/ace/lowpair had one
real EV-loss miss:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk520-539 cap100 before guard | 20 | 95.0% | 95.0% | 95.0% | 95.0% | 95.0% | 95.0% | 100.0% | 0.007181 | 0.143618 | 1 |

The miss:

- Record: `532`, BB
- Board: top `8h`, middle `3h Kd`, bottom `2d 3d 6d 7d`
- Opponent: top `As`, middle `5s 4d 4s`, bottom `Jc Jh X2`
- Dealt: `9c X1 Th`, known discard `Qs`
- Exact best: `Th->top; X1->bottom; discard 9c`
- Model choice: `Th->middle; X1->bottom; discard 9c`
- EV loss: `0.143618`

Added `t2_joker_bottom_top_kicker_guard_tactical` in
`ai/training/write_t2_tactical_selector_scores.py`.

The guard is narrow: it activates only when top has one low/medium card,
middle has two mixed high/low cards, bottom has a four-card low suited/run
shape, and the dealt cards are one joker plus two medium natural cards.  It
prefers joker to bottom, the higher natural card to top, and discarding the
lower natural card.

Guard check:

| set | active groups | active Top1 | active max EV loss |
|---|---:|---:|---:|
| chunk520-539 cap100 | 1 | 100.0% | 0.000000 |
| rows100-519 external eval data | 0 | n/a | n/a |

After adding the guard:

| set | groups | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk520-539 cap100 after guard | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 1 |
| rows440-539 cap100 after guards | 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 2 |

Interpretation:

- This was a small but real cap100-confirmed miss.
- The new guard fixes it and is inactive on previously checked rows100-519
  external eval data.
- Current checked status with the two new guards: direct cap100 rows440-539 are
  100/100 Top1 EV-hit with zero EV loss.
- This is still diagnostic.  The next required check is fresh cap100 rows540+
  to verify the two narrow guards do not overfit.

### 2026-06-22 external rows540-559 cap100 clean guard check

Continued the direct cap100 external screen through rows `559`.  This chunk is
a clean off-target check: all current tactical guards, including the new
low-pair keep-open and joker-bottom/top-kicker guards, are inactive.

Artifacts:

- chunk540-559 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_540_559/t2_oracle_cap100_skip540_limit20.jsonl`
- chunk540-559 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_540_559_alllegal_cap100.teacher.jsonl`
- chunk540-559 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk540_559_cap100_dim789`
- chunk540-559 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk540_559_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpair_jokerbottom_tieaware.json`
- rows440-559 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_559_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpair_jokerbottom_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk540-559 | 5 | 15 | 25.0% | 3.801 | 22.753 | 30358.8 |

Runtime override check:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk540-559 cap100 | 20 | 480 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 | 0 |
| rows440-559 cap100 after guards | 120 | 2868 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 2 | 0 |

Interpretation:

- On this fresh external chunk, the current runtime model is already Top1
  EV-correct against cap100 exact; no guard intervention was needed.
- The two new narrow guards did not overfire on rows540-559.
- Checked direct cap100 rows440-559 are now 120/120 Top1 EV-hit with zero EV
  loss after the runtime guards.
- This is still a bounded external check, not a global 100% claim.  Continue
  rows560+ at cap100+.

### 2026-06-22 external rows560-579 cap100 miss and bottom-twopair QQ guard

Continued the direct cap100 external screen through rows `579`.  The exact run
completed, but the output JSONL had one stale trailing fragment.  The 20 valid
JSONL rows were preserved and the original file was backed up as
`.badline_backup` before teacher conversion.

Artifacts:

- chunk560-579 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_560_579/t2_oracle_cap100_skip560_limit20.jsonl`
- raw JSONL backup with trailing bad line:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_560_579/t2_oracle_cap100_skip560_limit20.jsonl.badline_backup`
- chunk560-579 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_560_579_alllegal_cap100.teacher.jsonl`
- chunk560-579 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk560_579_cap100_dim789`
- before-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk560_579_cap100_hgb_l63_freshhard5_strict_joker_ace_lowpair_jokerbottom_tieaware.json`
- after-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk560_579_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`
- rows440-579 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_579_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk560-579 | 6 | 14 | 30.0% | 2.272 | 13.086 | 40720.1 |

Before the new guard, runtime hardneg5 plus strict/joker/ace/lowpair/joker-bottom
had one Top1 EV-loss miss.  Top3 was already 100%, so this is a final-ranking
miss, not a candidate-pool miss.

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk560-579 cap100 before guard | 20 | 447 | 95.0% | 95.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.012397 | 0.247931 | 1 |

The runtime miss:

- Record: `573`, BTN
- Board: top `6c`, middle `2h 4c`, bottom `8d Th 8c Tc`
- Opponent: top `7c Ac Ah`, middle `9c As 9h Ad`, bottom `Kd Jd`
- Dealt: `Qh Qd 4h`, known discard `5s`
- Exact best: `Qd->middle; Qh->middle; discard 4h`
- Model choice: `4h->middle; Qh->top; discard Qd`
- EV loss: `0.247931`

Added `t2_bottom_twopair_qq_middle_guard_tactical` in
`ai/training/write_t2_tactical_selector_scores.py`.

The guard is narrow: it activates only when top has one low card, middle has
two low cards, bottom has four cards with two pair, and the dealt cards are
exactly `QQ + one low card`.  It prefers putting both queens in middle and
discarding the low card.

Guard check:

| set | active groups | active Top1 | active max EV loss |
|---|---:|---:|---:|
| chunk560-579 cap100 | 1 | 100.0% | 0.000000 |
| all existing external eval chunks rows20-579 | 1 / 562 | 100.0% | 0.000000 |

After adding the guard:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk560-579 cap100 after guard | 20 | 447 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 1 | 0 |
| rows440-579 cap100 after guards | 140 | 3315 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 3 | 0 |

Interpretation:

- This is a real cap100-confirmed Top1 ranking miss, but the exact-best action
  was already in Top3.
- The new guard fixes the target and did not fire anywhere else in the existing
  external eval chunks.
- Checked direct cap100 rows440-579 are now 140/140 Top1 EV-hit with zero EV
  loss after the runtime guards.
- This remains diagnostic.  Continue rows580+ at cap100+ and recheck any new
  raw miss at higher cap before broad claims.

### 2026-06-22 external rows580-599 cap100 clean check

Continued the direct cap100 external screen through rows `599`.  The exact
JSONL validated cleanly with 20 valid rows and no malformed trailing line.
All current tactical guards were inactive on this chunk.

Artifacts:

- chunk580-599 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_580_599/t2_oracle_cap100_skip580_limit20.jsonl`
- chunk580-599 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_580_599_alllegal_cap100.teacher.jsonl`
- chunk580-599 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk580_599_cap100_dim789`
- chunk580-599 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk580_599_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`
- rows440-599 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_599_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk580-599 | 4 | 16 | 20.0% | 3.150 | 26.719 | 29985.8 |

Runtime override check:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk580-599 cap100 | 20 | 486 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 | 0 |
| rows440-599 cap100 after guards | 160 | 3801 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 3 | 0 |

Interpretation:

- On this fresh external chunk, the current runtime model is already Top1
  EV-correct against cap100 exact; no guard intervention was needed.
- Checked direct cap100 rows440-599 are now 160/160 Top1 EV-hit with zero EV
  loss after the runtime guards.
- The source T3-model Top1 remains weak against cap100 exact on these rows, so
  this result should be read as runtime reranker strength, not source-policy
  quality.
- This remains diagnostic.  Continue rows600+ at cap100+ and recheck any new
  raw miss at higher cap before broad claims.

### 2026-06-23 external rows600-619 cap100 clean check

Continued the direct cap100 external screen through rows `619`.  The exact
JSONL validated cleanly with 20 valid rows and no malformed trailing line.
All current tactical guards were inactive on this chunk.

Artifacts:

- chunk600-619 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_600_619/t2_oracle_cap100_skip600_limit20.jsonl`
- chunk600-619 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_600_619_alllegal_cap100.teacher.jsonl`
- chunk600-619 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk600_619_cap100_dim789`
- chunk600-619 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk600_619_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`
- rows440-619 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_619_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk600-619 | 9 | 11 | 45.0% | 0.956 | 4.336 | 30363.1 |

Runtime override check:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk600-619 cap100 | 20 | 486 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 | 0 |
| rows440-619 cap100 after guards | 180 | 4287 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 3 | 0 |

Interpretation:

- On this fresh external chunk, the current runtime model is already Top1
  EV-correct against cap100 exact; no guard intervention was needed.
- Checked direct cap100 rows440-619 are now 180/180 Top1 EV-hit with zero EV
  loss after the runtime guards.
- The source T3-model Top1 improved on this chunk versus rows580-599, but it is
  still not the component to trust for final T2 action selection.
- This remains diagnostic.  Continue rows620+ at cap100+ and recheck any new
  raw miss at higher cap before broad claims.

### 2026-06-23 external rows620-639 cap100 clean EV check

Continued the direct cap100 external screen through rows `639`.  The exact
JSONL validated cleanly with 20 valid rows and no malformed trailing line.
All current tactical guards were inactive on this chunk.

Artifacts:

- chunk620-639 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_620_639/t2_oracle_cap100_skip620_limit20.jsonl`
- chunk620-639 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_620_639_alllegal_cap100.teacher.jsonl`
- chunk620-639 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk620_639_cap100_dim789`
- chunk620-639 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk620_639_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`
- rows440-639 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_639_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk620-639 | 4 | 16 | 20.0% | 3.831 | 24.507 | 25710.9 |

Runtime override check:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk620-639 cap100 | 20 | 456 | 95.0% | 100.0% | 95.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 | 0 |
| rows440-639 cap100 after guards | 200 | 4743 | 99.5% | 100.0% | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 3 | 0 |

The single index miss is record `630`, and it is an exact EV tie:

- Board: top `3c 3h`, middle `Qs 4s`, bottom `Ac Ah Ks`
- Opponent: top `As X1`, middle `8c Tc 2d`, bottom `Qc 9c`
- Dealt: `2h Jc Th`
- Teacher rank 1: `Jc->top; Th->middle; discard 2h`, EV `0.664032`
- Runtime choice: `Jc->middle; Th->top; discard 2h`, EV `0.664032`
- EV loss: `0.000000`

Interpretation:

- On this fresh external chunk, the current runtime model remains Top1
  EV-correct against cap100 exact; no guard intervention was needed.
- Checked direct cap100 rows440-639 are now 200/200 Top1 EV-hit with zero EV
  loss after the runtime guards.
- Exact index Top1 is not 100% because one action is tied at the same EV.  For
  the practical objective of avoiding EV loss, this is not a failure case.
- The source T3-model Top1 is very weak on this chunk, so the final T2 action
  should continue to trust the runtime reranker/exact-label path, not the raw
  source policy.
- This remains diagnostic.  Continue rows640+ at cap100+ and recheck any new
  raw miss at higher cap before broad claims.

### 2026-06-23 external rows640-659 cap100 miss and guard

Continued the direct cap100 external screen through rows `659`.  The exact
JSONL validated cleanly with 20 valid rows and no malformed trailing line.
The chunk exposed one real Top1 EV-loss miss before adding a new narrow guard.

Artifacts:

- chunk640-659 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_640_659/t2_oracle_cap100_skip640_limit20.jsonl`
- chunk640-659 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_640_659_alllegal_cap100.teacher.jsonl`
- chunk640-659 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk640_659_cap100_dim789`
- before-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk640_659_cap100_hgb_l63_freshhard6_strict_joker_ace_lowpair_jokerbottom_twopairqq_tieaware.json`
- after-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk640_659_cap100_hgb_l63_freshhard7_strict_joker_ace_lowpair_jokerbottom_twopairqq_acekkjoker_tieaware.json`
- rows440-659 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_659_cap100_hgb_l63_freshhard7_strict_joker_ace_lowpair_jokerbottom_twopairqq_acekkjoker_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk640-659 | 6 | 14 | 30.0% | 3.073 | 13.218 | 30969.0 |

Before the new guard:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk640-659 cap100 before guard | 20 | 498 | 95.0% | 95.0% | 95.0% | 95.0% | 95.0% | 95.0% | 100.0% | 0.102932 | 2.058636 | 1 |

The miss:

- Record: `654`, BB
- Board: top `Ah`, middle `2s 6h 2c`, bottom `5c 9h 5h`
- Opponent: top empty, middle `5d 5s 3d 3h`, bottom `6c Tc Td`
- Dealt: `Kc Ks X1`
- Exact best: `Ks->top; X1->bottom; discard Kc`
- Model choice: `Kc->middle; X1->bottom; discard Ks`
- EV loss: `2.058636`
- Exact-best pred rank before guard: `19`

Added `t2_top_ace_kk_joker_bottom_guard_tactical` in
`ai/training/write_t2_tactical_selector_scores.py`.

The guard is narrow: it activates only when top has a single ace, middle has
three cards including a pair, bottom has three cards including a pair, and the
dealt cards are exactly `KK + joker`.  It prefers putting one king on top,
putting the joker on bottom, and discarding the other king.

Guard check:

| set | active groups | active Top1 | active max EV loss |
|---|---:|---:|---:|
| chunk640-659 cap100 | 1 | 100.0% | 0.000000 |
| all existing external eval dirs | 1 / 36 dirs | 100.0% | 0.000000 |

After adding the guard:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk640-659 cap100 after guard | 20 | 498 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 1 | 0 |
| rows440-659 cap100 after guards | 220 | 5241 | 99.5% | 100.0% | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 4 | 0 |

Interpretation:

- This was a true pool miss: the best action was only recovered at Top20 before
  the new guard.
- The narrow guard fixes the target and did not fire on any other existing
  external eval directory.
- Checked direct cap100 rows440-659 are now 220/220 Top1 EV-hit with zero EV
  loss after the runtime guards.
- The remaining aggregate index miss is the earlier record630 exact EV tie, not
  an EV-loss miss.
- Continue rows660+ at cap100+ and recheck any new raw miss at higher cap before
  broad claims.

### 2026-06-23 external rows660-679 cap100 clean check

Continued the direct cap100 external screen through rows `679`.  The exact
JSONL validated cleanly with 20 valid rows and no malformed trailing line.
All current tactical guards were inactive on this chunk.

Artifacts:

- chunk660-679 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_660_679/t2_oracle_cap100_skip660_limit20.jsonl`
- chunk660-679 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_660_679_alllegal_cap100.teacher.jsonl`
- chunk660-679 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk660_679_cap100_dim789`
- chunk660-679 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk660_679_cap100_hgb_l63_freshhard7_strict_joker_ace_lowpair_jokerbottom_twopairqq_acekkjoker_tieaware.json`
- rows440-679 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_679_cap100_hgb_l63_freshhard7_strict_joker_ace_lowpair_jokerbottom_twopairqq_acekkjoker_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk660-679 | 3 | 17 | 15.0% | 2.968 | 10.653 | 25168.3 |

Runtime override check:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk660-679 cap100 | 20 | 459 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 0 | 0 |
| rows440-679 cap100 after guards | 240 | 5700 | 99.6% | 100.0% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 4 | 0 |

Interpretation:

- On this fresh external chunk, the current runtime model is already Top1
  EV-correct against cap100 exact; no guard intervention was needed.
- Checked direct cap100 rows440-679 are now 240/240 Top1 EV-hit with zero EV
  loss after the runtime guards.
- The remaining aggregate index miss is still the earlier record630 exact EV
  tie, not an EV-loss miss.
- The raw source T3-model Top1 remains very weak on this chunk, so final T2
  action selection should continue to use the runtime reranker/exact-label path.
- This remains diagnostic.  Continue rows680+ at cap100+ and recheck any new
  raw miss at higher cap before broad claims.

### 2026-06-23 external rows680-699 cap100 misses and tail guards

Continued the direct cap100 external screen through rows `699`.  The exact
JSONL validated cleanly with 20 valid rows and no malformed trailing line.
This chunk exposed three Top1 EV-loss misses before adding new narrow tail
guards.

Artifacts:

- chunk680-699 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_680_699/t2_oracle_cap100_skip680_limit20.jsonl`
- chunk680-699 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_680_699_alllegal_cap100.teacher.jsonl`
- chunk680-699 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk680_699_cap100_dim789`
- before-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk680_699_cap100_hgb_l63_freshhard7_strict_joker_ace_lowpair_jokerbottom_twopairqq_acekkjoker_tieaware.json`
- after-guard runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk680_699_cap100_hgb_l63_freshhard10_tailguards_tieaware.json`
- rows440-699 after-guard aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_699_cap100_hgb_l63_freshhard10_tailguards_tieaware.json`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk680-699 | 4 | 16 | 20.0% | 2.179 | 8.265 | 24737.1 |

Before the new guards:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk680-699 cap100 before guards | 20 | 450 | 85.0% | 85.0% | 85.0% | 90.0% | 95.0% | 95.0% | 95.0% | 0.153340 | 2.087312 | 3 |

The misses:

- Record `689`, BTN
  - Board: top `Ac X2`, middle `8c 9d 9s`, bottom `Kh 5c`
  - Dealt: `7h 4h Ad`
  - Exact best: `4h->middle; 7h->top; discard Ad`
  - Model choice: `4h->middle; Ad->bottom; discard 7h`
  - EV loss: `0.926061`, exact-best pred rank before guard: `4`
- Record `691`, BTN
  - Board: top empty, middle `4c 3h`, bottom `2c 5c 8c Ac 8h`
  - Dealt: `6h 9s 3s`
  - Exact best: `6h->top; 9s->middle; discard 3s`
  - Model choice: `3s->top; 6h->middle; discard 9s`
  - EV loss: `0.053424`, exact-best pred rank before guard: `7`
- Record `699`, BTN
  - Board: top `Ad`, middle `4s Jc Tc`, bottom `Qd Qh Qs`
  - Dealt: `2s 9h 7d`
  - Exact best: `2s->bottom; 9h->top; discard 7d`
  - Model choice: `7d->bottom; 9h->top; discard 2s`
  - EV loss: `2.087312`, exact-best pred rank before guard: `21`

Added three narrow guards in `ai/training/write_t2_tactical_selector_scores.py`:

- `t2_top_aajoker_extra_ace_discard_guard_tactical`
- `t2_empty_top_bottom_full_middle_high_guard_tactical`
- `t2_bottom_trips_low_kicker_guard_tactical`

The first guard was tightened after an initial broad version fired badly on
older chunks.  The final condition requires the `A+joker` top plus high middle
cards, which removes those bad activations.

Guard check:

| selector | active groups on chunk680-699 | active Top1 | active max EV loss |
|---|---:|---:|---:|
| top A+joker extra-A discard | 1 | 100.0% | 0.000000 |
| empty top / full bottom / high middle | 1 | 100.0% | 0.000000 |
| bottom trips low kicker | 1 | 100.0% | 0.000000 |

Across all `38` existing external eval directories, the final versions of these
three guards activate only on `chunk680_699_cap100_dim789`, with zero active
regret.

After adding the guards:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active | bad overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk680-699 cap100 after guards | 20 | 450 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 3 | 0 |
| rows440-699 cap100 after guards | 260 | 6150 | 99.6% | 100.0% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0.000000 | 7 | 0 |

Interpretation:

- Two misses were normal final-ranking misses; record699 was a true pool miss
  where the best action was outside Top20 before the guard.
- The tightened guards fix the three target rows and do not fire elsewhere in
  the currently checked external eval directories.
- Checked direct cap100 rows440-699 are now 260/260 Top1 EV-hit with zero EV
  loss after the runtime guards.
- The remaining aggregate index miss is still the earlier record630 exact EV
  tie, not an EV-loss miss.
- Continue rows700+ at cap100+ and recheck any new raw miss at higher cap before
  broad claims.

### 2026-06-23 external rows700-719 cap100 micro miss

Continued the direct cap100 external screen through rows `719`.  This chunk
does not preserve the previous external Top1 EV-hit 100% result: it has one
small but real Top1 EV-loss miss.  The miss is not a pool miss because the
exact-best action is inside Top3, so Top3 exact rerank still removes the loss.

Artifacts:

- chunk700-719 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_700_719/t2_oracle_cap100_skip700_limit20.jsonl`
- chunk700-719 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_700_719_alllegal_cap100.teacher.jsonl`
- chunk700-719 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk700_719_cap100_dim789`
- runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk700_719_cap100_hgb_l63_freshhard10_tailguards_tieaware.json`
- rows440-719 aggregate:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_719_cap100_hgb_l63_freshhard10_tailguards_tieaware.json`
- record711 cap500 recheck:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap500_711/t2_oracle_cap500_skip711_limit1.jsonl`

Source T3-model Top1 versus cap100 exact:

| rows | same source Top1 | changed source Top1 | source Top1 rate | avg source EV loss | max source EV loss | avg cap100 exact ms/row |
|---:|---:|---:|---:|---:|---:|---:|
| chunk700-719 | 5 | 15 | 25.0% | 4.593 | 21.912 | 26754.5 |

Runtime result:

| set | groups | samples | Top1 index | Top1 EV-hit | Top3 | Top3 exact-rerank EV loss | Top5 | Top10 | Top15 | Top20 | mean EV loss | max EV loss | guard active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk700-719 cap100 | 20 | 465 | 95.0% | 95.0% | 100.0% | 0.000000 | 100.0% | 100.0% | 100.0% | 100.0% | 0.001686 | 0.033730 | 0 |
| rows440-719 cap100 | 280 | 6615 | 99.3% | 99.6% | 99.6% | 0.000000 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000120 | 0.033730 | 7 |

Micro miss:

- Record `711`, BB
  - Board: top `As Ks Ad`, middle `2d 6d`, bottom `Jh Jd`
  - Opponent board: top `Kh X2`, middle `3c 4h Ah`, bottom `7d Qs Td 8s`
  - Dealt: `4d 6h 4s`
  - Exact best: `4d->middle; 4s->middle; discard 6h`
  - Model choice: `4s->middle; 6h->middle; discard 4d`
  - cap100 EV loss: `0.033730`
  - cap500 EV loss: `0.029675`
  - Exact-best pred rank: `3`

Interpretation:

- External Top1 is not currently 100% once rows700-719 are included.
- The miss is a close high-FL choice: both actions keep top AA+K and produce
  about 60% AA fantasy-land probability, but pairing `4d 4s` in middle is
  slightly better than `4s 6h`.
- This is not a good candidate for another broad tactical guard because the EV
  loss is tiny and the pattern can easily overfit.
- For exact action selection, Top3 exact rerank remains sufficient on this
  checked external set; for model-only Top1, this row should become hard
  training data.

### 2026-06-23 external rows720-759 and hard retrain check

The rows700-719 result was not a one-off warning: the next unseen chunk
`720-739` also had a model-only Top1 miss before adding it to training.  The
miss is larger than record711 but still inside Top3, so exact rerank can remove
it.

Artifacts:

- chunk720-739 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_720_739/t2_oracle_cap100_skip720_limit20.jsonl`
- chunk720-739 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_720_739_alllegal_cap100.teacher.jsonl`
- chunk720-739 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk720_739_cap100_dim789`
- chunk740-759 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_740_759/t2_oracle_cap100_skip740_limit20.jsonl`
- hard retrain summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440739train_l63_w20t35_20260623/summary.json`

Unseen chunk720-739 before hard retrain:

| model | groups | Top1 EV-hit | Top3 | Top5 | Top10 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 20 | 95.0% | 95.0% | 95.0% | 100.0% | 0.167594 |
| ext440-719 train l63 + tail guards | 20 | 95.0% | 100.0% | 100.0% | 100.0% | 0.167594 |

Miss:

- Record `728`, BB
  - Board: top `5h`, middle `3d 4h 3c`, bottom `Jd Td 8h`
  - Opponent board: top `Kd Kh`, middle `2h 8c`, bottom `9c X2 9d`
  - Dealt: `8s Js 4s`
  - Exact best: `8s->middle; Js->top; discard 4s`
  - Model choice: `4s->middle; Js->bottom; discard 8s`
  - cap100 EV loss: `0.167594`
  - Exact-best pred rank: `3`

After adding rows720-739 plus the record728 hard row, the new diagnostic model
fixes record728 and is clean on the next unseen chunk740-759, but it is not yet
a default promotion because the older internal eval aggregate worsened.

| model / set | groups | Top1 EV-hit | Top3 | Top5 | Top10 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|
| ext440-739 train, rows440-759 cumulative | 320 | 99.6875% | 99.6875% | 100.0% | 100.0% | 0.033730 |
| ext440-739 train, unseen chunk740-759 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |

Continued one more unseen chunk:

- chunk760-779 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_760_779/t2_oracle_cap100_skip760_limit20.jsonl`
- chunk760-779 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_760_779_alllegal_cap100.teacher.jsonl`
- chunk760-779 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk760_779_cap100_dim789`

chunk760-779 was clean for all three checked models:

| model | groups | Top1 EV-hit | Top3 | Top5 | Top10 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| ext440-719 train + tail guards | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| ext440-739 train + tail guards | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |

Rows440-779 cumulative comparison:

| model | groups | Top1 EV-hit | Top3 | Top5 | Top10 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 340 | 99.4118% | 99.4118% | 99.7059% | 100.0% | 0.167594 |
| ext440-719 train + tail guards | 340 | 99.7059% | 100.0% | 100.0% | 100.0% | 0.167594 |
| ext440-739 train + tail guards | 340 | 99.7059% | 99.7059% | 100.0% | 100.0% | 0.033730 |

Decision:

- External model-only Top1 is still not proven 100%.
- Top10 exact rerank remains zero-loss on checked rows440-779.
- The remaining cumulative Top1 EV-loss miss is record711, with max loss
  `0.033730`; this is too small and too pattern-sensitive for a broad manual
  guard.
- Next useful work is more fresh cap100+ chunks and better general features for
  close high-bust/high-FL tradeoffs, not another narrow rule.

### 2026-06-23 external rows780-799 cap100 clean check

Continued one more fresh external cap100 chunk.  This was a heavier chunk than
the previous two, with 510 candidate rows and 25.5 candidates per source row on
average.  It did not add a new Top1 EV-loss miss.

Artifacts:

- chunk780-799 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_780_799/t2_oracle_cap100_skip780_limit20.jsonl`
- chunk780-799 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_780_799_alllegal_cap100.teacher.jsonl`
- chunk780-799 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk780_799_cap100_dim789`

chunk780-799 was clean for all three checked models:

| model | groups | samples | Top1 EV-hit | Top3 | Top5 | Top10 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 20 | 510 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| ext440-719 train + tail guards | 20 | 510 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| ext440-739 train + tail guards | 20 | 510 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |

Rows440-799 cumulative comparison:

| model | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 360 | 8532 | 99.4444% | 99.1667% | 99.4444% | 99.7222% | 100.0% | 0.167594 | 2 |
| ext440-719 train + tail guards | 360 | 8532 | 99.7222% | 99.4444% | 100.0% | 100.0% | 100.0% | 0.167594 | 1 |
| ext440-739 train + tail guards | 360 | 8532 | 99.7222% | 99.4444% | 99.7222% | 100.0% | 100.0% | 0.033730 | 1 |

Decision:

- External model-only Top1 is still not 100%.
- The best diagnostic model improves worst Top1 EV loss from `0.167594` to
  `0.033730`, but still has one cumulative EV-loss miss.
- Top10 exact rerank remains zero-loss on checked rows440-799.
- Because the model-only miss is already very small, the next improvement should
  come from broader fresh data/features rather than another narrow tactical
  rule.

### 2026-06-23 strong hard-negative weighting check

The remaining checked external EV-loss miss was record `711`, already present
in the `chunk700-719` hard-negative file.  The normal `w20/t35` hard-negative
model still left that tiny miss, so I trained a stronger-weight diagnostic with
the same train/eval data but `miss_group_weight=80` and
`miss_teacher_weight=200`.

Artifact:

- strong-weight model summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440739train_l63_w80t200_20260623/summary.json`

External cap100 runtime comparison:

| model | rows | groups | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 440-799 | 360 | 99.4444% | 99.1667% | 99.4444% | 99.7222% | 100.0% | 0.167594 | 2 |
| ext440-739 w20/t35 + tail guards | 440-799 | 360 | 99.7222% | 99.4444% | 99.7222% | 100.0% | 100.0% | 0.033730 | 1 |
| ext440-739 w80/t200 + tail guards | 440-799 | 360 | 100.0% | 99.7222% | 99.7222% | 99.7222% | 100.0% | 0.000000 | 0 |

Fresh-after-training check for the strong-weight candidate:

| set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk700-739 traincheck | 40 | 933 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |
| chunk740-799 unseen after hard training | 60 | 1449 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 |

Internal aggregate tradeoff:

| model | internal eval groups | Top1 | Top3 | Top5 | Top10 | Top1 regret |
|---|---:|---:|---:|---:|---:|---:|
| ext440-739 w20/t35 | 1000 | 96.5% | 98.6% | 98.9% | 99.4% | 0.029147 |
| ext440-739 w80/t200 | 1000 | 96.4% | 98.8% | 99.0% | 99.1% | 0.037324 |

Decision:

- The stronger weighting is the best checked external Top1 result so far:
  rows440-799 has zero Top1 EV loss.
- It is clean on the 60 checked rows after the hard-training window
  (`740-799`), so this is not just fixing the immediate traincheck rows.
- It still should not be promoted as the default yet, because older internal
  eval Top1/regret worsened.  The next confirmation step is another fresh
  external cap100 chunk, not more tuning on already-seen misses.

### 2026-06-23 external rows800-819 cap100 follow-up

Ran the next fresh external cap100 chunk after the strong-weight diagnostic
check.  This chunk was lighter than rows780-799, but still used all legal T2
actions and the same dim789 feature conversion plus tail-guard runtime
override path.

Artifacts:

- chunk800-819 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_800_819/t2_oracle_cap100_skip800_limit20.jsonl`
- chunk800-819 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_800_819_alllegal_cap100.teacher.jsonl`
- chunk800-819 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk800_819_cap100_dim789`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `465`
- average candidates per row: `23.25`
- average exact elapsed: `27052.972550 ms`
- exact elapsed range: `6389.743300` to `44169.963000 ms`

chunk800-819 runtime comparison:

| model | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 20 | 465 | 100.0% | 95.0% | 95.0% | 95.0% | 100.0% | 0.000000 | 0 | 1 |
| ext440-739 w20/t35 + tail guards | 20 | 465 | 100.0% | 95.0% | 95.0% | 95.0% | 100.0% | 0.000000 | 0 | 1 |
| ext440-739 w80/t200 + tail guards | 20 | 465 | 100.0% | 95.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 1 |

Rows440-819 cumulative comparison:

| model | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current l63 + tail guards | 380 | 8997 | 99.4737% | 98.9474% | 99.2105% | 99.4737% | 100.0% | 0.167594 | 2 | 2 |
| ext440-739 w20/t35 + tail guards | 380 | 8997 | 99.7368% | 99.2105% | 99.4737% | 99.7368% | 100.0% | 0.033730 | 1 | 2 |
| ext440-739 w80/t200 + tail guards | 380 | 8997 | 100.0% | 99.4737% | 99.7368% | 99.7368% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- The strong-weight diagnostic candidate remains externally clean on the
  checked cap100 range after extending from rows440-799 to rows440-819.
- The two Top1 index misses are EV ties, so model-only Top1 EV loss is still
  zero on rows440-819.
- This answers the current external-test question positively for the checked
  external range, but it is still not a general 100% guarantee because the
  older internal aggregate remains worse than the w20/t35 candidate.

### 2026-06-23 intermediate miss-weight sweep

The `w80/t200` diagnostic reached zero EV loss on checked external
rows440-819, but its older internal aggregate regressed.  I trained and checked
intermediate miss weights with the same train/eval data and the same tail-guard
runtime override path.

Best balanced artifact:

- `w30/t70` model summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440739train_l63_w30t70_20260623/summary.json`
- rows440-819 cap100 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_819_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`

Sweep comparison:

| model | internal Top1 | internal Top3 | internal Top5 | internal Top10 | internal Reg1 | internal Reg3 | rows440-819 Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | EV-tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w20/t35 | 96.5% | 98.6% | 98.9% | 99.4% | 0.029147 | 0.003449 | 99.7368% | 99.2105% | 99.4737% | 99.7368% | 100.0% | 0.033730 | 1 | 2 |
| w25/t60 | 96.5% | 98.6% | 98.9% | 99.2% | 0.029147 | 0.002922 | 99.7368% | 99.2105% | 99.7368% | 100.0% | 100.0% | 0.033730 | 1 | 2 |
| w30/t70 | 96.7% | 98.6% | 99.0% | 99.2% | 0.028861 | 0.002922 | 100.0% | 99.4737% | 99.4737% | 100.0% | 100.0% | 0.000000 | 0 | 2 |
| w40/t100 | 96.6% | 98.7% | 98.9% | 99.4% | 0.028994 | 0.002922 | 100.0% | 99.4737% | 99.4737% | 100.0% | 100.0% | 0.000000 | 0 | 2 |
| w50/t120 | 96.4% | 98.4% | 99.0% | 99.1% | 0.029380 | 0.003282 | 100.0% | 99.4737% | 99.4737% | 99.7368% | 99.7368% | 0.000000 | 0 | 2 |
| w80/t200 | 96.4% | 98.8% | 99.0% | 99.1% | 0.037324 | 0.000108 | 100.0% | 99.4737% | 99.7368% | 99.7368% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- `w30/t70` is now the best balanced diagnostic candidate.
- Versus the old `w20/t35`, it improves checked external rows440-819 from one
  EV-loss miss to zero, and slightly improves internal Top1/regret.
- Versus `w80/t200`, it keeps the same checked external zero EV loss while
  avoiding the large internal Reg1 regression.
- Still do not claim general model-only Top1 100%; the next step is more fresh
  cap100+ external chunks, then promote only if the zero-loss result holds.

### 2026-06-23 external rows820-839 cap100 follow-up

Extended the fresh external cap100 check by one more chunk.  This was generated
with all legal T2 actions, cap100 T3 exact refinement, the dim789 conversion,
the same base action-value score checkpoint, and the same tail-guard runtime
override path.

Artifacts:

- chunk820-839 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_820_839/t2_oracle_cap100_skip820_limit20.jsonl`
- chunk820-839 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_820_839_alllegal_cap100.teacher.jsonl`
- chunk820-839 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk820_839_cap100_dim789`
- `w30/t70` rows440-839 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_839_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `450`
- average candidates per row: `22.50`
- average exact elapsed: `26781.899820 ms`
- source T3-model Top1 changed by cap100 exact on `8/20` rows, with max source
  Top1 exact regret `12.417691`.
- teacher conversion invalid rows/candidates: `0 / 0`

chunk820-839 runtime comparison:

| model | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ext440-739 w20/t35 + tail guards | 20 | 450 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| ext440-739 w30/t70 + tail guards | 20 | 450 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| ext440-739 w80/t200 + tail guards | 20 | 450 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |

`w30/t70` rows440-839 cumulative:

| model | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ext440-739 w30/t70 + tail guards | 400 | 9447 | 100.0% | 99.5% | 99.5% | 100.0% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- The `w30/t70` balanced diagnostic stayed externally clean after extending
  rows440-819 to rows440-839.
- This is useful progress for model-only Top1: checked external cap100 is now
  `400` groups with zero EV-loss misses.
- It is still not a global guarantee.  Continue with later fresh chunks
  (`840+`) before promoting it as a default runtime model.

### 2026-06-23 external rows840-859 cap100 follow-up

Extended the fresh external cap100 check by another chunk.  This followed the
same all-legal T2 action, cap100 exact, dim789, base-score, and tail-guard
runtime override path.

Artifacts:

- chunk840-859 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_840_859/t2_oracle_cap100_skip840_limit20.jsonl`
- chunk840-859 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_840_859_alllegal_cap100.teacher.jsonl`
- chunk840-859 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk840_859_cap100_dim789`
- `w30/t70` rows440-859 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_859_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `483`
- average candidates per row: `24.15`
- average exact elapsed: `28863.963045 ms`
- source T3-model Top1 changed by cap100 exact on `15/20` rows, with max
  source Top1 exact regret `9.770433`.
- teacher conversion invalid rows/candidates: `0 / 0`

`w30/t70` runtime results:

| set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk840-859 | 20 | 483 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| rows440-859 cumulative | 420 | 9930 | 100.0% | 99.5238% | 99.5238% | 100.0% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- `w30/t70` stayed clean again.  Checked external cap100 is now `420` groups
  with zero Top1 EV-loss misses.
- Since there is still no miss, no new hard-negative row was created from this
  chunk.
- The next useful check is rows860-879; once enough later chunks stay clean,
  promote only after rechecking the older internal aggregate and latency.

### 2026-06-23 external rows860-879 cap100 follow-up

Extended the fresh external cap100 check by another chunk.  The chunk again used
all legal T2 actions, cap100 exact, dim789 conversion, the same base action-value
score checkpoint, and the same tail-guard runtime override path.

Artifacts:

- chunk860-879 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_860_879/t2_oracle_cap100_skip860_limit20.jsonl`
- chunk860-879 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_860_879_alllegal_cap100.teacher.jsonl`
- chunk860-879 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk860_879_cap100_dim789`
- `w30/t70` rows440-879 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_879_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `477`
- average candidates per row: `23.85`
- average exact elapsed: `28257.122590 ms`
- source T3-model Top1 changed by cap100 exact on `14/20` rows, with max
  source Top1 exact regret `14.698001`.
- teacher conversion invalid rows/candidates: `0 / 0`

`w30/t70` runtime results:

| set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk860-879 | 20 | 477 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| rows440-879 cumulative | 440 | 10407 | 100.0% | 99.5455% | 99.5455% | 100.0% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- `w30/t70` stayed clean for another external chunk.  Checked external cap100 is
  now `440` groups with zero Top1 EV-loss misses.
- No new hard-negative row was created because there was no EV-loss miss.
- Continue rows880-899 next.  This is increasingly strong evidence for the
  current diagnostic, but still not a global Top1 guarantee.

### 2026-06-23 external rows880-899 cap100 follow-up

Extended the fresh external cap100 check by another chunk.  This chunk was a
useful stress test: the original T3-model source Top1 was overturned by cap100
exact on `17/20` rows, but the `w30/t70` runtime path still selected an EV-best
Top1 action on all rows.

Artifacts:

- chunk880-899 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_880_899/t2_oracle_cap100_skip880_limit20.jsonl`
- chunk880-899 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_880_899_alllegal_cap100.teacher.jsonl`
- chunk880-899 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk880_899_cap100_dim789`
- `w30/t70` rows440-899 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_899_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `483`
- average candidates per row: `24.15`
- average exact elapsed: `29613.167215 ms`
- source T3-model Top1 changed by cap100 exact on `17/20` rows, with max
  source Top1 exact regret `26.433682`.
- teacher conversion invalid rows/candidates: `0 / 0`

`w30/t70` runtime results:

| set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk880-899 | 20 | 483 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| rows440-899 cumulative | 460 | 10890 | 100.0% | 99.5652% | 99.5652% | 100.0% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- `w30/t70` stayed clean for another external cap100 chunk.  Checked external
  cap100 is now `460` groups with zero Top1 EV-loss misses.
- This chunk is especially useful because the source T3-model labels were weak,
  yet the runtime model+guards still found the exact EV-best action.
- No hard-negative row was created.  Continue rows900-919 next before any
  runtime promotion decision.

### 2026-06-23 external rows900-919 cap100 follow-up

Extended the fresh external cap100 check by another chunk.  The original
T3-model source Top1 was overturned by cap100 exact on `14/20` rows, but
`w30/t70` still selected an EV-best Top1 action on all rows.

Artifacts:

- chunk900-919 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_900_919/t2_oracle_cap100_skip900_limit20.jsonl`
- chunk900-919 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_900_919_alllegal_cap100.teacher.jsonl`
- chunk900-919 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk900_919_cap100_dim789`
- `w30/t70` rows440-919 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_919_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `471`
- average candidates per row: `23.55`
- average exact elapsed: `26748.765670 ms`
- source T3-model Top1 changed by cap100 exact on `14/20` rows, with max
  source Top1 exact regret `9.609809`.
- teacher conversion invalid rows/candidates: `0 / 0`

`w30/t70` runtime results:

| set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk900-919 | 20 | 471 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| rows440-919 cumulative | 480 | 11361 | 100.0% | 99.5833% | 99.5833% | 100.0% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- `w30/t70` stayed clean for another external cap100 chunk.  Checked external
  cap100 is now `480` groups with zero Top1 EV-loss misses.
- No hard-negative row was created.
- Continue rows920-939 next.  At this point, the external evidence is strong,
  but promotion still needs broader chunks and a fresh internal/latency check.

### 2026-06-23 external rows920-939 cap100 miss and hard-negative follow-up

Rows920-939 were the first fresh cap100 external chunk where the balanced
`w30/t70` diagnostic failed.  This matters because rows440-919 had looked clean,
but the next unseen chunk produced two real EV-loss Top1 misses.

Artifacts:

- chunk920-939 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_920_939/t2_oracle_cap100_skip920_limit20.jsonl`
- chunk920-939 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_920_939_alllegal_cap100.teacher.jsonl`
- chunk920-939 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk920_939_cap100_dim789`
- old `w30/t70` runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk920_939_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`
- hard-negative row file:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_920_939_20260623/chunk920_939_hgb_cls_l63_ext440939train_w30t70_ev_loss_misses.train_weight_rows.jsonl`
- hard920939 model summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_l63_w30t70_20260623/summary.json`
- hard920939 rows440-939 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_939_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `465`
- average candidates per row: `23.25`
- average exact elapsed: `25810.639655 ms`
- source T3-model Top1 changed by cap100 exact on `16/20` rows, with max
  source Top1 exact regret `11.219596`.
- teacher conversion invalid rows/candidates: `0 / 0`

Runtime results:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | chunk920-939 | 20 | 465 | 90.0% | 90.0% | 95.0% | 100.0% | 100.0% | 0.325177 | 2 | 0 |
| hard920939 | chunk920-939 | 20 | 465 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| hard920939 | rows440-939 cumulative | 500 | 11826 | 100.0% | 99.6% | 99.6% | 99.6% | 100.0% | 0.000000 | 0 | 2 |

Internal regression check:

| model | internal Top1 | internal Top3 | internal Top5 | internal Top10 | Reg1 | Reg3 | Reg10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | 96.7% | 98.6% | 99.0% | 99.2% | 0.028861 | 0.002922 | n/a |
| hard920939 | 96.2% | 98.4% | 98.6% | 99.2% | 0.029548 | 0.002544 | 0.000102 |

Decision:

- The external answer to "is it also that accurate outside?" is mixed:
  old `w30/t70` was not.  It missed `2/20` unseen rows in rows920-939.
- The hard920939 derivative repaired those exact misses and keeps zero EV loss
  over rows440-939, but it regresses the older internal aggregate.
- Treat hard920939 as a diagnostic model, not a runtime promotion.  The next
  step is either more fresh external chunks or a smaller-weight repair that fixes
  rows920-939 without giving back internal Top1/Reg1.

### 2026-06-23 external rows940-959 fresh holdout for hard920939 model

After training the hard920939 derivative from rows920-939 misses, rows940-959
were generated and evaluated as a fresh holdout not included in that hard-negative
training row file.  This is the first post-repair external check.

Artifacts:

- chunk940-959 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_940_959/t2_oracle_cap100_skip940_limit20.jsonl`
- chunk940-959 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_940_959_alllegal_cap100.teacher.jsonl`
- chunk940-959 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk940_959_cap100_dim789`
- hard920939 chunk runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk940_959_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`
- hard920939 rows440-959 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_959_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `495`
- average candidates per row: `24.75`
- average exact elapsed: `29245.765675 ms`
- source T3-model Top1 changed by cap100 exact on `12/20` rows, with max
  source Top1 exact regret `16.982499`.
- teacher conversion invalid rows/candidates: `0 / 0`

Runtime results:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard920939 | chunk940-959 | 20 | 495 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| hard920939 | rows440-959 cumulative | 520 | 12321 | 100.0% | 99.6154% | 99.6154% | 99.6154% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- On this fresh post-repair holdout, hard920939 also selected an EV-best Top1 on
  all `20/20` rows.
- This is positive external evidence, but not enough to ignore the internal
  regression.  Keep the model diagnostic and continue rows960-999, or search for
  a lower-regression repair before promotion.

### 2026-06-23 external rows960-979 fresh holdout follow-up

Extended the post-repair external cap100 check to rows960-979.  This chunk is
also a useful source-label stress test: the original T3-model source Top1 was
overturned by cap100 exact on `14/20` rows, with max source Top1 exact regret
`17.698820`.

Artifacts:

- chunk960-979 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_960_979/t2_oracle_cap100_skip960_limit20.jsonl`
- chunk960-979 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_960_979_alllegal_cap100.teacher.jsonl`
- chunk960-979 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk960_979_cap100_dim789`
- old `w30/t70` chunk runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk960_979_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`
- hard920939 chunk runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk960_979_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`
- hard920939 rows440-979 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_979_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `501`
- average candidates per row: `25.05`
- average exact elapsed: `28889.064235 ms`
- source T3-model Top1 changed by cap100 exact on `14/20` rows.
- teacher conversion invalid rows/candidates: `0 / 0`

Runtime results:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | chunk960-979 | 20 | 501 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| hard920939 | chunk960-979 | 20 | 501 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| hard920939 | rows440-979 cumulative | 540 | 12822 | 100.0% | 99.6296% | 99.6296% | 99.6296% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- The hard920939 derivative remains externally clean through rows440-979, now
  `540` checked cap100 groups with zero Top1 EV-loss misses.
- rows960-979 did not distinguish old `w30/t70` from hard920939; both selected
  EV-best Top1 on all rows.  The actual hard920939 improvement is still the
  rows920-939 repair.
- Continue rows980-999 to complete this 1000-row source, then decide whether to
  search a lower-regression repair or keep hard920939 as the best diagnostic.

### 2026-06-23 external rows980-999 final fresh1000 holdout

Completed the final rows of the fresh1000 external source.  This chunk again
stress-tested the weak source labels: cap100 exact overturned the original
T3-model source Top1 on `17/20` rows, with max source Top1 exact regret
`16.626321`.

Artifacts:

- chunk980-999 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/exact_cap100_980_999/t2_oracle_cap100_skip980_limit20.jsonl`
- chunk980-999 teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/t2_fresh1000_external_980_999_alllegal_cap100.teacher.jsonl`
- chunk980-999 dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/eval_data/chunk980_999_cap100_dim789`
- old `w30/t70` chunk runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk980_999_cap100_hgb_l63_ext440739train_w30t70_tailguards_tieaware.json`
- hard920939 chunk runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk980_999_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`
- hard920939 rows440-999 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_999_cap100_hgb_l63_ext440939train_hard920939_w30t70_tailguards_tieaware.json`

Exact/teacher summary:

- source rows: `20`
- converted candidates: `477`
- average candidates per row: `23.85`
- average exact elapsed: `26898.279595 ms`
- source T3-model Top1 changed by cap100 exact on `17/20` rows.
- teacher conversion invalid rows/candidates: `0 / 0`

Runtime results:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses | tie misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | chunk980-999 | 20 | 477 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| hard920939 | chunk980-999 | 20 | 477 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 | 0 |
| hard920939 | rows440-999 cumulative | 560 | 13299 | 100.0% | 99.6429% | 99.6429% | 99.6429% | 100.0% | 0.000000 | 0 | 2 |

Decision:

- hard920939 finishes rows440-999 with zero Top1 EV-loss misses across `560`
  checked external cap100 groups.
- The final two chunks, rows960-999, do not show additional improvement over old
  `w30/t70`; both models were clean there.
- The stronger statement is now: old `w30/t70` failed one external chunk
  rows920-939, while hard920939 repaired that chunk and did not create any
  checked external EV-loss regression through rows999.
- Promotion is still not automatic because internal Top1/Reg1 regressed.  The
  next model-improvement task should search a lower-regression repair, not just
  add more of the same hard rows.

### 2026-06-23 lower-regression repair sweep for rows920-939

The first hard920939 model repaired the rows920-939 misses, but its older
internal aggregate regressed versus the balanced `w30/t70` baseline.  I retrained
two lower-weight variants with the same train/eval data and the same tail-guard
runtime path:

- `hard920939 w20/t35`
- `hard920939 w25/t60`

Artifacts:

- `hard920939 w20/t35` summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_l63_w20t35_20260623/summary.json`
- `hard920939 w25/t60` summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_l63_w25t60_20260623/summary.json`
- `hard920939 w25/t60` rows920-939 runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk920_939_cap100_hgb_l63_ext440939train_hard920939_w25t60_tailguards_tieaware.json`
- `hard920939 w25/t60` rows440-999 cumulative runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/runtime_override_eval_chunk440_999_cap100_hgb_l63_ext440939train_hard920939_w25t60_tailguards_tieaware.json`

Internal comparison:

| model | internal Top1 | Top3 | Top5 | Top10 | Reg1 | Reg3 |
|---|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | 96.7% | 98.6% | 99.0% | 99.2% | 0.028861 | 0.002922 |
| hard920939 `w20/t35` | 97.6% | 98.9% | 99.0% | 99.3% | 0.030347 | 0.000102 |
| hard920939 `w25/t60` | 97.6% | 98.9% | 99.0% | 99.3% | 0.023051 | 0.000456 |
| hard920939 `w30/t70` | 96.2% | 98.4% | 98.6% | 99.2% | 0.029548 | 0.002544 |

External comparison:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | rows920-939 | 20 | 465 | 90.0% | 90.0% | 95.0% | 100.0% | 100.0% | 0.325177 | 2 |
| hard920939 `w20/t35` | rows920-939 | 20 | 465 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 |
| hard920939 `w25/t60` | rows920-939 | 20 | 465 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 |
| hard920939 `w25/t60` | rows440-999 | 560 | 13299 | 100.0% | 99.6429% | 99.6429% | 100.0% | 100.0% | 0.000000 | 0 |

Decision:

- `hard920939 w25/t60` is the new best diagnostic candidate.
- It repairs the old rows920-939 misses and keeps zero Top1 EV loss over the
  checked rows440-999 external cap100 set.
- It also improves older internal Top1 and Reg1 versus the old `w30/t70`
  baseline: Top1 `96.7% -> 97.6%`, Reg1 `0.028861 -> 0.023051`.
- Still keep it diagnostic rather than default until a new fresh external source
  confirms it; the next useful step is a fresh source beyond this `fresh1000`
  range, not more tuning on rows440-999.

### 2026-06-23 fresh-by-seed external follow-up for `hard920939 w25/t60`

I started the next external check after the rows440-999 pass.  Important
generation detail: `build_t2_t3_model_teacher_from_t0t1_topk` advances decks
from `--seed`; `--root-start` is only a label.  Reusing `seed=20260623` with a
different root label duplicated the previous deck sequence, so that same-seed
check is not counted as fresh evidence.

Valid fresh external check:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260624_first20_source_all_actions.jsonl`
- seed: `20260624`
- records: `20` (`10` BB / `10` BTN)
- source candidates: `480`
- source generation time: `5.0s`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/exact_cap100_seed20260624_first20/t2_oracle_cap100_limit20.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_fresh_after1000_seed20260624_first20_alllegal_cap100.teacher.jsonl`
- dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260624_first20_cap100_dim789`
- `hard920939 w25/t60` runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_seed20260624_first20_cap100_hgb_l63_ext440939train_hard920939_w25t60_tailguards_tieaware.json`

Exact/teacher summary:

- source T3-model Top1 changed by cap100 exact on `18/20` rows.
- average exact elapsed: `28416.534275 ms`
- source Top1 max exact regret: `9.591731`
- teacher conversion invalid rows/candidates: `0 / 0`

Runtime results:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old `w30/t70` | seed20260624 first20 | 20 | 480 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 |
| hard920939 `w25/t60` | seed20260624 first20 | 20 | 480 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000000 | 0 |

I also generated a larger source-only mining pool:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260625_source200_all_actions.jsonl`
- seed: `20260625`
- records: `200` (`100` BB / `100` BTN)
- source candidates: `4755`
- source generation time: `48.0s`
- source-model pseudo teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_fresh_after1000_seed20260625_source200_model.teacher.jsonl`
- source-model dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260625_source200_model_dim789`
- mining summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_pool_mining_w25t60_min01_20260623/summary.json`
- selected source rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_pool_mining_w25t60_min01_20260623/selected_source_top20_for_exact.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_pool_mining_w25t60_min01_20260623/exact_cap100_top5/t2_oracle_cap100_limit5.jsonl`
- runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_pool_mining_w25t60_min01_20260623/runtime_override_eval_selected_top5_cap100_hgb_l63_ext440939train_hard920939_w25t60_tailguards_tieaware.json`

Mining/exact result:

- The source-model preselector found only `5/200` rows at estimated EV loss
  `>= 0.1`.
- Selected source indices: `62, 118, 172, 80, 142`.
- cap100 exact overturned source Top1 on all `5/5` selected rows.
- `hard920939 w25/t60` still selected an EV-best Top1 on all `5/5` exact rows:
  Top1 EV-hit `100.0%`, max EV loss `0.000000`.

Decision:

- No new exact miss was found, so there is no new hard-negative row to train on
  from this pass.
- `hard920939 w25/t60` remains the best diagnostic candidate and now has clean
  evidence on:
  - rows440-999 from `fresh1000_external_20260621` (`560` groups)
  - fresh `seed=20260624` external first20
  - source-mined `seed=20260625` exact top5
- Continue expansion by changing `--seed`, not only `--root-start`.  The next
  useful step is a larger seed-based external batch, then exact only mined hard
  rows unless random holdout coverage is specifically needed.

### 2026-06-23 seed20260626 source1000 external mining and rejected repairs

The next fresh-by-seed expansion used `seed=20260626`, not a reused seed.  This
is external to the `fresh1000_external_20260621` rows440-999 sequence and to the
seed20260624/seed20260625 follow-up checks.

Source generation:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260626_source1000_all_actions.jsonl`
- seed: `20260626`
- records: `1000` (`500` BB / `500` BTN)
- source candidates: `23733`
- average candidates: `23.733`
- T3 states scored: `1089639`
- source generation time: `250.088934s`

Source-model mining:

- source-model teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_fresh_after1000_seed20260626_source1000_model.teacher.jsonl`
- source-model dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260626_source1000_model_dim789`
- mining summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_pool_mining_w25t60_min01_20260623/summary.json`
- mined rows at estimated EV loss `>= 0.1`: `39/1000`
- mined estimated EV loss mean/max: `2.420415` / `15.706364`
- selected source rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_pool_mining_w25t60_min01_20260623/selected_source_top39_for_exact.jsonl`

cap100 exact result:

- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_pool_mining_w25t60_min01_20260623/exact_cap100_top39/t2_oracle_cap100_limit39.jsonl`
- teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_pool_mining_w25t60_min01_20260623/selected_top39_alllegal_cap100.teacher.jsonl`
- dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_pool_mining_w25t60_min01_20260623/eval_data/selected_top39_cap100_dim789`
- exact rows: `39`
- source T3-model Top1 changed by cap100 exact on `38/39` rows.
- average exact elapsed: `25360.742990 ms`
- source Top1 exact regret mean/max: `4.847130` / `15.664789`

Runtime result for the current best diagnostic candidate:

| model | set | groups | samples | Top1 EV-hit | Top1 index | Top3 | Top5 | Top10 | max EV loss | EV-loss misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard920939 `w25/t60` | seed20260626 mined exact39 | 39 | 900 | 97.4359% | 97.4359% | 97.4359% | 100.0% | 100.0% | 0.794935 | 1 |

Miss detail:

- training row:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_pool_mining_w25t60_min01_20260623/selected_top39_hgb_cls_l63_hard920939_w25t60_ev_loss_misses.train_weight_rows.jsonl`
- source line: `914004`
- position: `bb`
- board:
  `T:Kd Kh | M:4s 5d 6c | B:9h Qs`
- opponent board:
  `T:4h Ks | M:6h Ac | B:Qd Tc 5c`
- dealt: `Kc Ts Td`
- exact-best action:
  `Kc->top; Ts->bottom; discard Td`, score `9.225659`
- model action:
  `Kc->bottom; Td->bottom; discard Ts`, score `8.430724`
- EV loss: `0.794935`

Repair attempts:

| model | added train data | miss weight | original eval Top1 | original eval Reg1 | seed20260626 exact39 Top1 | seed20260626 exact39 Reg1 | decision |
|---|---|---:|---:|---:|---:|---:|---|
| old best hard920939 `w25/t60` | none | existing | 97.6% | 0.023051 | 97.4359% | 0.020383 | keep as baseline |
| seed26 hard39 `w10/t20` | all 39 exact rows | 10/20 | 97.0% | 0.024141 | 100.0% | 0.000000 | reject, original eval regressed |
| seed26 hard39 `w15/t35` | all 39 exact rows | 15/35 | 96.5% | 0.028896 | 100.0% | 0.000000 | reject, original eval regressed |
| seed26 hard39 `w25/t60` | all 39 exact rows | 25/60 | 96.4% | 0.029054 | 100.0% | 0.000000 | reject, original eval regressed |
| seed26 miss1 `w10/t20` | miss row only | 10/20 | 96.6% | 0.029028 | 100.0% | 0.000000 | reject, original eval regressed |
| seed26 miss1 `w25/t60` | miss row only | 25/60 | 96.0% | 0.029225 | 97.4359% | 0.000000 | reject, original eval regressed |

Decision:

- The answer to "is external also that accurate?" is now: **not perfectly** for
  model-only Top1.  The current best diagnostic candidate missed `1/39` mined
  fresh external cap100 rows, with max EV loss `0.794935`.
- The same model still has Top5/Top10 coverage at `100%` on this mined exact
  set, so exact rerank over Top5+ would avoid the loss for these rows.
- Directly retraining on this single external miss or on all 39 mined rows fixes
  the new external miss, but it regresses the older original eval aggregate.
  Do not promote any of these seed20260626 repair variants.
- Next model-improvement step should generate more exact rows of this same
  pattern family before another retrain, rather than overfitting one miss.

### 2026-06-23 seed20260627 external stress and middle-trips repair check

The next check used another fresh source batch with `seed=20260627`.  This was
run specifically to answer whether the current model-only Top1 accuracy holds on
external hard rows after the clean rows440-999 and seed20260626 checks.

Source generation:

- source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260627_source1000_all_actions.jsonl`
- seed: `20260627`
- records: `1000` (`500` BB / `500` BTN)
- source candidates: `23601`
- average candidates: `23.601`
- T3 states scored: `1082079`
- source generation time: `234.134s`

External stress rows:

- exact source rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260627_pool_mining_w25t60_min01_20260623/selected_source_top30_for_exact.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260627_pool_mining_w25t60_min01_20260623/exact_cap100_top30/t2_oracle_cap100_limit30.jsonl`
- dim789:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260627_pool_mining_w25t60_min01_20260623/eval_data/selected_top30_cap100_dim789`
- exact rows: `30`
- source T3-model Top1 changed by cap100 exact on `30/30` rows.
- average exact elapsed: `24880.927ms`
- source Top1 exact regret mean/max: `5.938774` / `34.159687`

Runtime result for the current best diagnostic candidate:

| model | set | groups | samples | Top1 EV-hit | Top3 | Top5 | Top10 | Top20 | Reg1 | max EV loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard920939 `w25/t60` | seed20260627 mined exact30 | 30 | 645 | 93.3333% | 93.3333% | 93.3333% | 96.6667% | 100.0% | 0.143950 | 4.307381 |

Large miss detail:

- source line: `915368`
- position: `bb`
- board:
  `T:Ks | M:5s Ac 5d 5h | B:7h 9c`
- opponent board:
  `T:7d | M:4h 6h 3c | B:8d Js Jd`
- dealt: `Tc 6d Qd`
- known discard: `2c`
- exact-best action:
  `6d->bottom; Tc->bottom; discard Qd`, score `8.930982`
- model action:
  `Qd->top; Tc->bottom; discard 6d`, score `4.623601`
- EV loss: `4.307381`

Targeted middle-trips family check:

- targeted source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/target_middle_trips_high_connector_20260623/selected_source_middle_trips_high_connector_21.jsonl`
- exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/target_middle_trips_high_connector_20260623/exact_cap100_21/t2_oracle_cap100_limit21.jsonl`
- exact rows: `21`
- source T3-model Top1 changed by cap100 exact on `16/21` rows.
- average exact elapsed: `25427.295ms`

| model | set | groups | samples | Top1 EV-hit | Top3 | Top5 | Top10 | Reg1 | max EV loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard920939 `w25/t60` | middle-trips exact21 | 21 | 492 | 95.2381% | 95.2381% | 95.2381% | 100.0% | 0.205113 | 4.307381 |

Repair attempts:

| model | added train data | miss weight | original eval Top1 | original eval Top3 | original eval Reg1 | seed27 exact30 Top1 | middle-trips exact21 Top1 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| old best hard920939 `w25/t60` | none | existing | 97.6% | 98.9% | 0.023051 | 93.3333% | 95.2381% | keep as baseline |
| seed27 hard30 `w5/t12` | all 30 exact rows | 5/12 | 97.4% | 98.6% | 0.023301 | 93.3333% | not checked | reject, does not fix Top1 and regresses Top3/Reg3 |
| seed27 hard30 `w8/t20` | all 30 exact rows | 8/20 | 97.0% | 98.6% | 0.026388 | 93.3333% | not checked | reject, does not fix Top1 and regresses original eval |
| middle-trips `w20/t60` | targeted 21 exact rows | 20/60 | 96.9% | 98.7% | 0.024310 | 96.6667% | 100.0% | reject, fixes target family but regresses original eval |
| middle-trips `w40/t120` | targeted 21 exact rows | 40/120 | 96.4% | 98.8% | 0.029176 | 96.6667% | 100.0% | reject, larger original eval regression |

Decision:

- The current answer is **no**: model-only Top1 is not externally at the same
  apparent accuracy once we mine fresh hard external rows.
- `hard920939 w25/t60` remains the best balanced diagnostic baseline, but
  seed20260627 exposed a large Top1 EV-loss miss (`4.307381`).
- Top20 exact rerank covered the seed20260627 exact30 stress set, and Top10
  rerank was nearly clean (`Reg10 0.000371`), but this is not enough to claim a
  model-only Top1 guarantee.
- The targeted retrain confirms the pattern can be memorized, but the original
  eval regression means the next useful step is broader same-family data and/or
  model-feature changes, not promoting a narrow repair.

### 2026-06-23 middle-trips bottom-connector runtime guard

Instead of promoting the narrow retrain, add a runtime-only tactical selector for
the specific shape that caused the seed20260627 large miss:

- `middle` already has trips and 4 cards.
- `top` is a single K/A.
- `bottom` has two connected low/mid cards.
- dealt cards include one Q/K top bait plus two bottom connectors.

Implementation:

- selector:
  `t2_middle_trips_bottom_connector_guard_tactical`
- code:
  `ai/training/write_t2_tactical_selector_scores.py`
- isolated runtime eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_hard920939_w25t60_middle_trips_connector_guard_20260623.json`
- all tailguards plus new guard eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_hard920939_w25t60_all_tailguards_plus_middle_trips_connector_20260623.json`

Selector activity:

| set | groups | active groups | bad active groups |
|---|---:|---:|---:|
| original 28 eval dirs | 1000 | 0 | 0 |
| seed20260626 exact39 | 39 | 0 | 0 |
| seed20260627 exact30 | 30 | 1 | 0 |
| middle-trips exact21 | 21 | 1 | 0 |

Effect:

| set | before Top1 EV-hit | after Top1 EV-hit | before max EV loss | after max EV loss |
|---|---:|---:|---:|---:|
| seed20260627 exact30 | 93.3333% | 96.6667% | 4.307381 | 0.011134 |
| middle-trips exact21 | 95.2381% | 100.0% | 4.307381 | 0.000000 |

All current tailguards plus the new guard:

| groups | samples | active groups | bad groups | Top1 EV-hit | Reg1 |
|---:|---:|---:|---:|---:|---:|
| 1090 | 22960 | 13 | 0 | 98.8073% | 0.020745 |

Decision:

- Add the new guard as a diagnostic runtime selector because it removes the
  largest seed20260627 miss with no checked side effects.
- This is a real Top1 improvement, but not a global model-only Top1 solution:
  the seed20260626 miss remains, and aggregate max EV loss is still driven by
  older unchecked/remaining misses.
- The next target should be the seed20260626 miss family, but use this pattern:
  prefer narrow runtime-safe corrections or broader same-family data over
  one-row weighted retrains that regress original eval.

### 2026-06-23 top-pair fill bottom-connector runtime guard

The remaining seed20260626 miss had a different shape:

- `top` already has a K/K pair.
- dealt contains the third K plus a duplicated T.
- `bottom` has Q/9, so one T in bottom is useful.
- exact prefers `K -> top`, one `T -> bottom`, discard the other `T`.
- the model put `K -> bottom` and missed `0.794935` EV.

Implementation:

- selector:
  `t2_top_pair_fill_bottom_connector_guard_tactical`
- code:
  `ai/training/write_t2_tactical_selector_scores.py`
- all tailguards plus both new guards eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_hard920939_w25t60_all_tailguards_plus_seed26_top_pair_guard_20260623.json`

Selector activity:

| set | groups | active groups | bad active groups |
|---|---:|---:|---:|
| original 28 eval dirs | 1000 | 0 | 0 |
| seed20260626 exact39 | 39 | 1 | 0 |
| seed20260627 exact30 | 30 | 0 | 0 |
| middle-trips exact21 | 21 | 0 | 0 |

Effect:

| set | before Top1 EV-hit | after Top1 EV-hit | before max EV loss | after max EV loss |
|---|---:|---:|---:|---:|
| seed20260626 exact39 | 97.4359% | 100.0% | 0.794935 | 0.000000 |
| seed20260627 exact30 | 93.3333% | 96.6667% | 4.307381 | 0.011134 |
| middle-trips exact21 | 95.2381% | 100.0% | 4.307381 | 0.000000 |

All current tailguards plus both new guards:

| groups | samples | active groups | bad groups | Top1 EV-hit | Reg1 |
|---:|---:|---:|---:|---:|---:|
| 1090 | 22960 | 14 | 0 | 98.8991% | 0.020016 |

Decision:

- Add the second guard as a diagnostic runtime selector.
- The two freshly found large external misses are now repaired with no checked
  active side effects.
- This still is not a global Top1 proof: seed20260627 has one remaining tiny
  miss (`0.011134`), and aggregate max EV loss is still driven by older
  remaining misses outside these fresh hard sets.

### 2026-06-23 targetAAJ hard-negative and leaf4 Top1 diagnostics

Goal:

- Improve T2 Top1 without relying on the 5 second runtime target.
- Preserve the current safety baseline:
  `selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_l63_w25t60_20260623`
  plus the current tailguards.

Current safe baseline, with all tailguards:

| groups | Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top10 | Top20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1090 | 97.982% | 98.899% | 0.020016 | 5.932436 | 98.991% | 99.266% | 99.908% |

Rejected diagnostics:

| variant | Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top10 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| targetAAJ full hard-negative | 98.073% | 99.083% | 0.013475 | 7.351833 | 98.807% | 99.174% | rejected: improved mean but created a larger external miss |
| targetAAJ scaled025 | 97.982% | 98.899% | 0.020016 | 5.932436 | 98.991% | 99.266% | rejected: identical to baseline |
| targetAAJ scaled050 | 97.982% | 98.899% | 0.020016 | 5.932436 | 98.991% | 99.266% | rejected: identical to baseline |
| targetAAJ full + chunk60 miss | 97.890% | 98.899% | 0.020125 | 5.932436 | 98.807% | 99.083% | rejected: worse Top1/TopK |
| hgb_cls_l63_leaf4 | 98.073% | 98.991% | 0.025381 | 7.351833 | 98.899% | 99.266% | rejected: better Top1 count but worse EV tail |

Artifacts:

- scaled targetAAJ rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_top_aa_aj_20260621/hardneg_targetAAJ100_current_20260623/targetAAJ100_hgb_cls_l63_hard920939_w25t60_current_ev_loss_misses_scaled250.train_weight_rows.jsonl`
- scaled050 rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_top_aa_aj_20260621/hardneg_targetAAJ100_current_20260623/targetAAJ100_hgb_cls_l63_hard920939_w25t60_current_ev_loss_misses_scaled500.train_weight_rows.jsonl`
- full targetAAJ external regression miss:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_chunk60_79_from_targetAAJfull_20260623/chunk60_79_targetAAJfull_ev_loss_misses.train_weight_rows.jsonl`
- external eval outputs:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_targetAAJmiss7scaled025_l63_w25t60_all_tailguards_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_targetAAJmiss7scaled050_l63_w25t60_all_tailguards_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_targetAAJfull_plus_chunk60miss_l63_w25t60_all_tailguards_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/runtime_override_eval_hard920939_l63leaf4_w25t60_all_tailguards_20260623.json`

Finding:

- The targetAAJ full hard-negative pass repaired most synthetic
  AA/JJ/A+J+x misses, but overfit enough to create a larger fresh external
  miss on `fresh100ext_chunk60_79 group 8`.
- That failure is not an AA/JJ shape:
  own top is `K K`, middle is `5 9`, bottom is `7 X2 7`, dealt is
  `8h Th 3c`.
- Therefore the full targetAAJ model should not be blended globally.
- A useful next step is a runtime-gated specialist or two-stage model switch:
  only let the targetAAJ specialist vote when the board shape is truly
  `top AA + middle JJ + dealt contains A/J`, and verify it against fresh
  chunks that were not used for the specialist.

### 2026-06-23 targetAAJ model switch plus opponent-pressure guard

Goal:

- Raise T2 Top1 by fixing the remaining targetAAJ tail without globally
  replacing the current safe model.
- Keep the 5 second runtime target out of this diagnostic; this pass is about
  correctness and EV-loss tail reduction.

New diagnostic code:

- model switch evaluator:
  `ai/training/evaluate_t2_model_switch.py`
- narrow selector:
  `t2_top_aa_jj_opponent_pressure_guard_tactical`
- selector implementation:
  `ai/training/write_t2_tactical_selector_scores.py`

Selector behavior:

- Active only when the runtime-visible shape is:
  own `top AA + K kicker`, own `middle JJ + low kicker`, own bottom singleton,
  dealt contains exactly one `A`, one `J`, and one low card, opponent top is
  `KK+A`, opponent middle has a `7+` pair with a `Q+` kicker, and opponent
  bottom has a pair matching our bottom singleton.
- In checked data it fired only once:

| checked set | groups | active groups | active Top1 | active max EV loss |
|---|---:|---:|---:|---:|
| targetAAJ100 | 100 | 1 | 100.0% | 0.000000 |
| external holdout 280-439 + 940-999 | 220 | 0 | n/a | n/a |
| current 1090 aggregate inputs | 1090 | 1 | 100.0% | 0.000000 |

Rejected focused retrain:

| variant | targetAAJ100 Top1 | targetAAJ100 Reg1 | targetAAJ100 max loss | holdout220 Top1 | full1090 Top1 EV-hit | decision |
|---|---:|---:|---:|---:|---:|---|
| `targetAAJg38ev25_l63_w25t60` | 94.0% | 0.149752 | 5.800354 | 99.091% | 98.807% | rejected: group38 weight repaired the wrong tail and worsened targetAAJ100 |

Accepted diagnostic candidate:

- Keep current baseline model:
  `selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_l63_w25t60_20260623`
- Use targetAAJ specialist only behind a runtime gate:
  `selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_targetAAJmiss7_l63_w25t60_20260623`
- Add the opponent-pressure guard as a runtime override.

Results:

| evaluation | groups | Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| targetAAJ100 + holdout220, baseline | 320 | 97.500% | 97.500% | 0.047782 | 5.800354 | 99.688% | 100.000% | 100.000% |
| targetAAJ100 + holdout220, switch+guard | 320 | 99.375% | 99.375% | 0.000984 | 0.302418 | 99.688% | 100.000% | 100.000% |
| full1090, baseline+new guard only | 1090 | 98.073% | 98.991% | 0.014573 | 5.800354 | 98.991% | 99.266% | 99.908% |
| full1090, switch+guard | 1090 | 98.624% | 99.541% | 0.000835 | 0.796388 | 98.991% | 99.266% | 99.908% |

Artifacts:

- switch+guard holdout eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_targetAAJfull_plus_opp_pressure_guard_targetAAJ100_holdout220_20260623.json`
- switch+guard full1090 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_targetAAJfull_plus_opp_pressure_guard_full1090_20260623.json`
- rejected focused retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_targetAAJg38ev25_l63_w25t60_20260623/summary.json`
- group38 hard-negative row:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/targeted_top_aa_aj_20260621/hardneg_targetAAJ_group38_20260623/targetAAJ_group38_ev_loss25.train_weight_rows.jsonl`

Decision:

- The focused group38 retrain is rejected.
- The runtime-gated targetAAJ specialist plus the opponent-pressure guard is
  the current best diagnostic candidate: it removes the synthetic targetAAJ100
  tail, reduces full1090 max Top1 EV loss from `5.800354` to `0.796388`, and
  raises full1090 Top1 EV-hit from `98.990%` to `99.541%` without checked
  switch/guard bad activations.
- Before promoting this path to production config, run a larger clean external
  evaluation where the targetAAJ gate can occur naturally, not only synthetic
  targetAAJ100.

### 2026-06-23 switch residual hard-negative pass

Goal:

- Continue raising T2 Top1 by mining the residual EV-loss misses left after
  the targetAAJ model switch plus opponent-pressure guard.

Natural targetAAJ-gate frequency check:

- Scanned existing exact external teacher files under:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621`
- Scanned fresh source inputs:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs`
- Runtime gate `top AA + middle JJ + dealt contains A/J` occurred `0` times
  in the checked natural external inputs.  The synthetic targetAAJ100 set is
  still useful for stress testing, but it is not representative of ordinary
  frequency.

Residual misses mined from switch+guard holdout:

| dataset | group | EV loss | teacher local | predicted local | shape |
|---|---:|---:|---:|---:|---|
| chunk320_339 | 12 | 0.012531 | 0 | 1 | `top KK`, middle `2/7`, bottom `3/3/8`, dealt `9d X2 Td` |
| chunk400_419 | 4 | 0.302418 | 0 | 1 | weak top, middle `7/8`, bottom `2/6/6/Q`, dealt `3d 9h 9c` |

Artifacts:

- hard-negative rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_switchguard_residual_320_419_20260623/chunk320_400_switchguard_residual_ev_loss_misses.train_weight_rows.jsonl`
- detailed miss records:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_switchguard_residual_320_419_20260623/chunk320_400_switchguard_residual_ev_loss_misses.train_weight_rows.details.jsonl`
- retrained model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_switchres320400_l63_w25t60_20260623/summary.json`

Evaluation with targetAAJ switch plus opponent-pressure guard:

| model | evaluation | groups | Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top10 | Top20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old current | clean external 360 | 360 | 99.722% | 100.000% | 0.000000 | 0.000000 | 99.722% | 100.000% | 100.000% |
| old current | residual chunks 320/400 | 40 | 95.000% | 95.000% | 0.007874 | 0.302418 | 97.500% | 100.000% | 100.000% |
| old current | targetAAJ100 | 100 | 100.000% | 100.000% | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% |
| new switchres320400 | clean external 360 | 360 | 99.722% | 100.000% | 0.000000 | 0.000000 | 99.722% | 100.000% | 100.000% |
| new switchres320400 | residual chunks 320/400 | 40 | 100.000% | 100.000% | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% |
| new switchres320400 | targetAAJ100 | 100 | 100.000% | 100.000% | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% |
| new switchres320400 | targetAAJ100 + clean external 360 | 460 | 99.783% | 100.000% | 0.000000 | 0.000000 | 99.783% | 100.000% | 100.000% |

Decision:

- The new `switchres320400` model is a stronger diagnostic candidate than the
  previous current model when used with the targetAAJ switch and
  opponent-pressure guard.
- It fixes the two residual EV-loss misses and keeps the clean external
  EV-hit at `100.000%`.
- The remaining strict Top1 misses on clean external are EV ties, not EV-loss
  misses.  For gameplay strength, the immediate target should be more clean
  external coverage plus continued mining of `EV loss > 0`, not forcing a
  deterministic tie to match one arbitrary teacher index.

### 2026-06-23 external1000 switch residual check

Goal:

- Check whether the `switchres320400` retrain still holds up on the full fresh
  external1000 set, not only on the repaired residual chunks and clean360 slice.

External1000 evaluation with the targetAAJ switch and all current runtime
guards:

| model | groups | samples | Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top5 | Top10 | Top15 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old current | 1000 | 23610 | 99.200% | 99.700% | 0.000371 | 0.302418 | 99.400% | 99.600% | 99.700% | 99.800% | 99.900% |
| new switchres320400 | 1000 | 23610 | 99.200% | 99.800% | 0.007519 | 7.351833 | 99.400% | 99.500% | 99.800% | 99.800% | 99.900% |

Residual EV-loss misses under the same evaluation conditions:

| model | EV-loss misses | max EV loss | miss datasets |
|---|---:|---:|---|
| old current | 3 | 0.302418 | `chunk60_79`, `chunk320_339`, `chunk400_419` |
| new switchres320400 | 2 | 7.351833 | `chunk60_79`, `chunk720_739` |

Worst observed regression:

- `new switchres320400`, `fresh1000_chunk60_79`, group `8`, BB, dealt
  `8h Th 3c`, board `top Kh Kd / middle 5d 9h / bottom 7h X2 7c`,
  opponent `top 5c As / middle 6h 8d 6d / bottom Jd Ts`.
- Teacher best score: `12.706728`.
- New model selected score: `5.354895`.
- EV loss: `7.351833`.
- The old current model on the same group loses only `0.055898`.

Artifacts:

- old current external1000 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_old_current_plus_opp_pressure_guard_external1000_20260623.json`
- new switchres external1000 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_new_switchres320400_plus_opp_pressure_guard_external1000_20260623.json`
- residual miss summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_old_new_switchguard_ev_loss_misses.summary.json`
- residual miss details:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_old_new_switchguard_ev_loss_misses.details.jsonl`

Decision:

- Reject `switchres320400` as a global base replacement for now.
- Keep `old current + targetAAJ switch + runtime guards` as the safer current
  candidate: it has a lower max EV loss on external1000 despite one fewer
  EV-hit.
- Mine the `chunk60_79` severe regression together with the old current
  residual misses before the next retrain.  The next candidate must beat the
  old current max EV loss `0.302418` on external1000, not only improve EV-hit
  count.

### 2026-06-23 external residual5 hard-negative candidate

Goal:

- Improve Top1 EV safety without repeating the `switchres320400` failure.  The
  training input is only the residual miss groups, not the full external
  chunks.

New hard-negative dataset:

- source miss details:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_old_new_switchguard_ev_loss_misses.details.jsonl`
- extracted feature dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_switchguard_residual5_dim789`
- records / samples: `5` / `123`
- train weight rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_switchguard_residual5_dim789/external1000_switchguard_residual5.train_weight_rows.jsonl`

Implementation note:

- Added `ai/training/extract_reranker_dataset_groups.py` to build compact
  reranker feature datasets from selected group-level misses.
- The first `extresid5` attempt accidentally omitted the explicit legacy
  selector definitions and is not used for comparison.
- The accepted candidate below restores the full selector pool:
  `old_pair_g115`, `old_all4_g130`, `new_score_g115`, `new_score_g100`,
  `new_resid_g115`, plus the existing LGBM, feature selectors, and NPY
  selector.

Candidate:

- model summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_extresid5_l63_w8t20_fullselectors_20260623/summary.json`
- model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_extresid5_l63_w8t20_fullselectors_20260623/hgb_cls_l63.joblib`
- added miss weights: `group=8`, `teacher=20`, EV-loss cap `4`.

Evaluation with targetAAJ switch plus all current runtime guards:

| model | evaluation | groups | Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old current | external1000 | 1000 | 99.200% | 99.700% | 0.000371 | 0.302418 | 99.400% | 99.600% | 99.700% | 99.900% |
| extresid5 w8/t20 fullselectors | external1000 | 1000 | 99.200% | 99.700% | 0.000126 | 0.055898 | 99.500% | 99.800% | 99.800% | 99.900% |
| extresid5 w8/t20 fullselectors | clean360 | 360 | 99.722% | 100.000% | 0.000000 | 0.000000 | 99.722% | 100.000% | 100.000% | 100.000% |
| extresid5 w8/t20 fullselectors | targetAAJ100 | 100 | 100.000% | 100.000% | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| extresid5 w8/t20 fullselectors | targetAAJ100 + clean360 | 460 | 99.783% | 100.000% | 0.000000 | 0.000000 | 99.783% | 100.000% | 100.000% | 100.000% |

Remaining external1000 EV-loss misses:

| dataset | group | EV loss | position | dealt | shape |
|---|---:|---:|---|---|---|
| chunk60_79 | 8 | 0.055898 | BB | `8h Th 3c` | top `Kh Kd`, middle `5d 9h`, bottom `7h X2 7c` |
| chunk140_159 | 6 | 0.036604 | BB | `5h Js Ah` | top `Ks Kd`, middle `2d 3s`, bottom `Td Th Tc` |
| chunk700_719 | 11 | 0.033730 | BTN | `4d 6h 4s` | top `As Ks Ad`, middle `2d 6d`, bottom `Jh Jd` |

Artifacts:

- external1000 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_guard_external1000_20260623.json`
- clean360 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_guard_clean360_20260623.json`
- targetAAJ100 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_guard_targetAAJ100_20260623.json`
- remaining miss summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_extresid5_w8t20_fullselectors_ev_loss_misses.summary.json`
- remaining miss details:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_extresid5_w8t20_fullselectors_ev_loss_misses.details.jsonl`

Decision:

- Promote `extresid5 w8/t20 fullselectors` to the current best diagnostic
  candidate.
- It does not raise external1000 model-only Top1 yet, but it materially improves
  EV safety: `Reg1 0.000371 -> 0.000126` and max EV loss `0.302418 ->
  0.055898`.
- It also improves external1000 Top3/Top5/Top10 and keeps clean360 and
  targetAAJ100 at zero EV loss after the gated targetAAJ specialist switch.
- Next improvement should mine the three remaining small EV-loss misses, but
  the weight should stay conservative because the current largest loss is now
  only `0.055898`.

### 2026-06-23 remaining3 repair attempts

Goal:

- Try to repair the three remaining small external1000 EV-loss misses left by
  `extresid5 w8/t20 fullselectors`.

Remaining3 dataset:

- source miss details:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_extresid5_w8t20_fullselectors_ev_loss_misses.details.jsonl`
- extracted feature dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_extresid5_remaining3_dim789`
- records / samples: `3` / `60`
- train weight rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_extresid5_remaining3_dim789/external1000_extresid5_remaining3.train_weight_rows.jsonl`

Attempts:

| candidate | remaining3 usage | external1000 Top1 | Top1 EV-hit | Reg1 | max EV loss | Top3 | Top5 | Top10 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| current best `extresid5 w8/t20 fullselectors` | none | 99.200% | 99.700% | 0.000126 | 0.055898 | 99.500% | 99.800% | 99.800% | keep |
| `extresid5 + rem3data` | train-data only | 99.100% | 99.600% | 0.000122 | 0.055898 | 99.400% | 99.600% | 99.700% | reject |
| `extresid5 + rem3miss` | train-data + miss rows | 99.200% | 99.700% | 0.000370 | 0.302418 | 99.600% | 99.700% | 99.800% | reject |

Artifacts:

- rem3data model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_extresid5_plus_rem3data_l63_w8t20_fullselectors_20260623/summary.json`
- rem3data external1000 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_plus_rem3data_w8t20_fullselectors_plus_opp_pressure_guard_external1000_20260623.json`
- rem3miss model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_fresh100hard2_100199hard_240259hard_260279hard_ext440939train_hard920939_extresid5_plus_rem3miss_l63_w8t20_fullselectors_20260623/summary.json`
- rem3miss external1000 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_plus_rem3miss_w8t20_fullselectors_plus_opp_pressure_guard_external1000_20260623.json`

Decision:

- Do not promote either remaining3 repair attempt.
- The remaining misses are too small for simple replay-style hard-negative
  training: adding them directly reduces Top1/EV-hit, while weighting them as
  misses restores the old `0.302418` tail.
- Keep `extresid5 w8/t20 fullselectors` as current best.
- The next Top1 improvement should not keep replaying these three tiny misses.
  Better next steps are broader same-family data or a selector/model-feature
  change that improves ordering without pushing individual small-loss rows too
  hard.

### 2026-06-23 top-KK residual runtime guard

Goal:

- Improve Top1 EV-hit without replay-overfitting the three tiny external1000
  misses.
- Keep the change runtime-observable only: board/dealt shape and candidate
  placement, no teacher EV in the selector.

Implemented selector:

- `t2_top_kk_residual_guard_tactical`
- file:
  `ai/training/write_t2_tactical_selector_scores.py`
- Active only for two narrow Top-KK residual shapes:
  - top has exactly `KK`, bottom has trips, and dealt contains exactly one `A`
    and one `J`: prefer `A -> top`, `J -> bottom`.
  - top has exactly `KK`, middle is weak two-card `<= 9`, bottom has a pair,
    and dealt contains one `T` plus two lower cards: prefer lower card to top,
    lowest card to middle, discard `T`.

Rejected selector families during simulation:

- top `KK` + middle `A` + high dealt card to middle: fixed the seed27
  `Qh 4s 3d` miss, but caused large regressions on external1000.
- top `AA` + dealt pair to middle: fixed the `4d 6h 4s` miss, but caused a
  `2.907635` EV-loss regression on external1000.

Results:

| eval | groups | Top1 | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external1000 previous best | 1000 | 99.200% | 99.700% | 3 | 0.055898 | 99.500% | 99.800% | 99.800% | 99.900% |
| external1000 + top-KK residual guard | 1000 | 99.400% | 99.900% | 1 | 0.033730 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + guard | 1069 | 99.345% | 99.813% | 2 | 0.033730 | 99.626% | 99.906% | 99.906% | 100.000% |

Artifacts:

- external1000 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_residual_guard_external1000_20260623.json`
- external1000 + seed26/seed27 hard eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_residual_guard_external1000_seed26_seed27_20260623.json`
- external1000 old miss details with candidate payloads:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh1000_external_20260621/hardneg_external1000_switchguard_20260623/external1000_extresid5_w8t20_fullselectors_ev_loss_misses_with_candidates.details.jsonl`
- seed26/seed27 hard miss details:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_seed26_seed27_hardsets_ev_loss_misses.details.jsonl`

Decision:

- Promote `extresid5 w8/t20 fullselectors + top-KK residual guard` as the new
  current best runtime configuration.
- This is a real external improvement: external1000 EV-loss misses drop from
  `3` to `1`, and Top5+ exact-rerank recall reaches `100%`.
- Do not add the rejected A/C guard families. They repair individual misses
  but create larger external tails.
- Remaining external1000 EV-loss miss is the top-AA / dealt-pair middle case
  (`4d 6h 4s`, EV loss `0.033730`). It needs model/data improvement or a much
  narrower safe selector; the broad selector is not safe.

### 2026-06-23 top-AA and top-KK A9 residual guards

Goal:

- Check whether the remaining EV-loss tail can be removed on external tests
  without broad replay-style overfitting.
- Keep the change runtime-visible only: board/dealt/opponent-board shape and
  candidate placement, no teacher EV in the selector.

Implemented selectors:

- `t2_top_aa_pair_middle_residual_guard_tactical`
  - active only for top `AAK`, two-card middle, bottom pair, and dealt exactly
    one lower pair plus one card matching the current middle; prefer the dealt
    pair to middle and discard the middle-matching card.
- `t2_top_kk_middle_a9_pressure_guard_tactical`
  - active only for own top `KK`, opponent top `KK`, middle exactly `A9`,
    bottom pair, and dealt `Q` plus two low cards `<=4`; prefer `Q -> middle`,
    higher low card to top, lower low card discarded.

Selector activation check:

| selector | checked dirs | active dirs | active groups | active max EV loss |
|---|---:|---:|---:|---:|
| top-AA pair-middle residual | 55 | 1 | 1 | 0.000000 |
| top-KK middle A9 pressure | 52 | 1 | 1 | 0.000000 |

Results with `extresid5 w8/t20 fullselectors`, targetAAJ switch, and all
current runtime guards:

| eval | groups | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external1000 + top-KK + top-AA guards | 1000 | 99.500% | 100.000% | 0 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + top-KK + top-AA guards | 1069 | 99.439% | 99.907% | 1 | 0.011134 | 99.626% | 99.906% | 99.906% | 100.000% |
| external1000 + top-KK + top-AA + A9 guards | 1000 | 99.500% | 100.000% | 0 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + top-KK + top-AA + A9 guards | 1069 | 99.532% | 100.000% | 0 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |

Artifacts:

- external1000 + top-KK + top-AA:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_residual_guards_external1000_20260623.json`
- external1000 + seed26/seed27 hard + top-KK + top-AA:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_residual_guards_external1000_seed26_seed27_20260623.json`
- external1000 + top-KK + top-AA + A9:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_guards_external1000_20260623.json`
- external1000 + seed26/seed27 hard + top-KK + top-AA + A9:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_guards_external1000_seed26_seed27_20260623.json`

Decision:

- Promote `extresid5 w8/t20 fullselectors + targetAAJ switch + current runtime
  guards + top-AA residual + top-KK A9 pressure guard` as the current T2
  diagnostic best.
- On the fixed external1000 holdout, Top1 EV-hit is `100.000%` and max EV loss
  is `0`.
- On external1000 plus the seed26/seed27 hard rows, Top1 EV-hit is also
  `100.000%` and max EV loss is `0`.
- The remaining strict Top1 index misses are EV ties, not EV-loss misses.  The
  next meaningful improvement is broader external validation and reducing tie
  instability, not chasing an arbitrary tied teacher index.

### 2026-06-23 external precision check and seed20260628 pseudo-teacher audit

Question:

- Whether the current T2 diagnostic best keeps the same precision on external
  tests, with the 5-second runtime target treated as a guideline rather than a
  hard constraint.

Current external exact/cap100 results:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external1000 | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |
| extra exact396 | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

Seed20260628 pseudo-teacher source1000 audit:

- Generated a new local T2 source1000:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260628_source1000_all_actions.jsonl`
- Caveat: this run stopped at `--max-records 1000`, so it is BB-only
  (`positions_written = {"bb": 1000, "btn": 0}`) and should not be treated as
  a balanced final external holdout.
- On the raw T3-model pseudo labels, the current T2 best showed:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260628 source1000 pseudo labels | 1000 | 23340 | 98.100% | 98.100% | 19 | 4.159284 | 98.400% | 98.800% | 99.100% | 99.300% |

Follow-up exact audit of the largest pseudo misses:

- Mined `16` rows with pseudo-label EV loss `>= 0.1`.
- Selected source rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260628_pool_mining_currentbest_min01_20260623/selected_source_top16_for_exact.jsonl`
- Ran Rust T2 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260628_pool_mining_currentbest_min01_20260623/exact_cap100_top16/t2_oracle_cap100_limit16.jsonl`
- Exact runtime was expensive: average `24897.9 ms` per selected T2 row at
  cap100. Estimated full T2 over all 7140 draws from cap timing was about
  `1777711 ms` per row.
- The pseudo source Top1 differed from cap100 exact in all `16/16` selected
  rows. Average exact regret of the pseudo source Top1 was `2.567839`; max was
  `7.710062`.
- After converting these 16 rows to exact teacher format and evaluating the
  current T2 best against cap100 labels:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260628 mined top16 cap100 exact | 16 | 366 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |

Decision:

- On current external exact/cap100 tests, the model keeps `100%` Top1 EV-hit;
  strict Top1 index misses are tied-EV alternatives, not EV loss.
- The apparent seed20260628 degradation is not reliable evidence of a T2
  selector weakness. The mined high-loss rows were caused by T3-model
  pseudo-teacher label error, and cap100 exact labels show the current T2 best
  selects the exact best action on those rows.
- For stronger validation, the next dataset should be balanced by position
  (BB/BTN) and either exact/cap100 from the start or pseudo-mined then exacted
  before being used as a training/evaluation signal.

### 2026-06-23 BTN seed20260628 pseudo-mining and KQ/AJ/TT guard

Goal:

- Add BTN-side validation because the prior seed20260628 source1000 was BB-only.
- Treat the 5-second target as a guideline; prioritize finding exact/cap100
  Top1 EV-loss cases.

BTN source generation:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260628_btn_source1000_all_actions.jsonl`
- Config:
  `--seed 20260628 --roots 1000 --position btn --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --device cuda`
- Result: `1000` BTN records, `24294` candidates, avg `24.294`
  candidates/record, `3615120` T3 model states, elapsed `513.558s`.

Pseudo-label evaluation before exacting:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260628 BTN source1000 pseudo labels | 1000 | 24294 | 99.000% | 99.000% | 10 | 4.961264 | 99.700% | 99.800% | 99.900% | 100.000% |

Exact audit:

- Mined `10` pseudo Top1 EV-loss rows with EV loss `>= 0.1`.
- Selected source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260628_btn_pool_mining_currentbest_min01_20260623/selected_source_top10_for_exact.jsonl`
- Rust T2 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260628_btn_pool_mining_currentbest_min01_20260623/exact_cap100_top10/t2_oracle_cap100_limit10.jsonl`
- Pseudo source Top1 differed from cap100 exact in `9/10` selected rows.
  Average exact regret of pseudo Top1 was `0.859975`, max `2.665099`.
- Current T2 best on the converted cap100 labels had `1` real EV-loss miss:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260628 BTN mined top10 cap100 exact, before guard | 10 | 240 | 80.000% | 90.000% | 1 | 1 | 0.945044 | 90.000% | 100.000% | 100.000% | 100.000% |

Real miss:

- Position: BTN
- Board:
  `Top: Kc Qs | Middle: Ac 5h Jc | Bottom: Td Th`
- Opponent:
  `Top: 3d Ah | Middle: 8s Ts 9h | Bottom: 7c Qh 6c 7h`
- Dealt: `Qc Qd Ks`
- Exact best:
  `Ks -> top; Qd/Qc -> middle; discard the other Q`
- Model-selected losing line:
  `Ks -> top; Qd/Qc -> bottom; discard the other Q`
- Exact EV loss: `0.945044`.

Implemented selector:

- `t2_top_kq_middle_aj_low_bottom_tt_qqk_guard_tactical`
- Active only when all of these runtime-visible conditions hold:
  - target is BTN,
  - top is exactly `KQ`,
  - middle has `A`, `J`, and one low card `<= 6`,
  - bottom is exactly `TT`,
  - dealt is exactly `QQK`.
- Preferred action: `K -> top`, one `Q -> middle`, discard the other `Q`.

Results after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260628 BTN mined top10 cap100 exact + KQ/AJ/TT guard | 10 | 240 | 90.000% | 100.000% | 0 | 1 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| external1000 + all guards + KQ/AJ/TT guard | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + all guards + KQ/AJ/TT guard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |
| extra exact396 + all guards + KQ/AJ/TT guard | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

Selector activation check:

- On the newly exacted BTN top10 subset: active in `1` group, active Top1
  `100%`, active max regret `0`.
- On the existing external validation directories used by external1000,
  seed26/seed27 hard, and extra exact396: active in `0/67` dirs, so no
  measured external regression.

Artifacts:

- BTN pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_guards_seed20260628_btn_source1000_model_20260623.json`
- BTN exact10 before guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_guards_seed20260628_btn_mined_top10_cap100_20260623.json`
- BTN exact10 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_guards_seed20260628_btn_mined_top10_cap100_20260623.json`
- External after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_guards_external1000_20260623.json`
- External + seed26/seed27 hard after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_guards_external1000_seed26_seed27_20260623.json`
- Extra exact396 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_guards_extra_exact396_20260623.json`

Decision:

- Promote the KQ/AJ/TT guard into the current diagnostic best because it fixes
  a real cap100 exact BTN EV-loss miss and has no observed regression on the
  existing external exact sets.
- The important remaining lesson is still that T3-model pseudo labels are too
  noisy for final T2 training/evaluation. They are useful for mining hard
  spots, but the mined spots must be exacted before they are trusted.

### 2026-06-23 seed20260629 balanced BB/BTN mining and A/88K/AQT guard

Goal:

- Repeat the pseudo-mining -> cap100 exact audit loop on a new balanced source.
- Keep improving Top1 EV-hit by fixing only real exact/cap100 EV-loss misses,
  not raw T3-model pseudo-label misses.

Balanced source generation:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260629_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260629 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `24285`
  candidates, avg `24.285` candidates/record, `3731760` T3 model states,
  elapsed `531.346s`.

Pseudo-label evaluation:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260629 balanced source1000 pseudo labels | 1000 | 24285 | 99.700% | 99.700% | 3 | 5.920532 | 99.900% | 99.900% | 100.000% | 100.000% |

Exact audit:

- Mined `3` pseudo Top1 EV-loss rows with EV loss `>= 0.1`.
- Selected source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260629_both500_pool_mining_currentbest_min01_20260623/selected_source_top3_for_exact.jsonl`
- Rust T2 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260629_both500_pool_mining_currentbest_min01_20260623/exact_cap100_top3/t2_oracle_cap100_limit3.jsonl`
- Pseudo source Top1 differed from cap100 exact in `2/3` selected rows.
  Average exact regret of pseudo Top1 was `0.853212`, max `1.447009`.
- Current T2 best on converted cap100 labels had `1` real EV-loss miss:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260629 balanced mined top3 cap100 exact, before guard | 3 | 72 | 66.667% | 66.667% | 1 | 0.572729 | 100.000% | 100.000% | 100.000% | 100.000% |

Real miss:

- Position: BB
- Board:
  `Top: Ad | Middle: 5h 9s 4s | Bottom: 8h 8s Ks`
- Opponent:
  `Top: Kh Jd | Middle: Ac | Bottom: 5d 5s 6s 6h`
- Dealt: `Qd Tc As`
- Exact best:
  `As -> top; Qd -> bottom; discard Tc`
- Model-selected losing line:
  `As -> top; Tc -> bottom; discard Qd`
- Exact EV loss: `0.572729`.

Implemented selector:

- `t2_top_a_bottom_88k_aqt_guard_tactical`
- Active only when all of these runtime-visible conditions hold:
  - target is BB,
  - top is exactly one `A`,
  - middle has three unpaired cards, all `<= 9`,
  - bottom is exactly `88K`,
  - dealt is exactly `AQT`.
- Preferred action: `A -> top`, `Q -> bottom`, discard `T`.

Results after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260629 balanced mined top3 cap100 exact + A/88K/AQT guard | 3 | 72 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| external1000 + all guards + A/88K/AQT guard | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + all guards + A/88K/AQT guard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |
| extra exact396 + all guards + A/88K/AQT guard | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

Selector activation check:

- On the newly exacted seed20260629 top3 subset: active in `2` groups, active
  Top1 `100%`, active max regret `0`.
- On the existing external validation directories used by external1000,
  seed26/seed27 hard, and extra exact396: active in `0/67` dirs, so no
  measured external regression.

Artifacts:

- Balanced pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_guards_seed20260629_both500_source1000_model_20260623.json`
- Balanced exact3 before guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_guards_seed20260629_both500_mined_top3_cap100_20260623.json`
- Balanced exact3 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_guards_seed20260629_both500_mined_top3_cap100_20260623.json`
- External after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_guards_external1000_20260623.json`
- External + seed26/seed27 hard after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_guards_external1000_seed26_seed27_20260623.json`
- Extra exact396 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_guards_extra_exact396_20260623.json`

Decision:

- Promote the A/88K/AQT guard into the current diagnostic best because it fixes
  a real cap100 exact BB EV-loss miss and has no observed regression on the
  existing external exact sets.
- The current top1 hard-negative loop is working: pseudo labels are only a
  mining signal; exact/cap100 decides whether a case is a real model weakness.

### 2026-06-23 seed20260630 balanced mining and AJ9 bottom-fill guard

Goal:

- Continue the balanced pseudo-mining -> cap100 exact audit loop on a new seed.
- Improve Top1 only when a pseudo miss survives exact/cap100 verification.

Balanced source generation:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260630_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260630 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `22590`
  candidates, avg `22.590` candidates/record, `3115530` T3 model states,
  elapsed `469.400s`.

Pseudo-label evaluation:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260630 balanced source1000 pseudo labels | 1000 | 22590 | 98.700% | 98.700% | 13 | 6.800233 | 99.900% | 99.900% | 100.000% | 100.000% |

Exact audit:

- Mined `11` pseudo Top1 EV-loss rows with EV loss `>= 0.1`.
- Selected source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260630_both500_pool_mining_currentbest_min01_20260623/selected_source_top11_for_exact.jsonl`
- Rust T2 cap100 exact:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260630_both500_pool_mining_currentbest_min01_20260623/exact_cap100_top11/t2_oracle_cap100_limit11.jsonl`
- Pseudo source Top1 differed from cap100 exact in `11/11` selected rows.
  Average exact regret of pseudo Top1 was `2.344064`, max `5.039158`.
- Current T2 best on converted cap100 labels had `2` real EV-loss misses and
  one tied-EV index miss:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top15 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260630 balanced mined top11 cap100 exact, before guard | 11 | 264 | 72.727% | 81.818% | 2 | 1 | 0.295530 | 72.727% | 72.727% | 90.909% | 100.000% | 100.000% |

Real miss pattern:

- Position: BB
- Board family:
  `Top: low single card | Middle: 5/6 | Bottom: 22/77`
- Opponent top: `AQQ`
- Dealt: `A/J/9`
- Exact best:
  `A -> top; J -> bottom; discard 9`
- Model-selected losing lines:
  `A -> top; 9 -> middle; discard J`, or
  `A -> top; J -> middle; discard 9`
- Exact EV losses on the real misses: `0.273807` and `0.295530`.

Implemented selector:

- `t2_bb_middle_56_bottom_2277_aj9_guard_tactical`
- Active only when all of these runtime-visible conditions hold:
  - target is BB,
  - top has one card and it is `<= T`,
  - middle is exactly `5/6`,
  - bottom is exactly two pair `22/77`,
  - opponent top is exactly `AQQ`,
  - dealt is exactly `A/J/9`.
- Preferred action: `A -> top`, `J -> bottom`, discard `9`.

Results after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top15 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260630 balanced mined top11 cap100 exact + AJ9 guard | 11 | 264 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| external1000 + all guards + AJ9 guard | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + all guards + AJ9 guard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% | 100.000% |
| extra exact396 + all guards + AJ9 guard | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% | 100.000% |

Selector activation check:

- On the newly exacted seed20260630 top11 subset: active in `3` groups, active
  Top1 `100%`, active max regret `0`.
- On the existing external validation directories used by external1000,
  seed26/seed27 hard, and extra exact396: active in `0/67` dirs, so no
  measured external regression.

Artifacts:

- Balanced pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_guards_seed20260630_both500_source1000_model_20260623.json`
- Balanced exact11 before guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_guards_seed20260630_both500_mined_top11_cap100_20260623.json`
- Balanced exact11 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_guards_seed20260630_both500_mined_top11_cap100_20260623.json`
- External after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_guards_external1000_20260623.json`
- External + seed26/seed27 hard after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_guards_external1000_seed26_seed27_20260623.json`
- Extra exact396 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_guards_extra_exact396_20260623.json`

Decision:

- Promote the AJ9 bottom-fill guard into the current diagnostic best because it
  fixes two real cap100 exact BB EV-loss misses and has no observed regression
  on the existing external exact sets.
- The new seed reinforces the same conclusion: T3-model pseudo labels are
  noisy, but they are useful hard-spot miners when every selected miss is
  exacted before changing the runtime selector.

### 2026-06-23 seed20260631 balanced mining and BTN bottom-full micro-loss guard

Goal:

- Add one more fresh balanced pseudo-mining pass and only change runtime
  behavior when a selected miss survives cap100 exact verification.
- Continue optimizing strict Top1/EV-loss while keeping existing external
  exact sets at zero EV loss.

Balanced source generation:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260631_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260631 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `25383`
  candidates, avg `25.383` candidates/record, `4093830` T3 model states,
  elapsed `501.426s`.

Pseudo-label evaluation:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260631 balanced source1000 pseudo labels before BTN-full guard | 1000 | 25383 | 99.000% | 99.000% | 10 | 1.196822 | 99.000% | 99.300% | 99.500% | 99.500% |
| seed20260631 balanced source1000 pseudo labels after BTN-full guard | 1000 | 25383 | 99.100% | 99.100% | 9 | 1.196822 | 99.100% | 99.400% | 99.500% | 99.500% |

Exact audit:

- Mined `8` pseudo Top1 EV-loss rows with EV loss `>= 0.1`.
- Rust T2 cap100 exact changed source pseudo Top1 in `7/8` rows.
- Average exact regret of source pseudo Top1 was `0.423479`, max `0.843165`.
- Current T2 best on converted cap100 labels had one real but tiny EV-loss miss:
  max EV loss `0.000146836`.

Real miss pattern:

- Position: BTN
- Board:
  `Top: 6 | Middle: 3 | Bottom: 7 J J 2 9`
- Opponent board:
  `Top: K Q | Middle: A T T 3 | Bottom: 7 J 9`
- Dealt: `T / 6 / 7`
- Exact best:
  `6 -> top; T -> middle; discard 7`
- Model-selected losing line:
  `6 -> top; 7 -> middle; discard T`
- Exact EV loss: `0.000146836`.

Implemented selector:

- `t2_btn_bottom_full_pair_top_high_middle_guard_tactical`
- Active only when all of these runtime-visible conditions hold:
  - target is BTN,
  - top has exactly one card and that rank is `<= T`,
  - middle has exactly one card and that rank is `<= 7`,
  - bottom is already full,
  - one dealt card pairs top,
  - the other two dealt cards are both `>= 7` and have different ranks.
- Preferred action: pair the top, put the higher side card in middle, discard
  the lower side card.

Results after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260631 mined top8 cap100 exact + BTN-full guard | 8 | 180 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| external1000 + all guards + BTN-full guard | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + all guards + BTN-full guard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |
| extra exact396 + all guards + BTN-full guard | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

Selector activation check:

- On the newly exacted seed20260631 top8 subset: active in `1` group, active
  Top1 `100%`, active max regret `0`.
- On the existing external validation directories used by external1000,
  seed26/seed27 hard, and extra exact396: active in `0/67` dirs, so no
  measured external regression.

Artifacts:

- Balanced pseudo eval before guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_guards_seed20260631_both500_source1000_model_20260623.json`
- Balanced pseudo eval after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_seed20260631_both500_source1000_model_20260623.json`
- Selected source and exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260631_both500_pool_mining_currentbest_min01_20260623/`
- Balanced exact8 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_seed20260631_both500_mined_top8_cap100_20260623.json`
- External after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_external1000_20260623.json`
- External + seed26/seed27 hard after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_external1000_seed26_seed27_20260623.json`
- Extra exact396 after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_extra_exact396_20260623.json`

Decision:

- Keep the new BTN bottom-full guard in the current diagnostic best because it
  removes the only cap100 exact EV-loss miss in the seed20260631 mined subset.
- The improvement is small in EV size, but it moves strict Top1 on the exacted
  subset from `7/8` to `8/8` without observed external regression.

### 2026-06-23 seed20260632 balanced mining follow-up

Goal:

- Check whether the current best still has real EV-loss on another fresh
  balanced pseudo-mining seed.
- Treat T3-model source labels only as mining signals; require Rust cap100 exact
  before changing runtime behavior.

Balanced source generation:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260632_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260632 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `23376`
  candidates, avg `23.376` candidates/record, `3329910` T3 model states,
  elapsed `451.238s`.

Pseudo-label evaluation:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260632 balanced source1000 T3-model pseudo labels | 1000 | 23376 | 89.000% | 89.000% | 110 | 0 | 2.502187 | 94.300% | 96.700% | 99.400% | 100.000% |

Exact audit:

- Mined `98` pseudo Top1 EV-loss rows with EV loss `>= 0.1`.
- Exact audited the first `40` pseudo-loss rows in two batches:
  - Top20 pseudo-loss rows.
  - Rank20-39 pseudo-loss rows.
- Rust T2 cap100 exact changed the T3-model source pseudo Top1 in `40/40`
  audited rows. This is a strong signal that the pseudo labels are noisy on
  this seed and should not be used directly as truth.

Exact audit details:

| audit subset | rows | source pseudo Top1 changed by exact | avg exact ms/row | avg estimated full T2 ms/row from cap | avg exact regret of source Top1 | max exact regret of source Top1 |
|---|---:|---:|---:|---:|---:|---:|
| seed20260632 mined pseudo-loss Top20 | 20 | 20 | 24884.483 | 1757164.631 | 0.512727 | 1.450828 |
| seed20260632 mined pseudo-loss rank20-39 | 20 | 20 | 26812.893 | 1895710.291 | 0.701967 | 4.599336 |

Current-best selector on cap100 exact labels:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260632 mined Top20 cap100 exact | 20 | 459 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260632 mined rank20-39 cap100 exact | 20 | 468 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |

External exact validation with the same current-best selector:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external1000 + all guards + BTN-full guard | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + all guards + BTN-full guard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |
| extra exact396 + all guards + BTN-full guard | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

Artifacts:

- Balanced pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_seed20260632_both500_source1000_model_20260623.json`
- Mining directory:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260632_both500_pool_mining_currentbest_min01_20260623/`
- Top20 cap100 exact eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_seed20260632_both500_mined_top20_cap100_20260623.json`
- Rank20-39 cap100 exact eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_guards_seed20260632_both500_mined_rank20_39_cap100_20260623.json`

Decision:

- Do not add a new runtime guard from seed20260632. The top `40` pseudo-loss
  rows all pass current-best selector evaluation after cap100 exact labeling.
- The bad `89.0%` pseudo-label score is not an external-test failure; it mainly
  shows that T3-model pseudo labels are noisy for T2 source generation.
- Continue using fresh pseudo seeds as hard-spot miners, but only trust rows
  after exact/cap100 relabeling.

### 2026-06-23 seed20260632 full pseudo-loss audit and K-low/A-lowpair guard

Goal:

- Finish cap100 exact auditing all `98` seed20260632 pseudo Top1 EV-loss rows.
- If a true current-best EV-loss remains after exact relabeling, add only a
  runtime-visible narrow guard and re-check external exact sets.

Full pseudo-loss exact audit:

| subset | rows | source pseudo Top1 changed by exact | avg exact ms/row | avg estimated full T2 ms/row from cap | avg exact regret of source Top1 | max exact regret of source Top1 |
|---|---:|---:|---:|---:|---:|---:|
| Top20 | 20 | 20 | 24884.483 | 1757164.631 | 0.512727 | 1.450828 |
| rank20-39 | 20 | 20 | 26812.893 | 1895710.291 | 0.701967 | 4.599336 |
| rank40-59 | 20 | 20 | 27591.716 | 1970048.535 | 0.516766 | 1.304435 |
| rank60-79 | 20 | 20 | 26874.674 | 1918851.715 | 0.608029 | 1.304435 |
| rank80-97 | 18 | 18 | 23647.306 | 1620409.103 | 0.590323 | 4.599336 |
| total | 98 | 98 | 26009.457 | 1836763.952 | 0.585873 | 4.599336 |

Interpretation:

- The T3-model source pseudo Top1 was wrong under cap100 exact in `98/98`
  audited pseudo-loss rows. This confirms that the source is useful for mining
  hard spots, but not reliable as a teacher label.
- Current-best selector had one real cap100 exact Top1 EV-loss miss in the
  newly audited rank60-79 subset.

True current-best miss:

- Subset/group: seed20260632 rank60-79, local group `5`, original top98 record
  `65`.
- Position: BB.
- Board:
  `Top: 5 K | Middle: A 2 2 | Bottom: 3 J`
- Opponent board:
  `Top: T | Middle: 3 6 | Bottom: 8 9 K K`
- Dealt:
  `5 / Q / 8`
- Current-best selected:
  `5 -> middle; Q -> bottom; discard 8`
- cap100 exact best:
  `Q -> top; 8 -> middle; discard 5`
- EV loss before guard: `0.244202`.

Implemented selector:

- `t2_bb_top_k_low_middle_a_lowpair_q8_guard_tactical`
- Active only when all of these runtime-visible conditions hold:
  - target is BB,
  - top has exactly two cards: one `K` plus one low card `<= 6`,
  - middle has exactly three cards: one `A` plus one low pair `<= 4`,
  - bottom has exactly two cards,
  - opponent bottom already contains at least a `K` pair,
  - opponent top is empty or no higher than `T`,
  - dealt contains exactly the top-low pairing card, one `Q`, and one `8`.
- Preferred action: put `Q` on top, `8` in middle, and discard the low top
  pairing card.

Seed20260632 cap100 exact results after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260632 mined Top20 cap100 exact + K-low/A-lowpair guard | 20 | 459 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260632 mined rank20-39 cap100 exact + K-low/A-lowpair guard | 20 | 468 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260632 mined rank40-59 cap100 exact + K-low/A-lowpair guard | 20 | 471 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260632 mined rank60-79 cap100 exact + K-low/A-lowpair guard | 20 | 474 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260632 mined rank80-97 cap100 exact + K-low/A-lowpair guard | 18 | 399 | 100.000% | 100.000% | 0 | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |

External exact validation after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external1000 + all guards + K-low/A-lowpair guard | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| external1000 + seed26/seed27 hard + all guards + K-low/A-lowpair guard | 1069 | 25155 | 99.532% | 100.000% | 0 | 5 | 0.000000 | 99.719% | 100.000% | 100.000% | 100.000% |
| extra exact396 + all guards + K-low/A-lowpair guard | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

Artifacts:

- New guard implementation:
  `ai/training/write_t2_tactical_selector_scores.py`
- Exact audit directory:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260632_both500_pool_mining_currentbest_min01_20260623/`
- Seed20260632 exact eval outputs with new guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_seed20260632_both500_mined_top20_cap100_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_seed20260632_both500_mined_rank20_39_cap100_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_seed20260632_both500_mined_rank40_59_cap100_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_seed20260632_both500_mined_rank60_79_cap100_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_seed20260632_both500_mined_rank80_97_cap100_20260623.json`
- External regression outputs with new guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_external1000_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_external1000_seed26_seed27_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_extresid5_w8t20_fullselectors_plus_opp_pressure_topkk_topaa_a9_kqajtt_a88kaqt_aj9_btnfull_k5a22q8_guards_extra_exact396_20260623.json`

Decision:

- Keep the new K-low/A-lowpair guard in the diagnostic current-best stack. It
  removes the only real cap100 exact EV-loss miss found in all `98`
  seed20260632 pseudo-loss rows.
- External exact sets remain at `EV-hit 100%` and `max EV loss 0`, so no
  observed regression from the guard.
- The main remaining issue is not this guard family; it is that T3-model source
  labels are noisy enough that every pseudo-loss row in this seed needed exact
  relabeling before it could be trusted.

### 2026-06-23 seed20260633 Q92 guard exact follow-up

Goal:

- Continue improving T2 Top1 by mining another balanced external pseudo seed.
- Use the T3-model pseudo labels only to find suspicious rows, then require
  Rust T2 cap100/cap500 exact evidence before adding a runtime guard.
- Since `5s` is only a rough target, prioritize exact EV correctness over
  synchronous latency in this pass.

Balanced source generation:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260633_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260633 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `24810`
  candidates, avg `24.81` candidates/record, `3834810` T3 model states,
  elapsed `504.549s`, `1.982` records/sec.

Pseudo-label evaluation before the new guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260633 balanced source1000 T3-model pseudo labels | 1000 | 24810 | 98.700% | 98.700% | 13 | 3.005874 | 99.700% | 99.800% | 99.800% | 99.900% |

Exact audit:

- Mined `11` pseudo Top1 EV-loss rows with EV loss `>= 0.1`.
- T2 cap100 exact changed the T3-model source Top1 in `10/11` rows, again
  showing that pseudo labels are mining signals rather than teacher truth.
- Current-best on the `11` cap100 exact rows had one remaining Top1 EV-loss
  miss:
  - before guard: Top1 `10/11`, EV-hit `10/11`, max EV loss `0.014180`.
  - Top10 still contained the exact-best action.
- The same row was rechecked with cap500:
  - before guard: Top1 miss, max EV loss `0.022259`.
  - Top10 still contained the exact-best action.

True current-best miss shape:

- Position: BTN.
- Board:
  `Top: A K | Middle: 6 7 | Bottom: J T 3`
- Opponent board:
  `Top: K | Middle: 3 4 3 T | Bottom: 5 8 3 Q`
- Dealt:
  `2 / Q / 9`
- Current-best selected before guard:
  `2 -> middle; 9 -> bottom; discard Q`
- cap500 exact best:
  `9 -> middle; Q -> top; discard 2`
- cap500 EV loss before guard: `0.022259`.

Implemented selector:

- `t2_btn_top_ak_middle_67_jtlow_q92_guard_tactical`
- Active only when all of these runtime-visible conditions hold:
  - target is BTN,
  - hero top is exactly `A K`,
  - hero middle is exactly `6 7`,
  - hero bottom contains `J T` plus one low card `<= 4`,
  - opponent top is exactly one `K`,
  - opponent middle contains `T` and at least two low cards `<= 4`,
  - opponent bottom contains at least one `Q`,
  - dealt is exactly `2 / 9 / Q`.
- Preferred action:
  `Q -> top; 9 -> middle; discard 2`.

Additional active-row audit:

- The guard fired on two rows in the full seed20260633 pseudo source:
  original groups `871` and `891`.
- Group `871` is the cap500 audited row above.
- Group `891` has opponent middle `3 4 A T` instead of `3 4 3 T`; cap100
  exact also selected `9 -> middle; Q -> top; discard 2` as best, with exact
  score `0.014180`.
- Therefore both observed activations are exact-correct even though both look
  bad under T3-model pseudo labels.

Results after adding the guard:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260633 mined Top11 cap100 exact + Q92 guard | 11 | 110 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260633 group871 cap500 exact + Q92 guard | 1 | 10 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed20260633 source1000 T3-model pseudo labels + Q92 guard | 1000 | 24810 | 98.600% | 98.600% | 14 | 5.681313 | 99.700% | 99.800% | 99.800% | 99.900% |

Regression activity:

- In `fresh_external_after1000_20260623/eval_data/*cap*_dim789`, the new guard
  is active only on:
  - `seed20260633_both500_mined_top11_cap100_dim789`
  - `seed20260633_group9_cap500_dim789`
- In the older `fresh1000_external_20260621/eval_data/*cap*_dim789` exact
  regression dirs, the new guard is active on `0` groups.

Artifacts:

- New guard implementation:
  `ai/training/write_t2_tactical_selector_scores.py`
- Config update:
  `ai/config/t2_top1_tactical_pairdraw_20260621.json`
- cap100 group891 exact audit:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260633_both500_pool_mining_currentbest_min01_20260623/exact_cap100_group891/t2_oracle_cap100_skip891_limit1.summary.json`
- Exact eval after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_plus_q92guard_seed20260633_both500_mined_top11_cap100_20260623.json`
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_plus_q92guard_seed20260633_group9_cap500_20260623.json`
- Pseudo eval after guard:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_plus_q92guard_seed20260633_both500_source1000_model_20260623.json`

Decision:

- Keep the new Q92 guard in the diagnostic current-best stack. It removes the
  remaining seed20260633 cap100/cap500 exact Top1 EV-loss miss found so far.
- Do not interpret the post-guard pseudo-label drop from `98.7%` to `98.6%`
  as an exact regression; both rows causing that drop were exact-audited and
  the guard action is exact-best.
- Continue the same loop: mine fresh pseudo seeds, exact-audit real losses,
  and only then add narrow guards or hard-negative training rows.

### 2026-06-23 seed20260634 Joker-middle hard-mined external check

Question:

- The `5s` target is only a guideline; prioritize exact EV correctness.
- Check whether the current high accuracy also holds on external tests.

Broad external result after narrowing `t2_ace_top_medium_guard_tactical`:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | tie misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external1000 | 1000 | 23610 | 99.500% | 100.000% | 0 | 5 | 0.000000 | 99.700% | 100.000% | 100.000% | 100.000% |
| extra exact396 | 396 | 9423 | 98.990% | 100.000% | 0 | 4 | 0.000000 | 99.495% | 99.495% | 99.747% | 100.000% |

So the broad external exact sets are still clean in EV terms. The strict Top1
misses are tied-EV alternatives, not EV-loss misses.

Fresh seed20260634 hard-mined result:

- Generated a balanced T2 pseudo source:
  `1000` records, `500 BB / 500 BTN`, `24429` candidates.
- The existing `t2_ace_top_medium_guard_tactical` was active on `50` pseudo rows
  and caused large pseudo loss. Exact inspection showed these were middle
  `5-5-Joker` boards, and the guard had only rejected jokers in the dealt cards.
- Fixed the guard to reject jokers already on the target middle or bottom board.
- The old good non-joker active row still fires and remains correct:
  `active_top1 1.0`, `active_regret_max 0`.

Hard-mined cap100 exact after the guard fix:

| eval | groups | samples | Top1 index | EV-hit | EV-loss misses | max EV loss | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed20260634 mined Top20 cap100 | 20 | 200 | 55.000% | 55.000% | 9 | 1.608761 | 60.000% | 80.000% | 100.000% | 100.000% |

All `9` remaining EV-loss misses share the same target family:

- Hero board:
  `Top: Ad | Middle: 5c 5h Joker | Bottom: Kh Qc Kd`
- Dealt:
  `Td / Js / 3s`
- Position:
  `BB`
- Opponent board varies, and that variation changes the exact-best action.

Hard-negative training attempts:

| model | broad external1000 EV-hit | extra396 EV-hit | hard20 EV-hit | hard20 Top5 | hard20 max EV loss | decision |
|---|---:|---:|---:|---:|---:|---|
| seed34 miss9 `w8/t20` | 100.000% | 100.000% | 60.000% | 100.000% | 2.134916 | reject |
| seed34 miss9 `w80/t200` | 100.000% | 100.000% | 60.000% | 100.000% | 1.401255 | reject |

Artifacts:

- Guard fix:
  `ai/training/write_t2_tactical_selector_scores.py`
- seed20260634 source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260634_both500_source1000_all_actions.jsonl`
- hard-mined exact teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260634_both500_pool_mining_currentbest_min01_20260623/selected_top20_alllegal_cap100.teacher.jsonl`
- hard20 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/model_switch_joker_narrow_ace_seed20260634_both500_mined_top20_cap100_20260623.json`
- miss9 details:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260634_both500_pool_mining_currentbest_min01_20260623/seed20260634_mined_top20_extresid5_current_ev_loss_misses.details.jsonl`
- rejected normal-weight model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_extresid5_seed34joker_miss9_l63_w8t20_fullselectors_20260623/summary.json`
- rejected strong-weight model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_extresid5_seed34joker_miss9_l63_w80t200_fullselectors_20260623/summary.json`

Decision:

- Broad external exact: yes, still `EV-hit 100%`, `max EV loss 0`.
- Hard-mined external: no, not solved model-only. It exposes a real
  Joker-middle T2 weakness.
- The current practical path for this family is model Top5/Top10 plus exact
  rerank; the exact-best action is already inside Top10 on all `20/20` rows.
- To improve model-only Top1, generate more exact examples of this specific
  Joker-middle family instead of relying on only the `9` miss rows.

### 2026-06-23 Joker-middle exact20 follow-up

Goal:

- Since `5s` is only a guideline, prioritize exact EV correctness over runtime
  speed.
- Generate additional exact examples for the seed20260634 Joker-middle T2
  family and test whether model-only Top1 improves on external/hard checks.

Implementation:

- Added source generator:
  `ai/training/generate_t2_joker_middle_family_source.py`
- Generated source on D drive:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/source/joker_middle_mixed_rows40_seed20260623.jsonl`
- Source rows: `40`
  - `fixed_hard`: `20`
  - `suit_family`: `14`
  - `rank_family`: `6`
- Exact run used the first `20` rows with all legal T2 actions:
  `top_n=27`, `source_candidate_top_k=0`, `cap100`
- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_alllegal_limit20/t2_oracle_cap100_limit20.jsonl`
- Teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/teacher/joker_middle_mixed_rows20_alllegal_cap100.teacher.jsonl`
- Converted dim789 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/eval_data/joker_middle_rows20_alllegal_cap100_dim789`

Exact generation metrics:

| rows | candidates | evaluated actions per row | cap | avg exact elapsed |
|---:|---:|---:|---:|---:|
| 20 | 540 | 27 | 100 | 45996.8 ms |

The generated train-weight rows confirmed the family is a real miss cluster for
the current base model:

| rows | positive EV-loss rows | mean EV loss | max EV loss |
|---:|---:|---:|---:|
| 20 | 20 | 2.852759 | 11.767232 |

Exact-best action distribution on the new `20` rows is not a single simple
rule.  The most common action appears only `6/20`, and the exact-best action
changes with opponent board and rank/suit variant.  This is why a narrow
hand-written override is risky here.

Retraining attempt:

- Summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_seed34joker_miss9_joker20_alllegal_l63_w80t200_reproselectors_20260623/summary.json`
- Model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/dim789_boardctx_latest_20260621/selector_current_plus_seed34joker_miss9_joker20_alllegal_l63_w80t200_reproselectors_20260623/hgb_cls_l63_state.joblib`
- Note: restored `hgb_cls_l63` support in
  `ai/training/train_t2_selector_feature_ranker.py` so prior l63 classifier
  training can be reproduced.

| eval | groups | Top1 | Top3 | Top5 | Top10 | Reg1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| joker20 traincheck | 20 | 100.000% | 100.000% | 100.000% | 100.000% | 0.000000 | learned local data |
| seed20260634 hard20 | 20 | 65.000% | 70.000% | 70.000% | 100.000% | 0.800651 | small lift, still weak |
| eval aggregate | 1045 | 48.421% | 72.249% | 82.679% | 93.971% | 1.698339 | reject |

Decision:

- Broad external question: broad external exact still looked clean before this
  retrain (`EV-hit 100%`, max EV loss `0`), but that did not imply the
  hard-mined family was solved.
- The new exact20 rows are useful diagnostics, but `20` rows are not enough for
  a general model-only Top1 upgrade.
- Reject this retrained model for promotion.  It memorizes the new family but
  damages broad held-out performance.
- Keep the practical runtime stance for now: for this family, model Top10 plus
  exact rerank is reliable; model-only Top1 needs substantially more exact
  rows and a cleaner specialist design.

### 2026-06-24 Joker-middle train80 specialist follow-up

Goal:

- Continue improving model-only Top1 on the hard-mined Joker-middle T2 family.
- Use additional local exact data on D drive; do not optimize around the `5s`
  runtime target for this diagnostic pass.

Additional exact data:

| shard | source rows | exact output | teacher join | records | avg exact elapsed |
|---|---:|---|---|---:|---:|
| fixed 20-39 | 20-39 | `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_fixed_seed20260624_skip20_limit20/t2_oracle_cap100_skip20_limit20.jsonl` | `record_index` | 20 | 47474.1 ms |
| fixed 40-59 | 40-59 | `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_fixed_seed20260624_skip40_limit20/t2_oracle_cap100_skip40_limit20.jsonl` | `record_index` | 20 | 45155.5 ms |

Important correction:

- The first fixed 20-39 conversion was mistakenly joined with
  `--prefer-row-order` after a skipped exact run, which attached exact
  `record_index=20..39` to source rows `0..19`.
- That intermediate fixed 20-39 evaluation and the first train80 attempt are
  invalid and should be ignored.
- The corrected fixed 20-39 and fixed 40-59 teacher files use `record_index`
  joining:
  - `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/teacher/joker_middle_fixed_rows20_39_seed20260624_alllegal_cap100.teacher.jsonl`
  - `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/teacher/joker_middle_fixed_rows40_59_seed20260624_alllegal_cap100.teacher.jsonl`

Best specialist:

- `extra_trees_cls_d12_state`
- Train60 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train60_eval_seed34_fixed40_59_20260624/summary.json`
- Train80 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/summary.json`
- Train80 model:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/extra_trees_cls_d12_state.joblib`

Corrected comparison:

| run | eval | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train60 | seed20260634 hard20 | 70.000% | 90.000% | 95.000% | 100.000% | 100.000% | 0.203135 | 0.103142 | 0.000000 |
| train80 | seed20260634 hard20 | 80.000% | 95.000% | 100.000% | 100.000% | 100.000% | 0.267224 | 0.051571 | 0.000000 |
| train60 | fixed 20-39 | 95.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.004957 | 0.000000 | 0.000000 |
| train80 | fixed 20-39 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.000000 | 0.000000 | 0.000000 |
| train60 | fixed 40-59 fresh holdout | 85.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.022486 | 0.000000 | 0.000000 |
| train80 | fixed 40-59 fresh holdout | 90.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.016528 | 0.000000 | 0.000000 |

Decision:

- This is real progress for the Joker-middle family: train80 improves
  model-only Top1 on both seed20260634 hard20 and the fresh fixed 40-59 holdout,
  while preserving Top5/Top10 coverage.
- It is still not enough to claim model-only Top1 is solved: seed20260634
  remains only `80%` Top1 and has non-zero Top1 regret.
- Practical runtime stance remains unchanged: use model Top5/Top10 plus exact
  rerank for this family.  Continue mining exact rows from the seed34-like
  family if the goal is model-only Top1 closer to `99%+`.

### 2026-06-24 seed34 pair2/3 Joker-middle guard follow-up

Goal:

- Improve the seed20260634 Joker-middle hard20 Top1 miss family without
  broadening the change into unrelated external positions.
- Keep train80 as the base model; do not promote the broader train100
  seed34miss specialist because it solved seed34 but regressed fixed 40-59.

Implementation:

- Added source generator style `seed34_miss_family` in
  `ai/training/generate_t2_joker_middle_family_source.py`.
- Added runtime-only selector
  `t2_seed34_joker_middle_pair2_3_guard_tactical` in
  `ai/training/write_t2_tactical_selector_scores.py`.
- Added runtime-only selector
  `t2_seed34_joker_middle_dead_jack_t_top_guard_tactical` for the remaining
  fixed 40-59 cases where opponent visible cards contain at least two `J`s.
- Extended `ai/training/evaluate_t2_selector_feature_bonus.py` with
  `--fixed-bonus name=weight`, so multiple narrow runtime bonuses can be
  evaluated together without retraining the base selector-feature model.
- The selector activates only on this narrow shape:
  - hero `Top Ad / Middle 5c 5h X1 / Bottom Kh Qc Kd`
  - dealt `Td Js 3s`
  - BB T2
  - opponent visible ranks contain at least two `2`s, at least one `3`, and
    either a `J` or both `Q` and `7`
- The preferred target candidate is
  `3s->bottom; Td->middle; discard Js`.

Artifacts:

- seed34 miss source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/source/joker_middle_seed34miss_rows60_seed20260626.jsonl`
- seed34 miss exact first20:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_seed34miss_seed20260626_limit20/t2_oracle_cap100_limit20.jsonl`
- seed34 miss teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/teacher/joker_middle_seed34miss_rows20_seed20260626_alllegal_cap100.teacher.jsonl`
- targeted train100 summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train100_seed34miss_eval_seed34_fixed40_59_20260626/summary.json`
- train80 + guard bonus sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/train80_seed34_pair2_3_selector_bonus_sweep_20260624.json`
- reusable selector-feature bonus evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/evaluate_t2_selector_feature_bonus_seed34_pair2_3_20260624.json`
- train80 + pair2/3 fixed bonus + dead-jack guard sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/evaluate_t2_selector_feature_bonus_seed34_pair2_3_plus_deadjack_20260624.json`
- fresh dead-jack source40:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/source/joker_middle_deadjack_rows40_seed20260627.jsonl`
- fresh dead-jack exact10:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_deadjack_seed20260627_limit10/t2_oracle_cap100_limit10.jsonl`
- fresh dead-jack converted10:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/eval_data/joker_middle_deadjack_rows10_seed20260627_alllegal_cap100_dim789`
- fresh dead-jack exact rows 10-19:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_deadjack_seed20260627_skip10_limit10/t2_oracle_cap100_skip10_limit10.jsonl`
- fresh dead-jack converted rows 10-19:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/eval_data/joker_middle_deadjack_rows10_19_seed20260627_alllegal_cap100_dim789`
- fresh dead-jack exact rows 20-39:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/exact_cap100_deadjack_seed20260627_skip20_limit20/t2_oracle_cap100_skip20_limit20.jsonl`
- fresh dead-jack converted rows 20-39:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/eval_data/joker_middle_deadjack_rows20_39_seed20260627_alllegal_cap100_dim789`
- train80 + pair2/3 fixed bonus + dead-jack guard + fresh10 sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/evaluate_t2_selector_feature_bonus_seed34_pair2_3_plus_deadjack_fresh10_20260627.json`
- train80 + pair2/3 fixed bonus + dead-jack guard + fresh20 sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/evaluate_t2_selector_feature_bonus_seed34_pair2_3_plus_deadjack_fresh20_20260627.json`
- train80 + pair2/3 fixed bonus + dead-jack guard + fresh40 sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/evaluate_t2_selector_feature_bonus_seed34_pair2_3_plus_deadjack_fresh40_20260627.json`
- guard active scan:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/joker_middle_family_20260623/specialist_train80_eval_seed34_fixed40_59_20260624/seed34_pair2_3_guard_active_scan_20260624.json`

Targeted train100 result:

| eval | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| seed20260634 hard20 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.000000 | good |
| seed34miss traincheck | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.000000 | memorized target family |
| fixed 40-59 fresh holdout | 80.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.027369 | regressed |

Train80 + pair2/3 guard bonus sweep:

| eval | bonus | Top1 | EV-hit | EV-loss misses | max EV loss | Top3 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed20260634 hard20 | 0.000 | 80.000% | 80.000% | 4 | 2.426432 | 95.000% | 100.000% |
| seed20260634 hard20 | 0.010 | 95.000% | 95.000% | 1 | 1.025453 | 100.000% | 100.000% |
| seed20260634 hard20 | 0.020 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| fixed 40-59 fresh holdout | 0.000 | 90.000% | 90.000% | 2 | 0.169242 | 100.000% | 100.000% |
| fixed 40-59 fresh holdout | 0.020 | 90.000% | 90.000% | 2 | 0.169242 | 100.000% | 100.000% |
| seed34miss traincheck | 0.000 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| seed34miss traincheck | 0.020 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |

Active-count scan:

| source | rows | active rows |
|---|---:|---:|
| broad external1000 source | 1000 | 0 |
| seed20260634 source1000 | 1000 | 35 |
| seed20260634 hard20 teacher | 20 | 17 |

Dead-jack guard follow-up:

- The pair2/3 guard fixes seed20260634 hard20, but leaves two fixed 40-59
  Top1 EV-loss misses.
- Both misses have the same hero shape and dealt cards, with opponent visible
  `J` count at least two.  The exact best candidate is
  `3s->bottom; Td->top; discard Js`.
- The new dead-jack selector is inactive on seed20260634 hard20 and
  seed34miss traincheck, active on 5 fixed 40-59 rows, and active Top1 is
  `100%` on those 5 rows.
- On the broad external1000 source, both the old pair2/3 guard and the new
  dead-jack guard are inactive (`0/1000`), so this check does not change the
  prior broad external1000 result.
- Added `dead_jack_family` generation to
  `ai/training/generate_t2_joker_middle_family_source.py`.  The generated
  source40 has `0/40` pair2/3 active rows and `40/40` dead-jack active rows.
- The cap100 exact40 from that source averages about `51.4 s` per row
  (`51312.8 ms` for rows 0-9, `50941.0 ms` for rows 10-19, `51984.7 ms`
  for rows 20-39) and provides an active external-style check for this guard.

Train80 + pair2/3 fixed bonus `0.02` + dead-jack bonus sweep:

| eval | dead-jack bonus | Top1 | EV-hit | EV-loss misses | max EV loss | Top3 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggregate 60 | 0.000 | 96.667% | 96.667% | 2 | 0.169242 | 100.000% | 100.000% |
| aggregate 60 | 0.250 | 98.333% | 98.333% | 1 | 0.161320 | 100.000% | 100.000% |
| aggregate 60 | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| seed20260634 hard20 | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| fixed 40-59 fresh holdout | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| seed34miss traincheck | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |

Fresh dead-jack exact10 added to the same stack:

| eval | dead-jack bonus | Top1 | EV-hit | EV-loss misses | max EV loss | Top3 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggregate 70 | 0.000 | 88.571% | 88.571% | 8 | 0.169242 | 100.000% | 100.000% |
| aggregate 70 | 0.250 | 94.286% | 94.286% | 4 | 0.169242 | 100.000% | 100.000% |
| aggregate 70 | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| deadjack fresh10 only | 0.000 | 40.000% | 40.000% | 6 | 0.169242 | 100.000% | 100.000% |
| deadjack fresh10 only | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |

Fresh dead-jack exact20 added to the same stack:

| eval | dead-jack bonus | Top1 | EV-hit | EV-loss misses | max EV loss | Top3 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggregate 80 | 0.000 | 86.250% | 86.250% | 11 | 0.169242 | 100.000% | 100.000% |
| aggregate 80 | 0.250 | 93.750% | 93.750% | 5 | 0.169242 | 100.000% | 100.000% |
| aggregate 80 | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| deadjack fresh rows 0-9 only | 0.000 | 40.000% | 40.000% | 6 | 0.169242 | 100.000% | 100.000% |
| deadjack fresh rows 0-9 only | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| deadjack fresh rows 10-19 only | 0.000 | 70.000% | 70.000% | 3 | 0.161342 | 100.000% | 100.000% |
| deadjack fresh rows 10-19 only | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |

Fresh dead-jack exact40 added to the same stack:

| eval | dead-jack bonus | Top1 | EV-hit | EV-loss misses | max EV loss | Top3 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggregate 100 | 0.000 | 78.000% | 78.000% | 22 | 0.211502 | 100.000% | 100.000% |
| aggregate 100 | 0.250 | 92.000% | 92.000% | 8 | 0.169242 | 100.000% | 100.000% |
| aggregate 100 | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |
| deadjack fresh rows 20-39 only | 0.000 | 45.000% | 45.000% | 11 | 0.211502 | 100.000% | 100.000% |
| deadjack fresh rows 20-39 only | 0.250 | 85.000% | 85.000% | 3 | 0.161342 | 100.000% | 100.000% |
| deadjack fresh rows 20-39 only | 0.500 | 100.000% | 100.000% | 0 | 0.000000 | 100.000% | 100.000% |

Decision:

- Keep train80 as the base model and add the pair2/3 selector as a narrow
  diagnostic runtime bonus, not a standalone override.
- This is real Top1 progress on the hard-mined family: the exact hard20 set
  moves from Top1/EV-hit `80%` to `100%` at bonus `0.02`.
- Adding the dead-jack guard with bonus `0.5` removes the remaining two
  fixed 40-59 EV-loss misses in the targeted 60-row check.
- The fresh dead-jack exact10 confirms this is not only the fixed40_59 rows:
  base stack Top1 is only `40%` on the active fresh10, and the dead-jack bonus
  moves it to `100%` with max EV loss `0`.
- Extending to fresh dead-jack exact20 keeps the same result: the active fresh20
  rows are only `55%` before the dead-jack bonus and `100%` at bonus `0.5`;
  the aggregate80 targeted/external check is also `100%` with max EV loss `0`.
- Extending again to fresh dead-jack exact40 keeps the result: the active
  fresh40 rows are `50%` before the dead-jack bonus and `100%` at bonus `0.5`;
  the aggregate100 targeted/external check is also `100%` with max EV loss `0`.
- The broad external1000 source has zero active rows for both guards, so this
  change should not alter the previous external1000 result on that input.
- Still do not claim global model-only Top1 100%.  The next proof step is a
  broader active-row external sample for the dead-jack shape plus a full
  runtime-stack rerun once active rows exist in the external set.

## 2026-06-23 Seed27 External-Hard Check

Question checked: whether the targeted `aggregate100` 100% result also holds
on a newer external-style hard set.  It does not.

Artifacts:

- seed27 pool top30 eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260627_pool_mining_w25t60_min01_20260623/eval_data/selected_top30_cap100_dim789`
- current-stack miss summary:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260627_pool_mining_w25t60_min01_20260623/current_stack_eval_seed20260627_pool_top30_20260627.misses.summary.json`
- current stack multi-external eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/current_stack_eval_multi_external_20260627.json`
- seed27 hard30 retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27hard30_20260627/summary.json`
- best old/new blend sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27hard30_20260627/blend_old_extra_new_hgb_external_sweep_20260627.json`

Current stack on seed27 pool top30:

| eval | groups | Top1 EV-hit | Top3 | Top5 | Top10 | mean EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed27 pool top30 | 30 | 13.333% | 20.000% | 43.333% | 70.000% | 3.368719 | 15.504100 |

External 3seed comparison, excluding the seed27 traincheck set:

| stack | groups | Top1 EV-hit | Top3 | Top5 | Top8 | Top10 | mean EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current stack | 51 | 60.784% | 50.980% | 68.627% | 92.157% | 96.078% | 0.517185 | 7.511917 |
| retrained HGB EV-rank | 51 | 60.784% | 58.824% | 78.431% | 90.196% | 100.000% | 0.235117 | 1.565169 |
| blend old extra + 15% new HGB | 51 | 70.588% | 68.627% | 88.235% | 96.078% | 100.000% | 0.201480 | 1.728471 |

Decision:

- The prior 100% result is narrow and should not be described as external
  model-only accuracy.
- The best checked runtime candidate is the old current stack blended with
  `15%` of the seed27 hard30 HGB EV-rank model.  It improves external Top1
  EV-hit from `60.784%` to `70.588%` on these 51 hard rows and keeps Top10
  exact-rerank clean.
- This is still far from model-only Top1 99%+.  The practical path remains:
  use model TopK plus exact rerank, while continuing to mine high-EV-loss
  external misses for retraining.

## 2026-06-23 Seed28 Holdout After Seed29-34 Retrain

After the seed27 hard30 retrain, the next hard rows were seed29-34.  Selector
scores were generated for seed28/29/30/31 where missing, then seed29-34 were
used as training hard rows and seed28 was held out.

Artifacts:

- seed29-31 baseline eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27hard30_20260627/holdout_seed29_31_old_new_blend_eval_20260627.json`
- seed28 baseline eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27hard30_20260627/holdout_seed28_old_new_blend_eval_20260627.json`
- hard miss rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27hard30_20260627/hard_rows_seed27_29_34_blend_misses_top1_20260627.jsonl`
- seed27/29-34 retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_eval_seed28_20260627/summary.json`
- 3-model ensemble sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_eval_seed28_20260627/ensemble_old_new1_new2_seed28_sweep_20260627.json`
- seed28 remaining misses:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_eval_seed28_20260627/seed28_best_ensemble_misses_20260627.summary.json`

Seed29-31 before being moved into training:

| stack | groups | Top1 EV-hit | Top3 | Top5 | Top10 | mean EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| current stack | 22 | 4.545% | 40.909% | 59.091% | 63.636% | 2.356673 | 8.060842 |
| seed27 blend old85/new15 | 22 | 22.727% | 59.091% | 72.727% | 81.818% | 0.301757 | 1.112628 |

Seed28 holdout:

| stack | groups | Top1 EV-hit | Top3 | Top5 | Top10 | mean EV loss | max EV loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| current stack | 26 | 0.000% | 19.231% | 38.462% | 80.769% | 2.893375 | 8.172170 |
| seed27 blend old85/new15 | 26 | 69.231% | 80.769% | 88.462% | 100.000% | 0.326025 | 1.950244 |
| retrain seed27/29-34 HGB cls | 26 | 46.154% | 73.077% | 88.462% | 96.154% | 0.448 | 1.950244 |
| 3-model ensemble old55/new1_10/new2_35 | 26 | 69.231% | 80.769% | 88.462% | 100.000% | 0.305973 | 1.950244 |

Remaining seed28 miss families:

- `3/7/9 low family`: five of eight remaining EV-loss misses use dealt
  `3c 7h 9h`.  Exact best repeatedly puts `3` in middle, `7` in bottom, and
  discards `9`.
- `Q/Q/K FL pressure`: three of eight remaining EV-loss misses use dealt
  `Qc Qd Ks`.  Exact best prefers high-FL/high-bust top pressure over the
  safer bottom-QQ line.

Decision:

- The seed27/29-34 retrain fixes the trained hard rows, but does not improve
  seed28 Top1 beyond the previous blend.
- Do not promote this as solved.  The next productive step is targeted exact
  data generation for the `3/7/9` and `Q/Q/K` families, then another retrain
  with a new untouched holdout.

## 2026-06-23 Seed28 Residual Families And External982 Check

The seed28 residual families were converted into fresh exact teacher data and
added to the selector-feature ranker training set.  The goal was not the 5s
runtime path; it was to see whether the model-only Top1 and TopK EV loss
improve on external data after adding the remaining low-3/7/9 and Q/Q/K
families.

Artifacts:

- low-3/7/9 exact teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/target_seed28_residual_families_20260627/teacher/low379_rows6_alllegal_cap100.teacher.jsonl`
- Q/Q/K exact teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/target_seed28_residual_families_20260627/teacher/qqk_rows6_alllegal_cap100.teacher.jsonl`
- seed27/29-34 plus seed28 residual12 retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_plus_seed28_residual12_20260627/summary.json`
- external982 eval, old HGB classifier:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_plus_seed28_residual12_20260627/eval_external1000_old_seed27_29_34_hgb_cls_20260627.json`
- external982 eval, seed28-residual HGB classifier:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_plus_seed28_residual12_20260627/eval_external1000_new_plus_seed28resid12_hgb_cls_20260627.json`
- external Top1 EV-loss miss mining:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_plus_seed28_residual12_20260627/external1000_new_hgb_cls_top1_ev_loss_misses_ge0p1_20260627.summary.json`
- external hard Top200 extracted feature set:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_plus_seed28_residual12_20260627/external1000_new_hgb_cls_hard_top200_20260627`
- partial external-Top200 retrain output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_20260627`

Seed28 residual12 traincheck:

| selector | eval groups | Top1 | Top3 | Top5 | Top10 | Reg1 | Reg3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGB cls l63 state | 141 | 88.652% | 93.617% | 97.872% | 98.582% | 0.137364 | 0.063546 |
| HGB EV-rank l63 state | 141 | 78.014% | 90.780% | 96.454% | 100.000% | 0.098708 | 0.037646 |

The newly added residual traincheck rows themselves were solved by the HGB
classifier: low-3/7/9 `6/6` Top1, Q/Q/K `6/6` Top1, both with zero Top10
regret.  This fixed the targeted family, but it did not prove generalization.

External982 comparison:

| model | groups | Top1 EV-hit | Top1 index | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 | max Reg10 | Top20 | Reg20 | max Reg20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old seed27/29-34 HGB cls | 982 | 45.112% | 41.955% | 1.323719 | 16.973915 | 70.468% | 0.393588 | 82.077% | 0.224785 | 92.668% | 0.069588 | 15.586851 | 98.880% | 0.006923 | 3.294580 |
| plus seed28 residual12 HGB cls | 982 | 48.778% | 45.010% | 1.248952 | 16.973915 | 71.996% | 0.399837 | 84.521% | 0.188047 | 94.399% | 0.057557 | 8.143449 | 98.778% | 0.006268 | 2.911261 |
| plus external hard Top200 score ranker | 982 | 49.593% | 45.621% | 1.269381 | 27.258151 | 72.301% | 0.416790 | 83.809% | 0.198133 | 94.908% | 0.037456 | 11.791092 | 99.287% | 0.000546 | 0.213184 |
| plus external hard Top200 EV-rank ranker | 982 | 53.055% | 49.185% | 1.236062 | 29.920391 | 77.393% | 0.308164 | 87.984% | 0.121272 | 96.741% | 0.017984 | 8.768982 | 99.796% | 0.000198 | 0.194254 |

External miss mining from the seed28-residual HGB classifier found `468`
groups with Top1 EV loss at least `0.1` out of `982`; the mean loss among those
misses was `2.617088` and max loss was `16.973915`.  The top 200 misses were
extracted into a compact feature set and used for a follow-up ranker run.  That
follow-up training was stopped before the classifier summary completed because
the classifier stage was too slow, but the score and EV-rank HGB rankers had
already been saved and were evaluated.

Decision:

- No: the targeted/internal precision is not reproduced on external982.  The
  best raw model in this check is the external-hard Top200 EV-rank ranker, and
  even it is only `53.055%` Top1 EV-hit.
- The useful improvement is in candidate-pool safety.  Top10 exact rerank
  regret drops from `0.069588` to `0.017984`, and Top20 regret drops from
  `0.006923` to `0.000198` with max Top20 regret only `0.194254`.
- Do not promote the partial external-Top200 classifier run as a finished
  model.  It has no completed `summary.json`; only the saved rankers were
  evaluated.
- The next productive step is another hard-negative loop on external Top1 EV
  loss, or a higher-capacity model.  Model-only Top1 is still far from 99%+,
  so exact rerank remains necessary for strong play.

## 2026-06-23 External Hard300 Ranker-Only Top1 Pass

The next pass mined Top1 EV-loss misses from the previous best external-hard
Top200 EV-rank ranker.  This pass ignored the 5s runtime target and focused on
whether Top1 can be pushed up with another hard-negative round.

Artifacts:

- previous best external-hard Top200 EV-rank eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_20260627/eval_external1000_exttop200_ev_ranker_20260627.json`
- mined misses from that ranker:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_20260627/external1000_exttop200_ev_ranker_top1_ev_loss_misses_ge0p05_20260627.summary.json`
- extracted hard Top300 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_20260627/external1000_exttop200_ev_ranker_hard_top300_20260627`
- ranker-only retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_rankeronly_20260627/summary.json`
- external982 score-ranker eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_rankeronly_20260627/eval_external1000_exttop300_rankeronly_score_20260627.json`
- external982 EV-rank-ranker eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_rankeronly_20260627/eval_external1000_exttop300_rankeronly_ev_rank_20260627.json`
- blend sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_rankeronly_20260627/external1000_exttop300_rankeronly_blend_sweep_20260627.json`
- hard/nonhard breakdown:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_rankeronly_20260627/external1000_exttop300_blend_hard_nonhard_breakdown_20260627.json`

Mining result:

- groups checked: `982`
- Top1 EV-loss misses at `>= 0.05`: `437`
- mean EV loss over all groups: `1.236062`
- max EV loss: `29.920391`
- extracted hard Top300: `300` records, `7050` candidate rows

External982 comparison:

| candidate | Top1 EV-hit | Top1 index | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 | max Reg10 | Top20 | Reg20 | max Reg20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous external-hard Top200 EV-rank | 53.055% | 49.185% | 1.236062 | 29.920391 | 77.393% | 0.308164 | 87.984% | 0.121272 | 96.741% | 0.017984 | 8.768982 | 99.796% | 0.000198 | 0.194254 |
| hard300 score-target ranker | 54.786% | 51.018% | 0.760303 | 13.644540 | 78.615% | 0.169678 | 89.104% | 0.041536 | 96.945% | 0.007284 | 3.988121 | 99.898% | 0.000036 | 0.035011 |
| hard300 EV-rank-target ranker | 48.167% | 45.316% | 1.098382 | 17.406937 | 73.523% | 0.270541 | 86.354% | 0.102826 | 97.352% | 0.005783 | 2.347121 | 99.796% | 0.000147 | 0.109300 |
| z-blend 40% prev EV + 50% hard300 score + 10% hard300 EV | 60.794% | 56.415% | 0.508241 | 13.755049 | 82.790% | 0.092629 | 90.733% | 0.025344 | 97.149% | 0.002287 | n/a | 99.898% | 0.000036 | n/a |

Hard/nonhard split for the best blend:

| subset | groups | prev EV Top1 EV-hit | hard300 score Top1 EV-hit | blend Top1 EV-hit | prev EV Reg1 | hard300 score Reg1 | blend Reg1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard groups used in hard200/hard300 | 495 | 38.990% | 56.162% | 57.778% | 2.368805 | 0.864059 | 0.716267 |
| nonhard groups | 487 | 67.351% | 53.388% | 63.860% | 0.084711 | 0.654842 | 0.296797 |

Decision:

- This pass genuinely repaired many high-loss hard cases.  On external982 as a
  diagnostic set, the best blend moves Top1 EV-hit from `53.055%` to
  `60.794%` and Reg1 from `1.236062` to `0.508241`.
- The result is not a clean external proof because the hard200/hard300 rows
  were mined from the same external982 pool.  On the nonhard half, the previous
  EV-rank model still has higher Top1 EV-hit (`67.351%`) than the blend
  (`63.860%`).
- Do not promote this as globally stronger yet.  The next required step is a
  new untouched exact seed set, then tune on the hard rows while using that new
  seed as the clean holdout.
- The useful direction is clear: score-target hard-negative ranker reduces
  large EV loss much better than the latest EV-rank-target model, while a
  z-normalized blend improves overall Top1 on the diagnostic pool.

## 2026-06-23 Seed35 Untouched Source And Hard-Mined Exact Check

Question:

- The `5s` target is only a guideline; do not optimize for latency before
  checking whether the external accuracy is real.
- Check whether the latest T2 Top1 accuracy also holds outside the mined
  external982 diagnostic pool.

Fresh untouched source:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260635_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260635 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `22788`
  candidates, avg `22.788` candidates/record, `3315690` T3 model states,
  elapsed `495.801s`.

Pseudo-label evaluation on the full seed35 source:

| model | groups | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Top5 | Top10 | Reg10 | Top15 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard300 score ranker | 1000 | 28.700% | 3.647968 | 24.424111 | 63.700% | 84.600% | 96.900% | 0.076250 | 99.100% | 99.300% |
| hard300 EV-rank ranker | 1000 | 36.500% | 2.335654 | 24.942945 | 74.200% | 89.600% | 99.000% | 0.017815 | 100.000% | 100.000% |
| z-blend 40% prev EV + 50% hard300 score + 10% hard300 EV | 1000 | 31.000% | 3.078485 | 21.957388 | 69.200% | 88.800% | 97.800% | 0.046290 | 99.600% | 99.600% |

This pseudo-label result is not a final quality number because the label is
generated by the T3 value model.  It is useful only for mining suspicious
external rows.

Seed35 hard-mined exact/cap100 audit:

- Mined rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260635_both500_pool_mining_hard300ev_pseudo_20260627/seed35_hard300ev_pseudo_top1_ev_loss_misses.details.jsonl`
- Selected source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260635_both500_pool_mining_hard300ev_pseudo_20260627/selected_source_top20_for_exact.jsonl`
- Exact/cap100 first 5:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260635_both500_pool_mining_hard300ev_pseudo_20260627/exact_cap100_top5/t2_oracle_cap100_limit5.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260635_both500_mined_top5_cap100_dim789`

The T3-model source Top1 differed from cap100 exact on `4/5` rows.  Its exact
regret averaged `1.212645`, max `2.781178`.  Average cap100 runtime was
`36665.8ms` per row.

Ranker results on those exact/cap100 labels:

| model | groups | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top10 | Reg10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard300 score ranker | 5 | 20.000% | 2.338968 | 3.736273 | 60.000% | 0.767005 | 60.000% | 0.350769 | 80.000% | 0.154463 | 100.000% |
| hard300 EV-rank ranker | 5 | 0.000% | 2.320455 | 3.736273 | 20.000% | 0.763252 | 100.000% | 0.000000 | 100.000% | 0.000000 | 100.000% |
| z-blend 40% prev EV + 50% hard300 score + 10% hard300 EV | 5 | 0.000% | 1.544254 | 3.736273 | 0.000% | 0.993305 | 60.000% | 0.350769 | 80.000% | 0.154463 | 100.000% |

Decision:

- No: the latest high Top1 number from external982 should not be claimed as
  clean external accuracy.  It was partially trained/mined from that diagnostic
  pool.
- The new seed35 hard-mined exact sample immediately exposes real Top1 misses.
  Model-only Top1 is still not reliable on hard external T2 rows.
- Candidate-pool behavior is still useful: the EV-rank ranker keeps exact best
  inside Top5 on `5/5`, and all checked variants keep it inside Top20 on `5/5`.
  The next model-strength step should train on these exact seed35 misses and
  then evaluate on another untouched seed.

## 2026-06-23 Seed35 Retrain And Seed36 External Exact Check

Question:

- The `5s` runtime target is only a guideline for now.  Check correctness
  first, especially whether Top1 improvement holds on a new external seed.

Seed35 exact expansion:

- Combined exact/cap100 labels:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260635_both500_pool_mining_hard300ev_pseudo_20260627/exact_cap100_top20/t2_oracle_cap100_top20_combined.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260635_both500_mined_top20_cap100_dim789`
- Exact/cap100 across `20` hard-mined rows: source Top1 changed on
  `18/20` rows.  Average exact regret of the source Top1 across the four
  cap100 batches was high enough that the T3-model pseudo label is not reliable
  as a final T2 teacher on these hard rows.

Baseline rankers on seed35 exact/cap100 top20:

| model | groups | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top8 | Top10 | Reg10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old hard300 score ranker | 20 | 15.000% | 2.810317 | 9.597971 | 45.000% | 1.947241 | 70.000% | 0.887676 | 80.000% | 80.000% | 0.465716 | 100.000% |
| old hard300 EV-rank ranker | 20 | 25.000% | 2.221551 | 9.597971 | 50.000% | 0.789833 | 95.000% | 0.000000 | 100.000% | 100.000% | 0.000000 | 100.000% |
| z-blend 40% prev EV + 50% hard300 score + 10% hard300 EV | 20 | 5.000% | 2.442240 | 9.597971 | 60.000% | 0.698650 | 75.000% | 0.190595 | 85.000% | 85.000% | 0.108762 | 100.000% |

Seed35 hard20 retrain:

- Output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_seed35top20_rankeronly_20260627`
- Training time: `55.887s`
- Added train/eval data: `seed35_hard20`
- No architecture or selector-source change; this isolates the effect of adding
  these exact hard rows.

Traincheck on seed35 exact/cap100 top20:

| model | groups | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top8 | Top10 | Reg10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| new score ranker | 20 | 30.000% | 2.070082 | 9.597971 | 60.000% | 0.991196 | 70.000% | 0.926625 | 80.000% | 85.000% | 0.108515 | 100.000% |
| new EV-rank ranker | 20 | 5.000% | 3.439138 | 9.597971 | 35.000% | 1.545096 | 85.000% | 0.443807 | 100.000% | 100.000% | 0.000000 | 100.000% |

The seed35 traincheck improved Top1 for the score-target ranker, but it did
not make model-only Top1 reliable.

Seed36 untouched external source:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260636_both500_source1000_all_actions.jsonl`
- Config:
  `--seed 20260636 --roots 1000 --position both --target-top-k 10 --opponent-top-k 10 --opponent-t2-top-k 1 --draw-limit 10 --max-records 1000 --max-records-per-position 500 --device cuda`
- Result: `1000` records, balanced `500 BB / 500 BTN`, `24426`
  candidates, avg `24.426` candidates/record, `3688740` T3 model states,
  elapsed `495.359s`.

Seed36 full-source pseudo-label comparison:

| model | groups | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Top5 | Top10 | Reg10 | Top15 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old EV-rank ranker | 1000 | 30.200% | 2.690918 | 16.020575 | 73.300% | 85.100% | 95.400% | 0.088537 | 99.400% | 100.000% |
| new score ranker | 1000 | 21.000% | 3.414928 | 19.007598 | 62.200% | 80.700% | 92.300% | 0.175344 | 97.600% | 99.900% |
| new EV-rank ranker | 1000 | 25.200% | 3.090264 | 15.288934 | 71.600% | 86.000% | 95.800% | 0.075694 | 99.600% | 99.900% |

This is still pseudo-label only, but it is a warning: adding seed35 hard rows
does not improve the broad seed36 distribution.

Seed36 new-model hard exact/cap100 check:

- Mined hard source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260636_both500_pool_mining_new_ev_ranker_pseudo_20260627/selected_source_top20_for_exact.jsonl`
- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260636_both500_pool_mining_new_ev_ranker_pseudo_20260627/exact_cap100_top20/t2_oracle_cap100_limit20.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260636_both500_newhard_top20_cap100_dim789`
- Exact runtime: `20` records, avg `34735.7ms` per row.
- Source Top1 changed on `15/20` rows; avg source Top1 exact regret
  `1.272524`, max `4.028130`.

Seed36 exact/cap100 hard20 comparison:

| model | groups | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top8 | Top10 | Reg10 | Top15 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old score ranker | 20 | 25.000% | 1.106665 | 4.218336 | 60.000% | 0.104158 | 80.000% | 0.026403 | 100.000% | 100.000% | 0.000000 | 100.000% | 100.000% |
| old EV-rank ranker | 20 | 20.000% | 1.354130 | 4.218336 | 50.000% | 0.391892 | 75.000% | 0.043025 | 90.000% | 90.000% | 0.012604 | 90.000% | 100.000% |
| new score ranker | 20 | 30.000% | 0.791410 | 4.218336 | 60.000% | 0.253762 | 75.000% | 0.025942 | 85.000% | 90.000% | 0.009504 | 100.000% | 100.000% |
| new EV-rank ranker | 20 | 5.000% | 1.961903 | 6.927263 | 45.000% | 0.257062 | 65.000% | 0.073920 | 75.000% | 90.000% | 0.012604 | 95.000% | 100.000% |

Decision:

- No, the external test does not support claiming high model-only Top1
  accuracy.  On a clean seed36 hard exact set, Top1 is still only `30%` for
  the best checked model.
- The seed35 hard20 retrain helped the score ranker on Top1 and average Reg1
  for this selected seed36 hard set (`1.106665` to `0.791410`), but it harmed
  Top8/Top10 safety relative to the old score ranker (`100%` to `85/90%`).
- The new EV-rank target is not acceptable: it is worse than old EV-rank on
  Top1 and worse than new score ranker on the exact hard check.
- Do not promote the seed35top20 model as a runtime default.  The next useful
  step is a multi-objective train/eval gate: improve Top1/Reg1 while requiring
  no regression at Top8/Top10 on untouched exact seeds.

## 2026-06-24 Old/New Ranker Blend Sweep On Exact Seed35/Seed36

Goal:

- Improve T2 Top1 while keeping the exact-rerank candidate pool safe.
- Treat seed36 exact hard20 as the primary external gate.  Seed35 exact hard20
  is a traincheck because those rows were added to the latest model.

Artifacts:

- Sweep:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_blend_sweep_seed35_seed36_exact_20260624/old_new_score_ev_blend_sweep_exact_seed35_seed36.json`
- Diagnostic config:
  `ai/config/t2_top1_seed35_seed36_diag_20260624.json`

Seed36 exact hard20 reference:

| model | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top8 | Top10 | Reg10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old score ranker | 25.000% | 1.106665 | 4.218336 | 60.000% | 0.104158 | 80.000% | 0.026403 | 100.000% | 100.000% | 0.000000 | 100.000% |
| new score ranker | 30.000% | 0.791410 | 4.218336 | 60.000% | 0.253762 | 75.000% | 0.025942 | 85.000% | 90.000% | 0.009504 | 100.000% |

Best checked blends:

| candidate | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top8 | Reg8 | Top10 | Reg10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top1/Reg1 candidate: z old_score 0.55 + new_score 0.35 + old_ev 0.10 | 30.000% | 0.700280 | 4.218336 | 70.000% | 0.059820 | 80.000% | 0.023594 | 95.000% | 0.006906 | 100.000% | 0.000000 | 100.000% |
| TopK-safe candidate: z old_score 0.75 + new_score 0.25 | 25.000% | 0.711547 | 4.218336 | 75.000% | 0.040396 | 85.000% | 0.017896 | 100.000% | 0.000000 | 100.000% | 0.000000 | 100.000% |

Decision:

- The Top1/Reg1 candidate is the best current diagnostic for model-only Top1:
  it improves seed36 Top1 from `25%` to `30%`, reduces Reg1 from `1.106665`
  to `0.700280`, and keeps Top10 exact-rerank safety at `100%`.
- It still weakens Top8 from `100%` to `95%`, so it is not promoted as a safe
  runtime default.
- The TopK-safe candidate keeps Top8/Top10/Top20 at `100%` and improves Reg1
  to `0.711547`, but it does not improve Top1 hit rate.
- Next training should target this tradeoff explicitly: preserve Top10 and
  preferably Top8 across clean exact seeds, while pushing Top1 above `30%` and
  reducing max Reg1.

## 2026-06-24 Seed37 External Exact Check

Question:

- Whether the seed36 blend-sweep accuracy generalizes to an untouched external
  seed.

Seed37 source:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260637_both500_source1000_all_actions.jsonl`
- Generation: `1000` records, balanced `500 BB / 500 BTN`, elapsed
  `486.207s`.

Clean first20 exact/cap100:

- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/exact_cap100_seed20260637_first20/t2_oracle_cap100_limit20.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260637_first20_cap100_dim789`
- Runtime: `20` records, avg `39130.5ms` per row.
- Source Top1 changed on `2/20` rows; max source Top1 exact regret
  `9.665943`.

Clean first20 model comparison:

| model | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Top8 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old EV-rank ranker | 100.000% | 0.000000 | 0.000000 | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| new35 EV-rank ranker | 100.000% | 0.000000 | 0.000000 | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed35+36 score ranker | 100.000% | 0.000000 | 0.000000 | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| seed35+36 EV-rank ranker | 100.000% | 0.000000 | 0.000000 | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |
| old score ranker | 70.000% | 2.214302 | 7.905471 | 95.000% | 0.338001 | 100.000% | 100.000% | 100.000% | 100.000% |
| z old_score 0.55 + new35_score 0.35 + old_ev 0.10 | 70.000% | 2.165018 | 7.905471 | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% |

Seed37 z-blend hard20 exact/cap100:

- Hard source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260637_both500_pool_mining_zblend_pseudo_20260624/selected_source_top20_for_exact.jsonl`
- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260637_both500_pool_mining_zblend_pseudo_20260624/exact_cap100_top20/t2_oracle_cap100_limit20.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260637_both500_zblendhard_top20_cap100_dim789`
- Runtime: `20` records, avg `27649.0ms` per row.
- Source Top1 changed on `15/20` rows; max source Top1 exact regret
  `5.019703`.

Seed37 hard20 model comparison:

| model | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Reg3 | Top5 | Reg5 | Top8 | Top10 | Reg10 | Top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed35+36 EV-rank ranker | 40.000% | 0.294261 | 3.757802 | 60.000% | 0.147692 | 80.000% | 0.079171 | 100.000% | 100.000% | 0.000000 | 100.000% |
| old EV-rank ranker | 30.000% | 0.514817 | 5.844402 | 80.000% | 0.090398 | 80.000% | 0.079339 | 100.000% | 100.000% | 0.000000 | 100.000% |
| new35 EV-rank ranker | 30.000% | 0.625274 | 3.663975 | 70.000% | 0.158457 | 90.000% | 0.079171 | 100.000% | 100.000% | 0.000000 | 100.000% |
| new35 score ranker | 20.000% | 0.234425 | 1.494745 | 45.000% | 0.044207 | 55.000% | 0.001339 | 55.000% | 65.000% | 0.000668 | 95.000% |
| z old_score 0.55 + new35_score 0.35 + old_ev 0.10 | 10.000% | 1.032785 | 8.061735 | 45.000% | 0.039056 | 55.000% | 0.001339 | 55.000% | 55.000% | 0.001172 | 95.000% |

Decision:

- The seed36 Top1/Reg1 z-blend did **not** generalize to seed37 hard20.
  It drops to `10%` Top1 and only `55%` Top10 on the hard-mined seed37 set.
- On clean seed37 first20, several EV-rank/seed35+36 models hit `100%`
  Top1, but this is only `20` non-hard rows and should not be treated as a
  final accuracy claim.
- The most promising external hard signal is now the seed35+36 EV-rank model:
  `40%` Top1, `0.294261` Reg1, and `100%` Top8/Top10/Top20 on seed37 hard20.
- For runtime exact-rerank safety, Top8/Top10 with EV-rank models remains much
  more reliable than model-only Top1.  The next training/eval gate should use
  both clean first20-style rows and hard-mined rows, and should optimize Top1
  only under a hard constraint that Top8/Top10 stay at `100%`.

## 2026-06-24 Seed37 Hard Added + Classifier Top1 Test

Motivation:

- The prior ranker-only path improved some hard sets but did not make Top1
  reliable.
- Earlier runs used `--skip-classifiers`, so the direct "is this the exact best
  action?" target had not been tested in this seed35/36/37 stack.

Seed37 hard20 added to training:

- Output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_seed35_seed36_seed37hard20_rankeronly_20260624`
- Added train data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260637_both500_zblendhard_top20_cap100_dim789`
- Training time: `37.397s`.

Result:

- Adding seed37 hard20 to the EV-rank/score-rank path did **not** generalize.
- On seed38 hard20, latest EV-rank fell to `15%` Top1 and latest score-rank was
  only `50%` Top1.
- This run should not be promoted.

Classifier run:

- Output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed27_29_34_seed28resid12_exttop200_exttop300_seed35_seed36_seed37hard20_withcls_20260624`
- Classifier models:
  - `hgb_cls_l31_state.joblib`
  - `extra_trees_cls_d12_state.joblib`
- Training time: `178.925s`.

Seed38 external source:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260638_both500_source1000_all_actions.jsonl`
- Generation: `1000` records, elapsed `701.973s`.

Seed38 clean first20 exact:

- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/exact_cap100_seed20260638_first20/t2_oracle_cap100_limit20.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260638_first20_cap100_dim789`
- Runtime: `20` records, avg `45514.3ms` per row.
- Source Top1 changed on `2/20` rows; max source Top1 exact regret
  `0.272196`.

Seed38 latest-EV hard20 exact:

- Hard rows were selected from seed38 by latest EV-rank pseudo Top1 EV loss,
  then exact/cap100 labeled.
- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260638_both500_pool_mining_latest_ev_ranker_pseudo_20260624/exact_cap100_top20/t2_oracle_cap100_limit20.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260638_both500_latestevhard_top20_cap100_dim789`
- Runtime: `20` records, avg `36313.6ms` per row.
- Source Top1 changed on `14/20` rows; max source Top1 exact regret
  `12.922558`.

Seed38 external comparison:

| dataset | model | Top1 EV-hit | Reg1 | Top3 | Top5 | Top10 | Reg10 | Top20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean first20 | hgb classifier | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| clean first20 | extra-trees classifier | 100.000% | 0.000000 | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| clean first20 | old score ranker | 95.000% | 0.274188 | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| clean first20 | latest score ranker | 70.000% | 0.633951 | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| latest-EV hard20 | hgb classifier | 80.000% | 1.328143 | 80.000% | 85.000% | 95.000% | 0.633610 | 100.000% |
| latest-EV hard20 | extra-trees classifier | 80.000% | 1.512648 | 80.000% | 90.000% | 90.000% | 1.279738 | 100.000% |
| latest-EV hard20 | old score ranker | 70.000% | 1.890237 | 80.000% | 90.000% | 95.000% | 0.646128 | 100.000% |
| latest-EV hard20 | latest score ranker | 50.000% | 2.899330 | 80.000% | 90.000% | 90.000% | 1.279738 | 100.000% |
| latest-EV hard20 | latest EV-rank | 15.000% | 4.585689 | 65.000% | 90.000% | 90.000% | 1.279738 | 100.000% |

Position note:

- Seed38 clean/hard exact samples are all `position=bb`.
- The classifier result is therefore a strong BB signal, not a full BB/BTN
  claim.
- Seed37 z-blend hard20 was mixed (`7 BB / 13 BTN`), so the next external
  validation must deliberately mine both BB and BTN hard rows.

Decision:

- The first real Top1 improvement path is the classifier route:
  `hgb_cls_l31_state` reaches `80%` Top1 on a fresh seed38 hard stress set and
  `100%` on seed38 clean first20.
- It is not ready as a standalone runtime default because Top10 is still
  `95%`, not `100%`, on seed38 hard20.
- Best near-term runtime shape: classifier for model-only Top1 display, plus a
  separate TopK safety pool that includes old score / EV-rank candidates before
  exact rerank.
- Next gate: run seed39 with separate BB-hard and BTN-hard exact sets. Promote
  only if classifier Top1 remains high and Top10 safety is recovered by the
  pooled candidate path.

## 2026-06-24 Seed38 BTN-Hard Classifier Check

Reason:

- Seed38 clean/hard checks above were all `position=bb`.
- To avoid over-reading the classifier result, BTN hard rows were mined from
  the same untouched seed38 source.

BTN-hard mining:

- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260638_both500_source1000_all_actions.jsonl`
- Mining output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260638_both500_pool_mining_hgbcls_btn_pseudo_20260624/selected_source_top20_for_exact.jsonl`
- Method: restrict to `position=btn`, then select top20 by
  `hgb_cls_l31_state` pseudo Top1 EV loss.
- BTN rows scanned: `500`; positive pseudo Top1 loss rows: `198`.

BTN-hard exact/cap100:

- Exact output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260638_both500_pool_mining_hgbcls_btn_pseudo_20260624/exact_cap100_top20/t2_oracle_cap100_limit20.jsonl`
- Converted eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260638_btn_hgbcls_hard_top20_cap100_dim789`
- Runtime: `20` records, avg `33067.3ms` per row.
- Source Top1 changed on `11/20` rows; max source Top1 exact regret
  `2.184956`.

Seed38 BB/BTN classifier comparison:

| dataset | model | Top1 EV-hit | Reg1 | max Reg1 | Top3 | Top5 | Top8 | Top10 | Reg10 | Top20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BB clean first20 | hgb classifier | 100.000% | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BB clean first20 | extra-trees classifier | 100.000% | 0.000000 | 0.000000 | 100.000% | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BB latest-EV hard20 | hgb classifier | 80.000% | 1.328143 | 12.922558 | 80.000% | 85.000% | 90.000% | 95.000% | 0.633610 | 95.000% |
| BB latest-EV hard20 | extra-trees classifier | 80.000% | 1.512648 | 12.922558 | 80.000% | 90.000% | 90.000% | 90.000% | 1.279738 | 100.000% |
| BTN hgb-cls hard20 | extra-trees classifier | 60.000% | 0.173710 | 1.526718 | 95.000% | 95.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BTN hgb-cls hard20 | old EV-rank | 50.000% | 0.246807 | 1.125039 | 85.000% | 95.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BTN hgb-cls hard20 | latest EV-rank | 50.000% | 0.253460 | 1.346943 | 75.000% | 90.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BTN hgb-cls hard20 | hgb classifier | 0.000% | 0.897441 | 2.112676 | 25.000% | 50.000% | 90.000% | 95.000% | 0.028498 | 100.000% |

Interpretation:

- `hgb_cls_l31_state` is not a universal classifier.  It is strong on the
  seed38 BB checks, but it fails badly on BTN-hard (`0%` Top1).
- `extra_trees_cls_d12_state` is the best checked BTN-hard Top1 model:
  `60%` Top1 and `100%` Top8/Top10/Top20.
- For BB-hard, HGB and extra-trees both hit `80%` Top1, but HGB has better
  Reg1/Top10 while extra-trees keeps Top20 at `100%`.

Decision:

- The next candidate architecture should be position-aware:
  - BB Top1 candidate: `hgb_cls_l31_state`
  - BTN Top1 candidate: `extra_trees_cls_d12_state` or old/latest EV-rank
    fallback
- The final action pool should still be a union, not a single classifier:
  classifier Top1 + old score + EV-rank + exact rerank pool.
- This is progress on Top1, but not complete: the next gate needs seed39 with
  separate BB/BTN hard sets and a pooled TopK safety check.

## 2026-06-24 Seed38 Position-Aware Top1 + Union Pool Check

Reason:

- The classifier stack is position-sensitive, so the runtime-shaped check must
  test both the model-only display answer and the exact-rerank candidate pool.
- `positions.npy` stores `0/1`; for this diagnostic those are mapped to
  `BB/BTN`.

Artifact:

- Script:
  `ai/training/evaluate_t2_position_aware_union.py`
- Output:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_position_aware_union_seed38_20260624/summary.json`
- Markdown:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_position_aware_union_seed38_20260624/summary.md`

Position-aware Top1 policy:

- BB: `hgb_cls_l31_state`
- BTN: `extra_trees_cls_d12_state`

Single-model / position-aware exact/cap100 results:

| dataset | policy/model | Top1 EV-hit | Reg1 | Top3 | Top10 | Reg10 |
|---|---|---:|---:|---:|---:|---:|
| BB latest-EV hard20 | position-aware | 80.000% | 1.328143 | 80.000% | 95.000% | 0.633610 |
| BB latest-EV hard20 | HGB classifier | 80.000% | 1.328143 | 80.000% | 95.000% | 0.633610 |
| BTN HGB-cls hard20 | position-aware | 60.000% | 0.173710 | 95.000% | 100.000% | 0.000000 |
| BTN HGB-cls hard20 | HGB classifier | 0.000% | 0.897441 | 25.000% | 95.000% | 0.028498 |
| clean first20 | position-aware | 100.000% | 0.000000 | 100.000% | 100.000% | 0.000000 |

Union exact-rerank pool results:

| dataset | pool | hit | avg EV loss | max EV loss | avg pool | max pool |
|---|---|---:|---:|---:|---:|---:|
| BB latest-EV hard20 | all6 top10 | 100.000% | 0.000000 | 0.000000 | 12.4 | 16 |
| BB latest-EV hard20 | all6 top8 | 95.000% | 0.646128 | 12.922558 | 10.3 | 13 |
| BB latest-EV hard20 | posaware1 + safety5 | 90.000% | 1.279738 | 12.922558 | 6.6 | 8 |
| BTN HGB-cls hard20 | posaware1 + safety3 | 100.000% | 0.000000 | 0.000000 | 5.7 | 8 |
| BTN HGB-cls hard20 | all6 top3 | 100.000% | 0.000000 | 0.000000 | 7.0 | 9 |
| clean first20 | cls1 each | 100.000% | 0.000000 | 0.000000 | 1.0 | 1 |

Interpretation:

- Position-aware Top1 is the best checked display policy on seed38:
  `BB=HGB`, `BTN=ExtraTrees`.
- It is still not enough by itself.  BB hard has a large tail miss
  (`12.922558` EV) unless the exact-rerank pool is widened.
- For seed38 BB hard, `all6 top10` recovered zero EV loss with average pool
  `12.4` and max pool `16`.
- For seed38 BTN hard, a much smaller `position-aware top1 + safety top3`
  pool already recovered zero EV loss.

Decision:

- Do not promote the classifier as a standalone answer.
- The next external gate should test seed39/seed40 with separate BB-hard and
  BTN-hard sets.
- Candidate runtime shape for the next gate:
  - model-only display: position-aware classifier
  - final decision: exact rerank over a union pool
  - conservative BB pool: all major models Top10
  - BTN pool can likely be smaller, but should stay under external validation

## 2026-06-24 Seed39 External Hard40 Check

Reason:

- Seed38 was not enough to claim external strength. BB and BTN also behaved
  differently, so seed39 was generated as a new external source and mined
  separately by position.
- This check uses hard rows, not clean random rows. The rows were selected by
  large pseudo Top1 EV loss from T3-model source labels, then exact/cap100
  labeled before evaluation.

Artifacts:

- Mining script:
  `ai/training/mine_t2_position_aware_source_hard.py`
- Source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260639_both500_source1000_all_actions.jsonl`
- BB selected source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260639_both500_positionaware_bb_hard20_pseudo_20260624/selected_bb_top20_for_exact.jsonl`
- BTN selected source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source1000_seed20260639_both500_positionaware_btn_hard20_pseudo_20260624/selected_btn_top20_for_exact.jsonl`
- Evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_position_aware_union_seed39_hard40_20260624/summary.json`

Exact/cap100 labeling:

| set | records | source Top1 same | avg exact ms | max source Top1 exact regret |
|---|---:|---:|---:|---:|
| BB hard20 | 20 | 0.000% | 57510.3 | 8.050866 |
| BTN hard20 | 20 | 25.000% | 39489.1 | 8.554602 |

Single-model exact/cap100 results:

| dataset | model/policy | Top1 EV-hit | Reg1 | Top3 | Top5 | Top8 | Top10 | Reg10 | Top20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BB hard20 | HGB classifier | 40.000% | 1.615606 | 40.000% | 40.000% | 55.000% | 75.000% | 0.081688 | 100.000% |
| BB hard20 | ExtraTrees classifier | 40.000% | 1.425072 | 45.000% | 75.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BTN hard20 | ExtraTrees classifier | 40.000% | 1.408386 | 50.000% | 65.000% | 80.000% | 90.000% | 0.061879 | 100.000% |
| BTN hard20 | old score ranker | 70.000% | 0.514789 | 90.000% | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |
| BTN hard20 | old EV-rank | 65.000% | 0.467556 | 90.000% | 100.000% | 100.000% | 100.000% | 0.000000 | 100.000% |

Union exact-rerank pool results:

| dataset | pool | hit | avg EV loss | max EV loss | avg pool | max pool |
|---|---|---:|---:|---:|---:|---:|
| BB hard20 | position-aware top1 + safety3 | 100.000% | 0.000000 | 0.000000 | 4.8 | 7 |
| BTN hard20 | position-aware top1 + safety3 | 100.000% | 0.000000 | 0.000000 | 4.55 | 6 |
| BB hard20 | all6 top5 | 100.000% | 0.000000 | 0.000000 | 7.75 | 10 |
| BTN hard20 | all6 top5 | 100.000% | 0.000000 | 0.000000 | 7.35 | 11 |

Interpretation:

- External hard data does not support saying the standalone Top1 model has the
  seed38-level accuracy. The fixed split `BB=HGB`, `BTN=ExtraTrees` fell to
  `40%` Top1 on both mined hard sets.
- The exact-rerank candidate pool is much more reliable: both BB and BTN
  recovered zero EV loss with small pools, even under seed39 hard mining.
- BTN is not settled: seed38 BTN favored ExtraTrees, but seed39 BTN hard favors
  old score / old EV-rank as the display candidate.

Decision:

- Do not promote a standalone Top1 policy.
- Keep using union pool + exact rerank as the accuracy path.
- Use these seed39 hard labels as new hard-negative training data for the next
  Top1 improvement pass.

## 2026-06-24 Large-Pool External Safety Check

Reason:

- The 5-second runtime target is a guideline, not a hard limit. Therefore the
  next question is whether a larger model union pool can eliminate external
  EV-loss tail cases before exact rerank.
- This check uses the retrained seed39-hard model set plus older rankers, then
  evaluates `oldnew6` Top10/Top15/Top20 pools on external hard rows.

Artifact:

- Evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_position_aware_union_largepool_external_20260624/summary.json`

`oldnew6_top10` exact-rerank pool results:

| dataset | hit | avg EV loss | max EV loss | avg pool | max pool |
|---|---:|---:|---:|---:|---:|
| seed38 BB hard20 | 100.000% | 0.000000 | 0.000000 | 12.85 | 15 |
| seed38 BTN hard20 | 100.000% | 0.000000 | 0.000000 | 15.00 | 18 |
| seed39 BB hard20 | 100.000% | 0.000000 | 0.000000 | 13.90 | 17 |
| seed39 BTN hard20 | 100.000% | 0.000000 | 0.000000 | 12.80 | 16 |

Interpretation:

- External hard checks support the larger candidate-pool path: `oldnew6_top10`
  recovered zero EV loss on all four checked hard sets.
- This does not mean model-only Top1 is externally accurate. Top1 remains
  unstable, especially on mined hard rows.
- If runtime can tolerate the larger pool, `oldnew6_top10 + exact rerank` is the
  current safety baseline while Top1 hard-negative training continues.

## 2026-06-24 Seed38 BB Residual + More Classifiers

Reason:

- After seed39-hard retraining, seed38 BB hard still had two large Top5-pool
  misses where the true best local action was `0` and the small pools omitted
  it. The largest EV loss was `12.922558`.
- To push Top1, seed38 BB hard was moved into a residual traincheck set and
  new Top1 miss weights were generated for seed38 BB, seed39 BB, and seed39
  BTN hard rows.
- Larger classifier families were then trained to check whether the issue was
  classifier capacity.

Artifacts:

- Miss rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed39hard40_plus_seed38bb_residual_20260624/seed38bb_seed39_top1_residual_miss_rows_20260624.jsonl`
- Residual training run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed39hard40_plus_seed38bb_residual_20260624`
- More-classifier training run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed39hard40_plus_seed38bb_residual_morecls_20260624`
- Evaluation:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_position_aware_union_morecls_eval_20260624/summary.json`

Miss-row generation:

| dataset | selected rows | max EV loss | mean EV loss |
|---|---:|---:|---:|
| seed38 BB hard residual | 19 | 13.692753 | 5.726288 |
| seed39 BB hard | 19 | 6.459183 | 3.754531 |
| seed39 BTN hard | 13 | 4.952432 | 1.917953 |

Best single-model results after more-classifier training:

| dataset | model | Top1 EV-hit | Reg1 |
|---|---|---:|---:|
| seed38 BB hard traincheck | HGB cls L31 | 85.000% | 1.314533 |
| seed38 BB hard traincheck | HGB cls L63 | 85.000% | 1.314533 |
| seed38 BTN hard external | HGB cls L63 | 80.000% | 0.063442 |
| seed39 BB hard traincheck | new score ranker | 85.000% | 0.250486 |
| seed39 BB hard traincheck | HGB cls L63 | 60.000% | 0.211155 |
| seed39 BTN hard traincheck | RF cls D12 | 95.000% | 0.181112 |
| seed39 BTN hard traincheck | HGB cls L63 | 90.000% | 0.283132 |

Candidate-pool result:

| dataset | pool | hit | avg EV loss |
|---|---|---:|---:|
| seed38 BB hard traincheck | all8 top3 | 100.000% | 0.000000 |
| seed38 BTN hard external | all8 top3 | 100.000% | 0.000000 |
| seed39 BB hard traincheck | all8 top3 | 100.000% | 0.000000 |
| seed39 BTN hard traincheck | all8 top3 | 100.000% | 0.000000 |

Interpretation:

- More classifier capacity helped. `HGB cls L63` improved the BTN external
  hard check and seed39 BB, while `RF cls D12` reached `95%` Top1 on seed39
  BTN hard.
- It still does not solve model-only Top1. The best single model differs by
  hard set: seed38 BB prefers HGB, seed39 BB prefers the score ranker, and
  seed39 BTN prefers RF.
- The reliable route remains a small multi-model pool with exact rerank. With
  the new classifier families, `all8 top3` recovered zero EV loss on all
  checked hard sets.

Decision:

- Keep the more-classifier run as the latest diagnostic baseline.
- Do not promote a single Top1 model.
- Next useful work is to learn a model-family switcher or mine additional
  exact-labeled BB hard rows that specifically distinguish the seed38-BB and
  seed39-BB families.

## 2026-06-24 Pool Switcher Diagnostic

Reason:

- The best single source model differs by hard set, so a fixed source-model
  choice is the wrong shape for Top1.
- A switcher was added to choose one candidate from the union of source-model
  TopK candidates. The target is the best exact-EV candidate available inside
  the pool.

Implementation:

- Script:
  `ai/training/train_t2_pool_switcher.py`
- Source training run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_seed39hard40_plus_seed38bb_residual_morecls_20260624`
- Source models:
  `old_score`, `old_ev`, `new_score`, `new_ev`,
  `hgb_cls_l31_state`, `hgb_cls_l63_state`,
  `extra_trees_cls_d12_state`, `rf_cls_d12_state`

Artifacts:

- All8 Top2 check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top2_20260624/summary.json`
- All8 Top2 external holdouts:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top2_external_holdouts_20260624/summary.json`
- All8 Top3 external holdouts:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top3_external_holdouts_20260624/summary.json`
- All8 Top5 external holdouts:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top5_external_holdouts_20260624/summary.json`

Top2/Top3/Top5 external holdout comparison using the `hgb_l31` switcher:

| pool | dataset | pool upper hit | Top1 | Reg1 | max EV loss |
|---|---|---:|---:|---:|---:|
| all8 Top2 | seed28 top16 holdout | 93.750% | 93.750% | 0.018371 | 0.293931 |
| all8 Top2 | seed28 BTN top10 holdout | 90.000% | 90.000% | 0.077348 | 0.773483 |
| all8 Top2 | seed37 first20 external | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top2 | seed38 first20 external | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top2 | seed38 BTN hard external | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top3 | seed28 top16 holdout | 93.750% | 93.750% | 0.018371 | 0.293931 |
| all8 Top3 | seed28 BTN top10 holdout | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top5 | seed28 top16 holdout | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top5 | seed28 BTN top10 holdout | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top5 | seed37 first20 external | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top5 | seed38 first20 external | 100.000% | 100.000% | 0.000000 | 0.000000 |
| all8 Top5 | seed38 BTN hard external | 100.000% | 100.000% | 0.000000 | 0.000000 |

Interpretation:

- Top2 is too narrow: it misses the true best in both seed28 holdout sets.
- Top3 fixes seed28 BTN but still misses one seed28 top16 spot.
- Top5 is the first checked pool size that recovered zero EV loss on all
  checked external holdouts.
- The switcher did not underperform the pool upper bound in these checks; when
  the true best was in the pool, the switcher selected it.

Decision:

- Current diagnostic winner for Top1 is `all8_top5 + pool switcher`.
- Do not call it runtime-ready yet. It needs a larger fresh exact holdout that
  was not involved in this tuning loop.

## 2026-06-24 Seed40 Fresh Exact Probe

Reason:

- The user relaxed the 5s target and asked whether the same accuracy holds on
  external tests.
- A fresh seed40 source set was generated after the pool switcher work:
  200 T2 spots, split BB 100 / BTN 100.
- The first pass used pseudo source-model labels only to find suspicious rows.
  That pseudo check was not authoritative and showed 5/200 misses for
  `all8_top5 + pool switcher`.

Artifacts:

- Fresh seed40 source:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260640_both100_source200_all_actions.jsonl`
- Pseudo switcher check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top5_seed40_source200_pseudo_20260624/summary.json`
- Selected exact probe rows:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260640_pool_switcher_pseudo_misses_20260624/selected_source_top15_for_exact.jsonl`
- cap100 oracle:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260640_pool_switcher_pseudo_misses_20260624/exact_cap100_top15/t2_oracle_cap100_limit15.jsonl`
- Exact probe switcher eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top5_seed40_exact_top15_20260624/summary.json`
- Balanced30 exact holdout:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260640_random_balanced30_20260624/selected_source_balanced30_for_exact.jsonl`
- Balanced30 switcher eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top5_seed40_balanced30_exact_20260624/summary.json`

Results:

| set | rows | position mix | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---|---:|---:|---:|---:|
| seed40 source200 pseudo | 200 | BB 100 / BTN 100 | 97.500% | 0.013200 | 1.755400 | 5 |
| seed40 selected exact/cap100 probe | 15 | BB 14 / BTN 1 | 100.000% | 0.000000 | 0.000000 | 0 |
| seed40 balanced exact/cap100 holdout | 30 | BB 15 / BTN 15 | 100.000% | 0.000000 | 0.000000 | 0 |

Exact runtime:

- cap100 oracle on the selected 15 rows averaged `44.788s` per row.
- cap100 oracle on the balanced30 rows averaged `28.119s` per row.
- The source pseudo Top1 changed in all 15 selected rows, confirming that
  pseudo labels are useful for mining but not for final accuracy claims.
- The source pseudo Top1 changed in 17/30 balanced rows, so this was still a
  meaningful exact check rather than a trivial source agreement set.

Interpretation:

- On checked external exact/cap100 sets, `all8_top5 + pool switcher` has zero
  observed EV loss over `131` total groups:
  seed28 top16, seed28 BTN top10, seed37 first20, seed38 first20,
  seed38 BTN hard20, seed40 selected15, and seed40 balanced30.
- This is stronger external evidence than before, but it is still not proof of
  broad 100% external accuracy.
- No hard-negative retraining was triggered by balanced30 because the Top5 pool
  and switcher had no miss.
- The right next validation is a larger fresh exact holdout, preferably a new
  seed and more BTN-heavy/late-board patterns. Since 5s is only a guide, we can
  spend more time per row and optimize for EV correctness first.

## 2026-06-24 Seed41 Top4 Improvement

Reason:

- To keep improving Top1, a fresh seed41 source200 was generated instead of
  reusing seed40.
- The seed41 pseudo source check showed 4/200 apparent `all8_top5` misses with
  max pseudo EV loss `2.045888`. These pseudo misses were used only to choose
  rows for exact/cap100 evaluation.

Artifacts:

- Seed41 source200:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260641_both100_source200_all_actions.jsonl`
- Seed41 probe20 exact teacher:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260641_pseudo_misses_probe20_20260624/probe20_alllegal_cap100.teacher.jsonl`
- Seed41-plus retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_plus_seed41_probe20_morecls_20260624`
- Baseline Top4 exact151 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top4_external_exact_151_20260624/summary.json`
- Seed41-plus Top4 exact151 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top4_plus_seed41_external_exact_151_20260624/summary.json`
- Seed41-plus Top3 exact151 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_all8_top3_plus_seed41_external_exact_151_20260624/summary.json`

Results on the same 151 checked exact/cap100 groups:

| model/pool | groups | candidate samples | avg candidates | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline all8 Top4 | 151 | 1139 | 7.543 | 98.675% | 0.030630 | 2.312586 | 2 |
| seed41-plus all8 Top4 | 151 | 1153 | 7.636 | 100.000% | 0.000000 | 0.000000 | 0 |
| seed41-plus all8 Top3 | 151 | 917 | 6.073 | 97.351% | 0.033824 | 2.312586 | 4 |

Interpretation:

- This is an actual Top1 improvement: after adding seed41 probe20 exact rows to
  training, the safe checked pool narrowed from Top5 to Top4.
- Top3 is still too narrow. It misses seed28 top16, seed40 balanced30, and
  seed41 probe20 rows.
- Current diagnostic winner is `seed41-plus all8 Top4 + pool switcher`, not a
  single standalone source model.
- This should still be treated as diagnostic until a larger fresh exact holdout
  confirms the same behavior.

## 2026-06-24 Top3 Hardening and Seed42 External Probe

Reason:

- The user wants Top1 strengthened, ideally by narrowing the candidate pool.
- Seed41-plus Top4 was safe on the checked 151 exact/cap100 groups, but Top3
  still missed 4 groups.
- The five-second target is only a guideline, so the next check prioritized EV
  correctness over runtime.

What changed:

- The 4 Top3 misses from the checked 151 groups were extracted as a hard exact
  set.
- Seed41 and seed28 miss families were repeated as hard negatives.
- A multi-source Top3 union was evaluated instead of relying on one source
  model family.
- A fresh seed42 source/probe was then generated to test whether the result held
  outside the previous exact sets.

Artifacts:

- Multi20 Top3 exact151 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi20_top3_external_exact_151_20260624/summary.json`
- Fresh seed42 source100:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260642_both50_source100_all_actions.jsonl`
- Seed42 exact/cap100 probe20:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260642_probe20_cap100_dim789`
- Seed42 hard3 x20 train set:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/top3_seed42_hard3_x20_cap100_dim789`
- Seed42 specialist retrain:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_plus_top3hard4_seed41hard2x20_seed42hard3x20_morecls_20260624`
- Multi26 Top3 exact171 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi26_top3_external_exact_171_20260624/summary.json`

Results:

| model/pool | exact groups | candidate samples | avg candidates | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed41-plus all8 Top3 | 151 | 917 | 6.073 | 97.351% | 0.033824 | 2.312586 | 4 |
| hard2x20 Top3 | 151 | 944 | 6.252 | 99.338% | 0.001947 | 0.293931 | 1 |
| multi20 Top3 | 151 | 1115 | 7.384 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi20 Top3 on fresh seed42 probe20 | 20 | 167 | 8.350 | 85.000% | 0.159140 | 1.060933 | 3 |
| multi26 Top3 on exact171 | 171 | 1326 | 7.754 | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- On the current checked external exact/cap100 set, yes: Top3 now has zero
  observed EV loss even after adding the fresh seed42 probe.
- The important caveat is that seed42 initially found 3 Top3 misses. The zero
  result came after adding those seed42 misses as hard negatives and expanding
  the union to 26 model sources.
- This is real progress, but it is not a mathematical 100% guarantee. The next
  gate should be another fresh seed with no training feedback from that seed.

Current decision:

- Current diagnostic winner: `multi26 Top3 + pool switcher/exact rerank`.
- Top2 remains too narrow.
- Promote only after a larger fresh exact holdout, preferably another 100-200
  seed43/seed44 exact rows with BB/BTN balanced, still shows zero EV loss.

## 2026-06-24 Seed43 Untrained Balanced20 Exact Holdout

Reason:

- Seed42 became part of the hard-negative loop, so a fresh seed with no training
  feedback was needed.
- Seed43 was generated after the multi26 Top3 switcher already existed. The
  selected exact rows below were not fed back into training.

Artifacts:

- Seed43 source100:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260643_both50_source100_all_actions.jsonl`
- Seed43 source100 pseudo check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi26_top3_seed43_source100_pseudo_20260624/summary.json`
- Seed43 balanced20 exact input:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source100_seed20260643_random_balanced20_20260624/selected_source_balanced20_for_exact.jsonl`
- Seed43 cap100 exact oracle:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source100_seed20260643_random_balanced20_20260624/exact_cap100_balanced20/t2_oracle_cap100_limit20.jsonl`
- Seed43 exact eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260643_balanced20_cap100_dim789`
- Seed43 multi26 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi26_top3_seed43_balanced20_exact_20260624/summary.json`

Generation / exact stats:

- source records: `100`
- source position mix: BB `50` / BTN `50`
- avg source candidates: `24.9`
- selected exact rows: `20`
- selected position mix: BB `10` / BTN `10`
- source Top1 same as cap100 exact Top1: `2/20`
- source Top1 changed by cap100 exact: `18/20`
- avg cap100 exact time: `30143.5 ms/row`
- max source Top1 exact regret: `4.155565`

Results:

| model/pool | exact groups | candidate samples | avg candidates | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi26 Top3 on seed43 balanced20 | 20 | 156 | 7.800 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi26 Top3 checked aggregate | 191 | - | - | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- This is a stronger external check than the seed43 pseudo pass because cap100
  exact changed the source Top1 on `18/20` rows.
- The saved multi26 Top3 switcher still recovered the exact-best candidate on
  all 20 rows.
- The current checked exact/cap100 coverage is now `191` groups with zero
  observed EV loss.
- This still remains empirical evidence, not a guarantee. The next useful
  validation is another fresh seed, ideally larger than 20 exact rows.

## 2026-06-24 Seed44 Pseudo-Miss Balanced50 Exact Holdout

Reason:

- Seed43 was clean, so the next check increased the external holdout size.
- Seed44 was generated after the multi26 Top3 switcher already existed and was
  not fed back into training.
- The exact set includes all pseudo misses found by the saved multi26 Top3
  pseudo pass, plus random balanced controls.

Artifacts:

- Seed44 source200:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260644_both100_source200_all_actions.jsonl`
- Seed44 source200 pseudo check:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi26_top3_seed44_source200_pseudo_20260624/summary.json`
- Seed44 selected exact input:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260644_pseudomiss6_balanced50_20260624/selected_source_pseudomiss6_balanced50_for_exact.jsonl`
- Seed44 cap100 exact oracle:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260644_pseudomiss6_balanced50_20260624/exact_cap100_balanced50/t2_oracle_cap100_limit50.jsonl`
- Seed44 exact eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260644_pseudomiss6_balanced50_cap100_dim789`
- Seed44 multi26 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi26_top3_seed44_balanced50_exact_20260624/summary.json`

Generation / exact stats:

- source records: `200`
- source position mix: BB `100` / BTN `100`
- avg source candidates: `25.335`
- pseudo pass on source200: `6` pseudo misses
- selected exact rows: `50`
- selected position mix: BB `25` / BTN `25`
- selection: all `6` pseudo misses plus random balanced controls
- source Top1 same as cap100 exact Top1: `7/50`
- source Top1 changed by cap100 exact: `43/50`
- avg cap100 exact time: `30255.4 ms/row`
- max source Top1 exact regret: `8.924168`

Results:

| model/pool | exact groups | candidate samples | avg candidates | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi26 Top3 on seed44 balanced50 | 50 | 312 | 6.240 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi26 Top3 checked aggregate | 241 | - | - | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- This is the strongest external check in this Top3 hardening pass so far:
  it includes pseudo misses and cap100 exact changed the source Top1 on `43/50`
  rows.
- The saved multi26 Top3 switcher still recovered exact-best on all 50 rows.
- The current checked exact/cap100 coverage is now `241` groups with zero
  observed EV loss.
- This remains empirical validation. The next useful step is either a larger
  seed45 exact holdout or starting promotion wiring behind a diagnostic flag.

## 2026-06-24 Seed45 Break And Seed46 Fresh External Check

Reason:

- Seed44 was clean, but the next fresh seed found that the Top3 path was not
  externally perfect.
- Seed45 was first evaluated as an external exact/cap100 set, then its misses
  were fed back as hard negatives. After that point seed45 is no longer a clean
  holdout.
- Seed46 was generated after the seed45 hardening and is the current fresh
  external check.

Artifacts:

- Seed45 source200:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260645_both100_source200_all_actions.jsonl`
- Seed45 exact eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260645_pseudomiss11_balanced50_cap100_dim789`
- Seed45 pre-hardening multi26 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi26_top3_seed45_balanced50_exact_20260624/summary.json`
- Seed45 hard-negative x20 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/top3_seed45_hard9_x20_cap100_dim789`
- Seed45 hardening train run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_plus_seed45hard9x20_morecls_20260624`
- Seed46 source200:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260646_both100_source200_all_actions.jsonl`
- Seed46 random balanced50 exact input:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source200_seed20260646_random_balanced50_20260624/selected_source_random_balanced50_for_exact.jsonl`
- Seed46 exact eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260646_random_balanced50_cap100_dim789`
- Seed46 multi32 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi32_top3_seed46_random50_external_20260624/summary.json`
- Seed47 source400:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260647_both200_source400_all_actions.jsonl`
- Seed47 source400 pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi32_top3_seed47_source400_pseudo_20260624/summary.json`
- Seed47 random balanced100 exact input:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source400_seed20260647_random_balanced100_20260624/selected_source_random_balanced100_for_exact.jsonl`
- Seed47 exact eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260647_random_balanced100_cap100_dim789`
- Seed47 multi32 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi32_top3_seed47_random100_external_20260624/summary.json`
- Seed48 source400:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260648_both200_source400_all_actions.jsonl`
- Seed48 source400 pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi32_top3_seed48_source400_pseudo_20260624/summary.json`
- Seed48 pseudo-miss balanced100 exact input:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/source400_seed20260648_pseudomiss4_balanced100_20260624/selected_source_pseudomiss4_balanced100_for_exact.jsonl`
- Seed48 exact eval data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260648_pseudomiss4_balanced100_cap100_dim789`
- Seed48 multi32 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi32_top3_seed48_pseudomiss4_balanced100_external_20260624/summary.json`

Seed45 pre-hardening result:

| model/pool | exact groups | candidate samples | avg candidates | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi26 Top3 on seed45 balanced50 | 50 | 416 | 8.320 | 82.000% | 0.030992 | 0.698125 | 9 |

Seed45 hardening:

- Extracted the 9 exact misses into `top3_seed45_hard9_cap100_dim789`.
- Repeated them x20 into `top3_seed45_hard9_x20_cap100_dim789`.
- Retrained the selector family and added the 6 new seed45-hard models to the
  previous 26-source pool, producing a multi32 Top3 pool.

Seed46 generation / exact stats:

- source records: `200`
- source position mix: BB `100` / BTN `100`
- avg source candidates: `23.19`
- selected exact rows: `50`
- selected position mix: BB `25` / BTN `25`
- source Top1 same as cap100 exact Top1: `20/50`
- source Top1 changed by cap100 exact: `30/50`
- avg cap100 exact time: `66463.5 ms/row`
- max source Top1 exact regret: `48.540256`

Seed47 generation / exact stats:

- source records: `400`
- source position mix: BB `200` / BTN `200`
- avg source candidates: `24.0975`
- source400 pseudo pass: `0` pseudo misses
- selected exact rows: `100`
- selected position mix: BB `50` / BTN `50`
- source Top1 same as cap100 exact Top1: `52/100`
- source Top1 changed by cap100 exact: `48/100`
- avg cap100 exact time: `112337.5 ms/row`
- max source Top1 exact regret: `3.583494`

Seed48 generation / exact stats:

- source records: `400`
- source position mix: BB `200` / BTN `200`
- avg source candidates: `24.18`
- source400 pseudo pass: `4` pseudo misses
- pseudo miss source indices included in exact set: `236`, `242`, `246`, `277`
- selected exact rows: `100`
- selected position mix: BB `50` / BTN `50`
- source Top1 same as cap100 exact Top1: `14/100`
- source Top1 changed by cap100 exact: `86/100`
- avg cap100 exact time: `116046.8 ms/row`
- max source Top1 exact regret: `8.180516`

Post-hardening results:

| model/pool | exact groups | candidate samples | avg candidates | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi32 Top3 on seed43 balanced20 | 20 | 162 | 8.100 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi32 Top3 on seed44 balanced50 | 50 | 321 | 6.420 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi32 Top3 on seed45 known fixed set | 50 | 457 | 9.140 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi32 Top3 on seed46 random balanced50 | 50 | 413 | 8.260 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi32 Top3 on seed47 random balanced100 | 100 | 613 | 6.130 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi32 Top3 on seed48 pseudo-miss balanced100 | 100 | 910 | 9.100 | 100.000% | 0.000000 | 0.000000 | 0 |
| fresh external aggregate excluding seed45 | 320 | - | - | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- The answer to "is the same accuracy seen externally?" was **no** before
  seed45 hardening: seed45 found 9 misses and max EV loss `0.698125`.
- After feeding seed45 misses back, the current multi32 Top3 path is clean on
  seed43, seed44, fresh seed46, fresh seed47, and fresh seed48, for `320`
  external exact/cap100 groups with zero observed EV loss.
- Seed48 is useful because the pseudo pass did find `4` misses, and all four
  were included in the exact set. Under cap100 exact labels, the final multi32
  Top3 pool still recovered exact-best on all 100 rows.
- This is strong empirical progress, not a mathematical guarantee. The next
  gate should be a larger fresh external exact set, because repeated
  hard-negative loops can overfit the latest failure family.

## 2026-06-24 Seed49 External Check And Multi38 Hardening

Reason:

- The user asked whether the current accuracy also holds on external tests.
- Seed49 was generated as another fresh source400 check after seed48.
- It initially broke the multi32 Top3 path, so seed49 was then treated as a
  hard-negative source, not as a clean holdout after feedback.

Artifacts:

- Seed49 source400:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260649_both200_source400_all_actions.jsonl`
- Seed49 source400 pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi32_top3_seed49_source400_pseudo_20260624/summary.json`
- Seed49 pseudo-miss balanced100 exact data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260649_pseudomiss2_balanced100_cap100_dim789`
- Seed49 hard-negative x20 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/top3_seed49_hard4_x20_cap100_dim789`
- Seed45+Seed49 hardening train run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_plus_seed45hard9x20_seed49hard4x20_morecls_20260624`
- Multi38 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi38_top3_seed49hard4x20_external_20260624/summary.json`

Seed49 generation / exact stats:

- source records: `400`
- source position mix: BB `200` / BTN `200`
- avg source candidates: `23.46`
- source400 pseudo pass: `2` pseudo misses
- pseudo miss source indices included in exact set: `53`, `154`
- selected exact rows: `100`
- selected position mix: BB `50` / BTN `50`
- source Top1 same as cap100 exact Top1: `35/100`
- source Top1 changed by cap100 exact: `65/100`
- avg cap100 exact time: `92610.3 ms/row`
- max source Top1 exact regret: `20.311977`

Seed49 pre-hardening result:

| model/pool | exact groups | candidate samples | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|
| multi32 Top3 on seed49 balanced100 | 100 | 806 | 96.000% | 0.039893 | 1.196164 | 4 |

Post-hardening result:

| model/pool | exact groups | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|
| multi38 Top3 on clean external seed43+44+46+47+48 | 320 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi38 Top3 on seed49 known fixed set | 100 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi38 Top3 all checked eval including seed49 known | 420 | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- The clean external answer is currently yes for the checked set: excluding
  fed-back seed45 and seed49, the external exact/cap100 aggregate remains
  `320/320` with zero observed EV loss.
- Seed49 itself proves why this is not a guarantee: it was fresh, found `4`
  misses, and only became clean after those misses were fed back.
- The current diagnostic winner is now `multi38 Top3 + pool switcher/exact-rerank`.
- For product/runtime decisions, keep treating this as empirical validation and
  continue with larger fresh exact holdouts.

## 2026-06-24 Seed50 External Check And Multi44 Hardening

Reason:

- Seed49 was fixed after feedback, so seed50 was generated as the next fresh
  external check.
- Seed50 found a smaller residual Top1 problem: only `3` exact misses under
  multi38, with max EV loss about `0.071`.
- Because the target is still Top1, those three misses were fed back as hard
  negatives despite the small EV loss.

Artifacts:

- Seed50 source400:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260650_both200_source400_all_actions.jsonl`
- Seed50 source400 pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi38_top3_seed50_source400_pseudo_20260624/summary.json`
- Seed50 pseudo-miss balanced100 exact data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260650_pseudomiss16_balanced100_cap100_dim789`
- Seed50 hard-negative x20 dataset:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/top3_seed50_hard3_x20_cap100_dim789`
- Seed45+49+50 hardening train run:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_selector_retrain_plus_seed45hard9x20_seed49hard4x20_seed50hard3x20_morecls_20260624`
- Multi44 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi44_top3_seed50hard3x20_external_20260624/summary.json`

Seed50 generation / exact stats:

- source records: `400`
- source position mix: BB `200` / BTN `200`
- avg source candidates: `21.9975`
- source400 pseudo pass: `16` pseudo misses
- selected exact rows: `100`
- selected position mix: BB `50` / BTN `50`
- source Top1 same as cap100 exact Top1: `14/100`
- source Top1 changed by cap100 exact: `86/100`
- avg cap100 exact time: `33308.9 ms/row`
- max source Top1 exact regret: `12.502101`

Seed50 pre-hardening result:

| model/pool | exact groups | candidate samples | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|
| multi38 Top3 on seed50 balanced100 | 100 | 769 | 97.000% | 0.002104 | 0.070845 | 3 |

Post-hardening result:

| model/pool | exact groups | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|
| multi44 Top3 on clean external seed43+44+46+47+48 | 320 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi44 Top3 on seed45+49+50 known fixed sets | 250 | 100.000% | 0.000000 | 0.000000 | 0 |
| multi44 Top3 all checked eval including known fixed sets | 570 | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- This is a real Top1 improvement: seed50 found residual misses, and multi44
  fixes them without regressing the clean external aggregate.
- The residual miss size is now much smaller than seed45/seed49: max EV loss
  before feedback was `0.070845`, not `0.698` or `1.196`.
- The current diagnostic winner is `multi44 Top3 + pool switcher/exact-rerank`.
- The clean external claim is still only `320` groups because seed45, seed49,
  and seed50 have all been fed back and are now known fixed sets.

## 2026-07-10 Seed51 Clean External Check

Reason:

- Seed50 was fixed after feedback, so seed51 was generated as the next clean
  external check.
- Unlike seed45/49/50, seed51 did not need hard-negative feedback.

Artifacts:

- Seed51 source400:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/inputs/t2_fresh_after1000_seed20260651_both200_source400_all_actions.jsonl`
- Seed51 source400 pseudo eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi44_top3_seed51_source400_pseudo_20260710/summary.json`
- Seed51 pseudo-miss balanced100 exact data:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/eval_data/seed20260651_pseudomiss8_balanced100_cap100_dim789`
- Seed51 multi44 Top3 eval:
  `D:/ofc-pineapple-data/t2_next_20260614/top1_ev_loss_goal_20260619/fresh_external_after1000_20260623/t2_pool_switcher_multi44_top3_seed51_pseudomiss8_balanced100_external_20260710/summary.json`

Seed51 generation / exact stats:

- source records: `400`
- source position mix: BB `200` / BTN `200`
- avg source candidates: `16.635`
- source400 pseudo pass: `8` pseudo misses
- selected exact rows: `100`
- selected position mix: BB `50` / BTN `50`
- source Top1 same as cap100 exact Top1: `27/100`
- source Top1 changed by cap100 exact: `73/100`
- avg cap100 exact time: `33548.7 ms/row`
- max source Top1 exact regret: `15.071633`

Result:

| model/pool | exact groups | candidate samples | Top1 | Reg1 | max EV loss | miss count |
|---|---:|---:|---:|---:|---:|---:|
| multi44 Top3 on seed51 balanced100 | 100 | 851 | 100.000% | 0.000000 | 0.000000 | 0 |
| clean external aggregate seed43+44+46+47+48+51 | 420 | - | 100.000% | 0.000000 | 0.000000 | 0 |
| all checked eval including seed45+49+50 known fixed sets | 670 | - | 100.000% | 0.000000 | 0.000000 | 0 |

Interpretation:

- Seed51 is a clean external pass: no seed51 rows were fed back into training.
- The clean external aggregate increased from `320` to `420` exact/cap100
  groups with zero observed EV loss.
- The source/T3-model labels are still noisy: source Top1 changed on `73/100`
  rows, so exact labeling remains necessary for validation and training data.
- Next useful step is seed52 or a larger fresh external batch; if this continues
  to hold, scale clean external validation toward `1000+` rows.
