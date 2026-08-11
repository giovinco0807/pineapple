# HU Turn1 Stage1 Pilot Plan

Status: GCP 2k MC1 teacher pilot completed with the fast T2 continuation;
schema and distribution checks passed. Production-scale T1 remains No-Go.
The current validation front is post-fix T1 TopK confirm with a deployable safe
selector veto. Plain `cse1/d0` reached the fired-count audit but remains No-Go.
Fresh `d3/cse1` also failed. Increasing independent confirm MC to `confirm16`
at `d0/cse1` produced enough fires but negative realized per-fire delta, so that
TopK confirm candidate is also No-Go. The next line is the same k8/confirm16
candidate with `safe0.5` as an additional veto. After Stage1e/f/g/h all failed
to beat the existing T1 baseline, the current active line is Stage1j clean
higher-MC teacher generation. MC4, MC16, MC32 first-only, and MC32 balanced
full-action pilots completed successfully, and a 110-record MC32 natural
partial run confirmed the label-quality profile while exposing a serious slow
tail. MC32 remains useful for selected refinement, but
production-scale T1 remains No-Go until a strict heldout model beats the
existing T1 baseline.

This starts the Turn1 work after the Turn2 continuation became available as
`stage9f_p2`.

## Fixed Continuation

- T1 target phase: `hu_turn1_5card`
- T2 continuation profile: `stage9f_p2`
- T2 runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T3 continuation: `stage7_m5_r10`
- Full replacement: disabled
- Stage9f remains selective override only

The `current` profile and production default are not changed by this pilot.

## Implemented CLI

Module:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot
```

The pilot samples natural HU T1 states from self-play, enumerates T1 legal
actions, and evaluates each action by rolling out T2-T4 with the fixed
`stage9f_p2` continuation.

Smoke command:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot --samples 1 --future-samples 1 --max-actions 1 --seed 2026062301 --profile stage9f_p2 --opponent-profile stage9f_p2 --opening-lookahead-samples 1 --output outputs\hu_turn1_stage1_pilot\smoke_stage9f_p2_mc1.jsonl --summary-output outputs\hu_turn1_stage1_pilot\smoke_stage9f_p2_mc1_summary.json
```

Full-action speed command:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot --samples 1 --future-samples 1 --max-actions 0 --seed 2026062302 --profile stage9f_p2 --opponent-profile stage9f_p2 --opening-lookahead-samples 1 --output outputs\hu_turn1_stage1_pilot\full_actions_stage9f_p2_mc1.jsonl --summary-output outputs\hu_turn1_stage1_pilot\full_actions_stage9f_p2_mc1_summary.json
```

## Observed Results

Smoke:

- samples: `1`
- future samples: `1`
- max actions: `1`
- seconds/sample: `4.18`
- output:
  `outputs/hu_turn1_stage1_pilot/smoke_stage9f_p2_mc1.jsonl`

Full action MC1:

- samples: `1`
- legal actions: `24`
- future samples: `1`
- seconds/sample: `49.00`
- best/worst action score: `+1.0 / -6.0`
- score gap: `1.0`
- output:
  `outputs/hu_turn1_stage1_pilot/full_actions_stage9f_p2_mc1.jsonl`

Profiled full-action MC1:

- samples: `1`
- legal actions: `27`
- future samples: `1`
- seconds/sample: `59.83`
- rollout time: `59.53s`
- total policy `choose_action` time: `59.52s`
- T2 `choose_action` time: `58.50s`
- T3 `choose_action` time: `0.85s`
- T4/final time: `0.04s`
- terminal score time: `0.01s`
- T2 decisions: `54`
- TopK overrides: `0`
- output:
  `outputs/hu_turn1_stage1_pilot/profile_full_actions_stage9f_p2_mc1.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/profile_full_actions_stage9f_p2_mc1_summary.json`

GCP 10-state MC1 pilot:

- run: `regular-hu-t1-stage1-pilot-20260623-001`
- samples: `10`
- completed shards: `10 / 10`
- missing shards: `0`
- mean seconds/sample: `10.91`
- max seconds/sample: `26.79`
- mean legal actions: `26.7`
- TopK decisions: `534`
- TopK overrides: `10`
- T2 choose_action total: `97.34s`
- rollout total: `107.59s`
- merged output:
  `outputs/hu_turn1_stage1_pilot/gcp_10_mc1/hu_turn1_stage1_pilot.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/gcp_10_mc1/summary.json`

Shards `4` and `9` were restarted after Spot interruption; the final aggregate
is complete.

T2 duplicate profile, local full-action MC1:

- samples: `1`
- legal actions: `27`
- T2 decisions: `54`
- T2 choose_action time: `14.71s`
- T2 state-only duplicate rate: `0.0%`
- T2 state plus decision-seed duplicate rate: `0.0%`
- output:
  `outputs/hu_turn1_stage1_pilot/dup_profile_full_actions_stage9f_p2_mc1.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/dup_profile_full_actions_stage9f_p2_mc1_summary.json`

The previous GCP 10-state run was generated before duplicate profiling was
added, so its aggregate summary reports `summary_count = 0` for duplicate
stats. Future shards will include this field automatically.

Fast T2 continuation candidate:

- profile: `stage9f_fast_t2_t1_teacher`
- status: local smoke passed; quality unproven
- runtime: Stage8b selective override without TopK MC/confirm
- thresholds: `predicted_delta >= 2.5`, `gate >= 0.9`, `reference_margin >= 0`
- seats: `first + second`
- T3 continuation: `stage7_m5_r10`
- production status: not production, does not replace `stage9f_p2`

Local 1-state full-action MC1 comparison:

| profile | seconds/sample | T2 choose_action | best action match vs `stage9f_p2` |
|---|---:|---:|---:|
| `stage9f_p2` | `5.52` | `4.51s` | `100%` |
| `stage9f_fast_t2_t1_teacher` | `1.62` | `0.87s` | `100%` |

This is only a smoke result. The speedup is large enough to justify a 10-50
state comparison, but the fast profile is not yet validated as a T1 teacher
continuation.

Local 10-state full-action MC1 comparison:

| profile | seconds/sample | T2 choose_action total | T1 best action match | Stage9f-regret mean/max |
|---|---:|---:|---:|---:|
| `stage9f_p2` | `7.37` | `63.04s` | `100%` | `0.0 / 0.0` |
| `stage9f_fast_t2_t1_teacher` | `4.09` | `23.27s` | `100%` | `0.0 / 0.0` |

The 10-state comparison used matched states (`state_match_rate = 100%`), and
the fast profile's best action was always the Stage9f P2 best action. Overall
speedup was about `1.8x`; T2 continuation speedup was about `2.7x`.

This is still not enough for broad T1 teacher adoption. It is a Go for a
50-state comparison.

Local 50-state full-action MC1 comparison:

| profile | seconds/sample | T2 choose_action total | T1 best action match | Stage9f-regret mean/max |
|---|---:|---:|---:|---:|
| `stage9f_p2` | `8.40` | `360.46s` | `100%` | `0.0 / 0.0` |
| `stage9f_fast_t2_t1_teacher` | `2.91` | `83.61s` | `100%` | `0.0 / 0.0` |

The 50-state comparison also used matched states (`state_match_rate = 100%`).
The fast profile's best action matched Stage9f P2 in all 50 states, even though
Stage9f P2 fired `45` TopK overrides across its `2646` T2 decisions. Overall
speedup was about `2.9x`; T2 continuation speedup was about `4.3x`.

This is a Go for a 2k-5k T1 pilot teacher using
`stage9f_fast_t2_t1_teacher`. Stage9f P2 should still be reserved for selected
refinement and audit, not removed.

GCP 2k fast-profile MC1 teacher pilot:

- run: `regular-hu-t1-stage1-fast2k-20260623-001`
- profile: `stage9f_fast_t2_t1_teacher`
- opponent profile: `stage9f_fast_t2_t1_teacher`
- T3 continuation: `stage7_m5_r10`
- samples: `2000`
- completed shards: `200 / 200`
- missing shards: `0`
- mean seconds/sample: `1.80`
- max seconds/sample: `2.32`
- mean legal actions: `26.394`
- total action rows: `52788`
- actions truncated: `0`
- invalid state rows: `0`
- invalid action rows: `0`
- action count mismatches: `0`
- best action missing/not legal: `0 / 0`
- best score mismatches: `0`
- score gap mismatches: `0`
- duplicate `hand_seed:player` state keys: `0`
- seat split: `first=1000`, `second=1000`
- raw score gap mean/median/p95: `2.923 / 0.000 / 18.227`
- zero raw-gap rows: `1347`
- raw gap >= 1 rows: `653`
- distinct score gap mean/median/p95: `8.360 / 6.000 / 23.227`
- merged output:
  `outputs/hu_turn1_stage1_pilot/gcp_fast2k_mc1/hu_turn1_stage1_pilot.jsonl`
- aggregate summary:
  `outputs/hu_turn1_stage1_pilot/gcp_fast2k_mc1/summary.json`
- analysis summary:
  `outputs/hu_turn1_stage1_pilot/gcp_fast2k_mc1/analysis/summary.json`

`sample_id` is shard-local in this pilot and therefore repeats; the replay/state
identity check should use `hand_seed:player`, which had no duplicates.

This is a Go for feature-cache construction and a T1 Stage1 training smoke. It
is not a Go for production-scale T1 generation or production runtime changes.

T1 Stage1 2k training smoke:

| model | feature family | holdout top1 | holdout top3 | holdout avg regret |
|---|---|---:|---:|---:|
| `turn1_stage1_hu_fast2k_smoke_ridge.npz` | self-board | `34.75%` | `54.25%` | `9.53` |
| `turn1_stage1_hu_fast2k_smoke_torch.pt` | self-board | `22.50%` | `40.50%` | `12.72` |
| `hu_turn1_stage1_fast2k_smoke_hgb.pkl` | HU-aware | `32.25%` | `50.25%` | `10.56` |
| `hu_turn1_stage1_fast2k_smoke_extra_trees.pkl` | HU-aware | `28.75%` | `46.50%` | `11.55` |

Outputs:

- metrics dir:
  `outputs/training/hu_turn1_stage1_fast2k_smoke/`
- margin bucket metrics:
  `outputs/training/hu_turn1_stage1_fast2k_smoke/margin_bucket_metrics.csv`

This training smoke is a No-Go for model adoption. The pipeline works, but the
2k MC1 labels are too weak/noisy for a usable T1 model. ExtraTrees overfit the
train split heavily and did not improve holdout. HU-aware features did not fix
the holdout regret, so capacity alone is not the primary blocker.

T1 selected refinement targets:

- source:
  `outputs/hu_turn1_stage1_pilot/gcp_fast2k_mc1/hu_turn1_stage1_pilot.jsonl`
- selected targets:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_selected_targets.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_selected_targets_summary.json`
- metrics:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_selected_targets_metrics.csv`
- targets: `200`
- seat split: `first=107`, `second=93`
- reason counts:
  `high_model_regret=80`, `high_score_gap=60`, `low_score_gap=40`,
  `random_cover=20`, `fill_cover=17`
- mean max model regret: `25.05`
- p90 max model regret: `39.33`
- mean score gap: `9.71`

Stage9f P2 relabel smoke:

- relabeled records: `4`
- output:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_selected_targets_stage9f_p2_relabel_smoke4.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_selected_targets_stage9f_p2_relabel_smoke4_summary.json`
- analysis:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_relabel_smoke4_analysis/summary.json`
- profile: `stage9f_p2`
- future samples: `1`
- schema checks: clean
- best action changed: `4 / 4`
- source best new regret mean/max: `2.75 / 6.00`
- relabel seconds total: `109.93s`
- T2 choose_action seconds total: `106.20s`

The relabel smoke confirms the selected targets are meaningful: the fast MC1
teacher's best action changed under Stage9f P2 on all 4 smoke states. Full
200-target relabeling should be sharded. The CLI supports `--skip-targets` and
`--max-targets` for Spot VM or local parallel chunks.

Example shard command:

```powershell
python -m ofc_regular.relabel_hu_turn1_refinement_targets --input outputs\hu_turn1_stage1_pilot\refinement_targets\stage1_selected_targets.jsonl --output outputs\hu_turn1_stage1_pilot\refinement_targets\stage1_selected_targets_stage9f_p2_relabel_part00.jsonl --summary-output outputs\hu_turn1_stage1_pilot\refinement_targets\stage1_selected_targets_stage9f_p2_relabel_part00_summary.json --profile stage9f_p2 --opponent-profile stage9f_p2 --future-samples 1 --max-actions 0 --skip-targets 0 --max-targets 20 --seed 2026062602 --opening-lookahead-samples 1
```

Full 200-target Stage9f P2 relabel:

- output:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_full_relabel/stage1_selected_targets_stage9f_p2_relabel_merged.jsonl`
- aggregate summary:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_full_relabel/stage1_selected_targets_stage9f_p2_relabel_aggregate_summary.json`
- records: `200`
- best action changed: `174 / 200` (`87.0%`)
- source-best new regret mean/max: `10.00 / 37.23`
- relabel seconds total: `7121.24s`
- T2 choose_action seconds total: `5663.21s`
- seat split: `first=107`, `second=93`
- schema checks: clean

This confirms the selected states are real disagreement/high-regret targets.
The fast T2 teacher is often not the same as Stage9f P2 on these states.

Merged teacher:

- base:
  `outputs/hu_turn1_stage1_pilot/gcp_fast2k_mc1/hu_turn1_stage1_pilot.jsonl`
- refinement:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_full_relabel/stage1_selected_targets_stage9f_p2_relabel_merged.jsonl`
- merged output:
  `outputs/hu_turn1_stage1_pilot/merged_stage9f_p2_refined/hu_turn1_stage1_fast2k_plus_stage9f_p2_refined200.jsonl`
- records: `2000`
- replaced records: `200`
- label source counts:
  `base_fast_t2=1800`, `stage9f_p2_refinement=200`
- merged analysis:
  `outputs/hu_turn1_stage1_pilot/merged_stage9f_p2_refined/analysis/summary.json`
- schema checks: clean

T1 refined200 training:

| model | feature family | holdout top1 | holdout top3 | holdout avg regret | note |
|---|---|---:|---:|---:|---|
| old `turn1_stage1_hu_fast2k_smoke_ridge.npz` on merged holdout | self-board | `37.25%` | `55.00%` | `8.37` | best diagnostic baseline |
| `turn1_stage1_hu_fast2k_refined200_ridge.npz` | self-board | `34.75%` | `53.00%` | `8.78` | worse than old ridge |
| `turn1_stage1_hu_fast2k_refined200_torch.pt` | self-board | `24.50%` | `40.75%` | `11.55` | overfits early |
| `hu_turn1_stage1_fast2k_refined200_hgb.pkl` | HU-aware | `36.00%` | `54.50%` | `9.21` | worse than old ridge |
| `hu_turn1_stage1_fast2k_refined200_hgb_w5.pkl` | HU-aware weighted | `35.00%` | `56.25%` | `9.21` | top3 improves, regret does not |
| `hu_turn1_stage1_fast2k_refined200_hgb_w10.pkl` | HU-aware weighted | `37.00%` | `56.25%` | `8.64` | best weighted HGB, still worse than old ridge |
| `hu_turn1_stage1_fast2k_refined200_hgb_w20.pkl` | HU-aware weighted | `35.00%` | `54.25%` | `9.24` | overweights refined labels |

Artifacts:

- metrics dir:
  `outputs/training/hu_turn1_stage1_fast2k_refined200/`
- margin bucket metrics:
  `outputs/training/hu_turn1_stage1_fast2k_refined200/margin_bucket_metrics.csv`
- source bucket metrics:
  `outputs/training/hu_turn1_stage1_fast2k_refined200/source_bucket_metrics.csv`

Decision: refined200 is a No-Go for T1 model adoption. The relabel targets are
valuable, but 200 strong labels mixed into 1,800 fast labels did not beat the
old self-board ridge baseline on the same merged holdout. Source-weighted HGB
shows that weighting can help, but the best weighted run still has higher
holdout regret than the old ridge baseline.

## Interpretation

The T1 teacher path works, but direct all-action rollout through Stage9f P2 is
too slow for large local generation. MC1 costs about `49-60s/state` on the
first full-action local samples. On GCP `e2-highcpu-4` Spot shards, the 10-state
MC1 pilot averaged `10.91s/state`, which is usable for small profiling but still
expensive for broad T1 teacher generation.

The bottleneck is not legal action generation, T3, final-turn exact search, or
terminal scoring. The bottleneck is T2 continuation:

```text
T2 choose_action: 58.50s / 59.83s
```

This is expected because each T1 action rollout invokes Stage9f TopK + confirm
at T2. The first duplicate-profile sample found no exact T2 state reuse, so
simple memoization is unlikely to be enough. For large T1 teacher generation,
either this path must be sharded heavily across Spot VMs or the T2 continuation
must be batched/distilled.

## Next Step

Do not start production-scale T1 generation yet.

Recommended next work:

1. Do not adopt the refined200 T1 models.
2. Generate a larger Stage9f P2 refined label set, or redesign the distilled
   objective so strong labels have enough influence without overfitting.
3. Keep the old self-board ridge as the diagnostic baseline until a new model
   beats it on the same heldout split and source buckets.
4. Only if holdout avg_regret drops materially below the old ridge baseline,
   run a small T1 seat-swap with the T1 candidate.
5. Production-scale T1 and T1 runtime adoption remain No-Go.

## Full 2k Stage9f P2 Relabel And Runtime Smoke

Full remaining-target relabel:

- additional input:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_remaining_targets_1800.jsonl`
- output dir:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_remaining1800_relabel_throttle6/`
- merged output:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_remaining1800_relabel_throttle6/stage1_remaining_targets_stage9f_p2_relabel_merged.jsonl`
- records: `1800`
- failed sessions: `0`
- best action changed: `1576 / 1800` (`87.56%`)
- source-best new-regret mean/max: `11.44 / 60.45`

Combined full2k relabel:

- output:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_full2k_relabel/stage1_all2k_targets_stage9f_p2_relabel_merged.jsonl`
- records: `2000`
- unique state keys: `2000`
- best action changed: `1750 / 2000` (`87.50%`)
- source-best new-regret mean/max: `11.30 / 60.45`
- seat split: `first=1000`, `second=1000`

Merged teacher:

- output:
  `outputs/hu_turn1_stage1_pilot/merged_stage9f_p2_full2k/hu_turn1_stage1_fast2k_stage9f_p2_relabel_full2k.jsonl`
- records: `2000`
- replaced records: `2000`
- label source counts: `stage9f_p2_refinement=2000`
- analysis summary:
  `outputs/hu_turn1_stage1_pilot/merged_stage9f_p2_full2k/analysis/summary.json`
- schema checks: clean
- score gap mean/median/p95: `3.18 / 0.00 / 19.01`
- distinct score gap mean/median/p95: `8.20 / 6.00 / 23.23`

Full2k training and sweep:

| model | holdout top1 | holdout top3 | holdout avg regret | note |
|---|---:|---:|---:|---|
| old self ridge, evaluated on full2k labels | `32.00%` | `52.50%` | `9.90` | old diagnostic baseline |
| old HU HGB, evaluated on full2k labels | `31.75%` | `47.75%` | `9.66` | old HU baseline |
| first new ridge | `31.00%` | `50.00%` | `10.12` | No-Go |
| first new HU HGB | `29.25%` | `45.75%` | `10.30` | No-Go |
| best HGB sweep4 `leaf3/lr0.01/l2=1` | `41.50%` | `46.00%` | `8.00` | best holdout regret |

Cross-seed validation for `leaf3/lr0.01/l2=1`:

- model:
  `models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl`
- output:
  `outputs/training/hu_turn1_stage1_stage9f_p2_full2k/cross_seed/cross_seed_model_comparison.csv`
- candidate mean avg regret across seeds: `7.90`
- old HU HGB mean avg regret across seeds: `9.82`
- old self ridge mean avg regret across seeds: `9.97`
- candidate minus old HU avg regret by seed:
  `-1.66`, `-1.97`, `-3.18`, `-0.53`, `-2.24`

Runtime integration:

- added profile: `stage9f_p2_hu_t1_stage1`
- candidate T1 model:
  `models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl`
- T2/T3 continuation: same as `stage9f_p2`
- T1 candidate is selective, not full replacement
- current T1 predicted-margin threshold:
  `DEFAULT_HU_TURN1_STAGE1_MIN_MARGIN = 1.0`

Runtime smoke results:

| profile A | profile B | games | seed stride | T1 behavior | EV/hand A | 95% CI | decision |
|---|---|---:|---:|---|---:|---:|---|
| `stage9f_p2_hu_t1_stage1` | `stage9f_p2` | `100` | `1009` | full replacement before margin gate | `-0.6268` | `[-1.8922, +0.6386]` | No-Go |
| `stage9f_p2_hu_t1_stage1` | `stage9f_p2` | `100` | `1009` | margin `5.0` | `0.0000` | `[0.0000, 0.0000]` | underfire, no effect |
| `stage9f_p2_hu_t1_stage1` | `stage9f_p2` | `100` | `1009` | margin `1.0` | `-0.2895` | `[-1.0207, +0.4417]` | No-Go smoke |

T1 margin distribution on the same 100 paired seeds:

- T1 states inspected: `200`
- HU candidate differs from self-board baseline: `135 / 200` (`67.5%`)
- margin p50/p75/p90/p95/max:
  `0.00 / 0.70 / 1.22 / 1.46 / 2.34`
- fire counts by threshold:
  `m0=97`, `m0.5=71`, `m1=35`, `m2=4`, `m3=0`, `m5=0`

Decision:

- The full2k relabel and HGB sweep are a real holdout improvement.
- The first runtime seat-swap smoke did not convert that holdout improvement
  into EV.
- T1 runtime adoption remains No-Go.
- Do not run a larger T1 seat-swap until the runtime objective is redesigned or
  the candidate is evaluated with a stronger conditional/per-action diagnostic.
- The next useful T1 work is to build a selective override training/evaluation
  target directly around realized T1 action deltas, not to promote the current
  HGB model as a full or margin-only replacement.

T1 decision-log audit support:

- policy field: `hu_turn1_decision_log`
- evaluation CLI:
  `python -m ofc_regular.evaluate_matchups --hu-turn1-decision-output <jsonl>`
- analyzer:
  `python -m ofc_regular.analyze_hu_turn1_decision_log`

The evaluator now attaches opposite-seat-swap realized deltas to T1 decision
rows, the same way TopK diagnostics are handled. This makes the primary metric
the realized fired-decision delta. The T1 predicted margin is a gate diagnostic,
not performance evidence.

Margin-1 audit:

- decision log:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_smoke_100_margin1_hu_t1.jsonl`
- analysis dir:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_margin1/`
- rows: `200`
- valid rows: `163`
- fired valid rows: `29`
- non-fired valid rows: `134`
- override rate on valid: `17.79%`
- fired realized delta mean: `-2.1003`
- fired realized delta 95% CI: `[-7.0926, +2.8920]`
- fired losses/wins/zeros: `12 / 6 / 11`
- non-fired nonzero count: `0`
- hard-negative targets:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_margin1/hu_turn1_hard_negative_targets.jsonl`
- hard-negative rows: `12`

Updated decision:

- The T1 logging/evaluation path is now usable.
- The current `stage9f_p2_hu_t1_stage1` model is still No-Go as runtime.
- The next model iteration should use the 12 hard negatives plus future
  positive fired rows to train a deployable safe-override head or selector.

500-paired-seed T1 decision-log run:

- matchup output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_500_margin1_t1log.json`
- T1 decision log:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_500_margin1_hu_t1.jsonl`
- analysis dir:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_500_margin1/`
- paired seeds / hands: `500 / 1000`
- full-hand EV/hand: `-0.4247`
- 95% CI: `[-0.7863, -0.0631]`
- T1 decision rows / valid rows: `1000 / 803`
- fired valid rows: `149`
- override rate on valid T1 rows: `18.56%`
- fired realized delta mean: `-2.0506`
- fired realized delta 95% CI: `[-4.1191, +0.0179]`
- fired losses/wins/zeros: `54 / 37 / 58`
- non-fired nonzero count: `0`
- labeled fired targets:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_500_margin1/hu_turn1_fired_labeled_targets.jsonl`
- positive targets: `37`
- hard-negative targets: `54`
- neutral targets: `58`

Margin bucket read:

| margin bucket | rows | mean realized delta | losses | wins | zeros |
|---|---:|---:|---:|---:|---:|
| `[1,1.25)` | `38` | `-6.4078` | `14` | `4` | `20` |
| `[1.25,1.5)` | `32` | `-2.5213` | `11` | `4` | `17` |
| `[1.5,1.75)` | `20` | `-0.0114` | `8` | `4` | `8` |
| `[1.75,2)` | `20` | `-1.6227` | `9` | `7` | `4` |
| `[2,3)` | `31` | `+0.8167` | `10` | `15` | `6` |
| `[3,5)` | `8` | `+3.2500` | `2` | `3` | `3` |

Decision after 500 paired seeds:

- `hu_turn1_min_margin=1.0` is a clear runtime No-Go.
- Low margin bands are strongly negative.
- Higher margins (`>=2`) may contain useful signal, but the sample is too small
  and still includes losses.
- Do not tune by prediction margin alone. Train a safe-override classifier or
  selector using the labeled fired rows, then validate with the same
  realized-fired-delta audit.

3000-paired-seed T1 decision-log run:

- matchup output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_3000_margin1_t1log.json`
- T1 decision log:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_3000_margin1_hu_t1.jsonl`
- analysis dir:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_3000_margin1/`
- paired seeds / hands: `3000 / 6000`
- full-hand EV/hand: `-0.4540`
- 95% CI: `[-0.6003, -0.3078]`
- T1 decision rows / valid rows: `6000 / 4777`
- fired valid rows: `963`
- override rate on valid T1 rows: `20.16%`
- fired realized delta mean: `-2.1835`
- fired realized delta 95% CI: `[-2.9724, -1.3945]`
- fired losses/wins/zeros: `326 / 234 / 403`
- non-fired nonzero count: `0`
- labeled fired targets:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_3000_margin1/hu_turn1_fired_labeled_targets.jsonl`
- positive targets: `234`
- hard-negative targets: `326`
- neutral targets: `403`

3000-run margin bucket read:

| margin bucket | rows | mean realized delta | 95% CI | losses | wins | zeros |
|---|---:|---:|---:|---:|---:|---:|
| `[1,1.25)` | `271` | `-1.3988` | `[-2.8605, +0.0628]` | `82` | `69` | `120` |
| `[1.25,1.5)` | `281` | `-2.8701` | `[-4.2831, -1.4571]` | `94` | `58` | `129` |
| `[1.5,1.75)` | `127` | `-2.3242` | `[-4.3444, -0.3041]` | `40` | `26` | `61` |
| `[1.75,2)` | `98` | `-0.2416` | `[-2.9325, +2.4492]` | `30` | `30` | `38` |
| `[2,3)` | `142` | `-3.7074` | `[-5.9242, -1.4906]` | `61` | `36` | `45` |
| `[3,5)` | `43` | `-1.3393` | `[-5.1465, +2.4680]` | `18` | `15` | `10` |
| `>=5` | `1` | `-14.2270` | `[-14.2270, -14.2270]` | `1` | `0` | `0` |

Safe-override selector smoke on the 3000-run fired labels:

- trainer:
  `python -m ofc_regular.train_hu_turn1_safe_override_selector`
- split fix:
  label-stratified deterministic train/val/test split was added because the
  500-run hash split put zero positives in validation.
- labeled rows: `963` with neutral, `560` excluding neutral.
- best validation AP with neutral:
  `0.3693` (`delta_plus_meta` ExtraTrees), `0.3321`
  (`candidate_delta_plus_meta` ExtraTrees), `0.2771` (`meta_only` ExtraTrees).
- best validation AP excluding neutral:
  `0.5077` (`delta_plus_meta` HGB), `0.4645`
  (`candidate_delta_plus_meta` HGB), `0.4592` (`meta_only` HGB).
- threshold transfer to test remains weak. The best sparse positive-looking
  runs fire only `1-2` test rows; broader thresholds are negative on test.

Decision after 3000 paired seeds:

- `stage9f_p2_hu_t1_stage1` with `hu_turn1_min_margin=1.0` is a confirmed
  runtime No-Go.
- Prediction margin is not a usable safety gate; even `margin >= 2` is negative
  on the 3000-run fired audit.
- The simple post-hoc safe selector trained on realized fired labels is not
  strong enough for runtime use. It shows weak out-of-sample ranking and sparse
  threshold transfer.
- T1 runtime adoption remains No-Go.
- The next useful T1 work is not threshold tuning. It should improve the
  teacher/objective: more reliable T1 labels, direct safe-override targets with
  independent replay, or a redesigned candidate generator that optimizes
  realized per-decision delta rather than holdout top1 alone.

Replay-ready decision-log and relabel pipeline:

- T1 decision records now include replay context when produced through
  `evaluate_matchups.trace_hand`:
  `visibility_model`, `discard_visibility`, `true_dead_cards`,
  `visible_dead_cards`, `hero_private_discards`, `opponent_private_discards`,
  and `replay_ready`.
- Legacy decision logs without these fields must remain replay-ineligible for
  exact relabeling. They are still useful for aggregate fired-delta diagnostics.
- extractor:
  `python -m ofc_regular.extract_hu_turn1_decision_relabel_targets`
- default behavior:
  only replay-ready fired positive / hard-negative rows are converted.
- target schema:
  `hu_turn1_stage1_decision_relabel_target_v1`
- the extractor writes the runtime candidate as source `best_action=0`, so
  `relabel_hu_turn1_refinement_targets` can report whether the stronger teacher
  changes the runtime choice and how much regret the runtime candidate has.

Replay-ready smoke:

- matchup output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_20_margin1_replay_ready_t1log.json`
- decision log:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_20_margin1_replay_ready_hu_t1.jsonl`
- analysis dir:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_20_margin1_replay_ready/`
- paired seeds / hands: `20 / 40`
- fired valid rows: `7`
- extracted replay-ready relabel targets: `4`
  (`hard_negative=3`, `positive=1`)
- Stage9f P2 relabel smoke:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_20_margin1_replay_ready/relabel_targets_stage9f_p2_smoke2.jsonl`
- relabeled records: `2`
- best action changed: `2 / 2`
- runtime candidate new-regret mean/max: `6.0 / 6.0`

Updated next step:

1. Generate a larger replay-ready T1 decision log with the current margin1
   candidate, or a broader candidate generator if available.
2. Extract fired positive / hard-negative replay targets with
   `extract_hu_turn1_decision_relabel_targets`.
3. Relabel these targets with Stage9f P2, sharded on Spot VM if needed.
4. Train the next T1 model/objective from the stronger relabels, prioritizing
   runtime candidate regret and safe override labels over holdout top1 alone.
5. Only then run another T1 runtime seat-swap.

Replay-ready 1000-paired audit and balanced relabel:

- matchup output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_1000_margin1_replay_ready_t1log.json`
- decision log:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_1000_margin1_replay_ready_hu_t1.jsonl`
- analysis dir:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/`
- paired seeds / hands: `1000 / 2000`
- full-hand EV/hand: `-0.7859`
- 95% CI: `[-1.0535, -0.5182]`
- valid T1 rows: `1592`
- fired valid rows: `322`
- fired realized delta mean: `-3.6012`
- fired realized delta 95% CI: `[-4.9796, -2.2227]`
- fired losses/wins/zeros: `125 / 71 / 126`
- non-fired nonzero count: `0`

Balanced replay target extraction:

- target output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_balanced100.jsonl`
- summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_balanced100_summary.json`
- written targets: `100`
- label counts: `hard_negative=50`, `positive=50`
- seat split: `first=45`, `second=55`

Stage9f P2 relabel of balanced100:

- output dir:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_balanced100_full/`
- merged output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_balanced100_full/relabel_targets_stage9f_p2_merged.jsonl`
- aggregate summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_balanced100_full/relabel_targets_stage9f_p2_aggregate_summary.json`
- records: `100`
- best action changed: `83 / 100`
- runtime candidate new-regret mean/max: `11.88 / 42.45`
- by label:
  - `runtime_hard_negative`: changed `41 / 50`, regret mean/max `11.94 / 42.45`
  - `runtime_positive`: changed `42 / 50`, regret mean/max `11.81 / 30.23`
- relabel seconds total: `1827.31`
- T2 choose-action seconds total: `1541.29`

Interpretation:

- The replay-ready pipeline is working and produces usable stronger-teacher
  labels.
- The margin1 T1 runtime candidate is worse than the 3000-run audit suggested
  on this seed range, with strongly negative whole-hand and fired-delta results.
- Runtime positive fired decisions are not reliable teacher positives. In the
  balanced100 relabel, `runtime_positive` rows were rejected by Stage9f P2 about
  as often as hard negatives.
- Therefore, do not train the next T1 model directly on realized positive vs
  hard-negative labels. Use the Stage9f P2 relabel outputs as the stronger
  action-value target, and treat realized fired delta as a sampling signal only.

Next larger step:

1. Collect a larger replay-ready T1 decision log only if more runtime-sampled
   targets are needed.
2. For training, prioritize Stage9f P2 relabeled targets over raw realized-delta
   labels.
3. A useful next dataset is `500-1000` replay-ready fired targets, balanced by
   runtime label and seat, then Stage9f P2 relabeled on Spot VM.
4. Do not run T1 seat-swap again until a model improves on Stage9f P2 relabeled
   holdout regret and has a deployable selective objective.

Runtime relabel100 augmented diagnostic:

- augment CLI:
  `python -m ofc_regular.augment_hu_turn1_teacher`
- augmented teacher:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_runtime_relabel100/hu_turn1_stage1_full2k_plus_runtime_relabel100.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_runtime_relabel100/summary.json`
- records: `2100`
  (`stage9f_p2_refinement=2000`, `stage9f_p2_runtime_relabel=100`)
- duplicate state keys: `0`
- trainer evaluation fix:
  HU top3 now uses stable descending sort; tied predictions can no longer make
  top3 lower than top1.

HGB `leaf3/lr0.01/l2=1` with runtime relabel source weights:

| model | runtime relabel weight | holdout top1 | holdout top3 | holdout avg regret |
|---|---:|---:|---:|---:|
| `w1` | `1` | `44.05%` | `62.62%` | `7.94` |
| `w5` | `5` | `50.71%` | `64.29%` | `6.87` |
| `w10` | `10` | `64.52%` | `74.29%` | `4.48` |

Runtime smoke, 100 paired seeds, same seed range:

| model | T1 margin | EV/hand | 95% CI | fired | fired delta mean | decision |
|---|---:|---:|---:|---:|---:|---|
| `w10` | `1.0` | `-0.8568` | `[-1.6705, -0.0431]` | `22` | `-8.38` | No-Go |
| `w5` | `1.0` | `-0.7368` | `[-1.6492, +0.1756]` | `24` | `-4.29` | No-Go |
| `w10` | `1.5` | `-0.3173` | `[-0.7693, +0.1348]` | `8` | `-7.93` | No-Go |
| `w5` | `1.5` | `-0.0261` | `[-0.2774, +0.2251]` | `8` | `-0.65` | not positive |
| `w10` | `2.0` | `0.0000` | `[0.0000, 0.0000]` | `1` | `0.00` | underfire |
| `w5` | `2.0` | `+0.0300` | `[-0.0288, +0.0888]` | `2` | `+3.00` | underfire |

Interpretation:

- The balanced100 Stage9f P2 relabels are useful diagnostic data and can move
  holdout action-value metrics a lot.
- They still do not produce a deployable T1 margin gate. `margin1` fires bad
  overrides, `margin2` mostly does nothing, and `margin1.5` is not convincingly
  positive.
- This confirms the next T1 step is not another small weighted HGB sweep. The
  objective/gate needs to predict safe override probability or use a stronger
  search teacher; current T1 runtime remains No-Go.

Runtime relabel196 expansion:

- all replay-ready fired nonzero targets from the 1000-paired decision log:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_all_fired_nonzero.jsonl`
- summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_all_fired_nonzero_summary.json`
- records: `196`
  (`hard_negative=125`, `positive=71`)
- seat split: `first=97`, `second=99`
- Stage9f P2 relabel output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_all196_full/relabel_targets_stage9f_p2_merged.jsonl`
- aggregate summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_all196_full/relabel_targets_stage9f_p2_aggregate_summary.json`
- relabeled records: `196`
- best action changed: `183 / 196` (`93.37%`)
- runtime candidate new-regret mean/max: `13.40 / 41.23`
- by runtime label:
  - `runtime_hard_negative`: changed `120 / 125`, regret mean/max
    `13.82 / 41.23`
  - `runtime_positive`: changed `63 / 71`, regret mean/max
    `12.67 / 36.23`

Augmented 2196-row teacher:

- output:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_runtime_relabel196/hu_turn1_stage1_full2k_plus_runtime_relabel196.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_runtime_relabel196/summary.json`
- records:
  `stage9f_p2_refinement=2000`, `stage9f_p2_runtime_relabel=196`

HGB `leaf3/lr0.01/l2=1` with runtime relabel196 source weights:

| model | runtime relabel weight | holdout top1 | holdout top3 | holdout avg regret |
|---|---:|---:|---:|---:|
| `w3` | `3` | `49.89%` | `62.87%` | `7.04` |
| `w5` | `5` | `50.57%` | `63.55%` | `6.76` |
| `w10` | `10` | `49.66%` | `62.64%` | `6.76` |

Runtime smoke for `w5`, 100 paired seeds:

| T1 margin | EV/hand | 95% CI | fired | fired delta mean | decision |
|---:|---:|---:|---:|---:|---|
| `1.0` | `-0.4434` | `[-0.9544, +0.0676]` | `16` | `-4.92` | No-Go |
| `1.5` | `-0.0150` | `[-0.1777, +0.1477]` | `7` | `-0.43` | not positive |
| `2.0` | `-0.0550` | `[-0.1628, +0.0528]` | `2` | `-5.50` | underfire/No-Go |

Interpretation:

- Expanding the replay-ready runtime relabel set from 100 to 196 makes the
  teacher signal stronger, not weaker: even runtime-positive fired rows are
  usually rejected by the Stage9f P2 teacher.
- The 2196-row weighted HGB models still do not produce a useful deployable T1
  selective override.
- T1 should now pivot away from margin-only action-value gating. The next
  useful experiment is a direct safe-override gate/head or stronger search
  teacher for T1, with runtime fire decisions judged by fired whole-game delta.

Teacher-regret safe gate diagnostic:

- input:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_all196_full/relabel_targets_stage9f_p2_merged.jsonl`
- label mode: `teacher_regret`
- positive label:
  `source_best_new_regret <= 0.25`
- rows: `196`
- positives / negatives: `40 / 156`
- regret distribution:
  - `<=0.25`: `40`
  - `<=1`: `42`
  - `<=2`: `48`
  - `<=5`: `59`
  - median: `12.0`
  - p95: `32.23`
  - max: `41.23`

Best diagnostic selector:

- model:
  `models/hu_turn1_safe_override_selector_teacher_regret_all196_accept0p25_gray1_delta_meta.pkl`
- output dir:
  `outputs/training/hu_turn1_safe_override_selector_teacher_regret_all196/accept0p25_gray1_delta_meta/`
- feature mode: `delta_plus_meta`
- best estimator: `logistic_balanced`
- val AP / AUC: `0.4901 / 0.6739`
- test AP / AUC: `0.2356 / 0.3696`
- heldout threshold behavior:
  - val threshold `0.9`: fires `2`, precision `0.50`,
    teacher regret mean/max `9.11 / 18.23`
  - test threshold `0.9`: fires `3`, precision `0.33`,
    teacher regret mean/max `17.15 / 27.23`

Interpretation:

- The direct safe-gate objective is the right kind of target, but `196` rows is
  not enough. The selector memorizes train rows and still fires high-regret
  rows on heldout.
- Do not integrate this gate into runtime.
- Next useful data step: collect/relabel substantially more diverse T1 fired
  decisions, or generate T1 teacher states directly from self-play rather than
  only from a bad margin1 runtime candidate.

Non-fired candidate relabel expansion:

- extractor:
  `python -m ofc_regular.extract_hu_turn1_decision_relabel_targets`
- new extractor mode:
  `--only-non-fired-candidates --min-predicted-margin 0.5`
- output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_non_fired_margin0p5_top100.jsonl`
- summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_non_fired_margin0p5_top100_summary.json`
- records: `100`
- seat split: `first=57`, `second=43`
- Stage9f P2 relabel output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_non_fired_margin0p5_top100/relabel_targets_stage9f_p2_merged.jsonl`
- relabeled records: `100`
- best action changed: `93 / 100`
- runtime candidate new-regret mean/max: `12.86 / 41.45`
- teacher-accepted rows:
  - `<=0.25`: `19`
  - `<=1`: `19`
  - `<=2`: `27`
  - `<=5`: `35`

Combined fired/non-fired teacher-regret safe gate diagnostic:

- combined relabel output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_combined296/relabel_targets_stage9f_p2_merged.jsonl`
- combined summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_combined296/summary.json`
- records: `296`
- teacher-accepted rows:
  - `<=0.25`: `59`
  - `<=1`: `61`
  - `<=2`: `75`
  - `<=5`: `94`
  - `<=10`: `141`
- source split:
  - fired runtime decisions: `196`, `<=0.25` rows `40`, regret mean `13.40`
  - non-fired candidate decisions: `100`, `<=0.25` rows `19`, regret mean
    `12.86`

Safe selector training on combined296:

| label | feature mode | val AP | test AP | heldout threshold result |
|---|---|---:|---:|---|
| `accept0.25/gray1` | `delta_plus_meta` | `0.3714` | `0.2295` | No-Go: test threshold `0.9` fires `4`, precision `0.25`, regret mean/max `14.36 / 25.23` |
| `accept0.25/gray1` | `candidate_delta_plus_meta` | `0.3060` | weaker | No-Go |
| `accept0.25/gray1` | `meta_only` | `0.3057` | weaker | No-Go |
| `accept2/gray5` | `delta_plus_meta` | `0.4124` | weak | No-Go: test threshold `0.9` fires `9`, `<=2` precision `0.11`, regret mean/max `15.21 / 36.23` |

Interpretation:

- Non-fired near-boundary candidates are useful data: `19 / 100` are accepted
  by the Stage9f P2 teacher at regret `<=0.25`.
- Most non-fired candidates are still high-regret, so this source must be
  treated as hard-negative-rich data, not as easy positives.
- Combining fired and non-fired relabels improves coverage but still does not
  produce a deployable safe selector. The heldout splits keep firing high-regret
  rows at high selector probabilities.
- Next useful step is a larger, stratified replay-ready T1 relabel set with both
  fired and non-fired candidate rows, not another runtime seat-swap or a
  margin-only threshold sweep.

Stratified670 relabel expansion:

- fired targets:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_fired_all322.jsonl`
- non-fired targets:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_non_fired_margin0p5_all.jsonl`
- combined target file:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/relabel_targets_stratified670/relabel_targets_stratified670.jsonl`
- Stage9f P2 relabel output:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_stratified670_full/relabel_targets_stage9f_p2_merged.jsonl`
- aggregate summary:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_1000_margin1_replay_ready/stage9f_p2_relabel_stratified670_full/relabel_targets_stage9f_p2_aggregate_summary.json`
- records: `670`
  - fired all labels: `322`
  - non-fired margin `>=0.5`: `348`
- Stage9f P2 changed runtime/source action: `598 / 670` (`89.25%`)
- source candidate regret mean/max: `11.77 / 53.45`
- teacher-accepted rows:
  - `<=0.25`: `172`
  - `<=1`: `178`
  - `<=2`: `214`
  - `<=5`: `260`
  - `<=10`: `360`
- by source:
  - fired all labels: `80 / 322` at regret `<=0.25`, regret mean `12.51`
  - non-fired margin `>=0.5`: `92 / 348` at regret `<=0.25`,
    regret mean `11.09`

Safe selector training on stratified670:

| label | feature mode | val AP | test AP | heldout threshold result |
|---|---|---:|---:|---|
| `accept0.25/gray1` | `delta_plus_meta` | `0.3138` | `0.3011` | No-Go: test threshold `0.9` fires `31`, precision `0.323`, regret mean/max `12.99 / 36.23` |
| `accept0.25/gray1` | `candidate_delta_plus_meta` | `0.3421` | `0.3269` | underfires at high threshold; threshold `0.3` still regret mean/max `14.27 / 30.23` |
| `accept2/gray5` | `delta_plus_meta` | `0.3676` | `0.3464` | No-Go: test threshold `0.9` fires `29`, `<=2` precision `0.207`, regret mean/max `13.51 / 36.23` |

Interpretation:

- Larger stratified relabeling confirms the runtime candidate action is usually
  not teacher-safe, even when it did not fire at margin `1.0`.
- The current post-hoc safe selector feature set cannot separate
  teacher-accepted rows from high-regret rows. Adding more rows from the same
  margin1 runtime distribution did not solve generalization.
- Do not run a T1 seat-swap with this selector. Do not promote any T1 runtime
  gate from these artifacts.
- The next useful T1 work should change the candidate generator/objective, not
  simply add another selector threshold. Candidate directions: train a T1
  action-value model on stratified relabel rows plus the full2k base, add richer
  teacher-derived features, or collect T1 states from broader self-play rather
  than only from the bad margin1 runtime distribution.

Stratified670 action-value candidate generator:

- augmented teacher:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_stratified670/hu_turn1_stage1_full2k_plus_stratified670.jsonl`
- summary:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_stratified670/summary.json`
- records:
  - full2k Stage9f P2 refinement: `2000`
  - stratified670 runtime relabel: `670`
  - total: `2670`

HGB `leaf3/lr0.01/l2=1`, stratified source weights:

| model | stratified weight | holdout top1 | holdout top3 | holdout avg regret |
|---|---:|---:|---:|---:|
| `w1` | `1` | `45.51%` | `61.42%` | `7.81` |
| `w3` | `3` | `44.01%` | `60.11%` | `8.12` |
| `w5` | `5` | `50.19%` | `63.11%` | `6.97` |
| `w10` | `10` | `50.56%` | `66.48%` | `6.66` |
| `w20` | `20` | `54.12%` | `70.97%` | `5.98` |

Source split for `w20`:

- full2k refinement holdout: top1 `54.31%`, top3 `71.07%`,
  avg regret `5.91`
- stratified670 holdout: top1 `53.57%`, top3 `70.71%`,
  avg regret `6.19`

Runtime smoke for `w20`:

| margin | paired seeds | EV/hand | 95% CI | fired | fired delta mean | decision |
|---:|---:|---:|---:|---:|---:|---|
| `1.0` | `100` | `+0.2000` | `[-0.1127, +0.5127]` | `18` | `+1.61` | promising smoke only |
| `1.5` | `100` | `-0.0111` | `[-0.3054, +0.2832]` | `8` | `-0.28` | No-Go |
| `2.0` | `100` | `-0.1161` | `[-0.2745, +0.0422]` | `3` | `-7.74` | No-Go |
| `1.0` | `500` | `-0.0870` | `[-0.3196, +0.1456]` | `84` | `-0.93` | No-Go |

Interpretation:

- As an offline candidate generator, stratified670 `w20` is a real improvement.
- The improvement still does not transfer to a deployable margin gate; the
  500-paired fired delta is negative.
- Margin bucket ordering is not reliable: higher margin does not imply safer
  T1 override on this model.

Tail84 feedback iteration:

- runtime fired targets from `w20 margin1`:
  `outputs/evals/hu_turn1_stage1_stratified670_w20/relabel_targets_w20_margin1_fired_all84.jsonl`
- Stage9f P2 relabel:
  `outputs/evals/hu_turn1_stage1_stratified670_w20/stage9f_p2_relabel_w20_margin1_fired_all84/relabel_targets_stage9f_p2_merged.jsonl`
- records: `84`
- teacher changed runtime/source action: `77 / 84` (`91.67%`)
- runtime fired delta mean: `-0.93`
- source candidate regret mean/max: `10.89 / 32.23`
- teacher-accepted rows:
  - `<=0.25`: `18`
  - `<=2`: `24`
  - `<=5`: `31`

Tail84-weighted action-value training:

- augmented teacher:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_stratified670_plus_w20_margin1_fired84/hu_turn1_stage1_full2k_plus_stratified670_plus_w20_margin1_fired84.jsonl`
- records:
  - full2k Stage9f P2 refinement: `2000`
  - stratified670 runtime relabel: `670`
  - w20 margin1 fired relabel: `84`
  - total: `2754`

| model | stratified weight | tail84 weight | holdout top1 | holdout top3 | holdout avg regret |
|---|---:|---:|---:|---:|---:|
| `w20_20` | `20` | `20` | `47.55%` | `66.24%` | `7.03` |
| `w20_50` | `20` | `50` | `50.09%` | `66.79%` | `6.64` |
| `w20_100` | `20` | `100` | `55.54%` | `68.78%` | `5.57` |

Runtime smoke for `w20_100`:

| margin | paired seeds | EV/hand | 95% CI | fired | fired delta mean | decision |
|---:|---:|---:|---:|---:|---:|---|
| `0.5` | `100` | `-0.2461` | `[-0.6369, +0.1446]` | `13` | `-4.71` | No-Go |
| `1.0` | `100` | `+0.0600` | `[-0.0576, +0.1776]` | `3` | `0.00` | underfire |
| `1.5` | `100` | `0.0000` | `[0.0000, 0.0000]` | `2` | `0.00` | underfire |
| `1.0` | `500` | `-0.1734` | `[-0.4064, +0.0597]` | `53` | `-3.27` | No-Go |

Interpretation:

- Tail84 weighting improves offline holdout again, but does not fix runtime
  conversion.
- Lowering margin to recover fire rate reintroduces large losses.
- T1 margin-gated selective override remains No-Go even with a better
  action-value candidate generator.
- Next step should be a different runtime objective: either an explicit T1
  search/rerank stage with independent confirmation, or a teacher-generated
  candidate policy trained from broader self-play states. More margin-only HGB
  and post-hoc selector sweeps are not the right direction.

T1 TopK + independent confirm runtime path:

- implementation status: post-fix plumbing passed; quality still unproven
- runtime profile:
  `stage9f_p2_hu_t1_topk_confirm`
- candidate model:
  `models/hu_turn1_stage1_stage9f_p2_full2k_plus_stratified670_tail84_w20_100_hgb_leaf3_lr01_l2_1.pkl`
- continuation:
  - T2: `stage9f_p2`
  - T3: `stage7_m5_r10`

Design:

- The T1 HU model is used only as a TopK candidate generator.
- Stage A evaluates topK candidates plus baseline with common-random-future
  MC rollouts.
- Stage B independently re-evaluates the Stage A champion against baseline
  using a separate confirm seed.
- Confirm delta is a gate diagnostic only. Strength must be judged by realized
  fired seat-swap delta from the matchup harness.

Important implementation fix:

- The first smoke implementation used the correct T1 TopK confirm code, but the
  actual gameplay policy did not inherit the T2 Stage9f P2 continuation.
- That produced a pre-fix `50` paired run before the continuation fix, so
  `outputs/evals/hu_turn1_topk_confirm_sweep_small/summary_50_mc2_confirm4_cse1.json`
  is obsolete for strength decisions.
- The old `non_fired_nonzero_count` field counts candidate counterfactual
  deltas, not actual final-action mismatches. Use
  `non_fired_final_mismatch_count` for cancellation checks.
- The profile now inherits the T2 Stage9f P2 policy, and the T1 helper no
  longer shadows the T2 rerank method.

Post-fix small sweep:

| config | paired seeds | EV/hand | 95% CI | T1 decisions | valid fired | fired delta mean | fired delta 95% CI | non-fired nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `k3/mc2/d0/confirm4/cse1/pd0/seat=first+second` | `10` | `+0.5000` | `[-1.2286, +2.2286]` | `20` | `6` | `+1.6667` | `[-4.2495, +7.5829]` | `0` |
| `k3/mc2/d0/confirm4/cse1.5/pd0/seat=first+second` | `10` | `-0.6614` | `[-3.7943, +2.4716]` | `20` | `3` | `-4.4090` | `[-28.1322, +19.3142]` | `0` |
| `k3/mc2/d0/confirm4/cse2/pd0/seat=first+second` | `10` | `0.0000` | `[0.0000, 0.0000]` | `20` | `0` | `0.0000` | `[0.0000, 0.0000]` | `0` |

Post-fix artifacts:

- `outputs/evals/hu_turn1_topk_confirm_sweep_small/summary_10_mc2_confirm4_cse1_after_t2fix.json`
- `outputs/evals/hu_turn1_topk_confirm_sweep_small/decisions_10_mc2_confirm4_cse1_after_t2fix.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_sweep_small/summary_10_mc2_confirm4_cse1p5_after_t2fix.json`
- `outputs/evals/hu_turn1_topk_confirm_sweep_small/decisions_10_mc2_confirm4_cse1p5_after_t2fix.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_sweep_small/summary_10_mc2_confirm4_cse2_after_t2fix.json`
- `outputs/evals/hu_turn1_topk_confirm_sweep_small/decisions_10_mc2_confirm4_cse2_after_t2fix.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_sweep_small/analysis_after_t2fix/hu_turn1_topk_confirm_summary.csv`

Extended `cse1/d0` audit:

| run | paired seeds | hands | EV/hand | 95% CI | valid fired | fired delta mean | fired delta 95% CI | p95 loss | non-fired final mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cse1_100` | `100` | `200` | `+0.1211` | `[-0.4873, +0.7296]` | `24` | `+0.3333` | `[-4.3622, +5.0288]` | `18.0770` | `0` |
| `cse1_add150` | `150` | `300` | `+0.1100` | `[-0.3313, +0.5513]` | `35` | `+1.4351` | not separately exported | not separately exported | `0` |
| combined | `250` | `500` | `+0.1145` weighted | component CIs cross zero | `59` | `+0.9869` | `[-1.8869, +3.8607]` | `17.3270` | `0` |

Extended artifacts:

- `outputs/evals/hu_turn1_topk_confirm_cse1_100/summary.json`
- `outputs/evals/hu_turn1_topk_confirm_cse1_100/decisions.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_cse1_add150/summary.json`
- `outputs/evals/hu_turn1_topk_confirm_cse1_add150/decisions.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_cse1_combined250/analysis/hu_turn1_topk_confirm_summary.csv`

Post-hoc confirm-delta floor diagnostic on the combined fired rows:

| virtual floor | fired kept | mean fired delta | 95% CI | p95 loss | max loss | note |
|---|---:|---:|---:|---:|---:|---|
| `d0` | `59` | `+0.9869` | `[-1.8869, +3.8607]` | `17.2270` | `29.2270` | actual audit config |
| `d2` | `28` | `+4.5958` | `[-0.0015, +9.1930]` | `8.0000` | `18.2270` | diagnostic only |
| `d3` | `23` | `+6.1264` | `[+0.8927, +11.3602]` | `6.0000` | `18.2270` | next validation candidate |
| `d6` | `12` | `+7.5568` | `[+0.9760, +14.1375]` | `2.0000` | `6.0000` | likely underfires |

Fresh `d3/cse1` validation:

| run | paired seeds | hands | EV/hand | 95% CI | valid fired | fired delta mean | fired delta 95% CI | p95 loss | non-fired final mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `d3_cse1_250` | `250` | `500` | `-0.1378` | `[-0.4231, +0.1475]` | `29` | `-1.7555` | `[-6.1392, +2.6283]` | `24.2270` | `0` |

Artifacts:

- `outputs/evals/hu_turn1_topk_confirm_d3_cse1_250/summary.json`
- `outputs/evals/hu_turn1_topk_confirm_d3_cse1_250/decisions.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_d3_cse1_250/analysis/hu_turn1_topk_confirm_summary.csv`

Interpretation:

- This is still not a T1 adoption signal. Fired counts are far below the `>=50`
  target in the small sweep, and the extended `cse1/d0` audit reached the fired
  target but still has a wide CI and large tail losses.
- The post-fix non-fired cancellation check is clean for all three configs, so
  fired-only realized delta analysis is usable.
- `cse1/d0` is No-Go for adoption: per-fire mean is positive but not
  statistically resolved, and p95 loss remains too high.
- The post-hoc `d3` floor did not transfer to fresh seeds. Fresh `d3/cse1`
  had negative fired delta and worse p95 loss, so it is No-Go.
- Several fresh `d3` losses still had large confirm deltas but high confirm SE.
  The next useful experiment should increase Stage B confirm MC
  (`confirm16` or `confirm32`) and keep the evaluation by realized fired delta.
- Do not compare configs by confirm delta; compare by realized fired
  seat-swap delta and tail loss.

GCP `d0/confirm16/cse1` validation:

| run | paired seeds | hands | EV/hand | valid fired | fired delta mean | fired delta 95% CI | p95 loss | non-fired final mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d0_confirm16_gcp2000` | `2000` | `4000` | `-0.0385` | `220` | `-0.9969` | `[-2.5018, +0.5081]` | `23.2270` | `0` |

Artifacts:

- `outputs/evals/hu_turn1_topk_confirm_d0_confirm16_gcp2000/aggregate/summary.json`
- `outputs/evals/hu_turn1_topk_confirm_d0_confirm16_gcp2000/aggregate/hu_turn1_decisions.jsonl`
- `outputs/evals/hu_turn1_topk_confirm_d0_confirm16_gcp2000/aggregate/hu_turn1_topk_confirm_audit/hu_turn1_topk_confirm_summary.md`

Interpretation:

- Non-fired final mismatch stayed clean at `0`, so the fired-only realized
  delta audit is valid.
- `confirm16` did not fix the issue: it reached a large fired sample (`220`)
  but the realized per-fire mean is negative and the p95 loss remains high.
- This rules out the current T1 TopK candidate/gate shape for adoption. The
  next useful T1 work is not another `confirm4/confirm16` threshold sweep; it
  should change the candidate generator or teacher objective before another
  large validation.

## Stage1b Candidate-Generator Classifier Diagnostic

After `d0/confirm16` failed, a validation-only T1 candidate-generator trainer
was added:

```powershell
python -m ofc_regular.train_hu_turn1_candidate_generator
```

Purpose:

- Train a classifier to score actions by whether the Stage9f P2 teacher treats
  them as near-best.
- Save it through the existing `HuSklearnActionValueModel` interface, so it can
  be used by the T1 TopK confirm runtime as a drop-in candidate model.
- Compare candidate-generator TopK recall and TopK regret against the current
  regression candidate on the same split.

Inputs:

- teacher:
  `outputs/hu_turn1_stage1_pilot/augmented_stage9f_p2_stratified670_plus_w20_margin1_fired84/hu_turn1_stage1_full2k_plus_stratified670_plus_w20_margin1_fired84.jsonl`
- baseline candidate:
  `models/hu_turn1_stage1_stage9f_p2_full2k_plus_stratified670_tail84_w20_100_hgb_leaf3_lr01_l2_1.pkl`

Results:

| model | label | holdout top1 regret | holdout top3 best regret | holdout top5 best regret | holdout top3 accepted recall | baseline top3 best regret |
|---|---|---:|---:|---:|---:|---:|
| `hu_turn1_stage1b_candidate_hgb_accept025_gray2_weighted_v2.pkl` | regret `<=0.25` | `9.8492` | `5.8452` | `3.8132` | `0.5100` | `2.8831` |
| `hu_turn1_stage1b_candidate_hgb_accept2_gray5_weighted.pkl` | regret `<=2.0` | `9.9656` | `6.2494` | `4.3389` | `0.5626` | `2.8831` |

Artifacts:

- `models/hu_turn1_stage1b_candidate_smoke200_logistic.pkl`
- `models/hu_turn1_stage1b_candidate_hgb_accept025_gray2_weighted_v2.pkl`
- `models/hu_turn1_stage1b_candidate_hgb_accept2_gray5_weighted.pkl`
- `outputs/training/hu_turn1_stage1b_candidate_hgb_accept025_gray2_weighted_v2/metrics.json`
- `outputs/training/hu_turn1_stage1b_candidate_hgb_accept2_gray5_weighted/metrics.json`

Interpretation:

- The classifier path is functional and runtime-loadable, but it is not a
  better candidate generator than the current regression HGB.
- Weighting stratified/tail relabel sources heavily improves train-source
  behavior but hurts holdout generalization on those same sources.
- `accept_regret=2.0` increases the positive class but still loses to the
  existing regression model on holdout TopK regret and recall.
- Do not run a seat-swap with these Stage1b classifier models.
- Next useful T1 work should improve the teacher data or action-value
  regression candidate itself, not replace it with the current binary
  classifier objective.
## Stage1c Natural Teacher Expansion Plan

Decision:

- T1 runtime adoption remains No-Go.
- Do not continue simple TopK confirm threshold sweeps on the current candidate.
- Do not use the Stage1b classifier models for seat-swap.
- Next work is to expand natural-distribution HU T1 teacher data with the fixed
  `stage9f_p2` continuation, then train a stronger regression/ranking candidate
  generator.

Fixed continuation:

- hero profile: `stage9f_p2`
- opponent profile: `stage9f_p2`
- T2 status: fixed continuation for T1 teacher work, not production
- T3 status: Stage7 `m5_r10` remains fixed inside the continuation

Local probe:

- `stage9f_p2`, `samples=2`, `future_samples=1`, full actions timed out after
  `120s` before writing output.
- `stage9f_fast_t2_t1_teacher`, `samples=1`, `future_samples=1`,
  `max_actions=2` also timed out after `120s` before writing output.
- Both residual `hu_turn1_teacher_pilot` processes were killed.
- Interpretation: this workstation is currently not suitable for T1 natural
  teacher smoke while other exact-oracle jobs are active. Use a small Spot/GCP
  shard smoke instead of local CPU.

GCP smoke command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage1c-natural-smoke-20260624 `
  -TotalSamples 4 `
  -ShardSamples 1 `
  -VmCount 4 `
  -FutureSamples 1 `
  -MaxActions 0 `
  -BaseSeed 2026062504 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -OpeningLookaheadSamples 1 `
  -CollectTopkLog `
  -CreateInstances
```

If the smoke passes, the first broad candidate is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage1c-natural-2000-20260624 `
  -TotalSamples 2000 `
  -ShardSamples 1 `
  -VmCount 40 `
  -FutureSamples 1 `
  -MaxActions 0 `
  -BaseSeed 2026062601 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -OpeningLookaheadSamples 1 `
  -CollectTopkLog `
  -CreateInstances
```

Smoke Go conditions:

- all shards complete
- records equal requested samples
- summary schema is `hu_turn1_stage1_pilot_summary_v1`
- no residual GCP VMs
- hidden-discard metadata remains present

GCP smoke result:

- run: `regular-hu-t1-stage1c-natural-smoke-20260624-0938`
- records: `4 / 4`
- completed shards: `4 / 4`
- mean seconds/sample: `11.43`
- max seconds/sample: `21.78`
- mean action count: `27.0`
- topk decisions during continuation: `216`
- topk overrides during continuation: `0`
- duplicate T2 state rate: `0.0`
- duplicate T2 state plus decision seed rate: `0.0`
- merged output:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/regular-hu-t1-stage1c-natural-smoke-20260624-0938/hu_turn1_stage1_pilot.jsonl`
- summary:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/regular-hu-t1-stage1c-natural-smoke-20260624-0938/summary.json`
- residual GCP VMs: `0`

Decision: Stage1c GCP smoke passed. Proceed to a larger natural teacher run
before retraining T1.

Broad Go conditions after smoke:

- mean seconds/sample is feasible on Spot VMs
- action-count distribution is not degenerate
- teacher labels contain enough nonzero regret and near-best alternatives
- Stage9f P2 continuation stays fixed

Production status remains No-Go.

Stage1c first-only 2k GCP run:

- run: `regular-hu-t1-stage1c-natural-2000-20260624-1010`
- records: `2000 / 2000`
- completed shards: `2000 / 2000`
- mean seconds/sample: `8.84`
- max seconds/sample: `28.94`
- mean action count: `26.37`
- topk decisions during continuation: `105474`
- topk overrides during continuation: `1143`
- analysis:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/regular-hu-t1-stage1c-natural-2000-20260624-1010/analysis/summary.json`
- issue: all records were first seat (`first=2000`, `second=0`) because
  `ShardSamples=1` stops after the first player in each shard.

Decision: schema/correctness passed, but this run is not training-ready for a
two-seat T1 model.

Stage1c balanced 2k GCP run:

- run: `regular-hu-t1-stage1c-natural-balanced2000-20260624-1130`
- records: `2000 / 2000`
- completed shards: `1000 / 1000`
- mean seconds/sample: `8.73`
- max seconds/sample: `25.82`
- mean action count: `26.397`
- topk decisions during continuation: `105588`
- topk overrides during continuation: `1079`
- seat split: `first=1000`, `second=1000`
- invalid state/action rows: `0`
- action count / best score / score gap mismatches: `0`
- duplicate state keys: `0`
- score gap mean / median / p95: `3.00 / 0.00 / 18.23`
- merged teacher:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/regular-hu-t1-stage1c-natural-balanced2000-20260624-1130/hu_turn1_stage1_pilot.jsonl`
- analysis:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/regular-hu-t1-stage1c-natural-balanced2000-20260624-1130/analysis/summary.json`

Decision: balanced teacher generation is clean. It is valid as a diagnostic and
training input, but not as runtime evidence.

Stage1c balanced 2k training smoke:

| model | eval set | top1 | top3 | avg regret | decision |
| --- | --- | ---: | ---: | ---: | --- |
| old `w20_100` | balanced2k holdout | `59.75%` | `71.00%` | `4.99` | keep as baseline |
| `stage1c_balanced2k_hgb_leaf3_lr01_l2_1` | balanced2k holdout | `40.25%` | `55.50%` | `7.98` | No-Go |
| `aug_natw0p5` | balanced2k holdout | `36.00%` | `54.75%` | `8.79` | No-Go |
| `aug_natw1` | balanced2k holdout | `37.00%` | `54.50%` | `8.49` | No-Go |
| `aug_natw3` | balanced2k holdout | `42.00%` | `56.75%` | `7.91` | No-Go |
| old `w20_100` | augmented holdout | `58.15%` | `70.56%` | `5.17` | keep as baseline |
| best augmented `natw3` | augmented holdout | `41.32%` | `58.68%` | `8.17` | No-Go |

Artifacts:

- single-source model:
  `models/hu_turn1_stage1c_balanced2k_hgb_leaf3_lr01_l2_1.pkl`
- augmented teacher:
  `outputs/hu_turn1_stage1_pilot/augmented_stage1c_balanced2k/hu_turn1_stage1_full2k_plus_stratified670_tail84_plus_stage1c_balanced2k.jsonl`
- same-split comparison:
  `outputs/training/hu_turn1_stage1c_balanced2k/same_split_model_comparison.csv`
- augmented comparison:
  `outputs/training/hu_turn1_stage1c_augmented_balanced2k/stage1c_model_comparison.csv`

Decision:

- Do not adopt the Stage1c balanced-only model.
- Do not adopt the Stage1c augmented models.
- Do not run T1 seat-swap from this line.
- Do not expand MC1 natural labels further; they add coverage but degrade the
  current action-value model.
- Next useful T1 work needs a better target: higher-confidence relabels,
  direct safe-override labels, or a stronger ranking/loss design. More MC1
  natural teacher rows are not the bottleneck.

Stage1d high-regret relabel probe from balanced2k:

- target source:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/regular-hu-t1-stage1c-natural-balanced2000-20260624-1130/hu_turn1_stage1_pilot.jsonl`
- selection model:
  `models/hu_turn1_stage1_stage9f_p2_full2k_plus_stratified670_tail84_w20_100_hgb_leaf3_lr01_l2_1.pkl`
- selected targets:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/stage1d_refinement_targets/balanced2k_stage1d_targets300.jsonl`
- selection mix:
  `high_model_regret=120`, `high_score_gap=80`, `low_score_gap=60`,
  `random_cover=40`, `max_targets=300`
- target seat split: `first=155`, `second=145`
- selected max model regret mean / p90: `14.88 / 33.23`
- selected score gap mean / p90: `7.87 / 24.23`

Relabel:

- smoke2 passed:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/stage1d_refinement_targets/balanced2k_stage1d_targets300_relabel_smoke2.jsonl`
- full relabel output:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/stage1d_refinement_targets/stage9f_p2_relabel_targets300_parallel/balanced2k_stage1d_targets300_stage9f_p2_relabel_merged.jsonl`
- records: `300 / 300`
- best action changed: `265 / 300` (`88.33%`)
- source best new regret mean / median / p90 / max:
  `11.66 / 8.00 / 27.23 / 49.45`
- relabel seconds mean: `15.29`

Stage1d action-value retraining:

| model | eval set | top1 | top3 | avg regret | decision |
| --- | --- | ---: | ---: | ---: | --- |
| old `w20_100` | Stage1d augmented holdout | `56.63%` | `71.69%` | `5.61` | keep as baseline |
| `stage1d_w20` | Stage1d augmented holdout | `37.81%` | `56.30%` | `8.59` | No-Go |
| `stage1d_w50` | Stage1d augmented holdout | `37.15%` | `55.48%` | `8.60` | No-Go |

Source breakdown on Stage1d holdout:

- old `w20_100` on Stage1d relabel source: avg regret `6.89`
- `stage1d_w20` on Stage1d relabel source: avg regret `8.96`
- `stage1d_w50` on Stage1d relabel source: avg regret `8.02`

Artifacts:

- merged Stage1d teacher:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/stage1d_refinement_targets/full2k_stratified670_tail84_plus_stage1d_relabel300.jsonl`
- Stage1d model comparison:
  `outputs/training/hu_turn1_stage1d_full2k_stratified670_tail84_plus_relabel300/stage1d_model_comparison.csv`
- Stage1d source comparison:
  `outputs/training/hu_turn1_stage1d_full2k_stratified670_tail84_plus_relabel300/stage1d_source_model_comparison.csv`

Decision:

- Stage1d relabels are diagnostically valuable: they expose many places where
  the old source labels or old model choice differ from the stronger
  continuation.
- The current HGB action-value retraining path does not absorb those labels;
  both Stage1d weighted candidates are worse than the existing `w20_100`
  baseline on the same holdout.
- Do not adopt Stage1d HGB models.
- The next useful T1 work should not be another HGB source-weight sweep. It
  should use the Stage1d relabels as hard examples for either:
  1. a candidate-generator/ranker loss that preserves old `w20_100` broad
     behavior while improving recall on Stage1d hard states, or
  2. a selective safe-override head trained on candidate-vs-baseline deltas.

Stage1d candidate-generator probe:

- training source:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/stage1d_refinement_targets/full2k_stratified670_tail84_plus_stage1d_relabel300.jsonl`
- baseline:
  `models/hu_turn1_stage1_stage9f_p2_full2k_plus_stratified670_tail84_w20_100_hgb_leaf3_lr01_l2_1.pkl`
- HGB candidates:
  - `models/hu_turn1_stage1d_candidate_hgb_accept025_gray2_weighted.pkl`
  - `models/hu_turn1_stage1d_candidate_hgb_accept2_gray5_weighted.pkl`
- metrics:
  `outputs/training/hu_turn1_stage1d_candidate_generator/`

Same-split candidate-generator result:

| model | holdout top1 regret | top3 best regret | top5 best regret | decision |
| --- | ---: | ---: | ---: | --- |
| old `w20_100` baseline | `4.88` | `2.57` | `1.70` | keep |
| HGB accept0.25/gray2 | `8.69` | `5.45` | `4.14` | No-Go as standalone |
| HGB accept2/gray5 | `8.63` | `5.50` | `3.82` | No-Go as standalone |

ExtraTrees accept0.25/gray2 was attempted but exceeded `300s` without writing
metrics. The partial model was deleted. It is not a viable local training path.

Union TopK diagnostic:

| candidate set | avg union size | teacher best recall | best avg regret |
| --- | ---: | ---: | ---: |
| old `w20_100` Top3 | `3.00` | `74.80%` | `2.57` |
| old `w20_100` Top5 | `5.00` | `80.52%` | `1.70` |
| old `w20_100` Top8 | `8.00` | `88.05%` | `0.90` |
| old Top5 + HGB accept0.25 Top5 | `7.51` | `84.45%` | `1.22` |
| old Top5 + HGB accept2 Top5 | `7.57` | `85.43%` | `1.16` |

Decision:

- The Stage1d candidate-generator HGBs are worse than the old action-value
  baseline as standalone candidate models.
- Combining them with old `w20_100` TopK helps versus old Top5, but does not
  beat simply increasing the old model to Top8.
- Do not add these Stage1d candidate-generator models to runtime.
- The useful runtime probe is old `w20_100` with larger TopK, not a second
  candidate generator.

T1 TopK confirm k8 runtime smoke:

- command profile: `stage9f_p2_hu_t1_topk_confirm`
- config: `k8/mc2/d0/confirm4/cse1.5/pd0/seat=first+second`
- output:
  `outputs/evals/hu_turn1_topk_confirm_k8_smoke2/summary.json`
- decision log:
  `outputs/evals/hu_turn1_topk_confirm_k8_smoke2/hu_turn1_decisions.jsonl`
- audit:
  `outputs/evals/hu_turn1_topk_confirm_k8_smoke2/analysis/`
- games / hands: `2 / 4`
- elapsed: `34.77s`
- overrides: `0`
- non-fired final matches baseline: `4`
- non-fired delta max abs: `0.0`

Decision:

- k8 TopK confirm wiring works and preserves non-fired pair cancellation.
- Local evaluation is too slow for meaningful k8 validation. The next runtime
  check should be a small GCP/Spot run, not a local long run.
- Candidate config for the next GCP smoke:
  `k8/mc8/d0/confirm16/cse1.5/pd0/seat=first+second`
- This is still validation-only. T1 runtime adoption remains No-Go until
  realized fired-delta is positive on a nontrivial fired sample.

T1 TopK confirm k8 GCP smoke:

- run name:
  `regular-hu-t1-k8-confirm16-smoke100-20260624-1420`
- profile: `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- config: `k8/mc8/d0/confirm16/cse1.5/pd0/seat=first+second`
- output:
  `outputs/evals/hu_turn1_topk_confirm_k8_gcp_smoke100/summary.json`
- decision log:
  `outputs/evals/hu_turn1_topk_confirm_k8_gcp_smoke100/hu_turn1_decisions.jsonl`
- audit:
  `outputs/evals/hu_turn1_topk_confirm_k8_gcp_smoke100/hu_turn1_topk_confirm_audit/`
- games / hands: `100 / 200`
- T1 decisions: `200`
- valid T1 decisions: `195`
- valid overrides: `5`
- override rate on valid decisions: `2.56%`
- realized per-fire delta mean: `-3.0454`
- realized per-fire CI95: `[-8.0918, +2.0010]`
- estimated EV per T1 decision: `-0.0781`
- fired loss / win / zero: `2 / 0 / 3`
- p95 loss: `10.9816`
- confirm delta mean on fired rows: `+5.4341`
- confirm metric role: gate diagnostic only; not allowed as performance claim
- non-fired counterfactual nonzero: `0`
- non-fired final mismatch: `0`
- residual GCP VMs: cleaned up after run

Decision:

- The GCP smoke validates the evaluation harness: non-fired T1 decisions cancel
  exactly and realized fired-delta is the correct metric.
- The runtime candidate itself is No-Go. Confirm MC still selected five fires
  whose independent game-level paired result was negative on average.
- Do not expand this `k8/mc8/confirm16/cse1.5` line to a larger run.
- Do not choose thresholds by confirm delta. The fired rows had positive
  confirm mean but negative realized per-fire delta, which is the remaining
  selection/cutoff bias.
- Next T1 work should add a deployable safe-override veto before scaling this
  line again.

T1 TopK confirm k8 safe-selector local smoke:

- profile: `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- config: `k8/mc2/d0/confirm4/cse1.5/safe0.5/pd0/seat=first+second`
- safe selector:
  `models/hu_turn1_safe_override_selector_teacher_regret_combined296_accept0p25_gray1_delta_plus_meta.pkl`
- output:
  `outputs/evals/hu_turn1_topk_confirm_k8_safe_smoke2/summary.json`
- decision log:
  `outputs/evals/hu_turn1_topk_confirm_k8_safe_smoke2/hu_turn1_decisions.jsonl`
- games / hands: `2 / 4`
- T1 decisions: `4`
- overrides: `0`
- `below_safe_selector`: `1`
- scored safe-selector value on vetoed row: `8.27e-7`
- non-fired counterfactual nonzero: `0`

Decision:

- Safe selector veto wiring works and preserves non-fired cancellation.
- This is not runtime evidence; it only proves that the veto can stop a
  confirm-passing candidate.
- Next validation candidate:
  `k8/mc8/d0/confirm16/cse1.5/safe0.5/pd0/seat=first+second`
- Run that as a small GCP/Spot smoke before any larger T1 runtime evaluation.

T1 TopK confirm k8 safe-selector GCP smoke:

- run name:
  `regular-hu-t1-k8-confirm16-safe05-smoke100-20260624-1434`
- profile: `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- config: `k8/mc8/d0/confirm16/cse1.5/safe0.5/pd0/seat=first+second`
- safe selector:
  `models/hu_turn1_safe_override_selector_teacher_regret_combined296_accept0p25_gray1_delta_plus_meta.pkl`
- output:
  `outputs/evals/hu_turn1_topk_confirm_k8_safe_gcp_smoke100/`
- audit:
  `outputs/evals/hu_turn1_topk_confirm_k8_safe_gcp_smoke100/hu_turn1_topk_confirm_audit/`
- games / hands: `100 / 200`
- T1 decisions / valid: `200 / 198`
- valid overrides: `2`
- override rate on valid decisions: `1.01%`
- realized per-fire delta mean: `0.0000`
- estimated EV/decision: `0.0000`
- fired loss / win / zero: `0 / 0 / 2`
- `below_safe_selector`: `3`
- safe selector scored rows: `5`
- non-fired final mismatch: `0`
- non-fired counterfactual nonzero: `2`
- residual GCP instances: `0`

Decision:

- The safe selector veto materially improved safety relative to the previous
  k8/confirm16 run: it allowed only `2 / 5` confirm-passing rows, and no fired
  row lost in realized seat-swap counterfactuals.
- This is still underpowered. Both fired rows were zero-delta, so the smoke is
  not positive runtime evidence.
- Do not adopt T1 runtime and do not scale Stage1c MC1 natural labels.
- If continuing this line, use a fired-count target rather than game count; at
  the observed `~1%` fire rate, `50` fires needs roughly `5k` paired decisions.

T1 TopK confirm k8 safe-selector fired-count GCP validation:

- run name:
  `regular-hu-t1-k8-safe05-firetarget2500-20260624-1507`
- profile: `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- config: `k8/mc8/d0/confirm16/cse1.5/safe0.5/pd0/seat=first+second`
- safe selector:
  `models/hu_turn1_safe_override_selector_teacher_regret_combined296_accept0p25_gray1_delta_plus_meta.pkl`
- output:
  `outputs/evals/hu_turn1_topk_confirm_k8_safe_firetarget2500/`
- audit:
  `outputs/evals/hu_turn1_topk_confirm_k8_safe_firetarget2500/hu_turn1_topk_confirm_audit/`
- paired seeds / hands: `2500 / 5000`
- T1 decisions / valid: `5000 / 4977`
- valid overrides: `23`
- override rate on valid decisions: `0.462%`
- realized per-fire delta mean: `-1.1166`
- realized per-fire CI95: `[-5.9836, +3.7505]`
- estimated EV per T1 decision: `-0.0052`
- estimated EV per T1 decision CI95: `[-0.0277, +0.0173]`
- fired loss / win / zero: `5 / 9 / 9`
- p95 loss / max loss: `22.6270 / 34.2270`
- confirm delta mean on fired rows: `+4.2893`
- confirm metric role: gate diagnostic only; not allowed as performance claim
- `below_safe_selector`: `162`
- non-fired final mismatch: `0`
- residual GCP instances: `0`

Decision:

- No-Go for T1 runtime adoption and for scaling this TopK+confirm+safe-selector
  line. The larger fired-count validation did not reproduce a positive
  realized per-fire delta.
- The safe selector reduced fire rate and prevented many confirm-passing rows,
  but still allowed large losses, including `-34.2270`, `-23.2270`, and
  `-17.2270`.
- The gap between `confirm_delta_mean_on_valid_fired = +4.2893` and realized
  per-fire mean `-1.1166` confirms that confirm MC remains a gate diagnostic
  only; it must not be used as a performance claim or threshold-selection
  objective.
- Non-fired final-action cancellation is preserved (`0` final mismatches), so
  the negative result is attributable to fired rows, not evaluation drift.
- Next T1 work should stop simple threshold/safe-score sweeping. Use these
  fired losses as hard negatives and redesign the T1 objective or runtime gate
  before any further large GCP validation.

T1 TopK confirm replay metadata fix:

- The fired-count run above is useful as runtime diagnostics only. Its
  `hu_turn1_decisions.jsonl` rows do not contain explicit `replay_ready`, so
  the exact relabel extractor correctly treats all `5000` rows as
  replay-ineligible when replay readiness is required.
- `hu_turn1_topk_confirm_decision_v1` logging now mirrors the regular T1
  decision log and writes:
  `visibility_model`, `discard_visibility`, `true_dead_cards`,
  `visible_dead_cards`, `hero_private_discards`, `opponent_private_discards`,
  and `replay_ready`.
- Smoke output:
  `outputs/evals/hu_turn1_topk_confirm_replay_meta_smoke/hu_turn1_decisions.jsonl`
- Smoke result: `2 / 2` TopK confirm decision rows have
  `replay_ready = true`, `visibility_model = hidden_discard`, and
  `discard_visibility = own_private_only`.
- The default replay extractor no longer skips the new rows as
  `not_replay_ready`; the smoke wrote `0` relabel targets only because it did
  not include realized seat-swap deltas.

Next:

- Re-run a small fired-target T1 TopK confirm collection with the new metadata.
- Extract replay-ready fired rows with
  `extract_hu_turn1_decision_relabel_targets` using the default
  replay-ready requirement.
- Relabel those rows with the Stage9f P2 teacher before any new T1 training.

T1 TopK confirm replay-ready extraction smoke:

- output:
  `outputs/evals/hu_turn1_topk_confirm_replay_ready_extract_smoke20/`
- profile: `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- config: `k1/mc1/d0/seat=first`
- paired seeds / hands: `20 / 40`
- T1 decision rows: `40`
- replay-ready decision rows: `40 / 40`
- realized T1 delta rows: `28`
- realized T1 override rows: `12`
- extracted relabel targets with replay-ready required: `5`
- label counts: `hard_negative = 1`, `positive = 4`
- extractor skipped `not_replay_ready = 0`
- Stage9f P2 relabel smoke:
  `outputs/evals/hu_turn1_topk_confirm_replay_ready_extract_smoke20/relabel_targets_stage9f_p2_smoke2.jsonl`
- relabel smoke records: `2`
- relabel smoke best action changed: `2 / 2`

Decision:

- The TopK confirm fired-row pipeline is now end-to-end usable:
  runtime decision log -> replay-ready target extraction -> Stage9f P2 relabel.
- This smoke used an intentionally cheap and aggressive config, so it is not
  runtime-strength evidence.
- Next real data collection should use the failed production-like line
  `k8/mc8/d0/confirm16/cse1.5/safe0.5/pd0/seat=first+second` or a deliberately
  diagnostic variant, but the output must be generated after the replay
  metadata fix.

T1 TopK confirm replay-ready firetarget GCP collection:

- run name:
  `regular-hu-t1-k8-safe05-replayready-20260624-1740`
- profile: `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- config: `k8/mc8/d0/confirm16/cse1.5/safe0.5/pd0/seat=first+second`
- output:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/`
- paired seeds / hands: `2500 / 5000`
- GCP shards: `250 / 250`
- T1 decision rows: `5000`
- replay-ready rows: `5000 / 5000`
- valid fired rows: `22`
- realized fired mean: `+3.8946`
- realized fired CI95: `[-0.6367, +8.4259]`
- estimated EV / T1 decision: `+0.0172`
- non-fired final mismatch: `0`
- no-override reason counts:
  `below_confirm_delta = 1372`, `below_confirm_se = 759`,
  `below_safe_selector = 163`, `mc_best_is_baseline = 1786`,
  `topk_empty = 898`, `override_fired = 22`
- TopK aggregate audit remains runtime-negative:
  realized TopK override mean `-1.2307`, CI95 `[-3.1788, +0.7174]`,
  p95 / max loss `13.0000 / 36.2270`

Replay-ready relabel target extraction:

- fired target output:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/relabel_targets.jsonl`
- fired targets: `13`
- fired labels: `positive = 8`, `hard_negative = 5`
- non-fired boundary target output:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/relabel_targets_boundary_margin0p5_top300.jsonl`
- non-fired boundary targets: `300`
- boundary split: `first = 152`, `second = 148`
- combined target output:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/relabel_targets_combined313.jsonl`

Stage9f P2 relabel of combined313:

- output dir:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/stage9f_p2_relabel_combined313_full/`
- merged teacher:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/stage9f_p2_relabel_combined313_full/relabel_targets_stage9f_p2_merged.jsonl`
- aggregate summary:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/stage9f_p2_relabel_combined313_full/relabel_targets_stage9f_p2_aggregate_summary.json`
- records: `313`
- source kinds: `runtime_fired_decision = 13`,
  `runtime_non_fired_candidate_decision = 300`
- labels: `positive = 9`, `hard_negative = 9`, `neutral = 295`
- seats: `first = 156`, `second = 157`
- Stage9f P2 best action changed: `272 / 313` (`86.9%`)
- fired target best action changed: `13 / 13`
- non-fired boundary best action changed: `259 / 300`
- source action regret mean / max under relabel:
  `11.8444 / 44.2270`

Decision:

- The replay-ready collection succeeded and is suitable as relabeling evidence,
  not as T1 runtime adoption evidence.
- The runtime TopK+confirm+safe selector is still No-Go. The broader TopK audit
  remains negative and confirm MC remains gate-diagnostic only.
- The Stage9f P2 relabel changed the best action on most selected targets,
  which strongly supports using these rows as additional T1 training /
  hard-negative data.
- Next step is to merge this relabel set into the existing T1 teacher corpus,
  train a Stage1e/Stage1d-plus-runtime-relabel candidate, and evaluate by
  holdout regret and seat-swap. Do not enable T1 runtime override.

Stage1e runtime313 merge and retraining:

- merged teacher:
  `outputs/hu_turn1_stage1c_natural_teacher_expansion/stage1e_runtime_relabel313/full2k_stratified670_tail84_stage1d300_plus_runtime313.jsonl`
- merged rows: `3367` (`3054` previous Stage1d rows + `313` runtime relabel rows)
- candidate generator outputs:
  - `models/hu_turn1_stage1e_candidate_hgb_accept025_gray2_runtime313_weighted.pkl`
  - `models/hu_turn1_stage1e_candidate_hgb_accept2_gray5_runtime313_weighted.pkl`
- candidate generator metrics:
  `outputs/training/hu_turn1_stage1e_candidate_generator_runtime313/`
- `accept0.25/gray2` holdout:
  - candidate top1 / top3 / top5 regret:
    `10.0257 / 5.9137 / 3.9723`
  - baseline top1 / top3 / top5 regret on same split:
    `5.8491 / 3.3448 / 2.1959`
- `accept2/gray5` holdout:
  - candidate top1 / top3 / top5 regret:
    `10.0450 / 5.8136 / 3.9636`
  - baseline top1 / top3 / top5 regret on same split:
    `5.8491 / 3.3448 / 2.1959`
- result: Stage1e HGB candidate-generator retraining is No-Go. It is
  substantially worse than the existing `w20_100` baseline on the same split.

Stage1e runtime313 safe selector retraining:

- input:
  `outputs/evals/hu_turn1_topk_confirm_replayready_firetarget_gcp/stage9f_p2_relabel_combined313_full/relabel_targets_stage9f_p2_merged.jsonl`
- comparison:
  `outputs/training/hu_turn1_safe_selector_stage1e_runtime313/selector_comparison.csv`
- trained models:
  - `models/hu_turn1_safe_selector_stage1e_runtime313_accept025_gray1_delta_plus_meta.pkl`
  - `models/hu_turn1_safe_selector_stage1e_runtime313_accept025_gray1_candidate_delta_plus_meta.pkl`
  - `models/hu_turn1_safe_selector_stage1e_runtime313_accept2_gray5_delta_plus_meta.pkl`
  - `models/hu_turn1_safe_selector_stage1e_runtime313_accept2_gray5_candidate_delta_plus_meta.pkl`
- best validation AP:
  - `accept0.25/gray1 + delta_plus_meta`: `0.4820`
  - `accept0.25/gray1 + candidate_delta_plus_meta`: `0.4144`
  - `accept2/gray5 + delta_plus_meta`: `0.4304`
  - `accept2/gray5 + candidate_delta_plus_meta`: `0.4367`
- best AP config:
  `accept0.25/gray1 + delta_plus_meta`
- best AP config test threshold behavior:
  - threshold `0.3`: fires `5`, precision `0.40`
  - threshold `0.5`: fires `1`, precision `0.00`
  - threshold `0.8`: fires `0`
- the broader `accept2/gray5 + candidate_delta_plus_meta` config fires more
  test rows, but still only reaches precision `0.50` at threshold `0.8`.

Decision:

- The runtime313 rows are valuable as hard-negative / relabel evidence.
- They do not produce a deployable Stage1e HGB candidate generator.
- They also do not produce a deployable standalone safe selector. AP improved
  over the older combined296 selector family, but test-set precision/fire-count
  is too weak for runtime gating.
- T1 runtime remains No-Go.
- Do not run a large T1 seat-swap with these Stage1e artifacts.
- Next T1 direction should be a model/objective redesign or larger teacher
  relabel pass, not another simple HGB candidate-generator or safe-score
  threshold sweep.

Stage1f candidate-generator objective probe:

- change:
  `train_hu_turn1_candidate_generator` now supports score-regression model
  types:
  `hgb_regressor`, `extra_trees_regressor`, and `random_forest_regressor`.
- reason:
  The Stage1e binary near-best classifier failed. A direct teacher-score
  regression objective is a cheap way to check whether the failure is caused by
  the binary label construction rather than by the feature/data mix itself.
- comparison CSV:
  `outputs/training/hu_turn1_stage1f_candidate_generator_runtime313_regression/stage1f_candidate_model_comparison.csv`
- classification weight sweep:
  `outputs/training/hu_turn1_stage1f_candidate_generator_runtime313_weight_sweep/`
- regression sweep:
  `outputs/training/hu_turn1_stage1f_candidate_generator_runtime313_regression/`
- best classification probe:
  `w50_hgb_leaf3_lr01_l2_1`
  - holdout top1 / top3 / top5 regret:
    `7.8262 / 5.5545 / 4.3423`
  - baseline top1 / top3 / top5 regret on same split:
    `5.5930 / 2.9181 / 1.9745`
- best regression probe:
  `regressor_w50_hgb_leaf3_lr01_l2_1`
  - holdout top1 / top3 / top5 regret:
    `8.6479 / 5.6929 / 4.3075`
  - baseline top3 regret on same split:
    `3.2301`

Decision:

- Stage1f objective probe is No-Go.
- Direct score regression did not solve the Stage1e failure.
- Existing baseline remains better than all Stage1e/Stage1f probes.
- The next useful T1 step is not another local model-type/threshold sweep.
  Either collect a larger and cleaner T1 teacher/relabel set, or redesign the
  T1 representation/model so it can learn from state-level action rankings
  without being dominated by ties, noisy MC1 labels, and runtime boundary
  relabel rows.

Stage1g pairwise ranking probe:

- change:
  `train_hu_turn1_candidate_generator` now supports `pairwise_logistic`.
  It trains on within-state action score differences and saves a runtime
  compatible per-action utility scorer.
- reason:
  If the action-level binary/regression objectives were the main problem,
  pairwise within-state ranking should improve candidate ordering without
  needing new teacher data.
- comparison CSV:
  `outputs/training/hu_turn1_stage1g_candidate_generator_pairwise_runtime313/stage1f_stage1g_candidate_model_comparison.csv`
- best pairwise probe:
  `pairwise_w50_gap025_pairs64`
  - model:
    `models/hu_turn1_stage1g_pairwise_runtime313_w50_gap025_pairs64.pkl`
  - holdout top1 / top3 / top5 regret:
    `9.8255 / 5.5454 / 3.8874`
  - baseline top3 regret on same split:
    `3.1578`
- larger pairwise probe:
  `pairwise_w100_gap025`
  - holdout top1 / top3 / top5 regret:
    `11.3539 / 6.5797 / 4.2123`
  - baseline top3 regret on same split:
    `3.6174`

Decision:

- Stage1g pairwise ranking is No-Go.
- Pairwise improves neither top1 nor top3 enough to challenge the existing
  baseline. Its best top3 regret remains around `5.55`, while the baseline is
  near `3.16`.
- This closes the cheap local objective/model probes for T1:
  binary near-best classification, score regression, safe selector, and
  pairwise ranking all fail to produce a better T1 candidate generator from the
  current mixed teacher/relabel data.
- Next useful work is data and representation, not another local sklearn
  objective sweep:
  - build a cleaner T1 teacher set with higher-MC / less MC1 tie noise, or
  - train a richer listwise neural model that consumes all legal actions for a
    state jointly, with strict state-level heldout splits.

Stage1h listwise neural probe:

- change:
  Added `train_hu_turn1_listwise_torch`, a state-level listwise softmax trainer
  that consumes all legal actions for a T1 state jointly and saves a runtime
  compatible `HuTorchActionValueModel`.
- reason:
  Stage1e/f/g ruled out simple action-level classification, score regression,
  safe-selector filtering, and linear pairwise ranking. A listwise neural model
  checks whether a modestly richer joint-action objective can absorb the
  current mixed teacher/relabel data.
- output dir:
  `outputs/training/hu_turn1_stage1h_listwise_torch_runtime313/`
- comparison CSV:
  `outputs/training/hu_turn1_stage1h_listwise_torch_runtime313/stage1h_listwise_comparison.csv`
- models:
  - `models/hu_turn1_stage1h_listwise_runtime313_w50_128x64_e10.pt`
  - `models/hu_turn1_stage1h_listwise_runtime313_unweighted_128x64_e10.pt`
- weighted run:
  - source weight: `stage9f_p2 = 50`
  - holdout top1 / top3 / top5 regret:
    `10.9668 / 6.1768 / 4.2106`
  - baseline top3 regret on same split:
    `3.1005`
- unweighted run:
  - holdout top1 / top3 / top5 regret:
    `10.6942 / 5.6865 / 4.1195`
  - baseline top3 regret on same split:
    `2.7106`

Decision:

- Stage1h listwise neural probe is No-Go.
- Training loss decreases, but holdout topK regret remains far worse than the
  existing T1 baseline.
- This makes the current blocker data quality / target quality rather than
  merely model capacity or action-level objective choice.
- Next T1 work should build a cleaner high-MC teacher/relabel set before more
  model work:
  - avoid MC1-heavy labels where possible,
  - separate natural, hard-negative, and boundary sources in heldout,
  - keep exact replay-ready metadata,
  - use state-level split only,
  - evaluate against the existing baseline before any runtime seat-swap.

Stage1i clean higher-MC T1 teacher pilot:

- purpose:
  Check whether clean full-action T1 teacher labels with the fixed
  `stage9f_p2` continuation are feasible before more T1 model work.
- fixed continuation:
  - T2 profile: `stage9f_p2`
  - opponent profile: `stage9f_p2`
  - T3 continuation: `stage7_m5_r10`
  - full replacement: disabled
- local full-action smokes:
  - MC2:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/local_full_actions_mc2_smoke_summary.json`
    - records: `1`
    - legal actions: `27`
    - seconds/sample: `13.27`
  - MC4:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/local_full_actions_mc4_smoke_summary.json`
    - records: `1`
    - legal actions: `27`
    - seconds/sample: `25.22`
- GCP MC4 pilot:
  - run: `regular-hu-t1-clean-mc4-pilot20-20260624-2128`
  - output:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/gcp_clean_mc4_pilot20/hu_turn1_stage1_pilot.jsonl`
  - records: `20/20`
  - missing shards: `0`
  - mean / max seconds per sample: `39.19 / 72.25`
  - mean action count: `26.25`
  - action SE mean / p90: `5.69 / 8.15`
  - existing baseline top1 / top3 / top5 regret:
    `2.3602 / 1.8017 / 0.4807`
  - existing baseline top3 teacher-best recall: `0.55`
- GCP MC16 pilot:
  - run: `regular-hu-t1-clean-mc16-pilot20-20260624-2138`
  - output:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/gcp_clean_mc16_pilot20/hu_turn1_stage1_pilot.jsonl`
  - records: `20/20`
  - missing shards: `0`
  - mean / max seconds per sample: `131.55 / 257.51`
  - mean action count: `26.40`
  - action SE mean / p90: `3.14 / 4.17`
  - existing baseline top1 / top3 / top5 regret:
    `1.6116 / 0.8143 / 0.7206`
  - existing baseline top3 teacher-best recall: `0.70`
- GCP MC32 first-only pilot:
  - run: `regular-hu-t1-clean-mc32-pilot20-20260624`
  - output:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/gcp_clean_mc32_pilot20/hu_turn1_stage1_pilot.jsonl`
  - note: one missing Spot shard (`15`) was requeued into the same run and
    completed cleanly.
  - distribution caveat: this run used `ShardSamples=1`, so it produced
    `first=20`, `second=0`; it is a speed/quality reference only, not a
    balanced training distribution.
  - records: `20/20`
  - missing shards: `0`
  - mean / max seconds per sample: `282.02 / 492.51`
  - mean action count: `26.40`
  - action SE mean / p90: `2.33 / 2.86`
  - existing baseline top1 / top3 / top5 regret:
    `1.4439 / 0.2087 / 0.0116`
  - existing baseline top3 teacher-best recall: `0.85`
- GCP MC32 balanced pilot:
  - primary run: `regular-hu-t1-clean-mc32-balanced-pilot20-20260624`
  - supplement run: `regular-hu-t1-clean-mc32-balanced-supplement2-20260624`
  - output:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/gcp_clean_mc32_balanced_pilot20_combined/hu_turn1_stage1_pilot.jsonl`
  - analysis:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/gcp_clean_mc32_balanced_pilot20_combined/analysis/summary.json`
  - combined summary:
    `outputs/hu_turn1_stage1i_clean_teacher_pilot/gcp_clean_mc32_balanced_pilot20_combined/combined_summary.json`
  - note: primary shard `5` (`seed=2031063001`) was stopped after becoming an
    extreme slow shard with no result; the balanced sample was completed with a
    separate two-record supplement run.
  - records: `20`
  - seat split: `first=10`, `second=10`
  - player split: `0=10`, `1=10`
  - future samples: `32`
  - mean / max seconds per sample from shard summaries: `250.83 / 376.65`
  - mean action count: `26.40`
  - action SE mean / p90: `2.38 / 2.93`
  - best-action SE mean: `2.79`
  - score gap mean / median: `1.177 / 0.921`
  - score gap `<0.50`: `7 / 20`
  - invalid state rows / invalid action rows / actions truncated:
    `0 / 0 / 0`
  - duplicate state keys: `0`
- comparison artifact:
  `outputs/hu_turn1_stage1i_clean_teacher_pilot/stage1i_clean_teacher_pilot_comparison.json`

Stage1j natural MC32 partial broad run:

- run: `regular-hu-t1-stage1j-natural-mc32-200-20260625`
- output:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc32_partial110/hu_turn1_stage1_pilot.jsonl`
- analysis:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc32_partial110/analysis/summary.json`
- refinement targets:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc32_partial110/refinement_targets/targets.jsonl`
- result:
  - requested broad target: `200`
  - completed records: `110`
  - completed shards: `55 / 100`
  - missing shards after stopping slow workers: `45`
  - future samples: `32`
  - seat split: `first=55`, `second=55`
  - player split: `0=55`, `1=55`
  - mean / max seconds per sample: `226.65 / 362.94`
  - mean action count: `26.51`
  - action SE mean / p90 / p95: `2.26 / 2.78 / 2.92`
  - best-action SE mean / p95: `2.62 / 3.09`
  - score gap mean / median / p90: `1.228 / 0.688 / 3.224`
  - score gap `<0.50`: `47 / 110`
  - invalid state rows / invalid action rows / actions truncated:
    `0 / 0 / 0`
  - duplicate state keys: `0`
  - extracted refinement targets: `100`
  - refinement target seat split: `first=50`, `second=50`
  - refinement target reasons:
    `high_model_regret=40`, `high_action_se=32`, `high_score_gap=30`,
    `low_score_gap=30`, `random_cover=20`, `fill_cover=10`
- operational diagnosis:
  The partial run validates MC32 label quality on a broader natural
  distribution, but the slow-shard tail is too large to scale this exact
  setup blindly. Remaining workers were deleted after partial aggregation.

Stage1j MC64 refinement relabel attempt:

- smoke run:
  `regular-hu-t1-stage1j-relabel-gcp-smoke-mc2-20260625`
  - records: `4 / 4`
  - future samples: `2`
  - receive/merge completed:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_smoke_mc2/summary.json`
- MC64 run:
  `regular-hu-t1-stage1j-refinement-relabel-mc64-100-20260625`
  - intended target count: `100`
  - completed records: `14`
  - completed shards: `7 / 50`
  - output:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc64_partial14/hu_turn1_refinement_relabel.jsonl`
  - summary:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc64_partial14/summary.json`
  - future samples: `64`
  - best action changed: `8 / 14`
  - source-best regret mean / max under MC64 relabel:
    `1.005 / 3.719`
  - seat split: `first=4`, `second=10`
  - mean relabel seconds per record: `319.57`
  - action taken:
    Remaining workers were deleted after the completion count stopped growing.
- operational diagnosis:
  MC64 confirms that some MC32 refinement targets change materially, but
  full-action Stage9f P2 MC64 relabeling is too slow for broad use in this
  implementation. Treat the 14 records as diagnostic high-MC support, not as a
  balanced training dataset.

Interpretation:

- The clean full-action teacher path works on GCP with no action truncation.
- MC32 scales at about `2.14x` MC16 wall time and about `7.20x` MC4 wall time
  on this 20-state pilot.
- MC32 materially improves label quality versus MC16:
  - action SE mean: `3.14 -> 2.33`
  - action SE p90: `4.17 -> 2.86`
  - existing baseline top3 regret: `0.8143 -> 0.2087`
  - existing baseline top3 teacher-best recall: `0.70 -> 0.85`
- The balanced MC32 smoke is clean enough to use MC32 as the broad
  clean-teacher baseline for the next pilot dataset, but
  close/high-regret/disagreement/high-SE states should still receive selected
  MC64/128 refinement before training.
- The Stage1j partial broad run shows that a naive MC32 broad pass has a
  problematic slow tail. The next broad pass should either use shorter
  one-sample shards with explicit timeout/replacement, or switch to an MC16
  broad pass followed by MC32/64 selected refinement.
- The Stage1j MC64 refinement attempt shows that full-action MC64 with
  `stage9f_p2` is not a viable 100-target relabel path as implemented.
  Higher-MC labels should be reserved for very small audits unless the T2
  continuation is further accelerated.
- Current T1 runtime adoption and production-scale T1 generation remain No-Go.

Next:

1. Build a stratified clean T1 teacher dataset with either MC16 broad labels
   plus MC32/64 refinement, or MC32 one-sample shards with strict
   timeout/replacement.
2. Include natural, hard-negative, boundary, high-SE, and replay-ready runtime sources
   with source labels preserved in train/heldout.
3. Use MC32 as the practical selected-refinement level. Use MC64 only for small
   audits or for the highest-impact/highest-uncertainty rows.
4. Only after the clean teacher set beats the existing T1 baseline on strict
   state-level heldout should another T1 runtime seat-swap be considered.

Stage1j MC16 broad500 plus MC32 selected refinement:

- MC16 broad run:
  `regular-hu-t1-stage1j-natural-mc16-broad500-20260625`
  - output:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc16_broad500/hu_turn1_stage1_pilot.jsonl`
  - summary:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc16_broad500/summary.json`
  - records: `500 / 500`
  - completed shards: `250 / 250`
  - missing shards: `0`
  - future samples: `16`
  - mean seconds per sample: `147.01`
  - max seconds per sample: `365.19`
  - seat split: `first=250`, `second=250`
  - invalid state rows / invalid action rows / actions truncated:
    `0 / 0 / 0`
  - duplicate state keys: `0`
  - action SE mean / p90 / p95: `3.272 / 4.212 / 4.507`
  - best-action SE mean / p95: `3.822 / 4.960`
  - score gap mean / median / p90: `1.556 / 0.952 / 3.830`
  - score gap `<0.50`: `163 / 500`
- extracted MC32 refinement targets:
  - output:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc16_broad500/refinement_targets/targets.jsonl`
  - summary:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc16_broad500/refinement_targets/summary.json`
  - targets: `250`
  - seat split: `first=130`, `second=120`
  - reason counts:
    `high_model_regret=100`, `high_action_se=92`, `high_score_gap=60`,
    `low_score_gap=47`, `random_cover=55`
- MC32 selected relabel run:
  `r-t1j-mc32-refine250-20260625`
  - output:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/hu_turn1_refinement_relabel.jsonl`
  - summary:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/summary.json`
  - analysis:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/analysis/summary.json`
  - intended targets: `250`
  - completed records: `124`
  - completed shards: `62 / 125`
  - future samples: `32`
  - best action changed: `83 / 124` (`66.94%`)
  - source-best regret mean / median / p95 / max under MC32 relabel:
    `1.841 / 1.241 / 6.063 / 7.689`
  - mean relabel seconds per record: `221.59`
  - seat split: `first=62`, `second=62`
  - action taken:
    Running workers were deleted after completion count stopped growing for
    roughly `8` minutes. Remaining targets are preserved for later targeted
    rerun or timeout-aware splitting.
- MC32 missing-target analysis:
  - output:
    `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/analysis/missing_target_summary.json`
  - missing targets: `126`
  - source bucket counts:
    `high_action_se=45`, `high_model_regret=30`, `high_score_gap=20`,
    `low_score_gap=16`, `random_cover=15`
  - seat split: `first=68`, `second=58`
  - selection max-action SE mean / p95:
    `4.423 / 5.279`
- merged teacher:
  - output:
    `outputs/hu_turn1_stage1j_stratified_teacher/stage1j_mc16_broad500_plus_mc32_partial124/hu_turn1_stage1j_merged_teacher.jsonl`
  - summary:
    `outputs/hu_turn1_stage1j_stratified_teacher/stage1j_mc16_broad500_plus_mc32_partial124/merge_summary.json`
  - records: `500`
  - label sources:
    `base_fast_t2=376`, `stage9f_p2_refinement=124`

Stage1j local training smoke on the merged teacher:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | baseline top1/top3/top5 on same split | decision |
|---|---:|---:|---:|---:|---|
| HGB `leaf31/lr0.05/l2=1` | `2.046` | `0.716` | `0.406` | `1.295 / 0.821 / 0.524` | TopK-only diagnostic; top1 worse |
| listwise torch `256,128/e30` | `2.319` | `0.771` | `0.446` | `1.255 / 0.664 / 0.326` | No-Go |
| HGB `leaf3/lr0.01/l2=1` | `1.249` | `0.560` | `0.348` | `0.815 / 0.345 / 0.197` | No-Go |
| ExtraTrees `n600/leaf5` | `2.816` | `1.225` | `0.537` | `1.098 / 0.558 / 0.321` | No-Go |
| pairwise logistic | `1.912` | `0.558` | `0.297` | `0.921 / 0.387 / 0.301` | No-Go; top5 roughly tied only |

Interpretation:

- MC16 broad500 is a clean complete dataset and confirms balanced first/second
  coverage with no action truncation or schema failures.
- MC32 relabel materially changes many selected labels, so the MC16 labels are
  not reliable enough by themselves for T1 adoption.
- MC32 selected full-action relabeling still has a severe slow-target tail:
  only `124 / 250` selected targets completed before workers stalled.
- The partial MC32 labels are useful diagnostic data, but the merged 500-row
  training smoke does not beat the existing T1 baseline on strict holdout.
- T1 runtime adoption, T1 production-scale generation, and T1 seat-swap remain
  No-Go.

Updated next step:

1. Do not promote any Stage1j model from the merged 500-row smoke.
2. Build a timeout-aware / target-level MC32 refinement path so slow selected
   targets are isolated instead of stalling whole shards.
3. Prefer `ShardTargets=1`, `VmCount=total_shards`, and a per-target runtime
   cap or slow-target skip list for the next MC32 selected run.
4. After a cleaner MC32 selected set is available, train against a strict
   heldout split and require top3/top5 regret to beat the existing baseline
   before any runtime seat-swap.

## Stage1j Timeout-Aware Refinement Path

Implemented after the `124 / 250` MC32 selected relabel stall.

Changes:

- `ofc_regular.relabel_hu_turn1_refinement_targets` accepts:
  - `--target-timeout-seconds`
  - `--skip-output`
  - `--continue-on-target-error`
- timed-out / failed targets are written as
  `hu_turn1_stage1_refinement_relabel_skip_v1` rows instead of silently
  disappearing.
- per-shard summaries now include:
  - `target_attempts`
  - `skipped_targets`
  - `timed_out_targets`
  - `failed_targets`
  - `skip_reason_counts`
- local chunk runner accepts:
  - `-TargetTimeoutSeconds`
  - `-PartTimeoutSeconds`
  - `-ContinueOnTargetError`
- GCP relabel launcher accepts:
  - `-TargetTimeoutSeconds`
  - `-ShardTimeoutSeconds`
  - `-ContinueOnTargetError`
- GCP receive aggregation now keeps skipped/timeout counts.

Recommended next GCP run shape for the missing `126` selected targets:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-GcpHuTurn1RefinementRelabelRun.ps1 `
  -RunName r-t1j-mc32-timeout-missing126-YYYYMMDD `
  -TargetInput outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/analysis/missing_targets.jsonl `
  -TotalTargets 126 `
  -ShardTargets 1 `
  -BaseSeed 2026063801 `
  -VmCount 126 `
  -MachineType e2-highcpu-4 `
  -FutureSamples 32 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -TargetTimeoutSeconds 420 `
  -ShardTimeoutSeconds 540 `
  -ContinueOnTargetError `
  -CreateInstances
```

Receive with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Receive-GcpHuTurn1RefinementRelabelRun.ps1 `
  -RunName r-t1j-mc32-timeout-missing126-YYYYMMDD `
  -OutputDir outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_timeout_missing126 `
  -AllowPartial
```

Decision rule:

- completed relabel rows are usable as selected MC32 labels.
- timed-out targets should not block the dataset; keep them in skip artifacts
  and only revisit with lower MC or a tiny MC64 audit if they are repeatedly
  important.
- T1 runtime seat-swap remains No-Go until a trained model beats the current
  baseline on a strict heldout split.

GCP smoke:

- run:
  `r-t1j-mc32-timeout-smoke2-20260625-063424`
- input:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/analysis/missing_targets.jsonl`
- output:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_timeout_smoke2`
- settings:
  `TotalTargets=2`, `ShardTargets=1`, `VmCount=2`, `FutureSamples=32`,
  `TargetTimeoutSeconds=180`, `ShardTimeoutSeconds=300`
- aggregate:
  - expected/completed shards: `2 / 2`
  - missing shards: `0`
  - target attempts: `2`
  - relabel records: `0`
  - skipped targets: `2`
  - timed-out targets: `2`
  - failed targets: `0`
- interpretation:
  The timeout-aware path is wired correctly on GCP. A low `180s` timeout was
  deliberately enough to prove skip aggregation; use the recommended `420s`
  target cap for the real missing-126 run.

## Stage1j MC32 Missing126 Timeout Completion

Timeout-aware relabel run for the `126` previously missing selected targets:

- run:
  `r-t1j-mc32-timeout-missing126-20260625-064554`
- output:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_timeout_missing126_final`
- target input:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/analysis/missing_targets.jsonl`
- settings:
  `FutureSamples=32`, `ShardTargets=1`, `TargetTimeoutSeconds=420`,
  `ShardTimeoutSeconds=540`, `ContinueOnTargetError=true`
- expected/completed shards: `126 / 126`
- missing shards: `0`
- target attempts: `126`
- relabel records: `104`
- skipped targets: `22`
- timed-out targets: `22`
- failed targets: `0`
- skip reason counts:
  `target_timeout=21`, `shard_timeout=1`
- best action changed:
  `71 / 104` (`68.27%`)
- seat split:
  `first=60`, `second=44`
- no residual GCP VMs after receive.

Combined MC32 refinement labels:

- partial124:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_partial124/hu_turn1_refinement_relabel.jsonl`
- missing126 completion:
  `outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel_mc32_timeout_missing126_final/hu_turn1_refinement_relabel.jsonl`
- combined refinement:
  `outputs/hu_turn1_stage1j_stratified_teacher/stage1j_mc16_broad500_plus_mc32_timeout_refinement/hu_turn1_refinement_relabel_mc32_combined.jsonl`
- combined rows:
  `228`

Merged teacher:

- output:
  `outputs/hu_turn1_stage1j_stratified_teacher/stage1j_mc16_broad500_plus_mc32_timeout_refinement/hu_turn1_stage1j_merged_teacher.jsonl`
- summary:
  `outputs/hu_turn1_stage1j_stratified_teacher/stage1j_mc16_broad500_plus_mc32_timeout_refinement/merge_summary.json`
- records:
  `500`
- replaced records:
  `228`
- label sources:
  `base_fast_t2=272`, `stage9f_p2_refinement=228`

Training smoke on the merged 228-refinement teacher:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | baseline top1/top3/top5 | decision |
|---|---:|---:|---:|---:|---|
| HGB `leaf31/lr0.05/l2=1` | `1.661` | `0.583` | `0.355` | `1.030 / 0.671 / 0.444` | TopK diagnostic only; top3/top5 improve, top1 worsens |
| HGB `leaf3/lr0.01/l2=1` | `1.236` | `0.500` | `0.386` | `0.705 / 0.332 / 0.205` | No-Go |
| ExtraTrees `n600/leaf5` | `2.767` | `1.188` | `0.440` | `1.100 / 0.509 / 0.271` | No-Go |
| pairwise logistic | `1.738` | `0.607` | `0.293` | `1.040 / 0.332 / 0.240` | No-Go |
| listwise torch `256,128/e30` | `2.166` | `0.654` | `0.368` | `0.706 / 0.415 / 0.380` | No-Go |

Interpretation:

- The timeout-aware path recovered usable MC32 selected labels and removed the
  stalled-shard blocker.
- The larger selected refinement set still does not produce a T1 model that
  beats the existing baseline on top1 holdout regret.
- HGB `leaf31/lr0.05/l2=1` is the only model with improved TopK candidate
  coverage (`top3/top5` regret lower than the baseline), but it is not a direct
  action policy and should not be promoted.
- T1 runtime seat-swap, production-scale T1 teacher generation, and production
  adoption remain No-Go.

Next step:

1. Treat HGB `leaf31/lr0.05/l2=1` as a diagnostic TopK candidate-generator
   only.
2. If continuing T1, evaluate it with a small independent TopK confirm path
   before spending on larger T1 teacher generation.
3. Do not run T1 production seat-swap unless TopK confirm shows positive
   fired whole-game delta and enough fire count.

Leaf31 TopK confirm smoke:

- profile:
  `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- candidate model:
  `models/hu_turn1_stage1j_mc16_broad500_plus_mc32_timeout_hgb_leaf31_lr005_l2_1.pkl`
- config:
  `k3/mc4/d0/confirm8/cse1/pd0/seat=first+second`
- output:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_smoke20/summary.json`
- decision log:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_smoke20/hu_turn1_decisions.jsonl`
- audit:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_smoke20/hu_turn1_topk_confirm_audit/`
- paired seeds / hands:
  `20 / 40`
- EV/hand:
  `0.0000`
- HU T1 decisions:
  `40`
- HU T1 realized override count:
  `2`
- realized override delta mean:
  `0.0000`
- non-fired final mismatch:
  `0`

Interpretation:

- The new HGB leaf31 candidate-generator wiring works with independent TopK
  confirm.
- Non-fired cancellation is preserved.
- The run is far too small and under-fired for quality evidence. It only
  supports running a larger fired-count diagnostic if T1 remains the active
  line.

Leaf31 TopK fire-count diagnostic:

- run:
  `r-t1j-leaf31-topk-fire500-20260625`
- profile:
  `stage9f_p2_hu_t1_topk_confirm` vs `stage9f_p2`
- candidate model:
  `models/hu_turn1_stage1j_mc16_broad500_plus_mc32_timeout_hgb_leaf31_lr005_l2_1.pkl`
- config:
  `k3/mc4/d0/confirm8/cse1/pd0/seat=first+second`
- output:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_fire500/summary.json`
- decision log:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_fire500/hu_turn1_decisions.jsonl`
- audit:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_fire500/hu_turn1_topk_confirm_audit/`
- paired seeds / hands:
  `500 / 1000`
- EV/hand:
  `-0.0101`
- HU T1 decisions / valid decisions:
  `1000 / 929`
- valid override count / rate:
  `71 / 7.64%`
- realized per-fire delta:
  `-0.1844`
- realized per-fire CI95:
  `[-2.6737, 2.3049]`
- estimated EV/decision:
  `-0.0141`
- non-fired final mismatch:
  `0`
- non-fired counterfactual nonzero:
  `8`
- p95 loss:
  `16.5`
- no residual GCP VMs after receive.

Interpretation:

- The larger fire-count diagnostic is cleanly received and confirms that
  non-fired final actions remain baseline-aligned.
- The candidate fires enough for a first real diagnostic, but realized
  whole-game per-fire delta is slightly negative and the aggregate
  seat-swap EV/hand is also negative.
- Confirm-delta remains gate-diagnostic only; it is not performance evidence.
- HGB leaf31 TopK should not be promoted to a T1 runtime policy or production
  seat-swap candidate.
- Next T1 work should not spend more on this exact leaf31 TopK gate. If T1 is
  resumed, change the objective/model/gate design rather than extending this
  diagnostic.
