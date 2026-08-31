# OFC Pineapple Tutor T1/T3 Strengthening Report

Date: 2026-05-23

## Goal

Strengthen the position evaluation model by adding targeted T1/T3 teacher data instead of precomputing every branch.

## Baseline

Model:
`ai/models/candidate_runs/joint-t0-aa-k20-trips45-ft-20260521/model/action_value_best.pt`

Route10 evaluation:

| scope | top1 | top3 | top10 | avg_regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 40.0% | 65.0% | 91.2% | 1.882 | 3.007 | 7.4% | 9.2% |
| T1 | 35.0% | 70.0% | 95.0% | 2.510 | 2.811 | 7.9% | 7.7% |
| T3 | 40.0% | 65.0% | 85.0% | 2.406 | 3.637 | 9.5% | 24.3% |

Primary weak spots were T1 and T3: high regret, FL-heavy decisions, AA/KK route selection, and bust-risk handling.

## Data Generated

Target extraction:

- Weak spots from Route10 evaluation: 16 decisions
- Self-play active targets: 500 decisions
- Local teacher labels generated so far: 100 self-play decisions at 300 sims for MC turns, plus 16 weak-spot decisions
- Combined reranker data: 116 decisions, 2,172 candidate samples

Generated data:

- `ai/data/tutor_route10_20260522/active_teacher_t1_t3_weakspots/`
- `ai/data/tutor_route10_20260522/active_teacher_t1_t3_from_selfplay/`
- `ai/data/tutor_route10_20260522/active_teacher_t1_t3_combined/`

## Experiments

### Active-only fine-tune

Checkpoint:
`ai/models/candidate_runs/tutor-route10-t1t3-active-ft-20260523/model/action_value_best.pt`

Result: not suitable as a replacement. T1 top1 and regret improved, but overall and T3 degraded.

| scope | top1 | top3 | top10 | avg_regret |
|---|---:|---:|---:|---:|
| overall | 37.5% | 62.5% | 87.5% | 2.108 |
| T1 | 45.0% | 65.0% | 90.0% | 1.759 |
| T3 | 35.0% | 60.0% | 90.0% | 3.057 |

### Base + active 16x mix

Training data:
`D:/ofc_data/tutor-route10-t1t3-active16x-mix-20260523/reranker_base_active16x`

Checkpoint:
`ai/models/candidate_runs/tutor-route10-t1t3-active16x-mix-ft-20260523/model/action_value_best.pt`

| scope | top1 | top3 | top10 | avg_regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 38.8% | 66.2% | 91.2% | 1.641 | 3.251 | 7.1% | 9.3% |
| T1 | 35.0% | 65.0% | 95.0% | 1.343 | 3.560 | 8.5% | 7.4% |
| T3 | 40.0% | 60.0% | 90.0% | 2.307 | 4.642 | 10.3% | 25.3% |

### Base + active 8x mix

Training data:
`D:/ofc_data/tutor-route10-t1t3-active8x-mix-20260523/reranker_base_active8x`

Checkpoint:
`ai/models/candidate_runs/tutor-route10-t1t3-active8x-mix-ft-20260523/model/action_value_best.pt`

| scope | top1 | top3 | top10 | avg_regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 38.8% | 66.2% | 93.8% | 1.640 | 3.236 | 7.1% | 9.3% |
| T1 | 35.0% | 65.0% | 95.0% | 1.343 | 3.489 | 8.3% | 7.5% |
| T3 | 40.0% | 60.0% | 90.0% | 2.307 | 4.618 | 10.5% | 25.1% |

## Decision

Do not replace the production baseline yet.

The mixed models reduced average regret, especially at T1, but top1 did not improve and T3 top3 recall dropped. The 8x mix is the best candidate for further testing because it improves overall regret and top10 while keeping the model closer to the baseline.

## Next Step

Generate the remaining 400 self-play active labels or run the same target set on GCP, then retrain the base+active mix. T3 should be judged with exact T3/T4 labels where possible, because the current Route10 merged evaluation still contains estimated 96-sim labels for many T3 candidates.
