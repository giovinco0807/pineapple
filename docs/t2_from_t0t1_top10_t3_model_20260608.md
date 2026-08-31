# T2 From T0/T1 Top10 via T3 Model - 2026-06-08

## Purpose

Generate T2 candidate-ranking data from realistic branch-expanded boards:

1. deal a two-player root,
2. expand the target player's T0 and T1 choices with model Top10,
3. advance the opponent with model Top1 by default,
4. at the target player's T2 decision, enumerate every legal T2 action,
5. score each T2 action by sampling T3 deals and choosing the best T3 action with a T3 action-value model.

This is a fast model-label path, not a full exact T2 oracle.

## Implementation

- Script: `ai/tutor/build_t2_t3_model_teacher_from_t0t1_topk.py`
- Default T0 model: `ai/models/candidate_runs/tutor-route10-t0top128-route3-mc30-1000-ft-20260525/model/action_value_best.pt`
- Default T1 model: `ai/models/candidate_runs/t1-runtime112-top40regret-x20-plus-currenthard15-x20-ft-20260604/model/action_value_best.pt`
- Default opponent T2 model: `ai/models/candidate_runs/t2-mix-broad80k-hard121-x10-ft-20260603/action_value_best.pt`
- Default T3 BB model: `D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/t3-20k-plus-gen5k-hardneginit-bb/action_value_best.pt`
- Default T3 BTN model: `D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/t3-20k-plus-gen5k-hardneginit-btn/action_value_best.pt`

## Local Smoke

- Output: `D:/ofc-pineapple-data/t2_t0t1_top10_t3_model_20260608/smoke_r1_pos5_s2/t2_t3_model_rows.jsonl`
- Records: 10
- Position split: BB 5, BTN 5
- T2 candidates: 249
- T3 states scored: 7,704
- Elapsed: 2.5s

## Root1 Full

- Output: `D:/ofc-pineapple-data/t2_t0t1_top10_t3_model_20260608/root1_full_s2/t2_t3_model_rows.jsonl`
- Summary: `D:/ofc-pineapple-data/t2_t0t1_top10_t3_model_20260608/root1_full_s2/t2_t3_model_rows.summary.json`
- Records: 200
- Position split: BB 100, BTN 100
- Target T0/T1 TopK: 10
- Opponent T0/T1 TopK: 1
- Opponent T2 TopK for BTN records: 1
- T3 draw limit per T2 candidate: 2
- T2 candidates: 4,842
- Avg candidates/record: 24.21
- T3 states scored: 149,850
- Elapsed: 27.4s
- Throughput: 7.30 records/s, 5,467 T3 states/s

## Root1 Distribution

- Candidate count: min 12, mean 24.21, max 27
- Best T3-model EV: min 1.269, mean 14.699, max 51.416
- Top1 minus Top2 EV gap: min 0.050, median 2.569, mean 3.793, max 28.114
- Near ties with gap < 0.1: 2 / 200
- Large gaps with gap > 5: 58 / 200

## Notes

- BTN T2 records see the BB board after BB takes a model Top1 T2 action, matching the actual action order.
- BB T2 records see the opponent board after BTN T1. This is the correct visible board at the BB T2 decision, but the T3-model label is still an approximation of future hidden opponent T2 play.
- The output rows contain all legal T2 actions sorted by `model_t3_value_score`, plus predicted T3 continuation bust/FL averages.
- For training, start with this as a cheap broad label source, then promote high-loss or close-call rows to capped Rust/exact labels.
