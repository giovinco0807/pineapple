# T2 Teacher Generation Status - 2026-06-08

## Purpose

Generate local T2 teacher labels without using VMs, using capped Rust T2 oracle runs backed by the existing `t3_exact_solver.exe`.

The goal is not to claim full T2 exact yet. The current labels are pilot `cap10/cap20` stable labels, useful for validating data format, finding MC300 label noise, and starting hard-case mining.

## Outputs

### Random T2 pilot

- Input: `ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl`
- Output dir: `D:/ofc-pineapple-data/t2_teacher_20260608/pilot_local500_cap10_20_limit5`
- Stable labels: `D:/ofc-pineapple-data/t2_teacher_20260608/pilot_local500_cap10_20_limit5/t2_stable/stable_t2_oracle_cap10_cap20.jsonl`
- Records: 5
- cap10/cap20 stable Top1: 5/5
- Source MC300 Top1 agreement:
  - cap10: 1/5
  - cap20: 1/5
- cap20 average elapsed: 60.2s / record
- cap20 max regret of source Top1: 6.672 EV

### High EV-loss hard-miss pilot

- Miss source: `D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_selector_benchmark/local500_limit100_selector_s1/teacher_misses.jsonl`
- Input: `D:/ofc-pineapple-data/t2_teacher_20260608/hard_miss_local500_top5_cap10_20/hard_miss_top5_input.jsonl`
- Output dir: `D:/ofc-pineapple-data/t2_teacher_20260608/hard_miss_local500_top5_cap10_20`
- Stable labels: `D:/ofc-pineapple-data/t2_teacher_20260608/hard_miss_local500_top5_cap10_20/t2_stable/stable_t2_oracle_cap10_cap20.jsonl`
- Records: 5
- Position split: BB 3, BTN 2
- Candidate count: min 12, mean 19.2, max 24
- cap10/cap20 stable Top1: 5/5
- Source MC300 Top1 agreement:
  - cap10: 3/5
  - cap20: 3/5
- cap20 average elapsed: 47.2s / record
- cap20 max regret of source Top1: 3.710 EV
- Total elapsed: 353.8s

## Notes

- A stronger `cap100/cap200` local run was attempted on 20 T2 records but was stopped after about 20 minutes with only 4 cap100 records completed. Local high-cap T2 oracle generation is too slow for bulk generation.
- The stable label rows include `board`, `opponent_board`, `known_discards`, `dealt`, all candidate actions, `best_idx`, candidate `ev`, `bust_prob`, `fl_rate`, `fl_type_rates`, and nested `exact.samples`.
- T2 source labels from MC300 are visibly noisy: even low-cap oracle labels change Top1 frequently. Future training should use oracle-generated labels, not raw MC300 Top1, as the ranking target.
- Top1 can be ambiguous when multiple actions tie in EV. Training and evaluation should emphasize TopK recall plus EV loss, not Top1 accuracy alone.

## Next Step

Build a repeatable local shard runner for T2 `cap10/cap20` labels, prioritizing:

1. high EV-loss misses,
2. diverse random T2 spots,
3. BB/BTN balance,
4. later promotion of a smaller subset to higher caps such as `cap20/cap50`.
