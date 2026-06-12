# Regular OFC AI Handoff Status

Last updated: 2026-06-12 JST

This document is a handoff note for another AI or engineer. It summarizes the
Regular OFC rules implemented in this repository and the current status of the
AI/model work. Treat this as a project status snapshot, not as a production
approval document.

## 1. Variant And Rules

The target game is Regular OFC Pineapple, heads-up focused.

Core rule implementation:

- Rule file: `src/ofc_regular/rules.py`
- Evaluator: `src/ofc_regular/evaluator.py`
- HU terminal scoring: `src/ofc_regular/teacher.py`

### Deck And Fantasyland

- Jokers are not used.
- `REGULAR_RULES.include_jokers = False`
- Fantasyland entry conditions are the same as the previous FL-entry rule:
  - top row QQ or better
  - top row trips
- In this regular mode, every FL entry type receives 14 cards:
  - QQ: 14
  - KK: 14
  - AA: 14
  - trips: 14
- FL stay also receives 14 cards.
- FL stay condition is unchanged:
  - trips on top, or
  - quads or better on bottom

Current default FL EV:

- `DEFAULT_FL_EV = {14: 12.196164}`

### Board And Bust Rule

Board shape:

- top/front: 3 cards
- middle: 5 cards
- bottom/back: 5 cards

The board is busted/fouled if:

- top hand strength is greater than middle hand strength, or
- middle hand strength is greater than bottom hand strength

If busted, row royalties and FL entry are not awarded for that board.

### Royalties

Top/front:

- Pair of 6s or better: `pair_rank - 5`
- Trips: `10 + rank - 2`

Middle:

- Royal flush: 50
- Straight flush: 30
- Quads: 20
- Full house: 12
- Flush: 8
- Straight: 4
- Trips: 2

Bottom/back:

- Royal flush: 25
- Straight flush: 15
- Quads: 10
- Full house: 6
- Flush: 4
- Straight: 2

### Heads-Up Terminal Score

Implemented in `_heads_up_terminal_score`.

If both players bust:

- score = 0

If hero busts and opponent does not:

- score = `-6 - opponent_royalty - opponent_FL_EV`

If opponent busts and hero does not:

- score = `+6 + hero_royalty + hero_FL_EV`

If neither busts:

- each row is worth +1 / -1 / 0
- sweeping all 3 rows adds another +3 or -3
- final score adds:
  - hero royalties minus opponent royalties
  - hero FL EV minus opponent FL EV

So a clean 3-row scoop is worth +6 before royalties and FL EV.

## 2. Turn Naming Used In This Work

The project uses turn labels such as T0, T1, T2, T3, and T4.

Practical interpretation in the current training/evaluation work:

- T0: opening 5-card placement
- T1/T2/T3: intermediate Pineapple turns
- T4/final: final completion turn

For HU selective override work:

- T3 means the late turn where the Stage7 HU T3 continuation model is applied.
- T2 means the earlier HU turn now being evaluated for selective override or
  candidate generation.

Do not assume a model is production-ready just because a turn label has a
checkpoint. Each turn has separate gate status below.

## 3. Fixed Baseline / Continuation Assumptions

### Current Self-Board Baseline Set

The baseline model set previously used as the fixed reference was:

- `models/opening_stage7_torch_wide.pt`
- `models/turn1_stage6_torch_wide.pt`
- `models/turn2_stage8.pkl`
- `models/turn3_stage6.pkl`

These are baseline/reference artifacts. They are not all equally strong, and
they are not all HU-aware.

### Fixed HU T3 Continuation

HU T3 uses Stage7_candidate_A as a selective override only.

Production/default runtime for the T3 continuation:

- model: `models/hu_turn3_stage7_reference_override_cached_rank_wide.pt`
- `hu_turn3_min_margin = 5.0`
- `hu_turn3_reference_min_margin = 10.0`
- fallback/default policy: Stage3 HU margin10

Important:

- Stage7 is not a full replacement.
- Stage7 only overrides Stage3 when the runtime safety gates pass.
- If model load, feature generation, legality, NaN/inf, or other safety checks
  fail, the action falls back to Stage3.
- `margin0.25` is explicitly excluded from production presets.

Current status:

- T3 Stage7_candidate_A m5_r10 is the fixed continuation policy for T2 work.
- It is considered usable as a conservative HU T3 selective override.

## 4. HU T2 Stage8 / Stage8b Status

T2 is not production-ready.

The main lesson so far:

- Teacher/oracle filters can identify strong-looking T2 improvements.
- Direct runtime proxy gates have not yet converted that teacher advantage into
  stable seat-swap EV.
- Stage8b is now being explored as a candidate generator for TopK + MC rerank,
  not as a direct production override gate.

### Stage8 20k Broad Training

Model:

- `models/hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt`

Teacher/evaluation foundation:

- 20k MC512 teacher data was generated.
- Teacher EV feature cache and replay-ready calibration artifacts were built.
- C1e and C1f calibration passed.

Important distinction:

- C1f `confidence_lcb*` filters use MC512 teacher EV LCB.
- These are oracle/calibration filters.
- They must not be used directly as runtime gates.

### Stage8 Direct Runtime Proxy Result

C2-small initially showed a positive tendency:

- EV/hand: +0.0173
- 95% CI: [-0.0035, +0.0382]

C3 larger seat-swap did not reproduce this:

- `m2.5_r0_g0.9`: EV/hand -0.0058, 95% CI [-0.0315, +0.0199]
- `m2.75_r0_g0.9`: EV/hand -0.0058
- `m2.5_r0_g0.925`: EV/hand -0.0058

Decision:

- execution: pass
- decision: No-Go
- production/P2 fixed: No-Go
- 50k teacher: No-Go
- T1 training: No-Go

Interpretation:

- Stage8 teacher/calibration work was useful.
- The current runtime proxy gate was too weak or underfired.
- Threshold search alone is not the right next step.

### Stage8b Safe Override Labels

Stage8b was created to address the gap between teacher oracle LCB and runtime
proxy gates.

Stage8b label prep status:

- 20,000 rows
- safe_lcb196 positive / gray / negative: 2,917 / 15,218 / 1,865
- hard negatives: 15
- high-MC label overrides: 50
- training smoke: passed

Purpose:

- Learn a runtime-available safe override/confidence signal.
- Reduce false positives and top-loss cases.
- Use `safe_override_probability` as the key gate signal instead of relying only
  on older `predicted_delta + gate_probability`.

Stage8b large training was allowed as a hypothesis test. It is still not a
production candidate by itself.

### Stage8b Direct Selective Override C3 Result

The direct/selective Stage8b runtime gate still did not reach production
evidence.

Decision:

- execution: pass
- decision: No-Go
- production/P2 fixed: No-Go
- 50k teacher: No-Go
- T1 training: No-Go

Current conclusion:

- Do not deploy T2 Stage8b as a direct selective override yet.
- Do not fix Stage8b as P2.
- Do not start T1 training based on this T2 policy.

## 5. Stage8b TopK + MC Rerank Experiment

The latest direction is to use Stage8b only as a candidate generator, then use
MC rerank against a reduced action set.

Implemented CLI:

- `src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py`

Supporting implementation:

- `src/ofc_regular/hu_turn2_teacher_data.py`
  - `evaluate_hu_turn2_actions(..., action_indices=...)`
  - MC rollout can now evaluate only selected legal action indices.

Concept:

1. Enumerate all legal HU T2 actions.
2. Score actions with Stage8b.
3. Keep TopK candidates plus the baseline action.
4. Run MC only on that reduced action set.
5. Override only if MC rerank clears the configured gates.

This is not production. It is an experiment to test whether the model is better
as a candidate generator than as a direct gate.

### Small Validation Result

Output directory:

- `outputs/evals/hu_turn2_stage8b_topk_mc_rerank/`

Run scale:

- seeds: `2026062101,2026062102,2026062103`
- games per seed: 30
- paired total: 90
- `--seed-stride 1000000`

Results:

| config | paired | EV/hand | CI low | CI high | overrides | override rate | avg MC gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `k3_mc64_d0.5_se0_seatfirst` | 90 | +0.2233 | -0.3163 | +0.7629 | 26 | 14.44% | +2.1009 |
| `k5_mc64_d0.25_se0_seatfirst` | 90 | +0.0611 | -0.5944 | +0.7166 | 36 | 20.00% | +1.7655 |
| `k3_mc64_d0.25_se1.5_seatfirst` | 90 | -0.0333 | -0.0987 | +0.0320 | 2 | 1.11% | +4.9449 |
| `k3_mc64_d0.25_se0_seatfirst` | 90 | -0.1111 | -0.4998 | +0.2776 | 34 | 18.89% | +1.6513 |

Decision:

- execution: pass
- decision: No-Go
- production/P2 fixed: No-Go
- 50k teacher: No-Go
- T1 training: No-Go

Interpretation:

- The TopK + MC path works mechanically.
- It fires much more often than the direct gate.
- The validation sample is far too small and confidence intervals are very wide.
- Promising configs for further validation are:
  - `k3_mc64_d0.5_se0_seatfirst`
  - `k5_mc64_d0.25_se0_seatfirst`
- Runtime is roughly about one second per reranked T2 decision at MC64 in the
  small validation. This is acceptable for analysis/adviser use or limited
  runtime testing, but not yet a proven fast production path.

## 6. Current Go / No-Go Summary

Current decisions:

- T3 Stage7_candidate_A m5_r10 selective override: usable as fixed T3 continuation
- T2 Stage8 broad teacher/training: useful foundation
- T2 Stage8 direct runtime gate: No-Go
- T2 Stage8b direct selective override: No-Go
- T2 Stage8b TopK + MC rerank: execution pass, decision No-Go pending larger validation
- 50k teacher generation: No-Go
- T1 training: No-Go
- production/P2 fixed T2 policy: No-Go

Use the phrase "execution pass / decision No-Go" when a run completed correctly
but did not provide enough evidence to adopt the policy.

## 7. Things Not To Mix Up

Do not confuse these:

- Teacher EV LCB filter:
  - uses MC teacher EV
  - oracle/calibration-only
  - not runtime-deployable

- Runtime proxy gate:
  - uses only runtime-available fields
  - examples: predicted delta, safe override probability, model rank, predicted
    EV margin, position
  - must be validated with seat-swap

- T3 reference margin:
  - `hu_turn3_reference_min_margin = 10.0`
  - validated as part of the Stage7 T3 selective override runtime

- T2 `reference_margin_raw`:
  - T2-scale diagnostic/reference value
  - not equivalent to T3 r10
  - must not be copied as a hard production gate without validation

Also:

- Always preserve `dead_cards` in runtime logs intended for replay or high-MC
  audit.
- Legacy logs without `dead_cards` are replay-ineligible.
- Preserve action identity with `original_index` or a stable action encoding.
  Do not trust a local action index if actions were filtered/reordered.
- Seat-swap evaluations must use non-overlapping seeds and `--seed-stride`.

## 8. Recommended Next Work

### Next Best Step

Run a larger TopK + MC rerank validation, preferably on Spot VM if local runtime
is too slow.

Recommended first larger configs:

- `k3/mc64/d0.5/se0/seat=first`
- `k5/mc64/d0.25/se0/seat=first`

Recommended validation size:

- 300 to 500 paired games per seed
- 3 to 5 non-overlapping seeds
- keep `--seed-stride`
- keep T3 continuation fixed to Stage7_candidate_A m5_r10

If MC64 looks stable, test MC128 on the best one or two configs.

### What To Measure

For each config:

- EV/hand
- 95% CI
- seed breakdown
- first/second position breakdown
- override rate
- average MC gain on override
- p95/p99 loss
- top override losses
- latency per reranked decision
- no-override reasons

### When To Move Forward

Move to a larger C-style validation only if:

- aggregate EV/hand is positive or close to positive,
- CI lower bound is not strongly negative,
- multiple seeds point the same way,
- override rate is not near zero,
- tail losses are acceptable,
- first/second or seat-specific behavior is understood,
- runtime latency is acceptable for the intended use case.

Still do not start T1 or 50k teacher until T2 has a stable policy or a clear
decision is made to use a baseline T2 continuation.

## 9. Useful Commands

Small TopK + MC smoke:

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 1 `
  --seeds 2026062101 `
  --seed-stride 1000000 `
  --configs "k3/mc8/d0.25/se0/seat=first" `
  --hu-turn2-stage8b-model models\hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt `
  --output-dir outputs\evals\hu_turn2_stage8b_topk_mc_rerank_smoke `
  --device auto `
  --prediction-threads 1 `
  --opening-lookahead-samples 8 `
  --progress-every 1 `
  --write-decision-log
```

Small validation command already used:

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 30 `
  --seeds 2026062101,2026062102,2026062103 `
  --seed-stride 1000000 `
  --configs "k3/mc64/d0.25/se0/seat=first,k5/mc64/d0.25/se0/seat=first,k3/mc64/d0.5/se0/seat=first,k3/mc64/d0.25/se1.5/seat=first" `
  --hu-turn2-stage8b-model models\hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt `
  --output-dir outputs\evals\hu_turn2_stage8b_topk_mc_rerank `
  --device auto `
  --prediction-threads 1 `
  --opening-lookahead-samples 32 `
  --progress-every 10 `
  --write-decision-log
```

Targeted tests:

```powershell
python -m pytest -p no:cacheprovider tests\test_hu_turn2_stage8b_training_prep.py
```

Full test command:

```powershell
python -m pytest -p no:cacheprovider
```

## 10. Current Bottom Line

T3 is stable enough to keep fixed as Stage7_candidate_A m5_r10.

T2 is still under active research:

- Stage8/Stage8b contain useful signal.
- Direct runtime override gates have not passed C3.
- Stage8b TopK + MC rerank is the current most promising direction, but only
  has a small execution-pass / decision-No-Go validation so far.

Do not deploy T2, do not mark it P2, do not start T1, and do not launch 50k
teacher generation until a larger TopK + MC or revised gate validation gives
stable seat-swap evidence.
