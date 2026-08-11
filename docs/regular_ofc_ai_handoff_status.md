# Regular OFC AI Handoff Status

Last updated: 2026-07-16 JST

This document is a handoff note for another AI or engineer. It summarizes the
Regular OFC rules implemented in this repository and the current status of the
AI/model work. Treat this as a project status snapshot, not as a production
approval document.

## 2026-07-16 full audit and revised execution order

The current local working tree has a correct normal-play information-set
adapter, deterministic ActionKey/RNG infrastructure, the M2 Python late-street
teacher, and the M3 Rust search engine. Full local regression on this audit was
`2379 passed, 2 skipped`; M3 Rust library/runner tests were `36 + 2` passed.
These are correctness results, not policy-strength certification.

The fixed chain remains operational only as an explicit legacy baseline:

```text
stage19_p0 -> stage18_p1 -> stage9f_p2 -> stage7_m5_r10
```

- T2 Stage9f and T3 Stage7/reference weights were trained from caches that
  encoded realized opponent private discards.
- P0/P1 decision features do not have the same direct leak, but their labels,
  selectors, and old acceptance evaluations used that downstream chain.
- M3.0 now provides an exact both-seat T4 runtime as an explicit compositional
  `m30_exact` option. `legacy` remains the default; no named profile or
  `current` mapping was changed. The fixed runtime's legacy first-seat shortcut
  remains available only as the rollback comparator.
- The local safety boundary is not yet reflected in upstream HEAD. Do not claim
  the GitHub branch is information-set safe until R0 is reviewed, committed,
  pushed, and reverified.

M4 through M4.3 Attempt13 are closed development history, not a completed T1
second-seat policy. Attempt13 passed 16 of 18 gates but failed fired-root p95
and maximum-loss gates; it did not open Audit50, fit, population acceptance,
runtime activation, or promotion. Its T2 continuation was the quarantined
`stage9f_p2`, so its evidence is valid only as legacy-chain architecture
diagnosis.

Authoritative next order:

```text
R0 safe release boundary
  -> M3.0 both-seat exact T4 source freeze
  -> M3.1 safe T3 reference/candidate rebuild
  -> M3.2 safe T2 both-seat rebuild and promotion
  -> M4R T1 both-seat rebuild
  -> M5 T0 both-seat rebuild
  -> M6 population self-play, ABR/NashConv approximation, and restricted CFR
```

No Spot-scale generation starts before local correctness, determinism,
scalar/batch/exact parity, profiling, and 100 -> 1,000 root pilots pass. No
legacy EV, teacher score, model top-1 accuracy, or holdout threshold reselection
may be used as promotion evidence. The authoritative detailed gates are in
`docs/hu_joint_policy_implementation_milestones.md`; the machine-readable
supersession is `configs/hu_joint_policy_revised_roadmap_20260716.json`.

## 2026-07-16 M3.0 exact T4 component

M3.0 is complete locally as an explicit opt-in T4 component. First seat
computes `max_a E_uniform-deal[min_b terminal HU score]` across all 2,024
three-card opponent deals and every legal exact response. Second seat fully
enumerates terminal placements against the completed opponent board. The API
returns every legal semantic ActionKey and its EV, not only a predicted top-1
placement.

The information boundary is fail-closed: only `ActorObservation` is accepted;
unknown fields, opponent private discards, realized deck tails, and world/replay
state aliases are rejected by Python and by scalar/batch Rust. The release DLL,
engine version, and SHA are pinned, with no runtime build or silent fallback.

The disjoint 100-root and 1,000-root pilots passed. The 1,000-root first-seat
p95/p99 latency was `42.91/54.53 ms`, second-seat p99 was `1.70 ms`, and batch
throughput was `156.04 roots/s`. A separate fresh 100-paired counterfactual
run produced `+0.295 EV/hand`, 95% CI `[+0.1204, +0.4696]`; 11 overrides averaged
`+5.36` realized points, with 0 false positives, zero observed tail loss, and
full-trajectory identity on all 89 non-fired pairs.

This is exact under the declared uniform exchangeable restart belief. It is not
a learned full-Bayesian posterior, a Nash-equilibrium proof, or mathematical
full-game optimality. The fixed P0/P1/P2/T3 chain remains quarantined, so M3.0
does not by itself certify the joint policy. See
`docs/hu_joint_policy_m30_t4_completion_audit.md` and
`configs/hu_joint_policy_m30_t4_runtime.json`.

## M1 information-set audit supersedes legacy promotion claims

The 2026-07-12 M1 audit found that the source feature caches used by the fixed
T2 Stage9f and T3 Stage7 models contained both players' private discards in the
legacy `dead_cards` feature and did not contain a versioned
`policy_observation`.  P0/P1 evaluations also used those continuations.  The
profiles and model files remain frozen as explicit legacy baselines, but every
old `Go`, EV, confidence interval, and acceptance statement below is historical
only under the required hidden-opponent-discard rules.  It is not valid
promotion evidence for the new joint policy.

Do not change `current`, delete the legacy chain, or silently retrain it. New
promotion evidence must use ActorObservation-only feature caches and belief-
conditioned teachers, then pass fresh paired seat-swap holdouts. M2 and M3 are
complete as teacher/search infrastructure; until safe T3 and then T2 artifacts
are rebuilt, policy promotion and large-scale data generation remain `No-Go`.
The authoritative implementation status is
`docs/hu_joint_policy_implementation_milestones.md`.

## 0. Latest Update 2026-07-12

HU T0 Stage19 is accepted as the fixed first-seat P0 selective override.

Accepted P0 runtime:

- profile: `stage19_p0`
- candidate model:
  `models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl`
- safe selector:
  `models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl`
- runtime: `topk60 / first margin 0.5 / first selector 0.6`
- allowed seat: `first`
- fallback and T1 continuation: `stage18_p1`
- T2 continuation: `stage9f_p2`
- T3 continuation: `stage7_m5_r10`
- mode: selective override only
- rollback/off profile: `stage18_p1` / `stage19_p0_off`

Fresh preregistered evidence:

- paired seeds / T0 events: `100000 / 200000`
- realized fires: `3747`
- EV/hand: `+0.017303`, 95% CI `[+0.008292, +0.026314]`
- realized per-fire delta: `+0.9236`, 95% CI `[+0.4434, +1.4037]`
- non-fired cancellation errors: `0`
- p99 realized loss: `35.767`
- independent worst-30 MC512 audit: mean `+0.8255`, negative rate `13.33%`
- formal acceptance verifier: `Go`
- current-code smoke: `10` paired seeds / `20` T0 rows, clean fallback logging
- full regression: `779 passed`
- final GCP RUNNING-instance audit: `0`

Current fixed chain:

```text
stage19_p0 -> stage18_p1 -> stage9f_p2 -> stage7_m5_r10
```

`current` remains unchanged. Second-seat T0 and T0 full replacement remain
outside the accepted scope. Detailed audit:
`docs/hu_turn0_stage19_p0_completion_audit.md`.

Older status sections below are retained as experiment history. Where they
conflict with this latest section, this section and the completion audits for
P0/P1/P2 take precedence.

## 0a. Previous Update 2026-07-11

HU T1 Stage18 is accepted as the fixed first-seat P1 continuation for
downstream T0 work.

Accepted T1 P1 runtime:

- profile: `stage18_p1`
- candidate model:
  `models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl`
- safe selector:
  `models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl`
- runtime:
  `k5/mc8/d3/confirm32/cse1.5/pd1.5/seat=first/safe0.7`
- fallback and T2 continuation: `stage9f_p2`
- T3 continuation: `stage7_m5_r10`
- mode: first-seat selective override only
- rollback/off profile: `stage9f_p2`

Fresh preregistered C6 evidence:

- artifact:
  `outputs/evals/hu_turn1_stage18_c6_safe07_cse15_d3_pd15_250k_final/`
- paired seeds / T1 decisions: `250000 / 500000`
- valid realized fires: `340`
- realized per-fire delta: `+3.5103`
- realized per-fire 95% CI: `[+1.9683, +5.0523]`
- estimated EV/decision: `+0.002389`
- estimated EV/decision 95% CI: `[+0.001339, +0.003438]`
- whole-hand EV: `+0.002387`
- invalid overrides: `0`
- non-fired cancellation errors: `0`
- p95 / p99 / max fired loss: `24.227 / 30.837 / 39.227`
- formal acceptance verifier: pass

Current decision:

- first-seat T1 P1 continuation: `Go`
- downstream T0 work: may use `stage18_p1`
- second-seat T1 override: `No-Go`
- T1 full replacement: `No-Go`
- global `current` replacement: not performed

Metric rule:

- Performance claims use realized fired whole-game seat-swap counterfactual
  deltas.
- Confirm-MC delta remains a gate diagnostic only.

Detailed audit:
`docs/hu_turn1_stage18_p1_completion_audit.md`

Older T1 `No-Go` statements below are retained as historical experiment
records. They are superseded for the accepted first-seat P1 scope by this
2026-07-11 section; they still apply to second-seat and full-replacement T1.

## 0a. Previous Update 2026-06-25

HU T2 Stage9f is accepted as the fixed P2 continuation profile for
post-acceptance experiments.

Accepted T2 P2 runtime:

- profile: `stage9f_p2`
- evidence profile: `stage9f_cse2_csemax2_bothseat`
- model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T3 continuation: `stage7_m5_r10`
- mode: selective override only, not full replacement
- rollback/off profile: `stage9f_off`

Primary evidence:

- C4 validation:
  `outputs/evals/hu_turn2_stage9f_bothseat_c4_chunked_40x12000_m5r10/`
  - paired hands `218116`
  - realized fires `4000`
  - estimated EV/hand `+0.02580`
  - per-fire delta `+2.79986`
  - per-fire 95% CI `[+2.51686, +3.08286]`
  - first/second per-fire deltas `+2.72177 / +2.87458`
  - non-fired cancellation clean
- production-like profile canary:
  `outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/`
  - decisions `60000`
  - realized fires `547`
  - estimated EV/decision `+0.02386`
  - per-fire 95% CI `[+1.87247, +3.36150]`
  - first/second CI lows `+1.58773 / +1.53843`
  - replay-ready `60000 / 60000`
  - p95 latency `895.28ms`
- guarded canary:
  `outputs/evals/hu_turn2_stage9f_guarded_production_canary/`
  - verifier status `pass`
  - decisions `20000`
  - realized fires `174`
  - estimated EV/decision `+0.02892`
  - per-fire 95% CI `[+1.87929, +4.76905]`
  - first/second CI lows `+0.93869 / +1.64419`
  - replay-ready `20000 / 20000`
  - p95 latency `907.40ms`

Post-goal wiring smoke:

- artifact:
  `outputs/evals/hu_turn2_stage9f_p2_post_goal_smoke20/`
- command profile: `stage9f_p2` vs `stage7_m5_r10`
- paired seeds / hands / decisions: `20 / 40 / 40`
- realized overrides: `1`
- realized fired delta: `+8.0`
- non-fired nonzero realized deltas: `0`
- replay-ready rows: `40 / 40`
- p95 latency: `598.21ms`

Current decision:

- T2 as P2 continuation: `Go`
- T2 production default: still not flipped; explicit enable required
- T2 full replacement: `No-Go`
- 50k teacher: not needed unless a later experiment shows a new data need
- T1/T0 work may use `stage9f_p2` as the fixed T2 continuation

Important metric rule:

- Confirm-MC deltas are gate diagnostics only.
- Performance claims use realized fired whole-game paired deltas.
- Non-fired cancellation must remain clean in future T2/T1 evaluations.

## 0b. Previous Update 2026-06-23

HU T1 Stage1 was advanced from selected `200` relabels to full `2,000`
Stage9f P2 relabels:

- full2k relabel artifact:
  `outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_full2k_relabel/stage1_all2k_targets_stage9f_p2_relabel_merged.jsonl`
- merged teacher:
  `outputs/hu_turn1_stage1_pilot/merged_stage9f_p2_full2k/hu_turn1_stage1_fast2k_stage9f_p2_relabel_full2k.jsonl`
- best HU T1 model:
  `models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl`
- cross-seed holdout result:
  candidate mean avg regret `7.90`, old HU HGB `9.82`, old self ridge `9.97`

Runtime support was added for a validation-only profile:

- profile: `stage9f_p2_hu_t1_stage1`
- T2/T3 continuation: same as `stage9f_p2`
- HU T1 candidate is selective, not full replacement
- current T1 margin gate:
  `DEFAULT_HU_TURN1_STAGE1_MIN_MARGIN = 1.0`

Seat-swap smoke is No-Go for adoption:

- full replacement smoke, `100` paired seeds:
  EV/hand `-0.6268`, 95% CI `[-1.8922, +0.6386]`
- margin `5.0` smoke:
  no T1 effect, EV/hand `0.0000`
- margin `1.0` smoke:
  EV/hand `-0.2895`, 95% CI `[-1.0207, +0.4417]`
- T1 decision-log audit for margin `1.0`:
  fired valid rows `29`, realized fired-delta mean `-2.1003`, hard-negative
  rows `12`, non-fired nonzero count `0`
- analysis:
  `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_margin1/`
- Larger 500-paired-seed T1 audit:
  - matchup:
    `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_500_margin1_t1log.json`
  - decision log:
    `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/seat_swap_500_margin1_hu_t1.jsonl`
  - analysis:
    `outputs/evals/hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_500_margin1/`
  - full-hand EV/hand: `-0.4247`, 95% CI `[-0.7863, -0.0631]`
  - fired valid T1 rows: `149`
  - fired realized delta mean: `-2.0506`, 95% CI `[-4.1191, +0.0179]`
  - fired losses/wins/zeros: `54 / 37 / 58`
  - extracted labeled fired targets: positives `37`, hard negatives `54`,
    neutral `58`

Interpretation:

- Full2k relabeling and HGB tuning improved teacher-holdout regret.
- That improvement did not yet transfer to real seat-swapped EV.
- T1 runtime adoption remains `No-Go`.
- Do not start production, P2 fixed status, or larger T1 seat-swap from this
  model. The next useful T1 work is a selective override objective/evaluation
  built around realized action deltas, not a full replacement.
- Use `--hu-turn1-decision-output` and
  `python -m ofc_regular.analyze_hu_turn1_decision_log` for future T1 gate
  work. Prediction margins are gate diagnostics only; realized fired-decision
  deltas are the performance metric.
- The current margin-only T1 runtime gate is now a stronger No-Go after the
  500-paired-seed audit. The useful output is the labeled fired target set for
  training a future safe-override head, not a runtime candidate.

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

- `DEFAULT_FL_EV = {14: 10.227020614683454}`
- Source config: `configs/fl_ev_regular_2k.json`
- This value is a direct HU FL-vs-normal fixed-point estimate. The previous
  chain estimate, `12.196164`, is retained only as legacy metadata.

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

- T3 means the late turn where the HU T3 continuation policy is applied.
  Hidden-discard T2 work defaults that continuation to the Stage3 HU reference
  policy; Stage7 `m5_r10` is an explicit opt-in legacy/experiment setting.
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

- Under the old open-discard/leaky-info evaluation, T3 Stage7_candidate_A
  `m5_r10` looked like a strong conservative selective override.
- Under the corrected hidden-discard information model, `m5_r10` is no longer
  confirmed as a positive fixed continuation policy.
- Do not launch new hidden-discard T2/T1 teacher generation assuming Stage7
  `m5_r10` is fixed. Use a Stage3-only continuation as the conservative
  baseline, or retrain/recalibrate T3 under the hidden-discard objective first.

Post-FL-EV recalibration check:

- After changing the default 14-card FL EV from `12.196164` to
  `10.227020614683454`, Stage7_candidate_A `m5_r10` was rerun against the
  Stage3 margin10 baseline.
- Output: `outputs/evals/stage7_candidate_A_fl_ev_10p227_m5r10_3k/`
- Scale: 3 golden seeds, `1,000` paired seeds per seed, `3,000` paired total.
- Aggregate EV/hand: `+0.0930`
- seed-mean 95% CI: `[+0.0649, +0.1212]`
- production override rate: `3.52%`
- seed EV/hand: `+0.1208`, `+0.0728`, `+0.0855`

Conclusion:

- The T3 selective override did not break under the new FL EV objective alone,
  but this result was still from the old open-discard/leaky-info path.

Hidden-discard recheck after fixing discard visibility:

- Output: `outputs/evals/stage7_m5_r10_hidden_discard_3k/`
- Scale: 3 non-overlapping seed blocks, `1,000` paired seeds per block,
  `3,000` paired total.
- Seeds: `2026062601`, `2027062601`, `2028062601`.
- Evaluator: `evaluate_matchups`, profile A `stage7_m5_r10`, profile B
  `stage3_baseline`.
- Weighted EV/hand: `-0.001341`
- seed-mean 95% CI: `[-0.132458, +0.129776]`
- Seed EV/hand: `-0.085362`, `+0.130841`, `-0.049500`
- Note: this evaluator measures paired seat-swap EV but does not log
  production override rate.

Updated conclusion:

- `m5_r10` is execution-pass but decision No-Go as a hidden-discard fixed T3
  continuation.
- The old `+0.0930` result should be treated as legacy open-discard evidence.
- Before further serious T2 work under standard hidden-discard rules, either
  retrain/recalibrate HU T3 on hidden-discard teacher data or fall back to a
  Stage3-only T3 continuation baseline.

T2 teacher implementation guard:

- `src/ofc_regular/hu_turn2_teacher_data.py` now defaults to
  `--t3-continuation stage3_reference_default`.
- This records `continuation_policy_T3 = Stage3_HU_reference_default` and uses
  the HU Stage3 reference action directly in the T2 rollout continuation.
- Stage7 `m5_r10` is still available only by explicitly passing
  `--t3-continuation stage7_m5_r10`.
- Do not read this T2 teacher default as a fresh hidden-discard validation of
  Stage7. It is a safety default to avoid silently generating new hidden-discard
  T2 labels on top of old Stage7 assumptions.
- The local shard scripts expose the same guard:
  - `scripts/Run-HuTurn2TeacherShard.ps1 -T3Continuation stage3_reference_default`
  - `scripts/Run-HuTurn2TeacherChunksParallel.ps1 -T3Continuation stage3_reference_default`
- The GCP pilot launcher also exposes `-T3Continuation`; it writes
  `t3_continuation` into the manifest and shard status JSON.

T2 evaluation implementation guard:

- `src/ofc_regular/evaluate_hu_turn2_stage8_seat_swap.py` now exposes
  `--t3-continuation`, with hidden-discard default
  `stage3_reference_default`.
- `src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py` exposes the
  same option for TopK + MC rerank validation.
- Both evaluators write `t3_continuation` and `t3_continuation_policy` into the
  summary/CSV artifacts; do not compare runs unless this field matches.
- The local and GCP wrappers for Stage8 C3 seat-swap and Stage8c TopK per-fire
  evaluation also expose/pass the same value:
  - `scripts/Run-HuTurn2Stage8C3LargerSeatSwap.ps1 -T3Continuation stage3_reference_default`
  - `scripts/Start-GcpHuTurn2Stage8C3SeatSwapRun.ps1 -T3Continuation stage3_reference_default`
  - `scripts/Run-HuTurn2Stage8cTopkPerFireEval.ps1 -T3Continuation stage3_reference_default`
  - `scripts/Run-HuTurn2Stage8cFirstSeatC4.ps1 -T3Continuation stage3_reference_default`
  - `scripts/Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -T3Continuation stage3_reference_default`
- The selected high-MC / replay diagnostics also make the continuation
  explicit:
  - `src/ofc_regular/audit_hu_turn2_stage8_high_mc.py --t3-continuation stage3_reference_default`
  - `src/ofc_regular/replay_hu_turn2_stage8b_topk_hard_negatives.py --t3-continuation stage3_reference_default`
  - `scripts/Start-GcpHuTurn2Stage8C4HighMcRun.ps1 -T3Continuation stage3_reference_default`
- Stage7 `m5_r10` remains explicit opt-in via
  `--t3-continuation stage7_m5_r10`.

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

### Post-FL-EV Recheck

After adopting `DEFAULT_FL_EV = {14: 10.227020614683454}`, the two promising
TopK + MC configs were rerun with the same small scale:

- output: `outputs/evals/hu_turn2_stage8b_topk_mc_rerank_fl_ev_10p227/`
- seeds: `2026062101,2026062102,2026062103`
- games per seed: `30`
- paired total: `90`
- `--seed-stride 1000000`
- T3 continuation: Stage7_candidate_A `m5_r10` opt-in / legacy-continuation
  setting

Results:

| config | paired | EV/hand | CI low | CI high | overrides | override rate | avg MC gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `k3_mc64_d0.5_se0_seatfirst` | 90 | +0.2124 | -0.2846 | +0.7094 | 24 | 13.33% | +2.1216 |
| `k5_mc64_d0.25_se0_seatfirst` | 90 | +0.0611 | -0.5573 | +0.6795 | 36 | 20.00% | +1.6664 |

Interpretation:

- The result is mechanically consistent with the old-FL-EV small validation.
- `k3/mc64/d0.5` remains the better of the two small-sample candidates.
- The confidence intervals are still too wide, and seed `2026062102` is
  negative for both configs.
- This does not justify production, P2-fixed status, T1 training, 50k teacher,
  or a large Spot VM run by itself.
- Treat this as execution pass / decision No-Go until a better validation
  design controls MC selection bias or a larger run shows stable seat-swap EV.

### Two-Stage Independent Confirm MC

To reduce winner's curse in TopK + MC rerank, the evaluator now supports a
two-stage rerank mode:

- Stage A: existing Stage8b TopK plus baseline, using `mc_samples`.
- Stage B: if Stage A selects a non-baseline champion, reroll only champion vs
  baseline with an independent future seed and `confirm_mc_samples`.
- The Stage B gate uses the independent paired delta estimate and can require
  `confirm_delta >= confirm_se_multiplier * confirm_delta_se`.
- Config tokens:
  - `confirm128` or `confirm_mc=128`
  - `cse2` or `confirm_se_multiplier=2`
- Output artifact:
  - `conditional_override_metrics.csv`
  - `cancellation_audit.csv`
  - `predicted_delta_safety_audit.csv`
- The evaluator now writes metric-role fields into seed/result rows,
  `conditional_override_metrics.csv`, `topk_rerank_summary.md`, and
  `go_nogo.md`:
  - `performance_metric_source=realized_fired_whole_game_delta`
  - `per_fire_performance_column` / `hand_ev_performance_column`
  - `rerank_delta_metric_role=gate_diagnostic_only`
  - `confirm_delta_metric_role=gate_diagnostic_only`
  - `confirm_delta_performance_claim_allowed=false`

Small confirm run:

- output:
  `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm128_small/`
- seeds: `2026062101,2026062102,2026062103`
- games per seed: `30`
- paired total: `90`
- configs:
  - `k3/mc64/confirm128/d0/se0/cse2/seat=first`
  - `k5/mc64/confirm128/d0/se0/cse2/seat=first`

Whole-hand seat-swap result:

| config | paired | EV/hand | CI low | CI high | overrides | override rate |
|---|---:|---:|---:|---:|---:|---:|
| `k3_mc64_d0_se0_confirm128_cse2_seatfirst` | 90 | -0.0333 | -0.0987 | +0.0320 | 4 | 2.22% |
| `k5_mc64_d0_se0_confirm128_cse2_seatfirst` | 90 | -0.0333 | -0.0987 | +0.0320 | 5 | 2.78% |

Important correction:

- The confirm MC delta is independent of Stage A, but it is still used by the
  confirm gate. Averaging only confirm deltas that passed `cse2` is a truncated
  selected sample and is not an unbiased per-fire estimate.
- Confirm delta must be treated as a gate diagnostic only.
- The unbiased small-run per-fire estimate must come from realized seat-swap
  counterfactual deltas on fired decisions, or from a third independent
  post-selection estimate.

Cancellation audit on this run:

- Non-fired realized deltas were exactly zero for both configs.
- Therefore the whole-hand EV difference is attributable to fired decisions in
  this small run.

Realized fired-decision estimate:

| config | overrides | non-fired nonzero | realized per-fire delta | confirm delta mean | realized EV/hand |
|---|---:|---:|---:|---:|---:|
| `k3_mc64_d0_se0_confirm128_cse2_seatfirst` | 4 | 0 | -1.5000 | +3.0003 | -0.0333 |
| `k5_mc64_d0_se0_confirm128_cse2_seatfirst` | 5 | 0 | -1.2000 | +2.9492 | -0.0333 |

Interpretation:

- The independent confirm stage is working mechanically; Stage A and Stage B
  use different future digests, and inflated Stage A candidates are often
  rejected by Stage B.
- However, `confirm128/cse2` still shows confirm-gate selection bias: fired
  decisions have strongly positive confirm deltas but negative realized
  seat-swap deltas in this small sample.
- `cse2` also underfires at this small scale.
- This is still not a production or P2 result. The next useful local/small-VM
  experiment should target enough fired decisions and use realized per-fire
  deltas as the primary metric. Do not select thresholds by average confirm
  delta. If using `confirm128`, start with a looser `cse1` run to gather at
  least 50 fires, then compare `cse1.5` and `cse2` only after the realized
  per-fire distribution is visible.

Confirm256 check:

- output:
  `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm256_small/`
- same 3 seeds and 30 paired seeds per seed.
- configs:
  - `k3/mc64/confirm256/d0/se0/cse2/seat=first`
  - `k5/mc64/confirm256/d0/se0/cse2/seat=first`

Results:

| config | paired | EV/hand | realized fires | realized per-fire | confirm delta mean | non-fired nonzero |
|---|---:|---:|---:|---:|---:|---:|
| `k3_mc64_d0_se0_confirm256_cse2_seatfirst` | 90 | -0.1056 | 10 | -1.9000 | +1.9878 | 0 |
| `k5_mc64_d0_se0_confirm256_cse2_seatfirst` | 90 | +0.0000 | 9 | +0.0000 | +2.2509 | 0 |

Interpretation:

- Increasing confirm MC from 128 to 256 did not produce positive realized
  per-fire evidence.
- `k3` is worse under confirm256 and should not be promoted.
- `k5` is neutral in this tiny run, but with only 9 fires and p95/max realized
  loss still visible it is not a Go signal.
- Do not launch 50k, T1, or production based on these results.

Failure postmortem:

- output:
  `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_failure_postmortem/`
- Combined fired rows from confirm128 and confirm256: `28`
- Realized loss rows: `6`
- All loss rows were diagnosed as `mc_overrode_negative_model_delta`.
- In these cases the Stage8b model's predicted delta for the MC champion was
  negative, but Stage A/B MC still promoted the action.
- Example repeated loss:
  `hand_seed=2026062101000023`, realized delta `-6`, confirm delta stayed
  strongly positive across `confirm128` and `confirm256`.
- The evaluator now writes `predicted_delta_safety_audit.csv` and mirrors the
  blocker in `go_nogo.md`. If a run has realized losses from fired actions with
  model `predicted_delta < 0`, it is a decision `No-Go` regardless of noisy
  aggregate EV.
- Backfilled audit results:
  - `confirm128`: k3 had `4/4` negative-predicted-delta overrides and `1`
    realized loss (`-6`); k5 had `4/5` and `1` realized loss (`-6`).
  - `confirm256`: k3 had `10/10` negative-predicted-delta overrides and `3`
    realized losses (`-19` total); k5 had `8/9` and `1` realized loss (`-6`).

Guard check:

- A deployable `min_predicted_delta` guard was added to the evaluator.
- Config token examples:
  - `pd0`
  - `pred0`
  - `min_predicted_delta=0`
- Output:
  `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm256_pd0_small/`
- Same seeds, `confirm256/cse2/pd0`.

Results:

| config | paired | EV/hand | overrides | realized per-fire | below_predicted_delta |
|---|---:|---:|---:|---:|---:|
| `k3_mc64_d0_se0_confirm256_cse2_pd0_seatfirst` | 90 | +0.0000 | 2 | +0.0000 | 23 |
| `k5_mc64_d0_se0_confirm256_cse2_pd0_seatfirst` | 90 | +0.0000 | 2 | +0.0000 | 30 |

Interpretation:

- `pd0` removes the observed loss pattern, but it also removes nearly all
  firing.
- This confirms the current TopK+MC path is mostly trying to overrule the
  learned Stage8b delta head rather than refine high-confidence positive model
  candidates.
- Next useful work is not 50k/T1/production. Either redesign the candidate
  pool so it starts from model-positive actions, or use these false positives
  as hard negatives for the next model/gate iteration.

Model-positive candidate pool check:

- The evaluator now applies `min_predicted_delta` during candidate-pool
  construction, not only as a post-MC safety fallback.
- Output:
  `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm256_pd0_rank3_small/`
- Configs:
  - `k3/mc64/confirm256/d0/se0/cse2/pd0/rank3/seat=first`
  - `k5/mc64/confirm256/d0/se0/cse2/pd0/rank3/seat=first`

Results:

| config | paired | EV/hand | overrides | realized per-fire | topk_empty |
|---|---:|---:|---:|---:|---:|
| `k3_mc64_d0_se0_confirm256_cse2_pd0_seatfirst_rank3` | 90 | +0.0000 | 1 | +0.0000 | 70 |
| `k5_mc64_d0_se0_confirm256_cse2_pd0_seatfirst_rank3` | 90 | +0.0000 | 0 | +0.0000 | 70 |

Interpretation:

- Requiring model-positive, rank-limited candidates removes the bad overrides
  but leaves essentially no usable T2 fires.
- This is evidence that current Stage8b does not surface enough deployable
  positive T2 override candidates.
- Do not keep tuning MC thresholds on this model. The next serious T2 step is
  model/teacher revision: add these false positives as hard negatives, improve
  deployable positive-label learning, or regenerate labels under the corrected
  FL EV before another runtime sweep.

Hard negative replay pack:

- Tool:
  `python -m ofc_regular.prepare_hu_turn2_stage8b_topk_hard_negatives`
- Output:
  `outputs/evals/hu_turn2_stage8b_topk_hard_negatives/`
- Inputs:
  - `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm128_small/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm256_small/runtime_decisions.jsonl`
- Dedupe key: canonical state signature plus final action signature.

Pack summary:

| group | rows | replay-ready | realized mean | confirm mean |
|---|---:|---:|---:|---:|
| false_positive | 3 | 3 | -6.3333 | +2.7684 |
| neutral_overconfirm | 13 | 13 | +0.0000 | +2.3225 |
| all_fired_deduped | 18 | 18 | -0.7222 | +2.3396 |

Important:

- `topk_false_positive_hard_negatives.jsonl` is replay-ready and should be
  used as hard-negative input for the next T2 model/gate iteration.
- Do not train directly from confirm MC deltas. Replay or feature-generate
  these state/action pairs under the current FL EV objective first.
- These examples reinforce the current No-Go for production, T1, and 50k
  teacher generation.

## 6. Current Go / No-Go Summary

Current fixed explicit profile chain:

- P0: `stage19_p0`, first-seat T0 selective override, `Go`
- P1: `stage18_p1`, first-seat T1 selective override, `Go`
- P2: `stage9f_p2`, guarded T2 continuation, `Go`
- T3: `stage7_m5_r10`, selective override, `Go`
- FL EV: `10.227020614683454`
- visibility model: hidden opponent discards / own private discards only

Scope limits:

- `current` is unchanged; the fixed chain requires explicit profile selection.
- second-seat P0 and P1 overrides are `No-Go`.
- full replacement at T0/T1/T2/T3 is not accepted.
- older Stage8/Stage8b direct-gate No-Go results remain historical evidence and
  do not describe the later accepted `stage9f_p2` runtime.

## 7. Critical Caveats After Code Audit

These are high-priority caveats. Do not spend large Spot VM budget or start T1
training until they are either fixed or explicitly accepted as part of the game
variant.

### 7.1 Discard Visibility / Information Model

Earlier runtime and teacher-generation code used a shared `dead_cards` list
during HU play and passed it into each policy together with the opponent's
public board:

- `src/ofc_regular/play_ai.py`
- `src/ofc_regular/evaluate_matchups.py`
- `src/ofc_regular/evaluate_stage7_production_candidate.py`
- `src/ofc_regular/trace_hu_turn3_overrides.py`
- `src/ofc_regular/self_play_teacher_data.py`
- `src/ofc_regular/hu_self_play_teacher_data.py`
- `src/ofc_regular/hu_turn2_teacher_data.py`

In standard Pineapple OFC, the opponent's discards are hidden. A player should
only condition on the opponent's public board plus their own private discards.
Showing both players' accumulated discards is an information leak and a
rule-model mismatch.

The leak also matters because dead-card masks are part of model features, for
example in the HU T3 Stage3 feature manifest.

Current implementation status:

- `play_ai.play_hand`, `evaluate_matchups.trace_hand`, and
  `evaluate_stage7_production_candidate.trace_hand_with_records` now pass only
  `opponent_board.all_cards() + own_private_discards` as policy-visible
  `dead_cards`.
- `trace_hu_turn3_overrides.play_traced_hand` and its counterfactual rollout
  helper use the same hidden-discard visibility model.
- `self_play_teacher_data.py`, `hu_self_play_teacher_data.py`, and
  `hu_turn2_teacher_data.py` now keep true unavailable discards separate from
  policy-visible dead cards. Teacher records can preserve replay-grade
  `dead_cards`, while model features and policy calls use only public opponent
  board cards plus the acting player's private discards.
- HU T3/T2 teacher helpers also use the hidden-discard fallback when
  `visible_dead_cards` is omitted: if `hero_private_discards` is present, the
  actor-visible dead cards are `opponent_board.all_cards() + hero_private`.
  HU T2 candidate-pool records now preserve both private discard lists so later
  replay/regeneration does not have to infer visibility from leaky `dead_cards`.
- T2 runtime decision logs intended for high-MC replay now separate
  replay-grade true unavailable cards from actor-visible information:
  `dead_cards` is the true unavailable discard set, `visible_dead_cards` is the
  policy-visible set, and `hero_private_discards` / `opponent_private_discards`
  are preserved explicitly.
- C1e/cache, high-MC audit, and TopK hard-negative replay readiness checks now
  require all four replay fields above. Rows missing hidden-discard replay
  metadata are marked replay-ineligible instead of being silently replayed from
  ambiguous `dead_cards`.
- `mine_hu_turn3_states.py` and `label_hu_turn3_states.py` now preserve
  hidden-discard metadata for mined HU T3 states:
  `visibility_model = hidden_discard`, `discard_visibility = own_private_only`,
  full `discarded_cards`, actor-visible `visible_dead_cards`, and both players'
  private discard lists.
- Old runtime/evaluation/teacher artifacts generated before this fix should be
  treated as open-discard/leaky-info artifacts and are not directly comparable
  to new hidden-discard evaluations.
- If this project intentionally models an open-discard variant later, the rule
  document must say so explicitly and these hidden-discard guards should not be
  mixed with open-discard teacher labels.

Hidden-discard smoke after the fix:

- Output dir: `outputs/hidden_discard_smoke/`
- T2 teacher smoke:
  `outputs/hidden_discard_smoke/hu_turn2_teacher_smoke.jsonl`
- T2 summary:
  `outputs/hidden_discard_smoke/hu_turn2_teacher_smoke_summary.json`
- T2 teacher Stage3-continuation default smoke:
  `outputs/hidden_discard_smoke/hu_turn2_teacher_stage3_continuation_smoke.jsonl`
- T2 teacher Stage3-continuation summary:
  `outputs/hidden_discard_smoke/hu_turn2_teacher_stage3_continuation_smoke_summary.json`
- T2 teacher Stage7 explicit opt-in smoke:
  `outputs/hidden_discard_smoke/hu_turn2_teacher_stage7_optin_smoke.jsonl`
- T2 teacher Stage7 explicit opt-in summary:
  `outputs/hidden_discard_smoke/hu_turn2_teacher_stage7_optin_smoke_summary.json`
- HU T3 self-play teacher smoke:
  `outputs/hidden_discard_smoke/hu_self_play_turn3_smoke.jsonl`
- self-board Turn3 teacher smoke:
  `outputs/hidden_discard_smoke/self_play_turn3_smoke.jsonl`
- T2 smoke record check:
  - `dead_cards` preserved true unavailable discards: `["7h", "Ah"]`
  - `visible_dead_cards` contained the opponent public board plus hero private
    discard `7h`
  - opponent private discard `Ah` was not in `visible_dead_cards`
- T2 continuation smoke check:
  - default run reported `t3_continuation = stage3_reference_default`,
    `continuation_policy_T3 = Stage3_HU_reference_default`, and
    `stage7_model_called_count = 0`
  - explicit opt-in run reported `t3_continuation = stage7_m5_r10`,
    `continuation_policy_T3 = Stage7_candidate_A_m5_r10`, and non-zero Stage7
    model calls
- T2 Stage8 seat-swap evaluator smoke:
  `outputs/hidden_discard_smoke/hu_turn2_stage8_seat_swap_stage3_continuation_smoke/`
  recorded `t3_continuation = stage3_reference_default` and
  `t3_continuation_policy = Stage3_HU_reference_default` in recommended,
  grid, seed, summary, and Go/No-Go artifacts.
- T2 Stage8b TopK evaluator smoke:
  `outputs/hidden_discard_smoke/hu_turn2_topk_stage3_continuation_smoke/`
  recorded `t3_continuation = stage3_reference_default` and
  `t3_continuation_policy = Stage3_HU_reference_default` in result, seed,
  decision-log, summary, and Go/No-Go artifacts.
- Full test suite after the replay-readiness hardening:
  `269 passed`, including runtime, matchup, self-play teacher, HU teacher, T2
  teacher, HU T3 batch continuation, Stage8/Stage8b/Stage8c guard tests,
  TopK per-fire analysis, high-MC audit, hard-negative replay, pilot training,
  Stage8b training-prep, exact teacher, FL EV legacy guard, and old T2/T3 model
  fixture tests.

### 7.2 FL EV Calibration

Current default:

- `DEFAULT_FL_EV = {14: 10.227020614683454}`
- config: `configs/fl_ev_regular_2k.json`

The Rust FL solver at `rust/regular_fl_solver` does perform fixed-point
iteration over the stay bonus. The older shortcut EV formula used manually
chosen adjustments:

- `opponent_avg_royalty = 5.0`
- `line_scoop_advantage = 4.0`
- resulting legacy FL EV: `12.196164`

That shortcut overestimated the current-policy 14-card FL value by about 2
points because it did not charge for the opponent entering FL during the
hero's FL hand.

The adopted recalibration path is direct FL-vs-normal HU simulation:

- hero receives 14 cards and is placed by `solve_fantasyland(stay_bonus=V)`
- opponent plays a normal 13-card hand with the current baseline policy stack
- hero current-FL next bonus is awarded only by FL-stay, not by ordinary QQ+
  entry
- opponent normal-hand FL entry subtracts the same fixed-point value
- update `V_next = E[score]` until stable

Implementation and result:

- `src/ofc_regular/estimate_hu_fl_ev_direct.py`
- GCP run: `regular-hu-fl-ev-direct-20260612-161630`
- output: `outputs/fl_ev_direct_hu_gcp/fl_ev_direct_aggregate_summary.json`
- total solved: `10,000`
- weighted FL EV: `10.227020614683454`
- 95% CI: `[9.961025542748054, 10.493015686618854]`
- SE: `0.1357117713956127`
- hero stay rate: `0.1034`
- opponent FL entry rate: `0.2394`
- opponent bust rate: `0.3892`
- hero average royalty: `12.3477`
- opponent average royalty: `4.0413`

Sanity check:

- Normal `current` vs `current` evaluation over 500 paired seeds produced
  `2,000` counted boards with bust rate `0.358`, FL entry rate `0.249`, and
  average royalty with bust as zero `4.269`.
- These are close to the direct-FL opponent rates, so the high opponent bust
  and FL entry rates appear to be normal current-policy behavior rather than
  an obvious empty-hero-board OOD failure.

Remaining caveat:

- Models already trained before this update still encode labels generated under
  the old `12.196164` value. Runtime scoring, exact final-turn solving, and new
  teacher generation should use `10.227020614683454`, but model retraining and
  sensitivity checks are still needed before making production/P2 decisions.
- Teacher EV caches generated before this update are not valid for new
  calibration, LCB threshold selection, or safe-override label decisions under
  the `10.227020614683454` scoring objective. They may still be useful as
  candidate generators or historical diagnostics, but any EV/LCB/gate decision
  that depends on terminal rollout scores must be rerolled with the new value.
- Sensitivity checks should start at T3. T2 validation depends on the selected
  T3 continuation policy, so first verify or explicitly choose that
  continuation under the current objective and information model. Do not
  silently carry the old Stage7 `m5_r10` continuation into hidden-discard T2
  candidate reruns.
- FL EV and policy co-evolve. Re-estimate direct HU FL EV after a material
  baseline policy change, especially after fixing discard visibility, reducing
  standalone foul-risk bias, or materially changing FL pursuit behavior.

### 7.3 TopK + MC Rerank Gain Is Selection-Biased

The current TopK + MC rerank experiment uses MC estimates both to choose the
candidate action and to report the selected candidate's gain. This can inflate
`avg MC gain` because the selected action is the winner of noisy estimates.

For the next validation, prefer one of these designs:

- two-stage MC: select candidate with one random stream, then confirm selected
  candidate vs baseline with an independent random stream
- SE-gated override: require `delta >= k * SE`, not only a small fixed delta
- sequential halving: cheap MC for TopK pruning, higher MC only for finalists,
  then independent confirmation against baseline

The current small TopK + MC results are useful as an execution smoke and
candidate-generation signal. They are not production evidence.

### 7.4 Validation Power

For small expected effects such as +0.01 to +0.03 EV/hand, ordinary whole-hand
seat-swap estimates may need many games. When testing selective overrides, also
report:

- override rate
- per-override paired EV
- conditional paired CI on fired hands
- whole-hand EV as `override_rate * per_override_EV`

This avoids hiding the signal in thousands of non-fired hands. It also helps
identify whether a policy is genuinely positive but underfiring, or simply not
better than baseline.

## 8. Things Not To Mix Up

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

- Runtime logs intended for replay or high-MC audit must preserve:
  `dead_cards` as true unavailable discards, `visible_dead_cards` as
  actor-visible cards, and both `hero_private_discards` and
  `opponent_private_discards`.
- Legacy logs without the full hidden-discard replay packet are
  replay-ineligible.
- Preserve action identity with `original_index` or a stable action encoding.
- Treat old `12.196164` teacher EV caches as calibration-expired. Do not use
  the old 20k MC512 teacher EV, C1e/C1f cache, or Stage8b `safe_lcb196` labels
  to set new thresholds under the `10.227020614683454` objective without
  rerolling the terminal EVs.
- Reconfirm T3 Stage7_candidate_A `m5_r10` under the current hidden-discard
  objective before using it as a continuation policy for T2/T1 work.
  Do not trust a local action index if actions were filtered/reordered.
- Seat-swap evaluations must use non-overlapping seeds and `--seed-stride`.

## 9. Recommended Next Work

### Next Best Step

Run a larger TopK + MC rerank validation, preferably on Spot VM if local runtime
is too slow.

Do not size this run by total games alone. Size it by realized fired decisions:
the primary metric is realized per-fire delta from the seat-swap
counterfactual, not the confirm MC delta that was used as a gate input.

Recommended first larger configs:

- `k3/mc64/d0/se0/confirm128/cse1/pd0/seat=first`
- `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first`

Recommended validation size:

- at least 50 realized fired decisions per config before reading per-fire EV
- use `--target-realized-overrides-per-seed` and treat `--games-per-seed` as a
  hard maximum budget, not the actual target
- start with loose `cse1` to collect fired rows, then compare `cse1.5` and
  `cse2` only after the realized per-fire distribution is visible
- 3 to 5 non-overlapping seeds
- keep `--seed-stride`
- for new hidden-discard runs, keep T3 continuation explicit; default to
  `--t3-continuation stage3_reference_default` unless the run is intentionally
  labeled as Stage7 `m5_r10` opt-in

If confirm128/cse1 has positive realized per-fire evidence, test confirm256 on
the best one or two configs. Do not choose thresholds by mean confirm delta.

### What To Measure

For each config:

- EV/hand
- 95% CI
- seed breakdown
- first/second position breakdown
- override rate
- average confirm MC gain on override, for diagnostics only
- realized per-fire delta and CI from `conditional_override_metrics.csv`
- `cancellation_audit.csv`, especially non-fired nonzero deltas
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

## 10. Useful Commands

Small TopK + MC smoke:

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 1 `
  --seeds 2026062101 `
  --seed-stride 1000000 `
  --configs "k3/mc8/d0.25/se0/pd0/seat=first" `
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
  --configs "k3/mc64/d0.25/se0/pd0/seat=first,k5/mc64/d0.25/se0/pd0/seat=first,k3/mc64/d0.5/se0/pd0/seat=first,k3/mc64/d0.25/se1.5/pd0/seat=first" `
  --hu-turn2-stage8b-model models\hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt `
  --output-dir outputs\evals\hu_turn2_stage8b_topk_mc_rerank `
  --device auto `
  --prediction-threads 1 `
  --opening-lookahead-samples 32 `
  --progress-every 10 `
  --write-decision-log
```

Fire-count-target validation command:

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 5000 `
  --target-realized-overrides-per-seed 20 `
  --seeds 2026062601,2026062602,2026062603 `
  --seed-stride 1000000 `
  --configs "k3/mc64/d0/se0/confirm128/cse1/pd0/seat=first,k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first" `
  --hu-turn2-stage8b-model models\hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt `
  --output-dir outputs\evals\hu_turn2_stage8b_topk_confirm128_cse1_fire_target `
  --device auto `
  --prediction-threads 1 `
  --opening-lookahead-samples 32 `
  --progress-every 100 `
  --write-decision-log
```

This command uses `--games-per-seed 5000` only as a cap. Each config/seed
stops earlier when `target_realized_overrides_per_seed` realized fired rows are
available, so the resulting evaluation is sized by the unbiased per-fire
sample, not by total games.

Fire-target smoke after adding the CLI option:

- `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_fire_target_smoke/`
  - config: `k3/mc64/d0/se0/confirm128/cse1/pd0/seat=first`
  - cap: 20 paired, target: 1 realized fire
  - result: 0 fires, `stop_reason = max_games_reached`
- `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_k5_fire_target_smoke/`
  - config: `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first`
  - cap: 20 paired, target: 1 realized fire
  - result: 0 fires, `stop_reason = max_games_reached`
- `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_no_pd_smoke/`
  - config: `k5/mc64/d0/se0/confirm128/cse1/seat=first`
  - cap: 20 paired, target: 1 realized fire
  - result: stopped at 11 paired after 1 realized fire
  - realized per-fire delta: `-12.0`
  - `negative_predicted_delta_override_count = 1`
  - `safety_blocker = 1`

Interpretation: the new fire-target evaluator works, but this smoke reinforces
the prior diagnosis. Without the deployable `pd0` guard, TopK+MC can still
override model-negative actions and immediately produce a tail loss. With
`pd0`, the same short smoke underfires. Do not treat `cse1` alone as a safe
fix; it is only a way to collect fired rows for diagnosis.

TopK score-order smoke:

- `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_smoke/`
- configs:
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate`
- `bygate_delta` reached one realized fire after 8 paired seeds:
  - predicted delta: `+2.2642`
  - gate probability: `0.9997`
  - candidate EV rank: `1`
  - realized per-fire delta: `-12.0`
  - safety blocker: `0` because `pd0` was satisfied
- `bygate` reached one realized fire after 7 paired seeds:
  - realized per-fire delta: `0.0`
  - safety blocker: `0`

Interpretation: changing TopK order can increase fires under `pd0`, but
`bygate_delta` exposed a stronger hard negative that passes all current
deployable guards. This is not a runtime Go signal. It is useful training/audit
data.

Hard-negative extraction and replay for this smoke:

- Hard-negative pack:
  `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_hard_negatives/`
- Extracted:
  - false-positive hard negatives: `1`
  - neutral overconfirm rows: `1`
- Replay:
  `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_hard_negative_replay/`
- MC512 replay result for the false positive:
  - action mapping: `ok`
  - candidate EV: `-0.1268`
  - baseline EV: `+0.3689`
  - delta vs baseline: `-0.4957`
  - paired delta SE: `0.4356`
  - LCB196: `-1.3495`
  - label: `safe_lcb196 = negative`, `hard_negative_label = 1`

This specific row is a local T2 hard negative, not merely a whole-game
counterfactual loss. It should be feature-generated or included in the next
Stage8c-style hard-negative training pass after current-FL-EV cache generation.

The Stage8b prep script now accepts repeated `--topk-hard-negatives-jsonl`
inputs. A merged-prep smoke was written to:

- `outputs/hu_turn2_stage8b_prelarge_training_with_score_smoke_hard_negatives/`
- TopK runtime hard-negative replay states: `4`
- scoring metadata status: `missing`
- large training remains `No-Go` until the base feature cache is rebuilt under
  `fl_ev_14 = 10.227020614683454`.

Current-FL-EV Stage8c cache/training smoke:

- replay teacher:
  `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_hard_negative_replay/topk_hard_negative_replay_teacher.jsonl`
- combined cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_smoke_cache/`
- inputs:
  - `outputs/hu_turn2_current_fl_ev_train_smoke18/teacher.jsonl`
  - replay teacher above
- cache state count: `19`
- run buckets:
  - `teacher`: `18`
  - `topk_hard_negative_replay_teacher`: `1`
- source buckets:
  - `natural`: `18`
  - `topk_hard_negative_replay`: `1`
- gate labels:
  - gray: `12`
  - negative: `7`
- hard-negative state metadata:
  - `state_id = topk_hard_negative_replay:2026062601000007`
  - `sample_id = topk_hard_negative_replay:2026062601000007:14`
  - `hand_seed = 2026062601000007`
  - `teacher_label = negative`
  - `pilot_gate_label = negative`
- scoring objective:
  - `fl_ev_14 = 10.227020614683454`
  - `fl_ev_14_status = unique`
- training smoke:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_smoke/`
  - one epoch, tiny CPU model
  - train/val/test states: `11 / 4 / 4`
  - `scoring_metadata_status.status = match`
  - `training_allowed = true`

This proves the current objective can ingest TopK replay hard negatives into a
feature cache and reach the training entrance. It is not a quality result and
does not change the No-Go status for production, T1, P2, or 50k teacher.

Current-FL-EV Stage8c local60 MC16 follow-up smoke:

- teacher:
  `outputs/hu_turn2_current_fl_ev_stage8c_local60_mc16/teacher.jsonl`
  - states: `60`
  - source bucket: `natural`
  - future samples: `16`
  - scoring objective: `fl_ev_14 = 10.227020614683454`
  - runtime: about `50.5s` wall, about `0.84s/state`
- combined cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_local60_mc16_cache/`
  - inputs:
    - `outputs/hu_turn2_current_fl_ev_stage8c_local60_mc16/teacher.jsonl`
    - `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_hard_negative_replay/topk_hard_negative_replay_teacher.jsonl`
  - states: `61`
  - action rows: `1,475`
  - run buckets:
    - `teacher`: `60`
    - `topk_hard_negative_replay_teacher`: `1`
  - source buckets:
    - `natural`: `60`
    - `topk_hard_negative_replay`: `1`
  - gate labels:
    - gray: `38`
    - negative: `23`
  - rollout counts:
    - `16`: `60`
    - `512`: `1`
  - scoring objective:
    - `fl_ev_14_status = unique`
    - `fl_ev_14_missing_records = 0`
- training smoke:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_local60_mc16/`
  - model output:
    `models/hu_turn2_current_fl_ev_stage8c_local60_mc16_smoke.pt`
  - two epochs, small GPU smoke
  - train/val/test states: `41 / 10 / 10`
  - `scoring_metadata_status.status = match`
  - `training_allowed = true`

This follow-up only proves the current-FL-EV data path remains healthy after
adding more rows and replay hard negatives. The metrics are too small and noisy
for model-quality or deployment decisions.

Current-FL-EV Stage8c local300 MC16 follow-up smoke:

- teacher:
  `outputs/hu_turn2_current_fl_ev_stage8c_local300_mc16/teacher.jsonl`
  - states: `300`
  - source bucket: `natural`
  - future samples: `16`
  - scoring objective: `fl_ev_14 = 10.227020614683454`
  - runtime: about `293s` wall, about `0.98s/state`
  - rough sizing from this run:
    - `1k` natural MC16 states: about `16 minutes` local
    - `5k` natural MC16 states: about `1.4 hours` local, before bucket filters
- combined cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_local300_mc16_cache/`
  - inputs:
    - `outputs/hu_turn2_current_fl_ev_stage8c_local300_mc16/teacher.jsonl`
    - `outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_hard_negative_replay/topk_hard_negative_replay_teacher.jsonl`
  - states: `301`
  - action rows: `7,298`
  - run buckets:
    - `teacher`: `300`
    - `topk_hard_negative_replay_teacher`: `1`
  - source buckets:
    - `natural`: `300`
    - `topk_hard_negative_replay`: `1`
  - gate labels:
    - gray: `188`
    - negative: `113`
  - rollout counts:
    - `16`: `300`
    - `512`: `1`
  - scoring objective:
    - `fl_ev_14_status = unique`
    - `fl_ev_14_missing_records = 0`
- training smoke:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_local300_mc16/`
  - model output:
    `models/hu_turn2_current_fl_ev_stage8c_local300_mc16_smoke.pt`
  - small GPU smoke, early-stopped after `4` epochs
  - train/val/test states: `213 / 44 / 44`
  - `scoring_metadata_status.status = match`
  - `training_allowed = true`

This is still not a model-quality result. It is the first larger current-FL-EV
pipeline check after the 18/60-state entrance smokes. The next useful scale is
not production training; it is a current-FL-EV `1k-5k` mixed-source cache plus
TopK hard-negative replay packs, followed by calibration and realized per-fire
validation.

Current-FL-EV Stage8c mixed-source MC16 probe:

- output dir:
  `outputs/hu_turn2_current_fl_ev_stage8c_mixed_probe_mc16/`
- per-bucket teacher files:
  - `teacher_disagreement.jsonl`: `50` states
    - attempts: `108`
    - accept rate: `46.3%`
    - wall seconds per written: about `1.55s`
    - skipped before full rollout by prefilter: `37`
  - `high_regret.jsonl`: `50` states
    - attempts: `139`
    - accept rate: `36.0%`
    - wall seconds per written: about `2.06s`
    - skipped before full rollout by prefilter: `51`
  - `low_margin.jsonl`: `50` states
    - attempts: `291`
    - accept rate: `17.2%`
    - wall seconds per written: about `3.44s`
    - skipped before full rollout by prefilter: `149`
    - this is the highest-cost bucket in the current probe
  - `random_off_policy.jsonl`: `50` states
    - attempts: `50`
    - accept rate: `100%`
    - wall seconds per written: about `0.54s`
- all bucket files record:
  - `fl_ev_14 = 10.227020614683454`
  - T3 continuation: Stage7_candidate_A `m5_r10` opt-in /
    legacy-continuation setting
  - feature mode: `rust_direct`

Current-FL-EV Stage8c mixed501 MC16 cache/training smoke:

- combined cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_mixed501_mc16_cache/`
  - inputs:
    - local300 natural teacher
    - `teacher_disagreement` 50
    - `high_regret` 50
    - `low_margin` 50
    - `random_off_policy` 50
    - current-FL-EV TopK hard-negative replay teacher 1
  - states: `501`
  - action rows: `11,921`
  - source buckets:
    - natural: `300`
    - teacher_disagreement: `50`
    - high_regret: `50`
    - low_margin: `50`
    - random_off_policy: `50`
    - topk_hard_negative_replay: `1`
  - gate labels:
    - gray: `352`
    - negative: `141`
    - positive: `8`
  - rollout counts:
    - `16`: `500`
    - `512`: `1`
  - scoring objective:
    - `fl_ev_14_status = unique`
    - `fl_ev_14_missing_records = 0`
- training smoke:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_mixed501_mc16/`
  - model output:
    `models/hu_turn2_current_fl_ev_stage8c_mixed501_mc16_smoke.pt`
  - small GPU smoke, `8` epochs
  - train/val/test states: `353 / 74 / 74`
  - `scoring_metadata_status.status = match`
  - `training_allowed = true`
  - `threshold_positive_config_count = 0`

Interpretation:

- The mixed-source current-FL-EV data path is mechanically healthy.
- Positive gate labels are still rare at this small MC16 scale, so this is not a
  gate-quality result.
- For the next `1k-5k` run, low_margin should be budgeted separately because it
  is about `3.4s/written` locally even with prefiltering.
- If the next mixed run targets multiple thousands of low_margin/high_regret
  rows, use Spot VM shards and merge afterward.

Current-FL-EV Stage8c mixed1001 MC16 cache/training smoke:

- additional teacher dir:
  `outputs/hu_turn2_current_fl_ev_stage8c_mixed1000_mc16_extra/`
- added rows beyond mixed501:
  - `natural_200.jsonl`: `200` states
    - accept rate: `100%`
    - wall seconds per written: about `0.82s`
  - `teacher_disagreement_100.jsonl`: `100` states
    - accept rate: `48.1%`
    - wall seconds per written: about `1.49s`
  - `high_regret_100.jsonl`: `100` states
    - accept rate: `35.5%`
    - wall seconds per written: about `3.19s`
  - `low_margin_50.jsonl`: `50` states
    - accept rate: `15.1%`
    - wall seconds per written: about `6.81s`
    - this confirms low_margin is the most expensive current-FL-EV bucket
  - `random_off_policy_50.jsonl`: `50` states
    - accept rate: `100%`
    - wall seconds per written: about `1.15s`
- combined cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_mixed1001_mc16_cache/`
  - states: `1,001`
  - action rows: `23,819`
  - source buckets:
    - natural: `500`
    - teacher_disagreement: `150`
    - high_regret: `150`
    - low_margin: `100`
    - random_off_policy: `100`
    - topk_hard_negative_replay: `1`
  - gate labels:
    - gray: `758`
    - negative: `231`
    - positive: `12`
  - rollout counts:
    - `16`: `1,000`
    - `512`: `1`
  - scoring objective:
    - `fl_ev_14_status = unique`
    - `fl_ev_14_missing_records = 0`
- training smoke:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_mixed1001_mc16/`
  - model output:
    `models/hu_turn2_current_fl_ev_stage8c_mixed1001_mc16_smoke.pt`
  - small GPU smoke, `12` epochs
  - train/val/test states: `703 / 149 / 149`
  - `scoring_metadata_status.status = match`
  - `training_allowed = true`
  - val:
    - avg_regret: `1.8080`
    - top3 recall: `0.5034`
    - gate_accuracy_pos_neg: `0.9444`
  - test:
    - avg_regret: `2.1866`
    - top3 recall: `0.5168`
    - gate_accuracy_pos_neg: `0.9444`
  - `threshold_positive_config_count = 0`

Interpretation:

- The `1k` current-FL-EV mixed-source pipeline is healthy enough to scale.
- This still is not a gate-quality or production result because positive labels
  are only `12 / 1,001`, and all non-hard-negative labels are MC16.
- The next useful generation is a `5k` current-FL-EV mixed-source run with more
  high_regret and low_margin rows, plus additional TopK hard-negative replay
  packs.
- Based on measured local cost, low_margin and high_regret should be sharded on
  Spot VM for the `5k` run. Natural and random_off_policy can remain local if
  needed, but using the same sharded pipeline for all buckets will simplify
  merge/audit.

Current-FL-EV Stage8c mixed5k runner readiness:

- shard runner:
  `scripts/Run-HuTurn2TeacherShard.ps1`
  - default `future_samples = 16`
  - default `stage3_feature_encoder_mode = rust_direct`
  - passes `--use-batched-continuation` by default
  - supports `MaxHands`, `PrefilterFutureSamples`, and `ProgressEvery`
- parallel runner:
  `scripts/Run-HuTurn2TeacherChunksParallel.ps1`
  - default target: `5,000` states
  - default output:
    `outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16/shards`
  - bucket weights:
    - natural: `50%`
    - teacher_disagreement: `15%`
    - high_regret: `15%`
    - low_margin: `10%`
    - random_off_policy: `10%`
  - parent process now fails hard if a shard child job fails, if no shard jobs
    are scheduled, if a teacher output is empty, or if a shard summary is
    missing
- cache builder:
  `scripts/Build-HuTurn2Stage8cCurrentFlEvCache.ps1`
  - excludes `*_final_turn_slow_states_top50.jsonl` from teacher inputs
  - optional extra input remains available for current-FL-EV hard-negative
    replay teachers
- training wrapper:
  `scripts/Run-HuTurn2Stage8cCurrentFlEvTraining.ps1`

Runner smoke completed:

```powershell
.\scripts\Run-HuTurn2TeacherChunksParallel.ps1 `
  -TotalSamples 10 `
  -Chunks 5 `
  -FutureSamples 1 `
  -PrefilterFutureSamples 1 `
  -MaxParallel 2 `
  -MaxHands 80 `
  -OutputDir outputs\script_smoke\chunks_parallel `
  -SeedBase 2026363000 `
  -ProgressEvery 100
```

Smoke result:

- all 5 buckets produced teacher JSONL
- total teacher states: `11`
- summary files written under `outputs/script_smoke/chunks_parallel/summaries/`
- scoring objective remained `fl_ev_14 = 10.227020614683454`
- `stage3_feature_mode = rust_direct`
- cache wrapper smoke succeeded from `6` teacher shard inputs:
  `outputs/script_smoke/chunks_parallel_cache/`
- training wrapper smoke succeeded using the existing mixed1001 cache for
  `1` CPU epoch:
  `outputs/training/script_smoke_stage8c/`

Do not treat this smoke as model quality evidence. It only proves the `5k`
current-FL-EV shard -> cache -> train wrappers are mechanically runnable.

Current-FL-EV Stage8c mixed5k MC16 run:

- teacher generation:
  - output:
    `outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16/shards`
  - states: `5,000`
  - shards/summaries: `100 / 100`
  - missing: `0`
  - scoring objective: `fl_ev_14 = 10.227020614683454`
  - feature encoder: `rust_direct`
  - bucket rows:
    - natural: `2,500`
    - teacher_disagreement: `750`
    - high_regret: `750`
    - low_margin: `500`
    - random_off_policy: `500`
  - measured accept rates:
    - high_regret: `0.308`
    - low_margin: `0.160`
    - teacher_disagreement: `0.494`
- cache:
  - output:
    `outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_cache`
  - inputs: the `100` teacher shards plus `1` current-FL-EV TopK
    hard-negative replay teacher row
  - states/actions: `5,001 / 117,308`
  - feature dim/dtype: `1,076 / float32`
  - `scoring_objective.fl_ev_14_status = unique`
  - `fl_ev_14_missing_records = 0`
  - split states: train/val/test `3,488 / 756 / 757`
  - seat split: first/second `2,491 / 2,510`
  - gate labels: gray/negative/positive `3,609 / 1,342 / 50`
- training:
  - output:
    `outputs/training/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16`
  - model:
    `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
  - device: `cuda`
  - epochs ran / best epoch: `28 / 23`
  - elapsed: `211.4s`
  - scoring metadata status: `match`
  - val EV MAE / avg_regret / top3:
    `4.1363 / 1.8530 / 0.5754`
  - test EV MAE / avg_regret / top3:
    `4.1359 / 2.1484 / 0.5125`
  - test delta baseline MAE: `3.5058`
  - test pairwise ranking accuracy: `0.7156`
  - test gate accuracy pos/neg: `0.9954`
  - threshold sweep:
    `threshold_positive_config_count = 0`

Interpretation:

- The current-FL-EV mixed5k teacher -> cache -> training path is now validated
  end to end.
- This is not a production candidate and not a P2-fixed model. The holdout
  threshold sweep produced no positive override config, and the training labels
  are still mostly MC16.
- The useful conclusion is pipeline readiness and a fresh 10.227-compatible
  baseline artifact. Gate quality still requires a larger current-FL-EV holdout,
  a realized per-fire/seat-swap validation, or a revised objective that models
  whole-game risk.

Current-FL-EV Stage8c TopK per-fire evaluation entrance:

- wrapper:
  `scripts/Run-HuTurn2Stage8cTopkPerFireEval.ps1`
  - default model:
    `models\hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
  - default output:
    `outputs\evals\hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_fire_target`
  - default configs:
    - `k3/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
    - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- default sizing is by fired decisions:
  `target_realized_overrides_per_seed = 50`
- reporting detail:
  - `confirm_delta_mean_on_fired` now prefers the independent Stage B
    `confirm_delta` when present and falls back to legacy `rerank_delta` for
    old logs.
  - primary performance metrics still use realized whole-game paired deltas,
    not confirm/rerank gate deltas.

Small direct smoke:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm32_smoke`
- command used the Stage8c mixed5k model with:
  `k5/mc16/d0/se0/confirm32/cse1/pd0/seat=first/bygate_delta`
- paired seeds: `10`
- override / realized override count: `1 / 1`
- realized per-fire delta: `+12.0`
- confirm delta mean on fired: `+8.0838`
- cancellation audit:
  - non-fired count: `18`
  - non-fired nonzero count: `0`
  - non-fired delta sum: `0.0`

Wrapper smoke:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_wrapper_smoke`
- config:
  `k5/mc8/d0/se0/confirm16/cse1/pd0/seat=first/bygate_delta`
- paired seeds: `5`
- override count: `0`
- cancellation audit:
  - non-fired count: `10`
  - non-fired nonzero count: `0`
  - non-fired delta sum: `0.0`

Interpretation:

- Stage8c current-FL-EV model can be used as a TopK candidate generator in the
  existing two-stage confirm MC evaluator.
- The evaluator emits the required realized per-fire files:
  `conditional_override_metrics.csv` and `cancellation_audit.csv`.
- The smoke is not evidence of model strength. It only proves the Stage8c
  model path, confirm gate, realized per-fire metric, and non-fired
  cancellation audit work together under the current FL EV objective.

Current-FL-EV Stage8c TopK confirm128/cse1 fire-target probe:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_target10_probe`
- configs:
  - `k3/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- seed: `2026063201`
- target: `10` realized fires per config, cap `500` paired seeds
- elapsed: `363.4s`

Probe results:

| config | paired | realized fires | EV/hand | realized per-fire | non-fired nonzero |
|---|---:|---:|---:|---:|---:|
| `k5...` | `366` | `10` | `+0.1017` | `+7.4454` | `0` |
| `k3...` | `476` | `10` | `+0.0260` | `+2.4773` | `0` |

Interpretation:

- Both configs reached the small target and kept non-fired deltas exactly
  canceled.
- `k5` is the better next candidate, but `10` fires is still only a sizing
  probe, not validation.
- Measured fire rate was roughly `1.1%` to `1.4%` of T2 decisions, or about
  `2%` to `3%` of paired seeds with the first-seat-only filter.

Current-FL-EV Stage8c k5 confirm128/cse1 target50 run:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50`
- aggregate output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_aggregate`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- seed: `2026063301`
- target: `50` realized fires, cap `2,500` paired seeds
- reached target after `1,533` paired seeds
- elapsed: `780.4s`

Results:

- realized fires: `50`
- EV/hand: `+0.0413`
- seed-mean CI: `[-0.0049, +0.0876]`
- realized per-fire delta: `+2.5336`
- realized per-fire CI: `[-0.2451, +5.3123]`
- confirm delta mean on fired: `+2.4881`
- cancellation audit:
  - non-fired count: `2,966`
  - non-fired nonzero count: `0`
  - non-fired delta sum: `0.0`
- safety audit:
  - negative-predicted-delta overrides: `0`
  - realized loss count: `6`
  - max realized loss: `24.2270`
  - p95 loss: `9.95`

Interpretation:

- This is a real improvement over the earlier small TopK runs: the fired
  distribution is positive on average, and the known bad pattern of overriding
  model-negative actions did not appear.
- It is still not production evidence. The CI still crosses zero, it is only
  one seed, and the tail loss is material.
- Use the aggregate analyzer for this read:
  `python -m ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire`.
  The analyzer treats confirm delta as diagnostic only and uses realized
  fired-hand deltas plus the cancellation audit for the primary metric.
- The next validation should use multiple independent seeds and keep the same
  realized per-fire primary metric.

Current-FL-EV Stage8c k5 confirm128/cse1 2-seed aggregate:

- aggregate output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_2seed_aggregate`
- inputs:
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063302`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- paired seeds: `3,407`
- realized fires: `100`
- EV/hand: `+0.0441`
- seed mean EV/hand: `+0.0439`
- seed-mean 95% CI: `[+0.0389, +0.0488]`
- realized per-fire delta: `+3.0059`
- realized per-fire CI: `[+0.9464, +5.0654]`
- estimated EV/hand from fire rate x per-fire delta: `+0.0441`
- estimated EV/hand CI: `[+0.0139, +0.0743]`
- confirm delta mean on fired: `+2.2842`
- cancellation audit:
  - non-fired count: `6,614`
  - non-fired nonzero count: `0`
  - non-fired delta sum: `0.0`
- safety audit:
  - negative-predicted-delta overrides: `0`
  - realized loss count: `10`
  - max realized loss: `24.2270`
  - p95 loss: `11.2270`

Interpretation:

- This is the first current-FL-EV Stage8c TopK result with `100` realized
  fires and a positive realized per-fire CI.
- It materially strengthens the `k5/confirm128/cse1/pd0/seat=first/bygate_delta`
  candidate as an offline TopK rerank path.
- It still does not authorize production, P2 fixed status, T1, or 50k teacher
  generation. It is only two seeds, first-seat-only, and still has meaningful
  tail losses.
- The next validation should add more independent seeds or run the same config
  on Spot VM, then repeat tail-loss replay/risk-target extraction on the
  enlarged fired set.

Current-FL-EV Stage8c k5 confirm128/cse1 3-seed aggregate:

- aggregate output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_3seed_aggregate`
- inputs:
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063302`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063303`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- paired seeds: `5,010`
- realized fires: `150`
- EV/hand: `+0.0376`
- seed mean EV/hand: `+0.0372`
- seed-mean 95% CI: `[+0.0237, +0.0506]`
- realized per-fire delta: `+2.5121`
- realized per-fire CI: `[+0.8808, +4.1434]`
- estimated EV/hand from fire rate x per-fire delta: `+0.0376`
- estimated EV/hand CI: `[+0.0132, +0.0620]`
- confirm delta mean on fired: `+2.2112`
- cancellation audit:
  - non-fired count: `9,720`
  - non-fired nonzero count: `0`
  - non-fired delta sum: `0.0`
- safety audit:
  - negative-predicted-delta overrides: `0`
  - realized loss count: `18`
  - max realized loss: `25.2270`
  - p95 loss: `13.2270`

Interpretation:

- The third seed was weaker than the first two but still positive on average:
  seed EV/hand `+0.0238`, per-fire `+1.5245`.
- The 3-seed aggregate keeps positive realized per-fire evidence with exact
  non-fired cancellation.
- This is now stronger validation evidence for the offline TopK rerank path,
  but it is still not a production runtime candidate. The path is expensive,
  first-seat-only, and the tail losses are still meaningful.
- Before any P2/production/T1 decision, run a larger multi-seed validation and
  repeat tail-loss replay/risk-target extraction over all fired/loss rows.

Current-FL-EV Stage8c k5 target50 tail-loss replay:

- hard-negative pack:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_hard_negatives`
  - false-positive realized-loss rows: `6`
  - neutral overconfirm rows: `22`
  - all fired deduped rows: `50`
- MC512 replay:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_hard_negative_replay_mc512`
  - rows replayed: `6 / 6`
  - action mapping: `ok` for all rows
  - hard-negative labels after local T2 replay: `0`
  - replay delta mean: `+1.7801`
- counterfactual risk targets:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_counterfactual_loss_targets`
  - target rows: `6`
  - realized delta sum: `-71.4540`
  - realized delta mean: `-11.9090`
  - recommended use: `whole_game_risk_only` for all `6`

Interpretation:

- The 6 realized losses are not local T2 EV hard negatives under current-FL
  MC512 replay. They are whole-game counterfactual risk examples.
- Do not train them as local EV/safe-LCB negative labels.
- If used for learning, they need a separate whole-game risk/counterfactual
  head or an audit that explicitly consumes realized-loss labels.
- This replay/risk-target extraction only covers the first target50 seed. After
  adding more fired rows, repeat the same tail-loss replay and risk-target
  extraction on the enlarged fired/loss set.
- The next large run should continue multi-seed k5 confirm128/cse1, sized by
  realized fires. Based on the three local target50 runs, expect roughly `13`
  to `21` minutes per `50` fires on this machine for one seed/config; larger
  multi-seed validation is a good Spot VM candidate.

Current-FL-EV Stage8c k5 3-seed tail-loss replay:

- hard-negative pack:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_3seed_hard_negatives`
  - all fired deduped rows: `150`
  - false-positive realized-loss rows: `18`
  - neutral overconfirm rows: `65`
- MC512 replay:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_3seed_hard_negative_replay_mc512`
  - rows replayed: `18 / 18`
  - action mapping: `ok` for all rows
  - hard-negative labels after local T2 replay: `2`
  - replay delta mean: `+1.8788`
- counterfactual risk targets:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_3seed_counterfactual_loss_targets`
  - target rows: `18`
  - realized delta sum: `-205.5891`
  - realized delta mean: `-11.4216`
  - recommended use:
    - local EV hard negative: `2`
    - whole-game risk only: `16`
  - local replay buckets:
    - local negative: `2`
    - local positive/gray: `1`
    - local positive LCB: `15`

Interpretation:

- Most realized whole-game losses still do not become local T2 EV hard
  negatives under MC512 replay.
- The `2` local-negative rows are valid candidates for the next Stage8c
  hard-negative training pass after feature generation.
- The remaining `16` loss rows should not train local EV/safe-LCB targets;
  use them only for a separate whole-game risk/counterfactual head or audit.

Current-FL-EV Stage8c k5 8-seed tail-loss replay:

- source:
  first-seat k5 target50 aggregate with `8` seeds / `400` realized fires.
- hard-negative pack:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_k5_target50_8seed_hard_negatives`
  - all fired deduped rows: `400`
  - false-positive realized-loss rows: `55`
  - neutral overconfirm rows: `156`
- MC512 replay:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_k5_target50_8seed_hard_negative_replay_mc512`
  - rows replayed: `55 / 55`
  - action mapping: `ok` for all rows
  - hard-negative labels after local T2 replay: `2`
  - replay delta mean: `+2.1453`
  - safe_lcb196 labels:
    - positive: `41`
    - gray: `12`
    - negative / hard negative: `2`
- counterfactual risk targets:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_k5_target50_8seed_counterfactual_loss_targets`
  - target rows: `55`
  - realized delta sum: `-692.2215`
  - realized delta mean: `-12.5858`
  - recommended use:
    - local EV hard negative: `2`
    - whole-game risk only: `53`
  - local replay buckets:
    - local negative: `2`
    - local positive/gray: `12`
    - local positive LCB: `41`

Interpretation:

- The enlarged 8-seed audit confirms the earlier pattern: most realized
  whole-game losses are not local T2 EV hard negatives under MC512 replay.
- Only `2 / 55` loss rows should be used as local EV/gate hard negatives after
  feature generation.
- The other `53` rows are valuable risk/counterfactual examples, but should not
  be mixed into local EV/safe-LCB targets until a separate whole-game risk head
  exists.
- This reduces the value of simply adding more local hard negatives. The next
  model improvement should either distill the strong first-seat TopK policy or
  add a separate risk/counterfactual head for the rare downstream losses.

Current-FL-EV Stage8c 8-seed all-fired replay and distillation smoke:

- all-fired MC128 replay:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_k5_target50_8seed_all_fired_replay_mc128`
  - input: 8-seed first-seat k5 target50 aggregate, `400` realized fired rows.
  - rows replayed: `400 / 400`.
  - future samples: `128`.
  - replay delta mean: `+1.9202`.
  - safe_lcb196 labels:
    - positive: `187`, mean local replay delta `+3.5722`
    - gray: `156`, mean local replay delta `+0.8120`
    - negative / local hard negative: `57`, mean local replay delta `-0.4665`
  - teacher JSONL:
    `topk_hard_negative_replay_teacher.jsonl`
- cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_8seed_all_fired_replay_mc128_cache`
  - inputs: existing current-FL-EV mixed5k MC16 shards plus the `400` MC128
    all-fired replay rows.
  - rollout count distribution: `5,000` rows at MC16 plus `400` rows at MC128.
  - scoring objective: `fl_ev_14 = 10.227020614683454`, unique and complete.
  - validation: no feature/target NaN or inf, all legal actions present.
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_8seed_all_fired_replay_mc128.pt`
- training output:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_8seed_all_fired_replay_mc128`
  - local CUDA, RTX 2060 SUPER.
  - epochs/best epoch: `20` / `15`.
  - val avg_regret / top1 / top3: `1.7711` / `0.2953` / `0.5711`.
  - test avg_regret / top1 / top3: `2.1866` / `0.2619` / `0.5459`.
  - threshold positive config count: `0`.
- first-seat TopK smoke, same seed `2026063902`:
  - deployable `pd0` config:
    `outputs/evals/hu_turn2_current_fl_ev_stage8c_distill_allfired_mc128_firstseat_smoke_seed3902_aggregate`
    - paired seeds: `1200`
    - fires: `2`
    - EV/hand: `+0.0050`
    - per-fire delta: `+6.0000`, CI `[-5.7600, +17.7600]`
    - non-fired nonzero: `0`
    - dominant no-override reason: `topk_empty` (`1182`)
  - relaxed `pd=-2` diagnostic:
    `outputs/evals/hu_turn2_current_fl_ev_stage8c_distill_allfired_mc128_firstseat_smoke_seed3902_pdneg2_aggregate`
    - paired seeds: `137`
    - fires: `10`
    - EV/hand: `+0.0738`
    - per-fire delta: `+2.0227`, CI `[-8.6443, +12.6897]`
    - losses / max loss: `3` / `25.2270`
    - negative predicted-delta overrides: `10`
    - non-fired nonzero: `0`
- same-seed base mixed5k comparison:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_smoke_seed3902_aggregate`
  - config: `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
  - paired seeds: `249`
  - fires: `10`
  - EV/hand: `+0.0507`
  - per-fire delta: `+2.5227`, CI `[-4.7195, +9.7649]`
  - losses / max loss: `2` / `16.2270`
  - negative predicted-delta overrides: `0`
  - non-fired nonzero: `0`
- auxiliary-gate variant:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_8seed_all_fired_replay_mc128_auxgate.pt`
  - training output:
    `outputs/training/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_8seed_all_fired_replay_mc128_auxgate`
  - implementation: `topk_hard_negative_replay` rows were treated as auxiliary
    partial teachers:
    - regression weight: `0.0`
    - ranking weight: `0.0`
    - listwise weight: `0.0`
    - gate weight: `1.0`
  - loss metadata records `400` auxiliary states, so this run does not use the
    partial two-action replay rows as ordinary EV/ranking teacher rows.
  - holdout:
    - val avg_regret / top3: `1.7638` / `0.5833`
    - test avg_regret / top3: `2.1854` / `0.5508`
  - first-seat `pd0` smoke:
    `outputs/evals/hu_turn2_current_fl_ev_stage8c_auxgate_allfired_mc128_firstseat_smoke_seed3902_aggregate`
    - paired seeds: `1200`
    - fires: `2`
    - EV/hand: `+0.0050`
    - per-fire delta: `+6.0000`, CI `[-5.7600, +17.7600]`
    - dominant no-override reason: `topk_empty` (`1184`)
    - non-fired nonzero: `0`

Interpretation:

- Replaying all fired rows is useful as an audit/distillation source, but
  naively mixing all-fired replay rows into the normal policy teacher is not a
  win yet.
- The all-fired replay model became too conservative under the deployable
  `pd0` guard: almost all first-seat opportunities were blocked by `topk_empty`.
- Relaxing to `pd=-2` can recover fires, but every override then has negative
  predicted delta, so it is a diagnostic only and should not be treated as a
  deployable gate.
- Treating the replay rows as auxiliary gate-only partial teachers is now
  supported by the trainer and is safer than using them as ordinary EV/ranking
  rows, but this first aux-gate run still did not restore candidate generation.
- The same-seed base mixed5k comparison is stronger at `pd0`; keep base
  mixed5k as the default first-seat TopK candidate generator.
- Next model work should not simply append all-fired replay rows as ordinary
  EV/ranking teacher data. Prefer either a separate risk/counterfactual head or
  a weighted/auxiliary distillation objective that preserves candidate
  generation while learning from replay labels.

Current-FL-EV Stage8c 8-seed post-hoc risk-filter audit:

- analyzer:
  `src/ofc_regular/analyze_hu_turn2_stage8c_risk_filter.py`
- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_k5_target50_8seed_risk_filter_audit`
- inputs: the 8 first-seat base mixed5k TopK decision logs used by the
  `400`-fire aggregate.
- grid rows: `45,361`.
- scoring rule: realized whole-game paired delta only. `confirm_delta` is used
  only as a runtime-available guard input, not as a performance claim.
- unfiltered baseline:
  - decisions: `25,666`
  - fires: `400`
  - EV/hand: `+0.0311288`
  - realized per-fire delta: `+1.9974`, CI `[+1.0058, +2.9890]`
  - losses / p95 loss / p99 loss / max loss:
    `55` / `14.0` / `26.2270` / `39.2270`
- best post-hoc EV guard with at least `50` kept fires:
  `pd0_cd0_cz1_rank10`
  - kept fires: `391`
  - EV/hand: `+0.0320426`
  - EV/hand delta vs unfiltered: `+0.0009138`
  - realized per-fire delta: `+2.1033`, CI low `+1.1039`
  - losses / max loss: `53` / `39.2270`
  - blocked losses / blocked gains: `2` / `2`
- better tail-loss tradeoff examples exist, but they give up a lot of EV:
  - `pd0.5_cd0_cz1.25_rank10`: `198` kept fires, `+0.0220807`
    EV/hand, p95 loss `10` or less, `24` realized losses.
  - `pd0_cd0_cz1.25_semax1.25_rank10`: `290` kept fires, `+0.0217124`
    EV/hand, p95 loss `10` or less, `31` realized losses.

Interpretation:

- A simple runtime-available post-hoc guard does not materially improve the
  already positive 8-seed first-seat base result.
- The best EV guard only adds about `+0.0009 EV/hand` and leaves the same
  `39.2270` max loss.
- Tail-risk guards are possible, but the best p95-loss-constrained variants
  cut EV/hand from about `+0.0311` to about `+0.021` to `+0.022`.
- Treat these guards only as hypotheses for non-overlapping validation. This
  audit does not approve production, P2 fixed status, T1, or 50k teacher
  generation.

Non-overlapping target20 validation of base vs the best post-hoc rank guard:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_vs_rank10_firstseat_nonoverlap_target20`
- aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_vs_rank10_firstseat_nonoverlap_target20_aggregate`
- seeds: `2026064001`, `2026064002`
- configs:
  - base:
    `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
  - rank guard:
    `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/rank10/bygate_delta`
- both configs reached `40 / 40` realized fires.
- non-fired cancellation:
  - base: `non_fired_nonzero_count = 0`
  - rank10: `non_fired_nonzero_count = 0`

| config | paired | fires | EV/hand | per-fire delta | per-fire CI | losses | p95 loss | max loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | `1,339` | `40` | `+0.0556` | `+3.7227` | `[+0.4982, +6.9472]` | `3` | `6.2` | `28.2270` |
| rank10 | `1,234` | `40` | `+0.0199` | `+1.2307` | `[-1.3635, +3.8249]` | `4` | `10.0` | `28.2270` |

Interpretation:

- The post-hoc `rank10` guard did not reproduce its tiny in-sample improvement
  on non-overlapping seeds.
- Base mixed5k remains the preferred first-seat TopK candidate generator.
- Do not add `rank10` to the default runtime/eval config based on the post-hoc
  audit.

Current-FL-EV Stage8c mixed5k plus 3-seed TopK replay training:

- cache:
  `outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_3seed_topk_replay_cache`
  - states/actions: `5,018` / `117,342`
  - rollout count distribution: `5,000` rows at MC16 plus `18` TopK replay
    rows at MC512
  - source bucket counts:
    - high_regret: `750`
    - low_margin: `500`
    - natural: `2,500`
    - random_off_policy: `500`
    - teacher_disagreement: `750`
    - topk_hard_negative_replay: `18`
  - scoring objective: `fl_ev_14 = 10.227020614683454`, unique and complete
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_3seed_topk_replay.pt`
- training output:
  `outputs/training/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_3seed_topk_replay`
- device: local CUDA, RTX 2060 SUPER
- epochs/best epoch: `23` / `18`
- holdout:
  - val EV MAE / avg_regret / top1 / top3:
    `4.1253` / `1.8481` / `0.3030` / `0.5718`
  - test EV MAE / avg_regret / top1 / top3:
    `4.1211` / `2.1881` / `0.2434` / `0.5224`
  - threshold positive config count: `0`

Comparison to the base current-FL-EV mixed5k model:

- base model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- val avg_regret: `1.8530` -> `1.8481` (`-0.0049`)
- test avg_regret: `2.1484` -> `2.1881` (`+0.0397`, worse)
- test top1: `0.2285` -> `0.2434`
- test top3: `0.5125` -> `0.5224`

Interpretation:

- The replay-enriched cache/training path is mechanically valid under the
  current FL EV objective.
- The model-quality signal is mixed: ranking metrics improved a little on test,
  but test avg_regret worsened and the threshold sweep still found zero
  positive teacher-holdout configs.
- Treat this as a training-pipeline pass and a candidate-generator variant, not
  as adoption evidence.

Current-FL-EV Stage8c plus3seed TopK target10 smoke:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_plus3seed_topk_confirm128_cse1_k5_target10_smoke`
- aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_plus3seed_topk_confirm128_cse1_k5_target10_smoke_aggregate`
- config:
  `k5_mc64_d0_se0_confirm128_cse1_pd0_seatfirst_bygate_delta`
- seed: `2026063401`
- paired seeds / hands: `543` / `1,086`
- realized overrides: `10`
- runtime override rate: `0.9208%`
- EV/hand: `+0.0306`
- realized per-fire delta: `+3.3227`
- realized per-fire 95% CI: `[-1.2243, +7.8697]`
- losses / max loss: `2` / `5.0000`
- confirm delta mean on fired: `+2.1041`
- cancellation audit:
  - non-fired count: `1,066`
  - non-fired nonzero count: `0`
  - fired delta sum: `+33.2270`

Interpretation:

- This is an execution smoke and small realized per-fire check only.
- Non-fired cancellation is intact, so realized fired-hand delta is the correct
  primary metric.
- The sample is too small to select the plus3seed model over the base mixed5k
  model. It does not approve production, P2 fixed status, T1, or 50k teacher
  generation.

Current-FL-EV Stage8c base vs plus3seed target50 comparison:

- purpose: compare the base mixed5k candidate generator against the plus3seed
  replay-enriched candidate generator on a non-training seed before deciding
  whether the replay-enriched model is actually better.
- shared config:
  `k5_mc64_d0_se0_confirm128_cse1_pd0_seatfirst_bygate_delta`
- seed: `2026063501`
- T3 continuation: Stage7_candidate_A `m5_r10` opt-in / legacy-continuation
  setting
- base output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target50_seed2026063501`
- base aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target50_seed2026063501_aggregate`
- plus3seed output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_plus3seed_topk_confirm128_cse1_k5_target50_seed2026063501_max5000`
- plus3seed aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_plus3seed_topk_confirm128_cse1_k5_target50_seed2026063501_aggregate`

| model | paired | fires | fire rate | EV/hand | per-fire delta | per-fire CI | losses | max loss | non-fired nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base mixed5k | `2,028` | `50` | `1.2327%` | `+0.0364` | `+2.9536` | `[+0.0063, +5.9009]` | `5` | `34.2270` | `0` |
| plus3seed | `4,704` | `50` | `0.5315%` | `+0.0297` | `+5.5854` | `[+2.1523, +9.0185]` | `6` | `19.2270` | `0` |

Fired-set overlap on the shared seed:

- base fired states: `50`
- plus3seed fired states: `50`
- overlap: `18`
- union: `82`
- Jaccard: `0.2195`
- same-action overlap: `17`

Interpretation:

- Both models pass the realized per-fire/cancellation check on this seed.
- plus3seed fires much less often but has a higher realized per-fire delta and
  lower max loss on this seed.
- base has better EV/hand on this seed because its fire rate is much higher.
- The fired sets are not the same; plus3seed is not merely a strict subset of
  base. It is changing candidate selection materially.
- This is useful diagnostic evidence, not model adoption evidence. A single
  seed cannot decide between base and plus3seed, and neither result approves
  production, P2 fixed status, T1, or 50k teacher generation.

Current-FL-EV Stage8c base vs plus3seed 2-seed target50 comparison:

- seeds: `2026063501`, `2026063502`
- comparison CLI:
  `python -m ofc_regular.analyze_hu_turn2_stage8c_candidate_generators`
- comparison output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_vs_plus3seed_k5_target50_seed3501_3502_comparison`
  - `candidate_generator_comparison.md`
  - `candidate_generator_metrics.csv`
  - `candidate_generator_overlap.csv`
- base aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target50_seed3501_3502_aggregate`
- plus3seed aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_plus3seed_topk_confirm128_cse1_k5_target50_seed3501_3502_aggregate`

| model | paired | fires | EV/hand | per-fire delta | per-fire CI | losses | max loss | non-fired nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base mixed5k | `3,515` | `100` | `+0.0379` | `+2.6614` | `[+0.6140, +4.7087]` | `12` | `34.2270` | `0` |
| plus3seed | `8,083` | `100` | `+0.0227` | `+3.6727` | `[+1.3134, +6.0320]` | `16` | `37.2270` | `0` |

Seed breakdown:

| model | seed | paired | fires | EV/hand | per-fire delta |
|---|---:|---:|---:|---:|---:|
| base mixed5k | `2026063501` | `2,028` | `50` | `+0.0364` | `+2.9536` |
| base mixed5k | `2026063502` | `1,487` | `50` | `+0.0398` | `+2.3691` |
| plus3seed | `2026063501` | `4,704` | `50` | `+0.0297` | `+5.5854` |
| plus3seed | `2026063502` | `3,379` | `50` | `+0.0130` | `+1.7600` |

Fired-set overlap across the two seeds:

- base fired states: `100`
- plus3seed fired states: `100`
- overlap: `35`
- union: `165`
- Jaccard: `0.2121`
- same-action overlap: `33`

Interpretation:

- Both candidate generators remain positive on realized per-fire metrics with
  exact non-fired cancellation.
- base mixed5k has the better 2-seed EV/hand because it reaches 100 fires with
  far fewer paired seeds.
- plus3seed has higher aggregate per-fire quality, but the lower fire rate
  costs too much EV/hand on these two seeds.
- The low fired-set overlap shows the models choose materially different T2
  spots; plus3seed should remain a precision-oriented diagnostic candidate, but
  the base mixed5k model is the better default candidate generator for the next
  multi-seed TopK rerank validation.
- This still does not approve T2 production, P2 fixed status, T1, or 50k
  teacher generation.

Current-FL-EV Stage8c base mixed5k 5-seed target50 aggregate:

- purpose: consolidate all available base mixed5k target50 TopK rerank seeds
  before launching more compute.
- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target50_5seed_aggregate`
- inputs:
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063302`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063303`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target50_seed2026063501`
  - `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target50_seed2026063502`
- config:
  `k5_mc64_d0_se0_confirm128_cse1_pd0_seatfirst_bygate_delta`
- paired seeds: `8,525`
- hands: `17,050`
- realized fires: `250`
- runtime fire rate: `1.4663%`
- EV/hand: `+0.0377`
- seed mean EV/hand: `+0.0375`
- seed mean EV/hand 95% CI: `[+0.0301, +0.0450]`
- realized per-fire delta: `+2.5718`
- realized per-fire 95% CI: `[+1.2982, +3.8454]`
- confirm delta mean on fired: `+2.3337`
- losses / p95 loss / max loss: `30` / `14.0000` / `34.2270`
- cancellation audit:
  - valid realized delta count: `16,800`
  - non-fired count: `16,550`
  - non-fired nonzero count: `0`
  - fired delta sum: `+642.9513`
- no negative-predicted-delta overrides: `0`

Seed breakdown:

| seed | paired | fires | EV/hand | per-fire delta |
|---:|---:|---:|---:|---:|
| `2026063301` | `1,533` | `50` | `+0.0413` | `+2.5336` |
| `2026063302` | `1,874` | `50` | `+0.0464` | `+3.4782` |
| `2026063303` | `1,603` | `50` | `+0.0238` | `+1.5245` |
| `2026063501` | `2,028` | `50` | `+0.0364` | `+2.9536` |
| `2026063502` | `1,487` | `50` | `+0.0398` | `+2.3691` |

Interpretation:

- This is the strongest current evidence for T2 so far: five non-overlapping
  seeds, `250` realized fires, exact non-fired cancellation, positive seed mean
  CI, and positive realized per-fire CI.
- It validates the base mixed5k model as the preferred offline TopK candidate
  generator under the current FL EV objective.
- The per-fire aggregate analyzer now reports hidden-discard replay packet
  completeness. Older target50 runtime logs may still be valid for realized
  seat-swap/cancellation metrics while missing replay-grade
  `visible_dead_cards` and private discard metadata.
- It still does not justify production/P2 because the path is expensive,
  first-seat-only in this validation, and uses online MC rerank rather than a
  cheap runtime policy.
- The next T2 step should be a C4-style validation plan for base mixed5k:
  add second-seat coverage or explicitly keep first-seat-only, and decide
  whether to spend compute on more seeds locally or on Spot VM.

Current-FL-EV Stage8c base mixed5k second-seat target20 probe:

- purpose: check whether the strong first-seat TopK rerank result transfers to
  second-seat before mixing seats in a larger validation.
- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target20_second_seed2026063601`
- aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_topk_confirm128_cse1_k5_target20_second_seed2026063601_aggregate`
- config:
  `k5_mc64_d0_se0_confirm128_cse1_pd0_seatsecond_bygate_delta`
- seed: `2026063601`
- paired seeds: `638`
- hands: `1,276`
- realized fires: `20`
- runtime fire rate: `1.5674%`
- EV/hand: `-0.0141`
- realized per-fire delta: `-0.9000`
- realized per-fire 95% CI: `[-6.6072, +4.8072]`
- confirm delta mean on fired: `+3.1077`
- losses / max loss: `4` / `33.2270`
- cancellation audit:
  - valid realized delta count: `1,256`
  - non-fired count: `1,236`
  - non-fired nonzero count: `0`
  - fired delta sum: `-18.0000`

Interpretation:

- This small probe does not prove second-seat is bad, but it clearly does not
  reproduce the first-seat signal.
- It is also a useful bias check: confirm MC looked positive on fired decisions,
  while realized per-fire was negative.
- Do not mix second-seat into the current first-seat C4 evidence without a
  separate second-seat gate/model or substantially more validation.
- Current practical split:
  - first-seat base mixed5k TopK rerank: continue validation.
  - second-seat base mixed5k TopK rerank: No-Go pending redesign or more
    targeted data.

First-seat C4 runner guard:

- script:
  `scripts/Run-HuTurn2Stage8cFirstSeatC4.ps1`
- purpose:
  launch the next base mixed5k first-seat C4 TopK rerank validation without
  accidentally mixing second-seat configs into the positive first-seat evidence.
- default config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- default model:
  `models\hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- default seeds:
  `2026063701,2026063702,2026063703`
- default target:
  `50` realized fires per seed, `3,500` paired seeds hard cap per seed
- default aggregate output:
  `${OutputDir}_aggregate`
- guard:
  the script rejects configs that do not contain `seat=first`, and also rejects
  `seat=second`, `seat=both`, `seat=all`, or wildcard seat scopes.
- compute guard:
  by default, the script rejects a live CUDA-looking Python job matching the
  OFC project command lines before launching C4. This avoids contaminating C4
  timing or starving another GPU run. Use `-AllowConcurrentPython` only if the
  overlap is intentional.
- CPU smoke exception:
  `-Device cpu` bypasses the CUDA job guard. This is only for wiring/smoke
  checks; quality validation should still wait for the GPU path unless the run
  is intentionally CPU-only.
- wait option:
  `-WaitForGpuClear` polls until detected CUDA Python jobs disappear before
  launching C4. `-GpuWaitTimeoutSeconds 0` means wait indefinitely; set a
  positive timeout for a bounded wait. `-GpuPollSeconds` controls the polling
  interval.
- runner invocation:
  the wrapper forwards parameters to
  `scripts\Run-HuTurn2Stage8cTopkPerFireEval.ps1` with a hashtable splat
  (`$runnerParams`) instead of positional string arguments. A previous smoke
  exposed that positional forwarding can bind `-Seeds` incorrectly.
- aggregation:
  after the evaluation succeeds, the script runs
  `python -m ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire` on the output
  directory. This preserves the realized per-fire/cancellation-audit workflow.
- `-NoDecisionLog` requires `-NoAggregate`, because aggregate evaluation needs
  `runtime_decisions.jsonl`.
- dry run:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 -DryRun
```

Dry run includes:

- `aggregate_enabled`
- `aggregate_output_dir`
- `concurrent_gpu_python_process_count`
- `concurrent_gpu_python_processes`

- rejection smoke:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 `
  -DryRun `
  -Configs 'k5/mc64/d0/se0/confirm128/cse1/pd0/seat=second/bygate_delta'
```

Expected result: throw before launching evaluation.

Concurrent GPU job smoke:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 `
  -GamesPerSeed 1 `
  -TargetRealizedOverridesPerSeed 1 `
  -Seeds 2026063799 `
  -OutputDir outputs\evals\guard_should_not_run
```

Expected result when another CUDA Python job is running: throw before launching
evaluation, with the detected PID and command line. Current local observation:
the guard detected a live T3 target-generation CUDA job and correctly blocked
C4 launch.

Bounded wait smoke:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 `
  -WaitForGpuClear `
  -GpuWaitTimeoutSeconds 1 `
  -GpuPollSeconds 1 `
  -GamesPerSeed 1 `
  -TargetRealizedOverridesPerSeed 1 `
  -Seeds 2026063799 `
  -OutputDir outputs\evals\guard_wait_should_not_run
```

Expected result while the GPU job is still running: print a wait line, then
throw before launching evaluation.

CPU wiring smoke:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 `
  -Device cpu `
  -GamesPerSeed 2 `
  -TargetRealizedOverridesPerSeed 1 `
  -Seeds 2026063798 `
  -OutputDir outputs\evals\hu_turn2_current_fl_ev_stage8c_firstseat_c4_cpu_smoke `
  -ProgressEvery 1
```

Observed result:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_firstseat_c4_cpu_smoke`
- aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_firstseat_c4_cpu_smoke_aggregate`
- paired seeds: `2`
- hands: `4`
- realized fires: `0`
- EV/hand: `0.0000`
- cancellation audit: non-fired nonzero `0`

This is a wrapper/aggregate smoke only. With zero realized fires it says
nothing about T2 model strength.

No-decision-log dry run must explicitly disable aggregate:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 -DryRun -NoDecisionLog -NoAggregate
```

Next first-seat C4 command:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 `
  -Seeds "2026063701,2026063702,2026063703" `
  -OutputDir outputs\evals\hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_c4_seed3701_3703
```

Safe wait-and-run variant:

```powershell
.\scripts\Run-HuTurn2Stage8cFirstSeatC4.ps1 `
  -WaitForGpuClear `
  -GpuPollSeconds 60 `
  -Seeds "2026063701,2026063702,2026063703" `
  -OutputDir outputs\evals\hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_c4_seed3701_3703
```

Spot VM variant:

- start script:
  `scripts\Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1`
- status script:
  `scripts\Get-GcpHuTurn2Stage8cTopkPerFireRunStatus.ps1`
- receive script:
  `scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1`
- phase:
  `hu_t2_stage8c_topk_per_fire`
- default:
  same first-seat C4 config/model/seeds as the local runner.
- safety:
  the GCP start script also rejects non-first-seat configs and builds/checks the
  Rust `rust_direct` feature encoder before running shards.
- aggregation:
  receive runs `python -m ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire`
  across all downloaded shard result directories.

Start on Spot VMs:

```powershell
.\scripts\Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1 `
  -RunName regular-hu-t2-stage8c-topk-c4-YYYYMMDD-HHMMSS `
  -CreateInstances
```

Check status:

```powershell
.\scripts\Get-GcpHuTurn2Stage8cTopkPerFireRunStatus.ps1 `
  -RunName regular-hu-t2-stage8c-topk-c4-YYYYMMDD-HHMMSS
```

Receive and aggregate:

```powershell
.\scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 `
  -RunName regular-hu-t2-stage8c-topk-c4-YYYYMMDD-HHMMSS `
  -OutputDir outputs\evals\hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_c4_seed3701_3703_gcp_aggregate
```

Completed Spot C4 run:

- run name:
  `regular-hu-t2-stage8c-topk-c4-20260614-093129`
- started:
  `2026-06-14T00:31Z` UTC
- shards:
  `3`
- first launch:
  shard `0`, `1`, `2`
- note:
  the first shard `0` instance disappeared before producing status/results, so
  shard `0` was relaunched with `-StartShards 0 -SkipExistingInstances`.
- final status:
  `3/3` shards complete, no failed shards, no running instances.
- shard elapsed seconds:
  `1209`, `1389`, `1008`.
- aggregate:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_c4_seed3701_3703_gcp_aggregate`
- C4-only result:
  - paired seeds: `4,308`
  - realized fires: `150`
  - EV/hand: `+0.0181`
  - seed-mean EV/hand CI: `[-0.0106, +0.0447]`
  - realized per-fire delta: `+1.0400`
  - realized per-fire CI: `[-0.5303, +2.6103]`
  - realized losses: `25`
  - p95 loss: `17.2270`
  - max loss: `39.2270`
  - non-fired nonzero: `0`
- interpretation:
  execution/cancellation passed. C4-only is positive on the mean but does not
  independently clear zero because the CI still crosses zero.

Eight-seed first-seat aggregate:

- output:
  `outputs/evals/hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_k5_target50_8seed_aggregate`
- inputs:
  the prior five first-seat target50 seeds plus the three C4 GCP shards above.
- paired seeds: `12,833`
- realized fires: `400`
- override rate: `1.5585%`
- EV/hand: `+0.0311`
- seed-mean EV/hand: `+0.0299`
- seed-mean EV/hand CI: `[+0.0174, +0.0423]`
- realized per-fire delta: `+1.9974`
- realized per-fire CI: `[+1.0058, +2.9890]`
- confirm delta mean on fired: `+2.4113`
- realized losses: `55`
- p95 loss: `14.0000`
- max loss: `39.2270`
- non-fired nonzero: `0`
- replay-ready fires after hidden-discard packet audit: `0 / 400`
- replay missing fields on fired logs:
  `visible_dead_cards`, `hero_private_discards`, `opponent_private_discards`
- interpretation:
  first-seat base mixed5k TopK+confirm rerank now has strong offline evidence
  across eight seeds, with exact non-fired cancellation. This supports continued
  first-seat-only validation/distillation. It does not authorize production,
  P2 fixed status, T1, or 50k teacher generation by itself.
  The current eight-seed logs predate the replay-packet logging hardening, so
  do not use them directly for exact high-MC replay or risk-target extraction
  that requires hidden-discard state reconstruction. New runs should show
  replay-ready fired rows.

Replay-packet logging smoke after hardening:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_smoke/`
- aggregate:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_smoke_aggregate/`
- config:
  `k5/mc1/d0/se0/confirm0/cse0/pd0/seat=first/bygate_delta`
- purpose:
  wiring smoke only; not a model-quality result.
- paired seeds: `6`
- realized fires: `1`
- `decision_replay_ready_count`: `12 / 12`
- `fired_replay_ready_count`: `1 / 1`
- missing replay field counts: `{}`
- runtime log fields confirmed:
  `dead_cards` as true unavailable discards, `visible_dead_cards` as
  actor-visible cards, plus `hero_private_discards` and
  `opponent_private_discards`.

Interpretation:

- New Stage8c TopK runs after the replay-packet hardening can produce exact
  replay-ready fired rows.
- The old eight-seed aggregate remains valid for realized per-fire/cancellation
  evidence, but its logs are not suitable for exact high-MC replay. Any next
  high-MC replay/risk extraction should be based on newly generated logs or a
  newly rerun validation with replay-ready metadata.

Real-config replay-ready smoke:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target5/`
- aggregate:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target5_aggregate/`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- T3 continuation:
  `stage3_reference_default`
- purpose:
  wiring/replayability smoke under the real first-seat TopK config; not a
  quality gate.
- paired seeds: `237`
- realized fires: `5`
- EV/hand: `+0.0853`
- realized per-fire delta: `+8.0908`
- realized per-fire CI: `[-3.1083, +19.2899]`
- `decision_replay_ready_count`: `474 / 474`
- `fired_replay_ready_count`: `5 / 5`
- missing replay field counts: `{}`

Replay pack from this log:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target5_replay_pack/`
- `all_fired_deduped`: `5 / 5` replay-ready
- `false_positive`: `0`
- `neutral_overconfirm`: `3 / 3` replay-ready

MC128 local T2 replay of all fired rows:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target5_replay_mc128/`
- rows replayed: `5 / 5`
- action mapping status: `ok` for all rows
- hard-negative labels after replay: `0`
- replay delta mean: `+3.7959`

Counterfactual loss target extraction from the same replay-ready log:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target5_counterfactual_loss_targets/`
- target rows: `0`
- mismatch rows: `0`

Interpretation:

- The fixed runtime logging is now sufficient for the complete path:
  runtime decision log -> replay pack -> local T2 MC replay -> counterfactual
  risk target extraction.
- This removes the replay-metadata blocker for future first-seat Stage8c TopK
  validation runs. It still does not approve production/P2/T1/50k.

This C4 runner is still validation-only. A positive result would justify a
larger first-seat validation or distillation work; it still would not authorize
production/P2 fixed status, T1, or 50k teacher generation by itself.

Targeted tests:

```powershell
python -m pytest -p no:cacheprovider `
  tests\test_stage8c_scripts.py `
  tests\test_hu_turn2_stage8c_candidate_generator_comparison.py `
  tests\test_hu_turn2_stage8c_topk_per_fire_analysis.py `
  tests\test_hu_turn2_stage8b_training_prep.py `
  tests\test_hu_turn2_stage8b_counterfactual_loss_targets.py `
  tests\test_hu_turn2_stage8b_topk_hard_negatives.py `
  tests\test_hu_turn2_stage8b_topk_hard_negative_replay.py
```

TopK per-fire aggregate:

```powershell
python -m ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire `
  --input-dir outputs\evals\hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50 `
  --input-dir outputs\evals\hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063302 `
  --input-dir outputs\evals\hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_seed2026063303 `
  --output-dir outputs\evals\hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_k5_target50_3seed_aggregate
```

Full test command:

```powershell
python -m pytest -p no:cacheprovider
```

Direct HU FL EV smoke:

```powershell
python -m ofc_regular.estimate_hu_fl_ev_direct `
  --trials 5 `
  --iterations 1 `
  --seed 2026062201 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --output-dir outputs\fl_ev_direct_hu_smoke
```

Direct HU FL EV production-style run should use much larger trials, preferably
on Spot VM:

```powershell
python -m ofc_regular.estimate_hu_fl_ev_direct `
  --trials 10000 `
  --iterations 5 `
  --seed 2026062201 `
  --opening-lookahead-samples 32 `
  --prediction-threads 1 `
  --output-dir outputs\fl_ev_direct_hu_10k
```

TopK hard-negative pack extraction:

```powershell
python -m ofc_regular.prepare_hu_turn2_stage8b_topk_hard_negatives `
  --decision-log outputs\evals\hu_turn2_stage8b_topk_mc_twostage_confirm128_small\runtime_decisions.jsonl `
  --decision-log outputs\evals\hu_turn2_stage8b_topk_mc_twostage_confirm256_small\runtime_decisions.jsonl `
  --output-dir outputs\evals\hu_turn2_stage8b_topk_hard_negatives
```

Replay extracted hard negatives under the current FL EV objective before using
them for training:

```powershell
python -m ofc_regular.replay_hu_turn2_stage8b_topk_hard_negatives `
  --input-jsonl outputs\evals\hu_turn2_stage8b_topk_hard_negatives\topk_false_positive_hard_negatives.jsonl `
  --future-samples 512 `
  --seed 2026062301 `
  --prediction-threads 1 `
  --opening-lookahead-samples 32 `
  --output-dir outputs\evals\hu_turn2_stage8b_topk_hard_negative_replay_mc512
```

Replay output:

- `topk_hard_negative_replay_teacher.jsonl`
- `topk_hard_negative_replay_summary.csv`
- `topk_hard_negative_replay_summary.md`
- `topk_hard_negative_replay_manifest.json`

Important interpretation rules:

- The replay CLI resolves actions by canonical placement/discard signature, not
  by trusting old local action indices.
- `delta_for_label` is the replay diagnostic. If paired candidate-vs-baseline
  stats are available, it uses the paired delta.
- Rows with `action_mapping_status != ok` must be inspected before training.
- High-MC replay labels are diagnostic/training inputs only. They still do not
  approve T2 production, P2 fixed status, T1, or 50k teacher generation.

Latest local replay result:

- `outputs/evals/hu_turn2_stage8b_topk_hard_negative_replay_mc2048/`
- 3/3 rows replayed successfully.
- Action mapping status was `ok` for all rows.
- MC2048 paired T2 deltas were all positive:
  - `+0.8496 ± 0.1892`
  - `+4.6758 ± 0.1921`
  - `+1.6634 ± 0.2758`
- `hard_negative_label` after replay: 0/3.

Interpretation: these old top-loss rows should not be injected as direct
negative labels from local T2 EV. They remain useful as evidence that
whole-game realized losses can disagree with local T2 MC replay, so the next
training iteration needs either replayed feature generation with explicit
gray/positive labels or a separate whole-game/counterfactual target. Do not
call them hard negatives unless an independent higher-MC or realized
counterfactual label is actually negative.

Counterfactual loss target audit:

- `outputs/evals/hu_turn2_stage8b_counterfactual_loss_targets/`
- Inputs:
  - `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm128_small/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage8b_topk_mc_twostage_confirm256_small/runtime_decisions.jsonl`
  - MC2048 local replay summary above
- realized whole-game loss rows: 6
- realized loss sum: `-37.0`
- realized loss mean: `-6.1667`
- local replay bucket: `local_positive_lcb` for all 6 rows
- recommended use: `whole_game_risk_only` for all 6 rows

Interpretation: the observed TopK losses are not local T2 EV hard negatives
under the current replay objective. They are examples where local T2 MC says the
candidate is good, but whole-game seat-swap counterfactual performance loses.
The next Stage8 iteration should not simply add these rows as negative labels
to the local EV/safe LCB head. A separate whole-game risk/counterfactual head,
or a training target that explicitly models downstream realized-loss risk, is
needed if these examples are used.

Pre-large prep now carries these rows forward separately:

- `outputs/hu_turn2_stage8b_prelarge_training_with_risk_targets/`
- `stage8b_safe_override_labels.csv`: existing local safe-LCB gate labels.
- `stage8b_whole_game_risk_targets.jsonl`: 6 risk-only loss targets.
- `stage8b_training_ready.json`:
  - `safe_lcb_training_ready = false` until the feature cache records the
    current scoring objective
  - `risk_head_training_ready = false`
  - `whole_game_risk_targets_require_separate_head = true`
  - `topk_local_ev_hard_negatives_from_counterfactual_audit = 0`
  - `scoring_metadata_status.status = missing` for the existing old 20k cache

Do not use `stage8b_whole_game_risk_targets.jsonl` as negative local EV labels.
They are only valid for a future risk/counterfactual head or an audit that
explicitly consumes whole-game realized-loss labels.
The Stage8b training label loader rejects rows marked as whole-game risk-only,
so accidentally passing those rows as `--stage8b-labels-csv` should fail before
training starts.
The Stage8b model training CLI also rejects feature caches whose
`scoring_objective.fl_ev_14` is missing or differs from the current
`10.227020614683454` value unless an explicit analysis-only override flag is
provided.

Current-FL-EV training entrance smoke:

- `outputs/hu_turn2_current_fl_ev_train_smoke18/`
- teacher generation:
  - 18 states
  - MC4 smoke only
  - `scoring_objective.fl_ev_14 = 10.227020614683454`
- feature cache:
  - `outputs/hu_turn2_current_fl_ev_train_smoke18/feature_cache/`
  - train/val/test states: `10 / 4 / 4`
  - `scoring_objective.fl_ev_14_status = unique`
- training smoke:
  - `outputs/training/hu_turn2_current_fl_ev_train_smoke18/`
  - one epoch, tiny CPU model, execution-only smoke
  - `scoring_metadata_status.status = match`
  - `training_allowed = true`

This smoke is not a model-quality result. It only proves the current 10.227
teacher -> feature cache -> training entrance path is wired correctly, while
old 12.196 or missing-scoring caches are blocked by default.

Replay-ready real-config target20 check:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20/`
- aggregate:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20_aggregate/`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- T3 continuation:
  `stage3_reference_default`
- paired seeds: `681`
- decisions: `1362`
- realized overrides: `20`
- override rate: `1.4684%`
- EV/hand: `+0.0215`
- estimated EV/hand CI from realized fired deltas: `[-0.0394, +0.0823]`
- realized per-fire delta: `+1.4614`
- realized per-fire CI: `[-2.6815, +5.6042]`
- confirm delta mean on fired: `+2.2188`
- realized loss count: `4`
- p95 loss: `6.0000`
- max loss: `18.2270`
- replay-ready decisions: `1362 / 1362`
- replay-ready fired rows: `20 / 20`
- replay missing fields: `{}`

Replay pack from this log:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20_replay_pack/`
- `all_fired_deduped`: `20 / 20` replay-ready
- `false_positive`: `4 / 4` replay-ready
- `neutral_overconfirm`: `7 / 7` replay-ready
- false-positive realized mean: `-7.0568`
- false-positive confirm mean: `+2.1633`

MC replay from the same replay-ready rows:

- all-fired MC128 output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20_replay_mc128/`
- all-fired MC128 rows: `20 / 20`
- action mapping status: `ok` for all rows
- false-positive-only MC512 output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20_false_positive_replay_mc512/`
- false-positive-only MC512 rows: `4 / 4`
- action mapping status: `ok` for all rows
- false-positive-only MC512 labels:
  all `4 / 4` are `safe_lcb196_label = positive`
- false-positive-only MC512 hard negatives: `0`

Counterfactual loss target extraction:

- MC128 output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20_counterfactual_loss_targets/`
- MC128 target rows: `4`
- MC128 recommended use:
  `2` local EV hard negatives, `2` whole-game risk only
- MC512 output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target20_counterfactual_loss_targets_mc512/`
- MC512 target rows: `4`
- MC512 recommended use:
  `4` whole-game risk only, `0` local EV hard negatives

Interpretation:

- The replay packet path now works on a nontrivial real-config target20 run:
  runtime decisions -> replay-ready fired rows -> replay pack -> local T2 MC
  replay -> counterfactual loss target extraction.
- The four realized whole-game loss rows are not local T2 EV hard negatives
  under MC512. They should be kept as whole-game risk/counterfactual examples,
  not as negative labels for the local EV or safe-LCB head.
- The result is still too small for model-quality adoption. It strengthens the
  pipeline readiness evidence, not the production case.

Replay-ready real-config target50 check:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target50_seed2026064007/`
- aggregate:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target50_seed2026064007_aggregate/`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- T3 continuation:
  `stage3_reference_default`
- paired seeds: `1846`
- decisions: `3692`
- realized overrides: `50`
- override rate: `1.3543%`
- EV/hand: `+0.0070`
- estimated EV/hand CI from realized fired deltas: `[-0.0305, +0.0444]`
- realized per-fire delta: `+0.5155`
- realized per-fire CI: `[-2.2505, +3.2814]`
- confirm delta mean on fired: `+2.4128`
- realized loss count: `10`
- p95 loss: `23.2270`
- max loss: `31.2270`
- non-fired nonzero count: `0`
- replay-ready decisions: `3692 / 3692`
- replay-ready fired rows: `50 / 50`
- replay missing fields: `{}`

Replay and loss-target extraction from this log:

- replay pack:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target50_seed2026064007_replay_pack/`
- `all_fired_deduped`: `50 / 50` replay-ready
- `false_positive`: `10 / 10` replay-ready
- `neutral_overconfirm`: `19 / 19` replay-ready
- all-fired MC128 replay:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target50_seed2026064007_replay_mc128/`
- all-fired MC128 labels:
  `10` negative, `21` gray, `19` positive
- false-positive-only MC512 replay:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target50_seed2026064007_false_positive_replay_mc512/`
- false-positive-only MC512 labels:
  `1` negative, `4` gray, `5` positive
- MC512 counterfactual loss targets:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_target50_seed2026064007_counterfactual_loss_targets_mc512/`
- MC512 target rows: `10`
- MC512 recommended use:
  `1` local EV hard negative, `9` whole-game risk only

Interpretation:

- This is the first replay-ready target50 run with exact non-fired
  cancellation under the current hidden-discard packet format.
- The whole-game realized metric is weakly positive but statistically
  unresolved. It is not adoption evidence.
- The gap between confirm mean (`+2.4128`) and realized per-fire (`+0.5155`)
  remains large, so confirm-gated deltas must stay diagnostic only.
- Most realized losses are not local EV hard negatives after MC512 replay.
  Future learning should separate the one local hard negative from the nine
  whole-game risk-only rows.

Combined replay-ready loss target collection:

- output:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_realconfig_loss_target_collection/`
- inputs:
  - target20 MC512 loss targets
  - target50 MC512 loss targets
- input rows: `14`
- deduped rows: `14`
- realized delta sum: `-149.1351`
- realized delta mean: `-10.6525`
- local EV hard negatives: `1`
- whole-game risk only: `13`
- requires local replay: `0`
- local replay buckets:
  - `local_negative`: `1`
  - `local_positive_gray`: `4`
  - `local_positive_lcb`: `9`

Files:

- `topk_counterfactual_loss_targets_merged.jsonl`
- `topk_local_ev_hard_negatives.jsonl`
- `topk_whole_game_risk_only.jsonl`
- `topk_requires_local_replay.jsonl`
- `topk_loss_target_collection_summary.md`

Interpretation:

- The current replay-ready loss evidence says the local EV/safe-LCB head has
  only one confirmed local hard negative from these runs.
- The much larger signal is downstream whole-game realized risk. Do not fold
  those `13` risk-only rows into local EV/gate negatives.
- The new collector preserves this split so future Spot VM runs can be merged
  without manually mixing incompatible labels.

Replay-ready postprocess wrapper:

- script:
  `scripts/Run-HuTurn2Stage8cReplayReadyPostprocess.ps1`
- purpose:
  run the complete postprocess chain for local or GCP-downloaded Stage8c TopK
  result directories:
  aggregate -> replay pack -> local replay -> counterfactual loss targets ->
  loss target collection.
- default T3 continuation:
  `stage3_reference_default`
- safety status emitted by the wrapper:
  production/P2 fixed `No-Go`, T1 `No-Go`, 50k teacher `No-Go`
- useful switches:
  - `-SkipAllFiredReplay`
  - `-SkipFalsePositiveReplay`
  - `-SkipLossTargetCollection`
  - `-DryRun`

Wrapper smoke:

```powershell
.\scripts\Run-HuTurn2Stage8cReplayReadyPostprocess.ps1 `
  -InputDirs `
    outputs\hidden_discard_smoke\stage8c_topk_replay_packet_realconfig_target20,`
    outputs\hidden_discard_smoke\stage8c_topk_replay_packet_realconfig_target50_seed2026064007 `
  -OutputRoot outputs\hidden_discard_smoke\stage8c_topk_replay_packet_postprocess_wrapper_smoke `
  -SkipAllFiredReplay `
  -FalsePositiveFutureSamples 512 `
  -ReplaySeed 2026064110 `
  -PredictionThreads 1 `
  -OpeningLookaheadSamples 32 `
  -T3Continuation stage3_reference_default
```

Smoke output:

- aggregate:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_postprocess_wrapper_smoke/aggregate/`
- replay pack:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_postprocess_wrapper_smoke/replay_pack/`
- false-positive MC512 replay:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_postprocess_wrapper_smoke/false_positive_replay_mc512/`
- loss target collection:
  `outputs/hidden_discard_smoke/stage8c_topk_replay_packet_postprocess_wrapper_smoke/loss_target_collection/`
- combined paired seeds: `2527`
- combined decisions: `5054`
- combined realized fires: `70`
- replay-ready decisions: `5054 / 5054`
- replay-ready fired rows: `70 / 70`
- realized per-fire: `+0.7857`
- realized per-fire CI: `[-1.5037, +3.0751]`
- confirm mean on fired: `+2.3574`
- realized losses: `14`
- loss target collection from new MC512 replay seed:
  - local EV hard negatives: `2`
  - whole-game risk only: `12`

Interpretation:

- The wrapper reproduces the full postprocess chain from multiple input dirs.
- The local hard-negative count moved from `1` in the prior manual MC512
  replay seed to `2` in the wrapper MC512 replay seed. Treat borderline
  local-negative labels as MC-sensitive until confirmed with larger MC or
  repeated replay seeds.
- The main stable conclusion remains unchanged: most realized losses are
  whole-game risk-only, not local EV hard negatives.

GCP replay-ready target50 run:

- run name:
  `regular-hu-t2-stage8c-rr-20260614-1426`
- phase:
  `hu_t2_stage8c_topk_per_fire`
- VM setup:
  `3` Spot VMs, `e2-highcpu-8`, zones `asia-northeast1-a/b/c`
- seeds:
  `2026064201`, `2026064202`, `2026064203`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- games per seed:
  `3500`
- target realized overrides per seed:
  `50`
- T3 continuation:
  `stage3_reference_default`
- status:
  `3 / 3` shards complete, `0` failed, no running instances after completion
- shard elapsed seconds:
  - seed `2026064201`: `1097`
  - seed `2026064202`: `1238`
  - seed `2026064203`: `1196`
- received output:
  `outputs/gcp_runs/regular-hu-t2-stage8c-rr-20260614-1426/`
- aggregate output:
  `outputs/evals/regular-hu-t2-stage8c-rr-20260614-1426_aggregate/`
- postprocess output:
  `outputs/evals/regular-hu-t2-stage8c-rr-20260614-1426_postprocess/`

GCP aggregate:

- paired seeds:
  `5300`
- decisions:
  `10600`
- realized overrides:
  `150`
- override rate:
  `1.4151%`
- EV/hand:
  `+0.0148`
- estimated EV/hand CI from realized fired deltas:
  `[-0.0077, +0.0374]`
- seed mean EV/hand:
  `+0.0138`
- seed CI:
  `[-0.0071, +0.0346]`
- realized per-fire:
  `+1.0482`
- realized per-fire CI:
  `[-0.5470, +2.6434]`
- confirm delta mean on fired:
  `+2.5802`
- realized loss count:
  `25`
- p95 loss:
  `17.2270`
- max loss:
  `39.2270`
- negative predicted-delta overrides:
  `0`
- replay-ready decisions:
  `10600 / 10600`
- replay-ready fired rows:
  `150 / 150`
- replay missing fields:
  `{}`

GCP postprocess:

- replay pack:
  `150` all-fired rows, `25` realized false-positive rows, `56`
  neutral-overconfirm rows
- false-positive-only MC512 replay:
  `25 / 25` rows replayed
- MC512 labels:
  - `positive`: `20`
  - `gray`: `4`
  - `negative`: `1`
- loss target collection:
  `outputs/evals/regular-hu-t2-stage8c-rr-20260614-1426_postprocess/loss_target_collection/`
- collection rows:
  `25`
- local EV hard negatives:
  `1`
- whole-game risk only:
  `24`
- risk target audit:
  `outputs/evals/regular-hu-t2-stage8c-rr-20260614-1426_postprocess/risk_target_audit/`
- risk target audit replay-ready rows:
  `25 / 25`
- risk target audit blockers:
  `risk_only_rows_lt_min`, `missing_non_loss_control_rows`,
  `single_seat_only`, `local_ev_hard_negatives_lt_min`

Interpretation:

- This is the first completed GCP multi-seed replay-ready target50 run under
  the hidden-discard/current-FL-EV Stage8c TopK path.
- It is directionally positive but not production evidence: EV/hand CI still
  includes zero and the run is first-seat only.
- The confirm-vs-realized gap persists (`+2.5802` confirm mean vs `+1.0482`
  realized per-fire), so confirm deltas remain gate diagnostics only.
- The loss-target split is now more stable across a larger sample: only `1 /
  25` realized losses is a local EV hard negative under MC512; `24 / 25` are
  whole-game risk-only. This reinforces the need for a separate downstream
  risk/counterfactual head rather than treating realized losses as local EV
  negatives.
- The risk-target audit confirms the rows are replay-ready but not yet ready
  for risk-head training: the sample is first-seat-only, has no non-loss
  controls, and is far below the current minimum row counts for both risk-only
  and local-EV hard-negative supervised training.

Follow-up risk-control extraction smoke:

- output:
  `outputs/evals/hu_turn2_stage8c_risk_control_extraction_smoke/`
- source:
  existing base mixed5k first-seat 8-seed / 400-fired all-fired replay MC128
- change:
  `prepare_hu_turn2_stage8b_counterfactual_loss_targets --include-non-loss-controls`
  now emits non-loss fired controls for a separate whole-game risk head.
- extracted rows:
  `400`
- replay-ready rows:
  `400 / 400`
- whole-game non-loss controls:
  `345`
- whole-game risk-only losses:
  `53`
- local EV hard negatives:
  `2`
- risk-head readiness:
  `False`
- remaining risk-head blockers:
  `risk_only_rows_lt_min`, `single_seat_only`,
- local EV hard-negative blockers:
  `local_ev_hard_negatives_lt_min`

Interpretation:

- The previous `missing_non_loss_control_rows` blocker is solved at the
  pipeline level when all-fired replay is available.
- The dataset is still not suitable for risk-head training because it is
  first-seat-only and has too few realized risk-only loss rows and local-EV
  hard negatives.
- Future Stage8c risk-head work should collect second-seat fired rows and more
  whole-game risk-only losses before any supervised risk-head training.

Second-seat replay-ready GCP risk-data run:

- run name:
  `regular-hu-t2-stage8c-risk-second-20260614-1517`
- purpose:
  explicit second-seat risk-data collection only; this is not production or
  P2-fixed evidence.
- launcher guard:
  `scripts/Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1` requires
  `-AllowRiskDataSeatScope` for `seat=second` configs and records
  `risk_data_collection = true`.
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=second/bygate_delta`
- scale:
  `3` Spot VM shards, `3` seeds, target `50` realized overrides per seed.
- aggregate output:
  `outputs/evals/regular-hu-t2-stage8c-risk-second-20260614-1517_aggregate/`
- paired hands:
  `4863`
- fired rows:
  `150`
- replay-ready fired rows:
  `150 / 150`
- EV/hand:
  `+0.0314`
- realized per-fire delta:
  `+2.0336`
- per-fire CI:
  `[+0.6801, +3.3871]`
- realized losses:
  `22`
- max realized loss:
  `23.2270`
- non-fired nonzero:
  `0`

Second-seat postprocess:

- output:
  `outputs/evals/regular-hu-t2-stage8c-risk-second-20260614-1517_postprocess/`
- all-fired replay:
  `150 / 150` rows at MC128
- false-positive replay:
  `22 / 22` rows at MC512
- loss target collection:
  `150` deduped rows
- whole-game non-loss controls:
  `128`
- whole-game risk-only losses:
  `21`
- local EV hard negatives:
  `1`
- risk-target audit blockers:
  `risk_only_rows_lt_min`, `single_seat_only`
- local EV hard-negative blockers:
  `local_ev_hard_negatives_lt_min`

Combined first+second risk-target audit:

- output:
  `outputs/evals/hu_turn2_stage8c_risk_first_second_combined_audit/`
- inputs:
  first-seat all-fired risk-control smoke plus the second-seat GCP postprocess.
- rows:
  `550`
- replay-ready rows:
  `550 / 550`
- seats:
  `first = 400`, `second = 150`
- whole-game non-loss controls:
  `473`
- whole-game risk-only losses:
  `74`
- local EV hard negatives:
  `3`
- realized delta mean:
  `+2.0073`
- local replay delta mean:
  `+1.9706`
- risk-head readiness:
  `False`
- remaining risk-head blockers:
  `risk_only_rows_lt_min`
- local EV hard-negative blockers:
  `local_ev_hard_negatives_lt_min`

Interpretation:

- The second-seat run is execution-pass and directionally positive in realized
  per-fire terms, with non-fired cancellation intact.
- The previous `single_seat_only` blocker is solved when combining the current
  first-seat and second-seat risk-data collections.
- The dataset is still not ready for risk-head training: it has enough
  non-loss controls and both seats, but still too few whole-game risk-only
  losses.
- Local EV hard-negative training is tracked separately and remains No-Go
  because there are still far too few local EV hard negatives.
- Do not train a risk head yet. The next useful collection should target more
  fired rows and especially more realized risk-only losses/local replay
  negatives, while preserving exact replay packets and non-fired cancellation.

Expanded risk-data collection:

- main run:
  `regular-hu-t2-stage8c-risk-expand-20260614-1548`
- purpose:
  collect enough first/second fired rows for a supervised whole-game risk head.
- configs:
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=second/bygate_delta`
- target:
  `5` seeds per config, `100` realized overrides per seed.
- completed shards:
  `8 / 10`; two Spot shards were missing and the run was aggregated with
  `Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -AllowPartial`.
- aggregate output:
  `outputs/evals/regular-hu-t2-stage8c-risk-expand-20260614-1548_aggregate_partial/`
- fired rows:
  `400` first-seat and `400` second-seat
- non-fired nonzero:
  `0` for both configs
- first-seat EV/hand:
  `+0.0294`
- first-seat realized per-fire delta:
  `+2.0054`
- second-seat EV/hand:
  `+0.0175`
- second-seat realized per-fire delta:
  `+1.1365`
- postprocess output:
  `outputs/evals/regular-hu-t2-stage8c-risk-expand-20260614-1548_partial_postprocess/`
- expanded loss target collection:
  `800` rows, `666` whole-game non-loss controls, `121` whole-game risk-only
  losses, `13` local EV hard negatives

Risk-data fill run:

- run:
  `regular-hu-t2-stage8c-risk-fill-20260614-1722`
- purpose:
  fill the final gap from `195 / 200` risk-only rows to risk-head readiness.
- target:
  one first-seat shard and one second-seat shard, `50` realized overrides each.
- aggregate output:
  `outputs/evals/regular-hu-t2-stage8c-risk-fill-20260614-1722_aggregate/`
- fired rows:
  `50` first-seat and `50` second-seat
- non-fired nonzero:
  `0` for both configs
- postprocess output:
  `outputs/evals/regular-hu-t2-stage8c-risk-fill-20260614-1722_postprocess/`
- fill loss target collection:
  `100` rows, `78` whole-game non-loss controls, `15` whole-game risk-only
  losses, `7` local EV hard negatives

Final combined risk-target audit:

- output:
  `outputs/evals/hu_turn2_stage8c_risk_combined_ready_audit/`
- inputs:
  first-seat risk-control smoke, second-seat GCP postprocess, expanded partial
  postprocess, and fill postprocess.
- rows:
  `1450`
- replay-ready rows:
  `1450 / 1450`
- seats:
  `first = 850`, `second = 600`
- whole-game non-loss controls:
  `1217`
- whole-game risk-only losses:
  `210`
- local EV hard negatives:
  `23`
- realized delta mean:
  `+1.7003`
- local replay delta mean:
  `+1.9257`
- risk-head readiness:
  `True`
- local EV hard-negative readiness:
  `True`
- blockers:
  `none`

Interpretation:

- The risk-data collection is now sufficient to start a supervised
  risk/counterfactual-head training smoke.
- This still does not approve production, P2 fixed status, T1, or 50k teacher
  generation.
- The next model work should train a separate whole-game risk head from the
  `whole_game_risk_only` and `whole_game_non_loss_control` rows, while keeping
  the `local_ev_hard_negative` rows separate for local EV/gate hard-negative
  training.

Stage8c whole-game risk-head smoke:

- trainer:
  `src/ofc_regular/train_hu_turn2_stage8c_risk_head.py`
- test coverage:
  `tests/test_hu_turn2_stage8c_risk_head_training.py`
- output:
  `outputs/training/hu_turn2_stage8c_risk_head_smoke/`
- model:
  `models/hu_turn2_stage8c_risk_head_smoke.pt`
- inputs:
  the four replay-ready risk target collections from the final combined audit.
- trainable rows:
  `1427`
- positives:
  `210` whole-game risk-only rows
- negatives:
  `1217` whole-game non-loss controls
- explicitly excluded:
  `23` local EV hard-negative rows
- best epoch:
  `4`
- validation metrics:
  AP `0.1790`, ROC AUC `0.5527`
- test metrics:
  AP `0.1963`, ROC AUC `0.5920`

Interpretation:

- The risk-head training path is now execution-pass: the replay-ready
  whole-game risk/control rows can be materialized into HU feature rows and
  trained without mixing local EV hard negatives into the whole-game label.
- The current smoke model is not strong enough for runtime integration. It
  overfits the train split and has weak validation/test ranking signal.
- Treat `hu_turn2_stage8c_risk_head_smoke.pt` as a pipeline proof only.
- Production, P2 fixed status, 50k teacher generation, and T1 training remain
  No-Go.
- The next useful work is a calibration/error audit of the risk head and/or
  more diverse risk/control data, not runtime deployment.

Risk-head calibration/error audit:

- auditor:
  `src/ofc_regular/analyze_hu_turn2_stage8c_risk_head.py`
- test coverage:
  `tests/test_hu_turn2_stage8c_risk_head_audit.py`
- output:
  `outputs/training/hu_turn2_stage8c_risk_head_smoke_audit/`
- decision:
  runtime integration `false`
- blockers:
  `val_auc_lt_0p65`, `test_auc_lt_0p65`,
  `train_test_auc_gap_gt_0p25`, `holdout_ap_lt_0p25`
- validation metrics:
  AP `0.1790`, ROC AUC `0.5527`
- test metrics:
  AP `0.1963`, ROC AUC `0.5920`
- threshold sweep:
  high thresholds look precise only on the full in-sample aggregate; on val/test
  they either fire almost nothing or have poor precision.
- decile audit:
  val/test top deciles do not cleanly concentrate the whole-game risk-only
  positives.

Low-capacity risk-head retrain probes:

- comparison artifact:
  `outputs/training/hu_turn2_stage8c_risk_head_model_comparison.csv`
- tried:
  - `tiny32_wd1e2_do0p2`
  - `tiny16_wd1e2_do0p1`
  - `small64_32_wd1e3_do0p3`
- best test ROC AUC among these probes:
  `0.5841`, still below the `0.65` audit floor
- best validation ROC AUC among these probes:
  `0.5276`, also below the audit floor

Runtime-meta risk-head probes:

- trainer now supports:
  `--feature-mode hu_plus_runtime_meta`
- appended runtime-available features:
  `predicted_delta`, `abs(predicted_delta)`, `gate_probability`,
  `logit(gate_probability)`, `confirm_delta`, `confirm_delta_se`,
  `confirm_delta / confirm_delta_se`, `confirm_delta - predicted_delta`,
  candidate rank features, and seat flag.
- tried:
  - `meta_256_128_do0p10`
  - `meta_tiny32_wd1e2_do0p2`
  - `meta_small64_32_wd1e3_do0p3`
- best validation ROC AUC with runtime meta:
  `0.6149`
- best test ROC AUC with runtime meta:
  `0.5852`
- best validation AP with runtime meta:
  `0.1995`
- best test AP with runtime meta:
  `0.2019`

Raw runtime score baseline audit:

- artifact:
  `outputs/training/hu_turn2_stage8c_risk_head_smoke_audit/risk_head_raw_score_baselines.csv`
- purpose:
  check whether existing runtime values separate whole-game risk-only losses
  without training a new head.
- best validation raw score:
  `confirm_delta`, AP `0.1967`, ROC AUC `0.6142`
- best test raw score:
  `gate_probability`, AP `0.2932`, ROC AUC `0.6223`
- other notable test scores:
  - risk head probability: AP `0.1963`, ROC AUC `0.5920`
  - confirm delta: AP `0.2265`, ROC AUC `0.5687`
  - confirm minus predicted delta: AP `0.2059`, ROC AUC `0.5711`

Source-seed holdout audit:

- trainer now supports:
  `--split-mode source_seed`
- purpose:
  keep all rows from the same source seed in a single split, instead of mixing
  nearby same-run rows across train/val/test.
- source-seed split check:
  `17` groups, `0` groups crossing splits.
- source-seed `hu_only_256_128`:
  - val AP `0.2342`, ROC AUC `0.6019`
  - test AP `0.2571`, ROC AUC `0.5790`
- source-seed `meta_small64_32`:
  - val AP `0.2446`, ROC AUC `0.6075`
  - test AP `0.2541`, ROC AUC `0.6063`
- artifact:
  `outputs/training/hu_turn2_stage8c_risk_head_model_comparison.csv`

Interpretation:

- The first risk-head failure is not fixed by simply shrinking/regularizing the
  MLP.
- Adding the obvious runtime TopK/confirm metadata improves validation AUC a
  little, but it still does not reach the audit floor and does not improve test
  AUC over the HU-only smoke.
- The raw runtime scores themselves are also weak. `confirm_delta` and
  `gate_probability` contain some signal, but not enough for a production-safe
  whole-game risk gate.
- The stricter source-seed holdout does not rescue the risk head. It gives a
  slightly cleaner estimate and AP improves because of the split composition,
  but AUC remains around `0.58`-`0.61`, still below the audit floor.
- Current features and labels do not yet separate whole-game risk-only losses
  from non-loss controls well enough on heldout rows.
- Continue treating the risk head as a diagnostic pipeline. The next useful
  step is not runtime deployment; it is either more diverse risk/control data,
  richer counterfactual/downstream features, or a revised target that separates
  downstream trajectory risk from local action quality more explicitly.

Risk-target gap analysis:

- analyzer:
  `src/ofc_regular/analyze_hu_turn2_stage8c_risk_target_gap.py`
- test coverage:
  `tests/test_hu_turn2_stage8c_risk_target_gap.py`
- output:
  `outputs/training/hu_turn2_stage8c_risk_target_gap_analysis/`
- inputs:
  the same four replay-ready risk target collections used by the risk-head
  training smoke.
- rows:
  `1450`
- whole-game risk-only losses:
  `210`
- whole-game non-loss controls:
  `1217`
- local EV hard negatives:
  `23`
- risk-only rows with local positive replay:
  `156 / 210` (`74.3%`)
- risk-only rows with local gray replay:
  `54 / 210`
- risk-only rows with local negative replay:
  `0 / 210`
- non-loss controls with local negative replay:
  `190 / 1217` (`15.6%`)
- downstream trajectory field coverage:
  `0 / 15` tracked fields are present in any row.

Interpretation:

- Most whole-game risk-only losses are locally positive at T2. The loss is not
  explained by local action EV alone.
- Some locally negative rows are whole-game non-loss controls, so local
  hard-negative labels and whole-game risk labels are not interchangeable.
- The current risk head is being asked to infer downstream foul/FL/royalty/scoop
  outcomes from a single T2 state/action snapshot. That explains why smaller
  models, runtime metadata, and source-seed splits did not rescue the AUC.
- Before another serious risk-head attempt, collect trajectory decomposition
  fields: post-T2 candidate/baseline boards, T3 continuation summaries,
  downstream override fired flags, final boards, foul flags, FL entry/stay
  flags, royalty deltas, line/scoop deltas, and per-future paired delta
  summaries.
- Runtime risk integration, production/P2 fixed status, 50k teacher generation,
  and T1 training remain `No-Go`.

Trajectory instrumentation added after this audit:

- runtime evaluator:
  `src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py`
- loss-target builder:
  `src/ofc_regular/prepare_hu_turn2_stage8b_counterfactual_loss_targets.py`
- tests:
  - `tests/test_hu_turn2_stage8b_training_prep.py`
  - `tests/test_hu_turn2_stage8b_counterfactual_loss_targets.py`
- New `runtime_decisions.jsonl` rows include:
  - `post_t2_candidate_board`
  - `post_t2_baseline_board`
  - candidate and baseline final hero/opponent boards
  - candidate and baseline foul flags
  - candidate and baseline royalty totals and deltas
  - candidate and baseline FL entry/value flags
  - candidate and baseline line/scoop/foul score components
  - component deltas versus the baseline trace when the opposite seat-swap is
    baseline-like.
- The loss-target builder propagates those fields into
  `topk_counterfactual_loss_targets*.jsonl`; unprefixed fields represent the
  candidate outcome, while `baseline_*` fields preserve the counterfactual
  baseline outcome.
- Existing historical risk-target artifacts remain trajectory-incomplete. New
  evidence must come from rerunning the evaluator and loss-target postprocess
  after this instrumentation.
- T3 decision summaries and downstream override-fired decomposition are still
  not available in the trace; add them before a second serious risk-head model
  if the final-board decomposition is not sufficient.

Trajectory schema smoke:

- output:
  `outputs/evals/hu_turn2_stage8c_trajectory_schema_smoke/`
- purpose:
  prove the new runtime trajectory fields survive through
  `prepare_hu_turn2_stage8b_counterfactual_loss_targets` and the risk-target
  gap analyzer.
- config:
  diagnostic-only `k3/mc4/d-999/se0/pd-999/seat=first/bygate_delta`.
- paired seeds:
  `2` before target realized overrides was reached.
- realized fired rows:
  `2`
- loss target output:
  `outputs/evals/hu_turn2_stage8c_trajectory_schema_smoke/loss_target_collection/`
- gap analysis output:
  `outputs/evals/hu_turn2_stage8c_trajectory_schema_smoke/risk_target_gap_analysis_collection_dir/`
- downstream field coverage:
  `13 / 15` tracked fields present in all rows.
- still missing by design:
  `t3_decision_summary`, `downstream_override_fired`.

Interpretation:

- The previous `0 / 15` coverage problem is fixed for new TopK evaluation logs.
- This smoke is not EV evidence; it uses a deliberately loose diagnostic config.
- The next real risk-data run can now produce final-board/foul/FL/royalty/line
  decomposition for fired rows, which is the minimum needed before another
  serious risk-head attempt.

Trajectory component analyzer:

- analyzer:
  `src/ofc_regular/analyze_hu_turn2_stage8c_trajectory_components.py`
- test coverage:
  `tests/test_hu_turn2_stage8c_trajectory_components.py`
- smoke output:
  `outputs/evals/hu_turn2_stage8c_trajectory_schema_smoke/trajectory_component_analysis/`
- purpose:
  explain realized candidate-vs-baseline deltas by terminal score components:
  foul, line, scoop, royalty, and FL.
- component coverage on the schema smoke:
  `10 / 10` tracked component fields present.
- smoke loss rows:
  `0`; this is expected because the schema smoke was not a quality/risk-data
  run.

Use this analyzer after the next trajectory-enriched risk-data run. The key
artifact is `trajectory_component_loss_breakdown.csv`, which should show whether
whole-game risk-only losses are mainly foul-risk, FL-overvaluation, royalty
swings, line/scoop swings, or unexplained by terminal components.

Trajectory real-small run with component decomposition:

- output:
  `outputs/evals/hu_turn2_stage8c_trajectory_real_small/`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- T3 continuation:
  `stage3_reference_default`
- seed:
  `2026069951`
- paired seeds reached:
  `463 / 500`
- decision rows:
  `926`
- runtime overrides:
  `10`
- realized overrides:
  `10`
- non-fired cancellation audit:
  `906` non-fired rows, `0` nonzero non-fired deltas.
- aggregate EV/hand:
  `-0.0132`
- realized per-fire mean:
  `-1.2227`
- confirm delta mean on fired rows:
  `+2.8945`
- per-fire p95 loss:
  `24.9270`
- max loss:
  `31.2270`
- loss target rows:
  `10`
- component coverage:
  `10 / 10` component fields present.
- downstream field coverage:
  `13 / 15` tracked fields present.
- still missing:
  `t3_decision_summary`, `downstream_override_fired`.
- loss component breakdown:
  - `royalty`: `1` row, realized delta `-31.2270`; includes
    `-15.0` hero royalty, `-10.2270` hero FL value, and `-6.0`
    foul-score delta versus baseline.
  - `fl`: `1` row, realized delta `-17.2270`; includes `-7.0` hero
    royalty and `-10.2270` hero FL value versus baseline.
  - `non_loss`: `8` rows, mean realized delta `+4.5284`.

Interpretation:

- This is not adoption evidence: it is a one-seed, first-seat, target10
  diagnostic run.
- The result confirms that non-fired cancellation works in this harness.
- The realized per-fire result is negative while the confirm delta on fired rows
  is strongly positive. That is another concrete example of why confirm-MC
  deltas must stay gate diagnostics, not performance claims.
- The two loss rows are explained by FL/royalty swings in the terminal
  decomposition, not by missing component data. This supports using the new
  component fields for future risk-head data rather than mixing whole-game
  losses directly into local EV labels.
- The top-loss audit JSONL now includes the replay/action context
  (`hero_board`, `cards_to_place`, `baseline_action`, `candidate_action`, and
  post-T2 candidate/baseline boards), so it is usable for manual failure review
  and selected replay targeting.

Trajectory target50 follow-up:

- output:
  `outputs/evals/hu_turn2_stage8c_trajectory_target50_seed2026069961/`
- purpose:
  collect enough fired decisions under the new trajectory instrumentation to
  read realized per-fire quality instead of confirm-delta diagnostics.
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- T3 continuation:
  `stage3_reference_default`
- seed:
  `2026069961`
- paired seeds reached:
  `1,313 / 5,000`
- realized override target:
  `50`
- target reached:
  yes
- elapsed:
  `647.52s`
- decision rows:
  `2,626`
- runtime overrides:
  `50`
- realized overrides:
  `50`
- runtime override rate:
  `1.904%`
- non-fired cancellation audit:
  `2,526` non-fired rows, `0` nonzero non-fired deltas.
- aggregate EV/hand:
  `+0.1173`
- seed-mean CI:
  `[+0.0471, +0.1876]`
- realized per-fire mean:
  `+6.1627`
- realized per-fire CI:
  `[+2.8438, +9.4816]`
- confirm delta mean on fired rows:
  `+2.4897`
- p95 loss:
  `2.0`
- max loss:
  `24.2270`
- predicted-delta safety audit:
  `0 / 50` overrides had negative model `predicted_delta`.
- position split:
  - first seat: `50 / 1,313` decisions fired, `+0.3345 EV/hand` on the
    first-seat half.
  - second seat: `0 / 1,313` decisions fired, `-0.0998 EV/hand` on the
    second-seat half from paired downstream variance.
- loss target rows:
  `50`
- component coverage:
  `10 / 10` component fields present.
- downstream field coverage:
  `13 / 15` tracked fields present.
- still missing:
  `t3_decision_summary`, `downstream_override_fired`.
- loss rows:
  `5 / 50`
- loss component breakdown:
  - `fl`: `2` rows, mean realized delta `-22.2270`, max loss `24.2270`.
  - `royalty`: `2` rows, mean realized delta `-2.0`.
  - `line`: `1` row, realized delta `-2.0`.
  - `non_loss`: `45` rows, mean realized delta `+7.9686`.

Interpretation:

- This is the first trajectory-enriched first-seat target50 result after the
  component instrumentation. It is a useful Conditional-Go signal for continuing
  TopK+confirm work, not a production/P2 signal.
- Non-fired cancellation remains exact, so the per-fire estimate is meaningful.
- Unlike the target10 diagnostic, realized per-fire is strongly positive and is
  not being propped up by confirm-delta reporting.
- Tail losses are still FL/royalty/line-component issues; use these rows for
  failure review and selected replay/risk-data enrichment.
- The evidence is still one seed and first-seat-only. Before any Spot-scale
  or production-oriented claim, repeat on non-overlapping seeds and include
  second-seat validation.

Trajectory target50 second-seat follow-up:

- output:
  `outputs/evals/hu_turn2_stage8c_trajectory_target50_second_seed2026069962/`
- purpose:
  check whether the same trajectory-enriched `confirm128/cse1/pd0` TopK path
  can fire profitably when restricted to second-seat decisions.
- config:
  `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=second/bygate_delta`
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- T3 continuation:
  `stage3_reference_default`
- seed:
  `2026069962`
- paired seeds reached:
  `1,400 / 5,000`
- realized override target:
  `50`
- target reached:
  yes
- elapsed:
  `427.23s`
- decision rows:
  `2,800`
- runtime overrides:
  `50`
- realized overrides:
  `50`
- runtime override rate:
  `1.786%`
- non-fired cancellation audit:
  `2,700` non-fired rows, `0` nonzero non-fired deltas.
- aggregate EV/hand:
  `+0.0533`
- seed-mean CI:
  `[+0.0102, +0.0964]`
- realized per-fire mean:
  `+2.9845`
- realized per-fire CI:
  `[+0.6893, +5.2797]`
- confirm delta mean on fired rows:
  `+2.3895`
- p95 loss:
  `5.0`
- max loss:
  `20.2270`
- predicted-delta safety audit:
  `0 / 50` overrides had negative model `predicted_delta`.
- loss target rows:
  `50`
- component coverage:
  `10 / 10` component fields present.
- downstream field coverage:
  `13 / 15` tracked fields present.
- still missing:
  `t3_decision_summary`, `downstream_override_fired`.
- loss rows:
  `6 / 50`
- loss component breakdown:
  - `fl`: `1` row, realized delta `-20.2270`.
  - `royalty`: `3` rows, mean realized delta `-4.3333`.
  - `scoop`: `1` row, realized delta `-5.0`.
  - `line`: `1` row, realized delta `-2.0`.
  - `non_loss`: `44` rows, mean realized delta `+4.3058`.

First+second target50 combined snapshot:

- rows:
  `100` fired decisions across the first-seat and second-seat target50 runs.
- realized delta sum:
  `+457.3621`
- combined realized per-fire mean:
  `+4.5736`
- combined decision rows:
  `5,426`
- implied EV/decision:
  `+0.0843`
- loss rows:
  `11 / 100`
- non-fired cancellation:
  exact in both runs.

Interpretation:

- This removes the immediate "first-seat only" blocker for continuing the
  TopK+confirm research path: both first-seat and second-seat target50 probes
  are positive with exact non-fired cancellation.
- Second-seat is weaker than first-seat but still positive on this one
  non-overlapping seed.
- This is still not production/P2 evidence. The current evidence is only one
  target50 seed per seat, and both use the same model/config family.
- Next useful validation is a multi-seed target50 run per seat, preferably on
  Spot VM if local runtime is inconvenient, followed by the same trajectory
  component postprocess. Do not move to T1 or 50k teacher from these two probes
  alone.

GCP multi-seed target50 first+second run:

- run:
  `regular-hu-t2-stage8c-target50-both-20260614-1930`
- aggregate output:
  `outputs/evals/regular-hu-t2-stage8c-target50-both-20260614-1930_aggregate/`
- downloaded shard output:
  `outputs/gcp_runs/regular-hu-t2-stage8c-target50-both-20260614-1930/`
- trajectory postprocess:
  `outputs/evals/regular-hu-t2-stage8c-target50-both-20260614-1930_trajectory_postprocess/`
- GCP setup:
  6 Spot VMs, `e2-highcpu-8`, one shard per config/seed.
- configs:
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta`
  - `k5/mc64/d0/se0/confirm128/cse1/pd0/seat=second/bygate_delta`
- seeds:
  `2026069971`, `2026069972`, `2026069973`
- T3 continuation:
  `stage3_reference_default`
- execution:
  6 / 6 shards complete, 0 failed, no running instances left.

First-seat aggregate:

- paired seeds:
  `4,993`
- fired decisions:
  `150`
- replay-ready fired decisions:
  `150 / 150`
- weighted EV/hand:
  `+0.0529`
- seed mean CI:
  `[+0.0280, +0.0786]`
- realized per-fire mean:
  `+3.5227`
- realized per-fire CI:
  `[+1.8655, +5.1799]`
- losses:
  `17 / 150`
- max loss:
  `23.2270`
- non-fired nonzero:
  `0`
- negative predicted-delta overrides:
  `0`
- seed EV/hand:
  - `2026069971`: `+0.0701`
  - `2026069972`: `+0.0279`
  - `2026069973`: `+0.0619`

Second-seat aggregate:

- paired seeds:
  `4,760`
- fired decisions:
  `150`
- replay-ready fired decisions:
  `150 / 150`
- weighted EV/hand:
  `+0.0169`
- seed mean CI:
  `[-0.0129, +0.0444]`
- realized per-fire mean:
  `+1.0733`
- realized per-fire CI:
  `[-0.3835, +2.5301]`
- losses:
  `26 / 150`
- max loss:
  `37.2270`
- non-fired nonzero:
  `0`
- negative predicted-delta overrides:
  `0`
- seed EV/hand:
  - `2026069971`: `-0.0130`
  - `2026069972`: `+0.0345`
  - `2026069973`: `+0.0258`

Trajectory component analysis over all 300 fired rows:

- rows:
  `300`
- loss rows:
  `43`
- mean realized delta:
  `+2.2980`
- component coverage:
  `10 / 10`
- downstream coverage:
  `13 / 15`; still missing `t3_decision_summary` and
  `downstream_override_fired`.
- primary loss components:
  - `royalty`: `22`
  - `fl`: `10`
  - `foul`: `5`
  - `line`: `4`
  - `scoop`: `2`
  - `non_loss`: `257`

Interpretation:

- The GCP multi-seed run confirms the first-seat TopK+confirm signal much more
  strongly than the one-seed local probes.
- The second-seat result is execution-pass but not yet decision-strong: it is
  positive on weighted EV/hand, but its seed CI and per-fire CI still cross
  zero, and one seed is negative.
- Non-fired cancellation is exact for both seats, so the per-fire estimates are
  meaningful.
- The remaining loss tail is mostly royalty/FL-driven, with a smaller foul
  component. This is a useful target for risk-data enrichment.
- Do not mark T2 as production/P2 and do not start T1 or 50k teacher from this
  artifact alone. A reasonable next step is to either:
  - expand second-seat target50 on more non-overlapping seeds, or
  - run a stricter second-seat/risk-filter hypothesis and compare against this
    baseline.

Post-hoc second-seat runtime risk-filter audit:

- second-seat output:
  `outputs/evals/regular-hu-t2-stage8c-target50-both-20260614-1930_second_seat_risk_filter/`
- first-seat comparison output:
  `outputs/evals/regular-hu-t2-stage8c-target50-both-20260614-1930_first_seat_risk_filter/`
- compact comparison:
  `outputs/evals/regular-hu-t2-stage8c-target50-both-20260614-1930_second_first_guard_comparison.csv`
- input:
  the same six GCP target50 shard `runtime_decisions.jsonl` files from
  `regular-hu-t2-stage8c-target50-both-20260614-1930`.
- method:
  post-hoc only; runtime-available filters were applied to already-fired
  decisions and scored by realized whole-game deltas. Confirm deltas remain
  gate inputs only, not performance claims.

Second-seat post-hoc guard highlights:

- unfiltered:
  `150` fires, `+0.0169 EV/hand`, `+1.0733` per fire,
  per-fire CI `[-0.3835, +2.5301]`, `26` losses, max loss `37.2270`.
- `pd0_cd1_cz1`:
  `115` fires, `+0.0231 EV/hand`, `+1.9083` per fire,
  per-fire CI `[+0.1285, +3.6881]`, `21` losses, max loss still `37.2270`.
  This is the best broad second-seat hypothesis: higher EV and positive
  per-fire CI while keeping enough fires.
- `pd0.5_cd1_cz1`:
  `73` fires, `+0.0209 EV/hand`, `+2.7217` per fire,
  per-fire CI `[+0.5613, +4.8820]`, `12` losses, max loss `18.2270`.
  This is a stricter tail-risk hypothesis: lower total EV than `pd0_cd1_cz1`,
  but it cuts the observed max loss roughly in half.
- `pd1_cd1_cz1`:
  `51` second-seat fires, `+0.0158 EV/hand`, max loss `17.2270`.
  Too sparse for a main candidate, but useful as a diagnostic strict guard.

First-seat comparison for the same guards:

- unfiltered remains strongest:
  `150` fires, `+0.0529 EV/hand`, `+3.5227` per fire,
  per-fire CI `[+1.8655, +5.1799]`.
- `pd0_cd1_cz1` stays positive but gives up EV:
  `125` fires, `+0.0450 EV/hand`.
- `pd0.5_cd1_cz1` stays positive but is much sparser:
  `79` fires, `+0.0302 EV/hand`.

Interpretation:

- The most plausible next validation is position-specific:
  keep first-seat at the current unfiltered `pd0/cse1` config and test second
  seat with either `pd0_cd1_cz1` or `pd0.5_cd1_cz1`.
- `pd0_cd1_cz1` is the main second-seat candidate because it improves the
  weak second-seat result without cutting too many fires.
- `pd0.5_cd1_cz1` is the conservative tail-risk comparison candidate.
- This is in-sample post-hoc analysis. It does not approve a runtime change.
  Any guard must be validated on non-overlapping seeds before it can affect
  C4/P2/production decisions.

Non-overlapping validation of second-seat `cd1` guard:

- run:
  `regular-hu-t2-stage8c-second-guard-cd1-20260614-validate`
- local output:
  `outputs/gcp_runs/regular-hu-t2-stage8c-second-guard-cd1-20260614-validate/`
- trajectory postprocess:
  `outputs/evals/regular-hu-t2-stage8c-second-guard-cd1-20260614-validate_trajectory_postprocess/`
- execution:
  6 / 6 Spot shards completed, 0 failed, no running instances left.
- seeds:
  `2026070101`, `2026070102`, `2026070103`
- T3 continuation:
  `stage3_reference_default`
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- configs:
  - `k5/mc64/d0/se0/confirm128/cse1/cd1/pd0/seat=second/bygate_delta`
  - `k5/mc64/d0/se0/confirm128/cse1/cd1/pd0.5/seat=second/bygate_delta`

Implementation note:

- `cd1` is now a runtime config token for `min_confirm_delta = 1.0`.
- It is different from raising `d` to `1`: `d` remains the base rerank delta
  threshold, while `cd` applies only to the independent confirm stage.
- Runtime logs, seed breakdown, and aggregate outputs include
  `min_confirm_delta` so these runs can be distinguished from `d1` runs.

Results:

| config | paired | fires | EV/hand | per-fire | per-fire CI | losses | max loss |
|---|---:|---:|---:|---:|---|---:|---:|
| `pd0_cd1` | 6,514 | 150 | +0.0243 | +2.1127 | [+0.5069, +3.7186] | 27 | 29.2270 |
| `pd0.5_cd1` | 9,105 | 150 | +0.0035 | +0.4303 | [-1.3500, +2.2107] | 37 | 35.2270 |

Seed split:

- `pd0_cd1`:
  - `2026070101`: `+0.0474 EV/hand`, `+3.4736` per fire
  - `2026070102`: `+0.0214 EV/hand`, `+2.1291` per fire
  - `2026070103`: `+0.0084 EV/hand`, `+0.7355` per fire
- `pd0.5_cd1`:
  - `2026070101`: `+0.0017 EV/hand`, `+0.1909` per fire
  - `2026070102`: `+0.0171 EV/hand`, `+2.1736` per fire
  - `2026070103`: `-0.0087 EV/hand`, `-1.0736` per fire

Cancellation and safety:

- non-fired nonzero:
  `0` for both configs.
- replay-ready fired rows:
  `150 / 150` for both configs.
- negative predicted-delta overrides:
  `0` for both configs.

Trajectory component analysis over the 300 fired rows:

- rows:
  `300`
- loss rows:
  `64`
- mean realized delta:
  `+1.2715`
- component coverage:
  all tracked component fields present.
- primary loss components:
  - `royalty`: `28`
  - `fl`: `19`
  - `line`: `9`
  - `foul`: `7`
  - `scoop`: `1`
- config-level component read:
  - `pd0_cd1`: `27 / 150` losses, mean realized delta `+2.1127`,
    p95 loss `15.0`, max loss `29.2270`.
  - `pd0.5_cd1`: `37 / 150` losses, mean realized delta `+0.4303`,
    p95 loss `22.2270`, max loss `35.2270`.

Interpretation:

- The broad second-seat `pd0_cd1` guard reproduced out of sample and is now the
  leading second-seat validation candidate.
- The stricter `pd0.5_cd1` guard did not reproduce the desired tail-risk
  improvement and should not be promoted.
- This is still not a production/P2 result. The next validation should test a
  position-specific policy on fresh seeds: first-seat remains unfiltered
  `pd0/cse1`, second-seat uses `pd0_cd1`.

Position-specific integrated `scd1` validation:

- run:
  `regular-hu-t2-stage8c-position-specific-scd1-20260614-validate`
- local output:
  `outputs/gcp_runs/regular-hu-t2-stage8c-position-specific-scd1-20260614-validate/`
- trajectory postprocess:
  `outputs/evals/regular-hu-t2-stage8c-position-specific-scd1-20260614-validate_trajectory_postprocess/`
- execution:
  3 / 3 Spot shards completed, 0 failed, no running instances left.
- seeds:
  `2026070201`, `2026070202`, `2026070203`
- T3 continuation:
  `stage3_reference_default`
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/scd1/pd0/bygate_delta`

Implementation note:

- `scd1` is a runtime config token for
  `second_min_confirm_delta = 1.0`.
- It leaves first-seat unfiltered at `pd0/cse1`, while applying the `cd1`
  confirm-stage floor only to second-seat decisions.
- The GCP launcher now requires `-AllowPositionSpecificSeatScope` for configs
  without explicit `seat=first` or `seat=second` but with `fcd`/`scd` guards.
  Broad or ambiguous `seat=both/all/*` configs remain rejected.
- Runtime logs, seed breakdown, and aggregate outputs include
  `first_min_confirm_delta`, `second_min_confirm_delta`, and
  `seat_min_confirm_delta`.

Aggregate result:

| metric | value |
|---|---:|
| paired seeds | 2,832 |
| decisions | 5,664 |
| realized fires | 150 |
| non-fired nonzero | 0 |
| fired replay-ready | 150 / 150 |
| whole-game paired EV/hand | +0.0369 |
| seed-mean EV/hand CI | [+0.0291, +0.0464] |
| conditional per-fire delta | +1.0852 |
| conditional per-fire CI | [-0.5958, +2.7661] |
| conditional estimated EV/hand | +0.0299 |
| realized losses | 24 |
| p95 loss | 20.2270 |
| max loss | 33.2270 |

Seed split:

- `2026070201`: `+0.0452 EV/hand`, `50` realized fires
- `2026070202`: `+0.0299 EV/hand`, `50` realized fires
- `2026070203`: `+0.0382 EV/hand`, `50` realized fires

Seat split over fired decisions:

- first-seat:
  `86` fires, mean realized delta `+1.3166`, CI `[-1.1421, +3.7753]`,
  `16` losses, max loss `33.2270`.
- second-seat:
  `64` fires, mean realized delta `+0.7742`, CI `[-1.3919, +2.9402]`,
  `8` losses, max loss `27.2270`.

Trajectory component analysis over the 150 fired rows:

- rows:
  `150`
- loss rows:
  `24`
- mean realized delta:
  `+1.0852`
- component coverage:
  all tracked component fields present.
- primary loss components:
  - `fl`: `12`
  - `royalty`: `10`
  - `foul`: `1`
  - `scoop`: `1`

Interpretation:

- This is an execution pass and directionally positive integrated validation:
  all three non-overlapping seeds are positive and non-fired cancellation is
  exact.
- It is not decision-strong enough for production/P2 because the conditional
  per-fire CI still crosses zero and both seat-specific per-fire CIs are wide.
- Keep `scd1` as the current integrated position-specific hypothesis. The next
  validation should increase realized fires or run a larger non-overlapping
  integrated check before any production/P2/T1/50k decision.

Larger position-specific integrated `scd1` validation:

- run:
  `regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614`
- local output:
  `outputs/gcp_runs/regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614/`
- trajectory postprocess:
  `outputs/evals/regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614_trajectory_postprocess/`
- execution:
  5 / 5 Spot shards completed, 0 failed, no running instances left.
- seeds:
  `2026070301`, `2026070302`, `2026070303`, `2026070304`, `2026070305`
- T3 continuation:
  `stage3_reference_default`
- model:
  `models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt`
- config:
  `k5/mc64/d0/se0/confirm128/cse1/scd1/pd0/bygate_delta`
- scale:
  `100` realized fires per seed target, `500` realized fires total.

Aggregate result:

| metric | value |
|---|---:|
| paired seeds | 9,829 |
| decisions | 19,658 |
| realized fires | 500 |
| non-fired nonzero | 0 |
| fired replay-ready | 500 / 500 |
| whole-game paired EV/hand | +0.0536 |
| seed-mean EV/hand CI | [+0.0397, +0.0684] |
| conditional per-fire delta | +1.9699 |
| conditional per-fire CI | [+1.0378, +2.9020] |
| conditional estimated EV/hand | +0.0513 |
| estimated EV/hand CI | [+0.0270, +0.0756] |
| realized losses | 82 |
| p95 loss | 17.2270 |
| max loss | 42.2270 |

Seed split:

- `2026070301`: `+0.0623 EV/hand`, `100` fires,
  per-fire `+1.8923`, CI `[-0.1700, +3.9546]`.
- `2026070302`: `+0.0620 EV/hand`, `100` fires,
  per-fire `+2.1900`, CI `[+0.4909, +3.8891]`.
- `2026070303`: `+0.0330 EV/hand`, `100` fires,
  per-fire `+1.3845`, CI `[-1.0397, +3.8088]`.
- `2026070304`: `+0.0720 EV/hand`, `100` fires,
  per-fire `+2.7559`, CI `[+0.8998, +4.6120]`.
- `2026070305`: `+0.0409 EV/hand`, `100` fires,
  per-fire `+1.6268`, CI `[-0.6944, +3.9480]`.

Seat split over fired decisions:

- first-seat:
  `264` fires, mean realized delta `+1.4711`, CI `[+0.2388, +2.7034]`,
  estimated EV/hand `+0.0395`, `39` losses, max loss `42.2270`.
- second-seat:
  `236` fires, mean realized delta `+2.5279`, CI `[+1.1146, +3.9412]`,
  estimated EV/hand `+0.0607`, `43` losses, max loss `29.2270`.

Trajectory component analysis over the 500 fired rows:

- rows:
  `500`
- loss rows:
  `82`
- mean realized delta:
  `+1.9699`
- component coverage:
  all tracked component fields present.
- primary loss components:
  - `royalty`: `31`
  - `fl`: `29`
  - `foul`: `11`
  - `line`: `10`
  - `scoop`: `1`

Replay target postprocess:

- false-positive local replay output:
  `outputs/evals/regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614_false_positive_replay_postprocess/`
- all-fired local replay output:
  `outputs/evals/regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614_all_fired_replay_postprocess/`
- final merged replay target output:
  `outputs/evals/regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614_final_replay_targets/`
- false-positive replay:
  `82` realized-loss rows replayed at MC512.
- all-fired replay:
  `512` deduped fired rows replayed at MC128.
- final merged collection:
  `500` rows, all replay-ready, missing replay fields `0`.

Final risk target audit:

| target type | rows |
|---|---:|
| whole-game non-loss control | 418 |
| whole-game risk only | 81 |
| local EV hard negative | 1 |

Audit interpretation:

- Only `1 / 82` whole-game losses is a local EV hard negative after MC512
  replay. The overwhelming majority of losses are downstream whole-game risk
  rows, not local T2 EV/gate mistakes.
- Do not feed these whole-game losses into safe-LCB/local EV labels. They are
  for a separate risk/counterfactual head.
- The final collection is schema-clean and replay-ready, but still not
  risk-head training-ready because `whole_game_risk_only` rows are below the
  current minimum. More fired/loss rows are needed before training that head.

Interpretation:

- This is the strongest integrated Stage8c TopK result so far under
  hidden-discard/current-FL-EV with Stage3 T3 continuation.
- The primary realized per-fire metric is now positive with CI above zero,
  and both first-seat and second-seat splits are positive with CI above zero.
- This still does not directly authorize production/P2/T1/50k. The runtime is
  expensive TopK+MC+confirm, not a cheap direct model gate. Treat it as strong
  evidence that the integrated `scd1` hypothesis is worth distillation, more
  risk-data collection, selected high-MC audit, or a larger final validation
  before deployment work.

## 11. Current Bottom Line

T3 Stage7_candidate_A m5_r10 is no longer confirmed as the fixed continuation
under standard hidden-discard rules. It remains useful historical/open-discard
evidence, but hidden-discard T2/T1 work should use Stage3 HU margin10 or first
rebuild the T3 selective override under the corrected information model.

T2 is still under active research:

- Stage8/Stage8b contain useful signal.
- Direct runtime override gates have not passed C3.
- Stage8b TopK + MC rerank is the current most promising direction, but only
  has small execution-pass / decision-No-Go validations so far.
- The `10.227` FL EV recheck did not rescue T2 by itself.
- A fresh current-FL-EV Stage8c mixed5k MC16 teacher/cache/training pass now
  exists and is mechanically clean, but its threshold sweep has zero positive
  override configs. Treat it as a compatible training artifact and pipeline
  proof, not as adoption evidence.
- A current-FL-EV Stage8c mixed5k plus 3-seed TopK replay variant also exists.
  It proves replay-enriched cache/training compatibility and passes a tiny
  target10 TopK smoke with exact non-fired cancellation, but its holdout signal
  is mixed and the smoke is too small to prefer it over the base mixed5k model.
- An all-fired 8-seed replay variant was also trained from `400` MC128 replayed
  fired rows. It is mechanically valid, but under the deployable `pd0` guard it
  mostly produces `topk_empty` and fires only `2 / 1200` paired seeds in the
  same-seed smoke. Relaxing to `pd=-2` increases fires but uses negative
  predicted-delta overrides, so it is diagnostic only. Do not prefer it over
  base mixed5k.
- The trainer now supports marking partial replay rows as auxiliary sources so
  they do not contribute to EV/ranking/listwise losses. The first aux-gate
  all-fired replay model is also No-Go for candidate generation (`2 / 1200`
  fires at `pd0`), so the issue is not solved by gate-only replay mixing.
- On two new target50 seeds, plus3seed has higher realized per-fire quality
  than base (`+3.67` vs `+2.66`) but much lower fire rate, so base has the
  better EV/hand (`+0.0379` vs `+0.0227`). Use base mixed5k as the default
  candidate generator for the next multi-seed TopK rerank validation; keep
  plus3seed as a precision-oriented diagnostic candidate.
- Base mixed5k now has an 8-seed / 400-fire first-seat aggregate at
  `+0.0311 EV/hand`, seed mean CI `[+0.0174, +0.0423]`, and per-fire CI
  `[+1.0058, +2.9890]`. This is strong offline TopK rerank evidence, but still
  first-seat-only and too expensive to call production/P2.
- A replay-ready GCP multi-seed target50 run now exists for both seats under
  the hidden-discard/current-FL-EV trajectory-instrumented path. First-seat is
  strong (`+0.0529 EV/hand`, per-fire CI `[+1.8655, +5.1799]` on `150` fires).
  Second-seat is directionally positive but not decision-strong (`+0.0169
  EV/hand`, per-fire CI `[-0.3835, +2.5301]` on `150` fires).
- Post-hoc second-seat runtime risk-filtering on those `150` second-seat fires
  found two validation hypotheses, not deployable settings: broad
  `pd0_cd1_cz1` (`+0.0231 EV/hand`, `115` fires) and stricter tail-risk
  `pd0.5_cd1_cz1` (`+0.0209 EV/hand`, `73` fires, max loss `18.2270`). The
  next validation should keep first-seat unfiltered and test these second-seat
  guards on non-overlapping seeds.
- The non-overlapping second-seat validation has now been run. Broad `pd0_cd1`
  reproduced (`+0.0243 EV/hand`, `+2.1127` per fire, CI `[+0.5069, +3.7186]`,
  150 fires, all three seeds positive). Stricter `pd0.5_cd1` did not
  reproduce (`+0.0035 EV/hand`, per-fire CI crosses zero, one seed negative).
  Keep `pd0_cd1` as the second-seat candidate and drop `pd0.5_cd1` from the
  main path.
- A position-specific integrated validation has now been run with first-seat
  unfiltered and second-seat `scd1`. It is directionally positive (`+0.0369`
  whole-game paired EV/hand, seed-mean CI `[+0.0291, +0.0464]`, 150 realized
  fires, all three seeds positive) with exact non-fired cancellation. However,
  the conditional per-fire CI still crosses zero (`+1.0852`, CI `[-0.5958,
  +2.7661]`), so this is a validation pass for the hypothesis, not a
  production/P2 approval.
- The larger integrated `scd1` validation reached `500` realized fires on
  fresh non-overlapping seeds and upgraded the evidence materially:
  conditional per-fire `+1.9699` with CI `[+1.0378, +2.9020]`, estimated
  EV/hand `+0.0513`, seed-mean whole-game EV/hand CI `[+0.0397, +0.0684]`,
  exact non-fired cancellation, and both first/second seat splits positive.
  This makes Stage8c TopK+confirm the current best T2 path, but still not a
  production/P2 path because it is an expensive search/rerank policy rather
  than a cheap deployable model gate.
- Postprocess of the 500-fire run produced a final replay-ready target
  collection: `418` whole-game non-loss controls, `81` whole-game risk-only
  rows, and only `1` local EV hard negative. This says the current losses are
  mostly downstream risk/counterfactual failures rather than local T2 EV
  mistakes. A future cheap model should keep local EV/gate training separate
  from a whole-game risk head.
- Two additional non-overlapping risk-data runs were generated for the same
  `scd1` config to make the risk-head target set training-ready:
  `riskplus` reached `800` fires (`+0.0656` estimated EV/hand, per-fire CI
  `[+1.7390, +3.2084]`, `112` realized losses, exact non-fired cancellation)
  and `riskplus2` reached `400` fires (`+0.0573` estimated EV/hand, per-fire
  CI `[+0.9451, +3.1252]`, `75` realized losses, exact non-fired
  cancellation). Both are validation/data-collection runs only.
- Combining the 500-fire, riskplus 800-fire, and riskplus2 400-fire replay
  targets produced a clean `1700`-row collection: `1431` whole-game non-loss
  controls, `244` whole-game risk-only rows, and `25` local EV hard negatives.
  Replay readiness is `1700 / 1700`, missing replay fields are `0`, and both
  `risk_head_training_ready` and `local_ev_hard_negative_training_ready` are
  true. This clears the data-readiness blocker for a risk-head smoke, but it
  still does not approve production/P2/T1/50k.
- A first whole-game risk-head smoke was trained on the combined 1700-fire
  collection with `hu_plus_runtime_meta` features and `source_seed` split:
  `244` positives, `1431` negatives, `25` local EV hard negatives excluded.
  The smoke executes and writes
  `models/hu_turn2_stage8c_risk_head_scd1_1700fires_smoke.pt`, but
  generalization is weak so far: train ROC AUC `0.9134`, val ROC AUC `0.5730`,
  test ROC AUC `0.5524`; val/test AP are only `0.1885` / `0.1321`. Treat this
  as a pipeline smoke and evidence that the current risk-head feature/model
  design needs improvement before any runtime filtering validation.
- The risk-head trainer now supports explicit BCE positive-class weighting via
  `--pos-weight-mode auto|none|value`. The audit tool also reports
  `source_run` and `source_seed` groups, so the 500-fire, riskplus, and
  riskplus2 sources can be separated instead of collapsing to `other`.
- A source-seed / runtime-meta / unweighted BCE diagnostic was run:
  `outputs/training/hu_turn2_stage8c_risk_head_scd1_1700fires_diag_source_seed_meta_unweighted/`.
  It improves calibration materially versus auto weighting:
  probability mean `0.1305` vs positive rate `0.1457`, Brier `0.1131` overall,
  val/test Brier `0.1336` / `0.1089`. It also improves validation AP/AUC to
  `0.2440` / `0.6332`, but test AP/AUC remain weak at `0.1893` / `0.5612`.
  The audit still blocks runtime integration with
  `val_auc_lt_0p65`, `test_auc_lt_0p65`, `train_test_auc_gap_gt_0p25`, and
  `holdout_ap_lt_0p25`.
- Interpretation of the 1700-fire risk-head diagnostics: unweighted BCE is the
  better calibration default for this risk target, but current runtime/HU
  features still do not generalize enough to deploy a whole-game risk filter.
  Use the unweighted model as the current diagnostic baseline, not as a runtime
  gate.
- A 1700-fire risk-target gap analysis was also run:
  `outputs/training/hu_turn2_stage8c_risk_target_gap_analysis_1700fires/`.
  It shows the risk target is not mainly a local T2 EV mistake: `178 / 244`
  whole-game risk-only rows are locally positive by replay, `0 / 244` are
  locally negative, and local replay deltas are similar for risk-only losses
  and non-loss controls (`+2.10` vs `+2.12`). Downstream trajectory coverage is
  now mostly present (`13 / 15` tracked fields), with only
  `t3_decision_summary` and `downstream_override_fired` missing everywhere.
  Therefore the current blocker is not missing terminal-component logging; it
  is that deployable runtime features do not yet separate downstream
  whole-game risk well enough.
- The TopK evaluator now writes `candidate_t3_decision_summary`,
  `baseline_t3_decision_summary`, top-level `t3_decision_summary`, and
  `downstream_override_fired` into `runtime_decisions.jsonl`. It also writes
  `stage_a_paired_delta_summary`, `confirm_paired_delta_summary`, and
  `paired_future_delta_summary` so future risk rows have a compact distribution
  summary for the candidate-vs-baseline common-future deltas. The
  counterfactual loss-target builder preserves these fields. Smoke output:
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_paired_future_summary_smoke/`,
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_paired_future_summary_fire_smoke/`,
  and
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_paired_future_summary_loss_targets/`.
  The smoke gap analyzer reports `0 / 16` tracked downstream/risk fields
  missing everywhere, so future newly generated Stage8c risk collections should
  no longer miss the T3 summary, downstream override flag, or paired-future
  delta summary fields.
- A fresh-schema target5 local smoke validated the full instrumentation and
  postprocess path:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target5_scd1/` and
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target5_scd1_postprocess/`.
  The run stopped after `98` paired seeds with `5` realized fires. Its EV/hand
  was `-0.0510` and realized per-fire delta was `-2.0`, but this is only a
  tiny schema validation run, not a performance estimate. The important result
  is coverage: all runtime decisions had `t3_decision_summary` and
  `downstream_override_fired`; all fired rows had `paired_future_delta_summary`;
  and the postprocess gap analyzer found `0 / 16` tracked fields missing
  everywhere. The resulting collection split was `3` whole-game non-loss
  controls, `1` whole-game risk-only row, and `1` local EV hard negative. This
  confirms fresh Stage8c rows can feed risk diagnostics without the previous
  missing-schema blocker, but it does not make risk-head training ready
  (`risk_only_rows_lt_min`, `single_seat_only`) and does not approve T2
  production, P2 fixed status, T1, or 50k teacher generation.
- A larger fresh-schema target50 local run was then executed with the same
  Stage3-reference T3 continuation:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1/` and
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_postprocess/`.
  It reached `50` realized fires after `788` paired seeds (`1,576` hands).
  Non-fired cancellation was exact (`non_fired_nonzero_count = 0`), so the
  conditional metric is based only on fired whole-game paired deltas. The run
  produced `+0.0504 EV/hand`, realized per-fire delta `+1.5891`, and
  confirm-delta mean on fired rows `+2.3977`; however this is still a single
  seed and has very wide per-fire uncertainty (`[-1.6243, +4.8024]`), so it is
  not production evidence. The position split is also unstable (`first`
  `-0.4541 EV/hand`, `second` `+0.5550 EV/hand`). Treat this as a data-path and
  target-collection validation, not a deployment decision.
- The target50 postprocess generated replay-ready risk rows with full schema
  coverage: `50 / 50` replay-ready, `0` missing replay fields, and `0 / 16`
  tracked downstream/risk fields missing everywhere. The collection split was
  `41` whole-game non-loss controls, `6` whole-game risk-only rows, and `3`
  local EV hard negatives. Risk-head training remains No-Go because
  `risk_only_rows_lt_min`; local EV hard-negative training remains No-Go because
  `local_ev_hard_negatives_lt_min`. The important next scaling target is not
  new columns, but more fresh fired rows with this instrumentation, preferably
  non-overlapping seeds and balanced first/second coverage.
- A second non-overlapping target50 seed was added:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_seed2026071302/`.
  It reached the target after `1,001` paired seeds (`2,002` hands), with
  `50` realized fires, exact non-fired cancellation, and realized per-fire
  delta `+3.0472`. The two-seed aggregate is:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_2seed_postprocess/`.
  Aggregate stats: `1,789` paired seeds, `100` realized fires, weighted EV/hand
  `+0.0665`, conditional EV/hand estimate `+0.0674`, realized per-fire delta
  `+2.3182` with CI `[+0.2340, +4.4024]`, `14` realized-loss fires, max loss
  `36.2270`, and `non_fired_nonzero_count = 0`. Replay coverage stayed clean:
  `3,578 / 3,578` decisions replay-ready, `100 / 100` fired rows replay-ready,
  no missing replay fields, and `0 / 16` tracked downstream/risk fields missing
  everywhere.
- The two-seed loss-target collection has `100` deduped fired rows:
  `86` whole-game non-loss controls, `10` whole-game risk-only rows, and `4`
  local EV hard negatives. This is useful evidence that the instrumentation and
  postprocess path can scale beyond a smoke test. It is still not enough for
  risk-head training (`risk_only_rows_lt_min`, `local_ev_hard_negatives_lt_min`)
  and still does not approve production, P2 fixed status, T1, or 50k teacher.
- Two more non-overlapping target50 seeds were added:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_seed2026071303/`
  and
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_seed2026071304/`.
  The four-seed aggregate is:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_4seed_postprocess/`.
  Aggregate stats: `3,488` paired seeds, `200` realized fires, weighted EV/hand
  `+0.0790`, seed-mean EV/hand `+0.0796` with seed CI
  `[+0.0352, +0.1240]`, conditional EV/hand estimate `+0.0797`, realized
  per-fire delta `+2.7243` with CI `[+1.2598, +4.1888]`, `32` realized-loss
  fires, max loss `36.2270`, and `non_fired_nonzero_count = 0`. Replay coverage
  stayed clean again: `6,976 / 6,976` decisions replay-ready, `200 / 200`
  fired rows replay-ready, no missing replay fields, and `0 / 16` tracked
  downstream/risk fields missing everywhere.
- The four-seed loss-target collection has `200` deduped fired rows:
  `168` whole-game non-loss controls, `25` whole-game risk-only rows, and `7`
  local EV hard negatives. This is the strongest current-FL-EV Stage8c TopK
  per-fire validation artifact so far, and it confirms that the measured gain
  is coming only from fired paired deltas rather than non-fired noise. It still
  does not make the risk head trainable (`risk_only_rows_lt_min`,
  `local_ev_hard_negatives_lt_min`) and still does not approve production, P2
  fixed status, T1, or 50k teacher. Next practical step: continue collecting
  non-overlapping target50 seeds with the same instrumentation until risk-only
  and local-hard-negative counts are large enough, or launch this per-fire
  target collection on Spot VM if local collection becomes the bottleneck.
- Two more non-overlapping target50 seeds were collected:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_seed2026071305/`
  and
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_seed2026071306/`.
  Seed `2026071305` is an important weak seed: it reached `50` fires with
  `-0.0317 EV/hand` and realized per-fire delta `-1.3227`. Seed `2026071306`
  was modestly positive (`+0.0381 EV/hand`, per-fire `+1.4200`). The six-seed
  aggregate is:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_6seed_postprocess/`.
  Aggregate stats: `5,462` paired seeds, `300` realized fires, weighted EV/hand
  `+0.0509`, seed-mean EV/hand `+0.0541` with seed CI
  `[+0.0084, +0.0999]`, conditional EV/hand estimate `+0.0510`, realized
  per-fire delta `+1.8324` with CI `[+0.5453, +3.1195]`, `55` realized-loss
  fires, max loss `36.2270`, and `non_fired_nonzero_count = 0`. Replay coverage
  remained clean: `10,924 / 10,924` decisions replay-ready, `300 / 300` fired
  rows replay-ready, no missing replay fields, and `0 / 16` tracked
  downstream/risk fields missing everywhere.
- The six-seed loss-target collection has `300` deduped fired rows:
  `245` whole-game non-loss controls, `43` whole-game risk-only rows, and `12`
  local EV hard negatives. The additional weak seed lowered the effect estimate
  but did not erase the positive per-fire signal. This is useful validation
  evidence for the TopK+confirm path, not production evidence. Risk-head
  training remains No-Go (`risk_only_rows_lt_min`), and local hard-negative
  training remains No-Go (`local_ev_hard_negatives_lt_min`). Continue collecting
  non-overlapping target50 seeds or move this exact collection to Spot VM before
  attempting another risk-head or production-gate decision.
- `analyze_hu_turn2_stage8c_topk_per_fire` now writes
  `aggregate_manifest.json` with a machine-readable primary-metric gate:
  `cancellation_audit_present`, `cancellation_clean`, and
  `primary_metric_valid`. A representative local six-seed rerun was written to
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_local6_aggregate_manifest_check/`.
  The manifest reports `primary_metric_valid=True`, `cancellation_clean=True`,
  `total_realized_fires=300`, `total_non_fired_nonzero_count=0`,
  best estimated EV/hand `+0.05099`, and per-fire CI low `+0.5453`. This is a
  validation-quality audit artifact only; it still leaves production/P2, T1,
  and 50k teacher as `No-Go`.
- Post-hoc runtime risk-filtering on those same `400` fires found no clear
  free improvement. The best EV guard only moves EV/hand to `+0.0320` and keeps
  the same `39.2270` max loss; p95-loss-constrained guards reduce tail risk but
  drop EV/hand to about `+0.021` to `+0.022`. Treat any such guard as a
  hypothesis requiring non-overlapping validation.
- A non-overlapping target20 check of the best post-hoc rank guard did not
  validate it: base scored `+0.0556 EV/hand` on `40` fires, while `rank10`
  scored `+0.0199 EV/hand` on `40` fires. Keep the default config without
  `rank10`.
- The earlier single-seed second-seat target20 probe was negative
  (`-0.0141 EV/hand`, `-0.9000` per fire on 20 fires), but it has now been
  superseded by the replay-ready multi-seed target50 result above. Treat
  second-seat as unresolved rather than failed: it needs a position-specific
  guard validation, not production adoption.
- Fresh-schema Stage8c `scd1` collection was moved to GCP Spot VM for the same
  deployable TopK+confirm config
  `k5/mc64/d0/se0/confirm128/cse1/scd1/pd0/bygate_delta`, using hidden-discard
  current-FL-EV scoring and `stage3_reference_default` T3 continuation. The
  first GCP run added `6` non-overlapping target50 seeds and passed with
  `300` fires, estimated EV/hand `+0.0733`, realized per-fire `+2.6726` with
  CI `[+1.4781, +3.8670]`, exact non-fired cancellation, and no missing replay
  fields. A second GCP run added `16` more target50 seeds and passed with
  `800` fires, estimated EV/hand `+0.0682`, realized per-fire `+2.4964` with
  CI `[+1.7645, +3.2283]`, exact non-fired cancellation, and no missing replay
  fields. Two shard losses in that run were Spot interruption/missing-status
  events and were safely re-run by shard id.
- Combining the local `6` seeds, the GCP `6` seeds, and the GCP `16` seeds gives
  a `28`-seed validation artifact:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_28seed_aggregate/`.
  It has `26,018` paired seeds, `1,400` realized fires, weighted EV/hand
  `+0.0646`, seed-mean EV/hand CI `[+0.0476, +0.0839]`, realized per-fire
  delta `+2.3919` with CI `[+1.8295, +2.9543]`, `223` realized-loss fires,
  max loss `37.2270`, exact non-fired cancellation (`non_fired_nonzero = 0`),
  and `1,400 / 1,400` fired rows replay-ready. This is the strongest validation
  evidence so far for the expensive Stage8c TopK+confirm policy, but it is
  still a search/rerank policy and not a production/P2 model gate.
- The `28`-seed postprocess produced `1,400` replay-ready fired loss-target
  rows: `1,177` whole-game non-loss controls, `197` whole-game risk-only rows,
  and `26` local EV hard negatives. This made local EV hard-negative training
  ready but missed the whole-game risk-head threshold by only `3` risk-only
  rows. A final small GCP `2`-seed top-up added `100` fires, and merging its
  loss targets with the `28`-seed collection produced:
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_30seed_loss_targets/`
  and
  `outputs/evals/hu_turn2_stage8c_fresh_schema_target50_scd1_30seed_risk_target_audit/`.
  The final `30`-seed target collection has `1,500` deduped fired rows,
  `1,500 / 1,500` replay-ready, `0` missing replay fields, `1,266`
  whole-game non-loss controls, `206` whole-game risk-only rows, and `28`
  local EV hard negatives. Both `risk_head_training_ready` and
  `local_ev_hard_negative_training_ready` are now `true`.
- This clears the data-readiness blocker for the next large Stage8c risk/local
  training pass. It does not approve T2 production, P2 fixed status, T1
  training, or 50k teacher generation. The next model work should train and
  audit a risk/local head using the `30`-seed collection while preserving the
  separation between local EV hard negatives and whole-game risk-only rows.
- A large Stage8c whole-game risk-head training pass was run on the `30`-seed
  collection:
  `outputs/training/hu_turn2_stage8c_risk_head_30seed_unweighted_hu_meta_source_seed/`
  with model
  `models/hu_turn2_stage8c_risk_head_30seed_unweighted_hu_meta_source_seed.pt`.
  It used `hu_plus_runtime_meta`, source-seed split, unweighted BCE, `206`
  positive whole-game risk rows, `1,266` controls, and excluded the `28` local
  EV hard negatives as intended. Execution passed, but the audit is decision
  No-Go: train AP/AUC `0.9470 / 0.9845`, val AP/AUC `0.1539 / 0.5889`, and
  test AP/AUC `0.1802 / 0.5489`. The head overfits source seeds and must not be
  wired into runtime.
- Lower-capacity ablations were also run, including `hu_plus_runtime_meta`
  MLPs, `hu_only`, and a new `runtime_meta_only` feature mode that uses only
  the 13 runtime-available TopK/confirm metadata fields. The comparison artifact
  is:
  `outputs/training/hu_turn2_stage8c_risk_head_30seed_comparison/`.
  None cleared the runtime-integration bar; best test AUC was only about
  `0.616`, and all variants had holdout AP below `0.25`. Current conclusion:
  the `30`-seed target set is sufficient for execution testing and audit
  plumbing, but not enough to train a generalizing whole-game risk filter.
  Collect more diverse fired rows/risk-only rows or redesign the label/features
  before trying another runtime risk gate.
- A separate local-EV-negative target mode was added to
  `src/ofc_regular/train_hu_turn2_stage8c_risk_head.py`. It trains a veto head
  from local replay labels only, keeping whole-game risk-only rows out of the
  local EV label. The best probe so far is:
  `models/hu_turn2_stage8c_local_ev_negative_30seed_local_runtime_h8_auto.pt`
  with output
  `outputs/training/hu_turn2_stage8c_local_ev_negative_30seed_local_runtime_h8_auto/`.
  It uses `runtime_meta_only`, hidden layer `8`, auto positive weight, and
  reached val/test AP-AUC about `0.5873 / 0.4769` and ROC-AUC about
  `0.7903 / 0.8099` for local-EV-negative detection. Treat this as a veto
  hypothesis only, not as production evidence.
- `analyze_hu_turn2_stage8c_risk_head` now records the audit label family in
  `risk_head_audit_manifest.json` as `target_mode` and
  `risk_probability_semantics`. It also separates
  `runtime_integration_ready` from `runtime_integration_approval`. The former
  is only a model-quality screen for whether a fixed-threshold fresh validation
  is worth trying; the latter remains `No-Go` unless a runtime gate has passed
  fresh seat-swap validation. This prevents the local-EV-negative audit
  manifest from being misread as production/runtime approval.
- `evaluate_hu_turn2_stage8b_topk_mc_rerank.py` now supports an explicit
  post-confirm veto:
  `--local-ev-risk-model` and `--local-ev-risk-threshold`. The risk probability
  is computed only after Stage B confirm has passed and is logged as
  `local_ev_risk_probability`; it is not a realized gain metric. If the
  probability is above threshold, the decision falls back to baseline with
  `no_override_reason = local_ev_risk_veto`.
- Offline application of the local-EV veto to the 30-seed fired target set is
  in `outputs/evals/hu_turn2_stage8c_local_ev_risk_veto_offline/`. At threshold
  `0.85`, it vetoes `37 / 1500` fired rows, catches `22 / 211` local-negative
  rows, and incorrectly vetoes `2 / 778` local-positive-LCB rows. This is
  conservative but low recall.
- A real small seat-swap wiring smoke with threshold `0.85` passed execution:
  `outputs/evals/hu_turn2_stage8c_local_ev_risk_veto_target5_threshold085/`.
  Non-fired cancellation stayed exact (`non_fired_nonzero_count = 0`), but
  there were `0` risk vetoes and the first `5` fired decisions had realized
  per-fire delta `-12.4908`. A stricter `pd0.5/g0.5` smoke
  `outputs/evals/hu_turn2_stage8c_local_ev_risk_veto_target5_threshold085_pd05_g05/`
  also passed execution with exact cancellation but stayed negative
  (`-2.8454` per fire on `5` fires). Adding `rank2` did not fix it
  (`-3.6454` per fire on `5` fires).
- The new loss examples are not in the old 30-seed loss-target set. The most
  important observed miss is seed `2026062701000077`, where confirm MC and model
  metadata looked strong but the realized result lost FL value and royalty. Add
  these fresh loss rows to the next high-MC/local replay hard-negative collection
  before relying on the local-EV veto head.
- That follow-up local replay has now been run:
  `outputs/evals/hu_turn2_stage8c_local_ev_risk_veto_rank2_postprocess/`.
  It replayed the `5` fired rows from the `pd0.5/g0.5/rank2` smoke at MC512.
  The collection split was `4` whole-game non-loss controls, `1` whole-game
  risk-only row, and `0` local EV hard negatives. The key miss
  `2026062701000077` remains local-positive by MC512 replay but whole-game
  risk-only by realized seat-swap outcome.
- The new `5` rows were merged with the previous 30-seed collection into
  `outputs/evals/hu_turn2_stage8c_30seed_plus_local_risk_veto_rank2_loss_targets/`
  and audited at
  `outputs/evals/hu_turn2_stage8c_30seed_plus_local_risk_veto_rank2_risk_audit/`.
  The merged set has `1,505` rows: `1,270` whole-game non-loss controls, `207`
  whole-game risk-only rows, and `28` local EV hard negatives. It remains
  replay-ready and both risk-head and local-EV-hard-negative training-ready.
- The updated risk-target-gap artifact is
  `outputs/training/hu_turn2_stage8c_risk_target_gap_30seed_plus_local_veto_rank2/`.
  It confirms the same structural diagnosis: `162 / 207` risk-only rows are
  local-positive, `45 / 207` are local-gray, and `0 / 207` are local-negative.
  Therefore the current tail losses are not mainly local T2 EV errors; a local
  EV veto can only catch a small subset. The next serious fix needs downstream
  trajectory/risk features or a better whole-game risk model, not another
  local-EV-negative threshold sweep.
- A follow-up runtime tail-distribution feature mode was added to
  `src/ofc_regular/train_hu_turn2_stage8c_risk_head.py`:
  `runtime_tail_meta_only` and `hu_plus_runtime_tail_meta`. These add
  post-confirm paired-delta distribution features such as min/p01/p05/p25,
  p95/p99/max, left-tail loss, and z-scaled tail values. Focused tests cover
  JSON-encoded confirm summaries and both feature modes.
- Four small whole-game risk-head ablations were trained on the merged
  `1,505`-row collection. None cleared the runtime-integration bar:
  `runtime_tail_meta_only` h16 unweighted reached val/test AUC
  `0.5346 / 0.5901`, h16 auto-weight reached `0.6267 / 0.5931`, h32x16
  unweighted reached `0.6236 / 0.6100`, and `hu_plus_runtime_tail_meta` h32
  overfit badly with train/test AUC `0.9987 / 0.6240`. The best test AUC moved
  only to about `0.624`, with holdout AP still weak. Runtime risk integration
  remains No-Go.
- Current conclusion after local-EV and tail-meta probes: the observed
  whole-game losses are not explained well enough by local T2 replay EV,
  confirm MC distribution shape, or the existing HU board/action features.
  Do not keep sweeping risk thresholds on this dataset. The next useful step is
  to collect more diverse fired rows with downstream trajectory labels, or to
  design features that explicitly represent how a T2 override changes later
  T3/final-turn/FL paths.
- The risk-target gap analyzer now also writes downstream trajectory component
  diagnostics:
  `outputs/training/hu_turn2_stage8c_downstream_trajectory_gap_30seed_plus_local_veto_rank2/`.
  Realized-after-the-fact trajectory fields explain the risk labels clearly:
  `royalty_delta_vs_baseline` and `hero_royalty_vs_baseline` have diagnostic
  AUC about `0.952` with low values meaning risk, `fl_delta_vs_baseline` /
  `hero_fl_value_vs_baseline` have AUC about `0.715`, and
  `foul_delta_vs_baseline` has AUC about `0.742`. Mean gaps also point to the
  same story: risk-only rows average `-5.42` royalty vs `+2.09` for controls,
  `-3.56` FL value vs `+1.30`, and `-2.09` foul delta vs `+1.27`.
- These trajectory fields are post-outcome diagnostics, not deployable runtime
  gate inputs. They show what went wrong: T2 overrides that look locally
  positive often give up later baseline royalty/FL/foul outcomes. The risk head
  must predict this opportunity cost before the future is known.
- A baseline-comparative HU feature mode was added to
  `train_hu_turn2_stage8c_risk_head.py`: `hu_delta_only` and
  `hu_delta_plus_runtime_tail_meta`, where `hu_delta = candidate_HU_features -
  baseline_HU_features`. This directly exposes the baseline action comparison
  that the previous candidate-only risk head lacked. Small ablations on the
  same `1,505` rows still failed source-seed holdout:
  `hu_delta_only` val/test AUC `0.4374 / 0.4818`,
  `hu_delta_plus_runtime_tail_meta` unweighted `0.5178 / 0.5642`, and
  auto-weighted `0.5432 / 0.5137`. Runtime integration remains No-Go.
- Updated interpretation: the post-outcome trajectory labels are highly
  explanatory, but current deployable pre-outcome features still do not
  generalize. The next attempt should not be another small threshold or MLP
  sweep. It should either collect substantially more diverse fired rows, or add
  explicit pre-outcome proxy features for baseline opportunity cost such as
  row-specific royalty/FL/foul potential deltas and downstream policy path
  summaries computed at T2.
- A first explicit pre-outcome opportunity-cost proxy was then added to
  `train_hu_turn2_stage8c_risk_head.py`: `opportunity_proxy_only` and
  `opportunity_proxy_plus_runtime_tail_meta`. These features use only T2-visible
  state/action information and compare candidate vs baseline after-boards on
  top FL potential, row royalty potential, partial flush/straight/full-house/
  quads potential, made royalty, and coarse inevitable-foul proxies.
- Training comparison artifact:
  `outputs/training/hu_turn2_stage8c_risk_head_opportunity_1505_comparison.csv`.
  On the same source-seed split, `opportunity_proxy_only` reached val/test AUC
  `0.6158 / 0.5609`; `opportunity_proxy_plus_runtime_tail_meta` unweighted
  improved to `0.6421 / 0.6206`; auto-weighted fell to `0.4917 / 0.5568`.
  This is the best deployable-feature probe so far, but it still misses the
  audit threshold (`val_auc_lt_0p65`, `test_auc_lt_0p65`) and fires too few
  high-risk vetoes at useful precision. Runtime integration remains No-Go.
- The opportunity proxy direction is plausible because it moves toward the
  observed post-outcome failure modes, but the current 1,505-row dataset and
  simple MLP are still insufficient. The next useful iteration should either
  (a) expand the fired-row collection with balanced source/seat coverage and
  keep these opportunity features, or (b) compute stronger T2-time proxy
  features with short lookahead for baseline/candidate FL/royalty/foul
  opportunity cost. Do not promote this risk head into the TopK runtime yet.
- A short-lookahead proxy feature path was then added to
  `train_hu_turn2_stage8c_risk_head.py`: `lookahead_proxy_only`,
  `lookahead_proxy_plus_runtime_tail_meta`, and
  `opportunity_lookahead_proxy_plus_runtime_tail_meta`. The current spike uses
  deterministic T2-time samples with `LOOKAHEAD_SAMPLES = 4` and
  `LOOKAHEAD_FINAL_SAMPLES = 4` to compare baseline and candidate after-boards
  by shallow continuation score summaries and non-bust-rate summaries. This is
  runtime-feasible only as an expensive diagnostic/proxy path, not as a
  production gate by itself.
- Lookahead comparison artifact:
  `outputs/training/hu_turn2_stage8c_risk_head_lookahead_1505_comparison.csv`.
  Pre-canonicalization comparison on the same `1,505`-row source-seed split:
  `lookahead_proxy_plus_runtime_tail_meta` h16 reached val/test AUC
  `0.6017 / 0.6062`, `opportunity_lookahead_proxy_plus_runtime_tail_meta` h16
  reached `0.6275 / 0.6279`, and a smaller h8 version of the combined feature
  set reached `0.5513 / 0.6337` with validation AP only `0.1464`.
- The lookahead proxy seed path was then hardened so equivalent card sets do not
  depend on JSON/list card order. Focused tests now cover finite materialization,
  card-order stability, and safe zero fallback when an after-board is not the
  expected T3-sized `9`-card board.
- Full canonicalized-feature rerun:
  `outputs/training/hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_1505_h16_canonical/`
  with audit output in the matching `_audit` directory. It trained on `1,477`
  rows, produced feature dim `212`, and completed training plus audit in
  `244.4s` training time. Audit result: val/test AUC `0.6201 / 0.6489`,
  val/test AP `0.2633 / 0.1890`; decision remains No-Go
  (`val_auc_lt_0p65`, `test_auc_lt_0p65`).
- The risk-head audit now reports explicit veto utility for threshold sweeps:
  selected rows are assumed to be vetoed back to baseline, and utility is
  `prevented realized loss - forfeited realized gain`. For the canonical h16
  lookahead model, this confirms the No-Go: common holdout thresholds have
  negative net utility. On test, thresholds `0.30 / 0.35 / 0.40 / 0.45 / 0.50`
  produce net veto gain per row `-0.5068 / -0.2589 / -0.0913 / -0.2751 /
  -0.0660`; on validation the only positive checked threshold is `0.50`
  at `+0.0863` per row, while lower thresholds are negative. Do not select a
  veto threshold from the all-split or train-influenced positive totals.
- The audit also writes `risk_head_threshold_group_sweep.csv` with
  `runtime_group_available`, so source/label-only diagnostic groups are not
  confused with deployable runtime conditions. Among runtime-available groups
  (`seat`, `config_id`, `candidate_rank_bucket`), the only mildly interesting
  holdout signal is `candidate_rank_bucket = rank_4_5` at threshold `0.35`:
  validation selected `6` rows with net `+0.5323` per row and test selected `3`
  rows with net `+0.1633` per row. This is too few rows for a decision, but it
  is a reasonable diagnostic hypothesis for a future fresh heldout run.
- A stricter stable-candidate artifact was added:
  `risk_head_runtime_group_candidates.csv`. With
  `runtime_group_min_selected = 10`, there are no stable runtime group
  candidates. The same `rank_4_5 @ 0.35` subset is marked only as
  `near_miss_positive_but_underpowered = 1`, with `min_selected = 3` and
  `stable_runtime_candidate = 0`. With `runtime_group_target_selected = 50`,
  the observed minimum holdout fire rate `0.09375` implies roughly `534` rows
  per fresh holdout split would be needed to get `50` selected rows for this
  exact hypothesis, before accounting for seed/source variance. Source-seed
  diversity for this near-miss is not obviously a one-seed artifact: validation
  selected `6` rows across `4` source seeds, and test selected `3` rows across
  `3` source seeds. It is still underpowered, not stable. The candidate artifact
  now also reports selected-row veto-gain SE and LCB95; for this near-miss the
  point means are positive but the LCB95 values are strongly negative
  (`val -6.2951`, `test -14.0933`), which confirms that the apparent signal is
  not statistically stable.
- The risk-head audit summary now separates runtime-available group diagnostics
  from diagnostic-only groups. `recommended_training_use`, `source_seed`, and
  similar rows can still explain where losses came from, but they are not
  deployable runtime gates. Read `Runtime-Available Group Candidate Check` and
  `Stable Runtime Group Candidates` first when deciding whether any runtime
  hypothesis deserves a fresh heldout run.
- The audit manifest now includes `runtime_group_candidate_decision`. For the
  canonical h16 lookahead run it is `Fresh-Heldout-Hypothesis`, with
  `stable_runtime_candidate_count = 0`,
  `near_miss_positive_but_underpowered_count = 1`, and best near-miss
  `candidate_rank_bucket/rank_4_5 @ 0.35`. This is explicitly not a runtime
  approval.
- The audit now also writes a dedicated fresh-heldout planning artifact:
  `risk_head_fresh_heldout_plan.csv` and `risk_head_fresh_heldout_plan.md`.
  For the canonical run the only planned runtime gate is
  `risk_probability >= 0.35 and 4 <= candidate_ev_rank <= 5`, with an estimated
  `534` rows per fresh holdout split to reach `50` selected rows. The plan
  repeats that production/P2, 50k teacher, and T1 are all `No-Go`, and that the
  metric must be fresh-heldout realized veto utility, not training/cache LCB.
- The TopK per-fire evaluator can now execute that gate directly with
  `--stage8c-risk-model`, `--stage8c-risk-threshold`,
  `--stage8c-risk-rank-min`, and `--stage8c-risk-rank-max`. The old
  `--local-ev-risk-model` flags remain backward-compatible, but the preferred
  wording for this work is Stage8c risk veto. Use:
  `--stage8c-risk-model models/hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_1505_h16_canonical.pt`
  `--stage8c-risk-threshold 0.35 --stage8c-risk-rank-min 4 --stage8c-risk-rank-max 5`.
  A 1-game CPU smoke wrote
  `outputs/evals/hu_turn2_stage8c_fresh_heldout_plan_smoke/` and confirmed the
  canonical risk head loads, the runtime row includes board/action/dead-card
  features, and the rank guard is logged. This smoke is execution evidence only,
  not performance evidence.
- `scripts/Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1` now accepts the same
  Stage8c risk gate arguments and packages the risk-head model for Spot VM
  workers. The GCP runner still rejects ambiguous seat configs by design, so
  fresh-heldout Spot runs should be launched as explicit seat-split jobs:
  `seat=first` normally, and `seat=second` only with `-AllowRiskDataSeatScope`.
  Both DryRuns were checked with the canonical risk gate and recorded
  `stage8c_risk_model`, `stage8c_risk_threshold`, `stage8c_risk_rank_min`, and
  `stage8c_risk_rank_max` in the dry-run JSON.
- Fresh-heldout Spot VM runs were started for both seats on 2026-06-15:
  `regular-hu-t2-stage8c-fresh-heldout-rank45-risk035-first-20260615` and
  `regular-hu-t2-stage8c-fresh-heldout-rank45-risk035-second-20260615`. Each run
  has `3` shards, `3500` games/seed, target `50` realized overrides/seed,
  `stage3_reference_default` T3 continuation, and the canonical Stage8c risk
  gate (`threshold=0.35`, rank `4..5`). Initial status check showed all six
  Spot instances running, `0/3` completed for each run, and no failures yet.
  Monitor with
  `.\scripts\Get-GcpHuTurn2Stage8cTopkPerFireRunStatus.ps1 -RunName <run>`.
  Receive/aggregate with
  `.\scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -RunName <run>`.
- Fresh-heldout Spot VM runs completed and were received/aggregated on
  2026-06-15. The second-seat run had one early Spot preemption on shard `0`;
  that shard was re-run with the same `RunName` and `-StartShards 0`. All
  first/second shards now have `DONE`, and no matching VM instances remain.
  Aggregates:
  `outputs/evals/hu_turn2_stage8c_fresh_heldout_rank45_risk035_first/` and
  `outputs/evals/hu_turn2_stage8c_fresh_heldout_rank45_risk035_second/`.
- Fresh-heldout first-seat result for the canonical rank `4..5` risk gate:
  `150` realized overrides, non-fired nonzero count `0`, per-fire delta
  `+2.8285` with CI95 `[+1.1742, +4.4827]`, and EV/hand `+0.0420` with CI95
  `[+0.0174, +0.0665]`. Seed-level EV/hand was positive on all three seeds.
- Fresh-heldout second-seat result for the same gate: `150` realized overrides,
  non-fired nonzero count `0`, per-fire delta `+2.1639` with CI95
  `[+0.4600, +3.8679]`, and EV/hand `+0.0364` with CI95
  `[+0.0077, +0.0650]`. Seed-level EV/hand was positive on all three seeds.
- Combined first+second fresh-heldout result: `19,034` decisions, `300` fired
  overrides, override rate `1.576%`, non-fired nonzero count `0`, per-fire
  delta `+2.4962` with CI95 `[+1.3102, +3.6822]`, and EV/hand `+0.0393` with
  CI95 `[+0.0206, +0.0580]`. This is the first clean positive fresh-heldout
  evidence for the Stage8c TopK + confirm-MC + rank `4..5` risk-gated path.
  It remains validation evidence only: production/P2 fixed, 50k teacher, and
  T1 are still `No-Go` until a larger C4-style validation and tail-loss audit
  confirm stability.
- C4-style larger validation was launched and completed on 2026-06-15:
  `regular-hu-t2-stage8c-c4-rank45-risk035-first-20260615` and
  `regular-hu-t2-stage8c-c4-rank45-risk035-second-20260615`. The run used
  `5` seeds per seat, target `100` realized overrides per seed, `8000`
  games/seed cap, `stage3_reference_default` T3 continuation, and the same
  Stage8c risk gate (`threshold=0.35`, rank `4..5`). A few Spot preemptions
  occurred; missing shards were rerun by `-StartShards` in different zones.
  All shards completed and no matching VM instances remain.
- C4 artifacts:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_first/`,
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_second/`, and
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_combined/`.
  Combined C4 result: `66,756` decisions, `1,000` fired overrides, override
  rate `1.498%`, non-fired nonzero count `0`, per-fire delta `+1.8362` with
  CI95 `[+1.2121, +2.4602]`, and EV/hand `+0.0275` with CI95
  `[+0.0182, +0.0369]`. First seat was positive but weaker (`+0.0180`
  EV/hand, CI95 `[+0.0055, +0.0306]`); second seat was stronger (`+0.0376`
  EV/hand, CI95 `[+0.0237, +0.0514]`). C4 is positive validation evidence, not
  production approval. Tail losses remain material (`combined` max loss
  `35.2270`, p95 loss `29.2270`), so the next step should be tail-loss audit
  and possibly selected high-MC replay before P2/production or T1.
- C4 tail-loss audit artifacts:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_hard_negatives/` and
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_tail_audit/`.
  The realized fired rows produced `157 / 1,000` whole-game false positives
  with mean realized delta `-11.1707`, but their confirm-MC mean was still
  positive (`+2.3241`). Therefore confirm-MC deltas must remain gate
  diagnostics only, not performance estimates. The rank breakdown also showed
  that tail loss is not isolated to the rank `4..5` risk-scored subset:
  rank `1..3` fires account for most of the loss mass. The current risk head
  only evaluates rank `4..5`, so future safety work needs a broader T2-time
  risk/opportunity-cost guard or more high-MC hard-negative supervision.
- The top `100` realized C4 false positives were replayed with independent
  MC512 under `stage3_reference_default`:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_tail_replay_mc512_top100/`.
  Result: `100 / 100` replay rows OK, mean local replay delta `+1.9788`,
  `69` LCB196-positive rows, `21` gray rows, and only `10` confirmed local EV
  hard negatives. This means most large realized losses were variance/whole-game
  risk cases, not local negative-EV T2 choices. Only
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_tail_audit/confirmed_hard_negative_mc512_teacher.jsonl`
  should be used as confirmed MC512 hard-negative supervision from this audit.
- The replay tool now supports chunk-safe `--offset` plus `--limit` so large
  all-fired replay can be split across local chunks or Spot VM shards without
  changing deterministic replay seeds. The global input row index is preserved
  in each chunk. Verified smoke:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_all_fired_replay_offset10_limit2_mc8_smoke/`
  matched rows `10` and `11` from the unchunked
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_all_fired_replay_limit12_mc8_parity/`
  run exactly on `replay_seed` and `delta_for_label`.
- A local all-fired MC128 probe on the first `20` rows wrote
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_all_fired_replay_offset0_limit20_mc128_probe/`.
  It completed in `30.6s`, found `1` hard-negative label, `11` LCB196-positive
  rows, and `8` gray rows. Scaling this linearly gives about `102` minutes for
  all `1,000` fired rows at MC512 on the current local worker, or about
  `10` minutes per `100`-row MC512 chunk. Use chunks or Spot VM for the full
  all-fired replay; do not run it as one fragile monolithic local job.
- The same first `20` rows were then replayed at MC512:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_all_fired_replay_offset0_limit20_mc512_probe/`.
  It completed in `110.0s`, found `1` hard-negative label, `15`
  LCB196-positive rows, and `4` gray rows. Compared with MC128, hard-negative
  labels matched on `18 / 20` rows and LCB196 bucket labels matched on
  `14 / 20` rows; the flips were boundary negative/gray or gray/positive
  cases. Use MC512+ for training labels when possible. MC128 is useful for
  speed probes, not final local-EV label assignment.
- A standalone chunk runner was added for this exact workflow:
  `scripts/Run-HuTurn2Stage8cReplayChunks.ps1`. It takes a replay-pack JSONL
  such as `topk_all_fired_deduped.jsonl`, splits by `StartOffset` /
  `ChunkSize`, runs `replay_hu_turn2_stage8b_topk_hard_negatives` with
  chunk-safe global row indexes, and merges per-chunk
  `topk_hard_negative_replay_summary.csv` plus teacher JSONL into a `merged/`
  directory. Dry-run for the full C4 all-fired pack produced `10` chunks of
  `100` rows each at MC512.
- The loss-target builder now also accepts the replay-pack schema
  `hu_turn2_stage8b_topk_hard_negative_v1` directly as `--decision-log`. This
  means original runtime decision logs are not required after the self-contained
  replay pack has been built. Verified smoke:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_replay_chunks_mc4_smoke/`
  ran `2` chunks x `2` rows at MC4, merged `4` replay rows, built
  `loss_targets_from_pack/` from the merged summary plus
  `topk_all_fired_deduped.jsonl`, and collected `1,000` target skeleton rows.
  Because only `4 / 1,000` rows were replayed in the smoke, most rows correctly
  stayed `missing` / `requires_local_replay`. Once all MC512 chunks finish, run
  the same loss-target command against the merged MC512 summary.
- C4 counterfactual target preparation was run from the same C4 logs plus the
  MC512 top100 replay summary:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_counterfactual_targets_mc512_top100/`.
  It produced `157` realized-loss target rows: `10` local EV hard negatives,
  `90` whole-game-risk-only rows, and `57` rows still requiring local replay.
  The audit output
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_counterfactual_targets_mc512_top100_audit/`
  now correctly separates readiness by target type: whole-game risk-head
  training remains No-Go (`risk_only_rows_lt_min`,
  `missing_non_loss_control_rows`), while local EV negative-head smoke is
  allowed because the local-trainable subset has `10` confirmed negatives and
  `69` local-positive controls with replay metadata.
- A local EV negative-head training smoke was run:
  `outputs/training/hu_turn2_stage8c_local_ev_negative_c4_top100_mc512_smoke/`
  with model
  `models/hu_turn2_stage8c_local_ev_negative_c4_top100_mc512_smoke.pt`.
  It materialized `79` trainable rows (`10` positive local EV negatives,
  `69` local-positive controls) using
  `opportunity_lookahead_proxy_plus_runtime_tail_meta`. The run proves the
  confirmed MC512 hard negatives can flow through the existing Stage8c trainer,
  but the dataset is too small for runtime or production conclusions: holdout
  splits have only `1` positive each, and the head is a smoke artifact only.
- The Stage8c risk-head trainer CLI was cleaned up after the smoke: repeated
  `--threshold` arguments now replace, rather than append to, the default
  report thresholds. A dedup smoke using explicit thresholds `0.42` and `0.84`
  wrote only those two thresholds per split at
  `outputs/training/hu_turn2_stage8c_threshold_dedup_smoke/`. Older
  threshold-metrics CSVs from before this fix may contain duplicated default
  thresholds when custom thresholds were also supplied.
- Full C4 all-fired independent local replay was then moved to GCP Spot:
  `regular-hu-t2-stage8c-c4-allfired-mc512-replay-20260615`.
  It replayed all `1,000` rows from
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_hard_negatives/topk_all_fired_deduped.jsonl`
  at MC512 with `10` shards x `100` rows. One early shard VM disappeared without
  status and was safely re-run by shard id; final status was `10 / 10` complete,
  `0` failed, and no running instances. Received output:
  `outputs/evals/hu_turn2_stage8c_c4_allfired_mc512_replay_gcp_received/`.
  The merged summary has `1,000` replay rows and `0` missing replay fields.
- The C4 all-fired MC512 target split is:
  `843` whole-game non-loss controls, `138` whole-game risk-only rows, and
  `19` local EV hard negatives. All rows are replay-ready and first/second seats
  are balanced (`500 / 500`), but this artifact alone is not training-ready:
  `risk_only_rows_lt_min` and `local_ev_hard_negatives_lt_min`.
- Combining the older `30`-seed collection with the C4 all-fired MC512 targets
  produced
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_plus_30seed_loss_targets/`
  with `2,505` deduped rows, `2,113` non-loss controls, `345` whole-game
  risk-only rows, `47` local EV hard negatives, and `0` rows requiring local
  replay. Audit output:
  `outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_plus_30seed_risk_target_audit/`.
  Both `risk_head_training_ready` and `local_ev_hard_negative_training_ready`
  are `true`.
- A larger whole-game risk-head rerun on the `2,505` rows used the previous best
  deployable feature family,
  `opportunity_lookahead_proxy_plus_runtime_tail_meta`, h16, source-seed split,
  and unweighted BCE:
  `outputs/training/hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_2505_h16_canonical/`
  with model
  `models/hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_2505_h16_canonical.pt`.
  Execution passed, but the audit remains decision No-Go:
  val/test AUC `0.5701 / 0.5841`, val/test AP `0.2085 / 0.1891`, no stable
  runtime group candidates, and negative test veto utility at common thresholds.
  Adding the C4 all-fired data did not make the whole-game risk head deployable.
- A separate local-EV-negative update on the same `2,505` rows used
  `runtime_meta_only`, h8, source-seed split, and auto positive weight:
  `outputs/training/hu_turn2_stage8c_local_ev_negative_2505_runtime_h8_auto/`
  with model
  `models/hu_turn2_stage8c_local_ev_negative_2505_runtime_h8_auto.pt`.
  This local replay label task is healthier than the whole-game risk task
  (val/test ROC-AUC `0.7000 / 0.7298`, AP `0.3770 / 0.3821`), but threshold
  `0.5` still has only about `0.37-0.39` precision on holdout and thresholds
  `0.7+` fire zero rows. Treat it as a veto hypothesis only, not runtime
  integration evidence.
- The Stage8c risk-head audit now also writes raw runtime-score veto sweeps:
  `risk_head_raw_score_threshold_sweep.csv` and
  `risk_head_raw_score_runtime_candidates.csv`. These use train-split quantile
  thresholds for scores such as `confirm_delta`, `gate_probability`,
  `negative_predicted_delta`, and candidate rank, then apply the fixed threshold
  to val/test. Re-audit output:
  `outputs/training/hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_2505_h16_canonical_audit_rawscore/`.
  Result: no stable raw-score runtime candidate and no near-miss raw-score
  candidate. Some test-only gates look positive, for example
  `gate_probability` q0.98 with `14` selected and net `+0.1936` per row, but the
  matching val split is non-positive. This rules out the cheap fix of replacing
  the weak risk head with a simple raw confirm/topk score threshold.
- The `2,505`-row collection was rechecked with the risk-target gap and
  trajectory-component analyzers:
  `outputs/training/hu_turn2_stage8c_risk_target_gap_2505_plus_c4_rank45/` and
  `outputs/training/hu_turn2_stage8c_trajectory_components_2505_plus_c4_rank45/`.
  The main target gap remains: `345` whole-game risk-only rows, `271` of those
  locally positive, `0` locally negative, and `292` non-loss controls that are
  locally negative. The realized loss path is still downstream: strongest
  post-outcome signals are `terminal_score_vs_baseline` AUC `0.8442`,
  `royalty_delta_vs_baseline` AUC `0.8039`, `foul_delta_vs_baseline` AUC
  `0.6543`, and `fl_delta_vs_baseline` AUC `0.6345`. These are explanatory
  labels after the hand resolves, not deployable T2-time gate inputs.
- Coverage caveat for that `2,505`-row analysis: the older `30`-seed collection
  has complete downstream trajectory fields (`1,505 / 1,505`), but the C4
  all-fired MC512 replay rows have only local replay/target metadata (`0 /
  1,000` complete downstream rows). The C4 rows are valid for local-EV
  hard-negative and whole-game risk-head labels, but they should not be used as
  evidence about foul/FL/royalty/scoop component causes unless rerun with
  trajectory instrumentation.
- `analyze_hu_turn2_stage8c_trajectory_components` now excludes incomplete
  rows from component metrics instead of treating missing component fields as
  zero. On the combined `2,505`-row collection, the trajectory component
  analysis therefore uses only `1,505` complete rows, excludes the `1,000` C4
  local-replay-only rows, and reports `235` loss rows. Among those complete
  rows, the primary loss components are `royalty=119`, `fl=50`, `foul=41`,
  `line=14`, and `scoop=11`; the previous `unexplained` bucket was an artifact
  of mixing incomplete C4 rows into component analysis.
- To prevent this confusion from recurring,
  `prepare_hu_turn2_stage8b_counterfactual_loss_targets` now writes
  `downstream_trajectory_complete`,
  `downstream_trajectory_present_fields`,
  `downstream_trajectory_total_fields`, and
  `topk_counterfactual_downstream_coverage.csv`. The C4 all-fired MC512
  loss-target summary now explicitly reports `0 / 1000` complete downstream
  trajectory rows. The combined collection summary now reports `1,505 / 2,505`
  complete downstream rows, and the risk-target audit keeps
  `risk_head_training_ready=True` separate from
  `trajectory_component_analysis_ready=False`. The corresponding manifests now
  carry the same audit fields (`downstream_trajectory_complete_rows`,
  `downstream_trajectory_incomplete_rows`,
  `downstream_trajectory_complete_rate/share`,
  `trajectory_component_complete_rows`, and
  `trajectory_component_incomplete_rows`), so downstream automation no longer
  has to open the CSV summaries to know that the combined collection is only
  partially trajectory-complete. `Run-HuTurn2Stage8cReplayReadyPostprocess.ps1` and
  `Receive-GcpHuTurn2Stage8cReplayChunksRun.ps1` also run
  `analyze_hu_turn2_stage8c_risk_target_gap` by default after collection, so
  future local/GCP postprocess runs leave the same coverage audit next to the
  risk-target readiness audit.
- The `downstream_trajectory_*` fields are audit metadata only. They must not
  become runtime risk-head features. `train_hu_turn2_stage8c_risk_head` now
  writes explicit `feature_column_names` to the training manifest, and the
  focused training tests verify that changing `downstream_trajectory_complete`,
  `downstream_trajectory_present_fields`, or
  `downstream_trajectory_total_fields` does not change any runtime feature
  vector. A CPU one-epoch `runtime_meta_only` smoke at
  `outputs/training/hu_turn2_stage8c_manifest_feature_names_smoke/` confirmed
  the manifest feature list contains only runtime meta columns.
- Updated risk-head conclusion: shallow lookahead adds a little signal, but it
  does not solve the whole-game risk target even after expanding to `2,505`
  replay-ready fired rows. The risk problem is still underdetermined by the
  available fired-row sample and deployable T2-time features. Do not integrate
  this risk head into Stage8c TopK runtime, do not use it to approve T2/P2, and
  do not start T1 or 50k teacher generation from it.
- A local-EV-negative veto runtime wiring smoke was run with the
  `30`-seed local-EV head:
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_local_ev_veto_rank45_t060_target1_smoke/`
  and aggregate output
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_local_ev_veto_rank45_t060_target1_smoke_aggregate/`.
  The config was
  `k5_mc64_d0_se0_confirm128_cse1_scd1_pd0_bygate_delta` with
  risk threshold `0.60` and candidate-rank guard `4..5`. Runtime loading,
  risk evaluation, rank guard, cancellation audit, and per-fire aggregation all
  executed. The single fired decision had risk probability `0.5224`, so it was
  not vetoed by threshold `0.60`; realized fired delta was `-2.0`. The aggregate
  manifest reports `primary_metric_valid=True`, `cancellation_clean=True`,
  `total_realized_fires=1`, `total_non_fired_nonzero_count=0`, and
  `best_estimated_ev_per_hand=-1.0`. This is only a wiring/cancellation smoke,
  not performance evidence and not runtime integration approval.
- A slightly larger local probe of the same rank45/threshold0.60 veto used
  `target_realized_overrides_per_seed=10`:
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_local_ev_veto_rank45_t060_target10_probe/`
  with aggregate output
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_local_ev_veto_rank45_t060_target10_probe_aggregate/`.
  It reached `10` realized fires after `176` paired seeds. Cancellation remained
  clean (`non_fired_nonzero_count=0`), but the risk model was evaluated only
  once and vetoed `0` rows. Realized per-fire delta was `-2.6227` with a very
  wide CI, so this is not a performance result. The important conclusion is
  operational: sizing this validation by realized overrides does not collect
  enough veto-selected rows.
- To support the intended fixed-threshold veto validation, the TopK evaluator
  and local/GCP runners now accept `--target-risk-vetoes-per-seed`
  / `TargetRiskVetoesPerSeed`. This is the correct stop target when validating
  a risk veto; realized override count remains useful for TopK per-fire
  validation but is the wrong size metric for a veto-only gate. A parser/stop
  smoke at
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_target_risk_veto_stop_smoke/`
  used a synthetic threshold `0.0`, stopped after `1` risk veto in `7` paired
  seeds, and wrote `target_risk_vetoes_reached_count=1`. A GCP DryRun with
  `TargetRealizedOverridesPerSeed=0`, `TargetRiskVetoesPerSeed=50`,
  threshold `0.60`, and rank guard `4..5` passed and emitted
  `target_risk_vetoes_per_seed=50` in the dry-run manifest. This still does not
  approve the rank45/threshold0.60 veto; it only makes the next fresh-heldout
  measurement correctly sizable.
- A second local probe showed why actual veto mode is insufficient for
  measuring veto utility: if the candidate is actually vetoed, the realized hand
  follows the baseline action and the would-be candidate delta is not observed.
  The evaluator and local/GCP runners therefore also support
  `--stage8c-risk-audit-only` / `Stage8cRiskAuditOnly`. In this mode the risk
  model scores and logs `local_ev_risk_would_veto=True`, but the candidate is
  still played so the realized seat-swap counterfactual delta can estimate veto
  utility.
- Audit-only smoke:
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_local_ev_veto_rank45_t030_audit_only_target2_probe/`
  with aggregate output
  `outputs/hidden_discard_smoke/hu_turn2_stage8c_local_ev_veto_rank45_t030_audit_only_target2_probe_aggregate/`.
  Config: threshold `0.30`, rank guard `4..5`,
  `target_risk_vetoes_per_seed=2`, `Stage8cRiskAuditOnly=True`. It reached
  `2` would-veto rows after `129` paired seeds.
  `risk_veto_candidate_metrics.csv` reports realized would-veto candidate
  delta mean `+5.0`, so the implied veto utility was `-5.0` per veto in this
  tiny sample. This is not performance evidence, but it proves the measurement
  path now separates would-veto audit rows from actual vetoed runtime rows.
- The next fresh-heldout risk-veto validation should use audit-only mode, not
  actual veto mode: `TargetRealizedOverridesPerSeed=0`,
  `TargetRiskVetoesPerSeed=50`, `Stage8cRiskAuditOnly`, then read
  `risk_veto_candidate_metrics.csv` as the primary veto-utility estimate. The
  ordinary TopK per-fire metrics remain useful but are not the risk-veto
  adoption metric.
- Fresh-heldout GCP audit-only risk-veto validation completed:
  `outputs/evals/regular-hu-t2-stage8c-risk-veto-audit-t030-rank45-20260615-142635/`.
  Config: `k5/mc64/d0/se0/confirm128/cse1/scd1/pd0/bygate_delta`,
  threshold `0.30`, rank guard `4..5`, `Stage8cRiskAuditOnly=True`,
  `TargetRiskVetoesPerSeed=50`, seeds `2026072001..2026072003`, and
  `t3_continuation=stage3_reference_default`. All 3 shards completed with no
  missing shards and no residual VM instances. The ordinary TopK per-fire
  aggregate was positive (`801` realized fires, per-fire mean `+2.5907`,
  95% CI low `+1.8645`, EV/hand `+0.07125`) and cancellation was clean
  (`non_fired_nonzero_count=0`). However, the risk-veto adoption metric failed:
  `95` would-veto rows were marked, `92` had realized candidate deltas, and
  the would-veto candidate delta mean was `+3.2539` with 95% CI
  `[+1.5468, +4.9611]`. The implied veto utility is therefore `-3.2539` per
  veto and `-0.01030` EV/hand. Conclusion: the current rank4-5 threshold0.30
  local-EV risk veto is No-Go; it blocks profitable candidates in heldout
  audit-only evaluation. Do not enable this veto in runtime.
- A pure TopK+confirm heldout replication without any risk-veto model completed:
  `outputs/evals/regular-hu-t2-stage8c-topk-confirm-replication-20260615-164114/`.
  Config: `k5/mc64/d0/se0/confirm128/cse1/scd1/pd0/bygate_delta`,
  seeds `2026072101..2026072103`, `TargetRealizedOverridesPerSeed=250`,
  `TargetRiskVetoesPerSeed=0`, and
  `t3_continuation=stage3_reference_default`. All 3 shards completed, no
  missing shards, no failed shards, and no residual VM instances. Cancellation
  was clean (`non_fired_nonzero_count=0`) and replay fields were complete.
  Result: `722` realized fires over `14,383` paired seeds, per-fire delta mean
  `+1.7653`, 95% CI low `+0.9731`, estimated EV/hand `+0.04529` with 95% CI
  `[+0.02497, +0.06561]`. This independently reproduces the positive
  TopK+confirm signal, though weaker than the risk-audit run.
- Combined TopK+confirm aggregate across the risk-audit run and pure
  replication is at
  `outputs/evals/regular-hu-t2-stage8c-topk-confirm-combined-20260615/`.
  Combined result: `1,523` realized fires over `29,383` paired seeds, per-fire
  delta mean `+2.1994`, 95% CI low `+1.6635`, estimated EV/hand `+0.05850`,
  and clean cancellation. This is strong evidence that Stage8c TopK+confirm is
  a useful offline/search policy and candidate data generator. It still does
  not approve production/P2 fixed status because the policy is computationally
  heavy and the deployable lightweight runtime gate has not been recovered.
- The combined aggregate now writes `aggregate_position_breakdown.csv`.
  Position-specific realized per-fire deltas are positive on both seats:
  first-seat `792` realized fires, per-fire mean `+2.1135`, 95% CI low
  `+1.4197`, estimated EV/hand `+0.05841`; second-seat `731` realized fires,
  per-fire mean `+2.2925`, 95% CI low `+1.4666`, estimated EV/hand `+0.05859`.
  This reduces the concern that the combined TopK+confirm signal is carried by
  only one seat. Continue to use realized per-fire/cancellation metrics for
  adoption decisions; do not use confirm-gate means as performance evidence.
- `analyze_hu_turn2_stage8c_topk_per_fire` now aggregates risk-veto audit
  rows directly from `runtime_decisions.jsonl` and writes
  `aggregate_risk_veto_candidate_metrics.csv`. The aggregate manifest includes
  `risk_veto_candidate_metric_present`, `risk_veto_candidate_realized_count`,
  `risk_veto_best_utility_per_veto`,
  `risk_veto_best_estimated_utility_per_hand`, and
  `risk_veto_adoption_decision`. This prevents future runs from relying on a
  manual post-hoc veto-utility calculation.
- `Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -StartShards` now uses
  `worker_stride=total_shards` for explicit shard retries. This means
  `-StartShards 0 -VmCount 1` runs only shard 0 instead of shard 0, 1, 2 in
  sequence. The previous behavior was harmless when later shards already had
  `DONE` markers, but unsafe for partially completed retries with multiple
  missing shards.
- Latest focused verification after the manifest coverage/readiness update:
  `python -m py_compile ...` passed, and the manifest/audit/gap focused pytest
  suite passed `21` tests.
- Latest follow-up verification for the fresh-heldout plan artifact:
  `python -m py_compile src/ofc_regular/analyze_hu_turn2_stage8c_risk_head.py`
  passed, and
  `python -m pytest -p no:cacheprovider tests/test_hu_turn2_stage8c_risk_head_audit.py`
  passed `11` tests.
- Latest runtime-gate verification:
  `python -m py_compile src/ofc_regular/train_hu_turn2_stage8c_risk_head.py src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py src/ofc_regular/analyze_hu_turn2_stage8c_risk_head.py`
  passed, and the focused runtime/audit/script suite passed `55` tests.
- Latest risk-veto target verification:
  `python -m py_compile src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py`
  passed, the focused TopK runtime/script tests passed `25` tests, the synthetic
  `--target-risk-vetoes-per-seed 1` stop smoke reached the risk-veto target,
  the audit-only target smoke wrote `risk_veto_candidate_metrics.csv`, and the
  GCP TopK runner DryRun carried both `target_risk_vetoes_per_seed=50` and
  `stage8c_risk_audit_only=true`.
- Latest aggregate/retry verification:
  `python -m py_compile src/ofc_regular/analyze_hu_turn2_stage8c_topk_per_fire.py src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py src/ofc_regular/analyze_hu_turn2_stage8c_risk_head.py`
  passed, the new analyzer reproduced the fresh-heldout GCP risk-veto result
  with `risk_veto_adoption_decision=No-Go`, `-StartShards 0 -VmCount 1 -DryRun`
  emitted `worker_stride=3`, and the focused runtime/audit/script suite passed
  `46` tests.
- Latest TopK+confirm replication verification:
  `Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1` received all 3 fresh heldout
  shards for `regular-hu-t2-stage8c-topk-confirm-replication-20260615-164114`,
  the analyzer wrote a Pass manifest with clean cancellation and
  `best_estimated_ev_per_hand=+0.04529`, and the combined six-shard aggregate
  wrote a Pass manifest with `best_estimated_ev_per_hand=+0.05850`.
- Latest position aggregate verification:
  `analyze_hu_turn2_stage8c_topk_per_fire` now writes
  `aggregate_position_breakdown.csv`; the combined six-shard aggregate shows
  first-seat and second-seat realized per-fire CI lows both positive, and the
  focused analyzer/script tests passed `22` tests.
- Latest GCP runner verification:
  `python -m pytest -p no:cacheprovider tests/test_stage8c_scripts.py` passed
  `11` tests, first-seat and second-seat `-DryRun` commands both passed, and the
  no-seat dry run correctly failed on the explicit-seat safety guard.
- Stage8c TopK+confirm distillation prep has been added:
  `src/ofc_regular/prepare_hu_turn2_stage8c_topk_distillation.py`.
  Combined risk-audit + pure-replication decision logs produced
  `outputs/training/hu_turn2_stage8c_topk_confirm_distillation_combined_20260615/`
  with `8,722` trainable rows: `477` realized-positive fired rows and `8,245`
  negatives (`1,046` realized-loss fired rows, `3,199` confirm-rejected rows,
  and `4,000` sampled topk_empty rows). The schema is
  `hu_turn2_stage8c_topk_confirm_distillation_v1`; it keeps realized
  fired-hand labels separate from confirm MC diagnostics and does not authorize
  production/P2/50k/T1.
- The Stage8c trainer now supports `--target-mode topk_confirm_fire`.
  This target is separate from `whole_game_risk` and `local_ev_negative`.
  It allows TopK+confirm distillation rows to be trained without pretending they
  are local replay labels.
- Distillation smoke results:
  - `runtime_tail_meta_only` trained quickly and achieved source-seed holdout
    ROC AUC around `0.95`, but it uses confirm MC/tail statistics. It is
    therefore not a deployable pre-confirm lightweight gate by itself.
  - `hu_delta_plus_runtime_tail_meta` overfit relative to runtime-tail-only
    under source-seed holdout.
  - `opportunity_proxy_plus_runtime_tail_meta` was the healthiest
    confirm-aware smoke (`val AP 0.477`, `test AP 0.393`, test ROC AUC
    `0.944`), but still assigns high probabilities to realized-loss fired
    rows, so it is not adoption-ready.
  - `opportunity_proxy_only` removes confirm MC features and is much weaker
    (`test AP 0.167`, test ROC AUC `0.765`). This means a truly deployable
    pre-confirm gate has not yet recovered the TopK+confirm signal.
  - `opportunity_proxy_plus_preconfirm_meta` adds deployable Stage8b metadata
    (`predicted_delta`, `gate_probability`, candidate rank, model/topk score,
    seat) while still excluding confirm MC and Stage-A MC statistics. It
    improves over opportunity-only but remains weak (`test AP 0.192`, test ROC
    AUC `0.818`). This is not strong enough to launch large training or a
    production-style seat-swap gate.
- The training threshold metrics now separate observed realized deltas from
  unknown deltas. This matters because confirm-rejected/topk_empty rows were
  not actually played, so their would-have-fired whole-game delta is unknown.
  Do not treat their placeholder zero as performance evidence. For the
  `opportunity_proxy_plus_preconfirm_meta` smoke, threshold `0.8` selected `68`
  test rows, but only `35` had observed realized deltas and `33` were unknown.
  Observed selected mean was positive (`+3.1493`) but this is still not a
  deployable gate result because many selected candidates need replay before
  their true whole-game delta is known.
- Current interpretation: TopK+confirm remains a strong offline/search policy
  and candidate data generator. The new distillation pipeline is useful, but
  the lightweight pre-confirm runtime gate is still No-Go. The next modeling
  step should either add better deployable pre-confirm features or train a
  teacher that mimics the TopK+confirm action/value before trying another
  seat-swap runtime gate.
- Added `extract_hu_turn2_stage8c_topk_replay_targets`, which joins
  `risk_head_predictions.csv` back to the TopK+confirm distillation JSONL and
  extracts only rows selected by a deployable pre-confirm head whose realized
  whole-game delta is still unknown. This keeps observed fired-hand metrics
  separate from replay-prep candidates.
- The `opportunity_proxy_plus_preconfirm_meta` threshold `0.8` extraction wrote
  `outputs/training/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_preconfirm_t080_20260615/`.
  It found `138` unknown selected candidates, all replay-ready. The heldout
  test-only extraction wrote
  `outputs/training/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_preconfirm_t080_test_20260615/`
  with `33` replay-ready unknown candidates. These are not performance
  evidence; they are the next independent replay targets needed before claiming
  the pre-confirm runtime head is useful.
- Replay compatibility smoke:
  `scripts/Run-HuTurn2Stage8cReplayChunks.ps1` accepted the test-only unknown
  target JSONL and ran `2` rows at MC4 under explicit
  `stage3_reference_default`, writing
  `outputs/evals/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_t080_test_mc4_smoke/`.
  Both rows mapped actions successfully and wrote replay teacher rows. MC4 is
  only a pipeline smoke, not an EV estimate.
- The unknown-target replay was expanded locally without Spot VM:
  - test-only `33` rows at MC128:
    `outputs/evals/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_t080_test_mc128_full33/`.
    All `33` mapped actions successfully. Mean replay delta was `+0.4901`,
    CI low `-0.3826`, and negative rate `42.4%`, so the pre-confirm head was
    still No-Go.
  - test-only `33` rows at MC512:
    `outputs/evals/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_t080_test_mc512_full33/`.
    Mean replay delta was `+0.6385`, CI `[-0.0397, +1.3167]`, negative rate
    `33.3%`, and hard negatives `11`; still No-Go.
  - all-split `138` unknown rows at MC512:
    `outputs/evals/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_t080_all_mc512_full138/`.
    Mean replay delta was `+0.5491`, CI `[-0.0014, +1.0997]`, negative rate
    `40.6%`, and hard negatives `56`; still No-Go as a runtime gate, but useful
    as new replay-labeled supervision.
- Added `analyze_hu_turn2_stage8c_unknown_replay`, producing replay summaries
  such as
  `outputs/evals/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_t080_all_mc512_full138_analysis/`.
  It joins replay summary rows back to the source JSONL, reports seat/source
  breakdowns, CI, negative rate, and top losses. The all-split MC512 replay
  produced `57` safe-LCB positive rows, `56` negative rows, and `25` gray rows.
- Added `merge_hu_turn2_stage8c_topk_replay_labels`, which replaces the old
  unknown placeholder labels in the TopK+confirm distillation file with MC512
  replay labels. The merged artifact is
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_distillation_t080_mc512_20260615/`
  with `8,722` rows, `138` replacements, `57` replay positives, `56` replay
  negatives, `25` replay gray/excluded rows, and `8,697` trainable
  `topk_confirm_fire` rows.
- A replay-labeled pre-confirm smoke using
  `opportunity_proxy_plus_preconfirm_meta` wrote
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_replay_labeled_t080_mc512_smoke/`
  and
  `models/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_replay_labeled_t080_mc512_smoke.pt`.
  Test AP improved to `0.3723` and ROC AUC to `0.8477`, but threshold precision
  remains too low (`0.8` test precision `0.516`, `0.9` test precision `0.636`).
  Replay positive/negative probabilities remain close, so this is still not a
  runtime gate.
- Added deployable lookahead+preconfirm feature modes:
  `lookahead_proxy_plus_preconfirm_meta` and
  `opportunity_lookahead_proxy_plus_preconfirm_meta`. A full local training run
  with `opportunity_lookahead_proxy_plus_preconfirm_meta` was stopped after a
  15-minute timeout with no completed artifact. Treat this mode as a
  Spot VM/optimization target before using it for full training.
- The Stage8c risk-head trainer now supports stratified row limiting for
  expensive feature-smoke runs:
  `--max-rows-mode stratified --max-rows-seed <seed>`. The default remains the
  historical prefix slice (`first`) for backward compatibility.
- Replay-labeled lookahead+preconfirm smoke runs:
  - Stratified `400` input rows with
    `opportunity_lookahead_proxy_plus_preconfirm_meta` completed in `67.4s`
    and wrote
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_lookahead_preconfirm_replay_labeled_t080_mc512_strat400_smoke/`.
    This was an execution/compatibility smoke only; 4 epochs left probabilities
    compressed around `0.52`.
  - Stratified `1,200` input rows with normal training settings completed in
    `195.3s` and wrote
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_lookahead_preconfirm_replay_labeled_t080_mc512_strat1200_smoke/`.
    Holdout metrics: test AP `0.4250`, test ROC AUC `0.7088`.
    Threshold `0.8` selected `5` test rows with precision `0.600`; threshold
    `0.9` selected no test rows. This proves the lookahead path runs and has
    some signal, but it is not a deployable gate.
- The full `8,722`-row replay-labeled
  `opportunity_lookahead_proxy_plus_preconfirm_meta` training was then run
  locally and completed in `1,732.7s` (`28.9` minutes), writing
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_lookahead_preconfirm_replay_labeled_t080_mc512_full/`
  and
  `models/hu_turn2_stage8c_topk_confirm_fire_opportunity_lookahead_preconfirm_replay_labeled_t080_mc512_full.pt`.
  Metrics: val AP `0.3732`, val AUC `0.8354`, test AP `0.3606`, test AUC
  `0.8574`. This is not an improvement over the lighter
  `opportunity_proxy_plus_preconfirm_meta` smoke on the key deployable-gate
  criteria: test AP was slightly lower (`0.3606` vs `0.3723`), threshold `0.8`
  precision was lower (`0.487` vs `0.516`), and threshold `0.9` precision was
  lower (`0.500` vs `0.636`). The lookahead model does separate MC512 replay
  positives and negatives more than the previous preconfirm model
  (`0.892` vs `0.754` mean probability), but negative replay rows still score
  far too high for a safe runtime gate.
- Current lookahead conclusion: the deployable short-lookahead features are
  mechanically usable and add some ranking signal, but they do not solve the
  TopK+confirm distillation problem. Do not promote this model to runtime,
  P2, 50k teacher, or T1. Future work should either cache/optimize lookahead
  feature generation for broader feature search, or redesign the target/model
  rather than simply increasing this training run.
- To test whether the MC512 replay negatives were merely underweighted, the
  Stage8c risk-head trainer now supports TopK-specific row weights:
  `--topk-replay-negative-weight`, `--topk-realized-loss-weight`, and
  `--topk-replay-positive-weight`. Defaults are `1.0`, so existing experiments
  are unchanged.
- Weighted pre-confirm experiments:
  - `replay_negative=8`, `realized_loss=2` wrote
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_replay_labeled_t080_mc512_hn8_loss2/`.
    It did not reduce replay-negative scores (`topk_confirm_replay_negative`
    mean probability `0.757`) and did not improve the gate.
  - `replay_negative=50`, `realized_loss=2` wrote
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_replay_labeled_t080_mc512_hn50_loss2/`.
    It did lower replay-negative mean probability to `0.439`, but at the cost
    of weaker holdout metrics (test AP `0.288`, test AUC `0.808`) and very low
    useful fire counts. Test threshold `0.8` selected only `13` rows; threshold
    `0.85` selected `4` rows and had negative observed mean delta. This means
    simple hard-negative weighting is not enough; the current deployable
    pre-confirm features still cannot reliably separate replay-positive and
    replay-negative TopK candidates.
- The next decomposition was to train a separate local-negative veto head on
  the MC512 replay-positive/replay-negative rows (`113` rows total) instead of
  forcing one fire head to solve both tasks. The trainer now supports fixed
  source-group splits via `--fixed-val-groups` and `--fixed-test-groups`, so
  fire and veto heads can be evaluated on aligned heldout seeds.
- Local-negative veto smokes:
  - `opportunity_proxy_plus_preconfirm_meta`, row-stratified: weak/noisy
    (`test AUC 0.611`).
  - `opportunity_lookahead_proxy_plus_preconfirm_meta`, row-stratified:
    high apparent test AUC (`0.875`) but only `17` test rows, so not trusted.
  - `opportunity_lookahead_proxy_plus_preconfirm_meta`, source-seed split:
    better overall (`test AUC 0.867`) and useful as a diagnostic, but still
    only `21` test rows.
  - Aligned split with val seed `2026072001` and test seed `2026072101` wrote
    `outputs/training/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_t080_mc512_aligned_2001_2101_smoke/`.
- Added `analyze_hu_turn2_stage8c_fire_veto_combo`, which joins a fire-head
  prediction CSV and a local-negative veto-head prediction CSV by
  state/action/baseline signatures. The aligned audit wrote
  `outputs/training/hu_turn2_stage8c_fire_preconfirm_plus_local_veto_lookahead_aligned_2001_2101_audit/`.
  On the aligned test seed (`24` joined replay rows), examples:
  - fire `0.8`, veto `<0.4`: kept `3`, precision `1.0`, mean replay delta
    `+2.80`.
  - fire `0.9`, veto `<0.4`: kept `3`, precision `1.0`, mean replay delta
    `+2.80`.
  - fire `0.95`, veto `<0.4`: kept `2`, precision `1.0`, mean replay delta
    `+3.17`.
  This is a useful direction, but the kept counts are far too small for a
  runtime decision. Treat it as evidence to collect more MC512/MC2048 replay
  rows around fire+veto boundary states, not as a production gate.
- The next T2 work should address MC selection bias, realized per-fire
  validation, and whole-game risk before launching a large Spot VM run.

Do not deploy T2, do not mark it P2, do not start T1, and do not launch 50k
teacher generation until a larger TopK + MC or revised gate validation gives
stable seat-swap evidence.

## 2026-06-15 Stage8c Fire + Veto Replay Target Prep

- Added `predict_hu_turn2_stage8c_risk_head`, an inference-only CLI that scores
  arbitrary Stage8c JSONL rows with a saved risk-head model. This is needed
  because the first local-negative veto prediction file only covered the `113`
  MC512 replay-labeled rows, not the unknown rows that need new replay.
- Scored the full replay-labeled TopK distillation file with the aligned
  local-negative veto model:
  `outputs/training/hu_turn2_stage8c_local_ev_negative_full_distillation_predict_20260615/`.
  Manifest:
  - input rows: `8,722`
  - predicted rows: `8,722`
  - skipped rows: `0`
  - model:
    `models/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_t080_mc512_aligned_2001_2101_smoke.pt`
  - feature mode: `opportunity_lookahead_proxy_plus_preconfirm_meta`
- Added `extract_hu_turn2_stage8c_fire_veto_replay_targets`, which joins fire
  predictions, full-row veto predictions, and the distillation JSONL by
  source/state/action/baseline keys. It emits only rows whose realized/local
  replay delta is still unknown.
- Extracted additional replay targets:
  `outputs/training/hu_turn2_stage8c_fire_veto_replay_targets_full_veto_20260615/`.
  Settings:
  - fire thresholds: `0.8, 0.85, 0.9, 0.95`
  - veto thresholds: `0.3, 0.4, 0.5`
  - fire/veto boundary width: `0.05`
  Results:
  - targets: `89`
  - replay-ready targets: `89`
  - targets with veto prediction: `89`
  - targets missing veto prediction: `0`
  - observed-delta rows skipped: `1,636`
  - target composition is mostly fire-boundary rows, with `15` rows kept by
    fire `0.8` + veto `<0.5`, `5` rows kept by fire `0.8` + veto `<0.4`, and
    `5` rows kept by fire `0.85` + veto `<0.5`.
- Ran a tiny replay compatibility smoke on the first `3` targets:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_full_veto_20260615_mc4_smoke/`.
  Settings:
  - future samples: `4`
  - T3 continuation: `Stage7_candidate_A_m5_r10`
  - rows: `3`
  - ok rows: `3`
  - action mapping: `ok` for all `3`
  This smoke only verifies schema/action-mapping compatibility. MC4 deltas are
  not evidence for gate quality.
- Current next step: replay the `89` targets with a meaningful independent MC
  count, preferably `MC512` first and then selected `MC2048/8192` if the
  boundary set contains both useful positives and false positives. Do not use
  fire/veto probabilities or confirm-gate means as performance evidence.
- The `89` targets were then replayed locally at `MC512` with T3 continuation
  `Stage7_candidate_A_m5_r10`:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_full_veto_20260615_mc512/`.
  All chunks completed (`89/89`, action mapping `ok`), and the analysis artifact
  is
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_full_veto_20260615_mc512_analysis/`.
  Overall replay result:
  - mean delta: `-0.3249`
  - 95% CI: `[-0.8152, +0.1654]`
  - negative rate: `58.4%`
  - hard-negative labels: `52`
  - safe-LCB196 positives: `16`
  - safe-LCB196 negatives: `52`
  - decision: `No-Go: replay CI low is not positive`
- Reason-level check:
  - `kept_after_fire_veto`: `15` rows, mean `+0.0793`, CI
    `[-1.0253, +1.1840]`, negative rate `60.0%`.
  - `just_kept_by_veto`: `7` rows, mean `-1.1522`, CI
    `[-1.8440, -0.4603]`, negative rate `85.7%`.
  - `veto_boundary`: `18` rows, mean `-0.0446`, CI
    `[-0.9748, +0.8856]`, negative rate `66.7%`.
  This means the current fire+veto boundary strategy is useful for finding hard
  negatives, but it is not a safe runtime gate.
- Updated next step: merge the `MC512` replay labels back into the Stage8c
  distillation/cache as additional hard negatives/positives, then retrain or
  redesign the veto target. Do not run C2/C3, P2, T1, 50k teacher, or production
  from this fire+veto gate.
- Merged those `89` MC512 labels into the existing replay-labeled TopK
  distillation:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_distillation_t080_mc512_plus_fire_veto_mc512_20260615/`.
  Replacements:
  - replay positives: `16`
  - replay negatives: `52`
  - replay gray/excluded: `21`
  - trainable `topk_confirm_fire` rows after merge: `8,676`
  - total replay negatives after merge: `108`
  - total replay positives after merge: `73`
- Ran a lightweight pre-confirm fire-head retrain on this merged dataset:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_plus_fire_veto_mc512_smoke/`
  and
  `models/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_plus_fire_veto_mc512_smoke.pt`.
  Metrics:
  - test AP: `0.4033`
  - test ROC AUC: `0.8501`
  - threshold `0.8`: selected `89`, precision `0.461`
  - threshold `0.9`: selected `46`, precision `0.565`
  - threshold `0.95`: selected `19`, precision `0.789`
  Compared with the previous pre-confirm smoke, AP improved, but threshold
  precision at useful fire counts did not. The model became broader rather than
  safer. Treat this as a better diagnostic/hard-negative dataset, not a runtime
  gate improvement.
- Retrained the local-negative veto head on the merged `181` replay-labeled rows
  (`108` local-negative positives and `73` positive-LCB controls):
  `outputs/training/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_t080_mc512_plus_fire_veto_aligned_2001_2101/`
  and
  `models/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_t080_mc512_plus_fire_veto_aligned_2001_2101.pt`.
  It reached `test AP 0.7584` and `test AUC 0.8137`. This was good enough as a
  diagnostic veto model, but still not runtime evidence.
- Scored the full `8,722`-row distillation file with the updated fire and veto
  heads, then extracted a second replay target set:
  `outputs/training/hu_turn2_stage8c_fire_veto_replay_targets_plus_fire_veto_models_20260615/`.
  Results:
  - targets: `111`
  - replay-ready targets: `111`
  - missing veto predictions: `0`
- Replayed those `111` rows at `MC512`:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_plus_fire_veto_models_20260615_mc512/`
  with analysis in
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_plus_fire_veto_models_20260615_mc512_analysis/`.
  Overall:
  - mean delta: `-0.3368`
  - 95% CI: `[-0.7939, +0.1203]`
  - negative rate: `48.65%`
  - decision: `No-Go`
  Reason-level checks showed useful pockets but not a safe gate:
  - `just_kept_by_veto`: `15` rows, mean `+1.1179`, CI
    `[+0.1767, +2.0590]`, negative rate `26.7%`
  - `kept_after_fire_veto`: `24` rows, mean `+0.7260`, CI
    `[-0.1271, +1.5792]`, negative rate `29.2%`
- Replayed the `15` `just_kept_by_veto` rows at `MC2048`:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_plus_fire_veto_models_20260615_just_kept_mc2048/`.
  Analysis:
  - mean delta: `+1.1333`
  - 95% CI: `[+0.2049, +2.0617]`
  - negative rate: `20.0%`
  - labels: `9` safe-LCB196 positives, `3` gray, `3` negatives
  This is a useful high-MC seed set, but the negative rate is still too high for
  a runtime gate.
- Merged the second `111` MC512 rows, then the `15` MC2048 rows, into the
  Stage8c distillation:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_distillation_t080_mc512_plus_two_fire_veto_rounds_jk_mc2048_20260615/`.
  Current replay-label totals:
  - replay positives: `106`
  - replay negatives: `161`
  - replay gray: `71`
  - trainable `topk_confirm_fire` rows: `8,651`
- Final smoke retrain on this merged data:
  - Fire head:
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_two_rounds_jk_mc2048_smoke/`
    and
    `models/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_two_rounds_jk_mc2048_smoke.pt`
    reached `test AP 0.4122` and `test AUC 0.8581`. Threshold precision remains
    insufficient for runtime use (`0.9` precision `0.588`, `0.95` precision
    `0.769` on tiny selected counts).
  - Local-negative veto head:
    `outputs/training/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_two_rounds_jk_mc2048_aligned_2001_2101/`
    and
    `models/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_two_rounds_jk_mc2048_aligned_2001_2101.pt`
    reached `test AP 0.8170` and `test AUC 0.7783`. This is now a better
    diagnostic veto head, but it still needs full-row combo inference and
    independent replay before it can support any runtime gate.
- Decision after the second fire/veto round:
  - T2/P2 fixed policy: `No-Go`
  - production runtime: `No-Go`
  - T1 training: `No-Go`
  - 50k teacher: `No-Go`
  - next empirical step, if continuing T2: run full-row inference with the
    final fire and final veto heads, extract the next boundary/keep set, and
    replay it independently. Do not use fire/veto probabilities or confirm-gate
    averages as performance evidence.
- That next empirical step was then executed:
  - Fire full-row inference:
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_full_distillation_predict_two_rounds_jk_mc2048_20260615/`
  - Veto full-row inference:
    `outputs/training/hu_turn2_stage8c_local_ev_negative_full_distillation_predict_two_rounds_jk_mc2048_20260615/`
  - Both scored `8,722 / 8,722` rows with `0` skipped rows.
  - Third-round replay target extraction:
    `outputs/training/hu_turn2_stage8c_fire_veto_replay_targets_two_rounds_jk_mc2048_models_20260615/`
    selected `103` replay-ready rows, all with veto predictions.
- Third-round MC512 replay:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_two_rounds_jk_mc2048_models_20260615_mc512/`.
  Results:
  - rows: `103 / 103`
  - mean delta: `-0.4170`
  - 95% CI: `[-0.8696, +0.0356]`
  - negative rate: `53.4%`
  - hard negatives: `55`
  - decision: `No-Go`
  Reason-level checks:
  - `kept_after_fire_veto`: `8` rows, mean `+0.0969`, CI
    `[-2.1450, +2.3388]`, negative rate `50.0%`
  - `just_kept_by_veto`: `5` rows, mean `+0.0718`, CI
    `[-2.4364, +2.5800]`, negative rate `60.0%`
  - `veto_boundary`: `8` rows, mean `-0.2617`, CI
    `[-2.4654, +1.9420]`, negative rate `62.5%`
- Updated Stage8c fire+veto conclusion:
  - The final 267-row veto head is a useful diagnostic, but the combined
    fire+veto runtime proxy still does not produce a safe replay target set.
  - Continue treating Stage8c as a data-generation / hard-negative-mining path.
  - Do not promote T2/P2, production, T1, or 50k teacher from this line.
- Merged the third-round `103` MC512 labels back into the Stage8c distillation:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_distillation_t080_mc512_plus_three_fire_veto_rounds_20260615/`.
  Current replay-label totals:
  - replay positives: `131`
  - replay negatives: `216`
  - replay gray: `94`
  - trainable `topk_confirm_fire` rows: `8,628`
- Three-round fire head retrain:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_three_rounds_smoke/`
  and
  `models/hu_turn2_stage8c_topk_confirm_fire_opportunity_preconfirm_three_rounds_smoke.pt`.
  Metrics:
  - trainable rows: `8,628`
  - positives / negatives: `608 / 8,020`
  - test AP: `0.4224`
  - test ROC AUC: `0.8506`
  - test threshold precision: `0.8 -> 0.495`, `0.9 -> 0.611`,
    `0.95 -> 0.615`
  This is a small AP improvement over the previous fire smoke (`0.4122`), but
  still not enough for a deployable runtime gate.
- Three-round local-negative veto head retrain:
  `outputs/training/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_three_rounds_aligned_2001_2101/`
  and
  `models/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_three_rounds_aligned_2001_2101.pt`.
  Metrics:
  - trainable rows: `347`
  - local-negative positives / positive-LCB controls: `216 / 131`
  - test AP: `0.8349`
  - test ROC AUC: `0.7781`
  - test threshold precision: `0.4 -> 0.725`, `0.5 -> 0.738`,
    `0.6 -> 0.829`
  This is a better diagnostic veto head, but it does not override the replay
  No-Go above because the fire head still selects too many bad rows.
- Updated three-round conclusion:
  - Stage8c fire/veto remains useful for hard-negative mining and model
    diagnosis.
  - The latest heads should not be promoted without another full-row inference,
    target extraction, and independent replay pass.
  - Given the weak fire precision, threshold tuning alone is unlikely to solve
    T2. Next serious progress should change the candidate generator or the
    fire target/model, not just run a larger seat-swap.
- Added `analyze_hu_turn2_stage8c_fire_veto_replay`, a fire/veto-specific
  replay diagnostic that joins replay summary rows back to the target JSONL and
  reports reason-member, probability-bin, source, seat, and top-loss breakdowns
  using independent replay deltas.
- Applied it to the third-round `103` MC512 replay:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_two_rounds_jk_mc2048_models_20260615_mc512_fire_veto_analysis/`.
  Key findings:
  - overall mean delta: `-0.4170`, CI `[-0.8696, +0.0356]`, negative rate
    `53.4%`
  - `topk_empty`: `33` rows, mean `-1.0230`, negative rate `60.6%`
  - second seat: `61` rows, mean `-0.6505`, CI low `-1.2974`
  - veto probability `>=0.60`: `60` rows, mean `-1.0103`, CI
    `[-1.5905, -0.4301]`
  - top losses are mostly second-seat `topk_empty` rows.
- Added replay-target extraction filters:
  - `--exclude-candidate-source`
  - `--max-veto-probability`
  These are replay-mining filters, not production runtime gates.
- Full-row inference was run for the three-round fire and veto heads:
  - fire:
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_full_distillation_predict_three_rounds_20260615/`
  - veto:
    `outputs/training/hu_turn2_stage8c_local_ev_negative_full_distillation_predict_three_rounds_20260615/`
  Both scored `8,722 / 8,722` rows with no skips.
- Filtered target-count diagnostic:
  `outputs/evals/hu_turn2_stage8c_fire_veto_replay_targets_two_rounds_jk_mc2048_models_20260615_mc512_fire_veto_analysis/three_round_filtered_target_count_comparison.csv`.
  Counts:
  - exclude `topk_empty` only: `16` replay-ready targets
  - exclude `topk_empty` and veto `<0.60`: `8` targets
  - exclude `topk_empty` and veto `<0.70`: `9` targets
  - exclude `topk_empty` and veto `<0.80`: `9` targets
- Updated actionable conclusion:
  - The current fire/veto stack can identify known-bad areas, but after removing
    them it has almost no unknown positive candidates left.
  - Do not spend Spot VM budget on another broad replay of this exact candidate
    generator.
  - The next useful T2 step is candidate-generator redesign: start from
    model-positive/rank-safe candidates or train a different fire target that
    does not rely on `topk_empty` and high-veto-risk rows. Threshold tuning on
    the current fire/veto heads is exhausted.

## 2026-06-16 Stage8c Two-Stage MC Evaluation Guardrails

- The Stage8c TopK+confirm path must be read as a two-stage search/evaluation
  procedure:
  - Stage A selects a candidate from TopK with common-random MC.
  - Stage B confirms the selected candidate versus baseline with independent
    common-random futures.
  - Stage B `confirm_delta` remains a gate diagnostic. It is not an unbiased
    performance metric after filtering on `confirm_delta >= cse * SE`.
- The only adoption-quality metric from this path is realized whole-game paired
  delta on fired hands, and only when non-fired rows cancel exactly:
  `cancellation_audit.csv` must have `non_fired_nonzero_count=0`,
  `non_fired_delta_sum=0`, and `non_fired_delta_max_abs=0`.
- `analyze_hu_turn2_stage8c_topk_per_fire` now makes this explicit:
  - `primary_metric_source=realized_fired_whole_game_delta`
  - `per_fire_performance_column=per_fire_delta_mean`
  - `hand_ev_performance_column=estimated_ev_per_hand`
  - `confirm_delta_metric_role=gate_diagnostic_only`
  - `confirm_delta_performance_claim_allowed=false`
  - `minimum_realized_fires_for_evidence=50`
  - `evidence_decision` requires clean cancellation, enough realized fires, and
    positive realized per-fire CI.
- Existing CSE1 evidence remains valid under this standard:
  `outputs/evals/regular-hu-t2-stage8c-topk-confirm-combined-20260615/` has
  `1,523` realized fires, clean cancellation, realized per-fire mean
  `+2.1994`, realized per-fire 95% CI low `+1.6635`, and estimated EV/hand
  `+0.0585`.
- A follow-up CSE1.5/CSE2 comparison was completed on GCP under the same
  standard with current hidden-discard/current-FL-EV rules and
  `stage3_reference_default` T3 continuation:
  `outputs/evals/regular-hu-t2-stage8c-cse15-cse2-20260616/`.
  Non-fired cancellation is exact (`non_fired_nonzero_count=0`,
  `non_fired_delta_sum=0`, `max_non_fired_delta_abs=0`) and both configs have
  `150` realized fires. CSE1.5 is the better setting in this comparison:
  realized per-fire mean `+3.9560`, 95% CI low `+2.0130`, estimated EV/hand
  `+0.0416`, EV/hand CI low `+0.0212`. CSE2 is also positive but lower:
  realized per-fire mean `+2.8039`, 95% CI low `+1.0344`, estimated EV/hand
  `+0.0191`, EV/hand CI low `+0.0070`.
- CSE1.5 runtime decision logs were converted into a fresh TopK+confirm
  distillation artifact:
  `outputs/training/hu_turn2_stage8c_topk_confirm_distillation_cse15_20260616/`.
  It contains `563` trainable rows: `58` realized-positive fired rows, `92`
  realized-loss fired rows, and `413` confirm-rejected negatives. A first
  `topk_confirm_fire` smoke using `opportunity_proxy_plus_preconfirm_meta` and
  `source_seed` split executed at
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_cse15_opportunity_preconfirm_source_seed_smoke_20260616/`.
  The model is not deployable: val/test ROC AUC are only `0.5851 / 0.6503`,
  and fixed threshold `0.5` has low precision (`0.126` val, `0.116` test).
  However, the new `risk_head_topk_metrics.csv` shows the top few scored rows
  have nonzero signal (test top3 precision `0.667`, test top5 precision `0.400`).
  Treat this as a diagnostic that a cheap pre-confirm fire head may need more
  high-quality CSE1.5 rows or a different objective, not as runtime approval.
- The Stage8c risk-head trainer now supports excluding specific
  `recommended_training_use` groups before sampling via
  `--exclude-recommended-use`. This is important because
  `topk_confirm_topk_empty` rows are easy "no opportunity" negatives and can
  inflate apparent classification quality while teaching little about fired
  candidate quality.
- A stricter CSE1 + CSE1.5 combined fire-head smoke was run with
  `topk_confirm_topk_empty` excluded:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_opportunity_preconfirm_source_seed_20260616/`.
  Inputs were the CSE1 combined distillation artifact plus the CSE1.5 artifact:
  `9,285` loaded rows, `4,000` `topk_confirm_topk_empty` rows excluded, and
  `5,285` trainable rows kept (`535` realized-positive, `1,138`
  realized-loss, `3,612` confirm-rejected). With
  `opportunity_proxy_plus_preconfirm_meta`, `source_seed` split, and auto
  positive weighting, val/test ROC AUC are `0.7042 / 0.7085`, AP
  `0.2807 / 0.2601`. TopK selection metrics are encouraging but still
  diagnostic only: test top1/top3/top5/top10 precision is
  `1.000 / 0.667 / 0.400 / 0.300`; test top20 precision is `0.400`.
- Two quick ablations were also run:
  - `pos_weight=none`:
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_opportunity_preconfirm_source_seed_posnone_20260616/`.
    It has similar AUC but fires no rows at thresholds `0.7+` and has poor
    test top1/top3 behavior, so it is not the preferred setting.
  - `topk_realized_loss_weight=2.0` with the same seed:
    `outputs/training/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_opportunity_preconfirm_source_seed_loss2_seed1604_20260616/`.
    It improves some val topK numbers but lowers test AP/AUC and weakens test
    top3/top5 selection, so it is not preferred over the unweighted
    no-topk-empty run.
- Current fire-head reading: the combined no-topk-empty model is a useful
  ranking/triage aid for choosing rows to send into expensive TopK+confirm or
  high-MC labeling, but it is not a standalone runtime gate. Threshold metrics
  still have low precision; continue judging strength by realized per-fire
  whole-game deltas after exact non-fired cancellation.
- A reproducible comparison artifact for these fire-head smokes now exists at
  `outputs/training/hu_turn2_stage8c_fire_head_cse1_cse15_no_topk_empty_comparison_20260616/`.
  It compares the base no-topk-empty run, two realized-loss-weight variants,
  and the no-positive-weight variant. Preferred triage run:
  `base_no_topk_empty`; runtime gate decision: `No-Go`. The comparison summary
  records test AP/AUC, TopK precision, threshold precision, and the explicit
  role boundary: ranking/triage only, not production/P2/T1/50k evidence.
- First MC512 replay-label expansion:
  - Fire-head predictions from
    `models/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_opportunity_preconfirm_source_seed.pt`
    selected `227` replay-ready unknown rows after excluding
    `topk_confirm_topk_empty`.
  - GCP replay run:
    `regular-hu-t2-stage8c-no-topk-empty-firehead-replay-mc512-20260616`
    completed `5 / 5` shards, `227 / 227` ok rows, no missing shards, and no
    residual running instances.
  - MC512 labels: `58` replay-positive, `109` replay-negative, `60` gray.
  - Replay-labeled distillation:
    `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_cse1_cse15_no_topk_empty_firehead_mc512_20260616/`.
  - Replay-labeled fire-head model:
    `models/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_mc512_replay_labeled.pt`.
    It improves test AP/AUC to `0.3164 / 0.7361` and test top10 precision to
    `0.400`, but fixed threshold precision remains insufficient for runtime.
  - Comparison artifact:
    `outputs/training/hu_turn2_stage8c_fire_head_cse1_cse15_no_topk_empty_mc512_replay_comparison_20260616/`.
    Preferred triage run became `replay_labeled_mc512`; runtime gate remains
    `No-Go`.
- Second MC512 replay-label expansion:
  - The first replay-labeled model selected `68` new replay-ready unknown rows
    at threshold `0.7` (`7` at `0.8`, `0` at `0.9`).
  - GCP replay run:
    `regular-hu-t2-stage8c-replay-labeled-firehead-next68-mc512-20260616`
    completed `2 / 2` shards, `68 / 68` ok rows, no missing shards, and no
    residual running instances.
  - MC512 labels: `12` replay-positive, `34` replay-negative, `22` gray.
  - Round2 replay-labeled distillation:
    `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_round2_cse1_cse15_no_topk_empty_firehead_mc512_20260616/`.
    Aggregate replay labels are now `70` replay-positive, `143`
    replay-negative, and `82` gray.
  - Round2 fire-head model:
    `models/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_mc512_replay_labeled_round2.pt`.
    Training metrics improved to test AP/AUC `0.3401 / 0.7673`; test top10
    precision is `0.500`. Fixed threshold `0.8` precision is only `0.408`, so
    this remains a triage/ranking model, not a runtime gate.
  - Round2 comparison:
    `outputs/training/hu_turn2_stage8c_fire_head_cse1_cse15_no_topk_empty_mc512_replay_round2_comparison_20260616/`.
    Preferred triage run: `replay_labeled_round2_mc512`; runtime gate decision:
    `No-Go`.
  - The round2 model left only `23` replay-ready unknown rows at threshold
    `0.7` (`3` at `0.8`, `0` at `0.9`) in:
    `outputs/training/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_cse1_cse15_no_topk_empty_replay_labeled_round2_firehead_20260616/`.
    This pack has now been replayed locally with MC512.
- Third MC512 replay-label expansion:
  - Local replay output:
    `outputs/evals/hu_turn2_stage8c_replay_labeled_round2_firehead_next23_mc512_20260616/`.
  - Scale: `23` rows, `512` futures, one local chunk, runtime about `99s`.
  - Replay status: `23 / 23` ok, action mapping `23 / 23` ok.
  - MC512 labels: `12` replay-positive, `7` replay-negative, `4` gray.
  - Round3 replay-labeled distillation:
    `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_round3_cse1_cse15_no_topk_empty_firehead_mc512_20260616/`.
    Aggregate replay labels are now `82` replay-positive, `150`
    replay-negative, and `86` gray.
  - Round3 fire-head model:
    `models/hu_turn2_stage8c_topk_confirm_fire_cse1_cse15_no_topk_empty_mc512_replay_labeled_round3.pt`.
    Metrics: test AP/AUC `0.3307 / 0.7389`, test top10 precision `0.400`.
  - Round3 comparison:
    `outputs/training/hu_turn2_stage8c_fire_head_cse1_cse15_no_topk_empty_mc512_replay_round3_comparison_20260616/`.
    The preferred triage run remains `replay_labeled_round2_mc512`, not round3:
    round2 has better test AP/AUC (`0.3401 / 0.7673`) and better test top10
    precision (`0.500`). Round3 improves test top3 precision but weakens the
    broader triage metrics.
  - Preferred round2 model rescored on the round3-labeled data leaves `0`
    replay-ready unknown rows at thresholds `0.7`, `0.8`, and `0.9`:
    `outputs/training/hu_turn2_stage8c_topk_confirm_unknown_replay_targets_round2_model_on_round3_labels_20260616/`.
    Within this CSE1+CSE1.5 no-topk-empty candidate pool, high-scoring unknown
    rows have therefore been closed out with MC512 replay labels.
- Do not cite `confirm_delta_mean_on_fired` as gain. In the combined CSE1 run it
  is `+2.5049`, about `+0.3055` above the realized per-fire mean; that gap is
  diagnostic evidence of residual confirm-gate selection bias, not extra EV.
- A small round3 fire-head ablation was run after closing out the high-scoring
  unknown rows:
  `outputs/training/hu_turn2_stage8c_fire_head_round3_h64_ablation_comparison_20260616/`.
  It compared the existing round2 preferred model against round3 labels with
  hidden size `64`, plus a `1.5x` realized-loss/replay-negative weight variant.
  The existing round2 model remains the preferred triage model:
  test AP/AUC `0.3401 / 0.7673`, test top10 precision `0.500`.
  The best new h64 run had slightly higher test AP/AUC (`0.3493 / 0.7719`)
  but worse test top10 precision (`0.400`) and lower comparison triage score.
  The loss-weighted h64 run weakened both AP and top10 precision. Conclusion:
  simple capacity increases or negative/loss row reweighting do not solve the
  deployable fire-gate problem; the round2 model is still only a triage/ranking
  aid for selecting rows for expensive replay.
- Added a research-only realized-delta head trainer:
  `src/ofc_regular/train_hu_turn2_stage8c_delta_head.py`.
  It keeps two target semantics explicit:
  - `policy_delta`: rejected/topk-empty rows are treated as zero because the
    policy did not fire.
  - `observed_delta`: only fired or MC replay-labeled rows are used, so rejected
    rows are not misread as candidate-action EV labels.
  This is a candidate-generator diagnostic only; it does not authorize runtime
  gates.
- Delta-head smokes on the round3 no-topk-empty dataset:
  - `policy_delta`:
    `outputs/training/hu_turn2_stage8c_delta_head_policy_round3_no_topk_empty_opportunity_preconfirm_h32_20260616/`.
    Test regression correlation is weak (Pearson `0.0836`, Spearman `0.0399`),
    but top-ranked rows are useful for replay triage: test top3 mean delta
    `+10.4090`, top5 `+5.4454`, top20 `+4.7936`.
  - `observed_delta`:
    `outputs/training/hu_turn2_stage8c_delta_head_observed_round3_no_topk_empty_opportunity_preconfirm_h32_20260616/`.
    It skips `3294` unobserved rejected rows. Test correlation is also weak
    (Pearson `0.0210`, Spearman `0.0563`), with test top5 mean delta
    `+6.9689` and top20 `+4.4453`.
  - Compared with the preferred round2 classifier, the delta heads are better
    only for very small topK slices, while the round2 classifier remains better
    around top20 (`+7.1982` test mean delta). Conclusion: delta regression is
    useful as an additional top-few replay triage signal, but it does not replace
    the round2 fire-head ranking model or solve runtime gating.
- Added a same-row prediction-ranker comparison:
  `src/ofc_regular/compare_hu_turn2_stage8c_prediction_rankers.py`.
  It joins the preferred round2 fire-head predictions with the policy/observed
  delta-head predictions by `source_log`, `hand_seed`, `candidate_index`,
  `baseline_index`, and `recommended_training_use`, with `topk_confirm_topk_empty`
  excluded by default. Output:
  `outputs/training/hu_turn2_stage8c_prediction_ranker_comparison_round3_20260616/`.
  Join result: `5,285` fire rows after filter, `5,285` policy-delta rows joined,
  and `1,991` observed-delta rows joined. The fire prediction CSV has empty
  split values, so the comparison fills split from the joined delta prediction
  rows.
- Ranker-comparison reading:
  - `observed_delta` and its combinations are strong on the subset where
    observed/replay labels exist (`329` eligible test rows). For example,
    test top10 `observed` and `fire_probability+observed` both have mean
    realized delta `+13.0681`.
  - This is not a general runtime/candidate ranker because unobserved rejected
    rows are absent from the observed-delta prediction set.
  - On the full no-topk-empty policy test slice (`799` eligible rows),
    `policy_delta` remains useful only as a top-few triage signal
    (test top3 `+10.4090`, top20 `+4.7936`). The best broader top20 signal in
    this comparison is still not a deployable gate; `confirm_delta` has top20
    `+9.8181` but also a `29.2270` max loss, and `fire_probability+policy`
    has top20 `+7.2465` with lower max loss `6.2270`.
  - Use this comparison only to prioritize expensive replay rows. It does not
    change runtime/P2/T1/50k No-Go status.
- Added a replay-target extractor for the ranker comparison:
  `src/ofc_regular/extract_hu_turn2_stage8c_ranker_replay_targets.py`.
  It joins `prediction_ranker_top_rows.csv` back to distillation rows and emits
  only unknown/unobserved candidate deltas by default. It keeps the same
  boundary as the other replay tools: selected rows are independent replay
  targets only, not performance evidence.
- Ranker replay target pack:
  `outputs/training/hu_turn2_stage8c_prediction_ranker_replay_targets_round3_20260616/`.
  Inputs:
  - distillation rows:
    `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_round3_cse1_cse15_no_topk_empty_firehead_mc512_20260616/topk_confirm_replay_labeled_distillation_rows.jsonl`
  - ranker top rows:
    `outputs/training/hu_turn2_stage8c_prediction_ranker_comparison_round3_20260616/prediction_ranker_top_rows.csv`
  - rankers: `confirm_delta`, `fire_probability`, `fire_probability+policy`,
    `policy`
  - split: `test`
  - max rank: `300`
  - max targets: `100`
  Result: `100` replay-ready targets, `0` missing distillation rows, `0`
  replay blockers. Target counts by ranker: `confirm_delta=30`,
  `fire_probability=30`, `fire_probability+policy=30`, `policy=10`.
- Ranker replay MC4 full smoke:
  `outputs/evals/hu_turn2_stage8c_ranker_replay_targets_round3_mc4_full_20260616/`.
  It replayed all `100` targets with `future_samples=4`, hidden-discard
  `stage3_reference_default` T3 continuation, and wrote `100 / 100` samples.
  Status/action mapping: `100 / 100 ok`. This proves the target pack is
  mechanically replayable; MC4 is not quality evidence. The MC4-only diagnostic
  had `39` hard-negative labels and mean delta `+0.3886`.
- Next replay command for meaningful labels:

```powershell
.\scripts\Run-HuTurn2Stage8cReplayChunks.ps1 `
  -InputJsonl outputs\training\hu_turn2_stage8c_prediction_ranker_replay_targets_round3_20260616\ranker_replay_targets.jsonl `
  -OutputRoot outputs\evals\hu_turn2_stage8c_prediction_ranker_replay_targets_round3_mc512 `
  -FutureSamples 512 `
  -ChunkSize 50 `
  -ReplaySeed 2026061609 `
  -T3Continuation stage3_reference_default
```

Dry-run result for that command: source rows `100`, chunks `2 x 50`, merged
output enabled. If MC2048 is desired, use the same input with
`-FutureSamples 2048`; for Spot VM, use
`scripts/Start-GcpHuTurn2Stage8cReplayChunksRun.ps1` with this input JSONL and
the same `stage3_reference_default` continuation.
- The 100-row MC512 replay was completed locally:
  `outputs/evals/hu_turn2_stage8c_prediction_ranker_replay_targets_round3_mc512/`.
  It ran in two chunks (`50 + 50`), wrote `100 / 100` samples, and kept
  `stage3_reference_default` T3 continuation. Merged output:
  `outputs/evals/hu_turn2_stage8c_prediction_ranker_replay_targets_round3_mc512/merged/`.
- MC512 ranker analysis:
  `outputs/evals/hu_turn2_stage8c_prediction_ranker_replay_targets_round3_mc512_analysis/`.
  Overall result across the selected 100 rows: mean delta `+0.2790`, 95% CI
  `[+0.0926, +0.4654]`, `39` hard negatives, `31` safe LCB196 positive rows.
  Ranker breakdown:
  - `confirm_delta`: `30` rows, mean `+0.7587`, CI `[+0.4861, +1.0313]`,
    negative rate `0.200`, `17` LCB196-positive rows.
  - `fire_probability`: `30` rows, mean `+0.1621`, CI `[-0.1528, +0.4771]`,
    negative rate `0.467`.
  - `fire_probability+policy`: `30` rows, mean `+0.0121`, CI
    `[-0.3180, +0.3422]`, negative rate `0.500`.
  - `policy`: `10` rows, mean `-0.0084`, CI `[-0.8162, +0.7994]`, negative
    rate `0.400`.
- Interpretation: among the current ranker candidates, only `confirm_delta`
  is worth follow-up. Fire-head and policy-delta rankers are still useful for
  broad triage/hard-negative mining but are not good enough as replay target
  selectors here.
- A focused `confirm_delta` 30-row MC2048 replay was then run:
  `outputs/evals/hu_turn2_stage8c_prediction_ranker_confirm_delta_round3_mc2048_20260616/`.
  It wrote `30 / 30` samples in about `318s` with `stage3_reference_default`.
  Analysis:
  `outputs/evals/hu_turn2_stage8c_prediction_ranker_confirm_delta_round3_mc2048_analysis_20260616/`.
  Result: mean delta `+0.5881`, 95% CI `[+0.3264, +0.8499]`, negative rate
  `0.233`, `7` hard negatives, and `17` LCB196-positive rows. MC512 vs MC2048
  sign agreement on these 30 rows was `25 / 30`.
- This is promising as a candidate-generation/replay-prior result, not as a
  runtime gate. A `confirm_delta` selector still leaves too many negatives
  (`7 / 30`) for production/P2/T1. Use the MC2048 rows to add both positives
  and hard negatives to the next training pass, or require an additional
  learned safety/veto layer before considering runtime validation.
- The focused MC2048 `confirm_delta` labels were merged back into the
  Stage8c distillation rows:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_round4_confirm_delta_mc2048_20260616/`.
  The merge replaced `30` rows: `17` positive, `7` negative, and `6` gray.
  The merge code now records replay label basis from the actual replay sample
  count, so these rows are marked `mc2048_independent_replay` rather than a
  stale MC512 label basis. The merge manifest also reports
  `replay_label_basis_counts = {"mc2048_independent_replay": 30}` and
  `local_replay_future_sample_counts = {"2048": 30}` for auditability.
- Round4 fire-head training on the merged file completed:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_round4_confirm_delta_mc2048_20260616/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_round4_confirm_delta_mc2048.pt`.
  Metrics: `5,193` trainable rows, `634` positives, test AP/AUC
  `0.3165 / 0.6962`, test top3 precision `0.333`, test top10 precision
  `0.400`, and threshold `0.8` precision `0.429` over `63` selected rows.
- Round2/Round3/Round4 comparison artifact:
  `outputs/training/hu_turn2_stage8c_fire_head_round4_confirm_delta_mc2048_comparison_20260616/`.
  Preferred triage run remains `round2`, with test AP/AUC `0.3401 / 0.7673`
  and test top10 precision `0.500`. Round4 does not improve the broad triage
  model and should not replace Round2 as the current fire-head triage model.
  Runtime gate decision remains `No-Go`.
- Current status remains unchanged:
  - Stage8c TopK+confirm is useful as an offline/search policy and data
    generator.
  - Fire/veto lightweight runtime heads remain No-Go.
  - T2/P2 fixed, production runtime, T1 training, and 50k teacher remain No-Go.

### 2026-06-16 evaluator Conditional-Go guardrail update

- The primary TopK+confirm evaluator now requires realized fired whole-game
  per-fire evidence before it can emit `Conditional-Go`.
- `topk_conditional_go` requires all of the following for the best config:
  clean non-fired cancellation, no negative-predicted-delta safety blocker,
  positive aggregate `aggregate_ev_per_hand`, non-disastrous seed CI low,
  `realized_override_count > 0`, `per_override_delta_mean > 0`, and
  `estimated_ev_per_hand > 0`.
- The generated `topk_rerank_summary.md` and `go_nogo.md` now include explicit
  fields:
  - `conditional_realized_per_fire_positive`
  - `conditional_realized_overrides_for_best_config`
  - `conditional_realized_per_fire_required_for_conditional_go=True`
  - `confirm_rerank_diagnostics_used_for_conditional_go=False`
- Safety blockers are now reported in two scopes:
  - `known_negative_predicted_delta_safety_blocker_for_best_config`
  - `known_negative_predicted_delta_safety_blocker_any_config`
  The `Conditional-Go` gate uses the best-config scope. The any-config field is
  still useful audit context, but a blocker in a non-best config no longer
  suppresses an otherwise clean best config.
- Non-fired cancellation is also reported in two scopes:
  - `cancellation_clean_for_best_config`
  - `cancellation_clean_all_configs`
  The `Conditional-Go` gate uses the best-config scope. The all-configs field
  remains audit context for spotting harness issues in other configs.
- This closes an ambiguity in earlier artifacts: `confirm_delta`,
  `rerank_delta`, and `confirm_delta_mean_on_fired` remain gate diagnostics
  only. They are not enough to justify `Conditional-Go` and must not be cited
  as performance evidence.
- Verification after this change:
  - related evaluator tests: `53 passed`
  - full pytest: `420 passed`
  - `git diff --check`: only LF/CRLF warnings
- Current status remains unchanged:
  - Stage8c TopK+confirm is useful as an offline/search policy and replay/data
    generator.
  - Runtime gates, T2/P2 fixed policy, production runtime, T1 training, and
    50k teacher remain `No-Go`.

### 2026-06-17 Stage8c prediction-ranker role guardrail

- `src/ofc_regular/compare_hu_turn2_stage8c_prediction_rankers.py` now emits
  role metadata for every ranker row:
  - `deployable_ranker_input`: model/runtime prediction columns such as
    `fire_probability`, `predicted_delta`, and deployable delta-head outputs.
  - `replay_triage_only`: rankers using `confirm_delta` or confirm/replay
    deltas. These are valid for selecting replay targets, not runtime evidence.
  - `diagnostic_only`: rankers using observed/replay/teacher/oracle fields.
  - `diagnostic_combo`: normalized score combinations that are analysis-only
    until separately implemented and validated as runtime features.
- The comparison tool now writes
  `prediction_ranker_recommendations.csv`, including `ranker_role`,
  `runtime_eligible`, `role_reason`, and `recommendation`. This prevents
  strong-looking `confirm_delta` or observed-delta rows from being mistaken for
  deployable runtime gates.
- `src/ofc_regular/extract_hu_turn2_stage8c_ranker_replay_targets.py` now
  defaults to deployable runtime rankers only. Non-runtime rankers such as
  `confirm_delta` require the explicit `--allow-non-runtime-rankers` opt-in.
  The hard safety check is based on the ranker name, so a malformed CSV cannot
  promote `confirm_delta` by setting `runtime_eligible=True`.
- Real-data smoke:
  `outputs/training/hu_turn2_stage8c_prediction_ranker_comparison_round3_roles_20260617/`
  was generated from the Round3 comparison inputs. All `confirm_delta` rankers
  are `runtime_eligible=False`. Runtime-eligible recommendations are limited to
  deployable prediction inputs.
- Runtime-only replay-target smoke:
  `outputs/training/hu_turn2_stage8c_ranker_replay_targets_round3_roles_runtime_only_smoke_20260617/`
  loaded `9,285` distillation rows and emitted `5` replay-ready targets, all
  from `predicted_delta`; `confirm_delta` rows were excluded by default.
  The replay-target summary now reports the selection audit: `60` deployable
  top rows considered (`20` each from `fire_probability`, `policy`, and
  `predicted_delta`) and `240` non-runtime ranker rows excluded, including all
  `confirm_delta`, observed/replay-derived, and analysis-combo rankers.
- The `5` runtime-only `predicted_delta` replay targets were independently
  replayed at MC512:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_predicted_delta5_mc512_20260617/`.
  Analysis:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_predicted_delta5_mc512_analysis_20260617/`.
  Result: `5 / 5` action mappings OK, mean delta `-0.2192`, 95% CI
  `[-0.5139, +0.0755]`, negative rate `0.800`, hard negatives `4`, and
  safe LCB196 positives `0`. This is a small sample, but it is enough to mark
  plain deployable `predicted_delta` target selection as weak for this pool.
  It should be used as hard-negative evidence, not as a reason to scale this
  ranker directly.
- The 5 MC512 labels were merged into a new replay-labeled distillation
  artifact:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_runtime_predicted_delta5_mc512_20260617/`.
  Merge result: rows `9,285`, replacements `5`, replay positives `0`, replay
  negatives `4`, replay gray `1`, all with `mc512_independent_replay` basis.
  The merged trainable topk-confirm-fire rows count is `9,192`.
- A compatibility fire-head smoke trained successfully on the merged artifact:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_predicted_delta5_mc512_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_runtime_predicted_delta5_mc512.pt`.
  Metrics: trainable rows `5,192`, positives `634`, negatives `4,558`, test
  AP/AUC `0.3402 / 0.7389`. This is a training-pipeline compatibility check,
  not a promotion. The 4 newly negative rows were in the test split; predicted
  probabilities were approximately `0.207` to `0.340`, with the row that old
  Round4 scored `0.597` now scored `0.340`.
- Fire-head comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_runtime_predicted_delta5_comparison_20260617/`.
  Compared `round2`, `round4_confirm_delta`, and
  `runtime_predicted_delta5`. Preferred triage run remains `round2`.
  Summary:
  - `round2`: test AP/AUC `0.3401 / 0.7673`, test top10 precision `0.500`,
    threshold 0.8 precision `0.408` over `49` selected rows.
  - `runtime_predicted_delta5`: test AP/AUC `0.3402 / 0.7389`, test top10
    precision `0.400`, threshold 0.8 precision `0.466` over `73` selected
    rows.
  - `round4_confirm_delta`: test AP/AUC `0.3165 / 0.6962`, test top10
    precision `0.400`, threshold 0.8 precision `0.429` over `63` selected
    rows.
  Interpretation: the 5 hard-negative labels improve one known false-positive
  probability and keep training compatible, but they do not beat `round2` as the
  current broad triage model. Do not promote the runtime-predicted-delta5 model.
- Broader runtime-only top100 replay:
  `outputs/training/hu_turn2_stage8c_ranker_replay_targets_round3_roles_runtime_only_top100_20260617/`
  selected `38` replay-ready deployable-ranker targets from the Round3 test
  top100 pool: `fire_probability=1`, `policy=17`, `predicted_delta=20`.
  It excluded `1,200` non-runtime ranker rows by default.
- MC512 replay:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_top100_mc512_20260617/`
  and analysis
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_top100_mc512_analysis_20260617/`.
  Overall result: `38 / 38` OK, mean delta `-0.1056`, 95% CI
  `[-0.3685, +0.1573]`, negative rate `0.632`, hard negatives `24`, safe
  LCB196 positives `6`. Ranker split:
  - `fire_probability`: `1` row, mean `+0.8273`, LCB196+ `1`.
  - `policy`: `17` rows, mean `-0.2104`, negative rate `0.706`, hard
    negatives `12`, LCB196+ `2`.
  - `predicted_delta`: `20` rows, mean `-0.0632`, negative rate `0.600`, hard
    negatives `12`, LCB196+ `3`.
  Seat split is also important: first seat mean `+0.1087`, second seat mean
  `-0.3438` with CI high below zero, so second-seat deployable-ranker target
  selection is especially weak here.
- The 38 MC512 labels were merged into
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_runtime_top100_mc512_20260617/`.
  Merge result: replacements `38`, replay positives `6`, replay negatives
  `24`, replay gray `8`, all with `mc512_independent_replay` basis.
- A compatibility fire-head smoke trained successfully:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top100_mc512_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_runtime_top100_mc512.pt`.
  Metrics: trainable rows `5,186`, positives `639`, negatives `4,547`, test
  AP/AUC `0.3355 / 0.7453`, test top10 precision `0.600`, threshold 0.8
  precision `0.467` over `60` selected rows.
- Four-way fire-head comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_runtime_top100_comparison_20260617/`.
  Preferred triage run remains `round2`. `runtime_top100` is close on aggregate
  triage score and has better top10 precision than the 5-row model, but its
  test AP/AUC remains below `round2`; it is still `No-Go` for runtime.
- Seat-specific follow-up:
  `src/ofc_regular/extract_hu_turn2_stage8c_ranker_replay_targets.py` now
  supports repeatable `--seat` filters. The selection manifest records
  `selected_seats`, and the audit reports `skipped_seat_filter`. This lets
  first/second-seat replay target pools be generated without mixing the weak
  second-seat evidence into first-seat triage.
- First-seat top100 runtime-only replay:
  `outputs/training/hu_turn2_stage8c_ranker_replay_targets_round3_roles_runtime_only_top100_first_20260617/`
  selected `28` first-seat deployable-ranker replay targets, all runtime
  eligible (`predicted_delta=20`, `policy=7`, `fire_probability=1`). MC512
  replay output:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_top100_first_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_top100_first_mc512_analysis_20260617/`.
  Result: `28 / 28` OK, mean delta `+0.0892`, 95% CI
  `[-0.3037, +0.4821]`, hard negatives `12`, safe LCB196 positives `5`.
  This looked better than the mixed-seat top100 pool but remained too small and
  too noisy for a runtime claim.
- First-seat top300 runtime-only replay:
  `outputs/training/hu_turn2_stage8c_prediction_ranker_comparison_round3_roles_top300_20260617/`
  was generated with `--top-row-limit 300`, then
  `outputs/training/hu_turn2_stage8c_ranker_replay_targets_round3_roles_runtime_only_top300_first_20260617/`
  selected `120` first-seat deployable-ranker replay targets
  (`fire_probability=50`, `policy=32`, `predicted_delta=38`). MC512 replay:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_top300_first_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_ranker_runtime_only_top300_first_mc512_analysis_20260617/`.
  Result: `120 / 120` OK, mean delta `-0.1638`, 95% CI
  `[-0.3603, +0.0326]`, hard negatives `68`, safe LCB196 positives `18`.
  Ranker split:
  - `fire_probability`: `50` rows, mean `-0.0046`, hard negatives `24`,
    LCB196+ `12`.
  - `policy`: `32` rows, mean `-0.2708`, hard negatives `21`, LCB196+ `2`.
  - `predicted_delta`: `38` rows, mean `-0.2833`, hard negatives `23`,
    LCB196+ `4`.
  Label split remains useful for training: `safe_lcb196_label=positive` rows
  are strong (`18` rows, mean `+1.4024`, CI `[+1.1419, +1.6629]`), while
  `negative` rows are clearly bad (`68` rows, mean `-0.8466`). The deployable
  runtime rankers do not isolate these rows well enough yet.
- The 120 first-seat MC512 labels were merged into
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_runtime_top300_first_mc512_20260617/`.
  Merge result: replacements `120`, replay positives `18`, replay negatives
  `68`, replay gray `34`, all with `mc512_independent_replay` basis.
- A fire-head smoke trained successfully on the first-seat top300 merged
  artifact:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_mc512_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_mc512.pt`.
  Metrics: trainable rows `5,163`, positives `650`, negatives `4,513`, test
  AP/AUC `0.3385 / 0.7478`, test top10 precision `0.600`, threshold 0.8
  precision `0.464` over `56` selected rows.
- Fire-head comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_runtime_top300_first_comparison_20260617/`.
  Preferred triage run moved to `runtime_top300_first` by a very small triage
  score margin (`1.5696` vs `1.5657` for `round2`). This is not a runtime
  promotion: fixed threshold precision is still weak and test AUC remains below
  `round2`. The artifact's decision remains `runtime gate: No-Go`.
- Current interpretation:
  - The top100 first-seat positive signal was not stable when expanded to
    top300.
  - First-seat labels are useful for hard-negative and safe-head training.
  - Runtime-deployable rankers still fail to isolate safe LCB-positive rows.
  - The next rational step is not 50k teacher or T1; it is either a
    seat-specific safe head / candidate generator, or an explicit second-seat
    veto/skip path before any larger seat-swap.
- Fire-selector audit tools:
  - `src/ofc_regular/analyze_hu_turn2_stage8c_fire_selector.py` evaluates
    high `risk_probability` as a fire selector, not as a veto. This avoids the
    old risk/veto audit's reversed interpretation for `topk_confirm_fire`
    models.
  - `src/ofc_regular/extract_hu_turn2_stage8c_fire_selector_replay_targets.py`
    extracts replay-ready unknown rows directly from fire-head prediction CSVs,
    with split/seat/threshold filters. It writes compatible `ranker`,
    `ranker_score`, and `fire_probability` fields so downstream replay analysis
    does not collapse them into an empty ranker group.
- Fire-selector audit for
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_mc512_20260617/risk_head_predictions.csv`:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_mc512_fire_selector_audit_20260617/`.
  Screening-only result: on existing test labels, `seat=second` looked strong
  (`threshold=0.8`: `30` selected, precision `0.533`, mean delta `+5.4504`;
  `threshold=0.9`: `10` selected, precision `0.700`, mean delta `+5.7770`).
  This was not fresh replay evidence.
- Fresh replay target extraction for that second-seat hypothesis:
  - `p>=0.9`: `10` selected prediction rows, but all had observed/replay delta,
    so `0` unknown replay targets.
  - `p>=0.95`: `3` selected prediction rows, all observed, so `0` unknown
    replay targets.
  - `p>=0.7`: `50` selected prediction rows, `7` unknown replay-ready targets.
    Target artifact:
    `outputs/training/hu_turn2_stage8c_fire_selector_replay_targets_runtime_top300_first_model_second_test_p70_20260617/`.
- The 7 unknown second-seat fire-selector targets were replayed at MC512:
  `outputs/evals/hu_turn2_stage8c_fire_selector_runtime_top300_first_model_second_test_p70_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_fire_selector_runtime_top300_first_model_second_test_p70_mc512_analysis_20260617/`.
  Result: `7 / 7` OK, mean delta `+0.1731`, 95% CI
  `[-0.3698, +0.7160]`, hard negatives `3`, safe LCB196 positives `1`.
  This is execution-pass but still decision No-Go: the average is slightly
  positive, but the sample is tiny and the hard-negative rate remains high.
- The 7 MC512 labels were merged on top of the first-seat top300 labeled
  artifact:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_runtime_top300_first_plus_second_p70_mc512_20260617/`.
  New replacements: `7` (`1` replay positive, `3` replay negative, `3` gray).
- A fire-head smoke trained on this combined artifact:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_plus_second_p70_mc512_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_plus_second_p70_mc512.pt`.
  Comparison artifact:
  `outputs/training/hu_turn2_stage8c_fire_head_runtime_top300_first_plus_second_p70_comparison_20260617/`.
  Result: `runtime_top300_first_plus_second_p70` is the new training-smoke
  preferred triage run by a small margin. Test AP/AUC `0.3458 / 0.7521`;
  threshold 0.8 precision `0.481` over `54` selected rows. This is a useful
  incremental improvement, but still `runtime gate: No-Go`.
- Fire-selector audit for the combined model:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_first_plus_second_p70_mc512_fire_selector_audit_20260617/`.
  Existing-label screen remains promising for `seat=second`, but it is not a
  deployment result. The right next step is to keep mining fresh unknown rows or
  regenerate a broader heldout pool; do not jump to 50k/T1/production from this.
- Verification:
  - `python -m py_compile src\ofc_regular\compare_hu_turn2_stage8c_prediction_rankers.py`
  - `python -m py_compile src\ofc_regular\extract_hu_turn2_stage8c_ranker_replay_targets.py`
  - `python -m py_compile src\ofc_regular\merge_hu_turn2_stage8c_topk_replay_labels.py`
  - `python -m py_compile src\ofc_regular\train_hu_turn2_stage8c_risk_head.py`
  - `python -m py_compile src\ofc_regular\compare_hu_turn2_stage8c_fire_head_training.py`
  - `python -m py_compile src\ofc_regular\analyze_hu_turn2_stage8c_fire_selector.py`
  - `python -m py_compile src\ofc_regular\extract_hu_turn2_stage8c_fire_selector_replay_targets.py`
  - `python -m pytest tests\test_hu_turn2_stage8c_ranker_replay_targets.py tests\test_hu_turn2_stage8c_prediction_ranker_comparison.py tests\test_hu_turn2_stage8c_ranker_replay_analysis.py tests\test_hu_turn2_stage8c_topk_replay_label_merge.py tests\test_hu_turn2_stage8c_risk_head_training.py tests\test_hu_turn2_stage8c_fire_head_training_comparison.py -p no:cacheprovider`
    passed with `55 passed`.
  - `python -m pytest tests\test_hu_turn2_stage8c_fire_selector_analysis.py tests\test_hu_turn2_stage8c_fire_selector_replay_targets.py -p no:cacheprovider`
    passed with `4 passed`.
- Current status remains unchanged:
  - Stage8c TopK+confirm is useful as an offline/search policy and replay/data
    generator.
  - Ranker comparisons are replay triage, not runtime approval.
  - Runtime-only deployable rankers need redesign or a broader candidate pool;
    plain `predicted_delta` is not good enough from this smoke.
  - Runtime gates, T2/P2 fixed policy, production runtime, T1 training, and
    50k teacher remain `No-Go`.

## 2026-06-17 continuation: fire-selector label mining after p70

- The combined `runtime_top300_first_plus_second_p70` model remains the preferred
  triage model, but not a runtime gate.
- First, all split/seat unknown rows with fire probability `>=0.70` under the
  combined model were extracted and replayed:
  `outputs/training/hu_turn2_stage8c_fire_selector_replay_targets_runtime_top300_plus_second_model_all_p70_20260617/`.
  This yielded `51` replay-ready targets after excluding `662` already observed
  rows. MC512 replay completed successfully:
  `outputs/evals/hu_turn2_stage8c_fire_selector_runtime_top300_plus_second_model_all_p70_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_fire_selector_runtime_top300_plus_second_model_all_p70_mc512_analysis_20260617/`.
  Result: `51 / 51` OK, mean delta `+0.1850`, 95% CI
  `[-0.1223, +0.4924]`, hard negatives `20`, safe LCB196 positives `12`.
  This is useful label data only; runtime gate remains `No-Go`.
- Those 51 labels were merged into
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_runtime_top300_plus_second_all_p70_mc512_20260617/`.
  Merge result: replay replacements `51`, positives `12`, negatives `20`,
  gray `19`, trainable topk-confirm-fire rows `9,141`.
- A fire-head smoke was trained:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_plus_second_all_p70_mc512_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_runtime_top300_plus_second_all_p70_mc512.pt`.
  Metrics: test AP/AUC `0.3474 / 0.7551`, threshold 0.8 precision `0.431`
  over `51` selected rows. Comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_runtime_top300_first_plus_second_all_p70_comparison_20260617/`.
  Decision: old `runtime_top300_first_plus_second_p70` remained preferred for
  triage because the all-p70 model improved AP/AUC slightly but worsened
  high-threshold precision and top10 precision.
- Next, using the preferred old `runtime_top300_first_plus_second_p70` prediction
  scores against the updated all-p70 distillation, all split/seat unknown rows
  with fire probability `>=0.60` were extracted:
  `outputs/training/hu_turn2_stage8c_fire_selector_replay_targets_preferred_all_p60_after_all_p70_20260617/`.
  This yielded `143` replay-ready targets after excluding `906` already observed
  rows. MC512 replay completed successfully:
  `outputs/evals/hu_turn2_stage8c_fire_selector_preferred_all_p60_after_all_p70_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_fire_selector_preferred_all_p60_after_all_p70_mc512_analysis_20260617/`.
  Result: `143 / 143` OK, mean delta `+0.1430`, 95% CI
  `[-0.0595, +0.3454]`, hard negatives `64`, safe LCB196 positives `37`.
  Seat split was similar: first `+0.1222`, second `+0.1716`. Label split was
  clean: safe LCB196 positive rows averaged `+1.5552`; negative rows averaged
  `-0.8706`.
- The 143 labels were merged into
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_preferred_all_p60_after_all_p70_mc512_20260617/`.
  Merge result: replay replacements `143`, positives `37`, negatives `64`,
  gray `42`, trainable topk-confirm-fire rows `9,099`. Total replay-label
  counts in that artifact are now positives `165`, negatives `311`, gray `186`.
- A fire-head smoke was trained:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_preferred_all_p60_after_all_p70_mc512_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_preferred_all_p60_after_all_p70_mc512.pt`.
  Metrics: test AP/AUC `0.3463 / 0.7580`, threshold 0.8 precision `0.455`
  over `55` selected rows.
- Comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_preferred_all_p60_after_all_p70_comparison_20260617/`.
  Preferred triage run remained `runtime_top300_first_plus_second_p70`:
  `test AP/AUC 0.3458 / 0.7521`, top10 precision `0.600`, threshold 0.8
  precision `0.481`. The p60-expanded model gained more MC512 labels and AUC
  but did not improve the triage objective enough to replace the old preferred
  model.
- Fire-selector audit for the p60-expanded model:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_preferred_all_p60_after_all_p70_mc512_fire_selector_audit_20260617/`.
  Existing-label screening still shows second seat as the strongest slice
  (`threshold=0.8`: `28` selected, precision `0.571`, mean delta `+5.7281`;
  `threshold=0.9`: `9` selected, precision `0.667`, mean delta `+6.1967`),
  but this is screening evidence only.
- Current interpretation:
  - Adding more p70/p60 MC512 labels is useful for hard-negative mining and
    diagnosing the safe-LCB boundary.
  - The current fire-head architecture is not yet converting those labels into
    a clearly better deployable selector.
  - The old `runtime_top300_first_plus_second_p70` model should remain the
    active triage model for selecting future replay rows.
  - Runtime gate, T2/P2 fixed policy, production, T1, and 50k teacher remain
    `No-Go`.

## 2026-06-17 continuation: replay-label weighting sweep

- Because adding p60/p70 labels alone did not clearly improve the fire head,
  the next experiment weighted MC512 replay labels more strongly during
  training. All runs used
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_preferred_all_p60_after_all_p70_mc512_20260617/topk_confirm_replay_labeled_distillation_rows.jsonl`
  with target mode `topk_confirm_fire`, feature mode
  `opportunity_proxy_plus_preconfirm_meta`, split mode `source_seed`, and
  `topk_confirm_topk_empty` excluded.
- Weight sweep output:
  `outputs/training/hu_turn2_stage8c_fire_head_preferred_all_p60_weight_sweep_comparison_20260617/`.
  Tested replay-positive/replay-negative weights:
  - `w2p2n`
  - `w2p3n`
  - `w2p5n`
  - `w3p3n`
  - `w4p6n`
- Comparison result:
  - `preferred_all_p60_w4p6n` became the best training-smoke triage run by
    top3 precision (`1.000`), with top10 precision `0.600`, but test AP/AUC
    dropped to `0.3371 / 0.7266` and threshold 0.8 precision was only `0.439`
    over `57` selected rows.
  - `preferred_all_p60_w2p5n` had a more conservative threshold profile:
    threshold 0.8 precision `0.571` over `14` selected rows, but no fresh
    p>=0.8 replay targets remained after observed rows were excluded.
  - The previous `runtime_top300_first_plus_second_p70` remained competitive:
    top10 precision `0.600`, threshold 0.8 precision `0.481`, test AP/AUC
    `0.3458 / 0.7521`.
  - All runs remain runtime gate `No-Go`.
- Fire-selector audit for `preferred_all_p60_w4p6n`:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_preferred_all_p60_w4p6n_fire_selector_audit_20260617/`.
  It exposed `58` fresh all-split/seat targets at probability `>=0.70`:
  `outputs/training/hu_turn2_stage8c_fire_selector_replay_targets_w4p6n_all_p70_after_p60_20260617/`.
  MC512 replay completed:
  `outputs/evals/hu_turn2_stage8c_fire_selector_w4p6n_all_p70_after_p60_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_fire_selector_w4p6n_all_p70_after_p60_mc512_analysis_20260617/`.
  Result: `58 / 58` OK, mean delta `+0.2941`, 95% CI
  `[+0.0666, +0.5216]`, hard negatives `21`, safe LCB196 positives `19`,
  max loss `1.9121`. Seat split: first `+0.4433` over `9` rows, second
  `+0.2667` over `49` rows. This is the cleanest fresh replay batch in this
  sequence, but still label-mining evidence only.
- Those 58 labels were merged:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_w4p6n_all_p70_after_p60_mc512_20260617/`.
  Merge result: replay replacements `58`, positives `19`, negatives `21`,
  gray `18`; total replay-label counts in that artifact are positives `184`,
  negatives `332`, gray `204`.
- Two follow-up heads were trained on that merged artifact:
  - `outputs/training/hu_turn2_stage8c_topk_confirm_fire_w4p6n_labels_w4p6n_weight_20260617/`
    with replay weights positive `4.0`, negative `6.0`.
  - `outputs/training/hu_turn2_stage8c_topk_confirm_fire_w4p6n_labels_w2p5n_weight_20260617/`
    with replay weights positive `2.0`, negative `5.0`.
- Comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_w4p6n_labels_comparison_20260617/`.
  Result:
  - `preferred_all_p60_w4p6n` remained preferred for replay triage.
  - `w4p6n_labels_w2p5n_weight` improved AP/AUC to `0.3513 / 0.7444` and
    threshold 0.8 precision to `0.500` over `22` selected rows, but top3
    precision fell to `0.667`.
  - `w4p6n_labels_w4p6n_weight` was weaker (`0.3310 / 0.7296`, top10
    precision `0.500`).
  - No model in this sweep is runtime-approved.
- Fresh-target availability after this point:
  - For `w4p6n_labels_w2p5n_weight`, probability `>=0.80` selected `178`
    rows but all were already observed; probability `>=0.70` selected `466`
    rows with only `1` fresh replay target.
  - For `preferred_all_p60_w2p5n`, probability `>=0.80` had `0` fresh replay
    targets and probability `>=0.70` had only `2`.
  - This means the current 9,285-row distillation pool is nearly exhausted in
    the high-confidence region.
- Current interpretation:
  - Replay-label weighting can improve triage locally, and `w4p6n` found a
    clean fresh batch, but repeatedly feeding labels back into the same small
    pool does not yet produce a deployable selector.
  - The best role for the current heads is replay-target mining.
  - Next meaningful progress requires a new candidate pool or a structural
    model/loss change; simply mining more high-probability rows from this
    same pool has diminishing returns.
  - Runtime gate, T2/P2 fixed policy, production, T1, and 50k teacher remain
    `No-Go`.

## 2026-06-17 continuation: delta-head ranker check

- To test a more rank/listwise-oriented direction, two Stage8c delta heads were
  trained from
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_w4p6n_all_p70_after_p60_mc512_20260617/topk_confirm_replay_labeled_distillation_rows.jsonl`.
  Both used feature mode `opportunity_proxy_plus_preconfirm_meta`, split mode
  `source_seed`, and excluded `topk_confirm_topk_empty`.
  - Policy delta head:
    `outputs/training/hu_turn2_stage8c_delta_policy_w4p6n_labels_20260617/`;
    model `models/hu_turn2_stage8c_delta_policy_w4p6n_labels.pt`.
    It treats rejected/no-fire rows as policy delta `0`, so it can score all
    rows.
  - Observed delta head:
    `outputs/training/hu_turn2_stage8c_delta_observed_w4p6n_labels_20260617/`;
    model `models/hu_turn2_stage8c_delta_observed_w4p6n_labels.pt`.
    It is diagnostic only because it only materializes observed/replay rows.
- Ranker comparison:
  `outputs/training/hu_turn2_stage8c_prediction_ranker_comparison_delta_w4p6n_labels_20260617/`.
  Result:
  - `policy` is a deployable ranker input and looked strong on existing test
    labels (`top5 mean +13.0908`, positive rate `0.800`; `top10 mean +9.6644`,
    positive rate `0.800`).
  - `fire_probability` also remained useful (`top10 mean +8.4547`).
  - `observed` and combos using observed labels are diagnostic only and must
    not be used as runtime gates.
- The policy-delta ranker was then tested with fresh independent replay instead
  of trusting the existing-label screen. Top300 policy rows were extracted from
  the ranker artifact:
  `outputs/training/hu_turn2_stage8c_ranker_replay_targets_policy_delta_top300_after_w4p6n_labels_20260617/`.
  This yielded `100` replay-ready targets after excluding `174` already
  observed rows.
- MC512 replay completed:
  `outputs/evals/hu_turn2_stage8c_policy_delta_top300_after_w4p6n_labels_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_policy_delta_top300_after_w4p6n_labels_mc512_analysis_20260617/`.
  Result: `100 / 100` OK, mean delta `-0.0797`, 95% CI
  `[-0.2859, +0.1265]`, hard negatives `47`, safe LCB196 positives `18`,
  max loss `4.4809`. Seat split was not favorable: first `-0.0404`, second
  `-0.1190`.
- Interpretation:
  - The policy delta head looked good on existing labels but did not generalize
    to fresh policy top300 replay targets.
  - Policy delta top300 should be treated as a hard-negative source, not a
    promoted candidate generator.
- The 100 policy-delta replay labels were merged:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_policy_delta_top300_mc512_20260617/`.
  Merge result: replacements `100`, positives `18`, negatives `47`, gray `35`;
  total replay-label counts in that artifact are positives `202`, negatives
  `379`, gray `239`.
- A follow-up fire head was trained with replay weights positive `2.0`,
  negative `5.0`:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_policy_delta_top300_labels_w2p5n_weight_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_policy_delta_top300_labels_w2p5n_weight.pt`.
  It did not improve triage: test AP/AUC `0.3430 / 0.7069`, top10 precision
  `0.500`, threshold 0.8 precision `0.452`.
- Comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_policy_delta_top300_labels_comparison_20260617/`.
  Preferred triage run remains `preferred_all_p60_w4p6n`. The conservative
  `w4p6n_labels_w2p5n_weight` remains a possible validation candidate, but
  this is still only training-smoke evidence.
- Current interpretation:
  - Delta/rank regression alone did not solve the selector problem.
  - The current small pool repeatedly shows the same pattern: screens look good
    on already observed labels, but fresh independent replay is much noisier
    and often worse.
  - Next progress should be a new candidate pool or a stronger structural
    objective with a true heldout replay set; do not promote policy delta or
    start T1/50k/production from this.
- Verification:
  - `python -m pytest tests\test_hu_turn2_stage8c_delta_head_training.py tests\test_hu_turn2_stage8c_prediction_ranker_comparison.py tests\test_hu_turn2_stage8c_ranker_replay_targets.py tests\test_hu_turn2_stage8c_ranker_replay_analysis.py tests\test_hu_turn2_stage8c_topk_replay_label_merge.py tests\test_hu_turn2_stage8c_risk_head_training.py tests\test_hu_turn2_stage8c_fire_head_training_comparison.py tests\test_hu_turn2_stage8c_fire_selector_analysis.py tests\test_hu_turn2_stage8c_fire_selector_replay_targets.py -p no:cacheprovider`
    passed with `64 passed`.

## 2026-06-17 continuation: stratified replay pool smoke

- Because the high-confidence rows in the current 9,285-row pool are mostly
  exhausted, a new extractor was added:
  `src/ofc_regular/extract_hu_turn2_stage8c_stratified_replay_targets.py`.
  It samples unknown rows across ranker-score buckets instead of only taking
  the very top rows. It refuses non-runtime rankers by default and marks all
  outputs as replay targets only, not runtime evidence.
- Unit tests:
  `tests/test_hu_turn2_stage8c_stratified_replay_targets.py`.
- Smoke target extraction used the latest policy-delta-merged distillation
  artifact and the ranker comparison top rows:
  `outputs/training/hu_turn2_stage8c_stratified_replay_targets_policy_fire_predicted_smoke_20260617/`.
  Settings:
  - rankers: `policy`, `fire_probability`, `predicted_delta`
  - splits: `test`, `val`
  - seats: `first`, `second`
  - score buckets: `5`
  - targets per bucket: `4`
  - max total: `80`
- Extraction result: `67` replay-ready targets. This confirms the stratified
  extractor can still find fresh replay targets where simple high-probability
  screens are mostly exhausted.
- MC512 replay completed locally:
  `outputs/evals/hu_turn2_stage8c_stratified_policy_fire_predicted_smoke_mc512_20260617/`;
  analysis:
  `outputs/evals/hu_turn2_stage8c_stratified_policy_fire_predicted_smoke_mc512_analysis_20260617/`.
  Result: `67 / 67` OK, overall mean delta `-0.2492`, 95% CI
  `[-0.4504, -0.0480]`, hard negatives `38`, safe LCB196 positives `6`.
  Ranker split:
  - `policy`: `6` rows, mean `+0.3359`, CI `[+0.0227, +0.6491]`
  - `predicted_delta`: `40` rows, mean `-0.1394`, CI `[-0.3595, +0.0808]`
  - `fire_probability`: `21` rows, mean `-0.6255`, CI
    `[-1.0600, -0.1910]`
- Interpretation:
  - The stratified pool is useful as hard-negative mining and coverage
    exploration.
  - It is not a positive performance signal. The negative overall mean and the
    weak `fire_probability` slice reinforce that current model-score screens
    are not reliable runtime gates.
- The 67 replay labels were merged:
  `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_stratified_policy_fire_predicted_mc512_20260617/`.
  Merge result: replacements `67`, positives `6`, negatives `38`, gray `23`;
  total replay-label counts in that artifact are positives `208`, negatives
  `417`, gray `262`.
- A conservative follow-up fire head was trained:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_stratified_policy_fire_predicted_labels_w2p5n_weight_20260617/`.
  Model:
  `models/hu_turn2_stage8c_topk_confirm_fire_stratified_policy_fire_predicted_labels_w2p5n_weight.pt`.
- Comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_stratified_policy_fire_predicted_comparison_20260617/`.
  Result:
  - preferred triage run remains `preferred_all_p60_w4p6n`
  - stratified-label model: test AP/AUC `0.3488 / 0.7138`, top10 precision
    `0.500`, threshold 0.8 precision `0.440` over `25` selected rows
  - runtime gate remains `No-Go`
- Current conclusion:
  - Stratified extraction is now available for building less biased replay
    batches.
  - This specific stratified batch is mainly a hard-negative update, not a
    promotion signal.
  - Continue toward a broader heldout/replay-first candidate pool or a
    structural selector objective. Do not promote current fire heads, do not
    start T1, do not launch 50k, and do not treat this as production evidence.

## 2026-06-17 continuation: new source-log pipeline smoke

- To verify that Stage8c can move beyond the exhausted 9,285-row pool, fresh
  TopK+confirm decision logs were generated locally with the current objective:
  - default FL EV `10.227020614683454`
  - hidden-discard-safe T3 continuation `stage3_reference_default`
  - Stage8b used only as a candidate generator
  - confirm/rerank deltas treated as gate diagnostics, not performance
    evidence
- First smoke:
  `outputs/evals/hu_turn2_stage8c_new_source_log_smoke_20260617/`.
  Config:
  `k5/mc32/d0/se0/confirm64/cse1/pd0/bydelta/seat=first`,
  seed `2026061801`, `20` paired seeds. Output:
  - runtime decisions: `40`
  - overrides: `0`
  - cancellation clean: `yes`
  - distillation output:
    `outputs/training/hu_turn2_stage8c_new_source_log_smoke_distillation_20260617/`
  - distillation rows: `11`
  - positives: `0`
  - negatives: `11`
  - note: `seat=first` wastes roughly half the decisions through
    `seat_not_allowed`, so it is not a good source-log generation setting.
- Second smoke with no seat restriction:
  `outputs/evals/hu_turn2_stage8c_new_source_log_seatall_50_smoke_20260617/`.
  Config:
  `k5/mc32/d0/se0/confirm64/cse1/pd0/bydelta`,
  seed `2026061802`, `50` paired seeds. Output:
  - runtime decisions: `100`
  - overrides: `2`
  - realized per-fire delta: `0.0000`
  - confirm delta mean on fired: `+2.9677`
  - cancellation clean: `yes`
  - negative-predicted-delta override blocker: `no`
  - decision: `No-Go` as performance evidence
- Distillation from the unrestricted smoke:
  `outputs/training/hu_turn2_stage8c_new_source_log_seatall_50_distillation_20260617/`.
  Output:
  - training rows: `43`
  - positives: `0`
  - negatives: `43`
  - recommended use counts:
    - `topk_confirm_realized_loss`: `2`
    - `topk_confirm_rejected`: `7`
    - `topk_confirm_topk_empty`: `34`
  - seats: first `18`, second `25`
  - skipped counts:
    - `candidate_same_as_baseline`: `36`
    - `no_candidate`: `21`
- Interpretation:
  - The new-source-log path works end to end:
    evaluator -> `runtime_decisions.jsonl` -> distillation rows.
  - Seat-unrestricted configs are better for source-log generation than
    `seat=first` filters.
  - This 50-pair smoke is not large enough and has no positive labels, so it
    should not be mixed as an improvement claim.
  - For the next useful run, generate a larger source log with no seat
    restriction and target realized overrides or risk-veto audit rows, then
    replay/label before any selector promotion.
- A slightly larger no-seat-restriction source-log smoke was then run with an
  early stop target of `10` realized overrides:
  `outputs/evals/hu_turn2_stage8c_new_source_log_seatall_target10_smoke_20260617/`.
  Config:
  `k5/mc32/d0/se0/confirm64/cse1/pd0/bydelta`,
  seed `2026061803`, hard max `250` paired seeds,
  `target_realized_overrides_per_seed=10`.
  It stopped after `48` paired seeds:
  - runtime decisions: `96`
  - overrides: `10`
  - realized per-fire delta: `+3.5227`
  - aggregate EV/hand: `+0.3669`
  - seed-level 95% CI: `[-0.1917, +0.9256]`
  - confirm delta mean on fired: `+3.4964`
  - non-fired cancellation clean: `yes`
  - negative-predicted-delta override blocker: `no`
  - realized loss count: `1`
  This is encouraging as a source-log setting but is still only `10` fired
  decisions and is not adoption evidence.
- Distillation from the target10 smoke:
  `outputs/training/hu_turn2_stage8c_new_source_log_seatall_target10_distillation_20260617/`.
  Output:
  - training rows: `45`
  - positives: `6`
  - negatives: `39`
  - recommended use counts:
    - `topk_confirm_realized_positive`: `6`
    - `topk_confirm_realized_loss`: `4`
    - `topk_confirm_rejected`: `10`
    - `topk_confirm_topk_empty`: `25`
  - seat split: first `27`, second `18`
  - fired rows: `10`, realized delta mean `+3.5227`
- Updated next step:
  - Use this unrestricted `k5/mc32/confirm64/cse1/pd0/bydelta` family as a
    candidate for broader source-log generation.
  - Size the next run by realized fired decisions, not only paired games.
    A useful local/VM target is at least `50` realized fires before drawing
    selector conclusions.
  - Keep the output as new-source distillation/replay-label material only.
    Runtime gate, T2/P2 fixed policy, production, T1, and 50k teacher remain
    `No-Go`.

## 2026-06-17 continuation: target50 new-source run

- The unrestricted `k5/mc32/confirm64/cse1/pd0/bydelta` source-log setting was
  scaled to a `50` realized-override target:
  `outputs/evals/hu_turn2_stage8c_new_source_log_seatall_target50_20260617/`.
  Inputs:
  - seed `2026061804`
  - hard max `1,500` paired seeds
  - `target_realized_overrides_per_seed=50`
  - T3 continuation `stage3_reference_default`
  - default FL EV `10.227020614683454`
- Run result:
  - stopped at `515` paired seeds
  - runtime decisions: `1,030`
  - overrides: `50`
  - override rate: `4.85%`
  - realized per-fire delta: `+1.0045`
  - aggregate EV/hand: `+0.0488`
  - 95% CI: `[-0.0572, +0.1547]`
  - confirm delta mean on fired: `+3.8378`
  - non-fired cancellation clean: `yes`
  - negative-predicted-delta overrides: `0`
  - realized loss count: `6`
- Interpretation:
  - This is a useful source-log generation setting. It produced enough fired
    rows for a small training/material audit and did not show the previous
    negative-predicted-delta safety blocker.
  - It is not adoption evidence. The CI still crosses zero, and per-fire gain
    is far lower than the confirm diagnostic mean, so the winner's-curse
    warning still applies.
- Distillation:
  `outputs/training/hu_turn2_stage8c_new_source_log_seatall_target50_distillation_20260617/`.
  Output:
  - input decisions: `1,030`
  - training rows: `524`
  - positives: `10`
  - negatives: `514`
  - recommended use counts:
    - `topk_confirm_realized_positive`: `10`
    - `topk_confirm_realized_loss`: `40`
    - `topk_confirm_rejected`: `88`
    - `topk_confirm_topk_empty`: `386`
  - fired rows in distillation: `50`
  - fired realized delta mean: `+1.0045`
- A mixed-source fire-head smoke was trained by combining:
  - `outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_stratified_policy_fire_predicted_mc512_20260617/topk_confirm_replay_labeled_distillation_rows.jsonl`
  - `outputs/training/hu_turn2_stage8c_new_source_log_seatall_target50_distillation_20260617/topk_confirm_distillation_rows.jsonl`
  Output:
  `outputs/training/hu_turn2_stage8c_topk_confirm_fire_plus_target50_source_w2p5n_weight_20260617/`;
  model:
  `models/hu_turn2_stage8c_topk_confirm_fire_plus_target50_source_w2p5n_weight.pt`.
- Comparison:
  `outputs/training/hu_turn2_stage8c_fire_head_plus_target50_source_comparison_20260617/`.
  Result:
  - preferred triage run remains `preferred_all_p60_w4p6n`
  - mixed-source model: test AP/AUC `0.3465 / 0.7195`
  - top10 precision `0.400`
  - threshold 0.8 precision `0.391` over `23` selected rows
  - runtime gate remains `No-Go`
- Current conclusion:
  - New-source generation is now working and can produce fired positives and
    hard negatives under the current FL EV / hidden-discard-safe continuation.
  - Simply appending one target50 source log to the current fire-head training
    does not improve the selector.
  - Next useful work should either generate multiple target50 source logs for
    a true heldout source split, or change the selector objective so it can
    learn from replay/source-log rows without overfitting the old 9,285-row
    pool.
  - Production, P2 fixed policy, T1, and 50k teacher remain `No-Go`.

## 2026-06-17 continuation: target50 multi-source heldout

- Two more target50 source logs were generated with the same unrestricted
  `k5/mc32/confirm64/cse1/pd0/bydelta` setting:
  - `outputs/evals/hu_turn2_stage8c_new_source_log_seatall_target50_seed2026061805_20260617/`
  - `outputs/evals/hu_turn2_stage8c_new_source_log_seatall_target50_seed2026061806_20260617/`
- Both reached `50` realized overrides with clean cancellation and no
  negative-predicted-delta safety blocker.
- Source results:
  - seed `2026061804`: `515` paired, override rate `4.85%`,
    realized per-fire `+1.0045`, EV/hand `+0.0488`,
    CI `[-0.0572, +0.1547]`, losses `6`.
  - seed `2026061805`: `444` paired, override rate `5.63%`,
    realized per-fire `-0.8491`, EV/hand `-0.0478`,
    CI `[-0.2600, +0.1644]`, losses `13`.
  - seed `2026061806`: `493` paired, override rate `5.07%`,
    realized per-fire `+1.1509`, EV/hand `+0.0584`,
    CI `[-0.0731, +0.1899]`, losses `8`.
- Distillation outputs:
  - `outputs/training/hu_turn2_stage8c_new_source_log_seatall_target50_distillation_20260617/`
    rows `524`, positives `10`, negatives `514`.
  - `outputs/training/hu_turn2_stage8c_new_source_log_seatall_target50_seed2026061805_distillation_20260617/`
    rows `454`, positives `13`, negatives `441`.
  - `outputs/training/hu_turn2_stage8c_new_source_log_seatall_target50_seed2026061806_distillation_20260617/`
    rows `492`, positives `17`, negatives `475`.
- Source-heldout training smokes were run with fixed source groups, using the
  three target50 source logs plus the current replay-labeled pool. Summary:
  `outputs/training/hu_turn2_stage8c_source_heldout_target50_summary_20260617/`.
  Fixed test metrics:
  - test source `2026061804`: AP/AUC `0.3129 / 0.8102`.
  - test source `2026061805`: AP/AUC `0.2131 / 0.5826`.
  - test source `2026061806`: AP/AUC `0.2710 / 0.5952`.
  Threshold behavior was unstable:
  - threshold `0.5` estimated realized delta/row:
    - `2026061804`: `+0.3567`
    - `2026061805`: `-0.3569`
    - `2026061806`: `+0.2040`
  - threshold `0.7` selected too few rows to be reliable and still varied by
    source.
- Interpretation:
  - Target50 source generation works and gives useful fired positives/losses.
  - Source-to-source variance is large. The current fire-head selector can look
    useful on one source and fail on another.
  - This confirms that source-level heldout must be the default validation
    unit for Stage8c selector work.
  - Selector promotion remains `No-Go`. The next improvement should be either
    more target50 sources plus source-heldout selection, or a stronger
    source-robust objective. Do not proceed to production, P2 fixed, T1, or
    50k teacher from these results.

## 2026-06-17 continuation: source-heldout analyzer

- Added a reusable source-heldout analysis CLI:
  `python -m ofc_regular.analyze_hu_turn2_stage8c_source_heldout`.
- Purpose:
  - treat source seed as the validation unit;
  - prevent a selector from being promoted just because it looks good on one
    target50 source;
  - keep selector training smoke evidence separate from runtime, P2, T1,
    production, or 50k teacher decisions.
- The CLI was run on the three fixed-source target50 training smokes:
  - `fixed_1804_test=outputs/training/hu_turn2_stage8c_topk_confirm_fire_plus_target50_sources_fixed_1804_test_20260617`
  - `fixed_1805_test=outputs/training/hu_turn2_stage8c_topk_confirm_fire_plus_target50_sources_fixed_1805_test_20260617`
  - `fixed_1806_test=outputs/training/hu_turn2_stage8c_topk_confirm_fire_plus_target50_sources_fixed_1806_test_20260617`
- Output:
  `outputs/training/hu_turn2_stage8c_source_heldout_target50_analysis_cli_20260617/`.
- Automated decision:
  - selector promotion: `No-Go`
  - reason: `no_threshold_positive_across_sources`
  - production / P2 fixed / T1 / 50k: `No-Go`
- Key threshold stability:
  - threshold `0.5`: selected `124` rows, aggregate estimated EV/row
    `+0.0629`, but one heldout source (`2026061805`) had negative estimated
    EV/row `-0.3569`; therefore `No-Go`.
  - threshold `0.7`: aggregate EV/row was positive, but it selected too few
    rows (`14` total; zero sources met the minimum source selection count);
    therefore `No-Go`.
  - thresholds `0.8` and `0.9`: underfire; `No-Go`.
- New tests:
  `tests/test_hu_turn2_stage8c_source_heldout_analysis.py`.
- Verification:
  `python -m pytest tests/test_hu_turn2_stage8c_source_heldout_analysis.py -p no:cacheprovider`
  passed with `4 passed`.

## 2026-06-18 continuation: T3 joint exact teacher foundation

- Added a new T3 strengthening foundation:
  `python -m ofc_regular.hu_turn3_joint_exact_teacher`.
- Purpose:
  - evaluate each legal T3 action on a 9-card hero board;
  - enumerate or sample the hero's final 3-card T4 future;
  - solve T4 with the existing exact final-turn solver;
  - aggregate joint EV plus diagnostic row/outcome statistics.
- Output schema:
  `hu_turn3_joint_exact_v1`.
- Primary target:
  `joint_ev_after_t4_exact`.
- Diagnostic-only fields:
  - row category counts for top/middle/bottom;
  - bust rate;
  - FL entry rate;
  - top/middle/bottom royalty means;
  - final exact action counts.
- Important limitation:
  - This is not yet a full simultaneous HU game-tree exact solver.
  - If the opponent board is complete, scores are HU terminal scores against
    that opponent board.
  - If the opponent board is incomplete, the final-turn exact score is
    standalone for the hero board. Use this as teacher/diagnostic material, not
    as production evidence.
- Hidden-discard compatibility:
  - The module accepts mined-state `visible_dead_cards` that include opponent
    public board cards and normalizes them so only extra/private dead cards are
    used for future-deck exclusion.
- Smoke output:
  - `outputs/hu_turn3_joint_exact_smoke_teacher.jsonl`
  - `outputs/hu_turn3_joint_exact_smoke_summary.csv`
  - smoke used `future_samples=4` and produced ranked action EV plus joint
    outcome statistics for one fixed T3 state.
- Verification:
  - `python -m pytest tests/test_hu_turn3_joint_exact_teacher.py -p no:cacheprovider`
    passed with `5 passed`.
- Next T3 work:
  - run this teacher on mined hidden-discard T3 states;
  - compare its EV best action against Stage3 and legacy Stage7 choices;
  - train a new HU T3 action-value model only after the exact/joint teacher
    output is validated on a source-heldout state set.

## 2026-06-18 continuation: T3 joint exact reference comparison

- Added a comparison CLI:
  `python -m ofc_regular.analyze_hu_turn3_joint_exact_references`.
- Purpose:
  - read `hu_turn3_joint_exact_v1` teacher JSONL;
  - compare joint-exact EV for recorded `selection.baseline_index`,
    `selection.hu_index`, and optional `selection.compare_hu_index`;
  - report baseline regret, HU-selection regret, and
    `delta_hu_vs_baseline`.
- Output artifacts:
  - `joint_exact_reference_rows.csv`
  - `joint_exact_reference_summary.json`
  - `joint_exact_reference_summary.md`
- Smoke output:
  `outputs/hu_turn3_joint_exact_reference_smoke_analysis/`.
  Result:
  - comparable samples: `1`
  - mean HU-selection delta vs baseline: `+3.0000`
  - materially positive / negative / near tie: `1 / 0 / 0`
  - decision: `Needs larger validation`
- This is still teacher diagnostics, not production evidence.
- Next usable T3 strengthening step:
  1. mine or reuse hidden-discard HU T3 states with recorded `selection`;
  2. run `hu_turn3_joint_exact_teacher` with enough T4 futures;
  3. run `analyze_hu_turn3_joint_exact_references`;
  4. use negative `delta_hu_vs_baseline` rows as hard audit targets and
     positive rows as candidate training labels for the next HU T3 model.

## 2026-06-18 continuation: T3 joint exact pilot on mined states

- Reused mined HU T3 states:
  `outputs/hu_turn3_stage1_cycle14_mined_margin8_12_states_100.jsonl`.
- Generated joint-exact T3 teacher data:
  `outputs/hu_turn3_joint_exact_cycle14_pilot100_mc64_teacher.jsonl`.
- Command:
  `python -m ofc_regular.hu_turn3_joint_exact_teacher --input outputs\hu_turn3_stage1_cycle14_mined_margin8_12_states_100.jsonl --output outputs\hu_turn3_joint_exact_cycle14_pilot100_mc64_teacher.jsonl --summary-csv outputs\hu_turn3_joint_exact_cycle14_pilot100_mc64_summary.csv --future-samples 64 --seed 20260618 --max-states 100`
- Compared recorded HU selection vs baseline selection:
  `outputs/hu_turn3_joint_exact_cycle14_pilot100_mc64_analysis/`.
- Result:
  - comparable samples: `100`
  - mean HU-selection delta vs baseline: `+0.9832`
  - materially positive / negative / near tie: `57 / 40 / 3`
  - best is baseline / HU: `31 / 33`
  - mean baseline regret: `3.8291`
  - mean HU-selection regret: `2.8459`
  - worst HU delta vs baseline: `-12.1127`
  - best HU delta vs baseline: `+15.1447`
- Interpretation:
  - Existing HU selection has signal on this mined disagreement set, but the
    false-positive/tail-loss rate is too large for full replacement.
  - Negative `delta_hu_vs_baseline` rows should be treated as hard audit /
    hard-negative material for the next T3 model or selective gate.
  - This remains teacher diagnostics, not source-heldout or production
    evidence.
- Added hard-audit output:
  `joint_exact_hu_loss_top30.jsonl`.
- Training path smoke:
  `python -m ofc_regular.train_hu_turn3 --input outputs\hu_turn3_joint_exact_cycle14_pilot100_mc64_teacher.jsonl --model-output outputs\hu_turn3_joint_exact_cycle14_pilot100_mc64_model.pkl --metrics-output outputs\hu_turn3_joint_exact_cycle14_pilot100_mc64_train_metrics.json --model-type hgb --max-iter 60 --holdout 0.2 --seed 20260618`
- Smoke training result:
  - total samples/actions: `100 / 1983`
  - holdout samples/actions: `20 / 384`
  - holdout top1/top3: `0.85 / 1.00`
  - holdout avg_regret: `0.4365`
- Next T3 strengthening gate:
  - run this pipeline on source-separated hidden-discard T3 states, not only
    the legacy cycle14 disagreement file;
  - use `future_samples >= 128` or exact enumeration for final teacher
    labeling on selected states;
  - then train a new T3 candidate and compare against Stage3/legacy Stage7
    on source-heldout joint-exact regret before any seat-swap promotion.

## 2026-06-18 continuation: T3 multisource joint exact pilot

- Extended `hu_turn3_joint_exact_teacher`:
  - `--input` can now be repeated;
  - teacher samples preserve `source_input_path`;
  - this allows source/file-level holdout instead of only random splits.
- Added model evaluation CLI:
  `python -m ofc_regular.evaluate_hu_turn3_model`.
  It reports overall metrics, source-input breakdown, and optional per-state
  regret rows for any HU T3 model on a teacher JSONL.
- Multisource teacher:
  `outputs/hu_turn3_joint_exact_multisource400_mc64_teacher.jsonl`.
- Inputs:
  - `outputs/hu_turn3_stage1_cycle13_mined_margin6_12_states_100.jsonl`
  - `outputs/hu_turn3_stage1_cycle14_mined_margin8_12_states_100.jsonl`
  - `outputs/stage5_mistake_mining/stage3_m8_12_disagree_states_200_seed2026060701.jsonl`
- Teacher command:
  `python -m ofc_regular.hu_turn3_joint_exact_teacher --input outputs\hu_turn3_stage1_cycle13_mined_margin6_12_states_100.jsonl --input outputs\hu_turn3_stage1_cycle14_mined_margin8_12_states_100.jsonl --input outputs\stage5_mistake_mining\stage3_m8_12_disagree_states_200_seed2026060701.jsonl --output outputs\hu_turn3_joint_exact_multisource400_mc64_teacher.jsonl --summary-csv outputs\hu_turn3_joint_exact_multisource400_mc64_summary.csv --future-samples 64 --seed 20260618 --progress-every 100`
- Existing HU selection vs baseline on the 400-state teacher:
  - samples: `400`
  - mean HU-selection delta vs baseline: `+0.2692`
  - materially positive / negative / near tie: `194 / 160 / 46`
  - best is baseline / HU: `94 / 119`
  - baseline regret mean: `2.6367`
  - HU regret mean: `2.3675`
- Stage5 holdout slice only:
  - samples: `200`
  - mean HU-selection delta vs baseline: `+1.6303`
  - materially positive / negative / near tie: `125 / 51 / 24`
  - baseline regret mean: `2.6735`
  - HU regret mean: `1.0432`
- Initial source-heldout model test:
  - trained on cycle13+cycle14 only:
    `models/hu_turn3_joint_exact_stage8_seed200_mc64_extra_trees.pkl`
  - evaluated on stage5 holdout:
    `outputs/evals/hu_turn3_joint_exact_stage8_seed200_extra_trees_on_stage5_holdout200_metrics.json`
  - result: top1/top3 `0.325 / 0.475`, avg_regret `3.2446`.
  - HGB was worse on the same source-heldout: avg_regret `3.9294`.
- Interpretation:
  - cycle13+cycle14 training does not generalize to the stage5 source.
  - The old HU selection is still stronger than this tiny source-heldout
    retrain on stage5 (`1.0432` HU regret vs `3.2446` candidate regret).
  - Therefore the current small model is not a promoted T3 improvement.
- Multisource 400-state training smoke:
  - model:
    `models/hu_turn3_joint_exact_stage8_multisource400_mc64_extra_trees.pkl`
  - random holdout: top1/top3 `0.775 / 0.900`, avg_regret `0.6541`
  - evaluation on the same 400-state set by source input:
    - cycle13: avg_regret `0.0431`
    - cycle14: avg_regret `0.0000`
    - stage5: avg_regret `0.2769`
  - This proves the learning path works, but it is not source-heldout
    promotion evidence because the model saw all three sources.
- Current T3 strengthening status:
  - joint-exact teacher generation works;
  - source-aware evaluation works;
  - existing HU selection has useful signal but unsafe tail losses;
  - small retrains are not enough for source-heldout generalization.
- Next T3 step:
  - generate a larger source-separated hidden-discard T3 teacher set, ideally
    at least several thousand states per source with `future_samples >= 128`;
  - train a new model with explicit source-heldout evaluation;
  - use `joint_exact_hu_loss_top30.jsonl` and related negative rows as hard
    negative/audit data for any selective override gate.

## 2026-06-18 continuation: T3 chunk runner and MC128 check

- Added chunk runner:
  `scripts/Run-HuTurn3JointExactTeacherChunksParallel.ps1`.
- Purpose:
  - consume mined T3 state chunks;
  - run `ofc_regular.hu_turn3_joint_exact_teacher` per chunk;
  - resume already complete chunks by line count;
  - write per-chunk summary CSV;
  - merge teacher JSONL in chunk order.
- Smoke command:
  `.\scripts\Run-HuTurn3JointExactTeacherChunksParallel.ps1 -InputDir outputs\hu_turn3_joint_exact_chunk_smoke_input -OutputDir outputs\hu_turn3_joint_exact_chunk_smoke -MergedOutput outputs\hu_turn3_joint_exact_chunk_smoke\merged.jsonl -FutureSamples 8 -MaxParallel 2 -Seed 20260618`
- Smoke result:
  - chunks: `2`
  - total lines: `4`
  - downstream comparison and model evaluation both completed.
- MC128 multisource teacher:
  `outputs/hu_turn3_joint_exact_multisource400_mc128_teacher.jsonl`.
- Existing HU selection vs baseline at MC128:
  - samples: `400`
  - mean HU-selection delta vs baseline: `+0.3265`
  - materially positive / negative / near tie: `199 / 155 / 46`
  - baseline regret mean: `2.6375`
  - HU regret mean: `2.3110`
- Multisource 400-state MC128 training smoke:
  - model:
    `models/hu_turn3_joint_exact_stage8_multisource400_mc128_extra_trees.pkl`
  - random holdout: top1/top3 `0.8375 / 0.9250`, avg_regret `0.4131`
  - same-400 source breakdown:
    - cycle13 avg_regret `0.0111`
    - cycle14 avg_regret `0.0000`
    - stage5 avg_regret `0.1725`
  - This is still not promotion evidence because all sources are included in
    training.
- Source-heldout MC128 check:
  - train: cycle13+cycle14 (`200` states)
  - holdout: stage5 (`200` states)
  - model:
    `models/hu_turn3_joint_exact_stage8_seed200_mc128_extra_trees.pkl`
  - stage5 holdout result: top1/top3 `0.330 / 0.480`,
    avg_regret `3.3487`.
- Conclusion:
  - MC128 improves random holdout stability, but does not solve source
    generalization.
  - The next useful work is not another tiny threshold/model tweak. It is a
    larger, source-separated hidden-discard teacher set generated with the
    chunk runner, then source-heldout training/evaluation.
- Suggested next local/VM command after mining chunks:
  `.\scripts\Run-HuTurn3JointExactTeacherChunksParallel.ps1 -InputDir <mined_chunk_dir> -OutputDir outputs\hu_turn3_joint_exact_stage9_mc128_chunks -MergedOutput outputs\hu_turn3_joint_exact_stage9_mc128.jsonl -FutureSamples 128 -MaxParallel <workers> -Seed 20260618`

## 2026-06-18 continuation: T3 Stage9 1k MC128 candidate

- Generated a larger stage5 teacher from existing mined chunks:
  `outputs/hu_turn3_joint_exact_stage5_1k_mc128.jsonl`.
- Command:
  `.\scripts\Run-HuTurn3JointExactTeacherChunksParallel.ps1 -InputDir outputs\stage5_mistake_mining\stage3_m8_12_disagree_states_1k_chunks -OutputDir outputs\hu_turn3_joint_exact_stage5_1k_mc128_chunks -MergedOutput outputs\hu_turn3_joint_exact_stage5_1k_mc128.jsonl -FutureSamples 128 -MaxParallel 4 -Seed 20260618`
- Result:
  - chunks: `40`
  - states/samples: `1000`
  - elapsed wall time: about `238s` locally with `MaxParallel=4`
- Existing HU selection vs baseline on this 1k teacher:
  - samples: `1000`
  - mean HU-selection delta vs baseline: `+2.9918`
  - materially positive / negative / near tie: `669 / 214 / 117`
  - baseline regret mean: `3.7384`
  - HU regret mean: `0.7466`
- Holdout split:
  - train: first `800` stage5 samples
  - holdout: final `200` stage5 samples
- Existing HU selection on the final-200 holdout:
  - mean HU-selection delta vs baseline: `+2.9367`
  - baseline regret mean: `3.7160`
  - HU regret mean: `0.7793`
- Stage5-only candidate:
  - model:
    `models/hu_turn3_joint_exact_stage9_stage5_train800_mc128_extra_trees.pkl`
  - train input:
    `outputs/hu_turn3_joint_exact_stage5_train800_mc128.jsonl`
  - final-200 holdout:
    top1/top3 `0.855 / 0.975`, avg_regret `0.0680`
  - This is a large teacher-regret improvement over existing HU selection on
    the same holdout (`0.7793 -> 0.0680`).
- Reverse source check:
  - stage5-only candidate evaluated on cycle13+cycle14:
    top1/top3 `0.210 / 0.630`, avg_regret `2.8933`
  - This confirms T3 distributions are source-sensitive.
- Mixed1000 candidate:
  - train:
    cycle13+cycle14 `200` + stage5 first `800`
  - model:
    `models/hu_turn3_joint_exact_stage9_mixed1000_mc128_extra_trees.pkl`
  - final-200 stage5 holdout:
    top1/top3 `0.855 / 0.975`, avg_regret `0.0680`
  - cycle13+cycle14 evaluation:
    top1/top3 `0.960 / 1.000`, avg_regret `0.0055`
- Mixed1200 candidate:
  - train:
    cycle13+cycle14 `200` + stage5 first `800` + independent stage5 `200`
  - model:
    `models/hu_turn3_joint_exact_stage9_mixed1200_mc128_extra_trees.pkl`
  - random holdout from train file:
    top1/top3 `0.875 / 0.9667`, avg_regret `0.1843`
  - final-200 stage5 holdout:
    top1/top3 `0.855 / 0.975`, avg_regret `0.0680`
  - cycle13+cycle14 evaluation:
    top1/top3 `0.960 / 1.000`, avg_regret `0.0055`
- Current T3 candidate status:
  - `models/hu_turn3_joint_exact_stage9_mixed1200_mc128_extra_trees.pkl`
    is the best current teacher-regret candidate.
  - It is not yet a production/runtime replacement; it needs runtime wiring
    and source-heldout/seat-swap evaluation.
- Recommended next gate:
  1. add a T3 candidate config pointing at
     `models/hu_turn3_joint_exact_stage9_mixed1200_mc128_extra_trees.pkl`;
  2. evaluate it as HU T3 selective or full candidate against Stage3 and
     legacy Stage7 on hidden-discard seat-swap;
  3. if seat-swap is positive, generate a larger multi-source MC128/MC256
     teacher on spot VM and retrain a wider model.

## 2026-06-18 continuation: T3 Stage9 runtime smoke

- Added evaluation presets:
  - `configs/hu_turn3_stage9_joint_exact_mixed1200_full_candidate.json`
  - `configs/hu_turn3_stage9_joint_exact_mixed1200_selective_m1.json`
  - `configs/hu_turn3_stage9_joint_exact_mixed1200_selective_m2.json`
- Added decision log flags to `evaluate_model_set_matchup`:
  - `--hu-turn3-decision-log-a`
  - `--hu-turn3-decision-log-b`
- Full HU T3 replacement smoke vs Stage3 reference:
  - command used Stage9 as `--hu-turn3-a` with margin `0`
  - `100` paired seeds
  - EV/hand: `-0.2939`
  - 95% CI: `[-1.2289, +0.6412]`
  - conclusion: full replacement is not currently attractive.
- Selective Stage9 over Stage3 reference:
  - candidate model:
    `models/hu_turn3_joint_exact_stage9_mixed1200_mc128_extra_trees.pkl`
  - reference model:
    `models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt`
  - `hu_turn3_reference_min_margin = 0.0`
- m1 smoke vs Stage3 reference:
  - `200` paired seeds / `400` hands
  - EV/hand: `+0.2378`
  - 95% CI: `[-0.3187, +0.7944]`
  - decisions: `400`
  - overrides: `52`
  - override rate: `13.0%`
  - no-override reasons:
    - `same_as_stage3`: `157`
    - `below_stage7_margin`: `191`
- m2 smoke vs Stage3 reference:
  - `200` paired seeds / `400` hands
  - EV/hand: `+0.2303`
  - 95% CI: `[-0.2711, +0.7318]`
  - decisions: `400`
  - overrides: `29`
  - override rate: `7.25%`
- m1 smoke vs legacy Stage7 m5/r10:
  - `100` paired seeds / `200` hands
  - EV/hand: `+0.1050`
  - 95% CI: `[-0.3179, +0.5279]`
- Interpretation:
  - Teacher-regret improvement does not transfer as full replacement.
  - Selective override over Stage3 reference is positive in smoke but still
    statistically unresolved.
  - m1 is the main larger-eval candidate; m2 is the conservative comparison.
  - No production/default change yet.
- Next gate:
  - run larger paired seat-swap for m1 and m2, using decision logs;
  - target at least `1000` paired seeds per config before promotion talk;
  - if m1/m2 remain positive, generate larger multi-source MC128/MC256 teacher
    on spot VM and retrain.

## 2026-06-18 continuation: T3 Stage9 larger seat-swap result

- Added `--seed-stride` to `evaluate_model_set_matchup` so larger checks do
  not depend on adjacent seeds only.
- m1 larger seat-swap:
  - config: `hu_turn3_stage9_joint_exact_mixed1200_selective_m1`
  - command used `--games 1000 --seed 2026061901 --seed-stride 17`
  - EV/hand: `-0.2711`
  - 95% CI: `[-0.5136, -0.0287]`
  - decisions: `2000`
  - overrides: `304`
  - override rate: `15.2%`
  - no-override reasons:
    - `same_as_stage3`: `719`
    - `below_stage7_margin`: `977`
  - decision: `Rejected`.
- m2 larger seat-swap:
  - config: `hu_turn3_stage9_joint_exact_mixed1200_selective_m2`
  - command used `--games 1000 --seed 2026061901 --seed-stride 17`
  - EV/hand: `-0.1453`
  - 95% CI: `[-0.3796, +0.0890]`
  - decisions: `2000`
  - overrides: `122`
  - override rate: `6.1%`
  - no-override reasons:
    - `same_as_stage3`: `719`
    - `below_stage7_margin`: `1159`
  - decision: `No-Go`.
- Interpretation:
  - Stage9 looked strong under joint-exact teacher regret but did not transfer
    to hidden-discard seat-swap.
  - The small 200-paired positive smoke was not stable.
  - The failure pattern is consistent with a teacher/runtime distribution
    mismatch rather than simply too little teacher holdout accuracy.
- Current T3 status:
  - Do not promote Stage9 full replacement.
  - Do not promote Stage9 m1/m2 selective override.
  - Stage7/current T3 remains the safer runtime baseline for now.
- Next T3 direction:
  - diagnose fired Stage9 losses using the decision logs;
  - compare fired actions against joint-exact replay on the actual runtime
    states, not only mined teacher states;
  - use those runtime fired false positives as hard negatives before training
    another Stage9b candidate.

## 2026-06-20 continuation: T2 Stage9 component-tail risk features

- Current T2 candidate generator remains:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- Current best T2 runtime candidate remains diagnostic, not production:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- C9+C10 combined evidence:
  - fired decisions: `800`
  - EV/hand: about `+0.0427`
  - per-fire realized delta: about `+4.84`
  - top-loss tail remains large, so P2/production/T1 are still `No-Go`.
- MC4096 replay of all `121` realized-loss fired rows found only `3` local
  EV hard negatives. Most top losses are therefore not locally bad T2 actions;
  they are whole-game trajectory risk where the local candidate-vs-baseline EV
  remains positive or gray.
- A whole-game risk head trained on the `121` loss labels improved over the
  top30 smoke but remains `No-Go` as a runtime veto:
  - val AUC about `0.648`
  - test AUC about `0.607`
  - holdout AP below target.
- Implemented new confirm-rollout component-tail logging:
  - paired total delta now includes `lt0_rate`, `le_neg6_rate`,
    `le_neg12_rate`, `le_neg20_rate`;
  - paired component summaries are now logged under
    `confirm_paired_delta_summary.component_delta_summaries`;
  - components: `terminal_score`, `royalty_delta`, `fl_delta`,
    `line_score_delta`, `scoop_delta`, `foul_delta`;
  - risk-head feature modes that include runtime tail metadata now consume
    these component-tail summaries when present.
- Smoke artifact:
  `outputs/evals/hu_turn2_stage9_component_tail_smoke/runtime_decisions.jsonl`
  confirms component summaries are emitted by TopK confirm.
- Verification:
  - `python -m pytest tests/test_hu_turn2_teacher_data.py tests/test_hu_turn2_stage8b_topk_runtime.py tests/test_hu_turn2_stage8c_risk_head_training.py -q -p no:cacheprovider`
    passed.
  - `python -m py_compile src\ofc_regular\hu_turn2_teacher_data.py src\ofc_regular\evaluate_hu_turn2_stage8b_topk_mc_rerank.py src\ofc_regular\train_hu_turn2_stage8c_risk_head.py`
    passed.
- Next T2 direction:
  - generate a fresh C11-style TopK confirm run with component-tail logging;
  - rebuild risk/trajectory labels from that component-rich runtime log;
  - retrain the risk/veto head using component-tail features;
  - only then run fixed-threshold seat-swap. Do not go to 50k teacher or T1
    until a component-tail veto/gate improves realized per-fire tail risk.

## 2026-06-20 continuation: T2 Stage9 C12 component-tail 30-seed run

- C12 TopK confirm run:
  `regular-hu-t2-stage9-run11-c12-k3-cse2-component-tail-30seed-20260620-001`.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  T3 `stage7_m5_r10`, with `hu_turn3_min_margin=5.0` and
  `hu_turn3_reference_min_margin=10.0`.
- Aggregate:
  - completed shards: `30/30`
  - decisions: `165084`
  - realized fires: `1500`
  - non-fired cancellation: clean; non-fired delta sum/max both `0`
  - EV/hand: `+0.03963`
  - per-fire realized delta: `+4.3619`
  - per-fire CI95 low: `+3.6441`
  - evidence decision: `Pass`
- Component-tail completeness:
  - confirm rows: `5876`
  - component rows: `5876`
  - missing component summaries: `0`
  - components:
    `terminal_score`, `royalty_delta`, `fl_delta`, `line_score_delta`,
    `scoop_delta`, `foul_delta`.
- Realized whole-game loss target set:
  `outputs/evals/hu_turn2_stage9_run11_c12_component_tail_counterfactual_loss_targets/topk_counterfactual_loss_targets.jsonl`.
  - rows: `1500`
  - realized losses: `239`
  - realized non-losses: `1261`
  - component rows: `1500/1500`
- Risk/veto head training on C12 improved over C11 but remains `No-Go` for
  runtime adoption:
  - best model among the first C12 sweep:
    `runtime_tail_h16`
  - test AP: `0.2958`
  - test ROC-AUC: `0.6429`
  - threshold 0.5 precision/recall: `0.206 / 0.525`
  - threshold 0.7 precision/recall: `0.500 / 0.150`
- Offline veto checks:
  - NN veto improves all-split EV slightly, but not robustly on test split.
  - raw component-tail vetoes also show only small test-set improvements on
    very small veto counts.
  - Therefore no C12 risk head or raw veto threshold is adoption-ready.
- Current T2 status:
  - TopK+confirm itself remains promising and statistically positive.
  - A runtime risk/veto layer is not solved yet.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

### C20/C21 selector and h64 threshold target-50 comparison

- Purpose:
  compare the validated C17 h32 fire-selector against the new h64
  fire-selector thresholds before any larger T2 scaling.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Candidate generator:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- C17 old selector:
  `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`
  with threshold `0.2`.
- C20/C21 h64 selector:
  `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`.
- Seeds:
  `2026063001,2026063002,2026063003`.
- Target:
  `50` realized fires per seed.
- Comparison output:
  `outputs/evals/hu_turn2_stage9_c21_h64_threshold_comparison/comparison_summary.md`.

| selector | paired | fires | fire rate | EV/hand | EV 95% CI | per-fire | per-fire 95% CI | losses | p95 loss | max loss |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| C17 old h32 t0.2 | 8083 | 150 | 0.9279% | +0.0432 | [+0.0227, +0.0637] | +4.6524 | [+2.4443, +6.8605] | 24 | 21.2270 | 37.2270 |
| C20 h64 t0.5 | 13142 | 150 | 0.5707% | +0.0390 | [+0.0260, +0.0520] | +6.8357 | [+4.5510, +9.1204] | 24 | 16.2270 | 27.2270 |
| C21 h64 t0.35 | 10132 | 150 | 0.7402% | +0.0428 | [+0.0264, +0.0593] | +5.7830 | [+3.5601, +8.0059] | 22 | 18.2270 | 27.2270 |
| C21 h64 t0.4 | 11238 | 150 | 0.6674% | +0.0381 | [+0.0230, +0.0531] | +5.7066 | [+3.4496, +7.9636] | 25 | 22.2270 | 27.2270 |

- Interpretation:
  - all four target-50 runs passed with clean non-fired cancellation and
    `150` realized fires;
  - C17 old h32 t0.2 remains the best EV/hand baseline on this comparison;
  - C21 h64 t0.35 is the best h64 threshold and is close to C17 on EV/hand
    while reducing tail loss (`max_loss` 27.2270 vs 37.2270);
  - C20 h64 t0.5 has the strongest per-fire quality and lowest p95 loss, but
    the lower fire rate costs aggregate EV/hand;
  - C21 h64 t0.4 is dominated by t0.35 in this sample.
- Current recommendation:
  - keep C17 old h32 t0.2 as the current EV/hand baseline;
  - keep C21 h64 t0.35 as the main safety-oriented replacement candidate;
  - next validation should be a same-seed target-100 or target-150 head-to-head
    between C17 old h32 t0.2 and C21 h64 t0.35;
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

### C18 TopK+confirm fire-selector distillation refresh

- Purpose:
  distill the positive C12/C13/C16/C17 TopK+confirm overlay behavior into a
  stronger pre-confirm fire selector, while keeping the realized fired
  whole-game delta as the only performance evidence.
- Prep implementation:
  `src/ofc_regular/prepare_hu_turn2_stage8c_topk_distillation.py` now streams
  decision logs instead of loading all JSONL rows into memory.
- Distillation source logs:
  - C12 component-tail run
  - C13 independent-heldout run
  - C16 t0.2 target-50 run
  - C17 t0.2 fresh-seed target-100 run
- Distillation artifact:
  `outputs/training/hu_turn2_stage9_c18_c12_c13_c16_c17_topk_confirm_distillation/`.
- Dataset:
  - source logs: `66`
  - input rows: `383982`
  - training rows: `33169`
  - positives: `1396`
  - negatives: `31773`
  - fired rows: `3441`
  - rejected rows: `9728`
  - topk-empty negatives: `20000`
  - realized fired-row mean delta: `+4.2200`
- h32 smoke:
  `models/hu_turn2_stage9_c18_topk_confirm_fire_opp_preconfirm_h32_smoke.pt`.
  - trainable rows after excluding topk-empty negatives: `13169`
  - test AP: `0.3689`
  - test ROC AUC: `0.7729`
  - result: smoke `Pass`
- h64/32 CUDA training:
  `models/hu_turn2_stage9_c18_topk_confirm_fire_opp_preconfirm_h64.pt`.
  - best epoch: `17`
  - elapsed: `201s`
  - test AP: `0.4230`
  - test ROC AUC: `0.8197`
  - all-row AP: `0.4837`
  - all-row ROC AUC: `0.8549`
  - top-100 training-table precision: `0.81`
- Local runtime smoke, fixed config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Candidate-generator model:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- Fire-selector model:
  `models/hu_turn2_stage9_c18_topk_confirm_fire_opp_preconfirm_h64.pt`.
- Local threshold smoke results on seeds `2026062501,2026062502`:

| fire selector threshold | paired | fires | fire rate | realized per-fire | per-fire CI low | estimated EV/hand | EV CI low | p95 loss | max loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1827 | 20 | 0.0055 | +5.6341 | +0.1173 | +0.0308 | +0.0006 | 14.5114 | 24.2270 |
| 0.7 | 2539 | 20 | 0.0039 | +10.8181 | +5.5868 | +0.0426 | +0.0220 | 0.7000 | 14.0000 |
| 0.8 | 3043 | 17 | 0.0028 | +10.7138 | +3.6291 | +0.0299 | +0.0101 | 16.0454 | 24.2270 |

- Interpretation:
  - h64/32 materially improved the fire-selector training metrics over the
    h32 smoke.
  - In small local runtime smoke, threshold `0.7` was the best balance:
    higher quality than `0.5`, more fires than `0.8`, and clean cancellation.
  - This is still underpowered: only `20` fired decisions for t0.7.
  - Next useful step is a GCP fresh-seed target-100 validation for h64 t0.7,
    with h64 t0.5 as a secondary comparison only if budget allows.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

### C18 h64 t0.7 fresh-seed target-100 validation

- GCP run:
  `regular-hu-t2-stage9-c18-h64-t07-target100-20260621-001`.
- Output:
  `outputs/evals/hu_turn2_stage9_c18_h64_t07_target100_aggregate/`.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Candidate-generator model:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- Fire-selector model:
  `models/hu_turn2_stage9_c18_topk_confirm_fire_opp_preconfirm_h64.pt`.
- Fire-selector threshold:
  `0.7`.
- Seeds:
  `2026062601,2026062602,2026062603`.
- Target:
  `100` realized fires per seed.
- Aggregate:
  - completed shards: `3/3`
  - paired seeds: `41779`
  - decisions: `83558`
  - realized fires: `300`
  - fired replay-ready rate: `100%`
  - non-fired cancellation: clean; non-fired nonzero count `0`
  - EV/hand from realized fired whole-game delta: `+0.0200`
  - EV/hand 95% CI: `[+0.0131, +0.0269]`
  - per-fire realized delta: `+5.5721`
  - per-fire 95% CI: `[+3.6554, +7.4887]`
  - confirm diagnostic mean on fired: `+7.3361`
  - confirm minus realized per-fire gap: `+1.7640`
  - realized losses: `64`
  - p95 loss: `25.2270`
  - max loss: `55.4540`
  - negative predicted-delta override count: `0`
- Seed-level EV/hand:
  - seed `2026062601`: `+0.0164`
  - seed `2026062602`: `+0.0147`
  - seed `2026062603`: `+0.0291`
- Interpretation:
  - h64/t0.7 is a validation `Pass`; it preserves clean cancellation and
    all three fresh seeds are positive.
  - Compared with C17 t0.2, h64/t0.7 has lower aggregate EV/hand
    (`+0.0200` vs `+0.0307`) but higher realized per-fire quality
    (`+5.5721` vs `+3.7727`) and lower firing rate.
  - The larger confirm-realized gap means confirm MC remains only a gate
    diagnostic; realized fired whole-game delta remains the performance
    metric.
  - This result supports continuing the distillation path, but still does not
    authorize production/P2.
  - Next useful work is to either:
    - validate a slightly lower h64 threshold or a combined h64 + old t0.2
      selector to recover fire rate without losing per-fire quality; or
    - train a direct lightweight candidate ranker from the C18 distilled rows
      to reduce the expensive TopK+confirm path.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.
- Next T2 direction:
  - evaluate TopK+confirm without veto on a larger fixed heldout if runtime cost
    is acceptable; or
  - collect another independent component-tail heldout specifically to validate
    a very small fixed veto rule, not to tune it; or
  - train a selector/ranker to choose safer candidates before confirm rather
    than adding a weak post-confirm veto.

### C18 h64 t0.5 fresh-seed target-100 validation

- GCP run:
  `regular-hu-t2-stage9-c18-h64-t05-target100-20260621-001`.
- Output:
  `outputs/evals/hu_turn2_stage9_c18_h64_t05_target100_aggregate/`.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Candidate-generator model:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- Fire-selector model:
  `models/hu_turn2_stage9_c18_topk_confirm_fire_opp_preconfirm_h64.pt`.
- Fire-selector threshold:
  `0.5`.
- Seeds:
  `2026062601,2026062602,2026062603`.
- Target:
  `100` realized fires per seed.
- Aggregate:
  - completed shards: `3/3`
  - paired seeds: `33595`
  - decisions: `67190`
  - realized fires: `300`
  - fired replay-ready rate: `100%`
  - non-fired cancellation: clean; non-fired nonzero count `0`
  - EV/hand from realized fired whole-game delta: `+0.0229`
  - EV/hand 95% CI: `[+0.0147, +0.0311]`
  - per-fire realized delta: `+5.1272`
  - per-fire 95% CI: `[+3.2970, +6.9574]`
  - confirm diagnostic mean on fired: `+6.6960`
  - confirm minus realized per-fire gap: `+1.5688`
  - realized losses: `59`
  - p95 loss: `24.2270`
  - max loss: `55.4540`
  - negative predicted-delta override count: `0`
- Seed-level EV/hand:
  - seed `2026062601`: `+0.0154`
  - seed `2026062602`: `+0.0173`
  - seed `2026062603`: `+0.0359`
- Comparison against the currently validated C17 and C18 variants:

| variant | paired | fires | EV/hand | EV CI | per-fire | per-fire CI | losses | p95 loss | max loss |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| C17 old selector t0.2 | `18439` | `300` | `+0.0307` | `[+0.0173, +0.0441]` | `+3.7727` | `[+2.1271, +5.4183]` | `49` | `24.2270` | `35.2270` |
| C18 h64 t0.7 | `41779` | `300` | `+0.0200` | `[+0.0131, +0.0269]` | `+5.5721` | `[+3.6554, +7.4887]` | `64` | `25.2270` | `55.4540` |
| C18 h64 t0.5 | `33595` | `300` | `+0.0229` | `[+0.0147, +0.0311]` | `+5.1272` | `[+3.2970, +6.9574]` | `59` | `24.2270` | `55.4540` |

- Interpretation:
  - h64/t0.5 is a validation `Pass`; it keeps all three seeds positive and
    non-fired cancellation clean.
  - h64/t0.5 recovers some fire rate versus h64/t0.7 and improves EV/hand
    (`+0.0229` vs `+0.0200`), but it still does not beat C17 t0.2
    (`+0.0307`).
  - C18 h64 raises per-fire quality versus C17, but the lower fire rate leaves
    aggregate EV/hand behind C17.
  - The shared `55.4540` max-loss tail in h64/t0.5 and h64/t0.7 is worse than
    C17's `35.2270` max loss and must be audited before any stronger claim.
  - C18 h64 should not replace C17 as the main runtime path. The next useful
    direction is C19: keep C17 as the fire-rate-preserving main selector and
    test whether C18 h64, a dedicated loss head, or a small top-loss rule can
    veto only the worst C17 candidates.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

### C19 loss-audit preparation

- Output:
  `outputs/evals/hu_turn2_stage9_c19_loss_audit_prep/`.
- Inputs:
  - C17 t0.2 target-100 `failure_top30.jsonl` from all three shards.
  - C18 h64 t0.7 target-100 `failure_top30.jsonl` from all three shards.
  - C18 h64 t0.5 target-100 `failure_top30.jsonl` from all three shards.
- Artifacts:
  - `top_loss_rows.csv`
  - `top_loss_overlap.csv`
  - `c19_loss_audit_prep_summary.md`
- Summary:

| run | top-loss rows | unique hands | largest row loss | rows loss>=25 | rows loss>=35 | median row loss | mean confirm delta | mean confirm z | rank1 rows | rank>5 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C17 t0.2 | `90` | `90` | `31.2270` | `12` | `0` | `10.0000` | `4.1497` | `3.4778` | `54` | `5` |
| C18 h64 t0.7 | `90` | `90` | `30.2270` | `13` | `0` | `10.0000` | `6.4694` | `5.3081` | `77` | `2` |
| C18 h64 t0.5 | `90` | `90` | `46.2270` | `13` | `1` | `10.0000` | `5.9947` | `4.8642` | `72` | `3` |

- Cross-run top-loss overlap:
  - rows appearing in at least two variants: `70`
  - rows appearing in all three variants: `0`
  - C18 h64 t0.5 and t0.7 share many of the same highest-loss hands because
    they used the same seed block and the same h64 selector, only with a
    threshold change.
- Immediate read:
  - C18's worst top-loss rows are not removed by confirm128+cse2; several have
    positive confirm deltas and high confirm z.
  - h64/t0.5 improves EV/hand over h64/t0.7, but the worse tail and lower
    aggregate EV/hand than C17 make C18 h64 a poor full replacement.
  - C19 should test `C17 main selector + top-loss veto`, not `C18 h64 replaces
    C17`.
  - The veto source should be validated as a blocker of realized bad C17
    candidates, not as another threshold-tuned selector on the same rows.

Mechanical C19 runtime smoke:

- Output:
  `outputs/evals/hu_turn2_stage9_c19_c17main_riskveto_audit_t05_path_smoke/`.
- Purpose:
  prove that the C17 main TopK+confirm path can run with a post-confirm
  Stage8c risk model in audit-only mode.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Candidate-generator model:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- Risk veto model:
  `models/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_t080_mc512_plus_fire_veto_aligned_2001_2101.pt`.
- Risk threshold:
  `0.5`, audit-only.
- Continuation:
  explicit `stage7_m5_r10`.
- Smoke size:
  seed `2026062701`, `500` games max, target `1` realized override.
- Result:
  - execution: `Pass`
  - elapsed: about `21s`
  - paired seeds: `24`
  - decisions: `48`
  - realized overrides: `1`
  - non-fired cancellation: clean
  - would-veto rows: `1`
  - actual veto rows: `0` because audit-only
- Interpretation:
  - This is path validation only, not model-quality evidence.
  - A meaningful C19 run must collect enough realized fires and enough
    would-veto rows to estimate veto utility from realized whole-game deltas.
  - The local target-15-per-seed attempt was too slow for interactive local
    execution, so C19 evidence should be gathered with the existing Spot VM
    shard wrapper.

C19 Spot VM audit launch:

- Run:
  `regular-hu-t2-stage9-c19-c17main-riskveto-audit-t05-target100-20260621-001`.
- Purpose:
  C17 main TopK+confirm path with the Stage8c local-EV-negative risk head in
  audit-only mode. This measures would-veto utility without blocking overrides.
- GCP:
  project `ofc-solver-485418`, bucket
  `pokerhu-ofc-solver-485418-training`.
- Shards/VMs:
  `3` shards on `e2-highcpu-8`.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Seeds:
  `2026062801,2026062802,2026062803`.
- Games/targets:
  `12000` games per seed, target `100` realized overrides per seed.
- Models:
  - candidate generator:
    `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`
  - risk veto:
    `models/hu_turn2_stage8c_local_ev_negative_opportunity_lookahead_preconfirm_t080_mc512_plus_fire_veto_aligned_2001_2101.pt`
- Risk settings:
  threshold `0.5`, audit-only `true`.
- Continuation:
  explicit `stage7_m5_r10`.
- Initial status:
  VM creation succeeded and all three instances were `RUNNING`; no status files
  were present immediately after startup.
- Expected decision use:
  - if would-veto rows have positive veto utility and preserve C17 fire-rate
    EV, test an actual-veto C19b run;
  - if would-veto utility is flat/negative or sparse, train a stronger loss
    head from the audited top-loss rows instead of widening 50k teacher.

C19 Spot VM audit result:

- Received and aggregated output:
  `outputs/evals/hu_turn2_stage9_c19_c17main_riskveto_audit_t05_target100_aggregate/`.
- Downloaded run package:
  `outputs/gcp_runs/regular-hu-t2-stage9-c19-c17main-riskveto-audit-t05-target100-20260621-001/`.
- Execution:
  - completed shards: `3 / 3`
  - missing shards: `0`
  - no residual matching GCP instances after receive
- Primary metric validity:
  - primary metric source: `realized_fired_whole_game_delta`
  - non-fired cancellation: clean
  - non-fired nonzero count: `0`
  - replay-ready decisions/fires: `31804 / 300`
- C17-main performance on these fresh C19 seeds:
  - paired seeds: `15902`
  - decisions: `31804`
  - realized fires: `300`
  - override rate: `0.9433%` of decisions
  - realized-est EV/hand: `+0.0466`
  - seed-mean EV/hand: `+0.0470`
  - seed-mean 95% CI: `[+0.0383, +0.0557]`
  - realized per-fire delta: `+4.9451`
  - per-fire 95% CI: `[+3.4760, +6.4143]`
  - confirm diagnostic mean on fired: `+4.8592`
  - confirm minus realized per-fire gap: `-0.0859`
  - realized losses: `39`
  - p95 loss: `17.2270`
  - max loss: `30.2270`
- Position:
  - first seat only fires: `300`
  - first-seat estimated EV/hand: `+0.0933`
  - second seat fires: `0` due to `seat=first` gate
- Risk-veto audit:
  - would-veto count: `300`
  - actual veto count: `0` because audit-only
  - realized candidate delta on would-veto rows: `+4.9451`
  - veto utility per veto: `-4.9451`
  - estimated veto utility per hand: `-0.0466`
  - risk-veto adoption decision: `No-Go`
- Interpretation:
  - C19 is execution-pass and validation-pass for the C17 main path, but
    risk-veto adoption is a clear `No-Go`.
  - The local-EV-negative risk model at threshold `0.5` marks profitable C17
    fires as veto candidates; an actual-veto C19b run would block value.
  - Do not integrate this risk head into runtime.
  - C17 main remains the strongest currently validated T2 overlay evidence,
    but it is still an offline TopK+confirm/search overlay, not P2/production.
  - Next useful work is either C17 practical-runtime/canary design and
    distillation, or a different whole-game-risk model with better labels and
    deployable features. Do not start `50k teacher`, `T1`, or production/P2
    fixing from the C19 risk-veto result.

C20 C17/C19 TopK-confirm distillation prep and smoke:

- C17+C19 combined prep:
  `outputs/training/hu_turn2_stage9_c20_c17_c19_topk_confirm_distillation_prep/`.
  - input decision rows: `68682`
  - training-ready rows: `7123`
  - positive rows: `252`
  - negative rows: `6871`
  - fired candidate rows: `600`
  - confirm-rejected rows: `1523`
  - sampled topk-empty easy negatives: `5000`
- C19-only incremental prep:
  `outputs/training/hu_turn2_stage9_c20_c19_topk_confirm_distillation_prep/`.
  - rows: `6138`
  - positive rows: `124`
  - negative rows: `6014`
- C20 h64 training:
  - output:
    `outputs/training/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64/`
  - model:
    `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`
  - training input:
    old C18 C12/C13/C16/C17 distillation pack plus the new C19-only pack
  - target mode: `topk_confirm_fire`
  - feature mode: `opportunity_proxy_plus_preconfirm_meta`
  - split mode: `source_seed`
  - hidden: `64`
  - trainable rows: `14307`
  - positives / negatives: `1520 / 12787`
  - excluded `topk_confirm_topk_empty`: `25000`
  - metrics:
    - val AP/AUC: `0.4275 / 0.8166`
    - test AP/AUC: `0.4511 / 0.8353`
    - all AP/AUC: `0.4732 / 0.8548`
  - Compared with C18 h64, test AP/AUC improved slightly while validation AP
    dropped. Treat this as a candidate for runtime smoke only, not as proof of
    improvement.
- C20 h64 runtime smoke, threshold `0.8`:
  - output:
    `outputs/evals/hu_turn2_stage9_c20_fire_selector_h64_t08_target5_smoke/`
  - seed: `2026062901`
  - max paired seeds: `1000`
  - realized fires: `4`
  - target `5` fires was not reached
  - non-fired cancellation: clean
  - fire-selector pass rate: `3.46%`
  - estimated EV/hand from fired deltas: `+0.0191`
  - per-fire delta: `+9.5568`
  - interpretation: path executes, but `t0.8` underfires and is not evidence.
- C20 h64 runtime smoke, threshold `0.5`:
  - output:
    `outputs/evals/hu_turn2_stage9_c20_fire_selector_h64_t05_target5_smoke/`
  - seed: `2026062901`
  - paired seeds: `615`
  - realized fires: `5`
  - non-fired cancellation: clean
  - fire-selector pass rate: `16.23%`
  - estimated EV/hand from fired deltas: `+0.0491`
  - per-fire delta: `+12.0908`
  - per-fire CI: `[+0.4228, +23.7588]`
  - realized losses: `0`
  - interpretation: execution and sizing look acceptable for the next
    target-50 comparison, but `5` fires on one seed is still only a smoke.
- C20 next decision:
  - Do not replace C17 with C20 yet.
  - Next useful validation is a fresh-seed target-50 comparison:
    C17 old selector `t0.2` vs C20 h64 `t0.5`, same config
    `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`,
    explicit `stage7_m5_r10`, and primary metric still realized fired
    whole-game delta with clean cancellation.
  - If C20 h64 `t0.5` cannot preserve C17 fire-rate or per-fire quality at
    target-50, keep C17 and continue distillation/ranker work instead of
    widening `50k teacher`.
- C20 target-50 fresh-seed GCP comparison launched:
  - shared seeds: `2026063001,2026063002,2026063003`
  - games per seed: `8000`
  - target realized overrides per seed: `50`
  - VM setup: `3` x `e2-highcpu-8`
  - T3 continuation: explicit `stage7_m5_r10`
  - C17 old selector run:
    `regular-hu-t2-stage9-c20-c17old-t02-target50-20260621-001`
    - selector:
      `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`
    - threshold: `0.2`
  - C20 h64 run:
    `regular-hu-t2-stage9-c20-h64-t05-target50-20260621-001`
    - selector:
      `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`
    - threshold: `0.5`
  - Initial status:
    both runs had `3 / 3` shards reporting `running` after launch.
  - Receive commands after completion:
    - `.\scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -RunName "regular-hu-t2-stage9-c20-c17old-t02-target50-20260621-001" -OutputDir "outputs/evals/hu_turn2_stage9_c20_c17old_t02_target50_aggregate"`
    - `.\scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -RunName "regular-hu-t2-stage9-c20-h64-t05-target50-20260621-001" -OutputDir "outputs/evals/hu_turn2_stage9_c20_h64_t05_target50_aggregate"`

C20 target-50 fresh-seed GCP comparison result:

- C17 old selector aggregate:
  `outputs/evals/hu_turn2_stage9_c20_c17old_t02_target50_aggregate/`.
- C20 h64 selector aggregate:
  `outputs/evals/hu_turn2_stage9_c20_h64_t05_target50_aggregate/`.
- Comparison artifact:
  `outputs/evals/hu_turn2_stage9_c20_target50_selector_comparison/`.
- Execution:
  - both runs completed `3 / 3` shards.
  - no matching GCP instances remained after receive.
  - both aggregates had clean non-fired cancellation:
    non-fired nonzero count `0`, non-fired delta sum `0`.
  - primary metric remained realized fired whole-game paired delta; confirm
    deltas are diagnostics only.

| selector | paired | fires | fire rate | EV/hand | EV CI | per-fire | per-fire CI | losses | p95 loss | max loss |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| C17 old h32 t0.2 | `8,083` | `150` | `0.9279%` | `+0.0432` | `[+0.0227, +0.0637]` | `+4.6524` | `[+2.4443, +6.8605]` | `24` | `21.2270` | `37.2270` |
| C20 h64 t0.5 | `13,142` | `150` | `0.5707%` | `+0.0390` | `[+0.0260, +0.0520]` | `+6.8357` | `[+4.5510, +9.1204]` | `24` | `16.2270` | `27.2270` |

- Seed-level summary:
  - C17 old h32 t0.2:
    - seed `2026063001`: EV/hand `+0.0716`, per-fire `+6.7809`
    - seed `2026063002`: EV/hand `+0.0407`, per-fire `+4.8718`
    - seed `2026063003`: EV/hand `+0.0212`, per-fire `+2.3045`
  - C20 h64 t0.5:
    - seed `2026063001`: EV/hand `+0.0571`, per-fire `+9.2436`
    - seed `2026063002`: EV/hand `+0.0296`, per-fire `+6.2409`
    - seed `2026063003`: EV/hand `+0.0328`, per-fire `+5.0227`
- Interpretation:
  - both selectors are validation `Pass` at target-50 and all three seeds are
    positive.
  - C20 h64 is stricter: lower fire rate, higher per-fire value, lower p95 and
    max realized loss.
  - C17 old remains the current EV/hand baseline because it fired more often
    and had slightly higher aggregate EV/hand on this same-seed target-50
    comparison.
  - Do not replace C17 with C20 h64 yet. Use C20 h64 as a stricter comparison
    candidate and consider a larger target-100/150 comparison if tail-risk
    reduction is worth the lower fire rate.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

C21 h64 threshold relaxation launch:

- Purpose:
  test whether the C20 h64 fire selector can recover C17-like fire rate while
  preserving its higher per-fire quality and lower tail loss.
- Pre-launch log audit from the C20 h64 t0.5 decision logs:
  - t0.5 passed `489` fire-selector decisions and blocked `1553`.
  - lowering to t0.4 would add up to `195` previously blocked candidate
    decisions to the MC/confirm path.
  - lowering to t0.35 would add up to `313` previously blocked candidate
    decisions and roughly matches the C17 old selector fire-rate scale if the
    downstream confirm pass rate holds.
- Shared settings:
  - seeds: `2026063001,2026063002,2026063003`
  - games per seed: `8000`
  - target realized overrides per seed: `50`
  - config:
    `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`
  - candidate generator:
    `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`
  - fire selector:
    `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`
  - T3 continuation: explicit `stage7_m5_r10`
  - VM setup: `3` x `e2-highcpu-8` per threshold.
- t0.35 run:
  `regular-hu-t2-stage9-c21-h64-t035-target50-20260621-001`.
- t0.4 run:
  `regular-hu-t2-stage9-c21-h64-t040-target50-20260621-001`.
- Initial status:
  both runs had `3 / 3` instances `RUNNING`; no result shards yet.
- Receive commands after completion:
  - `.\scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -RunName "regular-hu-t2-stage9-c21-h64-t035-target50-20260621-001" -OutputDir "outputs/evals/hu_turn2_stage9_c21_h64_t035_target50_aggregate"`
  - `.\scripts\Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1 -RunName "regular-hu-t2-stage9-c21-h64-t040-target50-20260621-001" -OutputDir "outputs/evals/hu_turn2_stage9_c21_h64_t040_target50_aggregate"`

## 2026-06-20 continuation: T2 Stage9 C13 independent heldout

- C13 independent heldout run:
  `regular-hu-t2-stage9-run11-c13-k3-cse2-independent-heldout-30seed-20260620-001`.
- Purpose:
  verify that the C12 TopK+confirm result reproduces on non-overlapping seeds
  before any production or P2 decision.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  T3 `stage7_m5_r10`.
- Aggregate:
  - completed shards: `30/30`
  - paired seeds: `82395`
  - decisions: `164790`
  - realized fires: `1491`
  - override rate: `0.9048%` over all decisions
  - non-fired cancellation: clean
  - EV/hand: `+0.03680`
  - EV/hand CI95: `[+0.03053, +0.04307]`
  - per-fire realized delta: `+4.0675`
  - per-fire CI95: `[+3.3745, +4.7606]`
  - evidence decision: `Pass`
- Combined C12+C13 fixed-config evidence:
  - paired seeds: `164937`
  - decisions: `329874`
  - realized fires: `2991`
  - non-fired cancellation: clean
  - EV/hand: `+0.03822`
  - EV/hand CI95: `[+0.03370, +0.04274]`
  - per-fire realized delta: `+4.2152`
  - per-fire CI95: `[+3.7163, +4.7141]`
  - realized loss count: `475`
  - p95 loss: `22.2270`
  - max loss: `42.4540`
- Latency from C12/C13 shard logs:
  - all decisions mean: about `0.19-0.20s`
  - reranked decisions mean: about `2.20-2.36s`
  - override-fired decisions mean: about `3.32-3.51s`
  - fired decision p95: about `4.4-4.6s`
  - this is strong enough for offline/search-overlay or canary advisory use,
    but too slow to treat as a normal low-latency policy without distillation.
- Position interpretation:
  - The config is `seat=first`, so all fired decisions are first-position only.
  - First-position estimated EV/hand is `+0.07644` on first-position hands.
  - HU all-hand EV is half of that, `+0.03822`, because second-position hands
    are intentionally unchanged.
- Current T2 interpretation:
  - TopK+confirm itself is now strongly reproduced across C12 and C13 under
    the explicit `stage7_m5_r10` continuation setting.
  - Because this is not the hidden-discard safety default, do not read C12/C13
    as a broad production proof for all current T2 paths.
  - The post-confirm risk/veto layer is still `No-Go`; do not attach the weak
    C12 risk head to runtime.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`
    until runtime cost and implementation safety are settled.
- Next T2 direction:
  - measure practical runtime cost/latency of this TopK+confirm config;
  - decide whether it is acceptable as an online search overlay or should be
    distilled into a faster T2 model/selector;
  - if used online, add a conservative runtime preset with explicit
    `seat=first` scope and keep it canary-only before any production decision.

## 2026-06-20 continuation: T2 Stage9 C14 fire-selector runtime wiring

- Implemented a validation-only Stage8c/Stage9 fire-selector runtime path in
  `src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py`.
- New CLI options:
  - `--stage8c-fire-selector-model`
  - `--stage8c-fire-selector-threshold`
  - `--stage8c-fire-selector-audit-only`
- Semantics:
  - the fire selector is a pre-confirm candidate filter;
  - high probability keeps a TopK candidate, unlike the post-confirm risk head
    where high probability vetoes;
  - confirm MC and realized whole-game paired delta remain the evaluation gates;
  - fire-selector probability is not a realized performance metric.
- Initial model wired for C14:
  `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`.
- Tests:
  - `python -m py_compile src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py tests/test_hu_turn2_stage8b_topk_runtime.py`
  - `python -m pytest tests/test_hu_turn2_stage8b_topk_runtime.py -p no:cacheprovider`
  - result: `15 passed`.
- Smoke outputs:
  - `outputs/evals/hu_turn2_stage9_c14_fire_selector_smoke/`
    confirms the new CLI fields and summary/log wiring under the C12/C13
    `stage7_m5_r10` continuation setting.
  - `outputs/evals/hu_turn2_stage9_c14_fire_selector_smoke_pass/`
    confirms the real fire-selector model is evaluated before MC:
    `stage8c_fire_selector_evaluated_count=3`,
    `stage8c_fire_selector_passed_count=3`.
- Current interpretation:
  - execution pass only;
  - no C14 EV claim yet;
  - do not promote to production/P2;
  - next evaluation is a small fixed C14 threshold sweep comparing
    thresholds such as `0.5`, `0.7`, and `0.85` against the C12/C13
    no-selector baseline.

### C14 small threshold comparison

- Output:
  `outputs/evals/hu_turn2_stage9_c14_fire_selector_small_compare/`.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`, matching C12/C13.
- Fire-selector model:
  `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`.
- First 20-paired probe:
  - no-selector: `0` fires
  - t0.1: `4/7` selector candidates passed, `0` fires
  - t0.2: `2/7` selector candidates passed, `0` fires
  - t0.3: `1/7` selector candidates passed, `0` fires
  - t0.5/t0.7/t0.85: `0/7` selector candidates passed, `0` fires
  - max observed fire-selector probability: `0.4295`
- Target-5 probe on the same seed block:
  - no-selector: `295` paired seeds, `5` fires, EV/hand `+0.0906`,
    realized per-fire `+10.6908`, mean latency `154.9ms`
  - t0.1: `295` paired seeds, `5` fires, EV/hand `+0.0906`,
    same fired set as no-selector, `72/87` selector candidates passed,
    mean latency `416.0ms`
  - t0.2: `295` paired seeds, `5` fires, EV/hand `+0.0906`,
    same fired set as no-selector, `54/87` selector candidates passed,
    mean latency `578.9ms`
- Interpretation:
  - `0.5/0.7/0.85` are too high for this runtime distribution.
  - `0.1/0.2` preserved the same five target fires in this seed block, but did
    not improve EV and added local inference overhead.
  - The current fire-selector is execution-pass but value-unproven. It should
    not be used as a production gate.
  - Next useful work is either a larger GCP threshold comparison sized by
    fired count, or retraining/calibrating the fire-selector so its probability
    scale and inference cost are useful in runtime.

### C15 fire-selector batch scoring

- Implemented batch scoring for the C14 fire-selector path in
  `src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py`.
  - `LocalEvRiskScorer.predict_probabilities(rows)` now builds one feature
    matrix and performs one torch inference pass for all TopK candidates.
  - The runtime selector uses the batch method when present and falls back to
    scalar `predict_probability(row)` for compatibility.
  - Semantics are unchanged: this is still a pre-confirm candidate selector,
    not a post-confirm risk veto and not a production gate.
- Tests:
  - `python -m py_compile src/ofc_regular/evaluate_hu_turn2_stage8b_topk_mc_rerank.py tests/test_hu_turn2_stage8b_topk_runtime.py`
  - `python -m pytest tests/test_hu_turn2_stage8b_topk_runtime.py -p no:cacheprovider`
  - result: `16 passed`.
- C15 target-5 latency check:
  - output summary:
    `outputs/evals/hu_turn2_stage9_c15_fire_selector_batch_compare/c15_fire_selector_batch_compare.md`
  - t0.1:
    - C14: `295` paired, `5` fires, EV/hand `+0.0906`,
      mean latency `416.0ms`, fired latency `7700.1ms`
    - C15: `295` paired, `5` fires, EV/hand `+0.0906`,
      mean latency `124.1ms`, fired latency `2418.1ms`
  - t0.2:
    - C14: `295` paired, `5` fires, EV/hand `+0.0906`,
      mean latency `578.9ms`, fired latency `11841.6ms`
    - C15: `295` paired, `5` fires, EV/hand `+0.0906`,
      mean latency `104.8ms`, fired latency `2503.2ms`
- Interpretation:
  - batch scoring fixed most of the local fire-selector runtime overhead;
  - fired set and realized EV were unchanged on the deterministic target-5
    seed block;
  - this remains an execution/latency improvement only. The current
    fire-selector still has no demonstrated EV advantage over no-selector.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

### C16 fire-selector target-50 comparison

- Output summary:
  `outputs/evals/hu_turn2_stage9_c16_fire_selector_target50_compare/c16_fire_selector_target50_compare.md`
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Fire-selector model:
  `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`.
- Compared variants:
  - no selector
  - fire-selector threshold `0.1`
  - fire-selector threshold `0.2`
- Seeds:
  `2026062301,2026062302,2026062303`.
- Target:
  `50` realized fires per seed.
- Aggregate results:

| run | paired | fires | EV/hand | EV CI low | EV CI high | per-fire | per-fire CI low | losses | p95 loss | max loss | selector pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-selector | 8253 | 150 | +0.0398 | +0.0200 | +0.0596 | +4.3775 | +2.1986 | 27 | 22.0 | 28.2270 | 0/0 |
| t0.1 | 8307 | 150 | +0.0404 | +0.0206 | +0.0602 | +4.4791 | +2.2863 | 28 | 22.0 | 28.2270 | 2134/2616 |
| t0.2 | 8615 | 150 | +0.0454 | +0.0258 | +0.0650 | +5.2118 | +2.9585 | 27 | 22.0 | 28.2270 | 1765/2709 |

- Interpretation:
  - all variants had clean non-fired cancellation;
  - `t0.2` was best in this target-50 comparison;
  - `t0.1` was too close to no-selector to justify preference;
  - this remained validation-only because the path is an offline
    TopK+confirm overlay, not a cheap production policy.

### C17 fire-selector t0.2 fresh-seed target-100 validation

- GCP run:
  `regular-hu-t2-stage9-c17-fs-t02-target100-20260621-001`.
- Output:
  `outputs/evals/hu_turn2_stage9_c17_fs_t02_target100_aggregate/`.
- Purpose:
  validate the C16 best variant on fresh non-overlapping seeds with a larger
  fired sample.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Fire-selector:
  `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`
  with threshold `0.2`.
- Seeds:
  `2026062401,2026062402,2026062403`.
- Target:
  `100` realized fires per seed.
- Aggregate:
  - completed shards: `3/3`
  - paired seeds: `18439`
  - decisions: `36878`
  - realized fires: `300`
  - fired replay-ready rate: `100%`
  - non-fired cancellation: clean; non-fired nonzero count `0`
  - EV/hand from realized fired whole-game delta: `+0.0307`
  - EV/hand 95% CI: `[+0.0173, +0.0441]`
  - per-fire realized delta: `+3.7727`
  - per-fire 95% CI: `[+2.1271, +5.4183]`
  - confirm diagnostic mean on fired: `+5.0787`
  - confirm minus realized per-fire gap: `+1.3059`
  - realized losses: `49`
  - p95 loss: `24.2270`
  - max loss: `35.2270`
  - negative predicted-delta override count: `0`
- Weighted latency:
  - all decisions: `145.5ms`
  - reranked decisions: `2254.3ms`
  - fired decisions: `3316.3ms`
- Seed-level EV/hand:
  - seed `2026062401`: `+0.0435`
  - seed `2026062402`: `+0.0245`
  - seed `2026062403`: `+0.0249`
- Interpretation:
  - C17 is a validation `Pass`; the C16 `t0.2` signal reproduced on fresh
    seeds and all three seeds were positive.
  - The realized effect is smaller than C16 but still clearly positive by the
    fired whole-game metric.
  - Confirm MC remains a gate diagnostic only; it overstates realized per-fire
    by about `1.3` points in this run.
  - Tail losses remain material, so this is not a production/P2 decision.
  - The next useful work is not `50k teacher` or `T1`; it is either:
    - make this `seat=first` TopK+confirm overlay practical as a canary/search
      runtime with explicit latency and safety guardrails; or
    - distill the positive C12/C13/C16/C17 overlay decisions into a faster T2
      selector/ranker while preserving realized per-fire quality.
  - `production`, `P2 fixed`, `50k teacher`, and `T1 training` remain `No-Go`.

### C22 same-seed target-100: C17 old vs h64 t0.35

- Purpose:
  resolve the C21 tradeoff between C17 old h32 t0.2 and C21 h64 t0.35 using
  the same fresh seeds and `100` realized fires per seed.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Candidate generator:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- C17 old selector:
  `models/hu_turn2_stage9_c12_c13_topk_confirm_distill_opp_preconfirm_h32.pt`
  with threshold `0.2`.
- h64 selector:
  `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`
  with threshold `0.35`.
- GCP runs:
  - `regular-hu-t2-stage9-c22-c17old-t02-target100-20260621-001`
  - `regular-hu-t2-stage9-c22-h64-t035-target100-20260621-001`
- Outputs:
  - `outputs/evals/hu_turn2_stage9_c22_c17old_t02_target100_aggregate/`
  - `outputs/evals/hu_turn2_stage9_c22_h64_t035_target100_aggregate/`
  - `outputs/evals/hu_turn2_stage9_c22_c17_vs_h64_t035_target100_comparison/comparison_summary.md`
- Seeds:
  `2026063101,2026063102,2026063103`.
- Target:
  `100` realized fires per seed.

| selector | paired | fires | fire rate | EV/hand | EV 95% CI | seed CI | per-fire | per-fire 95% CI | losses | p95 loss | max loss |
|---|---:|---:|---:|---:|---|---|---:|---|---:|---:|---:|
| C17 old h32 t0.2 | 19222 | 300 | 0.7804% | +0.0165 | [+0.0041, +0.0289] | [-0.0002, +0.0361] | +2.1176 | [+0.5284, +3.7068] | 60 | 24.2270 | 42.2270 |
| h64 t0.35 | 23829 | 300 | 0.6295% | +0.0199 | [+0.0095, +0.0302] | [+0.0145, +0.0257] | +3.1544 | [+1.5098, +4.7989] | 60 | 23.2270 | 42.2270 |

- Seed-level EV/hand:
  - C17 old:
    - `2026063101`: `+0.0238`
    - `2026063102`: `+0.0303`
    - `2026063103`: `-0.0002`
  - h64 t0.35:
    - `2026063101`: `+0.0221`
    - `2026063102`: `+0.0237`
    - `2026063103`: `+0.0144`
- Interpretation:
  - both runs completed `3/3` shards with clean non-fired cancellation and
    `300` realized fires;
  - on the same seeds, h64 t0.35 beat C17 old on EV/hand (`+0.0199` vs
    `+0.0165`) and per-fire value (`+3.1544` vs `+2.1176`);
  - C17 old had one nearly flat seed, while h64 t0.35 was positive on all
    three seeds;
  - extreme tail risk is not fixed (`max_loss` `42.2270` for both), but h64
    t0.35 has a slightly better p95 loss;
  - confirm deltas still overstate realized per-fire value by about `2.8-3.1`
    points, so confirm deltas remain gate diagnostics only.
- Current recommendation:
  - promote h64 t0.35 to the preferred T2 Stage9 TopK+confirm selector
    candidate for further validation;
  - keep C17 old h32 t0.2 as the older EV baseline, but no longer prefer it
    over h64 t0.35 after this same-seed target-100 comparison;
  - next useful step is a larger h64 t0.35 validation (`target150` or
    `target200`) before any production/P2 decision.
- Status:
  - `production`: `No-Go`
  - `P2 fixed`: `No-Go`
  - `50k teacher`: `No-Go`
  - `T1 training`: `No-Go`

### C23 h64 t0.35 fresh-seed target-200 validation

- Purpose:
  stress the preferred h64 t0.35 selector after C22 with a larger fired sample
  on fresh seeds.
- Runtime config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`.
- Continuation:
  explicit `stage7_m5_r10`.
- Candidate generator:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`.
- Fire selector:
  `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`
  with threshold `0.35`.
- GCP run:
  `regular-hu-t2-stage9-c23-h64-t035-target200-20260621-001`.
- Output:
  - `outputs/evals/hu_turn2_stage9_c23_h64_t035_target200_aggregate/`
  - `outputs/evals/hu_turn2_stage9_c23_h64_t035_target200_validation/validation_summary.md`
- Seeds:
  `2026063201,2026063202,2026063203`.
- Target:
  `200` realized fires per seed.
- Aggregate:
  - completed shards: `3/3`
  - paired seeds: `49524`
  - decisions: `99048`
  - realized fires: `600`
  - fired replay-ready rate: `100%`
  - non-fired cancellation: clean; non-fired nonzero count `0`
  - EV/hand from realized fired whole-game delta: `+0.0312`
  - EV/hand 95% CI: `[+0.0242, +0.0382]`
  - seed mean EV/hand 95% CI: `[+0.0252, +0.0371]`
  - per-fire realized delta: `+5.1484`
  - per-fire 95% CI: `[+3.9977, +6.2990]`
  - confirm diagnostic mean on fired: `+5.7276`
  - confirm minus realized per-fire gap: `+0.5792`
  - realized losses: `89`
  - p95 loss: `22.2270`
  - max loss: `43.2270`
  - negative predicted-delta override count: `0`
- Seed-level EV/hand:
  - seed `2026063201`: `+0.0308`
  - seed `2026063202`: `+0.0367`
  - seed `2026063203`: `+0.0261`
- Interpretation:
  - C23 is a stronger validation `Pass`; h64 t0.35 reproduced on fresh seeds
    with `600` fired decisions and all three seeds positive.
  - Per-fire quality improved versus C22 target-100 (`+5.1484` vs `+3.1544`).
  - The confirm diagnostic bias shrank substantially in this run, but confirm
    deltas are still diagnostics only.
  - Tail risk remains material (`max_loss` `43.2270`), so this is still not a
    production/P2 decision.
- Current recommendation:
  - h64 t0.35 is now the preferred T2 Stage9 TopK+confirm selector candidate;
  - C17 old h32 t0.2 remains only as the old baseline;
  - next useful work is either a larger h64 t0.35 stress run
    (`target300`/`target500`) or productization/latency work for this offline
    TopK+confirm overlay;
  - do not start `50k teacher`, `T1 training`, `production`, or `P2 fixed` yet.

### C23 h64 t0.35 productization and latency check

- Purpose:
  quantify whether the C23-winning h64 t0.35 overlay is usable as a runtime
  policy, or whether it must be distilled/optimized first.
- Output:
  `outputs/evals/hu_turn2_stage9_c23_h64_t035_productization/productization_summary.md`.
- Aggregate latency:

| scope | count | mean runtime ms | shard p95 range ms | mean MC rerank ms | mean confirm ms |
|---|---:|---:|---:|---:|---:|
| all decisions | 99048 | 79.01 | 21.91-27.80 | 34.89 | 27.57 |
| reranked | 3015 | 2076.43 | 5221.52-5496.96 | 1146.21 | 905.80 |
| fired | 600 | 3157.94 | 5638.15-6750.89 | 1153.67 | 1980.27 |

- Runtime rates:
  - reranked decisions:
    `3015 / 99048 = 3.0440%` of all decisions, or `6.0880%` of first-seat
    decisions;
  - fired overrides:
    `600 / 99048 = 0.6058%` of all decisions, or `1.2115%` of first-seat
    decisions;
  - fire-selector evaluated:
    `15971 / 99048 = 16.1245%` of all decisions, or `32.2490%` of first-seat
    decisions.
- No-override counts:
  - `seat_not_allowed`: `49524`
  - `topk_empty`: `41686`
  - `below_fire_selector_threshold`: `4823`
  - `mc_best_is_baseline`: `1635`
  - `below_confirm_se`: `469`
  - `below_confirm_delta`: `311`
  - fired: `600`
- Interpretation:
  - strength evidence is good enough to keep h64 t0.35 as the preferred T2
    Stage9 candidate;
  - current implementation is still too heavy for real-time production: fired
    decisions average about `3.16s` and have shard p95 values above `5.6s`;
  - this is plausible for offline/advisor/canary usage if multi-second T2
    decisions are acceptable;
  - production/P2 remains `No-Go` until the heavy MC64 + confirm128 path is
    distilled or optimized and then revalidated with clean non-fired
    cancellation.
- Recommended next implementation:
  build `Stage9d` as a distillation/productization step.
  - Extract replay-ready fired, near-fired, and blocked-by-fire-selector states
    from C22/C23.
  - Preserve the measurement rule:
    confirm deltas are gate diagnostics only; realized fired whole-game paired
    delta is the performance label.
  - Train a cheap selector/ranker that approximates h64 t0.35 fire decisions
    without running MC64 + confirm128 on every reranked state.
  - Target runtime:
    all-decision mean `<25ms`, fired-decision mean `<500ms`.
  - Validate with the same C23 metric:
    realized fired whole-game paired delta, non-fired cancellation clean.
- Status:
  - `Stage9d distillation/productization`: `Go`
  - `larger target300/target500 stress`: optional, lower priority than
    productization
  - `production`: `No-Go`
  - `P2 fixed`: `No-Go`
  - `50k teacher`: `No-Go`
  - `T1 training`: `No-Go`

### Stage9d C23 distillation smoke

- Purpose:
  start converting the validated but heavy h64 t0.35 TopK+confirm overlay into
  a cheaper runtime gate.
- Code changes:
  - `prepare_hu_turn2_stage8c_topk_distillation.py` now supports
    `below_fire_selector_threshold` as a negative training group:
    `topk_confirm_fire_selector_rejected`.
  - `train_hu_turn2_stage8c_risk_head.py` maps
    `topk_confirm_fire_selector_rejected` to label `0` under
    `target-mode=topk_confirm_fire`.
  - `evaluate_hu_turn2_stage8b_topk_mc_rerank.py` now logs action payloads in
    `stage8c_fire_selector_candidates`, so future fire-selector rejected rows
    are replay/train ready.
- Important limitation:
  the already-generated C23 raw logs only stored
  `stage8c_fire_selector_candidates` as action indices, not action payloads.
  Therefore C23's `below_fire_selector_threshold` rows cannot all be recovered
  as action-level distillation negatives from the existing logs. Future runs
  after the logging patch can recover them.
- C23 distillation extraction:
  - output:
    `outputs/training/hu_turn2_stage9d_c23_h64_t035_distillation/`
  - input rows:
    `99048`
  - training rows after dedupe/sampling:
    `4380`
  - positives:
    `289`
  - negatives:
    `4091`
  - training-ready rows:
    `4380`
  - groups:
    - `topk_confirm_realized_positive`: `289`
    - `topk_confirm_realized_loss`: `311`
    - `topk_confirm_rejected`: `780`
    - `topk_confirm_topk_empty`: `3000`
  - absent from current C23 extraction:
    `topk_confirm_fire_selector_rejected`, because old C23 logs lack candidate
    action payloads for those rows.
- Smoke training:
  - command used:
    `python -m ofc_regular.train_hu_turn2_stage8c_risk_head --input-jsonl outputs\training\hu_turn2_stage9d_c23_h64_t035_distillation\topk_confirm_distillation_rows.jsonl --output-dir outputs\training\hu_turn2_stage9d_c23_h64_t035_fire_selector_smoke --model-output models\hu_turn2_stage9d_c23_h64_t035_fire_selector_smoke.pt --target-mode topk_confirm_fire --feature-mode hu_delta_plus_preconfirm_meta --epochs 8 --batch-size 256 --learning-rate 0.001 --patience 3 --seed 2026062101 --device cpu --threshold 0.5 --threshold 0.7 --threshold 0.8 --threshold 0.9 --threshold 0.95`
  - output:
    `outputs/training/hu_turn2_stage9d_c23_h64_t035_fire_selector_smoke/`
  - model:
    `models/hu_turn2_stage9d_c23_h64_t035_fire_selector_smoke.pt`
  - feature mode:
    `hu_delta_plus_preconfirm_meta`
    (chosen because Stage9d should avoid confirm128 as a runtime dependency)
  - test AP:
    `0.2534`
  - test ROC AUC:
    `0.7883`
  - risk-head smoke:
    `Pass`
- Interpretation:
  - the Stage9d data and training path is now mechanically working;
  - the smoke model is not an adoption candidate;
  - the next meaningful data run should regenerate logs after the action-payload
    patch, so `below_fire_selector_threshold` rows become usable negatives.
- Recommended next gate:
  - run a small fresh h64 t0.35 evaluation after the logging patch;
  - extract Stage9d distillation rows and verify
    `topk_confirm_fire_selector_rejected > 0`;
  - retrain the fire selector with those rows;
  - only then compare its runtime and C23-style realized fired whole-game delta.
- Status:
  - `Stage9d path`: `smoke Pass`
  - `Stage9d production candidate`: `No-Go`
  - `production`: `No-Go`
  - `P2 fixed`: `No-Go`
  - `50k teacher`: `No-Go`
  - `T1 training`: `No-Go`

### Stage9d logging-patch smoke

- Purpose:
  verify that the logging patch makes `below_fire_selector_threshold` rows
  action-recoverable for Stage9d distillation.
- Local smoke command:
  `python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank --games-per-seed 50 --target-realized-overrides-per-seed 0 --seeds 2026064101 --seed-stride 1000000 --configs "k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta" --hu-turn2-stage8b-model models\hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt --t3-continuation stage7_m5_r10 --stage8c-fire-selector-model models\hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt --stage8c-fire-selector-threshold 0.35 --device cpu --prediction-threads 1 --stage3-feature-encoder-mode rust_direct --output-dir outputs\evals\hu_turn2_stage9d_logging_patch_smoke_50 --write-decision-log --progress-every 25`
- Output:
  `outputs/evals/hu_turn2_stage9d_logging_patch_smoke_50/`
- Runtime:
  about `59s` local CPU for `50` paired hands.
- No-override counts:
  - `below_fire_selector_threshold`: `5`
  - `topk_empty`: `41`
  - `below_confirm_se`: `2`
  - `below_confirm_delta`: `1`
  - `mc_best_is_baseline`: `1`
  - `seat_not_allowed`: `50`
- Schema check:
  the first `below_fire_selector_threshold` row had
  `stage8c_fire_selector_candidates[0]` keys:
  `action`, `action_index`, `candidate_ev_rank`, `gate_probability`,
  `model_score`, `passed`, `post_t2_board`, `predicted_delta`, `probability`.
  This proves future fire-selector rejected rows are action-recoverable.
- Distillation extraction:
  - output:
    `outputs/training/hu_turn2_stage9d_logging_patch_smoke_50_distillation/`
  - input rows:
    `100`
  - training rows:
    `32`
  - `topk_confirm_fire_selector_rejected`:
    `5`
  - `topk_confirm_rejected`:
    `3`
  - `topk_confirm_topk_empty`:
    `24`
  - positive rows:
    `0`
- Combined smoke training:
  - inputs:
    - `outputs/training/hu_turn2_stage9d_c23_h64_t035_distillation/topk_confirm_distillation_rows.jsonl`
    - `outputs/training/hu_turn2_stage9d_logging_patch_smoke_50_distillation/topk_confirm_distillation_rows.jsonl`
  - output:
    `outputs/training/hu_turn2_stage9d_c23_plus_logging_patch_fire_selector_smoke/`
  - model:
    `models/hu_turn2_stage9d_c23_plus_logging_patch_fire_selector_smoke.pt`
  - rows:
    `4412`
  - positives:
    `289`
  - negatives:
    `4123`
  - target group counts:
    - `topk_confirm_fire_selector_rejected`: `5`
    - `topk_confirm_realized_loss`: `311`
    - `topk_confirm_realized_positive`: `289`
    - `topk_confirm_rejected`: `783`
    - `topk_confirm_topk_empty`: `3024`
  - test AP / ROC AUC:
    `0.2116 / 0.7429`
- Interpretation:
  - the new logging/extraction path is verified;
  - the combined smoke model is not a model-quality improvement because the
    fresh fire-selector rejected set has only `5` rows;
  - the next useful run is a larger fresh logging-patched data run whose main
    goal is to collect enough `topk_confirm_fire_selector_rejected` negatives
    for Stage9d training.
- Recommended next gate:
  run a fresh local/GCP Stage9d data run targeting at least `500-1000`
  action-recoverable fire-selector rejected rows, then retrain and validate
  the lightweight selector.

### Stage9d fire-selector rejection target smoke

- Purpose:
  add a controlled data-collection stop condition for Stage9d distillation:
  `--target-fire-selector-rejections-per-seed`.
- Local smoke command:
  `python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank --games-per-seed 200 --target-realized-overrides-per-seed 0 --target-fire-selector-rejections-per-seed 10 --seeds 2026064102 --seed-stride 1000000 --configs "k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta" --hu-turn2-stage8b-model models\hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt --t3-continuation stage7_m5_r10 --stage8c-fire-selector-model models\hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt --stage8c-fire-selector-threshold 0.35 --device cpu --prediction-threads 1 --stage3-feature-encoder-mode rust_direct --output-dir outputs\evals\hu_turn2_stage9d_fire_selector_target10_smoke --write-decision-log --progress-every 25`
- Output:
  `outputs/evals/hu_turn2_stage9d_fire_selector_target10_smoke/`
- Result:
  - paired seeds:
    `65`
  - decisions:
    `130`
  - stop reason:
    `target_fire_selector_rejections_reached`
  - `below_fire_selector_threshold`:
    `10`
- Distillation extraction:
  - output:
    `outputs/training/hu_turn2_stage9d_fire_selector_target10_smoke_distillation/`
  - training rows:
    `42`
  - `topk_confirm_fire_selector_rejected`:
    `10`
  - `topk_confirm_topk_empty`:
    `32`
  - positive rows:
    `0`
- Schema check:
  the extracted fire-selector rejected row preserved `candidate_action`,
  `stage8c_fire_selector_probability`, `predicted_delta`, `gate_probability`,
  `candidate_ev_rank`, and `model_score`.
- Status:
  - target-stop implementation:
    `Pass`
  - action-recoverable rejection extraction:
    `Pass`
  - Stage9d production candidate:
    `No-Go`
  - production / P2 fixed / 50k teacher / T1:
    `No-Go`
- Recommended next gate:
  run a larger fresh logging-patched data collection on Spot VM/GCP targeting
  `500-1000` `topk_confirm_fire_selector_rejected` rows.

### Stage9d fresh fire-selector rejected collection and retrain

- GCP run:
  `regular-hu-t2-stage9d-fsreject-t200-20260621-001`
- Purpose:
  collect action-recoverable `below_fire_selector_threshold` examples for
  Stage9d distillation.
- Runtime target:
  `--target-fire-selector-rejections-per-seed 200`
- Inputs:
  - candidate generator:
    `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`
  - old fire selector:
    `models/hu_turn2_stage9_c20_topk_confirm_fire_opp_preconfirm_h64.pt`
  - old fire selector threshold:
    `0.35`
  - T3 continuation:
    `stage7_m5_r10`
- Result:
  - completed shards:
    `3/3`
  - paired seeds:
    `6629`
  - decisions:
    `13258`
  - `below_fire_selector_threshold` rows:
    `600`
  - cancellation audit:
    clean
- Output:
  `outputs/evals/hu_turn2_stage9d_fsreject_t200_aggregate/`
- Distillation output:
  `outputs/training/hu_turn2_stage9d_fsreject_t200_distillation/`
- Distillation rows:
  - total:
    `3780`
  - positive:
    `41`
  - negative:
    `3739`
  - `topk_confirm_fire_selector_rejected`:
    `600`
  - `topk_confirm_realized_loss`:
    `41`
  - `topk_confirm_realized_positive`:
    `41`
  - `topk_confirm_rejected`:
    `98`
  - `topk_confirm_topk_empty`:
    `3000`

Stage9d retrain:

- Training inputs:
  - `outputs/training/hu_turn2_stage9d_c23_h64_t035_distillation/topk_confirm_distillation_rows.jsonl`
  - `outputs/training/hu_turn2_stage9d_fsreject_t200_distillation/topk_confirm_distillation_rows.jsonl`
- Model:
  `models/hu_turn2_stage9d_fsreject_t200_fire_selector_h64.pt`
- Output:
  `outputs/training/hu_turn2_stage9d_fsreject_t200_fire_selector_h64/`
- Rows:
  - total:
    `8160`
  - positive:
    `330`
  - negative:
    `7830`
  - `topk_confirm_fire_selector_rejected`:
    `600`
- Metrics:
  - test AP / ROC AUC:
    `0.2811 / 0.8354`
  - all AP / ROC AUC:
    `0.5223 / 0.9308`
- Interpretation:
  the retrain improves over the previous combined smoke (`test AP/AUC`
  `0.2116 / 0.7429`), but it is still only a validation candidate.

Local threshold smokes with the new selector:

- t0.8 smoke:
  - output:
    `outputs/evals/hu_turn2_stage9d_fsreject_t200_selector_t08_smoke100/`
  - paired seeds:
    `100`
  - realized overrides:
    `1`
  - EV/hand:
    `+0.0200`
  - fire-selector pass rate:
    `4.76%`
- t0.7 smoke:
  - output:
    `outputs/evals/hu_turn2_stage9d_fsreject_t200_selector_t07_smoke100/`
  - paired seeds:
    `100`
  - realized overrides:
    `0`
  - fire-selector pass rate:
    `4.17%`
- t0.5 smoke:
  - output:
    `outputs/evals/hu_turn2_stage9d_fsreject_t200_selector_t05_smoke100/`
  - paired seeds:
    `100`
  - realized overrides:
    `0`
  - fire-selector pass rate:
    `48.65%`
- Interpretation:
  t0.5 passes many candidates but MC/confirm still filters almost all of them;
  t0.8 is very conservative. A larger validation is needed before deciding
  whether this retrained selector helps productization.

Stage9d t0.5 validation run:

- GCP run:
  `regular-hu-t2-stage9d-fsreject-h64-t05-target50-20260621-001`
- Purpose:
  validate the retrained selector at threshold `0.5` with target realized
  fires.
- Target:
  `--target-realized-overrides-per-seed 50`
- Status at handoff:
  running on Spot VM.
- Production / P2 fixed / 50k teacher / T1:
  `No-Go`

Stage9d t0.5 validation result:

- GCP run:
  `regular-hu-t2-stage9d-fsreject-h64-t05-target50-20260621-001`
- Aggregate output:
  `outputs/evals/hu_turn2_stage9d_fsreject_h64_t05_target50_aggregate/`
- Completed shards:
  `3/3`
- Paired seeds:
  `12408`
- Decisions:
  `24816`
- Realized fires:
  `150`
- Runtime override rate:
  about `0.604%`
- EV/hand:
  `+0.03524`
- Realized per-fire delta:
  `+5.8297`
- Per-fire 95% CI:
  `[+3.5464, +8.1129]`
- p95 loss / max loss:
  `22.2270 / 26.2270`
- Non-fired cancellation:
  clean (`non_fired_nonzero_count = 0`)
- Seed split:
  - `2026064501`:
    EV/hand `+0.03944`, realized fires `50`, avg realized delta `+5.7209`
  - `2026064502`:
    EV/hand `+0.02992`, realized fires `50`, avg realized delta `+5.1872`
  - `2026064503`:
    EV/hand `+0.03699`, realized fires `50`, avg realized delta `+6.5809`
- Interpretation:
  this retrained Stage9d selector at threshold `0.5` preserves the C23
  validation strength and is slightly better on this target-50 validation:
  C23 h64 t0.35 was EV/hand `+0.0312`, per-fire `+5.1484`.

Stage9d t0.5 productization check:

- Output:
  `outputs/evals/hu_turn2_stage9d_fsreject_h64_t05_productization/`
- All decisions:
  mean runtime `71.93ms`, p95 `20.04ms`
- Reranked decisions:
  mean runtime `1962.20ms`, p95 `3869.27ms`
- Fired decisions:
  mean runtime `2953.56ms`, p95 `4166.87ms`
- Fire-selector blocked decisions:
  mean runtime `19.03ms`, p95 `20.37ms`
- Comparison to C23 h64 t0.35 productization:
  - all-decision mean improved from `79.01ms` to `71.93ms`
  - reranked mean improved from `2076.43ms` to `1962.20ms`
  - fired mean improved from `3157.94ms` to `2953.56ms`
- Productization decision:
  `No-Go` for real-time production. The selector improved strength and gives
  a small latency improvement, but MC64 + confirm128 still dominate fired and
  reranked states.
- Recommended next gate:
  Stage9e should distill or replace the MC/confirm stage itself. The target
  remains all-decision mean `<25ms` and fired-decision mean `<500ms` while
  preserving positive realized per-fire delta with clean non-fired cancellation.

### Stage9e direct-fire smoke

Implementation:

- Added experimental runtime flag:
  `--stage8c-fire-selector-direct-fire`
- Behavior:
  after the Stage8c fire selector filters TopK candidates, choose the highest
  selector-probability candidate directly and skip MC64/confirm128.
- Safety:
  this mode requires `--stage8c-fire-selector-model`, cannot be combined with
  `--stage8c-fire-selector-audit-only`, and is validation-only.
- Existing Stage9d behavior is unchanged unless the direct-fire flag is set.

Smoke commands used:

- Candidate generator:
  `models/hu_turn2_stage9_candidate_generator_current_fl_5k_run11_train_pairwise_highmc230.pt`
- Fire selector:
  `models/hu_turn2_stage9d_fsreject_t200_fire_selector_h64.pt`
- T3 continuation:
  `stage7_m5_r10`
- Config:
  `k3/mc64/d0/se0/confirm128/cse2/pd0/seat=first/bygate_delta`

Results:

| threshold | paired | decisions | fires | pass rate | EV/hand | realized per-fire | all mean latency | fired mean latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.95` | `200` | `400` | `0` | `0.00%` | `0.0000` | `0.0000` | not material | not material |
| `0.90` | `300` | `600` | `0` | `0.00%` | `0.0000` | `0.0000` | not material | not material |
| `0.80` | `300` | `600` | `1` | `1.03%` | `-0.0387` | `-23.2270` | `12.44ms` | `16.18ms` |
| `0.50` | `201` | `402` | `20` | `36.49%` | `-0.1205` | `-2.4227` | `12.61ms` | `16.88ms` |

Interpretation:

- Direct-fire meets the latency target mechanically:
  MC rerank and confirm latency are `0.0ms`, all-decision mean is around
  `12-13ms`, and fired-decision mean is around `16-17ms`.
- The current Stage9d fire selector is not sufficient as a direct production
  gate:
  high thresholds under-fire completely, while permissive threshold `0.50`
  fires often enough but has negative realized per-fire delta.
- Therefore, Stage9e direct-fire with the current selector is `No-Go`.

Next gate:

- Build a dedicated direct-fire selector trained on realized whole-game fired
  deltas and MC/confirm outcomes, not just the pre-confirm
  `topk_confirm_fire` label.
- Use Stage9d validated logs as positives, and include hard negatives from:
  fire-selector-passed but MC/confirm-rejected candidates, direct-fire
  false positives, top losses, and selector-passed non-fires.
- Target:
  all-decision mean `<25ms`, fired-decision mean `<500ms`, clean non-fired
  cancellation, and positive realized per-fire delta.
- Production / P2 fixed / 50k teacher / T1:
  still `No-Go`.

## HU Turn1 Stage2 Stage9f P2 Candidate-Subset Teacher Update

Context:

- T2 `stage9f_p2` is fixed only for post-acceptance experiments, not production.
- T3 continuation remains fixed to `stage7_m5_r10`.
- T1 leaf31 TopK line remains No-Go.
- Current T1 work is a restart using `hu_turn1_teacher_pilot` with
  `stage9f_p2` continuation.

All-action T1 teacher result:

- Run:
  `regular-hu-t1-stage2-stage9f-p2-pilot100-mc16-20260625-001`
- Output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_pilot100_mc16_partial/`
- Completed:
  `10` records from `2 / 20` shards before intentional abort.
- Mean / max seconds per sample:
  `117.22 / 129.07`
- Invalid action/state rows:
  `0 / 0`
- Decision:
  all-action `stage9f_p2` MC16 T1 teacher is too slow to broaden directly.

Candidate-subset implementation:

- `hu_turn1_teacher_pilot` now supports `--candidate-model` and
  `--candidate-topk`.
- GCP launcher `scripts/Start-GcpHuTurn1PilotRun.ps1` packages and forwards the
  candidate model/topK arguments.
- Top5 candidate20 MC16:
  - output:
    `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_top5_mc16/`
  - records:
    `20 / 20`
  - mean / max seconds per sample:
    `28.16 / 46.06`
  - invalid action/state rows:
    `0 / 0`
  - decision:
    execution-clean but undercovers all-action teacher best.
- Top10 candidate20 MC16:
  - run:
    `regular-hu-t1-stage2-stage9f-p2-top10-candidate20-mc16-20260625-001`
  - output:
    `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_top10_mc16/`
  - records:
    `20 / 20`
  - mean / max seconds per sample:
    `38.01 / 82.58`
  - invalid action/state rows:
    `0 / 0`
  - seat split:
    `first=10`, `second=10`
  - T2/T3 continuation:
    `stage9f_p2 / stage7_m5_r10`
  - decision:
    execution pass, but coverage is still unresolved.

Coverage audit on the 10 all-action partial records:

| K | teacher-best recall | avg TopK regret | max TopK regret |
|---|---:|---:|---:|
| `1` | `3/10` | `2.3991` | `7.1959` |
| `3` | `4/10` | `2.0838` | `6.8068` |
| `5` | `6/10` | `1.2454` | `5.3209` |
| `8` | `8/10` | `0.7912` | `5.3209` |
| `10` | `9/10` | `0.5321` | `5.3209` |

Decision:

- Candidate-subset generation is now practical, but Top10 labels should not be
  treated as complete all-action teacher labels.
- Broad 1k T1 training is still No-Go until candidate coverage is improved or
  the incomplete-label design is made explicit.
- Next useful step:
  Top15 probe, improved T1 candidate generator, or a larger cheap all-action
  audit to measure candidate coverage.

Follow-up all-action coverage audit:

- Run:
  `regular-hu-t1-stage2-stage9f-p2-allaction-audit40-mc4-20260625-001`
- Output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction_audit40_mc4/`
- Coverage artifact:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction_audit40_mc4/coverage/coverage_summary.json`
- Records:
  `40 / 40`
- Future samples:
  `4`
- Mean / max seconds per sample:
  `35.59 / 64.06`
- Invalid action/state rows:
  `0 / 0`
- Seat split:
  `first=20`, `second=20`

Coverage of `models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl`:

| K | teacher-best recall | avg TopK regret | max TopK regret |
|---|---:|---:|---:|
| `5` | `21/40` | `1.9199` | `10.3635` |
| `10` | `29/40` | `0.7585` | `7.0568` |
| `15` | `33/40` | `0.3605` | `7.0568` |
| `20` | `35/40` | `0.1403` | `3.0568` |

Updated decision:

- Current T1 candidate model coverage is No-Go for broad candidate-subset
  teacher generation.
- Do not launch broad 1k from Top10/Top15 labels with this model.
- Next useful T1 work is a better candidate generator or a teacher/training
  design that explicitly handles incomplete candidate-subset labels.

## Latest Status: Stage9f P2 Accepted, Production Still Off

As of 2026-06-25, the T2 fixed continuation profile is accepted for
post-acceptance experiments:

- fixed T2 P2 profile:
  `stage9f_p2`
- evidence profile:
  `stage9f_cse2_csemax2_bothseat`
- runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T3 continuation:
  `stage7_m5_r10`
- model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- acceptance artifact:
  `configs/hu_turn2_stage9f_p2_acceptance_candidate.json`
- review:
  `docs/hu_turn2_stage9f_p2_acceptance_review.md`

What is accepted:

- Use `stage9f_p2` as the fixed T2 P2 continuation for new T2/T1 experiments.
- Stage9f remains selective override only.
- Full replacement remains disabled.
- Confirm-MC delta remains a gate diagnostic only; performance claims must use
  realized fired whole-game paired deltas.

What is still not accepted:

- production default is not enabled,
- `current` profile is not replaced,
- full replacement is not enabled,
- 50k teacher generation remains blocked unless a later experiment shows it is
  needed.

Important T1 update:

- The T1 leaf31 TopK line is now a decision No-Go.
- Fire-count diagnostic:
  `outputs/evals/hu_turn1_stage1j_mc32_timeout_leaf31_topk_fire500/`
- It reached `71` valid overrides but had EV/hand `-0.0101` and realized
  per-fire delta `-0.1844`.
- If T1 resumes, do not extend this exact leaf31 TopK gate. Use a revised
  objective/model/gate design with `stage9f_p2` as the fixed T2 continuation.

### HU Turn1 Stage2 Restart With Stage9f P2

T1 Stage2 was restarted with `stage9f_p2` fixed as the T2 continuation and
`stage7_m5_r10` fixed as the T3 continuation.

- Plan:
  `configs/hu_turn1_stage2_stage9f_p2_plan.json`
- Notes:
  `docs/hu_turn1_stage2_stage9f_p2_plan.md`
- Wiring smoke:
  `outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/`
- All-action speed smoke:
  `outputs/hu_turn1_stage2_stage9f_p2/speed1_mc1_allactions/`

All-action Stage9f P2 MC16 teacher is currently speed No-Go:

- GCP run:
  `regular-hu-t1-stage2-stage9f-p2-pilot100-mc16-20260625-001`
- Partial output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_pilot100_mc16_partial/`
- completed:
  `10` records from `2/20` shards
- mean seconds/sample:
  `117.22`
- residual VMs after abort:
  `0`

Candidate-subset teacher support was added to `hu_turn1_teacher_pilot`:

- CLI:
  `--candidate-model ... --candidate-topk ...`
- GCP runner:
  `scripts/Start-GcpHuTurn1PilotRun.ps1`
- Top5 GCP pilot:
  `regular-hu-t1-stage2-stage9f-p2-top5-candidate20-mc16-20260625-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_top5_mc16/`
- records:
  `20/20`
- mean seconds/sample:
  `28.16`
- missing:
  `0`
- residual VMs:
  `0`

However, Top5 undercovers the all-action teacher on the 10 completed all-action
states:

- Top5 teacher-best recall:
  `6/10`
- Top10 teacher-best recall:
  `9/10`
- Top10 max TopK regret:
  `5.3209`

Decision: candidate-subset execution path is viable, but Top5 is too narrow.
The next T1 gate should be a Top10/Top15 candidate-subset probe or a stronger
candidate-generator retrain. Do not broaden Top5 labels as if they were complete
teacher labels.

### Stage9f cheap independent confirm

Reason:

- Stage9e direct-fire removed MC/confirm completely and failed. The issue was
  not latency alone; direct-fire could not reproduce the safe subset.
- Stage9f keeps the independent confirmation step but makes it cheaper:
  Stage A `MC16`, Stage B independent `confirm32`.
- Primary metric remains realized fired whole-game paired delta. Confirm delta
  remains a gate diagnostic only.

Local smoke:

- With Round3 fire selector at threshold `0.5`, `mc16/confirm32/cse1.5`
  fired only `3` times in `300` paired seeds. It was positive but too sparse.
- Without fire selector:
  - `cse1`: `9` fires, EV/hand `-0.0017`, per-fire `-0.1111`
  - `cse1.5`: `4` fires, EV/hand `+0.0500`, per-fire `+7.5000`
- This justified a small GCP run to get enough fires.

GCP run:

- Run:
  `regular-hu-t2-stage9f-cheap-confirm-mc16c32-20260621-001`
- Output:
  `outputs/evals/hu_turn2_stage9f_cheap_confirm_mc16c32_aggregate/`
- Configs:
  - `k3/mc16/d0/se0/confirm32/cse1/pd0/seat=first/bygate_delta`
  - `k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta`
- Seeds:
  `2026065001`, `2026065002`, `2026065003`
- Completed shards:
  `6 / 6`
- Non-fired cancellation:
  clean

Aggregate:

| config | paired | fires | EV/hand | EV/hand CI | per-fire | per-fire CI | fired latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mc16_confirm32_cse1` | `8738` | `218` | `+0.0459` | `[+0.0249,+0.0668]` | `+3.6778` | `[+1.9975,+5.3581]` | `786ms` |
| `mc16_confirm32_cse1.5` | `9000` | `144` | `+0.0297` | `[+0.0131,+0.0463]` | `+3.7127` | `[+1.6403,+5.7851]` | `841ms` |

Latency:

- `mc16_confirm32_cse1`
  - all decisions mean:
    `57.7ms`
  - reranked mean:
    `562.9ms`
  - fired mean:
    `786.3ms`
- `mc16_confirm32_cse1.5`
  - all decisions mean:
    `59.1ms`
  - reranked mean:
    `570.5ms`
  - fired mean:
    `841.4ms`

Interpretation:

- Stage9f is the first T2 path in this phase that is both:
  - clearly positive on realized fired whole-game paired delta, and
  - substantially faster than Stage9d/C23 confirm128.
- Best current config:
  `k3/mc16/d0/se0/confirm32/cse1/pd0/seat=first/bygate_delta`
- It is not production-ready yet:
  only `3` seeds, first-seat only, and still an offline evaluation path.

Decision:

- `Stage9f cheap confirm`: `C4-Go`
- `Stage9e direct-fire`: still `No-Go`
- `50k teacher`: still `No-Go`
- `T1`: still `No-Go`
- `production / P2 fixed`: still `No-Go`

Larger C4 validation:

- Run:
  `regular-hu-t2-stage9f-c4-mc16c32-20260621-001`
- Output:
  `outputs/evals/hu_turn2_stage9f_c4_larger_mc16c32_aggregate/`
- T3 continuation:
  `stage7_m5_r10` explicit opt-in
- Scale:
  `5` non-overlapping seeds x `3,000` paired hands x `2` configs
- Completed shards:
  `10 / 10`
- GCP residual VMs:
  none after receive
- Non-fired cancellation:
  clean (`non_fired_nonzero_count = 0`, `non_fired_delta_sum = 0.0`)

Aggregate C4 result:

| config | paired | fires | fire rate | EV/hand | EV/hand CI | per-fire | per-fire CI | losses | p95 loss | max loss | all-decision latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mc16_confirm32_cse1.5` | `15,000` | `227` | `0.76%` | `+0.0260` | `[+0.0155,+0.0365]` | `+3.4337` | `[+2.0488,+4.8186]` | `26` | `11.2270` | `31.2270` | `59.08ms` |
| `mc16_confirm32_cse1` | `15,000` | `377` | `1.26%` | `+0.0236` | `[+0.0077,+0.0395]` | `+1.8765` | `[+0.6121,+3.1409]` | `64` | `23.2270` | `40.2270` | `59.58ms` |

Seed stability:

- `mc16_confirm32_cse1.5`: all `5 / 5` seeds were positive.
  Seed EV/hand:
  `+0.0202`, `+0.0178`, `+0.0407`, `+0.0244`, `+0.0268`.
- `mc16_confirm32_cse1`: all `5 / 5` seeds were positive.
  Seed EV/hand:
  `+0.0114`, `+0.0207`, `+0.0215`, `+0.0503`, `+0.0139`.

Interpretation after C4:

- Stage9f cheap independent confirm reproduced on a larger C4 run.
- `cse1.5` is now the preferred Stage9f T2 config because it has:
  - higher EV/hand than `cse1`,
  - clearly positive EV/hand CI,
  - stronger per-fire CI,
  - substantially lighter tail loss.
- `cse1` remains useful as a looser comparison, but it is not the conservative
  best because its p95/max losses are much larger.
- The evidence is still first-seat only. Second-seat decisions were explicitly
  blocked by `seat_not_allowed`.
- Confirm delta is still diagnostic only. The adoption metric is realized
  fired whole-game paired delta.

Updated decision:

- `Stage9f cheap confirm C4`: `Pass`
- Preferred config:
  `k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta`
- `Stage9e direct-fire`: still `No-Go`
- `50k teacher`: still `No-Go`
- `T1`: still `No-Go`
- `production / P2 fixed`: still `No-Go`

Next after larger C4:

- Run a tail-loss audit for the `cse1.5` fired losses.
- Decide whether T2 deployment is allowed as a first-seat-only selective
  override, or build a separate second-seat path.
- Package a validation-only runtime preset for `cse1.5` and verify baseline
  fallback behavior.
- Do not start T1 or 50k teacher until the first-seat-only scope and tail-loss
  audit are accepted.

C4 `cse1.5` tail-loss audit:

- Output:
  `outputs/evals/hu_turn2_stage9f_c4_cse1p5_tail_loss_audit/`
- Fired losses:
  `26 / 227`
- All fired losses are replay-ready.
- Average realized loss:
  `10.7027`
- p95 realized loss:
  `25.2270`
- max realized loss:
  `31.2270`
- Primary realized-loss labels:
  - lost FL value: `10`
  - royalty regression: `10`
  - foul regression: `5`
  - line regression: `1`
- Component counts:
  - royalty regression: `22`
  - high confirm SE: `16`
  - non-top-rank candidate: `13`
  - foul regression: `10`
  - lost FL value: `10`

Guard sweep from the same C4 fired set:

- `all`:
  `227` fires, EV/all-decision `+0.0260`, EV/first-decision `+0.0520`,
  per-fire `+3.4337`, max loss `31.2270`
- `rank<=2`:
  `185` fires, EV/all-decision `+0.0263`, EV/first-decision `+0.0526`,
  per-fire `+4.2631`, max loss `31.2270`
- `confirm_se<=3.0`:
  `205` fires, EV/all-decision `+0.0252`, EV/first-decision `+0.0505`,
  per-fire `+3.6938`, max loss `17.2270`

Interpretation:

- `confirm_se<=3.0` is the best immediate tail guard candidate. It preserves
  almost all C4 EV while cutting the worst realized tail from `31.2270` to
  `17.2270`.
- `rank<=2` improves mean EV but does not remove the worst tail loss.
- These guard sweep numbers are still derived from the same C4 fired set, so
  they are diagnostic until validated on non-overlapping seeds.

C4 `cse1.5` fired-loss MC512 replay:

- Run:
  `regular-hu-t2-stage9f-c4-cse1p5-tail-mc512-20260621-001`
- Output:
  `outputs/evals/hu_turn2_stage9f_c4_cse1p5_tail_loss_replay_mc512/`
- Input:
  `26` replay-ready fired-loss rows from the C4 `cse1.5` audit.
- Future samples:
  `512`
- T3 continuation:
  `stage7_m5_r10`
- Completed shards:
  `6 / 6`
- GCP residual VMs:
  none
- Replay status:
  `26 / 26` rows ok
- MC512 result:
  - positive delta rows: `23`
  - negative delta rows: `3`
  - safe LCB196 positive rows: `20`
  - hard-negative rows: `3`
  - mean MC512 delta: `+4.1307`
  - mean old realized delta on the same rows: `-10.7027`
  - mean old confirm delta: `+5.5555`
  - max MC512 loss: `1.6759`

Interpretation after MC512 replay:

- Most C4 realized losses were bad realized futures, not negative local EV
  decisions.
- The actual hard-negative set is small: `3 / 26` of the realized-loss rows
  remained negative under MC512.
- The next model/rule improvement should use these `3` hard negatives plus the
  `confirm_se<=3.0` guard candidate, rather than treating all `26` realized
  losses as bad actions.

Updated next after tail audit:

- Validate `cse1.5 + confirm_se<=3.0` on non-overlapping seeds.
- Runtime config syntax for this guard is:
  `k3/mc16/d0/se0/confirm32/cse1.5/csemax3/pd0/seat=first/bygate_delta`.
  The `csemax3` token blocks an override when the independent confirm-stage
  delta SE is above `3.0` and records `no_override_reason=above_confirm_se`.
- Keep `cse1.5` without this guard as the comparison baseline.
- Keep the `3` MC512 hard negatives as future hard-negative training / veto
  examples.
- Still do not start T1 or 50k teacher until the first-seat-only deployment
  scope and guarded C4 validation are accepted.

C4 `csemax3` non-overlapping validation:

- Run:
  `regular-hu-t2-stage9f-c4-csemax3-20260621-001`
- Output:
  `outputs/evals/hu_turn2_stage9f_c4_csemax3_aggregate/`
- T3 continuation:
  `stage7_m5_r10`
- Scale:
  `5` non-overlapping seeds, target `50` realized fires per seed, two configs:
  - `k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta`
  - `k3/mc16/d0/se0/confirm32/cse1.5/csemax3/pd0/seat=first/bygate_delta`
- Completed shards:
  `10 / 10`
- GCP residual VMs:
  none
- Non-fired cancellation:
  clean for both configs (`non_fired_nonzero_count = 0`)

Aggregate result:

| config | paired | fires | EV/hand | EV/hand CI | per-fire | per-fire CI | losses | p95 loss | max loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cse1.5` | `8,426` | `250` | `+0.0690` | `[+0.0458,+0.0922]` | `+4.6505` | `[+3.0846,+6.2164]` | `30` | `17.2270` | `31.2270` |
| `cse1.5+csemax3` | `9,529` | `250` | `+0.0336` | `[+0.0150,+0.0522]` | `+2.5625` | `[+1.1457,+3.9793]` | `38` | `18.2270` | `35.2270` |

Interpretation after `csemax3` validation:

- The `confirm_se<=3.0` tail guard did not reproduce the same-fireset
  diagnostic benefit on non-overlapping seeds.
- It blocked `95` candidate decisions but lowered EV and worsened the observed
  max loss.
- The current preferred Stage9f config remains unguarded `cse1.5`.
- `csemax3` should not be adopted as a runtime guard.
- This is still first-seat-only validation. Second seat remains explicitly
  blocked by `seat_not_allowed`.

Updated decision after `csemax3` validation:

- `Stage9f cse1.5`: validation `Pass`, current best first-seat T2 candidate.
- `Stage9f cse1.5+csemax3`: `No-Go` as a guard.
- `production / P2 fixed`: still `No-Go` until first-seat-only scope,
  runtime packaging, and rollback/preset behavior are reviewed.
- `50k teacher`: still `No-Go`.
- `T1`: still `No-Go`.

Stage9f runtime preflight packaging:

- Added preflight config:
  `configs/hu_turn2_stage9f_cse1p5_firstseat_preflight.json`
- Added canary/off preset matrix:
  `configs/hu_turn2_stage9f_canary_presets.json`
- Added rollout/preflight note:
  `docs/hu_turn2_stage9f_rollout_note.md`
- The validation-default preset is:
  `stage9f_cse1p5_firstseat`
- Runtime config string:
  `k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta`
- Validation AI profile:
  `stage9f_cse1p5_firstseat`
- Explicit exclusions:
  - `stage9f_cse1p5_csemax3_firstseat`
  - all second-seat presets
  - full replacement
- Production default:
  none. These configs/profile hooks are preflight/canary artifacts only; the
  `current` profile is not changed.
- Safety framing:
  baseline T2 remains the fallback/default policy; Stage9f is a selective
  first-seat override/search overlay only.
- Model-load failure fallback is explicit preflight-only behavior. The default
  CLI fails closed if the Stage8b/Stage9f candidate model is missing; the
  `--allow-missing-stage8b-model-fallback` flag is used only for the fallback
  smoke at
  `outputs/evals/hu_turn2_stage9f_preflight_missing_model_fallback_smoke/`.
- Missing-model fallback smoke result:
  `stage8b_model_loaded = False`, `stage8b_model_load_failed = True`,
  `override_fired = False` for all `6` decisions, with `no_override_reason`
  split between `model_load_failed` and `seat_not_allowed`.
- AI profile wiring smoke:
  `python -m ofc_regular.evaluate_matchups --profile-a
  stage9f_cse1p5_firstseat --profile-b stage7_m5_r10 --games 1 ...`
  completed with real model files and wrote
  `outputs/evals/hu_turn2_stage9f_profile_smoke1/`. This is wiring evidence
  only, not strength evidence.
- Added profile-evaluation TopK decision logging:
  `ofc_regular.evaluate_matchups --topk-decision-output ...` writes Stage9f
  TopK decision JSONL when the explicit `stage9f_cse1p5_firstseat` validation
  profile is used.
- AI profile log smoke:
  `outputs/evals/hu_turn2_stage9f_profile_log_smoke5/` completed `5` paired
  seeds / `10` hands and wrote `10` TopK decision rows. No overrides fired in
  this tiny smoke; reasons were `seat_not_allowed`, `topk_empty`,
  `mc_best_is_baseline`, and `below_confirm_se`.
- Added profile canary analyzer:
  `src/ofc_regular/analyze_hu_turn2_stage9f_profile_canary.py`
  summarizes TopK profile decision logs for runtime plumbing only.
- Superseded AI profile canary 100:
  `outputs/evals/hu_turn2_stage9f_profile_canary100/` completed `100` paired
  seeds / `200` hands locally in about `31s`, but it exposed a profile wiring
  bug. The Stage9f profile used Stage7 `m5_r10` for T2 confirm rollouts while
  the actual T3 runtime did not receive the same `m5/r10` thresholds. Treat this
  artifact as bug-discovery evidence only, not performance evidence.
- Stage9f profile T3 runtime wiring fix:
  `src/ofc_regular/ai_profiles.py` now passes
  `hu_turn3_min_margin = 5.0`, `hu_turn3_reference_min_margin = 10.0`, and
  `hu_turn3_stage7_enabled = True` into the actual
  `stage9f_cse1p5_firstseat` policy runtime.
- AI profile realized-delta canary 100 after the T3 `m5_r10` wiring fix:
  `outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/`
  completed `100` paired seeds / `200` hands locally in about `30s`, wrote
  `200` TopK decision rows, had `4` first-seat overrides and `0` second-seat
  overrides, replay-ready `200 / 200`, and no missing replay fields. Runtime
  audit:
  - realized per-fire delta: `+10.3068`,
  - estimated EV/decision: `+0.2061`,
  - non-fired nonzero realized deltas: `0`,
  - p95 latency: `530.59ms`,
  - reasons: `seat_not_allowed=100`, `topk_empty=63`,
    `mc_best_is_baseline=14`, `below_confirm_delta=11`,
    `below_confirm_se=8`, `override_fired=4`.
  This confirms runtime/logging/counterfactual cancellation plumbing. The fired
  sample is only `4`, so this is not a C4 performance replacement and not
  production evidence.
- AI profile realized-delta canary 1250 after the same T3 `m5_r10` wiring fix:
  `outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/`
  completed `1250` paired seeds / `2500` hands locally in about `416s`, wrote
  `2500` TopK decision rows, had `38` first-seat overrides and `0` second-seat
  overrides, replay-ready `2500 / 2500`, and no missing replay fields. Runtime
  audit:
  - overall EV/hand from paired seat-swap summary: `+0.0426`,
    95% CI `[-0.0116, +0.0967]`,
  - realized per-fire delta: `+2.8014`,
  - overall override rate: `1.52%`,
  - first-seat override rate: `3.04%`,
  - non-fired nonzero realized deltas: `0`,
  - p95 latency: `567.03ms` overall and `756.90ms` for first-seat decisions,
  - reasons: `seat_not_allowed=1250`, `topk_empty=846`,
    `mc_best_is_baseline=176`, `below_confirm_se=101`,
    `below_confirm_delta=89`, `override_fired=38`.
  This is still short of the preferred `>=50` fired-decision target, so it is
  not production evidence. It is positive runtime-canary evidence and supports
  moving the same explicit profile to a larger C4/Spot validation.
- AI profile realized-delta canary 3500:
  `outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/`
  completed `3500` paired seeds / `7000` hands locally in about `1057s`, wrote
  `7000` TopK decision rows, had `109` first-seat overrides and `0`
  second-seat overrides, replay-ready `7000 / 7000`, and no missing replay
  fields. Runtime audit:
  - whole-hand EV/hand: `+0.0230`, 95% CI `[-0.0103, +0.0562]`,
  - realized per-fire delta: `+1.4741`,
  - realized per-fire 95% CI: `[-0.6510, +3.5992]`,
  - positive / negative / zero fired deltas: `31 / 21 / 57`,
  - overall override rate: `1.56%`,
  - first-seat override rate: `3.11%`,
  - non-fired nonzero realized deltas: `0`,
  - p95 / max fired loss: `18.2270 / 36.2270`,
  - p95 latency: `533.74ms` overall and `641.80ms` for first-seat decisions,
  - reasons: `seat_not_allowed=3500`, `topk_empty=2366`,
    `mc_best_is_baseline=537`, `below_confirm_delta=254`,
    `below_confirm_se=234`, `override_fired=109`.
  Additional hard-negative input:
  `outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/audit/profile_canary_fired_losses_top30.jsonl`.
  Tail-loss guard sweep on the same fired set:
  - best simple guard: `confirm_se<=2.5`,
  - `81` fires,
  - estimated EV/decision `+0.0244`,
  - per-fire `+2.1128`, 95% CI `[-0.0141, +4.2397]`,
  - p95 / max loss `8.0 / 30.2270`.
  Added experiment-only validation profile:
  `stage9f_cse1p5_csemax2p5_firstseat`.
  Wiring smoke:
  `outputs/evals/hu_turn2_stage9f_csemax2p5_profile_smoke1/` completed `1`
  paired seed / `2` hands and confirmed `runtime_profile =
  stage9f_cse1p5_csemax2p5_firstseat`, `max_confirm_se = 2.5`, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.
  Decision: runtime plumbing and cancellation are now good, but this profile is
  still not production/P2 evidence because per-fire CI crosses zero and fired
  tail loss is heavy. Next work should independently validate
  `stage9f_cse1p5_csemax2p5_firstseat` on non-overlapping seeds and/or train a
  tail-risk guard using the fired losses instead of scaling the same unguarded
  profile.
- `stage9f_cse1p5_csemax2p5_firstseat` independent canary:
  `outputs/evals/hu_turn2_stage9f_csemax2p5_profile_canary3500_realized_m5r10/`
  completed `3500` paired seeds / `7000` hands with seed `2026065901`.
  Results:
  - whole-hand EV/hand `+0.0104`, 95% CI `[-0.0098, +0.0306]`,
  - `64` first-seat overrides,
  - realized per-fire `+1.1406`, 95% CI `[-1.0687, +3.3500]`,
  - positive / negative / zero fires `23 / 11 / 30`,
  - p95 / max fired loss `12.0 / 26.2270`,
  - non-fired nonzero realized deltas `0`,
  - p95 latency `615.77ms` overall and `951.44ms` first-seat.
  Decision: `csemax2.5` reduced tail risk but weakened EV and fire rate; still
  No-Go for production/P2.
  Retrospective sweep on this run points to `confirm_z>=2.0`, i.e.
  `stage9f_cse2_csemax2p5_firstseat`: `46` fires, estimated EV/decision
  `+0.0200`, per-fire `+3.0365`, 95% CI `[+0.7919, +5.2812]`, p95/max loss
  `5.0 / 12.0`. Added that as an experiment-only profile; next step is
  non-overlapping-seed validation.
  Wiring smoke:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_smoke1/` completed
  `1` paired seed / `2` hands and confirmed `confirm_se_multiplier = 2.0`,
  `max_confirm_se = 2.5`, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.
- `stage9f_cse2_csemax2p5_firstseat` independent canary:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_canary3500_realized_m5r10/`
  completed `3500` paired seeds / `7000` hands with seed `2026066101`.
  Results:
  - whole-hand EV/hand `+0.0134`, 95% CI `[-0.0086, +0.0353]`,
  - `46` first-seat overrides,
  - realized per-fire `+2.0316`, 95% CI `[-1.3005, +5.3637]`,
  - positive / negative / zero fires `14 / 4 / 28`,
  - p95 / max fired loss `5.0 / 36.2270`,
  - non-fired nonzero realized deltas `0`,
  - p95 latency `589.55ms` overall and `859.51ms` first-seat.
  Decision: cse2+csemax2.5 reduces negative fires, but still has a severe
  tail loss and weak EV evidence. Retrospective sweep points to
  `stage9f_cse2_csemax2_firstseat`: `33` fires, estimated EV/decision
  `+0.0180`, per-fire `+3.8085`, 95% CI `[+0.5267, +7.0903]`, p95/max loss
  `0.0 / 2.0`. Added that as an experiment-only profile; next step is
  non-overlapping-seed validation.
  Wiring smoke:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_smoke1/` completed `1`
  paired seed / `2` hands and confirmed `confirm_se_multiplier = 2.0`,
  `max_confirm_se = 2.0`, `no_override_reason = above_confirm_se` on the
  first-seat row, and `t3_continuation_policy = Stage7_candidate_A_m5_r10`.
- `stage9f_cse2_csemax2_firstseat` independent canary:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/`
  completed `3500` paired seeds / `7000` hands with seed `2026066301`.
  Results:
  - whole-hand EV/hand `+0.0174`, 95% CI `[+0.0030, +0.0319]`,
  - `39` first-seat overrides,
  - realized per-fire `+3.1258`, 95% CI `[+0.6907, +5.5610]`,
  - positive / negative / zero fires `14 / 2 / 23`,
  - p95 / max fired loss `2.0 / 19.2270`,
  - non-fired nonzero realized deltas `0`,
  - p95 latency `592.63ms` overall and `844.84ms` first-seat.
  Decision: this is the first Stage9f profile canary in this sequence with
  both whole-hand EV/hand and realized per-fire CI lower bounds above zero.
  It is still validation-only, not production/P2 fixed, because the fired
  sample is `39`, second-seat is disabled, and one `19.2270` tail loss remains.
  Retrospective sweep on this run points to `rank<=1`: `21` fires, estimated
  EV/decision `+0.0131`, per-fire `+4.3766`, 95% CI
  `[+1.1565, +7.5967]`, p95/max loss `0.0 / 2.0`. Because `rank<=1` is
  selected from this run, the next useful step is an independent non-overlap
  validation of the `rank<=1` guard, not production.
  Added experiment-only profile/preset:
  `stage9f_cse2_csemax2_rank1_firstseat` with config
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/pd0/rank1/seat=first/bygate_delta`.
  It is validation-only and must not become production default without an
  independent run.
  Wiring smoke:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_smoke1/`
  completed `1` paired seed / `2` hands and confirmed
  `candidate_ev_rank_max = 1`, `confirm_se_multiplier = 2.0`,
  `max_confirm_se = 2.0`, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.
  Partial independent validation:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_canary10000_realized_m5r10/`
  was started for `10000` paired seeds with seed `2026066701` but stopped after
  `4047` paired seeds / `8094` decisions because the independent sample
  contradicted the retrospective result. Results at stop:
  - `28` first-seat overrides,
  - estimated EV/decision `-0.0014`,
  - realized per-fire `-0.4010`, 95% CI `[-2.9542, +2.1523]`,
  - positive / negative / zero fires `8 / 5 / 15`,
  - p95 / max fired loss `22.0270 / 24.2270`,
  - non-fired nonzero realized deltas `0`.
  Decision: `rank<=1` is rejected and should not be promoted. The broader
  `stage9f_cse2_csemax2_firstseat` profile remains the better validation
  candidate; future work should either run a larger non-overlapping validation
  of that profile or train a fresh tail-risk guard from independent fired-loss
  rows.
- `stage9f_cse2_csemax2_firstseat` larger independent canary:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/`
  completed `10000` paired seeds / `20000` hands with seed `2026066901`.
  Results:
  - whole-hand EV/hand `+0.0129`, 95% CI `[+0.0036, +0.0222]`,
  - `90` first-seat overrides,
  - realized per-fire `+2.8571`, 95% CI `[+0.8641, +4.8500]`,
  - positive / negative / zero fires `31 / 7 / 52`,
  - p95 / max fired loss `8.0 / 31.2270`,
  - non-fired nonzero realized deltas `0`,
  - p95 latency `596.27ms`.
  Artifacts:
  - `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/audit/profile_canary_summary.md`
  - `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/tail_loss_audit/tail_loss_guard_sweep.csv`
  - `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/tail_loss_audit/tail_loss_top.jsonl`
  Decision: the positive T2 Stage9f first-seat signal reproduced at larger
  scale, and non-fired cancellation remained exact. This is still
  validation-only, not production/P2 fixed, because the fired tail still has
  `7` losses, p95 loss `8.0`, and max loss `31.2270`. Same-run guard sweeps
  remain diagnostic only: `rank<=1` is rejected by independent partial
  validation, and `confirm_z>=2.5` still leaves max loss `26.2270`. Next useful
  work is an independently trained/validated tail-risk guard or selected
  high-MC replay of the worst losses, not T1, 50k teacher, or production.
- Canonical tail-guard/replay target set for the next step:
  `outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/`.
  It is generated from the `10000` canary only, not from the earlier canary, to
  avoid mixing potentially overlapping seed ranges. Counts:
  - `114` targets,
  - `114 / 114` replay-ready,
  - source fired realized rows `90`,
  - tail losses `7`,
  - severe tail losses at threshold `8.0`: `5`,
  - positive controls `28`,
  - zero controls `52`,
  - confirm-z boundary targets `27`.
  This is a guard-training / selected high-MC input artifact, not performance
  evidence. Production, P2 fixed, 50k teacher, and T1 remain No-Go.
- Stage9f tail-guard replay runner:
  `ofc_regular.replay_hu_turn2_stage9f_tail_guard_targets` now consumes those
  runtime-log targets directly, verifies baseline/candidate legal action
  mapping, and writes `replay_results.partial.jsonl` plus `progress.json`
  incrementally so Spot VM/interrupted runs can resume.
  Readiness:
  `outputs/evals/hu_turn2_stage9f_tail_guard_replay_readiness_10000/`
  recovered `114 / 114` targets with exact action mapping.
  Tail-loss selected replay:
  `outputs/evals/hu_turn2_stage9f_tail_guard_replay_tail_loss_mc512/`
  replayed the `7` input tail-loss rows at MC512:
  - successes `7 / 7`,
  - high-MC gain mean `+1.3560`,
  - min / max gain `-1.9255 / +3.9536`,
  - high-MC losses `1 / 7`,
  - LCB95-positive rows `5 / 7`,
  - sign flips vs input realized deltas `6 / 7`.
  Interpretation: most large seat-swap realized losses do not remain negative
  under independent MC512; they are useful tail-risk examples but should not all
  be hard negatives. The one MC512-negative row is the current first true
  Stage9f hard-negative seed. Full `114` target replay should use the
  partial/resume path or Spot VM. Production/P2 fixed/50k/T1 remain No-Go.
- Stage9f tail-guard replay labels:
  `outputs/training/hu_turn2_stage9f_tail_guard_labels_tail_loss_mc512/`
  labels the `7` MC512 tail-loss replays for future tail-risk guard training:
  `1` hard negative, `5` safe positives, and `1` gray row. These are local
  high-MC training labels, not runtime thresholds and not production evidence.
  Production/P2 fixed/50k/T1 remain No-Go.
- Stage9f full tail-guard replay on GCP Spot:
  run `regular-hu-t2-stage9f-tail-guard-replay-20260622-221333` completed
  `15 / 15` shards and replayed `114 / 114` targets at MC512. Output:
  `outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/`.
  Raw labels are `8` hard negatives, `72` safe positives, and `34` gray rows,
  but `25` rows are duplicate source events across target buckets. The deduped
  training labels are therefore the safer next input:
  `outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/labels/stage9f_tail_guard_labels_dedup.csv`
  with `89` unique events = `5` hard negatives, `57` safe positives, and `27`
  gray rows. Use raw `114` only for audit/source accounting, not direct
  training. Production/P2 fixed/50k/T1 remain No-Go.

Stage9g tail-risk guard smoke:

- Config:
  `configs/hu_turn2_stage9g_tail_guard_training_plan.json`
- Training rows:
  `outputs/training/hu_turn2_stage9g_tail_guard_training/stage9g_tail_guard_training_rows.jsonl`
- Rows excluding gray:
  `62`
- Hard negatives / safe controls:
  `5 / 57`
- Smoke models:
  - `preconfirm_meta_only`: all AP/AUC `0.4137 / 0.6982`
  - `opportunity_proxy_plus_preconfirm_meta`: all AP/AUC `0.3884 / 0.6737`
  - `hu_delta_plus_preconfirm_meta`: all AP/AUC `0.5967 / 0.9404`
- Current best offline ranking candidate:
  `hu_delta_plus_preconfirm_meta`
- Runtime guard:
  `No-Go`. Probabilities are not calibrated for thresholds yet; thresholds
  `>= 0.7` select zero rows in the smoke.
- Large Stage9g training:
  `No-Go` until more labels. Minimum target is `50` deduped hard negatives;
  `100+` is preferred.

Next Stage9g action:

- Collect more selected high-MC labels from Stage9f `cse2/csemax2` fired
  losses, near-confirm-boundary rows, rank/confirm disagreements, large-gain
  positive controls, and zero controls. Use GCP Spot VM if the selected replay
  is too slow locally.

Stage9g expanded labels and smoke:

- Mixed Stage9f target extraction:
  `outputs/training/hu_turn2_stage9g_more_tail_guard_targets_stage9f_mixed/`
- Event-deduped replay targets:
  `317`
- Replay-ready:
  `317 / 317`
- GCP run:
  `regular-hu-t2-stage9g-more-tailguard-replay-20260623-001`
- GCP result:
  `40 / 40` shards, `317 / 317` replay rows, `0` failures
- Output:
  `outputs/evals/hu_turn2_stage9g_more_tail_guard_replay_stage9f_mixed_gcp/`
- Labels:
  `21` hard negatives, `195` safe positives, `101` gray
- Training rows:
  `outputs/training/hu_turn2_stage9g_tail_guard_training_stage9f_mixed_mc512/`
- Rows excluding gray:
  `216`
- New smoke:
  - `hu_delta_plus_preconfirm_meta`: all AP/AUC `0.8389 / 0.9516`,
    test AP/AUC `0.5111 / 0.8161`, threshold `0.6` precision `0.90`
    on `10` selected rows, threshold `0.7` selects `0`.
  - `preconfirm_meta_only`: all AP/AUC `0.2508 / 0.7282`, weaker.

Updated Stage9g decision:

- Candidate for offline tail-risk ranking:
  `hu_delta_plus_preconfirm_meta`
- Runtime guard:
  still `No-Go`
- Large training:
  still `No-Go`; hard negatives increased from `5` to `21`, but minimum gate is
  `50`, preferred `100+`.
- Production/P2 fixed/50k/T1:
  still `No-Go`.

### Stage9e direct-fire hard-negative loop

Purpose:

- The first Stage9e direct-fire smoke showed that the existing Stage9d fire
  selector was fast enough but not safe enough without MC64/confirm128.
- Next experiment: mine direct-fire false positives and retrain the lightweight
  selector with those fired losses upweighted.

Round 1 data:

- Direct-fire smoke distillation:
  `outputs/training/hu_turn2_stage9e_direct_fire_smoke_distillation/`
- Source logs:
  - `outputs/evals/hu_turn2_stage9e_direct_fire_t050_smoke300/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage9e_direct_fire_t080_smoke300/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage9e_direct_fire_t090_smoke300/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage9e_direct_fire_t095_smoke200/runtime_decisions.jsonl`
- Rows:
  `70`
- Positive / negative:
  `7 / 63`
- Fired direct-fire rows:
  `20`, with realized mean `-2.4227`

Round 1 model:

- Model:
  `models/hu_turn2_stage9e_direct_fire_hardneg_h64.pt`
- Output:
  `outputs/training/hu_turn2_stage9e_direct_fire_hardneg_h64/`
- Training inputs:
  Stage9d C23 distillation + Stage9d fsreject distillation + Round 1
  direct-fire smoke distillation.
- Loss weighting:
  `topk_confirm_realized_loss` weight `6.0`
- Rows:
  `8230`
- Positive / negative:
  `337 / 7893`
- Metrics:
  - test AP / ROC AUC:
    `0.1869 / 0.7904`
  - all AP / ROC AUC:
    `0.4986 / 0.9142`

Round 1 runtime smokes:

| threshold | paired | fires | EV/hand | realized per-fire | all mean latency | fired mean latency |
|---:|---:|---:|---:|---:|---:|---:|
| `0.90` | `300` | `0` | `0.0000` | `0.0000` | `12.45ms` | not material |
| `0.80` | `500` | `1` | `0.0000` | `0.0000` | `16.76ms` | `22.53ms` |
| `0.70` | `500` | `6` | `-0.0040` | `-0.6667` | `13.85ms` | `16.67ms` |
| `0.50` | `500` | `28` | `-0.0040` | `-0.1429` | `12.55ms` | `16.15ms` |

Interpretation:

- Latency remains good; all tested direct-fire paths have `0ms` MC/confirm
  latency.
- Adding the first hard negatives removed the severe t0.50 collapse, but did
  not produce a reliably positive direct-fire policy.
- The model still under-fires at safe thresholds and is approximately break-even
  at permissive thresholds on these small local smokes.

Round 2 data:

- Distillation:
  `outputs/training/hu_turn2_stage9e_direct_fire_round2_mine_distillation/`
- Source logs:
  - `outputs/evals/hu_turn2_stage9e_hardneg_direct_fire_t050_mine500/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage9e_hardneg_direct_fire_t070_smoke500/runtime_decisions.jsonl`
  - `outputs/evals/hu_turn2_stage9e_hardneg_direct_fire_t080_smoke500/runtime_decisions.jsonl`
- Rows:
  `179`
- Positive / negative:
  `8 / 171`

Round 2 model:

- Model:
  `models/hu_turn2_stage9e_direct_fire_round2_h64.pt`
- Output:
  `outputs/training/hu_turn2_stage9e_direct_fire_round2_h64/`
- Loss weighting:
  `topk_confirm_realized_loss` weight `8.0`
- Rows:
  `8409`
- Positive / negative:
  `345 / 8064`
- Metrics:
  - test AP / ROC AUC:
    `0.1844 / 0.7513`
  - all AP / ROC AUC:
    `0.5103 / 0.9114`

Round 2 runtime smoke:

| threshold | paired | fires | EV/hand | realized per-fire | all mean latency | fired mean latency |
|---:|---:|---:|---:|---:|---:|---:|
| `0.60` | `500` | `4` | `-0.0242` | not enough | not final | not final |

Decision:

- `Stage9e direct-fire`: still `No-Go`.
- It is not a latency problem anymore. It is a data/target problem: there are
  not enough high-quality direct-fire positives and hard negatives to replace
  MC64/confirm128.

GCP mining note:

- Initial GCP mining run
  `regular-hu-t2-stage9e-directfire-r2-t05-mine80-20260621-001`
  was stopped because the VM startup script did not read the
  `STAGE8C_FIRE_SELECTOR_DIRECT_FIRE` metadata key; the Python process was
  therefore missing `--stage8c-fire-selector-direct-fire` and was running the
  slower MC/confirm path.
- Fixed script:
  `scripts/Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1`
  now reads `STAGE8C_FIRE_SELECTOR_DIRECT_FIRE` in startup bash and writes the
  flag to running/complete/failed status JSON.
- Corrected GCP run:
  `regular-hu-t2-stage9e-directfire-r2-t05-mine80-fix-20260621-001`
- VM command check:
  all shards included `--stage8c-fire-selector-direct-fire`.
- Completed shards:
  `3 / 3`
- Aggregate output:
  `outputs/evals/hu_turn2_stage9e_directfire_r2_t05_mine80_fix_aggregate/`
- Aggregate realized fires:
  `240`
- Aggregate realized per-fire:
  `-1.2443`
- Aggregate EV/hand:
  `-0.0253`
- Non-fired cancellation:
  clean

Round 3 data:

- Distillation:
  `outputs/training/hu_turn2_stage9e_directfire_r2_t05_mine80_fix_distillation/`
- Source logs:
  corrected GCP run shard `runtime_decisions.jsonl` files.
- Rows:
  `944`
- Positive / negative:
  `57 / 887`

Round 3 model:

- Model:
  `models/hu_turn2_stage9e_direct_fire_round3_h64.pt`
- Output:
  `outputs/training/hu_turn2_stage9e_direct_fire_round3_h64/`
- Training rows:
  `9353`
- Positive / negative:
  `402 / 8951`
- Realized-loss hard negatives:
  `574`
- Loss weighting:
  `topk_confirm_realized_loss` weight `8.0`

Round 3 runtime smokes:

| threshold | paired | fires | EV/hand | realized per-fire | p95 loss | max loss |
|---:|---:|---:|---:|---:|---:|---:|
| `0.50` | `500` | `19` | `-0.0705` | `-3.7081` | `31.2270` | `40.2270` |
| `0.60` | `500` | `7` | `-0.0182` | `-2.6039` | `16.9589` | `24.2270` |

Updated decision:

- `Stage9e direct-fire`: still `No-Go`.
- The hard-negative loop improved the training set, but the runtime result is
  still negative. Simple direct-fire replacement of MC64/confirm128 is not
  currently viable.
- Keep Stage9d/C23-style TopK+MC+confirm as the strength reference and treat
  direct-fire as a research path only.
- Next useful action is not more threshold tweaking. Either:
  - improve the selector target/features using third-sample or high-MC labels,
    or
  - design a cheaper confirm approximation that keeps an independent
    verification step.
- Production / P2 fixed / 50k teacher / T1:
  still `No-Go`.

### T1 Stage2 candidate-subset update

Context:

- Fixed continuation remains `stage9f_p2` for T2 and `stage7_m5_r10` for T3.
- All-action T1 MC16 teacher generation is too slow to broaden directly.
- Current full2k T1 candidate model coverage on all-action audit40 is No-Go.

Implementation added:

- `hu_turn1_teacher_pilot` now supports multi-model candidate union:
  `--candidate-models ... --candidate-topk K --candidate-union-cap N`.
- `analyze_hu_turn1_candidate_coverage` now supports the same union coverage
  audit.
- `scripts/Start-GcpHuTurn1PilotRun.ps1` now packages and passes
  `-CandidateModels` / `-CandidateUnionCap` to Spot VM workers.

All-action audit40 model comparison:

- Current full2k model Top15:
  recall `33/40`, avg regret `0.3605`, max regret `7.0568`.
- `stage1b accept2 gray5` Top15:
  recall `35/40`, avg regret `0.0938`, max regret `1.4432`.
- `stage1b accept2 gray5` Top18:
  recall `36/40`, avg regret `0.0375`, max regret `1.25`.
- `stage1b accept2 gray5` Top25:
  recall `40/40`, avg regret `0.0`, max regret `0.0`.
- Best simple union checked, current + stage1j + stage1b Top15 with cap20:
  recall `38/40`, avg regret `0.0313`, max regret `1.25`.

Interpretation:

- Union support is useful infrastructure, but the simple union configs did not
  clearly beat the simpler `stage1b accept2 gray5` path for the next probe.
- `stage1b Top15` is currently the best speed/coverage candidate from audit40.
- Because it was selected after seeing audit40, it must pass a fresh
  non-overlapping all-action coverage audit before broad generation.

Stage1b Top15 MC16 candidate smoke:

- Balanced GCP run:
  `regular-hu-t1-stage2-stage9f-p2-stage1b-top15-mc16-balanced20-20260625-001`
- Output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_stage1b_top15_mc16_balanced/`
- Records:
  `20 / 20`
- Candidate actions per state:
  `15`
- Mean / max seconds per sample:
  `75.71 / 140.11`
- Invalid action/state rows:
  `0 / 0`
- Seat split:
  `first=10`, `second=10`
- Residual GCP VMs:
  `0`

Decision:

- `stage1b Top15` candidate-subset teacher execution is clean and balanced.
- Broad T1 teacher generation:
  still `No-Go` until fresh heldout all-action coverage confirms the candidate
  model.
- Next gate:
  run a fresh all-action audit40 MC4 with non-overlapping seeds and compare
  `stage1b Top15/18/20` coverage. If it holds, run a larger candidate-subset
  teacher pilot with `stage1b Top15` or `Top18`.

### T1 Stage2 fresh audit and union Top18 smoke

Fresh heldout all-action audit:

- Run:
  `regular-hu-t1-stage2-stage9f-p2-allaction-fresh-audit40-mc4-20260625-001`
- Output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction_fresh_audit40_mc4/`
- Records:
  `40 / 40`
- Seat split:
  `first=20`, `second=20`
- Mean / max seconds per sample:
  `30.62 / 55.67`
- Mean legal actions:
  `26.63`
- Invalid action/state rows:
  `0 / 0`

Fresh coverage highlights:

- `stage1b Top15`:
  recall `36/40`, avg regret `0.2188`, max regret `5.1932`.
- `stage1b Top18`:
  recall `37/40`, avg regret `0.1423`, max regret `5.1932`.
- `stage1b Top25`:
  recall `40/40`, avg regret `0.0`, max regret `0.0`.
- `stage1e Top20`:
  recall `39/40`, avg regret `0.0`, max regret `0.0`.
- `stage1b + stage1e union Top15 cap18`:
  recall `40/40`, avg regret `0.0`, max regret `0.0`.

Union Top18 execution smoke:

- Run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-top18-mc16-20260625-001`
- Output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_union_stage1b_stage1e_top18_mc16/`
- Records:
  `20 / 20`
- Candidate models:
  `stage1b accept2 gray5`, `stage1e accept2 runtime313`
- Candidate selection:
  Top15 per model, union cap `18`
- Mean / max seconds per sample:
  `99.98 / 168.51`
- Mean action count:
  `17.2`
- Seat split:
  `first=10`, `second=10`
- Invalid action/state rows:
  `0 / 0`

Decision:

- `stage1b Top15/18` is not enough for broad T1 teacher generation; the fresh
  audit still has a large miss.
- `stage1e Top20` and `stage1b+stage1e union Top18` are the best observed
  candidates on the fresh audit.
- `stage1b+stage1e union Top18` is execution-clean but slower than
  `stage1b Top15`, and earlier audit evidence was not as clean.
- Broad T1 teacher generation remains `No-Go` unless another heldout confirms
  robust coverage, or the training objective explicitly treats candidate-subset
  rows as incomplete labels rather than all-action teacher labels.

Stage1e Top20 execution smoke:

- Run:
  `regular-hu-t1-stage2-stage9f-p2-stage1e-top20-mc16-20260625-001`
- Output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_stage1e_top20_mc16/`
- Records:
  `20 / 20`
- Candidate model:
  `models/hu_turn1_stage1e_candidate_hgb_accept2_gray5_runtime313_weighted.pkl`
- Candidate TopK:
  `20`
- Mean / max seconds per sample:
  `108.98 / 257.36`
- Mean action count:
  `20.0`
- Seat split:
  `first=10`, `second=10`
- Invalid action/state rows:
  `0 / 0`

Updated T1 Stage2 interpretation:

- `stage1e Top20` is execution-clean and has the best single-model fresh-audit
  coverage, but it is slower than union Top18 and much slower than
  `stage1b Top15`.
- `stage1b+stage1e union Top18` is a good diagnostic candidate set, but not yet
  robust enough to treat as a complete all-action teacher.
- Do not launch broad T1 teacher generation as if these candidate-subset labels
  are complete. Either run another non-overlapping all-action audit for
  `stage1e Top20` / union Top18, or explicitly design the next training pass
  around incomplete candidate-subset labels.

### T1 Stage2 union Top20 cap22 pilot100

Latest candidate-subset pilot:

- Run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-top20-cap22-pilot100-mc16-20260625-001`
- Final output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate100_union_stage1b_stage1e_top20_cap22_mc16/hu_turn1_stage1_pilot.jsonl`
- Analysis:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate100_union_stage1b_stage1e_top20_cap22_mc16/analysis_summary.json`
- Records:
  `100 / 100`
- Future samples:
  `16`
- Candidate models:
  `stage1b accept2 gray5` + `stage1e accept2 runtime313`
- Candidate selection:
  Top20 per model, union cap `22`
- Mean action count:
  `21.63` with min/max `20 / 22`
- Actions truncated:
  `100 / 100`
- Invalid action/state rows:
  `0 / 0`
- Seat split:
  `first=60`, `second=40`; this is expected for 20 shards of 5 T1 records,
  because each shard emits `first, second, first, second, first`.
- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`

Rescue details:

- Original Spot run completed `13 / 20` shards.
- `rescue2` supplied `27` exact-skip partial rows.
- `rescue4` supplied the final `8` missing rows using `--fast-skip-records`
  and on-demand VMs.
- The old `rescue_sXXX` outputs are intentionally excluded because they
  generated the first row for each seed and skewed the seat distribution.
- `--fast-skip-records` preserves the target T1 state by replaying actual
  prior T0/T1 actions but does not reproduce the skipped rows' MC future RNG.
  This is acceptable for rescue teacher rows because each evaluated row still
  has common futures internally; do not use it when exact MC-future parity with
  a full shard is required.

Decision:

- `union Top20 cap22` is the best current candidate-subset teacher line.
- The pilot100 dataset is schema-clean and usable for a T1 candidate-subset
  training smoke.
- It is not complete all-action teacher data. Any training must treat the rows
  as incomplete/candidate-set labels, not as proof that the true all-action best
  was included.
- T1 production/runtime adoption remains `No-Go`.
- Next useful step: train a T1 candidate/listwise model on this pilot100
  candidate-set data, then evaluate candidate-regret and run another heldout
  all-action coverage audit before scaling beyond pilot size.

### T1 Stage2 candidate100 training smoke

The candidate-set training smoke completed on the union Top20 cap22 pilot100
dataset.  All rows remain incomplete candidate labels, not complete all-action
teacher labels.

Artifacts:

- comparison:
  `outputs/training/hu_turn1_stage2_candidate100_commonseed_model_comparison.json`
- HGB score-regressor:
  `models/hu_turn1_stage2_candidate100_commonseed_hgb_regressor.pkl`
- pairwise logistic:
  `models/hu_turn1_stage2_candidate100_commonseed_pairwise_logistic.pkl`
- Torch listwise:
  `models/hu_turn1_stage2_candidate100_commonseed_listwise_torch.pt`

Common split `2026062699`, holdout `20`:

- `stage2 HGB regressor`:
  top1 avg regret `1.7541`, top3 best avg regret `0.8175`,
  top5 best avg regret `0.6535`, top10 best avg regret `0.0716`,
  top3 accepted recall `0.50`.
- `stage1e` baseline on the same holdout:
  top1 avg regret `2.9335`, top3 best avg regret `1.3794`,
  top5 best avg regret `0.7023`, top10 best avg regret `0.0851`,
  top3 accepted recall `0.35`.
- `pairwise_logistic` and `listwise_torch` did not beat the HGB regressor on
  this common holdout.  Listwise overfit the small 80-row train split.

Implementation note:

- Pairwise model saving was fixed by moving `DecisionFunctionAsScoreEstimator`
  into importable `hu_turn3_model`; a fresh-process load regression test was
  added.

Decision:

- Carry forward only `stage2 HGB regressor` as a candidate-set generator.
- Do not use the pilot100 HGB regressor directly as the broad teacher candidate
  selector; as a single model it needs Top25 to become clean on the checked
  all-action audits.
- The original `stage1b + stage1e` union Top20 cap22 label source remains the
  correct broad-data generator.  Cap22 has zero EV regret on the checked
  original/fresh/heldout2 all-action audit40 sets.
- Broad 1k MC16 candidate-set teacher generation is `Go` for training data
  only, using `stage1b + stage1e` Top20 per model with union cap `22`.
- T1 runtime / production remains `No-Go`.

### T1 Stage2 broad1k candidate-set teacher and model

Broad `1k` MC16 candidate-set teacher generation completed.

- run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-top20-cap22-broad1000-mc16-20260625-001`
- merged output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_top20_cap22_mc16/hu_turn1_stage1_pilot.jsonl`
- aggregate summary:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_top20_cap22_mc16/summary.json`
- analysis summary:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_top20_cap22_mc16/analysis_summary.json`
- records/shards:
  `1000 / 1000`, `200 / 200`
- candidate source:
  `stage1b + stage1e`, Top20 per model, union cap `22`
- mean action count:
  `21.559`
- invalid action/state rows:
  `0 / 0`
- seat split:
  `first=600`, `second=400`
- continuation:
  T2 `stage9f_p2`, T3 `stage7_m5_r10`

Generation note:

- The initial `e2-highcpu-4` Spot run stalled at `59 / 200` shards with
  memory-pressure symptoms.  Completed shards were retained, the stalled VMs
  were deleted, and the missing shards were rescued with `e2-standard-4` Spot
  VMs plus one final single-shard rescue for shard `196`.

Training completed:

- model:
  `models/hu_turn1_stage2_candidate1000_union_top20_cap22_hgb_regressor.pkl`
- metrics:
  `outputs/training/hu_turn1_stage2_candidate1000_union_top20_cap22_hgb_regressor/metrics.json`
- split:
  seed `2026062701`, train/holdout `800 / 200`

Candidate-set holdout:

- Stage2 broad1k HGB:
  top1 avg regret `1.9634`, top3 best avg regret `0.7852`,
  top5 best avg regret `0.4409`, top3 accepted recall `0.635`,
  top5 accepted recall `0.775`.
- Stage1e baseline on same holdout:
  top1 avg regret `3.8236`, top3 best avg regret `2.1083`,
  top5 best avg regret `1.3641`, top3 accepted recall `0.415`,
  top5 accepted recall `0.545`.

All-action coverage:

- broad1k HGB single model:
  original audit40 Top25 clean, fresh audit40 Top22 clean, heldout2 audit40
  Top27 clean.
- `stage1b + stage1e + broad1k HGB` union, Top20/model:
  original audit40 Top22 EV-regret clean with recall `0.975`, fresh audit40
  Top20 clean, heldout2 audit40 Top25 clean.

Decision:

- The broad1k HGB model is a genuine candidate-set improvement and should be
  carried forward as a candidate-set model.
- It is not yet a T1 runtime policy.  T1 runtime, production, and P2-style
  fixing remain `No-Go`.
- Next gate: larger all-action audit and/or a T1 candidate-set reranking design
  before any runtime adoption.
