# HU joint-policy M1 completion audit

Status date: 2026-07-12

Decision: **M1 foundation Go; policy promotion, safe-artifact claims, and large
compute No-Go.**

M1 establishes the information-set, action identity, RNG, replay, and parity
boundaries needed by later milestones. It does not certify the strength of the
fixed models and it does not complete the M2 late-street teacher.

## Requirement-by-requirement evidence

| Requirement | Result | Authoritative evidence |
|---|---|---|
| Simulator truth is separate from policy input | Pass | `WorldState`, immutable `ActorObservation`, and geometry validation in `src/ofc_regular/hu_infoset.py`; invariance tests in `tests/test_hu_infoset.py` |
| Opponent private discards cannot enter normal runtime observation | Pass | `play_ai.py` and `evaluate_matchups.py` call `choose_action_observation`; `tests/test_policy_play.py` checks both paths |
| Replay truth is attached only after a policy returns | Pass | `src/ofc_regular/decision_trace.py`; attachment-order tests in `tests/test_hu_infoset.py` and `tests/test_policy_play.py` |
| T1/T2 labels do not condition on realized hidden cards | Pass | mandatory `ActorObservation` plus `HiddenCardParticleBatch`; opponent-truth invariance tests in `tests/test_hu_turn1_teacher_pilot.py` and `tests/test_hu_turn2_teacher_data.py` |
| Feature caches fail closed on ambiguous legacy rows | Pass | `policy_feature_sample_from_record`, T2 cache v2, T3 cache v2, and rejection tests in `tests/test_hu_infoset.py`, `tests/test_hu_turn2_pilot_training.py`, and `tests/test_cache_hu_turn3_features.py` |
| Legal actions have order-independent identity | Pass | fixed-width `ActionKey`, set/order digests, checked resolution, all T0-T4 dealt permutations in `tests/test_action_key.py` |
| Runtime and training mapping cannot silently drift | Pass | semantic key-first runtime choice, cache remapping, and fail-closed T1 selector tests in `tests/test_train_hu_turn1_safe_override_selector.py` |
| Counter RNG is candidate-order independent | Pass | `src/ofc_regular/counter_rng.py`; candidate-order and domain-separation tests in `tests/test_counter_rng.py`, T0/T1/T2 teacher tests |
| Scalar/batch/exact behavior agrees | Pass | T3 and full T2 rollout scalar/batch tests for first and second seats; independent T4 exhaustive terminal-score checks |
| Non-fire counterfactuals cancel exactly | Pass | same-snapshot full-trajectory digest plus semantic action equality in `evaluate_matchups.py`; negative tests in `tests/test_evaluate_matchups.py` |
| Baselines remain available and `current` is unchanged | Pass | 16/16 fixed config/model SHA-256 matches against M0; explicit profile chain in both manifests; current registry M0-hash reconstruction in `validation_summary.json` |
| Full regression remains green | Pass | `python -B -m pytest -q`: 876 passed in 26.28 seconds |

`CardFreeMetadata` rejects known card/world fields and card-bearing objects on
every mutation path. It is defense in depth, not a proof against an arbitrary
new alias containing card-shaped strings. `ActorObservation` is the enforced
card-bearing policy boundary.

## Legacy artifact audit

The fixed artifacts were inspected rather than assumed safe.

T2 source audited:

`outputs/hu_turn2_stage8_20k_mc512/natural.jsonl`

- 100 rows: 50 first seat, 50 second seat
- legacy private `dead_cards` lengths: 2 for 50 rows, 3 for 50 rows
- missing `hero_private_discards`: 100/100
- missing `policy_observation`: 100/100
- missing `replay_truth`: 100/100
- the v1 cache builder passed the raw rows into the action encoder

T3 Stage7 source audited:

`outputs/gcp_runs/regular-hu-t3-stage7-stage3-override-mc2048-m4-25-50k-2h-20260608-001/hu_turn3_selfplay_merged.jsonl`

- 100 rows: 71 first seat, 29 second seat
- private discard identities beyond opponent public cards: 4 for 71 rows, 5
  for 29 rows
- missing visible/private split and `policy_observation`: 100/100
- the legacy T3 cache builder encoded raw rows

Consequences:

- `stage9f_p2` and `stage7_m5_r10` stay available as legacy baselines, not as
  hidden-discard-safe models.
- P0/P1 teacher and promotion evaluations used those continuations, so their
  old acceptance evidence is also diagnostic only.
- The reported T0 100,000-paired gain of `+0.0173028 EV/hand` and its confidence
  interval must be rerun after safe downstream artifacts exist.
- No file was deleted and no profile was promoted or made current.

The machine-readable quarantine is
`configs/hu_joint_policy_m1_quarantine.json`. In particular, the old
`hu_turn3_joint_exact_teacher.py`, raw-dead-card T2 risk trainer, and implicit
`current` self-play/mining entrypoints cannot be used as new-policy evidence.

## M1 versus M2 boundary

M1 is complete because every M1 foundation gate has direct test evidence. The
following remain deliberately No-Go and are M2 or later work:

- correct first-seat T4 sequential opponent response under hidden-card belief;
- repaired T3 exact/MC teacher with disjoint candidate/evaluation samples;
- fresh ActorObservation-only T2/T3 cache generation and retraining;
- 100-1,000 state correctness/speed pilot;
- any Spot-scale generation, policy promotion, or old-EV reuse.

The immediate next milestone is M2, beginning with the first-seat T4 response
model and a scalar reference teacher before any Rust or cloud scaling.
