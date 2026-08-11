# M3.1 T3 fresh-quality v1 contract

## Purpose

This stage answers one question: after Candidate02 passes the one-shot
performance-lock v4, is the accepted `8/32/4/exact-T4` search sufficiently
accurate on a fresh, pre-registered T3 population?

It is not match EV, training data authorization, model promotion, or permission
to change `current`.

## Frozen experiment

| Population | Paired hands | Roots | Budget |
|---|---:|---:|---|
| Primary quality | 50 | 100 | `8/32/4/0`, exact T4 |
| Independent confirmation | 5 | 10 | primary `8/32/4/0` plus locked `8/128/4/0` |

The confirmation hands are not a subset of the primary 50. Each population has
its own hand, behavior, candidate, evaluation, child, and confirmation seed
base. All 330 reserved quality seed values are unique, the two populations are
disjoint, and they are disjoint from performance-lock v4.

Both populations are balanced over the five frozen behavior profiles:

- `stage19_p0`
- `stage9f_p2`
- `stage7_m5_r10`
- `stage3_baseline`
- `random_exact_final`

Primary contains 10 paired hands per profile. Confirmation contains one paired
hand per profile. Every paired hand contributes one first-seat and one
second-seat T3 information set.

## Information and action safety

Roots contain only `ActorObservation`:

- hero board
- opponent public board
- hero private discards
- dealt cards
- seat/action order
- street and scoring context

Opponent-private discards, world state, future cards, remaining deck, replay
truth, and realized deck tail are forbidden recursively.

The result merger reuses the accepted Step 6c decision-certificate validator.
For every root it recomputes:

- the complete legal action set
- canonical `ActionKey` order and original-index mapping
- selection and locked-evaluation Q values and regret arithmetic
- selected action mapping
- search/result certificates
- belief digests and individual particle RNG keys
- exact-T4 runtime identity

Unknown fields, missing/extra results, duplicated fingerprints, ActionKey
drift, hidden fields, or any RNG overlap fail before a scientific merge can be
created.

## Immutable lifecycle

The implementation is split into:

- `src/ofc_regular/hu_m31_t3_step6d_fresh_quality_v1.py`
  - performance-lock authorization
  - deterministic plan and seed schedule
  - 55 paired-root files
  - source-replayed materialization receipt and root seal
  - 10 primary jobs plus 5 confirmation jobs
  - deterministic package ZIP and independent package validator
- `src/ofc_regular/hu_m31_t3_step6d_fresh_quality_gate_v1.py`
  - strict worker-result contract
  - source-replayed 15-job merge
  - frozen quality gate

All scientific JSON files are canonical and create-only. A partial
materialization or package is retained for inspection and cannot be silently
resumed under the same output path.

No quality plan can be created until
`hu_m31_t3_step6d_performance_lock_v4_production_final_receipt_v1` is fully
source-replayed and states:

- performance lock finalized and qualified
- one-shot lock consumed
- quality pilot authorized
- `current_profile_changed = false`

## Gate

The 10 independent confirmation roots use nearest-rank percentiles. All four
conditions must pass:

| Metric | Maximum |
|---|---:|
| mean regret | 0.75 |
| p95 regret | 3.0 |
| p99 regret | 6.0 |
| max regret | 15.0 |

In addition, hidden/unknown/ActionKey/RNG/missing counts must all be zero and
the exact 50+5 paired, seat-balanced, profile-balanced grid must replay.

A pass opens only the next 25-paired data-shard pilot.

- It does not authorize the full 9,000-paired fanout.
- It does not make labels realized match EV.
- It does not make a model training-eligible.
- It does not add or activate a profile.
- A failure cannot be repaired by retuning thresholds on these seeds.

## Local correctness commands

```powershell
$env:PYTHONPATH = "src"
pytest -q tests/test_hu_m31_t3_step6d_fresh_quality_v1.py
pytest -q tests/test_hu_m31_t3_step6d_fresh_quality_gate_v1.py
python -m py_compile `
  src/ofc_regular/hu_m31_t3_step6d_fresh_quality_v1.py `
  src/ofc_regular/hu_m31_t3_step6d_fresh_quality_gate_v1.py
```

Cloud startup, IAM, launch, receiver, and cleanup are intentionally outside
this foundation. They must consume the sealed 15-job package and publish the
strict result schema; they may not reinterpret the seed grid or thresholds.
