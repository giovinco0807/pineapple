# M4 T1 second-seat completion audit

Date: 2026-07-13

## Decision

M4's implementation, correctness pilot, training path, locked evaluation path,
and fixed acceptance decision are complete. The policy decision is **No-Go**.
The pilot artifact is not promoted, no runtime profile is activated, and
`current` is unchanged.

This is not a claim that T1 second-seat play is solved. It is the intended
fail-closed result of the milestone gates: the bounded pilot did not produce a
safe candidate worth scaling. M5 must not start from this continuation.

## What was implemented

- `hu_m4_teacher_contract.py` freezes the T1-second live deal order and child
  policy identity.
- `hu_m4_t1_teacher.py` evaluates every legal root action with disjoint
  candidate-selection and locked-evaluation beliefs. T2 uses an explicit
  observation-only fixed policy; T3/T4 use the M3 Rust engine.
- `generate_hu_m4_t1_data.py` creates small restartable shards with checkpoint
  and heartbeat output. Resume validates a canonical config hash, checkpoint
  hash, contiguous root indices, seeds, profiles, search settings, and safely
  truncates an uncommitted tail.
- `audit_hu_m4_t1_data.py` rejects hidden truth, incomplete ActionKey mappings,
  RNG overlap, duplicate roots, and split leakage.
- `hu_m4_joint_model.py` and `train_hu_m4_joint_model.py` implement one
  versioned seat-aware artifact with policy, value, baseline-delta,
  uncertainty, and safety heads. The action/safety roles are locked to the same
  artifact SHA-256.
- `hu_m4_t1_policy.py` is a baseline-first composition. It can override only
  T1 second-seat, delegates T1 first-seat exactly, and fails closed on model,
  threshold, schema, non-finite, or legality failures.
- `evaluate_hu_m4_population.py` compares candidate and `stage18_p1` on the
  same hand, physical seat, opponent, and policy seeds. Invalid
  counterfactuals are excluded from primary metrics; population confidence
  intervals cluster by hand seed after averaging opponents.
- `validate_hu_m4_acceptance.py` applies fixed realized-play gates. Insufficient
  evidence produces `complete_no_go`, not a false promotion.

The M1-era `stage18_p1` safe-selector observation regression was also repaired:
the validated `ActorObservation` now reaches reranking, the selector, and its
decision log. No profile configuration changed.

## Correctness and determinism

The native one-root smoke evaluated all 27 legal T1 actions. Candidate and
evaluation RNG overlap was zero. Repeating the same root produced byte-identical
JSONL:

`b2230b646d3059d72b01fcaca6ee6e68aa2720239b3b8eea1172ddfef74e1cfe`

The full 112-state pilot used four small local shards:

- train: 64
- calibration: 24
- locked holdout: 24
- `--seed-stride`: 1,000,003
- cross-split seed overlap: 0
- cross-split observation-fingerprint overlap: 0
- hidden-truth records: 0
- candidate/evaluation RNG overlap: 0
- data audit: pass

Teacher values are recorded only as diagnostic search labels. They are not
reported as match EV and are not used directly as an LCB runtime gate.

## Pilot model result

The shared action encoder has 1,076 features. Diagnostic top-1 accuracy was
71.875% on train and 25% on the locked holdout. These figures are not promotion
criteria.

Calibration rejected the candidate:

- candidate opportunities: 22
- safe-label rate: 18.18%
- threshold 0.0-0.1: 22 fires, mean diagnostic delta -3.5103/fire,
  false-positive rate 81.82%, p95 loss 25.9157
- threshold 0.2 and above: zero fires
- frozen threshold: 1.0
- safety enabled: false

The safe response was therefore total fallback, not threshold relaxation.

## Fresh population smoke

The disabled artifact was replayed on two fresh paired seeds per opponent
against:

- `stage19_p0`
- `stage9f_p2`
- `stage7_m5_r10`
- `random_exact_final`

Across 16 candidate/baseline comparisons (32 traced hands):

- overrides: 0
- invalid counterfactuals: 0
- non-fire trajectory mismatches: 0
- non-fire non-zero deltas: 0
- non-fire unknowns: 0
- first-seat delta: exactly 0
- second-seat delta: exactly 0
- paired seat-swap delta: exactly 0

This proves fallback cancellation for the smoke; it does not prove a strength
gain.

## Acceptance result

17 of 23 gates passed. The six failures all follow from the candidate being
disabled and having zero realized overrides:

- minimum 300 valid overrides
- realized gain/override CI lower bound greater than zero
- paired seat-swap delta CI lower bound greater than zero
- second-seat delta CI lower bound greater than zero
- false-positive rate with actual fires
- p95/p99/max tail bounds with actual fires

No Spot VM was started and cloud cost was zero because the pre-scale Go gate
failed.

## Regression validation

- focused M4/related tests: 86 passed
- full Python suite: 993 passed in 53.91 seconds
- Rust library: 36 passed
- Rust runner: 2 passed
- Clippy: warnings denied, pass
- `ai_profiles.py` SHA-256 remains
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`

## Required M4.1 redesign before another scale decision

1. Improve label signal beyond the 1/1 correctness-pilot budget and measure
   maximization bias with independent higher-MC evaluation.
2. Diversify T1 roots across historical, conservative, exploitative, and
   randomized prefix policies instead of one root policy.
3. Separate safety-head fitting from threshold calibration (or use OOF safety
   predictions) while preserving the locked holdout.
4. Batch or move T2 continuation into Rust before a high-MC 1,000-state run.
5. Repeat the 100-state gate. Start Spot shards only if calibration produces a
   non-zero safe region with positive diagnostics and acceptable tails.
6. Require at least 300 fresh realized fires and all fixed population gates
   before adding any explicit runtime profile.

`stage18_p1` remains the rollback baseline. M5 is blocked by policy quality,
not by missing M4 infrastructure.
