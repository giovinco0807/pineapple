# M4.3 attempt02 pre-calibration No-Go

Date: 2026-07-14

## Decision

Attempt02 is complete with decision `no_go_precalibration`. The fresh-train
out-of-fold proposal gate failed before calibration was opened, so no model was
written and no safety fit, threshold selection, locked evaluation, population
evaluation, or runtime activation was allowed.

This result is an offline proposal-policy diagnostic. Its teacher deltas are
not realized match EV and are not a runtime gate.

## Immutable r2 evidence

- Run: `regular-hu-m43-attempt02-v4-model-r2-20260714-0117`
- Receive receipt SHA-256:
  `6d04af784850fe7c8fa38087e93d96c584bc1091132eae41eee4d13083d22654`
- Training manifest SHA-256:
  `6b061a690006ec21d4f52305bc21a2fd7c8ec0cd374fe47ad22dced1d7ce84ae`
- The receipt binds that exact training-manifest SHA-256 and reports
  `model_sha256: null`.

The assembler completed all 30 jobs with exact outer/inner coverage. The
failure is therefore the predeclared out-of-fold proposal gate, not an
incomplete Spot run or an artifact-receive failure.

## Failed gates

Two of 13 gates failed; the other 11 passed.

1. `minimum_proposal_positive_rate`: 69 of 200 proposals were positive,
   `0.345`, below the required `>= 0.40`.
2. `positive_mean_teacher_delta`: the unweighted mean was
   `-1.1242242603887602`, below the required `> 0`. The normal 95% interval,
   computed as the sample mean plus or minus 1.96 sample standard errors, was
   `[-1.4943, -0.7542]`.

The 11 passing gates were:

- all out-of-fold identities excluded from fit;
- exact-zero baseline score;
- exactly 200 states and the expected five-fold set;
- minimum all-action safe and unsafe row counts;
- minimum states per fold;
- minimum tail-safe and tail-unsafe proposal counts;
- a non-baseline proposal for every state; and
- exactly-once out-of-fold state coverage.

These correctness and coverage passes do not override the negative proposal
quality result.

## Fail-closed boundary

The training manifest records all of the following as false:

- calibration data-contract opened;
- fresh calibration rows opened;
- safety fit performed;
- threshold selection performed;
- inherited locked-holdout content opened or accepted as input;
- model written;
- `current` profile mutated;
- runtime policy activated; and
- full replacement enabled.

No model artifact exists in the received r2 directory. The baseline policy
registry remains bound to SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

Attempt02 must not be rescued by lowering a gate, opening calibration, or
selecting a threshold after observing this result. Any later experiment is a
separate attempt with its own predeclared evidence boundary; this status does
not modify or authorize it.
