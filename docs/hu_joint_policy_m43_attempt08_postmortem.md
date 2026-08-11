# M4.3 Attempt08 development closeout

Date: 2026-07-14

## Decision

Attempt08 is closed as `complete_no_go_development`. The fixed Development200
selector was executed exactly once. Thirteen of fourteen gates passed; the
strict maximum per-fired-root loss gate was the only failure. Audit50,
distillation, threshold selection, runtime activation, and profile promotion
remain unopened. `current` and the existing P0/P1/P2/T3 baselines are
unchanged.

## Result

- final fires: 41 (minimum 40)
- minimum fires for every opponent profile: 3 or more
- E-like A256 mean delta/state: +0.0668743732
- A256 mean delta/fire: +0.3262164546
- false-positive rate/fire: 0.3170731707 (limit 0.40)
- maximum fired-root p95 loss: 24.3067551537 (limit 25)
- maximum fired-root p99 loss: 37.2270206147 (limit 40)
- maximum fired-root loss: 52.4540412294 (limit 50; failed)
- mapping, RNG, hidden-information, reserve, and locked-action violations: 0

The sole failing root was index 72 against `stage7_m5_r10`, acting second.
Its V256 minimum was -37.2270 and its X512 minimum was -44.4540, so the fixed
-45 stress filter passed. The independent A256 batch then observed -52.4540.
This is a rare-tail miss across independent Monte Carlo batches, not a hidden
discard leak, action-index drift, or counterfactual-cancellation defect.

## Boundary

The opened Attempt08 rows may be used only as postmortem architecture evidence.
They must not be relabelled as a pass, re-run with alternate seeds, used to
reselect a threshold, or substituted for a fresh audit. The immutable artifact
bindings are recorded in
`configs/hu_joint_policy_m43_attempt08_closeout.json`.

## Fresh Attempt09 design implication

Attempt09 keeps the frozen ranker, R128, K4, and V256 eligibility unchanged.
Instead of trusting one locked X512 result, it carries every V-eligible K4
candidate in original R order through two independent profile-blind tail
filters, X512 and C256. The first candidate surviving both filters is locked;
an independent E256 batch is diagnostics-only and supplies the new one-shot
Development200 gates. This preserves the separation between search MC and
evaluation MC and gives an X-cancelled root a safe candidate fallback without
using opponent identity at runtime.

Descriptively only, applying the unchanged -45 confirmation bound to the
already-open Attempt08 A256 vectors would remove root 72 and leave exactly 40
fires. Those vectors are not an Attempt09 evaluation and make no fresh strength
claim; Attempt09 uses wholly disjoint seed namespaces.
