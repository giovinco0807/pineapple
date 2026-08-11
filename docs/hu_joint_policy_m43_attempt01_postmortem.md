# M4.3 attempt01 postmortem

Date: 2026-07-13

## Decision

The first M4.3 distributed model attempt is a calibration **No-Go**.  The
model, threshold, and runtime profile were not frozen or activated.  The
locked holdout was not evaluated, its one-shot marker was not claimed, and
`current` remains unchanged.

This is a model/scoring failure rather than a teacher-data, action-mapping, or
Spot execution failure.  The immutable completed artifacts remain useful as a
reproducibility record, but they are not promotion evidence.

## Verified execution

- Teacher run: `regular-hu-m43-c2e16-pilot200-20260713-1810`
  - train 100 / calibration 60 / locked holdout 40
  - candidate samples 2 / independent evaluation samples 16
  - receipt status `verified_and_audited`
- Fold-model run: `regular-hu-m43-fold-model-20260713-2052`
  - 30 of 30 fold jobs completed
  - a preempted shard was resumed without changing the frozen run
  - every estimator, manifest, `DONE`, source, and local rebind hash verified
- Model SHA-256:
  `a17119badf8efe31e78b18d04dbbe873ce3db9c2d36c0b9b06895d32ed0ce657`
- Training-manifest SHA-256:
  `70698a4ec2adb096616c4368c74c04b7f92b303cc11643f5a17028f9a2d5a3ff`

## Failure mechanism

The teacher data contains usable non-baseline opportunities:

- teacher-best non-baseline action: 126 / 160 train+calibration states
- strictly positive paired-delta opportunity: 125 / 160 states
- positive non-baseline actions: 830 / 4,010 actions

Nevertheless, the trained model selected the exact-zero baseline on all 160
states.  Every non-baseline composite score was negative.  On calibration the
largest score was `-0.2801` and the median state maximum was `-12.4485`.

The proposal score mixed predicted gain with an absolute downside penalty.
Calibration predictions averaged approximately `-3.38` points of paired
delta, `31.06` points of downside, and `1.51` points of fold disagreement.
The `0.5 * downside` term therefore suppressed every proposal before the
safety model could see it.  Candidate and safety rows both became zero, the
safety fallback became constant probability zero, and the only valid locked
threshold was `1.0` with safety disabled.

ActionKey resolution, legal-action coverage, card-order remapping, baseline
mapping, paired-label fields, RNG-domain separation, fold lineage, and artifact
hashes all passed.  No evidence points to corrupted labels or index drift.

## Locked boundary

The locked 40-row file was used only for structural hashing by the sealed data
contract.  It was not passed to model fitting, calibration, or threshold
selection.  No freeze manifest, locked receipt, or consumption marker exists.
Its immutable hashes are:

- file SHA-256:
  `0a88e1cc8b079906e0ca14ddf4a8c3df7b108a09d0f363c7cb665cab89b40a1e`
- canonical rows SHA-256:
  `fa99408bc4be031a7b4f2a8dfe05f4af07d90ebb7450e1eb03ba1fa596a87890`
- identity SHA-256:
  `3946c110de49831359316439fa230d9096fc41046b0cb07a92506eb931e0e629`

## Attempt02 boundary

Attempt02 is a clean, predeclared experiment rather than a threshold retune on
the consumed calibration set.

- Exclude all 252 M4/M4.1/M4.2 roots and all 160 attempt01 train/calibration
  roots.  The 412-identity union digest is
  `c7598ba0f74528562b79966e53dd843fe2225f23033b791377327f6ea6ab3b3e`.
- Generate fresh train 200 and calibration 100 roots on Spot, balanced over the
  five frozen prefix policies, using candidate samples 2 and evaluation
  samples 64.
- Keep calibration roles sealed as safety-fit 50 and threshold-lock 50.
- Rank all non-baseline proposals using paired-delta prediction only.  Move
  downside and fold disagreement into the separate safety decision.
- Exclude the exact baseline row from paired-regressor fitting and normalize
  non-baseline action weights inside each state.
- Generate out-of-fold safety rows independently of whether the proposal beats
  baseline, then require an explicit train-OOF proposal gate before calibration
  is opened.
- Reuse the unopened locked 40 only through a versioned inherited-holdout
  contract with a global one-shot marker.  A model or threshold change after
  seeing it requires a new locked set.
- Even a locked pass cannot promote the policy.  A separately seeded policy
  population must still produce at least 300 valid realized overrides and pass
  the fixed M4 EV, false-positive, tail-loss, robustness, and cancellation
  gates.

