# M4.2 T1 second-seat completion audit

Date: 2026-07-13

## Decision

M4.2 is **complete No-Go**. It completed the paired-delta teacher contract,
strict nested identity-group cross-fitting, resumable Spot shards, immutable
source/model/native manifests, fresh-seed population provenance, exact non-fire
counterfactual cancellation, and the fixed acceptance validator. The resulting
ranker did not produce a safe positive override region.

The candidate is not promoted, no runtime profile is activated, `current` is
unchanged, further scaling of this ranker is not authorized, and M5 remains
blocked. This is an engineering and measured policy-quality decision, not a
claim of Nash play or mathematical optimality.

## Frozen chain and correctness smoke

The rollback and continuation chain remained:

- T1 baseline: `stage18_p1`;
- T2: `stage9f_p2`;
- T3: M3 Rust `stage7_m5_r10` continuation;
- T4: M3 Rust exact selector;
- hidden opponent discards: excluded;
- `current`: never resolved.

The local paired-contract smoke used one root at c1/e2 and ran twice. The two
JSONL files were byte-identical with SHA-256
`148b5ada18ec6523bcfde4f37673dc7d7fbd7ae413745a9e920cecd9f9d44d4c`.
Wall times were 32.2741 and 30.2040 seconds. The baseline paired delta and SE
were exactly zero; non-baseline paired SE values remained non-zero.

## Gate40 Spot teacher evidence

Successful run:
`regular-hu-m42-c2e8-gate40-20260713-1335`.

- 40 roots: train 20, calibration 10, locked holdout 10;
- 8 shards of 5 roots;
- candidate/evaluation samples: 2/8 with independent RNG domains;
- seed stride: 1,000,003;
- five root policy families, exactly 8 roots each;
- paired-delta schema v2 required for every record;
- zero cross-split hand-seed and observation-fingerprint overlap;
- unique shard paths, seeds, and fingerprints within each split;
- hidden-discard, legal-action mapping, holdout-lock, and `current` guards: pass;
- receipt: 8/8 shards and 40/40 roots verified;
- data audit SHA-256:
  `b0df1118df38b1855a31af00a52de45ba31dba002e3c40be98a54cfb7b9e08ee`.

The source/native chain was immutable and verified. The source SHA-256 was
`902c200873927fb31ffd1204588d4f0492880ab7ff29fb579125742c54e12530`;
the Linux native manifest SHA-256 was
`2176f712b0ec2345c9f3c7dbd17a6269149108d874ba9cfd9b8ecc2a710c2735`.

## Nested ranker and calibration result

Model run:
`regular-hu-m42-ranker-i30-gate40-20260713-1423`.

The opt-in `negative_regret_ranker_v2` used 30 boosting iterations and five
outer folds. Nested predictor-lineage auditing passed: every train identity was
predicted exactly once OOF and no validation seed, observation fingerprint, or
row hash entered its base, meta, uncertainty, or residual-target lineage.
Safety fitting used train OOF examples plus the independent safety-fit subset;
threshold selection used only threshold-lock. Locked holdout was used for
neither fitting nor threshold search.

Diagnostic model metrics were:

| Split | States / actions | Top-1 | Score MAE | Score RMSE |
|---|---:|---:|---:|---:|
| train | 20 / 531 | 0.0000 | 3.503104 | 4.313195 |
| locked holdout | 10 / 261 | 0.1000 | 3.774028 | 4.684634 |

Top-1 is diagnostic only and was not a promotion gate. Teacher values are also
diagnostic search labels, not realized match EV and not a runtime LCB gate.

Calibration split into five safety-fit and five threshold-lock roots with zero
identity overlap. Safety-fit had four candidate overrides and a 50% positive
label rate. Threshold-lock had five candidates and no positive labels.
Thresholds 0.0 through 0.4 fired all five candidates:

- teacher diagnostic mean delta per fire: -4.267027;
- false-positive rate: 100%;
- p95 / p99 / max loss: 12.3681 / 13.4417 / 13.7101.

Threshold 0.5 and above fired zero candidates. Calibration therefore returned
No-Go, froze threshold 1.0, and disabled safety. The single model artifact
SHA-256 is
`84ec54ace99e3fe8b2967b4252c6dff89d27a76acbf112407bf5cc4fa349f5f4`.

## Failure analysis

This is not a threshold-only failure.

1. The model had 1,076 input features but only 20 independent train states.
   Strict nested roles reduced some effective base/meta/target fits to 5/5/6
   state identities; action rows from one state are not independent samples.
2. The objective ranked all legal actions, while the runtime decision is a
   baseline-relative selective override. The baseline action was not an
   explicit paired input and was not structurally anchored to score zero.
3. The meta ranker learned from OOF base-head distributions but was applied to
   full-refit base heads. Its train top-1 of zero shows that this composition
   was unstable even before holdout generalization.
4. The safety classifier received 20 combined examples and retained the
   histogram booster default `min_samples_leaf=20`. All 30 trees had one node
   and every candidate safety probability was exactly 0.4, permitting only an
   all-fire or zero-fire threshold.
5. c2/e8 paired labels remained noisy. Scaling the same ranker would spend more
   compute without first correcting its baseline-relative objective and
   calibration capacity.

For diagnostic context only, removing the safety gate on the locked split
would select nine overrides: two positive and seven non-positive. Their mean
teacher delta was -3.5713 per fire and -3.2142 per state, with 77.78% false
positives. This postmortem consumes the M4.2 locked result; it must not be
reused to select M4.3 architecture or thresholds.

## Fresh population and acceptance

The frozen disabled artifact was evaluated on two fresh paired seeds per
opponent against `stage19_p0`, `stage9f_p2`, `stage7_m5_r10`, and
`random_exact_final`. The population seed was 2,126,071,901 with stride
1,000,003. The new explicit provenance gate compared both evaluation seeds
against all 40 teacher hand seeds and found zero overlap.

Across 16 candidate hands, 16 baseline hands, and 32 traces:

- overrides: 0;
- invalid counterfactuals: 0;
- non-fire mismatches, non-zero deltas, and unknowns: 0;
- first-seat, second-seat, and paired seat-swap delta: exactly 0;
- CI unit: hand-seed cluster after opponent averaging.

This proves inactive-policy cancellation, not strength. The fixed validator
returned `complete_no_go`: **19 of 25 gates passed**. The extra passing gate
relative to M4.1 was `fresh_evaluation_seed_disjoint`. The six failures were:

- minimum 300 valid overrides;
- realized gain per override CI95 low greater than zero;
- paired seat-swap delta CI95 low greater than zero;
- second-seat delta CI95 low greater than zero;
- defined false-positive override rate;
- defined override-loss tails.

Teacher metrics and top-1 accuracy were excluded from promotion.

## Compute and billing boundary

Exact cloud billing was not read back from a billing export or invoice, so the
M4.2 USD cost is deliberately recorded as `null` / unverified rather than
estimated as an exact amount.

Observed compute evidence is:

- successful teacher: 8 Spot `c4-standard-4` workers, 40 roots, shard times
  233.9866 to 529.4849 seconds, 1,197.4180 seconds from frozen manifest time to
  the last DONE; all workers self-deleted;
- model: one Spot `c4-highcpu-16` worker, 136.7373 seconds from manifest to
  DONE, deleted after completion;
- two failed preflight runs: eight GCS worker status objects each and zero
  completed roots;
- local population evaluation: no cloud VM, 13.1858 seconds.

The observed total is 25 Spot VM/worker status records: 8 + 8 failed
preflights, 8 successful teacher workers, and 1 model worker. Live closure
readback found zero active M4.2 VMs.

These observed counts and timings are operational evidence only, not an exact
billing statement.

## M4.3 redesign boundary

The highest-value next step is an opt-in `baseline_paired_delta_risk_ensemble`:

1. encode candidate versus explicit baseline action and fix baseline score to
   exactly zero;
2. train precision-weighted paired-delta, soft positive-gain, and downside-tail
   heads rather than optimizing all-action top-1 alone;
3. use the cross-fit fold ensemble at runtime so meta features do not move from
   OOF to a different full-refit distribution;
4. start safety with a low-capacity regularized calibrator trained only on OOF
   plus safety-fit predictions;
5. select the threshold only on threshold-lock, then evaluate once on a fresh
   M4.3 locked holdout;
6. require a new small fresh pilot signal before Spot scale, without lowering
   any acceptance gate.

M4.3 runtime must not consume teacher values or teacher LCBs. M5 and large
scale remain blocked until a fresh M4.3 candidate passes its bounded gate.

## Current and frozen evidence

`src/ofc_regular/ai_profiles.py` remains at SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.
No `current` mapping or runtime profile was changed.

## Final validation

- focused M4.2 Python regression: 56 passed in 0.75 seconds;
- full Python regression: 1,075 passed in 116.72 seconds;
- `ofc_hu_m3_engine`: 36 library tests and 2 runner tests passed;
- target-package Rust format check and clippy with warnings denied: passed;
- `git diff --check`: passed;
- acceptance status SHA-256:
  `3d1fcc15f5c3dbc410d08f0545b5ab84fcbe1bf46fcc86d25a2c716357f261c1`.

The whole-workspace Rust format check is not an M4.2 gate: it reports existing
unformatted files in other user work. Those files were deliberately preserved;
the M4.2 target package itself passes its format check.

- `configs/hu_joint_policy_m42_status.json`
- `outputs/hu_joint_policy/m42_complete/validation_summary.json`
- `outputs/hu_joint_policy/m42_contract_smoke/`
- `outputs/hu_joint_policy/m42_spot/regular-hu-m42-c2e8-gate40-20260713-1335/receipt.json`
- `outputs/hu_joint_policy/m42_spot/regular-hu-m42-c2e8-gate40-20260713-1335/data_audit.json`
- `outputs/hu_joint_policy/m42_spot/regular-hu-m42-c2e8-gate40-20260713-1335/model_runs/regular-hu-m42-ranker-i30-gate40-20260713-1423/training_manifest.json`
- `outputs/hu_joint_policy/m42_spot/regular-hu-m42-c2e8-gate40-20260713-1335/model_runs/regular-hu-m42-ranker-i30-gate40-20260713-1423/population_smoke.json`
- `outputs/hu_joint_policy/m42_spot/regular-hu-m42-c2e8-gate40-20260713-1335/model_runs/regular-hu-m42-ranker-i30-gate40-20260713-1423/acceptance_status.json`
