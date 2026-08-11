# M4.1 T1 second-seat completion audit

Date: 2026-07-13

## Decision

M4.1 is **complete No-Go**. The redesigned local pilot proved deterministic
`c2/e4` teacher generation, five-policy root coverage, disjoint safety fitting
and threshold locking, strict multi-shard auditing, and exact runtime fallback.
It did not produce an acceptable T1 second-seat override policy.

The model is not promoted, no runtime profile is activated, `current` is
unchanged, and M5 remains blocked. This result is a bounded engineering and
quality decision, not a claim of Nash play, mathematical optimality, or solved
T1 strategy.

## Frozen M4.1 policy and data chain

Every root used the same explicit continuation contract:

- rollback baseline: `stage18_p1`;
- T2 continuation: `stage9f_p2`;
- T3/T4 continuation: the M3 Rust selectors;
- candidate-selection samples: 2;
- locked-evaluation samples: 4;
- batched information-set child selectors: enabled;
- native batch threads: 4.

Root generation used five profiles with equal requested and realized weight:

| Root profile | Policy family | Roots |
|---|---|---:|
| `stage19_p0` | selective opening fixed chain | 20 |
| `stage9f_p2` | selective T2 fixed chain | 20 |
| `stage7_m5_r10` | conservative T3 margin | 20 |
| `stage3_baseline` | baseline | 20 |
| `random_exact_final` | random/off-policy with exact final | 20 |

The population schedule was the deterministic
`weighted_quota_seeded_shuffle_v1` schedule. No row resolved `current`, and the
opponent's private discards were not policy or model inputs.

## Determinism and speed evidence

The one-root `c2/e4` smoke was run twice with the same root and configuration.
Both complete JSONL files had SHA-256:

`885e5ca59581bd83a62d0e87812969c14241c94b79c96770cbb481d8b8e1c740`

The two wall times were 48.0181 and 55.3122 seconds. The output recorded
`batched_infoset_locked_v1`, M3 Rust T3/T4 selectors, four native batch threads,
two candidate RNG digests, four evaluation RNG digests, and zero overlap.

This is practical local throughput evidence, not a controlled scalar-versus-
batch benchmark. It nevertheless shows that the higher `c2/e4` label budget was
feasible locally: four parallel pilot shards completed 100 roots, with the
slowest shard taking 3,111.0507 seconds (51.8508 minutes). The four shard times
were:

| Shard | Roots | Wall seconds |
|---|---:|---:|
| train A | 25 | 2,534.2126 |
| train B | 25 | 2,829.9705 |
| calibration | 30 | 3,111.0507 |
| locked holdout | 20 | 2,132.0981 |

## Pilot data audit

The fresh local pilot contained 100 roots:

- train: 50 roots in two independent shards;
- calibration: 30 roots;
- locked holdout: 20 roots;
- seed stride: 1,000,003;
- each of the five root profiles: exactly 20 roots.

The frozen data audit passed. It found zero cross-split hand-seed overlap and
zero cross-split observation-fingerprint overlap. Within every split, shard
paths, hand seeds, and observation fingerprints were unique. All legal actions
were mapped by `ActionKey`; candidate/evaluation RNG domains were disjoint; and
the hidden-discard, holdout-lock, and `current` guards passed.

Teacher values remain diagnostic search labels. They are not realized match EV
and were not used as a direct runtime LCB gate.

## Model and calibration result

The shared joint artifact used the 1,076-feature action encoder with policy,
value, baseline-delta, uncertainty, and safety heads. Diagnostic top-1 accuracy
was 48% on train and 20% on the locked holdout. These accuracy values are not
promotion criteria, but the large generalization gap is evidence that candidate
ranking needs improvement.

The 30 calibration roots were deterministically divided into two independent
sets:

- safety-estimator fit: 15 roots;
- threshold lock: 15 roots;
- hand-seed, observation-fingerprint, and row-hash overlap: 0;
- locked holdout used for either role: false.

On the threshold-lock subset, thresholds 0.0 through 0.3 fired all 15 candidate
opportunities. Their diagnostic mean delta was -1.611351 per fire, the
false-positive rate was 60%, and p95 loss was 13.5385. Threshold 0.4 and every
higher threshold fired zero times. There was therefore no threshold with at
least five fires, positive mean diagnostic delta, false-positive rate at most
30%, and the fixed tail bounds.

Calibration correctly returned No-Go. The frozen threshold is 1.0 and safety is
disabled. Lowering the threshold would activate a region that the independent
threshold-lock subset measured as negative and false-positive-heavy; it is not
a valid remedy.

The unpromoted model artifact SHA-256 is:

`4e1c85a9f2813ab6b8b6c3e6c4da2b75817956d214eb32a25968a0783c7ed46e`

## Fresh population smoke

The disabled artifact was evaluated on two fresh paired seeds per opponent
against `stage19_p0`, `stage9f_p2`, `stage7_m5_r10`, and
`random_exact_final`. Across 16 candidate/baseline comparisons and 32 traced
hands:

- overrides: 0;
- invalid counterfactuals: 0;
- non-fire trajectory mismatches: 0;
- non-fire non-zero deltas: 0;
- non-fire cancellation unknowns: 0;
- first-seat delta: exactly 0;
- second-seat delta: exactly 0;
- paired seat-swap delta: exactly 0.

The population confidence interval used hand-seed clusters after averaging the
opponent population. This smoke proves exact inactive-policy cancellation; it
does not provide evidence of a strength gain.

## Acceptance result

The fixed validator returned `complete_no_go`: 18 of 24 gates passed. The six
failed gates were:

- minimum 300 valid overrides;
- realized gain per override CI lower bound greater than zero;
- paired seat-swap delta CI lower bound greater than zero;
- second-seat delta CI lower bound greater than zero;
- a defined false-positive rate from actual fires;
- defined p95/p99/max loss tails from actual fires.

All provenance, data correctness, fit/lock separation, frozen-threshold,
artifact-hash, invalid-counterfactual, non-fire cancellation, first-seat exact
delegation, and opponent-population gates passed.

## Regression validation

- focused M4.1 integration suite: 70 passed;
- full Python suite: 1,030 passed in 43.55 seconds;
- Rust library: 36 passed;
- Rust runner: 2 passed;
- Rust Clippy with warnings denied: pass;
- Rust package format check: pass.

## Cost, rollback, and activation boundary

All M4.1 generation and evaluation was local. No Spot VM or other cloud compute
was started, so cloud cost was USD 0. The fixed rollback remains
`stage18_p1`; no M4.1 model was added to a runtime profile.

`src/ofc_regular/ai_profiles.py` still has SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`,
the frozen M1 registry hash. The `current` mapping is unchanged.

## Required M4.2 work

M4.2 should improve the learned decision, not weaken the gate:

1. Generate materially more train, safety-fit, threshold-lock, and locked
   holdout roots while preserving all seed/fingerprint separation and the five
   root-policy families.
2. Use out-of-fold or cross-fitted predictions for safety training so the safety
   head learns candidate errors without in-sample ranking optimism.
3. Improve candidate ranking and uncertainty calibration. The 48% train versus
   20% locked diagnostic top-1 result and the negative all-fire region both
   point to model/generalization quality, not just threshold selection.
4. Mine model/teacher disagreements, near-ties, high-SE actions, tail losses,
   and profile-specific failures; use higher independent evaluation MC where it
   changes the candidate ordering.
5. Repeat the bounded local gate before any Spot scale. Require a non-zero safe
   region before paying for large shards, then retain the fixed requirement for
   at least 300 fresh realized overrides.

M4.2 must not simply lower the safety threshold: the locked evidence already
rejects that action. It also should not switch to pure terminal-reward RL from
scratch. The corrected search teacher, exact late-street engine, population
coverage, and supervised candidate signal should be retained; self-play/CFR or
RL can be added later as a controlled residual or population-iteration layer.

## Frozen evidence

- `configs/hu_joint_policy_m41_status.json`
- `outputs/hu_joint_policy/m41_complete/smoke/c2e4_a.jsonl`
- `outputs/hu_joint_policy/m41_complete/smoke/c2e4_b.jsonl`
- `outputs/hu_joint_policy/m41_complete/pilot100/data_audit.json`
- `outputs/hu_joint_policy/m41_complete/pilot100/training_manifest.json`
- `outputs/hu_joint_policy/m41_complete/pilot100/population_smoke.json`
- `outputs/hu_joint_policy/m41_complete/pilot100/acceptance_config.json`
- `outputs/hu_joint_policy/m41_complete/pilot100/acceptance_status.json`
