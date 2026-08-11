# HU joint-policy M3.1 T3 Step 5 completion audit

Date: 2026-07-16

Decision: **Step 5 is complete locally. The numeric science gates and all
development, calibration, locked-holdout, and approximate-best-response seed
namespaces are frozen. Only immutable Linux Spot shard 0 is now authorized;
production data fanout, model training, strength claims, and activation remain
No-Go.**

The validator passed the live Step 4 status, summary, audit, native library,
and policy-registry SHA locks. It enumerated all 66,000 reserved seed values and
found them globally unique. The smallest planned seed is `300108071901`, above
the maximum seed-like integer in the 109 pre-existing config files,
`256108071901`.

No VM was created, no GCP charge was incurred, no holdout was opened, and no
profile or `current` mapping was changed.

## Why the local1000 budget is not the production label budget

Step 4 established that `4/8/2/0` is deterministic, restartable, and locally
scalable. It did not establish that eight locked-evaluation particles are
sufficient for final model labels. Step 5 therefore assigns budgets by role:

| Role | candidate | evaluation | downstream T3 | downstream T4 |
| --- | ---: | ---: | ---: | ---: |
| Linux infrastructure canary | 4 | 8 | 2 | 0 (exact) |
| Production teacher labels | 8 | 32 | 4 | 0 (exact) |

Ten percent of every teacher split must also receive an independent 128-sample
all-legal-action confirmation pass in a third RNG namespace. Confirmation
regret, not top-1 agreement alone, controls label-quality Go/No-Go. Canary rows
are infrastructure-only and cannot enter training.

## Frozen teacher and evaluation splits

All teacher schedules generate both physical seats for every paired hand and
use equal quotas across `stage19_p0`, `stage9f_p2`, `stage7_m5_r10`,
`stage3_baseline`, and `random_exact_final`.

| Split | Paired indices | T3 roots | Use |
| --- | ---: | ---: | --- |
| infrastructure canary | 250 | 500 | Linux parity, resume, and throughput only |
| train | 6,000 | 12,000 | reference/candidate action learning |
| safety fit | 1,000 | 2,000 | safety/uncertainty estimator only |
| threshold lock | 1,000 | 2,000 | threshold selection only |
| diagnostic teacher holdout | 1,000 | 2,000 | opened after model and threshold freeze; diagnostic only |
| development population | 250 per opponent | realized development evaluation |
| locked population | 1,000 per opponent | realized promotion holdout |
| locked ABR probe | 500 per response | approximate exploitability proxy |

The seed stride is `1000003`. Every schedule has six independent namespaces.
Teacher schedules separate hand, behavior, candidate, evaluation, child-policy,
and confirmation RNG. Population schedules use the same separation while
candidate and baseline deliberately share the frozen actor-policy randomness
inside each paired counterfactual.

Seed audit:

| Metric | Result |
| --- | ---: |
| Historical config files scanned | 109 |
| Historical maximum | 256,108,071,901 |
| Planned values | 66,000 |
| Unique planned values | 66,000 |
| Planned minimum | 300,108,071,901 |
| Planned maximum | 470,607,073,398 |
| Planned set SHA-256 | `6610c740fbb1b22405a0eb960fd3beff5f1c747328ed140187a8f8a277385b24` |

Alternate seeds after observing results and post-hoc sample extension are both
forbidden. An underpowered or failed locked evaluation is a completed No-Go.

## Frozen label and model gates

Teacher data requires zero hidden-truth rows, unknown fields, invalid action
mappings, non-finite values, duplicate observation fingerprints, and RNG-domain
overlap. Exact-T4 and complete legal-action value coverage must both be 100%.
First/second roots must be exactly balanced.

On the independent confirmation subset, selected-action regret must satisfy:

- mean at most 0.75 points;
- p95 at most 3 points;
- p99 at most 6 points;
- maximum at most 15 points.

Ambiguous rows are retained with an uncertainty weight floor of 0.1 rather than
silently discarded. The candidate starts as one shared seat-aware model with
seat-specific calibration. A split-seat model is allowed only under the frozen
interference test. Model top-1 placement accuracy and teacher EV/LCB remain
diagnostic and cannot authorize runtime use.

## Frozen realized promotion gates

The locked population uses 1,000 paired seeds for each of five opponent
profiles with full seat swap and candidate/baseline common randomness. It must
have:

- zero invalid counterfactuals;
- exact full-trajectory cancellation on every non-fire;
- at least 300 valid overrides overall and 100 for each seat;
- positive exclusive 95% CI lower bounds for overall paired EV/hand, first-seat
  EV/hand, second-seat EV/hand, and realized gain per override;
- false-positive override rate at most 0.30 overall and 0.35 for each seat;
- fired loss p95/p99/max at most 25/40/50 points;
- each opponent mean delta at least -0.005 EV/hand and CI lower bound at least
  -0.02 EV/hand.

Threshold reselection, replacement seeds, and sample extension after content
read are prohibited.

The exploitability proxy freezes three response families: greedy search, foul
pressure, and royalty denial. They must be trained without locked probe seeds
and frozen before the 500-paired-seed response holdout. This is an approximate
best-response robustness check, not Nash or mathematical optimality evidence.

## Spot authorization boundary

Step 5 authorizes only `infrastructure_canary` shard 0 on a Spot
`c4-standard-16` VM:

- 25 paired hands / 50 roots;
- immutable source and native hashes;
- Linux semantic parity against frozen golden roots;
- an intentional preemption/resume drill;
- atomic write-once paired-hand checkpoints;
- heartbeat upload every 60 seconds;
- `DONE` committed last.

The remaining nine canary shards require verified shard-0 completion. Even a
passed 500-root infrastructure canary does not authorize production fanout. A
separate 100-root `8/32/4/0` production-label quality pilot must pass the frozen
confirmation-regret gates first.

## Validation

- Step 5 unit tests: `11 passed in 0.41s` after final source formatting.
- Combined M3/M3.0/M3.1 focused suite: `124 passed in 32.80s`.
- Full Python suite: `2464 passed, 2 skipped, 50 warnings in 831.79s`.
- Black check passed for the new validator and tests.
- The live contract validator passed all prerequisite, semantic, and seed gates.
- `src/ofc_regular/ai_profiles.py` remains SHA-256
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

## Source and artifact anchors

| Path | SHA-256 |
| --- | --- |
| `configs/hu_joint_policy_m31_t3_step5_contract.json` | `5f9fab4d844f1a7411a99f9dada2fe289314d4dea578d0c0f6a6d14918263c93` |
| canonical contract JSON | `04c4298feaed78f327f7fb6601f70a6821c3a91dcd2996cc34becbc4bb38e0b4` |
| `src/ofc_regular/validate_hu_m31_t3_step5_contract.py` | `6b1cce48a4749b9d656527e9f7666e456020e0eb5a14307afffc325aaf9db5b5` |
| `tests/test_hu_m31_t3_step5_contract.py` | `533dd713d7ac4a8a63397a02543cda74af5c7fdcc074f9508a2815f0098cda8d` |
| `outputs/hu_joint_policy/m31_t3_step5/contract_validation_v2.json` | `055b900e06db174f69d3ecb25fd1466f9e812d272222529121a40b71dd003bcc` |

## Remaining boundary and next step

M3.1 remains incomplete. No new T3 reference, candidate, uncertainty head, or
safety head exists yet, and no realized strength evaluation has run.

The next action is Step 6a: build an immutable Linux package and resumable Spot
runner for exactly shard 0, dry-run it locally without teacher generation, then
launch only that 50-root canary. Production fanout remains explicitly disabled.

