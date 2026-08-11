# HU joint-policy M3.1 T3 Step 6c completion audit

Date: 2026-07-17

Decision: **Step 6c is complete as a fixed No-Go experiment. The 100-root
production-label quality pilot passed every confirmation-regret, integrity,
RNG, second-seat latency, and memory gate, but failed the frozen first-seat
primary p95 latency gate. No same-data threshold reselection, reseed,
extension, training, production fanout, profile activation, or `current`
change is authorized.**

The accepted run is
`regular-hu-m31-step6c-quality-20260717-003`. It used two `c4-standard-16`
Spot VMs, 25 paired hands / 50 roots per shard, two worker processes per VM,
and eight Rayon threads per worker. Both shards recovered one deliberately
interrupted task, committed `DONE.json` last, and self-deleted. No Step 6c VM
remains.

`current` was neither resolved nor changed. No model was trained, no profile
was added, and the pilot rows remain ineligible for training. Teacher values
remain `diagnostic_not_match_EV` and are not realized HU match EV.

## What was completed

- Frozen the 6,000-hand equal-quota block-shuffle schedule and the disjoint
  50-hand pilot grid, with exactly ten hands from each of five behavior
  profiles and balanced first/second roots.
- Ran the production-label budget `8/32/4/0` on all 100 roots and an independent
  `8/128/4/0` confirmation evaluation on the precommitted ten-root subset.
- Locked the primary ActionKey during confirmation; the 128-sample argmax could
  not replace the selected action.
- Recomputed every recovered root from its frozen seed/profile and rejected
  hidden opponent discards, unknown fields, stale Q-value certificates,
  action-index drift, and RNG overlap.
- Bound the source ZIP, models, native libraries, schedule, manifest, local
  Linux dry-run, launch authorization, shard summaries, result files, and
  receive receipt by exact schemas and SHA-256.
- Proved restart behavior with write-once paired-hand checkpoints. Partial
  result recovery merges immutable remote results after progress recovery, and
  upload order is `parity -> roots -> tasks -> summary -> DONE`.
- Received and independently revalidated the exact 100-root grid locally,
  including all action values and all 5,280 RNG keys.

## Launch hardening and accepted package

Two earlier local-only packages were superseded before launch:

- `...-001` predated the final RSS, resume-count, and partial-upload hardening;
- `...-002` predated the bounded non-confirmation resume-drill order.

Neither package was authorized or published. Only `...-003` is acceptance
evidence. Its formal Linux dry-run fresh-extracted the immutable source ZIP,
passed portable parity in `119.7574` seconds for first seat and `3.1052`
seconds for second seat, then completed two atomic tasks with
`resumed_task_count=1`.

The resume drill uses the lowest non-confirmation hands to keep the operational
smoke bounded. Full production execution retains confirmation-first ordering;
the frozen confirmation subset, budgets, seeds, ActionKeys, and final summary
order are unchanged.

## Merged 100-root result

| Gate | Observed | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Paired hands / roots | 50 / 100 | exact | pass |
| First / second roots | 50 / 50 | exact | pass |
| Profile hands / roots | 10 / 20 each | exact | pass |
| Unique observation fingerprints | 100 | 100 | pass |
| Candidate RNG keys | 800 unique | 800 | pass |
| Evaluation RNG keys | 3,200 unique | 3,200 | pass |
| Confirmation RNG keys | 1,280 unique | 1,280 | pass |
| Cross-domain RNG overlap | 0 | 0 | pass |
| Confirmation regret mean | 0.2802 | at most 0.75 | pass |
| Confirmation regret p95 | 2.3299 | at most 3.0 | pass |
| Confirmation regret p99 | 2.3299 | at most 6.0 | pass |
| Confirmation regret max | 2.3299 | at most 15.0 | pass |
| First primary p95 latency | 326.2574 s | at most 180 s | **fail** |
| Second primary p95 latency | 3.7300 s | at most 6 s | pass |
| Peak process RSS | 504,397,824 bytes | at most 1 GiB | pass |
| Recovered task | 1 per shard | at least 1 | pass |

The confirmation result is strong: mean, p95, p99, and max regret all pass by
substantial margins. The No-Go is operational, not an action-quality failure.
The first-seat distribution had mean `157.2997`, p95 `326.2574`, p99/max
`353.0594` seconds. Running two eight-thread workers on a 16-vCPU VM did not
meet the precommitted p95 bound even though the isolated Linux parity root did.

The gate is not relaxed after seeing the data. Both shard summaries are
`no_go`, and the merged decision is
`production_label_quality_pilot_no_go_no_same_data_reselection`.

## Validation and adversarial review

- Step 6c focused tests: `113 passed`.
- Combined M3/M3.0/M3.1 focused tests: `286 passed in 75.08s`.
- Full Python regression: `2636 passed, 2 skipped, 50 warnings in 912.06s`.
- Black passed for all Step 6c Python source and tests.
- Bash 5.2 syntax validation passed for the startup script.
- Three independent review passes added negative coverage for decision/Q
  tampering, seed-inconsistent roots, provenance mixing, shard No-Go reversal,
  parity forgery, status/gate contradictions, RSS summary forgery, boolean
  resume counts, post-summary recovery, and partial immutable uploads.

The 50 warnings are the pre-existing sklearn feature-name warnings in
`test_hu_m43_attempt05_training.py`; Step 6c introduced no new warning class.

## Artifact anchors

| Artifact | SHA-256 |
| --- | --- |
| Immutable source ZIP | `b603a174120074508e45b98a8ff2a534a692c5357bc86c29373c3e29bacc16eb` |
| Package manifest | `467246d002abbb5e27e22aa0d96ced8919616c7321af2b133d2960aafb6de75b` |
| Local dry-run receipt | `06de27ed315c206ae0152f4bac1ed5d19b39da4cb108fcd835082f08544b929c` |
| Launch authorization | `e5d64771e250fc5bc764d18725ea6f3464fba95d4be6408847305a7d10aea488` |
| Receive receipt | `755d44f67a33a3470e14f0c4ece790f86f3dd3dbf9c51374761ef6670a51e71f` |
| Merged quality validation | `66ddae312ee9561a0101ba94deb5e5e867d407dda78af00b03da10fc96739731` |
| Shard 000 `DONE.json` | `4e1bf037d04437963b324a91718d4765729a9978b78869442289d62b4ffb7ed6` |
| Shard 001 `DONE.json` | `cd4b6a75a8b05e79cb63414c6c6fe10968d5d65697d405e7047665e730c0a2da` |
| Native T3 library | `3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0` |
| Feature encoder | `82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411` |
| Policy registry / `current` mapping | `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3` |

Received shards are under
`outputs/hu_joint_policy/m31_t3_step6c/received_v1`. The merged result is
`outputs/hu_joint_policy/m31_t3_step6c/quality100_validation_v1.json`. The
accepted immutable package and authorization are under
`outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003`.

## Boundary and next step

Step 6c is not repeated to search for a friendlier threshold, seed, subset, or
VM outcome. Its scientific quality evidence may inform engineering, but its
rows cannot train a model because the frozen operational gate failed.

The next action is a separately versioned M3.1 performance-repair step. Profile
first-seat primary search by geometry and concurrency, then optimize the Rust
engine or the worker/Rayon allocation under a precommitted benchmark. Only
after that repair passes on disjoint performance states may a new, disjoint
quality pilot be proposed. M3.1 artifact rebuilding, strength evaluation,
profile activation, and production fanout remain No-Go.
