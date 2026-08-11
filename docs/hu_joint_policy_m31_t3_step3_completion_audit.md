# HU joint-policy M3.1 T3 Step 3 completion audit

Date: 2026-07-16

Decision: **Step 3 is complete locally as a 100-root correctness and
performance boundary. M3.1 model rebuilding, strength evaluation, and policy
promotion are not complete.**

The accepted artifact contains exactly 100 general-board T3 roots, balanced
50/50 by seat, at the frozen `4/8/2/0` search budget. It passed every
pre-registered integrity gate in 3,498.657 seconds (58.31 minutes). No profile,
`current` mapping, policy wrapper, Spot VM, or cloud job was changed or started.

## Exactness boundary

- Root T3 remains common-random Monte Carlo over hidden-card particles.
- Candidate selection uses 4 particles and locked evaluation uses 8 disjoint
  particles.
- Each nested T3-second information set uses 2 particles.
- Every downstream T4 child uses the accepted exact native T4 semantics with
  `downstream_t4_samples = 0`.
- The artifact is diagnostic teacher search, not realized match EV, a policy
  strength result, or a proof of full-game optimality.

## Sample-budget decision

The initial `1/1/1` integrity probe passed, but it was not stable enough for the
100-root run. On the frozen two-root ladder, the first-seat `1/1/1` selection
had regret 8.0 under the independent `2/3/2` evaluation batch. It was therefore
rejected rather than promoted from action agreement or top-1 accuracy.

The pre-registered final gate compared `2/3/2` with the existing teacher
default `4/8/2`. For both seats:

- the selected action agreed;
- the `2/3/2` selection had regret 0.0 under the `4/8/2` evaluation batch;
- the `4/8/2` locked selection had evaluation regret 0.0;
- candidate and evaluation particle prefixes were nested and disjoint.

This authorized `4/8/2/0` only for the local 100-root correctness/performance
pilot.

## Rust performance repair

The original sequential engine projected the accepted budget above the frozen
60-minute limit. Its 8-process probe was an honest No-Go at 111.07 projected
minutes. Step 3 did not start the 100-root job from that result.

The bottleneck was exact T4-first child evaluation. Two order-preserving loops
were parallelized with Rayon:

1. generation and scoring of opponent responses for all 2,024 future deals;
2. scoring each legal hero action against the ordered response table.

The sum order inside each action is unchanged, so all action values remain
bit-identical. Across both seats and all three `1/1/1`, `2/3/2`, and `4/8/2`
budgets, all six old/new decisions had exactly identical action-value rows,
selected actions, gaps, and regrets. Runtime result digests intentionally differ
between binaries because both digest scopes bind `native_library_sha256`.

The accepted M3.0 DLL remains at its original path and hash. The optimized DLL
was built into a separate target:

| Native artifact | SHA-256 |
| --- | --- |
| `target/m30_build/release/ofc_hu_m3_engine.dll` | `03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69` |
| `target/m31_opt_build/release/ofc_hu_m3_engine.dll` | `d2d8f66ae56d978297a9228ca8b271622bb1b73cd2437237de8ef89915ea73e4` |

At `4/8/2`, the frozen two-root batch improved from 135.6315 seconds to
31.4883 seconds. The final `2 processes x 8 Rayon threads` probe projected the
complete workflow at 3,431.455 seconds (57.19 minutes), so the 100-root command
was allowed to start.

## Accepted 100-root gate

Frozen contract:

- 100 roots / 50 generated hands;
- root seed `2026073201`;
- seed stride `1000003`;
- run ID `hu-m31-step3-local100-v1`;
- candidate/evaluation/downstream-T3/downstream-T4 budget `4/8/2/0`;
- 2 worker processes, each with 8 Rayon threads;
- 50 primary-hand tasks, 10 deterministic-rerun tasks, and 10 permutation
  tasks;
- atomic write-once task artifacts and a five-second parent heartbeat.

Results:

| Gate | Result |
| --- | --- |
| Unique live-shape roots | 100 / 100 |
| Seat balance | 50 first / 50 second |
| Forbidden hidden-truth fields | 0 |
| Scalar/batch semantic and mapping parity | 100 / 100 |
| Deterministic rerun, both digest scopes | 20 / 20 |
| Six dealt-card permutations | 10 / 10 roots; one semantic result each |
| Local ordered mapping under permutations | 60 / 60 valid |
| Candidate RNG keys | 400 unique |
| Evaluation RNG keys | 800 unique |
| Candidate/evaluation RNG overlap | 0 |
| Exact T4 children | all decisions |
| Finite values and complete legal mappings | all decisions |
| Write-once checkpoint tasks | 70 / 70 |
| Peak RSS per process | 83,664,896 bytes |
| End-to-end wall time | 3,498.657 seconds |

Latency is strongly seat-dependent and remains a profiling concern for the next
stage:

| Seat | mean | p50 | p95 | p99 / max |
| --- | ---: | ---: | ---: | ---: |
| first | 43.658 s | 28.857 s | 139.641 s | 158.190 s |
| second | 1.654 s | 1.300 s | 3.569 s | 3.678 s |

These are search latencies, not EV or loss-tail measurements.

## Validation

- New Step 3 and combined M3/M3.0/M3.1 focused Python tests:
  `104 passed in 84.37s`.
- Full Python suite: `2444 passed, 2 skipped, 50 warnings in 864.95s`.
- Rust release tests: 36 library tests and 2 runner tests passed.
- Rust format check passed.
- Rust release clippy passed with warnings denied.
- `src/ofc_regular/ai_profiles.py` remains SHA-256
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

The attempted `tests ai/tests` command was not a test failure: this nested
checkout has no `ai/tests` directory, so it collected nothing. The accepted
full-suite result above uses repository-root pytest discovery, matching the
earlier M3.1 audits.

## Source and artifact anchors

| Path | SHA-256 |
| --- | --- |
| `src/ofc_regular/hu_m31_t3_runtime.py` | `d9230a82b8350dde3937f63988a90c96e3cb1b6fe6f0b83858f550897276998a` |
| `src/ofc_regular/validate_hu_m31_t3_profile.py` | `58999bcb3b24bfe46d1f239e335fccbc7612bd41be41ab37568004a41f2eaa14` |
| `src/ofc_regular/validate_hu_m31_t3_convergence.py` | `e894040e2f57b621f47c35487ce0176babcfb8a53dd6b1a60b5e6b357b11a1ae` |
| `src/ofc_regular/profile_hu_m31_t3_parallelism.py` | `65ce9b0262df3952505dad1f7bb4ea70462e8938fc6afdc9d496720022539efb` |
| `src/ofc_regular/validate_hu_m31_t3_local100.py` | `630c2569c553429767c9e074236752290a663847b4a76d60727828c56c9475c4` |
| `rust/hu_m3_engine/src/search.rs` | `8a89cd02459f142153381af0bec7877096bb7e5aceac243b98644fefdebe2f56` |
| `outputs/hu_joint_policy/m31_t3_step3/local10_profile_rayon.json` | `e4caa2130c844b90289feacb4d58ff3ee1ea013dc9905554cb3ea40ea6399e43` |
| `outputs/hu_joint_policy/m31_t3_step3/convergence_2root_rayon_v2.json` | `9b0883d723ac15846ee13b1f7feb22386a585970bdf6d8b0c06a89f198d4beb1` |
| `outputs/hu_joint_policy/m31_t3_step3/parallel2x8_profile_v3.json` | `fb995ee4eb944d4ac9dcb4c61bf306f8e1efe894e7910a1814acb85fe7f35ffa` |
| `outputs/hu_joint_policy/m31_t3_step3/local100_v1/summary.json` | `c59fb28e3d5f407715e1688376537da44741c4c1221de12cbb88c5ccc3ac6f68` |

The rejected artifacts are retained as evidence rather than overwritten:

- `parallel8_profile.json`: original sequential engine, 111.07-minute No-Go;
- `parallel2x8_profile.json`: speed passed but exposed the incorrect PID-count
  validator;
- `parallel2x8_profile_v2.json`: corrected gate before engine provenance was
  added.

## Remaining boundary and next step

Step 3 does not create or activate a T3 model. The following remain incomplete:

- local 1,000-root profiling pilot;
- versioned T3 teacher row generation;
- new T3 reference and selective candidate training;
- population and multi-opponent paired seat-swap holdout;
- realized gain, false-positive override, p95/p99 loss-tail, and exploitability
  proxy evaluation;
- any named profile or `current` activation;
- any Spot VM expansion.

The next authorized action is the local 1,000-root profiling pilot using the
same hidden-discard-safe observation and optimized exact-T4 engine. Spot shards
remain No-Go until that local gate passes. M3.1 completion and policy promotion
require the later training and locked strength gates; the 100-root artifact by
itself is not promotion evidence.
