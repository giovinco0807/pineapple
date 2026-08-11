# HU joint-policy M3.1 T3 Step 4 completion audit

Date: 2026-07-16

Decision: **Step 4 is complete locally as a 1,000-root correctness,
determinism, and performance boundary. M3.1 teacher artifact rebuilding,
strength evaluation, and policy promotion are not complete.**

The accepted artifact contains exactly 1,000 general-board T3 roots, balanced
500/500 by seat, at the frozen `4/8/2/0` search budget. It passed every frozen
integrity gate in 9,749.527 seconds (2.708 hours), below the 3.6-hour local
limit. No profile, `current` mapping, runtime policy, full replacement, Spot VM,
or cloud job was changed or started.

## Exactness and information boundary

- Root T3 remains common-random Monte Carlo over hidden-card particles; this is
  not an exact full-deck T3 solution.
- Candidate selection uses 4 particles and locked evaluation uses 8 disjoint
  particles.
- Each nested T3-second information set uses 2 particles.
- Every downstream T4 child uses the accepted exact native T4 semantics with
  `downstream_t4_samples = 0`.
- Inputs are actor observations: hero board, opponent public board, hero private
  discards, dealt cards, seat/order, turn, FL state, and scoring context. The
  opponent's private discards and realized hidden deck are not exposed.
- Teacher values remain diagnostic action estimates. They are not realized
  match EV, promotion evidence, or a proof of mathematical optimality.

## Local CPU allocation decision

The same 8-hand / 16-root workload and seeds were compared before the full run:

| Allocation | Wall time | Result |
| --- | ---: | --- |
| 4 worker processes x 4 Rayon threads | 157.575 s | pass |
| 2 worker processes x 8 Rayon threads | 137.518 s | pass |

The faster `2 x 8` allocation was frozen for local1000. The Step 3 measurements
projected 12,761.076 seconds (3.545 hours) with the preregistered 10% buffer,
which passed the 12,960-second start gate. Actual wall time was 9,749.527
seconds (2.708 hours).

## Restart and task-integrity contract

The runner writes one atomic, write-once JSON artifact per task and maintains a
parent heartbeat. The accepted run completed the exact expected set of 510
tasks: 500 primary two-seat hand tasks plus 10 deterministic rerun tasks. A
resume validates the frozen contract digest and existing task contents before
reusing them; corrupt or incompatible tasks fail closed.

Frozen contract:

- 1,000 roots / 500 generated hands;
- deterministic rerun of the first 20 roots / 10 hands;
- root seed `2026074201` and stride `1000003`;
- run ID `hu-m31-step4-local1000-v1`;
- candidate/evaluation/downstream-T3/downstream-T4 budget `4/8/2/0`;
- 2 worker processes, each with 8 Rayon threads;
- optimized native library SHA-256
  `d2d8f66ae56d978297a9228ca8b271622bb1b73cd2437237de8ef89915ea73e4`;
- maximum local wall time 12,960 seconds.

## Accepted 1,000-root gate

| Gate | Result |
| --- | --- |
| Unique observation fingerprints | 1,000 / 1,000 |
| Seat balance | 500 first / 500 second |
| Step 3 fingerprint overlap | 0 |
| Unique hand seeds | 500 / 500 |
| Step 3 hand-seed overlap | 0 |
| Candidate RNG keys | 4,000 unique |
| Evaluation RNG keys | 8,000 unique |
| Candidate/evaluation RNG overlap | 0 |
| Deterministic rerun, both digest scopes | 20 / 20 |
| Legal-action geometries | 3, 9, 12, and 21 all present |
| Atomic write-once task set | 510 / 510 |
| Peak process RSS | 57,225,216 bytes |
| End-to-end wall time | 9,749.527 seconds |

Legal-action geometry counts were 126 roots with 3 actions, 411 with 9, 252
with 12, and 211 with 21.

Search latency remains strongly seat-dependent:

| Seat | mean | p50 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| first | 36.858 s | 25.167 s | 129.333 s | 143.481 s | 153.766 s |
| second | 1.473 s | 1.173 s | 3.137 s | 3.541 s | 3.807 s |

These are teacher-search latencies, not EV or loss-tail measurements.

## Validation

- Step 4 and combined M3/M3.0/M3.1 focused Python tests:
  `113 passed in 27.17s`.
- Full Python suite:
  `2453 passed, 2 skipped, 50 warnings in 731.78s`.
- Rust release tests: 36 library tests and 2 runner tests passed.
- Rust format check passed.
- Rust release clippy passed with warnings denied.
- `src/ofc_regular/ai_profiles.py` remains SHA-256
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

## Source and artifact anchors

| Path | SHA-256 |
| --- | --- |
| `src/ofc_regular/validate_hu_m31_t3_local1000.py` | `36eba747c4fa0857e01a13ce3cecfb136ae390b454b28e24f8807f686870db5f` |
| `tests/test_hu_m31_t3_local1000.py` | `759a56e288cb4124885651f7b252bad518a1fa730ffb671692eff865d9338d29` |
| `outputs/hu_joint_policy/m31_t3_step4/parallel4x4_profile.json` | `c264cc584c4a0e2ee3926253a95bf5c797e052e49bb08032787863c5d5d1517c` |
| `outputs/hu_joint_policy/m31_t3_step4/parallel2x8_same8_profile.json` | `b8353cad2552eaa1b7bcb549c55ce2b1243d51fde6585b7e22bebf93371b6961` |
| `outputs/hu_joint_policy/m31_t3_step4/local1000_v1/smoke.json` | `ff422377da704e0033d778affb10b3cf1ce3c8ed1c06fb91f3ad3bf4ce0c3cb5` |
| `outputs/hu_joint_policy/m31_t3_step4/local1000_v1/summary.json` | `8fc6599192a9dcc1bc15d9368eccff8bbde0b5cb3c03ee34a2b8851d4421bb71` |
| `outputs/hu_joint_policy/m31_t3_step4/local1000_v1/heartbeat.json` | `4ca381d43662b1366e0a9a97dde5f574f478731d186297e65fe07bdea3a1d4cc` |

## Remaining boundary and next step

Step 4 proves that the current hidden-discard-safe T3 teacher path is
deterministic, integrity-checked, restartable, and locally scalable to 1,000
roots. It does not create or activate a T3 model and does not establish policy
strength.

The next action is Step 5: freeze numeric data-quality and promotion acceptance
gates plus non-overlapping development, calibration, and holdout seed ranges in
a versioned manifest. Only after that contract passes may small resumable Spot
teacher-data shards be authorized. No Spot job is authorized by this document
alone.

