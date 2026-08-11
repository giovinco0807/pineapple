# HU joint-policy M3.1 T3 Step 6a completion audit

Date: 2026-07-17

Decision: **Step 6a is complete. Immutable Linux Spot shard 0 passed parity,
restart recovery, integrity, latency, and memory gates. M3.1 model building,
the remaining nine infrastructure shards, production teacher generation,
training, strength evaluation, and runtime activation remain No-Go.**

The accepted run is `regular-hu-m31-step6a-s0-20260717-004`. It evaluated 25
paired hands, yielding 50 balanced T3 seat roots. The VM used two worker
processes with eight Rayon threads each on a Spot `c4-standard-16` in
`asia-northeast1-b`. `DONE.json` was uploaded last, all 61 result objects were
received and checked against its file manifest, and the VM self-deleted.

`current` was neither resolved nor changed. No profile was added or activated.
Canary rows are diagnostic infrastructure evidence and are explicitly
ineligible for training.

## What was implemented

- `hu_m31_t3_behavior_roots.py` generates actor-visible T3 roots with equal
  quotas over `stage19_p0`, `stage9f_p2`, `stage7_m5_r10`,
  `stage3_baseline`, and `random_exact_final`.
- `run_hu_m31_t3_step6a_shard.py` freezes shard-0 seeds, materializes atomic
  paired-hand checkpoints, validates existing checkpoints on restart, and
  produces both-seat search results with candidate/evaluation RNG separation.
- Linux process creation is fixed to `spawn`. This avoids inheriting an
  initialized Rust/Rayon pool through `fork` after the parity probe.
- `hu_m31_t3_step6a_spot.py` creates and validates an immutable package,
  records the local Linux receipt, writes a shard-0-only authorization,
  publishes objects without overwrite, launches Spot, reports status, and
  verifies every received object hash.
- `startup_hu_m31_t3_step6a.sh` pins the Debian snapshot, checks all package
  hashes, uploads a 60-second heartbeat, deliberately stops after one task,
  wipes local result state, recovers from GCS, resumes the remaining work,
  commits `DONE` last, and self-deletes the VM.
- Both the T3 native engine and the Rust Stage3 feature encoder are bundled and
  bound by SHA-256.

## Accepted shard-0 evidence

| Gate | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Paired hands / T3 roots | 25 / 50 | exactly 25 / 50 | pass |
| First-seat roots | 25 | exactly 25 | pass |
| Second-seat roots | 25 | exactly 25 | pass |
| Recovered task count | 1 | at least 1 | pass |
| Unique fingerprints | 50 / 50 | 50 / 50 | pass |
| Candidate RNG keys | 200 unique | 200 | pass |
| Evaluation RNG keys | 400 unique | 400 | pass |
| Candidate/evaluation overlap | 0 | 0 | pass |
| First p95 latency | 45.2134 s | at most 180 s | pass |
| Second p95 latency | 1.2317 s | at most 6 s | pass |
| Peak process RSS | 508,878,848 bytes | at most 1 GiB | pass |
| Full resumed wall time | 340.17 s | diagnostic | pass |

The corrected latency summaries are monotone:

- first: p50 `24.3584`, p95 `45.2134`, p99/max `45.8900` seconds;
- second: p50 `0.9461`, p95 `1.2317`, p99/max `1.2420` seconds.

All five behavior profiles contributed exactly five paired hands. All 50
observation fingerprints were unique. Neither opponent private discards nor
realized remaining-deck truth is present in the root schema.

## Failure and repair audit

Failed or rejected runs were preserved rather than overwritten:

1. `...001` failed closed before any search task because the Linux Stage3
   feature encoder was absent from the package. The encoder was added with a
   frozen hash.
2. `...002` was not launched. Its local Linux smoke exposed the
   fork-after-Rayon deadlock; the executor now uses `spawn` on every OS.
3. `...003` completed search and hash-checked receipt, but its aggregate was
   rejected because percentile inputs were not sorted. Although even its
   maxima were inside the frozen limits, the reporting artifact was not used
   as completion evidence.
4. `...004` includes the sorted-percentile repair and passed locally and on
   Spot. This is the only accepted run.

This sequence is important: a successful process exit alone was not treated
as a correctness or promotion gate.

## Validation

- Step 6a unit tests: `12 passed in 0.58s`.
- Combined M3/M3.0/M3.1 focused tests: `126 passed in 30.86s`.
- Full Python suite: `2476 passed, 2 skipped, 50 warnings in 831.96s`.
- Black check passed for the Step 6a Python source and tests.
- Bash syntax validation passed in `bash:5.2`.
- Linux portable decision digests matched the frozen first/second goldens
  exactly.
- GCS receive validated the final marker plus all result file hashes.

The warnings are the pre-existing sklearn feature-name warnings in
`test_hu_m43_attempt05_training.py`; there were no test failures.

## Artifact anchors

| Artifact | SHA-256 |
| --- | --- |
| immutable source zip | `224506d27a15f237baaa5c200b4ef41bc2a8c61b817edac4b5a2f9c71d704be8` |
| package manifest | `a2d7cf23ed22fb4bbf3fee6fdecd6df807439117093e92783edb72d4fc665188` |
| launch authorization | `e3b6340885d2002a2b552a6f54ae730c7fc9b6c47f358ee22de74ac09ff02ffc` |
| native T3 library | `3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0` |
| Stage3 feature encoder | `82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411` |
| `DONE.json` | `45928bbd0c1938506e33c7e15fa7053f82436f1da9b445cd58701ff8f2bfe852` |
| `summary.json` | `4cdde3f061ef67354236e3297b7d40511550c7a00daf028bbc45b234c685717d` |
| receive receipt | `febe15eccf141d0fe1689fb8bdc5cdbfc7788459d5b8253943307bc41dd621ee` |

Local accepted artifacts are under
`outputs/hu_joint_policy/m31_t3_step6a/received_v4`. The lifecycle package and
authorization are under
`outputs/gcp_runs/regular-hu-m31-step6a-s0-20260717-004`.

## Remaining boundary and next step

Step 6a proves that the Linux search package, checkpoint/recovery path, and
bounded shard shape work. It does not create a T3 model and does not establish
realized match EV, low exploitability, Nash proximity, or mathematical
optimality. Teacher search values remain `diagnostic_not_match_EV`.

The next action is M3.1 Step 6b: explicitly authorize only the remaining nine
25-paired-hand infrastructure shards, merge and validate the full 500-root
canary, and stop again. Production fanout remains disabled. Only after that
boundary passes may the separately frozen 100-root `8/32/4/0` production-label
quality pilot be considered.
