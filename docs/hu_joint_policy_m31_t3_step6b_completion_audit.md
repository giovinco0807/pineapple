# HU joint-policy M3.1 T3 Step 6b completion audit

Date: 2026-07-17

Decision: **Step 6b is complete. The remaining nine immutable Linux Spot
shards and the merged 500-root infrastructure canary passed restart, integrity,
latency, and memory gates. M3.1 artifact rebuilding, production teacher
generation, training, strength evaluation, and runtime activation remain
No-Go.**

Step 6b used run `regular-hu-m31-step6b-canary-20260717-001` for shards 1-9
and retained accepted Step 6a run `regular-hu-m31-step6a-s0-20260717-004` as
immutable shard 0. Together they cover 250 paired hands and 500 balanced T3
seat roots. All nine Step 6b VMs completed, uploaded `DONE.json` last, and
self-deleted. There are no remaining Step 6b instances.

`current` was neither resolved nor changed. No profile was added or activated,
no model was trained, and production fanout was never authorized. These rows
are infrastructure diagnostics and are not eligible for training. Teacher
values remain `diagnostic_not_match_EV`.

## What was implemented

- `run_hu_m31_t3_step6b_shard.py` extends the accepted Step 6a task contract to
  shards 1-9, freezes each global hand range, validates every recovered row,
  action mapping, fingerprint, seed, and RNG key, and resumes only valid atomic
  paired-hand checkpoints.
- `hu_m31_t3_step6b_spot.py` creates a write-once package, binds the accepted
  Step 6a hashes, authorizes exactly shards 1-9, caps launches at three VMs per
  wave, restricts zones to `asia-northeast1-b/c`, and hash-checks every received
  object.
- `startup_hu_m31_t3_step6b.sh` pins the Debian snapshot and dependencies,
  validates every packaged source entry, runs Linux parity, deliberately stops
  after one task, wipes local task state, recovers from GCS, resumes, uploads
  `DONE.json` last, and self-deletes.
- `validate_hu_m31_t3_step6b_canary.py` treats accepted Step 6a shard 0 as a
  fixed trust anchor and verifies the exact 10-shard hand/root grid, both-seat
  balance, profile quotas, hidden-information boundary, seed namespaces, RNG
  disjointness, restart evidence, file hashes, and performance limits.

Before cloud launch, a real Linux local recovery smoke under
`outputs/hu_joint_policy/m31_t3_step6b/local_resume_smoke_v2` proved parity and
recovered one interrupted task. The first local invocation was preserved as a
failed command-construction attempt; PowerShell expanded a Bash variable before
the container started. It did not expose a search-engine defect and was not
used as acceptance evidence.

## Spot execution

The nine `c4-standard-16` Spot VMs were launched in bounded waves `(1,2,3)`,
`(4,5,6)`, and `(7,8,9)`, alternating between the two authorized zones. Every
shard completed exactly 25 paired hands / 50 roots and recovered exactly one
task from GCS. Each shard produced 61 progress objects and 61 result objects;
the receive lifecycle validated all hashes before accepting them.

| Shard | Elapsed | Recovered tasks | Result |
| ---: | ---: | ---: | --- |
| 1 | 6:52.09 | 1 | pass |
| 2 | 4:16.72 | 1 | pass |
| 3 | 5:26.46 | 1 | pass |
| 4 | 4:24.33 | 1 | pass |
| 5 | 6:52.14 | 1 | pass |
| 6 | 5:12.28 | 1 | pass |
| 7 | 6:11.66 | 1 | pass |
| 8 | 4:20.15 | 1 | pass |
| 9 | 5:01.62 | 1 | pass |

The package authorizes only shards 1-9. Its launch authorization records
`production_fanout_authorized=false` and `current_profile_changed=false`.

## Merged 500-root evidence

| Gate | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Shards / paired hands / roots | 10 / 250 / 500 | exact | pass |
| First / second roots | 250 / 250 | 250 / 250 | pass |
| Profile hands | 50 each across five profiles | exactly 50 each | pass |
| Profile roots | 100 each across five profiles | exactly 100 each | pass |
| Unique fingerprints | 500 / 500 | 500 / 500 | pass |
| Namespace seed values | 1,500 unique | 1,500 | pass |
| Candidate RNG keys | 2,000 unique | 2,000 | pass |
| Evaluation RNG keys | 4,000 unique | 4,000 | pass |
| Candidate/evaluation overlap | 0 | 0 | pass |
| Shards with recovered task | 10 / 10 | 10 / 10 | pass |
| First p95 latency | 53.9792 s | at most 180 s | pass |
| Second p95 latency | 1.3149 s | at most 6 s | pass |
| Peak process RSS | 720,109,568 bytes | at most 1 GiB | pass |

Latency tails remained inside the frozen infrastructure bounds:

- first: mean `25.8325`, p50 `22.7577`, p95 `53.9792`, p99 `61.6501`, max
  `64.0623` seconds;
- second: mean `0.7574`, p50 `0.7754`, p95 `1.3149`, p99 `1.4757`, max
  `1.5298` seconds.

The merged validator also proved the exact hand grid `0..249`, root grid
`0..499`, shard set `0..9`, and zero use of opponent private discards.

## Validation

- Step 6b-related tests: `47 passed in 36.56s`.
- Combined M3/M3.0/M3.1 focused tests: `173 passed in 68.27s`.
- Full Python suite: `2523 passed, 2 skipped, 50 warnings in 816.03s`.
- Black passed for all Step 6b Python source and tests.
- Bash syntax passed under `bash:5.2`.
- The local Linux parity and restart smoke passed.
- `git diff --check` passed.

The 50 warnings are the existing sklearn feature-name warnings in
`test_hu_m43_attempt05_training.py`; there were no test failures.

## Artifact anchors

| Artifact | SHA-256 |
| --- | --- |
| Step 6b immutable source zip | `4dc166990ff2c0015870aabcadf428151c02e13fcfe6d6515ba90352d57b4645` |
| Step 6b package manifest | `e24811edf22605f3b7acb7fcbc3507a2da157de8cefe8dffd9ae8e4755f0c855` |
| Step 6b shard schedule | `13923bb934e651a277673ac174f94079f7792df2433018615d3ed9854ff815e1` |
| Step 6b launch authorization | `def2fa90b34f0699133ce6c3945946811c6d8b6d7bd8af37b989614c019ad309` |
| Step 6b receive receipt | `63b289f3f436662ac56c74b1b16b58054a8e95e44362e9077179f6c2e5cf70cf` |
| Merged canary validation | `f95445cc71b402d97eb120cde2cfdbd1235ee57ca7c83b9227bcf5f9c3a238da` |
| Native T3 library | `3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0` |
| Stage3 feature encoder | `82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411` |
| Accepted Step 6a `DONE.json` | `45928bbd0c1938506e33c7e15fa7053f82436f1da9b445cd58701ff8f2bfe852` |

Received shards are under
`outputs/hu_joint_policy/m31_t3_step6b/received_v1`. The merged result is
`outputs/hu_joint_policy/m31_t3_step6b/canary500_validation_v1.json`. The
immutable package and authorization are under
`outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001`.

## Known contract gap

The Step 5 contract names its behavior schedule
`equal_quota_seeded_shuffle_v1`, while Step 6a/6b implemented the deterministic
five-profile cycle `global_hand_index % 5`. This does not invalidate the
infrastructure canary: the merged grid has exactly 50 hands and 100 roots per
profile, and shard/restart determinism is stronger with the fixed cycle. It is
nevertheless a versioned-contract mismatch. The schedule must be explicitly
resolved and frozen before any production-label pilot is authorized; the
canary implementation must not silently define the production distribution.

## Remaining boundary and next step

Step 6b proves that the portable Linux package, nine-shard Spot lifecycle,
restart path, merged integrity validator, and bounded runtime shape work. It
does not build a T3 artifact or establish action quality, realized match EV,
low exploitability, Nash proximity, or mathematical optimality.

The next action is Step 6c, not production fanout: freeze and explicitly
authorize only a 100-root `8/32/4/0` production-label quality pilot, use
independent candidate/evaluation RNG, and re-evaluate a frozen 10% subset with
128 samples. Confirmation regret and integrity gates must pass before any
training or larger generation is considered. Step 6c was not authorized or
started as part of this completion.
