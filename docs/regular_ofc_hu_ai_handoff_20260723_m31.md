# Regular OFC HU AI M3.1 handoff

Updated: 2026-07-23 JST

Repository:

`C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\regular-ofc-pineapple`

Branch:

`codex/regular-ofc-pineapple`

## 1. Non-negotiable constraints

- Do not delete, reset, clean, stage, or overwrite the existing dirty worktree.
- `git status --short` currently has about 1,194 entries. Many M3.1 files are
  untracked but are active research state, not disposable build output.
- Do not change `current`.
- Do not replace or remove the named P0/P1/P2/T3 baselines.
- Do not activate a full replacement before locked practical promotion.
- Do not expose opponent private discards or realized deck tail to a model.
- Do not use teacher EV LCB directly as a runtime gate.
- Do not call a policy mathematically optimal unless a proof exists.
- Performance, quality, data, threshold-lock, and diagnostic holdout seeds must
  remain disjoint.
- Candidate-selection randomness and evaluation randomness must remain
  independent.
- A non-fired override must return the exact baseline `Action` object and must
  not consume policy RNG or change the trajectory.

Pinned profile file:

`src/ofc_regular/ai_profiles.py`

Pinned SHA-256:

`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`

The `current` profile still resolves to the legacy stage9d policy. The stronger
fixed chain is selected explicitly:

`stage19_p0 -> stage18_p1 -> stage9f_p2 -> stage7_m5_r10 -> T4 exact`

## 2. Meaning of the final goal

M3.1 is complete only when T3 has:

1. passed an independent one-shot performance lock;
2. passed fresh quality and high-precision confirmation;
3. produced the frozen 9,000-paired label set;
4. trained and calibrated `StreetPolicyNetV1`;
5. preserved exact non-fire cancellation;
6. passed five-opponent and at least three-ABR-family promotion;
7. been added only as an explicit opt-in profile.

The realistic claim is a strong practical self-play policy with empirical
low-exploitability evidence. It is not a proof of Nash equilibrium and not a
mathematically complete solution.

## 3. Completed foundations

### T4

- T4 uses the exact runtime solver.
- It evaluates every legal final placement using the actor's valid information.
- Opponent private discards are not required or exposed.

### T3 engine and contract

- Step6d runner, independent validator, deterministic seed namespaces, resume,
  tamper detection, ActionKey checks, and candidate/reference process isolation
  exist.
- Candidate01 matched the reference bit-for-bit and was about 1.44x faster.
- Candidate02/compact-scoring work is the active performance candidate.
- Existing baseline `stage7_m5_r10` remains intact.

### Accepted performance-development run

Run root:

`D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009`

Run name:

`regular-hu-m31-c02-f100wv2-20260723-009`

Execution identity:

`dd88e7361639f3ada186b0cfa4661681d4c715465dc9b9b7562523221bc902ba`

Scientific receipt:

`D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009\scientific-gate-receipt.json`

Receipt file SHA-256:

`664f86262436d41b62a8324cf1ba8df52e0f6af09f80f2071af825862b39e6b2`

Observed result:

- status: `pass`
- decision: `full100_wave_v2_go_open_one_shot_performance_lock_only`
- exactly 100 paired hands / 200 roots
- exactly 20 jobs / 440 accepted objects
- portable semantic parity: 1.0
- first-seat p95: 88.183210156 s
- first-seat p99: 90.621403529 s
- first-seat max: 91.630463564 s
- second-seat p95: 1.059373219 s
- peak RSS: 123,621,376 bytes
- candidate/reference first-seat geometric-mean speedup:
  2.77465048668492x
- missing/censored records: zero

This run authorizes only the new one-shot performance lock. It is not the
performance lock itself and does not authorize training or quality acceptance.

## 4. Frozen performance-lock v4 input

Root:

`outputs/hu_joint_policy/m31_t3_step6d/performance_lock_v4`

Frozen files:

| File | SHA-256 |
|---|---|
| `performance_lock_v4_plan.json` | `2ad08116835a58f5b5927e4de986f2717915d0dd128e7a5f3fa288e0cac6e5be` |
| `execution/MATERIALIZATION_CLAIM.json` | `cfceeb9d4bd89807e43ef7b8899b0ed04bf70fdef987648b0c57c9c743984d3f` |
| `execution/MATERIALIZATION_RECEIPT.json` | `be9554ef78e9234b994a3e1e537c9e3a2c1618c61205f8c8a90578380e86666e` |
| `execution/ROOT_SEAL.json` | `f38cc30eba09ef5e3a6380a41b79e990ca8394943f4415fbc8f4eba463bff401` |
| `scientific_source_package/manifest.json` | `d01933aaf861ab714d13a21d8961ba04ed749631af24894d71b428820f099ddd` |
| `scientific_source_package/PACKAGE_READY.json` | `3322d1e70e27b6f143e4f2be0f9b37649d1e961b0544669c1d24e52d1ded8d10` |
| source ZIP | `acfd573fd4ea033bf3e154b37b325a7a61fc49188a77b522a6494a0a4c8dacf9` |

The package above cannot execute. Its source ZIP omits
`configs/hu_joint_policy_m31_t3_step6d_contract.json`, which
`run_hu_m31_t3_step6d_performance_v2.run_source_shard` reads from the extracted
science root, so every worker died with `FileNotFoundError` right after the
wheelhouse install. The packager now carries that contract
(`hu_m31_t3_step6d_performance_lock_v4_spot_package.CONTRACT_RELATIVE_PATH`).

Use this replacement package instead. Root:
`D:\ofc-gcp-runs\v4-science-package-contractfix`

| File | SHA-256 |
|---|---|
| `manifest.json` | `618f3aa95209ac7d86cb3fa5af941d17b9f6dda246985f383d02de0016a889cc` |
| `PACKAGE_READY.json` | `d433dc6ea3dba65f8ca01e951f55180efd5021b53be1b22a3c77eae8282c08f4` |
| source ZIP | `b850f7b6e0832239916790e860b460663dcc7ef544d232e6132a097395e7224e` |

The replacement ZIP holds 643 members against the original 597: the contract
plus 45 untracked `src/ofc_regular/*.py` modules that appeared in the worktree
after the original package was built. The packager globs the whole package
directory, and the dirty worktree must not be cleaned, so they are carried
along. They do not affect execution — the startup script validates every
archive member against the manifest it was built from, and the runner imports
only what it needs — but a payload rebuilt from a clean tree would be smaller.

Other frozen identities:

- run contract digest:
  `669c1efa1afeebe41fcc531c6458c9d72fffdd5df2cca751c99988a872f3e2b6`
- seed-set SHA-256:
  `f63b2f0cb9212e9f16d9a05c79cdd6946fec54daccfd4a212f0db08c498a9847`
- v4 startup SHA-256:
  `80c489030a67536f3584d062e16b265486659be2fd1a38b2bfe03b4a4ec0d68b`
- old development startup SHA-256, which must remain unchanged:
  `204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87`

The v4 package contains 100 sealed roots and 20 job descriptors. Candidate and
reference are separate jobs/VMs. The intended wave layout is 8, 8, and 4 VMs,
not 16 simultaneous VMs.

## 5. Implemented v4 production code

Important modules:

- `src/ofc_regular/hu_m31_t3_step6d_candidate02_performance_lock_v4_plan.py`
- `src/ofc_regular/hu_m31_t3_step6d_performance_lock_v4_spot_package.py`
- `src/ofc_regular/hu_m31_t3_step6d_performance_lock_v4_gate.py`
- `src/ofc_regular/merge_hu_m31_t3_step6d_candidate02_performance_lock_v4.py`
- `src/ofc_regular/hu_m31_t3_step6d_performance_lock_v4_production_bridge.py`
- `src/ofc_regular/hu_m31_t3_step6d_full100_wave_science_registry_v2.py`
- `src/ofc_regular/hu_m31_t3_step6d_full100_wave_plan_v2.py`
- `src/ofc_regular/hu_m31_t3_step6d_full100_wave_package_v2.py`
- `src/ofc_regular/hu_m31_t3_step6d_full100_wave_run_prepare_v2.py`
- `src/ofc_regular/hu_m31_t3_step6d_full100_wave_production_orchestrator_v2.py`
- `scripts/startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh`

The science registry now selects startup path, startup hash, execution scope,
and package validator from the validated plan descriptor. Explicit caller
values must match the descriptor or fail closed. The development descriptor
and startup remain supported without changing their bytes.

The v4 final production bridge validates:

- full pure-source replay;
- plan, materialization receipt, and root seal;
- root seal -> archive -> outer package -> prelaunch chronology;
- exactly 20 accepted jobs and 440 accepted objects;
- only attempt `a00`, with zero retry/reseed/failure;
- ten candidate/reference VM and process namespace pairs;
- exact merge-view/pure-merge source equality;
- pinned `current` profile hash;
- final qualified/no-go flags;
- write-once and tamper rejection.

The bridge currently exposes Python APIs but has no dedicated production CLI:

- `build_performance_lock_v4_production_receipt`
- `write_performance_lock_v4_production_receipt`
- `validate_performance_lock_v4_production_receipt`
- `validate_performance_lock_v4_production_receipt_value`

Do not use the old development scientific gate as a substitute for this v4
bridge.

## 6. Test state

Focused suites that passed during the current implementation include:

- pure v4 gate/merger: 39 passed;
- v4 plan/package/gate/merger combined: 51 passed;
- v4 source package: 5 passed;
- runner/recovery boundary: 37 passed;
- science registry plus plan: 12 passed;
- outer package including actual v4 source replay: 25 passed;
- v4 startup including 100-root/20-job replay: 3 passed;
- v4 production bridge: 8 passed;
- related bridge regression: 53 passed;
- existing run-prepare suite: 9 passed.

These groups overlap and must not be summed.

The complete repository suite was 779 passed before the later v4 dispatch
changes. A final broad startup/transport regression was interrupted after the
actual v4 Phase-A test process exited and before all controller/receiver suites
were collected. Treat that final broad regression as unverified and rerun it.

No pytest process is currently running.

Recommended focused rerun:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_hu_m31_t3_step6d_full100_wave_science_registry_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_plan_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_package_v2.py `
  tests/test_startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_run_prepare_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_launch_bundle_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_controller_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_gce_adapter_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_result_receiver_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_production_receiver_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_production_cleanup_orchestrator_v2.py `
  tests/test_hu_m31_t3_step6d_full100_wave_production_scientific_bridge_v2.py `
  tests/test_hu_m31_t3_step6d_performance_lock_v4_gate.py `
  tests/test_merge_hu_m31_t3_step6d_candidate02_performance_lock_v4.py `
  tests/test_hu_m31_t3_step6d_performance_lock_v4_production_bridge.py `
  --disable-warnings --maxfail=1
```

Afterward, recheck the pinned profile hash.

## 7. Current cloud state

Project:

`ofc-solver-485418`

Region/zone:

`asia-northeast1` / `asia-northeast1-b`

Bucket:

`gs://pokerhu-ofc-solver-485418-training`

Live readback on 2026-07-23:

- VM count: 0
- disk count: 0
- C4-family regional quota: 128 vCPU
- `c4-standard-16` capacity per VM: 16 vCPU
- maximum accepted simultaneous layout: 8 VMs / 128 vCPU

Eight worker service-account objects named `ofc-f100-worker-00` through
`ofc-f100-worker-07` still exist. The last audit found no target project or
bucket bindings. Do not delete or grant them outside the production lifecycle.

The one-shot performance lock has since been collected twice. Live readback on
2026-07-26 is again VM count 0 and disk count 0.

| run | root | state |
|---|---|---|
| `...-plock-20260723-004` | `D:\ofc-gcp-runs\...-004` (Windows) | 20/20 accepted; lock **qualified**; cannot authorize quality — see section 13 |
| `...-plock-20260726-005` | `~/ofc-runs/...-005` (WSL ext4) | 20/20 accepted, zero preemptions; lock stages pending |

**Run -005 is the operative run.** Run -004 proved the science and produced a
qualified final receipt, but every local path in it is a Windows string, so the
portable receipt cannot replay from Linux and fresh quality cannot consume it.
Run -005 repeats the same collection entirely on Linux so that the same receipt
is replayable where the frozen `.so` files load. The scientific content of the
two runs is identical — same frozen plan, same seeds, same binaries.

## 8. Reusable immutable inputs

Wheelhouse:

`D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009\phase-a\outer_package\content\wheelhouse\wheelhouse.zip`

Wheelhouse SHA-256:

`8704107c8e63f2947128f7f5a0c3ba3b45c0aace3636115d0720b77a74b40405`

Wheelhouse manifest:

`D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009\phase-a\outer_package\content\wheelhouse\wheelhouse_manifest.json`

Wheelhouse manifest SHA-256:

`f8c1b3fc02d075ff44e6823c3121cf379360cef296def89ec07365cceeb6fb3d`

Accepted image digest:

`sha256:9dd85299f559ea3b143b1a764a9c69e0e535672036c2b45bf1cff25b88da3c0d`

Heavy output must go to `D:`. `C:` had only about 2.3 GiB free during the last
check.

## 9. Immediate next execution

Only after the focused regression above is green:

```powershell
$repo = "C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\regular-ofc-pineapple"
$runName = "regular-hu-m31-c02-performance-lock-v4-20260723-001"
$runRoot = "D:\ofc-gcp-runs\$runName"
$env:PYTHONPATH = "$repo\src"

$saltBytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($saltBytes)
$rng.Dispose()
$env:OFC_FULL100_IDENTITY_SALT = [Convert]::ToBase64String($saltBytes)

python "$repo\scripts\prepare_hu_m31_t3_step6d_full100_wave_run_v2.py" phase-a `
  --output-dir "$runRoot\phase-a" `
  --run-name $runName `
  --scientific-package-dir "$repo\outputs\hu_joint_policy\m31_t3_step6d\performance_lock_v4\scientific_source_package" `
  --startup-script "$repo\scripts\startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh" `
  --expected-startup-sha256 "80c489030a67536f3584d062e16b265486659be2fd1a38b2bfe03b4a4ec0d68b" `
  --wheelhouse-archive "D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009\phase-a\outer_package\content\wheelhouse\wheelhouse.zip" `
  --wheelhouse-manifest "D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009\phase-a\outer_package\content\wheelhouse\wheelhouse_manifest.json" `
  --image-digest "sha256:9dd85299f559ea3b143b1a764a9c69e0e535672036c2b45bf1cff25b88da3c0d" `
  --full100-plan "$repo\outputs\hu_joint_policy\m31_t3_step6d\performance_lock_v4\performance_lock_v4_plan.json"

Remove-Item Env:OFC_FULL100_IDENTITY_SALT

python "$repo\scripts\prepare_hu_m31_t3_step6d_full100_wave_run_v2.py" collect-absence `
  --phase-a-dir "$runRoot\phase-a" `
  --output "$runRoot\all_owned_absence_receipt.json" `
  --project "ofc-solver-485418" `
  --zone "asia-northeast1-b"

python "$repo\scripts\prepare_hu_m31_t3_step6d_full100_wave_run_v2.py" phase-b `
  --phase-a-dir "$runRoot\phase-a" `
  --absence-receipt "$runRoot\all_owned_absence_receipt.json" `
  --output-dir "$runRoot\phase-b" `
  --expected-startup-sha256 "80c489030a67536f3584d062e16b265486659be2fd1a38b2bfe03b4a4ec0d68b"
```

Never print or persist the identity salt. Phase-A is write-once; if the command
partially fails, inspect its receipt and resume contract rather than deleting
the directory.

Before any cloud mutation, run the production orchestrator in `--mode plan`
using:

- `phase-a/wave_plan.json`
- `phase-b/attempt_ledger_v0.json`
- `phase-b/resume_plan_wave0.json`
- `phase-a/outer_package`
- a new control root on `D:`

Then execute exactly three waves: 8, 8, and 4 jobs. Receive and perform the
production cleanup lifecycle after each wave. Wave 1 and wave 2 must consume the
receiver-produced ledger/resume plan from the preceding wave.

Every wave must share one canonical set of roots, because the scientific bridge
reads the lifecycle from `<run-root>/{receiver,control,cleanup}/<namespace>` and
requires the `controller_journal_dir` sealed in each `receiver_request.json` to
equal `<run-root>/control/<namespace>/controller-journal` exactly:

- `--control-root <run-root>\control`
- `--cleanup-root <run-root>\cleanup`
- receiver `--output-dir <run-root>\receiver`
- receiver `--destination <run-root>\accepted` (the same path for every wave;
  the final receipt seals it, and the bridge's `--accepted-root` must match)

Per-wave root names such as `control-wave0` look harmless — each wave already
gets its own `execution-NNN-<hash>` namespace inside the root — but they leave
the bridge unable to consume the run. The journal path is sealed at launch time
inside `execution_manifest.json`, so re-running cleanup or receive afterwards
cannot repair it; only a fresh collection can. Run
`regular-hu-m31-c02-f100wv2-plock-20260723-003` reached 20/20 accepted jobs and
440 objects and still could not be merged for this reason.

From wave 1 onward, pass `--source-content-handoff`. The immutable content
prefix is content-addressed and shared across waves, so a second staging attempt
fails with `immutable content prefix is not empty`. Build the handoff from the
wave-0 evidence with
`production_orchestrator_v2.build_source_content_handoff(stage_plan=...,
content_preflight_receipt=..., source_stage_receipt=...)` using
`control/<wave-0-namespace>/static/content_stage_plan.json` and the
`content-prefix-preflight` / `stage-content` step events.

Spot preemption is normal: wave 1 of run `-003` lost 4 of 8 workers. The
receiver reports them in `failed_job_ids` and emits a resume plan selecting
`a01` for exactly those jobs. Re-launching that resume plan into the same
canonical control root is the designed path and gets its own namespace.

Do not hand-construct GCE or IAM commands. Use the existing production
orchestrator, receiver, and cleanup modules. Use a fresh OAuth access token only
in process memory. Verify zero owned VM/disk/IAM residue after every wave.

After 20 accepted jobs / 440 accepted objects:

1. build the accepted lifecycle merge view;
2. run the pure v4 merger;
3. issue the v4 final production receipt through
   `hu_m31_t3_step6d_performance_lock_v4_production_bridge.py`;
4. open quality only if every performance flag passes.

## 10. Work not yet implemented or executed

**This section is stale on two points.** The v4 one-shot Spot performance lock
is now complete — run `regular-hu-m31-c02-f100wv2-plock-20260723-004` issued
`performance_lock_v4_production_receipt.json` with
`status: qualified` and
`decision: performance_lock_v4_finalized_qualified_open_quality_pilot_only`,
which sets `quality_pilot_authorized: true`. And the "no implementation found"
claim below is wrong: `street_policy_net_v1.py` (1,277 lines),
`hu_m31_t3_abr_v1.py` (1,984), `hu_m31_t3_abr_teacher_v1.py` (1,861),
`hu_m31_t3_street_policy_training_v1.py` (1,858),
`hu_m31_t3_step6d_fresh_quality_v1.py` (1,531),
`hu_m31_t3_step6d_locked_promotion_v1.py` (1,643),
`hu_m31_t3_street_policy_runtime_v1.py` (1,348) and
`hu_m31_t3_opt_in_registration_v1.py` (1,645) all exist, with tests. They are
untracked, which is why a `git`-based search missed them.

So the remaining work is **execution, not implementation**. Nothing after the
lock has ever run: `outputs/` contains zero artifacts for street_policy,
fresh_quality, locked_promotion, abr, or opt_in.

That distinction matters, because every stage executed for the first time in
this milestone contained a blocking defect. See section 10b.

The following are still unfinished:

- fresh 50-paired quality plan/package/run/merge;
- separate 5-paired `8/128/4/0` confirmation;
- immutable 9,000-paired split and generator;
- 25-paired shard smoke for that generator;
- `StreetPolicyNetV1`;
- risk/safety calibration and locked thresholds;
- T3 runtime override integration;
- exact non-fire trajectory proof for the new model;
- five-opponent promotion run;
- three learned ABR families;
- final opt-in T3 profile and M3.1 closeout.

Search of the current source tree found no M3.1 implementation of
`StreetPolicyNetV1`, `hu_population_league_v1.py`, or `hu_abr_v1.py`.
`validate_hu_m31_t3_step6c_quality.py` is an older Step6c component and is not a
replacement for the new v4 fresh-quality contract.

## 10b. Defects found by executing the lock

Five blocking defects surfaced while completing the lock. All five were in code
that was implemented and unit-tested but had never been run. Expect the same
density in every remaining stage.

1. **Staged startup object name.** The packager hardcoded the development
   startup filename, so the v4 script was staged under the wrong name. The
   script re-derives its own object name and aborts, killing all eight workers
   in 77 ms with nothing in the serial console — line 67 redirects the rest of
   the run into `/var/log/ofc-full100-wave-v2/startup.log`. Fixed by resolving
   the path through `science_registry.startup_relative_paths_by_sha256()`.
2. **Missing performance contract in the science payload.** The v4 packager
   collected only `pyproject.toml` and `src/ofc_regular/**/*.py`, but
   `run_source_shard` reads `configs/hu_joint_policy_m31_t3_step6d_contract.json`
   from the extracted root. Workers died with `FileNotFoundError` after the
   wheelhouse install. Fixed with `CONTRACT_RELATIVE_PATH`; see section 4.
3. **Per-wave control roots.** A process error, not a code defect — see the
   canonical-layout rule in section 9.
4. **Scientific merge was development-only, five layers deep.** The bridge
   called `merge_candidate02_full100` directly; the registry had no merger
   binding; the v4 merger needs frozen materialization and root-seal evidence
   the merge view never carried; the downstream summary validator assumed the
   development shape; and receipt replay read the plan from the summary, which
   the v4 shape does not carry. Each fix exposed the next.
5. **Snapshot handoff gap between the two bridges.** The scientific bridge kept
   only `accepted_snapshot_sha256`; the production bridge needs the snapshot
   body, and it cannot be rebuilt from the digests — a reconstruction attempt
   produced a different digest and was discarded rather than used. The bridge
   now also stores `accepted_results_snapshot`.

Defect 5's first fix made that key mandatory, which broke
`load_run009_scientific_gate_receipt`: the frozen run009 receipt is pinned by
file hash and has no such key, so the exact-key check rejected it and the whole
v4 plan module stopped loading. The key is now optional, validated against the
adjacent digest when present. **Any schema change to an artifact that older
frozen receipts are also read through must stay backward compatible**, and the
focused regression is what caught it.

## 10c. Actual dependency chain

Section 2 lists seven completion criteria; they are a strict chain, not a menu.
In particular fresh quality cannot start before the lock is fully finalized:

```text
accepted lifecycle merge view      (scientific bridge, --mode merge)
  -> pure v4 merger                (merge_and_write_candidate02_performance_lock_v4)
  -> v4 final production receipt   (performance_lock_v4_production_bridge)
  -> fresh quality local staging   (--performance-receipt consumes that receipt)
```

`build_fresh_quality_plan` rejects the scientific gate receipt with
`performance receipt is not the exact portable pin`; only the production final
receipt authorizes it. Neither the pure merger nor the production bridge has a
CLI — both must be driven from their module APIs, and the production bridge
takes twelve inputs, of which eleven come from the gate receipt and the frozen
lineage.

## 11. Required post-lock gates

### Fresh quality

- 50 fresh paired hands / 100 roots;
- separate 5-paired `8/128/4/0` confirmation;
- confirmation regret mean/p95/p99/max at most 0.75/3/6/15;
- hidden truth, unknown field, ActionKey drift, RNG overlap, and missing rows:
  zero.

### Data split

| Role | Paired hands |
|---|---:|
| train | 6,000 |
| safety-fit | 1,000 |
| threshold-lock | 1,000 |
| diagnostic holdout | 1,000 |

Exactly 10% of each split receives preregistered high-precision confirmation.
Core weights use train only. Safety/risk fitting uses safety-fit only.
Threshold-lock cannot update weights. Diagnostic holdout cannot select a
threshold.

### Runtime gate

Override only when:

- candidate and baseline semantic ActionKeys differ;
- predicted delta minus downside p95 and ensemble disagreement is positive;
- seat-calibrated safe probability exceeds the frozen threshold;
- input schema, ActionKey mapping, and model hashes match exactly.

### Promotion

- five opponent policies and at least three ABR families;
- paired seat-swap overall and by seat;
- at least 300 valid overrides and at least 100 per seat;
- gain/override and EV/hand CI95 lower bounds above zero;
- false-positive rate at most 0.30 overall and 0.35 per seat;
- override-loss p95/p99/max at most 25/40/50;
- every opponent mean at least -0.005 and CI95 lower bound at least -0.02;
- non-fire trajectory mismatch zero;
- learned ABR worst response at least -0.01 point/hand.

## 11b. Measured override economics from turn 1

The promotion thresholds in section 11 are easier to read against the turn-1
results already in `outputs/evals/`, which used the same override design. Read
these before planning any T3 run size; they are the only measured prior.

Raw candidate/baseline disagreement, margin >= 1
(`hu_turn1_stage1_stage9f_p2_full2k/t1_decision_analysis_3000_margin1`):

| Quantity | Value |
|---|---|
| fired rows | 963 of 4,777 valid (`override_rate_on_valid` 0.202) |
| mean delta | **-2.18** points per firing |
| CI95 | [-2.97, -1.39] — significantly negative |
| median / min / max | 0.0 / -40.2 / +49.5 |
| implied sigma | ~12.5 points (`std_error` 0.4025 x sqrt(963)) |
| `non_fired_delta_max_abs` | 0.0 — exact non-fire confirmed |

Taking the candidate action on every disagreement *loses*. The safe-override
selector exists to find the positive subset, and it does
(`hu_turn1_stage11_selector_c2_500`, 500 paired seeds):

| Quantity | Value |
|---|---|
| realized overrides | 15 of 2,000 decisions (~0.75%) |
| mean delta per firing | **+4.08** points |
| `avg_score_per_hand_for_a` | +0.042 (stage12: +0.085) |
| `non_fired_final_mismatch_count` | 0 |

Two consequences for sizing a T3 run:

1. **Section 11's "at least 300 valid overrides" is conservative, not tight.**
   With paired seat-swap and exact non-fire, non-firing hands contribute zero
   variance, so the effective sample size is the firing count. CI95 above zero
   needs `n_fire > (1.96 * sigma / mu)^2`; at sigma 12.5 and mu +4.08 that is
   36 firings. 300 is roughly 8x the statistical minimum.
2. **The firing rate, not the threshold, drives cost.** At the turn-1 rate of
   ~1%, reaching 300 firings needs ~40,000 decisions, i.e. ~20,000 paired
   hands. Promotion hands are cheap — they are played by the runtime policy
   (second-seat p95 ~1.1 s), not by the 92 s teacher — but the count is large,
   and section 11 does not say whether 300 is required per opponent, per ABR
   family, or overall.

The T3 firing rate is unmeasured. Its upper bound is the raw disagreement rate,
which needs no trained selector — only the teacher and the baseline — and is
therefore worth measuring before committing to the 9,000-paired label run.

## 11c. T3 measurement, and what it does not establish

The turn-1 numbers in 11b can be reproduced for T3 locally, free, in minutes,
with `ofc_regular.trace_hu_turn3_overrides`. Two candidate models over 2,000
paired hands at `--hu-turn3-min-margin 1`:

| model | fire% | zero-delta% | take-all/hand | oracle/hand | sigma |
|---|---:|---:|---:|---:|---:|
| `hu_turn3_stage2_mc32_500k` (7 MB) | 34.2% | 70.7% | +0.355 | +1.489 | 8.78 |
| `hu_turn3_joint_exact_stage8_mc128` (79 MB) | 15.8% | 53.6% | -0.349 | +0.893 | 12.18 |

A margin sweep (0/1/5/10/20, 1,000 hands each) puts both at their own operating
point. At margin 0 — every disagreement, the fair comparison — the two oracles
are +1.672 and +1.707, indistinguishable within sampling error.

**Do not conclude from this that teacher labels are worthless.** The measurement
is against `trace_hu_turn3_overrides`' default base models, not the production
chain, and the direction of that bias is not neutral:

> The oracle ceiling is the *baseline's* stock of mistakes. Against a weak
> baseline almost any candidate finds the easy wins, so candidates look
> interchangeable. Against a strong baseline only a sharper candidate finds
> what is left. This measurement sits in exactly the regime where candidate
> quality is least distinguishable, so it cannot rule out that a better teacher
> matters against `stage9f_p2`.

What **is** established, because 11b is measured against `stage9f_p2` itself:
large headroom exists, and selection is the binding constraint — the turn-1
selector captures about 6.5% of its oracle.

The T3 version of that measurement needs a `stage9f_p2` + T3-override composite
profile. None exists (turn 1 has `stage9f_p2_hu_t1_topk_confirm`; T3 has no
counterpart), and creating one edits the pinned `ai_profiles.py`, which section
2 makes the *last* step of M3.1. So this measurement only becomes available
after T3 runtime override integration lands.

## 11d. Revised strategy

| Direction | Status |
|---|---|
| Prioritise the discriminator over candidate quality | **Supported** — 6.5% capture against the production baseline |
| Shrink the 9,000-paired teacher label run | **On hold** — the evidence for this was the weak-baseline comparison above and does not carry |
| Defer statistical proof from per-street gates to one joint promotion | **Supported in form** — cost scales as `(sigma/mu)^2`, so proving one combined effect is far cheaper than five small ones; cross-street additivity is untested |
| Measure locally before spending in cloud | **Supported and demonstrated** — every number in 11b/11c cost nothing |

The label-run decision should be made **after** T3 runtime integration makes the
production-baseline measurement possible, not before. Cutting it now would be a
bet without evidence; committing the full `$330-500` now would be a bet against
the one production-regime datapoint we have. Sequencing the integration ahead of
the label run resolves this at no extra cost, because the integration is
required for M3.1 regardless.

## 12. Expected remaining time

If every frozen gate passes on its first attempt:

- realistic: 6-9 calendar days;
- best case with overlap: 4-5 days;
- one redesign cycle: 10-14 days.

The largest wall-clock item is the 9,000-paired teacher-label run. With the
current accepted C4 layout, no more than eight `c4-standard-16` workers may run
simultaneously.

That item has now been costed against measurement rather than estimate. Run
`regular-hu-m31-c02-f100wv2-plock-20260723-003` collected 100 paired hands for
about **18.6 c4-standard-16 VM-hours** end to end, including one Spot
preemption and its `a01` resume. Scaling linearly:

| Quantity | 100 paired hands (measured) | 9,000 paired hands (extrapolated) |
|---|---|---|
| VM-hours | 18.6 | ~1,670 |
| Wall clock at 8 workers | ~2.5 h | **~8.7 days** |
| Spot cost at $0.20-0.30/VM-h | ~$5 | **~$330-500** |

So the label run alone plausibly consumes the entire USD 500 authorisation and
the whole "realistic 6-9 calendar days" budget. Two things follow:

- The wall-clock half is fixable for free. The eight-worker ceiling is the C4
  quota (128 vCPU), not a contract; raising it to 512 vCPU gives 32 workers and
  cuts ~8.7 days to ~2.2 days. Quota increases cost nothing and one has already
  been granted for this project (24 -> 128).
- The money half is not fixable by parallelism, because VM-hours are unchanged.
  It needs a smaller label set, a cheaper teacher configuration, or a new
  budget decision. Section 1's rule against expanding spend without a fresh
  frozen pilot applies here.

Section 11b explains why the T3 firing rate should be measured before this run
is authorised: it sets the promotion evaluation size, the other unbounded cost.

The currently authorized M3.1 cloud hard cap is USD 500. Do not expand later
milestone spending without a new frozen pilot and budget decision.


## 13. Execution environment

**The local scientific stages require a Linux host end to end.** This is not a
preference; three independent requirements enforce it, and none of them is
visible until late in the pipeline.

1. **The frozen native libraries are Linux `.so` files.** Fresh-quality root
   materialisation dereferences them, so on Windows it fails with
   `[WinError 193]` — but only at that stage. Everything before it, including
   the entire performance lock, completes happily on Windows, so the mismatch
   surfaces only after a full collection.
2. **Create-only publishing needs `renameat2(RENAME_NOREPLACE)`.** DrvFs — any
   `/mnt/c` or `/mnt/d` path from WSL — returns `EINVAL`, so outer-package
   publish fails there. Run roots must live on a real Linux filesystem; WSL's
   ext4 works and had 934 GB free against ~324 MB per run.
3. **The portable receipt prefers full source replay.**
   `load_preferred_or_pinned_receipt` dereferences the paths recorded in the
   receipt and only falls back to `PINNED_RECEIPT_FILE_SHA256` when they are
   absent. Every local path is sealed at launch, cleanup and receive time, so a
   chain produced on Windows cannot be replayed from Linux and vice versa. No
   file matching the pin currently exists in the tree, so the fallback is not
   an escape hatch.

Together these mean a run started on Windows cannot be finished on Linux: the
sealed `local_destination` and `controller_journal_dir` are Windows strings that
no Linux path resolves to. Start on the platform you intend to finish on.

There is a fourth requirement, and it is the one that costs the most attempts:
**the frozen wheelhouse pins versions, not just distributions.** Installing the
right packages at their latest versions gets you scikit-learn 1.9 against models
pickled with 1.8, which fails at root materialisation with the unhelpful
`No module named '_loss'` — the pickle names an internal module that moved.
Read `wheelhouse_manifest.json`'s `version` field and install exactly those.
Only the ABI tag may differ from the workers' (cp311 there, cp312 under Ubuntu
24.04); the versions must not.

Working WSL setup (no root required):

```bash
python3 -m venv ~/ofc-fq-venv                  # ensurepip ships with Ubuntu 24.04
V=~/ofc-fq-venv/bin
$V/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
$V/pip install numpy==2.2.6 scikit-learn==1.8.0 scipy==1.15.3 lightgbm==4.6.0 \
               joblib==1.4.2 threadpoolctl==3.6.0 networkx==3.4.2 sympy==1.13.1 \
               mpmath==1.3.0 filelock fsspec jinja2 markupsafe pygments \
               typing-extensions==4.13.2
export CLOUDSDK_CONFIG=/mnt/c/Users/<user>/AppData/Roaming/gcloud
export PYTHONPATH=<repo>/src
```

Verify before running anything expensive:

```bash
$V/python -c "import ctypes; ctypes.CDLL('<dev>/native/candidate/release/libofc_hu_m3_engine.so')"
PYTHONPATH=<repo>/src $V/python -c "
from ofc_regular.hu_turn3_model import load_hu_action_value_model
load_hu_action_value_model('<repo>/models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl')"
```

The second line is the check that would have caught the version skew directly,
rather than after a staging attempt.

`CLOUDSDK_CONFIG` matters: WSL's `gcloud` resolves to the Windows SDK through
`/mnt/c` but reads its own empty config, so it reports no active account until
pointed at the Windows credential store. The frozen cp311 wheelhouse is *not*
installed locally — staging only repackages and hash-checks it — so matching its
pins in a local venv cannot perturb pinned evidence; diverging from them only
breaks the local replay.

## 14. Run history, and why wave 0 ran five times

| run | outcome | cause |
|---|---|---|
| `-20260723-001` | all 8 workers dead in 77 ms | defect 1, staged startup name |
| `-20260723-002` | all 8 workers dead in 64 s | defect 2, missing contract |
| `-20260723-003` | 20/20 accepted, merge blocked | per-wave control roots (process error) |
| `-20260723-004` | 20/20 accepted, lock **qualified**, quality blocked | Windows platform mismatch |
| `-20260726-005` | Linux-native, in progress | — |

Only the first two were unavoidable discovery. The other three were preventable,
and the same failure explains all three: **expensive cloud work was started
before the cheap local check that would have blocked it.**

- Defects 1 and 2 were both reproduced afterwards in a WSL harness in seconds,
  using `OFC_FULL100_WAVE_V2_TEST_METADATA` and the real launch bundle. Building
  that harness *first* would have found both without creating a single VM.
- The control-root layout was a five-minute read of what the scientific bridge
  requires. It was generalised from an unrelated epoch-exhaustion incident
  instead.
- The platform mismatch was visible in a file extension.

Before the next collection, verify offline: that every module the *post*-
collection stages import loads; that the native libraries load; that
`renameat2(RENAME_NOREPLACE)` works on the run root; and that cloud credentials
resolve. All four are seconds of work and together they gate roughly three hours
and USD 5 per attempt.

### Operational notes from the Linux run

Two near-misses worth repeating, because both made working steps look broken:

- **Output filters are platform-dependent.** The orchestrators print compact
  JSON on Windows and pretty-printed JSON under WSL, so a `grep '"status":"…"'`
  that worked on Windows silently matched nothing on Linux and made a
  *successful* cleanup look like a failure. Match the spaced form, or drop the
  filter and read the tail. Never conclude failure from an empty filter.
- **Do not `sed`-edit multi-line shell scripts.** Rewriting a runner with `sed`
  broke its line continuations and produced
  `--attempt-ledger: command not found`. Keep one small script per step and
  parameterise it instead.

Run -005 completed 8/8, 8/8 and 4/4 with zero preemptions, which is also the
first collection where no defect surfaced — every fix from section 10b was
already in place.

## 15. Revised plan from here

1. **Finish the lock on run -005**: scientific bridge `--mode merge`, then the
   pure v4 merger, then the production bridge. All three are local, free, and
   minutes; the calling patterns are in section 10c. Verify the resulting
   receipt authorizes quality by calling `build_fresh_quality_plan` on it — it
   raises `performance receipt is not the exact portable pin` if the paths do
   not dereference, which is the exact failure that ended run -004.
2. **Fresh quality (item 2)**: 50 paired hands plus the 5-paired confirmation,
   roughly USD 3. Its local staging is create-only and fails closed with a
   forensic marker, so a failed attempt is cheap and must be retried in a new
   directory.
3. **T3 runtime override integration (item 5-ish)** *before* committing to the
   9,000-paired label run. This is the reordering section 11d argues for: the
   integration is required for M3.1 anyway, and only once it exists can the T3
   override be measured against `stage9f_p2`. That measurement decides whether
   the label run is worth USD 330-500, and it costs nothing to run afterwards.
4. **Then** the label set, training, non-fire proof, promotion, opt-in profile.

Do not start step 4's label run on the strength of the section 11c numbers.
They were measured against the wrong baseline, and section 11c explains why the
bias runs in the direction that would flatter that decision.

## 16. Fresh quality: local staging complete

Run -005's lock finalized on Linux and authorizes quality:

- `performance_lock_v4_production_receipt.json` — `status: qualified`,
  `decision: performance_lock_v4_finalized_qualified_open_quality_pilot_only`
- `build_fresh_quality_plan` on it returns `quality_pilot_authorized: true`,
  which is the check that failed for run -004 with
  `performance receipt is not the exact portable pin`

Staging output, `~/ofc-runs/regular-hu-m31-t3-fqv1-20260726-004`:

```json
{"status": "complete_local_staging_ready_cloud_not_authorized",
 "paired_hand_count": 55, "root_count": 110,
 "job_count": 15, "wave_job_counts": [8, 7], "cloud_called": false}
```

That matches the section 11 contract: 50 primary plus 5 confirmation paired
hands, 110 actor observations, 15 jobs over two waves.

Three earlier attempts under `-fqv1-20260726-001..003` are failed-closed
directories, each holding a `LOCAL_STAGING_FAILED.json` naming the stage and
error. They are retained deliberately — staging is create-only and a failure
must be retried in a new directory, never repaired in place. `-004` is the
authoritative one.

Next: the cloud run via `run_hu_m31_t3_step6d_fresh_quality_gcp_v1.py`, which
consumes this staging directory. Fifteen jobs, roughly USD 3.

## 17. Where the remaining headroom is, per street

Every street already carries a model, and T0/T1/T2 already carry the same
override-plus-safe-selector pair this milestone is building for T3:

```
DEFAULT_HU_TURN0_STAGE19_P0_MODEL / ..._SAFE_SELECTOR_MODEL
DEFAULT_HU_TURN1_STAGE18_P1_MODEL / ..._SAFE_SELECTOR_MODEL
DEFAULT_HU_TURN2_STAGE8B_MODEL
```

`models/` holds 38 turn0, 228 turn1, 169 turn2 and 63 turn3 files. So the M3.x
curriculum is not building street policies from nothing; it is rebuilding them
on top of the exact T4 solver, with a far stricter verification process. The
backward order follows from that: a street cannot be evaluated until the play
below it is correct.

That reframes the open question. It is not "which street is unbuilt" but:

> **How much headroom does each already-built street still have?**

Only turn 1 has been measured, and only against `stage9f_p2`:

| | value |
|---|---:|
| oracle ceiling | +0.92 pts/hand |
| achieved by the shipped selector | +0.06 pts/hand |
| **capture rate** | **~6.5%** |

A street that has already been through one pass is leaving roughly 93% of its
available value unclaimed. T0, T2 and T3 have never been measured this way.

The measurement is cheap — `ofc_regular.trace_hu_turn3_overrides` and its
turn-0/1/2 equivalents run locally, free, in minutes, and section 11c shows the
method. Running it for every street produces a headroom table, and that table
is what should decide where the next USD 330-500 goes.

Two caveats keep this from being a conclusion:

1. The oracle is an unreachable upper bound. It selects on realized outcomes,
   much of which is downstream randomness no decision-time predictor can see.
   A fair version would fit the best predictor available at decision time and
   report *its* capture, not the oracle's.
2. Section 11c's T3 numbers were measured against the trace tool's default
   baseline, not `stage9f_p2`, and that bias flatters the ceiling. Per-street
   comparison must use the production baseline for every street or none.

A further structural question, unresolved: the teacher's budget is
`8/32/4/0` — eight candidate placements, thirty-two evaluations, four
downstream T3 rollouts, and an exact T4. A first-seat decision costs about 92
seconds, and almost all of it is the exact T4 solve rather than search breadth.
So the teacher pays exact-solver prices for a narrow search. Whether a wider
candidate set with exact T4 reserved for the final few would distil better is
untested, and it bears directly on whether more labels of the *current* teacher
are worth buying.

## 18. The T4 evaluation is the cost, and it is the one place with perfect labels

Measured from run `-20260726-005`'s accepted candidate hands (100 hands, 200
roots), reading `source_result.decision`:

| quantity | first seat | second seat | ratio |
|---|---:|---:|---:|
| `child_information_set_count` | 91,116 | 1,332 | 68x |
| `native_latency_ms` | 55,838 | 822 | 68x |
| wall seconds (mean) | 50.90 | 0.78 | 65x |

Latency tracks the child information-set count almost exactly, and
`validation_latency_ms` is 13 ms against 55,838 ms of native search. **The T3
decision cost is the number of T4 evaluations and essentially nothing else.**
The first seat is expensive because the opponent has not placed yet, so the
child set is two orders of magnitude larger.

The budget is `candidate_samples 8 / evaluation_samples 32 /
downstream_t3_samples 4 / downstream_t4_samples 0`, with
`use_t4_action_cache: True` and `runtime_id: hu_m31_t3_crn_exact_t4_v1`. The
zero is not a shortcut: T4 children are solved exactly rather than sampled. The
cache only helps a position that recurs.

### Why this matters for where the money goes

T4 is the only street in this game with a perfect teacher. It is exactly
solvable, so (position -> exact EV, exact best placement) pairs are unlimited
and noise-free. Every other street's teacher is itself an approximation — T3's
32 evaluation samples carry roughly 18% standard error.

So the current error budget is lopsided:

| error source | current |
|---|---|
| T4 leaf evaluation | **zero** — exact |
| unknown-card sampling | 32 samples, ~18% s.e. |
| opponent T3 response | 4 samples |

Exactness removed the smallest term. The dominant error is sampling, and
sampling is starved precisely because the exact leaf is expensive.

A learned T4 evaluator inverts that trade. If it carried, say, 2% error but
made leaves cheap enough to raise evaluation samples from 32 to 512, sampling
error would fall to about 4.4% and the total would improve. `models/` currently
holds no T4 model at all — 38 turn0, 228 turn1, 169 turn2, 63 turn3 files, and
nothing for the final street.

The speedup compounds backwards: cheaper T4 makes T3 labels cheaper, T2 search
uses T3, and so on up the backward curriculum. The 9,000-paired label run
in section 11d is priced against the current leaf cost.

### What such a model must not smooth over

1. **Fouling is a cliff, not a gradient.** It is also decidable exactly and
   cheaply once a placement is fixed, so the natural split is to check fouling
   exactly and have the model predict EV conditional on a legal board.
2. **Fantasy Land is a discrete jump** worth tens of points at the QQ boundary.
   A regressor that interpolates across it will be badly wrong exactly where
   the decision matters. It wants its own head or an explicit feature.
3. **The exactness claim is worth keeping.** It does not have to be given up:
   evaluate the search interior with the model and re-solve the chosen line,
   or the top few, exactly. What gets reported stays exact; only the 91,116
   interior evaluations get cheap.

This pattern is already running in the sibling `ai/` codebase, where a value
network (corr 0.921, MAE 4.08) backs an MCTS evaluator instead of exact
evaluation at every leaf.

### Cheapest way to test it

Nothing here requires cloud spend:

1. Generate exact T4 labels locally — they are free and unlimited.
2. Fit a small evaluator and measure held-out error against exact values.
3. Re-run `evaluate_t3` with the model at the leaf and compare both the
   decision agreement and the wall time against the exact runtime, on the same
   roots the lock already measured.

Step 3 answers the real question directly: does the decision change, and by how
much does 55 seconds fall.

## 19. Exact T4 measured: cost, seat asymmetry, and what to model

Measured locally against the frozen candidate library, on T4 states built from
the lock roots (one T3 placement applied, opponent advanced to eleven cards):

| seat / to_act | n | mean | p95 | legal actions | best_score mean | sd |
|---|---:|---:|---:|---:|---:|---:|
| first / first | 80 | **3.29 ms** | 4.59 ms | 4.2 | +0.919 | 8.465 |
| second / second | 80 | **0.29 ms** | 0.43 ms | 4.4 | -1.131 | 10.668 |

**The seat is not a detail; it is the shape of the problem.** Acting second at
T4 means the opponent's board is already complete, so the actor scores four
legal placements deterministically — 0.29 ms, and nothing worth approximating.
Acting first means the opponent still holds two cards, so every completion has
to be expanded. Same action count, eleven times the cost, and `best_score` even
changes sign between the two.

So a learned leaf should cover **T4-first only**. Second seat stays exact: it is
already fast and it is a deterministic scoring, not a search.

That is also precisely the leaf the expensive path expands. A first-seat T3
decision visits 91,116 child information sets (section 18), and those children
are T4-first states.

### Label economics

At 3.29 ms per exact evaluation, one core produces about **281 labels per
second — roughly one million per hour**, with zero label noise because the
value is exact. Sixteen cores make that sixteen million an hour. T4 is the only
street in this game where the teacher is free, unlimited and correct.

### Expected effect

A small evaluator answers in tens of microseconds, so:

| | exact leaf | learned leaf |
|---|---:|---:|
| per leaf | 3.29 ms | ~0.03 ms |
| first-seat T3 decision | 55.8 s measured | order of 0.5 s |

The same 92-second budget would then buy `evaluation_samples` in the thousands
rather than 32, taking sampling error from roughly 18% to under 2% — which is
the dominant error term identified in section 18.

### What the model has to predict

`evaluate_t4` returns `best_score`, `actions`, `legal_action_count`,
`selected_action_key`, `selection_score_gap` and
`evaluation_sample_regret_of_locked_selection`. With only ~4 legal actions, an
accurate `best_score` makes the action choice nearly free, so the regression
target is the scalar exact EV.

Difficulty is bounded but real: `best_score` has a standard deviation of 8.5 to
10.7 points. For comparison the sibling `ai/` value network reaches corr 0.921 /
MAE 4.08 on a harder problem; T4-first is easier — two cards left, ~4 legal
actions, opponent nearly complete — so a tighter fit should be achievable, but
MAE well under 1.0 is the bar worth aiming at rather than assuming.

Fouling and Fantasy Land still need the treatment in section 18: fouling stays
an exact check, and the FL jump at the QQ boundary wants its own head or an
explicit feature rather than being smoothed across.

## 20. Building the T4-first evaluator: measured pipeline

### Scoring context is frozen into the labels

The lock ran with, and these labels therefore encode:

```json
{"fl_ev": {"14": 10.227020614683454}, "fantasyland_cards": 14,
 "foul_enabled": true, "hu_line_points": true,
 "middle_trips_royalty": 2, "scoop_bonus": 3}
```

`ScoringContext.fl_ev` is a list of (cards, value) pairs, so a Fantasy Land
chain is representable, but only the 14-card entry is set and its value is taken
to already account for chaining, chains being rare. **Changing `fl_ev` later
invalidates every label and the model fitted on them**, so it is worth treating
as frozen alongside the plan.

### Where the states come from

`hu_m31_t3_behavior_roots.generate_behavior_t3_roots` plays T0-T2 with one of
the five frozen behavior profiles and returns both sequential T3 roots, keeping
the opponent's private discards out of every model input. Fresh quality uses the
same routine, so labels built on it inherit that information discipline.

T4-first states are then produced the way the search reaches them: apply a legal
T3 placement, advance the opponent, and deal the remaining unknowns. That keeps
the training distribution equal to the search's query distribution instead of to
uniform random boards.

Stage-3 profiles need the Rust feature encoder. Rather than building one, pin
the frozen copy the lock already uses, via
`hu_turn3_stage3_feature_rust.pinned_feature_encoder_library`, SHA-256
`82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411`.

### Measured costs

| step | rate | per 10,000 |
|---|---:|---:|
| T3 skeleton generation (5 profiles, self-play T0-T2) | 7.5 roots/s | 22 min |
| exact T4-first label | 308 labels/s | 32 s |

Skeleton generation is 40x the cost of labelling, so it is the constraint. About
ten minutes buys 1,000 skeletons and roughly 100,000 labels.

### Label quality, first 2,000

| property | value |
|---|---|
| `best_score` | mean +0.119, sd 8.106, range -29.8 .. +33.2 |
| distinct observations | 2,000 of 2,000 |
| QQ+ on top (FL-qualifying) | 17.6% |
| `legal_action_count` | only 3 or 6 |

FL coverage at 17.6% matters: the model has to learn the discrete jump at the QQ
boundary, and a set that rarely entered it would teach nothing there. The Q pair
count dominating K and A is the signature of real play aiming at Fantasy Land,
which random boards would not reproduce.

The `legal_action_count` taking only two values exposed the limit of seeding
from the lock roots alone: 200 roots means about 100 board skeletons, and
resampling completions multiplies rows without adding structure. That is why the
skeletons are generated fresh rather than reused.

## 21. The T4 evaluator was built and measured. It does not pay off.

Sections 18-20 argued that the exact T4 leaf is where a T3 decision spends its
time, and that a learned leaf should therefore buy a large speedup. The model
was built and measured end to end. **The conclusion does not hold**, and the
reason is worth keeping.

### The model works

Trained on 324,360 exact T4-first labels, held out by skeleton:

| features | corr | MAE |
|---|---:|---:|
| raw card one-hots | 0.038 | 7.06 |
| + `score_board` outcomes per legal action | 0.838 | 3.63 |
| + row-versus-row comparison against the opponent | **0.895** | **2.86** |

The first attempt memorised its training rows at corr 0.9999 while generalising
at 0.038: the encoding separated states but carried no structure to generalise
over. Poker value is a non-linear function of card combinations and asking a
tree to rediscover hand ranking from one-hots is not reasonable. Feeding
`score_board`'s own outputs fixed it, and describing each hero action *against
the opponent's rows* — which is how OFC actually pays — fixed it further.

A useful negative result along the way: correcting the Fantasy Land flag (
`FantasylandEntry` is a dataclass, so `if score.fl_entry` is always true; the
field is `.qualifies`) moved MAE by 0.003. FL was not the binding constraint;
the opponent's description was.

### But the substitution does not pay

Measured on 120 fresh roots from an unused seed base, comparing the T3 action
chosen with exact leaves against the same choice with learned leaves:

| quantity | value |
|---|---|
| same action chosen | **67.5%** |
| regret when it differs | mean +0.247, p95 +1.44, max +2.90 |
| exact leaf time | 118.76 s |
| learned leaf time | 32.19 s |
| **speedup** | **3.7x**, not the ~100x estimated |

Two separate problems. The decision changes one time in three, which defeats
the point of a drop-in leaf — though the regret when it differs is small, so
the model is confusing near-ties rather than blundering. And the speedup is an
order of magnitude below the estimate because **feature extraction costs more
than inference**: `encode` runs `generate_turn_actions` and up to eight
`score_board` calls per leaf, in Python.

### Why the premise was wrong

`score_t4_first_actions` in `rust/hu_m3_engine/src/search.rs` is already
optimised in exactly the way that would have made a learned leaf worthwhile:

```rust
// Opponent terminal boards depend on the future deal, but not on which
// hero action is being evaluated. Score them once and share that exact
// response table across every legal hero action.
let opponent_responses = precompute_t4_opponent_responses(...);
```

The opponent response table is built once per deal set and shared across every
hero action; both loops are `par_iter`; scoring uses a `CompactBoardScore` and a
`_trusted` path that skips validation; ActionKey hashing was deliberately
removed as "pure overhead". What is left for a model to replace is a minimax
over precomputed compact scores — plausibly faster than any learned evaluator.

So the arithmetic in section 18, "91,116 child information sets x 3.29 ms", is
misleading: the children are not 91,116 independent exact solves, because the
expensive part is shared between them. The measured 3.7x is what that structure
actually permits.

### What this does and does not close

- A T4-first evaluator is learnable to corr 0.895 with cheap, exact, unlimited
  labels. That remains true and the pipeline is in section 20.
- Replacing the leaf **does not** change the price of the 9,000-paired label run
  in section 11d. That premise is withdrawn.
- If T3 search time is worth attacking, the place to look is the T3 layer —
  pruning dominated T3 actions, or sharing structure across `evaluation_samples`
  x `downstream_t3_samples` — not inside T4. That is untested speculation, not a
  measurement.
