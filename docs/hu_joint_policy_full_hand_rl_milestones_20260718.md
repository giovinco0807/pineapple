# Regular OFC HU joint-policy revised roadmap

Status date: 2026-07-18

> ⚠️ **注意 (2026-07-21追記)**: この文書の冒頭要約と末尾には
> Step12c以前の時点の記述が残っている(例:「GCE bootstrap未実装、
> `cloud_executable=false`」はStep12b/12cで更新済み)。現在地の判断は
> I1 Step12cチェックリストと以下を優先すること:
> - `docs/hu_joint_policy_m31_t3_step12c_phase2_schema_failure_20260721.md`
> - `docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md`

## 0. 日本語要約

最終目標は、先攻・後攻の両プレイヤーが同じゲーム内で T0 から T4
まで対戦する full-hand 強化学習です。ただし、いきなり最終得点だけで
学習させません。完成済みの後段 search/exact teacher を利用し、
`T3 -> RL環境 -> T2 -> T1 -> T0 -> FL -> league/CFR/ABR`
の順に広げます。

現在地は次の通りです。

- T4 exact component と T3 の性能開発は完了しています。ただしT4の
  release activationは、旧契約が固定したDLLの復旧または新versionの
  exact/parity/performance再検証（R0b）が終わるまで禁止です。
- 再現性 snapshot、fresh T3 roots、Windows/Linux parity、immutable
  package、20ジョブ分の startup smoke はローカルで合格しています。
- 1台診断と2台candidate/reference診断の契約に加え、accepted
  candidate/reference native、feature encoder、exact 3 jobs、runnerを
  含むimmutable worker payloadとLinux startupを実装・検証済みです。
- fake remote上ではgeneration-match、prefix/known-object collision、
  claim/auth分離、upload/heartbeat/DONE、attempt 1 resume、success
  self-delete、failure時のbounded shutdown、全VM不在後のreceive、
  stage 2 controller receiptまで実装・fault injection済みです。
- ただし、GCE metadata bootstrap、package download、実object upload、
  remote collision/quota/price照会、実claim/authorization、VM
  create/deleteは未実装です。`cloud_executable=false`、
  `launch_ready=false`のままです。
- performance-lock、fresh quality、9,000 paired labels、T3モデル、
  実戦promotion、full-hand RL環境はまだ未完了です。
- `current` と既存 P0/P1/P2/T3 baseline は変更していません。

直近の安全な順序は次です。

1. GCE metadata bootstrap、immutable package download、result upload、
   heartbeat、self-deleteを別versionのdirect transportとして実装する。
2. direct transportをfake backendへ接続し、現在合格済みの同じ
   lifecycle contractを再利用する。
3. read-onlyの実remote collision/quota/price preflightを行い、小額cost
   guard内でstage 1 authorizationを別途固定する。
4. 明示的に認可した後、1 VMだけでlifecycleを確認する。
5. 合格時だけcandidate/referenceの2 VMを実行する。
6. それも合格した場合だけ20 VMのT3 performance-lockへ進む。
7. fresh quality、教師label、モデル学習、複数相手とABR評価を通して
   T3を明示opt-in profileとしてpromotionする。
8. その後、両プレイヤーがT0-T4を通して対戦するRust batch環境を作り、
   T2、T1、T0の順にjoint policyを学習する。

「完成」は数学的な完全最適/Nash証明を意味しません。目標は、paired
seat-swapで両seatが強く、複数相手に頑健で、近似best response gapが
小さいことをfresh holdout上で実証したpolicyです。

This document is the current execution roadmap from the accepted T4 component
to a strong, both-seat, full-hand HU policy. It supersedes only the future
execution order and point-in-time status text in
`docs/hu_joint_policy_implementation_milestones.md`,
`configs/hu_joint_policy_revised_roadmap_20260716.json`, and
`docs/regular_ofc_ai_handoff_status.md`. Historical completion audits,
postmortems, contracts, and No-Go artifacts remain immutable evidence; this
document does not rewrite their results.

## 1. Final objective and honest optimality boundary

The target is a Heads-Up Regular OFC Pineapple policy that:

- plays both first and second seat from T0 through T4;
- uses no joker and follows the 5-card T0 / 3-choose-2 T1-T4 rules;
- never observes the opponent's private discards;
- remains robust across a population of opponent policies;
- has positive paired seat-swap value and controlled tail loss;
- has a decreasing learned approximate-best-response gap.

The project distinguishes four different claims:

1. best response against one fixed opponent;
2. a strong self-play policy;
3. an empirically low-exploitability policy;
4. a mathematically proven optimal/Nash policy.

The delivery target is item 2 plus strong empirical evidence toward item 3.
Approximate best response and restricted CFR are evaluation and improvement
tools. They do not provide a proof of item 4.

### Algorithm decision

| Method | Decision for Regular OFC HU |
|---|---|
| iterative supervised learning from search | use immediately for high-quality Q/delta/risk labels |
| Expert Iteration | primary backward-curriculum learning loop |
| AlphaZero-style policy/value search | use its guided-search ideas, not a perfect-information monolith |
| MCCFR | use as a correctness oracle and restricted late-street improver |
| Deep CFR | do not start full-game training; action/chance/belief scale is too costly before the native environment is proven |
| NFSP | do not use as the primary learner; historical/BR mixture is handled by the explicit league |
| ReBeL-style belief plus search | add after the public-history belief and batch environment pass RLE |
| population/league self-play | mandatory for robustness and anti-overfitting |
| PPO/DQN | do not train from sparse terminal reward from scratch; allow low-weight off-policy V-trace only after supervised initialization |
| search + RL/CFR hybrid | selected final architecture |

## 2. Current verified state

Repository state at this revision:

- branch: `codex/regular-ofc-pineapple`;
- HEAD: `623e3948cb70b05be2958f0b1c63b6f02c7fb756`;
- initial audit working tree: 70 tracked modifications and 884 untracked
  entries; this roadmap and its new local control artifacts add untracked
  entries without deleting or overwriting the initial state;
- all existing changes must be preserved;
- rearm2 Compute Engine VM count and rearm2 cloud mutation count: zero;
- project-wide VM count is not a valid rearm2 invariant: eight concurrently
  running `s13a-canary-*` instances belong to the separate `C:\pokerHU`
  workflow and are deliberately not modified by this roadmap;
- `ai_profiles.py` SHA-256:
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`;
- `current` and the legacy P0/P1/P2/T3 rollback chain are unchanged.

Completed:

- M3.0 T4 exact runtime is complete locally as an explicit opt-in component.
- T4 second seat is exhaustive terminal evaluation.
- T4 first seat is exact under the declared uniform exchangeable
  information-set belief.
- Candidate02 T3 performance-development passed 100 paired hands / 200 roots:
  - semantic parity: 100%;
  - first-seat p95 / max: 86.05 / 88.01 seconds;
  - second-seat p95: 1.048 seconds;
  - first-seat geometric-mean speedup: 2.700x;
- peak RSS: 124,416,000 bytes.
- R0a-v2 reproducibility freeze is Go with 798 payload entries.
- The local I1 rearm2 lifecycle is complete through the immutable package and
  exhaustive actual-package startup smoke:
  - fresh roots: 100 paired hands / 200 seat observations;
  - every root, observation fingerprint, and seed is unique;
  - overlap with development, old-v1, and rearm1: zero;
  - fixed 512-row Windows/Linux encoder output: bit-exact;
  - actual packaged startup manifests: 20/20 passed with all 100 temporary
    root payloads poisoned and zero root reads;
  - Spot claim, launch authorization, VM, and cloud mutation: none.
- A separate fail-closed diagnostic canary contract is locally complete:
  - stage 1 freezes exactly one candidate job;
  - stage 2 freezes exactly one candidate and one reference job over identical
    stage-2 roots;
  - both stages use only performance-development roots and cannot become
    performance-lock, quality, training, or promotion evidence;
  - contract SHA-256:
    `f1ad641fc4bc4953258b13f0b31f202553210f50b937fdac2b6d560c190bb80d`;
  - it is deliberately not cloud-capable; no VM-deployable diagnostic package,
    remote startup/receive lifecycle, claim, or launch authorization exists
    yet.
- Its local lifecycle implementation is also complete:
  - immutable diagnostic package and standalone pre-content verifier;
  - exact ordered, write-once uploads and deterministic heartbeats;
  - `DONE` published last and immutable receive receipts;
  - stage 2 cannot open without a validated stage-1 receive receipt;
  - interrupted/resumed and clean executions produce byte-identical aggregate
    output;
  - hidden fields, out-of-order work, file-set drift, and receipt tampering fail
    closed;
  - stage-open v2 embeds and revalidates the canonical stage-1 receipt, so a
    forged 64-character prerequisite hash no longer passes either the Python
    validator or standalone startup verifier;
  - the combined rearm2 diagnostic and existing package/startup/recovery/parity
    regression set passes 55 tests;
  - this is local lifecycle evidence only, not a VM or cloud result.
- The local-only diagnostic VM preview adapter is complete:
  - exact stage membership is fixed at one VM identity for stage 1 and two for
    stage 2;
  - stage 2 revalidates the actual stage-1 receive directory and binds its
    package, run, ordered jobs, receipt hash, and semantic aggregate;
  - metadata, remote URI manifests, ordered uploads/heartbeats, `DONE`-last,
    receive, stale/collision rejection, success self-delete request, and bounded
    failure shutdown are represented and validated without a cloud call;
  - result payloads use an exact-key schema rather than the former generic
    hidden-field denylist;
  - the planning ceiling is `$0.57/VM-hour`, 4,200 seconds per attempt, at most
    two attempts per job, `$3.99` combined maximum, and a `$4.00` diagnostic
    phase cap;
  - `launch_ready` is explicitly false: the direct GCE transport and read-only
    real-cloud preflight now exist, but no package provisioning, real
    controller claim/authorization, worker launch permission, upload, or VM
    create has been authorized.
- The immutable diagnostic worker payload and integrated fake transport are
  locally complete:
  - the package contains the exact three runner-compatible jobs, 33 Python
    modules, three configs, one pinned runtime requirement, 20 development
    roots, accepted candidate/reference Linux libraries, and the feature
    encoder;
  - package validation, the real Linux startup entrypoint smoke, isolated
    imports, all 30 seeded roots, and all three ELF libraries pass locally;
  - result identity is retry-invariant while attempt control identity is
    attempt-specific;
  - generation-match-zero, empty-prefix and known-object collision rejection,
    trailing-upload recovery, full-prefix/no-DONE recovery, DONE-last,
    success self-delete, failure preservation with bounded shutdown, and
    controller-proven VM absence pass fault injection;
  - stage 2 accepts only an opaque stage-1 controller proof bound to exact
    receive content, success lifecycle records, and VM absence;
  - the current complete rearm2 regression selection passes 228 tests;
  - this remains local transport-contract evidence only. Runner result content
    validation is deliberately deferred to a future real receiver, and none of
    these fixtures is scientific, quality, training, or promotion evidence.

Current release-boundary exception:

- the M3.0 config pins
  `target/m30_build/release/ofc_hu_m3_engine.dll` at SHA-256
  `03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69`;
- that pinned DLL is currently absent, so the current checkout cannot pass R0b
  or activate M3.0 exact runtime;
- a 2026-07-18 isolated validation-target rebuild currently has SHA-256
  `faf1eec9a86a4c2278b6e2d323d6df2dbf9fa1050911d6807894d9b4e6566261`
  at `target/m30_rebuild_20260718/release/` as an unaccepted candidate;
- the rebuilt binary is not a replacement for the pinned M3.0 binary unless a
  fresh parity, exactness, performance, and fail-closed validation freezes a
  new versioned runtime contract.

Not completed:

- independent Candidate02 performance-lock;
- fresh T3 quality pilot and high-precision confirmation;
- 9,000 paired T3 labels;
- `StreetPolicyNetV1` training and calibration;
- T3 population/ABR promotion;
- Rust full-hand T0-T4 RL environment;
- T2, T1, T0 safe curriculum;
- explicit 14-card FL transition;
- joint league iteration and low-exploitability promotion.

The two failed Spot lock attempts failed before root/result content was read:

- the first attempt failed on Windows/Linux path casing;
- rearm1 failed on a startup job-manifest schema mismatch on all 20 VMs;
- neither failure is Candidate02 scientific performance evidence;
- the canonical rearm1 closeout is now frozen at SHA-256
  `fa9041b064a11db2b24e9b2051aee0bc72661b384fd4ff84b89c62331eeb4214`;
- the rearm2 pre-content plan is now frozen at SHA-256
  `8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5`;
- the rearm2 scientific claim is frozen at SHA-256
  `e7f53bc050319f46059bd315e4c9649dc1534bf2c91b87b7fbc547eb1a1309d0`;
- root materialization / seal SHA-256 values are
  `88e535478f892e7d981295cb43fab915b9ecdee89b451b851587c9747535e724`
  and
  `a25b2d67247c92407d1fd44ba2193d8a200800cc4c93bc004f548be4d866f4de`;
- aggregate fresh-root SHA-256 is
  `bb0cf7fb538eab401f10464a555aa2d9b3bebf870444ecf6c0b7a42190da883b`;
- parity receipt, package manifest, package source, and actual-package smoke
  receipt SHA-256 values are respectively
  `9585453e6b0c77cc4cbdf86c9f097cba87add9e7c058cd018b1729e963a62f4e`,
  `92c7977f8a701c83afb53f06ca4ee96e46ba67c891c5769babdc7c70614c05cd`,
  `8aa762cd31c61b33ec8ee984786f723a9b2372b4aa673cfad20e32ed6187efea`,
  and
  `d194f1235cdeb5f74ae96d203c84b381be58c2280aca7833cc55e335395acb77`;
- no rearm2 Spot claim, launch authorization, VM, cloud result, or scientific
  performance-lock result is open.

## 3. Revised dependency order

```text
R0a reproducible T3 research snapshot
  -> I1 proven package/lifecycle path
  -> M3.1 T3 practical promotion
       ^ requires R0b exact-T4 release-runtime recovery before activation
  -> RLA/RLB full-hand contract and Rust mechanics
  -> RLC/RLD batch, replay, resume, and population
  -> RLE RL-entry pilot
  -> M3.2 T2 curriculum
  -> M4R T1 curriculum
  -> M5 T0 curriculum
  -> M5F explicit FL
  -> M6 joint full-hand league iteration
  -> M7 empirical low-exploitability opt-in release
```

T2, T1, and T0 curriculum stages are not separate games. Every trajectory is
a complete two-player T0-T4 hand. During backward curriculum, accepted later
street components are frozen while the current street is improved. M6 then
jointly unfreezes the learned components.

## 4. Milestones and Go/No-Go gates

### R0a - Reproducible T3 research snapshot

Purpose: preserve, rather than clean, the current dirty working tree while
making the accepted Candidate02 Linux research inputs and every later T3 result
reproducible.

Status: Go. Immutable R0a-v2 contains 798 payload entries from a 799-entry
inventory, including all 385 `src/ofc_regular` Python files, the three
historical 100-root trees and controls, startup/parity sources, accepted
binaries/models, rearm2 closeout/plan, and focused lifecycle tests. Only the
R0b Windows DLL role is deferred. The independently verified snapshot is
`outputs/hu_joint_policy/r0_snapshot/hu-joint-policy-r0a-v2-20260718`:

- spec SHA-256:
  `98c35653bb6651850228ec4639e82a3fc2900f37072dcfefc0c07b44f21229d5`;
- payload SHA-256:
  `5c8ee942d25e9b037061ad152dd4420a4b5a4d41f8fe1c4c3fb968a312ff8568`;
- READY SHA-256:
  `af8338cdc12e141800c0a2009a101781b5cee9dac8ddd33559e4aa14107b9620`.

The earlier 84-entry R0a-v1 snapshot remains immutable evidence but is not
sufficient by itself to reconstruct the rearm2 lifecycle.

Deliverables:

- content-addressed inventory of source, config, Rust binary, model, and
  decision-critical artifacts;
- classification of source changes versus generated outputs;
- recorded Git HEAD, branch, dirty fingerprints, command, profile chain, and
  required hashes;
- Python/Rust focused regressions, information-set guards, and current-profile
  hash verification;
- no staging, deletion, profile activation, or cloud creation.

Go:

- every required input is hashed and resolves inside the declared repository or
  immutable artifact root;
- unknown or missing required inputs fail closed;
- `current`, `stage19_p0`, `stage18_p1`, `stage9f_p2`, and
  `stage7_m5_r10` remain unchanged;
- no hidden opponent discard can reach an `ActorObservation`.

Estimated calendar time: 0.5-2 days.

### R0b - Exact-T4 release-runtime recovery

Purpose: restore an activatable Windows exact-T4 runtime without changing the
historical M3.0 contract.

Status: No-Go. The pinned M3.0 DLL is absent. The rebuilt candidate passed 20
focused exact/runtime tests, but its fresh 100-state pilot failed only the
frozen second-seat p99 latency gate (`2.3487 ms > 2.0 ms`). Exact enumeration,
Python parity, scalar/batch parity, determinism, first-seat latency, and all
other pilot gates passed. The failed seed is consumed and the 1,000-state
expansion did not run.

Allowed resolution:

1. recover the exact pinned DLL and verify its SHA-256; or
2. create a new versioned candidate/runtime contract, use fresh non-overlapping
   seeds, and rerun the full M3.0 validation ladder.

The old M3.0 config and evidence remain unchanged. R0b may proceed in parallel
with I1/T3 data infrastructure, but it must be Go before T3 runtime promotion
or any full-hand release candidate is activated.

### I1 - Proven package and lifecycle path

Scientific identity and deployment identity are separated:

- scientific identity: roots, seeds, binary, budgets, and locked gates;
- deployment identity: archive, startup script, image, job manifest, VM
  allocation, and receipt paths.

Status: local Go through step 10c.1 below. The canonical closeout, plan, one-shot
scientific claim, 100-root materialization, disjointness seal, platform parity,
immutable package, 20 actual job manifests, and write-once preauthorization
smoke receipt are frozen and independently validated. The run contract digest
is
`39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5`.
No Spot claim, launch authorization, VM, cloud result, profile change, quality
claim, training claim, or promotion claim exists.

The local diagnostic canary contract needed by steps 11 and 12 is also frozen
and audited. Its separate local lifecycle now covers the accepted native worker
payload, pre-content Linux startup, heartbeat/upload/DONE/receive, stage
dependency, resume, controller receipt, and tamper rejection. It does not
weaken or subset the accepted all-20 launcher. The direct GCE metadata
bootstrap, immutable package download, real remote upload/receive,
success/failure VM shutdown implementation, cloud authorization, and read-only
remote preflight remain to be implemented before either diagnostic stage can
touch cloud resources.

Required order:

1. [x] Close rearm1 as a canonical pre-content infrastructure failure.
2. [x] Freeze the rearm2 pre-content plan.
3. [x] Run the read-only pre-open validation while claim, roots, and cloud
   remain absent.
4. [x] After R0a Go, create the one-shot scientific root-open claim.
5. [x] Materialize and seal the fresh rearm2 roots.
6. [x] Create/reuse the bounded platform probes and freeze fresh Windows/Linux
   parity evidence.
7. [x] Build the actual archive and all 20 actual job manifests.
8. [x] Execute the real startup-contract verifier for all 20 packaged jobs
   without reading root contents.
9. [x] Persist and validate the package-bound write-once preauthorization
   receipt.
10. [x] Verify resume, heartbeat, upload, DONE, and result validation locally
     where possible.
    - [x] 10a: upgrade the local stage-open contract to v2 and bind the
      canonical validated stage-1 receipt.
    - [x] 10b: freeze and test the no-cloud VM metadata/remote lifecycle
      preview with `launch_ready=false`.
    - [x] 10c.1: build the immutable solver/native diagnostic worker payload,
      Linux worker startup, exact result-tree adapter, and pass fake-remote
      fault-injection tests.
    - [x] 10c.2: implement the direct GCE metadata bootstrap plus immutable
      package download/result upload transport, then pass a fresh actual
      GET-only real-cloud prefix/quota/price preflight while
      `launch_ready=false`.
      - [x] Implement and harden the direct transport, generation-bound
        receiver, and fixed-endpoint GET-only collector.
      - [x] Pass 110 focused 10c.2 tests and the complete 228-test rearm2
        regression selection.
      - [x] Freeze a local wheel-complete v3 contract and outer package:
        contract SHA
        `484308d6861df2bde2652afcbef2c7dfc2efb614a59602a8b3e055c23d4426b3`,
        outer identity
        `c6f880b77172215728f15ee4b8ab181d62f0fff0d964b502ff724f2e50ad2cb3`,
        and direct-stage identity
        `b264b6361eed8485a4a84acdbdec8618a47bf311456ec8c009c20bbf57cb1d05`.
      - [x] Execute one actual GET-only v1 diagnostic preflight. It stopped
        fail-closed at the dedicated service-account endpoint with HTTP 404;
        access tokens and authorization headers were not recorded, and no
        cloud mutation, GCS write, IAM change, claim, authorization, or VM
        create occurred.
      - [x] Provision and independently authorize the dedicated service
        account plus exact condition-scoped `storage.objects.get/create`
        bindings, then run one fresh v3 read-only preflight through image,
        quota, retry-name collision, prefix, and Billing SKU checks.
      - [x] The first v3 attempt remained fail-closed on an obsolete quota
        source, and `quota_fix1` remained fail-closed on an incorrect local
        package path. Both immutable No-Go receipts are retained; neither
        attempt performed a cloud mutation or VM create.
      - [x] Fresh actual GET-only v3 `quota_fix2` passed the receiver-only
        gate and is retained at
        `outputs/hu_joint_policy/m31_t3_step6d/rearm2_diagnostic_10c2_actual_readonly_preflight_v3_quota_fix2.json`;
        independent review then found its `$0.60` receiver ceiling did not
        match the legacy controller's `$0.50` cost guard. It is historical
        evidence, not the 10c.2 closeout receipt.
      - [x] Freeze a coherent wheel-complete v4 contract with a shared
        `$0.57/VM-hour` ceiling and `$3.99 < $4.00` all-stage/all-attempt
        diagnostic guard. Contract file SHA-256 is
        `963d8571de3e44b395a758b6fd11e00baf05a4ba8a7528a5817ab7fe2a07916e`,
        outer identity is
        `200246e7ff15787f8a98ac805214edf4db657e59da3d87f7498d480345e49b01`,
        and direct-stage identity is
        `f48a46cce21f19f9e8ceca83fa1b9b8f100b9f06abd07754cbec85fc89c6bd79`.
        The exact objectCreator condition was rotated to the v4 stage prefix;
        the namespace-wide objectViewer condition did not change.
      - [x] The first v4 GET-only receipt remained fail-closed because eight
        unrelated `solver-v2` Spot VMs started in the shared project after the
        prior observation. No VM was deleted. The inventory gate now accepts
        only strictly sized `*-standard-N` instances, counts every status as a
        conservative global/C4 usage upper bound, keeps reservations,
        nodeGroups, and futureReservations empty-only, and rejects ambiguous
        machine types or reservation consumption.
      - [x] Fresh actual GET-only v4 `inventory_fix1` passed with
        `read_only_observation_passed=true`. The immutable closeout receipt is
        `outputs/hu_joint_policy/m31_t3_step6d/rearm2_diagnostic_10c2_actual_readonly_preflight_v4_inventory_fix1.json`;
        embedded artifact SHA-256 is
        `14f6f34c2aabf00640b369594a4fa09338ebd9d18ebbd99cc5e6f559fd4e61d9`,
        file SHA-256 is
        `2e9a27577514592e8e30006bcc65cfad85d16e38352699d2c04d0b50834ac69b`,
        and retrieval time is `2026-07-18T14:06:00Z`.
      - [x] The closeout receipt binds the exact project/bucket/region/zone,
        `ofc-m31-t3-diagnostic` service account, two exact condition-scoped
        bucket bindings, attempt-0/attempt-1 instance GET404 observations,
        empty package/stage prefixes, eight unrelated standard instances and
        their 128-vCPU conservative global usage upper bound, global
        `512-128=384`, Tokyo C4 `24-0=24`, and Tokyo Spot
        `468-64=404` vCPU headroom for a 16-vCPU request. It also binds Billing
        SKUs `8E5C-A4DA-B065`/`AEC9-84A2-D49E`; the observed
        `c4-standard-16` Spot price was `$0.55932/VM-hour`, below the frozen
        `$0.57/VM-hour` ceiling. Unrelated VM names are not serialized.
      - [x] The pinned Debian image is still `READY` but is now
        `DEPRECATED`, with replacement
        `debian-12-bookworm-v20260714`; this is recorded as a launch warning,
        not silently substituted.

The actual read-only gate is deliberately separate from launch readiness.
The completed GET-only preflight is read-only eligibility evidence only and
cannot authorize a VM. `compute.instances.delete`, the worker OAuth scope,
package provisioning, asymmetric controller authorization/claim, and VM create
remain separate Step 11 requirements. The current v4 contract remains
`cloud_executable=false`, `launch_ready=false`, `launch_authorized=false`,
`claim_created=false`, and `vm_created=false`.
11. [x] After separate cloud authorization, run one lifecycle-only VM smoke
    with diagnostic seeds.
    - [x] Before launch, cover shell/import/argparse failures with the same
      bounded shutdown guarantee.
    - [x] Verify UBLA, collect live project/bucket/service-account IAM
      evidence, and explicitly record the bounded shared-project exception.
      The strict ancestor-deny/effective-permission gate was not claimed; the
      exception authorized only this exact Step 11 VM and authorizes neither a
      retry nor Step 12.
    - [x] Run exactly one authorized candidate attempt-0 VM and prove
      insert/provider identity, signed claim CAS, immediate post-claim
      launch/actAs revocation, and fail-closed exact VM/disk/IAM cleanup.
    - [x] Complete the lifecycle result path. v10 stopped because a
      condition-scoped controller GET returned HTTP 403 while polling a
      not-yet-existing `DONE` object. The independent GET-only collector fixed
      that edge. A separately authorized v11 VM then reached signed claim and
      post-claim revocation, but its inner transport failed before the first
      heartbeat because the six-file bootstrap closure imported an undeclared
      development module. The retained v11 closure reproduces that ImportError
      with repository and `site-packages` access removed. The fix3 VM completed
      all 10 hands and published 10 heartbeats, 10 upload envelopes, and a
      complete 23-object runner tree before self-deleting.
    - [x] Prepare and freeze the local v12 correction: make runtime contract validation
      independent of the development import graph, lazily load adapter-only
      dependencies after worker extraction, accept exactly one newline-delimited
      OAuth scope, add an isolated bootstrap-closure regression, add secret-free
      phase markers, and fail fast when the exact VM becomes terminal before
      `DONE`. Runner validation now uses the offline worker venv, accepts the
      normal Debian venv Python symlink, verifies `pyvenv.cfg` and
      `sys.prefix`, and does not require NumPy in the bootstrap parent.
      Focused regression: `136 passed`.
      Complete rearm2 diagnostic regression: `293 passed`.
    - [x] Build and freeze a new content-addressed outer package and pass its
      fresh local preflight at
      `outputs/hu_joint_policy/m31_t3_step6d/step11_v12_local_preflight_fix3`.
      Receipt SHA is
      `c6dec3db3be51f93120536b93a577261812eff785e8ada8f9d1ea5b91312df5e`;
      outer identity is
      `593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548`;
      direct-stage identity is
      `8e27e55dab68ae9697c1c40c80353e8905ad1339423c52d51070f559fbceb44a`.
      The immutable v9/v10/v11 package was not overwritten.
    - [x] Obtain fresh explicit authorization for exactly one new candidate
      attempt-0 VM. Exactly one VM was created; no attempt-1 or second VM was
      launched. See
      `docs/hu_joint_policy_m31_t3_step11_attempt0_postmortem.md`.
    - [x] Recover and independently validate the fix3 result. The controller
      initially returned a false negative because it polled the legacy
      `hu-m31-r2diag-worker-v1` preview URI while the worker correctly
      published to the content-addressed direct-v1 result URI. The retained
      direct-v1 result was then generation-pinned, strictly remapped from the
      legacy provenance URIs, freshly materialized, and passed the runner
      validator. Recovery receipt SHA:
      `e92e5e8af28e7977870e51443108c4a17ebac18f5e6692228c7265d6844051c7`.
      The controller now monitors and materializes from `remote_layout`; a
      final `DONE` re-read also covers the genuine publication/self-delete
      race.
12. [ ] Run a two-VM candidate/reference end-to-end canary.
    - [x] Re-contract and validate a diagnostic-only two-VM topology using one
      `c4-standard-8` candidate and one `c4-standard-8` reference. The frozen
      inner contract remains `c4-standard-16` with Rayon 16, so this
      oversubscribed topology is lifecycle evidence only and can never become
      performance, quality, training, or promotion evidence.
    - [x] Add pair-wide exact attempt-0 identities, common run-contract
      binding, direct-v1 result monitoring, six-endpoint authoritative capacity
      evidence, exact IAM-set readback, both-claim-before-revoke, launch-time
      expiry checks, and independent two-VM/two-disk/all-IAM cleanup. The final
      frozen runner SHA-256 was
      `2699d53f0fbb27549f83c7fa505336837190d5249129caca365351f3d2f7382b`.
    - [x] Execute the separately authorized command once. It failed closed
      before the pair controller and before any Compute insert because the
      controller service-account access token did not become available within
      the 39-second 403-only propagation loop. All ten temporary IAM bindings
      were removed.
    - [x] Independently close out the failed execution: live VM/disk/result
      counts are zero, exact temporary IAM is zero, exact insert/operation
      counts are zero, all 40 canonical/cross-reference checks pass, and
      `current` is unchanged. Closeout receipt SHA-256:
      `717c89023fddc7185aa65448a7706f58e86329e6c28e25d03023a0113c44cb29`.
      See `docs/hu_joint_policy_m31_t3_step12_preinsert_failure_20260719.md`.
    - [x] Implement and validate the two-phase controller-token propagation
      barrier and a new direct-v2 exact-pair runner. Before launch, the runner,
      failure cleanup, independent zero readback, exact 44+44 receiver, and
      immutable-output boundaries passed the complete Step 12b regression.
    - [x] Execute the separately authorized direct-v2 command once. Token mint
      and Token Creator removal both succeeded, but the normal empty IAM policy
      response omitted `bindings`; the token barrier failed closed because its
      post-remove reader did not normalize that omission. Phase2 and Compute
      insert were never entered.
    - [x] Independently close out the direct-v2 failure. Two VM and two disk
      GETs are 404, the run-scoped service account is absent, all eight Phase2
      bindings are zero, direct-v2 results are zero, the three fresh source
      objects remain generation-pinned, old artifacts and `current` are
      unchanged, and no credential was persisted. Closeout receipt SHA-256:
      `c66e75f8dfbc877e7602467231d361054bc382161e724ecd3e5acda88696b2b0`.
      See
      `docs/hu_joint_policy_m31_t3_step12b_token_barrier_failure_20260720.md`.
    - [x] Normalize an omitted IAM `bindings` field to an exact empty list,
      retain rejection of malformed values, preserve bounded no-remint
      readback, and persist future token-barrier failure receipts. Focused
      tests are `47 passed`; the complete Step 12b regression is `263 passed`.
    - [x] Add the versioned Step12c fresh-identity entrypoint and reject every
      terminal Step12b identity/output surface before constructing a live
      backend. Correct the live `c4-standard-8` memory contract from the stale
      mocked `32768 MiB` value to the provider's exact `30720 MiB` readback.
    - [x] Execute the separately authorized Step12c attempt once. It crossed
      the corrected token barrier but stopped before the first Phase2 IAM
      mutation because the Phase2 validator's exact success-field allowlist
      omitted the two new TokenCreator zero-readback fields. No Compute insert
      was issued.
    - [x] Independently close out Step12c: all eight Phase2 bindings are zero,
      two VM and two disk GETs are 404, the controller service account is
      absent, results are zero, three source objects remain generation-pinned,
      old Step12b artifacts and `current` are unchanged, and no credential was
      persisted. Closeout receipt SHA-256:
      `31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef`.
      See
      `docs/hu_joint_policy_m31_t3_step12c_phase2_schema_failure_20260721.md`.
    - [x] Extend the Phase2 validator to accept only the exact new fields and
      validate the full nested zero-readback evidence. The actual Step12c
      success receipt is now a regression fixture.
    - [x] Re-run the complete Step12b/Step12c regression boundary after the
      correction: `275 passed in 166.97s`.
    - [ ] Obtain fresh explicit authorization before another pair canary. The
      consumed Step12c identity is terminal with `retry=false` and
      `resume=false`; signer, contract, source prefix, output root, service
      account, VM names, and disk names must all be new.
13. [ ] Only after the two-VM canary passes, redesign or raise the Tokyo C4
    family quota for performance-lock. The current quota is 24 vCPU; the old
    twenty-`c4-standard-16` plan requires 320 vCPU and is not launchable.

Canary seeds never become performance-lock or quality evidence.

Go:

- actual-package preauthorization is 20/20;
- startup schema is derived from the implementation rather than duplicated as
  stale shell constants;
- candidate and reference use separate processes/VMs for performance evidence;
- interruption/resume produces byte-identical aggregates;
- no root is consumed before the content-access claim is valid.

Estimated calendar time: 1-2 days.

### M3.1 - Hidden-discard-safe T3 practical promotion

#### M3.1a Independent performance-lock

- fresh 100 paired hands / 200 roots;
- semantic parity: 100%;
- first-seat p95 <= 150 s and p99/max <= 240 s;
- second-seat p95 <= 5 s;
- candidate/reference first-seat geometric-mean speedup >= 1.55x;
- RSS <= 0.8 GiB;
- missing, censored, mixed-binary, image, and allocation records: zero.

#### M3.1b Fresh quality

- fresh 50 paired hands / 100 roots;
- separate 5-paired `8/128/4/0` confirmation;
- confirmation regret mean/p95/p99/max <= 0.75/3/6/15;
- hidden truth, unknown field, ActionKey drift, RNG overlap, and missing
  records: zero.

#### M3.1c Data and model

Fixed paired-hand split:

| Role | Paired hands |
|---|---:|
| train | 6,000 |
| safety-fit | 1,000 |
| threshold-lock | 1,000 |
| diagnostic holdout | 1,000 |

Exactly 10% of each split receives the separately budgeted high-precision
confirmation label. Confirmation membership is preregistered and cannot be
changed after any Q, regret, timing, or model output is observed.

The first 25-paired shard must complete, resume, receive, and validate before
fanout.

The model uses:

- order-invariant card/zone set encoder;
- public-history and observation-derived belief encoder;
- semantic `ActionKey` action encoder;
- policy, state-value, action-Q, baseline-delta, uncertainty, and safety heads;
- shared first/second backbone with seat embedding and seat calibration;
- street-specific adapters/checkpoints.

Core model weights use train only. Risk/safety fit uses safety-fit only.
Threshold-lock cannot modify model weights. Diagnostic holdout cannot select
thresholds.

#### M3.1d Runtime and practical promotion

An override may fire only when:

- candidate and baseline semantic ActionKeys differ;
- predicted delta minus downside p95 and ensemble disagreement is positive;
- seat-calibrated safety probability exceeds the frozen threshold;
- input schema, action mapping, and model hashes match exactly.

On non-fire, return the exact baseline Action object without consuming policy
RNG or changing the future deck or trajectory.

Promotion requires:

- five-opponent population and at least three ABR families;
- paired seat-swap reporting overall and by seat;
- at least 300 valid overrides and at least 100 per seat;
- gain/override and EV/hand CI95 lower bounds above zero;
- false-positive override rate <= 0.30 overall and <= 0.35 in each seat;
- override-loss p95 / p99 / max <= 25 / 40 / 50 points;
- non-fire trajectory mismatch: zero;
- each opponent-family mean >= -0.005 point/hand and CI95 lower bound
  >= -0.02 point/hand;
- learned ABR worst-response result >= -0.01 point/hand;
- explicit opt-in profile only.

Estimated M3.1 calendar time after I1: 3-6 days.

### RLA - Full-hand contract

A normal hand has ten actor decisions:

```text
T0 first -> T0 second -> ... -> T4 first -> T4 second -> terminal HU score
```

Contract:

- 52 cards, no joker;
- T0 places all five cards; T1-T4 place two of three and privately discard one;
- terminal rows are top 3 / middle 5 / bottom 5;
- HU score includes normal royalties, foul, and scoop;
- middle trips royalty is exactly 2 points;
- normal-hand FL entry uses the standard Regular OFC rule;
- FL entry and stay both use explicit 14-card mode;
- canonical ActionKey order, at most 232 legal actions;
- illegal action masking with no fallback-to-action-zero behavior;
- versioned public placement history;
- actor-relative observation containing hero board, opponent public board,
  hero private discards, dealt cards, seat/order/turn, FL/scoring context, and
  observation-derived belief features;
- opponent discard, realized deck tail, and world identity excluded.

Go on Python/Rust golden-schema and full street/seat geometry parity, public
history reconstruction, zero hidden-field injection, and zero ActionKey
collision.

### RLB - Rust full-hand mechanics

Add `rust/hu_rl_engine` on top of the accepted HU M3 engine.

Private `WorldState` contains deck/cursor, both private discards, actor/street,
policy assignment, RNG counters, and audit truth. Public API exposes only
ActorObservation, legal ActionKeys/mask, reward/done, and audit token.

Go:

- 10,000 scripted/random legal full hands match the Python reference for every
  action, board, discard count, and terminal score;
- 52-card conservation and zero-sum scoring are exact;
- illegal ActionKey raises an error;
- hidden truth exposure is zero.

### RLC - PyO3 batch API

Minimum API:

- `reset_batch`;
- `observe_batch`;
- `legal_actions_batch`;
- `step_batch`;
- `snapshot_batch` / `restore_batch`;
- later `search_label_batch` and `rollout_population_batch`.

Go:

- scalar/batch byte parity;
- lane order, chunk width, and thread-count invariance;
- dealt-card permutation invariance;
- 100,000 paired hands resume to a byte-identical aggregate;
- mechanics throughput >= 20,000 actor decisions/s on `c4-standard-16`;
- >= 10x Python scalar mechanics throughput;
- 4,096 environments use <= 2 GiB RSS.

### RLD - Immutable replay and population

Replay stores only:

- ActorObservation and public history;
- chosen ActionKey and legal-set digest;
- behavior log-probability;
- reward/done;
- policy/checkpoint ID;
- RNG provenance and audit token.

Truth/audit data is a separate access-controlled artifact and is never a policy
training input.

Go on one million off-policy episodes with 100% trajectory/return
reconstruction, exact seat/population quotas, and zero duplicate, gap, RNG
collision, or unknown field.

### RLE - RL-entry acceptance

- connect accepted T3/T4 search labels;
- run a small shared seat-aware model through all-street forward/backward;
- probability outside the legal mask is zero;
- checkpoint, replay, and return reconstruction is 100%;
- complete a 100-1,000 state pilot before any large generation.

RLE Go means the project can safely start RL. It is not a strength claim.

Estimated RLA-RLE calendar time: 5-10 days.

### Common search, learning, and league contract

The following rules apply to T3, T2, T1, and T0:

- enumerate the complete legal action set and sort it by semantic `ActionKey`;
- use common random futures when comparing actions within one state;
- give candidate selection/search and final evaluation disjoint RNG namespaces,
  seed ranges, and artifacts;
- require every scalable generator/evaluator CLI to expose and record
  `--seed-stride`, reject overlapping seed sets, and include RNG provenance in
  its immutable manifest;
- never use evaluation futures to choose candidates, thresholds, or models;
- never report teacher Q/EV as realized full-hand EV;
- keep search labels, behavior trajectories, audit truth, and locked evaluation
  results as distinct artifact roles;
- balance first/second seat at root collection and full-hand evaluation;
- reject a shared-seat model if either seat fails its separately locked
  calibration or EV gate.

Every street uses two Expert Iteration cycles:

1. collect balanced full-hand population roots;
2. label Q, baseline delta, uncertainty, and hard negatives with search;
3. distill policy/value/Q/safety heads;
4. add disagreement, tail, and false-positive hard negatives;
5. add low-weight V-trace only in cycle 2;
6. evaluate on fresh population and ABR seeds.

Initial supervised loss weights are frozen before the first large shard:

| Loss | Weight |
|---|---:|
| action-Q Huber | 1.00 |
| baseline-delta Huber | 1.00 |
| teacher-policy KL | 0.25 |
| state-value Huber | 0.25 |
| pairwise/listwise ranking | 0.50 |
| uncertainty quantile | 0.20 |
| safe-action BCE | 0.50 |

Cycle-2 V-trace weight is selected on the preregistered pilot only and then
frozen. It cannot be tuned on threshold-lock or diagnostic holdout.

The full-hand opponent mixture is:

| Opponent source | Share |
|---|---:|
| current checkpoint | 35% |
| historical checkpoints | 25% |
| exploitative responses | 15% |
| conservative policies | 10% |
| random/off-policy policies | 10% |
| legacy rollback chain | 5% |

Common street-promotion Go requires:

- immutable split/seed/model manifests and 100% score, visibility, legal-mask,
  ActionKey, scalar/batch, and replay parity on the applicable audit sample;
- at least 300 valid changed actions and at least 100 in each seat;
- gain/change and full-hand EV CI95 lower bounds above zero overall, first
  seat, and second seat;
- false-positive rate <= 0.30 overall and <= 0.35 per seat;
- changed-action loss p95 / p99 / max <= 25 / 40 / 50 points;
- every opponent-family mean >= -0.005 point/hand and CI95 lower bound
  >= -0.02 point/hand;
- non-fire counterfactual trajectory mismatch exactly zero;
- learned ABR worst-response result >= -0.01 point/hand and no regression in
  the multi-family ABR gap.

Failure of either seat, any opponent family, any hidden-information check, or
any tail gate is a stage No-Go even if aggregate EV is positive.

### M3.2 - T2 curriculum

- full-hand population self-play supplies balanced T2 roots;
- T3/T4 remain accepted and frozen;
- enumerate all legal T2 actions;
- use accepted continuation search and disjoint candidate/evaluation RNG;
- run two Expert Iteration cycles;
- add low-weight off-policy V-trace in cycle 2;
- start with a 100-1,000 state pilot before the planned 18,000 roots.

T2 Go additionally requires full legal-action enumeration, accepted T3/T4
continuation parity, and the common street-promotion gates above. No-Go keeps
`stage9f_p2` as the rollback baseline and does not unblock T1.

Estimated calendar time: 4-8 days.

### M4R - T1 curriculum

- full-hand trajectories with accepted T2-T4 continuation;
- both first and second seat are rebuilt;
- roots, valid changes, calibration, EV, false positives, and tails are
  reported separately for first and second seat;
- historical T1 attempts remain architecture evidence only;
- use new thresholds, seeds, contract, and population holdout;
- planned scale: 22,000 roots after pilot Go.

T1 Go requires both seats to pass the common street-promotion gates against
the accepted T1 baseline. A positive first-seat result cannot compensate for a
second-seat failure. No-Go preserves `stage18_p1` and does not unblock T0.

T1 second-seat completion order:

1. reserve a 50/50 first/second root quota and a second-seat calibration split;
2. construct the T1 second-seat observation from the opponent's public board,
   hero board/hand/private discard, order, and belief only;
3. enumerate every legal ActionKey and label it with common-future accepted
   T2-T4 continuation search;
4. train the shared model, then fit seat calibration without changing core
   weights;
5. if the preregistered gradient/interference gate fails, add a small
   second-seat adapter rather than silently tuning the shared backbone;
6. lock thresholds once and run fresh paired seat-swap population/ABR
   evaluation with at least 100 valid second-seat changes.

Estimated calendar time: 5-10 days.

### M5 - T0 curriculum

- both seats rebuilt against accepted T1-T4 continuation;
- roots, valid changes, calibration, EV, false positives, and tails are
  reported separately for first and second seat;
- evaluate all 232 actions coarsely;
- search top 24 plus the baseline;
- apply high-precision MC to the top 8;
- planned scale: 34,000 roots after pilot Go.

T0 Go requires both seats to pass the common street-promotion gates, plus zero
candidate/evaluation RNG overlap and a locked audit showing that top-24/top-8
pruning does not introduce an unacceptable regret tail. No-Go preserves
`stage19_p0` and blocks joint unfreezing.

T0 second-seat completion order:

1. reserve a 50/50 first/second root quota and never mix the first-seat empty
   opponent board with the second-seat visible opponent T0 board;
2. enumerate all 232 T0 ActionKeys and mask only genuinely illegal actions;
3. score all actions coarsely, include the rollback baseline unconditionally,
   search top 24 plus baseline, then evaluate top 8 at high precision;
4. use common random futures inside each comparison and independent futures
   for final locked evaluation;
5. calibrate second-seat uncertainty/safety separately and apply the same
   interference rule as T1;
6. run fresh paired seat-swap population/ABR evaluation with at least 100
   valid second-seat changes and require both seat CI lower bounds above zero.

Estimated calendar time: 7-14 days.

### M5F - Explicit 14-card FL

The RL environment contains 14-card FL mode from the start. Initial normal-hand
training may bootstrap entry with the frozen FL EV
`10.227020614683454`. Before joint promotion, entry and stay must become
explicit 14-card transitions with frozen-solver parity.

Estimated calendar time: 2-5 days.

### M6 - Joint full-hand league iteration

All learned street components are jointly unfrozen. T4 exact remains available
as the terminal search/runtime component.

Role allocation:

- search teacher: high-quality Q, delta, and hard negatives;
- Expert Iteration: primary policy/value/Q learning;
- V-trace actor-critic: terminal-result fine-tuning;
- population league: robustness against current, historical, exploitative,
  conservative, random/off-policy, and legacy rollback opponents;
- ReBeL-style public-history belief: hidden-card belief representation;
- restricted MCCFR: late-street public-belief subgame improvement;
- learned ABR: independent exploitability proxy.

Do not start with pure terminal-reward PPO/DQN or full-game Deep CFR.

M6 runs at most three league iterations. Go requires the learned multi-family
ABR-gap 95% upper bound to improve by at least 20% versus M5, an absolute
empirical proxy <= 0.10 point/hand, and non-regression in population
worst-case, both seats, and loss tails. Two consecutive plateaus are a No-Go
closeout, not permission to tune on the same locked evaluation.

Estimated calendar time: 2-4 weeks.

### M7 - Empirical low-exploitability promotion

Go:

- fresh paired seat-swap results are positive overall and separately by seat;
- population worst-case and override-loss tails do not regress;
- measured multi-family ABR gap improves by at least 20% from the preceding
  accepted policy;
- target empirical proxy <= 0.10 point/hand;
- improvement occurs within three league iterations;
- two consecutive plateaus close the experiment No-Go;
- release is a new explicit opt-in profile.

An ABR confidence bound is not a mathematical NashConv upper bound.

## 5. Common stop rules

- No multi-VM fanout without a completed lifecycle canary.
- Two consecutive failures at the same pre-content layer stop cloud launches
  until that layer is redesigned and proven locally.
- Stop fanout if pilot VM time exceeds the forecast by more than 2x.
- Any integrity, hidden-information, ActionKey, unknown-field, or RNG violation
  makes the entire affected dataset No-Go.
- Do not modify thresholds, add seeds, or rerun a failed locked holdout.
- Teacher EV and top-1 accuracy are diagnostic only.
- Do not hide an opponent-family or seat regression behind overall mean EV.
- Do not automatically change `current`, delete a rollback baseline, or enable
  a full replacement.

## 6. Time and compute boundary

If gates pass on their first frozen attempt, the complete route is estimated at
7-12 weeks. A realistic research range allowing redesign is 10-16 weeks.

Coding can finish substantially faster than these ranges. The irreducible
calendar work is independent search generation, fresh holdout evaluation,
population play, and independently trained ABRs.

Local development assumes the existing RTX 2060 SUPER 8 GB, 64 GB RAM, and
8-core / 16-thread CPU. Model training, calibration, replay validation, and
small pilots stay local. Native search and large self-play generation use
restart-safe `c4-standard-16` Spot shards only after their local gate passes.

The 10c.2 diagnostic contract uses a conservative
`$0.57 per c4-standard-16 VM-hour` ceiling while retaining its `$4.00` phase
cap. This is not a promise of the live price or a blanket price assumption for
later phases. Google states that Spot prices are variable and may change
daily, so the current regional SKU must be checked immediately before every
authorization:
[Google Cloud Spot VM pricing](https://cloud.google.com/spot-vms/pricing?hl=en).

| Phase | Calendar | Local GPU estimate | Spot VM-hours planning range | Cloud budget range |
|---|---:|---:|---:|---:|
| M3.1 remaining T3 | 3-6 days | 8-24 h | 20-1,000 | hard cap `$500` |
| RLA-RLE RL-ready | 5-10 days | 2-8 h | 0-200 | `$20-100` |
| M3.2 T2 | 4-8 days | 12-36 h | 500-1,400 | `$250-700` |
| M4R T1 | 5-10 days | 18-60 h | 1,200-3,000 | `$600-1,500` |
| M5 T0 | 7-14 days | 24-96 h | 1,800-5,000 | `$900-2,500` |
| M5F FL | 2-5 days | 4-16 h | 200-1,000 | `$100-500` |
| M6/M7 joint league/ABR | 2-4 weeks | 80-300 h | 4,000-12,000 | `$2,000-6,000` |

These are order-of-magnitude planning ranges, not preauthorization. The
rearm2 package currently freezes a much smaller immediate guard: initial
20-job estimate `$11.67`, all attempt-0/attempt-1 jobs `$23.33`, and phase cap
`$25`. Later phase budgets are frozen separately only after their 100-1,000
state pilot. The whole research program is expected to fall near
`$4,000-12,000` if all stages are pursued; no later-stage amount is currently
authorized.

R0a and the local I1 package gate are now complete. The rearm2 cloud lifecycle
remains off pending bounded canaries; unrelated project workloads are outside
this statement. A future rearm2 Spot fanout requires:

1. local actual-package and lifecycle Go;
2. a measured one-VM lifecycle smoke and two-VM paired canary;
3. a frozen cost cap;
4. explicit launch authorization.

## 7. Immediate execution sequence

The write-once claim, roots, package, and smoke receipt already exist and must
not be recreated. The safe read-only verification commands are:

```powershell
git status --short --branch
git rev-parse HEAD
Get-FileHash -Algorithm SHA256 src/ofc_regular/ai_profiles.py

$env:PYTHONPATH = "src"
$package = "D:\ofc-gcp-runs\regular-hu-m31-c02-performance-lock-rearm2-20260718-001\package"

python -m ofc_regular.hu_m31_t3_step6d_performance_lock_rearm2_spot `
  validate-package --run-dir $package

python -m ofc_regular.hu_m31_t3_step6d_performance_lock_rearm2_spot `
  validate-preauthorize-smoke-receipt `
  --run-dir $package `
  --receipt "$package\ACTUAL_PACKAGE_PREAUTHORIZE_SMOKE.json"
```

Next:

1. [x] build an immutable worker payload containing the three exact
   runner-compatible shard manifests, Python allowlist, accepted candidate and
   reference native libraries, feature encoder, and a diagnostic-only Linux
   startup;
2. [x] implement the integrated fake remote execution adapter and prove
   generation-match writes, prefix and known-object collision rejection,
   claim/auth separation, interruption/recovery, `DONE` readback, success
   self-delete, bounded failure shutdown, and controller receipt binding;
3. [x] implement a separate direct GCE bootstrap/download/upload transport and
   add only read-only real-cloud preflight for prefix, quota, and current Spot
   price; keep `launch_ready=false` until it passes;
4. [x] freeze the `$4.00` diagnostic phase guard; [ ] keep an explicit one-VM
   authorization as a separate change;
5. [x] build and locally validate the fix3 content-addressed outer package
   containing the import-closure, scope, and worker-venv validation
   corrections; retain the old package unchanged;
6. [x] obtain fresh explicit authorization and rerun exactly one corrected
   candidate attempt-0 VM; v10 proved insert/claim/revocation/cleanup but hit
   the missing-object collector edge, and v11 proved the collector correction
   but reproduced the bootstrap import-closure failure; fix3 completed all 10
   diagnostic hands and self-deleted;
7. [ ] if and only if its receipt passes, separately authorize and run exactly
   one candidate/reference pair. The first separately authorized Step 12
   command stopped before Compute insert because the 39-second controller-token
   propagation barrier expired. The later direct-v2 command minted the token
   and removed Token Creator, but stopped before Phase2 because an empty IAM
   policy omitted `bindings` and exposed a normalization mismatch. Both
   executions closed with exact IAM/VM/disk/result zero evidence. The parser
   correction and full regression now pass locally, but a new canary still
   requires fresh explicit authorization and a wholly new identity;
8. inspect runtime, RSS, heartbeat, checkpoint, self-delete, and byte-identical
   receive/resume evidence;
9. only after those gates pass, freeze the one-shot 20-VM authorization.

In parallel, R0b must either recover the exact pinned M3.0 DLL or validate a
new candidate as a new runtime version. No hash in the existing M3.0 contract
may be silently replaced.

## 8. Planned implementation map

Existing rollback/runtime components remain in place:

- `src/ofc_regular/ai_profiles.py`;
- `src/ofc_regular/policy.py`;
- `src/ofc_regular/hu_turn2_stage8_runtime.py`;
- `src/ofc_regular/hu_turn3_model.py`;
- `rust/hu_m3_engine`;
- the current exact-T4 implementation and every named baseline artifact.

T3/I1 control modules remain versioned rather than replacing older lifecycles:

- `src/ofc_regular/hu_m31_t3_step6d_performance_lock_rearm2_open.py`;
- `src/ofc_regular/hu_m31_t3_step6d_performance_lock_rearm2_spot.py`;
- `src/ofc_regular/verify_hu_m31_t3_feature_encoder_platform_parity_rearm2.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_plan.py`;
- `configs/hu_joint_policy_m31_t3_step6d_rearm2_diagnostic_canary_v1.json`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_plan.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_local.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_local.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package.py`;
- `configs/hu_m31_t3_step6d_rearm2_diagnostic_runtime_requirements_v1.txt`;
- `scripts/startup_hu_m31_t3_step6d_rearm2_diagnostic_canary_v1.sh`;
- `scripts/verify_hu_m31_t3_step6d_rearm2_diagnostic_canary_v1.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_fake_remote.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_fake_remote.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1.py`;
- `scripts/bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_v1.sh`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_receiver_preflight.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_receiver_preflight.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1.py`;
- `scripts/bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh`;
- `scripts/prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1.py`;
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1.py`;
- `scripts/prepare_hu_m31_t3_step6d_step11_v12_local_v1.py`;
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1.py`.

The diagnostic canary files above freeze the plan and complete the offline
package/startup/receive/resume lifecycle, the accepted solver/native worker
payload, the direct GCE metadata/download/upload implementation, generation-
bound atomic receive, and the real GET-only preflight collector. The fresh
actual v4 GET-only preflight passed and is frozen by
`outputs/hu_joint_policy/m31_t3_step6d/rearm2_diagnostic_10c2_actual_readonly_preflight_v4_inventory_fix1.json`
at embedded artifact SHA-256
`14f6f34c2aabf00640b369594a4fa09338ebd9d18ebbd99cc5e6f559fd4e61d9`.
This closes 10c.2 read-only eligibility only. Later Step 11 v10 and v11
diagnostic-only attempts separately proved package provisioning,
insert/provider identity, asymmetric authorization/claim, post-claim
launch/actAs revocation, and exact cleanup. Neither attempt produced a
heartbeat, validated `DONE` tree, successful self-delete receipt, performance
evidence, or scientific result. The old content-addressed package remains
immutable. The corrected fix3 outer identity and fresh local preflight are now
frozen, but remain explicitly local-only: package upload, live readback,
authorization, claim, IAM change, and VM creation are all false. A fresh
explicit one-VM authorization must bind that exact fix3 identity before any
new upload or launch. The frozen all-20 production launch path remains
untouched.

RL-ready additions are planned as:

- `rust/hu_rl_engine/` for private world state, legal actions, step, batch
  rollout, search labels, snapshot/restore, and PyO3 bindings;
- `src/ofc_regular/hu_rl_contract.py` for versioned observation, ActionKey,
  replay, audit-token, and RNG schemas;
- `src/ofc_regular/hu_rl_reference.py` for the slow Python correctness oracle;
- `src/ofc_regular/street_policy_net_v1.py` for the shared
  state/action/policy/value/Q/delta/uncertainty/safety model;
- `src/ofc_regular/hu_search_teacher_v1.py` for street-dependent search and
  independent selection/evaluation namespaces;
- `src/ofc_regular/hu_expert_iteration_v1.py` for dataset roles, losses,
  checkpoints, and V-trace cycle 2;
- `src/ofc_regular/hu_population_league_v1.py` for fixed opponent/seat quotas;
- `src/ofc_regular/hu_abr_v1.py` for learned best-response evaluation;
- milestone-specific immutable JSON contracts under `configs/`;
- compressed immutable replay/search shards under `outputs/`, never imported
  as policy source code.

Each new module gets focused tests for canonical serialization, tamper
rejection, hidden-field rejection, ActionKey order/mask parity, deterministic
resume, scalar/batch parity, non-fire cancellation, seat quotas, and
write-once artifacts. No module is wired into `current` until M7 explicitly
promotes a new opt-in profile.
