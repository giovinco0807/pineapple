# HU joint-policy implementation milestones

Status date: 2026-07-16

This plan preserves `stage19_p0`, `stage18_p1`, `stage9f_p2`, and
`stage7_m5_r10` as explicit baselines. It does not change the `current` profile,
and no new policy becomes a full replacement before locked holdout evaluation.

## 2026-07-16 audit revision

The fixed chain is operational, but it is a legacy comparison chain rather
than a hidden-discard-safe promoted joint policy. The fixed T2 Stage9f model,
the T3 Stage7 model, and the Stage7 reference were trained from legacy feature
caches that encoded realized opponent private discards. P0 and P1 do not have
that direct feature leak at their decision streets, but their labels, selector
targets, and promotion evaluations used the quarantined downstream chain.

M2 and M3 repaired the late-street teacher and native search engine; they did
not rebuild or promote T3 or T2 model artifacts. M4 through M4.3 Attempt13 used
`stage9f_p2` as the fixed T2 continuation. Those experiments remain useful as
search-architecture and fail-closed evaluation evidence, but cannot certify the
final safe joint policy. Attempt13 closed `complete_no_go_development`: 16 of 18
gates passed, while fired-root p95 loss and maximum loss failed. It did not open
Audit50, fit a model, activate runtime, or run population acceptance.

The revised dependency order is:

```text
R0 release boundary
  -> M3.0 both-seat exact T4 runtime closure
  -> M3.1 safe T3 rebuild
  -> M3.2 safe T2 rebuild and promotion
  -> M4R T1 both-seat rebuild
  -> M5 T0 both-seat rebuild
  -> M6 joint population iteration and exploitability reduction
```

No later milestone may cite old teacher EV, placement accuracy, or a legacy
profile's historical Go result as fresh promotion evidence.

Machine-readable supersession status:
`configs/hu_joint_policy_revised_roadmap_20260716.json`. Historical M3/M4/M4.1/
M4.2/Attempt13 status and closeout artifacts remain immutable evidence and are
not rewritten by this plan revision.

## Milestone 0 - Baseline freeze and reproducibility

Status: complete (2026-07-12)

Deliverables:

- A run manifest containing the Git commit, dirty-tree fingerprints, explicit
  profile chain, command, seed roles, `seed_stride`, and SHA-256 hashes for every
  explicitly supplied config, model, input, or binary.
- Atomic manifest writes with no overwrite by default.
- A local correctness manifest and a full regression-test result.

Go when the manifest round-trip tests pass, the fixed chain is explicit, and
the existing test suite remains green. No-Go if a run can silently depend on
`current` or an unhashed required artifact.

Evidence:

- `outputs/hu_joint_policy/m0_baseline/run_manifest.json`
- 18 explicitly selected artifacts hashed; fixed profile chain only
- The M0 snapshot remains immutable; final M1 validation is recorded separately

## Milestone 1 - Information-set and deterministic action foundations

Status: complete (2026-07-12) for foundations. Policy promotion and large-scale
generation remain No-Go pending M2 and safe artifact rebuilding.

Deliverables:

- Separate full simulator `WorldState` from the policy-facing
  `ActorObservation`.
- Exclude opponent private discards by construction.
- Introduce an order-independent `ActionKey`, stable legal-action sorting, and
  checked index/key serialization.
- Introduce counter-derived RNG keys independent of candidate enumeration order.

Go when visibility, card-permutation, scalar/batch mapping, and repeated-run
determinism tests all pass. No-Go on any hidden-discard exposure or action-index
drift.

Implementation:

- Added fixed-width four-mask `ActionKey` tokens, key-first resolution, and
  separate legal-set/order digests without changing legacy action enumeration.
- Verified canonical action ordering across all dealt-card permutations at
  T0-T4. T1 selector training now rejects key/payload/index/digest conflicts
  instead of silently accepting a saved positional index.
- Added immutable `WorldState -> ActorObservation`; the observation has no
  opponent-discard identity or true deck tail.
- Added post-decision-only `ReplayTruth`; mutable policy metadata rejects replay,
  world, card, and observation fields. Routed normal `play_hand` and matchup
  tracing through the observation adapter while preserving legacy baselines.
- Added exchangeable hidden-card particles rooted only in ActorObservation.
  T1 and T2 teacher APIs require both an observation and a validated belief
  batch; realized opponent discards and the realized deck tail cannot condition
  their labels.
- Changed the T2 feature cache to rebuild the model sample from the actor
  observation. Added the same fail-closed boundary to the T3 feature cache. Raw
  replay `dead_cards` and opponent private discards are not forwarded to either
  encoder.
- Added semantic counter RNG keys and removed `action_index` from downstream T1
  teacher policy seeds, so candidates share decision randomness. Runtime seeds
  are domain-separated by actor, street, decision ordinal, and observation
  fingerprint.
- Added action keys and mapping digests to new T0/T1/T2/T3/T4 decision and
  teacher records. Cache hits remap semantic actions into the current legal
  order.
- Non-fired decisions cancel only when a same-snapshot shadow run reproduces the
  entire trajectory and the final and baseline semantic ActionKeys agree.

Completion evidence:

- Full suite: `876 passed in 26.28s`
- Focused M1 suite: `188 passed in 10.46s`
- T3 scalar/batch parity and full T2 rollout parity cover both seats
- T4 exact search is checked against independent exhaustive terminal scoring on
  deterministic randomized states
- T2 values and all relevant digests are invariant across all six permutations
  of the dealt cards
- `outputs/hu_joint_policy/m1_complete/validation_summary.json`
- `outputs/hu_joint_policy/m1_complete/run_manifest.json`
- `configs/hu_joint_policy_m1_quarantine.json`

Artifact audit and quarantine:

- The first 100 rows of the fixed T2 model's source contained 50 two-discard and
  50 three-discard legacy `dead_cards` rows; all 100 lacked
  `hero_private_discards`, `policy_observation`, and `replay_truth`.
- The first 100 rows of the fixed T3 Stage7 source contained four or five
  private discard identities beyond the opponent public board; all 100 lacked
  the safe split and versioned observation.
- Therefore the fixed P0/P1/P2/T3 chain is retained only as an explicit legacy
  baseline. Its old EV and acceptance evidence, including the T0 100,000-paired
  result, is not valid hidden-discard promotion evidence.
- Legacy generators/trainers and the unfinished joint-exact T3 teacher are
  listed in the machine-readable quarantine registry. They cannot be cited as
  M1-safe data sources.
- All 16 fixed config/model hashes still match M0. `ai_profiles.py` differs only
  by the M1 card-free metadata guard; removing that guard in memory reproduces
  the M0 registry hash exactly. The `current` mapping was not changed.

## Milestone 2 - Correct late-street teacher

Status: complete (2026-07-13) as a Python correctness reference. Runtime
promotion and large-scale generation remain No-Go until M3 parity and pilot.

Deliverables:

- Keep second-seat T4 exhaustive evaluation.
- Replace first-seat T4's self-board shortcut with sequential opponent response
  and hidden-card belief evaluation.
- Use exact future enumeration at T3 where tractable and common-random Monte
  Carlo otherwise.
- Keep candidate-selection samples disjoint from reported evaluation samples.

Go on brute-force toy parity, scalar/batch parity, non-fire trajectory identity,
and unbiased fresh-state checks. No-Go if first-seat values omit the opponent's
future deal/action or use private opponent discards.

Completion evidence:

- T4-first exact uniform marginal: all 2,024 opponent deals and all exact
  responses; fixed shortcut counterexample changes the selected ActionKey.
- T4-second exhaustive terminal parity.
- Exact declared finite-support T3 oracle and live event-order oracle for both
  seats; normal full-deck T3 is explicitly CRN Monte Carlo.
- Separate candidate/evaluation RNG domains and seeds, strategy-fusion guard,
  scalar/batch parity, and hidden-truth invariance.
- T2 scalar/batch continuation can opt into the safe selector from its CLI and
  shard runners without changing existing defaults or `current`.
- Full suite `921 passed`; focused M2 suite `181 passed`.
- `docs/hu_joint_policy_m2_completion_audit.md`
- `outputs/hu_joint_policy/m2_complete/validation_summary.json`

## Milestone 3 - Rust HU rollout and search engine

Status: complete (2026-07-13) as the native correctness/search engine. Runtime
policy promotion remains No-Go. The 2026-07-16 audit inserts R0, M3.1, and M3.2
before any new T1 promotion attempt.

Deliverables:

- Rust state, action, scoring, information-set, belief, RNG, T4, T3, rollout,
  batch, and Python-binding modules.
- Resumable small shards with atomic checkpoints and heartbeat output.
- 100-1,000 state pilot after correctness and deterministic parity.

Go only if Python/Rust decisions and values meet fixed tolerances and profiling
shows enough speedup to justify expansion. No-Go triggers redesign before any
Spot-scale generation.

Completion evidence:

- Rust cards/state/action/ActionKey/scoring/infoset/belief/RNG/T4/T3/batch/FFI
  modules, plus exact declared-support T3 and restart-safe shard runner.
- T4 first full 2,024-deal parity, T4 second exhaustive parity, and T3 parity
  for both seats; 9 Python/Rust focused tests.
- Full Python regression: 940 passed. Also 36 Rust library tests, 2 runner CLI
  tests, 10 pilot CLI tests, and Clippy with warnings denied.
- Release benchmark on four fixed fresh roots: 8.5879x aggregate speedup with
  every ActionKey and per-action value matching Python.
- Gated local pilots: 100 roots (50/50 seats) in 6.291 s, then 1,000 roots
  (500/500 seats) in 71.123 s. All RNG separation, deterministic rerun,
  checkpoint, heartbeat, unique-root, and result-mapping gates passed.
- `docs/hu_joint_policy_m3_completion_audit.md`
- `outputs/hu_joint_policy/m3_complete/validation_summary.json`

## Prerequisite R0 - Safe release and provenance boundary

Status: next. This phase changes no profile selection and activates no model.

Deliverables:

- Separate the local M1-M4 source changes from unrelated working-tree changes,
  preserve every legacy model/profile, and record a fresh manifest.
- Make `ActorObservation` the enforced card-bearing runtime API. Compatibility
  entrypoints accepting arbitrary `dead_cards` must either be private or reject
  calls that cannot prove actor-visible provenance.
- Complete the quarantine registry so it includes the Stage3 T3 reference and
  the P0/P1 selector artifacts, not only the four primary candidate models.
- Re-run full Python tests, M3 Rust tests, package fmt/clippy, profile smoke, and
  the non-fire full-trajectory cancellation test.
- After explicit release authorization, commit and push the safe boundary to
  `codex/regular-ofc-pineapple`; verify the remote branch no longer contains the
  global cross-player `dead_cards` runtime path.

Go only when the source scope is reviewable, the policy-registry/current hash is
unchanged except for already frozen safety guards, all required tests pass, and
local and upstream information-set boundaries agree. No-Go on any hidden truth,
unhashed required artifact, unrelated deletion, implicit `current`, or profile
activation.

## Milestone 3.0 - Both-seat exact T4 runtime closure

Status: complete locally (2026-07-16) as an explicit opt-in component. Source
release and any named-profile activation remain gated by R0; `legacy` remains
the default and `current` is unchanged.

Decision semantics:

- First seat evaluates every legal hero action as
  `max_a E_uniform-deal[min_b terminal HU score]`. The declared restart belief
  has 24 unknown cards, all 2,024 three-card opponent deals, and every legal
  exact opponent response for each deal.
- Second seat evaluates every legal terminal placement directly against the
  completed opponent board.
- Both paths return the selected semantic `ActionKey`, selected EV, and EV for
  every legal action. Illegal actions are never passed to the native engine.
- This is exact under the declared uniform exchangeable information-set belief.
  It is not a full Bayesian posterior, a Nash-equilibrium proof, or a proof of
  mathematical full-game optimality.

Runtime boundary:

- The policy API accepts `ActorObservation` only. Python, scalar Rust, and
  batch Rust reject unknown fields and hidden-truth aliases such as opponent
  private discards, the realized remaining deck, and replay/world state.
- The prebuilt release library, engine version, and SHA-256 must match before
  play. There is no runtime build and no silent fallback to legacy T4.
- Activation is compositional and explicit through `--t4-mode-a m30_exact` or
  `--t4-mode-b m30_exact`. No existing named profile, fixed profile, or
  `current` mapping was edited.

Local Go evidence:

- Rust and Python exact parity, scalar/batch parity, dealt-card permutation
  invariance, deterministic reruns, strict-schema injection tests, and
  fail-closed native corruption/version/hash tests pass.
- The disjoint 100-root and 1,000-root pilots passed every frozen gate. In the
  1,000-root pilot, first-seat p95/p99 latency was `42.91/54.53 ms`, second-seat
  p99 was `1.70 ms`, and native batch throughput was `156.04 roots/s`.
- A fresh 100-paired off-policy counterfactual run changed 11 first-seat T4
  actions and produced `+0.295 EV/hand`, 95% CI `[+0.1204, +0.4696]`. Realized
  gain was `+5.36` points per override, with 0 false positives, zero p95/p99/max
  tail loss, and exact full-trajectory cancellation on all 89 non-fired pairs.
- The fixed `stage19_p0` chain remains a quarantined distribution sampler, not
  safe joint-policy promotion evidence. Its independent 100-paired smoke had a
  positive `+0.065 EV/hand` point estimate but a 95% CI crossing zero.

Evidence:

- `docs/hu_joint_policy_m30_t4_completion_audit.md`
- `configs/hu_joint_policy_m30_t4_runtime.json`
- `outputs/hu_joint_policy/m30_t4_complete/pilot_100.json`
- `outputs/hu_joint_policy/m30_t4_complete/pilot_1000.json`
- `outputs/hu_joint_policy/m30_t4_complete/paired_random100_v3_summary.json`
- `outputs/hu_joint_policy/m30_t4_complete/paired_stage19_100_summary.json`

## Milestone 3.1 - Hidden-discard-safe T3 artifact rebuild

Status: in progress after a completed Step 6c No-Go. The strict runtime
boundary, Step 2 correctness matrix, general-board profiling, Step 3 local
100-root gate, Step 4 local 1,000-root gate, Step 5 numeric gate/seed freeze,
Step 6a immutable Linux Spot shard-0, Step 6b merged 500-root infrastructure
canary, and the fixed Step 6c 100-root production-label quality experiment are
complete. Step 6c passed every confirmation-regret and integrity gate but
failed first-seat primary p95 latency (`326.2574 s > 180 s`). Artifact rebuild,
strength evaluation, and promotion remain No-Go pending a separately frozen
performance repair. See
`docs/hu_joint_policy_m31_t3_step2_correctness_audit.md`,
`docs/hu_joint_policy_m31_t3_step3_completion_audit.md`, and
`docs/hu_joint_policy_m31_t3_step4_completion_audit.md`, and
`docs/hu_joint_policy_m31_t3_step5_completion_audit.md`, and
`docs/hu_joint_policy_m31_t3_step6a_completion_audit.md`, and
`docs/hu_joint_policy_m31_t3_step6b_completion_audit.md`, and
`docs/hu_joint_policy_m31_t3_step6c_completion_audit.md`. Step 6b combined the
accepted Step 6a shard with nine restart-tested Spot shards and passed all
frozen integrity/performance gates across 250 paired hands / 500 roots. The
rows remain ineligible for training. Step 6c then evaluated 50 disjoint paired
hands with `8/32/4/0` labels and a frozen ten-root `8/128/4/0` confirmation
subset. Confirmation mean/p95/p99/max regret all passed, but the frozen
first-seat performance gate failed. The same data may not be reseeded,
extended, or threshold-selected to reverse that result. The next boundary is a
versioned first-seat performance repair before any new disjoint quality pilot.
This remains the first model-building milestone in the revised plan because
T2, T1, and T0 all depend on T3 continuation quality.

Deliverables:

- Generate new versioned T3 rows only from `ActorObservation`, with explicit
  hero-private/public visibility, `ActionKey`, legal-set and ordered-mapping
  digests, source hashes, seat, opponent policy, and RNG provenance.
- Build both a new T3 reference and a new selective candidate. Legacy Stage3
  and Stage7 weights may be opponents or rollback comparators, but cannot be
  label sources, safety targets, or promotion evidence.
- Use M3 Rust T3 search and the accepted M3.0 exact T4 runtime. Candidate selection and
  locked evaluation use disjoint counter-RNG domains and common random futures
  within each candidate comparison.
- Balance first/second seats and include historical, conservative,
  exploitative, and randomized behavior policies in state generation.
- Keep the new profile explicit and opt-in; preserve `stage7_m5_r10` unchanged.

Validation ladder:

1. schema/visibility/action-mapping smoke;
2. determinism and scalar/batch/exact parity;
3. local 100-root pilot;
4. local 1,000-root pilot and profiling;
5. freeze numeric acceptance gates and non-overlapping seed ranges;
6. only then authorize small resumable Spot shards;
7. fresh paired seat-swap multi-opponent holdout.

Go requires zero integrity violations, exact non-fire cancellation, positive
locked realized gain per override, positive overall paired EV/hand LCB, no
material first/second regression, and frozen p95/p99/max-loss and false-positive
gates. Teacher values and model placement accuracy remain diagnostic. No-Go
returns to local diagnosis without threshold search on the holdout.

## Milestone 3.2 - Hidden-discard-safe T2 artifact rebuild and promotion

Status: pending M3.1 Go.

Deliverables:

- Regenerate all T2 feature caches from `ActorObservation`; raw legacy
  `dead_cards` rows fail closed rather than being adapted heuristically.
- Use the accepted M3.1 T3 continuation and M3.0 exact T4 runtime in both
  scalar and batched generation paths.
- Enumerate every legal T2 action, rank by semantic `ActionKey`, and train a
  shared seat-aware action-value/delta/uncertainty/safety model with seat-specific
  calibration. Split seat models only if a preregistered interference test
  rejects sharing.
- Preserve `stage9f_p2` as the rollback baseline and publish any candidate under
  a new explicit opt-in profile.

Use the same validation ladder and integrity gates as M3.1. Promotion also
requires both-seat fire coverage, positive first/second realized results on
fresh seeds, robustness across opponent families, bounded tail loss, and no
holdout threshold reselection. M3.2 must finish before T1 labels are rebuilt.

## Milestone 4R - T1 both-seat policy rebuild

Status: historical M4, M4.1, M4.2, and M4.3 attempts are complete No-Go;
promotion remains blocked pending M3.2 Go. The next T1 attempt must use a new
identifier, new disjoint seeds, and a separately frozen contract.

Deliverables:

- Relabel balanced second-seat T1 states with the corrected teacher.
- Relabel first-seat T1 as well; old Stage18 first-seat labels and holdouts are
  diagnostic because their continuations were quarantined.
- Train a shared seat-aware policy/value/action-delta/safety model, retaining
  a selective baseline fallback.
- Lock thresholds before fresh paired seat-swap evaluation against a policy
  population.

Go requires positive paired EV/hand confidence, positive realized gain per
override, bounded p95/p99 loss, acceptable false-positive rate, and no material
population worst-case regression. Teacher EV and top-1 accuracy are diagnostic
only.

Audit correction:

- Existing M4 teachers enforced an observation-only T2 call and used M3 Rust at
  T3/T4, but the fixed T2 policy was `stage9f_p2`, whose weights remain in M1
  quarantine. Existing M4 evidence therefore measures behavior relative to the
  legacy continuation and is architecture evidence only.
- Attempt13 Development200 fired 44 times and had positive mean E512 deltas,
  but failed fired-root p95 loss (`29.227 > 25`) and maximum loss
  (`57.454 > 50`). E512 is not realized match EV. Audit50, fit, population
  acceptance, runtime activation, and profile promotion did not occur.
- M4R must not reuse Attempt13 thresholds or evaluation seeds. Architecture
  diagnosis may inform a new frozen proposal only after M3.2 is accepted.

Historical M4 implementation evidence:

- T1-second teacher with an observation-safe decision boundary, independent
  candidate/evaluation beliefs, and M3 Rust T3/T4 continuation. Because its T2
  continuation used quarantined Stage9f weights, it is not end-to-end
  promotion-safe evidence.
- Restart-safe shard generation, strict data audit, shared seat-aware joint
  model, baseline-first runtime composition, fresh population evaluator, and
  fixed acceptance validator.
- 112 fresh roots: 64 train, 24 calibration, 24 locked holdout; zero seed or
  fingerprint overlap and data audit pass.
- Calibration No-Go: low-threshold false-positive rate 81.82% with negative
  diagnostic delta; higher thresholds produced zero fires. Safety was disabled.
- Fresh four-opponent fallback smoke: invalid/mismatch/nonzero/unknown all zero
  and first/second/paired deltas exactly zero.
- Acceptance: 17/23 gates passed; strength/realized-fire gates failed because
  the artifact correctly remained inactive.
- Full suite 993 passed; Rust 36+2 passed; Clippy passed.
- `docs/hu_joint_policy_m4_completion_audit.md`
- `outputs/hu_joint_policy/m4_complete/pilot112/acceptance_status.json`

M4.1 follow-up evidence:

- 100 fresh `c2/e4` roots: 50 train, 30 calibration, 20 locked holdout;
  five root profiles contributed exactly 20 roots each.
- Batched M3 Rust child selectors made the local pilot feasible; the slowest of
  four parallel shards completed in 3,111.051 seconds.
- The 30-root calibration split into independent 15/15 safety-fit and
  threshold-lock subsets with zero seed, fingerprint, or row-hash overlap.
- Calibration remained No-Go: thresholds through 0.3 had negative mean
  diagnostic delta and 60% false positives; 0.4 and above had zero fires.
- Safety stayed disabled at threshold 1.0; fresh fallback smoke cancelled
  exactly; acceptance finished `complete_no_go` with 18/24 gates passed.
- `docs/hu_joint_policy_m41_completion_audit.md`
- `outputs/hu_joint_policy/m41_complete/pilot100/acceptance_status.json`

M4.2 follow-up evidence:

- 40 fresh `c2/e8` roots generated on resumable Spot shards: 20 train, 10
  calibration, and 10 locked holdout; all five root profiles contributed
  exactly 8 roots and all paired common-future delta contracts passed.
- The opt-in negative-regret ranker used strict identity-group nested 5-fold
  cross-fitting. Every base/meta/uncertainty OOF prediction occurred exactly
  once, predictor-lineage leakage was zero, and the locked holdout was never
  used for training or threshold selection.
- The bounded quality gate remained No-Go: all five threshold-lock candidates
  were non-positive, thresholds through 0.4 had diagnostic mean delta
  `-4.2670` per fire and 100% false positives, while 0.5 and above had no fires.
  The frozen threshold is 1.0 and safety is disabled.
- A fresh four-opponent inactive smoke had exact first/second/paired
  counterfactual cancellation, zero invalid/mismatch/nonzero/unknown records,
  and zero overlap between its evaluation seeds and all 40 teacher seeds.
- The acceptance validator passed 19 of 25 gates after adding strict nested-OOF
  and fresh-evaluation-seed provenance checks. Only the six realized-fire and
  strength gates failed; no 300-root expansion or runtime activation occurred.
- `docs/hu_joint_policy_m42_completion_audit.md`
- `outputs/hu_joint_policy/m42_complete/validation_summary.json`

## Milestone 5 - T0 both-seat policy rebuild

Status: pending M4R Go.

Deliverables and gates mirror M4R, using the accepted safe T1-T4 chain and all
232 legal opening actions. Both seats are relabeled: the old Stage19 first-seat
candidate may remain a proposal generator and rollback baseline, but its old
teacher, selector, and 100,000-paired acceptance evidence are not reusable for
promotion. T0 does not start until the T1 continuation is accepted for both
seats.

## Milestone 6 - Joint population iteration and exploitability reduction

Status: pending M5 Go.

Deliverables:

- Shared first/second joint policy with seat adapters if needed.
- Historical, exploitative, conservative, and randomized opponent population.
- Approximate best-response evaluation plus restricted late-street CFR.
- Repeated search-teacher / Expert Iteration cycles with frozen evaluation sets.

Promotion requires gains across seats and opponent families, stable tail risk,
and a decreasing approximate NashConv/ABR gap. This is a practical
low-exploitability target, not a mathematical proof of full-game optimality.

## Immediate R0 command sequence

Run read-only inventory and correctness checks before any additional model or
cloud work:

```powershell
git status --short --branch
git rev-parse HEAD
git rev-parse '@{upstream}'
Get-FileHash -Algorithm SHA256 src/ofc_regular/ai_profiles.py
rg -n "choose_action|dead_cards|choose_action_observation" src/ofc_regular

$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider `
  tests/test_hu_infoset.py `
  tests/test_action_key.py `
  tests/test_policy_play.py `
  tests/test_evaluate_matchups.py
python -m pytest tests -q

cargo test -p ofc_hu_m3_engine
cargo fmt -p ofc_hu_m3_engine -- --check
cargo clippy -p ofc_hu_m3_engine --all-targets -- -D warnings
gcloud compute instances list --project ofc-solver-485418
```

R0 performs no training and starts no VM. Any commit/push is a separate,
explicitly reviewed release action after the dirty-tree scope is partitioned.
