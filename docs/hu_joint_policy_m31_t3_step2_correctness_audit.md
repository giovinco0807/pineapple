# HU joint-policy M3.1 T3 Step 2 correctness audit

Status date: 2026-07-16

Decision: **Step 2 complete locally as a correctness boundary. M3.1 policy,
strength, 100-root pilot, model training, and profile promotion remain No-Go.**

Step 2 proves that the pinned Rust T3 component preserves semantic action
values across dealt-card permutations and scalar/batch execution, composes
with the accepted exact-T4 kernel, and agrees with the finite-support Python
oracle on non-zero both-seat fixtures. It does not claim exact full-deck T3,
realized match EV, Nash play, or mathematical full-game optimality.

## What changed

### Two deliberately different result digests

`hu_m31_t3_runtime_decision_v2` retains the original `result_digest` as an
ordered-mapping certificate. It includes the legacy original indices and
legal-action order digest, so it is expected to change when the same three
cards arrive in a different order.

The runtime now also emits
`hu_m31_t3_semantic_result_digest_v1`. Its payload contains the canonical
observation fingerprint, legal ActionKey set, selected ActionKey, all
ActionKey-sorted ranks and unrounded selection/evaluation/regret values,
belief and RNG provenance, exact-T4 contract, solver identity, engine version,
and pinned library hash. It excludes positional indices, action-order digest,
latency, and execution mode. Signed zero is normalized to `0.0`; no other
numeric rounding is allowed.

This separation is important: the semantic digest proves policy/value
invariance, while the mapping-bound digest and explicit index/key validation
continue to detect a corrupted positional action mapping.

### Finite-support public contract

The public Python-to-Rust finite-support request helper now fails closed unless
every declared world has the exact field schema, a finite strictly positive
weight, and total weight `1.0` within absolute tolerance `1e-12`. This matches
the Python exact-oracle domain. Card validity, hidden-card overlap, consumed
card counts, duplicate worlds, and unique non-empty world IDs remain enforced
by the native evaluator.

The accepted DLL was not rebuilt. Its lower-level raw kernel still implements
a more general normalized weighted mean, but the formal public binding used by
this matrix exposes only the positive normalized contract.

## Exactness matrix

| Scope | Chance treatment | Action treatment | Correct classification |
|---|---|---|---|
| T4 first | All `C(24,3) = 2,024` opponent deals under the uniform exchangeable restart belief | Every legal hero placement and exact opponent response | Exact for this T4 information-set model |
| T4 second | Terminal; no future deal | Every legal hero placement | Terminal exact |
| Normal full-deck T3 | Finite disjoint candidate/evaluation CRN particle batches | Every legal root action; each downstream T4 call exact | **T3 Monte Carlo with exact T4 children** |
| Declared finite-support T3 | Every positive normalized caller-declared world | Every legal root action under one fixed information-set-safe continuation | Exact only over the declared support |

`downstream_t4_samples = 0` therefore means exact child T4 enumeration. It
does not convert the root T3 hidden-card distribution into a full exact tree.

## Both-seat permutation and execution-mode result

For each seat, all six permutations of the same three dealt cards were run in
scalar and native-batch modes with the pinned `1/1/1/0` contract.

| Check per seat | Unique count / result |
|---|---:|
| Observation fingerprint | 1 |
| Legal ActionKey set digest | 1 |
| Selected ActionKey | 1 |
| Complete ActionKey rank/selection/evaluation/regret map | 1 |
| Candidate/evaluation belief and RNG provenance | 1 |
| Child information-set count | 1 |
| Semantic result digest | 1 |
| Legacy ordered-action mapping digest | 6 |
| Mapping-bound result digest | 6 |
| Every local original-index to ActionKey binding | Pass |
| Scalar/batch equality in both digest scopes | Pass |
| Deterministic repeat in both digest scopes | Pass |

The fast artifact took 4.2589 seconds scalar and 1.4972 seconds batch for 12
constrained roots. These timings are a correctness smoke, not representative
general-board latency.

## Python/Rust composition parity

The deliberately slow reference check ran normal T3 CRN with exact T4
children in both implementations. Every ActionKey, original index, rank,
selection EV, locked evaluation EV, regret, selected key, and envelope field
matched at absolute tolerance `1e-12`.

| Seat | Python | Rust | Speedup | Maximum action-value difference |
|---|---:|---:|---:|---:|
| first | 96.3318 s | 0.4942 s | 194.91x | 0.0 |
| second | 23.4982 s | 0.1072 s | 219.12x | 0.0 |

This constrained composition fixture has tied zero root values, so it proves
the mapping and execution path but is weak numeric coverage. Numeric coverage
comes from the separate non-zero finite-support fixtures below.

## Non-zero finite-support parity

Python exact scalar, Rust scalar, and Rust batch matched for both seats,
including support digest, selected ActionKey, every ActionKey EV, rank,
original index, future count, and regret.

- first-seat ranked scores:
  `6, 6, 6, 4.5, 4.5, 4.5, 1.5, 0, 0`
- second-seat ranked scores:
  `-0.5, -0.5, -2.25, -2.25, -6, -6, -6, -6, -6`

Invalid zero, negative, NaN, bool, and non-normalized weights, malformed world
schemas, and invalid basic field types are rejected before native dispatch.

## Regression evidence

- focused M3.1 runtime plus finite-support matrix:
  `38 passed in 8.19s`
- combined M3, M3.0, M3.1, and Python T3 oracle:
  `84 passed in 32.14s`
- full Python:
  `2417 passed, 2 skipped, 50 warnings in 809.79s`
- Rust M3 library: `36 passed`
- Rust M3 runner: `2 passed`
- `cargo fmt -p ofc_hu_m3_engine -- --check`: pass
- `cargo clippy -p ofc_hu_m3_engine --all-targets -- -D warnings`: pass

The workspace-wide fmt check remains red only because pre-existing unrelated
edits in `rust/ofc_stage3_feature_encoder/src/lib.rs` and
`rust/regular_fl_solver/src/main.rs` are not rustfmt-clean. Step 2 did not edit
or reformat those files.

## Fixed anchors and artifacts

- accepted DLL:
  `target/m30_build/release/ofc_hu_m3_engine.dll`
- DLL SHA-256:
  `03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69`
- engine version: `ofc_hu_m3_engine/0.1.0`
- full correctness artifact:
  `outputs/hu_joint_policy/m31_t3_step2/validation_summary.json`
- fast permutation artifact:
  `outputs/hu_joint_policy/m31_t3_step2/correctness_matrix.json`

Source hashes at freeze time:

| File | SHA-256 |
|---|---|
| `src/ofc_regular/hu_m31_t3_runtime.py` | `d9230a82b8350dde3937f63988a90c96e3cb1b6fe6f0b83858f550897276998a` |
| `src/ofc_regular/validate_hu_m31_t3_correctness.py` | `87b7dad054990e0b7012a77cb879babf72ad543efdedb4f86e7874990cca0c19` |
| `src/ofc_regular/hu_m3_rust.py` | `6de5bc5cf45d0088593b1c3d2eacbdd4fa2606a4c1063a128fc8c2a463514b78` |
| `tests/test_hu_m31_t3_runtime.py` | `8ea093130c6747b60cd6a5c54cf5a7316b329791bb3f9696e9d9f2a359b9e399` |
| `tests/test_hu_m31_t3_finite_support_matrix.py` | `b477fb09a66c2805c9986477bb3971e25a21af35bef1a70d090493c0a39e890f` |
| full correctness artifact | `aead786a2d03f33cd030446ba7451f0cb38bf94015f6634aa3c43e9ae4ec7f2a` |

## Preserved invariants

- `stage19_p0`, `stage18_p1`, `stage9f_p2`, and `stage7_m5_r10` remain rollback baselines.
- `src/ofc_regular/ai_profiles.py` remains at SHA-256
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.
- No named profile or `current` mapping was changed.
- No policy wrapper, replacement, model, or runtime gate was activated.
- No Spot VM or other cloud action was started.

## Frozen entry conditions for Step 3

Step 3 starts with a 10-root general-board profiling probe before allocating a
100-root pilot. The 100-root job remains No-Go unless that probe shows a
projected local wall time no greater than 60 minutes, completes with no native
failure or fallback, and keeps peak process memory below 16 GiB. If it fails,
the next action is local Rust profiling/optimization, not Spot expansion.

The first accepted 100-root artifact must additionally satisfy all of these
pre-registered integrity gates:

1. exactly 50 first-seat and 50 second-seat unique live-shape roots;
2. root seed `2026073201`, `--seed-stride 1000003`, and run ID
   `hu-m31-step3-local100-v1`;
3. zero forbidden hidden-truth fields and zero action-key/index/payload errors;
4. every root returns every legal action with finite values and exact T4 mode;
5. candidate/evaluation particle RNG domains are disjoint globally and per root;
6. scalar/batch semantic and mapping-bound parity on all 100 roots;
7. deterministic rerun of the first 20 roots in both digest scopes;
8. all six dealt-card permutations on a fixed 10-root subset have one semantic
   digest per root while every local ordered mapping remains valid;
9. unique observation fingerprints and non-overlapping future evaluation seeds;
10. all raw teacher values remain labeled diagnostic, never match EV.

Latency percentiles and the multi-budget convergence ladder are not invented
from constrained fixtures. They must be measured on the 10-root general-board
probe and frozen before the 100-root command is allowed to run. A 1,000-root
run, training-data generation, Spot VM, and any policy/profile activation
remain No-Go after Step 2.

## Reproduction commands

Fast correctness matrix (write-once output path required):

```powershell
python -m ofc_regular.validate_hu_m31_t3_correctness `
  --library target/m30_build/release/ofc_hu_m3_engine.dll `
  --library-sha256 03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69 `
  --output outputs/hu_joint_policy/m31_t3_step2/<new-fast-artifact>.json
```

Slow both-seat Python/Rust composition parity:

```powershell
python -m ofc_regular.validate_hu_m31_t3_correctness `
  --library target/m30_build/release/ofc_hu_m3_engine.dll `
  --library-sha256 03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69 `
  --python-exact-t4-parity `
  --output outputs/hu_joint_policy/m31_t3_step2/<new-full-artifact>.json
```
