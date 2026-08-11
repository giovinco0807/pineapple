# M3.1 T3 strict runtime boundary - Step 1 completion audit

Status date: 2026-07-16

Status: complete locally as a search component. No policy/profile is activated,
`current` is unchanged, and no Spot VM was started.

## Scope

Step 1 adds a production-style trust boundary around the existing Rust T3
search without changing the accepted M3.0 native engine. The new component is
`src/ofc_regular/hu_m31_t3_runtime.py`.

The component:

- accepts only a T3 `ActorObservation`;
- requires a prebuilt release native library;
- verifies the native SHA-256 and engine version before execution;
- never builds a missing library and has no legacy fallback;
- fixes `downstream_t4_samples` to zero inside an immutable
  `JointExactConfig`;
- therefore uses the same pinned M3.0 native T4 kernel: T4-first enumerates all
  2,024 opponent deals and exact responses, while T4-second is terminal
  exhaustive;
- exposes scalar and batch search, but no policy wrapper or named profile.

T3 root chance remains common-random Monte Carlo. This is not an exact T3
solver and its values are diagnostic teacher/search estimates, not realized
match EV.

## Strict result contract

Before returning an action, the wrapper validates:

- exact scalar/batch envelope schema and native engine version;
- complete coverage of every freshly generated Python legal action;
- ActionKey, action payload, original index, rank, legal-set digest, and legal
  order digest;
- candidate-score canonical argmax and locked independent evaluation value;
- JSON numeric types without bool-as-int or numeric-string coercion;
- finite selection/evaluation values, gaps, and per-action regrets;
- candidate and evaluation belief schema, seed, run ID, sample count, and root
  fingerprint;
- deterministic Python regeneration of every particle digest and RNG key
  digest from the ActorObservation;
- candidate/evaluation RNG-domain disjointness;
- continuation policy ID, strategy-fusion guard, downstream T3 budget, and
  integer `downstream_t4_samples == 0`;
- the full search contract in the decision artifact and semantic result digest.

Any mismatch raises `HuM31T3RuntimeError`. There is no fallback action.

## Fixed native boundary

- accepted Windows library:
  `target/m30_build/release/ofc_hu_m3_engine.dll`
- SHA-256:
  `03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69`
- engine version: `ofc_hu_m3_engine/0.1.0`
- M3.0 runtime config SHA-256:
  `a5dfea8105e5145fdf9fa5bae889dd9d1ef0aeec27fc953f7310d1c8a0b12561`
- M3.0 completion audit SHA-256:
  `c8b62c5aedb4b96c1283b536e043aa3179a453e298468a122ae2a2541f9b668e`

The nested T4 calls use Rust's internal `evaluate_t4` in this same pinned
library. They are semantically the accepted M3.0 exact native kernel, but do
not pass through the Python `HuM3T4ExactSolver` wrapper for every child.

## Validation

- M3.1 focused: `27 passed in 3.73s`
- M3/M3.0/M3.1 combined: `56 passed in 22.20s`
- full Python: `2406 passed, 2 skipped, 50 warnings in 741.14s`
- Rust library: `36 passed`
- Rust runner: `2 passed`
- Cargo fmt: pass
- Cargo clippy with warnings denied: pass
- final constrained both-seat smoke with candidate/evaluation/downstream-T3
  samples `2/3/2` and downstream-T4 `0`: pass

Final constrained smoke:

| Seat | Latency | Legal actions | Child infosets | Result digest |
|---|---:|---:|---:|---|
| first | 1.7131 s | 3 | 225 | `107c43d22c9646c59a8527048a7336c2f350c4c36f3b8baf66b30c79cb01bf47` |
| second | 0.2574 s | 3 | 30 | `35f7e563057645ffaef4154df298c5f85b0c5c530352469f30a4f3cfd257b972` |

These constrained timings prove the exact-T4 bridge and strict boundary; they
are not general-board runtime estimates or promotion evidence.

## Preserved invariants

- `stage19_p0`, `stage18_p1`, `stage9f_p2`, and `stage7_m5_r10` remain intact.
- `src/ofc_regular/ai_profiles.py` remains at SHA-256
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.
- No named profile or `current` mapping was edited by Step 1.
- No model was trained, promoted, or activated.
- No cloud action was performed.

## Next gate

Step 2 will expand correctness coverage to dealt-card permutations and the
formal exact/scalar/batch matrix, then run bounded 100-root sample-convergence
and latency profiling. A 1,000-root run and Spot VM remain No-Go until that
100-root gate passes.
