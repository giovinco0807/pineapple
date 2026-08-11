# HU joint policy M3.0 exact T4 completion audit

Status date: 2026-07-16

## Decision

M3.0 is `Go` as a local, explicit opt-in T4 component. It is not activated by
any existing named profile and does not change `current`. The release/source
boundary remains pending R0 review.

The runtime returns the best legal action, its EV, and the EV of every legal
action. Online play does not require a learned approximation at T4: the Rust
engine evaluates the declared information-set objective directly. Legacy T4 is
not executed on the normal no-logging path.

## Exactness boundary

First seat solves

```text
max hero action
  expectation over every uniformly weighted unknown opponent three-card deal
    min exact opponent response
      terminal HU score
```

At a normal T4-first root this is all 2,024 combinations from 24 unknown cards
and every legal opponent response for each deal. Second seat enumerates all
legal hero terminal placements against the completed public opponent board.

This is exact under the declared uniform exchangeable restart belief. It is not
a posterior learned from the opponent's public action history, a Nash
equilibrium proof, or mathematical full-game optimality.

## Runtime and information boundary

- The card-bearing entrypoint requires `ActorObservation` at T4.
- Hero board, opponent public board, hero private discards, dealt cards, seat,
  action order, street, and scoring context are accepted.
- Opponent private discards, the realized remaining deck, `WorldState`, replay
  truth, and unknown fields are rejected in Python and in scalar/batch Rust.
- The scoring context used to decide is propagated through normal play and
  terminal scoring.
- A prebuilt release library, matching engine version, and pinned SHA-256 are
  required. Missing/mismatched/corrupt native results raise; there is no build
  or legacy fallback.
- Runtime activation is only through `--t4-mode-a m30_exact` or
  `--t4-mode-b m30_exact`. Omitting the option retains legacy behavior.

## Action and value semantics

Legal actions are freshly enumerated in Python and bound to order-independent
semantic `ActionKey` values. Rust must return exactly the same legal set, order
digest, payload, index, and ActionKey mapping. The selected native result is
remapped to the Python legal action rather than trusted as an unchecked
payload. Dealt-card permutations produce identical semantic values and choice.

First-seat EV means expected terminal HU points under the declared restart
belief and exact response. Second-seat EV means terminal HU points against the
completed board. Neither value is reported as realized match EV.

## Correctness, determinism, and parity

The following checks pass:

- independent Python/Rust exact action-value parity for both seats;
- scalar/native-batch parity and batch input-order preservation;
- all legal ActionKeys and EVs returned;
- all six dealt-card permutations preserve semantic values;
- deterministic reruns preserve result digests;
- T4-second agrees with legacy terminal exact selection;
- hidden-truth aliases and unknown request/config/board fields fail closed;
- missing library, version/hash mismatch, and corrupt native result do not
  fall back;
- no-logging runtime does not call legacy T4;
- focused Python suite: 89 passed;
- full Python suite: 2,379 passed, 2 skipped;
- Rust library suite: 36 passed.

## 100 to 1,000 performance pilot

Both pilots used balanced first/second roots generated from live legal
trajectories without consulting a model or `current`. The 1,000-root seed range
starts after the 100-root range, and both record `seed_stride`.

| Pilot | Result | First p95 | First p99 | Second p99 | Batch throughput |
|---|---:|---:|---:|---:|---:|
| 100 roots | pass | 39.13 ms | 43.75 ms | 1.48 ms | 129.60 roots/s |
| 1,000 roots | pass | 42.91 ms | 54.53 ms | 1.70 ms | 156.04 roots/s |

The 1,000-root first-seat maximum was 91.72 ms. First-seat exact changed the
legacy action on 96 of 500 roots. Candidate samples and evaluation samples are
both zero because T4 is exhaustive.

## Fresh paired evaluation

The strongest clean runtime diagnostic is a fresh same-profile, seat-stable
counterfactual against `random_exact_final`. Only profile A received M3.0 T4;
profile B retained legacy T4. The two seat-swapped hands use the same deck and
seat-stable policy randomness.

| Metric | Result |
|---|---:|
| Paired seeds / hands | 100 / 200 |
| EV/hand | +0.295 |
| 95% CI | [+0.1204, +0.4696] |
| Override events | 11 first / 0 second |
| Realized points per override | +5.36 |
| False-positive overrides | 0 |
| p95 / p99 / max override tail loss | 0 / 0 / 0 |
| Non-fired pairs | 89 |
| Non-fire full-trajectory mismatches | 0 |

The non-fire check hashes every action, board, discard, final board, and
terminal score; score cancellation alone is not sufficient.

A separate fresh 100-paired fixed-chain smoke produced `+0.065 EV/hand`, 95%
CI `[-0.0069, +0.1369]`, with 4 winning, 0 losing, and 96 tied paired seeds.
Its point estimate is positive but the interval crosses zero. More importantly,
the upstream `stage19_p0 -> stage18_p1 -> stage9f_p2 -> stage7_m5_r10` chain is
quarantined, so this is only a state-distribution smoke and not joint-policy
strength acceptance.

## Activation and rollback boundary

`stage19_p0`, `stage18_p1`, `stage9f_p2`, and `stage7_m5_r10` remain unchanged
as rollback baselines. No M3.0 profile was added to `ai_profiles.py`, whose
frozen SHA-256 remains
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.
`current` was not changed. Rollback consists of omitting the explicit
`m30_exact` mode.

## Files and artifacts

Implementation:

- `src/ofc_regular/hu_m3_t4_runtime.py`
- `src/ofc_regular/hu_m3_rust.py`
- `src/ofc_regular/hu_infoset.py`
- `src/ofc_regular/evaluate_matchups.py`
- `src/ofc_regular/validate_hu_m30_t4_runtime.py`
- `src/ofc_regular/play_ai.py`
- `src/ofc_regular/decision_trace.py`
- `rust/hu_m3_engine/src/search.rs`
- `rust/hu_m3_engine/src/infoset.rs`
- `rust/hu_m3_engine/src/state.rs`
- `rust/hu_m3_engine/src/lib.rs`
- `tests/test_hu_m3_t4_runtime.py`

Evidence:

- `configs/hu_joint_policy_m30_t4_runtime.json`
- `outputs/hu_joint_policy/m30_t4_complete/pilot_100.json`
- `outputs/hu_joint_policy/m30_t4_complete/pilot_1000.json`
- `outputs/hu_joint_policy/m30_t4_complete/paired_random100_v3_summary.json`
- `outputs/hu_joint_policy/m30_t4_complete/paired_random100_v3_t4_decisions.jsonl`
- `outputs/hu_joint_policy/m30_t4_complete/paired_stage19_100_summary.json`

## Remaining No-Go scope

- Do not change `current` or activate a named profile before R0 source review.
- Do not describe M3.0 as full-Bayesian, Nash, or full-game mathematical
  optimality.
- Do not treat the quarantined fixed chain as a safe joint-policy promotion.
- Do not start large Spot generation for M3.1 until its local correctness and
  speed gates are frozen and pass.

The next implementation milestone is M3.1: rebuild the T3 reference and
selective candidate from hidden-discard-safe observations, using M3.0 exact T4
as the accepted continuation.
