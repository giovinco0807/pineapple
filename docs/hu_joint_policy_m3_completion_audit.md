# HU joint-policy M3 completion audit

Status date: 2026-07-13

Decision: **M3 Go as the native rollout/search engine; runtime policy
promotion remains No-Go.**

M3 ports the M2 information-set, ActionKey, deterministic RNG, hidden-card
belief, HU scoring, T4 and T3 contracts into Rust. It provides a deterministic
Python binding and a restart-safe JSONL shard runner. It does not change
`current`, activate a replacement policy, start a Spot VM, or turn teacher
scores into match-EV evidence.

## Native engine boundary

The policy-facing root is `ActorObservation`. The Rust request schema has no
field for an opponent private discard, replay world, or realized deck tail.
The hidden-card sampler constructs an exchangeable particle only from visible
cards. Every T3 child decision receives a newly constructed actor observation;
the outer particle is not passed into that policy decision.

Legal actions retain their legacy positional mapping, while every cross-module
identity and tie break uses the four-mask `ActionKey`. Both legal-set and
ordered-mapping digests are emitted. Candidate-selection and locked-evaluation
samples use separate counter-RNG domains and expose provenance digests.

## Search semantics

- T4 second seat evaluates every legal terminal placement.
- T4 first seat computes `max_a E[min_b u]`. With sample counts zero it
  enumerates all `C(24,3) = 2,024` future deals and every exact response.
- Normal T3 follows the live first/second action order and uses disjoint root
  CRN batches plus local-belief child actions cached by observation
  fingerprint.
- The separate T3 finite-support oracle exactly evaluates a caller-declared
  support under the canonical information-set continuation. It explicitly
  reports `full_52_card_tree_claimed: false`.
- These are teacher values under the recorded belief/continuation, not
  realized heads-up match EV and not a proof of Nash or mathematical
  optimality.

## Runner durability

The Rust runner requires a versioned JSONL row and unique `root_id`. Its
checkpoint records the full input hash, committed input-prefix hash, committed
output-prefix hash, configuration hash, byte boundaries, and last root. Output
is flushed and fsynced before an atomic checkpoint replacement. Resume
validates all hashes, truncates only an uncommitted suffix, and rejects corrupt
or incompatible state. Heartbeat and final output are atomic.

## Parity and speed evidence

| Gate | Result |
|---|---|
| Full Python regression | 940 passed in 47.95 s |
| Rust library tests | 36 passed |
| Rust runner CLI tests | 2 passed |
| Python/Rust focused parity | 9 passed |
| Pilot CLI tests | 10 passed |
| T4 first full 2,024-future parity | Pass |
| T4 second exhaustive parity | Pass |
| T3 first/second 1/1/1/1 parity | Pass; all ActionKeys and values |
| T3 declared-support exact parity | Pass for both seats |
| Scalar/batch parity | Pass |
| Six dealt-card permutations | Semantic policy/value parity |
| Raw `WorldState` request | Rejected |
| Clippy | Pass with warnings denied |

The locked four-root release benchmark used the same fresh roots and the same
1/1/1/1 search configuration in Python and Rust:

- Python: 2.2702 seconds
- Rust: 0.2643 seconds
- aggregate speedup: 8.5879x
- required gate: at least 5x
- selected ActionKeys and every per-action selection/evaluation value matched

## Bounded 100 -> 1,000 pilot

The 100-root phase completed before the CLI was allowed to start 1,000 roots.
Both phases used `seed_start=2026071301`, `seed_stride=1000003`, balanced seats,
and 1/1/1/1 samples.

| Phase | Seats | Runner time | Result |
|---|---:|---:|---|
| 100 roots | 50 first / 50 second | 6.291 s | Pass |
| 1,000 roots | 500 first / 500 second | 71.123 s | Pass |

For both phases: every fingerprint and result envelope was unique; all
ActionKey/index mappings were well formed; candidate/evaluation RNG provenance
was disjoint per root and globally; checkpoint and heartbeat were complete;
and a fresh deterministic subset rerun matched byte-for-semantic-result.

The pilot was local only. It loaded no profile or model, did not read
`current`, and performed no cloud or Spot action. All reported teacher values
are marked `diagnostic_not_match_EV`.

The 17 M0 direct config/model/project artifacts outside the policy registry
still match their frozen SHA-256 bytes. `ai_profiles.py` remains at the M1
card-free-guard hash
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`;
the `current` mapping is unchanged.

## Files and artifacts

Native implementation:

- `rust/hu_m3_engine/src/cards.rs`
- `rust/hu_m3_engine/src/state.rs`
- `rust/hu_m3_engine/src/action.rs`
- `rust/hu_m3_engine/src/action_key.rs`
- `rust/hu_m3_engine/src/scoring.rs`
- `rust/hu_m3_engine/src/infoset.rs`
- `rust/hu_m3_engine/src/counter_rng.rs`
- `rust/hu_m3_engine/src/belief.rs`
- `rust/hu_m3_engine/src/search.rs`
- `rust/hu_m3_engine/src/explicit_support.rs`
- `rust/hu_m3_engine/src/runner_support.rs`
- `rust/hu_m3_engine/src/lib.rs`
- `rust/hu_m3_engine/src/bin/runner.rs`

Python and validation:

- `src/ofc_regular/hu_m3_rust.py`
- `src/ofc_regular/validate_hu_m3_engine.py`
- `tests/test_hu_m3_rust.py`
- `tests/test_validate_hu_m3_engine.py`

Frozen evidence:

- `configs/hu_joint_policy_m3_status.json`
- `outputs/hu_joint_policy/m3_complete/validation_summary.json`
- `outputs/hu_joint_policy/m3_complete/run_manifest.json`
- versioned 100/1,000 input, output, checkpoint, heartbeat, and deterministic
  rerun artifacts in the same directory

## Remaining No-Go scope

- no runtime activation or full replacement;
- no model promotion based on teacher EV or placement accuracy;
- no reuse of hidden-discard-leaking legacy artifacts;
- no Spot generation until M4 freezes its data manifest, shard budget, and
  acceptance gates;
- no claim of a full Bayesian posterior, low exploitability, Nash equilibrium,
  or mathematical complete optimality from M3.

The next milestone is M4: build and evaluate the T1 second-seat policy using
this native teacher, with thresholds locked before fresh multi-opponent paired
seat-swap evaluation.
