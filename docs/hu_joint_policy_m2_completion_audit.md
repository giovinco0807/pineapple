# HU joint-policy M2 completion audit

Status date: 2026-07-13

Decision: **M2 Go as a Python correctness reference; runtime promotion and
large-scale generation remain No-Go.**

M2 repairs the late-street teacher. It does not activate a new runtime policy,
revalidate the fixed P0/P1/P2/T3 model chain, or establish Nash optimality.

## Exactness boundary

- T4 second seat is exhaustive over every legal placement against a complete
  opponent board.
- T4 first seat evaluates `max_a E_deal[min_response u]`. Under the declared
  uniform exchangeable prior, the opponent's future deal is marginally uniform
  over all `C(24,3) = 2024` unknown three-card sets. Every response is terminal
  exhaustive.
- This T4-first result is exact under the declared uniform-restart belief. It
  does not posterior-reweight hidden discards from the likelihood of observed
  opponent placements and is not a globally exact Bayesian solution.
- T3 has an exact path for a declared finite support and fixed deterministic
  information-set continuation. Both seats are checked against an independent
  weighted finite-tree oracle. Normal 52-card T3 trees are not tractable in the
  Python reference and use common-random Monte Carlo.
- A normal T3 value is `Q^pi` under the serialized local-belief continuation,
  not a policy-independent game value and not realized match EV.

## Correct action order

The teacher now follows the live sequence.

- T3 first: hero T3 action, opponent T3 deal/action, hero T4 first-seat belief
  action, opponent T4 second-seat exact response.
- T3 second: hero T3 action, opponent T4 first-seat belief action, hero T4
  second-seat exact response.
- Every child policy receives a newly constructed `ActorObservation`. The
  outer realized deck and the other player's private discards are never passed
  into that decision.

Two outer worlds that reach the same child information set but have different
later deck tails are tested to produce the same child action. This is the
strategy-fusion/non-clairvoyance guard.

## Candidate and evaluation separation

Candidate selection and reported evaluation use separate counter-RNG domains
and independently configurable seeds. The locked selected action is evaluated
on the evaluation batch; finite-sample maxima are named sample quantities, not
oracle values. T4 MC uses a fixed counter-based partial Fisher-Yates contract
with a golden vector for the future Rust port.

## T2 integration

The M2 T4 selector is threaded through both scalar and batched T2 continuation
paths. It is reachable from the real T2 generator CLI with
`--enable-m2-t4-search` and from both PowerShell shard runners. When enabled,
model loading uses the explicitly selected T3 continuation profile rather than
`current`.

The default remains the legacy path to avoid silently changing existing
artifacts and baselines. Therefore any new hidden-discard-safe T2 artifact must
record the enable flag and full T4 search config. Old audit/replay entrypoints
without that opt-in remain quarantined.

## Validation evidence

| Gate | Result |
|---|---|
| T4 first full `2024 x all responses` counterexample | Pass; corrected action EV `+6`, legacy standalone action EV `0` |
| T4 second independent terminal brute force | Pass |
| T3 first/second exact finite-support oracle | Pass |
| T3 first/second live event-order oracle | Pass |
| Scalar/batch parity | Pass |
| Candidate/evaluation seed independence | Pass on both seats |
| Strategy-fusion/non-clairvoyance | Pass |
| T3/T4 hidden-belief partition and truth invariance | Pass |
| Raw replay/dead-card input rejection | Pass |
| Non-fire same-snapshot trajectory identity | Pass |
| Full regression | `921 passed in 39.79s` |
| Focused M2 regression | `181 passed in 18.59s` |
| PowerShell parser | 3/3 scripts pass |
| Fixed baseline artifacts | 16/16 SHA-256 matches M1 |
| Policy registry/current mapping | SHA-256 matches M1; unchanged |

Fresh bounded smoke:

- 4 hands / 8 roots, exactly 4 first and 4 second seat
- source seeds `202607130000 + hand_index * 1009`
- candidate/evaluation samples `2/4`, downstream T3/T4 samples `1/2`
- 8/8 unique observation fingerprints
- candidate/evaluation RNG overlap `0`
- deterministic rerun digest match
- p50 `2.353`, p95 `5.827`, max `5.838` seconds/root
- total `30.581` seconds
- no profiles, models, `current`, cloud, or Spot VM loaded/used

This smoke is speed and correctness evidence only. Its teacher scores are
explicitly marked `diagnostic_not_match_EV`; it is not a holdout, match-EV, or
promotion result.

## Files and artifacts

Core implementation:

- `src/ofc_regular/hu_late_street_teacher.py`
- `src/ofc_regular/hu_turn3_joint_exact_teacher.py`
- `src/ofc_regular/hu_belief.py`
- `src/ofc_regular/hu_turn2_teacher_data.py`
- `src/ofc_regular/validate_hu_m2_teacher.py`

Frozen evidence:

- `configs/hu_joint_policy_m2_teacher.json`
- `configs/hu_joint_policy_m2_status.json`
- `outputs/hu_joint_policy/m2_complete/correctness_pilot.json`
- `outputs/hu_joint_policy/m2_complete/validation_summary.json`
- `outputs/hu_joint_policy/m2_complete/run_manifest.json`

## Remaining No-Go items

- no full replacement or runtime profile activation;
- no reuse of old teacher EV as match EV;
- no claim that uniform-restart belief is a fully Bayesian posterior;
- no 100-1,000 state scaled pilot until the Rust M3 engine reaches parity;
- no Spot VM generation, retraining, or P0/P1/P2/T3 promotion.

The next milestone is M3: port the frozen semantics, ActionKey, counter RNG,
belief, T4, T3 and batch contracts to Rust, then run the 100-1,000 state pilot.
