# All-turn HU AI status (2026-07-12)

## Goal

Select the best legal placement on T0 through T4 for both BB/first and
BTN/second, without using an opponent's hidden draw or discard. The final
policy target is approximate equilibrium/self-play value under the canonical
two-Joker rules, not a game UI or a perfect-information best response.

## Promotion status

| Turn | BB / first | BTN / second | Current evidence |
|---:|---|---|---|
| T4 | not final-promoted | promoted exact terminal response | BB candidate/terminal mechanics are exact, but the hidden BTN-discard range is currently uniform rather than history-weighted |
| T3 | not promoted | not promoted | exact reduced goldens and exact T4 leaves pass; PIMC pilot is not public-belief/equilibrium safe |
| T2 | not promoted | not promoted | canonical ingress and rollout order fixed; former `t2_hu_*` continuation skips required global opponent actions |
| T1 | not promoted | not promoted | canonical ingress and rollout order fixed; no promoted T2 continuation target yet |
| T0 | not promoted | not promoted | canonical self-play order fixed; no promoted downstream continuation target yet |

“Not promoted” means the path must not produce final training labels or become
the serving default. It does not mean no code exists for that turn.

## Canonical contract now enforced

- Every normal street is BB/non-button first, then BTN/button second.
- Decision board shapes are validated separately for both roles.
- `position_contract_version` is `bb_first_v1`.
- T1/T2 dict ingress rejects missing, contradictory, or impossible role/board
  combinations.
- Rollouts complete a pending BTN action on the same street after a BB root,
  then start every later street with BB.
- Public history retains only turn, actor, and public placements. Private
  deals, discards, model scores, and branch metadata are stripped.

## T3 evidence

The balanced 100-position/two-seed bootstrap produced:

- BTN outer=8: 50/50 identical Top-1 decisions.
- BB outer=2/inner=1: 26/50 identical Top-1 decisions.
- BB's 20 hardest positions at outer=4/inner=4: 15/20.
- The same 20 at outer=8/inner=4: 15/20.
- The remaining five at outer=16/inner=4: 2/5, with runtime p95 above 36 s.

The Rust joint T4 evaluator made the representative BB sampled path 2.84x
faster without changing any candidate value or selected action. The remaining
problem is not terminal accuracy: PIMC solves future policies separately for
hidden-card worlds. More uniform sampling is therefore stopped.

The BB T4 leaf was also reclassified after an independent information-set
audit. Enumerating every unseen BTN draw is the exact marginal only when BTN's
three hidden discards are exchangeable/uniform. Public placement history can
make that posterior nonuniform. The leaf now reports `hu_exact=false`,
`uniform_range_exact=true`, and must be aggregated over history-weighted
physical particles before the final BB policy is selected.

## Required next path

1. Build a T3 information-set solver whose policies are shared across worlds
   indistinguishable to the acting player.
2. Because public placements signal private card information, validate the
   grouped bootstrap and then train a public-tree CFR/MCCFR policy; local
   max/min backups alone are not a final equilibrium solution.
3. Re-run root-disjoint T3 gates for each role and Joker stratum.
4. Feed only promoted T3 continuation values into a corrected global T2
   sequence, then repeat backward for T1 and T0.
5. Finish with alternating-button paired self-play and exploitability/league
   evaluation against frozen policies.

Phase 1 and the first bounded phase-2 invariants of item 1 are now implemented:

- canonical `PrivateRecall`, `JointParticle`, and information-set keys;
- exact Bayes range updates with epsilon-smoothed behavior likelihood;
- exact grouped `max(E)` / `min(E)` diagnostics that expose PIMC strategy
  fusion;
- a deterministic reduced public signaling CFR+ reference with exact best
  responses and exploitability;
- a strict Rust adapter returning every BB T4 candidate value under an
  explicit uniform hidden-discard range without selecting a BB action.
- strict physical-world validation across both players' recall, public cards,
  the acting draw, and the represented undealt range;
- a separate conditioned physical T4 API that returns every candidate vector
  for exact hidden-discard or explicit reduced-card ranges without selection;
- one shared policy/regret node per `InfoSetKey`, with complete particle action
  vectors chance-aggregated before either BB/max or BTN/min regret updates.
- recursive synchronous CFR+ with counterfactual reach, own-reach average
  strategies, exhaustive infoset-policy best responses, and NashConv;
- commitment-checked compilation of every conditioned Rust T4 candidate into
  recursive tree leaves without a hidden-world preselection;
- a content-bound `promotion_gate_v2` scoped only to M2 reduced-reference
  readiness across BB/BTN and Joker 0/1/2. It re-derives metrics from raw run
  records and verifies every embedded provenance manifest hash.

The reduced bluff golden converges near its theoretical mixed strategy
(weak bet 1/3, call 2/3) with exploitability about 0.00046 at 20,000
iterations. The new particle layer proves the no-strategy-fusion aggregation
contract, and the recursive reduced reference now solves signaling goldens.
The six canonical actor/Joker reduced fixtures and the M2 gate artifact are
complete. Save/readback validation passes all six strata with maximum
exploitability 0.000152017689331152 and zero Python/Rust leaf difference,
order-replay policy TV, and BR re-evaluation residual. This promotes only the
finite reduced reference. Full-card range construction and runtime integration
remain unfinished, so full-card T3/T4 stays unpromoted.

A BTN-root uniform-range baseline is also implemented. It makes BB select its
T4 action from a PlayerView that excludes BTN's hidden discards, then scores
that fixed action in the actual physical particle. Natural and X1/X2 tests
prove the BB policy is invariant when only BTN's hidden discard identity is
changed. This removes that PIMC leak, but it ignores history-based range
weights and is therefore `equilibrium_approx=false`, `hu_exact=false`, and not
promoted.

## Existing CFR audit

The repository's former Python and Rust CFR runs are not continuation
checkpoints for this plan.

- The saved Python strategy was generated under the old button-first order.
  Its snapshot infoset omits ordered public/private recall, standard depth
  never reached a terminal in the saved run, and its leaf heuristic is not
  zero-sum symmetric.
- The Rust solver still acts P0 then P1 with P0 as button, omits the opponent
  public board from its infoset, and uses hand-array indices whose action
  meaning changes when the same cards are ordered differently. Its current
  outcome-sampling reach update is also not a valid final estimator.

Reusable parts are the canonical scorer, board/action-capacity machinery, and
regret-store structure. The strategy/checkpoints and traversal math are not
reused. New Python CFR checkpoints now require contract/rules/action/info/FL
metadata; old files are rejected by default before their pickle store is
loaded, with an explicit diagnostic-only override.

Supporting reports:

- `ai/reports/late_hu_status_20260712/README.md`
- `ai/reports/t3_hu_sampling_pilot_100_20260712/README.md`
