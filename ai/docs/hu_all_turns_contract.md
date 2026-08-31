# Heads-up all-turn AI contract

Status: canonical for new oracle, runtime, self-play, and promotion work from 2026-07-12 onward.

Machine-readable version: `position_contract_version = "bb_first_v1"`.
Artifacts or checkpoints without this field are legacy until their action order
and decision-board shapes are audited; they must not be loaded silently by the
canonical runtime.

## Objective

Build one AI system that selects placements on T0 through T4 for both heads-up positions. Every decision maximizes expected zero-sum session value under the information legally visible to the acting player.

The system may use turn/position specialists internally, but they must share one state contract, rules engine, scoring function, action identity, and promotion gate.

## Normal-hand action order

- The non-button player acts first on every turn. The project retains the legacy label `bb` for this first position.
- The button acts second after seeing the first player's public placement. Its label is `btn`.
- `is_btn` means button/second; it never means first.
- Preferred external wording is `first` and `second`; `bb` and `btn` remain stable storage labels.

Canonical seat order is implemented in `ai.engine.turn_order.action_order`.

Expected public board counts at each decision are:

| Turn | first (`bb`) Hero/Opp | second (`btn`) Hero/Opp |
|---:|---:|---:|
| T0 | 0 / 0 | 0 / 5 |
| T1 | 5 / 5 | 5 / 7 |
| T2 | 7 / 7 | 7 / 9 |
| T3 | 9 / 9 | 9 / 11 |
| T4 | 11 / 11 | 11 / 13 |

Fantasyland requires a separate action-order contract because its board can remain hidden until set.

## Information boundary

An action may depend on:

- the acting player's public board;
- the opponent's placements already made public;
- the acting player's current private draw;
- the acting player's own discarded cards;
- turn, position, Fantasyland state, and session state.
- the public placement history (turn, actor, and placed cards only).

It may not depend on:

- the opponent's current private draw before it is placed;
- the opponent's face-down discards;
- future draws;
- a determinization identifier or any hidden cards used only by the evaluator.

Sampled evaluators must choose one action per information set. They must not choose an earlier action separately for each sampled future trajectory (strategy fusion).

The public history is required for the final system because a face-down discard
can be inferred probabilistically from the two cards an opponent chose to show.
Snapshot-only evaluators that assume uniform hidden discards are bootstrap PIMC
evaluators and must identify their information model explicitly.

## Value target

Bootstrap target:

1. exact terminal row score, scoop, and royalty difference;
2. calibrated direct Fantasyland continuation value;
3. zero-sum value from the acting player's perspective.

Final target:

- played-out Fantasyland and session continuation value;
- chip caps and session stopping rules;
- approximate equilibrium/self-play value, not best response to a single frozen opponent only.

## Solver hierarchy

| Turn | Target evaluation |
|---:|---|
| T4 | exact terminal mechanics; history-weighted hidden-discard range for first, direct terminal response for second |
| T3 | sampled public-state expectimax; inner T4 exact |
| T2 | sampled next-turn search using promoted T3 policy/value |
| T1 | sampled next-turn search using promoted T2 policy/value |
| T0 | broad candidate generation plus T1 continuation search |

No sampled, capped, pooled, or model-only result may be labeled globally exact. Every result records `method`, `candidate_scope`, position, seed, sample count, uncertainty, and runtime.

## Promotion

Each turn is promoted separately for first and second position, with Joker and non-Joker strata. Required evidence includes legal-action coverage, candidate recall, EV regret mean/p95/max, seed stability, paired confidence for the best-vs-runner-up gap, and p50/p95/max runtime.

The final all-turn promotion additionally requires paired-seat self-play with button alternation and an exploitability/league evaluation against frozen historical policies.
