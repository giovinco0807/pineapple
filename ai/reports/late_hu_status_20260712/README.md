# Late-turn HU status (2026-07-12)

## Completion decision

- T4 BTN: complete. Every legal terminal action is scored against the completed BB board.
- T4 BB mechanics: complete for a uniform/exchangeable hidden-discard range.
  It is not final-HU complete because public history can make BTN's three
  hidden discards nonuniform.
- T3: not HU-complete. The current oracle exactly solves Hero's own remaining T4 draw/placement objective, but it skips intervening opponent actions.

The runtime exposes this distinction through `exact_scope` and `hu_exact`:

| State | `exact_scope` | `hu_exact` |
|---|---|---:|
| T3 BB/BTN | `t3_self_board_all_t4_draws_best_t4` | false |
| T4 BB generic exact-late path | `t4_all_opponent_draws_best_response_given_exclude` | false |
| T4 BB uniform-range adapter | `t4_uniform_hidden_discard_marginal_all_draws_best_response` | false |
| T4 BTN, opponent is complete | `t4_terminal_vs_complete_opponent` | true |
| T4 offline self-board fixture | `t4_terminal_self_board` | false |

## T4 implementation

For the uniform-range BB leaf, the canonical Python reference and Rust runtime
use the same sequence:

1. Apply one candidate BB action.
2. Remove BB's discard from the live cards.
3. Enumerate every three-card BTN draw.
4. Enumerate every legal BTN placement for that draw.
5. Select the response that maximizes BTN's canonical HU score.
6. Negate the zero-sum score and average it for BB.

This enumerates the correct marginal when the unknown BTN discard identities
are uniform/exchangeable. A history-weighted public solver must instead score
every BB action in physical hidden-discard particles and average the complete
action vector at the shared BB information set before selecting an action.

That conditioned physical-vector layer is now implemented separately from the
uniform adapter. It accepts either both exact three-card hidden-discard sets or
an explicit complete reduced remaining-card range, verifies every Rust
candidate and `C(n,3)` draw count, and returns all action vectors without a
`best`/`chosen_action`. Its output remains `hu_exact=false` and
`public_policy_safe=false` until history-weighted particles are aggregated at
one shared `InfoSetKey`.

The evaluator uses the current 54-card/two-Joker rules and direct progressive Fantasyland values from `ai/config/fl_ev.json`. The legacy `t4_exact_solver` is not used because its schema, Joker identity, 52-card generator, and FL value treatment differ from the canonical runtime.

## Verification

- Independent four-draw golden: BB EV/raw score `-20.5`.
- Two-Joker golden: `X1` and `X2` remain distinct; Hero EV/raw score `+2`, royalty `7`, QQ Fantasyland rate `1.0`.
- Python/Rust full-candidate parity, including Joker-heavy positions: Top1 and all checked metrics agree.
- Runtime API, teacher generation, and Rust-teacher conversion preserve the opponent-response mechanics scope.
- Ambiguous partial opponent boards are rejected instead of being labeled exact.
- Rust tests: 12 passed (`ofc_core` 7, late solver 5).
- Targeted Python tests: 40 passed.

A representative full live-deck position had 29 unseen cards, 3,654 BTN draws,
and six BB actions. Rust took about 13 ms inside the solver and about 40 ms
including process startup. These timings and parity certify terminal mechanics,
not a history-weighted hidden-discard belief.

## T3 re-evaluation

The independent 5,000-position gate contains:

- BB: 2,500 positions with Hero 9 cards and opponent 9 cards.
- BTN: 2,500 positions with Hero 9 cards and opponent 11 cards.

The missing sequence is role-dependent:

- BB T3 must model BTN T3 response, then BB T4, then BTN T4 response.
- BTN T3 must model BB T4 action, then BTN T4 response.

Therefore the former 1,000/1,000 recall and zero EV loss certify only the self-board oracle/candidate pool, not HU strategy. Exact full enumeration is too large for the five-second runtime target. The next evaluator should keep the inner T4 subgame exact and sample the outer T3 chance/action nodes with common random numbers and confidence estimates.

Opponent discards are hidden. A T3 evaluator must preserve the public-information belief state; a perfect-information determinization must be labeled as PIMC/sampled rather than HU-exact.

## Next gate

The first three planned bootstrap steps are complete:

1. Reduced-deck exact BTN/BB T3 goldens, including X1/X2, pass.
2. The sampled evaluator uses explicit BB/BTN roles, canonical action order,
   common random tapes, and exact T4 leaves.
3. The balanced 100-position/two-seed pilot completed.

The pilot did not promote T3. BTN was 50/50 seed-stable at outer=8, while BB
was 26/50 at outer=2/inner=1. Increasing the 20 hardest BB positions to
outer=4/inner=4 and outer=8/inner=4 produced 15/20 seed agreement at both
budgets. The remaining five positions were still only 2/5 at
outer=16/inner=4, with runtime p95 above 36 seconds.

The revised gate is:

1. Group every future decision by the acting player's public/private
   information set instead of solving it independently per determinization.
2. Validate that public-belief evaluator against reduced exact games that
   intentionally make PIMC choose a different action.
3. Repeat the two-seed BB/BTN pilot with the corrected information model.
4. Promote only after an unused root-disjoint BB/BTN gate reports EV regret,
   seed stability, paired confidence, and runtime separately by Joker stratum.
5. Only a promoted T3 continuation may generate T2, T1, and T0 targets.

Detailed evidence is in
`ai/reports/t3_hu_sampling_pilot_100_20260712/README.md`.
