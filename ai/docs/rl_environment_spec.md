# RL Session Environment Specification

Last updated: 2026-05-09

## Purpose

This document fixes the rules for the real-game reinforcement learning environment.

The RL target is not one-hand raw EV. The RL target is session EV under the real 200-point match rules, including Fantasyland play, score caps, and the 40-point session stop rule.

## Session Rules

- Both players start each session with 200 points.
- Total points are conserved at 400.
- Points never go below 0.
- The first BTN/BB assignment is random for each new session.
- BTN/BB alternate after every hand.
- Before starting a new normal hand, if neither player is in Fantasyland and the score gap is at least 40 points, the session ends.
- If either player is in Fantasyland, the session continues even if the score gap is at least 40 points.
- After Fantasyland ends, if both players are non-FL and the score gap is at least 40 points, the session ends before the next normal hand.
- After session end, a new session starts at 200-200 with a new random initial BTN.

## Capped Score Transfer

Raw hand score is converted into an actual chip transfer capped by the loser remaining points.

Example:

- Before hand: P0 = 395, P1 = 5
- Raw result: P0 wins 20
- Actual transfer: min(20, 5) = 5
- After hand: P0 = 400, P1 = 0

This means the same board can have different strategic value depending on chip state. The RL state must include both players' scores and the score gap.

## Fantasyland

Fantasyland must be played out. It should not be replaced only by a fixed EV in the final RL target.

Track:

- FL entry
- FL card count: 14, 15, 16, or 17
- FL hand score
- FL stay result
- FL chain id
- FL chain total score

Fixed FL EV values may still be used as bootstrap priors or search bonuses, but the value target for RL is the observed session return after actual FL play.

## Agent Interface

Agents receive:

- observation
- legal actions
- legal mask

Agents return:

- chosen action
- chosen action index
- candidate action indices
- optional policy logits/probabilities
- optional value estimates

Agents must not select illegal actions. The environment rejects illegal actions.

The final player should not blindly take NN top-1. It should evaluate multiple candidates using policy priors, value estimates, rollout, or shallow search.

## Decision Log Schema

Each decision record should contain:

- `session_id`
- `hand_id`
- `seat`
- `turn`
- `is_btn`
- `is_fl`
- `opp_is_fl`
- `chips_self`
- `chips_opp`
- `score_gap`
- `board_self`
- `board_opponent`
- `dealt_cards`
- `known_discards`
- `state`
- `legal_mask`
- `chosen_action`
- `chosen_action_index`
- `candidate_actions`
- `policy_logits`
- `policy_probs`
- `value_estimates`
- `hand_raw_score`
- `capped_score_delta`
- `session_return`
- `fl_entry`
- `fl_card_count`
- `fl_hand_score`
- `fl_stayed`
- `fl_chain_id`
- `fl_chain_total_score`

## Learning Target

Primary value target:

```text
session_return = final_chips[seat] - 200
```

Store raw hand score and capped score delta as auxiliary labels, but the main RL value target is the capped session return.

