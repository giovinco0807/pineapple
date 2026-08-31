# Bottom-Up Oracle to Reinforcement Learning Roadmap

Last updated: 2026-05-09

## Goal

Build a full-game OFC Pineapple AI that can eventually improve through self-play reinforcement learning.

The bottom-up oracle models are not the final target. They are used to create a strong, stable initial policy and value estimate so that later RL does not start from random play.

## Target Real-Game Session Rules

The final training/evaluation environment should match the intended real game as closely as possible.

Session rules to preserve:

- Both players start with 200 points.
- Total points are conserved at 400.
- Points never go below 0.
- Hand scores are capped by the loser's remaining points. For example, if a player has 5 points left, the opponent can win at most 5 actual session points from that hand.
- BB and BTN alternate by hand during a session.
- Fantasyland must be played out as part of the same session, not replaced only by a fixed EV.
- Before a new normal hand starts, if neither player is currently in Fantasyland and the score gap is 40 points or more, the session ends.
- After a session ends, start a new session with both players reset to 200 points.
- At the start of a new session, choose positions randomly again.

This means the final RL return is session-level value, not only one-hand value.

The value function should eventually learn:

- one-hand EV,
- session continuation value,
- stack/score-gap pressure,
- capped-score effects when one player is short on points,
- Fantasyland entry value,
- Fantasyland continuation value,
- and the effect of alternating BB/BTN positions.

Fantasyland EV is the most important target. Fixed FL constants can be used for bootstrapping, but the final system should learn FL value from played-out FL returns.

## Current Model Structure

The current T3 and T4 models are unified models, not separate BB and BTN files.

They receive `is_btn` as part of the 522-dimensional state and learn BB/BTN behavior inside one model.

Current adopted models:

- T4 unified model: `ai/data/t4_oracle_v2/t4_policyvalue_best.pt`
- T3 unified model: `ai/data/t3_oracle_rust_discard_sensitive_ft/t3_policyvalue_v2_best.pt`

Current status:

- T4: trained unified model exists.
- T3: trained unified model exists and was improved with discard-sensitive fine-tuning.
- T2: teacher data generation is running on GCP.
- T1/T0: not yet rebuilt on top of the current T2/T3/T4 stack.

## Important Design Point: BB vs BTN

BTN states are simpler because BTN often acts after BB's current-turn placement is already known.

BB states are harder because BB acts first and the value of a BB action should include the opponent's later response. For BB-specific data, the teacher should evaluate:

1. BB candidate action.
2. Sample or enumerate BTN draw.
3. Let BTN choose the best response using the next-turn oracle.
4. Average the resulting value from BB's perspective.

This is already reflected in the T3 generation design. T2 and T1 should eventually get the same treatment for BB-specific training.

The practical sequence is:

1. Train a unified/simple model first.
2. Evaluate regret and P99 failures.
3. Add BB-specific teacher data if the unified model is weak on BB decisions.

## Bottom-Up Training Plan

### Step 1: T4

T4 is the leaf model near the end of the hand.

The model uses a 27-slot regular-turn action head, even though T4 usually has fewer valid actions. Invalid slots are masked.

T4 should remain the base evaluator for T3 teacher generation until RL value heads become stronger.

### Step 2: T3

T3 is already usable.

Current best T3 model:

- `ai/data/t3_oracle_rust_discard_sensitive_ft/t3_policyvalue_v2_best.pt`

The main known issue after the first T3 fix was high-regret discard mistakes. Hard fine-tuning and discard-sensitive fine-tuning improved P99.

Keep using P99 extraction after each major retrain. The important metrics are:

- mean regret
- median regret
- p90 regret
- p99 regret
- max regret
- top-1 / top-3

### Step 3: T2

T2 teacher data is currently generated with:

- random full deck
- rule/heuristic T0 and T1 placements
- optional non-ace trips branching at T0
- T2 actions evaluated by sampling T3 deals and querying the T3 value oracle

Current T2 data is an initial self-board oracle, not a final BB-specific expectimax teacher.

Use a staged training approach:

1. Train on 0.5M to 2M samples first.
2. Evaluate regret and P99.
3. Train on more samples only if the learning curve still improves.
4. Add hard/P99 fine-tuning if needed.
5. Add BB-specific T2 data if BB decisions are weak.

The large GCP run should be treated as a data pool. It is not necessary to train on the entire pool immediately.

### Step 4: T1

After T2 is good enough, use it as the leaf evaluator for T1.

For each T1 state:

1. Enumerate valid T1 actions.
2. Apply each action.
3. Sample T2 draws.
4. Use the T2 model to estimate continuation EV.
5. Store action EVs and masks.

As with T2, start with a unified/simple teacher and later add BB-specific response modeling.

### Step 5: T0

After T1 is good enough, use it for T0.

T0 has a much larger initial placement action space than T1-T4. Existing T0 solver/PIMC/DFS data can be mixed with the bottom-up labels.

T0 should eventually combine:

- exact or high-quality T0 solver records
- bottom-up T1-evaluated labels
- P99/hard cases
- special branches such as non-ace trips, joker hands, and FL-ready patterns

## Transition to Reinforcement Learning

The final objective is not just to imitate oracle labels. The final objective is full-game EV under actual play.

The bottom-up models should initialize:

- policy heads for action selection
- value heads for state EV
- rollout or search priors

Then self-play RL should fine-tune the full-game policy/value.

Implementation spec for the session environment is tracked in `ai/docs/rl_environment_spec.md`.

The RL/search player should not blindly take only the NN top-1 action. The model should provide priors and value estimates, but the player should still explore alternatives and search for better actions.

Recommended action selection stack:

1. Use the NN policy head to rank candidate actions.
2. Keep several plausible candidates, not only top-1.
3. Use value evaluation, rollout, or shallow search to compare them.
4. Add controlled exploration during self-play so RL can discover improvements over the supervised policy.
5. Preserve exact valid-action masks so invalid or semantically aliased actions are never selected.

Recommended RL sequence:

1. Build a full-game player using T0/T1/T2/T3/T4 models.
2. Run full sessions using the real-game session rules above.
3. Log every decision state, action, legal mask, NN logits, value estimate, score, position, and FL status.
4. Play out Fantasyland hands instead of replacing them only with a constant.
5. Compute capped session return, including royalties, Fantasyland outcomes, remaining-point caps, and session-ending score-gap rules.
6. Train value heads from Monte Carlo return or TD return.
7. Fine-tune policy using advantage-weighted imitation, PPO-style updates, or another stable policy-gradient method.
8. Oversample rare but important states, especially Fantasyland and high-regret cases.

## Fantasyland Handling

Fantasyland EV should not remain a fixed constant in the final system.

During the bottom-up phase, fixed or calibrated FL EV constants are acceptable as bootstrap values. They stabilize early teacher generation and reduce compute.

During RL, FL value should be learned from actual returns:

- FL entry value
- FL continuation value
- different FL hand types
- foul risk while preserving FL
- sacrifice decisions where entering or maintaining FL trades off against immediate row value

The final value model should infer FL value from the state, not from a hard-coded table alone.

Fantasyland should be treated as a first-class state in the RL environment:

- whether each player is in FL,
- FL card count/type,
- whether FL continuation is possible,
- resulting score swings,
- and transition back into normal hands.

Because the session stops only when neither player is in FL and the score gap is at least 40 points, FL can extend a session and materially changes session EV. This makes FL value more important than a one-hand bonus estimate.

Practical approach:

1. Keep FL constants for bootstrapping.
2. In self-play, actually play out FL hands.
3. Store cumulative returns across normal and FL hands.
4. Train the value head to predict those returns.
5. Reduce reliance on fixed FL constants as the learned value improves.

## Evaluation Loop

Every model stage should follow the same loop:

1. Train a baseline model.
2. Evaluate on held-out data.
3. Extract P99 / max-regret samples.
4. Analyze the failure modes.
5. Generate targeted data or oversample failures.
6. Fine-tune.
7. Re-evaluate on the original validation set and a hard validation set.

Avoid judging only top-1 accuracy. For OFC, regret distribution is more important.

Primary metrics:

- mean regret
- median regret
- p90 regret
- p99 regret
- max regret
- zero-regret rate
- low-regret rate
- top-1 / top-3

## Near-Term Next Steps

1. Let the current T2 GCP run finish.
2. Download a small subset first, around 0.5M to 2M samples.
3. Train a first T2 model.
4. Evaluate T2 regret and extract P99.
5. Decide whether to:
   - train on more of the generated pool,
   - add T2 hard cases,
   - add BB-specific T2 data,
   - or move on to T1 teacher generation.
