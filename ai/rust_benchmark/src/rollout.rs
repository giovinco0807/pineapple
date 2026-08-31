//! RolloutEvaluator for T1+ decisions
//!
//! Hybrid: VN prefilter top-K candidates → N rollouts with BC playout → best EV

use rand::prelude::*;
use std::collections::HashMap;

use crate::types::*;
use crate::encoding::*;
use crate::onnx_models::OnnxModels;
use crate::game_engine::*;

pub struct RolloutConfig {
    pub n_rollouts: usize,          // default: 250 (full rollout mode)
    pub vn_top_k: usize,            // default: 10
    pub bc_temperature: f32,        // default: 0.8
    pub vn_truncate_depth: usize,   // 0 = full rollout (default), 1+ = VN-truncated
    pub vn_truncate_n: usize,       // rollouts per candidate in truncated mode (default: 500)
    pub vn_hybrid: bool,            // true = blend full + VN, false = VN only
    pub value_scale: f32,           // VN denormalization scale (default: 30.0)
}

impl Default for RolloutConfig {
    fn default() -> Self {
        RolloutConfig {
            n_rollouts: 250,
            vn_top_k: 10,
            bc_temperature: 0.8,
            vn_truncate_depth: 0,
            vn_truncate_n: 500,
            vn_hybrid: false,
            value_scale: 30.0,
        }
    }
}

/// Result of rollout evaluation, including per-action EVs for soft-label training
pub struct RolloutResult {
    pub best_action_idx: usize,
    pub n_total_actions: usize,       // total legal actions (before VN prefilter)
    pub top_k_indices: Vec<usize>,    // which actions were evaluated
    pub action_evs: Vec<f64>,         // EV for each top-K action
}

/// Select best T1+ action using VN prefilter + rollout evaluation.
/// If vn_truncate_depth > 0, uses VN-truncated mode instead of full rollout.
pub fn rollout_select_action(
    board: &Board,
    opp_board: &Board,
    dealt_cards: &[CardIdx; 3],
    turn: u8,
    discards: &[CardIdx],
    is_btn: bool,
    models: &mut OnnxModels,
    config: &RolloutConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> RolloutResult {
    let actions = gen_turn_actions(dealt_cards, board);
    if actions.len() <= 1 {
        return RolloutResult {
            best_action_idx: 0,
            n_total_actions: actions.len(),
            top_k_indices: vec![0],
            action_evs: vec![0.0],
        };
    }

    // VN-truncated mode
    if config.vn_truncate_depth > 0 {
        return rollout_select_action_truncated(
            board, opp_board, dealt_cards, turn, discards, is_btn,
            models, config, &actions, fl_ev, rng,
        );
    }

    // VN prefilter: evaluate all candidates, keep top-K
    let top_k_indices = vn_prefilter(
        board, opp_board, &actions, dealt_cards, turn, discards, is_btn, models, config.vn_top_k,
    );

    // Rollout each candidate
    let mut best_idx = top_k_indices[0];
    let mut best_ev = f64::NEG_INFINITY;
    let mut evs: Vec<f64> = Vec::with_capacity(top_k_indices.len());

    for &action_idx in &top_k_indices {
        let action = &actions[action_idx];
        let ev = evaluate_action_rollout(
            board, opp_board, action, dealt_cards, turn, discards, is_btn,
            models, config, fl_ev, rng,
        );
        evs.push(ev);
        if ev > best_ev {
            best_ev = ev;
            best_idx = action_idx;
        }
    }

    RolloutResult {
        best_action_idx: best_idx,
        n_total_actions: actions.len(),
        top_k_indices,
        action_evs: evs,
    }
}

/// VN prefilter: score all candidates with VN, return top-K indices
fn vn_prefilter(
    board: &Board,
    opp_board: &Board,
    actions: &[TurnAction],
    dealt_cards: &[CardIdx; 3],
    turn: u8,
    discards: &[CardIdx],
    is_btn: bool,
    models: &mut OnnxModels,
    top_k: usize,
) -> Vec<usize> {
    if actions.len() <= top_k {
        return (0..actions.len()).collect();
    }

    // Build states for all candidates
    let mut states: Vec<[f32; STATE_DIM]> = Vec::with_capacity(actions.len());
    for action in actions {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board.place_mut(card, row);
        }
        let mut new_discards: Vec<CardIdx> = discards.to_vec();
        new_discards.push(action.discard);

        let obs = Observation {
            board_self: &new_board,
            board_opponent: opp_board,
            dealt_cards: &[],
            known_discards_self: &new_discards,
            turn,
            is_btn,
            chips_self: 200,
            chips_opponent: 200,
        };
        states.push(encode_state(&obs));
    }

    // Batch VN inference
    let state_refs: Vec<&[f32; STATE_DIM]> = states.iter().collect();
    let vn_results = models.vn_inference_batch(&state_refs);

    // Sort by value, take top-K
    let mut scored: Vec<(usize, f32)> = vn_results.iter()
        .enumerate()
        .map(|(i, &(value, _, _, _))| (i, value))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scored.iter().take(top_k).map(|&(idx, _)| idx).collect()
}

/// Evaluate one action by N rollouts
fn evaluate_action_rollout(
    board: &Board,
    opp_board: &Board,
    action: &TurnAction,
    dealt_cards: &[CardIdx; 3],
    turn: u8,
    discards: &[CardIdx],
    is_btn: bool,
    models: &mut OnnxModels,
    config: &RolloutConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> f64 {
    let mut total = 0.0;

    for _ in 0..config.n_rollouts {
        let score = single_rollout(
            board, opp_board, action, dealt_cards, turn, discards, is_btn,
            models, config, fl_ev, rng,
        );
        total += score;
    }

    total / config.n_rollouts as f64
}

/// Single rollout: apply action, then playout remaining turns with BC
fn single_rollout(
    board: &Board,
    opp_board: &Board,
    action: &TurnAction,
    _dealt_cards: &[CardIdx; 3],
    turn: u8,
    discards: &[CardIdx],
    is_btn: bool,
    models: &mut OnnxModels,
    config: &RolloutConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> f64 {
    // Apply action
    let mut my_board = board.clone();
    for &(card, row) in &action.placements {
        my_board.place_mut(card, row);
    }
    let mut my_opp = opp_board.clone();

    // Build unseen card pool
    let mut seen = [false; NUM_CARDS];
    for &c in my_board.all_card_indices().iter() { seen[c as usize] = true; }
    for &c in my_opp.all_card_indices().iter() { seen[c as usize] = true; }
    for &c in discards { seen[c as usize] = true; }
    seen[action.discard as usize] = true;
    // Original dealt cards are also seen (the placed ones already counted, discard counted)

    let mut unseen: Vec<CardIdx> = (0..NUM_CARDS as CardIdx)
        .filter(|&i| !seen[i as usize])
        .collect();
    unseen.shuffle(rng);

    let mut card_idx = 0usize;
    let mut my_discards: Vec<CardIdx> = discards.to_vec();
    my_discards.push(action.discard);
    let mut opp_discards: Vec<CardIdx> = Vec::new();

    // Playout remaining turns
    for t in (turn + 1)..5 {
        // My turn
        if !my_board.is_complete() {
            if card_idx + 3 > unseen.len() { break; }
            let my_cards: [CardIdx; 3] = [
                unseen[card_idx], unseen[card_idx + 1], unseen[card_idx + 2]
            ];
            card_idx += 3;
            bc_playout_turn(
                &mut my_board, &my_opp, &my_cards, t, &mut my_discards, is_btn, models, config, rng,
            );
        }

        // Opponent turn
        if !my_opp.is_complete() {
            if card_idx + 3 > unseen.len() { break; }
            let opp_cards: [CardIdx; 3] = [
                unseen[card_idx], unseen[card_idx + 1], unseen[card_idx + 2]
            ];
            card_idx += 3;
            bc_playout_turn(
                &mut my_opp, &my_board, &opp_cards, t, &mut opp_discards, !is_btn, models, config, rng,
            );
        }
    }

    compute_rollout_score(&my_board, &my_opp, fl_ev)
}

/// VN-truncated rollout evaluation.
/// Play `depth` turns with BC playout, then batch VN eval instead of full playout.
/// If hybrid, also completes full playout and blends 0.5*exact + 0.5*VN.
fn rollout_select_action_truncated(
    board: &Board,
    opp_board: &Board,
    dealt_cards: &[CardIdx; 3],
    turn: u8,
    discards: &[CardIdx],
    is_btn: bool,
    models: &mut OnnxModels,
    config: &RolloutConfig,
    actions: &[TurnAction],
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> RolloutResult {
    // VN prefilter: evaluate all candidates, keep top-K
    let top_k_indices = vn_prefilter(
        board, opp_board, actions, dealt_cards, turn, discards, is_btn, models, config.vn_top_k,
    );

    let n_rollouts = config.vn_truncate_n;
    let depth = config.vn_truncate_depth;
    let hybrid = config.vn_hybrid;
    let scale = config.value_scale as f64;

    // Phase 1: Collect all VN snapshot states (and optional exact scores)
    let mut all_states: Vec<[f32; STATE_DIM]> = Vec::with_capacity(top_k_indices.len() * n_rollouts);
    let mut all_exact: Vec<f64> = Vec::with_capacity(if hybrid { top_k_indices.len() * n_rollouts } else { 0 });
    let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(top_k_indices.len());

    for &action_idx in &top_k_indices {
        let action = &actions[action_idx];
        let start = all_states.len();

        for _ in 0..n_rollouts {
            let (state, exact) = single_rollout_snapshot(
                board, opp_board, action, turn, discards, is_btn,
                models, config, depth, hybrid, fl_ev, rng,
            );
            all_states.push(state);
            if hybrid {
                all_exact.push(exact);
            }
        }

        offsets.push((start, n_rollouts));
    }

    // Phase 2: Batch VN eval all collected states
    let state_refs: Vec<&[f32; STATE_DIM]> = all_states.iter().collect();
    let vn_results = models.vn_inference_batch(&state_refs);

    // Phase 3: Compute EVs per candidate
    let mut best_idx = top_k_indices[0];
    let mut best_ev = f64::NEG_INFINITY;
    let mut evs: Vec<f64> = Vec::with_capacity(top_k_indices.len());

    for (i, &(start, count)) in offsets.iter().enumerate() {
        let mut ev_sum = 0.0;
        for j in start..start + count {
            let vn_val = vn_results[j].0 as f64 * scale;
            if hybrid {
                ev_sum += 0.5 * all_exact[j] + 0.5 * vn_val;
            } else {
                ev_sum += vn_val;
            }
        }
        let ev = ev_sum / count as f64;
        evs.push(ev);
        if ev > best_ev {
            best_ev = ev;
            best_idx = top_k_indices[i];
        }
    }

    RolloutResult {
        best_action_idx: best_idx,
        n_total_actions: actions.len(),
        top_k_indices,
        action_evs: evs,
    }
}

/// Single rollout with VN snapshot at depth.
/// Returns (encoded_state_at_depth, exact_score_if_hybrid).
fn single_rollout_snapshot(
    board: &Board,
    opp_board: &Board,
    action: &TurnAction,
    turn: u8,
    discards: &[CardIdx],
    is_btn: bool,
    models: &mut OnnxModels,
    config: &RolloutConfig,
    depth: usize,
    hybrid: bool,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> ([f32; STATE_DIM], f64) {
    // Apply action
    let mut my_board = board.clone();
    for &(card, row) in &action.placements {
        my_board.place_mut(card, row);
    }
    let mut my_opp = opp_board.clone();

    // Build unseen card pool
    let mut seen = [false; NUM_CARDS];
    for &c in my_board.all_card_indices().iter() { seen[c as usize] = true; }
    for &c in my_opp.all_card_indices().iter() { seen[c as usize] = true; }
    for &c in discards { seen[c as usize] = true; }
    seen[action.discard as usize] = true;

    let mut unseen: Vec<CardIdx> = (0..NUM_CARDS as CardIdx)
        .filter(|&i| !seen[i as usize])
        .collect();
    unseen.shuffle(rng);

    let mut card_idx = 0usize;
    let mut my_discards: Vec<CardIdx> = discards.to_vec();
    my_discards.push(action.discard);
    let mut opp_discards: Vec<CardIdx> = Vec::new();

    let mut turns_played = 0usize;
    let mut snapshot_state: Option<[f32; STATE_DIM]> = None;

    // Playout remaining turns
    for t in (turn + 1)..5 {
        // My turn
        if !my_board.is_complete() {
            if card_idx + 3 > unseen.len() { break; }
            let my_cards: [CardIdx; 3] = [
                unseen[card_idx], unseen[card_idx + 1], unseen[card_idx + 2]
            ];
            card_idx += 3;
            bc_playout_turn(
                &mut my_board, &my_opp, &my_cards, t, &mut my_discards, is_btn, models, config, rng,
            );
        }

        // Opponent turn
        if !my_opp.is_complete() {
            if card_idx + 3 > unseen.len() { break; }
            let opp_cards: [CardIdx; 3] = [
                unseen[card_idx], unseen[card_idx + 1], unseen[card_idx + 2]
            ];
            card_idx += 3;
            bc_playout_turn(
                &mut my_opp, &my_board, &opp_cards, t, &mut opp_discards, !is_btn, models, config, rng,
            );
        }

        turns_played += 1;

        // Take snapshot at depth point
        if turns_played == depth && snapshot_state.is_none() {
            let obs = Observation {
                board_self: &my_board,
                board_opponent: &my_opp,
                dealt_cards: &[],
                known_discards_self: &my_discards,
                turn: t,
                is_btn,
                chips_self: 200,
                chips_opponent: 200,
            };
            snapshot_state = Some(encode_state(&obs));

            if !hybrid {
                break; // truncated mode: stop here
            }
        }
    }

    // If depth was never reached (board completed early), snapshot at current point
    let state = snapshot_state.unwrap_or_else(|| {
        let obs = Observation {
            board_self: &my_board,
            board_opponent: &my_opp,
            dealt_cards: &[],
            known_discards_self: &my_discards,
            turn: turn + turns_played as u8,
            is_btn,
            chips_self: 200,
            chips_opponent: 200,
        };
        encode_state(&obs)
    });

    let exact_score = if hybrid {
        compute_rollout_score(&my_board, &my_opp, fl_ev)
    } else {
        0.0
    };

    (state, exact_score)
}

/// BC-guided playout for one turn
fn bc_playout_turn(
    board: &mut Board,
    opp_board: &Board,
    cards: &[CardIdx; 3],
    turn: u8,
    discards: &mut Vec<CardIdx>,
    is_btn: bool,
    models: &mut OnnxModels,
    config: &RolloutConfig,
    rng: &mut impl Rng,
) {
    if board.is_complete() { return; }

    let actions = gen_turn_actions(cards, board);
    if actions.is_empty() { return; }

    let chosen_idx = if actions.len() == 1 {
        0
    } else {
        // Encode state
        let obs = Observation {
            board_self: board,
            board_opponent: opp_board,
            dealt_cards: cards,
            known_discards_self: discards,
            turn,
            is_btn,
            chips_self: 200,
            chips_opponent: 200,
        };
        let state = encode_state(&obs);
        let logits = models.bc_inference(&state, turn);

        // Temperature sampling
        OnnxModels::bc_select_temperature(&logits, actions.len(), config.bc_temperature, rng)
    };

    let chosen = &actions[chosen_idx];
    for &(card, row) in &chosen.placements {
        board.place_mut(card, row);
    }
    discards.push(chosen.discard);
}
