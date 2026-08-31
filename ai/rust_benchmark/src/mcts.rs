//! MCTS for T0 action selection
//!
//! 2-ply PUCT search matching Python's MultiTurnMCTS:
//!   1. PUCT select T0 action
//!   2. Apply T0 → deal random T1 (3 hero + 3 opp)
//!   3. Opponent plays T1 with BC greedy
//!   4. Batch-eval ALL T1 hero candidates with VN + FL bonus + bust penalty
//!   5. Return MAX value → backprop
//!   6. Select most-visited action

use rand::prelude::*;
use std::collections::HashMap;

use crate::types::*;
use crate::encoding::*;
use crate::onnx_models::OnnxModels;
use crate::game_engine::*;

pub struct MctsConfig {
    pub num_simulations: u32,
    pub c_puct: f32,
    pub bust_penalty: f32,     // Python default: 0.0
    pub fl_ev_scale: f32,      // Python default: 1.0
    pub value_scale: f32,      // Python default: 30.0
    pub score_mean: f64,       // VN denormalization (0.0 if no norm_stats)
    pub score_std: f64,        // VN denormalization (1.0 if no norm_stats)
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            num_simulations: 400,
            c_puct: 1.5,
            bust_penalty: 0.0,
            fl_ev_scale: 1.0,
            value_scale: 30.0,
            score_mean: 0.0,
            score_std: 1.0,
        }
    }
}

/// Result of MCTS search, including visit distribution for soft-label training
pub struct MctsResult {
    pub best_action_idx: usize,
    pub visit_counts: Vec<u32>,
    pub q_values: Vec<f64>,
    pub total_simulations: u32,
}

struct MctsNode {
    visit_count: u32,
    total_value: f64,
    prior: f32,
}

impl MctsNode {
    fn q_value(&self) -> f64 {
        if self.visit_count == 0 { 0.0 }
        else { self.total_value / self.visit_count as f64 }
    }

    fn uct_score(&self, parent_visits: u32, c_puct: f32) -> f64 {
        let explore = c_puct as f64
            * self.prior as f64
            * ((parent_visits as f64).sqrt() / (1.0 + self.visit_count as f64));
        self.q_value() + explore
    }
}

/// Check FL card count from top row cards (matching Python _check_fl_cards)
fn check_fl_cards_from_top(top: &[CardIdx], top_n: u8) -> u8 {
    if top_n < 3 { return 0; }

    let mut ranks: Vec<u8> = Vec::new();  // 0-12 rank values
    let mut jokers: u8 = 0;

    for i in 0..top_n as usize {
        let card = top[i];
        if card >= 52 {
            jokers += 1;
        } else {
            ranks.push(card / 4);  // rank index: 0=2, 10=Q, 11=K, 12=A
        }
    }

    // Check trips: any rank with count + jokers >= 3
    let mut rank_counts: HashMap<u8, u8> = HashMap::new();
    for &r in &ranks {
        *rank_counts.entry(r).or_insert(0) += 1;
    }
    for (_, &count) in &rank_counts {
        if count + jokers >= 3 {
            return 17; // trips
        }
    }

    // Check pairs: Q(10)=14, K(11)=15, A(12)=16
    let rank_to_fl: [(u8, u8); 3] = [(10, 14), (11, 15), (12, 16)]; // Q, K, A
    let mut best: u8 = 0;
    for &(rank_val, fl_val) in &rank_to_fl {
        let count = rank_counts.get(&rank_val).copied().unwrap_or(0);
        if count + jokers >= 2 {
            best = best.max(fl_val);
        }
    }
    best
}

/// FL partial bonus for incomplete top row (matching Python _fl_partial_bonus)
fn fl_partial_bonus(top: &[CardIdx], top_n: u8, fl_ev: &HashMap<u8, f64>) -> f64 {
    if top_n >= 3 || top_n == 0 { return 0.0; }

    let mut ranks: Vec<u8> = Vec::new();
    let mut jokers: u8 = 0;
    for i in 0..top_n as usize {
        let card = top[i];
        if card >= 52 { jokers += 1; }
        else { ranks.push(card / 4); }  // rank index: 0=2, 10=Q, 11=K, 12=A
    }

    let rank_to_fl: [(u8, u8); 3] = [(10, 14), (11, 15), (12, 16)]; // Q=10, K=11, A=12

    if top_n == 1 {
        // 1 card, 2 slots: ~20-25% chance to pair
        if jokers == 1 {
            return fl_ev.get(&16).copied().unwrap_or(52.4) * 0.25;
        }
        for &(rank_val, fl_val) in &rank_to_fl {
            if ranks.contains(&rank_val) {
                return fl_ev.get(&fl_val).copied().unwrap_or(0.0) * 0.20;
            }
        }
        return 0.0;
    }

    if top_n == 2 {
        // 2 cards, 1 slot
        let mut rank_counts: HashMap<u8, u8> = HashMap::new();
        for &r in &ranks { *rank_counts.entry(r).or_insert(0) += 1; }

        // Already have a pair?
        for &(rank_val, fl_val) in &rank_to_fl {
            let count = rank_counts.get(&rank_val).copied().unwrap_or(0);
            if count >= 2 {
                return fl_ev.get(&fl_val).copied().unwrap_or(0.0) * 0.85;
            }
        }

        // Joker + high card?
        if jokers >= 1 && !ranks.is_empty() {
            for &(rank_val, fl_val) in &rank_to_fl {
                if ranks.contains(&rank_val) {
                    return fl_ev.get(&fl_val).copied().unwrap_or(0.0) * 0.85;
                }
            }
            return 0.0;
        }

        // Two jokers
        if jokers >= 2 {
            return fl_ev.get(&16).copied().unwrap_or(52.4) * 0.85;
        }

        // Two different cards, 1 slot: ~8% chance
        let mut bonus = 0.0f64;
        for &(rank_val, fl_val) in &rank_to_fl {
            if ranks.contains(&rank_val) {
                let ev = fl_ev.get(&fl_val).copied().unwrap_or(0.0) * 0.08;
                if ev > bonus { bonus = ev; }
            }
        }
        return bonus;
    }

    0.0
}

/// Evaluate a post-action board with VN + FL bonus + bust penalty
/// Returns value normalized to [-1, 1] (matching Python)
fn vn_evaluate_board(
    board: &Board,
    opp_board: &Board,
    discards: &[CardIdx],
    turn: u8,
    is_btn: bool,
    models: &mut OnnxModels,
    config: &MctsConfig,
    fl_ev: &HashMap<u8, f64>,
) -> f64 {
    let obs = Observation {
        board_self: board,
        board_opponent: opp_board,
        dealt_cards: &[],
        known_discards_self: discards,
        turn,
        is_btn,
        chips_self: 200,
        chips_opponent: 200,
    };
    let state = encode_state(&obs);
    let (value, bust_prob, _fl_prob, _royalty) = models.vn_inference(&state);

    // Denormalize
    let mut score = value as f64 * config.score_std + config.score_mean;

    // Bust penalty
    score -= bust_prob as f64 * config.bust_penalty as f64;

    // FL bonus
    let fl_cards = check_fl_cards_from_top(&board.top, board.top_n);
    if fl_cards > 0 {
        score += fl_ev.get(&fl_cards).copied().unwrap_or(0.0) * config.fl_ev_scale as f64;
    } else {
        score += fl_partial_bonus(&board.top, board.top_n, fl_ev) * config.fl_ev_scale as f64;
    }

    // Normalize to [-1, 1]
    (score / config.value_scale as f64).clamp(-1.0, 1.0)
}

/// 2-ply evaluation: apply T0 action → random T1 deal → opp BC T1 → batch VN eval → MAX
fn evaluate_2ply(
    board: &Board,
    opp_board: &Board,
    t0_action: &T0Action,
    is_btn: bool,
    unseen: &[CardIdx],
    models: &mut OnnxModels,
    config: &MctsConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> f64 {
    // 1. Apply T0 action
    let mut post_board = board.clone();
    for &(card, row) in t0_action {
        post_board.place_mut(card, row);
    }

    // 2. Shuffle unseen, deal T1
    let mut shuffled = unseen.to_vec();
    shuffled.shuffle(rng);

    let opp_complete = opp_board.is_complete();
    let cards_needed = if opp_complete { 3 } else { 6 };
    if shuffled.len() < cards_needed {
        // Not enough cards, evaluate post-T0 directly
        return vn_evaluate_board(
            &post_board, opp_board, &[], 0, is_btn, models, config, fl_ev,
        );
    }

    let my_t1_cards: [CardIdx; 3] = [shuffled[0], shuffled[1], shuffled[2]];

    // 3. Opponent plays T1 with BC greedy
    let mut post_opp = opp_board.clone();
    if !opp_complete {
        let opp_t1_cards: [CardIdx; 3] = [shuffled[3], shuffled[4], shuffled[5]];
        bc_opponent_turn(
            &mut post_opp, &post_board, &opp_t1_cards, 1, is_btn, models,
        );
    }

    // 4. Generate T1 hero candidates
    let t1_actions = gen_turn_actions(&my_t1_cards, &post_board);
    if t1_actions.is_empty() {
        return vn_evaluate_board(
            &post_board, &post_opp, &[], 1, is_btn, models, config, fl_ev,
        );
    }

    if t1_actions.len() == 1 {
        let mut test_board = post_board.clone();
        for &(card, row) in &t1_actions[0].placements {
            test_board.place_mut(card, row);
        }
        let disc = vec![t1_actions[0].discard];
        return vn_evaluate_board(
            &test_board, &post_opp, &disc, 1, is_btn, models, config, fl_ev,
        );
    }

    // 5. Batch-evaluate ALL T1 candidates with VN
    let mut states: Vec<[f32; STATE_DIM]> = Vec::with_capacity(t1_actions.len());
    let mut t1_boards: Vec<Board> = Vec::with_capacity(t1_actions.len());

    for action in &t1_actions {
        let mut test_board = post_board.clone();
        for &(card, row) in &action.placements {
            test_board.place_mut(card, row);
        }
        t1_boards.push(test_board.clone());

        let disc = vec![action.discard];
        let obs = Observation {
            board_self: &test_board,
            board_opponent: &post_opp,
            dealt_cards: &[],
            known_discards_self: &disc,
            turn: 1,
            is_btn,
            chips_self: 200,
            chips_opponent: 200,
        };
        states.push(encode_state(&obs));
    }

    // Batch VN inference
    let state_refs: Vec<&[f32; STATE_DIM]> = states.iter().collect();
    let vn_results = models.vn_inference_batch(&state_refs);

    // Score each candidate: denormalize + bust penalty + FL bonus
    let mut best_value = f64::NEG_INFINITY;
    for (i, &(value, bust_prob, _fl_prob, _royalty)) in vn_results.iter().enumerate() {
        let mut score = value as f64 * config.score_std + config.score_mean;
        score -= bust_prob as f64 * config.bust_penalty as f64;

        let fl_cards = check_fl_cards_from_top(&t1_boards[i].top, t1_boards[i].top_n);
        if fl_cards > 0 {
            score += fl_ev.get(&fl_cards).copied().unwrap_or(0.0) * config.fl_ev_scale as f64;
        } else {
            score += fl_partial_bonus(&t1_boards[i].top, t1_boards[i].top_n, fl_ev)
                * config.fl_ev_scale as f64;
        }

        let normalized = (score / config.value_scale as f64).clamp(-1.0, 1.0);
        if normalized > best_value {
            best_value = normalized;
        }
    }

    best_value
}

/// Opponent plays one turn using BC greedy
fn bc_opponent_turn(
    opp_board: &mut Board,
    my_board: &Board,
    cards: &[CardIdx; 3],
    turn: u8,
    is_btn: bool,
    models: &mut OnnxModels,
) {
    if opp_board.is_complete() { return; }

    let actions = gen_turn_actions(cards, opp_board);
    if actions.is_empty() { return; }

    let chosen_idx = if actions.len() == 1 {
        0
    } else {
        let obs = Observation {
            board_self: opp_board,
            board_opponent: my_board,
            dealt_cards: cards,
            known_discards_self: &[],
            turn,
            is_btn: !is_btn,
            chips_self: 200,
            chips_opponent: 200,
        };
        let state = encode_state(&obs);
        let logits = models.bc_inference(&state, turn);
        OnnxModels::bc_select_greedy(&logits, actions.len())
    };

    let chosen = &actions[chosen_idx];
    for &(card, row) in &chosen.placements {
        opp_board.place_mut(card, row);
    }
}

/// Select T0 action using 2-ply MCTS with PUCT.
///
/// Matches Python's MultiTurnMCTS._search_t0:
/// 1. BC softmax priors for all T0 actions
/// 2. PUCT select → 2-ply eval (T0 → random T1 → opp BC → batch VN MAX)
/// 3. Backprop normalized value
/// 4. Return most-visited action
pub fn mcts_select_t0_action(
    board: &Board,
    opp_board: &Board,
    dealt_cards: &[CardIdx; 5],
    is_btn: bool,
    models: &mut OnnxModels,
    config: &MctsConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut impl Rng,
) -> MctsResult {
    let actions = gen_t0_actions(dealt_cards, board);
    if actions.len() <= 1 {
        return MctsResult {
            best_action_idx: 0,
            visit_counts: vec![config.num_simulations],
            q_values: vec![0.0],
            total_simulations: config.num_simulations,
        };
    }

    // Get BC priors (softmax)
    let obs = Observation {
        board_self: board,
        board_opponent: opp_board,
        dealt_cards: dealt_cards,
        known_discards_self: &[],
        turn: 0,
        is_btn,
        chips_self: 200,
        chips_opponent: 200,
    };
    let state = encode_state(&obs);
    let logits = models.bc_inference(&state, 0);

    let n_actions = actions.len();
    let max_logit = logits[..n_actions].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut priors: Vec<f32> = logits[..n_actions].iter()
        .map(|&l| (l - max_logit).exp())
        .collect();
    let sum: f32 = priors.iter().sum();
    for p in priors.iter_mut() { *p /= sum; }

    // Initialize nodes
    let mut nodes: Vec<MctsNode> = (0..n_actions).map(|i| MctsNode {
        visit_count: 0,
        total_value: 0.0,
        prior: priors[i],
    }).collect();

    // Build unseen pool
    let mut seen = [false; NUM_CARDS];
    for &c in dealt_cards { seen[c as usize] = true; }
    for &c in opp_board.all_card_indices().iter() { seen[c as usize] = true; }
    let unseen: Vec<CardIdx> = (0..NUM_CARDS as CardIdx)
        .filter(|&i| !seen[i as usize])
        .collect();

    // Run simulations
    for _ in 0..config.num_simulations {
        // PUCT select
        let parent_n: u32 = nodes.iter().map(|n| n.visit_count).sum();
        let selected = nodes.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.uct_score(parent_n, config.c_puct)
                    .partial_cmp(&b.uct_score(parent_n, config.c_puct))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        // 2-ply evaluation
        let value = evaluate_2ply(
            board, opp_board, &actions[selected], is_btn,
            &unseen, models, config, fl_ev, rng,
        );

        // Backpropagate
        nodes[selected].visit_count += 1;
        nodes[selected].total_value += value;
    }

    // Return most-visited action with full result
    let best = nodes.iter()
        .enumerate()
        .max_by_key(|(_, n)| n.visit_count)
        .map(|(i, _)| i)
        .unwrap();

    MctsResult {
        best_action_idx: best,
        visit_counts: nodes.iter().map(|n| n.visit_count).collect(),
        q_values: nodes.iter().map(|n| n.q_value()).collect(),
        total_simulations: config.num_simulations,
    }
}
