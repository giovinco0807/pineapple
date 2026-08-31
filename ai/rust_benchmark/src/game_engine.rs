//! OFC Pineapple Game Engine
//!
//! Two-player game loop with deck management, scoring, and FL detection.

use ofc_core::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::types::*;

/// FL Expected Values (chain formula)
pub fn default_fl_ev() -> HashMap<u8, f64> {
    let mut m = HashMap::new();
    m.insert(14, 14.0);  // QQ
    m.insert(15, 27.9);  // KK
    m.insert(16, 52.4);  // AA
    m.insert(17, 104.5); // Trips
    m
}

/// Full game state for a 2-player game
pub struct GameState {
    pub deck: Vec<CardIdx>,
    pub boards: [Board; 2],
    pub discards: [Vec<CardIdx>; 2],
    pub dealt_cards: [Vec<CardIdx>; 2],
    pub deck_idx: usize,
    pub turn: u8,
    pub btn: u8, // button player (acts first)
}

impl GameState {
    /// Create new game with given shuffled deck
    pub fn new(deck: Vec<CardIdx>, btn: u8) -> Self {
        GameState {
            deck,
            boards: [Board::new(), Board::new()],
            discards: [Vec::new(), Vec::new()],
            dealt_cards: [Vec::new(), Vec::new()],
            deck_idx: 0,
            turn: 0,
            btn,
        }
    }

    /// Deal initial 5 cards to each player
    pub fn deal_initial(&mut self) {
        // Button player gets first 5, other gets next 5
        let first = self.btn as usize;
        let second = 1 - first;
        self.dealt_cards[first] = self.deck[0..5].to_vec();
        self.dealt_cards[second] = self.deck[5..10].to_vec();
        self.deck_idx = 10;
    }

    /// Deal 3 cards for turn (T1-T4)
    pub fn deal_turn(&mut self) {
        let first = self.btn as usize;
        let second = 1 - first;
        self.dealt_cards[first] = self.deck[self.deck_idx..self.deck_idx + 3].to_vec();
        self.dealt_cards[second] = self.deck[self.deck_idx + 3..self.deck_idx + 6].to_vec();
        self.deck_idx += 6;
        self.turn += 1;
    }

    /// Apply T0 action for a player
    pub fn apply_t0_action(&mut self, seat: usize, action: &T0Action) {
        for &(card, row) in action {
            self.boards[seat].place_mut(card, row);
        }
        self.dealt_cards[seat].clear();
    }

    /// Apply turn action for a player
    pub fn apply_turn_action(&mut self, seat: usize, action: &TurnAction) {
        for &(card, row) in &action.placements {
            self.boards[seat].place_mut(card, row);
        }
        self.discards[seat].push(action.discard);
        self.dealt_cards[seat].clear();
    }

    pub fn is_complete(&self) -> bool {
        self.boards[0].is_complete() && self.boards[1].is_complete()
    }
}

/// Game result from one player's perspective
#[derive(Clone)]
pub struct GameResult {
    pub busted: [bool; 2],
    pub fl_entry: [bool; 2],
    pub fl_card_count: [u8; 2],
    pub fl_stay: [bool; 2],  // FL stay conditions met (top trips OR bot quads+)
    pub royalties: [i32; 2],
    pub raw_score: f64,      // from hero perspective
    pub fl_type: Option<String>,
}

/// Compute final score between two completed boards (from hero=0's perspective).
/// Uses joker bust-prevention: jokers pick strongest hand that doesn't bust.
pub fn compute_game_result(boards: &[Board; 2], fl_ev: &HashMap<u8, f64>) -> GameResult {
    // Evaluate with joker bust-prevention (bottom-up constraint)
    let eval: [BoardEval; 2] = [
        evaluate_board_with_joker_constraint(
            &boards[0].top_cards(), &boards[0].mid_cards(), &boards[0].bot_cards(),
        ),
        evaluate_board_with_joker_constraint(
            &boards[1].top_cards(), &boards[1].mid_cards(), &boards[1].bot_cards(),
        ),
    ];

    let busted = [eval[0].busted, eval[1].busted];

    // Use constrained cards for all downstream evaluation
    let top = [&eval[0].top, &eval[1].top];
    let mid = [&eval[0].mid, &eval[1].mid];
    let bot = [&eval[0].bot, &eval[1].bot];

    // Royalties
    let mut royalties = [0i32; 2];
    for i in 0..2 {
        if !busted[i] {
            royalties[i] = get_top_royalty(top[i])
                + get_middle_royalty(mid[i])
                + get_bottom_royalty(bot[i]);
        }
    }

    // FL entry (from constrained top)
    let mut fl_entry = [false; 2];
    let mut fl_card_count = [0u8; 2];
    for i in 0..2 {
        if !busted[i] {
            let (entry, cards) = check_fl_entry(top[i]);
            fl_entry[i] = entry;
            fl_card_count[i] = cards;
        }
    }

    // FL stay (top trips OR bot quads+) from constrained cards
    let mut fl_stay = [false; 2];
    for i in 0..2 {
        if !busted[i] {
            fl_stay[i] = check_fl_stay(top[i], mid[i], bot[i]);
        }
    }

    let fl_type = if fl_entry[0] {
        Some(match fl_card_count[0] {
            17 => "trips".to_string(),
            16 => "AA".to_string(),
            15 => "KK".to_string(),
            14 => "QQ".to_string(),
            _ => "unknown".to_string(),
        })
    } else {
        None
    };

    // Score
    let raw_score = if busted[0] && busted[1] {
        0.0
    } else if busted[0] {
        -(6.0 + royalties[1] as f64)
    } else if busted[1] {
        6.0 + royalties[0] as f64
    } else {
        let mut line_total = 0i32;
        // Top (compare constrained 3-card hands)
        let top_val = [evaluate_3_card(top[0]), evaluate_3_card(top[1])];
        match compare_3_card_vals(&top_val[0], &top_val[1]) {
            1 => line_total += 1,
            -1 => line_total -= 1,
            _ => {}
        }
        // Mid
        if compare_5_hands(mid[0], mid[1]) > 0 { line_total += 1; }
        else if compare_5_hands(mid[0], mid[1]) < 0 { line_total -= 1; }
        // Bot
        if compare_5_hands(bot[0], bot[1]) > 0 { line_total += 1; }
        else if compare_5_hands(bot[0], bot[1]) < 0 { line_total -= 1; }

        let mut score = line_total as f64;
        if line_total.abs() == 3 {
            score += if line_total > 0 { 3.0 } else { -3.0 };
        }
        score += (royalties[0] - royalties[1]) as f64;
        score
    };

    GameResult {
        busted,
        fl_entry,
        fl_card_count,
        fl_stay,
        royalties,
        raw_score,
        fl_type,
    }
}

fn compare_3_card_vals(a: &(HandRank3, u32), b: &(HandRank3, u32)) -> i32 {
    if (a.0 as u8) != (b.0 as u8) {
        return if (a.0 as u8) > (b.0 as u8) { 1 } else { -1 };
    }
    if a.1 > b.1 { 1 } else if a.1 < b.1 { -1 } else { 0 }
}

/// Score computation for rollout.
/// Uses joker bust-prevention for correct scoring.
pub fn compute_rollout_score(
    my_board: &Board,
    opp_board: &Board,
    fl_ev: &HashMap<u8, f64>,
) -> f64 {
    let my_eval = evaluate_board_with_joker_constraint(
        &my_board.top_cards(), &my_board.mid_cards(), &my_board.bot_cards(),
    );
    let opp_eval = evaluate_board_with_joker_constraint(
        &opp_board.top_cards(), &opp_board.mid_cards(), &opp_board.bot_cards(),
    );

    let my_busted = my_eval.busted;
    let opp_busted = opp_eval.busted;

    let mut my_royalty = 0i32;
    let mut opp_royalty = 0i32;
    if !my_busted {
        my_royalty = get_top_royalty(&my_eval.top)
            + get_middle_royalty(&my_eval.mid)
            + get_bottom_royalty(&my_eval.bot);
    }
    if !opp_busted {
        opp_royalty = get_top_royalty(&opp_eval.top)
            + get_middle_royalty(&opp_eval.mid)
            + get_bottom_royalty(&opp_eval.bot);
    }

    if my_busted && opp_busted { return 0.0; }
    if my_busted { return -(6.0 + opp_royalty as f64); }
    if opp_busted { return 6.0 + my_royalty as f64; }

    // Line comparison (constrained cards)
    let mut line_total = 0i32;
    let my_tv = evaluate_3_card(&my_eval.top);
    let opp_tv = evaluate_3_card(&opp_eval.top);
    match compare_3_card_vals(&my_tv, &opp_tv) {
        1 => line_total += 1,
        -1 => line_total -= 1,
        _ => {}
    }
    if compare_5_hands(&my_eval.mid, &opp_eval.mid) > 0 { line_total += 1; }
    else if compare_5_hands(&my_eval.mid, &opp_eval.mid) < 0 { line_total -= 1; }
    if compare_5_hands(&my_eval.bot, &opp_eval.bot) > 0 { line_total += 1; }
    else if compare_5_hands(&my_eval.bot, &opp_eval.bot) < 0 { line_total -= 1; }

    let scoop = if line_total.abs() == 3 { 3.0 * line_total.signum() as f64 } else { 0.0 };
    let mut score = line_total as f64 + scoop + (my_royalty - opp_royalty) as f64;

    // FL EV (from constrained top)
    let (my_fl, my_fl_cards) = check_fl_entry(&my_eval.top);
    let (opp_fl, opp_fl_cards) = check_fl_entry(&opp_eval.top);
    if my_fl && !my_busted {
        score += fl_ev.get(&my_fl_cards).copied().unwrap_or(0.0);
    }
    if opp_fl && !opp_busted {
        score -= fl_ev.get(&opp_fl_cards).copied().unwrap_or(0.0);
    }

    score
}

/// Create a full 54-card deck as CardIdx array
pub fn create_full_deck() -> Vec<CardIdx> {
    (0..54).collect()
}
