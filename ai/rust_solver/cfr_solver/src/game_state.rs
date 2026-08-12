//! Game state for OFC Pineapple CFR traversal.
//!
//! Tracks both players' boards, deck, hands, discards, turn, FL status.
//! Provides terminal utility calculation.

use ofc_core::{
    Card, evaluate_3_card,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint, check_fl_entry, compare_5_hands,
};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::action_gen::{Board, Action, get_placed_cards, get_discard, generate_t0_actions, generate_turn_actions};

/// FL Chain EV values — calibrated via imperfect-info rollouts.
/// Formula: Delta = (R_FL - R_N) / (1 - S + p_FL)
/// R_N = 2.850, p_FL = 0.4436 (measured [10,6,3] nesting, 20 hands × 200 rollouts)
pub fn fl_chain_ev(fl_cards: u8) -> f64 {
    match fl_cards {
        14 => 11.79,   // QQ   (S=38.44%)
        15 => 17.20,   // KK   (S=48.69%)
        16 => 26.10,   // AA   (S=64.09%)
        17 => 38.27,   // Trips (S=77.60%)
        _ => 0.0,
    }
}

/// Node type in the game tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeType {
    Player0,
    Player1,
    Chance,
    Terminal,
}

/// Complete game state for CFR traversal.
#[derive(Clone, Debug)]
pub struct GameState {
    pub boards: [Board; 2],
    pub deck: Vec<Card>,
    pub hands: [Vec<Card>; 2],
    pub discards: [Vec<Card>; 2],
    pub turn: u8,
    pub placed: [bool; 2],
    pub btn: u8,
    pub is_fl: [bool; 2],
}

impl GameState {
    /// Create a fresh initial state with a shuffled deck.
    pub fn new_random(rng: &mut impl Rng) -> Self {
        let mut deck = ofc_core::create_deck(true); // 54 cards with jokers
        deck.shuffle(rng);

        let hand0: Vec<Card> = deck.drain(..5).collect();
        let hand1: Vec<Card> = deck.drain(..5).collect();

        GameState {
            boards: [Board::new(), Board::new()],
            deck,
            hands: [hand0, hand1],
            discards: [Vec::new(), Vec::new()],
            turn: 0,
            placed: [false, false],
            btn: 0,
            is_fl: [false, false],
        }
    }

    /// Determine the current node type.
    pub fn node_type(&self) -> NodeType {
        if self.boards[0].is_complete() && self.boards[1].is_complete() {
            return NodeType::Terminal;
        }
        if self.placed[0] && self.placed[1] {
            return NodeType::Chance;
        }
        if !self.placed[0] {
            NodeType::Player0
        } else {
            NodeType::Player1
        }
    }

    /// Get legal actions for a player.
    pub fn get_legal_actions(&self, player: usize) -> Vec<Action> {
        if self.turn == 0 {
            generate_t0_actions(&self.hands[player], &self.boards[player])
        } else {
            generate_turn_actions(&self.hands[player], &self.boards[player])
        }
    }

    /// Apply an action for a player, returning a new state.
    pub fn apply_action(&self, player: usize, action: &Action) -> GameState {
        let mut new_state = self.clone();

        // Apply placements
        let placements = get_placed_cards(&self.hands[player], action);
        for (card, row) in placements {
            new_state.boards[player].row_mut(row).push(card);
        }

        // Handle discard
        if let Some(discard) = get_discard(&self.hands[player], action) {
            new_state.discards[player].push(discard);
        }

        new_state.placed[player] = true;
        new_state.hands[player].clear();

        new_state
    }

    /// Deal cards for the next turn (chance node transition).
    pub fn deal_next_turn(&self, rng: &mut impl Rng) -> GameState {
        let mut new_state = self.clone();
        new_state.turn += 1;
        new_state.placed = [false, false];

        new_state.deck.shuffle(rng);

        if new_state.deck.len() >= 6 {
            new_state.hands[0] = new_state.deck.drain(..3).collect();
            new_state.hands[1] = new_state.deck.drain(..3).collect();
        }

        new_state
    }

    /// Compute terminal utility for a given player.
    pub fn terminal_utility(&self, player: usize) -> f64 {
        let opp = 1 - player;

        let my_board = &self.boards[player];
        let opp_board = &self.boards[opp];

        let my_eval = evaluate_board_with_joker_constraint(
            &my_board.top,
            &my_board.middle,
            &my_board.bottom,
        );
        let opp_eval = evaluate_board_with_joker_constraint(
            &opp_board.top,
            &opp_board.middle,
            &opp_board.bottom,
        );
        let my_bust = my_eval.busted;
        let opp_bust = opp_eval.busted;

        let my_royalties = if my_bust {
            0
        } else {
            get_top_royalty(&my_eval.top)
                + get_middle_royalty(&my_eval.mid)
                + get_bottom_royalty(&my_eval.bot)
        };

        let opp_royalties = if opp_bust {
            0
        } else {
            get_top_royalty(&opp_eval.top)
                + get_middle_royalty(&opp_eval.mid)
                + get_bottom_royalty(&opp_eval.bot)
        };

        let mut score: f64;

        if my_bust && opp_bust {
            score = 0.0;
        } else if my_bust {
            score = -6.0 - opp_royalties as f64;
        } else if opp_bust {
            score = 6.0 + my_royalties as f64;
        } else {
            let mut lines_won = 0i32;
            let mut lines_lost = 0i32;

            // Top (3-card comparison)
            let top_cmp = compare_3_hands(&my_eval.top, &opp_eval.top);
            if top_cmp > 0 { lines_won += 1; }
            else if top_cmp < 0 { lines_lost += 1; }

            // Middle
            let mid_cmp = compare_5_hands(&my_eval.mid, &opp_eval.mid);
            if mid_cmp > 0 { lines_won += 1; }
            else if mid_cmp < 0 { lines_lost += 1; }

            // Bottom
            let bot_cmp = compare_5_hands(&my_eval.bot, &opp_eval.bot);
            if bot_cmp > 0 { lines_won += 1; }
            else if bot_cmp < 0 { lines_lost += 1; }

            let line_score = lines_won - lines_lost;

            let scoop = if lines_won == 3 { 3 }
                       else if lines_lost == 3 { -3 }
                       else { 0 };

            score = (line_score + scoop) as f64 + (my_royalties - opp_royalties) as f64;
        }

        // FL Chain EV bonus (only in normal rounds)
        if !self.is_fl[player] && !my_bust {
            let (fl_qualifies, fl_cards) = check_fl_entry(&my_eval.top);
            if fl_qualifies {
                score += fl_chain_ev(fl_cards);
            }
        }
        if !self.is_fl[opp] && !opp_bust {
            let (opp_fl_qualifies, opp_fl_cards) = check_fl_entry(&opp_eval.top);
            if opp_fl_qualifies {
                score -= fl_chain_ev(opp_fl_cards);
            }
        }

        score
    }

    /// Generate an info set key for a player.
    pub fn info_set_key(&self, player: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        self.turn.hash(&mut hasher);
        (player == self.btn as usize).hash(&mut hasher);

        let board = &self.boards[player];
        hash_sorted_cards(&board.top, &mut hasher);
        hash_sorted_cards(&board.middle, &mut hasher);
        hash_sorted_cards(&board.bottom, &mut hasher);

        hash_sorted_cards(&self.hands[player], &mut hasher);
        hash_sorted_cards(&self.discards[player], &mut hasher);

        hasher.finish()
    }
}

/// Hash a slice of cards in sorted order.
fn hash_sorted_cards(cards: &[Card], hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    let mut keys: Vec<u16> = cards.iter()
        .map(|c| (c.rank as u16) * 10 + c.suit as u16)
        .collect();
    keys.sort_unstable();
    keys.hash(hasher);
}

/// Compare two 3-card top hands.
fn compare_3_hands(a: &[Card], b: &[Card]) -> i32 {
    let (ra, sa) = evaluate_3_card(a);
    let (rb, sb) = evaluate_3_card(b);

    if (ra as u8) != (rb as u8) {
        return if (ra as u8) > (rb as u8) { 1 } else { -1 };
    }
    if sa > sb { 1 } else if sa < sb { -1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    #[test]
    fn terminal_utility_uses_constrained_joker_rows() {
        let hero = Board {
            top: vec![card(12, 0), card(12, 1), card(0, 4)],
            middle: vec![card(13, 0), card(13, 1), card(9, 2), card(8, 3), card(7, 0)],
            bottom: vec![card(14, 0), card(14, 1), card(14, 2), card(5, 3), card(4, 2)],
        };
        let opponent = Board {
            top: vec![card(4, 0), card(3, 1), card(2, 2)],
            middle: vec![card(13, 2), card(11, 1), card(9, 0), card(7, 3), card(5, 2)],
            bottom: vec![card(14, 3), card(12, 2), card(10, 1), card(8, 0), card(6, 3)],
        };
        let state = GameState {
            boards: [hero, opponent],
            deck: Vec::new(),
            hands: [Vec::new(), Vec::new()],
            discards: [Vec::new(), Vec::new()],
            turn: 4,
            placed: [true, true],
            btn: 0,
            is_fl: [false, false],
        };

        // Base score is scoop 6 + QQ royalty 7. QQ also enters 14-card FL.
        assert!((state.terminal_utility(0) - (13.0 + fl_chain_ev(14))).abs() < 1e-9);
    }
}
