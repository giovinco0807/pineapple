//! 520-dimensional state encoding matching Python's ai/engine/encoding.py exactly.
//!
//! Layout:
//!   54 cards × 9 locations = 486 dims (card matrix)
//!   4 meta dims
//!   30 game-aware dims
//!   Total = 520

use crate::types::*;
use std::collections::HashMap;

pub const STATE_DIM: usize = 520;

// Location indices in the 9-dim one-hot vector per card
const LOC_MY_TOP: usize = 0;
const LOC_MY_MID: usize = 1;
const LOC_MY_BOT: usize = 2;
const LOC_OPP_TOP: usize = 3;
const LOC_OPP_MID: usize = 4;
const LOC_OPP_BOT: usize = 5;
const LOC_IN_HAND: usize = 6;
const LOC_MY_DISCARD: usize = 7;
const LOC_UNSEEN: usize = 8;

/// Game observation for encoding
pub struct Observation<'a> {
    pub board_self: &'a Board,
    pub board_opponent: &'a Board,
    pub dealt_cards: &'a [CardIdx],
    pub known_discards_self: &'a [CardIdx],
    pub turn: u8,
    pub is_btn: bool,
    pub chips_self: u16,
    pub chips_opponent: u16,
}

/// Encode game state into 520-dim float vector.
/// Must exactly match Python's encode_state() output.
pub fn encode_state(obs: &Observation) -> [f32; STATE_DIM] {
    let mut state = [0.0f32; STATE_DIM];

    // ─── Card matrix (54 × 9 = 486 dims) ─────────────────────────
    // Each card gets a 9-dim one-hot at offset card_idx * 9

    // Self board
    for i in 0..obs.board_self.top_n as usize {
        let idx = obs.board_self.top[i] as usize;
        state[idx * 9 + LOC_MY_TOP] = 1.0;
    }
    for i in 0..obs.board_self.mid_n as usize {
        let idx = obs.board_self.mid[i] as usize;
        state[idx * 9 + LOC_MY_MID] = 1.0;
    }
    for i in 0..obs.board_self.bot_n as usize {
        let idx = obs.board_self.bot[i] as usize;
        state[idx * 9 + LOC_MY_BOT] = 1.0;
    }

    // Opponent board
    for i in 0..obs.board_opponent.top_n as usize {
        let idx = obs.board_opponent.top[i] as usize;
        state[idx * 9 + LOC_OPP_TOP] = 1.0;
    }
    for i in 0..obs.board_opponent.mid_n as usize {
        let idx = obs.board_opponent.mid[i] as usize;
        state[idx * 9 + LOC_OPP_MID] = 1.0;
    }
    for i in 0..obs.board_opponent.bot_n as usize {
        let idx = obs.board_opponent.bot[i] as usize;
        state[idx * 9 + LOC_OPP_BOT] = 1.0;
    }

    // Hand
    for &c in obs.dealt_cards {
        state[c as usize * 9 + LOC_IN_HAND] = 1.0;
    }

    // Own discards
    for &c in obs.known_discards_self {
        state[c as usize * 9 + LOC_MY_DISCARD] = 1.0;
    }

    // Unseen: any card not placed in any of the above locations
    let mut seen = [false; NUM_CARDS];
    for i in 0..obs.board_self.top_n as usize { seen[obs.board_self.top[i] as usize] = true; }
    for i in 0..obs.board_self.mid_n as usize { seen[obs.board_self.mid[i] as usize] = true; }
    for i in 0..obs.board_self.bot_n as usize { seen[obs.board_self.bot[i] as usize] = true; }
    for i in 0..obs.board_opponent.top_n as usize { seen[obs.board_opponent.top[i] as usize] = true; }
    for i in 0..obs.board_opponent.mid_n as usize { seen[obs.board_opponent.mid[i] as usize] = true; }
    for i in 0..obs.board_opponent.bot_n as usize { seen[obs.board_opponent.bot[i] as usize] = true; }
    for &c in obs.dealt_cards { seen[c as usize] = true; }
    for &c in obs.known_discards_self { seen[c as usize] = true; }
    for i in 0..NUM_CARDS {
        if !seen[i] {
            state[i * 9 + LOC_UNSEEN] = 1.0;
        }
    }

    // ─── Meta features (4 dims at offset 486) ────────────────────
    let base = 486;
    state[base] = obs.turn as f32 / 4.0;
    state[base + 1] = if obs.is_btn { 1.0 } else { 0.0 };
    state[base + 2] = obs.chips_self as f32 / 200.0;
    state[base + 3] = obs.chips_opponent as f32 / 200.0;

    // ─── Game-aware features (30 dims at offset 490) ─────────────
    let ga = 490;

    // Row slots remaining (6 dims)
    let bs = obs.board_self;
    let bo = obs.board_opponent;
    state[ga] = (3 - bs.top_n) as f32 / 3.0;
    state[ga + 1] = (5 - bs.mid_n) as f32 / 5.0;
    state[ga + 2] = (5 - bs.bot_n) as f32 / 5.0;
    state[ga + 3] = (3 - bo.top_n) as f32 / 3.0;
    state[ga + 4] = (5 - bo.mid_n) as f32 / 5.0;
    state[ga + 5] = (5 - bo.bot_n) as f32 / 5.0;

    // FL features - self (6 dims)
    let fl_self = fl_features(&bs.top[..bs.top_n as usize]);
    for i in 0..6 { state[ga + 6 + i] = fl_self[i]; }

    // FL features - opponent (4 dims: has_Q, has_K, has_A, fl_ready)
    let fl_opp = fl_features(&bo.top[..bo.top_n as usize]);
    state[ga + 12] = fl_opp[0]; // has_Q
    state[ga + 13] = fl_opp[1]; // has_K
    state[ga + 14] = fl_opp[2]; // has_A
    state[ga + 15] = fl_opp[5]; // fl_ready

    // Hand rank features (6 dims)
    state[ga + 16] = row_rank_numeric(&bs.top[..bs.top_n as usize], 3);
    state[ga + 17] = row_rank_numeric(&bs.mid[..bs.mid_n as usize], 5);
    state[ga + 18] = row_rank_numeric(&bs.bot[..bs.bot_n as usize], 5);
    state[ga + 19] = row_rank_numeric(&bo.top[..bo.top_n as usize], 3);
    state[ga + 20] = row_rank_numeric(&bo.mid[..bo.mid_n as usize], 5);
    state[ga + 21] = row_rank_numeric(&bo.bot[..bo.bot_n as usize], 5);

    // Bust risk (2 dims)
    state[ga + 22] = if state[ga + 16] > state[ga + 17]
        && bs.top_n > 0 && bs.mid_n > 0 { 1.0 } else { 0.0 };
    state[ga + 23] = if state[ga + 19] > state[ga + 20]
        && bo.top_n > 0 && bo.mid_n > 0 { 1.0 } else { 0.0 };

    // Draw features (6 dims)
    let draw_mid = draw_features(&bs.mid[..bs.mid_n as usize]);
    let draw_bot = draw_features(&bs.bot[..bs.bot_n as usize]);
    state[ga + 24] = draw_mid[0]; // flush_draw_mid
    state[ga + 25] = draw_mid[1]; // straight_potential_mid
    state[ga + 26] = draw_mid[2]; // pair_count_mid
    state[ga + 27] = draw_bot[0]; // flush_draw_bot
    state[ga + 28] = draw_bot[1]; // straight_potential_bot
    state[ga + 29] = draw_bot[2]; // pair_count_bot

    state
}

// ─── Helper functions matching Python exactly ──────────────────

/// Get rank value (0-12) from CardIdx. Returns -1 for jokers.
fn card_rank(idx: CardIdx) -> i8 {
    if cardidx_is_joker(idx) { return -1; }
    (idx % 13) as i8
}

/// Get suit (0-3) from CardIdx. Returns 255 for jokers.
fn card_suit(idx: CardIdx) -> u8 {
    if cardidx_is_joker(idx) { return 255; }
    idx / 13
}

/// Python RANK_VALUES: 2=0, 3=1, ..., A=12
/// Q=10, K=11, A=12
const RANK_Q: i8 = 10;
const RANK_K: i8 = 11;
const RANK_A: i8 = 12;

/// FL features from top row cards (6 dims).
/// [has_Q, has_K, has_A, has_pair, pair_rank/14, fl_ready]
fn fl_features(top_cards: &[CardIdx]) -> [f32; 6] {
    if top_cards.is_empty() {
        return [0.0; 6];
    }

    let mut ranks: Vec<i8> = Vec::new();
    let mut n_jokers = 0u8;
    for &c in top_cards {
        let r = card_rank(c);
        if r >= 0 {
            ranks.push(r);
        } else {
            n_jokers += 1;
        }
    }

    if ranks.is_empty() && n_jokers == 0 {
        return [0.0; 6];
    }

    let has_q = if ranks.contains(&RANK_Q) { 1.0 } else { 0.0 };
    let has_k = if ranks.contains(&RANK_K) { 1.0 } else { 0.0 };
    let mut has_a = if ranks.contains(&RANK_A) { 1.0 } else { 0.0 };

    // Count ranks
    let mut rank_counts: HashMap<i8, u8> = HashMap::new();
    for &r in &ranks {
        *rank_counts.entry(r).or_insert(0) += 1;
    }

    // With jokers, boost counts
    let mut effective_counts = rank_counts.clone();
    let mut jokers_left = n_jokers;

    // Boost existing cards (highest first) to form pairs
    let mut sorted_keys: Vec<i8> = effective_counts.keys().cloned().collect();
    sorted_keys.sort_unstable_by(|a, b| b.cmp(a));
    for &r in &sorted_keys {
        if jokers_left == 0 { break; }
        if effective_counts[&r] == 1 {
            *effective_counts.get_mut(&r).unwrap() = 2;
            jokers_left -= 1;
        }
    }

    // If jokers still left and no real cards
    if jokers_left > 0 && ranks.is_empty() {
        effective_counts.insert(RANK_A, jokers_left.min(2));
        has_a = 1.0;
    } else if jokers_left > 0 && !ranks.is_empty() {
        let best_r = *effective_counts.keys().max().unwrap();
        *effective_counts.get_mut(&best_r).unwrap() += jokers_left;
    }

    // Find pairs
    let pairs: Vec<(i8, u8)> = effective_counts.iter()
        .filter(|(_, &c)| c >= 2)
        .map(|(&r, &c)| (r, c))
        .collect();

    let has_pair = if !pairs.is_empty() { 1.0 } else { 0.0 };
    let pair_rank = if !pairs.is_empty() {
        pairs.iter().map(|(r, _)| *r).max().unwrap() as f32 / 14.0
    } else {
        0.0
    };

    // FL ready: pair of QQ+ or trips QQ+
    let mut fl_ready = 0.0;
    if !pairs.is_empty() {
        let best_pair = pairs.iter().map(|(r, _)| *r).max().unwrap();
        if best_pair >= RANK_Q {
            fl_ready = 1.0;
        }
    }
    let trips: Vec<i8> = effective_counts.iter()
        .filter(|(_, &c)| c >= 3)
        .map(|(&r, _)| r)
        .collect();
    if !trips.is_empty() && *trips.iter().max().unwrap() >= RANK_Q {
        fl_ready = 1.0;
    }

    [has_q, has_k, has_a, has_pair, pair_rank, fl_ready]
}

/// Compute normalized hand rank (0.0-1.0) for a row.
fn row_rank_numeric(cards: &[CardIdx], expected_size: u8) -> f32 {
    if cards.is_empty() { return 0.0; }

    let mut ranks: Vec<i8> = cards.iter()
        .filter(|&&c| !cardidx_is_joker(c))
        .map(|&c| card_rank(c))
        .collect();
    if ranks.is_empty() { return 0.0; }

    ranks.sort_unstable_by(|a, b| b.cmp(a));

    let mut rank_counts: HashMap<i8, u8> = HashMap::new();
    for &r in &ranks {
        *rank_counts.entry(r).or_insert(0) += 1;
    }
    let mut counts: Vec<u8> = rank_counts.values().cloned().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));

    if expected_size == 3 {
        let score = if counts[0] >= 3 {
            8.0
        } else if counts[0] >= 2 {
            4.0 + ranks[0] as f32 / 12.0
        } else {
            *ranks.iter().max().unwrap() as f32 / 12.0
        };
        (score / 9.0).min(1.0)
    } else {
        // 5-card row
        let real_cards_with_suits: Vec<(i8, u8)> = cards.iter()
            .filter(|&&c| !cardidx_is_joker(c))
            .map(|&c| (card_rank(c), card_suit(c)))
            .collect();

        let mut suit_counts: HashMap<u8, u8> = HashMap::new();
        for &(_, s) in &real_cards_with_suits {
            if s < 4 {
                *suit_counts.entry(s).or_insert(0) += 1;
            }
        }
        let max_suited = suit_counts.values().cloned().max().unwrap_or(0);

        let score = if counts[0] >= 4 {
            7.0
        } else if counts[0] >= 3 && counts.len() > 1 && counts[1] >= 2 {
            6.0
        } else if real_cards_with_suits.len() >= 4 && max_suited >= 4 {
            5.5
        } else if counts[0] >= 3 {
            3.5
        } else if counts[0] >= 2 && counts.len() > 1 && counts[1] >= 2 {
            2.5
        } else if counts[0] >= 2 {
            1.5
        } else {
            ranks.iter().max().map(|&r| r as f32 / 12.0).unwrap_or(0.0)
        };
        (score / 9.0).min(1.0)
    }
}

/// Draw detection features for a row (3 dims: flush_draw, straight_pot, pair_count).
fn draw_features(cards: &[CardIdx]) -> [f32; 3] {
    let real_cards: Vec<CardIdx> = cards.iter()
        .filter(|&&c| !cardidx_is_joker(c))
        .cloned()
        .collect();

    if real_cards.is_empty() {
        return [0.0; 3];
    }

    // Flush draw: max suited count / 5
    let mut suit_counts = [0u8; 4];
    for &c in &real_cards {
        let s = card_suit(c);
        if s < 4 { suit_counts[s as usize] += 1; }
    }
    let flush_draw = *suit_counts.iter().max().unwrap() as f32 / 5.0;

    // Straight potential: count consecutive ranks / 5
    let mut unique_ranks: Vec<i8> = real_cards.iter()
        .map(|&c| card_rank(c))
        .filter(|&r| r >= 0)
        .collect();
    unique_ranks.sort_unstable();
    unique_ranks.dedup();

    let straight_pot = if unique_ranks.len() >= 2 {
        let mut max_consecutive = 1u8;
        let mut current = 1u8;
        for i in 1..unique_ranks.len() {
            if unique_ranks[i] == unique_ranks[i - 1] + 1 {
                current += 1;
                max_consecutive = max_consecutive.max(current);
            } else {
                current = 1;
            }
        }
        max_consecutive as f32 / 5.0
    } else {
        0.0
    };

    // Pair count
    let mut rank_counts: HashMap<i8, u8> = HashMap::new();
    for &c in &real_cards {
        let r = card_rank(c);
        if r >= 0 { *rank_counts.entry(r).or_insert(0) += 1; }
    }
    let pair_count = rank_counts.values().filter(|&&c| c >= 2).count() as f32 / 3.0;

    [flush_draw, straight_pot, pair_count]
}

/// Create action mask: first N bits true, rest false (matches Python create_action_mask)
pub fn create_action_mask(n_actions: usize) -> [bool; 250] {
    let mut mask = [false; 250];
    for i in 0..n_actions.min(250) {
        mask[i] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_idx_mapping() {
        // Python: ALL_CARDS = [f"{r}{s}" for s in "hdcs" for r in "23456789TJQKA"] + ["X1", "X2"]
        // So: 2h=0, 3h=1, ..., Ah=12, 2d=13, ..., Ad=25, 2c=26, ..., Ac=38, 2s=39, ..., As=51
        assert_eq!(card_str_to_idx("2h"), 0);
        assert_eq!(card_str_to_idx("Ah"), 12);
        assert_eq!(card_str_to_idx("2d"), 13);
        assert_eq!(card_str_to_idx("2c"), 26);
        assert_eq!(card_str_to_idx("2s"), 39);
        assert_eq!(card_str_to_idx("As"), 51);
        assert_eq!(card_str_to_idx("X1"), 52);
        assert_eq!(card_str_to_idx("X2"), 53);

        // Round-trip
        for i in 0..52 {
            let s = cardidx_to_str(i);
            assert_eq!(card_str_to_idx(&s), i, "Round-trip failed for idx {}: {}", i, s);
        }
    }

    #[test]
    fn test_fl_features_empty() {
        let f = fl_features(&[]);
        assert_eq!(f, [0.0; 6]);
    }

    #[test]
    fn test_fl_features_pair_aa() {
        let cards = [card_str_to_idx("Ah"), card_str_to_idx("Ad")];
        let f = fl_features(&cards);
        assert_eq!(f[2], 1.0); // has_A
        assert_eq!(f[3], 1.0); // has_pair
        assert_eq!(f[5], 1.0); // fl_ready
    }

    #[test]
    fn test_encode_state_matches_python() {
        // Same board state as ai/verify_encoding.py
        // Hero: top=[Ah, Kh], mid=[Qs, Js, Ts], bot=[9c, 8c, 7c, 6c, 5c]
        // Opp: top=[2h], mid=[3s, 4s], bot=[5s, 6s, 7s]
        // Dealt: [Ad, Kd, Qd], Discards: [2c], Turn 2, is_btn=true

        let mut hero = Board::new();
        hero.place_mut(card_str_to_idx("Ah"), 0); // top
        hero.place_mut(card_str_to_idx("Kh"), 0);
        hero.place_mut(card_str_to_idx("Qs"), 1); // mid
        hero.place_mut(card_str_to_idx("Js"), 1);
        hero.place_mut(card_str_to_idx("Ts"), 1);
        hero.place_mut(card_str_to_idx("9c"), 2); // bot
        hero.place_mut(card_str_to_idx("8c"), 2);
        hero.place_mut(card_str_to_idx("7c"), 2);
        hero.place_mut(card_str_to_idx("6c"), 2);
        hero.place_mut(card_str_to_idx("5c"), 2);

        let mut opp = Board::new();
        opp.place_mut(card_str_to_idx("2h"), 0);
        opp.place_mut(card_str_to_idx("3s"), 1);
        opp.place_mut(card_str_to_idx("4s"), 1);
        opp.place_mut(card_str_to_idx("5s"), 2);
        opp.place_mut(card_str_to_idx("6s"), 2);
        opp.place_mut(card_str_to_idx("7s"), 2);

        let dealt = [card_str_to_idx("Ad"), card_str_to_idx("Kd"), card_str_to_idx("Qd")];
        let discards = vec![card_str_to_idx("2c")];

        let obs = Observation {
            board_self: &hero,
            board_opponent: &opp,
            dealt_cards: &dealt,
            known_discards_self: &discards,
            turn: 2,
            is_btn: true,
            chips_self: 200,
            chips_opponent: 200,
        };

        let state = encode_state(&obs);

        // Python reference non-zero indices and values
        let expected: Vec<(usize, f32)> = vec![
            (3, 1.0), (17, 1.0), (26, 1.0), (35, 1.0), (44, 1.0),
            (53, 1.0), (62, 1.0), (71, 1.0), (80, 1.0), (89, 1.0),
            (98, 1.0), (99, 1.0), (108, 1.0), (125, 1.0), (134, 1.0),
            (143, 1.0), (152, 1.0), (161, 1.0), (170, 1.0), (179, 1.0),
            (188, 1.0), (197, 1.0), (206, 1.0), (213, 1.0), (222, 1.0),
            (231, 1.0), (241, 1.0), (251, 1.0), (260, 1.0), (263, 1.0),
            (272, 1.0), (281, 1.0), (290, 1.0), (299, 1.0), (314, 1.0),
            (323, 1.0), (332, 1.0), (341, 1.0), (350, 1.0), (359, 1.0),
            (364, 1.0), (373, 1.0), (383, 1.0), (392, 1.0), (401, 1.0),
            (413, 1.0), (422, 1.0), (424, 1.0), (433, 1.0), (442, 1.0),
            (458, 1.0), (467, 1.0), (476, 1.0), (485, 1.0),
            (486, 0.5), (487, 1.0), (488, 1.0), (489, 1.0),
            (490, 0.333333), (491, 0.4), (493, 0.666667), (494, 0.6), (495, 0.4),
            (497, 1.0), (498, 1.0),
            (506, 0.111111), (507, 0.092593), (508, 0.611111),
            (510, 0.018519), (511, 0.046296), (512, 1.0),
            (514, 0.6), (515, 0.6),
            (517, 1.0), (518, 1.0),
        ];

        // Check all expected non-zero values
        let mut mismatches = Vec::new();
        for &(idx, expected_val) in &expected {
            let actual = state[idx];
            if (actual - expected_val).abs() > 1e-4 {
                mismatches.push(format!(
                    "[{}] expected={:.6}, got={:.6}", idx, expected_val, actual
                ));
            }
        }

        // Check no unexpected non-zeros
        let expected_set: std::collections::HashSet<usize> = expected.iter().map(|&(i, _)| i).collect();
        for i in 0..STATE_DIM {
            if !expected_set.contains(&i) && state[i].abs() > 1e-6 {
                mismatches.push(format!(
                    "[{}] expected=0.0, got={:.6} (unexpected non-zero)", i, state[i]
                ));
            }
        }

        if !mismatches.is_empty() {
            panic!("Encoding mismatches ({}):\n{}", mismatches.len(), mismatches.join("\n"));
        }
    }
}
