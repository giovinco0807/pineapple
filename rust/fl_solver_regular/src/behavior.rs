//! Hero behavior policy and T4-vs-FL root generation.
//!
//! # What "vs FL" changes about a hero root
//!
//! The opponent is dealt 14 Fantasyland cards at the start of the hand and the
//! hero never sees any of them, nor any opponent board as it is built. So the
//! hero's five streets are played against *nothing observable*: 14 cards
//! vanish unseen, and the hero draws 17 cards for their own T0..T4.
//!
//! Because the hero's 17 cards are a uniform 17-subset of the deck either way,
//! it is equivalent -- and simpler -- to deal the hero 17 cards from a shuffled
//! deck and let the remaining 35 be the unseen set the opponent's Fantasyland
//! hand is later sampled from. The concrete 14 that "vanished" never enters any
//! label, because the label must be a function of the hero's information set.
//!
//! # Integration debt, stated plainly
//!
//! This is *not* the production learned chain. `ofc_hu_m3_engine`'s
//! `ActorObservation` has no Fantasyland street -- `Street` is `T0..T4` and its
//! own comment notes Fantasyland "geometry is not this one" -- and no way to
//! say "the opponent exists but is entirely hidden". Wiring the learned chain
//! into a hidden-Fantasyland observation needs an observation-geometry change
//! that this crate deliberately does not make. `bp_rollout_v1` below is a
//! self-contained stand-in whose bust and Fantasyland-entry rates are measured
//! against the repo's own baseline numbers instead of assumed.

use crate::eval::{
    bottom_royalty, eval3, eval5, fl_entry_from_top, is_foul, middle_royalty, top_royalty,
};
use crate::rng::SplitMix64;
use serde::{Deserialize, Serialize};

pub const ROW_TOP: u8 = 0;
pub const ROW_MIDDLE: u8 = 1;
pub const ROW_BOTTOM: u8 = 2;
pub const ROW_CAPACITY: [u8; 3] = [3, 5, 5];

/// A hero board under construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartialBoard {
    pub rows: [[u8; 5]; 3],
    pub lengths: [u8; 3],
}

impl PartialBoard {
    #[inline(always)]
    pub fn open_slots(&self, row: u8) -> u8 {
        ROW_CAPACITY[row as usize] - self.lengths[row as usize]
    }

    #[inline(always)]
    pub fn card_count(&self) -> u8 {
        self.lengths[0] + self.lengths[1] + self.lengths[2]
    }

    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        self.card_count() == 13
    }

    #[inline(always)]
    pub fn push(&mut self, card: u8, row: u8) {
        let slot = self.lengths[row as usize] as usize;
        self.rows[row as usize][slot] = card;
        self.lengths[row as usize] += 1;
    }

    pub fn cards(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.card_count() as usize);
        for row in 0..3_usize {
            out.extend_from_slice(&self.rows[row][..self.lengths[row] as usize]);
        }
        out
    }

    pub fn top(&self) -> [u8; 3] {
        [self.rows[0][0], self.rows[0][1], self.rows[0][2]]
    }

    pub fn middle(&self) -> [u8; 5] {
        self.rows[1]
    }

    pub fn bottom(&self) -> [u8; 5] {
        self.rows[2]
    }
}

/// A turn action: which dealt card goes where, and which is discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnAction {
    /// Indices into the three dealt cards, ascending, that get placed.
    pub kept: [u8; 2],
    /// Destination row for `kept[0]` and `kept[1]`.
    pub rows: [u8; 2],
    /// Index into the three dealt cards that is discarded.
    pub discard: u8,
}

/// Legal turn actions, matching `hu_m3_engine::action::generate_turn_actions`:
/// choose two of three dealt cards, then an ordered pair of destination rows.
pub fn generate_turn_actions(board: &PartialBoard, out: &mut Vec<TurnAction>) {
    out.clear();
    for first in 0..2_u8 {
        for second in (first + 1)..3 {
            let discard = 3 - first - second;
            for row_first in 0..3_u8 {
                for row_second in 0..3_u8 {
                    let need_first = 1;
                    let need_second = 1;
                    if row_first == row_second {
                        if board.open_slots(row_first) < need_first + need_second {
                            continue;
                        }
                    } else {
                        if board.open_slots(row_first) < need_first
                            || board.open_slots(row_second) < need_second
                        {
                            continue;
                        }
                    }
                    out.push(TurnAction {
                        kept: [first, second],
                        rows: [row_first, row_second],
                        discard,
                    });
                }
            }
        }
    }
}

/// Opening actions: all five dealt cards are placed, none discarded.
pub fn generate_opening_actions(out: &mut Vec<[u8; 5]>) {
    out.clear();
    for code in 0..243_u32 {
        let mut assignment = [0_u8; 5];
        let mut counts = [0_u8; 3];
        let mut value = code;
        for slot in 0..5 {
            let row = (value % 3) as u8;
            value /= 3;
            assignment[slot] = row;
            counts[row as usize] += 1;
        }
        if counts[0] > ROW_CAPACITY[0] {
            continue;
        }
        out.push(assignment);
    }
}

#[inline(always)]
pub fn apply_turn(board: &PartialBoard, dealt: &[u8; 3], action: &TurnAction) -> PartialBoard {
    let mut next = *board;
    next.push(dealt[action.kept[0] as usize], action.rows[0]);
    next.push(dealt[action.kept[1] as usize], action.rows[1]);
    next
}

/// Behavior-policy configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehaviorConfig {
    /// Rollouts per candidate action at the decision streets.
    pub rollouts: usize,
    /// Value of a hero board that enters Fantasyland, read from a config.
    pub fl_ev: f64,
    /// Cost of fouling: the 6-point line swing plus what the Fantasyland
    /// opponent is worth on average. Measured, not guessed -- see the
    /// provenance string.
    pub foul_penalty: f64,
    pub foul_penalty_provenance: String,
}

impl BehaviorConfig {
    pub fn identity(&self) -> String {
        format!(
            "bp_rollout_v1/rollouts={}/fl_ev={:.6}/foul_penalty={:.6}",
            self.rollouts, self.fl_ev, self.foul_penalty
        )
    }
}

/// Terminal value of a completed hero board, ignoring the opponent's own
/// royalties (which the hero cannot influence).
#[inline(always)]
pub fn terminal_value(board: &PartialBoard, config: &BehaviorConfig) -> f64 {
    debug_assert!(board.is_complete());
    let top_key = eval3(&board.top());
    let middle_key = eval5(&board.middle());
    let bottom_key = eval5(&board.bottom());
    if is_foul(top_key, middle_key, bottom_key) {
        return -config.foul_penalty;
    }
    let royalty = top_royalty(top_key) + middle_royalty(middle_key) + bottom_royalty(bottom_key);
    let entry = if fl_entry_from_top(top_key).is_some() {
        config.fl_ev
    } else {
        0.0
    };
    royalty as f64 + entry
}

/// Score of a partial board for the greedy playout: the value already banked
/// in completed rows, minus the foul penalty if two completed rows are already
/// in the wrong order. Only completed rows are compared, so a penalty here
/// means a foul that is certain, not one that is merely possible.
///
/// A completed top row banks its Fantasyland entry as well as its royalty.
/// This is not an extra heuristic -- it is the same [`terminal_value`] applied
/// to the row that decides entry. Leaving it out was worth roughly nine points
/// of silent bias: the playout would happily bury junk on top to protect the
/// row ordering, so no rollout ever showed the caller what a Fantasyland entry
/// was worth, and the generated hero boards entered Fantasyland at a fraction
/// of the rate the repo's own baseline does.
#[inline(always)]
fn partial_value(board: &PartialBoard, config: &BehaviorConfig) -> f64 {
    let mut banked = 0_i32;
    let mut entry = 0.0_f64;
    let mut keys: [Option<u32>; 3] = [None; 3];
    if board.lengths[0] == 3 {
        let key = eval3(&board.top());
        banked += top_royalty(key);
        if fl_entry_from_top(key).is_some() {
            entry = config.fl_ev;
        }
        keys[0] = Some(key);
    }
    if board.lengths[1] == 5 {
        let key = eval5(&board.middle());
        banked += middle_royalty(key);
        keys[1] = Some(key);
    }
    if board.lengths[2] == 5 {
        let key = eval5(&board.bottom());
        banked += bottom_royalty(key);
        keys[2] = Some(key);
    }
    let mut certain_foul = false;
    if let (Some(top), Some(middle)) = (keys[0], keys[1]) {
        certain_foul |= top > middle;
    }
    if let (Some(middle), Some(bottom)) = (keys[1], keys[2]) {
        certain_foul |= middle > bottom;
    }
    if certain_foul {
        return banked as f64 - config.foul_penalty;
    }
    banked as f64 + entry
}

/// Greedy playout used inside a rollout: at each remaining street take the
/// action with the best immediate value, exactly evaluating the final street.
fn greedy_playout(
    mut board: PartialBoard,
    deck: &[u8],
    mut cursor: usize,
    config: &BehaviorConfig,
    actions: &mut Vec<TurnAction>,
) -> f64 {
    while !board.is_complete() {
        let dealt = [deck[cursor], deck[cursor + 1], deck[cursor + 2]];
        cursor += 3;
        generate_turn_actions(&board, actions);
        let mut best_value = f64::NEG_INFINITY;
        let mut best: Option<PartialBoard> = None;
        for action in actions.iter() {
            let next = apply_turn(&board, &dealt, action);
            let value = if next.is_complete() {
                terminal_value(&next, config)
            } else {
                partial_value(&next, config)
            };
            if value > best_value {
                best_value = value;
                best = Some(next);
            }
        }
        board = best.expect("an incomplete board always has a legal turn action");
    }
    terminal_value(&board, config)
}

/// Estimate an action's value by rolling the rest of the hand out.
fn action_value(
    board: &PartialBoard,
    remaining: &[u8],
    config: &BehaviorConfig,
    rng: &mut SplitMix64,
    scratch_deck: &mut Vec<u8>,
    actions: &mut Vec<TurnAction>,
) -> f64 {
    if board.is_complete() {
        return terminal_value(board, config);
    }
    let needed = 3 * (((13 - board.card_count() as usize) + 1) / 2);
    let mut total = 0.0;
    for _ in 0..config.rollouts {
        scratch_deck.clear();
        scratch_deck.extend_from_slice(remaining);
        rng.partial_shuffle(scratch_deck, needed);
        total += greedy_playout(*board, scratch_deck, 0, config, actions);
    }
    total / config.rollouts as f64
}

/// A generated hero decision point at T4: eleven cards placed, three dealt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeroT4Root {
    pub root_index: u64,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub dealt: [u8; 3],
    /// The hero's own three discards from T1..T3.
    pub discards: [u8; 3],
    /// Everything the hero has seen: board, dealt and discards. 17 cards.
    pub seen_mask: u64,
}

impl HeroT4Root {
    pub fn board(&self) -> PartialBoard {
        let mut board = PartialBoard::default();
        for card in &self.top {
            board.push(*card, ROW_TOP);
        }
        for card in &self.middle {
            board.push(*card, ROW_MIDDLE);
        }
        for card in &self.bottom {
            board.push(*card, ROW_BOTTOM);
        }
        board
    }
}

/// Play a hero hand with `bp_rollout_v1` up to, but not including, the T4
/// decision. Deterministic in `(seed_base, root_index)`.
pub fn generate_root(seed_base: u64, root_index: u64, config: &BehaviorConfig) -> HeroT4Root {
    let mut rng = SplitMix64::for_stream(seed_base, root_index);
    let mut deck: Vec<u8> = (0..52).collect();
    rng.partial_shuffle(&mut deck, 17);
    let hero_cards: Vec<u8> = deck[..17].to_vec();

    // Rollouts must draw from what the hero has not seen *at that street*,
    // which includes the hero's own future cards and the opponent's hidden
    // Fantasyland hand alike. Sampling from the post-hoc unseen remainder
    // instead would quietly tell the policy which cards it is about to be
    // dealt.
    let mut seen_mask_so_far = 0_u64;
    let mut board = PartialBoard::default();
    let mut discards = [0_u8; 3];
    let mut actions: Vec<TurnAction> = Vec::with_capacity(27);
    let mut openings: Vec<[u8; 5]> = Vec::with_capacity(232);
    let mut scratch_deck: Vec<u8> = Vec::with_capacity(52);

    // T0: place all five.
    generate_opening_actions(&mut openings);
    let opening_cards: [u8; 5] = hero_cards[..5].try_into().expect("five opening cards");
    seen_mask_so_far |= crate::cards::mask_of(&opening_cards);
    let unseen = crate::cards::unseen_from_mask(seen_mask_so_far);
    let mut best_value = f64::NEG_INFINITY;
    let mut best_board = board;
    for assignment in openings.iter() {
        let mut next = board;
        let mut legal = true;
        for (slot, row) in assignment.iter().enumerate() {
            if next.open_slots(*row) == 0 {
                legal = false;
                break;
            }
            next.push(opening_cards[slot], *row);
        }
        if !legal {
            continue;
        }
        let value = action_value(
            &next,
            &unseen,
            config,
            &mut rng,
            &mut scratch_deck,
            &mut actions,
        );
        if value > best_value {
            best_value = value;
            best_board = next;
        }
    }
    board = best_board;

    // T1..T3: place two, discard one.
    for turn in 0..3_usize {
        let base = 5 + turn * 3;
        let dealt: [u8; 3] = hero_cards[base..base + 3]
            .try_into()
            .expect("three dealt cards");
        seen_mask_so_far |= crate::cards::mask_of(&dealt);
        let unseen = crate::cards::unseen_from_mask(seen_mask_so_far);
        generate_turn_actions(&board, &mut actions);
        let candidates = actions.clone();
        let mut best_value = f64::NEG_INFINITY;
        let mut best: Option<(PartialBoard, u8)> = None;
        for action in candidates.iter() {
            let next = apply_turn(&board, &dealt, action);
            let value = action_value(
                &next,
                &unseen,
                config,
                &mut rng,
                &mut scratch_deck,
                &mut actions,
            );
            if value > best_value {
                best_value = value;
                best = Some((next, dealt[action.discard as usize]));
            }
        }
        let (next_board, discarded) = best.expect("T1..T3 always have a legal action");
        board = next_board;
        discards[turn] = discarded;
    }

    let dealt: [u8; 3] = hero_cards[14..17].try_into().expect("three T4 cards");
    let seen_mask = crate::cards::mask_of(&hero_cards);

    HeroT4Root {
        root_index,
        top: board.rows[0][..board.lengths[0] as usize].to_vec(),
        middle: board.rows[1][..board.lengths[1] as usize].to_vec(),
        bottom: board.rows[2][..board.lengths[2] as usize].to_vec(),
        dealt,
        discards,
        seen_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BehaviorConfig {
        BehaviorConfig {
            rollouts: 4,
            fl_ev: 9.109,
            foul_penalty: 11.0,
            foul_penalty_provenance: "test".to_owned(),
        }
    }

    #[test]
    fn opening_actions_respect_the_top_row_capacity() {
        let mut out = Vec::new();
        generate_opening_actions(&mut out);
        assert_eq!(out.len(), 232);
        for assignment in out {
            let tops = assignment.iter().filter(|row| **row == ROW_TOP).count();
            assert!(tops <= 3);
        }
    }

    #[test]
    fn turn_actions_match_the_engine_shape_on_an_empty_middle_game_board() {
        let mut board = PartialBoard::default();
        for card in 0..5_u8 {
            board.push(card, ROW_BOTTOM);
        }
        for card in 5..9_u8 {
            board.push(card, ROW_MIDDLE);
        }
        // Bottom full, middle has one slot, top has three: the legal ordered
        // row pairs are (top,top), (top,middle), (middle,top).
        let mut actions = Vec::new();
        generate_turn_actions(&board, &mut actions);
        assert_eq!(actions.len(), 3 * 3);
        for action in actions {
            assert!(action.rows.iter().all(|row| *row != ROW_BOTTOM));
        }
    }

    #[test]
    fn a_t4_board_has_between_three_and_six_actions() {
        // Eleven placed leaves two open slots, so the T4 branching factor is
        // three (both slots in one row) or six (slots in two rows) -- not the
        // ~27 a T3 root has.
        let mut same_row = PartialBoard::default();
        for card in 0..3_u8 {
            same_row.push(card, ROW_TOP);
        }
        for card in 3..8_u8 {
            same_row.push(card, ROW_MIDDLE);
        }
        for card in 8..11_u8 {
            same_row.push(card, ROW_BOTTOM);
        }
        let mut actions = Vec::new();
        generate_turn_actions(&same_row, &mut actions);
        assert_eq!(actions.len(), 3);

        let mut two_rows = PartialBoard::default();
        for card in 0..2_u8 {
            two_rows.push(card, ROW_TOP);
        }
        for card in 2..7_u8 {
            two_rows.push(card, ROW_MIDDLE);
        }
        for card in 7..11_u8 {
            two_rows.push(card, ROW_BOTTOM);
        }
        generate_turn_actions(&two_rows, &mut actions);
        assert_eq!(actions.len(), 6);
    }

    fn board_of(top: &str, middle: &str, bottom: &str) -> PartialBoard {
        let mut board = PartialBoard::default();
        for card in crate::cards::parse_cards(top).unwrap() {
            board.push(card, ROW_TOP);
        }
        for card in crate::cards::parse_cards(middle).unwrap() {
            board.push(card, ROW_MIDDLE);
        }
        for card in crate::cards::parse_cards(bottom).unwrap() {
            board.push(card, ROW_BOTTOM);
        }
        board
    }

    #[test]
    fn the_value_function_pays_for_fantasyland_and_charges_for_a_foul() {
        // If bp_rollout_v1 under-enters Fantasyland it must be the search that
        // is myopic, not the objective it is searching against. Pin both ends.
        let config = config();

        let entering = board_of("Qh Qs 2d", "Kh Kd 6c 8s Tc", "9c 9d 9s Kc Ad");
        let plain = board_of("2h 3d 4c", "Kh Kd 6c 8s Tc", "9c 9d 9s Kc Ad");
        assert!(entering.is_complete() && plain.is_complete());
        let with_entry = terminal_value(&entering, &config);
        let without_entry = terminal_value(&plain, &config);
        // Same middle and bottom, so the whole gap is the top row: a queens
        // pair pays `rank - 5 = 7` royalty plus the Fantasyland entry.
        const QUEENS_TOP_ROYALTY: f64 = 7.0;
        assert!(
            (with_entry - without_entry - (QUEENS_TOP_ROYALTY + config.fl_ev)).abs() < 1e-12,
            "gap was {}",
            with_entry - without_entry
        );

        let fouled = board_of("Ah As Kd", "2h 3h 4c 5d 7s", "4d 4s 8c 9c Td");
        assert_eq!(terminal_value(&fouled, &config), -config.foul_penalty);
    }

    #[test]
    fn a_completed_top_row_banks_its_fantasyland_entry_in_the_playout_score() {
        // partial_value drives every greedy playout step. A completed QQ top
        // has to be worth the entry there too, or no rollout ever reports what
        // Fantasyland is worth back to the caller.
        let config = config();
        let with_queens = board_of("Qh Qs 2d", "Kh Kd 6c 8s Tc", "9c 9d");
        let with_junk = board_of("2h 3d 4c", "Kh Kd 6c 8s Tc", "9c 9d");
        let gap = partial_value(&with_queens, &config) - partial_value(&with_junk, &config);
        assert!((gap - (7.0 + config.fl_ev)).abs() < 1e-12, "gap was {gap}");
    }

    #[test]
    fn generated_roots_are_deterministic_and_well_formed() {
        let config = config();
        let first = generate_root(997_000_000, 3, &config);
        let again = generate_root(997_000_000, 3, &config);
        assert_eq!(first.top, again.top);
        assert_eq!(first.middle, again.middle);
        assert_eq!(first.bottom, again.bottom);
        assert_eq!(first.dealt, again.dealt);

        let board = first.board();
        assert_eq!(board.card_count(), 11);
        assert_eq!(first.seen_mask.count_ones(), 17);
        // Board, dealt and discards partition the seen set exactly.
        let mut seen = board.cards();
        seen.extend_from_slice(&first.dealt);
        seen.extend_from_slice(&first.discards);
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 17);
        assert_eq!(crate::cards::mask_of(&seen), first.seen_mask);
    }
}
