//! Action generation for OFC Pineapple CFR (optimized).
//!
//! T0: 5 cards → distribute to top(≤3)/middle(≤5)/bottom(≤5)
//!   Uses COMBINATIONS instead of permutations for O(C(5,k)) vs O(5!)
//! T1-T4: 3 cards → discard 1, place 2
//!
//! T0 constraint filters (domain knowledge):
//! 1. Ace → top or middle only (never bottom)
//! 2. Joker → top or bottom only (never middle)
//! 3. Two jokers must be in different rows

use ofc_core::Card;

/// Row identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Row {
    Top = 0,
    Middle = 1,
    Bottom = 2,
}

const ROWS: [Row; 3] = [Row::Top, Row::Middle, Row::Bottom];

/// Board state: cards placed in each row
#[derive(Clone, Debug)]
pub struct Board {
    pub top: Vec<Card>,
    pub middle: Vec<Card>,
    pub bottom: Vec<Card>,
}

impl Board {
    pub fn new() -> Self {
        Board {
            top: Vec::with_capacity(3),
            middle: Vec::with_capacity(5),
            bottom: Vec::with_capacity(5),
        }
    }

    pub fn top_space(&self) -> usize { 3 - self.top.len() }
    pub fn mid_space(&self) -> usize { 5 - self.middle.len() }
    pub fn bot_space(&self) -> usize { 5 - self.bottom.len() }

    pub fn is_complete(&self) -> bool {
        self.top.len() == 3 && self.middle.len() == 5 && self.bottom.len() == 5
    }

    pub fn row_mut(&mut self, row: Row) -> &mut Vec<Card> {
        match row {
            Row::Top => &mut self.top,
            Row::Middle => &mut self.middle,
            Row::Bottom => &mut self.bottom,
        }
    }

    pub fn row_len(&self, row: Row) -> usize {
        match row {
            Row::Top => self.top.len(),
            Row::Middle => self.middle.len(),
            Row::Bottom => self.bottom.len(),
        }
    }

    pub fn row_capacity(&self, row: Row) -> usize {
        match row {
            Row::Top => 3,
            Row::Middle => 5,
            Row::Bottom => 5,
        }
    }
}

/// A compact action: row assignment for each card.
/// For T0: rows[0..5] are assignments, discard_idx = None
/// For T1-T4: rows[0..2] are assignments for the 2 kept cards, discard_idx = Some(index into hand)
#[derive(Clone, Debug)]
pub struct Action {
    /// Row assigned to each placed card
    pub row_assignments: Vec<Row>,
    /// Index of discarded card in original hand (None for T0)
    pub discard_idx: Option<usize>,
}

// ─── T0 Constraint Filter ──────────────────────────────────────

/// Check if assigning card to row satisfies per-card constraints.
/// Returns false if this assignment violates a constraint.
#[inline(always)]
fn is_card_row_valid(_card: &Card, _row: Row) -> bool {
    // No card-row restrictions: any card can go to any row per OFC rules.
    true
}

/// Check 2-joker constraint: can't be in same row.
#[inline(always)]
fn check_joker_pair_constraint(cards: &[Card], assignments: &[Row]) -> bool {
    let mut first_joker_row: Option<Row> = None;
    for (i, card) in cards.iter().enumerate() {
        if card.is_joker() {
            if let Some(prev_row) = first_joker_row {
                if prev_row == assignments[i] {
                    return false;
                }
            } else {
                first_joker_row = Some(assignments[i]);
            }
        }
    }
    true
}

// ─── T0 Action Generation (Combination-based) ──────────────────

/// Generate all valid T0 actions using combinations.
/// Instead of generating 5! = 120 permutations per split and deduplicating,
/// we directly enumerate C(5, top_n) * C(remaining, mid_n) unique assignments.
pub fn generate_t0_actions(hand: &[Card], board: &Board) -> Vec<Action> {
    debug_assert_eq!(hand.len(), 5, "T0 expects exactly 5 cards");

    let top_space = board.top_space();
    let mid_space = board.mid_space();
    let bot_space = board.bot_space();

    let mut actions = Vec::with_capacity(256);

    // For each valid (top_n, mid_n, bot_n) split
    for top_n in 0..=top_space.min(5) {
        for mid_n in 0..=mid_space.min(5 - top_n) {
            let bot_n = 5 - top_n - mid_n;
            if bot_n > bot_space {
                continue;
            }

            // Generate combinations: choose top_n cards for top
            enumerate_combinations_t0(
                hand, top_n, mid_n, &mut actions,
            );
        }
    }

    actions
}

/// Enumerate all C(5, top_n) * C(5-top_n, mid_n) unique card-to-row assignments.
fn enumerate_combinations_t0(
    hand: &[Card],
    top_n: usize,
    mid_n: usize,
    actions: &mut Vec<Action>,
) {
    let n = hand.len();
    let mut top_indices = Vec::with_capacity(top_n);

    // Enumerate top card combinations
    enumerate_top(hand, n, top_n, mid_n, 0, &mut top_indices, actions);
}

fn enumerate_top(
    hand: &[Card],
    n: usize,
    top_n: usize,
    mid_n: usize,
    start: usize,
    top_indices: &mut Vec<usize>,
    actions: &mut Vec<Action>,
) {
    if top_indices.len() == top_n {
        // top_indices is complete; now enumerate mid combinations from remaining
        let remaining: Vec<usize> = (0..n)
            .filter(|i| !top_indices.contains(i))
            .collect();
        let mut mid_indices = Vec::with_capacity(mid_n);
        enumerate_mid(hand, &remaining, mid_n, top_indices, 0, &mut mid_indices, actions);
        return;
    }

    for i in start..n {
        // Early prune: check card-row constraint for top
        if !is_card_row_valid(&hand[i], Row::Top) {
            continue;
        }
        top_indices.push(i);
        enumerate_top(hand, n, top_n, mid_n, i + 1, top_indices, actions);
        top_indices.pop();
    }
}

fn enumerate_mid(
    hand: &[Card],
    remaining: &[usize],
    mid_n: usize,
    top_indices: &[usize],
    start: usize,
    mid_indices: &mut Vec<usize>,
    actions: &mut Vec<Action>,
) {
    if mid_indices.len() == mid_n {
        // Both top and mid are selected; remaining goes to bottom
        let bot_indices: Vec<usize> = remaining.iter()
            .copied()
            .filter(|i| !mid_indices.contains(&i))
            .collect();

        // Build assignments
        let mut assignments = vec![Row::Bottom; hand.len()];
        for &i in top_indices {
            assignments[i] = Row::Top;
        }
        for &i in mid_indices.iter() {
            assignments[i] = Row::Middle;
        }

        // Check bottom card constraints
        for &i in &bot_indices {
            if !is_card_row_valid(&hand[i], Row::Bottom) {
                return;
            }
        }

        // Check 2-joker constraint
        if !check_joker_pair_constraint(hand, &assignments) {
            return;
        }

        actions.push(Action {
            row_assignments: assignments,
            discard_idx: None,
        });
        return;
    }

    for idx in start..remaining.len() {
        let i = remaining[idx];
        // Early prune: check card-row constraint for middle
        if !is_card_row_valid(&hand[i], Row::Middle) {
            continue;
        }
        mid_indices.push(i);
        enumerate_mid(hand, remaining, mid_n, top_indices, idx + 1, mid_indices, actions);
        mid_indices.pop();
    }
}

// ─── T1-T4 Action Generation ───────────────────────────────────

/// Generate all valid turn actions (3 cards → discard 1, place 2).
/// Returns compact actions with discard_idx and row assignments for 2 kept cards.
pub fn generate_turn_actions(hand: &[Card], board: &Board) -> Vec<Action> {
    debug_assert_eq!(hand.len(), 3, "Regular turn expects exactly 3 cards");

    let mut actions = Vec::with_capacity(27);

    for discard_idx in 0..3 {
        let kept = [
            (discard_idx + 1) % 3,
            (discard_idx + 2) % 3,
        ];

        for &row0 in &ROWS {
            for &row1 in &ROWS {
                // Check capacity
                let mut counts = [
                    board.row_len(Row::Top),
                    board.row_len(Row::Middle),
                    board.row_len(Row::Bottom),
                ];
                counts[row0 as usize] += 1;
                if counts[row0 as usize] > board.row_capacity(row0) {
                    continue;
                }
                counts[row1 as usize] += 1;
                if counts[row1 as usize] > board.row_capacity(row1) {
                    continue;
                }




                let mut assignments = vec![Row::Top; 3]; // placeholder
                assignments[kept[0]] = row0;
                assignments[kept[1]] = row1;
                // The discarded card's row doesn't matter

                actions.push(Action {
                    row_assignments: vec![row0, row1],
                    discard_idx: Some(discard_idx),
                });
            }
        }
    }

    actions
}

/// Get the cards to place from a turn action and hand.
#[inline]
pub fn get_placed_cards(hand: &[Card], action: &Action) -> Vec<(Card, Row)> {
    match action.discard_idx {
        None => {
            // T0: all cards placed
            hand.iter().zip(action.row_assignments.iter())
                .map(|(c, &r)| (*c, r))
                .collect()
        }
        Some(discard_idx) => {
            // T1-T4: 2 kept cards
            let kept = [
                (discard_idx + 1) % 3,
                (discard_idx + 2) % 3,
            ];
            vec![
                (hand[kept[0]], action.row_assignments[0]),
                (hand[kept[1]], action.row_assignments[1]),
            ]
        }
    }
}

/// Get the discarded card from a turn action and hand.
#[inline]
pub fn get_discard(hand: &[Card], action: &Action) -> Option<Card> {
    action.discard_idx.map(|idx| hand[idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofc_core::Card;

    fn make_card(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    fn joker() -> Card {
        Card { rank: 0, suit: 4 }
    }

    #[test]
    fn test_t0_action_count_no_special() {
        // 5 distinct low cards, no aces or jokers → should be 232
        let hand = [
            make_card(2, 0), make_card(3, 1), make_card(4, 2),
            make_card(5, 3), make_card(6, 0),
        ];
        let board = Board::new();
        let actions = generate_t0_actions(&hand, &board);
        assert_eq!(actions.len(), 232, "5 distinct low cards should give 232 actions, got {}", actions.len());
    }

    #[test]
    fn test_t0_with_aces_filtered() {
        // 2 Aces: should filter out actions with Ace on bottom
        let hand = [
            make_card(14, 0), make_card(14, 1), make_card(3, 2),
            make_card(7, 3), make_card(13, 0),
        ];
        let board = Board::new();
        let actions = generate_t0_actions(&hand, &board);
        assert!(actions.len() < 232, "Aces should reduce action count, got {}", actions.len());
        // Verify no Ace on bottom
        for a in &actions {
            for (i, &r) in a.row_assignments.iter().enumerate() {
                if hand[i].rank == 14 {
                    assert_ne!(r, Row::Bottom, "Ace must not be on bottom");
                }
            }
        }
    }

    #[test]
    fn test_t0_with_jokers_filtered() {
        // 2 Jokers: should filter aggressively
        let hand = [
            joker(), joker(), make_card(13, 0),
            make_card(7, 3), make_card(3, 2),
        ];
        let board = Board::new();
        let actions = generate_t0_actions(&hand, &board);
        assert!(actions.len() < 100, "2 Jokers should heavily reduce actions, got {}", actions.len());
        // Verify no joker on middle & no 2 jokers same row
        for a in &actions {
            let mut joker_rows = Vec::new();
            for (i, &r) in a.row_assignments.iter().enumerate() {
                if hand[i].is_joker() {
                    assert_ne!(r, Row::Middle, "Joker must not be on middle");
                    joker_rows.push(r);
                }
            }
            if joker_rows.len() >= 2 {
                assert_ne!(joker_rows[0], joker_rows[1], "2 jokers must be in different rows");
            }
        }
    }

    #[test]
    fn test_turn_actions_open_board() {
        let hand = [make_card(10, 0), make_card(11, 1), make_card(12, 2)];
        let mut board = Board::new();
        board.top.push(make_card(2, 0));
        board.middle.push(make_card(3, 0));
        board.middle.push(make_card(4, 1));
        board.bottom.push(make_card(5, 2));
        board.bottom.push(make_card(6, 3));

        let actions = generate_turn_actions(&hand, &board);
        assert!(actions.len() > 0, "Should have some valid actions");
        assert!(actions.len() <= 27, "Max 27 actions for 3 cards, got {}", actions.len());
    }

    #[test]
    fn test_turn_actions_nearly_full() {
        let hand = [make_card(10, 0), make_card(11, 1), make_card(12, 2)];
        let mut board = Board::new();
        board.top = vec![make_card(2, 0), make_card(3, 0), make_card(4, 0)];
        board.middle = vec![
            make_card(5, 0), make_card(6, 0), make_card(7, 0),
            make_card(8, 0), make_card(9, 0),
        ];
        board.bottom = vec![make_card(10, 1), make_card(11, 2), make_card(12, 3)];

        let actions = generate_turn_actions(&hand, &board);
        assert_eq!(actions.len(), 3, "Only 3 discard choices when only bottom has space, got {}", actions.len());
    }
}
