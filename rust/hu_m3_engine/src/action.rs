//! Legal action generation for all regular Pineapple streets.

use crate::cards::{validate_cards, Card};
use crate::state::{Board, Row, ALL_ROWS};
use serde::{Deserialize, Serialize};

/// A semantic action. Tuple placement serde matches Python `[card, row]` pairs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub placements: Vec<(Card, Row)>,
    #[serde(default)]
    pub discards: Vec<Card>,
}

impl Action {
    pub fn new(placements: Vec<(Card, Row)>, discards: Vec<Card>) -> Result<Self, String> {
        let action = Self {
            placements,
            discards,
        };
        action.validate()?;
        Ok(action)
    }

    pub fn validate(&self) -> Result<(), String> {
        let cards: Vec<_> = self
            .placements
            .iter()
            .map(|(card, _)| *card)
            .chain(self.discards.iter().copied())
            .collect();
        validate_cards(&cards)
    }

    pub fn apply(&self, board: &Board) -> Result<Board, String> {
        self.validate()?;
        board.place(&self.placements)
    }

    /// Apply an action produced by the validated native legal-action
    /// generator. This avoids repeating card/capacity checks at every exact
    /// rollout node while keeping the public boundary fail-closed.
    pub(crate) fn apply_trusted(&self, board: &Board) -> Board {
        board.place_trusted(&self.placements)
    }
}

pub fn generate_actions(board: &Board, dealt_cards: &[Card]) -> Result<Vec<Action>, String> {
    if board.card_count() == 0 && dealt_cards.len() == 5 {
        generate_initial_actions(board, dealt_cards)
    } else {
        generate_turn_actions(board, dealt_cards)
    }
}

/// Generate T0 in the same legacy enumeration order as Python `itertools.product`.
pub fn generate_initial_actions(
    board: &Board,
    dealt_cards: &[Card],
) -> Result<Vec<Action>, String> {
    board.validate()?;
    validate_cards(dealt_cards)?;
    if board.card_count() != 0 {
        return Err("opening actions require an empty board".to_string());
    }
    if dealt_cards.len() != 5 {
        return Err("opening actions require exactly five dealt cards".to_string());
    }

    let mut actions = Vec::new();
    for rows in row_products(5) {
        if !rows_fit(board, &rows) {
            continue;
        }
        let placements = dealt_cards.iter().copied().zip(rows).collect::<Vec<_>>();
        if board.place(&placements).is_ok() {
            actions.push(Action {
                placements,
                discards: Vec::new(),
            });
        }
    }
    Ok(actions)
}

/// Generate T1--T4 actions: place two of three and privately discard one.
///
/// The one-slot compatibility path matches Python and places one card while
/// discarding the rest.
pub fn generate_turn_actions(board: &Board, dealt_cards: &[Card]) -> Result<Vec<Action>, String> {
    board.validate()?;
    validate_cards(dealt_cards)?;
    let board_mask = board
        .all_cards()
        .into_iter()
        .fold(0_u64, |mask, card| mask | card.bit());
    if dealt_cards.iter().any(|card| board_mask & card.bit() != 0) {
        return Err("dealt cards overlap board cards".to_string());
    }

    Ok(generate_turn_actions_trusted(board, dealt_cards))
}

/// Generate actions for an internally constructed observation whose board,
/// deal, disjointness, and geometry were already validated.
pub(crate) fn generate_turn_actions_trusted(board: &Board, dealt_cards: &[Card]) -> Vec<Action> {
    let open_total = 13_usize.saturating_sub(board.card_count());
    if open_total == 0 {
        return Vec::new();
    }
    let place_count = 2_usize.min(open_total).min(dealt_cards.len());
    if place_count == 0 {
        return Vec::new();
    }

    let mut actions = Vec::new();
    for indices in index_combinations(dealt_cards.len(), place_count) {
        let place_cards: Vec<Card> = indices.iter().map(|&index| dealt_cards[index]).collect();
        let discards = dealt_cards
            .iter()
            .enumerate()
            .filter_map(|(index, &card)| (!indices.contains(&index)).then_some(card))
            .collect::<Vec<_>>();
        for rows in row_products(place_count) {
            if !rows_fit(board, &rows) {
                continue;
            }
            let placements = place_cards.iter().copied().zip(rows).collect::<Vec<_>>();
            actions.push(Action {
                placements,
                discards: discards.clone(),
            });
        }
    }
    actions
}

fn rows_fit(board: &Board, rows: &[Row]) -> bool {
    ALL_ROWS.iter().all(|&row| {
        rows.iter().filter(|&&candidate| candidate == row).count() <= board.open_slots(row)
    })
}

fn row_products(length: usize) -> Vec<Vec<Row>> {
    fn visit(length: usize, prefix: &mut Vec<Row>, output: &mut Vec<Vec<Row>>) {
        if prefix.len() == length {
            output.push(prefix.clone());
            return;
        }
        for row in ALL_ROWS {
            prefix.push(row);
            visit(length, prefix, output);
            prefix.pop();
        }
    }

    let mut output = Vec::new();
    visit(length, &mut Vec::with_capacity(length), &mut output);
    output
}

fn index_combinations(size: usize, choose: usize) -> Vec<Vec<usize>> {
    fn visit(
        size: usize,
        choose: usize,
        start: usize,
        prefix: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if prefix.len() == choose {
            output.push(prefix.clone());
            return;
        }
        let need = choose - prefix.len();
        if need > size.saturating_sub(start) {
            return;
        }
        for index in start..=size - need {
            prefix.push(index);
            visit(size, choose, index + 1, prefix, output);
            prefix.pop();
        }
    }

    let mut output = Vec::new();
    if choose <= size {
        visit(
            size,
            choose,
            0,
            &mut Vec::with_capacity(choose),
            &mut output,
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cards(values: &[&str]) -> Vec<Card> {
        values.iter().map(|value| value.parse().unwrap()).collect()
    }

    #[test]
    fn t0_has_the_python_232_semantic_actions() {
        let actions =
            generate_initial_actions(&Board::empty(), &cards(&["Ah", "Kd", "Qc", "Js", "Th"]))
                .unwrap();
        assert_eq!(actions.len(), 232);
        assert!(actions.iter().all(|action| action.placements.len() == 5));
        assert!(actions.iter().all(|action| action.discards.is_empty()));
    }

    #[test]
    fn final_turn_places_two_and_fits_open_rows() {
        let board = Board::new(
            cards(&["Qh", "2d"]),
            cards(&["Kh", "Kd", "6c", "8s", "Th"]),
            cards(&["9c", "9d", "9s", "Kc"]),
        )
        .unwrap();
        let actions = generate_turn_actions(&board, &cards(&["Qs", "Ah", "7d"])).unwrap();
        assert_eq!(actions.len(), 6);
        for action in actions {
            assert_eq!(action.placements.len(), 2);
            assert_eq!(action.discards.len(), 1);
            assert!(action.apply(&board).unwrap().is_complete());
        }
    }

    #[test]
    fn one_slot_path_discards_two() {
        let board = Board::new(
            cards(&["Qh", "2d"]),
            cards(&["Kh", "Kd", "6c", "8s", "Th"]),
            cards(&["9c", "9d", "9s", "Kc", "Ah"]),
        )
        .unwrap();
        let actions = generate_actions(&board, &cards(&["Qs", "Ac", "7d"])).unwrap();
        assert_eq!(actions.len(), 3);
        assert!(actions.iter().all(|action| action.placements.len() == 1));
        assert!(actions.iter().all(|action| action.discards.len() == 2));
    }
}
