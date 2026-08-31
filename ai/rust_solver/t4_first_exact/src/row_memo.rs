//! Terminal evaluation memoized over added-card identities.
//!
//! Profiling the T3-vs-FL teacher put terminal evaluation at ~97% of solve
//! time, split between the draw-sweep pattern loop and joint_block's pair
//! table.  Both share one shape: a base board with two open slots, completed
//! by two cards drawn from a pool, evaluated thousands of times with heavy
//! repetition in the added cards.  This module caches at two levels, keyed by
//! card identity so hits accumulate across the whole sweep:
//!
//! - per-row evaluations (value, royalty, FL entry), valid because rows are
//!   independent exactly when the final top and middle are joker-free -- the
//!   same condition as ofc_core's evaluate_board_with_joker_constraint fast
//!   path, which this reproduces verbatim;
//! - whole terminals for the joker-in-top/mid case, where the constrained
//!   joint evaluation cannot decompose by row.
//!
//! The two jokers collide on one cache id, which is sound: they are
//! interchangeable in evaluation.

use ofc_core::{
    check_fl_entry, evaluate_hand_value, get_bottom_royalty, get_middle_royalty, get_top_royalty,
    Card,
};
use std::collections::HashMap;

use super::{terminal_of, CoreBoard, Terminal};

#[derive(Clone, Copy)]
pub(crate) struct RowEvalEntry {
    pub(crate) value: u32,
    pub(crate) royalty: i32,
    /// FL entry count when this row is the top; 0 otherwise or unqualified.
    pub(crate) fl_count: u8,
    pub(crate) has_joker: bool,
}

fn card_id(card: &Card) -> u32 {
    if card.is_joker() {
        52
    } else {
        card.suit as u32 * 13 + (card.rank as u32 - 2)
    }
}

pub(crate) struct TerminalMemo {
    rows: HashMap<u32, RowEvalEntry>,
    slow: HashMap<(u32, u32, u32), Terminal>,
    board: CoreBoard,
    scratch: Vec<Card>,
}

impl TerminalMemo {
    pub(crate) fn new(base: &CoreBoard) -> Self {
        TerminalMemo {
            rows: HashMap::new(),
            slow: HashMap::new(),
            board: base.clone(),
            scratch: Vec::with_capacity(5),
        }
    }

    /// Key: row tag plus the sorted identities of the added cards (up to
    /// five slots, 0x3F = empty), 32 bits total.
    fn row_key(row: usize, added: &[Card]) -> u32 {
        debug_assert!(added.len() <= 5);
        let mut ids = [0x3Fu32; 5];
        for (slot, card) in added.iter().enumerate() {
            ids[slot] = card_id(card);
        }
        ids.sort_unstable();
        let mut key = (row as u32) << 30;
        for (slot, id) in ids.iter().enumerate() {
            key ^= id << (slot * 6).min(24);
        }
        key
    }

    pub(crate) fn row_entry(&mut self, row: usize, added: &[Card]) -> RowEvalEntry {
        let key = Self::row_key(row, added);
        if let Some(entry) = self.rows.get(&key) {
            return *entry;
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.board.rows[row]);
        self.scratch.extend_from_slice(added);
        let cards = &self.scratch;
        let has_joker = cards.iter().any(|card| card.is_joker());
        let expected = if row == 0 { 3 } else { 5 };
        let entry = RowEvalEntry {
            value: evaluate_hand_value(cards, expected),
            royalty: match row {
                0 => get_top_royalty(cards),
                1 => get_middle_royalty(cards),
                _ => get_bottom_royalty(cards),
            },
            fl_count: if row == 0 {
                let (qualified, count) = check_fl_entry(cards);
                if qualified {
                    count
                } else {
                    0
                }
            } else {
                0
            },
            has_joker,
        };
        self.rows.insert(key, entry);
        entry
    }

    /// Terminal of the base board with any number of `additions` placed.
    /// Same fast/slow structure as the two-card path: joker-free final top
    /// and middle assemble from cached row evaluations, anything else runs
    /// the constrained joint evaluation, memoized on the full addition key.
    pub(crate) fn terminal_n(&mut self, additions: &[(usize, Card)]) -> anyhow::Result<Terminal> {
        let mut adds: [[Card; 5]; 3] = [[Card { rank: 0, suit: 0 }; 5]; 3];
        let mut lens = [0usize; 3];
        for (row, card) in additions {
            if lens[*row] >= 5 {
                anyhow::bail!("row {} over capacity in terminal_n", row);
            }
            adds[*row][lens[*row]] = *card;
            lens[*row] += 1;
        }
        let top = self.row_entry(0, &adds[0][..lens[0]]);
        let mid = self.row_entry(1, &adds[1][..lens[1]]);
        if !top.has_joker && !mid.has_joker {
            let bot = self.row_entry(2, &adds[2][..lens[2]]);
            let busted = !(top.value <= mid.value && mid.value <= bot.value);
            return Ok(Terminal {
                busted,
                royalty: if busted {
                    0
                } else {
                    top.royalty + mid.royalty + bot.royalty
                },
                fl_card_count: if busted { 0 } else { top.fl_count },
                values: [top.value, mid.value, bot.value],
            });
        }
        let slow_key = (
            Self::row_key(0, &adds[0][..lens[0]]),
            Self::row_key(1, &adds[1][..lens[1]]),
            Self::row_key(2, &adds[2][..lens[2]]),
        );
        if let Some(cached) = self.slow.get(&slow_key) {
            return Ok(*cached);
        }
        for (row, card) in additions {
            self.board.rows[*row].push(*card);
        }
        let terminal = terminal_of(&self.board);
        for (row, _) in additions.iter().rev() {
            self.board.rows[*row].pop();
        }
        self.slow.insert(slow_key, terminal);
        Ok(terminal)
    }

    /// Terminal of the base board with `additions` = [(row, card); 2]
    /// placed.  Byte-equivalent to pushing the cards and calling terminal_of.
    pub(crate) fn terminal(&mut self, additions: &[(usize, Card); 2]) -> Terminal {
        let mut adds: [[Card; 2]; 3] = [[additions[0].1; 2]; 3];
        let mut lens = [0usize; 3];
        for (row, card) in additions {
            adds[*row][lens[*row]] = *card;
            lens[*row] += 1;
        }
        let top = self.row_entry(0, &adds[0][..lens[0]]);
        let mid = self.row_entry(1, &adds[1][..lens[1]]);
        if !top.has_joker && !mid.has_joker {
            // Independent rows: assemble from cached row evaluations;
            // matches terminal_of's fast path.
            let bot = self.row_entry(2, &adds[2][..lens[2]]);
            let busted = !(top.value <= mid.value && mid.value <= bot.value);
            return Terminal {
                busted,
                royalty: if busted {
                    0
                } else {
                    top.royalty + mid.royalty + bot.royalty
                },
                fl_card_count: if busted { 0 } else { top.fl_count },
                values: [top.value, mid.value, bot.value],
            };
        }
        // Joker in the final top or middle: the constrained joint
        // evaluation, memoized on the full added-card key.
        let slow_key = (
            Self::row_key(0, &adds[0][..lens[0]]),
            Self::row_key(1, &adds[1][..lens[1]]),
            Self::row_key(2, &adds[2][..lens[2]]),
        );
        if let Some(cached) = self.slow.get(&slow_key) {
            return *cached;
        }
        self.board.rows[additions[0].0].push(additions[0].1);
        self.board.rows[additions[1].0].push(additions[1].1);
        let terminal = terminal_of(&self.board);
        self.board.rows[additions[1].0].pop();
        self.board.rows[additions[0].0].pop();
        self.slow.insert(slow_key, terminal);
        terminal
    }
}
