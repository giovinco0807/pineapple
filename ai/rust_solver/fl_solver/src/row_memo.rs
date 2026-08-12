//! Terminal evaluation memoized over added-card identities.
//!
//! Splitting a T3 solve with `teach --own-only`, which skips the opponent
//! scoring entirely, put the Fantasyland frontier scan at 8% of a root (eight
//! dealt roots, 60 opponents, one thread: 8.05 s scored against 7.39 s
//! own-only) and terminal evaluation at the other 92%.  All of that lands in
//! [`crate::t3_labels::completion_value`], which cloned the base board and ran
//! a whole-board evaluation for every (pair, placement) -- while within one
//! call the base board is FIXED and only two cards move.  Every row outside
//! `pattern` was being re-evaluated identically hundreds of times.
//!
//! `t4_first_exact::row_memo` already solved this exact shape for the T3-vs-FL
//! teacher.  This is that module against fl_solver's `Card` and `HeroTerminal`
//! rather than a dependency on it: `t4_first_exact` depends on `fl_solver`, so
//! the arrow cannot also point the other way.
//!
//! What is cached is per-row evaluations (value, royalty, FL entry), keyed by
//! card identity so hits accumulate across the whole sweep.  Rows are
//! independent exactly when the final top and middle are joker-free -- the same
//! condition as ofc_core's `evaluate_board_with_joker_constraint` fast path,
//! which this reproduces.  The two jokers collide on one cache id, which is
//! sound: they are interchangeable in evaluation.
//!
//! # What is deliberately NOT cached
//!
//! `t4_first_exact::row_memo` has a second level holding whole terminals for
//! the joker-in-top/mid case, where the constrained joint evaluation cannot
//! decompose by row.  It is not reproduced here because it cannot hit:
//! `completion_value` visits each (pair, placement) exactly once, so every key
//! it would store is used once and thrown away.  That case goes straight to
//! [`crate::t3_labels::hero_terminal`], which leaves it strictly cheaper than
//! before this module existed -- it saves the caller's board clone and adds
//! nothing.  Which matters more than it sounds: a root whose middle holds two
//! jokers takes EVERY completion down that path, and one thread prices it in
//! 33 s against 0.35 s for a joker-free root.  One such root sets the wall time
//! of a whole batch, so a few percent there outweighs the rest of it.

use crate::vs_fl::HeroTerminal;
use crate::Card;
use std::collections::HashMap;

/// One row's terminal facts, independent of the other two.
#[derive(Clone, Copy)]
struct RowEval {
    value: u32,
    royalty: i32,
    /// FL entry width when this row is the top; 0 otherwise or unqualified.
    fl_width: u8,
}

/// 0..=51 for a natural, 52 for either joker.
fn card_id(card: &Card) -> u32 {
    if card.is_joker() {
        52
    } else {
        card.suit as u32 * 13 + (card.rank as u32 - 2)
    }
}

pub(crate) struct TerminalMemo {
    rows: HashMap<u32, RowEval>,
    /// The base board, used to rebuild whole rows on the joker path.
    board: [Vec<Card>; 3],
    base_core: [Vec<ofc_core::Card>; 3],
    /// Each row with nothing added.  A placement touches at most two rows, so
    /// the third reads its value, royalty and FL entry from here rather than
    /// from a hash lookup or an evaluation.
    base_eval: [RowEval; 3],
    /// Whether each base row already holds a joker.
    ///
    /// Kept apart from `base_eval` because the fast-path test has to be
    /// answerable without evaluating anything.  Taking it from the top and
    /// middle row evaluations instead made a root whose middle already holds a
    /// joker -- where every completion is joint and both row evaluations are
    /// thrown away -- 70% SLOWER than no memo at all: 24 dealt roots went 48.5 s
    /// wall before, 83.4 s after, and the whole regression sat in one root.
    base_joker: [bool; 3],
}

impl TerminalMemo {
    pub(crate) fn new(base: &[Vec<Card>; 3]) -> Self {
        let base_core = [
            to_core(&base[0]),
            to_core(&base[1]),
            to_core(&base[2]),
        ];
        let base_eval = [
            eval_row(&base_core[0], 0, &[]),
            eval_row(&base_core[1], 1, &[]),
            eval_row(&base_core[2], 2, &[]),
        ];
        let base_joker = [
            base[0].iter().any(Card::is_joker),
            base[1].iter().any(Card::is_joker),
            base[2].iter().any(Card::is_joker),
        ];
        TerminalMemo {
            rows: HashMap::new(),
            board: base.clone(),
            base_core,
            base_eval,
            base_joker,
        }
    }

    /// Key: row tag plus the sorted identities of the added cards.  At most two
    /// cards are ever added -- a T4 completion places exactly two -- so six bits
    /// a slot with 0x3F for empty is injective in fourteen bits.
    fn row_key(row: usize, added: &[Card]) -> u32 {
        debug_assert!(added.len() <= 2);
        let mut ids = [0x3Fu32; 2];
        for (slot, card) in added.iter().enumerate() {
            ids[slot] = card_id(card);
        }
        ids.sort_unstable();
        ((row as u32) << 12) | (ids[0] << 6) | ids[1]
    }

    fn row_entry(&mut self, row: usize, added: &[Card]) -> RowEval {
        if added.is_empty() {
            return self.base_eval[row];
        }
        let key = Self::row_key(row, added);
        if let Some(entry) = self.rows.get(&key) {
            return *entry;
        }
        let entry = eval_row(&self.base_core[row], row, added);
        self.rows.insert(key, entry);
        entry
    }

    /// Terminal of the base board with `additions` = [(row, card); 2] placed.
    /// Bit-identical to pushing the cards and calling
    /// [`crate::t3_labels::hero_terminal`].
    pub(crate) fn terminal(&mut self, additions: &[(usize, Card); 2]) -> HeroTerminal {
        // The base is an eleven-card board, which is what makes `base_eval`
        // usable: a row the placement does not touch has no slots left, so the
        // value it holds now is its final one.  On a shorter board an untouched
        // row would still be growing and its cached evaluation would be a
        // hand of the wrong length.
        debug_assert_eq!(
            self.board.iter().map(Vec::len).sum::<usize>(),
            11,
            "the memo prices completions of an eleven-card board"
        );
        let mut adds: [[Card; 2]; 3] = [[additions[0].1; 2]; 3];
        let mut lens = [0usize; 3];
        for (row, card) in additions {
            adds[*row][lens[*row]] = *card;
            lens[*row] += 1;
        }
        let joker_above = |row: usize| {
            self.base_joker[row] || adds[row][..lens[row]].iter().any(Card::is_joker)
        };
        if !joker_above(0) && !joker_above(1) {
            // Independent rows: assemble from the cached row evaluations, which
            // is what evaluate_board_with_joker_constraint's fast path plus
            // hero_terminal add up to.
            let top = self.row_entry(0, &adds[0][..lens[0]]);
            let mid = self.row_entry(1, &adds[1][..lens[1]]);
            let bot = self.row_entry(2, &adds[2][..lens[2]]);
            if !(top.value <= mid.value && mid.value <= bot.value) {
                return HeroTerminal {
                    busted: true,
                    top: 0,
                    mid: 0,
                    bot: 0,
                    royalty: 0,
                    entry_width: 0,
                };
            }
            return HeroTerminal {
                busted: false,
                top: top.value,
                mid: mid.value,
                bot: bot.value,
                royalty: top.royalty + mid.royalty + bot.royalty,
                entry_width: top.fl_width,
            };
        }
        // Joker in the final top or middle: the joker's rank is chosen against
        // the row below it, so no row has a value of its own.  Nothing to
        // memoize (see the module header) -- push, evaluate, pop.
        self.board[additions[0].0].push(additions[0].1);
        self.board[additions[1].0].push(additions[1].1);
        let terminal = crate::t3_labels::hero_terminal(&self.board);
        self.board[additions[1].0].pop();
        self.board[additions[0].0].pop();
        terminal
    }
}

fn to_core(cards: &[Card]) -> Vec<ofc_core::Card> {
    cards
        .iter()
        .map(|card| ofc_core::Card { rank: card.rank, suit: card.suit })
        .collect()
}

/// One row of `base` plus `added`, evaluated on its own.
fn eval_row(base: &[ofc_core::Card], row: usize, added: &[Card]) -> RowEval {
    let mut cards = [ofc_core::Card { rank: 0, suit: 0 }; 5];
    let mut len = 0usize;
    for card in base {
        cards[len] = *card;
        len += 1;
    }
    for card in added {
        cards[len] = ofc_core::Card { rank: card.rank, suit: card.suit };
        len += 1;
    }
    let cards = &cards[..len];
    RowEval {
        value: ofc_core::evaluate_hand_value(cards, if row == 0 { 3 } else { 5 }),
        royalty: match row {
            0 => ofc_core::get_top_royalty(cards),
            1 => ofc_core::get_middle_royalty(cards),
            _ => ofc_core::get_bottom_royalty(cards),
        },
        fl_width: if row == 0 {
            let (qualifies, width) = ofc_core::check_fl_entry(cards);
            if qualifies {
                width
            } else {
                0
            }
        } else {
            0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::deal;
    use crate::t3_labels::{hero_terminal, open_patterns, unseen_from};

    fn same(a: &HeroTerminal, b: &HeroTerminal) -> bool {
        a.busted == b.busted
            && a.top == b.top
            && a.mid == b.mid
            && a.bot == b.bot
            && a.royalty == b.royalty
            && a.entry_width == b.entry_width
    }

    /// Every eleven-card board shape.  There are six, and each admits a
    /// different set of patterns -- two into one row for `(1,5,5)`, `(3,3,5)`
    /// and `(3,5,3)`, one into each of two rows for the rest -- so the sweep
    /// covers both key shapes rather than whichever the dealt path happens to
    /// reach.
    const ELEVEN_CARD_SHAPES: [[usize; 3]; 6] =
        [[1, 5, 5], [2, 4, 5], [2, 5, 4], [3, 3, 5], [3, 4, 4], [3, 5, 3]];

    /// The memo is an optimisation, so the only property that matters is that
    /// it moves nothing.  Checked against the unmemoized `hero_terminal` on
    /// every completion of every board shape, with the jokers put where they
    /// decide which path the memo takes.
    ///
    /// Put there rather than dealt: an earlier version of this test built the
    /// board from a deal and asserted afterwards on the joker counts it had
    /// happened to see, and sixteen deals produced no board holding two.  The
    /// joint path is exactly the part a row-independent memo has no right to
    /// take, so it is arranged, not hoped for.
    #[test]
    fn the_memo_agrees_with_hero_terminal_on_every_completion() {
        let joker = Card { rank: 0, suit: 4 };
        let mut checked = 0usize;
        let mut joint = 0usize;
        for attempt in 0..6u64 {
            // Eleven naturals, so the only jokers on the board are the ones
            // this test puts there and `unseen_from` still offers both.
            let naturals: Vec<Card> = deal(0x9E11_0000 + attempt, attempt, 20)
                .into_iter()
                .filter(|card| !card.is_joker())
                .take(11)
                .collect();
            for shape in ELEVEN_CARD_SHAPES {
                let mid = shape[0];
                for spots in [
                    &[][..],
                    &[0][..],
                    &[mid][..],
                    &[mid, mid + 1][..],
                    &[0, mid][..],
                ] {
                    let mut flat = naturals.clone();
                    for spot in spots {
                        flat[*spot] = joker;
                    }
                    let mut rows: [Vec<Card>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                    let mut next = 0usize;
                    for (row, count) in shape.iter().enumerate() {
                        for _ in 0..*count {
                            rows[row].push(flat[next]);
                            next += 1;
                        }
                    }
                    // A short pool keeps the sweep to seconds.  Taken from both
                    // ends: `unseen_from` puts whatever jokers are left last,
                    // and a prefix would never add one.
                    let pool = unseen_from(&flat);
                    let mut unseen: Vec<Card> = pool.iter().copied().take(10).collect();
                    unseen.extend(pool.iter().rev().take(4).copied());
                    let patterns = open_patterns(&rows);
                    let mut memo = TerminalMemo::new(&rows);
                    for first in 0..unseen.len() {
                        for second in (first + 1)..unseen.len() {
                            for pattern in &patterns {
                                let mut expected = rows.clone();
                                expected[pattern[0]].push(unseen[first]);
                                expected[pattern[1]].push(unseen[second]);
                                let want = hero_terminal(&expected);
                                let got = memo.terminal(&[
                                    (pattern[0], unseen[first]),
                                    (pattern[1], unseen[second]),
                                ]);
                                assert!(
                                    same(&want, &got),
                                    "shape {shape:?} jokers at {spots:?} pattern \
                                     {pattern:?} moved: want {want:?} got {got:?}"
                                );
                                let top_or_mid =
                                    expected[0].iter().chain(expected[1].iter());
                                if top_or_mid.clone().any(Card::is_joker) {
                                    joint += 1;
                                }
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0 && joint > 0);
        println!(
            "checked {checked} completions, {joint} of them down the joint \
             joker path"
        );
    }

    /// Both jokers share a cache id, so a completion adding two of them must
    /// still be told apart from one adding a joker and a natural.
    #[test]
    fn a_second_joker_is_not_confused_with_the_first() {
        let joker = Card { rank: 0, suit: 4 };
        let ace = Card { rank: 14, suit: 1 };
        // (1,5,5): both cards go to the top, so every completion reshapes the
        // row the joker constraint reads.
        let rows: [Vec<Card>; 3] = [
            vec![Card { rank: 14, suit: 0 }],
            vec![
                Card { rank: 5, suit: 0 },
                Card { rank: 6, suit: 0 },
                Card { rank: 7, suit: 0 },
                Card { rank: 8, suit: 0 },
                Card { rank: 10, suit: 3 },
            ],
            vec![
                Card { rank: 9, suit: 1 },
                Card { rank: 9, suit: 2 },
                Card { rank: 9, suit: 3 },
                Card { rank: 2, suit: 1 },
                Card { rank: 3, suit: 1 },
            ],
        ];
        let mut memo = TerminalMemo::new(&rows);
        for addition in [[joker, joker], [joker, ace], [ace, Card { rank: 14, suit: 2 }]] {
            let got = memo.terminal(&[(0, addition[0]), (0, addition[1])]);
            let mut expected = rows.clone();
            expected[0].push(addition[0]);
            expected[0].push(addition[1]);
            let want = hero_terminal(&expected);
            assert!(same(&want, &got), "want {want:?} got {got:?}");
        }
    }
}
