//! Core types for OFC Pineapple Benchmark
//!
//! Board, CardIdx, Action types reused/adapted from backward solver.

use ofc_core::Card;

pub type CardIdx = u8;

pub const JOKER1: CardIdx = 52;
pub const JOKER2: CardIdx = 53;
pub const EMPTY: CardIdx = 0xFF;
pub const NUM_CARDS: usize = 54;

pub const ROW_TOP: u8 = 0;
pub const ROW_MID: u8 = 1;
pub const ROW_BOT: u8 = 2;

/// Convert CardIdx (0-53) to ofc_core::Card
pub fn cardidx_to_card(idx: CardIdx) -> Card {
    if idx >= 52 {
        Card { rank: 0, suit: 4 }
    } else {
        Card {
            rank: (idx / 4) + 2,
            suit: idx % 4,
        }
    }
}

/// Convert ofc_core::Card back to CardIdx
pub fn card_to_cardidx(card: &Card) -> CardIdx {
    if card.is_joker() {
        // Ambiguous for two jokers, caller must track
        52
    } else {
        (card.rank - 2) * 4 + card.suit
    }
}

pub fn cardidx_is_joker(idx: CardIdx) -> bool {
    idx >= 52
}

/// Convert card string (e.g. "Ah", "X1") to CardIdx
pub fn card_str_to_idx(s: &str) -> CardIdx {
    if s.starts_with('X') {
        if s == "X1" { return JOKER1; }
        return JOKER2;
    }
    let rank = match s.as_bytes()[0] {
        b'2' => 0, b'3' => 1, b'4' => 2, b'5' => 3,
        b'6' => 4, b'7' => 5, b'8' => 6, b'9' => 7,
        b'T' => 8, b'J' => 9, b'Q' => 10, b'K' => 11, b'A' => 12,
        _ => panic!("Unknown rank in card: {}", s),
    };
    let suit = match s.as_bytes()[1] {
        b'h' => 0, b'd' => 1, b'c' => 2, b's' => 3,
        _ => panic!("Unknown suit in card: {}", s),
    };
    // Python order: for s in "hdcs" for r in "23456789TJQKA"
    // suit_idx * 13 + rank_idx
    suit * 13 + rank
}

/// Convert CardIdx back to card string
pub fn cardidx_to_str(idx: CardIdx) -> String {
    if idx == JOKER1 { return "X1".to_string(); }
    if idx == JOKER2 { return "X2".to_string(); }
    let rank_idx = (idx % 13) as usize;
    let suit_idx = (idx / 13) as usize;
    let ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
    let suits = ['h', 'd', 'c', 's'];
    format!("{}{}", ranks[rank_idx], suits[suit_idx])
}

// ============================================================
//  Board
// ============================================================

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Board {
    pub top: [CardIdx; 3],
    pub mid: [CardIdx; 5],
    pub bot: [CardIdx; 5],
    pub top_n: u8,
    pub mid_n: u8,
    pub bot_n: u8,
}

impl Board {
    pub fn new() -> Self {
        Board {
            top: [EMPTY; 3],
            mid: [EMPTY; 5],
            bot: [EMPTY; 5],
            top_n: 0,
            mid_n: 0,
            bot_n: 0,
        }
    }

    pub fn place(&self, card: CardIdx, row: u8) -> Board {
        let mut b = self.clone();
        match row {
            ROW_TOP => {
                b.top[b.top_n as usize] = card;
                b.top_n += 1;
                b.top[..b.top_n as usize].sort();
            }
            ROW_MID => {
                b.mid[b.mid_n as usize] = card;
                b.mid_n += 1;
                b.mid[..b.mid_n as usize].sort();
            }
            ROW_BOT => {
                b.bot[b.bot_n as usize] = card;
                b.bot_n += 1;
                b.bot[..b.bot_n as usize].sort();
            }
            _ => unreachable!(),
        }
        b
    }

    pub fn place_mut(&mut self, card: CardIdx, row: u8) {
        match row {
            ROW_TOP => {
                self.top[self.top_n as usize] = card;
                self.top_n += 1;
                self.top[..self.top_n as usize].sort();
            }
            ROW_MID => {
                self.mid[self.mid_n as usize] = card;
                self.mid_n += 1;
                self.mid[..self.mid_n as usize].sort();
            }
            ROW_BOT => {
                self.bot[self.bot_n as usize] = card;
                self.bot_n += 1;
                self.bot[..self.bot_n as usize].sort();
            }
            _ => unreachable!(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.top_n == 3 && self.mid_n == 5 && self.bot_n == 5
    }

    pub fn total_cards(&self) -> u8 {
        self.top_n + self.mid_n + self.bot_n
    }

    pub fn top_cards(&self) -> Vec<Card> {
        self.top[..self.top_n as usize].iter().map(|&i| cardidx_to_card(i)).collect()
    }

    pub fn mid_cards(&self) -> Vec<Card> {
        self.mid[..self.mid_n as usize].iter().map(|&i| cardidx_to_card(i)).collect()
    }

    pub fn bot_cards(&self) -> Vec<Card> {
        self.bot[..self.bot_n as usize].iter().map(|&i| cardidx_to_card(i)).collect()
    }

    pub fn all_card_indices(&self) -> Vec<CardIdx> {
        let mut cards = Vec::new();
        for i in 0..self.top_n as usize { cards.push(self.top[i]); }
        for i in 0..self.mid_n as usize { cards.push(self.mid[i]); }
        for i in 0..self.bot_n as usize { cards.push(self.bot[i]); }
        cards
    }
}

// ============================================================
//  Actions
// ============================================================

/// Turn 0 action: place 5 cards
pub type T0Action = [(CardIdx, u8); 5];

/// Turn 1-4 action: discard 1, place 2
#[derive(Clone)]
pub struct TurnAction {
    pub discard: CardIdx,
    pub placements: [(CardIdx, u8); 2],
}

// ============================================================
//  Action Generation
// ============================================================

pub fn gen_t0_actions(cards: &[CardIdx; 5], board: &Board) -> Vec<T0Action> {
    let top_cap = 3 - board.top_n;
    let mid_cap = 5 - board.mid_n;
    let bot_cap = 5 - board.bot_n;

    let mut actions: Vec<T0Action> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let rows = [ROW_TOP, ROW_MID, ROW_BOT];
    for &r0 in &rows {
        for &r1 in &rows {
            for &r2 in &rows {
                for &r3 in &rows {
                    for &r4 in &rows {
                        let assignment = [r0, r1, r2, r3, r4];
                        let top_count = assignment.iter().filter(|&&r| r == ROW_TOP).count() as u8;
                        let mid_count = assignment.iter().filter(|&&r| r == ROW_MID).count() as u8;
                        let bot_count = assignment.iter().filter(|&&r| r == ROW_BOT).count() as u8;

                        if top_count > top_cap || mid_count > mid_cap || bot_count > bot_cap {
                            continue;
                        }

                        let mut b = board.clone();
                        for i in 0..5 {
                            b = b.place(cards[i], assignment[i]);
                        }

                        if seen.insert(b) {
                            actions.push([
                                (cards[0], r0), (cards[1], r1), (cards[2], r2),
                                (cards[3], r3), (cards[4], r4),
                            ]);
                        }
                    }
                }
            }
        }
    }
    actions
}

pub fn gen_turn_actions(cards: &[CardIdx; 3], board: &Board) -> Vec<TurnAction> {
    let top_cap = 3 - board.top_n;
    let mid_cap = 5 - board.mid_n;
    let bot_cap = 5 - board.bot_n;
    let rows = [ROW_TOP, ROW_MID, ROW_BOT];

    let mut actions: Vec<TurnAction> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for disc in 0..3u8 {
        if cardidx_is_joker(cards[disc as usize]) {
            continue;
        }

        let remaining: Vec<CardIdx> = (0..3u8)
            .filter(|&i| i != disc)
            .map(|i| cards[i as usize])
            .collect();

        for &r0 in &rows {
            for &r1 in &rows {
                let mut cap = [top_cap, mid_cap, bot_cap];
                if cap[r0 as usize] == 0 { continue; }
                cap[r0 as usize] -= 1;
                if cap[r1 as usize] == 0 { continue; }

                let mut b = board.clone();
                b = b.place(remaining[0], r0);
                b = b.place(remaining[1], r1);

                if seen.insert(b) {
                    actions.push(TurnAction {
                        discard: cards[disc as usize],
                        placements: [(remaining[0], r0), (remaining[1], r1)],
                    });
                }
            }
        }
    }
    actions
}
