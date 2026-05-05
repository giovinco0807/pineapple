use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitBoard(pub u64);

impl BitBoard {
    pub const EMPTY: BitBoard = BitBoard(0);
    pub const FULL_DECK: BitBoard = BitBoard((1 << 54) - 1);
    pub const FULL_DECK_NO_JOKERS: BitBoard = BitBoard((1 << 52) - 1);

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub fn contains(&self, card: u8) -> bool {
        (self.0 & (1 << card)) != 0
    }

    #[inline]
    pub fn add(&mut self, card: u8) {
        self.0 |= 1 << card;
    }

    #[inline]
    pub fn remove(&mut self, card: u8) {
        self.0 &= !(1 << card);
    }

    #[inline]
    pub fn merge(&self, other: BitBoard) -> BitBoard {
        BitBoard(self.0 | other.0)
    }

    #[inline]
    pub fn intersect(&self, other: BitBoard) -> BitBoard {
        BitBoard(self.0 & other.0)
    }

    #[inline]
    pub fn difference(&self, other: BitBoard) -> BitBoard {
        BitBoard(self.0 & !other.0)
    }

    #[inline]
    pub fn has_jokers(&self) -> bool {
        (self.0 & (3 << 52)) != 0
    }
    
    #[inline]
    pub fn num_jokers(&self) -> u32 {
        (self.0 & (3 << 52)).count_ones()
    }

    // Iterate over cards
    pub fn cards(&self) -> BitBoardIter {
        BitBoardIter { bb: self.0 }
    }

    pub fn from_string(s: &str) -> Option<BitBoard> {
        let mut bb = BitBoard::EMPTY;
        for part in s.split_whitespace() {
            if let Some(c) = string_to_card(part) {
                bb.add(c);
            } else {
                return None;
            }
        }
        Some(bb)
    }
}

impl std::ops::BitOr for BitBoard {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        BitBoard(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for BitBoard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for BitBoard {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        BitBoard(self.0 & rhs.0)
    }
}

impl std::ops::Sub for BitBoard {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        BitBoard(self.0 & !rhs.0)
    }
}

pub struct BitBoardIter {
    bb: u64,
}

impl Iterator for BitBoardIter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bb == 0 {
            None
        } else {
            let card = self.bb.trailing_zeros() as u8;
            self.bb &= self.bb - 1; // clear the lowest set bit
            Some(card)
        }
    }
}

// Helper to convert to human readable string
pub fn card_to_string(card: u8) -> String {
    if card == 52 || card == 53 {
        return "JK".to_string();
    }
    let ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
    let suits = ['s', 'h', 'd', 'c'];
    
    let rank_idx = (card % 13) as usize;
    let suit_idx = (card / 13) as usize;
    
    format!("{}{}", ranks[rank_idx], suits[suit_idx])
}

// Convert from "As", "Kh", "JK" to u8
pub fn string_to_card(s: &str) -> Option<u8> {
    if s == "JK" || s == "J1" {
        return Some(52);
    }
    if s == "J2" {
        return Some(53);
    }
    if s.len() != 2 {
        return None;
    }
    let chars: Vec<char> = s.chars().collect();
    let r = chars[0];
    let suit_char = chars[1];

    let rank_idx = match r {
        '2' => 0, '3' => 1, '4' => 2, '5' => 3, '6' => 4, '7' => 5, '8' => 6,
        '9' => 7, 'T' => 8, 'J' => 9, 'Q' => 10, 'K' => 11, 'A' => 12,
        _ => return None,
    };

    let suit_idx = match suit_char {
        's' => 0, 'h' => 1, 'd' => 2, 'c' => 3,
        _ => return None,
    };

    Some(suit_idx * 13 + rank_idx)
}
