//! No-joker regular OFC cards.
//!
//! The numeric identity deliberately matches Python `ofc_regular.cards.ALL_CARDS`:
//! suits are `h,d,c,s`, ranks within each suit are `2..A`.

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, str::FromStr};

pub const RANKS: &str = "23456789TJQKA";
pub const SUITS: &str = "hdcs";

const CARD_TOKENS: [&str; 52] = [
    "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "Th", "Jh", "Qh", "Kh", "Ah", "2d", "3d", "4d",
    "5d", "6d", "7d", "8d", "9d", "Td", "Jd", "Qd", "Kd", "Ad", "2c", "3c", "4c", "5c", "6c", "7c",
    "8c", "9c", "Tc", "Jc", "Qc", "Kc", "Ac", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "Ts",
    "Js", "Qs", "Ks", "As",
];

/// Compact card identity in Python `ALL_CARDS` order.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Card(u8);

impl Card {
    pub const fn from_index_unchecked(index: u8) -> Self {
        Self(index)
    }

    pub fn from_index(index: u8) -> Result<Self, String> {
        if index < 52 {
            Ok(Self(index))
        } else {
            Err(format!("card index is outside the 52-card domain: {index}"))
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn rank(self) -> u8 {
        (self.0 % 13) + 2
    }

    pub const fn suit_index(self) -> u8 {
        self.0 / 13
    }

    pub const fn as_str(self) -> &'static str {
        CARD_TOKENS[self.0 as usize]
    }

    pub const fn bit(self) -> u64 {
        1_u64 << self.0
    }
}

pub const ALL_CARDS: [Card; 52] = [
    Card(0),
    Card(1),
    Card(2),
    Card(3),
    Card(4),
    Card(5),
    Card(6),
    Card(7),
    Card(8),
    Card(9),
    Card(10),
    Card(11),
    Card(12),
    Card(13),
    Card(14),
    Card(15),
    Card(16),
    Card(17),
    Card(18),
    Card(19),
    Card(20),
    Card(21),
    Card(22),
    Card(23),
    Card(24),
    Card(25),
    Card(26),
    Card(27),
    Card(28),
    Card(29),
    Card(30),
    Card(31),
    Card(32),
    Card(33),
    Card(34),
    Card(35),
    Card(36),
    Card(37),
    Card(38),
    Card(39),
    Card(40),
    Card(41),
    Card(42),
    Card(43),
    Card(44),
    Card(45),
    Card(46),
    Card(47),
    Card(48),
    Card(49),
    Card(50),
    Card(51),
];

impl fmt::Display for Card {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Card {
    type Err = String;

    fn from_str(token: &str) -> Result<Self, Self::Err> {
        let bytes = token.as_bytes();
        if bytes.len() != 2 {
            return Err(format!("invalid regular-rule card: {token:?}"));
        }
        let rank_index = RANKS.as_bytes().iter().position(|rank| *rank == bytes[0]);
        let suit_index = SUITS.as_bytes().iter().position(|suit| *suit == bytes[1]);
        match (rank_index, suit_index) {
            (Some(rank), Some(suit)) => Ok(Self((suit * 13 + rank) as u8)),
            _ => Err(format!("invalid regular-rule card: {token:?}")),
        }
    }
}

impl Serialize for Card {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Card {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Use an owned string so `serde_json::from_value` works as well as
        // borrowed `from_str`/`from_slice` inputs at the FFI boundary.
        let token = String::deserialize(deserializer)?;
        token.parse().map_err(de::Error::custom)
    }
}

/// Validate uniqueness. Every `Card` value is already in-domain by construction.
pub fn validate_cards(cards: &[Card]) -> Result<(), String> {
    let mut mask = 0_u64;
    for &card in cards {
        let bit = card.bit();
        if mask & bit != 0 {
            return Err(format!("duplicate card: {:?}", card.as_str()));
        }
        mask |= bit;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_all_cards_order_and_serde_are_exact() {
        assert_eq!(ALL_CARDS[0].as_str(), "2h");
        assert_eq!(ALL_CARDS[12].as_str(), "Ah");
        assert_eq!(ALL_CARDS[13].as_str(), "2d");
        assert_eq!(ALL_CARDS[51].as_str(), "As");
        assert_eq!("Qc".parse::<Card>().unwrap().index(), 36);
        assert_eq!(
            serde_json::to_string(&"Qc".parse::<Card>().unwrap()).unwrap(),
            "\"Qc\""
        );
        assert_eq!(
            serde_json::from_str::<Card>("\"Qc\"").unwrap().as_str(),
            "Qc"
        );
    }

    #[test]
    fn duplicate_cards_fail_closed() {
        let ace = "Ah".parse().unwrap();
        assert!(validate_cards(&[ace, ace])
            .unwrap_err()
            .contains("duplicate"));
        assert!("1h".parse::<Card>().is_err());
        assert!(Card::from_index(52).is_err());
    }
}
