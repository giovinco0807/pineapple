//! Board state primitives with regular OFC row capacities.

use crate::cards::{validate_cards, Card};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::{fmt, str::FromStr};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Row {
    Top,
    Middle,
    Bottom,
}

pub const ALL_ROWS: [Row; 3] = [Row::Top, Row::Middle, Row::Bottom];

impl Row {
    pub const fn capacity(self) -> usize {
        match self {
            Self::Top => 3,
            Self::Middle | Self::Bottom => 5,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::Middle => "middle",
            Self::Bottom => "bottom",
        }
    }
}

impl fmt::Display for Row {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Row {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "top" => Ok(Self::Top),
            "middle" => Ok(Self::Middle),
            "bottom" => Ok(Self::Bottom),
            _ => Err(format!("unknown row: {value}")),
        }
    }
}

/// A public OFC board. Deserialization validates capacities and duplicates.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Board {
    pub top: Vec<Card>,
    pub middle: Vec<Card>,
    pub bottom: Vec<Card>,
}

impl<'de> Deserialize<'de> for Board {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawBoard {
            #[serde(default)]
            top: Vec<Card>,
            #[serde(default)]
            middle: Vec<Card>,
            #[serde(default)]
            bottom: Vec<Card>,
        }

        let raw = RawBoard::deserialize(deserializer)?;
        Self::new(raw.top, raw.middle, raw.bottom).map_err(de::Error::custom)
    }
}

impl Board {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn new(top: Vec<Card>, middle: Vec<Card>, bottom: Vec<Card>) -> Result<Self, String> {
        let board = Self {
            top,
            middle,
            bottom,
        };
        board.validate()?;
        Ok(board)
    }

    pub fn validate(&self) -> Result<(), String> {
        for row in ALL_ROWS {
            if self.cards(row).len() > row.capacity() {
                return Err(format!("{} row exceeds capacity", row.as_str()));
            }
        }
        validate_cards(&self.all_cards())
    }

    pub fn cards(&self, row: Row) -> &[Card] {
        match row {
            Row::Top => &self.top,
            Row::Middle => &self.middle,
            Row::Bottom => &self.bottom,
        }
    }

    fn cards_mut(&mut self, row: Row) -> &mut Vec<Card> {
        match row {
            Row::Top => &mut self.top,
            Row::Middle => &mut self.middle,
            Row::Bottom => &mut self.bottom,
        }
    }

    pub fn all_cards(&self) -> Vec<Card> {
        self.top
            .iter()
            .chain(&self.middle)
            .chain(&self.bottom)
            .copied()
            .collect()
    }

    pub fn card_count(&self) -> usize {
        self.top.len() + self.middle.len() + self.bottom.len()
    }

    pub fn open_slots(&self, row: Row) -> usize {
        row.capacity().saturating_sub(self.cards(row).len())
    }

    pub fn is_complete(&self) -> bool {
        self.card_count() == 13 && ALL_ROWS.iter().all(|&row| self.open_slots(row) == 0)
    }

    pub fn place(&self, placements: &[(Card, Row)]) -> Result<Self, String> {
        self.validate()?;
        let next = self.place_trusted(placements);
        next.validate()?;
        Ok(next)
    }

    /// Apply already validated, capacity-safe placements inside the native
    /// search tree. Public callers continue to use [`Board::place`].
    pub(crate) fn place_trusted(&self, placements: &[(Card, Row)]) -> Self {
        let mut next = self.clone();
        for &(card, row) in placements {
            next.cards_mut(row).push(card);
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(value: &str) -> Card {
        value.parse().unwrap()
    }

    #[test]
    fn capacities_and_placement_match_python_board() {
        let board = Board::new(
            vec![card("Qh"), card("2d")],
            vec![card("Kh"), card("Kd"), card("6c"), card("8s"), card("Th")],
            vec![card("9c"), card("9d"), card("9s"), card("Kc")],
        )
        .unwrap();
        assert_eq!(board.card_count(), 11);
        assert_eq!(board.open_slots(Row::Top), 1);
        assert_eq!(board.open_slots(Row::Middle), 0);
        assert_eq!(board.open_slots(Row::Bottom), 1);
        let complete = board
            .place(&[(card("Qs"), Row::Top), (card("Ah"), Row::Bottom)])
            .unwrap();
        assert!(complete.is_complete());
    }

    #[test]
    fn invalid_deserialized_board_fails_closed() {
        let duplicate = r#"{"top":["Ah"],"middle":["Ah"],"bottom":[]}"#;
        assert!(serde_json::from_str::<Board>(duplicate).is_err());
        let overflow = r#"{"top":["2h","3h","4h","5h"],"middle":[],"bottom":[]}"#;
        assert!(serde_json::from_str::<Board>(overflow).is_err());
    }
}
