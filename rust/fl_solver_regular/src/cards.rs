//! Card identity, byte-for-byte compatible with `ofc_hu_m3_engine::cards`.
//!
//! Index layout is `suit * 13 + rank_index`, suits `h,d,c,s`, ranks `2..A`.
//! This mirrors Python `ofc_regular.cards.ALL_CARDS`, so a `u8` here and a
//! `Card` there denote the same physical card and the same JSON token.

pub const RANKS: &[u8; 13] = b"23456789TJQKA";
pub const SUITS: &[u8; 4] = b"hdcs";

pub const DECK_SIZE: usize = 52;

/// Rank in `2..=14` for a card index.
#[inline(always)]
pub const fn rank_of(card: u8) -> u8 {
    (card % 13) + 2
}

/// Suit index in `0..=3` for a card index.
#[inline(always)]
pub const fn suit_of(card: u8) -> u8 {
    card / 13
}

pub fn card_to_token(card: u8) -> String {
    let rank = RANKS[(card % 13) as usize] as char;
    let suit = SUITS[(card / 13) as usize] as char;
    format!("{rank}{suit}")
}

pub fn parse_card(token: &str) -> Result<u8, String> {
    let bytes = token.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("invalid regular-rule card: {token:?}"));
    }
    let rank = RANKS
        .iter()
        .position(|value| *value == bytes[0])
        .ok_or_else(|| format!("invalid regular-rule card: {token:?}"))?;
    let suit = SUITS
        .iter()
        .position(|value| *value == bytes[1])
        .ok_or_else(|| format!("invalid regular-rule card: {token:?}"))?;
    Ok((suit * 13 + rank) as u8)
}

pub fn parse_cards(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut seen = 0_u64;
    for token in text
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
    {
        let card = parse_card(token)?;
        if seen & (1 << card) != 0 {
            return Err(format!("duplicate card: {token:?}"));
        }
        seen |= 1 << card;
        out.push(card);
    }
    Ok(out)
}

pub fn cards_to_tokens(cards: &[u8]) -> Vec<String> {
    cards.iter().copied().map(card_to_token).collect()
}

/// Full deck in engine order.
pub fn full_deck() -> Vec<u8> {
    (0..DECK_SIZE as u8).collect()
}

/// Cards of the 52-card deck not present in `mask`.
pub fn unseen_from_mask(mask: u64) -> Vec<u8> {
    (0..DECK_SIZE as u8)
        .filter(|card| mask & (1_u64 << card) == 0)
        .collect()
}

pub fn mask_of(cards: &[u8]) -> u64 {
    cards.iter().fold(0_u64, |mask, card| mask | (1_u64 << card))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_roundtrip_matches_python_all_cards_order() {
        assert_eq!(card_to_token(0), "2h");
        assert_eq!(card_to_token(12), "Ah");
        assert_eq!(card_to_token(13), "2d");
        assert_eq!(card_to_token(51), "As");
        assert_eq!(parse_card("Qc").unwrap(), 36);
        for card in 0..52_u8 {
            assert_eq!(parse_card(&card_to_token(card)).unwrap(), card);
        }
    }

    #[test]
    fn rank_and_suit_helpers_agree_with_index_layout() {
        assert_eq!(rank_of(0), 2);
        assert_eq!(rank_of(12), 14);
        assert_eq!(suit_of(0), 0);
        assert_eq!(suit_of(51), 3);
    }
}
