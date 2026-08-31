//! The T0-BB policy net, served: five dealt cards in, an ordering over all
//! 232 openings out.
//!
//! This is the ranker's successor at one node.  A shortlist ranker scores
//! each opening by encoding the board it reaches and reading a value net --
//! 232 encodings, one net read each.  The policy reads the *hand* once and
//! emits a logit per action, so the shortlist costs one forward pass instead
//! of 232 encodings, and it is trained on the ordering it is asked for rather
//! than on a value it must be argmaxed through.
//!
//! Everything below is a transcription of `ai/tutor/train_t0_policy.py`
//! (`canonical`, `feats`, `action_index`, `legal_mask`).  The training script
//! is the canon; a disagreement here is not a difference of opinion, it is
//! the model being served an index that means a different placement than the
//! one it was taught.  The order rules that matter, in the order they bite:
//!
//!   1. suits are relabelled by (count desc, ascending-rank-list desc), and
//!      **ties fall back to first appearance in the dealt list** -- Python's
//!      `sorted(dict, ..., reverse=True)` is stable over insertion order, and
//!      a pair like `Ac Ad` ties on both key components;
//!   2. cards then sort by (rank desc, relabelled suit in `cdhs`), jokers
//!      last as `X1`, `X2` by name;
//!   3. the action index is `sum(row_i * 3^i)` over *that* order, with row 0
//!      = top.
//!
//! Consequence of (1) worth stating out loud: the canonicalisation is a
//! function of the dealt list's order, not of the set.  The parity harness
//! must feed Rust and Python the same order, and serving must pass the deal
//! order it actually received.

use anyhow::{anyhow, bail, Result};

use crate::evaluator;

const SUITS: [u8; 4] = [b'c', b'd', b'h', b's'];
const RANKS: &[u8; 13] = b"23456789TJQKA";

pub const FEATURE_SIZE: usize = 54;
pub const ACTION_SIZE: usize = 243;
/// Openings that respect the three-card top; the other 11 indices are masked.
pub const LEGAL_ACTIONS: usize = 232;

fn rank_of(name: &str) -> Result<u8> {
    let ch = name.as_bytes().first().copied().ok_or_else(|| anyhow!("empty card"))?;
    RANKS
        .iter()
        .position(|r| *r == ch)
        .map(|index| index as u8 + 2)
        .ok_or_else(|| anyhow!("card {name} has no rank"))
}

fn suit_index(ch: u8) -> Result<usize> {
    SUITS
        .iter()
        .position(|s| *s == ch)
        .ok_or_else(|| anyhow!("suit {} is not one of cdhs", ch as char))
}

/// `idx` places at most three cards on top.
pub fn legal(index: usize) -> bool {
    let mut code = index;
    let mut tops = 0usize;
    for _ in 0..5 {
        if code % 3 == 0 {
            tops += 1;
        }
        code /= 3;
    }
    tops <= 3
}

/// One dealt hand, canonicalised, with its net read done once.
pub struct T0Policy {
    /// The canonical five, in the order the action index counts positions in.
    canon: Vec<String>,
    /// Original card name -> canonical card name.
    mapping: Vec<(String, String)>,
    /// The raw 243 logits; illegal slots are never consulted.
    logits: Vec<f32>,
}

impl T0Policy {
    /// Canonicalise `draw`, run the net, keep the logits.
    ///
    /// `draw` is the dealt list **in deal order**: see the tie rule above.
    pub fn new(model: &evaluator::Model, draw: &[String]) -> Result<Self> {
        if draw.len() != 5 {
            bail!("the T0 policy wants exactly five cards, got {}", draw.len());
        }
        if model.input_dim != FEATURE_SIZE || model.output_dim != ACTION_SIZE {
            bail!(
                "the T0 policy image is {}->{}, expected {FEATURE_SIZE}->{ACTION_SIZE}",
                model.input_dim,
                model.output_dim
            );
        }
        let (canon, mapping) = canonicalise(draw)?;
        let mut features = vec![0.0f32; FEATURE_SIZE];
        for name in &canon {
            features[feature_slot(name)?] = 1.0;
        }
        let mut logits: Vec<f32> = Vec::new();
        model.predict_all(&features, &mut logits);
        Ok(Self { canon, mapping, logits })
    }

    /// The action index of one opening, given as rows of original card names.
    pub fn action_index(&self, rows: &[Vec<String>; 3]) -> Result<usize> {
        let mut row_of: Vec<(usize, &str)> = Vec::with_capacity(5);
        for (row, cards) in rows.iter().enumerate() {
            for card in cards {
                let canon = self
                    .mapping
                    .iter()
                    .find(|(orig, _)| orig == card)
                    .map(|(_, c)| c.as_str())
                    .ok_or_else(|| anyhow!("{card} is not one of the dealt five"))?;
                row_of.push((row, canon));
            }
        }
        if row_of.len() != 5 {
            bail!("an opening places five cards, got {}", row_of.len());
        }
        let mut index = 0usize;
        let mut power = 1usize;
        for card in &self.canon {
            let row = row_of
                .iter()
                .find(|(_, c)| *c == card.as_str())
                .map(|(row, _)| *row)
                .ok_or_else(|| anyhow!("canonical card {card} is unplaced"))?;
            index += row * power;
            power *= 3;
        }
        Ok(index)
    }

    /// The logit this opening was given.
    pub fn score(&self, rows: &[Vec<String>; 3]) -> Result<f32> {
        let index = self.action_index(rows)?;
        if !legal(index) {
            bail!("opening maps to masked action {index}");
        }
        Ok(self.logits[index])
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn canon(&self) -> &[String] {
        &self.canon
    }
}

fn feature_slot(canon_name: &str) -> Result<usize> {
    let bytes = canon_name.as_bytes();
    if bytes[0] == b'X' {
        let which = canon_name[1..]
            .parse::<usize>()
            .map_err(|_| anyhow!("joker {canon_name} has no index"))?;
        if which < 1 || which > 2 {
            bail!("joker {canon_name} is out of range");
        }
        return Ok(52 + which - 1);
    }
    Ok((rank_of(canon_name)? as usize - 2) * 4 + suit_index(bytes[1])?)
}

/// Suit-canonicalise and order the five; returns the canonical order and the
/// original-to-canonical name map.
fn canonicalise(cards: &[String]) -> Result<(Vec<String>, Vec<(String, String)>)> {
    let mut plain: Vec<&String> = Vec::new();
    let mut jokers: Vec<&String> = Vec::new();
    for card in cards {
        if card.starts_with('X') {
            jokers.push(card);
        } else {
            plain.push(card);
        }
    }
    // Python sorts the joker *names*, then renames them X1..Xn by that order.
    jokers.sort();

    // Insertion-ordered by first appearance, which is the tie-break Python's
    // stable sort over a dict inherits.
    let mut by_suit: Vec<(u8, Vec<u8>)> = Vec::new();
    for card in &plain {
        let suit = card.as_bytes()[1];
        let rank = rank_of(card)?;
        match by_suit.iter_mut().find(|(s, _)| *s == suit) {
            Some((_, ranks)) => ranks.push(rank),
            None => by_suit.push((suit, vec![rank])),
        }
    }
    for (_, ranks) in by_suit.iter_mut() {
        ranks.sort();
    }
    // Descending on (count, ascending-rank-list); `sort_by` is stable, so a
    // full tie keeps first-appearance order exactly as Python does.
    let mut order: Vec<(u8, Vec<u8>)> = by_suit;
    order.sort_by(|a, b| (b.1.len(), &b.1).cmp(&(a.1.len(), &a.1)));
    let relabel: Vec<(u8, u8)> = order
        .iter()
        .enumerate()
        .map(|(index, (suit, _))| (*suit, SUITS[index]))
        .collect();

    let mut renamed: Vec<String> = Vec::with_capacity(plain.len());
    for card in &plain {
        let suit = card.as_bytes()[1];
        let to = relabel
            .iter()
            .find(|(from, _)| *from == suit)
            .map(|(_, to)| *to)
            .ok_or_else(|| anyhow!("suit {} was not relabelled", suit as char))?;
        renamed.push(format!("{}{}", card.as_bytes()[0] as char, to as char));
    }
    // (rank desc, suit index in cdhs); the five are distinct, so no tie.
    let mut keyed: Vec<(u8, usize, String)> = Vec::with_capacity(renamed.len());
    for name in &renamed {
        let rank = rank_of(name)?;
        let suit = suit_index(name.as_bytes()[1])?;
        keyed.push((rank, suit, name.clone()));
    }
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut canon: Vec<String> = keyed.into_iter().map(|(_, _, name)| name).collect();

    let mut mapping: Vec<(String, String)> = Vec::with_capacity(cards.len());
    for (card, name) in plain.iter().zip(renamed.iter()) {
        mapping.push(((*card).clone(), name.clone()));
    }
    for (index, joker) in jokers.iter().enumerate() {
        let name = format!("X{}", index + 1);
        canon.push(name.clone());
        mapping.push(((*joker).clone(), name));
    }
    Ok((canon, mapping))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn legal_count_matches_the_python_mask() {
        assert_eq!((0..ACTION_SIZE).filter(|i| legal(*i)).count(), LEGAL_ACTIONS);
    }

    #[test]
    fn suits_relabel_by_count_then_ranks() {
        // Hearts hold three, diamonds two: hearts become c, diamonds d.
        let (canon, _) = canonicalise(&names(&["2d", "Ah", "Kh", "5d", "7h"])).unwrap();
        assert_eq!(canon, names(&["Ac", "Kc", "7c", "5d", "2d"]));
    }

    #[test]
    fn a_full_tie_keeps_first_appearance() {
        // Both singleton suits hold {A}; the one dealt first becomes c.
        let (_, map) = canonicalise(&names(&["Ad", "Ac", "2h", "3h", "4h"])).unwrap();
        let of = |name: &str| {
            map.iter().find(|(o, _)| o == name).map(|(_, c)| c.clone()).unwrap()
        };
        // Hearts win on count and take c; the tie is then d before h.
        assert_eq!(of("Ad"), "Ad");
        assert_eq!(of("Ac"), "Ah");
    }

    #[test]
    fn jokers_sort_last_by_name() {
        let (canon, map) = canonicalise(&names(&["X2", "Ah", "X1", "Kh", "2h"])).unwrap();
        assert_eq!(canon, names(&["Ac", "Kc", "2c", "X1", "X2"]));
        assert_eq!(map.iter().find(|(o, _)| o == "X2").unwrap().1, "X2");
    }

    #[test]
    fn action_index_counts_positions_in_canonical_order() {
        // Ac Kc 7c 5d 2d, everything on top would be index 0; put the two
        // lowest on the bottom (row 2) at positions 3 and 4.
        let (canon, _) = canonicalise(&names(&["2d", "Ah", "Kh", "5d", "7h"])).unwrap();
        assert_eq!(canon.len(), 5);
        let expected = 2 * 3usize.pow(3) + 2 * 3usize.pow(4);
        assert!(legal(expected));
    }
}
