//! The T0-BTN policy net, served: hero's five dealt cards and BB's placed
//! five in, an ordering over hero's 232 openings out.
//!
//! The Button analogue of `t0_policy` (the T0-BB fence).  Within a street BB
//! acts first, so the Button opening is chosen against five cards already on
//! the table, and the decision state is the pair (hero's five, BB's board).
//! The net reads both halves in one forward pass and emits a logit per
//! action: no per-candidate encoding, and no ranker at that node.
//!
//! Everything below is one half of a contract shared with the Python trainer
//! (`ai/tutor/train_t0_btn_policy.py`), whose acceptance test is bit-for-bit
//! rank agreement over all 232 openings on real states.  It is a
//! transcription, not a design: a disagreement here is the model being
//! served an index that means a different placement than the one it was
//! taught.  The rules, in the order they bite:
//!
//!   1. **Joint suit canonicalisation, order-independent.**  For each suit
//!      s in "cdhs", key(s) = (hero_mask, opp_top_mask, opp_mid_mask,
//!      opp_bot_mask): 13-bit rank masks over the naturals, 2 = bit 0 ...
//!      A = bit 12.  Suits sort by key DESCENDING (lexicographic tuple
//!      compare); a residual tie (all four masks equal) falls back to the
//!      original suit's index in "cdhs" ASCENDING.  The sorted suits become
//!      c, d, h, s in that order.  This is a function of the card *sets* --
//!      neither the dealt order nor a row's internal order enters, which is
//!      where it differs from the BB rule and its first-appearance tie.
//!   2. Hero naturals sort by (rank desc, canonical suit index asc); hero
//!      jokers are renumbered X1..Xk by sorted original name and go last.
//!      That is the position order the action index counts in.
//!   3. Opponent jokers carry no name: a per-row count only.  Jokers are
//!      numbered X1 then X2 across the whole deal, so hero may hold X2 while
//!      BB holds X1; rule 2 folds that away.
//!   4. Features, 213 dims in this order: hero 54 (natural slot
//!      `(rank-2)*4 + canonical suit`, joker slots 52 + i); opponent
//!      top/mid/bot, 3 x 52 one-hot naturals; opponent joker count per row,
//!      x 3, as 0/1/2.  No standardisation: the image carries mean 0, std 1.
//!   5. The action index is `sum(row_i * 3^i)` over the canonical hero order
//!      with row 0 = top, 243 slots, of which the 11 that put more than
//!      three on top are masked; `t0_policy::legal` is the one decoder.
//!
//! `t0_policy` (the BB type) is not touched by any of this: its rules differ
//! in (1), and the two nets are trained on their own indices.

use anyhow::{anyhow, bail, Result};

use crate::evaluator;
use crate::t0_policy::{legal, ACTION_SIZE};

const SUITS: [u8; 4] = [b'c', b'd', b'h', b's'];
const RANKS: &[u8; 13] = b"23456789TJQKA";
const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

/// hero 54 + opponent 3 x 52 + opponent joker counts 3.
pub const FEATURE_SIZE: usize = 213;
/// Hero's two joker slots sit after the 52 naturals.
const HERO_JOKER_BASE: usize = 52;
const HERO_BLOCK: usize = 54;
const OPP_ROW_BLOCK: usize = 52;
const OPP_JOKER_BASE: usize = HERO_BLOCK + 3 * OPP_ROW_BLOCK;

/// One suit's sort key: (hero, opp top, opp mid, opp bot) rank masks.
pub(crate) type SuitKey = [u16; 4];

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

/// A natural card as (rank, original suit index in "cdhs").
fn natural(name: &str) -> Result<(u8, usize)> {
    let bytes = name.as_bytes();
    if bytes.len() != 2 {
        bail!("bad card name {name} (want e.g. As, Td, 7c, X1)");
    }
    Ok((rank_of(name)?, suit_index(bytes[1])?))
}

fn is_joker(name: &str) -> bool {
    name.starts_with('X')
}

/// The four suit keys of a decision state, indexed by original suit.
///
/// Masks only: the checks on the state (five and five, no shared card, at
/// most two jokers) live in `BtnState::new`, which is the one entry point.
pub(crate) fn suit_keys(hero: &[String], opp: &[Vec<String>; 3]) -> Result<[SuitKey; 4]> {
    let mut keys: [SuitKey; 4] = [[0; 4]; 4];
    for card in hero {
        if is_joker(card) {
            continue;
        }
        let (rank, suit) = natural(card)?;
        keys[suit][0] |= 1 << (rank - 2);
    }
    for (row, cards) in opp.iter().enumerate() {
        for card in cards {
            if is_joker(card) {
                continue;
            }
            let (rank, suit) = natural(card)?;
            keys[suit][1 + row] |= 1 << (rank - 2);
        }
    }
    Ok(keys)
}

/// One decision state canonicalised, no net involved: the position order the
/// action index counts in, the original-to-canonical name map, and the 213
/// features.  Split from the net so the contract can be tested without
/// weights.
pub struct BtnState {
    canon: Vec<String>,
    mapping: Vec<(String, String)>,
    features: Vec<f32>,
}

impl BtnState {
    /// `hero` is the dealt five in any order; `opp` is BB's placed board,
    /// rows in any internal order.  Both may spell jokers with whatever
    /// numbers the deal gave them.
    pub fn new(hero: &[String], opp: &[Vec<String>; 3]) -> Result<Self> {
        if hero.len() != 5 {
            bail!("the T0-BTN policy wants exactly five hero cards, got {}", hero.len());
        }
        let placed: usize = opp.iter().map(Vec::len).sum();
        if placed != 5 {
            bail!("the T0-BTN policy wants BB's placed five, got {placed}");
        }
        for (row, cards) in opp.iter().enumerate() {
            if cards.len() > ROW_CAPACITY[row] {
                bail!("opponent row {row} holds {} of {}", cards.len(), ROW_CAPACITY[row]);
            }
        }
        // Naturals are compared by name and jokers by count, the split the
        // deck accounting uses everywhere.
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut hero_naturals: Vec<(u8, usize, &String)> = Vec::with_capacity(5);
        let mut hero_jokers: Vec<&String> = Vec::new();
        let mut opp_jokers = [0usize; 3];
        for card in hero {
            if is_joker(card) {
                hero_jokers.push(card);
                continue;
            }
            let (rank, suit) = natural(card)?;
            if !seen.insert(card) {
                bail!("card {card} appears twice");
            }
            hero_naturals.push((rank, suit, card));
        }
        for (row, cards) in opp.iter().enumerate() {
            for card in cards {
                if is_joker(card) {
                    opp_jokers[row] += 1;
                    continue;
                }
                natural(card)?;
                if !seen.insert(card) {
                    bail!("card {card} appears twice");
                }
            }
        }
        if hero_jokers.len() + opp_jokers.iter().sum::<usize>() > 2 {
            bail!("more than two jokers between hero's five and BB's five");
        }
        // Rule 2's joker tail: renumbered by sorted original name.
        hero_jokers.sort();
        if hero_jokers.windows(2).any(|pair| pair[0] == pair[1]) {
            bail!("hero joker {} appears twice", hero_jokers[0]);
        }

        // Rule 1: suits by key descending, residual ties by original index
        // ascending.  `relabel[original] = canonical`.
        let keys = suit_keys(hero, opp)?;
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|a, b| keys[*b].cmp(&keys[*a]).then(a.cmp(b)));
        let mut relabel = [0usize; 4];
        for (canonical, original) in order.iter().enumerate() {
            relabel[*original] = canonical;
        }

        // Rule 2: hero naturals by (rank desc, canonical suit asc).
        let mut keyed: Vec<(u8, usize, String, &String)> = hero_naturals
            .iter()
            .map(|(rank, suit, name)| {
                let canonical = relabel[*suit];
                let renamed =
                    format!("{}{}", RANKS[(*rank - 2) as usize] as char, SUITS[canonical] as char);
                (*rank, canonical, renamed, *name)
            })
            .collect();
        keyed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        // Rule 4: the feature layout.
        let mut features = vec![0.0f32; FEATURE_SIZE];
        let mut canon: Vec<String> = Vec::with_capacity(5);
        let mut mapping: Vec<(String, String)> = Vec::with_capacity(5);
        for (rank, suit, renamed, original) in &keyed {
            features[(*rank as usize - 2) * 4 + suit] = 1.0;
            canon.push(renamed.clone());
            mapping.push(((*original).clone(), renamed.clone()));
        }
        for (index, joker) in hero_jokers.iter().enumerate() {
            features[HERO_JOKER_BASE + index] = 1.0;
            let renamed = format!("X{}", index + 1);
            canon.push(renamed.clone());
            mapping.push(((*joker).clone(), renamed));
        }
        for (row, cards) in opp.iter().enumerate() {
            for card in cards {
                if is_joker(card) {
                    continue;
                }
                let (rank, suit) = natural(card)?;
                features[HERO_BLOCK + row * OPP_ROW_BLOCK + (rank as usize - 2) * 4 + relabel[suit]] =
                    1.0;
            }
            features[OPP_JOKER_BASE + row] = opp_jokers[row] as f32;
        }
        Ok(Self { canon, mapping, features })
    }

    /// Rule 5: the action index of one opening, given as rows of original
    /// card names.
    pub fn action_index(&self, rows: &[Vec<String>; 3]) -> Result<usize> {
        let mut row_of: Vec<(usize, &str)> = Vec::with_capacity(5);
        for (row, cards) in rows.iter().enumerate() {
            for card in cards {
                let canon = self
                    .mapping
                    .iter()
                    .find(|(orig, _)| orig == card)
                    .map(|(_, c)| c.as_str())
                    .ok_or_else(|| anyhow!("{card} is not one of hero's five"))?;
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

    /// The canonical five, in the order the action index counts positions in.
    pub fn canon(&self) -> &[String] {
        &self.canon
    }

    /// Original card name -> canonical card name, hero's five only.
    pub fn mapping(&self) -> &[(String, String)] {
        &self.mapping
    }

    pub fn features(&self) -> &[f32] {
        &self.features
    }
}

/// One decision state, canonicalised, with its net read done once.
pub struct T0PolicyBtn {
    state: BtnState,
    /// The raw 243 logits; illegal slots are never consulted.
    logits: Vec<f32>,
}

impl T0PolicyBtn {
    /// Canonicalise (hero's five, BB's placed five), run the net, keep the
    /// logits.
    pub fn new(model: &evaluator::Model, hero: &[String], opp: &[Vec<String>; 3]) -> Result<Self> {
        if model.input_dim != FEATURE_SIZE || model.output_dim != ACTION_SIZE {
            bail!(
                "the T0-BTN policy image is {}->{}, expected {FEATURE_SIZE}->{ACTION_SIZE}",
                model.input_dim,
                model.output_dim
            );
        }
        let state = BtnState::new(hero, opp)?;
        let mut logits: Vec<f32> = Vec::new();
        model.predict_all(&state.features, &mut logits);
        Ok(Self { state, logits })
    }

    pub fn action_index(&self, rows: &[Vec<String>; 3]) -> Result<usize> {
        self.state.action_index(rows)
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
        self.state.canon()
    }

    pub fn features(&self) -> &[f32] {
        self.state.features()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::t0_policy::LEGAL_ACTIONS;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn rows(list: [&[&str]; 3]) -> [Vec<String>; 3] {
        [names(list[0]), names(list[1]), names(list[2])]
    }

    /// Every arrangement of the five by *list* position: (mask, rows).
    fn openings_by_mask(hero: &[String]) -> Vec<(usize, [Vec<String>; 3])> {
        let mut out = Vec::new();
        for mask in 0..3usize.pow(5) {
            let mut placed: [Vec<String>; 3] = Default::default();
            let mut code = mask;
            for card in hero {
                placed[code % 3].push(card.clone());
                code /= 3;
            }
            if placed[0].len() > ROW_CAPACITY[0] {
                continue;
            }
            out.push((mask, placed));
        }
        out
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        fn go(current: &mut Vec<usize>, used: &mut Vec<bool>, out: &mut Vec<Vec<usize>>) {
            if current.len() == used.len() {
                out.push(current.clone());
                return;
            }
            for i in 0..used.len() {
                if !used[i] {
                    used[i] = true;
                    current.push(i);
                    go(current, used, out);
                    current.pop();
                    used[i] = false;
                }
            }
        }
        let mut out = Vec::new();
        go(&mut Vec::new(), &mut vec![false; n], &mut out);
        out
    }

    fn permute_suits(name: &str, perm: &[usize]) -> String {
        if is_joker(name) {
            return name.to_string();
        }
        let suit = suit_index(name.as_bytes()[1]).unwrap();
        format!("{}{}", name.as_bytes()[0] as char, SUITS[perm[suit]] as char)
    }

    fn permute_rows(rows: &[Vec<String>; 3], perm: &[usize]) -> [Vec<String>; 3] {
        [0, 1, 2].map(|r| rows[r].iter().map(|n| permute_suits(n, perm)).collect())
    }

    /// A two-layer T4F1 image of any width, mean 0 / std 1, deterministic
    /// pseudo-random weights.  The stand-in for a trained policy net.
    pub(crate) fn synthetic_wide_model(
        input_dim: usize,
        hidden: usize,
        output_dim: usize,
        seed: u32,
    ) -> evaluator::Model {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"T4F1");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version
        bytes.extend_from_slice(&2u32.to_le_bytes()); // two layers
        bytes.extend_from_slice(&(input_dim as u32).to_le_bytes());
        for _ in 0..input_dim {
            bytes.extend_from_slice(&0.0f32.to_le_bytes()); // mean
        }
        for _ in 0..input_dim {
            bytes.extend_from_slice(&1.0f32.to_le_bytes()); // std
        }
        let mut counter = 0u32;
        let mut next = |scale: f32| -> f32 {
            counter = counter.wrapping_add(1);
            let mixed = counter.wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((mixed % 2_003) as f32 / 1_001.0 - 1.0) * scale
        };
        for (inputs, outputs) in [(input_dim, hidden), (hidden, output_dim)] {
            bytes.extend_from_slice(&(inputs as u32).to_le_bytes());
            bytes.extend_from_slice(&(outputs as u32).to_le_bytes());
            for _ in 0..inputs * outputs {
                bytes.extend_from_slice(&next(0.5).to_le_bytes());
            }
            for _ in 0..outputs {
                bytes.extend_from_slice(&next(0.1).to_le_bytes());
            }
        }
        evaluator::Model::load_wide(&bytes).expect("synthetic wide model image")
    }

    fn example() -> (Vec<String>, [Vec<String>; 3]) {
        (
            names(&["As", "Kd", "7c", "7h", "2s"]),
            rows([&["Qs"], &["Jd", "Th"], &["9c", "8c"]]),
        )
    }

    fn set_slots(features: &[f32]) -> Vec<usize> {
        features
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// The layout written out by hand on one state.
    #[test]
    fn the_feature_layout_on_a_hand_built_example() {
        let (hero, opp) = example();
        let state = BtnState::new(&hero, &opp).unwrap();
        // Keys: s (4097,1024,0,0) > d (2048,0,512,0) > h (32,0,256,0)
        // > c (32,0,0,192), so s->c, d->d, h->h, c->s.
        assert_eq!(state.canon(), names(&["Ac", "Kd", "7h", "7s", "2c"]));
        assert_eq!(state.features().len(), FEATURE_SIZE);
        // Hero: Ac 48, Kd 45, 7h 22, 7s 23, 2c 0.  Opp top Qc 54+40 = 94;
        // mid Jd 106+37 = 143, Th 106+34 = 140; bot 9s 158+31 = 189,
        // 8s 158+27 = 185.  No jokers anywhere: 210..213 stay 0.
        assert_eq!(set_slots(state.features()), vec![0, 22, 23, 45, 48, 94, 140, 143, 185, 189]);
        assert!(state.features()[OPP_JOKER_BASE..].iter().all(|v| *v == 0.0));
        // top 2s | mid 7c,7h | bot As,Kd: rows over the canonical order are
        // [2, 2, 1, 1, 0] -> 2 + 6 + 9 + 27 + 0.
        let opening = rows([&["2s"], &["7c", "7h"], &["As", "Kd"]]);
        assert_eq!(state.action_index(&opening).unwrap(), 44);
        assert!(legal(44));
        // Everything on top is masked, and `score` refuses it.
        let stacked = rows([&["As", "Kd", "7c", "7h", "2s"], &[], &[]]);
        assert!(!legal(state.action_index(&stacked).unwrap()));
    }

    /// Hero holding the deal's X2 while BB holds X1 is the same state as the
    /// other way round: hero's joker is X1, BB's is a count.
    #[test]
    fn hero_x2_beside_the_opponents_x1_is_renumbered_x1() {
        let hero = names(&["X2", "Ah", "Kh", "2h", "3d"]);
        let opp = rows([&["X1"], &["Qc", "Jc"], &["9s", "8s"]]);
        let state = BtnState::new(&hero, &opp).unwrap();
        // h (6145,..) > d (2,..) > c (0,0,1536,0) > s (0,0,0,192):
        // h->c, d->d, c->h, s->s.
        assert_eq!(state.canon(), names(&["Ac", "Kc", "3d", "2c", "X1"]));
        assert_eq!(state.mapping().iter().find(|(o, _)| o == "X2").unwrap().1, "X1");
        let f = state.features();
        // Ac 48, Kc 44, 3d 5, 2c 0, X1 52; mid Qh 148, Jh 144; bot 9s 189,
        // 8s 185; BB's joker is the top-row count at 210.
        assert_eq!(set_slots(f), vec![0, 5, 44, 48, 52, 144, 148, 185, 189, 210]);
        assert_eq!(f[52], 1.0);
        assert_eq!(f[53], 0.0);
        assert_eq!(&f[210..213], &[1.0, 0.0, 0.0]);
        assert_eq!(f.iter().sum::<f32>(), 10.0);
        // The deal numbering the other way round reads identically ...
        let hero2 = names(&["X1", "Ah", "Kh", "2h", "3d"]);
        let opp2 = rows([&["X2"], &["Qc", "Jc"], &["9s", "8s"]]);
        let state2 = BtnState::new(&hero2, &opp2).unwrap();
        assert_eq!(state2.canon(), state.canon());
        assert_eq!(state2.features(), state.features());
        // ... and "the joker on top" is the same action under both spellings.
        let a = rows([&["X2"], &["Ah", "Kh"], &["2h", "3d"]]);
        let b = rows([&["X1"], &["Ah", "Kh"], &["2h", "3d"]]);
        assert_eq!(state.action_index(&a).unwrap(), state2.action_index(&b).unwrap());
        // Two hero jokers: X1 then X2 by sorted name, slots 52 and 53.
        let hero3 = names(&["X2", "X1", "Ah", "Kh", "2h"]);
        let opp3 = rows([&["Qc"], &["Jc", "3d"], &["9s", "8s"]]);
        let state3 = BtnState::new(&hero3, &opp3).unwrap();
        assert_eq!(state3.canon(), names(&["Ac", "Kc", "2c", "X1", "X2"]));
        assert_eq!(state3.features()[52], 1.0);
        assert_eq!(state3.features()[53], 1.0);
    }

    /// Relabelling the suits of hero and BB together changes nothing: not
    /// the features, not the index of any of the 232 openings.
    #[test]
    fn canonicalisation_is_invariant_under_every_joint_suit_permutation() {
        let states = [
            example(),
            (
                names(&["X2", "Ah", "Kh", "2h", "3d"]),
                rows([&["X1"], &["Qc", "Jc"], &["9s", "8s"]]),
            ),
            (
                names(&["Tc", "Td", "Th", "Ts", "X1"]),
                rows([&["2c", "3d"], &["4h", "5s", "6c"], &[]]),
            ),
        ];
        for (hero, opp) in &states {
            // With a residual tie the index would depend on the tie rule
            // rather than on the permutation, and the test would be asserting
            // the wrong thing; that rule has its own test below.
            let keys = suit_keys(hero, opp).unwrap();
            for a in 0..4 {
                for b in a + 1..4 {
                    assert_ne!(keys[a], keys[b], "state {hero:?} has a residual tie");
                }
            }
            let base = BtnState::new(hero, opp).unwrap();
            let openings = openings_by_mask(hero);
            assert_eq!(openings.len(), LEGAL_ACTIONS);
            let perms = permutations(4);
            assert_eq!(perms.len(), 24);
            for perm in &perms {
                let hero_p: Vec<String> = hero.iter().map(|n| permute_suits(n, perm)).collect();
                let opp_p = permute_rows(opp, perm);
                let state = BtnState::new(&hero_p, &opp_p).unwrap();
                assert_eq!(state.canon(), base.canon(), "perm {perm:?}");
                assert_eq!(state.features(), base.features(), "perm {perm:?}");
                for (mask, placed) in &openings {
                    let placed_p = permute_rows(placed, perm);
                    assert_eq!(
                        state.action_index(&placed_p).unwrap(),
                        base.action_index(placed).unwrap(),
                        "perm {perm:?} mask {mask}"
                    );
                }
            }
        }
    }

    /// A function of the sets: the dealt order and the rows' internal order
    /// leave the canon, the map, the features and every index unchanged.
    #[test]
    fn the_dealt_order_and_the_rows_internal_order_do_not_enter() {
        let (hero, opp) = example();
        let base = BtnState::new(&hero, &opp).unwrap();
        let openings = openings_by_mask(&hero);
        let mut base_map = base.mapping().to_vec();
        base_map.sort();
        let perms = permutations(5);
        assert_eq!(perms.len(), 120);
        for perm in &perms {
            let shuffled: Vec<String> = perm.iter().map(|i| hero[*i].clone()).collect();
            for flip in [false, true] {
                let opp_r: [Vec<String>; 3] = [0, 1, 2].map(|r| {
                    let mut row = opp[r].clone();
                    if flip {
                        row.reverse();
                    }
                    row
                });
                let state = BtnState::new(&shuffled, &opp_r).unwrap();
                assert_eq!(state.canon(), base.canon(), "perm {perm:?} flip {flip}");
                assert_eq!(state.features(), base.features(), "perm {perm:?} flip {flip}");
                let mut map = state.mapping().to_vec();
                map.sort();
                assert_eq!(map, base_map, "perm {perm:?} flip {flip}");
                for (mask, placed) in &openings {
                    assert_eq!(
                        state.action_index(placed).unwrap(),
                        base.action_index(placed).unwrap(),
                        "perm {perm:?} flip {flip} mask {mask}"
                    );
                }
            }
        }
    }

    /// All four masks equal: the original "cdhs" index decides, ascending,
    /// whichever order the cards arrived in.
    #[test]
    fn a_residual_tie_falls_back_to_the_original_suit_index() {
        // d and h both hold exactly {A} for hero and nothing for BB.
        let hero = names(&["Ah", "Ad", "Kc", "Qc", "Jc"]);
        let opp = rows([&["2s"], &["3s", "4s"], &["5s", "6s"]]);
        let keys = suit_keys(&hero, &opp).unwrap();
        assert_eq!(keys[1], keys[2], "the test needs a full tie between d and h");
        let state = BtnState::new(&hero, &opp).unwrap();
        let of = |s: &BtnState, name: &str| {
            s.mapping().iter().find(|(o, _)| o == name).map(|(_, c)| c.clone()).unwrap()
        };
        // d (index 1) before h (index 3): d -> c, h -> d, then c -> h, s -> s.
        assert_eq!(of(&state, "Ad"), "Ac");
        assert_eq!(of(&state, "Ah"), "Ad");
        assert_eq!(of(&state, "Kc"), "Kh");
        assert_eq!(state.canon(), names(&["Ac", "Ad", "Kh", "Qh", "Jh"]));
        assert_eq!(set_slots(state.features()).len(), 10);
        // Dealt the other way round: the same set, the same tie, the same
        // answer -- there is no first-appearance rule here.
        let reversed: Vec<String> = hero.iter().rev().cloned().collect();
        let again = BtnState::new(&reversed, &opp).unwrap();
        assert_eq!(of(&again, "Ad"), "Ac");
        assert_eq!(of(&again, "Ah"), "Ad");
        assert_eq!(again.canon(), state.canon());
        assert_eq!(again.features(), state.features());
    }

    #[test]
    fn the_232_openings_map_onto_the_232_legal_actions() {
        let hero = names(&["X2", "Ah", "Kh", "2h", "3d"]);
        let opp = rows([&["X1"], &["Qc", "Jc"], &["9s", "8s"]]);
        let state = BtnState::new(&hero, &opp).unwrap();
        let indices: std::collections::BTreeSet<usize> = openings_by_mask(&hero)
            .iter()
            .map(|(_, placed)| state.action_index(placed).unwrap())
            .collect();
        assert_eq!(indices.len(), LEGAL_ACTIONS);
        assert!(indices.iter().all(|i| legal(*i)));
        assert_eq!(indices.len(), (0..ACTION_SIZE).filter(|i| legal(*i)).count());
    }

    /// The net reads 213 and nothing else; a BB image is refused by width,
    /// and a malformed state by shape.
    #[test]
    fn the_net_reads_213_and_refuses_the_bb_width() {
        let (hero, opp) = example();
        let wide = synthetic_wide_model(FEATURE_SIZE, 16, ACTION_SIZE, 0x5555_0000);
        let policy = T0PolicyBtn::new(&wide, &hero, &opp).unwrap();
        assert_eq!(policy.logits().len(), ACTION_SIZE);
        assert_eq!(policy.features().len(), FEATURE_SIZE);
        let opening = rows([&["2s"], &["7c", "7h"], &["As", "Kd"]]);
        assert_eq!(policy.score(&opening).unwrap(), policy.logits()[44]);
        let stacked = rows([&["As", "Kd", "7c", "7h", "2s"], &[], &[]]);
        assert!(policy.score(&stacked).is_err());
        let bb = synthetic_wide_model(54, 16, ACTION_SIZE, 0x5555_0000);
        assert!(T0PolicyBtn::new(&bb, &hero, &opp).is_err());
        // Four hero cards, four placed, a shared card, four on top.
        assert!(BtnState::new(&hero[..4], &opp).is_err());
        assert!(BtnState::new(&hero, &rows([&["Qs"], &["Jd", "Th"], &["9c"]])).is_err());
        assert!(BtnState::new(&hero, &rows([&["As"], &["Jd", "Th"], &["9c", "8c"]])).is_err());
        assert!(BtnState::new(&hero, &rows([&["Qs", "Jd", "Th", "9c"], &["8c"], &[]])).is_err());
        // Three jokers across the pair cannot be dealt.
        let hero_j = names(&["X1", "X2", "Ah", "Kh", "2h"]);
        assert!(BtnState::new(&hero_j, &rows([&["X1"], &["Qc", "Jc"], &["9s", "8s"]])).is_err());
    }
}
