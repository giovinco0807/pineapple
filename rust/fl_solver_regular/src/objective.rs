//! The Fantasyland placement objective and its reference hero distribution.
//!
//! # Why the objective is shaped this way
//!
//! Write the Fantasyland player's terminal score against a normal hero exactly
//! as `ofc_regular.estimate_hu_fl_ev_direct.score_fl_vs_normal` does, and split
//! on whether the hero fouls. `P` is the hero's foul probability, `R` the FL
//! player's royalty total, `F` the FL player's next-Fantasyland value:
//!
//! ```text
//! E[score] = P * (6 + R + F)
//!          + (1 - P) * ( E[line + scoop | hero legal] + R - E[R_hero] + F - E[F_hero] )
//!          = R + F + (1 - P) * E[line + scoop | hero legal] + constant
//! ```
//!
//! The hero cannot see the Fantasyland board, so `P` does not depend on the
//! arrangement and the hero's own royalty and entry value are constants. Three
//! consequences drive the implementation:
//!
//! * `R` and `F` both carry coefficient one, so the familiar
//!   `royalty + stay_bonus` objective is the exact argmax whenever the line
//!   term is dropped. That is objective `pure_v1`.
//! * The line term is scaled by `1 - P`, not by one. `hero_foul_prob` is a
//!   configured constant with recorded provenance, never a literal.
//! * The reference set may be restricted to non-fouling hero boards without
//!   changing the argmax, because fouling hero boards contribute only the
//!   arrangement-independent `6 * P`.
//!
//! `F` is the *stay* value, not the ordinary QQ+ entry value: a player already
//! in Fantasyland continues only under `ofc_regular.rules.check_fl_stay`.

use crate::eval::{eval3, eval5, is_foul, HandKey};
use crate::rng::SplitMix64;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Reference hero boards are packed 64 to a machine word.
pub const REF_WORD_BITS: usize = 64;

/// A fixed set of reference hero boards, used to price the line/scoop term.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceSet {
    /// Human-readable statement of how these boards were produced. This is the
    /// provenance the mission requires to travel with every artifact.
    pub provenance: String,
    pub sample_count: usize,
    pub top: Vec<HandKey>,
    pub middle: Vec<HandKey>,
    pub bottom: Vec<HandKey>,
}

impl ReferenceSet {
    pub fn words(&self) -> usize {
        self.sample_count.div_ceil(REF_WORD_BITS)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.sample_count == 0 || self.sample_count % REF_WORD_BITS != 0 {
            return Err(format!(
                "reference sample_count must be a positive multiple of {REF_WORD_BITS}, got {}",
                self.sample_count
            ));
        }
        if self.top.len() != self.sample_count
            || self.middle.len() != self.sample_count
            || self.bottom.len() != self.sample_count
        {
            return Err("reference row arrays must all have sample_count entries".to_owned());
        }
        Ok(())
    }

    /// Uniform-random-legal reference boards.
    ///
    /// Provenance, stated plainly because it matters for how far the line term
    /// can be trusted: draw 13 cards uniformly from the full 52-card deck, then
    /// pick uniformly among that hand's non-fouling 3/5/5 partitions. This is a
    /// *placeholder*. It is not a playing opponent -- it has no preference for
    /// royalties and no foul avoidance beyond legality -- so it understates how
    /// strong a real hero's rows are. It exists so objective `line_equity_v1`
    /// has a well-defined, reproducible distribution to price against until a
    /// measured hero-row distribution replaces it.
    pub fn uniform_random_legal(sample_count: usize, seed_base: u64) -> Self {
        let mut top = Vec::with_capacity(sample_count);
        let mut middle = Vec::with_capacity(sample_count);
        let mut bottom = Vec::with_capacity(sample_count);
        let mut stream = 0_u64;
        while top.len() < sample_count {
            let mut rng = SplitMix64::for_stream(seed_base, stream);
            stream += 1;
            let mut deck: Vec<u8> = (0..52).collect();
            rng.partial_shuffle(&mut deck, 13);
            let hand: [u8; 13] = deck[..13].try_into().expect("13 cards drawn");
            if let Some(rows) = uniform_legal_partition(&hand, &mut rng) {
                top.push(rows.0);
                middle.push(rows.1);
                bottom.push(rows.2);
            }
        }
        Self {
            provenance: format!(
                "uniform_random_legal_v1: 13 cards drawn uniformly from the 52-card deck, then one \
                 non-fouling 3/5/5 partition chosen uniformly at random; seed_base={seed_base}, \
                 sample_count={sample_count}. PLACEHOLDER -- this is not a playing hero."
            ),
            sample_count,
            top,
            middle,
            bottom,
        }
    }

    /// Reference set assembled from concrete hero boards, e.g. the final boards
    /// a behavior policy actually produced. `boards` are `(top, middle, bottom)`
    /// card-index rows; fouling boards are rejected (they do not change the
    /// argmax, see the module docs).
    pub fn from_boards(
        provenance: String,
        boards: &[([u8; 3], [u8; 5], [u8; 5])],
    ) -> Result<Self, String> {
        let mut top = Vec::new();
        let mut middle = Vec::new();
        let mut bottom = Vec::new();
        for (row_top, row_middle, row_bottom) in boards {
            let top_key = eval3(row_top);
            let middle_key = eval5(row_middle);
            let bottom_key = eval5(row_bottom);
            if is_foul(top_key, middle_key, bottom_key) {
                continue;
            }
            top.push(top_key);
            middle.push(middle_key);
            bottom.push(bottom_key);
        }
        let usable = top.len() - (top.len() % REF_WORD_BITS);
        if usable == 0 {
            return Err(format!(
                "need at least {REF_WORD_BITS} non-fouling reference boards, got {}",
                top.len()
            ));
        }
        top.truncate(usable);
        middle.truncate(usable);
        bottom.truncate(usable);
        Ok(Self {
            provenance,
            sample_count: usable,
            top,
            middle,
            bottom,
        })
    }
}

/// Uniformly choose a non-fouling 3/5/5 partition of 13 cards.
fn uniform_legal_partition(hand: &[u8; 13], rng: &mut SplitMix64) -> Option<(HandKey, HandKey, HandKey)> {
    let mut legal: Vec<(HandKey, HandKey, HandKey)> = Vec::new();
    let mut top_positions = [0_u8; 3];
    for a in 0..11_usize {
        for b in (a + 1)..12 {
            for c in (b + 1)..13 {
                top_positions = [a as u8, b as u8, c as u8];
                let top_key = eval3(&[hand[a], hand[b], hand[c]]);
                let rest: Vec<u8> = (0..13_u8)
                    .filter(|slot| !top_positions.contains(slot))
                    .map(|slot| hand[slot as usize])
                    .collect();
                for middle_positions in COMBINATIONS_10_CHOOSE_5.iter() {
                    let middle_cards = [
                        rest[middle_positions[0] as usize],
                        rest[middle_positions[1] as usize],
                        rest[middle_positions[2] as usize],
                        rest[middle_positions[3] as usize],
                        rest[middle_positions[4] as usize],
                    ];
                    let middle_key = eval5(&middle_cards);
                    if top_key > middle_key {
                        continue;
                    }
                    let mut bottom_cards = [0_u8; 5];
                    let mut slot = 0;
                    for (index, card) in rest.iter().enumerate() {
                        if !middle_positions.contains(&(index as u8)) {
                            bottom_cards[slot] = *card;
                            slot += 1;
                        }
                    }
                    let bottom_key = eval5(&bottom_cards);
                    if middle_key > bottom_key {
                        continue;
                    }
                    legal.push((top_key, middle_key, bottom_key));
                }
            }
        }
    }
    let _ = top_positions;
    if legal.is_empty() {
        return None;
    }
    let pick = rng.below(legal.len());
    Some(legal[pick])
}

/// All `C(10,5) = 252` index combinations, built once.
static COMBINATIONS_10_CHOOSE_5: std::sync::LazyLock<Vec<[u8; 5]>> =
    std::sync::LazyLock::new(|| {
        let mut out = Vec::with_capacity(252);
        for a in 0..6_u8 {
            for b in (a + 1)..7 {
                for c in (b + 1)..8 {
                    for d in (c + 1)..9 {
                        for e in (d + 1)..10 {
                            out.push([a, b, c, d, e]);
                        }
                    }
                }
            }
        }
        out
    });

/// Which terms the placement objective prices.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveKind {
    /// `royalty + stay * fl_ev_stay`. The exact argmax when the line term is
    /// dropped; matches what `ofc_regular.fantasyland.solve_fantasyland` and
    /// `rust/regular_fl_solver` already optimise.
    PureV1,
    /// `pure_v1 + (1 - hero_foul_prob) * E_ref[line + scoop]`.
    LineEquityV1,
}

/// Objective configuration. Every FL value carries the config it came from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectiveConfig {
    pub kind: ObjectiveKind,
    /// Value of continuing in Fantasyland, awarded on *stay*.
    pub fl_ev_stay: f64,
    /// Path the FL EV was read from. Never a literal.
    pub fl_ev_config_path: String,
    /// Card count key the value was read under (regular rules: always 14).
    pub fl_ev_cards: u8,
    /// Probability the normal hero fouls; scales the line term.
    pub hero_foul_prob: f64,
    pub hero_foul_prob_provenance: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceSet>,
}

impl ObjectiveConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.fl_ev_stay.is_finite() {
            return Err("fl_ev_stay must be finite".to_owned());
        }
        if !(0.0..=1.0).contains(&self.hero_foul_prob) {
            return Err("hero_foul_prob must lie in [0, 1]".to_owned());
        }
        match self.kind {
            ObjectiveKind::PureV1 => Ok(()),
            ObjectiveKind::LineEquityV1 => match &self.reference {
                Some(reference) => reference.validate(),
                None => Err("line_equity_v1 requires a reference set".to_owned()),
            },
        }
    }

    /// Short identity string suitable for artifact provenance fields.
    pub fn identity(&self) -> String {
        let kind = match self.kind {
            ObjectiveKind::PureV1 => "pure_v1",
            ObjectiveKind::LineEquityV1 => "line_equity_v1",
        };
        format!(
            "{kind}/fl_ev_stay={:.6}@{}[{}]/hero_foul_prob={:.6}/refs={}",
            self.fl_ev_stay,
            self.fl_ev_config_path,
            self.fl_ev_cards,
            self.hero_foul_prob,
            self.reference
                .as_ref()
                .map(|reference| reference.sample_count)
                .unwrap_or(0)
        )
    }
}

/// The FL EV table as read from a repo config, with the path it came from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlEvConfig {
    pub path: String,
    pub cards: u8,
    pub value: f64,
}

/// Read `fl_ev[cards]` out of a repo FL EV config.
///
/// This is the only place the solver learns an FL value. There is deliberately
/// no default and no fallback constant: a missing or malformed config is an
/// error, so an artifact can never be built against an unrecorded FL EV.
pub fn load_fl_ev(path: &Path, cards: u8) -> Result<FlEvConfig, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read FL EV config {}: {error}", path.display()))?;
    let payload: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("invalid FL EV config {}: {error}", path.display()))?;
    let table = payload
        .get("fl_ev")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("FL EV config {} has no fl_ev mapping", path.display()))?;
    let value = table
        .get(&cards.to_string())
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!(
                "FL EV config {} has no fl_ev entry for {cards} cards",
                path.display()
            )
        })?;
    if !value.is_finite() {
        return Err(format!("FL EV config {} holds a non-finite value", path.display()));
    }
    Ok(FlEvConfig {
        path: path.display().to_string(),
        cards,
        value,
    })
}

/// Locate `configs/fl_ev_regular_v4_selfplay.json` by walking up from `start`.
///
/// v4 supersedes v3's 9.109 as the reader default. Both superseded files stay
/// on disk: a corpus labelled under one of them validates against it, which is
/// only possible while the file it names still exists.
pub fn find_default_fl_ev_config(start: &Path) -> Option<PathBuf> {
    const RELATIVE: &str = "configs/fl_ev_regular_v4_selfplay.json";
    let mut cursor = Some(start);
    while let Some(directory) = cursor {
        let candidate = directory.join(RELATIVE);
        if candidate.is_file() {
            return Some(candidate);
        }
        cursor = directory.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_random_legal_reference_is_seed_reproducible() {
        let first = ReferenceSet::uniform_random_legal(64, 997_000_000);
        let again = ReferenceSet::uniform_random_legal(64, 997_000_000);
        assert_eq!(first.top, again.top);
        assert_eq!(first.middle, again.middle);
        assert_eq!(first.bottom, again.bottom);
        first.validate().unwrap();
        assert!(first.provenance.contains("PLACEHOLDER"));
    }

    #[test]
    fn every_uniform_reference_board_is_legal() {
        let reference = ReferenceSet::uniform_random_legal(64, 997_000_001);
        for index in 0..reference.sample_count {
            assert!(!is_foul(
                reference.top[index],
                reference.middle[index],
                reference.bottom[index]
            ));
        }
    }

    #[test]
    fn objective_validation_rejects_a_missing_reference() {
        let config = ObjectiveConfig {
            kind: ObjectiveKind::LineEquityV1,
            fl_ev_stay: 9.0,
            fl_ev_config_path: "x".to_owned(),
            fl_ev_cards: 14,
            hero_foul_prob: 0.25,
            hero_foul_prob_provenance: "test".to_owned(),
            reference: None,
        };
        assert!(config.validate().is_err());
    }
}
