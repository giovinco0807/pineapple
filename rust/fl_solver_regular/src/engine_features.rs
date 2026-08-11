//! The bridge to the engine's vs-Fantasyland fast features.
//!
//! The T1 teacher continues its candidates with the distilled T2/T3 rankers,
//! and those rankers were trained on the engine's 168-wide row:
//! `structural | hero outlook` real in `[0, HERO_BLOCK_END)`, opponent tail
//! zeroed. A continuation that scored a different encoding would be a different
//! policy than the one whose regret was measured, so this calls the same engine
//! entry points the offline extractor did rather than reimplementing them.
//!
//! One context per decision node, reused across that node's whole fan: the
//! observation and the unknown set are properties of the node, and
//! `FastOutlookCache` is where the outlook work is amortized.

use ofc_hu_m3_engine::cards::{Card, ALL_CARDS};
use ofc_hu_m3_engine::fast_features::{fast_encode_hidden_opponent, FastOutlookCache};
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::state::Board as EngineBoard;
use ofc_hu_m3_engine::t3_features::unknown_cards;
use ofc_hu_m3_engine::t3first_features::FEATURE_SIZE;

use crate::behavior::PartialBoard;

/// Width of one encoded candidate. Re-exported so callers do not have to reach
/// into the engine for it.
pub const ROW_WIDTH: usize = FEATURE_SIZE;

/// Card identifiers in this crate are indices into the engine's card order.
/// That is the same assumption the offline extractor made, and the labels it
/// produced trained the models being loaded here.
fn cards(ids: &[u8]) -> Vec<Card> {
    ids.iter().map(|id| ALL_CARDS[*id as usize]).collect()
}

fn engine_board(board: &PartialBoard) -> Result<EngineBoard, String> {
    EngineBoard::new(
        cards(&board.rows[0][..board.lengths[0] as usize]),
        cards(&board.rows[1][..board.lengths[1] as usize]),
        cards(&board.rows[2][..board.lengths[2] as usize]),
    )
    .map_err(|error| format!("engine board: {error}"))
}

/// Everything a single decision node needs to encode its fan.
pub struct NodeFeatures {
    observation: ActorObservation,
    unknown: Vec<Card>,
    cache: FastOutlookCache,
}

impl NodeFeatures {
    /// `street` is the hero's decision street; `thrown` is the hero's own prior
    /// discards, which the engine needs to compute the unknown set correctly.
    ///
    /// `dealt` is a slice rather than a three-card array because T0 is dealt
    /// five. The encoding itself is street-agnostic -- all four vs-Fantasyland
    /// streets produce the same 168 columns -- so the only thing that had to
    /// widen was the way in.
    pub fn new(
        board: &PartialBoard,
        dealt: &[u8],
        thrown: &[u8],
        street: Street,
    ) -> Result<Self, String> {
        let observation = ActorObservation::new_vs_fantasyland(
            engine_board(board)?,
            cards(dealt),
            cards(thrown),
            Seat::First,
            street,
            ActOrder::First,
            ScoringContext::default(),
        )
        .map_err(|error| format!("observation: {error}"))?;
        let unknown = unknown_cards(&observation);
        Ok(Self {
            observation,
            unknown,
            cache: FastOutlookCache::new(),
        })
    }

    /// Encode one candidate -- the hero board AFTER the action.
    ///
    /// The engine hands back a fixed-width array, which is returned as-is: the
    /// width is a compile-time property of the encoding, so a runtime length
    /// check would be checking something the type already guarantees.
    pub fn encode(&mut self, candidate: &PartialBoard) -> Result<[f32; ROW_WIDTH], String> {
        let placed = engine_board(candidate)?;
        fast_encode_hidden_opponent(&self.observation, &placed, &self.unknown, &mut self.cache)
            .map_err(|error| format!("encode: {error}"))
    }
}

/// The engine generation linked into this binary.
///
/// The plan pins the solver binary by digest, which already fixes the engine
/// bytes inside it. This is the human-readable half: a package diff can name
/// which engine generation produced a label without disassembling anything.
pub fn engine_features_rev() -> &'static str {
    option_env!("OFC_ENGINE_FEATURES_REV").unwrap_or("unpinned")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(top: &[u8], middle: &[u8], bottom: &[u8]) -> PartialBoard {
        let mut out = PartialBoard::default();
        for (index, row) in [top, middle, bottom].iter().enumerate() {
            out.lengths[index] = row.len() as u8;
            out.rows[index][..row.len()].copy_from_slice(row);
        }
        out
    }

    #[test]
    fn encodes_a_t2_node_with_a_zeroed_opponent_tail() {
        // 7 placed, 1 thrown: a T2-vs-FL decision.
        let hero = board(&[0], &[4, 8], &[12, 16, 20, 24]);
        let mut node = NodeFeatures::new(&hero, &[28, 32, 36], &[40], Street::T2)
            .expect("T2 observation is accepted");
        let candidate = board(&[0, 28], &[4, 8, 32], &[12, 16, 20, 24]);
        let row = node.encode(&candidate).expect("encodes");
        assert_eq!(row.len(), ROW_WIDTH);
        let tail_start = ofc_hu_m3_engine::fast_features::HERO_BLOCK_END;
        assert!(
            row[tail_start..].iter().all(|value| *value == 0.0),
            "the opponent tail must be exactly zero"
        );
        assert!(
            row[..tail_start].iter().any(|value| *value != 0.0),
            "the hero block must carry real numbers"
        );
    }

    #[test]
    fn encodes_a_t3_node() {
        // 9 placed, 2 thrown.
        let hero = board(&[0, 1], &[4, 8, 12, 16], &[20, 24, 28]);
        let mut node = NodeFeatures::new(&hero, &[32, 36, 40], &[44, 48], Street::T3)
            .expect("T3 observation is accepted");
        let candidate = board(&[0, 1, 32], &[4, 8, 12, 16, 36], &[20, 24, 28]);
        assert_eq!(node.encode(&candidate).expect("encodes").len(), ROW_WIDTH);
    }

    #[test]
    fn the_same_node_encodes_identically_twice() {
        // The outlook cache is reused across a fan; reuse must not drift.
        let hero = board(&[0], &[4, 8], &[12, 16, 20, 24]);
        let candidate = board(&[0, 28], &[4, 8, 32], &[12, 16, 20, 24]);
        let mut node = NodeFeatures::new(&hero, &[28, 32, 36], &[40], Street::T2).expect("node");
        let first = node.encode(&candidate).expect("encodes");
        let second = node.encode(&candidate).expect("encodes");
        assert_eq!(first, second);
    }
}
