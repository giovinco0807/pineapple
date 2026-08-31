//! The 207-dim HU pair encoding, in Rust.
//!
//! The teachers build this vector in Python (`ai/tutor/encode_hu_teacher.py`),
//! which is fine for labelling but hopeless for playing hands: a match asks
//! for it a hundred and seventy times a hand.  This is the same vector, laid
//! out in the same order, so a model trained through the Python path can be
//! served here:
//!
//! ```text
//!   own  : actor 48 | rowwise 41 | joint 8 | context 7 | alloc 6   = 110
//!   opp  : actor 48 | rowwise 41 | joint 8                          =  97
//! ```
//!
//! Both boards are judged over **one** pool -- everything the deciding seat
//! has not seen -- which is what puts the two outlooks on a shared deck and
//! makes contested-ness implicit.  A parity harness pins this against the
//! Python encoder; without that the two drift and the model is silently
//! served a different vector than it learnt.

use anyhow::Result;

use super::evaluator;
use super::joint_outlook;
use super::playout::RowwiseMemo;
use super::{Card, CoreBoard, FlEv};

pub const OWN_SIZE: usize = 110;
pub const OPP_SIZE: usize = 97;
pub const HU_FEATURE_SIZE: usize = OWN_SIZE + OPP_SIZE;

/// Joint-block settings, mirroring what the Python encoder passes.
///
/// The sample count is the whole cost of serving: eight model decisions a
/// hand, five encodings each, four hundred sampled completions apiece is
/// sixteen thousand completions per hand.  It is overridable so the trade --
/// cheaper features against a distribution the model did not train on -- can
/// be measured rather than assumed.  Training must keep 400.
pub const JOINT_SAMPLES: usize = 400;
const JOINT_ARRANGEMENTS: usize = 32;

/// Serve-time sample count; 0 means the trained 400.  Carried per arm rather
/// than globally because the only way to price the trade is to seat two
/// sample counts against each other in one match.
pub fn joint_samples(requested: usize) -> usize {
    if requested == 0 { JOINT_SAMPLES } else { requested }
}

/// One board's blocks, appended to `out`.
#[allow(clippy::too_many_arguments)]
fn board_blocks(
    rows: &CoreBoard,
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    seed: &str,
    with_context_and_alloc: bool,
    samples: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    evaluator::actor_block(&rows.rows, out);
    let _categories =
        evaluator::opponent_rowwise_block_shared(&rows.rows, pool, fl_table, memo, pool_key, out);
    // Always sampled, never the exact shortcut two open slots would allow:
    // `encode_hu_teacher.py` passes four hundred samples at every street, so
    // exact blocks here would serve the model a feature it never trained on.
    // (Lap two should make both sides exact where they can be; that changes
    // the corpus, so it is not a serving-side decision.)  The `joint/` prefix
    // reproduces what the block binary prepends.
    //
    // One board is exempt and it is not exempted here: a board nobody has
    // played to -- the first actor's opponent at T0 -- returns eight zeros.
    // That rule is inside `joint_outlook::sampled_joint_block_shared` so this
    // serving path and the teachers' `--joint-outlook` path cannot state it
    // differently; see `joint_outlook::unplayed_joint_block`.
    let block = joint_outlook::sampled_joint_block(
        rows,
        pool,
        joint_samples(samples),
        JOINT_ARRANGEMENTS,
        &format!("joint/{seed}"),
        fl_ev,
        joint_outlook::CompletionSampler::Sha256,
    )?;
    for value in block {
        out.push(value as f32);
    }
    if with_context_and_alloc {
        evaluator::fl14_context_block(pool, out);
        evaluator::allocation_rank_block(&rows.rows, out);
    }
    Ok(())
}

/// The pair vector for (deciding seat's board, opponent's visible board).
///
/// `seed` must not vary with the action: two candidates compared on
/// different sampled completions differ by the sample before they differ by
/// the move.  Callers pass the node's seed, never the candidate's.
/// The opponent's 97 dimensions, which do not vary with the candidate.
///
/// The pool a decision is judged against is fixed before the candidates are
/// enumerated -- it comes from the pre-decision boards, the discards so far
/// and the draw -- and the opponent's board does not move on this turn.  So
/// this half of the vector is the same for every candidate, and computing it
/// inside the loop repeats the most expensive block in the encoding: at the
/// first seat's opening the opponent's board is *empty*, thirteen open slots
/// of sampled completions, recomputed once per candidate for all 232 of them.
#[allow(clippy::too_many_arguments)]
pub fn opponent_tail(
    opp: &CoreBoard,
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    seed: &str,
    samples: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    board_blocks(opp, pool, fl_ev, fl_table, memo, pool_key, seed, false, samples, out)
}

/// `encode_pair` with the opponent's half already computed.
#[allow(clippy::too_many_arguments)]
pub fn encode_pair_with_tail(
    own: &CoreBoard,
    tail: &[f32],
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    seed: &str,
    samples: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    board_blocks(own, pool, fl_ev, fl_table, memo, pool_key, seed, true, samples, out)?;
    out.extend_from_slice(tail);
    debug_assert_eq!(out.len(), HU_FEATURE_SIZE);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_pair(
    own: &CoreBoard,
    opp: &CoreBoard,
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    seed: &str,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    board_blocks(
        own, pool, fl_ev, fl_table, memo, pool_key, seed, true, JOINT_SAMPLES, out,
    )?;
    board_blocks(
        opp, pool, fl_ev, fl_table, memo, pool_key, seed, false, JOINT_SAMPLES, out,
    )?;
    debug_assert_eq!(out.len(), HU_FEATURE_SIZE);
    Ok(())
}

// ---------------------------------------------------------------------------
// Hybrid encoding for distilled evaluators: 432 card one-hots + the cheap 191.
//
// The card blocks replicate `encode_hu_cards.py` exactly, including its card
// order -- hearts, diamonds, clubs, spades, ranks 2..A, jokers at 52/53 --
// which differs from this crate's own s/h/d/c `all_cards()`.  The nets were
// trained on the Python layout; an encoder that is merely equivalent rather
// than identical scores every candidate with scrambled inputs and no error.
// Within a block the k-th joker present fills the k-th joker slot; jokers are
// counted, never name-matched, because the seats' names collide.

pub const HYBRID_CARD_BLOCKS: usize = 8;
pub const HYBRID_FEATURE_SIZE: usize = 54 * HYBRID_CARD_BLOCKS + (HU_FEATURE_SIZE - 16);

fn py_card_slot(name: &str) -> usize {
    let bytes = name.as_bytes();
    let rank = match bytes[0] {
        b'2'..=b'9' => (bytes[0] - b'2') as usize,
        b'T' => 8,
        b'J' => 9,
        b'Q' => 10,
        b'K' => 11,
        _ => 12,
    };
    let suit = match bytes[1] {
        b'h' => 0,
        b'd' => 1,
        b'c' => 2,
        _ => 3,
    };
    suit * 13 + rank
}

fn card_block<'a>(cards: impl Iterator<Item = &'a String>, block: usize, out: &mut [f32]) {
    let base = block * 54;
    let mut jokers = 0usize;
    for name in cards {
        let slot = if name.starts_with('X') {
            jokers += 1;
            51 + jokers
        } else {
            py_card_slot(name)
        };
        out[base + slot] = 1.0;
    }
}

/// actor + rowwise (+ context + alloc for the deciding side), no joint.
#[allow(clippy::too_many_arguments)]
fn board_blocks_cheap(
    rows: &CoreBoard,
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    with_context_and_alloc: bool,
    out: &mut Vec<f32>,
) {
    let _ = fl_ev;
    evaluator::actor_block(&rows.rows, out);
    let _categories =
        evaluator::opponent_rowwise_block_shared(&rows.rows, pool, fl_table, memo, pool_key, out);
    if with_context_and_alloc {
        evaluator::fl14_context_block(pool, out);
        evaluator::allocation_rank_block(&rows.rows, out);
    }
}

/// The 623-dim vector a distilled evaluator reads: no sampling anywhere.
#[allow(clippy::too_many_arguments)]
pub fn encode_hybrid(
    own_rows: &[Vec<String>; 3],
    opp_rows: &[Vec<String>; 3],
    dead: &[String],
    pool_names: &[String],
    own: &CoreBoard,
    opp: &CoreBoard,
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(54 * HYBRID_CARD_BLOCKS, 0.0);
    for (block, rows) in [(0usize, own_rows), (3, opp_rows)] {
        for (offset, row) in rows.iter().enumerate() {
            card_block(row.iter(), block + offset, out);
        }
    }
    card_block(dead.iter(), 6, out);
    card_block(pool_names.iter(), 7, out);
    board_blocks_cheap(own, pool, fl_ev, fl_table, memo, pool_key, true, out);
    board_blocks_cheap(opp, pool, fl_ev, fl_table, memo, pool_key, false, out);
    debug_assert_eq!(out.len(), HYBRID_FEATURE_SIZE);
}

/// The 207-dim layout with the sixteen joint dims left at zero -- what the
/// no-joint shortlist rankers were trained on, at none of the joint's cost.
#[allow(clippy::too_many_arguments)]
pub fn encode_cheap207(
    own: &CoreBoard,
    opp: &CoreBoard,
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    out: &mut Vec<f32>,
) {
    out.clear();
    board_blocks_cheap(own, pool, fl_ev, fl_table, memo, pool_key, false, out);
    out.resize(out.len() + 8, 0.0);
    evaluator::fl14_context_block(pool, out);
    evaluator::allocation_rank_block(&own.rows, out);
    board_blocks_cheap(opp, pool, fl_ev, fl_table, memo, pool_key, false, out);
    out.resize(out.len() + 8, 0.0);
    debug_assert_eq!(out.len(), HU_FEATURE_SIZE);
}
