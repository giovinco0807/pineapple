//! Played T1 and T2 roots, as input files for the labellers.
//!
//! `fl_solver teach-t2` deals eleven cards and assigns the first seven to rows
//! by position -- two to the top, three to the middle, two to the bottom, one
//! dead -- then labels the last three as the draw.  Every such position is
//! legal and none of them is one the chain plays, so a model trained on those
//! labels is trained on a distribution it will never meet.  This module deals
//! the same cards from the same generator and *plays* the ones the labeller
//! would have assigned: T0 places five, and at T2 T1 places two of three and
//! discards one.  The reached position goes out in exactly the record shape
//! the labeller reads, so pointing a teacher at played roots is a change of
//! input file and nothing else.
//!
//! Two emitters, one street apart:
//!
//! | mode              | dealt | played      | record                        |
//! |-------------------|-------|-------------|-------------------------------|
//! | `--emit-t1-roots` | 8     | T0          | `PlayedT1Root` (5 placed)     |
//! | `--emit-t2-roots` | 11    | T0, T1      | `PlayedRoot`   (7 placed)     |
//!
//! # Agreeing with the labeller's own dealing
//!
//! Two conventions are copied rather than reinvented, because a root file that
//! disagreed with them would relabel a different population without failing:
//!
//!   * the deal is `pool::deal(deal_seed, ordinal, width)`, the same call the
//!     labeller makes for a self-dealt root;
//!   * the id is `deal_seed.wrapping_add(ordinal)` in decimal, which is the id
//!     the labeller gives that root -- so a played root and the assigned root
//!     it replaces are keyed the same, and the two runs can be joined.
//!
//! # Cost
//!
//! One chooser call per T1 root and two per T2 root, with no move played at
//! the street being emitted: that street's encode is most of a played deal,
//! and the emitted root is the input to the decision rather than its outcome.
//! Roots are independent, so a chunk of them runs on the global rayon pool and
//! is written back in ordinal order -- the file is a function of
//! `(deal_seed, offset, count)` alone, never of how the threads landed.

use anyhow::{bail, Result};
use rayon::prelude::*;
use std::io::Write;

use super::evaluator;
use super::play_roots::{self, PlayedRoot};
use super::self_play::name_of;
use super::FlEv;

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

/// Roots per parallel chunk, and so also the progress reporting interval.
const CHUNK: usize = 1_000;

/// The `width` cards of one root, in this crate's spelling.
///
/// `pool::deal` is `fl_solver`'s, so the cards are the ones the labeller would
/// have dealt itself for this ordinal; `name_of` is the chain's, so the two
/// jokers become X1 and X2 in deal order and stay two distinct deck slots.
fn deal_names(seed: u64, ordinal: u64, width: usize) -> Vec<String> {
    let mut jokers = 0usize;
    fl_solver::pool::deal(seed, ordinal, width)
        .iter()
        .map(|card| name_of(card, &mut jokers))
        .collect()
}

/// Everything a labeller assumes about a root, checked before it is written.
///
/// A T0 that places five and a T1 that places two of three are always legal,
/// so none of this can fail on a well-formed chain -- which is the reason to
/// check it here rather than skip the root: a failure means the chain moved,
/// and a run that quietly dropped those roots would hand the labeller a
/// silently filtered population.
fn validate(
    id: &str,
    rows: [&[String]; 3],
    dead: &[String],
    draw: &[String],
    cards: &[String],
    placed_expected: usize,
    dead_expected: usize,
) -> Result<()> {
    for row in 0..3 {
        if rows[row].len() > ROW_CAPACITY[row] {
            bail!(
                "{id}: row {row} holds {} of {}",
                rows[row].len(),
                ROW_CAPACITY[row]
            );
        }
    }
    let placed: usize = rows.iter().map(|row| row.len()).sum();
    if placed != placed_expected {
        bail!("{id}: {placed} cards placed, this root holds {placed_expected}");
    }
    if dead.len() != dead_expected {
        bail!("{id}: {} dead, expected {dead_expected}", dead.len());
    }
    if draw.len() != 3 {
        bail!("{id}: {} drawn, a decision draws three", draw.len());
    }

    // The dealt names, partitioned: nothing invented and nothing lost.
    let mut emitted: Vec<&str> = rows
        .iter()
        .flat_map(|row| row.iter())
        .chain(dead.iter())
        .chain(draw.iter())
        .map(String::as_str)
        .collect();
    emitted.sort_unstable();
    let mut given: Vec<&str> = cards.iter().map(String::as_str).collect();
    given.sort_unstable();
    if emitted != given {
        bail!("{id}: the emitted root is not the deal");
    }

    // And the naturals appear once each.  Redundant with the partition above
    // while the deal itself is sound, and the assertion that would catch a
    // placement duplicating a card if it ever stopped being.
    let mut naturals: Vec<&str> = emitted
        .iter()
        .copied()
        .filter(|name| !name.starts_with('X'))
        .collect();
    let count = naturals.len();
    naturals.dedup();
    if naturals.len() != count {
        bail!("{id}: a natural card appears twice");
    }
    Ok(())
}

/// Write `count` records starting at `offset`, one JSON record a line.
///
/// Parallel within a chunk and written back in ordinal order, so the bytes do
/// not depend on the thread schedule; flushed and reported per chunk, so a
/// preempted shard keeps the roots it already played.  Returns the number
/// written, which is `count` or an error: a root that cannot be played is a
/// broken chain, not a root to skip.
fn emit_chunked<F>(
    count: usize,
    offset: u64,
    label: &str,
    out: &mut impl Write,
    line_of: F,
) -> Result<usize>
where
    F: Fn(u64) -> Result<String> + Sync,
{
    let started = std::time::Instant::now();
    let mut written = 0usize;
    while written < count {
        let take = CHUNK.min(count - written);
        let first = offset + written as u64;
        let lines: Vec<String> = (0..take as u64)
            .into_par_iter()
            .map(|step| line_of(first + step))
            .collect::<Result<Vec<String>>>()?;
        for line in &lines {
            writeln!(out, "{line}")?;
        }
        written += take;
        out.flush()?;
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "{label}: {written}/{count} roots, {:.0}/min ({:.1} s)",
            written as f64 * 60.0 / elapsed.max(1e-9),
            elapsed
        );
    }
    Ok(written)
}

/// Played T1 roots: eight cards dealt, five placed by the T0 chooser, three
/// left as the draw the T1 labeller prices.
#[allow(clippy::too_many_arguments)]
pub fn emit_t1(
    count: usize,
    offset: u64,
    seed: u64,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    t0: &evaluator::Model,
    out: &mut impl Write,
) -> Result<usize> {
    emit_chunked(count, offset, "emit-t1-roots", out, |ordinal| {
        let cards = deal_names(seed, ordinal, 8);
        let id = format!("{}", seed.wrapping_add(ordinal));
        let root = play_roots::play_t1_root(&id, &cards, fl_ev, fl_table, t0)?;
        validate(
            &root.id,
            [&root.board.top, &root.board.middle, &root.board.bottom],
            &root.dead,
            &root.draw,
            &cards,
            5,
            0,
        )?;
        Ok(serde_json::to_string(&root)?)
    })
}

/// Played T2 roots: eleven cards dealt, seven placed by the T0 and T1
/// choosers, one discarded, three left as the draw.
#[allow(clippy::too_many_arguments)]
pub fn emit_t2(
    count: usize,
    offset: u64,
    seed: u64,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    out: &mut impl Write,
) -> Result<usize> {
    emit_chunked(count, offset, "emit-t2-roots", out, |ordinal| {
        let cards = deal_names(seed, ordinal, 11);
        let id = format!("{}", seed.wrapping_add(ordinal));
        let root: PlayedRoot = play_roots::play_t2_root(&id, &cards, fl_ev, fl_table, t0, t1)?;
        validate(
            &root.id,
            [&root.rows[0], &root.rows[1], &root.rows[2]],
            &root.dead,
            &root.draw,
            &cards,
            7,
            1,
        )?;
        Ok(serde_json::to_string(&root)?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playout::tests::linear_model;

    /// Two width-60 choosers, the cheapest widths `encode_for` dispatches on
    /// (actor block plus FL context, nothing sampled).  Synthetic rather than
    /// the shipped weights so the test runs anywhere: a test gated behind
    /// `D:/ofc_data` is a test that stops being run.
    fn choosers() -> (evaluator::Model, evaluator::Model) {
        (linear_model(60, 0x0BADC0DE), linear_model(60, 0x5EED_1234))
    }

    fn fl_ev() -> FlEv {
        let mut by_card_count = std::collections::BTreeMap::new();
        for (slot, count) in [14u8, 15, 16, 17].into_iter().enumerate() {
            by_card_count.insert(count, [0.0, 10.7, 29.9, 63.5][slot]);
        }
        FlEv {
            by_card_count,
            config_sha256: String::new(),
        }
    }

    /// Which street an assertion is about, so the two emitters share one body.
    #[derive(Clone, Copy, PartialEq)]
    enum Street {
        T1,
        T2,
    }

    fn emitted(street: Street, count: usize, offset: u64, seed: u64) -> String {
        let (t0, t1) = choosers();
        let fl_ev = fl_ev();
        let fl_table: evaluator::FlTable = [0.0, 10.7, 29.9, 63.5];
        let mut out: Vec<u8> = Vec::new();
        let written = match street {
            Street::T1 => emit_t1(count, offset, seed, &fl_ev, &fl_table, &t0, &mut out),
            Street::T2 => emit_t2(count, offset, seed, &fl_ev, &fl_table, &t0, &t1, &mut out),
        }
        .expect("emit");
        assert_eq!(written, count, "a root was skipped rather than played");
        String::from_utf8(out).expect("utf8")
    }

    /// **A root file is a function of (seed, offset, count) alone.**
    ///
    /// Same arguments, same bytes -- across a rayon pool that is free to
    /// schedule the chunk differently on the second run.  This is what lets a
    /// shard be replayed, and what makes two boxes agree about which roots a
    /// teacher was pointed at.
    #[test]
    fn the_same_seed_and_offset_emit_the_same_bytes() {
        for street in [Street::T1, Street::T2] {
            let first = emitted(street, 6, 0, 0xFCA1_1234);
            let again = emitted(street, 6, 0, 0xFCA1_1234);
            assert_eq!(first, again, "the same request emitted a different file");
            assert_eq!(first.lines().count(), 6);
        }
    }

    /// **A shard is a slice of the whole run.**
    ///
    /// The ordinal has to reach the deal and nothing else: roots [4, 8) asked
    /// for on their own must be byte-identical to lines 5..8 of a run that
    /// started at zero, or a fleet's shards do not reassemble into the file a
    /// single box would have written.
    #[test]
    fn a_shard_is_the_same_bytes_as_that_slice_of_the_whole() {
        for street in [Street::T1, Street::T2] {
            let whole = emitted(street, 8, 0, 0xFCA1_1234);
            let shard = emitted(street, 4, 4, 0xFCA1_1234);
            let slice: String = whole
                .lines()
                .skip(4)
                .map(|line| format!("{line}\n"))
                .collect();
            assert_eq!(shard, slice, "a shard is not the slice it claims to be");
        }
    }

    /// **Different offsets are different roots, and different seeds too.**
    ///
    /// The failure this rules out is an ordinal that never reaches the deal:
    /// a file of one position repeated is a teacher pointed at one root.
    #[test]
    fn a_different_offset_reaches_different_boards() {
        for street in [Street::T1, Street::T2] {
            let here = emitted(street, 4, 0, 0xFCA1_1234);
            assert_ne!(
                here,
                emitted(street, 4, 100, 0xFCA1_1234),
                "two offsets emitted the same roots"
            );
            assert_ne!(
                here,
                emitted(street, 4, 0, 0xFCA1_9999),
                "two seeds emitted the same roots"
            );
            let mut lines: Vec<&str> = here.lines().collect();
            let count = lines.len();
            lines.sort_unstable();
            lines.dedup();
            assert_eq!(lines.len(), count, "a file repeated a root");
        }
    }

    /// **Every emitted record is the position its labeller expects.**
    ///
    /// Re-checked from the parsed JSON rather than trusting `validate`, so a
    /// record that serialised wrongly fails even though the struct it came
    /// from was sound.  The T1 shape is `PlayedT1Root`: a `board` object, an
    /// empty `dead`, and the `opp_count` the labeller reads.
    #[test]
    fn every_t1_record_is_a_legal_played_t1_root() {
        let text = emitted(Street::T1, 8, 42, 0xFCA1_1234);
        for (index, line) in text.lines().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line).expect("record parses");
            assert_eq!(
                value["id"].as_str().expect("id"),
                format!("{}", 0xFCA1_1234u64.wrapping_add(42 + index as u64)),
                "the id is not the labeller's id for this ordinal"
            );
            assert_eq!(value["opp_count"].as_u64(), Some(14));
            assert_eq!(value["dead"].as_array().expect("dead").len(), 0);
            assert_eq!(value["draw"].as_array().expect("draw").len(), 3);

            let board = &value["board"];
            let mut names: Vec<String> = Vec::new();
            let mut placed = 0usize;
            for (row, key) in ["top", "middle", "bottom"].iter().enumerate() {
                let held = board[*key].as_array().expect("row");
                assert!(
                    held.len() <= ROW_CAPACITY[row],
                    "{key} holds {} of {}",
                    held.len(),
                    ROW_CAPACITY[row]
                );
                placed += held.len();
                names.extend(held.iter().map(|n| n.as_str().expect("card").to_string()));
            }
            assert_eq!(placed, 5, "line {}: {placed} placed", index + 1);

            names.extend(
                value["draw"]
                    .as_array()
                    .expect("draw")
                    .iter()
                    .map(|n| n.as_str().expect("card").to_string()),
            );
            names.sort();
            let mut given = deal_names(0xFCA1_1234, 42 + index as u64, 8);
            given.sort();
            assert_eq!(names, given, "line {}: not the deal", index + 1);
        }
    }

    /// The same for the T2 shape: flat `rows`, one dead, seven placed.
    #[test]
    fn every_t2_record_is_a_legal_played_t2_root() {
        let text = emitted(Street::T2, 8, 42, 0xFCA1_1234);
        for (index, line) in text.lines().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line).expect("record parses");
            assert_eq!(
                value["id"].as_str().expect("id"),
                format!("{}", 0xFCA1_1234u64.wrapping_add(42 + index as u64))
            );
            let rows = value["rows"].as_array().expect("rows");
            assert_eq!(rows.len(), 3);
            let mut placed = 0usize;
            for (row, capacity) in ROW_CAPACITY.iter().enumerate() {
                let held = rows[row].as_array().expect("row").len();
                assert!(held <= *capacity, "row {row} holds {held} of {capacity}");
                placed += held;
            }
            assert_eq!(placed, 7, "line {}: {placed} placed", index + 1);
            assert_eq!(value["dead"].as_array().expect("dead").len(), 1);
            assert_eq!(value["draw"].as_array().expect("draw").len(), 3);

            let mut names: Vec<String> = rows
                .iter()
                .flat_map(|row| row.as_array().expect("row").iter())
                .chain(value["dead"].as_array().expect("dead").iter())
                .chain(value["draw"].as_array().expect("draw").iter())
                .map(|name| name.as_str().expect("card name").to_string())
                .collect();
            names.sort();
            let mut given = deal_names(0xFCA1_1234, 42 + index as u64, 11);
            given.sort();
            assert_eq!(names, given, "line {}: not the deal", index + 1);
        }
    }
}
