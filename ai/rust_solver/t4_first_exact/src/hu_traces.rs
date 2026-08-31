//! Two-seat synchronized traces: the root corpus of the HU phase.
//!
//! Both seats are played by the generation-3 own-hand chain (owner's
//! instruction), which never looks across the table -- so the seats can be
//! played independently and the *visibility* reassembled afterwards under the
//! confirmed rules:
//!
//! * within a street, the second seat (BTN) sees the first seat's (BB's)
//!   placement of that same street before placing;
//! * discards are face down: a seat's dead cards never appear in what the
//!   opponent can see, and stay inside the opponent's unseen set.
//!
//! So a BB decision at street k sees BTN's board after street k-1, and a BTN
//! decision at street k sees BB's board after street k.  Every emitted row is
//! one decision: the seat's own board, dead and draw -- exactly what its
//! chooser saw -- plus the opponent board it was entitled to see and did not
//! use.  The HU teachers will use it.

use anyhow::Result;
use serde::Serialize;

use super::evaluator;
use super::self_play::{play_normal_traced, settle, NormalTrace};
use super::FlEv;

#[derive(Serialize)]
struct DecisionRow<'a> {
    hand: u64,
    /// "bb" acts first within a street, "btn" second.
    seat: &'static str,
    street: usize,
    board: &'a [Vec<String>; 3],
    dead: &'a [String],
    draw: &'a [String],
    opp_board: &'a [Vec<String>; 3],
    board_after: &'a [Vec<String>; 3],
}

#[derive(Serialize)]
struct HandRow<'a> {
    hand: u64,
    summary: bool,
    bb_final: &'a [Vec<String>; 3],
    btn_final: &'a [Vec<String>; 3],
    /// Real settlement from BB's side, entry credit not included.
    settle_bb: i32,
    bb_foul: bool,
    btn_foul: bool,
    bb_entry: u8,
    btn_entry: u8,
}

fn emit(hand: u64, bb: &NormalTrace, btn: &NormalTrace, out: &mut Vec<String>) -> Result<()> {
    let empty: [Vec<String>; 3] = Default::default();
    for street in 0..5 {
        let bb_state = &bb.streets[street];
        // BB places first: it sees BTN as of the previous street.
        let bb_sees = if street == 0 {
            &empty
        } else {
            &btn.streets[street - 1].board_after
        };
        out.push(serde_json::to_string(&DecisionRow {
            hand,
            seat: "bb",
            street,
            board: &bb_state.board_before,
            dead: &bb_state.dead_before,
            draw: &bb_state.draw,
            opp_board: bb_sees,
            board_after: &bb_state.board_after,
        })?);
        // BTN places second: it sees BB's placement of this very street.
        let btn_state = &btn.streets[street];
        out.push(serde_json::to_string(&DecisionRow {
            hand,
            seat: "btn",
            street,
            board: &btn_state.board_before,
            dead: &btn_state.dead_before,
            draw: &btn_state.draw,
            opp_board: &bb_state.board_after,
            board_after: &btn_state.board_after,
        })?);
    }
    out.push(serde_json::to_string(&HandRow {
        hand,
        summary: true,
        bb_final: &bb.streets[4].board_after,
        btn_final: &btn.streets[4].board_after,
        settle_bb: settle(&bb.finished, &btn.finished),
        bb_foul: bb.finished.busted,
        btn_foul: btn.finished.busted,
        bb_entry: bb.finished.entry_width,
        btn_entry: btn.finished.entry_width,
    })?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    hands: usize,
    seed: u64,
    workers: usize,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    t2: &evaluator::Model,
    output: &std::path::Path,
) -> Result<()> {
    use rayon::prelude::*;
    use std::io::Write;

    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    let started = std::time::Instant::now();
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    let chunk = 256usize.max(workers);
    let mut written = 0usize;
    for start in (0..hands).step_by(chunk) {
        let end = (start + chunk).min(hands);
        let lines: Result<Vec<Vec<String>>> = (start..end)
            .into_par_iter()
            .map(|index| {
                // One 54-card shuffle a hand; BB takes the first seventeen.
                let dealt = fl_solver::pool::deal(seed, index as u64, 34);
                let mut rows: Vec<String> = Vec::with_capacity(11);
                let bb = play_normal_traced(
                    &format!("hu/{seed}/{index}/bb"),
                    &dealt[..17],
                    fl_ev,
                    fl_table,
                    &table,
                    t0,
                    t1,
                    t2,
                )?;
                let btn = play_normal_traced(
                    &format!("hu/{seed}/{index}/btn"),
                    &dealt[17..34],
                    fl_ev,
                    fl_table,
                    &table,
                    t0,
                    t1,
                    t2,
                )?;
                emit(index as u64, &bb, &btn, &mut rows)?;
                Ok(rows)
            })
            .collect();
        for rows in lines? {
            for line in rows {
                writeln!(out, "{line}")?;
                written += 1;
            }
        }
        out.flush()?;
    }
    eprintln!(
        "hu-traces: {hands} hands, {written} rows, {:.1} s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
