//! Fantasyland-only simulator: measure `fl_ev(w)` at a width you choose.
//!
//! Self-play answers "what is a Fantasyland entry worth?" only where self-play
//! happens to go, and it does not go to the widths that matter: sixty thousand
//! gen-2 hands bought 389 chains at width 17 (SE 3.6 against a 30-point
//! question) and put 93% of their Fantasyland observations at widths 15 and 16.
//! Here the hero is **pinned** in Fantasyland at one width, so every trial is
//! an observation of that width, and the chain of stays is followed to its end.
//!
//! What is measured is the owner's frozen definition (2026-08-15), unchanged:
//! **episodic** -- one entry, the whole chain of same-width re-entries it
//! becomes, summed -- in **real settled points**, with **no baseline
//! subtracted**.  `ai/tutor/analyze_selfplay_flev.py` is the canonical
//! statement of that methodology; this module reproduces its game, not its
//! arithmetic.
//!
//! The opponent is not held at normal.  It runs the same two-state machine as
//! `self_play.rs` (enter on qualifying, leave on not staying), because the
//! reference numbers this is calibrated against are chain totals that include
//! Fantasyland-versus-Fantasyland hands -- 41% of the hero's Fantasyland hands
//! in the reference data were played against an opponent who was also in
//! Fantasyland, and those hands are worth about four points *less* per hand at
//! width 15 than the versus-normal ones are worth more.  Pinning the opponent
//! to normal would delete that 41% of the environment and the chain totals
//! would float up by points.  `solo` (the opponent normal for the chain's whole
//! life) is recorded per hand and separated by the analyzer instead, which is
//! the direction that works: a conditional reading can be extracted from the
//! full process, but the full process cannot be synthesized from a pinned one.
//!
//! What this world does **not** have is stacks, sessions, the 200-point buy-in
//! or the gap-40 reset.  Those are not omissions to be patched: the gap reset
//! fires only when *both* players are normal, which never happens while the
//! hero is pinned, so a stack process here would not be the game's stack
//! process but an invention.  Chains end one way only -- the hero does not
//! stay.  Settlements are raw.  In the reference data raw and paid differ by
//! less than the standard error at widths 14-16; at width 17 paid is 2.45
//! points lower, which is the 200-point stack truncating long chains, and any
//! report of `fl_ev(17)` from this simulator has to carry that note.
//!
//! One invocation measures **one pass under one table**.  `fl_ev` is
//! self-referential -- the stay decision and the placement both read the table
//! being measured -- so a fixed point has to be iterated, and the iteration
//! lives in `ai/tutor/fl_sim_driver.py`.  This module knows nothing about where
//! its table came from; it stamps the table's SHA-256 into the output so the
//! provenance is auditable after the fact.

use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::evaluator;
use super::self_play::{play_fl, play_normal_traced, settle, Finished};
use super::FlEv;

/// One block is a deterministic function of `(seed, block)`; `1_000_003` is the
/// project's stride for splitting a seed into independent streams.
const BLOCK_STRIDE: u64 = 1_000_003;

/// A block that reaches this many deals has stopped being a Fantasyland chain
/// and become a bug.  Stay probability is under 0.8, so a chain of 10^5 hands
/// has probability around 10^-9000; arriving here means the stay flag or the
/// loop's exit condition is wrong, and failing loudly beats writing a file that
/// looks like data.
const MAX_HANDS_PER_BLOCK: u32 = 100_000;

/// The opponent's state.
///
/// This is `self_play::State` (self_play.rs:402-406), redeclared rather than
/// shared: it is six lines, and making it visible would mean editing a file
/// whose byte-for-byte output is a regression test for every other mode.
#[derive(Clone, Copy, PartialEq)]
enum OppState {
    Normal,
    Fl(u8),
}

impl OppState {
    fn label(&self) -> String {
        match self {
            OppState::Normal => "normal".to_string(),
            OppState::Fl(width) => format!("fl{width}"),
        }
    }

    /// Cards this state is dealt.
    fn need(&self) -> usize {
        match self {
            OppState::Normal => 17,
            OppState::Fl(width) => *width as usize,
        }
    }
}

pub struct Config {
    /// The width the hero enters and re-enters at, 14..=17.
    pub width: u8,
    /// Independent blocks; block `b` is a deterministic function of `(seed, b)`.
    pub blocks: u64,
    /// Complete chains per block.  Chain 0 starts the opponent at normal and is
    /// flagged `burn_in`; the headline reading uses chains `1..K`.
    pub chains_per_block: u32,
    pub seed: u64,
    /// The three chooser paths as given on the command line, for the metadata
    /// line only.  Provenance: a run whose models cannot be identified after
    /// the fact cannot be compared with another run.
    pub model_paths: [String; 3],
}

/// One Fantasyland hand of a chain, from the hero's side.
#[derive(Serialize)]
struct HandRow {
    /// Deal index within the block -- what `deal(block_seed, h, ..)` was given,
    /// so any hand here can be re-dealt on its own.
    h: u32,
    /// The opponent's state at the **start** of this hand.
    opp: String,
    settle: i32,
    hero_roy: i32,
    hero_stay: bool,
    opp_roy: i32,
    opp_foul: bool,
    /// The opponent's qualifying width from a normal hand; 0 when it busted and
    /// 0 for every Fantasyland hand (a Fantasyland hand does not requalify --
    /// it continues through `stays`, the frozen rule).
    opp_entry: u8,
    /// The opponent's stay from a Fantasyland hand; false on normal hands.
    opp_stay: bool,
}

/// One entry and every same-width re-entry it became.  `sum_settle` is the
/// sample of `fl_ev(width)`.
#[derive(Serialize)]
struct ChainRow {
    block: u64,
    chain: u32,
    burn_in: bool,
    sum_settle: i64,
    n_hands: u32,
    solo: bool,
    opp_at_start: String,
    hands: Vec<HandRow>,
}

/// A block's serialized chains, kept with its number so the writer can restore
/// the order rayon completed them out of.
struct BlockOut {
    block: u64,
    lines: Vec<String>,
    hands: u64,
    headline_chains: u64,
    headline_sum: i64,
}

pub fn run(
    config: &Config,
    fl_ev: &FlEv,
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    t2: &evaluator::Model,
    output: &std::path::Path,
) -> Result<()> {
    if !(14..=17).contains(&config.width) {
        bail!(
            "--fl-sim-width is {} but a Fantasyland entry is 14..=17 cards",
            config.width
        );
    }
    if config.chains_per_block < 2 {
        bail!(
            "--fl-sim-chains-per-block is {} but must be >= 2: chain 0 of every block \
             is the burn-in that mixes the opponent's state and is excluded from the \
             headline, so K=1 would produce no samples at all",
            config.chains_per_block
        );
    }
    if config.blocks < 1 {
        bail!("--fl-sim-blocks must be >= 1");
    }

    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    for (slot, value) in table.iter().enumerate() {
        if *value < 0.0 {
            bail!(
                "fl_ev[{}] = {value} is negative.  `build_frontier` asserts fl_ev >= 0 \
                 (frontier.rs:634): below zero the per-row joker maximum stops being \
                 the best substitution and the frontier would be missing arrangements \
                 rather than merely mispriced.  Failing here beats panicking an hour in.",
                slot + 14
            );
        }
    }
    let fl_table: evaluator::FlTable = [
        table[0] as f32,
        table[1] as f32,
        table[2] as f32,
        table[3] as f32,
    ];

    eprintln!(
        "fl-sim: width {}, {} blocks x {} chains, seed {}, table [{:.2}, {:.2}, {:.2}, {:.2}], \
         choosers {} / {} / {} dims",
        config.width,
        config.blocks,
        config.chains_per_block,
        config.seed,
        table[0],
        table[1],
        table[2],
        table[3],
        t0.input_dim,
        t1.input_dim,
        t2.input_dim
    );

    let started = std::time::Instant::now();
    let finished_blocks = AtomicUsize::new(0);

    // Blocks are independent -- each deals from its own (seed, block) stream --
    // and each is played strictly sequentially inside, because the opponent's
    // state carries from one chain to the next.  The per-decision rayon inside
    // `play_normal_traced` still applies; the scheduler interleaves both levels.
    let collected: Result<Vec<BlockOut>> = (0..config.blocks)
        .into_par_iter()
        .map(|block| -> Result<BlockOut> {
            let out = play_block(block, config, fl_ev, &fl_table, &table, t0, t1, t2)?;
            let done = finished_blocks.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 100 == 0 || done as u64 == config.blocks {
                // stderr only: the output file is a byte-for-byte determinism
                // test and progress must never reach it.
                eprintln!(
                    "fl-sim: {done}/{} blocks, {:.1} s",
                    config.blocks,
                    started.elapsed().as_secs_f64()
                );
            }
            Ok(out)
        })
        .collect();

    // Completion order is whatever the scheduler chose; the file is by block.
    let mut blocks = collected?;
    blocks.sort_by_key(|out| out.block);

    let meta = serde_json::json!({
        "schema": "ofc_flsim/v1",
        "width": config.width,
        "seed": config.seed,
        "blocks": config.blocks,
        "chains_per_block": config.chains_per_block,
        "fl_ev": table,
        "fl_ev_config_sha256": fl_ev.config_sha256.clone(),
        "models": {
            "t0": {"path": config.model_paths[0], "input_dim": t0.input_dim},
            "t1": {"path": config.model_paths[1], "input_dim": t1.input_dim},
            "t2": {"path": config.model_paths[2], "input_dim": t2.input_dim},
        },
        "deal": "block_seed=seed+b*1000003; deal(block_seed,h,width+opp_need); hero first",
    });

    use std::io::Write;
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output)?);
    writeln!(writer, "{}", serde_json::to_string(&meta)?)?;
    let mut chains = 0u64;
    let mut hands = 0u64;
    let mut headline_chains = 0u64;
    let mut headline_sum = 0i64;
    for block in &blocks {
        for line in &block.lines {
            writeln!(writer, "{line}")?;
            chains += 1;
        }
        hands += block.hands;
        headline_chains += block.headline_chains;
        headline_sum += block.headline_sum;
    }
    writer.flush()?;

    eprintln!(
        "fl-sim: {chains} chains ({hands} hands), width {}, mean(all,raw) {:+.3} over {headline_chains} \
         non-burn-in chains, {:.1} s",
        config.width,
        headline_sum as f64 / headline_chains.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// One block: `K` complete chains played back to back, the opponent's state
/// carrying across chain boundaries so that chains 1.. start from the
/// stationary distribution rather than from a forced normal.
#[allow(clippy::too_many_arguments)]
fn play_block(
    block: u64,
    config: &Config,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    table: &[f64; 4],
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    t2: &evaluator::Model,
) -> Result<BlockOut> {
    let block_seed = config.seed.wrapping_add(block.wrapping_mul(BLOCK_STRIDE));
    let mut opp_state = OppState::Normal;
    let mut hand_in_block: u32 = 0;
    let mut lines: Vec<String> = Vec::with_capacity(config.chains_per_block as usize);
    let mut block_hands = 0u64;
    let mut headline_chains = 0u64;
    let mut headline_sum = 0i64;

    for chain in 0..config.chains_per_block {
        let burn_in = chain == 0;
        let opp_at_start = opp_state.label();
        let mut hands: Vec<HandRow> = Vec::new();
        let mut sum_settle = 0i64;
        let mut solo = true;

        loop {
            let opp_label = opp_state.label();
            let need = config.width as usize + opp_state.need();
            let dealt = fl_solver::pool::deal(block_seed, hand_in_block as u64, need);
            let (hero_cards, opp_cards) = dealt.split_at(config.width as usize);

            // The normal player finishes first and the Fantasyland player
            // answers the board it can see (the 2026-08-05 rule); two
            // Fantasyland players are simultaneous and blind.
            let (opp_finished, opp_stay) = match opp_state {
                OppState::Normal => {
                    // The id seeds the chooser's sampling, so it has to be
                    // unique per deal: sharing one would make different hands
                    // share a completion draw.
                    let id = format!(
                        "flsim/{}/{}/{block}/{hand_in_block}",
                        config.seed, config.width
                    );
                    let trace = play_normal_traced(
                        &id, opp_cards, fl_ev, fl_table, table, t0, t1, t2,
                    )?;
                    (trace.finished, false)
                }
                OppState::Fl(m) => {
                    solo = false;
                    play_fl(opp_cards, m, table, None)
                }
            };

            // A busted opponent is still passed in: `play_fl` recognizes dead
            // rows and falls back to its blind maximum, which is where the
            // branch belongs (self_play.rs:374-388, hu_match.rs:1396-1401).
            let opponent: Option<&Finished> = match opp_state {
                OppState::Normal => Some(&opp_finished),
                OppState::Fl(_) => None,
            };
            let (hero_finished, hero_stay) =
                play_fl(hero_cards, config.width, table, opponent);

            // Real points only.  The opponent's Fantasyland credit is not money;
            // it is the value of a future hand, and that future hand is dealt by
            // the transition below and settles into this same chain.
            let settle_raw = settle(&hero_finished, &opp_finished);
            sum_settle += settle_raw as i64;
            hands.push(HandRow {
                h: hand_in_block,
                opp: opp_label,
                settle: settle_raw,
                hero_roy: hero_finished.royalty,
                hero_stay,
                opp_roy: opp_finished.royalty,
                opp_foul: opp_finished.busted,
                opp_entry: opp_finished.entry_width,
                opp_stay,
            });

            opp_state = match opp_state {
                OppState::Normal => {
                    if !opp_finished.busted && opp_finished.entry_width >= 14 {
                        OppState::Fl(opp_finished.entry_width)
                    } else {
                        OppState::Normal
                    }
                }
                OppState::Fl(m) => {
                    if opp_stay {
                        OppState::Fl(m)
                    } else {
                        OppState::Normal
                    }
                }
            };

            hand_in_block += 1;
            if hand_in_block >= MAX_HANDS_PER_BLOCK {
                bail!(
                    "block {block} reached {MAX_HANDS_PER_BLOCK} deals at width {}: a chain \
                     that long has probability around 10^-9000, so the stay flag or the \
                     loop's exit is broken rather than the run being unlucky",
                    config.width
                );
            }

            // The hero re-enters at the same width -- the frozen rule -- so a
            // stay continues this chain and nothing else ends it.
            if !hero_stay {
                break;
            }
        }

        let n_hands = hands.len() as u32;
        block_hands += n_hands as u64;
        if !burn_in {
            headline_chains += 1;
            headline_sum += sum_settle;
        }
        lines.push(serde_json::to_string(&ChainRow {
            block,
            chain,
            burn_in,
            sum_settle,
            n_hands,
            solo,
            opp_at_start,
            hands,
        })?);
    }

    Ok(BlockOut {
        block,
        lines,
        hands: block_hands,
        headline_chains,
        headline_sum,
    })
}
