//! T3-vs-FL labels by direct library scoring -- no learned model anywhere.
//!
//! Measurement drove this module: the learned T4 leaf injects MAE 0.645
//! (max 2.36) into T3 labels because its systematic per-board errors are
//! correlated across a root's draws and survive the 300-draw average.  Here
//! every T4 completion is scored directly against the FL board library, so
//! the label error is Monte Carlo only (draws and library sampling).
//!
//! The response also carries, per action, the blocks the 109-dim encoder
//! needs: the own-board joint completion block and the per-row completion
//! outlook, both computed over the hero's unseen pool.

use anyhow::{bail, Result};
use ofc_core::{
    check_fl_entry, evaluate_hand_value, get_bottom_royalty, get_middle_royalty, get_top_royalty,
    Card,
};
use rayon::prelude::*;
use serde::Serialize;

use super::evaluator;
use super::t3_second;
use super::t3_vs_fl::T3VsFlRequest;
use super::{all_cards, apply, legal_actions, terminal_of, to_core_card, CoreBoard, FlEv};

pub struct FlLibrary {
    pub(crate) masks: Vec<u64>,
    pub(crate) values: Vec<[u32; 3]>,
    pub(crate) royalty: Vec<f64>,
    pub(crate) stay: Vec<bool>,
    pub(crate) busted: Vec<bool>,
}

impl FlLibrary {
    pub fn load(dirs: &[std::path::PathBuf]) -> Result<Self> {
        let mut lib = FlLibrary {
            masks: Vec::new(),
            values: Vec::new(),
            royalty: Vec::new(),
            stay: Vec::new(),
            busted: Vec::new(),
        };
        for dir in dirs {
            let mut shards: Vec<_> = std::fs::read_dir(dir)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.starts_with("shard_") && name.ends_with(".jsonl"))
                        .unwrap_or(false)
                })
                .collect();
            shards.sort();
            for shard in shards {
                for line in std::fs::read_to_string(&shard)?.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let row: serde_json::Value = serde_json::from_str(line)?;
                    lib.masks.push(row["mask"].as_u64().unwrap());
                    let values = row["values"].as_array().unwrap();
                    lib.values.push([
                        values[0].as_u64().unwrap() as u32,
                        values[1].as_u64().unwrap() as u32,
                        values[2].as_u64().unwrap() as u32,
                    ]);
                    lib.royalty.push(row["royalty"].as_f64().unwrap());
                    lib.stay.push(row["stay"].as_bool().unwrap());
                    lib.busted.push(row["busted"].as_bool().unwrap());
                }
            }
        }
        if lib.masks.is_empty() {
            bail!("FL library is empty");
        }
        Ok(lib)
    }

    pub fn len(&self) -> usize {
        self.masks.len()
    }
}

/// Per-count FL board libraries.  The opponent's board distribution differs
/// by entry count (a 17-card Fantasyland places far stronger boards than a
/// 14-card one: measured mean royalty 27.7 vs 15.3, stay 78% vs 35%), so
/// scoring selects the matching library.  Counts without their own library
/// fall back to the 14-card one -- the pre-fix behavior, kept so old runs
/// stay reproducible when the extra libraries are not passed.
pub struct LibrarySet {
    base: FlLibrary,
    extra: [Option<FlLibrary>; 3],
}

impl LibrarySet {
    pub fn load(
        base_dirs: &[std::path::PathBuf],
        extra_dirs: [Option<&std::path::PathBuf>; 3],
    ) -> Result<Self> {
        let base = FlLibrary::load(base_dirs)?;
        let mut extra: [Option<FlLibrary>; 3] = [None, None, None];
        for (slot, dir) in extra_dirs.into_iter().enumerate() {
            if let Some(dir) = dir {
                extra[slot] = Some(FlLibrary::load(std::slice::from_ref(dir))?);
            }
        }
        Ok(LibrarySet { base, extra })
    }

    pub fn for_count(&self, opp_count: u8) -> &FlLibrary {
        match opp_count {
            15..=17 => self.extra[(opp_count - 15) as usize]
                .as_ref()
                .unwrap_or(&self.base),
            _ => &self.base,
        }
    }
}

pub(crate) fn card_bit(card: &Card) -> Result<u64> {
    // Index order pinned to Python ALL_CARDS: suits s,h,d,c x ranks 2..A,
    // then X1=52.  Both jokers map to bit 52|53; the mask test only needs
    // "is this physical card taken", and jokers in the library mask carry
    // their own bits, so hero-side jokers must cover both.
    if card.is_joker() {
        return Ok((1u64 << 52) | (1u64 << 53));
    }
    let suit_order = [0u64, 1, 2, 3]; // s,h,d,c already match ofc_core suits
    let index = suit_order[card.suit as usize] * 13 + (card.rank as u64 - 2);
    Ok(1u64 << index)
}

/// One matched library row, packed contiguously so the per-pattern scoring
/// loop scans L1-resident memory instead of gathering five 120k-wide arrays
/// through an index vector (measured: that gather was ~97% of teacher time).
#[derive(Clone, Copy)]
pub(crate) struct MatchedRow {
    pub(crate) values: [u32; 3],
    pub(crate) royalty: f64,
    pub(crate) stay: bool,
    pub(crate) busted: bool,
}

/// Everything score_mean reads from a terminal: against a fixed matched set
/// the score is a pure function of this key, so per-draw memoization over
/// the placement patterns is exact.
pub(crate) type TerminalKey = (bool, i32, u8, [u32; 3]);

pub(crate) fn terminal_key(terminal: &super::Terminal) -> TerminalKey {
    (
        terminal.busted,
        terminal.royalty,
        terminal.fl_card_count,
        terminal.values,
    )
}

/// Mean canonical score of one completed hero board over the matched rows.
/// Iteration order matches the pre-optimization code exactly, so the float
/// sum -- and therefore every emitted label byte -- is unchanged.
pub(crate) fn score_mean(
    terminal: &super::Terminal,
    matched: &[MatchedRow],
    opp_count: u8,
    fl_ev: &FlEv,
) -> f64 {
    let hero_entry_ev = if terminal.busted {
        0.0
    } else {
        fl_ev.value(terminal.fl_card_count)
    };
    let stay_cost = fl_ev.value(opp_count);
    let mut total = 0.0f64;
    for row in matched {
        let fl_busted = row.busted;
        let fl_royalty = row.royalty;
        let base = if terminal.busted {
            if fl_busted {
                0.0
            } else {
                -6.0 - fl_royalty
            }
        } else if fl_busted {
            6.0 + terminal.royalty as f64
        } else {
            let fl_values = &row.values;
            let mut lines = 0i32;
            for row in 0..3 {
                let (a, b) = (terminal.values[row], fl_values[row]);
                lines += (a > b) as i32 - (a < b) as i32;
            }
            let scoop = if lines == 3 {
                3.0
            } else if lines == -3 {
                -3.0
            } else {
                0.0
            };
            lines as f64 + scoop + terminal.royalty as f64 - fl_royalty
        };
        let stay_term = if row.stay && !fl_busted {
            stay_cost
        } else {
            0.0
        };
        total += base + hero_entry_ev - stay_term;
    }
    total / matched.len().max(1) as f64
}

#[derive(Serialize)]
pub struct LibActionValue {
    pub action_key: String,
    pub value: f64,
    pub draws: usize,
    pub mean_fl_samples: f64,
    pub own_joint_block: [f64; 8],
    pub own_rowwise_block: Vec<f32>,
    /// Full 109-dim encoder vector (actor + rowwise + joint + FL context),
    /// emitted under T3FL_EMIT_FEATURES=1 for cross-language parity checks
    /// and consumed by the in-engine T3 policy during T2 playouts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<f32>>,
}

#[derive(Serialize)]
pub struct LibResponse {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    pub library_boards: usize,
    pub actions: Vec<LibActionValue>,
}

pub fn solve(
    request: &T3VsFlRequest,
    fl_ev: &FlEv,
    libraries: &LibrarySet,
    fl_table: &evaluator::FlTable,
) -> Result<LibResponse> {
    let library = libraries.for_count(request.opp_count);
    let base = CoreBoard::from_str_board(&request.board)?;
    if base.card_count() != 9 {
        bail!("T3-vs-FL needs a 9-card hero board");
    }

    // The seen set is board+draw+dead -- action-independent -- so the unseen
    // pool and the library shortlist are computed once per root and shared
    // across every action (they were previously rebuilt per action).
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for card in request
        .board
        .top
        .iter()
        .chain(&request.board.middle)
        .chain(&request.board.bottom)
        .chain(&request.draw)
        .chain(&request.dead)
    {
        seen.insert(card.clone());
    }
    let unseen: Vec<Card> = all_cards()
        .into_iter()
        .filter(|card| !seen.contains(card))
        .map(|card| to_core_card(&card))
        .collect::<Result<Vec<_>>>()?;
    let mut seen_mask = 0u64;
    for name in &seen {
        seen_mask |= card_bit(&to_core_card(name)?)?;
    }
    // Jokers the hero holds cover both joker bits; recompute the
    // hero's actual joker usage so an unseen joker stays available.
    let hero_jokers = seen
        .iter()
        .filter(|name| *name == "X1" || *name == "X2")
        .count();
    if hero_jokers == 1 {
        // Only one joker is seen; leave the other bit clear.  The
        // library masks distinguish X1 and X2, and either being held
        // by the hero blocks only itself -- approximate by clearing
        // the higher bit, which the sampler never favours.
        seen_mask &= !(1u64 << 53);
    }

    // Two-stage filter: boards compatible with the pre-draw seen set,
    // then per-draw refinement inside that shortlist.
    let shortlist: Vec<u32> = (0..library.len() as u32)
        .filter(|&index| library.masks[index as usize] & seen_mask == 0)
        .collect();

    let actions = legal_actions(&base, &request.draw);
    let results: Result<Vec<LibActionValue>> = actions
        .par_iter()
        .map(|action| {
            let after = apply(&base, action)?;
            let timings = std::env::var("T3FL_PROFILE").as_deref() == Ok("timings");
            let mut phase = [0.0f64; 5]; // draws, filter, terminal, score, blocks
            let clock = std::time::Instant::now();
            let draw_sets = super::t3_vs_fl::draws_for(
                request,
                &unseen,
                &format!("{}/{}", request.id, action.key()),
            )?;
            phase[0] = clock.elapsed().as_secs_f64();
            let patterns = t3_second::placement_patterns(&after);

            let mut total = 0.0f64;
            let mut sample_total = 0usize;
            let mut matched: Vec<MatchedRow> = Vec::with_capacity(shortlist.len());
            let mut memo: std::collections::HashMap<TerminalKey, f64> =
                std::collections::HashMap::with_capacity(patterns.len());
            let mut terminal_memo = super::row_memo::TerminalMemo::new(&after);
            for draw in &draw_sets {
                let mark = std::time::Instant::now();
                let draw_cards = [unseen[draw[0]], unseen[draw[1]], unseen[draw[2]]];
                let mut draw_mask = 0u64;
                for card in &draw_cards {
                    draw_mask |= card_bit(card)?;
                }
                // Gather the matched rows once per draw into a contiguous
                // scratch buffer; the pattern loop then scans L1-resident
                // rows instead of re-walking the index indirection 20 times.
                matched.clear();
                for &index in &shortlist {
                    let index = index as usize;
                    if library.masks[index] & draw_mask == 0 {
                        matched.push(MatchedRow {
                            values: library.values[index],
                            royalty: library.royalty[index],
                            stay: library.stay[index],
                            busted: library.busted[index],
                        });
                    }
                }
                phase[1] += mark.elapsed().as_secs_f64();
                if matched.is_empty() {
                    continue;
                }
                sample_total += matched.len();
                let mut best = f64::NEG_INFINITY;
                memo.clear();
                for pattern in &patterns {
                    let mark = std::time::Instant::now();
                    let terminal = terminal_memo.terminal(&[
                        (pattern.rows[0], draw_cards[pattern.cards[0]]),
                        (pattern.rows[1], draw_cards[pattern.cards[1]]),
                    ]);
                    phase[2] += mark.elapsed().as_secs_f64();
                    // Distinct patterns often complete into the same terminal
                    // (values, royalty, bust, FL count); against a fixed
                    // matched set the score is a pure function of that key.
                    let mark = std::time::Instant::now();
                    let key = terminal_key(&terminal);
                    let value = match memo.get(&key) {
                        Some(cached) => *cached,
                        None => {
                            let computed = score_mean(
                                &terminal, &matched, request.opp_count, fl_ev,
                            );
                            memo.insert(key, computed);
                            computed
                        }
                    };
                    phase[3] += mark.elapsed().as_secs_f64();
                    if value > best {
                        best = value;
                    }
                }
                total += best;
            }
            let effective_draws = draw_sets.len().max(1);

            // T3FL_PROFILE=blocks_off zeroes the encoder blocks; labels keep
            // their exact path.  Debug-only: never set in production runs.
            let mark = std::time::Instant::now();
            let profile_off = std::env::var("T3FL_PROFILE").as_deref() == Ok("blocks_off");
            let mut joint_elapsed = 0.0f64;
            let (rowwise, joint) = if profile_off {
                (vec![0.0f32; 41], [0.0f64; 8])
            } else {
                let mut rowwise: Vec<f32> = Vec::with_capacity(evaluator::OPPONENT_SIZE);
                let _categories = evaluator::opponent_rowwise_block(
                    &after.rows, &unseen, fl_table, &mut rowwise,
                );
                let joint_mark = std::time::Instant::now();
                let joint = t3_second::joint_block(&after, &unseen, fl_ev)?;
                joint_elapsed = joint_mark.elapsed().as_secs_f64();
                (rowwise, joint)
            };
            phase[4] = mark.elapsed().as_secs_f64() - joint_elapsed;
            if timings {
                eprintln!(
                    "PHASE draws={:.3} filter={:.3} terminal={:.3} score={:.3} rowwise={:.3} joint={:.3}",
                    phase[0], phase[1], phase[2], phase[3], phase[4], joint_elapsed,
                );
            }

            let features = if std::env::var("T3FL_EMIT_FEATURES").as_deref() == Ok("1") {
                let mut vector: Vec<f32> = Vec::with_capacity(109);
                evaluator::actor_block(&after.rows, &mut vector);
                vector.extend_from_slice(&rowwise);
                vector.extend(joint.iter().map(|value| *value as f32));
                super::t3_vs_fl::fl_context(
                    &unseen, request.opp_count, fl_ev, &mut vector,
                );
                if vector.len() != 109 {
                    bail!("t3-vs-fl feature vector drifted: {}", vector.len());
                }
                Some(vector)
            } else {
                None
            };

            Ok(LibActionValue {
                action_key: action.key(),
                value: total / effective_draws as f64,
                draws: draw_sets.len(),
                mean_fl_samples: sample_total as f64 / effective_draws as f64,
                own_joint_block: joint,
                own_rowwise_block: rowwise,
                features,
            })
        })
        .collect();

    let mut actions_out = results?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(LibResponse {
        id: request.id.clone(),
        schema: "ofc_t3_vs_fl_value_library/v1",
        leaf: "direct_fl_library_scoring",
        library_boards: library.len(),
        actions: actions_out,
    })
}
