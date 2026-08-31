//! Gen-2 vs-FL teacher: the value of a T0 opening when T1 is played well.
//!
//! The gen-1 referee (`fl_t0_deep`) replays production exactly, so a T0
//! candidate is worth whatever the shipped T1/T2 choosers happen to make of
//! it.  That is the right thing to audit and the wrong thing to teach: a
//! label built that way carries the chooser's mistakes into the next
//! generation.  Here every T1 node is *raced* instead of chosen, so a T0
//! candidate is worth what it is worth when T1 is played well -- one
//! bootstrap layer removed -- and the race's own ordering falls out as T1
//! training material for free.
//!
//! # The structural saving
//!
//! T0 places all five cards and discards nothing, so **the T1 draw does not
//! depend on the T0 choice**.  One set of T1 draws is therefore sampled once
//! per root and shared by every T0 candidate, which is both a variance
//! reduction (candidates are compared on the same futures) and most of the
//! cost saving.  One level down, the T2-onward futures are shared across the
//! T1 candidates racing at a node, for the same two reasons: those candidates
//! differ only in which two of three drawn cards were placed, and the third
//! is discarded-but-seen either way, so they face an identical deck.
//!
//! # Seed bands
//!
//! 910M T1 draws, 920M race futures, 930M the winner's extra measurement.
//! Everything at or below 900M is spent: 850/860 the miner, 870/880 the race,
//! 890 holdout sharpening, 900 the ship gate.  Keying on the root's index
//! rather than on a position within a run is what lets a shard and a local
//! run produce the same numbers.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::evaluator;
use super::hu_match;
use super::play_roots;
use super::playout;
use super::self_play::{self, own_worth};
use super::{to_core_card, CoreBoard, FlEv};

/// One T1 move: the board it reaches and the card it throws away.
#[derive(Clone)]
pub struct T1Move {
    pub key: String,
    pub rows: [Vec<String>; 3],
    pub discard: String,
}

#[derive(serde::Serialize)]
pub struct Scored {
    pub key: String,
    pub mean: f64,
    pub se: f64,
    pub n: usize,
}

/// A T1 decision with the ranking the race produced.
#[derive(serde::Serialize)]
pub struct T1Label {
    pub root: String,
    pub t0_key: String,
    /// 1-based rank of the T0 candidate under the shipped width-120 net.
    pub t0_rank: usize,
    /// The board T1 acts on, and the three cards it drew.
    pub board: [Vec<String>; 3],
    pub draw: Vec<String>,
    pub candidates: Vec<Scored>,
    pub winner: String,
    /// True when the T0 candidate is one the shipped net would plausibly
    /// serve, so the state is one production actually reaches.
    pub on_policy: bool,
}

#[derive(serde::Serialize)]
pub struct T0Label {
    pub key: String,
    pub t0_rank: usize,
    pub mean: f64,
    pub se: f64,
    /// T1 draws averaged over.
    pub n: usize,
    /// Hands played under this candidate, for the cost report.
    pub hands: usize,
}

/// Legal T1 moves from a board and a draw, jokers collapsed.
///
/// `playout::candidates` already dedupes on card identity, which is what
/// makes two interchangeable jokers one move rather than two; reusing it
/// keeps this enumeration the same one the choosers were trained against.
pub fn t1_moves(rows: &[Vec<String>; 3], draw_names: &[String]) -> Result<Vec<T1Move>> {
    let mut board = CoreBoard {
        rows: [Vec::new(), Vec::new(), Vec::new()],
    };
    for row in 0..3 {
        for name in &rows[row] {
            board.rows[row].push(to_core_card(name)?);
        }
    }
    let mut draw = [ofc_core::Card { rank: 0, suit: 0 }; 3];
    for (slot, name) in draw_names.iter().enumerate() {
        draw[slot] = to_core_card(name)?;
    }
    let mut out = Vec::new();
    for candidate in playout::candidates(&board, &draw) {
        // Recover which drawn NAME each placement used: the two jokers are
        // one Card but two deck slots, so placements must consume names.
        let mut left: Vec<usize> = (0..3).collect();
        let mut after = rows.clone();
        let mut ok = true;
        for (row, card) in candidate.placements.iter() {
            match left
                .iter()
                .position(|slot| to_core_card(&draw_names[*slot]).map(|c| c == *card).unwrap_or(false))
            {
                Some(at) => {
                    after[*row].push(draw_names[left[at]].clone());
                    left.remove(at);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || left.len() != 1 {
            continue;
        }
        let discard = draw_names[left[0]].clone();
        out.push(T1Move {
            key: hu_match::t0_key_of(&after) + "/" + &discard,
            rows: after,
            discard,
        });
    }
    if out.is_empty() {
        bail!("no legal T1 move from {rows:?} with {draw_names:?}");
    }
    Ok(out)
}

/// Mean and standard error of a sample.
fn stats(values: &[f32]) -> (f64, f64) {
    let n = values.len().max(1);
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let var = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / (n.saturating_sub(1).max(1)) as f64;
    (mean, (var / n as f64).sqrt())
}

pub struct Config {
    pub top_k: usize,
    pub t1_top: usize,
    pub n1: usize,
    /// Cumulative particles per race round.
    pub schedule: Vec<usize>,
    pub winner_extra: usize,
    pub sigma: f64,
    pub floor: f64,
    /// Cumulative particles a candidate needs before the flat floor may
    /// retire it.  Phase 1 measured this: gated at a hard 24 while the first
    /// round only ever reached 6, the floor could not fire at teacher
    /// budgets and 46 of 48 nodes carried every candidate to the end.  It now
    /// defaults to the first round's count, so elimination starts working at
    /// the first point there is anything to work with.
    pub floor_min_n: usize,
    pub root_index: u64,
}

/// Play one hand: T0 and T1 pinned, T2 by the chooser, T3/T4 exact own-worth.
#[allow(clippy::too_many_arguments)]
fn play_hand(
    id: &str,
    hero: &[fl_solver::Card],
    t0_rows: &[Vec<String>; 3],
    rows: &[Vec<String>; 3],
    discard: &str,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    table: &[f64; 4],
    models: [&evaluator::Model; 3],
    t2_fence: Option<(&evaluator::Model, usize)>,
) -> Result<f64> {
    let pinned = (rows.clone(), discard.to_string());
    // BOTH streets are pinned: T0 to the candidate being valued and T1 to the
    // move being raced.  Leaving T0 to the chooser would silently value a
    // different opening -- and the T1 diff would then fail against a board it
    // never built.
    let finished = self_play::play_normal_traced_forced(
        id, hero, fl_ev, fl_table, table, models[0], models[1], models[2],
        Some(t0_rows), Some(&pinned), t2_fence,
    )?
    .finished;
    Ok(own_worth(&finished, table))
}

/// One root: T0 labels, and the T1 rankings collected on the way.
#[allow(clippy::too_many_arguments)]
pub fn teach_root(
    root: &str,
    hero_names: &[String],
    config: &Config,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    models: [&evaluator::Model; 3],
    t2_fence: Option<(&evaluator::Model, usize)>,
) -> Result<(Vec<T0Label>, Vec<T1Label>, usize)> {
    let (hero, hero_cards) = hu_match::normalise_hero(hero_names)?;
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];

    // T0 field: the shipped width-120 net's own ordering, top-K.
    let ranked = super::fl_t0_deep::rank(&hero, fl_ev, fl_table, models[0])?;
    let field: Vec<(usize, String, [Vec<String>; 3])> = ranked
        .iter()
        .take(config.top_k.min(ranked.len()))
        .map(|row| {
            let rows = hu_match::t0_candidate_keys(&hero, Some(&[row.key.clone()]))?
                .pop()
                .expect("one key gives one board")
                .0;
            Ok((row.own_rank, row.key.clone(), rows))
        })
        .collect::<Result<Vec<_>>>()?;

    // The T1 draws, sampled once and shared by every T0 candidate: T0 places
    // all five and discards nothing, so the draw cannot depend on the choice.
    let mut draws: Vec<Vec<String>> = Vec::with_capacity(config.n1);
    for index in 0..config.n1 as u64 {
        let shuffled = hu_match::deal_names(910_000_000 + config.root_index * 10, index, 54);
        let mut need: Vec<fl_solver::Card> = hero_cards.clone();
        let mut rest: Vec<fl_solver::Card> = Vec::new();
        for card in shuffled {
            match need
                .iter()
                .position(|h| h.rank == card.rank && h.suit == card.suit)
            {
                Some(at) => {
                    need.swap_remove(at);
                }
                None => rest.push(card),
            }
        }
        let mut jokers = hero.iter().filter(|n| n.starts_with('X')).count();
        draws.push(
            rest[..3]
                .iter()
                .map(|card| {
                    if card.is_joker() {
                        jokers += 1;
                        format!("X{jokers}")
                    } else {
                        let rank = b"23456789TJQKA"[(card.rank - 2) as usize] as char;
                        let suit = b"shdc"[card.suit as usize] as char;
                        format!("{rank}{suit}")
                    }
                })
                .collect(),
        );
    }

    // Every (T0 candidate, T1 draw) node is independent, so the box is filled
    // at that level and the race inside a node stays sequential and cheap.
    let nodes: Vec<(usize, usize)> = (0..field.len())
        .flat_map(|c| (0..draws.len()).map(move |d| (c, d)))
        .collect();
    let done: Result<Vec<(usize, usize, f64, T1Label, usize)>> = nodes
        .par_iter()
        .map(|(ci, di)| {
            let (rank, t0_key, t0_rows) = &field[*ci];
            let draw = &draws[*di];
            let moves = t1_moves(t0_rows, draw)?;
            // Pre-rank by the shipped T1 chooser and race only the head of the
            // list.  The chooser costs no rollouts (96-dim, nothing sampled),
            // and racing twenty-odd moves at meaningful particle counts is
            // what puts this design over budget; the T0 field is cut the same
            // way for the same reason.
            let mut scored: Vec<(f32, usize)> = Vec::with_capacity(moves.len());
            for (index, mv) in moves.iter().enumerate() {
                let mut names: Vec<String> = mv.rows.iter().flatten().cloned().collect();
                names.push(mv.discard.clone());
                let score = play_roots::t1_move_score(
                    &mv.rows, &names, fl_ev, fl_table, models[1],
                )?;
                scored.push((score, index));
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut live: Vec<usize> = scored
                .iter()
                .take(config.t1_top.min(scored.len()))
                .map(|(_s, i)| *i)
                .collect();

            // The futures every T1 candidate at this node is judged on.  Same
            // deck for all of them (the discard is seen either way), so the
            // comparison is paired and the sampling noise cancels.
            let future_seed = 920_000_000 + config.root_index * 10;
            let node = (*ci * draws.len() + *di) as u64;
            let mut samples: Vec<Vec<f32>> = vec![Vec::new(); moves.len()];
            let mut hands = 0usize;
            let mut played = 0usize;
            for (round, want) in config.schedule.iter().enumerate() {
                if live.len() <= 1 {
                    break;
                }
                for index in live.iter() {
                    let mv = &moves[*index];
                    while samples[*index].len() < *want {
                        let tick = samples[*index].len() as u64;
                        let id = format!("g2/{future_seed}/{node}/{tick}");
                        let hero17 = future_hand(
                            future_seed, node, tick, &hero_cards, draw, &mv.discard,
                        )?;
                        let value = play_hand(
                            &id, &hero17, t0_rows, &mv.rows, &mv.discard,
                            fl_ev, fl_table, &table, models, t2_fence,
                        )?;
                        samples[*index].push(value as f32);
                        hands += 1;
                    }
                }
                played = *want;
                if round + 1 == config.schedule.len() {
                    break;
                }
                // Gap to the leader, in paired standard errors: a candidate
                // survives unless it trails by more than `sigma` of the
                // difference, or by more than the floor once it has been
                // measured enough for that to mean something.
                let leader = *live
                    .iter()
                    .max_by(|a, b| {
                        stats(&samples[**a])
                            .0
                            .partial_cmp(&stats(&samples[**b]).0)
                            .unwrap()
                    })
                    .expect("a live field is not empty");
                let lead_mean = stats(&samples[leader]).0;
                live.retain(|index| {
                    if *index == leader {
                        return true;
                    }
                    let diff: Vec<f32> = samples[leader]
                        .iter()
                        .zip(samples[*index].iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    let (gap, se) = stats(&diff);
                    let hopeless = played >= config.floor_min_n && gap > config.floor;
                    !(gap > config.sigma * se.max(1e-9) || hopeless)
                });
            }
            // The winner carries the T0 value, so it gets the extra particles.
            let winner = *live
                .iter()
                .max_by(|a, b| {
                    stats(&samples[**a])
                        .0
                        .partial_cmp(&stats(&samples[**b]).0)
                        .unwrap()
                })
                .expect("a race always leaves a leader");
            let extra_seed = 930_000_000 + config.root_index * 10;
            while samples[winner].len() < config.winner_extra {
                let tick = samples[winner].len() as u64;
                let id = format!("g2x/{extra_seed}/{node}/{tick}");
                let hero17 = future_hand(
                    extra_seed, node, tick, &hero_cards, draw, &moves[winner].discard,
                )?;
                let value = play_hand(
                    &id, &hero17, t0_rows, &moves[winner].rows, &moves[winner].discard,
                    fl_ev, fl_table, &table, models, t2_fence,
                )?;
                samples[winner].push(value as f32);
                hands += 1;
            }
            let (mean, _se) = stats(&samples[winner]);
            let candidates: Vec<Scored> = moves
                .iter()
                .enumerate()
                .filter(|(index, _)| !samples[*index].is_empty())
                .map(|(index, mv)| {
                    let (m, se) = stats(&samples[index]);
                    Scored {
                        key: mv.key.clone(),
                        mean: m,
                        se,
                        n: samples[index].len(),
                    }
                })
                .collect();
            let label = T1Label {
                root: root.to_string(),
                t0_key: t0_key.clone(),
                t0_rank: *rank,
                board: t0_rows.clone(),
                draw: draw.clone(),
                candidates,
                winner: moves[winner].key.clone(),
                on_policy: *rank <= 4,
            };
            Ok((*ci, *di, mean, label, hands))
        })
        .collect();
    let done = done?;

    let mut per_candidate: Vec<Vec<f32>> = vec![Vec::new(); field.len()];
    let mut per_hands: Vec<usize> = vec![0; field.len()];
    let mut t1_labels = Vec::with_capacity(done.len());
    let mut total_hands = 0usize;
    for (ci, _di, mean, label, hands) in done {
        per_candidate[ci].push(mean as f32);
        per_hands[ci] += hands;
        total_hands += hands;
        t1_labels.push(label);
    }
    let mut t0_labels: Vec<T0Label> = field
        .iter()
        .enumerate()
        .map(|(ci, (rank, key, _rows))| {
            let (mean, se) = stats(&per_candidate[ci]);
            T0Label {
                key: key.clone(),
                t0_rank: *rank,
                mean,
                se,
                n: per_candidate[ci].len(),
                hands: per_hands[ci],
            }
        })
        .collect();
    t0_labels.sort_by(|a, b| b.mean.partial_cmp(&a.mean).unwrap());
    Ok((t0_labels, t1_labels, total_hands))
}

/// Hero's seventeen for one future: the eight already known plus nine drawn.
fn future_hand(
    seed: u64,
    node: u64,
    tick: u64,
    hero_cards: &[fl_solver::Card],
    draw: &[String],
    _discard: &str,
) -> Result<Vec<fl_solver::Card>> {
    let mut known: Vec<fl_solver::Card> = hero_cards.to_vec();
    for name in draw {
        known.push(super::hu_match::card_of(name)?);
    }
    let shuffled = hu_match::deal_names(seed.wrapping_add(node), tick, 54);
    let mut need = known.clone();
    let mut rest: Vec<fl_solver::Card> = Vec::new();
    for card in shuffled {
        match need
            .iter()
            .position(|h| h.rank == card.rank && h.suit == card.suit)
        {
            Some(at) => {
                need.swap_remove(at);
            }
            None => rest.push(card),
        }
    }
    if !need.is_empty() {
        bail!("future deck exclusion failed");
    }
    let mut hand = known;
    hand.extend(rest[..9].iter().copied());
    Ok(hand)
}
