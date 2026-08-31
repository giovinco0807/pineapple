//! Referee for hero's T0 when the opponent is already in Fantasyland.
//!
//! The T0-BB miner audits the shipped opening against a *normal* opponent.
//! This is the same loop for the other half of the serving surface: hero opens
//! five cards knowing the opponent is in Fantasyland, and the question is
//! whether `own_lap4/t0.bin`'s argmax is the placement that actually pays.
//!
//! # The objective is hero's own board (owner ruling, 2026-08-30)
//!
//! Against a Fantasyland opponent the referee scores hero's result alone:
//! royalties from all three rows, plus the Fantasyland entry hero earns, minus
//! six flat for a foul.  That is `self_play::own_worth` -- called here and by
//! the chain's own T4 greedy from the same function -- and it is the
//! 2026-08-14 own-hand EV objective the whole `own_lap4` chain was trained
//! under.
//!
//! So there is no settlement, no opponent stay cost, and no Fantasyland hand
//! is dealt or played at all.  The earlier version of this module scored
//! `settle(hero, opp)` against `play_fl`'s best response; that made the T0
//! ordering a function of an opponent the chooser was never trained to see.
//! What remains ranks openings by the yardstick the chain declares, which is
//! the only comparison whose disagreements are the chain's own errors.
//!
//! # What is still pinned
//!
//! **The chain is production's, not a better one.** T0 is pinned to the
//! candidate, T1/T2 run the shipped `own_lap4` choosers and T3/T4 the
//! own-worth greedy the chain ships with.  A referee that upgraded the
//! downstream streets would report the value of a hand nobody plays, and the
//! T0 ordering it produced would be for a different game.
//!
//! Rollout `r` deals one set of twelve future cards and every candidate gets
//! it, so `scores[r]` of two rows differ only by the opening and their
//! difference is the paired statistic the caller averages.  Candidate
//! differences are small next to draw variance -- pairing is what makes a
//! batch of a few hundred rollouts say anything.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::evaluator;
use super::hu_match::{self, T0DeepRow};
use super::play_roots;
use super::self_play::{self, own_worth};
use super::FlEv;

/// One opening as the shipped T0 chooser ranks it, no rollouts.
///
/// The serving decision is this argmax -- there is no ranker or policy fence
/// on this path -- so `own_rank` 1 is the placement the referee is auditing.
#[derive(serde::Serialize)]
pub struct FlT0RankRow {
    pub key: String,
    pub score: f32,
    pub own_rank: usize,
}

/// What `own_lap4/t0.bin` thinks of every opening, in its own order.
///
/// The scores come from `play_roots::t0_scores`, the same call a played hand
/// makes, so the top row is the played T0 by construction and not by a second
/// encoder that agrees today.  Ties break the way the chain breaks them: the
/// sort is stable over the chooser's own enumeration order, matching the
/// strict `>` argmax the chain runs.
pub fn rank(
    hero: &[String],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    t0: &evaluator::Model,
) -> Result<Vec<FlT0RankRow>> {
    let (hero, _cards) = hu_match::normalise_hero(hero)?;
    // The id only reaches the sampled FL14 encoder widths; the 96-dim chooser
    // this chain ships with never reads it, which is what lets a ranking run
    // and a hand agree without sharing one.
    let mut scored: Vec<(usize, [Vec<String>; 3], f32)> =
        play_roots::t0_scores("fl-t0-rank", &hero, fl_ev, fl_table, t0)?
            .into_iter()
            .enumerate()
            .map(|(order, (assignment, score))| {
                let mut rows: [Vec<String>; 3] = Default::default();
                for slot in 0..5 {
                    rows[assignment[slot]].push(hero[slot].clone());
                }
                (order, rows, score)
            })
            .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    Ok(scored
        .into_iter()
        .enumerate()
        .map(|(rank, (_order, rows, score))| FlT0RankRow {
            key: hu_match::t0_key_of(&rows),
            score,
            own_rank: rank + 1,
        })
        .collect())
}

/// Hero's seventeen for one rollout: the five under test plus twelve drawn.
///
/// Dealing a full 54 and filtering keeps the shuffle identical to the match
/// harness rather than inventing a second one.  The twelve are a function of
/// `(seed, r)` alone -- no candidate can steer them, which is the whole point
/// of the pairing.  Nothing is dealt to an opponent: under the own-hand
/// objective there is no opponent to deal to.
fn deal(seed: u64, rollout: u64, hero_cards: &[fl_solver::Card]) -> Result<Vec<fl_solver::Card>> {
    let shuffled = hu_match::deal_names(seed, rollout, 54);
    let mut need: Vec<fl_solver::Card> = hero_cards.to_vec();
    let mut rest: Vec<fl_solver::Card> = Vec::with_capacity(49);
    for card in shuffled {
        // By value and one slot at a time: X1 and X2 are two deck slots that
        // compare equal, so removing "the joker" would remove both.
        if let Some(at) = need
            .iter()
            .position(|h| h.rank == card.rank && h.suit == card.suit)
        {
            need.swap_remove(at);
        } else {
            rest.push(card);
        }
    }
    if !need.is_empty() {
        bail!("deck exclusion failed for rollout {rollout}");
    }
    let mut hero: Vec<fl_solver::Card> = hero_cards.to_vec();
    hero.extend(rest[..12].iter().copied());
    Ok(hero)
}

/// Deep evaluation of hero's opening, scored on hero's own finished board.
///
/// One invocation is one batch on one seed.  A verdict wants several: a single
/// batch's argmax carries the winner's curse, and the caller re-scores any
/// disagreement on fresh seeds for exactly that reason.
///
/// `t2_fence` cascades the chain's own T2 street the way production would be
/// asked to serve it; `None` replays the uncascaded chain.  A referee run and
/// the serving it audits have to agree on this, or the ordering is for a
/// chain nobody plays.
#[allow(clippy::too_many_arguments)]
pub fn deep_eval(
    hero: &[String],
    wanted: Option<&[String]>,
    rollouts: usize,
    seed: u64,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    models: [&evaluator::Model; 3],
    t2_fence: Option<(&evaluator::Model, usize)>,
) -> Result<Vec<T0DeepRow>> {
    let (hero, hero_cards) = hu_match::normalise_hero(hero)?;
    let chosen = hu_match::t0_candidate_keys(&hero, wanted)?;
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    // Dealt up front so the parallel map is flat over (rollout, candidate):
    // a 232-candidate batch at four rollouts has only four rollout-shaped
    // tasks, which would leave most of the box idle.
    let deals: Vec<Vec<fl_solver::Card>> = (0..rollouts as u64)
        .map(|rollout| deal(seed, rollout, &hero_cards))
        .collect::<Result<Vec<_>>>()?;

    let flat: Result<Vec<(usize, usize, f32)>> = (0..rollouts * chosen.len())
        .into_par_iter()
        .map(|index| {
            let rollout = index / chosen.len();
            let candidate = index % chosen.len();
            // The id seeds the encoders' sample streams; candidates share it
            // so identical sub-states sample identically.
            let id = format!("d/{seed}/{rollout}");
            let hero_board = self_play::play_normal_traced_forced(
                &id,
                &deals[rollout],
                fl_ev,
                fl_table,
                &table,
                models[0],
                models[1],
                models[2],
                Some(&chosen[candidate].0),
                None,
                t2_fence,
            )?
            .finished;
            Ok((rollout, candidate, own_worth(&hero_board, &table) as f32))
        })
        .collect();

    let mut per_candidate: Vec<Vec<f32>> = vec![vec![0.0; rollouts]; chosen.len()];
    for (rollout, candidate, score) in flat? {
        per_candidate[candidate][rollout] = score;
    }
    let mut out: Vec<T0DeepRow> = chosen
        .iter()
        .enumerate()
        .map(|(index, (_rows, key))| {
            let scores = std::mem::take(&mut per_candidate[index]);
            let mean = scores.iter().map(|s| *s as f64).sum::<f64>() / scores.len().max(1) as f64;
            T0DeepRow {
                key: key.clone(),
                n: scores.len(),
                mean,
                scores,
            }
        })
        .collect();
    out.sort_by(|a, b| b.mean.partial_cmp(&a.mean).unwrap());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_play::Finished;

    fn hero() -> Vec<fl_solver::Card> {
        // Ah,7c,X1,9d,2s -- one joker, so the exclusion has a value that
        // matches two deck slots and must consume only one.
        vec![
            fl_solver::Card { rank: 14, suit: 1 },
            fl_solver::Card { rank: 7, suit: 3 },
            fl_solver::Card { rank: 0, suit: 4 },
            fl_solver::Card { rank: 9, suit: 2 },
            fl_solver::Card { rank: 2, suit: 0 },
        ]
    }

    /// **A rollout's twelve future cards are a function of (seed, r) alone.**
    ///
    /// The common-random-numbers claim in one assertion: repeat the deal and
    /// the same twelve come back, which is what makes `scores[r]` differences
    /// paired across candidates.  Different rollouts must not collide, or the
    /// batch is one deal counted many times.
    #[test]
    fn a_rollout_deals_the_same_cards_every_time() {
        let hero = hero();
        for rollout in 0..8u64 {
            let first = deal(0xFEED_0001, rollout, &hero).expect("deal");
            let again = deal(0xFEED_0001, rollout, &hero).expect("deal");
            assert_eq!(first.len(), 17, "a played hand is five plus twelve");
            assert_eq!(&first[..5], &hero[..], "hero's five moved");
            for slot in 0..17 {
                assert_eq!(
                    (first[slot].rank, first[slot].suit),
                    (again[slot].rank, again[slot].suit),
                    "rollout {rollout} dealt hero differently"
                );
            }
        }
        let a = deal(0xFEED_0001, 0, &hero).expect("deal");
        let b = deal(0xFEED_0001, 1, &hero).expect("deal");
        assert!(
            (5..17).any(|slot| (a[slot].rank, a[slot].suit) != (b[slot].rank, b[slot].suit)),
            "two rollouts drew the same twelve cards"
        );
    }

    /// **Hero's five leave the deck exactly once.**
    ///
    /// The joker is the case worth a test: it matches two deck slots by value,
    /// so a hero holding one must still leave the other available to be drawn.
    #[test]
    fn the_deal_partitions_the_deck() {
        let hero = hero();
        let dealt = deal(0xFEED_0002, 3, &hero).expect("deal");
        let jokers = dealt.iter().filter(|card| card.rank == 0).count();
        assert!(jokers <= 2, "{jokers} jokers dealt out of a two-joker deck");
        let mut seen: Vec<(u8, u8)> = dealt
            .iter()
            .filter(|card| card.rank != 0)
            .map(|card| (card.rank, card.suit))
            .collect();
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count, "a natural card was dealt twice");
    }

    /// **The referee's score is the chain's own T4 comparator.**
    ///
    /// Both call `self_play::own_worth`, so this pins the contract that
    /// function carries rather than re-deriving it: a foul is -6 flat and
    /// earns no royalties, and an entry is priced from the fl_ev table.
    #[test]
    fn the_objective_is_royalties_plus_entry_and_a_flat_foul() {
        let table = [6.57, 16.61, 38.76, 70.07];
        let scored = |busted, royalty, entry_width| {
            own_worth(
                &Finished {
                    busted,
                    top: 0,
                    mid: 0,
                    bot: 0,
                    royalty,
                    entry_width,
                },
                &table,
            )
        };
        assert_eq!(scored(false, 9, 0), 9.0, "royalties, no entry");
        assert_eq!(scored(false, 9, 14), 9.0 + 6.57, "royalties plus the entry");
        assert_eq!(scored(false, 0, 17), 70.07, "the widest entry");
        assert_eq!(scored(true, 25, 0), -6.0, "a foul is flat and keeps nothing");
    }
}
