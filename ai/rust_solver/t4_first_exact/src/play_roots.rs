//! The T3 root the trained chain actually reaches, from fourteen dealt cards.
//!
//! `fl_solver teach` labels T3 roots it *assigns*: the deal's first two cards
//! into the top row, the next four into the middle, the next three into the
//! bottom, then two discards and a draw.  Every one of those is a legal
//! position and none of them is a played one, so a model trained on them is
//! trained on a distribution it will never meet.  Here the same fourteen cards
//! are played instead -- T0 assigns the first five, T1 and T2 each place two
//! of three and discard one -- which also makes the two teachers directly
//! comparable, since they can be handed the same deals.
//!
//! Nothing is scored here.  Each street's model only picks a move, so no
//! opponent is drawn and no Fantasyland pool is needed; the pricing happens
//! downstream in `fl_solver teach --roots-file`, against the pool it always
//! used.
//!
//! # Where a deal's time goes
//!
//! Three nodes: the 232-opening T0 fan-out, then one T1 and one T2 fan-out of
//! about twenty candidates each.  T2 is served by the 104-dim model rather
//! than a 96-dim ranker, deliberately -- it is the model that serves the
//! street in production -- and that one choice is most of what a deal costs.
//! A T2 candidate's board still has four open slots, so its joint block
//! samples 400 completions instead of enumerating, and `encode_for` records
//! that block as ~98% of a candidate encode.
//!
//! Measured on 20 deals, single-threaded, seconds per deal:
//!
//! ```text
//!                          total     t0      t1      t2
//!   104-dim T2 (j100)      1.841   0.287   0.047   1.508
//!    96-dim T2 (ranker96)  0.359   0.287   0.047   0.025
//! ```
//!
//! So the joint block is 59x the rest of the T2 node and 82% of a deal, and
//! dropping to the 96-dim ranker would make the play step 5.1x cheaper --
//! against a T2 model whose held-out regret is 0.312 rather than 0.147.  With
//! every core busy the 104-dim shape measured 0.335 s/deal wall over 40 deals.

use anyhow::{anyhow, bail, Result};
use ofc_core::Card;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::evaluator;
use super::playout::{self, RowwiseMemo};
use super::t0_vs_fl::t0_candidates;
use super::{all_cards, to_core_card, CoreBoard, FlEv};

/// The width these models were trained under.  `encode_for` only forwards
/// `opp_count` to the non-FL14 context block, so no width reaches the features
/// on this path; naming the one they were trained under keeps a future width
/// from arriving silently.
const OPP_COUNT: u8 = 14;

/// One deal: five dealt at T0, then three drawn at each of T1, T2 and T3.
#[derive(Deserialize)]
pub struct PlayRequest {
    pub id: String,
    pub cards: Vec<String>,
}

/// The T3 position the chain reached, in exactly the shape `fl_solver teach`
/// deals for itself.
#[derive(Serialize)]
pub struct PlayedRoot {
    pub id: String,
    pub rows: [Vec<String>; 3],
    pub dead: Vec<String>,
    pub draw: Vec<String>,
}

/// Seconds inside each street's fan-out, so a run can say where a deal went
/// rather than leave the T2 encoder's cost to be guessed at.
#[derive(Default, Clone, Copy)]
pub struct StreetSeconds {
    pub t0: f64,
    pub t1: f64,
    pub t2: f64,
}

impl StreetSeconds {
    pub fn add(&mut self, other: &StreetSeconds) {
        self.t0 += other.t0;
        self.t1 += other.t1;
        self.t2 += other.t2;
    }
}

/// The deck minus what the deal has already handed out, by name.
///
/// By name and not by `Card`: X1 and X2 are two deck slots that compare equal
/// as `Card`, so removing a drawn joker by value would remove both and leave
/// every downstream encode short a card.
fn unseen_after(taken: &[String]) -> Result<Vec<Card>> {
    let held: BTreeSet<&str> = taken.iter().map(String::as_str).collect();
    all_cards()
        .iter()
        .filter(|name| !held.contains(name.as_str()))
        .map(|name| to_core_card(name))
        .collect()
}

/// Which drawn slot each of a candidate's placements took.
///
/// `candidates` dedupes on card identity and hands back `Card`s, but the root
/// this emits has to give back the names it was handed, and the two jokers are
/// separate deck slots that compare equal as `Card`.  Consuming one matching
/// slot per placement keeps the fourteen names a partition; which joker went
/// where is not a question the board can answer, and does not need to be.
fn slots_of(draw: &[Card; 3], placements: &[(usize, Card); 2]) -> Result<[usize; 2]> {
    let mut used = [false; 3];
    let mut out = [usize::MAX; 2];
    for (position, (_, card)) in placements.iter().enumerate() {
        let slot = (0..3)
            .find(|&slot| !used[slot] && draw[slot] == *card)
            .ok_or_else(|| anyhow!("a chosen placement holds a card the draw does not"))?;
        used[slot] = true;
        out[position] = slot;
    }
    Ok(out)
}

/// The board a street's model would leave, with the discarded name.
///
/// `node_seed` is the node's and not the candidate's: at T2 the joint block is
/// sampled, and two candidates judged on different completions differ by the
/// sampling noise before they differ by the move.
#[allow(clippy::too_many_arguments)]
fn play_street(
    model: &evaluator::Model,
    board: &mut CoreBoard,
    names: &mut [Vec<String>; 3],
    draw: &[Card; 3],
    draw_names: &[String],
    unseen: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    node_seed: &str,
) -> Result<String> {
    let mut features: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    let mut best = f32::NEG_INFINITY;
    let mut chosen: Option<[(usize, Card); 2]> = None;
    for candidate in playout::candidates(board, draw) {
        let mut next = board.clone();
        next.rows[candidate.placements[0].0].push(candidate.placements[0].1);
        next.rows[candidate.placements[1].0].push(candidate.placements[1].1);
        playout::encode_for(
            model, &next, unseen, OPP_COUNT, fl_ev, fl_table, memo, pool_key, node_seed,
            &mut features,
        )?;
        let predicted = model.predict(&features, &mut scratch);
        if predicted > best {
            best = predicted;
            chosen = Some(candidate.placements);
        }
    }
    let placements = chosen.ok_or_else(|| {
        anyhow!("no legal placement of the draw; the board reached this street full")
    })?;
    let slots = slots_of(draw, &placements)?;
    for (position, (row, card)) in placements.iter().enumerate() {
        board.rows[*row].push(*card);
        names[*row].push(draw_names[slots[position]].clone());
    }
    let discarded = (0..3)
        .find(|slot| !slots.contains(slot))
        .expect("two of three slots are placed");
    Ok(draw_names[discarded].clone())
}

/// Play T0, T1 and T2 with the trained models and hand back the T3 root.
pub fn play(
    request: &PlayRequest,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    t0_model: &evaluator::Model,
    t1_model: &evaluator::Model,
    t2_model: &evaluator::Model,
) -> Result<(PlayedRoot, StreetSeconds)> {
    if request.cards.len() != 14 {
        bail!(
            "a deal is fourteen cards -- five at T0 and three at each of T1, T2 \
             and T3 -- but {} has {}",
            request.id,
            request.cards.len()
        );
    }
    // The deck accounting below is by name, so a name the deck does not spell
    // has to be refused rather than resolved: "JK" parses as a joker and
    // matches neither X1 nor X2, which would leave both in the unseen pool.
    let deck: BTreeSet<String> = all_cards().into_iter().collect();
    let mut held: BTreeSet<&str> = BTreeSet::new();
    for name in &request.cards {
        if !deck.contains(name.as_str()) {
            bail!("{name} is not a deck card; spell the jokers X1 and X2");
        }
        if !held.insert(name.as_str()) {
            bail!("{name} appears twice in {}", request.id);
        }
    }

    let mut timing = StreetSeconds::default();
    let memo: RowwiseMemo = std::sync::Mutex::new(std::collections::HashMap::new());
    let mut board = CoreBoard {
        rows: [Vec::new(), Vec::new(), Vec::new()],
    };
    let mut names: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // T0 places all five with no draw and no discard, so the street has no
    // candidate list in the shared playout's sense; the openings the T0
    // teacher enumerated are the openings played here.
    let started = std::time::Instant::now();
    let mut dealt = [Card { rank: 0, suit: 0 }; 5];
    for (slot, name) in request.cards[0..5].iter().enumerate() {
        dealt[slot] = to_core_card(name)?;
    }
    let unseen_t0 = unseen_after(&request.cards[0..5])?;
    let mut features: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    let mut best = f32::NEG_INFINITY;
    let mut opening: Option<[usize; 5]> = None;
    for assignment in t0_candidates(&dealt) {
        let mut next = CoreBoard {
            rows: [Vec::new(), Vec::new(), Vec::new()],
        };
        for slot in 0..5 {
            next.rows[assignment[slot]].push(dealt[slot]);
        }
        playout::encode_for(
            t0_model,
            &next,
            &unseen_t0,
            OPP_COUNT,
            fl_ev,
            fl_table,
            &memo,
            // Each street sees a different unseen pool, so the memo's rowwise
            // slices must not carry across one; the street index is what keeps
            // them apart.
            0,
            &format!("play/{}/t0", request.id),
            &mut features,
        )?;
        let predicted = t0_model.predict(&features, &mut scratch);
        if predicted > best {
            best = predicted;
            opening = Some(assignment);
        }
    }
    let opening = opening.ok_or_else(|| anyhow!("no T0 opening for {}", request.id))?;
    for slot in 0..5 {
        board.rows[opening[slot]].push(dealt[slot]);
        names[opening[slot]].push(request.cards[slot].clone());
    }
    timing.t0 = started.elapsed().as_secs_f64();

    let mut dead: Vec<String> = Vec::with_capacity(2);
    for (street, first) in [(1usize, 5usize), (2, 8)] {
        let started = std::time::Instant::now();
        let draw_names = &request.cards[first..first + 3];
        let mut draw = [Card { rank: 0, suit: 0 }; 3];
        for (slot, name) in draw_names.iter().enumerate() {
            draw[slot] = to_core_card(name)?;
        }
        // The discard is seen, so it leaves the pool with the two placed
        // cards -- the same unseen set every candidate at the node is judged
        // against, and the one the teacher's encoder built.
        let unseen = unseen_after(&request.cards[0..first + 3])?;
        let model = if street == 1 { t1_model } else { t2_model };
        dead.push(play_street(
            model,
            &mut board,
            &mut names,
            &draw,
            draw_names,
            &unseen,
            fl_ev,
            fl_table,
            &memo,
            street as u64,
            &format!("play/{}/t{street}", request.id),
        )?);
        let elapsed = started.elapsed().as_secs_f64();
        if street == 1 {
            timing.t1 = elapsed;
        } else {
            timing.t2 = elapsed;
        }
    }

    Ok((
        PlayedRoot {
            id: request.id.clone(),
            rows: names,
            dead,
            draw: request.cards[11..14].to_vec(),
        },
        timing,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_solver::pool;
    use std::path::Path;

    const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

    fn models() -> Option<(evaluator::Model, evaluator::Model, evaluator::Model)> {
        let paths = [
            "D:/ofc_data/fl14_t0_model_v1/evaluator.bin",
            "D:/ofc_data/fl14_t1_model_v1/evaluator.bin",
            "D:/ofc_data/fl14_t2_model_j100/evaluator.bin",
        ];
        if paths.iter().any(|path| !Path::new(path).exists()) {
            return None;
        }
        let loaded: Vec<evaluator::Model> = paths
            .iter()
            .map(|path| {
                evaluator::Model::load(&std::fs::read(path).expect("read model"))
                    .expect("load model")
            })
            .collect();
        let mut models = loaded.into_iter();
        Some((
            models.next().unwrap(),
            models.next().unwrap(),
            models.next().unwrap(),
        ))
    }

    fn fl_ev() -> FlEv {
        FlEv::load(Path::new("../../config/fl_ev.json")).expect("fl_ev config")
    }

    /// A card name for a dealt `Card`, choosing X1 before X2 so a two-joker
    /// deal spells two distinct deck slots.
    fn name_of(card: &Card, jokers_used: &mut usize) -> String {
        if card.is_joker() {
            *jokers_used += 1;
            return format!("X{}", jokers_used);
        }
        let rank = "23456789TJQKA"
            .chars()
            .nth(card.rank as usize - 2)
            .expect("rank");
        let suit = "shdc".chars().nth(card.suit as usize).expect("suit");
        format!("{rank}{suit}")
    }

    /// The fourteen names `pool::deal` produces for one root, in deal order.
    fn dealt_names(seed: u64, root: u64) -> Vec<String> {
        let mut jokers = 0usize;
        pool::deal(seed, root, 14)
            .iter()
            .map(|card| {
                name_of(
                    &Card {
                        rank: card.rank,
                        suit: card.suit,
                    },
                    &mut jokers,
                )
            })
            .collect()
    }

    /// The arrangement `fl_solver teach` assigns to the same fourteen cards.
    fn dealt_root(id: &str, cards: &[String]) -> PlayedRoot {
        PlayedRoot {
            id: id.to_string(),
            rows: [
                cards[0..2].to_vec(),
                cards[2..6].to_vec(),
                cards[6..9].to_vec(),
            ],
            dead: cards[9..11].to_vec(),
            draw: cards[11..14].to_vec(),
        }
    }

    fn key_of(root: &PlayedRoot) -> String {
        let rows: Vec<String> = root
            .rows
            .iter()
            .map(|row| {
                let mut names = row.clone();
                names.sort();
                names.join(",")
            })
            .collect();
        format!("{}|{}", rows.join("|"), root.dead.join(","))
    }

    /// Seeds chosen for their joker counts, so the deck accounting is
    /// exercised where it can actually go wrong: two cards that compare equal
    /// as `Card` but occupy separate deck slots.
    fn roots_by_joker_count() -> Vec<(u64, usize)> {
        let seed = 0xD00D_0001u64;
        let mut found: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();
        for root in 0..4000u64 {
            let jokers = pool::deal(seed, root, 14)
                .iter()
                .filter(|card| card.rank == 0)
                .count();
            found.entry(jokers).or_insert(root);
        }
        (0..=2)
            .map(|jokers| {
                (
                    *found
                        .get(&jokers)
                        .unwrap_or_else(|| panic!("no deal with {jokers} jokers in 4000")),
                    jokers,
                )
            })
            .collect()
    }

    /// **The emitted root is a legal T3 position over exactly the deal.**
    ///
    /// Nine placed within capacity, two dead, three still to come, and the
    /// fourteen input names partitioned with nothing invented or lost -- the
    /// property the whole downstream teacher rests on, checked on deals with
    /// zero, one and two jokers.
    #[test]
    fn a_played_root_partitions_the_deal_into_a_legal_position() {
        let Some((t0, t1, t2)) = models() else {
            eprintln!("skipped: the FL14 chain models are not on this box");
            return;
        };
        let fl_ev = fl_ev();
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        for (root, jokers) in roots_by_joker_count() {
            let cards = dealt_names(0xD00D_0001, root);
            let request = PlayRequest {
                id: format!("play-test-{root}"),
                cards: cards.clone(),
            };
            let (played, _) = play(&request, &fl_ev, &fl_table, &t0, &t1, &t2)
                .unwrap_or_else(|error| panic!("root {root} ({jokers} jokers): {error}"));

            for row in 0..3 {
                assert!(
                    played.rows[row].len() <= ROW_CAPACITY[row],
                    "root {root}: row {row} holds {} of {}",
                    played.rows[row].len(),
                    ROW_CAPACITY[row]
                );
            }
            let placed: usize = played.rows.iter().map(Vec::len).sum();
            assert_eq!(placed, 9, "root {root}: {placed} cards placed at T3");
            assert_eq!(played.dead.len(), 2, "root {root}: {:?} dead", played.dead);
            assert_eq!(played.draw.len(), 3, "root {root}");
            assert_eq!(played.draw, cards[11..14], "root {root}: the draw moved");

            let mut emitted: Vec<String> = played
                .rows
                .iter()
                .flatten()
                .chain(played.dead.iter())
                .chain(played.draw.iter())
                .cloned()
                .collect();
            emitted.sort();
            let mut given = cards.clone();
            given.sort();
            assert_eq!(
                emitted, given,
                "root {root} ({jokers} jokers): the emitted root is not the deal"
            );

            let jokers_out = emitted.iter().filter(|name| name.starts_with('X')).count();
            assert_eq!(jokers_out, jokers, "root {root}: joker count moved");
        }
    }

    /// **Playing is not assigning.**
    ///
    /// Same fourteen cards, run through the chain and through the arrangement
    /// `teach` deals for itself.  If these agreed there would be nothing to
    /// build.
    #[test]
    fn the_played_arrangement_is_not_the_dealt_one() {
        let Some((t0, t1, t2)) = models() else {
            eprintln!("skipped: the FL14 chain models are not on this box");
            return;
        };
        let fl_ev = fl_ev();
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let mut differed = 0usize;
        let roots = 12u64;
        for root in 0..roots {
            let cards = dealt_names(0xD00D_0001, root);
            let request = PlayRequest {
                id: format!("{}", 0xD00D_0001u64.wrapping_add(root)),
                cards: cards.clone(),
            };
            let (played, _) = play(&request, &fl_ev, &fl_table, &t0, &t1, &t2).expect("play");
            if key_of(&played) != key_of(&dealt_root(&request.id, &cards)) {
                differed += 1;
            }
        }
        assert_eq!(
            differed, roots as usize,
            "{} of {roots} played roots landed on the dealt arrangement",
            roots as usize - differed
        );
    }
}
