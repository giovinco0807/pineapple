//! T2 labels against a best-responding Fantasyland opponent.
//!
//! Hero holds seven, draws three, places two and discards one.  Two streets
//! remain, so unlike T4 there is a continuation to value and unlike T3 it
//! cannot be enumerated: a T3 draw is one of C(43,3) = 12,341, and each one
//! costs a full T3 solve.  So the T3 draw is the one sampled quantity added at
//! this street, on top of the opponent draw T3 and T4 already had.
//!
//! # What stays exact
//!
//! Everything below the sampled T3 draw.  Given a draw, every T3 placement is
//! enumerated and each is priced by `completion_value`, which enumerates every
//! C(40,3) T4 draw and every placement of it.  So a T2 action's value is
//!
//! ```text
//!     mean over sampled T3 draws of
//!       max over T3 placements of
//!         mean over ALL T4 draws of
//!           max over T4 placements of
//!             hero's exact score against the opponent's best response
//! ```
//!
//! The two maxima are hero playing on, not an assumption about hero: at T3 and
//! T4 hero sees the draw before choosing, so taking the best is what hero does.
//!
//! # Common random numbers, twice over
//!
//! Both sampled quantities are drawn once per root and shared by every action.
//!
//! The opponents can be shared because hero's seen cards are the same whatever
//! hero places.  The T3 draws can be shared for the same reason, and it is
//! worth being explicit about why: hero's discard is *seen*, so the unseen set
//! after a T2 action does not depend on which card was discarded.  All twenty
//! or so actions therefore face one list of T3 draws, and the draw cancels in
//! every action-to-action difference the teacher is consumed for.

use rayon::prelude::*;

use crate::pool::{draw, mask_of, mix64, Pool, PoolEntry, ShortDraw};
use crate::t3_labels::{
    board_key, completion_value, open_patterns, sampled_completion_value, t3_placements,
    unseen_from,
};
use crate::Card;

/// Mixed into the stream so the T3-draw sample and the opponent sample are
/// independent axes.
const T3_DRAW_STREAM: u64 = 0x7432;

/// A third independent axis: the sampled T4 draws under a T3 placement.
const T4_DRAW_STREAM: u64 = 0x7433;

pub struct T2Request {
    pub id: String,
    /// Hero's seven placed cards, by row.
    pub rows: [Vec<Card>; 3],
    /// Hero's earlier discard -- seen by hero, so it conditions the draw.
    pub dead: Vec<Card>,
    pub draw: [Card; 3],
    pub opponents: usize,
    /// T3 draws sampled per action.
    pub t3_draws: usize,
    /// T4 draws sampled under each T3 placement; 0 enumerates all C(40,3).
    ///
    /// Enumerating is exact and costs about thirteen times more.  A first lap
    /// that wants a whole chain sooner samples here; a final one does not.
    pub t4_draws: usize,
}

pub struct T2ActionValue {
    pub action_key: String,
    pub value: f64,
    pub t3_draws: usize,
    pub opponents: usize,
}

/// `count` draws of three from `pool_len`, deterministic in `stream`.
///
/// Indices into the root's unseen list, not cards, so the caller can prove the
/// list is action-independent without knowing what hero placed.
pub fn sampled_t3_draws(pool_len: usize, count: usize, stream: u64) -> Vec<[usize; 3]> {
    let mut out = Vec::with_capacity(count);
    let mut tick = 0u64;
    while out.len() < count {
        let mut picked = [0usize; 3];
        let mut filled = 0usize;
        while filled < 3 {
            let value = mix64(stream ^ (tick.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            tick += 1;
            let index = (value % pool_len as u64) as usize;
            if !picked[..filled].contains(&index) {
                picked[filled] = index;
                filled += 1;
            }
        }
        picked.sort_unstable();
        out.push(picked);
    }
    out
}

/// Every distinct nine-card board a T2 draw reaches, with the key naming it.
fn t2_placements(rows: &[Vec<Card>; 3], draw: &[Card; 3]) -> Vec<([Vec<Card>; 3], String, Card)> {
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard).collect();
        for pattern in open_patterns(rows) {
            let mut after = rows.clone();
            after[pattern[0]].push(draw[kept[0]]);
            after[pattern[1]].push(draw[kept[1]]);
            let key = board_key(&after, &draw[discard]);
            if seen.insert(key.clone()) {
                out.push((after, key, draw[discard]));
            }
        }
    }
    out
}

/// The best eleven-card board one T3 draw reaches from a nine-card board.
fn best_after_t3_draw(
    rows: &[Vec<Card>; 3],
    t3_draw: &[Card; 3],
    unseen_after: &[Card],
    opponents: &[&PoolEntry],
    fl_ev: &[f64; 4],
    t4_draws: usize,
    stream: u64,
) -> f64 {
    t3_placements(rows, t3_draw)
        .iter()
        .enumerate()
        .map(|(index, after)| {
            if t4_draws == 0 {
                let (total, draws) =
                    completion_value(after, unseen_after, opponents, fl_ev, false, None);
                total / draws.max(1) as f64
            } else {
                // The stream varies with the placement so two placements at one
                // node are not compared on the same T4 draws by accident --
                // sharing them there would bias the max toward whichever
                // placement those particular draws happened to suit.
                sampled_completion_value(
                    after,
                    unseen_after,
                    t4_draws,
                    stream.wrapping_add(index as u64),
                    opponents,
                    fl_ev,
                )
            }
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Every action's value at one T2 root.
pub fn solve(
    request: &T2Request,
    pool: &Pool,
    fl_ev: &[f64; 4],
    stream: u64,
) -> Result<Vec<T2ActionValue>, ShortDraw> {
    let mut seen: Vec<Card> = request.dead.clone();
    for row in &request.rows {
        seen.extend_from_slice(row);
    }
    seen.extend_from_slice(&request.draw);
    let (hero_naturals, hero_jokers) = mask_of(&seen);
    let opponents: Vec<&PoolEntry> =
        draw(pool, hero_naturals, hero_jokers, request.opponents, stream)?;

    // Action-independent, because hero's discard is seen either way.
    let unseen = unseen_from(&seen);
    // A separate stream from the opponents', so raising one sample count does
    // not silently reshuffle the other.
    let draws = sampled_t3_draws(unseen.len(), request.t3_draws, stream ^ T3_DRAW_STREAM);

    let actions = t2_placements(&request.rows, &request.draw);
    // Flat over (action, draw): twenty actions alone would leave most of a
    // sixteen-core machine idle, and the pairs are equal-cost.
    let work: Vec<(usize, usize)> = (0..actions.len())
        .flat_map(|a| (0..draws.len()).map(move |d| (a, d)))
        .collect();
    let priced: Vec<((usize, usize), f64)> = work
        .par_iter()
        .map(|(action_index, draw_index)| {
            let (rows, _, _) = &actions[*action_index];
            let picked = draws[*draw_index];
            let t3_draw = [unseen[picked[0]], unseen[picked[1]], unseen[picked[2]]];
            let unseen_after: Vec<Card> = unseen
                .iter()
                .enumerate()
                .filter(|(index, _)| !picked.contains(index))
                .map(|(_, card)| *card)
                .collect();
            (
                (*action_index, *draw_index),
                best_after_t3_draw(
                    rows,
                    &t3_draw,
                    &unseen_after,
                    &opponents,
                    fl_ev,
                    request.t4_draws,
                    stream
                        ^ T4_DRAW_STREAM
                        ^ ((*action_index as u64) << 32)
                        ^ (*draw_index as u64),
                ),
            )
        })
        .collect();

    let mut totals = vec![0.0f64; actions.len()];
    for ((action_index, _), value) in &priced {
        totals[*action_index] += value;
    }
    let mut values: Vec<T2ActionValue> = actions
        .iter()
        .enumerate()
        .map(|(index, (_, key, _))| T2ActionValue {
            action_key: key.clone(),
            value: totals[index] / draws.len() as f64,
            t3_draws: draws.len(),
            opponents: opponents.len(),
        })
        .collect();
    values.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::{build_entry, deal};

    const TABLE: [f64; 4] = [0.0, 10.7, 29.9, 63.5];

    fn tiny_pool(entries: usize) -> Pool {
        Pool {
            width: 14,
            fl_ev: TABLE,
            seed: 0x7211_0001,
            entries: (0..entries as u64)
                .map(|index| build_entry(0x7211_0001, index, 14, TABLE[0]))
                .collect(),
        }
    }

    fn request_from(seed: u64, opponents: usize, t3_draws: usize) -> T2Request {
        let cards = deal(seed, 0, 11);
        T2Request {
            id: "t".into(),
            rows: [cards[0..2].to_vec(), cards[2..5].to_vec(), cards[5..7].to_vec()],
            dead: cards[7..8].to_vec(),
            draw: [cards[8], cards[9], cards[10]],
            opponents,
            t3_draws,
            t4_draws: 0,
        }
    }

    /// Every legal placement appears once and finishes at nine cards.
    #[test]
    fn placements_are_distinct_and_complete() {
        let request = request_from(0x7211_1000, 1, 1);
        let list = t2_placements(&request.rows, &request.draw);
        let keys: std::collections::BTreeSet<&String> =
            list.iter().map(|(_, key, _)| key).collect();
        assert_eq!(keys.len(), list.len(), "a placement key repeats");
        for (rows, _, _) in &list {
            assert_eq!(rows[0].len() + rows[1].len() + rows[2].len(), 9);
            assert!(rows[0].len() <= 3 && rows[1].len() <= 5 && rows[2].len() <= 5);
        }
    }

    /// The T3 draw list depends on the root, not on the action.
    ///
    /// This is the property that lets the draw cancel between actions, and it
    /// holds only because hero's discard is seen: if the unseen set moved with
    /// the discard, sampling indices into it would silently give each action a
    /// different set of draws.
    #[test]
    fn the_t3_draws_do_not_depend_on_what_hero_placed() {
        let request = request_from(0x7211_2000, 1, 8);
        let mut seen: Vec<Card> = request.dead.clone();
        for row in &request.rows {
            seen.extend_from_slice(row);
        }
        seen.extend_from_slice(&request.draw);
        let root_unseen = unseen_from(&seen);
        for (rows, _, discard) in t2_placements(&request.rows, &request.draw) {
            let mut after_seen: Vec<Card> = request.dead.clone();
            after_seen.push(discard);
            for row in &rows {
                after_seen.extend_from_slice(row);
            }
            let after_unseen = unseen_from(&after_seen);
            assert_eq!(
                after_unseen.len(),
                root_unseen.len(),
                "an action changed how many cards are unseen"
            );
            let same = root_unseen
                .iter()
                .zip(&after_unseen)
                .all(|(a, b)| a.rank == b.rank && a.suit == b.suit);
            assert!(same, "an action changed WHICH cards are unseen");
        }
    }

    /// Sampled draws are three distinct indices, and reproducible.
    #[test]
    fn sampled_draws_are_distinct_and_deterministic() {
        let first = sampled_t3_draws(43, 20, 99);
        let again = sampled_t3_draws(43, 20, 99);
        assert_eq!(first, again, "the same stream gave a different list");
        assert_ne!(first, sampled_t3_draws(43, 20, 100), "streams collide");
        for picked in &first {
            assert!(picked[0] < picked[1] && picked[1] < picked[2], "not distinct");
            assert!(picked[2] < 43);
        }
    }

    /// A short draw propagates rather than averaging over whatever turned up.
    #[test]
    fn a_short_draw_is_reported() {
        let pool = tiny_pool(8);
        let request = request_from(0x7211_4000, 500, 2);
        match solve(&request, &pool, &TABLE, 1) {
            Err(ShortDraw { wanted, .. }) => assert_eq!(wanted, 500),
            Ok(_) => panic!("an 8-entry pool satisfied a 500-opponent draw"),
        }
    }

    /// An action's value is the mean of its per-draw bests, recomputed here
    /// from the parts so a change to the aggregation shows up.
    #[test]
    fn an_action_is_the_mean_of_its_per_draw_bests() {
        let pool = tiny_pool(3000);
        let request = request_from(0x7211_5000, 3, 2);
        let Ok(values) = solve(&request, &pool, &TABLE, 21) else {
            return;
        };
        let mut seen: Vec<Card> = request.dead.clone();
        for row in &request.rows {
            seen.extend_from_slice(row);
        }
        seen.extend_from_slice(&request.draw);
        let (naturals, jokers) = mask_of(&seen);
        let opponents = draw(&pool, naturals, jokers, request.opponents, 21).expect("draw");
        let unseen = unseen_from(&seen);
        let draws = sampled_t3_draws(unseen.len(), request.t3_draws, 21 ^ T3_DRAW_STREAM);
        let by_key: std::collections::HashMap<&str, f64> =
            values.iter().map(|v| (v.action_key.as_str(), v.value)).collect();
        for (rows, key, _) in t2_placements(&request.rows, &request.draw) {
            let mut total = 0.0f64;
            for picked in &draws {
                let t3_draw = [unseen[picked[0]], unseen[picked[1]], unseen[picked[2]]];
                let after: Vec<Card> = unseen
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !picked.contains(index))
                    .map(|(_, card)| *card)
                    .collect();
                total += best_after_t3_draw(
                    &rows, &t3_draw, &after, &opponents, &TABLE, 0, 0,
                );
            }
            let expected = total / draws.len() as f64;
            assert_eq!(
                by_key[key.as_str()].to_bits(),
                expected.to_bits(),
                "action {key} is not the mean of its per-draw bests"
            );
        }
    }
}
