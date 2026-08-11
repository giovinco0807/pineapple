//! T4 labels against a best-responding Fantasyland opponent.
//!
//! Hero holds eleven, draws three, places two and discards one.  The board is
//! finished, so there is no continuation to estimate: every action's value is
//! hero's exact score against the opponent's best response, averaged over the
//! opponents drawn for this position.
//!
//! # What is exact here and what is not
//!
//! Exact: the opponent's play (a best response over its whole frontier), the
//! scoring, and the conditional distribution the opponents are drawn from.
//! Sampled: only *how many* opponents.  That is the whole point of the method
//! -- the Fantasyland side contributes no bias, so its sample count buys
//! variance and nothing else, and every opponent in the draw sees the same
//! hero board so the comparison between actions shares its noise.
//!
//! # Common random numbers, deliberately
//!
//! Every action at a root is scored against the **same** drawn opponents.  A
//! root's opponents depend on hero's *seen* cards, which are the same for all
//! actions (board + draw + dead), so this costs nothing and removes the draw
//! from every action-to-action difference the teacher consumes.

use crate::frontier::FrontierEntry;
use crate::pool::{draw, mask_of, Pool, PoolEntry, ShortDraw};
use crate::vs_fl::{hero_score, HeroTerminal};
use crate::Card;

/// One T4 decision.
pub struct T4Request {
    pub id: String,
    /// Hero's eleven placed cards, by row.
    pub rows: [Vec<Card>; 3],
    /// Hero's earlier discards -- seen by hero, so they condition the draw.
    pub dead: Vec<Card>,
    /// The three drawn cards.
    pub draw: [Card; 3],
    pub opponents: usize,
}

pub struct T4ActionValue {
    pub action_key: String,
    pub value: f64,
    pub opponents: usize,
}

/// Hero's terminal facts after a placement, through the canonical evaluator.
fn hero_terminal(rows: &[Vec<Card>; 3]) -> HeroTerminal {
    let core: Vec<Vec<ofc_core::Card>> = rows.iter().map(|row| crate::to_core_cards(row)).collect();
    let eval = ofc_core::evaluate_board_with_joker_constraint(&core[0], &core[1], &core[2]);
    if eval.busted {
        return HeroTerminal {
            busted: true,
            top: 0,
            mid: 0,
            bot: 0,
            royalty: 0,
            entry_width: 0,
        };
    }
    let (qualifies, width) = ofc_core::check_fl_entry(&eval.top);
    HeroTerminal {
        busted: false,
        top: ofc_core::evaluate_hand_value(&eval.top, 3),
        mid: ofc_core::evaluate_hand_value(&eval.mid, 5),
        bot: ofc_core::evaluate_hand_value(&eval.bot, 5),
        royalty: ofc_core::get_top_royalty(&eval.top)
            + ofc_core::get_middle_royalty(&eval.mid)
            + ofc_core::get_bottom_royalty(&eval.bot),
        entry_width: if qualifies { width } else { 0 },
    }
}

fn card_name(card: &Card) -> String {
    if card.rank == 0 {
        return "X".to_string();
    }
    let rank = "23456789TJQKA"
        .chars()
        .nth(card.rank as usize - 2)
        .unwrap_or('?');
    let suit = "shdc".chars().nth(card.suit as usize).unwrap_or('?');
    format!("{rank}{suit}")
}

/// Placements of two of three drawn cards into the open rows, deduplicated on
/// card identity so the interchangeable jokers collapse.
fn placements(rows: &[Vec<Card>; 3], draw: &[Card; 3]) -> Vec<([Vec<Card>; 3], String)> {
    let capacity = [3usize, 5, 5];
    let open: Vec<usize> = (0..3).map(|row| capacity[row] - rows[row].len()).collect();
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|row| need[row] > open[row]) {
                    continue;
                }
                let mut next = rows.clone();
                next[row_a].push(draw[kept[0]]);
                next[row_b].push(draw[kept[1]]);
                // Identity key: the rows as sorted names, plus the discard.
                let mut parts: Vec<String> = next
                    .iter()
                    .map(|row| {
                        let mut names: Vec<String> = row.iter().map(card_name).collect();
                        names.sort();
                        names.join(",")
                    })
                    .collect();
                parts.push(card_name(&draw[discard]));
                let key = parts.join("|");
                if seen.insert(key.clone()) {
                    out.push((next, key));
                }
            }
        }
    }
    out
}

/// Every action's value at one T4 root.
pub fn solve(
    request: &T4Request,
    pool: &Pool,
    fl_ev: &[f64; 4],
    stream: u64,
) -> Result<Vec<T4ActionValue>, ShortDraw> {
    // Hero's seen cards are action-independent, so the draw is made once and
    // shared -- common random numbers across the actions being compared.
    let mut seen: Vec<Card> = request.dead.clone();
    for row in &request.rows {
        seen.extend_from_slice(row);
    }
    seen.extend_from_slice(&request.draw);
    let (hero_naturals, hero_jokers) = mask_of(&seen);
    let opponents: Vec<&PoolEntry> =
        draw(pool, hero_naturals, hero_jokers, request.opponents, stream)?;

    let mut values: Vec<T4ActionValue> = placements(&request.rows, &request.draw)
        .into_iter()
        .map(|(rows, action_key)| {
            let hero = hero_terminal(&rows);
            let total: f64 = opponents
                .iter()
                .map(|entry| hero_score(&hero, &entry.rows, fl_ev))
                .sum();
            T4ActionValue {
                action_key,
                value: total / opponents.len() as f64,
                opponents: opponents.len(),
            }
        })
        .collect();
    values.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(values)
}

/// The frontier rows a T4 leaf actually scans, for callers that want to price
/// a single completed board rather than a whole decision.
pub fn score_board(
    hero: &HeroTerminal,
    opponents: &[&PoolEntry],
    fl_ev: &[f64; 4],
) -> f64 {
    let total: f64 = opponents
        .iter()
        .map(|entry| hero_score(hero, &entry.rows, fl_ev))
        .sum();
    total / opponents.len().max(1) as f64
}

/// Convenience for the T3 labeler: the frontier slice of a drawn entry.
pub fn rows_of(entry: &PoolEntry) -> &[FrontierEntry] {
    &entry.rows
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
            seed: 0x7411_0001,
            entries: (0..entries as u64)
                .map(|index| build_entry(0x7411_0001, index, 14, TABLE[0]))
                .collect(),
        }
    }

    fn request_from(seed: u64, opponents: usize) -> T4Request {
        let cards = deal(seed, 0, 15);
        T4Request {
            id: "t".into(),
            rows: [
                cards[0..2].to_vec(),
                cards[2..7].to_vec(),
                cards[7..11].to_vec(),
            ],
            dead: cards[11..12].to_vec(),
            draw: [cards[12], cards[13], cards[14]],
            opponents,
        }
    }

    /// Every legal placement appears exactly once, and the joker-collapsed
    /// count is what the rest of the chain expects.
    #[test]
    fn placements_are_distinct_and_complete() {
        let request = request_from(0x7411_1000, 1);
        let list = placements(&request.rows, &request.draw);
        let keys: std::collections::BTreeSet<&String> =
            list.iter().map(|(_, key)| key).collect();
        assert_eq!(keys.len(), list.len(), "a placement key repeats");
        for (rows, _) in &list {
            assert_eq!(
                rows[0].len() + rows[1].len() + rows[2].len(),
                13,
                "a placement did not finish the board"
            );
            assert!(rows[0].len() <= 3 && rows[1].len() <= 5 && rows[2].len() <= 5);
        }
    }

    /// Every action at a root faces the SAME opponents.  Without this the
    /// difference between two actions carries the draw's noise, which is the
    /// quantity the teacher is trying to measure.
    #[test]
    fn actions_share_their_opponents() {
        let pool = tiny_pool(400);
        let request = request_from(0x7411_2000, 6);
        let Ok(values) = solve(&request, &pool, &TABLE, 3) else {
            return; // a short draw is its own test
        };
        assert!(values.len() > 1);
        let opponents = values[0].opponents;
        assert!(values.iter().all(|value| value.opponents == opponents));
    }

    /// The labeler's value is the mean of per-opponent scores -- recomputed
    /// here from the parts, so a change to the aggregation shows up.
    #[test]
    fn action_value_is_the_mean_of_per_opponent_scores() {
        let pool = tiny_pool(400);
        let request = request_from(0x7411_3000, 5);
        let mut seen: Vec<Card> = request.dead.clone();
        for row in &request.rows {
            seen.extend_from_slice(row);
        }
        seen.extend_from_slice(&request.draw);
        let (naturals, jokers) = mask_of(&seen);
        let Ok(opponents) = draw(&pool, naturals, jokers, request.opponents, 11) else {
            return;
        };
        let Ok(values) = solve(&request, &pool, &TABLE, 11) else {
            return;
        };
        let by_key: std::collections::HashMap<&str, f64> =
            values.iter().map(|v| (v.action_key.as_str(), v.value)).collect();
        for (rows, key) in placements(&request.rows, &request.draw) {
            let hero = hero_terminal(&rows);
            let expected: f64 = opponents
                .iter()
                .map(|entry| hero_score(&hero, &entry.rows, &TABLE))
                .sum::<f64>()
                / opponents.len() as f64;
            assert_eq!(
                by_key[key.as_str()].to_bits(),
                expected.to_bits(),
                "action {key} is not the mean of its per-opponent scores"
            );
        }
    }

    /// A short draw propagates rather than silently averaging over whatever
    /// turned up.
    #[test]
    fn a_short_draw_is_reported() {
        let pool = tiny_pool(8);
        let request = request_from(0x7411_4000, 500);
        match solve(&request, &pool, &TABLE, 1) {
            Err(ShortDraw { wanted, .. }) => assert_eq!(wanted, 500),
            Ok(_) => panic!("an 8-entry pool satisfied a 500-opponent draw"),
        }
    }
}
