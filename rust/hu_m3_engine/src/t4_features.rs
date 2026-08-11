//! Feature encoding for a learned T4-first evaluator.
//!
//! This module only describes positions. It is not wired into [`crate::search`]
//! and changes no existing result, so the exactness the engine reports about T4
//! remains true while this is developed and measured.
//!
//! The encoding exists because a first attempt that fed raw cards reached a
//! held-out correlation of 0.038: separating states is not the same as carrying
//! structure a model can generalise over, and rediscovering hand ranking from
//! card identities is not a reasonable thing to ask. Reading the errors of the
//! version that followed showed the remaining failures were not hard positions
//! but positions whose deciding facts were absent from the input -- an opponent
//! already committed to fouling, a completed opponent flush described as a high
//! card, and a kicker that decided a line but was never encoded. All three are
//! exactly computable, so they are computed here rather than learned.
//!
//! The opponent block is independent of which action the hero takes, so it is
//! produced once per node by [`opponent_outlook`] and shared across every legal
//! action, which is what keeps it affordable relative to an exact solve.

use crate::cards::Card;
use crate::infoset::ActorObservation;
use crate::scoring::{
    bottom_royalty_from_value, evaluate_3_card_trusted, evaluate_5_card_trusted,
    fl_entry_from_top_value, middle_royalty_from_value, top_royalty_from_value, BoardScore,
    HandValue,
};
use crate::state::{Board, Row};

/// Hero block, opponent block, per-row comparison, joint shape, context.
pub const V4_SIZE: usize = 71;
/// Category histograms, room, forced foul, slack, Fantasy Land, royalty.
pub const OUTLOOK_SIZE: usize = 36;
/// Per-row win rates plus the scoop conditions.
pub const OUTCOME_SIZE: usize = 6;
pub const FEATURE_SIZE: usize = V4_SIZE + OUTLOOK_SIZE + OUTCOME_SIZE;

const CATEGORIES: usize = 9;
const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

/// The made hand a row currently holds.
///
/// A completed row is evaluated for real, which is what makes flushes and
/// straights visible. An incomplete row cannot hold either yet, so ranking it
/// by rank multiplicity is exact rather than approximate, and it is a sound
/// lower bound on what the row will finish as: every card already placed counts
/// toward the final hand.
pub fn partial_value(cards: &[Card], capacity: usize) -> HandValue {
    if cards.is_empty() {
        return HandValue(0, Vec::new());
    }
    if cards.len() == capacity {
        return if capacity == 3 {
            evaluate_3_card_trusted(cards)
        } else {
            evaluate_5_card_trusted(cards)
        };
    }
    let mut counts = [0u8; 15];
    for card in cards {
        counts[card.rank() as usize] += 1;
    }
    let mut groups: Vec<(u8, u8)> = (2u8..15)
        .filter(|rank| counts[*rank as usize] > 0)
        .map(|rank| (rank, counts[rank as usize]))
        .collect();
    groups.sort_by(|left, right| right.1.cmp(&left.1).then(right.0.cmp(&left.0)));
    let best = groups[0].1;
    let pairs = groups.iter().filter(|group| group.1 >= 2).count();
    let category = if best >= 4 {
        7
    } else if best == 3 && pairs >= 2 {
        6
    } else if best == 3 {
        3
    } else if pairs >= 2 {
        2
    } else if best == 2 {
        1
    } else {
        0
    };
    HandValue(category, groups.into_iter().map(|group| group.0).collect())
}

fn royalty_of(row: usize, value: &HandValue) -> f32 {
    match row {
        0 => top_royalty_from_value(value) as f32,
        1 => middle_royalty_from_value(value) as f32,
        _ => bottom_royalty_from_value(value) as f32,
    }
}

/// One finished row: its value, what it pays, and which unknown cards it spent.
struct Completion {
    value: HandValue,
    royalty: f32,
    used: [Option<Card>; 2],
    fantasyland: bool,
}

fn used_overlaps(left: &Completion, right: &Completion) -> bool {
    left.used
        .iter()
        .flatten()
        .any(|card| right.used.iter().flatten().any(|other| other == card))
}

fn completions(cards: &[Card], row: usize, unknown: &[Card]) -> Vec<Completion> {
    let capacity = ROW_CAPACITY[row];
    let room = capacity - cards.len();
    let mut buffer = Vec::with_capacity(capacity);

    let mut finish = |extra: &[Card]| -> Completion {
        buffer.clear();
        buffer.extend_from_slice(cards);
        buffer.extend_from_slice(extra);
        let value = partial_value(&buffer, capacity);
        let fantasyland = row == 0 && fl_entry_from_top_value(&value).qualifies;
        let royalty = royalty_of(row, &value);
        let mut used = [None, None];
        for (slot, card) in extra.iter().enumerate() {
            used[slot] = Some(*card);
        }
        Completion {
            value,
            royalty,
            used,
            fantasyland,
        }
    };

    match room {
        0 => vec![finish(&[])],
        1 => unknown.iter().map(|card| finish(&[*card])).collect(),
        _ => {
            let mut out = Vec::with_capacity(unknown.len() * (unknown.len() - 1) / 2);
            for first in 0..unknown.len() {
                for second in first + 1..unknown.len() {
                    out.push(finish(&[unknown[first], unknown[second]]));
                }
            }
            out
        }
    }
}

/// Everything the opponent's two remaining cards can still produce.
///
/// At T4 the opponent finishes on thirteen cards, so every open slot must be
/// filled and there is no choice about which row each card goes to. Completing
/// the rows is therefore exact, and the joint pass only has to reject the pairs
/// that would reuse a card.
///
/// Returns the shared feature block and every legal finished board, which
/// [`hero_outcome`] consumes without re-evaluating anything.
pub fn opponent_outlook(observation: &ActorObservation) -> ([f32; OUTLOOK_SIZE], Vec<[HandValue; 3]>) {
    let mut out = [0.0f32; OUTLOOK_SIZE];
    let opponent = &observation.opponent_public_board;
    let rows = [
        opponent.cards(Row::Top),
        opponent.cards(Row::Middle),
        opponent.cards(Row::Bottom),
    ];

    let mut seen = 0u64;
    for card in observation
        .hero_board
        .all_cards()
        .iter()
        .chain(opponent.all_cards().iter())
        .chain(observation.hero_private_discards.iter())
        .chain(observation.dealt_cards.iter())
    {
        seen |= 1u64 << card.index();
    }
    let unknown: Vec<Card> = (0u8..52)
        .map(Card::from_index_unchecked)
        .filter(|card| seen & (1u64 << card.index()) == 0)
        .collect();

    let filled: Vec<Vec<Completion>> = (0..3)
        .map(|row| completions(rows[row], row, &unknown))
        .collect();

    for row in 0..3 {
        let mut counts = [0.0f64; CATEGORIES];
        for completion in &filled[row] {
            counts[(completion.value.0 as usize).min(CATEGORIES - 1)] += 1.0;
        }
        let total: f64 = counts.iter().sum();
        for (index, count) in counts.iter().enumerate() {
            out[row * CATEGORIES + index] = (count / total.max(1.0)) as f32;
        }
    }

    let mut base = 3 * CATEGORIES;
    let mut room = [0usize; 3];
    for row in 0..3 {
        room[row] = ROW_CAPACITY[row] - rows[row].len();
        out[base + row] = room[row] as f32 / 5.0;
    }
    base += 3;

    // Fouling despite best play: an opponent with a choice takes a legal
    // arrangement whenever one exists, so what matters is the deals where none
    // does.
    let open: Vec<usize> = (0..3).filter(|row| room[*row] > 0).collect();
    let mut legal: Vec<[HandValue; 3]> = Vec::new();
    let mut total = 0usize;
    let mut fantasyland = 0usize;
    let mut royalty = 0.0f64;

    {
        let mut consider = |triple: [&Completion; 3]| {
            total += 1;
            if !(triple[0].value <= triple[1].value && triple[1].value <= triple[2].value) {
                return;
            }
            royalty += (triple[0].royalty + triple[1].royalty + triple[2].royalty) as f64;
            if triple[0].fantasyland {
                fantasyland += 1;
            }
            legal.push([
                triple[0].value.clone(),
                triple[1].value.clone(),
                triple[2].value.clone(),
            ]);
        };

        if open.len() == 1 {
            let only = open[0];
            for candidate in &filled[only] {
                let mut triple = [&filled[0][0], &filled[1][0], &filled[2][0]];
                triple[only] = candidate;
                consider(triple);
            }
        } else {
            let (first, second) = (open[0], open[1]);
            for left in &filled[first] {
                for right in &filled[second] {
                    if used_overlaps(left, right) {
                        continue;
                    }
                    let mut triple = [&filled[0][0], &filled[1][0], &filled[2][0]];
                    triple[first] = left;
                    triple[second] = right;
                    consider(triple);
                }
            }
        }
    }

    out[base] = (total - legal.len()) as f32 / total.max(1) as f32;
    out[base + 1] = if total > 0 && legal.is_empty() { 1.0 } else { 0.0 };
    base += 2;

    let category_min = |row: usize| filled[row].iter().map(|c| c.value.0).min().unwrap_or(0);
    let category_max = |row: usize| filled[row].iter().map(|c| c.value.0).max().unwrap_or(0);
    out[base] = (category_min(1) as f32 - category_max(2) as f32) / 8.0;
    out[base + 1] = (category_min(0) as f32 - category_max(1) as f32) / 8.0;
    base += 2;

    // The category histogram drops ranks, so a pair of deuces and a pair of
    // queens land in the same bin. Fantasy Land turns on exactly that
    // distinction and is worth more than ten points, so it gets its own term.
    let denominator = legal.len().max(1) as f32;
    out[base] = fantasyland as f32 / denominator;
    out[base + 1] = (royalty as f32 / denominator) / 10.0;
    (out, legal)
}

/// How the hero's finished rows fare against every legal opponent finish.
///
/// The rows are not independent -- the scoop pays on all three at once -- so
/// they are counted jointly over the same completions rather than multiplied.
pub fn hero_outcome(hero: &[HandValue; 3], legal: &[[HandValue; 3]]) -> [f32; OUTCOME_SIZE] {
    let mut out = [0.0f32; OUTCOME_SIZE];
    if legal.is_empty() {
        return out;
    }
    let mut wins = [0.0f64; 3];
    let (mut scoop, mut scooped, mut lines) = (0.0f64, 0.0f64, 0.0f64);
    for triple in legal {
        let mut won = 0;
        for row in 0..3 {
            if hero[row] > triple[row] {
                wins[row] += 1.0;
                won += 1;
            }
        }
        lines += won as f64;
        if won == 3 {
            scoop += 1.0;
        } else if won == 0 {
            scooped += 1.0;
        }
    }
    let count = legal.len() as f64;
    for row in 0..3 {
        out[row] = (wins[row] / count) as f32;
    }
    out[3] = (scoop / count) as f32;
    out[4] = (scooped / count) as f32;
    out[5] = (lines / count / 3.0) as f32;
    out
}

fn spread(value: &HandValue, out: &mut [f32]) {
    out[0] = value.0 as f32 / 8.0;
    for index in 0..5 {
        out[1 + index] = value.1.get(index).map_or(0.0, |rank| *rank as f32 / 14.0);
    }
}

fn compare(left: &HandValue, right: &HandValue) -> f32 {
    match left.cmp(right) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Equal => 0.0,
    }
}

/// The action-dependent block: the hero's finished board against what the
/// opponent holds now.
pub fn encode_v4(observation: &ActorObservation, score: &BoardScore) -> [f32; V4_SIZE] {
    let mut out = [0.0f32; V4_SIZE];
    let opponent = &observation.opponent_public_board;
    let rows = [
        opponent.cards(Row::Top),
        opponent.cards(Row::Middle),
        opponent.cards(Row::Bottom),
    ];

    out[0] = if score.busted { 1.0 } else { 0.0 };
    out[1] = score.total_royalty as f32 / 10.0;
    out[2] = score.top_royalty as f32 / 10.0;
    out[3] = score.middle_royalty as f32 / 10.0;
    out[4] = score.bottom_royalty as f32 / 10.0;
    out[5] = if score.fl_entry.qualifies { 1.0 } else { 0.0 };
    out[6] = score.fl_entry.card_count as f32 / 17.0;

    let hero = [
        score.top_value.clone(),
        score.middle_value.clone(),
        score.bottom_value.clone(),
    ];
    for row in 0..3 {
        spread(&hero[row], &mut out[7 + row * 6..13 + row * 6]);
    }

    let opponent_values: Vec<HandValue> = (0..3)
        .map(|row| partial_value(rows[row], ROW_CAPACITY[row]))
        .collect();
    for row in 0..3 {
        spread(&opponent_values[row], &mut out[25 + row * 6..31 + row * 6]);
    }

    let mut room = [0usize; 3];
    for row in 0..3 {
        room[row] = ROW_CAPACITY[row] - rows[row].len();
        out[43 + row] = room[row] as f32 / 5.0;
        let mut suits = [0u8; 4];
        for card in rows[row] {
            suits[card.index() / 13] += 1;
        }
        out[46 + row] = *suits.iter().max().unwrap_or(&0) as f32 / 5.0;
    }

    // A full row can no longer move, so a row below it that already wins fixes
    // a foul regardless of the remaining deal.
    let locked_middle = room[2] == 0 && opponent_values[1].0 > opponent_values[2].0;
    let locked_top = room[1] == 0 && opponent_values[0].0 > opponent_values[1].0;
    out[49] = if locked_middle { 1.0 } else { 0.0 };
    out[50] = if locked_top { 1.0 } else { 0.0 };
    out[51] = out[49].max(out[50]);
    out[52] = (opponent_values[1].0 as f32 - opponent_values[2].0 as f32) / 8.0;

    let mut won = 0;
    for row in 0..3 {
        let sign = compare(&hero[row], &opponent_values[row]);
        if sign > 0.0 {
            won += 1;
        }
        let slot = 53 + row * 4;
        out[slot] = sign;
        out[slot + 1] = (hero[row].0 as f32 - opponent_values[row].0 as f32) / 8.0;
        let head = |value: &HandValue| value.1.first().copied().unwrap_or(0) as f32;
        out[slot + 2] = (head(&hero[row]) - head(&opponent_values[row])) / 14.0;
        out[slot + 3] = if room[row] == 0 { sign } else { 0.0 };
    }

    out[65] = won as f32 / 3.0;
    out[66] = if won == 3 { 1.0 } else { 0.0 };
    out[67] = if won == 0 { 1.0 } else { 0.0 };
    out[68] = room.iter().sum::<usize>() as f32 / 5.0;

    out[69] = observation.hero_private_discards.len() as f32 / 3.0;
    out[70] = observation
        .dealt_cards
        .iter()
        .map(|card| card.rank())
        .max()
        .unwrap_or(0) as f32
        / 14.0;
    out
}

/// The full vector for one hero action, given the node's shared outlook.
pub fn encode(
    observation: &ActorObservation,
    score: &BoardScore,
    outlook: &[f32; OUTLOOK_SIZE],
    legal: &[[HandValue; 3]],
) -> [f32; FEATURE_SIZE] {
    let mut out = [0.0f32; FEATURE_SIZE];
    out[..V4_SIZE].copy_from_slice(&encode_v4(observation, score));
    out[V4_SIZE..V4_SIZE + OUTLOOK_SIZE].copy_from_slice(outlook);
    let hero = [
        score.top_value.clone(),
        score.middle_value.clone(),
        score.bottom_value.clone(),
    ];
    out[V4_SIZE + OUTLOOK_SIZE..].copy_from_slice(&hero_outcome(&hero, legal));
    out
}

/// Convenience for callers that hold a board rather than a score.
pub fn encode_board(
    observation: &ActorObservation,
    board: &Board,
    outlook: &[f32; OUTLOOK_SIZE],
    legal: &[[HandValue; 3]],
) -> [f32; FEATURE_SIZE] {
    let score = crate::scoring::score_board_trusted(board);
    encode(observation, &score, outlook, legal)
}
