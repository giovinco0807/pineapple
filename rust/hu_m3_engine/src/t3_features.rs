//! Feature encoding for a learned T3 second-seat evaluator.
//!
//! Like [`crate::t4_features`] this only describes positions; nothing in
//! [`crate::search`] calls it.
//!
//! T4 let one side be described exactly: once the action was chosen the hero's
//! board was final and could simply be scored. Nothing here is. Both players sit
//! on eleven cards with two slots left, so every row is a partial hand with a
//! draw attached and the two sides have to be described symmetrically.
//!
//! The enumeration that made T4 exact still applies, and now to both sides:
//! thirteen cards is where everyone finishes, so every open slot must be filled
//! and there is no choice about which row each card goes to. That yields, per
//! side, the distribution over each row's final category, the chance of fouling
//! despite best play, and the Fantasy Land rate -- which the category histogram
//! cannot express on its own because it drops the rank that decides it.
//!
//! The two sides are then compared as finished boards rather than as
//! histograms. A histogram files two one-pair middles in the same bin and calls
//! the line a tie when the rank decides it, the same blindness that produced the
//! worst errors the T4 model made. Both sides draw from one pool so their
//! completions are not independent; each is enumerated against the full unknown
//! set, which is the approximation this makes deliberately.

use crate::cards::Card;
use crate::infoset::ActorObservation;
use crate::scoring::{
    bottom_royalty_from_value, fl_entry_from_top_value, middle_royalty_from_value,
    top_royalty_from_value, HandValue,
};
use crate::state::{Board, Row};
use crate::t4_features::partial_value;

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];
const CATEGORIES: usize = 9;

const PER_ROW: usize = 9;
const PER_SIDE: usize = 3 * PER_ROW + 4;
const COMPARE: usize = 15;
const JOINT: usize = 5;
const CONTEXT: usize = 4;
/// Structural block: both sides' rows, their comparison, and the joint shape.
pub const STRUCTURAL_SIZE: usize = 2 * PER_SIDE + COMPARE + JOINT + CONTEXT;

/// One side's exact outlook: histograms, room, forced foul, slack, FL, royalty.
pub const SIDE_OUTLOOK_SIZE: usize = 3 * CATEGORIES + 3 + 2 + 2 + 2;
/// Row win rates, the scoop conditions, and the royalty and Fantasy Land edges.
pub const HEAD_TO_HEAD_SIZE: usize = 10;
pub const FEATURE_SIZE: usize =
    STRUCTURAL_SIZE + 2 * SIDE_OUTLOOK_SIZE + HEAD_TO_HEAD_SIZE;

/// How many finishes each side contributes to the comparison.
const SAMPLE: usize = 24;

fn rows_of(board: &Board) -> [&[Card]; 3] {
    [
        board.cards(Row::Top),
        board.cards(Row::Middle),
        board.cards(Row::Bottom),
    ]
}

fn head(value: &HandValue) -> f32 {
    value.1.first().copied().unwrap_or(0) as f32 / 14.0
}

fn spread(value: &HandValue, out: &mut [f32]) {
    out[0] = value.0 as f32 / 8.0;
    for index in 0..5 {
        out[1 + index] = value.1.get(index).map_or(0.0, |rank| *rank as f32 / 14.0);
    }
}

/// One row: what it holds now, and what its remaining slots still allow.
fn row_block(cards: &[Card], index: usize, out: &mut [f32]) -> (HandValue, usize) {
    let capacity = ROW_CAPACITY[index];
    let value = partial_value(cards, capacity);
    let room = capacity - cards.len();
    spread(&value, &mut out[..6]);
    out[6] = room as f32 / 5.0;

    let mut suits = [0u8; 4];
    for card in cards {
        suits[card.index() / 13] += 1;
    }
    out[7] = *suits.iter().max().unwrap_or(&0) as f32 / 5.0;

    let mut ranks: Vec<u8> = cards.iter().map(|card| card.rank()).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let span = if ranks.len() > 1 {
        ranks[ranks.len() - 1] - ranks[0]
    } else {
        0
    };
    out[8] = if capacity == 5 && span <= 4 && ranks.len() >= 3 {
        1.0
    } else {
        0.0
    };
    (value, room)
}

fn side(board: &Board, out: &mut [f32]) -> [HandValue; 3] {
    let rows = rows_of(board);
    let mut values = Vec::with_capacity(3);
    let mut room = [0usize; 3];
    for index in 0..3 {
        let (value, slots) = row_block(rows[index], index, &mut out[index * PER_ROW..]);
        values.push(value);
        room[index] = slots;
    }

    let base = 3 * PER_ROW;
    // Fouling as an ordering between this side's own rows. A full row can no
    // longer move, so a lower row already losing to the one above it is a foul
    // the remaining cards cannot undo.
    out[base] = (values[1].0 as f32 - values[2].0 as f32) / 8.0;
    out[base + 1] = (values[0].0 as f32 - values[1].0 as f32) / 8.0;
    let locked = (room[2] == 0 && values[1].0 > values[2].0)
        || (room[1] == 0 && values[0].0 > values[1].0);
    out[base + 2] = if locked { 1.0 } else { 0.0 };

    // Fantasy Land turns on a pair of queens or better up top, which a category
    // alone cannot express.
    let top = &values[0];
    out[base + 3] = if top.0 >= 2 {
        1.0
    } else if top.0 == 1 && top.1.first().is_some_and(|rank| *rank >= 12) {
        1.0
    } else if room[0] > 0 {
        0.5
    } else {
        0.0
    };
    [values[0].clone(), values[1].clone(), values[2].clone()]
}

/// The structural block, which needs no enumeration.
pub fn encode_structural(
    observation: &ActorObservation,
    hero_board: &Board,
) -> [f32; STRUCTURAL_SIZE] {
    let mut out = [0.0f32; STRUCTURAL_SIZE];
    let hero_values = side(hero_board, &mut out[..PER_SIDE]);
    let opponent = &observation.opponent_public_board;
    let opponent_values = side(opponent, &mut out[PER_SIDE..2 * PER_SIDE]);

    let hero_rows = rows_of(hero_board);
    let opponent_rows = rows_of(opponent);

    let mut base = 2 * PER_SIDE;
    let mut won = 0;
    for row in 0..3 {
        let (left, right) = (&hero_values[row], &opponent_values[row]);
        let sign = match left.cmp(right) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        };
        if sign > 0.0 {
            won += 1;
        }
        let slot = base + row * 5;
        out[slot] = sign;
        out[slot + 1] = (left.0 as f32 - right.0 as f32) / 8.0;
        out[slot + 2] = head(left) - head(right);
        // A lead is only as safe as the slots the other side has left to answer
        // with, so the comparison is paired with how much can still change.
        out[slot + 3] = (ROW_CAPACITY[row] - opponent_rows[row].len()) as f32 / 5.0;
        out[slot + 4] = (ROW_CAPACITY[row] - hero_rows[row].len()) as f32 / 5.0;
    }

    base += COMPARE;
    out[base] = won as f32 / 3.0;
    out[base + 1] = if won == 3 { 1.0 } else { 0.0 };
    out[base + 2] = if won == 0 { 1.0 } else { 0.0 };
    out[base + 3] = (0..3)
        .map(|row| ROW_CAPACITY[row] - hero_rows[row].len())
        .sum::<usize>() as f32;
    out[base + 4] = (0..3)
        .map(|row| ROW_CAPACITY[row] - opponent_rows[row].len())
        .sum::<usize>() as f32;

    base += JOINT;
    out[base] = observation.hero_private_discards.len() as f32 / 3.0;
    let ranks = observation.dealt_cards.iter().map(|card| card.rank());
    out[base + 1] = ranks.clone().max().unwrap_or(0) as f32 / 14.0;
    out[base + 2] = ranks.min().unwrap_or(0) as f32 / 14.0;
    out[base + 3] = observation.dealt_cards.len() as f32 / 3.0;
    out
}

/// One way a row can finish: its value, what it pays, and the cards it spent.
struct RowFinish {
    value: HandValue,
    royalty: f32,
    used: [Option<Card>; 2],
    fantasyland: bool,
}

/// A finished legal board.
pub struct Finish {
    values: [HandValue; 3],
    royalty: f32,
    fantasyland: bool,
}

impl Finish {
    /// Used by the first-seat encoder, whose sampled joint block builds
    /// finishes of its own but compares them through the same head-to-head.
    pub fn new(values: [HandValue; 3], royalty: f32, fantasyland: bool) -> Self {
        Self {
            values,
            royalty,
            fantasyland,
        }
    }
}

fn royalty_of(row: usize, value: &HandValue) -> f32 {
    match row {
        0 => top_royalty_from_value(value) as f32,
        1 => middle_royalty_from_value(value) as f32,
        _ => bottom_royalty_from_value(value) as f32,
    }
}

fn completions(cards: &[Card], row: usize, unknown: &[Card]) -> Vec<RowFinish> {
    let capacity = ROW_CAPACITY[row];
    let room = capacity - cards.len();
    let mut buffer = Vec::with_capacity(capacity);

    let mut finish = |extra: &[Card]| -> RowFinish {
        buffer.clear();
        buffer.extend_from_slice(cards);
        buffer.extend_from_slice(extra);
        let value = partial_value(&buffer, capacity);
        let complete = buffer.len() == capacity;
        let mut used = [None, None];
        for (slot, card) in extra.iter().enumerate() {
            used[slot] = Some(*card);
        }
        RowFinish {
            royalty: if complete { royalty_of(row, &value) } else { 0.0 },
            fantasyland: row == 0 && complete && fl_entry_from_top_value(&value).qualifies,
            value,
            used,
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

fn overlaps(left: &RowFinish, right: &RowFinish) -> bool {
    left.used
        .iter()
        .flatten()
        .any(|card| right.used.iter().flatten().any(|other| other == card))
}

/// Exact outlook over one board's two remaining cards.
pub fn side_outlook(
    board: &Board,
    unknown: &[Card],
) -> ([f32; SIDE_OUTLOOK_SIZE], Vec<Finish>) {
    let mut out = [0.0f32; SIDE_OUTLOOK_SIZE];
    let rows = rows_of(board);
    let filled: Vec<Vec<RowFinish>> = (0..3)
        .map(|row| completions(rows[row], row, unknown))
        .collect();

    for row in 0..3 {
        let mut counts = [0.0f64; CATEGORIES];
        for entry in &filled[row] {
            counts[(entry.value.0 as usize).min(CATEGORIES - 1)] += 1.0;
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

    let open: Vec<usize> = (0..3).filter(|row| room[*row] > 0).collect();
    let mut legal: Vec<Finish> = Vec::new();
    let mut total = 0usize;
    let mut fantasyland = 0usize;
    let mut royalty_sum = 0.0f64;

    {
        let mut consider = |triple: [&RowFinish; 3]| {
            total += 1;
            if !(triple[0].value <= triple[1].value && triple[1].value <= triple[2].value) {
                return;
            }
            let royalty = triple[0].royalty + triple[1].royalty + triple[2].royalty;
            royalty_sum += royalty as f64;
            if triple[0].fantasyland {
                fantasyland += 1;
            }
            legal.push(Finish {
                values: [
                    triple[0].value.clone(),
                    triple[1].value.clone(),
                    triple[2].value.clone(),
                ],
                royalty,
                fantasyland: triple[0].fantasyland,
            });
        };

        let fixed = [&filled[0][0], &filled[1][0], &filled[2][0]];
        match open.len() {
            0 => consider(fixed),
            1 => {
                let only = open[0];
                for candidate in &filled[only] {
                    let mut triple = fixed;
                    triple[only] = candidate;
                    consider(triple);
                }
            }
            _ => {
                let (first, second) = (open[0], open[1]);
                for left in &filled[first] {
                    for right in &filled[second] {
                        if overlaps(left, right) {
                            continue;
                        }
                        let mut triple = fixed;
                        triple[first] = left;
                        triple[second] = right;
                        consider(triple);
                    }
                }
            }
        }
    }

    out[base] = (total - legal.len()) as f32 / total.max(1) as f32;
    out[base + 1] = if total > 0 && legal.is_empty() { 1.0 } else { 0.0 };
    base += 2;

    let lowest = |row: usize| filled[row].iter().map(|e| e.value.0).min().unwrap_or(0);
    let highest = |row: usize| filled[row].iter().map(|e| e.value.0).max().unwrap_or(0);
    out[base] = (lowest(1) as f32 - highest(2) as f32) / 8.0;
    out[base + 1] = (lowest(0) as f32 - highest(1) as f32) / 8.0;
    base += 2;

    let denominator = legal.len().max(1) as f32;
    out[base] = fantasyland as f32 / denominator;
    out[base + 1] = (royalty_sum as f32 / denominator) / 10.0;
    (out, legal)
}

/// Evenly spaced sample, so the features are reproducible without an RNG.
fn stride(items: &[Finish], count: usize) -> Vec<&Finish> {
    if items.len() <= count {
        return items.iter().collect();
    }
    let step = items.len() as f64 / count as f64;
    (0..count)
        .map(|index| &items[(index as f64 * step) as usize])
        .collect()
}

/// Compare finished boards against finished boards.
pub fn head_to_head(hero: &[Finish], opponent: &[Finish]) -> [f32; HEAD_TO_HEAD_SIZE] {
    let mut out = [0.0f32; HEAD_TO_HEAD_SIZE];
    if hero.is_empty() || opponent.is_empty() {
        // An empty list means every completion fouls, which is itself the signal.
        out[6] = if hero.is_empty() { 1.0 } else { 0.0 };
        out[7] = if opponent.is_empty() { 1.0 } else { 0.0 };
        return out;
    }

    let ours = stride(hero, SAMPLE);
    let theirs = stride(opponent, SAMPLE);
    let mut wins = [0.0f64; 3];
    let (mut scoop, mut scooped, mut lines) = (0.0f64, 0.0f64, 0.0f64);
    let (mut royalty, mut fantasyland) = (0.0f64, 0.0f64);
    let mut pairs = 0usize;

    for mine in &ours {
        for other in &theirs {
            pairs += 1;
            let mut won = 0;
            for row in 0..3 {
                if mine.values[row] > other.values[row] {
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
            royalty += (mine.royalty - other.royalty) as f64;
            fantasyland += (if mine.fantasyland { 1.0 } else { 0.0 })
                - (if other.fantasyland { 1.0 } else { 0.0 });
        }
    }

    let count = pairs as f64;
    for row in 0..3 {
        out[row] = (wins[row] / count) as f32;
    }
    out[3] = (scoop / count) as f32;
    out[4] = (scooped / count) as f32;
    out[5] = (lines / count / 3.0) as f32;
    out[8] = ((royalty / count) as f32) / 10.0;
    // Fantasy Land is worth more than ten points and swings on which side gets
    // it, so the difference is carried rather than left to be inferred.
    out[9] = (fantasyland / count) as f32;
    out
}

/// Cards neither the hero nor the opponent has revealed.
///
/// All three dealt cards end up known to the hero -- two placed, one discarded --
/// so this does not depend on which action was taken and one call serves every
/// legal action at the node.
pub fn unknown_cards(observation: &ActorObservation) -> Vec<Card> {
    let mut seen = 0u64;
    for card in observation
        .hero_board
        .all_cards()
        .iter()
        .chain(observation.opponent_public_board.all_cards().iter())
        .chain(observation.hero_private_discards.iter())
        .chain(observation.dealt_cards.iter())
    {
        seen |= 1u64 << card.index();
    }
    (0u8..52)
        .map(Card::from_index_unchecked)
        .filter(|card| seen & (1u64 << card.index()) == 0)
        .collect()
}

/// The full vector for one hero action, given the node's shared opponent block.
pub fn encode(
    observation: &ActorObservation,
    hero_board: &Board,
    unknown: &[Card],
    opponent_outlook: &[f32; SIDE_OUTLOOK_SIZE],
    opponent_finishes: &[Finish],
) -> [f32; FEATURE_SIZE] {
    let mut out = [0.0f32; FEATURE_SIZE];
    out[..STRUCTURAL_SIZE].copy_from_slice(&encode_structural(observation, hero_board));
    let (hero_outlook, hero_finishes) = side_outlook(hero_board, unknown);
    let mut base = STRUCTURAL_SIZE;
    out[base..base + SIDE_OUTLOOK_SIZE].copy_from_slice(&hero_outlook);
    base += SIDE_OUTLOOK_SIZE;
    out[base..base + SIDE_OUTLOOK_SIZE].copy_from_slice(opponent_outlook);
    base += SIDE_OUTLOOK_SIZE;
    out[base..].copy_from_slice(&head_to_head(&hero_finishes, opponent_finishes));
    out
}
