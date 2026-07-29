//! Feature encoder and dense-net inference for the T4 first-seat evaluator.
//!
//! Semantics are pinned to `ai/tutor/t4_first_features.py` and
//! `ai/tutor/train_t4_first_evaluator.py`; the parity test drives this module
//! against those.  Only the hero and joint blocks are computed here per action
//! -- the opponent and context blocks are node-shared and supplied by the
//! caller, which is what keeps the per-action cost to a few microseconds.

use anyhow::{anyhow, bail, Result};
use ofc_core::{
    check_fl_entry, evaluate_board_with_joker_constraint, evaluate_hand_value,
    get_bottom_royalty, get_middle_royalty, get_top_royalty, Card,
};

pub const CATEGORIES: usize = 9;
pub const HERO_SIZE: usize = 42;
pub const OPPONENT_SIZE: usize = 49;
pub const JOINT_SIZE: usize = 12;
pub const CONTEXT_SIZE: usize = 6;
pub const FEATURE_SIZE: usize = HERO_SIZE + OPPONENT_SIZE + JOINT_SIZE + CONTEXT_SIZE;

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];
const MAX_ROYALTY: f32 = 25.0;
const MAX_FL_EV: f32 = 63.5;
const B: u32 = 15;

/// Category index of an encoded hand value, matching `hand_category`.
fn category_of(value: u32) -> usize {
    (value / B.pow(5)) as usize
}

fn row_royalty(row: usize, cards: &[Card]) -> i32 {
    match row {
        0 => get_top_royalty(cards),
        1 => get_middle_royalty(cards),
        _ => get_bottom_royalty(cards),
    }
}

/// Category one-hot plus the two leading rank tiebreaks.
fn spread(value: u32, out: &mut Vec<f32>) {
    let category = category_of(value).min(CATEGORIES - 1);
    let start = out.len();
    out.resize(start + CATEGORIES + 2, 0.0);
    out[start + category] = 1.0;
    out[start + CATEGORIES] = ((value / B.pow(4)) % B) as f32 / 14.0;
    out[start + CATEGORIES + 1] = ((value / B.pow(3)) % B) as f32 / 14.0;
}

/// Category of an incomplete row by rank multiplicity, jokers counted wild.
/// An incomplete row cannot hold a straight or flush yet, so this is exact
/// rather than approximate, and a sound lower bound on the finished row.
pub fn partial_category(cards: &[Card], capacity: usize) -> usize {
    if cards.is_empty() {
        return 0;
    }
    if cards.len() == capacity {
        return category_of(evaluate_hand_value(cards, capacity)).min(CATEGORIES - 1);
    }
    let mut counts = [0u8; 15];
    let mut jokers = 0u8;
    for card in cards {
        if card.is_joker() {
            jokers += 1;
        } else {
            counts[card.rank as usize] += 1;
        }
    }
    let best = counts.iter().copied().max().unwrap_or(0) + jokers;
    let pairs = counts.iter().filter(|count| **count >= 2).count();
    if best >= 4 {
        7
    } else if best == 3 && pairs >= 2 {
        6
    } else if best >= 3 {
        3
    } else if pairs >= 2 {
        2
    } else if best == 2 {
        1
    } else {
        0
    }
}

/// Fantasyland EV by card count 14..17, from ai/config/fl_ev.json.
pub type FlTable = [f32; 4];

fn fl_ev_for(table: &FlTable, card_count: u8) -> f32 {
    match card_count {
        14..=17 => table[(card_count - 14) as usize],
        _ => 0.0,
    }
}

/// Exact terminal facts of the completed hero board (42 dims).
pub fn hero_block(rows: &[Vec<Card>; 3], fl_table: &FlTable, out: &mut Vec<f32>) {
    let eval = evaluate_board_with_joker_constraint(&rows[0], &rows[1], &rows[2]);
    let busted = eval.busted;
    let final_rows = [eval.top, eval.mid, eval.bot];
    let royalties: [i32; 3] = if busted {
        [0, 0, 0]
    } else {
        [
            row_royalty(0, &final_rows[0]),
            row_royalty(1, &final_rows[1]),
            row_royalty(2, &final_rows[2]),
        ]
    };
    let (fl_qualified, fl_count) = if busted {
        (false, 0u8)
    } else {
        check_fl_entry(&final_rows[0])
    };
    let fl_ev = if fl_qualified {
        fl_ev_for(fl_table, fl_count)
    } else {
        0.0
    };
    out.push(if busted { 1.0 } else { 0.0 });
    out.push(royalties.iter().sum::<i32>() as f32 / MAX_ROYALTY);
    for index in 0..3 {
        out.push(royalties[index] as f32 / MAX_ROYALTY);
    }
    out.push(if fl_qualified { 1.0 } else { 0.0 });
    out.push(fl_count as f32 / 17.0);
    out.push(fl_ev / MAX_FL_EV);
    for index in 0..3 {
        spread(
            evaluate_hand_value(&final_rows[index], ROW_CAPACITY[index]),
            out,
        );
    }
    let jokers = rows.iter().flatten().filter(|card| card.is_joker()).count();
    out.push(jokers as f32 / 2.0);
}

/// Facts the per-row histograms cannot express (12 dims).
pub fn joint_block(
    hero_rows: &[Vec<Card>; 3],
    opponent_rows: &[Vec<Card>; 3],
    opponent_categories: &[usize; 3],
    out: &mut Vec<f32>,
) {
    let rooms: [usize; 3] = [
        ROW_CAPACITY[0] - opponent_rows[0].len(),
        ROW_CAPACITY[1] - opponent_rows[1].len(),
        ROW_CAPACITY[2] - opponent_rows[2].len(),
    ];
    let eval = evaluate_board_with_joker_constraint(&hero_rows[0], &hero_rows[1], &hero_rows[2]);
    let hero_final = [eval.top, eval.mid, eval.bot];
    let hero_values: [u32; 3] = [
        evaluate_hand_value(&hero_final[0], 3),
        evaluate_hand_value(&hero_final[1], 5),
        evaluate_hand_value(&hero_final[2], 5),
    ];

    let locked_middle = rooms[2] == 0 && opponent_categories[1] > opponent_categories[2];
    let locked_top = rooms[1] == 0 && opponent_categories[0] > opponent_categories[1];
    out.push(if locked_middle { 1.0 } else { 0.0 });
    out.push(if locked_top { 1.0 } else { 0.0 });
    out.push(if locked_middle || locked_top { 1.0 } else { 0.0 });
    out.push((opponent_categories[1] as f32 - opponent_categories[2] as f32) / 8.0);
    out.push((opponent_categories[0] as f32 - opponent_categories[1] as f32) / 8.0);

    let mut wins = 0;
    for index in 0..3 {
        let sign = if rooms[index] == 0 {
            let opponent_value =
                evaluate_hand_value(&opponent_rows[index], ROW_CAPACITY[index]);
            (hero_values[index] > opponent_value) as i32
                - (hero_values[index] < opponent_value) as i32
        } else {
            let hero_category = category_of(hero_values[index]);
            (hero_category > opponent_categories[index]) as i32
                - (hero_category < opponent_categories[index]) as i32
        };
        out.push(sign as f32);
        if sign > 0 {
            wins += 1;
        }
    }
    out.push(wins as f32 / 3.0);
    out.push(if wins == 3 { 1.0 } else { 0.0 });
    out.push(if eval.busted { 1.0 } else { 0.0 });
    out.push(rooms.iter().sum::<usize>() as f32 / 5.0);
}

// ------------------------------------------------------------------
// Dense net
// ------------------------------------------------------------------

struct Layer {
    inputs: usize,
    outputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

pub struct Model {
    pub input_dim: usize,
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
    layers: Vec<Layer>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32> {
        if self.offset + 4 > self.bytes.len() {
            bail!("model image is truncated");
        }
        let value = u32::from_le_bytes(
            self.bytes[self.offset..self.offset + 4]
                .try_into()
                .map_err(|_| anyhow!("bad u32"))?,
        );
        self.offset += 4;
        Ok(value)
    }

    fn floats(&mut self, count: usize) -> Result<Vec<f32>> {
        if self.offset + count * 4 > self.bytes.len() {
            bail!("model image is truncated");
        }
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let start = self.offset + index * 4;
            out.push(f32::from_le_bytes(
                self.bytes[start..start + 4]
                    .try_into()
                    .map_err(|_| anyhow!("bad f32"))?,
            ));
        }
        self.offset += count * 4;
        Ok(out)
    }
}

impl Model {
    pub fn load(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || &bytes[..4] != b"T4F1" {
            bail!("model image does not start with the expected magic");
        }
        let mut reader = Reader { bytes, offset: 4 };
        if reader.u32()? != 1 {
            bail!("unsupported model image version");
        }
        let layer_count = reader.u32()? as usize;
        let input_dim = reader.u32()? as usize;
        let mean = reader.floats(input_dim)?;
        let std = reader.floats(input_dim)?;
        if std.iter().any(|value| !value.is_finite() || *value == 0.0) {
            bail!("model image has a zero or non-finite standard deviation");
        }
        let inverse_std = std.iter().map(|value| 1.0 / value).collect();
        let mut layers = Vec::with_capacity(layer_count);
        let mut expected = input_dim;
        for index in 0..layer_count {
            let inputs = reader.u32()? as usize;
            let outputs = reader.u32()? as usize;
            if inputs != expected {
                bail!("layer {index} expects {inputs} inputs, previous stage gives {expected}");
            }
            let weight = reader.floats(inputs * outputs)?;
            let bias = reader.floats(outputs)?;
            layers.push(Layer {
                inputs,
                outputs,
                weight,
                bias,
            });
            expected = outputs;
        }
        if expected != 1 {
            bail!("model image must end in a single output");
        }
        Ok(Self {
            input_dim,
            mean,
            inverse_std,
            layers,
        })
    }

    /// Standardize then run the stack; ReLU on every layer but the last.
    pub fn predict(&self, features: &[f32], scratch: &mut Vec<f32>) -> f32 {
        scratch.clear();
        for index in 0..self.input_dim {
            scratch.push((features[index] - self.mean[index]) * self.inverse_std[index]);
        }
        let mut current = std::mem::take(scratch);
        let mut next: Vec<f32> = Vec::new();
        for (position, layer) in self.layers.iter().enumerate() {
            next.clear();
            next.reserve(layer.outputs);
            for output in 0..layer.outputs {
                let row = &layer.weight[output * layer.inputs..(output + 1) * layer.inputs];
                let mut sum = layer.bias[output];
                for input in 0..layer.inputs {
                    sum += row[input] * current[input];
                }
                if position + 1 < self.layers.len() {
                    sum = sum.max(0.0);
                }
                next.push(sum);
            }
            std::mem::swap(&mut current, &mut next);
        }
        let value = current[0];
        *scratch = current;
        value
    }
}
