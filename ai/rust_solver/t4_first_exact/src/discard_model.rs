//! The trained discard predictor: P(this unseen card is in the opponent's
//! discards | their visible board).
//!
//! A twelve-feature relation vector and at most one hidden layer, so the
//! Rust forward pass is a page and the Python trainer's featureizer can be
//! mirrored exactly.  `ai/tutor/train_discard_model.py` is the source of
//! truth for both the features and the weights; a fixture test pins the two
//! implementations together.

use anyhow::{anyhow, bail, Result};

use super::Card;

pub struct DiscardModel {
    /// Dense layers as (weights row-major [out][in], bias).
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

const FEATURES: usize = 12;

pub fn features(card: &Card, board: &[Vec<Card>; 3]) -> [f32; FEATURES] {
    let mut out = [0.0f32; FEATURES];
    if card.is_joker() {
        out[0] = 0.5;
        out[1] = 1.0;
        return out;
    }
    let rank = card.rank as i32;
    let suit = card.suit;
    let mut same_rank = [0f32; 3];
    let mut same_suit = [0f32; 3];
    let mut near_rank = [0f32; 3];
    for (slot, row) in board.iter().enumerate() {
        for other in row {
            if other.is_joker() {
                continue;
            }
            if other.rank as i32 == rank {
                same_rank[slot] += 1.0;
            }
            if other.suit == suit {
                same_suit[slot] += 1.0;
            }
            let dr = (other.rank as i32 - rank).abs();
            if dr > 0 && dr <= 2 {
                near_rank[slot] += 1.0;
            }
        }
    }
    out[0] = (rank - 2) as f32 / 12.0;
    out[1] = 0.0;
    out[2] = same_rank[0];
    out[3] = same_rank[1];
    out[4] = same_rank[2];
    out[5] = same_suit[1];
    out[6] = same_suit[2];
    out[7] = near_rank[1];
    out[8] = near_rank[2];
    out[9] = same_suit[0] + same_suit[1] + same_suit[2];
    out[10] = if rank <= 6 { 1.0 } else { 0.0 };
    out[11] = if rank >= 11 { 1.0 } else { 0.0 };
    out
}

impl DiscardModel {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let state = parsed
            .get("state")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("discard model has no state"))?;
        // Keys are net.weight/net.bias (linear) or net.0.*, net.2.* (mlp).
        let matrix = |key: &str| -> Option<Vec<Vec<f32>>> {
            state.get(key).and_then(|v| v.as_array()).map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_f64().unwrap() as f32)
                            .collect()
                    })
                    .collect()
            })
        };
        let vector = |key: &str| -> Option<Vec<f32>> {
            state.get(key).and_then(|v| v.as_array()).map(|row| {
                row.iter().map(|x| x.as_f64().unwrap() as f32).collect()
            })
        };
        let mut layers = Vec::new();
        for (w_key, b_key) in [
            ("net.weight", "net.bias"),
            ("net.0.weight", "net.0.bias"),
            ("net.2.weight", "net.2.bias"),
        ] {
            if let (Some(weight), Some(bias)) = (matrix(w_key), vector(b_key)) {
                let flat: Vec<f32> = weight.iter().flatten().copied().collect();
                layers.push((flat, bias));
            }
        }
        if layers.is_empty() {
            bail!("discard model state has no recognised layers");
        }
        Ok(Self { layers })
    }

    /// P(card in the board owner's discards).
    pub fn predict(&self, card: &Card, board: &[Vec<Card>; 3]) -> f32 {
        let mut activations: Vec<f32> = features(card, board).to_vec();
        for (index, (weight, bias)) in self.layers.iter().enumerate() {
            let inputs = activations.len();
            let outputs = bias.len();
            let mut next = vec![0.0f32; outputs];
            for o in 0..outputs {
                let mut sum = bias[o];
                for i in 0..inputs {
                    sum += weight[o * inputs + i] * activations[i];
                }
                next[o] = if index + 1 < self.layers.len() {
                    sum.max(0.0)
                } else {
                    sum
                };
            }
            activations = next;
        }
        1.0 / (1.0 + (-activations[0]).exp())
    }
}
