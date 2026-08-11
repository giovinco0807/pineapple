//! Sampled joint completion outlook for boards with more than two open
//! slots -- the block whose absence failed the T2 gate at 0.265 over floor.
//!
//! The exact T3 joint block enumerates every 2-card completion; with 4-8
//! slots that explodes, so this samples K completions of the open slots and,
//! for each, takes the best self-value arrangement (exactly when the
//! arrangement count is small, over a hash-sampled subset otherwise).  The 8
//! summary statistics keep the T3 block's exact semantics: foul rate, mean
//! self-value, survive-conditional royalty and Fantasyland EV, standard
//! deviation, two tail probabilities, and survive-mean.
//!
//! Row evaluations go through the generalized row memo (fast path when the
//! final top and middle are joker-free, the constrained joint evaluation
//! otherwise), so repeated arrangements across samples cost lookups.

use anyhow::Result;
use ofc_core::Card;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::row_memo::TerminalMemo;
use super::{BoardStr, CoreBoard, FlEv};

#[derive(Deserialize)]
pub struct JointOutlookRequest {
    pub id: String,
    /// The actor's partial board (any number of open slots >= 1).
    pub board: BoardStr,
    /// Cards the completion is drawn from (the actor's unseen pool).
    pub pool: Vec<String>,
    /// Completions sampled.
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// Arrangements tried per completion when the exact count is too big.
    #[serde(default = "default_arrangements")]
    pub max_arrangements: usize,
}

fn default_samples() -> usize {
    150
}

fn default_arrangements() -> usize {
    32
}

#[derive(Serialize)]
pub struct JointOutlookResponse {
    pub id: String,
    pub schema: &'static str,
    pub joint_block: [f64; 8],
    /// Per-row completion outlook (41 dims) over the same pool.  Emitted here
    /// so an encoder needs one solver call, not two over the same position --
    /// two calls is two chances for the board and the pool to drift apart.
    pub rowwise_block: Vec<f32>,
    pub samples: usize,
}

/// Deterministic choice of `want` distinct pool indices.
fn sampled_subset(pool_len: usize, want: usize, seed: &str, tick: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..pool_len).collect();
    let mut counter: u64 = 0;
    for position in (1..pool_len).rev() {
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        hasher.update(tick.to_le_bytes());
        hasher.update(counter.to_le_bytes());
        counter += 1;
        let digest = hasher.finalize();
        let value = u64::from_le_bytes(digest[..8].try_into().unwrap());
        indices.swap(position, (value % (position as u64 + 1)) as usize);
    }
    indices.truncate(want);
    indices
}

/// Every distinct way to deal `cards` into the open slots, as row tags per
/// card, deduplicated on card identity; hash-pruned to `cap`.
fn arrangements(open: [usize; 3], cards: &[Card], cap: usize, seed: &str) -> Vec<Vec<usize>> {
    let id = |card: &Card| -> u32 {
        if card.is_joker() {
            52
        } else {
            card.suit as u32 * 13 + card.rank as u32 - 2
        }
    };
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::BTreeSet<Vec<u32>> = std::collections::BTreeSet::new();
    let mut stack = vec![0usize; cards.len()];
    fn recurse(
        slot: usize,
        cards: &[Card],
        open: &[usize; 3],
        counts: &mut [usize; 3],
        stack: &mut Vec<usize>,
        seen: &mut std::collections::BTreeSet<Vec<u32>>,
        out: &mut Vec<Vec<usize>>,
        id: &dyn Fn(&Card) -> u32,
    ) {
        if slot == cards.len() {
            let mut signature: Vec<u32> = (0..cards.len())
                .map(|position| (stack[position] as u32) << 8 | id(&cards[position]))
                .collect();
            signature.sort_unstable();
            if seen.insert(signature) {
                out.push(stack.clone());
            }
            return;
        }
        for row in 0..3 {
            if counts[row] < open[row] {
                counts[row] += 1;
                stack[slot] = row;
                recurse(slot + 1, cards, open, counts, stack, seen, out, id);
                counts[row] -= 1;
            }
        }
    }
    let mut counts = [0usize; 3];
    recurse(0, cards, &open, &mut counts, &mut stack, &mut seen, &mut out, &id);
    prune(out, cap, seed)
}

/// Hash-shuffle then truncate: deterministic arrangement sampling.
fn prune(mut all: Vec<Vec<usize>>, cap: usize, seed: &str) -> Vec<Vec<usize>> {
    if all.len() <= cap {
        return all;
    }
    let mut counter: u64 = 0;
    for position in (1..all.len()).rev() {
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        hasher.update(counter.to_le_bytes());
        counter += 1;
        let digest = hasher.finalize();
        let value = u64::from_le_bytes(digest[..8].try_into().unwrap());
        all.swap(position, (value % (position as u64 + 1)) as usize);
    }
    all.truncate(cap);
    all
}

/// Every `need`-card subset of the pool, iterated without materialization.
struct SubsetIter {
    indices: Vec<usize>,
    pool_len: usize,
    done: bool,
}

impl SubsetIter {
    fn new(pool_len: usize, need: usize) -> Self {
        SubsetIter {
            indices: (0..need).collect(),
            pool_len,
            done: need > pool_len,
        }
    }
}

impl Iterator for SubsetIter {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        let current = self.indices.clone();
        let need = self.indices.len();
        let mut position = need;
        loop {
            if position == 0 {
                self.done = true;
                break;
            }
            position -= 1;
            if self.indices[position] != position + self.pool_len - need {
                self.indices[position] += 1;
                for later in (position + 1)..need {
                    self.indices[later] = self.indices[later - 1] + 1;
                }
                break;
            }
        }
        Some(current)
    }
}

/// samples == 0 enumerates every completion exactly -- affordable up to four
/// open slots (C(43,4) ~ 123k subsets x <= 12 arrangements through the row
/// memo).  Larger rooms must sample: the caller keeps samples > 0 there.
pub fn sampled_joint_block(
    board: &CoreBoard,
    pool: &[Card],
    samples: usize,
    max_arrangements: usize,
    seed: &str,
    fl_ev: &FlEv,
) -> Result<[f64; 8]> {
    let open = board.open_slots();
    let need: usize = open.iter().sum();
    let mut memo = TerminalMemo::new(board);
    let mut best_self: Vec<f64> = Vec::with_capacity(samples.max(1024));
    let (mut fouls, mut survivors) = (0usize, 0usize);
    let (mut survive_royalty, mut survive_fl) = (0.0f64, 0.0f64);
    let mut additions: Vec<(usize, Card)> = Vec::with_capacity(need);

    let subsets: Box<dyn Iterator<Item = Vec<usize>>> = if samples == 0 {
        if need > 4 {
            anyhow::bail!("exact joint outlook is only affordable up to 4 open slots");
        }
        Box::new(SubsetIter::new(pool.len(), need))
    } else {
        let seed_owned = seed.to_string();
        let pool_len = pool.len();
        Box::new(
            (0..samples).map(move |tick| sampled_subset(pool_len, need, &seed_owned, tick as u64)),
        )
    };

    for subset in subsets {
        let cards: Vec<Card> = subset.iter().map(|index| pool[*index]).collect();
        let plans = arrangements(open, &cards, max_arrangements, seed);
        let mut best = f64::NEG_INFINITY;
        let mut parts = (0.0f64, 0.0f64);
        for plan in &plans {
            additions.clear();
            for (slot, row) in plan.iter().enumerate() {
                additions.push((*row, cards[slot]));
            }
            let terminal = memo.terminal_n(&additions)?;
            let value = if terminal.busted {
                -6.0
            } else {
                terminal.royalty as f64 + fl_ev.value(terminal.fl_card_count)
            };
            if value > best {
                best = value;
                parts = if terminal.busted {
                    (0.0, 0.0)
                } else {
                    (
                        terminal.royalty as f64,
                        fl_ev.value(terminal.fl_card_count),
                    )
                };
            }
        }
        if best <= -6.0 {
            fouls += 1;
        } else {
            survivors += 1;
            survive_royalty += parts.0;
            survive_fl += parts.1;
        }
        best_self.push(best);
    }

    let count = best_self.len().max(1) as f64;
    let mean = best_self.iter().sum::<f64>() / count;
    let variance =
        best_self.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count;
    let denominator = survivors.max(1) as f64;
    const MAX_ROYALTY: f64 = 25.0;
    const MAX_FL_EV: f64 = 63.5;
    Ok([
        fouls as f64 / count,
        mean / MAX_ROYALTY,
        (survive_royalty / denominator) / MAX_ROYALTY,
        (survive_fl / denominator) / MAX_FL_EV,
        variance.sqrt() / MAX_ROYALTY,
        best_self.iter().filter(|v| **v >= 6.0).count() as f64 / count,
        best_self.iter().filter(|v| **v >= 15.0).count() as f64 / count,
        (best_self.iter().filter(|v| **v > -6.0).sum::<f64>() / denominator) / MAX_ROYALTY,
    ])
}

pub fn solve(request: &JointOutlookRequest, fl_ev: &FlEv) -> Result<JointOutlookResponse> {
    let board = CoreBoard::from_str_board(&request.board)?;
    let pool: Vec<Card> = request
        .pool
        .iter()
        .map(|name| super::to_core_card(name))
        .collect::<Result<Vec<_>>>()?;
    let block = sampled_joint_block(
        &board,
        &pool,
        request.samples,
        request.max_arrangements,
        &format!("joint/{}", request.id),
        fl_ev,
    )?;
    let mut rowwise: Vec<f32> = Vec::with_capacity(super::evaluator::OPPONENT_SIZE);
    let fl_table: super::evaluator::FlTable = [
        fl_ev.value(14) as f32,
        fl_ev.value(15) as f32,
        fl_ev.value(16) as f32,
        fl_ev.value(17) as f32,
    ];
    let _categories =
        super::evaluator::opponent_rowwise_block(&board.rows, &pool, &fl_table, &mut rowwise);
    Ok(JointOutlookResponse {
        id: request.id.clone(),
        schema: "ofc_sampled_joint_outlook/v2",
        joint_block: block,
        rowwise_block: rowwise,
        samples: request.samples,
    })
}
