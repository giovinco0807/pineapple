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
    /// What the completion sample is drawn from; defaults to `id`.
    ///
    /// Every action at a root leaves the SAME unseen set -- hero's discard is
    /// seen either way -- so a caller can hand every action of a root one seed
    /// and have each candidate board judged on the same drawn completions.
    /// Without that the sample differs per action and its noise lands straight
    /// on the within-root ordering, which is the only thing the ranking gate
    /// reads.  Same reason the opponents and the T3 draws are shared in the
    /// labelers; this block was the one sampled quantity still left free.
    #[serde(default)]
    pub seed: Option<String>,
    /// "sha256" (default, the legacy sampler) or "splitmix".  Part of the
    /// encoder contract: two samplers draw different completions from the
    /// same seed, so a model is trained and served under exactly one of them.
    #[serde(default)]
    pub sampler: Option<String>,
}

/// How completion subsets are drawn.  Sha256 is the legacy sampler: a full
/// Fisher-Yates whose every swap is a SHA-256 -- at T2 that is ~95% of the
/// whole sampled block's cost (the exact path evaluates 123k subsets through
/// the row memo in 126 ms, ~1 us each, while 400 sampled subsets cost ~8 ms).
/// SplitMix hashes the seed once and drives a partial Fisher-Yates with
/// splitmix64: same distribution, same determinism, microseconds.  Widths
/// that already shipped keep Sha256 forever; new widths take SplitMix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompletionSampler {
    Sha256,
    SplitMix,
}

impl CompletionSampler {
    pub fn parse(name: Option<&str>) -> Result<Self> {
        match name.unwrap_or("sha256") {
            "sha256" => Ok(CompletionSampler::Sha256),
            "splitmix" => Ok(CompletionSampler::SplitMix),
            other => anyhow::bail!("unknown completion sampler {other:?}"),
        }
    }
}

/// The room a board has before anybody has played to it: three, five, five.
pub const UNPLAYED_ROOM: [usize; 3] = [3, 5, 5];

/// The joint block of a board nobody has played to yet -- eight zeros -- or
/// `None` when the block has to be computed.
///
/// **This is the only place the rule is written down**, and every route to a
/// joint block passes through it, because it sits inside
/// `sampled_joint_block_shared`.  That matters more than it looks: the block
/// is reached from two directions that never meet in code.  Serving goes
/// `hu_match` -> `hu_encode::opponent_tail` -> `board_blocks`; the teachers go
/// `encode_hu_teacher.py` -> the binary's `--joint-outlook` mode -> `solve`.
/// A rule stated once per route is a rule that drifts, and a drifted rule
/// serves a model a vector it never trained on -- which is the failure this
/// project has already paid for elsewhere.
///
/// # What it is for
///
/// The first actor's opening street.  Seat BB acts first, so at T0 its
/// opponent's board is empty and the opponent's half of the 207-dim pair
/// vector is built over thirteen open slots.  Three things are true of that
/// block at once:
///
/// * it is **constant across candidates** -- `hu_match` already hoists it out
///   of the candidate loop precisely because it cannot vary with hero's move,
///   so it cannot contribute to ranking 232 openings;
/// * it is **almost all of the work** -- `arrangements()` enumerates before it
///   prunes, and thirteen slots is 13!/(3!5!5!) = 72,072 arrangements a
///   sample against at most 560 for hero's eight;
/// * it is **not information** -- the opponent has not moved, so there is
///   nothing about the opponent to encode.  The regular track reached the same
///   conclusion and zeroes the same block
///   (`hu_m3_engine/src/fast_features.rs::fast_encode_hidden_opponent`).
///
/// # Why keyed on the room and not on a caller-supplied flag
///
/// A flag would have to be set by both routes, and the teacher route has
/// nowhere honest to set it from: `encode_hu_teacher.py` deduplicates its
/// block requests on `(board, pool)` and drops the own/opponent role before
/// the request is written.  A rule that is a pure function of `(board, pool)`
/// is therefore the only kind guaranteed to agree between training and
/// serving; a role-dependent one would also break that deduplication.
///
/// # Why this does not disturb the Fantasyland encoders
///
/// `sampled_joint_block` is shared with the FL14 teachers
/// (`encode_fl14_teacher.py`, `encode_fl14_t1.py`), the vs-FL regret gates
/// (`ai/tutor/joint_blocks.py`) and the FL14 playout widths in `playout.rs`,
/// and an *empty board is a legitimate Fantasyland position* -- thirteen cards
/// go onto one.  Every one of those callers encodes the board a candidate
/// **reaches**, never the position it started from, so the board always
/// carries at least one card and never has this room.  That was checked, not
/// assumed, and the test below pins the boundary: a single placed card is
/// still computed.
pub fn unplayed_joint_block(open: [usize; 3]) -> Option<[f64; 8]> {
    if open == UNPLAYED_ROOM {
        Some([0.0; 8])
    } else {
        None
    }
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

/// splitmix64: one multiply-xor-shift step per draw.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed string hashed once; every subset of a block call shares this.
fn seed_hash(seed: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

/// `sampled_subset`'s job at splitmix cost: `want` distinct indices via a
/// partial Fisher-Yates, O(want) rng draws instead of a SHA-256 per swap.
/// The modulo bias at pool sizes <= 54 is < 2^-58 -- unmeasurable next to
/// the sampling error K itself carries.
fn sampled_subset_fast(pool_len: usize, want: usize, seed_hash: u64, tick: u64) -> Vec<usize> {
    let mut state = seed_hash ^ tick.wrapping_mul(0xA076_1D64_78BD_642F);
    let mut indices: Vec<usize> = (0..pool_len).collect();
    for position in 0..want.min(pool_len) {
        let remaining = (pool_len - position) as u64;
        let choice = position + (splitmix64(&mut state) % remaining) as usize;
        indices.swap(position, choice);
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
    recurse(
        0,
        cards,
        &open,
        &mut counts,
        &mut stack,
        &mut seen,
        &mut out,
        &id,
    );
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
    sampler: CompletionSampler,
) -> Result<[f64; 8]> {
    let mut memo = TerminalMemo::new(board);
    sampled_joint_block_shared(
        &mut memo,
        &[],
        board.open_slots(),
        pool,
        samples,
        max_arrangements,
        seed,
        fl_ev,
        sampler,
    )
}

/// The same block, against a memo whose base board sits **above** this
/// candidate.
///
/// A T3 node offers about twenty-one placements of its draw and every one of
/// them is encoded, each enumerating the same C(40,2) completions of the same
/// pool.  Building the memo from the candidate board throws that away: the
/// rows a placement did not touch are re-evaluated once per candidate, and two
/// candidates that reach the same row learn it twice.  Keying on the node's
/// board instead -- with the placement carried as the first entries of every
/// addition list -- makes every row evaluation reachable by all of them.
///
/// `base_additions` is the placement relative to the memo's board and `open`
/// is the candidate's remaining room, since the memo no longer knows either.
pub fn sampled_joint_block_shared(
    memo: &mut TerminalMemo,
    base_additions: &[(usize, Card)],
    open: [usize; 3],
    pool: &[Card],
    samples: usize,
    max_arrangements: usize,
    seed: &str,
    fl_ev: &FlEv,
    sampler: CompletionSampler,
) -> Result<[f64; 8]> {
    // The one gate; see `unplayed_joint_block` for why it lives here and not
    // at either caller.  It is first so the untouched board costs nothing:
    // thirteen open slots is where the sampling bill actually is.
    if let Some(block) = unplayed_joint_block(open) {
        return Ok(block);
    }
    let need: usize = open.iter().sum();
    let mut best_self: Vec<f64> = Vec::with_capacity(samples.max(1024));
    let (mut fouls, mut survivors) = (0usize, 0usize);
    let (mut survive_royalty, mut survive_fl) = (0.0f64, 0.0f64);
    let mut additions: Vec<(usize, Card)> = Vec::with_capacity(need + base_additions.len());

    let subsets: Box<dyn Iterator<Item = Vec<usize>>> = if samples == 0 {
        if need > 4 {
            anyhow::bail!("exact joint outlook is only affordable up to 4 open slots");
        }
        Box::new(SubsetIter::new(pool.len(), need))
    } else if sampler == CompletionSampler::SplitMix {
        // Drawn up front and iterated in sorted order: the statistics are a
        // mean over the drawn multiset, so order changes nothing they say,
        // but lexicographic iteration keeps consecutive subsets sharing row
        // fills and the row memo hot -- measured 13.2 us per random-order
        // sample against 1.7 us per subset on the exact path's ordered walk.
        let hashed = seed_hash(seed);
        let mut drawn: Vec<Vec<usize>> = (0..samples)
            .map(|tick| {
                let mut subset = sampled_subset_fast(pool.len(), need, hashed, tick as u64);
                subset.sort_unstable();
                subset
            })
            .collect();
        drawn.sort_unstable();
        Box::new(drawn.into_iter())
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
            additions.extend_from_slice(base_additions);
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
                    (terminal.royalty as f64, fl_ev.value(terminal.fl_card_count))
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
    let variance = best_self
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / count;
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
        &format!("joint/{}", request.seed.as_deref().unwrap_or(&request.id)),
        fl_ev,
        CompletionSampler::parse(request.sampler.as_deref())?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn row(names: &[&str]) -> Vec<Card> {
        names
            .iter()
            .map(|name| crate::to_core_card(name).expect("card"))
            .collect()
    }

    #[test]
    fn open_board_joint_columns_match_the_golden_bits() {
        let board = CoreBoard {
            rows: [
                row(&["2s", "3d"]),
                row(&["7h", "7d", "Ac", "Kh"]),
                row(&["9s", "Ts", "Js", "Qs", "Ks"]),
            ],
        };
        let pool = row(&["2c", "7c", "4c", "5c"]);
        let fl_ev = FlEv::load(Path::new("../../config/fl_ev.json")).expect("fl_ev");
        let block = sampled_joint_block(
            &board, &pool, 0, 32, "compat-seed", &fl_ev, CompletionSampler::Sha256)
            .expect("joint outlook");

        // Captured from the established release binary on this exact-
        // enumeration fixture.  This exercises every non-Fantasyland joint
        // statistic and guards the historical eight columns.
        let legacy = [0.0, 0.64, 0.64, 0.0, 0.04, 1.0, 1.0, 0.64];
        assert_eq!(block.map(f64::to_bits), legacy.map(f64::to_bits));
    }

    /// The board the first actor's opponent shows at T0: nothing played.
    #[test]
    fn an_unplayed_board_gets_a_zero_joint_block() {
        let board = CoreBoard { rows: [Vec::new(), Vec::new(), Vec::new()] };
        assert_eq!(board.open_slots(), UNPLAYED_ROOM);
        let pool: Vec<Card> = crate::all_cards()
            .iter()
            .map(|name| crate::to_core_card(name).expect("card"))
            .collect();
        let fl_ev = FlEv::load(Path::new("../../config/fl_ev.json")).expect("fl_ev");
        let block = sampled_joint_block(
            &board, &pool, 200, 32, "unplayed-seed", &fl_ev, CompletionSampler::Sha256)
            .expect("block");
        assert_eq!(block, [0.0; 8]);
        // Not merely equal to zero by arithmetic: the seed must not reach the
        // sampler at all, so a second seed gives the same eight zeros.
        let again = sampled_joint_block(
            &board, &pool, 200, 32, "other-seed", &fl_ev, CompletionSampler::Sha256)
            .expect("block");
        assert_eq!(again, [0.0; 8]);
    }

    /// The boundary, pinned: one placed card is a played board, and a played
    /// board is still computed.  Without this the rule could widen to "nearly
    /// empty" and silently blank the Fantasyland encoders, which do send
    /// sparse boards over the same `--joint-outlook` mode.
    #[test]
    fn one_placed_card_is_still_computed() {
        let board = CoreBoard { rows: [Vec::new(), Vec::new(), row(&["Ts"])] };
        assert_ne!(board.open_slots(), UNPLAYED_ROOM);
        let pool = row(&["Th", "Td", "Tc", "9s", "9h", "2c", "3c", "4c"]);
        let fl_ev = FlEv::load(Path::new("../../config/fl_ev.json")).expect("fl_ev");
        let block = sampled_joint_block(
            &board, &pool, 32, 32, "one-card-seed", &fl_ev, CompletionSampler::Sha256)
            .expect("block");
        assert_ne!(block, [0.0; 8]);
    }

    /// The splitmix sampler is deterministic in the seed, distinct seeds draw
    /// distinct completions, and samples=0 ignores the sampler entirely --
    /// the exact path must be byte-identical under either sampler.
    #[test]
    fn splitmix_sampler_is_deterministic_and_exact_path_ignores_it() {
        let board = CoreBoard {
            rows: [row(&["2s", "3d", "4h"]), row(&["7h", "7d", "Ac", "Kh"]), row(&["9s", "Ts"])],
        };
        let pool = row(&["2c", "7c", "4c", "5c", "8d", "9d", "Th", "Jc", "Qd", "Kd"]);
        let fl_ev = FlEv::load(Path::new("../../config/fl_ev.json")).expect("fl_ev");
        let one = sampled_joint_block(
            &board, &pool, 64, 32, "seed-a", &fl_ev, CompletionSampler::SplitMix)
            .expect("block");
        let two = sampled_joint_block(
            &board, &pool, 64, 32, "seed-a", &fl_ev, CompletionSampler::SplitMix)
            .expect("block");
        assert_eq!(one.map(f64::to_bits), two.map(f64::to_bits));
        let other = sampled_joint_block(
            &board, &pool, 64, 32, "seed-b", &fl_ev, CompletionSampler::SplitMix)
            .expect("block");
        assert_ne!(one, other);
        let exact_sha = sampled_joint_block(
            &board, &pool, 0, 1_000_000, "seed-a", &fl_ev, CompletionSampler::Sha256)
            .expect("block");
        let exact_fast = sampled_joint_block(
            &board, &pool, 0, 1_000_000, "seed-b", &fl_ev, CompletionSampler::SplitMix)
            .expect("block");
        assert_eq!(exact_sha.map(f64::to_bits), exact_fast.map(f64::to_bits));
    }
}
