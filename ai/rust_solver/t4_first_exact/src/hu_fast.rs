//! Distilled candidate evaluators for a referee's continuations.
//!
//! A referee rollout is ~130 ms and over half of it is the two sampled joint
//! blocks inside the 207-dim pair vector, paid at five model decisions per
//! hand (t1_bb, t1_btn, t2_bb, t2_btn, t3_bb -- t3_btn and t4 are exact or
//! closed-form and consult no model).  These nets read 487 deterministic
//! features off one candidate board and score it; a move is the argmax over
//! the street's ~24 candidates.
//!
//! # What this is for, and what it is not
//!
//! It buys particle count, not accuracy.  Dev top-1 agreement with the
//! champion is 0.60 / 0.51 / 0.43 at t1_bb / t2_btn / t3_bb, so a fast-net
//! referee plays a *different, weaker* continuation than the bundle would.
//! Its numbers answer "what is this opening worth when the rest is played by
//! the distilled chain" and are not comparable with a champion-continuation
//! run.  It is therefore refused anywhere near serving: `play_hand` cannot
//! reach these nets at all, and only the referee entry points can pass them.
//!
//! # The feature contract
//!
//! `ai/tutor/train_hu_fast_eval.py::featurise` is the reference and this
//! reproduces it exactly, including one thing worth writing down because it
//! looks like a bug and is not: the eighth per-row statistic is **not** flush
//! outs.  The Python computes `flush_outs` and then does not return it; the
//! slot it occupies carries `sum(unseen_rank[r] for r in ranks) / 9`, the
//! unseen-card count of the row's own ranks.  The nets were trained against
//! that vector, so that is the vector, and `--dump-fast-features` exists so
//! the claim can be diffed rather than believed.

use anyhow::{anyhow, bail, Result};
use std::path::Path;

/// Width of one candidate's feature vector.
pub const FAST_FEATURES: usize = 487;

const RANKS: &[u8] = b"23456789TJQKA";
const SUITS: &[u8] = b"shdc";
const CAP: [usize; 3] = [3, 5, 5];

/// The five decisions a fast referee replaces, as `street * 2 + seat`.
///
/// t3_btn (7) and both seats of t4 are absent on purpose: they are solved
/// exactly and there is nothing there to distil.  T0 is absent because it is
/// the decision under test.
pub const FAST_SLOTS: [(&str, usize); 5] = [
    ("t1_bb", 2),
    ("t1_btn", 3),
    ("t2_bb", 4),
    ("t2_btn", 5),
    ("t3_bb", 6),
];

/// The ten straight windows, wheel first, exactly `WINDOWS` in the trainer.
const WINDOWS: [[u8; 5]; 10] = [
    [14, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9],
    [6, 7, 8, 9, 10],
    [7, 8, 9, 10, 11],
    [8, 9, 10, 11, 12],
    [9, 10, 11, 12, 13],
    [10, 11, 12, 13, 14],
];

struct Layer {
    inputs: usize,
    outputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// One distilled candidate evaluator.
pub struct FastNet {
    pub input_dim: usize,
    pub slot: u32,
    layers: Vec<Layer>,
}

impl FastNet {
    /// Read an `HUF1` image.
    ///
    /// The magic is checked rather than the width alone: a champion image and
    /// a fast image could in principle agree on a number and disagree on every
    /// feature, and a net read by the wrong encoder is silently wrong instead
    /// of loudly broken.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 20 || &bytes[..4] != b"HUF1" {
            bail!("not a fast-net image (magic is not HUF1)");
        }
        let word = |at: usize| -> u32 {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        if word(4) != 1 {
            bail!("unsupported fast-net image version {}", word(4));
        }
        let layer_count = word(8) as usize;
        let input_dim = word(12) as usize;
        let slot = word(16);
        if input_dim != FAST_FEATURES {
            bail!("fast net reads {input_dim} features, this encoder writes {FAST_FEATURES}");
        }
        let mut at = 20usize;
        let mut layers = Vec::with_capacity(layer_count);
        let mut expected = input_dim;
        for index in 0..layer_count {
            if at + 8 > bytes.len() {
                bail!("fast-net image is truncated at layer {index}");
            }
            let inputs = word(at) as usize;
            let outputs = word(at + 4) as usize;
            at += 8;
            if inputs != expected {
                bail!("layer {index} takes {inputs} inputs, the stage before gives {expected}");
            }
            let weights = inputs * outputs;
            if at + (weights + outputs) * 4 > bytes.len() {
                bail!("fast-net image is truncated inside layer {index}");
            }
            let mut weight = Vec::with_capacity(weights);
            for slot in 0..weights {
                weight.push(f32::from_le_bytes([
                    bytes[at + slot * 4],
                    bytes[at + slot * 4 + 1],
                    bytes[at + slot * 4 + 2],
                    bytes[at + slot * 4 + 3],
                ]));
            }
            at += weights * 4;
            let mut bias = Vec::with_capacity(outputs);
            for slot in 0..outputs {
                bias.push(f32::from_le_bytes([
                    bytes[at + slot * 4],
                    bytes[at + slot * 4 + 1],
                    bytes[at + slot * 4 + 2],
                    bytes[at + slot * 4 + 3],
                ]));
            }
            at += outputs * 4;
            layers.push(Layer { inputs, outputs, weight, bias });
            expected = outputs;
        }
        if expected != 1 {
            bail!("a candidate evaluator ends in one score, this one ends in {expected}");
        }
        Ok(Self { input_dim, slot, layers })
    }

    /// Run the stack.  ReLU on every layer but the last, and no
    /// standardisation: the features are bounded by construction, which is why
    /// the image carries no mean/std block.
    pub fn predict(&self, features: &[f32], scratch: &mut Vec<f32>) -> f32 {
        let mut current: Vec<f32> = features[..self.input_dim].to_vec();
        for (position, layer) in self.layers.iter().enumerate() {
            scratch.clear();
            scratch.reserve(layer.outputs);
            for output in 0..layer.outputs {
                let row = &layer.weight[output * layer.inputs..(output + 1) * layer.inputs];
                let mut sum = layer.bias[output];
                for (index, value) in row.iter().enumerate() {
                    sum += value * current[index];
                }
                if position + 1 < self.layers.len() && sum < 0.0 {
                    sum = 0.0;
                }
                scratch.push(sum);
            }
            std::mem::swap(&mut current, scratch);
        }
        current[0]
    }
}

/// The five nets a fast referee needs, all of them or none.
pub struct FastNets {
    nets: Vec<(usize, FastNet)>,
}

impl FastNets {
    /// Load `t1_bb.bin t1_btn.bin t2_bb.bin t2_btn.bin t3_bb.bin` from a
    /// directory.
    ///
    /// A missing file is a hard error naming the slot.  Falling back to the
    /// champion for the slots that happen to be present would produce a third
    /// chain -- part distilled, part champion -- whose numbers answer no
    /// question anyone asked.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut nets = Vec::with_capacity(FAST_SLOTS.len());
        for (name, code) in FAST_SLOTS {
            let path = dir.join(format!("{name}.bin"));
            if !path.exists() {
                bail!(
                    "--hu-fast-nets is missing the {name} slot ({}); a referee with only \
                     some streets distilled is neither chain and its numbers mean nothing",
                    path.display()
                );
            }
            let image = std::fs::read(&path)
                .map_err(|e| anyhow!("{}: {e}", path.display()))?;
            let net = FastNet::load(&image)
                .map_err(|e| anyhow!("{}: {e}", path.display()))?;
            if net.slot as usize != code {
                bail!(
                    "{} carries slot {} but is named {name} (slot {code}); a net serving \
                     the wrong street would never fail, only mis-play",
                    path.display(),
                    net.slot
                );
            }
            nets.push((code, net));
        }
        Ok(Self { nets })
    }

    /// The net for this decision, if this is one of the five.
    pub fn get(&self, street: usize, seat: usize) -> Option<&FastNet> {
        let code = street * 2 + seat;
        self.nets
            .iter()
            .find(|(slot, _)| *slot == code)
            .map(|(_, net)| net)
    }

    /// The loaded slots, for the run's log.
    pub fn loaded(&self) -> String {
        FAST_SLOTS
            .iter()
            .filter(|(_, code)| self.nets.iter().any(|(slot, _)| slot == code))
            .map(|(name, _)| *name)
            .collect::<Vec<&str>>()
            .join(",")
    }
}

/// Index of a card in the 54-slot one-hot: naturals by suit-major order, then
/// X1 and X2.  Matches `train_hu_fast_eval.card_index`, jokers included: any
/// `X` name that is not exactly `X1` lands on 53.
fn card_index(name: &str) -> Result<usize> {
    if name.starts_with('X') {
        return Ok(52 + usize::from(name != "X1"));
    }
    let bytes = name.as_bytes();
    if bytes.len() != 2 {
        bail!("bad card name {name}");
    }
    let rank = RANKS
        .iter()
        .position(|r| *r == bytes[0])
        .ok_or_else(|| anyhow!("bad rank in {name}"))?;
    let suit = SUITS
        .iter()
        .position(|s| *s == bytes[1])
        .ok_or_else(|| anyhow!("bad suit in {name}"))?;
    Ok(suit * 13 + rank)
}

/// 2..14 for a natural; jokers are excluded from every rank statistic, so the
/// trainer's sentinel 15 never reaches one.
fn rank_of(name: &str) -> Result<u8> {
    if name.starts_with('X') {
        return Ok(15);
    }
    let bytes = name.as_bytes();
    RANKS
        .iter()
        .position(|r| *r == bytes[0])
        .map(|index| index as u8 + 2)
        .ok_or_else(|| anyhow!("bad rank in {name}"))
}

/// How many of each natural rank are still unseen.
///
/// Indexed by rank, so `by_rank[14]` is the aces nobody can see.  Jokers are
/// not counted: the trainer sweeps the 52 naturals and nothing else.
pub struct Unseen {
    by_rank: [u32; 16],
}

/// Unseen tallies against a set of visible card names.
pub fn unseen_of(seen: &[String]) -> Result<Unseen> {
    let mut visible = [false; 54];
    for name in seen {
        visible[card_index(name)?] = true;
    }
    let mut by_rank = [0u32; 16];
    for suit in 0..4usize {
        for rank in 0..13usize {
            if !visible[suit * 13 + rank] {
                by_rank[rank + 2] += 1;
            }
        }
    }
    Ok(Unseen { by_rank })
}

/// The eight statistics the trainer emits for one row.
fn row_stats(cards: &[String], unseen: &Unseen, out: &mut Vec<f32>) -> Result<()> {
    let mut ranks: Vec<u8> = Vec::with_capacity(cards.len());
    let mut suits: Vec<u8> = Vec::with_capacity(cards.len());
    let mut jokers = 0usize;
    for name in cards {
        if name.starts_with('X') {
            jokers += 1;
            continue;
        }
        ranks.push(rank_of(name)?);
        suits.push(name.as_bytes()[1]);
    }
    let mut best_suit_n = 0usize;
    for suit in SUITS {
        let count = suits.iter().filter(|s| *s == suit).count();
        if count > best_suit_n {
            best_suit_n = count;
        }
    }
    let mut pairs = 0usize;
    let mut counted: Vec<u8> = Vec::new();
    for rank in &ranks {
        if counted.contains(rank) {
            continue;
        }
        counted.push(*rank);
        if ranks.iter().filter(|r| *r == rank).count() > 1 {
            pairs += 1;
        }
    }
    // A window scores only if every natural in the row falls inside it with a
    // distinct rank; an empty row passes every window and scores on its jokers
    // alone, exactly as the trainer's loop does.
    let mut fill = 0.0f32;
    for window in WINDOWS {
        let mut filled: Vec<u8> = Vec::with_capacity(5);
        let mut ok = true;
        for rank in &ranks {
            if !window.contains(rank) || filled.contains(rank) {
                ok = false;
                break;
            }
            filled.push(*rank);
        }
        if ok {
            let value = ((filled.len() + jokers) as f32 / 5.0).min(1.0);
            if value > fill {
                fill = value;
            }
        }
    }
    let max_rank = ranks.iter().copied().max().unwrap_or(0);
    let rank_sum: u32 = ranks.iter().map(|r| *r as u32).sum();
    // The eighth slot: unseen cards sharing the row's ranks, counted with
    // multiplicity.  NOT flush outs -- see the module header.
    let unseen_of_ranks: u32 = ranks.iter().map(|r| unseen.by_rank[*r as usize]).sum();
    out.push(cards.len() as f32 / 5.0);
    out.push(jokers as f32 / 2.0);
    out.push(max_rank as f32 / 15.0);
    out.push(rank_sum as f32 / 70.0);
    out.push(best_suit_n as f32 / 5.0);
    out.push(pairs as f32 / 2.0);
    out.push(fill);
    out.push(unseen_of_ranks as f32 / 9.0);
    Ok(())
}

/// One candidate's 487 features, in the trainer's order.
#[allow(clippy::too_many_arguments)]
pub fn featurise(
    after: &[Vec<String>; 3],
    opp: &[Vec<String>; 3],
    seen: &[String],
    discard: &str,
    street: usize,
    unseen: &Unseen,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    out.resize(FAST_FEATURES, 0.0);
    let mut at = 0usize;
    for rows in [after, opp] {
        for row in rows {
            for card in row {
                out[at + card_index(card)?] = 1.0;
            }
            at += 54;
        }
    }
    for card in seen {
        out[at + card_index(card)?] = 1.0;
    }
    at += 54;
    out[at + card_index(discard)?] = 1.0;
    at += 54;
    let mut stats: Vec<f32> = Vec::with_capacity(48);
    for rows in [after, opp] {
        for row in rows {
            row_stats(row, unseen, &mut stats)?;
        }
    }
    out[at..at + 48].copy_from_slice(&stats);
    at += 48;
    for row in 0..3usize {
        out[at + row] = (CAP[row] - after[row].len().min(CAP[row])) as f32 / 5.0;
    }
    at += 3;
    out[at + street.min(3)] = 1.0;
    at += 4;
    debug_assert_eq!(at, FAST_FEATURES);
    Ok(())
}

/// The cards a seat can see when it acts: its own board before the move, the
/// opponent's visible board, its own draw, and its own discards.
///
/// The opponent's discards are face down and deliberately absent.
pub fn seen_of(
    own_before: &[Vec<String>; 3],
    opp: &[Vec<String>; 3],
    draw: &[String],
    own_discards: &[String],
) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for row in own_before {
        seen.extend(row.iter().cloned());
    }
    for row in opp {
        seen.extend(row.iter().cloned());
    }
    seen.extend(draw.iter().cloned());
    seen.extend(own_discards.iter().cloned());
    seen
}

/// The trainer's own candidate enumeration, used only by the parity dump.
///
/// `placements` in `hu_match` reaches the same set of boards but dedupes and
/// enumerates in its own order.  The dump has to line up row for row with the
/// Python so a diff is a diff and not an alignment puzzle, so it reproduces
/// `train_hu_fast_eval.candidates` instead: the draw sorted, then discard
/// position major, and the action index it names.
pub fn python_candidates(
    own: &[Vec<String>; 3],
    draw: &[String],
) -> Vec<(usize, [Vec<String>; 3], String)> {
    let mut order: Vec<String> = draw.to_vec();
    order.sort();
    let room: Vec<usize> = (0..3).map(|row| CAP[row] - own[row].len().min(CAP[row])).collect();
    let mut out = Vec::new();
    for discard_pos in 0..3usize {
        let kept: Vec<&String> = order
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != discard_pos)
            .map(|(_, card)| card)
            .collect();
        for r1 in 0..3usize {
            for r2 in 0..3usize {
                let mut need = [0usize; 3];
                need[r1] += 1;
                need[r2] += 1;
                if (0..3).any(|row| need[row] > room[row]) {
                    continue;
                }
                let mut after = own.clone();
                after[r1].push(kept[0].clone());
                after[r2].push(kept[1].clone());
                out.push((discard_pos * 9 + r1 * 3 + r2, after, order[discard_pos].clone()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// **The 54-slot index is the trainer's.**
    ///
    /// Suit-major over "shdc" with ranks "23456789TJQKA", then the two jokers.
    /// An index that disagreed by one would still produce a valid-looking
    /// vector and a net reading the wrong card everywhere.
    #[test]
    fn the_card_index_matches_the_trainer() {
        assert_eq!(card_index("2s").unwrap(), 0);
        assert_eq!(card_index("As").unwrap(), 12);
        assert_eq!(card_index("2h").unwrap(), 13);
        assert_eq!(card_index("Ac").unwrap(), 51);
        assert_eq!(card_index("X1").unwrap(), 52);
        assert_eq!(card_index("X2").unwrap(), 53);
        assert!(card_index("Zz").is_err());
    }

    /// **The eighth row statistic is the unseen-rank sum, not flush outs.**
    ///
    /// The trainer computes `flush_outs` and returns something else in that
    /// slot; the nets were fitted against what it returns.  Pinned here
    /// because it reads like a bug and "fixing" it would silently break every
    /// net trained so far.
    #[test]
    fn the_eighth_row_statistic_counts_unseen_ranks() {
        // Nothing seen: every rank has four unseen cards.
        let unseen = unseen_of(&[]).unwrap();
        let mut out = Vec::new();
        row_stats(&row(&["As", "Ad"]), &unseen, &mut out).unwrap();
        // Two aces in the row, four aces unseen each (the row's own cards are
        // not in `seen` here), so 8/9.
        assert!((out[7] - 8.0 / 9.0).abs() < 1e-6, "eighth stat is {}", out[7]);
        // A flush-outs reading would have been 13 spades-ish / 13; it is not.
        assert!((out[7] - 1.0).abs() > 1e-6);

        // With the aces visible, the same row's unseen-rank sum drops.
        let unseen = unseen_of(&row(&["As", "Ad", "Ah"])).unwrap();
        let mut out = Vec::new();
        row_stats(&row(&["As", "Ad"]), &unseen, &mut out).unwrap();
        assert!((out[7] - 2.0 / 9.0).abs() < 1e-6, "eighth stat is {}", out[7]);
    }

    /// **The straight window is the trainer's, wheel included.**
    #[test]
    fn the_straight_fill_scores_the_wheel_and_rejects_duplicates() {
        let unseen = unseen_of(&[]).unwrap();
        let fill = |cards: &[&str]| -> f32 {
            let mut out = Vec::new();
            row_stats(&row(cards), &unseen, &mut out).unwrap();
            out[6]
        };
        // A,2,3 is a wheel draw: three of five.
        assert!((fill(&["As", "2d", "3h"]) - 0.6).abs() < 1e-6);
        // A pair fits no window with distinct ranks.
        assert!(fill(&["As", "Ad"]).abs() < 1e-6);
        // Jokers count toward the fill and cap it at one.
        assert!((fill(&["9s", "Td", "X1"]) - 0.6).abs() < 1e-6);
        // An empty row passes every window on its jokers alone.
        assert!(fill(&[]).abs() < 1e-6);
        // A,K,Q is the top window, not the wheel.
        assert!((fill(&["As", "Kd", "Qh"]) - 0.6).abs() < 1e-6);
    }

    /// **The vector is 487 wide and its blocks land where the trainer puts
    /// them.**
    #[test]
    fn the_feature_vector_is_laid_out_in_the_trainers_order() {
        let after = [row(&["As"]), row(&["2h", "3h"]), row(&[])];
        let opp = [row(&[]), row(&["Kc"]), row(&[])];
        let seen = row(&["As", "2h", "3h", "Kc", "7d"]);
        let unseen = unseen_of(&seen).unwrap();
        let mut out = Vec::new();
        featurise(&after, &opp, &seen, "7d", 2, &unseen, &mut out).unwrap();
        assert_eq!(out.len(), FAST_FEATURES);
        // Own row 0 holds As at 12.
        assert_eq!(out[12], 1.0);
        // Own row 1 (offset 54) holds 2h (13) and 3h (14).
        assert_eq!(out[54 + 13], 1.0);
        assert_eq!(out[54 + 14], 1.0);
        // Opponent row 1 is the fifth 54-block: Kc at 50.
        assert_eq!(out[54 * 4 + 50], 1.0);
        // The seen block is the seventh, the discard the eighth.
        assert_eq!(out[54 * 6 + card_index("7d").unwrap()], 1.0);
        assert_eq!(out[54 * 7 + card_index("7d").unwrap()], 1.0);
        // Room after: (3-1, 5-2, 5-0)/5.
        let room = 54 * 8 + 48;
        assert!((out[room] - 0.4).abs() < 1e-6);
        assert!((out[room + 1] - 0.6).abs() < 1e-6);
        assert!((out[room + 2] - 1.0).abs() < 1e-6);
        // Street 2 one-hot.
        assert_eq!(out[room + 3 + 2], 1.0);
        assert_eq!(out[room + 3 + 3], 0.0);
        // Street 4 folds onto slot 3.
        let mut late = Vec::new();
        featurise(&after, &opp, &seen, "7d", 4, &unseen, &mut late).unwrap();
        assert_eq!(late[room + 3 + 3], 1.0);
    }

    /// **A fast image is refused where a champion image is expected, and the
    /// reverse.**
    #[test]
    fn only_an_huf1_image_loads() {
        assert!(FastNet::load(b"T4F1\x01\x00\x00\x00").is_err());
        assert!(FastNet::load(b"").is_err());
        assert!(FastNet::load(b"HUF1\x09\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").is_err());
    }

    /// **The parity dump enumerates candidates the way the trainer does.**
    ///
    /// Sorted draw, discard position major, and the action index the trainer
    /// names -- otherwise a row-for-row diff would compare different moves.
    #[test]
    fn the_dump_enumerates_candidates_like_the_trainer() {
        let own = [row(&["Kd"]), row(&["4s", "4c"]), row(&["Qs", "Qc"])];
        let draw = row(&["3h", "5s", "As"]);
        let out = python_candidates(&own, &draw);
        // Sorted draw is [3h, 5s, As]; discard 0 throws 3h and keeps 5s, As.
        assert_eq!(out[0].0, 0);
        assert_eq!(out[0].2, "3h");
        // Action index is discard*9 + r1*3 + r2, and the first legal one from
        // this board is (0,0) -- the top row has two free slots.
        assert!(out.iter().any(|(action, _, _)| *action == 0));
        // Every action index is distinct and in range.
        let mut seen: Vec<usize> = out.iter().map(|(action, _, _)| *action).collect();
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count);
        assert!(seen.iter().all(|action| *action < 27));
        // Each candidate places two cards and throws the third.
        for (_, after, discard) in &out {
            let placed: usize = after.iter().map(Vec::len).sum();
            assert_eq!(placed, 7, "a candidate placed {placed} cards");
            assert!(draw.contains(discard));
        }
    }
}
