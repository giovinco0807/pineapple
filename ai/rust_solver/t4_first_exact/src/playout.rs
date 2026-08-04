//! Chain playouts against a Fantasyland opponent, shared by every street.
//!
//! One recursion covers T1/T2/T3: from a board with an even number of open
//! slots, sample the next street's draw, let that street's learned model pick
//! among the candidate placements, and descend.  At 11 cards the recursion
//! stops and the position is priced exactly -- mean over T4 draws of the best
//! completion's direct FL-library score.  Models therefore only ever choose
//! moves; every number that reaches a label comes from exact scoring, which
//! is the error-containment structure the T3-vs-FL v1 failure established.
//!
//! Feature width selects the encoder, so one code path serves every street's
//! model without a street tag:
//!   60  = actor + FL context                    (light playout policy)
//!   101 = actor + rowwise + FL context          (light-lap T1/T2 evaluators)
//!   109 = actor + rowwise + joint + FL context  (full T3 evaluator; the
//!         joint block needs exactly two open slots, so it is only ever
//!         reachable from an 11-card board)

use anyhow::{bail, Result};
use ofc_core::Card;
use std::collections::HashMap;
use std::sync::Mutex;

/// Rowwise slices cached across the whole root: common random numbers make
/// every action walk the same draw paths, so the pool at a node is
/// identified by its seed string and row contents repeat massively across
/// the action fan-out.  Key = (hash of the node's seed, row-cards key).
pub(crate) type RowwiseMemo = Mutex<HashMap<(u64, u64), ([f32; 12], usize)>>;

fn seed_hash(seed: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    hasher.finish()
}

use super::evaluator;
use super::row_memo::TerminalMemo;
use super::t3_second;
use super::t3_vs_fl::sampled_draws;
use super::t3_vs_fl_lib::{card_bit, score_mean, terminal_key, FlLibrary, MatchedRow, TerminalKey};
use super::{CoreBoard, FlEv};

/// Everything the descent needs that does not change between nodes.
pub(crate) struct Context<'a> {
    pub(crate) fl_ev: &'a FlEv,
    pub(crate) library: &'a FlLibrary,
    pub(crate) fl_table: &'a evaluator::FlTable,
    /// Move choosers, ordered from the shallowest street down to T3.  The
    /// descent pops the front as it goes: a T1 request passes [T2, T3], a T2
    /// request passes [T3], a T3 request passes [].
    pub(crate) models: &'a [&'a evaluator::Model],
    /// Draws sampled per street, aligned with `models`.
    pub(crate) samples: &'a [usize],
    pub(crate) opp_count: u8,
    /// T4 draws sampled at the terminal; 0 enumerates all C(n,3).
    pub(crate) t4_draw_sample: usize,
    /// Root-wide rowwise cache; see RowwiseMemo.
    pub(crate) rowwise_memo: RowwiseMemo,
}

/// One candidate placement of two drawn cards into rows, at the Card level.
pub(crate) struct Candidate {
    pub(crate) placements: [(usize, Card); 2],
}

/// Distinct placements of two of three drawn cards.  Keyed on card identity,
/// so the interchangeable jokers collapse instead of duplicating work.
pub(crate) fn candidates(board: &CoreBoard, draw: &[Card; 3]) -> Vec<Candidate> {
    let open = board.open_slots();
    let mut out: Vec<Candidate> = Vec::new();
    let mut seen: std::collections::BTreeSet<(u32, u32, u32)> =
        std::collections::BTreeSet::new();
    let id = |card: &Card| -> u32 {
        if card.is_joker() {
            52
        } else {
            card.suit as u32 * 13 + card.rank as u32 - 2
        }
    };
    for discard_index in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard_index).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|row| need[row] > open[row]) {
                    continue;
                }
                let mut pair = [
                    (row_a as u32) << 8 | id(&draw[kept[0]]),
                    (row_b as u32) << 8 | id(&draw[kept[1]]),
                ];
                pair.sort_unstable();
                if seen.insert((pair[0], pair[1], id(&draw[discard_index]))) {
                    out.push(Candidate {
                        placements: [(row_a, draw[kept[0]]), (row_b, draw[kept[1]])],
                    });
                }
            }
        }
    }
    out
}

/// Encode a board for `model`, choosing the block set by its input width.
/// `rowwise_memo` caches per-row 12-dim slices across a node's candidates:
/// a candidate changes at most two rows and the pool is fixed within the
/// node, so most rows repeat -- the difference between minutes and hours on
/// a T0 root (232 candidates x C(pool,2) evaluations per uncached row).
pub(crate) fn encode_for(
    model: &evaluator::Model,
    board: &CoreBoard,
    unseen: &[Card],
    opp_count: u8,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    evaluator::actor_block(&board.rows, out);
    if model.input_dim >= 101 {
        let _categories = evaluator::opponent_rowwise_block_shared(
            &board.rows, unseen, fl_table, memo, pool_key, out,
        );
    }
    if model.input_dim >= 109 {
        for value in t3_second::joint_block(board, unseen, fl_ev)? {
            out.push(value as f32);
        }
    }
    super::t3_vs_fl::fl_context(unseen, opp_count, fl_ev, out);
    if out.len() != model.input_dim {
        bail!(
            "playout feature width {} does not match model input {}",
            out.len(),
            model.input_dim
        );
    }
    Ok(())
}

/// The board the model would choose from this draw.
fn choose(
    model: &evaluator::Model,
    board: &CoreBoard,
    draw: &[Card; 3],
    unseen: &[Card],
    context: &Context<'_>,
    pool_key: u64,
    features: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) -> Result<Option<CoreBoard>> {
    let mut best_score = f32::NEG_INFINITY;
    let mut chosen: Option<CoreBoard> = None;
    for candidate in candidates(board, draw) {
        let mut next = board.clone();
        next.rows[candidate.placements[0].0].push(candidate.placements[0].1);
        next.rows[candidate.placements[1].0].push(candidate.placements[1].1);
        encode_for(
            model,
            &next,
            unseen,
            context.opp_count,
            context.fl_ev,
            context.fl_table,
            &context.rowwise_memo,
            pool_key,
            features,
        )?;
        let predicted = model.predict(features, scratch);
        if predicted > best_score {
            best_score = predicted;
            chosen = Some(next);
        }
    }
    Ok(chosen)
}

/// Exact T4 continuation value of an 11-card board: mean over T4 draws of the
/// best completion's library score, on an already-filtered shortlist.
fn terminal_value(
    board: &CoreBoard,
    unseen: &[Card],
    shortlist: &[u32],
    context: &Context<'_>,
    seed: &str,
) -> Result<(f64, f64)> {
    let draw_sets = sampled_draws(unseen.len(), context.t4_draw_sample, seed);
    let patterns = t3_second::placement_patterns(board);
    let mut terminal_memo = TerminalMemo::new(board);
    let mut memo: std::collections::HashMap<TerminalKey, f64> =
        std::collections::HashMap::with_capacity(patterns.len());
    let mut matched: Vec<MatchedRow> = Vec::with_capacity(shortlist.len());
    let mut total = 0.0f64;
    let mut sample_total = 0usize;
    for draw in &draw_sets {
        let draw_cards = [unseen[draw[0]], unseen[draw[1]], unseen[draw[2]]];
        let mut draw_mask = 0u64;
        for card in &draw_cards {
            draw_mask |= card_bit(card)?;
        }
        matched.clear();
        for &index in shortlist {
            let index = index as usize;
            if context.library.masks[index] & draw_mask == 0 {
                matched.push(MatchedRow {
                    values: context.library.values[index],
                    royalty: context.library.royalty[index],
                    stay: context.library.stay[index],
                    busted: context.library.busted[index],
                });
            }
        }
        if matched.is_empty() {
            continue;
        }
        sample_total += matched.len();
        let mut best = f64::NEG_INFINITY;
        memo.clear();
        for pattern in &patterns {
            let terminal = terminal_memo.terminal(&[
                (pattern.rows[0], draw_cards[pattern.cards[0]]),
                (pattern.rows[1], draw_cards[pattern.cards[1]]),
            ]);
            let key = terminal_key(&terminal);
            let value = match memo.get(&key) {
                Some(cached) => *cached,
                None => {
                    let computed =
                        score_mean(&terminal, &matched, context.opp_count, context.fl_ev);
                    memo.insert(key, computed);
                    computed
                }
            };
            if value > best {
                best = value;
            }
        }
        total += best;
    }
    let draws = draw_sets.len().max(1);
    Ok((total / draws as f64, sample_total as f64 / draws as f64))
}

/// Value of `board` (9 or fewer cards: models remain) or its exact terminal
/// value (11 cards: none remain), as (value, mean FL samples per terminal).
///
/// `seen_mask` carries the library-disjointness bits for every card already
/// revealed on this line, so the FL hands sampled at the terminal stay
/// consistent with the whole playout, not just the root.
pub(crate) fn descend(
    board: &CoreBoard,
    unseen: &[Card],
    seen_mask: u64,
    jokers_seen: usize,
    context: &Context<'_>,
    depth: usize,
    seed: &str,
    features: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) -> Result<(f64, f64)> {
    if depth == context.models.len() {
        if board.card_count() != 11 {
            bail!(
                "playout ran out of models on a {}-card board",
                board.card_count()
            );
        }
        // card_bit sets both joker bits for either joker, since the library
        // masks distinguish X1 from X2 and a mask test only asks "is this
        // physical card taken".  With exactly one joker seen the other is
        // still available, so its bit is reopened here.
        let mut mask = seen_mask;
        if jokers_seen == 1 {
            mask &= !(1u64 << 53);
        }
        let shortlist: Vec<u32> = (0..context.library.len() as u32)
            .filter(|&index| context.library.masks[index as usize] & mask == 0)
            .collect();
        return terminal_value(board, unseen, &shortlist, context, seed);
    }

    let model = context.models[depth];
    let triples = sampled_draws(unseen.len(), context.samples[depth], seed);
    let mut total = 0.0f64;
    let mut fl_samples = 0.0f64;
    let mut lines = 0usize;
    for (index, triple) in triples.iter().enumerate() {
        let draw = [unseen[triple[0]], unseen[triple[1]], unseen[triple[2]]];
        let mut drawn_mask = 0u64;
        for card in &draw {
            drawn_mask |= card_bit(card)?;
        }
        let next_unseen: Vec<Card> = unseen
            .iter()
            .enumerate()
            .filter(|(position, _)| !triple.contains(position))
            .map(|(_, card)| *card)
            .collect();
        let child_seed = format!("{seed}/{index}");
        let Some(next_board) = choose(
            model,
            board,
            &draw,
            &next_unseen,
            context,
            seed_hash(&child_seed),
            features,
            scratch,
        )?
        else {
            continue;
        };
        let drawn_jokers = draw.iter().filter(|card| card.is_joker()).count();
        let (value, samples) = descend(
            &next_board,
            &next_unseen,
            seen_mask | drawn_mask,
            jokers_seen + drawn_jokers,
            context,
            depth + 1,
            &child_seed,
            features,
            scratch,
        )?;
        total += value;
        fl_samples += samples;
        lines += 1;
    }
    let effective = lines.max(1);
    Ok((total / effective as f64, fl_samples / effective as f64))
}
