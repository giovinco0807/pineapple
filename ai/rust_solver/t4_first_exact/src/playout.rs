//! Chain playouts against a Fantasyland opponent, shared by every street.
//!
//! One recursion covers T1/T2/T3: from a board with an even number of open
//! slots, sample the next street's draw, let that street's learned model pick
//! among the candidate placements, and descend.  At 11 cards the recursion
//! stops and the position is priced exactly -- mean over T4 draws of the best
//! completion's score against the opponents the leaf was given.  Models
//! therefore only ever choose moves; every number that reaches a label comes
//! from exact scoring, which is the error-containment structure the T3-vs-FL
//! v1 failure established.
//!
//! Which opponents those are is the leaf's one degree of freedom, and the two
//! choices are not the same game.  See [`Opponents`].
//!
//! Feature width selects the encoder, so one code path serves every street's
//! model without a street tag:
//!   60  = actor + FL context                    (light playout policy)
//!   101 = actor + rowwise + FL context          (light-lap T1/T2 evaluators)
//!   104 = actor + rowwise + sampled joint + deck  (FL14 best-response
//!         teachers; see `ai/tutor/encode_fl14_teacher.py`.  A different
//!         joint block and a different tail from 109, not a truncation of
//!         it -- the joint here is the completion outlook that admits more
//!         than two open slots, and the seven-dim tail drops the columns a
//!         fixed opponent width makes constant)
//!   109 = actor + rowwise + joint + FL context  (full T3 evaluator; the
//!         joint block needs exactly two open slots, so it is only ever
//!         reachable from an 11-card board)
//!   110 = the 104-dim FL14 vector plus six placed-card rank tiebreaks

use anyhow::{anyhow, bail, Result};
use fl_solver::pool::{Pool, PoolEntry};
use fl_solver::vs_fl::{self, HeroTerminal};
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
use super::joint_outlook;
use super::row_memo::TerminalMemo;
use super::t3_second;
use super::t3_vs_fl::sampled_draws;
use super::t3_vs_fl_lib::{
    card_bit, score_mean, terminal_key, FlLibrary, LibrarySet, MatchedRow, TerminalKey,
};
use super::{CoreBoard, FlEv, Terminal};

/// The 52 natural cards, in the pool's rank-major bit order.
const POOL_NATURALS: u64 = (1u64 << 52) - 1;

/// actor 48 + rowwise 41 + joint 8 + deck 7 = 104, the legacy FL14 width.
/// This prefix is frozen because existing models carry only their width, not
/// an encoder schema identifier.
pub(crate) const FL14_FEATURE_SIZE_V1: usize =
    evaluator::ACTOR_SIZE + evaluator::OPPONENT_SIZE + evaluator::FL14_CONTEXT_SIZE;
/// v2 appends six category-aware placed-rank tiebreaks after the v1 prefix.
pub(crate) const FL14_FEATURE_SIZE: usize = FL14_FEATURE_SIZE_V1 + evaluator::ALLOCATION_RANK_SIZE;

/// The same blocks minus the joint one: what a search can afford.
///
/// `OPPONENT_SIZE` is rowwise 41 and joint 8 added together, so the eight come
/// off here by name rather than by a constant of their own.  This remains the
/// v1 96-dim ranker; a v2 ranker is deliberately not inferred from the full
/// width and needs an explicit encoder role before it is introduced.
const FL14_RANKER_SIZE: usize = FL14_FEATURE_SIZE_V1 - 8;

/// The 96-dim ranker's blocks plus sixteen deterministic draw descriptors.
///
/// 112 collides with no other width this crate dispatches on (60, 96, 101,
/// 104, 109, 110 here; 207 and the hybrid in `hu_encode`), which is what lets
/// a model of this width select the encoder by width alone -- so shipping one
/// is a model swap and nothing else.  The point of the width is that it costs
/// what 96 costs: no sampling, no completions, just counts over the board and
/// the unseen pool.
const FL14_CHEAP_SIZE: usize = FL14_RANKER_SIZE + evaluator::CHEAP_DRAW_SIZE;

/// v2 of the same idea, 24 dims wide: 120, likewise colliding with no other
/// width this crate dispatches on.  v1 stays reachable so the two can be
/// compared on one binary.
const FL14_CHEAP_V2_SIZE: usize = FL14_RANKER_SIZE + evaluator::CHEAP_DRAW_V2_SIZE;

/// 128 = the 120 assembly with the eight joint statistics appended -- the
/// oracle-proven composition (dev regret 0.4036 vs 110's 0.4042 on the T2
/// own corpus) at splitmix sampling cost.  The joint block here is drawn by
/// `CompletionSampler::SplitMix`, never Sha256: at T2 the legacy sampler's
/// SHA-256 shuffle is ~95% of the block's cost, and a width that has never
/// shipped has no compatibility to keep.  Order: actor 48 | rowwise 41 |
/// context 7 | cheap v2 24 | joint 8 -- joint LAST, unlike 104/110, because
/// the corpus this width trains on is built by appending the joint to the
/// 120 vector.
const FL14_CHEAP_V2_JOINT_SIZE: usize = FL14_CHEAP_V2_SIZE + 8;

/// What `encode_fl14_teacher` passes the block binary: `--joint-samples`
/// defaults to 400 at T2 and to exact enumeration at T3/T4, and
/// `max_arrangements` is 32 at every street.  The street is recoverable from
/// the board -- a T2 placement leaves four open slots, a T3 placement two --
/// so the encoder needs no street tag to reproduce the teacher's setting.
const FL14_JOINT_SAMPLES: usize = 400;
const FL14_JOINT_ARRANGEMENTS: usize = 32;

/// What the leaf prices hero's finished board against.
///
/// The two variants are different games, not different precisions.  The
/// library is a shelf of Fantasyland boards each solved once against nobody,
/// which is what the rule correction invalidated: a Fantasyland player sets its
/// thirteen *after* hero's board is complete, so scoring hero against a board
/// chosen in advance lets hero beat an opponent who was not allowed to react.
/// The pool keeps every drawn hand's whole best-response frontier and lets it
/// answer the board hero actually finished with.
///
/// The library path survives because runs labelled through it have to stay
/// reproducible, not because it is right.
pub(crate) enum Opponents<'a> {
    Library(&'a FlLibrary),
    /// Drawn once per root, so every action at that root prices the same
    /// opponents and the draw cancels in the action-to-action differences a
    /// teacher is consumed for.
    ///
    /// `opp_count` is not consulted on this path: the pool's stay term is
    /// already inside each row's `static_value`, priced at the width the pool
    /// was built for and pinned by its header.
    Pool(&'a [&'a PoolEntry]),
}

/// Where a labelling run's opponents come from, before a root turns them into
/// an [`Opponents`].
///
/// Not the same type as `Opponents` because a pool has to be drawn from once
/// per root and a library does not: the library is filtered per leaf against
/// whatever hero has seen, while the pool's draw is a sample and has to be
/// fixed before the action fan-out so the fan-out shares it.
pub(crate) enum OpponentSource<'a> {
    Libraries(&'a LibrarySet),
    Pool(&'a Pool),
}

/// The opponents one root faces, drawn once for the whole action fan-out.
///
/// `seen` is everything hero can see at the root -- board, discards and draw
/// -- which no action changes, so one draw serves every action and the sampled
/// opponents cancel in the action-to-action differences a teacher is read for.
/// Same argument as the common random numbers the sampled continuations
/// already share; drawing per action would put an independent sample between
/// two numbers that are only ever subtracted.
pub(crate) fn root_opponents<'a>(
    pool: &'a Pool,
    seen: &[Card],
    want: usize,
    stream: u64,
) -> Result<Vec<&'a PoolEntry>> {
    let (naturals, jokers) = pool_mask_of(seen);
    fl_solver::pool::draw(pool, naturals, jokers, want, stream).map_err(|short| {
        anyhow!(
            "pool yielded {} of {} opponents for this root; a label quietly \
             averaged over however many turned up is the expensive kind of \
             silent failure",
            short.found,
            short.wanted
        )
    })
}

/// The opponent stream for a root, from the request's own id.
///
/// FNV-1a rather than `DefaultHasher`: the stream decides which opponents a
/// label was priced against, and a hasher the standard library is free to
/// change between toolchains would make a rerun a different run.
pub(crate) fn root_stream(id: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fl_solver::pool::mix64(hash)
}

/// Everything the descent needs that does not change between nodes.
pub(crate) struct Context<'a> {
    pub(crate) fl_ev: &'a FlEv,
    pub(crate) opponents: Opponents<'a>,
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
    /// Price hero's finished board by its own worth -- royalty plus the
    /// Fantasyland entry it earns, a foul flat at -6 -- and never consult an
    /// opponent.  The 2026-08-14 objective: with it set, nothing at the leaf
    /// is sampled and the T4 mean is exact over its draw set.
    pub(crate) own_only: bool,
    /// Depth at which the chooser's own value is taken as the line's value
    /// instead of playing the line out.
    ///
    /// `None` plays every line to an eleven-card board and prices it against
    /// the opponents, so no number a label is built from comes from a model --
    /// the containment structure the T3-vs-FL v1 failure established, where a
    /// learned T4 leaf put MAE 0.645 into T3 labels and its per-board errors
    /// were correlated enough across a root's draws to survive a 300-draw
    /// average.
    ///
    /// `Some(d)` truncates, which is orders of magnitude faster and reopens
    /// exactly that risk.  It exists to be measured against the untruncated
    /// labels on the same roots, not to be switched on by default.
    pub(crate) truncate_depth: Option<usize>,
    /// A cheap pre-ranking pass for the T2 node: `(fence model, K)`.
    ///
    /// The T2 fan-out is where the chain's time goes -- the serving evaluator's
    /// joint block samples 400 completions per candidate -- and the cheap
    /// width-120 chooser scores the same candidates for nothing.  With the
    /// fence set, every candidate is scored by the fence first and only its
    /// top-K reach the serving model's argmax.  Offline on sharp labels,
    /// K=12 matched full evaluation (regret 0.1237 vs 0.1245) at half the
    /// cost; K=8 cost +0.003 for 2.8x.
    ///
    /// `None` is the untouched chain: `choose` must emit bit-identical
    /// decisions in that case, so the fence is a strict addition and never a
    /// re-ordering of the field.
    pub(crate) t2_fence: Option<(&'a evaluator::Model, usize)>,
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
    let mut seen: std::collections::BTreeSet<(u32, u32, u32)> = std::collections::BTreeSet::new();
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
///
/// `node_seed` only reaches the FL14 width, whose joint block is sampled: it
/// picks the completions, and every candidate at a node has to be judged on
/// the same ones or the sampling noise lands straight on the ordering the
/// chooser reads.  The `joint/` prefix mirrors `joint_outlook::solve`, so a
/// playout node and a teacher row encoded from the same seed draw the same
/// completions.
pub(crate) fn encode_for(
    model: &evaluator::Model,
    board: &CoreBoard,
    unseen: &[Card],
    opp_count: u8,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &RowwiseMemo,
    pool_key: u64,
    node_seed: &str,
    out: &mut Vec<f32>,
    // `joint`: a memo whose base board is the node above `board`, together
    // with the placement that separates them.  `None` builds a private one,
    // which is what a caller encoding a single board wants; `choose` passes
    // one it keeps across the node's candidates.
    joint: Option<(&mut TerminalMemo, &[(usize, Card)])>,
) -> Result<()> {
    match model.input_dim {
        // Historical actor/context, rowwise, FL14 v1/v2, and full-T3 widths.
        // Width is the only encoder role in the v1 model image, so experimental
        // 113/116 images must fail loudly instead of falling through to 109.
        60 | FL14_RANKER_SIZE | 101 | FL14_FEATURE_SIZE_V1 | 109 | FL14_FEATURE_SIZE
        | FL14_CHEAP_SIZE | FL14_CHEAP_V2_SIZE | FL14_CHEAP_V2_JOINT_SIZE => {}
        unsupported => bail!("unsupported playout model input width {unsupported}"),
    }
    out.clear();
    evaluator::actor_block(&board.rows, out);
    // 96 is below the 101 threshold but still wants the rowwise block -- it is
    // 104 with the joint eight removed, not a narrower feature set.
    if model.input_dim >= 101 || model.input_dim == FL14_RANKER_SIZE {
        let _categories = evaluator::opponent_rowwise_block_shared(
            &board.rows,
            unseen,
            fl_table,
            memo,
            pool_key,
            out,
        );
    }
    if model.input_dim == FL14_RANKER_SIZE {
        // The FL14 blocks without the joint one.
        //
        // Not a smaller model -- a differently-employed one.  The joint block
        // is worth 0.085 of regret at T2 (0.134 with it, 0.219 without) and
        // the 104-dim evaluator keeps it to serve the street.  But it costs
        // 98% of a candidate encode -- one T1 request measured 0.77 s without
        // it and 48.6 s with -- and a search encodes thousands of candidates
        // per root.  So the street is served by the model that sees the most
        // and searched by the model that costs the least.
        evaluator::fl14_context_block(unseen, out);
    } else if model.input_dim == FL14_CHEAP_SIZE {
        // Exactly the 96-dim assembly, then the draw descriptors appended.
        // Sharing the prefix is deliberate: a 112 model is a 96 model that has
        // been told what the rows are reaching for, so the two must not be
        // able to disagree about the first 96 dims.
        evaluator::fl14_context_block(unseen, out);
        evaluator::cheap_draw_block(&board.rows, unseen, out);
    } else if model.input_dim == FL14_CHEAP_V2_SIZE {
        evaluator::fl14_context_block(unseen, out);
        evaluator::cheap_draw_block_v2(&board.rows, unseen, out);
    } else if model.input_dim == FL14_CHEAP_V2_JOINT_SIZE {
        evaluator::fl14_context_block(unseen, out);
        evaluator::cheap_draw_block_v2(&board.rows, unseen, out);
        let open: usize = board.open_slots().iter().sum();
        let samples = if open <= 2 { 0 } else { FL14_JOINT_SAMPLES };
        let seed = format!("joint/{node_seed}");
        let block = match joint {
            Some((shared, placement)) => joint_outlook::sampled_joint_block_shared(
                shared,
                placement,
                board.open_slots(),
                unseen,
                samples,
                FL14_JOINT_ARRANGEMENTS,
                &seed,
                fl_ev,
                joint_outlook::CompletionSampler::SplitMix,
            )?,
            None => joint_outlook::sampled_joint_block(
                board,
                unseen,
                samples,
                FL14_JOINT_ARRANGEMENTS,
                &seed,
                fl_ev,
                joint_outlook::CompletionSampler::SplitMix,
            )?,
        };
        for value in block {
            out.push(value as f32);
        }
    } else if model.input_dim == FL14_FEATURE_SIZE_V1 || model.input_dim == FL14_FEATURE_SIZE {
        let open: usize = board.open_slots().iter().sum();
        let samples = if open <= 2 { 0 } else { FL14_JOINT_SAMPLES };
        let seed = format!("joint/{node_seed}");
        let block = match joint {
            Some((shared, placement)) => joint_outlook::sampled_joint_block_shared(
                shared,
                placement,
                board.open_slots(),
                unseen,
                samples,
                FL14_JOINT_ARRANGEMENTS,
                &seed,
                fl_ev,
                joint_outlook::CompletionSampler::Sha256,
            )?,
            None => joint_outlook::sampled_joint_block(
                board,
                unseen,
                samples,
                FL14_JOINT_ARRANGEMENTS,
                &seed,
                fl_ev,
                joint_outlook::CompletionSampler::Sha256,
            )?,
        };
        for value in block {
            out.push(value as f32);
        }
        evaluator::fl14_context_block(unseen, out);
        if model.input_dim == FL14_FEATURE_SIZE {
            evaluator::allocation_rank_block(&board.rows, out);
        }
    } else {
        if model.input_dim == 109 {
            for value in t3_second::joint_block(board, unseen, fl_ev)? {
                out.push(value as f32);
            }
        }
        super::t3_vs_fl::fl_context(unseen, opp_count, fl_ev, out);
    }
    if out.len() != model.input_dim {
        bail!(
            "playout feature width {} does not match model input {}",
            out.len(),
            model.input_dim
        );
    }
    Ok(())
}

/// The cards a T2 decision is taken with already on the board: T0 placed five
/// and T1 placed two, so the draw the fence pre-ranks lands on a seven-card
/// board.  Named because the fence must fire at exactly one street and a bare
/// `7` in the middle of `choose` would not say which.
pub(crate) const T2_BOARD_CARDS: usize = 7;

/// The `topk` candidate indices by fence score, in the field's own order.
///
/// Two properties the fence rests on, both of them here rather than inline so
/// they can be tested without a model:
///
///   * **Deterministic under ties.**  Equal scores keep the lower index, so a
///     rerun cuts the same field.  `total_cmp` rather than `partial_cmp`: a
///     NaN from a model would otherwise make the comparator intransitive and
///     the surviving set arbitrary.  A NaN is ordered below every number
///     rather than at `total_cmp`'s own position above `+inf`, so a broken
///     fence score loses the cut the same way it loses the strict `>` argmax
///     downstream -- a fence must not be able to promote what the evaluator
///     would refuse.
///   * **Order-preserving.**  The survivors come back ascending, so the
///     evaluator's second pass walks them in the same relative order it would
///     have walked the whole field -- which is what makes a K at or above the
///     field size a no-op rather than a reshuffle, and keeps the strict `>`
///     argmax breaking ties the way the unfenced chain breaks them.
pub(crate) fn fence_topk(scores: &[f32], topk: usize) -> Vec<usize> {
    let rank = |score: f32| -> f32 {
        if score.is_nan() {
            f32::NEG_INFINITY
        } else {
            score
        }
    };
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| rank(scores[*b]).total_cmp(&rank(scores[*a])).then(a.cmp(b)));
    order.truncate(topk);
    order.sort_unstable();
    order
}

/// The board the model would choose from this draw.
///
/// With `context.t2_fence` set and the board at T2, the field is cut by the
/// cheap fence before `model` sees it; see [`Context::t2_fence`].  Everything
/// else -- the enumeration order, the memo, the strict `>` argmax -- is
/// untouched, so an unfenced call emits the same bits it always did.
fn choose(
    model: &evaluator::Model,
    board: &CoreBoard,
    draw: &[Card; 3],
    unseen: &[Card],
    context: &Context<'_>,
    pool_key: u64,
    node_seed: &str,
    features: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) -> Result<(Option<CoreBoard>, f32)> {
    let mut best_score = f32::NEG_INFINITY;
    let mut chosen: Option<CoreBoard> = None;
    // One memo for the node, not one per candidate: they complete the same
    // pool from the same board and differ only by where two cards went.  The
    // fence pass shares it: it is a pure cache keyed on the placement, so a
    // width that reads it only warms it for the pass that follows, and the
    // width-120 fence never reaches it at all.
    let mut joint_memo = TerminalMemo::new(board);
    let field = candidates(board, draw);

    // The pre-ranking pass.  Skipped outright when the field already fits
    // under K, so the fence can only ever remove work.
    let shortlist: Option<Vec<usize>> = match context.t2_fence {
        Some((fence, topk)) if board.card_count() == T2_BOARD_CARDS && field.len() > topk => {
            let mut scores: Vec<f32> = Vec::with_capacity(field.len());
            for candidate in &field {
                let mut next = board.clone();
                next.rows[candidate.placements[0].0].push(candidate.placements[0].1);
                next.rows[candidate.placements[1].0].push(candidate.placements[1].1);
                encode_for(
                    fence,
                    &next,
                    unseen,
                    context.opp_count,
                    context.fl_ev,
                    context.fl_table,
                    &context.rowwise_memo,
                    pool_key,
                    node_seed,
                    features,
                    Some((&mut joint_memo, &candidate.placements)),
                )?;
                scores.push(fence.predict(features, scratch));
            }
            Some(fence_topk(&scores, topk))
        }
        _ => None,
    };

    for (index, candidate) in field.iter().enumerate() {
        if let Some(keep) = &shortlist {
            // Linear over a list of at most K, and reached only on the fenced
            // path: unfenced, this is a `None` test in front of the loop body
            // that always was.
            if !keep.contains(&index) {
                continue;
            }
        }
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
            node_seed,
            features,
            Some((&mut joint_memo, &candidate.placements)),
        )?;
        let predicted = model.predict(features, scratch);
        if predicted > best_score {
            best_score = predicted;
            chosen = Some(next);
        }
    }
    Ok((chosen, best_score))
}

/// The opponents that survive everything hero has seen on this line, before a
/// T4 draw narrows them again.  Two stages because the line-level filter is
/// paid once per leaf and the draw-level one once per T4 draw.
enum Shortlist<'a> {
    Library {
        library: &'a FlLibrary,
        indices: Vec<u32>,
    },
    /// Hero's joker count travels with the entries: the pool counts jokers
    /// rather than masking them, so the per-draw refinement has to add the
    /// draw's count to the line's rather than test the draw's alone.
    Pool {
        entries: Vec<&'a PoolEntry>,
        hero_jokers: u32,
    },
}

/// Mask and joker count of a card set in the pool's terms.
///
/// Not `card_bit`: the pool's 52-bit order is rank-major where the library's is
/// suit-major, and its jokers are counted rather than given bits, which is the
/// distinction that makes two hero jokers filter correctly.  Routed through
/// `fl_solver::pool::natural_bit` so the order cannot drift apart from the
/// pool file's.
pub(crate) fn pool_mask_of(cards: &[Card]) -> (u64, u32) {
    let mut naturals = 0u64;
    let mut jokers = 0u32;
    for card in cards {
        let converted = fl_solver::Card {
            rank: card.rank,
            suit: card.suit,
        };
        match fl_solver::pool::natural_bit(&converted) {
            Some(bit) => naturals |= bit,
            None => jokers += 1,
        }
    }
    (naturals, jokers)
}

/// The four-width table `vs_fl` prices against, from this crate's config.
///
/// A pool pins the same four numbers in its header and refuses to load under
/// any others, so a leaf scoring under a table the pool was not built for is a
/// load error rather than a quiet re-pricing.
fn fl_ev_table(fl_ev: &FlEv) -> [f64; 4] {
    [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ]
}

/// Mean of hero's score over drawn opponents, each best-responding to this
/// exact board.
///
/// Empty is scored as zero rather than divided by, mirroring `score_mean`: the
/// draw loop already skips a draw whose filter matched nothing, so this only
/// ever guards the arithmetic.
fn pool_score_mean(terminal: &Terminal, drawn: &[&PoolEntry], fl_ev: &[f64; 4]) -> f64 {
    let hero = HeroTerminal {
        busted: terminal.busted,
        top: terminal.values[0],
        mid: terminal.values[1],
        bot: terminal.values[2],
        royalty: terminal.royalty,
        entry_width: terminal.fl_card_count,
    };
    let mut total = 0.0f64;
    for entry in drawn {
        total += vs_fl::hero_score(&hero, &entry.rows, fl_ev);
    }
    total / drawn.len().max(1) as f64
}

/// Exact T4 continuation value of an 11-card board: mean over T4 draws of the
/// best completion's score, on an already-filtered shortlist.
fn terminal_value(
    board: &CoreBoard,
    unseen: &[Card],
    shortlist: &Shortlist<'_>,
    context: &Context<'_>,
    seed: &str,
) -> Result<(f64, f64)> {
    if context.own_only && context.t4_draw_sample == 0 {
        // Exact T4 under the own-hand objective, through the pair table: one
        // C(n,2) sweep answers every C(n,3) draw, the same arithmetic the T3
        // teacher's `completion_value` is trusted for.  The sampled loop
        // below reaches the same expectation in about twenty-seven times the
        // work, which is what made `--t4-draw-sample 0` unaffordable here.
        let to_fl = |cards: &[Card]| -> Vec<fl_solver::Card> {
            cards
                .iter()
                .map(|c| fl_solver::Card {
                    rank: c.rank,
                    suit: c.suit,
                })
                .collect()
        };
        let rows: [Vec<fl_solver::Card>; 3] = [
            to_fl(&board.rows[0]),
            to_fl(&board.rows[1]),
            to_fl(&board.rows[2]),
        ];
        let table = fl_ev_table(context.fl_ev);
        let exact =
            fl_solver::t3_labels::completion_value(&rows, &to_fl(unseen), &[], &table, true, None);
        return Ok((exact.mean(), 1.0));
    }
    let draw_sets = sampled_draws(unseen.len(), context.t4_draw_sample, seed);
    let patterns = t3_second::placement_patterns(board);
    let mut terminal_memo = TerminalMemo::new(board);
    let mut memo: std::collections::HashMap<TerminalKey, f64> =
        std::collections::HashMap::with_capacity(patterns.len());
    let mut matched: Vec<MatchedRow> = Vec::with_capacity(match shortlist {
        Shortlist::Library { indices, .. } => indices.len(),
        Shortlist::Pool { .. } => 0,
    });
    let mut drawn: Vec<&PoolEntry> = Vec::with_capacity(match shortlist {
        Shortlist::Library { .. } => 0,
        Shortlist::Pool { entries, .. } => entries.len(),
    });
    let table = fl_ev_table(context.fl_ev);
    let mut total = 0.0f64;
    let mut sample_total = 0usize;
    for draw in &draw_sets {
        let draw_cards = [unseen[draw[0]], unseen[draw[1]], unseen[draw[2]]];
        let opponents = if context.own_only {
            // No opponent is consulted; one keeps the draw in the mean and
            // keeps `mean_fl_samples` distinguishable from the bootstrap's
            // hard 0.0.
            1
        } else { match shortlist {
            Shortlist::Library { library, indices } => {
                let mut draw_mask = 0u64;
                for card in &draw_cards {
                    draw_mask |= card_bit(card)?;
                }
                matched.clear();
                for &index in indices {
                    let index = index as usize;
                    if library.masks[index] & draw_mask == 0 {
                        matched.push(MatchedRow {
                            values: library.values[index],
                            royalty: library.royalty[index],
                            stay: library.stay[index],
                            busted: library.busted[index],
                        });
                    }
                }
                matched.len()
            }
            Shortlist::Pool {
                entries,
                hero_jokers,
            } => {
                let (draw_naturals, draw_jokers) = pool_mask_of(&draw_cards);
                drawn.clear();
                for entry in entries {
                    if entry.naturals & draw_naturals == 0
                        && entry.jokers + hero_jokers + draw_jokers <= 2
                    {
                        drawn.push(entry);
                    }
                }
                drawn.len()
            }
        } };
        if opponents == 0 {
            continue;
        }
        sample_total += opponents;
        let mut best = f64::NEG_INFINITY;
        memo.clear();
        for pattern in &patterns {
            let terminal = terminal_memo.terminal(&[
                (pattern.rows[0], draw_cards[pattern.cards[0]]),
                (pattern.rows[1], draw_cards[pattern.cards[1]]),
            ]);
            // Both leaves read exactly the terminal key and nothing else, so
            // memoizing over it stays exact for either.
            let key = terminal_key(&terminal);
            let value = match memo.get(&key) {
                Some(cached) => *cached,
                None => {
                    let computed = if context.own_only {
                        let hero = HeroTerminal {
                            busted: terminal.busted,
                            top: terminal.values[0],
                            mid: terminal.values[1],
                            bot: terminal.values[2],
                            royalty: terminal.royalty,
                            entry_width: terminal.fl_card_count,
                        };
                        vs_fl::hero_own(&hero, &table)
                    } else {
                        match shortlist {
                            Shortlist::Library { .. } => {
                                score_mean(&terminal, &matched, context.opp_count, context.fl_ev)
                            }
                            Shortlist::Pool { .. } => pool_score_mean(&terminal, &drawn, &table),
                        }
                    };
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
        let shortlist = match &context.opponents {
            Opponents::Library(library) => {
                // card_bit sets both joker bits for either joker, since the
                // library masks distinguish X1 from X2 and a mask test only
                // asks "is this physical card taken".  With exactly one joker
                // seen the other is still available, so its bit is reopened
                // here.
                let mut mask = seen_mask;
                if jokers_seen == 1 {
                    mask &= !(1u64 << 53);
                }
                Shortlist::Library {
                    library,
                    indices: (0..library.len() as u32)
                        .filter(|&index| library.masks[index as usize] & mask == 0)
                        .collect(),
                }
            }
            Opponents::Pool(entries) => {
                // The pool's disjointness test wants every card hero can see on
                // this line, and `unseen` is the complement of exactly that --
                // the T4 draw is still in it, and stays there for the per-draw
                // refinement.  Deriving the mask here rather than threading a
                // second one keeps the two orders from having to agree on the
                // way down; the joker halves agreeing is asserted instead.
                let (unseen_naturals, unseen_jokers) = pool_mask_of(unseen);
                let hero_jokers = 2u32.saturating_sub(unseen_jokers);
                debug_assert_eq!(
                    hero_jokers as usize, jokers_seen,
                    "the unseen list is no longer the complement of the line's seen cards"
                );
                Shortlist::Pool {
                    entries: entries
                        .iter()
                        .copied()
                        .filter(|entry| {
                            entry.compatible(POOL_NATURALS & !unseen_naturals, hero_jokers)
                        })
                        .collect(),
                    hero_jokers,
                }
            }
        };
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
        let (chosen, best_score) = choose(
            model,
            board,
            &draw,
            &next_unseen,
            context,
            seed_hash(&child_seed),
            &child_seed,
            features,
            scratch,
        )?;
        let Some(next_board) = chosen else {
            continue;
        };
        if context.truncate_depth == Some(depth) {
            // The chooser already scored every candidate; taking its best is
            // free, where playing the line out is the whole cost.
            total += best_score as f64;
            lines += 1;
            continue;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{all_cards, terminal_of, to_core_card};
    use fl_solver::frontier::FrontierEntry;

    /// The table `D:/ofc_data/fl_pools/fl14_v1.jfl1` pins in its header, so a
    /// fixture entry is priced the way the shipped pool's entries are.
    const TABLE: [f64; 4] = [0.0, 10.7, 29.9, 63.5];

    fn fl_ev() -> FlEv {
        let mut by_card_count = std::collections::BTreeMap::new();
        for (slot, count) in [14u8, 15, 16, 17].into_iter().enumerate() {
            by_card_count.insert(count, TABLE[slot]);
        }
        FlEv {
            by_card_count,
            config_sha256: String::new(),
        }
    }

    /// One Fantasyland hand's frontier, built rather than solved.
    ///
    /// `build_frontier` sweeps about a million arrangements per hand, which is
    /// minutes in a debug test binary; these tests are about the leaf's
    /// filtering and arithmetic, and a frontier written out by hand exercises
    /// both while keeping the opponent's choice visible in the fixture.
    /// `static_value` is royalty alone because `fl_ev[0]` is zero, which is
    /// exactly what `pool::deserialize` computes for a width-14 pool.
    fn entry(naturals: u64, jokers: u32, rows: &[(u32, u32, u32, i32, bool)]) -> PoolEntry {
        PoolEntry {
            naturals,
            jokers,
            rows: rows
                .iter()
                .map(|&(top, mid, bot, royalty, stays)| FrontierEntry {
                    top,
                    mid,
                    bot,
                    royalty,
                    stays,
                    static_value: royalty as f64 + if stays { TABLE[0] } else { 0.0 },
                })
                .collect(),
        }
    }

    /// An 11-card hero board with two open slots, joker-free so a failure is
    /// never about the line's joker accounting.
    fn hero_board() -> CoreBoard {
        let row = |names: &[&str]| -> Vec<Card> {
            names
                .iter()
                .map(|name| to_core_card(name).unwrap())
                .collect()
        };
        CoreBoard {
            rows: [
                row(&["Ks", "Kh", "2c"]),
                row(&["Qs", "Qh", "7d", "3c"]),
                row(&["As", "Ah", "Ad", "9c"]),
            ],
        }
    }

    fn unseen_for(board: &CoreBoard) -> Vec<Card> {
        let placed: Vec<Card> = board.rows.iter().flatten().copied().collect();
        all_cards()
            .iter()
            .map(|name| to_core_card(name).unwrap())
            .filter(|card| !placed.contains(card))
            .collect()
    }

    /// A pool whose hands are drawn from what hero cannot see, so every entry
    /// survives the line filter and the per-draw filter is what decides.
    fn fixture_pool(unseen: &[Card]) -> Vec<PoolEntry> {
        let naturals: Vec<Card> = unseen.iter().copied().filter(|c| !c.is_joker()).collect();
        (0..6usize)
            .map(|index| {
                let hand: Vec<Card> = (0..14)
                    .map(|slot| naturals[(index * 7 + slot * 3) % naturals.len()])
                    .collect();
                let (mask, _) = pool_mask_of(&hand);
                // A frontier the opponent has to choose within: a big-royalty
                // arrangement with weak rows, and a flat one with strong rows.
                entry(
                    mask,
                    (index % 3 == 0) as u32,
                    &[
                        (100 + index as u32, 200, 300, 12, true),
                        (400, 500, 600 + index as u32, 2, false),
                        (250, 450, 650, 7, false),
                    ],
                )
            })
            .collect()
    }

    fn leaf(
        board: &CoreBoard,
        unseen: &[Card],
        entries: &[&PoolEntry],
        draws: usize,
        seed: &str,
    ) -> (f64, f64) {
        let fl_ev = fl_ev();
        let fl_table: evaluator::FlTable = [0.0, 10.7, 29.9, 63.5];
        // No models left to descend through, so `descend` prices the terminal
        // on entry -- the leaf under test with nothing else in front of it.
        let context = Context {
            fl_ev: &fl_ev,
            opponents: Opponents::Pool(entries),
            fl_table: &fl_table,
            models: &[],
            samples: &[],
            opp_count: 14,
            t4_draw_sample: draws,
            rowwise_memo: Mutex::new(HashMap::new()),
            truncate_depth: None,
            own_only: false,
            t2_fence: None,
        };
        let mut features: Vec<f32> = Vec::new();
        let mut scratch: Vec<f32> = Vec::new();
        descend(
            board,
            unseen,
            0,
            0,
            &context,
            0,
            seed,
            &mut features,
            &mut scratch,
        )
        .expect("terminal leaf")
    }

    /// **The bust branch.** With hero's rows dead the opponent's choice cannot
    /// depend on them, so it takes its best `static_value` and hero pays the
    /// foul plus that arrangement's royalty -- not the frontier's largest
    /// royalty, which is a different number as soon as `fl_ev` stops being zero.
    #[test]
    fn a_busted_hero_pays_the_foul_and_the_opponents_best_static_arrangement() {
        let unseen = unseen_for(&hero_board());
        let pool = fixture_pool(&unseen);
        let entries: Vec<&PoolEntry> = pool.iter().collect();
        let expected = entries
            .iter()
            .map(|entry| {
                let best = entry
                    .rows
                    .iter()
                    .max_by(|a, b| a.static_value.partial_cmp(&b.static_value).unwrap())
                    .expect("non-empty frontier");
                -6.0 - best.royalty as f64
            })
            .sum::<f64>()
            / entries.len() as f64;

        let busted = Terminal {
            busted: true,
            royalty: 0,
            fl_card_count: 0,
            values: [0, 0, 0],
        };
        assert_eq!(
            pool_score_mean(&busted, &entries, &TABLE).to_bits(),
            expected.to_bits(),
            "busted leaf priced at {} not {expected}",
            pool_score_mean(&busted, &entries, &TABLE)
        );
        // Nothing about hero's dead board can move that price, including a
        // royalty and a Fantasyland entry the foul has already cancelled.
        let dressed = Terminal {
            busted: true,
            royalty: 22,
            fl_card_count: 17,
            values: [9_999, 9_999, 9_999],
        };
        assert_eq!(
            pool_score_mean(&dressed, &entries, &TABLE).to_bits(),
            expected.to_bits(),
            "the bust price moved with hero's dead rows"
        );
    }

    /// **The leaf's identity.** Mean over T4 draws of the best completion's
    /// mean score against the opponents that draw leaves compatible.
    ///
    /// Recomputed here without the terminal memo and without the per-draw
    /// gather, so a memo that returned a stale value or a filter that kept the
    /// wrong entries would show up as a different number rather than as the
    /// same bug twice.
    #[test]
    fn the_leaf_is_the_mean_over_draws_of_the_best_completion_against_the_pool() {
        let board = hero_board();
        let unseen = unseen_for(&board);
        let pool = fixture_pool(&unseen);
        let entries: Vec<&PoolEntry> = pool.iter().collect();
        let seed = "playout/pool/leaf";
        let draws = 4usize;
        let (value, samples) = leaf(&board, &unseen, &entries, draws, seed);

        let (unseen_naturals, unseen_jokers) = pool_mask_of(&unseen);
        assert_eq!(unseen_jokers, 2, "the fixture board is joker-free");
        let hero_naturals = POOL_NATURALS & !unseen_naturals;
        let shortlist: Vec<&PoolEntry> = entries
            .iter()
            .copied()
            .filter(|entry| entry.compatible(hero_naturals, 0))
            .collect();
        assert_eq!(
            shortlist.len(),
            entries.len(),
            "the fixture pool was meant to survive the line filter intact"
        );

        let draw_sets = sampled_draws(unseen.len(), draws, seed);
        let patterns = t3_second::placement_patterns(&board);
        let mut total = 0.0f64;
        let mut matched_total = 0usize;
        let mut filtered_any = false;
        for draw in &draw_sets {
            let cards = [unseen[draw[0]], unseen[draw[1]], unseen[draw[2]]];
            let (draw_naturals, draw_jokers) = pool_mask_of(&cards);
            let matched: Vec<&PoolEntry> = shortlist
                .iter()
                .copied()
                .filter(|entry| {
                    entry.naturals & draw_naturals == 0 && entry.jokers + draw_jokers <= 2
                })
                .collect();
            filtered_any |= matched.len() < shortlist.len();
            if matched.is_empty() {
                continue;
            }
            matched_total += matched.len();
            let mut best = f64::NEG_INFINITY;
            for pattern in &patterns {
                let mut completed = board.clone();
                completed.rows[pattern.rows[0]].push(cards[pattern.cards[0]]);
                completed.rows[pattern.rows[1]].push(cards[pattern.cards[1]]);
                let terminal = terminal_of(&completed);
                let hero = HeroTerminal {
                    busted: terminal.busted,
                    top: terminal.values[0],
                    mid: terminal.values[1],
                    bot: terminal.values[2],
                    royalty: terminal.royalty,
                    entry_width: terminal.fl_card_count,
                };
                let mean = matched
                    .iter()
                    .map(|entry| vs_fl::hero_score(&hero, &entry.rows, &TABLE))
                    .sum::<f64>()
                    / matched.len() as f64;
                if mean > best {
                    best = mean;
                }
            }
            total += best;
        }
        let expected = total / draw_sets.len().max(1) as f64;

        assert!(
            matched_total > 0,
            "no opponent survived any draw; the fixture proves nothing"
        );
        assert!(
            filtered_any,
            "no draw ever removed an entry; the per-draw filter is untested"
        );
        assert_eq!(
            value.to_bits(),
            expected.to_bits(),
            "leaf gave {value}, recomputation {expected}"
        );
        assert_eq!(
            samples.to_bits(),
            (matched_total as f64 / draw_sets.len() as f64).to_bits(),
            "reported opponents per draw disagrees with the recomputation"
        );
    }

    /// A leaf whose opponents are all incompatible is worth zero, not NaN.
    #[test]
    fn an_all_incompatible_pool_is_zero_rather_than_a_division() {
        let board = hero_board();
        let unseen = unseen_for(&board);
        // Every entry claims a card hero is holding, so nothing survives the
        // line filter and every draw finds an empty matched set.
        let (held, _) = pool_mask_of(&board.rows[0]);
        let blocked: Vec<PoolEntry> = (0..4i32)
            .map(|index| entry(held, 0, &[(1, 1, 1, index, false)]))
            .collect();
        let entries: Vec<&PoolEntry> = blocked.iter().collect();
        let (value, samples) = leaf(&board, &unseen, &entries, 3, "playout/pool/empty");
        assert_eq!(value, 0.0, "an empty match set priced the leaf at {value}");
        assert_eq!(samples, 0.0);

        // And the mean itself divides by one rather than by zero.
        let terminal = Terminal {
            busted: false,
            royalty: 4,
            fl_card_count: 14,
            values: [100, 200, 300],
        };
        assert_eq!(pool_score_mean(&terminal, &[], &TABLE), 0.0);
    }

    /// A seven-card board: the position a T2 decision is taken from, and the
    /// only card count the fence is allowed to fire at.  Two open slots in the
    /// top row would collapse the field, so the rows are left uneven.
    fn t2_board() -> CoreBoard {
        let row = |names: &[&str]| -> Vec<Card> {
            names
                .iter()
                .map(|name| to_core_card(name).unwrap())
                .collect()
        };
        CoreBoard {
            rows: [
                row(&["Ks", "Kh"]),
                row(&["Qs", "7d", "3c"]),
                row(&["As", "9c"]),
            ],
        }
    }

    /// A one-layer linear model of the given width, assembled as an image so
    /// it arrives through exactly the reader a shipped model arrives through.
    /// `seed` moves every weight, which is what lets a fence rank a field
    /// differently from the evaluator it stands in front of.
    fn linear_model(input_dim: usize, seed: u32) -> evaluator::Model {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"T4F1");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one layer
        bytes.extend_from_slice(&(input_dim as u32).to_le_bytes());
        for _ in 0..input_dim {
            bytes.extend_from_slice(&0.0f32.to_le_bytes()); // mean
        }
        for _ in 0..input_dim {
            bytes.extend_from_slice(&1.0f32.to_le_bytes()); // std
        }
        bytes.extend_from_slice(&(input_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for index in 0..input_dim as u32 {
            let mixed = index.wrapping_mul(2_654_435_761).wrapping_add(seed);
            bytes.extend_from_slice(&((mixed % 2_003) as f32 / 1_001.0 - 1.0).to_le_bytes());
        }
        bytes.extend_from_slice(&0.0f32.to_le_bytes()); // bias
        evaluator::Model::load(&bytes).expect("synthetic model image")
    }

    /// A context carrying nothing but what `choose` reads.  No opponent is
    /// ever consulted on this path: the fence decides which candidates get
    /// encoded, and encoding is the whole of what it changes.
    fn fence_context<'a>(
        fl_ev: &'a FlEv,
        fl_table: &'a evaluator::FlTable,
        fence: Option<(&'a evaluator::Model, usize)>,
    ) -> Context<'a> {
        Context {
            fl_ev,
            opponents: Opponents::Pool(&[]),
            fl_table,
            models: &[],
            samples: &[],
            opp_count: 14,
            t4_draw_sample: 0,
            rowwise_memo: Mutex::new(HashMap::new()),
            truncate_depth: None,
            own_only: true,
            t2_fence: fence,
        }
    }

    /// **The cut is a top-K, taken in descending score and handed back in the
    /// field's own order.**
    ///
    /// Both halves matter: descending is what makes it a fence, ascending is
    /// what makes the second pass walk the survivors in the order the unfenced
    /// loop would have walked them -- so a strict `>` argmax breaks its ties
    /// the same way either side of the cut.
    #[test]
    fn the_fence_keeps_the_top_k_in_the_fields_own_order() {
        assert_eq!(fence_topk(&[1.0, 1.5, 0.5, 2.0, 1.0], 3), vec![0, 1, 3]);
        assert_eq!(fence_topk(&[1.0, 1.5, 0.5, 2.0, 1.0], 1), vec![3]);
        // Negative scores are ordinary scores, not an absent candidate.
        assert_eq!(fence_topk(&[-9.0, -1.0, -5.0], 2), vec![1, 2]);
    }

    /// **A tie keeps the lower index, so a rerun cuts the same field.**
    ///
    /// An all-equal field is the case a naive comparator gets away with until
    /// the sort implementation changes underneath it; pinning it here means a
    /// fenced run is reproducible rather than reproducible-so-far.
    #[test]
    fn a_tied_fence_cuts_by_index_and_not_by_luck() {
        assert_eq!(fence_topk(&[1.0, 1.0, 1.0, 1.0], 2), vec![0, 1]);
        assert_eq!(fence_topk(&[2.0, 1.0, 2.0, 1.0, 2.0], 2), vec![0, 2]);
        // A NaN cannot make the surviving set arbitrary, and cannot promote
        // the candidate carrying it: it is ordered below every number, which
        // is where the strict `>` argmax downstream would put it too.
        assert_eq!(fence_topk(&[f32::NAN, 3.0, 1.0], 2), vec![1, 2]);
        assert_eq!(fence_topk(&[f32::NAN, -3.0, -1.0], 1), vec![2]);
    }

    /// **A K at or above the field size is a no-op, not a reshuffle.**
    ///
    /// The identity permutation is what the bit-identity claim rests on: with
    /// nothing to cut, the second pass is the original loop.
    #[test]
    fn a_fence_wider_than_the_field_selects_everything_in_order() {
        for topk in 3..8usize {
            assert_eq!(fence_topk(&[0.5, 2.0, 1.0], topk), vec![0, 1, 2]);
        }
        assert!(fence_topk(&[1.0, 2.0], 0).is_empty());
        assert!(fence_topk(&[], 4).is_empty());
    }

    /// **A fence that cannot bite leaves the decision bit-identical.**
    ///
    /// Three ways of not biting, all against the same unfenced run: no fence,
    /// a K the field already fits under, and a fence that IS the evaluator --
    /// the last one actually runs the pre-ranking pass and cuts the field, so
    /// it pins the cut itself rather than only the guard in front of it.  The
    /// score is compared by bits because a fence that moved a decision by an
    /// ulp would still be a fence that moved a decision.
    #[test]
    fn a_fence_that_cannot_bite_changes_no_decision() {
        let fl_ev = fl_ev();
        let fl_table: evaluator::FlTable = [0.0, 10.7, 29.9, 63.5];
        let board = t2_board();
        assert_eq!(board.card_count(), T2_BOARD_CARDS);
        let draw = [
            to_core_card("Td").unwrap(),
            to_core_card("4h").unwrap(),
            to_core_card("8s").unwrap(),
        ];
        let placed: Vec<Card> = board.rows.iter().flatten().copied().collect();
        let unseen: Vec<Card> = all_cards()
            .iter()
            .map(|name| to_core_card(name).unwrap())
            .filter(|card| !placed.contains(card) && !draw.contains(card))
            .collect();
        let field = candidates(&board, &draw).len();
        assert!(field > 4, "the fixture leaves no field to cut ({field})");

        // Width 60 is the cheapest encoder this crate dispatches on -- actor
        // block plus FL context, nothing sampled -- so the test measures the
        // fence and not the joint block.
        let served = linear_model(60, 0x1234_5678);
        let other = linear_model(60, 0x9E37_79B9);

        let run = |fence: Option<(&evaluator::Model, usize)>| -> ([Vec<Card>; 3], u32) {
            let context = fence_context(&fl_ev, &fl_table, fence);
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            let (chosen, best) = choose(
                &served,
                &board,
                &draw,
                &unseen,
                &context,
                2,
                "fence/t2",
                &mut features,
                &mut scratch,
            )
            .expect("a seven-card board has a legal placement");
            (chosen.expect("a chosen board").rows, best.to_bits())
        };

        let plain = run(None);
        assert_eq!(
            run(Some((&other, field))),
            plain,
            "a K at the field size cut something"
        );
        assert_eq!(
            run(Some((&other, field + 5))),
            plain,
            "a K above the field size cut something"
        );
        assert_eq!(
            run(Some((&served, 3))),
            plain,
            "the evaluator's own argmax did not survive its own cut"
        );
    }
}
