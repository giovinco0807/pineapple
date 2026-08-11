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
//!   109 = actor + rowwise + joint + FL context  (full T3 evaluator; the
//!         joint block needs exactly two open slots, so it is only ever
//!         reachable from an 11-card board)

use anyhow::{bail, Result};
use fl_solver::pool::PoolEntry;
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
use super::row_memo::TerminalMemo;
use super::t3_second;
use super::t3_vs_fl::sampled_draws;
use super::t3_vs_fl_lib::{card_bit, score_mean, terminal_key, FlLibrary, MatchedRow, TerminalKey};
use super::{CoreBoard, FlEv, Terminal};

/// The 52 natural cards, in the pool's rank-major bit order.
const POOL_NATURALS: u64 = (1u64 << 52) - 1;

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
    #[allow(dead_code)] // Constructed once the labelers thread a pool through.
    Pool(&'a [&'a PoolEntry]),
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
fn pool_mask_of(cards: &[Card]) -> (u64, u32) {
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
        let opponents = match shortlist {
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
        };
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
                    let computed = match shortlist {
                        Shortlist::Library { .. } => {
                            score_mean(&terminal, &matched, context.opp_count, context.fl_ev)
                        }
                        Shortlist::Pool { .. } => pool_score_mean(&terminal, &drawn, &table),
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
            names.iter().map(|name| to_core_card(name).unwrap()).collect()
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
}
