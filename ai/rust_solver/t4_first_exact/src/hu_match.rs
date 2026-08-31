//! Head-up matches: two chains playing the real interleaved game.
//!
//! The own-hand simulator plays each seat's whole hand independently, which
//! is exactly right when neither side looks across the table and exactly
//! wrong here.  In HU the seats alternate inside every street -- BB places,
//! BTN places seeing it -- so the hand is one loop over ten decisions, not
//! two loops over five.
//!
//! What plays each decision:
//!
//! | street | BB (acts first)         | BTN (acts second)        |
//! |--------|-------------------------|--------------------------|
//! | T0-T2  | HU model                | HU model                 |
//! | T3     | HU model                | **exact**: -V4(opp, own) |
//! | T4     | **exact** V4 on the draw| **closed form**          |
//!
//! The exact ends are not a luxury: they are cheaper than the models there
//! (0.8 ms a V4 state, a closed form is a max over nine placements), and
//! they keep the measurement from grading a model against its own teacher.
//!
//! An arm may be an HU chain or an own-hand chain, so the generations can be
//! seated across from each other and settled in real points -- the only
//! number in this tree that says which is stronger.

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;

use super::evaluator;
use super::hu_encode;
use super::playout;
use super::self_play::{settle, Finished};
use super::t0_policy;
use super::v4_first::{self, V4FirstRequest};
use super::{all_cards, to_core_card, BoardStr, Card, CoreBoard, FlEv, Terminal};

/// `fl_solver::pool::deal` speaks its own Card type; the two are identical
/// structs in different crates.
pub(crate) fn deal_names(seed: u64, index: u64, width: usize) -> Vec<fl_solver::Card> {
    fl_solver::pool::deal(seed, index, width)
}

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

/// One side's models.
///
/// Two families, because two situations: against a normal opponent the board
/// is public and the HU evaluators read it, but a Fantasyland opponent's
/// thirteen are **face down**, so the pair encoding has nothing to read and
/// the own-hand chain plays those hands.  An own-hand arm simply leaves `hu`
/// empty and plays every hand with the second set.
pub struct Arm<'a> {
    /// [street][seat] street/seat evaluators on the 207-dim pair vector;
    /// seat 0 is BB.  A street may be `None` -- the ladder is built from the
    /// bottom, so a partial chain is the normal state of affairs and the
    /// streets that have no HU model yet fall back to the own-hand chooser.
    pub hu: Vec<Option<[&'a evaluator::Model; 2]>>,
    /// T0/T1/T2 own-hand choosers, used against Fantasyland and as the whole
    /// chain when `hu` is empty.
    pub own: [&'a evaluator::Model; 3],
    /// [street][seat] shortlist rankers.  When a street has one and `topk`
    /// is set, every candidate is scored by the ranker at none of the
    /// joint's cost and only the top K meet the real evaluator -- measured
    /// at T0: K=4 keeps 0.0003 of the EV and 1/58th of the encoding.
    pub rankers: Vec<Option<[&'a evaluator::Model; 2]>>,
    pub topk: usize,
    /// The ranker's successor at exactly one node: T0, first actor.  Given
    /// one, the shortlist there is the policy's top `policy_topk` and the
    /// ranker is not called at all.  `None` leaves that node on the ranker.
    pub t0_policy: Option<&'a evaluator::Model>,
    pub policy_topk: usize,
    /// Serve-time joint samples; 0 keeps the trained 400.  Per arm so two
    /// sample counts can be seated against each other in one match.
    pub joint_samples: usize,
}

/// Score one candidate with whatever encoding the model asks for by width:
/// 623 = hybrid (no sampling), 207 with a tail = the full pair encoding,
/// 207 without = the joint-zeroed cheap layout the rankers train on.
#[allow(clippy::too_many_arguments)]
fn score_with(
    model: &evaluator::Model,
    own_board: &CoreBoard,
    opp_board: &CoreBoard,
    after: &[Vec<String>; 3],
    toss: &Option<String>,
    dead: &[String],
    pool_names: &[String],
    opp_rows: &[Vec<String>; 3],
    pool: &[Card],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    memo: &playout::RowwiseMemo,
    node_seed: &str,
    features: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    opp_tail: &[f32],
    samples: usize,
) -> Result<f32> {
    if model.input_dim == hu_encode::HYBRID_FEATURE_SIZE {
        let mut full_dead: Vec<String> = dead.to_vec();
        if let Some(card) = toss {
            full_dead.push(card.clone());
        }
        hu_encode::encode_hybrid(
            after, opp_rows, &full_dead, pool_names, own_board, opp_board,
            pool, fl_ev, fl_table, memo, 0, features,
        );
    } else if opp_tail.is_empty() {
        hu_encode::encode_cheap207(
            own_board, opp_board, pool, fl_ev, fl_table, memo, 0, features,
        );
    } else {
        hu_encode::encode_pair_with_tail(
            own_board, opp_tail, pool, fl_ev, fl_table, memo, 0, node_seed, samples,
            features,
        )?;
    }
    Ok(model.predict(features, scratch))
}

fn board_of(rows: &[Vec<String>; 3]) -> Result<CoreBoard> {
    CoreBoard::from_str_board(&BoardStr {
        top: rows[0].clone(),
        middle: rows[1].clone(),
        bottom: rows[2].clone(),
    })
}

fn finish_of(rows: &[Vec<String>; 3]) -> Result<Finished> {
    let board = board_of(rows)?;
    let terminal: Terminal = super::terminal_of(&board);
    Ok(Finished {
        busted: terminal.busted,
        top: terminal.values[0],
        mid: terminal.values[1],
        bot: terminal.values[2],
        royalty: terminal.royalty,
        entry_width: terminal.fl_card_count,
    })
}

/// Placements of two of three drawn cards, deduplicated on the board they
/// reach plus what they threw.
fn placements(
    board: &[Vec<String>; 3],
    draw: &[String],
) -> Vec<([Vec<String>; 3], String)> {
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for toss in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|k| *k != toss).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|r| board[r].len() + need[r] > ROW_CAPACITY[r]) {
                    continue;
                }
                let mut after = board.clone();
                after[row_a].push(draw[kept[0]].clone());
                after[row_b].push(draw[kept[1]].clone());
                let mut key: Vec<String> = after
                    .iter()
                    .map(|row| {
                        let mut sorted = row.clone();
                        sorted.sort();
                        sorted.join(",")
                    })
                    .collect();
                key.push(draw[toss].clone());
                if seen.insert(key.join("|")) {
                    out.push((after, draw[toss].clone()));
                }
            }
        }
    }
    out
}

/// Every distinct arrangement of the opening five.
pub(crate) fn openings(draw: &[String]) -> Vec<[Vec<String>; 3]> {
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Assign each of the five cards to one of three rows, respecting capacity.
    for mask in 0..3usize.pow(5) {
        let mut rows: [Vec<String>; 3] = Default::default();
        let mut code = mask;
        let mut legal = true;
        for card in draw.iter().take(5) {
            let row = code % 3;
            code /= 3;
            rows[row].push(card.clone());
            if rows[row].len() > ROW_CAPACITY[row] {
                legal = false;
                break;
            }
        }
        if !legal {
            continue;
        }
        let key: Vec<String> = rows
            .iter()
            .map(|row| {
                let mut sorted = row.clone();
                sorted.sort();
                sorted.join(",")
            })
            .collect();
        if seen.insert(key.join("|")) {
            out.push(rows);
        }
    }
    out
}

struct Seat {
    board: [Vec<String>; 3],
    dead: Vec<String>,
}

/// One seat's turn, recorded so the hand can be read rather than scored.
///
/// Every other consumer of this harness wants a number.  This exists because
/// a placement is the thing being judged, and a chain that scores well while
/// placing badly is a claim nobody can check against a settlement column.
#[derive(serde::Serialize, Clone)]
pub struct TraceStep {
    pub street: usize,
    /// 0 is BB, who acts first on every street.
    pub seat: usize,
    pub draw: Vec<String>,
    pub board: [Vec<String>; 3],
    pub discard: Option<String>,
    /// What chose it: "model", "exact-v4", or "closed-form".
    pub by: &'static str,
}

#[derive(serde::Serialize)]
pub struct TracedHand {
    pub hand: u64,
    pub steps: Vec<TraceStep>,
    pub bb_busted: bool,
    pub btn_busted: bool,
    pub bb_royalty: i32,
    pub btn_royalty: i32,
    pub bb_entry: u8,
    pub btn_entry: u8,
    /// Settlement from BB's side, lines plus royalties.
    pub bb_points: i32,
}

/// The unseen pool from one seat's view: the deck minus both boards, its own
/// dead and its own draw.  Jokers are counted, never name-matched.
fn pool_of(
    own: &[Vec<String>; 3],
    opp: &[Vec<String>; 3],
    dead: &[String],
    draw: &[String],
) -> Result<(Vec<String>, Vec<Card>)> {
    let mut naturals: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut jokers = 0usize;
    for name in own
        .iter()
        .flatten()
        .chain(opp.iter().flatten())
        .chain(dead)
        .chain(draw)
    {
        if name.starts_with('X') {
            jokers += 1;
        } else if !naturals.insert(name) {
            bail!("card {name} appears twice");
        }
    }
    let mut names: Vec<String> = all_cards()
        .into_iter()
        .filter(|n| !n.starts_with('X') && !naturals.contains(n.as_str()))
        .collect();
    for slot in 0..(2usize.saturating_sub(jokers)) {
        names.push(format!("X{}", slot + 1));
    }
    let cards = names
        .iter()
        .map(|n| to_core_card(n))
        .collect::<Result<Vec<_>>>()?;
    Ok((names, cards))
}

/// The exact closed form for the last actor: it sees thirteen opposing cards.
fn closed_form_t4(
    own: &[Vec<String>; 3],
    draw: &[String],
    opponent: &Finished,
    table: &[f64; 4],
) -> Result<[Vec<String>; 3]> {
    let mut best = f64::NEG_INFINITY;
    let mut chosen = None;
    for (after, _toss) in placements(own, draw) {
        let mine = finish_of(&after)?;
        let entry = |f: &Finished| -> f64 {
            if !f.busted && f.entry_width >= 14 {
                table[(f.entry_width - 14) as usize]
            } else {
                0.0
            }
        };
        let score = settle(&mine, opponent) as f64 + entry(&mine) - entry(opponent);
        if score > best {
            best = score;
            chosen = Some(after);
        }
    }
    chosen.ok_or_else(|| anyhow!("no legal T4 placement"))
}

/// Placements the harness must take as given, keyed by (street, seat).
pub type Forced = std::collections::HashMap<(usize, usize), [Vec<String>; 3]>;

/// One hand.  Returns both finished boards.
///
/// `forced` pins the placement at any (street, seat) instead of consulting a
/// model: a deep evaluator judges a *placement*, so the placement under test
/// -- and the history that led to it -- must be the one thing the harness
/// never chooses for itself.  The discard is inferred as the drawn card the
/// forced board does not contain.
#[allow(clippy::too_many_arguments)]
fn play_hand(
    id: &str,
    dealt: &[fl_solver::Card],
    arms: &[&Arm<'_>; 2],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    table: &[f64; 4],
    mut trace: Option<&mut Vec<TraceStep>>,
    forced: Option<&Forced>,
) -> Result<[Finished; 2]> {
    // Seat 0 is BB (acts first every street), seat 1 is BTN.
    let mut names: [Vec<String>; 2] = Default::default();
    let mut jokers = 0usize;
    for seat in 0..2usize {
        let start = seat * 17;
        names[seat] = dealt[start..start + 17]
            .iter()
            .map(|card| {
                if card.is_joker() {
                    jokers += 1;
                    format!("X{jokers}")
                } else {
                    let ranks = b"23456789TJQKA";
                    let suits = b"shdc";
                    format!(
                        "{}{}",
                        ranks[(card.rank - 2) as usize] as char,
                        suits[card.suit as usize] as char
                    )
                }
            })
            .collect();
    }
    let mut seats = [
        Seat { board: Default::default(), dead: Vec::new() },
        Seat { board: Default::default(), dead: Vec::new() },
    ];

    for street in 0..5usize {
        for seat in 0..2usize {
            let draw: Vec<String> = if street == 0 {
                names[seat][..5].to_vec()
            } else {
                names[seat][2 + street * 3..5 + street * 3].to_vec()
            };
            if let Some(rows) = forced.and_then(|map| map.get(&(street, seat))) {
                // Every placed card must come from this seat's board-so-far
                // plus its draw, and exactly one drawn card is thrown after
                // street 0.  A mismatch means the forced history does not
                // belong to this deal, which is worth failing loudly.
                let mut placed: Vec<String> = rows.iter().flatten().cloned().collect();
                placed.sort();
                let mut expect: Vec<String> = seats[seat]
                    .board
                    .iter()
                    .flatten()
                    .cloned()
                    .chain(draw.iter().cloned())
                    .collect();
                expect.sort();
                let tossed: Vec<String> = {
                    let mut left = expect.clone();
                    for card in &placed {
                        if let Some(at) = left.iter().position(|c| c == card) {
                            left.remove(at);
                        }
                    }
                    left
                };
                let want_toss = if street == 0 { 0 } else { 1 };
                if tossed.len() != want_toss || placed.len() + tossed.len() != expect.len() {
                    bail!(
                        "{id}: forced ({street},{seat}) places {placed:?} against {expect:?}"
                    );
                }
                seats[seat].board = rows.clone();
                seats[seat].dead.extend(tossed);
                record(&mut trace, street, seat, &draw, &seats[seat], "forced");
                continue;
            }
            let opp = 1 - seat;
            let (pool_names, pool) =
                pool_of(&seats[seat].board, &seats[opp].board, &seats[seat].dead, &draw)?;

            // The last street's second actor has a closed form; the first
            // actor has an exact V4.  Neither consults a model.
            if street == 4 {
                if seat == 1 {
                    let opponent = finish_of(&seats[0].board)?;
                    seats[1].board = closed_form_t4(&seats[1].board, &draw, &opponent, table)?;
                } else {
                    let request = V4FirstRequest {
                        id: format!("{id}/t4"),
                        board: BoardStr {
                            top: seats[0].board[0].clone(),
                            middle: seats[0].board[1].clone(),
                            bottom: seats[0].board[2].clone(),
                        },
                        dead: seats[0].dead.clone(),
                        draw: Some(draw.clone()),
                        opp_dead: Vec::new(),
                        opp_board: BoardStr {
                            top: seats[1].board[0].clone(),
                            middle: seats[1].board[1].clone(),
                            bottom: seats[1].board[2].clone(),
                        },
                    };
                    let solved = v4_first::solve(&request, table)?;
                    let best = solved
                        .actions
                        .iter()
                        .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap().then(b.key.cmp(&a.key)))
                        .ok_or_else(|| anyhow!("no T4 action"))?;
                    // The key is "c1@row,c2@row|discard".
                    let (placed, discard) = best
                        .key
                        .split_once('|')
                        .ok_or_else(|| anyhow!("bad action key"))?;
                    for part in placed.split(',') {
                        let (card, row) = part
                            .split_once('@')
                            .ok_or_else(|| anyhow!("bad placement"))?;
                        let slot = match row {
                            "top" => 0,
                            "middle" => 1,
                            _ => 2,
                        };
                        seats[0].board[slot].push(card.to_string());
                    }
                    seats[0].dead.push(discard.to_string());
                }
                record(&mut trace, street, seat, &draw, &seats[seat],
                       if seat == 1 { "closed-form" } else { "exact-v4" });
                continue;
            }

            // T3 second actor: exact, by the swapped V4.
            if street == 3 && seat == 1 {
                let scored: Vec<(f64, [Vec<String>; 3], String)> =
                    placements(&seats[1].board, &draw)
                    .into_par_iter()
                    .map(|(after, toss)| {
                    let mut dead = seats[1].dead.clone();
                    dead.push(toss.clone());
                    let request = V4FirstRequest {
                        id: format!("{id}/t3btn"),
                        board: BoardStr {
                            top: seats[0].board[0].clone(),
                            middle: seats[0].board[1].clone(),
                            bottom: seats[0].board[2].clone(),
                        },
                        dead,
                        draw: None,
                        opp_dead: Vec::new(),
                        opp_board: BoardStr {
                            top: after[0].clone(),
                            middle: after[1].clone(),
                            bottom: after[2].clone(),
                        },
                    };
                    let value = -v4_first::solve(&request, table)?.value;
                    Ok((value, after, toss))
                })
                    .collect::<Result<Vec<_>>>()?;
                let (_, after, toss) = scored
                    .into_iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.2.cmp(&a.2)))
                    .ok_or_else(|| anyhow!("no T3 placement"))?;
                seats[1].board = after;
                seats[1].dead.push(toss);
                record(&mut trace, street, seat, &draw, &seats[seat], "exact-v4");
                continue;
            }

            // Everything else is a model decision.
            let candidates: Vec<([Vec<String>; 3], Option<String>)> = if street == 0 {
                openings(&draw).into_iter().map(|rows| (rows, None)).collect()
            } else {
                placements(&seats[seat].board, &draw)
                    .into_iter()
                    .map(|(rows, toss)| (rows, Some(toss)))
                    .collect()
            };
            let opp_board = board_of(&seats[opp].board)?;
            let memo: playout::RowwiseMemo =
                std::sync::Mutex::new(std::collections::HashMap::new());
            let node_seed = format!("{id}/{street}/{seat}");
            // The opponent's half of the pair vector is the same for every
            // candidate; computing it once turns the most expensive block in
            // the encoding from a per-candidate cost into a per-decision one.
            let hu_model_dim = arms[seat]
                .hu
                .get(street)
                .and_then(|slot| slot.as_ref())
                .map(|pair| pair[seat].input_dim);
            let mut opp_tail: Vec<f32> = Vec::new();
            if hu_model_dim == Some(hu_encode::HU_FEATURE_SIZE) {
                // Only the 207-dim path pays for the opponent joint; the
                // hybrid encodings sample nothing to begin with.
                hu_encode::opponent_tail(
                    &opp_board, &pool, fl_ev, fl_table, &memo, 0, &node_seed,
                    arms[seat].joint_samples, &mut opp_tail,
                )?;
            }
            // Shortlist pass: rank everything cheaply, keep the top K for
            // the real evaluator.  Only where the model is the expensive
            // 207-dim kind -- a hybrid model already reads every candidate
            // for less than the ranker would cost.
            // The policy shortlist, where there is one: T0 first actor only.
            // It reads the hand once instead of encoding every opening, so it
            // is gated on nothing but its own presence -- and where it fires
            // the ranker below is skipped, not merely outvoted.
            let policy = if street == 0 && seat == 0 {
                arms[seat].t0_policy
            } else {
                None
            };
            let candidates: Vec<([Vec<String>; 3], Option<String>)> = match policy {
                Some(net)
                    if arms[seat].policy_topk > 0
                        && candidates.len() > arms[seat].policy_topk =>
                {
                    let policy = t0_policy::T0Policy::new(net, &draw)?;
                    let mut ranked: Vec<(f32, ([Vec<String>; 3], Option<String>))> = candidates
                        .into_iter()
                        .map(|(after, toss)| {
                            policy.score(&after).map(|score| (score, (after, toss)))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    // Same comparator and same stable sort as the ranker path,
                    // so ties land on the K boundary the same way.
                    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    ranked
                        .into_iter()
                        .take(arms[seat].policy_topk)
                        .map(|(_, candidate)| candidate)
                        .collect()
                }
                _ => candidates,
            };
            let ranker = arms[seat]
                .rankers
                .get(street)
                .and_then(|slot| slot.as_ref())
                .map(|pair| pair[seat])
                .filter(|_| policy.is_none());
            let candidates: Vec<([Vec<String>; 3], Option<String>)> = match ranker {
                Some(model)
                    if arms[seat].topk > 0
                        && candidates.len() > arms[seat].topk
                        && hu_model_dim == Some(hu_encode::HU_FEATURE_SIZE) =>
                {
                    let mut ranked: Vec<(f32, ([Vec<String>; 3], Option<String>))> =
                        candidates
                            .into_par_iter()
                            .map(|(after, toss)| {
                                let mut features: Vec<f32> = Vec::new();
                                let mut scratch: Vec<f32> = Vec::new();
                                let own_board = board_of(&after)?;
                                score_with(
                                    model, &own_board, &opp_board, &after, &toss,
                                    &seats[seat].dead, &pool_names, &seats[opp].board,
                                    &pool, fl_ev, fl_table, &memo, &node_seed,
                                    &mut features, &mut scratch, &[],
                                    arms[seat].joint_samples,
                                )
                                .map(|score| (score, (after, toss)))
                            })
                            .collect::<Result<Vec<_>>>()?;
                    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    ranked
                        .into_iter()
                        .take(arms[seat].topk)
                        .map(|(_, candidate)| candidate)
                        .collect()
                }
                _ => candidates,
            };
            // Every candidate is an independent read of the same nets against
            // the same pool, and at T0 there are 232 of them.  Serially this
            // decision alone runs for the better part of a minute on one core
            // while fifteen sit idle.
            let scored: Vec<(f32, [Vec<String>; 3], Option<String>)> = candidates
                .into_par_iter()
                .map(|(after, toss)| {
                let mut features: Vec<f32> = Vec::new();
                let mut scratch: Vec<f32> = Vec::new();
                let own_board = board_of(&after)?;
                let hu_model = arms[seat]
                    .hu
                    .get(street)
                    .and_then(|slot| slot.as_ref())
                    .map(|pair| pair[seat]);
                let score = if let Some(model) = hu_model {
                    score_with(
                        model, &own_board, &opp_board, &after, &toss,
                        &seats[seat].dead, &pool_names, &seats[opp].board,
                        &pool, fl_ev, fl_table, &memo, &node_seed,
                        &mut features, &mut scratch, &opp_tail,
                        arms[seat].joint_samples,
                    )?
                } else {
                    {
                        let model = arms[seat].own[street.min(2)];
                        playout::encode_for(
                            model,
                            &own_board,
                            &pool,
                            14,
                            fl_ev,
                            fl_table,
                            &memo,
                            0,
                            &node_seed,
                            &mut features,
                            None,
                        )?;
                        model.predict(&features, &mut scratch)
                    }
                };
                Ok((score, after, toss))
            })
                .collect::<Result<Vec<_>>>()?;
            let (_, after, toss) = scored
                .into_iter()
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .ok_or_else(|| anyhow!("no legal placement"))?;
            seats[seat].board = after;
            if let Some(toss) = toss {
                seats[seat].dead.push(toss);
            }
            record(&mut trace, street, seat, &draw, &seats[seat], "model");
        }
    }
    Ok([finish_of(&seats[0].board)?, finish_of(&seats[1].board)?])
}

fn record(
    trace: &mut Option<&mut Vec<TraceStep>>,
    street: usize,
    seat: usize,
    draw: &[String],
    state: &Seat,
    by: &'static str,
) {
    if let Some(sink) = trace.as_mut() {
        sink.push(TraceStep {
            street,
            seat,
            draw: draw.to_vec(),
            board: state.board.clone(),
            discard: state.dead.last().cloned(),
            by,
        });
    }
}

/// Play `hands` normal-versus-normal hands and keep every placement.
/// Trace hands `[first, first + hands)`.
///
/// The range exists so a shard can checkpoint: a trace writes nothing until
/// its last hand, so a preemption at hour two costs every hand of it (tr2
/// lost thirteen shards that way).  The caller loops over chunks, appending
/// and flushing each, and a lost worker costs one chunk.
pub fn trace_range(
    first: u64,
    hands: usize,
    seed: u64,
    arms: &[Arm<'_>; 2],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
) -> Result<Vec<TracedHand>> {
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    let seated: [&Arm<'_>; 2] = [&arms[0], &arms[1]];
    // Hands are independent -- each deals from its own (seed, hand) stream --
    // so a corpus-sized run parallelises over hands.  The per-decision rayon
    // inside play_hand still applies; the scheduler interleaves both levels.
    let out: Result<Vec<TracedHand>> = (first..first + hands as u64)
        .into_par_iter()
        .map(|hand| {
        let dealt = deal_names(seed, hand, 34);
        let id = format!("t/{seed}/{hand}");
        let mut steps = Vec::new();
        let both = play_hand(&id, &dealt, &seated, fl_ev, fl_table, &table, Some(&mut steps), None)?;
        Ok(TracedHand {
            hand,
            steps,
            bb_busted: both[0].busted,
            btn_busted: both[1].busted,
            bb_royalty: both[0].royalty,
            btn_royalty: both[1].royalty,
            bb_entry: both[0].entry_width,
            btn_entry: both[1].entry_width,
            bb_points: settle(&both[0], &both[1]),
        })
    })
        .collect();
    let mut out = out?;
    out.sort_by_key(|hand| hand.hand);
    Ok(out)
}

/// One candidate's deep evaluation: full-game rollouts under a forced T0.
#[derive(serde::Serialize)]
pub struct T0DeepRow {
    pub key: String,
    pub n: usize,
    pub mean: f64,
    /// Per-rollout scores, index-aligned across candidates: scores[r] of two
    /// rows share the deal, so their difference is the paired statistic.
    pub scores: Vec<f32>,
}

pub(crate) fn t0_key_of(rows: &[Vec<String>; 3]) -> String {
    rows.iter()
        .map(|row| {
            let mut sorted = row.clone();
            sorted.sort();
            sorted.join(",")
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// One deal played twice with the seats traded: the duplicate-bridge trick.
///
/// Every number here is arm A's.  Orientation 0 seats A first (BB), 1 seats
/// it second; the deal, and the sampling streams inside the encoders, are
/// identical across the two.  Deal luck therefore cancels in the average of
/// the orientations, and two identical arms score exactly zero on every hand
/// -- which is the acceptance test.
///
/// Entry widths ride along raw rather than pre-priced: `fl_ev` is a borrowed
/// constant this project has not yet re-measured, and the two arms differ in
/// how often they enter Fantasyland, so a verdict must be re-pricable without
/// replaying the match.
#[derive(serde::Serialize)]
pub struct MirrorRow {
    pub hand: u64,
    /// Line settlement from A's side, per orientation.
    pub settle: [i32; 2],
    /// A's Fantasyland entry width (0 = none), per orientation.
    pub a_entry: [u8; 2],
    pub b_entry: [u8; 2],
    pub a_foul: [bool; 2],
    pub b_foul: [bool; 2],
    pub a_royalty: [i32; 2],
    pub b_royalty: [i32; 2],
}

/// Play `hands` deals twice each, trading seats, and report arm A's side.
pub fn mirror_hands(
    hands: usize,
    seed: u64,
    arms: &[Arm<'_>; 2],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
) -> Result<Vec<MirrorRow>> {
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    let out: Result<Vec<MirrorRow>> = (0..hands as u64)
        .into_par_iter()
        .map(|hand| {
            let dealt = deal_names(seed, hand, 34);
            // One id for both orientations: the encoders seed their sampling
            // from it, so a sub-state common to both tables samples alike.
            let id = format!("mir/{seed}/{hand}");
            let mut row = MirrorRow {
                hand,
                settle: [0; 2],
                a_entry: [0; 2],
                b_entry: [0; 2],
                a_foul: [false; 2],
                b_foul: [false; 2],
                a_royalty: [0; 2],
                b_royalty: [0; 2],
            };
            for orientation in 0..2usize {
                let seated: [&Arm<'_>; 2] = if orientation == 0 {
                    [&arms[0], &arms[1]]
                } else {
                    [&arms[1], &arms[0]]
                };
                let both = play_hand(&id, &dealt, &seated, fl_ev, fl_table, &table, None, None)?;
                // Seat 0 is whoever the orientation seated first.
                let (a, b) = if orientation == 0 {
                    (&both[0], &both[1])
                } else {
                    (&both[1], &both[0])
                };
                row.settle[orientation] = if orientation == 0 {
                    settle(a, b)
                } else {
                    -settle(&both[0], &both[1])
                };
                row.a_entry[orientation] = if a.busted { 0 } else { a.entry_width };
                row.b_entry[orientation] = if b.busted { 0 } else { b.entry_width };
                row.a_foul[orientation] = a.busted;
                row.b_foul[orientation] = b.busted;
                row.a_royalty[orientation] = a.royalty;
                row.b_royalty[orientation] = b.royalty;
            }
            Ok(row)
        })
        .collect();
    let mut out = out?;
    out.sort_by_key(|row| row.hand);
    Ok(out)
}

/// Deep-evaluate one decision out of a traced hand.
///
/// The history up to that decision is replayed exactly (both seats, forced),
/// the candidate under test is forced, and everything after it is played by
/// the arms against a **re-dealt future**: the cards the opponent would go on
/// to draw are unknown at the moment of choosing, so judging the choice
/// against the one future that actually happened would grade luck.  Rollout
/// `r` deals the same future to every candidate, so the comparison is paired.
#[allow(clippy::too_many_arguments)]
pub fn deep_replay(
    steps: &[(usize, usize, Vec<String>, [Vec<String>; 3])],
    street: usize,
    seat: usize,
    wanted: Option<&[String]>,
    rollouts: usize,
    seed: u64,
    arms: &[Arm<'_>; 2],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
) -> Result<Vec<T0DeepRow>> {
    let order = |s: usize, t: usize| s * 2 + t;
    let target = order(street, seat);

    // History: every decision strictly before the target, by both seats.
    let mut history: Forced = Forced::new();
    let mut draws: std::collections::HashMap<(usize, usize), Vec<String>> = Default::default();
    let mut own_before: [Vec<String>; 3] = Default::default();
    let mut target_draw: Vec<String> = Vec::new();
    for (s, t, draw, board) in steps {
        draws.insert((*s, *t), draw.clone());
        if order(*s, *t) < target {
            history.insert((*s, *t), board.clone());
        }
        if *s == street && *t == seat {
            target_draw = draw.clone();
        }
        if *t == seat && order(*s, *t) < target {
            own_before = board.clone();
        }
    }
    if target_draw.len() != if street == 0 { 5 } else { 3 } {
        bail!("no draw recorded for street {street} seat {seat}");
    }

    // Known at the moment of choosing: both boards, both discards, own draw.
    let mut seen: Vec<String> = Vec::new();
    for board in history.values() {
        seen.extend(board.iter().flatten().cloned());
    }
    for (s, t, draw, board) in steps {
        if order(*s, *t) < target && *s > 0 {
            let placed: Vec<&String> = board.iter().flatten().collect();
            for card in draw {
                if !placed.iter().any(|c| *c == card) {
                    seen.push(card.clone());  // a discard, face down but gone
                }
            }
        }
    }
    seen.extend(target_draw.iter().cloned());
    seen.sort();
    seen.dedup();

    let candidates: Vec<([Vec<String>; 3], String)> = if street == 0 {
        openings(&target_draw).into_iter().map(|rows| {
            let key = t0_key_of(&rows);
            (rows, key)
        }).collect()
    } else {
        placements(&own_before, &target_draw)
            .into_iter()
            .map(|(rows, _toss)| {
                let key = t0_key_of(&rows);
                (rows, key)
            })
            .collect()
    };
    let chosen: Vec<([Vec<String>; 3], String)> = match wanted {
        None => candidates,
        Some(keys) => {
            let by_key: std::collections::BTreeMap<String, [Vec<String>; 3]> =
                candidates.into_iter().map(|(rows, key)| (key, rows)).collect();
            let unknown: Vec<&str> = keys.iter().map(String::as_str)
                .filter(|k| !by_key.contains_key(*k)).collect();
            if !unknown.is_empty() {
                bail!("{} unknown keys of {}: {unknown:?}", unknown.len(), by_key.len());
            }
            keys.iter().map(|k| (by_key[k].clone(), k.clone())).collect()
        }
    };

    let table = [fl_ev.value(14), fl_ev.value(15), fl_ev.value(16), fl_ev.value(17)];
    let seated: [&Arm<'_>; 2] = [&arms[0], &arms[1]];

    // No rollouts asked for: report what the serving model thinks, which is
    // the cheap prior a cascade sieves with.  Same encoding path play_hand
    // uses, so the ranking is the one the engine actually acts on.
    if rollouts == 0 {
        let opp_rows: [Vec<String>; 3] = history
            .iter()
            .filter(|((_s, t), _)| *t == 1 - seat)
            .max_by_key(|((s, t), _)| order(*s, *t))
            .map(|(_k, board)| board.clone())
            .unwrap_or_default();
        let mut dead: Vec<String> = Vec::new();
        for (s, t, draw, board) in steps {
            if *t == seat && *s > 0 && order(*s, *t) < target {
                let placed: Vec<&String> = board.iter().flatten().collect();
                for card in draw {
                    if !placed.iter().any(|c| *c == card) {
                        dead.push(card.clone());
                    }
                }
            }
        }
        let (pool_names, pool) = pool_of(&own_before, &opp_rows, &dead, &target_draw)?;
        let opp_board = board_of(&opp_rows)?;
        let memo: playout::RowwiseMemo =
            std::sync::Mutex::new(std::collections::HashMap::new());
        let node_seed = format!("replay-rank/{street}/{seat}");
        let model = arms[0]
            .hu
            .get(street)
            .and_then(|slot| slot.as_ref())
            .map(|pair| pair[seat])
            .ok_or_else(|| anyhow!("arm A has no model for street {street}"))?;
        let mut opp_tail: Vec<f32> = Vec::new();
        if model.input_dim == hu_encode::HU_FEATURE_SIZE {
            hu_encode::opponent_tail(
                &opp_board, &pool, fl_ev, fl_table, &memo, 0, &node_seed,
                arms[0].joint_samples, &mut opp_tail,
            )?;
        }
        let mut scored: Vec<T0DeepRow> = chosen
            .par_iter()
            .map(|(rows, key)| {
                let mut features: Vec<f32> = Vec::new();
                let mut scratch: Vec<f32> = Vec::new();
                let own_board = board_of(rows)?;
                let toss: Option<String> = if street == 0 {
                    None
                } else {
                    let placed: Vec<&String> = rows.iter().flatten().collect();
                    target_draw.iter().find(|c| !placed.iter().any(|p| p == c)).cloned()
                };
                let score = score_with(
                    model, &own_board, &opp_board, rows, &toss, &dead, &pool_names,
                    &opp_rows, &pool, fl_ev, fl_table, &memo, &node_seed,
                    &mut features, &mut scratch, &opp_tail, arms[0].joint_samples,
                )?;
                Ok(T0DeepRow { key: key.clone(), n: 0, mean: score as f64, scores: Vec::new() })
            })
            .collect::<Result<Vec<_>>>()?;
        scored.sort_by(|a, b| b.mean.partial_cmp(&a.mean).unwrap());
        return Ok(scored);
    }

    let per_rollout: Result<Vec<Vec<f32>>> = (0..rollouts as u64)
        .into_par_iter()
        .map(|rollout| {
            // A fresh shuffle supplies every card not yet seen; the streets
            // already played keep their real draws.
            let shuffled = deal_names(seed, rollout, 54);
            let mut fresh: Vec<String> = Vec::new();
            for card in shuffled {
                let name = if card.is_joker() {
                    "X".to_string()
                } else {
                    let ranks = b"23456789TJQKA";
                    let suits = b"shdc";
                    format!("{}{}", ranks[(card.rank - 2) as usize] as char,
                            suits[card.suit as usize] as char)
                };
                if name != "X" && !seen.contains(&name) {
                    fresh.push(name);
                }
            }
            let jokers_seen = seen.iter().filter(|c| c.starts_with('X')).count();
            for slot in 0..(2 - jokers_seen.min(2)) {
                fresh.insert((slot * 7 + 3).min(fresh.len()), format!("X{}", slot + 1));
            }
            let mut next = fresh.into_iter();
            let mut names: [Vec<String>; 2] = Default::default();
            for s in 0..5usize {
                for t in 0..2usize {
                    let want = if s == 0 { 5 } else { 3 };
                    let known = if order(s, t) <= target { draws.get(&(s, t)) } else { None };
                    match known {
                        Some(cards) if cards.len() == want => names[t].extend(cards.clone()),
                        _ => for _ in 0..want {
                            names[t].push(next.next().ok_or_else(|| anyhow!("deck exhausted"))?);
                        },
                    }
                }
            }
            let dealt: Vec<fl_solver::Card> = names[0].iter().chain(names[1].iter())
                .map(|name| {
                    if name.starts_with('X') {
                        Ok(fl_solver::Card { rank: 0, suit: 4 })
                    } else {
                        let b = name.as_bytes();
                        let rank = b"23456789TJQKA".iter().position(|r| *r == b[0])
                            .ok_or_else(|| anyhow!("bad card {name}"))? as u8 + 2;
                        let suit = b"shdc".iter().position(|s| *s == b[1])
                            .ok_or_else(|| anyhow!("bad card {name}"))? as u8;
                        Ok(fl_solver::Card { rank, suit })
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let id = format!("rp/{seed}/{rollout}");
            let entry = |f: &Finished| -> f64 {
                if !f.busted && f.entry_width >= 14 {
                    table[(f.entry_width - 14) as usize]
                } else { 0.0 }
            };
            chosen.iter().map(|(rows, _key)| {
                let mut forced = history.clone();
                forced.insert((street, seat), rows.clone());
                let both = play_hand(&id, &dealt, &seated, fl_ev, fl_table, &table,
                                     None, Some(&forced))?;
                let (me, them) = (&both[seat], &both[1 - seat]);
                let sign = if seat == 0 { 1.0 } else { -1.0 };
                Ok((sign * settle(&both[0], &both[1]) as f64
                    + entry(me) - entry(them)) as f32)
            }).collect::<Result<Vec<f32>>>()
        })
        .collect();
    let per_rollout = per_rollout?;
    let mut out: Vec<T0DeepRow> = chosen.iter().enumerate().map(|(index, (_rows, key))| {
        let scores: Vec<f32> = per_rollout.iter().map(|row| row[index]).collect();
        let mean = scores.iter().map(|s| *s as f64).sum::<f64>() / scores.len().max(1) as f64;
        T0DeepRow { key: key.clone(), n: scores.len(), mean, scores }
    }).collect();
    out.sort_by(|a, b| b.mean.partial_cmp(&a.mean).unwrap());
    Ok(out)
}

/// One opening under arm A's T0 machinery: what the evaluator scores it, and
/// what the shortlist ranker scores it.
///
/// Two scores because serving uses two.  The evaluator alone reads all 232
/// openings only in this audit; in a real hand the ranker cuts to `topk`
/// first and the evaluator never sees the rest, so the evaluator's own first
/// row is not the served card unless the ranker also kept it.
pub struct T0ModelRow {
    pub key: String,
    pub score: f32,
    /// The ranker's score and its rank among all openings (1 = the ranker's
    /// first choice).  `None` when serving would not shortlist this decision
    /// at all -- no T0 ranker loaded, `--hu-topk 0`, or an evaluator cheap
    /// enough to read every candidate -- and then the served opening is the
    /// evaluator's own first row.
    pub ranker_score: Option<f32>,
    pub ranker_rank: Option<usize>,
    /// The policy's logit and its rank among all 232 openings (1 = the
    /// policy's first choice).  `None` when the arm carries no T0 policy, and
    /// then the ranker columns describe the shortlist instead.  These are the
    /// parity harness's read: Python's per-action ordering must reproduce
    /// `policy_rank` exactly, or the served index means a different placement
    /// than the trained one.
    pub policy_score: Option<f32>,
    pub policy_rank: Option<usize>,
}

/// What arm A's T0 machinery thinks of every opening, no rollouts.
///
/// The cheap prior that a cascade would sieve with.  Comparing this ranking
/// against the playout ranking is the only way to know whether a model-first
/// cut is safe -- a prior that drops the true best makes every later rollout
/// worthless, and nothing downstream would ever report it.
///
/// The ranker columns exist because the evaluator column is not the served
/// choice.  Outside the shortlist the evaluator is reading boards no ranker
/// would ever hand it -- off its training distribution, where it has been
/// seen to crown nonsense -- so an audit that reports only its argmax is
/// auditing a decision the program never makes.
pub fn t0_model_scores(
    hero: &[String],
    arm: &Arm<'_>,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
) -> Result<Vec<T0ModelRow>> {
    let empty: [Vec<String>; 3] = Default::default();
    let (pool_names, pool) = pool_of(&empty, &empty, &[], hero)?;
    let opp_board = board_of(&empty)?;
    let memo: playout::RowwiseMemo = std::sync::Mutex::new(std::collections::HashMap::new());
    let node_seed = "model-rank/t0";
    let model = arm
        .hu
        .first()
        .and_then(|slot| slot.as_ref())
        .map(|pair| pair[0])
        .ok_or_else(|| anyhow!("arm A has no T0 model"))?;
    let mut opp_tail: Vec<f32> = Vec::new();
    if model.input_dim == hu_encode::HU_FEATURE_SIZE {
        hu_encode::opponent_tail(
            &opp_board, &pool, fl_ev, fl_table, &memo, 0, node_seed,
            arm.joint_samples, &mut opp_tail,
        )?;
    }
    let candidates = openings(hero);
    // The policy shortlist, gated exactly as `play_hand` gates it, so "no
    // policy column" here means "no policy shortlist there" -- and where it
    // fires the ranker is skipped here too.
    let policy_rows: std::collections::HashMap<String, (f32, usize)> = match arm.t0_policy {
        None => std::collections::HashMap::new(),
        Some(net) => {
            let policy = t0_policy::T0Policy::new(net, hero)?;
            let mut ranked: Vec<(f32, String)> = candidates
                .iter()
                .map(|after| Ok((policy.score(after)?, t0_key_of(after))))
                .collect::<Result<Vec<_>>>()?;
            ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            ranked
                .into_iter()
                .enumerate()
                .map(|(index, (score, key))| (key, (score, index + 1)))
                .collect()
        }
    };
    // The shortlist, gated on exactly the three conditions `play_hand` gates
    // it on, so "no ranker column" here means "no shortlist there".
    let ranker = arm
        .rankers
        .first()
        .and_then(|slot| slot.as_ref())
        .map(|pair| pair[0])
        .filter(|_| {
            arm.t0_policy.is_none()
                && arm.topk > 0
                && candidates.len() > arm.topk
                && model.input_dim == hu_encode::HU_FEATURE_SIZE
        });
    let shortlist: std::collections::HashMap<String, (f32, usize)> = match ranker {
        None => std::collections::HashMap::new(),
        Some(ranker) => {
            // Same `score_with`, same empty tail -- the ranker is fed by the
            // serve path's call, not by a second encoding written here that
            // could drift from it.
            let mut ranked: Vec<(f32, String)> = candidates
                .par_iter()
                .map(|after| {
                    let mut features: Vec<f32> = Vec::new();
                    let mut scratch: Vec<f32> = Vec::new();
                    let own_board = board_of(after)?;
                    let score = score_with(
                        ranker, &own_board, &opp_board, after, &None, &[], &pool_names,
                        &empty, &pool, fl_ev, fl_table, &memo, node_seed,
                        &mut features, &mut scratch, &[], arm.joint_samples,
                    )?;
                    Ok((score, t0_key_of(after)))
                })
                .collect::<Result<Vec<_>>>()?;
            // The serve path's comparator over the serve path's candidate
            // order, and `sort_by` is stable in both places: ties land on the
            // K boundary the same way here as they do in a hand.
            ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            ranked
                .into_iter()
                .enumerate()
                .map(|(index, (score, key))| (key, (score, index + 1)))
                .collect()
        }
    };
    let mut scored: Vec<T0ModelRow> = candidates
        .par_iter()
        .map(|after| {
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            let own_board = board_of(after)?;
            let score = score_with(
                model, &own_board, &opp_board, after, &None, &[], &pool_names,
                &empty, &pool, fl_ev, fl_table, &memo, node_seed,
                &mut features, &mut scratch, &opp_tail, arm.joint_samples,
            )?;
            let key = t0_key_of(after);
            let ranked = shortlist.get(&key).copied();
            let by_policy = policy_rows.get(&key).copied();
            Ok(T0ModelRow {
                key,
                score,
                ranker_score: ranked.map(|(score, _)| score),
                ranker_rank: ranked.map(|(_, rank)| rank),
                policy_score: by_policy.map(|(score, _)| score),
                policy_rank: by_policy.map(|(_, rank)| rank),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    Ok(scored)
}

/// One card name as an `fl_solver::Card`.  Jokers carry no identity here:
/// X1 and X2 are the same card, and only the deck accounting tells them
/// apart.
pub(crate) fn card_of(name: &str) -> Result<fl_solver::Card> {
    if name.starts_with('X') {
        return Ok(fl_solver::Card { rank: 0, suit: 4 });
    }
    let bytes = name.as_bytes();
    let rank = bytes
        .first()
        .and_then(|r| b"23456789TJQKA".iter().position(|x| x == r))
        .map(|i| i as u8 + 2);
    let suit = bytes
        .get(1)
        .and_then(|s| b"shdc".iter().position(|x| x == s))
        .map(|i| i as u8);
    match (rank, suit, bytes.len()) {
        (Some(rank), Some(suit), 2) => Ok(fl_solver::Card { rank, suit }),
        _ => bail!("bad card name {name}"),
    }
}

/// The five `--t0-cards` as canonical names plus their `Card` values.
///
/// Joker names are renumbered by position because `play_hand` renames the
/// deal's jokers X1 then X2 in deal order: a caller that spelled them the
/// other way round would otherwise write candidate keys no board can match.
pub(crate) fn normalise_hero(hero: &[String]) -> Result<(Vec<String>, Vec<fl_solver::Card>)> {
    if hero.len() != 5 {
        bail!("--t0-cards wants exactly five cards, got {}", hero.len());
    }
    let mut jokers = 0usize;
    let hero: Vec<String> = hero
        .iter()
        .map(|name| {
            if name.starts_with('X') {
                jokers += 1;
                format!("X{jokers}")
            } else {
                name.clone()
            }
        })
        .collect();
    let mut unique: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for name in &hero {
        if !unique.insert(name.as_str()) {
            bail!("card {name} appears twice in --t0-cards");
        }
    }
    let mut hero_cards: Vec<fl_solver::Card> = Vec::with_capacity(5);
    for name in &hero {
        if name.starts_with('X') {
            hero_cards.push(fl_solver::Card { rank: 0, suit: 4 });
            continue;
        }
        let bytes = name.as_bytes();
        let rank = bytes
            .first()
            .and_then(|r| b"23456789TJQKA".iter().position(|x| x == r))
            .map(|i| i as u8 + 2);
        let suit = bytes
            .get(1)
            .and_then(|s| b"shdc".iter().position(|x| x == s))
            .map(|i| i as u8);
        match (rank, suit, bytes.len()) {
            (Some(rank), Some(suit), 2) => hero_cards.push(fl_solver::Card { rank, suit }),
            _ => bail!("bad card name {name} (want e.g. As, Td, 7c, X1)"),
        }
    }
    Ok((hero, hero_cards))
}

/// The candidate boards a batch will evaluate, in the requested order.
///
/// Discipline: unknown or duplicate keys are enumerated errors, never skipped
/// -- a silently dropped candidate reads downstream as "measured and lost".
pub(crate) fn t0_candidate_keys(
    hero: &[String],
    wanted: Option<&[String]>,
) -> Result<Vec<([Vec<String>; 3], String)>> {
    let all = openings(hero);
    let chosen: Vec<([Vec<String>; 3], String)> = match wanted {
        None => all
            .into_iter()
            .map(|rows| {
                let key = t0_key_of(&rows);
                (rows, key)
            })
            .collect(),
        Some(keys) => {
            let by_key: std::collections::BTreeMap<String, [Vec<String>; 3]> = all
                .into_iter()
                .map(|rows| (t0_key_of(&rows), rows))
                .collect();
            let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            let duplicates: Vec<&str> = keys
                .iter()
                .map(String::as_str)
                .filter(|key| !seen.insert(key))
                .collect();
            if !duplicates.is_empty() {
                bail!("duplicate keys: {duplicates:?}");
            }
            let unknown: Vec<&str> = keys
                .iter()
                .map(String::as_str)
                .filter(|key| !by_key.contains_key(*key))
                .collect();
            if !unknown.is_empty() {
                bail!(
                    "{} unknown keys (of {} candidates): {unknown:?}",
                    unknown.len(),
                    by_key.len()
                );
            }
            keys.iter()
                .map(|key| (by_key[key].clone(), key.clone()))
                .collect()
        }
    };
    if chosen.is_empty() {
        bail!("no candidates to evaluate");
    }
    Ok(chosen)
}

/// Deep evaluation of a specified BB (first-actor) T0 hand.
///
/// The regular track's restricted-evaluation lesson, translated: the sieve
/// (street teacher, boundary-net read) proposes, but the verdict comes from
/// playing the actual game to the end.  Every candidate placement is forced
/// as BB's opening and the remaining four streets run under both arms'
/// serving policies; rollout `r` deals the same opponent hand and the same
/// future draws to every candidate, so candidate differences are paired.
///
/// Discipline (also borrowed): unknown or duplicate keys are enumerated
/// errors, never skipped -- a silently dropped candidate reads as "measured".
/// One invocation is one batch; verdicts want 4+ seeds and a CI clear of
/// zero.
pub fn t0_deep_eval(
    hero: &[String],
    wanted: Option<&[String]>,
    rollouts: usize,
    seed: u64,
    arms: &[Arm<'_>; 2],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
) -> Result<Vec<T0DeepRow>> {
    let (hero, hero_cards) = normalise_hero(hero)?;
    let chosen = t0_candidate_keys(&hero, wanted)?;

    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    let seated: [&Arm<'_>; 2] = [&arms[0], &arms[1]];
    let per_rollout: Result<Vec<Vec<f32>>> = (0..rollouts as u64)
        .into_par_iter()
        .map(|rollout| {
            // A full-deck shuffle with the hero's five removed keeps the
            // dealing machinery identical to the match harness.
            let shuffled = deal_names(seed, rollout, 54);
            let mut need = hero_cards.clone();
            let mut rest: Vec<fl_solver::Card> = Vec::with_capacity(49);
            for card in shuffled {
                if let Some(at) = need
                    .iter()
                    .position(|h| h.rank == card.rank && h.suit == card.suit)
                {
                    need.swap_remove(at);
                } else {
                    rest.push(card);
                }
            }
            if !need.is_empty() {
                bail!("deck exclusion failed for rollout {rollout}");
            }
            let mut dealt: Vec<fl_solver::Card> = Vec::with_capacity(34);
            dealt.extend(hero_cards.iter().copied());
            dealt.extend(rest[..29].iter().copied());
            // The id seeds the encoders' sample streams; candidates share it
            // so identical sub-states sample identically.
            let id = format!("d/{seed}/{rollout}");
            let entry = |f: &Finished| -> f64 {
                if !f.busted && f.entry_width >= 14 {
                    table[(f.entry_width - 14) as usize]
                } else {
                    0.0
                }
            };
            chosen
                .iter()
                .map(|(rows, _key)| {
                    let mut forced: Forced = Forced::new();
                    forced.insert((0, 0), rows.clone());
                    let both = play_hand(
                        &id, &dealt, &seated, fl_ev, fl_table, &table, None, Some(&forced),
                    )?;
                    Ok((settle(&both[0], &both[1]) as f64 + entry(&both[0]) - entry(&both[1]))
                        as f32)
                })
                .collect::<Result<Vec<f32>>>()
        })
        .collect();
    let per_rollout = per_rollout?;

    let mut out: Vec<T0DeepRow> = chosen
        .iter()
        .enumerate()
        .map(|(index, (_rows, key))| {
            let scores: Vec<f32> = per_rollout.iter().map(|row| row[index]).collect();
            let mean = scores.iter().map(|s| *s as f64).sum::<f64>() / scores.len().max(1) as f64;
            T0DeepRow {
                key: key.clone(),
                n: scores.len(),
                mean,
                scores,
            }
        })
        .collect();
    out.sort_by(|a, b| b.mean.partial_cmp(&a.mean).unwrap());
    Ok(out)
}

pub struct MatchConfig {
    pub hands: usize,
    pub workers: usize,
    pub seed: u64,
    pub stack: i32,
    pub gap: i32,
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Normal,
    Fl(u8),
}

impl State {
    fn label(&self) -> &'static str {
        match self {
            State::Normal => "normal",
            State::Fl(14) => "fl14",
            State::Fl(15) => "fl15",
            State::Fl(16) => "fl16",
            _ => "fl17",
        }
    }
    fn cards(&self) -> usize {
        match self {
            State::Normal => 17,
            State::Fl(width) => *width as usize,
        }
    }
}

/// The full match: sessions, Fantasyland, seat alternation, real points.
#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &MatchConfig,
    arms: &[Arm<'_>; 2],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    output: &std::path::Path,
) -> Result<()> {
    use std::io::Write;
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    let per_worker = config.hands / config.workers.max(1);
    let started = std::time::Instant::now();

    let chunks: Vec<Result<Vec<String>>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..config.workers.max(1) {
            let table = table;
            handles.push(scope.spawn(move || -> Result<Vec<String>> {
                let mut lines: Vec<String> = Vec::with_capacity(per_worker);
                let mut stacks = [config.stack, config.stack];
                let mut states = [State::Normal, State::Normal];
                let mut session = 0usize;
                for local in 0..per_worker {
                    let global = (worker * per_worker + local) as u64;
                    // Arm 0 sits BB on even hands and BTN on odd ones, so the
                    // seat cancels out of the difference between the arms.
                    let flip = global % 2 == 1;
                    // seats[0] is BB (acts first); arm_at[seat] is the arm there.
                    let arm_at: [usize; 2] = if flip { [1, 0] } else { [0, 1] };
                    let seated: [&Arm<'_>; 2] = [&arms[arm_at[0]], &arms[arm_at[1]]];
                    let seat_state = [states[arm_at[0]], states[arm_at[1]]];

                    let dealt = deal_names(
                        config.seed,
                        global,
                        seat_state[0].cards() + seat_state[1].cards(),
                    );
                    let (first_cards, second_cards) = dealt.split_at(seat_state[0].cards());
                    let id = format!("m/{}/{global}", config.seed);

                    let mut finished: [Option<Finished>; 2] = [None, None];
                    let mut stays = [false, false];
                    match (seat_state[0], seat_state[1]) {
                        (State::Normal, State::Normal) => {
                            let both = play_hand(
                                &id,
                                &dealt,
                                &seated,
                                fl_ev,
                                fl_table,
                                &table,
                                None,
                                None,
                            )?;
                            finished = [Some(both[0]), Some(both[1])];
                        }
                        _ => {
                            // At most one seat is normal; it plays its own hand
                            // blind (a Fantasyland board is face down), then any
                            // Fantasyland seat answers it.
                            for seat in 0..2usize {
                                if let State::Normal = seat_state[seat] {
                                    let cards = if seat == 0 { first_cards } else { second_cards };
                                    let arm = seated[seat];
                                    let trace = super::self_play::play_normal_traced(
                                        &format!("{id}/{seat}"),
                                        cards,
                                        fl_ev,
                                        fl_table,
                                        &table,
                                        arm.own[0],
                                        arm.own[1],
                                        arm.own[2],
                                    )?;
                                    finished[seat] = Some(trace.finished);
                                }
                            }
                            for seat in 0..2usize {
                                if let State::Fl(width) = seat_state[seat] {
                                    let cards = if seat == 0 { first_cards } else { second_cards };
                                    let opponent = match seat_state[1 - seat] {
                                        State::Normal => finished[1 - seat].as_ref(),
                                        State::Fl(_) => None,
                                    };
                                    let (board, stay) =
                                        super::self_play::play_fl(cards, width, &table, opponent);
                                    finished[seat] = Some(board);
                                    stays[seat] = stay;
                                }
                            }
                        }
                    }
                    let bb = finished[0].expect("BB played");
                    let btn = finished[1].expect("BTN played");

                    let entry_credit = |f: &Finished| -> f64 {
                        if !f.busted && f.entry_width >= 14 {
                            table[(f.entry_width - 14) as usize]
                        } else {
                            0.0
                        }
                    };
                    // Real points only; the Fantasyland credit is not money,
                    // it is the value of a future hand, and that future hand
                    // gets played.
                    let raw = settle(&bb, &btn);
                    let paid = if raw > 0 {
                        raw.min(stacks[arm_at[1]])
                    } else {
                        -((-raw).min(stacks[arm_at[0]]))
                    };
                    stacks[arm_at[0]] += paid;
                    stacks[arm_at[1]] -= paid;

                    let next = |state: State, f: &Finished, stay: bool| -> State {
                        match state {
                            State::Normal => {
                                if !f.busted && f.entry_width >= 14 {
                                    State::Fl(f.entry_width)
                                } else {
                                    State::Normal
                                }
                            }
                            State::Fl(width) => {
                                if stay {
                                    State::Fl(width)
                                } else {
                                    State::Normal
                                }
                            }
                        }
                    };
                    let pending_seat = [
                        next(seat_state[0], &bb, stays[0]),
                        next(seat_state[1], &btn, stays[1]),
                    ];
                    states[arm_at[0]] = pending_seat[0];
                    states[arm_at[1]] = pending_seat[1];

                    let end = if stacks[0] == 0 || stacks[1] == 0 {
                        Some("zero")
                    } else if (stacks[0] - stacks[1]).abs() >= config.gap
                        && states[0] == State::Normal
                        && states[1] == State::Normal
                    {
                        Some("gap")
                    } else {
                        None
                    };

                    // Everything from arm 0's side.
                    let (a, b) = if flip { (&btn, &bb) } else { (&bb, &btn) };
                    let a_points = if flip { -paid } else { paid };
                    lines.push(format!(
                        "{{\"hand\":{global},\"session\":{session},\"a_seat\":\"{}\",\
                         \"a_state\":\"{}\",\"b_state\":\"{}\",\
                         \"a_points\":{a_points},\"raw\":{},\
                         \"a_foul\":{},\"b_foul\":{},\"a_entry\":{},\"b_entry\":{},\
                         \"a_royalty\":{},\"b_royalty\":{},\
                         \"a_credit\":{:.4},\"b_credit\":{:.4},\
                         \"stack_a\":{},\"stack_b\":{},\"end\":{}}}",
                        if flip { "btn" } else { "bb" },
                        states_label(seat_state, arm_at, 0),
                        states_label(seat_state, arm_at, 1),
                        if flip { -raw } else { raw },
                        a.busted,
                        b.busted,
                        a.entry_width,
                        b.entry_width,
                        a.royalty,
                        b.royalty,
                        entry_credit(a),
                        entry_credit(b),
                        stacks[0],
                        stacks[1],
                        match end {
                            Some(reason) => format!("\"{reason}\""),
                            None => "null".to_string(),
                        },
                    ));

                    if end.is_some() {
                        stacks = [config.stack, config.stack];
                        states = [State::Normal, State::Normal];
                        session += 1;
                    }
                }
                Ok(lines)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    let mut written = 0usize;
    for chunk in chunks {
        for line in chunk? {
            writeln!(out, "{line}")?;
            written += 1;
        }
    }
    out.flush()?;
    eprintln!(
        "hu-match: {written} hands, {} workers, {:.1} s",
        config.workers,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The state label of the arm sitting in `arm` slot, at the hand it played.
fn states_label(seat_state: [State; 2], arm_at: [usize; 2], arm: usize) -> &'static str {
    let seat = if arm_at[0] == arm { 0 } else { 1 };
    seat_state[seat].label()
}
