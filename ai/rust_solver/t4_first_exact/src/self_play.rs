//! Mirror self-play under the frozen session rules (owner, 2026-08-14/15).
//!
//! Two copies of the served chain play each other and every hand is settled in
//! **real points** -- lines with the scoop bonus, royalties both ways, fouls --
//! not in the own-hand objective the choosers were trained on.  The objective
//! prices a board; the scoreboard here is the game.  Fantasyland hands are
//! played, not credited: a Fantasyland player facing a normal opponent sets its
//! board **after** seeing the opponent's finished one (the 2026-08-05 rule),
//! and two simultaneous Fantasyland players set theirs blind.
//!
//! What this exists to measure is the Fantasyland EV table.  Every hand line
//! carries both players' states, the raw settlement and the stack-clamped
//! transfer, entries, stays and session boundaries, so the analyzer can
//! decompose by opponent state (FL-vs-normal is the value, FL-vs-FL the
//! convolution, normal-vs-normal the zero check) without re-simulating.
//!
//! Session rules, frozen:
//! * stacks start at 200 and never go below zero -- the loser pays what it has
//!   (zero-sum: the winner receives only what the loser pays);
//! * positions alternate every hand;
//! * after a hand, if the stacks differ by 40 or more AND neither player is in
//!   Fantasyland for the next hand, the session ends and restarts at 200/200;
//! * a stack hitting zero ends the session immediately, pending Fantasyland
//!   included.

use anyhow::{anyhow, bail, Result};
use fl_solver::frontier::{self, FrontierEntry};
use fl_solver::t3_labels::{completion_value, open_patterns};
use fl_solver::Card as FlCard;

use super::evaluator;
use super::play_roots::{self, PlayRequest};
use super::FlEv;

const RANKS: &[u8] = b"23456789TJQKA";
const SUITS: &[u8] = b"shdc";

fn name_of(card: &FlCard, jokers_named: &mut usize) -> String {
    if card.is_joker() {
        *jokers_named += 1;
        format!("X{}", *jokers_named)
    } else {
        format!(
            "{}{}",
            RANKS[(card.rank - 2) as usize] as char,
            SUITS[card.suit as usize] as char
        )
    }
}

fn fl_card_of(name: &str) -> Result<FlCard> {
    if name == "X1" || name == "X2" {
        return Ok(FlCard { rank: 0, suit: 4 });
    }
    let bytes = name.as_bytes();
    if bytes.len() != 2 {
        bail!("bad card name {name}");
    }
    let rank = RANKS
        .iter()
        .position(|r| *r == bytes[0])
        .ok_or_else(|| anyhow!("bad rank in {name}"))? as u8
        + 2;
    let suit = SUITS
        .iter()
        .position(|s| *s == bytes[1])
        .ok_or_else(|| anyhow!("bad suit in {name}"))? as u8;
    Ok(FlCard { rank, suit })
}

fn core_of(cards: &[FlCard]) -> Vec<ofc_core::Card> {
    cards
        .iter()
        .map(|c| ofc_core::Card {
            rank: c.rank,
            suit: c.suit,
        })
        .collect()
}

/// A finished board reduced to what settlement needs.
#[derive(Clone, Copy)]
pub struct Finished {
    pub busted: bool,
    pub top: u32,
    pub mid: u32,
    pub bot: u32,
    pub royalty: i32,
    /// 14..17 when the top row earns Fantasyland, 0 otherwise.
    pub entry_width: u8,
}

fn finish_of(rows: &[Vec<FlCard>; 3]) -> Finished {
    let top = core_of(&rows[0]);
    let mid = core_of(&rows[1]);
    let bot = core_of(&rows[2]);
    let eval = ofc_core::evaluate_board_with_joker_constraint(&top, &mid, &bot);
    if eval.busted {
        return Finished {
            busted: true,
            top: 0,
            mid: 0,
            bot: 0,
            royalty: 0,
            entry_width: 0,
        };
    }
    let (qualifies, width) = ofc_core::check_fl_entry(&eval.top);
    Finished {
        busted: false,
        top: ofc_core::evaluate_hand_value(&eval.top, 3),
        mid: ofc_core::evaluate_hand_value(&eval.mid, 5),
        bot: ofc_core::evaluate_hand_value(&eval.bot, 5),
        royalty: ofc_core::get_top_royalty(&eval.top)
            + ofc_core::get_middle_royalty(&eval.mid)
            + ofc_core::get_bottom_royalty(&eval.bot),
        entry_width: if qualifies { width } else { 0 },
    }
}

/// A finished board in the own-hand objective the chain was trained under
/// (owner, 2026-08-14): royalties plus the Fantasyland entry it earns, a foul
/// flat at -6, and no opponent anywhere in it.
///
/// This is the T4 greedy's own comparator, lifted out so the vs-FL referee
/// scores hands with the identical expression rather than a copy of it.  The
/// two drifting apart would mean the referee ranks openings by a yardstick the
/// chain's last street does not optimise, which is the whole failure mode a
/// referee exists to rule out.  Note the foul is -6 flat, not royalties minus
/// six: a fouled board earns nothing.
pub fn own_worth(finished: &Finished, table: &[f64; 4]) -> f64 {
    if finished.busted {
        -6.0
    } else {
        finished.royalty as f64
            + if finished.entry_width >= 14 {
                table[(finished.entry_width - 14) as usize]
            } else {
                0.0
            }
    }
}

fn sign(own: u32, other: u32) -> i32 {
    match own.cmp(&other) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

/// Real points, from the first player's side.  A foul pays six plus the
/// opponent's royalties and wins nothing; two fouls wash.
pub fn settle(a: &Finished, b: &Finished) -> i32 {
    match (a.busted, b.busted) {
        (true, true) => 0,
        (true, false) => -(6 + b.royalty),
        (false, true) => 6 + a.royalty,
        (false, false) => {
            let lines = frontier::scoop_aware_line(
                sign(a.top, b.top),
                sign(a.mid, b.mid),
                sign(a.bot, b.bot),
            );
            lines + a.royalty - b.royalty
        }
    }
}

/// One street of a traced normal hand, all cards by name (X1/X2 kept apart).
#[derive(Clone)]
pub struct StreetState {
    pub board_before: [Vec<String>; 3],
    pub dead_before: Vec<String>,
    pub draw: Vec<String>,
    pub board_after: [Vec<String>; 3],
}

/// A whole normal hand, street by street, plus the finished board.
pub struct NormalTrace {
    pub streets: [StreetState; 5],
    pub finished: Finished,
}

fn board_names(board: &super::play_roots::PlayedBoard) -> [Vec<String>; 3] {
    [board.top.clone(), board.middle.clone(), board.bottom.clone()]
}

/// The chain plays one normal hand from seventeen cards: T0-T2 by the trained
/// choosers, T3 and T4 by exact own-hand solves.
#[allow(clippy::too_many_arguments)]
fn play_normal(
    id: &str,
    cards: &[FlCard],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    table: &[f64; 4],
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    t2: &evaluator::Model,
) -> Result<Finished> {
    play_normal_traced(id, cards, fl_ev, fl_table, table, t0, t1, t2).map(|t| t.finished)
}

/// `play_normal`, keeping what it saw and did at every street.  The HU root
/// emitter needs the street-by-street boards of both seats to reconstruct
/// what each decision could legally see.
#[allow(clippy::too_many_arguments)]
pub fn play_normal_traced(
    id: &str,
    cards: &[FlCard],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    table: &[f64; 4],
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    t2: &evaluator::Model,
) -> Result<NormalTrace> {
    play_normal_traced_forced(id, cards, fl_ev, fl_table, table, t0, t1, t2, None, None, None)
}

/// `play_normal_traced` with T0 optionally pinned.
///
/// The forced board is spelled in the names `name_of` gives these seventeen
/// cards, jokers numbered in deal order -- so a caller that pins an opening
/// must have named the same five cards the same way.  Nothing else about the
/// chain moves: T1/T2 stay with the trained choosers and T3/T4 stay on the
/// own-worth greedy, because a referee that upgraded the chain would be
/// scoring a hand production never plays.
///
/// `t2_fence` is the T2 cascade, passed straight through to
/// `play_roots::play_trace_forced`; `None` plays the chain unchanged.
#[allow(clippy::too_many_arguments)]
pub fn play_normal_traced_forced(
    id: &str,
    cards: &[FlCard],
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    table: &[f64; 4],
    t0: &evaluator::Model,
    t1: &evaluator::Model,
    t2: &evaluator::Model,
    forced_t0: Option<&[Vec<String>; 3]>,
    forced_t1: Option<&([Vec<String>; 3], String)>,
    t2_fence: Option<(&evaluator::Model, usize)>,
) -> Result<NormalTrace> {
    debug_assert_eq!(cards.len(), 17);
    let mut jokers_named = 0usize;
    let names: Vec<String> = cards.iter().map(|c| name_of(c, &mut jokers_named)).collect();
    let request = PlayRequest {
        id: id.to_string(),
        cards: names[..14].to_vec(),
    };
    let (trace, _timing) = play_roots::play_trace_forced(
        &request, fl_ev, fl_table, t0, t1, t2, forced_t0, forced_t1, t2_fence,
    )?;

    let mut rows: [Vec<FlCard>; 3] = Default::default();
    let mut rows_names: [Vec<String>; 3] = Default::default();
    for (slot, row) in trace.t3.rows.iter().enumerate() {
        rows[slot] = row
            .iter()
            .map(|n| fl_card_of(n))
            .collect::<Result<Vec<_>>>()?;
        rows_names[slot] = row.clone();
    }
    let draw: Vec<FlCard> = trace
        .t3
        .draw
        .iter()
        .map(|n| fl_card_of(n))
        .collect::<Result<Vec<_>>>()?;
    let draw_names: Vec<String> = trace.t3.draw.clone();

    // The unseen pool for the exact T3 solve, by name so the two jokers keep
    // their count.
    let mut seen: std::collections::BTreeSet<&str> =
        trace.t3.dead.iter().map(|s| s.as_str()).collect();
    for row in &trace.t3.rows {
        for n in row {
            seen.insert(n);
        }
    }
    for n in &trace.t3.draw {
        seen.insert(n);
    }
    let unseen: Vec<FlCard> = super::all_cards()
        .into_iter()
        .filter(|n| !seen.contains(n.as_str()))
        .map(|n| fl_card_of(&n))
        .collect::<Result<Vec<_>>>()?;

    // T3: every placement priced exactly (all C(40,3) T4 draws through the
    // pair table), hero's own worth at the leaf.
    let mut best_value = f64::NEG_INFINITY;
    let mut board11: Option<[Vec<FlCard>; 3]> = None;
    let mut board11_names: Option<[Vec<String>; 3]> = None;
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|k| *k != discard).collect();
        for pattern in open_patterns(&rows) {
            let mut after = rows.clone();
            after[pattern[0]].push(draw[kept[0]]);
            after[pattern[1]].push(draw[kept[1]]);
            let value = completion_value(&after, &unseen, &[], table, true, None).mean();
            if value > best_value {
                best_value = value;
                let mut names_after = rows_names.clone();
                names_after[pattern[0]].push(draw_names[kept[0]].clone());
                names_after[pattern[1]].push(draw_names[kept[1]].clone());
                board11 = Some(after);
                board11_names = Some(names_after);
            }
        }
    }
    let board11 = board11.ok_or_else(|| anyhow!("{id}: no legal T3 placement"))?;
    let board11_names = board11_names.expect("names travel with the boards");

    // T4: the actual three cards, placed for the best own worth.
    let t4_draw = &cards[14..17];
    let t4_names: Vec<String> = names[14..17].to_vec();
    let mut best_own = f64::NEG_INFINITY;
    let mut final_board: Option<Finished> = None;
    let mut final_names: Option<[Vec<String>; 3]> = None;
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|k| *k != discard).collect();
        for pattern in open_patterns(&board11) {
            let mut after = board11.clone();
            after[pattern[0]].push(t4_draw[kept[0]]);
            after[pattern[1]].push(t4_draw[kept[1]]);
            let finished = finish_of(&after);
            let own = own_worth(&finished, table);
            if own > best_own {
                best_own = own;
                let mut names_after = board11_names.clone();
                names_after[pattern[0]].push(t4_names[kept[0]].clone());
                names_after[pattern[1]].push(t4_names[kept[1]].clone());
                final_board = Some(finished);
                final_names = Some(names_after);
            }
        }
    }
    let finished = final_board.ok_or_else(|| anyhow!("{id}: no legal T4 placement"))?;
    let final_names = final_names.expect("names travel with the boards");

    // What each street threw away, recovered as draw minus what the after
    // board gained.
    fn discard_of(
        before: &[Vec<String>; 3],
        after: &[Vec<String>; 3],
        draw: &[String],
    ) -> Vec<String> {
        let mut gained: Vec<&String> = Vec::new();
        for row in 0..3 {
            let mut base: Vec<&String> = before[row].iter().collect();
            for name in &after[row] {
                if let Some(slot) = base.iter().position(|b| *b == name) {
                    base.remove(slot);
                } else {
                    gained.push(name);
                }
            }
        }
        draw.iter()
            .filter(|n| !gained.contains(n))
            .cloned()
            .collect()
    }

    let empty: [Vec<String>; 3] = Default::default();
    let t0_after = board_names(&trace.t1.board);
    let t1_after = [
        trace.t2.rows[0].clone(),
        trace.t2.rows[1].clone(),
        trace.t2.rows[2].clone(),
    ];
    let t2_after = rows_names.clone();
    let mut t4_dead = trace.t3.dead.clone();
    t4_dead.extend(discard_of(&t2_after, &board11_names, &trace.t3.draw));
    let streets = [
        StreetState {
            board_before: empty,
            dead_before: Vec::new(),
            draw: names[..5].to_vec(),
            board_after: t0_after.clone(),
        },
        StreetState {
            board_before: t0_after.clone(),
            dead_before: Vec::new(),
            draw: trace.t1.draw.clone(),
            board_after: t1_after.clone(),
        },
        StreetState {
            board_before: t1_after,
            dead_before: trace.t2.dead.clone(),
            draw: trace.t2.draw.clone(),
            board_after: t2_after.clone(),
        },
        StreetState {
            board_before: t2_after,
            dead_before: trace.t3.dead.clone(),
            draw: trace.t3.draw.clone(),
            board_after: board11_names.clone(),
        },
        StreetState {
            board_before: board11_names,
            dead_before: t4_dead,
            draw: t4_names,
            board_after: final_names,
        },
    ];
    Ok(NormalTrace { streets, finished })
}

/// A Fantasyland hand: the whole width at once, blind or as a best response.
pub(crate) fn play_fl(cards: &[FlCard], width: u8, table: &[f64; 4], opponent: Option<&Finished>) -> (Finished, bool) {
    let stay_value = table[(width - 14) as usize];
    let entries = frontier::build_frontier(cards, stay_value);
    let pick: &FrontierEntry = match opponent {
        Some(other) if !other.busted => entries
            .iter()
            .max_by(|a, b| {
                a.score_against(other.top, other.mid, other.bot)
                    .partial_cmp(&b.score_against(other.top, other.mid, other.bot))
                    .unwrap()
            })
            .expect("a Fantasyland hand always has an arrangement"),
        // Blind, or the opponent's rows are dead and cannot push back.
        _ => entries
            .iter()
            .max_by(|a, b| a.static_value.partial_cmp(&b.static_value).unwrap())
            .expect("a Fantasyland hand always has an arrangement"),
    };
    (
        Finished {
            busted: false,
            top: pick.top,
            mid: pick.mid,
            bot: pick.bot,
            royalty: pick.royalty,
            entry_width: 0,
        },
        pick.stays,
    )
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Normal,
    Fl(u8),
}

impl State {
    fn label(&self) -> String {
        match self {
            State::Normal => "normal".to_string(),
            State::Fl(width) => format!("fl{width}"),
        }
    }
}

pub struct Config {
    pub hands: usize,
    pub workers: usize,
    pub seed: u64,
    pub stack: i32,
    pub gap: i32,
}

/// One side's three choosers.  A mirror match passes the same set twice; an
/// A/B match seats two generations at one table, alternating positions like
/// any other pair of players.
pub struct Arm<'a> {
    pub t0: &'a evaluator::Model,
    pub t1: &'a evaluator::Model,
    pub t2: &'a evaluator::Model,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &Config,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    arms: &[Arm<'_>; 2],
    output: &std::path::Path,
) -> Result<()> {
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
                let mut hand_in_session = 0usize;
                for local in 0..per_worker {
                    let global = (worker * per_worker + local) as u64;
                    // The first-position player alternates every hand.
                    let first = (global % 2) as usize;
                    let second = 1 - first;
                    let need = |state: State| -> usize {
                        match state {
                            State::Normal => 17,
                            State::Fl(width) => width as usize,
                        }
                    };
                    let dealt = fl_solver::pool::deal(
                        config.seed,
                        global,
                        need(states[first]) + need(states[second]),
                    );
                    let (cards_first, cards_second) = dealt.split_at(need(states[first]));
                    let mut hands: [&[FlCard]; 2] = [&[], &[]];
                    hands[first] = cards_first;
                    hands[second] = cards_second;

                    // Play.  A normal player always finishes before a
                    // Fantasyland player responds; two Fantasyland players are
                    // simultaneous and blind.
                    let mut finished: [Option<Finished>; 2] = [None, None];
                    let mut stays = [false, false];
                    for p in 0..2 {
                        if let State::Normal = states[p] {
                            finished[p] = Some(play_normal(
                                &format!("sp/{}/{}", config.seed, global * 2 + p as u64),
                                hands[p],
                                fl_ev,
                                fl_table,
                                &table,
                                arms[p].t0,
                                arms[p].t1,
                                arms[p].t2,
                            )?);
                        }
                    }
                    for p in 0..2 {
                        if let State::Fl(width) = states[p] {
                            let opponent = match states[1 - p] {
                                State::Normal => finished[1 - p].as_ref(),
                                State::Fl(_) => None,
                            };
                            let (board, stay) = play_fl(hands[p], width, &table, opponent);
                            finished[p] = Some(board);
                            stays[p] = stay;
                        }
                    }
                    let a = finished[0].expect("player 0 played");
                    let b = finished[1].expect("player 1 played");

                    let raw = settle(&a, &b);
                    let paid = if raw > 0 {
                        raw.min(stacks[1])
                    } else {
                        -((-raw).min(stacks[0]))
                    };
                    stacks[0] += paid;
                    stacks[1] -= paid;

                    let next = |state: State, finished: &Finished, stay: bool| -> State {
                        match state {
                            State::Normal => {
                                if !finished.busted && finished.entry_width >= 14 {
                                    State::Fl(finished.entry_width)
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
                    let pending = [next(states[0], &a, stays[0]), next(states[1], &b, stays[1])];

                    let end = if stacks[0] == 0 || stacks[1] == 0 {
                        Some("zero")
                    } else if (stacks[0] - stacks[1]).abs() >= config.gap
                        && pending[0] == State::Normal
                        && pending[1] == State::Normal
                    {
                        Some("gap")
                    } else {
                        None
                    };

                    lines.push(format!(
                        "{{\"worker\":{worker},\"session\":{session},\"hand\":{hand_in_session},\
                         \"global\":{global},\"first\":{first},\
                         \"state_a\":\"{}\",\"state_b\":\"{}\",\
                         \"settle_raw\":{raw},\"settle_paid\":{paid},\
                         \"royalty_a\":{},\"royalty_b\":{},\
                         \"foul_a\":{},\"foul_b\":{},\
                         \"entry_a\":{},\"entry_b\":{},\
                         \"stay_a\":{},\"stay_b\":{},\
                         \"stack_a\":{},\"stack_b\":{},\
                         \"end\":{}}}",
                        states[0].label(),
                        states[1].label(),
                        a.royalty,
                        b.royalty,
                        a.busted,
                        b.busted,
                        a.entry_width,
                        b.entry_width,
                        stays[0],
                        stays[1],
                        stacks[0],
                        stacks[1],
                        match end {
                            Some(reason) => format!("\"{reason}\""),
                            None => "null".to_string(),
                        },
                    ));

                    if let Some(_reason) = end {
                        stacks = [config.stack, config.stack];
                        states = [State::Normal, State::Normal];
                        session += 1;
                        hand_in_session = 0;
                    } else {
                        states = pending;
                        hand_in_session += 1;
                    }
                }
                Ok(lines)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    use std::io::Write;
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
        "self-play: {written} hands, {} workers, {:.1} s ({:.1} ms/hand/worker)",
        config.workers,
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() * 1000.0 * config.workers.max(1) as f64
            / written.max(1) as f64
    );
    Ok(())
}
