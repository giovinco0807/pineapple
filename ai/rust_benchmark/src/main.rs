//! OFC Pineapple Benchmark
//!
//! Hero: MCTS(T0) + RolloutEvaluator(T1+)
//! Opponent: BC greedy (benchmark mode) or same AI (selfplay mode)
//!
//! Usage:
//!   cargo run --release -- --games 5000 --bc models/bc.onnx --vn models/vn.onnx
//!   cargo run --release -- --games 5000 --mode selfplay --output data/selfplay.jsonl

mod types;
mod encoding;
mod onnx_models;
mod game_engine;
mod rollout;
mod mcts;
mod fl_placement;

use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::Serialize;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use types::*;
use encoding::*;
use onnx_models::*;
use game_engine::*;
use rollout::*;
use mcts::*;

#[derive(Parser)]
#[command(name = "ofc_benchmark")]
#[command(about = "OFC Pineapple AI Benchmark")]
struct Args {
    /// Number of games to play
    #[arg(long, default_value_t = 5000)]
    games: u32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of parallel threads
    #[arg(long, default_value_t = 1)]
    threads: u32,

    /// MCTS simulations for T0
    #[arg(long, default_value_t = 400)]
    mcts_sims: u32,

    /// Rollout count for T1+
    #[arg(long, default_value_t = 250)]
    rollouts: u32,

    /// VN prefilter top-K for rollout
    #[arg(long, default_value_t = 10)]
    vn_top_k: u32,

    /// PUCT exploration constant
    #[arg(long, default_value_t = 1.5)]
    c_puct: f32,

    /// BC temperature for rollout playout
    #[arg(long, default_value_t = 0.8)]
    bc_temperature: f32,

    /// MCTS bust penalty (VN bust_prob * penalty)
    #[arg(long, default_value_t = 0.0)]
    bust_penalty: f32,

    /// MCTS FL EV scale
    #[arg(long, default_value_t = 1.0)]
    fl_ev_scale: f32,

    /// Path to common BC ONNX model
    #[arg(long, default_value = "models/bc.onnx")]
    bc: String,

    /// Path to VN ONNX model
    #[arg(long, default_value = "models/vn.onnx")]
    vn: String,

    /// Per-turn BC model paths (T1-T4)
    #[arg(long)]
    bc_t1: Option<String>,
    #[arg(long)]
    bc_t2: Option<String>,
    #[arg(long)]
    bc_t3: Option<String>,
    #[arg(long)]
    bc_t4: Option<String>,

    /// VN truncation depth for T1+ (0=disabled, 1=1-turn lookahead + VN)
    #[arg(long, default_value_t = 0)]
    vn_truncate_depth: u32,

    /// Rollout count per candidate in VN-truncated mode
    #[arg(long, default_value_t = 500)]
    vn_truncate_n: u32,

    /// Hybrid mode: blend full rollout score + VN prediction (0.5/0.5)
    #[arg(long, default_value_t = false)]
    vn_hybrid: bool,

    /// Mode: "benchmark" (hero vs BC greedy) or "selfplay" (hero vs hero)
    #[arg(long, default_value = "benchmark")]
    mode: String,

    /// Output JSONL path for self-play data logging
    #[arg(long)]
    output: Option<String>,

    /// Include 520-dim state vectors in JSONL (large, default off)
    #[arg(long, default_value_t = false)]
    log_states: bool,
}

/// Per-game result (for statistics)
#[derive(Clone)]
struct GameRecord {
    busted: bool,
    fl_entry: bool,       // hero qualifies for FL (QQ+ on top)
    is_fl_hand: bool,     // this hand was played as FL (not normal)
    fl_type: Option<String>,
    royalty: i32,
    normal_score: f64,
    fl_score: f64,
    total_score: f64,
    duration_ms: f64,
    // Opponent info for JSONL
    opp_busted: bool,
    opp_fl: bool,
    opp_fl_cards: u8,
    opp_royalty: i32,
    hero_fl_cards: u8,
    // Session info
    chips_before: [i32; 2],  // chips at start of this hand
    chips_after: [i32; 2],   // chips after this hand
    chip_delta: i32,         // hero chip change
    session_id: u32,
}

/// Per-turn log entry for JSONL output
#[derive(Serialize)]
struct TurnLog {
    turn: u8,
    player: String,       // "hero" or "opp"
    is_btn: bool,
    chips_self: i32,
    chips_opponent: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<Vec<f32>>,
    board_self: BoardJson,
    board_opponent: BoardJson,
    dealt_cards: Vec<u8>,
    discards_self: Vec<u8>,
    action_idx: usize,
    n_actions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    visit_counts: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    q_values: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_evs: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k_indices: Option<Vec<usize>>,
}

/// Board state for JSON serialization
#[derive(Serialize, Clone)]
struct BoardJson {
    top: Vec<u8>,
    middle: Vec<u8>,
    bottom: Vec<u8>,
}

impl BoardJson {
    fn from_board(b: &Board) -> Self {
        BoardJson {
            top: b.top[..b.top_n as usize].to_vec(),
            middle: b.mid[..b.mid_n as usize].to_vec(),
            bottom: b.bot[..b.bot_n as usize].to_vec(),
        }
    }
}

/// Game-level JSONL record
#[derive(Serialize)]
struct GameLog {
    game_id: u32,
    seed: u64,
    mode: String,
    session_id: u32,
    is_fl_hand: bool,
    fl_active: [bool; 2],       // which seats were in FL this hand
    fl_card_count: [u8; 2],     // inherited FL card counts (0 if not in FL)
    chips_before: [i32; 2],
    chips_after: [i32; 2],
    turns: Vec<TurnLog>,
    result: GameResultJson,
}

#[derive(Serialize)]
struct GameResultJson {
    normal_score: f64,
    fl_score: f64,
    total_score: f64,
    hero_busted: bool,
    hero_fl: bool,           // FL entry (from normal hand)
    hero_fl_stay: bool,      // FL stay (from FL hand)
    hero_fl_type: Option<String>,
    hero_fl_cards: u8,       // new FL entry card count
    hero_royalty: i32,
    opp_busted: bool,
    opp_fl: bool,
    opp_fl_stay: bool,
    opp_fl_cards: u8,
    opp_royalty: i32,
}

/// Play one full game with optional turn logging
fn play_one_game(
    deck: &[CardIdx],
    models: &mut OnnxModels,
    mcts_config: &MctsConfig,
    rollout_config: &RolloutConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut StdRng,
    selfplay: bool,
    log_states: bool,
    hero_is_btn: bool,
    chips: [i32; 2],  // current session chips [hero, opp]
) -> (GameRecord, GameResult, Vec<TurnLog>) {
    let start = Instant::now();
    let hero: usize = 0;
    let opp: usize = 1;
    // Seat order: btn_seat acts second in T0, first in T1+
    let btn_seat: usize = if hero_is_btn { hero } else { opp };
    let bb_seat: usize = 1 - btn_seat;
    let mut turn_logs: Vec<TurnLog> = Vec::new();

    let mut boards = [Board::new(), Board::new()];
    let mut discards: [Vec<CardIdx>; 2] = [Vec::new(), Vec::new()];
    let mut deck_idx: usize = 0;

    // T0: deal 5 cards to each seat (seat 0 = hero, seat 1 = opp)
    let cards_t0: [[CardIdx; 5]; 2] = [
        [deck[0], deck[1], deck[2], deck[3], deck[4]],
        [deck[5], deck[6], deck[7], deck[8], deck[9]],
    ];
    deck_idx = 10;

    // T0 action order: BB acts first, BTN acts second
    let t0_order = [bb_seat, btn_seat];
    for &seat in &t0_order {
        let is_hero_seat = seat == hero;
        let other = 1 - seat;
        let is_btn_seat = seat == btn_seat;
        let use_mcts = is_hero_seat || selfplay;

        let t0_actions = gen_t0_actions(&cards_t0[seat], &boards[seat]);

        let chosen_idx;
        let mut mcts_result: Option<MctsResult> = None;

        let seat_chips = chips[seat] as u16;
        let other_chips = chips[other] as u16;

        if use_mcts {
            let mr = mcts_select_t0_action(
                &boards[seat], &boards[other], &cards_t0[seat], is_btn_seat,
                models, mcts_config, fl_ev, rng,
            );
            chosen_idx = mr.best_action_idx;
            mcts_result = Some(mr);
        } else {
            let obs = Observation {
                board_self: &boards[seat], board_opponent: &boards[other],
                dealt_cards: &cards_t0[seat], known_discards_self: &[],
                turn: 0, is_btn: is_btn_seat, chips_self: seat_chips, chips_opponent: other_chips,
            };
            let state = encode_state(&obs);
            let logits = models.bc_inference(&state, 0);
            chosen_idx = OnnxModels::bc_select_greedy(&logits, t0_actions.len());
        }

        // Log (hero always, opp only in selfplay)
        if is_hero_seat || selfplay {
            turn_logs.push(TurnLog {
                turn: 0,
                player: if is_hero_seat { "hero" } else { "opp" }.to_string(),
                is_btn: is_btn_seat,
                chips_self: chips[seat],
                chips_opponent: chips[other],
                state: if log_states {
                    let obs = Observation {
                        board_self: &boards[seat], board_opponent: &boards[other],
                        dealt_cards: &cards_t0[seat], known_discards_self: &[],
                        turn: 0, is_btn: is_btn_seat, chips_self: seat_chips, chips_opponent: other_chips,
                    };
                    Some(encode_state(&obs).to_vec())
                } else { None },
                board_self: BoardJson::from_board(&boards[seat]),
                board_opponent: BoardJson::from_board(&boards[other]),
                dealt_cards: cards_t0[seat].to_vec(),
                discards_self: Vec::new(),
                action_idx: chosen_idx,
                n_actions: t0_actions.len(),
                visit_counts: mcts_result.as_ref().map(|r| r.visit_counts.clone()),
                q_values: mcts_result.as_ref().map(|r| r.q_values.clone()),
                action_evs: None,
                top_k_indices: None,
            });
        }

        let action = &t0_actions[chosen_idx];
        for &(card, row) in action {
            boards[seat].place_mut(card, row);
        }
    }

    // T1-T4: BTN acts first, BB acts second
    let turn_order = [btn_seat, bb_seat];
    for turn in 1..5u8 {
        if boards[0].is_complete() && boards[1].is_complete() { break; }
        if deck_idx + 6 > deck.len() { break; }

        // Deal: BTN gets first 3 cards
        let mut deal: [[CardIdx; 3]; 2] = [[0; 3]; 2];
        deal[btn_seat] = [deck[deck_idx], deck[deck_idx + 1], deck[deck_idx + 2]];
        deal[bb_seat] = [deck[deck_idx + 3], deck[deck_idx + 4], deck[deck_idx + 5]];
        deck_idx += 6;

        for &seat in &turn_order {
            if boards[seat].is_complete() { continue; }
            let is_hero_seat = seat == hero;
            let other = 1 - seat;
            let is_btn_seat = seat == btn_seat;
            let seat_chips = chips[seat] as u16;
            let other_chips = chips[other] as u16;

            let actions = gen_turn_actions(&deal[seat], &boards[seat]);
            if actions.is_empty() { continue; }

            let (chosen_idx, rollout_result) = if boards[seat].total_cards() == 11 {
                (direct_eval_last_turn(&boards[seat], &boards[other], &actions, fl_ev), None)
            } else if is_hero_seat || selfplay {
                let rr = rollout_select_action(
                    &boards[seat], &boards[other], &deal[seat], turn,
                    &discards[seat], is_btn_seat, models, rollout_config, fl_ev, rng,
                );
                let idx = rr.best_action_idx;
                (idx, Some(rr))
            } else {
                // Benchmark: BC greedy for opponent
                let obs = Observation {
                    board_self: &boards[seat], board_opponent: &boards[other],
                    dealt_cards: &deal[seat], known_discards_self: &discards[seat],
                    turn, is_btn: is_btn_seat, chips_self: seat_chips, chips_opponent: other_chips,
                };
                let state = encode_state(&obs);
                let logits = models.bc_inference(&state, turn);
                (OnnxModels::bc_select_greedy(&logits, actions.len()), None)
            };

            // Log (hero always, opp only in selfplay)
            if is_hero_seat || selfplay {
                turn_logs.push(TurnLog {
                    turn,
                    player: if is_hero_seat { "hero" } else { "opp" }.to_string(),
                    is_btn: is_btn_seat,
                    chips_self: chips[seat],
                    chips_opponent: chips[other],
                    state: if log_states {
                        let obs = Observation {
                            board_self: &boards[seat], board_opponent: &boards[other],
                            dealt_cards: &deal[seat], known_discards_self: &discards[seat],
                            turn, is_btn: is_btn_seat, chips_self: seat_chips, chips_opponent: other_chips,
                        };
                        Some(encode_state(&obs).to_vec())
                    } else { None },
                    board_self: BoardJson::from_board(&boards[seat]),
                    board_opponent: BoardJson::from_board(&boards[other]),
                    dealt_cards: deal[seat].to_vec(),
                    discards_self: discards[seat].clone(),
                    action_idx: chosen_idx,
                    n_actions: actions.len(),
                    visit_counts: None,
                    q_values: None,
                    action_evs: rollout_result.as_ref().map(|r| r.action_evs.clone()),
                    top_k_indices: rollout_result.as_ref().map(|r| r.top_k_indices.clone()),
                });
            }

            let chosen = &actions[chosen_idx];
            for &(card, row) in &chosen.placements {
                boards[seat].place_mut(card, row);
            }
            discards[seat].push(chosen.discard);
        }
    }

    let result = compute_game_result(&boards, fl_ev);
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    let hero_fl = result.fl_entry[hero] && !result.busted[hero];
    let opp_fl = result.fl_entry[1 - hero] && !result.busted[1 - hero];
    let mut fl_score = 0.0;
    if hero_fl {
        fl_score += fl_ev.get(&result.fl_card_count[hero]).copied().unwrap_or(0.0);
    }
    if opp_fl {
        fl_score -= fl_ev.get(&result.fl_card_count[1 - hero]).copied().unwrap_or(0.0);
    }

    let record = GameRecord {
        busted: result.busted[hero],
        fl_entry: hero_fl,
        is_fl_hand: false,
        fl_type: result.fl_type.clone(),
        royalty: result.royalties[hero],
        normal_score: result.raw_score,
        fl_score,
        total_score: result.raw_score + fl_score,
        duration_ms,
        opp_busted: result.busted[1 - hero],
        opp_fl,
        opp_fl_cards: result.fl_card_count[1 - hero],
        opp_royalty: result.royalties[1 - hero],
        hero_fl_cards: result.fl_card_count[hero],
        // Session fields filled by caller
        chips_before: [0; 2],
        chips_after: [0; 2],
        chip_delta: 0,
        session_id: 0,
    };

    (record, result, turn_logs)
}

/// Direct evaluation for the last turn (card_count == 11)
fn direct_eval_last_turn(
    board: &Board,
    opp_board: &Board,
    actions: &[TurnAction],
    fl_ev: &HashMap<u8, f64>,
) -> usize {
    if actions.len() <= 1 { return 0; }

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (i, action) in actions.iter().enumerate() {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board.place_mut(card, row);
        }
        let score = compute_rollout_score(&new_board, opp_board, fl_ev);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

/// Play one FL hand (one or both players in Fantasyland)
///
/// Game flow:
/// - Both FL: deal FL cards to each, solve independently (max royalty + stay bonus)
/// - One FL, one normal: normal player plays T0-T4 first (FL board invisible),
///   then FL player places all 13 cards at once against completed opponent board
fn play_fl_hand(
    deck: &[CardIdx],
    fl_active: [bool; 2],
    fl_card_count: [u8; 2],
    models: &mut OnnxModels,
    mcts_config: &MctsConfig,
    rollout_config: &RolloutConfig,
    fl_ev: &HashMap<u8, f64>,
    rng: &mut StdRng,
    selfplay: bool,
    log_states: bool,
    hero_is_btn: bool,
    chips: [i32; 2],
) -> (GameRecord, GameResult, Vec<TurnLog>) {
    let start = Instant::now();
    let hero: usize = 0;
    let opp: usize = 1;
    let btn_seat: usize = if hero_is_btn { hero } else { opp };
    let mut turn_logs: Vec<TurnLog> = Vec::new();
    let mut boards = [Board::new(), Board::new()];
    let mut discards: [Vec<CardIdx>; 2] = [Vec::new(), Vec::new()];
    let mut deck_idx: usize = 0;

    let both_fl = fl_active[0] && fl_active[1];

    if both_fl {
        // ===== BOTH FL =====
        // Deal FL cards to each player from deck
        let fl_count = [fl_card_count[0] as usize, fl_card_count[1] as usize];
        let fl_dealt: [Vec<CardIdx>; 2] = [
            deck[0..fl_count[0]].to_vec(),
            deck[fl_count[0]..fl_count[0] + fl_count[1]].to_vec(),
        ];

        // Each player solves independently (max royalty + stay bonus)
        for seat in 0..2 {
            if let Some((board, disc)) = fl_placement::solve_fl_both(&fl_dealt[seat]) {
                boards[seat] = board;
                discards[seat] = disc;
            }
        }
    } else {
        // ===== ONE FL, ONE NORMAL =====
        let fl_seat = if fl_active[0] { 0 } else { 1 };
        let normal_seat = 1 - fl_seat;
        let fl_count = fl_card_count[fl_seat] as usize;

        // Deal FL cards first (invisible to normal player)
        let fl_dealt: Vec<CardIdx> = deck[0..fl_count].to_vec();
        deck_idx = fl_count;

        // Normal player plays T0-T4 (sees empty FL board)
        let is_hero_normal = normal_seat == hero;
        let is_btn_normal = normal_seat == btn_seat;
        let use_smart_ai = is_hero_normal || selfplay;
        let normal_chips = chips[normal_seat] as u16;
        let fl_chips = chips[fl_seat] as u16;

        // --- T0 ---
        let t0_cards: [CardIdx; 5] = [
            deck[deck_idx], deck[deck_idx+1], deck[deck_idx+2],
            deck[deck_idx+3], deck[deck_idx+4],
        ];
        deck_idx += 5;

        let t0_actions = gen_t0_actions(&t0_cards, &boards[normal_seat]);
        let chosen_idx;
        let mut mcts_result: Option<MctsResult> = None;

        if use_smart_ai {
            let mr = mcts_select_t0_action(
                &boards[normal_seat], &boards[fl_seat], &t0_cards, is_btn_normal,
                models, mcts_config, fl_ev, rng,
            );
            chosen_idx = mr.best_action_idx;
            mcts_result = Some(mr);
        } else {
            let obs = Observation {
                board_self: &boards[normal_seat],
                board_opponent: &boards[fl_seat],
                dealt_cards: &t0_cards,
                known_discards_self: &[],
                turn: 0, is_btn: is_btn_normal,
                chips_self: normal_chips, chips_opponent: fl_chips,
            };
            let state = encode_state(&obs);
            let logits = models.bc_inference(&state, 0);
            chosen_idx = OnnxModels::bc_select_greedy(&logits, t0_actions.len());
        }

        // Log T0
        if is_hero_normal || selfplay {
            turn_logs.push(TurnLog {
                turn: 0,
                player: if is_hero_normal { "hero" } else { "opp" }.to_string(),
                is_btn: is_btn_normal,
                chips_self: chips[normal_seat],
                chips_opponent: chips[fl_seat],
                state: if log_states {
                    let obs = Observation {
                        board_self: &boards[normal_seat],
                        board_opponent: &boards[fl_seat],
                        dealt_cards: &t0_cards,
                        known_discards_self: &[],
                        turn: 0, is_btn: is_btn_normal,
                        chips_self: normal_chips, chips_opponent: fl_chips,
                    };
                    Some(encode_state(&obs).to_vec())
                } else { None },
                board_self: BoardJson::from_board(&boards[normal_seat]),
                board_opponent: BoardJson::from_board(&boards[fl_seat]),
                dealt_cards: t0_cards.to_vec(),
                discards_self: Vec::new(),
                action_idx: chosen_idx,
                n_actions: t0_actions.len(),
                visit_counts: mcts_result.as_ref().map(|r| r.visit_counts.clone()),
                q_values: mcts_result.as_ref().map(|r| r.q_values.clone()),
                action_evs: None,
                top_k_indices: None,
            });
        }

        let action = &t0_actions[chosen_idx];
        for &(card, row) in action {
            boards[normal_seat].place_mut(card, row);
        }

        // --- T1-T4 ---
        for turn in 1..5u8 {
            if boards[normal_seat].is_complete() { break; }
            if deck_idx + 3 > deck.len() { break; }

            let deal: [CardIdx; 3] = [deck[deck_idx], deck[deck_idx+1], deck[deck_idx+2]];
            deck_idx += 3;

            let actions = gen_turn_actions(&deal, &boards[normal_seat]);
            if actions.is_empty() { continue; }

            let (chosen_idx, rollout_result) = if boards[normal_seat].total_cards() == 11 {
                (direct_eval_last_turn(&boards[normal_seat], &boards[fl_seat], &actions, fl_ev), None)
            } else if use_smart_ai {
                let rr = rollout_select_action(
                    &boards[normal_seat], &boards[fl_seat], &deal, turn,
                    &discards[normal_seat], is_btn_normal, models, rollout_config, fl_ev, rng,
                );
                let idx = rr.best_action_idx;
                (idx, Some(rr))
            } else {
                let obs = Observation {
                    board_self: &boards[normal_seat],
                    board_opponent: &boards[fl_seat],
                    dealt_cards: &deal,
                    known_discards_self: &discards[normal_seat],
                    turn, is_btn: is_btn_normal,
                    chips_self: normal_chips, chips_opponent: fl_chips,
                };
                let state = encode_state(&obs);
                let logits = models.bc_inference(&state, turn);
                (OnnxModels::bc_select_greedy(&logits, actions.len()), None)
            };

            // Log
            if is_hero_normal || selfplay {
                turn_logs.push(TurnLog {
                    turn,
                    player: if is_hero_normal { "hero" } else { "opp" }.to_string(),
                    is_btn: is_btn_normal,
                    chips_self: chips[normal_seat],
                    chips_opponent: chips[fl_seat],
                    state: if log_states {
                        let obs = Observation {
                            board_self: &boards[normal_seat],
                            board_opponent: &boards[fl_seat],
                            dealt_cards: &deal,
                            known_discards_self: &discards[normal_seat],
                            turn, is_btn: is_btn_normal,
                            chips_self: normal_chips, chips_opponent: fl_chips,
                        };
                        Some(encode_state(&obs).to_vec())
                    } else { None },
                    board_self: BoardJson::from_board(&boards[normal_seat]),
                    board_opponent: BoardJson::from_board(&boards[fl_seat]),
                    dealt_cards: deal.to_vec(),
                    discards_self: discards[normal_seat].clone(),
                    action_idx: chosen_idx,
                    n_actions: actions.len(),
                    visit_counts: None,
                    q_values: None,
                    action_evs: rollout_result.as_ref().map(|r| r.action_evs.clone()),
                    top_k_indices: rollout_result.as_ref().map(|r| r.top_k_indices.clone()),
                });
            }

            let chosen = &actions[chosen_idx];
            for &(card, row) in &chosen.placements {
                boards[normal_seat].place_mut(card, row);
            }
            discards[normal_seat].push(chosen.discard);
        }

        // FL player places against completed normal board
        if let Some((board, disc)) = fl_placement::solve_fl_vs_normal(
            &fl_dealt, &boards[normal_seat], fl_ev,
        ) {
            boards[fl_seat] = board;
            discards[fl_seat] = disc;
        }
    }

    // Log FL placement(s) as turn=-1
    if both_fl {
        for seat in 0..2 {
            let fl_count = fl_card_count[seat] as usize;
            let dealt_start: usize = if seat == 0 { 0 } else { fl_card_count[0] as usize };
            let fl_dealt: Vec<CardIdx> = deck[dealt_start..dealt_start + fl_count].to_vec();
            let player_name = if seat == hero { "hero" } else { "opponent" };
            turn_logs.push(TurnLog {
                turn: 255, // FL placement marker
                player: player_name.to_string(),
                is_btn: seat == btn_seat,
                chips_self: chips[seat],
                chips_opponent: chips[1 - seat],
                state: None,
                board_self: BoardJson::from_board(&boards[seat]),
                board_opponent: BoardJson::from_board(&boards[1 - seat]),
                dealt_cards: fl_dealt,
                discards_self: discards[seat].clone(),
                action_idx: 0,
                n_actions: 0,
                visit_counts: None,
                q_values: None,
                action_evs: None,
                top_k_indices: None,
            });
        }
    } else {
        let fl_seat = if fl_active[0] { 0 } else { 1 };
        let fl_count = fl_card_count[fl_seat] as usize;
        let fl_dealt: Vec<CardIdx> = deck[0..fl_count].to_vec();
        let player_name = if fl_seat == hero { "hero" } else { "opponent" };
        turn_logs.push(TurnLog {
            turn: 255, // FL placement marker
            player: player_name.to_string(),
            is_btn: fl_seat == btn_seat,
            chips_self: chips[fl_seat],
            chips_opponent: chips[1 - fl_seat],
            state: None,
            board_self: BoardJson::from_board(&boards[fl_seat]),
            board_opponent: BoardJson::from_board(&boards[1 - fl_seat]),
            dealt_cards: fl_dealt,
            discards_self: discards[fl_seat].clone(),
            action_idx: 0,
            n_actions: 0,
            visit_counts: None,
            q_values: None,
            action_evs: None,
            top_k_indices: None,
        });
    }

    // Score the hand
    let result = compute_game_result(&boards, fl_ev);
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    // FL score for FL hands:
    // - Player IN FL: use fl_stay (not fl_entry) for chain continuation
    //   → fl_ev uses inherited card count (fl_card_count arg), not new entry count
    // - Player NOT in FL (normal): use fl_entry for new FL entry
    let mut fl_score = 0.0;
    for seat in 0..2 {
        let sign = if seat == hero { 1.0 } else { -1.0 };
        if fl_active[seat] {
            // Was in FL → chain continues only if fl_stay
            if !result.busted[seat] && result.fl_stay[seat] {
                fl_score += sign * fl_ev.get(&fl_card_count[seat]).copied().unwrap_or(0.0);
            }
        } else {
            // Normal player → new FL entry
            if result.fl_entry[seat] && !result.busted[seat] {
                fl_score += sign * fl_ev.get(&result.fl_card_count[seat]).copied().unwrap_or(0.0);
            }
        }
    }

    let hero_fl = result.fl_entry[hero] && !result.busted[hero];
    let opp_fl = result.fl_entry[opp] && !result.busted[opp];

    let record = GameRecord {
        busted: result.busted[hero],
        fl_entry: hero_fl,
        is_fl_hand: true,
        fl_type: result.fl_type.clone(),
        royalty: result.royalties[hero],
        normal_score: result.raw_score,
        fl_score,
        total_score: result.raw_score + fl_score,
        duration_ms,
        opp_busted: result.busted[opp],
        opp_fl,
        opp_fl_cards: result.fl_card_count[opp],
        opp_royalty: result.royalties[opp],
        hero_fl_cards: result.fl_card_count[hero],
        chips_before: [0; 2],
        chips_after: [0; 2],
        chip_delta: 0,
        session_id: 0,
    };

    (record, result, turn_logs)
}

fn main() {
    let args = Args::parse();
    let selfplay = args.mode == "selfplay";

    println!("============================================================");
    println!("  OFC Pineapple {} (Rust)", if selfplay { "Self-Play" } else { "Benchmark" });
    println!("============================================================");
    println!("  Games: {}, Seed: {}, Threads: {}", args.games, args.seed, args.threads);
    println!("  MCTS sims: {}, Rollouts: {}, VN top-K: {}", args.mcts_sims, args.rollouts, args.vn_top_k);
    if args.vn_truncate_depth > 0 {
        println!("  VN-truncated: depth={}, n={}, hybrid={}", args.vn_truncate_depth, args.vn_truncate_n, args.vn_hybrid);
    }
    println!("  Mode: {}", args.mode);
    if let Some(ref out) = args.output {
        println!("  Output: {}", out);
    }
    println!("  BC: {}", args.bc);
    println!("  VN: {}", args.vn);
    if args.bc_t3.is_some() { println!("  BC T3: {}", args.bc_t3.as_ref().unwrap()); }
    if args.bc_t4.is_some() { println!("  BC T4: {}", args.bc_t4.as_ref().unwrap()); }
    println!();

    // Load models
    let bc_t_paths: [Option<String>; 5] = [
        None,
        args.bc_t1.clone(),
        args.bc_t2.clone(),
        args.bc_t3.clone(),
        args.bc_t4.clone(),
    ];
    let mut models = OnnxModels::load(&args.bc, &args.vn, &bc_t_paths)
        .expect("Failed to load ONNX models");
    println!("  Models loaded.");

    let fl_ev = default_fl_ev();
    let mcts_config = MctsConfig {
        num_simulations: args.mcts_sims,
        c_puct: args.c_puct,
        bust_penalty: args.bust_penalty,
        fl_ev_scale: args.fl_ev_scale,
        value_scale: 30.0,
        score_mean: 0.0,
        score_std: 1.0,
    };
    let rollout_config = RolloutConfig {
        n_rollouts: args.rollouts as usize,
        vn_top_k: args.vn_top_k as usize,
        bc_temperature: args.bc_temperature,
        vn_truncate_depth: args.vn_truncate_depth as usize,
        vn_truncate_n: args.vn_truncate_n as usize,
        vn_hybrid: args.vn_hybrid,
        value_scale: 30.0,
    };

    // Generate decks
    let mut master_rng = StdRng::seed_from_u64(args.seed);
    let mut decks: Vec<Vec<CardIdx>> = Vec::with_capacity(args.games as usize);
    for _ in 0..args.games {
        let mut deck = create_full_deck();
        deck.shuffle(&mut master_rng);
        decks.push(deck);
    }

    // Open JSONL output file if specified
    let mut jsonl_writer: Option<BufWriter<File>> = args.output.as_ref().map(|path| {
        let file = File::create(path).expect("Failed to create output file");
        BufWriter::new(file)
    });

    let total_start = Instant::now();
    let completed = AtomicU32::new(0);

    // ================================================================
    // Session state
    // ================================================================
    let mut chips: [i32; 2] = [200, 200];        // [hero, opp]
    let mut fl_active: [bool; 2] = [false; 2];   // FL/FL-stay active for next hand
    let mut fl_cards: [u8; 2] = [0; 2];          // FL card count (14-17)
    let mut session_btn_parity: usize = master_rng.gen_range(0..2usize); // random initial BTN
    let mut hand_in_session: u32 = 0;
    let mut session_id: u32 = 0;
    let mut total_sessions: u32 = 0;
    let mut fl_hands_played: u32 = 0;
    let mut both_fl_hands: u32 = 0;

    // Play games with session tracking
    let mut results: Vec<GameRecord> = Vec::with_capacity(args.games as usize);
    for (i, deck) in decks.iter().enumerate() {
        // Check session end BEFORE starting hand:
        // diff >= 40 AND neither player is going to FL/FL-stay
        let chip_diff = (chips[0] - chips[1]).abs();
        if chip_diff >= 40 && !fl_active[0] && !fl_active[1] {
            // Session ends → reset
            total_sessions += 1;
            session_id += 1;
            chips = [200, 200];
            fl_active = [false; 2];
            fl_cards = [0; 2];
            session_btn_parity = master_rng.gen_range(0..2usize);
            hand_in_session = 0;
        }

        // BTN alternates within session
        let hero_is_btn = (session_btn_parity + hand_in_session as usize) % 2 == 0;

        let mut rng = StdRng::seed_from_u64(args.seed + i as u64 + 1);
        let chips_before = chips;

        let is_fl_hand = fl_active[0] || fl_active[1];
        // Snapshot FL state before update (for JSONL)
        let fl_was_active_0 = fl_active[0];
        let fl_was_active_1 = fl_active[1];
        let fl_was_cards_0 = fl_cards[0];
        let fl_was_cards_1 = fl_cards[1];
        if is_fl_hand {
            fl_hands_played += 1;
            if fl_active[0] && fl_active[1] { both_fl_hands += 1; }
        }
        let (mut record, game_result, turn_logs) = if is_fl_hand {
            play_fl_hand(
                deck, fl_active, fl_cards,
                &mut models, &mcts_config, &rollout_config, &fl_ev, &mut rng,
                selfplay, args.log_states, hero_is_btn, chips,
            )
        } else {
            play_one_game(
                deck, &mut models, &mcts_config, &rollout_config, &fl_ev, &mut rng,
                selfplay, args.log_states, hero_is_btn, chips,
            )
        };

        // ============================================================
        // Chip transfer (zero-sum, floor at 0)
        // ============================================================
        // raw_score is from hero (seat 0) perspective
        // Use normal_score only for chips (FL_EV is a planning heuristic, not actual score)
        let chip_delta = record.normal_score.round() as i32;
        if chip_delta >= 0 {
            // Hero wins → transfer capped by opp's chips
            let transfer = chip_delta.min(chips[1]);
            chips[0] += transfer;
            chips[1] -= transfer;
        } else {
            // Opp wins → transfer capped by hero's chips
            let transfer = (-chip_delta).min(chips[0]);
            chips[1] += transfer;
            chips[0] -= transfer;
        }

        // ============================================================
        // Update FL state for next hand
        // ============================================================
        let hero: usize = 0;
        for seat in 0..2 {
            if fl_active[seat] {
                // Was in FL → check FL STAY (top trips OR bot quads+)
                if !game_result.busted[seat] && game_result.fl_stay[seat] {
                    fl_active[seat] = true;
                    // fl_cards[seat] stays the same (inherits from original FL entry)
                } else {
                    fl_active[seat] = false;
                    fl_cards[seat] = 0;
                }
            } else {
                // Not in FL → check FL ENTRY
                if game_result.fl_entry[seat] && !game_result.busted[seat] {
                    fl_active[seat] = true;
                    fl_cards[seat] = game_result.fl_card_count[seat];
                } else {
                    fl_active[seat] = false;
                    fl_cards[seat] = 0;
                }
            }
        }

        // Fill session info in record
        record.chips_before = chips_before;
        record.chips_after = chips;
        record.chip_delta = chips[hero] - chips_before[hero];
        record.session_id = session_id;

        // Write JSONL
        if let Some(ref mut writer) = jsonl_writer {
            let game_log = GameLog {
                game_id: i as u32,
                seed: args.seed + i as u64 + 1,
                mode: args.mode.clone(),
                session_id,
                is_fl_hand,
                fl_active: [fl_was_active_0, fl_was_active_1],
                fl_card_count: [fl_was_cards_0, fl_was_cards_1],
                chips_before,
                chips_after: chips,
                turns: turn_logs,
                result: GameResultJson {
                    normal_score: record.normal_score,
                    fl_score: record.fl_score,
                    total_score: record.total_score,
                    hero_busted: record.busted,
                    hero_fl: record.fl_entry,
                    hero_fl_stay: game_result.fl_stay[0],
                    hero_fl_type: record.fl_type.clone(),
                    hero_fl_cards: record.hero_fl_cards,
                    hero_royalty: record.royalty,
                    opp_busted: record.opp_busted,
                    opp_fl: record.opp_fl,
                    opp_fl_stay: game_result.fl_stay[1],
                    opp_fl_cards: record.opp_fl_cards,
                    opp_royalty: record.opp_royalty,
                },
            };
            serde_json::to_writer(&mut *writer, &game_log).expect("JSONL write failed");
            writer.write_all(b"\n").expect("JSONL newline failed");
        }

        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 10 == 0 {
            let elapsed = total_start.elapsed().as_secs_f64();
            let fl_tag = if fl_active[0] || fl_active[1] { " FL" } else { "" };
            eprintln!("  [{:4}/{}] {:.0}s  ({:.1}s/game)  session={} chips=[{},{}]{}",
                done, args.games, elapsed, elapsed / done as f64,
                session_id, chips[0], chips[1], fl_tag);
        }

        results.push(record);
        hand_in_session += 1;
    }
    // Count final session
    total_sessions += 1;

    // Flush JSONL
    if let Some(ref mut writer) = jsonl_writer {
        writer.flush().expect("JSONL flush failed");
        eprintln!("  JSONL written to {}", args.output.as_ref().unwrap());
    }

    let total_time = total_start.elapsed().as_secs_f64();

    // ================================================================
    // Compute statistics
    // ================================================================
    let n = results.len() as f64;
    let busts = results.iter().filter(|r| r.busted).count();
    let normal_hands = results.iter().filter(|r| !r.is_fl_hand).count();
    let fl_entries = results.iter().filter(|r| r.fl_entry && !r.is_fl_hand).count();
    let fl_stays = results.iter().filter(|r| r.fl_entry && r.is_fl_hand).count();
    let mut fl_types: HashMap<String, u32> = HashMap::new();
    for r in &results {
        if r.fl_entry && !r.is_fl_hand {
            if let Some(ref ft) = r.fl_type {
                *fl_types.entry(ft.clone()).or_insert(0) += 1;
            }
        }
    }

    let normal_scores: Vec<f64> = results.iter().map(|r| r.normal_score).collect();
    let normal_mean = normal_scores.iter().sum::<f64>() / n;

    let fl_scores: Vec<f64> = results.iter().map(|r| r.fl_score).collect();
    let fl_mean = fl_scores.iter().sum::<f64>() / n;

    let total_scores: Vec<f64> = results.iter().map(|r| r.total_score).collect();
    let total_mean = total_scores.iter().sum::<f64>() / n;
    let total_std = (total_scores.iter().map(|&s| (s - total_mean).powi(2)).sum::<f64>() / n).sqrt();

    let royalties: Vec<f64> = results.iter().map(|r| r.royalty as f64).collect();
    let royalty_mean = royalties.iter().sum::<f64>() / n;

    let chip_deltas: Vec<f64> = results.iter().map(|r| r.chip_delta as f64).collect();
    let chip_delta_mean = chip_deltas.iter().sum::<f64>() / n;

    let win_rate = results.iter().filter(|r| r.total_score > 0.0).count() as f64 / n;

    let total_se = total_std / n.sqrt();
    let bust_rate = busts as f64 / n;
    let bust_se = (bust_rate * (1.0 - bust_rate) / n).sqrt();

    println!();
    println!("============================================================");
    println!("  Results ({} games, {} sessions, {:.0}s total)", args.games, total_sessions, total_time);
    println!("============================================================");
    println!("  Bust rate:     {:.1}%", bust_rate * 100.0);
    println!("  FL entry rate: {:.1}% ({}/{} normal hands)", fl_entries as f64 / normal_hands.max(1) as f64 * 100.0, fl_entries, normal_hands);
    if fl_entries > 0 {
        for ft in &["AA", "KK", "QQ", "trips"] {
            let count = fl_types.get(*ft).copied().unwrap_or(0);
            println!("    {:5}: {:3} ({:.0}%)", ft, count, count as f64 / fl_entries as f64 * 100.0);
        }
    }
    println!("  FL stay rate:  {}/{} FL hands", fl_stays, results.iter().filter(|r| r.is_fl_hand).count());
    println!("  Avg royalty:   {:.2}", royalty_mean);
    println!("  Normal score:  {:+.2}/hand", normal_mean);
    println!("  FL bonus:      {:+.2}/hand (planning EV)", fl_mean);
    println!("  Total score:   {:+.2}/hand +/- {:.2}", total_mean, total_std);
    println!("  Chip delta:    {:+.2}/hand (actual)", chip_delta_mean);
    println!("  Win rate:      {:.1}%", win_rate * 100.0);
    println!("  Speed:         {:.2}s/hand", total_time / n);
    println!("  Avg hands/ses: {:.1}", n / total_sessions as f64);
    println!("  FL hands:      {} ({:.0}%, both={})", fl_hands_played,
        fl_hands_played as f64 / n * 100.0, both_fl_hands);
    println!();
    println!("  95% CI:");
    println!("    Score: [{:+.2}, {:+.2}]", total_mean - 1.96 * total_se, total_mean + 1.96 * total_se);
    println!("    Bust:  [{:.1}%, {:.1}%]", (bust_rate - 1.96 * bust_se) * 100.0, (bust_rate + 1.96 * bust_se) * 100.0);
    println!("============================================================");
}
