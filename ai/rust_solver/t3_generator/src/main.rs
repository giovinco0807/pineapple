use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use fl_solver::solve_fantasyland_v2_fast;
use ofc_core::{create_deck, rank_to_char, suit_to_char, Card};

mod action;
mod encoding;

use action::{get_semantic_action_index, get_turn_actions, Action};
use encoding::{encode_state, Board, Observation, ACTION_DIM, STATE_DIM};

#[derive(Parser, Debug)]
#[command(author, version, about = "Rust T3 teacher generator backed by a T4 ONNX oracle")]
struct Args {
    #[arg(long, default_value_t = 1000)]
    states: usize,

    #[arg(long, default_value = "mixed")]
    mode: String,

    #[arg(long, default_value_t = 64)]
    btn_samples: usize,

    #[arg(long, default_value = "ai/data/t4_oracle_v2/t4_oracle.onnx")]
    onnx_model: PathBuf,

    #[arg(long, default_value = "ai/data/t3_rust_teacher/t3_teacher.jsonl")]
    output: PathBuf,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 100)]
    log_interval: usize,

    #[arg(long, default_value_t = false)]
    exact_input: bool,

    #[arg(long, default_value_t = false)]
    include_jokers: bool,
}

#[derive(Serialize)]
struct T3DataRow {
    state: Vec<f32>,
    action_evs: Vec<f32>,
    action_masks: Vec<bool>,
    valid_masks: Vec<bool>,
    actions: usize,
    best_evs: f32,
    rewards: f32,
    turns: u8,
    is_btn: bool,
}

#[derive(Serialize)]
struct ExactBoardPayload {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
}

#[derive(Serialize)]
struct ExactInputRow {
    source: &'static str,
    source_line: usize,
    turn: u8,
    board: ExactBoardPayload,
    opponent_board: ExactBoardPayload,
    dealt: Vec<String>,
    known_discards: Vec<String>,
    exclude: Vec<String>,
    is_btn: bool,
    position: &'static str,
    reasons: Vec<&'static str>,
    generator_mode: String,
    include_jokers: bool,
}

struct T3Case {
    bb_board: Board,
    btn_board: Board,
    bb_discards: Vec<Card>,
    btn_discards: Vec<Card>,
    dealt_cards: Vec<Card>,
    remaining_deck: Vec<Card>,
    is_btn: bool,
}

fn peel_card(row: &mut Vec<Card>, discards: &mut Vec<Card>, rng: &mut impl Rng) -> bool {
    let valid_indices: Vec<usize> = row
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_joker())
        .map(|(i, _)| i)
        .collect();

    if let Some(&idx) = valid_indices.choose(rng) {
        discards.push(row.remove(idx));
        true
    } else {
        false
    }
}

fn generate_t3_board(cards: &[Card], rng: &mut impl Rng) -> Option<(Board, Vec<Card>, Vec<Card>)> {
    let fl_cards: Vec<fl_solver::Card> = cards
        .iter()
        .map(|c| fl_solver::Card {
            rank: c.rank,
            suit: c.suit,
        })
        .collect();
    let placement = solve_fantasyland_v2_fast(&fl_cards)?;
    if placement.discards.is_empty() {
        return None;
    }

    let mut top: Vec<Card> = placement
        .top
        .iter()
        .map(|c| Card {
            rank: c.rank,
            suit: c.suit,
        })
        .collect();
    let mut middle: Vec<Card> = placement
        .middle
        .iter()
        .map(|c| Card {
            rank: c.rank,
            suit: c.suit,
        })
        .collect();
    let mut bottom: Vec<Card> = placement
        .bottom
        .iter()
        .map(|c| Card {
            rank: c.rank,
            suit: c.suit,
        })
        .collect();
    let mut peeled: Vec<Card> = placement
        .discards
        .iter()
        .map(|c| Card {
            rank: c.rank,
            suit: c.suit,
        })
        .collect();

    while top.len() + middle.len() + bottom.len() > 9 {
        let mut rows = [0, 1, 2];
        rows.shuffle(rng);
        let mut success = false;
        for r in rows {
            success = match r {
                0 => peel_card(&mut top, &mut peeled, rng),
                1 => peel_card(&mut middle, &mut peeled, rng),
                _ => peel_card(&mut bottom, &mut peeled, rng),
            };
            if success {
                break;
            }
        }
        if !success {
            return None;
        }
    }

    peeled.shuffle(rng);
    let known_discards = peeled.iter().take(2).copied().collect();
    let hidden_cards = peeled.iter().skip(2).copied().collect();

    Some((
        Board {
            top,
            middle,
            bottom,
        },
        known_discards,
        hidden_cards,
    ))
}

fn sorted_sample(deck: &[Card], n: usize, rng: &mut impl Rng) -> Vec<Card> {
    deck.choose_multiple(rng, n).copied().collect()
}

fn remove_cards(deck: &mut Vec<Card>, cards: &[Card]) {
    let remove: HashSet<Card> = cards.iter().copied().collect();
    deck.retain(|c| !remove.contains(c));
}

fn apply_action(board: &Board, action: &Action) -> Board {
    let mut next = board.clone();
    for (card, row) in &action.placements {
        match row {
            0 => next.top.push(*card),
            1 => next.middle.push(*card),
            2 => next.bottom.push(*card),
            _ => unreachable!(),
        }
    }
    next
}

fn create_regular_turn_mask(dealt_cards: &[Card], board: &Board) -> Vec<bool> {
    let mut mask = vec![false; ACTION_DIM];
    for action in get_turn_actions(
        dealt_cards,
        board.top.len(),
        board.middle.len(),
        board.bottom.len(),
    ) {
        let idx = get_semantic_action_index(&action, dealt_cards);
        mask[idx] = true;
    }
    mask
}

fn encode_observation(
    self_board: &Board,
    opp_board: &Board,
    dealt_cards: &[Card],
    discards: &[Card],
    turn: u8,
    is_btn: bool,
) -> Vec<f32> {
    encode_state(&Observation {
        board_self: self_board.clone(),
        board_opponent: opp_board.clone(),
        dealt_cards: dealt_cards.to_vec(),
        known_discards_self: discards.to_vec(),
        turn,
        is_btn,
        is_fl: false,
        opp_is_fl: false,
        chips_self: 200,
        chips_opponent: 200,
    })
}

fn encode_t4_state(
    bb_board: &Board,
    btn_board: &Board,
    dealt_cards: &[Card],
    bb_discards: &[Card],
) -> Vec<f32> {
    encode_observation(bb_board, btn_board, dealt_cards, bb_discards, 4, false)
}

fn evaluate_t4_batch(session: &mut Session, states: &[Vec<f32>], masks: &[Vec<bool>]) -> Vec<f32> {
    if states.is_empty() {
        return vec![];
    }

    let batch_size = states.len();
    let mut flat_states = Vec::with_capacity(batch_size * STATE_DIM);
    for s in states {
        flat_states.extend_from_slice(s);
    }
    let states_array = Array2::from_shape_vec((batch_size, STATE_DIM), flat_states).unwrap();

    let mut flat_masks = Vec::with_capacity(batch_size * ACTION_DIM);
    for m in masks {
        flat_masks.extend(m.iter().copied());
    }
    let masks_array = Array2::from_shape_vec((batch_size, ACTION_DIM), flat_masks).unwrap();

    let states_tensor = Value::from_array(states_array).unwrap();
    let masks_tensor = Value::from_array(masks_array).unwrap();

    let outputs = session
        .run(ort::inputs![
            "state" => states_tensor,
            "mask" => masks_tensor,
        ])
        .unwrap();

    let values_tensor = outputs["value"].try_extract_tensor::<f32>().unwrap();
    values_tensor.1.iter().copied().collect()
}

fn sample_case(is_btn: bool, rng: &mut impl Rng) -> Option<T3Case> {
    sample_case_with_options(is_btn, false, rng)
}

fn sample_case_with_options(is_btn: bool, include_jokers: bool, rng: &mut impl Rng) -> Option<T3Case> {
    let mut deck = create_deck(false);
    if include_jokers {
        deck.push(Card { rank: 0, suit: 4 });
        deck.push(Card { rank: 0, suit: 5 });
    }
    deck.shuffle(rng);

    let bb_seed: Vec<Card> = deck.drain(0..14).collect();
    let btn_seed: Vec<Card> = deck.drain(0..14).collect();

    let (mut bb_board, mut bb_discards, bb_hidden) = generate_t3_board(&bb_seed, rng)?;
    let (btn_board, btn_discards, btn_hidden) = generate_t3_board(&btn_seed, rng)?;

    deck.extend(bb_hidden);
    deck.extend(btn_hidden);
    deck.shuffle(rng);

    if is_btn {
        let bb_dealt: Vec<Card> = deck.drain(0..3).collect();
        let bb_actions = get_turn_actions(
            &bb_dealt,
            bb_board.top.len(),
            bb_board.middle.len(),
            bb_board.bottom.len(),
        );
        if bb_actions.is_empty() {
            return None;
        }
        let bb_action = bb_actions.choose(rng)?.clone();
        bb_board = apply_action(&bb_board, &bb_action);
        if let Some(discard) = bb_action.discard {
            bb_discards.push(discard);
        }
    }

    let dealt_cards: Vec<Card> = deck.drain(0..3).collect();
    Some(T3Case {
        bb_board,
        btn_board,
        bb_discards,
        btn_discards,
        dealt_cards,
        remaining_deck: deck,
        is_btn,
    })
}

fn card_name(card: &Card) -> String {
    if card.is_joker() {
        if card.suit == 5 {
            "X2".to_string()
        } else {
            "X1".to_string()
        }
    } else {
        format!("{}{}", rank_to_char(card.rank), suit_to_char(card.suit))
    }
}

fn board_payload(board: &Board) -> ExactBoardPayload {
    ExactBoardPayload {
        top: board.top.iter().map(card_name).collect(),
        middle: board.middle.iter().map(card_name).collect(),
        bottom: board.bottom.iter().map(card_name).collect(),
    }
}

fn board_card_names(board: &Board) -> Vec<String> {
    board
        .top
        .iter()
        .chain(board.middle.iter())
        .chain(board.bottom.iter())
        .map(card_name)
        .collect()
}

fn exact_input_row(index: usize, case: &T3Case, mode: &str, include_jokers: bool) -> ExactInputRow {
    let (board, opponent_board, known_discards, position) = if case.is_btn {
        (&case.btn_board, &case.bb_board, &case.btn_discards, "btn")
    } else {
        (&case.bb_board, &case.btn_board, &case.bb_discards, "bb")
    };
    let mut exclude = board_card_names(opponent_board);
    exclude.extend(known_discards.iter().map(card_name));
    ExactInputRow {
        source: "rust_t3_generator",
        source_line: index + 1,
        turn: 3,
        board: board_payload(board),
        opponent_board: board_payload(opponent_board),
        dealt: case.dealt_cards.iter().map(card_name).collect(),
        known_discards: known_discards.iter().map(card_name).collect(),
        exclude,
        is_btn: case.is_btn,
        position,
        reasons: vec!["turn_3", "rust_generated", "exact_input"],
        generator_mode: mode.to_string(),
        include_jokers,
    }
}

fn evaluate_btn_row(
    session: &mut Session,
    case: &T3Case,
    rng: &mut impl Rng,
) -> Option<T3DataRow> {
    let board = &case.btn_board;
    let actions = get_turn_actions(
        &case.dealt_cards,
        board.top.len(),
        board.middle.len(),
        board.bottom.len(),
    );
    if actions.is_empty() {
        return None;
    }

    let state = encode_observation(
        &case.btn_board,
        &case.bb_board,
        &case.dealt_cards,
        &case.btn_discards,
        3,
        true,
    );
    let valid_mask = create_regular_turn_mask(&case.dealt_cards, board);
    let mut evs = vec![-1.0e9; ACTION_DIM];
    let mut t4_states = Vec::new();
    let mut t4_masks = Vec::new();
    let mut idxs = Vec::new();

    for action in &actions {
        let action_idx = get_semantic_action_index(action, &case.dealt_cards);
        let next_btn = apply_action(&case.btn_board, action);
        let mut deck = case.remaining_deck.clone();
        if let Some(discard) = action.discard {
            remove_cards(&mut deck, &[discard]);
        }
        let t4_dealt = sorted_sample(&deck, 3, rng);
        if t4_dealt.len() != 3 {
            continue;
        }
        let mask = create_regular_turn_mask(&t4_dealt, &case.bb_board);
        if !mask.iter().any(|&b| b) {
            continue;
        }
        t4_states.push(encode_t4_state(
            &case.bb_board,
            &next_btn,
            &t4_dealt,
            &case.bb_discards,
        ));
        t4_masks.push(mask);
        idxs.push(action_idx);
    }

    let values = evaluate_t4_batch(session, &t4_states, &t4_masks);
    for (action_idx, bb_value) in idxs.into_iter().zip(values) {
        evs[action_idx] = -bb_value;
    }

    build_row(state, evs, valid_mask, true)
}

fn evaluate_bb_row(
    session: &mut Session,
    case: &T3Case,
    btn_samples: usize,
    rng: &mut impl Rng,
) -> Option<T3DataRow> {
    let actions = get_turn_actions(
        &case.dealt_cards,
        case.bb_board.top.len(),
        case.bb_board.middle.len(),
        case.bb_board.bottom.len(),
    );
    if actions.is_empty() {
        return None;
    }

    let state = encode_observation(
        &case.bb_board,
        &case.btn_board,
        &case.dealt_cards,
        &case.bb_discards,
        3,
        false,
    );
    let valid_mask = create_regular_turn_mask(&case.dealt_cards, &case.bb_board);
    let mut evs = vec![-1.0e9; ACTION_DIM];

    for action in &actions {
        let action_idx = get_semantic_action_index(action, &case.dealt_cards);
        let next_bb = apply_action(&case.bb_board, action);
        let mut after_bb_deck = case.remaining_deck.clone();
        let mut bb_discards = case.bb_discards.clone();
        if let Some(discard) = action.discard {
            remove_cards(&mut after_bb_deck, &[discard]);
            bb_discards.push(discard);
        }

        let mut bb_values = Vec::new();
        let mut t4_states = Vec::new();
        let mut t4_masks = Vec::new();
        let mut draw_ids = Vec::new();
        for _ in 0..btn_samples.max(1) {
            let btn_dealt = sorted_sample(&after_bb_deck, 3, rng);
            if btn_dealt.len() != 3 {
                continue;
            }
            let btn_actions = get_turn_actions(
                &btn_dealt,
                case.btn_board.top.len(),
                case.btn_board.middle.len(),
                case.btn_board.bottom.len(),
            );
            if btn_actions.is_empty() {
                continue;
            }

            let draw_id = bb_values.len();
            bb_values.push(f32::NEG_INFINITY);
            for btn_action in &btn_actions {
                let next_btn = apply_action(&case.btn_board, btn_action);
                let mut t4_deck = after_bb_deck.clone();
                remove_cards(&mut t4_deck, &btn_dealt);
                if let Some(discard) = btn_action.discard {
                    remove_cards(&mut t4_deck, &[discard]);
                }
                let t4_dealt = sorted_sample(&t4_deck, 3, rng);
                if t4_dealt.len() != 3 {
                    continue;
                }
                let mask = create_regular_turn_mask(&t4_dealt, &next_bb);
                if !mask.iter().any(|&b| b) {
                    continue;
                }
                t4_states.push(encode_t4_state(
                    &next_bb,
                    &next_btn,
                    &t4_dealt,
                    &bb_discards,
                ));
                t4_masks.push(mask);
                draw_ids.push(draw_id);
            }
        }

        let values = evaluate_t4_batch(session, &t4_states, &t4_masks);
        for (draw_id, bb_value) in draw_ids.into_iter().zip(values) {
            let btn_ev = -bb_value;
            if btn_ev > bb_values[draw_id] {
                bb_values[draw_id] = btn_ev;
            }
        }

        let completed: Vec<f32> = bb_values
            .into_iter()
            .filter(|v| v.is_finite() && *v > f32::NEG_INFINITY)
            .map(|btn_best| -btn_best)
            .collect();
        if !completed.is_empty() {
            evs[action_idx] = completed.iter().sum::<f32>() / completed.len() as f32;
        }
    }

    build_row(state, evs, valid_mask, false)
}

fn build_row(
    state: Vec<f32>,
    action_evs: Vec<f32>,
    valid_mask: Vec<bool>,
    is_btn: bool,
) -> Option<T3DataRow> {
    let mut best_idx = None;
    let mut best_ev = f32::NEG_INFINITY;
    for (idx, (&ev, &valid)) in action_evs.iter().zip(&valid_mask).enumerate() {
        if valid && ev > best_ev && ev > -1.0e8 {
            best_ev = ev;
            best_idx = Some(idx);
        }
    }
    let actions = best_idx?;
    Some(T3DataRow {
        state,
        action_evs,
        action_masks: valid_mask.clone(),
        valid_masks: valid_mask,
        actions,
        best_evs: best_ev,
        rewards: best_ev,
        turns: 3,
        is_btn,
    })
}

fn next_is_btn(mode: &str, i: usize, rng: &mut impl Rng) -> bool {
    match mode {
        "btn" => true,
        "bb" => false,
        "mixed" => i % 2 == 1,
        _ => rng.gen_bool(0.5),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output dir {}", parent.display()))?;
    }

    let file = File::create(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let started = Instant::now();
    let mut written = 0usize;
    let mut attempts = 0usize;

    if args.exact_input {
        while written < args.states {
            attempts += 1;
            let is_btn = next_is_btn(&args.mode, written, &mut rng);
            let Some(case) = sample_case_with_options(is_btn, args.include_jokers, &mut rng) else {
                continue;
            };
            let row = exact_input_row(written, &case, &args.mode, args.include_jokers);
            serde_json::to_writer(&mut writer, &row).context("failed to serialize exact input row")?;
            writer.write_all(b"\n").context("failed to write newline")?;
            written += 1;

            if written % args.log_interval.max(1) == 0 {
                let elapsed = started.elapsed().as_secs_f64();
                eprintln!(
                    "written={} attempts={} speed={:.2} states/sec output={}",
                    written,
                    attempts,
                    written as f64 / elapsed.max(1e-6),
                    args.output.display()
                );
            }
        }

        writer.flush().context("failed to flush output")?;
        eprintln!(
            "done written={} attempts={} elapsed={:.1}s output={}",
            written,
            attempts,
            started.elapsed().as_secs_f64(),
            args.output.display()
        );
        return Ok(());
    }

    ort::init().commit();
    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("failed to create ONNX session builder: {e:?}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("failed to set ONNX optimization level: {e:?}"))?
        .commit_from_file(&args.onnx_model)
        .map_err(|e| anyhow::anyhow!("failed to load ONNX model {}: {e:?}", args.onnx_model.display()))?;

    while written < args.states {
        attempts += 1;
        let is_btn = next_is_btn(&args.mode, written, &mut rng);
        let Some(case) = sample_case(is_btn, &mut rng) else {
            continue;
        };
        let row = if case.is_btn {
            evaluate_btn_row(&mut session, &case, &mut rng)
        } else {
            evaluate_bb_row(&mut session, &case, args.btn_samples, &mut rng)
        };
        let Some(row) = row else {
            continue;
        };

        serde_json::to_writer(&mut writer, &row).context("failed to serialize row")?;
        writer.write_all(b"\n").context("failed to write newline")?;
        written += 1;

        if written % args.log_interval.max(1) == 0 {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "written={} attempts={} speed={:.2} states/sec output={}",
                written,
                attempts,
                written as f64 / elapsed.max(1e-6),
                args.output.display()
            );
        }
    }

    writer.flush().context("failed to flush output")?;
    eprintln!(
        "done written={} attempts={} elapsed={:.1}s output={}",
        written,
        attempts,
        started.elapsed().as_secs_f64(),
        args.output.display()
    );
    Ok(())
}
