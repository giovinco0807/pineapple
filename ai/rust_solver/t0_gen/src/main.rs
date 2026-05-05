//! T0 Training Data Generator using Nested Expectimax (Rust)
//!
//! Evaluates all valid T0 placements for a given hand.
//! Pipeline:
//! 1. Generate random T0 hand (5 cards)
//! 2. Generate all valid placements (~232)
//! 3. For each placement, sample n1 T1 deals.
//! 4. For each T1 deal, evaluate all T1 actions via Expectimax (n2, n3, n4).
//! 5. Compute the EV of the T0 placement by averaging the max T1 EV across the n1 deals.
//! 6. Output JSONL with hand and all placements ranked by EV.
//!
//! Usage:
//!   t0_gen --n-hands 1000 --n1 5 --n2 8 --n3 6 --n4 5 --seed 42 -o t0_data.jsonl

use std::io::{BufWriter, Write};
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use ofc_core::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::Serialize;
use clap::Parser;

pub type CardIdx = u8;
const JOKER1: CardIdx = 52;
const JOKER2: CardIdx = 53;
const EMPTY: CardIdx = 0xFF;
const DECK_SIZE: usize = 54;

fn cardidx_to_card(idx: CardIdx) -> Card {
    if idx >= 52 { Card { rank: 0, suit: 4 } }
    else { Card { rank: (idx / 4) + 2, suit: idx % 4 } }
}

fn cardidx_is_joker(idx: CardIdx) -> bool { idx >= 52 }

fn cardidx_to_string(idx: CardIdx) -> String {
    if idx == JOKER1 { return "X1".to_string(); }
    if idx == JOKER2 { return "X2".to_string(); }
    let ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
    let suits = ['s','h','d','c'];
    format!("{}{}", ranks[(idx/4) as usize], suits[(idx%4) as usize])
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct Board {
    top: [CardIdx; 3], mid: [CardIdx; 5], bot: [CardIdx; 5],
    top_n: u8, mid_n: u8, bot_n: u8,
}

impl Board {
    fn new() -> Self { Board { top: [EMPTY; 3], mid: [EMPTY; 5], bot: [EMPTY; 5], top_n: 0, mid_n: 0, bot_n: 0 } }
    fn place(&self, card: CardIdx, row: u8) -> Board {
        let mut b = self.clone();
        match row {
            0 => { b.top[b.top_n as usize] = card; b.top_n += 1; b.top[..b.top_n as usize].sort(); }
            1 => { b.mid[b.mid_n as usize] = card; b.mid_n += 1; b.mid[..b.mid_n as usize].sort(); }
            2 => { b.bot[b.bot_n as usize] = card; b.bot_n += 1; b.bot[..b.bot_n as usize].sort(); }
            _ => unreachable!(),
        }
        b
    }
    fn is_complete(&self) -> bool { self.top_n == 3 && self.mid_n == 5 && self.bot_n == 5 }
    fn top_cards(&self) -> Vec<Card> { self.top[..self.top_n as usize].iter().map(|&i| cardidx_to_card(i)).collect() }
    fn mid_cards(&self) -> Vec<Card> { self.mid[..self.mid_n as usize].iter().map(|&i| cardidx_to_card(i)).collect() }
    fn bot_cards(&self) -> Vec<Card> { self.bot[..self.bot_n as usize].iter().map(|&i| cardidx_to_card(i)).collect() }
}

const ROW_TOP: u8 = 0; const ROW_MID: u8 = 1; const ROW_BOT: u8 = 2;

type T0Action = [(CardIdx, u8); 5];

fn gen_t0_actions(cards: &[CardIdx; 5]) -> Vec<T0Action> {
    let rows = [ROW_TOP, ROW_MID, ROW_BOT];
    let mut actions = Vec::new();
    let mut seen = Vec::new();
    let board = Board::new();
    for &r0 in &rows { for &r1 in &rows { for &r2 in &rows { for &r3 in &rows { for &r4 in &rows {
        let a = [r0,r1,r2,r3,r4];
        let tc = a.iter().filter(|&&r| r==0).count() as u8;
        let mc = a.iter().filter(|&&r| r==1).count() as u8;
        let bc = a.iter().filter(|&&r| r==2).count() as u8;
        if tc > 3 || mc > 5 || bc > 5 { continue; }
        let mut b = board.clone();
        for i in 0..5 { b = b.place(cards[i], a[i]); }
        if !seen.contains(&b) {
            seen.push(b);
            actions.push([(cards[0],r0),(cards[1],r1),(cards[2],r2),(cards[3],r3),(cards[4],r4)]);
        }
    }}}}}
    actions
}

struct TurnAction { discard: CardIdx, placements: [(CardIdx, u8); 2] }

fn gen_turn_actions(cards: &[CardIdx; 3], board: &Board) -> Vec<TurnAction> {
    let top_cap = 3 - board.top_n; let mid_cap = 5 - board.mid_n; let bot_cap = 5 - board.bot_n;
    let rows = [ROW_TOP, ROW_MID, ROW_BOT];
    let mut actions = Vec::new(); let mut seen = Vec::new();
    for disc in 0..3u8 {
        if cardidx_is_joker(cards[disc as usize]) { continue; }
        let rem: Vec<CardIdx> = (0..3u8).filter(|&i| i != disc).map(|i| cards[i as usize]).collect();
        for &r0 in &rows { for &r1 in &rows {
            let mut cap = [top_cap, mid_cap, bot_cap];
            if cap[r0 as usize] == 0 { continue; }
            cap[r0 as usize] -= 1;
            if cap[r1 as usize] == 0 { continue; }
            let mut b = board.clone();
            b = b.place(rem[0], r0); b = b.place(rem[1], r1);
            if !seen.contains(&b) {
                seen.push(b);
                actions.push(TurnAction { discard: cards[disc as usize], placements: [(rem[0],r0),(rem[1],r1)] });
            }
        }}
    }
    actions
}

const BUST_PENALTY: f64 = -6.0;

fn evaluate_terminal(board: &Board) -> f64 {
    let top = board.top_cards(); let mid = board.mid_cards(); let bot = board.bot_cards();
    if !is_valid_placement(&top, &mid, &bot) { return BUST_PENALTY; }
    let tr = get_top_royalty(&top); let mr = get_middle_royalty(&mid); let br = get_bottom_royalty(&bot);
    let total = (tr + mr + br) as f64;
    let (fl_q, fl_c) = check_fl_entry(&top);
    let fl_bonus = if fl_q { match fl_c { 14 => 16.8, 15 => 27.9, 16 => 52.4, 17 => 104.5, _ => 0.0 } } else { 0.0 };
    total + fl_bonus
}

fn sample_3(remaining: &[CardIdx], rng: &mut StdRng) -> [CardIdx; 3] {
    let n = remaining.len(); let mut buf: Vec<usize> = (0..n).collect();
    for i in 0..3 { let j = rng.gen_range(i..n); buf.swap(i, j); }
    [remaining[buf[0]], remaining[buf[1]], remaining[buf[2]]]
}

fn remove_dealt(remaining: &[CardIdx], dealt: &[CardIdx; 3]) -> Vec<CardIdx> {
    let mut result = Vec::with_capacity(remaining.len() - 3);
    let mut used = [false; 3];
    for &c in remaining {
        let mut found = false;
        for i in 0..3 { if !used[i] && c == dealt[i] { used[i] = true; found = true; break; } }
        if !found { result.push(c); }
    }
    result
}

#[derive(Clone)]
struct NestParams { n1: usize, n2: usize, n3: usize, n4: usize }

fn expectimax_t4(board: &Board, remaining: &[CardIdx], params: &NestParams, rng: &mut StdRng) -> f64 {
    if board.is_complete() { return evaluate_terminal(board); }
    let n = remaining.len(); if n < 3 { return evaluate_terminal(board); }
    if params.n4 == 0 {
        let mut total = 0.0; let mut count = 0u64;
        for i in 0..n { for j in (i+1)..n { for k in (j+1)..n {
            let deal = [remaining[i], remaining[j], remaining[k]];
            total += choice_node_terminal(&deal, board); count += 1;
        }}}
        if count == 0 { BUST_PENALTY } else { total / count as f64 }
    } else {
        let mut total = 0.0;
        for _ in 0..params.n4 {
            let deal = sample_3(remaining, rng);
            total += choice_node_terminal(&deal, board);
        }
        total / params.n4 as f64
    }
}

fn choice_node_terminal(deal: &[CardIdx; 3], board: &Board) -> f64 {
    let actions = gen_turn_actions(deal, board);
    if actions.is_empty() { return evaluate_terminal(board); }
    let mut best = f64::NEG_INFINITY;
    for a in &actions {
        let mut b2 = board.clone();
        for &(card, row) in &a.placements { b2 = b2.place(card, row); }
        let v = if b2.is_complete() { evaluate_terminal(&b2) } else { BUST_PENALTY };
        if v > best { best = v; }
    }
    best
}

fn expectimax_t3(board: &Board, remaining: &[CardIdx], params: &NestParams, rng: &mut StdRng) -> f64 {
    if board.is_complete() { return evaluate_terminal(board); }
    let mut total = 0.0;
    for _ in 0..params.n3 {
        let deal = sample_3(remaining, rng); let rest = remove_dealt(remaining, &deal);
        let actions = gen_turn_actions(&deal, board);
        if actions.is_empty() { total += evaluate_terminal(board); continue; }
        let mut best = f64::NEG_INFINITY;
        for a in &actions {
            let mut b2 = board.clone();
            for &(card, row) in &a.placements { b2 = b2.place(card, row); }
            let v = expectimax_t4(&b2, &rest, params, rng);
            if v > best { best = v; }
        }
        total += best;
    }
    total / params.n3 as f64
}

fn expectimax_t2(board: &Board, remaining: &[CardIdx], params: &NestParams, rng: &mut StdRng) -> f64 {
    if board.is_complete() { return evaluate_terminal(board); }
    let mut total = 0.0;
    for _ in 0..params.n2 {
        let deal = sample_3(remaining, rng); let rest = remove_dealt(remaining, &deal);
        let actions = gen_turn_actions(&deal, board);
        if actions.is_empty() { total += evaluate_terminal(board); continue; }
        let mut best = f64::NEG_INFINITY;
        for a in &actions {
            let mut b2 = board.clone();
            for &(card, row) in &a.placements { b2 = b2.place(card, row); }
            let v = expectimax_t3(&b2, &rest, params, rng);
            if v > best { best = v; }
        }
        total += best;
    }
    total / params.n2 as f64
}

#[derive(Serialize)]
struct T0Record {
    hand_idx: usize,
    t0_hand: Vec<String>,
    n_placements: usize,
    nesting: String,
    placements: Vec<T0PlacementResult>,
}

#[derive(Serialize)]
struct T0PlacementResult {
    p: String,
    placement: [u8; 5],
    ev: f64,
}

fn format_t0_action(action: &T0Action) -> String {
    let row_names = ["Top", "Middle", "Bottom"];
    action.iter().map(|&(c,r)| format!("{}→{}", cardidx_to_string(c), row_names[r as usize])).collect::<Vec<_>>().join(", ")
}

fn evaluate_t0_hand(
    hand_idx: usize,
    deck: &[CardIdx; DECK_SIZE],
    params: &NestParams,
    seed: u64,
) -> T0Record {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffled = deck.clone();
    for i in 0..5 { let j = rng.gen_range(i..DECK_SIZE); shuffled.swap(i, j); }
    let mut t0_hand = [EMPTY; 5]; t0_hand.copy_from_slice(&shuffled[0..5]);
    let rest_deck = &shuffled[5..];
    
    let actions = gen_t0_actions(&t0_hand);
    
    // Pre-sample n1 T1 deals for variance reduction
    let mut t1_deals = Vec::new();
    for _ in 0..params.n1 { t1_deals.push(sample_3(rest_deck, &mut rng)); }
    
    // Evaluate all placements in parallel
    let mut results: Vec<T0PlacementResult> = actions.par_iter().enumerate().map(|(pi, action)| {
        let mut board = Board::new();
        for &(card, row) in action { board = board.place(card, row); }
        
        let mut total_ev = 0.0;
        let mut eval_rng = StdRng::seed_from_u64(seed.wrapping_add((pi * 9999 + 13) as u64));
        
        for deal in &t1_deals {
            let rest = remove_dealt(rest_deck, deal);
            let t1_actions = gen_turn_actions(deal, &board);
            if t1_actions.is_empty() { total_ev += evaluate_terminal(&board); continue; }
            let mut best = f64::NEG_INFINITY;
            for a in &t1_actions {
                let mut b2 = board.clone();
                for &(card, row) in &a.placements { b2 = b2.place(card, row); }
                let v = expectimax_t2(&b2, &rest, params, &mut eval_rng);
                if v > best { best = v; }
            }
            total_ev += best;
        }
        let ev = total_ev / params.n1 as f64;
        
        let p = format_t0_action(action);
        let placement_arr = [action[0].1, action[1].1, action[2].1, action[3].1, action[4].1];
        T0PlacementResult { p, placement: placement_arr, ev: (ev * 1000.0).round() / 1000.0 }
    }).collect();
    
    results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());
    
    let t0_hand_strs: Vec<String> = t0_hand.iter().map(|&c| cardidx_to_string(c)).collect();
    let nesting_str = if params.n4 == 0 { format!("n1={},n2={},n3={},n4=all", params.n1, params.n2, params.n3) }
                      else { format!("n1={},n2={},n3={},n4={}", params.n1, params.n2, params.n3, params.n4) };
    
    T0Record { hand_idx, t0_hand: t0_hand_strs, n_placements: results.len(), nesting: nesting_str, placements: results }
}

#[derive(Parser)]
#[command(name = "t0_gen", about = "T0 Expectimax training data generator")]
struct Cli {
    #[arg(short, long, default_value_t = 1000)]
    n_hands: usize,
    #[arg(long, default_value_t = 5)]
    n1: usize,
    #[arg(long, default_value_t = 8)]
    n2: usize,
    #[arg(long, default_value_t = 6)]
    n3: usize,
    #[arg(long, default_value_t = 5)]
    n4: usize,
    #[arg(short, long, default_value = "t0_data.jsonl")]
    output: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();
    let params = NestParams { n1: cli.n1, n2: cli.n2, n3: cli.n3, n4: cli.n4 };
    let deck: Vec<CardIdx> = (0..DECK_SIZE as CardIdx).collect();
    let deck_arr: [CardIdx; DECK_SIZE] = deck.try_into().unwrap();

    let t4_desc = if cli.n4 == 0 { "all".to_string() } else { cli.n4.to_string() };

    eprintln!("=== T0 Expectimax Generator (Rust) ===");
    eprintln!("  T0 limits:   {} random hands", cli.n_hands);
    eprintln!("  Nesting:     n1={}, n2={}, n3={}, n4={}", cli.n1, cli.n2, cli.n3, t4_desc);
    eprintln!("  Output:      {}", cli.output);
    eprintln!("  Threads:     {} (rayon)", rayon::current_num_threads());

    let start = Instant::now();
    let counter = Arc::new(AtomicUsize::new(0));

    let all_records: Vec<T0Record> = (0..cli.n_hands)
        .into_par_iter()
        .map(|i| {
            let rec = evaluate_t0_hand(
                i, &deck_arr, &params,
                cli.seed.wrapping_add(i as u64 * 100003),
            );
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == cli.n_hands {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let eta = (cli.n_hands - done) as f64 / rate;
                eprint!("\r  [{:>6}/{}] rate={:.2}/s ETA={:.0}s ({:.1}h)   ",
                    done, cli.n_hands, rate, eta, eta / 3600.0);
            }
            rec
        })
        .collect();

    eprintln!();

    let file = File::create(&cli.output).expect("Cannot create output file");
    let mut writer = BufWriter::new(file);
    for rec in &all_records {
        serde_json::to_writer(&mut writer, rec).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    writer.flush().unwrap();

    let elapsed = start.elapsed();
    eprintln!("=== Done === {:.0}s ({:.1}h) | {} hands | {}",
        elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0,
        all_records.len(), cli.output);
}
