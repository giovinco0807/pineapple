//! T1 Training Data Generator using Nested Expectimax (Rust)
//!
//! Pipeline:
//! 1. Deal 5 random cards (T0), pick a random placement
//! 2. Sample n1 T1 deals, enumerate ALL T1 actions, compute EV via:
//!    - T2: n2 sampled deals × all actions → max
//!    - T3: n3 sampled deals × all actions → max
//!    - T4: n4 sampled deals × all actions → max → terminal
//!      (n4=0 means exhaustive: all C(remaining,3) deals)
//! 3. Output JSONL: one line per (T0-placement, T1-deal) with all T1 action EVs
//!
//! Usage:
//!   t1_gen --n-hands 50000 --n1 5 --n2 3 --n3 3 --n4 30 --seed 42 -o t1_data.jsonl

use std::collections::HashSet;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use ofc_core::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use clap::Parser;

// ============================================================
//  Card representation (same as backward solver)
// ============================================================

pub type CardIdx = u8;
const JOKER1: CardIdx = 52;
const JOKER2: CardIdx = 53;
const EMPTY: CardIdx = 0xFF;
const DECK_SIZE: usize = 54;

fn cardidx_to_card(idx: CardIdx) -> Card {
    if idx >= 52 {
        Card { rank: 0, suit: 4 }
    } else {
        Card { rank: (idx / 4) + 2, suit: idx % 4 }
    }
}

fn cardidx_is_joker(idx: CardIdx) -> bool {
    idx >= 52
}

fn cardidx_to_string(idx: CardIdx) -> String {
    if idx == JOKER1 { return "X1".to_string(); }
    if idx == JOKER2 { return "X2".to_string(); }
    let ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
    let suits = ['s','h','d','c'];
    format!("{}{}", ranks[(idx/4) as usize], suits[(idx%4) as usize])
}

// ============================================================
//  Board (compact, hashable)
// ============================================================

#[derive(Clone, Eq, PartialEq, Hash)]
struct Board {
    top: [CardIdx; 3],
    mid: [CardIdx; 5],
    bot: [CardIdx; 5],
    top_n: u8,
    mid_n: u8,
    bot_n: u8,
}

impl Board {
    fn new() -> Self {
        Board { top: [EMPTY; 3], mid: [EMPTY; 5], bot: [EMPTY; 5], top_n: 0, mid_n: 0, bot_n: 0 }
    }
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
    fn fmt_row(cards: &[CardIdx], n: u8) -> String {
        cards[..n as usize].iter().map(|&c| cardidx_to_string(c)).collect::<Vec<_>>().join(" ")
    }
}

const ROW_TOP: u8 = 0;
const ROW_MID: u8 = 1;
const ROW_BOT: u8 = 2;

// ============================================================
//  T0 Action generation
// ============================================================

type T0Action = [(CardIdx, u8); 5];

fn gen_t0_actions(cards: &[CardIdx; 5]) -> Vec<T0Action> {
    let rows = [ROW_TOP, ROW_MID, ROW_BOT];
    let mut actions = Vec::new();
    let mut seen = HashSet::new();
    let board = Board::new();
    for &r0 in &rows { for &r1 in &rows { for &r2 in &rows { for &r3 in &rows { for &r4 in &rows {
        let a = [r0,r1,r2,r3,r4];
        let tc = a.iter().filter(|&&r| r==0).count() as u8;
        let mc = a.iter().filter(|&&r| r==1).count() as u8;
        let bc = a.iter().filter(|&&r| r==2).count() as u8;
        if tc > 3 || mc > 5 || bc > 5 { continue; }
        let mut b = board.clone();
        for i in 0..5 { b = b.place(cards[i], a[i]); }
        if seen.insert(b) {
            actions.push([(cards[0],r0),(cards[1],r1),(cards[2],r2),(cards[3],r3),(cards[4],r4)]);
        }
    }}}}}
    actions
}

// ============================================================
//  Turn Action generation (T1-T4)
// ============================================================

struct TurnAction {
    discard: CardIdx,
    placements: [(CardIdx, u8); 2],
}

fn gen_turn_actions(cards: &[CardIdx; 3], board: &Board) -> Vec<TurnAction> {
    let top_cap = 3 - board.top_n;
    let mid_cap = 5 - board.mid_n;
    let bot_cap = 5 - board.bot_n;
    let rows = [ROW_TOP, ROW_MID, ROW_BOT];
    let mut actions = Vec::new();
    let mut seen = HashSet::new();
    for disc in 0..3u8 {
        if cardidx_is_joker(cards[disc as usize]) { continue; }
        let rem: Vec<CardIdx> = (0..3u8).filter(|&i| i != disc).map(|i| cards[i as usize]).collect();
        for &r0 in &rows { for &r1 in &rows {
            let mut cap = [top_cap, mid_cap, bot_cap];
            if cap[r0 as usize] == 0 { continue; }
            cap[r0 as usize] -= 1;
            if cap[r1 as usize] == 0 { continue; }
            let mut b = board.clone();
            b = b.place(rem[0], r0);
            b = b.place(rem[1], r1);
            if seen.insert(b) {
                actions.push(TurnAction { discard: cards[disc as usize], placements: [(rem[0],r0),(rem[1],r1)] });
            }
        }}
    }
    actions
}

// ============================================================
//  Terminal Evaluation (using ofc_core)
// ============================================================

const BUST_PENALTY: f64 = -6.0;

fn evaluate_terminal(board: &Board) -> f64 {
    let top = board.top_cards();
    let mid = board.mid_cards();
    let bot = board.bot_cards();
    if !is_valid_placement(&top, &mid, &bot) { return BUST_PENALTY; }
    let tr = get_top_royalty(&top);
    let mr = get_middle_royalty(&mid);
    let br = get_bottom_royalty(&bot);
    let total = (tr + mr + br) as f64;
    let (fl_q, fl_c) = check_fl_entry(&top);
    let fl_bonus = if fl_q {
        match fl_c { 14 => 16.8, 15 => 27.9, 16 => 52.4, 17 => 104.5, _ => 0.0 }
    } else { 0.0 };
    total + fl_bonus
}

// ============================================================
//  Deck utilities
// ============================================================

fn sample_3(remaining: &[CardIdx], rng: &mut StdRng) -> [CardIdx; 3] {
    let n = remaining.len();
    let mut buf: Vec<usize> = (0..n).collect();
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

// ============================================================
//  Nested Expectimax: T4 (sampled or exhaustive), T3, T2
// ============================================================

/// Nesting parameters
#[derive(Clone)]
struct NestParams {
    n2: usize,
    n3: usize,
    n4: usize,  // 0 = exhaustive
}

/// T4 chance node: if n4==0 exhaustive, else sample n4 deals
fn expectimax_t4(board: &Board, remaining: &[CardIdx], params: &NestParams, rng: &mut StdRng) -> f64 {
    if board.is_complete() { return evaluate_terminal(board); }
    let n = remaining.len();
    if n < 3 { return evaluate_terminal(board); }

    if params.n4 == 0 {
        // Exhaustive: enumerate all C(n,3) deals
        let mut total = 0.0;
        let mut count = 0u64;
        for i in 0..n {
            for j in (i+1)..n {
                for k in (j+1)..n {
                    let deal = [remaining[i], remaining[j], remaining[k]];
                    let v = choice_node_terminal(&deal, board);
                    total += v;
                    count += 1;
                }
            }
        }
        if count == 0 { BUST_PENALTY } else { total / count as f64 }
    } else {
        // Sampled
        let mut total = 0.0;
        for _ in 0..params.n4 {
            let deal = sample_3(remaining, rng);
            let v = choice_node_terminal(&deal, board);
            total += v;
        }
        total / params.n4 as f64
    }
}

/// Choice node at T4: just pick best action → terminal eval
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

/// T3 chance node: sample n3 deals, all actions → max, then T4
fn expectimax_t3(board: &Board, remaining: &[CardIdx], params: &NestParams, rng: &mut StdRng) -> f64 {
    if board.is_complete() { return evaluate_terminal(board); }
    let mut total = 0.0;
    for _ in 0..params.n3 {
        let deal = sample_3(remaining, rng);
        let rest = remove_dealt(remaining, &deal);
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

/// T2 chance node: sample n2 deals, all actions → max, then T3
fn expectimax_t2(board: &Board, remaining: &[CardIdx], params: &NestParams, rng: &mut StdRng) -> f64 {
    if board.is_complete() { return evaluate_terminal(board); }
    let mut total = 0.0;
    for _ in 0..params.n2 {
        let deal = sample_3(remaining, rng);
        let rest = remove_dealt(remaining, &deal);
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

// ============================================================
//  Output format
// ============================================================

#[derive(Serialize)]
struct T1Record {
    hand_idx: usize,
    turn: u8,
    t0_hand: Vec<String>,
    t0_action: String,
    board: String,
    hand: String,
    n_placements: usize,
    nesting: String,
    placements: Vec<PlacementResult>,
}

#[derive(Serialize)]
struct PlacementResult {
    d: String,
    p: String,
    ev: f64,
}

fn format_turn_action_str(a: &TurnAction) -> (String, String) {
    let row_names = ["Top", "Middle", "Bottom"];
    let d = cardidx_to_string(a.discard);
    let p = format!("{}→{}, {}→{}",
        cardidx_to_string(a.placements[0].0), row_names[a.placements[0].1 as usize],
        cardidx_to_string(a.placements[1].0), row_names[a.placements[1].1 as usize]);
    (d, p)
}

fn format_t0_action(action: &T0Action) -> String {
    let row_names = ["Top", "Middle", "Bottom"];
    action.iter().map(|&(c,r)| format!("{}→{}", cardidx_to_string(c), row_names[r as usize])).collect::<Vec<_>>().join(", ")
}

// ============================================================
//  Per-hand worker: parallel over T1 actions within each deal
// ============================================================

#[derive(Deserialize)]
struct PreGenT0 {
    t0_hand: [CardIdx; 5],
    top50: Vec<[u8; 5]>,
}

fn generate_one_pregen(
    hand_idx: usize,
    pregen: &PreGenT0,
    n1: usize,
    params: &NestParams,
    seed: u64,
) -> Vec<T1Record> {
    let mut deck: Vec<CardIdx> = (0..DECK_SIZE as CardIdx).collect();
    deck.retain(|c| !pregen.t0_hand.contains(c));
    let t0_hand = pregen.t0_hand;
    let t0_hand_strs: Vec<String> = t0_hand.iter().map(|&c| cardidx_to_string(c)).collect();

    // We will generate records for all 50 placements in parallel
    pregen.top50.par_iter().enumerate().flat_map(|(pi, t0_action_raw)| {
        let mut t0_action = [(0, 0); 5];
        for i in 0..5 {
            t0_action[i] = (t0_hand[i], t0_action_raw[i]);
        }
        
        let mut board = Board::new();
        for &(card, row) in &t0_action { board = board.place(card, row); }
        let t0_action_str = format_t0_action(&t0_action);

        let nesting_str = if params.n4 == 0 {
            format!("n2={},n3={},n4=all", params.n2, params.n3)
        } else {
            format!("n2={},n3={},n4={}", params.n2, params.n3, params.n4)
        };

        let mut local_records = Vec::new();
        // T1: sample n1 deals
        for s in 0..n1 {
            let mut deal_rng = StdRng::seed_from_u64(seed.wrapping_add((pi * 9999 + s * 13) as u64));
            let deal = sample_3(&deck, &mut deal_rng);
            let rest = remove_dealt(&deck, &deal);

            let actions = gen_turn_actions(&deal, &board);
            if actions.is_empty() { continue; }

            // Evaluate ALL T1 actions
            let mut results: Vec<PlacementResult> = Vec::new();
            for (ai, a) in actions.iter().enumerate() {
                let mut b2 = board.clone();
                for &(card, row) in &a.placements { b2 = b2.place(card, row); }
                let mut eval_rng = StdRng::seed_from_u64(seed.wrapping_add((pi * 177 + s * 9999 + ai * 777 + 13) as u64));
                let ev = expectimax_t2(&b2, &rest, params, &mut eval_rng);
                let (d, p) = format_turn_action_str(a);
                results.push(PlacementResult { d, p, ev: (ev * 1000.0).round() / 1000.0 });
            }
            results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

            let board_str = format!("Top[{}] Mid[{}] Bot[{}]",
                Board::fmt_row(&board.top, board.top_n),
                Board::fmt_row(&board.mid, board.mid_n),
                Board::fmt_row(&board.bot, board.bot_n));
            let hand_str = deal.iter().map(|&c| cardidx_to_string(c)).collect::<Vec<_>>().join(" ");

            local_records.push(T1Record {
                hand_idx: hand_idx * n1 + s,
                turn: 1,
                t0_hand: t0_hand_strs.clone(),
                t0_action: t0_action_str.clone(),
                board: board_str,
                hand: hand_str,
                n_placements: results.len(),
                nesting: nesting_str.clone(),
                placements: results,
            });
        }
        local_records
    }).collect()
}

// ============================================================
//  CLI & main
// ============================================================

#[derive(Parser)]
#[command(name = "t1_gen", about = "T1 Expectimax training data generator")]
struct Cli {
    /// Input JSONL file of pre-generated T0 states
    #[arg(short, long)]
    input: String,
    /// Number of T0 hands to process (default all)
    #[arg(long)]
    n_hands: Option<usize>,
    /// T1 deal samples per T0 placement
    #[arg(long, default_value_t = 5)]
    n1: usize,
    /// T2 deal samples
    #[arg(long, default_value_t = 3)]
    n2: usize,
    /// T3 deal samples
    #[arg(long, default_value_t = 3)]
    n3: usize,
    /// T4 deal samples (0 = exhaustive all C(remaining,3))
    #[arg(long, default_value_t = 30)]
    n4: usize,
    /// Output file
    #[arg(short, long, default_value = "t1_data.jsonl")]
    output: String,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();
    let params = NestParams { n2: cli.n2, n3: cli.n3, n4: cli.n4 };
    let file = File::open(&cli.input).expect("Cannot open input file");
    let reader = BufReader::new(file);
    let mut pregen_data: Vec<PreGenT0> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let pregen: PreGenT0 = serde_json::from_str(&line).expect("Invalid JSONL in input");
        pregen_data.push(pregen);
    }

    let n_hands = cli.n_hands.unwrap_or(pregen_data.len());
    let pregen_data = &pregen_data[..n_hands];

    let total_records = n_hands * 50 * cli.n1;

    let t4_desc = if cli.n4 == 0 { "all".to_string() } else { cli.n4.to_string() };

    eprintln!("=== T1 Expectimax Generator (Rust) ===");
    eprintln!("  Input:       {} ({} items)", cli.input, pregen_data.len());
    eprintln!("  T0 limits:   {} hands, Top 50 placements each", n_hands);
    eprintln!("  T1 samples:  {:>8} (per T0 placement)", cli.n1);
    eprintln!("  Nesting:     T2={}, T3={}, T4={}", cli.n2, cli.n3, t4_desc);
    eprintln!("  Total T1 expected records: ~{}", total_records);
    eprintln!("  Output:      {}", cli.output);
    eprintln!("  Threads:     {} (rayon)", rayon::current_num_threads());

    let start = Instant::now();
    let counter = Arc::new(AtomicUsize::new(0));

    // Generate all hands in parallel using rayon
    let all_records: Vec<T1Record> = pregen_data
        .into_par_iter()
        .enumerate()
        .flat_map(|(i, pregen)| {
            let recs = generate_one_pregen(
                i, pregen, cli.n1, &params,
                cli.seed.wrapping_add(i as u64 * 100003),
            );
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == n_hands {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let eta = (n_hands - done) as f64 / rate;
                eprint!("\r  [{:>6}/{}] rate={:.1}/s ETA={:.0}s ({:.1}h)   ",
                    done, n_hands, rate, eta, eta / 3600.0);
            }
            recs
        })
        .collect();

    eprintln!();

    // Write output
    let file = File::create(&cli.output).expect("Cannot create output file");
    let mut writer = BufWriter::new(file);
    for rec in &all_records {
        serde_json::to_writer(&mut writer, rec).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    writer.flush().unwrap();

    let elapsed = start.elapsed();
    eprintln!("=== Done === {:.0}s ({:.1}h) | {} records | {}",
        elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0,
        all_records.len(), cli.output);
}
