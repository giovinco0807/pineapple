//! Expectimax Deep Search Engine for OFC Pineapple
//!
//! Computes optimal placements for T0→T4 using Expectimax
//! (expected value maximization over unknown future cards).
//!
//! Usage:
//!   backward expectimax --t0 "Ah,Kh,Qd,7c,3s" --n1 10 --n2 3 --n3 2 --n4 2
//!   backward enumerate  # list all canonical T0 patterns
//!   backward stdin       # JSON protocol on stdin/stdout

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use ofc_core::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================
//  CardIdx: compact card representation (0-53)
// ============================================================

pub type CardIdx = u8;

const JOKER1: CardIdx = 52;
const JOKER2: CardIdx = 53;
const EMPTY: CardIdx = 0xFF;

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

// ============================================================
//  Board: compact board state
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
        Board {
            top: [EMPTY; 3],
            mid: [EMPTY; 5],
            bot: [EMPTY; 5],
            top_n: 0,
            mid_n: 0,
            bot_n: 0,
        }
    }

    fn place(&self, card: CardIdx, row: u8) -> Board {
        let mut b = self.clone();
        match row {
            0 => {
                b.top[b.top_n as usize] = card;
                b.top_n += 1;
                b.top[..b.top_n as usize].sort();
            }
            1 => {
                b.mid[b.mid_n as usize] = card;
                b.mid_n += 1;
                b.mid[..b.mid_n as usize].sort();
            }
            2 => {
                b.bot[b.bot_n as usize] = card;
                b.bot_n += 1;
                b.bot[..b.bot_n as usize].sort();
            }
            _ => unreachable!(),
        }
        b
    }

    fn is_complete(&self) -> bool {
        self.top_n == 3 && self.mid_n == 5 && self.bot_n == 5
    }

    fn total_cards(&self) -> u8 {
        self.top_n + self.mid_n + self.bot_n
    }

    fn top_cards(&self) -> Vec<Card> {
        self.top[..self.top_n as usize].iter().map(|&i| cardidx_to_card(i)).collect()
    }
    fn mid_cards(&self) -> Vec<Card> {
        self.mid[..self.mid_n as usize].iter().map(|&i| cardidx_to_card(i)).collect()
    }
    fn bot_cards(&self) -> Vec<Card> {
        self.bot[..self.bot_n as usize].iter().map(|&i| cardidx_to_card(i)).collect()
    }
}

// ============================================================
//  Action generation
// ============================================================

const ROW_TOP: u8 = 0;
const ROW_MID: u8 = 1;
const ROW_BOT: u8 = 2;

type T0Action = [(CardIdx, u8); 5];

struct TurnAction {
    discard: CardIdx,
    placements: [(CardIdx, u8); 2],
}

fn gen_t0_actions(cards: &[CardIdx; 5], board: &Board) -> Vec<T0Action> {
    let top_cap = 3 - board.top_n;
    let mid_cap = 5 - board.mid_n;
    let bot_cap = 5 - board.bot_n;

    let mut actions: Vec<T0Action> = Vec::new();
    let mut seen: std::collections::HashSet<Board> = std::collections::HashSet::new();

    let rows = [ROW_TOP, ROW_MID, ROW_BOT];
    for &r0 in &rows {
        for &r1 in &rows {
            for &r2 in &rows {
                for &r3 in &rows {
                    for &r4 in &rows {
                        let assignment = [r0, r1, r2, r3, r4];
                        let top_count = assignment.iter().filter(|&&r| r == ROW_TOP).count() as u8;
                        let mid_count = assignment.iter().filter(|&&r| r == ROW_MID).count() as u8;
                        let bot_count = assignment.iter().filter(|&&r| r == ROW_BOT).count() as u8;

                        if top_count > top_cap || mid_count > mid_cap || bot_count > bot_cap {
                            continue;
                        }

                        let mut b = board.clone();
                        for i in 0..5 {
                            b = b.place(cards[i], assignment[i]);
                        }

                        if seen.insert(b) {
                            let action = [
                                (cards[0], r0), (cards[1], r1), (cards[2], r2),
                                (cards[3], r3), (cards[4], r4),
                            ];
                            actions.push(action);
                        }
                    }
                }
            }
        }
    }
    actions
}

/// Domain-pruned T0 action generation.
/// Rules:
///   1. Q/J/T pair or trips → all must go to Bot
///   2. 2-6 trips → all must go to Bot or Top (not Mid)
///   3. A/K pair → must go to Bot or Top (not Mid separately)
///   4. Jokers are never discarded (enforced elsewhere too)
fn gen_t0_actions_pruned(cards: &[CardIdx; 5], board: &Board) -> Vec<T0Action> {
    let all = gen_t0_actions(cards, board);

    // Analyze the hand
    let ranks: Vec<u8> = cards.iter().map(|&c| if c >= 52 { 99 } else { c / 4 }).collect();
    let mut rank_count: [u8; 13] = [0; 13];
    for &r in &ranks {
        if r < 13 { rank_count[r as usize] += 1; }
    }

    // Find pairs and trips
    let mut pair_ranks: Vec<u8> = Vec::new();
    let mut trips_ranks: Vec<u8> = Vec::new();
    for r in 0..13u8 {
        if rank_count[r as usize] >= 3 { trips_ranks.push(r); }
        else if rank_count[r as usize] >= 2 { pair_ranks.push(r); }
    }

    // Determine constraints per card index
    // None = no constraint, Some(rows) = card must go to one of these rows
    let mut constraints: Vec<Option<Vec<u8>>> = vec![None; 5];

    let has_joker = cards.iter().any(|&c| c >= 52);

    // Rule 5: 4 same-suit cards (no A/K in them, no pair) → 4 suited to Bot, remainder to Mid if not A/K
    if !has_joker && pair_ranks.is_empty() && trips_ranks.is_empty() {
        let suits: Vec<u8> = cards.iter().map(|&c| if c >= 52 { 99 } else { c % 4 }).collect();
        let mut suit_count = [0u8; 4];
        for &s in &suits { if s < 4 { suit_count[s as usize] += 1; } }
        if let Some(flush_suit) = suit_count.iter().position(|&c| c == 4) {
            let flush_suit = flush_suit as u8;
            // Check no A/K among the 4 suited cards
            let suited_ranks: Vec<u8> = (0..5).filter(|&i| suits[i] == flush_suit)
                .map(|i| ranks[i]).collect();
            if !suited_ranks.iter().any(|&r| r == 12 || r == 11) {
                for i in 0..5 {
                    if suits[i] == flush_suit {
                        constraints[i] = Some(vec![ROW_BOT]);
                    } else {
                        // Remaining card: Mid if not A/K, else no constraint
                        if ranks[i] != 12 && ranks[i] != 11 {
                            constraints[i] = Some(vec![ROW_MID]);
                        }
                    }
                }
            }
        }
    }

    for (i, &r) in ranks.iter().enumerate() {
        // Rule 4: Non-A/K quads (no joker) → must go to Bot
        if r < 13 && r != 12 && r != 11 && rank_count[r as usize] >= 4 && !has_joker {
            constraints[i] = Some(vec![ROW_BOT]);
            continue;  // Skip other rules for this card
        }
        // Rule 1: Q/J/T (rank 10,9,8) pair or trips → must go to Bot
        if (r == 10 || r == 9 || r == 8) && rank_count.get(r as usize).copied().unwrap_or(0) >= 2 {
            constraints[i] = Some(vec![ROW_BOT]);
        }
        // Rule 2: 2-6 (rank 0-4) trips → Bot or Top
        if r <= 4 && trips_ranks.contains(&r) {
            constraints[i] = Some(vec![ROW_BOT, ROW_TOP]);
        }
        // Rule 3: A/K (rank 12,11) pair → Top or Bot
        if (r == 12 || r == 11) && rank_count.get(r as usize).copied().unwrap_or(0) >= 2 {
            constraints[i] = Some(vec![ROW_TOP, ROW_BOT]);
        }
    }

    let has_constraints = constraints.iter().any(|c| c.is_some());
    if !has_constraints {
        return all;
    }

    // Filter actions
    let filtered: Vec<T0Action> = all.into_iter().filter(|action| {
        for (i, &(_card, row)) in action.iter().enumerate() {
            if let Some(ref allowed) = constraints[i] {
                if !allowed.contains(&row) {
                    return false;
                }
            }
        }
        true
    }).collect();

    // Safety: if pruning removed everything, return unpruned
    if filtered.is_empty() {
        return gen_t0_actions(cards, board);
    }
    filtered
}

fn gen_turn_actions(cards: &[CardIdx; 3], board: &Board) -> Vec<TurnAction> {
    let top_cap = 3 - board.top_n;
    let mid_cap = 5 - board.mid_n;
    let bot_cap = 5 - board.bot_n;
    let rows = [ROW_TOP, ROW_MID, ROW_BOT];

    let mut actions: Vec<TurnAction> = Vec::new();
    let mut seen: std::collections::HashSet<Board> = std::collections::HashSet::new();

    for disc in 0..3u8 {
        if cardidx_is_joker(cards[disc as usize]) { continue; }

        let remaining: Vec<CardIdx> = (0..3u8)
            .filter(|&i| i != disc)
            .map(|i| cards[i as usize])
            .collect();

        for &r0 in &rows {
            for &r1 in &rows {
                let mut cap = [top_cap, mid_cap, bot_cap];
                if cap[r0 as usize] == 0 { continue; }
                cap[r0 as usize] -= 1;
                if cap[r1 as usize] == 0 { continue; }

                let mut b = board.clone();
                b = b.place(remaining[0], r0);
                b = b.place(remaining[1], r1);

                if seen.insert(b) {
                    actions.push(TurnAction {
                        discard: cards[disc as usize],
                        placements: [(remaining[0], r0), (remaining[1], r1)],
                    });
                }
            }
        }
    }
    actions
}

// ============================================================
//  Terminal Evaluation
// ============================================================

struct FlEvConfig {
    bust_penalty: f64,
    fl_ev: HashMap<u8, f64>,
}

fn evaluate_terminal(board: &Board, config: &FlEvConfig) -> f64 {
    let top = board.top_cards();
    let mid = board.mid_cards();
    let bot = board.bot_cards();

    if !is_valid_placement(&top, &mid, &bot) {
        return config.bust_penalty;
    }

    let top_r = get_top_royalty(&top);
    let mid_r = get_middle_royalty(&mid);
    let bot_r = get_bottom_royalty(&bot);
    let total = (top_r + mid_r + bot_r) as f64;

    let (fl_qualified, fl_cards) = check_fl_entry(&top);
    let fl_bonus = if fl_qualified {
        config.fl_ev.get(&fl_cards).copied().unwrap_or(0.0)
    } else {
        0.0
    };

    total + fl_bonus
}

// ============================================================
//  Expectimax Engine
// ============================================================

#[derive(Clone)]
struct SamplingParams {
    n1: usize,
    n2: usize,
    n3: usize,
    n4: usize,
}

impl SamplingParams {
    fn samples_for_turn(&self, turn: u8) -> usize {
        match turn {
            1 => self.n1,
            2 => self.n2,
            3 => self.n3,
            4 => self.n4,
            _ => 1,
        }
    }
}

/// Result for a single T0 action
#[derive(Clone, Serialize)]
struct ActionResult {
    action_idx: usize,
    action_desc: String,
    ev: f64,
}

/// Result for a single T1-T4 turn action (within a ChoiceRecord)
#[derive(Clone, Serialize)]
struct TurnActionResult {
    desc: String,
    ev: f64,
}

/// Record of one T1-T4 choice node decision (board state + deal + all action EVs)
#[derive(Clone, Serialize)]
struct ChoiceRecord {
    turn: u8,
    top: Vec<String>,
    mid: Vec<String>,
    bot: Vec<String>,
    deal: Vec<String>,
    actions: Vec<TurnActionResult>,
}

/// Full result of expectimax solve for one T0 pattern
#[derive(Serialize)]
struct ExpectimaxResult {
    t0_hand: Vec<String>,
    n_actions: usize,
    params: [usize; 4],
    best_action_idx: usize,
    best_ev: f64,
    all_actions: Vec<ActionResult>,
    choice_records: Vec<ChoiceRecord>,
    elapsed_ms: u64,
}

/// Sample 3 cards from remaining deck without replacement
fn sample_3_cards(remaining: &[CardIdx], rng: &mut StdRng) -> [CardIdx; 3] {
    let n = remaining.len();
    debug_assert!(n >= 3);
    // Fisher-Yates partial shuffle for 3 elements
    let mut indices = [0usize; 3];
    let mut buf: Vec<usize> = (0..n).collect();
    for i in 0..3 {
        let j = rng.gen_range(i..n);
        buf.swap(i, j);
        indices[i] = buf[i];
    }
    [remaining[indices[0]], remaining[indices[1]], remaining[indices[2]]]
}

/// Remove 3 dealt cards from remaining deck
fn remove_dealt(remaining: &[CardIdx], dealt: &[CardIdx; 3]) -> Vec<CardIdx> {
    let mut result = Vec::with_capacity(remaining.len() - 3);
    let mut dealt_remaining = [true; 3];
    for &c in remaining {
        let mut found = false;
        for i in 0..3 {
            if dealt_remaining[i] && c == dealt[i] {
                dealt_remaining[i] = false;
                found = true;
                break;
            }
        }
        if !found {
            result.push(c);
        }
    }
    result
}

/// Chance node: average over N random 3-card draws
fn chance_node(
    board: &Board,
    turn: u8,
    remaining: &[CardIdx],
    config: &FlEvConfig,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> f64 {
    let n_samples = params.samples_for_turn(turn);
    let mut total = 0.0;
    for _ in 0..n_samples {
        let deal = sample_3_cards(remaining, rng);
        let new_remaining = remove_dealt(remaining, &deal);
        let val = choice_node(board, turn, &deal, &new_remaining, config, params, rng);
        total += val;
    }
    total / n_samples as f64
}

/// Choice node: pick placement with highest EV
fn choice_node(
    board: &Board,
    turn: u8,
    deal: &[CardIdx; 3],
    remaining: &[CardIdx],
    config: &FlEvConfig,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> f64 {
    let actions = gen_turn_actions(deal, board);
    let mut best = f64::NEG_INFINITY;
    for action in &actions {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board = new_board.place(card, row);
        }

        let val = if new_board.is_complete() {
            evaluate_terminal(&new_board, config)
        } else {
            chance_node(&new_board, turn + 1, remaining, config, params, rng)
        };
        if val > best { best = val; }
    }
    best
}

/// Top-level Expectimax solve: evaluate all T0 actions in parallel
fn expectimax_solve(
    t0_hand: &[CardIdx; 5],
    config: &FlEvConfig,
    params: &SamplingParams,
    seed: u64,
) -> ExpectimaxResult {
    let start = std::time::Instant::now();

    let actions = gen_t0_actions(t0_hand, &Board::new());
    let n_actions = actions.len();

    // Remaining deck after T0
    let remaining: Vec<CardIdx> = (0..54u8)
        .filter(|c| !t0_hand.contains(c))
        .collect();

    // Evaluate all T0 actions in parallel using rayon
    let action_results: Vec<ActionResult> = actions.par_iter().enumerate()
        .map(|(i, action)| {
            let mut board = Board::new();
            for &(card, row) in action {
                board = board.place(card, row);
            }

            // Each T0 action gets its own independent RNG
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));

            let ev = chance_node(&board, 1, &remaining, config, params, &mut rng);

            // Format action description
            let desc = format_t0_action(action);

            ActionResult {
                action_idx: i,
                action_desc: desc,
                ev,
            }
        })
        .collect();

    // Sort by EV descending
    let mut sorted = action_results.clone();
    sorted.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

    let best_ev = sorted[0].ev;
    let best_action_idx = sorted[0].action_idx;

    // Recording pass: re-run best T0 action's subtree to collect T1-T4 decisions.
    // Uses same seed as original evaluation → same T1 deals initially.
    // Overhead: ~0.5% of total computation (1/n_actions of one T0 subtree).
    let mut choice_records = Vec::new();
    {
        let best_action = &actions[best_action_idx];
        let mut board_after_t0 = Board::new();
        for &(card, row) in best_action {
            board_after_t0 = board_after_t0.place(card, row);
        }
        let mut record_rng = StdRng::seed_from_u64(seed.wrapping_add(best_action_idx as u64));
        chance_node_recording(
            &board_after_t0, 1, &remaining, config, params,
            &mut record_rng, &mut choice_records,
        );
    }

    let elapsed = start.elapsed().as_millis() as u64;

    ExpectimaxResult {
        t0_hand: t0_hand.iter().map(|&c| cardidx_to_string(c)).collect(),
        n_actions,
        params: [params.n1, params.n2, params.n3, params.n4],
        best_action_idx,
        best_ev,
        all_actions: sorted,
        choice_records,
        elapsed_ms: elapsed,
    }
}

/// Format a T0 action as human-readable string
fn format_t0_action(action: &T0Action) -> String {
    let row_names = ["T", "M", "B"];
    let mut parts: Vec<String> = Vec::new();
    for &(card, row) in action {
        parts.push(format!("{}→{}", cardidx_to_string(card), row_names[row as usize]));
    }
    parts.join(" ")
}

/// Format a T1-T4 turn action as human-readable string
fn format_turn_action(action: &TurnAction) -> String {
    let row_names = ["T", "M", "B"];
    format!("d:{} {}→{} {}→{}",
        cardidx_to_string(action.discard),
        cardidx_to_string(action.placements[0].0),
        row_names[action.placements[0].1 as usize],
        cardidx_to_string(action.placements[1].0),
        row_names[action.placements[1].1 as usize],
    )
}

/// Chance node with recording: samples deals and records choice nodes along best path.
/// Called only for the best T0 action's subtree (adds ~0.5% overhead).
fn chance_node_recording(
    board: &Board,
    turn: u8,
    remaining: &[CardIdx],
    config: &FlEvConfig,
    params: &SamplingParams,
    rng: &mut StdRng,
    records: &mut Vec<ChoiceRecord>,
) -> f64 {
    let n_samples = params.samples_for_turn(turn);
    let mut total = 0.0;
    for _ in 0..n_samples {
        let deal = sample_3_cards(remaining, rng);
        let new_remaining = remove_dealt(remaining, &deal);
        let val = choice_node_recording(
            board, turn, &deal, &new_remaining, config, params, rng, records,
        );
        total += val;
    }
    total / n_samples as f64
}

/// Choice node with recording: evaluates all actions, records the decision,
/// then recurses into best action's subtree for deeper records.
fn choice_node_recording(
    board: &Board,
    turn: u8,
    deal: &[CardIdx; 3],
    remaining: &[CardIdx],
    config: &FlEvConfig,
    params: &SamplingParams,
    rng: &mut StdRng,
    records: &mut Vec<ChoiceRecord>,
) -> f64 {
    let actions = gen_turn_actions(deal, board);
    let mut best_val = f64::NEG_INFINITY;
    let mut best_idx = 0;
    let mut action_evs: Vec<f64> = Vec::with_capacity(actions.len());

    // Evaluate all actions using NON-recording functions (exact same computation)
    for (i, action) in actions.iter().enumerate() {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board = new_board.place(card, row);
        }
        let val = if new_board.is_complete() {
            evaluate_terminal(&new_board, config)
        } else {
            chance_node(&new_board, turn + 1, remaining, config, params, rng)
        };
        action_evs.push(val);
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }

    // Build sorted action results for this record
    let mut turn_actions: Vec<TurnActionResult> = actions
        .iter()
        .zip(action_evs.iter())
        .map(|(a, &ev)| TurnActionResult {
            desc: format_turn_action(a),
            ev,
        })
        .collect();
    turn_actions.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

    // Record this choice node
    records.push(ChoiceRecord {
        turn,
        top: board.top[..board.top_n as usize]
            .iter()
            .map(|&c| cardidx_to_string(c))
            .collect(),
        mid: board.mid[..board.mid_n as usize]
            .iter()
            .map(|&c| cardidx_to_string(c))
            .collect(),
        bot: board.bot[..board.bot_n as usize]
            .iter()
            .map(|&c| cardidx_to_string(c))
            .collect(),
        deal: deal.iter().map(|&c| cardidx_to_string(c)).collect(),
        actions: turn_actions,
    });

    // Recurse into best action's subtree WITH recording (for deeper turn records)
    let best_action = &actions[best_idx];
    let mut new_board = board.clone();
    for &(card, row) in &best_action.placements {
        new_board = new_board.place(card, row);
    }
    if !new_board.is_complete() {
        chance_node_recording(&new_board, turn + 1, remaining, config, params, rng, records);
    }

    best_val
}

// ============================================================
//  Card String Parsing
// ============================================================

fn parse_card_str(s: &str) -> Option<CardIdx> {
    let s = s.trim();
    if s == "X1" { return Some(JOKER1); }
    if s == "X2" { return Some(JOKER2); }
    if s.len() != 2 { return None; }
    let chars: Vec<char> = s.chars().collect();
    let rank = match chars[0] {
        '2' => 0, '3' => 1, '4' => 2, '5' => 3, '6' => 4, '7' => 5,
        '8' => 6, '9' => 7, 'T' => 8, 'J' => 9, 'Q' => 10, 'K' => 11, 'A' => 12,
        _ => return None,
    };
    let suit = match chars[1] {
        's' => 0, 'h' => 1, 'd' => 2, 'c' => 3,
        _ => return None,
    };
    Some(rank * 4 + suit)
}

fn cardidx_to_string(idx: CardIdx) -> String {
    if idx == JOKER1 { return "X1".to_string(); }
    if idx == JOKER2 { return "X2".to_string(); }
    let rank = idx / 4;
    let suit = idx % 4;
    let r = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'][rank as usize];
    let s = ['s','h','d','c'][suit as usize];
    format!("{}{}", r, s)
}

// ============================================================
//  JSON Protocol (stdin mode)
// ============================================================

#[derive(Deserialize)]
struct SolveRequest {
    t0_hand: Vec<String>,
    #[serde(default = "default_n1")]
    n1: usize,
    #[serde(default = "default_n2")]
    n2: usize,
    #[serde(default = "default_n3")]
    n3: usize,
    #[serde(default = "default_n4")]
    n4: usize,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    fl_ev: HashMap<String, f64>,
    #[serde(default = "default_bust_penalty")]
    bust_penalty: f64,
    /// How many top actions to return (0 = all)
    #[serde(default)]
    top_k: usize,
}

fn default_n1() -> usize { 500 }
fn default_n2() -> usize { 100 }
fn default_n3() -> usize { 20 }
fn default_n4() -> usize { 5 }
fn default_seed() -> u64 { 42 }
fn default_bust_penalty() -> f64 { -6.0 }

fn run_stdin() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() { continue; }

        let req: SolveRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                writeln!(stdout, "{{\"error\":\"{}\"}}", e).unwrap();
                stdout.flush().unwrap();
                continue;
            }
        };

        let hand_indices: Vec<CardIdx> = req.t0_hand.iter()
            .filter_map(|s| parse_card_str(s))
            .collect();
        if hand_indices.len() != 5 {
            writeln!(stdout, "{{\"error\":\"need exactly 5 cards\"}}").unwrap();
            stdout.flush().unwrap();
            continue;
        }
        let mut t0: [CardIdx; 5] = [0; 5];
        t0.copy_from_slice(&hand_indices);

        let mut fl_ev_map = HashMap::new();
        for (k, v) in &req.fl_ev {
            if let Ok(cards) = k.parse::<u8>() {
                fl_ev_map.insert(cards, *v);
            }
        }
        let config = FlEvConfig {
            bust_penalty: req.bust_penalty,
            fl_ev: fl_ev_map,
        };
        let params = SamplingParams {
            n1: req.n1, n2: req.n2, n3: req.n3, n4: req.n4,
        };

        let mut result = expectimax_solve(&t0, &config, &params, req.seed);

        // Trim to top_k if requested
        if req.top_k > 0 && req.top_k < result.all_actions.len() {
            result.all_actions.truncate(req.top_k);
        }

        writeln!(stdout, "{}", serde_json::to_string(&result).unwrap()).unwrap();
        stdout.flush().unwrap();
    }
}

// ============================================================
//  CLI
// ============================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "expectimax" {
        run_expectimax_cli(&args[2..]);
    } else if args.len() > 1 && args[1] == "stdin" {
        run_stdin();
    } else if args.len() > 1 && args[1] == "enumerate" {
        run_enumerate();
    } else if args.len() > 1 && args[1] == "t3eval" {
        run_t3_eval(&args[2..]);
    } else if args.len() > 1 && args[1] == "t2eval" {
        run_t2_eval(&args[2..]);
    } else {
        eprintln!("Usage:");
        eprintln!("  backward expectimax --t0 Ah,Kh,Qd,7c,3s --n1 10 --n2 3 --n3 2 --n4 2");
        eprintln!("  backward enumerate  # list canonical T0 patterns");
        eprintln!("  backward stdin      # JSON protocol on stdin/stdout");
        eprintln!("  backward t3eval     # T3-focused evaluation (bottom-up)");
        eprintln!("  backward t2eval     # T2-focused deep evaluation (with pruning)");
        eprintln!("    --exhaustive      # enumerate ALL T4 deals (exact EV)");
        eprintln!("    --top-k N         # keep only top-N actions at each level (default: 4)");
    }
}

fn run_expectimax_cli(args: &[String]) {
    let mut t0_str = String::new();
    let mut n1 = 500usize;
    let mut n2 = 100usize;
    let mut n3 = 20usize;
    let mut n4 = 5usize;
    let mut seed = 42u64;
    let mut top_k = 20usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--t0" => { i += 1; t0_str = args[i].clone(); }
            "--n1" => { i += 1; n1 = args[i].parse().unwrap(); }
            "--n2" => { i += 1; n2 = args[i].parse().unwrap(); }
            "--n3" => { i += 1; n3 = args[i].parse().unwrap(); }
            "--n4" => { i += 1; n4 = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--top-k" | "--top" => { i += 1; top_k = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }

    let hand_indices: Vec<CardIdx> = t0_str.split(',')
        .filter_map(|s| parse_card_str(s))
        .collect();
    if hand_indices.len() != 5 {
        eprintln!("Error: need exactly 5 cards, got {}", hand_indices.len());
        std::process::exit(1);
    }
    let mut t0: [CardIdx; 5] = [0; 5];
    t0.copy_from_slice(&hand_indices);

    let params = SamplingParams { n1, n2, n3, n4 };

    // Load FL EV from config
    let config = FlEvConfig {
        bust_penalty: -6.0,
        fl_ev: [(14u8, 15.8), (15, 22.7), (16, 28.6), (17, 35.1)]
            .iter().cloned().collect(),
    };

    eprintln!("=== Expectimax Deep Search ===");
    eprintln!("T0 hand: {}", t0.iter().map(|&c| cardidx_to_string(c)).collect::<Vec<_>>().join(", "));
    eprintln!("Samples: N1={}, N2={}, N3={}, N4={}", n1, n2, n3, n4);
    eprintln!("Seed: {}", seed);

    let result = expectimax_solve(&t0, &config, &params, seed);

    eprintln!("Completed in {:.1}s", result.elapsed_ms as f64 / 1000.0);
    eprintln!("{} T0 actions evaluated", result.n_actions);

    // Print results
    let show = if top_k == 0 { result.all_actions.len() } else { top_k.min(result.all_actions.len()) };
    println!("=== Top {} / {} actions (sorted by EV) ===", show, result.n_actions);
    for (rank, ar) in result.all_actions.iter().take(show).enumerate() {
        let marker = if rank == 0 { " <<<BEST" } else { "" };
        println!("#{:>3}  EV={:>7.2}  {}{}", rank + 1, ar.ev, ar.action_desc, marker);
    }

    // Also show bottom 5
    if result.all_actions.len() > show {
        println!("\n=== Bottom 5 actions ===");
        let start = result.all_actions.len().saturating_sub(5);
        for (i, ar) in result.all_actions[start..].iter().enumerate() {
            println!("#{:>3}  EV={:>7.2}  {}", start + i + 1, ar.ev, ar.action_desc);
        }
    }

    println!("\nBest EV: {:.2}", result.best_ev);
    println!("Elapsed: {:.1}s", result.elapsed_ms as f64 / 1000.0);

    // Choice records summary
    let mut by_turn = [0usize; 5];
    for r in &result.choice_records {
        if (r.turn as usize) < by_turn.len() {
            by_turn[r.turn as usize] += 1;
        }
    }
    println!("\nChoice records: {} total (T1={}, T2={}, T3={}, T4={})",
        result.choice_records.len(), by_turn[1], by_turn[2], by_turn[3], by_turn[4]);
}

// ============================================================
//  T3-focused bottom-up evaluation
// ============================================================

/// Quick 1-sample look-ahead for picking "reasonable" actions at T1/T2.
/// Evaluates each action by recursing with N=1 per future turn.
fn quick_future_eval(
    board: &Board,
    remaining: &[CardIdx],
    config: &FlEvConfig,
    rng: &mut StdRng,
) -> f64 {
    if board.is_complete() {
        return evaluate_terminal(board, config);
    }
    if remaining.len() < 3 {
        return evaluate_terminal(board, config);
    }
    let deal = sample_3_cards(remaining, rng);
    let new_remaining = remove_dealt(remaining, &deal);
    let actions = gen_turn_actions(&deal, board);
    let mut best = f64::NEG_INFINITY;
    for action in &actions {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board = new_board.place(card, row);
        }
        let val = quick_future_eval(&new_board, &new_remaining, config, rng);
        if val > best { best = val; }
    }
    best
}

/// Pick best action at a turn using quick_future_eval (N=1 per future level).
fn quick_pick_best(
    board: &Board,
    deal: &[CardIdx; 3],
    remaining: &[CardIdx],
    config: &FlEvConfig,
    rng: &mut StdRng,
    n_quick: usize,
) -> Option<TurnAction> {
    let actions = gen_turn_actions(deal, board);
    if actions.is_empty() { return None; }
    if actions.len() == 1 {
        return Some(TurnAction {
            discard: actions[0].discard,
            placements: actions[0].placements,
        });
    }
    let mut best_val = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, action) in actions.iter().enumerate() {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board = new_board.place(card, row);
        }
        let mut total = 0.0;
        for _ in 0..n_quick {
            total += quick_future_eval(&new_board, remaining, config, rng);
        }
        let val = total / n_quick as f64;
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Some(TurnAction {
        discard: actions[best_idx].discard,
        placements: actions[best_idx].placements,
    })
}

/// Analytical T4 evaluation: sample N4 T4 deals, pick best terminal for each.
fn t4_analytical_eval(
    board: &Board,
    remaining: &[CardIdx],
    config: &FlEvConfig,
    n4: usize,
    rng: &mut StdRng,
) -> f64 {
    if board.is_complete() {
        return evaluate_terminal(board, config);
    }
    if remaining.len() < 3 {
        return evaluate_terminal(board, config);
    }
    let mut total = 0.0;
    for _ in 0..n4 {
        let deal = sample_3_cards(remaining, rng);
        let actions = gen_turn_actions(&deal, board);
        let mut best = f64::NEG_INFINITY;
        for action in &actions {
            let mut final_board = board.clone();
            for &(card, row) in &action.placements {
                final_board = final_board.place(card, row);
            }
            let val = evaluate_terminal(&final_board, config);
            if val > best { best = val; }
        }
        if best > f64::NEG_INFINITY {
            total += best;
        }
    }
    total / n4 as f64
}

/// Exhaustive T4 evaluation: enumerate ALL C(remaining, 3) deals.
/// Returns exact expected value (no sampling noise).
/// Optimized: pre-converts cards, uses stack arrays, avoids redundant hand evals.
fn t4_exhaustive_eval(
    board: &Board,
    remaining: &[CardIdx],
    config: &FlEvConfig,
) -> f64 {
    if board.is_complete() {
        return evaluate_terminal(board, config);
    }
    let n = remaining.len();
    if n < 3 {
        return evaluate_terminal(board, config);
    }

    // Pre-convert board cards to Card (once)
    let top_n = board.top_n as usize;
    let mid_n = board.mid_n as usize;
    let bot_n = board.bot_n as usize;
    let mut base_top = [Card { rank: 0, suit: 4 }; 3];
    let mut base_mid = [Card { rank: 0, suit: 4 }; 5];
    let mut base_bot = [Card { rank: 0, suit: 4 }; 5];
    for i in 0..top_n { base_top[i] = cardidx_to_card(board.top[i]); }
    for i in 0..mid_n { base_mid[i] = cardidx_to_card(board.mid[i]); }
    for i in 0..bot_n { base_bot[i] = cardidx_to_card(board.bot[i]); }

    let top_cap = 3 - top_n;
    let mid_cap = 5 - mid_n;
    let bot_cap = 5 - bot_n;

    // Pre-evaluate rows that are already complete (won't change)
    let top_complete = top_cap == 0;
    let mid_complete = mid_cap == 0;
    let bot_complete = bot_cap == 0;
    let cached_top = if top_complete { Some(eval_row_3(&base_top[..3])) } else { None };
    let cached_mid = if mid_complete { Some(eval_row_5(&base_mid[..5])) } else { None };
    let cached_bot = if bot_complete { Some(eval_row_5(&base_bot[..5])) } else { None };

    // Pre-convert remaining deck to Card (once)
    let rem_cards: Vec<Card> = remaining.iter().map(|&i| cardidx_to_card(i)).collect();

    let mut total = 0.0;
    let mut count = 0u64;

    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                let cidx = [remaining[i], remaining[j], remaining[k]];
                let cards = [rem_cards[i], rem_cards[j], rem_cards[k]];

                let mut best = f64::NEG_INFINITY;

                // Enumerate placements: 3 discard choices × valid (r0, r1) pairs
                for disc in 0..3usize {
                    if cardidx_is_joker(cidx[disc]) { continue; }
                    let c0 = cards[(disc + 1) % 3];
                    let c1 = cards[(disc + 2) % 3];

                    for r0 in 0..3u8 {
                        let cap0 = match r0 { 0 => top_cap, 1 => mid_cap, 2 => bot_cap, _ => 0 };
                        if cap0 == 0 { continue; }

                        for r1 in 0..3u8 {
                            let cap1 = if r0 == r1 { cap0 - 1 } else {
                                match r1 { 0 => top_cap, 1 => mid_cap, 2 => bot_cap, _ => 0 }
                            };
                            if cap1 == 0 { continue; }

                            // Build final rows on stack
                            let mut ft = base_top;
                            let mut fm = base_mid;
                            let mut fb = base_bot;
                            let mut tn = top_n;
                            let mut mn = mid_n;
                            let mut bn = bot_n;

                            // Track which rows changed
                            let mut top_changed = false;
                            let mut mid_changed = false;
                            let mut bot_changed = false;

                            match r0 {
                                0 => { ft[tn] = c0; tn += 1; top_changed = true; }
                                1 => { fm[mn] = c0; mn += 1; mid_changed = true; }
                                2 => { fb[bn] = c0; bn += 1; bot_changed = true; }
                                _ => {}
                            }
                            match r1 {
                                0 => { ft[tn] = c1; top_changed = true; }
                                1 => { fm[mn] = c1; mid_changed = true; }
                                2 => { fb[bn] = c1; bot_changed = true; }
                                _ => {}
                            }

                            // Use cached eval for unchanged rows, recompute only changed ones
                            let te = if top_changed { eval_row_3(&ft[..3]) } else { cached_top.unwrap() };
                            let me = if mid_changed { eval_row_5(&fm[..5]) } else { cached_mid.unwrap() };
                            let be = if bot_changed { eval_row_5(&fb[..5]) } else { cached_bot.unwrap() };

                            let val = score_from_evals(&te, &me, &be, &ft[..3], &fm[..5], &fb[..5], config);
                            if val > best { best = val; }
                        }
                    }
                }

                if best > f64::NEG_INFINITY {
                    total += best;
                    count += 1;
                }
            }
        }
    }

    if count == 0 { 0.0 } else { total / count as f64 }
}

/// Cached hand evaluation for a row.
#[derive(Clone, Copy)]
struct RowEval5 {
    rank: HandRank,
    strength: u32,
    royalty_mid: i32,
    royalty_bot: i32,
}

#[derive(Clone, Copy)]
struct RowEval3 {
    rank: HandRank3,
    royalty: i32,
    fl_qualified: bool,
    fl_cards: u8,
}

fn eval_row_5(cards: &[Card]) -> RowEval5 {
    let (rank, strength) = evaluate_5_card(cards);
    RowEval5 {
        rank, strength,
        royalty_mid: match rank {
            HandRank::Trips => 2, HandRank::Straight => 4, HandRank::Flush => 8,
            HandRank::FullHouse => 12, HandRank::Quads => 20,
            HandRank::StraightFlush => 30, HandRank::RoyalFlush => 50, _ => 0,
        },
        royalty_bot: match rank {
            HandRank::Straight => 2, HandRank::Flush => 4, HandRank::FullHouse => 6,
            HandRank::Quads => 10, HandRank::StraightFlush => 15,
            HandRank::RoyalFlush => 25, _ => 0,
        },
    }
}

fn eval_row_3(cards: &[Card]) -> RowEval3 {
    let (rank, _) = evaluate_3_card(cards);
    let rc = count_ranks(cards);
    let j = count_jokers(cards);
    let royalty = match rank {
        HandRank3::Trips => {
            (2..=14).rev().find(|&r| rc[r] + j >= 3 && rc[r] >= 1)
                .map(|r| 10 + r as i32 - 2).unwrap_or(0)
        }
        HandRank3::OnePair => {
            (6..=14).rev().find(|&r| rc[r] + j >= 2 && rc[r] >= 1)
                .map(|r| r as i32 - 5).unwrap_or(0)
        }
        HandRank3::HighCard => 0,
    };
    let (fl_qualified, fl_cards) = {
        let mut fq = false; let mut fc = 0u8;
        for r in (2..=14).rev() {
            if rc[r] + j >= 3 && rc[r] >= 1 { fq = true; fc = 17; break; }
            if rc[r] + j >= 2 && rc[r] >= 1 {
                match r { 14 => { fq = true; fc = 16; } 13 => { fq = true; fc = 15; }
                           12 => { fq = true; fc = 14; } _ => {} }
                break;
            }
        }
        (fq, fc)
    };
    RowEval3 { rank, royalty, fl_qualified, fl_cards }
}

/// Score from cached row evaluations. Returns bust_penalty or royalties+FL.
fn score_from_evals(
    te: &RowEval3, me: &RowEval5, be: &RowEval5,
    top: &[Card], mid: &[Card], bot: &[Card],
    config: &FlEvConfig,
) -> f64 {
    // Bot >= Mid check
    if (be.rank as u8) < (me.rank as u8) { return config.bust_penalty; }
    if (be.rank as u8) == (me.rank as u8) && be.strength < me.strength {
        if compare_5_hands(bot, mid) < 0 { return config.bust_penalty; }
    }

    // Top <= Mid check
    let top_f = match te.rank { HandRank3::HighCard => 0.0, HandRank3::OnePair => 1.0, HandRank3::Trips => 2.5 };
    let mid_f = me.rank as u8 as f64;
    if top_f > mid_f + 0.01 { return config.bust_penalty; }
    if (top_f - mid_f).abs() < 0.01 {
        // Same category — detailed check
        match te.rank {
            HandRank3::HighCard => {
                let th = top.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
                let mh = mid.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
                if th > mh { return config.bust_penalty; }
            }
            HandRank3::OnePair => {
                let tp = get_pair_rank(top); let mp = get_pair_rank(mid);
                if tp > mp { return config.bust_penalty; }
                if tp == mp {
                    let tk = top.iter().filter(|c| !c.is_joker() && c.rank != tp).map(|c| c.rank).max().unwrap_or(0);
                    let mk = mid.iter().filter(|c| !c.is_joker() && c.rank != mp).map(|c| c.rank).max().unwrap_or(0);
                    if tk > mk { return config.bust_penalty; }
                }
            }
            HandRank3::Trips => {
                if me.rank == HandRank::Trips {
                    if get_trips_rank(top) > get_trips_rank(mid) { return config.bust_penalty; }
                }
            }
        }
    }

    let total = (te.royalty + me.royalty_mid + be.royalty_bot) as f64;
    let fl_bonus = if te.fl_qualified { config.fl_ev.get(&te.fl_cards).copied().unwrap_or(0.0) } else { 0.0 };
    total + fl_bonus
}

/// T3 choice record for bottom-up data
#[derive(Serialize)]
struct T3Record {
    turn: u8,
    top: Vec<String>,
    mid: Vec<String>,
    bot: Vec<String>,
    deal: Vec<String>,
    actions: Vec<TurnActionResult>,
}

/// T3-focused evaluation result for one pattern
#[derive(Serialize)]
struct T3EvalResult {
    t0_hand: Vec<String>,
    t3_records: Vec<T3Record>,
    elapsed_ms: u64,
}

/// Run T3-focused bottom-up evaluation for one pattern.
/// T0: use existing best action. T1/T2: quick eval. T3: deep N4 eval.
/// If exhaustive_t4=true, uses exact C(remaining,3) enumeration instead of N4 sampling.
fn t3_eval_pattern(
    t0_hand: &[CardIdx; 5],
    config: &FlEvConfig,
    n1: usize,
    n2: usize,
    n3: usize,
    n4_deep: usize,
    n_quick: usize,
    seed: u64,
    exhaustive_t4: bool,
) -> T3EvalResult {
    let start = std::time::Instant::now();

    // Find best T0 action using quick eval
    let t0_actions = gen_t0_actions(t0_hand, &Board::new());
    let remaining_after_t0: Vec<CardIdx> = (0..54u8)
        .filter(|c| !t0_hand.contains(c))
        .collect();

    // Quick-evaluate T0 actions in parallel
    let t0_evs: Vec<(usize, f64)> = t0_actions.par_iter().enumerate()
        .map(|(i, action)| {
            let mut board = Board::new();
            for &(card, row) in action {
                board = board.place(card, row);
            }
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            // Quick eval with a few samples
            let mut total = 0.0;
            for _ in 0..3 {
                total += quick_future_eval(&board, &remaining_after_t0, config, &mut rng);
            }
            (i, total / 3.0)
        })
        .collect();

    let best_t0_idx = t0_evs.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|x| x.0)
        .unwrap_or(0);

    let best_t0 = &t0_actions[best_t0_idx];
    let mut board_t0 = Board::new();
    for &(card, row) in best_t0 {
        board_t0 = board_t0.place(card, row);
    }

    // Now do T1→T2→T3 traversal
    let mut all_t3_records: Vec<T3Record> = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));

    for _ in 0..n1 {
        // T1: sample deal, quick pick best
        if remaining_after_t0.len() < 3 { break; }
        let deal_t1 = sample_3_cards(&remaining_after_t0, &mut rng);
        let rem_t1 = remove_dealt(&remaining_after_t0, &deal_t1);
        let best_t1 = match quick_pick_best(&board_t0, &deal_t1, &rem_t1, config, &mut rng, n_quick) {
            Some(a) => a,
            None => continue,
        };
        let mut board_t1 = board_t0.clone();
        for &(card, row) in &best_t1.placements {
            board_t1 = board_t1.place(card, row);
        }

        for _ in 0..n2 {
            // T2: sample deal, quick pick best
            if rem_t1.len() < 3 { break; }
            let deal_t2 = sample_3_cards(&rem_t1, &mut rng);
            let rem_t2 = remove_dealt(&rem_t1, &deal_t2);
            let best_t2 = match quick_pick_best(&board_t1, &deal_t2, &rem_t2, config, &mut rng, n_quick) {
                Some(a) => a,
                None => continue,
            };
            let mut board_t2 = board_t1.clone();
            for &(card, row) in &best_t2.placements {
                board_t2 = board_t2.place(card, row);
            }

            for _ in 0..n3 {
                // T3: sample deal, DEEP eval ALL actions
                if rem_t2.len() < 3 { break; }
                let deal_t3 = sample_3_cards(&rem_t2, &mut rng);
                let rem_t3 = remove_dealt(&rem_t2, &deal_t3);
                let actions = gen_turn_actions(&deal_t3, &board_t2);
                if actions.is_empty() { continue; }

                let mut turn_results: Vec<TurnActionResult> = Vec::new();
                for action in &actions {
                    let mut board_t3 = board_t2.clone();
                    for &(card, row) in &action.placements {
                        board_t3 = board_t3.place(card, row);
                    }
                    let ev = if exhaustive_t4 {
                        t4_exhaustive_eval(&board_t3, &rem_t3, config)
                    } else {
                        t4_analytical_eval(&board_t3, &rem_t3, config, n4_deep, &mut rng)
                    };
                    turn_results.push(TurnActionResult {
                        desc: format_turn_action(action),
                        ev,
                    });
                }

                // Sort by EV desc
                turn_results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

                all_t3_records.push(T3Record {
                    turn: 3,
                    top: board_t2.top[..board_t2.top_n as usize]
                        .iter().map(|&c| cardidx_to_string(c)).collect(),
                    mid: board_t2.mid[..board_t2.mid_n as usize]
                        .iter().map(|&c| cardidx_to_string(c)).collect(),
                    bot: board_t2.bot[..board_t2.bot_n as usize]
                        .iter().map(|&c| cardidx_to_string(c)).collect(),
                    deal: deal_t3.iter().map(|&c| cardidx_to_string(c)).collect(),
                    actions: turn_results,
                });
            }
        }
    }

    let elapsed = start.elapsed().as_millis() as u64;
    T3EvalResult {
        t0_hand: t0_hand.iter().map(|&c| cardidx_to_string(c)).collect(),
        t3_records: all_t3_records,
        elapsed_ms: elapsed,
    }
}

/// Run T3 evaluation in stdin mode: one JSON line per pattern.
fn run_t3_eval(args: &[String]) {
    let mut n1 = 50usize;
    let mut n2 = 10usize;
    let mut n3 = 3usize;
    let mut n4_deep = 20usize;
    let mut n_quick = 3usize;
    let mut seed = 42u64;
    let mut patterns_file = String::new();
    let mut exhaustive_t4 = false;
    let mut bust_penalty = -6.0f64;
    let mut fl_ev_14 = 14.0f64;
    let mut fl_ev_15 = 27.9f64;
    let mut fl_ev_16 = 52.4f64;
    let mut fl_ev_17 = 104.5f64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n1" => { i += 1; n1 = args[i].parse().unwrap(); }
            "--n2" => { i += 1; n2 = args[i].parse().unwrap(); }
            "--n3" => { i += 1; n3 = args[i].parse().unwrap(); }
            "--n4" => { i += 1; n4_deep = args[i].parse().unwrap(); }
            "--n-quick" => { i += 1; n_quick = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--patterns" => { i += 1; patterns_file = args[i].clone(); }
            "--exhaustive" => { exhaustive_t4 = true; }
            "--bust-penalty" => { i += 1; bust_penalty = args[i].parse().unwrap(); }
            "--fl-ev-14" => { i += 1; fl_ev_14 = args[i].parse().unwrap(); }
            "--fl-ev-15" => { i += 1; fl_ev_15 = args[i].parse().unwrap(); }
            "--fl-ev-16" => { i += 1; fl_ev_16 = args[i].parse().unwrap(); }
            "--fl-ev-17" => { i += 1; fl_ev_17 = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }

    let config = FlEvConfig {
        bust_penalty,
        fl_ev: [(14u8, fl_ev_14), (15, fl_ev_15), (16, fl_ev_16), (17, fl_ev_17)]
            .iter().cloned().collect(),
    };

    eprintln!("=== T3-Focused Bottom-Up Evaluation ===");
    if exhaustive_t4 {
        eprintln!("  Mode: EXHAUSTIVE T4 (all C(remaining,3) deals)");
    }
    eprintln!("  N1={}, N2={}, N3={}{}, N_quick={}", n1, n2, n3,
        if exhaustive_t4 { " (N4=exhaustive)".to_string() } else { format!(", N4_deep={}", n4_deep) },
        n_quick);
    eprintln!("  bust_penalty={}, FL EV: 14={}, 15={}, 16={}, 17={}",
        bust_penalty, fl_ev_14, fl_ev_15, fl_ev_16, fl_ev_17);
    eprintln!("  T3 states/pattern = {} × {} × {} = {}", n1, n2, n3, n1*n2*n3);

    // Read patterns from file or enumerate
    let patterns: Vec<[CardIdx; 5]> = if !patterns_file.is_empty() {
        let content = std::fs::read_to_string(&patterns_file).unwrap();
        content.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                let cards: Vec<CardIdx> = l.split(',')
                    .filter_map(|s| parse_card_str(s))
                    .collect();
                if cards.len() == 5 {
                    let mut arr = [0u8; 5];
                    arr.copy_from_slice(&cards);
                    Some(arr)
                } else { None }
            })
            .collect()
    } else {
        // Read from stdin: one pattern per line (5 comma-separated cards)
        let stdin = io::stdin();
        stdin.lock().lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                let cards: Vec<CardIdx> = l.split(',')
                    .filter_map(|s| parse_card_str(s))
                    .collect();
                if cards.len() == 5 {
                    let mut arr = [0u8; 5];
                    arr.copy_from_slice(&cards);
                    Some(arr)
                } else { None }
            })
            .collect()
    };

    eprintln!("  Patterns: {}", patterns.len());
    let total_start = std::time::Instant::now();

    // Process patterns in parallel using rayon
    let results: Vec<T3EvalResult> = patterns.par_iter().enumerate()
        .map(|(idx, pattern)| {
            let result = t3_eval_pattern(
                pattern, &config, n1, n2, n3, n4_deep, n_quick,
                seed.wrapping_add(idx as u64 * 100000),
                exhaustive_t4,
            );
            if (idx + 1) % 50 == 0 || idx + 1 == patterns.len() {
                eprintln!("  [{}/{}] {} T3 records in {}ms",
                    idx + 1, patterns.len(),
                    result.t3_records.len(),
                    result.elapsed_ms);
            }
            result
        })
        .collect();

    // Output all results as JSON (one per line)
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    for result in &results {
        writeln!(stdout, "{}", serde_json::to_string(result).unwrap()).unwrap();
    }
    stdout.flush().unwrap();

    let total_elapsed = total_start.elapsed();
    let total_records: usize = results.iter().map(|r| r.t3_records.len()).sum();
    eprintln!("\n=== T3 Evaluation Complete ===");
    eprintln!("  Total patterns: {}", patterns.len());
    eprintln!("  Total T3 records: {}", total_records);
    eprintln!("  Total time: {:.1}s", total_elapsed.as_secs_f64());
}

// ============================================================
//  Pruning: Partial Bust Detection
// ============================================================

/// Check if a partial board is guaranteed to bust based on filled rows.
/// Returns true if bust is certain given current placements.
fn can_bust_partial(board: &Board) -> bool {
    let top_n = board.top_n as usize;
    let mid_n = board.mid_n as usize;
    let bot_n = board.bot_n as usize;

    // Check top vs mid only when both are complete
    if top_n == 3 && mid_n == 5 {
        let top = board.top_cards();
        let mid = board.mid_cards();
        if !is_top_le_mid_partial(&top, &mid) {
            return true;
        }
    }

    // Check bot vs mid only when both are complete
    if mid_n == 5 && bot_n == 5 {
        let mid = board.mid_cards();
        let bot = board.bot_cards();
        if compare_5_hands(&bot, &mid) < 0 {
            return true;
        }
    }

    false
}

/// Relaxed top <= mid check for partial bust detection.
/// Uses same logic as score_from_evals bust check.
fn is_top_le_mid_partial(top: &[Card], mid: &[Card]) -> bool {
    let (top_rank, _) = evaluate_3_card(top);
    let (mid_rank, _) = evaluate_5_card(mid);

    let top_f: f64 = match top_rank {
        HandRank3::HighCard => 0.0,
        HandRank3::OnePair => 1.0,
        HandRank3::Trips => 2.5,
    };
    let mid_f = mid_rank as u8 as f64;

    if top_f < mid_f - 0.01 { return true; }
    if top_f > mid_f + 0.01 { return false; }

    // Same category — detailed check
    match top_rank {
        HandRank3::HighCard => {
            let th = top.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
            let mh = mid.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
            th <= mh
        }
        HandRank3::OnePair => {
            let tp = get_pair_rank(top);
            let mp = get_pair_rank(mid);
            if tp != mp { return tp < mp; }
            let tk = top.iter().filter(|c| !c.is_joker() && c.rank != tp).map(|c| c.rank).max().unwrap_or(0);
            let mk = mid.iter().filter(|c| !c.is_joker() && c.rank != mp).map(|c| c.rank).max().unwrap_or(0);
            tk <= mk
        }
        HandRank3::Trips => {
            if mid_rank == HandRank::Trips {
                get_trips_rank(top) <= get_trips_rank(mid)
            } else {
                true
            }
        }
    }
}



/// Generate turn actions with bust pruning + scatter-card rule + row ordering check.
/// Filters out:
///   - Actions that lead to guaranteed bust
///   - Actions placing 4+ unpaired cards in mid/bot without flush or straight draw
///   - Actions where mid's floor > bot's ceiling (guaranteed ordering violation)
fn gen_turn_actions_pruned(cards: &[CardIdx; 3], board: &Board) -> Vec<TurnAction> {
    let actions = gen_turn_actions(cards, board);
    actions.into_iter().filter(|action| {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board = new_board.place(card, row);
        }
        // Bust check (complete rows)
        if can_bust_partial(&new_board) { return false; }
        // Scatter-card check for mid and bot (only when 4+ cards placed)
        if new_board.mid_n >= 4 && has_scatter_no_draw(&new_board.mid[..new_board.mid_n as usize]) {
            return false;
        }
        if new_board.bot_n >= 4 && has_scatter_no_draw(&new_board.bot[..new_board.bot_n as usize]) {
            return false;
        }
        // Low-top rule: reject if top has 2+ cards, all J or lower (rank<=9), no pair/joker
        if new_board.top_n >= 2 && has_low_top_no_pair(&new_board.top[..new_board.top_n as usize]) {
            return false;
        }
        // Row ordering: fast pair-dominance check
        // If mid has pair rank > bot's best group rank and bot can't catch up → prune
        if new_board.mid_n >= 2 && new_board.bot_n >= 3 {
            if mid_dominates_bot(&new_board) { return false; }
        }
        true
    }).collect()
}

/// Check if a row has 4+ unpaired cards with no flush or straight draw.
/// Returns true if the row is "scattered" (no draw potential).
fn has_scatter_no_draw(row: &[CardIdx]) -> bool {
    if row.len() < 4 { return false; }

    let mut ranks: Vec<u8> = Vec::new();
    let mut suits: Vec<u8> = Vec::new();
    let mut n_jokers = 0u8;
    for &c in row {
        if c >= 52 {
            n_jokers += 1;
        } else {
            ranks.push(c / 4);
            suits.push(c % 4);
        }
    }

    // Check for pairs (including with joker)
    let mut rc = [0u8; 13];
    for &r in &ranks { rc[r as usize] += 1; }
    let max_count = rc.iter().max().copied().unwrap_or(0) + n_jokers;
    if max_count >= 2 { return false; }  // Has pair or better

    // Check flush draw (3+ same suit, or 2+ with joker)
    let mut sc = [0u8; 4];
    for &s in &suits { sc[s as usize] += 1; }
    let max_suit = sc.iter().max().copied().unwrap_or(0) + n_jokers;
    if max_suit >= 3 { return false; }  // Flush draw

    // Check straight draw (3+ cards within 5-card window, or with jokers)
    for low in 0..=8u8 {  // windows: 2-6, 3-7, ..., T-A
        let high = low + 4;
        let in_window = ranks.iter().filter(|&&r| r >= low && r <= high).count() as u8 + n_jokers;
        if in_window >= 3 { return false; }  // Straight draw
    }
    // A-low straight: A,2,3,4,5 -> ranks 12,0,1,2,3
    let a_low = ranks.iter().filter(|&&r| r == 12 || r <= 3).count() as u8 + n_jokers;
    if a_low >= 3 { return false; }

    true  // No draw found -> scatter
}

/// Check if top row has 2+ cards, all J or lower, no pair.
/// J = rank index 9. Returns true if this poor placement is detected.
fn has_low_top_no_pair(row: &[CardIdx]) -> bool {
    if row.len() < 2 { return false; }

    let mut ranks: Vec<u8> = Vec::new();
    for &c in row {
        if c >= 52 { return false; }  // Joker present → always makes a pair, OK
        ranks.push(c / 4);
    }

    // All cards must be J(9) or lower
    if ranks.iter().any(|&r| r > 9) { return false; }  // Has Q/K/A → fine

    // Check for pair
    let mut rc = [0u8; 13];
    for &r in &ranks { rc[r as usize] += 1; }
    if rc.iter().any(|&c| c >= 2) { return false; }  // Has pair → fine

    true  // All J or lower, no pair → bad top
}

/// Fast check: does mid's confirmed group dominate bot's best possible?
/// Returns true if mid has pair/trips rank X > bot's best group rank Y,
/// and bot has no flush or straight draw to overtake.
fn mid_dominates_bot(board: &Board) -> bool {
    let mid_n = board.mid_n as usize;
    let bot_n = board.bot_n as usize;
    if mid_n < 2 || bot_n < 3 { return false; }

    // Analyze mid: find best group rank
    let mut mid_rc = [0u8; 13];
    let mut mid_jokers = 0u8;
    for i in 0..mid_n {
        let c = board.mid[i];
        if c >= 52 { mid_jokers += 1; } else { mid_rc[(c / 4) as usize] += 1; }
    }
    let mid_best_group = mid_rc.iter().max().copied().unwrap_or(0) + mid_jokers;
    // Find the rank of mid's best group
    let mid_best_rank = if mid_best_group >= 2 {
        mid_rc.iter().enumerate().rev()
            .find(|(_, &c)| c + mid_jokers >= mid_best_group)
            .map(|(i, _)| i as u8).unwrap_or(0)
    } else {
        return false;  // Mid has no pair, nothing to dominate with
    };

    // Analyze bot
    let mut bot_rc = [0u8; 13];
    let mut bot_sc = [0u8; 4];
    let mut bot_jokers = 0u8;
    let mut bot_ranks: [u8; 5] = [0; 5];
    let mut bot_rank_count = 0usize;
    for i in 0..bot_n {
        let c = board.bot[i];
        if c >= 52 { bot_jokers += 1; }
        else {
            let r = c / 4;
            let s = c % 4;
            bot_rc[r as usize] += 1;
            bot_sc[s as usize] += 1;
            if bot_rank_count < 5 { bot_ranks[bot_rank_count] = r; bot_rank_count += 1; }
        }
    }
    let bot_remaining = 5 - bot_n;
    let bot_best_group = bot_rc.iter().max().copied().unwrap_or(0) + bot_jokers;
    let bot_best_rank = bot_rc.iter().enumerate().rev()
        .find(|(_, &c)| c + bot_jokers >= bot_best_group)
        .map(|(i, _)| i as u8).unwrap_or(0);

    // Can bot make a higher category? Check flush/straight draw
    let bot_max_suit = bot_sc.iter().max().copied().unwrap_or(0) + bot_jokers;
    if bot_max_suit + bot_remaining as u8 >= 5 { return false; }  // Flush possible

    // Straight draw check
    for low in 0..=8u8 {
        let high = low + 4;
        let have = bot_ranks[..bot_rank_count].iter()
            .filter(|&&r| r >= low && r <= high).count() as u8 + bot_jokers;
        if have + bot_remaining as u8 >= 5 { return false; }  // Straight possible
    }
    let a_low = bot_ranks[..bot_rank_count].iter()
        .filter(|&&r| r == 12 || r <= 3).count() as u8 + bot_jokers;
    if a_low + bot_remaining as u8 >= 5 { return false; }

    // No flush/straight possible. Compare group strengths.
    // Bot's ceiling group size = bot_best_group + bot_remaining (can add matching cards)
    let bot_ceiling_group = bot_best_group + bot_remaining as u8;

    // If mid and bot are in same group category, compare ranks
    if mid_best_group > bot_ceiling_group {
        return true;  // Mid's category beats bot's ceiling category
    }
    if mid_best_group == bot_ceiling_group {
        // Same category potential - but bot could pull any card rank
        // So bot could potentially get trips/quads of rank > mid's pair rank
        // Only prune if all bot's existing cards are BELOW mid's pair rank
        // AND bot can't overtake by drawing high cards
        if bot_remaining >= 2 { return false; }  // Can draw 2+ cards, could get anything
        // 1 remaining: bot needs to improve with 1 card
        // Bot's best with 1 card = match existing pair → trips
        // If bot's pair rank < mid's pair rank, and bot can only make pairs/trips of that rank
        if bot_best_rank < mid_best_rank {
            return true;  // Bot's best group rank < mid's → mid dominates
        }
    }
    false
}

// ============================================================
//  Pruning: Top-K Action Pre-filter
// ============================================================

/// Pre-filter actions to top-K by cheap 1-sample evaluation.
/// If actions.len() <= k, returns all actions unchanged.
fn topk_filter_actions(
    actions: Vec<TurnAction>,
    board: &Board,
    remaining: &[CardIdx],
    config: &FlEvConfig,
    rng: &mut StdRng,
    k: usize,
) -> Vec<TurnAction> {
    if actions.len() <= k {
        return actions;
    }

    // Score each action with cheap quick_future_eval (1 sample)
    let mut scored: Vec<(usize, f64)> = actions.iter().enumerate().map(|(i, action)| {
        let mut new_board = board.clone();
        for &(card, row) in &action.placements {
            new_board = new_board.place(card, row);
        }
        let val = if new_board.is_complete() {
            evaluate_terminal(&new_board, config)
        } else {
            quick_future_eval(&new_board, remaining, config, rng)
        };
        (i, val)
    }).collect();

    // Sort by EV descending, keep top-K
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored.sort_by_key(|(i, _)| *i); // Restore original order for stability

    scored.into_iter().map(|(i, _)| {
        TurnAction {
            discard: actions[i].discard,
            placements: actions[i].placements,
        }
    }).collect()
}

// ============================================================
//  T2 Deep Evaluation with Pruning
// ============================================================

/// Evaluate a T3 state by sampling n3 deals, then for each deal evaluate
/// all actions (pruned + top-K) with T4 exhaustive.
fn t3_sampling_pruned(
    board: &Board,
    remaining: &[CardIdx],
    config: &FlEvConfig,
    n3: usize,
    top_k: usize,
    exhaustive_t4: bool,
    n4_deep: usize,
    rng: &mut StdRng,
) -> f64 {
    if board.is_complete() {
        return evaluate_terminal(board, config);
    }
    if remaining.len() < 3 {
        return evaluate_terminal(board, config);
    }

    let mut total = 0.0;
    for _ in 0..n3 {
        let deal = sample_3_cards(remaining, rng);
        let rem = remove_dealt(remaining, &deal);

        // Generate actions with bust pruning
        let actions = gen_turn_actions_pruned(&deal, board);
        if actions.is_empty() {
            // All actions bust — fall back to unpruned
            let actions = gen_turn_actions(&deal, board);
            if actions.is_empty() { continue; }
            // Just pick best terminal from unpruned
            let mut best = f64::NEG_INFINITY;
            for action in &actions {
                let mut new_board = board.clone();
                for &(card, row) in &action.placements {
                    new_board = new_board.place(card, row);
                }
                let ev = if exhaustive_t4 {
                    t4_exhaustive_eval(&new_board, &rem, config)
                } else {
                    t4_analytical_eval(&new_board, &rem, config, n4_deep, rng)
                };
                if ev > best { best = ev; }
            }
            total += best;
            continue;
        }

        // Top-K filter
        let actions = topk_filter_actions(actions, board, &rem, config, rng, top_k);

        // Deep eval each remaining action with T4
        let mut best = f64::NEG_INFINITY;
        for action in &actions {
            let mut new_board = board.clone();
            for &(card, row) in &action.placements {
                new_board = new_board.place(card, row);
            }
            let ev = if exhaustive_t4 {
                t4_exhaustive_eval(&new_board, &rem, config)
            } else {
                t4_analytical_eval(&new_board, &rem, config, n4_deep, rng)
            };
            if ev > best { best = ev; }
        }
        total += best;
    }
    total / n3 as f64
}

/// T2 deep evaluation result for one T2 deal.
#[derive(Serialize)]
struct T2ActionResult {
    desc: String,
    ev: f64,
}

/// T2 record: board state + deal + all action EVs
#[derive(Serialize)]
struct T2Record {
    turn: u8,
    top: Vec<String>,
    mid: Vec<String>,
    bot: Vec<String>,
    deal: Vec<String>,
    actions: Vec<T2ActionResult>,
    n_pruned: usize,
    n_total: usize,
}

/// T2 evaluation result for one pattern
#[derive(Serialize)]
struct T2EvalResult {
    t0_hand: Vec<String>,
    t2_records: Vec<T2Record>,
    t3_records: Vec<T3Record>,
    elapsed_ms: u64,
}

/// Run T2-deep evaluation for one pattern.
/// T0: quick eval best action. T1: sample + quick_pick_best.
/// T2: sample n2 deals, DEEP eval ALL actions (pruned + top-K) with T3+T4.
/// Also collects T3 records along the way.
fn t2_deep_eval_pattern(
    t0_hand: &[CardIdx; 5],
    config: &FlEvConfig,
    n1: usize,
    n2: usize,
    n3_deep: usize,
    n_quick: usize,
    top_k: usize,
    seed: u64,
    exhaustive_t4: bool,
    n4_deep: usize,
) -> T2EvalResult {
    let start = std::time::Instant::now();

    // Find best T0 action using quick eval (same as t3_eval_pattern)
    let t0_actions = gen_t0_actions_pruned(t0_hand, &Board::new());
    let remaining_after_t0: Vec<CardIdx> = (0..54u8)
        .filter(|c| !t0_hand.contains(c))
        .collect();

    let t0_evs: Vec<(usize, f64)> = t0_actions.par_iter().enumerate()
        .map(|(i, action)| {
            let mut board = Board::new();
            for &(card, row) in action {
                board = board.place(card, row);
            }
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut total = 0.0;
            for _ in 0..3 {
                total += quick_future_eval(&board, &remaining_after_t0, config, &mut rng);
            }
            (i, total / 3.0)
        })
        .collect();

    let best_t0_idx = t0_evs.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|x| x.0)
        .unwrap_or(0);

    let best_t0 = &t0_actions[best_t0_idx];
    let mut board_t0 = Board::new();
    for &(card, row) in best_t0 {
        board_t0 = board_t0.place(card, row);
    }

    let mut all_t2_records: Vec<T2Record> = Vec::new();
    let mut all_t3_records: Vec<T3Record> = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));

    for _ in 0..n1 {
        // T1: sample deal, quick pick best
        if remaining_after_t0.len() < 3 { break; }
        let deal_t1 = sample_3_cards(&remaining_after_t0, &mut rng);
        let rem_t1 = remove_dealt(&remaining_after_t0, &deal_t1);
        let best_t1 = match quick_pick_best(&board_t0, &deal_t1, &rem_t1, config, &mut rng, n_quick) {
            Some(a) => a,
            None => continue,
        };
        let mut board_t1 = board_t0.clone();
        for &(card, row) in &best_t1.placements {
            board_t1 = board_t1.place(card, row);
        }

        for _ in 0..n2 {
            // T2: sample deal, DEEP eval ALL actions
            if rem_t1.len() < 3 { break; }
            let deal_t2 = sample_3_cards(&rem_t1, &mut rng);
            let rem_t2 = remove_dealt(&rem_t1, &deal_t2);

            // Generate actions with bust pruning
            let all_actions = gen_turn_actions(&deal_t2, &board_t1);
            let pruned_actions = gen_turn_actions_pruned(&deal_t2, &board_t1);
            let n_total = all_actions.len();
            let n_after_bust = pruned_actions.len();

            let actions = if pruned_actions.is_empty() {
                all_actions  // Fall back to all if everything busts
            } else {
                // Top-K filter
                topk_filter_actions(pruned_actions, &board_t1, &rem_t2, config, &mut rng, top_k)
            };

            if actions.is_empty() { continue; }

            // Deep eval each T2 action using T3 sampling + T4 exhaustive
        // Parallelize T2 deep eval across all T2 actions for the current T1 board
        let t2_action_results: Vec<T2ActionResult> = actions.par_iter().map(|action| {
            let mut board_t2 = board_t1.clone();
            for &(card, row) in &action.placements {
                board_t2 = board_t2.place(card, row);
            }
            // Need a thread-local RNG for the inner T3 sampling
            let mut local_rng = StdRng::seed_from_u64(seed.wrapping_add(action.placements[0].0 as u64));
            let ev = t3_sampling_pruned(
                &board_t2, &rem_t2, config, n3_deep, top_k,
                exhaustive_t4, n4_deep, &mut local_rng,
            );
            T2ActionResult {
                desc: format_turn_action(action),
                ev,
            }
        }).collect();

        // Sort by EV desc in a separate mutable copy
        let mut t2_action_results_sorted = t2_action_results;
        t2_action_results_sorted.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

            all_t2_records.push(T2Record {
                turn: 2,
                top: board_t1.top[..board_t1.top_n as usize]
                    .iter().map(|&c| cardidx_to_string(c)).collect(),
                mid: board_t1.mid[..board_t1.mid_n as usize]
                    .iter().map(|&c| cardidx_to_string(c)).collect(),
                bot: board_t1.bot[..board_t1.bot_n as usize]
                    .iter().map(|&c| cardidx_to_string(c)).collect(),
                deal: deal_t2.iter().map(|&c| cardidx_to_string(c)).collect(),
                actions: t2_action_results_sorted,
                n_pruned: n_total - n_after_bust,
                n_total,
            });

            // Also collect T3 records from the best T2 action
            // (Re-evaluate best T2 action's subtree to get T3 data)
            if !actions.is_empty() {
                // Find best T2 action (the one with highest EV from t2_action_results)
                let best_t2_desc = &all_t2_records.last().unwrap().actions[0].desc;
                let best_action = actions.iter()
                    .find(|a| format_turn_action(a) == *best_t2_desc)
                    .unwrap_or(&actions[0]);

                let mut board_t2 = board_t1.clone();
                for &(card, row) in &best_action.placements {
                    board_t2 = board_t2.place(card, row);
                }

                // Collect T3 records from this T2 board state
                let n3_record = 3.min(n3_deep); // A few T3 samples for records
                for _ in 0..n3_record {
                    if rem_t2.len() < 3 { break; }
                    let deal_t3 = sample_3_cards(&rem_t2, &mut rng);
                    let rem_t3 = remove_dealt(&rem_t2, &deal_t3);
                    let t3_actions_raw = gen_turn_actions_pruned(&deal_t3, &board_t2);
                    let t3_actions = if t3_actions_raw.is_empty() {
                        gen_turn_actions(&deal_t3, &board_t2)
                    } else {
                        t3_actions_raw
                    };
                    if t3_actions.is_empty() { continue; }

                    // Parallelize T4 exhaustive evaluations across T3 actions
                    let mut turn_results: Vec<TurnActionResult> = t3_actions.par_iter().enumerate().map(|(i, action)| {
                        let mut board_t3 = board_t2.clone();
                        for &(card, row) in &action.placements {
                            board_t3 = board_t3.place(card, row);
                        }
                        let mut local_rng = StdRng::seed_from_u64(seed.wrapping_add((i * 10) as u64));
                        let ev = if exhaustive_t4 {
                            t4_exhaustive_eval(&board_t3, &rem_t3, config)
                        } else {
                            t4_analytical_eval(&board_t3, &rem_t3, config, n4_deep, &mut local_rng)
                        };
                        TurnActionResult { desc: format_turn_action(action), ev }
                    }).collect();
                    turn_results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

                    all_t3_records.push(T3Record {
                        turn: 3,
                        top: board_t2.top[..board_t2.top_n as usize]
                            .iter().map(|&c| cardidx_to_string(c)).collect(),
                        mid: board_t2.mid[..board_t2.mid_n as usize]
                            .iter().map(|&c| cardidx_to_string(c)).collect(),
                        bot: board_t2.bot[..board_t2.bot_n as usize]
                            .iter().map(|&c| cardidx_to_string(c)).collect(),
                        deal: deal_t3.iter().map(|&c| cardidx_to_string(c)).collect(),
                        actions: turn_results,
                    });
                }
            }
        }
    }

    let elapsed = start.elapsed().as_millis() as u64;
    T2EvalResult {
        t0_hand: t0_hand.iter().map(|&c| cardidx_to_string(c)).collect(),
        t2_records: all_t2_records,
        t3_records: all_t3_records,
        elapsed_ms: elapsed,
    }
}

/// Run T2 deep evaluation: CLI entry point.
fn run_t2_eval(args: &[String]) {
    let mut n1 = 50usize;
    let mut n2 = 100usize;
    let mut n3_deep = 30usize;
    let mut n4_deep = 20usize;
    let mut n_quick = 3usize;
    let mut top_k = 4usize;
    let mut seed = 42u64;
    let mut patterns_file = String::new();
    let mut exhaustive_t4 = false;
    let mut bust_penalty = -6.0f64;
    let mut fl_ev_14 = 14.0f64;
    let mut fl_ev_15 = 27.9f64;
    let mut fl_ev_16 = 52.4f64;
    let mut fl_ev_17 = 104.5f64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n1" => { i += 1; n1 = args[i].parse().unwrap(); }
            "--n2" => { i += 1; n2 = args[i].parse().unwrap(); }
            "--n3-deep" | "--n3" => { i += 1; n3_deep = args[i].parse().unwrap(); }
            "--n4" => { i += 1; n4_deep = args[i].parse().unwrap(); }
            "--n-quick" => { i += 1; n_quick = args[i].parse().unwrap(); }
            "--top-k" => { i += 1; top_k = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--patterns" => { i += 1; patterns_file = args[i].clone(); }
            "--exhaustive" => { exhaustive_t4 = true; }
            "--bust-penalty" => { i += 1; bust_penalty = args[i].parse().unwrap(); }
            "--fl-ev-14" => { i += 1; fl_ev_14 = args[i].parse().unwrap(); }
            "--fl-ev-15" => { i += 1; fl_ev_15 = args[i].parse().unwrap(); }
            "--fl-ev-16" => { i += 1; fl_ev_16 = args[i].parse().unwrap(); }
            "--fl-ev-17" => { i += 1; fl_ev_17 = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }

    let config = FlEvConfig {
        bust_penalty,
        fl_ev: [(14u8, fl_ev_14), (15, fl_ev_15), (16, fl_ev_16), (17, fl_ev_17)]
            .iter().cloned().collect(),
    };

    eprintln!("=== T2-Deep Evaluation with Pruning ===");
    if exhaustive_t4 {
        eprintln!("  T4 Mode: EXHAUSTIVE (all C(remaining,3) deals)");
    }
    eprintln!("  N1={}, N2={}, N3_deep={}, top_K={}", n1, n2, n3_deep, top_k);
    eprintln!("  N_quick={}, bust_penalty={}", n_quick, bust_penalty);
    eprintln!("  FL EV: 14={}, 15={}, 16={}, 17={}", fl_ev_14, fl_ev_15, fl_ev_16, fl_ev_17);
    eprintln!("  T2 deep states/pattern = {} × {}", n1, n2);
    eprintln!("  Each T2 action: n3={} T3 samples × T4 {}",
        n3_deep, if exhaustive_t4 { "exhaustive" } else { "sampling" });

    // Read patterns from file or stdin
    let patterns: Vec<[CardIdx; 5]> = if !patterns_file.is_empty() {
        let content = std::fs::read_to_string(&patterns_file).unwrap();
        content.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                let cards: Vec<CardIdx> = l.split(',')
                    .filter_map(|s| parse_card_str(s))
                    .collect();
                if cards.len() == 5 {
                    let mut arr = [0u8; 5];
                    arr.copy_from_slice(&cards);
                    Some(arr)
                } else { None }
            })
            .collect()
    } else {
        let stdin = io::stdin();
        stdin.lock().lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                let cards: Vec<CardIdx> = l.split(',')
                    .filter_map(|s| parse_card_str(s))
                    .collect();
                if cards.len() == 5 {
                    let mut arr = [0u8; 5];
                    arr.copy_from_slice(&cards);
                    Some(arr)
                } else { None }
            })
            .collect()
    };

    eprintln!("  Patterns: {}", patterns.len());
    let total_start = std::time::Instant::now();

    // Process patterns in parallel using rayon
    let results: Vec<T2EvalResult> = patterns.par_iter().enumerate()
        .map(|(idx, pattern)| {
            let result = t2_deep_eval_pattern(
                pattern, &config, n1, n2, n3_deep, n_quick, top_k,
                seed.wrapping_add(idx as u64 * 100000),
                exhaustive_t4, n4_deep,
            );
            if (idx + 1) % 10 == 0 || idx + 1 == patterns.len() {
                eprintln!("  [{}/{}] T2={} T3={} records in {}ms",
                    idx + 1, patterns.len(),
                    result.t2_records.len(),
                    result.t3_records.len(),
                    result.elapsed_ms);
            }
            result
        })
        .collect();

    // Output all results as JSON (one per line)
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    for result in &results {
        writeln!(stdout, "{}", serde_json::to_string(result).unwrap()).unwrap();
    }
    stdout.flush().unwrap();

    let total_elapsed = total_start.elapsed();
    let total_t2: usize = results.iter().map(|r| r.t2_records.len()).sum();
    let total_t3: usize = results.iter().map(|r| r.t3_records.len()).sum();
    let avg_pruned: f64 = if total_t2 > 0 {
        results.iter()
            .flat_map(|r| r.t2_records.iter())
            .map(|r| r.n_pruned as f64)
            .sum::<f64>() / total_t2 as f64
    } else { 0.0 };

    eprintln!("\n=== T2 Deep Evaluation Complete ===");
    eprintln!("  Total patterns: {}", patterns.len());
    eprintln!("  Total T2 records: {}", total_t2);
    eprintln!("  Total T3 records: {}", total_t3);
    eprintln!("  Avg bust-pruned actions per T2 deal: {:.1}", avg_pruned);
    eprintln!("  Total time: {:.1}s", total_elapsed.as_secs_f64());
    eprintln!("  Avg per pattern: {:.1}s",
        total_elapsed.as_secs_f64() / patterns.len().max(1) as f64);
}

// ============================================================
//  Suit Isomorphism & Pattern Enumeration
// ============================================================

const SUIT_PERMS: [[u8; 4]; 24] = [
    [0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],
    [1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],[1,3,0,2],[1,3,2,0],
    [2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],
    [3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0],
];

fn permute_card(card: CardIdx, perm: &[u8; 4]) -> CardIdx {
    if card >= 52 { return card; }
    let rank = card / 4;
    let suit = card % 4;
    rank * 4 + perm[suit as usize]
}

fn canonicalize(hand: &[CardIdx; 5]) -> [CardIdx; 5] {
    let mut best = *hand;
    best.sort();
    for perm in &SUIT_PERMS {
        let mut p = [0u8; 5];
        for i in 0..5 { p[i] = permute_card(hand[i], perm); }
        p.sort();
        if p < best { best = p; }

        // Joker swap (X1↔X2)
        let mut p2 = [0u8; 5];
        for i in 0..5 {
            let c = permute_card(hand[i], perm);
            p2[i] = if c == JOKER1 { JOKER2 } else if c == JOKER2 { JOKER1 } else { c };
        }
        p2.sort();
        if p2 < best { best = p2; }
    }
    best
}

fn run_enumerate() {
    let start = std::time::Instant::now();
    let mut canonical_set: std::collections::HashSet<[CardIdx; 5]> = std::collections::HashSet::new();

    let n = 54u8;
    let mut count_total = 0u64;
    for a in 0..n {
        for b in (a+1)..n {
            for c in (b+1)..n {
                for d in (c+1)..n {
                    for e in (d+1)..n {
                        count_total += 1;
                        let hand = [a, b, c, d, e];
                        let canon = canonicalize(&hand);
                        canonical_set.insert(canon);
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!("Total combinations: {}", count_total);
    eprintln!("Canonical patterns: {}", canonical_set.len());
    eprintln!("Compression ratio: {:.1}x", count_total as f64 / canonical_set.len() as f64);
    eprintln!("Elapsed: {:.2}s", elapsed.as_secs_f64());

    let mut patterns: Vec<[CardIdx; 5]> = canonical_set.into_iter().collect();
    patterns.sort();
    println!("{}", patterns.len());
    for pat in &patterns {
        let s: Vec<String> = pat.iter().map(|&c| cardidx_to_string(c)).collect();
        println!("{}", s.join(","));
    }
}
