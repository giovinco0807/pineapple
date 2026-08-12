use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const B: i64 = 15;
const B5: i64 = 759_375;
const ROWS: [&str; 3] = ["top", "middle", "bottom"];
const FULL_DECK_CARDS: [&str; 54] = [
    "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "Th", "Jh", "Qh", "Kh", "Ah", "2d", "3d", "4d",
    "5d", "6d", "7d", "8d", "9d", "Td", "Jd", "Qd", "Kd", "Ad", "2c", "3c", "4c", "5c", "6c", "7c",
    "8c", "9c", "Tc", "Jc", "Qc", "Kc", "Ac", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "Ts",
    "Js", "Qs", "Ks", "As", "X1", "X2",
];

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Exact T3/T4 evaluator for OFC Pineapple tutor positions"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    limit: usize,

    #[arg(long, default_value_t = 0)]
    skip: usize,

    #[arg(long, default_value_t = 20)]
    top_n: usize,

    #[arg(long, default_value_t = 0)]
    t2_draw_limit: usize,

    #[arg(long, default_value_t = 0)]
    source_candidate_top_k: usize,

    #[arg(long, default_value_t = false)]
    position_parallel: bool,

    #[arg(long, default_value = "ai/config/fl_ev.json")]
    fl_config: PathBuf,

    #[arg(long, default_value_t = false)]
    summary_only: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct BoardPayload {
    #[serde(default)]
    top: Vec<String>,
    #[serde(default, alias = "mid")]
    middle: Vec<String>,
    #[serde(default, alias = "bot")]
    bottom: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct PositionPayload {
    #[serde(default)]
    board: BoardPayload,
    #[serde(default)]
    opponent_board: BoardPayload,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    known_discards: Vec<String>,
    #[serde(default)]
    dealt: Vec<String>,
    #[serde(default)]
    turn: Option<u8>,
    #[serde(default)]
    candidate_actions: Vec<ActionPayload>,
    #[serde(default)]
    candidates: Vec<SourceCandidatePayload>,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceCandidatePayload {
    #[serde(default)]
    action: Option<ActionPayload>,
    #[serde(default)]
    placements: Vec<(String, String)>,
    #[serde(default)]
    discard: String,
}

impl SourceCandidatePayload {
    fn action_payload(&self) -> Option<ActionPayload> {
        if let Some(action) = &self.action {
            return Some(action.clone());
        }
        if self.placements.is_empty() || self.discard.is_empty() {
            return None;
        }
        Some(ActionPayload {
            placements: self.placements.clone(),
            discard: self.discard.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct PositionResult {
    record_index: usize,
    turn: u8,
    legal_actions: usize,
    evaluated_actions: usize,
    requested_actions: usize,
    elapsed_ms: f64,
    best: Option<CandidateResult>,
    candidates: Vec<CandidateResult>,
}

#[derive(Clone, Debug, Serialize)]
struct CandidateResult {
    action: ActionPayload,
    board: BoardPayload,
    metrics: Metrics,
    row_eval_cache: RowEvalStats,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ActionPayload {
    placements: Vec<(String, String)>,
    discard: String,
}

#[derive(Clone, Debug, Serialize)]
struct Metrics {
    score: f64,
    ev: f64,
    raw_score: f64,
    royalty: f64,
    bust_rate: f64,
    fl_rate: f64,
    fl_type_rates: BTreeMap<String, f64>,
    samples: usize,
    source: &'static str,
    forced_bust: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enumerated_draws: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remaining_deck_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_turn: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opponent_response: Option<bool>,
}

#[derive(Clone, Debug, Default)]
struct Board {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
}

#[derive(Clone, Debug)]
struct Action {
    placements: Vec<(String, usize)>,
    discard: String,
}

#[derive(Clone, Debug, Default)]
struct Totals {
    score: f64,
    raw_score: f64,
    royalty: f64,
    bust: f64,
    fl_any: f64,
    fl_qq: f64,
    fl_kk: f64,
    fl_aa: f64,
    fl_trips: f64,
}

#[derive(Clone, Debug)]
struct Terminal {
    score: f64,
    raw_score: f64,
    royalty: f64,
    bust: bool,
    fl_type: Option<u8>,
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn compare_terminal_quality(a: &Terminal, b: &Terminal) -> Ordering {
    cmp_f64(a.score, b.score)
        .then_with(|| (!a.bust).cmp(&(!b.bust)))
        .then_with(|| a.fl_type.unwrap_or(0).cmp(&b.fl_type.unwrap_or(0)))
        .then_with(|| cmp_f64(a.raw_score, b.raw_score))
        .then_with(|| cmp_f64(a.royalty, b.royalty))
}

fn compare_metrics_quality(a: &Metrics, b: &Metrics) -> Ordering {
    cmp_f64(a.score, b.score)
        .then_with(|| cmp_f64(b.bust_rate, a.bust_rate))
        .then_with(|| cmp_f64(a.fl_rate, b.fl_rate))
        .then_with(|| cmp_f64(a.raw_score, b.raw_score))
        .then_with(|| cmp_f64(a.royalty, b.royalty))
}

#[derive(Clone, Debug)]
struct FlEv {
    values: BTreeMap<u8, f64>,
}

#[derive(Clone, Debug)]
struct BoardEval {
    vals: [i64; 3],
    complete: bool,
    busted: bool,
    royalty: i32,
    fl_key: u8,
    fl_type: Option<u8>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RowEvalStats {
    entries: usize,
    hits: usize,
    misses: usize,
    board_entries: usize,
    board_hits: usize,
    board_misses: usize,
}

#[derive(Debug, Default)]
struct RowEvalCache {
    hand_values: HashMap<u64, i64>,
    board_evals: HashMap<BoardEvalKey, BoardEval>,
    constrained_rows: HashMap<u64, Vec<String>>,
    hits: usize,
    misses: usize,
    board_hits: usize,
    board_misses: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BoardEvalKey {
    row_lens: [u8; 3],
    codes: [u8; 13],
}

#[derive(Debug)]
struct EvalOutcome {
    metrics: Metrics,
    row_eval_cache: RowEvalStats,
}

impl RowEvalCache {
    fn evaluate_hand(&mut self, cards: &[String], expected_count: usize) -> i64 {
        let key = hand_eval_key(cards, expected_count);
        if let Some(value) = self.hand_values.get(&key).copied() {
            self.hits += 1;
            return value;
        }
        self.misses += 1;
        let value = evaluate_hand(cards, expected_count);
        self.hand_values.insert(key, value);
        value
    }

    fn stats(&self) -> RowEvalStats {
        RowEvalStats {
            entries: self.hand_values.len(),
            hits: self.hits,
            misses: self.misses,
            board_entries: self.board_evals.len(),
            board_hits: self.board_hits,
            board_misses: self.board_misses,
        }
    }
}

impl Board {
    fn from_payload(payload: &BoardPayload) -> Self {
        Self {
            top: normalize_cards(&payload.top),
            middle: normalize_cards(&payload.middle),
            bottom: normalize_cards(&payload.bottom),
        }
    }

    fn to_payload(&self) -> BoardPayload {
        BoardPayload {
            top: self.top.clone(),
            middle: self.middle.clone(),
            bottom: self.bottom.clone(),
        }
    }

    fn all_cards(&self) -> impl Iterator<Item = &String> {
        self.top
            .iter()
            .chain(self.middle.iter())
            .chain(self.bottom.iter())
    }

    fn is_complete(&self) -> bool {
        self.top.len() == 3 && self.middle.len() == 5 && self.bottom.len() == 5
    }
}

fn normalize_card(card: &str, joker_count: &mut usize) -> Option<String> {
    if card.is_empty() {
        return None;
    }
    if card == "JK" || card == "Xj" {
        *joker_count += 1;
        return Some(if *joker_count == 1 { "X1" } else { "X2" }.to_string());
    }
    Some(card.to_string())
}

fn normalize_cards(cards: &[String]) -> Vec<String> {
    let mut jokers = 0usize;
    cards
        .iter()
        .filter_map(|c| normalize_card(c, &mut jokers))
        .collect()
}

fn normalize_unique_cards(cards: &[String]) -> Vec<String> {
    let mut out = normalize_cards(cards);
    out.sort();
    out.dedup();
    out
}

fn canonical_exclude_cards(
    cards: &[String],
    board: &Board,
    opponent: &Board,
    dealt: &[String],
) -> Vec<String> {
    let mut live: HashSet<String> = HashSet::new();
    live.extend(board.all_cards().cloned());
    live.extend(opponent.all_cards().cloned());
    live.extend(dealt.iter().cloned());
    normalize_unique_cards(cards)
        .into_iter()
        .filter(|card| !live.contains(card))
        .collect()
}

fn append_exclude_card(exclude: &[String], card: &str) -> Vec<String> {
    let mut out = exclude.to_vec();
    if !out.iter().any(|existing| existing == card) {
        out.push(card.to_string());
    }
    out
}

fn row_mut<'a>(board: &'a mut Board, row: usize) -> &'a mut Vec<String> {
    match row {
        0 => &mut board.top,
        1 => &mut board.middle,
        2 => &mut board.bottom,
        _ => unreachable!(),
    }
}

fn apply_action(board: &Board, action: &Action) -> Board {
    let mut out = board.clone();
    for (card, row_idx) in &action.placements {
        row_mut(&mut out, *row_idx).push(card.clone());
    }
    out
}

fn action_to_payload(action: &Action) -> ActionPayload {
    ActionPayload {
        placements: action
            .placements
            .iter()
            .map(|(card, row_idx)| (card.clone(), ROWS[*row_idx].to_string()))
            .collect(),
        discard: action.discard.clone(),
    }
}

fn get_turn_actions(dealt: &[String], board: &Board) -> Vec<Action> {
    assert_eq!(dealt.len(), 3);
    let mut cards = dealt.to_vec();
    cards.sort();
    let limits = [3usize, 5, 5];
    let lens = [board.top.len(), board.middle.len(), board.bottom.len()];
    let mut seen = BTreeSet::new();
    let mut actions = Vec::new();

    for discard_idx in 0..3 {
        let discard = cards[discard_idx].clone();
        let remaining: Vec<String> = (0..3)
            .filter(|i| *i != discard_idx)
            .map(|i| cards[i].clone())
            .collect();
        for pos0 in 0..3 {
            for pos1 in 0..3 {
                let mut counts = lens;
                counts[pos0] += 1;
                if counts[pos0] > limits[pos0] {
                    continue;
                }
                counts[pos1] += 1;
                if counts[pos1] > limits[pos1] {
                    continue;
                }
                let placements = vec![(remaining[0].clone(), pos0), (remaining[1].clone(), pos1)];
                let key = canonical_action_key(&discard, &placements);
                if seen.insert(key) {
                    actions.push(Action {
                        placements,
                        discard: discard.clone(),
                    });
                }
            }
        }
    }
    actions
}

fn canonical_action_key(discard: &str, placements: &[(String, usize)]) -> String {
    let mut by_row: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (card, row_idx) in placements {
        by_row[*row_idx].push(card.clone());
    }
    let mut parts = vec![format!("d={discard}")];
    for (idx, cards) in by_row.iter_mut().enumerate() {
        cards.sort();
        if !cards.is_empty() {
            parts.push(format!("{}={}", ROWS[idx], cards.join(",")));
        }
    }
    parts.join("|")
}

fn action_payload_key(action: &ActionPayload) -> String {
    let placements: Vec<(String, usize)> = action
        .placements
        .iter()
        .filter_map(|(card, row)| row_index(row).map(|row_idx| (card.clone(), row_idx)))
        .collect();
    canonical_action_key(&action.discard, &placements)
}

fn joker_wildcard_card(card: &str) -> String {
    if card == "X1" || card == "X2" || card == "JK" || card == "Xj" {
        "Xj".to_string()
    } else {
        card.to_string()
    }
}

fn action_wildcard_key(action: &Action) -> String {
    let placements: Vec<(String, usize)> = action
        .placements
        .iter()
        .map(|(card, row_idx)| (joker_wildcard_card(card), *row_idx))
        .collect();
    canonical_action_key(&joker_wildcard_card(&action.discard), &placements)
}

fn action_payload_wildcard_key(action: &ActionPayload) -> Option<String> {
    let placements: Vec<(String, usize)> = action
        .placements
        .iter()
        .filter_map(|(card, row)| {
            row_index(row).map(|row_idx| (joker_wildcard_card(card), row_idx))
        })
        .collect();
    if placements.len() != action.placements.len() {
        return None;
    }
    Some(canonical_action_key(
        &joker_wildcard_card(&action.discard),
        &placements,
    ))
}

fn row_index(row: &str) -> Option<usize> {
    match row {
        "top" => Some(0),
        "middle" | "mid" => Some(1),
        "bottom" | "bot" => Some(2),
        _ => None,
    }
}

fn action_from_payload(payload: &ActionPayload) -> Option<Action> {
    let mut joker_count = 0usize;
    let discard = normalize_card(&payload.discard, &mut joker_count)?;
    let mut placements = Vec::new();
    for (card, row) in &payload.placements {
        let normalized = normalize_card(card, &mut joker_count)?;
        let row_idx = row_index(row)?;
        placements.push((normalized, row_idx));
    }
    Some(Action {
        placements,
        discard,
    })
}

fn requested_action_subset(
    payload_actions: &[ActionPayload],
    legal_actions: &[Action],
) -> Vec<Action> {
    if payload_actions.is_empty() {
        return legal_actions.to_vec();
    }
    let legal_by_key: HashMap<String, Action> = legal_actions
        .iter()
        .map(|action| {
            (
                canonical_action_key(&action.discard, &action.placements),
                action.clone(),
            )
        })
        .collect();
    let legal_by_wildcard_key: HashMap<String, Action> = legal_actions
        .iter()
        .map(|action| (action_wildcard_key(action), action.clone()))
        .collect();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for payload_action in payload_actions {
        let mut matched = None;
        if let Some(action) = action_from_payload(payload_action) {
            let key = canonical_action_key(&action.discard, &action.placements);
            matched = legal_by_key.get(&key).cloned();
        }
        if matched.is_none() {
            if let Some(wildcard_key) = action_payload_wildcard_key(payload_action) {
                matched = legal_by_wildcard_key.get(&wildcard_key).cloned();
            }
        }
        let Some(legal) = matched else {
            continue;
        };
        let key = canonical_action_key(&legal.discard, &legal.placements);
        if !seen.insert(key.clone()) {
            continue;
        }
        out.push(legal);
    }
    out
}

fn remaining_deck(
    board: &Board,
    opponent: &Board,
    dealt: &[String],
    exclude: &[String],
) -> Vec<String> {
    let mut used: HashSet<&str> = HashSet::new();
    used.extend(board.all_cards().map(String::as_str));
    used.extend(opponent.all_cards().map(String::as_str));
    used.extend(dealt.iter().map(String::as_str));
    used.extend(exclude.iter().map(String::as_str));
    FULL_DECK_CARDS
        .iter()
        .filter(|card| !used.contains(**card))
        .map(|card| (*card).to_string())
        .collect()
}

fn rank_value(card: &str) -> Option<u8> {
    if is_joker(card) {
        return None;
    }
    match card.as_bytes().first().copied()? as char {
        '2' => Some(2),
        '3' => Some(3),
        '4' => Some(4),
        '5' => Some(5),
        '6' => Some(6),
        '7' => Some(7),
        '8' => Some(8),
        '9' => Some(9),
        'T' => Some(10),
        'J' => Some(11),
        'Q' => Some(12),
        'K' => Some(13),
        'A' => Some(14),
        _ => None,
    }
}

fn suit_value(card: &str) -> Option<char> {
    if is_joker(card) {
        return None;
    }
    card.chars().nth(1)
}

fn is_joker(card: &str) -> bool {
    card == "X1" || card == "X2" || card == "JK"
}

fn card_code(card: &str) -> u8 {
    if card == "X1" {
        return 52;
    }
    if card == "X2" || card == "JK" {
        return 53;
    }
    let bytes = card.as_bytes();
    if bytes.len() < 2 {
        return 63;
    }
    let rank = match bytes[0] as char {
        '2' => 0,
        '3' => 1,
        '4' => 2,
        '5' => 3,
        '6' => 4,
        '7' => 5,
        '8' => 6,
        '9' => 7,
        'T' => 8,
        'J' => 9,
        'Q' => 10,
        'K' => 11,
        'A' => 12,
        _ => 15,
    };
    let suit = match bytes[1] as char {
        'h' => 0,
        'd' => 1,
        'c' => 2,
        's' => 3,
        _ => 3,
    };
    (rank * 4 + suit).min(63)
}

fn hand_eval_key_from_codes(mut codes: [u8; 5], len: usize, expected_count: usize) -> u64 {
    codes[..len].sort_unstable();
    let mut key = ((expected_count as u64) << 4) | len as u64;
    for code in &codes[..len] {
        key = (key << 6) | (*code as u64 + 1);
    }
    key
}

fn hand_eval_key(cards: &[String], expected_count: usize) -> u64 {
    let mut codes = [0u8; 5];
    for (idx, card) in cards.iter().enumerate() {
        codes[idx] = card_code(card);
    }
    hand_eval_key_from_codes(codes, cards.len(), expected_count)
}

fn board_eval_key(board: &Board) -> BoardEvalKey {
    let rows = [&board.top, &board.middle, &board.bottom];
    let mut row_lens = [0u8; 3];
    let mut codes = [0u8; 13];
    let mut offset = 0usize;
    for (row_idx, row) in rows.iter().enumerate() {
        row_lens[row_idx] = row.len() as u8;
        let end = offset + row.len();
        for (idx, card) in row.iter().enumerate() {
            codes[offset + idx] = card_code(card).saturating_add(1);
        }
        codes[offset..end].sort_unstable();
        offset = end;
    }
    BoardEvalKey { row_lens, codes }
}

fn encode_hand(cat: i64, ranks: &[i64]) -> i64 {
    let mut val = cat;
    for i in 0..5 {
        val = val * B + ranks.get(i).copied().unwrap_or(0);
    }
    val
}

fn hand_category(val: i64) -> i64 {
    val / B5
}

fn check_straight(sorted_ranks: &[u8], jokers: usize) -> bool {
    if sorted_ranks.len() + jokers < 5 {
        return false;
    }
    let unique: BTreeSet<u8> = sorted_ranks.iter().copied().collect();
    for high in (5..=14).rev() {
        let needed: BTreeSet<u8> = if high == 5 {
            [14u8, 5, 4, 3, 2].into_iter().collect()
        } else {
            ((high - 4)..=high).collect()
        };
        let missing = needed.difference(&unique).count();
        if missing <= jokers {
            return true;
        }
    }
    false
}

fn straight_high(sorted_ranks: &[u8], jokers: usize) -> u8 {
    if jokers == 0 {
        if sorted_ranks == [14, 5, 4, 3, 2] {
            return 5;
        }
        return sorted_ranks.first().copied().unwrap_or(0);
    }
    let unique: BTreeSet<u8> = sorted_ranks.iter().copied().collect();
    for high in (5..=14).rev() {
        let needed: BTreeSet<u8> = if high == 5 {
            [14u8, 5, 4, 3, 2].into_iter().collect()
        } else {
            ((high - 4)..=high).collect()
        };
        let missing = needed.difference(&unique).count();
        let extra = unique.difference(&needed).count();
        if missing <= jokers && extra == 0 {
            return high;
        }
    }
    sorted_ranks.first().copied().unwrap_or(0)
}

fn rank_counts(ranks: &[u8]) -> BTreeMap<u8, usize> {
    let mut counts = BTreeMap::new();
    for rank in ranks {
        *counts.entry(*rank).or_insert(0) += 1;
    }
    counts
}

fn row_has_joker(cards: &[String]) -> bool {
    cards.iter().any(|card| is_joker(card))
}

fn available_subs_for_row(cards: &[String]) -> Vec<String> {
    let used: HashSet<&str> = cards
        .iter()
        .filter(|card| !is_joker(card))
        .map(String::as_str)
        .collect();
    FULL_DECK_CARDS
        .iter()
        .filter(|card| **card != "X1" && **card != "X2" && !used.contains(**card))
        .map(|card| (*card).to_string())
        .collect()
}

fn constrained_row_key(cards: &[String], ref_val: i64, expected_count: usize) -> u64 {
    hand_eval_key(cards, expected_count).wrapping_mul(1_099_511_628_211) ^ ref_val as u64
}

fn constrain_row_to_value(
    cards: &[String],
    ref_val: i64,
    expected_count: usize,
    cache: &mut RowEvalCache,
) -> Vec<String> {
    let key = constrained_row_key(cards, ref_val, expected_count);
    if let Some(row) = cache.constrained_rows.get(&key).cloned() {
        return row;
    }

    let value = cache.evaluate_hand(cards, expected_count);
    if value <= ref_val {
        let row = cards.to_vec();
        cache.constrained_rows.insert(key, row.clone());
        return row;
    }

    let non_jokers: Vec<String> = cards
        .iter()
        .filter(|card| !is_joker(card))
        .cloned()
        .collect();
    let n_jokers = cards.len().saturating_sub(non_jokers.len());
    let subs = available_subs_for_row(cards);
    let mut best: Option<(Vec<String>, i64)> = None;

    if n_jokers == 1 {
        for sub in &subs {
            let mut test = non_jokers.clone();
            test.push(sub.clone());
            let test_value = cache.evaluate_hand(&test, expected_count);
            if test_value <= ref_val
                && best
                    .as_ref()
                    .map_or(true, |(_, best_value)| test_value > *best_value)
            {
                best = Some((test, test_value));
            }
        }
    } else if n_jokers == 2 {
        for i in 0..subs.len() {
            for j in (i + 1)..subs.len() {
                let mut test = non_jokers.clone();
                test.push(subs[i].clone());
                test.push(subs[j].clone());
                let test_value = cache.evaluate_hand(&test, expected_count);
                if test_value <= ref_val
                    && best
                        .as_ref()
                        .map_or(true, |(_, best_value)| test_value > *best_value)
                {
                    best = Some((test, test_value));
                }
            }
        }
    }

    let row = best
        .map(|(cards, _)| cards)
        .unwrap_or_else(|| cards.to_vec());
    cache.constrained_rows.insert(key, row.clone());
    row
}

fn evaluate_hand(cards: &[String], expected_count: usize) -> i64 {
    if cards.len() != expected_count {
        return 0;
    }
    let joker_count = cards.iter().filter(|card| is_joker(card)).count();
    if joker_count == 0 {
        return evaluate_natural_hand(cards, expected_count);
    }

    let non_jokers: Vec<String> = cards
        .iter()
        .filter(|card| !is_joker(card))
        .cloned()
        .collect();
    let substitutions = available_subs_for_row(cards);
    let mut best_value = -1;
    if joker_count == 1 {
        for substitution in substitutions {
            let mut natural = non_jokers.clone();
            natural.push(substitution);
            best_value = best_value.max(evaluate_natural_hand(&natural, expected_count));
        }
    } else if joker_count == 2 {
        for first in 0..substitutions.len() {
            for second in (first + 1)..substitutions.len() {
                let mut natural = non_jokers.clone();
                natural.push(substitutions[first].clone());
                natural.push(substitutions[second].clone());
                best_value = best_value.max(evaluate_natural_hand(&natural, expected_count));
            }
        }
    }
    best_value
}

fn evaluate_natural_hand(cards: &[String], expected_count: usize) -> i64 {
    if cards.len() != expected_count {
        return 0;
    }
    let mut ranks = Vec::new();
    let mut suits = Vec::new();
    let mut jokers = 0usize;
    for card in cards {
        if is_joker(card) {
            jokers += 1;
        } else if let Some(rank) = rank_value(card) {
            ranks.push(rank);
            if let Some(suit) = suit_value(card) {
                suits.push(suit);
            }
        }
    }
    ranks.sort_by(|a, b| b.cmp(a));
    let counts = rank_counts(&ranks);
    let is_flush =
        suits.iter().collect::<BTreeSet<_>>().len() == 1 && suits.len() + jokers == expected_count;
    let is_straight = check_straight(&ranks, jokers);

    if expected_count == 3 {
        let best = counts.values().copied().max().unwrap_or(0);
        if best + jokers >= 3 {
            let r = if best >= 3 {
                counts
                    .iter()
                    .filter(|(_, &c)| c >= 3)
                    .map(|(&r, _)| r)
                    .max()
                    .unwrap_or(0)
            } else if best >= 2 {
                counts
                    .iter()
                    .filter(|(_, &c)| c >= 2)
                    .map(|(&r, _)| r)
                    .max()
                    .unwrap_or(0)
            } else {
                ranks.first().copied().unwrap_or(14)
            };
            return encode_hand(3, &[r as i64]);
        }
        if best + jokers >= 2 {
            let (pr, kickers): (u8, Vec<u8>) = if best >= 2 {
                let pair_rank = counts
                    .iter()
                    .filter(|(_, &c)| c >= 2)
                    .map(|(&r, _)| r)
                    .max()
                    .unwrap_or(0);
                let k = ranks.iter().copied().filter(|r| *r != pair_rank).collect();
                (pair_rank, k)
            } else {
                (
                    ranks.first().copied().unwrap_or(0),
                    ranks.iter().copied().skip(1).collect(),
                )
            };
            return encode_hand(
                1,
                &[pr as i64, kickers.first().copied().unwrap_or(0) as i64],
            );
        }
        return encode_hand(0, &ranks.iter().map(|r| *r as i64).collect::<Vec<_>>());
    }

    let best = counts.values().copied().max().unwrap_or(0);
    let mut pairs: Vec<u8> = counts
        .iter()
        .filter(|(_, &c)| c >= 2)
        .map(|(&r, _)| r)
        .collect();
    pairs.sort_by(|a, b| b.cmp(a));

    if is_flush && is_straight {
        return encode_hand(8, &[straight_high(&ranks, jokers) as i64]);
    }
    if best + jokers >= 4 {
        let qr = if best >= 4 {
            counts
                .iter()
                .filter(|(_, &c)| c >= 4)
                .map(|(&r, _)| r)
                .max()
                .unwrap_or(0)
        } else if best >= 3 {
            counts
                .iter()
                .filter(|(_, &c)| c >= 3)
                .map(|(&r, _)| r)
                .max()
                .unwrap_or(0)
        } else {
            pairs
                .first()
                .copied()
                .or_else(|| ranks.first().copied())
                .unwrap_or(0)
        };
        let kicker = ranks
            .iter()
            .copied()
            .filter(|r| *r != qr)
            .max()
            .unwrap_or(0);
        return encode_hand(7, &[qr as i64, kicker as i64]);
    }
    if best >= 3 {
        let tr = counts
            .iter()
            .filter(|(_, &c)| c >= 3)
            .map(|(&r, _)| r)
            .max()
            .unwrap_or(0);
        let pc: Vec<u8> = counts
            .iter()
            .filter(|(&r, &c)| c >= 2 && r != tr)
            .map(|(&r, _)| r)
            .collect();
        if !pc.is_empty() {
            return encode_hand(6, &[tr as i64, pc.into_iter().max().unwrap_or(0) as i64]);
        }
    }
    if jokers >= 1 && pairs.len() >= 2 {
        return encode_hand(6, &[pairs[0] as i64, pairs[1] as i64]);
    }
    if is_flush {
        return encode_hand(
            5,
            &ranks.iter().take(5).map(|r| *r as i64).collect::<Vec<_>>(),
        );
    }
    if is_straight {
        return encode_hand(4, &[straight_high(&ranks, jokers) as i64]);
    }
    if best + jokers >= 3 {
        let tr = if best >= 3 {
            counts
                .iter()
                .filter(|(_, &c)| c >= 3)
                .map(|(&r, _)| r)
                .max()
                .unwrap_or(0)
        } else if best >= 2 {
            counts
                .iter()
                .filter(|(_, &c)| c >= 2)
                .map(|(&r, _)| r)
                .max()
                .unwrap_or(0)
        } else {
            ranks.first().copied().unwrap_or(0)
        };
        let k: Vec<u8> = ranks.iter().copied().filter(|r| *r != tr).collect();
        return encode_hand(
            3,
            &[
                tr as i64,
                k.get(0).copied().unwrap_or(0) as i64,
                k.get(1).copied().unwrap_or(0) as i64,
            ],
        );
    }
    if pairs.len() >= 2 {
        let kicker = ranks
            .iter()
            .copied()
            .filter(|r| !pairs[..2].contains(r))
            .max()
            .unwrap_or(0);
        return encode_hand(2, &[pairs[0] as i64, pairs[1] as i64, kicker as i64]);
    }
    if best >= 2 || jokers >= 1 {
        let (pr, k): (u8, Vec<u8>) = if !pairs.is_empty() {
            (
                pairs[0],
                ranks.iter().copied().filter(|r| *r != pairs[0]).collect(),
            )
        } else {
            (
                ranks.first().copied().unwrap_or(0),
                ranks.iter().copied().skip(1).collect(),
            )
        };
        return encode_hand(
            1,
            &[
                pr as i64,
                k.get(0).copied().unwrap_or(0) as i64,
                k.get(1).copied().unwrap_or(0) as i64,
                k.get(2).copied().unwrap_or(0) as i64,
            ],
        );
    }
    encode_hand(
        0,
        &ranks.iter().take(5).map(|r| *r as i64).collect::<Vec<_>>(),
    )
}

fn middle_royalty_from_value(val: i64) -> i32 {
    let cat = hand_category(val);
    let r1 = (val / B.pow(4)) % B;
    match cat {
        8 if r1 == 14 => 50,
        8 => 30,
        7 => 20,
        6 => 12,
        5 => 8,
        4 => 4,
        3 => 2,
        _ => 0,
    }
}

fn bottom_royalty_from_value(val: i64) -> i32 {
    let cat = hand_category(val);
    let r1 = (val / B.pow(4)) % B;
    match cat {
        8 if r1 == 14 => 25,
        8 => 15,
        7 => 10,
        6 => 6,
        5 => 4,
        4 => 2,
        _ => 0,
    }
}

fn top_rank_from_value(val: i64) -> i64 {
    (val / B.pow(4)) % B
}

fn top_royalty_from_value(val: i64) -> i32 {
    let cat = hand_category(val);
    let r1 = top_rank_from_value(val);
    match cat {
        3 => 10 + (r1 as i32 - 2),
        1 if r1 >= 6 => r1 as i32 - 5,
        _ => 0,
    }
}

fn fl_card_count_from_top_value(val: i64) -> u8 {
    let cat = hand_category(val);
    let r1 = top_rank_from_value(val);
    match (cat, r1) {
        (3, _) => 17,
        (1, 14) => 16,
        (1, 13) => 15,
        (1, 12) => 14,
        _ => 0,
    }
}

fn constrained_rows(
    board: &Board,
    cache: &mut RowEvalCache,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let bottom = board.bottom.clone();
    let bottom_value = if bottom.len() == 5 {
        cache.evaluate_hand(&bottom, 5)
    } else {
        0
    };
    let middle = if board.middle.len() == 5 && bottom.len() == 5 && row_has_joker(&board.middle) {
        constrain_row_to_value(&board.middle, bottom_value, 5, cache)
    } else {
        board.middle.clone()
    };
    let middle_value = if middle.len() == 5 {
        cache.evaluate_hand(&middle, 5)
    } else {
        0
    };
    let top = if board.top.len() == 3 && middle.len() == 5 && row_has_joker(&board.top) {
        constrain_row_to_value(&board.top, middle_value, 3, cache)
    } else {
        board.top.clone()
    };
    (top, middle, bottom)
}

fn board_eval(board: &Board, cache: &mut RowEvalCache) -> BoardEval {
    let key = board_eval_key(board);
    if let Some(value) = cache.board_evals.get(&key).cloned() {
        cache.board_hits += 1;
        return value;
    }
    cache.board_misses += 1;
    let value = board_eval_uncached(board, cache);
    cache.board_evals.insert(key, value.clone());
    value
}

fn board_eval_uncached(board: &Board, cache: &mut RowEvalCache) -> BoardEval {
    let complete = board.is_complete();
    if complete {
        let (top, middle, bottom) = constrained_rows(board, cache);
        let vals = [
            cache.evaluate_hand(&top, 3),
            cache.evaluate_hand(&middle, 5),
            cache.evaluate_hand(&bottom, 5),
        ];
        let busted = vals[0] > vals[1] || vals[1] > vals[2];
        let fl_key = fl_card_count_from_top_value(vals[0]);
        let fl_type = match fl_key {
            14 | 15 | 16 | 17 => Some(fl_key),
            _ => None,
        };
        let royalty = if busted {
            0
        } else {
            top_royalty_from_value(vals[0])
                + middle_royalty_from_value(vals[1])
                + bottom_royalty_from_value(vals[2])
        };
        return BoardEval {
            vals,
            complete,
            busted,
            royalty,
            fl_key,
            fl_type,
        };
    }

    let vals = [
        if board.top.len() == 3 {
            cache.evaluate_hand(&board.top, 3)
        } else {
            0
        },
        if board.middle.len() == 5 {
            cache.evaluate_hand(&board.middle, 5)
        } else {
            0
        },
        if board.bottom.len() == 5 {
            cache.evaluate_hand(&board.bottom, 5)
        } else {
            0
        },
    ];
    let busted = complete && (vals[0] > vals[1] || vals[1] > vals[2]);
    let fl_key = if board.top.len() == 3 {
        fl_card_count_from_top_value(vals[0])
    } else {
        0
    };
    let fl_type = match fl_key {
        14 | 15 | 16 | 17 => Some(fl_key),
        _ => None,
    };
    let royalty = if busted {
        0
    } else {
        let top = if board.top.len() == 3 {
            top_royalty_from_value(vals[0])
        } else {
            0
        };
        let middle = if board.middle.len() == 5 {
            middle_royalty_from_value(vals[1])
        } else {
            0
        };
        let bottom = if board.bottom.len() == 5 {
            bottom_royalty_from_value(vals[2])
        } else {
            0
        };
        top + middle + bottom
    };
    BoardEval {
        vals,
        complete,
        busted,
        royalty,
        fl_key,
        fl_type,
    }
}

fn is_irreparably_busted(board: &Board, cache: &mut RowEvalCache) -> bool {
    let (top, middle, bottom) = constrained_rows(board, cache);
    if board.top.len() == 3 && board.middle.len() == 5 {
        let top_value = cache.evaluate_hand(&top, 3);
        let middle_value = cache.evaluate_hand(&middle, 5);
        if top_value > middle_value {
            return true;
        }
    }
    if board.middle.len() == 5 && board.bottom.len() == 5 {
        let middle_value = cache.evaluate_hand(&middle, 5);
        let bottom_value = cache.evaluate_hand(&bottom, 5);
        if middle_value > bottom_value {
            return true;
        }
    }
    false
}

fn compute_score_raw_eval(my: &BoardEval, opp: &BoardEval) -> f64 {
    if my.busted && opp.busted {
        return 0.0;
    }
    if my.busted {
        return (-6 - opp.royalty) as f64;
    }
    if opp.busted {
        return (6 + my.royalty) as f64;
    }
    let mut line_total: i32 = 0;
    for i in 0..3 {
        match my.vals[i].cmp(&opp.vals[i]) {
            Ordering::Greater => line_total += 1,
            Ordering::Less => line_total -= 1,
            Ordering::Equal => {}
        }
    }
    let scoop_bonus = if line_total.abs() == 3 { 3 } else { 0 };
    let mut score = line_total;
    score += if line_total > 0 {
        scoop_bonus
    } else if line_total < 0 {
        -scoop_bonus
    } else {
        0
    };
    score += my.royalty - opp.royalty;
    score as f64
}

fn terminal_metrics_from_eval(
    eval: &BoardEval,
    opponent_eval: &BoardEval,
    fl_ev: &FlEv,
) -> Terminal {
    if opponent_eval.complete {
        let raw_score = compute_score_raw_eval(eval, opponent_eval);
        let mut score = raw_score;
        if !eval.busted {
            score += fl_ev.get(eval.fl_key);
        }
        if !opponent_eval.busted {
            score -= fl_ev.get(opponent_eval.fl_key);
        }
        Terminal {
            score,
            raw_score,
            royalty: eval.royalty as f64,
            bust: eval.busted,
            fl_type: if eval.busted { None } else { eval.fl_type },
        }
    } else {
        let score = if eval.busted {
            0.0
        } else {
            eval.royalty as f64 + fl_ev.get(eval.fl_key)
        };
        Terminal {
            score,
            raw_score: if eval.busted {
                0.0
            } else {
                eval.royalty as f64
            },
            royalty: eval.royalty as f64,
            bust: eval.busted,
            fl_type: if eval.busted { None } else { eval.fl_type },
        }
    }
}

fn row_cards(board: &Board, row_idx: usize) -> &[String] {
    match row_idx {
        0 => &board.top,
        1 => &board.middle,
        2 => &board.bottom,
        _ => unreachable!(),
    }
}

fn t4_action_key(
    cards: &[&String; 3],
    discard_idx: usize,
    remaining: &[usize; 2],
    rows: &[usize; 2],
) -> u64 {
    let mut key = card_code(cards[discard_idx]) as u64 + 1;
    for row_idx in 0..3 {
        let mut row_codes = [0u8; 2];
        let mut row_len = 0usize;
        for placement_idx in 0..2 {
            if rows[placement_idx] == row_idx {
                row_codes[row_len] = card_code(cards[remaining[placement_idx]]);
                row_len += 1;
            }
        }
        row_codes[..row_len].sort_unstable();
        key = (key << 2) | row_len as u64;
        for code in &row_codes[..row_len] {
            key = (key << 6) | (*code as u64 + 1);
        }
    }
    key
}

fn row_cards_after_t4_placement(
    board: &Board,
    cards: &[&String; 3],
    remaining: &[usize; 2],
    rows: &[usize; 2],
    row_idx: usize,
) -> Vec<String> {
    let mut out = row_cards(board, row_idx).to_vec();
    for placement_idx in 0..2 {
        if rows[placement_idx] == row_idx {
            out.push((*cards[remaining[placement_idx]]).clone());
        }
    }
    out
}

fn board_eval_key_after_t4_placement(
    board: &Board,
    cards: &[&String; 3],
    remaining: &[usize; 2],
    rows: &[usize; 2],
) -> BoardEvalKey {
    let board_rows = [&board.top, &board.middle, &board.bottom];
    let mut row_lens = [0u8; 3];
    let mut codes = [0u8; 13];
    let mut offset = 0usize;
    for row_idx in 0..3 {
        let row = board_rows[row_idx];
        row_lens[row_idx] = (row.len()
            + rows
                .iter()
                .filter(|placement_row| **placement_row == row_idx)
                .count()) as u8;
        let mut row_len = 0usize;
        for card in row {
            codes[offset + row_len] = card_code(card).saturating_add(1);
            row_len += 1;
        }
        for placement_idx in 0..2 {
            if rows[placement_idx] == row_idx {
                codes[offset + row_len] =
                    card_code(cards[remaining[placement_idx]]).saturating_add(1);
                row_len += 1;
            }
        }
        codes[offset..offset + row_len].sort_unstable();
        offset += row_len;
    }
    BoardEvalKey { row_lens, codes }
}

fn board_eval_after_t4_placement(
    board: &Board,
    cards: &[&String; 3],
    remaining: &[usize; 2],
    rows: &[usize; 2],
    cache: &mut RowEvalCache,
) -> BoardEval {
    let key = board_eval_key_after_t4_placement(board, cards, remaining, rows);
    if let Some(eval) = cache.board_evals.get(&key).cloned() {
        cache.board_hits += 1;
        return eval;
    }
    cache.board_misses += 1;
    let final_board = Board {
        top: row_cards_after_t4_placement(board, cards, remaining, rows, 0),
        middle: row_cards_after_t4_placement(board, cards, remaining, rows, 1),
        bottom: row_cards_after_t4_placement(board, cards, remaining, rows, 2),
    };
    let eval = board_eval_uncached(&final_board, cache);
    cache.board_evals.insert(key, eval.clone());
    eval
}

fn terminal_metrics_after_t4_placement(
    board: &Board,
    cards: &[&String; 3],
    remaining: &[usize; 2],
    rows: &[usize; 2],
    opponent_eval: &BoardEval,
    fl_ev: &FlEv,
    cache: &mut RowEvalCache,
) -> Terminal {
    let eval = board_eval_after_t4_placement(board, cards, remaining, rows, cache);
    terminal_metrics_from_eval(&eval, opponent_eval, fl_ev)
}

fn accumulate(totals: &mut Totals, terminal: &Terminal) {
    totals.score += terminal.score;
    totals.raw_score += terminal.raw_score;
    totals.royalty += terminal.royalty;
    if terminal.bust {
        totals.bust += 1.0;
    }
    if let Some(fl) = terminal.fl_type {
        totals.fl_any += 1.0;
        match fl {
            14 => totals.fl_qq += 1.0,
            15 => totals.fl_kk += 1.0,
            16 => totals.fl_aa += 1.0,
            17 => totals.fl_trips += 1.0,
            _ => {}
        }
    }
}

fn metrics_from_totals(totals: Totals, n: usize) -> Metrics {
    metrics_from_totals_with_source(totals, n, "exact")
}

fn metrics_from_totals_with_source(totals: Totals, n: usize, source: &'static str) -> Metrics {
    let denom = n.max(1) as f64;
    let mut fl_type_rates = BTreeMap::new();
    fl_type_rates.insert("qq".to_string(), totals.fl_qq / denom);
    fl_type_rates.insert("kk".to_string(), totals.fl_kk / denom);
    fl_type_rates.insert("aa".to_string(), totals.fl_aa / denom);
    fl_type_rates.insert("trips".to_string(), totals.fl_trips / denom);
    Metrics {
        score: totals.score / denom,
        ev: totals.score / denom,
        raw_score: totals.raw_score / denom,
        royalty: totals.royalty / denom,
        bust_rate: totals.bust / denom,
        fl_rate: totals.fl_any / denom,
        fl_type_rates,
        samples: n,
        source,
        forced_bust: n > 0 && (totals.bust - n as f64).abs() < 1e-9,
        enumerated_draws: None,
        remaining_deck_size: None,
        start_turn: None,
        opponent_response: None,
    }
}

fn combinations3_count(n: usize, limit: usize) -> usize {
    if n < 3 {
        return 0;
    }
    let total = n * (n - 1) * (n - 2) / 6;
    if limit == 0 {
        total
    } else {
        total.min(limit)
    }
}

fn forced_bust_metrics(samples: usize, source: &'static str) -> Metrics {
    let mut totals = Totals::default();
    totals.bust = samples as f64;
    metrics_from_totals_with_source(totals, samples, source)
}

fn best_t4_completion(
    board: &Board,
    dealt: [&String; 3],
    opponent_eval: &BoardEval,
    fl_ev: &FlEv,
    cache: &mut RowEvalCache,
) -> Terminal {
    let mut cards = dealt;
    cards.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    let limits = [3usize, 5, 5];
    let lens = [board.top.len(), board.middle.len(), board.bottom.len()];
    let mut seen = Vec::with_capacity(24);
    let mut best: Option<Terminal> = None;

    for discard_idx in 0..3 {
        let remaining = match discard_idx {
            0 => [1, 2],
            1 => [0, 2],
            2 => [0, 1],
            _ => unreachable!(),
        };
        for pos0 in 0..3 {
            for pos1 in 0..3 {
                let mut counts = lens;
                counts[pos0] += 1;
                if counts[pos0] > limits[pos0] {
                    continue;
                }
                counts[pos1] += 1;
                if counts[pos1] > limits[pos1] {
                    continue;
                }
                let rows = [pos0, pos1];
                let action_key = t4_action_key(&cards, discard_idx, &remaining, &rows);
                if seen.iter().any(|key| *key == action_key) {
                    continue;
                }
                seen.push(action_key);
                let terminal = terminal_metrics_after_t4_placement(
                    board,
                    &cards,
                    &remaining,
                    &rows,
                    opponent_eval,
                    fl_ev,
                    cache,
                );
                let replace = match &best {
                    None => true,
                    Some(current) => {
                        compare_terminal_quality(&terminal, current) == Ordering::Greater
                    }
                };
                if replace {
                    best = Some(terminal);
                }
            }
        }
    }
    best.expect("T4 should have at least one legal action")
}

fn visit_t4_completion_evals<F>(
    board: &Board,
    dealt: [&String; 3],
    cache: &mut RowEvalCache,
    mut visit: F,
) -> usize
where
    F: FnMut(&BoardEval),
{
    let mut cards = dealt;
    cards.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    let limits = [3usize, 5, 5];
    let lens = [board.top.len(), board.middle.len(), board.bottom.len()];
    let mut seen = Vec::with_capacity(24);
    let mut n = 0usize;

    for discard_idx in 0..3 {
        let remaining = match discard_idx {
            0 => [1, 2],
            1 => [0, 2],
            2 => [0, 1],
            _ => unreachable!(),
        };
        for pos0 in 0..3 {
            for pos1 in 0..3 {
                let mut counts = lens;
                counts[pos0] += 1;
                if counts[pos0] > limits[pos0] {
                    continue;
                }
                counts[pos1] += 1;
                if counts[pos1] > limits[pos1] {
                    continue;
                }
                let rows = [pos0, pos1];
                let action_key = t4_action_key(&cards, discard_idx, &remaining, &rows);
                if seen.iter().any(|key| *key == action_key) {
                    continue;
                }
                seen.push(action_key);
                let eval = board_eval_after_t4_placement(board, &cards, &remaining, &rows, cache);
                visit(&eval);
                n += 1;
            }
        }
    }
    n
}

fn exact_candidate_metrics_for_state(
    next_board: &Board,
    opponent: &Board,
    opponent_eval: &BoardEval,
    next_exclude: &[String],
    fl_ev: &FlEv,
    cache: &mut RowEvalCache,
) -> Metrics {
    let deck = remaining_deck(&next_board, opponent, &[], &next_exclude);
    if is_irreparably_busted(&next_board, cache) {
        return forced_bust_metrics(combinations3_count(deck.len(), 0), "exact");
    }
    let mut totals = Totals::default();
    let mut n = 0usize;

    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            for k in (j + 1)..deck.len() {
                let draw = [&deck[i], &deck[j], &deck[k]];
                let best = best_t4_completion(&next_board, draw, opponent_eval, fl_ev, cache);
                accumulate(&mut totals, &best);
                n += 1;
            }
        }
    }
    metrics_from_totals(totals, n)
}

fn exact_candidate_metrics(
    board: &Board,
    action: &Action,
    opponent: &Board,
    opponent_eval: &BoardEval,
    exclude: &[String],
    fl_ev: &FlEv,
) -> EvalOutcome {
    let next_board = apply_action(board, action);
    let next_exclude = append_exclude_card(exclude, &action.discard);
    let mut cache = RowEvalCache::default();
    let metrics = exact_candidate_metrics_for_state(
        &next_board,
        opponent,
        opponent_eval,
        &next_exclude,
        fl_ev,
        &mut cache,
    );
    EvalOutcome {
        metrics,
        row_eval_cache: cache.stats(),
    }
}

fn exact_t4_candidate_metrics(
    board: &Board,
    action: &Action,
    opponent: &Board,
    opponent_eval: &BoardEval,
    exclude: &[String],
    fl_ev: &FlEv,
) -> EvalOutcome {
    let next_board = apply_action(board, action);
    let next_exclude = append_exclude_card(exclude, &action.discard);
    let mut cache = RowEvalCache::default();
    let hero_eval = board_eval(&next_board, &mut cache);
    let opponent_card_count = opponent.all_cards().count();

    let metrics = if opponent_card_count == 11 {
        // Hero is BB and acts first on T4.  Enumerate every possible BTN draw,
        // let BTN choose its best canonical terminal placement, then mirror the
        // zero-sum result back to Hero while retaining Hero's board statistics.
        let deck = remaining_deck(&next_board, opponent, &[], &next_exclude);
        let mut totals = Totals::default();
        let n = visit_combinations3(&deck, 0, |draw| {
            let opponent_best = best_t4_completion(opponent, draw, &hero_eval, fl_ev, &mut cache);
            let hero_terminal = Terminal {
                score: -opponent_best.score,
                raw_score: -opponent_best.raw_score,
                royalty: hero_eval.royalty as f64,
                bust: hero_eval.busted,
                fl_type: if hero_eval.busted {
                    None
                } else {
                    hero_eval.fl_type
                },
            };
            accumulate(&mut totals, &hero_terminal);
        });
        assert!(
            n > 0,
            "T4 opponent-response evaluation requires at least 3 live cards"
        );
        let mut metrics = metrics_from_totals_with_source(totals, n, "exact_hu_response");
        metrics.enumerated_draws = Some(n);
        metrics.remaining_deck_size = Some(deck.len());
        metrics.start_turn = Some(4);
        metrics.opponent_response = Some(true);
        metrics
    } else {
        // BTN faces a completed BB board.  Empty/partial opponent fixtures keep
        // the historical exact self-board objective used by offline teachers.
        let terminal = terminal_metrics_from_eval(&hero_eval, opponent_eval, fl_ev);
        let mut totals = Totals::default();
        accumulate(&mut totals, &terminal);
        let mut metrics = metrics_from_totals(totals, 1);
        metrics.enumerated_draws = Some(1);
        metrics.remaining_deck_size = Some(0);
        metrics.start_turn = Some(5);
        metrics.opponent_response = Some(false);
        metrics
    };

    EvalOutcome {
        metrics,
        row_eval_cache: cache.stats(),
    }
}

fn exact_t4_hu_candidate_metrics_joint(
    board: &Board,
    actions: &[Action],
    dealt: &[String],
    opponent: &Board,
    exclude: &[String],
    fl_ev: &FlEv,
) -> Vec<EvalOutcome> {
    if actions.is_empty() {
        return Vec::new();
    }
    assert_eq!(
        opponent.all_cards().count(),
        11,
        "joint T4 HU evaluation requires an 11-card opponent board"
    );

    // Every legal T4 action consumes the same three dealt cards: two are
    // placed and one is discarded.  The future BTN deck is therefore common
    // to every BB candidate even though the completed BB boards differ.
    let deck = remaining_deck(board, opponent, dealt, exclude);
    let mut cache = RowEvalCache::default();
    let hero_evals: Vec<BoardEval> = actions
        .iter()
        .map(|action| board_eval(&apply_action(board, action), &mut cache))
        .collect();
    let mut totals = vec![Totals::default(); actions.len()];

    let n = visit_combinations3(&deck, 0, |draw| {
        let mut opponent_bests: Vec<Option<Terminal>> = vec![None; hero_evals.len()];
        let response_count =
            visit_t4_completion_evals(opponent, draw, &mut cache, |opponent_eval| {
                for (candidate_index, hero_eval) in hero_evals.iter().enumerate() {
                    let terminal = terminal_metrics_from_eval(opponent_eval, hero_eval, fl_ev);
                    let replace = match &opponent_bests[candidate_index] {
                        None => true,
                        Some(current) => {
                            compare_terminal_quality(&terminal, current) == Ordering::Greater
                        }
                    };
                    if replace {
                        opponent_bests[candidate_index] = Some(terminal);
                    }
                }
            });
        assert!(
            response_count > 0,
            "T4 should have at least one legal response"
        );

        for (candidate_index, opponent_best) in opponent_bests.into_iter().enumerate() {
            let opponent_best = opponent_best.expect("T4 should have a best response");
            let hero_eval = &hero_evals[candidate_index];
            let hero_terminal = Terminal {
                score: -opponent_best.score,
                raw_score: -opponent_best.raw_score,
                royalty: hero_eval.royalty as f64,
                bust: hero_eval.busted,
                fl_type: if hero_eval.busted {
                    None
                } else {
                    hero_eval.fl_type
                },
            };
            accumulate(&mut totals[candidate_index], &hero_terminal);
        }
    });
    assert!(
        n > 0,
        "T4 opponent-response evaluation requires at least 3 live cards"
    );

    let cache_stats = cache.stats();
    totals
        .into_iter()
        .map(|candidate_totals| {
            let mut metrics =
                metrics_from_totals_with_source(candidate_totals, n, "exact_hu_response");
            metrics.enumerated_draws = Some(n);
            metrics.remaining_deck_size = Some(deck.len());
            metrics.start_turn = Some(4);
            metrics.opponent_response = Some(true);
            EvalOutcome {
                metrics,
                row_eval_cache: cache_stats.clone(),
            }
        })
        .collect()
}

fn visit_combinations3<F>(deck: &[String], limit: usize, mut visit: F) -> usize
where
    F: FnMut([&String; 3]),
{
    let max_len = if limit == 0 { usize::MAX } else { limit };
    let mut n = 0usize;
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            for k in (j + 1)..deck.len() {
                visit([&deck[i], &deck[j], &deck[k]]);
                n += 1;
                if n >= max_len {
                    return n;
                }
            }
        }
    }
    n
}

fn best_t3_metrics(
    board: &Board,
    dealt: &[String],
    opponent: &Board,
    opponent_eval: &BoardEval,
    exclude: &[String],
    fl_ev: &FlEv,
    cache: &mut RowEvalCache,
) -> Metrics {
    let actions = get_turn_actions(dealt, board);
    actions
        .iter()
        .map(|action| {
            let next_board = apply_action(board, action);
            let next_exclude = append_exclude_card(exclude, &action.discard);
            exact_candidate_metrics_for_state(
                &next_board,
                opponent,
                opponent_eval,
                &next_exclude,
                fl_ev,
                cache,
            )
        })
        .max_by(compare_metrics_quality)
        .expect("T3 should have at least one legal action")
}

fn exact_t2_candidate_metrics(
    board: &Board,
    action: &Action,
    opponent: &Board,
    opponent_eval: &BoardEval,
    exclude: &[String],
    fl_ev: &FlEv,
    draw_limit: usize,
) -> EvalOutcome {
    let next_board = apply_action(board, action);
    let next_exclude = append_exclude_card(exclude, &action.discard);
    let deck = remaining_deck(&next_board, opponent, &[], &next_exclude);
    let mut cache = RowEvalCache::default();
    let source = if draw_limit == 0 {
        "exact_t2"
    } else {
        "exact_t2_capped"
    };
    if is_irreparably_busted(&next_board, &mut cache) {
        return EvalOutcome {
            metrics: forced_bust_metrics(combinations3_count(deck.len(), draw_limit), source),
            row_eval_cache: cache.stats(),
        };
    }
    let mut totals = Totals::default();

    let n = visit_combinations3(&deck, draw_limit, |draw| {
        let dealt = [draw[0].clone(), draw[1].clone(), draw[2].clone()];
        let best = best_t3_metrics(
            &next_board,
            &dealt,
            opponent,
            opponent_eval,
            &next_exclude,
            fl_ev,
            &mut cache,
        );
        totals.score += best.score;
        totals.raw_score += best.raw_score;
        totals.royalty += best.royalty;
        totals.bust += best.bust_rate;
        totals.fl_any += best.fl_rate;
        totals.fl_qq += best.fl_type_rates.get("qq").copied().unwrap_or(0.0);
        totals.fl_kk += best.fl_type_rates.get("kk").copied().unwrap_or(0.0);
        totals.fl_aa += best.fl_type_rates.get("aa").copied().unwrap_or(0.0);
        totals.fl_trips += best.fl_type_rates.get("trips").copied().unwrap_or(0.0);
    });

    EvalOutcome {
        metrics: metrics_from_totals_with_source(totals, n, source),
        row_eval_cache: cache.stats(),
    }
}

fn evaluate_position(
    record_index: usize,
    payload: &PositionPayload,
    fl_ev: &FlEv,
    top_n: usize,
    t2_draw_limit: usize,
    source_candidate_top_k: usize,
) -> PositionResult {
    let started = Instant::now();
    let board = Board::from_payload(&payload.board);
    let opponent = Board::from_payload(&payload.opponent_board);
    let mut opponent_eval_cache = RowEvalCache::default();
    let opponent_eval = board_eval(&opponent, &mut opponent_eval_cache);
    let dealt = normalize_cards(&payload.dealt);
    let mut exclude_src = payload.exclude.clone();
    exclude_src.extend(payload.known_discards.clone());
    let exclude = canonical_exclude_cards(&exclude_src, &board, &opponent, &dealt);
    let legal_actions = get_turn_actions(&dealt, &board);
    let mut requested_payload_actions = payload.candidate_actions.clone();
    if requested_payload_actions.is_empty() && source_candidate_top_k > 0 {
        requested_payload_actions = payload
            .candidates
            .iter()
            .filter_map(SourceCandidatePayload::action_payload)
            .take(source_candidate_top_k)
            .collect();
    }
    let actions = requested_action_subset(&requested_payload_actions, &legal_actions);
    let turn = payload.turn.unwrap_or(3);
    let mut candidates: Vec<CandidateResult> = if turn == 4 && opponent.all_cards().count() == 11 {
        let outcomes = exact_t4_hu_candidate_metrics_joint(
            &board, &actions, &dealt, &opponent, &exclude, fl_ev,
        );
        actions
            .iter()
            .zip(outcomes)
            .map(|(action, outcome)| CandidateResult {
                action: action_to_payload(action),
                board: apply_action(&board, action).to_payload(),
                metrics: outcome.metrics,
                row_eval_cache: outcome.row_eval_cache,
            })
            .collect()
    } else {
        actions
            .par_iter()
            .map(|action| {
                let next_board = apply_action(&board, action);
                let outcome = match turn {
                    2 => exact_t2_candidate_metrics(
                        &board,
                        action,
                        &opponent,
                        &opponent_eval,
                        &exclude,
                        fl_ev,
                        t2_draw_limit,
                    ),
                    4 => exact_t4_candidate_metrics(
                        &board,
                        action,
                        &opponent,
                        &opponent_eval,
                        &exclude,
                        fl_ev,
                    ),
                    _ => exact_candidate_metrics(
                        &board,
                        action,
                        &opponent,
                        &opponent_eval,
                        &exclude,
                        fl_ev,
                    ),
                };
                CandidateResult {
                    action: action_to_payload(action),
                    board: next_board.to_payload(),
                    metrics: outcome.metrics,
                    row_eval_cache: outcome.row_eval_cache,
                }
            })
            .collect()
    };
    candidates.sort_by(|a, b| {
        let quality_order = compare_metrics_quality(&b.metrics, &a.metrics);
        if quality_order == Ordering::Equal {
            action_payload_key(&a.action).cmp(&action_payload_key(&b.action))
        } else {
            quality_order
        }
    });
    let best = candidates.first().cloned();
    candidates.truncate(top_n);
    PositionResult {
        record_index,
        turn,
        legal_actions: legal_actions.len(),
        evaluated_actions: actions.len(),
        requested_actions: requested_payload_actions.len(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        best,
        candidates,
    }
}

impl FlEv {
    fn get(&self, key: u8) -> f64 {
        self.values.get(&key).copied().unwrap_or(0.0)
    }
}

fn load_fl_ev(path: &PathBuf) -> FlEv {
    let fallback = BTreeMap::from([(14, 3.0), (15, 10.0), (16, 15.0), (17, 20.0)]);
    let Ok(text) = std::fs::read_to_string(path) else {
        return FlEv { values: fallback };
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return FlEv { values: fallback };
    };
    if value.get("reward_mode").and_then(Value::as_str) == Some("direct") {
        if let Some(map) = value.get("fl_ev_direct").and_then(Value::as_object) {
            let mut values = BTreeMap::new();
            for (key, val) in map {
                if let (Ok(k), Some(v)) = (key.parse::<u8>(), val.as_f64()) {
                    values.insert(k, v);
                }
            }
            if !values.is_empty() {
                return FlEv { values };
            }
        }
    }
    let Some(opp) = value.get("opponent_avg_royalty").and_then(Value::as_f64) else {
        return FlEv { values: fallback };
    };
    let Some(stats) = value.get("fl_stats").and_then(Value::as_object) else {
        return FlEv { values: fallback };
    };
    let mut values = BTreeMap::new();
    for (key, stat) in stats {
        let Some(r) = stat.get("R").and_then(Value::as_f64) else {
            continue;
        };
        let Some(stay_rate) = stat.get("stay_rate").and_then(Value::as_f64) else {
            continue;
        };
        if let Ok(k) = key.parse::<u8>() {
            values.insert(k, (r - opp) / (1.0 - stay_rate));
        }
    }
    if values.is_empty() {
        FlEv { values: fallback }
    } else {
        FlEv { values }
    }
}

fn parse_payload(line: &str) -> Result<PositionPayload> {
    let value: Value = serde_json::from_str(line.trim_start_matches('\u{feff}'))?;
    let payload: PositionPayload = serde_json::from_value(value)?;
    Ok(payload)
}

fn read_payloads(args: &Args) -> Result<Vec<(usize, PositionPayload)>> {
    let reader = BufReader::new(
        File::open(&args.input)
            .with_context(|| format!("failed to open {}", args.input.display()))?,
    );
    let mut out = Vec::new();
    let mut eligible_seen = 0usize;
    for (idx, line) in reader.lines().enumerate() {
        if args.limit > 0 && out.len() >= args.limit {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let payload =
            parse_payload(&line).with_context(|| format!("failed to parse line {}", idx + 1))?;
        if !matches!(payload.turn.unwrap_or(3), 2 | 3 | 4) {
            continue;
        }
        if eligible_seen < args.skip {
            eligible_seen += 1;
            continue;
        }
        eligible_seen += 1;
        out.push((idx, payload));
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let fl_ev = load_fl_ev(&args.fl_config);
    let mut writer: Box<dyn Write> = if let Some(path) = &args.output {
        Box::new(BufWriter::new(File::create(path).with_context(|| {
            format!("failed to create {}", path.display())
        })?))
    } else {
        Box::new(std::io::stdout())
    };

    let started = Instant::now();
    let records = read_payloads(&args)?;
    let mut elapsed_ms = Vec::new();

    let mut results: Vec<PositionResult> = if args.position_parallel {
        records
            .par_iter()
            .map(|(idx, payload)| {
                evaluate_position(
                    *idx,
                    payload,
                    &fl_ev,
                    args.top_n,
                    args.t2_draw_limit,
                    args.source_candidate_top_k,
                )
            })
            .collect()
    } else {
        records
            .iter()
            .map(|(idx, payload)| {
                evaluate_position(
                    *idx,
                    payload,
                    &fl_ev,
                    args.top_n,
                    args.t2_draw_limit,
                    args.source_candidate_top_k,
                )
            })
            .collect()
    };
    results.sort_by_key(|result| result.record_index);
    for result in results {
        elapsed_ms.push(result.elapsed_ms);
        if !args.summary_only {
            serde_json::to_writer(&mut writer, &result)?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    if args.summary_only {
        elapsed_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        let avg = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len().max(1) as f64;
        let p50 = percentile(&elapsed_ms, 0.50);
        let p95 = percentile(&elapsed_ms, 0.95);
        let max = elapsed_ms.last().copied().unwrap_or(0.0);
        eprintln!(
            "positions={} total_ms={:.1} avg_ms={:.1} p50_ms={:.1} p95_ms={:.1} max_ms={:.1}",
            elapsed_ms.len(),
            total_ms,
            avg,
            p50,
            p95,
            max
        );
    }
    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(cards: &[&str]) -> Vec<String> {
        cards.iter().map(|card| (*card).to_string()).collect()
    }

    #[test]
    fn canonical_top_joker_downgrade_matches_python() {
        let board = Board {
            top: strings(&["Qh", "Qs", "X1"]),
            middle: strings(&["Kh", "Ks", "9d", "8c", "7h"]),
            bottom: strings(&["Ah", "Ad", "Ac", "5s", "4d"]),
        };
        let mut cache = RowEvalCache::default();
        let eval = board_eval(&board, &mut cache);

        assert!(!eval.busted);
        assert_eq!(eval.royalty, 7);
        assert_eq!(eval.fl_key, 14);
    }

    #[test]
    fn canonical_two_jokers_choose_quads() {
        let value = evaluate_hand(&strings(&["As", "2s", "2h", "X1", "X2"]), 5);
        assert_eq!(hand_category(value), 7);
        assert_eq!(bottom_royalty_from_value(value), 10);
    }

    #[test]
    fn joker_flush_and_quads_kickers_are_exhaustive() {
        assert_eq!(
            evaluate_hand(&strings(&["Kh", "Qh", "9h", "3h", "X1"]), 5),
            4_552_338,
        );
        assert_eq!(
            evaluate_hand(&strings(&["Qh", "Qs", "Qd", "X1", "X2"]), 5),
            5_970_375,
        );
    }

    #[test]
    fn top_trips_above_middle_trips_is_bust() {
        let board = Board {
            top: strings(&["Ah", "Ad", "Ac"]),
            middle: strings(&["2h", "2d", "2c", "Kh", "Qh"]),
            bottom: strings(&["3h", "3d", "3c", "3s", "4h"]),
        };
        let mut cache = RowEvalCache::default();
        assert!(board_eval(&board, &mut cache).busted);
    }

    #[test]
    fn t4_bb_candidate_mirrors_btn_best_joker_reply() {
        let hero = Board {
            top: strings(&["2c", "3c"]),
            middle: strings(&["6h", "6d", "7s", "8s", "Tc"]),
            bottom: strings(&["Jh", "Jd", "Qc", "Qd"]),
        };
        let opponent = Board {
            top: strings(&["Qh", "Qs"]),
            middle: strings(&["Kh", "Ks", "9d", "8c", "7h"]),
            bottom: strings(&["Ah", "Ad", "Ac", "5s"]),
        };
        let action = Action {
            placements: vec![("4c".to_string(), 0), ("Kc".to_string(), 2)],
            discard: "5d".to_string(),
        };
        let reply_draw = strings(&["X1", "4d", "2s", "9s"]);
        let mut live: HashSet<String> = HashSet::new();
        live.extend(hero.all_cards().cloned());
        live.extend(opponent.all_cards().cloned());
        live.extend(strings(&["4c", "Kc", "5d"]));
        live.extend(reply_draw.iter().cloned());
        let exclude: Vec<String> = FULL_DECK_CARDS
            .iter()
            .filter(|card| !live.contains(**card))
            .map(|card| (*card).to_string())
            .collect();
        let fl_ev = FlEv {
            values: BTreeMap::from([(14, 0.0), (15, 10.7), (16, 29.9), (17, 63.5)]),
        };

        let final_hero = apply_action(&hero, &action);
        let mut expected_cache = RowEvalCache::default();
        let hero_eval = board_eval(&final_hero, &mut expected_cache);
        let opponent_eval = board_eval(&opponent, &mut expected_cache);
        let outcome =
            exact_t4_candidate_metrics(&hero, &action, &opponent, &opponent_eval, &exclude, &fl_ev);

        assert_eq!(outcome.metrics.source, "exact_hu_response");
        assert_eq!(outcome.metrics.samples, 4);
        assert_eq!(outcome.metrics.enumerated_draws, Some(4));
        assert_eq!(outcome.metrics.remaining_deck_size, Some(4));
        assert_eq!(outcome.metrics.opponent_response, Some(true));
        assert_eq!(outcome.metrics.score, -20.5);
        assert_eq!(outcome.metrics.raw_score, -20.5);
        assert_eq!(outcome.metrics.royalty, hero_eval.royalty as f64);
        assert_eq!(
            outcome.metrics.bust_rate,
            if hero_eval.busted { 1.0 } else { 0.0 }
        );
    }

    fn test_fl_ev() -> FlEv {
        FlEv {
            values: BTreeMap::from([(14, 3.0), (15, 10.7), (16, 29.9), (17, 63.5)]),
        }
    }

    fn exclude_all_except(
        board: &Board,
        opponent: &Board,
        dealt: &[String],
        future_live: &[String],
    ) -> Vec<String> {
        let mut retained: HashSet<String> = HashSet::new();
        retained.extend(board.all_cards().cloned());
        retained.extend(opponent.all_cards().cloned());
        retained.extend(dealt.iter().cloned());
        retained.extend(future_live.iter().cloned());
        FULL_DECK_CARDS
            .iter()
            .filter(|card| !retained.contains(**card))
            .map(|card| (*card).to_string())
            .collect()
    }

    fn assert_close(label: &str, actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "{label}: actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_metrics_equivalent(actual: &Metrics, expected: &Metrics) {
        assert_close("score", actual.score, expected.score);
        assert_close("ev", actual.ev, expected.ev);
        assert_close("raw_score", actual.raw_score, expected.raw_score);
        assert_close("royalty", actual.royalty, expected.royalty);
        assert_close("bust_rate", actual.bust_rate, expected.bust_rate);
        assert_close("fl_rate", actual.fl_rate, expected.fl_rate);
        assert_eq!(actual.fl_type_rates.len(), expected.fl_type_rates.len());
        for (key, expected_value) in &expected.fl_type_rates {
            assert_close(
                key,
                *actual.fl_type_rates.get(key).expect("missing FL type rate"),
                *expected_value,
            );
        }
        assert_eq!(actual.samples, expected.samples);
        assert_eq!(actual.source, expected.source);
        assert_eq!(actual.forced_bust, expected.forced_bust);
        assert_eq!(actual.enumerated_draws, expected.enumerated_draws);
        assert_eq!(actual.remaining_deck_size, expected.remaining_deck_size);
        assert_eq!(actual.start_turn, expected.start_turn);
        assert_eq!(actual.opponent_response, expected.opponent_response);
    }

    fn assert_joint_t4_matches_reference(
        board: Board,
        opponent: Board,
        dealt: Vec<String>,
        future_live: Vec<String>,
    ) {
        let actions = get_turn_actions(&dealt, &board);
        assert!(
            actions.len() > 1,
            "fixture should exercise candidate ordering"
        );
        let exclude = exclude_all_except(&board, &opponent, &dealt, &future_live);
        let fl_ev = test_fl_ev();
        let mut opponent_cache = RowEvalCache::default();
        let opponent_eval = board_eval(&opponent, &mut opponent_cache);
        let reference: Vec<EvalOutcome> = actions
            .iter()
            .map(|action| {
                exact_t4_candidate_metrics(
                    &board,
                    action,
                    &opponent,
                    &opponent_eval,
                    &exclude,
                    &fl_ev,
                )
            })
            .collect();
        let joint = exact_t4_hu_candidate_metrics_joint(
            &board, &actions, &dealt, &opponent, &exclude, &fl_ev,
        );
        assert_eq!(joint.len(), reference.len());
        for (joint_outcome, reference_outcome) in joint.iter().zip(&reference) {
            assert_metrics_equivalent(&joint_outcome.metrics, &reference_outcome.metrics);
            assert_eq!(joint_outcome.metrics.source, "exact_hu_response");
            assert_eq!(
                joint_outcome.metrics.enumerated_draws,
                Some(combinations3_count(future_live.len(), 0))
            );
            assert_eq!(
                joint_outcome.metrics.remaining_deck_size,
                Some(future_live.len())
            );
            assert_eq!(joint_outcome.metrics.start_turn, Some(4));
            assert_eq!(joint_outcome.metrics.opponent_response, Some(true));
        }

        let mut reference_order: Vec<usize> = (0..actions.len()).collect();
        reference_order.sort_by(|left, right| {
            let quality_order =
                compare_metrics_quality(&reference[*right].metrics, &reference[*left].metrics);
            if quality_order == Ordering::Equal {
                action_payload_key(&action_to_payload(&actions[*left]))
                    .cmp(&action_payload_key(&action_to_payload(&actions[*right])))
            } else {
                quality_order
            }
        });

        let payload = PositionPayload {
            board: board.to_payload(),
            opponent_board: opponent.to_payload(),
            exclude,
            known_discards: Vec::new(),
            dealt: dealt.clone(),
            turn: Some(4),
            candidate_actions: Vec::new(),
            candidates: Vec::new(),
        };
        let result = evaluate_position(0, &payload, &fl_ev, actions.len(), 0, 0);
        assert_eq!(result.legal_actions, actions.len());
        assert_eq!(result.evaluated_actions, actions.len());
        assert_eq!(result.candidates.len(), actions.len());
        for (actual, expected_index) in result.candidates.iter().zip(reference_order) {
            assert_eq!(
                action_payload_key(&actual.action),
                action_payload_key(&action_to_payload(&actions[expected_index]))
            );
            assert_metrics_equivalent(&actual.metrics, &reference[expected_index].metrics);
        }
    }

    #[test]
    fn joint_t4_hu_matches_reference_for_all_natural_candidates() {
        let board = Board {
            top: strings(&["2c", "3c"]),
            middle: strings(&["6h", "6d", "7s", "8s", "Tc"]),
            bottom: strings(&["Jh", "Jd", "Qc", "Qd"]),
        };
        let opponent = Board {
            top: strings(&["Qh", "Qs"]),
            middle: strings(&["Kh", "Ks", "9d", "8c", "7h"]),
            bottom: strings(&["Ah", "Ad", "Ac", "5s"]),
        };
        assert_joint_t4_matches_reference(
            board,
            opponent,
            strings(&["4c", "Kc", "5d"]),
            strings(&["4d", "2s", "9s", "Th", "7c"]),
        );
    }

    #[test]
    fn joint_t4_hu_matches_reference_for_all_x1_x2_candidates() {
        let board = Board {
            top: strings(&["2c", "3c"]),
            middle: strings(&["6h", "6d", "7s", "8s", "X1"]),
            bottom: strings(&["Jh", "Jd", "Qc", "Qd"]),
        };
        let opponent = Board {
            top: strings(&["Qh", "Qs"]),
            middle: strings(&["Kh", "Ks", "9d", "8c", "7h"]),
            bottom: strings(&["Ah", "Ad", "Ac", "5s"]),
        };
        assert_joint_t4_matches_reference(
            board,
            opponent,
            strings(&["4c", "Kc", "5d"]),
            strings(&["X2", "4d", "2s", "9s", "7c"]),
        );
    }
}
