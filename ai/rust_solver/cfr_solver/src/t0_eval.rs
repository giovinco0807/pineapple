//! T0 EV Evaluator: Imperfect-Information Monte Carlo
//!
//! For a given 5-card hand, enumerate all valid T0 placements,
//! then evaluate each using nested Monte Carlo with imperfect information:
//! at each turn, only the current 3 dealt cards are visible; future cards
//! are estimated by sampling. Sample counts decrease at deeper levels
//! to keep computation tractable.

use ofc_core::{
    Card, create_deck, card_to_string,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    is_valid_placement, check_fl_entry,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use crate::action_gen::{Board, Row, generate_t0_actions, generate_turn_actions};
use crate::game_state::fl_chain_ev;

// ─── Compact Board for fast tree search (stack-allocated) ───

/// Fixed-size board representation to avoid heap allocations in the tree search.
#[derive(Clone, Copy)]
pub struct CompactBoard {
    top: [Card; 3],
    mid: [Card; 5],
    bot: [Card; 5],
    top_len: u8,
    mid_len: u8,
    bot_len: u8,
}

impl CompactBoard {
    fn from_board(board: &Board) -> Self {
        let mut cb = CompactBoard {
            top: [Card { rank: 0, suit: 0 }; 3],
            mid: [Card { rank: 0, suit: 0 }; 5],
            bot: [Card { rank: 0, suit: 0 }; 5],
            top_len: board.top.len() as u8,
            mid_len: board.middle.len() as u8,
            bot_len: board.bottom.len() as u8,
        };
        for (i, c) in board.top.iter().enumerate() { cb.top[i] = *c; }
        for (i, c) in board.middle.iter().enumerate() { cb.mid[i] = *c; }
        for (i, c) in board.bottom.iter().enumerate() { cb.bot[i] = *c; }
        cb
    }

    fn to_board(&self) -> Board {
        Board {
            top: self.top[..self.top_len as usize].to_vec(),
            middle: self.mid[..self.mid_len as usize].to_vec(),
            bottom: self.bot[..self.bot_len as usize].to_vec(),
        }
    }

    #[inline]
    fn push(&mut self, row: Row, card: Card) {
        match row {
            Row::Top => {
                self.top[self.top_len as usize] = card;
                self.top_len += 1;
            }
            Row::Middle => {
                self.mid[self.mid_len as usize] = card;
                self.mid_len += 1;
            }
            Row::Bottom => {
                self.bot[self.bot_len as usize] = card;
                self.bot_len += 1;
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.top_len == 3 && self.mid_len == 5 && self.bot_len == 5
    }
}

/// Score a completed board.
#[inline]
fn score_final_board(cb: &CompactBoard) -> f64 {
    let top = &cb.top[..cb.top_len as usize];
    let mid = &cb.mid[..cb.mid_len as usize];
    let bot = &cb.bot[..cb.bot_len as usize];

    if !is_valid_placement(top, mid, bot) {
        return -6.0;
    }

    let royalties = get_top_royalty(top)
        + get_middle_royalty(mid)
        + get_bottom_royalty(bot);

    let mut score = royalties as f64;

    let (fl_qualifies, fl_cards) = check_fl_entry(top);
    if fl_qualifies {
        score += fl_chain_ev(fl_cards);
    }

    score
}

/// Deals for T1-T4: 4 groups of 3 cards, stored as flat array.
type Deals = [[Card; 3]; 4];

/// Sample deals from remaining deck.
#[inline]
fn sample_deals(remaining: &[Card], rng: &mut SmallRng) -> Deals {
    let mut deck = [Card { rank: 0, suit: 0 }; 49];
    deck[..remaining.len()].copy_from_slice(remaining);
    let len = remaining.len();

    // Fisher-Yates shuffle (only first 12 positions needed)
    for i in 0..12.min(len) {
        let j = i + (rng.gen_range(0..len - i) as usize);
        deck.swap(i, j);
    }

    [
        [deck[0], deck[1], deck[2]],
        [deck[3], deck[4], deck[5]],
        [deck[6], deck[7], deck[8]],
        [deck[9], deck[10], deck[11]],
    ]
}

use rand::Rng;

/// Inline turn action generation to avoid Vec allocations.
/// For each turn: discard 1 of 3 cards, place 2 in valid rows.
/// Calls the callback with each valid (discard_idx, row0, row1) combo.
#[inline]
fn for_each_turn_action(
    hand: &[Card; 3],
    cb: &CompactBoard,
    mut callback: impl FnMut(usize, Row, Row),
) {
    let top_space = 3 - cb.top_len as usize;
    let mid_space = 5 - cb.mid_len as usize;
    let bot_space = 5 - cb.bot_len as usize;

    for discard_idx in 0..3usize {
        let kept = [
            (discard_idx + 1) % 3,
            (discard_idx + 2) % 3,
        ];

        for &row0 in &[Row::Top, Row::Middle, Row::Bottom] {
            for &row1 in &[Row::Top, Row::Middle, Row::Bottom] {
                // Check capacity
                let mut t = top_space;
                let mut m = mid_space;
                let mut b = bot_space;

                match row0 {
                    Row::Top => { if t == 0 { continue; } t -= 1; }
                    Row::Middle => { if m == 0 { continue; } m -= 1; }
                    Row::Bottom => { if b == 0 { continue; } b -= 1; }
                }
                match row1 {
                    Row::Top => { if t == 0 { continue; } }
                    Row::Middle => { if m == 0 { continue; } }
                    Row::Bottom => { if b == 0 { continue; } }
                }

                callback(discard_idx, row0, row1);
            }
        }
    }
}

/// Exhaustive search from a given turn onwards.
/// Returns the maximum achievable score with optimal play.
fn best_score_from_turn(cb: &CompactBoard, deals: &Deals, turn: usize) -> f64 {
    if turn >= 4 || cb.is_complete() {
        return score_final_board(cb);
    }

    let hand = &deals[turn];
    let mut best = f64::NEG_INFINITY;

    for_each_turn_action(hand, cb, |discard_idx, row0, row1| {
        let kept0 = (discard_idx + 1) % 3;
        let kept1 = (discard_idx + 2) % 3;

        let mut new_cb = *cb;
        new_cb.push(row0, hand[kept0]);
        new_cb.push(row1, hand[kept1]);

        let score = best_score_from_turn(&new_cb, deals, turn + 1);
        if score > best {
            best = score;
        }
    });

    if best == f64::NEG_INFINITY {
        score_final_board(cb)
    } else {
        best
    }
}

// ─── Imperfect-Information Monte Carlo Evaluation ───

/// Sample counts for nested MC at each depth level (main evaluation).
/// Index 0 = how many T2 futures to sample when evaluating a T1 action,
/// Index 1 = how many T3 futures when evaluating T2,
/// Index 2 = how many T4 futures when evaluating T3.
/// Higher = more accurate but slower. Total cost ∝ N0 × N1 × N2.
const NESTED_SAMPLES: [usize; 3] = [10, 6, 3];

/// Lighter nesting for FL stats rollouts.
/// Each rollout is cheap; statistical accuracy comes from many rollouts.
const ROLLOUT_NESTED: [usize; 3] = [10, 6, 3];

/// Evaluate the best action at a given turn with imperfect information.
/// `nesting` controls sample counts at each depth level.
/// Returns the score of the best action.
fn imperfect_best_score(
    cb: &CompactBoard,
    hand: &[Card; 3],
    remaining: &[Card],
    turn: usize,
    nesting: &[usize; 3],
    rng: &mut SmallRng,
) -> f64 {
    if turn >= 4 || cb.is_complete() {
        return score_final_board(cb);
    }

    let mut best = f64::NEG_INFINITY;

    for_each_turn_action(hand, cb, |discard_idx, row0, row1| {
        let kept0 = (discard_idx + 1) % 3;
        let kept1 = (discard_idx + 2) % 3;

        let mut new_cb = *cb;
        new_cb.push(row0, hand[kept0]);
        new_cb.push(row1, hand[kept1]);

        let score = if turn >= 3 || new_cb.is_complete() {
            score_final_board(&new_cb)
        } else {
            // remaining already excludes the 3 dealt cards for this turn;
            // the discard card is permanently out of play (not added back)
            estimate_future_ev(&new_cb, remaining, turn + 1, nesting, rng)
        };

        if score > best {
            best = score;
        }
    });

    if best == f64::NEG_INFINITY {
        score_final_board(cb)
    } else {
        best
    }
}

/// Estimate the expected value of a board position by sampling future hands.
fn estimate_future_ev(
    cb: &CompactBoard,
    remaining: &[Card],
    turn: usize,
    nesting: &[usize; 3],
    rng: &mut SmallRng,
) -> f64 {
    if turn >= 4 || cb.is_complete() {
        return score_final_board(cb);
    }

    let n_samples = nesting.get(turn.saturating_sub(1))
        .copied()
        .unwrap_or(5);

    let mut deck_buf = [Card { rank: 0, suit: 0 }; 54];
    let len = remaining.len();
    deck_buf[..len].copy_from_slice(remaining);

    let mut total = 0.0;
    for _ in 0..n_samples {
        // Partial Fisher-Yates: only need 3 cards
        for i in 0..3.min(len) {
            let j = i + rng.gen_range(0..len - i);
            deck_buf.swap(i, j);
        }

        let hand: [Card; 3] = [deck_buf[0], deck_buf[1], deck_buf[2]];
        let next_remaining: Vec<Card> = deck_buf[3..len].to_vec();

        total += imperfect_best_score(cb, &hand, &next_remaining, turn, nesting, rng);
    }

    total / n_samples as f64
}

/// Top-level imperfect-info evaluation for a single T0 placement + one T1 sample.
fn evaluate_t0_sample_imperfect(
    t0_board: &CompactBoard,
    remaining: &[Card],
    nesting: &[usize; 3],
    rng: &mut SmallRng,
) -> f64 {
    let len = remaining.len();
    let mut deck_buf = [Card { rank: 0, suit: 0 }; 54];
    deck_buf[..len].copy_from_slice(remaining);

    for i in 0..3.min(len) {
        let j = i + rng.gen_range(0..len - i);
        deck_buf.swap(i, j);
    }

    let t1_hand: [Card; 3] = [deck_buf[0], deck_buf[1], deck_buf[2]];
    let t1_remaining: Vec<Card> = deck_buf[3..len].to_vec();

    imperfect_best_score(t0_board, &t1_hand, &t1_remaining, 0, nesting, rng)
}

/// Play out a complete game (T1-T4) from a T0 board using imperfect-info decisions.
/// At each turn, samples a random hand, evaluates all actions via nested MC,
/// picks the best action, and continues. Returns (total_score, final_board).
fn rollout_imperfect(
    t0_board: &CompactBoard,
    remaining: &[Card],
    nesting: &[usize; 3],
    rng: &mut SmallRng,
) -> (f64, CompactBoard) {
    let mut cb = *t0_board;
    let mut deck_buf = [Card { rank: 0, suit: 0 }; 54];
    let mut len = remaining.len();
    deck_buf[..len].copy_from_slice(remaining);

    for turn in 0..4 {
        if cb.is_complete() { break; }
        if len < 3 { break; }

        // Deal 3 cards from shuffled remaining
        for i in 0..3 {
            let j = i + rng.gen_range(0..len - i);
            deck_buf.swap(i, j);
        }
        let hand: [Card; 3] = [deck_buf[0], deck_buf[1], deck_buf[2]];
        let turn_remaining = deck_buf[3..len].to_vec();

        // Use imperfect-info evaluation to pick best action
        let mut best_score = f64::NEG_INFINITY;
        let mut best_board = cb;

        for_each_turn_action(&hand, &cb, |discard_idx, row0, row1| {
            let kept0 = (discard_idx + 1) % 3;
            let kept1 = (discard_idx + 2) % 3;

            let mut new_cb = cb;
            new_cb.push(row0, hand[kept0]);
            new_cb.push(row1, hand[kept1]);

            let score = if turn >= 3 || new_cb.is_complete() {
                score_final_board(&new_cb)
            } else {
                estimate_future_ev(&new_cb, &turn_remaining, turn + 1, nesting, rng)
            };

            if score > best_score {
                best_score = score;
                best_board = new_cb;
            }
        });

        cb = best_board;

        // Shift remaining deck: remove dealt 3 cards
        deck_buf.copy_within(3..len, 0);
        len -= 3;
    }

    let final_score = score_final_board(&cb);
    (final_score, cb)
}


fn apply_t0_action(hand: &[Card], action: &crate::action_gen::Action) -> CompactBoard {
    let mut cb = CompactBoard {
        top: [Card { rank: 0, suit: 0 }; 3],
        mid: [Card { rank: 0, suit: 0 }; 5],
        bot: [Card { rank: 0, suit: 0 }; 5],
        top_len: 0,
        mid_len: 0,
        bot_len: 0,
    };

    for (i, &row) in action.row_assignments.iter().enumerate() {
        cb.push(row, hand[i]);
    }

    cb
}

/// Format a T0 placement for display.
fn format_placement(hand: &[Card], action: &crate::action_gen::Action) -> String {
    let mut top_cards = Vec::new();
    let mut mid_cards = Vec::new();
    let mut bot_cards = Vec::new();

    for (i, &row) in action.row_assignments.iter().enumerate() {
        let cs = card_to_string(&hand[i]);
        match row {
            Row::Top => top_cards.push(cs),
            Row::Middle => mid_cards.push(cs),
            Row::Bottom => bot_cards.push(cs),
        }
    }

    // Sort cards within each row for canonical matching with Python
    top_cards.sort();
    mid_cards.sort();
    bot_cards.sort();

    format!(
        "Top[{}] Mid[{}] Bot[{}]",
        top_cards.join(" "),
        mid_cards.join(" "),
        bot_cards.join(" "),
    )
}

/// Evaluate all T0 placements for a given hand.
/// Returns sorted (placement_description, EV) pairs.
pub fn evaluate_t0(hand: &[Card; 5], n_samples: usize, seed: u64, nesting: [usize; 3]) -> Vec<(String, f64)> {
    let board = Board::new();
    let actions = generate_t0_actions(hand, &board);

    println!("Hand: {}", hand.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" "));
    println!("Valid T0 placements: {}", actions.len());
    println!("Samples per placement: {} (imperfect-info MC, nested: {:?})", n_samples, nesting);
    println!();

    // Build remaining deck (54 cards = 52 + 2 Jokers, minus the 5 in hand)
    // Note: Both Jokers are identical (rank=0, suit=4), so we must remove exactly
    // one card per hand card, not all matching cards.
    let full_deck = create_deck(true); // 54 cards
    let mut remaining = full_deck.clone();
    for h in hand.iter() {
        if let Some(pos) = remaining.iter().position(|c| c.rank == h.rank && c.suit == h.suit) {
            remaining.remove(pos);
        }
    }
    // 54 - 5 = 49
    assert_eq!(remaining.len(), 49, "Should have 49 remaining cards (deck=54, hand=5)");

    // Flatten all work units: (action_idx, sample_idx) for maximum parallelism
    let total_tasks = actions.len() * n_samples;
    let n_actions = actions.len();
    println!("Total tasks: {} ({}×{})  using rayon parallel + imperfect-info MC", total_tasks, n_actions, n_samples);

    // Pre-compute T0 boards
    let t0_boards: Vec<CompactBoard> = actions.iter()
        .map(|action| apply_t0_action(hand, action))
        .collect();

    // Parallel eval with imperfect-info MC
    let scores: Vec<f64> = (0..total_tasks)
        .into_par_iter()
        .map(|task_id| {
            let action_idx = task_id / n_samples;
            let sample_idx = task_id % n_samples;

            let combined_seed = seed
                .wrapping_add(action_idx as u64 * 10_000_019)
                .wrapping_add(sample_idx as u64);
            let mut rng = SmallRng::seed_from_u64(combined_seed);
            evaluate_t0_sample_imperfect(&t0_boards[action_idx], &remaining, &nesting, &mut rng)
        })
        .collect();

    // Aggregate scores per action
    let results: Vec<(String, f64)> = actions.iter()
        .enumerate()
        .map(|(action_idx, action)| {
            let desc = format_placement(hand, action);
            let start = action_idx * n_samples;
            let end = start + n_samples;
            let total: f64 = scores[start..end].iter().sum();
            let ev = total / n_samples as f64;
            (desc, ev)
        })
        .collect();

    let mut sorted = results;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    sorted
}

// ─── Batch evaluation ───

use std::io::Write;

/// Quiet version of evaluate_t0 -- no stdout, just returns results.
/// Backward-compatible wrapper: evaluates all placements.
pub fn evaluate_t0_quiet(hand: &[Card; 5], n_samples: usize, seed: u64, nesting: &[usize; 3]) -> Vec<(String, f64)> {
    evaluate_t0_quiet_topk(hand, n_samples, seed, nesting, 0)
}

/// Two-pass top-K evaluation:
///   Pass 1: Screen ALL placements with cheap nesting=[1,1,1], samples=5
///   Pass 2: Only evaluate top-K placements with full nesting & samples
/// If top_k == 0, evaluates all placements (no screening).
pub fn evaluate_t0_quiet_topk(hand: &[Card; 5], n_samples: usize, seed: u64, nesting: &[usize; 3], top_k: usize) -> Vec<(String, f64)> {
    let board = Board::new();
    let actions = generate_t0_actions(hand, &board);

    let full_deck = create_deck(true); // 54 cards
    let mut remaining = full_deck.clone();
    for h in hand.iter() {
        if let Some(pos) = remaining.iter().position(|c| c.rank == h.rank && c.suit == h.suit) {
            remaining.remove(pos);
        }
    }
    if remaining.len() != 49 {
        return Vec::new();
    }

    let n_actions = actions.len();

    let t0_boards: Vec<CompactBoard> = actions.iter()
        .map(|action| apply_t0_action(hand, action))
        .collect();

    // Determine which action indices to deeply evaluate
    let deep_indices: Vec<usize> = if top_k > 0 && top_k < n_actions {
        // --- Pass 1: Cheap screening ---
        let screen_nesting: [usize; 3] = [1, 1, 1];
        let screen_samples: usize = 5;
        let screen_tasks = n_actions * screen_samples;

        let screen_scores: Vec<f64> = (0..screen_tasks)
            .into_par_iter()
            .map(|task_id| {
                let action_idx = task_id / screen_samples;
                let sample_idx = task_id % screen_samples;
                let combined_seed = seed
                    .wrapping_add(action_idx as u64 * 10_000_019)
                    .wrapping_add(sample_idx as u64)
                    .wrapping_add(77777); // offset to avoid seed collision with pass 2
                let mut rng = SmallRng::seed_from_u64(combined_seed);
                evaluate_t0_sample_imperfect(&t0_boards[action_idx], &remaining, &screen_nesting, &mut rng)
            })
            .collect();

        // Aggregate screening scores
        let mut screen_evs: Vec<(usize, f64)> = (0..n_actions)
            .map(|ai| {
                let start = ai * screen_samples;
                let end = start + screen_samples;
                let total: f64 = screen_scores[start..end].iter().sum();
                (ai, total / screen_samples as f64)
            })
            .collect();

        // Sort by EV descending, take top-K
        screen_evs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        screen_evs.iter().take(top_k).map(|(idx, _)| *idx).collect()
    } else {
        // No screening: evaluate all
        (0..n_actions).collect()
    };

    let n_deep = deep_indices.len();

    // --- Pass 2: Deep evaluation of selected placements ---
    let deep_tasks = n_deep * n_samples;
    let scores: Vec<f64> = (0..deep_tasks)
        .into_par_iter()
        .map(|task_id| {
            let deep_idx = task_id / n_samples;
            let sample_idx = task_id % n_samples;
            let action_idx = deep_indices[deep_idx];
            let combined_seed = seed
                .wrapping_add(action_idx as u64 * 10_000_019)
                .wrapping_add(sample_idx as u64);
            let mut rng = SmallRng::seed_from_u64(combined_seed);
            evaluate_t0_sample_imperfect(&t0_boards[action_idx], &remaining, nesting, &mut rng)
        })
        .collect();

    let results: Vec<(String, f64)> = deep_indices.iter()
        .enumerate()
        .map(|(deep_idx, &action_idx)| {
            let desc = format_placement(hand, &actions[action_idx]);
            let start = deep_idx * n_samples;
            let end = start + n_samples;
            let total: f64 = scores[start..end].iter().sum();
            let ev = total / n_samples as f64;
            (desc, ev)
        })
        .collect();

    let mut sorted = results;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted
}
/// Generate a random 5-card hand from the deck.
fn generate_random_hand(rng: &mut SmallRng) -> [Card; 5] {
    let mut deck = create_deck(true);
    deck.shuffle(rng);
    [deck[0], deck[1], deck[2], deck[3], deck[4]]
}

/// Classify hand type for analysis.
fn classify_hand(hand: &[Card; 5]) -> String {
    let joker_count = hand.iter().filter(|c| c.rank == 0).count();
    let mut ranks: Vec<u8> = hand.iter().filter(|c| c.rank > 0).map(|c| c.rank).collect();
    ranks.sort_unstable();

    // Count pairs
    let mut pairs = Vec::new();
    let mut trips = false;
    let mut i = 0;
    while i < ranks.len() {
        let mut count = 1;
        while i + count < ranks.len() && ranks[i] == ranks[i + count] {
            count += 1;
        }
        if count == 2 { pairs.push(ranks[i]); }
        if count >= 3 { trips = true; }
        i += count;
    }

    let has_ace = ranks.contains(&14);
    let suit_count = hand.iter().filter(|c| c.rank > 0).map(|c| c.suit).collect::<std::collections::HashSet<_>>().len();
    let max_rank = ranks.last().copied().unwrap_or(0);

    let mut parts = Vec::new();

    if joker_count > 0 { parts.push(format!("Jo×{}", joker_count)); }
    if trips { parts.push("Trips".to_string()); }
    else if pairs.len() == 2 { parts.push("2Pair".to_string()); }
    else if pairs.len() == 1 {
        let pr = pairs[0];
        let rank_char = ofc_core::rank_to_char(pr);
        parts.push(format!("Pair{}", rank_char));
    }
    else { parts.push("NoPair".to_string()); }

    if has_ace && pairs.iter().all(|&p| p != 14) { parts.push("Ace".to_string()); }
    if suit_count <= 2 && joker_count == 0 { parts.push("Suited".to_string()); }

    let high = ofc_core::rank_to_char(max_rank);
    parts.push(format!("Hi{}", high));

    parts.join("_")
}

/// Run batch evaluation: generate N random hands, evaluate each, save to JSONL.
/// Saves top-K placements (or all if top_k=0) with EVs for supervised learning.
/// Uses 2-pass screening when top_k > 0 to save compute.
/// Supports resumption by counting existing lines in the output file.
pub fn run_batch(n_hands: usize, n_samples: usize, output_path: &str, seed: u64, nesting: [usize; 3], top_k: usize) {
    use std::fs::{OpenOptions};
    use std::io::{BufRead, BufReader};
    use rand::seq::SliceRandom;

    // Check how many hands already completed (for resumption)
    let completed = if std::path::Path::new(output_path).exists() {
        let file = std::fs::File::open(output_path).unwrap();
        BufReader::new(file).lines().count()
    } else {
        0
    };

    if completed >= n_hands {
        println!("Already completed {} hands (target: {}). Nothing to do.", completed, n_hands);
        return;
    }

    let remaining_hands = n_hands - completed;
    println!("=== T0 Batch Evaluation (Imperfect-Info MC) ===");
    println!("Target: {} hands | Samples/hand: {} | Nesting: {:?}", n_hands, n_samples, nesting);
    if top_k > 0 {
        println!("Two-pass mode: screen all → deep eval top-{}", top_k);
    }
    println!("Output: {} | Format: {} placements per hand", output_path, if top_k > 0 { format!("top-{}", top_k) } else { "ALL".to_string() });
    if completed > 0 {
        println!("Resuming from hand #{} ({} already done)", completed + 1, completed);
    }
    println!();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Failed to open output file");

    let start = std::time::Instant::now();

    for hand_idx in completed..n_hands {
        let hand_seed = seed.wrapping_add(hand_idx as u64 * 999_999_937);
        let mut rng = SmallRng::seed_from_u64(hand_seed);
        let hand = generate_random_hand(&mut rng);

        let hand_str = hand.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" ");
        let hand_type = classify_hand(&hand);

        let eval_seed = seed.wrapping_add(hand_idx as u64 * 1_000_000_007);
        let results = evaluate_t0_quiet_topk(&hand, n_samples, eval_seed, &nesting, top_k);

        if results.is_empty() {
            continue;
        }

        // Build JSON with placements (sorted by EV desc)
        let all_placements: Vec<String> = results.iter().map(|(desc, ev)| {
            format!("{{\"p\":\"{}\",\"ev\":{:.3}}}", desc, ev)
        }).collect();

        let json = format!(
            "{{\"hand_idx\":{},\"hand\":\"{}\",\"type\":\"{}\",\"n_placements\":{},\"n_samples\":{},\"nesting\":\"{:?}\",\"top_k\":{},\"placements\":[{}]}}",
            hand_idx,
            hand_str,
            hand_type,
            results.len(),
            n_samples,
            nesting,
            top_k,
            all_placements.join(","),
        );

        writeln!(file, "{}", json).expect("Failed to write");
        file.flush().unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let done = hand_idx - completed + 1;
        let avg = elapsed / done as f64;
        let eta = avg * (remaining_hands - done) as f64;

        println!(
            "[{:>4}/{}] {} ({}) | {} placements | Best: {} EV:{:+.2} | {:.0}s/hand | ETA: {:.0}min",
            hand_idx + 1, n_hands,
            hand_str, hand_type,
            results.len(),
            results[0].0, results[0].1,
            avg, eta / 60.0,
        );
    }

    let total = start.elapsed().as_secs_f64();
    println!("\n=== Batch Complete ===");
    println!("Hands evaluated: {}", remaining_hands);
    println!("Total time: {:.1}min", total / 60.0);
    println!("Avg: {:.1}s/hand", total / remaining_hands as f64);
    println!("Output: {}", output_path);
}

/// Run filtered batch evaluation: read pre-filtered placements from Python JSON,
/// evaluate only those placements with MC, and save results to JSONL.
/// This is the same as run_batch but only evaluates the top-K placements
/// selected by the PolicyNet, enabling higher sample counts.
pub fn run_batch_filtered(input_path: &str, n_samples: usize, output_path: &str, seed: u64, nesting: [usize; 3]) {
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader};

    // Read pre-filtered JSON
    let input_data: String = std::fs::read_to_string(input_path)
        .expect("Failed to read input JSON");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&input_data)
        .expect("Failed to parse input JSON");

    let n_hands = entries.len();

    // Check how many hands already completed (for resumption)
    let completed = if std::path::Path::new(output_path).exists() {
        let file = File::open(output_path).unwrap();
        BufReader::new(file).lines().count()
    } else {
        0
    };

    if completed >= n_hands {
        println!("Already completed {} hands (target: {}). Nothing to do.", completed, n_hands);
        return;
    }

    let remaining_hands = n_hands - completed;
    println!("=== T0 Filtered Batch Evaluation (PolicyNet Pruned) ===");
    println!("Input: {} | Hands: {} | Samples/hand: {} | Nesting: {:?}", input_path, n_hands, n_samples, nesting);
    println!("Output: {} | Format: filtered placements per hand", output_path);
    if completed > 0 {
        println!("Resuming from hand #{} ({} already done)", completed + 1, completed);
    }
    println!();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Failed to open output file");

    let start = std::time::Instant::now();

    for hand_idx in completed..n_hands {
        let entry = &entries[hand_idx];
        let hand_str = entry["hand"].as_str().unwrap();
        let hand = match parse_hand(hand_str) {
            Some(h) => h,
            None => {
                eprintln!("Skipping invalid hand: {}", hand_str);
                continue;
            }
        };

        let hand_type = classify_hand(&hand);

        // Get the filtered placement descriptions from Python
        let filtered_descs: Vec<String> = entry["filtered_placements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        // Generate all T0 actions and filter to match
        let board = Board::new();
        let all_actions = generate_t0_actions(&hand, &board);

        // Build a set of filtered placement descriptions for fast lookup
        let filter_set: std::collections::HashSet<String> = filtered_descs.iter().cloned().collect();

        // Filter actions: keep only those whose format_placement matches
        let filtered_actions: Vec<_> = all_actions.iter()
            .filter(|action| {
                let desc = format_placement(&hand, action);
                filter_set.contains(&desc)
            })
            .collect();

        let n_filtered = filtered_actions.len();
        if n_filtered == 0 {
            eprintln!("[{}/{}] No matching placements for hand {}", hand_idx + 1, n_hands, hand_str);
            continue;
        }

        // Pre-compute T0 boards for filtered actions
        // Use position-based removal to handle duplicate Jokers correctly
        let full_deck = create_deck(true); // 54 cards
        let mut remaining = full_deck.clone();
        for h in hand.iter() {
            if let Some(pos) = remaining.iter().position(|c| c.rank == h.rank && c.suit == h.suit) {
                remaining.remove(pos);
            }
        }
        // 54 - 5 = 49
        if remaining.len() != 49 { continue; }

        let t0_boards: Vec<CompactBoard> = filtered_actions.iter()
            .map(|action| apply_t0_action(&hand, action))
            .collect();

        let eval_seed = seed.wrapping_add(hand_idx as u64 * 1_000_000_007);
        let total_tasks = n_filtered * n_samples;

        // Parallel MC evaluation (same as evaluate_t0_quiet)
        let scores: Vec<f64> = (0..total_tasks)
            .into_par_iter()
            .map(|task_id| {
                let action_idx = task_id / n_samples;
                let sample_idx = task_id % n_samples;
                let combined_seed = eval_seed
                    .wrapping_add(action_idx as u64 * 10_000_019)
                    .wrapping_add(sample_idx as u64);
                let mut rng = SmallRng::seed_from_u64(combined_seed);
                evaluate_t0_sample_imperfect(&t0_boards[action_idx], &remaining, &nesting, &mut rng)
            })
            .collect();

        // Aggregate results
        let mut results: Vec<(String, f64)> = filtered_actions.iter()
            .enumerate()
            .map(|(action_idx, action)| {
                let desc = format_placement(&hand, action);
                let start_i = action_idx * n_samples;
                let end_i = start_i + n_samples;
                let total: f64 = scores[start_i..end_i].iter().sum();
                let ev = total / n_samples as f64;
                (desc, ev)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build JSON output (same format as run_batch)
        let all_placements: Vec<String> = results.iter().map(|(desc, ev)| {
            format!("{{\"p\":\"{}\",\"ev\":{:.3}}}", desc, ev)
        }).collect();

        let json = format!(
            "{{\"hand_idx\":{},\"hand\":\"{}\",\"type\":\"{}\",\"n_placements\":{},\"n_samples\":{},\"nesting\":\"{:?}\",\"placements\":[{}]}}",
            hand_idx,
            hand_str,
            hand_type,
            results.len(),
            n_samples,
            nesting,
            all_placements.join(","),
        );

        writeln!(file, "{}", json).expect("Failed to write");
        file.flush().unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let done = hand_idx - completed + 1;
        let avg = elapsed / done as f64;
        let eta = avg * (remaining_hands - done) as f64;

        println!(
            "[{:>4}/{}] {} ({}) | {}/{} placements | Best: {} EV:{:+.2} | {:.0}s/hand | ETA: {:.0}min",
            hand_idx + 1, n_hands,
            hand_str, hand_type,
            n_filtered, all_actions.len(),
            results[0].0, results[0].1,
            avg, eta / 60.0,
        );
    }

    let total = start.elapsed().as_secs_f64();
    println!("\n=== Filtered Batch Complete ===");
    println!("Hands evaluated: {}", remaining_hands);
    println!("Total time: {:.1}min", total / 60.0);
    println!("Avg: {:.1}s/hand", total / remaining_hands as f64);
    println!("Output: {}", output_path);
}

// ─── Real-time Turn Evaluator (T1-T4) ───

/// Sample deals for remaining turns from the deck.
/// `n_future_turns`: how many turns remain after this one (0-3).
#[inline]
fn sample_future_deals(remaining: &[Card], n_future_turns: usize, rng: &mut SmallRng) -> [[Card; 3]; 3] {
    let mut deck = [Card { rank: 0, suit: 0 }; 46];
    let len = remaining.len().min(46);
    deck[..len].copy_from_slice(&remaining[..len]);

    let needed = n_future_turns * 3;
    for i in 0..needed.min(len) {
        let j = i + rng.gen_range(0..len - i);
        deck.swap(i, j);
    }

    let mut deals = [[Card { rank: 0, suit: 0 }; 3]; 3];
    for t in 0..n_future_turns.min(3) {
        deals[t] = [deck[t * 3], deck[t * 3 + 1], deck[t * 3 + 2]];
    }
    deals
}

/// Exhaustive search for future turns (after current action is applied).
fn best_score_future(cb: &CompactBoard, deals: &[[Card; 3]; 3], turns_remaining: usize, turn_offset: usize) -> f64 {
    if turns_remaining == 0 || cb.is_complete() {
        return score_final_board(cb);
    }

    let hand = &deals[turn_offset];
    let mut best = f64::NEG_INFINITY;

    for_each_turn_action(hand, cb, |discard_idx, row0, row1| {
        let kept0 = (discard_idx + 1) % 3;
        let kept1 = (discard_idx + 2) % 3;

        let mut new_cb = *cb;
        new_cb.push(row0, hand[kept0]);
        new_cb.push(row1, hand[kept1]);

        let score = best_score_future(&new_cb, deals, turns_remaining - 1, turn_offset + 1);
        if score > best {
            best = score;
        }
    });

    if best == f64::NEG_INFINITY { score_final_board(cb) } else { best }
}

/// Represents a turn action result.
pub struct TurnActionResult {
    pub discard: String,
    pub placement_desc: String,
    pub ev: f64,
}

/// Evaluate all actions at the current turn.
///
/// Arguments:
/// - `board`: current board state
/// - `hand`: the 3 cards dealt this turn
/// - `known_cards`: all cards already placed + the 3 in hand (to remove from deck)
/// - `turns_after`: how many turns remain after this one (T1ↁE, T2ↁE, T3ↁE, T4ↁE)
/// - `n_samples`: Monte Carlo samples for future turns
/// - `seed`: random seed
pub fn evaluate_turn(
    board: &CompactBoard,
    hand: &[Card; 3],
    known_cards: &[Card],
    turns_after: usize,
    n_samples: usize,
    seed: u64,
) -> Vec<TurnActionResult> {
    // Build remaining deck
    let full_deck = create_deck(true);
    let remaining: Vec<Card> = full_deck.iter()
        .filter(|c| !known_cards.iter().any(|k| k.rank == c.rank && k.suit == c.suit))
        .copied()
        .collect();

    // Enumerate all valid actions at this turn
    let mut actions: Vec<(usize, Row, Row)> = Vec::new();
    for_each_turn_action(hand, board, |d, r0, r1| {
        actions.push((d, r0, r1));
    });

    if turns_after == 0 {
        // T4: no future sampling needed, just score directly
        let results: Vec<TurnActionResult> = actions.iter().map(|&(discard_idx, row0, row1)| {
            let kept0 = (discard_idx + 1) % 3;
            let kept1 = (discard_idx + 2) % 3;

            let mut cb = *board;
            cb.push(row0, hand[kept0]);
            cb.push(row1, hand[kept1]);

            let ev = score_final_board(&cb);
            let discard_str = card_to_string(&hand[discard_idx]);
            let desc = format!(
                "{}→{:?}, {}→{:?}",
                card_to_string(&hand[kept0]), row0,
                card_to_string(&hand[kept1]), row1,
            );

            TurnActionResult { discard: discard_str, placement_desc: desc, ev }
        }).collect();

        return sort_results(results);
    }

    // For T1-T3: parallel Monte Carlo over (action, sample)
    let n_actions = actions.len();
    let total_tasks = n_actions * n_samples;

    let scores: Vec<f64> = (0..total_tasks)
        .into_par_iter()
        .map(|task_id| {
            let action_idx = task_id / n_samples;
            let sample_idx = task_id % n_samples;
            let (discard_idx, row0, row1) = actions[action_idx];

            let kept0 = (discard_idx + 1) % 3;
            let kept1 = (discard_idx + 2) % 3;

            let mut cb = *board;
            cb.push(row0, hand[kept0]);
            cb.push(row1, hand[kept1]);

            let combined_seed = seed
                .wrapping_add(action_idx as u64 * 10_000_019)
                .wrapping_add(sample_idx as u64);
            let mut rng = SmallRng::seed_from_u64(combined_seed);
            let deals = sample_future_deals(&remaining, turns_after, &mut rng);
            best_score_future(&cb, &deals, turns_after, 0)
        })
        .collect();

    let results: Vec<TurnActionResult> = actions.iter().enumerate().map(|(i, &(discard_idx, row0, row1))| {
        let kept0 = (discard_idx + 1) % 3;
        let kept1 = (discard_idx + 2) % 3;

        let start = i * n_samples;
        let end = start + n_samples;
        let total: f64 = scores[start..end].iter().sum();
        let ev = total / n_samples as f64;

        let discard_str = card_to_string(&hand[discard_idx]);
        let desc = format!(
            "{}→{:?}, {}→{:?}",
            card_to_string(&hand[kept0]), row0,
            card_to_string(&hand[kept1]), row1,
        );

        TurnActionResult { discard: discard_str, placement_desc: desc, ev }
    }).collect();

    sort_results(results)
}

fn sort_results(mut results: Vec<TurnActionResult>) -> Vec<TurnActionResult> {
    results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());
    results
}

/// Build a CompactBoard from board description strings.
pub fn parse_board(top_str: &str, mid_str: &str, bot_str: &str) -> Option<CompactBoard> {
    let mut cb = CompactBoard {
        top: [Card { rank: 0, suit: 0 }; 3],
        mid: [Card { rank: 0, suit: 0 }; 5],
        bot: [Card { rank: 0, suit: 0 }; 5],
        top_len: 0,
        mid_len: 0,
        bot_len: 0,
    };

    for token in top_str.split_whitespace() {
        if token == "-" || token.is_empty() { continue; }
        let card = parse_card(token)?;
        if cb.top_len >= 3 { return None; }
        cb.top[cb.top_len as usize] = card;
        cb.top_len += 1;
    }
    for token in mid_str.split_whitespace() {
        if token == "-" || token.is_empty() { continue; }
        let card = parse_card(token)?;
        if cb.mid_len >= 5 { return None; }
        cb.mid[cb.mid_len as usize] = card;
        cb.mid_len += 1;
    }
    for token in bot_str.split_whitespace() {
        if token == "-" || token.is_empty() { continue; }
        let card = parse_card(token)?;
        if cb.bot_len >= 5 { return None; }
        cb.bot[cb.bot_len as usize] = card;
        cb.bot_len += 1;
    }

    Some(cb)
}

/// Collect all cards currently on the board.
pub fn board_cards(cb: &CompactBoard) -> Vec<Card> {
    let mut cards = Vec::new();
    for i in 0..cb.top_len as usize { cards.push(cb.top[i]); }
    for i in 0..cb.mid_len as usize { cards.push(cb.mid[i]); }
    for i in 0..cb.bot_len as usize { cards.push(cb.bot[i]); }
    cards
}

// ─── Card parsing ───

/// Parse a card string like "Ad", "8c", "Ts", "Jo"
pub fn parse_card(s: &str) -> Option<Card> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("jo") || s.eq_ignore_ascii_case("joker") {
        return Some(Card { rank: 0, suit: 4 });
    }

    if s.len() != 2 {
        return None;
    }

    let chars: Vec<char> = s.chars().collect();
    let rank = match chars[0].to_ascii_uppercase() {
        '2' => 2,
        '3' => 3,
        '4' => 4,
        '5' => 5,
        '6' => 6,
        '7' => 7,
        '8' => 8,
        '9' => 9,
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        _ => return None,
    };
    let suit = match chars[1].to_ascii_lowercase() {
        's' => 0,
        'h' => 1,
        'd' => 2,
        'c' => 3,
        _ => return None,
    };

    Some(Card { rank, suit })
}

/// Parse a hand string like "Ad 8c 4s 3d 2s"
pub fn parse_hand(s: &str) -> Option<[Card; 5]> {
    let cards: Vec<Card> = s.split_whitespace()
        .filter_map(|token| parse_card(token))
        .collect();

    if cards.len() != 5 {
        return None;
    }

    Some([cards[0], cards[1], cards[2], cards[3], cards[4]])
}

// ─── FL Statistics Measurement ───

/// Score a completed board WITHOUT FL bonus (raw royalties only).
#[inline]
fn score_raw_royalty(cb: &CompactBoard) -> f64 {
    let top = &cb.top[..cb.top_len as usize];
    let mid = &cb.mid[..cb.mid_len as usize];
    let bot = &cb.bot[..cb.bot_len as usize];

    if !is_valid_placement(top, mid, bot) {
        return -6.0;
    }

    (get_top_royalty(top) + get_middle_royalty(mid) + get_bottom_royalty(bot)) as f64
}

/// Exhaustive search returning both score AND final board.
/// Uses the FULL score (including FL bonus) for decision-making,
/// but returns the completed board for post-hoc analysis.
fn best_board_from_turn(cb: &CompactBoard, deals: &Deals, turn: usize) -> (f64, CompactBoard) {
    if turn >= 4 || cb.is_complete() {
        return (score_final_board(cb), *cb);
    }

    let hand = &deals[turn];
    let mut best_score = f64::NEG_INFINITY;
    let mut best_board = *cb;

    for_each_turn_action(hand, cb, |discard_idx, row0, row1| {
        let kept0 = (discard_idx + 1) % 3;
        let kept1 = (discard_idx + 2) % 3;

        let mut new_cb = *cb;
        new_cb.push(row0, hand[kept0]);
        new_cb.push(row1, hand[kept1]);

        let (score, board) = best_board_from_turn(&new_cb, deals, turn + 1);
        if score > best_score {
            best_score = score;
            best_board = board;
        }
    });

    if best_score == f64::NEG_INFINITY {
        (score_final_board(cb), *cb)
    } else {
        (best_score, best_board)
    }
}

/// Per-sample result for FL stats.
struct SampleResult {
    raw_royalty: f64,
    total_score: f64,
    fl_entered: bool,
    fl_type: u8, // 0=none, 14=QQ, 15=KK, 16=AA, 17=Trips
    is_bust: bool,
}

/// Measure FL statistics across N random hands.
/// Phase 1: Fast perfect-info MC for T0 action selection.
/// Phase 2: Imperfect-info rollouts for stats (bust rate, FL rate, royalties).
pub fn measure_fl_stats(n_hands: usize, n_samples: usize, seed: u64) {
    println!("=== FL Statistics Measurement (Imperfect-Info Rollouts) ===");
    println!("Hands: {} | Rollouts/hand: {} | Rollout nesting: {:?}",
        n_hands, n_samples, ROLLOUT_NESTED);
    println!();

    let start = std::time::Instant::now();

    // Running totals
    let mut sum_raw = 0.0f64;
    let mut sum_total = 0.0f64;
    let mut count = 0u64;
    let mut fl_total = 0u64;
    let mut fl_qq_total = 0u64;
    let mut fl_kk_total = 0u64;
    let mut fl_aa_total = 0u64;
    let mut fl_trips_total = 0u64;
    let mut bust_total = 0u64;

    for hand_idx in 0..n_hands {
        let hand_seed = seed.wrapping_add(hand_idx as u64 * 999_999_937);
        let mut rng = SmallRng::seed_from_u64(hand_seed);
        let hand = generate_random_hand(&mut rng);

        let board = crate::action_gen::Board::new();
        let actions = crate::action_gen::generate_t0_actions(&hand, &board);

        let full_deck = create_deck(true); // 54 cards
        let mut remaining = full_deck.clone();
        for h in hand.iter() {
            if let Some(pos) = remaining.iter().position(|c| c.rank == h.rank && c.suit == h.suit) {
                remaining.remove(pos);
            }
        }
        if remaining.len() != 49 { continue; }

        let t0_boards: Vec<CompactBoard> = actions.iter()
            .map(|action| apply_t0_action(&hand, action))
            .collect();
        let n_actions = actions.len();
        let eval_seed = seed.wrapping_add(hand_idx as u64 * 1_000_000_007);

        // Phase 1: Fast T0 selection using perfect-info MC (100 samples)
        let select_samples = 100usize.min(n_samples);
        let task_count = n_actions * select_samples;
        let scores: Vec<f64> = (0..task_count)
            .into_par_iter()
            .map(|task_id| {
                let action_idx = task_id / select_samples;
                let sample_idx = task_id % select_samples;
                let combined_seed = eval_seed
                    .wrapping_add(action_idx as u64 * 10_000_019)
                    .wrapping_add(sample_idx as u64);
                let mut rng = SmallRng::seed_from_u64(combined_seed);
                let deals = sample_deals(&remaining, &mut rng);
                best_score_from_turn(&t0_boards[action_idx], &deals, 0)
            })
            .collect();

        let best_action_idx = (0..n_actions)
            .max_by(|&a, &b| {
                let avg_a: f64 = scores[a * select_samples..(a + 1) * select_samples].iter().sum::<f64>() / select_samples as f64;
                let avg_b: f64 = scores[b * select_samples..(b + 1) * select_samples].iter().sum::<f64>() / select_samples as f64;
                avg_a.partial_cmp(&avg_b).unwrap()
            })
            .unwrap_or(0);

        // Phase 2: Collect FL stats using imperfect-info rollouts
        let results: Vec<SampleResult> = (0..n_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let combined_seed = eval_seed
                    .wrapping_add(best_action_idx as u64 * 10_000_019)
                    .wrapping_add(sample_idx as u64 + 1_000_000);
                let mut rng = SmallRng::seed_from_u64(combined_seed);
                let (total, final_board) = rollout_imperfect(
                    &t0_boards[best_action_idx], &remaining, &ROLLOUT_NESTED, &mut rng
                );
                let raw = score_raw_royalty(&final_board);
                let top = &final_board.top[..final_board.top_len as usize];
                let (fl_entered, fl_cards) = check_fl_entry(top);
                SampleResult {
                    raw_royalty: raw,
                    total_score: total,
                    fl_entered,
                    fl_type: if fl_entered { fl_cards } else { 0 },
                    is_bust: raw <= -5.0,
                }
            })
            .collect();

        // Accumulate into running totals
        let mut hand_fl = 0u64;
        for r in &results {
            sum_raw += r.raw_royalty;
            sum_total += r.total_score;
            count += 1;
            if r.fl_entered {
                fl_total += 1;
                hand_fl += 1;
                match r.fl_type {
                    14 => fl_qq_total += 1,
                    15 => fl_kk_total += 1,
                    16 => fl_aa_total += 1,
                    17 => fl_trips_total += 1,
                    _ => {}
                }
            }
            if r.is_bust { bust_total += 1; }
        }

        // Per-hand stats for progress display
        let hand_raw: f64 = results.iter().map(|r| r.raw_royalty).sum::<f64>() / n_samples as f64;
        let hand_total: f64 = results.iter().map(|r| r.total_score).sum::<f64>() / n_samples as f64;
        let hand_fl_rate = hand_fl as f64 / n_samples as f64;
        let hand_bust_rate = results.iter().filter(|r| r.is_bust).count() as f64 / n_samples as f64;
        let min_raw = results.iter().map(|r| r.raw_royalty).fold(f64::INFINITY, f64::min);
        let min_total = results.iter().map(|r| r.total_score).fold(f64::INFINITY, f64::min);
        let hand_str = hand.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" ");
        let hand_type = classify_hand(&hand);

        let elapsed = start.elapsed().as_secs_f64();
        let avg_per_hand = elapsed / (hand_idx + 1) as f64;
        let eta = avg_per_hand * (n_hands - hand_idx - 1) as f64;

        println!(
            "[{:>3}/{}] {} ({}) R_raw={:.2} R_total={:.2} FL={:.1}% bust={:.1}% min_raw={:.1} min_total={:.1} | {:.0}s/h ETA {:.0}min",
            hand_idx + 1, n_hands,
            hand_str, hand_type,
            hand_raw, hand_total, hand_fl_rate * 100.0, hand_bust_rate * 100.0,
            min_raw, min_total,
            avg_per_hand, eta / 60.0,
        );
    }

    // Final aggregation
    let n = count as f64;
    let r_n = sum_raw / n;
    let r_total = sum_total / n;
    let p_fl = fl_total as f64 / n;
    let bust_rate = bust_total as f64 / n;

    println!("\n========================================");
    println!("  FL Statistics  EFinal Results");
    println!("========================================");
    println!("Total samples: {}", count);
    println!();
    println!("R_N (raw royalty, no FL bonus): {:.3}", r_n);
    println!("R_total (with FL bonus):        {:.3}", r_total);
    println!("FL contribution:                {:.3}", r_total - r_n);
    println!("Bust rate:                      {:.2}%", bust_rate * 100.0);
    println!();
    println!("p_FL (FL entry rate):  {:.4} ({:.2}%)", p_fl, p_fl * 100.0);
    println!("  QQ:    {:.4} ({:.2}%)", fl_qq_total as f64 / n, fl_qq_total as f64 / n * 100.0);
    println!("  KK:    {:.4} ({:.2}%)", fl_kk_total as f64 / n, fl_kk_total as f64 / n * 100.0);
    println!("  AA:    {:.4} ({:.2}%)", fl_aa_total as f64 / n, fl_aa_total as f64 / n * 100.0);
    println!("  Trips: {:.4} ({:.2}%)", fl_trips_total as f64 / n, fl_trips_total as f64 / n * 100.0);
    println!();

    // Known FL data (from KI)
    let fl_data = [
        (14u8, "QQ   ", 15.34f64, 0.3844f64),
        (15,   "KK   ", 19.30, 0.4869),
        (16,   "AA   ", 23.80, 0.6409),
        (17,   "Trips", 28.40, 0.776),
    ];

    println!("=== Corrected FL Chain EV (Delta) ===");
    println!("Formula: Delta = (R_FL - R_N) / (1 - S + p_FL)");
    println!("R_N = {:.3}, p_FL = {:.4}", r_n, p_fl);
    println!();
    println!("{:<8} {:>8} {:>8} {:>10} {:>10} {:>10}", "Entry", "R_FL", "S", "Old EV", "New EV", "Change");
    println!("{}", "-".repeat(60));

    let old_evs = [14.0f64, 27.9, 52.4, 104.5];

    for (i, &(_fl_cards, name, r_fl, s)) in fl_data.iter().enumerate() {
        let denom = 1.0 - s + p_fl;
        let delta = (r_fl - r_n) / denom;
        let old = old_evs[i];
        let change_pct = (delta - old) / old * 100.0;
        println!("{:<8} {:>8.2} {:>7.2}% {:>10.2} {:>10.2} {:>+9.1}%",
            name, r_fl, s * 100.0, old, delta, change_pct);
    }
    println!();

    let total_time = start.elapsed().as_secs_f64();
    println!("Total time: {:.1}min", total_time / 60.0);
}
