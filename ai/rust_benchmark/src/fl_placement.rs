//! FL (Fantasyland) Placement Solver
//!
//! Handles FL card placement for two scenarios:
//! - Both FL: maximize royalty + FL stay bonus
//! - FL vs Normal: candidates within 4 royalty of max, scored vs opponent board

use ofc_core::*;
use rayon::prelude::*;
use itertools::Itertools;
use crate::types::*;
use crate::game_engine::*;
use std::collections::HashMap;

/// FL placement candidate (internal)
struct FlCandidate {
    board: Board,
    discards: Vec<CardIdx>,
    royalty: i32,
    can_stay: bool,
}

/// Solve FL placement for "both FL" case.
/// Returns the placement with max (royalty + stay_bonus).
pub fn solve_fl_both(cards: &[CardIdx]) -> Option<(Board, Vec<CardIdx>)> {
    let n = cards.len();
    if n < 14 || n > 17 { return None; }
    let card_objs: Vec<Card> = cards.iter().map(|&idx| cardidx_to_card(idx)).collect();

    let bot_combos: Vec<Vec<usize>> = (0..n).combinations(5).collect();

    bot_combos.into_par_iter()
        .filter_map(|bot_idx| {
            best_for_bot(cards, &card_objs, n, &bot_idx)
        })
        .max_by(|a, b| {
            let sa = a.royalty as f64 + if a.can_stay { 30.0 } else { 0.0 };
            let sb = b.royalty as f64 + if b.can_stay { 30.0 } else { 0.0 };
            sa.partial_cmp(&sb).unwrap()
        })
        .map(|c| (c.board, c.discards))
}

/// Solve FL placement against a completed normal opponent board.
/// 1. Find max royalty among all valid placements
/// 2. Collect candidates within 4 royalty of max
/// 3. Score each vs opponent (lines + scoop + royalties + FL stay EV)
/// 4. Return the best
pub fn solve_fl_vs_normal(
    cards: &[CardIdx],
    opp_board: &Board,
    fl_ev: &HashMap<u8, f64>,
) -> Option<(Board, Vec<CardIdx>)> {
    let n = cards.len();
    if n < 14 || n > 17 { return None; }
    let card_objs: Vec<Card> = cards.iter().map(|&idx| cardidx_to_card(idx)).collect();
    let current_fl_cards = n as u8;

    let bot_combos: Vec<Vec<usize>> = (0..n).combinations(5).collect();

    // Pass 1: find max royalty (no Board construction)
    let max_royalty: i32 = bot_combos.par_iter()
        .filter_map(|bot_idx| max_royalty_for_bot(&card_objs, n, bot_idx))
        .max()
        .unwrap_or(0);

    let threshold = max_royalty - 4;

    // Pass 2: collect candidates with royalty >= threshold
    let candidates: Vec<FlCandidate> = bot_combos.into_par_iter()
        .flat_map(|bot_idx| {
            candidates_for_bot(cards, &card_objs, n, &bot_idx, threshold)
        })
        .collect();

    if candidates.is_empty() { return None; }

    // Score each candidate vs opponent, pick best
    candidates.into_iter()
        .max_by(|a, b| {
            let sa = score_vs_opp(&a.board, opp_board, fl_ev, a.can_stay, current_fl_cards);
            let sb = score_vs_opp(&b.board, opp_board, fl_ev, b.can_stay, current_fl_cards);
            sa.partial_cmp(&sb).unwrap()
        })
        .map(|c| (c.board, c.discards))
}

// ============================================================
//  Internal helpers
// ============================================================

/// Find the single best candidate for a given bottom selection (by royalty + stay bonus)
fn best_for_bot(
    cards: &[CardIdx],
    card_objs: &[Card],
    n: usize,
    bot_idx: &[usize],
) -> Option<FlCandidate> {
    let remaining: Vec<usize> = (0..n).filter(|i| !bot_idx.contains(i)).collect();
    let bot_c: Vec<Card> = bot_idx.iter().map(|&i| card_objs[i]).collect();
    let mut best: Option<FlCandidate> = None;
    let mut best_score = f64::NEG_INFINITY;

    for mid_idx in remaining.iter().copied().combinations(5) {
        let mid_c: Vec<Card> = mid_idx.iter().map(|&i| card_objs[i]).collect();

        // Early pruning: if mid has no joker and bot < mid, all tops will bust
        let mid_has_joker = mid_c.iter().any(|c| c.is_joker());
        if !mid_has_joker && compare_5_hands(&bot_c, &mid_c) < 0 { continue; }

        let after_mid: Vec<usize> = remaining.iter().copied()
            .filter(|i| !mid_idx.contains(i)).collect();

        for top_idx in after_mid.iter().copied().combinations(3) {
            let disc_idx: Vec<usize> = after_mid.iter().copied()
                .filter(|i| !top_idx.contains(i)).collect();

            if disc_idx.iter().any(|&i| card_objs[i].is_joker()) { continue; }

            let top_c: Vec<Card> = top_idx.iter().map(|&i| card_objs[i]).collect();

            let eval = evaluate_board_with_joker_constraint(&top_c, &mid_c, &bot_c);
            if eval.busted { continue; }

            let royalty = get_top_royalty(&eval.top)
                + get_middle_royalty(&eval.mid)
                + get_bottom_royalty(&eval.bot);
            let can_stay = check_fl_stay(&eval.top, &eval.mid, &eval.bot);
            let score = royalty as f64 + if can_stay { 30.0 } else { 0.0 };

            if score > best_score {
                best_score = score;
                let mut board = Board::new();
                for &i in bot_idx { board.place_mut(cards[i], ROW_BOT); }
                for &i in &mid_idx { board.place_mut(cards[i], ROW_MID); }
                for &i in &top_idx { board.place_mut(cards[i], ROW_TOP); }
                best = Some(FlCandidate {
                    board,
                    discards: disc_idx.iter().map(|&i| cards[i]).collect(),
                    royalty,
                    can_stay,
                });
            }
        }
    }

    best
}

/// Find max royalty for a given bottom (lightweight, no Board construction)
fn max_royalty_for_bot(card_objs: &[Card], n: usize, bot_idx: &[usize]) -> Option<i32> {
    let remaining: Vec<usize> = (0..n).filter(|i| !bot_idx.contains(i)).collect();
    let bot_c: Vec<Card> = bot_idx.iter().map(|&i| card_objs[i]).collect();
    let mut max_r = i32::MIN;

    for mid_idx in remaining.iter().copied().combinations(5) {
        let mid_c: Vec<Card> = mid_idx.iter().map(|&i| card_objs[i]).collect();

        // Early pruning: if mid has no joker and bot < mid, skip
        let mid_has_joker = mid_c.iter().any(|c| c.is_joker());
        if !mid_has_joker && compare_5_hands(&bot_c, &mid_c) < 0 { continue; }

        let after_mid: Vec<usize> = remaining.iter().copied()
            .filter(|i| !mid_idx.contains(i)).collect();

        for top_idx in after_mid.iter().copied().combinations(3) {
            let disc_idx: Vec<usize> = after_mid.iter().copied()
                .filter(|i| !top_idx.contains(i)).collect();

            if disc_idx.iter().any(|&i| card_objs[i].is_joker()) { continue; }

            let top_c: Vec<Card> = top_idx.iter().map(|&i| card_objs[i]).collect();

            let eval = evaluate_board_with_joker_constraint(&top_c, &mid_c, &bot_c);
            if eval.busted { continue; }

            let royalty = get_top_royalty(&eval.top)
                + get_middle_royalty(&eval.mid)
                + get_bottom_royalty(&eval.bot);
            if royalty > max_r { max_r = royalty; }
        }
    }

    if max_r > i32::MIN { Some(max_r) } else { None }
}

/// Collect candidates with royalty >= threshold for a given bottom
fn candidates_for_bot(
    cards: &[CardIdx],
    card_objs: &[Card],
    n: usize,
    bot_idx: &[usize],
    threshold: i32,
) -> Vec<FlCandidate> {
    let remaining: Vec<usize> = (0..n).filter(|i| !bot_idx.contains(i)).collect();
    let bot_c: Vec<Card> = bot_idx.iter().map(|&i| card_objs[i]).collect();
    let mut result = Vec::new();

    for mid_idx in remaining.iter().copied().combinations(5) {
        let mid_c: Vec<Card> = mid_idx.iter().map(|&i| card_objs[i]).collect();

        // Early pruning: if mid has no joker and bot < mid, skip
        let mid_has_joker = mid_c.iter().any(|c| c.is_joker());
        if !mid_has_joker && compare_5_hands(&bot_c, &mid_c) < 0 { continue; }

        let after_mid: Vec<usize> = remaining.iter().copied()
            .filter(|i| !mid_idx.contains(i)).collect();

        for top_idx in after_mid.iter().copied().combinations(3) {
            let disc_idx: Vec<usize> = after_mid.iter().copied()
                .filter(|i| !top_idx.contains(i)).collect();

            if disc_idx.iter().any(|&i| card_objs[i].is_joker()) { continue; }

            let top_c: Vec<Card> = top_idx.iter().map(|&i| card_objs[i]).collect();

            let eval = evaluate_board_with_joker_constraint(&top_c, &mid_c, &bot_c);
            if eval.busted { continue; }

            let royalty = get_top_royalty(&eval.top)
                + get_middle_royalty(&eval.mid)
                + get_bottom_royalty(&eval.bot);
            let can_stay = check_fl_stay(&eval.top, &eval.mid, &eval.bot);

            // Keep if within royalty window OR has FL stay (always valuable)
            if royalty < threshold && !can_stay { continue; }

            let mut board = Board::new();
            for &i in bot_idx { board.place_mut(cards[i], ROW_BOT); }
            for &i in &mid_idx { board.place_mut(cards[i], ROW_MID); }
            for &i in &top_idx { board.place_mut(cards[i], ROW_TOP); }

            result.push(FlCandidate {
                board,
                discards: disc_idx.iter().map(|&i| cards[i]).collect(),
                royalty,
                can_stay,
            });
        }
    }

    result
}

/// Score FL placement vs completed opponent board
/// Returns: raw_score (lines + scoop + royalty diff) + FL stay chain EV
fn score_vs_opp(
    fl_board: &Board,
    opp_board: &Board,
    fl_ev: &HashMap<u8, f64>,
    can_stay: bool,
    current_fl_cards: u8,
) -> f64 {
    let boards = [fl_board.clone(), opp_board.clone()];
    let result = compute_game_result(&boards, fl_ev);
    let mut score = result.raw_score;

    // Add FL chain EV if this placement achieves FL stay
    if can_stay {
        // If FL entry (QQ+ on top), use that card count; otherwise keep current
        let next_cards = if result.fl_entry[0] && !result.busted[0] {
            result.fl_card_count[0]
        } else {
            current_fl_cards
        };
        score += fl_ev.get(&next_cards).copied().unwrap_or(0.0);
    }

    score
}
