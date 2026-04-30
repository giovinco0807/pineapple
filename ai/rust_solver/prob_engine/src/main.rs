//! OFC Probability Calculation Engine
//!
//! Computes exact hand probability distributions for each row
//! by enumerating all possible card combinations from the remaining deck.
//! Supports bust approximation, FL rate, royalty estimation, and EV calculation.

use std::collections::HashSet;

use clap::Parser;
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::{SeedableRng, RngCore};
use rand::rngs::StdRng;
use rayon::prelude::*;
use ofc_core::{
    Card, HandRank3,
    evaluate_5_card, evaluate_3_card,
    create_deck, rank_to_char, SUIT_CHARS,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fl_entry, count_ranks, compare_5_hands,
};
use serde::Serialize;

// ============================================================
//  Hand value: combined (category, strength) into single u32
// ============================================================

fn hand_value_5(cards: &[Card]) -> u32 {
    let (rank, strength) = evaluate_5_card(cards);
    (rank as u32) * 1_000_000 + strength
}

fn hand_value_3(cards: &[Card]) -> u32 {
    let (rank, strength) = evaluate_3_card(cards);
    (rank as u32) * 1_000_000 + strength
}

/// Map 3-card hand to a value comparable with 5-card hand_value_5.
/// Trips maps to 2.5 on the category scale (between TwoPair=2 and Trips=3).
fn top_comparable_value_5(cards: &[Card]) -> u32 {
    let (rank, strength) = evaluate_3_card(cards);
    match rank {
        HandRank3::HighCard => strength,               // category 0
        HandRank3::OnePair => 1_000_000 + strength,    // category 1
        HandRank3::Trips   => 2_500_000 + strength,    // between TwoPair(2) and Trips(3)
    }
}

// ============================================================
//  Histogram bins
// ============================================================

const BINS_5: usize = 20;
const BINS_3: usize = 10;

fn bin_5(value: u32) -> usize {
    let category = (value / 1_000_000) as usize;
    let strength = value % 1_000_000;
    let sub = if strength >= 500_000 { 1 } else { 0 };
    (category * 2 + sub).min(BINS_5 - 1)
}

fn bin_3(value: u32) -> usize {
    let category = (value / 1_000_000) as usize;
    let strength = value % 1_000_000;
    match category {
        0 => {
            if strength < 250_000 { 0 }
            else if strength < 500_000 { 1 }
            else if strength < 750_000 { 2 }
            else { 3 }
        }
        1 => {
            if strength < 250_000 { 4 }
            else if strength < 500_000 { 5 }
            else if strength < 750_000 { 6 }
            else { 7 }
        }
        2 => {
            if strength < 500_000 { 8 } else { 9 }
        }
        _ => 9,
    }
}

// ============================================================
//  Fine-grained 100-bin histogram (unified scale)
// ============================================================

const FINE_BINS: usize = 100;

/// Extract primary rank from 3-card hand for sub-bin ordering.
/// Returns the most significant rank for hand comparison:
///   Trips → trips rank, OnePair → pair rank, HighCard → highest card.
fn primary_rank_3(cards: &[Card]) -> u8 {
    let rc = count_ranks(cards);
    for r in (2..=14).rev() {
        if rc[r] >= 3 { return r as u8; }
    }
    for r in (2..=14).rev() {
        if rc[r] >= 2 { return r as u8; }
    }
    for r in (2..=14).rev() {
        if rc[r] >= 1 { return r as u8; }
    }
    0
}

/// Extract primary rank from 5-card hand for sub-bin ordering.
/// Quads rank > Trips rank > Pair rank > High card.
fn primary_rank_5(cards: &[Card]) -> u8 {
    let rc = count_ranks(cards);
    for r in (2..=14).rev() {
        if rc[r] >= 4 { return r as u8; }
    }
    for r in (2..=14).rev() {
        if rc[r] >= 3 { return r as u8; }
    }
    for r in (2..=14).rev() {
        if rc[r] >= 2 { return r as u8; }
    }
    for r in (2..=14).rev() {
        if rc[r] >= 1 { return r as u8; }
    }
    0
}

/// Map (category, primary_rank) to fine bin 0-99.
/// 10 sub-bins per category, subdivided by primary rank (2-14 → 0-9).
fn fine_bin_from(category: usize, primary_rank: u8) -> usize {
    let sub = ((primary_rank as usize).saturating_sub(2) * 10 / 13).min(9);
    (category * 10 + sub).min(FINE_BINS - 1)
}

/// Compute fine bin for a 3-card top hand on the comparable 5-card scale.
fn fine_bin_top(cards: &[Card]) -> usize {
    let (rank, _) = evaluate_3_card(cards);
    let pr = primary_rank_3(cards);
    match rank {
        HandRank3::HighCard => fine_bin_from(0, pr),
        HandRank3::OnePair => fine_bin_from(1, pr),
        HandRank3::Trips   => {
            // Map to category 2.5: bins 25-29 (between TwoPair and Trips)
            let sub = ((pr as usize).saturating_sub(2) * 5 / 13).min(4);
            25 + sub
        }
    }
}

/// Compute fine bin for a 5-card hand.
fn fine_bin_5card(cards: &[Card]) -> usize {
    let (rank, _) = evaluate_5_card(cards);
    let pr = primary_rank_5(cards);
    fine_bin_from(rank as usize, pr)
}

// ============================================================
//  Row probability distribution
// ============================================================

#[derive(Debug, Clone, Serialize)]
struct RowDistribution {
    histogram: Vec<f64>,
    categories: Vec<(String, f64)>,
    mean_value: f64,
    std_value: f64,
    n_combinations: usize,
    expected_royalty: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparable_hist_5: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fl_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fl_qq_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fl_kk_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fl_aa_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fl_trips_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fl_comparable_hist_5: Option<Vec<f64>>,
    /// Fine-grained 100-bin histogram on unified comparable scale
    fine_hist: Vec<f64>,
    /// Per-bin expected royalty (E[royalty | bin])
    #[serde(skip_serializing)]
    bin_royalty: Vec<f64>,
    /// Per-bin FL rate (P(FL | bin)), top row only
    #[serde(skip_serializing)]
    bin_fl_rate: Vec<f64>,
    /// Per-bin FL EV (E[FL_EV | bin, FL]), top row only
    #[serde(skip_serializing)]
    bin_fl_ev: Vec<f64>,
}

fn compute_top_distribution(existing: &[Card], remaining_deck: &[Card]) -> RowDistribution {
    let slots = 3 - existing.len();
    assert!(slots >= 1 && slots <= 3, "Top needs 1-3 slots, got {}", slots);

    let mut histogram = vec![0u64; BINS_3];
    let mut comparable_hist = vec![0u64; BINS_5];
    let mut category_counts = vec![0u64; 3];
    let mut sum_value = 0u64;
    let mut sum_value_sq = 0u64;
    let mut sum_royalty = 0i64;
    let mut fl_count = 0u64;
    let mut fl_qq = 0u64;
    let mut fl_kk = 0u64;
    let mut fl_aa = 0u64;
    let mut fl_trips = 0u64;
    let mut fl_comparable_hist = vec![0u64; BINS_5];
    let mut n = 0u64;

    // Fine bin accumulators
    let mut fine_counts = vec![0u64; FINE_BINS];
    let mut fine_royalty_sum = vec![0i64; FINE_BINS];
    let mut fine_fl_count = vec![0u64; FINE_BINS];
    let mut fine_fl_ev_sum = vec![0.0f64; FINE_BINS];

    for combo in remaining_deck.iter().combinations(slots) {
        let mut hand = Vec::with_capacity(3);
        hand.extend_from_slice(existing);
        for &card in &combo {
            hand.push(*card);
        }

        let value = hand_value_3(&hand);
        let comp_value = top_comparable_value_5(&hand);
        let category = (value / 1_000_000) as usize;

        histogram[bin_3(value)] += 1;
        comparable_hist[bin_5(comp_value)] += 1;
        category_counts[category.min(2)] += 1;
        sum_value += value as u64;
        sum_value_sq += (value as u64) * (value as u64);
        let royalty = get_top_royalty(&hand);
        sum_royalty += royalty as i64;

        // Fine bin accumulation (use primary rank, not raw strength)
        let fbin = fine_bin_top(&hand);
        fine_counts[fbin] += 1;
        fine_royalty_sum[fbin] += royalty as i64;

        let (fl, fl_cards) = check_fl_entry(&hand);
        if fl {
            fl_count += 1;
            fl_comparable_hist[bin_5(comp_value)] += 1;
            fine_fl_count[fbin] += 1;
            fine_fl_ev_sum[fbin] += fl_ev_for_cards(fl_cards);
            match fl_cards {
                14 => fl_qq += 1,
                15 => fl_kk += 1,
                16 => fl_aa += 1,
                17 => fl_trips += 1,
                _ => {}
            }
        }
        n += 1;
    }

    let nf = n as f64;
    let mean = sum_value as f64 / nf;
    let variance = (sum_value_sq as f64 / nf) - mean * mean;
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    let cat_names = ["high_card", "pair", "trips"];
    let categories: Vec<(String, f64)> = cat_names.iter().enumerate()
        .map(|(i, name)| (name.to_string(), category_counts[i] as f64 / nf))
        .collect();

    let fl_comp = if fl_count > 0 {
        let fl_nf = fl_count as f64;
        Some(fl_comparable_hist.iter().map(|&c| c as f64 / fl_nf).collect())
    } else {
        None
    };

    // Normalize fine bin data
    let fine_hist: Vec<f64> = fine_counts.iter().map(|&c| c as f64 / nf).collect();
    let bin_royalty: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_counts[i] > 0 { fine_royalty_sum[i] as f64 / fine_counts[i] as f64 } else { 0.0 }
    }).collect();
    let bin_fl_rate: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_counts[i] > 0 { fine_fl_count[i] as f64 / fine_counts[i] as f64 } else { 0.0 }
    }).collect();
    let bin_fl_ev: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_fl_count[i] > 0 { fine_fl_ev_sum[i] / fine_fl_count[i] as f64 } else { 0.0 }
    }).collect();

    RowDistribution {
        histogram: histogram.iter().map(|&c| c as f64 / nf).collect(),
        categories,
        mean_value: mean,
        std_value: std,
        n_combinations: n as usize,
        expected_royalty: sum_royalty as f64 / nf,
        comparable_hist_5: Some(comparable_hist.iter().map(|&c| c as f64 / nf).collect()),
        fl_rate: Some(fl_count as f64 / nf),
        fl_qq_rate: Some(fl_qq as f64 / nf),
        fl_kk_rate: Some(fl_kk as f64 / nf),
        fl_aa_rate: Some(fl_aa as f64 / nf),
        fl_trips_rate: Some(fl_trips as f64 / nf),
        fl_comparable_hist_5: fl_comp,
        fine_hist, bin_royalty, bin_fl_rate, bin_fl_ev,
    }
}

fn compute_5card_distribution(existing: &[Card], remaining_deck: &[Card], is_middle: bool) -> RowDistribution {
    let slots = 5 - existing.len();
    assert!(slots >= 1 && slots <= 5, "Row needs 1-5 slots, got {}", slots);

    let mut histogram = vec![0u64; BINS_5];
    let mut category_counts = vec![0u64; 10];
    let mut sum_value = 0u64;
    let mut sum_value_sq = 0u64;
    let mut sum_royalty = 0i64;
    let mut n = 0u64;

    // Fine bin accumulators
    let mut fine_counts = vec![0u64; FINE_BINS];
    let mut fine_royalty_sum = vec![0i64; FINE_BINS];

    for combo in remaining_deck.iter().combinations(slots) {
        let mut hand = Vec::with_capacity(5);
        hand.extend_from_slice(existing);
        for &card in &combo {
            hand.push(*card);
        }

        let value = hand_value_5(&hand);
        let category = (value / 1_000_000) as usize;

        histogram[bin_5(value)] += 1;
        category_counts[category.min(9)] += 1;
        sum_value += value as u64;
        sum_value_sq += (value as u64) * (value as u64);

        let royalty = if is_middle {
            get_middle_royalty(&hand)
        } else {
            get_bottom_royalty(&hand)
        };
        sum_royalty += royalty as i64;

        // Fine bin accumulation (use primary rank, not raw strength)
        let fbin = fine_bin_5card(&hand);
        fine_counts[fbin] += 1;
        fine_royalty_sum[fbin] += royalty as i64;

        n += 1;
    }

    let nf = n as f64;
    let mean = sum_value as f64 / nf;
    let variance = (sum_value_sq as f64 / nf) - mean * mean;
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    let cat_names = [
        "high_card", "one_pair", "two_pair", "trips",
        "straight", "flush", "full_house", "quads",
        "straight_flush", "royal_flush",
    ];
    let categories: Vec<(String, f64)> = cat_names.iter().enumerate()
        .map(|(i, name)| (name.to_string(), category_counts[i] as f64 / nf))
        .collect();

    // Normalize fine bin data
    let fine_hist: Vec<f64> = fine_counts.iter().map(|&c| c as f64 / nf).collect();
    let bin_royalty: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_counts[i] > 0 { fine_royalty_sum[i] as f64 / fine_counts[i] as f64 } else { 0.0 }
    }).collect();

    RowDistribution {
        histogram: histogram.iter().map(|&c| c as f64 / nf).collect(),
        categories,
        mean_value: mean,
        std_value: std,
        n_combinations: n as usize,
        expected_royalty: sum_royalty as f64 / nf,
        comparable_hist_5: None,
        fl_rate: None, fl_qq_rate: None, fl_kk_rate: None,
        fl_aa_rate: None, fl_trips_rate: None,
        fl_comparable_hist_5: None,
        fine_hist, bin_royalty,
        bin_fl_rate: vec![0.0; FINE_BINS],
        bin_fl_ev: vec![0.0; FINE_BINS],
    }
}

// ============================================================
//  FL EV constants
// ============================================================

fn fl_ev_for_cards(cards: u8) -> f64 {
    match cards {
        14 => 14.0,
        15 => 27.9,
        16 => 52.4,
        17 => 104.5,
        _  => 0.0,
    }
}

// ============================================================
//  Joint evaluation via triple loop (Top × Mid × Bot)
// ============================================================

struct JointStats {
    bust_prob: f64,
    bust_top_mid: f64,   // marginal P(top > mid), diagnostic
    bust_mid_bot: f64,   // marginal P(mid > bot), diagnostic
    fl_rate: f64,         // P(FL and not bust)
    fl_ev_contribution: f64,
    expected_royalty: f64, // unconditional E[royalty] (for display)
    ev: f64,
}

/// Compute bust/FL/EV via exact triple loop over fine histogram bins.
/// Correctly handles the constraint that the SAME mid must satisfy top <= mid AND mid <= bot.
fn compute_joint_stats(
    top: &RowDistribution,
    mid: &RowDistribution,
    bot: &RowDistribution,
) -> JointStats {
    let mut not_bust = 0.0;
    let mut ev_sum = 0.0;      // royalty + FL bonus for non-bust outcomes
    let mut fl_sum = 0.0;      // P(FL and not bust)
    let mut fl_ev_sum = 0.0;   // FL EV contribution

    for t in 0..FINE_BINS {
        let pt = top.fine_hist[t];
        if pt == 0.0 { continue; }
        for m in t..FINE_BINS {
            let pm = mid.fine_hist[m];
            if pm == 0.0 { continue; }
            let p_tm = if m > t { 1.0 } else { 0.5 }; // same bin = 50% tie
            let ptm = pt * pm * p_tm;

            for b in m..FINE_BINS {
                let pb = bot.fine_hist[b];
                if pb == 0.0 { continue; }
                let p_mb = if b > m { 1.0 } else { 0.5 };
                let p = ptm * pb * p_mb;

                not_bust += p;
                let royalty = top.bin_royalty[t] + mid.bin_royalty[m] + bot.bin_royalty[b];
                let fl_bonus = top.bin_fl_rate[t] * top.bin_fl_ev[t];
                ev_sum += p * (royalty + fl_bonus);
                fl_sum += p * top.bin_fl_rate[t];
                fl_ev_sum += p * fl_bonus;
            }
        }
    }

    let bust_prob = (1.0 - not_bust).clamp(0.0, 1.0);
    let ev = ev_sum - bust_prob * 6.0;

    // Marginal diagnostics (from fine histograms)
    let bust_top_mid = prob_a_gt_b_fine(&top.fine_hist, &mid.fine_hist);
    let bust_mid_bot = prob_a_gt_b_fine(&mid.fine_hist, &bot.fine_hist);

    let expected_royalty = top.expected_royalty + mid.expected_royalty + bot.expected_royalty;

    JointStats {
        bust_prob, bust_top_mid, bust_mid_bot,
        fl_rate: fl_sum, fl_ev_contribution: fl_ev_sum,
        expected_royalty, ev,
    }
}

/// P(A > B) from fine histograms of any equal length.
fn prob_a_gt_b_fine(hist_a: &[f64], hist_b: &[f64]) -> f64 {
    let n = hist_a.len();
    assert_eq!(n, hist_b.len());
    let mut p = 0.0;
    for i in 0..n {
        if hist_a[i] == 0.0 { continue; }
        let mut b_lower = 0.0;
        for j in 0..i {
            b_lower += hist_b[j];
        }
        p += hist_a[i] * b_lower;
        p += hist_a[i] * hist_b[i] * 0.5;
    }
    p
}

// ============================================================
//  Board evaluation
// ============================================================

#[derive(Debug, Clone, Serialize)]
struct BoardEvaluation {
    top: RowDistribution,
    mid: RowDistribution,
    bot: RowDistribution,
    bust_prob: f64,
    bust_top_mid: f64,
    bust_mid_bot: f64,
    fl_rate: f64,
    fl_ev_contribution: f64,
    expected_royalty: f64,
    ev: f64,
}

fn make_deterministic_top(cards: &[Card]) -> RowDistribution {
    let value = hand_value_3(cards);
    let comp_value = top_comparable_value_5(cards);
    let royalty = get_top_royalty(cards) as f64;
    let (fl, fl_cards) = check_fl_entry(cards);
    let cat_idx = (value / 1_000_000) as usize;

    let mut hist = vec![0.0; BINS_3];
    hist[bin_3(value)] = 1.0;
    let mut comp_hist = vec![0.0; BINS_5];
    comp_hist[bin_5(comp_value)] = 1.0;

    let cat_names = ["high_card", "pair", "trips"];
    let categories: Vec<(String, f64)> = cat_names.iter().enumerate()
        .map(|(i, name)| (name.to_string(), if i == cat_idx { 1.0 } else { 0.0 }))
        .collect();

    // Fine bin data (use primary rank for proper ordering)
    let fbin = fine_bin_top(cards);
    let mut fine_hist = vec![0.0; FINE_BINS];
    fine_hist[fbin] = 1.0;
    let mut bin_royalty = vec![0.0; FINE_BINS];
    bin_royalty[fbin] = royalty;
    let mut bin_fl_rate = vec![0.0; FINE_BINS];
    let mut bin_fl_ev = vec![0.0; FINE_BINS];
    if fl {
        bin_fl_rate[fbin] = 1.0;
        bin_fl_ev[fbin] = fl_ev_for_cards(fl_cards);
    }

    RowDistribution {
        histogram: hist, categories,
        mean_value: value as f64, std_value: 0.0,
        n_combinations: 1, expected_royalty: royalty,
        comparable_hist_5: Some(comp_hist.clone()),
        fl_rate: Some(if fl { 1.0 } else { 0.0 }),
        fl_qq_rate: Some(if fl && fl_cards == 14 { 1.0 } else { 0.0 }),
        fl_kk_rate: Some(if fl && fl_cards == 15 { 1.0 } else { 0.0 }),
        fl_aa_rate: Some(if fl && fl_cards == 16 { 1.0 } else { 0.0 }),
        fl_trips_rate: Some(if fl && fl_cards == 17 { 1.0 } else { 0.0 }),
        fl_comparable_hist_5: if fl { Some(comp_hist) } else { None },
        fine_hist, bin_royalty, bin_fl_rate, bin_fl_ev,
    }
}

fn make_deterministic_5(cards: &[Card], is_middle: bool) -> RowDistribution {
    let value = hand_value_5(cards);
    let royalty = if is_middle { get_middle_royalty(cards) } else { get_bottom_royalty(cards) } as f64;
    let cat_idx = (value / 1_000_000) as usize;

    let mut hist = vec![0.0; BINS_5];
    hist[bin_5(value)] = 1.0;

    let cat_names = [
        "high_card", "one_pair", "two_pair", "trips",
        "straight", "flush", "full_house", "quads",
        "straight_flush", "royal_flush",
    ];
    let categories: Vec<(String, f64)> = cat_names.iter().enumerate()
        .map(|(i, name)| (name.to_string(), if i == cat_idx { 1.0 } else { 0.0 }))
        .collect();

    // Fine bin data (use primary rank for proper ordering)
    let fbin = fine_bin_5card(cards);
    let mut fine_hist = vec![0.0; FINE_BINS];
    fine_hist[fbin] = 1.0;
    let mut bin_royalty = vec![0.0; FINE_BINS];
    bin_royalty[fbin] = royalty;

    RowDistribution {
        histogram: hist, categories,
        mean_value: value as f64, std_value: 0.0,
        n_combinations: 1, expected_royalty: royalty,
        comparable_hist_5: None,
        fl_rate: None, fl_qq_rate: None, fl_kk_rate: None,
        fl_aa_rate: None, fl_trips_rate: None,
        fl_comparable_hist_5: None,
        fine_hist, bin_royalty,
        bin_fl_rate: vec![0.0; FINE_BINS],
        bin_fl_ev: vec![0.0; FINE_BINS],
    }
}

/// Sampled 5-card distribution: random N combinations instead of exhaustive.
/// Used for MC internal evaluation when slots >= 3.
fn compute_5card_distribution_sampled(
    existing: &[Card], remaining_deck: &[Card], is_middle: bool,
    n_samples: usize, rng: &mut StdRng,
) -> RowDistribution {
    let slots = 5 - existing.len();

    let mut fine_counts = vec![0u64; FINE_BINS];
    let mut fine_royalty_sum = vec![0i64; FINE_BINS];
    let mut n = 0u64;
    let mut sum_royalty = 0i64;

    let deck_len = remaining_deck.len();
    let mut indices: Vec<usize> = (0..deck_len).collect();

    for _ in 0..n_samples {
        // Fisher-Yates partial shuffle for `slots` elements
        for i in 0..slots {
            let j = i + (rng.next_u64() as usize % (deck_len - i));
            indices.swap(i, j);
        }

        let mut hand = Vec::with_capacity(5);
        hand.extend_from_slice(existing);
        for i in 0..slots {
            hand.push(remaining_deck[indices[i]]);
        }

        let royalty = if is_middle {
            get_middle_royalty(&hand)
        } else {
            get_bottom_royalty(&hand)
        };
        sum_royalty += royalty as i64;

        let fbin = fine_bin_5card(&hand);
        fine_counts[fbin] += 1;
        fine_royalty_sum[fbin] += royalty as i64;
        n += 1;
    }

    let nf = n as f64;
    let fine_hist: Vec<f64> = fine_counts.iter().map(|&c| c as f64 / nf).collect();
    let bin_royalty: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_counts[i] > 0 { fine_royalty_sum[i] as f64 / fine_counts[i] as f64 } else { 0.0 }
    }).collect();

    RowDistribution {
        histogram: vec![], categories: vec![],
        mean_value: 0.0, std_value: 0.0,
        n_combinations: n as usize,
        expected_royalty: sum_royalty as f64 / nf,
        comparable_hist_5: None,
        fl_rate: None, fl_qq_rate: None, fl_kk_rate: None,
        fl_aa_rate: None, fl_trips_rate: None,
        fl_comparable_hist_5: None,
        fine_hist, bin_royalty,
        bin_fl_rate: vec![0.0; FINE_BINS],
        bin_fl_ev: vec![0.0; FINE_BINS],
    }
}

/// Sampled top distribution for slots >= 2.
fn compute_top_distribution_sampled(
    existing: &[Card], remaining_deck: &[Card],
    n_samples: usize, rng: &mut StdRng,
) -> RowDistribution {
    let slots = 3 - existing.len();

    let mut fine_counts = vec![0u64; FINE_BINS];
    let mut fine_royalty_sum = vec![0i64; FINE_BINS];
    let mut fine_fl_count = vec![0u64; FINE_BINS];
    let mut fine_fl_ev_sum = vec![0.0f64; FINE_BINS];
    let mut n = 0u64;
    let mut sum_royalty = 0i64;

    let deck_len = remaining_deck.len();
    let mut indices: Vec<usize> = (0..deck_len).collect();

    for _ in 0..n_samples {
        for i in 0..slots {
            let j = i + (rng.next_u64() as usize % (deck_len - i));
            indices.swap(i, j);
        }

        let mut hand = Vec::with_capacity(3);
        hand.extend_from_slice(existing);
        for i in 0..slots {
            hand.push(remaining_deck[indices[i]]);
        }

        let royalty = get_top_royalty(&hand);
        sum_royalty += royalty as i64;
        let fbin = fine_bin_top(&hand);
        fine_counts[fbin] += 1;
        fine_royalty_sum[fbin] += royalty as i64;

        let (fl, fl_cards) = check_fl_entry(&hand);
        if fl {
            fine_fl_count[fbin] += 1;
            fine_fl_ev_sum[fbin] += fl_ev_for_cards(fl_cards);
        }
        n += 1;
    }

    let nf = n as f64;
    let fine_hist: Vec<f64> = fine_counts.iter().map(|&c| c as f64 / nf).collect();
    let bin_royalty: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_counts[i] > 0 { fine_royalty_sum[i] as f64 / fine_counts[i] as f64 } else { 0.0 }
    }).collect();
    let bin_fl_rate: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_counts[i] > 0 { fine_fl_count[i] as f64 / fine_counts[i] as f64 } else { 0.0 }
    }).collect();
    let bin_fl_ev: Vec<f64> = (0..FINE_BINS).map(|i| {
        if fine_fl_count[i] > 0 { fine_fl_ev_sum[i] / fine_fl_count[i] as f64 } else { 0.0 }
    }).collect();

    RowDistribution {
        histogram: vec![], categories: vec![],
        mean_value: 0.0, std_value: 0.0,
        n_combinations: n as usize,
        expected_royalty: sum_royalty as f64 / nf,
        comparable_hist_5: None,
        fl_rate: None, fl_qq_rate: None, fl_kk_rate: None,
        fl_aa_rate: None, fl_trips_rate: None,
        fl_comparable_hist_5: None,
        fine_hist, bin_royalty, bin_fl_rate, bin_fl_ev,
    }
}

/// Fast board evaluation using sampling for rows with many empty slots.
/// Used internally by MC simulation for T1-T3 candidate comparison.
fn evaluate_board_fast(
    top_cards: &[Card], mid_cards: &[Card], bot_cards: &[Card],
    remaining_deck: &[Card],
    rng: &mut StdRng,
) -> f64 {
    let n_samples = 1000;

    let top_dist = if top_cards.len() >= 3 {
        make_deterministic_top(top_cards)
    } else if 3 - top_cards.len() <= 1 {
        compute_top_distribution(top_cards, remaining_deck)
    } else {
        compute_top_distribution_sampled(top_cards, remaining_deck, n_samples, rng)
    };

    let mid_dist = if mid_cards.len() >= 5 {
        make_deterministic_5(mid_cards, true)
    } else if 5 - mid_cards.len() <= 2 {
        compute_5card_distribution(mid_cards, remaining_deck, true)
    } else {
        compute_5card_distribution_sampled(mid_cards, remaining_deck, true, n_samples, rng)
    };

    let bot_dist = if bot_cards.len() >= 5 {
        make_deterministic_5(bot_cards, false)
    } else if 5 - bot_cards.len() <= 2 {
        compute_5card_distribution(bot_cards, remaining_deck, false)
    } else {
        compute_5card_distribution_sampled(bot_cards, remaining_deck, false, n_samples, rng)
    };

    let joint = compute_joint_stats(&top_dist, &mid_dist, &bot_dist);
    joint.ev
}

fn evaluate_board(
    top_cards: &[Card], mid_cards: &[Card], bot_cards: &[Card],
    remaining_deck: &[Card],
) -> BoardEvaluation {
    let top_dist = if top_cards.len() < 3 {
        compute_top_distribution(top_cards, remaining_deck)
    } else {
        make_deterministic_top(top_cards)
    };

    let mid_dist = if mid_cards.len() < 5 {
        compute_5card_distribution(mid_cards, remaining_deck, true)
    } else {
        make_deterministic_5(mid_cards, true)
    };

    let bot_dist = if bot_cards.len() < 5 {
        compute_5card_distribution(bot_cards, remaining_deck, false)
    } else {
        make_deterministic_5(bot_cards, false)
    };

    // Joint triple-loop evaluation
    let joint = compute_joint_stats(&top_dist, &mid_dist, &bot_dist);

    BoardEvaluation {
        top: top_dist,
        mid: mid_dist,
        bot: bot_dist,
        bust_prob: joint.bust_prob,
        bust_top_mid: joint.bust_top_mid,
        bust_mid_bot: joint.bust_mid_bot,
        fl_rate: joint.fl_rate,
        fl_ev_contribution: joint.fl_ev_contribution,
        expected_royalty: joint.expected_royalty,
        ev: joint.ev,
    }
}

// ============================================================
//  Candidate generation (T1-T4: 3 cards → place 2, discard 1)
// ============================================================

#[derive(Debug, Clone)]
struct Candidate {
    placements: Vec<(Card, usize)>,
    discard: Card,
}

fn generate_candidates(
    top: &[Card], mid: &[Card], bot: &[Card],
    dealt: &[Card],
) -> Vec<Candidate> {
    assert_eq!(dealt.len(), 3);
    let current_counts = [top.len(), mid.len(), bot.len()];
    let limits = [3usize, 5, 5];

    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    for discard_idx in 0..3 {
        let discard = dealt[discard_idx];
        let remaining: Vec<Card> = dealt.iter().enumerate()
            .filter(|&(i, _)| i != discard_idx)
            .map(|(_, &c)| c)
            .collect();

        for pos0 in 0..3usize {
            for pos1 in 0..3usize {
                let mut counts = current_counts;
                counts[pos0] += 1;
                if counts[pos0] > limits[pos0] { continue; }
                counts[pos1] += 1;
                if counts[pos1] > limits[pos1] { continue; }

                let mut key_placements = vec![
                    (remaining[0].rank, remaining[0].suit, pos0 as u8),
                    (remaining[1].rank, remaining[1].suit, pos1 as u8),
                ];
                if pos0 == pos1 {
                    key_placements.sort();
                }
                let key = (key_placements, discard.rank, discard.suit);

                if seen.contains(&key) { continue; }
                seen.insert(key);

                candidates.push(Candidate {
                    placements: vec![
                        (remaining[0], pos0),
                        (remaining[1], pos1),
                    ],
                    discard,
                });
            }
        }
    }

    candidates
}

fn generate_t0_candidates(dealt: &[Card]) -> Vec<Candidate> {
    assert_eq!(dealt.len(), 5);
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    for top_n in 0..=3usize {
        for mid_n in 0..=(5 - top_n).min(5) {
            let bot_n = 5 - top_n - mid_n;
            if bot_n > 5 { continue; }

            for perm in dealt.iter().permutations(5) {
                let mut top_cards: Vec<&Card> = perm[..top_n].to_vec();
                let mut mid_cards: Vec<&Card> = perm[top_n..top_n + mid_n].to_vec();
                let mut bot_cards: Vec<&Card> = perm[top_n + mid_n..].to_vec();
                top_cards.sort_by_key(|c| (c.rank, c.suit));
                mid_cards.sort_by_key(|c| (c.rank, c.suit));
                bot_cards.sort_by_key(|c| (c.rank, c.suit));

                let key: Vec<(u8, u8, u8)> = top_cards.iter().map(|c| (c.rank, c.suit, 0))
                    .chain(mid_cards.iter().map(|c| (c.rank, c.suit, 1)))
                    .chain(bot_cards.iter().map(|c| (c.rank, c.suit, 2)))
                    .collect();

                if seen.contains(&key) { continue; }
                seen.insert(key);

                let mut placements = Vec::new();
                for &c in &top_cards { placements.push((*c, 0)); }
                for &c in &mid_cards { placements.push((*c, 1)); }
                for &c in &bot_cards { placements.push((*c, 2)); }

                candidates.push(Candidate {
                    placements,
                    discard: Card { rank: 0, suit: 0 },
                });
            }
        }
    }

    candidates
}

// ============================================================
//  Candidate evaluation
// ============================================================

#[derive(Debug, Clone, Serialize)]
struct CandidateResult {
    placements: Vec<(String, String)>,
    discard: String,
    ev: f64,
    bust_prob: f64,
    fl_rate: f64,
    expected_royalty: f64,
    fl_ev_contribution: f64,
}

fn row_name(idx: usize) -> &'static str {
    match idx {
        0 => "top",
        1 => "middle",
        2 => "bottom",
        _ => "unknown",
    }
}

fn evaluate_candidates(
    top: &[Card], mid: &[Card], bot: &[Card],
    dealt: &[Card], remaining_deck: &[Card],
    turn: usize,
) -> Vec<CandidateResult> {
    let candidates = if turn == 0 {
        generate_t0_candidates(dealt)
    } else {
        generate_candidates(top, mid, bot, dealt)
    };

    let mut results: Vec<CandidateResult> = candidates.par_iter().map(|cand| {
        let mut new_top = top.to_vec();
        let mut new_mid = mid.to_vec();
        let mut new_bot = bot.to_vec();

        for &(card, pos) in &cand.placements {
            match pos {
                0 => new_top.push(card),
                1 => new_mid.push(card),
                2 => new_bot.push(card),
                _ => {}
            }
        }

        // Remove placed cards and discard from remaining deck
        let used: HashSet<(u8, u8)> = cand.placements.iter()
            .map(|(c, _)| (c.rank, c.suit))
            .chain(std::iter::once((cand.discard.rank, cand.discard.suit)))
            .collect();
        let row_deck: Vec<Card> = remaining_deck.iter()
            .filter(|c| !used.contains(&(c.rank, c.suit)))
            .copied()
            .collect();

        let eval = evaluate_board(&new_top, &new_mid, &new_bot, &row_deck);

        CandidateResult {
            placements: cand.placements.iter()
                .map(|(c, pos)| (card_to_string(c), row_name(*pos).to_string()))
                .collect(),
            discard: card_to_string(&cand.discard),
            ev: eval.ev,
            bust_prob: eval.bust_prob,
            fl_rate: eval.fl_rate,
            expected_royalty: eval.expected_royalty,
            fl_ev_contribution: eval.fl_ev_contribution,
        }
    }).collect();

    results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ============================================================
//  Card parsing
// ============================================================

fn parse_card(s: &str) -> Option<Card> {
    let s = s.trim();
    if s == "X1" || s == "X2" || s == "Xj" {
        return Some(Card { rank: 0, suit: 4 });
    }
    if s.len() != 2 { return None; }
    let chars: Vec<char> = s.chars().collect();
    let rank = match chars[0] {
        '2' => 2, '3' => 3, '4' => 4, '5' => 5, '6' => 6, '7' => 7,
        '8' => 8, '9' => 9, 'T' => 10, 'J' => 11, 'Q' => 12, 'K' => 13, 'A' => 14,
        _ => return None,
    };
    let suit = match chars[1] {
        's' => 0, 'h' => 1, 'd' => 2, 'c' => 3,
        _ => return None,
    };
    Some(Card { rank, suit })
}

fn parse_cards(s: &str) -> Vec<Card> {
    if s.is_empty() { return vec![]; }
    s.split(',').filter_map(|c| parse_card(c.trim())).collect()
}

fn card_to_string(c: &Card) -> String {
    if c.is_joker() {
        "Xj".to_string()
    } else if c.rank == 0 && c.suit == 0 {
        "-".to_string()
    } else {
        format!("{}{}", rank_to_char(c.rank), SUIT_CHARS[c.suit as usize])
    }
}

// ============================================================
//  CLI
// ============================================================

#[derive(Parser)]
#[command(name = "prob_engine", about = "OFC probability calculation engine")]
struct Cli {
    /// Mode: "row" (single row), "board" (full board), "candidates" (evaluate all placements)
    #[arg(long, default_value = "row")]
    mode: String,

    /// Row cards for "row" mode
    #[arg(long, default_value = "")]
    row: String,

    /// Row type for "row" mode: "top", "mid", "bot"
    #[arg(long, default_value = "top")]
    row_type: String,

    /// Top row cards (for board/candidates mode)
    #[arg(long, default_value = "")]
    top: String,

    /// Middle row cards (for board/candidates mode)
    #[arg(long, default_value = "")]
    mid: String,

    /// Bottom row cards (for board/candidates mode)
    #[arg(long, default_value = "")]
    bot: String,

    /// Dealt cards (for candidates mode)
    #[arg(long, default_value = "")]
    dealt: String,

    /// Known cards NOT in the deck (comma-separated)
    #[arg(long, default_value = "")]
    exclude: String,

    /// Turn number (0-4)
    #[arg(long, default_value_t = 1)]
    turn: usize,

    /// Position: "btn", "bb", "fl_btn", "fl_bb", "fl_vs_fl", "fl_vs_normal"
    #[arg(long, default_value = "bb")]
    position: String,

    /// Number of MC simulations per candidate (for "mc" mode)
    #[arg(long, default_value_t = 50)]
    sims: usize,
}

// ============================================================
//  Output types
// ============================================================

#[derive(Serialize)]
struct RowOutput {
    row_cards: Vec<String>,
    row_type: String,
    remaining_deck_size: usize,
    distribution: RowDistribution,
}

#[derive(Serialize)]
struct BoardOutput {
    top_cards: Vec<String>,
    mid_cards: Vec<String>,
    bot_cards: Vec<String>,
    remaining_deck_size: usize,
    turn: usize,
    position: String,
    evaluation: BoardEvaluation,
}

#[derive(Serialize)]
struct CandidatesOutput {
    top_cards: Vec<String>,
    mid_cards: Vec<String>,
    bot_cards: Vec<String>,
    dealt_cards: Vec<String>,
    remaining_deck_size: usize,
    turn: usize,
    position: String,
    n_candidates: usize,
    candidates: Vec<CandidateResult>,
}

// ============================================================
//  Monte Carlo simulation
// ============================================================

/// Evaluate a complete 13-card board. Returns (score, is_fl, fl_cards).
/// score = royalty if not bust, -6 if bust. fl_cards = 14-17 or 0.
fn evaluate_final_board(top: &[Card], mid: &[Card], bot: &[Card]) -> (f64, bool, u8) {
    assert_eq!(top.len(), 3);
    assert_eq!(mid.len(), 5);
    assert_eq!(bot.len(), 5);

    // Bust check: top <= mid <= bot (using comparable values)
    let top_val = top_comparable_value_5(top);
    let mid_val = hand_value_5(mid);

    let mid_le_bot = compare_5_hands(mid, bot) <= 0;
    let top_le_mid = top_val <= mid_val;

    if !top_le_mid || !mid_le_bot {
        return (-6.0, false, 0);
    }

    let royalty = get_top_royalty(top) + get_middle_royalty(mid) + get_bottom_royalty(bot);
    let (fl, fl_cards) = check_fl_entry(top);
    let fl_bonus = if fl { fl_ev_for_cards(fl_cards) } else { 0.0 };

    (royalty as f64 + fl_bonus, fl, if fl { fl_cards } else { 0 })
}

/// T1-T3: select best placement via sampled prob_engine histogram EV.
fn select_best_placement_pe(
    top: &[Card], mid: &[Card], bot: &[Card],
    dealt: &[Card], remaining_deck: &[Card],
    rng: &mut StdRng,
) -> (Vec<Card>, Vec<Card>, Vec<Card>) {
    let candidates = generate_candidates(top, mid, bot, dealt);

    let mut best_ev = f64::NEG_INFINITY;
    let mut best_top = top.to_vec();
    let mut best_mid = mid.to_vec();
    let mut best_bot = bot.to_vec();

    for cand in &candidates {
        let mut new_top = top.to_vec();
        let mut new_mid = mid.to_vec();
        let mut new_bot = bot.to_vec();

        for &(card, pos) in &cand.placements {
            match pos {
                0 => new_top.push(card),
                1 => new_mid.push(card),
                2 => new_bot.push(card),
                _ => {}
            }
        }

        let used: HashSet<(u8, u8)> = cand.placements.iter()
            .map(|(c, _)| (c.rank, c.suit))
            .chain(std::iter::once((cand.discard.rank, cand.discard.suit)))
            .collect();
        let row_deck: Vec<Card> = remaining_deck.iter()
            .filter(|c| !used.contains(&(c.rank, c.suit)))
            .copied()
            .collect();

        let ev = evaluate_board_fast(&new_top, &new_mid, &new_bot, &row_deck, rng);

        if ev > best_ev {
            best_ev = ev;
            best_top = new_top;
            best_mid = new_mid;
            best_bot = new_bot;
        }
    }

    (best_top, best_mid, best_bot)
}

/// T4: board complete after placement → evaluate_final_board directly (μs).
fn select_best_placement_final(
    top: &[Card], mid: &[Card], bot: &[Card],
    dealt: &[Card],
) -> (Vec<Card>, Vec<Card>, Vec<Card>) {
    let candidates = generate_candidates(top, mid, bot, dealt);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_top = top.to_vec();
    let mut best_mid = mid.to_vec();
    let mut best_bot = bot.to_vec();

    for cand in &candidates {
        let mut new_top = top.to_vec();
        let mut new_mid = mid.to_vec();
        let mut new_bot = bot.to_vec();

        for &(card, pos) in &cand.placements {
            match pos {
                0 => new_top.push(card),
                1 => new_mid.push(card),
                2 => new_bot.push(card),
                _ => {}
            }
        }

        if new_top.len() == 3 && new_mid.len() == 5 && new_bot.len() == 5 {
            let (score, _, _) = evaluate_final_board(&new_top, &new_mid, &new_bot);
            if score > best_score {
                best_score = score;
                best_top = new_top;
                best_mid = new_mid;
                best_bot = new_bot;
            }
        }
    }

    (best_top, best_mid, best_bot)
}

/// Single MC simulation from a given board state.
/// T1-T3: prob_engine histogram EV. T4: direct final eval (μs).
fn run_one_simulation(
    top: &[Card], mid: &[Card], bot: &[Card],
    remaining_deck: &[Card],
    start_turn: usize,
    rng: &mut StdRng,
) -> (f64, bool, bool, u8) {
    let mut deck = remaining_deck.to_vec();
    deck.shuffle(rng);

    let mut cur_top = top.to_vec();
    let mut cur_mid = mid.to_vec();
    let mut cur_bot = bot.to_vec();
    let mut deck_idx = 0;

    for turn in start_turn..=4 {
        if cur_top.len() + cur_mid.len() + cur_bot.len() >= 13 { break; }
        if deck_idx + 3 > deck.len() { break; }

        let dealt: Vec<Card> = deck[deck_idx..deck_idx + 3].to_vec();
        deck_idx += 3;
        let future_deck: Vec<Card> = deck[deck_idx..].to_vec();

        let (new_top, new_mid, new_bot) = if turn == 4 {
            select_best_placement_final(
                &cur_top, &cur_mid, &cur_bot, &dealt,
            )
        } else {
            select_best_placement_pe(
                &cur_top, &cur_mid, &cur_bot, &dealt, &future_deck, rng,
            )
        };
        cur_top = new_top;
        cur_mid = new_mid;
        cur_bot = new_bot;
    }

    if cur_top.len() == 3 && cur_mid.len() == 5 && cur_bot.len() == 5 {
        let (score, is_fl, fl_cards) = evaluate_final_board(&cur_top, &cur_mid, &cur_bot);
        (score, score <= -6.0, is_fl, fl_cards)
    } else {
        (-6.0, true, false, 0)
    }
}

#[derive(Debug, Clone, Serialize)]
struct McResult {
    simulations: usize,
    avg_score: f64,
    bust_rate: f64,
    fl_rate: f64,
    avg_royalty_no_bust: f64,
    fl_type_rates: FlTypeRates,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct FlTypeRates {
    qq: f64,
    kk: f64,
    aa: f64,
    trips: f64,
}

#[derive(Debug, Clone, Serialize)]
struct McCandidate {
    placements: Vec<(String, String)>,
    discard: String,
    mc: McResult,
}

#[derive(Debug, Clone, Serialize)]
struct McCandidatesOutput {
    dealt_cards: Vec<String>,
    n_candidates: usize,
    simulations_per_candidate: usize,
    candidates: Vec<McCandidate>,
    elapsed_ms: u64,
}

/// Run MC simulation for a single board state.
fn run_mc(
    top: &[Card], mid: &[Card], bot: &[Card],
    remaining_deck: &[Card],
    start_turn: usize,
    n_sims: usize,
) -> McResult {
    let start = std::time::Instant::now();

    // Run simulations in parallel
    let results: Vec<(f64, bool, bool, u8)> = (0..n_sims).into_par_iter().map(|i| {
        let mut rng = StdRng::seed_from_u64(i as u64 + 42);
        run_one_simulation(top, mid, bot, remaining_deck, start_turn, &mut rng)
    }).collect();

    let n = results.len() as f64;
    let total_score: f64 = results.iter().map(|r| r.0).sum();
    let bust_count = results.iter().filter(|r| r.1).count();
    let fl_count = results.iter().filter(|r| r.2).count();
    let qq_count = results.iter().filter(|r| r.3 == 14).count();
    let kk_count = results.iter().filter(|r| r.3 == 15).count();
    let aa_count = results.iter().filter(|r| r.3 == 16).count();
    let trips_count = results.iter().filter(|r| r.3 == 17).count();

    let non_bust: Vec<&(f64, bool, bool, u8)> = results.iter().filter(|r| !r.1).collect();
    let avg_royalty_no_bust = if non_bust.is_empty() {
        0.0
    } else {
        non_bust.iter().map(|r| r.0).sum::<f64>() / non_bust.len() as f64
    };

    McResult {
        simulations: n_sims,
        avg_score: total_score / n,
        bust_rate: bust_count as f64 / n,
        fl_rate: fl_count as f64 / n,
        avg_royalty_no_bust,
        fl_type_rates: FlTypeRates {
            qq: qq_count as f64 / n,
            kk: kk_count as f64 / n,
            aa: aa_count as f64 / n,
            trips: trips_count as f64 / n,
        },
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

/// Evaluate all T0 candidates via MC simulation.
fn evaluate_t0_mc(
    dealt: &[Card],
    remaining_deck: &[Card],
    n_sims: usize,
) -> Vec<McCandidate> {
    let t0_candidates = generate_t0_candidates(dealt);

    let mut results: Vec<McCandidate> = t0_candidates.par_iter().map(|cand| {
        let mut new_top = Vec::new();
        let mut new_mid = Vec::new();
        let mut new_bot = Vec::new();

        for &(card, pos) in &cand.placements {
            match pos {
                0 => new_top.push(card),
                1 => new_mid.push(card),
                2 => new_bot.push(card),
                _ => {}
            }
        }

        let mc = run_mc(&new_top, &new_mid, &new_bot, remaining_deck, 1, n_sims);

        McCandidate {
            placements: cand.placements.iter()
                .map(|(c, pos)| (card_to_string(c), row_name(*pos).to_string()))
                .collect(),
            discard: card_to_string(&cand.discard),
            mc,
        }
    }).collect();

    results.sort_by(|a, b| b.mc.avg_score.partial_cmp(&a.mc.avg_score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Evaluate all T1+ candidates via MC simulation.
/// For each candidate placement, apply it to the board, then run MC from next turn.
fn evaluate_t1plus_mc(
    top: &[Card], mid: &[Card], bot: &[Card],
    dealt: &[Card],
    remaining_deck: &[Card],
    current_turn: usize,
    n_sims: usize,
) -> Vec<McCandidate> {
    let candidates = generate_candidates(top, mid, bot, dealt);

    let mut results: Vec<McCandidate> = candidates.par_iter().map(|cand| {
        let mut new_top = top.to_vec();
        let mut new_mid = mid.to_vec();
        let mut new_bot = bot.to_vec();

        for &(card, pos) in &cand.placements {
            match pos {
                0 => new_top.push(card),
                1 => new_mid.push(card),
                2 => new_bot.push(card),
                _ => {}
            }
        }

        // Remove placed + discarded cards from remaining deck
        let used: HashSet<(u8, u8)> = cand.placements.iter()
            .map(|(c, _)| (c.rank, c.suit))
            .chain(std::iter::once((cand.discard.rank, cand.discard.suit)))
            .collect();
        let cand_deck: Vec<Card> = remaining_deck.iter()
            .filter(|c| !used.contains(&(c.rank, c.suit)))
            .copied()
            .collect();

        let mc = run_mc(&new_top, &new_mid, &new_bot, &cand_deck, current_turn + 1, n_sims);

        McCandidate {
            placements: cand.placements.iter()
                .map(|(c, pos)| (card_to_string(c), row_name(*pos).to_string()))
                .collect(),
            discard: card_to_string(&cand.discard),
            mc,
        }
    }).collect();

    results.sort_by(|a, b| b.mc.avg_score.partial_cmp(&a.mc.avg_score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn build_remaining_deck(all_known: &[Card]) -> Vec<Card> {
    let full_deck = create_deck(true);
    let used: HashSet<(u8, u8)> = all_known.iter()
        .map(|c| (c.rank, c.suit))
        .collect();
    full_deck.into_iter()
        .filter(|c| !used.contains(&(c.rank, c.suit)))
        .collect()
}

fn main() {
    let cli = Cli::parse();

    match cli.mode.as_str() {
        "row" => {
            let row_cards = parse_cards(&cli.row);
            let exclude_cards = parse_cards(&cli.exclude);
            let all_known: Vec<Card> = row_cards.iter()
                .chain(exclude_cards.iter())
                .copied().collect();
            let remaining = build_remaining_deck(&all_known);

            let dist = match cli.row_type.as_str() {
                "top" => compute_top_distribution(&row_cards, &remaining),
                "mid" => compute_5card_distribution(&row_cards, &remaining, true),
                "bot" => compute_5card_distribution(&row_cards, &remaining, false),
                _ => {
                    eprintln!("Unknown row type: {}. Use 'top', 'mid', or 'bot'.", cli.row_type);
                    std::process::exit(1);
                }
            };

            let output = RowOutput {
                row_cards: row_cards.iter().map(card_to_string).collect(),
                row_type: cli.row_type,
                remaining_deck_size: remaining.len(),
                distribution: dist,
            };
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }

        "board" => {
            let top_cards = parse_cards(&cli.top);
            let mid_cards = parse_cards(&cli.mid);
            let bot_cards = parse_cards(&cli.bot);
            let exclude_cards = parse_cards(&cli.exclude);

            let all_known: Vec<Card> = top_cards.iter()
                .chain(mid_cards.iter())
                .chain(bot_cards.iter())
                .chain(exclude_cards.iter())
                .copied().collect();
            let remaining = build_remaining_deck(&all_known);

            let eval = evaluate_board(&top_cards, &mid_cards, &bot_cards, &remaining);

            let output = BoardOutput {
                top_cards: top_cards.iter().map(card_to_string).collect(),
                mid_cards: mid_cards.iter().map(card_to_string).collect(),
                bot_cards: bot_cards.iter().map(card_to_string).collect(),
                remaining_deck_size: remaining.len(),
                turn: cli.turn,
                position: cli.position,
                evaluation: eval,
            };
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }

        "candidates" => {
            let top_cards = parse_cards(&cli.top);
            let mid_cards = parse_cards(&cli.mid);
            let bot_cards = parse_cards(&cli.bot);
            let dealt_cards = parse_cards(&cli.dealt);
            let exclude_cards = parse_cards(&cli.exclude);

            let all_known: Vec<Card> = top_cards.iter()
                .chain(mid_cards.iter())
                .chain(bot_cards.iter())
                .chain(dealt_cards.iter())
                .chain(exclude_cards.iter())
                .copied().collect();
            let remaining = build_remaining_deck(&all_known);

            let results = evaluate_candidates(
                &top_cards, &mid_cards, &bot_cards,
                &dealt_cards, &remaining, cli.turn,
            );

            let output = CandidatesOutput {
                top_cards: top_cards.iter().map(card_to_string).collect(),
                mid_cards: mid_cards.iter().map(card_to_string).collect(),
                bot_cards: bot_cards.iter().map(card_to_string).collect(),
                dealt_cards: dealt_cards.iter().map(card_to_string).collect(),
                remaining_deck_size: remaining.len(),
                turn: cli.turn,
                position: cli.position,
                n_candidates: results.len(),
                candidates: results,
            };
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }

        "mc" => {
            let start = std::time::Instant::now();
            let dealt_cards = parse_cards(&cli.dealt);
            let top_cards = parse_cards(&cli.top);
            let mid_cards = parse_cards(&cli.mid);
            let bot_cards = parse_cards(&cli.bot);
            let exclude_cards = parse_cards(&cli.exclude);

            let all_known: Vec<Card> = top_cards.iter()
                .chain(mid_cards.iter())
                .chain(bot_cards.iter())
                .chain(dealt_cards.iter())
                .chain(exclude_cards.iter())
                .copied().collect();
            let remaining = build_remaining_deck(&all_known);

            if cli.turn == 0 {
                // T0: evaluate all placement candidates via MC
                let results = evaluate_t0_mc(&dealt_cards, &remaining, cli.sims);
                let output = McCandidatesOutput {
                    dealt_cards: dealt_cards.iter().map(card_to_string).collect(),
                    n_candidates: results.len(),
                    simulations_per_candidate: cli.sims,
                    candidates: results,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                };
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            } else {
                // T1+: evaluate all placement candidates via MC
                let results = evaluate_t1plus_mc(
                    &top_cards, &mid_cards, &bot_cards,
                    &dealt_cards, &remaining, cli.turn, cli.sims,
                );
                let output = McCandidatesOutput {
                    dealt_cards: dealt_cards.iter().map(card_to_string).collect(),
                    n_candidates: results.len(),
                    simulations_per_candidate: cli.sims,
                    candidates: results,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                };
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            }
        }

        _ => {
            eprintln!("Unknown mode: {}. Use 'row', 'board', 'candidates', or 'mc'.", cli.mode);
            std::process::exit(1);
        }
    }
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_1_slot() {
        let row = vec![
            Card { rank: 14, suit: 0 },
            Card { rank: 14, suit: 1 },
        ];
        let full = create_deck(true);
        let keep: HashSet<(u8, u8)> = [
            (14, 2), (13, 1), (12, 0), (2, 3), (3, 2),
        ].iter().cloned().collect();
        let mut exclude = Vec::new();
        for c in &full {
            if !keep.contains(&(c.rank, c.suit))
                && (c.rank, c.suit) != (14, 0)
                && (c.rank, c.suit) != (14, 1)
            {
                exclude.push(*c);
            }
        }

        let remaining: Vec<Card> = full.into_iter()
            .filter(|c| {
                let key = (c.rank, c.suit);
                key != (14, 0) && key != (14, 1) && !exclude.iter().any(|e| (e.rank, e.suit) == key)
            })
            .collect();

        assert_eq!(remaining.len(), 5);

        let dist = compute_top_distribution(&row, &remaining);
        assert_eq!(dist.n_combinations, 5);

        let trips_prob: f64 = dist.categories.iter()
            .find(|(name, _)| name == "trips")
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        let pair_prob: f64 = dist.categories.iter()
            .find(|(name, _)| name == "pair")
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        assert!((trips_prob - 0.2).abs() < 0.01, "trips_prob={}", trips_prob);
        assert!((pair_prob - 0.8).abs() < 0.01, "pair_prob={}", pair_prob);

        // Royalty: Trips AAA = 22, AA pair = 9
        let expected_royalty = (22.0 + 9.0 * 4.0) / 5.0;
        assert!((dist.expected_royalty - expected_royalty).abs() < 0.1,
            "expected_royalty={} vs {}", dist.expected_royalty, expected_royalty);

        // FL: all 5 completions give AA pair or Trips → 100% FL
        assert!((dist.fl_rate.unwrap() - 1.0).abs() < 0.01,
            "fl_rate={}", dist.fl_rate.unwrap());
    }

    #[test]
    fn test_bot_2_slots() {
        let row = vec![
            Card { rank: 13, suit: 0 },
            Card { rank: 13, suit: 1 },
            Card { rank: 11, suit: 2 },
        ];
        let keep: HashSet<(u8, u8)> = [
            (13, 2), (13, 3), (11, 1), (2, 0), (3, 0), (4, 0),
        ].iter().cloned().collect();

        let full = create_deck(false);
        let row_set: HashSet<(u8, u8)> = row.iter().map(|c| (c.rank, c.suit)).collect();
        let remaining: Vec<Card> = full.into_iter()
            .filter(|c| !row_set.contains(&(c.rank, c.suit)) && keep.contains(&(c.rank, c.suit)))
            .collect();

        assert_eq!(remaining.len(), 6);

        let dist = compute_5card_distribution(&row, &remaining, false);
        assert_eq!(dist.n_combinations, 15);

        assert!(dist.expected_royalty > 0.0, "should have positive royalty");
    }

    #[test]
    fn test_bust_high_prob() {
        // Strong top vs weak mid → high bust (using fine histograms)
        let mut top_hist = vec![0.0; FINE_BINS];
        top_hist[12] = 1.0; // OnePair range

        let mut mid_hist = vec![0.0; FINE_BINS];
        mid_hist[2] = 1.0; // HighCard low

        let mut bot_hist = vec![0.0; FINE_BINS];
        bot_hist[52] = 1.0; // Flush range

        let top_mid = prob_a_gt_b_fine(&top_hist, &mid_hist);
        assert!(top_mid > 0.9, "top_mid={}", top_mid);

        // Joint test: top=12 > mid=2, so mid can't satisfy top<=mid → bust
        // Build minimal RowDistributions for joint test
        let top = make_fine_test_dist(12, 0.0, 0.0, 0.0);
        let mid = make_fine_test_dist(2, 0.0, 0.0, 0.0);
        let bot = make_fine_test_dist(52, 0.0, 0.0, 0.0);
        let joint = compute_joint_stats(&top, &mid, &bot);
        assert!(joint.bust_prob > 0.99, "joint bust={}", joint.bust_prob);
    }

    #[test]
    fn test_bust_low_prob() {
        // Weak top, strong mid, stronger bot → no bust
        let top = make_fine_test_dist(1, 0.0, 0.0, 0.0);
        let mid = make_fine_test_dist(22, 0.0, 0.0, 0.0);
        let bot = make_fine_test_dist(62, 0.0, 0.0, 0.0);
        let joint = compute_joint_stats(&top, &mid, &bot);
        assert!(joint.bust_prob < 0.01, "bust={}", joint.bust_prob);
    }

    /// Helper: create a RowDistribution with a single fine bin for testing
    fn make_fine_test_dist(bin: usize, royalty: f64, fl_rate: f64, fl_ev: f64) -> RowDistribution {
        let mut fine_hist = vec![0.0; FINE_BINS];
        fine_hist[bin] = 1.0;
        let mut bin_royalty = vec![0.0; FINE_BINS];
        bin_royalty[bin] = royalty;
        let mut bin_fl_rate_v = vec![0.0; FINE_BINS];
        bin_fl_rate_v[bin] = fl_rate;
        let mut bin_fl_ev_v = vec![0.0; FINE_BINS];
        bin_fl_ev_v[bin] = fl_ev;
        RowDistribution {
            histogram: vec![], categories: vec![],
            mean_value: 0.0, std_value: 0.0,
            n_combinations: 1, expected_royalty: royalty,
            comparable_hist_5: None,
            fl_rate: None, fl_qq_rate: None, fl_kk_rate: None,
            fl_aa_rate: None, fl_trips_rate: None,
            fl_comparable_hist_5: None,
            fine_hist, bin_royalty,
            bin_fl_rate: bin_fl_rate_v, bin_fl_ev: bin_fl_ev_v,
        }
    }

    #[test]
    fn test_board_evaluation() {
        let top = vec![Card { rank: 14, suit: 0 }];
        let mid = vec![Card { rank: 13, suit: 0 }, Card { rank: 13, suit: 1 }];
        let bot = vec![Card { rank: 14, suit: 1 }, Card { rank: 14, suit: 2 }];

        let remaining = vec![
            Card { rank: 12, suit: 0 }, Card { rank: 12, suit: 1 },
            Card { rank: 11, suit: 0 }, Card { rank: 11, suit: 1 },
            Card { rank: 10, suit: 0 }, Card { rank: 10, suit: 1 },
            Card { rank: 9, suit: 0 },  Card { rank: 9, suit: 1 },
            Card { rank: 8, suit: 0 },  Card { rank: 8, suit: 1 },
            Card { rank: 7, suit: 0 },  Card { rank: 7, suit: 1 },
        ];

        let eval = evaluate_board(&top, &mid, &bot, &remaining);
        // KK vs AA on same small deck → similar distributions, bust_mid_bot ≈ 0.5
        assert!(eval.bust_mid_bot <= 0.6, "bust_mid_bot={}", eval.bust_mid_bot);
        assert!(eval.ev.is_finite(), "ev should be finite");
        println!("Board eval: ev={:.2}, bust={:.3}, fl={:.3}, royalty={:.2}",
            eval.ev, eval.bust_prob, eval.fl_rate, eval.expected_royalty);
    }

    #[test]
    fn test_candidate_generation() {
        let top = vec![Card { rank: 14, suit: 0 }];
        let mid = vec![Card { rank: 13, suit: 0 }, Card { rank: 13, suit: 1 }];
        let bot = vec![Card { rank: 14, suit: 1 }, Card { rank: 14, suit: 2 }];

        let dealt = vec![
            Card { rank: 12, suit: 0 },
            Card { rank: 11, suit: 0 },
            Card { rank: 10, suit: 0 },
        ];

        let candidates = generate_candidates(&top, &mid, &bot, &dealt);
        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 27);

        for cand in &candidates {
            assert_eq!(cand.placements.len(), 2);
        }
        println!("Generated {} candidates", candidates.len());
    }

    #[test]
    fn test_candidate_evaluation() {
        let top = vec![Card { rank: 14, suit: 0 }];
        let mid = vec![Card { rank: 13, suit: 0 }, Card { rank: 13, suit: 1 }];
        let bot = vec![Card { rank: 14, suit: 1 }, Card { rank: 14, suit: 2 }];

        let dealt = vec![
            Card { rank: 12, suit: 0 },
            Card { rank: 11, suit: 0 },
            Card { rank: 10, suit: 0 },
        ];

        let remaining = vec![
            Card { rank: 9, suit: 0 },  Card { rank: 9, suit: 1 },
            Card { rank: 8, suit: 0 },  Card { rank: 8, suit: 1 },
            Card { rank: 7, suit: 0 },  Card { rank: 7, suit: 1 },
            Card { rank: 6, suit: 0 },  Card { rank: 6, suit: 1 },
            Card { rank: 5, suit: 0 },  Card { rank: 5, suit: 1 },
        ];

        let results = evaluate_candidates(&top, &mid, &bot, &dealt, &remaining, 1);
        assert!(!results.is_empty());

        let best = &results[0];
        let worst = results.last().unwrap();
        assert!(best.ev >= worst.ev, "sorted by EV desc");

        println!("Best: ev={:.2}, bust={:.3}, fl={:.3}", best.ev, best.bust_prob, best.fl_rate);
        println!("Worst: ev={:.2}, bust={:.3}, fl={:.3}", worst.ev, worst.bust_prob, worst.fl_rate);
    }
}
