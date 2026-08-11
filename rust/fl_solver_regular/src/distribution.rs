//! Opponent-Fantasyland board distribution.
//!
//! From a hero's information set the opponent's 14 Fantasyland cards are an
//! unordered uniform draw from everything the hero has not seen -- the hero's
//! own board, the cards dealt to them and their own discards. The opponent's
//! hand is exchangeable with the rest of the unseen deck, so sampling it after
//! the fact is the exact posterior, not an approximation.
//!
//! Sampling is addressed by `(seed_base, sample_index)` rather than drawn from
//! a running stream, which is what makes common random numbers work: hero
//! candidate A and hero candidate B are scored against the *same* opponent
//! boards, so their difference is not contaminated by sampling noise.

use crate::cards::unseen_from_mask;
use crate::eval::{category, HAND_FLUSH, HAND_FULL_HOUSE, HAND_QUADS, HAND_STRAIGHT,
    HAND_STRAIGHT_FLUSH, HAND_TRIPS};
use crate::objective::ObjectiveConfig;
use crate::rng::SplitMix64;
use crate::solver::{FlSolver, Solution};
use serde::{Deserialize, Serialize};

/// One sampled and solved opponent Fantasyland deal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampledFlBoard {
    pub sample_index: u64,
    pub deal: [u8; 14],
    pub solution: Solution,
}

/// Draw the 14-card Fantasyland deal for sample `sample_index`.
///
/// Deterministic in `(seed_base, sample_index)` and in the unseen set.
pub fn sample_deal(seen_mask: u64, seed_base: u64, sample_index: u64) -> Result<[u8; 14], String> {
    let mut unseen = unseen_from_mask(seen_mask);
    if unseen.len() < 14 {
        return Err(format!(
            "opponent Fantasyland needs 14 unseen cards, the hero has left {}",
            unseen.len()
        ));
    }
    let mut rng = SplitMix64::for_stream(seed_base, sample_index);
    rng.partial_shuffle(&mut unseen, 14);
    let mut deal: [u8; 14] = unseen[..14].try_into().expect("14 cards drawn");
    // Canonical order keeps the deal a set, not a sequence, so a solved board
    // is a function of the cards alone and replays byte-identically.
    deal.sort_unstable();
    Ok(deal)
}

/// Sample and solve `count` opponent Fantasyland boards from a hero's unseen
/// remainder. Sample `i` uses stream `(seed_base, first_sample + i)`.
pub fn sample_and_solve(
    seen_mask: u64,
    count: usize,
    seed_base: u64,
    first_sample: u64,
    solver: &mut FlSolver,
) -> Result<Vec<SampledFlBoard>, String> {
    let mut out = Vec::with_capacity(count);
    for offset in 0..count as u64 {
        let sample_index = first_sample + offset;
        let deal = sample_deal(seen_mask, seed_base, sample_index)?;
        let solution = solver
            .solve(&deal)
            .ok_or_else(|| "a 14-card Fantasyland deal must have a legal placement".to_owned())?;
        out.push(SampledFlBoard {
            sample_index,
            deal,
            solution,
        });
    }
    Ok(out)
}

/// Row-strength histogram bins, coarse enough to read at a glance.
pub const CATEGORY_LABELS: [&str; 9] = [
    "high", "pair", "two_pair", "trips", "straight", "flush", "full_house", "quads",
    "straight_flush",
];

/// Aggregate statistics over a set of solved Fantasyland boards.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DistributionStats {
    pub solved: usize,
    pub stays: usize,
    pub fouls: usize,
    pub royalty_sum: f64,
    pub royalty_sum_squares: f64,
    pub royalty_histogram: Vec<u64>,
    pub top_categories: Vec<u64>,
    pub middle_categories: Vec<u64>,
    pub bottom_categories: Vec<u64>,
    pub stay_by_reason: Vec<(String, u64)>,
    /// Fantasyland *entry* rate under ordinary QQ+ rules, reported only to
    /// show how far entry and stay diverge; it is not what a Fantasyland
    /// player continues on.
    pub qq_plus_top: usize,
}

/// Royalty histogram upper edges; the final bin is open-ended.
pub const ROYALTY_BIN_EDGES: [i32; 12] = [0, 2, 4, 6, 8, 10, 13, 16, 20, 26, 35, 50];

impl DistributionStats {
    pub fn new() -> Self {
        Self {
            royalty_histogram: vec![0; ROYALTY_BIN_EDGES.len() + 1],
            top_categories: vec![0; CATEGORY_LABELS.len()],
            middle_categories: vec![0; CATEGORY_LABELS.len()],
            bottom_categories: vec![0; CATEGORY_LABELS.len()],
            ..Default::default()
        }
    }

    pub fn observe(&mut self, solution: &Solution) {
        self.solved += 1;
        let royalty = solution.total_royalty;
        self.royalty_sum += royalty as f64;
        self.royalty_sum_squares += (royalty as f64) * (royalty as f64);
        let bin = ROYALTY_BIN_EDGES
            .iter()
            .position(|edge| royalty <= *edge)
            .unwrap_or(ROYALTY_BIN_EDGES.len());
        self.royalty_histogram[bin] += 1;
        self.top_categories[category(solution.top_key) as usize] += 1;
        self.middle_categories[category(solution.middle_key) as usize] += 1;
        self.bottom_categories[category(solution.bottom_key) as usize] += 1;
        if let Some(reason) = &solution.stay {
            self.stays += 1;
            match self
                .stay_by_reason
                .iter_mut()
                .find(|(label, _)| label == reason)
            {
                Some((_, count)) => *count += 1,
                None => self.stay_by_reason.push((reason.clone(), 1)),
            }
        }
        if solution.fl_entry().is_some() {
            self.qq_plus_top += 1;
        }
        if crate::eval::is_foul(solution.top_key, solution.middle_key, solution.bottom_key) {
            self.fouls += 1;
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.solved += other.solved;
        self.stays += other.stays;
        self.fouls += other.fouls;
        self.qq_plus_top += other.qq_plus_top;
        self.royalty_sum += other.royalty_sum;
        self.royalty_sum_squares += other.royalty_sum_squares;
        for (slot, value) in other.royalty_histogram.iter().enumerate() {
            self.royalty_histogram[slot] += value;
        }
        for (slot, value) in other.top_categories.iter().enumerate() {
            self.top_categories[slot] += value;
        }
        for (slot, value) in other.middle_categories.iter().enumerate() {
            self.middle_categories[slot] += value;
        }
        for (slot, value) in other.bottom_categories.iter().enumerate() {
            self.bottom_categories[slot] += value;
        }
        for (label, count) in other.stay_by_reason.iter() {
            match self
                .stay_by_reason
                .iter_mut()
                .find(|(existing, _)| existing == label)
            {
                Some((_, existing)) => *existing += count,
                None => self.stay_by_reason.push((label.clone(), *count)),
            }
        }
    }

    pub fn stay_rate(&self) -> f64 {
        self.stays as f64 / self.solved.max(1) as f64
    }

    pub fn mean_royalty(&self) -> f64 {
        self.royalty_sum / self.solved.max(1) as f64
    }

    pub fn royalty_standard_deviation(&self) -> f64 {
        let count = self.solved.max(1) as f64;
        let mean = self.mean_royalty();
        (self.royalty_sum_squares / count - mean * mean).max(0.0).sqrt()
    }
}

/// Which hand categories can produce a Fantasyland stay, for reporting.
pub fn stay_capable_categories() -> [(&'static str, u32); 6] {
    [
        ("trips (top)", HAND_TRIPS),
        ("straight", HAND_STRAIGHT),
        ("flush", HAND_FLUSH),
        ("full_house", HAND_FULL_HOUSE),
        ("quads (bottom)", HAND_QUADS),
        ("straight_flush (bottom)", HAND_STRAIGHT_FLUSH),
    ]
}

/// Full-deck run: sample `count` Fantasyland deals from a fresh 52-card deck.
pub fn fresh_deck_stats(
    count: usize,
    seed_base: u64,
    config: &ObjectiveConfig,
) -> Result<DistributionStats, String> {
    let mut solver = FlSolver::new(config)?;
    let mut stats = DistributionStats::new();
    for sample_index in 0..count as u64 {
        let deal = sample_deal(0, seed_base, sample_index)?;
        let solution = solver
            .solve(&deal)
            .ok_or_else(|| "a 14-card Fantasyland deal must have a legal placement".to_owned())?;
        stats.observe(&solution);
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::mask_of;
    use crate::objective::{ObjectiveConfig, ObjectiveKind};

    fn pure_config() -> ObjectiveConfig {
        ObjectiveConfig {
            kind: ObjectiveKind::PureV1,
            fl_ev_stay: 9.109,
            fl_ev_config_path: "test".to_owned(),
            fl_ev_cards: 14,
            hero_foul_prob: 0.2482,
            hero_foul_prob_provenance: "test".to_owned(),
            reference: None,
        }
    }

    #[test]
    fn sampling_is_deterministic_and_avoids_the_seen_cards() {
        let seen = mask_of(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let first = sample_deal(seen, 997_000_000, 5).unwrap();
        let again = sample_deal(seen, 997_000_000, 5).unwrap();
        assert_eq!(first, again);
        for card in first {
            assert_eq!(seen & (1_u64 << card), 0);
        }
        assert_ne!(first, sample_deal(seen, 997_000_000, 6).unwrap());
    }

    #[test]
    fn common_random_numbers_reuse_the_same_deals_across_callers() {
        let seen = mask_of(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut solver = FlSolver::new(&pure_config()).unwrap();
        let first = sample_and_solve(seen, 4, 997_000_000, 0, &mut solver).unwrap();
        let again = sample_and_solve(seen, 4, 997_000_000, 0, &mut solver).unwrap();
        for (left, right) in first.iter().zip(again.iter()) {
            assert_eq!(left.deal, right.deal);
            assert_eq!(left.solution, right.solution);
        }
    }

    #[test]
    fn a_solved_fantasyland_board_never_fouls() {
        let mut solver = FlSolver::new(&pure_config()).unwrap();
        let solved = sample_and_solve(0, 40, 997_000_123, 0, &mut solver).unwrap();
        for board in solved {
            assert!(!crate::eval::is_foul(
                board.solution.top_key,
                board.solution.middle_key,
                board.solution.bottom_key
            ));
        }
    }

    #[test]
    fn refusing_to_sample_without_enough_unseen_cards() {
        // Exactly 14 unseen is the boundary: fine at 38 seen, refused at 39.
        let seen = (0..38_u8).fold(0_u64, |mask, card| mask | (1_u64 << card));
        assert!(sample_deal(seen, 1, 0).is_ok());
        let seen = (0..39_u8).fold(0_u64, |mask, card| mask | (1_u64 << card));
        assert!(sample_deal(seen, 1, 0).is_err());
    }
}
