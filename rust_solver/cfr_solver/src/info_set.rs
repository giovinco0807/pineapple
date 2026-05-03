//! Information set storage and regret matching for MCCFR.

use rustc_hash::FxHashMap;

/// Data stored per information set.
#[derive(Clone, Debug)]
pub struct InfoSetData {
    pub cumulative_regret: Vec<f64>,
    pub cumulative_strategy: Vec<f64>,
    pub n_actions: usize,
}

impl InfoSetData {
    pub fn new(n_actions: usize) -> Self {
        InfoSetData {
            cumulative_regret: vec![0.0; n_actions],
            cumulative_strategy: vec![0.0; n_actions],
            n_actions,
        }
    }

    /// Regret Matching+: compute current strategy from cumulative regrets.
    /// Negative regrets are clamped to 0 (RM+).
    pub fn get_strategy(&self) -> Vec<f64> {
        let positive_sum: f64 = self.cumulative_regret.iter()
            .map(|&r| r.max(0.0))
            .sum();

        if positive_sum > 0.0 {
            self.cumulative_regret.iter()
                .map(|&r| r.max(0.0) / positive_sum)
                .collect()
        } else {
            // Uniform
            let p = 1.0 / self.n_actions as f64;
            vec![p; self.n_actions]
        }
    }

    /// Update regret for a specific action.
    pub fn add_regret(&mut self, action_idx: usize, regret: f64) {
        self.cumulative_regret[action_idx] += regret;
        // RM+: clamp to 0
        if self.cumulative_regret[action_idx] < 0.0 {
            self.cumulative_regret[action_idx] = 0.0;
        }
    }

    /// Accumulate strategy for averaging.
    pub fn add_strategy(&mut self, strategy: &[f64]) {
        for (i, &s) in strategy.iter().enumerate() {
            self.cumulative_strategy[i] += s;
        }
    }

}

/// Storage for all information sets.
pub struct InfoSetStore {
    pub data: FxHashMap<u64, InfoSetData>,
}

impl InfoSetStore {
    pub fn new() -> Self {
        InfoSetStore {
            data: FxHashMap::default(),
        }
    }

    /// Get or create an info set entry.
    pub fn get_or_create(&mut self, key: u64, n_actions: usize) -> &mut InfoSetData {
        self.data.entry(key)
            .or_insert_with(|| InfoSetData::new(n_actions))
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Apply Linear CFR discounting.
    pub fn apply_discount(&mut self, iteration: u64, alpha: f64, beta: f64, gamma: f64) {
        let t = iteration as f64;
        let pos_discount = t.powf(alpha) / (t.powf(alpha) + 1.0);
        let neg_discount = t.powf(beta) / (t.powf(beta) + 1.0);
        let strat_discount = (t / (t + 1.0)).powf(gamma);

        for info in self.data.values_mut() {
            for r in info.cumulative_regret.iter_mut() {
                if *r > 0.0 {
                    *r *= pos_discount;
                } else {
                    *r *= neg_discount;
                }
            }
            for s in info.cumulative_strategy.iter_mut() {
                *s *= strat_discount;
            }
        }
    }
}
