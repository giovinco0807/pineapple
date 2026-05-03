//! MCCFR Engine: Outcome Sampling with Regret Matching+.
//!
//! Outcome Sampling: BOTH traverser AND opponent sample ONE action.
//! Regrets are updated using importance sampling weights.
//! Each iteration is a single path through the tree → O(depth) per iteration.
//! Requires many more iterations but each is extremely fast.

use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use crate::game_state::{GameState, NodeType};
use crate::info_set::InfoSetStore;

/// MCCFR Solver configuration.
pub struct MCCFRConfig {
    pub discount_alpha: f64,
    pub discount_beta: f64,
    pub discount_gamma: f64,
    pub discount_interval: u64,
    pub exploration_eps: f64,    // ε for ε-greedy exploration in OS-MCCFR
}

impl Default for MCCFRConfig {
    fn default() -> Self {
        MCCFRConfig {
            discount_alpha: 1.5,
            discount_beta: 0.0,
            discount_gamma: 2.0,
            discount_interval: 10000,
            exploration_eps: 0.6,  // Standard: 0.6
        }
    }
}

/// MCCFR Solver with Outcome Sampling.
pub struct MCCFRSolver {
    pub store: InfoSetStore,
    pub config: MCCFRConfig,
    pub iteration: u64,

    // Stats
    pub nodes_visited: u64,
    pub terminal_reached: u64,
}

impl MCCFRSolver {
    pub fn new(config: MCCFRConfig) -> Self {
        MCCFRSolver {
            store: InfoSetStore::new(),
            config,
            iteration: 0,
            nodes_visited: 0,
            terminal_reached: 0,
        }
    }

    /// Run one full CFR iteration (both players as traverser).
    pub fn run_iteration(&mut self, rng: &mut impl Rng) {
        let state = GameState::new_random(rng);

        // Alternate traverser
        let traverser = (self.iteration % 2) as usize;

        self.os_cfr_traverse(&state, traverser, 1.0, 1.0, 1.0, rng);

        self.iteration += 1;

        // Apply discounting periodically
        if self.iteration > 0 && self.iteration % self.config.discount_interval == 0 {
            self.store.apply_discount(
                self.iteration,
                self.config.discount_alpha,
                self.config.discount_beta,
                self.config.discount_gamma,
            );
        }
    }

    /// Outcome Sampling MCCFR traversal.
    ///
    /// `pi_i` = reach probability for traverser
    /// `pi_neg_i` = reach probability for opponent
    /// `s` = sampling probability along this path
    fn os_cfr_traverse(
        &mut self,
        state: &GameState,
        traverser: usize,
        pi_i: f64,
        pi_neg_i: f64,
        s: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        self.nodes_visited += 1;

        match state.node_type() {
            NodeType::Terminal => {
                self.terminal_reached += 1;
                // Return utility / sampling probability for IS correction
                state.terminal_utility(traverser) / s
            }

            NodeType::Chance => {
                let new_state = state.deal_next_turn(rng);
                self.os_cfr_traverse(&new_state, traverser, pi_i, pi_neg_i, s, rng)
            }

            NodeType::Player0 | NodeType::Player1 => {
                let acting_player = match state.node_type() {
                    NodeType::Player0 => 0usize,
                    NodeType::Player1 => 1usize,
                    _ => unreachable!(),
                };

                let actions = state.get_legal_actions(acting_player);
                if actions.is_empty() {
                    let mut new_state = state.clone();
                    new_state.placed[acting_player] = true;
                    return self.os_cfr_traverse(&new_state, traverser, pi_i, pi_neg_i, s, rng);
                }

                let n_actions = actions.len();
                let info_key = state.info_set_key(acting_player);
                let eps = self.config.exploration_eps;

                // Get current strategy
                let strategy = {
                    let info = self.store.get_or_create(info_key, n_actions);
                    info.get_strategy()
                };

                // ε-greedy sampling distribution:
                // σ'(a) = ε/|A| + (1-ε) × σ(a)
                let sample_probs: Vec<f64> = strategy.iter()
                    .map(|&s| eps / n_actions as f64 + (1.0 - eps) * s)
                    .collect();

                // Sample one action
                let action_idx = sample_action(&sample_probs, rng);
                let sigma_prime_a = sample_probs[action_idx];

                // Recurse down the sampled action
                let new_pi_i;
                let new_pi_neg_i;
                let new_s;

                if acting_player == traverser {
                    new_pi_i = pi_i * strategy[action_idx];
                    new_pi_neg_i = pi_neg_i;
                    new_s = s * sigma_prime_a;
                } else {
                    new_pi_i = pi_i;
                    new_pi_neg_i = pi_neg_i * strategy[action_idx];
                    new_s = s * sigma_prime_a;
                }

                let next_state = state.apply_action(acting_player, &actions[action_idx]);
                let u = self.os_cfr_traverse(&next_state, traverser, new_pi_i, new_pi_neg_i, new_s, rng);

                if acting_player == traverser {
                    // Compute counterfactual value estimates
                    // W = u × pi_neg_i / s
                    let w = u * pi_neg_i;

                    let info = self.store.get_or_create(info_key, n_actions);
                    for a in 0..n_actions {
                        let regret = if a == action_idx {
                            w * (1.0 - strategy[action_idx])
                        } else {
                            -w * strategy[action_idx]
                        };
                        info.add_regret(a, regret);
                    }

                    // Accumulate strategy (weighted by opponent reach / sampling)
                    let strat_weight = pi_neg_i / s;
                    let weighted_strat: Vec<f64> = strategy.iter()
                        .map(|&s| s * strat_weight)
                        .collect();
                    info.add_strategy(&weighted_strat);
                }

                u
            }
        }
    }
}

/// Sample an action index according to probabilities.
fn sample_action(probs: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..probs.len());
    }

    match WeightedIndex::new(probs) {
        Ok(dist) => dist.sample(rng),
        Err(_) => rng.gen_range(0..probs.len()),
    }
}
