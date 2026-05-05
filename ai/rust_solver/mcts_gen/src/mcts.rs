use std::collections::HashMap;
use crate::bitboard::BitBoard;
use crate::state::GameState;
use crate::action_gen::{get_initial_actions, get_turn_actions};
use crate::inference::Evaluator;
use crate::eval::compute_score;
use rand::Rng;
use rand_distr::Dirichlet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Action {
    pub to_top: BitBoard,
    pub to_middle: BitBoard,
    pub to_bottom: BitBoard,
    pub discards: BitBoard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    Action(Action),
    Cards(BitBoard),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeType {
    Decision,
    Chance,
}

pub struct MctsNode {
    pub node_type: NodeType,
    pub player_is_p1: bool,
    pub visits: u32,
    pub value_sum: f64,
    pub children: HashMap<Edge, MctsNode>,
    pub valid_actions: Vec<Action>,
    pub priors: Vec<f32>,
    pub is_expanded: bool,
}

impl MctsNode {
    pub fn new(node_type: NodeType, player_is_p1: bool) -> Self {
        MctsNode {
            node_type,
            player_is_p1,
            visits: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            valid_actions: Vec::new(),
            priors: Vec::new(),
            is_expanded: false,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / (self.visits as f64)
        }
    }

    pub fn ucb_score(&self, parent_visits: u32, prior: f32, c_puct: f64, invert: bool) -> f64 {
        let exploration = c_puct * (prior as f64) * (parent_visits as f64).sqrt() / (1.0 + self.visits as f64);
        let mut q = self.q_value();
        if invert {
            q = -q;
        }
        q + exploration
    }
}

pub struct IsMcts {
    pub root: MctsNode,
    pub c_puct: f64,
    pub progressive_widening_c: f64,
    pub progressive_widening_alpha: f64,
    pub max_children: usize,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
}

impl IsMcts {
    pub fn new() -> Self {
        IsMcts {
            root: MctsNode::new(NodeType::Decision, true),
            c_puct: 1.5,
            progressive_widening_c: 2.5,
            progressive_widening_alpha: 0.5,
            max_children: 100,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }

    pub fn search(&mut self, state: &GameState, num_simulations: usize, model: &mut impl Evaluator) -> Action {
        let valid_actions = if state.turn == 0 {
            get_initial_actions(state.current_hand, if state.is_p1_turn { &state.p1_board } else { &state.p2_board })
        } else {
            get_turn_actions(state.current_hand, if state.is_p1_turn { &state.p1_board } else { &state.p2_board })
        };

        if valid_actions.is_empty() {
            // Should not happen unless terminal
            return Action { to_top: BitBoard::EMPTY, to_middle: BitBoard::EMPTY, to_bottom: BitBoard::EMPTY, discards: BitBoard::EMPTY };
        }
        if valid_actions.len() == 1 {
            return valid_actions[0];
        }

        if !self.root.is_expanded {
            self.root.player_is_p1 = state.is_p1_turn;
            Self::expand_node(&mut self.root, state, &valid_actions, model);
            
            // Add Dirichlet noise
            if self.dirichlet_epsilon > 0.0 && self.root.priors.len() > 1 {
                let alpha = vec![self.dirichlet_alpha; self.root.priors.len()];
                if let Ok(dir) = Dirichlet::new(&alpha) {
                    let mut rng = rand::thread_rng();
                    let noise: Vec<f64> = rng.sample(dir);
                    let eps = self.dirichlet_epsilon as f32;
                    for (i, prior) in self.root.priors.iter_mut().enumerate() {
                        *prior = (*prior * (1.0 - eps)) + (noise[i] as f32 * eps);
                    }
                }
            }
        }

        for _ in 0..num_simulations {
            let mut sim_state = state.determinize();
            Self::traverse(&mut self.root, &mut sim_state, model, self.c_puct, self.progressive_widening_c, self.progressive_widening_alpha, self.max_children);
        }

        // Select best action based on visits
        let mut best_action = valid_actions[0];
        let mut max_visits = 0;

        for action in &self.root.valid_actions {
            if let Some(child) = self.root.children.get(&Edge::Action(*action)) {
                if child.visits > max_visits {
                    max_visits = child.visits;
                    best_action = *action;
                }
            }
        }

        best_action
    }

    fn expand_node(node: &mut MctsNode, state: &GameState, valid_actions: &Vec<Action>, model: &mut impl Evaluator) -> f64 {
        // Run inference
        let (logits, ev) = model.evaluate(state).unwrap_or((vec![0.0; valid_actions.len()], 0.0));
        
        let priors = if logits.len() >= valid_actions.len() {
            // Softmax
            let max_l = logits[..valid_actions.len()].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            let mut exp_l = Vec::with_capacity(valid_actions.len());
            for &l in &logits[..valid_actions.len()] {
                let e = (l - max_l).exp();
                sum_exp += e;
                exp_l.push(e);
            }
            if sum_exp > 0.0 {
                exp_l.iter().map(|&e| e / sum_exp).collect()
            } else {
                vec![1.0 / valid_actions.len() as f32; valid_actions.len()]
            }
        } else {
            vec![1.0 / valid_actions.len() as f32; valid_actions.len()]
        };

        node.valid_actions = valid_actions.clone();
        node.priors = priors;
        node.is_expanded = true;
        
        // Evaluate the other player? If we only predicted for current player, EV is from their perspective.
        // Return EV from perspective of Player 1.
        if state.is_p1_turn {
            ev as f64
        } else {
            -(ev as f64)
        }
    }

    fn traverse(
        node: &mut MctsNode, 
        sim_state: &mut GameState, 
        model: &mut impl Evaluator,
        c_puct: f64,
        pw_c: f64,
        pw_alpha: f64,
        max_children: usize
    ) -> f64 {
        if sim_state.is_terminal() {
            let score = compute_score(
                sim_state.p1_board.top, sim_state.p1_board.middle, sim_state.p1_board.bottom,
                sim_state.p2_board.top, sim_state.p2_board.middle, sim_state.p2_board.bottom
            );
            return if sim_state.is_p1_turn { score } else { -score };
        }

        if sim_state.current_hand.is_empty() {
            // Needs to deal cards (CHANCE node)
            sim_state.deal_cards();
            
            if node.node_type != NodeType::Chance {
                panic!("Expected Chance node!");
            }

            let cards = sim_state.current_hand;
            if !node.children.contains_key(&Edge::Cards(cards)) {
                node.children.insert(Edge::Cards(cards), MctsNode::new(NodeType::Decision, sim_state.is_p1_turn));
            }

            let next_node = node.children.get_mut(&Edge::Cards(cards)).unwrap();
            let v = Self::traverse(next_node, sim_state, model, c_puct, pw_c, pw_alpha, max_children);
            node.visits += 1;
            // Value sum for chance node? Usually sum over children, or just simple average.
            node.value_sum += v;
            return v;
        }

        if !node.is_expanded {
            let valid_actions = if sim_state.turn == 0 {
                get_initial_actions(sim_state.current_hand, if sim_state.is_p1_turn { &sim_state.p1_board } else { &sim_state.p2_board })
            } else {
                get_turn_actions(sim_state.current_hand, if sim_state.is_p1_turn { &sim_state.p1_board } else { &sim_state.p2_board })
            };
            let v = Self::expand_node(node, sim_state, &valid_actions, model);
            node.visits += 1;
            node.value_sum += v;
            return v;
        }

        // Select action UCB
        let max_k = std::cmp::min(
            node.valid_actions.len(),
            std::cmp::max(1, (pw_c * (node.visits as f64).powf(pw_alpha)).ceil() as usize)
        );
        let k = std::cmp::min(max_k, max_children);

        let mut sorted_actions: Vec<(usize, &Action)> = node.valid_actions.iter().enumerate().collect();
        sorted_actions.sort_by(|a, b| node.priors[b.0].partial_cmp(&node.priors[a.0]).unwrap());
        let candidates = &sorted_actions[..k];

        let invert = !node.player_is_p1;
        let mut best_score = std::f64::NEG_INFINITY;
        let mut best_action = node.valid_actions[0];

        for &(idx, action) in candidates {
            if let Some(child) = node.children.get(&Edge::Action(*action)) {
                let score = child.ucb_score(node.visits, node.priors[idx], c_puct, invert);
                if score > best_score {
                    best_score = score;
                    best_action = *action;
                }
            } else {
                // Unexplored action has infinite UCB
                best_action = *action;
                break;
            }
        }

        sim_state.apply_action(best_action);

        if !node.children.contains_key(&Edge::Action(best_action)) {
            let next_node_type = if sim_state.current_hand.is_empty() { NodeType::Chance } else { NodeType::Decision };
            node.children.insert(Edge::Action(best_action), MctsNode::new(next_node_type, sim_state.is_p1_turn));
        }

        let next_node = node.children.get_mut(&Edge::Action(best_action)).unwrap();
        let v = Self::traverse(next_node, sim_state, model, c_puct, pw_c, pw_alpha, max_children);

        node.visits += 1;
        node.value_sum += v;
        v
    }
}
