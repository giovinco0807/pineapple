use crate::{GameState, Row};
use ofc_core::Card;

#[derive(Clone, Debug)]
pub struct PlacementAction {
    pub cards: Vec<(Card, Row)>,
    pub discard: Option<Card>,
}

/// ニューラルネットから出力される「候補手」と「その確率」
#[derive(Clone, Debug)]
pub struct Action {
    pub placement: PlacementAction,
    pub prior_prob: f64, // P(s, a): ニューラルネットが予測したこの手の良さ
}

/// MCTSの探索木の各ノード
pub struct Node {
    pub state: GameState,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub action_taken: Option<Action>,
    pub visits: u32,       // N(s, a): 訪問回数
    pub total_score: f64,  // W(s, a): 累計スコア
}

impl Node {
    pub fn new(state: GameState, parent: Option<usize>, action_taken: Option<Action>) -> Self {
        Self {
            state,
            parent,
            children: Vec::new(),
            action_taken,
            visits: 0,
            total_score: 0.0,
        }
    }

    pub fn is_expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

pub struct MCTS {
    pub nodes: Vec<Node>,
    pub c_puct: f64, // 探索の度合いを決めるハイパーパラメータ
}

impl MCTS {
    pub fn new(initial_state: GameState, c_puct: f64) -> Self {
        let root = Node::new(initial_state, None, None);
        Self {
            nodes: vec![root],
            c_puct,
        }
    }

    /// PUCTアルゴリズムに基づき、最も有望な子ノードを選択する
    pub fn select_child(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        let mut best_child = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        let parent_visits = node.visits as f64;
        let sqrt_parent_visits = parent_visits.sqrt();

        for &child_idx in &node.children {
            let child = &self.nodes[child_idx];
            
            // Q値: その手の平均スコア（未探索なら0）
            let q_value = if child.visits > 0 {
                child.total_score / child.visits as f64
            } else {
                0.0
            };

            // U値: 事前確率(Prior)と訪問回数に基づく探索ボーナス
            let prior = child.action_taken.as_ref().unwrap().prior_prob;
            let u_value = self.c_puct * prior * sqrt_parent_visits / (1.0 + child.visits as f64);
            
            let ucb = q_value + u_value;

            if ucb > best_ucb {
                best_ucb = ucb;
                best_child = child_idx;
            }
        }
        best_child
    }

    /// Neural Netの出力を受け取り、ノードを展開する
    pub fn expand(&mut self, node_idx: usize, actions: Vec<Action>) {
        for action in actions {
            // 現在の状態をコピー
            let mut new_state = self.nodes[node_idx].state.clone();
            
            // アクション（配置）を適用してターン/プレイヤーを更新
            new_state.apply_placement(&action);

            let child_node = Node::new(new_state, Some(node_idx), Some(action));
            let child_idx = self.nodes.len();
            self.nodes.push(child_node);
            self.nodes[node_idx].children.push(child_idx);
        }
    }

    /// シミュレーション結果（勝敗スコア）をルートまで逆伝播させる
    pub fn backpropagate(&mut self, mut node_idx: usize, mut score: f64) {
        loop {
            let node = &mut self.nodes[node_idx];
            node.visits += 1;
            node.total_score += score;
            
            let parent_opt = node.parent;
            if let Some(parent_idx) = parent_opt {
                node_idx = parent_idx;
                
                // 相手のターンになるため、相対スコアを反転させる (Zero-Sum Game)
                score = -score;
            } else {
                break; // ルート到達
            }
        }
    }
}
