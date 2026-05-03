pub mod mcts;
pub mod inference;

use ofc_core::{Card, get_top_royalty, get_middle_royalty, get_bottom_royalty, evaluate_board_with_joker_constraint, compare_5_hands, evaluate_3_card};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Row {
    Top,
    Middle,
    Bottom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerBoard {
    pub top: Vec<Card>,
    pub mid: Vec<Card>,
    pub bot: Vec<Card>,
    pub discards: Vec<Card>,
    pub busted: bool,
}

impl PlayerBoard {
    pub fn new() -> Self {
        Self {
            top: Vec::new(),
            mid: Vec::new(),
            bot: Vec::new(),
            discards: Vec::new(),
            busted: false,
        }
    }

    /// 盤面の最終評価（バースト判定含む）
    pub fn finalize(&mut self) {
        if self.top.len() == 3 && self.mid.len() == 5 && self.bot.len() == 5 {
            let eval = evaluate_board_with_joker_constraint(&self.top, &self.mid, &self.bot);
            self.busted = eval.busted;
        }
    }

    pub fn total_royalty(&self) -> i32 {
        if self.busted {
            return 0;
        }
        get_top_royalty(&self.top) + get_middle_royalty(&self.mid) + get_bottom_royalty(&self.bot)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Turn {
    T0, T1, T2, T3, T4, Showdown
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    pub p1_board: PlayerBoard, // BB (先手)
    pub p2_board: PlayerBoard, // BTN (後手)
    pub deck: Vec<Card>,
    pub turn: Turn,
    pub current_player: u8, // 1 for P1, 2 for P2
}

impl GameState {
    pub fn new(deck: Vec<Card>) -> Self {
        Self {
            p1_board: PlayerBoard::new(),
            p2_board: PlayerBoard::new(),
            deck,
            turn: Turn::T0,
            current_player: 1, // 先手からスタート
        }
    }

    pub fn apply_placement(&mut self, action: &crate::mcts::Action) {
        let board = if self.current_player == 1 {
            &mut self.p1_board
        } else {
            &mut self.p2_board
        };

        // カードを配置
        for &(card, dest) in &action.placement.cards {
            match dest {
                crate::Row::Top => board.top.push(card),
                crate::Row::Middle => board.mid.push(card),
                crate::Row::Bottom => board.bot.push(card),
            }
        }
        if let Some(card) = action.placement.discard {
            board.discards.push(card);
        }

        // ターン進行
        if self.current_player == 1 {
            self.current_player = 2;
        } else {
            self.current_player = 1;
            self.turn = match self.turn {
                Turn::T0 => Turn::T1,
                Turn::T1 => Turn::T2,
                Turn::T2 => Turn::T3,
                Turn::T3 => Turn::T4,
                Turn::T4 => Turn::Showdown,
                Turn::Showdown => Turn::Showdown,
            };
        }

        if self.turn == Turn::Showdown {
            self.p1_board.finalize();
            self.p2_board.finalize();
        }
    }

    pub fn get_legal_actions(&self, hand: &[Card]) -> Vec<crate::mcts::PlacementAction> {
        use itertools::Itertools;
        let mut actions = Vec::new();
        
        let board = if self.current_player == 1 {
            &self.p1_board
        } else {
            &self.p2_board
        };
        
        let top_space = 3 - board.top.len();
        let mid_space = 5 - board.mid.len();
        let bot_space = 5 - board.bot.len();
        
        if hand.len() == 5 {
            // T0: place 5 cards
            let mut unique_actions = std::collections::HashSet::new();
            for assignment in std::iter::repeat(vec![Row::Top, Row::Middle, Row::Bottom]).take(5).multi_cartesian_product() {
                let mut t = 0; let mut m = 0; let mut b = 0;
                for &row in &assignment {
                    match row { Row::Top => t+=1, Row::Middle => m+=1, Row::Bottom => b+=1 }
                }
                if t <= top_space && m <= mid_space && b <= bot_space {
                    let mut cards = Vec::new();
                    for (i, &row) in assignment.iter().enumerate() {
                        cards.push((hand[i], row));
                    }
                    // Sort to avoid duplicate equivalent actions
                    cards.sort_by_key(|&(c, r)| (r as u8, c.rank, c.suit));
                    unique_actions.insert(cards);
                }
            }
            for cards in unique_actions {
                actions.push(crate::mcts::PlacementAction { cards, discard: None });
            }
        } else if hand.len() == 3 {
            // T1-T4: choose 2 to place, 1 to discard
            let mut unique_actions = std::collections::HashSet::new();
            for combo in (0..3).combinations(2) {
                let d_idx = 3 - combo[0] - combo[1];
                let discard = hand[d_idx];
                
                let c1 = hand[combo[0]];
                let c2 = hand[combo[1]];
                
                let mut rows = Vec::new();
                if top_space > 0 { rows.push(Row::Top); }
                if mid_space > 0 { rows.push(Row::Middle); }
                if bot_space > 0 { rows.push(Row::Bottom); }
                
                for &r1 in &rows {
                    for &r2 in &rows {
                        let t = (if r1 == Row::Top {1} else {0}) + (if r2 == Row::Top {1} else {0});
                        let m = (if r1 == Row::Middle {1} else {0}) + (if r2 == Row::Middle {1} else {0});
                        let b = (if r1 == Row::Bottom {1} else {0}) + (if r2 == Row::Bottom {1} else {0});
                        
                        if t <= top_space && m <= mid_space && b <= bot_space {
                            let mut cards = vec![(c1, r1), (c2, r2)];
                            cards.sort_by_key(|&(c, r)| (r as u8, c.rank, c.suit));
                            unique_actions.insert((cards, discard));
                        }
                    }
                }
            }
            for (cards, discard) in unique_actions {
                actions.push(crate::mcts::PlacementAction { cards, discard: Some(discard) });
            }
        }
        
        actions
    }

    /// 2つの完成した盤面を比較し、P1視点の相対スコア（勝敗ポイント）を計算する
    pub fn calculate_score(p1: &PlayerBoard, p2: &PlayerBoard) -> i32 {
        if p1.busted && p2.busted {
            return 0; // 引き分け
        }
        if p1.busted {
            return - (p2.total_royalty() + 6); // P2のロイヤルティ + スクープボーナス(6点)
        }
        if p2.busted {
            return p1.total_royalty() + 6; // P1のロイヤルティ + スクープボーナス(6点)
        }

        // 行ごとの勝敗 (1: P1勝ち, -1: P2勝ち, 0: 引き分け)
        let top_win = Self::compare_3_hands(&p1.top, &p2.top);
        let mid_win = compare_5_hands(&p1.mid, &p2.mid);
        let bot_win = compare_5_hands(&p1.bot, &p2.bot);

        let mut row_score = 0;
        row_score += top_win;
        row_score += mid_win;
        row_score += bot_win;

        // スクープボーナス判定 (+3 or -3)
        let scoop_bonus = if top_win > 0 && mid_win > 0 && bot_win > 0 {
            3
        } else if top_win < 0 && mid_win < 0 && bot_win < 0 {
            -3
        } else {
            0
        };

        // ロイヤルティの差分
        let royalty_diff = p1.total_royalty() - p2.total_royalty();

        row_score + scoop_bonus + royalty_diff
    }

    fn compare_3_hands(a: &[Card], b: &[Card]) -> i32 {
        let (ra, sa) = evaluate_3_card(a);
        let (rb, sb) = evaluate_3_card(b);
        if (ra as u8) != (rb as u8) {
            if (ra as u8) > (rb as u8) { 1 } else { -1 }
        } else {
            if sa > sb { 1 } else if sa < sb { -1 } else { 0 }
        }
    }

    pub fn get_fl_ev(board: &PlayerBoard) -> f64 {
        if board.busted {
            return 0.0;
        }
        let (is_fl, cards) = ofc_core::check_fl_entry(&board.top);
        if !is_fl {
            return 0.0;
        }
        match cards {
            14 => 24.9,
            15 => 31.4,
            16 => 39.8,
            17 => 47.7,
            _ => 0.0,
        }
    }

    pub fn calculate_reward(p1: &PlayerBoard, p2: &PlayerBoard) -> f64 {
        let base_score = Self::calculate_score(p1, p2) as f64;
        let p1_fl_ev = Self::get_fl_ev(p1);
        let p2_fl_ev = Self::get_fl_ev(p2);
        base_score + p1_fl_ev - p2_fl_ev
    }
}
