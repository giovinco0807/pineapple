use crate::bitboard::BitBoard;
use serde::{Deserialize, Serialize};
use crate::mcts::Action;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerBoard {
    pub top: BitBoard,
    pub middle: BitBoard,
    pub bottom: BitBoard,
    pub discards: BitBoard,
    
    pub top_count: u8,
    pub middle_count: u8,
    pub bottom_count: u8,
}

impl PlayerBoard {
    pub fn new() -> Self {
        PlayerBoard {
            top: BitBoard::EMPTY,
            middle: BitBoard::EMPTY,
            bottom: BitBoard::EMPTY,
            discards: BitBoard::EMPTY,
            top_count: 0,
            middle_count: 0,
            bottom_count: 0,
        }
    }
    
    pub fn total_placed(&self) -> u8 {
        self.top_count + self.middle_count + self.bottom_count
    }

    pub fn place(&mut self, to_top: BitBoard, to_middle: BitBoard, to_bottom: BitBoard, discards: BitBoard) {
        self.top |= to_top;
        self.middle |= to_middle;
        self.bottom |= to_bottom;
        self.discards |= discards;

        self.top_count += to_top.count() as u8;
        self.middle_count += to_middle.count() as u8;
        self.bottom_count += to_bottom.count() as u8;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    pub p1_board: PlayerBoard,
    pub p2_board: PlayerBoard,
    pub deck: BitBoard,
    
    pub is_p1_turn: bool,
    pub turn: u8, // 0 for T0, 1 for T1..T4
    pub current_hand: BitBoard,
    pub p1_chips: f32,
    pub p2_chips: f32,
}

impl GameState {
    pub fn initial() -> Self {
        GameState {
            p1_board: PlayerBoard::new(),
            p2_board: PlayerBoard::new(),
            deck: BitBoard::FULL_DECK,
            is_p1_turn: true,
            turn: 0,
            current_hand: BitBoard::EMPTY,
            p1_chips: 200.0,
            p2_chips: 200.0,
        }
    }
    
    pub fn is_terminal(&self) -> bool {
        self.p1_board.total_placed() == 13 && self.p2_board.total_placed() == 13
    }

    pub fn apply_action(&mut self, action: Action) {
        if self.is_p1_turn {
            self.p1_board.place(action.to_top, action.to_middle, action.to_bottom, action.discards);
        } else {
            self.p2_board.place(action.to_top, action.to_middle, action.to_bottom, action.discards);
        }

        let p1_placed = self.p1_board.total_placed();
        let p2_placed = self.p2_board.total_placed();

        self.current_hand = BitBoard::EMPTY;

        if p1_placed == p2_placed {
            self.turn += 1;
            self.is_p1_turn = true;
        } else {
            self.is_p1_turn = !self.is_p1_turn;
        }
    }

    pub fn deal_cards(&mut self) {
        self.deal_cards_with_rng(&mut rand::thread_rng());
    }

    pub fn deal_cards_with_rng(&mut self, rng: &mut impl rand::Rng) {
        if self.is_terminal() { return; }
        
        let num_cards = if self.turn == 0 { 5 } else { 3 };

        let mut avail_cards = [0u8; 54];
        let mut avail_len = 0;
        for c in self.deck.cards() {
            avail_cards[avail_len] = c;
            avail_len += 1;
        }

        for i in 0..num_cards {
            let swap_idx = rng.gen_range(i..avail_len);
            avail_cards.swap(i, swap_idx);
            let card = avail_cards[i];
            self.current_hand.add(card);
            self.deck.remove(card);
        }
    }


    pub fn determinize(&self) -> GameState {
        self.determinize_with_rng(&mut rand::thread_rng())
    }

    pub fn determinize_with_rng(&self, rng: &mut impl rand::Rng) -> GameState {
        let mut sim_state = self.clone();

        let mut known_cards = sim_state.p1_board.top
            | sim_state.p1_board.middle
            | sim_state.p1_board.bottom
            | sim_state.p2_board.top
            | sim_state.p2_board.middle
            | sim_state.p2_board.bottom;

        if self.is_p1_turn {
            known_cards |= sim_state.p1_board.discards;
            known_cards |= sim_state.current_hand;
        } else {
            known_cards |= sim_state.p2_board.discards;
            known_cards |= sim_state.current_hand;
        }

        let mut avail_cards = [0u8; 54];
        let mut avail_len = 0;
        for c in BitBoard::FULL_DECK.difference(known_cards).cards() {
            avail_cards[avail_len] = c;
            avail_len += 1;
        }

        let villain_discards_needed = if self.is_p1_turn {
            let needed = (sim_state.p2_board.total_placed().saturating_sub(5)) / 2;
            sim_state.p2_board.discards = BitBoard::EMPTY;
            needed
        } else {
            let needed = (sim_state.p1_board.total_placed().saturating_sub(5)) / 2;
            sim_state.p1_board.discards = BitBoard::EMPTY;
            needed
        };

        for i in 0..(villain_discards_needed as usize) {
            let swap_idx = rng.gen_range(i..avail_len);
            avail_cards.swap(i, swap_idx);
            let card = avail_cards[i];
            if self.is_p1_turn {
                sim_state.p2_board.discards.add(card);
            } else {
                sim_state.p1_board.discards.add(card);
            }
        }

        let mut new_deck = BitBoard::EMPTY;
        for i in (villain_discards_needed as usize)..avail_len {
            new_deck.add(avail_cards[i]);
        }
        sim_state.deck = new_deck;

        sim_state
    }
}
