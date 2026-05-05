use mcts_gen::mcts::{IsMcts};
use mcts_gen::state::{GameState, PlayerBoard};
use mcts_gen::bitboard::BitBoard;
use mcts_gen::inference::Evaluator;

// A dummy network that always returns uniform policy and 0 value
struct DummyNetwork;

impl Evaluator for DummyNetwork {
    fn evaluate(&mut self, _state: &GameState) -> Option<(Vec<f32>, f32)> {
        None
    }
}

#[test]
fn test_mcts_avoids_bust() {
    let mut state = GameState::initial();
    
    // Fast forward to turn 4
    state.turn = 4;
    state.is_p1_turn = true;
    
    // Setup P1 Board
    let mut p1 = PlayerBoard::new();
    p1.place(
        BitBoard::from_string("Ah 2h").unwrap(),
        BitBoard::from_string("3c 4c 5c 6c 7c").unwrap(),
        BitBoard::from_string("8d 9d Td Jd").unwrap(),
        BitBoard::from_string("2s 3s 4s 5s").unwrap() // some discards
    );
    state.p1_board = p1;
    
    // Setup P2 Board
    let mut p2 = PlayerBoard::new();
    p2.place(
        BitBoard::from_string("2d 3d").unwrap(),
        BitBoard::from_string("4d 5d 6d 7d 8h").unwrap(),
        BitBoard::from_string("9h Th Jh Qh").unwrap(),
        BitBoard::from_string("6s 7s 8s 9s").unwrap()
    );
    state.p2_board = p2;
    
    // P1's current hand: Qd, Kh, 2c
    state.current_hand = BitBoard::from_string("Qd Kh 2c").unwrap();
    
    let mut mcts = IsMcts::new();
    // Turn off Dirichlet noise for reproducible testing
    mcts.dirichlet_epsilon = 0.0;
    
    let mut network = DummyNetwork;
    
    // 10000 simulations to ensure it finds the valid terminal state
    let action = mcts.search(&state, 10000, &mut network);
    
    // The safe action is putting Qd at bottom (makes straight flush), Kh at top, 2c discard
    let expected_to_bottom = BitBoard::from_string("Qd").unwrap();
    let expected_to_top = BitBoard::from_string("Kh").unwrap();
    
    assert_eq!(action.to_bottom, expected_to_bottom, "MCTS should have placed Qd at bottom to make a straight flush");
    assert_eq!(action.to_top, expected_to_top, "MCTS should have placed Kh at top to avoid busting");
}
