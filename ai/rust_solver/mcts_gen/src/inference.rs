use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use crate::state::GameState;
use crate::bitboard::BitBoard;

pub trait Evaluator {
    fn evaluate(&mut self, state: &GameState) -> Option<(Vec<f32>, f32)>;
}

/// Map a Rust bitboard card index to the Python encoding.py card index.
///
/// Rust bitboard order:  suits = [s, h, d, c] → card = suit_rust * 13 + rank
/// Python encoding.py:  SUITS = "hdcs"        → card = suit_py   * 13 + rank
///
/// Mapping: s(0)→3, h(1)→0, d(2)→1, c(3)→2
#[inline]
fn rust_to_python_idx(rust_card: u8) -> usize {
    if rust_card >= 52 {
        // Jokers stay at 52, 53
        return rust_card as usize;
    }
    let rank = (rust_card % 13) as usize;
    let rust_suit = (rust_card / 13) as usize;
    const SUIT_MAP: [usize; 4] = [3, 0, 1, 2]; // s→3, h→0, d→1, c→2
    SUIT_MAP[rust_suit] * 13 + rank
}

pub struct PolicyValueSession {
    session: Session,
}

impl PolicyValueSession {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        // ort 2.x doesn't require explicit Environment creation
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(PolicyValueSession { session })
    }

    pub fn predict(&mut self, state: &GameState) -> ort::Result<(Vec<f32>, f32)> {
        let mut features = Array2::<f32>::zeros((1, 490));

        let (my_board, opp_board) = if state.is_p1_turn {
            (&state.p1_board, &state.p2_board)
        } else {
            (&state.p2_board, &state.p1_board)
        };

        let mut seen = BitBoard(0);

        // Use rust_to_python_idx to align card positions with Python's encoding
        let mut set_loc = |bb: &BitBoard, loc: usize| {
            for c in bb.cards() {
                let py_idx = rust_to_python_idx(c);
                features[[0, py_idx * 9 + loc]] = 1.0;
                seen.add(c as u8);
            }
        };

        set_loc(&my_board.top, 0);
        set_loc(&my_board.middle, 1);
        set_loc(&my_board.bottom, 2);
        
        set_loc(&opp_board.top, 3);
        set_loc(&opp_board.middle, 4);
        set_loc(&opp_board.bottom, 5);

        set_loc(&state.current_hand, 6);
        set_loc(&my_board.discards, 7);

        // Unseen — also remap via rust_to_python_idx
        for c in 0..54u8 {
            if !seen.contains(c) {
                let py_idx = rust_to_python_idx(c);
                features[[0, py_idx * 9 + 8]] = 1.0;
            }
        }

        // Meta features
        features[[0, 486]] = (state.turn as f32) / 8.0;
        features[[0, 487]] = if state.is_p1_turn { 1.0 } else { 0.0 };
        let (my_chips, opp_chips) = if state.is_p1_turn {
            (state.p1_chips, state.p2_chips)
        } else {
            (state.p2_chips, state.p1_chips)
        };
        features[[0, 488]] = my_chips / 200.0;
        features[[0, 489]] = opp_chips / 200.0;

        let input_tensor = Tensor::from_array(features)?;
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        // Extract raw slices
        let (_, row_logits_val) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, ev_pred_val) = outputs[1].try_extract_tensor::<f32>()?;

        let logits = row_logits_val.to_vec();
        let ev = ev_pred_val[0];

        Ok((logits, ev))
    }
}

impl Evaluator for PolicyValueSession {
    fn evaluate(&mut self, state: &GameState) -> Option<(Vec<f32>, f32)> {
        self.predict(state).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_to_python_idx() {
        // Rust: 2s = index 0, Python: 2s = index 39 (suit s is 4th in "hdcs")
        assert_eq!(rust_to_python_idx(0), 39);
        // Rust: As = index 12, Python: As = index 51
        assert_eq!(rust_to_python_idx(12), 51);
        // Rust: 2h = index 13, Python: 2h = index 0 (suit h is 1st in "hdcs")
        assert_eq!(rust_to_python_idx(13), 0);
        // Rust: Ah = index 25, Python: Ah = index 12
        assert_eq!(rust_to_python_idx(25), 12);
        // Rust: 2d = index 26, Python: 2d = index 13
        assert_eq!(rust_to_python_idx(26), 13);
        // Rust: 2c = index 39, Python: 2c = index 26
        assert_eq!(rust_to_python_idx(39), 26);
        // Jokers stay same
        assert_eq!(rust_to_python_idx(52), 52);
        assert_eq!(rust_to_python_idx(53), 53);
    }
}
