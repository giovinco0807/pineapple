use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use crate::state::GameState;
use crate::bitboard::BitBoard;

pub trait Evaluator {
    fn evaluate(&mut self, state: &GameState) -> Option<(Vec<f32>, f32)>;
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

        let mut set_loc = |bb: &BitBoard, loc: usize| {
            for c in bb.cards() {
                features[[0, (c as usize) * 9 + loc]] = 1.0;
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

        // Unseen
        for c in 0..54 {
            if !seen.contains(c as u8) {
                features[[0, (c as usize) * 9 + 8]] = 1.0;
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
