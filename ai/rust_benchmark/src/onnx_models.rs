//! ONNX Runtime model management and inference
//!
//! Loads BC (policy) and VN (value) models, supports per-turn BC.

use ort::session::Session;
use ort::value::TensorRef;
use ndarray::Array2;
use std::path::Path;
use rand::Rng;

use crate::encoding::STATE_DIM;

const MAX_ACTIONS: usize = 250;

pub struct OnnxModels {
    pub bc: Session,
    pub vn: Session,
    pub bc_per_turn: [Option<Session>; 5], // T0-T4
}

impl OnnxModels {
    pub fn load(
        bc_path: &str,
        vn_path: &str,
        bc_t_paths: &[Option<String>; 5],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bc = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(bc_path)?;
        let vn = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(vn_path)?;

        let mut bc_per_turn: [Option<Session>; 5] = Default::default();
        for (i, path_opt) in bc_t_paths.iter().enumerate() {
            if let Some(path) = path_opt {
                if Path::new(path).exists() {
                    bc_per_turn[i] = Some(
                        Session::builder()?
                            .with_intra_threads(1)?
                            .commit_from_file(path)?
                    );
                }
            }
        }

        Ok(OnnxModels { bc, vn, bc_per_turn })
    }

    fn run_session(session: &mut Session, input: Array2<f32>) -> Vec<f32> {
        let input_ref = TensorRef::from_array_view(&input).unwrap();
        let outputs = session.run(ort::inputs![input_ref]).unwrap();
        let (_, data) = outputs[0].try_extract_tensor::<f32>().unwrap();
        data.to_vec()
    }

    /// Run BC inference for a single state. Returns logits (250 dims).
    pub fn bc_inference(&mut self, state: &[f32; STATE_DIM], turn: u8) -> Vec<f32> {
        let input = Array2::from_shape_vec((1, STATE_DIM), state.to_vec()).unwrap();
        // Need to select session without borrowing self twice
        let use_per_turn = self.bc_per_turn[turn as usize].is_some();
        if use_per_turn {
            let session = self.bc_per_turn[turn as usize].as_mut().unwrap();
            Self::run_session(session, input)
        } else {
            Self::run_session(&mut self.bc, input)
        }
    }

    /// Run BC inference for a batch of states. Returns logits for each state.
    pub fn bc_inference_batch(
        &mut self,
        states: &[&[f32; STATE_DIM]],
        turn: u8,
    ) -> Vec<Vec<f32>> {
        if states.is_empty() { return Vec::new(); }

        let n = states.len();
        let mut flat = Vec::with_capacity(n * STATE_DIM);
        for s in states {
            flat.extend_from_slice(s.as_slice());
        }
        let input = Array2::from_shape_vec((n, STATE_DIM), flat).unwrap();

        let use_per_turn = self.bc_per_turn[turn as usize].is_some();
        let data = if use_per_turn {
            let session = self.bc_per_turn[turn as usize].as_mut().unwrap();
            Self::run_session(session, input)
        } else {
            Self::run_session(&mut self.bc, input)
        };

        let row_len = data.len() / n;
        (0..n).map(|i| {
            data[i * row_len..(i + 1) * row_len].to_vec()
        }).collect()
    }

    /// Run VN inference for a single state.
    /// Returns (value, bust_prob, fl_prob, royalty_ev).
    pub fn vn_inference(&mut self, state: &[f32; STATE_DIM]) -> (f32, f32, f32, f32) {
        let input = Array2::from_shape_vec((1, STATE_DIM), state.to_vec()).unwrap();
        let data = Self::run_session(&mut self.vn, input);
        (data[0], data[1], data[2], data[3])
    }

    /// Run VN inference for a batch of states.
    pub fn vn_inference_batch(
        &mut self,
        states: &[&[f32; STATE_DIM]],
    ) -> Vec<(f32, f32, f32, f32)> {
        if states.is_empty() { return Vec::new(); }

        let n = states.len();
        let mut flat = Vec::with_capacity(n * STATE_DIM);
        for s in states {
            flat.extend_from_slice(s.as_slice());
        }
        let input = Array2::from_shape_vec((n, STATE_DIM), flat).unwrap();
        let data = Self::run_session(&mut self.vn, input);

        (0..n).map(|i| {
            let base = i * 4;
            (data[base], data[base + 1], data[base + 2], data[base + 3])
        }).collect()
    }

    /// Select best action from BC logits given valid action count.
    pub fn bc_select_greedy(logits: &[f32], n_valid: usize) -> usize {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..n_valid.min(logits.len()) {
            if logits[i] > best_val {
                best_val = logits[i];
                best_idx = i;
            }
        }
        best_idx
    }

    /// Select action from BC logits with temperature sampling.
    pub fn bc_select_temperature(
        logits: &[f32],
        n_valid: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> usize {
        if n_valid <= 1 { return 0; }

        let valid_logits = &logits[..n_valid];
        let max_logit = valid_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut probs: Vec<f64> = valid_logits.iter()
            .map(|&l| ((l - max_logit) as f64 / temperature as f64).exp())
            .collect();
        let sum: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum; }

        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        n_valid - 1
    }
}
