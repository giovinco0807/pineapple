//! Inference for the distilled vs-Fantasyland rankers.
//!
//! The T1 teacher continues its candidates with the trained T2-vs-FL and
//! T3-vs-FL models rather than searching those streets exhaustively, so those
//! models have to run inside the labeller. This is the reader and the forward
//! pass, and nothing else.
//!
//! # Why not T4M1
//!
//! The engine's [`crate`-external] `T4M1` image stores a standard deviation and
//! refuses zeros, and it has no field for an input clamp. Both matter here:
//!
//! * The winning arms are the warm ones, whose standardization is the
//!   pretrained one with the opponent tail's inverse standard deviation forced
//!   to exactly zero -- the tail is not available in this situation and is
//!   imputed to its pretrained mean. A `std`-shaped field cannot express that.
//! * Their forward pass clamps standardized inputs to +/-8, because under the
//!   pretrained statistics a few vs-FL dims land absurdly outside the range the
//!   function was fitted on (feature 81 is a constant 6 in normal-vs-normal data
//!   and a constant 13 here, which standardizes to 7000).
//!
//! Writing these weights as `T4M1` would silently drop both and produce a model
//! that is not the model that was measured. `VFL1` stores the inverse std
//! directly and records the clamp.
//!
//! # Determinism
//!
//! Every value is `f32`, the layer loop is ordered, and the dot product
//! accumulates in a single fixed order -- no split accumulators, no rayon
//! inside a prediction. Two runs on the same bytes and the same input produce
//! bit-identical scores, which is what lets a label be reproduced from its
//! pinned digest. `predict` is the only entry point and it borrows scratch
//! space so a caller ranking a fan allocates once.

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"VFL1";
const VERSION: u32 = 1;

struct Layer {
    inputs: usize,
    outputs: usize,
    /// Row-major `[output][input]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

pub struct VflModel {
    input_dim: usize,
    clamp: f32,
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
    layers: Vec<Layer>,
    widest: usize,
}

pub struct Scratch {
    current: Vec<f32>,
    next: Vec<f32>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, String> {
        let end = self.offset + 4;
        if end > self.bytes.len() {
            return Err("image ends inside a length field".to_owned());
        }
        let value = u32::from_le_bytes(self.bytes[self.offset..end].try_into().unwrap());
        self.offset = end;
        Ok(value)
    }

    fn f32(&mut self) -> Result<f32, String> {
        let end = self.offset + 4;
        if end > self.bytes.len() {
            return Err("image ends inside a float field".to_owned());
        }
        let value = f32::from_le_bytes(self.bytes[self.offset..end].try_into().unwrap());
        self.offset = end;
        Ok(value)
    }

    fn floats(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let end = self.offset + count * 4;
        if end > self.bytes.len() {
            return Err(format!("image ends inside a block of {count} floats"));
        }
        let values = self.bytes[self.offset..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        self.offset = end;
        Ok(values)
    }
}

impl VflModel {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 || &bytes[..4] != MAGIC {
            return Err("image does not start with the expected magic".to_owned());
        }
        let mut reader = Reader { bytes, offset: 4 };
        let version = reader.u32()?;
        if version != VERSION {
            return Err(format!("image version {version} is not supported"));
        }
        let input_dim = reader.u32()? as usize;
        let layer_count = reader.u32()? as usize;
        if layer_count == 0 {
            return Err("image declares no layers".to_owned());
        }
        let clamp = reader.f32()?;
        if !clamp.is_finite() || clamp <= 0.0 {
            return Err(format!("image clamp {clamp} is not a positive finite value"));
        }
        let mean = reader.floats(input_dim)?;
        let inverse_std = reader.floats(input_dim)?;
        // Zero is legal here and meaningful: it pins a dim to its mean. NaN and
        // infinity are not.
        if inverse_std.iter().any(|value| !value.is_finite())
            || mean.iter().any(|value| !value.is_finite())
        {
            return Err("image has a non-finite standardization term".to_owned());
        }

        let mut layers = Vec::with_capacity(layer_count);
        let mut expected_inputs = input_dim;
        for index in 0..layer_count {
            let inputs = reader.u32()? as usize;
            let outputs = reader.u32()? as usize;
            if inputs != expected_inputs {
                return Err(format!(
                    "layer {index} expects {inputs} inputs but the previous stage \
                     produces {expected_inputs}"
                ));
            }
            let weight = reader.floats(inputs * outputs)?;
            let bias = reader.floats(outputs)?;
            layers.push(Layer {
                inputs,
                outputs,
                weight,
                bias,
            });
            expected_inputs = outputs;
        }
        if expected_inputs != 1 {
            return Err(format!(
                "the final layer must produce a single value, not {expected_inputs}"
            ));
        }
        if reader.offset != bytes.len() {
            return Err(format!(
                "image has {} trailing bytes",
                bytes.len() - reader.offset
            ));
        }
        let widest = layers.iter().map(|layer| layer.outputs).max().unwrap_or(1);
        Ok(Self {
            input_dim,
            clamp,
            mean,
            inverse_std,
            layers,
            widest,
        })
    }

    /// Load and refuse anything whose bytes are not the pinned ones.
    pub fn load_pinned(bytes: &[u8], expected_sha256: &str) -> Result<Self, String> {
        let digest = Sha256::digest(bytes);
        let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        if digest != expected_sha256 {
            return Err(format!(
                "model image digest {digest} does not match the pinned {expected_sha256}"
            ));
        }
        Self::load(bytes)
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn clamp(&self) -> f32 {
        self.clamp
    }

    pub fn scratch(&self) -> Scratch {
        let width = self.widest.max(self.input_dim);
        Scratch {
            current: vec![0.0; width],
            next: vec![0.0; width],
        }
    }

    pub fn predict(&self, features: &[f32], scratch: &mut Scratch) -> Result<f32, String> {
        if features.len() != self.input_dim {
            return Err(format!(
                "expected {} features, got {}",
                self.input_dim,
                features.len()
            ));
        }
        for index in 0..self.input_dim {
            let value = (features[index] - self.mean[index]) * self.inverse_std[index];
            scratch.current[index] = value.clamp(-self.clamp, self.clamp);
        }
        let last = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            let input = &scratch.current[..layer.inputs];
            for output in 0..layer.outputs {
                let row = &layer.weight[output * layer.inputs..(output + 1) * layer.inputs];
                let mut sum = layer.bias[output];
                for (weight, value) in row.iter().zip(input.iter()) {
                    sum += weight * value;
                }
                scratch.next[output] = if index == last { sum } else { sum.max(0.0) };
            }
            std::mem::swap(&mut scratch.current, &mut scratch.next);
        }
        Ok(scratch.current[0])
    }

    /// Index of the highest-scoring row, which is the continuation policy's
    /// pick. Ties go to the lower index so a label does not depend on the
    /// iteration order of anything upstream.
    pub fn argmax(&self, rows: &[Vec<f32>], scratch: &mut Scratch) -> Result<usize, String> {
        if rows.is_empty() {
            return Err("argmax over an empty fan".to_owned());
        }
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (index, row) in rows.iter().enumerate() {
            let score = self.predict(row, scratch)?;
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-layer image built by hand, so the reader is tested against bytes
    /// whose answer is known without running the exporter.
    fn tiny_image(clamp: f32) -> Vec<u8> {
        let mut image = Vec::new();
        image.extend_from_slice(MAGIC);
        image.extend_from_slice(&VERSION.to_le_bytes());
        image.extend_from_slice(&2u32.to_le_bytes()); // input_dim
        image.extend_from_slice(&2u32.to_le_bytes()); // layer_count
        image.extend_from_slice(&clamp.to_le_bytes());
        for value in [0.0f32, 0.0] {
            image.extend_from_slice(&value.to_le_bytes()); // mean
        }
        for value in [1.0f32, 1.0] {
            image.extend_from_slice(&value.to_le_bytes()); // inverse std
        }
        // layer 0: 2 -> 2, identity with zero bias
        image.extend_from_slice(&2u32.to_le_bytes());
        image.extend_from_slice(&2u32.to_le_bytes());
        for value in [1.0f32, 0.0, 0.0, 1.0] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0f32, 0.0] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        // layer 1: 2 -> 1, sum with bias 0.5
        image.extend_from_slice(&2u32.to_le_bytes());
        image.extend_from_slice(&1u32.to_le_bytes());
        for value in [1.0f32, 1.0] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        image.extend_from_slice(&0.5f32.to_le_bytes());
        image
    }

    #[test]
    fn reads_and_predicts() {
        let model = VflModel::load(&tiny_image(8.0)).expect("loads");
        let mut scratch = model.scratch();
        // ReLU on layer 0 kills the negative, so 3 + 0 + 0.5.
        let value = model.predict(&[3.0, -1.0], &mut scratch).expect("predicts");
        assert!((value - 3.5).abs() < 1e-6, "got {value}");
    }

    #[test]
    fn clamp_is_applied() {
        let model = VflModel::load(&tiny_image(2.0)).expect("loads");
        let mut scratch = model.scratch();
        // Both inputs clamp to +2, so 2 + 2 + 0.5.
        let value = model.predict(&[100.0, 50.0], &mut scratch).expect("predicts");
        assert!((value - 4.5).abs() < 1e-6, "got {value}");
    }

    #[test]
    fn rejects_bad_images() {
        let mut wrong_magic = tiny_image(8.0);
        wrong_magic[0] = b'X';
        assert!(VflModel::load(&wrong_magic).is_err());

        let mut trailing = tiny_image(8.0);
        trailing.push(0);
        assert!(VflModel::load(&trailing).is_err());

        let truncated = tiny_image(8.0);
        assert!(VflModel::load(&truncated[..truncated.len() - 4]).is_err());
    }

    #[test]
    fn zero_inverse_std_pins_a_dim_to_its_mean() {
        let mut image = tiny_image(8.0);
        // inverse std lives right after the two mean floats, at offset 20 + 8.
        let at = 4 + 4 + 4 + 4 + 4 + 8;
        image[at..at + 4].copy_from_slice(&0.0f32.to_le_bytes());
        let model = VflModel::load(&image).expect("zero inverse std is legal");
        let mut scratch = model.scratch();
        let a = model.predict(&[1000.0, 1.0], &mut scratch).expect("predicts");
        let b = model.predict(&[-1000.0, 1.0], &mut scratch).expect("predicts");
        assert_eq!(a, b, "a pinned dim must not move the score");
    }

    #[test]
    fn argmax_prefers_the_lower_index_on_ties() {
        let model = VflModel::load(&tiny_image(8.0)).expect("loads");
        let mut scratch = model.scratch();
        let rows = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]];
        assert_eq!(model.argmax(&rows, &mut scratch).expect("argmax"), 0);
    }

    #[test]
    fn pinning_rejects_a_different_image() {
        let image = tiny_image(8.0);
        assert!(VflModel::load_pinned(&image, &"0".repeat(64)).is_err());
    }
}
