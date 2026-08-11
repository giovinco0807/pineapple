//! Inference for the learned T4-first evaluator.
//!
//! Like [`crate::t4_features`], this is not wired into [`crate::search`]. It
//! evaluates a feature vector and nothing else.
//!
//! A dense network rather than the boosted ensemble that scored marginally
//! better: the search reaches its leaves one position at a time, and measured
//! at batch size one the ensemble cost 9,006 us against an exact first-seat
//! solve of roughly 1,100 us -- slower than the thing it would replace. Four
//! matrix multiplies also port without bringing a tree format or a dependency
//! along with them.
//!
//! Weights arrive as a flat little-endian image so they can be pinned by digest
//! the way this project pins its other inputs. A model paired with a drifted
//! encoder is silently wrong rather than loudly broken, so [`Model::load_pinned`]
//! exists and callers on any path that matters should prefer it.

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"T4M1";
const VERSION: u32 = 1;
/// Version 2 adds a clamp and stores `inverse_std` where version 1 stores
/// `std`. Both are accepted; version 1 keeps an infinite clamp so its
/// arithmetic is bit-for-bit what it always was.
const VERSION_CLAMPED: u32 = 2;

/// One fully connected layer, weights row-major as `[output][input]`.
struct Layer {
    inputs: usize,
    outputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

pub struct Model {
    input_dim: usize,
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
    layers: Vec<Layer>,
    widest: usize,
    /// Standardised features are held to +/-this before the first layer.
    /// Infinite for version 1 images, which were fitted without one.
    clamp: f32,
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, String> {
        let end = self.offset + 4;
        if end > self.bytes.len() {
            return Err("model image ends inside a length field".to_owned());
        }
        let value = u32::from_le_bytes(self.bytes[self.offset..end].try_into().unwrap());
        self.offset = end;
        Ok(value)
    }

    fn floats(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let end = self.offset + count * 4;
        if end > self.bytes.len() {
            return Err(format!("model image ends inside a block of {count} floats"));
        }
        let values = self.bytes[self.offset..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        self.offset = end;
        Ok(values)
    }
}

impl Model {
    /// Load without checking the digest. Use [`Model::load_pinned`] anywhere the
    /// identity of the weights matters.
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[..4] != MAGIC {
            return Err("model image does not start with the expected magic".to_owned());
        }
        let mut reader = Reader {
            bytes,
            offset: 4,
        };
        let version = reader.u32()?;
        if version != VERSION && version != VERSION_CLAMPED {
            return Err(format!("model image version {version} is not supported"));
        }
        let layer_count = reader.u32()? as usize;
        if layer_count == 0 {
            return Err("model image declares no layers".to_owned());
        }
        let input_dim = reader.u32()? as usize;
        let clamp = if version == VERSION_CLAMPED {
            let value = reader.floats(1)?[0];
            if !(value > 0.0) {
                return Err("clamped model image declares a non-positive clamp".to_owned());
            }
            value
        } else {
            f32::INFINITY
        };
        let mean = reader.floats(input_dim)?;
        let inverse_std: Vec<f32> = if version == VERSION_CLAMPED {
            // Stored directly, and zero is meaningful: a feature with no spread
            // is pinned to its mean rather than divided by something tiny.
            let values = reader.floats(input_dim)?;
            if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err("model image has a negative or non-finite inverse std".to_owned());
            }
            values
        } else {
            let std = reader.floats(input_dim)?;
            if std.iter().any(|value| !value.is_finite() || *value == 0.0) {
                return Err("model image has a zero or non-finite standard deviation".to_owned());
            }
            std.iter().map(|value| 1.0 / value).collect()
        };

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
                "model image has {} trailing bytes",
                bytes.len() - reader.offset
            ));
        }
        let widest = layers.iter().map(|layer| layer.outputs).max().unwrap_or(1);
        Ok(Self {
            input_dim,
            mean,
            inverse_std,
            layers,
            widest,
            clamp,
        })
    }

    /// Load and refuse anything whose bytes are not the expected ones.
    pub fn load_pinned(bytes: &[u8], expected_sha256: &str) -> Result<Self, String> {
        let digest = hex(&Sha256::digest(bytes));
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

    /// Scratch space, so a caller scoring many actions allocates once.
    pub fn scratch(&self) -> Scratch {
        Scratch {
            current: vec![0.0; self.widest.max(self.input_dim)],
            next: vec![0.0; self.widest.max(self.input_dim)],
        }
    }

    pub fn predict(&self, features: &[f32]) -> Result<f32, String> {
        let mut scratch = self.scratch();
        self.predict_with(features, &mut scratch)
    }

    pub fn predict_with(&self, features: &[f32], scratch: &mut Scratch) -> Result<f32, String> {
        if features.len() != self.input_dim {
            return Err(format!(
                "expected {} features, got {}",
                self.input_dim,
                features.len()
            ));
        }
        for index in 0..self.input_dim {
            let standardised =
                (features[index] - self.mean[index]) * self.inverse_std[index];
            scratch.current[index] = standardised.clamp(-self.clamp, self.clamp);
        }
        let last = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            let input = &scratch.current[..layer.inputs];
            for output in 0..layer.outputs {
                let row = &layer.weight[output * layer.inputs..(output + 1) * layer.inputs];
                let sum = layer.bias[output] + dot(row, input);
                scratch.next[output] = if index == last { sum } else { sum.max(0.0) };
            }
            std::mem::swap(&mut scratch.current, &mut scratch.next);
        }
        Ok(scratch.current[0])
    }
}

pub struct Scratch {
    current: Vec<f32>,
    next: Vec<f32>,
}

/// Dot product with independent partial sums.
///
/// Floating point addition is not associative, so a single running accumulator
/// forces the compiler to keep the additions in order and the loop stays
/// scalar. Splitting the accumulator lets it vectorise; measured on this
/// network that is the difference between 51 us and single digits per action,
/// which decides whether the evaluator is cheaper than the solve it replaces.
/// The summation order differs from PyTorch's, which the parity fixture
/// tolerates because it compares against a bound rather than bit equality.
const LANES: usize = 8;

fn dot(left: &[f32], right: &[f32]) -> f32 {
    let mut partial = [0.0f32; LANES];
    let mut left_chunks = left.chunks_exact(LANES);
    let mut right_chunks = right.chunks_exact(LANES);
    for (a, b) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        for lane in 0..LANES {
            partial[lane] += a[lane] * b[lane];
        }
    }
    let mut sum = 0.0;
    for value in partial {
        sum += value;
    }
    for (a, b) in left_chunks
        .remainder()
        .iter()
        .zip(right_chunks.remainder())
    {
        sum += a * b;
    }
    sum
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_image() -> Vec<u8> {
        let mut image = Vec::new();
        image.extend_from_slice(MAGIC);
        image.extend_from_slice(&VERSION.to_le_bytes());
        image.extend_from_slice(&1u32.to_le_bytes());
        image.extend_from_slice(&2u32.to_le_bytes());
        for value in [0.0f32, 0.0] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        for value in [1.0f32, 1.0] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        image.extend_from_slice(&2u32.to_le_bytes());
        image.extend_from_slice(&1u32.to_le_bytes());
        for value in [2.0f32, 3.0] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        image.extend_from_slice(&0.5f32.to_le_bytes());
        image
    }

    #[test]
    fn a_single_layer_image_round_trips_through_the_forward_pass() {
        let model = Model::load(&tiny_image()).expect("image loads");
        assert_eq!(model.input_dim(), 2);
        let value = model.predict(&[1.0, 1.0]).expect("predicts");
        assert!((value - 5.5).abs() < 1e-6, "got {value}");
    }

    #[test]
    fn a_truncated_or_extended_image_is_rejected_rather_than_guessed_at() {
        let image = tiny_image();
        assert!(Model::load(&image[..image.len() - 4]).is_err());
        let mut longer = image.clone();
        longer.push(0);
        assert!(Model::load(&longer).is_err());
        let mut wrong_magic = image.clone();
        wrong_magic[0] = b'X';
        assert!(Model::load(&wrong_magic).is_err());
    }

    #[test]
    fn a_digest_that_does_not_match_refuses_to_load() {
        let image = tiny_image();
        let digest = hex(&Sha256::digest(&image));
        assert!(Model::load_pinned(&image, &digest).is_ok());
        assert!(Model::load_pinned(&image, &"0".repeat(64)).is_err());
    }

    #[test]
    fn a_feature_vector_of_the_wrong_width_is_refused() {
        let model = Model::load(&tiny_image()).expect("image loads");
        assert!(model.predict(&[1.0]).is_err());
        assert!(model.predict(&[1.0, 1.0, 1.0]).is_err());
    }
}
