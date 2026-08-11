//! Counter-addressed deterministic RNG coordinates shared with Python M2.
//!
//! This module intentionally does not expose a mutable PRNG.  Every value is
//! addressed by semantic coordinates, so shards and candidate enumeration can
//! be reordered without changing chance samples.

use blake2b_simd::Params;
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

pub const COUNTER_RNG_SCHEMA: &str = "regular_ofc_counter_rng_v1";
const PERSONALIZATION: &[u8] = b"OFC-RNG-v1";
const SEED_MASK: u64 = (1_u64 << 63) - 1;

/// Python's `actor` coordinate can be a seat index or one of three names.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum CounterActor {
    Seat(u8),
    Hero,
    Opponent,
    Chance,
}

impl CounterActor {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Seat(0 | 1) | Self::Hero | Self::Opponent | Self::Chance => Ok(()),
            Self::Seat(value) => Err(format!("integer actor must be 0 or 1, got {value}")),
        }
    }

    fn json_value(&self) -> Value {
        match self {
            Self::Seat(value) => Value::Number(Number::from(*value)),
            Self::Hero => Value::String("hero".to_owned()),
            Self::Opponent => Value::String("opponent".to_owned()),
            Self::Chance => Value::String("chance".to_owned()),
        }
    }
}

/// Stable coordinates for one 63-bit deterministic value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CounterRngKey {
    pub base_seed: i64,
    pub run_id: String,
    pub phase: String,
    pub sample_index: u64,
    pub actor: CounterActor,
    pub street: String,
    pub stream: String,
    pub counter: u64,
    pub root_fingerprint: String,
}

impl CounterRngKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_seed: i64,
        run_id: impl Into<String>,
        phase: impl Into<String>,
        sample_index: u64,
        actor: CounterActor,
        street: impl Into<String>,
        stream: impl Into<String>,
        counter: u64,
        root_fingerprint: impl Into<String>,
    ) -> Result<Self, String> {
        let key = Self {
            base_seed,
            run_id: run_id.into(),
            phase: phase.into(),
            sample_index,
            actor,
            street: street.into(),
            stream: stream.into(),
            counter,
            root_fingerprint: root_fingerprint.into(),
        };
        key.validate()?;
        Ok(key)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.actor.validate()?;
        for (name, value) in [
            ("run_id", self.run_id.as_str()),
            ("phase", self.phase.as_str()),
            ("street", self.street.as_str()),
            ("stream", self.stream.as_str()),
        ] {
            if value.is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        Ok(())
    }

    /// JSON payload with the exact Python schema. Canonicalization happens at
    /// hash time and recursively sorts keys.
    pub fn payload(&self) -> Value {
        let mut payload = Map::new();
        payload.insert(
            "schema".to_owned(),
            Value::String(COUNTER_RNG_SCHEMA.to_owned()),
        );
        payload.insert(
            "base_seed".to_owned(),
            Value::Number(Number::from(self.base_seed)),
        );
        payload.insert("run_id".to_owned(), Value::String(self.run_id.clone()));
        payload.insert("phase".to_owned(), Value::String(self.phase.clone()));
        payload.insert(
            "sample_index".to_owned(),
            Value::Number(Number::from(self.sample_index)),
        );
        payload.insert("actor".to_owned(), self.actor.json_value());
        payload.insert("street".to_owned(), Value::String(self.street.clone()));
        payload.insert("stream".to_owned(), Value::String(self.stream.clone()));
        payload.insert(
            "counter".to_owned(),
            Value::Number(Number::from(self.counter)),
        );
        payload.insert(
            "root_fingerprint".to_owned(),
            Value::String(self.root_fingerprint.clone()),
        );
        Value::Object(payload)
    }

    /// Python parity: BLAKE2b-128 with `person=b"OFC-RNG-v1"`, first eight
    /// bytes as big endian, then masked into the non-negative 63-bit domain.
    pub fn seed(&self) -> u64 {
        // Invalid public struct literals fail closed instead of silently using
        // a coordinate outside the Python contract.
        self.validate().expect("invalid CounterRngKey");
        let encoded = canonical_json_ascii(&self.payload());
        let digest = Params::new()
            .hash_length(16)
            .personal(PERSONALIZATION)
            .hash(encoded.as_bytes());
        let mut first = [0_u8; 8];
        first.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_be_bytes(first) & SEED_MASK
    }
}

pub fn common_future_seed(
    base_seed: i64,
    run_id: &str,
    root_fingerprint: &str,
    sample_index: u64,
    street: &str,
    stream: &str,
) -> Result<u64, String> {
    Ok(CounterRngKey::new(
        base_seed,
        run_id,
        "common_future",
        sample_index,
        CounterActor::Chance,
        street,
        stream,
        0,
        root_fingerprint,
    )?
    .seed())
}

#[allow(clippy::too_many_arguments)]
pub fn policy_decision_seed(
    base_seed: i64,
    run_id: &str,
    root_fingerprint: &str,
    future_index: u64,
    actor: CounterActor,
    street: &str,
    decision_ordinal: u64,
    stream: &str,
) -> Result<u64, String> {
    if matches!(actor, CounterActor::Chance) {
        return Err("policy actor must be hero, opponent, or seat 0/1".to_owned());
    }
    Ok(CounterRngKey::new(
        base_seed,
        run_id,
        "rollout_policy",
        future_index,
        actor,
        street,
        stream,
        decision_ordinal,
        root_fingerprint,
    )?
    .seed())
}

/// Equivalent to Python `json.dumps(value, sort_keys=True, separators=(",", ":"),
/// ensure_ascii=True)` for the JSON types used by M2/M3 artifacts.
pub(crate) fn canonical_json_ascii(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => json_ascii_string(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json_ascii)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|key| format!(
                        "{}:{}",
                        json_ascii_string(key),
                        canonical_json_ascii(&values[key])
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

fn json_ascii_string(value: &str) -> String {
    // serde_json already handles quotes, slashes and control characters. Its
    // only difference from ensure_ascii=True is leaving non-ASCII scalar
    // values literal, so escape those while preserving its existing escapes.
    let serialized = serde_json::to_string(value).expect("string serialization cannot fail");
    let mut output = String::with_capacity(serialized.len());
    for character in serialized.chars() {
        if character.is_ascii() {
            output.push(character);
        } else {
            let scalar = character as u32;
            if scalar <= 0xffff {
                output.push_str(&format!("\\u{scalar:04x}"));
            } else {
                let adjusted = scalar - 0x1_0000;
                let high = 0xd800 + (adjusted >> 10);
                let low = 0xdc00 + (adjusted & 0x3ff);
                output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    output
}

pub(crate) fn sha256_hex_json(value: &Value) -> String {
    sha256_hex(canonical_json_ascii(value).as_bytes())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_counter_rng_golden_vector() {
        let key = CounterRngKey::new(
            2026071201,
            "m1-smoke",
            "common_future",
            7,
            CounterActor::Chance,
            "T2",
            "future_cards",
            3,
            "abc123",
        )
        .unwrap();
        assert_eq!(key.seed(), 1_867_823_025_197_256_593);
        assert_eq!(key.seed(), key.clone().seed());
    }

    #[test]
    fn canonical_json_matches_python_ascii_and_sorting() {
        let value = serde_json::json!({"z": "é😀", "a": {"y": 2, "x": true}});
        assert_eq!(
            canonical_json_ascii(&value),
            r#"{"a":{"x":true,"y":2},"z":"\u00e9\ud83d\ude00"}"#
        );
    }

    #[test]
    fn coordinates_are_domain_separated() {
        let base = |sample, actor, street: &str| {
            CounterRngKey::new(5, "run", "phase", sample, actor, street, "default", 0, "")
                .unwrap()
                .seed()
        };
        let seeds = [
            base(1, CounterActor::Seat(0), "T2"),
            base(2, CounterActor::Seat(0), "T2"),
            base(1, CounterActor::Seat(1), "T2"),
            base(1, CounterActor::Seat(0), "T3"),
        ];
        for left in 0..seeds.len() {
            for right in left + 1..seeds.len() {
                assert_ne!(seeds[left], seeds[right]);
            }
        }
    }
}
