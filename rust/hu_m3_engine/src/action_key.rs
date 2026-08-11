//! Stable, card-order-independent semantic action identities.

use crate::action::{generate_actions, Action};
use crate::cards::{Card, ALL_CARDS};
use crate::state::{Board, Row};
use serde::{de, ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

pub const ACTION_KEY_SCHEMA: &str = "regular_ofc_action_key_v1";
const ACTION_KEY_PREFIX: &str = "rak1";
const MASK_HEX_WIDTH: usize = 13;
const MAX_MASK: u64 = (1_u64 << 52) - 1;

/// Four disjoint 52-bit masks, ordered top/middle/bottom/discard.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ActionKey {
    pub top_mask: u64,
    pub middle_mask: u64,
    pub bottom_mask: u64,
    pub discard_mask: u64,
}

impl ActionKey {
    pub fn new(
        top_mask: u64,
        middle_mask: u64,
        bottom_mask: u64,
        discard_mask: u64,
    ) -> Result<Self, String> {
        let key = Self {
            top_mask,
            middle_mask,
            bottom_mask,
            discard_mask,
        };
        key.validate()?;
        Ok(key)
    }

    pub fn validate(&self) -> Result<(), String> {
        let masks = self.masks();
        if masks.iter().any(|&mask| mask > MAX_MASK) {
            return Err("ActionKey mask is outside the 52-card domain".to_string());
        }
        let mut union = 0_u64;
        for mask in masks {
            if union & mask != 0 {
                return Err("ActionKey masks must be pairwise disjoint".to_string());
            }
            union |= mask;
        }
        Ok(())
    }

    pub const fn masks(&self) -> [u64; 4] {
        [
            self.top_mask,
            self.middle_mask,
            self.bottom_mask,
            self.discard_mask,
        ]
    }

    pub fn from_action(action: &Action) -> Result<Self, String> {
        action.validate()?;
        let mut top = 0_u64;
        let mut middle = 0_u64;
        let mut bottom = 0_u64;
        for &(card, row) in &action.placements {
            match row {
                Row::Top => top |= card.bit(),
                Row::Middle => middle |= card.bit(),
                Row::Bottom => bottom |= card.bit(),
            }
        }
        let discard = action
            .discards
            .iter()
            .fold(0_u64, |mask, card| mask | card.bit());
        Self::new(top, middle, bottom, discard)
    }

    pub fn from_token(token: &str) -> Result<Self, String> {
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() != 5 || parts[0] != ACTION_KEY_PREFIX {
            return Err(format!("invalid ActionKey token: {token:?}"));
        }
        let mut masks = [0_u64; 4];
        for (index, part) in parts[1..].iter().enumerate() {
            if part.len() != MASK_HEX_WIDTH
                || !part.bytes().all(|character| {
                    character.is_ascii_digit() || (b'a'..=b'f').contains(&character)
                })
            {
                return Err(format!("invalid ActionKey mask encoding: {token:?}"));
            }
            masks[index] = u64::from_str_radix(part, 16)
                .map_err(|_| format!("invalid ActionKey mask encoding: {token:?}"))?;
        }
        Self::new(masks[0], masks[1], masks[2], masks[3])
    }

    pub fn to_token(&self) -> String {
        format!(
            "{ACTION_KEY_PREFIX}:{:013x}:{:013x}:{:013x}:{:013x}",
            self.top_mask, self.middle_mask, self.bottom_mask, self.discard_mask
        )
    }

    pub fn stable_token(&self) -> String {
        self.to_token()
    }

    pub fn cards(&self, group: &str) -> Result<Vec<Card>, String> {
        let mask = match group {
            "top" => self.top_mask,
            "middle" => self.middle_mask,
            "bottom" => self.bottom_mask,
            "discards" => self.discard_mask,
            _ => return Err(format!("unknown ActionKey group: {group}")),
        };
        Ok(ALL_CARDS
            .iter()
            .copied()
            .filter(|card| mask & card.bit() != 0)
            .collect())
    }
}

impl Serialize for ActionKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ActionKey", 2)?;
        state.serialize_field("schema", ACTION_KEY_SCHEMA)?;
        state.serialize_field("token", &self.to_token())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ActionKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Payload {
            schema: String,
            token: String,
        }
        let payload = Payload::deserialize(deserializer)?;
        if payload.schema != ACTION_KEY_SCHEMA {
            return Err(de::Error::custom(format!(
                "unsupported action key schema: {:?}",
                payload.schema
            )));
        }
        Self::from_token(&payload.token).map_err(de::Error::custom)
    }
}

pub fn action_key(action: &Action) -> Result<ActionKey, String> {
    ActionKey::from_action(action)
}

pub fn canonicalize_actions(actions: &[Action]) -> Result<Vec<Action>, String> {
    let mut keyed = actions
        .iter()
        .map(|action| Ok((action_key(action)?, action.clone())))
        .collect::<Result<Vec<_>, String>>()?;
    ensure_unique_keys(keyed.iter().map(|(key, _)| *key))?;
    keyed.sort_by_key(|(key, _)| *key);
    Ok(keyed.into_iter().map(|(_, action)| action).collect())
}

pub fn generate_canonical_actions(
    board: &Board,
    dealt_cards: &[Card],
) -> Result<Vec<Action>, String> {
    canonicalize_actions(&generate_actions(board, dealt_cards)?)
}

pub fn canonical_argmax_index(values: &[f64], actions: &[Action]) -> Result<usize, String> {
    if values.is_empty() || values.len() != actions.len() {
        return Err("values/actions length mismatch".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("values must be finite".to_string());
    }
    let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut tied = values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value == best).then_some(index));
    let first = tied.next().expect("non-empty values produce a best value");
    tied.try_fold(first, |best_index, index| {
        Ok(
            if action_key(&actions[index])? < action_key(&actions[best_index])? {
                index
            } else {
                best_index
            },
        )
    })
}

pub fn canonical_descending_indices(
    values: &[f64],
    actions: &[Action],
) -> Result<Vec<usize>, String> {
    if values.len() != actions.len() {
        return Err("values/actions length mismatch".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("values must be finite".to_string());
    }
    let keys = actions
        .iter()
        .map(action_key)
        .collect::<Result<Vec<_>, _>>()?;
    let mut indices: Vec<usize> = (0..actions.len()).collect();
    indices.sort_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| keys[left].cmp(&keys[right]))
    });
    Ok(indices)
}

pub fn index_actions_by_key(actions: &[Action]) -> Result<HashMap<ActionKey, usize>, String> {
    let mut mapping = HashMap::new();
    for (index, action) in actions.iter().enumerate() {
        let key = action_key(action)?;
        if let Some(previous) = mapping.insert(key, index) {
            return Err(format!(
                "duplicate semantic action key at indices {previous} and {index}"
            ));
        }
    }
    Ok(mapping)
}

pub fn resolve_action_key(actions: &[Action], key: ActionKey) -> Result<usize, String> {
    index_actions_by_key(actions)?
        .get(&key)
        .copied()
        .ok_or_else(|| format!("action key is not legal in this state: {}", key.to_token()))
}

pub fn ordered_action_mapping_digest(actions: &[Action]) -> Result<String, String> {
    let payload = actions
        .iter()
        .map(|action| Ok(action_key(action)?.to_token()))
        .collect::<Result<Vec<_>, String>>()?
        .join("\n");
    Ok(hex_sha256(payload.as_bytes()))
}

pub fn legal_action_set_digest(actions: &[Action]) -> Result<String, String> {
    let keys = actions
        .iter()
        .map(action_key)
        .collect::<Result<Vec<_>, _>>()?;
    ensure_unique_keys(keys.iter().copied())?;
    let payload = keys
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|key| key.to_token())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(hex_sha256(payload.as_bytes()))
}

fn ensure_unique_keys(keys: impl IntoIterator<Item = ActionKey>) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for key in keys {
        if !seen.insert(key) {
            return Err(format!("duplicate semantic action key: {}", key.to_token()));
        }
    }
    Ok(())
}

fn hex_sha256(payload: &[u8]) -> String {
    Sha256::digest(payload)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(value: &str) -> Card {
        value.parse().unwrap()
    }

    #[test]
    fn token_is_a_python_golden_fixture() {
        let action = Action::new(
            vec![
                (card("Ah"), Row::Top),
                (card("2h"), Row::Top),
                (card("Kd"), Row::Middle),
            ],
            vec![card("Qs"), card("3c")],
        )
        .unwrap();
        let key = action_key(&action).unwrap();
        assert_eq!(
            key.to_token(),
            "rak1:0000000001001:0000001000000:0000000000000:2000008000000"
        );
        assert_eq!(ActionKey::from_token(&key.to_token()).unwrap(), key);
        assert_eq!(key.cards("top").unwrap(), vec![card("2h"), card("Ah")]);
        assert_eq!(key.cards("discards").unwrap(), vec![card("3c"), card("Qs")]);
        let encoded = serde_json::to_string(&key).unwrap();
        assert_eq!(serde_json::from_str::<ActionKey>(&encoded).unwrap(), key);
    }

    #[test]
    fn canonical_tie_break_is_dealt_order_independent() {
        let board = Board::empty();
        let first = vec![card("Ah"), card("Kd"), card("Qc"), card("Js"), card("Th")];
        let second = vec![card("Th"), card("Js"), card("Qc"), card("Kd"), card("Ah")];
        let left = generate_canonical_actions(&board, &first).unwrap();
        let right = generate_canonical_actions(&board, &second).unwrap();
        let left_keys = left
            .iter()
            .map(action_key)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let right_keys = right
            .iter()
            .map(action_key)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(left_keys, right_keys);
        assert_eq!(left_keys.len(), 232);
        assert_eq!(
            canonical_argmax_index(&vec![0.0; left.len()], &left).unwrap(),
            0
        );
    }

    #[test]
    fn overlapping_masks_and_duplicate_keys_fail_closed() {
        assert!(ActionKey::new(1, 0, 0, 1).unwrap_err().contains("disjoint"));
        let action = Action::new(vec![(card("Ah"), Row::Top)], vec![card("Kd")]).unwrap();
        assert!(legal_action_set_digest(&[action.clone(), action])
            .unwrap_err()
            .contains("duplicate"));
    }
}
