//! Policy-facing full-hand observation contract.

use crate::{history::PublicPlacement, HuRlError, HuRlResult};
use ofc_hu_m3_engine::{
    action::Action,
    action_key::{
        action_key, generate_canonical_actions, legal_action_set_digest,
        ordered_action_mapping_digest, ActionKey, ACTION_KEY_SCHEMA,
    },
    cards::Card,
    infoset::{ActorObservation, Seat, Street},
    state::{Board, ALL_ROWS},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const LEGAL_ACTION_MAPPING_SCHEMA: &str = "regular_ofc_hu_rl_legal_action_mapping_v1";
pub const HU_RL_ACTOR_VIEW_SCHEMA: &str = "regular_ofc_hu_rl_actor_view_v1";
pub const MAX_LEGAL_ACTIONS: usize = 232;

/// Exact tensor-index to semantic-action mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegalActionMappingV1 {
    action_keys: Vec<ActionKey>,
    action_set_digest: String,
    action_order_digest: String,
}

impl LegalActionMappingV1 {
    pub fn for_observation(observation: &ActorObservation) -> HuRlResult<Self> {
        observation.validate().map_err(HuRlError::from)?;
        let actions = generate_canonical_actions(&observation.hero_board, &observation.dealt_cards)
            .map_err(HuRlError::from)?;
        Self::from_ordered_actions(&actions)
    }

    pub fn from_ordered_actions(actions: &[Action]) -> HuRlResult<Self> {
        if actions.is_empty() {
            return Err(HuRlError::new("legal action mapping must not be empty"));
        }
        if actions.len() > MAX_LEGAL_ACTIONS {
            return Err(HuRlError::new(format!(
                "legal action mapping exceeds capacity {MAX_LEGAL_ACTIONS}"
            )));
        }
        let action_keys = actions
            .iter()
            .map(action_key)
            .collect::<Result<Vec<_>, _>>()
            .map_err(HuRlError::from)?;
        if action_keys.iter().copied().collect::<HashSet<_>>().len() != action_keys.len() {
            return Err(HuRlError::new(
                "legal action mapping contains duplicate ActionKeys",
            ));
        }
        Ok(Self {
            action_keys,
            action_set_digest: legal_action_set_digest(actions).map_err(HuRlError::from)?,
            action_order_digest: ordered_action_mapping_digest(actions).map_err(HuRlError::from)?,
        })
    }

    pub fn action_keys(&self) -> &[ActionKey] {
        &self.action_keys
    }

    pub fn action_count(&self) -> usize {
        self.action_keys.len()
    }

    pub fn action_set_digest(&self) -> &str {
        &self.action_set_digest
    }

    pub fn action_order_digest(&self) -> &str {
        &self.action_order_digest
    }

    pub fn key_at(&self, index: usize) -> HuRlResult<ActionKey> {
        self.action_keys
            .get(index)
            .copied()
            .ok_or_else(|| HuRlError::new("legal action index is out of range"))
    }

    pub fn index_for(&self, key: ActionKey) -> HuRlResult<usize> {
        self.action_keys
            .iter()
            .position(|candidate| *candidate == key)
            .ok_or_else(|| HuRlError::new(format!("ActionKey is not legal: {}", key.to_token())))
    }

    pub fn mask(&self) -> [bool; MAX_LEGAL_ACTIONS] {
        let mut mask = [false; MAX_LEGAL_ACTIONS];
        mask[..self.action_keys.len()].fill(true);
        mask
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema": LEGAL_ACTION_MAPPING_SCHEMA,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "max_actions": MAX_LEGAL_ACTIONS,
            "action_count": self.action_count(),
            "action_keys": self.action_keys.iter().map(ActionKey::to_token).collect::<Vec<_>>(),
            "action_set_digest": self.action_set_digest,
            "action_order_digest": self.action_order_digest,
        })
    }
}

/// Complete normal-hand input visible to one policy decision.
#[derive(Clone, Debug, PartialEq)]
pub struct HuRlActorViewV1 {
    observation: ActorObservation,
    public_history: Vec<PublicPlacement>,
    legal_action_mapping: LegalActionMappingV1,
}

impl HuRlActorViewV1 {
    pub fn from_observation(
        observation: ActorObservation,
        public_history: Vec<PublicPlacement>,
    ) -> HuRlResult<Self> {
        observation.validate().map_err(HuRlError::from)?;
        validate_public_history(&observation, &public_history)?;
        let legal_action_mapping = LegalActionMappingV1::for_observation(&observation)?;
        Ok(Self {
            observation,
            public_history,
            legal_action_mapping,
        })
    }

    pub fn observation(&self) -> &ActorObservation {
        &self.observation
    }

    pub fn public_history(&self) -> &[PublicPlacement] {
        &self.public_history
    }

    pub fn legal_action_mapping(&self) -> &LegalActionMappingV1 {
        &self.legal_action_mapping
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema": HU_RL_ACTOR_VIEW_SCHEMA,
            "observation": canonical_observation_json(&self.observation),
            "public_history": self.public_history.iter().map(PublicPlacement::to_json).collect::<Vec<_>>(),
            "legal_action_mapping": self.legal_action_mapping.to_json(),
        })
    }

    pub fn canonical_json(&self) -> String {
        canonical_json_ascii(&self.to_json())
    }

    pub fn digest(&self) -> String {
        Sha256::digest(self.canonical_json().as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn validate_public_history(
    observation: &ActorObservation,
    history: &[PublicPlacement],
) -> HuRlResult<()> {
    let current_order = event_order(observation.street, observation.seat);
    let mut previous_order: Option<usize> = None;
    let mut seen_events = HashSet::new();
    let mut seen_cards = 0_u64;
    let mut seen_by_seat = [0_u64; 2];
    let mut seen_by_seat_row = [[0_u64; 3]; 2];
    let hero_board_mask = cards_mask(&observation.hero_board.all_cards());
    let opponent_board_mask = cards_mask(&observation.opponent_public_board.all_cards());
    for event in history {
        event.validate()?;
        let order = event_order(event.street(), event.acting_seat());
        if !seen_events.insert((event.street(), event.acting_seat())) {
            return Err(HuRlError::new(
                "public_history contains a duplicate seat/street event",
            ));
        }
        if previous_order.is_some_and(|previous| order <= previous) {
            return Err(HuRlError::new(
                "public_history is not in canonical chronological order",
            ));
        }
        if order >= current_order {
            return Err(HuRlError::new(
                "public_history contains the current or a future action",
            ));
        }
        if seen_cards & event.placement_mask() != 0 {
            return Err(HuRlError::new(
                "public_history places the same card more than once",
            ));
        }
        let visible_board = if event.acting_seat() == observation.seat {
            &observation.hero_board
        } else {
            &observation.opponent_public_board
        };
        for (row_index, row) in ALL_ROWS.iter().copied().enumerate() {
            let placement_mask = event.placement_masks()[row_index];
            let visible_row_mask = cards_mask(visible_board.cards(row));
            if placement_mask & !visible_row_mask != 0 {
                return Err(HuRlError::new(
                    "public_history placement rows disagree with the public boards",
                ));
            }
            seen_by_seat_row[seat_index(event.acting_seat())][row_index] |= placement_mask;
        }
        previous_order = Some(order);
        seen_cards |= event.placement_mask();
        seen_by_seat[seat_index(event.acting_seat())] |= event.placement_mask();
    }
    if history.len() != current_order {
        return Err(HuRlError::new(
            "public_history is not the complete trajectory prefix for this decision",
        ));
    }
    let (first_board, second_board) = match observation.seat {
        Seat::First => (&observation.hero_board, &observation.opponent_public_board),
        Seat::Second => (&observation.opponent_public_board, &observation.hero_board),
    };
    for (seat, board) in [first_board, second_board].into_iter().enumerate() {
        if seen_by_seat[seat] != cards_mask(&board.all_cards()) {
            return Err(HuRlError::new(
                "public_history does not exactly reconstruct the public board",
            ));
        }
        for (row_index, row) in ALL_ROWS.iter().copied().enumerate() {
            if seen_by_seat_row[seat][row_index] != cards_mask(board.cards(row)) {
                return Err(HuRlError::new(
                    "public_history does not exactly reconstruct every public board row",
                ));
            }
        }
    }
    if seen_cards != hero_board_mask | opponent_board_mask {
        return Err(HuRlError::new(
            "public_history does not exactly reconstruct both public boards",
        ));
    }
    Ok(())
}

const fn seat_index(seat: Seat) -> usize {
    match seat {
        Seat::First => 0,
        Seat::Second => 1,
    }
}

fn event_order(street: Street, seat: Seat) -> usize {
    let street = match street {
        Street::T0 => 0,
        Street::T1 => 1,
        Street::T2 => 2,
        Street::T3 => 3,
        Street::T4 => 4,
    };
    let seat = match seat {
        Seat::First => 0,
        Seat::Second => 1,
    };
    street * 2 + seat
}

fn canonical_observation_json(observation: &ActorObservation) -> Value {
    json!({
        "schema": ofc_hu_m3_engine::infoset::OBSERVATION_SCHEMA,
        "hero_board": sorted_board_json(&observation.hero_board),
        "opponent_public_board": sorted_board_json(&observation.opponent_public_board),
        "dealt_cards": sorted_cards(&observation.dealt_cards),
        "hero_private_discards": sorted_cards(&observation.hero_private_discards),
        "seat": observation.seat,
        "street": observation.street,
        "to_act_order": observation.to_act_order,
        "scoring": observation.scoring,
        "hero_in_fantasyland": observation.hero_in_fantasyland,
        "opponent_in_fantasyland": observation.opponent_in_fantasyland,
        "opponent_discard_count": observation.opponent_discard_count(),
    })
}

fn sorted_board_json(board: &Board) -> Value {
    json!({
        "top": sorted_cards(&board.top),
        "middle": sorted_cards(&board.middle),
        "bottom": sorted_cards(&board.bottom),
    })
}

fn sorted_cards(cards: &[Card]) -> Vec<Card> {
    let mut sorted = cards.to_vec();
    sorted.sort_unstable_by_key(|card| card.index());
    sorted
}

fn cards_mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0_u64, |mask, card| mask | card.bit())
}

/// Python-compatible canonical JSON for the ASCII-only actor contract.
fn canonical_json_ascii(value: &Value) -> String {
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
