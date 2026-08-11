//! Bounded JSON validation bridge for scalar Python/Rust parity.

use crate::{Board, Card, HuRlError, HuRlResult, ScalarHuRlEnv, DECISION_COUNT};
use serde::Deserialize;
use serde_json::{json, Value};

pub const HU_RL_SCALAR_TRACE_REQUEST_SCHEMA: &str = "regular_ofc_hu_rl_scalar_trace_request_v1";
pub const HU_RL_SCALAR_TRACE_RESULT_SCHEMA: &str = "regular_ofc_hu_rl_scalar_trace_result_v1";
pub const HU_RL_SCALAR_TRACE_ARTIFACT_ROLE: &str = "privileged_correctness_audit_only";
pub const MAX_SCALAR_TRACE_REQUEST_BYTES: usize = 16 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalarTraceRequest {
    schema: String,
    explicit_deck: Vec<Card>,
    selected_indices: Vec<usize>,
}

/// Decode one bounded request from JSON bytes and execute its ten decisions.
///
/// Parse failures are deliberately collapsed to a field/type error so neither
/// an invalid card token nor any other request value is reflected to callers.
pub fn run_scalar_trace_json(input: &[u8]) -> HuRlResult<Value> {
    if input.len() > MAX_SCALAR_TRACE_REQUEST_BYTES {
        return Err(HuRlError::new("scalar trace request exceeds byte limit"));
    }
    let request: ScalarTraceRequest = serde_json::from_slice(input)
        .map_err(|_| HuRlError::new("scalar trace request has invalid fields or types"))?;
    evaluate_decoded_request(request)
}

/// Value-based convenience boundary for in-process validators.
pub fn evaluate_scalar_trace_request(request: Value) -> HuRlResult<Value> {
    let encoded = serde_json::to_vec(&request)
        .map_err(|_| HuRlError::new("scalar trace request cannot be encoded"))?;
    run_scalar_trace_json(&encoded)
}

fn evaluate_decoded_request(request: ScalarTraceRequest) -> HuRlResult<Value> {
    if request.schema != HU_RL_SCALAR_TRACE_REQUEST_SCHEMA {
        return Err(HuRlError::new("unsupported scalar trace request schema"));
    }
    if request.explicit_deck.len() != 52 {
        return Err(HuRlError::new(
            "explicit_deck must contain exactly 52 cards",
        ));
    }
    if request.selected_indices.len() != DECISION_COUNT {
        return Err(HuRlError::new(
            "selected_indices must contain exactly 10 entries",
        ));
    }

    // Collapse all deck-domain/duplicate diagnostics. The scalar environment's
    // detailed error can name a duplicate card, which is useful internally but
    // must not echo request values at this external validation boundary.
    let mut environment = ScalarHuRlEnv::new(&request.explicit_deck)
        .map_err(|_| HuRlError::new("explicit_deck is not a valid complete regular deck"))?;
    let mut decisions = Vec::with_capacity(DECISION_COUNT);

    for (ordinal, &selected_index) in request.selected_indices.iter().enumerate() {
        let view = environment
            .observe()
            .map_err(|_| trace_stage_error("observation", ordinal))?;
        let mapping = view.legal_action_mapping();
        let selected_action = mapping.key_at(selected_index).map_err(|_| {
            HuRlError::new(format!(
                "selected_indices entry is outside the legal range at decision {ordinal}"
            ))
        })?;
        let actor_view = view.to_json();
        let actor_view_digest = view.digest();
        let action_count = mapping.action_count();
        let action_set_digest = mapping.action_set_digest().to_owned();
        let action_order_digest = mapping.action_order_digest().to_owned();

        let step = environment
            .step(selected_action)
            .map_err(|_| trace_stage_error("transition", ordinal))?;
        decisions.push(json!({
            "ordinal": ordinal,
            "actor": step.actor,
            "street": step.street,
            "actor_view": actor_view,
            "actor_view_digest": actor_view_digest,
            "legal_action_mapping": {
                "action_count": action_count,
                "action_set_digest": action_set_digest,
                "action_order_digest": action_order_digest,
            },
            "selected_index": selected_index,
            "selected_action_key": selected_action.to_token(),
            "step": {
                "public_event": step.public_placement.to_json(),
                "done": step.done,
                "rewards": step.rewards,
            },
        }));
    }

    if !environment.done() {
        return Err(HuRlError::new(
            "scalar trace did not reach the terminal state",
        ));
    }
    let rewards = environment
        .terminal_rewards()
        .map_err(|_| HuRlError::new("terminal scoring failed"))?;
    let boards = environment
        .boards()
        .iter()
        .map(sorted_board_json)
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": HU_RL_SCALAR_TRACE_RESULT_SCHEMA,
        "artifact_role": HU_RL_SCALAR_TRACE_ARTIFACT_ROLE,
        "policy_input_eligible": false,
        "replay_eligible": false,
        "training_eligible": false,
        "contains_cross_actor_private_information": true,
        "decisions": decisions,
        "terminal": {
            "boards": boards,
            "rewards": rewards,
        },
    }))
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

fn trace_stage_error(stage: &str, ordinal: usize) -> HuRlError {
    HuRlError::new(format!("scalar trace {stage} failed at decision {ordinal}"))
}
