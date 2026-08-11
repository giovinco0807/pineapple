//! Exact T3 evaluation over a caller-declared finite chance support.
//!
//! This is an oracle for reduced supports and regression fixtures.  It does
//! not claim to enumerate the full 52-card T3 game.  Every downstream policy
//! decision is made from a freshly constructed [`ActorObservation`] and the
//! selector has no API through which outer-world truth can enter.

use crate::action::{generate_turn_actions, Action};
use crate::action_key::{
    action_key, canonical_argmax_index, canonical_descending_indices, ActionKey,
};
use crate::cards::{validate_cards, Card};
use crate::counter_rng::sha256_hex_json;
use crate::infoset::{ActOrder, ActorObservation, Seat, Street};
use crate::scoring::terminal_score;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;

pub const T3_EXPLICIT_SUPPORT_SCHEMA: &str = "hu_turn3_exact_explicit_support_v1";
pub const CANONICAL_SELECTOR_MODE: &str = "canonical_min_action_key_v1";

/// One weighted world in a declared finite T3 chance support.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplicitSupportWorld {
    pub opponent_private_discards: Vec<Card>,
    pub future_cards: Vec<Card>,
    pub weight: f64,
    pub world_id: String,
}

/// Deliberately narrow exact-support configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplicitSupportConfig {
    pub worlds: Vec<ExplicitSupportWorld>,
    pub continuation_policy_id: String,
    pub selector_mode: String,
}

#[derive(Clone, Debug)]
struct CanonicalWorld {
    opponent_private_discards: Vec<Card>,
    future_cards: Vec<Card>,
    weight: f64,
    world_id: String,
}

impl CanonicalWorld {
    fn draw(&self, count: usize, offset: usize) -> Result<&[Card], String> {
        let end = offset
            .checked_add(count)
            .ok_or_else(|| "explicit T3 world draw is out of bounds".to_owned())?;
        self.future_cards
            .get(offset..end)
            .ok_or_else(|| "explicit T3 world draw is out of bounds".to_owned())
    }

    fn digest_payload(&self) -> Value {
        json!({
            "world_id": self.world_id,
            "weight": self.weight,
            "opponent_private_discards": self.opponent_private_discards,
            "future_cards": self.future_cards,
        })
    }
}

/// Evaluate every legal root action exactly over a declared finite support.
///
/// Weights may be any finite, non-negative values with a positive total.  The
/// reported action values are weighted means (`sum(w*x) / sum(w)`), so scaling
/// every weight by a common factor cannot change the policy or its values.
pub fn evaluate_t3_explicit_support(
    observation: &ActorObservation,
    config: &ExplicitSupportConfig,
) -> Result<Value, String> {
    require_t3_observation(observation)?;
    if config.selector_mode != CANONICAL_SELECTOR_MODE {
        return Err(format!(
            "unsupported explicit-support selector_mode: {:?}",
            config.selector_mode
        ));
    }
    if config.continuation_policy_id.is_empty() {
        return Err("continuation_policy_id must not be empty".to_owned());
    }

    let (worlds, weight_sum) = validate_and_canonicalize_worlds(observation, &config.worlds)?;
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T3 explicit-support root has no legal actions".to_owned());
    }
    let fl_ev_14 = observation
        .scoring
        .fl_ev
        .get(&14)
        .copied()
        .ok_or_else(|| "T3 explicit support requires a 14-card Fantasyland EV".to_owned())?;

    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let mut weighted_sum = 0.0;
        for world in &worlds {
            let score = match observation.to_act_order {
                ActOrder::First => rollout_t3_first(observation, action, world, fl_ev_14)?,
                ActOrder::Second => rollout_t3_second(observation, action, world, fl_ev_14)?,
            };
            weighted_sum += world.weight * score;
        }
        values.push(weighted_sum / weight_sum);
    }

    let selected = canonical_argmax_index(&values, &actions)?;
    let ranked = canonical_descending_indices(&values, &actions)?;
    let best_score = values[selected];
    let rows = ranked
        .iter()
        .enumerate()
        .map(|(sorted_index, &original_index)| {
            action_row(
                &actions[original_index],
                original_index,
                sorted_index,
                values[original_index],
                best_score,
                worlds.len(),
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let support_payload = worlds
        .iter()
        .map(CanonicalWorld::digest_payload)
        .collect::<Vec<_>>();
    // Match the Python M2 explicit-support policy fingerprint.  Its selectors
    // are supplied at that API boundary; this narrower Rust boundary fixes the
    // same information-set-safe selector via `selector_mode`.
    let policy_fingerprint = sha256_hex_json(&json!({
        "id": config.continuation_policy_id,
        "selectors": "caller_supplied_infoset_only",
    }));

    Ok(json!({
        "schema": T3_EXPLICIT_SUPPORT_SCHEMA,
        "mode": "exact_over_declared_finite_support",
        "full_52_card_tree_claimed": false,
        "observation_fingerprint": observation.fingerprint(),
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "continuation_policy_id": config.continuation_policy_id,
        "continuation_policy_fingerprint": policy_fingerprint,
        "selector_mode": config.selector_mode,
        "support_count": worlds.len(),
        "support_weight_sum": weight_sum,
        "support_digest": sha256_hex_json(&Value::Array(support_payload)),
        "legal_action_count": actions.len(),
        "selected_action_original_index": selected,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "best_score": best_score,
        "actions": rows,
        "teacher_notes": {
            "exactness": "exact_only_over_declared_support_and_fixed_continuation",
            "strategy_fusion_guard": "selectors_receive_only_ActorObservation",
            "weight_semantics": "weighted_mean_divided_by_support_weight_sum",
        },
    }))
}

fn require_t3_observation(observation: &ActorObservation) -> Result<(), String> {
    observation.validate()?;
    if observation.street != Street::T3 {
        return Err("T3 explicit support requires a T3 observation".to_owned());
    }
    let expected_opponent = match observation.to_act_order {
        ActOrder::First => 9,
        ActOrder::Second => 11,
    };
    if observation.hero_board.card_count() != 9
        || observation.opponent_public_board.card_count() != expected_opponent
    {
        return Err("invalid live T3 geometry for the requested action order".to_owned());
    }
    Ok(())
}

fn validate_and_canonicalize_worlds(
    observation: &ActorObservation,
    raw_worlds: &[ExplicitSupportWorld],
) -> Result<(Vec<CanonicalWorld>, f64), String> {
    if raw_worlds.is_empty() {
        return Err("explicit T3 support must not be empty".to_owned());
    }
    let future_count = match observation.to_act_order {
        ActOrder::First => 9,
        ActOrder::Second => 6,
    };
    let visible_mask = observation
        .known_unavailable_cards()
        .into_iter()
        .fold(0_u64, |mask, card| mask | card.bit());
    let mut world_ids = HashSet::with_capacity(raw_worlds.len());
    let mut world_contents = HashSet::with_capacity(raw_worlds.len());
    let mut worlds = Vec::with_capacity(raw_worlds.len());
    let mut weight_sum = 0.0;

    for raw in raw_worlds {
        if raw.world_id.is_empty() || !world_ids.insert(raw.world_id.clone()) {
            return Err("explicit T3 world IDs must be non-empty and unique".to_owned());
        }
        if !raw.weight.is_finite() || raw.weight < 0.0 {
            return Err("explicit T3 world weights must be finite and non-negative".to_owned());
        }
        if raw.opponent_private_discards.len() != observation.opponent_discard_count() {
            return Err("explicit T3 opponent discard count is inconsistent".to_owned());
        }
        if raw.future_cards.len() != future_count {
            return Err(
                "explicit T3 world must contain exactly the future cards consumed by this root"
                    .to_owned(),
            );
        }

        let mut opponent_private_discards = raw.opponent_private_discards.clone();
        opponent_private_discards.sort_unstable_by_key(|card| card.index());
        let mut future_cards = raw.future_cards.clone();
        for chunk in future_cards.chunks_mut(3) {
            chunk.sort_unstable_by_key(|card| card.index());
        }
        let hidden = opponent_private_discards
            .iter()
            .chain(&future_cards)
            .copied()
            .collect::<Vec<_>>();
        validate_cards(&hidden)?;
        if hidden.iter().any(|card| visible_mask & card.bit() != 0) {
            return Err("explicit T3 world overlaps actor-visible cards".to_owned());
        }
        if !world_contents.insert((opponent_private_discards.clone(), future_cards.clone())) {
            return Err("explicit T3 support contains a duplicate world".to_owned());
        }
        weight_sum += raw.weight;
        worlds.push(CanonicalWorld {
            opponent_private_discards,
            future_cards,
            weight: raw.weight,
            world_id: raw.world_id.clone(),
        });
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err("explicit T3 support weight sum must be finite and positive".to_owned());
    }
    Ok((worlds, weight_sum))
}

/// Fixed information-set policy.  The only input is the acting player's
/// observation; no `CanonicalWorld` or realized outer deck tail is accepted.
fn canonical_min_action(observation: &ActorObservation) -> Result<Action, String> {
    observation.validate()?;
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    actions
        .into_iter()
        .map(|action| Ok((action_key(&action)?, action)))
        .collect::<Result<Vec<(ActionKey, Action)>, String>>()?
        .into_iter()
        .min_by_key(|(key, _)| *key)
        .map(|(_, action)| action)
        .ok_or_else(|| "explicit-support child observation has no legal actions".to_owned())
}

/// T3 second -> opponent T4 first -> hero T4 second.
fn rollout_t3_second(
    observation: &ActorObservation,
    root_action: &Action,
    world: &CanonicalWorld,
    fl_ev_14: f64,
) -> Result<f64, String> {
    let after_root = root_action.apply(&observation.hero_board)?;
    let opponent_t4_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        world.draw(3, 0)?.to_vec(),
        world.opponent_private_discards.clone(),
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = canonical_min_action(&opponent_t4_observation)?;
    let opponent_final = opponent_t4_action.apply(&observation.opponent_public_board)?;

    let mut hero_private_discards = observation.hero_private_discards.clone();
    hero_private_discards.extend(root_action.discards.iter().copied());
    let hero_t4_observation = ActorObservation::new(
        after_root.clone(),
        opponent_final.clone(),
        world.draw(3, 3)?.to_vec(),
        hero_private_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = canonical_min_action(&hero_t4_observation)?;
    let hero_final = hero_t4_action.apply(&after_root)?;
    terminal_score(&hero_final, Some(&opponent_final), fl_ev_14).map(|(score, _)| score)
}

/// T3 first -> opponent T3 second -> hero T4 first -> opponent T4 second.
fn rollout_t3_first(
    observation: &ActorObservation,
    root_action: &Action,
    world: &CanonicalWorld,
    fl_ev_14: f64,
) -> Result<f64, String> {
    let after_root = root_action.apply(&observation.hero_board)?;
    let opponent_t3_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        world.draw(3, 0)?.to_vec(),
        world.opponent_private_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = canonical_min_action(&opponent_t3_observation)?;
    let opponent_after_t3 = opponent_t3_action.apply(&observation.opponent_public_board)?;

    let mut hero_private_discards = observation.hero_private_discards.clone();
    hero_private_discards.extend(root_action.discards.iter().copied());
    let hero_t4_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t3.clone(),
        world.draw(3, 3)?.to_vec(),
        hero_private_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = canonical_min_action(&hero_t4_observation)?;
    let hero_final = hero_t4_action.apply(&after_root)?;

    let mut opponent_private_discards = world.opponent_private_discards.clone();
    opponent_private_discards.extend(opponent_t3_action.discards.iter().copied());
    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_final.clone(),
        world.draw(3, 6)?.to_vec(),
        opponent_private_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = canonical_min_action(&opponent_t4_observation)?;
    let opponent_final = opponent_t4_action.apply(&opponent_after_t3)?;
    terminal_score(&hero_final, Some(&opponent_final), fl_ev_14).map(|(score, _)| score)
}

fn action_row(
    action: &Action,
    original_index: usize,
    sorted_index: usize,
    score: f64,
    best_score: f64,
    future_count: usize,
) -> Result<Value, String> {
    Ok(json!({
        "original_index": original_index,
        "action_key": action_key(action)?.to_token(),
        "placements": action.placements,
        "discards": action.discards,
        "sorted_index": sorted_index,
        "score": score,
        "joint_ev": score,
        "future_count": future_count,
        "regret_vs_best": best_score - score,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY_ID: &str = "unit_test_canonical_first_v1";

    fn decode_fixture(
        observation: &str,
        worlds: &str,
    ) -> (ActorObservation, ExplicitSupportConfig) {
        let observation = serde_json::from_str(observation).unwrap();
        let worlds = serde_json::from_str(worlds).unwrap();
        (
            observation,
            ExplicitSupportConfig {
                worlds,
                continuation_policy_id: POLICY_ID.to_owned(),
                selector_mode: CANONICAL_SELECTOR_MODE.to_owned(),
            },
        )
    }

    #[test]
    fn python_first_seat_nonzero_golden_matches_all_action_values() {
        let (observation, config) = decode_fixture(
            r#"{"schema":"regular_ofc_actor_observation_v1","hero_board":{"top":[],"middle":["2c","Td","7c","2s"],"bottom":["9h","Js","Ad","Jc","Kc"]},"opponent_public_board":{"top":[],"middle":["Qc","Tc","2h","9s"],"bottom":["2d","Th","3h","4s","5c"]},"dealt_cards":["Ts","8h","4h"],"hero_private_discards":["9c","3s"],"seat":"first","street":"T3","to_act_order":"first","scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14},"hero_in_fantasyland":false,"opponent_in_fantasyland":false,"opponent_discard_count":2}"#,
            r#"[{"opponent_private_discards":["3d","4d"],"future_cards":["Kh","8d","Ac","5h","7h","Qh","Ah","4c","5s"],"weight":0.25,"world_id":"first-world-0"},{"opponent_private_discards":["5h","6s"],"future_cards":["Jh","8d","Jd","5s","8s","Ks","7h","5d","3c"],"weight":0.75,"world_id":"first-world-1"}]"#,
        );
        let result = evaluate_t3_explicit_support(&observation, &config).unwrap();
        assert_eq!(
            result["support_digest"],
            "8c641aabb1227956e2ea528d87bb2f5419f31710f91cf8f0c32d2c2f72ce7264"
        );
        assert_eq!(
            result["selected_action_key"],
            "rak1:0000000000004:0000000000040:0000000000000:0800000000000"
        );
        assert_eq!(result["best_score"], 6.0);
        let scores = result["actions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["score"].as_f64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(scores, vec![6.0, 6.0, 6.0, 4.5, 4.5, 4.5, 1.5, 0.0, 0.0]);
    }

    #[test]
    fn python_second_seat_nonzero_golden_matches_all_action_values() {
        let (observation, config) = decode_fixture(
            r#"{"schema":"regular_ofc_actor_observation_v1","hero_board":{"top":[],"middle":["Qc","7d","Th","2d"],"bottom":["2c","2s","7h","8c","7s"]},"opponent_public_board":{"top":["4c"],"middle":["4d","3c","3h","5h","9c"],"bottom":["3d","Ah","6s","5c","6h"]},"dealt_cards":["8d","8s","As"],"hero_private_discards":["Js","6d"],"seat":"second","street":"T3","to_act_order":"second","scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14},"hero_in_fantasyland":false,"opponent_in_fantasyland":false,"opponent_discard_count":3}"#,
            r#"[{"opponent_private_discards":["9d","Ad","Ac"],"future_cards":["Jh","9s","Qs","5d","Kd","Jc"],"weight":0.25,"world_id":"second-world-0"},{"opponent_private_discards":["4h","7c","9s"],"future_cards":["8h","Kd","5s","2h","Jh","Jc"],"weight":0.75,"world_id":"second-world-1"}]"#,
        );
        let result = evaluate_t3_explicit_support(&observation, &config).unwrap();
        assert_eq!(
            result["support_digest"],
            "37b3dede5e7f90daef5534d049408a113c57ce16f6b4d28fe7fa778168f5300e"
        );
        assert_eq!(
            result["selected_action_key"],
            "rak1:0000000080000:8000000000000:0000000000000:0200000000000"
        );
        assert_eq!(result["best_score"], -0.5);
        let scores = result["actions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["score"].as_f64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            scores,
            vec![-0.5, -0.5, -2.25, -2.25, -6.0, -6.0, -6.0, -6.0, -6.0]
        );
    }

    #[test]
    fn weights_are_nonnegative_normalized_and_future_chunks_are_canonical() {
        let (observation, mut config) = decode_fixture(
            r#"{"schema":"regular_ofc_actor_observation_v1","hero_board":{"top":[],"middle":["Qc","7d","Th","2d"],"bottom":["2c","2s","7h","8c","7s"]},"opponent_public_board":{"top":["4c"],"middle":["4d","3c","3h","5h","9c"],"bottom":["3d","Ah","6s","5c","6h"]},"dealt_cards":["8d","8s","As"],"hero_private_discards":["Js","6d"],"seat":"second","street":"T3","to_act_order":"second","scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14},"hero_in_fantasyland":false,"opponent_in_fantasyland":false,"opponent_discard_count":3}"#,
            r#"[{"opponent_private_discards":["Ac","Ad","9d"],"future_cards":["Qs","Jh","9s","Jc","5d","Kd"],"weight":1.0,"world_id":"second-world-0"},{"opponent_private_discards":["9s","4h","7c"],"future_cards":["5s","Kd","8h","Jc","Jh","2h"],"weight":3.0,"world_id":"second-world-1"},{"opponent_private_discards":["Td","Qd","Ks"],"future_cards":["Tc","7c","9s","6c","Qh","Kc"],"weight":0.0,"world_id":"zero-weight-world"}]"#,
        );
        let result = evaluate_t3_explicit_support(&observation, &config).unwrap();
        assert_eq!(result["support_weight_sum"], 4.0);
        assert_eq!(result["best_score"], -0.5);

        config.worlds[0].weight = f64::NAN;
        assert!(evaluate_t3_explicit_support(&observation, &config)
            .unwrap_err()
            .contains("finite and non-negative"));
    }

    #[test]
    fn support_validation_fails_closed_on_truth_overlap_duplicates_and_mode() {
        let (observation, mut config) = decode_fixture(
            r#"{"schema":"regular_ofc_actor_observation_v1","hero_board":{"top":[],"middle":["Qc","7d","Th","2d"],"bottom":["2c","2s","7h","8c","7s"]},"opponent_public_board":{"top":["4c"],"middle":["4d","3c","3h","5h","9c"],"bottom":["3d","Ah","6s","5c","6h"]},"dealt_cards":["8d","8s","As"],"hero_private_discards":["Js","6d"],"seat":"second","street":"T3","to_act_order":"second","scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14},"hero_in_fantasyland":false,"opponent_in_fantasyland":false,"opponent_discard_count":3}"#,
            r#"[{"opponent_private_discards":["9d","Ad","Ac"],"future_cards":["Jh","9s","Qs","5d","Kd","Jc"],"weight":1.0,"world_id":"world"}]"#,
        );
        config.selector_mode = "outer_truth_cheat".to_owned();
        assert!(evaluate_t3_explicit_support(&observation, &config)
            .unwrap_err()
            .contains("unsupported explicit-support selector_mode"));
        config.selector_mode = CANONICAL_SELECTOR_MODE.to_owned();
        config.worlds[0].future_cards[0] = observation.dealt_cards[0];
        assert!(evaluate_t3_explicit_support(&observation, &config)
            .unwrap_err()
            .contains("overlaps actor-visible cards"));
        config.worlds[0].future_cards[0] = config.worlds[0].future_cards[1];
        assert!(evaluate_t3_explicit_support(&observation, &config)
            .unwrap_err()
            .contains("duplicate card"));
    }
}
