use ofc_hu_m3_engine::cards::{Card, ALL_CARDS};
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{evaluate_t3, evaluate_t4, T3Config, T4Config};
use ofc_hu_m3_engine::state::Board;
use serde_json::json;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn constrained_board(cards: &[Card]) -> Board {
    let top_count = cards.len().min(3);
    let middle_count = (cards.len() - top_count).min(5);
    Board::new(
        cards[..top_count].to_vec(),
        cards[top_count..top_count + middle_count].to_vec(),
        cards[top_count + middle_count..].to_vec(),
    )
    .unwrap()
}

fn t3_observation(order: ActOrder) -> ActorObservation {
    let opponent_count = if order == ActOrder::First { 9 } else { 11 };
    let mut cursor = 0;
    let hero = constrained_board(&ALL_CARDS[cursor..cursor + 9]);
    cursor += 9;
    let opponent = constrained_board(&ALL_CARDS[cursor..cursor + opponent_count]);
    cursor += opponent_count;
    let dealt = ALL_CARDS[cursor..cursor + 3].to_vec();
    cursor += 3;
    let discards = ALL_CARDS[cursor..cursor + 2].to_vec();
    ActorObservation::new(
        hero,
        opponent,
        dealt,
        discards,
        if order == ActOrder::First {
            Seat::First
        } else {
            Seat::Second
        },
        Street::T3,
        order,
        ScoringContext::default(),
    )
    .unwrap()
}

fn t4_first_observation() -> ActorObservation {
    serde_json::from_value(json!({
        "schema": "regular_ofc_actor_observation_v1",
        "hero_board": {
            "top": ["Kh"],
            "middle": ["7d", "Jh", "Ac", "6s", "3c"],
            "bottom": ["Qs", "Qc", "Ks", "As", "Js"]
        },
        "opponent_public_board": {
            "top": ["Ad"],
            "middle": ["3s", "8c", "4c", "7c", "6d"],
            "bottom": ["Td", "3d", "9d", "8d", "6h"]
        },
        "dealt_cards": ["4h", "2c", "Ah"],
        "hero_private_discards": ["Tc", "7h", "Th"],
        "seat": "first",
        "street": "T4",
        "to_act_order": "first",
        "scoring": {
            "schema": "regular_ofc_scoring_context_v1",
            "fl_ev": {"14": 10.227020614683454},
            "middle_trips_royalty": 2,
            "hu_line_points": true,
            "scoop_bonus": 3,
            "foul_enabled": true,
            "fantasyland_cards": 14
        }
    }))
    .unwrap()
}

#[test]
fn t4_mask_cache_preserves_full_t3_result_and_expected_child_cardinality() {
    for (order, expected_child_count) in [(ActOrder::First, 108_u64), (ActOrder::Second, 24_u64)] {
        let observation = t3_observation(order);
        let suffix = if order == ActOrder::First {
            "first"
        } else {
            "second"
        };
        let cached_config = T3Config {
            candidate_samples: 2,
            evaluation_samples: 2,
            downstream_t3_samples: 1,
            downstream_t4_samples: 0,
            seed: 808,
            candidate_seed: 809,
            evaluation_seed: 810,
            run_id: format!("candidate01-cache-audit-{suffix}"),
            use_t4_action_cache: true,
            ..Default::default()
        };
        let mut uncached_config = cached_config.clone();
        uncached_config.use_t4_action_cache = false;

        let cached = evaluate_t3(&observation, &cached_config).unwrap();
        let uncached = evaluate_t3(&observation, &uncached_config).unwrap();

        assert_eq!(cached, uncached);
        assert_eq!(
            cached["child_information_set_count"].as_u64(),
            Some(expected_child_count)
        );
    }
}

#[test]
fn public_t4_rejects_invalid_opponent_board_before_trusted_path() {
    let mut observation = t4_first_observation();
    let moved = observation
        .opponent_public_board
        .middle
        .drain(0..3)
        .collect::<Vec<_>>();
    observation.opponent_public_board.top.extend(moved);
    assert!(observation.validate().is_err());

    let config = T4Config {
        candidate_samples: 1,
        evaluation_samples: 1,
        seed: 42,
        candidate_seed: 43,
        evaluation_seed: 44,
        run_id: "candidate01-public-boundary".to_owned(),
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| evaluate_t4(&observation, &config)));
    assert!(
        outcome.is_ok(),
        "public evaluate_t4 must return Err, not panic"
    );
    assert!(outcome.unwrap().is_err());
}
