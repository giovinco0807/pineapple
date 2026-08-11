use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_rl_engine::{
    evaluate_scalar_trace_request, run_scalar_trace_json, ActionKey, Card, HuRlError,
    HU_RL_ACTOR_VIEW_SCHEMA, HU_RL_SCALAR_TRACE_ARTIFACT_ROLE, HU_RL_SCALAR_TRACE_REQUEST_SCHEMA,
    HU_RL_SCALAR_TRACE_RESULT_SCHEMA, MAX_SCALAR_TRACE_REQUEST_BYTES,
};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeSet,
    io::Write,
    process::{Command, Stdio},
};

fn request(indices: Vec<Value>) -> Value {
    json!({
        "schema": HU_RL_SCALAR_TRACE_REQUEST_SCHEMA,
        "explicit_deck": ALL_CARDS.to_vec(),
        "selected_indices": indices,
    })
}

fn zero_index_request() -> Value {
    request(vec![json!(0); 10])
}

fn keys(object: &Map<String, Value>) -> BTreeSet<&str> {
    object.keys().map(String::as_str).collect()
}

#[test]
fn trace_result_is_exactly_classified_as_privileged_audit_only() {
    let result = evaluate_scalar_trace_request(zero_index_request()).unwrap();
    let root = result.as_object().unwrap();
    assert_eq!(
        keys(root),
        BTreeSet::from([
            "artifact_role",
            "contains_cross_actor_private_information",
            "decisions",
            "policy_input_eligible",
            "replay_eligible",
            "schema",
            "terminal",
            "training_eligible",
        ])
    );
    assert_eq!(
        root["schema"],
        Value::String(HU_RL_SCALAR_TRACE_RESULT_SCHEMA.to_owned())
    );
    assert_eq!(
        root["artifact_role"],
        json!(HU_RL_SCALAR_TRACE_ARTIFACT_ROLE)
    );
    assert_eq!(root["policy_input_eligible"], json!(false));
    assert_eq!(root["replay_eligible"], json!(false));
    assert_eq!(root["training_eligible"], json!(false));
    assert_eq!(
        root["contains_cross_actor_private_information"],
        json!(true)
    );

    let decisions = root["decisions"].as_array().unwrap();
    assert_eq!(decisions.len(), 10);
    for (ordinal, decision) in decisions.iter().enumerate() {
        let decision = decision.as_object().unwrap();
        assert_eq!(
            keys(decision),
            BTreeSet::from([
                "actor",
                "actor_view",
                "actor_view_digest",
                "legal_action_mapping",
                "ordinal",
                "selected_action_key",
                "selected_index",
                "step",
                "street",
            ])
        );
        assert_eq!(decision["ordinal"], json!(ordinal));
        assert_eq!(decision["actor"], json!(ordinal % 2));
        assert_eq!(decision["street"], json!(format!("T{}", ordinal / 2)));
        assert_eq!(decision["selected_index"], json!(0));
        assert_eq!(
            decision["actor_view"]["schema"],
            json!(HU_RL_ACTOR_VIEW_SCHEMA)
        );
        assert_eq!(decision["actor_view_digest"].as_str().unwrap().len(), 64);
        if ordinal == 0 {
            assert_eq!(
                decision["actor_view_digest"],
                json!("9b151c2c92bade6151ac6c63c337c603ff80cdc678dc61d2e783f0c586c79ca7")
            );
        }

        let mapping = decision["legal_action_mapping"].as_object().unwrap();
        assert_eq!(
            keys(mapping),
            BTreeSet::from(["action_count", "action_order_digest", "action_set_digest"])
        );
        assert!(mapping["action_count"].as_u64().unwrap() > 0);
        assert_eq!(mapping["action_set_digest"].as_str().unwrap().len(), 64);
        assert_eq!(mapping["action_order_digest"].as_str().unwrap().len(), 64);
        assert_eq!(
            decision["selected_action_key"],
            decision["actor_view"]["legal_action_mapping"]["action_keys"][0]
        );

        let step = decision["step"].as_object().unwrap();
        assert_eq!(
            keys(step),
            BTreeSet::from(["done", "public_event", "rewards"])
        );
        assert_eq!(step["done"], json!(ordinal == 9));
        assert_eq!(
            step["rewards"],
            if ordinal == 9 {
                json!([-1.0, 1.0])
            } else {
                json!([0.0, 0.0])
            }
        );
        let public_event = step["public_event"].as_object().unwrap();
        assert_eq!(
            keys(public_event),
            BTreeSet::from([
                "acting_seat",
                "bottom_placement_mask",
                "discard_count",
                "middle_placement_mask",
                "schema",
                "street",
                "top_placement_mask",
            ])
        );
    }

    let terminal = root["terminal"].as_object().unwrap();
    assert_eq!(keys(terminal), BTreeSet::from(["boards", "rewards"]));
    assert_eq!(terminal["rewards"], json!([-1.0, 1.0]));
    let boards = terminal["boards"].as_array().unwrap();
    assert_eq!(boards.len(), 2);
    for board in boards {
        let board = board.as_object().unwrap();
        assert_eq!(keys(board), BTreeSet::from(["bottom", "middle", "top"]));
        for row in ["top", "middle", "bottom"] {
            let indices = board[row]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().parse::<Card>().unwrap().index())
                .collect::<Vec<_>>();
            assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    let forbidden = BTreeSet::from([
        "deck",
        "deck_tail",
        "discard_mask",
        "explicit_deck",
        "opponent_private_discard",
        "opponent_private_discards",
        "remaining_deck",
        "world_state",
    ]);
    assert!(all_object_keys(&result).is_disjoint(&forbidden));

    // The aggregate deliberately contains each actor's own observations and
    // chosen semantic actions.  Across both seats that is enough to recover
    // both private discard sets, so this result must never be mistaken for an
    // actor-scoped observation or ordinary replay shard.
    let mut reconstructed = [BTreeSet::new(), BTreeSet::new()];
    for decision in decisions {
        let actor = decision["actor"].as_u64().unwrap() as usize;
        for card in decision["actor_view"]["observation"]["hero_private_discards"]
            .as_array()
            .unwrap()
        {
            reconstructed[actor].insert(card.as_str().unwrap().to_owned());
        }
        let key = ActionKey::from_token(decision["selected_action_key"].as_str().unwrap()).unwrap();
        for card in key.cards("discards").unwrap() {
            reconstructed[actor].insert(card.as_str().to_owned());
        }
    }
    assert_eq!(reconstructed[0].len(), 4);
    assert_eq!(reconstructed[1].len(), 4);
}

#[test]
fn request_schema_unknown_types_lengths_and_ranges_fail_closed() {
    let mut unknown = zero_index_request();
    unknown["LEAK_UNKNOWN_FIELD"] = json!("LEAK_UNKNOWN_VALUE");
    assert_sanitized_error(
        evaluate_scalar_trace_request(unknown).unwrap_err(),
        &["LEAK_UNKNOWN_FIELD", "LEAK_UNKNOWN_VALUE"],
    );

    let mut wrong_schema = zero_index_request();
    wrong_schema["schema"] = json!("LEAK_SCHEMA");
    assert_sanitized_error(
        evaluate_scalar_trace_request(wrong_schema).unwrap_err(),
        &["LEAK_SCHEMA"],
    );

    for invalid in [json!(true), json!(-1), json!(0.5), json!("LEAK_INDEX")] {
        let mut indices = vec![json!(0); 10];
        indices[0] = invalid;
        assert_sanitized_error(
            evaluate_scalar_trace_request(request(indices)).unwrap_err(),
            &["LEAK_INDEX"],
        );
    }

    assert!(evaluate_scalar_trace_request(request(vec![json!(0); 9])).is_err());
    let mut out_of_range = vec![json!(0); 10];
    out_of_range[0] = json!(232);
    let error = evaluate_scalar_trace_request(request(out_of_range)).unwrap_err();
    assert!(error.message().contains("outside the legal range"));
    assert!(!error.message().contains("232"));

    let mut bad_deck = zero_index_request();
    bad_deck["explicit_deck"][0] = json!("LEAK_CARD");
    assert_sanitized_error(
        evaluate_scalar_trace_request(bad_deck).unwrap_err(),
        &["LEAK_CARD"],
    );

    let mut duplicate_deck = zero_index_request();
    duplicate_deck["explicit_deck"][51] = duplicate_deck["explicit_deck"][0].clone();
    let error = evaluate_scalar_trace_request(duplicate_deck).unwrap_err();
    assert!(error
        .message()
        .contains("not a valid complete regular deck"));
    assert!(!error.message().contains("2h"));

    let oversized = vec![b' '; MAX_SCALAR_TRACE_REQUEST_BYTES + 1];
    assert!(run_scalar_trace_json(&oversized)
        .unwrap_err()
        .message()
        .contains("byte limit"));
}

#[test]
fn duplicate_and_concatenated_json_requests_are_rejected() {
    let request = serde_json::to_string(&zero_index_request()).unwrap();
    let duplicate = request.replacen(
        &format!("\"schema\":\"{HU_RL_SCALAR_TRACE_REQUEST_SCHEMA}\""),
        &format!(
            "\"schema\":\"{HU_RL_SCALAR_TRACE_REQUEST_SCHEMA}\",\"schema\":\"{HU_RL_SCALAR_TRACE_REQUEST_SCHEMA}\""
        ),
        1,
    );
    assert!(run_scalar_trace_json(duplicate.as_bytes()).is_err());
    assert!(run_scalar_trace_json(format!("{request}{request}").as_bytes()).is_err());
}

#[test]
fn cli_emits_one_result_and_sanitizes_failure_stderr() {
    let input = serde_json::to_vec(&zero_index_request()).unwrap();
    let success = run_cli(&input);
    assert!(success.status.success());
    assert!(success.stderr.is_empty());
    let cli_result: Value = serde_json::from_slice(&success.stdout).unwrap();
    assert_eq!(
        cli_result,
        evaluate_scalar_trace_request(zero_index_request()).unwrap()
    );

    let mut invalid = zero_index_request();
    invalid["LEAK_CLI_FIELD"] = json!("LEAK_CLI_VALUE");
    let failure = run_cli(&serde_json::to_vec(&invalid).unwrap());
    assert!(!failure.status.success());
    assert!(failure.stdout.is_empty());
    let stderr = String::from_utf8(failure.stderr).unwrap();
    assert!(!stderr.contains("LEAK_CLI_FIELD"));
    assert!(!stderr.contains("LEAK_CLI_VALUE"));
    let payload: Value = serde_json::from_str(stderr.trim()).unwrap();
    assert_eq!(payload["status"], json!("error"));
}

fn run_cli(input: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hu_rl_scalar_trace"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn assert_sanitized_error(error: HuRlError, forbidden: &[&str]) {
    for value in forbidden {
        assert!(!error.message().contains(value));
    }
}

fn all_object_keys(value: &Value) -> BTreeSet<&str> {
    fn visit<'a>(value: &'a Value, output: &mut BTreeSet<&'a str>) {
        match value {
            Value::Object(object) => {
                output.extend(object.keys().map(String::as_str));
                for nested in object.values() {
                    visit(nested, output);
                }
            }
            Value::Array(values) => {
                for nested in values {
                    visit(nested, output);
                }
            }
            _ => {}
        }
    }
    let mut output = BTreeSet::new();
    visit(value, &mut output);
    output
}
