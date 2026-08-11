from __future__ import annotations

import json

import pytest

from ofc_regular.action_space import generate_actions
from ofc_regular.extract_hu_turn0_fired_replay_targets import extract_fired_targets
from ofc_regular.policy import action_to_json
from ofc_regular.replay_hu_turn0_fired_targets import parse_args, replay_target
from ofc_regular.state import Board


def _event() -> dict:
    board = Board.from_rows()
    dealt = ("As", "Kh", "Qd", "Jc", "Ts")
    actions = generate_actions(board, dealt)
    return {
        "event_id": "1:first",
        "config_id": "cfg",
        "seed": 1,
        "seat": "first",
        "override_fired": True,
        "fallback_action_index": 0,
        "candidate_action_index": 1,
        "final_action_index": 1,
        "hu_turn0_predicted_margin": 3.0,
        "realized_delta": 2.0,
        "decision": {
            "replay_ready": True,
            "visibility_model": "hidden_discard",
            "discard_visibility": "own_private_only",
            "hero_board": {"top": [], "middle": [], "bottom": []},
            "opponent_board": {"top": [], "middle": [], "bottom": []},
            "cards_to_place": list(dealt),
            "dead_cards": [],
            "visible_dead_cards": [],
            "baseline_action": action_to_json(board, actions[0]),
            "hu_turn0_action": action_to_json(board, actions[1]),
        },
    }


def test_extract_fired_targets_preserves_replay_and_action_mapping(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(_event()) + "\n", encoding="utf-8")

    rows, summary = extract_fired_targets(
        [(path, "cfg")], candidate_model="model.pkl", candidate_topk=60
    )

    assert len(rows) == 1
    assert rows[0]["baseline_action_index"] == 0
    assert rows[0]["candidate_action_index"] == 1
    assert rows[0]["replay_ready"] is True
    assert summary["replay_ready"] == 1
    assert summary["missing_action_mapping"] == 0


def test_replay_target_uses_only_baseline_and_candidate_with_common_futures() -> None:
    event = _event()
    target = {
        "schema": "hu_turn0_fired_replay_target_v1",
        "target_id": "a" * 64,
        "hand_seed": 1,
        "seat": "first",
        "hero_board": event["decision"]["hero_board"],
        "opponent_board": event["decision"]["opponent_board"],
        "cards_to_place": event["decision"]["cards_to_place"],
        "baseline_action_index": 0,
        "candidate_action_index": 1,
        "baseline_action": event["decision"]["baseline_action"],
        "candidate_action": event["decision"]["hu_turn0_action"],
    }

    def fake_build(**kwargs):
        assert kwargs["action_indices"] == (0, 1)
        assert kwargs["future_samples"] == 128
        return {
            "actions": [
                {
                    "original_index": 1,
                    "ev": 2.0,
                    "delta_vs_baseline": 1.0,
                    "delta_se_vs_baseline": 0.25,
                    "delta_z_vs_baseline": 4.0,
                },
                {
                    "original_index": 0,
                    "ev": 1.0,
                    "delta_vs_baseline": 0.0,
                    "delta_se_vs_baseline": 0.0,
                    "delta_z_vs_baseline": 0.0,
                },
            ],
            "baseline_action_index": 0,
            "future_seed": 123,
            "common_random_future_digest": "digest",
            "common_random_futures_verified": True,
        }

    row = replay_target(
        target,
        policies=(object(), object()),
        future_samples=128,
        replay_seed=7,
        build_sample_fn=fake_build,
    )

    assert row["candidate_delta_vs_baseline"] == 1.0
    assert row["candidate_delta_lcb196"] == pytest.approx(0.51)
    assert row["safe_override_label"] == "positive"
    assert row["action_mapping_verified"] is True


def test_replay_target_rejects_action_mapping_drift() -> None:
    event = _event()
    target = {
        "target_id": "b" * 64,
        "hand_seed": 1,
        "seat": "first",
        "hero_board": event["decision"]["hero_board"],
        "opponent_board": event["decision"]["opponent_board"],
        "cards_to_place": event["decision"]["cards_to_place"],
        "baseline_action_index": 0,
        "candidate_action_index": 1,
        "baseline_action": event["decision"]["hu_turn0_action"],
        "candidate_action": event["decision"]["hu_turn0_action"],
    }

    with pytest.raises(ValueError, match="baseline action mapping changed"):
        replay_target(
            target,
            policies=(object(), object()),
            future_samples=1,
            replay_seed=7,
            build_sample_fn=lambda **kwargs: {},
        )


def test_replay_cli_accepts_generic_pilot_shard_arguments() -> None:
    args = parse_args(
        [
            "--samples",
            "5",
            "--skip-records",
            "10",
            "--future-samples",
            "32",
            "--seed",
            "123",
            "--max-actions",
            "0",
            "--profile",
            "stage18_p1",
            "--opponent-profile",
            "stage18_p1",
            "--source-bucket",
            "runtime_fired_pair_replay",
            "--opening-lookahead-samples",
            "1",
            "--seats",
            "first",
            "second",
            "--output",
            "out.jsonl",
            "--summary-output",
            "summary.json",
        ]
    )

    assert args.input.as_posix() == "inputs/targets.jsonl"
    assert args.samples == 5
    assert args.skip_records == 10
    assert args.seed == 123
