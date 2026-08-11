from types import SimpleNamespace
from copy import deepcopy

from ofc_regular.action_key import resolve_action_index
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.extract_hu_turn1_refinement_targets import output_record, score_records, select_targets
from ofc_regular.merge_hu_turn1_refinement_labels import merge_records
from ofc_regular.relabel_hu_turn1_refinement_targets import (
    board_from_json,
    actor_observation_for_record,
    compare_source_to_relabel,
    dealt_from_record,
    private_discards_from_record,
    player_from_record,
    read_jsonl,
    remaining_cards_for_record,
    relabel_record,
    skipped_target_record,
    source_action_indices,
    summarize,
)


def make_record(sample_id: int, *, seat: str = "first", gap: float = 1.0) -> dict:
    best_score = 10.0 + gap
    opponent_board = {
        "top": ["Qs"],
        "middle": ["Jd"],
        "bottom": ["3c", "3d", "Ts"],
    }
    opponent_private_discards: list[str] = []
    if seat == "second":
        opponent_board["middle"].append("7d")
        opponent_board["bottom"].append("8d")
        opponent_private_discards = ["8h"]
    visible_dead_cards = [
        *opponent_board["top"],
        *opponent_board["middle"],
        *opponent_board["bottom"],
    ]
    return {
        "schema": "hu_turn1_stage1_pilot_v1",
        "sample_id": sample_id,
        "hand_seed": 100 + sample_id,
        "player": 0 if seat == "first" else 1,
        "seat": seat,
        "board": {
            "top": ["Ah"],
            "middle": ["Kd"],
            "bottom": ["2c", "2d", "9s"],
        },
        "opponent_board": opponent_board,
        "dealt": ["4h", "5h", "6h"],
        "dead_cards": list(visible_dead_cards),
        "visible_dead_cards": list(visible_dead_cards),
        "true_dead_cards": list(opponent_private_discards),
        "hero_private_discards": [],
        "opponent_private_discards": opponent_private_discards,
        "action_count": 2,
        "best_action": 0,
        "score_gap": gap,
        "actions": [
            {
                "placements": [["4h", "top"], ["5h", "middle"]],
                "discards": ["6h"],
                "score": best_score,
                "ev": best_score,
                "se": 0.0,
            },
            {
                "placements": [["4h", "middle"], ["5h", "bottom"]],
                "discards": ["6h"],
                "score": 10.0,
                "ev": 10.0,
                "se": 0.0,
            },
        ],
    }


def make_high_se_record(sample_id: int, *, seat: str = "first") -> dict:
    record = make_record(sample_id, seat=seat, gap=0.5)
    record["actions"][0]["se"] = 3.5
    record["actions"][1]["se"] = 2.0
    return record


def test_extract_selects_balanced_targets_without_models() -> None:
    records = [
        make_record(0, seat="first", gap=5.0),
        make_record(1, seat="second", gap=4.0),
        make_record(2, seat="first", gap=0.0),
        make_record(3, seat="second", gap=0.0),
    ]
    scored = score_records(records, self_model_path=None, hu_model_path=None)
    targets = select_targets(
        scored,
        high_regret_count=0,
        high_se_count=0,
        high_se_threshold=3.0,
        high_gap_count=2,
        low_gap_count=2,
        random_count=0,
        low_gap_threshold=0.01,
        max_targets=4,
        seed=1,
    )

    assert len(targets) == 4
    assert {target.seat for target in targets} == {"first", "second"}
    assert any("high_score_gap" in target.selection_reasons for target in targets)
    assert any("low_score_gap" in target.selection_reasons for target in targets)


def test_extract_selects_high_se_targets_and_outputs_source_bucket() -> None:
    records = [
        make_high_se_record(0, seat="first"),
        make_high_se_record(1, seat="second"),
        make_record(2, seat="first", gap=4.0),
        make_record(3, seat="second", gap=4.0),
    ]
    scored = score_records(records, self_model_path=None, hu_model_path=None)
    targets = select_targets(
        scored,
        high_regret_count=0,
        high_se_count=2,
        high_se_threshold=3.0,
        high_gap_count=0,
        low_gap_count=0,
        random_count=0,
        low_gap_threshold=0.01,
        max_targets=2,
        seed=1,
    )

    assert len(targets) == 2
    assert {target.seat for target in targets} == {"first", "second"}
    assert all("high_action_se" in target.selection_reasons for target in targets)
    row = output_record(targets[0], 0)
    assert row["source_bucket"] == "high_action_se"
    assert row["source_bucket_group"] == "turn1_refinement"
    assert row["selection_max_action_se"] >= 3.0


def test_legacy_relabel_helpers_are_audit_only() -> None:
    record = make_record(0, seat="first")
    board = board_from_json(record["board"])
    opponent = board_from_json(record["opponent_board"])
    private_discards = private_discards_from_record(record)
    remaining = remaining_cards_for_record(
        board=board,
        opponent_board=opponent,
        dealt=record["dealt"],
        private_discards=private_discards,
    )

    assert private_discards == [[], []]
    assert "Ah" not in remaining
    assert "4h" not in remaining
    assert "9s" not in remaining
    assert len(remaining) == 52 - 13

    observation = actor_observation_for_record(record)
    assert observation.hero_private_discards == ()
    assert observation.legacy_dead_cards() == tuple(record["visible_dead_cards"])


def test_relabel_helpers_support_decision_log_schema_without_player_or_dealt() -> None:
    record = {
        **make_record(0, seat="second"),
        "player": None,
        "dealt": None,
        "cards_to_place": ["4h", "5h", "6h"],
    }
    del record["player"]
    del record["dealt"]

    assert player_from_record(record) == 1
    assert dealt_from_record(record) == ("4h", "5h", "6h")
    assert private_discards_from_record(record) == [["8h"], []]


def test_relabel_compare_detects_changed_best_action() -> None:
    source = make_record(0)
    relabeled_actions = [source["actions"][1], source["actions"][0]]
    relabeled_actions[0] = {**relabeled_actions[0], "score": 20.0}
    relabeled_actions[1] = {**relabeled_actions[1], "score": 10.0}

    result = compare_source_to_relabel(source, relabeled_actions)

    assert result["best_action_changed"] is True
    assert result["source_best_new_rank"] == 2
    assert result["source_best_new_regret"] == 10.0


def test_relabel_read_jsonl_supports_skip_and_limit(tmp_path) -> None:
    path = tmp_path / "targets.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"target_id":0}',
                '{"target_id":1}',
                '{"target_id":2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_jsonl(path, skip_targets=1, max_targets=1)

    assert [row["target_id"] for row in rows] == [1]


def test_relabel_skipped_target_record_preserves_replay_metadata() -> None:
    source = {
        **make_record(7, seat="second"),
        "target_id": 99,
        "source_bucket": "high_action_se",
        "source_bucket_group": "turn1_refinement",
        "selection_reasons": ["high_action_se", "random_cover"],
    }

    skipped = skipped_target_record(
        source,
        reason="target_timeout",
        message="target exceeded 120.000s",
        profile="stage9f_p2",
        opponent_profile="stage9f_p2",
        future_samples=32,
        max_actions=0,
        timeout_seconds=120.0,
    )

    assert skipped["schema"] == "hu_turn1_stage1_refinement_relabel_skip_v1"
    assert skipped["target_id"] == 99
    assert skipped["sample_id"] == 7
    assert skipped["seat"] == "second"
    assert skipped["source_bucket"] == "high_action_se"
    assert skipped["selection_reasons"] == ["high_action_se", "random_cover"]
    assert skipped["skip_reason"] == "target_timeout"
    assert skipped["future_samples"] == 32


def test_relabel_source_action_indices_are_extracted_in_order() -> None:
    record = make_record(0)
    record["actions"][0]["action_index"] = 4
    record["actions"][1]["original_index"] = 7
    record["actions"].append({"action_index": 4})
    record["actions"].append({"action_index": -1})
    record["actions"].append({"action_index": "bad"})

    assert source_action_indices(record) == [4, 7]


def test_relabel_source_action_indices_include_runtime_pair_indices() -> None:
    record = make_record(0)
    record["runtime_candidate_action_index"] = 2
    record["runtime_baseline_action_index"] = 8
    record["actions"] = [{"action_index": 2}]

    assert source_action_indices(record) == [2, 8]


def test_relabel_summary_counts_skipped_timeout_targets(tmp_path) -> None:
    args = SimpleNamespace(
        input=tmp_path / "targets.jsonl",
        output=tmp_path / "relabel.jsonl",
        skip_output=tmp_path / "skipped.jsonl",
        profile="stage9f_p2",
        opponent_profile="stage9f_p2",
        future_samples=32,
        max_actions=0,
        candidate_model=None,
        candidate_models=[],
        candidate_topk=0,
        candidate_union_cap=0,
        candidate_union_mode="min_rank",
        target_timeout_seconds=120.0,
        continue_on_target_error=True,
        seed=2026062501,
    )
    good_row = {
        **make_record(0),
        "relabel_compare": {"best_action_changed": True, "source_best_new_regret": 2.5},
        "relabel_seconds": 10.0,
    }
    skipped = [
        {"skip_reason": "target_timeout"},
        {"skip_reason": "target_error"},
    ]

    result = summarize([good_row], [{"choose_action_T2_seconds": 8.0}], skipped, args)

    assert result["target_attempts"] == 3
    assert result["records"] == 1
    assert result["skipped_targets"] == 2
    assert result["timed_out_targets"] == 1
    assert result["failed_targets"] == 1
    assert result["skip_reason_counts"] == {"target_error": 1, "target_timeout": 1}
    assert result["candidate_model_path"] is None
    assert result["candidate_model_paths"] == []
    assert result["candidate_topk"] == 0
    assert result["candidate_union_cap"] == 0
    assert result["candidate_union_mode"] == "min_rank"
    assert result["best_action_changed"] == 1
    assert result["t2_choose_action_seconds_sum"] == 8.0


def test_relabel_record_overwrites_stale_candidate_selector_metadata(monkeypatch) -> None:
    source = {
        **make_record(3),
        "target_id": 3,
        "candidate_topk": 15,
        "candidate_union_cap": 20,
        "candidate_selector": {"topk": 15, "union_cap": 20, "selected_indices": [0]},
        "total_legal_actions": 2,
        "evaluated_action_count": 1,
    }
    replacement_action = {
        **source["actions"][1],
        "score": 13.0,
        "ev": 13.0,
        "action_index": 5,
        "original_index": 5,
    }
    old_action = {
        **source["actions"][0],
        "score": 10.0,
        "ev": 10.0,
        "action_index": 2,
        "original_index": 2,
    }

    def fake_evaluate_turn1_action_subset(**kwargs):
        assert kwargs["candidate_topk"] == 20
        assert kwargs["candidate_union_cap"] == 25
        assert kwargs["candidate_union_mode"] == "max_z_score"
        return {
            "actions": [replacement_action, old_action],
            "actions_truncated": True,
            "total_legal_actions": 27,
            "evaluated_action_count": 22,
            "candidate_selector": {
                "mode": "candidate_model_union_topk",
                "topk": 20,
                "union_cap": 25,
                "selected_indices": [5, 2],
            },
        }

    monkeypatch.setattr(
        "ofc_regular.relabel_hu_turn1_refinement_targets.evaluate_turn1_action_subset",
        fake_evaluate_turn1_action_subset,
    )

    relabeled, _profile = relabel_record(
        source,
        policies=[],
        profile="stage9f_p2",
        opponent_profile="stage9f_p2",
        future_samples=32,
        max_actions=0,
        candidate_model=None,
        candidate_models=[object(), object()],
        candidate_topk=20,
        candidate_union_cap=25,
        candidate_union_mode="max_z_score",
        source_actions_only=False,
        seed=2026062601,
    )

    assert relabeled["candidate_topk"] == 20
    assert relabeled["candidate_union_cap"] == 25
    assert relabeled["candidate_union_mode"] == "max_z_score"
    assert relabeled["source_actions_only"] is False
    assert relabeled["candidate_selector"]["topk"] == 20
    assert relabeled["candidate_selector"]["union_cap"] == 25
    assert relabeled["total_legal_actions"] == 27
    assert relabeled["evaluated_action_count"] == 22
    assert relabeled["relabel_compare"]["best_action_changed"] is True


def test_relabel_record_can_evaluate_source_actions_only(monkeypatch) -> None:
    source = make_record(0)
    source["actions"][0]["action_index"] = 4
    source["actions"][1]["action_index"] = 7
    seen = {}

    def fake_evaluate_turn1_action_subset(**kwargs):
        seen["action_indices"] = kwargs["action_indices"]
        return {
            "actions": [
                {**source["actions"][0], "score": 11.0, "action_index": 4, "original_index": 4},
                {**source["actions"][1], "score": 10.0, "action_index": 7, "original_index": 7},
            ],
            "actions_truncated": True,
            "total_legal_actions": 27,
            "evaluated_action_count": 2,
        }

    monkeypatch.setattr(
        "ofc_regular.relabel_hu_turn1_refinement_targets.evaluate_turn1_action_subset",
        fake_evaluate_turn1_action_subset,
    )

    relabeled, _profile = relabel_record(
        source,
        policies=[],
        profile="stage9f_p2",
        opponent_profile="stage9f_p2",
        future_samples=32,
        max_actions=0,
        candidate_model=None,
        candidate_models=[],
        candidate_topk=0,
        candidate_union_cap=0,
        candidate_union_mode="min_rank",
        source_actions_only=True,
        seed=2026062601,
    )

    legal_actions = generate_turn_actions(
        board_from_json(source["board"]), dealt_from_record(source)
    )
    expected = [
        resolve_action_index(legal_actions, payload=action).index
        for action in source["actions"]
    ]
    assert seen["action_indices"] == expected
    assert relabeled["source_actions_only"] is True
    assert relabeled["source_action_indices"] == expected
    assert relabeled["evaluated_action_count"] == 2


def test_relabel_belief_is_invariant_to_hidden_opponent_discard_identity(
    monkeypatch,
) -> None:
    first = make_record(11, seat="second")
    first["target_id"] = 41
    second = deepcopy(first)
    second["true_dead_cards"] = ["7h"]
    second["opponent_private_discards"] = ["7h"]
    calls = []

    def fake_evaluate_turn1_action_subset(**kwargs):
        calls.append(kwargs)
        assert "remaining_cards" not in kwargs
        assert "private_discards" not in kwargs
        assert kwargs["observation"].hero_private_discards == ()
        return {
            "actions": deepcopy(first["actions"]),
            "actions_truncated": False,
            "total_legal_actions": 27,
            "evaluated_action_count": 2,
            "belief_batch_digest": kwargs["belief_batch"].digest(),
            "belief_schema": kwargs["belief_batch"].to_dict()["belief_schema"],
            "belief_prior": kwargs["belief_batch"].prior,
        }

    monkeypatch.setattr(
        "ofc_regular.relabel_hu_turn1_refinement_targets.evaluate_turn1_action_subset",
        fake_evaluate_turn1_action_subset,
    )

    def relabel(source):
        return relabel_record(
            source,
            policies=[],
            profile="stage9f_p2",
            opponent_profile="stage9f_p2",
            future_samples=4,
            max_actions=0,
            candidate_model=None,
            candidate_models=[],
            candidate_topk=0,
            candidate_union_cap=0,
            candidate_union_mode="min_rank",
            source_actions_only=False,
            seed=2026062601,
        )[0]

    first_result = relabel(first)
    second_result = relabel(second)

    assert calls[0]["observation"] == calls[1]["observation"]
    assert calls[0]["belief_batch"].digest() == calls[1]["belief_batch"].digest()
    assert first_result["belief_batch_digest"] == second_result["belief_batch_digest"]
    assert first_result["actions"] == second_result["actions"]
    assert first_result["teacher_conditioning"] == "actor_observation_belief_v1"
    assert first_result["replay_truth_audit_only"] is True
    assert first_result["replay_truth_used_for_label"] is False
    # Replay truth is retained only for audit and does not enter the root.
    assert first_result["opponent_private_discards"] == ["8h"]
    assert second_result["opponent_private_discards"] == ["7h"]


def test_merge_refinement_replaces_matching_state() -> None:
    base = [make_record(0), make_record(1, seat="second")]
    refinement = [{**make_record(1, seat="second"), "profile": "stage9f_p2", "score_gap": 9.0}]

    merged, summary = merge_records(base, refinement)

    assert summary["output_records"] == 2
    assert summary["replaced_records"] == 1
    assert summary["missing_refinement_keys"] == 0
    assert merged[0]["label_source"] == "base_fast_t2"
    assert merged[1]["label_source"] == "stage9f_p2_refinement"
    assert merged[1]["profile"] == "stage9f_p2"
    assert merged[1]["score_gap"] == 9.0
