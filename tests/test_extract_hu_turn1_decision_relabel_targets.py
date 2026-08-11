import json

from ofc_regular.extract_hu_turn1_decision_relabel_targets import (
    limit_per_label,
    selected_rows,
    to_target,
)


def _row(delta=6.0, *, replay_ready=True):
    return {
        "hand_id": 101,
        "game_id": 101,
        "hand_seed": 101,
        "paired_index": 3,
        "seat_swap": "ab",
        "seat": "first",
        "hero_board": {"top": ["Ah"], "middle": ["7c"], "bottom": ["Th", "Jc", "Tc"]},
        "opponent_board": {"top": ["Ks"], "middle": ["Jh", "As"], "bottom": ["9c", "6d"]},
        "cards_to_place": ["Jd", "3d", "Js"],
        "dead_cards": ["Ks", "Jh", "As", "9c", "6d", "2c"],
        "visible_dead_cards": ["Ks", "Jh", "As", "9c", "6d", "2c"],
        "true_dead_cards": ["2c", "3d"],
        "hero_private_discards": ["2c"],
        "opponent_private_discards": ["3d"],
        "visibility_model": "hidden_discard",
        "discard_visibility": "own_private_only",
        "replay_ready": replay_ready,
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": delta,
        "hu_turn1_predicted_margin": 1.25,
        "candidate_score": 2.0,
        "fallback_score": 0.5,
        "candidate_action_index": 2,
        "fallback_action_index": 1,
        "hu_turn1_action": {
            "placements": [["Jd", "middle"], ["Js", "middle"]],
            "discards": ["3d"],
        },
        "baseline_action": {
            "placements": [["Jd", "bottom"], ["Js", "bottom"]],
            "discards": ["3d"],
        },
    }


def _seat_row(delta=6.0, *, seat="first"):
    row = _row(delta)
    row["seat"] = seat
    return row


def _non_fired_row(*, margin=0.75, same_as_baseline=False):
    row = _row(0.0)
    row["override_fired"] = False
    row["no_override_reason"] = "below_hu_turn1_margin"
    row["hu_turn1_predicted_margin"] = margin
    if same_as_baseline:
        row["baseline_action"] = dict(row["hu_turn1_action"])
    return row


def test_extract_hu_turn1_decision_relabel_target_preserves_replay_fields():
    target = to_target(_row(-4.0), target_id=7)

    assert target["schema"] == "hu_turn1_stage1_decision_relabel_target_v1"
    assert target["target_id"] == 7
    assert target["player"] == 0
    assert target["dealt"] == ["Jd", "3d", "Js"]
    assert target["visible_dead_cards"] == ["Ks", "Jh", "As", "9c", "6d", "2c"]
    assert target["true_dead_cards"] == ["2c", "3d"]
    assert target["hero_private_discards"] == ["2c"]
    assert target["opponent_private_discards"] == ["3d"]
    assert target["safe_override_label"] == "hard_negative"
    assert target["safe_override_label_id"] == 0
    assert target["best_action"] == 0
    assert len(target["actions"]) == 2
    assert target["actions"][0]["source_role"] == "runtime_candidate"
    assert target["actions"][1]["source_role"] == "runtime_baseline"
    assert target["runtime_candidate_action_signature"]
    assert target["runtime_baseline_action_signature"]


def test_extract_hu_turn1_decision_relabel_targets_requires_replay_ready_by_default():
    rows = [_row(6.0, replay_ready=False), _row(-6.0, replay_ready=True)]

    selected, skipped = selected_rows(
        rows,
        labels={"positive", "hard_negative"},
        min_abs_delta=0.0,
        require_replay_ready=True,
    )

    assert len(selected) == 1
    assert skipped["not_replay_ready"] == 1
    assert selected[0]["realized_candidate_seat_delta"] == -6.0


def test_extract_hu_turn1_decision_relabel_targets_can_allow_legacy_rows():
    rows = [_row(6.0, replay_ready=False)]

    selected, skipped = selected_rows(
        rows,
        labels={"positive"},
        min_abs_delta=0.0,
        require_replay_ready=False,
    )

    assert len(selected) == 1
    assert not skipped


def test_extract_hu_turn1_decision_relabel_target_is_json_serializable():
    json.dumps(to_target(_row(0.0), target_id=1))


def test_extract_hu_turn1_decision_relabel_targets_can_limit_per_label():
    rows = [_row(-float(index + 1)) for index in range(3)]
    rows += [_row(float(index + 1)) for index in range(3)]

    limited = limit_per_label(rows, 2)

    assert len(limited) == 4
    assert sum(1 for row in limited if row["realized_candidate_seat_delta"] < 0) == 2
    assert sum(1 for row in limited if row["realized_candidate_seat_delta"] > 0) == 2


def test_extract_hu_turn1_decision_relabel_targets_can_include_non_fired_candidates():
    rows = [
        _non_fired_row(margin=0.8),
        _non_fired_row(margin=0.2),
        _non_fired_row(margin=0.9, same_as_baseline=True),
        _row(-6.0),
    ]

    selected, skipped = selected_rows(
        rows,
        labels={"positive", "hard_negative"},
        min_abs_delta=0.0,
        require_replay_ready=True,
        include_non_fired_candidates=True,
        only_non_fired_candidates=True,
        min_predicted_margin=0.5,
    )

    assert len(selected) == 1
    assert selected[0]["override_fired"] is False
    assert skipped["below_min_predicted_margin"] == 1
    assert skipped["non_fired_same_as_baseline"] == 1
    assert skipped["fired_excluded"] == 1

    target = to_target(selected[0], target_id=9)
    assert target["source_kind"] == "runtime_non_fired_candidate_decision"
    assert target["selection_reasons"] == ["runtime_non_fired_candidate"]
    assert target["safe_override_label"] == "neutral"
    assert len(target["actions"]) == 2


def test_extract_hu_turn1_decision_relabel_targets_can_filter_by_seat():
    rows = [_seat_row(6.0, seat="first"), _seat_row(-6.0, seat="second")]

    selected, skipped = selected_rows(
        rows,
        labels={"positive", "hard_negative"},
        min_abs_delta=0.0,
        require_replay_ready=True,
        seats={"second"},
    )

    assert len(selected) == 1
    assert selected[0]["seat"] == "second"
    assert skipped["seat_excluded"] == 1


def test_extract_hu_turn1_decision_relabel_targets_can_include_all_replay_ready_states():
    kept_baseline = _non_fired_row(margin=0.0, same_as_baseline=True)
    fired = _row(-6.0)
    fired["hand_seed"] = 202
    fired["hand_id"] = 202

    selected, skipped = selected_rows(
        [kept_baseline, fired],
        labels={"positive", "hard_negative"},
        min_abs_delta=0.0,
        require_replay_ready=True,
        include_all_replay_ready_states=True,
        seats={"first"},
    )

    assert len(selected) == 2
    assert not skipped
