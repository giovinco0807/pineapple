from ofc_regular.extract_hu_turn1_value_add_replay_targets import (
    action_pair_key,
    enriched_target,
    selected_rows,
)


def _row(label="positive", *, fired=True, reason="", delta=4.0, replay_ready=True, same_action=False, seed=101):
    candidate = {
        "placements": [["Jd", "middle"], ["Js", "middle"]],
        "discards": ["3d"],
    }
    baseline = candidate if same_action else {
        "placements": [["Jd", "bottom"], ["Js", "bottom"]],
        "discards": ["3d"],
    }
    return {
        "hand_id": seed,
        "game_id": seed,
        "hand_seed": seed,
        "paired_index": 3,
        "seat_swap": "ab",
        "seat": "second",
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
        "override_fired": fired,
        "no_override_reason": "" if fired else reason,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": delta,
        "hu_turn1_realized_delta": delta,
        "hu_turn1_value_add_label": label,
        "hu_turn1_predicted_margin": 1.25,
        "candidate_score": 2.0,
        "fallback_score": 0.5,
        "candidate_action_index": 2,
        "fallback_action_index": 1,
        "hu_turn1_action": candidate,
        "baseline_action": baseline,
        "confirm_delta": 1.5,
        "confirm_delta_se": 0.5,
        "stage10_source_run": "run-a",
    }


def test_value_add_replay_targets_select_positive_negative_and_limit_neutral():
    rows = [
        _row("positive", fired=True, delta=5.0, seed=1),
        _row("negative", fired=True, delta=-4.0, seed=2),
        _row("positive", fired=False, reason="below_confirm_delta", delta=3.0, seed=3),
        _row("neutral", fired=False, reason="below_confirm_se", delta=0.0, seed=4),
        _row("neutral", fired=False, reason="below_confirm_delta", delta=0.0, seed=5),
    ]

    selected, skipped = selected_rows(rows, max_neutral=1)

    assert len(selected) == 4
    labels = [row["hu_turn1_value_add_label"] for row in selected]
    assert labels.count("positive") == 2
    assert labels.count("negative") == 1
    assert labels.count("neutral") == 1
    assert skipped["neutral_over_limit"] == 1


def test_value_add_replay_targets_skip_same_action_and_replay_ineligible():
    rows = [
        _row("positive", fired=True, same_action=True, seed=1),
        _row("negative", fired=True, replay_ready=False, seed=2),
        _row("neutral", fired=False, reason="mc_best_is_baseline", seed=3),
    ]

    selected, skipped = selected_rows(rows, max_neutral=5)

    assert selected == []
    assert skipped["same_action_or_missing_action"] == 1
    assert skipped["not_replay_ready"] == 1
    assert skipped["not_value_add_target"] == 1


def test_value_add_replay_targets_can_select_fired_only_including_neutral():
    rows = [
        _row("positive", fired=True, delta=5.0, seed=1),
        _row("neutral", fired=True, delta=0.0, seed=2),
        _row("negative", fired=False, reason="below_confirm_delta", delta=-4.0, seed=3),
    ]

    selected, skipped = selected_rows(rows, max_neutral=-1, fired_only=True)

    assert {row["hand_id"] for row in selected} == {1, 2}
    assert skipped["not_fired"] == 1


def test_value_add_replay_targets_can_exclude_existing_action_pairs():
    rows = [
        _row("positive", fired=True, delta=5.0, seed=1),
        _row("negative", fired=True, delta=-4.0, seed=2),
    ]

    selected, skipped = selected_rows(
        rows,
        max_neutral=-1,
        fired_only=True,
        exclude_action_pair_keys={action_pair_key(rows[0])},
    )

    assert [row["hand_id"] for row in selected] == [2]
    assert skipped["excluded_action_pair"] == 1


def test_value_add_replay_target_enriches_existing_relabel_schema():
    target = enriched_target(
        _row("negative", fired=False, reason="below_safe_selector", delta=-6.0),
        target_id=9,
    )

    assert target["schema"] == "hu_turn1_value_add_replay_target_v1"
    assert target["safe_override_label"] == "hard_negative"
    assert target["safe_override_label_id"] == 0
    assert target["value_add_selection_bucket"] == "hard_negative_near_miss_below_safe_selector"
    assert target["source_bucket_group"] == "hu_turn1_value_add"
    assert target["stage10_source_run"] == "run-a"
    assert target["confirm_delta"] == 1.5
    assert len(target["actions"]) == 2
