from ofc_regular.audit_hu_turn2_stage8_high_mc import (
    parse_configs,
    runtime_fired_rows,
    select_audit_states,
    stratified_for_replay,
    threshold_sweep_rows,
)


def _row(**overrides):
    base = {
        "state_index": "1",
        "split": "test",
        "bucket_group": "natural",
        "run_bucket": "natural",
        "seat": "first",
        "pilot_gate_label": "gray",
        "actual_high_regret": "False",
        "actual_low_margin": "False",
        "actual_teacher_disagreement": "False",
        "actual_delta_candidate_vs_baseline": "0.0",
        "actual_delta_best_vs_baseline": "0.0",
        "predicted_delta_vs_baseline": "0.0",
        "reference_margin_raw": "0.0",
        "gate_probability": "0.0",
        "SE_delta": "0.1",
        "candidate_is_baseline": "0",
        "teacher_best_margin": "0.2",
    }
    base.update({key: str(value) for key, value in overrides.items()})
    return base


def test_runtime_fired_rows_marks_missing_dead_cards_replay_ineligible(tmp_path):
    path = tmp_path / "runtime.jsonl"
    path.write_text(
        '{"override_fired":true,"config_id":"m2.5_r0_g0.9","hand_id":1}\n'
        '{"override_fired":true,"config_id":"m2.5_r0_g0.9","hand_id":2,"dead_cards":["2c"]}\n',
        encoding="utf-8",
    )

    rows = runtime_fired_rows(path)

    assert rows[0]["replay_ready"] is False
    assert rows[0]["replay_ineligible"] is True
    assert rows[0]["exclude_from_exact_replay"] is True
    assert rows[0]["legacy_runtime_log"] is True
    assert rows[0]["missing_dead_cards"] is True
    assert rows[0]["replay_blocker"] == "missing_dead_cards_in_legacy_runtime_log"
    assert rows[1]["replay_ready"] is True
    assert rows[1]["replay_ineligible"] is False


def test_select_audit_states_covers_fired_near_missed_and_suspected_fp():
    configs = parse_configs("2.75/0.05/0.90")
    rows = [
        _row(
            state_index=10,
            predicted_delta_vs_baseline=3.0,
            reference_margin_raw=0.10,
            gate_probability=0.95,
            actual_delta_candidate_vs_baseline=-0.2,
            actual_low_margin=True,
        ),
        _row(
            state_index=11,
            predicted_delta_vs_baseline=2.60,
            reference_margin_raw=0.10,
            gate_probability=0.85,
            actual_delta_candidate_vs_baseline=0.1,
        ),
        _row(
            state_index=12,
            predicted_delta_vs_baseline=1.0,
            reference_margin_raw=0.20,
            gate_probability=0.4,
            actual_delta_candidate_vs_baseline=1.2,
            SE_delta=0.2,
        ),
    ]

    selected = select_audit_states(
        rows,
        configs,
        split="test",
        max_fired=10,
        max_near_fired=10,
        max_missed_positive=10,
        max_suspected_false_positive=10,
        near_margin_window=0.5,
        near_gate_low=0.75,
        near_gate_high=0.95,
        near_reference_low=0.0,
        near_reference_high=0.30,
        missed_positive_delta=0.50,
        missed_positive_se_multiple=2.0,
    )

    groups = {item.audit_group for item in selected}
    assert "fired_teacher" in groups
    assert "suspected_false_positive" in groups
    assert "near_fired" in groups
    assert "missed_positive" in groups


def test_threshold_sweep_uses_high_mc_delta_for_override_quality():
    rows = [
        {
            "candidate_original_index": 3,
            "baseline_original_index": 1,
            "predicted_delta": 3.0,
            "reference_margin_raw": 0.10,
            "gate_probability": 0.92,
            "high_mc_delta_candidate_vs_baseline": 1.5,
        },
        {
            "candidate_original_index": 4,
            "baseline_original_index": 2,
            "predicted_delta": 3.0,
            "reference_margin_raw": 0.10,
            "gate_probability": 0.92,
            "high_mc_delta_candidate_vs_baseline": -0.5,
        },
    ]

    sweep = threshold_sweep_rows(rows)
    target = next(
        row
        for row in sweep
        if row["hu_turn2_min_margin"] == 2.75
        and row["hu_turn2_reference_min_margin"] == 0.05
        and row["hu_turn2_gate_threshold"] == 0.90
    )

    assert target["override_count"] == 2
    assert target["false_positive_count"] == 1
    assert target["high_mc_avg_gain_on_override"] == 0.5


def test_stratified_for_replay_honors_origin_buckets_without_duplicates():
    configs = parse_configs("2.75/0.05/0.90")
    rows = [
        _row(state_index=1, predicted_delta_vs_baseline=3.0, reference_margin_raw=0.1, gate_probability=0.95),
        _row(state_index=2, predicted_delta_vs_baseline=3.2, reference_margin_raw=0.1, gate_probability=0.96),
        _row(state_index=3, predicted_delta_vs_baseline=1.0, actual_delta_candidate_vs_baseline=2.0, SE_delta=0.2),
        _row(state_index=4, predicted_delta_vs_baseline=2.7, reference_margin_raw=0.1, gate_probability=0.85),
    ]
    selected = select_audit_states(
        rows,
        configs,
        split="test",
        max_fired=10,
        max_near_fired=10,
        max_missed_positive=10,
        max_suspected_false_positive=10,
        near_margin_window=0.5,
        near_gate_low=0.75,
        near_gate_high=0.95,
        near_reference_low=0.0,
        near_reference_high=0.30,
        missed_positive_delta=0.50,
        missed_positive_se_multiple=2.0,
    )

    replay = stratified_for_replay(
        selected,
        include_fired=1,
        include_suspected_false_positive=0,
        include_missed_positive=1,
        include_near_fired=1,
    )

    assert [row["replay_origin_group"] for row in replay[:3]] == [
        "fired_teacher",
        "missed_positive",
        "near_fired",
    ]
    assert len({row["state_index"] for row in replay}) == len(replay)
