from __future__ import annotations

import json

import pytest

from ofc_regular.validate_hu_m2_teacher import (
    M2_PILOT_SCHEMA,
    ROOT_GENERATION_POLICY,
    TEACHER_VALUE_STATUS,
    M2PilotConfig,
    generate_fresh_t3_roots,
    main,
    run_m2_correctness_pilot,
)


def test_fresh_root_generation_is_deterministic_and_has_both_live_t3_seats():
    first = generate_fresh_t3_roots(2026071307)
    repeated = generate_fresh_t3_roots(2026071307)

    assert [root.fingerprint() for root in first] == [
        root.fingerprint() for root in repeated
    ]
    assert len({root.fingerprint() for root in first}) == 2
    assert [root.seat for root in first] == ["first", "second"]
    assert [root.to_act_order for root in first] == ["first", "second"]
    assert [root.hero_board.card_count() for root in first] == [9, 9]
    assert [root.opponent_public_board.card_count() for root in first] == [9, 11]
    assert [len(root.hero_private_discards) for root in first] == [2, 2]
    assert [len(root.dealt_cards) for root in first] == [3, 3]


def test_one_hand_sample_one_pilot_reports_required_correctness_gates(tmp_path):
    output = tmp_path / "m2-pilot.json"
    config = M2PilotConfig(
        hands=1,
        seed_start=2026071311,
        seed_stride=1_000_003,
        teacher_seed=71,
        candidate_samples=1,
        evaluation_samples=1,
        downstream_t3_samples=1,
        downstream_t4_samples=1,
        run_id="m2-pilot-test",
    )

    report = run_m2_correctness_pilot(config, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == report
    assert report["schema"] == M2_PILOT_SCHEMA
    assert report["status"] == "smoke_pass"
    assert report["completion_gate_aggregate"] is False
    assert report["teacher_values"] == TEACHER_VALUE_STATUS
    assert report["promotion_evidence"] is False
    assert report["match_ev_reported"] is False
    assert report["configuration"]["seed_stride"] == 1_000_003
    assert report["root_count"] == 2
    assert report["seat_counts"] == {"first": 1, "second": 1}
    assert report["unique_root_fingerprint_count"] == 2
    assert len(report["unique_root_fingerprints"]) == 2
    assert report["candidate_evaluation_rng_overlap_count"] == 0
    assert report["global_candidate_evaluation_rng_overlap_count"] == 0
    assert report["deterministic_rerun"]["match"] is True
    assert all(report["gates"].values())
    assert report["root_generation"] == {
        "policy": ROOT_GENERATION_POLICY,
        "profile_selection": "none",
        "current_profile_read": False,
        "model_artifacts_loaded": [],
    }
    assert report["compute_scope"]["cloud_actions_performed"] is False
    timings = report["seconds_per_root"]
    assert timings["count"] == 2
    assert 0.0 <= timings["p50"] <= timings["p95"] <= timings["max"]
    assert {root["seat"] for root in report["roots"]} == {"first", "second"}
    assert all(
        root["teacher_value_status"] == TEACHER_VALUE_STATUS
        and root["candidate_evaluation_rng_overlap_count"] == 0
        for root in report["roots"]
    )
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError, match="already exists"):
        run_m2_correctness_pilot(config, output)


def test_cli_writes_atomic_one_hand_report(tmp_path, capsys):
    output = tmp_path / "cli-pilot.json"

    main(
        [
            "--output",
            str(output),
            "--hands",
            "1",
            "--seed-start",
            "2026071321",
            "--seed-stride",
            "1000003",
            "--teacher-seed",
            "81",
            "--candidate-samples",
            "1",
            "--evaluation-samples",
            "1",
            "--downstream-t3-samples",
            "1",
            "--downstream-t4-samples",
            "1",
            "--run-id",
            "m2-cli-test",
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert emitted["status"] == "smoke_pass"
    assert emitted["root_count"] == 2
    assert emitted["teacher_values"] == TEACHER_VALUE_STATUS
    assert persisted["status"] == "smoke_pass"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hands": 5}, "hands must be between"),
        ({"seed_stride": 0}, "seed_stride must be positive"),
        ({"candidate_samples": 9}, "candidate_samples must be between"),
        ({"downstream_t4_samples": 0}, "downstream_t4_samples must be between"),
    ],
)
def test_pilot_config_enforces_small_local_compute_caps(kwargs, message):
    with pytest.raises(ValueError, match=message):
        M2PilotConfig(**kwargs)
