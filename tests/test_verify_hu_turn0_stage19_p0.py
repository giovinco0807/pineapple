from __future__ import annotations

from copy import deepcopy

from ofc_regular.verify_hu_turn0_stage19_p0 import verify_holdout


def _plan() -> dict:
    return {
        "candidate_model": "model.pkl",
        "candidate_topk": 60,
        "evaluation": {"paired_seeds_per_config": 5000},
        "acceptance": {
            "selection_order": "main then conservative",
            "aggregate_avg_delta_per_hand_min_exclusive": 0.0,
            "aggregate_ci95_low_min": 0.0,
            "avg_delta_per_fire_min_exclusive": 0.0,
            "per_fire_ci95_low_min": 0.0,
            "minimum_fires": 300,
            "non_fired_nonzero_count": 0,
            "p99_fire_loss_max": 40.0,
            "max_fire_loss_max": 50.0,
            "duplicate_event_count": 0,
            "seed_set_mismatch_configs": [],
        },
        "candidates": [
            {
                "id": "main",
                "priority": 1,
                "min_margin_by_seat": {"first": 0.5, "second": 1.5},
                "allowed_seats": ["first", "second"],
            },
            {
                "id": "conservative",
                "priority": 2,
                "min_margin_by_seat": {"first": 0.75, "second": 999.0},
                "allowed_seats": ["first"],
            },
        ],
    }


def _row(config_id: str, *, first: float = 0.1, second: float = 0.1) -> dict:
    margins = (
        {"first": 0.5, "second": 1.5}
        if config_id == "main"
        else {"first": 0.75, "second": 999.0}
    )
    seats = ["first", "second"] if config_id == "main" else ["first"]
    return {
        "config_id": config_id,
        "candidate_model": "model.pkl",
        "candidate_topk": 60,
        "paired_seeds": 5000,
        "min_margin_by_seat": margins,
        "allowed_seats": seats,
        "fires": 500,
        "non_fired_nonzero_count": 0,
        "avg_delta_per_hand": 0.02,
        "ci95_low": 0.001,
        "avg_delta_per_fire": 0.5,
        "per_fire_ci95_low": 0.1,
        "p99_fire_loss": 30.0,
        "max_fire_loss": 40.0,
        "by_seat": {
            "first": {"avg_delta_per_hand": first},
            "second": {"avg_delta_per_hand": second},
        },
    }


def _summary() -> dict:
    return {
        "duplicate_event_count": 0,
        "seed_set_mismatch_configs": [],
        "config_results": [_row("main"), _row("conservative")],
    }


def test_stage19_verifier_selects_first_passing_candidate() -> None:
    result = verify_holdout(_plan(), _summary())

    assert result["passed"] is True
    assert result["selected_candidate"] == "main"


def test_stage19_verifier_falls_through_to_preregistered_conservative() -> None:
    summary = _summary()
    summary["config_results"][0]["ci95_low"] = -0.01

    result = verify_holdout(_plan(), summary)

    assert result["passed"] is True
    assert result["selected_candidate"] == "conservative"
    assert result["candidate_results"][0]["passed"] is False


def test_stage19_verifier_rejects_dirty_cancellation_and_negative_enabled_seat() -> None:
    summary = deepcopy(_summary())
    for row in summary["config_results"]:
        row["non_fired_nonzero_count"] = 1
        row["by_seat"]["first"]["avg_delta_per_hand"] = -0.01

    result = verify_holdout(_plan(), summary)

    assert result["passed"] is False
    assert result["selected_candidate"] is None
    failed = {
        check["check"]
        for candidate in result["candidate_results"]
        for check in candidate["checks"]
        if not check["passed"]
    }
    assert "non_fired_nonzero_count" in failed
    assert "enabled_seat_point_estimates_nonnegative" in failed


def test_stage19_verifier_rejects_runtime_metadata_drift() -> None:
    summary = _summary()
    summary["config_results"][0]["candidate_topk"] = 59
    summary["config_results"][1]["candidate_model"] = "other.pkl"

    result = verify_holdout(_plan(), summary)

    assert result["passed"] is False


def test_stage19_verifier_checks_optional_safe_selector_metadata() -> None:
    plan = _plan()
    plan["safe_selector_model"] = "selector.pkl"
    for candidate in plan["candidates"]:
        candidate["safe_selector_threshold_by_seat"] = {
            "first": 0.9,
            "second": 1.0,
        }
    summary = _summary()
    for row in summary["config_results"]:
        row["safe_selector_model"] = "selector.pkl"
        row["safe_selector_threshold_by_seat"] = {"first": 0.9, "second": 1.0}

    assert verify_holdout(plan, summary)["passed"] is True
    summary["config_results"][0]["safe_selector_threshold_by_seat"]["first"] = 0.8
    summary["config_results"][1]["safe_selector_model"] = "wrong.pkl"

    result = verify_holdout(plan, summary)

    assert result["passed"] is False


def test_stage19_verifier_normalizes_windows_path_separators() -> None:
    plan = _plan()
    plan["candidate_model"] = r"models\candidate.pkl"
    plan["safe_selector_model"] = r"models\selector.pkl"
    for candidate in plan["candidates"]:
        candidate["safe_selector_threshold_by_seat"] = {
            "first": 0.9,
            "second": 1.0,
        }
    summary = _summary()
    for row in summary["config_results"]:
        row["candidate_model"] = "models/candidate.pkl"
        row["safe_selector_model"] = "models/selector.pkl"
        row["safe_selector_threshold_by_seat"] = {"first": 0.9, "second": 1.0}

    assert verify_holdout(plan, summary)["passed"] is True


def test_stage19_verifier_supports_tail_audit_acceptance_without_hard_max() -> None:
    plan = _plan()
    del plan["acceptance"]["max_fire_loss_max"]
    plan["acceptance"]["max_fire_loss_report_only"] = True
    plan["tail_audit"] = {
        "minimum_records": 30,
        "delta_mean_min_exclusive": 0.0,
        "negative_rate_max": 0.2,
    }
    summary = _summary()
    for row in summary["config_results"]:
        row["max_fire_loss"] = 99.0
    tail_audit = {
        "records": 30,
        "missing_shards": [],
        "common_random_futures_verified": True,
        "action_mapping_verified": True,
        "delta_mean": 0.8,
        "label_counts": {"positive": 9, "gray": 17, "negative": 4},
    }

    result = verify_holdout(plan, summary, tail_audit)

    assert result["passed"] is True
    checks = {
        check["check"]: check
        for candidate in result["candidate_results"]
        for check in candidate["checks"]
    }
    assert "max_fire_loss" not in checks
    assert checks["tail_audit_negative_rate"]["passed"] is True


def test_stage19_verifier_rejects_missing_or_bad_tail_audit() -> None:
    plan = _plan()
    del plan["acceptance"]["max_fire_loss_max"]
    plan["tail_audit"] = {
        "minimum_records": 30,
        "delta_mean_min_exclusive": 0.0,
        "negative_rate_max": 0.2,
    }

    assert verify_holdout(plan, _summary(), None)["passed"] is False
    bad_tail = {
        "records": 30,
        "missing_shards": [],
        "common_random_futures_verified": True,
        "action_mapping_verified": True,
        "delta_mean": -0.1,
        "label_counts": {"negative": 10, "gray": 20},
    }
    assert verify_holdout(plan, _summary(), bad_tail)["passed"] is False
