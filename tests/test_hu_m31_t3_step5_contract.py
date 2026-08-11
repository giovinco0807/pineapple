from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular.validate_hu_m31_t3_step5_contract import (
    EXPECTED_CANONICAL_CONTRACT_SHA256,
    EXPECTED_HISTORICAL_CONFIG_SEED_MAX,
    EXPECTED_PLANNED_SEED_MIN,
    STEP5_VALIDATION_SCHEMA,
    _planned_seed_values,
    _write_json_atomic,
    canonical_contract_sha256,
    scan_historical_config_seed_max,
    validate_contract_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs/hu_joint_policy_m31_t3_step5_contract.json"


def _contract() -> dict[str, object]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_live_step5_contract_matches_frozen_science_and_seed_hash():
    payload = _contract()
    result = validate_contract_payload(
        payload,
        repo_root=REPO_ROOT,
        verify_prerequisites=False,
        verify_historical_scan=False,
    )

    assert canonical_contract_sha256(payload) == EXPECTED_CANONICAL_CONTRACT_SHA256
    assert result["schema"] == STEP5_VALIDATION_SCHEMA
    assert result["status"] == "pass"
    assert result["spot_canary_authorized"] is True
    assert result["production_fanout_authorized"] is False
    assert result["current_profile_changed"] is False
    assert result["m31_complete"] is False


def test_all_66000_planned_seed_values_are_unique_and_above_history():
    payload = _contract()
    values, rows = _planned_seed_values(payload["seed_contract"])

    assert len(values) == 66_000
    assert len(set(values)) == 66_000
    assert min(values) == EXPECTED_PLANNED_SEED_MIN
    assert min(values) > EXPECTED_HISTORICAL_CONFIG_SEED_MAX
    assert [row["name"] for row in rows] == [
        "infrastructure_canary",
        "train",
        "safety_fit",
        "threshold_lock",
        "diagnostic_teacher_holdout",
        "development_population",
        "locked_population",
        "locked_abr_probe",
    ]


def test_seed_overlap_mutation_fails_closed():
    payload = _contract()
    mutated = deepcopy(payload)
    train = mutated["seed_contract"]["schedules"][1]
    train["namespace_bases"]["candidate_seed_base"] = train["namespace_bases"][
        "hand_seed_base"
    ]

    with pytest.raises(ValueError, match="canonical contract SHA-256 changed"):
        validate_contract_payload(
            mutated,
            repo_root=REPO_ROOT,
            verify_prerequisites=False,
            verify_historical_scan=False,
        )

    values, _ = _planned_seed_values(mutated["seed_contract"])
    assert len(set(values)) < len(values)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["promotion_evaluation"].__setitem__(
            "false_positive_override_rate_max", 0.31
        ),
        lambda payload: payload["promotion_evaluation"].__setitem__(
            "first_seat_delta_ev_per_hand_ci95_low_min_exclusive", -0.01
        ),
        lambda payload: payload["teacher_search"][
            "production_label_budget"
        ].__setitem__("evaluation_samples", 8),
        lambda payload: payload["information_set"].__setitem__(
            "opponent_private_discards", True
        ),
        lambda payload: payload["spot_execution"].__setitem__(
            "production_fanout_authorized", True
        ),
        lambda payload: payload["activation_guards"].__setitem__(
            "current_profile_changed", True
        ),
    ),
)
def test_weakened_science_or_activation_contract_fails_closed(mutate):
    payload = _contract()
    mutate(payload)
    with pytest.raises(ValueError, match="canonical contract SHA-256 changed"):
        validate_contract_payload(
            payload,
            repo_root=REPO_ROOT,
            verify_prerequisites=False,
            verify_historical_scan=False,
        )


def test_historical_seed_scan_excludes_step5_contract_and_status(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "a.json").write_text('{"seed":10}', encoding="utf-8")
    (configs / "b.json").write_text(
        '{"nested":{"candidate_seed_base":20}}', encoding="utf-8"
    )
    (configs / "hu_joint_policy_m31_t3_step5_contract.json").write_text(
        '{"seed":999}', encoding="utf-8"
    )
    (configs / "hu_joint_policy_m31_t3_step5_status.json").write_text(
        '{"seed":888}', encoding="utf-8"
    )

    maximum, file_count = scan_historical_config_seed_max(tmp_path)
    assert maximum == 20
    assert file_count == 2


def test_step5_validation_artifact_is_atomic_and_write_once(tmp_path):
    output = tmp_path / "validation.json"
    _write_json_atomic(output, {"schema": STEP5_VALIDATION_SCHEMA})
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == (
        STEP5_VALIDATION_SCHEMA
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(output, {"schema": "changed"})
