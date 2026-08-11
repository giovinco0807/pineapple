from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_candidate02_windows_fallback_evidence_v1 as subject


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "configs/hu_joint_policy_m31_t3_candidate02_windows_fallback_evidence_v1.json"
)
WINDOWS_CANDIDATE = (
    ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "candidate02_frozen/ofc_hu_m3_engine_candidate02_4d07d13eadc24356.dll"
)
LINUX_CANDIDATE = (
    ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "candidate02_frozen/libofc_hu_m3_engine_candidate02_4050e04b22d7943d.so"
)
WINDOWS_FEATURE = ROOT / "target/release/ofc_stage3_feature_encoder.dll"
LINUX_FEATURE = (
    ROOT
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001/"
    "package_src/target/release/libofc_stage3_feature_encoder.so"
)


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="ascii"))


def _reseal(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = subject.canonical_sha256(result)
    return result


def test_sealed_candidate_and_feature_platform_evidence_replays() -> None:
    value = subject.validate_evidence_value(_evidence())
    rows = value["candidate_engine_parity"]["rows"]

    assert [row["root_index"] for row in rows] == [0, 1]
    assert [row["seat"] for row in rows] == ["first", "second"]
    assert [len(row["portable_payload"]["action_values"]) for row in rows] == [9, 21]
    assert [
        row["portable_payload"]["child_information_set_count"] for row in rows
    ] == [61_560, 1_680]
    assert all(
        row["portable_decision_sha256"]
        == subject.canonical_sha256(row["portable_payload"])
        for row in rows
    )
    assert value["feature_encoder_parity"]["row_count"] == 512
    assert value["feature_encoder_parity"]["input_bit_exact"] is True
    assert value["feature_encoder_parity"]["output_bit_exact"] is True
    assert value["fallback_boundary"]["primary_platform"] == "accepted_linux_or_wsl"
    assert value["fallback_boundary"]["windows_role"] == "slow_local_fallback_only"
    assert value["fallback_boundary"]["current_profile_changed"] is False

    assert subject.sha256_file(WINDOWS_CANDIDATE) == (
        subject.WINDOWS_CANDIDATE_SHA256
    )
    assert subject.sha256_file(LINUX_CANDIDATE) == (
        subject.PINNED_CANDIDATE_LIBRARY_SHA256
    )
    assert subject.sha256_file(WINDOWS_FEATURE) == subject.WINDOWS_FEATURE_SHA256
    assert subject.sha256_file(LINUX_FEATURE) == subject.LINUX_FEATURE_SHA256


def test_q_or_safety_tamper_fails_even_when_outer_receipt_is_resealed() -> None:
    q_tamper = _evidence()
    q_tamper["candidate_engine_parity"]["rows"][0]["portable_payload"][
        "action_values"
    ][0]["evaluation_q"] += 0.25
    with pytest.raises(ValueError, match="safety boundary"):
        subject.validate_evidence_value(_reseal(q_tamper))

    boundary_tamper = _evidence()
    boundary_tamper["fallback_boundary"]["training_eligible"] = True
    with pytest.raises(ValueError, match="safety boundary"):
        subject.validate_evidence_value(_reseal(boundary_tamper))
