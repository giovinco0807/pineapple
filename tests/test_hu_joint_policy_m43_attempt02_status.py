from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "configs" / "hu_joint_policy_m43_attempt02_status.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status() -> dict[str, object]:
    return _read_json(STATUS_PATH)


def _artifact(status: dict[str, object], name: str) -> Path:
    artifacts = status["artifacts"]
    assert isinstance(artifacts, dict)
    binding = artifacts[name]
    assert isinstance(binding, dict)
    relative = binding["path"]
    assert isinstance(relative, str)
    return ROOT / relative


def test_attempt02_is_an_immutable_precalibration_no_go() -> None:
    status = _status()
    assert status["schema"] == "hu_joint_policy_m43_attempt02_status_v1"
    assert status["status"] == "complete_no_go_precalibration"
    assert status["decision"] == "no_go_precalibration"

    guards = status["guards"]
    assert guards == {
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
        "calibration_opened_after_no_go": False,
        "locked_holdout_opened_after_no_go": False,
        "population_evaluation_started": False,
        "large_scale_authorized": False,
    }
    assert status["model"] == {
        "written": False,
        "artifact_present": False,
        "sha256": None,
    }

    expected = status["baseline_hash_audit"]["policy_registry_expected_sha256"]
    actual = _sha256(ROOT / "src" / "ofc_regular" / "ai_profiles.py")
    assert actual == expected
    assert status["baseline_hash_audit"]["current_mapping_changed"] is False


def test_attempt02_status_binds_the_exact_r2_receipt_and_manifest() -> None:
    status = _status()
    artifacts = status["artifacts"]
    receipt_path = _artifact(status, "receive_receipt")
    manifest_path = _artifact(status, "training_manifest")

    assert _sha256(receipt_path) == artifacts["receive_receipt"]["file_sha256"]
    assert _sha256(manifest_path) == artifacts["training_manifest"]["file_sha256"]

    receipt = _read_json(receipt_path)
    manifest = _read_json(manifest_path)
    assert receipt["status"] == "no_go_precalibration"
    assert manifest["status"] == "no_go_precalibration"
    assert manifest["promotion_status"] == "no_go_precalibration"
    assert receipt["training_manifest_sha256"] == _sha256(manifest_path)
    assert manifest["receipt_sha256"] == artifacts["training_manifest"][
        "assembler_pre_final_receipt_sha256"
    ]
    assert receipt["model_sha256"] is None
    assert manifest["model_written"] is False
    assert not any(
        path.is_file()
        for path in receipt_path.parent.iterdir()
        if path.suffix.lower() in {".onnx", ".pt", ".pkl", ".joblib"}
    )


def test_attempt02_precalibration_metrics_and_all_gate_results_are_frozen() -> None:
    status = _status()
    manifest = _read_json(_artifact(status, "training_manifest"))
    frozen = status["precalibration_gate"]
    observed = manifest["crossfit"]["precalibration_gate"]

    assert observed["status"] == "no_go"
    assert observed["config_sha256"] == frozen["config_sha256"]
    assert observed["metrics"]["states"] == 200
    assert observed["metrics"]["proposal_positive_count"] == 69
    assert observed["metrics"]["proposal_positive_rate"] == 0.345
    assert observed["metrics"]["proposal_mean_teacher_delta"] == (
        -1.1242242603887602
    )

    gates = observed["gates"]
    passed = sorted(name for name, value in gates.items() if value)
    failed = sorted(name for name, value in gates.items() if not value)
    assert passed == sorted(frozen["passed_gates"])
    assert failed == sorted(row["name"] for row in frozen["failed_gates"])
    assert (len(passed), len(failed)) == (11, 2)
    assert (frozen["gates_total"], frozen["gates_passed"], frozen["gates_failed"]) == (
        13,
        11,
        2,
    )

    deltas = [row["teacher_delta"] for row in observed["proposal_rows"]]
    mean = statistics.fmean(deltas)
    margin = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    ci = frozen["failed_gates"][1]["ci95_normal"]
    assert math.isclose(mean, -1.1242242603887602, abs_tol=1e-15)
    assert math.isclose(mean - margin, ci["low"], abs_tol=1e-14)
    assert math.isclose(mean + margin, ci["high"], abs_tol=1e-14)
    assert [round(mean - margin, 4), round(mean + margin, 4)] == ci[
        "rounded_4dp"
    ]


def test_attempt02_never_opened_calibration_or_inherited_locked_content() -> None:
    status = _status()
    manifest = _read_json(_artifact(status, "training_manifest"))
    boundaries = status["unopened_boundaries"]

    assert boundaries["calibration"] == manifest["calibration"]
    assert manifest["crossfit"]["calibration_opened"] is False
    assert manifest["crossfit"]["precalibration_gate"]["calibration_opened"] is False
    assert boundaries["inherited_locked_holdout"]["content_opened"] is False
    assert boundaries["inherited_locked_holdout"]["input_accepted"] is False
    assert manifest["inherited_holdout_content_opened"] is False
    assert manifest["inherited_holdout_input_accepted"] is False

    receipt = _read_json(_artifact(status, "receive_receipt"))
    assert receipt["precalibration_no_go_opens_local_calibration_or_contract"] is False
    assert receipt["current_profile_mutated"] is False
    assert receipt["runtime_policy_activated"] is False


def test_attempt02_postmortem_records_the_no_go_without_promotion_claims() -> None:
    status = _status()
    doc = (ROOT / "docs" / "hu_joint_policy_m43_attempt02_postmortem.md").read_text(
        encoding="utf-8"
    )
    assert "`no_go_precalibration`" in doc
    assert "69 of 200" in doc
    assert "`0.345`" in doc
    assert "`-1.1242242603887602`" in doc
    assert "`[-1.4943, -0.7542]`" in doc
    assert status["artifacts"]["receive_receipt"]["file_sha256"] in doc
    assert status["artifacts"]["training_manifest"]["file_sha256"] in doc
    assert "not realized match EV" in doc
