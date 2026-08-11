from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.freeze_hu_m43_attempt09_development as freeze
from ofc_regular.hu_m43_attempt09_contract import M43_ATTEMPT09_PLAN_SHA256, M43_ATTEMPT09_PROFILES
from ofc_regular.hu_m43_attempt09_spot import RECEIVE_SCHEMA, canonical_json_bytes
from ofc_regular.select_hu_m43_attempt09_development import (
    ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA,
)


RUN_NAME = "attempt09-development-freeze-test"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    run = tmp_path / RUN_NAME
    received = tmp_path / "received"
    selector_dir = tmp_path / "selector"
    run.mkdir()
    (run / "execution_authorization.json").write_bytes(
        canonical_json_bytes({"schema": "test-execution", "status": "authorized"})
    )
    execution_sha = _sha(run / "execution_authorization.json")
    source_sha = "1" * 64
    schedule_sha = "2" * 64
    startup_sha = "3" * 64
    manifest = {
        "mode": "development", "total_shards": 200, "run_name": RUN_NAME,
        "plan_sha256": M43_ATTEMPT09_PLAN_SHA256, "source_sha256": source_sha,
        "schedule_sha256": schedule_sha, "startup_sha256": startup_sha,
    }
    _write(run / "manifest.json", manifest)
    manifest_sha = _sha(run / "manifest.json")
    monkeypatch.setattr(freeze.spot, "validate_launch", lambda path: (manifest, {"status": "authorized"}))

    merged = received / "merged" / "teacher.jsonl"
    merged.parent.mkdir(parents=True)
    merged.write_bytes(b"canonical-development200-merged\n")
    merged_sha = _sha(merged)
    receive = {
        "schema": RECEIVE_SCHEMA, "status": "complete", "run_name": RUN_NAME,
        "mode": "development", "roots": 200, "root_indices": list(range(200)),
        "profiles": [M43_ATTEMPT09_PROFILES[index % 5] for index in range(200)],
        "manifest_sha256": manifest_sha, "schedule_sha256": schedule_sha,
        "source_sha256": source_sha, "authorization_sha256": execution_sha,
        "merged_sha256": merged_sha, "audit_sha256": "4" * 64,
        "batch_boundary_validation_count": 1,
        "per_shard_boundary_revalidation_count": 0, "selector_executed": False,
        "current_profile_mutated": False, "runtime_policy_activated": False,
    }
    receive_file = received / "merged" / "receive_receipt.json"
    _write(receive_file, receive)

    selector_source_sha = _sha(
        Path(freeze.__file__).with_name("select_hu_m43_attempt09_development.py")
    )
    decision = {
        "schema": ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA,
        "status": "go_write_separate_search_freeze_only", "decision": "go",
        "search_freeze_authorized": True, "selected_arm": None,
        "selected_threshold": None,
        "source": {
            "input_jsonl_sha256": merged_sha, "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
            "authorization_sha256": execution_sha, "source_package_sha256": source_sha,
            "run_name": RUN_NAME, "selector_source_sha256": selector_source_sha,
            "root_identity_sha256": "5" * 64,
        },
        "development_population": {"roots": 200}, "metrics": {"overall": {}},
        "gates": [{"name": "all", "passed": True, "observed": True, "requirement": "true"}],
        "decision_contract": {
            "single_frozen_search_architecture": True, "arm_selection_performed": False,
            "threshold_selection_performed": False, "gate_evaluation_count": 1,
            "all_gates_required": True,
        },
        "integrity": {"violations": 0},
        "science_boundary": {
            "assessment_source": "disjoint_E256_locked_final_nonbaseline_output_vs_explicit_baseline",
            "teacher_values_are_realized_match_ev": False,
            "future_audit_authorized": False, "fit_performed": False,
            "threshold_selected": False, "runtime_policy_activated": False,
            "current_profile_mutated": False, "full_replacement_enabled": False,
        },
    }
    decision_file = selector_dir / "decision.json"
    _write(decision_file, decision)
    receipt = {
        "schema": ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA,
        "status": "single_frozen_gate_evaluation_complete", "run_name": RUN_NAME,
        "decision_sha256": _sha(decision_file), "decision": "go",
        "search_freeze_authorized": True, "gate_evaluation_count": 1,
        "selector_executed": True, "future_audit_authorized": False,
        "fit_performed": False, "threshold_selected": False,
        "runtime_policy_activated": False, "current_profile_mutated": False,
    }
    receipt_file = selector_dir / "decision_receipt.json"
    _write(receipt_file, receipt)
    return {
        "run": run, "received": received, "decision": decision_file,
        "selector_receipt": receipt_file, "receive": receive_file,
        "manifest": run / "manifest.json", "execution": run / "execution_authorization.json",
        "manifest_payload": manifest,
    }


def _freeze(paths: dict, output: Path) -> dict:
    return freeze.freeze_attempt09_development(
        run_dir=paths["run"], received_dir=paths["received"],
        decision_path=paths["decision"],
        decision_receipt_path=paths["selector_receipt"], output=output,
    )


def test_clean_go_writes_canonical_future_audit_only_freeze(tmp_path, monkeypatch):
    paths = _bundle(tmp_path, monkeypatch)
    output = tmp_path / "attempt09-development-go-freeze.json"
    result = _freeze(paths, output)
    assert output.read_bytes() == canonical_json_bytes(result)
    assert result["status"] == "go_freeze_attempt09_development"
    assert result["decision"] == "go"
    assert result["authorization_scope"]["future_audit_package_authorized"] is True
    assert result["authorization_scope"]["future_audit_authorization_artifact_authorized"] is True
    assert result["authorization_scope"]["future_audit_launch_authorized"] is False
    assert result["future_audit_authorized"] is False
    assert result["fit_performed"] is False
    assert result["runtime_policy_activated"] is False
    assert result["current_profile_mutated"] is False


def test_no_go_is_rejected_even_with_hash_consistent_receipt(tmp_path, monkeypatch):
    paths = _bundle(tmp_path, monkeypatch)
    decision = json.loads(paths["decision"].read_text())
    decision.update(status="no_go_close_attempt09_development", decision="no_go", search_freeze_authorized=False)
    decision["gates"][0]["passed"] = False
    _write(paths["decision"], decision)
    receipt = json.loads(paths["selector_receipt"].read_text())
    receipt.update(decision_sha256=_sha(paths["decision"]), decision="no_go", search_freeze_authorized=False)
    _write(paths["selector_receipt"], receipt)
    with pytest.raises(ValueError, match="Go boundary"):
        _freeze(paths, tmp_path / "freeze.json")


@pytest.mark.parametrize("mutation", ["decision_receipt", "receive", "manifest", "execution", "source"])
def test_tampered_closure_is_rejected(tmp_path, monkeypatch, mutation):
    paths = _bundle(tmp_path, monkeypatch)
    if mutation == "decision_receipt":
        payload = json.loads(paths["selector_receipt"].read_text())
        payload["gate_evaluation_count"] = 2
        _write(paths["selector_receipt"], payload)
    elif mutation == "receive":
        payload = json.loads(paths["receive"].read_text())
        payload["source_sha256"] = "9" * 64
        _write(paths["receive"], payload)
    elif mutation == "manifest":
        paths["manifest"].write_bytes(paths["manifest"].read_bytes() + b" ")
    elif mutation == "execution":
        paths["execution"].write_bytes(paths["execution"].read_bytes() + b" ")
    else:
        decision = json.loads(paths["decision"].read_text())
        decision["source"]["source_package_sha256"] = "9" * 64
        _write(paths["decision"], decision)
        receipt = json.loads(paths["selector_receipt"].read_text())
        receipt["decision_sha256"] = _sha(paths["decision"])
        _write(paths["selector_receipt"], receipt)
    with pytest.raises(ValueError):
        _freeze(paths, tmp_path / "freeze.json")


def test_noncanonical_selector_artifact_is_rejected(tmp_path, monkeypatch):
    paths = _bundle(tmp_path, monkeypatch)
    payload = json.loads(paths["decision"].read_text())
    paths["decision"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        _freeze(paths, tmp_path / "freeze.json")


def test_freeze_never_overwrites_existing_output(tmp_path, monkeypatch):
    paths = _bundle(tmp_path, monkeypatch)
    output = tmp_path / "freeze.json"
    first = _freeze(paths, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        _freeze(paths, output)
    assert output.read_bytes() == original == canonical_json_bytes(first)
