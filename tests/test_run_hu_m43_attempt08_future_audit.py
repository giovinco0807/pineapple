from __future__ import annotations

import json
from pathlib import Path

import pytest

from ofc_regular import run_hu_m43_attempt08_future_audit as audit


def _write_canonical(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_missing_audit_authorization_fails_before_any_audit_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future-audit authorization"):
        audit.run_future_audit_shard(
            root_index=200,
            output=tmp_path / "must-not-exist.jsonl",
            run_id="dormant-audit",
            audit_open_authorization=tmp_path / "missing-audit-auth.json",
            development_pass_freeze=tmp_path / "missing-freeze.json",
            run_dir=tmp_path / "missing-run",
            launch_authorization=tmp_path / "missing-launch.json",
            development_decision=tmp_path / "missing-decision.json",
            selector_receipt=tmp_path / "missing-receipt.json",
            model=tmp_path / "missing-model.pkl",
            model_sha256="0" * 64,
            plan=tmp_path / "missing-plan.json",
            ai_profiles=tmp_path / "missing-ai-profiles.py",
        )
    assert not (tmp_path / "must-not-exist.jsonl").exists()


def test_audit_authorization_cannot_bypass_missing_development_freeze(
    tmp_path: Path,
) -> None:
    authorization = {key: None for key in audit._AUTHORIZATION_KEYS}
    authorization.update(
        {
            "schema": audit.ATTEMPT08_FUTURE_AUDIT_OPEN_AUTHORIZATION_SCHEMA,
            "status": "separately_authorized_after_immutable_development_pass_freeze",
            "future_audit_authorized": True,
        }
    )
    auth_path = tmp_path / "audit-auth.json"
    _write_canonical(auth_path, authorization)
    with pytest.raises(ValueError, match="development-pass freeze"):
        audit.load_and_validate_future_audit_open_authorization(
            auth_path,
            development_pass_freeze_path=tmp_path / "missing-freeze.json",
            run_dir=tmp_path / "missing-run",
            launch_authorization_path=tmp_path / "missing-launch.json",
            development_decision_path=tmp_path / "missing-decision.json",
            selector_receipt_path=tmp_path / "missing-receipt.json",
        )


def test_closed_freeze_does_not_itself_authorize_future_audit(tmp_path: Path) -> None:
    authorization = {key: None for key in audit._AUTHORIZATION_KEYS}
    freeze = {key: None for key in audit._FREEZE_KEYS}
    authorization.update(
        {
            "schema": audit.ATTEMPT08_FUTURE_AUDIT_OPEN_AUTHORIZATION_SCHEMA,
            "status": "separately_authorized_after_immutable_development_pass_freeze",
            "population": audit.ATTEMPT08_FUTURE_AUDIT_POPULATION,
            "root_index_first": 200,
            "root_index_last": 249,
            "total_roots": 50,
            "search_core": audit.ATTEMPT08_FUTURE_AUDIT_CORE,
            "development_passed": True,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
    )
    freeze.update(
        {
            "schema": audit.ATTEMPT08_DEVELOPMENT_PASS_FREEZE_SCHEMA,
            "status": "development_go_frozen_without_future_audit_authorization",
            "search_core": audit.ATTEMPT08_FUTURE_AUDIT_CORE,
            "search_freeze_authorized": True,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
    )
    auth_path = tmp_path / "audit-auth.json"
    freeze_path = tmp_path / "development-pass-freeze.json"
    _write_canonical(auth_path, authorization)
    _write_canonical(freeze_path, freeze)
    with pytest.raises(ValueError, match="not separately authorized"):
        audit.load_and_validate_future_audit_open_authorization(
            auth_path,
            development_pass_freeze_path=freeze_path,
            run_dir=tmp_path / "missing-run",
            launch_authorization_path=tmp_path / "missing-launch.json",
            development_decision_path=tmp_path / "missing-decision.json",
            selector_receipt_path=tmp_path / "missing-receipt.json",
        )


def test_dormant_audit_contract_is_disjoint_from_development() -> None:
    assert audit.ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST == 200
    assert audit.ATTEMPT08_FUTURE_AUDIT_ROOT_LAST == 249
    assert audit.ATTEMPT08_FUTURE_AUDIT_ROOTS == 50
    assert audit.ATTEMPT08_FUTURE_AUDIT_ROW_SCHEMA != (
        "hu_m43_attempt08_development_shard_row_v1"
    )
    assert audit.ATTEMPT08_FUTURE_AUDIT_PROVENANCE_SCHEMA != (
        "hu_m43_attempt08_development_provenance_v1"
    )

