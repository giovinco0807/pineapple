from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_performance_lock_v4_portable_receipt_v1 as subject,
)


_CANDIDATE_SHA = "1" * 64
_REFERENCE_SHA = "2" * 64
_PROFILE_SHA = "3" * 64
_CONTRACT_SHA = "4" * 64
_SEED_SHA = "5" * 64
_ACCEPTED_SHA = "6" * 64
_LIFECYCLE_SHA = "7" * 64
_MERGE_VIEW_SHA = "8" * 64
_PURE_SHA = "9" * 64
_TRANSPORT_SHA = "a" * 64


def _patch_structural_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_performance_lock_v4_plan",
        lambda value: deepcopy(value),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_materialization_receipt",
        lambda value: deepcopy(value),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_root_seal",
        lambda value: deepcopy(value),
    )
    monkeypatch.setattr(
        subject.pure_v4,
        "validate_candidate02_performance_lock_v4_value",
        lambda value, *, replay_sources: (
            deepcopy(value)
            if replay_sources is False
            else pytest.fail("portable validation requested source replay")
        ),
    )
    monkeypatch.setattr(
        subject.wave_v2,
        "validate_wave_plan",
        lambda value: deepcopy(value),
    )
    monkeypatch.setattr(
        subject.wave_v2,
        "validate_attempt_ledger",
        lambda _plan, value: deepcopy(value),
    )
    monkeypatch.setattr(
        subject.wave_bridge,
        "validate_validated_lifecycle_chain",
        lambda *, wave_plan, attempt_ledger, value: deepcopy(value),
    )


def _patch_scientific_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (
        ("PINNED_CANDIDATE_LIBRARY_SHA256", _CANDIDATE_SHA),
        ("PINNED_REFERENCE_LIBRARY_SHA256", _REFERENCE_SHA),
        ("PINNED_PROFILE_REGISTRY_SHA256", _PROFILE_SHA),
        ("PINNED_RUN_CONTRACT_DIGEST", _CONTRACT_SHA),
        ("PINNED_SEED_SET_SHA256", _SEED_SHA),
        ("PINNED_ACCEPTED_SNAPSHOT_SHA256", _ACCEPTED_SHA),
        ("PINNED_LIFECYCLE_CHAIN_SHA256", _LIFECYCLE_SHA),
        ("PINNED_MERGE_VIEW_MANIFEST_SHA256", _MERGE_VIEW_SHA),
        ("PINNED_PURE_V4_MERGE_SHA256", _PURE_SHA),
        ("PINNED_TRANSPORT_AUDIT_SHA256", _TRANSPORT_SHA),
    ):
        monkeypatch.setattr(subject, name, value)


def _receipt() -> dict:
    receipt = {
        key: None
        for key in subject.production._RECEIPT_KEYS  # type: ignore[attr-defined]
    }
    receipt.update(
        {
            "schema": subject.production.RECEIPT_SCHEMA,
            "status": "qualified",
            "decision": subject.production.QUALIFIED_DECISION,
            "source_paths": {
                key: rf"C:\retired-evidence\{key}.json"
                for key in subject.production._SOURCE_PATH_KEYS  # type: ignore[attr-defined]
            },
            "source_file_sha256": {
                "current_profile_registry": _PROFILE_SHA,
            },
            "performance_lock_plan": {
                "run_contract": {
                    "candidate_library_sha256": _CANDIDATE_SHA,
                    "reference_library_sha256": _REFERENCE_SHA,
                },
                "run_contract_digest": _CONTRACT_SHA,
                "seed_contract": {"seed_set_sha256": _SEED_SHA},
                "current_profile_registry": {"sha256": _PROFILE_SHA},
            },
            "materialization_receipt": {"schema": "materialization"},
            "root_seal": {"schema": "seal"},
            "pure_v4_merge": {"summary_sha256": _PURE_SHA},
            "wave_plan": {"schema": "wave"},
            "attempt_ledger": {"schema": "ledger"},
            "accepted_results_snapshot": {
                "accepted_root": r"D:\retired-evidence\accepted",
                "snapshot_sha256": _ACCEPTED_SHA,
                "accepted_job_count": 20,
                "accepted_object_count": 440,
                "observed_inventory": {
                    "object_count": 440,
                    "objects": [{"index": index} for index in range(440)],
                },
            },
            "validated_lifecycle_chain": {"chain_sha256": _LIFECYCLE_SHA},
            "merge_view_manifest": {"manifest_sha256": _MERGE_VIEW_SHA},
            "prelaunch_root_chronology": {},
            "one_shot_attempt_audit": {},
            "process_isolation_audit": {},
            "transport_audit": {
                "audit_sha256": _TRANSPORT_SHA,
                "accepted_snapshot_sha256": _ACCEPTED_SHA,
                "validated_lifecycle_chain_sha256": _LIFECYCLE_SHA,
                "merge_view_manifest_sha256": _MERGE_VIEW_SHA,
                "pure_v4_merge_sha256": _PURE_SHA,
                "run_contract_digest": _CONTRACT_SHA,
                "accepted_job_count": 20,
                "accepted_object_count": 440,
                "paired_hand_count": 100,
                "root_count": 200,
                "transport_lineage_validated": True,
                "current_profile_changed": False,
            },
            "all_gates_passed": True,
            "performance_lock_finalized": True,
            "one_shot_lock_consumed": True,
            "transport_lineage_validated": True,
            "performance_lock_qualified": True,
            "candidate_finalized_no_go": False,
            "quality_pilot_authorized": True,
            "rerun_authorized": False,
            "reseed_authorized": False,
            "artifact_fanout_authorized": False,
            "training_eligible": False,
            "training_authorized": False,
            "promotion_evidence": False,
            "promotion_authorized": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
            "bridge_cloud_network_invoked": False,
        }
    )
    body = deepcopy(receipt)
    body.pop("receipt_sha256")
    receipt["receipt_sha256"] = subject.canonical_sha256(body)
    return receipt


def _write_and_pin(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    receipt: dict,
) -> None:
    raw = subject.canonical_bytes(receipt)
    path.write_bytes(raw)
    monkeypatch.setattr(
        subject,
        "PINNED_RECEIPT_SHA256",
        receipt["receipt_sha256"],
    )
    monkeypatch.setattr(
        subject,
        "PINNED_RECEIPT_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )


@pytest.fixture
def portable_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, dict]:
    _patch_structural_validators(monkeypatch)
    _patch_scientific_pins(monkeypatch)
    receipt = _receipt()
    path = tmp_path / "performance_receipt.json"
    _write_and_pin(monkeypatch, path, receipt)
    return path, receipt


def test_exact_pin_validates_without_dereferencing_embedded_source_paths(
    monkeypatch: pytest.MonkeyPatch,
    portable_receipt: tuple[Path, dict],
):
    path, receipt = portable_receipt
    original_exists = Path.exists

    def reject_embedded_path(target: Path) -> bool:
        if "retired-evidence" in str(target):
            pytest.fail("portable validator dereferenced an embedded source path")
        return original_exists(target)

    monkeypatch.setattr(Path, "exists", reject_embedded_path)
    assert subject.validate_pinned_portable_receipt(
        path,
        expected_profile_sha256=_PROFILE_SHA,
    ) == receipt


def test_unavailable_paths_select_exact_pin_without_full_replay(
    monkeypatch: pytest.MonkeyPatch,
    portable_receipt: tuple[Path, dict],
):
    path, receipt = portable_receipt
    monkeypatch.setattr(
        subject,
        "source_replay_paths_natively_available",
        lambda _receipt: False,
    )
    monkeypatch.setattr(
        subject.production,
        "validate_performance_lock_v4_production_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "unavailable native paths must not request full replay"
        ),
    )
    loaded, audit = subject.load_preferred_or_pinned_receipt(
        path,
        expected_profile_sha256=_PROFILE_SHA,
    )
    assert loaded == receipt
    assert audit["mode"] == "exact_pinned_portable_no_source_path_dereference"
    assert audit["source_paths_dereferenced"] is False
    assert audit["new_seed_authorized"] is False
    assert audit["replacement_binary_authorized"] is False
    assert audit["current_profile_changed"] is False


def test_available_paths_require_full_replay_and_never_fallback(
    monkeypatch: pytest.MonkeyPatch,
    portable_receipt: tuple[Path, dict],
):
    path, _receipt = portable_receipt
    monkeypatch.setattr(
        subject,
        "source_replay_paths_natively_available",
        lambda _value: True,
    )

    def reject_full_replay(*_args, **_kwargs):
        raise PermissionError("source evidence replay failed")

    monkeypatch.setattr(
        subject.production,
        "validate_performance_lock_v4_production_receipt",
        reject_full_replay,
    )
    monkeypatch.setattr(
        subject,
        "validate_pinned_portable_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "full replay failure must not downgrade to the portable pin"
        ),
    )
    with pytest.raises(PermissionError, match="source evidence replay failed"):
        subject.load_preferred_or_pinned_receipt(
            path,
            expected_profile_sha256=_PROFILE_SHA,
        )


def test_canonical_byte_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    portable_receipt: tuple[Path, dict],
):
    path, receipt = portable_receipt
    changed = deepcopy(receipt)
    changed["promotion_authorized"] = True
    path.write_bytes(subject.canonical_bytes(changed))
    with pytest.raises(PermissionError, match="exact portable pin"):
        subject.validate_pinned_portable_receipt(
            path,
            expected_profile_sha256=_PROFILE_SHA,
        )


def test_wrong_scientific_identity_fails_even_if_bytes_are_re_pinned(
    monkeypatch: pytest.MonkeyPatch,
    portable_receipt: tuple[Path, dict],
):
    path, receipt = portable_receipt
    changed = deepcopy(receipt)
    changed["performance_lock_plan"]["run_contract"][
        "candidate_library_sha256"
    ] = "f" * 64
    body = deepcopy(changed)
    body.pop("receipt_sha256")
    changed["receipt_sha256"] = subject.canonical_sha256(body)
    _write_and_pin(monkeypatch, path, changed)
    with pytest.raises(PermissionError, match="scientific identity changed"):
        subject.validate_pinned_portable_receipt(
            path,
            expected_profile_sha256=_PROFILE_SHA,
        )


def test_wrong_profile_pin_fails_before_authorization(
    portable_receipt: tuple[Path, dict],
):
    path, _receipt = portable_receipt
    with pytest.raises(PermissionError, match="profile pin changed"):
        subject.validate_pinned_portable_receipt(
            path,
            expected_profile_sha256="f" * 64,
        )
