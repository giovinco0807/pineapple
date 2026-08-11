from __future__ import annotations

import json
from pathlib import Path

import pytest

import ofc_regular.validate_hu_m31_t3_step6c_quality as subject
from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_step6c_contract import (
    ACCEPTED_FEATURE_ENCODER_SHA256,
    ACCEPTED_NATIVE_LIBRARY_SHA256,
    CONFIRMATION_ROOT_INDICES,
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    PILOT_HAND_INDICES,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_RUN_ID,
    STEP5_CONTRACT_BYTE_SHA256,
    STEP5_CONTRACT_CANONICAL_SHA256,
    STEP5_VALIDATION_SHA256,
    STEP6B_STATUS_SHA256,
    STEP6B_VALIDATION_SHA256,
    behavior_profile_for_train_index,
    canonical_sha256,
    schedule_rows,
    train_seed_values,
)
from ofc_regular.hu_m31_t3_step6c_spot import (
    EXPECTED_IMAGE_ID,
    EXPECTED_IMAGE_NAME,
    EXPECTED_MACHINE_TYPE,
    EXPECTED_STEP6B_RECEIPT_SHA256,
    SCHEDULE_NAME,
    SOURCE_NAME,
    STARTUP_NAME,
    STEP6C_AUTHORIZATION_SCHEMA,
    STEP6C_PACKAGE_SCHEMA,
)
from ofc_regular.run_hu_m31_t3_step6b_shard import EXPECTED_MODELS
from ofc_regular.run_hu_m31_t3_step6c_shard import STEP6C_PARITY_SCHEMA, RowEvidence
from ofc_regular.state import Board


_RUN_NAME = "regular-hu-m31-step6c-quality-test-001"
_SOURCE_SHA256 = "a" * 64
_SCHEDULE_SHA256 = "b" * 64
_STARTUP_SHA256 = "c" * 64
_STEP6C_VALIDATION_SHA256 = "d" * 64
_PARITY_GOLDEN_SHA256 = "9" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    )


def _received_fixture(
    tmp_path: Path, *, shard_statuses: tuple[str, str] = ("pass", "pass")
) -> Path:
    received = tmp_path / "received"
    native = {
        "path": "native/release/libofc_hu_m3_engine.so",
        "sha256": ACCEPTED_NATIVE_LIBRARY_SHA256,
        "bytes": 101,
        "engine_version": "ofc_hu_m3_engine/0.1.0",
    }
    feature = {
        "path": "target/release/libofc_stage3_feature_encoder.so",
        "sha256": ACCEPTED_FEATURE_ENCODER_SHA256,
        "bytes": 102,
    }
    source_hashes = {
        "configs/hu_joint_policy_m31_t3_step5_contract.json": (
            STEP5_CONTRACT_BYTE_SHA256
        ),
        "configs/hu_joint_policy_m31_t3_step6b_status.json": STEP6B_STATUS_SHA256,
        "configs/hu_joint_policy_m31_t3_step6c_contract.json": (
            EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        ),
        "artifacts/step5/contract_validation.json": STEP5_VALIDATION_SHA256,
        "artifacts/step6b/canary_validation.json": STEP6B_VALIDATION_SHA256,
        "artifacts/step6b/receive_receipt.json": EXPECTED_STEP6B_RECEIPT_SHA256,
        "artifacts/step6c/contract_validation.json": _STEP6C_VALIDATION_SHA256,
        "artifacts/step6c/parity_golden.json": _PARITY_GOLDEN_SHA256,
        native["path"]: native["sha256"],
        feature["path"]: feature["sha256"],
        **EXPECTED_MODELS,
    }
    source_entries = {
        path: {
            "sha256": sha256,
            "bytes": (
                native["bytes"]
                if path == native["path"]
                else feature["bytes"] if path == feature["path"] else 1
            ),
        }
        for path, sha256 in source_hashes.items()
    }
    manifest = {
        "schema": STEP6C_PACKAGE_SCHEMA,
        "status": "packaged_local_no_gcloud",
        "run_name": _RUN_NAME,
        "source_name": SOURCE_NAME,
        "source_sha256": _SOURCE_SHA256,
        "source_bytes": 10_000,
        "schedule_name": SCHEDULE_NAME,
        "schedule_sha256": _SCHEDULE_SHA256,
        "startup_name": STARTUP_NAME,
        "startup_sha256": _STARTUP_SHA256,
        "total_shards": 2,
        "authorized_shards": [0, 1],
        "paired_hands_per_shard": 25,
        "roots_per_shard": 50,
        "pilot_hand_indices": list(PILOT_HAND_INDICES),
        "confirmation_hand_indices": [5, 16, 29, 39, 45],
        "step6c_run_id": STEP6C_RUN_ID,
        "behavior_schedule_schema": STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
        "teacher_schedule_canonical_sha256": canonical_sha256(
            schedule_rows(PILOT_HAND_INDICES)
        ),
        "step5_contract_byte_sha256": STEP5_CONTRACT_BYTE_SHA256,
        "step5_contract_canonical_sha256": STEP5_CONTRACT_CANONICAL_SHA256,
        "step5_validation_sha256": STEP5_VALIDATION_SHA256,
        "step6b_status_sha256": STEP6B_STATUS_SHA256,
        "step6b_validation_sha256": STEP6B_VALIDATION_SHA256,
        "step6b_receive_receipt_sha256": EXPECTED_STEP6B_RECEIPT_SHA256,
        "step6c_contract_byte_sha256": EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
        "step6c_contract_canonical_sha256": (EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256),
        "step6c_validation_sha256": _STEP6C_VALIDATION_SHA256,
        "native_library": native,
        "feature_encoder_library": feature,
        "models": EXPECTED_MODELS,
        "parity_golden_path": "artifacts/step6c/parity_golden.json",
        "parity_golden_sha256": _PARITY_GOLDEN_SHA256,
        "production_label_budget": subject._PRIMARY_BUDGET,
        "confirmation_budget": subject._CONFIRMATION_BUDGET,
        "source_entries": source_entries,
        "source_entry_count": len(source_entries),
        "machine_type": EXPECTED_MACHINE_TYPE,
        "image_name": EXPECTED_IMAGE_NAME,
        "image_id": EXPECTED_IMAGE_ID,
        "checkpoint_unit": "completed_paired_hand",
        "heartbeat_interval_seconds": 60,
        "resume_drill_required_each_shard": True,
        "quality_pilot_authorized": True,
        "pilot_rows_training_eligible": False,
        "production_fanout_authorized": False,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "m31_complete": False,
    }
    _write_json(received / "package_manifest.json", manifest)
    manifest_sha256 = subject._sha256(received / "package_manifest.json")
    authorization = {
        "schema": STEP6C_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": _RUN_NAME,
        "manifest_sha256": manifest_sha256,
        "source_sha256": _SOURCE_SHA256,
        "schedule_sha256": _SCHEDULE_SHA256,
        "startup_sha256": _STARTUP_SHA256,
        "step5_contract_canonical_sha256": STEP5_CONTRACT_CANONICAL_SHA256,
        "step6b_validation_sha256": STEP6B_VALIDATION_SHA256,
        "step6c_contract_canonical_sha256": (EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256),
        "dry_run_receipt_sha256": "e" * 64,
        "authorized_shards": [0, 1],
        "spot_authorized": True,
        "quality_pilot_only": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "root_execution_started": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "authorized_unix_seconds": 1.0,
    }
    _write_json(received / "launch_authorization.json", authorization)
    authorization_sha256 = subject._sha256(received / "launch_authorization.json")
    _write_json(
        received / "receive_receipt.json",
        {
            "schema": subject.STEP6C_RECEIVE_SCHEMA,
            "status": "pass",
            "run_name": _RUN_NAME,
            "source_sha256": _SOURCE_SHA256,
            "manifest_sha256": manifest_sha256,
            "schedule_sha256": _SCHEDULE_SHA256,
            "authorization_sha256": authorization_sha256,
            "step5_contract_canonical_sha256": STEP5_CONTRACT_CANONICAL_SHA256,
            "step6b_validation_sha256": STEP6B_VALIDATION_SHA256,
            "step6c_contract_canonical_sha256": (
                EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
            ),
            "native_library_sha256": ACCEPTED_NATIVE_LIBRARY_SHA256,
            "feature_encoder_sha256": ACCEPTED_FEATURE_ENCODER_SHA256,
            "all_shards_received": True,
            "quality_result_pending_validation": True,
            "training_eligible": False,
            "production_fanout_authorized": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "m31_complete": False,
            "per_shard": {
                f"{shard:03d}": {
                    "done_sha256": f"{shard + 11:064x}",
                    "summary_sha256": f"{shard + 1:064x}",
                    "task_count": 25,
                    "root_task_count": 25,
                    "resumed_task_count": 1,
                    "shard_status": shard_statuses[shard],
                }
                for shard in range(2)
            },
        },
    )
    for shard in range(2):
        (received / "shards" / f"shard-{shard:03d}").mkdir(parents=True)
    return received


def _key(domain: int, root: int, sample: int) -> str:
    return f"{domain:02x}{root:04x}{sample:04x}".ljust(64, "0")


def _parity_payload(*, manifest_sha256: str) -> dict[str, object]:
    gates = {
        "both_seats": True,
        "portable_action_values_exact": True,
        "linux_native_hash_bound": True,
        "manifest_source_and_golden_bound": True,
        "first_production_budget_speed_smoke_within_180_seconds": True,
        "second_production_budget_speed_smoke_within_6_seconds": True,
        "no_profile_or_current_resolution": True,
    }
    return {
        "schema": STEP6C_PARITY_SCHEMA,
        "status": "pass",
        "source_package_sha256": _SOURCE_SHA256,
        "manifest_sha256": manifest_sha256,
        "parity_golden_sha256": _PARITY_GOLDEN_SHA256,
        "native_library_sha256": ACCEPTED_NATIVE_LIBRARY_SHA256,
        "rows": [
            {
                "seat": seat,
                "observation_fingerprint": f"{index + 1:064x}",
                "expected_portable_sha256": f"{index + 3:064x}",
                "observed_portable_sha256": f"{index + 3:064x}",
                "wall_seconds": 1.0,
                "match": True,
            }
            for index, seat in enumerate(("first", "second"))
        ],
        "gates": gates,
        "all_gates_passed": True,
        "current_profile_changed": False,
        "teacher_generation_started": False,
        "production_fanout_authorized": False,
    }


def _validated_shard(
    shard: int,
    *,
    regrets: dict[int, float],
    overlap: bool = False,
    resumed: int = 1,
    status: str = "pass",
    run_name: str = _RUN_NAME,
    source_sha256: str = _SOURCE_SHA256,
    manifest_sha256: str,
) -> subject.ValidatedShard:
    start = shard * 25
    hands = tuple(range(start, start + 25))
    profiles = tuple(behavior_profile_for_train_index(index) for index in hands)
    evidence = []
    fingerprints = []
    seats = []
    for index in hands:
        for offset, seat in enumerate(("first", "second")):
            root = index * 2 + offset
            candidate = frozenset(_key(1, root, sample) for sample in range(8))
            evaluation = frozenset(_key(2, root, sample) for sample in range(32))
            confirmation = (
                frozenset(_key(3, root, sample) for sample in range(128))
                if root in CONFIRMATION_ROOT_INDICES
                else frozenset()
            )
            if overlap and root == 0:
                evaluation = frozenset({*evaluation, next(iter(candidate))})
            evidence.append(
                RowEvidence(
                    candidate_keys=candidate,
                    evaluation_keys=evaluation,
                    confirmation_keys=confirmation,
                    confirmation_regret=regrets.get(root),
                    primary_wall_seconds=1.0 if seat == "first" else 0.1,
                    confirmation_wall_seconds=(
                        2.0 if root in CONFIRMATION_ROOT_INDICES else None
                    ),
                    legal_action_count=21,
                    seat=seat,
                )
            )
            fingerprints.append(f"{root:064x}")
            seats.append(seat)
    return subject.ValidatedShard(
        shard=shard,
        status=status,
        run_name=run_name,
        source_sha256=source_sha256,
        manifest_sha256=manifest_sha256,
        hand_indices=hands,
        profiles=profiles,
        fingerprints=tuple(fingerprints),
        seats=tuple(seats),
        evidence=tuple(evidence),
        peak_rss_bytes=128 * 1024 * 1024,
        resumed_task_count=resumed,
        summary_sha256=f"{shard + 1:064x}",
    )


def _patch_shards(
    monkeypatch: pytest.MonkeyPatch,
    *,
    received: Path,
    regret_values: list[float],
    overlap: bool = False,
    resumed: int = 1,
    statuses: tuple[str, str] = ("pass", "pass"),
    source_sha256: str = _SOURCE_SHA256,
) -> None:
    regrets = dict(zip(CONFIRMATION_ROOT_INDICES, regret_values, strict=True))
    receipt = json.loads(
        (received / "receive_receipt.json").read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        subject, "load_model_bundle", lambda *_args, **_kwargs: object()
    )

    def fake_validate(
        _directory: Path,
        shard: int,
        *,
        bundle: object,
        expected_run_name: str,
        source_package_sha256: str,
        manifest_sha256: str,
        parity_golden_sha256: str,
    ) -> subject.ValidatedShard:
        assert bundle is not None
        assert expected_run_name == receipt["run_name"]
        assert source_package_sha256 == receipt["source_sha256"]
        assert manifest_sha256 == receipt["manifest_sha256"]
        assert parity_golden_sha256 == _PARITY_GOLDEN_SHA256
        return _validated_shard(
            shard,
            regrets=regrets,
            overlap=overlap,
            resumed=resumed,
            status=statuses[shard],
            source_sha256=source_sha256,
            manifest_sha256=receipt["manifest_sha256"],
        )

    monkeypatch.setattr(subject, "_validate_shard", fake_validate)


def test_quality_gate_passes_exact_balanced_rng_grid(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    _patch_shards(monkeypatch, received=received, regret_values=[0.5] * 10)
    result = subject.validate_quality(received_dir=received)
    assert result["status"] == "pass"
    assert result["integrity"]["paired_hands"] == 50
    assert result["integrity"]["roots"] == 100
    assert result["integrity"]["candidate_rng_keys"] == 800
    assert result["integrity"]["evaluation_rng_keys"] == 3200
    assert result["integrity"]["confirmation_rng_keys"] == 1280
    assert result["confirmation_quality"]["root_indices"] == list(
        CONFIRMATION_ROOT_INDICES
    )
    assert all(
        count == 10 for count in result["integrity"]["profile_hand_counts"].values()
    )
    assert result["training_eligible"] is False
    assert result["production_fanout_authorized"] is False


def test_nearest_rank_ten_root_p95_and_p99_are_the_maximum(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 9 + [3.0])
    result = subject.validate_quality(received_dir=received)
    assert result["confirmation_quality"]["p95"] == 3.0
    assert result["confirmation_quality"]["p99"] == 3.0
    assert result["status"] == "pass"

    received2 = _received_fixture(tmp_path / "second")
    _patch_shards(monkeypatch, received=received2, regret_values=[0.0] * 9 + [3.0001])
    failed = subject.validate_quality(received_dir=received2)
    assert failed["status"] == "no_go"
    assert failed["gates"]["confirmation_regret_p95_at_most_3"] is False


def test_rng_overlap_and_missing_resume_are_no_go(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    receipt_path = received / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for row in receipt["per_shard"].values():
        row["resumed_task_count"] = 0
    _write_json(receipt_path, receipt)
    _patch_shards(
        monkeypatch,
        received=received,
        regret_values=[0.0] * 10,
        overlap=True,
        resumed=0,
    )
    result = subject.validate_quality(received_dir=received)
    assert result["status"] == "no_go"
    assert result["gates"]["candidate_evaluation_confirmation_overlap_zero"] is False
    assert result["gates"]["every_shard_resumed_at_least_one_task"] is False


def test_quality_output_is_atomic_write_once(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    _patch_shards(monkeypatch, received=received, regret_values=[0.25] * 10)
    output = tmp_path / "quality.json"
    result = subject.validate_quality(received_dir=received, output_path=output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == result["status"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.validate_quality(received_dir=received, output_path=output)


@pytest.mark.parametrize(
    ("status", "failed_gate", "all_gates_passed"),
    [
        ("pass", "peak_rss_within_1_gib", False),
        ("no_go", None, True),
    ],
)
def test_shard_summary_status_must_match_exact_gate_outcome(
    status: str, failed_gate: str | None, all_gates_passed: bool
):
    gates = {key: True for key in subject.STEP6C_SHARD_GATE_KEYS}
    if failed_gate is not None:
        gates[failed_gate] = False
    summary = {
        "schema": subject.STEP6C_SUMMARY_SCHEMA,
        "status": status,
        "shard": 0,
        "hand_indices": list(range(25)),
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "completed_tasks": 25,
        "pending_tasks": 0,
        "resumed_task_count": 1,
        "contract": {
            "primary_budget": subject._PRIMARY_BUDGET,
            "confirmation_budget": subject._CONFIRMATION_BUDGET,
            "run_id": STEP6C_RUN_ID,
        },
        "integrity": {},
        "confirmation": {"quality_thresholds_applied": False},
        "performance": {},
        "gates": gates,
        "all_gates_passed": all_gates_passed,
        "teacher_value_status": "diagnostic_not_match_EV",
        "quality_pilot_authorized": True,
        "training_eligible": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }
    with pytest.raises(ValueError, match="summary boundary changed"):
        subject._validate_summary_envelope(
            summary, shard=0, hand_indices=tuple(range(25))
        )


def test_shard_peak_rss_is_recomputed_from_tasks_and_bound_to_summary():
    summary = {"performance": {"peak_process_rss_bytes": 300}}
    assert subject._validated_peak_rss([100, 300, 200], summary) == 300

    summary["performance"]["peak_process_rss_bytes"] = 200
    with pytest.raises(ValueError, match="peak RSS summary changed"):
        subject._validated_peak_rss([100, 300, 200], summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("all_shards_received", False),
        ("quality_result_pending_validation", False),
        ("training_eligible", True),
        ("production_fanout_authorized", True),
    ],
)
def test_receive_receipt_mutations_fail_closed(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: object,
):
    received = _received_fixture(tmp_path)
    receipt_path = received / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    _write_json(receipt_path, receipt)
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 10)
    with pytest.raises(ValueError, match="receive receipt boundary changed"):
        subject.validate_quality(received_dir=received)


@pytest.mark.parametrize(
    "filename", ["package_manifest.json", "launch_authorization.json"]
)
def test_missing_received_provenance_artifact_fails_closed(
    monkeypatch, tmp_path: Path, filename: str
):
    received = _received_fixture(tmp_path)
    (received / filename).unlink()
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 10)
    with pytest.raises(ValueError, match="missing or unsafe"):
        subject.validate_quality(received_dir=received)


def test_tampered_package_manifest_anchor_fails_closed(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    manifest_path = received / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["step6b_validation_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    receipt_path = received / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = subject._sha256(manifest_path)
    _write_json(receipt_path, receipt)
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 10)
    with pytest.raises(ValueError, match="package manifest provenance changed"):
        subject.validate_quality(received_dir=received)


def test_consistently_rehashed_manifest_model_tamper_fails_closed(
    monkeypatch, tmp_path: Path
):
    received = _received_fixture(tmp_path)
    manifest_path = received / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = {"models/unbound.pkl": "0" * 64}
    _write_json(manifest_path, manifest)
    manifest_sha256 = subject._sha256(manifest_path)

    authorization_path = received / "launch_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["manifest_sha256"] = manifest_sha256
    _write_json(authorization_path, authorization)

    receipt_path = received / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = manifest_sha256
    receipt["authorization_sha256"] = subject._sha256(authorization_path)
    _write_json(receipt_path, receipt)
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 10)
    with pytest.raises(ValueError, match="package manifest provenance changed"):
        subject.validate_quality(received_dir=received)


def test_per_shard_receipt_unknown_field_fails_closed(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    receipt_path = received / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["per_shard"]["000"]["unbound"] = True
    _write_json(receipt_path, receipt)
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 10)
    with pytest.raises(ValueError, match="receive receipt boundary changed"):
        subject.validate_quality(received_dir=received)


def test_tampered_launch_authorization_anchor_fails_closed(monkeypatch, tmp_path: Path):
    received = _received_fixture(tmp_path)
    authorization_path = received / "launch_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["step6c_contract_canonical_sha256"] = "0" * 64
    _write_json(authorization_path, authorization)
    receipt_path = received / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["authorization_sha256"] = subject._sha256(authorization_path)
    _write_json(receipt_path, receipt)
    _patch_shards(monkeypatch, received=received, regret_values=[0.0] * 10)
    with pytest.raises(ValueError, match="launch authorization provenance changed"):
        subject.validate_quality(received_dir=received)


def test_shard_summary_provenance_must_match_receive_receipt(
    monkeypatch, tmp_path: Path
):
    received = _received_fixture(tmp_path)
    _patch_shards(
        monkeypatch,
        received=received,
        regret_values=[0.0] * 10,
        source_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="per-shard binding changed"):
        subject.validate_quality(received_dir=received)


def test_shard_no_go_cannot_be_promoted_by_merged_quality_gate(
    monkeypatch, tmp_path: Path
):
    statuses = ("pass", "no_go")
    received = _received_fixture(tmp_path, shard_statuses=statuses)
    _patch_shards(
        monkeypatch,
        received=received,
        regret_values=[0.0] * 10,
        statuses=statuses,
    )
    result = subject.validate_quality(received_dir=received)
    assert result["status"] == "no_go"
    assert result["gates"]["all_shard_summaries_pass"] is False
    assert [row["status"] for row in result["shards"]] == ["pass", "no_go"]


def test_shard_parity_is_revalidated_and_hash_bound(tmp_path: Path):
    shard_dir = tmp_path / "shard-000"
    parity_path = shard_dir / "parity.json"
    manifest_sha256 = "f" * 64
    _write_json(parity_path, _parity_payload(manifest_sha256=manifest_sha256))
    summary = {
        "shard": 0,
        "parity_report_sha256": subject._sha256(parity_path),
    }
    subject._validate_shard_parity(
        shard_dir,
        summary,
        source_package_sha256=_SOURCE_SHA256,
        manifest_sha256=manifest_sha256,
        parity_golden_sha256=_PARITY_GOLDEN_SHA256,
    )

    summary["parity_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="summary parity binding changed"):
        subject._validate_shard_parity(
            shard_dir,
            summary,
            source_package_sha256=_SOURCE_SHA256,
            manifest_sha256=manifest_sha256,
            parity_golden_sha256=_PARITY_GOLDEN_SHA256,
        )


def test_shard_parity_provenance_tamper_fails_closed(tmp_path: Path):
    shard_dir = tmp_path / "shard-000"
    parity_path = shard_dir / "parity.json"
    manifest_sha256 = "f" * 64
    parity = _parity_payload(manifest_sha256=manifest_sha256)
    parity["source_package_sha256"] = "0" * 64
    _write_json(parity_path, parity)
    with pytest.raises(ValueError, match="Linux parity provenance changed"):
        subject._validate_shard_parity(
            shard_dir,
            {"shard": 0, "parity_report_sha256": subject._sha256(parity_path)},
            source_package_sha256=_SOURCE_SHA256,
            manifest_sha256=manifest_sha256,
            parity_golden_sha256=_PARITY_GOLDEN_SHA256,
        )


def _t3_observation(*, seat: str, replacement: bool = False) -> ActorObservation:
    cards = create_deck(shuffle=False)
    if seat == "first":
        hero = Board.from_rows(cards[0:2], cards[2:5], cards[5:9])
        opponent = Board.from_rows(cards[9:11], cards[11:14], cards[14:18])
        dealt = (cards[23] if replacement else cards[18], cards[19], cards[20])
        discards = tuple(cards[21:23])
    else:
        hero = Board.from_rows(cards[0:2], cards[2:5], cards[5:9])
        opponent = Board.from_rows(cards[9:12], cards[12:16], cards[16:20])
        dealt = (cards[25] if replacement else cards[20], cards[21], cards[22])
        discards = tuple(cards[23:25])
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=discards,
        seat=seat,
        street="T3",
        to_act_order=seat,
    )


def test_quality_rejects_valid_but_seed_inconsistent_replacement_root(
    monkeypatch, tmp_path: Path
):
    shard_dir = tmp_path / "shard-000"
    manifest_sha256 = "f" * 64
    _write_json(
        shard_dir / "summary.json",
        {
            "run_name": _RUN_NAME,
            "source_package_sha256": _SOURCE_SHA256,
            "manifest_sha256": manifest_sha256,
            "status": "pass",
        },
    )
    hand_indices = tuple(range(25))
    for index in hand_indices:
        _write_json(shard_dir / "roots" / f"hand_{index:03d}.json", {})
        _write_json(shard_dir / "tasks" / f"hand_{index:03d}.json", {})

    spec = subject.ShardSpec(_RUN_NAME, 0, hand_indices)
    seeds = train_seed_values(0)
    replacement = (
        _t3_observation(seat="first", replacement=True),
        _t3_observation(seat="second", replacement=True),
    )
    root = {
        "schema": subject.STEP6C_ROOT_TASK_SCHEMA,
        "contract_digest": subject._root_contract_digest(
            manifest_sha256=manifest_sha256,
            spec=spec,
            global_hand_index=0,
        ),
        "global_hand_index": 0,
        "profile": behavior_profile_for_train_index(0),
        "seeds": seeds,
        "confirmation_required": False,
        "observations": [
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation": observation.to_dict(),
            }
            for observation in replacement
        ],
        "current_profile_resolved": False,
        "opponent_private_discards_used": False,
    }
    _write_json(shard_dir / "roots" / "hand_000.json", root)
    expected = (
        _t3_observation(seat="first"),
        _t3_observation(seat="second"),
    )
    monkeypatch.setattr(subject, "_validate_summary_envelope", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_validate_shard_parity", lambda *_a, **_k: None)
    monkeypatch.setattr(
        subject, "generate_behavior_t3_roots", lambda **_kwargs: expected
    )
    with pytest.raises(ValueError, match="root observation changed"):
        subject._validate_shard(
            shard_dir,
            0,
            bundle=object(),
            expected_run_name=_RUN_NAME,
            source_package_sha256=_SOURCE_SHA256,
            manifest_sha256=manifest_sha256,
            parity_golden_sha256=_PARITY_GOLDEN_SHA256,
        )


def test_contract_pilot_indices_are_the_complete_quality_grid():
    assert tuple(PILOT_HAND_INDICES) == tuple(range(50))
    assert tuple(CONFIRMATION_ROOT_INDICES) == (
        10,
        11,
        32,
        33,
        58,
        59,
        78,
        79,
        90,
        91,
    )
