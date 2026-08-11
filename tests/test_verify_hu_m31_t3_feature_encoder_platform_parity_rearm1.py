from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm1_open as rearm_open,
)
from ofc_regular import hu_m31_t3_step6d_performance_lock_spot_v1 as spot_v1
from ofc_regular import (
    verify_hu_m31_t3_feature_encoder_platform_parity_rearm1 as subject,
)
from scripts import verify_hu_m31_t3_feature_encoder_platform_parity as probe_v1


REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_SOURCE = REPO_ROOT / "rust/ofc_stage3_feature_encoder/src/lib.rs"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"
AI_PROFILES = REPO_ROOT / "src/ofc_regular/ai_profiles.py"
PINNED_PROBE_SHA = "a1905bc951ac54680e4993a5f63cc85a2d2579f0f962529fd3a0c8968bfcd921"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(probe_v1.canonical_bytes(value))


def _probe(platform: str) -> dict[str, Any]:
    suffix = ".dll" if platform == "windows" else ".so"
    digest = (
        spot_v1.EXPECTED_WINDOWS_FEATURE_ENCODER_SHA256
        if platform == "windows"
        else rearm_open.FEATURE_ENCODER_SHA256
    )
    return {
        "schema": probe_v1.PROBE_SCHEMA,
        "status": "fixed_512_row_feature_encoder_probe_complete",
        "platform_label": platform,
        "generator": probe_v1._generator_record(),
        "build_inputs": probe_v1._default_build_inputs(),
        "library": {
            "path": f"/frozen/feature_encoder{suffix}",
            "bytes": 199_680 if platform == "windows" else 489_144,
            "sha256": digest,
        },
        "input_bytes": probe_v1.EXPECTED_INPUT_BYTES,
        "input_sha256": probe_v1.EXPECTED_INPUT_SHA256,
        "encoder_status": 0,
        "feature_dim": probe_v1.FEATURE_DIM,
        "row_count": probe_v1.ROW_COUNT,
        "output_encoding": probe_v1.OUTPUT_ENCODING,
        "output_bytes": probe_v1.EXPECTED_OUTPUT_BYTES,
        "output_sha256": spot_v1.EXPECTED_MATERIALIZER_PARITY_OUTPUT_SHA256,
    }


def _chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    claim = {key: {} for key in rearm_open._CLAIM_KEYS}
    claim.update(
        {
            "schema": rearm_open.CLAIM_SCHEMA,
            "status": rearm_open.CLAIM_STATUS,
            "scope": "candidate02_performance_lock_rearm1_fresh_roots_only",
            "accepted_binaries": {
                "candidate": {},
                "reference": {},
                "feature_encoder": {
                    "path": "/frozen/feature_encoder.so",
                    "bytes": 489_144,
                    "sha256": rearm_open.FEATURE_ENCODER_SHA256,
                },
            },
            "ai_profiles_current": probe_v1._file_record(
                AI_PROFILES, "ai_profiles"
            ),
            "rearm_guards": {
                "new_global_claim": True,
                "claim_before_new_root_path_touch": True,
                "crash_consumes_claim": True,
                "deterministic_exact_identity_resume_only": True,
                "fresh_700_series_seed_schedule": True,
                "old_v1_attempt1_used": False,
                "old_v1_attempt1_reused": False,
                "old_v1_root_reused": False,
                "old_v1_package_reused": False,
            },
            "restrictions": {
                field: False
                for field in (
                    "timing_used_for_root_selection",
                    "q_used_for_root_selection",
                    "ev_used_for_root_selection",
                    "old_v1_attempt1_allowed",
                    "old_v1_root_reuse_allowed",
                    "alternate_seed_allowed",
                    "reseed_allowed",
                    "post_claim_reseed_allowed",
                    "cloud_authorized",
                    "training_authorized",
                    "quality_authorized",
                    "promotion_authorized",
                    "current_profile_resolution_allowed",
                    "runtime_activation_allowed",
                    "opponent_private_discards_allowed",
                )
            },
        }
    )
    hashes = [
        hashlib.sha256(f"rearm-root-{index}".encode("ascii")).hexdigest()
        for index in range(100)
    ]
    materialization = {key: None for key in rearm_open._MATERIALIZATION_KEYS}
    materialization.update(
        {
            "schema": rearm_open.MATERIALIZATION_SCHEMA,
            "status": rearm_open.MATERIALIZATION_STATUS,
            "global_claim_sha256": probe_v1.canonical_sha256(claim),
            "plan_sha256": "3" * 64,
            "run_contract_digest": "4" * 64,
            "hand_indices": list(range(100)),
            "root_count": 100,
            "root_artifact_sha256": hashes,
            "aggregate_root_sha256": probe_v1.canonical_sha256(hashes),
            "same_identity_resume_only": True,
            "fresh_recovery_seed_schedule": True,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
            "reseeded": False,
            "training_eligible": False,
            "current_profile_changed": False,
        }
    )
    seal = {key: None for key in rearm_open._SEAL_KEYS}
    seal.update(
        {
            "schema": rearm_open.SEAL_SCHEMA,
            "status": rearm_open.SEAL_STATUS,
            "global_claim_sha256": probe_v1.canonical_sha256(claim),
            "materialization_sha256": probe_v1.canonical_sha256(materialization),
            "plan_sha256": materialization["plan_sha256"],
            "run_contract_digest": materialization["run_contract_digest"],
            "hand_indices": list(range(100)),
            "root_count": 100,
            "observation_count": 200,
            "seat_counts": {"first": 100, "second": 100},
            "root_artifact_sha256": hashes,
            "aggregate_root_sha256": materialization["aggregate_root_sha256"],
            "root_artifact_unique": True,
            "observation_fingerprint_unique": True,
            "visibility": {
                "opponent_private_discards_used": False,
                "current_profile_resolved": False,
            },
            "selection_inputs": {
                "all_100_preregistered_hands_used": True,
                "timing_used": False,
                "q_used": False,
                "ev_used": False,
            },
            "old_performance_lock_comparison": {
                "old_v1_global_claim_sha256": (
                    rearm_open.OLD_V1_GLOBAL_CLAIM_SHA256
                ),
                "old_v1_seal_sha256": rearm_open.OLD_V1_SEAL_SHA256,
                "old_v1_root_count": 100,
                "rearm1_fingerprint_overlap_count": 0,
                "rearm1_root_hash_overlap_count": 0,
                "rearm1_seed_overlap_count": 0,
                "old_v1_attempt1_reused": False,
                "old_v1_root_reused": False,
            },
            "rearm_guards": {
                "incident_receipt_sha256": (
                    rearm_open.STARTUP_FAILURE_RECEIPT_SHA256
                ),
                "fresh_700_series_seed_schedule": True,
                "same_identity_resume_only": True,
                "old_v1_attempt1_reused": False,
                "old_v1_root_reused": False,
                "post_claim_reseeded": False,
            },
            "training_eligible": False,
            "quality_evidence": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
        }
    )
    return claim, materialization, seal


def _fixture(tmp_path: Path) -> dict[str, Path]:
    claim, materialization, seal = _chain()
    paths = {
        "windows_probe": tmp_path / probe_v1.WINDOWS_PROBE_NAME,
        "linux_probe": tmp_path / probe_v1.LINUX_PROBE_NAME,
        "global_claim": tmp_path / "GLOBAL_PERFORMANCE_LOCK_REARM1_CLAIM.json",
        "materialization": tmp_path / "materialization.json",
        "seal": tmp_path / "seal.json",
        "output": tmp_path / probe_v1.PARITY_RECEIPT_NAME,
    }
    for path, value in (
        (paths["windows_probe"], _probe("windows")),
        (paths["linux_probe"], _probe("linux")),
        (paths["global_claim"], claim),
        (paths["materialization"], materialization),
        (paths["seal"], seal),
    ):
        _write(path, value)
    return paths


def test_rearm1_wrapper_writes_base_compatible_receipt_and_restores_v1(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    original = probe_v1._validate_lock_chain
    receipt = subject.write_comparison_receipt(
        windows_probe_path=paths["windows_probe"],
        linux_probe_path=paths["linux_probe"],
        global_claim_path=paths["global_claim"],
        materialization_path=paths["materialization"],
        seal_path=paths["seal"],
        rust_source_path=RUST_SOURCE,
        cargo_lock_path=CARGO_LOCK,
        ai_profiles_path=AI_PROFILES,
        output_path=paths["output"],
    )
    assert probe_v1._validate_lock_chain is original
    assert receipt["schema"] == probe_v1.RECEIPT_SCHEMA
    assert receipt["generator"]["script"]["sha256"] == PINNED_PROBE_SHA
    assert receipt["lock_chain"]["contract"] == "performance_lock_rearm1_v2"
    assert receipt["lock_chain"]["validator_sha256"] == probe_v1.sha256_file(
        Path(subject.__file__)
    )
    assert receipt["lock_chain"]["old_v1_root_reused"] is False
    validated = spot_v1._validate_materializer_platform_parity(
        windows_probe_path=paths["windows_probe"],
        linux_probe_path=paths["linux_probe"],
        parity_receipt_path=paths["output"],
        parity_script_path=Path(probe_v1.__file__),
        rust_source_path=RUST_SOURCE,
        cargo_lock_path=CARGO_LOCK,
        global_claim_path=paths["global_claim"],
        materialization_path=paths["materialization"],
        seal_path=paths["seal"],
        ai_profiles_path=AI_PROFILES,
    )
    assert validated == receipt


def test_rearm1_wrapper_rejects_old_lock_reuse_and_restores_v1(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    seal = probe_v1._read_canonical(paths["seal"], "seal")
    seal["old_performance_lock_comparison"]["old_v1_root_reused"] = True
    _write(paths["seal"], seal)
    original = probe_v1._validate_lock_chain
    with pytest.raises(ValueError, match="old-lock no-reuse proof changed"):
        subject.compare_probes(
            windows_probe_path=paths["windows_probe"],
            linux_probe_path=paths["linux_probe"],
            global_claim_path=paths["global_claim"],
            materialization_path=paths["materialization"],
            seal_path=paths["seal"],
            rust_source_path=RUST_SOURCE,
            cargo_lock_path=CARGO_LOCK,
            ai_profiles_path=AI_PROFILES,
        )
    assert probe_v1._validate_lock_chain is original


def test_original_probe_script_remains_byte_pinned() -> None:
    assert probe_v1.sha256_file(Path(probe_v1.__file__)) == PINNED_PROBE_SHA
