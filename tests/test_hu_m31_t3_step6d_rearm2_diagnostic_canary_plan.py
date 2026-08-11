from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_spot_v1 as production_transport
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "configs/hu_joint_policy_m31_t3_step6d_"
    "rearm2_diagnostic_canary_v1.json"
)
REARM2_PACKAGE = Path(
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-rearm2-20260718-001/package"
)


def _file_state(path: Path) -> tuple[bool, str | None, int | None]:
    if not path.exists():
        return False, None, None
    return True, subject.sha256_file(path), path.stat().st_size


def test_frozen_contract_has_exact_bounded_stages_and_distinct_identities() -> None:
    contract = subject.validate_frozen_contract(CONTRACT_PATH)
    stage1, stage2 = contract["stages"]

    assert stage1["stage_id"] == subject.STAGE1_ID
    assert stage1["selected_job_ids"] == ["candidate-shard-00"]
    assert stage1["source_roles"] == ["candidate"]
    assert stage1["vm_count"] == 1
    assert stage1["hand_indices"] == list(subject.STAGE1_HAND_INDICES)

    assert stage2["stage_id"] == subject.STAGE2_ID
    assert stage2["selected_job_ids"] == [
        "candidate-shard-01",
        "reference-shard-01",
    ]
    assert stage2["source_roles"] == ["candidate", "reference"]
    assert stage2["vm_count"] == 2
    assert stage2["hand_indices"] == list(subject.STAGE2_HAND_INDICES)

    assert set(stage1["hand_indices"]).isdisjoint(stage2["hand_indices"])
    assert stage1["run_name"] != stage2["run_name"]
    assert stage1["claim_name"] != stage2["claim_name"]
    assert stage1["result_name"] != stage2["result_name"]
    assert subject.EXPECTED_REARM2_RUN_NAME not in {
        stage1["run_name"],
        stage2["run_name"],
    }
    assert production_transport.LAUNCH_CLAIM_NAME not in {
        stage1["claim_name"],
        stage2["claim_name"],
    }
    assert production_transport.LAUNCH_RESULT_NAME not in {
        stage1["result_name"],
        stage2["result_name"],
    }
    assert all(stage["cloud_capable"] is False for stage in contract["stages"])
    assert all(
        len(stage["selected_job_ids"]) < production_transport.MAX_LOGICAL_JOBS
        for stage in contract["stages"]
    )


def test_frozen_contract_uses_only_development_roots_and_authorizes_nothing() -> None:
    contract = subject.validate_frozen_contract(CONTRACT_PATH)
    roots = contract["development_root_source"]

    assert roots["root_directory"] == subject.DEVELOPMENT_ROOTS_RELATIVE
    assert roots["classification"].endswith("_diagnostic_only")
    assert roots["rearm2_locked_root_directory_used"] is False
    assert roots["new_seed_namespace_opened"] is False
    assert "roots-open" not in roots["root_directory"]
    for stage in contract["stages"]:
        for record in stage["root_records"]:
            assert record["path"].startswith(
                subject.DEVELOPMENT_ROOTS_RELATIVE + "/"
            )
            assert "performance-lock-rearm2" not in record["path"]

    assert set(contract["authorization"].values()) == {False}
    assert contract["identity_guards"]["all20_job_set_selected"] is False
    assert contract["implementation_boundary"] == {
        "cloud_capable": False,
        "existing_rearm2_launch_reused": False,
        "existing_full100_startup_compatible_with_subset_claim": False,
        "reason": (
            "accepted startup and launch-chain validators require the exact "
            "all20 initial job set"
        ),
        "required_before_any_cloud_canary": [
            "versioned diagnostic-only package and authorization schema",
            "versioned startup verifier for the exact frozen stage job set",
            "stage-specific heartbeat upload DONE and receive validators",
            "write-once stage receipt whose validation gates the next stage",
        ],
        "production_launch_semantics_changed": False,
    }


@pytest.mark.skipif(
    not (REARM2_PACKAGE / "manifest.json").is_file(),
    reason="local immutable rearm2 package is not present",
)
def test_live_audit_matches_frozen_contract_without_cloud_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cloud_forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("diagnostic contract audit must not invoke a cloud helper")

    monkeypatch.setattr(production_transport.tail_spot, "_run", cloud_forbidden)
    monkeypatch.setattr(
        production_transport.tail_spot,
        "_subprocess_run",
        cloud_forbidden,
    )

    protected = (
        REARM2_PACKAGE / "manifest.json",
        REARM2_PACKAGE / "ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip",
        REARM2_PACKAGE / "startup_hu_m31_t3_step6d_full100_v1.sh",
        REARM2_PACKAGE / "ACTUAL_PACKAGE_PREAUTHORIZE_SMOKE.json",
        REPO_ROOT
        / "outputs/hu_joint_policy/m31_t3_step6d/"
        "GLOBAL_PERFORMANCE_LOCK_REARM2_CLAIM.json",
        REPO_ROOT / subject.CURRENT_PROFILE_RELATIVE,
    )
    absent_cloud_paths = (
        REARM2_PACKAGE / production_transport.AUTHORIZATION_NAME,
        REARM2_PACKAGE / production_transport.LAUNCH_CLAIM_NAME,
        REARM2_PACKAGE / production_transport.LAUNCH_RESULT_NAME,
        REPO_ROOT / subject.DEFAULT_REARM2_GLOBAL_SPOT_CLAIM,
    )
    before = {path: _file_state(path) for path in protected}
    assert all(not path.exists() for path in absent_cloud_paths)

    report = subject.audit_frozen_contract(contract_path=CONTRACT_PATH)

    after = {path: _file_state(path) for path in protected}
    assert before == after
    assert all(not path.exists() for path in absent_cloud_paths)
    assert report["status"] == "pass_local_only_cloud_not_authorized"
    assert report["stage1_job_count"] == 1
    assert report["stage2_job_count"] == 2
    assert report["max_stage_vm_count"] == 2
    assert report["gcloud_invoked"] is False
    assert report["all20_launch_authorized"] is False
    assert report["current_profile_changed"] is False


def test_structural_tampering_fails_closed() -> None:
    contract = subject.validate_frozen_contract(CONTRACT_PATH)
    mutations = []

    changed = deepcopy(contract)
    changed["stages"][0]["selected_job_ids"] = ["candidate-shard-00", "candidate-shard-01"]
    mutations.append(changed)

    changed = deepcopy(contract)
    changed["stages"][1]["selected_job_ids"] = list(
        production_transport.authorized_job_ids()
    )
    mutations.append(changed)

    changed = deepcopy(contract)
    changed["stages"][1]["source_roles"] = ["candidate", "candidate"]
    mutations.append(changed)

    changed = deepcopy(contract)
    changed["stages"][1]["run_name"] = changed["stages"][0]["run_name"]
    mutations.append(changed)

    changed = deepcopy(contract)
    changed["development_root_source"]["rearm2_locked_root_directory_used"] = True
    mutations.append(changed)

    changed = deepcopy(contract)
    changed["authorization"]["cloud_launch_authorized"] = True
    mutations.append(changed)

    changed = deepcopy(contract)
    changed["implementation_boundary"]["cloud_capable"] = True
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(ValueError):
            subject.validate_diagnostic_canary_contract(mutation)


def test_builder_refuses_a_spot_claim_before_reading_production_content(
    tmp_path: Path,
) -> None:
    spot_claim = tmp_path / "GLOBAL_PERFORMANCE_LOCK_REARM2_SPOT_CLAIM.json"
    spot_claim.write_text("occupied\n", encoding="utf-8")
    inputs = subject.DiagnosticCanaryInputs(
        rearm2_global_spot_claim_path=spot_claim,
    )
    with pytest.raises(
        FileExistsError,
        match="must precede production cloud authorization",
    ):
        subject.build_diagnostic_canary_contract(inputs)


def test_builder_refuses_locked_rearm2_roots_as_diagnostic_input() -> None:
    inputs = subject.DiagnosticCanaryInputs(
        development_root_directory=Path(
            "D:/ofc-gcp-runs/"
            "regular-hu-m31-c02-performance-lock-rearm2-20260718-001/"
            "roots-open"
        )
    )
    with pytest.raises(ValueError, match="escaped repository|root path changed"):
        subject.build_diagnostic_canary_contract(inputs)


def test_frozen_production_launch_still_rejects_a_subset_before_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cloud_forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("subset rejection must happen before a cloud helper")

    monkeypatch.setattr(production_transport, "_quota_snapshot", cloud_forbidden)
    monkeypatch.setattr(production_transport, "_instance_rows", cloud_forbidden)
    monkeypatch.setattr(production_transport, "_object_exists", cloud_forbidden)

    with pytest.raises(ValueError, match="exactly all twenty"):
        production_transport.preflight_launch(
            manifest={},
            selected=["candidate-shard-00"],
        )

