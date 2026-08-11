from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_attempt0_identity_v1
    as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def rebound_inputs() -> tuple[
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    stage1 = _read(STEP11_ROOT / "transport_contract.json")
    done = _read(STEP11_ROOT / "late_done_envelope.json")
    provision = _read(STEP11_ROOT / "package_provision_receipt.json")
    receive = adapter.build_receive(
        stage1["adapter_preview"], done_records=[done]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"]
        == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    signer = controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    common = {
        "package_dir": PACKAGE_DIR,
        "stage_id": plan.STAGE2_ID,
        "attempt_index": 0,
        "offline_wheel_record": wheel,
        "controller_public_key_record": signer.public_record,
        "prerequisite_stage1_preview": stage1["adapter_preview"],
        "prerequisite_stage1_receive": receive,
    }
    candidate = subject.build_rebound_job_contract(
        job_id=subject.SOURCE_JOB_IDS[0], **common
    )
    reference = subject.build_rebound_job_contract(
        job_id=subject.SOURCE_JOB_IDS[1], **common
    )
    return signer, candidate, reference, provision


def _identity_kwargs(
    signer: Any,
    provision: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "controller_public_key_sha256": subject.canonical_sha256(
            signer.public_record
        ),
        "gate_plan_sha256": "a" * 64,
        "immutable_package_source_receipt_sha256": provision[
            "receipt_sha256"
        ],
        "immutable_package_generations_sha256": provision[
            "package_generations_sha256"
        ],
        "issued_at_unix_seconds": 1_800_000_000,
        "expires_at_unix_seconds": 1_800_006_000,
        "output_root": output_root,
    }


def test_rebound_contract_is_fresh_but_local_planning_only(
    rebound_inputs: tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    _signer, candidate, reference, _provision = rebound_inputs
    assert [
        candidate["metadata_binding"]["instance_name"],
        reference["metadata_binding"]["instance_name"],
    ] == list(subject.ATTEMPT0_INSTANCE_NAMES)
    assert candidate["adapter_preview"]["run_name"] == (
        subject.EXECUTION_RUN_NAME
    )
    assert candidate["direct_stage_identity_sha256"] == (
        subject.EXPECTED_DIRECT_STAGE_IDENTITY_SHA256
    )
    assert candidate["remote_layout"]["stage_prefix"] == (
        subject.EXPECTED_DIRECT_STAGE_PREFIX
    )
    assert candidate["remote_layout"] == reference["remote_layout"]
    assert subject.OLD_STEP12_DIRECT_STAGE_PREFIX not in (
        candidate["remote_layout"]["stage_prefix"]
    )
    assert candidate["capabilities"]["cloud_executable"] is False
    assert candidate["capabilities"]["cloud_launch_authorized"] is False


def test_rebound_patch_restores_globals_and_rejects_attempt1(
    rebound_inputs: tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    _signer, candidate, _reference, _provision = rebound_inputs
    original_stage = adapter._stage
    original_name = transport.deterministic_instance_name
    original_run = transport.STAGE2_RUN_NAME
    with pytest.raises(RuntimeError, match="not re-entrant"):
        with subject.rebound_transport_identity():
            with subject.rebound_transport_identity():
                pass
    assert adapter._stage is original_stage
    assert transport.deterministic_instance_name is original_name
    assert transport.STAGE2_RUN_NAME is original_run

    with pytest.raises(ValueError, match="attempt1"):
        subject.build_rebound_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=plan.STAGE2_ID,
            job_id=candidate["metadata_binding"]["job_id"],
            attempt_index=1,
        )


def test_execution_identity_uses_trusted_inputs_and_validates_existing_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rebound_inputs: tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    signer, candidate, reference, provision = rebound_inputs
    actual_root = tmp_path / subject.DEFAULT_OUTPUT_ROOT_NAME
    monkeypatch.setattr(
        subject, "EXPECTED_ACTUAL_OUTPUT_ROOT", actual_root.resolve()
    )
    kwargs = _identity_kwargs(signer, provision, actual_root)
    identity = subject.build_execution_identity(
        candidate, reference, **kwargs
    )
    assert identity["local_alias_planning_only"] is True
    assert identity["inner_fresh_process_rebind_implemented"] is False
    assert identity["rebound_transport_contract_cloud_executable"] is False
    assert identity["v2_alias_bridge_required_before_cloud_launch"] is True
    assert identity["cloud_launch_authorized"] is False
    assert identity["retry_authorized"] is False
    assert identity["attempt1_authorized"] is False
    assert identity["immutable_package_reuse"][
        "old_immutable_package_bytes_may_be_reused"
    ] is True

    actual_root.mkdir()
    assert subject.validate_execution_identity(
        identity, candidate, reference, **kwargs
    ) == identity

    changed = copy.deepcopy(identity)
    changed["gate_plan_sha256"] = "b" * 64
    unsigned = dict(changed)
    unsigned.pop("identity_sha256")
    changed["identity_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="identity changed"):
        subject.validate_execution_identity(
            changed, candidate, reference, **kwargs
        )

    with pytest.raises(ValueError, match="payload mapping"):
        subject.validate_execution_identity(
            identity,
            candidate,
            reference,
            **{
                **kwargs,
                "controller_public_key_sha256": "c" * 64,
            },
        )


def test_actual_and_local_smoke_roots_are_disjoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actual = tmp_path / subject.DEFAULT_OUTPUT_ROOT_NAME
    smoke = tmp_path / subject.LOCAL_SMOKE_OUTPUT_ROOT_NAME
    monkeypatch.setattr(
        subject, "EXPECTED_ACTUAL_OUTPUT_ROOT", actual.resolve()
    )
    assert subject.require_fresh_output_root(actual) == actual.resolve()
    assert subject.require_local_smoke_output_root(smoke) == smoke.resolve()
    with pytest.raises(FileExistsError):
        subject.require_fresh_output_root(smoke)
    with pytest.raises(FileExistsError):
        subject.require_local_smoke_output_root(actual)
    wrong_parent = (
        tmp_path
        / "other"
        / subject.DEFAULT_OUTPUT_ROOT_NAME
    )
    with pytest.raises(FileExistsError):
        subject.require_fresh_output_root(wrong_parent)
    actual.mkdir()
    with pytest.raises(FileExistsError):
        subject.require_fresh_output_root(actual)


def test_immutable_package_receipt_binds_exact_old_generations(
    rebound_inputs: tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    _signer, candidate, reference, provision = rebound_inputs
    by_uri = {
        row["uri"]: row
        for row in candidate["remote_layout"]["package_inventory"][
            "records"
        ]
    }
    observed = [
        {
            "uri": uri,
            "generation": generation,
            "sha256": by_uri[uri]["sha256"],
            "bytes": by_uri[uri]["bytes"],
        }
        for uri, generation in provision["package_generations"].items()
    ]
    receipt = subject.build_immutable_package_readback_receipt(
        candidate,
        reference,
        execution_identity_sha256="d" * 64,
        source_package_receipt_sha256=provision["receipt_sha256"],
        expected_package_generations=provision[
            "package_generations"
        ],
        observed_at_unix_seconds=1_800_000_000,
        observed_records=observed,
    )
    assert receipt["record_count"] == 16
    assert receipt["package_generations_sha256"] == provision[
        "package_generations_sha256"
    ]
    assert receipt["package_reuse_only"] is True
    assert receipt["result_stage_prefix_reuse"] is False
    assert receipt["cloud_mutation_performed"] is False
    assert subject.validate_immutable_package_readback_receipt(
        receipt,
        candidate,
        reference,
        execution_identity_sha256="d" * 64,
        source_package_receipt_sha256=provision["receipt_sha256"],
        expected_package_generations=provision[
            "package_generations"
        ],
        now_unix_seconds=1_800_000_120,
    ) == receipt

    wrong_generation = copy.deepcopy(observed)
    wrong_generation[0]["generation"] += 1
    with pytest.raises(ValueError, match="generation/SHA/bytes"):
        subject.build_immutable_package_readback_receipt(
            candidate,
            reference,
            execution_identity_sha256="d" * 64,
            source_package_receipt_sha256=provision["receipt_sha256"],
            expected_package_generations=provision[
                "package_generations"
            ],
            observed_at_unix_seconds=1_800_000_000,
            observed_records=wrong_generation,
        )
    with pytest.raises(ValueError, match="stale"):
        subject.validate_immutable_package_readback_receipt(
            receipt,
            candidate,
            reference,
            execution_identity_sha256="d" * 64,
            source_package_receipt_sha256=provision["receipt_sha256"],
            expected_package_generations=provision[
                "package_generations"
            ],
            now_unix_seconds=1_800_000_121,
        )


def test_patch_context_does_not_rewrite_old_sources() -> None:
    paths = [
        Path(adapter.__file__),
        Path(transport.__file__),
        Path(plan.__file__),
    ]
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    with subject.rebound_transport_identity():
        assert adapter._stage is not None
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    assert after == before
