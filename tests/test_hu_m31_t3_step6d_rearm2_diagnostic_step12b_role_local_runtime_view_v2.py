from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
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
STEP12_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12_pair_v1_actual"
)
RUN_NONCE = "c3" * 32


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    candidate = _read(
        STEP12_ROOT / "candidate_transport_contract.json"
    )
    reference = _read(
        STEP12_ROOT / "reference_transport_contract.json"
    )
    public_key = _read(STEP12_ROOT / "controller_public_key.json")
    deployment = subject.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
    )
    return candidate, reference, public_key, deployment


@pytest.mark.parametrize("role_index", [0, 1])
def test_role_local_runtime_view_accepts_exact_selected_role_only(
    inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
    role_index: int,
) -> None:
    candidate, reference, public_key, deployment = inputs
    payload = (candidate, reference)[role_index]
    view = subject.validate_role_local_runtime_view(
        deployment,
        selected_payload_contract=payload,
        controller_public_key_record=public_key,
        external_job_id=deployment["selected_job_ids"][role_index],
    )
    assert view["schema"] == subject.ROLE_LOCAL_RUNTIME_VIEW_SCHEMA
    assert view["source_role"] == ("candidate", "reference")[role_index]
    assert view["selected_payload_contract_sha256"] == (
        subject.canonical_sha256(payload)
    )
    assert view["unselected_payload_reconstructed"] is False
    assert view["unselected_payload_validated_on_vm"] is False
    unsigned = dict(view)
    digest = unsigned.pop("role_local_runtime_view_sha256")
    assert subject.canonical_sha256(unsigned) == digest


def test_cross_role_payload_swap_fails_closed(
    inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, reference, public_key, deployment = inputs
    with pytest.raises(ValueError, match="role-local"):
        subject.validate_role_local_runtime_view(
            deployment,
            selected_payload_contract=reference,
            controller_public_key_record=public_key,
            external_job_id=deployment["selected_job_ids"][0],
        )
    with pytest.raises(ValueError, match="role-local"):
        subject.validate_role_local_runtime_view(
            deployment,
            selected_payload_contract=candidate,
            controller_public_key_record=public_key,
            external_job_id=deployment["selected_job_ids"][1],
        )


def test_resealed_deployment_mapping_still_fails_closed(
    inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, _reference, public_key, deployment = inputs
    changed = copy.deepcopy(deployment)
    changed["instances"][0]["inner_job_id"] = "reference-shard-01"
    unsigned = dict(changed)
    unsigned.pop("deployment_contract_sha256")
    changed["deployment_contract_sha256"] = subject.canonical_sha256(
        unsigned
    )
    with pytest.raises(ValueError, match="role-local"):
        subject.validate_role_local_runtime_view(
            changed,
            selected_payload_contract=candidate,
            controller_public_key_record=public_key,
            external_job_id=deployment["selected_job_ids"][0],
        )


def test_selected_payload_omission_fails_closed(
    inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    _candidate, _reference, public_key, deployment = inputs
    with pytest.raises(ValueError):
        subject.validate_role_local_runtime_view(
            deployment,
            selected_payload_contract={},
            controller_public_key_record=public_key,
            external_job_id=deployment["selected_job_ids"][0],
        )


def test_wrong_public_key_binding_fails_closed(
    inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, _reference, _public_key, deployment = inputs
    wrong_public_key = _read(
        STEP11_ROOT / "controller_public_key.json"
    )
    with pytest.raises(ValueError, match="role-local"):
        subject.validate_role_local_runtime_view(
            deployment,
            selected_payload_contract=candidate,
            controller_public_key_record=wrong_public_key,
            external_job_id=deployment["selected_job_ids"][0],
        )
