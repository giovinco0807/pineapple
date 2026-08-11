from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_content_v2
    as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP12_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12_pair_v1_actual"
)


def _read(name: str) -> dict[str, Any]:
    value = json.loads((STEP12_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sources() -> dict[str, str]:
    return {
        path: (REPO_ROOT / "src" / Path(path)).read_text(encoding="utf-8")
        for path in subject.REQUIRED_RUNTIME_SOURCE_PATHS
    }


def test_content_binding_is_uri_and_generation_free_and_reconstructable() -> None:
    candidate = _read("candidate_transport_contract.json")
    reference = _read("reference_transport_contract.json")
    binding = subject.build_bootstrap_source_content_binding(
        runtime_source_files=_sources(),
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
    )
    checked = subject.validate_bootstrap_source_content_binding(
        binding,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
    )
    assert checked["source_object_count"] == 3
    assert checked["final_source_prefix_present"] is False
    assert checked["generation_present"] is False
    assert checked["prefix_derivation_rule"] == (
        subject.PREFIX_DERIVATION_RULE
    )
    assert checked["runtime_source_bundle"]["bytes"] > 262_144
    serialized = subject.canonical_bytes(checked)
    assert b"gs://" not in serialized
    assert b"generation" in serialized
    assert b'"generation_present":false' in serialized

    bundle = subject.build_runtime_source_bundle(_sources())
    assert (
        subject.validate_runtime_bundle_against_content_binding(
            bundle,
            content_binding=checked,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )
        == bundle
    )


def test_content_binding_rejects_resealed_runtime_or_role_drift() -> None:
    candidate = _read("candidate_transport_contract.json")
    reference = _read("reference_transport_contract.json")
    binding = subject.build_bootstrap_source_content_binding(
        runtime_source_files=_sources(),
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
    )
    changed = copy.deepcopy(binding)
    changed["runtime_source_bundle"]["bytes"] += 1
    body = dict(changed)
    body.pop("bootstrap_source_content_binding_sha256")
    changed["bootstrap_source_content_binding_sha256"] = (
        subject.canonical_sha256(body)
    )
    with pytest.raises(ValueError, match="content binding"):
        subject.validate_bootstrap_source_content_binding(
            changed,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )

    wrong_reference = copy.deepcopy(reference)
    wrong_reference["metadata_binding"]["source_role"] = "candidate"
    with pytest.raises(ValueError):
        subject.validate_bootstrap_source_content_binding(
            binding,
            candidate_payload_contract=candidate,
            reference_payload_contract=wrong_reference,
        )


def test_content_binding_requires_exact_compiling_source_closure() -> None:
    candidate = _read("candidate_transport_contract.json")
    reference = _read("reference_transport_contract.json")
    missing = _sources()
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="closure"):
        subject.build_bootstrap_source_content_binding(
            runtime_source_files=missing,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )
    broken = _sources()
    key = next(iter(broken))
    broken[key] = "def broken(:\n"
    with pytest.raises(ValueError, match="compile"):
        subject.build_bootstrap_source_content_binding(
            runtime_source_files=broken,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )
