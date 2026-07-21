"""Step12d mandatory regression guards.

Two Step12c No-Go root causes are pinned here before any Step12d cloud run:

1. producer/validator schema drift: the token-barrier SUCCESS receipt is
   produced by the REAL producer code path (not a hand-written fixture) and
   must cross the Phase2 validator boundary unchanged;
2. machine-type contract drift: the c4-standard-8 memory contract must stay
   bound to the provider's actual 30720 MiB shape and the preflight check
   must fail closed on any other readback.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_preflight_gate_v2
    as preflight_gate,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_lifecycle_v2
    as phase2_lifecycle,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)


CONTROLLER = (
    "ofc-m31-s2d-controller-rt@"
    "ofc-solver-485418.iam.gserviceaccount.com"
)


@dataclass
class _Mutation:
    changed: bool
    attempts: int = 1


class _Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def unix(self) -> int:
        return 1_900_000_000 + int(self.elapsed)

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class _Admin:
    controller_service_account = CONTROLLER

    def __init__(self) -> None:
        self.binding: dict[str, Any] | None = None

    def add_binding(
        self,
        _target: Any,
        *,
        role: str,
        member: str,
        condition: dict[str, Any],
    ) -> _Mutation:
        self.binding = {
            "role": role,
            "members": [member],
            "condition": dict(condition),
        }
        return _Mutation(True)

    def remove_binding(
        self,
        _target: Any,
        *,
        role: str,
        member: str,
        condition: dict[str, Any],
    ) -> _Mutation:
        assert self.binding == {
            "role": role,
            "members": [member],
            "condition": dict(condition),
        }
        self.binding = None
        return _Mutation(True)

    def get_policy(self, _target: Any) -> dict[str, Any]:
        return {
            "bindings": []
            if self.binding is None
            else [dict(self.binding)]
        }


class _Token:
    expire_time = "2030-01-01T01:00:00Z"

    def access_token(self) -> str:
        return "ya29.step12d-roundtrip-secret"


class _Generator:
    controller_service_account = CONTROLLER

    def generate_controller_access_token(
        self, *, lifetime_seconds: int
    ) -> _Token:
        assert lifetime_seconds == 3_600
        return _Token()


def _real_producer_outcome() -> token_barrier.TokenBarrierOutcome:
    clock = _Clock()
    return token_barrier.run_token_creator_barrier(
        admin=_Admin(),
        token_generator=_Generator(),
        binding=token_barrier.build_token_creator_binding(
            controller_service_account=CONTROLLER,
            expires_at_rfc3339="2030-01-01T00:10:00Z",
        ),
        now_monotonic=clock.monotonic,
        now_unix_seconds=clock.unix,
        sleep=clock.sleep,
    )


def _plan() -> dict[str, Any]:
    return {
        "principals": {
            "controller_service_account": {"email": CONTROLLER}
        }
    }


def test_real_producer_receipt_crosses_phase2_validator() -> None:
    outcome = _real_producer_outcome()
    receipt = outcome.receipt
    assert receipt["status"] == token_barrier.SUCCESS_STATUS
    assert set(receipt) == phase2_lifecycle._TOKEN_SUCCESS_FIELDS
    assert (
        phase2_lifecycle._validate_token_barrier(outcome, _plan())
        == receipt["receipt_sha256"]
    )


def test_step12c_failure_fields_stay_in_shared_allowlist() -> None:
    for field in (
        "token_creator_revoke_zero_readback",
        "token_creator_revoke_zero_readback_evidence_sha256",
    ):
        assert field in phase2_lifecycle._TOKEN_SUCCESS_FIELDS


def test_extra_producer_field_fails_closed_at_validator() -> None:
    outcome = _real_producer_outcome()
    grown = copy.deepcopy(dict(outcome.receipt))
    grown.pop("receipt_sha256")
    grown["newly_added_field"] = True
    grown["receipt_sha256"] = token_barrier.canonical_sha256(grown)
    with pytest.raises(ValueError, match="fields changed"):
        phase2_lifecycle._validate_token_barrier(
            token_barrier.TokenBarrierOutcome(object(), grown), _plan()
        )


def _machine_payload(memory_mb: int) -> dict[str, Any]:
    project = preflight_gate.payload_transport.PROJECT
    zone = preflight_gate.payload_transport.ZONE
    name = deployment_v2.ACTUAL_MACHINE_TYPE
    return {
        "name": name,
        "guestCpus": deployment_v2.ACTUAL_VCPUS_PER_VM,
        "memoryMb": memory_mb,
        "zone": zone,
        "selfLink": (
            f"https://www.googleapis.com/compute/v1/projects/{project}"
            f"/zones/{zone}/machineTypes/{name}"
        ),
    }


def _region_payload() -> dict[str, Any]:
    project = preflight_gate.payload_transport.PROJECT
    zone = preflight_gate.payload_transport.ZONE
    region = zone.rsplit("-", 1)[0]
    return {
        "name": region,
        "status": "UP",
        "selfLink": (
            f"https://www.googleapis.com/compute/v1/projects/{project}"
            f"/regions/{region}"
        ),
        "quotas": [
            {"metric": "CPUS", "limit": 24, "usage": 0},
            {"metric": "PREEMPTIBLE_CPUS", "limit": 24, "usage": 0},
        ],
    }


def test_machine_contract_is_bound_to_provider_actual_30720() -> None:
    assert deployment_v2.ACTUAL_MEMORY_MB == 30_720
    facts = preflight_gate._machine_region_facts(
        _machine_payload(30_720), _region_payload()
    )
    assert facts["memory_mb"] == 30_720
    assert facts["machine_type"] == "c4-standard-8"


def test_old_wrong_32768_memory_readback_fails_closed() -> None:
    with pytest.raises(ValueError, match="machine or region readback"):
        preflight_gate._machine_region_facts(
            _machine_payload(32_768), _region_payload()
        )
