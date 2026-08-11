"""Complete-Go-gated explicit opt-in profile for Attempt13.

This sibling package is intentionally absent from ``ai_profiles``.  Importing
it does not register or activate anything and never consults ``current``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from ofc_regular.hu_m43_attempt13_contract import M43_ATTEMPT13_PLAN_SHA256
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt13_distilled_model import (
    BoundHuM43Attempt13DistilledModel,
    is_bound_attempt13_distilled_model,
)
from ofc_regular.hu_m43_attempt13_distilled_runtime import (
    ATTEMPT13_BOUND_EXECUTION_MODULES,
    validate_frozen_execution_modules,
)
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.validate_hu_m43_attempt13_acceptance import (
    ATTEMPT13_BASELINE_PROFILE,
    ATTEMPT13_OPPONENTS,
    ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
    ATTEMPT13_PROFILE_ID,
    load_bound_attempt13_distilled_model,
)


_COMPLETE_GO_ISSUER = object()


class _Attempt13CompleteGoCapability:
    """Opaque proof issued only by the byte-addressed public profile builder."""

    __slots__ = ("complete_go_sha256", "model_sha256", "__issuer")

    def __init__(
        self,
        *,
        complete_go_sha256: str,
        model_sha256: str,
        issuer: object,
    ) -> None:
        if issuer is not _COMPLETE_GO_ISSUER:
            raise TypeError("Attempt13 Complete-Go capabilities are builder-issued")
        self.complete_go_sha256 = complete_go_sha256
        self.model_sha256 = model_sha256
        self.__issuer = issuer

    def is_valid(self) -> bool:
        return self.__issuer is _COMPLETE_GO_ISSUER


class Attempt13SelectiveOverridePolicy(HuM4T1SelectiveOverridePolicy):
    """Second-seat-only wrapper accepting only a loader-issued capability."""

    def __init__(
        self,
        baseline_policy: object,
        *,
        bound_model: BoundHuM43Attempt13DistilledModel,
        _complete_go_capability: _Attempt13CompleteGoCapability | None = None,
        decision_log: list[dict[str, Any]] | None = None,
        decision_log_path: str | Path | None = None,
    ) -> None:
        if not _called_by_complete_go_builder(
            bound_model, _complete_go_capability
        ):
            raise TypeError(
                "stage20_m4_attempt13 requires a builder-issued Complete-Go capability"
            )
        if not is_bound_attempt13_distilled_model(bound_model):
            raise TypeError("stage20_m4_attempt13 requires a bound Attempt13 model")
        if bound_model.safety_enabled is not True or bound_model.winner_frozen is not True:
            raise ValueError("stage20_m4_attempt13 requires a frozen safe winner")
        if float(bound_model.safety_threshold) != 0.5:
            raise ValueError("stage20_m4_attempt13 safety threshold changed")
        if int(bound_model.minimum_fold_votes) != 4:
            raise ValueError("stage20_m4_attempt13 fold-vote gate changed")
        baseline_seat = _validate_stage19_baseline_policy(baseline_policy)
        super().__init__(
            baseline_policy,
            action_value_model=bound_model,
            safety_model=bound_model,
            safety_probability_threshold=0.5,
            enabled=True,
            allowed_seats=("second",),
            decision_log=decision_log,
            decision_log_path=decision_log_path,
            policy_id=ATTEMPT13_PROFILE_ID,
            runtime_binding_verified=True,
        )
        self.baseline_seat = baseline_seat
        self.__attempt13_baseline_policy = baseline_policy
        self.__attempt13_bound_model = bound_model

    def choose_action_observation(
        self, observation: ActorObservation, **kwargs: Any
    ) -> Any:
        current_seat = _validate_stage19_baseline_policy(self.baseline_policy)
        if (
            self.baseline_policy is not self.__attempt13_baseline_policy
            or current_seat != self.baseline_seat
            or observation.seat != self.baseline_seat
        ):
            raise ValueError("stage20_m4_attempt13 baseline/observation seat mismatch")
        if (
            self.action_value_model is not self.__attempt13_bound_model
            or self.safety_model is not self.__attempt13_bound_model
            or not is_bound_attempt13_distilled_model(self.__attempt13_bound_model)
            or self.enabled is not True
            or self.allowed_seats != ("second",)
            or self.runtime_binding_verified is not True
            or self.policy_id != ATTEMPT13_PROFILE_ID
            or float(self.safety_probability_threshold) != 0.5
        ):
            raise RuntimeError("stage20_m4_attempt13 runtime binding drifted")
        return super().choose_action_observation(observation, **kwargs)


def load_attempt13_complete_go(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_model_sha256: str,
) -> dict[str, Any]:
    """Re-open a byte-addressed Complete-Go capability, or fail closed."""

    artifact = Path(path)
    actual_sha = _file_sha256(artifact)
    if actual_sha != _sha256(expected_sha256, "acceptance artifact SHA"):
        raise ValueError("Attempt13 Complete-Go artifact SHA mismatch")
    value = json.loads(artifact.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("Attempt13 Complete-Go artifact must be a mapping")
    status = dict(value)
    gates = status.get("gates")
    if (
        status.get("schema") != ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA
        or status.get("status") != "complete_go"
        or status.get("profile_id") != ATTEMPT13_PROFILE_ID
        or status.get("search_plan_sha256") != M43_ATTEMPT13_PLAN_SHA256
        or status.get("model_sha256")
        != _sha256(expected_model_sha256, "expected model SHA")
        or status.get("baseline_profile") != ATTEMPT13_BASELINE_PROFILE
        or status.get("opponents") != list(ATTEMPT13_OPPONENTS)
        or status.get("promotion_eligible") is not True
        or status.get("explicit_opt_in_authorized") is not True
        or status.get("automatic_activation_authorized") is not False
        or status.get("teacher_values_reported_as_realized_match_ev") is not False
        or status.get("threshold_reselection_performed") is not False
        or status.get("audit50_rows_used_for_fit") != 0
        or status.get("current_profile_mutated") is not False
        or status.get("runtime_policy_activated") is not False
        or status.get("full_replacement") is not False
        or isinstance(status.get("passed_gates"), bool)
        or not isinstance(status.get("passed_gates"), int)
        or status.get("passed_gates") != status.get("total_gates")
        or not isinstance(gates, list)
        or not gates
        or len(gates) != status.get("total_gates")
        or not all(isinstance(gate, Mapping) and gate.get("passed") is True for gate in gates)
    ):
        raise ValueError("Attempt13 Complete-Go artifact is not activation authority")
    for field in (
        "population_plan_sha256",
        "records_sha256",
        "evaluation_sha256",
        "merge_manifest_sha256",
    ):
        _sha256(status.get(field), field)
    return status


def build_stage20_m4_attempt13(
    baseline_policy: object,
    *,
    complete_go_path: str | Path,
    expected_complete_go_sha256: str,
    model_path: str | Path,
    expected_model_sha256: str,
    runtime_freeze_path: str | Path,
    training_manifest_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
    decision_log: list[dict[str, Any]] | None = None,
    decision_log_path: str | Path | None = None,
) -> Attempt13SelectiveOverridePolicy:
    """Build the profile only from explicit baseline and immutable Go inputs."""

    _validate_stage19_baseline_policy(baseline_policy)
    status = load_attempt13_complete_go(
        complete_go_path,
        expected_sha256=expected_complete_go_sha256,
        expected_model_sha256=expected_model_sha256,
    )
    if _file_sha256(model_path) != status["model_sha256"]:
        raise ValueError("Attempt13 Complete-Go/model bytes mismatch")
    validate_frozen_execution_modules(
        extracted_root=runtime_source_root,
        manifest=runtime_source_manifest_path,
        module_names=ATTEMPT13_BOUND_EXECUTION_MODULES,
    )
    bound_model = load_bound_attempt13_distilled_model(
        model_path,
        expected_sha256=expected_model_sha256,
        runtime_freeze=runtime_freeze_path,
        training_manifest_path=training_manifest_path,
        runtime_source_manifest_path=runtime_source_manifest_path,
        runtime_source_root=runtime_source_root,
        runtime_dependency_root=runtime_dependency_root,
    )
    complete_go_capability = _Attempt13CompleteGoCapability(
        complete_go_sha256=_sha256(
            expected_complete_go_sha256, "acceptance artifact SHA"
        ),
        model_sha256=_sha256(expected_model_sha256, "expected model SHA"),
        issuer=_COMPLETE_GO_ISSUER,
    )
    return Attempt13SelectiveOverridePolicy(
        baseline_policy,
        bound_model=bound_model,
        _complete_go_capability=complete_go_capability,
        decision_log=decision_log,
        decision_log_path=decision_log_path,
    )


def _validate_stage19_baseline_policy(baseline_policy: object) -> str:
    context = getattr(baseline_policy, "decision_context", None)
    if not isinstance(context, Mapping):
        raise TypeError("stage20_m4_attempt13 requires stage19_p0 decision_context")
    expected = {
        "runtime_profile": "stage19_p0",
        "runtime_status": "p0_fixed",
        "fallback_policy": "stage18_p1",
        "t1_continuation": "stage18_p1",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
        "selective_override_only": True,
        "full_replacement_enabled": False,
    }
    changed = {
        name: (context.get(name), value)
        for name, value in expected.items()
        if context.get(name) != value
    }
    if changed:
        raise ValueError(f"stage20_m4_attempt13 fixed baseline chain changed: {changed}")
    if getattr(baseline_policy, "t3_continuation", None) != "stage7_m5_r10":
        raise ValueError("stage20_m4_attempt13 T3 continuation changed")
    seat = getattr(baseline_policy, "seat", None)
    if seat not in {"first", "second"}:
        raise ValueError("stage20_m4_attempt13 baseline seat is invalid")
    return str(seat)


def _called_by_complete_go_builder(
    bound_model: BoundHuM43Attempt13DistilledModel,
    capability: _Attempt13CompleteGoCapability | None,
) -> bool:
    if not isinstance(capability, _Attempt13CompleteGoCapability):
        return False
    frame = inspect.currentframe()
    try:
        init_frame = None if frame is None else frame.f_back
        caller = None if init_frame is None else init_frame.f_back
        return bool(
            caller is not None
            and caller.f_globals.get("__name__") == __name__
            and caller.f_code is build_stage20_m4_attempt13.__code__
            and caller.f_locals.get("bound_model") is bound_model
            and caller.f_locals.get("complete_go_capability") is capability
            and capability.is_valid()
        )
    finally:
        del frame


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} is invalid")
    return normalized


__all__ = [
    "ATTEMPT13_PROFILE_ID",
    "Attempt13SelectiveOverridePolicy",
    "build_stage20_m4_attempt13",
    "load_attempt13_complete_go",
]
