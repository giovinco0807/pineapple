"""Promotion-gated StreetPolicyNetV1 runtime for the M3.1 T3 candidate.

This module intentionally does not register a named profile.  The only public
factory replays a frozen Step6d population/ABR promotion receipt and all bound
artifacts before it returns an explicit opt-in wrapper.

At a T3 decision the wrapper:

* accepts only :class:`ActorObservation`;
* calls the unchanged ``stage7_m5_r10`` baseline exactly once;
* evaluates the complete legal ActionKey set in canonical order;
* applies the frozen delta/downside/disagreement/safety gate; and
* returns the exact baseline ``Action`` object on every valid non-fire.

The wrapper owns no RNG and stores no per-decision mutable state.  Promotion
evaluators that need the decision record can use
``choose_action_observation_with_audit`` and persist the returned immutable
record outside the policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import hu_m31_t3_promotion_runtime_closure_v1 as runtime_closure
from . import hu_m31_t3_street_policy_training_v1 as training
from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_infoset import ActorObservation
from .street_policy_net_v1 import (
    FEATURE_SCHEMA_HASH,
    LOSS_SCHEMA_HASH,
    MAX_LEGAL_ACTIONS,
    encode_street_policy_batch,
    model_state_sha256,
)


HU_M31_T3_RUNTIME_SCHEMA = "hu_m31_t3_street_policy_runtime_v1"
HU_M31_T3_DECISION_SCHEMA = "hu_m31_t3_street_policy_runtime_decision_v1"
HU_M31_T3_PROMOTION_RECEIPT_SCHEMA = (
    "hu_m31_t3_street_policy_qualified_promotion_receipt_v1"
)
HU_M31_T3_EVALUATION_RECEIPT_SCHEMA = (
    "hu_m31_t3_street_policy_locked_evaluation_receipt_v1"
)
OPT_IN_PROFILE_CANDIDATE = "stage7_m31_street_policy_v1_opt_in_candidate"
EVALUATION_ONLY_CANDIDATE = (
    "stage7_m31_street_policy_v1_locked_evaluation_only"
)
BASELINE_PROFILE = "stage7_m5_r10"
RUNTIME_SCOPE_EVALUATION_ONLY = "locked_population_abr_evaluation_only"
RUNTIME_SCOPE_QUALIFIED_OPT_IN = "qualified_explicit_opt_in_candidate"
EVALUATION_ONLY_WATERMARK = (
    "EVALUATION_ONLY_NOT_PROMOTION_EVIDENCE_NOT_RUNTIME_ACTIVATION"
)
_SHA256 = frozenset("0123456789abcdef")
_MODEL_OUTPUT_FIELDS = frozenset(
    {
        "policy_logits",
        "state_value",
        "action_q",
        "baseline_delta",
        "uncertainty_p95",
        "safe_logits",
        "safe_probability",
        "legal_action_mask",
        "baseline_indices",
    }
)
_FACTORY_TOKEN = object()


class HuM31T3RuntimeError(RuntimeError):
    """Raised when a frozen runtime or information-set binding changes."""


@dataclass(frozen=True)
class PromotionEvidencePaths:
    """Files needed to replay the locked population/ABR qualification."""

    plan_path: Path
    merge_path: Path
    gate_path: Path
    expected_gate_file_sha256: str
    compatibility_threshold_lock_path: Path
    policy_registry_path: Path
    evaluation_runtime_closure_path: Path


@dataclass(frozen=True)
class LockedEvaluationEvidencePaths:
    """Pre-promotion plan and files for the evaluation-only runtime."""

    plan_path: Path
    expected_plan_file_sha256: str
    compatibility_threshold_lock_path: Path
    policy_registry_path: Path
    evaluation_runtime_closure_path: Path


@dataclass(frozen=True)
class LockedEvaluationReceipt:
    """Source-replayed permission to evaluate, never to activate."""

    schema: str
    runtime_scope: str
    candidate_id: str
    audit_watermark: str
    plan_file_sha256: str
    plan_sha256: str
    model_manifest_file_sha256: str
    checkpoint_bundle_identity_sha256: str
    training_threshold_lock_file_sha256: str
    compatibility_threshold_lock_file_sha256: str
    policy_registry_file_sha256: str
    evaluation_runtime_closure_file_sha256: str
    seat_safe_probability_thresholds: Mapping[str, float]
    seat_enabled: Mapping[str, bool]
    plan_and_artifacts_source_replayed: bool
    evaluation_only: bool
    promotion_gate_required_for_evaluation: bool
    scientific_promotion_passed: bool
    separate_opt_in_profile_candidate_authorized: bool
    named_profile_added: bool
    current_profile_changed: bool
    runtime_activated: bool
    full_replacement_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedPromotionReceipt:
    """Source-replayed authorization for this exact runtime closure."""

    schema: str
    profile_candidate: str
    gate_file_sha256: str
    plan_sha256: str
    merge_sha256: str
    model_manifest_file_sha256: str
    checkpoint_bundle_identity_sha256: str
    training_threshold_lock_file_sha256: str
    compatibility_threshold_lock_file_sha256: str
    policy_registry_file_sha256: str
    evaluation_runtime_closure_file_sha256: str
    seat_safe_probability_thresholds: Mapping[str, float]
    seat_enabled: Mapping[str, bool]
    population_and_abr_source_replayed: bool
    scientific_promotion_passed: bool
    separate_opt_in_profile_candidate_authorized: bool
    named_profile_added: bool
    current_profile_changed: bool
    runtime_activated: bool
    full_replacement_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HuM31T3RuntimeDecision:
    """Immutable audit record returned separately from the gameplay action."""

    schema: str
    runtime_schema: str
    runtime_scope: str
    evaluation_only: bool
    audit_watermark: str | None
    profile_candidate: str
    baseline_profile: str
    observation_fingerprint: str
    seat: str
    action_key_schema: str
    legal_action_count: int
    legal_action_set_sha256: str
    canonical_action_mapping_sha256: str
    baseline_action_key: str
    candidate_action_key: str
    final_action_key: str
    baseline_index: int
    candidate_index: int
    predicted_delta: float
    downside_p95: float
    ensemble_disagreement: float
    safe_probability: float
    safe_probability_threshold: float
    lower_bound: float
    override_fired: bool
    nonfire_reason: str | None
    feature_schema_hash: str
    loss_schema_hash: str
    training_config_sha256: str
    checkpoint_bundle_identity_sha256: str
    threshold_lock_sha256: str
    promotion_gate_file_sha256: str | None
    model_state_sha256: tuple[str, ...]
    teacher_values_used: bool
    opponent_private_discards_used: bool
    current_profile_resolved: bool
    runtime_activated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_canonical_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HuM31T3RuntimeError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuM31T3RuntimeError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise HuM31T3RuntimeError(f"{label} is not a canonical object")
    return value, raw


def _require_complete_pass(gate: Mapping[str, Any]) -> None:
    gates = gate.get("gates")
    if (
        gate.get("status") != "pass"
        or gate.get("all_gates_passed") is not True
        or gate.get("scientific_promotion_passed") is not True
        or gate.get("separate_opt_in_profile_candidate_authorized") is not True
        or gate.get("teacher_values_used") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or gate.get("named_profile_added") is not False
        or gate.get("current_profile_changed") is not False
        or gate.get("runtime_activated") is not False
        or gate.get("full_replacement_enabled") is not False
    ):
        raise PermissionError(
            "M3.1 T3 runtime requires a fully passing locked population/ABR "
            "receipt; activation remains closed"
        )


def _validate_compatibility_lock(
    compatibility_lock: Mapping[str, Any],
    *,
    training_threshold_lock: Mapping[str, Any],
    training_threshold_lock_file_sha256: str,
    checkpoint_bundle_identity_sha256: str,
) -> tuple[dict[str, float], dict[str, bool]]:
    if compatibility_lock.get(
        "state_action_input_schema_sha256"
    ) != FEATURE_SCHEMA_HASH:
        raise HuM31T3RuntimeError(
            "promotion threshold compatibility lock has another feature schema"
        )
    training_seats = training_threshold_lock["seat_thresholds"]
    expected_seat_thresholds = {
        seat: float(training_seats[seat]["safe_probability_threshold"])
        for seat in training.SEATS
    }
    expected_seat_enabled = {
        seat: bool(training_seats[seat]["enabled"])
        for seat in training.SEATS
    }
    if (
        compatibility_lock.get("seat_thresholds")
        != expected_seat_thresholds
        or compatibility_lock.get("seat_enabled") != expected_seat_enabled
        or compatibility_lock.get(
            "source_training_threshold_lock_sha256"
        )
        != training_threshold_lock_file_sha256
        or compatibility_lock.get(
            "source_checkpoint_bundle_identity_sha256"
        )
        != checkpoint_bundle_identity_sha256
    ):
        raise HuM31T3RuntimeError(
            "promotion compatibility lock differs from the rich threshold "
            "lock or checkpoint bundle"
        )
    return expected_seat_thresholds, expected_seat_enabled


def _replay_runtime_closure(
    *,
    closure_path: str | Path,
    validated_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replay the self-contained source/model/native execution closure."""

    binding = validated_plan["artifact_binding"]
    try:
        manifest = runtime_closure.validate_runtime_closure_package(
            closure_path,
            expected_sha256=binding["evaluation_runtime_closure"][
                "sha256"
            ],
            source_replay_root=Path(__file__).resolve().parents[2],
        )
    except (
        OSError,
        TypeError,
        ValueError,
        runtime_closure.PromotionRuntimeClosureError,
    ) as exc:
        raise HuM31T3RuntimeError(
            "evaluation runtime closure package failed semantic source replay"
        ) from exc
    candidate = manifest["candidate_artifacts"]
    entries = manifest["entries"]
    registry_entry = entries.get("src/ofc_regular/ai_profiles.py")
    runtime_entry = entries.get(
        "src/ofc_regular/hu_m31_t3_street_policy_runtime_v1.py"
    )
    if (
        manifest["schema"] != runtime_closure.CLOSURE_SCHEMA
        or candidate["checkpoint_manifest_sha256"]
        != binding["model"]["sha256"]
        or registry_entry is None
        or registry_entry["sha256"]
        != binding["policy_registry"]["sha256"]
        or runtime_entry is None
        or runtime_entry["sha256"] != _sha256_file(Path(__file__).resolve())
        or manifest["schema_bindings"]["candidate_runtime_schema"]
        != HU_M31_T3_RUNTIME_SCHEMA
        or manifest["factory_contract"]["baseline"]["profile_id"]
        != BASELINE_PROFILE
        or manifest["factory_contract"]["candidate"]["candidate_id"]
        != EVALUATION_ONLY_CANDIDATE
        or manifest["factory_contract"]["current_profile_allowed"] is not False
        or manifest["factory_contract"][
            "implicit_profile_resolution_allowed"
        ]
        is not False
        or manifest["information_safety"]["actor_observation_only"]
        is not True
        or manifest["information_safety"]["opponent_private_discards_used"]
        is not False
    ):
        raise HuM31T3RuntimeError(
            "evaluation runtime closure is bound to another policy/runtime"
        )
    return manifest


def _load_locked_evaluation_receipt(
    *,
    evidence: LockedEvaluationEvidencePaths,
    model_manifest_path: Path,
    training_threshold_lock: Mapping[str, Any],
    training_threshold_lock_file_sha256: str,
    checkpoint_bundle_identity_sha256: str,
) -> LockedEvaluationReceipt:
    """Replay the frozen pre-promotion plan without claiming promotion."""

    if (
        not _is_sha256(evidence.expected_plan_file_sha256)
        or not _is_sha256(training_threshold_lock_file_sha256)
        or not _is_sha256(checkpoint_bundle_identity_sha256)
    ):
        raise ValueError(
            "evaluation plan, training threshold, and checkpoint bundle "
            "identities must be pinned"
        )
    plan, plan_raw = _read_canonical_object(
        Path(evidence.plan_path), "locked promotion plan"
    )
    plan_file_sha = hashlib.sha256(plan_raw).hexdigest()
    if plan_file_sha != evidence.expected_plan_file_sha256:
        raise HuM31T3RuntimeError(
            "pinned locked-evaluation plan file SHA-256 changed"
        )
    try:
        validated_plan = promotion.validate_locked_promotion_plan(plan)
        promotion.validate_artifact_files(
            validated_plan,
            model_path=model_manifest_path,
            threshold_lock_path=evidence.compatibility_threshold_lock_path,
            policy_registry_path=evidence.policy_registry_path,
            evaluation_runtime_closure_path=(
                evidence.evaluation_runtime_closure_path
            ),
        )
    except (OSError, TypeError, ValueError, PermissionError) as exc:
        raise HuM31T3RuntimeError(
            "locked evaluation plan/artifacts failed source replay"
        ) from exc
    if (
        validated_plan["evaluation_contract"]["baseline_profile"]
        != BASELINE_PROFILE
        or validated_plan["named_profile_added"] is not False
        or validated_plan["current_profile_changed"] is not False
        or validated_plan["runtime_activated"] is not False
        or validated_plan["full_replacement_enabled"] is not False
        or validated_plan["cloud_execution_started"] is not False
    ):
        raise HuM31T3RuntimeError(
            "locked evaluation plan crossed its pre-promotion boundary"
        )
    _replay_runtime_closure(
        closure_path=evidence.evaluation_runtime_closure_path,
        validated_plan=validated_plan,
    )
    binding = validated_plan["artifact_binding"]
    seat_thresholds, seat_enabled = _validate_compatibility_lock(
        binding["threshold_lock_content"],
        training_threshold_lock=training_threshold_lock,
        training_threshold_lock_file_sha256=(
            training_threshold_lock_file_sha256
        ),
        checkpoint_bundle_identity_sha256=(
            checkpoint_bundle_identity_sha256
        ),
    )
    return LockedEvaluationReceipt(
        schema=HU_M31_T3_EVALUATION_RECEIPT_SCHEMA,
        runtime_scope=RUNTIME_SCOPE_EVALUATION_ONLY,
        candidate_id=EVALUATION_ONLY_CANDIDATE,
        audit_watermark=EVALUATION_ONLY_WATERMARK,
        plan_file_sha256=plan_file_sha,
        plan_sha256=promotion.canonical_sha256(validated_plan),
        model_manifest_file_sha256=binding["model"]["sha256"],
        checkpoint_bundle_identity_sha256=(
            checkpoint_bundle_identity_sha256
        ),
        training_threshold_lock_file_sha256=(
            training_threshold_lock_file_sha256
        ),
        compatibility_threshold_lock_file_sha256=binding[
            "threshold_lock"
        ]["sha256"],
        policy_registry_file_sha256=binding["policy_registry"]["sha256"],
        evaluation_runtime_closure_file_sha256=binding[
            "evaluation_runtime_closure"
        ]["sha256"],
        seat_safe_probability_thresholds=seat_thresholds,
        seat_enabled=seat_enabled,
        plan_and_artifacts_source_replayed=True,
        evaluation_only=True,
        promotion_gate_required_for_evaluation=False,
        scientific_promotion_passed=False,
        separate_opt_in_profile_candidate_authorized=False,
        named_profile_added=False,
        current_profile_changed=False,
        runtime_activated=False,
        full_replacement_enabled=False,
    )


def _load_qualified_promotion_receipt(
    *,
    evidence: PromotionEvidencePaths,
    model_manifest_path: Path,
    training_threshold_lock: Mapping[str, Any],
    training_threshold_lock_file_sha256: str,
    checkpoint_bundle_identity_sha256: str,
) -> QualifiedPromotionReceipt:
    """Replay the pinned gate, source shards, and every bound runtime file."""

    if (
        not _is_sha256(evidence.expected_gate_file_sha256)
        or not _is_sha256(training_threshold_lock_file_sha256)
        or not _is_sha256(checkpoint_bundle_identity_sha256)
    ):
        raise ValueError(
            "promotion gate, training threshold, and checkpoint bundle "
            "identities must be pinned"
        )
    plan, _plan_raw = _read_canonical_object(
        Path(evidence.plan_path), "locked promotion plan"
    )
    merge, _merge_raw = _read_canonical_object(
        Path(evidence.merge_path), "locked promotion merge"
    )
    gate, gate_raw = _read_canonical_object(
        Path(evidence.gate_path), "locked promotion gate"
    )
    gate_file_sha = hashlib.sha256(gate_raw).hexdigest()
    if gate_file_sha != evidence.expected_gate_file_sha256:
        raise HuM31T3RuntimeError("pinned promotion gate file SHA-256 changed")
    try:
        validated_gate = promotion.validate_locked_promotion_gate(
            gate,
            plan=plan,
            merge=merge,
            replay_sources=True,
        )
        promotion.validate_artifact_files(
            plan,
            model_path=model_manifest_path,
            threshold_lock_path=evidence.compatibility_threshold_lock_path,
            policy_registry_path=evidence.policy_registry_path,
            evaluation_runtime_closure_path=(
                evidence.evaluation_runtime_closure_path
            ),
        )
    except (OSError, TypeError, ValueError, PermissionError) as exc:
        raise HuM31T3RuntimeError(
            "locked promotion evidence failed source replay"
        ) from exc
    _require_complete_pass(validated_gate)

    validated_plan = promotion.validate_locked_promotion_plan(plan)
    _replay_runtime_closure(
        closure_path=evidence.evaluation_runtime_closure_path,
        validated_plan=validated_plan,
    )
    binding = validated_plan["artifact_binding"]
    expected_seat_thresholds, expected_seat_enabled = (
        _validate_compatibility_lock(
            binding["threshold_lock_content"],
            training_threshold_lock=training_threshold_lock,
            training_threshold_lock_file_sha256=(
                training_threshold_lock_file_sha256
            ),
            checkpoint_bundle_identity_sha256=(
                checkpoint_bundle_identity_sha256
            ),
        )
    )
    return QualifiedPromotionReceipt(
        schema=HU_M31_T3_PROMOTION_RECEIPT_SCHEMA,
        profile_candidate=OPT_IN_PROFILE_CANDIDATE,
        gate_file_sha256=gate_file_sha,
        plan_sha256=promotion.canonical_sha256(validated_plan),
        merge_sha256=promotion.canonical_sha256(merge),
        model_manifest_file_sha256=binding["model"]["sha256"],
        checkpoint_bundle_identity_sha256=(
            checkpoint_bundle_identity_sha256
        ),
        training_threshold_lock_file_sha256=(
            training_threshold_lock_file_sha256
        ),
        compatibility_threshold_lock_file_sha256=binding[
            "threshold_lock"
        ]["sha256"],
        policy_registry_file_sha256=binding["policy_registry"]["sha256"],
        evaluation_runtime_closure_file_sha256=binding[
            "evaluation_runtime_closure"
        ]["sha256"],
        seat_safe_probability_thresholds=expected_seat_thresholds,
        seat_enabled=expected_seat_enabled,
        population_and_abr_source_replayed=True,
        scientific_promotion_passed=True,
        separate_opt_in_profile_candidate_authorized=True,
        named_profile_added=False,
        current_profile_changed=False,
        runtime_activated=False,
        full_replacement_enabled=False,
    )


class HuM31T3StreetPolicyRuntime:
    """T3-only selective override over an unchanged Stage7 baseline."""

    def __init__(
        self,
        *,
        baseline_policy: object,
        torch: Any,
        models: Sequence[Any],
        training_config: training.StreetPolicyTrainingConfig,
        threshold_lock_bytes: bytes,
        checkpoint_manifest: Mapping[str, Any],
        authorization_receipt: (
            LockedEvaluationReceipt | QualifiedPromotionReceipt
        ),
        runtime_scope: str,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise PermissionError(
                "construct M3.1 T3 runtime only through the promotion-gated "
                "opt-in factory"
            )
        chooser = getattr(baseline_policy, "choose_action_observation", None)
        if not callable(chooser):
            raise TypeError(
                "baseline_policy must implement choose_action_observation"
            )
        if len(models) != training_config.ensemble_size:
            raise ValueError("runtime ensemble size differs from training config")
        if runtime_scope not in {
            RUNTIME_SCOPE_EVALUATION_ONLY,
            RUNTIME_SCOPE_QUALIFIED_OPT_IN,
        }:
            raise ValueError("unknown M3.1 T3 runtime scope")
        if (
            runtime_scope == RUNTIME_SCOPE_EVALUATION_ONLY
            and not isinstance(
                authorization_receipt, LockedEvaluationReceipt
            )
        ) or (
            runtime_scope == RUNTIME_SCOPE_QUALIFIED_OPT_IN
            and not isinstance(
                authorization_receipt, QualifiedPromotionReceipt
            )
        ):
            raise PermissionError(
                "M3.1 T3 runtime scope/authorization receipt mismatch"
            )
        self._baseline_policy = baseline_policy
        self._torch = torch
        self._models = tuple(models)
        self._training_config = training_config
        self._threshold_lock_bytes = bytes(threshold_lock_bytes)
        self._checkpoint_manifest = dict(checkpoint_manifest)
        self._authorization_receipt = authorization_receipt
        self._runtime_scope = runtime_scope
        self._locked_model_hashes = tuple(
            str(record["model_state_sha256"])
            for record in checkpoint_manifest["models"]
        )
        for model in self._models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._baseline_policy, name)

    @property
    def runtime_id(self) -> str:
        return HU_M31_T3_RUNTIME_SCHEMA

    @property
    def profile_candidate(self) -> str:
        return (
            EVALUATION_ONLY_CANDIDATE
            if self.evaluation_only
            else OPT_IN_PROFILE_CANDIDATE
        )

    @property
    def promotion_receipt(self) -> QualifiedPromotionReceipt:
        if not isinstance(
            self._authorization_receipt, QualifiedPromotionReceipt
        ):
            raise PermissionError(
                "evaluation-only runtime has no promotion-qualified receipt"
            )
        return self._authorization_receipt

    @property
    def authorization_receipt(
        self,
    ) -> LockedEvaluationReceipt | QualifiedPromotionReceipt:
        return self._authorization_receipt

    @property
    def runtime_scope(self) -> str:
        return self._runtime_scope

    @property
    def evaluation_only(self) -> bool:
        return self._runtime_scope == RUNTIME_SCOPE_EVALUATION_ONLY

    def _threshold_lock(self) -> dict[str, Any]:
        try:
            value = json.loads(self._threshold_lock_bytes.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HuM31T3RuntimeError(
                "in-memory threshold lock bytes changed"
            ) from exc
        if not isinstance(value, dict):
            raise HuM31T3RuntimeError("in-memory threshold lock changed")
        return training._validate_threshold_lock(  # type: ignore[attr-defined]
            value,
            training_config=self._training_config,
            expected_dataset_identity_sha256=self._checkpoint_manifest[
                "training_view_identity_sha256"
            ],
            expected_model_hashes=self._locked_model_hashes,
        )

    def spawn_with_baseline(
        self,
        baseline_policy: object,
        *,
        baseline_profile_id: str,
    ) -> "HuM31T3StreetPolicyRuntime":
        """Share frozen models while giving one hand a fresh baseline policy."""

        if baseline_profile_id != BASELINE_PROFILE:
            raise ValueError(
                "spawned M3.1 runtime requires explicit stage7_m5_r10"
            )
        self._verify_model_state()
        return HuM31T3StreetPolicyRuntime(
            baseline_policy=baseline_policy,
            torch=self._torch,
            models=self._models,
            training_config=self._training_config,
            threshold_lock_bytes=self._threshold_lock_bytes,
            checkpoint_manifest=self._checkpoint_manifest,
            authorization_receipt=self._authorization_receipt,
            runtime_scope=self._runtime_scope,
            _factory_token=_FACTORY_TOKEN,
        )

    def _verify_model_state(self) -> None:
        observed = tuple(model_state_sha256(model) for model in self._models)
        if observed != self._locked_model_hashes:
            raise HuM31T3RuntimeError(
                "loaded StreetPolicyNetV1 model state no longer matches its lock"
            )

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        action, _decision = self.choose_action_observation_with_audit(
            observation,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
        )
        return action

    def choose_action_observation_with_audit(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> tuple[Action, HuM31T3RuntimeDecision | None]:
        if not isinstance(observation, ActorObservation):
            raise TypeError(
                "M3.1 T3 runtime requires ActorObservation; mappings, "
                "WorldState, replay truth, and hidden discards are forbidden"
            )
        expected_seat = getattr(self._baseline_policy, "seat", None)
        if expected_seat not in training.SEATS or observation.seat != expected_seat:
            raise HuM31T3RuntimeError(
                "observation seat does not match the explicit baseline policy"
            )
        if observation.street != "T3":
            return (
                self._baseline_policy.choose_action_observation(
                    observation,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                ),
                None,
            )

        # Integrity checks perform no inference and consume no RNG.  A tampered
        # runtime aborts before the gameplay baseline is advanced.
        self._verify_model_state()
        threshold_lock = self._threshold_lock()

        # This is the one and only baseline call for a valid T3 decision.
        baseline_action = self._baseline_policy.choose_action_observation(
            observation,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
        )
        try:
            baseline_key = action_key(baseline_action)
        except (AttributeError, TypeError, ValueError) as exc:
            raise HuM31T3RuntimeError(
                "stage7_m5_r10 returned an invalid Action"
            ) from exc

        actions = canonicalize_actions(
            generate_turn_actions(
                observation.hero_board, observation.dealt_cards
            )
        )
        if not actions or len(actions) > MAX_LEGAL_ACTIONS:
            raise HuM31T3RuntimeError("T3 complete legal action set changed")
        keys = tuple(action_key(action) for action in actions)
        try:
            baseline_index = keys.index(baseline_key)
        except ValueError as exc:
            raise HuM31T3RuntimeError(
                "stage7_m5_r10 ActionKey is not legal at the observation"
            ) from exc

        encoded = encode_street_policy_batch(
            [observation],
            [keys],
            [baseline_key],
            require_complete_legal_set=True,
        )
        encoded_tokens = tuple(
            token
            for token in encoded.action_key_tokens[0]
            if token is not None
        )
        expected_tokens = tuple(key.to_token() for key in keys)
        if (
            encoded_tokens != expected_tokens
            or int(encoded.baseline_indices[0]) != baseline_index
            or int(encoded.legal_action_mask[0].sum()) != len(actions)
        ):
            raise HuM31T3RuntimeError(
                "canonical ActionKey/index/mask mapping changed"
            )

        prediction = self._predict(
            observation=observation,
            encoded=encoded,
            baseline_key=baseline_key,
            baseline_index=baseline_index,
            legal_count=len(actions),
        )
        override_fired = training.safe_override_decision(
            prediction,
            threshold_lock,
            training_config=self._training_config,
        )
        candidate_action = actions[prediction.action_index]
        if action_key(candidate_action).to_token() != prediction.action_key:
            raise HuM31T3RuntimeError(
                "candidate ActionKey no longer resolves to its canonical index"
            )
        final_action = candidate_action if override_fired else baseline_action
        final_key = (
            prediction.action_key
            if override_fired
            else baseline_key.to_token()
        )
        seat_gate = threshold_lock["seat_thresholds"][observation.seat]
        if override_fired:
            nonfire_reason = None
        elif prediction.action_index == baseline_index:
            nonfire_reason = "candidate_matches_baseline"
        elif prediction.lower_bound <= 0.0:
            nonfire_reason = "nonpositive_risk_adjusted_delta"
        elif seat_gate["enabled"] is not True:
            nonfire_reason = "seat_threshold_disabled"
        else:
            nonfire_reason = "safe_probability_below_locked_threshold"
        decision = HuM31T3RuntimeDecision(
            schema=HU_M31_T3_DECISION_SCHEMA,
            runtime_schema=HU_M31_T3_RUNTIME_SCHEMA,
            runtime_scope=self.runtime_scope,
            evaluation_only=self.evaluation_only,
            audit_watermark=(
                EVALUATION_ONLY_WATERMARK
                if self.evaluation_only
                else None
            ),
            profile_candidate=self.profile_candidate,
            baseline_profile=BASELINE_PROFILE,
            observation_fingerprint=observation.fingerprint(),
            seat=observation.seat,
            action_key_schema=ACTION_KEY_SCHEMA,
            legal_action_count=len(actions),
            legal_action_set_sha256=legal_action_set_digest(actions),
            canonical_action_mapping_sha256=ordered_action_mapping_digest(
                actions
            ),
            baseline_action_key=baseline_key.to_token(),
            candidate_action_key=prediction.action_key,
            final_action_key=final_key,
            baseline_index=baseline_index,
            candidate_index=prediction.action_index,
            predicted_delta=prediction.predicted_delta,
            downside_p95=prediction.downside_p95,
            ensemble_disagreement=prediction.ensemble_disagreement,
            safe_probability=prediction.safe_probability,
            safe_probability_threshold=float(
                seat_gate["safe_probability_threshold"]
            ),
            lower_bound=prediction.lower_bound,
            override_fired=override_fired,
            nonfire_reason=nonfire_reason,
            feature_schema_hash=FEATURE_SCHEMA_HASH,
            loss_schema_hash=LOSS_SCHEMA_HASH,
            training_config_sha256=self._training_config.identity_sha256,
            checkpoint_bundle_identity_sha256=self._checkpoint_manifest[
                "bundle_identity_sha256"
            ],
            threshold_lock_sha256=threshold_lock["threshold_lock_sha256"],
            promotion_gate_file_sha256=(
                None
                if self.evaluation_only
                else self.promotion_receipt.gate_file_sha256
            ),
            model_state_sha256=self._locked_model_hashes,
            teacher_values_used=False,
            opponent_private_discards_used=False,
            current_profile_resolved=False,
            runtime_activated=False,
        )
        if not override_fired and final_action is not baseline_action:
            raise AssertionError(
                "M3.1 non-fire must preserve baseline Action object identity"
            )
        return final_action, decision

    def _predict(
        self,
        *,
        observation: ActorObservation,
        encoded: Any,
        baseline_key: ActionKey,
        baseline_index: int,
        legal_count: int,
    ) -> training.GatePrediction:
        inputs = encoded.to_torch(self._torch, device="cpu")
        outputs: list[Mapping[str, Any]] = []
        for model in self._models:
            model.eval()
            with self._torch.inference_mode():
                output = model(**inputs)
            if not isinstance(output, Mapping) or set(output) != _MODEL_OUTPUT_FIELDS:
                raise HuM31T3RuntimeError(
                    "StreetPolicyNetV1 output field set changed"
                )
            if not bool(
                self._torch.equal(
                    output["legal_action_mask"],
                    inputs["legal_action_mask"],
                )
            ) or not bool(
                self._torch.equal(
                    output["baseline_indices"],
                    inputs["baseline_indices"],
                )
            ):
                raise HuM31T3RuntimeError(
                    "StreetPolicyNetV1 returned another legal mask or baseline index"
                )
            outputs.append(output)

        deltas = np.stack(
            [
                output["baseline_delta"].detach().cpu().numpy()
                for output in outputs
            ],
            axis=0,
        )
        downsides = np.stack(
            [
                output["uncertainty_p95"].detach().cpu().numpy()
                for output in outputs
            ],
            axis=0,
        )
        safe_probabilities = np.stack(
            [
                output["safe_probability"].detach().cpu().numpy()
                for output in outputs
            ],
            axis=0,
        )
        mean_delta = deltas.mean(axis=0)[0, :legal_count]
        disagreement = deltas.std(axis=0, ddof=0)[0, :legal_count]
        mean_downside = downsides.mean(axis=0)[0, :legal_count]
        mean_safe = safe_probabilities.mean(axis=0)[0, :legal_count]
        # Legal actions occupy a canonical prefix.  numpy's first-index tie
        # break is therefore the frozen ActionKey.sort_key tie break.
        candidate_index = int(np.argmax(mean_delta))
        predicted_delta = float(mean_delta[candidate_index])
        downside = float(mean_downside[candidate_index])
        disagreement_value = float(disagreement[candidate_index])
        safe_probability = float(mean_safe[candidate_index])
        lower_bound = (
            predicted_delta
            - self._training_config.downside_multiplier * downside
            - self._training_config.disagreement_multiplier
            * disagreement_value
        )
        numeric = (
            predicted_delta,
            downside,
            disagreement_value,
            safe_probability,
            lower_bound,
        )
        if (
            any(not math.isfinite(value) for value in numeric)
            or downside < 0.0
            or disagreement_value < 0.0
            or not 0.0 <= safe_probability <= 1.0
        ):
            raise HuM31T3RuntimeError(
                "StreetPolicyNetV1 gate output is non-finite or out of range"
            )
        candidate_token = encoded.action_key_tokens[0][candidate_index]
        if not isinstance(candidate_token, str):
            raise HuM31T3RuntimeError(
                "candidate index resolved to a padded ActionKey slot"
            )
        return training.GatePrediction(
            example_identity=observation.fingerprint(),
            seat=observation.seat,
            action_key=candidate_token,
            action_index=candidate_index,
            baseline_action_key=baseline_key.to_token(),
            baseline_index=baseline_index,
            predicted_delta=predicted_delta,
            downside_p95=downside,
            ensemble_disagreement=disagreement_value,
            safe_probability=safe_probability,
            lower_bound=lower_bound,
        )

    def choose_action(
        self, board: Any, dealt_cards: Sequence[str], **kwargs: Any
    ) -> Action:
        if getattr(board, "card_count", lambda: -1)() == 9:
            raise HuM31T3RuntimeError(
                "M3.1 T3 runtime requires choose_action_observation; legacy "
                "dead_cards cannot prove the information-set boundary"
            )
        chooser = getattr(self._baseline_policy, "choose_action", None)
        if not callable(chooser):
            raise TypeError("baseline policy has no legacy choose_action")
        return chooser(board, dealt_cards, **kwargs)


@dataclass(frozen=True)
class _LoadedRuntimeArtifacts:
    models: tuple[Any, ...]
    checkpoint_manifest: Mapping[str, Any]
    threshold_lock: Mapping[str, Any]
    threshold_lock_bytes: bytes
    threshold_lock_file_sha256: str
    model_manifest_path: Path


def _load_runtime_artifacts(
    *,
    baseline_profile_id: str,
    torch: Any,
    checkpoint_bundle_path: str | Path,
    expected_dataset_identity_sha256: str,
    training_config: training.StreetPolicyTrainingConfig,
    expected_bundle_identity_sha256: str,
    threshold_lock_path: str | Path,
    expected_threshold_lock_file_sha256: str,
) -> _LoadedRuntimeArtifacts:
    if baseline_profile_id != BASELINE_PROFILE:
        raise ValueError(
            "M3.1 runtime baseline must be explicit stage7_m5_r10, never "
            "current or an implicit profile"
        )
    if not _is_sha256(expected_dataset_identity_sha256):
        raise ValueError("expected training dataset identity must be pinned")
    if not _is_sha256(expected_bundle_identity_sha256):
        raise ValueError("expected checkpoint bundle identity must be pinned")
    if not _is_sha256(expected_threshold_lock_file_sha256):
        raise ValueError("expected threshold-lock file SHA-256 must be pinned")

    bundle_path = Path(checkpoint_bundle_path)
    models, checkpoint_manifest = training.load_ensemble_checkpoint_bundle(
        bundle_path,
        torch=torch,
        expected_dataset_identity_sha256=expected_dataset_identity_sha256,
        expected_training_config=training_config,
        expected_stage="risk",
        expected_bundle_identity_sha256=expected_bundle_identity_sha256,
    )
    if checkpoint_manifest["completed_epoch"] != training_config.risk_epochs:
        raise HuM31T3RuntimeError(
            "runtime requires the fully completed risk-stage checkpoint"
        )
    model_hashes = [model_state_sha256(model) for model in models]
    declared_hashes = [
        str(record["model_state_sha256"])
        for record in checkpoint_manifest["models"]
    ]
    if model_hashes != declared_hashes:
        raise HuM31T3RuntimeError(
            "loaded model state differs from checkpoint bundle manifest"
        )

    threshold_path = Path(threshold_lock_path)
    threshold_lock, threshold_raw = _read_canonical_object(
        threshold_path, "StreetPolicyNetV1 threshold lock"
    )
    threshold_file_sha = hashlib.sha256(threshold_raw).hexdigest()
    if threshold_file_sha != expected_threshold_lock_file_sha256:
        raise HuM31T3RuntimeError(
            "pinned StreetPolicyNetV1 threshold-lock file SHA-256 changed"
        )
    threshold_lock = training._validate_threshold_lock(  # type: ignore[attr-defined]
        threshold_lock,
        training_config=training_config,
        expected_dataset_identity_sha256=expected_dataset_identity_sha256,
        expected_model_hashes=model_hashes,
    )
    if any(
        threshold_lock["seat_thresholds"][seat]["enabled"] is not True
        for seat in training.SEATS
    ):
        raise PermissionError(
            "both first and second seat thresholds must be enabled for the "
            "M3.1 both-seat candidate"
        )
    return _LoadedRuntimeArtifacts(
        models=tuple(models),
        checkpoint_manifest=checkpoint_manifest,
        threshold_lock=threshold_lock,
        threshold_lock_bytes=threshold_raw,
        threshold_lock_file_sha256=threshold_file_sha,
        model_manifest_path=bundle_path / "manifest.json",
    )


def build_locked_evaluation_t3_policy_candidate(
    *,
    baseline_policy: object,
    baseline_profile_id: str,
    torch: Any,
    checkpoint_bundle_path: str | Path,
    expected_dataset_identity_sha256: str,
    training_config: training.StreetPolicyTrainingConfig,
    expected_bundle_identity_sha256: str,
    threshold_lock_path: str | Path,
    expected_threshold_lock_file_sha256: str,
    evaluation_evidence: LockedEvaluationEvidencePaths,
) -> HuM31T3StreetPolicyRuntime:
    """Build the non-registered population/ABR evaluation-only candidate."""

    loaded = _load_runtime_artifacts(
        baseline_profile_id=baseline_profile_id,
        torch=torch,
        checkpoint_bundle_path=checkpoint_bundle_path,
        expected_dataset_identity_sha256=expected_dataset_identity_sha256,
        training_config=training_config,
        expected_bundle_identity_sha256=expected_bundle_identity_sha256,
        threshold_lock_path=threshold_lock_path,
        expected_threshold_lock_file_sha256=(
            expected_threshold_lock_file_sha256
        ),
    )
    receipt = _load_locked_evaluation_receipt(
        evidence=evaluation_evidence,
        model_manifest_path=loaded.model_manifest_path,
        training_threshold_lock=loaded.threshold_lock,
        training_threshold_lock_file_sha256=(
            loaded.threshold_lock_file_sha256
        ),
        checkpoint_bundle_identity_sha256=loaded.checkpoint_manifest[
            "bundle_identity_sha256"
        ],
    )
    if (
        receipt.schema != HU_M31_T3_EVALUATION_RECEIPT_SCHEMA
        or receipt.runtime_scope != RUNTIME_SCOPE_EVALUATION_ONLY
        or receipt.candidate_id != EVALUATION_ONLY_CANDIDATE
        or receipt.audit_watermark != EVALUATION_ONLY_WATERMARK
        or receipt.plan_and_artifacts_source_replayed is not True
        or receipt.evaluation_only is not True
        or receipt.promotion_gate_required_for_evaluation is not False
        or receipt.scientific_promotion_passed is not False
        or receipt.separate_opt_in_profile_candidate_authorized is not False
        or receipt.named_profile_added is not False
        or receipt.current_profile_changed is not False
        or receipt.runtime_activated is not False
        or receipt.full_replacement_enabled is not False
    ):
        raise PermissionError(
            "locked evaluation receipt crossed into promotion or activation"
        )
    _validate_receipt_artifact_binding(loaded, receipt)
    return HuM31T3StreetPolicyRuntime(
        baseline_policy=baseline_policy,
        torch=torch,
        models=loaded.models,
        training_config=training_config,
        threshold_lock_bytes=loaded.threshold_lock_bytes,
        checkpoint_manifest=loaded.checkpoint_manifest,
        authorization_receipt=receipt,
        runtime_scope=RUNTIME_SCOPE_EVALUATION_ONLY,
        _factory_token=_FACTORY_TOKEN,
    )


def _validate_receipt_artifact_binding(
    loaded: _LoadedRuntimeArtifacts,
    receipt: LockedEvaluationReceipt | QualifiedPromotionReceipt,
) -> None:
    manifest_file_sha = _sha256_file(loaded.model_manifest_path)
    if (
        receipt.model_manifest_file_sha256 != manifest_file_sha
        or receipt.checkpoint_bundle_identity_sha256
        != loaded.checkpoint_manifest["bundle_identity_sha256"]
        or receipt.training_threshold_lock_file_sha256
        != loaded.threshold_lock_file_sha256
    ):
        raise HuM31T3RuntimeError(
            "runtime receipt is bound to another checkpoint or rich "
            "threshold lock"
        )
    expected_seat_thresholds = {
        seat: float(
            loaded.threshold_lock["seat_thresholds"][seat][
                "safe_probability_threshold"
            ]
        )
        for seat in training.SEATS
    }
    if (
        dict(receipt.seat_safe_probability_thresholds)
        != expected_seat_thresholds
        or dict(receipt.seat_enabled)
        != {seat: True for seat in training.SEATS}
    ):
        raise HuM31T3RuntimeError(
            "runtime receipt is bound to another threshold or seat grid"
        )


def build_opt_in_t3_policy_candidate(
    *,
    baseline_policy: object,
    baseline_profile_id: str,
    torch: Any,
    checkpoint_bundle_path: str | Path,
    expected_dataset_identity_sha256: str,
    training_config: training.StreetPolicyTrainingConfig,
    expected_bundle_identity_sha256: str,
    threshold_lock_path: str | Path,
    expected_threshold_lock_file_sha256: str,
    promotion_evidence: PromotionEvidencePaths,
) -> HuM31T3StreetPolicyRuntime:
    """Build the opt-in candidate only after all frozen evidence replays."""

    loaded = _load_runtime_artifacts(
        baseline_profile_id=baseline_profile_id,
        torch=torch,
        checkpoint_bundle_path=checkpoint_bundle_path,
        expected_dataset_identity_sha256=expected_dataset_identity_sha256,
        training_config=training_config,
        expected_bundle_identity_sha256=expected_bundle_identity_sha256,
        threshold_lock_path=threshold_lock_path,
        expected_threshold_lock_file_sha256=(
            expected_threshold_lock_file_sha256
        ),
    )
    receipt = _load_qualified_promotion_receipt(
        evidence=promotion_evidence,
        model_manifest_path=loaded.model_manifest_path,
        training_threshold_lock=loaded.threshold_lock,
        training_threshold_lock_file_sha256=(
            loaded.threshold_lock_file_sha256
        ),
        checkpoint_bundle_identity_sha256=loaded.checkpoint_manifest[
            "bundle_identity_sha256"
        ],
    )
    if (
        receipt.schema != HU_M31_T3_PROMOTION_RECEIPT_SCHEMA
        or receipt.profile_candidate != OPT_IN_PROFILE_CANDIDATE
        or receipt.population_and_abr_source_replayed is not True
        or receipt.scientific_promotion_passed is not True
        or receipt.separate_opt_in_profile_candidate_authorized is not True
        or receipt.named_profile_added is not False
        or receipt.current_profile_changed is not False
        or receipt.runtime_activated is not False
        or receipt.full_replacement_enabled is not False
    ):
        raise PermissionError(
            "promotion receipt does not authorize only the explicit opt-in "
            "M3.1 T3 candidate"
        )
    _validate_receipt_artifact_binding(loaded, receipt)
    return HuM31T3StreetPolicyRuntime(
        baseline_policy=baseline_policy,
        torch=torch,
        models=loaded.models,
        training_config=training_config,
        threshold_lock_bytes=loaded.threshold_lock_bytes,
        checkpoint_manifest=loaded.checkpoint_manifest,
        authorization_receipt=receipt,
        runtime_scope=RUNTIME_SCOPE_QUALIFIED_OPT_IN,
        _factory_token=_FACTORY_TOKEN,
    )


__all__ = [
    "BASELINE_PROFILE",
    "EVALUATION_ONLY_CANDIDATE",
    "EVALUATION_ONLY_WATERMARK",
    "HU_M31_T3_DECISION_SCHEMA",
    "HU_M31_T3_EVALUATION_RECEIPT_SCHEMA",
    "HU_M31_T3_PROMOTION_RECEIPT_SCHEMA",
    "HU_M31_T3_RUNTIME_SCHEMA",
    "OPT_IN_PROFILE_CANDIDATE",
    "RUNTIME_SCOPE_EVALUATION_ONLY",
    "RUNTIME_SCOPE_QUALIFIED_OPT_IN",
    "HuM31T3RuntimeDecision",
    "HuM31T3RuntimeError",
    "HuM31T3StreetPolicyRuntime",
    "LockedEvaluationEvidencePaths",
    "LockedEvaluationReceipt",
    "PromotionEvidencePaths",
    "QualifiedPromotionReceipt",
    "build_locked_evaluation_t3_policy_candidate",
    "build_opt_in_t3_policy_candidate",
]
