"""Authenticated solve-root adapters for T3/T4 distillation teachers.

The sampled path is intentionally restricted to the supplied MCCFR root
observations for T3 BB, T3 BTN and T4 BB.  It requires a persisted checkpoint,
derives solver/payoff seeds from the pre-label split group, restores the exact
root visit count from the checkpoint, uses the exact root prior as reach, and
freshly replays every legal root action under the frozen checkpoint average
strategy.  No missing continuation fallback exists.

T4 BTN is a separate seedless/checkpointless exact branch.  Durable evidence
is algorithm-teacher-only and never changes serving or promotes a policy.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ai.tutor.runtime_semantic_anchor import (
    register_module_function_anchor,
    registered_module_function_anchor,
    verify_module_function_anchor,
)
from ai.tutor.t3_hu_full_card_mccfr import FullCardGenerativeAdapter
from ai.tutor.t3_hu_multi_root_mccfr import (
    MultiRootChanceEntry,
    MultiRootExternalSamplingMccfrResult,
    VerifiedMultiRootCheckpointSnapshot,
    verify_multi_root_checkpoint_against_result,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey
from ai.tutor.t3_hu_public_tree import (
    PendingChanceState,
    PublicTreeDecisionState,
    PublicTreeTerminalState,
)
from ai.tutor.t3_t4_distillation_teacher import (
    MCCFR_SOLVER_METHOD,
    MANIFEST_NAME as TEACHER_MANIFEST_FILENAME,
    PAYOFF_SEED_NAMESPACE,
    SOLVER_SEED_NAMESPACE,
    build_split_assignment,
    build_t4_btn_exact_teacher_row,
    build_teacher_row,
    canonical_json,
    canonical_sha256,
    derive_payoff_seed,
    derive_solver_seed,
    self_hash,
    verify_split_assignment,
    verify_teacher_bundle,
    verify_teacher_row,
    write_teacher_bundle,
)
from ai.tutor.t3_t4_infoset_encoder import semantic_action_ids
from ai.tutor.t3_t4_public_online_resolve import (
    OnlinePublicResolveResult,
    resolve_compiled_public_root_mixture,
    verify_online_public_resolve,
)
from ai.tutor.t3_t4_public_root_mixture import (
    CompiledPublicRootMixture,
    PublicRootContext,
    verify_compiled_public_root_mixture,
)
from ai.tutor.t4_btn_exact_resolver import (
    resolve_t4_second_btn_exact,
    verify_t4_second_btn_exact,
)


PRELABEL_PLAN_SCHEMA = "ofc_t3_t4_solve_root_prelabel_plan/v1"
SOLVE_ROOT_EVIDENCE_SCHEMA = "ofc_t3_t4_solve_root_teacher_evidence/v1"
EVIDENCE_BOUND_TEACHER_SCHEMA = "ofc_t3_t4_evidence_bound_teacher_bundle/v1"
PRELABEL_PLAN_ARTIFACT = "prelabel_solve_root_teacher_plan"
SOLVE_ROOT_EVIDENCE_ARTIFACT = "verified_solve_root_teacher_evidence"
PRELABEL_PLAN_FILENAME = "prelabel-plan.json"
EVIDENCE_MANIFEST_FILENAME = "solve-root-teacher-evidence.json"
EVIDENCE_BOUND_MANIFEST_FILENAME = "evidence-bound-teacher.json"

_SAMPLED_PHASE_ACTORS = frozenset({
    ("t3_first", "bb", 3),
    ("t3_second", "btn", 3),
    ("t4_first", "bb", 4),
})
_EXACT_PHASE_ACTOR = ("t4_second", "btn", 4)
_FLOAT_TOL = 1e-12

_SOURCE_PATHS = (
    "ai/tutor/t3_t4_solve_root_teacher_adapter.py",
    "ai/tutor/t3_hu_multi_root_mccfr.py",
    "ai/tutor/t3_hu_full_card_mccfr.py",
    "ai/tutor/t3_hu_full_card_range.py",
    "ai/tutor/t3_t4_public_root_mixture.py",
    "ai/tutor/t3_t4_public_online_resolve.py",
    "ai/tutor/t3_t4_distillation_teacher.py",
    "ai/tutor/t3_t4_infoset_encoder.py",
    "ai/tutor/t4_btn_exact_resolver.py",
    "ai/tutor/exact_late.py",
    "ai/engine/action_space.py",
    "ai/engine/encoding.py",
    "ai/engine/game_engine.py",
    "ai/engine/scoring.py",
    "ai/engine/turn_order.py",
    "ai/mcts/rollout_evaluator.py",
    "ai/config/fl_ev.json",
)


class SolveRootTeacherAdapterError(ValueError):
    """A pre-label plan, solver checkpoint, or teacher evidence failed."""


_VERIFIED_EVIDENCE_SEAL = (
    "ofc_t3_t4_verified_solve_root_evidence_handle/v1",
    "verifier-issued-only",
)


@dataclass(frozen=True, slots=True)
class PreLabelSolveRootTeacherPlan:
    manifest_json: str
    manifest_sha256: str

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.manifest_json))


@dataclass(frozen=True, slots=True)
class VerifiedSolveRootTeacherEvidence:
    root: Path
    manifest: Mapping[str, Any]
    teacher_row: Mapping[str, Any]
    _verification_seal: object = dataclasses.field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class VerifiedEvidenceBoundTeacherBundle:
    root: Path
    manifest: Mapping[str, Any]
    teacher_manifest: Mapping[str, Any]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SolveRootTeacherAdapterError(f"{label}: object required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise SolveRootTeacherAdapterError(
            f"{label}: exact fields required; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SolveRootTeacherAdapterError(f"{label}: lowercase SHA256 required")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SolveRootTeacherAdapterError(f"{label}: integer >= {minimum} required")
    if maximum is not None and value > maximum:
        raise SolveRootTeacherAdapterError(f"{label}: integer <= {maximum} required")
    return value


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.resolve(strict=True).read_bytes()).hexdigest()
    except OSError as exc:
        raise SolveRootTeacherAdapterError(f"cannot hash required file {path}") from exc


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _read_canonical_file(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise SolveRootTeacherAdapterError("artifact must be an existing regular file")
    for parent in (target.parent, *target.parent.parents):
        if parent.is_symlink():
            raise SolveRootTeacherAdapterError("artifact path traverses a symlink")
    try:
        raw = target.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise SolveRootTeacherAdapterError(f"cannot read artifact: {exc}") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise SolveRootTeacherAdapterError("artifact must contain one canonical JSON line")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise SolveRootTeacherAdapterError(f"duplicate JSON key {key!r}")
            output[key] = item
        return output

    try:
        value = json.loads(text[:-1], object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise SolveRootTeacherAdapterError("artifact is not valid JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != text[:-1]:
        raise SolveRootTeacherAdapterError("artifact is not canonical JSON")
    return value


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise SolveRootTeacherAdapterError("artifact target traverses a symlink")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise SolveRootTeacherAdapterError("existing artifact is unreadable") from exc
        if existing != payload:
            raise SolveRootTeacherAdapterError("immutable artifact already exists with other content")
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if path.read_bytes() != payload:
        raise SolveRootTeacherAdapterError("immutable artifact readback mismatch")


def _live_source_binding() -> dict[str, Any]:
    _require_runtime_contract()
    root = _repo_root()
    files = []
    for relative in _SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise SolveRootTeacherAdapterError(f"source binding missing {relative}")
        files.append({"path": relative, "sha256": _file_sha256(path)})
    payload: dict[str, Any] = {
        "schema": "ofc_t3_t4_solve_root_teacher_source/v1",
        "files": files,
        "runtime_semantic_anchor_verified": True,
        "promotion_eligible": False,
        "serving_changed": False,
    }
    payload["binding_sha256"] = self_hash(payload, "binding_sha256")
    return payload


def _compiled_mixture_binding(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
) -> dict[str, Any]:
    verify_compiled_public_root_mixture(context, mixture)
    payload: dict[str, Any] = {
        "schema": "ofc_t3_t4_teacher_compiled_mixture_binding/v1",
        "public_context_digest": context.digest(),
        "support_sha256": mixture.support_sha256,
        "audit_manifest_sha256": mixture.audit_manifest_sha256,
        "actual_private_hand_compiler_input": False,
    }
    payload["binding_sha256"] = self_hash(payload, "binding_sha256")
    return payload


def _select_root(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    root_id_sha256: str,
) -> tuple[int, MultiRootChanceEntry, str]:
    verify_compiled_public_root_mixture(context, mixture)
    wanted = _sha(root_id_sha256, label="root_id_sha256")
    matches = [
        (index, entry)
        for index, entry in enumerate(mixture.entries)
        if entry.root_id_sha256 == wanted
    ]
    if len(matches) != 1:
        raise SolveRootTeacherAdapterError("selected root is not exactly one compiled member")
    index, entry = matches[0]
    if type(entry.adapter) is not FullCardGenerativeAdapter:
        raise SolveRootTeacherAdapterError("teacher evidence requires canonical FullCard root")
    commitment = mixture.private_type_commitments[index]
    return index, entry, commitment


def _branch_for_key(key: InfoSetKey) -> str:
    identity = (key.phase, key.actor, key.turn)
    if identity in _SAMPLED_PHASE_ACTORS:
        return "sampled_mccfr"
    if identity == _EXACT_PHASE_ACTOR:
        return "t4_btn_exact"
    raise SolveRootTeacherAdapterError("unsupported T3/T4 solve-root phase/actor")


def _selected_root_payload(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    entry: MultiRootChanceEntry,
    private_type_commitment: str,
) -> dict[str, Any]:
    full_range = entry.adapter.root_range
    return {
        "public_context_digest": context.digest(),
        "public_root_commitment_sha256": context.digest(),
        "root_id_sha256": entry.root_id_sha256,
        "observation_sha256": entry.adapter.observation.digest(),
        "private_type_commitment_sha256": private_type_commitment,
        "prior_mass_exact": _fraction_text(entry.prior_mass),
        "range_content_sha256": full_range.range_content_sha256,
        "range_build_sha256": full_range.range_build_sha256,
        "behavior_model_sha256": full_range.behavior_model_sha256,
        "compiled_support_sha256": mixture.support_sha256,
    }


def _lineage_payload(
    *,
    assignment: Mapping[str, Any],
    selected_root: Mapping[str, Any],
    seat_swap_index: int,
    suit_augmentation_index: int,
) -> dict[str, Any]:
    seat = _integer(seat_swap_index, label="seat_swap_index", maximum=1)
    suit = _integer(
        suit_augmentation_index,
        label="suit_augmentation_index",
        maximum=23,
    )
    return {
        "descendant_public_path_sha256": canonical_sha256(
            {
                "schema": "ofc_t3_t4_solve_root_empty_descendant_path/v1",
                "public_root_family_commitment_sha256": assignment[
                    "public_root_family_commitment_sha256"
                ],
                "path": [],
            }
        ),
        "restricted_variant_root_commitment_sha256": canonical_sha256(
            {
                "schema": "ofc_t3_t4_restricted_variant_root/v1",
                "public_root_family_commitment_sha256": assignment[
                    "public_root_family_commitment_sha256"
                ],
                "observation_sha256": selected_root["observation_sha256"],
                "range_content_sha256": selected_root["range_content_sha256"],
                "range_build_sha256": selected_root["range_build_sha256"],
                "seat_swap_index": seat,
                "suit_augmentation_index": suit,
            }
        ),
        "restricted_private_type_commitment_sha256": selected_root[
            "private_type_commitment_sha256"
        ],
        "seat_swap_index": seat,
        "suit_augmentation_index": suit,
    }


def _plan_payload(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    *,
    root_id_sha256: str,
    full_deal_commitment_sha256: str,
    public_root_family_commitment_sha256: str,
    seat_swap_index: int,
    suit_augmentation_index: int,
    solver_seed_index: int | None,
    payoff_seed_index: int | None,
    solver_iterations: int | None,
    max_infosets: int | None,
    linear_averaging: bool | None,
    samples_per_action: int | None,
) -> dict[str, Any]:
    _index, entry, private_commitment = _select_root(
        context, mixture, root_id_sha256
    )
    assignment = build_split_assignment(
        full_deal_commitment_sha256=_sha(
            full_deal_commitment_sha256,
            label="full_deal_commitment_sha256",
        ),
        public_root_family_commitment_sha256=_sha(
            public_root_family_commitment_sha256,
            label="public_root_family_commitment_sha256",
        ),
    )
    selected_root = _selected_root_payload(
        context, mixture, entry, private_commitment
    )
    branch = _branch_for_key(entry.adapter.observation)
    lineage = _lineage_payload(
        assignment=assignment,
        selected_root=selected_root,
        seat_swap_index=seat_swap_index,
        suit_augmentation_index=suit_augmentation_index,
    )
    if branch == "sampled_mccfr":
        solver_index = _integer(
            solver_seed_index,
            label="solver_seed_index",
            maximum=(1 << 31) - 1,
        )
        payoff_index = _integer(
            payoff_seed_index,
            label="payoff_seed_index",
            maximum=(1 << 31) - 1,
        )
        iterations = _integer(
            solver_iterations, label="solver_iterations", minimum=1
        )
        maximum_infosets = _integer(
            max_infosets, label="max_infosets", minimum=1
        )
        sample_count = _integer(
            samples_per_action, label="samples_per_action", minimum=2
        )
        if not isinstance(linear_averaging, bool):
            raise SolveRootTeacherAdapterError("linear_averaging: bool required")
        group = assignment["split_group_sha256"]
        sampled_plan: dict[str, Any] | None = {
            "solver_seed_namespace": SOLVER_SEED_NAMESPACE,
            "solver_seed_index": solver_index,
            "derived_solver_seed": derive_solver_seed(group, solver_index),
            "payoff_seed_namespace": PAYOFF_SEED_NAMESPACE,
            "payoff_seed_index": payoff_index,
            "derived_payoff_seed": derive_payoff_seed(group, payoff_index),
            "solver_iterations": iterations,
            "max_infosets": maximum_infosets,
            "linear_averaging": linear_averaging,
            "samples_per_action": sample_count,
        }
    else:
        if any(
            value is not None
            for value in (
                solver_seed_index,
                payoff_seed_index,
                solver_iterations,
                max_infosets,
                linear_averaging,
                samples_per_action,
            )
        ):
            raise SolveRootTeacherAdapterError(
                "T4 BTN exact pre-label plans must not declare sampled provenance"
            )
        sampled_plan = None
    payload: dict[str, Any] = {
        "schema": PRELABEL_PLAN_SCHEMA,
        "artifact_kind": PRELABEL_PLAN_ARTIFACT,
        "assignment_before_labels": True,
        "label_fields_present": False,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "serving_changed": False,
        "split_assignment": assignment,
        "lineage": lineage,
        "selected_root": selected_root,
        "compiled_mixture": _compiled_mixture_binding(context, mixture),
        "label_branch": branch,
        "sampled_plan": sampled_plan,
    }
    payload["plan_sha256"] = self_hash(payload, "plan_sha256")
    return payload


def build_prelabel_solve_root_teacher_plan(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    *,
    root_id_sha256: str,
    full_deal_commitment_sha256: str,
    public_root_family_commitment_sha256: str,
    seat_swap_index: int = 0,
    suit_augmentation_index: int = 0,
    solver_seed_index: int | None = None,
    payoff_seed_index: int | None = None,
    solver_iterations: int | None = None,
    max_infosets: int | None = None,
    linear_averaging: bool | None = None,
    samples_per_action: int | None = None,
) -> PreLabelSolveRootTeacherPlan:
    """Create a label-free split/lineage/seed plan for one compiled root."""

    _require_runtime_contract()
    payload = _plan_payload(
        context,
        mixture,
        root_id_sha256=root_id_sha256,
        full_deal_commitment_sha256=full_deal_commitment_sha256,
        public_root_family_commitment_sha256=(
            public_root_family_commitment_sha256
        ),
        seat_swap_index=seat_swap_index,
        suit_augmentation_index=suit_augmentation_index,
        solver_seed_index=solver_seed_index,
        payoff_seed_index=payoff_seed_index,
        solver_iterations=solver_iterations,
        max_infosets=max_infosets,
        linear_averaging=linear_averaging,
        samples_per_action=samples_per_action,
    )
    manifest_json = canonical_json(payload)
    plan = PreLabelSolveRootTeacherPlan(
        manifest_json=manifest_json,
        manifest_sha256=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )
    return verify_prelabel_solve_root_teacher_plan(context, mixture, plan)


_PLAN_KEYS = {
    "schema",
    "artifact_kind",
    "assignment_before_labels",
    "label_fields_present",
    "promotion_eligible",
    "runtime_integrated",
    "serving_changed",
    "split_assignment",
    "lineage",
    "selected_root",
    "compiled_mixture",
    "label_branch",
    "sampled_plan",
    "plan_sha256",
}


def verify_prelabel_solve_root_teacher_plan(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    plan: PreLabelSolveRootTeacherPlan,
) -> PreLabelSolveRootTeacherPlan:
    _require_runtime_contract()
    if type(plan) is not PreLabelSolveRootTeacherPlan:
        raise TypeError("plan must be an exact PreLabelSolveRootTeacherPlan")
    try:
        raw = json.loads(plan.manifest_json)
    except json.JSONDecodeError as exc:
        raise SolveRootTeacherAdapterError("pre-label plan is not JSON") from exc
    if canonical_json(raw) != plan.manifest_json:
        raise SolveRootTeacherAdapterError("pre-label plan is not canonical JSON")
    digest = hashlib.sha256(plan.manifest_json.encode("utf-8")).hexdigest()
    if digest != plan.manifest_sha256:
        raise SolveRootTeacherAdapterError("pre-label plan manifest SHA256 mismatch")
    _exact_keys(raw, _PLAN_KEYS, label="prelabel_plan")
    fixed = {
        "schema": PRELABEL_PLAN_SCHEMA,
        "artifact_kind": PRELABEL_PLAN_ARTIFACT,
        "assignment_before_labels": True,
        "label_fields_present": False,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "serving_changed": False,
    }
    for field, expected in fixed.items():
        if raw.get(field) != expected:
            raise SolveRootTeacherAdapterError(f"prelabel_plan.{field}: mismatch")
    if raw.get("plan_sha256") != self_hash(raw, "plan_sha256"):
        raise SolveRootTeacherAdapterError("pre-label plan self-hash mismatch")
    assignment = verify_split_assignment(raw.get("split_assignment"))
    selected = _require_mapping(raw.get("selected_root"), label="selected_root")
    root_id = _sha(selected.get("root_id_sha256"), label="selected_root.root_id")
    sampled = raw.get("sampled_plan")
    if raw.get("label_branch") == "sampled_mccfr":
        sampled_map = _require_mapping(sampled, label="sampled_plan")
        rebuilt = _plan_payload(
            context,
            mixture,
            root_id_sha256=root_id,
            full_deal_commitment_sha256=assignment[
                "full_deal_commitment_sha256"
            ],
            public_root_family_commitment_sha256=assignment[
                "public_root_family_commitment_sha256"
            ],
            seat_swap_index=raw["lineage"]["seat_swap_index"],
            suit_augmentation_index=raw["lineage"]["suit_augmentation_index"],
            solver_seed_index=sampled_map.get("solver_seed_index"),
            payoff_seed_index=sampled_map.get("payoff_seed_index"),
            solver_iterations=sampled_map.get("solver_iterations"),
            max_infosets=sampled_map.get("max_infosets"),
            linear_averaging=sampled_map.get("linear_averaging"),
            samples_per_action=sampled_map.get("samples_per_action"),
        )
    elif raw.get("label_branch") == "t4_btn_exact":
        if sampled is not None:
            raise SolveRootTeacherAdapterError("exact pre-label plan has sampled fields")
        rebuilt = _plan_payload(
            context,
            mixture,
            root_id_sha256=root_id,
            full_deal_commitment_sha256=assignment[
                "full_deal_commitment_sha256"
            ],
            public_root_family_commitment_sha256=assignment[
                "public_root_family_commitment_sha256"
            ],
            seat_swap_index=raw["lineage"]["seat_swap_index"],
            suit_augmentation_index=raw["lineage"]["suit_augmentation_index"],
            solver_seed_index=None,
            payoff_seed_index=None,
            solver_iterations=None,
            max_infosets=None,
            linear_averaging=None,
            samples_per_action=None,
        )
    else:
        raise SolveRootTeacherAdapterError("pre-label plan branch is unknown")
    if raw != rebuilt:
        raise SolveRootTeacherAdapterError("pre-label plan does not match live root inputs")
    return PreLabelSolveRootTeacherPlan(plan.manifest_json, plan.manifest_sha256)


def write_prelabel_solve_root_teacher_plan(
    path: str | Path,
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    plan: PreLabelSolveRootTeacherPlan,
) -> Path:
    verified = verify_prelabel_solve_root_teacher_plan(context, mixture, plan)
    target = Path(path)
    _write_no_replace(target, (verified.manifest_json + "\n").encode("utf-8"))
    read_prelabel_solve_root_teacher_plan(target, context, mixture)
    return target


def read_prelabel_solve_root_teacher_plan(
    path: str | Path,
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
) -> PreLabelSolveRootTeacherPlan:
    raw = _read_canonical_file(path)
    text = canonical_json(raw)
    return verify_prelabel_solve_root_teacher_plan(
        context,
        mixture,
        PreLabelSolveRootTeacherPlan(
            manifest_json=text,
            manifest_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ),
    )


def resolve_prelabel_solve_root_plan(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    plan: PreLabelSolveRootTeacherPlan,
    *,
    checkpoint_path: str | Path,
) -> OnlinePublicResolveResult:
    """Run the whole public mixture using only the locked sampled plan."""

    verified = verify_prelabel_solve_root_teacher_plan(context, mixture, plan)
    manifest = verified.manifest
    if manifest["label_branch"] != "sampled_mccfr":
        raise SolveRootTeacherAdapterError("exact T4 BTN plans do not run MCCFR")
    sampled = manifest["sampled_plan"]
    return resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=sampled["solver_iterations"],
        seed=sampled["derived_solver_seed"],
        max_infosets=sampled["max_infosets"],
        linear_averaging=sampled["linear_averaging"],
        checkpoint_path=checkpoint_path,
    )


def _sample_public_policy(
    key: InfoSetKey,
    legal_action_ids: Sequence[str],
    distribution: Mapping[str, float],
    rng: random.Random,
) -> str:
    if not isinstance(key, InfoSetKey):
        raise TypeError("continuation policy key must be InfoSetKey")
    key.canonical_json()
    action_ids = tuple(str(action_id) for action_id in legal_action_ids)
    if action_ids != tuple(sorted(set(action_ids))) or set(distribution) != set(
        action_ids
    ):
        raise SolveRootTeacherAdapterError(
            "continuation policy/legal action support mismatch"
        )
    probabilities: list[tuple[str, float]] = []
    for action_id in action_ids:
        value = distribution[action_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SolveRootTeacherAdapterError("continuation probability is not numeric")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0:
            raise SolveRootTeacherAdapterError("continuation probability is invalid")
        probabilities.append((action_id, probability))
    total = math.fsum(probability for _action_id, probability in probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_FLOAT_TOL):
        raise SolveRootTeacherAdapterError(
            "continuation probabilities do not sum to one"
        )
    threshold = rng.random()
    cumulative = 0.0
    for action_id, probability in probabilities:
        cumulative += probability
        if threshold < cumulative:
            return action_id
    # The exact endpoint is unreachable for random.Random.random(); tolerate
    # only the final floating-point accumulation edge after full validation.
    return action_ids[-1]


def _physical_entropy_seed(
    *,
    split_group_sha256: str,
    payoff_seed: int,
    sample_index: int,
    stage: str,
    chance_index: int,
) -> int:
    return int(
        canonical_sha256(
            {
                "domain": "ofc_t3_t4_teacher_physical_chance/v1",
                "split_group_sha256": split_group_sha256,
                "payoff_seed": payoff_seed,
                "sample_index": sample_index,
                "stage": stage,
                "chance_index": chance_index,
            }
        ),
        16,
    )


def _policy_entropy_seed(
    *,
    split_group_sha256: str,
    payoff_seed: int,
    root_action_id: str,
    sample_index: int,
    decision_index: int,
    infoset_digest: str,
) -> int:
    return int(
        canonical_sha256(
            {
                "domain": "ofc_t3_t4_teacher_public_policy/v1",
                "split_group_sha256": split_group_sha256,
                "payoff_seed": payoff_seed,
                "root_action_id": root_action_id,
                "sample_index": sample_index,
                "decision_index": decision_index,
                "infoset_digest": infoset_digest,
            }
        ),
        16,
    )


def _rollout_root_action(
    *,
    adapter: FullCardGenerativeAdapter,
    root_action_id: str,
    root_actor: str,
    split_group_sha256: str,
    payoff_seed: int,
    sample_index: int,
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]],
    required_infosets: set[str],
) -> float:
    state: PublicTreeDecisionState | PendingChanceState | PublicTreeTerminalState
    state = adapter.sample_root_for_traversal(
        random.Random(
            _physical_entropy_seed(
                split_group_sha256=split_group_sha256,
                payoff_seed=payoff_seed,
                sample_index=sample_index,
                stage="root_posterior",
                chance_index=0,
            )
        )
    ).state
    state = adapter.apply_action_id(state, root_action_id)
    chance_index = 0
    decision_index = 0
    while not isinstance(state, PublicTreeTerminalState):
        if isinstance(state, PendingChanceState):
            state = adapter.sample_next_draw(
                state,
                random.Random(
                    _physical_entropy_seed(
                        split_group_sha256=split_group_sha256,
                        payoff_seed=payoff_seed,
                        sample_index=sample_index,
                        stage="future_draw",
                        chance_index=chance_index,
                    )
                ),
            ).state
            chance_index += 1
            continue
        if not isinstance(state, PublicTreeDecisionState):  # pragma: no cover
            raise AssertionError("unsupported physical teacher rollout state")
        key = adapter.information_key(state)
        required_infosets.add(key.digest())
        distribution = average_strategy.get(key)
        if distribution is None:
            raise SolveRootTeacherAdapterError(
                "frozen average strategy is missing an exact continuation InfoSetKey; "
                "fallback is forbidden"
            )
        legal_action_ids = tuple(
            action_id for action_id, _action in adapter.legal_actions(state)
        )
        selected = _sample_public_policy(
            key,
            legal_action_ids,
            distribution,
            random.Random(
                _policy_entropy_seed(
                    split_group_sha256=split_group_sha256,
                    payoff_seed=payoff_seed,
                    root_action_id=root_action_id,
                    sample_index=sample_index,
                    decision_index=decision_index,
                    infoset_digest=key.digest(),
                )
            ),
        )
        state = adapter.apply_action_id(state, selected)
        decision_index += 1
    utility_bb = float(adapter.terminal_utility_bb(state))
    if not math.isfinite(utility_bb):
        raise SolveRootTeacherAdapterError("terminal payoff is non-finite")
    return utility_bb if root_actor == "bb" else -utility_bb


def _evaluate_action_raw_moments(
    *,
    entry: MultiRootChanceEntry,
    split_group_sha256: str,
    payoff_seed: int,
    samples_per_action: int,
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]],
) -> dict[str, Any]:
    observation = entry.adapter.observation
    adapter = FullCardGenerativeAdapter(observation, entry.adapter.root_range)
    probe = adapter.sample_root_for_traversal(
        random.Random(
            _physical_entropy_seed(
                split_group_sha256=split_group_sha256,
                payoff_seed=payoff_seed,
                sample_index=0,
                stage="root_posterior",
                chance_index=0,
            )
        )
    )
    legal_action_ids = tuple(
        action_id for action_id, _action in adapter.legal_actions(probe.state)
    )
    semantic_legal = tuple(
        action_id
        for action_id in semantic_action_ids(observation)
        if action_id is not None
    )
    if set(legal_action_ids) != set(semantic_legal):
        raise SolveRootTeacherAdapterError(
            "physical and semantic legal root actions do not match"
        )
    root_policy = average_strategy.get(observation)
    if root_policy is None or set(root_policy) != set(legal_action_ids):
        raise SolveRootTeacherAdapterError(
            "checkpoint average strategy lacks the exact solve-root policy"
        )
    adapter = FullCardGenerativeAdapter(observation, entry.adapter.root_range)
    values: dict[str, list[float]] = {
        action_id: [] for action_id in legal_action_ids
    }
    required_infosets = {observation.digest()}
    for action_id in legal_action_ids:
        for sample_index in range(samples_per_action):
            values[action_id].append(
                _rollout_root_action(
                    adapter=adapter,
                    root_action_id=action_id,
                    root_actor=observation.actor,
                    split_group_sha256=split_group_sha256,
                    payoff_seed=payoff_seed,
                    sample_index=sample_index,
                    average_strategy=average_strategy,
                    required_infosets=required_infosets,
                )
            )
    moments: dict[str, dict[str, Any]] = {}
    for action_id in legal_action_ids:
        action_values = values[action_id]
        total = math.fsum(action_values)
        total_squares = math.fsum(value * value for value in action_values)
        moments[action_id] = {
            "count": len(action_values),
            "sum": total,
            "sum_squares": total_squares,
        }
    audit = copy.deepcopy(dict(adapter.sampling_audit()))
    expected_traversals = len(legal_action_ids) * samples_per_action
    if (
        audit.get("traversals") != expected_traversals
        or audit.get("root_posterior_samples") != expected_traversals
    ):
        raise SolveRootTeacherAdapterError(
            "teacher payoff root sampling accounting mismatch"
        )
    return {
        "moments": moments,
        "required_infoset_digests": sorted(required_infosets),
        "sampling_audit": audit,
    }


def _teacher_bindings(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    entry: MultiRootChanceEntry,
    *,
    source_binding_sha256: str,
    checkpoint: VerifiedMultiRootCheckpointSnapshot | None,
) -> dict[str, str]:
    root_range = entry.adapter.root_range
    bindings = {
        "public_root_commitment_sha256": context.digest(),
        "public_root_mixture_sha256": _compiled_mixture_binding(
            context, mixture
        )["binding_sha256"],
        "range_content_sha256": root_range.range_content_sha256,
        "range_build_sha256": root_range.range_build_sha256,
        "behavior_model_sha256": root_range.behavior_model_sha256,
        "source_manifest_sha256": source_binding_sha256,
    }
    if checkpoint is not None:
        bindings.update(
            {
                "solver_source_sha256": checkpoint.source_binding_sha256,
                "solver_config_sha256": checkpoint.solver_config_sha256,
                "solver_checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
        )
    return bindings


def _root_visit_and_reach(
    entry: MultiRootChanceEntry,
    checkpoint: VerifiedMultiRootCheckpointSnapshot,
) -> tuple[int, Fraction]:
    observation = entry.adapter.observation
    support = checkpoint.infoset_root_support_opaque_ids.get(observation)
    visits = checkpoint.infoset_root_visit_counts_by_opaque_id.get(observation)
    if support != (entry.root_id_sha256,) or visits is None or set(visits) != {
        entry.root_id_sha256
    }:
        raise SolveRootTeacherAdapterError(
            "solve-root checkpoint table has non-exact root support"
        )
    count = visits[entry.root_id_sha256]
    sampled = checkpoint.sampling_stats.get("root_samples_by_opaque_id")
    if not isinstance(sampled, Mapping) or sampled.get(entry.root_id_sha256) != count:
        raise SolveRootTeacherAdapterError(
            "solve-root visit count does not match checkpoint root samples"
        )
    if count <= 0:
        raise SolveRootTeacherAdapterError("selected solve root was never visited")
    return count, entry.prior_mass


def _payoff_evidence(
    *,
    plan: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    source_binding_sha256: str,
) -> dict[str, Any]:
    sampled = plan["sampled_plan"]
    moments = copy.deepcopy(evaluation["moments"])
    return {
        "schema": "ofc_t3_t4_solve_root_payoff_evidence/v1",
        "namespace": PAYOFF_SEED_NAMESPACE,
        "seed_index": sampled["payoff_seed_index"],
        "derived_seed": sampled["derived_payoff_seed"],
        "samples_per_action": sampled["samples_per_action"],
        "evaluator_source_binding_sha256": source_binding_sha256,
        "all_legal_root_actions_evaluated": True,
        "policy_key_type": "InfoSetKey",
        "hidden_cards_passed_to_policy": False,
        "missing_infoset_fallback": False,
        "required_continuation_infoset_digests": copy.deepcopy(
            evaluation["required_infoset_digests"]
        ),
        "action_raw_moments": moments,
        "action_raw_moments_sha256": canonical_sha256(moments),
        "sampling_audit": copy.deepcopy(evaluation["sampling_audit"]),
        "fresh_deterministic_replay_required": True,
    }


def _checkpoint_evidence(
    checkpoint: VerifiedMultiRootCheckpointSnapshot,
    entry: MultiRootChanceEntry,
    *,
    relative_path: str,
    root_visit_count: int,
) -> dict[str, Any]:
    root_policy = checkpoint.average_strategy[entry.adapter.observation]
    return {
        "schema": "ofc_t3_t4_solve_root_checkpoint_evidence/v1",
        "required": True,
        "relative_path": relative_path,
        "payload_sha256": checkpoint.checkpoint_sha256,
        "file_sha256": checkpoint.checkpoint_file_sha256,
        "source_binding_sha256": checkpoint.source_binding_sha256,
        "solver_config_sha256": checkpoint.solver_config_sha256,
        "prior_manifest_sha256": checkpoint.prior_manifest_sha256,
        "completed_iterations": checkpoint.completed_iterations,
        "root_table_sha256": canonical_sha256(
            {
                "observation_sha256": entry.adapter.observation.digest(),
                "support_opaque_ids": list(
                    checkpoint.infoset_root_support_opaque_ids[
                        entry.adapter.observation
                    ]
                ),
                "visits": dict(
                    checkpoint.infoset_root_visit_counts_by_opaque_id[
                        entry.adapter.observation
                    ]
                ),
            }
        ),
        "root_visit_count": root_visit_count,
        "root_support_opaque_ids": list(
            checkpoint.infoset_root_support_opaque_ids[
                entry.adapter.observation
            ]
        ),
        "average_strategy_sha256": canonical_sha256(
            [
                {
                    "action_id": action_id,
                    "probability_float_hex": float(root_policy[action_id]).hex(),
                }
                for action_id in sorted(root_policy)
            ]
        ),
        "solver_checkpoint_reexecuted": False,
        "checkpoint_content_restored": True,
    }


def _online_evidence(
    resolved: OnlinePublicResolveResult,
    mixture: CompiledPublicRootMixture,
) -> dict[str, Any]:
    manifest = resolved.manifest
    solver_binding = manifest["solver_result_binding"]
    return {
        "schema": "ofc_t3_t4_teacher_online_resolve_binding/v1",
        "manifest_sha256": resolved.manifest_sha256,
        "solver_result_binding_sha256": solver_binding["binding_sha256"],
        "compiled_support_sha256": mixture.support_sha256,
        "actual_infoset_consumed_by_solver": False,
        "actual_infoset_consumed_by_mixture_compilation": False,
    }


def _evidence_payload(
    *,
    branch: str,
    plan_relative_path: str,
    plan: PreLabelSolveRootTeacherPlan,
    plan_file_sha256: str,
    selected_root: Mapping[str, Any],
    online_evidence: Mapping[str, Any] | None,
    checkpoint_evidence: Mapping[str, Any] | None,
    payoff_evidence: Mapping[str, Any] | None,
    exact_evidence: Mapping[str, Any] | None,
    teacher_row: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SOLVE_ROOT_EVIDENCE_SCHEMA,
        "artifact_kind": SOLVE_ROOT_EVIDENCE_ARTIFACT,
        "branch": branch,
        "prelabel_plan_relative_path": plan_relative_path,
        "prelabel_plan_sha256": plan.manifest_sha256,
        "prelabel_plan_file_sha256": plan_file_sha256,
        "selected_root": copy.deepcopy(dict(selected_root)),
        "online_resolve_binding": (
            None if online_evidence is None else copy.deepcopy(dict(online_evidence))
        ),
        "checkpoint_evidence": (
            None
            if checkpoint_evidence is None
            else copy.deepcopy(dict(checkpoint_evidence))
        ),
        "payoff_evidence": (
            None if payoff_evidence is None else copy.deepcopy(dict(payoff_evidence))
        ),
        "exact_evidence": (
            None if exact_evidence is None else copy.deepcopy(dict(exact_evidence))
        ),
        "teacher_row": copy.deepcopy(dict(teacher_row)),
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "serving_changed": False,
        "opponent_hidden_cards_in_model_input": False,
        "bare_teacher_row_ingestion_allowed": False,
    }
    payload["evidence_sha256"] = self_hash(payload, "evidence_sha256")
    return payload


def _prepare_empty_output_dir(output_dir: str | Path) -> Path:
    supplied = Path(output_dir).absolute()
    if supplied.is_symlink() or any(parent.is_symlink() for parent in supplied.parents):
        raise SolveRootTeacherAdapterError("evidence output path traverses a symlink")
    root = supplied.resolve()
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise SolveRootTeacherAdapterError("evidence output directory must be empty")
    else:
        root.mkdir(parents=True)
    return root


def _copy_prelabel_plan(
    root: Path,
    source_path: str | Path,
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
) -> tuple[PreLabelSolveRootTeacherPlan, str, str]:
    plan = read_prelabel_solve_root_teacher_plan(source_path, context, mixture)
    source = Path(source_path)
    raw = source.read_bytes()
    relative = f"inputs/prelabel-plan-{plan.manifest_sha256}.json"
    target = root / relative
    _write_no_replace(target, raw)
    if _read_canonical_file(target) != dict(plan.manifest):
        raise SolveRootTeacherAdapterError("copied pre-label plan changed")
    return plan, relative, hashlib.sha256(raw).hexdigest()


def _snapshot_checkpoint(
    root: Path,
    source_path: str | Path,
    mixture: CompiledPublicRootMixture,
    result: MultiRootExternalSamplingMccfrResult,
    expected_checkpoint_sha256: str,
) -> tuple[VerifiedMultiRootCheckpointSnapshot, str]:
    source_snapshot = verify_multi_root_checkpoint_against_result(
        mixture.entries,
        result,
        checkpoint_path=source_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    source = Path(source_path)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_snapshot.checkpoint_file_sha256:
        raise SolveRootTeacherAdapterError("checkpoint changed before immutable copy")
    relative = f"checkpoints/{expected_checkpoint_sha256}.json"
    target = root / relative
    _write_no_replace(target, raw)
    immutable = verify_multi_root_checkpoint_against_result(
        mixture.entries,
        result,
        checkpoint_path=target,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if immutable != source_snapshot:
        raise SolveRootTeacherAdapterError("immutable checkpoint snapshot mismatch")
    return immutable, relative


def _validate_sampled_result_plan(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    plan: Mapping[str, Any],
    resolved: OnlinePublicResolveResult,
) -> None:
    verify_online_public_resolve(
        context,
        mixture,
        resolved,
        expected_manifest_sha256=resolved.manifest_sha256,
    )
    sampled = plan["sampled_plan"]
    result = resolved.solver_result
    checks = {
        "seed": (result.seed, sampled["derived_solver_seed"]),
        "iterations": (result.iterations, sampled["solver_iterations"]),
        "max_infosets": (
            result.metadata.get("max_infosets"),
            sampled["max_infosets"],
        ),
        "linear_averaging": (
            result.metadata.get("linear_averaging"),
            sampled["linear_averaging"],
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected or type(actual) is not type(expected):
            raise SolveRootTeacherAdapterError(
                f"solver result does not match pre-label {label}"
            )
    checkpoint_binding = resolved.manifest["checkpoint_binding"]
    if (
        checkpoint_binding.get("checkpoint_enabled") is not True
        or checkpoint_binding.get("checkpoint_output_persisted") is not True
        or checkpoint_binding.get("final_checkpoint_payload_sha256")
        != result.metadata.get("checkpoint_sha256")
    ):
        raise SolveRootTeacherAdapterError(
            "sampled teacher requires one persisted final checkpoint"
        )


def _build_sampled_components(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    plan: PreLabelSolveRootTeacherPlan,
    resolved: OnlinePublicResolveResult,
    checkpoint: VerifiedMultiRootCheckpointSnapshot,
    checkpoint_relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_manifest = plan.manifest
    _index, entry, _commitment = _select_root(
        context,
        mixture,
        plan_manifest["selected_root"]["root_id_sha256"],
    )
    sampled = plan_manifest["sampled_plan"]
    if checkpoint.seed != sampled["derived_solver_seed"]:
        raise SolveRootTeacherAdapterError("checkpoint seed is not split-derived")
    source_binding = _live_source_binding()
    visit_count, reach = _root_visit_and_reach(entry, checkpoint)
    evaluation = _evaluate_action_raw_moments(
        entry=entry,
        split_group_sha256=plan_manifest["split_assignment"][
            "split_group_sha256"
        ],
        payoff_seed=sampled["derived_payoff_seed"],
        samples_per_action=sampled["samples_per_action"],
        average_strategy=checkpoint.average_strategy,
    )
    payoff = _payoff_evidence(
        plan=plan_manifest,
        evaluation=evaluation,
        source_binding_sha256=source_binding["binding_sha256"],
    )
    row = build_teacher_row(
        entry.adapter.observation,
        split_assignment=plan_manifest["split_assignment"],
        lineage=plan_manifest["lineage"],
        bindings=_teacher_bindings(
            context,
            mixture,
            entry,
            source_binding_sha256=source_binding["binding_sha256"],
            checkpoint=checkpoint,
        ),
        solver_method=MCCFR_SOLVER_METHOD,
        solver_iterations_completed=checkpoint.completed_iterations,
        solver_seed_index=sampled["solver_seed_index"],
        payoff_seed_index=sampled["payoff_seed_index"],
        average_strategy_by_action_id=checkpoint.average_strategy[
            entry.adapter.observation
        ],
        action_payoff_moments_by_action_id=evaluation["moments"],
        infoset_visit_count=visit_count,
        infoset_reach_probability=reach,
    )
    checkpoint_payload = _checkpoint_evidence(
        checkpoint,
        entry,
        relative_path=checkpoint_relative_path,
        root_visit_count=visit_count,
    )
    return _online_evidence(resolved, mixture), checkpoint_payload, payoff, row


def _build_exact_components(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    plan: PreLabelSolveRootTeacherPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_manifest = plan.manifest
    _index, entry, _commitment = _select_root(
        context,
        mixture,
        plan_manifest["selected_root"]["root_id_sha256"],
    )
    information = entry.adapter.observation
    if _branch_for_key(information) != "t4_btn_exact":
        raise SolveRootTeacherAdapterError("exact evidence requires T4 BTN solve root")
    source_binding = _live_source_binding()
    resolution = resolve_t4_second_btn_exact(information)
    verification = verify_t4_second_btn_exact(
        information,
        resolution,
        expected_manifest_sha256=resolution.manifest_sha256,
    )
    row = build_t4_btn_exact_teacher_row(
        information,
        resolution=resolution,
        split_assignment=plan_manifest["split_assignment"],
        lineage=plan_manifest["lineage"],
        bindings=_teacher_bindings(
            context,
            mixture,
            entry,
            source_binding_sha256=source_binding["binding_sha256"],
            checkpoint=None,
        ),
    )
    exact = {
        "schema": "ofc_t4_btn_exact_teacher_evidence/v1",
        "resolver_manifest_sha256": resolution.manifest_sha256,
        "resolver_result_sha256": row["solver"]["resolver_result_sha256"],
        "selected_action_id": verification["selected_action_id"],
        "chance_sampling_used": False,
        "policy_sampling_used": False,
        "solver_seed_used": False,
        "payoff_seed_used": False,
        "checkpoint_used": False,
        "fresh_resolver_reexecution_required": True,
    }
    return exact, row


def build_solve_root_teacher_evidence(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    *,
    prelabel_plan_path: str | Path,
    output_dir: str | Path,
    resolved: OnlinePublicResolveResult | None = None,
    checkpoint_path: str | Path | None = None,
) -> VerifiedSolveRootTeacherEvidence:
    """Publish one immutable solve-root evidence artifact and verify it."""

    _require_runtime_contract()
    root = _prepare_empty_output_dir(output_dir)
    plan, plan_relative, plan_file_sha256 = _copy_prelabel_plan(
        root, prelabel_plan_path, context, mixture
    )
    manifest = plan.manifest
    branch = manifest["label_branch"]
    if branch == "sampled_mccfr":
        if type(resolved) is not OnlinePublicResolveResult or checkpoint_path is None:
            raise SolveRootTeacherAdapterError(
                "sampled evidence requires exact online result and checkpoint path"
            )
        _validate_sampled_result_plan(context, mixture, manifest, resolved)
        expected_checkpoint = _sha(
            resolved.solver_result.metadata.get("checkpoint_sha256"),
            label="solver checkpoint SHA256",
        )
        checkpoint, checkpoint_relative = _snapshot_checkpoint(
            root,
            checkpoint_path,
            mixture,
            resolved.solver_result,
            expected_checkpoint,
        )
        online, checkpoint_payload, payoff, row = _build_sampled_components(
            context,
            mixture,
            plan,
            resolved,
            checkpoint,
            checkpoint_relative,
        )
        exact = None
    else:
        if resolved is not None or checkpoint_path is not None:
            raise SolveRootTeacherAdapterError(
                "T4 BTN exact evidence forbids sampled result/checkpoint provenance"
            )
        exact, row = _build_exact_components(context, mixture, plan)
        online = None
        checkpoint_payload = None
        payoff = None
    evidence = _evidence_payload(
        branch=branch,
        plan_relative_path=plan_relative,
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        selected_root=manifest["selected_root"],
        online_evidence=online,
        checkpoint_evidence=checkpoint_payload,
        payoff_evidence=payoff,
        exact_evidence=exact,
        teacher_row=row,
    )
    _write_no_replace(
        root / EVIDENCE_MANIFEST_FILENAME,
        _canonical_file_bytes(evidence),
    )
    return verify_solve_root_teacher_evidence(
        context,
        mixture,
        root,
        resolved=resolved,
    )


_EVIDENCE_KEYS = {
    "schema",
    "artifact_kind",
    "branch",
    "prelabel_plan_relative_path",
    "prelabel_plan_sha256",
    "prelabel_plan_file_sha256",
    "selected_root",
    "online_resolve_binding",
    "checkpoint_evidence",
    "payoff_evidence",
    "exact_evidence",
    "teacher_row",
    "algorithm_teacher_only",
    "promotion_eligible",
    "runtime_integrated",
    "serving_changed",
    "opponent_hidden_cards_in_model_input",
    "bare_teacher_row_ingestion_allowed",
    "evidence_sha256",
}


def _safe_relative_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SolveRootTeacherAdapterError(f"{label}: canonical relative path required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise SolveRootTeacherAdapterError(f"{label}: unsafe relative path")
    target = root / path
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise SolveRootTeacherAdapterError(f"{label}: symlink forbidden")
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise SolveRootTeacherAdapterError(f"{label}: file missing or escapes root") from exc
    if not resolved.is_file():
        raise SolveRootTeacherAdapterError(f"{label}: regular file required")
    return resolved


def _artifact_file_set(root: Path) -> set[str]:
    output: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SolveRootTeacherAdapterError("evidence artifact contains symlink")
        if path.is_file():
            output.add(path.relative_to(root).as_posix())
    return output


def _artifact_directory_set(root: Path) -> set[str]:
    output: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SolveRootTeacherAdapterError("evidence artifact contains symlink")
        if path.is_dir():
            output.add(path.relative_to(root).as_posix())
        elif not path.is_file():
            raise SolveRootTeacherAdapterError(
                "evidence artifact contains a non-regular path"
            )
    return output


def verify_solve_root_teacher_evidence(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    evidence_dir: str | Path,
    *,
    resolved: OnlinePublicResolveResult | None = None,
    expected_evidence_sha256: str | None = None,
) -> VerifiedSolveRootTeacherEvidence:
    """Freshly restore/replay all evidence; a bare teacher row is insufficient."""

    _require_runtime_contract()
    root = Path(evidence_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        raise SolveRootTeacherAdapterError("evidence root must be a real directory")
    raw = _read_canonical_file(root / EVIDENCE_MANIFEST_FILENAME)
    _exact_keys(raw, _EVIDENCE_KEYS, label="solve_root_evidence")
    fixed = {
        "schema": SOLVE_ROOT_EVIDENCE_SCHEMA,
        "artifact_kind": SOLVE_ROOT_EVIDENCE_ARTIFACT,
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "serving_changed": False,
        "opponent_hidden_cards_in_model_input": False,
        "bare_teacher_row_ingestion_allowed": False,
    }
    for field, expected in fixed.items():
        if raw.get(field) != expected:
            raise SolveRootTeacherAdapterError(f"solve_root_evidence.{field}: mismatch")
    evidence_sha = _sha(raw.get("evidence_sha256"), label="evidence_sha256")
    if evidence_sha != self_hash(raw, "evidence_sha256"):
        raise SolveRootTeacherAdapterError("solve-root evidence self-hash mismatch")
    if expected_evidence_sha256 is not None and evidence_sha != _sha(
        expected_evidence_sha256, label="expected_evidence_sha256"
    ):
        raise SolveRootTeacherAdapterError("solve-root evidence is not externally pinned")
    plan_path = _safe_relative_file(
        root, raw.get("prelabel_plan_relative_path"), label="pre-label plan"
    )
    if _file_sha256(plan_path) != _sha(
        raw.get("prelabel_plan_file_sha256"), label="prelabel_plan_file_sha256"
    ):
        raise SolveRootTeacherAdapterError("pre-label plan file hash mismatch")
    plan = read_prelabel_solve_root_teacher_plan(plan_path, context, mixture)
    if plan.manifest_sha256 != raw.get("prelabel_plan_sha256"):
        raise SolveRootTeacherAdapterError("pre-label plan semantic hash mismatch")
    if dict(plan.manifest["selected_root"]) != raw.get("selected_root"):
        raise SolveRootTeacherAdapterError("evidence selected root changed from plan")
    branch = plan.manifest["label_branch"]
    if raw.get("branch") != branch:
        raise SolveRootTeacherAdapterError("evidence branch changed from plan")
    expected_files = {
        EVIDENCE_MANIFEST_FILENAME,
        raw["prelabel_plan_relative_path"],
    }
    if branch == "sampled_mccfr":
        if type(resolved) is not OnlinePublicResolveResult:
            raise SolveRootTeacherAdapterError(
                "sampled evidence verification requires exact online result"
            )
        _validate_sampled_result_plan(context, mixture, plan.manifest, resolved)
        checkpoint_raw = _require_mapping(
            raw.get("checkpoint_evidence"), label="checkpoint_evidence"
        )
        checkpoint_relative = checkpoint_raw.get("relative_path")
        checkpoint_path = _safe_relative_file(
            root, checkpoint_relative, label="immutable checkpoint"
        )
        expected_files.add(str(checkpoint_relative))
        expected_checkpoint_sha = _sha(
            checkpoint_raw.get("payload_sha256"), label="checkpoint payload SHA256"
        )
        checkpoint = verify_multi_root_checkpoint_against_result(
            mixture.entries,
            resolved.solver_result,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha,
        )
        online, rebuilt_checkpoint, payoff, row = _build_sampled_components(
            context,
            mixture,
            plan,
            resolved,
            checkpoint,
            str(checkpoint_relative),
        )
        exact = None
    else:
        if resolved is not None:
            raise SolveRootTeacherAdapterError(
                "exact evidence verification forbids sampled online result"
            )
        exact, row = _build_exact_components(context, mixture, plan)
        online = None
        rebuilt_checkpoint = None
        payoff = None
    verified_row = verify_teacher_row(raw.get("teacher_row"))
    if verified_row != row:
        raise SolveRootTeacherAdapterError("teacher row does not rebuild from evidence")
    rebuilt = _evidence_payload(
        branch=branch,
        plan_relative_path=raw["prelabel_plan_relative_path"],
        plan=plan,
        plan_file_sha256=raw["prelabel_plan_file_sha256"],
        selected_root=plan.manifest["selected_root"],
        online_evidence=online,
        checkpoint_evidence=rebuilt_checkpoint,
        payoff_evidence=payoff,
        exact_evidence=exact,
        teacher_row=row,
    )
    if raw != rebuilt:
        raise SolveRootTeacherAdapterError(
            "solve-root evidence does not match fresh reconstruction"
        )
    if _artifact_file_set(root) != expected_files:
        raise SolveRootTeacherAdapterError("solve-root evidence has missing/orphan files")
    expected_directories = {"inputs"}
    if branch == "sampled_mccfr":
        expected_directories.add("checkpoints")
    if _artifact_directory_set(root) != expected_directories:
        raise SolveRootTeacherAdapterError(
            "solve-root evidence has missing/orphan directories"
        )
    return VerifiedSolveRootTeacherEvidence(
        root=root.resolve(),
        manifest=MappingProxyType(copy.deepcopy(raw)),
        teacher_row=MappingProxyType(copy.deepcopy(row)),
        _verification_seal=_VERIFIED_EVIDENCE_SEAL,
    )


_EVIDENCE_ENTRY_KEYS = {
    "schema",
    "index",
    "branch",
    "evidence_sha256",
    "evidence_manifest_file_sha256",
    "root_id_sha256",
    "observation_sha256",
    "split_assignment_sha256",
    "row_identity_sha256",
    "row_sha256",
    "entry_sha256",
}


def _assert_verified_evidence_current(
    evidence: VerifiedSolveRootTeacherEvidence,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck the immutable files represented by an already verified handle."""

    if type(evidence) is not VerifiedSolveRootTeacherEvidence:
        raise TypeError(
            "teacher ingestion requires exact VerifiedSolveRootTeacherEvidence objects; "
            "bare teacher rows are forbidden"
        )
    if evidence._verification_seal is not _VERIFIED_EVIDENCE_SEAL:
        raise SolveRootTeacherAdapterError(
            "teacher ingestion requires a verifier-issued evidence handle"
        )
    root = Path(evidence.root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise SolveRootTeacherAdapterError("verified evidence root is no longer a directory")
    manifest_path = root / EVIDENCE_MANIFEST_FILENAME
    manifest = _read_canonical_file(manifest_path)
    if manifest != dict(evidence.manifest):
        raise SolveRootTeacherAdapterError("verified evidence manifest changed after verification")
    if manifest.get("evidence_sha256") != self_hash(manifest, "evidence_sha256"):
        raise SolveRootTeacherAdapterError("verified evidence self-hash changed")
    row = verify_teacher_row(dict(evidence.teacher_row))
    if row != manifest.get("teacher_row"):
        raise SolveRootTeacherAdapterError("verified evidence teacher row changed")
    plan_path = _safe_relative_file(
        root,
        manifest.get("prelabel_plan_relative_path"),
        label="bound pre-label plan",
    )
    if _file_sha256(plan_path) != manifest.get("prelabel_plan_file_sha256"):
        raise SolveRootTeacherAdapterError("bound pre-label plan changed")
    expected_files = {
        EVIDENCE_MANIFEST_FILENAME,
        manifest["prelabel_plan_relative_path"],
    }
    expected_directories = {"inputs"}
    if manifest.get("branch") == "sampled_mccfr":
        checkpoint = _require_mapping(
            manifest.get("checkpoint_evidence"), label="bound checkpoint evidence"
        )
        checkpoint_path = _safe_relative_file(
            root, checkpoint.get("relative_path"), label="bound checkpoint"
        )
        if _file_sha256(checkpoint_path) != checkpoint.get("file_sha256"):
            raise SolveRootTeacherAdapterError("bound checkpoint file changed")
        expected_files.add(checkpoint["relative_path"])
        expected_directories.add("checkpoints")
    elif manifest.get("branch") != "t4_btn_exact":
        raise SolveRootTeacherAdapterError("verified evidence branch is unknown")
    if _artifact_file_set(root) != expected_files:
        raise SolveRootTeacherAdapterError("bound evidence has missing/orphan files")
    if _artifact_directory_set(root) != expected_directories:
        raise SolveRootTeacherAdapterError("bound evidence has missing/orphan directories")
    return manifest, row


def _evidence_entries(
    evidences: Sequence[VerifiedSolveRootTeacherEvidence],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(evidences, (str, bytes, bytearray)) or not isinstance(
        evidences, Sequence
    ):
        raise TypeError("evidences must be a sequence of verified evidence objects")
    if not evidences:
        raise SolveRootTeacherAdapterError("at least one verified evidence is required")
    staged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    evidence_hashes: set[str] = set()
    row_identities: set[str] = set()
    for evidence in evidences:
        manifest, row = _assert_verified_evidence_current(evidence)
        evidence_hash = _sha(
            manifest.get("evidence_sha256"), label="bound evidence SHA256"
        )
        row_identity = _sha(
            row.get("row_identity_sha256"), label="bound row identity SHA256"
        )
        if evidence_hash in evidence_hashes:
            raise SolveRootTeacherAdapterError("duplicate solve-root evidence")
        if row_identity in row_identities:
            raise SolveRootTeacherAdapterError("duplicate teacher row identity")
        evidence_hashes.add(evidence_hash)
        row_identities.add(row_identity)
        selected = _require_mapping(
            manifest.get("selected_root"), label="bound selected root"
        )
        assignment = _require_mapping(
            row.get("split_assignment"), label="bound split assignment"
        )
        entry: dict[str, Any] = {
            "schema": "ofc_t3_t4_evidence_bound_teacher_entry/v1",
            "index": -1,
            "branch": manifest.get("branch"),
            "evidence_sha256": evidence_hash,
            "evidence_manifest_file_sha256": _file_sha256(
                Path(evidence.root) / EVIDENCE_MANIFEST_FILENAME
            ),
            "root_id_sha256": _sha(
                selected.get("root_id_sha256"), label="bound root ID"
            ),
            "observation_sha256": _sha(
                selected.get("observation_sha256"), label="bound observation"
            ),
            "split_assignment_sha256": _sha(
                assignment.get("assignment_sha256"),
                label="bound split assignment SHA256",
            ),
            "row_identity_sha256": row_identity,
            "row_sha256": _sha(row.get("row_sha256"), label="bound row SHA256"),
        }
        staged.append((entry, row))
    staged.sort(key=lambda item: item[0]["row_identity_sha256"])
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (entry, row) in enumerate(staged):
        entry["index"] = index
        entry["entry_sha256"] = self_hash(entry, "entry_sha256")
        entries.append(entry)
        rows.append(row)
    return entries, rows


def _evidence_bound_payload(
    *,
    entries: Sequence[Mapping[str, Any]],
    teacher_manifest: Mapping[str, Any],
    teacher_manifest_file_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": EVIDENCE_BOUND_TEACHER_SCHEMA,
        "artifact_kind": "evidence_bound_t3_t4_distillation_teacher_bundle",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "training_performed": False,
        "runtime_integrated": False,
        "serving_changed": False,
        "bare_teacher_row_ingestion_allowed": False,
        "all_rows_bound_to_verified_solve_root_evidence": True,
        "evidence_count": len(entries),
        "row_count": len(entries),
        "evidence_sha256_order": [entry["evidence_sha256"] for entry in entries],
        "row_identity_sha256_order": [
            entry["row_identity_sha256"] for entry in entries
        ],
        "evidence_entries": [copy.deepcopy(dict(entry)) for entry in entries],
        "teacher_bundle_relative_path": "teacher",
        "teacher_manifest_sha256": teacher_manifest["manifest_sha256"],
        "teacher_manifest_file_sha256": teacher_manifest_file_sha256,
        "manifest_written_last": True,
    }
    payload["manifest_sha256"] = self_hash(payload, "manifest_sha256")
    return payload


_EVIDENCE_BOUND_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "training_performed",
    "runtime_integrated",
    "serving_changed",
    "bare_teacher_row_ingestion_allowed",
    "all_rows_bound_to_verified_solve_root_evidence",
    "evidence_count",
    "row_count",
    "evidence_sha256_order",
    "row_identity_sha256_order",
    "evidence_entries",
    "teacher_bundle_relative_path",
    "teacher_manifest_sha256",
    "teacher_manifest_file_sha256",
    "manifest_written_last",
    "manifest_sha256",
}


def write_evidence_bound_teacher_bundle(
    output_dir: str | Path,
    evidences: Sequence[VerifiedSolveRootTeacherEvidence],
    *,
    shard_size: int,
) -> VerifiedEvidenceBoundTeacherBundle:
    """Publish teacher shards only from verified solve-root evidence handles."""

    _require_runtime_contract()
    entries, rows = _evidence_entries(evidences)
    root = _prepare_empty_output_dir(output_dir)
    teacher = write_teacher_bundle(root / "teacher", rows, shard_size=shard_size)
    teacher_manifest_path = teacher.root / TEACHER_MANIFEST_FILENAME
    manifest = _evidence_bound_payload(
        entries=entries,
        teacher_manifest=teacher.manifest,
        teacher_manifest_file_sha256=_file_sha256(teacher_manifest_path),
    )
    _write_no_replace(
        root / EVIDENCE_BOUND_MANIFEST_FILENAME,
        _canonical_file_bytes(manifest),
    )
    return verify_evidence_bound_teacher_bundle(root, evidences)


def verify_evidence_bound_teacher_bundle(
    output_dir: str | Path,
    evidences: Sequence[VerifiedSolveRootTeacherEvidence],
    *,
    expected_manifest_sha256: str | None = None,
) -> VerifiedEvidenceBoundTeacherBundle:
    """Verify row shards and their one-to-one solve-root evidence bindings."""

    _require_runtime_contract()
    entries, rows = _evidence_entries(evidences)
    root = Path(output_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        raise SolveRootTeacherAdapterError(
            "evidence-bound teacher root must be a real directory"
        )
    manifest = _read_canonical_file(root / EVIDENCE_BOUND_MANIFEST_FILENAME)
    _exact_keys(manifest, _EVIDENCE_BOUND_KEYS, label="evidence_bound_teacher")
    fixed = {
        "schema": EVIDENCE_BOUND_TEACHER_SCHEMA,
        "artifact_kind": "evidence_bound_t3_t4_distillation_teacher_bundle",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "training_performed": False,
        "runtime_integrated": False,
        "serving_changed": False,
        "bare_teacher_row_ingestion_allowed": False,
        "all_rows_bound_to_verified_solve_root_evidence": True,
        "evidence_count": len(entries),
        "row_count": len(entries),
        "teacher_bundle_relative_path": "teacher",
        "manifest_written_last": True,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise SolveRootTeacherAdapterError(
                f"evidence_bound_teacher.{field}: mismatch"
            )
    manifest_sha = _sha(
        manifest.get("manifest_sha256"), label="evidence-bound manifest SHA256"
    )
    if manifest_sha != self_hash(manifest, "manifest_sha256"):
        raise SolveRootTeacherAdapterError("evidence-bound manifest self-hash mismatch")
    if expected_manifest_sha256 is not None and manifest_sha != _sha(
        expected_manifest_sha256, label="expected evidence-bound manifest SHA256"
    ):
        raise SolveRootTeacherAdapterError(
            "evidence-bound teacher manifest is not externally pinned"
        )
    teacher = verify_teacher_bundle(root / "teacher")
    if list(teacher.rows) != rows:
        raise SolveRootTeacherAdapterError(
            "teacher shards do not exactly match evidence-bound rows"
        )
    rebuilt = _evidence_bound_payload(
        entries=entries,
        teacher_manifest=teacher.manifest,
        teacher_manifest_file_sha256=_file_sha256(
            teacher.root / TEACHER_MANIFEST_FILENAME
        ),
    )
    if manifest != rebuilt:
        raise SolveRootTeacherAdapterError(
            "evidence-bound teacher manifest does not rebuild"
        )
    expected_files = {EVIDENCE_BOUND_MANIFEST_FILENAME}
    expected_files.update(
        f"teacher/{path.relative_to(teacher.root).as_posix()}"
        for path in teacher.root.rglob("*")
        if path.is_file()
    )
    if _artifact_file_set(root) != expected_files:
        raise SolveRootTeacherAdapterError(
            "evidence-bound teacher has missing/orphan files"
        )
    if _artifact_directory_set(root) != {"teacher", "teacher/shards"}:
        raise SolveRootTeacherAdapterError(
            "evidence-bound teacher has missing/orphan directories"
        )
    return VerifiedEvidenceBoundTeacherBundle(
        root=root.resolve(),
        manifest=MappingProxyType(copy.deepcopy(manifest)),
        teacher_manifest=MappingProxyType(copy.deepcopy(teacher.manifest)),
    )


def _runtime_guard_factory(
    aliases: tuple[tuple[str, Any], ...],
    values: tuple[tuple[str, Any], ...],
    verifier: Any,
    registered_anchor: Any,
) -> Any:
    """Close over import-time dependencies used by every public trust boundary."""

    def require_runtime_contract() -> None:
        for name, canonical in aliases:
            if globals().get(name) is not canonical:
                raise SolveRootTeacherAdapterError(
                    f"solve-root teacher runtime callable alias drift: {name}"
                )
        for name, expected in values:
            if globals().get(name) != expected:
                raise SolveRootTeacherAdapterError(
                    f"solve-root teacher runtime contract value drift: {name}"
                )
        anchor = globals().get("_MODULE_RUNTIME_ANCHOR")
        if anchor is not globals().get("_MODULE_RUNTIME_ANCHOR_MIRROR"):
            raise SolveRootTeacherAdapterError(
                "solve-root teacher runtime anchor identity drift"
            )
        try:
            if anchor is not registered_anchor(__name__, globals()):
                raise RuntimeError(
                    "solve-root teacher runtime anchor registry drift"
                )
            verifier(globals(), anchor)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise SolveRootTeacherAdapterError(
                f"solve-root teacher runtime semantic drift: {exc}"
            ) from exc

    return require_runtime_contract


_CANONICAL_RUNTIME_ALIASES = (
    ("register_module_function_anchor", register_module_function_anchor),
    ("registered_module_function_anchor", registered_module_function_anchor),
    ("verify_module_function_anchor", verify_module_function_anchor),
    ("verify_multi_root_checkpoint_against_result", verify_multi_root_checkpoint_against_result),
    ("build_split_assignment", build_split_assignment),
    ("verify_split_assignment", verify_split_assignment),
    ("derive_solver_seed", derive_solver_seed),
    ("derive_payoff_seed", derive_payoff_seed),
    ("build_teacher_row", build_teacher_row),
    ("build_t4_btn_exact_teacher_row", build_t4_btn_exact_teacher_row),
    ("verify_teacher_row", verify_teacher_row),
    ("write_teacher_bundle", write_teacher_bundle),
    ("verify_teacher_bundle", verify_teacher_bundle),
    ("resolve_compiled_public_root_mixture", resolve_compiled_public_root_mixture),
    ("verify_online_public_resolve", verify_online_public_resolve),
    ("verify_compiled_public_root_mixture", verify_compiled_public_root_mixture),
    ("resolve_t4_second_btn_exact", resolve_t4_second_btn_exact),
    ("verify_t4_second_btn_exact", verify_t4_second_btn_exact),
    ("semantic_action_ids", semantic_action_ids),
    ("canonical_json", canonical_json),
    ("canonical_sha256", canonical_sha256),
    ("self_hash", self_hash),
)
_CANONICAL_RUNTIME_VALUES = (
    ("PRELABEL_PLAN_SCHEMA", PRELABEL_PLAN_SCHEMA),
    ("SOLVE_ROOT_EVIDENCE_SCHEMA", SOLVE_ROOT_EVIDENCE_SCHEMA),
    ("EVIDENCE_BOUND_TEACHER_SCHEMA", EVIDENCE_BOUND_TEACHER_SCHEMA),
    ("MCCFR_SOLVER_METHOD", MCCFR_SOLVER_METHOD),
    ("SOLVER_SEED_NAMESPACE", SOLVER_SEED_NAMESPACE),
    ("PAYOFF_SEED_NAMESPACE", PAYOFF_SEED_NAMESPACE),
    ("_SAMPLED_PHASE_ACTORS", _SAMPLED_PHASE_ACTORS),
    ("_EXACT_PHASE_ACTOR", _EXACT_PHASE_ACTOR),
    ("_SOURCE_PATHS", _SOURCE_PATHS),
    ("_VERIFIED_EVIDENCE_SEAL", _VERIFIED_EVIDENCE_SEAL),
)
_require_runtime_contract = _runtime_guard_factory(
    _CANONICAL_RUNTIME_ALIASES,
    _CANONICAL_RUNTIME_VALUES,
    verify_module_function_anchor,
    registered_module_function_anchor,
)
_MODULE_RUNTIME_ANCHOR = register_module_function_anchor(__name__, globals())
_MODULE_RUNTIME_ANCHOR_MIRROR = _MODULE_RUNTIME_ANCHOR


__all__ = [
    "EVIDENCE_MANIFEST_FILENAME",
    "EVIDENCE_BOUND_MANIFEST_FILENAME",
    "EVIDENCE_BOUND_TEACHER_SCHEMA",
    "PRELABEL_PLAN_FILENAME",
    "PRELABEL_PLAN_SCHEMA",
    "SOLVE_ROOT_EVIDENCE_SCHEMA",
    "PreLabelSolveRootTeacherPlan",
    "SolveRootTeacherAdapterError",
    "VerifiedEvidenceBoundTeacherBundle",
    "VerifiedSolveRootTeacherEvidence",
    "build_prelabel_solve_root_teacher_plan",
    "build_solve_root_teacher_evidence",
    "read_prelabel_solve_root_teacher_plan",
    "resolve_prelabel_solve_root_plan",
    "verify_evidence_bound_teacher_bundle",
    "verify_prelabel_solve_root_teacher_plan",
    "verify_solve_root_teacher_evidence",
    "write_evidence_bound_teacher_bundle",
    "write_prelabel_solve_root_teacher_plan",
]
