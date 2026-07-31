"""Bounded-memory sharded calibration bootstrap for the T3 fixed point.

This is the production-scale sibling of :mod:`calibrated_behavior_bootstrap`.
It starts from four immutable shard trees plus one saved sharded temperature
calibration artifact.  Construction always loads the four exact known
policy-value checkpoints, freshly verifies every collection/evaluation shard,
rebuilds the complete calibration result in bounded memory, and then creates
the existing strict T1/T2 bootstrap dispatch type.

The resulting runtime is deliberately not promotion evidence.  Even when its
source calibration passed every production gate it remains
``promotion_eligible=false``, ``fixed_point_bootstrap_only=true``, and
``strategic_strength=false``.  Its only purpose is to provide the no-fallback
T1/T2 likelihood routes needed while solving the endogenous T3-BB fixed point.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import ai.tutor.calibrated_behavior_bootstrap as bootstrap_runtime
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.behavior_calibration_contract import canonical_snapshot
from ai.tutor.behavior_logit_evaluator_torch import (
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.behavior_temperature_calibration_shards import (
    SHARDED_CALIBRATION_SCHEMA,
    read_sharded_behavior_temperature_calibration,
)
from ai.tutor.calibrated_behavior_bootstrap import (
    BOOTSTRAP_MODEL_TYPE,
    BOOTSTRAP_ROUTE_SCOPE,
    FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
)
from ai.tutor.calibrated_behavior_runtime import (
    APPROVED_HU_POLICY_VALUE_CHECKPOINTS,
    EXPECTED_ROUTES,
    RUNTIME_SCHEMA,
    _require_mapping,
    _require_sha256,
    _validated_identity_evaluator,
    _validated_runtime_child,
)
from ai.tutor.frozen_behavior_torch import (
    DEFAULT_QUANTIZATION_DENOMINATOR,
    KNOWN_HU_POLICY_VALUE_ASSETS,
    RULES_VERSION,
    TorchPolicyValueBehaviorModel,
)
from ai.tutor.promotion_gate_m3_full_card_strength import canonical_sha256
from ai.tutor.t3_hu_full_card_range import FrozenBehaviorModel


SHARDED_BOOTSTRAP_EVIDENCE_TYPE = (
    "fresh_verified_bounded_memory_sharded_temperature_calibration_v1"
)
_EXPECTED_ROUTE_SET = frozenset(EXPECTED_ROUTES)
_DATASET_KEYS = ("natural", "challenge")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_calibration_status(
    artifact: Mapping[str, Any],
) -> tuple[bool, bool, tuple[str, ...]]:
    """Return an internally consistent sharded source-gate status."""

    if artifact.get("schema") != SHARDED_CALIBRATION_SCHEMA:
        raise ValueError("verified artifact has an unsupported sharded schema")
    artifact_eligible = artifact.get("promotion_eligible")
    if type(artifact_eligible) is not bool:
        raise TypeError("sharded calibration promotion_eligible must be boolean")
    gate = _require_mapping(artifact.get("gate_result"), label="gate_result")
    gate_eligible = gate.get("promotion_eligible")
    all_passed = gate.get("all_required_gates_passed")
    if type(gate_eligible) is not bool or type(all_passed) is not bool:
        raise TypeError("sharded calibration gate status must be boolean")
    failures = gate.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) or not failure for failure in failures
    ):
        raise TypeError("sharded calibration gate failures must be strings")
    if artifact_eligible != gate_eligible or gate_eligible != all_passed:
        raise ValueError("sharded calibration promotion status is inconsistent")
    if all_passed != (failures == []):
        raise ValueError("sharded calibration failures/status is inconsistent")
    return artifact_eligible, all_passed, tuple(failures)


def _input_hash_bindings(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the four freshly verified shard-tree identities.

    Absolute paths are intentionally not runtime identity.  A byte-identical
    immutable shard tree may be relocated, while swapping a path to different
    evidence fails the fresh rebuild before this function is reached.
    """

    raw_inputs = _require_mapping(
        artifact.get("input_bindings"), label="input_bindings"
    )
    if set(raw_inputs) != set(_DATASET_KEYS):
        raise ValueError("sharded calibration requires natural/challenge inputs")
    result: dict[str, Any] = {}
    for dataset in _DATASET_KEYS:
        dataset_binding = _require_mapping(
            raw_inputs[dataset], label=f"input_bindings.{dataset}"
        )
        collection = _require_mapping(
            dataset_binding.get("collection"),
            label=f"input_bindings.{dataset}.collection",
        )
        evaluation = _require_mapping(
            dataset_binding.get("evaluation"),
            label=f"input_bindings.{dataset}.evaluation",
        )
        collection_hashes = {
            "content_sha256": _require_sha256(
                collection.get("collection_content_sha256"),
                label=f"{dataset} collection content",
            ),
            "layout_sha256": _require_sha256(
                collection.get("layout_sha256"),
                label=f"{dataset} collection layout",
            ),
            "manifest_sha256": _require_sha256(
                collection.get("manifest_sha256"),
                label=f"{dataset} collection manifest",
            ),
            "ordered_entry_chain_sha256": _require_sha256(
                collection.get("ordered_shard_entry_chain_sha256"),
                label=f"{dataset} collection ordered entry chain",
            ),
        }
        evaluation_hashes = {
            "content_sha256": _require_sha256(
                evaluation.get("evaluation_content_sha256"),
                label=f"{dataset} evaluation content",
            ),
            "layout_sha256": _require_sha256(
                evaluation.get("layout_sha256"),
                label=f"{dataset} evaluation layout",
            ),
            "manifest_sha256": _require_sha256(
                evaluation.get("manifest_sha256"),
                label=f"{dataset} evaluation manifest",
            ),
            "ordered_entry_chain_sha256": _require_sha256(
                evaluation.get("ordered_evaluation_shard_entry_chain_sha256"),
                label=f"{dataset} evaluation ordered entry chain",
            ),
            "evaluator_set_sha256": _require_sha256(
                evaluation.get("evaluator_set_sha256"),
                label=f"{dataset} evaluation evaluator set",
            ),
        }
        result[dataset] = {
            "collection": collection_hashes,
            "evaluation": evaluation_hashes,
            "raw_record_set_sha256": _require_sha256(
                dataset_binding.get("raw_record_set_sha256"),
                label=f"{dataset} raw record set",
            ),
            "evaluation_row_set_sha256": _require_sha256(
                dataset_binding.get("evaluation_row_set_sha256"),
                label=f"{dataset} evaluation row set",
            ),
        }
    return result


def _validate_paths(
    natural_collection_dir: str | Path,
    natural_evaluation_dir: str | Path,
    challenge_collection_dir: str | Path,
    challenge_evaluation_dir: str | Path,
    calibration_artifact_path: str | Path,
    workspace_root: str | Path,
) -> tuple[tuple[Path, Path, Path, Path], Path, Path]:
    roots = tuple(
        Path(value).resolve()
        for value in (
            natural_collection_dir,
            natural_evaluation_dir,
            challenge_collection_dir,
            challenge_evaluation_dir,
        )
    )
    if len(set(roots)) != len(roots):
        raise ValueError(
            "natural/challenge collection/evaluation roots must be distinct"
        )
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError(
                    "sharded evidence directories must not contain one another"
                )
    for path in roots:
        if not path.is_dir():
            raise FileNotFoundError(f"sharded evidence directory not found: {path}")
    artifact_path = Path(calibration_artifact_path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"sharded calibration artifact not found: {artifact_path}"
        )
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"workspace root not found: {root}")
    return roots, artifact_path, root


def _validated_scratch_dir(
    scratch_dir: str | Path | None, roots: tuple[Path, Path, Path, Path]
) -> Path | None:
    if scratch_dir is None:
        return None
    scratch = Path(scratch_dir).resolve()
    if scratch.exists() and not scratch.is_dir():
        raise ValueError("scratch_dir must be a directory")
    if any(scratch == root or root in scratch.parents for root in roots):
        raise ValueError("scratch_dir must be outside immutable evidence directories")
    return scratch


def build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch_from_shards(
    natural_collection_dir: str | Path,
    natural_evaluation_dir: str | Path,
    challenge_collection_dir: str | Path,
    challenge_evaluation_dir: str | Path,
    calibration_artifact_path: str | Path,
    *,
    workspace_root: str | Path,
    scratch_dir: str | Path | None = None,
    quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
    require_promoted_source: bool = False,
    model_id: str = "fixed_point_sharded_bootstrap_calibrated_hu_t1_t2_v1",
) -> FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch:
    """Freshly verify sharded calibration evidence and build four T1/T2 routes."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if type(require_promoted_source) is not bool:
        raise TypeError("require_promoted_source must be boolean")
    if (
        isinstance(quantization_denominator, bool)
        or not isinstance(quantization_denominator, int)
        or quantization_denominator != DEFAULT_QUANTIZATION_DENOMINATOR
    ):
        raise ValueError("sharded bootstrap runtime requires the exact Q32 denominator")

    roots, artifact_path, root = _validate_paths(
        natural_collection_dir,
        natural_evaluation_dir,
        challenge_collection_dir,
        challenge_evaluation_dir,
        calibration_artifact_path,
        workspace_root,
    )
    scratch = _validated_scratch_dir(scratch_dir, roots)
    if set(KNOWN_HU_POLICY_VALUE_ASSETS) != _EXPECTED_ROUTE_SET:
        raise ValueError("known HU policy-value asset registry has drifted")
    if set(APPROVED_HU_POLICY_VALUE_CHECKPOINTS) != _EXPECTED_ROUTE_SET:
        raise AssertionError("approved HU checkpoint registry is incomplete")

    # These evaluators read and hash the four actual checkpoint files.  The
    # same exact identities then verify every sharded evaluation row and are
    # independently checked again before constructing each runtime child.
    evaluators = build_known_hu_policy_value_logit_evaluators(
        root, quantization_denominator=quantization_denominator
    )
    if set(evaluators) != _EXPECTED_ROUTE_SET:
        raise ValueError("known logit evaluator loader returned incomplete routes")
    artifact_file_sha256 = _file_sha256(artifact_path)
    verified_artifact = read_sharded_behavior_temperature_calibration(
        artifact_path,
        roots[0],
        roots[1],
        roots[2],
        roots[3],
        evaluators,
        scratch_dir=scratch,
    )
    if _file_sha256(artifact_path) != artifact_file_sha256:
        raise ValueError("sharded calibration artifact changed during verification")
    (
        source_calibration_promotion_eligible,
        calibration_gate_passed,
        gate_failures,
    ) = _source_calibration_status(verified_artifact)
    if require_promoted_source and not (
        source_calibration_promotion_eligible and calibration_gate_passed
    ):
        raise ValueError(
            "production sharded bootstrap requires a promotion-eligible source "
            "calibration with all required gates passed"
        )

    artifact_sha256 = _require_sha256(
        verified_artifact.get("artifact_sha256"), label="artifact_sha256"
    )
    role_temperatures = bootstrap_runtime._artifact_role_temperatures(
        verified_artifact
    )
    role_bindings = bootstrap_runtime._artifact_role_bindings(verified_artifact)
    input_hash_bindings = _input_hash_bindings(verified_artifact)

    routes: dict[tuple[int, str], FrozenBehaviorModel] = {}
    route_manifest: list[dict[str, Any]] = []
    for turn, actor in EXPECTED_ROUTES:
        role = f"t{turn}_{actor}"
        asset = _require_mapping(
            KNOWN_HU_POLICY_VALUE_ASSETS[(turn, actor)],
            label=f"known asset {role}",
        )
        if set(asset) != {"relative_path", "checkpoint_sha256"}:
            raise ValueError(f"known asset {role} has unexpected fields")
        relative_path = asset.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"known asset {role} path is invalid")
        known_checkpoint_sha256 = _require_sha256(
            asset.get("checkpoint_sha256"),
            label=f"known asset {role} checkpoint_sha256",
        )
        if (
            known_checkpoint_sha256
            != APPROVED_HU_POLICY_VALUE_CHECKPOINTS[(turn, actor)]
        ):
            raise ValueError(
                f"known asset {role} checkpoint is not the approved exact hash"
            )
        binding = role_bindings[role]
        _validated_identity_evaluator(
            evaluators[(turn, actor)],
            role=role,
            expected_binding=binding,
            expected_checkpoint_sha256=known_checkpoint_sha256,
        )
        temperature = role_temperatures[role]
        child = TorchPolicyValueBehaviorModel(
            root / relative_path,
            model_id=(
                f"fixed_point_sharded_bootstrap_{role}_policyvalue_q32_"
                f"{artifact_sha256[:12]}"
            ),
            supported_turns=(turn,),
            supported_actors=(actor,),
            training_scope_id=f"hu_{role}_visible_opponent_board_2m_v1",
            temperature=temperature,
            quantization_denominator=quantization_denominator,
            expected_checkpoint_sha256=known_checkpoint_sha256,
        )
        child_sha256, child_manifest = _validated_runtime_child(
            child,
            turn=turn,
            actor=actor,
            role=role,
            binding=binding,
            temperature=temperature,
            quantization_denominator=quantization_denominator,
        )
        routes[(turn, actor)] = child
        route_manifest.append(
            {
                "turn": turn,
                "actor": actor,
                "role": role,
                "temperature": temperature,
                "checkpoint_sha256": binding["checkpoint_sha256"],
                "calibration_source_model_sha256": binding["model_sha256"],
                "row_extractor_sha256": binding["row_extractor_sha256"],
                "adapter_source_sha256": binding["adapter_source_sha256"],
                "child_model_id": child.model_id,
                "child_model_sha256": child_sha256,
                "child_model_type": child_manifest.get("model_type"),
            }
        )

    gate_config = _require_mapping(
        verified_artifact.get("gate_config"), label="gate_config"
    )
    gate_result = _require_mapping(
        verified_artifact.get("gate_result"), label="gate_result"
    )
    source_hashes = _require_mapping(
        verified_artifact.get("source_hashes"), label="source_hashes"
    )
    manifest: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "model_id": model_id,
        # Keep the existing type exactly so the fixed-point iteration router
        # accepts this object without an adapter or a weakened type check.
        "model_type": BOOTSTRAP_MODEL_TYPE,
        "bootstrap_evidence_type": SHARDED_BOOTSTRAP_EVIDENCE_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "bb_first": True,
        "rules_version": RULES_VERSION,
        "calibrated": True,
        "calibration_verified_from_raw": True,
        "calibration_verified_from_shards": True,
        "bounded_memory_verification": True,
        "raw_json_corpus_retained_in_memory": False,
        "source_calibration_schema": SHARDED_CALIBRATION_SCHEMA,
        "source_calibration_promotion_eligible": (
            source_calibration_promotion_eligible
        ),
        "source_calibration_all_required_gates_passed": calibration_gate_passed,
        "source_calibration_gate_failure_count": len(gate_failures),
        "source_calibration_gate_failures_sha256": canonical_sha256(
            list(gate_failures)
        ),
        "source_promotion_required_at_construction": require_promoted_source,
        "promotion_eligible": False,
        "fixed_point_bootstrap_only": True,
        "strategic_strength": False,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "route_scope": BOOTSTRAP_ROUTE_SCOPE,
        "no_fallback": True,
        "t3_bb_route_included": False,
        "t3_bb_likelihood_binding_included": False,
        "calibrated_behavior_bridge_included": False,
        "temperature_application": (
            "checkpoint_logits_div_temperature_then_softmax"
        ),
        "probability_quantization": "largest_remainder_exact_q32",
        "quantization_denominator": quantization_denominator,
        "calibration_artifact_sha256": artifact_sha256,
        "calibration_artifact_file_sha256": artifact_file_sha256,
        "calibration_gate_config_sha256": _require_sha256(
            gate_config.get("gate_config_sha256"), label="gate_config_sha256"
        ),
        "calibration_gate_result_sha256": _require_sha256(
            gate_result.get("gate_result_sha256"), label="gate_result_sha256"
        ),
        "calibration_source_hashes": canonical_snapshot(source_hashes),
        "sharded_input_hash_bindings": input_hash_bindings,
        "sharded_input_hash_binding_sha256": canonical_sha256(
            input_hash_bindings
        ),
        "evidence_path_policy": (
            "paths_not_identity_each_resolved_tree_freshly_content_verified"
        ),
        "raw_record_set_sha256": input_hash_bindings["natural"][
            "raw_record_set_sha256"
        ],
        "evaluation_row_set_sha256": input_hash_bindings["natural"][
            "evaluation_row_set_sha256"
        ],
        "challenge_raw_record_set_sha256": input_hash_bindings["challenge"][
            "raw_record_set_sha256"
        ],
        "challenge_evaluation_row_set_sha256": input_hash_bindings["challenge"][
            "evaluation_row_set_sha256"
        ],
        "routes": sorted(
            route_manifest, key=lambda row: (row["turn"], row["actor"])
        ),
    }
    return FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch(
        routes,
        manifest,
        _construction_token=bootstrap_runtime._CONSTRUCTION_TOKEN,
    )


# Shorter public spelling for command/orchestrator callers.  It is an exact
# alias, not a second construction path.
build_sharded_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch = (
    build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch_from_shards
)


__all__ = [
    "SHARDED_BOOTSTRAP_EVIDENCE_TYPE",
    "build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch_from_shards",
    "build_sharded_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch",
]
