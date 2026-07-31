"""Fail-closed production holdout evaluator for shared-infoset T3 candidates.

The evaluator deliberately keeps the physical world and the public policy on
opposite sides of a small boundary.  Candidate root and continuation actions
are looked up only by exact :class:`InfoSetKey`, with a missing continuation
failing closed; posterior particles and later draws remain inside a validated
:class:`FullCardGenerativeAdapter`.  Every legal root action is evaluated once
with frozen-candidate continuation and separately with the locked uniform
reference policy.  Candidate/reference, all root actions, and both seat-swap
roots share physical chance entropy, while policy sampling uses a separate
domain.

All durable files are one-line canonical JSON, written atomically and read
back after replacement.  Root/range inputs are copied into an immutable,
content-addressed bundle.  Evaluation work is resumable in exactly one
root-by-evaluation-seed shard, while the bundle manifest is written last.
Every artifact is algorithm-validation-only, explicitly non-promoting, and
makes no exploitability or unseen-root generalization claim.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import ai.engine.action_space as action_space_impl
import ai.tutor.exact_late as exact_late_impl
import ai.tutor.t3_hu_full_card_mccfr as full_card_impl
import ai.tutor.t3_hu_public_tree as public_tree_impl
from ai.tutor.promotion_gate_m3_full_card_strength import (
    REQUIRED_AUDITS,
    REQUIRED_STRATA,
    canonical_json,
    canonical_sha256,
    self_hash,
)
from ai.tutor.t3_bb_fixed_point_gate_v2 import reconstruct_infoset_key
from ai.tutor.t3_bb_fixed_point_gate_v2 import visible_joker_count
from ai.tutor.t3_bb_range_evidence import read_restricted_range_evidence
from ai.tutor.t3_hu_full_card_mccfr import (
    DRAW_SAMPLING_CONTRACT,
    POLICY_IDENTITY_CONTRACT,
    ROOT_SAMPLING_CONTRACT,
    FullCardGenerativeAdapter,
)
from ai.tutor.t3_hu_full_card_range import FullCardRange, verify_full_card_range
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import (
    PendingChanceState,
    PublicTreeDecisionState,
    PublicTreeTerminalState,
)
from ai.tutor.t3_shared_multi_root_strength_gate import (
    _verify_root_manifest,
    build_holdout_evaluator_manifest,
    verify_public_strategy_profile,
    verify_shared_multi_root_candidate_bundle,
)


SOURCE_MANIFEST_SCHEMA = "ofc_t3_production_holdout_source/v1"
ROOT_RANGE_BUNDLE_SCHEMA = "ofc_t3_production_holdout_root_range_bundle/v1"
SHARD_SCHEMA = "ofc_t3_production_holdout_shard/v1"
EVALUATION_BUNDLE_SCHEMA = "ofc_t3_production_holdout_evaluation_bundle/v1"
REFERENCE_POLICY_CONTRACT = "uniform_legal_public_continuation_v1"
EVALUATOR_METHOD = "canonical_full_card_candidate_reference_paired_crn_v2"

SOURCE_PATHS = (
    "ai/config/fl_ev.json",
    "ai/engine/action_space.py",
    "ai/engine/encoding.py",
    "ai/engine/game_engine.py",
    "ai/engine/scoring.py",
    "ai/engine/turn_order.py",
    "ai/mcts/rollout_evaluator.py",
    "ai/tutor/exact_late.py",
    "ai/tutor/promotion_gate_m3_full_card_strength.py",
    "ai/tutor/t3_bb_fixed_point_gate_v2.py",
    "ai/tutor/t3_bb_range_evidence.py",
    "ai/tutor/t3_hu_full_card_mccfr.py",
    "ai/tutor/t3_hu_full_card_range.py",
    "ai/tutor/t3_hu_public_cfr.py",
    "ai/tutor/t3_hu_public_tree.py",
    "ai/tutor/t3_production_holdout_evaluator.py",
    "ai/tutor/t3_shared_multi_root_strength_gate.py",
)

_ROOT_RANGE_MANIFEST_NAME = "root-range-bundle.json"
_EVALUATION_MANIFEST_NAME = "evaluation-bundle.json"
_CANDIDATE_RELATIVE_PATH = "inputs/candidate-bundle.json"
_RANGE_FILENAME = "restricted-range.json"
_FLOAT_TOL = 1e-12


class ProductionHoldoutError(ValueError):
    """An immutable input or production holdout artifact failed verification."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionHoldoutError(f"{label}: object required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ProductionHoldoutError(
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
        raise ProductionHoldoutError(f"{label}: lowercase SHA256 required")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductionHoldoutError(f"{label}: integer >= {minimum} required")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionHoldoutError(f"{label}: finite number required")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ProductionHoldoutError(f"{label}: invalid finite number")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionHoldoutError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical(path: str | Path) -> Any:
    target = Path(path)
    try:
        raw = target.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProductionHoldoutError(f"cannot read {target}: {exc}") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise ProductionHoldoutError(f"{target}: one canonical JSON line required")
    try:
        value = json.loads(text[:-1], object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ProductionHoldoutError(f"{target}: invalid JSON: {exc}") from exc
    if canonical_json(value) != text[:-1]:
        raise ProductionHoldoutError(f"{target}: noncanonical JSON")
    return value


def _atomic_write(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    if _read_canonical(target) != value:
        raise ProductionHoldoutError(f"{target}: atomic write readback mismatch")
    return target


def _regular_file_below(root: Path, relative_path: str, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ProductionHoldoutError(f"{label}: canonical POSIX relative path required")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_path:
        raise ProductionHoldoutError(f"{label}: unsafe relative path")
    if root.is_symlink():
        raise ProductionHoldoutError(f"{label}: symlink root forbidden")
    root_resolved = root.resolve()
    target = root / relative
    cursor = target
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ProductionHoldoutError(f"{label}: symlinks forbidden")
        cursor = cursor.parent
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ProductionHoldoutError(f"{label}: file escapes or is missing") from exc
    if not resolved.is_file():
        raise ProductionHoldoutError(f"{label}: regular file required")
    return resolved


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _list_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProductionHoldoutError(f"{path}: symlinks forbidden")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _verify_live_canonical_bindings(workspace_root: Path) -> None:
    """Reject runtime monkeypatches that a file hash alone cannot observe."""

    identity_bindings = (
        (FullCardGenerativeAdapter, full_card_impl.FullCardGenerativeAdapter, "adapter class"),
        (full_card_impl.get_turn_actions, action_space_impl.get_turn_actions, "legal action generator"),
        (full_card_impl.action_key, exact_late_impl.action_key, "stable action key"),
        (full_card_impl.terminal_metrics, exact_late_impl.terminal_metrics, "terminal scorer"),
        (
            full_card_impl.apply_public_tree_action,
            public_tree_impl.apply_public_tree_action,
            "canonical public transition",
        ),
        (
            full_card_impl.apply_public_tree_terminal_action,
            public_tree_impl.apply_public_tree_terminal_action,
            "canonical terminal transition",
        ),
        (
            full_card_impl.resolve_supplied_chance,
            public_tree_impl.resolve_supplied_chance,
            "canonical chance transition",
        ),
    )
    for actual, expected, label in identity_bindings:
        if actual is not expected:
            raise ProductionHoldoutError(f"live canonical {label} binding was replaced")
    method_names = (
        "sample_root_for_traversal",
        "information_key",
        "legal_actions",
        "apply_action_id",
        "sample_next_draw",
        "terminal_metrics_bb",
        "terminal_utility_bb",
    )
    expected_file = (workspace_root / "ai/tutor/t3_hu_full_card_mccfr.py").resolve()
    for name in method_names:
        method = getattr(FullCardGenerativeAdapter, name)
        function = getattr(method, "__func__", method)
        source_file = inspect.getsourcefile(function)
        if source_file is None or Path(source_file).resolve() != expected_file:
            raise ProductionHoldoutError(
                f"live canonical adapter method {name} was replaced"
            )


def build_source_manifest(workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Hash the exact implementation/rules surface used by the evaluator."""

    root = Path(workspace_root) if workspace_root is not None else _repo_root()
    _verify_live_canonical_bindings(root)
    files = []
    for relative in SOURCE_PATHS:
        path = _regular_file_below(root, relative, label=f"source.{relative}")
        files.append({"path": relative, "sha256": _file_sha256(path)})
    manifest: dict[str, Any] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "live_binding_contract": "canonical_full_card_runtime_identity_v1",
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "files": files,
    }
    manifest["source_manifest_sha256"] = self_hash(
        manifest, "source_manifest_sha256"
    )
    return verify_source_manifest(manifest, workspace_root=root)


def verify_source_manifest(
    value: Any, *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    raw = _require_mapping(value, label="source_manifest")
    _exact_keys(
        raw,
        {
            "schema",
            "live_binding_contract",
            "promotion_eligible",
            "exact_exploitability_computed",
            "files",
            "source_manifest_sha256",
        },
        label="source_manifest",
    )
    if raw.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ProductionHoldoutError("source_manifest.schema: mismatch")
    if raw.get("live_binding_contract") != "canonical_full_card_runtime_identity_v1":
        raise ProductionHoldoutError("source_manifest.live_binding_contract: mismatch")
    if raw.get("promotion_eligible") is not False:
        raise ProductionHoldoutError("source_manifest.promotion_eligible: must be false")
    if raw.get("exact_exploitability_computed") is not False:
        raise ProductionHoldoutError(
            "source_manifest.exact_exploitability_computed: must be false"
        )
    if raw.get("source_manifest_sha256") != self_hash(raw, "source_manifest_sha256"):
        raise ProductionHoldoutError("source_manifest: self-hash mismatch")
    rows = raw.get("files")
    if not isinstance(rows, list) or len(rows) != len(SOURCE_PATHS):
        raise ProductionHoldoutError("source_manifest.files: exact source set required")
    paths: list[str] = []
    root = Path(workspace_root) if workspace_root is not None else _repo_root()
    _verify_live_canonical_bindings(root)
    for index, row_value in enumerate(rows):
        row = _require_mapping(row_value, label=f"source_manifest.files[{index}]")
        _exact_keys(row, {"path", "sha256"}, label=f"source_manifest.files[{index}]")
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ProductionHoldoutError("source manifest path must be a string")
        paths.append(relative)
        supplied = _sha(row.get("sha256"), label=f"source.{relative}.sha256")
        current = _file_sha256(_regular_file_below(root, relative, label=f"source.{relative}"))
        if supplied != current:
            raise ProductionHoldoutError(f"source.{relative}: stale source binding")
    if tuple(paths) != SOURCE_PATHS:
        raise ProductionHoldoutError("source_manifest.files: canonical exact order required")
    return copy.deepcopy(dict(raw))


def write_candidate_bundle(path: str | Path, bundle: Mapping[str, Any]) -> Path:
    verified = verify_shared_multi_root_candidate_bundle(bundle)
    _atomic_write(path, verified)
    read_candidate_bundle(path)
    return Path(path)


def read_candidate_bundle(path: str | Path) -> dict[str, Any]:
    return verify_shared_multi_root_candidate_bundle(_read_canonical(path))


def _recall(value: Mapping[str, Any]) -> PrivateRecall:
    dealt = tuple(
        (int(row["turn"]), tuple(str(card) for card in row["cards"]))
        for row in value["dealt_by_turn"]
    )
    discards = tuple(
        (int(row["turn"]), str(row["card"]))
        for row in value["discards_by_turn"]
    )
    recall = PrivateRecall(dealt, discards)
    if recall.to_canonical_dict() != dict(value):
        raise ProductionHoldoutError("range recall is not canonical")
    return recall


def _materialize_range(artifact: Mapping[str, Any]) -> tuple[InfoSetKey, FullCardRange]:
    observation = reconstruct_infoset_key(artifact["observation"])
    particles = tuple(
        JointParticle(
            bb_recall=_recall(row["bb_recall"]),
            btn_recall=_recall(row["btn_recall"]),
            undealt_cards=tuple(row["undealt_cards"]),
            weight=Fraction(row["weight"]),
        )
        for row in artifact["particles"]
    )
    full_range = FullCardRange(
        observation_digest=artifact["observation_digest"],
        particles=particles,
        particle_commitments=tuple(artifact["particle_commitments"]),
        behavior_model_id=artifact["behavior_model_id"],
        behavior_model_sha256=artifact["behavior_model_sha256"],
        epsilon=Fraction(artifact["epsilon"]),
        evidence_normalizer=Fraction(
            artifact["opponent_public_evidence_normalizer_exact"]
        ),
        effective_sample_size=Fraction(artifact["effective_sample_size_exact"]),
        range_sha256=artifact["range_sha256"],
        range_content_sha256=artifact["range_content_sha256"],
        range_build_sha256=artifact["range_build_sha256"],
        metadata={
            "content_manifest": copy.deepcopy(artifact["content_manifest"]),
            "build_manifest": copy.deepcopy(artifact["build_manifest"]),
            "behavior_model_manifest": copy.deepcopy(artifact["behavior_model_manifest"]),
            "opponent_public_evidence_normalizer_exact": artifact[
                "opponent_public_evidence_normalizer_exact"
            ],
            "opponent_public_evidence_scope": artifact["build_manifest"].get(
                "opponent_public_evidence_scope"
            ),
        },
    )
    verify_full_card_range(observation, full_range)
    return observation, full_range


def _verify_root_observation_semantics(
    root: Mapping[str, Any], observation: InfoSetKey
) -> None:
    if observation.actor != root.get("actor"):
        raise ProductionHoldoutError("root actor disagrees with real InfoSetKey")
    expected_phase = "t3_first" if observation.actor == "bb" else "t3_second"
    if observation.turn != 3 or observation.phase != expected_phase:
        raise ProductionHoldoutError("holdout root must be its canonical T3 decision")
    joker_count = visible_joker_count(observation)
    if joker_count != root.get("visible_joker_count"):
        raise ProductionHoldoutError(
            "root visible Joker stratum disagrees with real InfoSetKey"
        )
    if root.get("stratum") != f"{observation.actor}_joker{joker_count}":
        raise ProductionHoldoutError("root stratum disagrees with real InfoSetKey")


def _range_relative(root_commitment: str) -> str:
    return f"ranges/{root_commitment}/{_RANGE_FILENAME}"


def build_root_range_bundle(
    *,
    root_manifest: Mapping[str, Any],
    range_evidence_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy and lock real restricted FullCardRange assets; manifest is last."""

    roots, roots_by_id, _counts = _verify_root_manifest(
        root_manifest, min_roots_per_stratum=1
    )
    if set(range_evidence_paths) != set(roots_by_id):
        raise ProductionHoldoutError("range_evidence_paths: exact holdout root set required")
    output = Path(output_dir)
    manifest_path = output / _ROOT_RANGE_MANIFEST_NAME
    if manifest_path.exists():
        existing = verify_root_range_bundle(output, workspace_root=workspace_root)[0]
        if existing["root_manifest_sha256"] != roots["root_manifest_sha256"]:
            raise ProductionHoldoutError("existing root/range bundle binds another root manifest")
        return existing
    if _list_files(output):
        raise ProductionHoldoutError("root/range bundle has orphan files before manifest")

    source = build_source_manifest(workspace_root)
    entries: list[dict[str, Any]] = []
    for root in roots["roots"]:
        source_path = Path(range_evidence_paths[root["root_id"]])
        artifact, _audit = read_restricted_range_evidence(source_path)
        observation, full_range = _materialize_range(artifact)
        _verify_root_observation_semantics(root, observation)
        if artifact["root_id"] != root["root_id"]:
            raise ProductionHoldoutError("range root_id binding mismatch")
        if artifact["root_commitment_sha256"] != root["root_commitment_sha256"]:
            raise ProductionHoldoutError("range root commitment binding mismatch")
        if observation.digest() != root["observation_digest"]:
            raise ProductionHoldoutError("range InfoSetKey/root digest mismatch")
        relative = _range_relative(root["root_commitment_sha256"])
        target = output / relative
        _atomic_write(target, artifact)
        copied, _copied_audit = read_restricted_range_evidence(target)
        if copied != artifact:
            raise ProductionHoldoutError("restricted range copy readback mismatch")
        entries.append(
            {
                "root": copy.deepcopy(root),
                "observation": observation.to_canonical_dict(),
                "observation_digest": observation.digest(),
                "range_asset": relative,
                "range_file_sha256": _file_sha256(target),
                "range_artifact_sha256": artifact["artifact_sha256"],
                "range_sha256": full_range.range_sha256,
                "range_content_sha256": full_range.range_content_sha256,
                "range_build_sha256": full_range.range_build_sha256,
                "range_solver_seed": artifact["solver_seed"],
                "particle_count": len(full_range.particles),
            }
        )
    bundle: dict[str, Any] = {
        "schema": ROOT_RANGE_BUNDLE_SCHEMA,
        "artifact_kind": "immutable_restricted_full_card_holdout_roots",
        "restricted_hidden_information": True,
        "immutable": True,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "root_manifest": roots,
        "root_manifest_sha256": roots["root_manifest_sha256"],
        "source_manifest": source,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "entries": entries,
    }
    bundle["bundle_sha256"] = self_hash(bundle, "bundle_sha256")
    _atomic_write(manifest_path, bundle)
    return verify_root_range_bundle(output, workspace_root=workspace_root)[0]


def verify_root_range_bundle(
    bundle_dir: str | Path, *, workspace_root: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, tuple[InfoSetKey, FullCardRange]]]:
    root = Path(bundle_dir)
    raw = _require_mapping(_read_canonical(root / _ROOT_RANGE_MANIFEST_NAME), label="root_range_bundle")
    expected_keys = {
        "schema", "artifact_kind", "restricted_hidden_information", "immutable",
        "promotion_eligible", "exact_exploitability_computed", "root_manifest",
        "root_manifest_sha256", "source_manifest", "source_manifest_sha256",
        "entries", "bundle_sha256",
    }
    _exact_keys(raw, expected_keys, label="root_range_bundle")
    fixed = {
        "schema": ROOT_RANGE_BUNDLE_SCHEMA,
        "artifact_kind": "immutable_restricted_full_card_holdout_roots",
        "restricted_hidden_information": True,
        "immutable": True,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
    }
    for field, wanted in fixed.items():
        if raw.get(field) != wanted:
            raise ProductionHoldoutError(f"root_range_bundle.{field}: mismatch")
    if raw.get("bundle_sha256") != self_hash(raw, "bundle_sha256"):
        raise ProductionHoldoutError("root_range_bundle: self-hash mismatch")
    roots, by_id, _counts = _verify_root_manifest(raw.get("root_manifest"), min_roots_per_stratum=1)
    if raw.get("root_manifest_sha256") != roots["root_manifest_sha256"]:
        raise ProductionHoldoutError("root_range_bundle: root manifest binding mismatch")
    source = verify_source_manifest(raw.get("source_manifest"), workspace_root=workspace_root)
    if raw.get("source_manifest_sha256") != source["source_manifest_sha256"]:
        raise ProductionHoldoutError("root_range_bundle: source binding mismatch")
    entries = raw.get("entries")
    if not isinstance(entries, list) or len(entries) != len(by_id):
        raise ProductionHoldoutError("root_range_bundle.entries: exact root count required")
    materialized: dict[str, tuple[InfoSetKey, FullCardRange]] = {}
    seen_range_content: set[str] = set()
    seen_range_build: set[str] = set()
    expected_files = {_ROOT_RANGE_MANIFEST_NAME}
    order: list[tuple[str, str]] = []
    entry_keys = {
        "root", "observation", "observation_digest", "range_asset",
        "range_file_sha256", "range_artifact_sha256", "range_sha256",
        "range_content_sha256", "range_build_sha256", "range_solver_seed",
        "particle_count",
    }
    for index, entry_value in enumerate(entries):
        entry = _require_mapping(entry_value, label=f"root_range_bundle.entries[{index}]")
        _exact_keys(entry, entry_keys, label=f"root_range_bundle.entries[{index}]")
        root_record = _require_mapping(entry.get("root"), label=f"entries[{index}].root")
        root_id = root_record.get("root_id")
        expected_root = by_id.get(str(root_id))
        if expected_root is None or dict(root_record) != expected_root:
            raise ProductionHoldoutError("root_range_bundle entry root mismatch")
        relative = _range_relative(expected_root["root_commitment_sha256"])
        if entry.get("range_asset") != relative:
            raise ProductionHoldoutError("root_range_bundle range path mismatch")
        expected_files.add(relative)
        path = _regular_file_below(root, relative, label=f"range.{root_id}")
        if entry.get("range_file_sha256") != _file_sha256(path):
            raise ProductionHoldoutError("restricted range file hash mismatch")
        artifact, _audit = read_restricted_range_evidence(path)
        observation, full_range = _materialize_range(artifact)
        _verify_root_observation_semantics(expected_root, observation)
        bindings = {
            "range_artifact_sha256": artifact["artifact_sha256"],
            "range_sha256": full_range.range_sha256,
            "range_content_sha256": full_range.range_content_sha256,
            "range_build_sha256": full_range.range_build_sha256,
            "range_solver_seed": artifact["solver_seed"],
            "particle_count": len(full_range.particles),
            "observation_digest": observation.digest(),
        }
        for field, wanted in bindings.items():
            if entry.get(field) != wanted:
                raise ProductionHoldoutError(f"root_range_bundle entry {field} mismatch")
        if full_range.range_content_sha256 in seen_range_content:
            raise ProductionHoldoutError("root_range_bundle: range content asset reused")
        if full_range.range_build_sha256 in seen_range_build:
            raise ProductionHoldoutError("root_range_bundle: range build asset reused")
        seen_range_content.add(full_range.range_content_sha256)
        seen_range_build.add(full_range.range_build_sha256)
        if artifact["root_id"] != root_id or artifact["root_commitment_sha256"] != expected_root["root_commitment_sha256"]:
            raise ProductionHoldoutError("restricted range root binding mismatch")
        if observation.to_canonical_dict() != entry.get("observation") or observation.digest() != expected_root["observation_digest"]:
            raise ProductionHoldoutError("restricted range real InfoSetKey binding mismatch")
        materialized[str(root_id)] = (observation, full_range)
        order.append((expected_root["stratum"], str(root_id)))
    if order != sorted(order):
        raise ProductionHoldoutError("root_range_bundle.entries: canonical root order required")
    if _list_files(root) != expected_files:
        raise ProductionHoldoutError("root_range_bundle: gap or orphan file detected")
    return copy.deepcopy(dict(raw)), materialized


def reference_policy_sha256() -> str:
    return canonical_sha256(
        {
            "contract": REFERENCE_POLICY_CONTRACT,
            "policy_input": "InfoSetKey_and_lexical_legal_action_ids_only",
            "root_distribution": "uniform_legal",
            "continuation_distribution": "uniform_legal",
            "hidden_cards_passed_to_policy": False,
        }
    )


def _distribution(value: Any, *, label: str) -> dict[str, float]:
    raw = _require_mapping(value, label=label)
    if not raw:
        raise ProductionHoldoutError(f"{label}: non-empty distribution required")
    result: dict[str, float] = {}
    for action_id, probability_value in raw.items():
        if not isinstance(action_id, str) or not action_id:
            raise ProductionHoldoutError(f"{label}: non-empty action IDs required")
        probability = _finite(probability_value, label=f"{label}.{action_id}", nonnegative=True)
        result[action_id] = probability
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=_FLOAT_TOL):
        raise ProductionHoldoutError(f"{label}: probabilities must sum to one")
    return result


def _candidate_policy_profile(
    candidate: Mapping[str, Any], *, stratum: str, observation: InfoSetKey
) -> tuple[
    dict[str, float], dict[InfoSetKey, dict[str, float]], Mapping[str, Any]
]:
    artifact = candidate["strategies"][stratum]
    _profile, distributions, infosets = verify_public_strategy_profile(
        artifact["strategy_profile"]
    )
    # The only lookup identity is the reconstructed public InfoSetKey.  Root
    # IDs, particle commitments and remaining cards do not enter this map.
    policy_by_key = {
        infosets[digest]: distributions[digest] for digest in sorted(distributions)
    }
    policy = policy_by_key.get(observation)
    if policy is None:
        raise ProductionHoldoutError(
            f"candidate {stratum} has no exact InfoSetKey for holdout root"
        )
    return copy.deepcopy(policy), copy.deepcopy(policy_by_key), artifact


def _randbelow_public(rng: random.Random, upper: int) -> int:
    if upper <= 0:
        raise ProductionHoldoutError("public action sampler needs positive support")
    if upper == 1:
        return 0
    bits = (upper - 1).bit_length()
    while True:
        value = rng.getrandbits(bits)
        if value < upper:
            return value


def _uniform_public_action(
    key: InfoSetKey, legal_action_ids: Sequence[str], rng: random.Random
) -> str:
    """Choose using public information only; physical state is unrepresentable."""

    if not isinstance(key, InfoSetKey):
        raise TypeError("uniform continuation policy requires InfoSetKey")
    key.canonical_json()
    action_ids = tuple(str(action_id) for action_id in legal_action_ids)
    if not action_ids or action_ids != tuple(sorted(set(action_ids))):
        raise ProductionHoldoutError("continuation legal actions must be canonical")
    return action_ids[_randbelow_public(rng, len(action_ids))]


def _sample_public_distribution(
    key: InfoSetKey,
    legal_action_ids: Sequence[str],
    distribution: Mapping[str, float],
    rng: random.Random,
) -> str:
    if not isinstance(key, InfoSetKey):
        raise TypeError("candidate continuation policy requires InfoSetKey")
    key.canonical_json()
    action_ids = tuple(str(action_id) for action_id in legal_action_ids)
    if action_ids != tuple(sorted(set(action_ids))) or set(distribution) != set(action_ids):
        raise ProductionHoldoutError(
            "candidate continuation policy/legal action support mismatch"
        )
    threshold = rng.random()
    cumulative = 0.0
    for action_id in action_ids:
        probability = _finite(
            distribution[action_id],
            label=f"candidate continuation.{action_id}",
            nonnegative=True,
        )
        cumulative += probability
        if threshold < cumulative:
            return action_id
    if not math.isclose(cumulative, 1.0, rel_tol=0.0, abs_tol=_FLOAT_TOL):
        raise ProductionHoldoutError("candidate continuation probabilities do not sum to one")
    return action_ids[-1]


def _physical_entropy_seed(
    *,
    payoff_seed: int,
    seat_swap_pair_id: str,
    sample_index: int,
    stage: str,
    chance_index: int,
) -> int:
    payload = {
        "domain": "ofc_t3_holdout_paired_physical_chance_v2",
        "payoff_seed": payoff_seed,
        "seat_swap_pair_id": seat_swap_pair_id,
        "sample_index": sample_index,
        "stage": stage,
        "chance_index": chance_index,
    }
    return int(canonical_sha256(payload), 16)


def _policy_entropy_seed(
    *,
    payoff_seed: int,
    root_commitment: str,
    root_action_id: str,
    sample_index: int,
    policy_kind: str,
    decision_index: int,
    infoset_digest: str,
) -> int:
    return int(
        canonical_sha256(
            {
                "domain": "ofc_t3_holdout_public_policy_sample_v2",
                "payoff_seed": payoff_seed,
                "root_commitment_sha256": root_commitment,
                "root_action_id": root_action_id,
                "sample_index": sample_index,
                "policy_kind": policy_kind,
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
    root: Mapping[str, Any],
    payoff_seed: int,
    sample_index: int,
    policy_kind: str,
    candidate_policy_by_key: Mapping[InfoSetKey, Mapping[str, float]],
    encountered_candidate_infosets: set[str],
) -> float:
    state: PublicTreeDecisionState | PendingChanceState | PublicTreeTerminalState
    state = adapter.sample_root_for_traversal(
        random.Random(
            _physical_entropy_seed(
                payoff_seed=payoff_seed,
                seat_swap_pair_id=root["seat_swap_pair_id"],
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
                        payoff_seed=payoff_seed,
                        seat_swap_pair_id=root["seat_swap_pair_id"],
                        sample_index=sample_index,
                        stage="future_draw",
                        chance_index=chance_index,
                    )
                ),
            ).state
            chance_index += 1
            continue
        if not isinstance(state, PublicTreeDecisionState):  # pragma: no cover
            raise AssertionError("unsupported physical rollout state")
        public_key = adapter.information_key(state)
        legal_action_ids = tuple(action_id for action_id, _ in adapter.legal_actions(state))
        policy_rng = random.Random(
            _policy_entropy_seed(
                payoff_seed=payoff_seed,
                root_commitment=root["root_commitment_sha256"],
                root_action_id=root_action_id,
                sample_index=sample_index,
                policy_kind=policy_kind,
                decision_index=decision_index,
                infoset_digest=public_key.digest(),
            )
        )
        if policy_kind == "candidate":
            encountered_candidate_infosets.add(public_key.digest())
            distribution = candidate_policy_by_key.get(public_key)
            if distribution is None:
                raise ProductionHoldoutError(
                    "candidate continuation missing exact InfoSetKey; fallback forbidden"
                )
            selected = _sample_public_distribution(
                public_key, legal_action_ids, distribution, policy_rng
            )
        elif policy_kind == "reference_uniform":
            selected = _uniform_public_action(public_key, legal_action_ids, policy_rng)
        else:  # pragma: no cover
            raise AssertionError("unknown continuation policy")
        state = adapter.apply_action_id(state, selected)
        decision_index += 1
    utility_bb = adapter.terminal_utility_bb(state)
    return utility_bb if root["actor"] == "bb" else -utility_bb


def _aggregate_statistics(values: Sequence[float]) -> tuple[dict[str, Any], float, float]:
    if len(values) < 2:
        raise ProductionHoldoutError("at least two payoff samples per action required")
    total = math.fsum(values)
    total_squares = math.fsum(value * value for value in values)
    mean = total / len(values)
    centered = max(0.0, total_squares - total * total / len(values))
    sample_variance = centered / (len(values) - 1)
    standard_error = math.sqrt(sample_variance / len(values))
    return (
        {"count": len(values), "sum": total, "sum_squares": total_squares},
        mean,
        standard_error,
    )


def _weighted(distribution: Mapping[str, float], payoffs: Mapping[str, float]) -> float:
    return math.fsum(
        distribution[action_id] * payoffs[action_id] for action_id in sorted(payoffs)
    )


def _evaluate_raw_aggregates(
    *,
    observation: InfoSetKey,
    full_range: FullCardRange,
    root: Mapping[str, Any],
    payoff_seed: int,
    samples_per_action: int,
    root_policy: Mapping[str, float],
    candidate_policy_by_key: Mapping[InfoSetKey, Mapping[str, float]],
) -> dict[str, Any]:
    adapter = FullCardGenerativeAdapter(observation, full_range)
    sampled = adapter.sample_root_for_traversal(
        random.Random(
            _physical_entropy_seed(
                payoff_seed=payoff_seed,
                seat_swap_pair_id=root["seat_swap_pair_id"],
                sample_index=0,
                stage="root_posterior",
                chance_index=0,
            )
        )
    )
    legal_action_ids = tuple(action_id for action_id, _ in adapter.legal_actions(sampled.state))
    if set(root_policy) != set(legal_action_ids):
        raise ProductionHoldoutError("candidate root policy does not cover exact legal actions")
    # The probe is not part of an action payoff and would pollute the sampling
    # audit.  Recreate the adapter before the actual fixed-seed traversals.
    adapter = FullCardGenerativeAdapter(observation, full_range)
    candidate_values: dict[str, list[float]] = {
        action_id: [] for action_id in legal_action_ids
    }
    reference_values: dict[str, list[float]] = {
        action_id: [] for action_id in legal_action_ids
    }
    encountered_candidate_infosets = {observation.digest()}
    for action_id in legal_action_ids:
        for sample_index in range(samples_per_action):
            candidate_values[action_id].append(
                _rollout_root_action(
                    adapter=adapter,
                    root_action_id=action_id,
                    root=root,
                    payoff_seed=payoff_seed,
                    sample_index=sample_index,
                    policy_kind="candidate",
                    candidate_policy_by_key=candidate_policy_by_key,
                    encountered_candidate_infosets=encountered_candidate_infosets,
                )
            )
            reference_values[action_id].append(
                _rollout_root_action(
                    adapter=adapter,
                    root_action_id=action_id,
                    root=root,
                    payoff_seed=payoff_seed,
                    sample_index=sample_index,
                    policy_kind="reference_uniform",
                    candidate_policy_by_key=candidate_policy_by_key,
                    encountered_candidate_infosets=encountered_candidate_infosets,
                )
            )
    candidate_aggregates: dict[str, dict[str, Any]] = {}
    candidate_estimates: dict[str, float] = {}
    candidate_errors: dict[str, float] = {}
    reference_aggregates: dict[str, dict[str, Any]] = {}
    reference_estimates: dict[str, float] = {}
    reference_errors: dict[str, float] = {}
    for action_id in legal_action_ids:
        aggregate, mean, error = _aggregate_statistics(candidate_values[action_id])
        candidate_aggregates[action_id] = aggregate
        candidate_estimates[action_id] = mean
        candidate_errors[action_id] = error
        aggregate, mean, error = _aggregate_statistics(reference_values[action_id])
        reference_aggregates[action_id] = aggregate
        reference_estimates[action_id] = mean
        reference_errors[action_id] = error
    reference_distribution = {
        action_id: 1.0 / len(legal_action_ids) for action_id in legal_action_ids
    }
    paired_deltas = [
        math.fsum(
            root_policy[action_id] * candidate_values[action_id][sample_index]
            for action_id in legal_action_ids
        )
        - math.fsum(
            reference_distribution[action_id]
            * reference_values[action_id][sample_index]
            for action_id in legal_action_ids
        )
        for sample_index in range(samples_per_action)
    ]
    paired_aggregate, paired_mean, paired_error = _aggregate_statistics(paired_deltas)
    return {
        "candidate_aggregates": candidate_aggregates,
        "candidate_estimates": candidate_estimates,
        "candidate_errors": candidate_errors,
        "reference_aggregates": reference_aggregates,
        "reference_estimates": reference_estimates,
        "reference_errors": reference_errors,
        "reference_distribution": reference_distribution,
        "paired_aggregate": paired_aggregate,
        "paired_mean": paired_mean,
        "paired_error": paired_error,
        "required_candidate_infoset_digests": sorted(
            encountered_candidate_infosets
        ),
        "candidate_profile_infoset_count": len(candidate_policy_by_key),
        "sampling_audit": copy.deepcopy(dict(adapter.sampling_audit())),
    }


def _build_shard(
    *,
    root: Mapping[str, Any],
    observation: InfoSetKey,
    full_range: FullCardRange,
    range_entry: Mapping[str, Any],
    candidate: Mapping[str, Any],
    evaluator_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    evaluation_seed: int,
    payoff_seed: int,
    samples_per_action: int,
) -> dict[str, Any]:
    policy, candidate_policy_by_key, artifact = _candidate_policy_profile(
        candidate, stratum=root["stratum"], observation=observation
    )
    started = time.perf_counter_ns()
    results = _evaluate_raw_aggregates(
        observation=observation,
        full_range=full_range,
        root=root,
        payoff_seed=payoff_seed,
        samples_per_action=samples_per_action,
        root_policy=policy,
        candidate_policy_by_key=candidate_policy_by_key,
    )
    runtime_ms = max((time.perf_counter_ns() - started) / 1_000_000.0, 1e-9)
    aggregates = results["candidate_aggregates"]
    payoffs = results["candidate_estimates"]
    standard_errors = results["candidate_errors"]
    reference_aggregates = results["reference_aggregates"]
    reference_payoffs = results["reference_estimates"]
    reference_errors = results["reference_errors"]
    sampling_audit = results["sampling_audit"]
    action_ids = sorted(payoffs)
    reference = results["reference_distribution"]
    policy_payoff = _weighted(policy, payoffs)
    reference_payoff = _weighted(reference, reference_payoffs)
    candidate_uniform_payoff = _weighted(reference, payoffs)
    best = max(payoffs.values())
    expected_traversals = 2 * len(action_ids) * samples_per_action
    if sampling_audit["traversals"] != expected_traversals:
        raise ProductionHoldoutError("physical root sampling audit mismatch")
    shard: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "artifact_kind": "one_root_by_evaluation_seed_full_card_payoff_shard",
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "algorithm_validation_only": True,
        "production_promotion_supported": False,
        "root": copy.deepcopy(root),
        "policy_infoset_digest": observation.digest(),
        "range_artifact_sha256": range_entry["range_artifact_sha256"],
        "range_content_sha256": full_range.range_content_sha256,
        "range_build_sha256": full_range.range_build_sha256,
        "candidate_bundle_sha256": candidate["candidate_bundle_sha256"],
        "candidate_scope_manifest_sha256": candidate[
            "candidate_scope_manifest_sha256"
        ],
        "candidate_strategy_artifact_sha256": artifact["artifact_sha256"],
        "candidate_strategy_sha256": artifact["average_strategy_sha256"],
        "evaluator_manifest_sha256": evaluator_manifest["manifest_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "reference_policy_sha256": reference_policy_sha256(),
        "evaluator_method": EVALUATOR_METHOD,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "root_sampling_contract": ROOT_SAMPLING_CONTRACT,
        "draw_sampling_contract": DRAW_SAMPLING_CONTRACT,
        "candidate_continuation_policy_contract": (
            "frozen_candidate_exact_infoset_no_fallback_v1"
        ),
        "reference_continuation_policy_contract": REFERENCE_POLICY_CONTRACT,
        "common_physical_chance_contract": (
            "pair_payoff_sample_phase_common_random_numbers_v1"
        ),
        "candidate_selected_only_by_infoset_key": True,
        "candidate_missing_infoset_fallback": False,
        "required_candidate_infoset_digests": results[
            "required_candidate_infoset_digests"
        ],
        "candidate_profile_infoset_count": results[
            "candidate_profile_infoset_count"
        ],
        "candidate_missing_infoset_count": 0,
        "candidate_fallback_count": 0,
        "hidden_cards_passed_to_policy": False,
        "all_legal_root_actions_evaluated": True,
        "evaluation_seed": evaluation_seed,
        "payoff_sample_seed": payoff_seed,
        "samples_per_action": samples_per_action,
        "policy_action_distribution": policy,
        "reference_action_distribution": reference,
        "action_payoff_aggregates": aggregates,
        "action_payoff_estimates": payoffs,
        "action_payoff_sample_counts": {
            action_id: aggregates[action_id]["count"] for action_id in action_ids
        },
        "action_payoff_standard_errors": standard_errors,
        "max_action_payoff_standard_error": max(standard_errors.values()),
        "reference_action_payoff_aggregates": reference_aggregates,
        "reference_action_payoff_estimates": reference_payoffs,
        "reference_action_payoff_sample_counts": {
            action_id: reference_aggregates[action_id]["count"]
            for action_id in action_ids
        },
        "reference_action_payoff_standard_errors": reference_errors,
        "max_reference_action_payoff_standard_error": max(reference_errors.values()),
        "policy_payoff_estimate": policy_payoff,
        "reference_payoff_estimate": reference_payoff,
        "candidate_continuation_uniform_root_payoff_estimate": (
            candidate_uniform_payoff
        ),
        "policy_reference_paired_aggregate": results["paired_aggregate"],
        "policy_reference_delta_estimate": results["paired_mean"],
        "policy_reference_delta_standard_error": results["paired_error"],
        "best_action_payoff_estimate": best,
        "ev_regret_estimate": max(0.0, best - policy_payoff),
        "sampling_audit": copy.deepcopy(dict(sampling_audit)),
        "audits": {field: 0 for field in REQUIRED_AUDITS},
        "runtime_ms": runtime_ms,
    }
    shard["shard_sha256"] = self_hash(shard, "shard_sha256")
    return shard


def _shard_relative(root_commitment: str, evaluation_seed: int) -> str:
    return f"shards/{root_commitment}/{evaluation_seed}.json"


_SHARD_KEYS = {
    "schema", "artifact_kind", "promotion_eligible", "exact_exploitability_computed",
    "algorithm_validation_only", "production_promotion_supported",
    "root", "policy_infoset_digest", "range_artifact_sha256",
    "range_content_sha256", "range_build_sha256", "candidate_bundle_sha256",
    "candidate_scope_manifest_sha256",
    "candidate_strategy_artifact_sha256", "candidate_strategy_sha256",
    "evaluator_manifest_sha256", "source_manifest_sha256",
    "reference_policy_sha256", "evaluator_method", "policy_identity_contract",
    "root_sampling_contract", "draw_sampling_contract",
    "candidate_continuation_policy_contract", "reference_continuation_policy_contract",
    "common_physical_chance_contract", "candidate_selected_only_by_infoset_key",
    "candidate_missing_infoset_fallback", "hidden_cards_passed_to_policy",
    "required_candidate_infoset_digests", "candidate_profile_infoset_count",
    "candidate_missing_infoset_count", "candidate_fallback_count",
    "all_legal_root_actions_evaluated",
    "evaluation_seed", "payoff_sample_seed", "samples_per_action",
    "policy_action_distribution", "reference_action_distribution",
    "action_payoff_aggregates", "action_payoff_estimates",
    "action_payoff_sample_counts", "action_payoff_standard_errors",
    "max_action_payoff_standard_error", "reference_action_payoff_aggregates",
    "reference_action_payoff_estimates", "reference_action_payoff_sample_counts",
    "reference_action_payoff_standard_errors",
    "max_reference_action_payoff_standard_error", "policy_payoff_estimate",
    "reference_payoff_estimate", "candidate_continuation_uniform_root_payoff_estimate",
    "policy_reference_paired_aggregate", "policy_reference_delta_estimate",
    "policy_reference_delta_standard_error", "best_action_payoff_estimate",
    "ev_regret_estimate", "sampling_audit", "audits", "runtime_ms",
    "shard_sha256",
}


def _moment_view(
    value: Any, *, label: str, samples_per_action: int
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, float]]:
    raw = _require_mapping(value, label=label)
    aggregates: dict[str, dict[str, Any]] = {}
    estimates: dict[str, float] = {}
    errors: dict[str, float] = {}
    for action_id, aggregate_value in raw.items():
        if not isinstance(action_id, str) or not action_id:
            raise ProductionHoldoutError(f"{label}: invalid action ID")
        aggregate = _require_mapping(aggregate_value, label=f"{label}.{action_id}")
        _exact_keys(aggregate, {"count", "sum", "sum_squares"}, label=f"{label}.{action_id}")
        count = _integer(aggregate.get("count"), label=f"{label}.{action_id}.count", minimum=2)
        if count != samples_per_action:
            raise ProductionHoldoutError(f"{label}.{action_id}: sample target mismatch")
        total = _finite(aggregate.get("sum"), label=f"{label}.{action_id}.sum")
        squares = _finite(
            aggregate.get("sum_squares"),
            label=f"{label}.{action_id}.sum_squares",
            nonnegative=True,
        )
        minimum = total * total / count
        tolerance = _FLOAT_TOL * max(1.0, abs(squares), abs(minimum))
        if squares + tolerance < minimum:
            raise ProductionHoldoutError(f"{label}.{action_id}: impossible raw moments")
        centered = max(0.0, squares - minimum)
        estimates[action_id] = total / count
        errors[action_id] = math.sqrt(centered / (count - 1) / count)
        aggregates[action_id] = {"count": count, "sum": total, "sum_squares": squares}
    if not aggregates:
        raise ProductionHoldoutError(f"{label}: non-empty action set required")
    return aggregates, estimates, errors


def verify_shard(
    value: Any,
    *,
    root: Mapping[str, Any],
    observation: InfoSetKey,
    full_range: FullCardRange,
    range_entry: Mapping[str, Any],
    candidate: Mapping[str, Any],
    evaluator_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    evaluation_seed: int,
    payoff_seed: int,
    samples_per_action: int,
) -> dict[str, Any]:
    raw = _require_mapping(value, label="evaluation_shard")
    _exact_keys(raw, _SHARD_KEYS, label="evaluation_shard")
    fixed = {
        "schema": SHARD_SCHEMA,
        "artifact_kind": "one_root_by_evaluation_seed_full_card_payoff_shard",
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "algorithm_validation_only": True,
        "production_promotion_supported": False,
        "evaluator_method": EVALUATOR_METHOD,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "root_sampling_contract": ROOT_SAMPLING_CONTRACT,
        "draw_sampling_contract": DRAW_SAMPLING_CONTRACT,
        "candidate_continuation_policy_contract": (
            "frozen_candidate_exact_infoset_no_fallback_v1"
        ),
        "reference_continuation_policy_contract": REFERENCE_POLICY_CONTRACT,
        "common_physical_chance_contract": (
            "pair_payoff_sample_phase_common_random_numbers_v1"
        ),
        "candidate_selected_only_by_infoset_key": True,
        "candidate_missing_infoset_fallback": False,
        "candidate_missing_infoset_count": 0,
        "candidate_fallback_count": 0,
        "hidden_cards_passed_to_policy": False,
        "all_legal_root_actions_evaluated": True,
    }
    for field, wanted in fixed.items():
        if raw.get(field) != wanted:
            raise ProductionHoldoutError(f"evaluation_shard.{field}: mismatch")
    if raw.get("shard_sha256") != self_hash(raw, "shard_sha256"):
        raise ProductionHoldoutError("evaluation_shard: self-hash mismatch")
    if raw.get("root") != dict(root):
        raise ProductionHoldoutError("evaluation_shard: stale root binding")
    bindings = {
        "policy_infoset_digest": observation.digest(),
        "range_artifact_sha256": range_entry["range_artifact_sha256"],
        "range_content_sha256": full_range.range_content_sha256,
        "range_build_sha256": full_range.range_build_sha256,
        "candidate_bundle_sha256": candidate["candidate_bundle_sha256"],
        "candidate_scope_manifest_sha256": candidate[
            "candidate_scope_manifest_sha256"
        ],
        "evaluator_manifest_sha256": evaluator_manifest["manifest_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "reference_policy_sha256": reference_policy_sha256(),
        "evaluation_seed": evaluation_seed,
        "payoff_sample_seed": payoff_seed,
        "samples_per_action": samples_per_action,
    }
    policy, candidate_policy_by_key, artifact = _candidate_policy_profile(
        candidate, stratum=root["stratum"], observation=observation
    )
    required_infosets = raw.get("required_candidate_infoset_digests")
    if (
        not isinstance(required_infosets, list)
        or not required_infosets
        or required_infosets != sorted(set(required_infosets))
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in required_infosets
        )
        or observation.digest() not in required_infosets
    ):
        raise ProductionHoldoutError(
            "evaluation_shard: canonical required candidate infosets missing"
        )
    if raw.get("candidate_profile_infoset_count") != len(candidate_policy_by_key):
        raise ProductionHoldoutError(
            "evaluation_shard: candidate profile infoset count mismatch"
        )
    bindings.update(
        {
            "candidate_strategy_artifact_sha256": artifact["artifact_sha256"],
            "candidate_strategy_sha256": artifact["average_strategy_sha256"],
        }
    )
    for field, wanted in bindings.items():
        if raw.get(field) != wanted:
            raise ProductionHoldoutError(f"evaluation_shard.{field}: stale binding")
    published_policy = _distribution(raw.get("policy_action_distribution"), label="shard.policy")
    if published_policy != policy:
        raise ProductionHoldoutError("evaluation_shard: candidate policy mismatch")
    reference = _distribution(raw.get("reference_action_distribution"), label="shard.reference")
    aggregates, derived_payoffs, derived_errors = _moment_view(
        raw.get("action_payoff_aggregates"),
        label="shard.action_payoff_aggregates",
        samples_per_action=samples_per_action,
    )
    action_ids = set(aggregates)
    if set(policy) != action_ids or set(reference) != action_ids:
        raise ProductionHoldoutError("evaluation_shard: incomplete legal action support")
    expected_reference = {action_id: 1.0 / len(action_ids) for action_id in sorted(action_ids)}
    if reference != expected_reference:
        raise ProductionHoldoutError("evaluation_shard: reference policy mismatch")
    published_payoffs = _require_mapping(raw.get("action_payoff_estimates"), label="shard.payoffs")
    published_counts = _require_mapping(raw.get("action_payoff_sample_counts"), label="shard.counts")
    published_errors = _require_mapping(raw.get("action_payoff_standard_errors"), label="shard.errors")
    if set(published_payoffs) != action_ids or set(published_counts) != action_ids or set(published_errors) != action_ids:
        raise ProductionHoldoutError("evaluation_shard: payoff field action support mismatch")
    for action_id in sorted(action_ids):
        actual_payoff = _finite(published_payoffs[action_id], label=f"shard.payoffs.{action_id}")
        actual_error = _finite(published_errors[action_id], label=f"shard.errors.{action_id}", nonnegative=True)
        if not math.isclose(actual_payoff, derived_payoffs[action_id], rel_tol=0.0, abs_tol=_FLOAT_TOL):
            raise ProductionHoldoutError("evaluation_shard: payoff not rederived from raw moments")
        if published_counts[action_id] != samples_per_action:
            raise ProductionHoldoutError("evaluation_shard: sample count mismatch")
        if not math.isclose(actual_error, derived_errors[action_id], rel_tol=0.0, abs_tol=_FLOAT_TOL):
            raise ProductionHoldoutError("evaluation_shard: uncertainty not rederived")
    max_error = max(derived_errors.values())
    if not math.isclose(
        _finite(raw.get("max_action_payoff_standard_error"), label="shard.max_error", nonnegative=True),
        max_error,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOL,
    ):
        raise ProductionHoldoutError("evaluation_shard: max uncertainty mismatch")
    reference_aggregates, derived_reference_payoffs, derived_reference_errors = (
        _moment_view(
            raw.get("reference_action_payoff_aggregates"),
            label="shard.reference_action_payoff_aggregates",
            samples_per_action=samples_per_action,
        )
    )
    if set(reference_aggregates) != action_ids:
        raise ProductionHoldoutError(
            "evaluation_shard: reference payoff action support mismatch"
        )
    published_reference_payoffs = _require_mapping(
        raw.get("reference_action_payoff_estimates"),
        label="shard.reference_payoffs",
    )
    published_reference_counts = _require_mapping(
        raw.get("reference_action_payoff_sample_counts"),
        label="shard.reference_counts",
    )
    published_reference_errors = _require_mapping(
        raw.get("reference_action_payoff_standard_errors"),
        label="shard.reference_errors",
    )
    if (
        set(published_reference_payoffs) != action_ids
        or set(published_reference_counts) != action_ids
        or set(published_reference_errors) != action_ids
    ):
        raise ProductionHoldoutError(
            "evaluation_shard: reference payoff field action support mismatch"
        )
    for action_id in sorted(action_ids):
        if not math.isclose(
            _finite(
                published_reference_payoffs[action_id],
                label=f"shard.reference_payoffs.{action_id}",
            ),
            derived_reference_payoffs[action_id],
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        ):
            raise ProductionHoldoutError(
                "evaluation_shard: reference payoff not rederived from raw moments"
            )
        if published_reference_counts[action_id] != samples_per_action:
            raise ProductionHoldoutError("evaluation_shard: reference sample count mismatch")
        if not math.isclose(
            _finite(
                published_reference_errors[action_id],
                label=f"shard.reference_errors.{action_id}",
                nonnegative=True,
            ),
            derived_reference_errors[action_id],
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        ):
            raise ProductionHoldoutError(
                "evaluation_shard: reference uncertainty not rederived"
            )
    max_reference_error = max(derived_reference_errors.values())
    if not math.isclose(
        _finite(
            raw.get("max_reference_action_payoff_standard_error"),
            label="shard.max_reference_error",
            nonnegative=True,
        ),
        max_reference_error,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOL,
    ):
        raise ProductionHoldoutError("evaluation_shard: max reference uncertainty mismatch")
    policy_payoff = _weighted(policy, derived_payoffs)
    reference_payoff = _weighted(reference, derived_reference_payoffs)
    candidate_uniform_payoff = _weighted(reference, derived_payoffs)
    best = max(derived_payoffs.values())
    derived_scalars = {
        "policy_payoff_estimate": policy_payoff,
        "reference_payoff_estimate": reference_payoff,
        "candidate_continuation_uniform_root_payoff_estimate": (
            candidate_uniform_payoff
        ),
        "best_action_payoff_estimate": best,
        "ev_regret_estimate": max(0.0, best - policy_payoff),
    }
    for field, wanted in derived_scalars.items():
        if not math.isclose(_finite(raw.get(field), label=f"shard.{field}"), wanted, rel_tol=0.0, abs_tol=_FLOAT_TOL):
            raise ProductionHoldoutError(f"evaluation_shard.{field}: raw-derived mismatch")
    paired_map, paired_means, paired_errors = _moment_view(
        {"paired": raw.get("policy_reference_paired_aggregate")},
        label="shard.policy_reference_paired",
        samples_per_action=samples_per_action,
    )
    paired_mean = paired_means["paired"]
    paired_error = paired_errors["paired"]
    if not math.isclose(
        paired_mean,
        policy_payoff - reference_payoff,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOL,
    ):
        raise ProductionHoldoutError(
            "evaluation_shard: paired delta/raw payoff mismatch"
        )
    if not math.isclose(
        _finite(raw.get("policy_reference_delta_estimate"), label="shard.delta"),
        paired_mean,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOL,
    ):
        raise ProductionHoldoutError("evaluation_shard: paired delta estimate mismatch")
    if not math.isclose(
        _finite(
            raw.get("policy_reference_delta_standard_error"),
            label="shard.delta_se",
            nonnegative=True,
        ),
        paired_error,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOL,
    ):
        raise ProductionHoldoutError("evaluation_shard: paired delta uncertainty mismatch")
    audits = _require_mapping(raw.get("audits"), label="shard.audits")
    if set(audits) != set(REQUIRED_AUDITS) or any(audits[field] != 0 or isinstance(audits[field], bool) for field in REQUIRED_AUDITS):
        raise ProductionHoldoutError("evaluation_shard.audits: exact integer zeros required")
    runtime_ms = _finite(raw.get("runtime_ms"), label="shard.runtime_ms", nonnegative=True)
    if runtime_ms <= 0.0:
        raise ProductionHoldoutError("evaluation_shard.runtime_ms: positive required")
    sampling = _require_mapping(raw.get("sampling_audit"), label="shard.sampling_audit")
    expected_traversals = 2 * len(action_ids) * samples_per_action
    if sampling.get("traversals") != expected_traversals or sampling.get("root_posterior_samples") != expected_traversals:
        raise ProductionHoldoutError("evaluation_shard: root/chance sample accounting mismatch")
    replay_results = _evaluate_raw_aggregates(
        observation=observation,
        full_range=full_range,
        root=root,
        payoff_seed=payoff_seed,
        samples_per_action=samples_per_action,
        root_policy=policy,
        candidate_policy_by_key=candidate_policy_by_key,
    )
    replay_bindings = {
        "candidate_aggregates": aggregates,
        "candidate_estimates": derived_payoffs,
        "candidate_errors": derived_errors,
        "reference_aggregates": reference_aggregates,
        "reference_estimates": derived_reference_payoffs,
        "reference_errors": derived_reference_errors,
        "paired_aggregate": paired_map["paired"],
        "paired_mean": paired_mean,
        "paired_error": paired_error,
        "required_candidate_infoset_digests": required_infosets,
        "candidate_profile_infoset_count": len(candidate_policy_by_key),
    }
    for field, wanted in replay_bindings.items():
        if replay_results[field] != wanted:
            raise ProductionHoldoutError(
                "evaluation_shard: deterministic raw payoff replay mismatch"
            )
    return copy.deepcopy(dict(raw))


def _normalize_seed_plan(
    *,
    roots: Sequence[Mapping[str, Any]],
    evaluation_seeds: Sequence[int],
    payoff_seeds: Mapping[str, Mapping[int | str, int]],
    training_seeds: set[int],
    range_seeds: set[int],
) -> tuple[list[int], dict[str, dict[str, int]]]:
    normalized_evaluation = [
        _integer(seed, label="evaluation_seed") for seed in evaluation_seeds
    ]
    if len(normalized_evaluation) < 2 or normalized_evaluation != sorted(set(normalized_evaluation)):
        raise ProductionHoldoutError("evaluation_seeds: at least two unique canonical seeds required")
    pairs = sorted({str(root["seat_swap_pair_id"]) for root in roots})
    if set(payoff_seeds) != set(pairs):
        raise ProductionHoldoutError("payoff_seeds: exact seat-swap pair set required")
    normalized_payoff: dict[str, dict[str, int]] = {}
    used_payoffs: set[int] = set()
    for pair in pairs:
        raw = _require_mapping(payoff_seeds[pair], label=f"payoff_seeds.{pair}")
        converted = {str(key): value for key, value in raw.items()}
        expected = {str(seed) for seed in normalized_evaluation}
        if set(converted) != expected:
            raise ProductionHoldoutError(f"payoff_seeds.{pair}: exact evaluation seed set required")
        normalized_payoff[pair] = {}
        for seed in normalized_evaluation:
            payoff = _integer(converted[str(seed)], label=f"payoff_seeds.{pair}.{seed}")
            if payoff in used_payoffs:
                raise ProductionHoldoutError("payoff seed reused across pair/evaluation owners")
            used_payoffs.add(payoff)
            normalized_payoff[pair][str(seed)] = payoff
    evaluation_set = set(normalized_evaluation)
    seed_sets = {
        "training": training_seeds,
        "range_build": range_seeds,
        "evaluation": evaluation_set,
        "payoff": used_payoffs,
    }
    names = tuple(seed_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if seed_sets[left] & seed_sets[right]:
                raise ProductionHoldoutError(
                    f"{left} and {right} seed sets overlap"
                )
    return normalized_evaluation, normalized_payoff


def _candidate_training_seeds_and_independence(
    candidate: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    range_entries: Sequence[Mapping[str, Any]],
) -> set[int]:
    training_roots = {
        row["root_identity_commitment_sha256"]
        for stratum in REQUIRED_STRATA
        for row in candidate["strategies"][stratum]["training_roots"]
    }
    holdout_roots = {
        root["root_identity_commitment_sha256"] for root in roots
    }
    if training_roots & holdout_roots:
        raise ProductionHoldoutError(
            "candidate training root reused by independent holdout"
        )
    training_observations = {
        prior_root["observation_sha256"]
        for stratum in REQUIRED_STRATA
        for prior_root in candidate["strategies"][stratum]["root_prior_manifest"]["roots"]
    }
    holdout_observations = {root["observation_digest"] for root in roots}
    if training_observations & holdout_observations:
        raise ProductionHoldoutError(
            "candidate training observation relabeled as independent holdout"
        )
    training_range_content = {
        prior_root["range_content_sha256"]
        for stratum in REQUIRED_STRATA
        for prior_root in candidate["strategies"][stratum]["root_prior_manifest"]["roots"]
    }
    training_range_build = {
        prior_root["range_build_sha256"]
        for stratum in REQUIRED_STRATA
        for prior_root in candidate["strategies"][stratum]["root_prior_manifest"]["roots"]
    }
    holdout_range_content = {
        entry["range_content_sha256"] for entry in range_entries
    }
    holdout_range_build = {entry["range_build_sha256"] for entry in range_entries}
    if training_range_content & holdout_range_content:
        raise ProductionHoldoutError(
            "candidate training range content relabeled as holdout"
        )
    if training_range_build & holdout_range_build:
        raise ProductionHoldoutError(
            "candidate training range build relabeled as holdout"
        )
    return {
        int(candidate["strategies"][stratum]["producer_contract"]["seed"])
        for stratum in REQUIRED_STRATA
    }


_EVALUATION_BUNDLE_KEYS = {
    "schema", "artifact_kind", "promotion_eligible", "exact_exploitability_computed",
    "algorithm_validation_only", "production_promotion_supported",
    "manifest_written_after_all_shards", "root_range_bundle_sha256",
    "root_manifest_sha256", "source_manifest_sha256", "candidate_relative_path",
    "candidate_file_sha256", "candidate_bundle_sha256", "holdout_evaluator_manifest",
    "candidate_scope_manifest_sha256",
    "evaluation_seeds", "payoff_seeds", "samples_per_action", "shards",
    "evaluation_bundle_sha256",
}


def build_evaluation_bundle(
    *,
    root_range_bundle_dir: str | Path,
    candidate_bundle_path: str | Path,
    output_dir: str | Path,
    evaluation_seeds: Sequence[int],
    payoff_seeds: Mapping[str, Mapping[int | str, int]],
    samples_per_action: int,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume all root×seed shards and write the manifest last."""

    sample_target = _integer(
        samples_per_action, label="samples_per_action", minimum=2
    )
    range_bundle, materialized = verify_root_range_bundle(
        root_range_bundle_dir, workspace_root=workspace_root
    )
    roots = range_bundle["root_manifest"]["roots"]
    candidate_source = read_candidate_bundle(candidate_bundle_path)
    training_seeds = _candidate_training_seeds_and_independence(
        candidate_source, roots, range_bundle["entries"]
    )
    range_seeds = {int(entry["range_solver_seed"]) for entry in range_bundle["entries"]}
    normalized_evaluation, normalized_payoff = _normalize_seed_plan(
        roots=roots,
        evaluation_seeds=evaluation_seeds,
        payoff_seeds=payoff_seeds,
        training_seeds=training_seeds,
        range_seeds=range_seeds,
    )
    output = Path(output_dir)
    manifest_path = output / _EVALUATION_MANIFEST_NAME
    if manifest_path.exists():
        existing, _shards = verify_evaluation_bundle(
            output,
            root_range_bundle_dir=root_range_bundle_dir,
            workspace_root=workspace_root,
        )
        requested = {
            "candidate_bundle_sha256": candidate_source["candidate_bundle_sha256"],
            "candidate_scope_manifest_sha256": candidate_source[
                "candidate_scope_manifest_sha256"
            ],
            "root_range_bundle_sha256": range_bundle["bundle_sha256"],
            "evaluation_seeds": normalized_evaluation,
            "payoff_seeds": normalized_payoff,
            "samples_per_action": sample_target,
        }
        for field, wanted in requested.items():
            if existing[field] != wanted:
                raise ProductionHoldoutError(
                    f"existing evaluation bundle has different {field}"
                )
        return existing

    candidate_target = output / _CANDIDATE_RELATIVE_PATH
    if candidate_target.exists():
        if read_candidate_bundle(candidate_target) != candidate_source:
            raise ProductionHoldoutError("stale candidate input in resumable bundle")
    else:
        write_candidate_bundle(candidate_target, candidate_source)

    source_sha = range_bundle["source_manifest_sha256"]
    evaluator_manifest = build_holdout_evaluator_manifest(
        candidate_bundle_sha256=candidate_source["candidate_bundle_sha256"],
        root_manifest_sha256=range_bundle["root_manifest_sha256"],
        reference_policy_sha256=reference_policy_sha256(),
        source_manifest_sha256=source_sha,
    )
    entries_by_id = {
        entry["root"]["root_id"]: entry for entry in range_bundle["entries"]
    }
    expected_shard_paths = {
        _shard_relative(root["root_commitment_sha256"], seed)
        for root in roots
        for seed in normalized_evaluation
    }
    allowed_pre_manifest = {_CANDIDATE_RELATIVE_PATH, *expected_shard_paths}
    extras = _list_files(output) - allowed_pre_manifest
    if extras:
        raise ProductionHoldoutError(
            f"evaluation bundle has orphan files before manifest: {sorted(extras)}"
        )

    shard_records: list[dict[str, Any]] = []
    for root in roots:
        observation, full_range = materialized[root["root_id"]]
        range_entry = entries_by_id[root["root_id"]]
        for evaluation_seed in normalized_evaluation:
            payoff_seed = normalized_payoff[root["seat_swap_pair_id"]][str(evaluation_seed)]
            relative = _shard_relative(root["root_commitment_sha256"], evaluation_seed)
            path = output / relative
            if path.exists():
                shard = verify_shard(
                    _read_canonical(path),
                    root=root,
                    observation=observation,
                    full_range=full_range,
                    range_entry=range_entry,
                    candidate=candidate_source,
                    evaluator_manifest=evaluator_manifest,
                    source_manifest_sha256=source_sha,
                    evaluation_seed=evaluation_seed,
                    payoff_seed=payoff_seed,
                    samples_per_action=sample_target,
                )
            else:
                shard = _build_shard(
                    root=root,
                    observation=observation,
                    full_range=full_range,
                    range_entry=range_entry,
                    candidate=candidate_source,
                    evaluator_manifest=evaluator_manifest,
                    source_manifest_sha256=source_sha,
                    evaluation_seed=evaluation_seed,
                    payoff_seed=payoff_seed,
                    samples_per_action=sample_target,
                )
                _atomic_write(path, shard)
                shard = verify_shard(
                    _read_canonical(path),
                    root=root,
                    observation=observation,
                    full_range=full_range,
                    range_entry=range_entry,
                    candidate=candidate_source,
                    evaluator_manifest=evaluator_manifest,
                    source_manifest_sha256=source_sha,
                    evaluation_seed=evaluation_seed,
                    payoff_seed=payoff_seed,
                    samples_per_action=sample_target,
                )
            shard_records.append(
                {
                    "root_id": root["root_id"],
                    "root_commitment_sha256": root["root_commitment_sha256"],
                    "evaluation_seed": evaluation_seed,
                    "payoff_sample_seed": payoff_seed,
                    "relative_path": relative,
                    "shard_sha256": shard["shard_sha256"],
                    "file_sha256": _file_sha256(path),
                }
            )
    manifest: dict[str, Any] = {
        "schema": EVALUATION_BUNDLE_SCHEMA,
        "artifact_kind": "resumable_full_card_holdout_evaluation_bundle",
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "algorithm_validation_only": True,
        "production_promotion_supported": False,
        "manifest_written_after_all_shards": True,
        "root_range_bundle_sha256": range_bundle["bundle_sha256"],
        "root_manifest_sha256": range_bundle["root_manifest_sha256"],
        "source_manifest_sha256": source_sha,
        "candidate_relative_path": _CANDIDATE_RELATIVE_PATH,
        "candidate_file_sha256": _file_sha256(candidate_target),
        "candidate_bundle_sha256": candidate_source["candidate_bundle_sha256"],
        "candidate_scope_manifest_sha256": candidate_source[
            "candidate_scope_manifest_sha256"
        ],
        "holdout_evaluator_manifest": evaluator_manifest,
        "evaluation_seeds": normalized_evaluation,
        "payoff_seeds": normalized_payoff,
        "samples_per_action": sample_target,
        "shards": shard_records,
    }
    manifest["evaluation_bundle_sha256"] = self_hash(
        manifest, "evaluation_bundle_sha256"
    )
    # Deliberately the final filesystem mutation in a successful build.
    _atomic_write(manifest_path, manifest)
    return verify_evaluation_bundle(
        output,
        root_range_bundle_dir=root_range_bundle_dir,
        workspace_root=workspace_root,
    )[0]


def verify_evaluation_bundle(
    bundle_dir: str | Path,
    *,
    root_range_bundle_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = Path(bundle_dir)
    raw = _require_mapping(
        _read_canonical(output / _EVALUATION_MANIFEST_NAME),
        label="evaluation_bundle",
    )
    _exact_keys(raw, _EVALUATION_BUNDLE_KEYS, label="evaluation_bundle")
    fixed = {
        "schema": EVALUATION_BUNDLE_SCHEMA,
        "artifact_kind": "resumable_full_card_holdout_evaluation_bundle",
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "algorithm_validation_only": True,
        "production_promotion_supported": False,
        "manifest_written_after_all_shards": True,
        "candidate_relative_path": _CANDIDATE_RELATIVE_PATH,
    }
    for field, wanted in fixed.items():
        if raw.get(field) != wanted:
            raise ProductionHoldoutError(f"evaluation_bundle.{field}: mismatch")
    if raw.get("evaluation_bundle_sha256") != self_hash(raw, "evaluation_bundle_sha256"):
        raise ProductionHoldoutError("evaluation_bundle: self-hash mismatch")
    range_bundle, materialized = verify_root_range_bundle(
        root_range_bundle_dir, workspace_root=workspace_root
    )
    range_bindings = {
        "root_range_bundle_sha256": range_bundle["bundle_sha256"],
        "root_manifest_sha256": range_bundle["root_manifest_sha256"],
        "source_manifest_sha256": range_bundle["source_manifest_sha256"],
    }
    for field, wanted in range_bindings.items():
        if raw.get(field) != wanted:
            raise ProductionHoldoutError(f"evaluation_bundle.{field}: stale binding")
    candidate_path = _regular_file_below(output, _CANDIDATE_RELATIVE_PATH, label="candidate input")
    if raw.get("candidate_file_sha256") != _file_sha256(candidate_path):
        raise ProductionHoldoutError("evaluation_bundle: candidate file tampered")
    candidate = read_candidate_bundle(candidate_path)
    if raw.get("candidate_bundle_sha256") != candidate["candidate_bundle_sha256"]:
        raise ProductionHoldoutError("evaluation_bundle: stale candidate binding")
    if raw.get("candidate_scope_manifest_sha256") != candidate[
        "candidate_scope_manifest_sha256"
    ]:
        raise ProductionHoldoutError(
            "evaluation_bundle: stale candidate scope binding"
        )
    evaluator_manifest = raw.get("holdout_evaluator_manifest")
    expected_evaluator = build_holdout_evaluator_manifest(
        candidate_bundle_sha256=candidate["candidate_bundle_sha256"],
        root_manifest_sha256=range_bundle["root_manifest_sha256"],
        reference_policy_sha256=reference_policy_sha256(),
        source_manifest_sha256=range_bundle["source_manifest_sha256"],
    )
    if evaluator_manifest != expected_evaluator:
        raise ProductionHoldoutError("evaluation_bundle: evaluator manifest mismatch")
    training_seeds = _candidate_training_seeds_and_independence(
        candidate,
        range_bundle["root_manifest"]["roots"],
        range_bundle["entries"],
    )
    range_seeds = {int(entry["range_solver_seed"]) for entry in range_bundle["entries"]}
    evaluation_seeds, payoff_seeds = _normalize_seed_plan(
        roots=range_bundle["root_manifest"]["roots"],
        evaluation_seeds=raw.get("evaluation_seeds"),
        payoff_seeds=_require_mapping(raw.get("payoff_seeds"), label="evaluation_bundle.payoff_seeds"),
        training_seeds=training_seeds,
        range_seeds=range_seeds,
    )
    if raw.get("payoff_seeds") != payoff_seeds:
        raise ProductionHoldoutError("evaluation_bundle.payoff_seeds: noncanonical")
    samples_per_action = _integer(
        raw.get("samples_per_action"), label="evaluation_bundle.samples_per_action", minimum=2
    )
    roots = range_bundle["root_manifest"]["roots"]
    roots_by_id = {root["root_id"]: root for root in roots}
    entries_by_id = {entry["root"]["root_id"]: entry for entry in range_bundle["entries"]}
    expected_keys = {
        (root["root_id"], seed) for root in roots for seed in evaluation_seeds
    }
    records = raw.get("shards")
    if not isinstance(records, list) or len(records) != len(expected_keys):
        raise ProductionHoldoutError("evaluation_bundle.shards: incomplete coverage")
    record_keys: set[tuple[str, int]] = set()
    verified_shards: list[dict[str, Any]] = []
    expected_files = {_EVALUATION_MANIFEST_NAME, _CANDIDATE_RELATIVE_PATH}
    record_fields = {
        "root_id", "root_commitment_sha256", "evaluation_seed",
        "payoff_sample_seed", "relative_path", "shard_sha256", "file_sha256",
    }
    order: list[tuple[str, str, int]] = []
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, label=f"evaluation_bundle.shards[{index}]")
        _exact_keys(record, record_fields, label=f"evaluation_bundle.shards[{index}]")
        root = roots_by_id.get(str(record.get("root_id")))
        seed = _integer(record.get("evaluation_seed"), label=f"shards[{index}].evaluation_seed")
        if root is None or (root["root_id"], seed) not in expected_keys:
            raise ProductionHoldoutError("evaluation_bundle.shards: root/seed outside plan")
        key = (root["root_id"], seed)
        if key in record_keys:
            raise ProductionHoldoutError("evaluation_bundle.shards: duplicate root/seed")
        record_keys.add(key)
        payoff = payoff_seeds[root["seat_swap_pair_id"]][str(seed)]
        relative = _shard_relative(root["root_commitment_sha256"], seed)
        bindings = {
            "root_commitment_sha256": root["root_commitment_sha256"],
            "payoff_sample_seed": payoff,
            "relative_path": relative,
        }
        for field, wanted in bindings.items():
            if record.get(field) != wanted:
                raise ProductionHoldoutError(f"evaluation_bundle shard record {field} mismatch")
        expected_files.add(relative)
        path = _regular_file_below(output, relative, label=f"shard.{root['root_id']}.{seed}")
        if record.get("file_sha256") != _file_sha256(path):
            raise ProductionHoldoutError("evaluation bundle shard file tampered")
        observation, full_range = materialized[root["root_id"]]
        shard = verify_shard(
            _read_canonical(path),
            root=root,
            observation=observation,
            full_range=full_range,
            range_entry=entries_by_id[root["root_id"]],
            candidate=candidate,
            evaluator_manifest=expected_evaluator,
            source_manifest_sha256=range_bundle["source_manifest_sha256"],
            evaluation_seed=seed,
            payoff_seed=payoff,
            samples_per_action=samples_per_action,
        )
        if record.get("shard_sha256") != shard["shard_sha256"]:
            raise ProductionHoldoutError("evaluation bundle shard content binding mismatch")
        verified_shards.append(shard)
        order.append((root["stratum"], root["root_id"], seed))
    if record_keys != expected_keys:
        raise ProductionHoldoutError("evaluation_bundle.shards: gap detected")
    if order != sorted(order):
        raise ProductionHoldoutError("evaluation_bundle.shards: canonical order required")
    if _list_files(output) != expected_files:
        raise ProductionHoldoutError("evaluation_bundle: gap or orphan file detected")
    return copy.deepcopy(dict(raw)), verified_shards


def build_strength_rows_from_evaluation_bundle(
    bundle_dir: str | Path,
    *,
    root_range_bundle_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Adapt verified shards into the v2 shared-strength gate row schema."""

    manifest, shards = verify_evaluation_bundle(
        bundle_dir,
        root_range_bundle_dir=root_range_bundle_dir,
        workspace_root=workspace_root,
    )
    rows: list[dict[str, Any]] = []
    for shard in shards:
        root = shard["root"]
        digest = shard["policy_infoset_digest"]
        row: dict[str, Any] = {
            "run_id": f"full-card-{root['root_id']}-{shard['evaluation_seed']}",
            "root_id": root["root_id"],
            "root_commitment_sha256": root["root_commitment_sha256"],
            "root_identity_commitment_sha256": root["root_identity_commitment_sha256"],
            "stratum": root["stratum"],
            "actor": root["actor"],
            "visible_joker_count": root["visible_joker_count"],
            "seat_swap_pair_id": root["seat_swap_pair_id"],
            "evaluation_seed": shard["evaluation_seed"],
            "payoff_sample_seed": shard["payoff_sample_seed"],
            "candidate_bundle_sha256": shard["candidate_bundle_sha256"],
            "candidate_strategy_artifact_sha256": shard["candidate_strategy_artifact_sha256"],
            "candidate_strategy_sha256": shard["candidate_strategy_sha256"],
            "policy_infoset_digest": digest,
            "range_content_sha256": shard["range_content_sha256"],
            "range_build_sha256": shard["range_build_sha256"],
            "policy_action_distribution": copy.deepcopy(shard["policy_action_distribution"]),
            "reference_policy_sha256": shard["reference_policy_sha256"],
            "reference_action_distribution": copy.deepcopy(shard["reference_action_distribution"]),
            "action_payoff_estimates": copy.deepcopy(shard["action_payoff_estimates"]),
            "action_payoff_sample_counts": copy.deepcopy(shard["action_payoff_sample_counts"]),
            "action_payoff_aggregates": copy.deepcopy(shard["action_payoff_aggregates"]),
            "action_payoff_standard_errors": copy.deepcopy(shard["action_payoff_standard_errors"]),
            "max_action_payoff_standard_error": shard["max_action_payoff_standard_error"],
            "reference_action_payoff_estimates": copy.deepcopy(
                shard["reference_action_payoff_estimates"]
            ),
            "reference_action_payoff_sample_counts": copy.deepcopy(
                shard["reference_action_payoff_sample_counts"]
            ),
            "reference_action_payoff_aggregates": copy.deepcopy(
                shard["reference_action_payoff_aggregates"]
            ),
            "reference_action_payoff_standard_errors": copy.deepcopy(
                shard["reference_action_payoff_standard_errors"]
            ),
            "max_reference_action_payoff_standard_error": shard[
                "max_reference_action_payoff_standard_error"
            ],
            "policy_payoff_estimate": shard["policy_payoff_estimate"],
            "reference_payoff_estimate": shard["reference_payoff_estimate"],
            "candidate_continuation_uniform_root_payoff_estimate": shard[
                "candidate_continuation_uniform_root_payoff_estimate"
            ],
            "policy_reference_paired_aggregate": copy.deepcopy(
                shard["policy_reference_paired_aggregate"]
            ),
            "policy_reference_delta_estimate": shard[
                "policy_reference_delta_estimate"
            ],
            "policy_reference_delta_standard_error": shard[
                "policy_reference_delta_standard_error"
            ],
            "best_action_payoff_estimate": shard["best_action_payoff_estimate"],
            "ev_regret_estimate": shard["ev_regret_estimate"],
            "eligible_infoset_digests": copy.deepcopy(
                shard["required_candidate_infoset_digests"]
            ),
            "encountered_infoset_digests": copy.deepcopy(
                shard["required_candidate_infoset_digests"]
            ),
            "eligible_infoset_count": len(
                shard["required_candidate_infoset_digests"]
            ),
            "encountered_infoset_count": len(
                shard["required_candidate_infoset_digests"]
            ),
            "encountered_infoset_coverage": 1.0,
            "audits": copy.deepcopy(shard["audits"]),
            "runtime_ms": shard["runtime_ms"],
            "exact_exploitability_computed": False,
        }
        row["row_sha256"] = self_hash(row, "row_sha256")
        rows.append(row)
    if len(rows) != len(manifest["shards"]):  # pragma: no cover
        raise AssertionError("verified shard/row count drift")
    return rows


__all__ = [
    "EVALUATION_BUNDLE_SCHEMA",
    "EVALUATOR_METHOD",
    "ProductionHoldoutError",
    "REFERENCE_POLICY_CONTRACT",
    "ROOT_RANGE_BUNDLE_SCHEMA",
    "SHARD_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "build_evaluation_bundle",
    "build_root_range_bundle",
    "build_source_manifest",
    "build_strength_rows_from_evaluation_bundle",
    "read_candidate_bundle",
    "reference_policy_sha256",
    "verify_evaluation_bundle",
    "verify_root_range_bundle",
    "verify_shard",
    "verify_source_manifest",
    "write_candidate_bundle",
]
