"""Merge and independently validate source-isolated Step 6d v2 artifacts.

The v2 runner puts the reference and candidate native images in different
processes and publishes immutable per-source shards.  This module pairs those
shards only after every canonical byte, manifest hash, run-contract field,
work index, root, source role, binary, allocation, budget, and decision has
been checked again.

Only two scopes are valid: the precommitted ten-hand candidate01 tail
diagnostic and the complete frozen 100-hand performance-development set.  A
tail result cannot apply the full gate, and an arbitrary partial set cannot be
merged at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import run_hu_m31_t3_step6d_performance as v1
from . import run_hu_m31_t3_step6d_performance_v2 as v2
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from .validate_hu_m31_t3_step6d_performance import (
    MAX_FIRST_P95_SECONDS,
    MAX_FIRST_P99_SECONDS,
    MAX_FIRST_SECONDS,
    MAX_PEAK_RSS_BYTES,
    MAX_SECOND_P95_SECONDS,
)


MERGE_SCHEMA = "hu_m31_t3_step6d_performance_merge_v2"
VALIDATION_SCHEMA = "hu_m31_t3_step6d_performance_merge_validation_v2"

TAIL_DIAGNOSTIC_SCOPE = "candidate01_tail_diagnostic"
FULL_PERFORMANCE_SCOPE = "full_performance_development"
TAIL_DIAGNOSTIC_HAND_INDICES = tuple(v2.TAIL_HAND_INDICES)
TAIL_RANDOM_HAND_INDICES = (9, 29)
TAIL_HEAVY_HAND_INDICES = tuple(
    index
    for index in TAIL_DIAGNOSTIC_HAND_INDICES
    if index not in TAIL_RANDOM_HAND_INDICES
)
FULL_HAND_INDICES = tuple(v2.CONTRACT_HAND_INDICES)

TAIL_PAIRED_PARITY_REQUIRED = 10
TAIL_MAX_PEAK_RSS_BYTES = 858_993_459
TAIL_HEAVY_FIRST_MEDIAN_SECONDS_MAX = 135.0
TAIL_HEAVY_FIRST_MAX_SECONDS_MAX = 145.0
TAIL_GEOMETRIC_MEAN_SPEEDUP_MIN = 1.55
TAIL_SECOND_MAX_SECONDS_MAX = 5.0
PERCENTILE_METHOD = "nearest_rank_ceiling_n_times_p"

_DONE_INPUT_KEYS = frozenset(
    {"path", "sha256", "shard_manifest_digest", "work_hand_indices"}
)
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "run_contract",
        "run_contract_digest",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "budget",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "source_done_inputs",
        "paired_artifacts",
        "integrity",
        "performance",
        "gate_mode",
        "gates",
        "all_gates_passed",
        "candidate01_tail_qualified",
        "performance_candidate_frozen",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)


@dataclass(frozen=True)
class _HandArtifact:
    role: str
    hand_index: int
    hand_path: Path
    hand_sha256: str
    root_path: Path
    root_sha256: str
    done_path: Path
    done_sha256: str
    shard_manifest_digest: str
    root: dict[str, Any]
    hand: dict[str, Any]


@dataclass(frozen=True)
class _RoleArtifacts:
    role: str
    run_contract: dict[str, Any]
    run_contract_digest: str
    hands: tuple[_HandArtifact, ...]
    done_inputs: tuple[dict[str, Any], ...]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite and in range")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.casefold()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    original = Path(path)
    if original.is_symlink():
        raise ValueError(f"{label} is a symlink")
    resolved = original.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != v2.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_root_canonical(path: Path, label: str) -> dict[str, Any]:
    """Read the retained v1 root format (canonical JSON without a newline)."""

    original = Path(path)
    if original.is_symlink():
        raise ValueError(f"{label} is a symlink")
    resolved = original.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != v1._canonical_bytes(value):
        raise ValueError(f"{label} is not canonical v1 root JSON")
    return value


def _safe_artifact_path(base: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    fragment = Path(relative)
    if fragment.is_absolute() or ".." in fragment.parts:
        raise ValueError(f"{label} path escapes its shard")
    base = base.resolve()
    unresolved = base / fragment
    if unresolved.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its shard") from error
    if not resolved.is_file():
        raise ValueError(f"{label} path is missing")
    return resolved


def _scope_for_indices(indices: Sequence[int], requested_scope: str = "auto") -> str:
    observed = tuple(indices)
    if observed == TAIL_DIAGNOSTIC_HAND_INDICES:
        inferred = TAIL_DIAGNOSTIC_SCOPE
    elif observed == FULL_HAND_INDICES:
        inferred = FULL_PERFORMANCE_SCOPE
    else:
        raise ValueError(
            "Step 6d v2 merge requires exactly the frozen tail or hands 0..99"
        )
    if requested_scope not in {"auto", inferred}:
        raise ValueError(
            f"Step 6d {requested_scope} gate cannot be applied to {list(indices)}"
        )
    return inferred


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-rank percentile requires values and p in (0,1]")
    ordered = sorted(_finite(value, "percentile value") for value in values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def _latencies(values: Sequence[float]) -> dict[str, Any]:
    rows = [_finite(value, "latency", minimum=0.0) for value in values]
    if not rows:
        raise ValueError("latency summary requires values")
    return {
        "count": len(rows),
        "mean_seconds": statistics.fmean(rows),
        "median_seconds": statistics.median(rows),
        "p95_seconds": nearest_rank_percentile(rows, 0.95),
        "p99_seconds": nearest_rank_percentile(rows, 0.99),
        "max_seconds": max(rows),
        "percentile_method": PERCENTILE_METHOD,
    }


def geometric_mean(values: Sequence[float]) -> float:
    rows = [_finite(value, "geometric-mean value", minimum=0.0) for value in values]
    if not rows or any(value <= 0.0 for value in rows):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in rows))


def _validate_shared_contract(
    contract: Mapping[str, Any],
    *,
    contract_variant: str = v2.CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    value = v2.validate_run_contract(contract)
    if (
        v2.contract_variant(value) != contract_variant
        or value["reference_library_sha256"] != v1.REFERENCE_NATIVE_LIBRARY_SHA256
        or value["allocation"] != v2.ALLOCATION
        or value["budget"] != v1.PERFORMANCE_BUDGET
        or value["contract_hand_indices"] != list(FULL_HAND_INDICES)
    ):
        raise ValueError("Step 6d v2 binary/allocation/budget contract changed")
    return value


def _expected_json_files(
    base: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    work_hand_indices: Sequence[int],
) -> set[Path]:
    expected = {
        base / "DONE.json",
        base / "run_contract.json",
        base / "shard_manifest.json",
        *(base / "roots" / f"hand_{index:03d}.json" for index in work_hand_indices),
    }
    for record in records:
        expected.add(
            _safe_artifact_path(base, record.get("path"), "Step 6d DONE artifact")
        )
    return {path.resolve() for path in expected}


def _load_role(
    done_paths: Sequence[Path],
    role: str,
    *,
    contract_variant: str = v2.CANDIDATE01_VARIANT,
) -> _RoleArtifacts:
    if role not in v2.SOURCE_ROLES or not done_paths:
        raise ValueError("candidate and reference DONE paths are both required")
    resolved_done: set[Path] = set()
    hands: dict[int, _HandArtifact] = {}
    done_inputs: list[dict[str, Any]] = []
    shared_contract: dict[str, Any] | None = None
    shared_digest: str | None = None

    for supplied in done_paths:
        supplied = Path(supplied)
        if supplied.is_symlink():
            raise ValueError("Step 6d DONE path is a symlink")
        done_path = supplied.resolve()
        if done_path in resolved_done:
            raise ValueError(f"duplicate {role} DONE input")
        resolved_done.add(done_path)
        if done_path.name != "DONE.json":
            raise ValueError("Step 6d source completion marker must be DONE.json")
        base = done_path.parent
        done = _read_canonical(done_path, f"Step 6d {role} DONE")
        manifest_path = base / "shard_manifest.json"
        manifest = v2.validate_shard_manifest(
            _read_canonical(manifest_path, f"Step 6d {role} shard manifest")
        )
        if manifest["source_role"] != role or done.get("source_role") != role:
            raise ValueError("Step 6d source role mix detected")
        validated_done = v2.validate_done(
            done, output_dir=base, shard_manifest=manifest
        )
        contract = _validate_shared_contract(
            manifest["run_contract"], contract_variant=contract_variant
        )
        digest = v2.canonical_sha256(contract)
        if validated_done["run_contract_digest"] != digest:
            raise ValueError("Step 6d DONE run-contract digest mismatch")
        run_contract_path = base / "run_contract.json"
        if (
            _read_canonical(run_contract_path, f"Step 6d {role} run contract")
            != contract
        ):
            raise ValueError("Step 6d run-contract artifact mismatch")
        if shared_contract is None:
            shared_contract = contract
            shared_digest = digest
        elif shared_contract != contract or shared_digest != digest:
            raise ValueError(f"mixed {role} run-contract digests")

        records = _array(
            validated_done["artifact_manifest"], "Step 6d DONE artifact manifest"
        )
        work = tuple(validated_done["work_hand_indices"])
        if tuple(validated_done["completed_hand_indices"]) != work:
            raise ValueError("Step 6d DONE completed-hand coverage mismatch")
        if (
            validated_done["artifact_count"] != len(records)
            or len(records) != 2 * len(work)
            or validated_done["artifact_manifest_sha256"]
            != v2.canonical_sha256(records)
        ):
            raise ValueError("Step 6d DONE artifact-manifest aggregate mismatch")
        expected_paths = _expected_json_files(base, records, work_hand_indices=work)
        discovered_json = list(base.rglob("*.json"))
        if any(path.is_symlink() for path in discovered_json):
            raise ValueError("Step 6d shard contains an unsafe JSON symlink")
        actual_paths = {path.resolve() for path in discovered_json if path.is_file()}
        if actual_paths != expected_paths:
            raise ValueError(
                "Step 6d shard has missing or out-of-contract JSON artifacts"
            )
        record_by_path: dict[str, Mapping[str, Any]] = {}
        for raw_record in records:
            record = _mapping(raw_record, "Step 6d DONE artifact record")
            relative = record.get("path")
            if not isinstance(relative, str) or relative in record_by_path:
                raise ValueError("Step 6d DONE has duplicate artifact paths")
            artifact_path = _safe_artifact_path(base, relative, "Step 6d DONE artifact")
            if _sha256(artifact_path) != _require_sha256(
                record.get("sha256"), "artifact SHA-256"
            ) or artifact_path.stat().st_size != _integer(
                record.get("bytes"), "artifact bytes", minimum=1
            ):
                raise ValueError("Step 6d artifact byte/hash mismatch")
            if (
                record.get("source_role") != role
                or record.get("hand_index") not in work
            ):
                raise ValueError("Step 6d artifact role/hand manifest mismatch")
            if relative.startswith("roots/"):
                _read_root_canonical(artifact_path, "Step 6d bound root artifact")
            else:
                _read_canonical(artifact_path, "Step 6d bound source artifact")
            record_by_path[relative] = record

        for index in work:
            if index in hands:
                raise ValueError(f"duplicate {role} hand {index}")
            root_relative = f"roots/hand_{index:03d}.json"
            hand_relative = f"hands/{role}/hand_{index:03d}.json"
            if (
                root_relative not in record_by_path
                or hand_relative not in record_by_path
            ):
                raise ValueError(f"missing {role} hand/root artifact {index}")
            root_path = base / root_relative
            hand_path = base / hand_relative
            root = _read_root_canonical(root_path, f"Step 6d root {index}")
            hand = _read_canonical(hand_path, f"Step 6d {role} hand {index}")
            validated_hand = v2._validate_source_hand(
                hand,
                root=root,
                run_contract=contract,
                source_role=role,
                library_sha256=contract[f"{role}_library_sha256"],
                run_contract_digest=digest,
                shard_manifest_sha256=v2.canonical_sha256(manifest),
                reference_library_sha256=contract["reference_library_sha256"],
                candidate_library_sha256=contract["candidate_library_sha256"],
            )
            hands[index] = _HandArtifact(
                role=role,
                hand_index=index,
                hand_path=hand_path.resolve(),
                hand_sha256=_sha256(hand_path),
                root_path=root_path.resolve(),
                root_sha256=_sha256(root_path),
                done_path=done_path,
                done_sha256=_sha256(done_path),
                shard_manifest_digest=v2.canonical_sha256(manifest),
                root=root,
                hand=validated_hand,
            )
        done_inputs.append(
            {
                "path": str(done_path),
                "sha256": _sha256(done_path),
                "shard_manifest_digest": v2.canonical_sha256(manifest),
                "work_hand_indices": list(work),
            }
        )

    assert shared_contract is not None and shared_digest is not None
    ordered_hands = tuple(hands[index] for index in sorted(hands))
    done_inputs.sort(key=lambda item: (item["work_hand_indices"], item["path"]))
    return _RoleArtifacts(
        role=role,
        run_contract=shared_contract,
        run_contract_digest=shared_digest,
        hands=ordered_hands,
        done_inputs=tuple(done_inputs),
    )


def _memory_peak(hand: Mapping[str, Any]) -> int:
    memory = _mapping(hand.get("memory"), "Step 6d source-hand memory")
    peak = _integer(memory.get("peak_rss_bytes"), "Step 6d peak RSS")
    result_peaks = []
    for raw in _array(hand.get("rows"), "Step 6d rows"):
        result = _mapping(
            _mapping(raw, "Step 6d row").get("source_result"), "source result"
        )
        snapshot = _mapping(result.get("rss_after"), "source result RSS")
        result_peaks.append(
            _integer(snapshot.get("peak_rss_bytes"), "source result peak RSS")
        )
    if peak < max(result_peaks):
        raise ValueError("Step 6d source-hand RSS aggregate understates a row")
    return peak


def _portable_decision_pair(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], label: str
) -> dict[str, Any]:
    reference_actions = _array(reference.get("action_values"), f"{label} reference Q")
    candidate_actions = _array(candidate.get("action_values"), f"{label} candidate Q")
    if reference_actions != candidate_actions:
        raise ValueError(f"{label} ordered ActionKey/all-Q parity mismatch")
    action_keys = [
        _mapping(row, f"{label} action row").get("action_key")
        for row in reference_actions
    ]
    if len(action_keys) != len(set(action_keys)):
        raise ValueError(f"{label} contains duplicate ActionKeys")
    selected_fields = (
        "selected_action_key",
        "selected_action",
        "selected_selection_ev",
        "selected_evaluation_ev",
        "selection_gap",
        "evaluation_sample_regret",
    )
    if any(reference.get(key) != candidate.get(key) for key in selected_fields):
        raise ValueError(f"{label} selected action/Q parity mismatch")
    rng_fields = (
        "run_id",
        "continuation_seed",
        "candidate_seed",
        "evaluation_seed",
        "candidate_belief_digest",
        "evaluation_belief_digest",
        "candidate_rng_digest",
        "evaluation_rng_digest",
        "search_contract_digest",
    )
    if any(reference.get(key) != candidate.get(key) for key in rng_fields):
        raise ValueError(f"{label} RNG parity mismatch")
    if reference.get("child_information_set_count") != candidate.get(
        "child_information_set_count"
    ):
        raise ValueError(f"{label} child-count parity mismatch")

    reference_semantic = deepcopy(dict(reference))
    candidate_semantic = deepcopy(dict(candidate))
    for value in (reference_semantic, candidate_semantic):
        for key in (
            "native_library_sha256",
            "native_latency_ms",
            "validation_latency_ms",
            "total_latency_ms",
            "semantic_result_digest",
            "result_digest",
        ):
            value.pop(key, None)
    if reference_semantic != candidate_semantic:
        raise ValueError(f"{label} portable semantic decision mismatch")
    parity = v1.compare_portable_decisions(reference, candidate)
    required = (
        "action_keys_exact",
        "selection_q_exact",
        "evaluation_q_exact",
        "selected_action_exact",
        "rng_exact",
        "child_information_set_count_exact",
        "portable_payload_exact",
    )
    if any(parity.get(key) is not True for key in required):
        raise ValueError(f"{label} portable parity mismatch")
    return parity


def _pair_hand(
    reference: _HandArtifact,
    candidate: _HandArtifact,
    *,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = reference.hand_index
    if candidate.hand_index != index:
        raise ValueError("Step 6d candidate/reference hand pairing changed")
    if (
        reference.root != candidate.root
        or reference.root_sha256 != candidate.root_sha256
    ):
        raise ValueError(f"Step 6d hand {index} root fingerprint mismatch")
    common_fields = (
        "contract_canonical_sha256",
        "schedule",
        "run_contract_digest",
        "hand_index",
        "root_indices",
        "schedule_row_sha256",
        "profile",
        "seeds",
        "budget",
        "root_artifact_sha256",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "engine_version",
        "teacher_value_status",
    )
    if any(reference.hand.get(key) != candidate.hand.get(key) for key in common_fields):
        raise ValueError(f"Step 6d hand {index} source provenance mismatch")
    reference_rows = _array(reference.hand["rows"], "reference rows")
    candidate_rows = _array(candidate.hand["rows"], "candidate rows")
    paired_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for offset, (reference_raw, candidate_raw) in enumerate(
        zip(reference_rows, candidate_rows, strict=True)
    ):
        reference_row = _mapping(reference_raw, "reference row")
        candidate_row = _mapping(candidate_raw, "candidate row")
        for key in (
            "root_index",
            "seat",
            "observation_fingerprint",
            "observation_sha256",
        ):
            if reference_row.get(key) != candidate_row.get(key):
                raise ValueError(
                    f"Step 6d hand {index} observation/root/geometry mismatch"
                )
        seat = "first" if offset == 0 else "second"
        if candidate_row.get("seat") != seat:
            raise ValueError(f"Step 6d hand {index} seat ordering changed")
        fingerprint = _require_sha256(
            candidate_row.get("observation_fingerprint"),
            f"Step 6d hand {index} observation fingerprint",
        )
        observation_sha256 = _require_sha256(
            candidate_row.get("observation_sha256"),
            f"Step 6d hand {index} observation SHA-256",
        )
        fingerprints.append(fingerprint)
        reference_result = _mapping(
            reference_row.get("source_result"), "reference result"
        )
        candidate_result = _mapping(
            candidate_row.get("source_result"), "candidate result"
        )
        reference_portable = _mapping(
            reference_row.get("portable_decision"), "reference portable decision"
        )
        candidate_portable = _mapping(
            candidate_row.get("portable_decision"), "candidate portable decision"
        )
        if reference_row.get("portable_decision_sha256") != v2.canonical_sha256(
            reference_portable
        ) or candidate_row.get("portable_decision_sha256") != v2.canonical_sha256(
            candidate_portable
        ):
            raise ValueError(
                f"Step 6d hand {index} {seat} ordered portable-Q digest mismatch"
            )
        reference_decision = _mapping(
            reference_result.get("decision"), "reference decision"
        )
        candidate_decision = _mapping(
            candidate_result.get("decision"), "candidate decision"
        )
        if reference_decision.get("engine_version") != reference.hand.get(
            "engine_version"
        ) or candidate_decision.get("engine_version") != candidate.hand.get(
            "engine_version"
        ):
            raise ValueError(f"Step 6d hand {index} {seat} engine provenance mismatch")
        parity = _portable_decision_pair(
            reference_decision,
            candidate_decision,
            f"Step 6d hand {index} {seat}",
        )
        if reference_portable != candidate_portable:
            raise ValueError(
                f"Step 6d hand {index} {seat} ordered portable-Q parity mismatch"
            )
        if reference_row.get("geometry") != candidate_row.get("geometry"):
            raise ValueError(f"Step 6d hand {index} {seat} source geometry mismatch")
        reference_seconds = _finite(
            reference_result.get("solve_wall_seconds"),
            "reference solve seconds",
            minimum=0.0,
        )
        candidate_seconds = _finite(
            candidate_result.get("solve_wall_seconds"),
            "candidate solve seconds",
            minimum=0.0,
        )
        if reference_seconds <= 0.0 or candidate_seconds <= 0.0:
            raise ValueError("Step 6d paired speedup requires positive solve times")
        geometry = _mapping(candidate_row.get("geometry"), "Step 6d geometry")
        action_count = _integer(
            geometry.get("hero_legal_action_count"), "Step 6d legal-action count"
        )
        if (
            action_count != len(candidate_decision["action_values"])
            or action_count != len(candidate_portable["action_values"])
            or geometry.get("candidate_q_count") != action_count
            or geometry.get("evaluation_q_count") != action_count
            or geometry.get("child_information_set_count")
            != candidate_decision.get("child_information_set_count")
        ):
            raise ValueError("Step 6d geometry/all-Q count mismatch")
        paired_rows.append(
            {
                "root_index": candidate_row["root_index"],
                "seat": seat,
                "observation_fingerprint": fingerprint,
                "observation_sha256": observation_sha256,
                "geometry": dict(geometry),
                "portable_decision_sha256": candidate_row["portable_decision_sha256"],
                "reference_solve_wall_seconds": reference_seconds,
                "candidate_solve_wall_seconds": candidate_seconds,
                "paired_speedup": reference_seconds / candidate_seconds,
                "portable_parity": parity,
            }
        )
    if len(set(fingerprints)) != 2:
        raise ValueError(f"Step 6d hand {index} has duplicate observations")
    heavy_21x21 = (
        paired_rows[0]["geometry"]["first_seat_action_matrix_rows"] == 21
        and paired_rows[0]["geometry"]["first_seat_action_matrix_columns"] == 21
        and paired_rows[0]["geometry"]["first_seat_action_matrix_cells"] == 441
    )
    if (
        scope == TAIL_DIAGNOSTIC_SCOPE
        and index in TAIL_HEAVY_HAND_INDICES
        and not heavy_21x21
    ):
        raise ValueError(f"Step 6d heavy tail hand {index} is not 21x21")
    paired = {
        "hand_index": index,
        "profile": candidate.hand["profile"],
        "root_artifact_sha256": candidate.hand["root_artifact_sha256"],
        "root_file_sha256": candidate.root_sha256,
        "reference_hand_sha256": reference.hand_sha256,
        "candidate_hand_sha256": candidate.hand_sha256,
        "reference_done_sha256": reference.done_sha256,
        "candidate_done_sha256": candidate.done_sha256,
        "paired_seat_parity_count": 2,
        "heavy_21x21": heavy_21x21,
        "rows": paired_rows,
    }
    return paired, paired_rows


def _performance(
    paired_rows: Sequence[Mapping[str, Any]],
    paired_artifacts: Sequence[Mapping[str, Any]],
    reference: _RoleArtifacts,
    candidate: _RoleArtifacts,
) -> dict[str, Any]:
    by_role = {
        role: {
            seat: _latencies(
                [
                    float(row[f"{role}_solve_wall_seconds"])
                    for row in paired_rows
                    if row["seat"] == seat
                ]
            )
            for seat in ("first", "second")
        }
        for role in ("reference", "candidate")
    }
    all_hands = [*reference.hands, *candidate.hands]
    peak_rss = max(_memory_peak(artifact.hand) for artifact in all_hands)
    first_speedups = [
        float(row["paired_speedup"]) for row in paired_rows if row["seat"] == "first"
    ]
    heavy_indices = set(TAIL_HEAVY_HAND_INDICES)
    heavy_rows = [
        row
        for artifact in paired_artifacts
        if artifact["hand_index"] in heavy_indices
        for row in artifact["rows"]
        if row["seat"] == "first"
    ]
    heavy = None
    if heavy_rows:
        heavy = {
            "hand_indices": [
                artifact["hand_index"]
                for artifact in paired_artifacts
                if artifact["hand_index"] in heavy_indices
            ],
            "candidate_first": _latencies(
                [float(row["candidate_solve_wall_seconds"]) for row in heavy_rows]
            ),
            "reference_first": _latencies(
                [float(row["reference_solve_wall_seconds"]) for row in heavy_rows]
            ),
            "paired_speedups": [float(row["paired_speedup"]) for row in heavy_rows],
            "geometric_mean_speedup": geometric_mean(
                [float(row["paired_speedup"]) for row in heavy_rows]
            ),
        }
    return {
        "reference_by_seat": by_role["reference"],
        "candidate_by_seat": by_role["candidate"],
        "first_paired_speedups": first_speedups,
        "first_geometric_mean_speedup_diagnostic": geometric_mean(first_speedups),
        "tail_heavy_first": heavy,
        "peak_source_process_rss_bytes": peak_rss,
    }


def _build_gates(
    *,
    scope: str,
    hand_indices: Sequence[int],
    paired_artifacts: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
) -> tuple[str, dict[str, bool], bool]:
    peak = int(performance["peak_source_process_rss_bytes"])
    candidate = _mapping(performance["candidate_by_seat"], "candidate performance")
    if scope == TAIL_DIAGNOSTIC_SCOPE:
        heavy = _mapping(performance["tail_heavy_first"], "tail-heavy performance")
        gates = {
            "exact_precommitted_tail_indices": tuple(hand_indices)
            == TAIL_DIAGNOSTIC_HAND_INDICES,
            "paired_semantic_parity_10_of_10": len(paired_artifacts)
            == TAIL_PAIRED_PARITY_REQUIRED
            and all(item["paired_seat_parity_count"] == 2 for item in paired_artifacts),
            "heavy_21x21_first_roots_8_of_8": sum(
                item["hand_index"] in TAIL_HEAVY_HAND_INDICES
                and item["heavy_21x21"] is True
                for item in paired_artifacts
            )
            == len(TAIL_HEAVY_HAND_INDICES),
            "peak_rss_within_858993459_bytes": peak <= TAIL_MAX_PEAK_RSS_BYTES,
            "heavy_candidate_first_median_within_135_seconds": float(
                heavy["candidate_first"]["median_seconds"]
            )
            <= TAIL_HEAVY_FIRST_MEDIAN_SECONDS_MAX,
            "heavy_candidate_first_max_within_145_seconds": float(
                heavy["candidate_first"]["max_seconds"]
            )
            <= TAIL_HEAVY_FIRST_MAX_SECONDS_MAX,
            "heavy_geometric_mean_speedup_at_least_1_55": float(
                heavy["geometric_mean_speedup"]
            )
            >= TAIL_GEOMETRIC_MEAN_SPEEDUP_MIN,
            "candidate_second_max_within_5_seconds": float(
                candidate["second"]["max_seconds"]
            )
            <= TAIL_SECOND_MAX_SECONDS_MAX,
        }
        return "tail_candidate01_qualification_only", gates, all(gates.values())

    profile_counts = {
        profile: sum(item["profile"] == profile for item in paired_artifacts)
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    first = _mapping(candidate["first"], "candidate first performance")
    second = _mapping(candidate["second"], "candidate second performance")
    gates = {
        "exactly_100_paired_hands": tuple(hand_indices) == FULL_HAND_INDICES,
        "exactly_200_roots": len(paired_rows) == 200,
        "exactly_100_first_and_100_second": sum(
            row["seat"] == "first" for row in paired_rows
        )
        == sum(row["seat"] == "second" for row in paired_rows)
        == 100,
        "exactly_20_hands_per_profile": set(profile_counts.values()) == {20},
        "portable_semantic_parity_fraction_one": all(
            row["portable_parity"]["portable_payload_exact"] is True
            for row in paired_rows
        ),
        "missing_or_censored_roots_zero": len(paired_rows) == 200,
        "first_p95_within_150_seconds": float(first["p95_seconds"])
        <= MAX_FIRST_P95_SECONDS,
        "first_p99_within_240_seconds": float(first["p99_seconds"])
        <= MAX_FIRST_P99_SECONDS,
        "first_max_within_240_seconds": float(first["max_seconds"])
        <= MAX_FIRST_SECONDS,
        "second_p95_within_5_seconds": float(second["p95_seconds"])
        <= MAX_SECOND_P95_SECONDS,
        "peak_rss_within_858993459_bytes": peak <= MAX_PEAK_RSS_BYTES,
    }
    return "frozen_full_100_hand_gate", gates, all(gates.values())


def merge_performance_v2(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    scope: str = "auto",
    contract_variant: str = v2.CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    """Build a deterministic merge summary from immutable source DONE paths."""

    candidate = _load_role(
        candidate_done_paths,
        "candidate",
        contract_variant=contract_variant,
    )
    reference = _load_role(
        reference_done_paths,
        "reference",
        contract_variant=contract_variant,
    )
    if (
        candidate.run_contract != reference.run_contract
        or candidate.run_contract_digest != reference.run_contract_digest
    ):
        raise ValueError("candidate/reference shared run-contract digest mismatch")
    candidate_indices = tuple(item.hand_index for item in candidate.hands)
    reference_indices = tuple(item.hand_index for item in reference.hands)
    if candidate_indices != reference_indices:
        missing_candidate = sorted(set(reference_indices) - set(candidate_indices))
        missing_reference = sorted(set(candidate_indices) - set(reference_indices))
        raise ValueError(
            "candidate/reference work coverage mismatch: "
            f"missing_candidate={missing_candidate}, missing_reference={missing_reference}"
        )
    inferred_scope = _scope_for_indices(candidate_indices, scope)
    paired_artifacts: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for reference_hand, candidate_hand in zip(
        reference.hands, candidate.hands, strict=True
    ):
        paired, rows = _pair_hand(reference_hand, candidate_hand, scope=inferred_scope)
        for row in rows:
            fingerprint = str(row["observation_fingerprint"])
            if fingerprint in fingerprints:
                raise ValueError("duplicate Step 6d observation fingerprint")
            fingerprints.add(fingerprint)
        paired_artifacts.append(paired)
        paired_rows.extend(rows)
    performance = _performance(paired_rows, paired_artifacts, reference, candidate)
    gate_mode, gates, all_gates = _build_gates(
        scope=inferred_scope,
        hand_indices=candidate_indices,
        paired_artifacts=paired_artifacts,
        paired_rows=paired_rows,
        performance=performance,
    )
    status = "pass" if all_gates else "no_go"
    if inferred_scope == TAIL_DIAGNOSTIC_SCOPE:
        decision = (
            "candidate01_tail_qualification_pass_full_gate_not_applied"
            if all_gates
            else "candidate01_tail_no_go_full_gate_not_applied"
        )
    else:
        decision = (
            "performance_development_candidate_freeze_pass_open_performance_lock_only"
            if all_gates
            else "performance_development_no_go_continue_engineering_on_same_set"
        )
    integrity = {
        "candidate_hand_count": len(candidate.hands),
        "reference_hand_count": len(reference.hands),
        "paired_hand_count": len(paired_artifacts),
        "paired_root_count": len(paired_rows),
        "unique_observation_fingerprint_count": len(fingerprints),
        "paired_hand_parity_count": sum(
            item["paired_seat_parity_count"] == 2 for item in paired_artifacts
        ),
        "paired_root_parity_count": sum(
            row["portable_parity"]["portable_payload_exact"] is True
            for row in paired_rows
        ),
        "missing_hand_indices": [],
        "duplicate_hand_indices": [],
        "out_of_contract_hand_indices": [],
    }
    full_pass = inferred_scope == FULL_PERFORMANCE_SCOPE and all_gates
    tail_pass = inferred_scope == TAIL_DIAGNOSTIC_SCOPE and all_gates
    return {
        "schema": MERGE_SCHEMA,
        "status": status,
        "decision": decision,
        "scope": inferred_scope,
        "run_contract": candidate.run_contract,
        "run_contract_digest": candidate.run_contract_digest,
        "hand_indices": list(candidate_indices),
        "paired_hand_count": len(paired_artifacts),
        "root_count": len(paired_rows),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "allocation": dict(v2.ALLOCATION),
        "reference_library_sha256": candidate.run_contract["reference_library_sha256"],
        "candidate_library_sha256": candidate.run_contract["candidate_library_sha256"],
        "source_done_inputs": {
            "candidate": list(candidate.done_inputs),
            "reference": list(reference.done_inputs),
        },
        "paired_artifacts": paired_artifacts,
        "integrity": integrity,
        "performance": performance,
        "gate_mode": gate_mode,
        "gates": gates,
        "all_gates_passed": all_gates,
        "candidate01_tail_qualified": tail_pass,
        "performance_candidate_frozen": full_pass,
        "performance_lock_authorized": full_pass,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def _input_paths(summary: Mapping[str, Any], role: str) -> list[Path]:
    source_inputs = _mapping(summary.get("source_done_inputs"), "source DONE inputs")
    rows = _array(source_inputs.get(role), f"{role} DONE inputs")
    paths: list[Path] = []
    for raw in rows:
        row = _mapping(raw, f"{role} DONE input")
        if set(row) != _DONE_INPUT_KEYS:
            raise ValueError(f"{role} DONE input fields changed")
        path = Path(str(row.get("path"))).resolve()
        if _sha256(path) != _require_sha256(row.get("sha256"), "DONE SHA-256"):
            raise ValueError(f"{role} DONE input hash mismatch")
        paths.append(path)
    return paths


def _validation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    all_gates = summary.get("all_gates_passed") is True
    return {
        "schema": VALIDATION_SCHEMA,
        "status": summary["status"],
        "decision": summary["decision"],
        "scope": summary["scope"],
        "summary_sha256": v2.canonical_sha256(summary),
        "run_contract_digest": summary["run_contract_digest"],
        "hand_indices": list(summary["hand_indices"]),
        "paired_hand_count": summary["paired_hand_count"],
        "root_count": summary["root_count"],
        "integrity_recomputed_from_source_artifacts": True,
        "gates": dict(summary["gates"]),
        "all_gates_passed": all_gates,
        "candidate01_tail_qualified": summary["candidate01_tail_qualified"],
        "performance_candidate_frozen": summary["performance_candidate_frozen"],
        "performance_lock_authorized": summary["performance_lock_authorized"],
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def validate_performance_merge_v2(
    *, summary_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    """Reload every source artifact and reject a forged merge aggregate."""

    summary_path = Path(summary_path).resolve()
    summary = _read_canonical(summary_path, "Step 6d v2 merge summary")
    if set(summary) != _SUMMARY_KEYS or summary.get("schema") != MERGE_SCHEMA:
        raise ValueError("Step 6d v2 merge summary schema changed")
    scope = str(summary.get("scope"))
    expected = merge_performance_v2(
        candidate_done_paths=_input_paths(summary, "candidate"),
        reference_done_paths=_input_paths(summary, "reference"),
        scope=scope,
    )
    if summary != expected:
        raise ValueError("Step 6d v2 merge summary aggregate/tamper mismatch")
    report = _validation_report(summary)
    if output_path is not None:
        _write_once(Path(output_path).resolve(), report)
    return report


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite Step 6d v2 merge output: {target}"
        )
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.perf_counter_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(v2.canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite Step 6d v2 merge output: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def merge_and_validate_performance_v2(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    summary_output_path: Path,
    validation_output_path: Path,
    scope: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    if summary_output == validation_output:
        raise ValueError("summary and validation outputs must be distinct")
    if summary_output.exists() or validation_output.exists():
        raise FileExistsError("Step 6d v2 merge outputs are write-once")
    summary = merge_performance_v2(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        scope=scope,
    )
    validation = _validation_report(summary)
    _write_once(summary_output, summary)
    _write_once(validation_output, validation)
    return summary, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-done", type=Path, action="append", required=True)
    parser.add_argument("--reference-done", type=Path, action="append", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("auto", TAIL_DIAGNOSTIC_SCOPE, FULL_PERFORMANCE_SCOPE),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, validation = merge_and_validate_performance_v2(
        candidate_done_paths=args.candidate_done,
        reference_done_paths=args.reference_done,
        summary_output_path=args.summary_output,
        validation_output_path=args.validation_output,
        scope=args.scope,
    )
    print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FULL_HAND_INDICES",
    "FULL_PERFORMANCE_SCOPE",
    "MERGE_SCHEMA",
    "TAIL_DIAGNOSTIC_HAND_INDICES",
    "TAIL_DIAGNOSTIC_SCOPE",
    "TAIL_GEOMETRIC_MEAN_SPEEDUP_MIN",
    "TAIL_HEAVY_FIRST_MAX_SECONDS_MAX",
    "TAIL_HEAVY_FIRST_MEDIAN_SECONDS_MAX",
    "TAIL_HEAVY_HAND_INDICES",
    "TAIL_MAX_PEAK_RSS_BYTES",
    "TAIL_SECOND_MAX_SECONDS_MAX",
    "VALIDATION_SCHEMA",
    "geometric_mean",
    "main",
    "merge_and_validate_performance_v2",
    "merge_performance_v2",
    "nearest_rank_percentile",
    "validate_performance_merge_v2",
]
