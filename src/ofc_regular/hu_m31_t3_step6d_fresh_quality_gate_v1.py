"""Source-replayed merge and frozen gate for M3.1 T3 fresh quality.

The worker result contract deliberately reuses the accepted Step 6c decision
certificate validator.  That validator recomputes legal ActionKeys, original
index/order mappings, Q/regret arithmetic, search/result certificates, belief
digests, and individual particle RNG keys from ActorObservation only.

Invalid, missing, duplicated, unknown-field, hidden-information, ActionKey, or
RNG evidence fails before a merge can exist.  Only confirmation regret may
produce a well-formed ``no_go`` gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import ActionKey
from .hu_infoset import ActorObservation
from . import hu_m31_t3_step6d_fresh_quality_v1 as quality
from . import run_hu_m31_t3_step6c_shard as step6c
from . import run_hu_m31_t3_step6d_performance as step6d_v1


RESULT_SCHEMA = "hu_m31_t3_step6d_fresh_quality_job_result_v1"
MERGE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_merge_v1"
GATE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gate_v1"
PERCENTILE_METHOD = "nearest_rank_ceil_n_times_q_v1"

REGRET_THRESHOLDS = {
    "mean_max": 0.75,
    "p95_max": 3.0,
    "p99_max": 6.0,
    "max_max": 15.0,
}

_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "job",
        "plan_sha256",
        "root_seal_sha256",
        "candidate_library_sha256",
        "root_artifacts",
        "rows",
        "peak_rss_bytes",
        "teacher_value_status",
        "teacher_values_are_realized_match_ev",
        "opponent_private_discards_used",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_RESULT_ROOT_KEYS = frozenset({"phase", "pair_index", "path", "sha256", "root_indices"})
_ROW_KEYS = frozenset(
    {
        "phase",
        "pair_index",
        "root_index",
        "seat",
        "observation_fingerprint",
        "observation_sha256",
        "primary_wall_seconds",
        "primary_decision",
        "confirmation_wall_seconds",
        "confirmation",
        "peak_rss_bytes",
    }
)
_SOURCE_PATH_KEYS = frozenset(
    {
        "plan",
        "materialization",
        "root_seal",
        "results_directory",
        "performance_receipt",
    }
)
_MERGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "source_paths",
        "source_sha256",
        "plan_sha256",
        "materialization_sha256",
        "root_seal_sha256",
        "result_records",
        "result_record_aggregate_sha256",
        "integrity",
        "confirmation_quality",
        "performance",
        "teacher_value_status",
        "teacher_values_are_realized_match_ev",
        "training_eligible",
        "full_data_fanout_authorized",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_GATE_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "merge",
        "merge_sha256",
        "thresholds",
        "gates",
        "all_gates_passed",
        "quality_pilot_passed",
        "data_pilot_25_paired_authorized",
        "full_9000_paired_fanout_authorized",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
        "teacher_values_are_realized_match_ev",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return quality.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return quality.canonical_sha256(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(
            f"{label} fields changed: missing={missing}, unknown={unknown}"
        )


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Any) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside the accepted range")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quality percentile input is empty")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _regret_metrics_any(values: Sequence[float]) -> dict[str, Any]:
    rows = [_finite(value, "confirmation regret", minimum=0.0) for value in values]
    if not rows:
        raise ValueError("fresh-quality confirmation regret input is empty")
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "p95": _nearest_rank(rows, 0.95),
        "p99": _nearest_rank(rows, 0.99),
        "max": max(rows),
        "percentile_method": PERCENTILE_METHOD,
        "selected_regrets": rows,
    }


def _regret_metrics(values: Sequence[float]) -> dict[str, Any]:
    metrics = _regret_metrics_any(values)
    if metrics["count"] != 10:
        raise ValueError("fresh-quality confirmation requires exactly 10 regrets")
    return metrics


def _latency(values: Sequence[float]) -> dict[str, Any]:
    rows = sorted(_finite(value, "latency", minimum=0.0) for value in values)
    if not rows:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
            "max_seconds": 0.0,
        }
    return {
        "count": len(rows),
        "mean_seconds": statistics.fmean(rows),
        "p50_seconds": _nearest_rank(rows, 0.50),
        "p95_seconds": _nearest_rank(rows, 0.95),
        "p99_seconds": _nearest_rank(rows, 0.99),
        "max_seconds": max(rows),
    }


def _root_map(
    *, plan: Mapping[str, Any], materialization: Mapping[str, Any]
) -> tuple[
    dict[int, tuple[dict[str, Any], ActorObservation]], dict[str, dict[str, Any]]
]:
    root_dir = Path(str(materialization["root_directory"])).resolve()
    by_root: dict[int, tuple[dict[str, Any], ActorObservation]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for record in materialization["root_records"]:
        relative = str(record["path"])
        root = quality.validate_root(
            _read_canonical(root_dir / relative, f"fresh-quality root {relative}"),
            plan=plan,
        )
        by_path[relative] = root
        for item in root["observations"]:
            observation = ActorObservation.from_dict(item["observation"])
            by_root[int(item["root_index"])] = (root, observation)
    if set(by_root) != set(range(110)) or len(by_path) != 55:
        raise ValueError("fresh-quality root lookup grid changed")
    return by_root, by_path


def _validate_confirmation_payload(
    *,
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> float:
    step6c._validate_confirmation_pair(primary, confirmation, payload)
    regret = _finite(payload.get("selected_regret"), "selected regret", minimum=0.0)
    confirmation_values = {
        str(row["action_key"]): float(row["evaluation_ev"])
        for row in confirmation["action_values"]
    }
    primary_key = str(primary["selected_action_key"])
    best = max(confirmation_values.values())
    expected = best - confirmation_values[primary_key]
    if not math.isclose(regret, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fresh-quality confirmation regret arithmetic changed")
    return regret


def _validate_result_row(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    root_lookup: Mapping[int, tuple[dict[str, Any], ActorObservation]],
) -> dict[str, Any]:
    row = deepcopy(dict(value))
    _exact_keys(row, _ROW_KEYS, "fresh-quality result row")
    root_index = _integer(row.get("root_index"), "root index")
    if root_index not in root_lookup:
        raise ValueError("fresh-quality result row root index is outside the plan")
    root, observation = root_lookup[root_index]
    phase = str(root["phase"])
    pair_index = int(root["pair_index"])
    item = next(
        record
        for record in root["observations"]
        if int(record["root_index"]) == root_index
    )
    if (
        phase != job["phase"]
        or pair_index not in job["pair_indices"]
        or row["phase"] != phase
        or row["pair_index"] != pair_index
        or row["seat"] != observation.seat
        or row["observation_fingerprint"] != observation.fingerprint()
        or row["observation_fingerprint"] != item["observation_fingerprint"]
        or row["observation_sha256"] != item["observation_sha256"]
    ):
        raise ValueError("fresh-quality result row provenance changed")
    primary_seconds = _finite(
        row["primary_wall_seconds"], "primary wall seconds", minimum=0.0
    )
    peak = _integer(row["peak_rss_bytes"], "row peak RSS", minimum=1)
    primary = row.get("primary_decision")
    primary_evidence = step6c._validate_decision_payload(
        primary,
        observation=observation,
        seeds=root["seeds"],
        budget=step6c._PRIMARY_BUDGET,
        evaluation_seed_key="evaluation",
        native_library_sha256=quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
    )
    confirmation_payload = row.get("confirmation")
    confirmation_seconds = row.get("confirmation_wall_seconds")
    confirmation_evidence = None
    regret = None
    if phase == quality.CONFIRMATION_PHASE:
        if not isinstance(confirmation_payload, Mapping):
            raise ValueError("fresh-quality confirmation row lacks locked evaluation")
        confirmation_seconds = _finite(
            confirmation_seconds, "confirmation wall seconds", minimum=0.0
        )
        confirmation_decision = confirmation_payload.get("decision")
        confirmation_evidence = step6c._validate_decision_payload(
            confirmation_decision,
            observation=observation,
            seeds=root["seeds"],
            budget=step6c._CONFIRMATION_BUDGET,
            evaluation_seed_key="confirmation",
            native_library_sha256=quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
        )
        if (
            confirmation_evidence.candidate_keys != primary_evidence.candidate_keys
            or confirmation_evidence.selection_by_key
            != primary_evidence.selection_by_key
        ):
            raise ValueError("fresh-quality locked confirmation changed candidate pass")
        regret = _validate_confirmation_payload(
            primary=primary,
            confirmation=confirmation_decision,
            payload=confirmation_payload,
        )
    elif confirmation_payload is not None or confirmation_seconds is not None:
        raise ValueError("primary quality row unexpectedly contains confirmation")
    candidate_keys = primary_evidence.candidate_keys
    evaluation_keys = primary_evidence.evaluation_keys
    confirmation_keys = (
        confirmation_evidence.evaluation_keys
        if confirmation_evidence is not None
        else frozenset()
    )
    if (
        candidate_keys & evaluation_keys
        or candidate_keys & confirmation_keys
        or evaluation_keys & confirmation_keys
    ):
        raise ValueError("fresh-quality row RNG namespaces overlap")
    step6d_v1._reject_hidden(row, "fresh_quality_result_row")
    return {
        "row": row,
        "phase": phase,
        "pair_index": pair_index,
        "root_index": root_index,
        "seat": observation.seat,
        "profile": root["profile"],
        "fingerprint": observation.fingerprint(),
        "candidate_keys": candidate_keys,
        "evaluation_keys": evaluation_keys,
        "confirmation_keys": confirmation_keys,
        "regret": regret,
        "primary_seconds": primary_seconds,
        "confirmation_seconds": confirmation_seconds,
        "peak_rss_bytes": peak,
    }


def validate_job_result(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    seal: Mapping[str, Any],
    root_lookup: Mapping[int, tuple[dict[str, Any], ActorObservation]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = deepcopy(dict(value))
    _exact_keys(result, _RESULT_KEYS, "fresh-quality job result")
    stored_job = result.get("job")
    if not isinstance(stored_job, Mapping):
        raise ValueError("fresh-quality result job descriptor is missing")
    validated_job = quality.validate_job_descriptor(stored_job, plan=plan, seal=seal)
    roots = result.get("root_artifacts")
    rows = result.get("rows")
    if not isinstance(roots, list) or not isinstance(rows, list):
        raise ValueError("fresh-quality result root/row evidence is missing")
    expected_root_records = []
    seal_by_path = {record["path"]: record for record in seal["root_records"]}
    expected_root_indices: list[int] = []
    for relative in validated_job["root_paths"]:
        record = seal_by_path[relative]
        expected_root_records.append(
            {
                "phase": record["phase"],
                "pair_index": record["pair_index"],
                "path": record["path"],
                "sha256": record["sha256"],
                "root_indices": record["root_indices"],
            }
        )
        expected_root_indices.extend(record["root_indices"])
    for record in roots:
        if not isinstance(record, Mapping):
            raise ValueError("fresh-quality result root record is missing")
        _exact_keys(record, _RESULT_ROOT_KEYS, "fresh-quality result root record")
    evidence = [
        _validate_result_row(row, job=validated_job, root_lookup=root_lookup)
        for row in rows
    ]
    if (
        result["schema"] != RESULT_SCHEMA
        or result["status"] != "complete_create_only_quality_job"
        or result["job"] != validated_job
        or result["plan_sha256"] != canonical_sha256(plan)
        or result["root_seal_sha256"] != canonical_sha256(seal)
        or result["candidate_library_sha256"]
        != quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or roots != expected_root_records
        or [item["root_index"] for item in evidence] != expected_root_indices
        or result["peak_rss_bytes"] != max(item["peak_rss_bytes"] for item in evidence)
        or result["teacher_value_status"] != "diagnostic_not_match_EV"
        or result["teacher_values_are_realized_match_ev"] is not False
        or result["opponent_private_discards_used"] is not False
        or any(
            result[field] is not False
            for field in (
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("fresh-quality job result contract changed")
    step6d_v1._reject_hidden(result, "fresh_quality_job_result")
    return result, evidence


def _validated_sources(source_paths: Mapping[str, Any]) -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paths = dict(source_paths)
    _exact_keys(paths, _SOURCE_PATH_KEYS, "fresh-quality merge source paths")
    normalized: dict[str, str] = {}
    for key, raw in paths.items():
        path = Path(str(raw))
        if not path.is_absolute():
            raise ValueError(f"fresh-quality source path {key} is not absolute")
        normalized[key] = str(path.resolve())
    plan = quality.validate_plan_authorization(
        _read_canonical(normalized["plan"], "fresh-quality plan"),
        performance_receipt_path=normalized["performance_receipt"],
    )
    materialization = quality.validate_materialization_receipt(
        _read_canonical(normalized["materialization"], "fresh-quality materialization"),
        plan=plan,
        replay_roots=True,
    )
    seal = quality.validate_root_seal(
        _read_canonical(normalized["root_seal"], "fresh-quality root seal"),
        plan=plan,
        materialization=materialization,
    )
    root_lookup, _root_by_path = _root_map(plan=plan, materialization=materialization)
    results_dir = Path(normalized["results_directory"])
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise ValueError("fresh-quality result directory is missing or unsafe")
    jobs = quality.build_job_descriptors(
        plan=plan,
        seal=seal,
        root_directory=materialization["root_directory"],
    )
    expected_names = {Path(job["result_path"]).name for job in jobs}
    actual_names = {
        path.name
        for path in results_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != expected_names or any(
        path.is_symlink() or not path.is_file() for path in results_dir.iterdir()
    ):
        raise ValueError("fresh-quality result grid has missing or extra artifacts")
    records: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for job in jobs:
        name = Path(job["result_path"]).name
        path = results_dir / name
        result, rows = validate_job_result(
            _read_canonical(path, f"fresh-quality result {name}"),
            job=job,
            plan=plan,
            seal=seal,
            root_lookup=root_lookup,
        )
        records.append(
            {
                "job_id": job["job_id"],
                "phase": job["phase"],
                "path": name,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
                "row_count": len(rows),
                "result_sha256": canonical_sha256(result),
            }
        )
        evidence.extend(rows)
    return normalized, plan, materialization, seal, records, evidence


def build_fresh_quality_merge(
    *,
    plan_path: str | Path,
    materialization_path: str | Path,
    root_seal_path: str | Path,
    results_directory: str | Path,
    performance_receipt_path: str | Path,
) -> dict[str, Any]:
    source_paths = {
        "plan": str(Path(plan_path).resolve()),
        "materialization": str(Path(materialization_path).resolve()),
        "root_seal": str(Path(root_seal_path).resolve()),
        "results_directory": str(Path(results_directory).resolve()),
        "performance_receipt": str(Path(performance_receipt_path).resolve()),
    }
    paths, plan, materialization, seal, records, evidence = _validated_sources(
        source_paths
    )
    root_indices = [row["root_index"] for row in evidence]
    fingerprints = [row["fingerprint"] for row in evidence]
    primary = [row for row in evidence if row["phase"] == quality.PRIMARY_PHASE]
    confirmation = [
        row for row in evidence if row["phase"] == quality.CONFIRMATION_PHASE
    ]
    candidate_sets = [row["candidate_keys"] for row in evidence]
    evaluation_sets = [row["evaluation_keys"] for row in evidence]
    confirmation_sets = [row["confirmation_keys"] for row in evidence]
    candidate_keys = set().union(*candidate_sets)
    evaluation_keys = set().union(*evaluation_sets)
    confirmation_keys = set().union(*confirmation_sets)
    if (
        root_indices != list(range(110))
        or len(set(fingerprints)) != 110
        or len(candidate_keys) != 880
        or len(evaluation_keys) != 3520
        or len(confirmation_keys) != 1280
        or sum(len(item) for item in candidate_sets) != 880
        or sum(len(item) for item in evaluation_sets) != 3520
        or sum(len(item) for item in confirmation_sets) != 1280
        or candidate_keys & evaluation_keys
        or candidate_keys & confirmation_keys
        or evaluation_keys & confirmation_keys
    ):
        raise ValueError("fresh-quality merged root/RNG grid changed")
    regrets = [row["regret"] for row in confirmation]
    if any(value is None for value in regrets):
        raise ValueError("fresh-quality confirmation regret is missing")
    seat_counts = {
        phase: {
            seat: sum(row["phase"] == phase and row["seat"] == seat for row in evidence)
            for seat in ("first", "second")
        }
        for phase in quality.PHASES
    }
    profile_pair_counts = {
        phase: {
            profile: len(
                {
                    row["pair_index"]
                    for row in evidence
                    if row["phase"] == phase and row["profile"] == profile
                }
            )
            for profile in quality.M31_T3_BEHAVIOR_PROFILES
        }
        for phase in quality.PHASES
    }
    if (
        seat_counts
        != {
            "primary": {"first": 50, "second": 50},
            "confirmation": {"first": 5, "second": 5},
        }
        or any(value != 10 for value in profile_pair_counts["primary"].values())
        or any(value != 1 for value in profile_pair_counts["confirmation"].values())
    ):
        raise ValueError("fresh-quality seat/profile quota changed")
    source_sha256 = {
        "plan": _file_sha256(Path(paths["plan"])),
        "materialization": _file_sha256(Path(paths["materialization"])),
        "root_seal": _file_sha256(Path(paths["root_seal"])),
        "performance_receipt": _file_sha256(Path(paths["performance_receipt"])),
    }
    merge = {
        "schema": MERGE_SCHEMA,
        "status": "complete_source_replayed_quality_merge",
        "source_paths": paths,
        "source_sha256": source_sha256,
        "plan_sha256": canonical_sha256(plan),
        "materialization_sha256": canonical_sha256(materialization),
        "root_seal_sha256": canonical_sha256(seal),
        "result_records": records,
        "result_record_aggregate_sha256": canonical_sha256(records),
        "integrity": {
            "paired_hands": 55,
            "roots": 110,
            "primary_paired_hands": 50,
            "primary_roots": 100,
            "confirmation_paired_hands": 5,
            "confirmation_roots": 10,
            "seat_counts": seat_counts,
            "profile_pair_counts": profile_pair_counts,
            "unique_observation_fingerprints": 110,
            "candidate_rng_keys": 880,
            "evaluation_rng_keys": 3520,
            "confirmation_rng_keys": 1280,
            "candidate_evaluation_overlap": 0,
            "candidate_confirmation_overlap": 0,
            "evaluation_confirmation_overlap": 0,
            "hidden_information_field_count": 0,
            "unknown_field_count": 0,
            "action_key_drift_count": 0,
            "missing_or_extra_result_count": 0,
        },
        "confirmation_quality": {
            "root_indices": [row["root_index"] for row in confirmation],
            **_regret_metrics([float(value) for value in regrets]),
            "by_seat_diagnostic": {
                seat: _regret_metrics_any(
                    [
                        float(row["regret"])
                        for row in confirmation
                        if row["seat"] == seat
                    ]
                )
                for seat in ("first", "second")
            },
            "top1_agreement_is_diagnostic_only": True,
        },
        "performance": {
            "primary_latency_by_seat": {
                seat: _latency(
                    [row["primary_seconds"] for row in evidence if row["seat"] == seat]
                )
                for seat in ("first", "second")
            },
            "confirmation_latency_by_seat": {
                seat: _latency(
                    [
                        float(row["confirmation_seconds"])
                        for row in confirmation
                        if row["seat"] == seat
                    ]
                )
                for seat in ("first", "second")
            },
            "peak_process_rss_bytes": max(row["peak_rss_bytes"] for row in evidence),
            "latency_is_diagnostic_not_quality_gate": True,
        },
        "teacher_value_status": "diagnostic_not_match_EV",
        "teacher_values_are_realized_match_ev": False,
        "training_eligible": False,
        "full_data_fanout_authorized": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    return merge


def validate_fresh_quality_merge_value(
    value: Mapping[str, Any], *, replay_sources: bool
) -> dict[str, Any]:
    merge = deepcopy(dict(value))
    _exact_keys(merge, _MERGE_KEYS, "fresh-quality merge")
    if replay_sources is not True:
        raise PermissionError("fresh-quality merge validation requires source replay")
    expected = build_fresh_quality_merge(
        plan_path=merge["source_paths"]["plan"],
        materialization_path=merge["source_paths"]["materialization"],
        root_seal_path=merge["source_paths"]["root_seal"],
        results_directory=merge["source_paths"]["results_directory"],
        performance_receipt_path=merge["source_paths"]["performance_receipt"],
    )
    if merge != expected:
        raise ValueError("fresh-quality merge differs from source replay")
    return merge


def write_fresh_quality_merge(
    *, output_path: str | Path, **source_paths: Any
) -> dict[str, Any]:
    merge = build_fresh_quality_merge(**source_paths)
    _write_once(output_path, merge)
    stored = _read_canonical(output_path, "stored fresh-quality merge")
    if validate_fresh_quality_merge_value(stored, replay_sources=True) != merge:
        raise ValueError("stored fresh-quality merge differs from source replay")
    return merge


def build_fresh_quality_gate(
    *, merge: Mapping[str, Any], replay_sources: bool
) -> dict[str, Any]:
    validated = validate_fresh_quality_merge_value(merge, replay_sources=replay_sources)
    metrics = validated["confirmation_quality"]
    integrity = validated["integrity"]
    gates = {
        "exact_50_fresh_paired_100_primary_roots": (
            integrity["primary_paired_hands"] == 50
            and integrity["primary_roots"] == 100
        ),
        "separate_5_paired_10_confirmation_roots": (
            integrity["confirmation_paired_hands"] == 5
            and integrity["confirmation_roots"] == 10
        ),
        "hidden_unknown_actionkey_rng_missing_zero": all(
            integrity[field] == 0
            for field in (
                "candidate_evaluation_overlap",
                "candidate_confirmation_overlap",
                "evaluation_confirmation_overlap",
                "hidden_information_field_count",
                "unknown_field_count",
                "action_key_drift_count",
                "missing_or_extra_result_count",
            )
        ),
        "confirmation_regret_mean_at_most_0_75": (
            metrics["mean"] <= REGRET_THRESHOLDS["mean_max"]
        ),
        "confirmation_regret_p95_at_most_3": (
            metrics["p95"] <= REGRET_THRESHOLDS["p95_max"]
        ),
        "confirmation_regret_p99_at_most_6": (
            metrics["p99"] <= REGRET_THRESHOLDS["p99_max"]
        ),
        "confirmation_regret_max_at_most_15": (
            metrics["max"] <= REGRET_THRESHOLDS["max_max"]
        ),
        "teacher_values_not_realized_match_ev": (
            validated["teacher_values_are_realized_match_ev"] is False
        ),
        "no_training_promotion_or_current_change": (
            validated["training_eligible"] is False
            and validated["promotion_evidence"] is False
            and validated["current_profile_changed"] is False
        ),
    }
    passed = all(gates.values())
    return {
        "schema": GATE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": (
            "fresh_quality_pass_open_25_paired_data_shard_only"
            if passed
            else "fresh_quality_no_go_no_same_seed_threshold_reselection"
        ),
        "merge": deepcopy(validated),
        "merge_sha256": canonical_sha256(validated),
        "thresholds": deepcopy(REGRET_THRESHOLDS),
        "gates": gates,
        "all_gates_passed": passed,
        "quality_pilot_passed": passed,
        "data_pilot_25_paired_authorized": passed,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "teacher_values_are_realized_match_ev": False,
    }


def validate_fresh_quality_gate_value(
    value: Mapping[str, Any], *, replay_sources: bool
) -> dict[str, Any]:
    gate = deepcopy(dict(value))
    _exact_keys(gate, _GATE_KEYS, "fresh-quality gate")
    merge = gate.get("merge")
    if not isinstance(merge, Mapping):
        raise ValueError("fresh-quality gate merge is missing")
    expected = build_fresh_quality_gate(merge=merge, replay_sources=replay_sources)
    if gate != expected:
        raise ValueError("fresh-quality gate differs from source replay")
    return gate


def write_fresh_quality_gate(
    *,
    merge_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    merge = _read_canonical(merge_path, "fresh-quality merge")
    gate = build_fresh_quality_gate(merge=merge, replay_sources=True)
    _write_once(output_path, gate)
    stored = _read_canonical(output_path, "stored fresh-quality gate")
    if validate_fresh_quality_gate_value(stored, replay_sources=True) != gate:
        raise ValueError("stored fresh-quality gate differs from source replay")
    return gate


__all__ = [
    "GATE_SCHEMA",
    "MERGE_SCHEMA",
    "PERCENTILE_METHOD",
    "REGRET_THRESHOLDS",
    "RESULT_SCHEMA",
    "build_fresh_quality_gate",
    "build_fresh_quality_merge",
    "canonical_bytes",
    "canonical_sha256",
    "validate_fresh_quality_gate_value",
    "validate_fresh_quality_merge_value",
    "validate_job_result",
    "write_fresh_quality_gate",
    "write_fresh_quality_merge",
]
