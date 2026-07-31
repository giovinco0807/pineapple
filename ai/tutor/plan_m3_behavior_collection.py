"""Preregister the deterministic production-sized M3 behavior collection.

This module plans, but deliberately does not execute, the large trace
collection.  It binds the production temperature-gate configuration to the
collector's exact root identities and immutable root-hash split.  The natural
range is the *shortest contiguous prefix* whose four role streams each meet
the fit/dev/locked-test minimums.  The targeted range is the shortest prefix
of the preregistered twelve-cell Joker cycle meeting every cell minimum.

The resulting artifact contains commitments, not the root preimages or any
decision labels.  It is never evidence of calibration quality, strategic
strength, or promotion eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import ai.tutor.behavior_calibration_contract as calibration_contract_module
import ai.tutor.collect_hu_behavior_traces as collector_module
from ai.tutor.behavior_calibration_contract import (
    SPLIT_NAMES,
    SPLIT_NAMESPACE,
    canonical_json,
    canonical_sha256,
    canonical_snapshot,
    root_split,
)
from ai.tutor.behavior_temperature_calibration import (
    GATE_CONFIG_SCHEMA,
    JOKER_KEYS,
    ROLE_KEYS,
    build_temperature_gate_config,
    verify_temperature_gate_config,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    ROOT_ID_SCHEMA,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
    derive_root_id,
    targeted_joker_cell,
)


PLAN_SCHEMA = "ofc_behavior_collection_plan/v1"
SIZE_OBSERVATION_SCHEMA = "ofc_behavior_collection_size_observation/v1"
DEFAULT_PLAN_ID = "m3_behavior_collection_plan_20260713"
DEFAULT_CREATED_ON = "2026-07-13"
DEFAULT_NATURAL_SEED_NAMESPACE = (
    "m3-behavior-calibration-production-natural-20260713-v1"
)
DEFAULT_CHALLENGE_SEED_NAMESPACE = (
    "m3-behavior-calibration-production-joker-grid-20260713-v1"
)
DEFAULT_SMOKE_REPORT = "ai/reports/m3_behavior_calibration_smoke_20260713"
DEFAULT_OUTPUT = "ai/reports/m3_behavior_collection_plan_20260713/plan.json"
DEFAULT_SHARD_SIZE = 1_000
SMOKE_REPORT_SCHEMA = "ofc_behavior_calibration_smoke_report/v2"

_SHA256_HEX = frozenset("0123456789abcdef")
_PAYLOAD_FILES = {
    "natural": {
        "decisions": ("natural/decisions.jsonl", "decision_file_sha256"),
        "hidden_roots": ("natural/roots.jsonl", "hidden_roots_file_sha256"),
        "model_evaluations": (
            "natural/model_evaluations.jsonl",
            "evaluation_file_sha256",
        ),
    },
    "joker_challenge": {
        "decisions": ("challenge/decisions.jsonl", "decision_file_sha256"),
        "hidden_roots": ("challenge/roots.jsonl", "hidden_roots_file_sha256"),
        "model_evaluations": (
            "challenge/model_evaluations.jsonl",
            "evaluation_file_sha256",
        ),
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _normalized_relative_path(value: str | Path, *, label: str) -> str:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must stay beneath the workspace root")
    normalized = raw.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{label} must name a relative path")
    return normalized


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise ValueError(f"{path} must be one canonical JSON line")
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw[:-1]:
        raise ValueError(f"{path} is not canonical JSON")
    return value


def build_smoke_size_observation(
    workspace_root: str | Path,
    smoke_report_relative_path: str | Path = DEFAULT_SMOKE_REPORT,
) -> dict[str, Any]:
    """Measure the saved 12-root smoke payloads used only for capacity estimates."""
    root = Path(workspace_root)
    report_relative = _normalized_relative_path(
        smoke_report_relative_path, label="smoke_report_relative_path"
    )
    report_dir = root / Path(report_relative)
    result_path = report_dir / "result.json"
    result = _read_canonical_json(result_path)
    if result.get("schema") != SMOKE_REPORT_SCHEMA:
        raise ValueError("size source is not the verified behavior calibration smoke")
    recorded_report_sha = _require_sha256(
        result.get("report_sha256"), label="smoke report SHA-256"
    )
    unsigned_result = dict(result)
    unsigned_result.pop("report_sha256")
    if canonical_sha256(unsigned_result) != recorded_report_sha:
        raise ValueError("smoke report SHA-256 mismatch")
    if (
        result.get("readback_verified") is not True
        or result.get("promotion_eligible") is not False
        or result.get("production_gate_passed") is not False
        or result.get("strategic_strength_evaluated") is not False
        or result.get("strategic_strength_claimed") is not False
    ):
        raise ValueError("size source must be the verified non-promoting smoke")

    populations: dict[str, Any] = {}
    for population, file_specs in _PAYLOAD_FILES.items():
        report_section = result[population]
        root_count = _require_positive_int(
            report_section.get("root_count"), label=f"{population} smoke root_count"
        )
        files: dict[str, Any] = {}
        for artifact, (relative_suffix, report_hash_key) in file_specs.items():
            file_relative = f"{report_relative}/{relative_suffix}"
            file_path = root / Path(file_relative)
            actual_hash = _sha256_file(file_path)
            expected_hash = _require_sha256(
                report_section.get(report_hash_key),
                label=f"{population}.{artifact} report hash",
            )
            if actual_hash != expected_hash:
                raise ValueError(f"{population}.{artifact} smoke file hash mismatch")
            files[artifact] = {
                "relative_path": file_relative,
                "observed_bytes": file_path.stat().st_size,
                "sha256": actual_hash,
            }
        populations[population] = {
            "observed_root_count": root_count,
            "files": files,
            "observed_payload_bytes": sum(
                entry["observed_bytes"] for entry in files.values()
            ),
        }

    observation: dict[str, Any] = {
        "schema": SIZE_OBSERVATION_SCHEMA,
        "purpose": "linear_capacity_estimate_source_only",
        "estimate_only": True,
        "source_report_relative_path": report_relative,
        "source_report_sha256": recorded_report_sha,
        "source_gate_config_sha256": _require_sha256(
            result.get("gate_config_sha256"), label="source gate config SHA-256"
        ),
        "source_is_nonpromoting_smoke": True,
        "populations": populations,
    }
    observation["observation_sha256"] = canonical_sha256(observation)
    return verify_size_observation(observation, workspace_root=root)


def verify_size_observation(
    observation: Mapping[str, Any], *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    raw = canonical_snapshot(observation)
    expected_keys = {
        "schema",
        "purpose",
        "estimate_only",
        "source_report_relative_path",
        "source_report_sha256",
        "source_gate_config_sha256",
        "source_is_nonpromoting_smoke",
        "populations",
        "observation_sha256",
    }
    if set(raw) != expected_keys:
        raise ValueError("size observation keys mismatch")
    if raw["schema"] != SIZE_OBSERVATION_SCHEMA:
        raise ValueError("unsupported size observation schema")
    if raw["purpose"] != "linear_capacity_estimate_source_only":
        raise ValueError("size observation purpose mismatch")
    if raw["estimate_only"] is not True or raw["source_is_nonpromoting_smoke"] is not True:
        raise ValueError("size observation must remain estimate-only and nonpromoting")
    _normalized_relative_path(
        raw["source_report_relative_path"], label="source_report_relative_path"
    )
    _require_sha256(raw["source_report_sha256"], label="source report SHA-256")
    _require_sha256(
        raw["source_gate_config_sha256"], label="source gate config SHA-256"
    )
    recorded = _require_sha256(
        raw["observation_sha256"], label="size observation SHA-256"
    )
    unsigned = dict(raw)
    unsigned.pop("observation_sha256")
    if canonical_sha256(unsigned) != recorded:
        raise ValueError("size observation SHA-256 mismatch")

    populations = raw["populations"]
    if not isinstance(populations, dict) or set(populations) != set(_PAYLOAD_FILES):
        raise ValueError("size observation populations mismatch")
    for population, expected_files in _PAYLOAD_FILES.items():
        section = populations[population]
        if not isinstance(section, dict) or set(section) != {
            "observed_root_count",
            "files",
            "observed_payload_bytes",
        }:
            raise ValueError(f"{population} size observation keys mismatch")
        _require_positive_int(
            section["observed_root_count"], label=f"{population}.observed_root_count"
        )
        files = section["files"]
        if not isinstance(files, dict) or set(files) != set(expected_files):
            raise ValueError(f"{population} size observation files mismatch")
        total = 0
        for artifact, entry in files.items():
            if not isinstance(entry, dict) or set(entry) != {
                "relative_path",
                "observed_bytes",
                "sha256",
            }:
                raise ValueError(f"{population}.{artifact} size entry keys mismatch")
            _normalized_relative_path(
                entry["relative_path"], label=f"{population}.{artifact}.relative_path"
            )
            observed_bytes = _require_positive_int(
                entry["observed_bytes"], label=f"{population}.{artifact}.observed_bytes"
            )
            _require_sha256(entry["sha256"], label=f"{population}.{artifact}.sha256")
            total += observed_bytes
            if workspace_root is not None:
                path = Path(workspace_root) / Path(entry["relative_path"])
                if path.stat().st_size != observed_bytes or _sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"{population}.{artifact} size source drift")
        if section["observed_payload_bytes"] != total:
            raise ValueError(f"{population} observed payload total mismatch")

    if workspace_root is not None:
        result_path = (
            Path(workspace_root)
            / Path(raw["source_report_relative_path"])
            / "result.json"
        )
        result = _read_canonical_json(result_path)
        if result.get("schema") != SMOKE_REPORT_SCHEMA:
            raise ValueError("size source report schema drift")
        result_sha = _require_sha256(
            result.get("report_sha256"), label="size source report SHA-256"
        )
        unsigned_result = dict(result)
        unsigned_result.pop("report_sha256")
        if (
            result_sha != raw["source_report_sha256"]
            or canonical_sha256(unsigned_result) != result_sha
        ):
            raise ValueError("size source report drift")
        if result.get("gate_config_sha256") != raw["source_gate_config_sha256"]:
            raise ValueError("size source gate binding drift")
        if (
            result.get("readback_verified") is not True
            or result.get("promotion_eligible") is not False
            or result.get("production_gate_passed") is not False
            or result.get("strategic_strength_evaluated") is not False
            or result.get("strategic_strength_claimed") is not False
        ):
            raise ValueError("size source report is no longer a nonpromoting smoke")
        for population, file_specs in _PAYLOAD_FILES.items():
            report_section = result[population]
            observed_section = populations[population]
            if report_section.get("root_count") != observed_section["observed_root_count"]:
                raise ValueError(f"{population} size source root-count drift")
            for artifact, (_suffix, report_hash_key) in file_specs.items():
                if report_section.get(report_hash_key) != observed_section["files"][artifact][
                    "sha256"
                ]:
                    raise ValueError(f"{population}.{artifact} report binding drift")
    return raw


def _all_minimums_met(counts: Mapping[str, int], minimums: Mapping[str, int]) -> bool:
    return all(counts.get(key, 0) >= value for key, value in minimums.items())


def _scan_natural_prefix(
    config: BehaviorTraceCollectionConfig, minimums: Mapping[str, int]
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    if set(minimums) != set(SPLIT_NAMES):
        raise ValueError("natural split minimum keys mismatch")
    for split, value in minimums.items():
        _require_positive_int(value, label=f"natural minimum {split}")
    root_ids: list[str] = []
    counts = {split: 0 for split in SPLIT_NAMES}
    previous_counts = dict(counts)
    root_index = 0
    while True:
        previous_counts = dict(counts)
        root_id = derive_root_id(config, root_index)
        root_ids.append(root_id)
        counts[root_split(root_id)] += 1
        root_index += 1
        if _all_minimums_met(counts, minimums):
            break
    return root_ids, counts, previous_counts


def _challenge_cell_key(root_index: int) -> str:
    turn, actor, joker_count = targeted_joker_cell(root_index)
    return f"t{turn}_{actor}.joker_{joker_count}"


def _challenge_cycle() -> list[dict[str, Any]]:
    result = []
    for offset in range(len(ROLE_KEYS) * len(JOKER_KEYS)):
        turn, actor, joker_count = targeted_joker_cell(offset)
        result.append(
            {
                "cycle_offset": offset,
                "role": f"t{turn}_{actor}",
                "visible_joker_count": joker_count,
                "cell": _challenge_cell_key(offset),
            }
        )
    if len({entry["cell"] for entry in result}) != 12:
        raise AssertionError("collector challenge cycle must contain twelve unique cells")
    return result


def _scan_challenge_prefix(
    config: BehaviorTraceCollectionConfig, minimum_per_cell: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    _require_positive_int(minimum_per_cell, label="challenge minimum_per_cell")
    cell_keys = [entry["cell"] for entry in _challenge_cycle()]
    minimums = {key: minimum_per_cell for key in cell_keys}
    counts = {key: 0 for key in cell_keys}
    previous_counts = dict(counts)
    root_ids: list[str] = []
    root_index = 0
    while True:
        previous_counts = dict(counts)
        root_ids.append(derive_root_id(config, root_index))
        counts[_challenge_cell_key(root_index)] += 1
        root_index += 1
        if _all_minimums_met(counts, minimums):
            break
    return root_ids, counts, previous_counts


def _range_commitments(
    *,
    config: BehaviorTraceCollectionConfig,
    root_ids: Sequence[str],
    shard_size: int,
    population: str,
) -> dict[str, Any]:
    shard_size = _require_positive_int(shard_size, label="shard_size")
    if not root_ids:
        raise ValueError("root range must not be empty")
    shards: list[dict[str, Any]] = []
    for start in range(0, len(root_ids), shard_size):
        stop = min(start + shard_size, len(root_ids))
        shard_ids = list(root_ids[start:stop])
        dimension_counts: Mapping[str, int]
        if population == "natural":
            dimension_counts = dict(
                sorted(Counter(root_split(root_id) for root_id in shard_ids).items())
            )
            dimension_name = "split_root_counts"
        elif population == "joker_challenge":
            dimension_counts = dict(
                sorted(Counter(_challenge_cell_key(index) for index in range(start, stop)).items())
            )
            dimension_name = "targeted_cell_root_counts"
        else:
            raise ValueError("unsupported collection population")
        shard: dict[str, Any] = {
            "shard_index": len(shards),
            "root_index_start": start,
            "root_index_stop_exclusive": stop,
            "root_count": len(shard_ids),
            "first_root_id": shard_ids[0],
            "last_root_id": shard_ids[-1],
            "root_id_order_sha256": canonical_sha256(shard_ids),
            dimension_name: dimension_counts,
        }
        shard["shard_sha256"] = canonical_sha256(shard)
        shards.append(shard)

    split_ids = {
        split: [root_id for root_id in root_ids if root_split(root_id) == split]
        for split in SPLIT_NAMES
    }
    collection_config = config.to_canonical_dict()
    result: dict[str, Any] = {
        "root_index_start": 0,
        "root_index_stop_exclusive": len(root_ids),
        "root_count": len(root_ids),
        "contiguous_prefix": True,
        "root_identity_input_sha256": canonical_sha256(
            {
                "root_identity_schema": ROOT_ID_SCHEMA,
                "collection_config": collection_config,
                "root_index_start": 0,
                "root_index_stop_exclusive": len(root_ids),
            }
        ),
        "root_id_order_sha256": canonical_sha256(list(root_ids)),
        "root_id_set_sha256": canonical_sha256(sorted(root_ids)),
        "root_list_commitments_by_split": {
            split: canonical_sha256(sorted(split_ids[split])) for split in SPLIT_NAMES
        },
        "root_id_order_commitments_by_split": {
            split: canonical_sha256(split_ids[split]) for split in SPLIT_NAMES
        },
        "shard_size_roots": shard_size,
        "shard_count": len(shards),
        "shards": shards,
        "shard_manifest_sha256": canonical_sha256(shards),
    }
    result["root_range_sha256"] = canonical_sha256(result)
    return result


def _capacity_estimate(
    observation: Mapping[str, Any], *, natural_roots: int, challenge_roots: int
) -> dict[str, Any]:
    planned_counts = {
        "natural": natural_roots,
        "joker_challenge": challenge_roots,
    }
    populations: dict[str, Any] = {}
    for population, planned_root_count in planned_counts.items():
        source = observation["populations"][population]
        observed_roots = source["observed_root_count"]
        files: dict[str, Any] = {}
        for artifact, entry in source["files"].items():
            # Ceil is conservative at the byte level; no binary float enters the artifact.
            estimated = (
                entry["observed_bytes"] * planned_root_count + observed_roots - 1
            ) // observed_roots
            files[artifact] = {
                "estimated_bytes": estimated,
                "source_observed_bytes": entry["observed_bytes"],
                "source_observed_root_count": observed_roots,
            }
        payload_total = sum(entry["estimated_bytes"] for entry in files.values())
        populations[population] = {
            "planned_root_count": planned_root_count,
            "files": files,
            "estimated_payload_bytes": payload_total,
        }
    total = sum(section["estimated_payload_bytes"] for section in populations.values())
    return {
        "label": "estimate_only_not_measured_production_bytes",
        "method": "ceil(smoke_file_bytes * planned_roots / smoke_roots) per JSONL file",
        "source_observation": canonical_snapshot(observation),
        "source_observation_sha256": observation["observation_sha256"],
        "compression_assumed": False,
        "manifest_calibration_and_shard_overhead_included": False,
        "production_line_lengths_assumed_equal_to_smoke_average": True,
        "populations": populations,
        "estimated_payload_bytes_total": total,
        "estimated_decimal_gigabytes_milli_ceil": (total * 1_000 + 10**9 - 1) // 10**9,
    }


def _source_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "collector_root_identity_schema": ROOT_ID_SCHEMA,
        "split_namespace": SPLIT_NAMESPACE,
        "split_names": list(SPLIT_NAMES),
        "collector_source_sha256": _sha256_file(Path(collector_module.__file__)),
        "calibration_contract_source_sha256": _sha256_file(
            Path(calibration_contract_module.__file__)
        ),
        "root_identity_function": "derive_root_id(collection_config, root_index)",
        "split_function": "root_split(root_id)",
        "natural_range_semantics": "minimum_satisfying_contiguous_prefix_from_index_zero",
        "challenge_cycle_function": "targeted_joker_cell(root_index)",
    }
    contract["source_contract_sha256"] = canonical_sha256(contract)
    return contract


def build_behavior_collection_plan(
    *,
    gate_config: Mapping[str, Any] | None = None,
    size_observation: Mapping[str, Any],
    natural_seed_namespace: str = DEFAULT_NATURAL_SEED_NAMESPACE,
    challenge_seed_namespace: str = DEFAULT_CHALLENGE_SEED_NAMESPACE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    plan_id: str = DEFAULT_PLAN_ID,
    created_on: str = DEFAULT_CREATED_ON,
) -> dict[str, Any]:
    """Build the deterministic, self-hashed collection preregistration."""
    gate = verify_temperature_gate_config(
        gate_config if gate_config is not None else build_temperature_gate_config()
    )
    observation = verify_size_observation(size_observation)
    _require_nonempty_string(plan_id, label="plan_id")
    _require_nonempty_string(created_on, label="created_on")
    _require_nonempty_string(natural_seed_namespace, label="natural_seed_namespace")
    _require_nonempty_string(challenge_seed_namespace, label="challenge_seed_namespace")
    shard_size = _require_positive_int(shard_size, label="shard_size")

    challenge_prefix = gate["challenge_contract"]["root_namespace_prefix"]
    if not challenge_prefix.endswith("/"):
        raise ValueError("gate challenge root namespace prefix must end with slash")
    challenge_id = challenge_prefix[:-1]
    natural_config = BehaviorTraceCollectionConfig(
        seed_namespace=natural_seed_namespace,
        root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
    )
    challenge_config = BehaviorTraceCollectionConfig(
        seed_namespace=challenge_seed_namespace,
        root_sampling_mode=TARGETED_JOKER_CHALLENGE,
        challenge_id=challenge_id,
        challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
    )

    minimums = gate["minimum_counts"]
    natural_minimums = minimums["decisions_per_role_by_split"]
    natural_ids, natural_counts, natural_previous = _scan_natural_prefix(
        natural_config, natural_minimums
    )
    challenge_minimum = max(
        minimums["challenge_decisions_per_role_joker"],
        minimums["challenge_roots_per_role_joker"],
    )
    challenge_ids, challenge_counts, challenge_previous = _scan_challenge_prefix(
        challenge_config, challenge_minimum
    )

    natural_range = _range_commitments(
        config=natural_config,
        root_ids=natural_ids,
        shard_size=shard_size,
        population="natural",
    )
    challenge_range = _range_commitments(
        config=challenge_config,
        root_ids=challenge_ids,
        shard_size=shard_size,
        population="joker_challenge",
    )
    natural: dict[str, Any] = {
        "collection_config": natural_config.to_canonical_dict(),
        "decisions_per_root_by_role": {role: 1 for role in ROLE_KEYS},
        "total_logged_decisions_per_root": len(ROLE_KEYS),
        "required_decisions_per_role_by_split": canonical_snapshot(natural_minimums),
        "achieved_root_and_decision_counts_by_split": natural_counts,
        "achieved_decision_counts_by_split_role": {
            split: {role: natural_counts[split] for role in ROLE_KEYS}
            for split in SPLIT_NAMES
        },
        "minimum_prefix_proof": {
            "first_satisfying_root_count": len(natural_ids),
            "counts_before_final_root": natural_previous,
            "previous_prefix_satisfied": _all_minimums_met(
                natural_previous, natural_minimums
            ),
            "unmet_splits_before_final_root": [
                split
                for split in SPLIT_NAMES
                if natural_previous[split] < natural_minimums[split]
            ],
        },
        "total_logged_decision_count": len(natural_ids) * len(ROLE_KEYS),
        "range": natural_range,
    }
    cycle = _challenge_cycle()
    challenge: dict[str, Any] = {
        "collection_config": challenge_config.to_canonical_dict(),
        "used_for_temperature_fit": False,
        "used_for_dev_selection": False,
        "used_for_locked_test_role_metrics": False,
        "targeted_diagnostics_only": True,
        "cycle_length_roots": len(cycle),
        "cycle": cycle,
        "total_logged_decisions_per_root": len(ROLE_KEYS),
        "targeted_decisions_per_root": 1,
        "required_targeted_roots_and_decisions_per_cell": challenge_minimum,
        "gate_required_challenge_decisions_per_role_joker": minimums[
            "challenge_decisions_per_role_joker"
        ],
        "gate_required_challenge_roots_per_role_joker": minimums[
            "challenge_roots_per_role_joker"
        ],
        "achieved_targeted_root_and_decision_counts_by_cell": challenge_counts,
        "minimum_prefix_proof": {
            "first_satisfying_root_count": len(challenge_ids),
            "counts_before_final_root": challenge_previous,
            "previous_prefix_satisfied": all(
                count >= challenge_minimum for count in challenge_previous.values()
            ),
            "unmet_cells_before_final_root": sorted(
                cell
                for cell, count in challenge_previous.items()
                if count < challenge_minimum
            ),
        },
        "targeted_decision_count": len(challenge_ids),
        "total_logged_decision_count": len(challenge_ids) * len(ROLE_KEYS),
        "range": challenge_range,
    }

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "plan_id": plan_id,
        "created_on": created_on,
        "purpose": "production_sample_count_collection_preregistration_only",
        "large_collection_executed": False,
        "calibration_quality_evaluated": False,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "promotion_eligible": False,
        "temperature_gate": {
            "schema": GATE_CONFIG_SCHEMA,
            "gate_id": gate["gate_id"],
            "gate_config_sha256": gate["gate_config_sha256"],
            "verified": True,
            "config": canonical_snapshot(gate),
        },
        "source_contract": _source_contract(),
        "natural": natural,
        "joker_challenge": challenge,
        "disk_size_estimate": _capacity_estimate(
            observation,
            natural_roots=len(natural_ids),
            challenge_roots=len(challenge_ids),
        ),
        "claims": {
            "proves_minimum_sample_count_plan": True,
            "proves_collection_was_run": False,
            "proves_temperature_gate_passed": False,
            "proves_m3_strength": False,
            "promotes_any_policy_or_behavior_model": False,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_behavior_collection_plan(
    plan: Mapping[str, Any],
    *,
    gate_config: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild every count/commitment and reject content or contract drift."""
    raw = canonical_snapshot(plan)
    expected_keys = {
        "schema",
        "plan_id",
        "created_on",
        "purpose",
        "large_collection_executed",
        "calibration_quality_evaluated",
        "strategic_strength_evaluated",
        "strategic_strength_claimed",
        "promotion_eligible",
        "temperature_gate",
        "source_contract",
        "natural",
        "joker_challenge",
        "disk_size_estimate",
        "claims",
        "plan_sha256",
    }
    if set(raw) != expected_keys:
        raise ValueError("behavior collection plan keys mismatch")
    if raw["schema"] != PLAN_SCHEMA:
        raise ValueError("unsupported behavior collection plan schema")
    recorded = _require_sha256(raw["plan_sha256"], label="plan SHA-256")
    unsigned = dict(raw)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != recorded:
        raise ValueError("behavior collection plan SHA-256 mismatch")

    embedded_gate = verify_temperature_gate_config(
        raw["temperature_gate"]["config"]
    )
    if gate_config is not None:
        current_gate = verify_temperature_gate_config(gate_config)
        if current_gate["gate_config_sha256"] != embedded_gate["gate_config_sha256"]:
            raise ValueError("temperature gate config drift")
    observation = verify_size_observation(
        raw["disk_size_estimate"]["source_observation"],
        workspace_root=workspace_root,
    )
    expected = build_behavior_collection_plan(
        gate_config=embedded_gate,
        size_observation=observation,
        natural_seed_namespace=raw["natural"]["collection_config"]["seed_namespace"],
        challenge_seed_namespace=raw["joker_challenge"]["collection_config"][
            "seed_namespace"
        ],
        shard_size=raw["natural"]["range"]["shard_size_roots"],
        plan_id=raw["plan_id"],
        created_on=raw["created_on"],
    )
    if canonical_json(expected) != canonical_json(raw):
        raise ValueError("behavior collection plan does not match exact rederivation")
    return raw


def write_behavior_collection_plan(path: str | Path, plan: Mapping[str, Any]) -> Path:
    verified = verify_behavior_collection_plan(plan)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(verified) + "\n").encode("utf-8")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if output.read_bytes() != payload:
        raise RuntimeError("behavior collection plan readback mismatch")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--smoke-report", default=DEFAULT_SMOKE_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--natural-seed-namespace", default=DEFAULT_NATURAL_SEED_NAMESPACE
    )
    parser.add_argument(
        "--challenge-seed-namespace", default=DEFAULT_CHALLENGE_SEED_NAMESPACE
    )
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace_root)
    observation = build_smoke_size_observation(workspace, args.smoke_report)
    gate = build_temperature_gate_config()
    plan = build_behavior_collection_plan(
        gate_config=gate,
        size_observation=observation,
        natural_seed_namespace=args.natural_seed_namespace,
        challenge_seed_namespace=args.challenge_seed_namespace,
        shard_size=args.shard_size,
    )
    verify_behavior_collection_plan(plan, gate_config=gate, workspace_root=workspace)
    output = Path(args.output)
    if not output.is_absolute():
        output = workspace / output
    write_behavior_collection_plan(output, plan)
    print(
        canonical_json(
            {
                "output": str(output),
                "plan_sha256": plan["plan_sha256"],
                "natural_root_count": plan["natural"]["range"]["root_count"],
                "challenge_root_count": plan["joker_challenge"]["range"][
                    "root_count"
                ],
                "large_collection_executed": False,
                "promotion_eligible": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHALLENGE_SEED_NAMESPACE",
    "DEFAULT_NATURAL_SEED_NAMESPACE",
    "DEFAULT_OUTPUT",
    "DEFAULT_SHARD_SIZE",
    "DEFAULT_SMOKE_REPORT",
    "PLAN_SCHEMA",
    "SIZE_OBSERVATION_SCHEMA",
    "build_behavior_collection_plan",
    "build_smoke_size_observation",
    "verify_behavior_collection_plan",
    "verify_size_observation",
    "write_behavior_collection_plan",
]
