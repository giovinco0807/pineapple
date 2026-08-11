"""Fail-closed candidate01/candidate02 native semantic differential.

This validator compares the complete raw JSON returned by two isolated M3
native libraries.  Every JSON float is compared by its IEEE-754 binary64 bit
pattern; object order is ignored, while array order and numeric JSON types are
preserved.  Consequently the comparison covers the ActionKey/index mapping,
all selection/evaluation Q values, the selected action, action ordering
digests, and the T3 child information-set count without a tolerance.

Step6d performance root artifacts can be supplied directly.  Bare
``ActorObservation`` JSON is also accepted, which makes T4 exact fixtures easy
to add without introducing another card-bearing schema.  Opponent private
discards and realized deck tails are never accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import (
    ActionKey,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_infoset import OBSERVATION_SCHEMA, ActorObservation
from .hu_late_street_teacher import T4SearchConfig
from .hu_m3_rust import (
    engine_version,
    evaluate_request,
    load_native_engine,
    t3_request,
    t4_request,
)
from .hu_turn3_joint_exact_teacher import JointExactConfig


SCHEMA = "hu_m31_t3_candidate02_equivalence_validation_v1"
STEP6D_ROOT_SCHEMA = "hu_m31_t3_step6d_performance_root_v1"
DEFAULT_SEED = 2026071701
DEFAULT_CANDIDATE_SEED = 2026071702
DEFAULT_EVALUATION_SEED = 2026071703

_STEP6D_ROOT_KEYS = frozenset(
    {
        "budget",
        "contract_canonical_sha256",
        "current_profile_resolved",
        "hand_index",
        "observations",
        "opponent_private_discards_used",
        "profile",
        "root_indices",
        "schedule",
        "schedule_row_sha256",
        "schema",
        "seeds",
        "training_eligible",
    }
)
_STEP6D_OBSERVATION_KEYS = frozenset(
    {"observation", "observation_fingerprint", "root_index", "seat"}
)
_BUDGET_KEYS = frozenset(
    {
        "candidate_samples",
        "evaluation_samples",
        "downstream_t3_samples",
        "downstream_t4_samples",
    }
)
_SEED_KEYS = frozenset(
    {"behavior", "candidate", "child", "confirmation", "evaluation", "hand"}
)


@dataclass(frozen=True)
class DifferentialConfig:
    candidate_samples: int = 1
    evaluation_samples: int = 1
    downstream_t3_samples: int = 1
    downstream_t4_samples: int = 0
    seed: int = DEFAULT_SEED
    candidate_seed: int = DEFAULT_CANDIDATE_SEED
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    run_id_prefix: str = "hu-m31-candidate02-equivalence-v1"
    use_root_contract: bool = False

    def __post_init__(self) -> None:
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.downstream_t4_samples != 0:
            raise ValueError(
                "candidate02 equivalence requires exact downstream T4 (zero samples)"
            )
        for name in ("seed", "candidate_seed", "evaluation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if not isinstance(self.run_id_prefix, str) or not self.run_id_prefix:
            raise ValueError("run_id_prefix must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": self.downstream_t4_samples,
            "seed": self.seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "run_id_prefix": self.run_id_prefix,
            "use_root_contract": self.use_root_contract,
        }


@dataclass(frozen=True)
class RootCase:
    case_id: str
    source_path: Path
    source_sha256: str
    observation: ActorObservation
    root_index: int | None
    budget: Mapping[str, int] | None = None
    seeds: Mapping[str, int] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def f64_bit_projection(value: Any) -> list[Any]:
    """Return an unambiguous typed tree with floats encoded as big-endian bits."""

    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("native result contains a non-finite float")
        return ["f64", struct.pack(">d", value).hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["array", [f64_bit_projection(item) for item in value]]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("native result object keys must be strings")
        return [
            "object",
            [[key, f64_bit_projection(value[key])] for key in sorted(value)],
        ]
    raise TypeError(f"unsupported native result value: {type(value).__name__}")


def f64_bit_projection_sha256(value: Any) -> str:
    return canonical_sha256(f64_bit_projection(value))


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "f64"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def first_bit_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """Return the first structural or f64-bit difference in canonical key order."""

    left_kind = _json_kind(left)
    right_kind = _json_kind(right)
    if left_kind != right_kind:
        return f"{path}:type:{left_kind}!={right_kind}"
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            return f"{path}:non_finite"
        left_bits = struct.pack(">d", left).hex()
        right_bits = struct.pack(">d", right).hex()
        return (
            None
            if left_bits == right_bits
            else f"{path}:f64_bits:{left_bits}!={right_bits}"
        )
    if isinstance(left, Mapping):
        left_keys = sorted(left)
        right_keys = sorted(right)
        if left_keys != right_keys:
            return f"{path}:keys:{left_keys!r}!={right_keys!r}"
        for key in left_keys:
            child = first_bit_difference(
                left[key], right[key], f"{path}[{json.dumps(key, ensure_ascii=True)}]"
            )
            if child is not None:
                return child
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}:length:{len(left)}!={len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            child = first_bit_difference(left_item, right_item, f"{path}[{index}]")
            if child is not None:
                return child
        return None
    return None if left == right else f"{path}:value:{left!r}!={right!r}"


def compare_raw_results(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    difference = first_bit_difference(left, right)
    return {
        "exact": difference is None,
        "left_f64_bit_projection_sha256": f64_bit_projection_sha256(left),
        "right_f64_bit_projection_sha256": f64_bit_projection_sha256(right),
        "first_difference": difference,
    }


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    return resolved


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys changed: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _parse_budget(value: Any) -> dict[str, int]:
    payload = _mapping(value, "Step6d root budget")
    _require_exact_keys(payload, _BUDGET_KEYS, "Step6d root budget")
    result = {
        "candidate_samples": _integer(
            payload["candidate_samples"], "candidate_samples", minimum=1
        ),
        "evaluation_samples": _integer(
            payload["evaluation_samples"], "evaluation_samples", minimum=1
        ),
        "downstream_t3_samples": _integer(
            payload["downstream_t3_samples"], "downstream_t3_samples", minimum=1
        ),
        "downstream_t4_samples": _integer(
            payload["downstream_t4_samples"], "downstream_t4_samples"
        ),
    }
    if result["downstream_t4_samples"] != 0:
        raise ValueError("candidate02 equivalence root must use exact downstream T4")
    return result


def _parse_seeds(value: Any) -> dict[str, int]:
    payload = _mapping(value, "Step6d root seeds")
    _require_exact_keys(payload, _SEED_KEYS, "Step6d root seeds")
    return {
        key: _integer(payload[key], f"Step6d {key} seed") for key in sorted(_SEED_KEYS)
    }


def load_root_cases(path: Path) -> list[RootCase]:
    resolved = _require_regular_file(path, "root artifact")
    source_sha = sha256_file(resolved)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"root artifact is not canonical JSON: {resolved}") from exc
    payload = _mapping(raw, "root artifact")
    schema = payload.get("schema")
    if schema == OBSERVATION_SCHEMA:
        observation = ActorObservation.from_dict(payload)
        return [
            RootCase(
                case_id=f"{source_sha[:12]}-{observation.street.lower()}-{observation.seat}-{observation.fingerprint()[:12]}",
                source_path=resolved,
                source_sha256=source_sha,
                observation=observation,
                root_index=None,
            )
        ]
    if schema != STEP6D_ROOT_SCHEMA:
        raise ValueError(f"unsupported root artifact schema: {schema!r}")
    _require_exact_keys(payload, _STEP6D_ROOT_KEYS, "Step6d root artifact")
    if payload.get("opponent_private_discards_used") is not False:
        raise ValueError("Step6d root artifact must not use opponent private discards")
    if payload.get("training_eligible") is not False:
        raise ValueError("Step6d performance roots must not be training eligible")
    if payload.get("current_profile_resolved") is not False:
        raise ValueError("Step6d root artifact must not resolve current")
    budget = _parse_budget(payload["budget"])
    seeds = _parse_seeds(payload["seeds"])
    rows = payload.get("observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Step6d root artifact observations must be a nonempty list")
    root_indices = payload.get("root_indices")
    if not isinstance(root_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in root_indices
    ):
        raise ValueError("Step6d root_indices must be an integer list")
    cases: list[RootCase] = []
    observed_indices: list[int] = []
    for position, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"Step6d observation {position}")
        _require_exact_keys(
            row, _STEP6D_OBSERVATION_KEYS, f"Step6d observation {position}"
        )
        root_index = _integer(row.get("root_index"), "Step6d root_index")
        observation = ActorObservation.from_dict(
            _mapping(row.get("observation"), "Step6d ActorObservation")
        )
        if observation.street != "T3":
            raise ValueError("Step6d performance root observations must be T3")
        if row.get("seat") != observation.seat:
            raise ValueError("Step6d observation seat disagrees with ActorObservation")
        fingerprint = observation.fingerprint()
        if row.get("observation_fingerprint") != fingerprint:
            raise ValueError("Step6d observation fingerprint mismatch")
        observed_indices.append(root_index)
        cases.append(
            RootCase(
                case_id=f"{source_sha[:12]}-root-{root_index:06d}-{observation.seat}-{fingerprint[:12]}",
                source_path=resolved,
                source_sha256=source_sha,
                observation=observation,
                root_index=root_index,
                budget=budget,
                seeds=seeds,
            )
        )
    if observed_indices != root_indices or len(set(observed_indices)) != len(
        observed_indices
    ):
        raise ValueError("Step6d root_indices disagree with ordered observations")
    return cases


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def validate_result_structure(
    result: Mapping[str, Any], observation: ActorObservation
) -> dict[str, int]:
    """Independently bind raw indices, ActionKeys, and digests to the root."""

    if result.get("status") != "ok" or result.get("kind") != observation.street.lower():
        raise ValueError("native result status/kind mismatch")
    if result.get("street") != observation.street:
        raise ValueError("native result street mismatch")
    if (
        result.get("seat") != observation.seat
        or result.get("to_act_order") != observation.to_act_order
    ):
        raise ValueError("native result seat/order mismatch")
    if result.get("observation_fingerprint") != observation.fingerprint():
        raise ValueError("native result observation fingerprint mismatch")
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise ValueError("equivalence root has no legal actions")
    if result.get("legal_action_count") != len(actions):
        raise ValueError("native result legal_action_count mismatch")
    if result.get("legal_action_order_digest") != ordered_action_mapping_digest(
        actions
    ):
        raise ValueError("native result legal action order digest mismatch")
    if result.get("legal_action_set_digest") != legal_action_set_digest(actions):
        raise ValueError("native result legal action set digest mismatch")
    raw_rows = result.get("actions")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(actions):
        raise ValueError("native result must cover every legal action")
    original_indices: set[int] = set()
    sorted_indices: set[int] = set()
    keys: set[str] = set()
    selected_flag_indices: list[int] = []
    for list_index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, f"native action row {list_index}")
        original_index = _integer(row.get("original_index"), "original_index")
        sorted_index = _integer(row.get("sorted_index"), "sorted_index")
        if original_index >= len(actions) or sorted_index >= len(actions):
            raise ValueError("native action index is outside legal action range")
        if sorted_index != list_index:
            raise ValueError("native action row order disagrees with sorted_index")
        token = row.get("action_key")
        if not isinstance(token, str):
            raise ValueError("native action row lacks ActionKey")
        parsed = ActionKey.from_token(token)
        expected = action_key(actions[original_index])
        if parsed != expected or action_key_from_payload(row) != expected:
            raise ValueError("native ActionKey/index/payload mapping mismatch")
        for field in ("selection_score", "score", "joint_ev"):
            _finite_number(row.get(field), f"native action {field}")
        if struct.pack(">d", _finite_number(row["score"], "score")) != struct.pack(
            ">d", _finite_number(row["joint_ev"], "joint_ev")
        ):
            raise ValueError("native score and joint_ev bits differ")
        selected = row.get("selected_by_candidate_plan")
        if not isinstance(selected, bool):
            raise ValueError("native selected_by_candidate_plan must be boolean")
        if selected:
            selected_flag_indices.append(original_index)
        original_indices.add(original_index)
        sorted_indices.add(sorted_index)
        keys.add(token)
    expected_range = set(range(len(actions)))
    if (
        original_indices != expected_range
        or sorted_indices != expected_range
        or len(keys) != len(actions)
    ):
        raise ValueError("native result action index/key coverage is incomplete")
    selected_index = _integer(
        result.get("selected_action_original_index"), "selected_action_original_index"
    )
    if selected_index >= len(actions):
        raise ValueError("native selected action index is outside legal range")
    selected_key = result.get("selected_action_key")
    if selected_key != action_key(actions[selected_index]).to_token():
        raise ValueError("native selected ActionKey/index mismatch")
    if selected_flag_indices != [selected_index]:
        raise ValueError(
            "native selected action flag does not match selected ActionKey/index"
        )
    child_count = 0
    if observation.street == "T3":
        child_count = _integer(
            result.get("child_information_set_count"),
            "child_information_set_count",
        )
    return {
        "legal_action_count": len(actions),
        "child_information_set_count": child_count,
    }


def _case_parameters(case: RootCase, config: DifferentialConfig) -> dict[str, int]:
    if not config.use_root_contract:
        return {
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "downstream_t3_samples": config.downstream_t3_samples,
            "downstream_t4_samples": config.downstream_t4_samples,
            "seed": config.seed,
            "candidate_seed": config.candidate_seed,
            "evaluation_seed": config.evaluation_seed,
        }
    if case.budget is None or case.seeds is None:
        raise ValueError(
            "--use-root-contract requires a Step6d performance root artifact"
        )
    parameters = {
        **dict(case.budget),
        "seed": case.seeds["child"],
        "candidate_seed": case.seeds["candidate"],
        "evaluation_seed": case.seeds["evaluation"],
    }
    if parameters["candidate_seed"] == parameters["evaluation_seed"]:
        raise ValueError("Step6d root candidate/evaluation seeds must be distinct")
    return parameters


def _requests_for_case(
    case: RootCase, config: DifferentialConfig
) -> list[tuple[str, dict[str, Any]]]:
    parameters = _case_parameters(case, config)
    run_id = f"{config.run_id_prefix}:{case.observation.fingerprint()}"
    if case.observation.street == "T4":
        request = t4_request(
            case.observation,
            config=T4SearchConfig(
                candidate_samples=0,
                evaluation_samples=0,
                seed=parameters["seed"],
                candidate_seed=parameters["candidate_seed"],
                evaluation_seed=parameters["evaluation_seed"],
                run_id=run_id,
            ),
        )
        return [("exact", request)]
    if case.observation.street != "T3":
        raise ValueError("candidate02 differential accepts only T3 or T4 observations")
    requests: list[tuple[str, dict[str, Any]]] = []
    for enabled in (True, False):
        request = t3_request(
            case.observation,
            config=JointExactConfig(
                candidate_samples=parameters["candidate_samples"],
                evaluation_samples=parameters["evaluation_samples"],
                downstream_t3_samples=parameters["downstream_t3_samples"],
                downstream_t4_samples=parameters["downstream_t4_samples"],
                seed=parameters["seed"],
                candidate_seed=parameters["candidate_seed"],
                evaluation_seed=parameters["evaluation_seed"],
                run_id=run_id,
                seat=case.observation.seat,
                to_act_order=case.observation.to_act_order,
                use_final_turn_cache=enabled,
            ),
        )
        requests.append(("cache_on" if enabled else "cache_off", request))
    return requests


def _result_record(
    result: Mapping[str, Any], geometry: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "f64_bit_projection_sha256": f64_bit_projection_sha256(result),
        "legal_action_count": geometry["legal_action_count"],
        "child_information_set_count": geometry["child_information_set_count"],
        "selected_action_key": result["selected_action_key"],
        "legal_action_order_digest": result["legal_action_order_digest"],
        "legal_action_set_digest": result["legal_action_set_digest"],
    }


def run_differential(
    *,
    candidate01_library_path: Path,
    candidate02_library_path: Path,
    root_paths: Sequence[Path],
    config: DifferentialConfig | None = None,
    expected_candidate01_sha256: str | None = None,
    expected_candidate02_sha256: str | None = None,
) -> dict[str, Any]:
    config = config or DifferentialConfig()
    old_path = _require_regular_file(candidate01_library_path, "candidate01 library")
    new_path = _require_regular_file(candidate02_library_path, "candidate02 library")
    if old_path == new_path:
        raise ValueError("candidate01 and candidate02 libraries must be distinct files")
    old_sha = sha256_file(old_path)
    new_sha = sha256_file(new_path)
    if old_sha == new_sha:
        raise ValueError(
            "candidate01 and candidate02 libraries must have distinct SHA-256"
        )
    if expected_candidate01_sha256 is not None and old_sha != _validate_sha(
        expected_candidate01_sha256, "expected candidate01 SHA-256"
    ):
        raise ValueError("candidate01 library SHA-256 does not match the pinned value")
    if expected_candidate02_sha256 is not None and new_sha != _validate_sha(
        expected_candidate02_sha256, "expected candidate02 SHA-256"
    ):
        raise ValueError("candidate02 library SHA-256 does not match the pinned value")
    if not root_paths:
        raise ValueError("at least one root artifact is required")
    cases: list[RootCase] = []
    seen_paths: set[Path] = set()
    seen_fingerprints: set[str] = set()
    for root_path in root_paths:
        resolved = _require_regular_file(root_path, "root artifact")
        if resolved in seen_paths:
            raise ValueError(f"duplicate root artifact: {resolved}")
        seen_paths.add(resolved)
        for case in load_root_cases(resolved):
            fingerprint = case.observation.fingerprint()
            if fingerprint in seen_fingerprints:
                raise ValueError(f"duplicate observation fingerprint: {fingerprint}")
            seen_fingerprints.add(fingerprint)
            cases.append(case)

    old_library = load_native_engine(path=old_path)
    new_library = load_native_engine(path=new_path)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "decision": "pending",
        "candidate01": {
            "path": str(old_path),
            "sha256": old_sha,
            "engine_version": engine_version(library=old_library),
        },
        "candidate02": {
            "path": str(new_path),
            "sha256": new_sha,
            "engine_version": engine_version(library=new_library),
        },
        "config": config.to_dict(),
        "root_artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "case_ids": [
                    case.case_id for case in cases if case.source_path == path
                ],
            }
            for path in sorted(seen_paths, key=str)
        ],
        "cases": [],
        "failure": None,
    }
    try:
        for case in cases:
            case_record: dict[str, Any] = {
                "case_id": case.case_id,
                "source_path": str(case.source_path),
                "source_sha256": case.source_sha256,
                "root_index": case.root_index,
                "street": case.observation.street,
                "seat": case.observation.seat,
                "observation_fingerprint": case.observation.fingerprint(),
                "modes": [],
                "cache_invariance": None,
            }
            mode_results: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
            for mode, request in _requests_for_case(case, config):
                old_result = evaluate_request(request, library=old_library)
                new_result = evaluate_request(request, library=new_library)
                old_geometry = validate_result_structure(old_result, case.observation)
                new_geometry = validate_result_structure(new_result, case.observation)
                comparison = compare_raw_results(old_result, new_result)
                case_record["modes"].append(
                    {
                        "mode": mode,
                        "request_sha256": canonical_sha256(request),
                        "candidate01": _result_record(old_result, old_geometry),
                        "candidate02": _result_record(new_result, new_geometry),
                        "candidate01_vs_candidate02": comparison,
                    }
                )
                if comparison["exact"] is not True:
                    raise ValueError(
                        f"{case.case_id} {mode} candidate01/candidate02 mismatch: "
                        f"{comparison['first_difference']}"
                    )
                mode_results[mode] = (old_result, new_result)
            if case.observation.street == "T3":
                old_cache = compare_raw_results(
                    mode_results["cache_on"][0], mode_results["cache_off"][0]
                )
                new_cache = compare_raw_results(
                    mode_results["cache_on"][1], mode_results["cache_off"][1]
                )
                case_record["cache_invariance"] = {
                    "candidate01": old_cache,
                    "candidate02": new_cache,
                }
                if old_cache["exact"] is not True or new_cache["exact"] is not True:
                    failing = (
                        "candidate01"
                        if old_cache["exact"] is not True
                        else "candidate02"
                    )
                    detail = old_cache if failing == "candidate01" else new_cache
                    raise ValueError(
                        f"{case.case_id} {failing} cache on/off mismatch: "
                        f"{detail['first_difference']}"
                    )
            artifact["cases"].append(case_record)
    except Exception as exc:  # Preserve one deterministic no-go reason in the artifact.
        artifact["status"] = "no_go"
        artifact["decision"] = "candidate02_equivalence_no_go"
        artifact["failure"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        return artifact
    artifact["status"] = "go"
    artifact["decision"] = "candidate02_bit_exact_equivalence_passed"
    return artifact


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON with create-if-absent semantics."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate01-library", type=Path, required=True)
    parser.add_argument("--candidate02-library", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidate01-sha256")
    parser.add_argument("--expected-candidate02-sha256")
    parser.add_argument("--candidate-samples", type=int, default=1)
    parser.add_argument("--evaluation-samples", type=int, default=1)
    parser.add_argument("--downstream-t3-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-seed", type=int, default=DEFAULT_CANDIDATE_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--run-id-prefix", default="hu-m31-candidate02-equivalence-v1")
    parser.add_argument("--use-root-contract", action="store_true")
    parser.add_argument("--rayon-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rayon_threads <= 0:
        raise ValueError("--rayon-threads must be positive")
    existing_threads = os.environ.get("RAYON_NUM_THREADS")
    if existing_threads is not None and existing_threads != str(args.rayon_threads):
        raise ValueError(
            "RAYON_NUM_THREADS conflicts with the locked --rayon-threads allocation"
        )
    os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    config = DifferentialConfig(
        candidate_samples=args.candidate_samples,
        evaluation_samples=args.evaluation_samples,
        downstream_t3_samples=args.downstream_t3_samples,
        downstream_t4_samples=0,
        seed=args.seed,
        candidate_seed=args.candidate_seed,
        evaluation_seed=args.evaluation_seed,
        run_id_prefix=args.run_id_prefix,
        use_root_contract=args.use_root_contract,
    )
    artifact = run_differential(
        candidate01_library_path=args.candidate01_library,
        candidate02_library_path=args.candidate02_library,
        root_paths=args.root,
        config=config,
        expected_candidate01_sha256=args.expected_candidate01_sha256,
        expected_candidate02_sha256=args.expected_candidate02_sha256,
    )
    write_json_once(args.output, artifact)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "decision": artifact["decision"],
                "output": str(args.output.resolve()),
                "case_count": len(artifact["cases"]),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "go" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"candidate02 equivalence validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


__all__ = [
    "DifferentialConfig",
    "RootCase",
    "SCHEMA",
    "canonical_sha256",
    "compare_raw_results",
    "f64_bit_projection",
    "f64_bit_projection_sha256",
    "first_bit_difference",
    "load_root_cases",
    "main",
    "run_differential",
    "validate_result_structure",
    "write_json_once",
]
