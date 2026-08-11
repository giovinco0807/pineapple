"""Fail-closed local executor for the M3.1 T3 teacher dataset.

The first 25-paired-hand shard is opened by a qualified fresh-quality receipt.
Every remaining shard requires a separately persisted, source-replayed smoke
gate in addition to that original fresh-quality receipt.  This module does not
launch cloud resources, train a model, add a profile, or resolve ``current``.

The lifecycle is:

1. source-replay and bind the qualified fresh-quality gate;
2. generate the two public ``ActorObservation`` roots for one pair;
3. retain the complete accepted Candidate02 decision certificates;
4. replay every ActionKey, Q, RNG, belief, and result certificate;
5. materialize the frozen dataset-contract pair JSON;
6. publish ``SHARD_DONE.json`` only after all 25 pairs and evidence files pass;
7. gate the remaining 8,975 pairs on the completed smoke shard;
8. merge the exact 360-shard grid without gaps or duplicates; and
9. optionally flatten the smoke merge to an immutable Apache Parquet artifact.

Search estimates remain teacher labels, not realized match EV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .ai_profiles import ModelBundle, ModelPaths, build_policy, load_model_bundle
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .hu_m31_t3_step6c_contract import STEP6C_RUN_ID
from . import hu_m31_t3_dataset_contract_v1 as contract
from . import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate
from . import run_hu_m31_t3_step6c_shard as step6c


AUTHORIZATION_SCHEMA = "hu_m31_t3_dataset_smoke_authorization_v1"
FULL_AUTHORIZATION_SCHEMA = "hu_m31_t3_dataset_full_authorization_v1"
PAIR_EVIDENCE_SCHEMA = "hu_m31_t3_dataset_pair_search_evidence_v1"
SMOKE_MERGE_SCHEMA = "hu_m31_t3_dataset_smoke_merge_v1"
PARQUET_MANIFEST_SCHEMA = "hu_m31_t3_dataset_smoke_parquet_manifest_v1"

AUTHORIZATION_NAME = "DATASET_AUTHORIZATION.json"
DONE_NAME = "SHARD_DONE.json"
EVIDENCE_DIRECTORY_NAME = "evidence"
SMOKE_MERGE_NAME = "SMOKE_MERGE.json"
PARQUET_FILE_NAME = "action_rows.parquet"
PARQUET_MANIFEST_NAME = "PARQUET_MANIFEST.json"

_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "shard_id",
        "fresh_quality_gate_schema",
        "fresh_quality_gate_sha256",
        "fresh_quality_merge_sha256",
        "fresh_quality_decision",
        "source_replayed",
        "data_pilot_25_paired_authorized",
        "full_9000_paired_fanout_authorized",
        "current_profile_registry_sha256",
        "current_profile_changed",
    }
)
_FULL_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "shard_id",
        "fresh_quality_gate_schema",
        "fresh_quality_gate_sha256",
        "fresh_quality_merge_sha256",
        "fresh_quality_decision",
        "dataset_smoke_gate_schema",
        "dataset_smoke_gate_sha256",
        "dataset_smoke_gate_decision",
        "smoke_shard_id",
        "smoke_shard_done_sha256",
        "source_replayed",
        "data_pilot_25_paired_authorized",
        "full_9000_paired_fanout_authorized",
        "current_profile_registry_sha256",
        "current_profile_changed",
    }
)
_PAIR_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "authorization_sha256",
        "candidate_library_sha256",
        "pair_contract",
        "pair_contract_sha256",
        "shard_id",
        "split",
        "local_pair_index",
        "global_pair_index",
        "root_indices",
        "profile",
        "seeds",
        "confirmation_required",
        "rows",
        "opponent_private_discards_used",
        "realized_deck_tail_used",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)
_EVIDENCE_ROW_KEYS = frozenset(
    {
        "root_index",
        "seat",
        "observation_fingerprint",
        "observation_sha256",
        "observation",
        "baseline_action_key",
        "primary_decision",
        "confirmation_decision",
    }
)
_EVIDENCE_RECORD_KEYS = frozenset(
    {
        "split",
        "local_pair_index",
        "global_pair_index",
        "shard_id",
        "path",
        "sha256",
        "bytes",
    }
)
_SMOKE_MERGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "shard_id",
        "authorization_sha256",
        "fresh_quality_gate_sha256",
        "shard_done_sha256",
        "pair_records",
        "pair_record_aggregate_sha256",
        "evidence_records",
        "evidence_record_aggregate_sha256",
        "paired_hand_count",
        "root_count",
        "seat_counts",
        "confirmation_pair_count",
        "action_row_count",
        "action_row_aggregate_sha256",
        "hidden_information_field_count",
        "unknown_field_count",
        "action_key_mapping_mismatch_count",
        "rng_overlap_count",
        "missing_or_duplicate_count",
        "data_pilot_25_paired_complete",
        "full_9000_paired_fanout_authorized",
        "training_eligible",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)
_PARQUET_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "smoke_merge_sha256",
        "shard_done_sha256",
        "authorization_sha256",
        "source_action_row_aggregate_sha256",
        "column_schema",
        "parquet_file",
        "parquet_sha256",
        "parquet_bytes",
        "row_count",
        "paired_hand_count",
        "root_count",
        "immutable",
        "source_replayed",
        "full_9000_paired_fanout_authorized",
        "training_eligible",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)

_EVIDENCE_FILE_PREFIX = "pair_"
_EVIDENCE_FILE_SUFFIX = ".json"
_SHA256_LENGTH = 64
_AI_PROFILES_PATH = Path(__file__).with_name("ai_profiles.py")

# Primitive-only schema keeps the training artifact portable.  Observation and
# semantic action identities are canonical JSON/token strings; nullable fields
# exist only for the pre-registered confirmation labels/certificates.
PARQUET_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("split", "string", False),
    ("shard_id", "string", False),
    ("local_pair_index", "int64", False),
    ("global_pair_index", "int64", False),
    ("profile", "string", False),
    ("root_index", "int64", False),
    ("seat", "string", False),
    ("confirmation_required", "bool", False),
    ("hand_seed", "int64", False),
    ("behavior_seed", "int64", False),
    ("candidate_seed", "int64", False),
    ("evaluation_seed", "int64", False),
    ("child_seed", "int64", False),
    ("confirmation_seed", "int64", False),
    ("observation_fingerprint", "string", False),
    ("observation_sha256", "string", False),
    ("observation_json", "string", False),
    ("legal_action_set_digest", "string", False),
    ("legal_action_order_digest", "string", False),
    ("action_index", "int64", False),
    ("action_key", "string", False),
    ("is_baseline", "bool", False),
    ("is_teacher_selected", "bool", False),
    ("search_selected_action_key", "string", False),
    ("primary_selection_q", "float64", False),
    ("primary_q", "float64", False),
    ("primary_delta", "float64", False),
    ("primary_rank", "int64", False),
    ("state_value", "float64", False),
    ("confirmation_q", "float64", True),
    ("confirmation_delta", "float64", True),
    ("confirmation_rank", "int64", True),
    ("confirmation_state_value", "float64", True),
    ("primary_search_contract_digest", "string", False),
    ("primary_semantic_result_digest", "string", False),
    ("primary_result_digest", "string", False),
    ("primary_candidate_rng_digest", "string", False),
    ("primary_evaluation_rng_digest", "string", False),
    ("confirmation_search_contract_digest", "string", True),
    ("confirmation_semantic_result_digest", "string", True),
    ("confirmation_result_digest", "string", True),
    ("confirmation_candidate_rng_digest", "string", True),
    ("confirmation_evaluation_rng_digest", "string", True),
    ("teacher_value_status", "string", False),
    ("teacher_values_are_realized_match_ev", "bool", False),
)

RootGenerator = Callable[
    [Mapping[str, Any], ModelBundle | None],
    Sequence[ActorObservation],
]
SearchAdapter = Callable[
    [
        ActorObservation,
        Mapping[str, Any],
        Mapping[str, int],
        str,
        Path | None,
    ],
    Mapping[str, Any],
]
BaselineAdapter = Callable[
    [ActorObservation, Mapping[str, Any], ModelBundle | None],
    ActionKey | str | Any,
]


def canonical_bytes(value: Any) -> bytes:
    return contract.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return contract.canonical_sha256(value)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} fields changed: "
            f"missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.casefold()


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


def _write_once(path: str | Path, value: Any) -> Path:
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
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {target}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _profile_sha256() -> str:
    observed = _file_sha256(_AI_PROFILES_PATH)
    if observed != contract.CURRENT_PROFILE_REGISTRY_SHA256:
        raise PermissionError("ai_profiles.py/current profile registry changed")
    return observed


def _read_quality_gate(path: str | Path) -> dict[str, Any]:
    """Read the one cross-contract artifact without weakening either schema.

    Fresh-quality artifacts use canonical JSON without a trailing LF, while
    dataset artifacts use exactly one trailing LF.  Historical dataset tests
    and already staged controller copies may contain the latter.  Accept only
    those two byte-exact forms, then let the fresh-quality validator perform
    the mandatory source replay.
    """

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("qualified fresh-quality gate is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "qualified fresh-quality gate is not canonical JSON"
        ) from exc
    quality_raw = quality_gate.canonical_bytes(value)
    if (
        not isinstance(value, dict)
        or raw not in (quality_raw, quality_raw + b"\n")
    ):
        raise ValueError(
            "qualified fresh-quality gate is not canonical JSON"
        )
    return value


def _gate_value(path: str | Path) -> tuple[dict[str, Any], str]:
    target = Path(path)
    gate = _read_quality_gate(target)
    validated = quality_gate.validate_fresh_quality_gate_value(
        gate, replay_sources=True
    )
    if (
        validated.get("schema") != quality_gate.GATE_SCHEMA
        or validated.get("status") != "pass"
        or validated.get("decision")
        != "fresh_quality_pass_open_25_paired_data_shard_only"
        or validated.get("all_gates_passed") is not True
        or validated.get("quality_pilot_passed") is not True
        or validated.get("data_pilot_25_paired_authorized") is not True
        or validated.get("full_9000_paired_fanout_authorized") is not False
        or validated.get("training_eligible") is not False
        or validated.get("promotion_evidence") is not False
        or validated.get("current_profile_changed") is not False
        or validated.get("teacher_values_are_realized_match_ev") is not False
        or not _is_sha256(validated.get("merge_sha256"))
    ):
        raise PermissionError(
            "fresh-quality gate does not authorize the 25-paired data pilot"
        )
    return validated, _file_sha256(target)


def _authorization_value(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    fresh_quality_gate_path: str | Path,
) -> dict[str, Any]:
    validated_plan = contract.validate_dataset_plan(plan)
    if shard_id != contract.SMOKE_SHARD_ID:
        raise PermissionError(
            "dataset executor v1 authorizes only the first 25-paired smoke shard"
        )
    gate, gate_file_sha256 = _gate_value(fresh_quality_gate_path)
    profile_sha256 = _profile_sha256()
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "qualified_fresh_quality_bound_to_smoke_shard",
        "plan_sha256": canonical_sha256(validated_plan),
        "shard_id": shard_id,
        "fresh_quality_gate_schema": gate["schema"],
        "fresh_quality_gate_sha256": gate_file_sha256,
        "fresh_quality_merge_sha256": gate["merge_sha256"],
        "fresh_quality_decision": gate["decision"],
        "source_replayed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
        "current_profile_registry_sha256": profile_sha256,
        "current_profile_changed": False,
    }


def bind_fresh_quality_authorization(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
) -> dict[str, Any]:
    """Replay and bind the sole receipt that may open the smoke shard."""

    expected = _authorization_value(
        plan=plan,
        shard_id=shard_id,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    path = Path(shard_directory) / AUTHORIZATION_NAME
    if path.exists():
        stored = _read_canonical(path, "M3.1 dataset authorization")
        _exact_keys(stored, _AUTHORIZATION_KEYS, "M3.1 dataset authorization")
        if stored != expected:
            raise PermissionError("M3.1 dataset authorization source changed")
        return stored
    _write_once(path, expected)
    stored = _read_canonical(path, "stored M3.1 dataset authorization")
    _exact_keys(stored, _AUTHORIZATION_KEYS, "M3.1 dataset authorization")
    if stored != expected:
        raise ValueError("stored M3.1 dataset authorization changed")
    return stored


def _existing_smoke_authorization(
    *,
    plan: Mapping[str, Any],
    smoke_shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
) -> dict[str, Any]:
    """Read-only replay of the authorization published before smoke compute."""

    expected = _authorization_value(
        plan=plan,
        shard_id=contract.SMOKE_SHARD_ID,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    path = Path(smoke_shard_directory) / AUTHORIZATION_NAME
    if not path.is_file() or path.is_symlink():
        raise PermissionError(
            "full fanout requires the original safe smoke authorization"
        )
    stored = _read_canonical(path, "M3.1 smoke dataset authorization")
    _exact_keys(stored, _AUTHORIZATION_KEYS, "M3.1 smoke authorization")
    if stored != expected:
        raise PermissionError("M3.1 smoke authorization source changed")
    return stored


def _full_authorization_value(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    fresh_quality_gate_path: str | Path,
    smoke_gate_receipt_path: str | Path,
    smoke_shard_directory: str | Path,
) -> dict[str, Any]:
    validated_plan = contract.validate_dataset_plan(plan)
    if shard_id == contract.SMOKE_SHARD_ID:
        raise PermissionError("the smoke shard uses its fresh-quality authorization")
    gate, gate_file_sha256 = _gate_value(fresh_quality_gate_path)
    _existing_smoke_authorization(
        plan=validated_plan,
        smoke_shard_directory=smoke_shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    receipt_path = Path(smoke_gate_receipt_path)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise PermissionError("full fanout requires a safe smoke-gate receipt")
    smoke_receipt = contract.validate_smoke_gate_receipt(
        _read_canonical(receipt_path, "M3.1 dataset smoke gate"),
        plan=validated_plan,
        smoke_shard_directory=smoke_shard_directory,
    )
    shard_authorization = contract.validate_shard_start_authorization(
        plan=validated_plan,
        shard_id=shard_id,
        smoke_gate_receipt=smoke_receipt,
        smoke_shard_directory=smoke_shard_directory,
    )
    if (
        smoke_receipt["status"] != "pass"
        or smoke_receipt["decision"] != "open_remaining_8975_paired_fanout"
        or smoke_receipt["all_gates_passed"] is not True
        or smoke_receipt["full_9000_paired_fanout_authorized"] is not True
        or shard_authorization["authorized_by_dataset_smoke_gate"] is not True
    ):
        raise PermissionError("M3.1 smoke gate did not authorize full fanout")
    return {
        "schema": FULL_AUTHORIZATION_SCHEMA,
        "status": "qualified_smoke_gate_bound_to_full_dataset_shard",
        "plan_sha256": canonical_sha256(validated_plan),
        "shard_id": shard_id,
        "fresh_quality_gate_schema": gate["schema"],
        "fresh_quality_gate_sha256": gate_file_sha256,
        "fresh_quality_merge_sha256": gate["merge_sha256"],
        "fresh_quality_decision": gate["decision"],
        "dataset_smoke_gate_schema": smoke_receipt["schema"],
        "dataset_smoke_gate_sha256": _file_sha256(receipt_path),
        "dataset_smoke_gate_decision": smoke_receipt["decision"],
        "smoke_shard_id": contract.SMOKE_SHARD_ID,
        "smoke_shard_done_sha256": smoke_receipt[
            "smoke_shard_done_sha256"
        ],
        "source_replayed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": True,
        "current_profile_registry_sha256": _profile_sha256(),
        "current_profile_changed": False,
    }


def bind_full_fanout_authorization(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_gate_receipt_path: str | Path,
    smoke_shard_directory: str | Path,
) -> dict[str, Any]:
    """Create once the source-replayed authorization for one non-smoke shard."""

    expected = _full_authorization_value(
        plan=plan,
        shard_id=shard_id,
        fresh_quality_gate_path=fresh_quality_gate_path,
        smoke_gate_receipt_path=smoke_gate_receipt_path,
        smoke_shard_directory=smoke_shard_directory,
    )
    path = Path(shard_directory) / AUTHORIZATION_NAME
    if path.exists():
        stored = _read_canonical(path, "M3.1 full-shard authorization")
        _exact_keys(
            stored, _FULL_AUTHORIZATION_KEYS, "M3.1 full-shard authorization"
        )
        if stored != expected:
            raise PermissionError("M3.1 full-shard authorization source changed")
        return stored
    _write_once(path, expected)
    stored = _read_canonical(path, "stored M3.1 full-shard authorization")
    _exact_keys(
        stored, _FULL_AUTHORIZATION_KEYS, "M3.1 full-shard authorization"
    )
    if stored != expected:
        raise ValueError("stored M3.1 full-shard authorization changed")
    return stored


def _default_root_generator(
    pair: Mapping[str, Any], bundle: ModelBundle | None
) -> Sequence[ActorObservation]:
    if bundle is None:
        raise ValueError("default dataset root generator requires a model bundle")
    return generate_behavior_t3_roots(
        hand_seed=int(pair["seeds"]["hand"]),
        behavior_seed=int(pair["seeds"]["behavior"]),
        profile=str(pair["profile"]),
        bundle=bundle,
    )


def _default_search_adapter(
    observation: ActorObservation,
    pair: Mapping[str, Any],
    budget: Mapping[str, int],
    evaluation_seed_key: str,
    library_path: Path | None,
) -> Mapping[str, Any]:
    if library_path is None:
        raise ValueError("default dataset search requires a Candidate02 library")
    seeds = pair["seeds"]
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
            library_path=library_path,
            run_id=STEP6C_RUN_ID,
            candidate_samples=int(budget["candidate_samples"]),
            evaluation_samples=int(budget["evaluation_samples"]),
            downstream_t3_samples=int(budget["downstream_t3_samples"]),
            seed=int(seeds["child"]),
            candidate_seed=int(seeds["candidate"]),
            evaluation_seed=int(seeds[evaluation_seed_key]),
        )
    )
    return solver.solve(observation).to_dict()


def _default_baseline_adapter(
    observation: ActorObservation,
    pair: Mapping[str, Any],
    bundle: ModelBundle | None,
) -> Any:
    if bundle is None:
        raise ValueError("default baseline adapter requires a model bundle")
    seat_offset = 0 if observation.seat == "first" else 1
    policy = build_policy(
        "stage7_m5_r10",
        bundle,
        seed=int(pair["seeds"]["behavior"]) + 20_000 + seat_offset,
        seat=observation.seat,
        opening_lookahead_samples=0,
    )
    return policy.choose_action_observation(
        observation,
        hand_id=int(pair["seeds"]["hand"]),
        game_id=int(pair["seeds"]["hand"]),
        decision_seed=int(pair["seeds"]["behavior"]) + 30_000 + seat_offset,
    )


def _baseline_token(value: Any) -> str:
    if isinstance(value, ActionKey):
        return value.to_token()
    if isinstance(value, str):
        return ActionKey.from_token(value).to_token()
    return action_key(value).to_token()


def evidence_artifact_path(
    shard_directory: str | Path, local_pair_index: int
) -> Path:
    if (
        isinstance(local_pair_index, bool)
        or not isinstance(local_pair_index, int)
        or local_pair_index < 0
    ):
        raise ValueError("local_pair_index must be a nonnegative integer")
    return (
        Path(shard_directory)
        / EVIDENCE_DIRECTORY_NAME
        / f"{_EVIDENCE_FILE_PREFIX}{local_pair_index:06d}{_EVIDENCE_FILE_SUFFIX}"
    )


def _validate_observations(
    values: Sequence[ActorObservation],
    *,
    pair: Mapping[str, Any],
) -> tuple[ActorObservation, ActorObservation]:
    if len(values) != 2:
        raise ValueError("dataset root generator must return exactly two observations")
    normalized: list[ActorObservation] = []
    for value, seat in zip(values, ("first", "second"), strict=True):
        if (
            not isinstance(value, ActorObservation)
            or value.seat != seat
            or value.to_act_order != seat
            or value.street != "T3"
        ):
            raise ValueError("dataset root generator returned an invalid T3 observation")
        # ActorObservation construction already enforces the public information
        # schema and visible-card consistency.
        normalized.append(ActorObservation.from_dict(value.to_dict()))
    if normalized[0].fingerprint() == normalized[1].fingerprint():
        raise ValueError("paired T3 roots unexpectedly share one information set")
    return normalized[0], normalized[1]


def _build_pair_evidence(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    pair: Mapping[str, Any],
    observations: Sequence[ActorObservation],
    search_adapter: SearchAdapter,
    baseline_adapter: BaselineAdapter,
    bundle: ModelBundle | None,
    library_path: Path | None,
) -> dict[str, Any]:
    validated_observations = _validate_observations(observations, pair=pair)
    rows: list[dict[str, Any]] = []
    for root_index, seat, observation in zip(
        pair["root_indices"],
        ("first", "second"),
        validated_observations,
        strict=True,
    ):
        baseline = _baseline_token(baseline_adapter(observation, pair, bundle))
        primary = deepcopy(
            dict(
                search_adapter(
                    observation,
                    pair,
                    contract.PRIMARY_BUDGET,
                    "evaluation",
                    library_path,
                )
            )
        )
        confirmation = None
        if pair["confirmation_required"]:
            confirmation = deepcopy(
                dict(
                    search_adapter(
                        observation,
                        pair,
                        contract.CONFIRMATION_BUDGET,
                        "confirmation",
                        library_path,
                    )
                )
            )
        payload = observation.to_dict()
        rows.append(
            {
                "root_index": root_index,
                "seat": seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation_sha256": canonical_sha256(payload),
                "observation": payload,
                "baseline_action_key": baseline,
                "primary_decision": primary,
                "confirmation_decision": confirmation,
            }
        )
    return {
        "schema": PAIR_EVIDENCE_SCHEMA,
        "status": "complete_create_only_candidate02_search_evidence",
        "plan_sha256": canonical_sha256(plan),
        "authorization_sha256": canonical_sha256(authorization),
        "candidate_library_sha256": contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
        "pair_contract": deepcopy(dict(pair)),
        "pair_contract_sha256": canonical_sha256(pair),
        "shard_id": pair["shard_id"],
        "split": pair["split"],
        "local_pair_index": pair["local_pair_index"],
        "global_pair_index": pair["global_pair_index"],
        "root_indices": pair["root_indices"],
        "profile": pair["profile"],
        "seeds": pair["seeds"],
        "confirmation_required": pair["confirmation_required"],
        "rows": rows,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def _canonical_legal_contract(
    observation: ActorObservation,
) -> tuple[list[str], str, str]:
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    return (
        [action_key(action).to_token() for action in actions],
        legal_action_set_digest(actions),
        ordered_action_mapping_digest(actions),
    )


def _rank_by_q(
    tokens: Sequence[str], values: Mapping[str, float]
) -> tuple[dict[str, int], str]:
    ordered = sorted(
        tokens,
        key=lambda token: (-float(values[token]), ActionKey.from_token(token).sort_key()),
    )
    return {token: rank for rank, token in enumerate(ordered)}, ordered[0]


def _pair_result_from_validated_evidence(
    *,
    plan: Mapping[str, Any],
    pair: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in evidence["rows"]:
        observation = ActorObservation.from_dict(raw["observation"])
        legal_tokens, legal_set_digest, legal_order_digest = (
            _canonical_legal_contract(observation)
        )
        primary = raw["primary_decision"]
        primary_rows = {
            str(item["action_key"]): item for item in primary["action_values"]
        }
        primary_q = {
            token: float(primary_rows[token]["evaluation_ev"])
            for token in legal_tokens
        }
        primary_rank, teacher_selected = _rank_by_q(legal_tokens, primary_q)
        baseline = str(raw["baseline_action_key"])
        baseline_q = primary_q[baseline]

        confirmation = raw["confirmation_decision"]
        confirmation_q: dict[str, float] | None = None
        confirmation_rank: dict[str, int] | None = None
        confirmation_state_value = None
        if confirmation is not None:
            confirmation_rows = {
                str(item["action_key"]): item
                for item in confirmation["action_values"]
            }
            confirmation_q = {
                token: float(confirmation_rows[token]["evaluation_ev"])
                for token in legal_tokens
            }
            confirmation_rank, confirmation_best = _rank_by_q(
                legal_tokens, confirmation_q
            )
            confirmation_state_value = confirmation_q[confirmation_best]
            confirmation_baseline = confirmation_q[baseline]
        else:
            confirmation_baseline = None

        targets: list[dict[str, Any]] = []
        for index, token in enumerate(legal_tokens):
            targets.append(
                {
                    "action_index": index,
                    "action_key": token,
                    "primary_q": primary_q[token],
                    "primary_delta": primary_q[token] - baseline_q,
                    "primary_rank": primary_rank[token],
                    "confirmation_q": (
                        confirmation_q[token]
                        if confirmation_q is not None
                        else None
                    ),
                    "confirmation_delta": (
                        confirmation_q[token] - float(confirmation_baseline)
                        if confirmation_q is not None
                        else None
                    ),
                    "confirmation_rank": (
                        confirmation_rank[token]
                        if confirmation_rank is not None
                        else None
                    ),
                }
            )
        rows.append(
            {
                "schema": contract.SEAT_ROW_SCHEMA,
                "root_index": raw["root_index"],
                "seat": raw["seat"],
                "observation_fingerprint": raw["observation_fingerprint"],
                "observation_sha256": raw["observation_sha256"],
                "observation": raw["observation"],
                "action_key_schema": ACTION_KEY_SCHEMA,
                "legal_action_keys": legal_tokens,
                "legal_action_set_digest": legal_set_digest,
                "legal_action_order_digest": legal_order_digest,
                "teacher": {
                    "schema": contract.TEACHER_LABEL_SCHEMA,
                    "primary_budget": dict(contract.PRIMARY_BUDGET),
                    "confirmation_budget": (
                        dict(contract.CONFIRMATION_BUDGET)
                        if pair["confirmation_required"]
                        else None
                    ),
                    "baseline_action_key": baseline,
                    "selected_action_key": teacher_selected,
                    "state_value": primary_q[teacher_selected],
                    "confirmation_state_value": confirmation_state_value,
                    "teacher_value_status": (
                        "search_estimate_not_realized_match_ev"
                    ),
                    "teacher_values_are_realized_match_ev": False,
                    "action_targets": targets,
                },
            }
        )
    value = {
        "schema": contract.PAIR_RESULT_SCHEMA,
        "plan_sha256": canonical_sha256(plan),
        "pair_contract_sha256": canonical_sha256(pair),
        "shard_id": pair["shard_id"],
        "split": pair["split"],
        "local_pair_index": pair["local_pair_index"],
        "global_pair_index": pair["global_pair_index"],
        "root_indices": pair["root_indices"],
        "profile": pair["profile"],
        "seeds": pair["seeds"],
        "confirmation_required": pair["confirmation_required"],
        "rows": rows,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    return contract.validate_pair_result(
        value,
        plan=plan,
        expected_split=str(pair["split"]),
        expected_local_pair_index=int(pair["local_pair_index"]),
    )


def validate_pair_evidence(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    expected_local_pair_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay complete decisions and return evidence plus derived pair JSON."""

    validated_plan = contract.validate_dataset_plan(plan)
    evidence = deepcopy(dict(value))
    _exact_keys(evidence, _PAIR_EVIDENCE_KEYS, "M3.1 dataset pair evidence")
    raw_pair = evidence.get("pair_contract")
    if not isinstance(raw_pair, Mapping):
        raise ValueError("M3.1 evidence pair contract is missing")
    local_index = evidence.get("local_pair_index")
    if (
        isinstance(local_index, bool)
        or not isinstance(local_index, int)
        or local_index < 0
    ):
        raise ValueError("M3.1 evidence local pair index changed")
    if (
        expected_local_pair_index is not None
        and local_index != expected_local_pair_index
    ):
        raise ValueError("M3.1 evidence local pair index changed")
    expected_pair = contract.pair_contract(
        validated_plan, str(evidence.get("split")), local_index
    )
    rows = evidence.get("rows")
    if (
        evidence["schema"] != PAIR_EVIDENCE_SCHEMA
        or evidence["status"]
        != "complete_create_only_candidate02_search_evidence"
        or evidence["plan_sha256"] != canonical_sha256(validated_plan)
        or evidence["authorization_sha256"] != canonical_sha256(authorization)
        or evidence["candidate_library_sha256"]
        != contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or dict(raw_pair) != expected_pair
        or evidence["pair_contract_sha256"] != canonical_sha256(expected_pair)
        or evidence["shard_id"] != expected_pair["shard_id"]
        or evidence["global_pair_index"] != expected_pair["global_pair_index"]
        or evidence["root_indices"] != expected_pair["root_indices"]
        or evidence["profile"] != expected_pair["profile"]
        or evidence["seeds"] != expected_pair["seeds"]
        or evidence["confirmation_required"]
        != expected_pair["confirmation_required"]
        or not isinstance(rows, list)
        or len(rows) != 2
        or evidence["opponent_private_discards_used"] is not False
        or evidence["realized_deck_tail_used"] is not False
        or evidence["teacher_values_are_realized_match_ev"] is not False
        or evidence["current_profile_changed"] is not False
    ):
        raise ValueError("M3.1 dataset pair evidence provenance changed")

    for position, (raw, seat, root_index) in enumerate(
        zip(rows, ("first", "second"), expected_pair["root_indices"], strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("M3.1 evidence row is missing")
        row = dict(raw)
        _exact_keys(row, _EVIDENCE_ROW_KEYS, "M3.1 dataset evidence row")
        observation_raw = row.get("observation")
        if not isinstance(observation_raw, Mapping):
            raise ValueError("M3.1 evidence observation is missing")
        observation = ActorObservation.from_dict(observation_raw)
        legal_tokens, _, _ = _canonical_legal_contract(observation)
        baseline = ActionKey.from_token(str(row["baseline_action_key"])).to_token()
        if (
            row["root_index"] != root_index
            or row["seat"] != seat
            or observation.seat != seat
            or observation.to_act_order != seat
            or observation.street != "T3"
            or observation.to_dict() != dict(observation_raw)
            or row["observation_fingerprint"] != observation.fingerprint()
            or row["observation_sha256"]
            != canonical_sha256(observation.to_dict())
            or baseline not in legal_tokens
        ):
            raise ValueError(f"M3.1 evidence observation changed at row {position}")

        primary = row.get("primary_decision")
        primary_evidence = step6c._validate_decision_payload(
            primary,
            observation=observation,
            seeds=expected_pair["seeds"],
            budget=step6c._PRIMARY_BUDGET,
            evaluation_seed_key="evaluation",
            native_library_sha256=contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
        )
        confirmation = row.get("confirmation_decision")
        confirmation_evidence = None
        if expected_pair["confirmation_required"]:
            if not isinstance(confirmation, Mapping):
                raise ValueError("M3.1 confirmation evidence is missing")
            confirmation_evidence = step6c._validate_decision_payload(
                confirmation,
                observation=observation,
                seeds=expected_pair["seeds"],
                budget=step6c._CONFIRMATION_BUDGET,
                evaluation_seed_key="confirmation",
                native_library_sha256=contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
            )
            if (
                primary_evidence.candidate_keys
                != confirmation_evidence.candidate_keys
                or primary_evidence.selection_by_key
                != confirmation_evidence.selection_by_key
            ):
                raise ValueError(
                    "M3.1 confirmation changed the locked candidate pass"
                )
        elif confirmation is not None:
            raise ValueError("nonconfirmation evidence contains a confirmation")

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
            raise ValueError("M3.1 dataset search RNG namespaces overlap")
    contract._reject_hidden(evidence, "dataset_pair_search_evidence")
    pair_result = _pair_result_from_validated_evidence(
        plan=validated_plan,
        pair=expected_pair,
        evidence=evidence,
    )
    return evidence, pair_result


def _evidence_indices(
    *, shard_directory: Path, expected_indices: set[int]
) -> dict[int, Path]:
    directory = shard_directory / EVIDENCE_DIRECTORY_NAME
    observed: dict[int, Path] = {}
    if not directory.exists():
        return observed
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("M3.1 evidence directory is unsafe")
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("M3.1 evidence artifact is unsafe")
        name = path.name
        if not (
            name.startswith(_EVIDENCE_FILE_PREFIX)
            and name.endswith(_EVIDENCE_FILE_SUFFIX)
            and name[
                len(_EVIDENCE_FILE_PREFIX) : -len(_EVIDENCE_FILE_SUFFIX)
            ].isdigit()
        ):
            raise ValueError(f"unknown M3.1 evidence artifact: {name}")
        index = int(
            name[len(_EVIDENCE_FILE_PREFIX) : -len(_EVIDENCE_FILE_SUFFIX)]
        )
        if index not in expected_indices or index in observed:
            raise ValueError("M3.1 shard contains extra/duplicate search evidence")
        observed[index] = path
    return observed


def _load_and_validate_evidence(
    *,
    path: Path,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    local_pair_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return validate_pair_evidence(
        _read_canonical(path, "M3.1 dataset pair search evidence"),
        plan=plan,
        authorization=authorization,
        expected_local_pair_index=local_pair_index,
    )


def _default_bundle() -> ModelBundle:
    profiles = set(M31_T3_BEHAVIOR_PROFILES) | {"stage7_m5_r10"}
    return load_model_bundle(ModelPaths(), profiles=profiles)


def _run_authorized_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
    authorization: Mapping[str, Any],
    library_path: str | Path | None = None,
    root_generator: RootGenerator | None = None,
    search_adapter: SearchAdapter | None = None,
    baseline_adapter: BaselineAdapter | None = None,
    bundle: ModelBundle | None = None,
    max_new_pairs: int | None = None,
) -> dict[str, Any]:
    """Run or safely resume one already-authorized 25-paired shard.

    ``max_new_pairs`` is a correctness/recovery hook.  Production callers leave
    it unset; tests and bounded pilots may stop after a fixed number of newly
    materialized pair files.
    """

    validated_plan = contract.validate_dataset_plan(plan)
    shard = next(
        (
            row
            for row in validated_plan["shards"]
            if row["shard_id"] == shard_id
        ),
        None,
    )
    if shard is None:
        raise KeyError(f"unknown M3.1 dataset shard: {shard_id}")
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    authorization_path = directory / AUTHORIZATION_NAME
    if not authorization_path.is_file() or authorization_path.is_symlink():
        raise PermissionError("M3.1 shard authorization was not persisted")
    stored_authorization = _read_canonical(
        authorization_path, "M3.1 persisted shard authorization"
    )
    if stored_authorization != dict(authorization):
        raise PermissionError("M3.1 persisted shard authorization changed")
    if (
        isinstance(max_new_pairs, bool)
        or (
            max_new_pairs is not None
            and (not isinstance(max_new_pairs, int) or max_new_pairs < 0)
        )
    ):
        raise ValueError("max_new_pairs must be a nonnegative integer or None")
    selected_root_generator = root_generator or _default_root_generator
    selected_search_adapter = search_adapter or _default_search_adapter
    selected_baseline_adapter = baseline_adapter or _default_baseline_adapter
    native_path = Path(library_path).resolve() if library_path is not None else None
    if search_adapter is None and (
        native_path is None
        or native_path.is_symlink()
        or not native_path.is_file()
        or _file_sha256(native_path)
        != contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256
    ):
        raise ValueError(
            "default dataset search requires the accepted Candidate02 library"
        )
    if bundle is None and (
        root_generator is None or baseline_adapter is None
    ):
        bundle = _default_bundle()

    expected_indices = set(int(value) for value in shard["local_pair_indices"])
    evidence_paths = _evidence_indices(
        shard_directory=directory, expected_indices=expected_indices
    )
    resume = contract.inspect_shard_resume(
        plan=validated_plan,
        shard_id=shard_id,
        shard_directory=directory,
    )
    completed = set(int(value) for value in resume["completed_pair_indices"])
    if not completed.issubset(evidence_paths):
        raise ValueError("M3.1 pair artifact exists without search evidence")

    # Every retained decision is replayed before any new work or DONE trust.
    for index in sorted(evidence_paths):
        _, derived = _load_and_validate_evidence(
            path=evidence_paths[index],
            plan=validated_plan,
            authorization=authorization,
            local_pair_index=index,
        )
        pair_path = contract.pair_artifact_path(directory, index)
        if pair_path.exists():
            stored_pair = _read_canonical(pair_path, "M3.1 dataset pair")
            if stored_pair != derived:
                raise ValueError(
                    "M3.1 dataset pair differs from replayed search evidence"
                )

    if resume["already_complete"]:
        if set(evidence_paths) != expected_indices:
            raise ValueError("completed M3.1 shard lacks complete search evidence")
        done = contract.validate_completed_shard(
            plan=validated_plan,
            shard_id=shard_id,
            shard_directory=directory,
        )
        _profile_sha256()
        return {
            "status": "already_complete",
            "shard_id": shard_id,
            "new_pair_count": 0,
            "resume": resume,
            "done": done,
            "authorization_sha256": canonical_sha256(authorization),
        }

    new_pair_count = 0
    for index in resume["pending_pair_indices"]:
        if max_new_pairs is not None and new_pair_count >= max_new_pairs:
            break
        pair = contract.pair_contract(
            validated_plan, str(shard["split"]), int(index)
        )
        evidence_path = evidence_artifact_path(directory, int(index))
        if evidence_path.exists():
            evidence, pair_result = _load_and_validate_evidence(
                path=evidence_path,
                plan=validated_plan,
                authorization=authorization,
                local_pair_index=int(index),
            )
        else:
            observations = selected_root_generator(pair, bundle)
            candidate = _build_pair_evidence(
                plan=validated_plan,
                authorization=authorization,
                pair=pair,
                observations=observations,
                search_adapter=selected_search_adapter,
                baseline_adapter=selected_baseline_adapter,
                bundle=bundle,
                library_path=native_path,
            )
            evidence, pair_result = validate_pair_evidence(
                candidate,
                plan=validated_plan,
                authorization=authorization,
                expected_local_pair_index=int(index),
            )
            _write_once(evidence_path, evidence)
            stored_evidence, stored_result = _load_and_validate_evidence(
                path=evidence_path,
                plan=validated_plan,
                authorization=authorization,
                local_pair_index=int(index),
            )
            if stored_evidence != evidence or stored_result != pair_result:
                raise ValueError("stored M3.1 search evidence changed")
        contract.write_pair_result(
            plan=validated_plan,
            shard_id=shard_id,
            shard_directory=directory,
            local_pair_index=int(index),
            value=pair_result,
        )
        new_pair_count += 1

    after = contract.inspect_shard_resume(
        plan=validated_plan,
        shard_id=shard_id,
        shard_directory=directory,
    )
    if after["pending_pair_count"]:
        _profile_sha256()
        return {
            "status": "partial_safe_to_resume",
            "shard_id": shard_id,
            "new_pair_count": new_pair_count,
            "resume": after,
            "done": None,
            "authorization_sha256": canonical_sha256(authorization),
        }

    evidence_paths = _evidence_indices(
        shard_directory=directory, expected_indices=expected_indices
    )
    if set(evidence_paths) != expected_indices:
        raise ValueError("M3.1 shard cannot finalize without all search evidence")
    for index in sorted(evidence_paths):
        _, derived = _load_and_validate_evidence(
            path=evidence_paths[index],
            plan=validated_plan,
            authorization=authorization,
            local_pair_index=index,
        )
        if _read_canonical(
            contract.pair_artifact_path(directory, index),
            "M3.1 dataset pair",
        ) != derived:
            raise ValueError("M3.1 pair/evidence replay changed before DONE")

    # No artifact in the shard directory is written after this call.
    done = contract.finalize_shard(
        plan=validated_plan,
        shard_id=shard_id,
        shard_directory=directory,
    )
    final_resume = contract.inspect_shard_resume(
        plan=validated_plan,
        shard_id=shard_id,
        shard_directory=directory,
    )
    if not final_resume["already_complete"]:
        raise ValueError("M3.1 SHARD_DONE did not publish last")
    _profile_sha256()
    return {
        "status": "complete_done_published_last",
        "shard_id": shard_id,
        "new_pair_count": new_pair_count,
        "resume": final_resume,
        "done": done,
        "authorization_sha256": canonical_sha256(authorization),
        "authorization_file_sha256": _file_sha256(authorization_path),
    }


def run_smoke_shard(
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
    library_path: str | Path | None = None,
    root_generator: RootGenerator | None = None,
    search_adapter: SearchAdapter | None = None,
    baseline_adapter: BaselineAdapter | None = None,
    bundle: ModelBundle | None = None,
    max_new_pairs: int | None = None,
) -> dict[str, Any]:
    """Run the first 25-paired correctness shard after fresh-quality pass."""

    validated_plan = contract.validate_dataset_plan(plan)
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    authorization = bind_fresh_quality_authorization(
        plan=validated_plan,
        shard_id=contract.SMOKE_SHARD_ID,
        shard_directory=directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    return _run_authorized_shard(
        plan=validated_plan,
        shard_id=contract.SMOKE_SHARD_ID,
        shard_directory=directory,
        authorization=authorization,
        library_path=library_path,
        root_generator=root_generator,
        search_adapter=search_adapter,
        baseline_adapter=baseline_adapter,
        bundle=bundle,
        max_new_pairs=max_new_pairs,
    )


def run_dataset_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_gate_receipt_path: str | Path,
    smoke_shard_directory: str | Path,
    library_path: str | Path | None = None,
    root_generator: RootGenerator | None = None,
    search_adapter: SearchAdapter | None = None,
    baseline_adapter: BaselineAdapter | None = None,
    bundle: ModelBundle | None = None,
    max_new_pairs: int | None = None,
) -> dict[str, Any]:
    """Run one non-smoke shard after replaying both prerequisite gates."""

    validated_plan = contract.validate_dataset_plan(plan)
    if shard_id == contract.SMOKE_SHARD_ID:
        raise ValueError("run_dataset_shard is for post-smoke shards only")
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    authorization = bind_full_fanout_authorization(
        plan=validated_plan,
        shard_id=shard_id,
        shard_directory=directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
        smoke_gate_receipt_path=smoke_gate_receipt_path,
        smoke_shard_directory=smoke_shard_directory,
    )
    return _run_authorized_shard(
        plan=validated_plan,
        shard_id=shard_id,
        shard_directory=directory,
        authorization=authorization,
        library_path=library_path,
        root_generator=root_generator,
        search_adapter=search_adapter,
        baseline_adapter=baseline_adapter,
        bundle=bundle,
        max_new_pairs=max_new_pairs,
    )


def _evidence_record(
    path: Path,
    evidence: Mapping[str, Any],
    *,
    shard_directory: Path,
) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "split": evidence["split"],
        "local_pair_index": evidence["local_pair_index"],
        "global_pair_index": evidence["global_pair_index"],
        "shard_id": evidence["shard_id"],
        "path": path.relative_to(shard_directory).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _load_smoke_sources(
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    validated_plan = contract.validate_dataset_plan(plan)
    directory = Path(shard_directory)
    authorization = bind_fresh_quality_authorization(
        plan=validated_plan,
        shard_id=contract.SMOKE_SHARD_ID,
        shard_directory=directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    done = contract.validate_completed_shard(
        plan=validated_plan,
        shard_id=contract.SMOKE_SHARD_ID,
        shard_directory=directory,
    )
    evidence_paths = _evidence_indices(
        shard_directory=directory,
        expected_indices=set(range(contract.SHARD_PAIR_COUNT)),
    )
    if set(evidence_paths) != set(range(contract.SHARD_PAIR_COUNT)):
        raise ValueError("M3.1 smoke merge lacks complete search evidence")
    evidence_values: list[dict[str, Any]] = []
    pair_values: list[dict[str, Any]] = []
    for index in range(contract.SHARD_PAIR_COUNT):
        evidence, derived = _load_and_validate_evidence(
            path=evidence_paths[index],
            plan=validated_plan,
            authorization=authorization,
            local_pair_index=index,
        )
        pair_path = contract.pair_artifact_path(directory, index)
        stored = _read_canonical(pair_path, "M3.1 dataset pair")
        if stored != derived:
            raise ValueError("M3.1 smoke pair differs from decision replay")
        evidence_values.append(evidence)
        pair_values.append(derived)
    return validated_plan, authorization, done, evidence_values, pair_values


def _flatten_action_rows(
    *,
    pairs: Sequence[Mapping[str, Any]],
    evidence_values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(pairs) != len(evidence_values):
        raise ValueError("M3.1 pair/evidence flattening count changed")
    flattened: list[dict[str, Any]] = []
    for pair, evidence in zip(pairs, evidence_values, strict=True):
        evidence_by_root = {
            int(row["root_index"]): row for row in evidence["rows"]
        }
        seeds = pair["seeds"]
        for seat_row in pair["rows"]:
            root_index = int(seat_row["root_index"])
            raw = evidence_by_root[root_index]
            teacher = seat_row["teacher"]
            primary = raw["primary_decision"]
            confirmation = raw["confirmation_decision"]
            primary_by_key = {
                str(item["action_key"]): item
                for item in primary["action_values"]
            }
            observation_json = json.dumps(
                seat_row["observation"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            for target in teacher["action_targets"]:
                token = str(target["action_key"])
                primary_raw = primary_by_key[token]
                row = {
                    "split": pair["split"],
                    "shard_id": pair["shard_id"],
                    "local_pair_index": pair["local_pair_index"],
                    "global_pair_index": pair["global_pair_index"],
                    "profile": pair["profile"],
                    "root_index": root_index,
                    "seat": seat_row["seat"],
                    "confirmation_required": pair["confirmation_required"],
                    "hand_seed": seeds["hand"],
                    "behavior_seed": seeds["behavior"],
                    "candidate_seed": seeds["candidate"],
                    "evaluation_seed": seeds["evaluation"],
                    "child_seed": seeds["child"],
                    "confirmation_seed": seeds["confirmation"],
                    "observation_fingerprint": seat_row[
                        "observation_fingerprint"
                    ],
                    "observation_sha256": seat_row["observation_sha256"],
                    "observation_json": observation_json,
                    "legal_action_set_digest": seat_row[
                        "legal_action_set_digest"
                    ],
                    "legal_action_order_digest": seat_row[
                        "legal_action_order_digest"
                    ],
                    "action_index": target["action_index"],
                    "action_key": token,
                    "is_baseline": token == teacher["baseline_action_key"],
                    "is_teacher_selected": token
                    == teacher["selected_action_key"],
                    "search_selected_action_key": primary[
                        "selected_action_key"
                    ],
                    "primary_selection_q": primary_raw["selection_ev"],
                    "primary_q": target["primary_q"],
                    "primary_delta": target["primary_delta"],
                    "primary_rank": target["primary_rank"],
                    "state_value": teacher["state_value"],
                    "confirmation_q": target["confirmation_q"],
                    "confirmation_delta": target["confirmation_delta"],
                    "confirmation_rank": target["confirmation_rank"],
                    "confirmation_state_value": teacher[
                        "confirmation_state_value"
                    ],
                    "primary_search_contract_digest": primary[
                        "search_contract_digest"
                    ],
                    "primary_semantic_result_digest": primary[
                        "semantic_result_digest"
                    ],
                    "primary_result_digest": primary["result_digest"],
                    "primary_candidate_rng_digest": primary[
                        "candidate_rng_digest"
                    ],
                    "primary_evaluation_rng_digest": primary[
                        "evaluation_rng_digest"
                    ],
                    "confirmation_search_contract_digest": (
                        confirmation["search_contract_digest"]
                        if confirmation is not None
                        else None
                    ),
                    "confirmation_semantic_result_digest": (
                        confirmation["semantic_result_digest"]
                        if confirmation is not None
                        else None
                    ),
                    "confirmation_result_digest": (
                        confirmation["result_digest"]
                        if confirmation is not None
                        else None
                    ),
                    "confirmation_candidate_rng_digest": (
                        confirmation["candidate_rng_digest"]
                        if confirmation is not None
                        else None
                    ),
                    "confirmation_evaluation_rng_digest": (
                        confirmation["evaluation_rng_digest"]
                        if confirmation is not None
                        else None
                    ),
                    "teacher_value_status": teacher["teacher_value_status"],
                    "teacher_values_are_realized_match_ev": False,
                }
                if set(row) != {name for name, _, _ in PARQUET_COLUMNS}:
                    raise RuntimeError("M3.1 flattened Parquet schema changed")
                if any(
                    isinstance(value, float) and not math.isfinite(value)
                    for value in row.values()
                ):
                    raise ValueError("M3.1 flattened row contains non-finite data")
                flattened.append(row)
    flattened.sort(
        key=lambda row: (
            int(row["global_pair_index"]),
            int(row["root_index"]),
            int(row["action_index"]),
        )
    )
    return flattened


def build_smoke_merge(
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
) -> dict[str, Any]:
    """Purely replay the completed smoke shard into a merge manifest."""

    (
        validated_plan,
        authorization,
        done,
        evidence_values,
        pair_values,
    ) = _load_smoke_sources(
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    directory = Path(shard_directory)
    evidence_records = [
        _evidence_record(
            evidence_artifact_path(directory, index),
            evidence,
            shard_directory=directory,
        )
        for index, evidence in enumerate(evidence_values)
    ]
    flattened = _flatten_action_rows(
        pairs=pair_values, evidence_values=evidence_values
    )
    value = {
        "schema": SMOKE_MERGE_SCHEMA,
        "status": "complete_source_replayed_25_paired_smoke_merge",
        "plan_sha256": canonical_sha256(validated_plan),
        "shard_id": contract.SMOKE_SHARD_ID,
        "authorization_sha256": canonical_sha256(authorization),
        "fresh_quality_gate_sha256": authorization[
            "fresh_quality_gate_sha256"
        ],
        "shard_done_sha256": canonical_sha256(done),
        "pair_records": deepcopy(done["pair_records"]),
        "pair_record_aggregate_sha256": done[
            "pair_record_aggregate_sha256"
        ],
        "evidence_records": evidence_records,
        "evidence_record_aggregate_sha256": canonical_sha256(
            evidence_records
        ),
        "paired_hand_count": contract.SHARD_PAIR_COUNT,
        "root_count": contract.SHARD_PAIR_COUNT * 2,
        "seat_counts": {"first": contract.SHARD_PAIR_COUNT, "second": contract.SHARD_PAIR_COUNT},
        "confirmation_pair_count": sum(
            bool(pair["confirmation_required"]) for pair in pair_values
        ),
        "action_row_count": len(flattened),
        "action_row_aggregate_sha256": canonical_sha256(flattened),
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "action_key_mapping_mismatch_count": 0,
        "rng_overlap_count": 0,
        "missing_or_duplicate_count": 0,
        "data_pilot_25_paired_complete": True,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    contract._reject_hidden(value, "dataset_smoke_merge")
    return value


def validate_smoke_merge_value(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
) -> dict[str, Any]:
    merge = deepcopy(dict(value))
    _exact_keys(merge, _SMOKE_MERGE_KEYS, "M3.1 dataset smoke merge")
    for record in merge.get("evidence_records", []):
        if not isinstance(record, Mapping):
            raise ValueError("M3.1 smoke evidence record is missing")
        _exact_keys(
            record, _EVIDENCE_RECORD_KEYS, "M3.1 smoke evidence record"
        )
    expected = build_smoke_merge(
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    if merge != expected:
        raise ValueError("M3.1 smoke merge differs from full source replay")
    return merge


def write_smoke_merge(
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path)
    directory = Path(shard_directory).resolve()
    if output.resolve().is_relative_to(directory):
        raise ValueError(
            "smoke merge must stay outside the DONE-last shard directory"
        )
    merge = build_smoke_merge(
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    _write_once(output, merge)
    return validate_smoke_merge_value(
        _read_canonical(output, "stored M3.1 smoke merge"),
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )


def _column_schema_value() -> list[dict[str, Any]]:
    return [
        {"name": name, "type": kind, "nullable": nullable}
        for name, kind, nullable in PARQUET_COLUMNS
    ]


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Apache Parquet export requires the optional 'pyarrow' package"
        ) from exc
    return pa, pq


def _arrow_schema(pa: Any) -> Any:
    kinds = {
        "string": pa.string(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
        "float64": pa.float64(),
    }
    return pa.schema(
        [
            pa.field(name, kinds[kind], nullable=nullable)
            for name, kind, nullable in PARQUET_COLUMNS
        ],
        metadata={
            b"schema": b"hu_m31_t3_dataset_action_rows_v1",
            b"teacher_value_status": b"search_estimate_not_realized_match_ev",
        },
    )


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("M3.1 Parquet artifact is missing or unsafe")
    raw = path.read_bytes()
    if len(raw) < 8 or raw[:4] != b"PAR1" or raw[-4:] != b"PAR1":
        raise ValueError("M3.1 Parquet magic changed")
    pa, pq = _pyarrow_modules()
    table = pq.read_table(path)
    if table.schema != _arrow_schema(pa):
        raise ValueError("M3.1 Parquet physical schema changed")
    rows = table.to_pylist()
    if any(set(row) != {name for name, _, _ in PARQUET_COLUMNS} for row in rows):
        raise ValueError("M3.1 Parquet columns changed")
    return rows


def _write_parquet_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    pa, pq = _pyarrow_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows], schema=_arrow_schema(pa))
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="2.0",
        )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parquet_manifest_value(
    *,
    plan: Mapping[str, Any],
    merge: Mapping[str, Any],
    done: Mapping[str, Any],
    authorization: Mapping[str, Any],
    parquet_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PARQUET_MANIFEST_SCHEMA,
        "status": "complete_immutable_source_replayed_parquet",
        "plan_sha256": canonical_sha256(plan),
        "smoke_merge_sha256": canonical_sha256(merge),
        "shard_done_sha256": canonical_sha256(done),
        "authorization_sha256": canonical_sha256(authorization),
        "source_action_row_aggregate_sha256": canonical_sha256(list(rows)),
        "column_schema": _column_schema_value(),
        "parquet_file": PARQUET_FILE_NAME,
        "parquet_sha256": _file_sha256(parquet_path),
        "parquet_bytes": parquet_path.stat().st_size,
        "row_count": len(rows),
        "paired_hand_count": contract.SHARD_PAIR_COUNT,
        "root_count": contract.SHARD_PAIR_COUNT * 2,
        "immutable": True,
        "source_replayed": True,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def write_parquet_export(
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_merge_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write or validate one immutable, flattened smoke-shard Parquet file."""

    validated_plan = contract.validate_dataset_plan(plan)
    shard_resolved = Path(shard_directory).resolve()
    output = Path(output_directory)
    if output.resolve().is_relative_to(shard_resolved):
        raise ValueError(
            "Parquet export must stay outside the DONE-last shard directory"
        )
    merge = validate_smoke_merge_value(
        _read_canonical(smoke_merge_path, "M3.1 smoke merge"),
        plan=validated_plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    (
        _,
        authorization,
        done,
        evidence_values,
        pair_values,
    ) = _load_smoke_sources(
        plan=validated_plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    rows = _flatten_action_rows(
        pairs=pair_values, evidence_values=evidence_values
    )
    if canonical_sha256(rows) != merge["action_row_aggregate_sha256"]:
        raise ValueError("Parquet source rows differ from the pure smoke merge")

    parquet_path = output / PARQUET_FILE_NAME
    manifest_path = output / PARQUET_MANIFEST_NAME
    if not parquet_path.exists():
        _write_parquet_once(parquet_path, rows)
    stored_rows = _read_parquet_rows(parquet_path)
    if stored_rows != rows:
        raise ValueError("M3.1 Parquet rows differ from source replay")
    expected = _parquet_manifest_value(
        plan=validated_plan,
        merge=merge,
        done=done,
        authorization=authorization,
        parquet_path=parquet_path,
        rows=rows,
    )
    if manifest_path.exists():
        stored = _read_canonical(
            manifest_path, "M3.1 Parquet immutable manifest"
        )
        _exact_keys(
            stored, _PARQUET_MANIFEST_KEYS, "M3.1 Parquet immutable manifest"
        )
        if stored != expected:
            raise ValueError("M3.1 Parquet immutable manifest changed")
        return stored
    _write_once(manifest_path, expected)
    stored = _read_canonical(
        manifest_path, "stored M3.1 Parquet immutable manifest"
    )
    _exact_keys(
        stored, _PARQUET_MANIFEST_KEYS, "M3.1 Parquet immutable manifest"
    )
    if stored != expected:
        raise ValueError("stored M3.1 Parquet immutable manifest changed")
    _profile_sha256()
    return stored


def validate_parquet_export(
    *,
    plan: Mapping[str, Any],
    shard_directory: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_merge_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    output = Path(output_directory)
    manifest_path = output / PARQUET_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("M3.1 Parquet immutable manifest is missing or unsafe")
    # The write function is idempotent only after full source and Parquet
    # replay; when both files exist it performs no mutation.
    return write_parquet_export(
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
        smoke_merge_path=smoke_merge_path,
        output_directory=output,
    )


def _cli_plan(path: str | Path) -> dict[str, Any]:
    return contract.validate_dataset_plan(
        _read_canonical(path, "M3.1 dataset plan")
    )


def _cli_shard_map(path: str | Path) -> dict[str, str]:
    value = _read_canonical(path, "M3.1 shard-directory map")
    expected_keys = {str(row["shard_id"]) for row in contract.build_dataset_plan()["shards"]}
    if set(value) != expected_keys or any(
        not isinstance(item, str) or not Path(item).is_absolute()
        for item in value.values()
    ):
        raise ValueError(
            "M3.1 shard-directory map must contain the exact absolute 360-shard grid"
        )
    return {str(key): str(item) for key, item in value.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed M3.1 T3 teacher-dataset lifecycle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run-smoke", help="run or resume the sole 25-paired smoke shard"
    )
    run.add_argument("--plan", required=True)
    run.add_argument("--fresh-quality-gate", required=True)
    run.add_argument("--shard-directory", required=True)
    run.add_argument("--library", required=True)
    run.add_argument("--max-new-pairs", type=int)

    smoke_gate = subparsers.add_parser(
        "write-smoke-gate",
        help="source-replay the completed smoke shard and open full fanout",
    )
    smoke_gate.add_argument("--plan", required=True)
    smoke_gate.add_argument("--smoke-shard-directory", required=True)
    smoke_gate.add_argument("--output", required=True)

    shard = subparsers.add_parser(
        "run-shard", help="run/resume one post-smoke 25-paired shard"
    )
    shard.add_argument("--plan", required=True)
    shard.add_argument("--shard-id", required=True)
    shard.add_argument("--shard-directory", required=True)
    shard.add_argument("--fresh-quality-gate", required=True)
    shard.add_argument("--smoke-gate", required=True)
    shard.add_argument("--smoke-shard-directory", required=True)
    shard.add_argument("--library", required=True)
    shard.add_argument("--max-new-pairs", type=int)

    dataset_merge = subparsers.add_parser(
        "merge-dataset",
        help="source-replay and merge the exact immutable 360-shard grid",
    )
    dataset_merge.add_argument("--plan", required=True)
    dataset_merge.add_argument("--shard-map", required=True)
    dataset_merge.add_argument("--output", required=True)

    merge = subparsers.add_parser(
        "merge-smoke", help="source-replay the complete smoke shard"
    )
    merge.add_argument("--plan", required=True)
    merge.add_argument("--fresh-quality-gate", required=True)
    merge.add_argument("--shard-directory", required=True)
    merge.add_argument("--output", required=True)

    parquet = subparsers.add_parser(
        "export-parquet", help="write/validate the immutable smoke Parquet"
    )
    parquet.add_argument("--plan", required=True)
    parquet.add_argument("--fresh-quality-gate", required=True)
    parquet.add_argument("--shard-directory", required=True)
    parquet.add_argument("--smoke-merge", required=True)
    parquet.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _cli_plan(args.plan)
    if args.command == "run-smoke":
        value = run_smoke_shard(
            plan=plan,
            shard_directory=args.shard_directory,
            fresh_quality_gate_path=args.fresh_quality_gate,
            library_path=args.library,
            max_new_pairs=args.max_new_pairs,
        )
    elif args.command == "write-smoke-gate":
        value = contract.write_smoke_gate_receipt(
            plan=plan,
            smoke_shard_directory=args.smoke_shard_directory,
            output_path=args.output,
        )
    elif args.command == "run-shard":
        value = run_dataset_shard(
            plan=plan,
            shard_id=args.shard_id,
            shard_directory=args.shard_directory,
            fresh_quality_gate_path=args.fresh_quality_gate,
            smoke_gate_receipt_path=args.smoke_gate,
            smoke_shard_directory=args.smoke_shard_directory,
            library_path=args.library,
            max_new_pairs=args.max_new_pairs,
        )
    elif args.command == "merge-dataset":
        value = contract.write_merge_manifest(
            plan=plan,
            shard_directories=_cli_shard_map(args.shard_map),
            output_path=args.output,
        )
    elif args.command == "merge-smoke":
        value = write_smoke_merge(
            plan=plan,
            shard_directory=args.shard_directory,
            fresh_quality_gate_path=args.fresh_quality_gate,
            output_path=args.output,
        )
    else:
        value = write_parquet_export(
            plan=plan,
            shard_directory=args.shard_directory,
            fresh_quality_gate_path=args.fresh_quality_gate,
            smoke_merge_path=args.smoke_merge,
            output_directory=args.output_directory,
        )
    print(canonical_bytes(value).decode("ascii"), end="")
    return 0


__all__ = [
    "AUTHORIZATION_NAME",
    "AUTHORIZATION_SCHEMA",
    "EVIDENCE_DIRECTORY_NAME",
    "PAIR_EVIDENCE_SCHEMA",
    "FULL_AUTHORIZATION_SCHEMA",
    "PARQUET_COLUMNS",
    "PARQUET_FILE_NAME",
    "PARQUET_MANIFEST_NAME",
    "PARQUET_MANIFEST_SCHEMA",
    "SMOKE_MERGE_NAME",
    "SMOKE_MERGE_SCHEMA",
    "bind_fresh_quality_authorization",
    "bind_full_fanout_authorization",
    "build_smoke_merge",
    "canonical_bytes",
    "canonical_sha256",
    "evidence_artifact_path",
    "main",
    "run_smoke_shard",
    "run_dataset_shard",
    "validate_pair_evidence",
    "validate_parquet_export",
    "validate_smoke_merge_value",
    "write_parquet_export",
    "write_smoke_merge",
]


if __name__ == "__main__":
    raise SystemExit(main())
