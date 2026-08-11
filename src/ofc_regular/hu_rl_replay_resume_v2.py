"""Deterministic, shard-resumable actor-safe HU RL replay pilot.

V2 deliberately does not serialize the native ``WorldState``.  A shard is
reconstructed from an externally supplied seed and counter-based policy RNG.
The shared ledger stores public transition bytes and RNG provenance, but no
packed actor observation and no raw semantic ActionKey.  This is necessary
because joining both seats' rows would otherwise reveal each opponent's
private discard.  Exact observations and ActionKeys exist only transiently
during fail-closed reconstruction.

This module is a scalable correctness contract, not the 100,000-pair
acceptance run and not the final Parquet storage layer.  Shards use
deterministic gzip-compressed canonical JSON so clean and interrupted/resumed
runs can be compared byte-for-byte without an extra dependency.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
import os
import re
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from .hu_rl_native import (
    MAX_BATCH_LANES,
    MAX_BATCH_THREADS,
    PACKED_STEP_RECORD_BYTES,
    NativeBatchHuRlEnvV1,
)
from .hu_rl_replay_resume_smoke import collect_replay_resume_provenance


MANIFEST_SCHEMA: Final = "regular_ofc_hu_rl_replay_resume_manifest_v2"
SHARD_SCHEMA: Final = "regular_ofc_hu_rl_actor_safe_replay_shard_v2"
RECORD_SCHEMA: Final = "regular_ofc_hu_rl_actor_safe_transition_v2"
CHECKPOINT_SCHEMA: Final = "regular_ofc_hu_rl_replay_checkpoint_v2"
AGGREGATE_SCHEMA: Final = "regular_ofc_hu_rl_replay_resume_aggregate_v2"
PROVENANCE_SCHEMA: Final = "regular_ofc_hu_rl_replay_resume_provenance_v3"
RNG_RECORD_SCHEMA: Final = "regular_ofc_hu_rl_counter_rng_provenance_v2"

DECK_RNG_NAMESPACE: Final = "regular_ofc_hu_rl_deck_pair_v2"
POLICY_RNG_NAMESPACE: Final = "regular_ofc_hu_rl_behavior_action_v2"
BEHAVIOR_MODE: Final = "uniform_hash_index_from_counter_v1"
TRAJECTORY_STORAGE: Final = "canonical_json_gzip_mtime0_pilot_v1"
DECISIONS_PER_HAND: Final = 10
LEGS_PER_PAIR: Final = 2
RECORDS_PER_PAIR: Final = DECISIONS_PER_HAND * LEGS_PER_PAIR
ACCEPTANCE_PAIR_TARGET: Final = 100_000
MAX_PAIRED_HANDS: Final = ACCEPTANCE_PAIR_TARGET
MAX_SHARD_PAIRS: Final = MAX_BATCH_LANES // LEGS_PER_PAIR
MAX_SHARDS: Final = 256
MAX_POLICY_ID_BYTES: Final = 128
MAX_RUN_ID_BYTES: Final = 128
MAX_COMPRESSED_SHARD_BYTES: Final = 64 * 1024 * 1024
MAX_UNCOMPRESSED_SHARD_BYTES: Final = 128 * 1024 * 1024
_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]*\Z")
_SOURCE_PATHS: Final = (
    "rust/hu_rl_engine/src/batch.rs",
    "rust/hu_rl_engine/src/lib.rs",
    "rust/hu_rl_engine/src/python.rs",
    "rust/hu_rl_engine/src/seeded_deck.rs",
    "scripts/run_hu_rl_replay_resume_v2.py",
    "src/ofc_regular/hu_rl_native.py",
    "src/ofc_regular/hu_rl_replay_resume_v2.py",
)
_DECISION_STREETS: Final = ("T0", "T0", "T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4")


class HuRlReplayResumeV2Error(ValueError):
    """V2 manifest, shard, checkpoint, or reconstruction failed closed."""


EnvFactory = Callable[..., Any]
ProvenanceProvider = Callable[[], Mapping[str, Any]]


def prepare_replay_resume_run(
    output_dir: Path,
    *,
    run_id: str,
    paired_hand_start: int,
    paired_hand_count: int,
    shard_pair_count: int,
    seed_base: int,
    seed_stride: int,
    policy_a_id: str,
    policy_b_id: str,
    chunk_width: int = 64,
    thread_count: int = 8,
    _provenance_provider: ProvenanceProvider | None = None,
) -> dict[str, Any]:
    """Create the write-once execution manifest without persisting ``seed_base``."""

    _validate_plan_values(
        run_id=run_id,
        paired_hand_start=paired_hand_start,
        paired_hand_count=paired_hand_count,
        shard_pair_count=shard_pair_count,
        seed_base=seed_base,
        seed_stride=seed_stride,
        policy_a_id=policy_a_id,
        policy_b_id=policy_b_id,
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise HuRlReplayResumeV2Error("replay v2 manifest already exists")
    provenance = dict((_provenance_provider or collect_replay_resume_v2_provenance)())
    _validate_provenance(provenance)
    shard_count = _ceil_div(paired_hand_count, shard_pair_count)
    seed_commitment = _seed_base_commitment(seed_base)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "artifact_role": "execution_control_not_policy_input",
        "run_id": run_id,
        "paired_hand_start": paired_hand_start,
        "paired_hand_count": paired_hand_count,
        "hand_count": paired_hand_count * LEGS_PER_PAIR,
        "shard_pair_count": shard_pair_count,
        "shard_count": shard_count,
        "decision_count_per_hand": DECISIONS_PER_HAND,
        "policy_a_id": policy_a_id,
        "policy_b_id": policy_b_id,
        "behavior_mode": BEHAVIOR_MODE,
        "chunk_width": chunk_width,
        "thread_count": thread_count,
        "rng": {
            "deck_namespace": DECK_RNG_NAMESPACE,
            "policy_namespace": POLICY_RNG_NAMESPACE,
            "seed_base_commitment_sha256": seed_commitment,
            "seed_stride": seed_stride,
            "pair_seed_formula": "seed_base_plus_global_pair_index_times_seed_stride",
            "native_deck_generator": "rust_cpython_mt19937_splitmix64_shuffle_v1",
            "paired_lane_order": "pair_major_ab_then_ba",
            "seat_swap_formula": "swap_0_5_5_10_10_13_13_16_16_19_19_22_22_25_25_28_28_31_31_34",
            "hidden_tail_coupling": "positions_34_through_51_identical",
            "deck_materialized_in_python": False,
            "policy_counter_formula": "global_pair_index_times_10_plus_population_role_times_5_plus_street",
            "raw_seed_persisted_in_trajectory": False,
        },
        "trajectory_contract": {
            "schema": RECORD_SCHEMA,
            "storage": TRAJECTORY_STORAGE,
            "actor_observation_persisted": False,
            "chosen_action_key_persisted": False,
            "action_key_reconstruction": "rust_seed_range_plus_policy_common_tape_fail_closed",
            "deck_reconstruction": "rust_paired_seed_range_fail_closed",
            "python_full_deck_materialized": False,
            "public_transition": "packed_public_step_v1",
            "opponent_private_discard_exposed": False,
            "realized_deck_tail_exposed": False,
            "world_state_exposed": False,
            "unknown_fields_allowed": False,
        },
        "durability": _durability_contract(),
        "acceptance": {
            "paired_hand_target": ACCEPTANCE_PAIR_TARGET,
            "target_scale_evaluated": False,
            "parquet_gate_evaluated": False,
        },
        "provenance": provenance,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = _self_digest(manifest, "manifest_sha256")
    validate_manifest(manifest, expected_provenance=provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    _write_json_exclusive(manifest_path, manifest)
    return manifest


def run_or_resume_replay(
    output_dir: Path,
    *,
    seed_base: int,
    max_new_shards: int | None = None,
    _env_factory: EnvFactory | None = None,
    _provenance_provider: ProvenanceProvider | None = None,
) -> dict[str, Any]:
    """Validate any prefix, append immutable shards, and finalize when complete."""

    output_dir = Path(output_dir)
    provider = _provenance_provider or collect_replay_resume_v2_provenance
    manifest = _load_manifest(output_dir, provider)
    if manifest["durability"] != _durability_contract():
        raise HuRlReplayResumeV2Error(
            "replay mutation runtime does not match the prepared durability mode"
        )
    _resync_artifact_directories(output_dir)
    _validate_seed_for_manifest(seed_base, manifest)
    if max_new_shards is not None and (
        not _strict_int(max_new_shards) or max_new_shards < 0
    ):
        raise HuRlReplayResumeV2Error("max_new_shards must be a nonnegative integer")

    _recover_missing_trailing_checkpoint(
        output_dir,
        manifest,
        seed_base=seed_base,
        env_factory=_env_factory,
    )
    existing = _validate_contiguous_prefix(
        output_dir,
        manifest,
        seed_base=seed_base,
        env_factory=_env_factory,
    )
    added = 0
    for shard_index in range(existing, manifest["shard_count"]):
        if max_new_shards is not None and added >= max_new_shards:
            break
        shard = _execute_shard(
            manifest,
            shard_index=shard_index,
            seed_base=seed_base,
            env_factory=_env_factory,
        )
        shard_bytes = _encode_shard(shard)
        shard_path = _shard_path(output_dir, shard_index)
        _write_bytes_atomic_create_only(shard_path, shard_bytes)
        checkpoint = _build_checkpoint(output_dir, manifest, shard_index + 1)
        _write_json_exclusive(_checkpoint_path(output_dir, shard_index + 1), checkpoint)
        added += 1

    completed = _discover_contiguous_shards(output_dir, manifest)
    if completed == manifest["shard_count"]:
        aggregate = _build_aggregate(output_dir, manifest)
        aggregate_path = output_dir / "aggregate.json"
        if aggregate_path.exists():
            observed = _read_json(aggregate_path)
            validate_aggregate(observed, manifest=manifest, output_dir=output_dir)
            if observed != aggregate:
                raise HuRlReplayResumeV2Error("existing aggregate disagrees with reconstruction")
        else:
            _write_json_exclusive(aggregate_path, aggregate)
        return {
            "status": "complete",
            "completed_shards": completed,
            "total_shards": manifest["shard_count"],
            "new_shards": added,
            "aggregate_sha256": aggregate["aggregate_sha256"],
        }
    if (output_dir / "aggregate.json").exists():
        raise HuRlReplayResumeV2Error("incomplete run unexpectedly contains an aggregate")
    return {
        "status": "paused",
        "completed_shards": completed,
        "total_shards": manifest["shard_count"],
        "new_shards": added,
        "aggregate_sha256": None,
    }


def validate_replay_resume_run(
    output_dir: Path,
    *,
    seed_base: int,
    require_complete: bool = True,
    _env_factory: EnvFactory | None = None,
    _provenance_provider: ProvenanceProvider | None = None,
) -> dict[str, Any]:
    """Reconstruct every completed transition and validate the durable prefix."""

    output_dir = Path(output_dir)
    manifest = _load_manifest(
        output_dir,
        _provenance_provider or collect_replay_resume_v2_provenance,
    )
    _validate_seed_for_manifest(seed_base, manifest)
    completed = _validate_contiguous_prefix(
        output_dir,
        manifest,
        seed_base=seed_base,
        env_factory=_env_factory,
    )
    complete = completed == manifest["shard_count"]
    if require_complete and not complete:
        raise HuRlReplayResumeV2Error("replay v2 run is incomplete")
    aggregate_sha: str | None = None
    if complete:
        aggregate_path = output_dir / "aggregate.json"
        if not aggregate_path.is_file():
            raise HuRlReplayResumeV2Error("complete replay v2 run lacks aggregate")
        aggregate = _read_json(aggregate_path)
        validate_aggregate(aggregate, manifest=manifest, output_dir=output_dir)
        expected = _build_aggregate(output_dir, manifest)
        if aggregate != expected:
            raise HuRlReplayResumeV2Error("aggregate is not byte-stable reconstruction")
        aggregate_sha = aggregate["aggregate_sha256"]
    return {
        "status": "validated_complete" if complete else "validated_prefix",
        "completed_shards": completed,
        "total_shards": manifest["shard_count"],
        "reconstructed_records": _completed_record_count(manifest, completed),
        "expected_records": manifest["paired_hand_count"] * RECORDS_PER_PAIR,
        "checkpoint_reconstruction_complete": complete,
        "replay_reconstruction_complete": complete,
        "return_reconstruction_complete": complete,
        "aggregate_sha256": aggregate_sha,
    }


def validate_manifest(
    manifest: Mapping[str, Any], *, expected_provenance: Mapping[str, Any] | None = None
) -> None:
    _require_exact_fields(
        manifest,
        {
            "schema", "status", "artifact_role", "run_id", "paired_hand_start",
            "paired_hand_count", "hand_count", "shard_pair_count", "shard_count",
            "decision_count_per_hand", "policy_a_id", "policy_b_id", "behavior_mode",
            "chunk_width", "thread_count", "rng", "trajectory_contract", "durability", "acceptance",
            "provenance", "manifest_sha256",
        },
        "manifest",
    )
    literals = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "artifact_role": "execution_control_not_policy_input",
        "decision_count_per_hand": DECISIONS_PER_HAND,
        "behavior_mode": BEHAVIOR_MODE,
    }
    for field, expected in literals.items():
        if manifest[field] != expected:
            raise HuRlReplayResumeV2Error(f"manifest {field} mismatch")
    # A placeholder seed of zero is sufficient for shape/range checks; the
    # actual seed is intentionally absent from the manifest.
    _validate_plan_values(
        run_id=manifest["run_id"], paired_hand_start=manifest["paired_hand_start"],
        paired_hand_count=manifest["paired_hand_count"], shard_pair_count=manifest["shard_pair_count"],
        seed_base=0, seed_stride=manifest["rng"].get("seed_stride") if isinstance(manifest["rng"], Mapping) else None,
        policy_a_id=manifest["policy_a_id"], policy_b_id=manifest["policy_b_id"],
        chunk_width=manifest["chunk_width"], thread_count=manifest["thread_count"],
    )
    if manifest["hand_count"] != manifest["paired_hand_count"] * LEGS_PER_PAIR:
        raise HuRlReplayResumeV2Error("manifest hand_count mismatch")
    if manifest["shard_count"] != _ceil_div(manifest["paired_hand_count"], manifest["shard_pair_count"]):
        raise HuRlReplayResumeV2Error("manifest shard_count mismatch")
    rng = manifest["rng"]
    _require_exact_fields(rng, {
        "deck_namespace", "policy_namespace", "seed_base_commitment_sha256", "seed_stride",
        "pair_seed_formula", "native_deck_generator", "paired_lane_order", "seat_swap_formula",
        "hidden_tail_coupling", "deck_materialized_in_python", "policy_counter_formula",
        "raw_seed_persisted_in_trajectory",
    }, "manifest rng")
    if rng["deck_namespace"] != DECK_RNG_NAMESPACE or rng["policy_namespace"] != POLICY_RNG_NAMESPACE:
        raise HuRlReplayResumeV2Error("manifest RNG namespace mismatch")
    if rng["pair_seed_formula"] != "seed_base_plus_global_pair_index_times_seed_stride" or rng["policy_counter_formula"] != "global_pair_index_times_10_plus_population_role_times_5_plus_street":
        raise HuRlReplayResumeV2Error("manifest RNG formula mismatch")
    if (
        rng["native_deck_generator"] != "rust_cpython_mt19937_splitmix64_shuffle_v1"
        or rng["paired_lane_order"] != "pair_major_ab_then_ba"
        or rng["seat_swap_formula"]
        != "swap_0_5_5_10_10_13_13_16_16_19_19_22_22_25_25_28_28_31_31_34"
        or rng["hidden_tail_coupling"] != "positions_34_through_51_identical"
        or rng["deck_materialized_in_python"] is not False
    ):
        raise HuRlReplayResumeV2Error("manifest native deck generator contract mismatch")
    if not _is_sha256(rng["seed_base_commitment_sha256"]):
        raise HuRlReplayResumeV2Error("manifest seed commitment is invalid")
    if rng["raw_seed_persisted_in_trajectory"] is not False:
        raise HuRlReplayResumeV2Error("manifest permits raw seed trajectory leakage")
    contract = manifest["trajectory_contract"]
    _require_exact_fields(contract, {
        "schema", "storage", "actor_observation_persisted",
        "chosen_action_key_persisted", "action_key_reconstruction",
        "deck_reconstruction", "python_full_deck_materialized",
        "public_transition", "opponent_private_discard_exposed", "realized_deck_tail_exposed",
        "world_state_exposed", "unknown_fields_allowed",
    }, "trajectory contract")
    expected_contract = {
        "schema": RECORD_SCHEMA, "storage": TRAJECTORY_STORAGE,
        "actor_observation_persisted": False, "chosen_action_key_persisted": False,
        "action_key_reconstruction": "rust_seed_range_plus_policy_common_tape_fail_closed",
        "deck_reconstruction": "rust_paired_seed_range_fail_closed",
        "python_full_deck_materialized": False,
        "public_transition": "packed_public_step_v1", "opponent_private_discard_exposed": False,
        "realized_deck_tail_exposed": False, "world_state_exposed": False,
        "unknown_fields_allowed": False,
    }
    if dict(contract) != expected_contract:
        raise HuRlReplayResumeV2Error("trajectory contract mismatch")
    _validate_durability_contract(manifest["durability"])
    acceptance = manifest["acceptance"]
    _require_exact_fields(acceptance, {"paired_hand_target", "target_scale_evaluated", "parquet_gate_evaluated"}, "acceptance")
    if acceptance["paired_hand_target"] != ACCEPTANCE_PAIR_TARGET or acceptance["parquet_gate_evaluated"] is not False:
        raise HuRlReplayResumeV2Error("acceptance contract mismatch")
    if acceptance["target_scale_evaluated"] is not False:
        raise HuRlReplayResumeV2Error("prepared manifest overclaims target-scale evaluation")
    _validate_provenance(manifest["provenance"])
    if expected_provenance is not None and manifest["provenance"] != expected_provenance:
        raise HuRlReplayResumeV2Error("source/wheel provenance changed")
    if manifest["manifest_sha256"] != _self_digest(manifest, "manifest_sha256"):
        raise HuRlReplayResumeV2Error("manifest digest mismatch")


def validate_aggregate(
    aggregate: Mapping[str, Any], *, manifest: Mapping[str, Any], output_dir: Path
) -> None:
    _require_exact_fields(aggregate, {
        "schema", "status", "artifact_role", "manifest_sha256", "paired_hand_count",
        "hand_count", "record_count", "shard_count", "ordered_shards",
        "terminal_returns_sha256", "checkpoint_sha256", "byte_identical_contract",
        "reconstruction", "durability", "acceptance", "aggregate_sha256",
    }, "aggregate")
    if aggregate["schema"] != AGGREGATE_SCHEMA or aggregate["status"] != "complete":
        raise HuRlReplayResumeV2Error("aggregate status/schema mismatch")
    if aggregate["artifact_role"] != "actor_safe_replay_resume_aggregate":
        raise HuRlReplayResumeV2Error("aggregate role mismatch")
    if aggregate["manifest_sha256"] != manifest["manifest_sha256"]:
        raise HuRlReplayResumeV2Error("aggregate manifest mismatch")
    expected_counts = (
        manifest["paired_hand_count"], manifest["hand_count"],
        manifest["paired_hand_count"] * RECORDS_PER_PAIR, manifest["shard_count"],
    )
    observed_counts = (aggregate["paired_hand_count"], aggregate["hand_count"], aggregate["record_count"], aggregate["shard_count"])
    if observed_counts != expected_counts:
        raise HuRlReplayResumeV2Error("aggregate counts mismatch")
    rows = aggregate["ordered_shards"]
    if not isinstance(rows, list) or len(rows) != manifest["shard_count"]:
        raise HuRlReplayResumeV2Error("aggregate shard list mismatch")
    for index, row in enumerate(rows):
        _require_exact_fields(row, {"shard_index", "file_sha256", "payload_sha256", "record_count"}, "aggregate shard row")
        path = _shard_path(Path(output_dir), index)
        if row["shard_index"] != index or row["file_sha256"] != _sha256_file(path):
            raise HuRlReplayResumeV2Error("aggregate shard file mismatch")
    for field in ("terminal_returns_sha256", "checkpoint_sha256", "aggregate_sha256"):
        if not _is_sha256(aggregate[field]):
            raise HuRlReplayResumeV2Error(f"aggregate {field} invalid")
    if aggregate["byte_identical_contract"] is not True:
        raise HuRlReplayResumeV2Error("aggregate byte-identical contract not asserted")
    reconstruction = aggregate["reconstruction"]
    expected_reconstruction = {
        "checkpoint": {"numerator": manifest["shard_count"], "denominator": manifest["shard_count"]},
        "replay": {"numerator": expected_counts[2], "denominator": expected_counts[2]},
        "returns": {"numerator": expected_counts[2], "denominator": expected_counts[2]},
    }
    if reconstruction != expected_reconstruction:
        raise HuRlReplayResumeV2Error("aggregate reconstruction counts mismatch")
    if aggregate["durability"] != manifest["durability"]:
        raise HuRlReplayResumeV2Error("aggregate durability contract mismatch")
    acceptance = aggregate["acceptance"]
    if acceptance != {
        "paired_hand_target": ACCEPTANCE_PAIR_TARGET,
        "target_scale_evaluated": False,
        "target_scale_pass": None,
        "parquet_gate_evaluated": False,
    }:
        raise HuRlReplayResumeV2Error("aggregate acceptance claims mismatch")
    if aggregate["aggregate_sha256"] != _self_digest(aggregate, "aggregate_sha256"):
        raise HuRlReplayResumeV2Error("aggregate digest mismatch")


def collect_replay_resume_v2_provenance() -> dict[str, Any]:
    base = collect_replay_resume_provenance()
    repository = Path(__file__).resolve().parents[2]
    hashes = dict(base["source_sha256"])
    for relative in _SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise HuRlReplayResumeV2Error("replay v2 source snapshot is incomplete")
        hashes[relative] = _sha256_file(path)
    return {
        "schema": PROVENANCE_SCHEMA,
        "package_name": base["package_name"],
        "package_version": base["package_version"],
        "wheel_filename": base["wheel_filename"],
        "wheel_sha256": base["wheel_sha256"],
        "native_extension_filename": base["native_extension_filename"],
        "native_extension_sha256": base["native_extension_sha256"],
        "source_sha256": hashes,
    }


def _execute_shard(
    manifest: Mapping[str, Any], *, shard_index: int, seed_base: int,
    env_factory: EnvFactory | None,
) -> dict[str, Any]:
    start_offset = shard_index * manifest["shard_pair_count"]
    pair_count = min(manifest["shard_pair_count"], manifest["paired_hand_count"] - start_offset)
    global_start = manifest["paired_hand_start"] + start_offset
    lane_meta: list[tuple[int, str]] = []
    for local in range(pair_count):
        pair_index = global_start + local
        lane_meta.extend(((pair_index, "ab"), (pair_index, "ba")))
    factory = env_factory or NativeBatchHuRlEnvV1
    seeded_constructor = getattr(factory, "from_paired_seed_range", None)
    if not callable(seeded_constructor):
        raise HuRlReplayResumeV2Error(
            "replay environment lacks the required native paired seed constructor"
        )
    env = seeded_constructor(
        seed_base=seed_base,
        global_pair_start=global_start,
        pair_count=pair_count,
        seed_stride=manifest["rng"]["seed_stride"],
        chunk_width=manifest["chunk_width"],
        thread_count=manifest["thread_count"],
    )
    lane_count = pair_count * LEGS_PER_PAIR
    if env.lane_count != lane_count:
        raise HuRlReplayResumeV2Error("native paired seed batch width mismatch")
    records_by_lane: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    terminal_returns: list[tuple[float, float]] | None = None
    for decision in range(DECISIONS_PER_HAND):
        actor_decision = env.actor_decision_batch_packed()
        legal = actor_decision.legal_actions
        selected_indices: list[int] = []
        rng_rows: list[dict[str, Any]] = []
        for lane, (pair_index, leg) in enumerate(lane_meta):
            actor = decision % 2
            policy_id = _acting_policy_id(manifest, leg, actor)
            population_role = _acting_population_role(leg, actor)
            count = legal.action_count(lane)
            index, rng_row = _policy_draw(
                seed_base=seed_base, seed_commitment=manifest["rng"]["seed_base_commitment_sha256"],
                pair_index=pair_index, population_role=population_role,
                decision=decision, policy_id=policy_id,
                action_count=count,
            )
            selected_indices.append(index)
            rng_rows.append(rng_row)
        selected = legal.select(tuple(selected_indices))
        step = env.step_batch_packed(selected)
        for lane, (pair_index, leg) in enumerate(lane_meta):
            outcome = step.lane(lane)
            public_step = step.payload[
                lane * PACKED_STEP_RECORD_BYTES:(lane + 1) * PACKED_STEP_RECORD_BYTES
            ]
            actor = decision % 2
            record: dict[str, Any] = {
                "schema": RECORD_SCHEMA,
                "pair_index": pair_index,
                "seat_swap_leg": leg,
                "decision_ordinal": decision,
                "acting_seat": "first" if actor == 0 else "second",
                "street": _DECISION_STREETS[decision],
                "policy_checkpoint_id": _acting_policy_id(manifest, leg, actor),
                "population_role": _acting_population_role(leg, actor),
                "actor_private_observation_persisted": False,
                "legal_action_count": legal.action_count(lane),
                "chosen_action_key_persisted": False,
                "action_key_reconstruction": "rust_seed_range_plus_policy_common_tape_fail_closed",
                "behavior_logprob_hex": float(-math.log(legal.action_count(lane))).hex(),
                "rng": rng_rows[lane],
                "public_step_b64": _b64(public_step),
                "rewards_hex": [float(value).hex() for value in outcome.rewards],
                "done": outcome.done,
                "returns_hex": None,
                "audit_token_sha256": None,
            }
            records_by_lane[lane].append(record)
            if decision == DECISIONS_PER_HAND - 1:
                terminal_returns = terminal_returns or []
                terminal_returns.append(tuple(float(value) for value in outcome.rewards))
    if not env.all_done or terminal_returns is None or len(terminal_returns) != lane_count:
        raise HuRlReplayResumeV2Error("shard did not reach every terminal state")
    records: list[dict[str, Any]] = []
    terminal_bytes = bytearray()
    for lane, lane_records in enumerate(records_by_lane):
        returns = terminal_returns[lane]
        terminal_bytes.extend(struct.pack("<2d", *returns))
        for record in lane_records:
            record["returns_hex"] = [value.hex() for value in returns]
            record["audit_token_sha256"] = _record_digest(record)
            validate_record(record)
            records.append(record)
    records.sort(key=lambda row: (row["pair_index"], 0 if row["seat_swap_leg"] == "ab" else 1, row["decision_ordinal"]))
    shard: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "status": "complete",
        "artifact_role": "public_replay_reconstruction_ledger",
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_index": shard_index,
        "paired_hand_start": global_start,
        "paired_hand_count": pair_count,
        "record_count": pair_count * RECORDS_PER_PAIR,
        "records": records,
        "terminal_returns_b64": _b64(bytes(terminal_bytes)),
        "terminal_returns_sha256": _sha256(bytes(terminal_bytes)),
        "hidden_truth_exposed": False,
        "opponent_private_discard_exposed": False,
        "raw_seed_exposed": False,
        "payload_sha256": None,
    }
    shard["payload_sha256"] = _self_digest(shard, "payload_sha256")
    validate_shard(shard, manifest=manifest, expected_index=shard_index)
    return shard


def validate_record(record: Mapping[str, Any]) -> None:
    _require_exact_fields(record, {
        "schema", "pair_index", "seat_swap_leg", "decision_ordinal", "acting_seat", "street",
        "policy_checkpoint_id", "population_role", "actor_private_observation_persisted",
        "legal_action_count", "chosen_action_key_persisted", "action_key_reconstruction",
        "behavior_logprob_hex", "rng", "public_step_b64",
        "rewards_hex", "done", "returns_hex", "audit_token_sha256",
    }, "trajectory record")
    if record["schema"] != RECORD_SCHEMA:
        raise HuRlReplayResumeV2Error("trajectory record schema mismatch")
    if not _strict_int(record["pair_index"]) or record["pair_index"] < 0:
        raise HuRlReplayResumeV2Error("trajectory pair index invalid")
    if record["seat_swap_leg"] not in {"ab", "ba"}:
        raise HuRlReplayResumeV2Error("trajectory seat-swap leg invalid")
    decision = record["decision_ordinal"]
    if not _strict_int(decision) or not 0 <= decision < DECISIONS_PER_HAND:
        raise HuRlReplayResumeV2Error("trajectory decision ordinal invalid")
    expected_actor = "first" if decision % 2 == 0 else "second"
    if record["acting_seat"] != expected_actor or record["street"] != _DECISION_STREETS[decision]:
        raise HuRlReplayResumeV2Error("trajectory schedule mismatch")
    _validate_id(record["policy_checkpoint_id"], "policy checkpoint id", MAX_POLICY_ID_BYTES)
    if record["population_role"] not in {"a", "b"}:
        raise HuRlReplayResumeV2Error("trajectory population role is invalid")
    if record["actor_private_observation_persisted"] is not False:
        raise HuRlReplayResumeV2Error("trajectory persisted an actor-private observation")
    if record["chosen_action_key_persisted"] is not False:
        raise HuRlReplayResumeV2Error("trajectory persisted a private ActionKey")
    if record["action_key_reconstruction"] != "rust_seed_range_plus_policy_common_tape_fail_closed":
        raise HuRlReplayResumeV2Error("trajectory ActionKey reconstruction contract mismatch")
    public_step = _unb64(record["public_step_b64"], "public step")
    if len(public_step) != PACKED_STEP_RECORD_BYTES:
        raise HuRlReplayResumeV2Error("trajectory packed geometry mismatch")
    count = record["legal_action_count"]
    if not _strict_int(count) or not 1 <= count <= 232:
        raise HuRlReplayResumeV2Error("trajectory legal action count invalid")
    for field in ("audit_token_sha256",):
        if not _is_sha256(record[field]):
            raise HuRlReplayResumeV2Error(f"trajectory {field} invalid")
    try:
        logprob = float.fromhex(record["behavior_logprob_hex"])
    except (TypeError, ValueError) as exc:
        raise HuRlReplayResumeV2Error("trajectory behavior logprob invalid") from exc
    if logprob != -math.log(count):
        raise HuRlReplayResumeV2Error("trajectory behavior logprob mismatch")
    for field in ("rewards_hex", "returns_hex"):
        values = record[field]
        if not isinstance(values, list) or len(values) != 2:
            raise HuRlReplayResumeV2Error(f"trajectory {field} invalid")
        try:
            parsed = tuple(float.fromhex(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise HuRlReplayResumeV2Error(f"trajectory {field} invalid") from exc
        if any(not math.isfinite(value) for value in parsed) or parsed[0] != -parsed[1]:
            raise HuRlReplayResumeV2Error(f"trajectory {field} is not finite zero-sum")
    if type(record["done"]) is not bool:
        raise HuRlReplayResumeV2Error("trajectory done flag invalid")
    _validate_rng_record(
        record["rng"],
        pair_index=record["pair_index"],
        population_role=record["population_role"],
        decision=decision,
    )
    if record["audit_token_sha256"] != _record_digest(record):
        raise HuRlReplayResumeV2Error("trajectory audit token mismatch")
    encoded_keys = "\n".join(str(key).lower() for key in record)
    for forbidden in ("opponent_private_discard", "deck_tail", "world_state", "explicit_deck", "run_seed", "seed_base"):
        if forbidden in encoded_keys:
            raise HuRlReplayResumeV2Error("trajectory schema exposes a forbidden hidden field")


def validate_shard(shard: Mapping[str, Any], *, manifest: Mapping[str, Any], expected_index: int) -> None:
    _require_exact_fields(shard, {
        "schema", "status", "artifact_role", "manifest_sha256", "shard_index",
        "paired_hand_start", "paired_hand_count", "record_count", "records",
        "terminal_returns_b64", "terminal_returns_sha256", "hidden_truth_exposed",
        "opponent_private_discard_exposed", "raw_seed_exposed", "payload_sha256",
    }, "shard")
    if shard["schema"] != SHARD_SCHEMA or shard["status"] != "complete" or shard["artifact_role"] != "public_replay_reconstruction_ledger":
        raise HuRlReplayResumeV2Error("shard identity mismatch")
    if shard["manifest_sha256"] != manifest["manifest_sha256"] or shard["shard_index"] != expected_index:
        raise HuRlReplayResumeV2Error("shard manifest/index mismatch")
    expected_start = manifest["paired_hand_start"] + expected_index * manifest["shard_pair_count"]
    expected_count = min(manifest["shard_pair_count"], manifest["paired_hand_count"] - expected_index * manifest["shard_pair_count"])
    if shard["paired_hand_start"] != expected_start or shard["paired_hand_count"] != expected_count:
        raise HuRlReplayResumeV2Error("shard pair range mismatch")
    if shard["record_count"] != expected_count * RECORDS_PER_PAIR:
        raise HuRlReplayResumeV2Error("shard record count mismatch")
    records = shard["records"]
    if not isinstance(records, list) or len(records) != shard["record_count"]:
        raise HuRlReplayResumeV2Error("shard records geometry mismatch")
    previous: tuple[int, int, int] | None = None
    for record in records:
        if not isinstance(record, Mapping):
            raise HuRlReplayResumeV2Error("shard record is not a mapping")
        validate_record(record)
        key = (record["pair_index"], 0 if record["seat_swap_leg"] == "ab" else 1, record["decision_ordinal"])
        if previous is not None and key <= previous:
            raise HuRlReplayResumeV2Error("shard records are duplicated or reordered")
        previous = key
    expected_keys = [
        (pair, leg, decision)
        for pair in range(expected_start, expected_start + expected_count)
        for leg in ("ab", "ba")
        for decision in range(DECISIONS_PER_HAND)
    ]
    observed_keys = [(row["pair_index"], row["seat_swap_leg"], row["decision_ordinal"]) for row in records]
    if observed_keys != expected_keys:
        raise HuRlReplayResumeV2Error("shard trajectory coverage is incomplete")
    terminal = _unb64(shard["terminal_returns_b64"], "terminal returns")
    if len(terminal) != expected_count * LEGS_PER_PAIR * 16 or shard["terminal_returns_sha256"] != _sha256(terminal):
        raise HuRlReplayResumeV2Error("shard terminal returns mismatch")
    if any(shard[field] is not False for field in ("hidden_truth_exposed", "opponent_private_discard_exposed", "raw_seed_exposed")):
        raise HuRlReplayResumeV2Error("shard claims hidden information exposure")
    if shard["payload_sha256"] != _self_digest(shard, "payload_sha256"):
        raise HuRlReplayResumeV2Error("shard payload digest mismatch")


def _reconstruct_shard(
    observed: Mapping[str, Any], manifest: Mapping[str, Any], *, shard_index: int,
    seed_base: int, env_factory: EnvFactory | None,
) -> None:
    expected = _execute_shard(manifest, shard_index=shard_index, seed_base=seed_base, env_factory=env_factory)
    if observed != expected:
        raise HuRlReplayResumeV2Error("trajectory replay/return reconstruction mismatch")


def _build_checkpoint(output_dir: Path, manifest: Mapping[str, Any], completed: int) -> dict[str, Any]:
    rows = []
    for index in range(completed):
        path = _shard_path(output_dir, index)
        shard = _decode_shard(path.read_bytes())
        validate_shard(shard, manifest=manifest, expected_index=index)
        rows.append({"shard_index": index, "file_sha256": _sha256_file(path), "payload_sha256": shard["payload_sha256"]})
    checkpoint: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "validated_prefix",
        "artifact_role": "resume_checkpoint_not_policy_input",
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_shards": completed,
        "completed_paired_hands": _completed_pair_count(manifest, completed),
        "ordered_shards": rows,
        "checkpoint_sha256": None,
    }
    checkpoint["checkpoint_sha256"] = _self_digest(checkpoint, "checkpoint_sha256")
    _validate_checkpoint(checkpoint, output_dir=output_dir, manifest=manifest, expected_completed=completed)
    return checkpoint


def _validate_checkpoint(checkpoint: Mapping[str, Any], *, output_dir: Path, manifest: Mapping[str, Any], expected_completed: int) -> None:
    _require_exact_fields(checkpoint, {"schema", "status", "artifact_role", "manifest_sha256", "completed_shards", "completed_paired_hands", "ordered_shards", "checkpoint_sha256"}, "checkpoint")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA or checkpoint["status"] != "validated_prefix" or checkpoint["artifact_role"] != "resume_checkpoint_not_policy_input":
        raise HuRlReplayResumeV2Error("checkpoint identity mismatch")
    if checkpoint["manifest_sha256"] != manifest["manifest_sha256"] or checkpoint["completed_shards"] != expected_completed:
        raise HuRlReplayResumeV2Error("checkpoint prefix mismatch")
    if checkpoint["completed_paired_hands"] != _completed_pair_count(manifest, expected_completed):
        raise HuRlReplayResumeV2Error("checkpoint pair count mismatch")
    rows = checkpoint["ordered_shards"]
    if not isinstance(rows, list) or len(rows) != expected_completed:
        raise HuRlReplayResumeV2Error("checkpoint shard rows mismatch")
    for index, row in enumerate(rows):
        _require_exact_fields(row, {"shard_index", "file_sha256", "payload_sha256"}, "checkpoint shard")
        path = _shard_path(output_dir, index)
        shard = _decode_shard(path.read_bytes())
        if row != {"shard_index": index, "file_sha256": _sha256_file(path), "payload_sha256": shard["payload_sha256"]}:
            raise HuRlReplayResumeV2Error("checkpoint shard commitment mismatch")
    if checkpoint["checkpoint_sha256"] != _self_digest(checkpoint, "checkpoint_sha256"):
        raise HuRlReplayResumeV2Error("checkpoint digest mismatch")


def _build_aggregate(output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    terminal = bytearray()
    for index in range(manifest["shard_count"]):
        path = _shard_path(output_dir, index)
        shard = _decode_shard(path.read_bytes())
        validate_shard(shard, manifest=manifest, expected_index=index)
        rows.append({"shard_index": index, "file_sha256": _sha256_file(path), "payload_sha256": shard["payload_sha256"], "record_count": shard["record_count"]})
        terminal.extend(_unb64(shard["terminal_returns_b64"], "terminal returns"))
    checkpoint_path = _checkpoint_path(output_dir, manifest["shard_count"])
    checkpoint = _read_json(checkpoint_path)
    _validate_checkpoint(checkpoint, output_dir=output_dir, manifest=manifest, expected_completed=manifest["shard_count"])
    records = manifest["paired_hand_count"] * RECORDS_PER_PAIR
    aggregate: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "artifact_role": "actor_safe_replay_resume_aggregate",
        "manifest_sha256": manifest["manifest_sha256"],
        "paired_hand_count": manifest["paired_hand_count"],
        "hand_count": manifest["hand_count"],
        "record_count": records,
        "shard_count": manifest["shard_count"],
        "ordered_shards": rows,
        "terminal_returns_sha256": _sha256(bytes(terminal)),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "byte_identical_contract": True,
        "reconstruction": {
            "checkpoint": {"numerator": manifest["shard_count"], "denominator": manifest["shard_count"]},
            "replay": {"numerator": records, "denominator": records},
            "returns": {"numerator": records, "denominator": records},
        },
        "durability": manifest["durability"],
        "acceptance": {
            "paired_hand_target": ACCEPTANCE_PAIR_TARGET,
            # A completed 100k run is only one side of the gate.  A separate
            # clean-vs-interrupted comparator must prove byte identity before
            # target_scale_evaluated/pass may ever become true.
            "target_scale_evaluated": False,
            "target_scale_pass": None,
            "parquet_gate_evaluated": False,
        },
        "aggregate_sha256": None,
    }
    aggregate["aggregate_sha256"] = _self_digest(aggregate, "aggregate_sha256")
    validate_aggregate(aggregate, manifest=manifest, output_dir=output_dir)
    return aggregate


def _validate_contiguous_prefix(output_dir: Path, manifest: Mapping[str, Any], *, seed_base: int, env_factory: EnvFactory | None) -> int:
    completed = _discover_contiguous_shards(output_dir, manifest)
    checkpoint_files = sorted((output_dir / "checkpoints").glob("checkpoint-*.json"))
    expected_names = {_checkpoint_path(output_dir, value).name for value in range(1, completed + 1)}
    if {path.name for path in checkpoint_files} != expected_names:
        raise HuRlReplayResumeV2Error("checkpoint set contains a gap or unexpected file")
    for index in range(completed):
        path = _shard_path(output_dir, index)
        shard = _decode_shard(path.read_bytes())
        validate_shard(shard, manifest=manifest, expected_index=index)
        _reconstruct_shard(shard, manifest, shard_index=index, seed_base=seed_base, env_factory=env_factory)
        checkpoint = _read_json(_checkpoint_path(output_dir, index + 1))
        _validate_checkpoint(checkpoint, output_dir=output_dir, manifest=manifest, expected_completed=index + 1)
    return completed


def _recover_missing_trailing_checkpoint(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    seed_base: int,
    env_factory: EnvFactory | None,
) -> None:
    """Repair only the atomicity gap after a durable shard was linked.

    A process can die between the create-only shard link and its checkpoint.
    The shard is accepted as the source of truth only after a full deterministic
    replay.  More than one missing checkpoint is not a normal crash boundary
    and remains fail-closed.
    """

    completed = _discover_contiguous_shards(output_dir, manifest)
    observed = {
        path.name
        for path in (output_dir / "checkpoints").glob("checkpoint-*.json")
    }
    expected = {
        _checkpoint_path(output_dir, value).name
        for value in range(1, completed + 1)
    }
    if observed == expected:
        return
    missing_trailing = _checkpoint_path(output_dir, completed).name if completed else None
    if completed == 0 or observed != expected - {missing_trailing}:
        raise HuRlReplayResumeV2Error(
            "checkpoint set is not a recoverable single trailing gap"
        )
    trailing_index = completed - 1
    shard_path = _shard_path(output_dir, trailing_index)
    shard = _decode_shard(shard_path.read_bytes())
    validate_shard(shard, manifest=manifest, expected_index=trailing_index)
    _reconstruct_shard(
        shard,
        manifest,
        shard_index=trailing_index,
        seed_base=seed_base,
        env_factory=env_factory,
    )
    # The interrupted writer may have linked the final shard name but failed
    # before syncing its directory.  Re-establish that boundary before a
    # reconstructed checkpoint can become visible.
    _fsync_parent_directory(shard_path.parent)
    checkpoint = _build_checkpoint(output_dir, manifest, completed)
    _write_json_exclusive(
        _checkpoint_path(output_dir, completed),
        checkpoint,
    )


def _discover_contiguous_shards(output_dir: Path, manifest: Mapping[str, Any]) -> int:
    files = sorted((output_dir / "shards").glob("shard-*.json.gz"))
    expected = []
    for index, path in enumerate(files):
        expected_path = _shard_path(output_dir, index)
        if path.name != expected_path.name or index >= manifest["shard_count"]:
            raise HuRlReplayResumeV2Error("shard set contains a gap or unexpected file")
        expected.append(path)
    return len(expected)


def _load_manifest(output_dir: Path, provider: ProvenanceProvider) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    provenance = dict(provider())
    validate_manifest(manifest, expected_provenance=provenance)
    return manifest


def _validate_plan_values(**values: Any) -> None:
    _validate_id(values["run_id"], "run_id", MAX_RUN_ID_BYTES)
    for field in ("paired_hand_start", "paired_hand_count", "shard_pair_count", "seed_base", "seed_stride", "chunk_width", "thread_count"):
        if not _strict_int(values[field]):
            raise HuRlReplayResumeV2Error(f"{field} must be an integer")
    if values["paired_hand_start"] < 0 or not 1 <= values["paired_hand_count"] <= MAX_PAIRED_HANDS:
        raise HuRlReplayResumeV2Error("paired hand range is invalid")
    if not 1 <= values["shard_pair_count"] <= MAX_SHARD_PAIRS:
        raise HuRlReplayResumeV2Error("shard_pair_count exceeds native lane capacity")
    if _ceil_div(values["paired_hand_count"], values["shard_pair_count"]) > MAX_SHARDS:
        raise HuRlReplayResumeV2Error(
            f"shard plan exceeds maximum {MAX_SHARDS} durable checkpoints"
        )
    if (
        not 0 <= values["seed_base"] <= (1 << 63) - 1
        or not 1 <= values["seed_stride"] <= (1 << 63) - 1
    ):
        raise HuRlReplayResumeV2Error("seed base/stride is invalid")
    maximum_seed = values["seed_base"] + (values["paired_hand_start"] + values["paired_hand_count"] - 1) * values["seed_stride"]
    if maximum_seed > (1 << 63) - 1:
        raise HuRlReplayResumeV2Error("paired seed range exceeds unsigned 63-bit domain")
    if values["chunk_width"] <= 0 or not 1 <= values["thread_count"] <= MAX_BATCH_THREADS:
        raise HuRlReplayResumeV2Error("batch execution hints are invalid")
    _validate_id(values["policy_a_id"], "policy_a_id", MAX_POLICY_ID_BYTES)
    _validate_id(values["policy_b_id"], "policy_b_id", MAX_POLICY_ID_BYTES)


def _validate_seed_for_manifest(seed_base: int, manifest: Mapping[str, Any]) -> None:
    if not _strict_int(seed_base) or not 0 <= seed_base <= (1 << 63) - 1:
        raise HuRlReplayResumeV2Error("seed_base is invalid")
    if _seed_base_commitment(seed_base) != manifest["rng"]["seed_base_commitment_sha256"]:
        raise HuRlReplayResumeV2Error("seed_base commitment mismatch")
    maximum = seed_base + (manifest["paired_hand_start"] + manifest["paired_hand_count"] - 1) * manifest["rng"]["seed_stride"]
    if maximum > (1 << 63) - 1:
        raise HuRlReplayResumeV2Error("seed range overflow")


def _policy_draw(*, seed_base: int, seed_commitment: str, pair_index: int, population_role: str, decision: int, policy_id: str, action_count: int) -> tuple[int, dict[str, Any]]:
    role_index = 0 if population_role == "a" else 1
    street_index = decision // 2
    counter = pair_index * 10 + role_index * 5 + street_index
    material = f"{POLICY_RNG_NAMESPACE}\0{seed_base}\0{counter}\0{policy_id}".encode("ascii")
    draw = _sha256(material)
    # Multiply-high maps one fixed 64-bit common draw into each legal support.
    # The draw itself is identical for the same population role/street in the
    # AB and BA legs; action-count is deliberately not hash input.
    index = (int(draw[:16], 16) * action_count) >> 64
    return index, {
        "schema": RNG_RECORD_SCHEMA,
        "namespace": POLICY_RNG_NAMESPACE,
        "coupling": "paired_seat_swap_common_tape",
        "seed_base_commitment_sha256": seed_commitment,
        "counter": counter,
        "draw_sha256": draw,
    }


def _validate_rng_record(value: Any, *, pair_index: int, population_role: str, decision: int) -> None:
    if not isinstance(value, Mapping):
        raise HuRlReplayResumeV2Error("trajectory RNG provenance must be a mapping")
    _require_exact_fields(value, {"schema", "namespace", "coupling", "seed_base_commitment_sha256", "counter", "draw_sha256"}, "trajectory RNG provenance")
    role_index = 0 if population_role == "a" else 1
    expected_counter = pair_index * 10 + role_index * 5 + decision // 2
    if value["schema"] != RNG_RECORD_SCHEMA or value["namespace"] != POLICY_RNG_NAMESPACE or value["coupling"] != "paired_seat_swap_common_tape" or value["counter"] != expected_counter:
        raise HuRlReplayResumeV2Error("trajectory RNG provenance mismatch")
    if not _is_sha256(value["seed_base_commitment_sha256"]) or not _is_sha256(value["draw_sha256"]):
        raise HuRlReplayResumeV2Error("trajectory RNG digest invalid")


def _acting_policy_id(manifest: Mapping[str, Any], leg: str, actor: int) -> str:
    if leg == "ab":
        return manifest["policy_a_id"] if actor == 0 else manifest["policy_b_id"]
    return manifest["policy_b_id"] if actor == 0 else manifest["policy_a_id"]


def _acting_population_role(leg: str, actor: int) -> str:
    if leg == "ab":
        return "a" if actor == 0 else "b"
    return "b" if actor == 0 else "a"


def _seed_base_commitment(seed_base: int) -> str:
    return _sha256(f"{DECK_RNG_NAMESPACE}\0{POLICY_RNG_NAMESPACE}\0{seed_base}".encode("ascii"))


def _record_digest(record: Mapping[str, Any]) -> str:
    return _self_digest(record, "audit_token_sha256")


def _validate_provenance(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise HuRlReplayResumeV2Error("provenance must be a mapping")
    _require_exact_fields(value, {"schema", "package_name", "package_version", "wheel_filename", "wheel_sha256", "native_extension_filename", "native_extension_sha256", "source_sha256"}, "provenance")
    if value["schema"] != PROVENANCE_SCHEMA:
        raise HuRlReplayResumeV2Error("provenance schema mismatch")
    for field in ("package_name", "package_version", "wheel_filename", "native_extension_filename"):
        if not isinstance(value[field], str) or not value[field]:
            raise HuRlReplayResumeV2Error(f"provenance {field} invalid")
    for field in ("wheel_sha256", "native_extension_sha256"):
        if not _is_sha256(value[field]):
            raise HuRlReplayResumeV2Error(f"provenance {field} invalid")
    hashes = value["source_sha256"]
    if not isinstance(hashes, Mapping) or not set(_SOURCE_PATHS).issubset(hashes) or any(not _is_sha256(digest) for digest in hashes.values()):
        raise HuRlReplayResumeV2Error("provenance source hashes invalid")


def _encode_shard(shard: Mapping[str, Any]) -> bytes:
    raw = (_canonical_json(shard) + "\n").encode("ascii")
    if len(raw) > MAX_UNCOMPRESSED_SHARD_BYTES:
        raise HuRlReplayResumeV2Error("uncompressed shard exceeds safety limit")
    encoded = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(encoded) > MAX_COMPRESSED_SHARD_BYTES:
        raise HuRlReplayResumeV2Error("compressed shard exceeds safety limit")
    return encoded


def _decode_shard(encoded: bytes) -> dict[str, Any]:
    if not encoded or len(encoded) > MAX_COMPRESSED_SHARD_BYTES:
        raise HuRlReplayResumeV2Error("compressed shard size invalid")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode="rb") as handle:
            raw = handle.read(MAX_UNCOMPRESSED_SHARD_BYTES + 1)
            if handle.read(1):
                raise HuRlReplayResumeV2Error(
                    "uncompressed shard exceeds safety limit"
                )
    except (OSError, EOFError) as exc:
        raise HuRlReplayResumeV2Error("shard gzip is invalid") from exc
    if len(raw) > MAX_UNCOMPRESSED_SHARD_BYTES:
        raise HuRlReplayResumeV2Error("uncompressed shard exceeds safety limit")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlReplayResumeV2Error("shard payload is invalid JSON") from exc
    if not isinstance(value, dict) or raw != (_canonical_json(value) + "\n").encode("ascii"):
        raise HuRlReplayResumeV2Error("shard payload is not canonical JSON")
    return value


def _write_bytes_atomic_create_only(path: Path, payload: bytes) -> None:
    if path.exists():
        raise HuRlReplayResumeV2Error(f"artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise HuRlReplayResumeV2Error(f"artifact already exists: {path.name}") from exc
        _fsync_parent_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_parent_directory(directory: Path) -> None:
    """Make the create-only directory entry durable before later checkpoints.

    The Spot runtime is Linux, where fsync on the containing directory is the
    durability boundary for a newly linked shard/checkpoint name.  Windows
    does not expose a portable directory fsync through ``os.open``; local
    Windows runs retain the atomic create-only behavior while the production
    Linux path fails closed if the directory cannot be synced.
    """

    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        raise HuRlReplayResumeV2Error(
            f"artifact parent directory cannot be opened durably: {directory.name}"
        ) from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise HuRlReplayResumeV2Error(
            f"artifact parent directory cannot be synced durably: {directory.name}"
        ) from exc
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_atomic_create_only(path, (_canonical_json(value) + "\n").encode("ascii"))


def _durability_contract() -> dict[str, Any]:
    posix_parent_fsync = os.name == "posix"
    return {
        "mode": (
            "posix_parent_directory_fsync"
            if posix_parent_fsync
            else "local_process_interruption_only"
        ),
        "parent_directory_fsync_enabled": posix_parent_fsync,
        "hard_power_loss_primitives_enabled": posix_parent_fsync,
        "fault_injection_evaluated": False,
        "production_spot_eligible": posix_parent_fsync,
    }


def _resync_artifact_directories(output_dir: Path) -> None:
    """Close any prior post-link fsync failure before accepting the prefix."""

    for directory in (
        output_dir,
        output_dir / "shards",
        output_dir / "checkpoints",
    ):
        if not directory.is_dir():
            raise HuRlReplayResumeV2Error(
                f"replay artifact directory is unavailable: {directory.name}"
            )
        _fsync_parent_directory(directory)


def _validate_durability_contract(value: Any) -> None:
    known = (
        {
            "mode": "posix_parent_directory_fsync",
            "parent_directory_fsync_enabled": True,
            "hard_power_loss_primitives_enabled": True,
            "fault_injection_evaluated": False,
            "production_spot_eligible": True,
        },
        {
            "mode": "local_process_interruption_only",
            "parent_directory_fsync_enabled": False,
            "hard_power_loss_primitives_enabled": False,
            "fault_injection_evaluated": False,
            "production_spot_eligible": False,
        },
    )
    if value not in known:
        raise HuRlReplayResumeV2Error("durability contract is unknown or overclaims safety")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HuRlReplayResumeV2Error(f"required artifact is unavailable: {path.name}") from exc
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlReplayResumeV2Error(f"artifact is invalid JSON: {path.name}") from exc
    if not isinstance(value, dict) or raw != (_canonical_json(value) + "\n").encode("ascii"):
        raise HuRlReplayResumeV2Error(f"artifact is not canonical JSON: {path.name}")
    return value


def _shard_path(output_dir: Path, index: int) -> Path:
    return output_dir / "shards" / f"shard-{index:06d}.json.gz"


def _checkpoint_path(output_dir: Path, completed: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint-{completed:06d}.json"


def _completed_pair_count(manifest: Mapping[str, Any], completed: int) -> int:
    return min(manifest["paired_hand_count"], completed * manifest["shard_pair_count"])


def _completed_record_count(manifest: Mapping[str, Any], completed: int) -> int:
    return _completed_pair_count(manifest, completed) * RECORDS_PER_PAIR


def _validate_id(value: Any, field: str, limit: int) -> None:
    try:
        encoded = value.encode("ascii", "strict") if isinstance(value, str) else b""
    except UnicodeEncodeError:
        encoded = b""
    if not isinstance(value, str) or not _ID_RE.fullmatch(value) or not encoded or len(encoded) > limit:
        raise HuRlReplayResumeV2Error(f"{field} is invalid")


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = None
    return _sha256(_canonical_json(payload).encode("ascii"))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HuRlReplayResumeV2Error("value is not canonical JSON") from exc


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise HuRlReplayResumeV2Error(f"{field} is not base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise HuRlReplayResumeV2Error(f"{field} is invalid base64") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HuRlReplayResumeV2Error(f"artifact cannot be hashed: {path.name}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_exact_fields(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise HuRlReplayResumeV2Error(f"{context} fields are invalid")


def _strict_int(value: Any) -> bool:
    return type(value) is int


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
