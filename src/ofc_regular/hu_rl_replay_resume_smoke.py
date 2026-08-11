"""Bounded cross-process deterministic replay/recovery smoke for HU RL.

This is not a serializable production checkpoint.  A fresh environment is
rebuilt from the same deterministic seed namespace and a private prefix of
semantic ActionKeys is replayed fail-closed.  The public receipt stores only
commitments/digests and actor-safe aggregate evidence; it never stores a deck,
deck tail, opponent private discard, or raw ActionKey.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .action_key import ActionKey
from .hu_rl_native import (
    MAX_BATCH_THREADS,
    NativeBatchHuRlEnvV1,
    PACKED_ACTION_BYTES,
    PACKED_STEP_RECORD_BYTES,
)
from .hu_rl_native_benchmark import (
    collect_native_benchmark_provenance,
    generate_benchmark_decks,
)


HU_RL_REPLAY_RESUME_SMOKE_SCHEMA: Final = (
    "regular_ofc_hu_rl_deterministic_replay_resume_smoke_v1"
)
HU_RL_REPLAY_RESUME_REQUEST_SCHEMA: Final = (
    "regular_ofc_hu_rl_deterministic_replay_resume_request_v1"
)
HU_RL_REPLAY_RESUME_WORKER_RESULT_SCHEMA: Final = (
    "regular_ofc_hu_rl_deterministic_replay_resume_worker_result_v1"
)
HU_RL_REPLAY_RESUME_PROVENANCE_SCHEMA: Final = (
    "regular_ofc_hu_rl_deterministic_replay_resume_provenance_v1"
)
SEED_NAMESPACE: Final = "regular_ofc_hu_rl_replay_resume_smoke_v1"
DEFAULT_RUN_SEED: Final = 2026072201
DEFAULT_LANE_COUNT: Final = 100
DEFAULT_PREFIX_DECISIONS: Final = 5
DECISIONS_PER_HAND: Final = 10
MIN_LANES: Final = 100
MAX_LANES: Final = 1_000
MAX_WORKER_REQUEST_BYTES: Final = 1_000_000
MAX_WORKER_OUTPUT_BYTES: Final = 2_000_000
WORKER_TIMEOUT_SECONDS: Final = 180

_NEW_SOURCE_PATHS: Final = (
    "scripts/run_hu_rl_replay_resume_smoke.py",
    "src/ofc_regular/hu_rl_replay_resume_smoke.py",
)
_REQUIRED_SOURCE_PATHS: Final = {
    "rust/hu_rl_engine/Cargo.toml",
    "rust/hu_rl_engine/src/batch.rs",
    "rust/hu_rl_engine/src/history.rs",
    "rust/hu_rl_engine/src/lib.rs",
    "rust/hu_rl_engine/src/observation.rs",
    "rust/hu_rl_engine/src/python.rs",
    "rust/hu_rl_engine/src/transition.rs",
    "rust/hu_rl_engine/src/world.rs",
    "scripts/benchmark_hu_rl_native_batch.py",
    "scripts/run_hu_rl_replay_resume_smoke.py",
    "src/ofc_regular/ai_profiles.py",
    "src/ofc_regular/hu_rl_native.py",
    "src/ofc_regular/hu_rl_native_benchmark.py",
    "src/ofc_regular/hu_rl_replay_resume_smoke.py",
}
_RECEIPT_FIELDS = {
    "schema",
    "status",
    "artifact_role",
    "scope",
    "resume_mechanism",
    "subprocess_boundary_exercised",
    "production_checkpoint_eligible",
    "serializable_snapshot_claimed",
    "cross_process_snapshot_claimed",
    "performance_gate_evaluated",
    "performance_gate_pass",
    "lane_count",
    "chunk_width",
    "thread_count",
    "decision_count_per_lane",
    "prefix_decisions",
    "seed_namespace",
    "seed_commitment_sha256",
    "deck_commitment_sha256",
    "prefix_action_history_sha256",
    "prefix_replay_checkpoint",
    "remaining_public_step_sha256",
    "terminal_rewards_sha256",
    "public_aggregate_sha256",
    "decision_counts_sha256",
    "byte_identical",
    "provenance",
    "runtime",
    "receipt_sha256",
}
_CHECKPOINT_FIELDS = {
    "packed_observation_sha256",
    "packed_action_keys_sha256",
    "packed_mask_sha256",
    "packed_action_counts_sha256",
    "packed_action_set_digests_sha256",
    "packed_action_order_digests_sha256",
}
_PROVENANCE_FIELDS = {
    "schema",
    "package_name",
    "package_version",
    "wheel_filename",
    "wheel_sha256",
    "native_extension_filename",
    "native_extension_sha256",
    "source_sha256",
}
_RUNTIME_FIELDS = {
    "python_version",
    "python_implementation",
    "platform",
    "machine",
    "processor",
    "cpu_count",
}
_REQUEST_FIELDS = {
    "schema",
    "seed_namespace",
    "run_seed",
    "seed_commitment_sha256",
    "deck_commitment_sha256",
    "lane_count",
    "chunk_width",
    "thread_count",
    "prefix_decisions",
    "prefix_actions",
    "prefix_action_history_sha256",
    "provenance_sha256",
}
_PREFIX_RECORD_FIELDS = {"decision_ordinal", "action_tokens"}
_WORKER_RESULT_FIELDS = {
    "schema",
    "lane_count",
    "prefix_decisions",
    "checkpoint",
    "remaining_public_steps_b64",
    "terminal_rewards_b64",
    "public_aggregate_b64",
    "decision_counts_b64",
    "all_done",
    "provenance_sha256",
}


class HuRlReplayResumeSmokeError(ValueError):
    """Replay/resume request, worker result, or receipt failed closed."""


WorkerRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_deterministic_replay_resume_smoke(
    *,
    run_seed: int = DEFAULT_RUN_SEED,
    lane_count: int = DEFAULT_LANE_COUNT,
    prefix_decisions: int = DEFAULT_PREFIX_DECISIONS,
    chunk_width: int = 32,
    thread_count: int = 8,
    _worker_runner: WorkerRunner | None = None,
) -> dict[str, Any]:
    """Run one bounded uninterrupted-vs-fresh-process deterministic pilot."""

    _validate_config(
        run_seed=run_seed,
        lane_count=lane_count,
        prefix_decisions=prefix_decisions,
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    provenance = collect_replay_resume_provenance()
    provenance_sha256 = _canonical_digest(provenance)
    decks = generate_benchmark_decks(lane_count=lane_count, seed=run_seed)
    deck_commitment = _deck_commitment(decks)
    seed_commitment = _seed_commitment(run_seed)

    uninterrupted, prefix_actions = _execute_full_hand(
        decks=decks,
        run_seed=run_seed,
        prefix_decisions=prefix_decisions,
        chunk_width=chunk_width,
        thread_count=thread_count,
        supplied_prefix=None,
        provenance_sha256=provenance_sha256,
    )
    prefix_digest = _prefix_action_digest(prefix_actions)
    request = {
        "schema": HU_RL_REPLAY_RESUME_REQUEST_SCHEMA,
        "seed_namespace": SEED_NAMESPACE,
        "run_seed": run_seed,
        "seed_commitment_sha256": seed_commitment,
        "deck_commitment_sha256": deck_commitment,
        "lane_count": lane_count,
        "chunk_width": chunk_width,
        "thread_count": thread_count,
        "prefix_decisions": prefix_decisions,
        "prefix_actions": prefix_actions,
        "prefix_action_history_sha256": prefix_digest,
        "provenance_sha256": provenance_sha256,
    }
    validate_replay_resume_request(request)
    worker_runner = _worker_runner or _run_resume_worker_subprocess
    resumed = dict(worker_runner(request))
    validate_replay_resume_worker_result(
        resumed,
        lane_count=lane_count,
        prefix_decisions=prefix_decisions,
        provenance_sha256=provenance_sha256,
    )
    if resumed != uninterrupted:
        raise HuRlReplayResumeSmokeError(
            "fresh-process replay did not reproduce uninterrupted execution"
        )
    if collect_replay_resume_provenance() != provenance:
        raise HuRlReplayResumeSmokeError(
            "source/wheel provenance changed during replay resume smoke"
        )

    summary = _summarize_worker_result(uninterrupted)
    receipt: dict[str, Any] = {
        "schema": HU_RL_REPLAY_RESUME_SMOKE_SCHEMA,
        "status": "deterministic_replay_resume_smoke_passed",
        "artifact_role": "deterministic_replay_resume_smoke",
        "scope": "bounded_100_to_1000_lane_local_pilot",
        "resume_mechanism": "fresh_env_seed_and_prefix_action_replay",
        "subprocess_boundary_exercised": _worker_runner is None,
        "production_checkpoint_eligible": False,
        "serializable_snapshot_claimed": False,
        "cross_process_snapshot_claimed": False,
        "performance_gate_evaluated": False,
        "performance_gate_pass": None,
        "lane_count": lane_count,
        "chunk_width": chunk_width,
        "thread_count": thread_count,
        "decision_count_per_lane": DECISIONS_PER_HAND,
        "prefix_decisions": prefix_decisions,
        "seed_namespace": SEED_NAMESPACE,
        "seed_commitment_sha256": seed_commitment,
        "deck_commitment_sha256": deck_commitment,
        "prefix_action_history_sha256": prefix_digest,
        **summary,
        "byte_identical": True,
        "provenance": provenance,
        "runtime": _runtime_identity(),
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    validate_replay_resume_receipt(
        receipt,
        require_subprocess_boundary=_worker_runner is None,
    )
    return receipt


def evaluate_replay_resume_worker_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Private worker entrypoint; raw prefix actions never enter the receipt."""

    validate_replay_resume_request(request)
    provenance = collect_replay_resume_provenance()
    provenance_sha256 = _canonical_digest(provenance)
    if provenance_sha256 != request["provenance_sha256"]:
        raise HuRlReplayResumeSmokeError("worker source/wheel provenance mismatch")
    run_seed = request["run_seed"]
    decks = generate_benchmark_decks(
        lane_count=request["lane_count"],
        seed=run_seed,
    )
    if (
        _seed_commitment(run_seed) != request["seed_commitment_sha256"]
        or _deck_commitment(decks) != request["deck_commitment_sha256"]
    ):
        raise HuRlReplayResumeSmokeError("worker seed/deck commitment mismatch")
    result, _ = _execute_full_hand(
        decks=decks,
        run_seed=run_seed,
        prefix_decisions=request["prefix_decisions"],
        chunk_width=request["chunk_width"],
        thread_count=request["thread_count"],
        supplied_prefix=request["prefix_actions"],
        provenance_sha256=provenance_sha256,
    )
    validate_replay_resume_worker_result(
        result,
        lane_count=request["lane_count"],
        prefix_decisions=request["prefix_decisions"],
        provenance_sha256=provenance_sha256,
    )
    return result


def validate_replay_resume_request(request: Mapping[str, Any]) -> None:
    _require_mapping(request, "replay resume request")
    _require_exact_fields(request, _REQUEST_FIELDS, "replay resume request")
    if request["schema"] != HU_RL_REPLAY_RESUME_REQUEST_SCHEMA:
        raise HuRlReplayResumeSmokeError("unsupported replay resume request schema")
    if request["seed_namespace"] != SEED_NAMESPACE:
        raise HuRlReplayResumeSmokeError("replay resume seed namespace mismatch")
    _validate_config(
        run_seed=request["run_seed"],
        lane_count=request["lane_count"],
        prefix_decisions=request["prefix_decisions"],
        chunk_width=request["chunk_width"],
        thread_count=request["thread_count"],
    )
    for field in (
        "seed_commitment_sha256",
        "deck_commitment_sha256",
        "prefix_action_history_sha256",
        "provenance_sha256",
    ):
        if not _is_sha256(request[field]):
            raise HuRlReplayResumeSmokeError(f"replay resume {field} is invalid")
    if _seed_commitment(request["run_seed"]) != request["seed_commitment_sha256"]:
        raise HuRlReplayResumeSmokeError("replay resume seed commitment mismatch")
    prefix = request["prefix_actions"]
    if isinstance(prefix, (str, bytes)) or not isinstance(prefix, Sequence):
        raise HuRlReplayResumeSmokeError("replay resume prefix must be a sequence")
    if len(prefix) != request["prefix_decisions"]:
        raise HuRlReplayResumeSmokeError("replay resume prefix length mismatch")
    seen_ordinals: set[int] = set()
    for expected_ordinal, record in enumerate(prefix):
        _require_mapping(record, "replay resume prefix record")
        _require_exact_fields(record, _PREFIX_RECORD_FIELDS, "replay resume prefix record")
        ordinal = record["decision_ordinal"]
        if not _strict_int(ordinal) or ordinal in seen_ordinals:
            raise HuRlReplayResumeSmokeError("replay resume prefix ordinal is duplicated")
        if ordinal != expected_ordinal:
            raise HuRlReplayResumeSmokeError("replay resume prefix ordinal is missing or reordered")
        seen_ordinals.add(ordinal)
        tokens = record["action_tokens"]
        if (
            isinstance(tokens, (str, bytes))
            or not isinstance(tokens, Sequence)
            or len(tokens) != request["lane_count"]
        ):
            raise HuRlReplayResumeSmokeError("replay resume prefix lane geometry mismatch")
        for token in tokens:
            if type(token) is not str:
                raise HuRlReplayResumeSmokeError("replay resume ActionKey type is invalid")
            try:
                ActionKey.from_token(token)
            except (TypeError, ValueError) as exc:
                raise HuRlReplayResumeSmokeError(
                    "replay resume ActionKey encoding is invalid"
                ) from exc
    if _prefix_action_digest(prefix) != request["prefix_action_history_sha256"]:
        raise HuRlReplayResumeSmokeError("replay resume prefix ActionKey digest mismatch")


def validate_replay_resume_worker_result(
    result: Mapping[str, Any],
    *,
    lane_count: int,
    prefix_decisions: int,
    provenance_sha256: str,
) -> None:
    _require_mapping(result, "replay resume worker result")
    _require_exact_fields(result, _WORKER_RESULT_FIELDS, "replay resume worker result")
    if result["schema"] != HU_RL_REPLAY_RESUME_WORKER_RESULT_SCHEMA:
        raise HuRlReplayResumeSmokeError("unsupported replay resume worker result schema")
    if result["lane_count"] != lane_count or not _strict_int(result["lane_count"]):
        raise HuRlReplayResumeSmokeError("worker lane count mismatch")
    if (
        result["prefix_decisions"] != prefix_decisions
        or not _strict_int(result["prefix_decisions"])
    ):
        raise HuRlReplayResumeSmokeError("worker prefix decision count mismatch")
    if result["provenance_sha256"] != provenance_sha256:
        raise HuRlReplayResumeSmokeError("worker provenance mismatch")
    if result["all_done"] is not True:
        raise HuRlReplayResumeSmokeError("worker did not reach terminal state")
    checkpoint = result["checkpoint"]
    _require_mapping(checkpoint, "worker replay checkpoint")
    _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, "worker replay checkpoint")
    if any(not _is_sha256(value) for value in checkpoint.values()):
        raise HuRlReplayResumeSmokeError("worker checkpoint digest is invalid")
    remaining = result["remaining_public_steps_b64"]
    if (
        isinstance(remaining, (str, bytes))
        or not isinstance(remaining, Sequence)
        or len(remaining) != DECISIONS_PER_HAND - prefix_decisions
    ):
        raise HuRlReplayResumeSmokeError("worker remaining-step geometry mismatch")
    if any(
        len(_decode_b64(value)) != lane_count * PACKED_STEP_RECORD_BYTES
        for value in remaining
    ):
        raise HuRlReplayResumeSmokeError("worker remaining-step bytes are invalid")
    if len(_decode_b64(result["terminal_rewards_b64"])) != lane_count * 16:
        raise HuRlReplayResumeSmokeError("worker terminal reward bytes are invalid")
    if len(_decode_b64(result["public_aggregate_b64"])) != (
        lane_count * PACKED_STEP_RECORD_BYTES * DECISIONS_PER_HAND
    ):
        raise HuRlReplayResumeSmokeError("worker public aggregate bytes are invalid")
    if len(_decode_b64(result["decision_counts_b64"])) != lane_count * 2:
        raise HuRlReplayResumeSmokeError("worker decision-count bytes are invalid")


def validate_replay_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    require_subprocess_boundary: bool = True,
) -> None:
    _require_mapping(receipt, "replay resume receipt")
    _require_exact_fields(receipt, _RECEIPT_FIELDS, "replay resume receipt")
    expected_literals = {
        "schema": HU_RL_REPLAY_RESUME_SMOKE_SCHEMA,
        "status": "deterministic_replay_resume_smoke_passed",
        "artifact_role": "deterministic_replay_resume_smoke",
        "scope": "bounded_100_to_1000_lane_local_pilot",
        "resume_mechanism": "fresh_env_seed_and_prefix_action_replay",
        "production_checkpoint_eligible": False,
        "serializable_snapshot_claimed": False,
        "cross_process_snapshot_claimed": False,
        "performance_gate_evaluated": False,
        "performance_gate_pass": None,
        "decision_count_per_lane": DECISIONS_PER_HAND,
        "seed_namespace": SEED_NAMESPACE,
        "byte_identical": True,
    }
    for field, expected in expected_literals.items():
        if receipt[field] != expected or (
            isinstance(expected, bool) and type(receipt[field]) is not bool
        ):
            raise HuRlReplayResumeSmokeError(f"replay resume receipt {field} is invalid")
    if type(receipt["subprocess_boundary_exercised"]) is not bool:
        raise HuRlReplayResumeSmokeError("replay resume subprocess flag is invalid")
    if require_subprocess_boundary and receipt["subprocess_boundary_exercised"] is not True:
        raise HuRlReplayResumeSmokeError("replay resume subprocess boundary was not exercised")
    _validate_config(
        run_seed=0,
        lane_count=receipt["lane_count"],
        prefix_decisions=receipt["prefix_decisions"],
        chunk_width=receipt["chunk_width"],
        thread_count=receipt["thread_count"],
    )
    for field in (
        "seed_commitment_sha256",
        "deck_commitment_sha256",
        "prefix_action_history_sha256",
        "terminal_rewards_sha256",
        "public_aggregate_sha256",
        "decision_counts_sha256",
    ):
        if not _is_sha256(receipt[field]):
            raise HuRlReplayResumeSmokeError(f"replay resume receipt {field} is invalid")
    checkpoint = receipt["prefix_replay_checkpoint"]
    _require_mapping(checkpoint, "receipt replay checkpoint")
    _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, "receipt replay checkpoint")
    if any(not _is_sha256(value) for value in checkpoint.values()):
        raise HuRlReplayResumeSmokeError("receipt replay checkpoint digest is invalid")
    remaining = receipt["remaining_public_step_sha256"]
    if (
        isinstance(remaining, (str, bytes))
        or not isinstance(remaining, Sequence)
        or len(remaining) != DECISIONS_PER_HAND - receipt["prefix_decisions"]
        or any(not _is_sha256(value) for value in remaining)
    ):
        raise HuRlReplayResumeSmokeError("receipt remaining public-step digests are invalid")
    _validate_replay_provenance(receipt["provenance"])
    runtime = receipt["runtime"]
    _require_mapping(runtime, "replay resume runtime")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, "replay resume runtime")
    for field in (
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "processor",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise HuRlReplayResumeSmokeError(f"replay resume runtime {field} is invalid")
    if not _strict_int(runtime["cpu_count"]) or runtime["cpu_count"] <= 0:
        raise HuRlReplayResumeSmokeError("replay resume runtime cpu_count is invalid")
    if (
        not _is_sha256(receipt["receipt_sha256"])
        or receipt["receipt_sha256"] != _receipt_digest(receipt)
    ):
        raise HuRlReplayResumeSmokeError("replay resume receipt digest mismatch")
    encoded = _canonical_json(receipt).lower()
    for forbidden in (
        '"deck_tail"',
        '"opponent_private_discard"',
        '"opponent_private_discards"',
        '"world_state"',
        '"action_tokens"',
        '"run_seed"',
    ):
        if forbidden in encoded:
            raise HuRlReplayResumeSmokeError("replay resume receipt exposed private input")


def canonical_replay_resume_json(receipt: Mapping[str, Any]) -> str:
    validate_replay_resume_receipt(receipt)
    return _canonical_json(receipt)


def collect_replay_resume_provenance() -> dict[str, Any]:
    base = collect_native_benchmark_provenance()
    repository = Path(__file__).resolve().parents[2]
    source_sha256 = dict(base["source_sha256"])
    for relative in _NEW_SOURCE_PATHS:
        path = repository / Path(relative)
        if not path.is_file():
            raise HuRlReplayResumeSmokeError("replay resume source snapshot is incomplete")
        source_sha256[relative] = _sha256_file(path)
    return {
        "schema": HU_RL_REPLAY_RESUME_PROVENANCE_SCHEMA,
        "package_name": base["package_name"],
        "package_version": base["package_version"],
        "wheel_filename": base["wheel_filename"],
        "wheel_sha256": base["wheel_sha256"],
        "native_extension_filename": base["native_extension_filename"],
        "native_extension_sha256": base["native_extension_sha256"],
        "source_sha256": source_sha256,
    }


def run_resume_worker_stdio() -> int:
    """Read one bounded private request from stdin and emit one result."""

    raw = sys.stdin.buffer.read(MAX_WORKER_REQUEST_BYTES + 1)
    if len(raw) > MAX_WORKER_REQUEST_BYTES:
        raise HuRlReplayResumeSmokeError("worker request exceeds byte limit")
    try:
        request = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlReplayResumeSmokeError("worker request is not canonical JSON") from exc
    result = evaluate_replay_resume_worker_request(request)
    encoded = _canonical_json(result)
    if len(encoded.encode("ascii")) > MAX_WORKER_OUTPUT_BYTES:
        raise HuRlReplayResumeSmokeError("worker result exceeds byte limit")
    print(encoded)
    return 0


def _execute_full_hand(
    *,
    decks: Sequence[Sequence[str]],
    run_seed: int,
    prefix_decisions: int,
    chunk_width: int,
    thread_count: int,
    supplied_prefix: Sequence[Mapping[str, Any]] | None,
    provenance_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lane_count = len(decks)
    env = NativeBatchHuRlEnvV1(
        decks,
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    prefix_records: list[dict[str, Any]] = []
    public_steps: list[bytes] = []
    remaining_steps: list[bytes] = []
    checkpoint: dict[str, str] | None = None
    final_step = None
    for decision in range(DECISIONS_PER_HAND):
        actor_decision = env.actor_decision_batch_packed()
        legal = actor_decision.legal_actions
        if decision == prefix_decisions:
            checkpoint = {
                "packed_observation_sha256": _sha256(
                    actor_decision.observations.payload
                ),
                "packed_action_keys_sha256": _sha256(legal.action_keys),
                "packed_mask_sha256": _sha256(legal.mask),
                "packed_action_counts_sha256": _sha256(legal.action_counts),
                "packed_action_set_digests_sha256": _sha256(
                    legal.action_set_digests
                ),
                "packed_action_order_digests_sha256": _sha256(
                    legal.action_order_digests
                ),
            }
        if decision < prefix_decisions and supplied_prefix is not None:
            selected = _pack_action_tokens(
                supplied_prefix[decision]["action_tokens"],
                lane_count=lane_count,
            )
        else:
            indices = tuple(
                _policy_index(
                    run_seed=run_seed,
                    lane=lane,
                    decision=decision,
                    action_count=legal.action_count(lane),
                )
                for lane in range(lane_count)
            )
            selected = legal.select(indices)
        if decision < prefix_decisions:
            prefix_records.append(
                {
                    "decision_ordinal": decision,
                    "action_tokens": _unpack_action_tokens(
                        selected,
                        lane_count=lane_count,
                    ),
                }
            )
        try:
            final_step = env.step_batch_packed(selected)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HuRlReplayResumeSmokeError("prefix replay or remaining step failed") from exc
        public_steps.append(final_step.payload)
        if decision >= prefix_decisions:
            remaining_steps.append(final_step.payload)
    if checkpoint is None or final_step is None or env.all_done is not True:
        raise HuRlReplayResumeSmokeError("full hand did not produce checkpoint and terminal state")
    if any(count != DECISIONS_PER_HAND for count in env.decision_counts):
        raise HuRlReplayResumeSmokeError("full hand decision counts changed")
    terminal_rewards = b"".join(
        struct.pack("<2d", *final_step.lane(lane).rewards)
        for lane in range(lane_count)
    )
    decision_counts = b"".join(
        struct.pack("<H", count) for count in env.decision_counts
    )
    result = {
        "schema": HU_RL_REPLAY_RESUME_WORKER_RESULT_SCHEMA,
        "lane_count": lane_count,
        "prefix_decisions": prefix_decisions,
        "checkpoint": checkpoint,
        "remaining_public_steps_b64": [_encode_b64(step) for step in remaining_steps],
        "terminal_rewards_b64": _encode_b64(terminal_rewards),
        "public_aggregate_b64": _encode_b64(b"".join(public_steps)),
        "decision_counts_b64": _encode_b64(decision_counts),
        "all_done": True,
        "provenance_sha256": provenance_sha256,
    }
    return result, prefix_records


def _summarize_worker_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prefix_replay_checkpoint": dict(result["checkpoint"]),
        "remaining_public_step_sha256": [
            _sha256(_decode_b64(value))
            for value in result["remaining_public_steps_b64"]
        ],
        "terminal_rewards_sha256": _sha256(
            _decode_b64(result["terminal_rewards_b64"])
        ),
        "public_aggregate_sha256": _sha256(
            _decode_b64(result["public_aggregate_b64"])
        ),
        "decision_counts_sha256": _sha256(
            _decode_b64(result["decision_counts_b64"])
        ),
    }


def _run_resume_worker_subprocess(request: Mapping[str, Any]) -> Mapping[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts" / "run_hu_rl_replay_resume_smoke.py"
    encoded = _canonical_json(request).encode("ascii")
    if len(encoded) > MAX_WORKER_REQUEST_BYTES:
        raise HuRlReplayResumeSmokeError("worker request exceeds byte limit")
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--resume-worker"],
            input=encoded,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repository,
            check=False,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HuRlReplayResumeSmokeError("resume worker process failed") from exc
    if (
        completed.returncode != 0
        or not completed.stdout
        or len(completed.stdout) > MAX_WORKER_OUTPUT_BYTES
    ):
        raise HuRlReplayResumeSmokeError("resume worker process failed closed")
    try:
        result = json.loads(completed.stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlReplayResumeSmokeError("resume worker output is invalid") from exc
    _require_mapping(result, "resume worker output")
    return result


def _validate_replay_provenance(value: object) -> None:
    _require_mapping(value, "replay resume provenance")
    _require_exact_fields(value, _PROVENANCE_FIELDS, "replay resume provenance")
    if value["schema"] != HU_RL_REPLAY_RESUME_PROVENANCE_SCHEMA:
        raise HuRlReplayResumeSmokeError("replay resume provenance schema is invalid")
    for field in ("package_name", "package_version"):
        if not isinstance(value[field], str) or not value[field]:
            raise HuRlReplayResumeSmokeError(f"replay resume provenance {field} is invalid")
    for field in ("wheel_filename", "native_extension_filename"):
        filename = value[field]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise HuRlReplayResumeSmokeError(f"replay resume provenance {field} is invalid")
    for field in ("wheel_sha256", "native_extension_sha256"):
        if not _is_sha256(value[field]):
            raise HuRlReplayResumeSmokeError(f"replay resume provenance {field} is invalid")
    hashes = value["source_sha256"]
    _require_mapping(hashes, "replay resume source hashes")
    if set(hashes) != _REQUIRED_SOURCE_PATHS or any(
        not _is_sha256(digest) for digest in hashes.values()
    ):
        raise HuRlReplayResumeSmokeError("replay resume source hashes are invalid")


def _validate_config(
    *,
    run_seed: object,
    lane_count: object,
    prefix_decisions: object,
    chunk_width: object,
    thread_count: object,
) -> None:
    if not _strict_int(run_seed) or not 0 <= run_seed <= (1 << 63) - 1:
        raise HuRlReplayResumeSmokeError("run seed is outside unsigned 63-bit domain")
    if not _strict_int(lane_count) or not MIN_LANES <= lane_count <= MAX_LANES:
        raise HuRlReplayResumeSmokeError("lane count is outside bounded pilot range")
    if (
        not _strict_int(prefix_decisions)
        or not 1 <= prefix_decisions < DECISIONS_PER_HAND
    ):
        raise HuRlReplayResumeSmokeError("prefix decision count is invalid")
    if not _strict_int(chunk_width) or chunk_width <= 0:
        raise HuRlReplayResumeSmokeError("chunk width is invalid")
    if (
        not _strict_int(thread_count)
        or not 1 <= thread_count <= MAX_BATCH_THREADS
    ):
        raise HuRlReplayResumeSmokeError("thread count is invalid")


def _policy_index(*, run_seed: int, lane: int, decision: int, action_count: int) -> int:
    mixed = (
        run_seed
        ^ ((lane + 1) * 0x9E3779B97F4A7C15)
        ^ ((decision + 1) * 0xBF58476D1CE4E5B9)
    ) & ((1 << 64) - 1)
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    return mixed % action_count


def _pack_action_tokens(tokens: Sequence[str], *, lane_count: int) -> bytes:
    if len(tokens) != lane_count:
        raise HuRlReplayResumeSmokeError("prefix ActionKey lane count mismatch")
    output = bytearray(lane_count * PACKED_ACTION_BYTES)
    for lane, token in enumerate(tokens):
        try:
            key = ActionKey.from_token(token)
        except (TypeError, ValueError) as exc:
            raise HuRlReplayResumeSmokeError("prefix ActionKey encoding is invalid") from exc
        struct.pack_into("<4Q", output, lane * PACKED_ACTION_BYTES, *key.masks)
    return bytes(output)


def _unpack_action_tokens(value: bytes, *, lane_count: int) -> list[str]:
    if type(value) is not bytes or len(value) != lane_count * PACKED_ACTION_BYTES:
        raise HuRlReplayResumeSmokeError("selected ActionKey geometry changed")
    return [
        ActionKey(*struct.unpack_from("<4Q", value, lane * PACKED_ACTION_BYTES)).to_token()
        for lane in range(lane_count)
    ]


def _deck_commitment(decks: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for lane, deck in enumerate(decks):
        digest.update(struct.pack("<I", lane))
        digest.update("\n".join(deck).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _seed_commitment(run_seed: int) -> str:
    return _sha256(f"{SEED_NAMESPACE}\0{run_seed}".encode("ascii"))


def _prefix_action_digest(prefix: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_json(prefix).encode("ascii"))


def _runtime_identity() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": (
            platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER")
            or "unknown"
        ),
        "cpu_count": os.cpu_count() or 1,
    }


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload["receipt_sha256"] = None
    return _canonical_digest(payload)


def _canonical_digest(value: object) -> str:
    return _sha256(_canonical_json(value).encode("ascii"))


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HuRlReplayResumeSmokeError("value is not canonical JSON") from exc


def _encode_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_b64(value: object) -> bytes:
    if not isinstance(value, str):
        raise HuRlReplayResumeSmokeError("worker binary field is not base64 text")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise HuRlReplayResumeSmokeError("worker binary field is invalid base64") from exc
    if not decoded:
        raise HuRlReplayResumeSmokeError("worker binary field is empty")
    return decoded


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HuRlReplayResumeSmokeError("source provenance file could not be read") from exc
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        raise HuRlReplayResumeSmokeError(f"{context} must be a mapping")


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise HuRlReplayResumeSmokeError(f"{context} fields are invalid")


def _strict_int(value: object) -> bool:
    return type(value) is int
