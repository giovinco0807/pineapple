"""Validate and merge the complete M3.1 T3 infrastructure canary.

This validator consumes the one accepted Step 6a shard-0 receive directory and
the immutable Step 6b receive tree for shards 1..9.  It does not average shard
summaries: every root and search-task artifact is content-validated and the
500-root integrity, RNG, latency, and memory gates are recomputed globally.

The output is a validation receipt, not teacher training data or match EV.
Nothing is written unless all ten shards pass, and an existing output is never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
)
from .run_hu_m31_t3_step6a_shard import (
    MAX_FIRST_P95_SECONDS,
    MAX_PEAK_RSS_BYTES,
    MAX_SECOND_P95_SECONDS,
    PAIRED_HANDS_PER_SHARD,
    RAYON_THREADS,
    ROOTS_PER_SHARD,
    STEP6A_HEARTBEAT_SCHEMA,
    STEP6A_PARITY_SCHEMA,
    STEP6A_ROOT_TASK_SCHEMA,
    STEP6A_RUN_ID,
    STEP6A_SEARCH_TASK_SCHEMA,
    STEP6A_SUMMARY_SCHEMA,
    WORKERS,
    seed_values,
)
from .validate_hu_m31_t3_convergence import REFERENCE_BUDGET
from .validate_hu_m31_t3_local1000 import _percentile


STEP6B_DONE_SCHEMA = "hu_m31_t3_step6b_done_v1"
STEP6B_PARITY_SCHEMA = "hu_m31_t3_step6b_linux_parity_v1"
STEP6B_RECEIVE_SCHEMA = "hu_m31_t3_step6b_receive_v1"
STEP6B_SUMMARY_SCHEMA = "hu_m31_t3_step6b_summary_v1"
STEP6B_VALIDATION_SCHEMA = "hu_m31_t3_step6b_canary_validation_v1"

TOTAL_SHARDS = 10
TOTAL_PAIRED_HANDS = 250
TOTAL_ROOTS = 500
EXPECTED_NATIVE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
EXPECTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)

# This is the only Step 6a result admitted to the ten-shard canary.  Tests use
# the private core with an explicit synthetic trust anchor; the CLI has no such
# override.
ACCEPTED_STEP6A_IDENTITY: Mapping[str, Any] = {
    "run_name": "regular-hu-m31-step6a-s0-20260717-004",
    "done_sha256": "45928bbd0c1938506e33c7e15fa7053f82436f1da9b445cd58701ff8f2bfe852",
    "summary_sha256": "4cdde3f061ef67354236e3297b7d40511550c7a00daf028bbc45b234c685717d",
    "receive_receipt_sha256": (
        "febe15eccf141d0fe1689fb8bdc5cdbfc7788459d5b8253943307bc41dd621ee"
    ),
    "source_sha256": "224506d27a15f237baaa5c200b4ef41bc2a8c61b817edac4b5a2f9c71d704be8",
    "manifest_sha256": "a2d7cf23ed22fb4bbf3fee6fdecd6df807439117093e92783edb72d4fc665188",
    "schedule_sha256": "19d2d5992d3a08435a6c255368503ddde241fa9704d571e206eafe46dbd39c64",
    "authorization_sha256": (
        "e3b6340885d2002a2b552a6f54ae730c7fc9b6c47f358ee22de74ac09ff02ffc"
    ),
    "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
    "feature_encoder_sha256": EXPECTED_FEATURE_ENCODER_SHA256,
}

_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_HIDDEN_KEYS = frozenset(
    {
        "opponent_private_discards",
        "true_dead_cards",
        "remaining_deck",
        "world_state",
        "replay_truth",
        "draw_pile",
        "future_cards",
        "deck_tail",
        "true_opponent_private_discards",
    }
)
_ROOT_KEYS = frozenset(
    {
        "schema",
        "contract_digest",
        "global_hand_index",
        "profile",
        "seeds",
        "observations",
        "current_profile_resolved",
        "opponent_private_discards_used",
    }
)
_OBSERVATION_ROW_KEYS = frozenset({"seat", "observation_fingerprint", "observation"})
_TASK_KEYS = frozenset(
    {
        "schema",
        "contract_digest",
        "global_hand_index",
        "profile",
        "seeds",
        "rayon_threads",
        "engine",
        "process_id",
        "rows",
        "memory",
        "wall_seconds",
        "all_gates_passed",
    }
)
_TASK_ROW_KEYS = frozenset(
    {
        "root_index",
        "global_hand_index",
        "seat",
        "observation_fingerprint",
        "legal_action_count",
        "wall_seconds",
        "decision",
        "integrity",
    }
)
_ACTION_VALUE_KEYS = frozenset(
    {
        "action_key",
        "original_index",
        "rank",
        "placements",
        "discards",
        "selection_ev",
        "evaluation_ev",
        "evaluation_regret",
    }
)
_INTEGRITY_KEYS = frozenset(
    {
        "observation_fingerprint",
        "seat",
        "budget",
        "exact_t4",
        "finite_values",
        "unique_action_keys",
        "original_index_mapping",
        "selected_action_mapping",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "selected_regret_consistent",
        "candidate_evaluation_rng_domains_distinct",
        "run_id",
        "continuation_seed",
        "candidate_seed",
        "evaluation_seed",
        "native_library_sha256",
        "execution_mode_scalar",
        "downstream_t4_exact",
        "teacher_value_diagnostic",
    }
)
_REQUIRED_REMOTE_FILES = frozenset(
    {
        "drill_stdout.json",
        "heartbeat.json",
        "parity.json",
        "run.log",
        "runner_stdout.json",
        "summary.json",
        "time.txt",
    }
)


@dataclass(frozen=True)
class _ValidatedShard:
    shard: int
    run_name: str
    source_sha256: str
    manifest_sha256: str
    schedule_sha256: str
    authorization_sha256: str
    done_sha256: str
    summary_sha256: str
    task_hashes: Mapping[int, str]
    hand_indices: tuple[int, ...]
    root_indices: tuple[int, ...]
    fingerprints: tuple[str, ...]
    profile_hands: tuple[str, ...]
    seats: tuple[str, ...]
    first_latencies: tuple[float, ...]
    second_latencies: tuple[float, ...]
    peak_rss_bytes: int
    resumed_task_count: int
    seed_values_flat: tuple[int, ...]
    candidate_rng_keys: frozenset[str]
    evaluation_rng_keys: frozenset[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable canonical JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields changed: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _hash_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _reject_hidden(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_HIDDEN_KEYS:
                raise ValueError(f"forbidden hidden-information field at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _relative_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"received shard contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _safe_relative_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("DONE file manifest contains an invalid relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError("DONE file manifest escapes the shard directory")
    path = root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"DONE file is absent or unsafe: {relative}")
    return path


def _validate_done_file_manifest(
    directory: Path,
    *,
    local_extra_files: frozenset[str],
) -> dict[str, Any]:
    done = _read_json(directory / "DONE.json", "DONE marker")
    expected_top_level = (
        set(_REQUIRED_REMOTE_FILES)
        | {"DONE.json", "roots", "tasks"}
        | set(local_extra_files)
    )
    top_level = list(directory.iterdir())
    if {path.name for path in top_level} != expected_top_level:
        raise ValueError("received shard top-level entry set changed")
    if any(path.is_symlink() for path in top_level) or any(
        not (directory / name).is_dir() for name in ("roots", "tasks")
    ):
        raise ValueError("received shard top-level layout is unsafe")
    files = done.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("DONE marker lacks a file manifest")
    expected_payload_files = {
        *(
            f"roots/hand_{index:03d}.json"
            for index in range(
                _integer(done.get("shard"), "DONE shard", minimum=0)
                * PAIRED_HANDS_PER_SHARD,
                (_integer(done.get("shard"), "DONE shard", minimum=0) + 1)
                * PAIRED_HANDS_PER_SHARD,
            )
        ),
        *(
            f"tasks/hand_{index:03d}.json"
            for index in range(
                _integer(done.get("shard"), "DONE shard", minimum=0)
                * PAIRED_HANDS_PER_SHARD,
                (_integer(done.get("shard"), "DONE shard", minimum=0) + 1)
                * PAIRED_HANDS_PER_SHARD,
            )
        ),
        *_REQUIRED_REMOTE_FILES,
    }
    if set(files) != expected_payload_files:
        raise ValueError(
            "DONE file manifest does not exactly cover the frozen shard files"
        )
    for relative, raw_record in files.items():
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "sha256",
            "bytes",
        }:
            raise ValueError(f"DONE file record changed: {relative}")
        path = _safe_relative_file(directory, relative)
        size = _integer(
            raw_record.get("bytes"), f"DONE bytes for {relative}", minimum=0
        )
        expected_hash = _hash_string(
            raw_record.get("sha256"), f"DONE hash for {relative}"
        )
        if path.stat().st_size != size or _sha256(path) != expected_hash:
            raise ValueError(f"received file differs from DONE manifest: {relative}")
    actual = _relative_files(directory)
    expected = set(files) | {"DONE.json"} | set(local_extra_files)
    if actual != expected:
        raise ValueError(
            "received shard file set changed: "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return done


def _latency(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency population must not be empty")
    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "p99_seconds": _percentile(ordered, 0.99),
        "max_seconds": max(ordered),
    }


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _validate_latency_summary(
    claimed: Any, values: Sequence[float], *, label: str
) -> dict[str, float | int]:
    if not isinstance(claimed, Mapping):
        raise ValueError(f"{label} latency summary is missing")
    recomputed = _latency(values)
    if set(claimed) != set(recomputed):
        raise ValueError(f"{label} latency summary fields changed")
    for key, expected in recomputed.items():
        if key == "count":
            if claimed.get(key) != expected:
                raise ValueError(f"{label} latency count changed")
        elif not _same_number(claimed.get(key), expected):
            raise ValueError(f"{label} latency {key} was not recomputed from rows")
    return recomputed


def _particle_evidence(
    observation: ActorObservation, seeds: Mapping[str, Any]
) -> tuple[frozenset[str], frozenset[str], str, str, str, str]:
    candidate = sample_hidden_card_particles(
        observation,
        base_seed=_integer(seeds.get("candidate"), "candidate seed", minimum=0),
        run_id=f"{STEP6A_RUN_ID}:candidate_selection",
        sample_count=REFERENCE_BUDGET.candidate_samples,
    )
    evaluation = sample_hidden_card_particles(
        observation,
        base_seed=_integer(seeds.get("evaluation"), "evaluation seed", minimum=0),
        run_id=f"{STEP6A_RUN_ID}:locked_evaluation",
        sample_count=REFERENCE_BUDGET.evaluation_samples,
    )
    candidate_keys = tuple(particle.rng_key_digest for particle in candidate.particles)
    evaluation_keys = tuple(
        particle.rng_key_digest for particle in evaluation.particles
    )
    return (
        frozenset(candidate_keys),
        frozenset(evaluation_keys),
        candidate.digest(),
        evaluation.digest(),
        _digest(candidate_keys),
        _digest(evaluation_keys),
    )


def _validate_decision(
    decision: Any,
    *,
    observation: ActorObservation,
    seeds: Mapping[str, Any],
    native_library_sha256: str,
) -> tuple[dict[str, bool], frozenset[str], frozenset[str]]:
    if not isinstance(decision, Mapping):
        raise ValueError("search row decision is missing")
    _reject_hidden(decision, "decision")
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    expected_by_index = {
        index: action_key(action).to_token() for index, action in enumerate(legal)
    }
    raw_values = decision.get("action_values")
    if not isinstance(raw_values, list) or len(raw_values) != len(legal):
        raise ValueError("decision action values do not cover every legal action")

    values_by_index: dict[int, str] = {}
    evaluation_by_key: dict[str, float] = {}
    ranks: list[int] = []
    finite_values = True
    tokens_in_output_order: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, Mapping):
            raise ValueError("decision action value must be an object")
        _require_exact_keys(raw, _ACTION_VALUE_KEYS, "action value")
        index = _integer(raw.get("original_index"), "action original_index", minimum=0)
        if index >= len(legal) or index in values_by_index:
            raise ValueError(
                "action original_index mapping is duplicated or outside legal set"
            )
        token = _hashable_action_token(raw.get("action_key"))
        if token != expected_by_index[index]:
            raise ValueError("action key does not match its original legal index")
        payload_token = action_key_from_payload(
            {"placements": raw.get("placements"), "discards": raw.get("discards")}
        ).to_token()
        if payload_token != token:
            raise ValueError("action placements/discards disagree with action key")
        values_by_index[index] = token
        tokens_in_output_order.append(token)
        rank = _integer(raw.get("rank"), "action rank", minimum=0)
        ranks.append(rank)
        selection = _finite(raw.get("selection_ev"), "selection EV")
        evaluation = _finite(raw.get("evaluation_ev"), "evaluation EV")
        regret = _finite(raw.get("evaluation_regret"), "evaluation regret", minimum=0.0)
        finite_values = finite_values and all(
            math.isfinite(number) for number in (selection, evaluation, regret)
        )
        evaluation_by_key[token] = evaluation
    if sorted(ranks) != list(range(len(legal))):
        raise ValueError("action ranks are not a complete permutation")
    canonical_tokens = sorted(
        expected_by_index.values(),
        key=lambda token: ActionKey.from_token(token).sort_key(),
    )
    if tokens_in_output_order != canonical_tokens:
        raise ValueError("action values are not in canonical semantic order")
    best_evaluation = max(evaluation_by_key.values())
    for raw in raw_values:
        if not math.isclose(
            float(raw["evaluation_regret"]),
            best_evaluation - float(raw["evaluation_ev"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("per-action evaluation regret is inconsistent")

    selected_token = _hashable_action_token(decision.get("selected_action_key"))
    selected_payload = decision.get("selected_action")
    if not isinstance(selected_payload, Mapping):
        raise ValueError("selected action payload is missing")
    selected_mapping = (
        action_key_from_payload(selected_payload).to_token() == selected_token
    )
    selected_evaluation = evaluation_by_key.get(selected_token)
    selected_row = next(
        (raw for raw in raw_values if raw.get("action_key") == selected_token), None
    )
    if selected_evaluation is None or selected_row is None:
        raise ValueError("selected action is outside the legal action values")
    if not _same_number(decision.get("selected_evaluation_ev"), selected_evaluation):
        raise ValueError("selected evaluation EV does not match action values")
    if not _same_number(
        decision.get("selected_selection_ev"), selected_row.get("selection_ev")
    ):
        raise ValueError("selected selection EV does not match action values")
    selected_regret = best_evaluation - selected_evaluation
    if not _same_number(decision.get("evaluation_sample_regret"), selected_regret):
        raise ValueError("selected evaluation regret is inconsistent")

    (
        candidate_keys,
        evaluation_keys,
        candidate_belief_digest,
        evaluation_belief_digest,
        candidate_rng_digest,
        evaluation_rng_digest,
    ) = _particle_evidence(observation, seeds)
    checks = {
        "observation_fingerprint": (
            decision.get("observation_fingerprint") == observation.fingerprint()
        ),
        "seat": decision.get("seat") == observation.seat,
        "budget": (
            decision.get("candidate_samples") == REFERENCE_BUDGET.candidate_samples
            and decision.get("evaluation_samples")
            == REFERENCE_BUDGET.evaluation_samples
            and decision.get("downstream_t3_samples")
            == REFERENCE_BUDGET.downstream_t3_samples
        ),
        "exact_t4": decision.get("use_t4_action_cache") is True,
        "finite_values": finite_values,
        "unique_action_keys": len(evaluation_by_key) == len(legal),
        "original_index_mapping": values_by_index == expected_by_index,
        "selected_action_mapping": selected_mapping,
        "legal_action_set_digest": (
            decision.get("legal_action_set_digest") == legal_action_set_digest(legal)
        ),
        "legal_action_order_digest": (
            decision.get("legal_action_order_digest")
            == ordered_action_mapping_digest(legal)
        ),
        "selected_regret_consistent": _same_number(
            decision.get("evaluation_sample_regret"), selected_regret
        ),
        "candidate_evaluation_rng_domains_distinct": not (
            candidate_keys & evaluation_keys
        ),
        "run_id": decision.get("run_id") == STEP6A_RUN_ID,
        "continuation_seed": decision.get("continuation_seed") == seeds.get("child"),
        "candidate_seed": decision.get("candidate_seed") == seeds.get("candidate"),
        "evaluation_seed": decision.get("evaluation_seed") == seeds.get("evaluation"),
        "native_library_sha256": (
            decision.get("native_library_sha256") == native_library_sha256
        ),
        "execution_mode_scalar": (
            decision.get("execution_mode") == "scalar"
            and decision.get("batch_size") == 1
        ),
        "downstream_t4_exact": (
            decision.get("downstream_t4_samples") == 0
            and decision.get("downstream_t4_mode") == "exact"
        ),
        "teacher_value_diagnostic": (
            decision.get("teacher_value_status") == "diagnostic_not_match_EV"
        ),
    }
    if decision.get("action_key_schema") != ACTION_KEY_SCHEMA:
        raise ValueError("decision action-key schema changed")
    if decision.get("candidate_belief_digest") != candidate_belief_digest:
        raise ValueError("candidate belief digest does not match actor observation")
    if decision.get("evaluation_belief_digest") != evaluation_belief_digest:
        raise ValueError("evaluation belief digest does not match actor observation")
    if decision.get("candidate_rng_digest") != candidate_rng_digest:
        raise ValueError("candidate RNG digest does not match frozen RNG domain")
    if decision.get("evaluation_rng_digest") != evaluation_rng_digest:
        raise ValueError("evaluation RNG digest does not match frozen RNG domain")
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise ValueError(f"decision integrity failed: {failed}")
    for field in (
        "search_contract_digest",
        "semantic_result_digest",
        "result_digest",
    ):
        _hash_string(decision.get(field), f"decision {field}")
    return checks, candidate_keys, evaluation_keys


def _hashable_action_token(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("action key must be a string")
    return ActionKey.from_token(value).to_token()


def _validate_root_task(
    value: Mapping[str, Any], *, global_hand_index: int
) -> tuple[dict[str, ActorObservation], tuple[str, ...], Mapping[str, Any]]:
    _require_exact_keys(value, _ROOT_KEYS, "root task")
    _reject_hidden(value.get("observations"), "root.observations")
    expected_seeds = seed_values(global_hand_index)
    expected_profile = behavior_profile_for_index(global_hand_index)
    _require(value.get("schema") == STEP6A_ROOT_TASK_SCHEMA, "root task schema changed")
    _hash_string(value.get("contract_digest"), "root contract digest")
    _require(
        value.get("global_hand_index") == global_hand_index,
        "root global hand index changed",
    )
    _require(value.get("profile") == expected_profile, "root behavior profile changed")
    _require(value.get("seeds") == expected_seeds, "root seed schedule changed")
    _require(
        value.get("current_profile_resolved") is False,
        "root resolved current profile",
    )
    _require(
        value.get("opponent_private_discards_used") is False,
        "root used opponent private discards",
    )
    raw_rows = value.get("observations")
    if not isinstance(raw_rows, list) or len(raw_rows) != 2:
        raise ValueError("root task must contain exactly two seat observations")
    observations: dict[str, ActorObservation] = {}
    fingerprints: list[str] = []
    for expected_seat, raw in zip(("first", "second"), raw_rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("root observation row must be an object")
        _require_exact_keys(raw, _OBSERVATION_ROW_KEYS, "root observation row")
        _require(raw.get("seat") == expected_seat, "root seat order changed")
        payload = raw.get("observation")
        if not isinstance(payload, Mapping):
            raise ValueError("root observation payload is missing")
        _reject_hidden(payload, "root.observation")
        observation = ActorObservation.from_dict(payload)
        _require(
            observation.to_dict() == dict(payload),
            "root observation is not canonical",
        )
        fingerprint = observation.fingerprint()
        _require(
            raw.get("observation_fingerprint") == fingerprint,
            "root observation fingerprint changed",
        )
        _require(observation.seat == expected_seat, "observation seat changed")
        _require(observation.street == "T3", "root is not a T3 observation")
        observations[expected_seat] = observation
        fingerprints.append(fingerprint)
    return observations, tuple(fingerprints), expected_seeds


def _validate_search_task(
    value: Mapping[str, Any],
    *,
    root_task: Mapping[str, Any],
    observations: Mapping[str, ActorObservation],
    global_hand_index: int,
    native_library_sha256: str,
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[float, ...],
    tuple[float, ...],
    int,
    frozenset[str],
    frozenset[str],
]:
    _require_exact_keys(value, _TASK_KEYS, "search task")
    _require(
        value.get("schema") == STEP6A_SEARCH_TASK_SCHEMA, "search task schema changed"
    )
    _require(
        value.get("contract_digest") == root_task.get("contract_digest"),
        "search/root contract digest mismatch",
    )
    _require(
        value.get("global_hand_index") == global_hand_index,
        "search global hand index changed",
    )
    _require(value.get("profile") == root_task.get("profile"), "search profile changed")
    _require(value.get("seeds") == root_task.get("seeds"), "search seed map changed")
    _require(
        value.get("rayon_threads") == str(RAYON_THREADS), "Rayon thread count changed"
    )
    _require(value.get("all_gates_passed") is True, "search task gate failed")
    engine = value.get("engine")
    if not isinstance(engine, Mapping):
        raise ValueError("search engine identity is missing")
    _require(
        engine.get("library_sha256") == native_library_sha256,
        "search engine library hash changed",
    )
    _require(engine.get("build_or_fallback") is False, "search used build/fallback")
    _require(
        engine.get("version") == "ofc_hu_m3_engine/0.1.0", "engine version changed"
    )
    memory = value.get("memory")
    if not isinstance(memory, Mapping):
        raise ValueError("search memory evidence is missing")
    peak_rss = _integer(memory.get("peak_rss_bytes"), "task peak RSS", minimum=0)
    _finite(value.get("wall_seconds"), "task wall seconds", minimum=0.0)
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("search task must contain exactly two seat rows")
    root_indices: list[int] = []
    seats: list[str] = []
    first_latencies: list[float] = []
    second_latencies: list[float] = []
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    stored_fingerprints = {
        str(raw["seat"]): str(raw["observation_fingerprint"])
        for raw in root_task["observations"]
    }
    for offset, (expected_seat, raw) in enumerate(
        zip(("first", "second"), rows, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("search row must be an object")
        _require_exact_keys(raw, _TASK_ROW_KEYS, "search row")
        expected_root = global_hand_index * 2 + offset
        _require(raw.get("root_index") == expected_root, "search root index changed")
        _require(
            raw.get("global_hand_index") == global_hand_index,
            "search row global hand changed",
        )
        _require(raw.get("seat") == expected_seat, "search row seat order changed")
        _require(
            raw.get("observation_fingerprint") == stored_fingerprints[expected_seat],
            "search/root observation fingerprint mismatch",
        )
        observation = observations[expected_seat]
        legal_count = len(
            generate_turn_actions(observation.hero_board, observation.dealt_cards)
        )
        _require(
            raw.get("legal_action_count") == legal_count, "legal action count changed"
        )
        latency = _finite(raw.get("wall_seconds"), "root wall seconds", minimum=0.0)
        checks, row_candidate, row_evaluation = _validate_decision(
            raw.get("decision"),
            observation=observation,
            seeds=root_task["seeds"],
            native_library_sha256=native_library_sha256,
        )
        integrity = raw.get("integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("stored decision integrity is missing")
        _require_exact_keys(integrity, _INTEGRITY_KEYS, "stored decision integrity")
        _require(
            dict(integrity) == checks, "stored decision integrity was not recomputed"
        )
        if candidate_keys & set(row_candidate) or evaluation_keys & set(row_evaluation):
            raise ValueError("RNG keys repeat within a search task")
        candidate_keys.update(row_candidate)
        evaluation_keys.update(row_evaluation)
        root_indices.append(expected_root)
        seats.append(expected_seat)
        (first_latencies if expected_seat == "first" else second_latencies).append(
            latency
        )
    return (
        tuple(root_indices),
        tuple(seats),
        tuple(first_latencies),
        tuple(second_latencies),
        peak_rss,
        frozenset(candidate_keys),
        frozenset(evaluation_keys),
    )


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    shard: int,
    summary_schema: str,
    native_library_sha256: str,
    fingerprints: Sequence[str],
    profiles: Sequence[str],
    first_latencies: Sequence[float],
    second_latencies: Sequence[float],
    peak_rss: int,
    task_hashes: Mapping[int, str],
    candidate_keys: frozenset[str],
    evaluation_keys: frozenset[str],
) -> int:
    _require(summary.get("schema") == summary_schema, "shard summary schema changed")
    _require(summary.get("status") == "pass", "shard summary is not pass")
    _require(summary.get("shard") == shard, "shard summary index changed")
    _require(summary.get("all_gates_passed") is True, "shard summary gate failed")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("shard summary contract is missing")
    _require(
        contract.get("paired_hands") == PAIRED_HANDS_PER_SHARD,
        "paired-hand budget changed",
    )
    _require(contract.get("roots") == ROOTS_PER_SHARD, "root budget changed")
    _require(contract.get("workers") == WORKERS, "worker count changed")
    _require(
        contract.get("rayon_threads_per_worker") == RAYON_THREADS,
        "Rayon budget changed",
    )
    _require(
        contract.get("budget") == REFERENCE_BUDGET.to_dict(), "teacher budget changed"
    )
    _require(contract.get("run_id") == STEP6A_RUN_ID, "teacher run id changed")
    _require(
        summary.get("native_library_sha256") == native_library_sha256,
        "summary native hash changed",
    )
    _hash_string(summary.get("source_package_sha256"), "summary source hash")
    _hash_string(summary.get("manifest_sha256"), "summary manifest hash")
    _require(
        summary.get("training_eligible") is False, "canary became training eligible"
    )
    _require(
        summary.get("teacher_value_status") == "diagnostic_not_match_EV",
        "teacher EV scope changed",
    )
    _require(summary.get("current_profile_changed") is False, "current profile changed")
    _require(
        summary.get("production_fanout_authorized") is False,
        "production fanout authorized",
    )
    _require(summary.get("m31_complete") is False, "M3.1 was marked complete by canary")
    gates = summary.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("shard summary contains a failed or non-boolean gate")
    expected_gate_names = {
        "exactly_50_roots",
        "exactly_25_each_seat",
        "five_profiles_equal_quota",
        "unique_observation_fingerprints",
        "all_task_and_decision_integrity",
        "candidate_evaluation_rng_disjoint",
        "candidate_rng_unique",
        "evaluation_rng_unique",
        "resume_drill_recovered_task",
        "first_p95_within_180_seconds",
        "second_p95_within_6_seconds",
        "peak_rss_within_1_gib",
        "canary_rows_not_training_eligible",
        "no_current_profile_or_production_fanout",
    }
    if summary_schema == STEP6B_SUMMARY_SCHEMA:
        expected_gate_names.update(
            {
                "exact_shard_hand_indices",
                "profiles_follow_frozen_cycle",
                "linux_portable_parity_bound",
            }
        )
        _require(
            summary.get("global_hand_start") == shard * PAIRED_HANDS_PER_SHARD,
            "Step 6b summary global hand start changed",
        )
        _require(
            summary.get("remaining_canary_shards_authorized") is True,
            "Step 6b summary lost bounded canary authorization",
        )
    else:
        expected_gate_names.add("linux_portable_parity")
    if not expected_gate_names.issubset(gates):
        raise ValueError("shard summary omitted a frozen gate")
    integrity = summary.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("shard summary integrity is missing")
    expected_profile_counts = {profile: 5 for profile in M31_T3_BEHAVIOR_PROFILES}
    expected_integrity = {
        "fingerprints": len(fingerprints),
        "unique_fingerprints": len(set(fingerprints)),
        "candidate_rng_keys": len(candidate_keys),
        "evaluation_rng_keys": len(evaluation_keys),
        "candidate_evaluation_overlap": len(candidate_keys & evaluation_keys),
        "profile_counts": expected_profile_counts,
    }
    for key, expected in expected_integrity.items():
        if integrity.get(key) != expected:
            raise ValueError(f"shard summary integrity mismatch: {key}")
    if {
        profile: profiles.count(profile) for profile in M31_T3_BEHAVIOR_PROFILES
    } != expected_profile_counts:
        raise ValueError("shard profile quota changed")
    performance = summary.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError("shard performance summary is missing")
    by_seat = performance.get("latency_by_seat")
    if not isinstance(by_seat, Mapping):
        raise ValueError("shard latency-by-seat summary is missing")
    first = _validate_latency_summary(
        by_seat.get("first"), first_latencies, label="first"
    )
    second = _validate_latency_summary(
        by_seat.get("second"), second_latencies, label="second"
    )
    if first["p95_seconds"] > MAX_FIRST_P95_SECONDS:
        raise ValueError("shard first-seat p95 exceeds 180 seconds")
    if second["p95_seconds"] > MAX_SECOND_P95_SECONDS:
        raise ValueError("shard second-seat p95 exceeds 6 seconds")
    if (
        performance.get("peak_process_rss_bytes") != peak_rss
        or peak_rss > MAX_PEAK_RSS_BYTES
    ):
        raise ValueError("shard peak RSS changed or exceeds 1 GiB")
    manifest = summary.get("task_manifest")
    if not isinstance(manifest, list) or len(manifest) != PAIRED_HANDS_PER_SHARD:
        raise ValueError("summary task manifest is incomplete")
    claimed_hashes: dict[int, str] = {}
    for record in manifest:
        if not isinstance(record, Mapping) or set(record) != {
            "global_hand_index",
            "sha256",
        }:
            raise ValueError("summary task manifest record changed")
        index = _integer(
            record.get("global_hand_index"), "task manifest hand", minimum=0
        )
        if index in claimed_hashes:
            raise ValueError("summary task manifest duplicates a hand")
        claimed_hashes[index] = _hash_string(record.get("sha256"), "task manifest hash")
    if claimed_hashes != dict(task_hashes):
        raise ValueError("summary task manifest does not match received tasks")
    resumed = _integer(
        summary.get("resumed_task_count"), "resumed task count", minimum=0
    )
    if not 1 <= resumed <= PAIRED_HANDS_PER_SHARD:
        raise ValueError("resume drill did not recover a completed task")
    return resumed


def _validate_parity(
    parity: Mapping[str, Any],
    *,
    schema: str,
    native_library_sha256: str,
    source_sha256: str,
    manifest_sha256: str,
) -> None:
    _require(parity.get("schema") == schema, "Linux parity schema changed")
    _require(parity.get("status") == "pass", "Linux parity is not pass")
    _require(parity.get("all_gates_passed") is True, "Linux parity gate failed")
    _require(
        parity.get("native_library_sha256") == native_library_sha256,
        "Linux parity native hash changed",
    )
    _require(
        parity.get("current_profile_changed") is False, "parity changed current profile"
    )
    gates = parity.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("Linux parity contains a failed gate")
    if schema == STEP6B_PARITY_SCHEMA:
        _require(
            parity.get("source_package_sha256") == source_sha256,
            "Step 6b parity source hash changed",
        )
        _require(
            parity.get("manifest_sha256") == manifest_sha256,
            "Step 6b parity manifest hash changed",
        )
        _require(
            parity.get("production_fanout_authorized") is False,
            "Step 6b parity authorized production fanout",
        )


def _validate_shard(
    directory: Path,
    *,
    shard: int,
    done_schema: str,
    parity_schema: str,
    summary_schema: str,
    local_extra_files: frozenset[str],
) -> _ValidatedShard:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"canary shard directory is missing or unsafe: {directory}")
    done = _validate_done_file_manifest(directory, local_extra_files=local_extra_files)
    _require(done.get("schema") == done_schema, f"shard {shard} DONE schema changed")
    _require(done.get("status") == "complete", f"shard {shard} DONE is incomplete")
    _require(done.get("shard") == shard, f"shard {shard} DONE identity changed")
    _require(
        done.get("training_eligible") is False, "DONE made canary training eligible"
    )
    _require(
        done.get("production_fanout_authorized") is False, "DONE authorized production"
    )
    _require(
        done.get("current_profile_changed") is False, "DONE changed current profile"
    )
    _require(done.get("resume_drill_passed") is True, "DONE lacks resume evidence")
    completed = _finite(
        done.get("completed_unix_seconds"), "DONE completion time", minimum=0.0
    )
    _require(completed >= 0.0, "DONE completion time changed")
    run_name = done.get("run_name")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("DONE run name is missing")
    source_sha = _hash_string(done.get("source_sha256"), "DONE source hash")
    manifest_sha = _hash_string(done.get("manifest_sha256"), "DONE manifest hash")
    schedule_sha = _hash_string(done.get("schedule_sha256"), "DONE schedule hash")
    authorization_sha = _hash_string(
        done.get("authorization_sha256"), "DONE authorization hash"
    )
    native_sha = _hash_string(
        done.get("native_library_sha256"), "DONE native library hash"
    )
    if native_sha != EXPECTED_NATIVE_LIBRARY_SHA256:
        raise ValueError("canary native library differs from accepted Step 6a")
    if done_schema == STEP6B_DONE_SCHEMA:
        _require(
            done.get("authorized_shards") == list(range(1, 10)),
            "Step 6b DONE authorization scope changed",
        )

    parity = _read_json(directory / "parity.json", "Linux parity")
    _validate_parity(
        parity,
        schema=parity_schema,
        native_library_sha256=native_sha,
        source_sha256=source_sha,
        manifest_sha256=manifest_sha,
    )
    heartbeat = _read_json(directory / "heartbeat.json", "final heartbeat")
    _require(
        heartbeat.get("schema") == STEP6A_HEARTBEAT_SCHEMA,
        "final heartbeat schema changed",
    )
    _require(heartbeat.get("status") == "pass", "final heartbeat is not pass")
    _require(
        heartbeat.get("total_tasks") == PAIRED_HANDS_PER_SHARD,
        "heartbeat total changed",
    )
    _require(
        heartbeat.get("completed_tasks") == PAIRED_HANDS_PER_SHARD,
        "heartbeat incomplete",
    )
    _require(heartbeat.get("pending_tasks") == 0, "heartbeat still has pending tasks")

    start = shard * PAIRED_HANDS_PER_SHARD
    hand_indices = tuple(range(start, start + PAIRED_HANDS_PER_SHARD))
    root_names = {path.name for path in (directory / "roots").iterdir()}
    task_names = {path.name for path in (directory / "tasks").iterdir()}
    expected_names = {f"hand_{index:03d}.json" for index in hand_indices}
    if root_names != expected_names or task_names != expected_names:
        raise ValueError(f"shard {shard} root/task hand grid changed")

    all_root_indices: list[int] = []
    all_fingerprints: list[str] = []
    profile_hands: list[str] = []
    all_seats: list[str] = []
    first_latencies: list[float] = []
    second_latencies: list[float] = []
    peak_rss = 0
    all_seed_values: list[int] = []
    all_candidate_keys: set[str] = set()
    all_evaluation_keys: set[str] = set()
    task_hashes: dict[int, str] = {}
    for index in hand_indices:
        root_path = directory / "roots" / f"hand_{index:03d}.json"
        task_path = directory / "tasks" / f"hand_{index:03d}.json"
        root = _read_json(root_path, "root task")
        observations, fingerprints, expected_seeds = _validate_root_task(
            root, global_hand_index=index
        )
        task = _read_json(task_path, "search task")
        (
            root_indices,
            seats,
            first,
            second,
            task_peak,
            candidate_keys,
            evaluation_keys,
        ) = _validate_search_task(
            task,
            root_task=root,
            observations=observations,
            global_hand_index=index,
            native_library_sha256=native_sha,
        )
        if all_candidate_keys & set(candidate_keys):
            raise ValueError("candidate RNG key repeats within shard")
        if all_evaluation_keys & set(evaluation_keys):
            raise ValueError("evaluation RNG key repeats within shard")
        all_candidate_keys.update(candidate_keys)
        all_evaluation_keys.update(evaluation_keys)
        all_root_indices.extend(root_indices)
        all_fingerprints.extend(fingerprints)
        profile_hands.append(str(root["profile"]))
        all_seats.extend(seats)
        first_latencies.extend(first)
        second_latencies.extend(second)
        peak_rss = max(peak_rss, task_peak)
        all_seed_values.extend(int(value) for value in expected_seeds.values())
        task_hashes[index] = _sha256(task_path)
    if all_candidate_keys & all_evaluation_keys:
        raise ValueError("candidate/evaluation RNG domains overlap within shard")

    summary = _read_json(directory / "summary.json", "shard summary")
    resumed = _validate_summary(
        summary,
        shard=shard,
        summary_schema=summary_schema,
        native_library_sha256=native_sha,
        fingerprints=all_fingerprints,
        profiles=profile_hands,
        first_latencies=first_latencies,
        second_latencies=second_latencies,
        peak_rss=peak_rss,
        task_hashes=task_hashes,
        candidate_keys=frozenset(all_candidate_keys),
        evaluation_keys=frozenset(all_evaluation_keys),
    )
    _require(summary.get("run_name") == run_name, "summary/DONE run name mismatch")
    _require(
        summary.get("source_package_sha256") == source_sha,
        "summary/DONE source mismatch",
    )
    _require(
        summary.get("manifest_sha256") == manifest_sha, "summary/DONE manifest mismatch"
    )
    if summary_schema == STEP6B_SUMMARY_SCHEMA:
        _require(
            summary.get("parity_report_sha256") == _sha256(directory / "parity.json"),
            "Step 6b summary parity hash mismatch",
        )
    if heartbeat.get("resumed_task_count") != resumed:
        raise ValueError("heartbeat/summary resume evidence mismatch")

    return _ValidatedShard(
        shard=shard,
        run_name=run_name,
        source_sha256=source_sha,
        manifest_sha256=manifest_sha,
        schedule_sha256=schedule_sha,
        authorization_sha256=authorization_sha,
        done_sha256=_sha256(directory / "DONE.json"),
        summary_sha256=_sha256(directory / "summary.json"),
        task_hashes=task_hashes,
        hand_indices=hand_indices,
        root_indices=tuple(all_root_indices),
        fingerprints=tuple(all_fingerprints),
        profile_hands=tuple(profile_hands),
        seats=tuple(all_seats),
        first_latencies=tuple(first_latencies),
        second_latencies=tuple(second_latencies),
        peak_rss_bytes=peak_rss,
        resumed_task_count=resumed,
        seed_values_flat=tuple(all_seed_values),
        candidate_rng_keys=frozenset(all_candidate_keys),
        evaluation_rng_keys=frozenset(all_evaluation_keys),
    )


def _validate_step6a_trust_anchor(
    directory: Path,
    shard: _ValidatedShard,
    *,
    accepted: Mapping[str, Any],
) -> None:
    receipt_path = directory / "receive_receipt.json"
    receipt = _read_json(receipt_path, "accepted Step 6a receive receipt")
    actual = {
        "run_name": shard.run_name,
        "done_sha256": shard.done_sha256,
        "summary_sha256": shard.summary_sha256,
        "receive_receipt_sha256": _sha256(receipt_path),
        "source_sha256": shard.source_sha256,
        "manifest_sha256": shard.manifest_sha256,
        "schedule_sha256": shard.schedule_sha256,
        "authorization_sha256": shard.authorization_sha256,
        "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
        "feature_encoder_sha256": accepted.get("feature_encoder_sha256"),
    }
    if actual != dict(accepted):
        raise ValueError("Step 6a shard 0 is not the accepted immutable v4 run")
    _require(receipt.get("status") == "pass", "Step 6a receive receipt is not pass")
    _require(receipt.get("all_gates_passed") is True, "Step 6a receive gate failed")
    _require(
        receipt.get("done_sha256") == shard.done_sha256, "Step 6a receipt/DONE mismatch"
    )
    _require(
        receipt.get("summary_sha256") == shard.summary_sha256,
        "Step 6a receipt/summary mismatch",
    )
    _require(
        receipt.get("training_eligible") is False, "Step 6a receipt allows training"
    )
    _require(
        receipt.get("production_fanout_authorized") is False,
        "Step 6a receipt authorizes production",
    )
    _require(
        receipt.get("current_profile_changed") is False,
        "Step 6a receipt changed current",
    )


def _validate_step6b_receive_receipt(
    path: Path,
    *,
    step6a: _ValidatedShard,
    step6b: Sequence[_ValidatedShard],
) -> dict[str, Any]:
    receipt = _read_json(path, "Step 6b aggregate receive receipt")
    _require(
        receipt.get("schema") == STEP6B_RECEIVE_SCHEMA, "Step 6b receipt schema changed"
    )
    _require(receipt.get("status") == "pass", "Step 6b receipt is not pass")
    _require(
        receipt.get("training_eligible") is False, "Step 6b receipt allows training"
    )
    _require(
        receipt.get("production_fanout_authorized") is False,
        "Step 6b receipt authorizes production",
    )
    _require(
        receipt.get("current_profile_changed") is False,
        "Step 6b receipt changed current",
    )
    _require(
        receipt.get("m31_complete") is False, "Step 6b receipt marks M3.1 complete"
    )
    _require(
        receipt.get("all_shards_received") is True,
        "Step 6b receipt does not close all nine shards",
    )
    _require(
        receipt.get("remaining_canary_shards_authorized") is True,
        "Step 6b receipt lost bounded canary authorization",
    )
    first = step6b[0]
    for field in (
        "run_name",
        "source_sha256",
        "manifest_sha256",
        "schedule_sha256",
        "authorization_sha256",
    ):
        if receipt.get(field) != getattr(first, field):
            raise ValueError(f"Step 6b receipt {field} mismatch")
    _require(
        receipt.get("native_library_sha256") == EXPECTED_NATIVE_LIBRARY_SHA256,
        "Step 6b receipt native hash mismatch",
    )
    _require(
        receipt.get("feature_encoder_sha256") == EXPECTED_FEATURE_ENCODER_SHA256,
        "Step 6b receipt feature encoder hash mismatch",
    )
    _require(
        receipt.get("step6a_done_sha256") == step6a.done_sha256,
        "Step 6b receipt accepted Step 6a DONE mismatch",
    )
    _require(
        receipt.get("step6a_summary_sha256") == step6a.summary_sha256,
        "Step 6b receipt accepted Step 6a summary mismatch",
    )
    records = receipt.get("per_shard")
    if not isinstance(records, Mapping) or set(records) != {
        f"{shard:03d}" for shard in range(1, 10)
    }:
        raise ValueError("Step 6b receipt shard hash manifest is incomplete")
    expected_records = {
        f"{shard.shard:03d}": {
            "done_sha256": shard.done_sha256,
            "summary_sha256": shard.summary_sha256,
            "heartbeat_sha256": _sha256(
                path.parent / "shards" / f"shard-{shard.shard:03d}" / "heartbeat.json"
            ),
            "task_count": PAIRED_HANDS_PER_SHARD,
            "root_task_count": PAIRED_HANDS_PER_SHARD,
            "resumed_task_count": shard.resumed_task_count,
        }
        for shard in step6b
    }
    if dict(records) != expected_records:
        raise ValueError("Step 6b receipt shard hash manifest changed")
    return receipt


def _validate_step6b_canary_core(
    *,
    step6a_shard0_dir: Path,
    step6b_received_dir: Path,
    accepted_step6a_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not step6b_received_dir.is_dir() or step6b_received_dir.is_symlink():
        raise ValueError("Step 6b receive directory is missing or unsafe")
    entries = list(step6b_received_dir.iterdir())
    if {path.name for path in entries} != {"shards", "receive_receipt.json"}:
        raise ValueError("Step 6b receive root file set changed")
    if any(path.is_symlink() for path in entries):
        raise ValueError("Step 6b receive root contains a symlink")
    shards_root = step6b_received_dir / "shards"
    shard_entries = list(shards_root.iterdir())
    expected_directories = {f"shard-{shard:03d}" for shard in range(1, 10)}
    if {path.name for path in shard_entries} != expected_directories or any(
        not path.is_dir() or path.is_symlink() for path in shard_entries
    ):
        raise ValueError("Step 6b received shard set is not exactly shard-001..009")

    step6a = _validate_shard(
        step6a_shard0_dir,
        shard=0,
        done_schema="hu_m31_t3_step6a_done_v1",
        parity_schema=STEP6A_PARITY_SCHEMA,
        summary_schema=STEP6A_SUMMARY_SCHEMA,
        local_extra_files=frozenset({"receive_receipt.json"}),
    )
    _validate_step6a_trust_anchor(
        step6a_shard0_dir, step6a, accepted=accepted_step6a_identity
    )
    step6b = [
        _validate_shard(
            shards_root / f"shard-{shard:03d}",
            shard=shard,
            done_schema=STEP6B_DONE_SCHEMA,
            parity_schema=STEP6B_PARITY_SCHEMA,
            summary_schema=STEP6B_SUMMARY_SCHEMA,
            local_extra_files=frozenset(),
        )
        for shard in range(1, 10)
    ]
    receive_receipt = _validate_step6b_receive_receipt(
        step6b_received_dir / "receive_receipt.json",
        step6a=step6a,
        step6b=step6b,
    )
    if (
        len(
            {
                (
                    shard.run_name,
                    shard.source_sha256,
                    shard.manifest_sha256,
                    shard.schedule_sha256,
                    shard.authorization_sha256,
                )
                for shard in step6b
            }
        )
        != 1
    ):
        raise ValueError("Step 6b shards used different immutable package identities")

    shards = [step6a, *step6b]
    hand_indices = [index for shard in shards for index in shard.hand_indices]
    root_indices = [index for shard in shards for index in shard.root_indices]
    fingerprints = [value for shard in shards for value in shard.fingerprints]
    profiles = [value for shard in shards for value in shard.profile_hands]
    seats = [value for shard in shards for value in shard.seats]
    seed_schedule = [value for shard in shards for value in shard.seed_values_flat]
    first_latencies = [value for shard in shards for value in shard.first_latencies]
    second_latencies = [value for shard in shards for value in shard.second_latencies]
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for shard in shards:
        if candidate_keys & set(shard.candidate_rng_keys):
            raise ValueError("candidate RNG keys overlap across shards")
        if evaluation_keys & set(shard.evaluation_rng_keys):
            raise ValueError("evaluation RNG keys overlap across shards")
        candidate_keys.update(shard.candidate_rng_keys)
        evaluation_keys.update(shard.evaluation_rng_keys)

    profile_hand_counts = {
        profile: profiles.count(profile) for profile in M31_T3_BEHAVIOR_PROFILES
    }
    seat_counts = {seat: seats.count(seat) for seat in ("first", "second")}
    first_performance = _latency(first_latencies)
    second_performance = _latency(second_latencies)
    peak_rss = max(shard.peak_rss_bytes for shard in shards)
    gates = {
        "exact_shard_set_0_through_9": [shard.shard for shard in shards]
        == list(range(TOTAL_SHARDS)),
        "exact_hand_grid_0_through_249": hand_indices
        == list(range(TOTAL_PAIRED_HANDS)),
        "exact_root_grid_0_through_499": root_indices == list(range(TOTAL_ROOTS)),
        "exactly_250_each_seat": seat_counts == {"first": 250, "second": 250},
        "five_profiles_exactly_50_hands_each": all(
            count == 50 for count in profile_hand_counts.values()
        ),
        "five_profiles_exactly_100_roots_each": all(
            count * 2 == 100 for count in profile_hand_counts.values()
        ),
        "all_500_observation_fingerprints_unique": (
            len(fingerprints) == len(set(fingerprints)) == TOTAL_ROOTS
        ),
        "all_1500_namespace_seeds_unique": (
            len(seed_schedule) == len(set(seed_schedule)) == TOTAL_PAIRED_HANDS * 6
        ),
        "all_2000_candidate_rng_keys_unique": (
            len(candidate_keys) == TOTAL_ROOTS * REFERENCE_BUDGET.candidate_samples
        ),
        "all_4000_evaluation_rng_keys_unique": (
            len(evaluation_keys) == TOTAL_ROOTS * REFERENCE_BUDGET.evaluation_samples
        ),
        "candidate_evaluation_rng_overlap_zero": not (candidate_keys & evaluation_keys),
        "all_ten_shards_recovered_resume_task": all(
            shard.resumed_task_count >= 1 for shard in shards
        ),
        "global_first_p95_within_180_seconds": (
            first_performance["p95_seconds"] <= MAX_FIRST_P95_SECONDS
        ),
        "global_second_p95_within_6_seconds": (
            second_performance["p95_seconds"] <= MAX_SECOND_P95_SECONDS
        ),
        "global_peak_rss_within_1_gib": peak_rss <= MAX_PEAK_RSS_BYTES,
        "canary_rows_not_training_eligible": True,
        "teacher_values_not_realized_match_ev": True,
        "no_current_profile_runtime_or_production_activation": True,
    }
    if not all(gates.values()):
        failed = sorted(key for key, passed in gates.items() if not passed)
        raise ValueError(f"merged Step 6b canary gates failed: {failed}")

    step6b_identity = step6b[0]
    return {
        "schema": STEP6B_VALIDATION_SCHEMA,
        "status": "pass",
        "decision": "infrastructure_canary_500_roots_pass_production_fanout_no_go",
        "scope": "infrastructure_only_not_training_not_match_ev_not_promotion",
        "contract": {
            "shards": TOTAL_SHARDS,
            "paired_hands": TOTAL_PAIRED_HANDS,
            "roots": TOTAL_ROOTS,
            "workers_per_shard": WORKERS,
            "rayon_threads_per_worker": RAYON_THREADS,
            "budget": REFERENCE_BUDGET.to_dict(),
            "run_id": STEP6A_RUN_ID,
        },
        "accepted_step6a": {
            "run_name": step6a.run_name,
            "done_sha256": step6a.done_sha256,
            "summary_sha256": step6a.summary_sha256,
            "source_sha256": step6a.source_sha256,
            "manifest_sha256": step6a.manifest_sha256,
        },
        "step6b": {
            "run_name": step6b_identity.run_name,
            "source_sha256": step6b_identity.source_sha256,
            "manifest_sha256": step6b_identity.manifest_sha256,
            "schedule_sha256": step6b_identity.schedule_sha256,
            "authorization_sha256": step6b_identity.authorization_sha256,
            "receive_receipt_sha256": _sha256(
                step6b_received_dir / "receive_receipt.json"
            ),
            "receive_receipt_schema": receive_receipt["schema"],
        },
        "shards": [
            {
                "shard": shard.shard,
                "run_name": shard.run_name,
                "done_sha256": shard.done_sha256,
                "summary_sha256": shard.summary_sha256,
                "resumed_task_count": shard.resumed_task_count,
                "peak_rss_bytes": shard.peak_rss_bytes,
            }
            for shard in shards
        ],
        "integrity": {
            "hand_indices": len(hand_indices),
            "root_indices": len(root_indices),
            "fingerprints": len(fingerprints),
            "unique_fingerprints": len(set(fingerprints)),
            "seat_counts": seat_counts,
            "profile_hand_counts": profile_hand_counts,
            "namespace_seed_values": len(seed_schedule),
            "unique_namespace_seed_values": len(set(seed_schedule)),
            "candidate_rng_keys": len(candidate_keys),
            "evaluation_rng_keys": len(evaluation_keys),
            "candidate_evaluation_rng_overlap": len(candidate_keys & evaluation_keys),
        },
        "performance": {
            "latency_by_seat": {
                "first": first_performance,
                "second": second_performance,
            },
            "peak_process_rss_bytes": peak_rss,
        },
        "gates": gates,
        "all_gates_passed": True,
        "training_eligible": False,
        "teacher_value_status": "diagnostic_not_match_EV",
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6b validation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Step 6b validation staging already exists: {temporary}")
    try:
        data = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_step6b_canary(
    *,
    step6a_shard0_dir: str | Path,
    step6b_received_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Step 6b validation: {output}")
    report = _validate_step6b_canary_core(
        step6a_shard0_dir=Path(step6a_shard0_dir).resolve(),
        step6b_received_dir=Path(step6b_received_dir).resolve(),
        accepted_step6a_identity=ACCEPTED_STEP6A_IDENTITY,
    )
    _write_once_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step6a-shard0", type=Path, required=True)
    parser.add_argument("--step6b-received", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_step6b_canary(
        step6a_shard0_dir=args.step6a_shard0,
        step6b_received_dir=args.step6b_received,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACCEPTED_STEP6A_IDENTITY",
    "STEP6B_VALIDATION_SCHEMA",
    "validate_step6b_canary",
]
