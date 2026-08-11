"""Run one resumable Attempt07 development root.

The runner derives every root property from the frozen Attempt07 contract.  It
loads only explicit named profiles, generates a canonical T1-second
information set, selects the explicit ``stage18_p1`` baseline, and evaluates
the frozen LambdaRank candidate generator with the explicit ``stage9f_p2``
continuation.  It never resolves ``current`` and never authorizes a runtime
policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .action_key import action_key
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .generate_hu_m4_t1_data import (
    _profile_policy_seed,
    generate_t1_second_root,
)
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_SHA256,
    FrozenAttempt06LambdaRanker,
    _require_concrete_stage9f_p2_policies,
)
from .hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
    M43_ATTEMPT07_PROFILES,
    enumerate_attempt07_seed_schedules,
    load_and_validate_attempt07_plan,
)
from .hu_m43_attempt07_teacher import (
    ATTEMPT07_TEACHER_SCHEMA,
    Attempt07TeacherConfig,
    evaluate_attempt07_t1_second,
)
from .play_ai import _choose_from_observation, _hand_decision_seed


ATTEMPT07_DEVELOPMENT_ROOTS = 100
ATTEMPT07_BASELINE_PROFILE = "stage18_p1"
ATTEMPT07_CONTINUATION_PROFILE = "stage9f_p2"
ATTEMPT07_SHARD_ROW_SCHEMA = "hu_m43_attempt07_development_shard_row_v1"
ATTEMPT07_PROVENANCE_SCHEMA = "hu_m43_attempt07_development_provenance_v1"
ATTEMPT07_CHECKPOINT_SCHEMA = "hu_m43_attempt07_development_checkpoint_v1"
ATTEMPT07_HEARTBEAT_SCHEMA = "hu_m43_attempt07_development_heartbeat_v1"
ATTEMPT07_SHARD_SUMMARY_SCHEMA = "hu_m43_attempt07_development_summary_v1"
ATTEMPT07_FIXED_CONTRACT_SCHEMA = "hu_m43_attempt07_development_contract_v1"
ATTEMPT07_NATIVE_BATCH_THREADS_DEFAULT = 4
ATTEMPT07_ROOT_GENERATION_POLICY = (
    "attempt07_explicit_mod5_per_root_policy_seed_v1"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
DEFAULT_AI_PROFILES_PATH = _REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
_SEED_DOMAINS = ("hand", "screen", "rerank", "veto", "assessment", "child")
_RNG_DOMAIN_NAMES = {
    "hand": "live_root_deal_and_explicit_profile_actions",
    "screen": "screen_s8",
    "rerank": "rerank_r64",
    "veto": "veto_v128",
    "assessment": "assessment_a128",
    "child": "infoset_child_policy",
}
_PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "root_index",
        "root_profile",
        "profile_assignment",
        "plan_sha256",
        "ai_profiles_sha256",
        "model_sha256",
        "config_sha256",
        "root_input_sha256",
        "seeds",
        "rng_domains",
        "root_generation_policy",
        "root_policy_profiles",
        "root_policy_seeds",
        "baseline_profile",
        "baseline_policy_seed",
        "continuation_profile",
        "continuation_policy_seeds",
        "explicit_loaded_profiles",
        "batch_child_selectors",
        "native_batch_threads",
        "current_profile_resolved",
        "opponent_private_discard_input_allowed",
        "teacher_value_status",
        "teacher_values_are_realized_match_ev",
        "teacher_ev_or_lcb_runtime_gate_allowed",
        "development_only",
        "fit_allowed",
        "threshold_selection_allowed",
        "runtime_activation_allowed",
        "full_replacement_enabled",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload).rstrip(b"\n")).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalized


def _canonical_path_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    identities: dict[str, str] = {}
    for name, path in paths.items():
        identity = _canonical_path_identity(path)
        if identity in identities:
            raise ValueError(
                f"Attempt07 paths must be distinct: {name} aliases "
                f"{identities[identity]}"
            )
        identities[identity] = name
    items = tuple(paths.items())
    for left_index, (left_name, left_path) in enumerate(items):
        if not left_path.exists():
            continue
        for right_name, right_path in items[left_index + 1 :]:
            if not right_path.exists():
                continue
            try:
                same_file = os.path.samefile(left_path, right_path)
            except OSError as exc:
                raise ValueError(
                    "Attempt07 could not verify existing path identities"
                ) from exc
            if same_file:
                raise ValueError(
                    "Attempt07 paths must be distinct: "
                    f"{right_name} hard-links or aliases {left_name}"
                )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _HeartbeatPump:
    def __init__(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        self.path = path
        self.payload = dict(payload)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="attempt07-development-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._write()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds + 1.0))
        if self.thread.is_alive():
            raise RuntimeError("Attempt07 heartbeat thread did not stop")
        if self.error is not None:
            raise RuntimeError("Attempt07 live heartbeat write failed") from self.error

    def _run(self) -> None:
        try:
            while not self.stop_event.wait(self.interval_seconds):
                self._write()
        except BaseException as exc:  # surfaced synchronously by stop
            self.error = exc
            self.stop_event.set()

    def _write(self) -> None:
        _atomic_json(
            self.path,
            {**self.payload, "updated_unix_seconds": time.time()},
        )


class _ShardFileLock:
    """Non-blocking one-byte process lock for a single output shard."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "_ShardFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                "Attempt07 shard is already owned by another worker"
            ) from exc
        self.handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self.handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self.handle = None


@contextmanager
def _native_batch_threads(enabled: bool, count: int) -> Iterator[None]:
    key = "OFC_HU_M3_BATCH_THREADS"
    prior = os.environ.get(key)
    if enabled:
        os.environ[key] = str(count)
    try:
        yield
    finally:
        if enabled:
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _development_seeds(
    plan: Mapping[str, Any], root_index: int
) -> dict[str, int]:
    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise TypeError("Attempt07 root_index must be an integer")
    if not 0 <= root_index < ATTEMPT07_DEVELOPMENT_ROOTS:
        raise ValueError("Attempt07 development root_index is outside 0..99")
    schedules = enumerate_attempt07_seed_schedules(plan, population="development")
    seeds = {domain: schedules[domain][root_index] for domain in _SEED_DOMAINS}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Attempt07 per-root seed domains overlap")
    return seeds


def _fixed_contract(
    *,
    root_index: int,
    root_profile: str,
    seeds: Mapping[str, int],
    run_id: str,
    plan_sha256: str,
    ai_profiles_sha256: str,
    model_sha256: str,
    batch_child_selectors: bool,
    native_batch_threads: int,
) -> dict[str, Any]:
    return {
        "schema": ATTEMPT07_FIXED_CONTRACT_SCHEMA,
        "teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
        "plan_sha256": plan_sha256,
        "ai_profiles_sha256": ai_profiles_sha256,
        "model_sha256": model_sha256,
        "root_index": root_index,
        "root_profile": root_profile,
        "seeds": dict(seeds),
        "run_id": run_id,
        "batch_child_selectors": batch_child_selectors,
        "native_batch_threads": native_batch_threads,
        "baseline_profile": ATTEMPT07_BASELINE_PROFILE,
        "continuation_profile": ATTEMPT07_CONTINUATION_PROFILE,
        "current_profile_resolved": False,
        "development_only": True,
    }


def _root_input_sha256(
    *,
    root_index: int,
    hand_seed: int,
    root_profile: str,
    observation: ActorObservation,
    baseline_action_key: str,
) -> str:
    return _canonical_sha256(
        {
            "root_index": root_index,
            "hand_seed": hand_seed,
            "root_profile": root_profile,
            "policy_observation": observation.to_dict(),
            "baseline_action_key": baseline_action_key,
        }
    )


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    fixed_contract: Mapping[str, Any],
    config_sha256: str,
    root_input_sha256: str,
) -> None:
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("Attempt07 provenance fields changed")
    expected = {
        "schema": ATTEMPT07_PROVENANCE_SCHEMA,
        "run_id": fixed_contract["run_id"],
        "root_index": fixed_contract["root_index"],
        "root_profile": fixed_contract["root_profile"],
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "plan_sha256": fixed_contract["plan_sha256"],
        "ai_profiles_sha256": fixed_contract["ai_profiles_sha256"],
        "model_sha256": fixed_contract["model_sha256"],
        "config_sha256": config_sha256,
        "root_input_sha256": root_input_sha256,
        "seeds": fixed_contract["seeds"],
        "rng_domains": _RNG_DOMAIN_NAMES,
        "root_generation_policy": ATTEMPT07_ROOT_GENERATION_POLICY,
        "root_policy_profiles": {
            "first": fixed_contract["root_profile"],
            "second": fixed_contract["root_profile"],
        },
        "root_policy_seeds": {
            seat: _profile_policy_seed(
                int(fixed_contract["seeds"]["hand"]),
                str(fixed_contract["root_profile"]),
                seat,
            )
            for seat in ("first", "second")
        },
        "baseline_profile": ATTEMPT07_BASELINE_PROFILE,
        "baseline_policy_seed": _profile_policy_seed(
            int(fixed_contract["seeds"]["hand"]),
            ATTEMPT07_BASELINE_PROFILE,
            "second",
        ),
        "continuation_profile": ATTEMPT07_CONTINUATION_PROFILE,
        "continuation_policy_seeds": {
            "first": int(fixed_contract["seeds"]["child"]),
            "second": int(fixed_contract["seeds"]["child"]) + 1,
        },
        "explicit_loaded_profiles": sorted(
            {
                str(fixed_contract["root_profile"]),
                ATTEMPT07_BASELINE_PROFILE,
                ATTEMPT07_CONTINUATION_PROFILE,
            }
        ),
        "batch_child_selectors": fixed_contract["batch_child_selectors"],
        "native_batch_threads": fixed_contract["native_batch_threads"],
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_value_status": "diagnostic_not_match_EV",
        "teacher_values_are_realized_match_ev": False,
        "teacher_ev_or_lcb_runtime_gate_allowed": False,
        "development_only": True,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "full_replacement_enabled": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Attempt07 provenance {key} changed")
def _reject_hidden_discard_fields(value: Any, *, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "opponent_private_discards":
                raise ValueError(
                    f"Attempt07 output exposes opponent private discards at {path}"
                )
            _reject_hidden_discard_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_hidden_discard_fields(child, path=f"{path}[{index}]")


def _resume_one_root(
    partial_path: Path,
    checkpoint_path: Path,
    *,
    config_sha256: str,
    plan_sha256: str,
    ai_profiles_sha256: str,
    model_sha256: str,
    expected_root_index: int,
) -> int:
    checkpoint: Mapping[str, Any] | None = None
    if checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, Mapping):
            raise ValueError("Attempt07 checkpoint must be a mapping")
        checkpoint = loaded
        expected = {
            "schema": ATTEMPT07_CHECKPOINT_SCHEMA,
            "config_sha256": config_sha256,
            "plan_sha256": plan_sha256,
            "ai_profiles_sha256": ai_profiles_sha256,
            "model_sha256": model_sha256,
            "target_roots": 1,
            "root_index": expected_root_index,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("Attempt07 checkpoint configuration or hash mismatch")
        if checkpoint.get("completed_roots") not in (0, 1):
            raise ValueError("Attempt07 checkpoint completed_roots is invalid")

    if not partial_path.exists():
        if checkpoint is None:
            return 0
        if (
            checkpoint.get("completed_roots") == 0
            and checkpoint.get("partial_sha256") == hashlib.sha256(b"").hexdigest()
        ):
            return 0
        raise ValueError("Attempt07 completed checkpoint has no partial output")

    raw = partial_path.read_bytes()
    if checkpoint is None:
        if raw:
            with partial_path.open("r+b") as handle:
                handle.truncate(0)
        return 0
    completed = int(checkpoint["completed_roots"])
    newline_offsets = [index + 1 for index, byte in enumerate(raw) if byte == 10]
    boundary = newline_offsets[completed - 1] if completed else 0
    if len(raw) != boundary:
        with partial_path.open("r+b") as handle:
            handle.truncate(boundary)
        raw = raw[:boundary]
    if hashlib.sha256(raw).hexdigest() != checkpoint.get("partial_sha256"):
        raise ValueError("Attempt07 partial hash disagrees with checkpoint")
    if completed:
        row = json.loads(raw.decode("utf-8").strip())
        if (
            row.get("schema") != ATTEMPT07_SHARD_ROW_SCHEMA
            or row.get("root_index") != expected_root_index
        ):
            raise ValueError("Attempt07 checkpointed row identity mismatch")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Attempt07 checkpointed row provenance is missing")
        expected_provenance = {
            "config_sha256": config_sha256,
            "plan_sha256": plan_sha256,
            "ai_profiles_sha256": ai_profiles_sha256,
            "model_sha256": model_sha256,
        }
        if any(
            provenance.get(key) != value
            for key, value in expected_provenance.items()
        ):
            raise ValueError("Attempt07 checkpointed row hash provenance mismatch")
    return completed


def _run_development_shard_locked(
    *,
    root_index: int,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    run_id: str,
    plan: str | Path = DEFAULT_PLAN_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    batch_child_selectors: bool = False,
    native_batch_threads: int = ATTEMPT07_NATIVE_BATCH_THREADS_DEFAULT,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    if not isinstance(batch_child_selectors, bool):
        raise TypeError("batch_child_selectors must be a bool")
    if batch_child_selectors is not True:
        raise ValueError("Attempt07 development requires batched child selectors")
    if (
        isinstance(native_batch_threads, bool)
        or not isinstance(native_batch_threads, int)
        or native_batch_threads != ATTEMPT07_NATIVE_BATCH_THREADS_DEFAULT
    ):
        raise ValueError("Attempt07 development native_batch_threads is fixed at 4")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must not be empty")

    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    checkpoint_path = Path(checkpoint)
    heartbeat_path = Path(heartbeat)
    model_path = Path(model)
    plan_path = Path(plan)
    ai_profiles_path = Path(ai_profiles)
    lock_path = output_path.with_name(output_path.name + ".lock")
    _require_distinct_paths(
        {
            "output": output_path,
            "partial": partial_path,
            "checkpoint": checkpoint_path,
            "heartbeat": heartbeat_path,
            "model": model_path,
            "plan": plan_path,
            "ai_profiles": ai_profiles_path,
            "lock": lock_path,
        }
    )
    if output_path.exists():
        raise FileExistsError(f"Attempt07 shard already complete: {output_path}")

    plan_payload = load_and_validate_attempt07_plan(plan_path)
    plan_hash = _sha256_file(plan_path)
    if plan_hash != M43_ATTEMPT07_PLAN_SHA256:
        raise ValueError("Attempt07 plan byte SHA-256 changed")
    ai_profiles_hash = _sha256_file(ai_profiles_path)
    if ai_profiles_hash != AI_PROFILES_SHA256:
        raise ValueError("Attempt07 ai_profiles.py SHA-256 changed")
    expected_model_hash = _require_sha256(model_sha256, name="model_sha256")
    if expected_model_hash != ATTEMPT06_FROZEN_MODEL_SHA256:
        raise ValueError("Attempt07 model is not the frozen Lambda artifact")

    seeds = _development_seeds(plan_payload, root_index)
    root_profile = M43_ATTEMPT07_PROFILES[root_index % len(M43_ATTEMPT07_PROFILES)]
    fixed_contract = _fixed_contract(
        root_index=root_index,
        root_profile=root_profile,
        seeds=seeds,
        run_id=run_id,
        plan_sha256=plan_hash,
        ai_profiles_sha256=ai_profiles_hash,
        model_sha256=expected_model_hash,
        batch_child_selectors=batch_child_selectors,
        native_batch_threads=native_batch_threads,
    )
    config_sha256 = _canonical_sha256(fixed_contract)
    completed = _resume_one_root(
        partial_path,
        checkpoint_path,
        config_sha256=config_sha256,
        plan_sha256=plan_hash,
        ai_profiles_sha256=ai_profiles_hash,
        model_sha256=expected_model_hash,
        expected_root_index=root_index,
    )
    started = time.perf_counter()
    if completed == 0:
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        starting_common = {
            "config_sha256": config_sha256,
            "plan_sha256": plan_hash,
            "ai_profiles_sha256": ai_profiles_hash,
            "model_sha256": expected_model_hash,
            "completed_roots": 0,
            "target_roots": 1,
            "root_index": root_index,
            "partial_sha256": empty_sha256,
        }
        _atomic_json(
            checkpoint_path,
            {
                "schema": ATTEMPT07_CHECKPOINT_SCHEMA,
                **starting_common,
                "updated_unix_seconds": time.time(),
            },
        )
        heartbeat_pump = _HeartbeatPump(
            heartbeat_path,
            {
                "schema": ATTEMPT07_HEARTBEAT_SCHEMA,
                "status": "running",
                **starting_common,
            },
        )
        heartbeat_pump.start()
        try:
            explicit_profiles = {
                root_profile,
                ATTEMPT07_BASELINE_PROFILE,
                ATTEMPT07_CONTINUATION_PROFILE,
            }
            if "current" in explicit_profiles:  # structural fail-closed guard
                raise AssertionError("Attempt07 must never resolve current")
            bundle = load_model_bundle(
                paths or ModelPaths(), profiles=explicit_profiles
            )
            root_policy_seeds = {
                seat: _profile_policy_seed(seeds["hand"], root_profile, seat)
                for seat in ("first", "second")
            }
            root_policies = {
                seat: build_policy(
                    root_profile,
                    bundle,
                    seed=root_policy_seeds[seat],
                    seat=seat,
                    opening_lookahead_samples=0,
                )
                for seat in ("first", "second")
            }
            observation = generate_t1_second_root(
                seeds["hand"], root_policies=root_policies
            )
            baseline_policy_seed = _profile_policy_seed(
                seeds["hand"], ATTEMPT07_BASELINE_PROFILE, "second"
            )
            baseline_policy = build_policy(
                ATTEMPT07_BASELINE_PROFILE,
                bundle,
                seed=baseline_policy_seed,
                seat="second",
                opening_lookahead_samples=0,
            )
            baseline_action = _choose_from_observation(
                baseline_policy,
                observation,
                hand_id=seeds["hand"],
                game_id=seeds["hand"],
                decision_seed=_hand_decision_seed(
                    base_seed=seeds["hand"], observation=observation
                ),
            )
            baseline_token = action_key(baseline_action).to_token()
            continuation_policy_seeds = {
                "first": seeds["child"],
                "second": seeds["child"] + 1,
            }
            t2_policies = {
                seat: build_policy(
                    ATTEMPT07_CONTINUATION_PROFILE,
                    bundle,
                    seed=continuation_policy_seeds[seat],
                    seat=seat,
                    opening_lookahead_samples=0,
                )
                for seat in ("first", "second")
            }
            _require_concrete_stage9f_p2_policies(t2_policies)
            ranker = FrozenAttempt06LambdaRanker.load(
                model_path, expected_sha256=expected_model_hash
            )
            teacher_config = Attempt07TeacherConfig(
                frozen_model_sha256=expected_model_hash,
                screen_seed=seeds["screen"],
                rerank_seed=seeds["rerank"],
                veto_seed=seeds["veto"],
                assessment_seed=seeds["assessment"],
                child_policy_seed=seeds["child"],
                run_id=(
                    f"{run_id}:root={root_index}:seed={seeds['hand']}:"
                    f"obs={observation.fingerprint()}"
                ),
                batch_child_selectors=batch_child_selectors,
            )
            with _native_batch_threads(
                batch_child_selectors, native_batch_threads
            ):
                teacher = evaluate_attempt07_t1_second(
                    observation,
                    baseline_action_key=baseline_token,
                    ranker=ranker,
                    t2_policies=t2_policies,
                    config=teacher_config,
                )
        finally:
            heartbeat_pump.stop()
        if (
            teacher.get("status") != "ok"
            or teacher.get("schema") != ATTEMPT07_TEACHER_SCHEMA
            or teacher.get("observation_fingerprint")
            != observation.fingerprint()
            or teacher.get("street") != "T1"
            or teacher.get("seat") != "second"
            or teacher.get("to_act_order") != "second"
            or teacher.get("baseline_action_key") != baseline_token
            or teacher.get("teacher_value_status") != "diagnostic_not_match_EV"
            or teacher.get("runtime_gate_allowed") is not False
            or teacher.get("profile_activation_allowed") is not False
            or teacher.get("current_profile_resolved") is not False
            or teacher.get("development_only") is not True
        ):
            raise ValueError("Attempt07 teacher result identity changed")
        root_input_hash = _root_input_sha256(
            root_index=root_index,
            hand_seed=seeds["hand"],
            root_profile=root_profile,
            observation=observation,
            baseline_action_key=baseline_token,
        )
        provenance = {
            "schema": ATTEMPT07_PROVENANCE_SCHEMA,
            "run_id": run_id,
            "root_index": root_index,
            "root_profile": root_profile,
            "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "plan_sha256": plan_hash,
            "ai_profiles_sha256": ai_profiles_hash,
            "model_sha256": expected_model_hash,
            "config_sha256": config_sha256,
            "root_input_sha256": root_input_hash,
            "seeds": dict(seeds),
            "rng_domains": dict(_RNG_DOMAIN_NAMES),
            "root_generation_policy": ATTEMPT07_ROOT_GENERATION_POLICY,
            "root_policy_profiles": {
                "first": root_profile,
                "second": root_profile,
            },
            "root_policy_seeds": root_policy_seeds,
            "baseline_profile": ATTEMPT07_BASELINE_PROFILE,
            "baseline_policy_seed": baseline_policy_seed,
            "continuation_profile": ATTEMPT07_CONTINUATION_PROFILE,
            "continuation_policy_seeds": continuation_policy_seeds,
            "explicit_loaded_profiles": sorted(explicit_profiles),
            "batch_child_selectors": batch_child_selectors,
            "native_batch_threads": native_batch_threads,
            "current_profile_resolved": False,
            "opponent_private_discard_input_allowed": False,
            "teacher_value_status": "diagnostic_not_match_EV",
            "teacher_values_are_realized_match_ev": False,
            "teacher_ev_or_lcb_runtime_gate_allowed": False,
            "development_only": True,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "full_replacement_enabled": False,
        }
        _validate_provenance(
            provenance,
            fixed_contract=fixed_contract,
            config_sha256=config_sha256,
            root_input_sha256=root_input_hash,
        )
        row = {
            "schema": ATTEMPT07_SHARD_ROW_SCHEMA,
            "root_index": root_index,
            "hand_seed": seeds["hand"],
            "root_profile": root_profile,
            "policy_observation": observation.to_dict(),
            "baseline_action_key": baseline_token,
            "provenance": provenance,
            "teacher": teacher,
        }
        _reject_hidden_discard_fields(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_json_bytes(row)
        with partial_path.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        common = {
            "config_sha256": config_sha256,
            "plan_sha256": plan_hash,
            "ai_profiles_sha256": ai_profiles_hash,
            "model_sha256": expected_model_hash,
            "completed_roots": 1,
            "target_roots": 1,
            "root_index": root_index,
            "partial_sha256": _sha256_file(partial_path),
            "updated_unix_seconds": time.time(),
        }
        _atomic_json(
            checkpoint_path,
            {"schema": ATTEMPT07_CHECKPOINT_SCHEMA, **common},
        )

    try:
        os.link(partial_path, output_path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Attempt07 shard concurrently completed: {output_path}"
        ) from exc
    partial_path.unlink()
    summary = {
        "schema": ATTEMPT07_SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "root_index": root_index,
        "completed_roots": 1,
        "target_roots": 1,
        "config_sha256": config_sha256,
        "plan_sha256": plan_hash,
        "ai_profiles_sha256": ai_profiles_hash,
        "model_sha256": expected_model_hash,
        "output_sha256": _sha256_file(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(
        heartbeat_path,
        {
            **summary,
            "schema": ATTEMPT07_HEARTBEAT_SCHEMA,
            "summary_schema": ATTEMPT07_SHARD_SUMMARY_SCHEMA,
        },
    )
    return summary


def run_development_shard(
    *,
    root_index: int,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    run_id: str,
    plan: str | Path = DEFAULT_PLAN_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    batch_child_selectors: bool = False,
    native_batch_threads: int = ATTEMPT07_NATIVE_BATCH_THREADS_DEFAULT,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Run one development root under a no-clobber process lock."""

    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    lock_path = output_path.with_name(output_path.name + ".lock")
    _require_distinct_paths(
        {
            "output": output_path,
            "partial": partial_path,
            "checkpoint": Path(checkpoint),
            "heartbeat": Path(heartbeat),
            "model": Path(model),
            "plan": Path(plan),
            "ai_profiles": Path(ai_profiles),
            "lock": lock_path,
        }
    )
    with _ShardFileLock(lock_path):
        return _run_development_shard_locked(
            root_index=root_index,
            output=output,
            checkpoint=checkpoint,
            heartbeat=heartbeat,
            model=model,
            model_sha256=model_sha256,
            run_id=run_id,
            plan=plan,
            ai_profiles=ai_profiles,
            batch_child_selectors=batch_child_selectors,
            native_batch_threads=native_batch_threads,
            paths=paths,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--ai-profiles", type=Path, default=DEFAULT_AI_PROFILES_PATH
    )
    parser.add_argument("--batch-child-selectors", action="store_true")
    parser.add_argument(
        "--native-batch-threads",
        type=int,
        default=ATTEMPT07_NATIVE_BATCH_THREADS_DEFAULT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_development_shard(
        root_index=args.root_index,
        output=args.output,
        checkpoint=args.checkpoint,
        heartbeat=args.heartbeat,
        model=args.model,
        model_sha256=args.model_sha256,
        run_id=args.run_id,
        plan=args.plan,
        ai_profiles=args.ai_profiles,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT07_CHECKPOINT_SCHEMA",
    "ATTEMPT07_HEARTBEAT_SCHEMA",
    "ATTEMPT07_PROVENANCE_SCHEMA",
    "ATTEMPT07_SHARD_ROW_SCHEMA",
    "ATTEMPT07_SHARD_SUMMARY_SCHEMA",
    "DEFAULT_AI_PROFILES_PATH",
    "DEFAULT_PLAN_PATH",
    "parse_args",
    "run_development_shard",
]
