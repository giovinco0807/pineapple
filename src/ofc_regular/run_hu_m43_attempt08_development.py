"""Run one deterministic, restartable M4.3 Attempt08 development root.

The runner derives the root, opponent profile, and every RNG domain from the
frozen Attempt08 contract.  It loads explicit profiles only, uses the fixed
``stage18_p1`` T1-second baseline and explicit ``stage9f_p2`` continuation,
and delegates search to the fixed Attempt08 teacher.  It never resolves
``current`` or authorizes audit, fitting, thresholding, or runtime activation.

Operational DONE records intentionally remain the responsibility of a future
Spot wrapper because only that layer can bind manifest, authorization,
schedule, and source-closure identities.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .action_key import action_key
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .generate_hu_m4_t1_data import _profile_policy_seed, generate_t1_second_root
from .finalize_hu_m43_attempt08_preflight import (
    load_and_validate_development_open_authorization,
)
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import _require_concrete_stage9f_p2_policies
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
)
from .hu_m43_attempt08_teacher import (
    ATTEMPT08_TEACHER_SCHEMA,
    Attempt08TeacherConfig,
    FrozenAttempt08LambdaRanker,
    evaluate_attempt08_t1_second,
    validate_attempt08_teacher_output,
)
from .hu_m43_attempt08_runtime_anchor import ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
from .hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
)
from .play_ai import _choose_from_observation, _hand_decision_seed


ATTEMPT08_DEVELOPMENT_ROOTS = 200
ATTEMPT08_BASELINE_PROFILE = "stage18_p1"
ATTEMPT08_CONTINUATION_PROFILE = "stage9f_p2"
ATTEMPT08_SHARD_ROW_SCHEMA = "hu_m43_attempt08_development_shard_row_v1"
ATTEMPT08_PROVENANCE_SCHEMA = "hu_m43_attempt08_development_provenance_v1"
ATTEMPT08_CHECKPOINT_SCHEMA = "hu_m43_attempt08_development_checkpoint_v1"
ATTEMPT08_HEARTBEAT_SCHEMA = "hu_m43_attempt08_development_heartbeat_v1"
ATTEMPT08_SHARD_SUMMARY_SCHEMA = "hu_m43_attempt08_development_summary_v1"
ATTEMPT08_FIXED_CONTRACT_SCHEMA = "hu_m43_attempt08_development_contract_v1"
ATTEMPT08_NATIVE_BATCH_THREADS = 4
ATTEMPT08_ROOT_GENERATION_POLICY = (
    "attempt08_explicit_mod5_per_root_policy_seed_v1"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt08.json"
DEFAULT_PREFLIGHT_PLAN_PATH = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt08_preflight.json"
)
DEFAULT_AI_PROFILES_PATH = _REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
_SEED_DOMAINS = (
    "hand",
    "rerank",
    "veto",
    "stress",
    "assessment",
    "child",
)
_RNG_DOMAIN_NAMES = {
    "hand": "live_root_deal_and_explicit_profile_actions",
    "rerank": "rerank_r128",
    "veto": "veto_v256",
    "stress": "stress_x512",
    "assessment": "assessment_a256",
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
        "development_open_authorization_sha256",
        "preflight_plan_sha256",
        "preflight_result_sha256",
        "preflight_proof_evidence_sha256",
        "preflight_proof_gates_sha256",
        "preflight_operational_gates_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
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
        "search_freeze_authorization_allowed",
        "future_audit_allowed",
        "fit_allowed",
        "threshold_selection_allowed",
        "runtime_activation_allowed",
        "full_replacement_enabled",
    }
)

_AUTHORIZATION_BINDING_KEYS = frozenset(
    {
        "development_open_authorization_sha256",
        "preflight_plan_sha256",
        "preflight_result_sha256",
        "preflight_proof_evidence_sha256",
        "preflight_proof_gates_sha256",
        "preflight_operational_gates_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
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


def load_attempt08_development_open_bindings(
    development_open_authorization: str | Path,
    *,
    plan: str | Path = DEFAULT_PLAN_PATH,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
) -> dict[str, str]:
    """Validate the Go-only authorization and return its immutable bindings.

    This function is intentionally called before a lock, checkpoint, heartbeat,
    model, policy, observation, or teacher is opened by the development runner.
    """

    authorization_path = Path(development_open_authorization)
    plan_path = Path(plan)
    preflight_plan_path = Path(preflight_plan)
    payload = load_and_validate_development_open_authorization(
        authorization_path,
        attempt08_plan_path=plan_path,
        preflight_plan_path=preflight_plan_path,
    )
    preflight = payload.get("preflight_plan")
    result = payload.get("preflight_result")
    evidence = payload.get("evidence")
    if not all(isinstance(value, Mapping) for value in (preflight, result, evidence)):
        raise ValueError("Attempt08 development authorization bindings changed")
    assert isinstance(preflight, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(evidence, Mapping)
    bindings = {
        "development_open_authorization_sha256": _sha256_file(authorization_path),
        "preflight_plan_sha256": str(preflight.get("sha256")),
        "preflight_result_sha256": str(result.get("sha256")),
        "preflight_proof_evidence_sha256": str(
            evidence.get("proof_evidence_sha256")
        ),
        "preflight_proof_gates_sha256": str(evidence.get("proof_gates_sha256")),
        "preflight_operational_gates_sha256": str(
            evidence.get("operational_gates_sha256")
        ),
        "preflight_execution_evidence_sha256": str(
            evidence.get("preflight_execution_evidence_sha256")
        ),
        "runtime_semantic_anchor_sha256": str(
            evidence.get("runtime_semantic_anchor_sha256")
        ),
        "runtime_source_closure_sha256": str(
            evidence.get("runtime_source_closure_sha256")
        ),
        "runtime_fingerprint_sha256": str(
            evidence.get("runtime_fingerprint_sha256")
        ),
        "runtime_requirements_sha256": str(
            evidence.get("runtime_requirements_sha256")
        ),
        "gcp_image_name": str(evidence.get("gcp_image_name")),
        "gcp_image_id": str(evidence.get("gcp_image_id")),
    }
    if set(bindings) != _AUTHORIZATION_BINDING_KEYS:
        raise AssertionError("Attempt08 authorization binding schema changed")
    expected = {
        "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
        "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
    }
    if any(bindings.get(name) != value for name, value in expected.items()):
        raise ValueError("Attempt08 external runtime authorization changed")
    return {
        name: (
            value
            if name in {"gcp_image_name", "gcp_image_id"}
            else _require_sha256(value, name=name)
        )
        for name, value in bindings.items()
    }


def _canonical_path_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _process_peak_rss_bytes() -> int:
    """Return this process's OS high-water RSS in bytes."""

    if os.name == "nt":
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        if not get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource
    import sys

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _require_distinct_paths(
    paths: Mapping[str, Path],
    *,
    allowed_hardlink_pairs: frozenset[frozenset[str]] = frozenset(),
) -> None:
    identities: dict[str, str] = {}
    for name, path in paths.items():
        identity = _canonical_path_identity(path)
        if identity in identities:
            raise ValueError(
                f"Attempt08 paths must be distinct: {name} aliases "
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
                    "Attempt08 could not verify existing path identities"
                ) from exc
            if same_file:
                if frozenset((left_name, right_name)) in allowed_hardlink_pairs:
                    continue
                raise ValueError(
                    "Attempt08 paths must be distinct: "
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
            name="attempt08-development-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._write()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds + 1.0))
        if self.thread.is_alive():
            raise RuntimeError("Attempt08 heartbeat thread did not stop")
        if self.error is not None:
            raise RuntimeError("Attempt08 live heartbeat write failed") from self.error

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
    """Non-blocking process lock for one output shard."""

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
                "Attempt08 shard is already owned by another worker"
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


def _development_seeds(plan: Mapping[str, Any], root_index: int) -> dict[str, int]:
    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise TypeError("Attempt08 root_index must be an integer")
    if not 0 <= root_index < ATTEMPT08_DEVELOPMENT_ROOTS:
        raise ValueError("Attempt08 development root_index is outside 0..199")
    schedules = enumerate_attempt08_seed_schedules(plan, population="development")
    if set(schedules) != set(_SEED_DOMAINS):
        raise ValueError("Attempt08 development seed domains changed")
    seeds = {domain: int(schedules[domain][root_index]) for domain in _SEED_DOMAINS}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Attempt08 per-root seed domains overlap")
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
    authorization_bindings: Mapping[str, str],
    batch_child_selectors: bool,
    native_batch_threads: int,
) -> dict[str, Any]:
    if set(authorization_bindings) != _AUTHORIZATION_BINDING_KEYS:
        raise ValueError("Attempt08 authorization binding fields changed")
    return {
        "schema": ATTEMPT08_FIXED_CONTRACT_SCHEMA,
        "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
        "plan_sha256": plan_sha256,
        "ai_profiles_sha256": ai_profiles_sha256,
        "model_sha256": model_sha256,
        **dict(authorization_bindings),
        "root_index": root_index,
        "root_profile": root_profile,
        "seeds": dict(seeds),
        "run_id": run_id,
        "batch_child_selectors": batch_child_selectors,
        "native_batch_threads": native_batch_threads,
        "baseline_profile": ATTEMPT08_BASELINE_PROFILE,
        "continuation_profile": ATTEMPT08_CONTINUATION_PROFILE,
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


def _expected_provenance(
    *,
    fixed_contract: Mapping[str, Any],
    config_sha256: str,
    root_input_sha256: str,
) -> dict[str, Any]:
    hand_seed = int(fixed_contract["seeds"]["hand"])
    root_profile = str(fixed_contract["root_profile"])
    child_seed = int(fixed_contract["seeds"]["child"])
    return {
        "schema": ATTEMPT08_PROVENANCE_SCHEMA,
        "run_id": fixed_contract["run_id"],
        "root_index": fixed_contract["root_index"],
        "root_profile": root_profile,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "plan_sha256": fixed_contract["plan_sha256"],
        "ai_profiles_sha256": fixed_contract["ai_profiles_sha256"],
        "model_sha256": fixed_contract["model_sha256"],
        "development_open_authorization_sha256": fixed_contract[
            "development_open_authorization_sha256"
        ],
        "preflight_plan_sha256": fixed_contract["preflight_plan_sha256"],
        "preflight_result_sha256": fixed_contract["preflight_result_sha256"],
        "preflight_proof_evidence_sha256": fixed_contract[
            "preflight_proof_evidence_sha256"
        ],
        "preflight_proof_gates_sha256": fixed_contract[
            "preflight_proof_gates_sha256"
        ],
        "preflight_operational_gates_sha256": fixed_contract[
            "preflight_operational_gates_sha256"
        ],
        "preflight_execution_evidence_sha256": fixed_contract[
            "preflight_execution_evidence_sha256"
        ],
        "runtime_semantic_anchor_sha256": fixed_contract[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": fixed_contract[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": fixed_contract[
            "runtime_fingerprint_sha256"
        ],
        "runtime_requirements_sha256": fixed_contract[
            "runtime_requirements_sha256"
        ],
        "gcp_image_name": fixed_contract["gcp_image_name"],
        "gcp_image_id": fixed_contract["gcp_image_id"],
        "config_sha256": config_sha256,
        "root_input_sha256": root_input_sha256,
        "seeds": dict(fixed_contract["seeds"]),
        "rng_domains": dict(_RNG_DOMAIN_NAMES),
        "root_generation_policy": ATTEMPT08_ROOT_GENERATION_POLICY,
        "root_policy_profiles": {"first": root_profile, "second": root_profile},
        "root_policy_seeds": {
            seat: _profile_policy_seed(hand_seed, root_profile, seat)
            for seat in ("first", "second")
        },
        "baseline_profile": ATTEMPT08_BASELINE_PROFILE,
        "baseline_policy_seed": _profile_policy_seed(
            hand_seed, ATTEMPT08_BASELINE_PROFILE, "second"
        ),
        "continuation_profile": ATTEMPT08_CONTINUATION_PROFILE,
        "continuation_policy_seeds": {
            "first": child_seed,
            "second": child_seed + 1,
        },
        "explicit_loaded_profiles": sorted(
            {
                root_profile,
                ATTEMPT08_BASELINE_PROFILE,
                ATTEMPT08_CONTINUATION_PROFILE,
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
        "search_freeze_authorization_allowed": False,
        "future_audit_allowed": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "full_replacement_enabled": False,
    }


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    fixed_contract: Mapping[str, Any],
    config_sha256: str,
    root_input_sha256: str,
) -> None:
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("Attempt08 provenance fields changed")
    expected = _expected_provenance(
        fixed_contract=fixed_contract,
        config_sha256=config_sha256,
        root_input_sha256=root_input_sha256,
    )
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Attempt08 provenance {key} changed")


def _reject_hidden_discard_fields(value: Any, *, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "opponent_private_discards":
                raise ValueError(
                    f"Attempt08 output exposes opponent private discards at {path}"
                )
            _reject_hidden_discard_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_hidden_discard_fields(child, path=f"{path}[{index}]")


def _validate_completed_row_bytes(
    raw: bytes,
    *,
    fixed_contract: Mapping[str, Any],
    config_sha256: str,
) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError("Attempt08 completed output must contain exactly one row")
    try:
        row = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt08 completed output row is invalid") from exc
    if not isinstance(row, Mapping) or raw != _canonical_json_bytes(row):
        raise ValueError("Attempt08 completed output row is not canonical")
    expected_keys = {
        "schema",
        "root_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
        "teacher",
    }
    seeds = fixed_contract["seeds"]
    if (
        set(row) != expected_keys
        or row.get("schema") != ATTEMPT08_SHARD_ROW_SCHEMA
        or row.get("root_index") != fixed_contract["root_index"]
        or row.get("hand_seed") != seeds["hand"]
        or row.get("root_profile") != fixed_contract["root_profile"]
        or not isinstance(row.get("baseline_action_key"), str)
    ):
        raise ValueError("Attempt08 checkpointed row identity mismatch")
    observation_payload = row.get("policy_observation")
    if not isinstance(observation_payload, Mapping):
        raise ValueError("Attempt08 checkpointed observation is missing")
    observation = ActorObservation.from_dict(observation_payload)
    if observation.to_dict() != dict(observation_payload):
        raise ValueError("Attempt08 checkpointed observation changed")
    baseline_action_key = str(row["baseline_action_key"])
    root_input_hash = _root_input_sha256(
        root_index=int(fixed_contract["root_index"]),
        hand_seed=int(seeds["hand"]),
        root_profile=str(fixed_contract["root_profile"]),
        observation=observation,
        baseline_action_key=baseline_action_key,
    )
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Attempt08 checkpointed row provenance is missing")
    _validate_provenance(
        provenance,
        fixed_contract=fixed_contract,
        config_sha256=config_sha256,
        root_input_sha256=root_input_hash,
    )
    teacher = row.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError("Attempt08 checkpointed teacher result is missing")
    if (
        teacher.get("status") != "ok"
        or teacher.get("schema") != ATTEMPT08_TEACHER_SCHEMA
        or teacher.get("observation_fingerprint") != observation.fingerprint()
        or teacher.get("baseline_action_key") != baseline_action_key
        or teacher.get("runtime_gate_allowed") is not False
        or teacher.get("profile_activation_allowed") is not False
        or teacher.get("current_profile_resolved") is not False
        or teacher.get("development_only") is not True
    ):
        raise ValueError("Attempt08 checkpointed teacher identity changed")
    _reject_hidden_discard_fields(row)
    return row


def _resume_one_root(
    partial_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    config_sha256: str,
    plan_sha256: str,
    ai_profiles_sha256: str,
    model_sha256: str,
    authorization_bindings: Mapping[str, str],
    expected_root_index: int,
    fixed_contract: Mapping[str, Any],
) -> tuple[int, float | None, int | None]:
    checkpoint: Mapping[str, Any] | None = None
    if checkpoint_path.exists():
        checkpoint_raw = checkpoint_path.read_bytes()
        loaded = json.loads(checkpoint_raw.decode("utf-8-sig"))
        if (
            not isinstance(loaded, Mapping)
            or checkpoint_raw != _canonical_json_bytes(loaded)
        ):
            raise ValueError("Attempt08 checkpoint must be a mapping")
        checkpoint = loaded
        expected = {
            "schema": ATTEMPT08_CHECKPOINT_SCHEMA,
            "config_sha256": config_sha256,
            "plan_sha256": plan_sha256,
            "ai_profiles_sha256": ai_profiles_sha256,
            "model_sha256": model_sha256,
            **dict(authorization_bindings),
            "target_roots": 1,
            "root_index": expected_root_index,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("Attempt08 checkpoint configuration or hash mismatch")
        base_keys = {
            *expected,
            "completed_roots",
            "partial_sha256",
            "updated_unix_seconds",
        }
        updated = checkpoint.get("updated_unix_seconds")
        if (
            isinstance(updated, bool)
            or not isinstance(updated, (int, float))
            or not math.isfinite(float(updated))
            or float(updated) < 0.0
        ):
            raise ValueError("Attempt08 checkpoint update time is invalid")
        if checkpoint.get("completed_roots") not in (0, 1):
            raise ValueError("Attempt08 checkpoint completed_roots is invalid")
        if checkpoint.get("completed_roots") == 0:
            if set(checkpoint) != base_keys:
                raise ValueError("Attempt08 incomplete checkpoint has generator metrics")
        else:
            if set(checkpoint) != base_keys | {
                "generator_elapsed_seconds",
                "generator_peak_rss_bytes",
            }:
                raise ValueError("Attempt08 completed checkpoint fields changed")
            elapsed = checkpoint.get("generator_elapsed_seconds")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
            ):
                raise ValueError("Attempt08 completed checkpoint generator elapsed is invalid")
            peak_rss = checkpoint.get("generator_peak_rss_bytes")
            if type(peak_rss) is not int or peak_rss < 0:
                raise ValueError("Attempt08 completed checkpoint generator RSS is invalid")

    if output_path.exists():
        if checkpoint is None or checkpoint.get("completed_roots") != 1:
            raise ValueError("Attempt08 completed output has no completed checkpoint")
        raw = output_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checkpoint.get("partial_sha256"):
            raise ValueError("Attempt08 completed output hash disagrees with checkpoint")
        _validate_completed_row_bytes(
            raw,
            fixed_contract=fixed_contract,
            config_sha256=config_sha256,
        )
        if partial_path.exists() and partial_path.read_bytes() != raw:
            raise ValueError("Attempt08 partial and completed output differ")
        return (
            1,
            float(checkpoint["generator_elapsed_seconds"]),
            int(checkpoint["generator_peak_rss_bytes"]),
        )

    if not partial_path.exists():
        if checkpoint is None:
            return 0, None, None
        if (
            checkpoint.get("completed_roots") == 0
            and checkpoint.get("partial_sha256") == hashlib.sha256(b"").hexdigest()
        ):
            return 0, None, None
        raise ValueError("Attempt08 completed checkpoint has no partial output")

    raw = partial_path.read_bytes()
    if checkpoint is None:
        if raw:
            with partial_path.open("r+b") as handle:
                handle.truncate(0)
        return 0, None, None
    completed = int(checkpoint["completed_roots"])
    newline_offsets = [index + 1 for index, byte in enumerate(raw) if byte == 10]
    boundary = newline_offsets[completed - 1] if completed else 0
    if len(raw) != boundary:
        with partial_path.open("r+b") as handle:
            handle.truncate(boundary)
        raw = raw[:boundary]
    if hashlib.sha256(raw).hexdigest() != checkpoint.get("partial_sha256"):
        raise ValueError("Attempt08 partial hash disagrees with checkpoint")
    if completed:
        _validate_completed_row_bytes(
            raw,
            fixed_contract=fixed_contract,
            config_sha256=config_sha256,
        )
    return (
        completed,
        float(checkpoint["generator_elapsed_seconds"]) if completed else None,
        int(checkpoint["generator_peak_rss_bytes"]) if completed else None,
    )


def _run_development_shard_locked(
    *,
    root_index: int,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    run_id: str,
    development_open_authorization: str | Path,
    preflight_plan: str | Path,
    plan: str | Path = DEFAULT_PLAN_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    batch_child_selectors: bool = False,
    native_batch_threads: int = ATTEMPT08_NATIVE_BATCH_THREADS,
    paths: ModelPaths | None = None,
    _validated_authorization_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    authorization_bindings = (
        dict(_validated_authorization_bindings)
        if _validated_authorization_bindings is not None
        else load_attempt08_development_open_bindings(
            development_open_authorization,
            plan=plan,
            preflight_plan=preflight_plan,
        )
    )
    if set(authorization_bindings) != _AUTHORIZATION_BINDING_KEYS:
        raise ValueError("Attempt08 authorization binding fields changed")
    if not isinstance(batch_child_selectors, bool):
        raise TypeError("batch_child_selectors must be a bool")
    if batch_child_selectors is not True:
        raise ValueError("Attempt08 development requires batched child selectors")
    if (
        isinstance(native_batch_threads, bool)
        or not isinstance(native_batch_threads, int)
        or native_batch_threads != ATTEMPT08_NATIVE_BATCH_THREADS
    ):
        raise ValueError("Attempt08 development native_batch_threads is fixed at 4")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must not be empty")

    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    checkpoint_path = Path(checkpoint)
    heartbeat_path = Path(heartbeat)
    model_path = Path(model)
    plan_path = Path(plan)
    ai_profiles_path = Path(ai_profiles)
    preflight_plan_path = Path(preflight_plan)
    authorization_path = Path(development_open_authorization)
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
            "preflight_plan": preflight_plan_path,
            "development_open_authorization": authorization_path,
            "lock": lock_path,
        },
        allowed_hardlink_pairs=frozenset({frozenset(("output", "partial"))}),
    )
    output_preexisting = output_path.exists()

    plan_payload = load_and_validate_attempt08_plan(plan_path)
    plan_hash = _sha256_file(plan_path)
    if plan_hash != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 plan byte SHA-256 changed")
    ai_profiles_hash = _sha256_file(ai_profiles_path)
    if ai_profiles_hash != AI_PROFILES_SHA256:
        raise ValueError("Attempt08 ai_profiles.py SHA-256 changed")
    expected_model_hash = _require_sha256(model_sha256, name="model_sha256")
    if expected_model_hash != ATTEMPT08_LAMBDA_MODEL_SHA256:
        raise ValueError("Attempt08 model is not the frozen Lambda artifact")

    seeds = _development_seeds(plan_payload, root_index)
    root_profile = M43_ATTEMPT08_PROFILES[
        root_index % len(M43_ATTEMPT08_PROFILES)
    ]
    fixed_contract = _fixed_contract(
        root_index=root_index,
        root_profile=root_profile,
        seeds=seeds,
        run_id=run_id,
        plan_sha256=plan_hash,
        ai_profiles_sha256=ai_profiles_hash,
        model_sha256=expected_model_hash,
        authorization_bindings=authorization_bindings,
        batch_child_selectors=batch_child_selectors,
        native_batch_threads=native_batch_threads,
    )
    config_sha256 = _canonical_sha256(fixed_contract)
    completed, generator_elapsed_seconds, generator_peak_rss_bytes = _resume_one_root(
        partial_path,
        output_path,
        checkpoint_path,
        config_sha256=config_sha256,
        plan_sha256=plan_hash,
        ai_profiles_sha256=ai_profiles_hash,
        model_sha256=expected_model_hash,
        authorization_bindings=authorization_bindings,
        expected_root_index=root_index,
        fixed_contract=fixed_contract,
    )
    if completed == 0:
        generation_started = time.perf_counter()
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        starting_common = {
            "config_sha256": config_sha256,
            "plan_sha256": plan_hash,
            "ai_profiles_sha256": ai_profiles_hash,
            "model_sha256": expected_model_hash,
            **authorization_bindings,
            "completed_roots": 0,
            "target_roots": 1,
            "root_index": root_index,
            "partial_sha256": empty_sha256,
        }
        _atomic_json(
            checkpoint_path,
            {
                "schema": ATTEMPT08_CHECKPOINT_SCHEMA,
                **starting_common,
                "updated_unix_seconds": time.time(),
            },
        )
        heartbeat_pump = _HeartbeatPump(
            heartbeat_path,
            {
                "schema": ATTEMPT08_HEARTBEAT_SCHEMA,
                "status": "running",
                **starting_common,
            },
        )
        heartbeat_pump.start()
        try:
            explicit_profiles = {
                root_profile,
                ATTEMPT08_BASELINE_PROFILE,
                ATTEMPT08_CONTINUATION_PROFILE,
            }
            if "current" in explicit_profiles:
                raise AssertionError("Attempt08 must never resolve current")
            bundle = load_model_bundle(paths or ModelPaths(), profiles=explicit_profiles)
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
                seeds["hand"], ATTEMPT08_BASELINE_PROFILE, "second"
            )
            baseline_policy = build_policy(
                ATTEMPT08_BASELINE_PROFILE,
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
                    ATTEMPT08_CONTINUATION_PROFILE,
                    bundle,
                    seed=continuation_policy_seeds[seat],
                    seat=seat,
                    opening_lookahead_samples=0,
                )
                for seat in ("first", "second")
            }
            _require_concrete_stage9f_p2_policies(t2_policies)
            ranker = FrozenAttempt08LambdaRanker.load(
                model_path, expected_sha256=expected_model_hash
            )
            teacher_config = Attempt08TeacherConfig(
                frozen_model_sha256=expected_model_hash,
                hand_seed=seeds["hand"],
                rerank_seed=seeds["rerank"],
                veto_seed=seeds["veto"],
                stress_seed=seeds["stress"],
                assessment_seed=seeds["assessment"],
                child_policy_seed=seeds["child"],
                run_id=(
                    f"{run_id}:root={root_index}:seed={seeds['hand']}:"
                    f"obs={observation.fingerprint()}"
                ),
                batch_child_selectors=batch_child_selectors,
            )
            with _native_batch_threads(batch_child_selectors, native_batch_threads):
                teacher = evaluate_attempt08_t1_second(
                    observation,
                    baseline_action_key=baseline_token,
                    ranker=ranker,
                    t2_policies=t2_policies,
                    config=teacher_config,
                )
        finally:
            heartbeat_pump.stop()

        validate_attempt08_teacher_output(
            observation,
            baseline_action_key=baseline_token,
            payload=teacher,
            config=teacher_config,
        )
        if (
            teacher.get("status") != "ok"
            or teacher.get("schema") != ATTEMPT08_TEACHER_SCHEMA
            or teacher.get("observation_fingerprint") != observation.fingerprint()
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
            raise ValueError("Attempt08 teacher result identity changed")
        root_input_hash = _root_input_sha256(
            root_index=root_index,
            hand_seed=seeds["hand"],
            root_profile=root_profile,
            observation=observation,
            baseline_action_key=baseline_token,
        )
        provenance = _expected_provenance(
            fixed_contract=fixed_contract,
            config_sha256=config_sha256,
            root_input_sha256=root_input_hash,
        )
        _validate_provenance(
            provenance,
            fixed_contract=fixed_contract,
            config_sha256=config_sha256,
            root_input_sha256=root_input_hash,
        )
        row = {
            "schema": ATTEMPT08_SHARD_ROW_SCHEMA,
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
        generator_elapsed_seconds = time.perf_counter() - generation_started
        generator_peak_rss_bytes = _process_peak_rss_bytes()
        common = {
            "config_sha256": config_sha256,
            "plan_sha256": plan_hash,
            "ai_profiles_sha256": ai_profiles_hash,
            "model_sha256": expected_model_hash,
            **authorization_bindings,
            "completed_roots": 1,
            "target_roots": 1,
            "root_index": root_index,
            "partial_sha256": _sha256_file(partial_path),
            "updated_unix_seconds": time.time(),
            "generator_elapsed_seconds": generator_elapsed_seconds,
            "generator_peak_rss_bytes": generator_peak_rss_bytes,
        }
        _atomic_json(
            checkpoint_path,
            {"schema": ATTEMPT08_CHECKPOINT_SCHEMA, **common},
        )

    if not output_preexisting:
        try:
            os.link(partial_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Attempt08 shard concurrently completed: {output_path}"
            ) from exc
    summary = {
        "schema": ATTEMPT08_SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "root_index": root_index,
        "completed_roots": 1,
        "target_roots": 1,
        "config_sha256": config_sha256,
        "plan_sha256": plan_hash,
        "ai_profiles_sha256": ai_profiles_hash,
        "model_sha256": expected_model_hash,
        **authorization_bindings,
        "output_sha256": _sha256_file(output_path),
        "elapsed_seconds": generator_elapsed_seconds,
        "generator_peak_rss_bytes": generator_peak_rss_bytes,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    completed_heartbeat = {
        **summary,
        "schema": ATTEMPT08_HEARTBEAT_SCHEMA,
        "summary_schema": ATTEMPT08_SHARD_SUMMARY_SCHEMA,
    }
    if output_preexisting and heartbeat_path.exists():
        heartbeat_raw = heartbeat_path.read_bytes()
        try:
            existing_heartbeat = json.loads(heartbeat_raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Attempt08 recovery heartbeat is invalid") from exc
        if (
            not isinstance(existing_heartbeat, Mapping)
            or heartbeat_raw != _canonical_json_bytes(existing_heartbeat)
        ):
            raise ValueError("Attempt08 recovery heartbeat is not canonical")
        if existing_heartbeat.get("status") == "complete":
            if dict(existing_heartbeat) != completed_heartbeat:
                raise ValueError("Attempt08 completed heartbeat disagrees with output")
            if partial_path.exists():
                partial_path.unlink()
            return summary
        if (
            existing_heartbeat.get("schema") != ATTEMPT08_HEARTBEAT_SCHEMA
            or existing_heartbeat.get("status") != "running"
            or existing_heartbeat.get("root_index") != root_index
            or existing_heartbeat.get("config_sha256") != config_sha256
        ):
            raise ValueError("Attempt08 recovery heartbeat identity changed")
    if partial_path.exists():
        partial_path.unlink()
    _atomic_json(heartbeat_path, completed_heartbeat)
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
    development_open_authorization: str | Path,
    preflight_plan: str | Path,
    plan: str | Path = DEFAULT_PLAN_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    batch_child_selectors: bool = False,
    native_batch_threads: int = ATTEMPT08_NATIVE_BATCH_THREADS,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Run exactly one development root under a no-clobber process lock."""

    authorization_bindings = load_attempt08_development_open_bindings(
        development_open_authorization,
        plan=plan,
        preflight_plan=preflight_plan,
    )

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
            "preflight_plan": Path(preflight_plan),
            "development_open_authorization": Path(
                development_open_authorization
            ),
            "lock": lock_path,
        },
        allowed_hardlink_pairs=frozenset({frozenset(("output", "partial"))}),
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
            development_open_authorization=development_open_authorization,
            preflight_plan=preflight_plan,
            plan=plan,
            ai_profiles=ai_profiles,
            batch_child_selectors=batch_child_selectors,
            native_batch_threads=native_batch_threads,
            paths=paths,
            _validated_authorization_bindings=authorization_bindings,
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
    parser.add_argument(
        "--development-open-authorization", type=Path, required=True
    )
    parser.add_argument("--preflight-plan", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--ai-profiles", type=Path, default=DEFAULT_AI_PROFILES_PATH)
    parser.add_argument("--batch-child-selectors", action="store_true")
    parser.add_argument(
        "--native-batch-threads",
        type=int,
        default=ATTEMPT08_NATIVE_BATCH_THREADS,
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
        development_open_authorization=args.development_open_authorization,
        preflight_plan=args.preflight_plan,
        plan=args.plan,
        ai_profiles=args.ai_profiles,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT08_CHECKPOINT_SCHEMA",
    "ATTEMPT08_HEARTBEAT_SCHEMA",
    "ATTEMPT08_PROVENANCE_SCHEMA",
    "ATTEMPT08_SHARD_ROW_SCHEMA",
    "ATTEMPT08_SHARD_SUMMARY_SCHEMA",
    "DEFAULT_AI_PROFILES_PATH",
    "DEFAULT_PLAN_PATH",
    "DEFAULT_PREFLIGHT_PLAN_PATH",
    "load_attempt08_development_open_bindings",
    "parse_args",
    "run_development_shard",
]
