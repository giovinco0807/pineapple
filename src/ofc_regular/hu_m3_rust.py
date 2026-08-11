"""ctypes binding for the M3 Rust HU rollout/search engine.

The binding sends one versioned JSON request through the same engine entrypoint
used by the resumable Rust shard runner.  It never supplies replay truth,
opponent private discards, or a realized deck tail to a policy root.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_infoset import ActorObservation
from .hu_late_street_teacher import T4SearchConfig
from .hu_turn3_joint_exact_teacher import JointExactConfig


HU_M3_REQUEST_SCHEMA = "hu_m3_engine_request_v1"
HU_M3_BATCH_REQUEST_SCHEMA = "hu_m3_engine_batch_request_v1"
HU_M3_BINDING_SCHEMA = "hu_m3_python_binding_v1"
_T3_EXPLICIT_SUPPORT_WORLD_FIELDS = frozenset(
    {
        "opponent_private_discards",
        "future_cards",
        "weight",
        "world_id",
    }
)


class HuM3RustError(RuntimeError):
    """Raised when the native engine rejects or cannot evaluate a request."""


@dataclass(frozen=True)
class HuM3BuildResult:
    library_path: Path
    command: tuple[str, ...]
    built: bool


_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_PATH: Path | None = None
_LOCK = threading.RLock()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cargo_target_directory(*, target_dir: str | Path | None = None) -> Path:
    """Resolve Cargo's effective target directory, including isolated builds."""

    configured = target_dir if target_dir is not None else os.environ.get("CARGO_TARGET_DIR")
    if not configured:
        return repository_root() / "target"
    path = Path(configured)
    return path if path.is_absolute() else repository_root() / path


def native_library_path(
    *, release: bool = True, target_dir: str | Path | None = None
) -> Path:
    profile = "release" if release else "debug"
    if sys.platform == "win32":
        name = "ofc_hu_m3_engine.dll"
    elif sys.platform == "darwin":
        name = "libofc_hu_m3_engine.dylib"
    else:
        name = "libofc_hu_m3_engine.so"
    return cargo_target_directory(target_dir=target_dir) / profile / name


def runner_path(*, release: bool = True, target_dir: str | Path | None = None) -> Path:
    profile = "release" if release else "debug"
    suffix = ".exe" if sys.platform == "win32" else ""
    return (
        cargo_target_directory(target_dir=target_dir)
        / profile
        / f"ofc_hu_m3_runner{suffix}"
    )


def build_native_engine(
    *, release: bool = True, target_dir: str | Path | None = None
) -> HuM3BuildResult:
    command = ["cargo", "build", "-p", "ofc_hu_m3_engine"]
    if release:
        command.append("--release")
    env = os.environ.copy()
    if target_dir is not None:
        env["CARGO_TARGET_DIR"] = str(
            cargo_target_directory(target_dir=target_dir).resolve()
        )
    env.setdefault("CARGO_TERM_COLOR", "never")
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise HuM3RustError(
            "Rust M3 build failed:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-8000:]
        )
    path = native_library_path(release=release, target_dir=target_dir)
    if not path.is_file():
        raise HuM3RustError(f"Rust build succeeded but library is missing: {path}")
    return HuM3BuildResult(path, tuple(command), True)


def load_native_engine(
    *,
    path: str | Path | None = None,
    build_if_missing: bool = False,
    release: bool = True,
) -> ctypes.CDLL:
    global _LIBRARY, _LIBRARY_PATH
    resolved = Path(path).resolve() if path is not None else native_library_path(release=release)
    with _LOCK:
        if _LIBRARY is not None and _LIBRARY_PATH == resolved:
            return _LIBRARY
        if not resolved.is_file():
            if not build_if_missing:
                raise HuM3RustError(f"Rust M3 library does not exist: {resolved}")
            resolved = build_native_engine(release=release).library_path.resolve()
        try:
            library = ctypes.CDLL(str(resolved))
        except OSError as exc:
            raise HuM3RustError(f"failed to load Rust M3 library {resolved}: {exc}") from exc
        library.ofc_hu_m3_engine_version.argtypes = []
        library.ofc_hu_m3_engine_version.restype = ctypes.c_char_p
        library.ofc_hu_m3_evaluate_alloc.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
        ]
        library.ofc_hu_m3_evaluate_alloc.restype = ctypes.c_void_p
        library.ofc_hu_m3_free_string.argtypes = [ctypes.c_void_p]
        library.ofc_hu_m3_free_string.restype = None
        _LIBRARY = library
        _LIBRARY_PATH = resolved
        return library


def engine_version(*, library: ctypes.CDLL | None = None) -> str:
    native = library or load_native_engine()
    raw = native.ofc_hu_m3_engine_version()
    if not raw:
        raise HuM3RustError("native engine returned a null version")
    return raw.decode("ascii")


def evaluate_request(
    payload: Mapping[str, Any],
    *,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    native = library or load_native_engine()
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    buffer = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
    pointer = native.ofc_hu_m3_evaluate_alloc(buffer, len(encoded))
    if not pointer:
        raise HuM3RustError("native engine returned a null response")
    try:
        raw = ctypes.string_at(pointer)
    finally:
        native.ofc_hu_m3_free_string(pointer)
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuM3RustError(f"native engine returned invalid JSON: {raw[:500]!r}") from exc
    if not isinstance(response, dict):
        raise HuM3RustError("native engine response must be a JSON object")
    if response.get("status") == "error":
        raise HuM3RustError(str(response.get("error", "native engine error")))
    return response


def t4_request(
    observation: ActorObservation,
    *,
    config: T4SearchConfig | None = None,
) -> dict[str, Any]:
    _require_observation(observation, street="T4")
    config = config or T4SearchConfig()
    return {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "t4",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": {
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "seed": config.seed,
            "candidate_seed": config.seed if config.candidate_seed is None else config.candidate_seed,
            "evaluation_seed": config.seed if config.evaluation_seed is None else config.evaluation_seed,
            "run_id": config.run_id,
        },
    }


def t3_request(
    observation: ActorObservation,
    *,
    config: JointExactConfig | None = None,
) -> dict[str, Any]:
    _require_observation(observation, street="T3")
    config = config or JointExactConfig(
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    return {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "t3",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": _joint_config_payload(config),
    }


def _joint_config_payload(config: JointExactConfig) -> dict[str, Any]:
    payload = {
        "candidate_samples": config.candidate_samples,
        "evaluation_samples": config.evaluation_samples,
        "downstream_t3_samples": config.downstream_t3_samples,
        "downstream_t4_samples": config.downstream_t4_samples,
        "seed": config.seed,
        "candidate_seed": config.seed if config.candidate_seed is None else config.candidate_seed,
        "evaluation_seed": config.seed if config.evaluation_seed is None else config.evaluation_seed,
        "run_id": config.run_id,
        "use_t4_action_cache": config.use_final_turn_cache,
    }
    # Emitted only when opted into, so a request that does not use the learned
    # leaf is byte-identical to one built before the option existed.
    if config.learned_t4_model_path is not None:
        payload["learned_t4_model_path"] = config.learned_t4_model_path
    if config.learned_t4_model_sha256 is not None:
        payload["learned_t4_model_sha256"] = config.learned_t4_model_sha256
    if config.learned_t3_second_model_path is not None:
        payload["learned_t3_second_model_path"] = config.learned_t3_second_model_path
    if config.learned_t3_second_model_sha256 is not None:
        payload["learned_t3_second_model_sha256"] = config.learned_t3_second_model_sha256
    if config.learned_t3_first_model_path is not None:
        payload["learned_t3_first_model_path"] = config.learned_t3_first_model_path
    if config.learned_t3_first_model_sha256 is not None:
        payload["learned_t3_first_model_sha256"] = config.learned_t3_first_model_sha256
    if config.learned_t2_second_model_path is not None:
        payload["learned_t2_second_model_path"] = config.learned_t2_second_model_path
    if config.learned_t2_second_model_sha256 is not None:
        payload["learned_t2_second_model_sha256"] = config.learned_t2_second_model_sha256
    if config.learned_t2_first_model_path is not None:
        payload["learned_t2_first_model_path"] = config.learned_t2_first_model_path
    if config.learned_t2_first_model_sha256 is not None:
        payload["learned_t2_first_model_sha256"] = config.learned_t2_first_model_sha256
    if config.learned_t1_second_model_path is not None:
        payload["learned_t1_second_model_path"] = config.learned_t1_second_model_path
    if config.learned_t1_second_model_sha256 is not None:
        payload["learned_t1_second_model_sha256"] = config.learned_t1_second_model_sha256
    if config.learned_t1_first_model_path is not None:
        payload["learned_t1_first_model_path"] = config.learned_t1_first_model_path
    if config.learned_t1_first_model_sha256 is not None:
        payload["learned_t1_first_model_sha256"] = config.learned_t1_first_model_sha256
    if config.learned_t0_second_model_path is not None:
        payload["learned_t0_second_model_path"] = config.learned_t0_second_model_path
    if config.learned_t0_second_model_sha256 is not None:
        payload["learned_t0_second_model_sha256"] = config.learned_t0_second_model_sha256
    # The T0 first-seat root policy, on the same emit-only-when-set terms as
    # everything above. Only a `decide` request reads it; no evaluation in the
    # engine has a T0 first-seat decision ahead of it.
    if config.learned_t0_first_model_path is not None:
        payload["learned_t0_first_model_path"] = config.learned_t0_first_model_path
    if config.learned_t0_first_model_sha256 is not None:
        payload["learned_t0_first_model_sha256"] = config.learned_t0_first_model_sha256
    # The coarse T2 pair, emitted only when pinned so a request that does not
    # use it stays byte-identical to one built before it existed. The engine
    # answers both T2 replies through these when they are present and reports
    # those evaluators as "learned_fast".
    if config.fast_t2_second_model_path is not None:
        payload["fast_t2_second_model_path"] = config.fast_t2_second_model_path
    if config.fast_t2_second_model_sha256 is not None:
        payload["fast_t2_second_model_sha256"] = config.fast_t2_second_model_sha256
    if config.fast_t2_first_model_path is not None:
        payload["fast_t2_first_model_path"] = config.fast_t2_first_model_path
    if config.fast_t2_first_model_sha256 is not None:
        payload["fast_t2_first_model_sha256"] = config.fast_t2_first_model_sha256
    # The coarse T1 pair, emitted only when pinned so a request that does not
    # use it stays byte-identical to one built before it existed. The engine
    # answers both T1 replies through these when they are present and reports
    # those evaluators as "learned_fast".
    if config.fast_t1_second_model_path is not None:
        payload["fast_t1_second_model_path"] = config.fast_t1_second_model_path
    if config.fast_t1_second_model_sha256 is not None:
        payload["fast_t1_second_model_sha256"] = config.fast_t1_second_model_sha256
    if config.fast_t1_first_model_path is not None:
        payload["fast_t1_first_model_path"] = config.fast_t1_first_model_path
    if config.fast_t1_first_model_sha256 is not None:
        payload["fast_t1_first_model_sha256"] = config.fast_t1_first_model_sha256
    # The coarse T0 second-seat reply, on the same terms. Only a T0 first-seat
    # evaluation reaches it; the engine reports it as "learned_fast" there.
    if config.fast_t0_second_model_path is not None:
        payload["fast_t0_second_model_path"] = config.fast_t0_second_model_path
    if config.fast_t0_second_model_sha256 is not None:
        payload["fast_t0_second_model_sha256"] = config.fast_t0_second_model_sha256
    # Emitted only when the two-stage schedule is actually requested, so a
    # single-stage request stays byte-identical to one built before it existed.
    if config.prefilter_samples:
        payload["prefilter_samples"] = config.prefilter_samples
    if config.prefilter_keep:
        payload["prefilter_keep"] = config.prefilter_keep
    # The two pruning-safety mechanisms, emitted on the same terms as
    # everything else optional above: only when asked for, so a request that
    # took neither is byte-identical to one built before they existed.
    if config.prefilter_margin:
        payload["prefilter_margin"] = config.prefilter_margin
    # Omitted when zero, so a request that does not narrow is byte-identical to
    # one emitted before the field existed.
    if getattr(config, "learned_prefilter_keep", 0):
        payload["learned_prefilter_keep"] = config.learned_prefilter_keep
    if config.audit_full_every:
        payload["audit_full_every"] = config.audit_full_every
    # Stage-two racing, on the same terms again. An empty schedule is the
    # uniform stage two, and emits nothing.
    if config.race_schedule:
        payload["race_schedule"] = list(config.race_schedule)
        payload["race_lcb_z"] = config.race_lcb_z
    return payload


def t2_request(
    observation: ActorObservation,
    *,
    config: JointExactConfig,
) -> dict[str, Any]:
    """The T2 request reuses the T3 config payload; the engine's T2 evaluator
    additionally refuses to run unless every learned-evaluator digest is set."""

    _require_observation(observation, street="T2")
    return {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "t2",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": _joint_config_payload(config),
    }


def evaluate_t2(
    observation: ActorObservation,
    *,
    config: JointExactConfig,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(t2_request(observation, config=config), library=library)


def t1_request(
    observation: ActorObservation,
    *,
    config: JointExactConfig,
) -> dict[str, Any]:
    """The T1 request reuses the T3 config payload; the engine's T1 evaluator
    additionally refuses to run unless every learned-evaluator digest is set,
    and refuses the first seat outright until its evaluator exists."""

    _require_observation(observation, street="T1")
    return {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "t1",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": _joint_config_payload(config),
    }


def evaluate_t1(
    observation: ActorObservation,
    *,
    config: JointExactConfig,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(t1_request(observation, config=config), library=library)


def t0_request(
    observation: ActorObservation,
    *,
    config: JointExactConfig,
) -> dict[str, Any]:
    """The T0 request reuses the T3 config payload; the engine's T0 evaluator
    additionally refuses to run unless every learned-evaluator digest its seat
    needs is set -- seven acting second, eight acting first."""

    _require_observation(observation, street="T0")
    return {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "t0",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": _joint_config_payload(config),
    }


def evaluate_t0(
    observation: ActorObservation,
    *,
    config: JointExactConfig,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(t0_request(observation, config=config), library=library)


def t3_abr_component_request(
    observation: ActorObservation,
    *,
    config: JointExactConfig | None = None,
) -> dict[str, Any]:
    """Build the diagnostic ABR request without adding any private truth."""

    request = t3_request(observation, config=config)
    request["kind"] = "t3_abr_components"
    return request


def t3_explicit_support_request(
    observation: ActorObservation,
    *,
    worlds: Sequence[Mapping[str, Any]],
    continuation_policy_id: str,
    selector_mode: str = "canonical_min_action_key_v1",
) -> dict[str, Any]:
    """Build the reduced-support exact oracle request.

    The declared worlds are offline oracle support, not policy inputs. Every
    downstream Rust selector is fixed to an ActorObservation-only canonical
    policy, and the result never claims full-tree exactness.
    """

    _require_observation(observation, street="T3")
    if not worlds:
        raise ValueError("explicit T3 support must not be empty")
    if not continuation_policy_id:
        raise ValueError("continuation_policy_id must not be empty")
    normalized_worlds = _validate_t3_explicit_support_worlds(worlds)
    return {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "t3_explicit_support",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": {
            "worlds": normalized_worlds,
            "continuation_policy_id": continuation_policy_id,
            "selector_mode": selector_mode,
        },
    }


def evaluate_t4(
    observation: ActorObservation,
    *,
    config: T4SearchConfig | None = None,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(t4_request(observation, config=config), library=library)


def evaluate_t3(
    observation: ActorObservation,
    *,
    config: JointExactConfig | None = None,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(t3_request(observation, config=config), library=library)


def evaluate_t3_abr_components(
    observation: ActorObservation,
    *,
    config: JointExactConfig | None = None,
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(
        t3_abr_component_request(observation, config=config),
        library=library,
    )


def evaluate_t3_explicit_support(
    observation: ActorObservation,
    *,
    worlds: Sequence[Mapping[str, Any]],
    continuation_policy_id: str,
    selector_mode: str = "canonical_min_action_key_v1",
    library: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    return evaluate_request(
        t3_explicit_support_request(
            observation,
            worlds=worlds,
            continuation_policy_id=continuation_policy_id,
            selector_mode=selector_mode,
        ),
        library=library,
    )


def evaluate_batch(
    requests: Sequence[Mapping[str, Any]],
    *,
    library: ctypes.CDLL | None = None,
) -> list[dict[str, Any]]:
    response = evaluate_request(
        {
            "schema": HU_M3_BATCH_REQUEST_SCHEMA,
            "requests": [dict(request) for request in requests],
        },
        library=library,
    )
    results = response.get("results")
    if not isinstance(results, list) or not all(isinstance(row, dict) for row in results):
        raise HuM3RustError("native batch response has no result list")
    return results


def _require_observation(value: object, *, street: str) -> ActorObservation:
    if not isinstance(value, ActorObservation):
        raise TypeError(
            "Rust M3 binding requires ActorObservation; WorldState and replay truth are forbidden"
        )
    if value.street != street:
        raise ValueError(f"expected {street} observation, got {value.street}")
    return value


def _validate_t3_explicit_support_worlds(
    worlds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the public finite-support envelope before native dispatch.

    The Python exact oracle accepts only finite, strictly positive weights
    normalized to one.  Keep that contract at this public binding even though
    the lower-level Rust evaluator also supports general weighted means.  Card
    validity, hidden-card overlap, consumed-card counts, duplicate worlds, and
    unique world IDs remain native validation responsibilities.
    """

    normalized: list[dict[str, Any]] = []
    weights: list[float] = []
    for index, world in enumerate(worlds):
        if not isinstance(world, Mapping):
            raise TypeError(f"explicit T3 world {index} must be a mapping")
        fields = frozenset(world)
        if fields != _T3_EXPLICIT_SUPPORT_WORLD_FIELDS:
            missing = sorted(_T3_EXPLICIT_SUPPORT_WORLD_FIELDS - fields)
            unknown = sorted(fields - _T3_EXPLICIT_SUPPORT_WORLD_FIELDS, key=str)
            raise ValueError(
                f"invalid explicit T3 world {index} schema: "
                f"missing={missing!r}, unknown={unknown!r}"
            )

        opponent_discards = world["opponent_private_discards"]
        future_cards = world["future_cards"]
        for field_name, cards in (
            ("opponent_private_discards", opponent_discards),
            ("future_cards", future_cards),
        ):
            if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
                raise TypeError(
                    f"explicit T3 world {index} {field_name} must be a card sequence"
                )
            if not all(isinstance(card, str) for card in cards):
                raise TypeError(
                    f"explicit T3 world {index} {field_name} must contain strings"
                )

        world_id = world["world_id"]
        if not isinstance(world_id, str):
            raise TypeError(f"explicit T3 world {index} world_id must be a string")

        raw_weight = world["weight"]
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"explicit T3 world {index} weight must be a real number")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "explicit T3 world weights must be finite and strictly positive"
            )
        weights.append(weight)
        normalized.append(
            {
                "opponent_private_discards": list(opponent_discards),
                "future_cards": list(future_cards),
                "weight": weight,
                "world_id": world_id,
            }
        )

    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("explicit T3 support weights must sum to one")
    return normalized
