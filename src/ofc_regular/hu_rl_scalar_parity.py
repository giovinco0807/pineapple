"""Python oracle and strict comparator for the Rust scalar HU RL bridge.

Each individual actor view remains information-set safe.  The complete trace
also records both actors' selected keys and therefore is privileged correctness
audit material: it is never eligible for policy input, replay, or training.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import hu_rl_reference as _hu_rl_reference_module
from . import teacher as _teacher_module
from .action_key import ActionKey
from .cards import ALL_CARDS, validate_cards
from .hu_infoset import FL_EV_CONFIG_RELPATH as _FL_EV_CONFIG_RELPATH
from .hu_rl_contract import HuRlActorViewV1, PublicPlacement
from .hu_rl_reference import HuRlReferenceEnv
from .state import Board, ROWS


SCALAR_TRACE_REQUEST_SCHEMA = "regular_ofc_hu_rl_scalar_trace_request_v1"
SCALAR_TRACE_RESULT_SCHEMA = "regular_ofc_hu_rl_scalar_trace_result_v1"
SCALAR_PARITY_SUMMARY_SCHEMA = "regular_ofc_hu_rl_scalar_parity_summary_v3"
SCALAR_PARITY_RUN_CONTRACT_SCHEMA = (
    "regular_ofc_hu_rl_scalar_parity_run_contract_v3"
)
EFFECTIVE_SCORING_SCHEMA = "regular_ofc_hu_rl_effective_scoring_v1"
DECISION_COUNT = 10
MAX_HANDS = 10_000
MAX_BINARY_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_SHARD_BYTES = 32 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 300.0
MAX_PARITY_WORKERS = 16

PRIVILEGED_AUDIT_ROLE = "privileged_correctness_audit_only"
_PYTHON_RUNTIME_SOURCE_RELPATHS = (
    # Every superseded config is still hashed alongside the current one: they
    # ship in the same trees, and a silent revert to one of them would
    # otherwise be invisible to the contract.
    "configs/fl_ev_regular_2k.json",
    "configs/fl_ev_regular_v3_direct2.json",
    "configs/fl_ev_regular_v4_selfplay.json",
    "scripts/run_hu_rl_scalar_parity.py",
    "src/ofc_regular/__init__.py",
    "src/ofc_regular/action_key.py",
    "src/ofc_regular/action_space.py",
    "src/ofc_regular/cards.py",
    "src/ofc_regular/evaluator.py",
    "src/ofc_regular/hu_infoset.py",
    "src/ofc_regular/hu_rl_contract.py",
    "src/ofc_regular/hu_rl_reference.py",
    "src/ofc_regular/hu_rl_scalar_parity.py",
    "src/ofc_regular/hu_turn0_safe_selector.py",
    "src/ofc_regular/hu_turn3_gate_model.py",
    "src/ofc_regular/hu_turn3_model.py",
    "src/ofc_regular/policy.py",
    "src/ofc_regular/rules.py",
    "src/ofc_regular/state.py",
    "src/ofc_regular/teacher.py",
    "src/ofc_regular/train_hu_turn0_safe_override_selector.py",
    "src/ofc_regular/turn3_model.py",
)
_RUST_CRATE_RELPATHS = (
    "rust/hu_m3_engine",
    "rust/hu_rl_engine",
)
_RUN_CONTRACT_FIELDS = {
    "schema",
    "seed",
    "global_hands",
    "native_binary",
    "profile_registry",
    "validator_sources",
    "validator_source_manifest_sha256",
    "python_runtime",
    "effective_scoring",
    "artifact_role",
    "contains_raw_cross_actor_private_information",
    "contains_reconstructable_hidden_oracle_state",
    "policy_input_eligible",
    "replay_eligible",
    "training_eligible",
}
_BINARY_RECORD_FIELDS = {
    "content_addressed_name",
    "content_addressed_path_identity",
    "sha256",
    "size_bytes",
    "read_only",
}
_FILE_RECORD_FIELDS = {"name", "sha256", "size_bytes"}
_SOURCE_RECORD_FIELDS = {"sha256", "size_bytes"}
_PYTHON_RUNTIME_FIELDS = {
    "implementation",
    "version",
    "cache_tag",
    "byteorder",
    "platform_system",
    "platform_machine",
    "executable_name",
    "executable_binary",
    "numpy",
}
_NUMPY_RUNTIME_FIELDS = {
    "distribution_name",
    "version",
    "module_file",
    "record_sha256",
    "record_size_bytes",
}
_EFFECTIVE_SCORING_FIELDS = {
    "schema",
    "config_relpath",
    "fl_ev",
    "canonical_sha256",
}
_SCORING_ENTRY_FIELDS = {"cards", "value_hex"}
_REQUIRED_LOADED_MODULE_RELPATHS = (
    "src/ofc_regular/__init__.py",
    "src/ofc_regular/hu_rl_reference.py",
    "src/ofc_regular/hu_rl_scalar_parity.py",
    "src/ofc_regular/teacher.py",
)
_SUMMARY_FIELDS = {
    "schema",
    "status",
    "hands",
    "hand_start",
    "hand_end_exclusive",
    "global_hands",
    "seed",
    "workers",
    "total_decisions",
    "unique_request_count",
    "proofs",
    "run_contract",
    "run_contract_sha256",
    "artifact_role",
    "contains_raw_cross_actor_private_information",
    "contains_reconstructable_hidden_oracle_state",
    "full_trace_persisted",
    "policy_input_eligible",
    "replay_eligible",
    "training_eligible",
    "current_profile_changed",
    "artifact_written",
}
_PROOF_FIELDS = {
    "hand_ordinal",
    "request_sha256",
    "result_sha256",
    "decision_count",
    "exact",
}

_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
_REQUEST_FIELDS = {"schema", "explicit_deck", "selected_indices"}
_RESULT_FIELDS = {
    "schema",
    "artifact_role",
    "policy_input_eligible",
    "replay_eligible",
    "training_eligible",
    "contains_cross_actor_private_information",
    "decisions",
    "terminal",
}
_DECISION_FIELDS = {
    "ordinal",
    "actor",
    "street",
    "actor_view",
    "actor_view_digest",
    "legal_action_mapping",
    "selected_index",
    "selected_action_key",
    "step",
}
_MAPPING_FIELDS = {
    "action_count",
    "action_set_digest",
    "action_order_digest",
}
_STEP_FIELDS = {"public_event", "done", "rewards"}
_TERMINAL_FIELDS = {"boards", "rewards"}


class HuRlScalarParityError(ValueError):
    """Scalar request/result is unsafe, malformed, or not bit-exact."""


class HuRlScalarBinaryError(RuntimeError):
    """Prebuilt scalar bridge failed without exposing child-process output."""


def runtime_source_relative_paths(source_root: str | Path) -> tuple[str, ...]:
    """Return the deterministic producer/validator source closure for RLB v3."""

    root = _resolved_source_root(source_root)
    relpaths = set(_PYTHON_RUNTIME_SOURCE_RELPATHS)
    for crate_relpath in _RUST_CRATE_RELPATHS:
        crate = root / crate_relpath
        if crate.is_symlink() or not crate.is_dir():
            raise HuRlScalarParityError("scalar parity Rust source root is missing or unsafe")
        for required in ("Cargo.toml",):
            relpaths.add(f"{crate_relpath}/{required}")
        lock = crate / "Cargo.lock"
        if lock.is_file() and not lock.is_symlink():
            relpaths.add(f"{crate_relpath}/Cargo.lock")
        build = crate / "build.rs"
        if build.is_file() and not build.is_symlink():
            relpaths.add(f"{crate_relpath}/build.rs")
        rust_sources = sorted(
            path
            for path in (crate / "src").rglob("*.rs")
            if "bin" not in path.relative_to(crate / "src").parts
        )
        if crate_relpath == "rust/hu_rl_engine":
            scalar_entrypoint = crate / "src/bin/hu_rl_scalar_trace.rs"
            if scalar_entrypoint.is_file() and not scalar_entrypoint.is_symlink():
                rust_sources.append(scalar_entrypoint)
                rust_sources.sort()
        if not rust_sources:
            raise HuRlScalarParityError("scalar parity Rust source set is empty")
        for path in rust_sources:
            if path.is_symlink() or not path.is_file():
                raise HuRlScalarParityError("scalar parity Rust source is unsafe")
            relpaths.add(path.relative_to(root).as_posix())
    return tuple(sorted(relpaths))


def validate_runtime_source_identity(
    source_root: str | Path,
    actual_path: str | Path,
    expected_relpath: str,
) -> None:
    """Require one executing Python file to be the manifest-owned source file."""

    root = _resolved_source_root(source_root)
    if (
        not isinstance(expected_relpath, str)
        or not expected_relpath
        or "\\" in expected_relpath
        or Path(expected_relpath).is_absolute()
    ):
        raise HuRlScalarParityError("runtime source relative path is invalid")
    expected = _resolved_regular_file(
        root / expected_relpath, "manifest runtime source"
    )
    actual = _resolved_regular_file(actual_path, "executing runtime source")
    if actual != expected:
        raise HuRlScalarParityError(
            "executing runtime source does not match frozen source_root"
        )


def pin_scalar_parity_binary(
    binary_path: str | Path, pin_directory: str | Path
) -> Path:
    """Copy one build output to a read-only, content-addressed executable path."""

    source = Path(binary_path)
    if source.is_symlink() or not source.is_file():
        raise HuRlScalarBinaryError("scalar parity binary is missing or unsafe")
    try:
        encoded = source.read_bytes()
    except OSError as exc:
        raise HuRlScalarBinaryError("scalar parity binary could not be pinned") from exc
    if not encoded:
        raise HuRlScalarBinaryError("scalar parity binary is empty")
    digest = hashlib.sha256(encoded).hexdigest()
    suffix = source.suffix
    stem = source.name[: -len(suffix)] if suffix else source.name
    content_addressed_name = (
        source.name if stem.endswith(f".{digest}") else f"{stem}.{digest}{suffix}"
    )
    target = Path(pin_directory) / content_addressed_name
    _atomic_write_no_clobber(target, encoded, HuRlScalarBinaryError)
    try:
        mode = target.stat().st_mode
        target.chmod(
            mode
            & ~(
                stat.S_IWUSR
                | stat.S_IWGRP
                | stat.S_IWOTH
            )
        )
    except OSError as exc:
        raise HuRlScalarBinaryError("pinned scalar parity binary is not immutable") from exc
    _native_binary_record(target)
    return target


def build_scalar_parity_run_contract(
    binary_path: str | Path,
    *,
    profile_path: str | Path,
    source_root: str | Path,
    seed: int,
    global_hands: int,
) -> dict[str, Any]:
    """Hash every frozen input needed to produce or validate an RLB v3 shard."""

    if not _strict_int(seed):
        raise HuRlScalarParityError("seed must be an integer")
    if not _strict_int(global_hands) or not 1 <= global_hands <= MAX_HANDS:
        raise HuRlScalarParityError(
            f"global_hands must be an integer in [1, {MAX_HANDS}]"
        )
    root = _resolved_source_root(source_root)
    source_relpaths = runtime_source_relative_paths(root)
    _validate_loaded_ofc_regular_sources(root, source_relpaths)
    source_manifest = {
        relpath: _source_file_record(root / relpath)
        for relpath in source_relpaths
    }
    effective_scoring = _effective_scoring_identity(root)
    runtime = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag or "none",
        "byteorder": sys.byteorder,
        "platform_system": platform.system() or "unknown",
        "platform_machine": platform.machine() or "unknown",
        "executable_name": Path(sys.executable).name or "python",
        "executable_binary": _resolved_named_file_record(
            sys.executable, "Python executable"
        ),
        "numpy": _numpy_runtime_identity(),
    }
    contract: dict[str, Any] = {
        "schema": SCALAR_PARITY_RUN_CONTRACT_SCHEMA,
        "seed": seed,
        "global_hands": global_hands,
        "native_binary": _native_binary_record(binary_path),
        "profile_registry": _named_file_record(profile_path, "profile registry"),
        "validator_sources": source_manifest,
        "validator_source_manifest_sha256": _canonical_digest(source_manifest),
        "python_runtime": runtime,
        "effective_scoring": effective_scoring,
        "artifact_role": PRIVILEGED_AUDIT_ROLE,
        "contains_raw_cross_actor_private_information": False,
        "contains_reconstructable_hidden_oracle_state": True,
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
    }
    validate_scalar_parity_run_contract(contract)
    return contract


def validate_scalar_parity_run_contract(contract: Mapping[str, Any]) -> None:
    """Validate the closed v3 provenance and safety schema."""

    _require_mapping(contract, "scalar parity run contract")
    _require_exact_fields(contract, _RUN_CONTRACT_FIELDS, "scalar parity run contract")
    if contract["schema"] != SCALAR_PARITY_RUN_CONTRACT_SCHEMA:
        raise HuRlScalarParityError("unsupported scalar parity run contract schema")
    if not _strict_int(contract["seed"]):
        raise HuRlScalarParityError("run contract seed must be an integer")
    if not _strict_int(contract["global_hands"]) or not 1 <= contract[
        "global_hands"
    ] <= MAX_HANDS:
        raise HuRlScalarParityError("run contract global_hands is invalid")
    binary = _mapping_value(contract["native_binary"], "run contract native binary")
    _require_exact_fields(binary, _BINARY_RECORD_FIELDS, "run contract native binary")
    _require_sha256(binary["sha256"], "run contract native binary sha256")
    if (
        not isinstance(binary["content_addressed_name"], str)
        or not binary["content_addressed_name"]
        or binary["content_addressed_path_identity"]
        != f"sha256/{binary['sha256']}/{binary['content_addressed_name']}"
        or not _strict_int(binary["size_bytes"])
        or binary["size_bytes"] <= 0
        or binary["read_only"] is not True
    ):
        raise HuRlScalarParityError("run contract native binary identity is invalid")
    profile = _mapping_value(
        contract["profile_registry"], "run contract profile registry"
    )
    _validate_named_file_record(profile, "run contract profile registry")
    sources = _mapping_value(
        contract["validator_sources"], "run contract validator sources"
    )
    if not sources:
        raise HuRlScalarParityError("run contract validator source set is empty")
    for relpath, record_value in sources.items():
        if not isinstance(relpath, str) or not relpath or "\\" in relpath:
            raise HuRlScalarParityError("run contract source identity is invalid")
        record = _mapping_value(record_value, "run contract source record")
        _require_exact_fields(record, _SOURCE_RECORD_FIELDS, "run contract source record")
        _require_sha256(record["sha256"], "run contract source sha256")
        if not _strict_int(record["size_bytes"]) or record["size_bytes"] <= 0:
            raise HuRlScalarParityError("run contract source size is invalid")
    _require_sha256(
        contract["validator_source_manifest_sha256"],
        "run contract validator source manifest sha256",
    )
    if contract["validator_source_manifest_sha256"] != _canonical_digest(sources):
        raise HuRlScalarParityError("run contract validator source manifest changed")
    runtime = _mapping_value(contract["python_runtime"], "run contract Python runtime")
    _require_exact_fields(runtime, _PYTHON_RUNTIME_FIELDS, "run contract Python runtime")
    for field in (
        "implementation",
        "version",
        "cache_tag",
        "byteorder",
        "platform_system",
        "platform_machine",
        "executable_name",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise HuRlScalarParityError(
                "run contract Python runtime identity is invalid"
            )
    executable = _mapping_value(
        runtime["executable_binary"], "run contract Python executable"
    )
    _validate_named_file_record(executable, "run contract Python executable")
    numpy_identity = _mapping_value(runtime["numpy"], "run contract NumPy runtime")
    _require_exact_fields(
        numpy_identity, _NUMPY_RUNTIME_FIELDS, "run contract NumPy runtime"
    )
    for field in ("distribution_name", "version"):
        if not isinstance(numpy_identity[field], str) or not numpy_identity[field]:
            raise HuRlScalarParityError("run contract NumPy identity is invalid")
    numpy_module = _mapping_value(
        numpy_identity["module_file"], "run contract NumPy module file"
    )
    _validate_named_file_record(numpy_module, "run contract NumPy module file")
    _require_sha256(numpy_identity["record_sha256"], "NumPy RECORD sha256")
    if (
        not _strict_int(numpy_identity["record_size_bytes"])
        or numpy_identity["record_size_bytes"] <= 0
    ):
        raise HuRlScalarParityError("run contract NumPy RECORD size is invalid")
    effective_scoring = _mapping_value(
        contract["effective_scoring"], "run contract effective scoring"
    )
    _validate_effective_scoring_identity(effective_scoring)
    expected_safety = {
        "artifact_role": PRIVILEGED_AUDIT_ROLE,
        "contains_raw_cross_actor_private_information": False,
        "contains_reconstructable_hidden_oracle_state": True,
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
    }
    if any(
        contract[field] != value or type(contract[field]) is not type(value)
        for field, value in expected_safety.items()
    ):
        raise HuRlScalarParityError("run contract privileged safety class changed")


def scalar_parity_run_contract_digest(contract: Mapping[str, Any]) -> str:
    validate_scalar_parity_run_contract(contract)
    return _canonical_digest(contract)


@dataclass(frozen=True, repr=False)
class ScalarTraceRequestV1:
    """The only hidden-oracle input accepted by the scalar bridge."""

    explicit_deck: tuple[str, ...]
    selected_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        deck = tuple(self.explicit_deck)
        indices = tuple(self.selected_indices)
        if len(deck) != len(ALL_CARDS):
            raise HuRlScalarParityError("explicit_deck must contain exactly 52 cards")
        try:
            validate_cards(deck)
        except ValueError as exc:
            raise HuRlScalarParityError(
                "explicit_deck is not a valid complete regular deck"
            ) from exc
        if set(deck) != set(ALL_CARDS):
            raise HuRlScalarParityError("explicit_deck is not a complete regular deck")
        if len(indices) != DECISION_COUNT:
            raise HuRlScalarParityError("selected_indices must contain exactly 10 values")
        if any(not _strict_int(index) or index < 0 for index in indices):
            raise HuRlScalarParityError(
                "selected_indices must contain non-negative integers"
            )
        object.__setattr__(self, "explicit_deck", deck)
        object.__setattr__(self, "selected_indices", indices)
        _validate_request_indices(self)

    def __repr__(self) -> str:
        return (
            "ScalarTraceRequestV1("
            f"selected_indices={self.selected_indices!r}, explicit_deck=<redacted>)"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScalarTraceRequestV1":
        _require_mapping(payload, "scalar trace request")
        _require_exact_fields(payload, _REQUEST_FIELDS, "scalar trace request")
        if payload["schema"] != SCALAR_TRACE_REQUEST_SCHEMA:
            raise HuRlScalarParityError("unsupported scalar trace request schema")
        deck = _string_sequence(payload["explicit_deck"], "explicit_deck")
        raw_indices = payload["selected_indices"]
        if isinstance(raw_indices, (str, bytes)) or not isinstance(
            raw_indices, Sequence
        ):
            raise HuRlScalarParityError("selected_indices must be a sequence")
        return cls(deck, tuple(raw_indices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCALAR_TRACE_REQUEST_SCHEMA,
            "explicit_deck": list(self.explicit_deck),
            "selected_indices": list(self.selected_indices),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def digest(self) -> str:
        return _canonical_digest(self.to_dict())


def generate_seeded_scalar_trace_requests(
    *, hands: int, seed: int
) -> tuple[ScalarTraceRequestV1, ...]:
    """Generate unique deterministic deck/index contracts without global RNG."""

    decks = _generate_unique_seeded_decks(hands=hands, seed=seed)
    requests: list[ScalarTraceRequestV1] = []
    for hand_ordinal, deck in enumerate(decks):
        indices = _seeded_valid_indices(deck, seed, hand_ordinal)
        # ``_seeded_valid_indices`` has already replayed all ten decisions
        # through the fail-closed Python oracle.  Avoid replaying the identical
        # hand in ``__post_init__`` while keeping the ordinary public
        # constructor strict for external inputs.
        requests.append(_validated_seeded_request(deck, indices))
    return tuple(requests)


def build_python_scalar_trace(
    request: ScalarTraceRequestV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact privileged correctness-audit result expected from Rust."""

    contract = _coerce_request(request)
    env = HuRlReferenceEnv(contract.explicit_deck)
    decisions: list[dict[str, Any]] = []
    for ordinal, selected_index in enumerate(contract.selected_indices):
        view = env.observe()
        mapping = view.legal_action_mapping
        selected = mapping.key_at(selected_index)
        step = env.step(selected)
        decisions.append(
            {
                "ordinal": ordinal,
                "actor": step.actor,
                "street": step.street,
                "actor_view": view.to_dict(),
                "actor_view_digest": view.digest(),
                "legal_action_mapping": {
                    "action_count": mapping.action_count,
                    "action_set_digest": mapping.action_set_digest,
                    "action_order_digest": mapping.action_order_digest,
                },
                "selected_index": selected_index,
                "selected_action_key": selected.to_token(),
                "step": {
                    "public_event": step.public_placement.to_dict(),
                    "done": step.done,
                    "rewards": list(step.rewards),
                },
            }
        )
    result = {
        "schema": SCALAR_TRACE_RESULT_SCHEMA,
        "artifact_role": "privileged_correctness_audit_only",
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
        "contains_cross_actor_private_information": True,
        "decisions": decisions,
        "terminal": {
            "boards": [_board_to_dict(board) for board in env.boards],
            "rewards": list(env.terminal_rewards()),
        },
    }
    validate_scalar_trace_result_shape(result)
    return result


def validate_scalar_trace_result_shape(result: Mapping[str, Any]) -> None:
    """Validate the closed Rust output schema without using oracle state."""

    _require_mapping(result, "scalar trace result")
    _require_exact_fields(result, _RESULT_FIELDS, "scalar trace result")
    if result["schema"] != SCALAR_TRACE_RESULT_SCHEMA:
        raise HuRlScalarParityError("unsupported scalar trace result schema")
    expected_classification = {
        "artifact_role": "privileged_correctness_audit_only",
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
        "contains_cross_actor_private_information": True,
    }
    if (
        result["artifact_role"] != expected_classification["artifact_role"]
        or any(
            type(result[field]) is not bool or result[field] is not value
            for field, value in expected_classification.items()
            if field != "artifact_role"
        )
    ):
        raise HuRlScalarParityError(
            "scalar trace privileged audit classification mismatch"
        )
    decisions = result["decisions"]
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise HuRlScalarParityError("result decisions must be a sequence")
    if len(decisions) != DECISION_COUNT:
        raise HuRlScalarParityError("result must contain exactly 10 decisions")
    for expected_ordinal, raw_decision in enumerate(decisions):
        decision = _mapping_value(raw_decision, f"decision[{expected_ordinal}]")
        _require_exact_fields(
            decision, _DECISION_FIELDS, f"decision[{expected_ordinal}]"
        )
        if decision["ordinal"] != expected_ordinal or not _strict_int(
            decision["ordinal"]
        ):
            raise HuRlScalarParityError("result decision ordinal mismatch")
        expected_actor = expected_ordinal % 2
        expected_street = f"T{expected_ordinal // 2}"
        if decision["actor"] != expected_actor or not _strict_int(decision["actor"]):
            raise HuRlScalarParityError("result decision actor mismatch")
        if decision["street"] != expected_street:
            raise HuRlScalarParityError("result decision street mismatch")

        raw_view = _mapping_value(
            decision["actor_view"], f"decision[{expected_ordinal}].actor_view"
        )
        try:
            view = HuRlActorViewV1.from_dict(raw_view)
        except (TypeError, ValueError) as exc:
            raise HuRlScalarParityError(
                f"invalid result actor view at decision {expected_ordinal}"
            ) from exc
        if raw_view != view.to_dict():
            raise HuRlScalarParityError(
                f"result actor view is not canonical at decision {expected_ordinal}"
            )
        _require_sha256(
            decision["actor_view_digest"],
            f"decision[{expected_ordinal}].actor_view_digest",
        )
        if decision["actor_view_digest"] != view.digest():
            raise HuRlScalarParityError(
                f"result actor view digest mismatch at decision {expected_ordinal}"
            )

        mapping = _mapping_value(
            decision["legal_action_mapping"],
            f"decision[{expected_ordinal}].legal_action_mapping",
        )
        _require_exact_fields(
            mapping,
            _MAPPING_FIELDS,
            f"decision[{expected_ordinal}].legal_action_mapping",
        )
        if not _strict_int(mapping["action_count"]) or mapping[
            "action_count"
        ] != view.legal_action_mapping.action_count:
            raise HuRlScalarParityError(
                f"result action count mismatch at decision {expected_ordinal}"
            )
        for field in ("action_set_digest", "action_order_digest"):
            _require_sha256(mapping[field], f"decision[{expected_ordinal}].{field}")
            if mapping[field] != getattr(view.legal_action_mapping, field):
                raise HuRlScalarParityError(
                    f"result {field.replace('_', ' ')} mismatch at decision {expected_ordinal}"
                )

        selected_index = decision["selected_index"]
        if (
            not _strict_int(selected_index)
            or not 0 <= selected_index < view.legal_action_mapping.action_count
        ):
            raise HuRlScalarParityError(
                f"result selected index is invalid at decision {expected_ordinal}"
            )
        selected_token = decision["selected_action_key"]
        if not isinstance(selected_token, str):
            raise HuRlScalarParityError("result selected_action_key must be a string")
        try:
            selected = ActionKey.from_token(selected_token)
        except ValueError as exc:
            raise HuRlScalarParityError(
                f"invalid selected ActionKey at decision {expected_ordinal}"
            ) from exc
        if selected != view.legal_action_mapping.key_at(selected_index):
            raise HuRlScalarParityError(
                f"result selected ActionKey mismatch at decision {expected_ordinal}"
            )

        step = _mapping_value(
            decision["step"], f"decision[{expected_ordinal}].step"
        )
        _require_exact_fields(step, _STEP_FIELDS, f"decision[{expected_ordinal}].step")
        public_event_payload = _mapping_value(
            step["public_event"],
            f"decision[{expected_ordinal}].step.public_event",
        )
        try:
            public_event = PublicPlacement.from_dict(public_event_payload)
        except (TypeError, ValueError) as exc:
            raise HuRlScalarParityError(
                f"invalid public event at decision {expected_ordinal}"
            ) from exc
        if public_event_payload != public_event.to_dict():
            raise HuRlScalarParityError(
                f"result public event is not canonical at decision {expected_ordinal}"
            )
        expected_seat = "first" if expected_actor == 0 else "second"
        if (public_event.street, public_event.acting_seat) != (
            expected_street,
            expected_seat,
        ):
            raise HuRlScalarParityError(
                f"result public event identity mismatch at decision {expected_ordinal}"
            )
        if public_event.placement_masks != selected.masks[:3] or (
            public_event.discard_count != selected.discard_mask.bit_count()
        ):
            raise HuRlScalarParityError(
                f"result public event/action mismatch at decision {expected_ordinal}"
            )
        if type(step["done"]) is not bool or step["done"] != (
            expected_ordinal == DECISION_COUNT - 1
        ):
            raise HuRlScalarParityError(
                f"result done mismatch at decision {expected_ordinal}"
            )
        rewards = _reward_pair(
            step["rewards"], f"decision[{expected_ordinal}].step.rewards"
        )
        if not step["done"] and rewards != (0.0, 0.0):
            raise HuRlScalarParityError(
                f"nonterminal result rewards are nonzero at decision {expected_ordinal}"
            )

    terminal = _mapping_value(result["terminal"], "result terminal")
    _require_exact_fields(terminal, _TERMINAL_FIELDS, "result terminal")
    raw_boards = terminal["boards"]
    if isinstance(raw_boards, (str, bytes)) or not isinstance(
        raw_boards, Sequence
    ) or len(raw_boards) != 2:
        raise HuRlScalarParityError("result terminal must contain exactly two boards")
    boards = tuple(_board_from_dict(board) for board in raw_boards)
    for raw, board in zip(raw_boards, boards, strict=True):
        if raw != _board_to_dict(board):
            raise HuRlScalarParityError("result terminal boards are not canonical")
    try:
        validate_cards((*boards[0].all_cards(), *boards[1].all_cards()))
    except ValueError as exc:
        raise HuRlScalarParityError("result terminal boards contain duplicate cards") from exc
    if not all(board.is_complete() for board in boards):
        raise HuRlScalarParityError("result terminal boards are incomplete")
    terminal_rewards = _reward_pair(terminal["rewards"], "result terminal.rewards")
    final_rewards = _reward_pair(decisions[-1]["step"]["rewards"], "final rewards")
    if terminal_rewards != final_rewards:
        raise HuRlScalarParityError("result terminal rewards disagree with final step")


def compare_rust_scalar_trace(
    request: ScalarTraceRequestV1 | Mapping[str, Any],
    rust_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact Python/Rust equality and return a value-redacted proof row."""

    contract = _coerce_request(request)
    validate_scalar_trace_result_shape(rust_result)
    expected = build_python_scalar_trace(contract)
    if _canonical_json(rust_result) != _canonical_json(expected):
        path = _first_mismatch_path(expected, rust_result) or "result.canonical_json"
        raise HuRlScalarParityError(f"Python/Rust scalar trace mismatch at {path}")
    return {
        "request_sha256": contract.digest(),
        "result_sha256": _canonical_digest(rust_result),
        "decision_count": DECISION_COUNT,
        "exact": True,
    }


def invoke_prebuilt_scalar_binary(
    binary_path: str | Path,
    request: ScalarTraceRequestV1 | Mapping[str, Any],
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Invoke one prebuilt binary once, using stdin/stdout JSON and no shell."""

    contract = _coerce_request(request)
    binary = Path(binary_path)
    if binary.is_symlink() or not binary.is_file():
        raise HuRlScalarBinaryError("prebuilt scalar parity binary is missing or unsafe")
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise HuRlScalarBinaryError("scalar parity timeout is invalid") from exc
    if not math.isfinite(timeout) or not 0.1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise HuRlScalarBinaryError(
            f"scalar parity timeout must be between 0.1 and {MAX_TIMEOUT_SECONDS:g} seconds"
        )
    try:
        completed = subprocess.run(
            [str(binary.resolve())],
            input=contract.canonical_json() + "\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HuRlScalarBinaryError("prebuilt scalar parity binary timed out") from exc
    except (OSError, UnicodeError) as exc:
        raise HuRlScalarBinaryError("prebuilt scalar parity binary could not run safely") from exc
    if completed.returncode != 0:
        raise HuRlScalarBinaryError(
            f"prebuilt scalar parity binary failed with exit code {completed.returncode}"
        )
    if completed.stderr:
        raise HuRlScalarBinaryError("prebuilt scalar parity binary wrote unexpected stderr")
    encoded = completed.stdout.encode("utf-8")
    if len(encoded) > MAX_BINARY_OUTPUT_BYTES:
        raise HuRlScalarBinaryError("prebuilt scalar parity binary output exceeded limit")
    try:
        result = json.loads(
            completed.stdout,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, HuRlScalarParityError) as exc:
        raise HuRlScalarBinaryError(
            "prebuilt scalar parity binary returned invalid JSON"
        ) from exc
    if not isinstance(result, Mapping):
        raise HuRlScalarBinaryError(
            "prebuilt scalar parity binary result is not a JSON object"
        )
    compare_rust_scalar_trace(contract, result)
    return dict(result)


def run_scalar_parity_suite(
    binary_path: str | Path,
    *,
    hands: int,
    seed: int,
    global_hands: int,
    profile_path: str | Path,
    source_root: str | Path,
    timeout_seconds: float = 30.0,
    workers: int = 1,
    hand_start: int = 0,
    expected_run_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Run exact deterministic parity and return no decks or full traces.

    Hand lanes are independent, so an optional process pool only changes wall
    time.  Proof rows are restored to canonical hand order before aggregation.
    """

    if not _strict_int(workers) or not 1 <= workers <= MAX_PARITY_WORKERS:
        raise HuRlScalarParityError(
            f"workers must be an integer in [1, {MAX_PARITY_WORKERS}]"
        )
    if not _strict_int(hands) or not 1 <= hands <= MAX_HANDS:
        raise HuRlScalarParityError(f"hands must be an integer in [1, {MAX_HANDS}]")
    if not _strict_int(hand_start) or hand_start < 0:
        raise HuRlScalarParityError("hand_start must be a non-negative integer")
    if not _strict_int(global_hands) or not 1 <= global_hands <= MAX_HANDS:
        raise HuRlScalarParityError(
            f"global_hands must be an integer in [1, {MAX_HANDS}]"
        )
    hand_end = hand_start + hands
    if hand_end > global_hands:
        raise HuRlScalarParityError(
            "hand_start + hands must not exceed global_hands"
        )
    resolved_root = _resolved_source_root(source_root)
    resolved_binary = _resolved_regular_file(
        binary_path, "pinned scalar parity binary"
    )
    resolved_profile = _resolved_regular_file(profile_path, "profile registry")
    run_contract = build_scalar_parity_run_contract(
        resolved_binary,
        profile_path=resolved_profile,
        source_root=resolved_root,
        seed=seed,
        global_hands=global_hands,
    )
    run_contract_sha256 = scalar_parity_run_contract_digest(run_contract)
    if expected_run_contract_sha256 is not None:
        _require_sha256(
            expected_run_contract_sha256, "expected_run_contract_sha256"
        )
        if expected_run_contract_sha256 != run_contract_sha256:
            raise HuRlScalarParityError("frozen scalar parity run contract changed")

    # Generate the complete prefix before slicing so collision resolution is
    # identical whether a range runs alone or as part of the full 10k suite.
    decks = _generate_unique_seeded_decks(hands=hand_end, seed=seed)[hand_start:]
    jobs = tuple(
        (
            hand_ordinal,
            str(resolved_binary),
            deck,
            seed,
            float(timeout_seconds),
        )
        for hand_ordinal, deck in enumerate(decks, start=hand_start)
    )
    if workers == 1:
        proofs = [_run_seeded_scalar_parity_job(job) for job in jobs]
    else:
        # ``spawn`` is the Windows default.  Every argument and result is a
        # value-only contract, and child processes never write artifacts.
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_scalar_parity_worker,
            initargs=(
                str(resolved_binary),
                str(resolved_profile),
                str(resolved_root),
                seed,
                global_hands,
                run_contract_sha256,
            ),
        ) as executor:
            proofs = list(
                executor.map(_run_seeded_scalar_parity_job, jobs, chunksize=1)
            )
    proofs.sort(key=lambda proof: proof["hand_ordinal"])
    if [proof["hand_ordinal"] for proof in proofs] != list(
        range(hand_start, hand_end)
    ):
        raise HuRlScalarParityError("parallel parity lane order changed")
    end_contract = build_scalar_parity_run_contract(
        resolved_binary,
        profile_path=resolved_profile,
        source_root=resolved_root,
        seed=seed,
        global_hands=global_hands,
    )
    if _canonical_json(end_contract) != _canonical_json(run_contract):
        raise HuRlScalarParityError("pinned scalar parity inputs changed during shard")
    summary = {
        "schema": SCALAR_PARITY_SUMMARY_SCHEMA,
        "status": "exact_python_rust_scalar_parity",
        "hands": hands,
        "hand_start": hand_start,
        "hand_end_exclusive": hand_end,
        "global_hands": global_hands,
        "seed": seed,
        "workers": workers,
        "total_decisions": hands * DECISION_COUNT,
        "unique_request_count": len({proof["request_sha256"] for proof in proofs}),
        "proofs": proofs,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "artifact_role": PRIVILEGED_AUDIT_ROLE,
        "contains_raw_cross_actor_private_information": False,
        "contains_reconstructable_hidden_oracle_state": True,
        "full_trace_persisted": False,
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "artifact_written": False,
    }
    validate_scalar_parity_summary(summary, expected_artifact_written=False)
    return summary


def validate_scalar_parity_summary(
    summary: Mapping[str, Any],
    *,
    expected_artifact_written: bool,
    expected_hand_start: int | None = None,
    expected_hands: int | None = None,
    expected_run_contract_sha256: str | None = None,
) -> None:
    """Validate one closed, value-redacted v3 shard summary."""

    _require_mapping(summary, "scalar parity summary")
    _require_exact_fields(summary, _SUMMARY_FIELDS, "scalar parity summary")
    if (
        summary["schema"] != SCALAR_PARITY_SUMMARY_SCHEMA
        or summary["status"] != "exact_python_rust_scalar_parity"
    ):
        raise HuRlScalarParityError("unsupported scalar parity summary identity")
    for field in (
        "hands",
        "hand_start",
        "hand_end_exclusive",
        "global_hands",
        "seed",
        "workers",
        "total_decisions",
        "unique_request_count",
    ):
        if not _strict_int(summary[field]):
            raise HuRlScalarParityError("scalar parity summary integer field changed")
    hands = summary["hands"]
    start = summary["hand_start"]
    end = summary["hand_end_exclusive"]
    global_hands = summary["global_hands"]
    if (
        not 1 <= hands <= MAX_HANDS
        or start < 0
        or end != start + hands
        or not 1 <= global_hands <= MAX_HANDS
        or end > global_hands
        or not 1 <= summary["workers"] <= MAX_PARITY_WORKERS
        or summary["total_decisions"] != hands * DECISION_COUNT
        or summary["unique_request_count"] != hands
    ):
        raise HuRlScalarParityError("scalar parity summary range/count changed")
    contract = _mapping_value(summary["run_contract"], "scalar parity run contract")
    validate_scalar_parity_run_contract(contract)
    _require_sha256(summary["run_contract_sha256"], "run_contract_sha256")
    if (
        summary["run_contract_sha256"] != scalar_parity_run_contract_digest(contract)
        or summary["seed"] != contract["seed"]
        or global_hands != contract["global_hands"]
    ):
        raise HuRlScalarParityError("scalar parity summary run contract changed")
    expected_safety = {
        "artifact_role": PRIVILEGED_AUDIT_ROLE,
        "contains_raw_cross_actor_private_information": False,
        "contains_reconstructable_hidden_oracle_state": True,
        "full_trace_persisted": False,
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "artifact_written": expected_artifact_written,
    }
    if any(
        summary[field] != value or type(summary[field]) is not type(value)
        for field, value in expected_safety.items()
    ):
        raise HuRlScalarParityError("scalar parity summary safety class changed")
    proofs = summary["proofs"]
    if not isinstance(proofs, list) or len(proofs) != hands:
        raise HuRlScalarParityError("scalar parity summary proof count changed")
    request_digests: set[str] = set()
    for offset, proof_value in enumerate(proofs):
        proof = _mapping_value(proof_value, "scalar parity proof")
        _require_exact_fields(proof, _PROOF_FIELDS, "scalar parity proof")
        if (
            not _strict_int(proof["hand_ordinal"])
            or proof["hand_ordinal"] != start + offset
            or not _strict_int(proof["decision_count"])
            or proof["decision_count"] != DECISION_COUNT
            or proof["exact"] is not True
        ):
            raise HuRlScalarParityError("scalar parity proof identity changed")
        _require_sha256(proof["request_sha256"], "request_sha256")
        _require_sha256(proof["result_sha256"], "result_sha256")
        if proof["request_sha256"] in request_digests:
            raise HuRlScalarParityError("scalar parity request digest overlapped")
        request_digests.add(proof["request_sha256"])
    if expected_hand_start is not None:
        if not _strict_int(expected_hand_start) or start != expected_hand_start:
            raise HuRlScalarParityError("scalar parity resume range changed")
    if expected_hands is not None:
        if not _strict_int(expected_hands) or hands != expected_hands:
            raise HuRlScalarParityError("scalar parity resume range changed")
    if expected_run_contract_sha256 is not None:
        _require_sha256(
            expected_run_contract_sha256, "expected_run_contract_sha256"
        )
        if summary["run_contract_sha256"] != expected_run_contract_sha256:
            raise HuRlScalarParityError("scalar parity resume contract changed")


def load_scalar_parity_summary(
    path: str | Path,
    *,
    expected_hand_start: int | None = None,
    expected_hands: int | None = None,
    expected_run_contract_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Load one canonical persisted shard for merge or validated resume."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HuRlScalarParityError("scalar parity shard is missing or unsafe")
    try:
        encoded = candidate.read_bytes()
    except OSError as exc:
        raise HuRlScalarParityError("scalar parity shard could not be read") from exc
    if not encoded or len(encoded) > MAX_SHARD_BYTES:
        raise HuRlScalarParityError("scalar parity shard size is invalid")
    try:
        payload = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlScalarParityError("scalar parity shard JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise HuRlScalarParityError("scalar parity shard must contain an object")
    if encoded != _canonical_bytes(payload) + b"\n":
        raise HuRlScalarParityError("scalar parity shard is not canonical JSON")
    validate_scalar_parity_summary(
        payload,
        expected_artifact_written=True,
        expected_hand_start=expected_hand_start,
        expected_hands=expected_hands,
        expected_run_contract_sha256=expected_run_contract_sha256,
    )
    return payload, encoded


def write_scalar_parity_summary(
    path: str | Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically persist or byte-validate one immutable shard for resume."""

    if summary.get("artifact_written") is True:
        persisted = dict(summary)
    else:
        validate_scalar_parity_summary(summary, expected_artifact_written=False)
        persisted = {**summary, "artifact_written": True}
    validate_scalar_parity_summary(persisted, expected_artifact_written=True)
    encoded = _canonical_bytes(persisted) + b"\n"
    _atomic_write_no_clobber(Path(path), encoded, HuRlScalarParityError)
    loaded, existing = load_scalar_parity_summary(
        path,
        expected_hand_start=persisted["hand_start"],
        expected_hands=persisted["hands"],
        expected_run_contract_sha256=persisted["run_contract_sha256"],
    )
    if existing != encoded or loaded != persisted:
        raise HuRlScalarParityError("persisted scalar parity shard changed")
    return persisted


def _initialize_scalar_parity_worker(
    binary_path: str,
    profile_path: str,
    source_root: str,
    seed: int,
    global_hands: int,
    expected_run_contract_sha256: str,
) -> None:
    """Prove every spawned worker executes the same frozen runtime closure."""

    contract = build_scalar_parity_run_contract(
        binary_path,
        profile_path=profile_path,
        source_root=source_root,
        seed=seed,
        global_hands=global_hands,
    )
    if scalar_parity_run_contract_digest(contract) != expected_run_contract_sha256:
        raise HuRlScalarParityError(
            "spawned parity worker runtime contract changed"
        )


def _run_seeded_scalar_parity_job(
    job: tuple[int, str, tuple[str, ...], int, float],
) -> dict[str, Any]:
    """Pickle-safe worker for one exact full-hand parity lane."""

    hand_ordinal, binary_path, deck, seed, timeout_seconds = job
    try:
        indices = _seeded_valid_indices(deck, seed, hand_ordinal)
        request = _validated_seeded_request(deck, indices)
        result = invoke_prebuilt_scalar_binary(
            binary_path,
            request,
            timeout_seconds=timeout_seconds,
        )
    except (HuRlScalarBinaryError, HuRlScalarParityError) as exc:
        # Ordinal is non-secret provenance and is enough to reproduce a
        # failing deterministic lane.  Never append deck/action values.
        raise type(exc)(f"hand ordinal {hand_ordinal}: {exc}") from exc
    # ``invoke_prebuilt_scalar_binary`` already performed the full semantic
    # and canonical Python/Rust comparison.  Recomputing it here would replay
    # the same Python hand a second time without adding evidence.
    return {
        "hand_ordinal": hand_ordinal,
        "request_sha256": request.digest(),
        "result_sha256": _canonical_digest(result),
        "decision_count": DECISION_COUNT,
        "exact": True,
    }


def _validate_request_indices(request: ScalarTraceRequestV1) -> None:
    env = HuRlReferenceEnv(request.explicit_deck)
    for ordinal, selected_index in enumerate(request.selected_indices):
        mapping = env.legal_mapping()
        if selected_index >= mapping.action_count:
            raise HuRlScalarParityError(
                f"selected_indices[{ordinal}] is outside its legal mapping"
            )
        env.step(mapping.key_at(selected_index))


def _seeded_deck(seed: int, hand_ordinal: int, collision_nonce: int) -> tuple[str, ...]:
    namespace = (
        f"regular-ofc-hu-rl-scalar-parity-v1|{seed}|{hand_ordinal}|"
        f"{collision_nonce}|deck|"
    ).encode("ascii")
    return tuple(
        sorted(
            ALL_CARDS,
            key=lambda card: (
                hashlib.sha256(namespace + card.encode("ascii")).digest(),
                _CARD_INDEX[card],
            ),
        )
    )


def _generate_unique_seeded_decks(
    *, hands: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    if not _strict_int(hands) or not 1 <= hands <= MAX_HANDS:
        raise HuRlScalarParityError(f"hands must be an integer in [1, {MAX_HANDS}]")
    if not _strict_int(seed):
        raise HuRlScalarParityError("seed must be an integer")

    decks: list[tuple[str, ...]] = []
    seen_decks: set[tuple[str, ...]] = set()
    for hand_ordinal in range(hands):
        collision_nonce = 0
        while True:
            deck = _seeded_deck(seed, hand_ordinal, collision_nonce)
            if deck not in seen_decks:
                break
            collision_nonce += 1
            if collision_nonce > MAX_HANDS:
                raise HuRlScalarParityError(
                    "deterministic deck generation exhausted collision namespace"
                )
        seen_decks.add(deck)
        decks.append(deck)
    return tuple(decks)


def _seeded_valid_indices(
    deck: tuple[str, ...], seed: int, hand_ordinal: int
) -> tuple[int, ...]:
    env = HuRlReferenceEnv(deck)
    indices: list[int] = []
    for decision in range(DECISION_COUNT):
        mapping = env.legal_mapping()
        material = (
            f"regular-ofc-hu-rl-scalar-parity-v1|{seed}|{hand_ordinal}|"
            f"{decision}|selected-index"
        ).encode("ascii")
        index = int.from_bytes(hashlib.sha256(material).digest(), "big") % mapping.action_count
        indices.append(index)
        env.step(mapping.key_at(index))
    return tuple(indices)


def _validated_seeded_request(
    deck: tuple[str, ...], indices: tuple[int, ...]
) -> ScalarTraceRequestV1:
    """Construct only after `_seeded_valid_indices` proved all ten actions."""

    request = object.__new__(ScalarTraceRequestV1)
    object.__setattr__(request, "explicit_deck", deck)
    object.__setattr__(request, "selected_indices", indices)
    return request


def _coerce_request(
    request: ScalarTraceRequestV1 | Mapping[str, Any],
) -> ScalarTraceRequestV1:
    if type(request) is ScalarTraceRequestV1:
        return request
    if isinstance(request, Mapping):
        return ScalarTraceRequestV1.from_dict(request)
    raise TypeError("scalar trace request must be ScalarTraceRequestV1 or a mapping")


def _board_to_dict(board: Board) -> dict[str, list[str]]:
    return {
        row: sorted(getattr(board, row), key=_CARD_INDEX.__getitem__) for row in ROWS
    }


def _board_from_dict(payload: Any) -> Board:
    mapping = _mapping_value(payload, "terminal board")
    _require_exact_fields(mapping, set(ROWS), "terminal board")
    try:
        return Board.from_rows(
            top=_string_sequence(mapping["top"], "terminal board.top"),
            middle=_string_sequence(mapping["middle"], "terminal board.middle"),
            bottom=_string_sequence(mapping["bottom"], "terminal board.bottom"),
        )
    except ValueError as exc:
        raise HuRlScalarParityError("invalid result terminal board") from exc


def _reward_pair(value: Any, context: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise HuRlScalarParityError(f"{context} must contain exactly two numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise HuRlScalarParityError(f"{context} must contain only numbers")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair):
        raise HuRlScalarParityError(f"{context} must be finite")
    if pair[0] + pair[1] != 0.0:
        raise HuRlScalarParityError(f"{context} must be exactly zero-sum")
    return pair


def _first_mismatch_path(expected: Any, actual: Any, path: str = "result") -> str:
    if type(expected) is not type(actual):
        return path
    if isinstance(expected, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in expected or key not in actual:
                return f"{path}.{key}"
            nested = _first_mismatch_path(expected[key], actual[key], f"{path}.{key}")
            if nested:
                return nested
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            nested = _first_mismatch_path(left, right, f"{path}[{index}]")
            if nested:
                return nested
        return ""
    return "" if expected == actual else path


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("ascii")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _resolved_source_root(source_root: str | Path) -> Path:
    candidate = Path(source_root)
    if candidate.is_symlink() or not candidate.is_dir():
        raise HuRlScalarParityError("scalar parity source root is missing or unsafe")
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise HuRlScalarParityError(
            "scalar parity source root could not be resolved"
        ) from exc


def _resolved_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HuRlScalarParityError(f"{label} is missing or unsafe")
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise HuRlScalarParityError(f"{label} could not be resolved") from exc


def _module_name_for_relpath(relpath: str) -> str:
    if relpath == "src/ofc_regular/__init__.py":
        return "ofc_regular"
    return f"ofc_regular.{Path(relpath).stem}"


def _validate_loaded_ofc_regular_sources(
    root: Path, source_relpaths: Sequence[str]
) -> None:
    manifest_modules = {
        _module_name_for_relpath(relpath): relpath
        for relpath in source_relpaths
        if relpath.startswith("src/ofc_regular/") and relpath.endswith(".py")
    }
    for required_relpath in _REQUIRED_LOADED_MODULE_RELPATHS:
        if _module_name_for_relpath(required_relpath) not in sys.modules:
            raise HuRlScalarParityError(
                "required scalar parity runtime module is not loaded"
            )
    for module_name, relpath in manifest_modules.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise HuRlScalarParityError(
                "loaded scalar parity runtime module has no source identity"
            )
        validate_runtime_source_identity(root, module_file, relpath)


def _canonical_scoring_entries(value: Any, context: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        raw_entries = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_entries = list(value)
    else:
        raise HuRlScalarParityError(f"{context} must be a mapping or pair sequence")
    parsed: dict[int, float] = {}
    for raw_entry in raw_entries:
        if (
            not isinstance(raw_entry, Sequence)
            or isinstance(raw_entry, (str, bytes))
            or len(raw_entry) != 2
        ):
            raise HuRlScalarParityError(f"{context} contains an invalid entry")
        raw_cards, raw_value = raw_entry
        if isinstance(raw_cards, str):
            if not raw_cards or not raw_cards.isascii() or not raw_cards.isdigit():
                raise HuRlScalarParityError(f"{context} card key is invalid")
            cards = int(raw_cards)
            if str(cards) != raw_cards:
                raise HuRlScalarParityError(f"{context} card key is not canonical")
        elif _strict_int(raw_cards):
            cards = raw_cards
        else:
            raise HuRlScalarParityError(f"{context} card key is invalid")
        if cards <= 0 or cards in parsed:
            raise HuRlScalarParityError(f"{context} card key is invalid or duplicated")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise HuRlScalarParityError(f"{context} EV is not numeric")
        numeric = float(raw_value)
        if not math.isfinite(numeric):
            raise HuRlScalarParityError(f"{context} EV is not finite")
        parsed[cards] = numeric
    if not parsed:
        raise HuRlScalarParityError(f"{context} is empty")
    return [
        {"cards": cards, "value_hex": parsed[cards].hex()}
        for cards in sorted(parsed)
    ]


def _effective_scoring_identity(root: Path) -> dict[str, Any]:
    relpath = _FL_EV_CONFIG_RELPATH
    config_path = root / relpath
    try:
        encoded = config_path.read_bytes()
    except OSError as exc:
        raise HuRlScalarParityError("FL EV config could not be read") from exc
    if not encoded:
        raise HuRlScalarParityError("FL EV config is empty")
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, HuRlScalarParityError) as exc:
        raise HuRlScalarParityError("FL EV config is not strict JSON") from exc
    if not isinstance(payload, Mapping) or "fl_ev" not in payload:
        raise HuRlScalarParityError("FL EV config is missing fl_ev")
    config_entries = _canonical_scoring_entries(payload["fl_ev"], "FL EV config")
    teacher_entries = _canonical_scoring_entries(
        _teacher_module.DEFAULT_FL_EV, "teacher effective FL EV"
    )
    reference_entries = _canonical_scoring_entries(
        _hu_rl_reference_module.DEFAULT_FL_EV, "reference effective FL EV"
    )
    scoring_context = getattr(_hu_rl_reference_module, "_SCORING", None)
    context_entries = _canonical_scoring_entries(
        getattr(scoring_context, "fl_ev", None), "reference scoring context FL EV"
    )
    if not (
        config_entries == teacher_entries == reference_entries == context_entries
    ):
        raise HuRlScalarParityError(
            "frozen FL EV config disagrees with effective Python scoring"
        )
    return {
        "schema": EFFECTIVE_SCORING_SCHEMA,
        "config_relpath": relpath,
        "fl_ev": config_entries,
        "canonical_sha256": _canonical_digest(config_entries),
    }


def _validate_effective_scoring_identity(identity: Mapping[str, Any]) -> None:
    _require_exact_fields(
        identity, _EFFECTIVE_SCORING_FIELDS, "run contract effective scoring"
    )
    if (
        identity["schema"] != EFFECTIVE_SCORING_SCHEMA
        or identity["config_relpath"] != _FL_EV_CONFIG_RELPATH
    ):
        raise HuRlScalarParityError("run contract effective scoring identity changed")
    entries = identity["fl_ev"]
    if not isinstance(entries, list) or not entries:
        raise HuRlScalarParityError("run contract effective scoring is empty")
    previous_cards = 0
    for raw_entry in entries:
        entry = _mapping_value(raw_entry, "run contract scoring entry")
        _require_exact_fields(entry, _SCORING_ENTRY_FIELDS, "run contract scoring entry")
        if (
            not _strict_int(entry["cards"])
            or entry["cards"] <= previous_cards
            or not isinstance(entry["value_hex"], str)
        ):
            raise HuRlScalarParityError("run contract scoring entry is invalid")
        try:
            value = float.fromhex(entry["value_hex"])
        except ValueError as exc:
            raise HuRlScalarParityError(
                "run contract scoring value is invalid"
            ) from exc
        if not math.isfinite(value) or value.hex() != entry["value_hex"]:
            raise HuRlScalarParityError("run contract scoring value is not canonical")
        previous_cards = entry["cards"]
    _require_sha256(identity["canonical_sha256"], "effective scoring sha256")
    if identity["canonical_sha256"] != _canonical_digest(entries):
        raise HuRlScalarParityError("run contract effective scoring digest changed")


def _resolved_named_file_record(path: str | Path, label: str) -> dict[str, Any]:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise HuRlScalarParityError(f"{label} could not be resolved") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise HuRlScalarParityError(f"{label} is missing or unsafe")
    return _named_file_record(resolved, label)


def _numpy_runtime_identity() -> dict[str, Any]:
    module_file = getattr(np, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise HuRlScalarParityError("NumPy module source identity is unavailable")
    try:
        distribution = importlib_metadata.distribution("numpy")
        record_text = distribution.read_text("RECORD")
    except importlib_metadata.PackageNotFoundError as exc:
        raise HuRlScalarParityError("NumPy distribution identity is unavailable") from exc
    if not isinstance(record_text, str) or not record_text:
        raise HuRlScalarParityError("NumPy distribution RECORD is unavailable")
    record_bytes = record_text.encode("utf-8")
    distribution_name = distribution.metadata.get("Name")
    distribution_version = distribution.version
    if not isinstance(distribution_name, str) or not distribution_name:
        raise HuRlScalarParityError("NumPy distribution name is unavailable")
    if distribution_name.casefold() != "numpy" or distribution_version != str(
        np.__version__
    ):
        raise HuRlScalarParityError(
            "NumPy module and distribution identities disagree"
        )
    return {
        "distribution_name": distribution_name,
        "version": distribution_version,
        "module_file": _resolved_named_file_record(module_file, "NumPy module file"),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "record_size_bytes": len(record_bytes),
    }


def _source_file_record(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HuRlScalarParityError("scalar parity validator source is missing or unsafe")
    try:
        encoded = candidate.read_bytes()
    except OSError as exc:
        raise HuRlScalarParityError("scalar parity validator source could not be read") from exc
    if not encoded:
        raise HuRlScalarParityError("scalar parity validator source is empty")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "size_bytes": len(encoded)}


def _named_file_record(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HuRlScalarParityError(f"{label} is missing or unsafe")
    try:
        encoded = candidate.read_bytes()
    except OSError as exc:
        raise HuRlScalarParityError(f"{label} could not be read") from exc
    if not encoded:
        raise HuRlScalarParityError(f"{label} is empty")
    return {
        "name": candidate.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _validate_named_file_record(record: Mapping[str, Any], context: str) -> None:
    _require_exact_fields(record, _FILE_RECORD_FIELDS, context)
    if not isinstance(record["name"], str) or not record["name"]:
        raise HuRlScalarParityError(f"{context} name is invalid")
    _require_sha256(record["sha256"], f"{context} sha256")
    if not _strict_int(record["size_bytes"]) or record["size_bytes"] <= 0:
        raise HuRlScalarParityError(f"{context} size is invalid")


def _native_binary_record(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HuRlScalarBinaryError("pinned scalar parity binary is missing or unsafe")
    try:
        encoded = candidate.read_bytes()
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise HuRlScalarBinaryError("pinned scalar parity binary could not be read") from exc
    if not encoded:
        raise HuRlScalarBinaryError("pinned scalar parity binary is empty")
    digest = hashlib.sha256(encoded).hexdigest()
    suffix = candidate.suffix
    stem = candidate.name[: -len(suffix)] if suffix else candidate.name
    if not stem.endswith(f".{digest}"):
        raise HuRlScalarBinaryError(
            "scalar parity binary path is not content-addressed"
        )
    read_only = not bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    if not read_only:
        raise HuRlScalarBinaryError("pinned scalar parity binary is writable")
    return {
        "content_addressed_name": candidate.name,
        "content_addressed_path_identity": f"sha256/{digest}/{candidate.name}",
        "sha256": digest,
        "size_bytes": len(encoded),
        "read_only": True,
    }


def _atomic_write_no_clobber(
    path: Path,
    encoded: bytes,
    error_type: type[Exception],
) -> None:
    """Publish complete bytes with fsync and an atomic no-clobber hard link."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise error_type("artifact directory could not be prepared") from exc
    if path.is_symlink():
        raise error_type("artifact target is unsafe")
    if path.exists():
        try:
            if not path.is_file() or path.read_bytes() != encoded:
                raise error_type("existing artifact does not match frozen bytes")
        except OSError as exc:
            raise error_type("existing artifact could not be validated") from exc
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise error_type("concurrent artifact does not match frozen bytes")
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    except error_type:
        raise
    except OSError as exc:
        raise error_type("artifact could not be published atomically") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HuRlScalarParityError("JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise HuRlScalarParityError("JSON contains an invalid numeric constant")


def _string_sequence(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HuRlScalarParityError(f"{context} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise HuRlScalarParityError(f"{context} must contain only strings")
    return tuple(value)


def _mapping_value(value: Any, context: str) -> Mapping[str, Any]:
    _require_mapping(value, context)
    return value


def _require_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise HuRlScalarParityError(f"{context} must be a mapping")


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    if missing:
        raise HuRlScalarParityError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise HuRlScalarParityError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _require_sha256(value: Any, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HuRlScalarParityError(f"{context} must be a lowercase SHA-256 digest")


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
