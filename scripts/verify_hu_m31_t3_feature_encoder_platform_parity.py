"""Prove fixed-corpus Windows/Linux parity for the M3.1 feature encoder.

The probe deliberately uses only the Python standard library and ``ctypes``.
It does not import OFC runtime code, resolve a policy profile, read a root,
or mutate a seed/model/root artifact.  Both platforms encode the same fixed
64-state x 8-action corpus.  The comparison receipt then binds that behavioral
parity to the already-open performance-lock claim/materialization/seal chain.

This is a fixed-corpus binary parity proof, not a reproducible-build
attestation and not a claim of mathematical solver optimality.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROBE_SCHEMA = "hu_m31_t3_feature_encoder_platform_probe_v1"
RECEIPT_SCHEMA = "hu_m31_t3_feature_encoder_platform_parity_receipt_v1"
PROBE_INPUT_SCHEMA = "hu_m31_t3_feature_encoder_platform_probe_input_v1"
CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_global_claim_v1"
MATERIALIZATION_SCHEMA = "hu_m31_t3_step6d_performance_lock_root_materialization_v1"
SEAL_SCHEMA = "hu_m31_t3_step6d_performance_lock_root_seal_v1"

STATE_COUNT = 64
ACTION_COUNT = 8
MAX_PLACEMENTS = 2
MAX_DISCARDS = 1
FEATURE_DIM = 1076
ROW_COUNT = STATE_COUNT * ACTION_COUNT
OUTPUT_ENCODING = "little_endian_ieee754_f32_features_then_i32_state_then_i16_action"
EXPECTED_LINUX_LIBRARY_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
EXPECTED_AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

WINDOWS_PROBE_NAME = "materializer_windows_probe.json"
LINUX_PROBE_NAME = "materializer_linux_probe.json"
PARITY_RECEIPT_NAME = "materializer_platform_parity.json"

_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "scope",
        "opened_unix_ns",
        "global_claim_path",
        "lock_output_directory",
        "precontent_plan",
        "development_go",
        "step6d_contract",
        "lock_run_contract",
        "lock_run_contract_digest",
        "runner_source",
        "ai_profiles_current",
        "accepted_binaries",
        "model_inputs",
        "root_generator_inputs",
        "image",
        "allocation",
        "seed_contract",
        "startup_source",
        "restrictions",
    }
)
_MATERIALIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_claim_sha256",
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "same_identity_resume_only",
        "reseeded",
        "training_eligible",
        "current_profile_changed",
    }
)
_SEAL_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_claim_sha256",
        "materialization_sha256",
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "observation_count",
        "profile_counts",
        "seat_counts",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "root_topology_sha256",
        "observation_fingerprint_sha256",
        "root_artifact_unique",
        "observation_fingerprint_unique",
        "development_comparison",
        "visibility",
        "selection_inputs",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)
_PROBE_KEYS = frozenset(
    {
        "schema",
        "status",
        "platform_label",
        "generator",
        "build_inputs",
        "library",
        "input_bytes",
        "input_sha256",
        "encoder_status",
        "feature_dim",
        "row_count",
        "output_encoding",
        "output_bytes",
        "output_sha256",
    }
)


class _RustProfile(ctypes.Structure):
    _fields_ = [
        ("total_seconds", ctypes.c_double),
        ("after_board_seconds", ctypes.c_double),
        ("row_summary_seconds", ctypes.c_double),
        ("global_summary_seconds", ctypes.c_double),
        ("action_delta_seconds", ctypes.c_double),
        ("rows", ctypes.c_uint64),
    ]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return target.resolve()


def _file_record(path: str | Path, label: str) -> dict[str, Any]:
    target = _safe_file(path, label)
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = _safe_file(path, label)
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd: int | None = None
    try:
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _state_deck(state_index: int) -> list[int]:
    deck = list(range(52))
    state = (0x6A09E667F3BCC909 + state_index * 0xD1342543DE82EF95) & ((1 << 64) - 1)
    for upper in range(51, 0, -1):
        state = _splitmix64(state)
        selected = state % (upper + 1)
        deck[upper], deck[selected] = deck[selected], deck[upper]
    return deck


def _mask(cards: Sequence[int]) -> int:
    value = 0
    for card in cards:
        if card < 0 or card >= 52:
            raise AssertionError("fixed probe generated an invalid card")
        value |= 1 << card
    return value


def _fixed_input_values() -> dict[str, list[int]]:
    values: dict[str, list[int]] = {
        "hero_board_masks": [],
        "opponent_board_masks": [],
        "dead_card_masks": [],
        "dealt_card_ids": [],
        "seat_ids": [],
        "order_ids": [],
        "action_counts": [],
        "action_placement_card_ids": [],
        "action_placement_row_ids": [],
        "action_discard_card_ids": [],
    }
    action_specs = (
        (0, 1, 0, 2, 1),
        (0, 1, 1, 2, 2),
        (0, 1, 0, 2, 2),
        (1, 0, 0, 2, 1),
        (1, 0, 1, 2, 2),
        (1, 0, 0, 2, 2),
        (2, 0, 0, 1, 1),
        (2, 0, 1, 1, 2),
    )

    for state_index in range(STATE_COUNT):
        deck = _state_deck(state_index)
        order_id = (state_index >> 1) & 1
        seat_id = state_index & 1
        hero = deck[:9]
        opponent_count = 11 if order_id else 9
        opponent = deck[9 : 9 + opponent_count]
        cursor = 9 + opponent_count
        dead = deck[cursor : cursor + 2]
        dealt = deck[cursor + 2 : cursor + 5]

        values["hero_board_masks"].extend(
            (_mask(hero[:2]), _mask(hero[2:5]), _mask(hero[5:9]))
        )
        if opponent_count == 9:
            opponent_rows = (opponent[:2], opponent[2:5], opponent[5:9])
        else:
            opponent_rows = (opponent[:3], opponent[3:7], opponent[7:11])
        values["opponent_board_masks"].extend(_mask(row) for row in opponent_rows)
        values["dead_card_masks"].append(_mask(dead))
        values["dealt_card_ids"].extend(dealt)
        values["seat_ids"].append(seat_id)
        values["order_ids"].append(order_id)
        values["action_counts"].append(ACTION_COUNT)

        for (
            discard_index,
            first_index,
            first_row,
            second_index,
            second_row,
        ) in action_specs:
            values["action_placement_card_ids"].extend(
                (dealt[first_index], dealt[second_index])
            )
            values["action_placement_row_ids"].extend((first_row, second_row))
            values["action_discard_card_ids"].append(dealt[discard_index])

    return values


_INPUT_LAYOUT = (
    ("hero_board_masks", "Q", ctypes.c_uint64),
    ("opponent_board_masks", "Q", ctypes.c_uint64),
    ("dead_card_masks", "Q", ctypes.c_uint64),
    ("dealt_card_ids", "h", ctypes.c_int16),
    ("seat_ids", "b", ctypes.c_int8),
    ("order_ids", "b", ctypes.c_int8),
    ("action_counts", "h", ctypes.c_int16),
    ("action_placement_card_ids", "h", ctypes.c_int16),
    ("action_placement_row_ids", "b", ctypes.c_int8),
    ("action_discard_card_ids", "h", ctypes.c_int16),
)


def _fixed_input_bytes(values: Mapping[str, Sequence[int]]) -> bytes:
    chunks = [
        PROBE_INPUT_SCHEMA.encode("ascii") + b"\0",
        struct.pack(
            "<6I",
            STATE_COUNT,
            ACTION_COUNT,
            MAX_PLACEMENTS,
            MAX_DISCARDS,
            FEATURE_DIM,
            ROW_COUNT,
        ),
    ]
    for name, format_code, _ctype in _INPUT_LAYOUT:
        name_bytes = name.encode("ascii")
        field = tuple(values[name])
        chunks.append(struct.pack("<H", len(name_bytes)))
        chunks.append(name_bytes)
        chunks.append(struct.pack("<Q", len(field)))
        chunks.append(struct.pack(f"<{len(field)}{format_code}", *field))
    return b"".join(chunks)


_FIXED_VALUES = _fixed_input_values()
_FIXED_INPUT_BYTES = _fixed_input_bytes(_FIXED_VALUES)
EXPECTED_INPUT_SHA256 = hashlib.sha256(_FIXED_INPUT_BYTES).hexdigest()
EXPECTED_INPUT_BYTES = len(_FIXED_INPUT_BYTES)
EXPECTED_OUTPUT_BYTES = (
    ROW_COUNT * FEATURE_DIM * ctypes.sizeof(ctypes.c_float)
    + ROW_COUNT * ctypes.sizeof(ctypes.c_int32)
    + ROW_COUNT * ctypes.sizeof(ctypes.c_int16)
)

PROBE_GENERATOR_CONTRACT = {
    "schema": "hu_m31_t3_feature_encoder_platform_probe_generator_v1",
    "input_schema": PROBE_INPUT_SCHEMA,
    "state_count": STATE_COUNT,
    "actions_per_state": ACTION_COUNT,
    "max_placements": MAX_PLACEMENTS,
    "max_discards": MAX_DISCARDS,
    "feature_dim": FEATURE_DIM,
    "row_count": ROW_COUNT,
    "input_bytes": EXPECTED_INPUT_BYTES,
    "input_sha256": EXPECTED_INPUT_SHA256,
    "output_encoding": OUTPUT_ENCODING,
    "output_bytes": EXPECTED_OUTPUT_BYTES,
    "opponent_private_discards_used": False,
    "root_or_seed_input_used": False,
}
PROBE_GENERATOR_CONTRACT_SHA256 = canonical_sha256(PROBE_GENERATOR_CONTRACT)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_build_inputs() -> dict[str, dict[str, Any]]:
    root = _repository_root()
    return {
        "rust_source": _file_record(
            root / "rust/ofc_stage3_feature_encoder/src/lib.rs",
            "Stage3 feature encoder Rust source",
        ),
        "cargo_lock": _file_record(root / "Cargo.lock", "Cargo.lock"),
    }


def _generator_record() -> dict[str, Any]:
    return {
        "contract": PROBE_GENERATOR_CONTRACT,
        "contract_sha256": PROBE_GENERATOR_CONTRACT_SHA256,
        "script": _file_record(Path(__file__), "platform parity script"),
    }


def _ctypes_inputs(
    values: Mapping[str, Sequence[int]],
) -> dict[str, ctypes.Array[Any]]:
    arrays: dict[str, ctypes.Array[Any]] = {}
    for name, _format_code, ctype in _INPUT_LAYOUT:
        field = tuple(values[name])
        arrays[name] = (ctype * len(field))(*field)
    return arrays


def _configure_library(library: ctypes.CDLL) -> None:
    library.ofc_stage3_feature_dim.argtypes = []
    library.ofc_stage3_feature_dim.restype = ctypes.c_size_t
    library.ofc_stage3_encode.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(_RustProfile),
    ]
    library.ofc_stage3_encode.restype = ctypes.c_int


def _load_library(path: Path) -> ctypes.CDLL:
    return ctypes.CDLL(str(path))


def _array_bytes(value: ctypes.Array[Any]) -> bytes:
    return ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))


def build_probe(library_path: str | Path, platform_label: str) -> dict[str, Any]:
    if platform_label not in {"windows", "linux"}:
        raise ValueError("platform label must be exactly 'windows' or 'linux'")
    if sys.byteorder != "little":
        raise ValueError("platform parity probe requires a little-endian host")
    library_record = _file_record(library_path, f"{platform_label} feature encoder")
    library_target = Path(library_record["path"])
    expected_suffix = ".dll" if platform_label == "windows" else ".so"
    if library_target.suffix.lower() != expected_suffix:
        raise ValueError(f"{platform_label} feature encoder must use {expected_suffix}")

    library = _load_library(library_target)
    _configure_library(library)
    feature_dim = int(library.ofc_stage3_feature_dim())
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"feature dimension changed: {feature_dim} != {FEATURE_DIM}")

    arrays = _ctypes_inputs(_FIXED_VALUES)
    features = (ctypes.c_float * (ROW_COUNT * FEATURE_DIM))()
    row_to_state = (ctypes.c_int32 * ROW_COUNT)()
    row_to_action = (ctypes.c_int16 * ROW_COUNT)()
    profile = _RustProfile()
    status = int(
        library.ofc_stage3_encode(
            STATE_COUNT,
            ACTION_COUNT,
            MAX_PLACEMENTS,
            MAX_DISCARDS,
            arrays["hero_board_masks"],
            arrays["opponent_board_masks"],
            arrays["dead_card_masks"],
            arrays["dealt_card_ids"],
            arrays["seat_ids"],
            arrays["order_ids"],
            arrays["action_counts"],
            arrays["action_placement_card_ids"],
            arrays["action_placement_row_ids"],
            arrays["action_discard_card_ids"],
            features,
            row_to_state,
            row_to_action,
            ctypes.byref(profile),
        )
    )
    if status != 0:
        raise RuntimeError(f"Rust Stage3 encoder returned status {status}")
    if int(profile.rows) != ROW_COUNT:
        raise RuntimeError(
            f"Rust Stage3 encoder row count changed: {profile.rows} != {ROW_COUNT}"
        )
    for row_index in range(ROW_COUNT):
        expected_state, expected_action = divmod(row_index, ACTION_COUNT)
        if (
            int(row_to_state[row_index]) != expected_state
            or int(row_to_action[row_index]) != expected_action
        ):
            raise RuntimeError("Rust Stage3 encoder row mapping changed")

    output = (
        _array_bytes(features)
        + _array_bytes(row_to_state)
        + _array_bytes(row_to_action)
    )
    if len(output) != EXPECTED_OUTPUT_BYTES:
        raise AssertionError("platform probe output layout changed")
    value = {
        "schema": PROBE_SCHEMA,
        "status": "fixed_512_row_feature_encoder_probe_complete",
        "platform_label": platform_label,
        "generator": _generator_record(),
        "build_inputs": _default_build_inputs(),
        "library": library_record,
        "input_bytes": len(_FIXED_INPUT_BYTES),
        "input_sha256": hashlib.sha256(_FIXED_INPUT_BYTES).hexdigest(),
        "encoder_status": status,
        "feature_dim": feature_dim,
        "row_count": int(profile.rows),
        "output_encoding": OUTPUT_ENCODING,
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }
    if set(value) != _PROBE_KEYS:
        raise AssertionError("platform probe schema implementation changed")
    return value


def write_probe(
    library_path: str | Path,
    platform_label: str,
    output_path: str | Path,
) -> dict[str, Any]:
    expected_name = (
        WINDOWS_PROBE_NAME if platform_label == "windows" else LINUX_PROBE_NAME
    )
    if Path(output_path).name != expected_name:
        raise ValueError(f"{platform_label} probe output must be named {expected_name}")
    value = build_probe(library_path, platform_label)
    _write_once(output_path, value)
    return value


def _validate_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _validate_file_record_shape(
    value: Any, label: str, *, expected_suffix: str | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} file record changed")
    path = value.get("path")
    byte_count = value.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} path changed")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise ValueError(f"{label} byte count changed")
    _validate_hash(value.get("sha256"), f"{label} hash")
    if expected_suffix is not None and Path(path).suffix.lower() != expected_suffix:
        raise ValueError(f"{label} suffix changed")
    return value


def _same_file_content(
    recorded: Mapping[str, Any], actual: Mapping[str, Any], label: str
) -> None:
    if recorded.get("bytes") != actual.get("bytes") or recorded.get(
        "sha256"
    ) != actual.get("sha256"):
        raise ValueError(f"{label} content changed")


def _validate_probe(
    value: Mapping[str, Any],
    *,
    expected_label: str,
    actual_script: Mapping[str, Any],
    actual_rust_source: Mapping[str, Any],
    actual_cargo_lock: Mapping[str, Any],
) -> None:
    if set(value) != _PROBE_KEYS:
        raise ValueError(f"{expected_label} probe fields changed")
    if (
        value.get("schema") != PROBE_SCHEMA
        or value.get("status") != "fixed_512_row_feature_encoder_probe_complete"
        or value.get("platform_label") != expected_label
    ):
        raise ValueError(f"{expected_label} probe identity changed")

    generator = value.get("generator")
    if not isinstance(generator, dict) or set(generator) != {
        "contract",
        "contract_sha256",
        "script",
    }:
        raise ValueError(f"{expected_label} generator record changed")
    if (
        generator.get("contract") != PROBE_GENERATOR_CONTRACT
        or generator.get("contract_sha256") != PROBE_GENERATOR_CONTRACT_SHA256
        or canonical_sha256(generator["contract"]) != generator["contract_sha256"]
    ):
        raise ValueError(f"{expected_label} generator contract changed")
    script_record = _validate_file_record_shape(
        generator.get("script"), f"{expected_label} probe script"
    )
    if Path(script_record["path"]).name != Path(__file__).name:
        raise ValueError(f"{expected_label} probe script path changed")
    _same_file_content(script_record, actual_script, f"{expected_label} probe script")

    build_inputs = value.get("build_inputs")
    if not isinstance(build_inputs, dict) or set(build_inputs) != {
        "rust_source",
        "cargo_lock",
    }:
        raise ValueError(f"{expected_label} build input record changed")
    rust_record = _validate_file_record_shape(
        build_inputs.get("rust_source"), f"{expected_label} Rust source"
    )
    lock_record = _validate_file_record_shape(
        build_inputs.get("cargo_lock"), f"{expected_label} Cargo.lock"
    )
    _same_file_content(rust_record, actual_rust_source, f"{expected_label} Rust source")
    _same_file_content(lock_record, actual_cargo_lock, f"{expected_label} Cargo.lock")

    suffix = ".dll" if expected_label == "windows" else ".so"
    _validate_file_record_shape(
        value.get("library"),
        f"{expected_label} feature encoder",
        expected_suffix=suffix,
    )
    expected_scalars = {
        "input_bytes": EXPECTED_INPUT_BYTES,
        "input_sha256": EXPECTED_INPUT_SHA256,
        "encoder_status": 0,
        "feature_dim": FEATURE_DIM,
        "row_count": ROW_COUNT,
        "output_encoding": OUTPUT_ENCODING,
        "output_bytes": EXPECTED_OUTPUT_BYTES,
    }
    for field, expected in expected_scalars.items():
        if value.get(field) != expected:
            raise ValueError(f"{expected_label} probe {field} changed")
    _validate_hash(value.get("output_sha256"), f"{expected_label} output hash")


def _validate_rust_source(path: str | Path) -> dict[str, Any]:
    record = _file_record(path, "Stage3 feature encoder Rust source")
    text = Path(record["path"]).read_text(encoding="utf-8")
    required = (
        "const FEATURES: usize = 1076;",
        'pub extern "C" fn ofc_stage3_feature_dim()',
        'pub unsafe extern "C" fn ofc_stage3_encode(',
    )
    if any(token not in text for token in required):
        raise ValueError("Stage3 feature encoder Rust ABI source changed")
    return record


def _validate_cargo_lock(path: str | Path) -> dict[str, Any]:
    record = _file_record(path, "Cargo.lock")
    raw = Path(record["path"]).read_bytes()
    if (
        b"# This file is automatically @generated by Cargo." not in raw[:256]
        or b'name = "regular_fl_solver"' not in raw
    ):
        raise ValueError("Cargo.lock no longer binds the encoder workspace package")
    return record


def _validate_ai_profiles(path: str | Path, claim: Mapping[str, Any]) -> dict[str, Any]:
    actual = _file_record(path, "ai_profiles.py")
    if actual["sha256"] != EXPECTED_AI_PROFILES_SHA256:
        raise ValueError("ai_profiles/current bytes changed")
    recorded = _validate_file_record_shape(
        claim.get("ai_profiles_current"), "claim ai_profiles/current"
    )
    _same_file_content(recorded, actual, "claim ai_profiles/current")
    if recorded["sha256"] != EXPECTED_AI_PROFILES_SHA256:
        raise ValueError("claim current profile anchor changed")
    return actual


def _validate_lock_chain(
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
    *,
    linux_probe: Mapping[str, Any],
    ai_profiles_path: str | Path,
) -> dict[str, Any]:
    if set(claim) != _CLAIM_KEYS or claim.get("schema") != CLAIM_SCHEMA:
        raise ValueError("performance-lock global claim schema changed")
    if (
        claim.get("status")
        != "global_one_shot_claim_persisted_before_lock_root_touch_crash_consumes_claim"
        or claim.get("scope") != "candidate02_performance_lock_roots_only"
    ):
        raise ValueError("performance-lock global claim identity changed")
    if (
        set(materialization) != _MATERIALIZATION_KEYS
        or materialization.get("schema") != MATERIALIZATION_SCHEMA
        or materialization.get("status")
        != "all_100_lock_roots_materialized_same_identity"
    ):
        raise ValueError("performance-lock materialization schema changed")
    if (
        set(seal) != _SEAL_KEYS
        or seal.get("schema") != SEAL_SCHEMA
        or seal.get("status")
        != "sealed_100_disjoint_hidden_safe_performance_lock_roots"
    ):
        raise ValueError("performance-lock seal schema changed")

    claim_sha = canonical_sha256(claim)
    materialization_sha = canonical_sha256(materialization)
    if (
        materialization.get("global_claim_sha256") != claim_sha
        or seal.get("global_claim_sha256") != claim_sha
        or seal.get("materialization_sha256") != materialization_sha
    ):
        raise ValueError(
            "performance-lock claim/materialization/seal hash chain changed"
        )

    accepted = claim.get("accepted_binaries")
    if not isinstance(accepted, dict) or set(accepted) != {
        "candidate",
        "reference",
        "feature_encoder",
    }:
        raise ValueError("claim accepted binary set changed")
    accepted_feature = _validate_file_record_shape(
        accepted.get("feature_encoder"),
        "claim accepted Linux feature encoder",
        expected_suffix=".so",
    )
    linux_library = linux_probe["library"]
    _same_file_content(
        accepted_feature, linux_library, "claim accepted Linux feature encoder"
    )
    if accepted_feature["sha256"] != EXPECTED_LINUX_LIBRARY_SHA256:
        raise ValueError("accepted Linux feature encoder hash changed")

    restrictions = claim.get("restrictions")
    required_false = (
        "alternate_seed_allowed",
        "reseed_allowed",
        "current_profile_resolution_allowed",
        "opponent_private_discards_allowed",
        "training_authorized",
        "promotion_authorized",
        "runtime_activation_allowed",
    )
    if not isinstance(restrictions, dict) or any(
        restrictions.get(field) is not False for field in required_false
    ):
        raise ValueError("performance-lock claim restrictions changed")

    hand_indices = list(range(100))
    root_hashes = materialization.get("root_artifact_sha256")
    if (
        materialization.get("root_count") != 100
        or materialization.get("hand_indices") != hand_indices
        or not isinstance(root_hashes, list)
        or len(root_hashes) != 100
        or len(set(root_hashes)) != 100
        or any(
            _validate_hash(value, "materialized root hash") != value
            for value in root_hashes
        )
        or materialization.get("aggregate_root_sha256") != canonical_sha256(root_hashes)
        or materialization.get("same_identity_resume_only") is not True
        or materialization.get("reseeded") is not False
        or materialization.get("training_eligible") is not False
        or materialization.get("current_profile_changed") is not False
    ):
        raise ValueError("performance-lock materialization invariants changed")

    matching_fields = (
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "root_artifact_sha256",
        "aggregate_root_sha256",
    )
    if any(seal.get(field) != materialization.get(field) for field in matching_fields):
        raise ValueError("performance-lock materialization/seal root identity changed")
    if (
        seal.get("observation_count") != 200
        or seal.get("seat_counts") != {"first": 100, "second": 100}
        or seal.get("root_artifact_unique") is not True
        or seal.get("observation_fingerprint_unique") is not True
        or seal.get("training_eligible") is not False
        or seal.get("quality_evidence") is not False
        or seal.get("promotion_evidence") is not False
        or seal.get("current_profile_changed") is not False
        or seal.get("named_profile_added") is not False
        or seal.get("runtime_policy_activated") is not False
    ):
        raise ValueError("performance-lock seal invariants changed")
    visibility = seal.get("visibility")
    if not isinstance(visibility, dict) or (
        visibility.get("opponent_private_discards_used") is not False
        or visibility.get("current_profile_resolved") is not False
    ):
        raise ValueError("performance-lock visibility invariants changed")
    selection = seal.get("selection_inputs")
    if not isinstance(selection, dict) or (
        selection.get("all_100_preregistered_hands_used") is not True
        or selection.get("timing_used") is not False
        or selection.get("q_used") is not False
        or selection.get("ev_used") is not False
    ):
        raise ValueError("performance-lock selection invariants changed")

    ai_profiles = _validate_ai_profiles(ai_profiles_path, claim)
    return {
        "global_claim_sha256": claim_sha,
        "materialization_sha256": materialization_sha,
        "seal_sha256": canonical_sha256(seal),
        "accepted_linux_feature_encoder_sha256": accepted_feature["sha256"],
        "ai_profiles_sha256": ai_profiles["sha256"],
        "root_count": 100,
        "reseeded": False,
        "current_profile_changed": False,
    }


def _artifact_record(path: str | Path, label: str) -> dict[str, Any]:
    value = _read_canonical(path, label)
    record = _file_record(path, label)
    record["canonical_sha256"] = canonical_sha256(value)
    if record["sha256"] != record["canonical_sha256"]:
        raise ValueError(f"{label} canonical/file digest mismatch")
    return record


def compare_probes(
    *,
    windows_probe_path: str | Path,
    linux_probe_path: str | Path,
    global_claim_path: str | Path,
    materialization_path: str | Path,
    seal_path: str | Path,
    rust_source_path: str | Path,
    cargo_lock_path: str | Path,
    ai_profiles_path: str | Path,
) -> dict[str, Any]:
    actual_script = _file_record(Path(__file__), "platform parity script")
    rust_source = _validate_rust_source(rust_source_path)
    cargo_lock = _validate_cargo_lock(cargo_lock_path)
    windows_probe = _read_canonical(windows_probe_path, "Windows platform probe")
    linux_probe = _read_canonical(linux_probe_path, "Linux platform probe")
    _validate_probe(
        windows_probe,
        expected_label="windows",
        actual_script=actual_script,
        actual_rust_source=rust_source,
        actual_cargo_lock=cargo_lock,
    )
    _validate_probe(
        linux_probe,
        expected_label="linux",
        actual_script=actual_script,
        actual_rust_source=rust_source,
        actual_cargo_lock=cargo_lock,
    )
    if (
        windows_probe["input_bytes"] != linux_probe["input_bytes"]
        or windows_probe["input_sha256"] != linux_probe["input_sha256"]
    ):
        raise ValueError("Windows/Linux platform probe input is not bit-exact")
    if (
        windows_probe["output_bytes"] != linux_probe["output_bytes"]
        or windows_probe["output_sha256"] != linux_probe["output_sha256"]
    ):
        raise ValueError("Windows/Linux feature output is not bit-exact")

    claim = _read_canonical(global_claim_path, "performance-lock global claim")
    materialization = _read_canonical(
        materialization_path, "performance-lock materialization"
    )
    seal = _read_canonical(seal_path, "performance-lock seal")
    chain = _validate_lock_chain(
        claim,
        materialization,
        seal,
        linux_probe=linux_probe,
        ai_profiles_path=ai_profiles_path,
    )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": (
            "verified_fixed_512_row_windows_linux_bit_exact_"
            "and_performance_lock_chain_intact"
        ),
        "generator": {
            "input_schema": PROBE_INPUT_SCHEMA,
            "contract_sha256": PROBE_GENERATOR_CONTRACT_SHA256,
            "script": actual_script,
        },
        "platform_parity": {
            "input_bit_exact": True,
            "output_bit_exact": True,
            "row_mapping_checked_by_each_probe": True,
            "row_count": ROW_COUNT,
            "input_bytes": windows_probe["input_bytes"],
            "input_sha256": windows_probe["input_sha256"],
            "output_bytes": windows_probe["output_bytes"],
            "output_sha256": windows_probe["output_sha256"],
            "accepted_linux_library_bound_to_global_claim": True,
            "fixed_corpus_behavioral_parity_not_reproducible_build_attestation": True,
        },
        "libraries": {
            "windows": windows_probe["library"],
            "linux": linux_probe["library"],
        },
        "build_inputs": {
            "same_rust_source_content_in_both_probes": True,
            "same_cargo_lock_content_in_both_probes": True,
            "rust_source": rust_source,
            "cargo_lock": cargo_lock,
        },
        "lock_chain": chain,
        "artifacts": {
            "windows_probe": _artifact_record(
                windows_probe_path, "Windows platform probe"
            ),
            "linux_probe": _artifact_record(linux_probe_path, "Linux platform probe"),
            "global_claim": _artifact_record(
                global_claim_path, "performance-lock global claim"
            ),
            "materialization": _artifact_record(
                materialization_path, "performance-lock materialization"
            ),
            "seal": _artifact_record(seal_path, "performance-lock seal"),
            "ai_profiles": _file_record(ai_profiles_path, "ai_profiles.py"),
        },
        "mutation_scope": {
            "only_new_receipt_written": True,
            "seed_changed": False,
            "model_changed": False,
            "root_changed": False,
            "current_profile_changed": False,
            "profile_resolved": False,
            "opponent_private_discards_used": False,
        },
        "quality_evidence": False,
        "promotion_evidence": False,
        "training_eligible": False,
    }
    return receipt


def write_comparison_receipt(
    *,
    windows_probe_path: str | Path,
    linux_probe_path: str | Path,
    global_claim_path: str | Path,
    materialization_path: str | Path,
    seal_path: str | Path,
    rust_source_path: str | Path,
    cargo_lock_path: str | Path,
    ai_profiles_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    if Path(output_path).name != PARITY_RECEIPT_NAME:
        raise ValueError(f"parity receipt must be named {PARITY_RECEIPT_NAME}")
    receipt = compare_probes(
        windows_probe_path=windows_probe_path,
        linux_probe_path=linux_probe_path,
        global_claim_path=global_claim_path,
        materialization_path=materialization_path,
        seal_path=seal_path,
        rust_source_path=rust_source_path,
        cargo_lock_path=cargo_lock_path,
        ai_profiles_path=ai_profiles_path,
    )
    _write_once(output_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--library", type=Path, required=True)
    probe.add_argument("--platform-label", choices=("windows", "linux"), required=True)
    probe.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--windows-probe", type=Path, required=True)
    compare.add_argument("--linux-probe", type=Path, required=True)
    compare.add_argument("--global-claim", type=Path, required=True)
    compare.add_argument("--materialization", type=Path, required=True)
    compare.add_argument("--seal", type=Path, required=True)
    compare.add_argument("--rust-source", type=Path, required=True)
    compare.add_argument("--cargo-lock", type=Path, required=True)
    compare.add_argument("--ai-profiles", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe":
        value = write_probe(args.library, args.platform_label, args.output)
    else:
        value = write_comparison_receipt(
            windows_probe_path=args.windows_probe,
            linux_probe_path=args.linux_probe,
            global_claim_path=args.global_claim,
            materialization_path=args.materialization,
            seal_path=args.seal,
            rust_source_path=args.rust_source,
            cargo_lock_path=args.cargo_lock,
            ai_profiles_path=args.ai_profiles,
            output_path=args.output,
        )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
