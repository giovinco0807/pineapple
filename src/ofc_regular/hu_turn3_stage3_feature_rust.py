"""ctypes bridge for the Rust HU Turn3 Stage3 feature encoder."""

from __future__ import annotations

import ctypes
import hashlib
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .action_space import Action
from .hu_turn3_model import CARD_INDEX, HU_FEATURE_DIM
from .hu_turn3_stage3_feature_fast import (
    FEATURE_SCHEMA_VERSION,
    FeatureBatch,
    Stage3StateFeatureCache,
)

ENCODER_MODE_RUST_DIRECT = "rust_direct"
ROW_INDEX = {"top": 0, "middle": 1, "bottom": 2}
_PINNED_LIBRARY: ContextVar[tuple[Path, str] | None] = ContextVar(
    "hu_turn3_stage3_pinned_feature_encoder_library",
    default=None,
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


def build_hu_turn3_stage3_feature_matrix_rust(
    t3_states: Sequence[Any],
    legal_actions_by_state: Sequence[list[Action]],
    feature_schema: str = FEATURE_SCHEMA_VERSION,
    *,
    state_keys: Sequence[tuple[Any, ...]] | None = None,
    state_feature_cache: Stage3StateFeatureCache | None = None,
    debug_sample_limit: int = 0,
    include_action_encodings: bool = False,
    encoder_mode: str = ENCODER_MODE_RUST_DIRECT,
) -> FeatureBatch:
    del state_keys, state_feature_cache, include_action_encodings
    if feature_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    if encoder_mode != ENCODER_MODE_RUST_DIRECT:
        raise ValueError(f"unsupported rust encoder mode: {encoder_mode}")
    if len(t3_states) != len(legal_actions_by_state):
        raise ValueError("states/actions length mismatch")

    lib = _load_library()
    arrays = compact_arrays_from_t3_states(t3_states, legal_actions_by_state)
    rows = int(np.asarray(arrays["action_counts"], dtype=np.int64).sum())
    X = np.zeros((rows, HU_FEATURE_DIM), dtype=np.float32)
    row_to_state_index = np.empty(rows, dtype=np.int32)
    row_to_action_index = np.empty(rows, dtype=np.int16)
    profile = _RustProfile()

    started_at = time.perf_counter()
    status = lib.ofc_stage3_encode(
        ctypes.c_size_t(len(t3_states)),
        ctypes.c_size_t(int(arrays["max_actions"])),
        ctypes.c_size_t(int(arrays["max_placements"])),
        ctypes.c_size_t(int(arrays["max_discards"])),
        _ptr(arrays["hero_board_masks"], ctypes.c_uint64),
        _ptr(arrays["opponent_board_masks"], ctypes.c_uint64),
        _ptr(arrays["dead_card_masks"], ctypes.c_uint64),
        _ptr(arrays["dealt_card_ids"], ctypes.c_int16),
        _ptr(arrays["seat_ids"], ctypes.c_int8),
        _ptr(arrays["order_ids"], ctypes.c_int8),
        _ptr(arrays["action_counts"], ctypes.c_int16),
        _ptr(arrays["action_placement_card_ids"], ctypes.c_int16),
        _ptr(arrays["action_placement_row_ids"], ctypes.c_int8),
        _ptr(arrays["action_discard_card_ids"], ctypes.c_int16),
        _ptr(X, ctypes.c_float),
        _ptr(row_to_state_index, ctypes.c_int32),
        _ptr(row_to_action_index, ctypes.c_int16),
        ctypes.byref(profile),
    )
    wall_seconds = time.perf_counter() - started_at
    if int(status) != 0:
        raise RuntimeError(f"Rust Stage3 feature encoder failed with status {status}")
    if int(profile.rows) != rows:
        raise RuntimeError(f"Rust Stage3 feature row mismatch: {profile.rows} != {rows}")

    non_encoder = max(
        0.0,
        float(profile.total_seconds)
        - float(profile.after_board_seconds)
        - float(profile.row_summary_seconds)
        - float(profile.global_summary_seconds)
        - float(profile.action_delta_seconds),
    )
    return FeatureBatch(
        X=X,
        row_to_state_index=row_to_state_index,
        row_to_action_index=row_to_action_index,
        action_encodings=[],
        feature_column_names=[f"f{index}" for index in range(HU_FEATURE_DIM)],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        debug_sample_rows=[] if debug_sample_limit <= 0 else _debug_rows(X, row_to_state_index, row_to_action_index, debug_sample_limit),
        profile={
            "stage3_feature_mode": ENCODER_MODE_RUST_DIRECT,
            "stage3_feature_generation_total": wall_seconds,
            "stage3_feature_rows": int(X.shape[0]),
            "stage3_state_feature_time": 0.0,
            "stage3_action_feature_time": wall_seconds,
            "stage3_matrix_assembly_time": max(0.0, wall_seconds - float(profile.total_seconds)),
            "stage3_sample_generation_time": 0.0,
            "stage3_context_generation_time": 0.0,
            "stage3_row_global_summary_time": float(profile.row_summary_seconds + profile.global_summary_seconds),
            "stage3_candidate_delta_feature_time": float(profile.action_delta_seconds),
            "stage3_encoder_matrix_build_seconds": wall_seconds,
            "stage3_after_board_construction_seconds": float(profile.after_board_seconds),
            "stage3_row_summary_seconds": float(profile.row_summary_seconds),
            "stage3_global_summary_seconds": float(profile.global_summary_seconds),
            "stage3_action_delta_seconds": float(profile.action_delta_seconds),
            "stage3_scalar_fallback_seconds": 0.0,
            "stage3_cache_lookup_update_seconds": 0.0,
            "stage3_column_validation_seconds": 0.0,
            "stage3_numpy_allocation_seconds": 0.0,
            "stage3_hgb_input_preparation_seconds": 0.0,
            "stage3_non_encoder_overhead_seconds": non_encoder,
            "stage3_state_feature_cache_hit": 0,
            "stage3_state_feature_cache_miss": len(t3_states),
            "stage3_state_feature_cache_hit_rate": 0.0,
            "feature_dtype": str(X.dtype),
            "feature_column_count": int(X.shape[1]) if X.ndim == 2 else 0,
            "direct_column_count": int(X.shape[1]) if X.ndim == 2 else 0,
            "scalar_fallback_column_count": 0,
            "direct_column_coverage_ratio": 1.0 if X.ndim == 2 and X.shape[1] else 0.0,
            "memory_peak_mb": float(X.nbytes) / (1024.0 * 1024.0),
            "rust_encoder_core_seconds": float(profile.total_seconds),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
    )


def compact_arrays_from_t3_states(
    states: Sequence[Any],
    actions_by_state: Sequence[list[Action]],
) -> dict[str, Any]:
    max_actions = max((len(actions) for actions in actions_by_state), default=0)
    max_placements = max(
        (len(action.placements) for actions in actions_by_state for action in actions),
        default=0,
    )
    max_discards = max(
        (len(action.discards) for actions in actions_by_state for action in actions),
        default=0,
    )
    state_count = len(states)
    hero_board_masks = np.zeros((state_count, 3), dtype=np.uint64)
    opponent_board_masks = np.zeros((state_count, 3), dtype=np.uint64)
    dead_card_masks = np.zeros(state_count, dtype=np.uint64)
    dealt_card_ids = np.full((state_count, 3), -1, dtype=np.int16)
    seat_ids = np.zeros(state_count, dtype=np.int8)
    order_ids = np.zeros(state_count, dtype=np.int8)
    action_counts = np.zeros(state_count, dtype=np.int16)
    placement_cards = np.full((state_count, max_actions, max_placements), -1, dtype=np.int16)
    placement_rows = np.full((state_count, max_actions, max_placements), -1, dtype=np.int8)
    discard_cards = np.full((state_count, max_actions, max_discards), -1, dtype=np.int16)

    for state_index, (state, actions) in enumerate(zip(states, actions_by_state)):
        hero_board_masks[state_index] = _board_masks(state.board)
        opponent_board_masks[state_index] = _board_masks(state.opponent_board)
        dead_card_masks[state_index] = _cards_mask(state.dead_cards)
        for card_index, card in enumerate(tuple(state.dealt_cards)[:3]):
            dealt_card_ids[state_index, card_index] = CARD_INDEX[card]
        seat_ids[state_index] = 1 if getattr(state, "seat", "first") == "second" else 0
        order_ids[state_index] = 1 if (getattr(state, "to_act_order", None) or "first") == "second" else 0
        action_counts[state_index] = len(actions)
        for action_index, action in enumerate(actions):
            for placement_index, (card, row) in enumerate(action.placements):
                placement_cards[state_index, action_index, placement_index] = CARD_INDEX[card]
                placement_rows[state_index, action_index, placement_index] = ROW_INDEX[row]
            for discard_index, card in enumerate(action.discards):
                discard_cards[state_index, action_index, discard_index] = CARD_INDEX[card]

    return {
        "hero_board_masks": np.ascontiguousarray(hero_board_masks),
        "opponent_board_masks": np.ascontiguousarray(opponent_board_masks),
        "dead_card_masks": np.ascontiguousarray(dead_card_masks),
        "dealt_card_ids": np.ascontiguousarray(dealt_card_ids),
        "seat_ids": np.ascontiguousarray(seat_ids),
        "order_ids": np.ascontiguousarray(order_ids),
        "action_counts": np.ascontiguousarray(action_counts),
        "action_placement_card_ids": np.ascontiguousarray(placement_cards),
        "action_placement_row_ids": np.ascontiguousarray(placement_rows),
        "action_discard_card_ids": np.ascontiguousarray(discard_cards),
        "max_actions": max_actions,
        "max_placements": max_placements,
        "max_discards": max_discards,
    }


def rust_direct_available() -> bool:
    try:
        _load_library()
        return True
    except RuntimeError:
        return False


@contextmanager
def pinned_feature_encoder_library(
    path: str | Path,
    *,
    expected_sha256: str,
) -> Iterator[Path]:
    """Temporarily bind the Rust encoder to one exact native library.

    This is intended for immutable scientific materialization outside the
    repository checkout.  The binding is context-local, nestable, and restored
    even when generation fails.  The file is revalidated when it is loaded.
    """

    target = _validate_pinned_library(path, expected_sha256=expected_sha256)
    token = _PINNED_LIBRARY.set((target, expected_sha256))
    try:
        yield target
    finally:
        _PINNED_LIBRARY.reset(token)


def _load_library() -> ctypes.CDLL:
    path = _library_path()
    if path is None:
        raise RuntimeError(
            "Rust Stage3 feature encoder library is not built. Run `cargo build --release` first."
        )
    lib = ctypes.CDLL(str(path))
    binding = _PINNED_LIBRARY.get()
    if binding is not None:
        _validate_pinned_library(path, expected_sha256=binding[1])
    lib.ofc_stage3_encode.argtypes = [
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
    lib.ofc_stage3_encode.restype = ctypes.c_int
    return lib


def _library_path() -> Path | None:
    binding = _PINNED_LIBRARY.get()
    if binding is not None:
        return _validate_pinned_library(
            binding[0],
            expected_sha256=binding[1],
        )
    root = Path(__file__).resolve().parents[2]
    if sys.platform.startswith("win"):
        name = "ofc_stage3_feature_encoder.dll"
    elif sys.platform == "darwin":
        name = "libofc_stage3_feature_encoder.dylib"
    else:
        name = "libofc_stage3_feature_encoder.so"
    candidates = [
        root / "target" / "release" / name,
        root / "target" / "debug" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_pinned_library(
    path: str | Path,
    *,
    expected_sha256: str,
) -> Path:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("feature encoder pin must be a lowercase SHA-256")
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_file():
        raise ValueError("pinned feature encoder must be an absolute non-symlink file")
    target = target.resolve()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise PermissionError("pinned feature encoder SHA-256 changed")
    return target


def _ptr(array: np.ndarray, ctype: Any) -> Any:
    return np.ascontiguousarray(array).ctypes.data_as(ctypes.POINTER(ctype))


def _board_masks(board: Any) -> np.ndarray:
    return np.asarray(
        [
            _cards_mask(board.top),
            _cards_mask(board.middle),
            _cards_mask(board.bottom),
        ],
        dtype=np.uint64,
    )


def _cards_mask(cards: Sequence[str]) -> np.uint64:
    mask = 0
    for card in cards:
        mask |= 1 << CARD_INDEX[card]
    return np.uint64(mask)


def _debug_rows(
    X: np.ndarray,
    row_to_state_index: np.ndarray,
    row_to_action_index: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(min(limit, X.shape[0])):
        rows.append(
            {
                "state_index": int(row_to_state_index[row_index]),
                "action_index": int(row_to_action_index[row_index]),
                "row_sum": float(X[row_index].sum()),
            }
        )
    return rows
