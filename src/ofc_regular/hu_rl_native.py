"""Typed Python wrapper for the optional deterministic HU RL native batch.

The explicit-deck constructor remains a correctness oracle. Production replay
uses the Rust-owned paired seed constructor/reset, so neither a full deck nor
its unrealized tail is materialized in Python. A public step result intentionally
excludes the selected ``ActionKey`` because that key contains the acting
player's private discard.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Final, Sequence

from .action_key import ActionKey
from .hu_rl_contract import (
    MAX_LEGAL_ACTIONS,
    HuRlActorViewV1,
    LegalActionMappingV1,
    PublicPlacement,
)

try:
    import _ofc_hu_rl_engine as _native
except ImportError:  # pragma: no cover - exercised in installations without the optional wheel
    _native = None


MAX_BATCH_LANES: Final = 4096
MAX_BATCH_THREADS: Final = 64
MAX_PAIRED_HANDS_PER_BATCH: Final = MAX_BATCH_LANES // 2
MAX_PAIRED_SEED: Final = (1 << 63) - 1
LEGAL_ACTION_BATCH_SCHEMA: Final = "regular_ofc_hu_rl_legal_action_batch_v1"
BATCH_STEP_OUTCOME_SCHEMA: Final = "regular_ofc_hu_rl_batch_step_outcome_v1"
PACKED_SCHEMA: Final = "regular_ofc_hu_rl_packed_boundary_v1"
PACKED_ENDIANNESS: Final = "little"
PACKED_SCORING_IDENTITY: Final = b"regular_ofc_hu_standard_score_v1"
PACKED_ACTION_U64S: Final = 4
PACKED_ACTION_BYTES: Final = 32
PACKED_ACTION_COUNT_BYTES: Final = 1
PACKED_DIGEST_BYTES: Final = 32
PACKED_HISTORY_SLOTS: Final = 9
PACKED_HISTORY_RECORD_BYTES: Final = 32
PACKED_OBSERVATION_PREFIX_BYTES: Final = 104
PACKED_OBSERVATION_RECORD_BYTES: Final = 392
PACKED_STEP_RECORD_BYTES: Final = 48
_CARD_DOMAIN_MASK: Final = (1 << 52) - 1
_STREETS: Final = ("T0", "T1", "T2", "T3", "T4")
_SEATS: Final = ("first", "second")
_ZERO_HISTORY_RECORD: Final = bytes(PACKED_HISTORY_RECORD_BYTES)


def _readonly_view(payload: bytes) -> memoryview:
    return memoryview(payload).toreadonly()


def _optional_numpy_view(payload: bytes, *, dtype: str, shape: tuple[int, ...]):
    """Return a zero-copy NumPy view without making NumPy an import dependency."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - depends on caller environment
        raise HuRlNativeUnavailableError("NumPy is not installed") from error
    view = np.frombuffer(payload, dtype=dtype)
    view.shape = shape
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class PackedPublicHistoryEventV1:
    street: str
    acting_seat: str
    placement_masks: tuple[int, int, int]
    discard_count: int

    def to_public_placement(self) -> PublicPlacement:
        return PublicPlacement(
            street=self.street,  # type: ignore[arg-type]
            acting_seat=self.acting_seat,  # type: ignore[arg-type]
            top_placement_mask=self.placement_masks[0],
            middle_placement_mask=self.placement_masks[1],
            bottom_placement_mask=self.placement_masks[2],
            discard_count=self.discard_count,
        )


@dataclass(frozen=True)
class PackedActorObservationLaneV1:
    hero_board_masks: tuple[int, int, int]
    opponent_public_board_masks: tuple[int, int, int]
    hero_private_discards_mask: int
    dealt_cards_mask: int
    seat: str
    street: str
    to_act_order: str
    hero_in_fantasyland: bool
    opponent_in_fantasyland: bool
    opponent_discard_count: int
    scoring_identity: bytes
    public_history: tuple[PackedPublicHistoryEventV1, ...]


@dataclass(frozen=True)
class PackedActorObservationBatchV1:
    """Immutable lane-major actor observations; no simulator-hidden fields exist."""

    payload: bytes
    lane_count: int
    schema: str = PACKED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PACKED_SCHEMA:
            raise HuRlNativeContractError("unsupported packed observation schema")
        if type(self.payload) is not bytes:
            raise HuRlNativeContractError("packed observations must be immutable bytes")
        if type(self.lane_count) is not int or not 1 <= self.lane_count <= MAX_BATCH_LANES:
            raise HuRlNativeContractError("packed observations have invalid lane count")
        if len(self.payload) != self.lane_count * PACKED_OBSERVATION_RECORD_BYTES:
            raise HuRlNativeContractError("packed observation geometry mismatch")
        for lane in range(self.lane_count):
            self._decode_lane(lane, materialize=False)

    @property
    def buffer(self) -> memoryview:
        return _readonly_view(self.payload)

    def numpy_u8_view(self):
        return _optional_numpy_view(
            self.payload,
            dtype="u1",
            shape=(self.lane_count, PACKED_OBSERVATION_RECORD_BYTES),
        )

    def lane(self, lane: int) -> PackedActorObservationLaneV1:
        if type(lane) is not int or not 0 <= lane < self.lane_count:
            raise IndexError("packed observation lane is out of range")
        decoded = self._decode_lane(lane, materialize=True)
        assert decoded is not None
        return decoded

    def _decode_lane(
        self, lane: int, *, materialize: bool
    ) -> PackedActorObservationLaneV1 | None:
        offset = lane * PACKED_OBSERVATION_RECORD_BYTES
        record = self.payload[offset : offset + PACKED_OBSERVATION_RECORD_BYTES]
        masks = struct.unpack_from("<8Q", record, 0)
        if any(mask & ~_CARD_DOMAIN_MASK for mask in masks):
            raise HuRlNativeContractError("packed observation mask is outside card domain")
        if _masks_overlap(masks):
            raise HuRlNativeContractError("packed observation visible-card masks overlap")
        if any(
            mask.bit_count() > capacity
            for mask, capacity in zip(masks[:6], (3, 5, 5, 3, 5, 5), strict=True)
        ):
            raise HuRlNativeContractError("packed observation row exceeds capacity")
        seat_code, street_code, order_code = record[64], record[65], record[66]
        if seat_code not in (0, 1) or street_code >= len(_STREETS) or order_code not in (0, 1):
            raise HuRlNativeContractError("packed observation enum is invalid")
        if seat_code != order_code:
            raise HuRlNativeContractError("packed observation seat/order disagree")
        if record[67] != 0 or record[68] != 0:
            raise HuRlNativeContractError("packed normal-hand observation contains FL state")
        if record[71] != 0:
            raise HuRlNativeContractError("packed observation reserved byte is nonzero")
        scoring_identity = record[72:PACKED_OBSERVATION_PREFIX_BYTES]
        if scoring_identity != PACKED_SCORING_IDENTITY:
            raise HuRlNativeContractError("packed scoring identity mismatch")
        history_count = record[70]
        expected_history_count = street_code * 2 + seat_code
        if history_count != expected_history_count or history_count > PACKED_HISTORY_SLOTS:
            raise HuRlNativeContractError("packed public history length is invalid")

        history: list[PackedPublicHistoryEventV1] | None = [] if materialize else None
        accumulated = [[0, 0, 0] for _ in range(2)]
        seen = 0
        for event_index in range(PACKED_HISTORY_SLOTS):
            event_offset = (
                PACKED_OBSERVATION_PREFIX_BYTES
                + event_index * PACKED_HISTORY_RECORD_BYTES
            )
            event_record = record[
                event_offset : event_offset + PACKED_HISTORY_RECORD_BYTES
            ]
            if event_index >= history_count:
                if event_record != _ZERO_HISTORY_RECORD:
                    raise HuRlNativeContractError("packed public history padding is nonzero")
                continue
            placement_masks = struct.unpack_from("<3Q", event_record, 0)
            event_street, event_seat = event_record[24], event_record[25]
            discard_count, present = event_record[26], event_record[27]
            if event_record[28:32] != b"\x00\x00\x00\x00" or present != 1:
                raise HuRlNativeContractError("packed public history metadata is invalid")
            if (event_street, event_seat) != (event_index // 2, event_index % 2):
                raise HuRlNativeContractError("packed public history order is invalid")
            expected_discards = 0 if event_street == 0 else 1
            expected_placements = 5 if event_street == 0 else 2
            if (
                discard_count != expected_discards
                or any(mask & ~_CARD_DOMAIN_MASK for mask in placement_masks)
                or _masks_overlap(placement_masks)
                or sum(mask.bit_count() for mask in placement_masks) != expected_placements
                or seen & _mask_union(placement_masks)
            ):
                raise HuRlNativeContractError("packed public history placement is invalid")
            seen |= _mask_union(placement_masks)
            for row, mask in enumerate(placement_masks):
                accumulated[event_seat][row] |= mask
            if history is not None:
                history.append(
                    PackedPublicHistoryEventV1(
                        street=_STREETS[event_street],
                        acting_seat=_SEATS[event_seat],
                        placement_masks=placement_masks,
                        discard_count=discard_count,
                    )
                )

        hero_masks = masks[:3]
        opponent_masks = masks[3:6]
        if tuple(accumulated[seat_code]) != hero_masks or tuple(
            accumulated[1 - seat_code]
        ) != opponent_masks:
            raise HuRlNativeContractError("packed public history does not reconstruct boards")
        expected_geometry = _regular_geometry(street_code, order_code)
        actual_geometry = (
            _mask_union(hero_masks).bit_count(),
            _mask_union(opponent_masks).bit_count(),
            masks[7].bit_count(),
            masks[6].bit_count(),
        )
        if actual_geometry != expected_geometry:
            raise HuRlNativeContractError("packed observation decision geometry mismatch")
        expected_opponent_discards = max(actual_geometry[1] - 5, 0) // 2
        if record[69] != min(expected_opponent_discards, 4):
            raise HuRlNativeContractError("packed opponent discard count mismatch")
        if history is None:
            return None
        return PackedActorObservationLaneV1(
            hero_board_masks=hero_masks,
            opponent_public_board_masks=opponent_masks,
            hero_private_discards_mask=masks[6],
            dealt_cards_mask=masks[7],
            seat=_SEATS[seat_code],
            street=_STREETS[street_code],
            to_act_order=_SEATS[order_code],
            hero_in_fantasyland=False,
            opponent_in_fantasyland=False,
            opponent_discard_count=record[69],
            scoring_identity=scoring_identity,
            public_history=tuple(history),
        )


@dataclass(frozen=True)
class PackedLegalActionBatchV1:
    """Immutable `[lanes,232,4]` u64 ActionKeys and mapping proofs."""

    action_keys: bytes
    mask: bytes
    action_counts: bytes
    action_set_digests: bytes
    action_order_digests: bytes
    lane_count: int
    schema: str = PACKED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PACKED_SCHEMA:
            raise HuRlNativeContractError("unsupported packed legal schema")
        fields = (
            self.action_keys,
            self.mask,
            self.action_counts,
            self.action_set_digests,
            self.action_order_digests,
        )
        if any(type(field) is not bytes for field in fields):
            raise HuRlNativeContractError("packed legal fields must be immutable bytes")
        if type(self.lane_count) is not int or not 1 <= self.lane_count <= MAX_BATCH_LANES:
            raise HuRlNativeContractError("packed legal actions have invalid lane count")
        expected = (
            self.lane_count * MAX_LEGAL_ACTIONS * PACKED_ACTION_BYTES,
            self.lane_count * MAX_LEGAL_ACTIONS,
            self.lane_count * PACKED_ACTION_COUNT_BYTES,
            self.lane_count * PACKED_DIGEST_BYTES,
            self.lane_count * PACKED_DIGEST_BYTES,
        )
        if tuple(map(len, fields)) != expected:
            raise HuRlNativeContractError("packed legal action geometry mismatch")
        key_lane_bytes = MAX_LEGAL_ACTIONS * PACKED_ACTION_BYTES
        for lane in range(self.lane_count):
            count = self.action_count(lane)
            if not 1 <= count <= MAX_LEGAL_ACTIONS:
                raise HuRlNativeContractError("packed legal action count is invalid")
            mask_offset = lane * MAX_LEGAL_ACTIONS
            mask_end = mask_offset + MAX_LEGAL_ACTIONS
            if (
                self.mask.count(1, mask_offset, mask_offset + count) != count
                or self.mask.count(0, mask_offset + count, mask_end)
                != MAX_LEGAL_ACTIONS - count
            ):
                raise HuRlNativeContractError("packed legal mask/count disagree")
            padding_start = lane * key_lane_bytes + count * PACKED_ACTION_BYTES
            padding_end = (lane + 1) * key_lane_bytes
            if self.action_keys.count(0, padding_start, padding_end) != (
                padding_end - padding_start
            ):
                raise HuRlNativeContractError("packed ActionKey padding is nonzero")

    @property
    def action_keys_buffer(self) -> memoryview:
        return _readonly_view(self.action_keys)

    @property
    def mask_buffer(self) -> memoryview:
        return _readonly_view(self.mask)

    def numpy_views(self) -> dict[str, Any]:
        return {
            "action_keys": _optional_numpy_view(
                self.action_keys,
                dtype="<u8",
                shape=(self.lane_count, MAX_LEGAL_ACTIONS, PACKED_ACTION_U64S),
            ),
            "mask": _optional_numpy_view(
                self.mask, dtype="u1", shape=(self.lane_count, MAX_LEGAL_ACTIONS)
            ),
            "counts": _optional_numpy_view(
                self.action_counts, dtype="u1", shape=(self.lane_count,)
            ),
        }

    def action_count(self, lane: int) -> int:
        _require_lane(lane, self.lane_count)
        return self.action_counts[lane]

    def action_key_at(self, lane: int, index: int) -> ActionKey:
        _require_lane(lane, self.lane_count)
        if type(index) is not int or not 0 <= index < self.action_count(lane):
            raise IndexError("packed legal action index is out of range")
        offset = (lane * MAX_LEGAL_ACTIONS + index) * PACKED_ACTION_BYTES
        return ActionKey(*struct.unpack_from("<4Q", self.action_keys, offset))

    def action_set_digest(self, lane: int) -> str:
        _require_lane(lane, self.lane_count)
        offset = lane * PACKED_DIGEST_BYTES
        return self.action_set_digests[offset : offset + PACKED_DIGEST_BYTES].hex()

    def action_order_digest(self, lane: int) -> str:
        _require_lane(lane, self.lane_count)
        offset = lane * PACKED_DIGEST_BYTES
        return self.action_order_digests[offset : offset + PACKED_DIGEST_BYTES].hex()

    def select(self, indices: Sequence[int]) -> bytes:
        if len(indices) != self.lane_count:
            raise HuRlNativeContractError("packed action selection lane count mismatch")
        selected = bytearray(self.lane_count * PACKED_ACTION_BYTES)
        for lane, index in enumerate(indices):
            if type(index) is not int or not 0 <= index < self.action_count(lane):
                raise IndexError("packed legal action index is out of range")
            source = (lane * MAX_LEGAL_ACTIONS + index) * PACKED_ACTION_BYTES
            masks = struct.unpack_from("<4Q", self.action_keys, source)
            _validate_action_key_masks(masks)
            target = lane * PACKED_ACTION_BYTES
            selected[target : target + PACKED_ACTION_BYTES] = self.action_keys[
                source : source + PACKED_ACTION_BYTES
            ]
        return bytes(selected)


@dataclass(frozen=True)
class PackedActorDecisionBatchV1:
    """One observation traversal encoded as actor views plus legal mappings."""

    observations: PackedActorObservationBatchV1
    legal_actions: PackedLegalActionBatchV1
    schema: str = PACKED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PACKED_SCHEMA:
            raise HuRlNativeContractError("unsupported packed actor-decision schema")
        if type(self.observations) is not PackedActorObservationBatchV1:
            raise HuRlNativeContractError(
                "packed actor-decision observations have invalid type"
            )
        if type(self.legal_actions) is not PackedLegalActionBatchV1:
            raise HuRlNativeContractError(
                "packed actor-decision legal actions have invalid type"
            )
        if self.observations.lane_count != self.legal_actions.lane_count:
            raise HuRlNativeContractError(
                "packed actor-decision lane geometry disagrees"
            )

    @property
    def lane_count(self) -> int:
        return self.observations.lane_count


@dataclass(frozen=True)
class PackedPublicStepV1:
    actor: int
    street: str
    done: bool
    placement_masks: tuple[int, int, int]
    discard_count: int
    rewards: tuple[float, float]


@dataclass(frozen=True)
class PackedStepBatchV1:
    payload: bytes
    lane_count: int
    schema: str = PACKED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PACKED_SCHEMA:
            raise HuRlNativeContractError("unsupported packed step schema")
        if type(self.payload) is not bytes:
            raise HuRlNativeContractError("packed step output must be immutable bytes")
        if type(self.lane_count) is not int or not 1 <= self.lane_count <= MAX_BATCH_LANES:
            raise HuRlNativeContractError("packed step output has invalid lane count")
        if len(self.payload) != self.lane_count * PACKED_STEP_RECORD_BYTES:
            raise HuRlNativeContractError("packed step geometry mismatch")
        for lane in range(self.lane_count):
            self._decode_lane(lane, materialize=False)

    @property
    def buffer(self) -> memoryview:
        return _readonly_view(self.payload)

    def lane(self, lane: int) -> PackedPublicStepV1:
        _require_lane(lane, self.lane_count)
        decoded = self._decode_lane(lane, materialize=True)
        assert decoded is not None
        return decoded

    def _decode_lane(
        self, lane: int, *, materialize: bool
    ) -> PackedPublicStepV1 | None:
        offset = lane * PACKED_STEP_RECORD_BYTES
        record = self.payload[offset : offset + PACKED_STEP_RECORD_BYTES]
        actor, street, done, discard_count = record[:4]
        if (
            actor not in (0, 1)
            or street >= len(_STREETS)
            or done not in (0, 1)
            or record[4:8] != b"\x00\x00\x00\x00"
        ):
            raise HuRlNativeContractError("packed public step metadata is invalid")
        placement_masks = struct.unpack_from("<3Q", record, 8)
        rewards = struct.unpack_from("<2d", record, 32)
        expected_discards = 0 if street == 0 else 1
        expected_placements = 5 if street == 0 else 2
        if (
            discard_count != expected_discards
            or any(mask & ~_CARD_DOMAIN_MASK for mask in placement_masks)
            or _masks_overlap(placement_masks)
            or sum(mask.bit_count() for mask in placement_masks) != expected_placements
            or any(not math.isfinite(reward) for reward in rewards)
        ):
            raise HuRlNativeContractError("packed public step placement/reward is invalid")
        if not done and rewards != (0.0, 0.0):
            raise HuRlNativeContractError("packed nonterminal step has rewards")
        if done and (street, actor) != (4, 1):
            raise HuRlNativeContractError("packed terminal step identity disagrees")
        if done and rewards[0] != -rewards[1]:
            raise HuRlNativeContractError("packed terminal rewards are not zero-sum")
        if not materialize:
            return None
        return PackedPublicStepV1(
            actor=actor,
            street=_STREETS[street],
            done=bool(done),
            placement_masks=placement_masks,
            discard_count=discard_count,
            rewards=rewards,
        )


def _require_lane(lane: int, lane_count: int) -> None:
    if type(lane) is not int or not 0 <= lane < lane_count:
        raise IndexError("packed lane is out of range")


def _mask_union(masks: Sequence[int]) -> int:
    union = 0
    for mask in masks:
        union |= mask
    return union


def _masks_overlap(masks: Sequence[int]) -> bool:
    union = 0
    for mask in masks:
        if union & mask:
            return True
        union |= mask
    return False


def _validate_action_key_masks(masks: Sequence[int]) -> None:
    if any(mask & ~_CARD_DOMAIN_MASK for mask in masks) or _masks_overlap(masks):
        raise HuRlNativeContractError("packed ActionKey is invalid")


def _regular_geometry(street: int, order: int) -> tuple[int, int, int, int]:
    return (
        ((0, 0, 5, 0), (0, 5, 5, 0)),
        ((5, 5, 3, 0), (5, 7, 3, 0)),
        ((7, 7, 3, 1), (7, 9, 3, 1)),
        ((9, 9, 3, 2), (9, 11, 3, 2)),
        ((11, 11, 3, 3), (11, 13, 3, 3)),
    )[street][order]


class HuRlNativeUnavailableError(RuntimeError):
    """The optional PyO3 extension is not installed in this interpreter."""


class HuRlNativeContractError(ValueError):
    """Native output failed the actor-safe Python boundary contract."""


@dataclass(frozen=True)
class LegalActionBatchV1:
    """Lane-ordered fixed-width legal ActionKeys and masks."""

    action_keys: tuple[tuple[str | None, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    action_counts: tuple[int, ...]
    action_set_digests: tuple[str, ...]
    action_order_digests: tuple[str, ...]
    schema: str = LEGAL_ACTION_BATCH_SCHEMA

    def __post_init__(self) -> None:
        lane_count = len(self.action_keys)
        if not 1 <= lane_count <= MAX_BATCH_LANES:
            raise HuRlNativeContractError("legal action batch has invalid lane count")
        parallel = (
            self.mask,
            self.action_counts,
            self.action_set_digests,
            self.action_order_digests,
        )
        if any(len(field) != lane_count for field in parallel):
            raise HuRlNativeContractError("legal action batch lane fields disagree")
        for lane, (keys, mask, count) in enumerate(
            zip(self.action_keys, self.mask, self.action_counts, strict=True)
        ):
            if len(keys) != MAX_LEGAL_ACTIONS or len(mask) != MAX_LEGAL_ACTIONS:
                raise HuRlNativeContractError(
                    f"legal action lane {lane} is not fixed width {MAX_LEGAL_ACTIONS}"
                )
            if type(count) is not int or not 1 <= count <= MAX_LEGAL_ACTIONS:
                raise HuRlNativeContractError(f"legal action lane {lane} has invalid count")
            if any(type(value) is not bool for value in mask):
                raise HuRlNativeContractError(f"legal action lane {lane} mask is not boolean")
            if mask != (True,) * count + (False,) * (MAX_LEGAL_ACTIONS - count):
                raise HuRlNativeContractError(f"legal action lane {lane} mask/count disagree")
            if any(not isinstance(token, str) for token in keys[:count]):
                raise HuRlNativeContractError(f"legal action lane {lane} has invalid key")
            if any(token is not None for token in keys[count:]):
                raise HuRlNativeContractError(f"legal action lane {lane} padding is not empty")
            try:
                parsed = tuple(ActionKey.from_token(token) for token in keys[:count])
                mapping = LegalActionMappingV1(parsed)
            except (TypeError, ValueError) as error:
                raise HuRlNativeContractError(
                    f"legal action lane {lane} has an invalid semantic mapping"
                ) from error
            if (
                mapping.action_count != count
                or mapping.action_set_digest != self.action_set_digests[lane]
                or mapping.action_order_digest != self.action_order_digests[lane]
            ):
                raise HuRlNativeContractError(
                    f"legal action lane {lane} mapping digest disagrees"
                )

    @property
    def lane_count(self) -> int:
        return len(self.action_keys)


class NativeBatchHuRlEnvV1:
    """Actor-safe wrapper around the fixed-width native batch coordinator."""

    def __init__(
        self,
        explicit_decks: Sequence[Sequence[str]],
        *,
        chunk_width: int = 64,
        thread_count: int = 1,
    ) -> None:
        native = _require_native()
        decks = _normalize_decks(explicit_decks)
        self._native = native.BatchHuRlEnv(
            decks,
            chunk_width=chunk_width,
            thread_count=thread_count,
        )

    @classmethod
    def from_paired_seed_range(
        cls,
        *,
        seed_base: int,
        global_pair_start: int,
        pair_count: int,
        seed_stride: int,
        chunk_width: int = 64,
        thread_count: int = 1,
    ) -> "NativeBatchHuRlEnvV1":
        """Create `[AB, BA]` pairs without constructing any deck in Python."""

        native = _require_native()
        _validate_paired_seed_range(
            seed_base=seed_base,
            global_pair_start=global_pair_start,
            pair_count=pair_count,
            seed_stride=seed_stride,
        )
        instance = cls.__new__(cls)
        instance._native = native.BatchHuRlEnv.from_paired_seed_range(
            seed_base,
            global_pair_start,
            pair_count,
            seed_stride,
            chunk_width=chunk_width,
            thread_count=thread_count,
        )
        return instance

    @property
    def lane_count(self) -> int:
        return int(self._native.lane_count)

    @property
    def decision_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._native.decision_counts)

    @property
    def all_done(self) -> bool:
        return bool(self._native.all_done)

    def reset_batch(self, explicit_decks: Sequence[Sequence[str]]) -> tuple[dict[str, Any], ...]:
        decks = _normalize_decks(explicit_decks)
        if len(decks) != self.lane_count:
            raise HuRlNativeContractError("reset_batch lane count is invalid")
        payloads = self._native.reset_batch(decks)
        return _decode_actor_views(payloads, operation="reset_batch")

    def reset_from_paired_seed_range(
        self,
        *,
        seed_base: int,
        global_pair_start: int,
        pair_count: int,
        seed_stride: int,
    ) -> tuple[dict[str, Any], ...]:
        """Atomically reset fixed-width lanes from a Rust-owned paired range."""

        _validate_paired_seed_range(
            seed_base=seed_base,
            global_pair_start=global_pair_start,
            pair_count=pair_count,
            seed_stride=seed_stride,
        )
        if pair_count * 2 != self.lane_count:
            raise HuRlNativeContractError(
                "paired seed reset lane count is invalid"
            )
        payloads = self._native.reset_from_paired_seed_range(
            seed_base,
            global_pair_start,
            pair_count,
            seed_stride,
        )
        return _decode_actor_views(
            payloads,
            operation="reset_from_paired_seed_range",
        )

    def observe_batch(self) -> tuple[dict[str, Any], ...]:
        return _decode_actor_views(
            self._native.observe_batch(), operation="observe_batch"
        )

    def observe_batch_packed(self) -> PackedActorObservationBatchV1:
        """Return the opt-in fixed-width actor-safe observation boundary."""

        payload = self._native.observe_batch_packed()
        return PackedActorObservationBatchV1(payload=payload, lane_count=self.lane_count)

    def legal_actions_batch(self) -> LegalActionBatchV1:
        keys, mask, counts, set_digests, order_digests = self._native.legal_actions_batch()
        return LegalActionBatchV1(
            action_keys=tuple(tuple(row) for row in keys),
            mask=tuple(tuple(row) for row in mask),
            action_counts=tuple(counts),
            action_set_digests=tuple(set_digests),
            action_order_digests=tuple(order_digests),
        )

    def legal_actions_batch_packed(self) -> PackedLegalActionBatchV1:
        """Return `[lanes,232,4]` little-endian u64 ActionKeys without strings."""

        keys, mask, counts, set_digests, order_digests = (
            self._native.legal_actions_batch_packed()
        )
        return PackedLegalActionBatchV1(
            action_keys=keys,
            mask=mask,
            action_counts=counts,
            action_set_digests=set_digests,
            action_order_digests=order_digests,
            lane_count=self.lane_count,
        )

    def actor_decision_batch_packed(self) -> PackedActorDecisionBatchV1:
        """Encode actor observations and their legal mappings in one traversal."""

        observations, keys, mask, counts, set_digests, order_digests = (
            self._native.actor_decision_batch_packed()
        )
        return PackedActorDecisionBatchV1(
            observations=PackedActorObservationBatchV1(
                payload=observations,
                lane_count=self.lane_count,
            ),
            legal_actions=PackedLegalActionBatchV1(
                action_keys=keys,
                mask=mask,
                action_counts=counts,
                action_set_digests=set_digests,
                action_order_digests=order_digests,
                lane_count=self.lane_count,
            ),
        )

    def step_batch(self, selected_action_keys: Sequence[str]) -> tuple[dict[str, Any], ...]:
        if isinstance(selected_action_keys, (str, bytes)):
            raise TypeError("selected_action_keys must be a sequence of ActionKey tokens")
        selected = list(selected_action_keys)
        if len(selected) != self.lane_count or any(type(token) is not str for token in selected):
            raise HuRlNativeContractError("step_batch action shape/type is invalid")
        results = _decode_json_documents(
            self._native.step_batch(selected), operation="step_batch"
        )
        for result in results:
            _validate_public_step_result(result)
        return results

    def step_batch_packed(
        self, selected_action_keys: bytes | bytearray | memoryview
    ) -> PackedStepBatchV1:
        """Atomically step one exact 4-u64 packed ActionKey per lane."""

        if isinstance(selected_action_keys, str):
            raise TypeError("packed selected actions require a bytes-like buffer")
        try:
            view = memoryview(selected_action_keys)
        except TypeError as error:
            raise TypeError("packed selected actions require a bytes-like buffer") from error
        if not view.contiguous:
            raise HuRlNativeContractError("packed selected actions must be contiguous")
        try:
            byte_view = view.cast("B")
        except (TypeError, ValueError) as error:
            raise HuRlNativeContractError(
                "packed selected actions must have byte-compatible geometry"
            ) from error
        expected = self.lane_count * PACKED_ACTION_BYTES
        if byte_view.nbytes != expected:
            raise HuRlNativeContractError("packed selected action geometry mismatch")
        payload_in = (
            selected_action_keys
            if type(selected_action_keys) is bytes
            else bytes(byte_view)
        )
        payload = self._native.step_batch_packed(payload_in)
        return PackedStepBatchV1(payload=payload, lane_count=self.lane_count)

    def snapshot_batch(self) -> object:
        """Return an opaque native checkpoint; it is intentionally not serializable."""

        return self._native.snapshot_batch()

    def restore_batch(self, snapshot: object) -> None:
        self._native.restore_batch(snapshot)


def native_available() -> bool:
    return _native is not None


def _require_native():
    if _native is None:
        raise HuRlNativeUnavailableError(
            "the optional _ofc_hu_rl_engine extension is not installed"
        )
    if int(_native.MAX_BATCH_LANES) != MAX_BATCH_LANES:
        raise HuRlNativeContractError("native MAX_BATCH_LANES contract mismatch")
    if int(_native.MAX_LEGAL_ACTIONS) != MAX_LEGAL_ACTIONS:
        raise HuRlNativeContractError("native MAX_LEGAL_ACTIONS contract mismatch")
    if int(_native.MAX_BATCH_THREADS) != MAX_BATCH_THREADS:
        raise HuRlNativeContractError("native MAX_BATCH_THREADS contract mismatch")
    if int(getattr(_native, "MAX_PAIRED_HANDS_PER_BATCH", -1)) != MAX_PAIRED_HANDS_PER_BATCH:
        raise HuRlNativeContractError("native paired batch capacity contract mismatch")
    if int(getattr(_native, "MAX_PAIRED_SEED", -1)) != MAX_PAIRED_SEED:
        raise HuRlNativeContractError("native paired seed domain contract mismatch")
    if _native.BATCH_STEP_OUTCOME_SCHEMA != BATCH_STEP_OUTCOME_SCHEMA:
        raise HuRlNativeContractError("native batch-step schema mismatch")
    packed_contract = {
        "PACKED_SCHEMA": PACKED_SCHEMA,
        "PACKED_ENDIANNESS": PACKED_ENDIANNESS,
        "PACKED_SCORING_IDENTITY": PACKED_SCORING_IDENTITY.decode("ascii"),
        "PACKED_ACTION_U64S": PACKED_ACTION_U64S,
        "PACKED_ACTION_BYTES": PACKED_ACTION_BYTES,
        "PACKED_ACTION_COUNT_BYTES": PACKED_ACTION_COUNT_BYTES,
        "PACKED_DIGEST_BYTES": PACKED_DIGEST_BYTES,
        "PACKED_HISTORY_SLOTS": PACKED_HISTORY_SLOTS,
        "PACKED_HISTORY_RECORD_BYTES": PACKED_HISTORY_RECORD_BYTES,
        "PACKED_OBSERVATION_PREFIX_BYTES": PACKED_OBSERVATION_PREFIX_BYTES,
        "PACKED_OBSERVATION_RECORD_BYTES": PACKED_OBSERVATION_RECORD_BYTES,
        "PACKED_STEP_RECORD_BYTES": PACKED_STEP_RECORD_BYTES,
    }
    for name, expected in packed_contract.items():
        if getattr(_native, name, None) != expected:
            raise HuRlNativeContractError(f"native {name} contract mismatch")
    return _native


def _validate_paired_seed_range(
    *,
    seed_base: object,
    global_pair_start: object,
    pair_count: object,
    seed_stride: object,
) -> None:
    values = (seed_base, global_pair_start, pair_count, seed_stride)
    if any(type(value) is not int for value in values):
        raise TypeError("paired seed range values must be integers")
    assert isinstance(seed_base, int)
    assert isinstance(global_pair_start, int)
    assert isinstance(pair_count, int)
    assert isinstance(seed_stride, int)
    if not 0 <= seed_base <= MAX_PAIRED_SEED:
        raise HuRlNativeContractError("paired seed base is invalid")
    if not 0 <= global_pair_start <= MAX_PAIRED_SEED:
        raise HuRlNativeContractError("global pair start is invalid")
    if not 1 <= pair_count <= MAX_PAIRED_HANDS_PER_BATCH:
        raise HuRlNativeContractError("paired seed pair count is invalid")
    if not 1 <= seed_stride <= MAX_PAIRED_SEED:
        raise HuRlNativeContractError("paired seed stride is invalid")
    final_pair_index = global_pair_start + pair_count - 1
    if (
        final_pair_index > MAX_PAIRED_SEED
        or seed_base + final_pair_index * seed_stride > MAX_PAIRED_SEED
    ):
        raise HuRlNativeContractError("paired seed range exceeds its fixed domain")


def _normalize_decks(explicit_decks: Sequence[Sequence[str]]) -> list[list[str]]:
    if isinstance(explicit_decks, (str, bytes)):
        raise TypeError("explicit_decks must be a sequence of decks")
    lane_count = len(explicit_decks)
    if not 1 <= lane_count <= MAX_BATCH_LANES:
        raise HuRlNativeContractError("explicit_decks lane count is invalid")
    normalized: list[list[str]] = []
    for deck in explicit_decks:
        if isinstance(deck, (str, bytes)) or len(deck) != 52:
            raise HuRlNativeContractError("every explicit deck must contain exactly 52 cards")
        cards = list(deck)
        if any(type(card) is not str for card in cards):
            raise HuRlNativeContractError("explicit deck cards must be strings")
        normalized.append(cards)
    return normalized


def _decode_json_documents(payloads: Sequence[str], *, operation: str) -> tuple[dict[str, Any], ...]:
    documents: list[dict[str, Any]] = []
    for payload in payloads:
        if type(payload) is not str:
            raise HuRlNativeContractError(f"{operation} returned a non-string document")
        try:
            document = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise HuRlNativeContractError(f"{operation} returned invalid JSON") from error
        if not isinstance(document, dict):
            raise HuRlNativeContractError(f"{operation} returned a non-object document")
        documents.append(document)
    return tuple(documents)


def _decode_actor_views(
    payloads: Sequence[str], *, operation: str
) -> tuple[dict[str, Any], ...]:
    documents = _decode_json_documents(payloads, operation=operation)
    for document in documents:
        try:
            view = HuRlActorViewV1.from_dict(document)
        except (TypeError, ValueError) as error:
            raise HuRlNativeContractError(
                f"{operation} returned an invalid actor view"
            ) from error
        if view.to_dict() != document:
            raise HuRlNativeContractError(
                f"{operation} returned a non-canonical actor view"
            )
    return documents


def _validate_public_step_result(result: dict[str, Any]) -> None:
    expected = {"schema", "actor", "street", "public_placement", "done", "rewards"}
    if set(result) != expected or result.get("schema") != BATCH_STEP_OUTCOME_SCHEMA:
        raise HuRlNativeContractError("step_batch returned an unsupported public result")
    placement = result.get("public_placement")
    if not isinstance(placement, dict):
        raise HuRlNativeContractError("step_batch returned an invalid public placement")
    forbidden = {"action_key", "discard_card", "discard_cards", "discard_mask", "world_state"}
    if forbidden.intersection(result) or forbidden.intersection(placement):
        raise HuRlNativeContractError("step_batch exposed private or audit state")
    actor = result.get("actor")
    street = result.get("street")
    done = result.get("done")
    rewards = result.get("rewards")
    if type(actor) is not int or actor not in (0, 1):
        raise HuRlNativeContractError("step_batch returned an invalid actor")
    if street not in ("T0", "T1", "T2", "T3", "T4"):
        raise HuRlNativeContractError("step_batch returned an invalid street")
    if type(done) is not bool:
        raise HuRlNativeContractError("step_batch returned an invalid done flag")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or any(type(value) not in (int, float) for value in rewards)
        or any(not math.isfinite(float(value)) for value in rewards)
    ):
        raise HuRlNativeContractError("step_batch returned invalid rewards")
    numeric_rewards = (float(rewards[0]), float(rewards[1]))
    if not done and numeric_rewards != (0.0, 0.0):
        raise HuRlNativeContractError("step_batch returned nonterminal rewards")
    if done and (street, actor) != ("T4", 1):
        raise HuRlNativeContractError("step_batch terminal identity disagrees")
    if done and numeric_rewards[0] != -numeric_rewards[1]:
        raise HuRlNativeContractError("step_batch terminal rewards are not zero-sum")
    try:
        parsed = PublicPlacement.from_dict(placement)
    except (TypeError, ValueError) as error:
        raise HuRlNativeContractError(
            "step_batch returned an invalid public placement"
        ) from error
    expected_seat = "first" if actor == 0 else "second"
    if (
        parsed.to_dict() != placement
        or parsed.street != street
        or parsed.acting_seat != expected_seat
    ):
        raise HuRlNativeContractError(
            "step_batch public placement identity disagrees"
        )
