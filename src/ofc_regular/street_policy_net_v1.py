"""Information-set-safe policy/value network foundation for regular HU OFC.

This module is deliberately independent of the M3.1 runtime selector.  It
defines the stable feature/action contract, a shared-seat PyTorch network, the
loss contract, and deterministic checkpoint I/O.  Importing the module does
not require PyTorch; callers pass/import PyTorch only when constructing a
network or tensors.

Only :class:`~ofc_regular.hu_infoset.ActorObservation` is accepted at the
policy boundary.  In particular, opponent private discards and a realized deck
tail can never enter this encoder.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .action_key import ACTION_KEY_SCHEMA, ActionKey, action_key
from .action_space import generate_actions
from .cards import ALL_CARDS
from .hu_infoset import ActorObservation, OBSERVATION_SCHEMA
from .state import ROW_CAPACITY, ROWS


STREET_POLICY_NET_V1_SCHEMA = "regular_ofc_street_policy_net_v1"
STREET_POLICY_FEATURE_SCHEMA = "regular_ofc_street_policy_features_v1"
STREET_POLICY_CHECKPOINT_SCHEMA = "regular_ofc_street_policy_checkpoint_v1"
STREET_POLICY_LOSS_SCHEMA = "regular_ofc_street_policy_loss_v1"

MAX_LEGAL_ACTIONS = 232
MAX_STATE_CARD_TOKENS = 52
MAX_ACTION_CARD_TOKENS = 5
PAD_CARD_ID = 0
PAD_ZONE_ID = 0

SEATS = ("first", "second")
# The 14-card FL transition has different action geometry and is intentionally
# kept out of this standard-street V1 contract.  It will receive an explicit
# transition encoder at the FL integration milestone.
STREETS = ("T0", "T1", "T2", "T3", "T4")

STATE_ZONES = (
    "hero_top",
    "hero_middle",
    "hero_bottom",
    "opponent_top",
    "opponent_middle",
    "opponent_bottom",
    "hero_private_discard",
    "dealt",
)
ACTION_ZONES = ("action_top", "action_middle", "action_bottom", "action_discard")
ALL_ZONES = (*STATE_ZONES, *ACTION_ZONES)

SCALAR_CONTEXT_NAMES = (
    "hero_top_fill",
    "hero_middle_fill",
    "hero_bottom_fill",
    "opponent_top_fill",
    "opponent_middle_fill",
    "opponent_bottom_fill",
    "dealt_fraction",
    "hero_discard_fraction",
    "opponent_discard_fraction",
    "known_card_fraction",
    "unknown_card_fraction",
    "exchangeable_remaining_deck_marginal",
    "hero_in_fantasyland",
    "opponent_in_fantasyland",
    "fl_ev_14_scaled",
    "middle_trips_royalty_scaled",
    "hu_line_points",
    "scoop_bonus_scaled",
    "foul_enabled",
    "fantasyland_cards_scaled",
)
MODEL_INPUT_FIELDS = (
    "state_card_ids",
    "state_zone_ids",
    "state_card_mask",
    "action_card_ids",
    "action_zone_ids",
    "action_card_mask",
    "legal_action_mask",
    "seat_ids",
    "street_ids",
    "scalar_context",
    "baseline_indices",
)

LOSS_WEIGHTS = {
    "action_q_huber": 1.0,
    "baseline_delta_huber": 1.0,
    "teacher_policy_kl": 0.25,
    "state_value_huber": 0.25,
    "ranking": 0.5,
    "uncertainty_quantile": 0.2,
    "safe_bce": 0.5,
}

_CARD_TO_ID = {card: index + 1 for index, card in enumerate(ALL_CARDS)}
_ID_TO_CARD = {index + 1: card for index, card in enumerate(ALL_CARDS)}
_ZONE_TO_ID = {zone: index + 1 for index, zone in enumerate(ALL_ZONES)}
_SEAT_TO_ID = {seat: index for index, seat in enumerate(SEATS)}
_STREET_TO_ID = {street: index for index, street in enumerate(STREETS)}
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def feature_schema_payload() -> dict[str, Any]:
    """Return the complete, serialization-stable feature contract."""

    return {
        "schema": STREET_POLICY_FEATURE_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "max_legal_actions": MAX_LEGAL_ACTIONS,
        "max_state_card_tokens": MAX_STATE_CARD_TOKENS,
        "max_action_card_tokens": MAX_ACTION_CARD_TOKENS,
        "card_vocabulary": list(ALL_CARDS),
        "state_zones": list(STATE_ZONES),
        "action_zones": list(ACTION_ZONES),
        "seats": list(SEATS),
        "streets": list(STREETS),
        "scalar_context_names": list(SCALAR_CONTEXT_NAMES),
        "model_input_fields": list(MODEL_INPUT_FIELDS),
        "pooling": {
            "state_cards": "masked_mean",
            "action_cards": "masked_mean",
        },
        "action_mapping": {
            "order": "ActionKey.sort_key",
            "padding": "right",
            "baseline_identity": "ActionKey",
            "baseline_delta_anchor": "exact_zero",
            "complete_legal_set_default": True,
        },
        "belief": {
            "source": "actor_observation_only",
            "type": "exchangeable_hidden_discard_prior",
            "forbidden": [
                "opponent_private_discards",
                "realized_deck_tail",
                "world_state",
                "replay_truth",
            ],
        },
    }


FEATURE_SCHEMA_HASH = _sha256_bytes(_canonical_json_bytes(feature_schema_payload()))
LOSS_SCHEMA_HASH = _sha256_bytes(
    _canonical_json_bytes(
        {
            "schema": STREET_POLICY_LOSS_SCHEMA,
            "weights": LOSS_WEIGHTS,
            "uncertainty_quantile": 0.95,
            "ranking": "pairwise_logistic_non_ties",
            "baseline_delta_target": "zero_at_baseline_ActionKey",
            "split_updates": {
                "train": "core",
                "safety-fit": "risk",
                "threshold-lock": "none",
                "diagnostic-holdout": "none",
            },
        }
    )
)


@dataclass(frozen=True)
class StreetPolicyNetV1Config:
    """Architecture parameters included in every checkpoint identity."""

    card_embedding_dim: int = 32
    zone_embedding_dim: int = 12
    token_hidden_dim: int = 64
    context_hidden_dim: int = 32
    seat_embedding_dim: int = 8
    street_embedding_dim: int = 8
    state_hidden_dim: int = 128
    action_hidden_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.card_embedding_dim,
            self.zone_embedding_dim,
            self.token_hidden_dim,
            self.context_hidden_dim,
            self.seat_embedding_dim,
            self.street_embedding_dim,
            self.state_hidden_dim,
            self.action_hidden_dim,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive
        ):
            raise ValueError("StreetPolicyNetV1 dimensions must be positive integers")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def schema_hash(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "schema": STREET_POLICY_NET_V1_SCHEMA,
                    "feature_schema_hash": FEATURE_SCHEMA_HASH,
                    "config": self.to_dict(),
                }
            )
        )


@dataclass(frozen=True)
class StreetPolicyEncodedBatch:
    """Dense numpy representation plus the exact ActionKey/index mapping."""

    state_card_ids: np.ndarray
    state_zone_ids: np.ndarray
    state_card_mask: np.ndarray
    action_card_ids: np.ndarray
    action_zone_ids: np.ndarray
    action_card_mask: np.ndarray
    legal_action_mask: np.ndarray
    seat_ids: np.ndarray
    street_ids: np.ndarray
    scalar_context: np.ndarray
    baseline_indices: np.ndarray
    action_key_tokens: tuple[tuple[str | None, ...], ...]
    feature_schema_hash: str = FEATURE_SCHEMA_HASH

    def __post_init__(self) -> None:
        batch = int(self.state_card_ids.shape[0])
        expected_shapes = {
            "state_card_ids": (batch, MAX_STATE_CARD_TOKENS),
            "state_zone_ids": (batch, MAX_STATE_CARD_TOKENS),
            "state_card_mask": (batch, MAX_STATE_CARD_TOKENS),
            "action_card_ids": (
                batch,
                MAX_LEGAL_ACTIONS,
                MAX_ACTION_CARD_TOKENS,
            ),
            "action_zone_ids": (
                batch,
                MAX_LEGAL_ACTIONS,
                MAX_ACTION_CARD_TOKENS,
            ),
            "action_card_mask": (
                batch,
                MAX_LEGAL_ACTIONS,
                MAX_ACTION_CARD_TOKENS,
            ),
            "legal_action_mask": (batch, MAX_LEGAL_ACTIONS),
            "seat_ids": (batch,),
            "street_ids": (batch,),
            "scalar_context": (batch, len(SCALAR_CONTEXT_NAMES)),
            "baseline_indices": (batch,),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} shape must be {shape}")
        if len(self.action_key_tokens) != batch or any(
            len(row) != MAX_LEGAL_ACTIONS for row in self.action_key_tokens
        ):
            raise ValueError("ActionKey mapping shape mismatch")
        if self.feature_schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError("feature schema hash mismatch")
        legal_counts = self.legal_action_mask.sum(axis=1)
        if np.any(legal_counts <= 0) or np.any(legal_counts > MAX_LEGAL_ACTIONS):
            raise ValueError("every encoded state requires 1..232 legal actions")
        for row, baseline in enumerate(self.baseline_indices.tolist()):
            if baseline < 0 or baseline >= int(legal_counts[row]):
                raise ValueError("baseline index must identify a legal ActionKey")

    @property
    def batch_size(self) -> int:
        return int(self.state_card_ids.shape[0])

    def to_torch(self, torch: Any, *, device: str | Any = "cpu") -> dict[str, Any]:
        """Copy the numeric contract to a PyTorch device."""

        return {
            "state_card_ids": torch.as_tensor(
                self.state_card_ids, dtype=torch.long, device=device
            ),
            "state_zone_ids": torch.as_tensor(
                self.state_zone_ids, dtype=torch.long, device=device
            ),
            "state_card_mask": torch.as_tensor(
                self.state_card_mask, dtype=torch.bool, device=device
            ),
            "action_card_ids": torch.as_tensor(
                self.action_card_ids, dtype=torch.long, device=device
            ),
            "action_zone_ids": torch.as_tensor(
                self.action_zone_ids, dtype=torch.long, device=device
            ),
            "action_card_mask": torch.as_tensor(
                self.action_card_mask, dtype=torch.bool, device=device
            ),
            "legal_action_mask": torch.as_tensor(
                self.legal_action_mask, dtype=torch.bool, device=device
            ),
            "seat_ids": torch.as_tensor(
                self.seat_ids, dtype=torch.long, device=device
            ),
            "street_ids": torch.as_tensor(
                self.street_ids, dtype=torch.long, device=device
            ),
            "scalar_context": torch.as_tensor(
                self.scalar_context, dtype=torch.float32, device=device
            ),
            "baseline_indices": torch.as_tensor(
                self.baseline_indices, dtype=torch.long, device=device
            ),
        }


def encode_street_policy_batch(
    observations: Sequence[ActorObservation | Mapping[str, Any]],
    legal_action_keys: Sequence[Sequence[ActionKey | str]],
    baseline_action_keys: Sequence[ActionKey | str],
    *,
    require_complete_legal_set: bool = True,
) -> StreetPolicyEncodedBatch:
    """Encode a batch in canonical ActionKey order.

    Mapping observations are parsed through ``ActorObservation.from_dict`` so
    an unknown or hidden-truth field fails closed.
    """

    if not observations:
        raise ValueError("StreetPolicyNetV1 batch cannot be empty")
    if not (
        len(observations) == len(legal_action_keys) == len(baseline_action_keys)
    ):
        raise ValueError("observation/action/baseline batch length mismatch")

    batch = len(observations)
    state_card_ids = np.zeros((batch, MAX_STATE_CARD_TOKENS), dtype=np.int64)
    state_zone_ids = np.zeros((batch, MAX_STATE_CARD_TOKENS), dtype=np.int64)
    state_card_mask = np.zeros((batch, MAX_STATE_CARD_TOKENS), dtype=np.bool_)
    action_card_ids = np.zeros(
        (batch, MAX_LEGAL_ACTIONS, MAX_ACTION_CARD_TOKENS), dtype=np.int64
    )
    action_zone_ids = np.zeros_like(action_card_ids)
    action_card_mask = np.zeros_like(action_card_ids, dtype=np.bool_)
    legal_mask = np.zeros((batch, MAX_LEGAL_ACTIONS), dtype=np.bool_)
    seat_ids = np.zeros(batch, dtype=np.int64)
    street_ids = np.zeros(batch, dtype=np.int64)
    scalar_context = np.zeros((batch, len(SCALAR_CONTEXT_NAMES)), dtype=np.float32)
    baseline_indices = np.full(batch, -1, dtype=np.int64)
    key_rows: list[tuple[str | None, ...]] = []

    for row_index, (raw_observation, raw_keys, raw_baseline) in enumerate(
        zip(
            observations,
            legal_action_keys,
            baseline_action_keys,
            strict=True,
        )
    ):
        observation = _safe_observation(raw_observation)
        if observation.street == "FL":
            raise ValueError(
                "StreetPolicyNetV1 covers T0-T4; FL requires its explicit "
                "14-card transition encoder"
            )
        state_tokens = _state_tokens(observation)
        for token_index, (card_id, zone_id) in enumerate(state_tokens):
            state_card_ids[row_index, token_index] = card_id
            state_zone_ids[row_index, token_index] = zone_id
            state_card_mask[row_index, token_index] = True

        keys = [_coerce_action_key(value) for value in raw_keys]
        if not keys or len(keys) > MAX_LEGAL_ACTIONS:
            raise ValueError("legal ActionKey count must be in 1..232")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate legal ActionKey")
        keys.sort(key=ActionKey.sort_key)
        for action_index, key in enumerate(keys):
            tokens = _action_tokens(observation, key)
            legal_mask[row_index, action_index] = True
            for token_index, (card_id, zone_id) in enumerate(tokens):
                action_card_ids[row_index, action_index, token_index] = card_id
                action_zone_ids[row_index, action_index, token_index] = zone_id
                action_card_mask[row_index, action_index, token_index] = True
        if require_complete_legal_set:
            expected = {
                action_key(action)
                for action in generate_actions(
                    observation.hero_board, observation.dealt_cards
                )
            }
            if set(keys) != expected:
                raise ValueError(
                    "ActionKey set is not the complete legal action set for "
                    "the observation"
                )

        baseline = _coerce_action_key(raw_baseline)
        try:
            baseline_indices[row_index] = keys.index(baseline)
        except ValueError as exc:
            raise ValueError("baseline ActionKey is not in the legal set") from exc
        tokens = [key.to_token() for key in keys]
        key_rows.append(tuple((*tokens, *(None for _ in range(MAX_LEGAL_ACTIONS - len(tokens))))))
        seat_ids[row_index] = _SEAT_TO_ID[observation.seat]
        street_ids[row_index] = _STREET_TO_ID[observation.street]
        scalar_context[row_index] = _scalar_context(observation)

    return StreetPolicyEncodedBatch(
        state_card_ids=state_card_ids,
        state_zone_ids=state_zone_ids,
        state_card_mask=state_card_mask,
        action_card_ids=action_card_ids,
        action_zone_ids=action_zone_ids,
        action_card_mask=action_card_mask,
        legal_action_mask=legal_mask,
        seat_ids=seat_ids,
        street_ids=street_ids,
        scalar_context=scalar_context,
        baseline_indices=baseline_indices,
        action_key_tokens=tuple(key_rows),
    )


def _safe_observation(
    value: ActorObservation | Mapping[str, Any],
) -> ActorObservation:
    if isinstance(value, ActorObservation):
        return value
    if isinstance(value, Mapping):
        return ActorObservation.from_dict(value)
    raise TypeError("policy encoder accepts only ActorObservation or its strict payload")


def _coerce_action_key(value: ActionKey | str) -> ActionKey:
    if isinstance(value, ActionKey):
        return value
    if isinstance(value, str):
        return ActionKey.from_token(value)
    raise TypeError("legal actions must be semantic ActionKey values")


def _state_tokens(observation: ActorObservation) -> list[tuple[int, int]]:
    zones: list[tuple[str, Iterable[str]]] = []
    for row in ROWS:
        zones.append((f"hero_{row}", getattr(observation.hero_board, row)))
    for row in ROWS:
        zones.append(
            (f"opponent_{row}", getattr(observation.opponent_public_board, row))
        )
    zones.extend(
        (
            ("hero_private_discard", observation.hero_private_discards),
            ("dealt", observation.dealt_cards),
        )
    )
    result: list[tuple[int, int]] = []
    for zone, cards in zones:
        # Sorting makes serialized features deterministic; masked mean pooling
        # independently provides invariance to the order within each zone.
        for card in sorted(cards, key=_CARD_TO_ID.__getitem__):
            result.append((_CARD_TO_ID[card], _ZONE_TO_ID[zone]))
    if len(result) > MAX_STATE_CARD_TOKENS:
        raise ValueError("observation contains more than 52 visible card tokens")
    return result


def _action_tokens(
    observation: ActorObservation, key: ActionKey
) -> list[tuple[int, int]]:
    dealt = set(observation.dealt_cards)
    groups = (
        ("top", key.top_mask),
        ("middle", key.middle_mask),
        ("bottom", key.bottom_mask),
        ("discard", key.discard_mask),
    )
    tokens: list[tuple[int, int]] = []
    group_counts: dict[str, int] = {}
    union: set[str] = set()
    for group, mask in groups:
        cards = [
            _ID_TO_CARD[card_id]
            for card_id in range(1, len(ALL_CARDS) + 1)
            if mask & (1 << (card_id - 1))
        ]
        group_counts[group] = len(cards)
        union.update(cards)
        for card in cards:
            tokens.append((_CARD_TO_ID[card], _ZONE_TO_ID[f"action_{group}"]))
    if union != dealt or len(tokens) != len(dealt):
        raise ValueError("ActionKey must account for every dealt card exactly once")
    expected_discard = 0 if observation.street == "T0" else 1
    if group_counts["discard"] != expected_discard:
        raise ValueError("ActionKey discard count disagrees with the street")
    for row in ROWS:
        if group_counts[row] > observation.hero_board.open_slots(row):
            raise ValueError(f"ActionKey exceeds open slots in {row}")
    if len(tokens) > MAX_ACTION_CARD_TOKENS:
        raise ValueError("ActionKey contains more than five cards")
    return tokens


def _scalar_context(observation: ActorObservation) -> np.ndarray:
    values: list[float] = []
    for row in ROWS:
        values.append(len(getattr(observation.hero_board, row)) / ROW_CAPACITY[row])
    for row in ROWS:
        values.append(
            len(getattr(observation.opponent_public_board, row)) / ROW_CAPACITY[row]
        )
    values.extend(
        (
            len(observation.dealt_cards) / 5.0,
            len(observation.hero_private_discards) / 4.0,
            observation.opponent_discard_count / 4.0,
        )
    )
    known_count = len(observation.known_unavailable_cards())
    unknown_count = len(ALL_CARDS) - known_count
    opponent_hidden_discards = observation.opponent_discard_count
    if opponent_hidden_discards > unknown_count:
        raise ValueError("opponent discard count exceeds the unknown-card pool")
    remaining_marginal = (
        (unknown_count - opponent_hidden_discards) / unknown_count
        if unknown_count
        else 0.0
    )
    fl_ev = dict(observation.scoring.fl_ev)
    values.extend(
        (
            known_count / len(ALL_CARDS),
            unknown_count / len(ALL_CARDS),
            remaining_marginal,
            float(observation.hero_in_fantasyland),
            float(observation.opponent_in_fantasyland),
            float(fl_ev.get(14, 0.0)) / 20.0,
            observation.scoring.middle_trips_royalty / 10.0,
            float(observation.scoring.hu_line_points),
            observation.scoring.scoop_bonus / 10.0,
            float(observation.scoring.foul_enabled),
            observation.scoring.fantasyland_cards / 20.0,
        )
    )
    if len(values) != len(SCALAR_CONTEXT_NAMES):
        raise AssertionError("StreetPolicyNetV1 scalar feature contract drift")
    return np.asarray(values, dtype=np.float32)


def build_street_policy_net_v1(
    torch: Any | None = None,
    config: StreetPolicyNetV1Config | None = None,
) -> Any:
    """Construct the shared first/second PyTorch network."""

    torch = torch or _import_torch()
    nn = torch.nn
    functional = torch.nn.functional
    resolved = config or StreetPolicyNetV1Config()

    class _SeatAffine(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_scale = nn.Parameter(torch.zeros(len(SEATS)))
            self.bias = nn.Parameter(torch.zeros(len(SEATS)))

        def forward(self, value: Any, seat_ids: Any) -> Any:
            scale = self.log_scale.exp()[seat_ids]
            bias = self.bias[seat_ids]
            while scale.ndim < value.ndim:
                scale = scale.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            return value * scale + bias

    class _SeatScale(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_scale = nn.Parameter(torch.zeros(len(SEATS)))

        def forward(self, value: Any, seat_ids: Any) -> Any:
            scale = self.log_scale.exp()[seat_ids]
            while scale.ndim < value.ndim:
                scale = scale.unsqueeze(-1)
            return value * scale

    class _StreetPolicyNetV1(nn.Module):
        schema = STREET_POLICY_NET_V1_SCHEMA
        feature_schema_hash = FEATURE_SCHEMA_HASH

        def __init__(self) -> None:
            super().__init__()
            self.config = resolved
            self.card_embedding = nn.Embedding(
                len(ALL_CARDS) + 1,
                resolved.card_embedding_dim,
                padding_idx=PAD_CARD_ID,
            )
            self.zone_embedding = nn.Embedding(
                len(ALL_ZONES) + 1,
                resolved.zone_embedding_dim,
                padding_idx=PAD_ZONE_ID,
            )
            token_input = resolved.card_embedding_dim + resolved.zone_embedding_dim
            self.state_token_encoder = nn.Sequential(
                nn.Linear(token_input, resolved.token_hidden_dim),
                nn.ReLU(),
                nn.Linear(resolved.token_hidden_dim, resolved.token_hidden_dim),
                nn.ReLU(),
            )
            self.action_token_encoder = nn.Sequential(
                nn.Linear(token_input, resolved.token_hidden_dim),
                nn.ReLU(),
                nn.Linear(resolved.token_hidden_dim, resolved.token_hidden_dim),
                nn.ReLU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(len(SCALAR_CONTEXT_NAMES), resolved.context_hidden_dim),
                nn.ReLU(),
            )
            self.seat_embedding = nn.Embedding(
                len(SEATS), resolved.seat_embedding_dim
            )
            self.street_embedding = nn.Embedding(
                len(STREETS), resolved.street_embedding_dim
            )
            state_input = (
                resolved.token_hidden_dim
                + resolved.context_hidden_dim
                + resolved.seat_embedding_dim
                + resolved.street_embedding_dim
            )
            self.state_backbone = nn.Sequential(
                nn.Linear(state_input, resolved.state_hidden_dim),
                nn.ReLU(),
                nn.Dropout(resolved.dropout),
                nn.Linear(resolved.state_hidden_dim, resolved.state_hidden_dim),
                nn.ReLU(),
            )
            self.action_backbone = nn.Sequential(
                nn.Linear(
                    resolved.state_hidden_dim + resolved.token_hidden_dim,
                    resolved.action_hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(resolved.dropout),
                nn.Linear(resolved.action_hidden_dim, resolved.action_hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(resolved.action_hidden_dim, 1)
            self.action_q_head = nn.Linear(resolved.action_hidden_dim, 1)
            self.baseline_delta_head = nn.Linear(resolved.action_hidden_dim, 1)
            self.state_value_head = nn.Linear(resolved.state_hidden_dim, 1)
            # These heads receive detached action features by construction:
            # safety-fit cannot silently modify the core/shared policy.
            self.uncertainty_head = nn.Linear(resolved.action_hidden_dim, 1)
            self.safe_head = nn.Linear(resolved.action_hidden_dim, 1)
            self.policy_seat_calibration = _SeatAffine()
            self.q_seat_calibration = _SeatAffine()
            self.delta_seat_calibration = _SeatScale()
            self.value_seat_calibration = _SeatAffine()
            self.uncertainty_seat_calibration = _SeatAffine()
            self.safe_seat_calibration = _SeatAffine()

        @staticmethod
        def _masked_mean(values: Any, mask: Any, dim: int) -> Any:
            weights = mask.to(values.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=dim).clamp_min(1.0)
            return (values * weights).sum(dim=dim) / denominator

        def forward(self, **batch: Any) -> dict[str, Any]:
            if set(batch) != set(MODEL_INPUT_FIELDS):
                raise ValueError(
                    "StreetPolicyNetV1 input field set mismatch; hidden truth "
                    "and undeclared tensors are forbidden"
                )
            legal = batch["legal_action_mask"]
            if legal.ndim != 2 or legal.shape[1] != MAX_LEGAL_ACTIONS:
                raise ValueError("legal action mask must have shape [B,232]")
            if not bool(torch.all(legal.any(dim=1)).item()):
                raise ValueError("each state needs at least one legal action")
            baseline_indices = batch["baseline_indices"]
            if tuple(baseline_indices.shape) != (legal.shape[0],):
                raise ValueError("baseline indices must have shape [B]")
            baseline_in_range = (baseline_indices >= 0) & (
                baseline_indices < MAX_LEGAL_ACTIONS
            )
            if not bool(torch.all(baseline_in_range).item()):
                raise ValueError("baseline index is outside [0,232)")
            if not bool(
                torch.all(
                    legal.gather(1, baseline_indices.unsqueeze(1)).squeeze(1)
                ).item()
            ):
                raise ValueError("baseline index does not identify a legal action")
            state_token = torch.cat(
                (
                    self.card_embedding(batch["state_card_ids"]),
                    self.zone_embedding(batch["state_zone_ids"]),
                ),
                dim=-1,
            )
            state_token = self.state_token_encoder(state_token)
            state_pool = self._masked_mean(
                state_token, batch["state_card_mask"], dim=1
            )
            context = self.context_encoder(batch["scalar_context"])
            seat = self.seat_embedding(batch["seat_ids"])
            street = self.street_embedding(batch["street_ids"])
            state_hidden = self.state_backbone(
                torch.cat((state_pool, context, seat, street), dim=-1)
            )

            action_token = torch.cat(
                (
                    self.card_embedding(batch["action_card_ids"]),
                    self.zone_embedding(batch["action_zone_ids"]),
                ),
                dim=-1,
            )
            action_token = self.action_token_encoder(action_token)
            action_pool = self._masked_mean(
                action_token, batch["action_card_mask"], dim=2
            )
            repeated_state = state_hidden.unsqueeze(1).expand(
                -1, MAX_LEGAL_ACTIONS, -1
            )
            action_hidden = self.action_backbone(
                torch.cat((repeated_state, action_pool), dim=-1)
            )
            seat_ids = batch["seat_ids"]
            policy = self.policy_seat_calibration(
                self.policy_head(action_hidden).squeeze(-1), seat_ids
            )
            action_q = self.q_seat_calibration(
                self.action_q_head(action_hidden).squeeze(-1), seat_ids
            )
            raw_delta = self.delta_seat_calibration(
                self.baseline_delta_head(action_hidden).squeeze(-1), seat_ids
            )
            baseline_delta = raw_delta.gather(
                1, baseline_indices.unsqueeze(1)
            )
            delta = raw_delta - baseline_delta
            value = self.value_seat_calibration(
                self.state_value_head(state_hidden).squeeze(-1), seat_ids
            )
            risk_hidden = action_hidden.detach()
            uncertainty = functional.softplus(
                self.uncertainty_seat_calibration(
                    self.uncertainty_head(risk_hidden).squeeze(-1), seat_ids
                )
            )
            safe_logit = self.safe_seat_calibration(
                self.safe_head(risk_hidden).squeeze(-1), seat_ids
            )
            negative_infinity = torch.full_like(policy, float("-inf"))
            zeros = torch.zeros_like(action_q)
            return {
                "policy_logits": torch.where(legal, policy, negative_infinity),
                "state_value": value,
                "action_q": torch.where(legal, action_q, zeros),
                "baseline_delta": torch.where(legal, delta, zeros),
                "uncertainty_p95": torch.where(legal, uncertainty, zeros),
                "safe_logits": torch.where(legal, safe_logit, zeros),
                "safe_probability": torch.where(
                    legal, torch.sigmoid(safe_logit), zeros
                ),
                "legal_action_mask": legal,
                "baseline_indices": baseline_indices,
            }

    return _StreetPolicyNetV1()


CORE_PARAMETER_PREFIXES = (
    "card_embedding.",
    "zone_embedding.",
    "state_token_encoder.",
    "action_token_encoder.",
    "context_encoder.",
    "seat_embedding.",
    "street_embedding.",
    "state_backbone.",
    "action_backbone.",
    "policy_head.",
    "action_q_head.",
    "baseline_delta_head.",
    "state_value_head.",
    "policy_seat_calibration.",
    "q_seat_calibration.",
    "delta_seat_calibration.",
    "value_seat_calibration.",
)
RISK_PARAMETER_PREFIXES = (
    "uncertainty_head.",
    "safe_head.",
    "uncertainty_seat_calibration.",
    "safe_seat_calibration.",
)


def parameter_names_for_update(model: Any, update_scope: str) -> tuple[str, ...]:
    """Return an exhaustive disjoint parameter set for an authorized update."""

    if update_scope not in {"core", "risk"}:
        raise ValueError("update scope must be core or risk")
    prefixes = CORE_PARAMETER_PREFIXES if update_scope == "core" else RISK_PARAMETER_PREFIXES
    selected = tuple(
        name
        for name, _parameter in model.named_parameters()
        if name.startswith(prefixes)
    )
    all_names = {name for name, _parameter in model.named_parameters()}
    core = {
        name for name in all_names if name.startswith(CORE_PARAMETER_PREFIXES)
    }
    risk = {
        name for name in all_names if name.startswith(RISK_PARAMETER_PREFIXES)
    }
    if core & risk or core | risk != all_names:
        raise RuntimeError("StreetPolicyNetV1 parameter ownership is not exhaustive")
    return selected


def authorize_weight_update(*, split_role: str, update_scope: str | None) -> None:
    """Fail closed on train/safety/threshold role leakage."""

    allowed = {
        "train": "core",
        "safety-fit": "risk",
        "threshold-lock": None,
        "diagnostic-holdout": None,
    }
    if split_role not in allowed:
        raise ValueError(f"unknown dataset split role: {split_role!r}")
    if update_scope != allowed[split_role]:
        raise PermissionError(
            f"{split_role} permits update scope {allowed[split_role]!r}, "
            f"not {update_scope!r}"
        )


def build_authorized_optimizer(
    torch: Any,
    model: Any,
    *,
    split_role: str,
    update_scope: str,
    learning_rate: float,
) -> Any:
    authorize_weight_update(split_role=split_role, update_scope=update_scope)
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    names = parameter_names_for_update(model, update_scope)
    parameters = dict(model.named_parameters())
    return torch.optim.AdamW(
        [parameters[name] for name in names],
        lr=float(learning_rate),
    )


def street_policy_training_loss(
    torch: Any,
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    split_role: str,
    update_scope: str,
) -> dict[str, Any]:
    """Compute only the losses authorized for one immutable split role."""

    authorize_weight_update(split_role=split_role, update_scope=update_scope)
    functional = torch.nn.functional
    legal = output["legal_action_mask"]
    if update_scope == "core":
        q_target = targets["action_q"]
        delta_target = targets["baseline_delta"]
        teacher_policy = targets["teacher_policy"]
        value_target = targets["state_value"]
        _require_shape(q_target, legal.shape, "action_q target")
        _require_shape(delta_target, legal.shape, "baseline_delta target")
        _require_shape(teacher_policy, legal.shape, "teacher_policy target")
        _require_shape(value_target, output["state_value"].shape, "state_value target")
        if bool(torch.any(teacher_policy[~legal] != 0).item()):
            raise ValueError("teacher policy assigns mass to an illegal action")
        target_baseline_delta = delta_target.gather(
            1, output["baseline_indices"].unsqueeze(1)
        ).squeeze(1)
        if not bool(
            torch.allclose(
                target_baseline_delta,
                torch.zeros_like(target_baseline_delta),
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("baseline delta target must be zero at the baseline ActionKey")
        row_mass = teacher_policy.sum(dim=1)
        if not bool(
            torch.allclose(
                row_mass,
                torch.ones_like(row_mass),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise ValueError("teacher policy must sum to one on every state")
        q_loss = functional.smooth_l1_loss(
            output["action_q"][legal], q_target[legal]
        )
        delta_loss = functional.smooth_l1_loss(
            output["baseline_delta"][legal], delta_target[legal]
        )
        log_policy = functional.log_softmax(output["policy_logits"], dim=1)
        positive = teacher_policy > 0
        policy_kl = (
            teacher_policy[positive]
            * (
                torch.log(teacher_policy[positive])
                - log_policy[positive]
            )
        ).sum() / legal.shape[0]
        value_loss = functional.smooth_l1_loss(
            output["state_value"], value_target
        )
        ranking = _pairwise_ranking_loss(
            torch,
            output["action_q"],
            q_target,
            legal,
        )
        weighted = {
            "action_q_huber": q_loss * LOSS_WEIGHTS["action_q_huber"],
            "baseline_delta_huber": delta_loss
            * LOSS_WEIGHTS["baseline_delta_huber"],
            "teacher_policy_kl": policy_kl * LOSS_WEIGHTS["teacher_policy_kl"],
            "state_value_huber": value_loss
            * LOSS_WEIGHTS["state_value_huber"],
            "ranking": ranking * LOSS_WEIGHTS["ranking"],
        }
    else:
        uncertainty_target = targets["downside_p95"]
        safe_target = targets["safe"]
        _require_shape(uncertainty_target, legal.shape, "downside_p95 target")
        _require_shape(safe_target, legal.shape, "safe target")
        if bool(torch.any(uncertainty_target[legal] < 0).item()):
            raise ValueError("downside target must be non-negative")
        if bool(
            torch.any((safe_target[legal] < 0) | (safe_target[legal] > 1)).item()
        ):
            raise ValueError("safe targets must be probabilities")
        error = uncertainty_target[legal] - output["uncertainty_p95"][legal]
        quantile = torch.maximum(0.95 * error, -0.05 * error).mean()
        safe_bce = functional.binary_cross_entropy_with_logits(
            output["safe_logits"][legal], safe_target[legal]
        )
        weighted = {
            "uncertainty_quantile": quantile
            * LOSS_WEIGHTS["uncertainty_quantile"],
            "safe_bce": safe_bce * LOSS_WEIGHTS["safe_bce"],
        }
    total = sum(weighted.values())
    return {
        "schema": STREET_POLICY_LOSS_SCHEMA,
        "loss_schema_hash": LOSS_SCHEMA_HASH,
        "split_role": split_role,
        "update_scope": update_scope,
        "components": weighted,
        "total": total,
    }


def _require_shape(value: Any, expected: Any, name: str) -> None:
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f"{name} shape mismatch")


def _pairwise_ranking_loss(
    torch: Any, prediction: Any, target: Any, legal: Any
) -> Any:
    losses = []
    for row in range(prediction.shape[0]):
        indices = torch.nonzero(legal[row], as_tuple=False).squeeze(-1)
        predicted = prediction[row, indices]
        expected = target[row, indices]
        target_difference = expected.unsqueeze(1) - expected.unsqueeze(0)
        upper = torch.triu(
            torch.ones_like(target_difference, dtype=torch.bool), diagonal=1
        )
        non_tie = upper & (target_difference != 0)
        if bool(non_tie.any().item()):
            predicted_difference = predicted.unsqueeze(1) - predicted.unsqueeze(0)
            signed = torch.sign(target_difference[non_tie])
            losses.append(
                torch.nn.functional.softplus(
                    -signed * predicted_difference[non_tie]
                ).mean()
            )
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def model_state_sha256(model: Any) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes deterministically."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_street_policy_checkpoint(
    path: str | Path,
    model: Any,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a byte-deterministic, write-once checkpoint ZIP."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    config = getattr(model, "config", None)
    if not isinstance(config, StreetPolicyNetV1Config):
        raise TypeError("model does not expose a StreetPolicyNetV1Config")
    canonical_provenance = _canonical_json_value(provenance)
    if not isinstance(canonical_provenance, dict):
        raise ValueError("checkpoint provenance must be a mapping")

    entries: dict[str, bytes] = {}
    weight_manifest: list[dict[str, Any]] = []
    for index, (name, tensor) in enumerate(sorted(model.state_dict().items())):
        array = tensor.detach().cpu().contiguous().numpy()
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        entry_name = f"weights/{index:04d}.npy"
        payload = buffer.getvalue()
        entries[entry_name] = payload
        weight_manifest.append(
            {
                "entry": entry_name,
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": _sha256_bytes(payload),
            }
        )
    identity_payload = {
        "schema": STREET_POLICY_CHECKPOINT_SCHEMA,
        "model_schema": STREET_POLICY_NET_V1_SCHEMA,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "loss_schema_hash": LOSS_SCHEMA_HASH,
        "config": config.to_dict(),
        "config_schema_hash": config.schema_hash,
        "model_state_sha256": model_state_sha256(model),
        "provenance": canonical_provenance,
        "weights": weight_manifest,
    }
    manifest = dict(identity_payload)
    manifest["checkpoint_identity_sha256"] = _sha256_bytes(
        _canonical_json_bytes(identity_payload)
    )
    entries["manifest.json"] = _canonical_json_bytes(manifest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 0
                info.external_attr = 0
                archive.writestr(info, entries[name])
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def load_street_policy_checkpoint(
    path: str | Path,
    *,
    torch: Any | None = None,
    map_location: str | Any = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Validate every byte before constructing and loading the network."""

    torch = torch or _import_torch()
    source = Path(path)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValueError("checkpoint ZIP has duplicate entries or no manifest")
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                raise ValueError("checkpoint manifest must be a mapping")
            if manifest_bytes != _canonical_json_bytes(manifest):
                raise ValueError("checkpoint manifest is not canonical JSON")
            _validate_checkpoint_manifest(manifest)
            expected_entries = {
                "manifest.json",
                *(row["entry"] for row in manifest["weights"]),
            }
            if set(names) != expected_entries:
                raise ValueError("checkpoint ZIP entry set mismatch")
            arrays: dict[str, np.ndarray] = {}
            for row in manifest["weights"]:
                payload = archive.read(row["entry"])
                if _sha256_bytes(payload) != row["sha256"]:
                    raise ValueError("checkpoint weight entry hash mismatch")
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                if array.dtype.str != row["dtype"] or list(array.shape) != row["shape"]:
                    raise ValueError("checkpoint tensor metadata mismatch")
                arrays[row["name"]] = array
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid StreetPolicyNetV1 checkpoint") from exc

    config = StreetPolicyNetV1Config(**manifest["config"])
    model = build_street_policy_net_v1(torch, config)
    expected_names = set(model.state_dict())
    if set(arrays) != expected_names:
        raise ValueError("checkpoint parameter name set mismatch")
    state = {
        name: torch.from_numpy(array.copy()).to(map_location)
        for name, array in arrays.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(map_location)
    if model_state_sha256(model) != manifest["model_state_sha256"]:
        raise ValueError("loaded model state digest mismatch")
    return model, manifest


def _validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "model_schema",
        "feature_schema_hash",
        "loss_schema_hash",
        "config",
        "config_schema_hash",
        "model_state_sha256",
        "provenance",
        "weights",
        "checkpoint_identity_sha256",
    }
    if set(manifest) != required:
        raise ValueError("checkpoint manifest field set mismatch")
    if manifest["schema"] != STREET_POLICY_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if manifest["model_schema"] != STREET_POLICY_NET_V1_SCHEMA:
        raise ValueError("model schema mismatch")
    if manifest["feature_schema_hash"] != FEATURE_SCHEMA_HASH:
        raise ValueError("feature schema hash mismatch")
    if manifest["loss_schema_hash"] != LOSS_SCHEMA_HASH:
        raise ValueError("loss schema hash mismatch")
    config = StreetPolicyNetV1Config(**manifest["config"])
    if manifest["config_schema_hash"] != config.schema_hash:
        raise ValueError("config schema hash mismatch")
    identity = dict(manifest)
    declared = identity.pop("checkpoint_identity_sha256")
    if declared != _sha256_bytes(_canonical_json_bytes(identity)):
        raise ValueError("checkpoint identity digest mismatch")
    weights = manifest["weights"]
    if not isinstance(weights, list) or not weights:
        raise ValueError("checkpoint weights must be a non-empty list")
    entries: set[str] = set()
    names: set[str] = set()
    for row in weights:
        if not isinstance(row, dict) or set(row) != {
            "entry",
            "name",
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError("checkpoint weight manifest row mismatch")
        if row["entry"] in entries or row["name"] in names:
            raise ValueError("checkpoint weight manifest has duplicates")
        entries.add(row["entry"])
        names.add(row["name"])


def _canonical_json_value(value: Any) -> Any:
    # Round-trip rejects non-JSON types and NaN/Inf while producing a detached,
    # stable representation.
    return json.loads(_canonical_json_bytes(value))


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on lean installs.
        raise RuntimeError(
            "StreetPolicyNetV1 requires the optional PyTorch training runtime"
        ) from exc
    return torch


__all__ = [
    "ACTION_ZONES",
    "FEATURE_SCHEMA_HASH",
    "LOSS_SCHEMA_HASH",
    "LOSS_WEIGHTS",
    "MAX_ACTION_CARD_TOKENS",
    "MAX_LEGAL_ACTIONS",
    "MAX_STATE_CARD_TOKENS",
    "MODEL_INPUT_FIELDS",
    "SCALAR_CONTEXT_NAMES",
    "STATE_ZONES",
    "STREET_POLICY_CHECKPOINT_SCHEMA",
    "STREET_POLICY_FEATURE_SCHEMA",
    "STREET_POLICY_LOSS_SCHEMA",
    "STREET_POLICY_NET_V1_SCHEMA",
    "StreetPolicyEncodedBatch",
    "StreetPolicyNetV1Config",
    "authorize_weight_update",
    "build_authorized_optimizer",
    "build_street_policy_net_v1",
    "encode_street_policy_batch",
    "feature_schema_payload",
    "load_street_policy_checkpoint",
    "model_state_sha256",
    "parameter_names_for_update",
    "save_street_policy_checkpoint",
    "street_policy_training_loss",
]
