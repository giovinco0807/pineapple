"""Content-addressed Torch policy adapter for full-card behavior beliefs.

This adapter is deliberately strict: it runs on CPU, supports regular T1-T4
decisions only, regenerates the complete legal action set, and quantizes the
masked neural distribution to exact rational probabilities.  A loaded legacy
checkpoint remains legacy unless its manifest scope is explicitly approved by
an M3 promotion gate.
"""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ai.engine.action_space import get_semantic_action_index, get_turn_actions
from ai.engine.encoding import Board, Observation, encode_state
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.models.networks import PolicyNetwork, _adapt_state
from ai.training.train_t2_oracle import T2PolicyValueNet
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    BehaviorInfoSet,
    FrozenBehaviorModel,
    _canonical_sha256,
)


RULES_VERSION = "canonical_joker_bottom_middle_top_20260711"
ADAPTER_VERSION = "torch_policy_behavior_q32_v1"
POLICY_VALUE_ADAPTER_VERSION = "torch_policyvalue_behavior_q32_v1"
ENCODING_VERSION = "legacy_state_encoding_522_with_520_adapter_v1"
POLICY_VALUE_ENCODING_VERSION = "state_encoding_522_prefix_v1"
ACTION_VERSION = "regular_turn_semantic_27_in_policy_output_v1"
DEFAULT_QUANTIZATION_DENOMINATOR = 1 << 32

KNOWN_HU_POLICY_VALUE_ASSETS = MappingProxyType(
    {
        (1, "bb"): MappingProxyType(
            {
                "relative_path": (
                    "ai/data/t1_hu_bb_gcp_2m_model/t2_policyvalue_model.pt"
                ),
                "checkpoint_sha256": (
                    "27dab713a658a5ede5637d2c96ae0d5330464b96738f562ae84ce7326c2562c6"
                ),
            }
        ),
        (1, "btn"): MappingProxyType(
            {
                "relative_path": (
                    "ai/data/t1_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"
                ),
                "checkpoint_sha256": (
                    "064ab29967d4294f76f4ac844000b7fad97553eb341cfb54065f023bbfe7a32f"
                ),
            }
        ),
        (2, "bb"): MappingProxyType(
            {
                "relative_path": (
                    "ai/data/t2_hu_bb_gcp_2m_model/t2_policyvalue_model.pt"
                ),
                "checkpoint_sha256": (
                    "fc47e5a7a02375c8d3fac32d04b0849d8e7b100fe5650007c8c7f8e6c7b69460"
                ),
            }
        ),
        (2, "btn"): MappingProxyType(
            {
                "relative_path": (
                    "ai/data/t2_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"
                ),
                "checkpoint_sha256": (
                    "a4d5dfcff1811515a6b3db972b7f06e071f00666db49f3cefe8a0079b7e6c83b"
                ),
            }
        ),
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch
        return torch.load(path, map_location="cpu")


def _load_state_dict(path: Path) -> tuple[Mapping[str, torch.Tensor], str]:
    payload = _safe_torch_load(path)
    checkpoint_format = "raw_state_dict"
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
        checkpoint_format = "model_state_dict_wrapper"
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
        checkpoint_format = "state_dict_wrapper"
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("behavior checkpoint does not contain a state dict")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("behavior checkpoint state dict must map strings to tensors")
        state_dict[key] = value.detach().cpu()
    return MappingProxyType(state_dict), checkpoint_format


def _required_checkpoint_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"policy-value checkpoint {key!r} must be a positive integer")
    return value


def _load_policy_value_checkpoint(
    path: Path,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(
            "policy-value behavior checkpoint requires a model_state_dict wrapper"
        )
    state_dim = _required_checkpoint_int(payload, "state_dim")
    n_actions = _required_checkpoint_int(payload, "n_actions")
    hidden = _required_checkpoint_int(payload, "hidden")
    n_blocks = _required_checkpoint_int(payload, "n_blocks")
    epoch = _required_checkpoint_int(payload, "epoch")
    if state_dim != 522:
        raise ValueError(
            f"policy-value behavior checkpoint requires 522 inputs, got {state_dim}"
        )
    if n_actions != 27:
        raise ValueError(
            f"policy-value behavior checkpoint requires 27 actions, got {n_actions}"
        )
    if hidden != 1024 or n_blocks != 4:
        raise ValueError(
            "unsupported policy-value behavior architecture: "
            f"hidden={hidden}, n_blocks={n_blocks}"
        )
    validation: dict[str, float] = {}
    for key in ("val_regret", "val_top1"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"policy-value checkpoint {key!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"policy-value checkpoint {key!r} must be finite")
        validation[key] = numeric

    raw_state_dict = payload["model_state_dict"]
    if not isinstance(raw_state_dict, Mapping) or not raw_state_dict:
        raise ValueError("policy-value model_state_dict must be a non-empty mapping")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(
                "policy-value model_state_dict must map strings to tensors"
            )
        state_dict[key] = value.detach().cpu()
    metadata = MappingProxyType(
        {
            "state_dim": state_dim,
            "n_actions": n_actions,
            "hidden": hidden,
            "n_blocks": n_blocks,
            "epoch": epoch,
            **validation,
        }
    )
    return MappingProxyType(state_dict), metadata


def _infer_policy_dimensions(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[int, int, tuple[int, ...]]:
    linear_weights: list[tuple[int, str, torch.Tensor]] = []
    for key, tensor in state_dict.items():
        if not key.startswith("net.") or not key.endswith(".weight") or tensor.ndim != 2:
            continue
        try:
            index = int(key.split(".")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid PolicyNetwork state key: {key!r}") from exc
        linear_weights.append((index, key, tensor))
    linear_weights.sort()
    if len(linear_weights) != 4:
        raise ValueError("behavior checkpoint must contain four PolicyNetwork linear layers")
    shapes = [tuple(int(value) for value in tensor.shape) for _index, _key, tensor in linear_weights]
    input_dim = shapes[0][1]
    output_dim = shapes[-1][0]
    hidden_dims = tuple(shape[0] for shape in shapes[:-1])
    expected = [
        (hidden_dims[0], input_dim),
        (hidden_dims[1], hidden_dims[0]),
        (hidden_dims[2], hidden_dims[1]),
        (output_dim, hidden_dims[2]),
    ]
    if shapes != expected or hidden_dims != (1024, 512, 256):
        raise ValueError(f"unsupported PolicyNetwork architecture: {shapes}")
    if input_dim not in (520, 522):
        raise ValueError(f"unsupported behavior checkpoint input dimension: {input_dim}")
    if output_dim < 27:
        raise ValueError("behavior checkpoint output must contain all 27 regular action slots")
    return input_dim, output_dim, hidden_dims


def _board(rows: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _quantize_largest_remainder(
    action_probabilities: Mapping[str, float],
    *,
    denominator: int,
) -> Mapping[str, Fraction]:
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("quantization denominator must be a positive integer")
    ordered = tuple(sorted(action_probabilities))
    if not ordered:
        raise ValueError("cannot quantize an empty behavior distribution")
    raw: dict[str, Fraction] = {}
    for action_id in ordered:
        probability = float(action_probabilities[action_id])
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("neural behavior probabilities must be finite and non-negative")
        raw[action_id] = Fraction.from_float(probability)
    total = sum(raw.values(), Fraction(0, 1))
    if total <= 0:
        raise ValueError("neural behavior probabilities must have positive mass")

    floors: dict[str, int] = {}
    remainders: list[tuple[Fraction, str]] = []
    for action_id in ordered:
        quota = raw[action_id] * denominator / total
        units = quota.numerator // quota.denominator
        floors[action_id] = units
        remainders.append((quota - units, action_id))
    units_left = denominator - sum(floors.values())
    if not 0 <= units_left <= len(ordered):
        raise AssertionError("largest-remainder quantization produced invalid residual units")
    for _remainder, action_id in sorted(
        remainders,
        key=lambda item: (-item[0], item[1]),
    )[:units_left]:
        floors[action_id] += 1
    quantized = {
        action_id: Fraction(floors[action_id], denominator) for action_id in ordered
    }
    if sum(quantized.values(), Fraction(0, 1)) != 1:
        raise AssertionError("quantized behavior distribution does not sum exactly to one")
    return MappingProxyType(quantized)


class TorchPolicyBehaviorModel(FrozenBehaviorModel):
    """Frozen PolicyNetwork checkpoint exposed as an exact behavior model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_id: str,
        supported_turns: Sequence[int],
        supported_actors: Sequence[str],
        training_scope_id: str,
        promotion_eligible: bool = False,
        temperature: Fraction | int | str = Fraction(1, 1),
        quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(training_scope_id, str) or not training_scope_id.strip():
            raise ValueError("training_scope_id must be a non-empty string")
        if not isinstance(promotion_eligible, bool):
            raise TypeError("promotion_eligible must be boolean")
        if promotion_eligible and training_scope_id.lower().startswith("legacy"):
            raise ValueError("legacy behavior training scopes cannot be promotion eligible")
        if isinstance(temperature, bool) or isinstance(temperature, float):
            raise TypeError("temperature must be an exact Fraction/int/str")
        exact_temperature = Fraction(temperature)
        if exact_temperature <= 0:
            raise ValueError("temperature must be positive")
        if (
            isinstance(quantization_denominator, bool)
            or not isinstance(quantization_denominator, int)
            or quantization_denominator <= 0
        ):
            raise ValueError("quantization_denominator must be a positive integer")
        turns = tuple(sorted({int(turn) for turn in supported_turns}))
        if not turns or any(turn not in (1, 2, 3, 4) for turn in turns):
            raise ValueError("supported_turns must be a non-empty subset of T1-T4")
        actors = tuple(sorted({str(actor) for actor in supported_actors}))
        if not actors or any(actor not in ("bb", "btn") for actor in actors):
            raise ValueError("supported_actors must contain only 'bb'/'btn'")

        checkpoint_sha256 = _sha256_file(path)
        if expected_checkpoint_sha256 is not None and (
            expected_checkpoint_sha256 != checkpoint_sha256
        ):
            raise ValueError("behavior checkpoint SHA256 does not match the expected hash")
        state_dict, checkpoint_format = _load_state_dict(path)
        input_dim, output_dim, hidden_dims = _infer_policy_dimensions(state_dict)
        model = PolicyNetwork(
            input_dim=input_dim,
            max_actions=output_dim,
            dropout=0.0,
        )
        model.load_state_dict(dict(state_dict), strict=True)
        model.to("cpu")
        model.eval()

        self._path = path
        self._model_id = model_id
        self._supported_turns = turns
        self._supported_actors = actors
        self._training_scope_id = training_scope_id
        self._promotion_eligible = promotion_eligible
        self._temperature = exact_temperature
        self._quantization_denominator = quantization_denominator
        self._checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_format = checkpoint_format
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dims = hidden_dims
        self._model = model
        self._adapter_sha256 = _sha256_file(Path(__file__).resolve())
        self._manifest = MappingProxyType(
            {
                "schema": "ofc_frozen_behavior_model/v1",
                "model_id": model_id,
                "model_type": "torch_policy_network_q32",
                "position_contract_version": POSITION_CONTRACT_VERSION,
                "rules_version": RULES_VERSION,
                "adapter_version": ADAPTER_VERSION,
                "adapter_source_sha256": self._adapter_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_format": checkpoint_format,
                "architecture": "PolicyNetwork_1024_512_256",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dims": list(hidden_dims),
                "encoding_version": ENCODING_VERSION,
                "action_version": ACTION_VERSION,
                "supported_turns": list(turns),
                "supported_actors": list(actors),
                "normal_hand_only": True,
                "training_scope_id": training_scope_id,
                "promotion_eligible": promotion_eligible,
                "temperature": (
                    f"{exact_temperature.numerator}/{exact_temperature.denominator}"
                ),
                "quantization": "largest_remainder_exact_fraction",
                "quantization_denominator": quantization_denominator,
                "device": "cpu",
                "inference_dtype": "torch.float32_logits_float64_softmax",
                "public_history_encoding": "boards_plus_actor_private_recall_discards",
            }
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def model_sha256(self) -> str:
        return _canonical_sha256(dict(self._manifest))

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if information.turn not in self._supported_turns:
            raise ValueError(f"behavior model does not support turn T{information.turn}")
        if information.actor not in self._supported_actors:
            raise ValueError(
                f"behavior model does not support actor {information.actor!r}"
            )
        if information.fantasy_state is not None:
            raise ValueError("Torch behavior adapter supports normal hands only")

        board_self_rows = (
            information.board_bb if information.actor == "bb" else information.board_btn
        )
        board_opponent_rows = (
            information.board_btn if information.actor == "bb" else information.board_bb
        )
        board_self = _board(board_self_rows)
        current_draw = list(information.current_draw)
        legal_actions = get_turn_actions(current_draw, board_self)
        legal_by_id = {action_key(action): action for action in legal_actions}
        if set(legal_by_id) != set(information.legal_action_ids):
            raise ValueError("behavior information legal actions do not match regeneration")
        semantic_indices: dict[str, int] = {}
        for action_id in information.legal_action_ids:
            index = get_semantic_action_index(legal_by_id[action_id], current_draw)
            if not 0 <= index < self._output_dim:
                raise ValueError("behavior action semantic index exceeds checkpoint output")
            semantic_indices[action_id] = index
        if len(set(semantic_indices.values())) != len(semantic_indices):
            raise ValueError("behavior legal actions collide in semantic action space")

        observation = Observation(
            board_self=board_self,
            board_opponent=_board(board_opponent_rows),
            dealt_cards=current_draw,
            known_discards_self=[
                card for _turn, card in information.own_recall_before.discards_by_turn
            ],
            turn=information.turn,
            is_btn=information.actor == "btn",
            is_fl=False,
            opp_is_fl=False,
        )
        state = encode_state(observation)
        if not isinstance(state, np.ndarray) or state.ndim != 1:
            raise ValueError("behavior state encoder returned an invalid vector")
        state_tensor = torch.from_numpy(state.astype(np.float32, copy=False)).unsqueeze(0)
        with torch.no_grad():
            adapted = _adapt_state(state_tensor, self._input_dim)
            logits = self._model.net(adapted).squeeze(0)
            selected = torch.stack(
                [logits[semantic_indices[action_id]] for action_id in information.legal_action_ids]
            ).to(dtype=torch.float64)
            selected = selected / float(self._temperature)
            probabilities = torch.softmax(selected, dim=0).cpu().tolist()
        probability_by_action = {
            action_id: float(probability)
            for action_id, probability in zip(
                information.legal_action_ids,
                probabilities,
            )
        }
        exact = _quantize_largest_remainder(
            probability_by_action,
            denominator=self._quantization_denominator,
        )
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=exact,
            source="model",
            used_fallback=False,
        )


class TorchPolicyValueBehaviorModel(FrozenBehaviorModel):
    """Frozen 522-input HU PolicyValueNet exposed as an uncalibrated prior.

    The existing position-specific HU checkpoints were optimized to rank
    actions by teacher EV.  Their logits are useful as a deterministic
    Boltzmann prior, but they are not calibrated observations of action
    frequency.  Consequently this adapter is deliberately non-promotable
    until an independent calibration artifact and gate are implemented.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_id: str,
        supported_turns: Sequence[int],
        supported_actors: Sequence[str],
        training_scope_id: str,
        temperature: Fraction | int | str = Fraction(1, 1),
        quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(training_scope_id, str) or not training_scope_id.strip():
            raise ValueError("training_scope_id must be a non-empty string")
        if isinstance(temperature, bool) or isinstance(temperature, float):
            raise TypeError("temperature must be an exact Fraction/int/str")
        exact_temperature = Fraction(temperature)
        if exact_temperature <= 0:
            raise ValueError("temperature must be positive")
        if (
            isinstance(quantization_denominator, bool)
            or not isinstance(quantization_denominator, int)
            or quantization_denominator <= 0
        ):
            raise ValueError("quantization_denominator must be a positive integer")
        turns = tuple(sorted({int(turn) for turn in supported_turns}))
        if not turns or any(turn not in (1, 2, 3, 4) for turn in turns):
            raise ValueError("supported_turns must be a non-empty subset of T1-T4")
        actors = tuple(sorted({str(actor) for actor in supported_actors}))
        if not actors or any(actor not in ("bb", "btn") for actor in actors):
            raise ValueError("supported_actors must contain only 'bb'/'btn'")

        checkpoint_sha256 = _sha256_file(path)
        if expected_checkpoint_sha256 is not None and (
            expected_checkpoint_sha256 != checkpoint_sha256
        ):
            raise ValueError("behavior checkpoint SHA256 does not match the expected hash")
        state_dict, checkpoint_metadata = _load_policy_value_checkpoint(path)
        model = T2PolicyValueNet(
            state_dim=int(checkpoint_metadata["state_dim"]),
            n_actions=int(checkpoint_metadata["n_actions"]),
            hidden=int(checkpoint_metadata["hidden"]),
            n_blocks=int(checkpoint_metadata["n_blocks"]),
            dropout=0.0,
        )
        model.load_state_dict(dict(state_dict), strict=True)
        model.to("cpu")
        model.eval()

        architecture_source = (
            Path(__file__).resolve().parents[1] / "training" / "train_t2_oracle.py"
        )
        if not architecture_source.is_file():
            raise FileNotFoundError(architecture_source)
        self._path = path
        self._model_id = model_id
        self._supported_turns = turns
        self._supported_actors = actors
        self._training_scope_id = training_scope_id
        self._temperature = exact_temperature
        self._quantization_denominator = quantization_denominator
        self._checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_metadata = checkpoint_metadata
        self._input_dim = int(checkpoint_metadata["state_dim"])
        self._output_dim = int(checkpoint_metadata["n_actions"])
        self._model = model
        self._adapter_sha256 = _sha256_file(Path(__file__).resolve())
        self._architecture_source_sha256 = _sha256_file(architecture_source)
        self._manifest = MappingProxyType(
            {
                "schema": "ofc_frozen_behavior_model/v1",
                "model_id": model_id,
                "model_type": "torch_policyvalue_boltzmann_q32",
                "position_contract_version": POSITION_CONTRACT_VERSION,
                "rules_version": RULES_VERSION,
                "adapter_version": POLICY_VALUE_ADAPTER_VERSION,
                "adapter_source_sha256": self._adapter_sha256,
                "architecture_source_sha256": self._architecture_source_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_format": "model_state_dict_wrapper",
                "architecture": "T2PolicyValueNet_residual_1024x4",
                "input_dim": self._input_dim,
                "output_dim": self._output_dim,
                "hidden": int(checkpoint_metadata["hidden"]),
                "n_blocks": int(checkpoint_metadata["n_blocks"]),
                "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
                "checkpoint_val_regret": float(checkpoint_metadata["val_regret"]),
                "checkpoint_val_top1": float(checkpoint_metadata["val_top1"]),
                "encoding_version": POLICY_VALUE_ENCODING_VERSION,
                "action_version": ACTION_VERSION,
                "supported_turns": list(turns),
                "supported_actors": list(actors),
                "normal_hand_only": True,
                "training_scope_id": training_scope_id,
                "promotion_eligible": False,
                "calibration_status": "uncalibrated_policy_ranking_logits",
                "behavior_likelihood_semantics": (
                    "boltzmann_over_teacher_ev_ranking_logits"
                ),
                "temperature": (
                    f"{exact_temperature.numerator}/{exact_temperature.denominator}"
                ),
                "quantization": "largest_remainder_exact_fraction",
                "quantization_denominator": quantization_denominator,
                "device": "cpu",
                "inference_dtype": "torch.float32_logits_float64_softmax",
                "public_history_encoding": (
                    "boards_plus_actor_private_recall_discards"
                ),
            }
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def model_sha256(self) -> str:
        return _canonical_sha256(dict(self._manifest))

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if information.turn not in self._supported_turns:
            raise ValueError(f"behavior model does not support turn T{information.turn}")
        if information.actor not in self._supported_actors:
            raise ValueError(
                f"behavior model does not support actor {information.actor!r}"
            )
        if information.fantasy_state is not None:
            raise ValueError("Torch behavior adapter supports normal hands only")

        board_self_rows = (
            information.board_bb if information.actor == "bb" else information.board_btn
        )
        board_opponent_rows = (
            information.board_btn if information.actor == "bb" else information.board_bb
        )
        board_self = _board(board_self_rows)
        current_draw = list(information.current_draw)
        legal_actions = get_turn_actions(current_draw, board_self)
        legal_by_id = {action_key(action): action for action in legal_actions}
        if set(legal_by_id) != set(information.legal_action_ids):
            raise ValueError("behavior information legal actions do not match regeneration")
        semantic_indices: dict[str, int] = {}
        for action_id in information.legal_action_ids:
            index = get_semantic_action_index(legal_by_id[action_id], current_draw)
            if not 0 <= index < self._output_dim:
                raise ValueError("behavior action semantic index exceeds checkpoint output")
            semantic_indices[action_id] = index
        if len(set(semantic_indices.values())) != len(semantic_indices):
            raise ValueError("behavior legal actions collide in semantic action space")

        observation = Observation(
            board_self=board_self,
            board_opponent=_board(board_opponent_rows),
            dealt_cards=current_draw,
            known_discards_self=[
                card for _turn, card in information.own_recall_before.discards_by_turn
            ],
            turn=information.turn,
            is_btn=information.actor == "btn",
            is_fl=False,
            opp_is_fl=False,
        )
        state = encode_state(observation)
        if not isinstance(state, np.ndarray) or state.ndim != 1:
            raise ValueError("behavior state encoder returned an invalid vector")
        if state.shape[0] < self._input_dim:
            raise ValueError(
                "behavior state encoder is shorter than the checkpoint input"
            )
        state_tensor = torch.from_numpy(
            state[: self._input_dim].astype(np.float32, copy=False)
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self._model(state_tensor, masks=None)
            selected = torch.stack(
                [
                    logits[0, semantic_indices[action_id]]
                    for action_id in information.legal_action_ids
                ]
            ).to(dtype=torch.float64)
            selected = selected / float(self._temperature)
            probabilities = torch.softmax(selected, dim=0).cpu().tolist()
        probability_by_action = {
            action_id: float(probability)
            for action_id, probability in zip(
                information.legal_action_ids,
                probabilities,
            )
        }
        exact = _quantize_largest_remainder(
            probability_by_action,
            denominator=self._quantization_denominator,
        )
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=exact,
            source="model",
            used_fallback=False,
        )


class TurnActorBehaviorDispatch(FrozenBehaviorModel):
    """Content-addressed exact router for turn/actor specialist models."""

    def __init__(
        self,
        routes: Mapping[tuple[int, str], FrozenBehaviorModel],
        *,
        model_id: str = "turn_actor_behavior_dispatch_v1",
    ) -> None:
        if not routes:
            raise ValueError("behavior dispatch requires at least one route")
        normalized: dict[tuple[int, str], FrozenBehaviorModel] = {}
        route_manifest: list[dict[str, Any]] = []
        all_children_promotion_eligible = True
        for (raw_turn, raw_actor), model in routes.items():
            turn = int(raw_turn)
            actor = str(raw_actor)
            if turn not in (1, 2, 3, 4) or actor not in ("bb", "btn"):
                raise ValueError("behavior dispatch routes require T1-T4 and bb/btn")
            key = (turn, actor)
            if key in normalized:
                raise ValueError(f"duplicate behavior dispatch route: {key}")
            manifest = model.model_manifest
            if not isinstance(manifest, Mapping):
                raise TypeError("child behavior model manifest must be a mapping")
            if _canonical_sha256(dict(manifest)) != model.model_sha256:
                raise ValueError("child behavior model manifest/hash mismatch")
            all_children_promotion_eligible = (
                all_children_promotion_eligible
                and manifest.get("promotion_eligible") is True
            )
            normalized[key] = model
            route_manifest.append(
                {
                    "turn": turn,
                    "actor": actor,
                    "child_model_id": model.model_id,
                    "child_model_sha256": model.model_sha256,
                }
            )
        self._routes = MappingProxyType(dict(sorted(normalized.items())))
        self._model_id = model_id
        self._manifest = MappingProxyType(
            {
                "schema": "ofc_frozen_behavior_model/v1",
                "model_id": model_id,
                "model_type": "turn_actor_dispatch",
                "position_contract_version": POSITION_CONTRACT_VERSION,
                "rules_version": RULES_VERSION,
                "promotion_eligible": all_children_promotion_eligible,
                "routes": sorted(route_manifest, key=lambda row: (row["turn"], row["actor"])),
            }
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def model_sha256(self) -> str:
        return _canonical_sha256(dict(self._manifest))

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        model = self._routes.get((information.turn, information.actor))
        if model is None:
            raise KeyError(
                f"no frozen behavior model for T{information.turn} {information.actor}"
            )
        return model.action_distribution(information)


def build_known_hu_policy_value_prior_dispatch(
    workspace_root: str | Path | None = None,
    *,
    temperature: Fraction | int | str = Fraction(1, 1),
    quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
) -> TurnActorBehaviorDispatch:
    """Load the four known T1/T2 position specialists as a frozen prior.

    The returned dispatch is intentionally not promotion eligible because the
    stored ranking logits have not yet passed an action-frequency calibration
    gate.  It is suitable for wiring and sensitivity experiments, not M3
    promotion evidence.
    """

    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    routes: dict[tuple[int, str], FrozenBehaviorModel] = {}
    for (turn, actor), asset in sorted(KNOWN_HU_POLICY_VALUE_ASSETS.items()):
        checkpoint_path = root / str(asset["relative_path"])
        routes[(turn, actor)] = TorchPolicyValueBehaviorModel(
            checkpoint_path,
            model_id=f"hu_t{turn}_{actor}_2m_policyvalue_prior",
            supported_turns=(turn,),
            supported_actors=(actor,),
            training_scope_id=(
                f"hu_t{turn}_{actor}_visible_opponent_board_2m_v1"
            ),
            temperature=temperature,
            quantization_denominator=quantization_denominator,
            expected_checkpoint_sha256=str(asset["checkpoint_sha256"]),
        )
    return TurnActorBehaviorDispatch(
        routes,
        model_id="known_hu_t1_t2_policyvalue_prior_dispatch_v1",
    )
