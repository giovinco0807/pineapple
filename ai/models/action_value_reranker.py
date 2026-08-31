"""Action-value reranker for OFC candidate placements.

The policy nets in this project classify a fixed action id.  This model scores
the board after a candidate action instead, so it can compare arbitrary legal
placements by predicted session-oriented EV plus auxiliary bust/FL signals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.engine.encoding import (
    ALL_CARDS,
    LOC_IN_HAND,
    LOC_MY_BOT,
    LOC_MY_DISCARD,
    LOC_MY_MID,
    LOC_MY_TOP,
    LOC_OPP_BOT,
    LOC_OPP_MID,
    LOC_OPP_TOP,
    STATE_DIM,
)
from ai.models.networks import _adapt_state

FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")
CARD_RANK_IDS = tuple(
    -1 if card.startswith("X") else "23456789TJQKA".index(card[0])
    for card in ALL_CARDS
)
RANK_VALUE_BY_LABEL = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "X": 15,
}
JOKER_INDICES = tuple(i for i, card in enumerate(ALL_CARDS) if card.startswith("X"))
BOARD_LOCATION_INDICES = (
    LOC_MY_TOP,
    LOC_MY_MID,
    LOC_MY_BOT,
    LOC_OPP_TOP,
    LOC_OPP_MID,
    LOC_OPP_BOT,
)


def _card_matrix_from_state(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] < 54 * 9:
        raise ValueError("state is too short to contain card-location features")
    return state[..., : 54 * 9].reshape(-1, 54, 9)


def _location_cards(matrix: torch.Tensor, location: int) -> torch.Tensor:
    return matrix[:, :, int(location)] > 0.5


def _top_has_pair_or_joker(matrix: torch.Tensor, location: int) -> torch.Tensor:
    cards = _location_cards(matrix, location)
    joker_idx = torch.tensor(JOKER_INDICES, dtype=torch.long, device=matrix.device)
    has_joker = cards.index_select(1, joker_idx).any(dim=1)
    rank_ids = torch.tensor(CARD_RANK_IDS, dtype=torch.long, device=matrix.device)
    has_pair = torch.zeros(cards.shape[0], dtype=torch.bool, device=matrix.device)
    for rank in range(13):
        rank_mask = rank_ids == rank
        has_pair = has_pair | (cards[:, rank_mask].sum(dim=1) >= 2)
    return has_joker | has_pair


def encoded_state_gate_mask(state: torch.Tensor, gate: str = "target_like_strict_state") -> torch.Tensor:
    """Return a per-state mask for conditional specialist blending.

    The runtime scorer receives post-action encoded states, not the original
    JSON rows.  These gates therefore use only card-location features that are
    available after the candidate action has been applied.
    """
    gate = str(gate or "target_like_strict_state")
    matrix = _card_matrix_from_state(state)
    joker_idx = torch.tensor(JOKER_INDICES, dtype=torch.long, device=matrix.device)
    visible_locations = torch.tensor(BOARD_LOCATION_INDICES, dtype=torch.long, device=matrix.device)
    visible_joker = matrix.index_select(1, joker_idx).index_select(2, visible_locations).amax(dim=(1, 2)) > 0.5
    in_hand_or_discard_joker = (
        matrix.index_select(1, joker_idx)[:, :, [LOC_IN_HAND, LOC_MY_DISCARD]].amax(dim=(1, 2)) > 0.5
    )
    own_top_cards = _location_cards(matrix, LOC_MY_TOP)
    own_middle_cards = _location_cards(matrix, LOC_MY_MID)
    opp_top_cards = _location_cards(matrix, LOC_OPP_TOP)
    own_top_len = own_top_cards.sum(dim=1)
    own_middle_len = own_middle_cards.sum(dim=1)
    opp_top_len = opp_top_cards.sum(dim=1)
    own_top_structured = _top_has_pair_or_joker(matrix, LOC_MY_TOP)
    opp_top_structured = _top_has_pair_or_joker(matrix, LOC_OPP_TOP)

    if gate == "all":
        return torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    if gate == "visible_joker_state":
        return visible_joker & ~in_hand_or_discard_joker
    if gate == "top_structured_state":
        return (own_top_len >= 2) & (own_top_structured | opp_top_structured)
    if gate == "top_single_unpaired_state":
        return (own_top_len == 1) & ~own_top_structured
    if gate == "opp_top_len_le_1_state":
        return opp_top_len <= 1
    if gate == "opp_top_len_le_1_dealt_max_rank_ge_j_state":
        return opp_top_len <= 1
    if gate == "own_middle_len_ge_3_state":
        return own_middle_len >= 3
    if gate == "opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3_state":
        return (opp_top_len <= 1) & (own_middle_len >= 3)
    if gate == "target_like_strict_state":
        return visible_joker & ~in_hand_or_discard_joker & (own_top_len >= 2) & (
            own_top_structured | opp_top_structured
        )
    raise ValueError(f"Unknown encoded state gate: {gate}")


def _rank_label(card: str) -> str:
    card = str(card)
    if card.startswith("X"):
        return "X"
    return card[0] if card else ""


def _rank_value(card: str) -> int:
    return int(RANK_VALUE_BY_LABEL.get(_rank_label(card), 0))


def _cards_have_joker(cards: Sequence[str]) -> bool:
    return any(str(card).startswith("X") for card in cards or [])


def _cards_have_pair_or_joker(cards: Sequence[str]) -> bool:
    cards = list(cards or [])
    if _cards_have_joker(cards):
        return True
    ranks = [_rank_label(card) for card in cards]
    return len(ranks) != len(set(ranks))


def _board_cards(board: Any) -> list[str]:
    if board is None:
        return []
    return list(getattr(board, "top", []) or []) + list(getattr(board, "middle", []) or []) + list(
        getattr(board, "bottom", []) or []
    )


def context_gate_accepts(obs: Any, gate: str = "target_like_strict_context") -> bool:
    """Gate from the original observation before candidate actions are applied."""
    gate = str(gate or "target_like_strict_context")
    if gate == "all":
        return True
    board = getattr(obs, "board_self", None)
    opponent = getattr(obs, "board_opponent", None)
    own_top = list(getattr(board, "top", []) or [])
    own_middle = list(getattr(board, "middle", []) or [])
    opp_top = list(getattr(opponent, "top", []) or [])
    dealt = list(getattr(obs, "dealt_cards", []) or [])
    visible = _board_cards(board) + _board_cards(opponent)
    visible_joker = _cards_have_joker(visible)
    dealt_joker = _cards_have_joker(dealt)
    own_top_structured = _cards_have_pair_or_joker(own_top)
    opp_top_structured = _cards_have_pair_or_joker(opp_top)
    if gate in {"opp_top_len_le_1", "opp_top_len_le_1_context"}:
        return bool(len(opp_top) <= 1)
    if gate in {"opp_top_len_le_1_dealt_max_rank_ge_j", "opp_top_len_le_1_dealt_max_rank_ge_j_context"}:
        return bool(len(opp_top) <= 1 and max([_rank_value(card) for card in dealt] or [0]) >= 11)
    if gate in {"own_middle_len_ge_3", "own_middle_len_ge_3_context"}:
        return bool(len(own_middle) >= 3)
    if gate in {
        "opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3",
        "opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3_context",
    }:
        return bool(
            len(opp_top) <= 1
            and max([_rank_value(card) for card in dealt] or [0]) >= 11
            and len(own_middle) >= 3
        )
    if gate in {"top_single_unpaired", "top_single_unpaired_context"}:
        return bool(len(own_top) == 1 and not own_top_structured)
    if gate in {"target_like_strict", "target_like_strict_context"}:
        return bool(
            visible_joker
            and not dealt_joker
            and len(own_top) >= 2
            and (own_top_structured or opp_top_structured)
        )
    if gate in {"target_like_v1", "target_like_v1_context"}:
        return bool(
            visible_joker
            and not dealt_joker
            and (len(own_top) >= 2 or own_top_structured or opp_top_structured)
        )
    raise ValueError(f"Unknown context gate: {gate}")


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class AdapterBlock(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(1, int(bottleneck))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
            nn.Dropout(dropout),
        )
        # Preserve checkpoint behavior at initialization; the adapter learns a
        # turn-specific residual only when fine-tuned.
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionValueReranker(nn.Module):
    """Predicts candidate EV, bust probability, and FL entry/type probability.

    The score head is trained on normalized EV.  Checkpoints store
    ``score_mean`` and ``score_std`` so inference can unnormalize the value.
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.1,
        turn_specific_heads: bool = False,
        turn_specific_adapters: bool = False,
        adapter_dim: int = 64,
        num_turns: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.turn_specific_heads = bool(turn_specific_heads)
        self.turn_specific_adapters = bool(turn_specific_adapters)
        self.num_turns = int(num_turns)
        self.score_mean = 0.0
        self.score_std = 1.0

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        head_dim = hidden // 2
        if self.turn_specific_adapters:
            self.turn_adapters = nn.ModuleList(
                [AdapterBlock(head_dim, adapter_dim, dropout) for _ in range(self.num_turns)]
            )
        self.score_head = nn.Linear(head_dim, 1)
        self.bust_head = nn.Linear(head_dim, 1)
        self.fl_head = nn.Linear(head_dim, 1)
        self.fl_type_head = nn.Linear(head_dim, len(FL_TYPE_KEYS))
        if self.turn_specific_heads:
            self.turn_score_heads = nn.ModuleList(
                [nn.Linear(head_dim, 1) for _ in range(self.num_turns)]
            )
            self.turn_bust_heads = nn.ModuleList(
                [nn.Linear(head_dim, 1) for _ in range(self.num_turns)]
            )
            self.turn_fl_heads = nn.ModuleList(
                [nn.Linear(head_dim, 1) for _ in range(self.num_turns)]
            )
            self.turn_fl_type_heads = nn.ModuleList(
                [nn.Linear(head_dim, len(FL_TYPE_KEYS)) for _ in range(self.num_turns)]
            )

    def infer_turn(self, state: torch.Tensor) -> torch.Tensor:
        """Infer turn index from encoded state metadata.

        The encoder stores ``turn / 4`` at feature index 486.  This keeps
        inference compatible with callers that only have candidate states.
        """
        if state.shape[-1] <= 486:
            return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
        turn = torch.round(state[..., 486] * 4.0).long()
        return turn.clamp(min=0, max=max(self.num_turns - 1, 0))

    def initialize_turn_heads_from_global(self) -> None:
        """Copy global head weights into each turn-specific head."""
        if not self.turn_specific_heads:
            return
        for head in self.turn_score_heads:
            head.load_state_dict(self.score_head.state_dict())
        for head in self.turn_bust_heads:
            head.load_state_dict(self.bust_head.state_dict())
        for head in self.turn_fl_heads:
            head.load_state_dict(self.fl_head.state_dict())
        for head in self.turn_fl_type_heads:
            head.load_state_dict(self.fl_type_head.state_dict())

    def _apply_turn_heads(
        self,
        features: torch.Tensor,
        turn: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        score = torch.empty(features.shape[0], dtype=features.dtype, device=features.device)
        bust = torch.empty_like(score)
        fl = torch.empty_like(score)
        fl_types = torch.empty(
            features.shape[0],
            len(FL_TYPE_KEYS),
            dtype=features.dtype,
            device=features.device,
        )
        for turn_id in range(self.num_turns):
            mask = turn == turn_id
            if not bool(mask.any()):
                continue
            selected = features[mask]
            score[mask] = self.turn_score_heads[turn_id](selected).squeeze(-1)
            bust[mask] = self.turn_bust_heads[turn_id](selected).squeeze(-1)
            fl[mask] = self.turn_fl_heads[turn_id](selected).squeeze(-1)
            fl_types[mask] = self.turn_fl_type_heads[turn_id](selected)
        return {
            "score": score,
            "bust_logit": bust,
            "fl_logit": fl,
            "fl_type_logits": fl_types,
        }

    def forward(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if turn is None and self.turn_specific_heads:
            turn = self.infer_turn(state)
        if turn is None and self.turn_specific_adapters:
            turn = self.infer_turn(state)
        state = _adapt_state(state, self.input_proj[0].in_features)
        x = self.input_proj(state)
        for block in self.blocks:
            x = block(x)
        x = self.trunk(x)
        if self.turn_specific_adapters:
            if turn is None:
                turn = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            turn = turn.to(device=x.device, dtype=torch.long).clamp(
                min=0,
                max=max(self.num_turns - 1, 0),
            )
            adapted = torch.empty_like(x)
            for turn_id in range(self.num_turns):
                mask = turn == turn_id
                if not bool(mask.any()):
                    continue
                adapted[mask] = self.turn_adapters[turn_id](x[mask])
            x = adapted
        if self.turn_specific_heads:
            if turn is None:
                turn = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            turn = turn.to(device=x.device, dtype=torch.long).clamp(
                min=0,
                max=max(self.num_turns - 1, 0),
            )
            return self._apply_turn_heads(x, turn)
        return {
            "score": self.score_head(x).squeeze(-1),
            "bust_logit": self.bust_head(x).squeeze(-1),
            "fl_logit": self.fl_head(x).squeeze(-1),
            "fl_type_logits": self.fl_type_head(x),
        }

    def predict_components(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(state, turn=turn)
        score = out["score"] * float(self.score_std) + float(self.score_mean)
        return {
            "score": score,
            "bust_prob": torch.sigmoid(out["bust_logit"]),
            "fl_prob": torch.sigmoid(out["fl_logit"]),
            "fl_type_probs": torch.sigmoid(out["fl_type_logits"]),
            "fl_qq": torch.sigmoid(out["fl_type_logits"][:, 0]),
            "fl_kk": torch.sigmoid(out["fl_type_logits"][:, 1]),
            "fl_aa": torch.sigmoid(out["fl_type_logits"][:, 2]),
            "fl_trips": torch.sigmoid(out["fl_type_logits"][:, 3]),
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: Optional[str | torch.device] = None,
    ) -> "ActionValueReranker":
        ckpt: Any = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            config = ckpt.get("model_config", {})
            model = cls(
                input_dim=int(ckpt.get("input_dim", config.get("input_dim", STATE_DIM))),
                hidden=int(config.get("hidden", 512)),
                n_blocks=int(config.get("n_blocks", 3)),
                dropout=float(config.get("dropout", 0.1)),
                turn_specific_heads=bool(config.get("turn_specific_heads", False)),
                turn_specific_adapters=bool(config.get("turn_specific_adapters", False)),
                adapter_dim=int(config.get("adapter_dim", 64)),
                num_turns=int(config.get("num_turns", 5)),
            )
            # v1 checkpoints do not have fl_type_head.  Keep them loadable so
            # old candidate runs can still be evaluated with type weights = 0.
            missing, unexpected = model.load_state_dict(
                ckpt["model_state_dict"],
                strict=False,
            )
            if unexpected:
                raise ValueError(f"Unexpected keys in checkpoint {path}: {unexpected}")
            norm = ckpt.get("normalization", {})
            model.score_mean = float(norm.get("score_mean", ckpt.get("score_mean", 0.0)))
            model.score_std = float(norm.get("score_std", ckpt.get("score_std", 1.0)))
            return model

        # Backward-compatible raw state_dict path.
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported checkpoint format: {path}")
        first_weight = ckpt.get("input_proj.0.weight")
        input_dim = int(first_weight.shape[1]) if first_weight is not None else STATE_DIM
        model = cls(input_dim=input_dim)
        model.load_state_dict(ckpt, strict=False)
        return model


class BlendedActionValueReranker(nn.Module):
    """Inference-only linear blend of action-value rerankers.

    This keeps independently trained checkpoints available at runtime
    without baking the blend into a new checkpoint.  The blend is applied to
    unnormalized EV scores and to probability components returned by
    ``predict_components``.
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        *,
        model_b_weight: float = 0.25,
        models: Optional[list[nn.Module]] = None,
        weights: Optional[list[float]] = None,
    ):
        super().__init__()
        if models is None:
            model_b_weight = float(model_b_weight)
            if not 0.0 <= model_b_weight <= 1.0:
                raise ValueError("model_b_weight must be between 0 and 1")
            models = [model_a, model_b]
            weights = [1.0 - model_b_weight, model_b_weight]
        if weights is None:
            raise ValueError("weights are required when models are provided")
        if len(models) < 2:
            raise ValueError("at least two models are required for a blend")
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length")
        weight_sum = float(sum(float(weight) for weight in weights))
        if weight_sum <= 0.0:
            raise ValueError("blend weights must sum to a positive value")
        normalized_weights = [float(weight) / weight_sum for weight in weights]
        if any(weight < 0.0 for weight in normalized_weights):
            raise ValueError("blend weights must be non-negative")

        self.models = nn.ModuleList(models)
        self.register_buffer(
            "weights",
            torch.tensor(normalized_weights, dtype=torch.float32),
            persistent=False,
        )
        self.model_a = self.models[0]
        self.model_b = self.models[1]
        self.model_a_weight = float(normalized_weights[0])
        self.model_b_weight = float(normalized_weights[1])
        self.input_dim = max(int(getattr(model, "input_dim", STATE_DIM)) for model in self.models)

    @classmethod
    def from_checkpoints(
        cls,
        model_a_path: str | Path,
        model_b_path: str | Path,
        *,
        model_b_weight: float = 0.25,
        map_location: Optional[str | torch.device] = None,
    ) -> "BlendedActionValueReranker":
        return cls(
            ActionValueReranker.from_checkpoint(model_a_path, map_location=map_location),
            ActionValueReranker.from_checkpoint(model_b_path, map_location=map_location),
            model_b_weight=model_b_weight,
        )

    @classmethod
    def from_weighted_checkpoints(
        cls,
        paths: list[str | Path],
        weights: list[float],
        *,
        map_location: Optional[str | torch.device] = None,
    ) -> "BlendedActionValueReranker":
        models = [
            ActionValueReranker.from_checkpoint(path, map_location=map_location)
            for path in paths
        ]
        return cls(models[0], models[1], models=models, weights=weights)

    @staticmethod
    def _predict(
        model: nn.Module,
        state: torch.Tensor,
        turn: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if hasattr(model, "predict_components"):
            if turn is None:
                return model.predict_components(state)
            try:
                return model.predict_components(state, turn=turn)
            except TypeError:
                return model.predict_components(state)
        raw = model(state)
        score = raw["score"] if isinstance(raw, dict) else raw.squeeze(-1)
        zeros = torch.zeros_like(score)
        return {
            "score": score,
            "bust_prob": zeros,
            "fl_prob": zeros,
            "fl_type_probs": torch.zeros((score.shape[0], len(FL_TYPE_KEYS)), dtype=score.dtype, device=score.device),
            "fl_qq": zeros,
            "fl_kk": zeros,
            "fl_aa": zeros,
            "fl_trips": zeros,
        }

    def forward(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.predict_components(state, turn=turn)

    def predict_components(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = [self._predict(model, state, turn) for model in self.models]

        def blend(key: str, fallback: torch.Tensor) -> torch.Tensor:
            result: Optional[torch.Tensor] = None
            for index, out in enumerate(outputs):
                value = out.get(key, fallback)
                weight = self.weights[index].to(dtype=value.dtype, device=value.device)
                term = weight * value
                result = term if result is None else result + term
            if result is None:
                return fallback
            return result

        score_fallback = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        type_fallback = torch.zeros(
            (state.shape[0], len(FL_TYPE_KEYS)),
            dtype=state.dtype,
            device=state.device,
        )
        score = blend("score", score_fallback)
        bust = blend("bust_prob", score_fallback)
        fl = blend("fl_prob", score_fallback)
        fl_types = blend("fl_type_probs", type_fallback)
        return {
            "score": score,
            "bust_prob": bust,
            "fl_prob": fl,
            "fl_type_probs": fl_types,
            "fl_qq": fl_types[:, 0],
            "fl_kk": fl_types[:, 1],
            "fl_aa": fl_types[:, 2],
            "fl_trips": fl_types[:, 3],
        }


class ConditionalBlendedActionValueReranker(nn.Module):
    """Blend a specialist only for encoded states that match a feature gate."""

    def __init__(
        self,
        base_model: nn.Module,
        specialist_model: nn.Module,
        *,
        specialist_weight: float,
        gate: str = "target_like_strict_state",
    ):
        super().__init__()
        specialist_weight = float(specialist_weight)
        if not 0.0 <= specialist_weight <= 1.0:
            raise ValueError("specialist_weight must be between 0 and 1")
        self.base_model = base_model
        self.specialist_model = specialist_model
        self.specialist_weight = specialist_weight
        self.gate = str(gate or "target_like_strict_state")
        self.input_dim = max(
            int(getattr(base_model, "input_dim", STATE_DIM)),
            int(getattr(specialist_model, "input_dim", STATE_DIM)),
        )

    @classmethod
    def from_checkpoints(
        cls,
        base_paths: list[str | Path],
        base_weights: list[float],
        specialist_path: str | Path,
        *,
        specialist_weight: float,
        gate: str = "target_like_strict_state",
        map_location: Optional[str | torch.device] = None,
    ) -> "ConditionalBlendedActionValueReranker":
        if len(base_paths) == 1:
            base_model: nn.Module = ActionValueReranker.from_checkpoint(
                base_paths[0],
                map_location=map_location,
            )
        else:
            base_model = BlendedActionValueReranker.from_weighted_checkpoints(
                list(base_paths),
                list(base_weights),
                map_location=map_location,
            )
        specialist_model = ActionValueReranker.from_checkpoint(
            specialist_path,
            map_location=map_location,
        )
        return cls(
            base_model,
            specialist_model,
            specialist_weight=specialist_weight,
            gate=gate,
        )

    def forward(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.predict_components(state, turn=turn)

    def predict_components(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gate = self.gate
        if gate.endswith("_context"):
            gate = f"{gate[: -len('_context')]}_state"
        elif not gate.endswith("_state"):
            gate = f"{gate}_state"
        return self.predict_components_with_gate_mask(
            state,
            gate_mask=encoded_state_gate_mask(state, gate),
            turn=turn,
        )

    def candidate_gate_mask(self, obs: Any, post_action_obs: Sequence[Any]) -> list[bool] | None:
        if self.gate.endswith("_state"):
            return None
        accepted = context_gate_accepts(obs, self.gate)
        return [accepted] * len(post_action_obs)

    def predict_components_with_gate_mask(
        self,
        state: torch.Tensor,
        *,
        gate_mask: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base = BlendedActionValueReranker._predict(self.base_model, state, turn)
        gate_mask = gate_mask.to(device=state.device, dtype=torch.bool)
        if not bool(gate_mask.any()) or self.specialist_weight <= 0.0:
            return base

        selected_state = state[gate_mask]
        selected_turn = turn[gate_mask] if turn is not None else None
        specialist = BlendedActionValueReranker._predict(
            self.specialist_model,
            selected_state,
            selected_turn,
        )
        w = float(self.specialist_weight)
        out: Dict[str, torch.Tensor] = {}
        for key, base_value in base.items():
            if key not in specialist:
                out[key] = base_value
                continue
            blended = base_value.clone()
            spec_value = specialist[key]
            blended[gate_mask] = (1.0 - w) * blended[gate_mask] + w * spec_value
            out[key] = blended
        fl_types = out.get("fl_type_probs")
        if fl_types is not None and fl_types.shape[-1] >= len(FL_TYPE_KEYS):
            out["fl_qq"] = fl_types[:, 0]
            out["fl_kk"] = fl_types[:, 1]
            out["fl_aa"] = fl_types[:, 2]
            out["fl_trips"] = fl_types[:, 3]
        return out


class ConditionalSwitchActionValueReranker(nn.Module):
    """Use a challenger action-value model only for states that match a gate."""

    def __init__(
        self,
        base_model: nn.Module,
        challenger_model: nn.Module,
        *,
        gate: str = "target_like_strict_state",
    ):
        super().__init__()
        self.base_model = base_model
        self.challenger_model = challenger_model
        self.gate = str(gate or "target_like_strict_state")
        self.input_dim = max(
            int(getattr(base_model, "input_dim", STATE_DIM)),
            int(getattr(challenger_model, "input_dim", STATE_DIM)),
        )

    @classmethod
    def from_checkpoints(
        cls,
        base_paths: list[str | Path],
        base_weights: list[float],
        challenger_paths: list[str | Path],
        challenger_weights: list[float],
        *,
        gate: str = "target_like_strict_state",
        map_location: Optional[str | torch.device] = None,
    ) -> "ConditionalSwitchActionValueReranker":
        if len(base_paths) == 1:
            base_model: nn.Module = ActionValueReranker.from_checkpoint(
                base_paths[0],
                map_location=map_location,
            )
        else:
            base_model = BlendedActionValueReranker.from_weighted_checkpoints(
                list(base_paths),
                list(base_weights),
                map_location=map_location,
            )
        if len(challenger_paths) == 1:
            challenger_model: nn.Module = ActionValueReranker.from_checkpoint(
                challenger_paths[0],
                map_location=map_location,
            )
        else:
            challenger_model = BlendedActionValueReranker.from_weighted_checkpoints(
                list(challenger_paths),
                list(challenger_weights),
                map_location=map_location,
            )
        return cls(base_model, challenger_model, gate=gate)

    def forward(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.predict_components(state, turn=turn)

    def predict_components(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gate = self.gate
        if gate.endswith("_context"):
            gate = f"{gate[: -len('_context')]}_state"
        elif not gate.endswith("_state"):
            gate = f"{gate}_state"
        return self.predict_components_with_gate_mask(
            state,
            gate_mask=encoded_state_gate_mask(state, gate),
            turn=turn,
        )

    def candidate_gate_mask(self, obs: Any, post_action_obs: Sequence[Any]) -> list[bool] | None:
        if self.gate.endswith("_state"):
            return None
        accepted = context_gate_accepts(obs, self.gate)
        return [accepted] * len(post_action_obs)

    def predict_components_with_gate_mask(
        self,
        state: torch.Tensor,
        *,
        gate_mask: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base = BlendedActionValueReranker._predict(self.base_model, state, turn)
        gate_mask = gate_mask.to(device=state.device, dtype=torch.bool)
        if not bool(gate_mask.any()):
            return base

        challenger = BlendedActionValueReranker._predict(
            self.challenger_model,
            state[gate_mask],
            turn[gate_mask] if turn is not None else None,
        )
        out: Dict[str, torch.Tensor] = {}
        for key, base_value in base.items():
            if key not in challenger:
                out[key] = base_value
                continue
            switched = base_value.clone()
            switched[gate_mask] = challenger[key]
            out[key] = switched
        fl_types = out.get("fl_type_probs")
        if fl_types is not None and fl_types.shape[-1] >= len(FL_TYPE_KEYS):
            out["fl_qq"] = fl_types[:, 0]
            out["fl_kk"] = fl_types[:, 1]
            out["fl_aa"] = fl_types[:, 2]
            out["fl_trips"] = fl_types[:, 3]
        return out


class CascadeSwitchActionValueReranker(nn.Module):
    """Apply ordered challenger switches on top of a base action-value model."""

    def __init__(
        self,
        base_model: nn.Module,
        switches: Sequence[tuple[str, nn.Module]],
    ):
        super().__init__()
        if not switches:
            raise ValueError("at least one cascade switch is required")
        self.base_model = base_model
        self.gates = [str(gate or "all") for gate, _model in switches]
        self.switch_models = nn.ModuleList([model for _gate, model in switches])
        self.input_dim = max(
            [int(getattr(base_model, "input_dim", STATE_DIM))]
            + [int(getattr(model, "input_dim", STATE_DIM)) for model in self.switch_models]
        )

    @staticmethod
    def _load_weighted(
        paths: list[str | Path],
        weights: list[float],
        *,
        map_location: Optional[str | torch.device] = None,
    ) -> nn.Module:
        if len(paths) == 1:
            return ActionValueReranker.from_checkpoint(paths[0], map_location=map_location)
        return BlendedActionValueReranker.from_weighted_checkpoints(
            list(paths),
            list(weights),
            map_location=map_location,
        )

    @classmethod
    def from_checkpoints(
        cls,
        base_paths: list[str | Path],
        base_weights: list[float],
        switch_specs: Sequence[tuple[str, list[str | Path], list[float]]],
        *,
        map_location: Optional[str | torch.device] = None,
    ) -> "CascadeSwitchActionValueReranker":
        base_model = cls._load_weighted(base_paths, base_weights, map_location=map_location)
        switches = [
            (
                gate,
                cls._load_weighted(list(paths), list(weights), map_location=map_location),
            )
            for gate, paths, weights in switch_specs
        ]
        return cls(base_model, switches)

    @staticmethod
    def _state_gate_name(gate: str) -> str:
        if gate.endswith("_context"):
            return f"{gate[: -len('_context')]}_state"
        if gate.endswith("_state"):
            return gate
        return f"{gate}_state"

    def forward(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.predict_components(state, turn=turn)

    def predict_components(
        self,
        state: torch.Tensor,
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gate_masks = {
            gate: encoded_state_gate_mask(state, self._state_gate_name(gate))
            for gate in self.gates
        }
        return self.predict_components_with_gate_masks(state, gate_masks=gate_masks, turn=turn)

    def candidate_gate_masks(self, obs: Any, post_action_obs: Sequence[Any]) -> dict[str, list[bool]] | None:
        masks: dict[str, list[bool]] = {}
        for gate in self.gates:
            if gate.endswith("_state"):
                continue
            accepted = context_gate_accepts(obs, gate)
            masks[gate] = [accepted] * len(post_action_obs)
        return masks or None

    def predict_components_with_gate_masks(
        self,
        state: torch.Tensor,
        *,
        gate_masks: dict[str, torch.Tensor],
        turn: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = BlendedActionValueReranker._predict(self.base_model, state, turn)
        normalized_masks = {
            str(gate): mask.to(device=state.device, dtype=torch.bool)
            for gate, mask in gate_masks.items()
        }
        for gate, model in zip(self.gates, self.switch_models):
            mask = normalized_masks.get(gate)
            if mask is None:
                mask = encoded_state_gate_mask(state, self._state_gate_name(gate))
            if not bool(mask.any()):
                continue
            challenger = BlendedActionValueReranker._predict(
                model,
                state[mask],
                turn[mask] if turn is not None else None,
            )
            for key, base_value in list(out.items()):
                if key not in challenger:
                    continue
                switched = base_value.clone()
                switched[mask] = challenger[key]
                out[key] = switched

        fl_types = out.get("fl_type_probs")
        if fl_types is not None and fl_types.shape[-1] >= len(FL_TYPE_KEYS):
            out["fl_qq"] = fl_types[:, 0]
            out["fl_kk"] = fl_types[:, 1]
            out["fl_aa"] = fl_types[:, 2]
            out["fl_trips"] = fl_types[:, 3]
        return out
