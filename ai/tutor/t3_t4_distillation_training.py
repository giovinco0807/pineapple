"""Deterministic, opt-in T3/T4 policy/value/Q distillation training.

Only :mod:`ai.tutor.t3_t4_distillation_dataset` tensors enter optimization.
The locked ``fit`` split is the sole gradient source, ``dev`` is the sole
checkpoint-selection source, and ``test`` is evaluated exactly once after the
selected checkpoint has been restored.  Artifacts are content-addressed and
remain candidate-only: this module has no serving/runtime integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ai.tutor.t3_t4_distillation_dataset as _dataset_module
import ai.tutor.t3_t4_distillation_teacher as _teacher_module
import ai.tutor.t3_t4_infoset_encoder as _encoder_module
import ai.tutor.runtime_semantic_anchor as _runtime_anchor_module
from ai.tutor.t3_t4_distillation_dataset import (
    DISTILLATION_DATASET_SCHEMA,
    DistillationDatasetError,
    DistillationSplit,
    VerifiedDistillationDataset,
    load_distillation_dataset,
    verify_distillation_dataset,
)
from ai.tutor.t3_t4_distillation_teacher import canonical_json, canonical_sha256
from ai.tutor.t3_t4_infoset_encoder import (
    ACTION_SEMANTICS_SHA256,
    INFOSET_ENCODER_MANIFEST_SHA256,
    INFOSET_VECTOR_DIM,
    infoset_encoder_manifest,
    legal_action_mask,
    semantic_action_ids,
    validate_infoset_encoder_manifest,
)


TRAINING_CONFIG_SCHEMA = "ofc_t3_t4_policy_value_q_training_config/v1"
MODEL_CONFIG_SCHEMA = "ofc_t3_t4_policy_value_q_model_config/v1"
TRAINING_HISTORY_SCHEMA = "ofc_t3_t4_policy_value_q_training_history/v1"
TRAINING_METRICS_SCHEMA = "ofc_t3_t4_policy_value_q_metrics/v1"
TRAINING_ARTIFACT_SCHEMA = "ofc_t3_t4_policy_value_q_training_artifact/v1"
CHECKPOINT_SCHEMA = "ofc_t3_t4_policy_value_q_tensor_checkpoint/v1"
EVALUATION_SCHEMA = "ofc_t3_t4_policy_value_q_split_evaluation/v1"
EVENT_SCHEMA = "ofc_t3_t4_policy_value_q_training_event/v1"
CHECKPOINT_MAGIC = b"OFC-T3T4-PVQ-CHECKPOINT-V1\n"
ACTION_COUNT = 27
MANIFEST_PREFIX = "training-manifest-"
MAX_CHECKPOINT_HEADER_BYTES = 16 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024

_LOAD_DISTILLATION_DATASET = load_distillation_dataset
_VERIFY_DISTILLATION_DATASET = verify_distillation_dataset
_INFOSET_ENCODER_MANIFEST = infoset_encoder_manifest
_LEGAL_ACTION_MASK = legal_action_mask
_SEMANTIC_ACTION_IDS = semantic_action_ids
_VALIDATE_INFOSET_ENCODER_MANIFEST = validate_infoset_encoder_manifest
_CANONICAL_JSON = canonical_json
_CANONICAL_SHA256 = canonical_sha256
_ASSERT_TRANSITIVE_RUNTIME_ANCHORS = (
    _dataset_module._assert_transitive_runtime_anchors
)
_PINNED_SOURCE_PATHS = MappingProxyType(
    {
        "trainer": Path(__file__).resolve(),
        "model": Path(__file__).resolve(),
        "dataset": Path(_dataset_module.__file__).resolve(),
        "teacher": Path(_teacher_module.__file__).resolve(),
        "encoder": Path(_encoder_module.__file__).resolve(),
        "runtime_anchor": Path(_runtime_anchor_module.__file__).resolve(),
    }
)
_PINNED_SOURCE_SHA256 = MappingProxyType(
    {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in _PINNED_SOURCE_PATHS.items()
    }
)


class DistillationTrainingError(ValueError):
    """Training inputs, deterministic execution, or artifacts failed closed."""


@dataclass(frozen=True)
class DistillationTrainingConfig:
    seed: int = 20260713
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    residual_blocks: int = 2
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    q_loss_weight: float = 0.5
    q_standard_error_floor: float = 0.05
    q_precision_weight_cap: float = 100.0
    huber_delta: float = 1.0
    evaluation_batch_size: int = 512


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    action_count: int
    hidden_dim: int
    residual_blocks: int
    activation: str = "gelu"
    dtype: str = "float32"


@dataclass(frozen=True)
class VerifiedDistillationTrainingArtifact:
    root: Path
    manifest: dict[str, Any]
    history: dict[str, Any]
    metrics: dict[str, Any]
    dataset: VerifiedDistillationDataset
    model: "T3T4PolicyValueQNet"


def _strict_int(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise DistillationTrainingError(
            f"{label}: integer in [{minimum}, {maximum}] required"
        )
    return value


def _strict_float(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistillationTrainingError(f"{label}: finite number required")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < minimum
        or result > maximum
        or (positive and result <= 0.0)
    ):
        raise DistillationTrainingError(
            f"{label}: finite value in [{minimum}, {maximum}] required"
        )
    return 0.0 if result == 0.0 else result


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DistillationTrainingError(f"{label}: lowercase SHA-256 required")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _CANONICAL_SHA256(payload)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DistillationTrainingError(
            f"{label}: exact fields required; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _config_payload(config: DistillationTrainingConfig) -> dict[str, Any]:
    if not isinstance(config, DistillationTrainingConfig):
        raise TypeError("config must be DistillationTrainingConfig")
    payload = asdict(config)
    payload["seed"] = _strict_int(
        payload["seed"],
        label="seed",
        minimum=0,
        maximum=(1 << 63) - 10_001,
    )
    payload["epochs"] = _strict_int(payload["epochs"], label="epochs", minimum=1, maximum=10_000)
    payload["batch_size"] = _strict_int(
        payload["batch_size"], label="batch_size", minimum=1, maximum=4096
    )
    payload["hidden_dim"] = _strict_int(
        payload["hidden_dim"], label="hidden_dim", minimum=4, maximum=2048
    )
    payload["residual_blocks"] = _strict_int(
        payload["residual_blocks"], label="residual_blocks", minimum=0, maximum=8
    )
    payload["evaluation_batch_size"] = _strict_int(
        payload["evaluation_batch_size"],
        label="evaluation_batch_size",
        minimum=1,
        maximum=4096,
    )
    for field, minimum, maximum, positive in (
        ("learning_rate", 0.0, 1.0, True),
        ("weight_decay", 0.0, 100.0, False),
        ("policy_loss_weight", 0.0, 1_000.0, False),
        ("value_loss_weight", 0.0, 1_000.0, False),
        ("q_loss_weight", 0.0, 1_000.0, False),
        ("q_standard_error_floor", 0.0, 1_000_000.0, True),
        ("q_precision_weight_cap", 0.0, 1_000_000_000.0, True),
        ("huber_delta", 0.0, 1_000_000.0, True),
    ):
        payload[field] = _strict_float(
            payload[field],
            label=field,
            minimum=minimum,
            maximum=maximum,
            positive=positive,
        )
    if (
        payload["policy_loss_weight"]
        + payload["value_loss_weight"]
        + payload["q_loss_weight"]
        <= 0.0
    ):
        raise DistillationTrainingError("at least one loss weight must be positive")
    return {"schema": TRAINING_CONFIG_SCHEMA, **payload}


def _config_from_payload(payload: Mapping[str, Any]) -> DistillationTrainingConfig:
    raw = dict(payload)
    if raw.pop("schema", None) != TRAINING_CONFIG_SCHEMA:
        raise DistillationTrainingError("training config schema mismatch")
    try:
        config = DistillationTrainingConfig(**raw)
    except TypeError as exc:
        raise DistillationTrainingError("training config fields mismatch") from exc
    if _config_payload(config) != dict(payload):
        raise DistillationTrainingError("training config is noncanonical")
    return config


def _model_config(config: DistillationTrainingConfig) -> ModelConfig:
    _config_payload(config)
    return ModelConfig(
        input_dim=INFOSET_VECTOR_DIM,
        action_count=ACTION_COUNT,
        hidden_dim=config.hidden_dim,
        residual_blocks=config.residual_blocks,
    )


def _model_config_payload(config: ModelConfig) -> dict[str, Any]:
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be ModelConfig")
    payload = {"schema": MODEL_CONFIG_SCHEMA, **asdict(config)}
    fixed = {
        "input_dim": INFOSET_VECTOR_DIM,
        "action_count": ACTION_COUNT,
        "activation": "gelu",
        "dtype": "float32",
    }
    for field, expected in fixed.items():
        if payload[field] != expected:
            raise DistillationTrainingError(f"model config {field} mismatch")
    _strict_int(payload["hidden_dim"], label="model hidden_dim", minimum=4, maximum=2048)
    _strict_int(
        payload["residual_blocks"],
        label="model residual_blocks",
        minimum=0,
        maximum=8,
    )
    return payload


def _model_config_from_payload(payload: Mapping[str, Any]) -> ModelConfig:
    raw = dict(payload)
    if raw.pop("schema", None) != MODEL_CONFIG_SCHEMA:
        raise DistillationTrainingError("model config schema mismatch")
    try:
        config = ModelConfig(**raw)
    except TypeError as exc:
        raise DistillationTrainingError("model config fields mismatch") from exc
    if _model_config_payload(config) != dict(payload):
        raise DistillationTrainingError("model config is noncanonical")
    return config


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.norm(value)
        value = F.gelu(self.fc1(value), approximate="none")
        value = self.fc2(value)
        return residual + value


class T3T4PolicyValueQNet(nn.Module):
    """One shared hidden-safe model for all late-turn seats and phases."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        _model_config_payload(config)
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(config.hidden_dim) for _ in range(config.residual_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(config.hidden_dim)
        self.policy_head = nn.Linear(config.hidden_dim, config.action_count)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.q_head = nn.Linear(config.hidden_dim, config.action_count)

    def forward(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        if states.ndim != 2 or states.shape[1] != self.config.input_dim:
            raise ValueError(
                f"states must have shape [N,{self.config.input_dim}]"
            )
        if states.device.type != "cpu" or states.dtype != torch.float32:
            raise ValueError("deterministic distillation model requires CPU float32 input")
        hidden = F.gelu(
            self.input_norm(self.input_projection(states)), approximate="none"
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.trunk_norm(hidden)
        return {
            "policy_logits": self.policy_head(hidden),
            "value": self.value_head(hidden).squeeze(-1),
            "q_values": self.q_head(hidden),
        }


def _loss_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    required = {
        "state",
        "legal_action_mask",
        "policy_target",
        "value_target",
        "q_target",
        "q_target_mask",
        "q_standard_error",
    }
    if not required.issubset(batch):
        raise DistillationTrainingError(
            f"loss batch missing fields {sorted(required - set(batch))}"
        )
    result = {
        "state": torch.as_tensor(batch["state"], dtype=torch.float32, device="cpu"),
        "legal_action_mask": torch.as_tensor(
            batch["legal_action_mask"], dtype=torch.bool, device="cpu"
        ),
        "policy_target": torch.as_tensor(
            batch["policy_target"], dtype=torch.float32, device="cpu"
        ),
        "value_target": torch.as_tensor(
            batch["value_target"], dtype=torch.float32, device="cpu"
        ),
        "q_target": torch.as_tensor(batch["q_target"], dtype=torch.float32, device="cpu"),
        "q_target_mask": torch.as_tensor(
            batch["q_target_mask"], dtype=torch.bool, device="cpu"
        ),
        "q_standard_error": torch.as_tensor(
            batch["q_standard_error"], dtype=torch.float32, device="cpu"
        ),
    }
    return result


def masked_policy_value_q_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    config: DistillationTrainingConfig,
) -> dict[str, torch.Tensor]:
    """Masked soft-policy, scalar-value, and precision-weighted Q losses."""

    config_payload = _config_payload(config)
    tensors = _loss_batch(batch)
    states = tensors["state"]
    row_count = states.shape[0]
    expected_action_shape = (row_count, ACTION_COUNT)
    for field in (
        "legal_action_mask",
        "policy_target",
        "q_target",
        "q_target_mask",
        "q_standard_error",
    ):
        if tuple(tensors[field].shape) != expected_action_shape:
            raise DistillationTrainingError(f"{field}: shape mismatch")
    if tuple(tensors["value_target"].shape) != (row_count,):
        raise DistillationTrainingError("value_target: shape mismatch")
    if not torch.equal(tensors["legal_action_mask"], tensors["q_target_mask"]):
        raise DistillationTrainingError("Q mask must equal legal-action mask")
    mask = tensors["legal_action_mask"]
    if row_count == 0 or not bool(mask.any(dim=1).all()):
        raise DistillationTrainingError("nonempty rows with legal actions required")
    if not bool(torch.isfinite(states).all()) or not bool(
        ((states == 0.0) | (states == 1.0)).all()
    ):
        raise DistillationTrainingError("states must remain finite exact binary vectors")
    if not bool(torch.isfinite(tensors["value_target"]).all()):
        raise DistillationTrainingError("value targets must be finite float32")
    policy_target = tensors["policy_target"]
    if not bool(torch.isfinite(policy_target).all()) or bool(
        (policy_target < 0.0).any()
    ):
        raise DistillationTrainingError("policy targets must be finite and nonnegative")
    if bool((policy_target.masked_select(~mask) != 0.0).any()):
        raise DistillationTrainingError("illegal actions have policy mass")
    if not torch.allclose(
        policy_target.sum(dim=1),
        torch.ones(row_count, dtype=torch.float32),
        rtol=0.0,
        atol=1e-6,
    ):
        raise DistillationTrainingError("policy target rows must sum to one")
    for field, shape in (
        ("policy_logits", expected_action_shape),
        ("q_values", expected_action_shape),
        ("value", (row_count,)),
    ):
        tensor = outputs.get(field)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise DistillationTrainingError(f"model output {field}: shape mismatch")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise DistillationTrainingError(
                f"model output {field}: CPU float32 required"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise DistillationTrainingError(f"model output {field}: non-finite value")

    masked_logits = outputs["policy_logits"].masked_fill(~mask, -1e30)
    log_policy = F.log_softmax(masked_logits, dim=1)
    policy_per_row = -(policy_target * log_policy).sum(dim=1)
    policy_loss = policy_per_row.mean()

    value_loss = F.smooth_l1_loss(
        outputs["value"],
        tensors["value_target"],
        beta=config_payload["huber_delta"],
        reduction="mean",
    )

    safe_q_target = torch.where(mask, tensors["q_target"], torch.zeros_like(tensors["q_target"]))
    safe_se = torch.where(
        mask,
        tensors["q_standard_error"],
        torch.ones_like(tensors["q_standard_error"]),
    )
    if not bool(torch.isfinite(safe_q_target).all()):
        raise DistillationTrainingError("legal Q targets must be finite")
    if not bool(torch.isfinite(safe_se).all()) or bool((safe_se[mask] < 0.0).any()):
        raise DistillationTrainingError("legal Q standard errors are invalid")
    precision = 1.0 / torch.clamp(
        safe_se.square()
        + config_payload["q_standard_error_floor"]
        * config_payload["q_standard_error_floor"],
        min=torch.finfo(torch.float32).tiny,
    )
    precision = torch.clamp(
        precision,
        max=config_payload["q_precision_weight_cap"],
    ).masked_fill(~mask, 0.0)
    q_element = F.smooth_l1_loss(
        outputs["q_values"],
        safe_q_target,
        beta=config_payload["huber_delta"],
        reduction="none",
    ).masked_fill(~mask, 0.0)
    q_loss = (q_element * precision).sum() / precision.sum().clamp(min=1e-12)
    total = (
        config_payload["policy_loss_weight"] * policy_loss
        + config_payload["value_loss_weight"] * value_loss
        + config_payload["q_loss_weight"] * q_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "q": q_loss,
    }


@contextmanager
def _deterministic_cpu(seed: int) -> Iterator[None]:
    seed = _strict_int(seed, label="deterministic seed", minimum=0, maximum=(1 << 63) - 1)
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_rng = torch.random.get_rng_state()
    prior_mkldnn = bool(torch.backends.mkldnn.enabled)
    prior_precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else None
    )
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(prior_rng)
        torch.backends.mkldnn.enabled = prior_mkldnn
        torch.use_deterministic_algorithms(prior_deterministic)
        if prior_precision is not None:
            torch.set_float32_matmul_precision(prior_precision)
        torch.set_num_threads(prior_threads)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _CANONICAL_JSON(
        {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_state_sha256(model: T3T4PolicyValueQNet) -> str:
    entries: list[dict[str, Any]] = []
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise DistillationTrainingError("model state must be CPU float32")
        array = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
        if not np.all(np.isfinite(array)):
            raise DistillationTrainingError("model state contains non-finite values")
        entries.append(
            {
                "name": name,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return _CANONICAL_SHA256(entries)


def _predict_split(
    model: T3T4PolicyValueQNet,
    split: DistillationSplit,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = _strict_int(
        batch_size,
        label="evaluation batch_size",
        minimum=1,
        maximum=4096,
    )
    if len(split) == 0:
        raise DistillationTrainingError(f"locked {split.name} split is empty")
    if next(model.parameters()).device.type != "cpu":
        raise DistillationTrainingError("evaluation model must remain on CPU")
    model.eval()
    policy: list[np.ndarray] = []
    values: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(split), size):
            stop = min(start + size, len(split))
            states = torch.from_numpy(
                np.array(split.states[start:stop], dtype=np.float32, copy=True)
            )
            output = model(states)
            if any(not bool(torch.isfinite(value).all()) for value in output.values()):
                raise DistillationTrainingError(
                    "model emitted non-finite evaluation predictions"
                )
            policy.append(output["policy_logits"].detach().numpy().astype("<f4", copy=True))
            values.append(output["value"].detach().numpy().astype("<f4", copy=True))
            q_values.append(output["q_values"].detach().numpy().astype("<f4", copy=True))
    return (
        np.ascontiguousarray(np.concatenate(policy, axis=0), dtype="<f4"),
        np.ascontiguousarray(np.concatenate(values, axis=0), dtype="<f4"),
        np.ascontiguousarray(np.concatenate(q_values, axis=0), dtype="<f4"),
    )


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    work = logits.astype(np.float64, copy=True)
    work[~mask] = -np.inf
    row_max = np.max(work, axis=1, keepdims=True)
    exponent = np.exp(work - row_max)
    exponent[~mask] = 0.0
    return exponent / exponent.sum(axis=1, keepdims=True)


def _metric_payload(
    indices: np.ndarray,
    *,
    split: DistillationSplit,
    policy_logits: np.ndarray,
    value_predictions: np.ndarray,
    q_predictions: np.ndarray,
    config: DistillationTrainingConfig,
) -> dict[str, Any]:
    if indices.ndim != 1 or len(indices) == 0:
        raise DistillationTrainingError("metric group requires nonempty 1-D indices")
    mask = split.legal_action_masks[indices]
    policy_target = split.policy_targets[indices]
    q_target = split.q_targets[indices]
    q_mask = split.q_target_masks[indices]
    q_se = split.q_standard_errors[indices]
    logits = policy_logits[indices].astype(np.float64)
    probabilities = _softmax_masked(logits, mask)
    cross_entropy = -np.sum(
        policy_target * np.log(np.clip(probabilities, 1e-300, 1.0)),
        axis=1,
    )
    masked_logits = np.where(mask, logits, -np.inf)
    predicted_action = np.argmax(masked_logits, axis=1)
    row_positions = np.arange(len(indices))
    safe_q_target = np.where(q_mask, q_target, -np.inf)
    best_q = np.max(safe_q_target, axis=1)
    chosen_q = q_target[row_positions, predicted_action]
    regret = np.maximum(0.0, best_q - chosen_q)
    best_index = split.best_action_indices[indices]
    top1 = predicted_action == best_index
    value_error = value_predictions[indices].astype(np.float64) - split.value_targets[indices]
    q_error = q_predictions[indices].astype(np.float64) - np.where(q_mask, q_target, 0.0)
    legal_q_error = q_error[q_mask]
    config_payload = _config_payload(config)
    precision = 1.0 / np.maximum(
        np.square(np.where(q_mask, q_se, 0.0))
        + config_payload["q_standard_error_floor"] ** 2,
        np.finfo(np.float64).tiny,
    )
    precision = np.minimum(precision, config_payload["q_precision_weight_cap"])
    legal_precision = precision[q_mask]
    weighted_q_mse = float(
        np.sum(legal_precision * np.square(legal_q_error))
        / np.sum(legal_precision)
    )
    identities = [split.provenance.row_identity_sha256[int(index)] for index in indices]
    return {
        "row_count": int(len(indices)),
        "legal_action_count": int(np.sum(mask)),
        "row_identity_set_sha256": _CANONICAL_SHA256(sorted(identities)),
        "policy_cross_entropy_mean": float(np.mean(cross_entropy)),
        "policy_top1_accuracy": float(np.mean(top1)),
        "policy_ev_regret_mean": float(np.mean(regret)),
        "policy_ev_regret_max": float(np.max(regret)),
        "value_mae": float(np.mean(np.abs(value_error))),
        "value_rmse": float(np.sqrt(np.mean(np.square(value_error)))),
        "q_mae": float(np.mean(np.abs(legal_q_error))),
        "q_rmse": float(np.sqrt(np.mean(np.square(legal_q_error)))),
        "q_se_weighted_rmse": float(math.sqrt(weighted_q_mse)),
    }


def _group_metrics(
    labels: Sequence[Any],
    *,
    split: DistillationSplit,
    policy_logits: np.ndarray,
    value_predictions: np.ndarray,
    q_predictions: np.ndarray,
    config: DistillationTrainingConfig,
    formatter: Any = str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(formatter(label), []).append(index)
    return {
        key: _metric_payload(
            np.asarray(indices, dtype=np.int64),
            split=split,
            policy_logits=policy_logits,
            value_predictions=value_predictions,
            q_predictions=q_predictions,
            config=config,
        )
        for key, indices in sorted(groups.items())
    }


def _evaluate_split(
    model: T3T4PolicyValueQNet,
    split: DistillationSplit,
    config: DistillationTrainingConfig,
    *,
    include_groups: bool,
) -> dict[str, Any]:
    policy, values, q_values = _predict_split(
        model,
        split,
        batch_size=config.evaluation_batch_size,
    )
    all_indices = np.arange(len(split), dtype=np.int64)
    result: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "split": split.name,
        "row_count": len(split),
        "row_identity_order_sha256": _CANONICAL_SHA256(
            list(split.provenance.row_identity_sha256)
        ),
        "prediction_binding": {
            "policy_logits_sha256": _array_sha256(policy),
            "value_predictions_sha256": _array_sha256(values),
            "q_predictions_sha256": _array_sha256(q_values),
        },
        "overall": _metric_payload(
            all_indices,
            split=split,
            policy_logits=policy,
            value_predictions=values,
            q_predictions=q_values,
            config=config,
        ),
        "groups_included": include_groups,
    }
    if include_groups:
        provenance = split.provenance
        result.update(
            {
                "by_public_root_family": _group_metrics(
                    provenance.public_root_family_commitment_sha256,
                    split=split,
                    policy_logits=policy,
                    value_predictions=values,
                    q_predictions=q_values,
                    config=config,
                ),
                "by_phase": _group_metrics(
                    provenance.phase,
                    split=split,
                    policy_logits=policy,
                    value_predictions=values,
                    q_predictions=q_values,
                    config=config,
                ),
                "by_actor": _group_metrics(
                    provenance.actor,
                    split=split,
                    policy_logits=policy,
                    value_predictions=values,
                    q_predictions=q_values,
                    config=config,
                ),
                "by_joker": _group_metrics(
                    provenance.visible_joker_count,
                    split=split,
                    policy_logits=policy,
                    value_predictions=values,
                    q_predictions=q_values,
                    config=config,
                    formatter=lambda value: f"joker_{value}",
                ),
                "by_phase_actor_joker": _group_metrics(
                    tuple(
                        zip(
                            provenance.phase,
                            provenance.actor,
                            provenance.visible_joker_count,
                        )
                    ),
                    split=split,
                    policy_logits=policy,
                    value_predictions=values,
                    q_predictions=q_values,
                    config=config,
                    formatter=lambda value: f"{value[0]}_{value[1]}_joker_{value[2]}",
                ),
            }
        )
    result["evaluation_sha256"] = _self_hash(result, "evaluation_sha256")
    return result


def _selection_key(evaluation: Mapping[str, Any], epoch: int) -> tuple[float, ...]:
    overall = evaluation["overall"]
    return (
        float(overall["policy_ev_regret_mean"]),
        float(overall["policy_cross_entropy_mean"]),
        float(overall["value_rmse"]),
        float(overall["q_se_weighted_rmse"]),
        float(epoch),
    )


def _clone_state_dict(model: T3T4PolicyValueQNet) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _train_and_select(
    fit: DistillationSplit,
    dev: DistillationSplit,
    config: DistillationTrainingConfig,
) -> tuple[T3T4PolicyValueQNet, list[dict[str, Any]], int, tuple[float, ...]]:
    """Train from fit only and select from dev only; test is not accepted here."""

    if fit.name != "fit" or dev.name != "dev":
        raise DistillationTrainingError("training requires locked fit and dev splits")
    if len(fit) == 0 or len(dev) == 0:
        raise DistillationTrainingError("fit and dev splits must be nonempty")
    config_payload = _config_payload(config)
    model = T3T4PolicyValueQNet(_model_config(config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_payload["learning_rate"],
        weight_decay=config_payload["weight_decay"],
        foreach=False,
    )
    fit_batch = fit.training_batch()
    records: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config_payload["epochs"] + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config_payload["seed"] + epoch)
        order = torch.randperm(len(fit), generator=generator).numpy()
        totals = {"total": 0.0, "policy": 0.0, "value": 0.0, "q": 0.0}
        processed = 0
        model.train()
        for start in range(0, len(fit), config_payload["batch_size"]):
            indices = order[start : start + config_payload["batch_size"]]
            batch = {key: value[indices] for key, value in fit_batch.items()}
            states = torch.as_tensor(batch["state"], dtype=torch.float32, device="cpu")
            output = model(states)
            losses = masked_policy_value_q_loss(output, batch, config)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = len(indices)
            processed += count
            for name in totals:
                totals[name] += float(losses[name].detach().item()) * count
        if processed != len(fit):
            raise AssertionError("fit epoch did not consume each row exactly once")
        dev_evaluation = _evaluate_split(
            model,
            dev,
            config,
            include_groups=False,
        )
        selection_key = _selection_key(dev_evaluation, epoch)
        state_sha256 = _model_state_sha256(model)
        record = {
            "epoch": epoch,
            "fit_rows_consumed": processed,
            "fit_loss_mean": {name: totals[name] / processed for name in sorted(totals)},
            "dev_evaluation_sha256": dev_evaluation["evaluation_sha256"],
            "dev_overall": dev_evaluation["overall"],
            "selection_key": list(selection_key),
            "model_state_sha256": state_sha256,
        }
        records.append(record)
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_epoch = epoch
            best_state = _clone_state_dict(model)
    if best_state is None or best_key is None or best_epoch < 1:
        raise AssertionError("dev checkpoint selection produced no candidate")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    if _model_state_sha256(model) != records[best_epoch - 1]["model_state_sha256"]:
        raise AssertionError("restored selected checkpoint state mismatch")
    return model, records, best_epoch, best_key


def _seed_contract(config: DistillationTrainingConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    contract: dict[str, Any] = {
        "schema": "ofc_t3_t4_policy_value_q_seed_contract/v1",
        "initialization_seed": payload["seed"],
        "epoch_shuffle_seed_formula": "training_seed_plus_one_based_epoch",
        "epoch_shuffle_seeds": [
            payload["seed"] + epoch for epoch in range(1, payload["epochs"] + 1)
        ],
        "data_loader_workers": 0,
        "fit_epoch_each_row_exactly_once": True,
        "cpu_thread_count": 1,
        "torch_deterministic_algorithms": True,
        "mkldnn_enabled": False,
    }
    contract["seed_contract_sha256"] = _self_hash(
        contract, "seed_contract_sha256"
    )
    return contract


def _loss_contract(config: DistillationTrainingConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    contract: dict[str, Any] = {
        "schema": "ofc_t3_t4_policy_value_q_loss_contract/v1",
        "policy": "legal_action_masked_soft_target_cross_entropy",
        "value": "smooth_l1_acting_player_policy_value",
        "q": "legal_action_masked_smooth_l1_all_legal_actions",
        "illegal_policy_logits_excluded": True,
        "illegal_q_predictions_excluded": True,
        "q_standard_error_weighting": "min(cap,1/(se_squared+floor_squared))",
        "q_standard_error_floor": payload["q_standard_error_floor"],
        "q_precision_weight_cap": payload["q_precision_weight_cap"],
        "huber_delta": payload["huber_delta"],
        "policy_loss_weight": payload["policy_loss_weight"],
        "value_loss_weight": payload["value_loss_weight"],
        "q_loss_weight": payload["q_loss_weight"],
    }
    contract["loss_contract_sha256"] = _self_hash(
        contract, "loss_contract_sha256"
    )
    return contract


def _source_file_paths() -> dict[str, Path]:
    paths = dict(_PINNED_SOURCE_PATHS)
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise DistillationTrainingError(f"source file {label} is not a real file")
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _source_hashes() -> dict[str, str]:
    current = {
        label: _file_sha256(path)
        for label, path in sorted(_source_file_paths().items())
    }
    if current != dict(_PINNED_SOURCE_SHA256):
        raise DistillationTrainingError(
            "source files changed after trainer module import"
        )
    return current


def _semantic_value_descriptor(
    value: Any,
    active: set[int] | None = None,
) -> Any:
    """Return an address-free descriptor for callable semantic state.

    Python function identity does not bind mutable ``__code__``, defaults, or
    closure cells.  This representation is deliberately composed only from
    canonical JSON values so it can be compared directly at the public API
    boundary and hashed into the artifact manifest.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if value is NotImplemented:
        return {"kind": "not_implemented"}

    seen = active if active is not None else set()
    identity = id(value)
    if identity in seen:
        value_type = type(value)
        return {
            "kind": "recursive_reference",
            "type_module": value_type.__module__,
            "type_qualname": value_type.__qualname__,
        }
    seen.add(identity)
    try:
        if isinstance(value, types.CodeType):
            return {
                "kind": "code",
                "argcount": value.co_argcount,
                "posonlyargcount": value.co_posonlyargcount,
                "kwonlyargcount": value.co_kwonlyargcount,
                "nlocals": value.co_nlocals,
                "stacksize": value.co_stacksize,
                "flags": value.co_flags,
                "bytecode_hex": value.co_code.hex(),
                "exceptiontable_hex": getattr(value, "co_exceptiontable", b"").hex(),
                "constants": [
                    _semantic_value_descriptor(item, seen) for item in value.co_consts
                ],
                "names": list(value.co_names),
                "varnames": list(value.co_varnames),
                "freevars": list(value.co_freevars),
                "cellvars": list(value.co_cellvars),
            }
        if isinstance(value, types.FunctionType):
            closure: list[Any] = []
            for cell in value.__closure__ or ():
                try:
                    contents = cell.cell_contents
                except ValueError:
                    closure.append({"kind": "empty_cell"})
                else:
                    closure.append(_semantic_value_descriptor(contents, seen))
            wrapped = getattr(value, "__wrapped__", None)
            return {
                "kind": "python_function",
                "module": value.__module__,
                "qualname": value.__qualname__,
                "name": value.__name__,
                "code": _semantic_value_descriptor(value.__code__, seen),
                "defaults": _semantic_value_descriptor(value.__defaults__, seen),
                "kwdefaults": _semantic_value_descriptor(value.__kwdefaults__, seen),
                "closure": closure,
                "wrapped": (
                    None
                    if wrapped is None
                    else _semantic_value_descriptor(wrapped, seen)
                ),
            }
        if isinstance(value, types.MethodType):
            owner = value.__self__
            owner_type = owner if isinstance(owner, type) else type(owner)
            return {
                "kind": "bound_method",
                "function": _semantic_value_descriptor(value.__func__, seen),
                "owner_module": owner_type.__module__,
                "owner_qualname": owner_type.__qualname__,
            }
        if isinstance(value, type):
            return {
                "kind": "class",
                "module": value.__module__,
                "qualname": value.__qualname__,
            }
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [_semantic_value_descriptor(item, seen) for item in value],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [_semantic_value_descriptor(item, seen) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            items = [_semantic_value_descriptor(item, seen) for item in value]
            items.sort(key=repr)
            return {"kind": type(value).__name__, "items": items}
        if isinstance(value, Mapping):
            items = [
                (
                    _semantic_value_descriptor(key, seen),
                    _semantic_value_descriptor(item, seen),
                )
                for key, item in value.items()
            ]
            items.sort(key=lambda pair: repr(pair[0]))
            return {
                "kind": "mapping",
                "items": [[key, item] for key, item in items],
            }
        if isinstance(value, types.ModuleType):
            return {"kind": "module", "name": value.__name__}
        if isinstance(value, os.PathLike):
            return {
                "kind": "pathlike",
                "type_module": type(value).__module__,
                "type_qualname": type(value).__qualname__,
                "path": os.fspath(value),
            }
        if hasattr(value, "__dataclass_fields__"):
            names = sorted(value.__dataclass_fields__)
            return {
                "kind": "dataclass",
                "type_module": type(value).__module__,
                "type_qualname": type(value).__qualname__,
                "fields": {
                    name: _semantic_value_descriptor(getattr(value, name), seen)
                    for name in names
                },
            }
        if callable(value):
            owner_class = getattr(value, "__objclass__", None)
            bound_owner = getattr(value, "__self__", None)
            if isinstance(bound_owner, types.ModuleType):
                bound_owner_descriptor: Any = {
                    "kind": "module",
                    "name": bound_owner.__name__,
                }
            elif bound_owner is None:
                bound_owner_descriptor = None
            else:
                bound_owner_type = (
                    bound_owner if isinstance(bound_owner, type) else type(bound_owner)
                )
                bound_owner_descriptor = {
                    "kind": "owner_type",
                    "module": bound_owner_type.__module__,
                    "qualname": bound_owner_type.__qualname__,
                }
            return {
                "kind": "native_or_callable_descriptor",
                "type_module": type(value).__module__,
                "type_qualname": type(value).__qualname__,
                "module": getattr(value, "__module__", None),
                "qualname": getattr(value, "__qualname__", None),
                "name": getattr(value, "__name__", None),
                "text_signature": getattr(value, "__text_signature__", None),
                "owner_class": (
                    None
                    if owner_class is None
                    else {
                        "module": owner_class.__module__,
                        "qualname": owner_class.__qualname__,
                    }
                ),
                "bound_owner": bound_owner_descriptor,
            }
        raise DistillationTrainingError(
            "unsupported runtime semantic value type: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    finally:
        seen.remove(identity)


def _callable_semantic_descriptor(value: Any) -> dict[str, Any]:
    descriptor = _semantic_value_descriptor(value)
    if not isinstance(descriptor, dict):
        raise DistillationTrainingError("callable semantic descriptor must be object")
    return descriptor


def _freeze_semantic_descriptor(value: Any) -> tuple[Any, ...]:
    """Deep-freeze a descriptor so boundary checks need no mutable hasher."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, list):
        return (
            "list",
            tuple(_freeze_semantic_descriptor(item) for item in value),
        )
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DistillationTrainingError(
                "semantic descriptor object keys must be strings"
            )
        return (
            "mapping",
            tuple(
                (key, _freeze_semantic_descriptor(value[key]))
                for key in sorted(value)
            ),
        )
    raise DistillationTrainingError(
        "semantic descriptor is not canonical JSON data: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _resolve_runtime_semantic_target(
    root_name: str,
    attributes: Sequence[str],
) -> Any:
    target = globals().get(root_name)
    if target is None:
        raise DistillationTrainingError(
            f"runtime semantic root missing: {root_name}"
        )
    for attribute in attributes:
        try:
            target = getattr(target, attribute)
        except AttributeError as exc:
            path = ".".join((root_name, *attributes))
            raise DistillationTrainingError(
                f"runtime semantic target missing: {path}"
            ) from exc
    return target


def _assert_runtime_semantic_graph(
    expected_bindings: Sequence[
        tuple[str, str, tuple[str, ...], Any, tuple[Any, ...]]
    ],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for label, root_name, attributes, canonical, expected_state in expected_bindings:
        current = _resolve_runtime_semantic_target(root_name, attributes)
        if current is not canonical:
            raise DistillationTrainingError(
                f"trainer runtime semantic target drift: {label}"
            )
        descriptor = _callable_semantic_descriptor(current)
        if _freeze_semantic_descriptor(descriptor) != expected_state:
            raise DistillationTrainingError(
                f"trainer runtime semantic descriptor drift: {label}"
            )
        descriptor_json = _CANONICAL_JSON(descriptor)
        hashes[label] = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
    return hashes


def _runtime_semantic_graph() -> dict[str, Any]:
    binding_hashes = _assert_runtime_semantic_graph(
        _CANONICAL_RUNTIME_SEMANTIC_BINDINGS
    )
    graph: dict[str, Any] = {
        "schema": "ofc_t3_t4_training_runtime_semantic_graph/v1",
        "binding_descriptor_sha256": binding_hashes,
    }
    graph["runtime_semantic_graph_sha256"] = _self_hash(
        graph, "runtime_semantic_graph_sha256"
    )
    return graph


def _live_semantic_bindings() -> dict[str, Any]:
    encoder_manifest = _INFOSET_ENCODER_MANIFEST()
    try:
        _VALIDATE_INFOSET_ENCODER_MANIFEST(encoder_manifest)
    except (TypeError, ValueError) as exc:
        raise DistillationTrainingError(
            f"live encoder manifest validation failed: {exc}"
        ) from exc
    if encoder_manifest.get("manifest_sha256") != INFOSET_ENCODER_MANIFEST_SHA256:
        raise DistillationTrainingError("live encoder manifest binding mismatch")
    # Hash the live callable identities' semantic outputs rather than relying on
    # source text alone.  The fixed encoder manifest binds action generation,
    # while the callable descriptors detect runtime alias substitution.
    bindings: dict[str, Any] = {
        "schema": "ofc_t3_t4_training_live_semantic_bindings/v1",
        "infoset_encoder_manifest_sha256": INFOSET_ENCODER_MANIFEST_SHA256,
        "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
        "infoset_vector_dimension": INFOSET_VECTOR_DIM,
        "dataset_schema": DISTILLATION_DATASET_SCHEMA,
        "runtime_semantic_graph": _runtime_semantic_graph(),
        "callable_bindings": {
            "infoset_encoder_manifest": {
                "module": _INFOSET_ENCODER_MANIFEST.__module__,
                "qualname": _INFOSET_ENCODER_MANIFEST.__qualname__,
            },
            "legal_action_mask": {
                "module": _LEGAL_ACTION_MASK.__module__,
                "qualname": _LEGAL_ACTION_MASK.__qualname__,
            },
            "semantic_action_ids": {
                "module": _SEMANTIC_ACTION_IDS.__module__,
                "qualname": _SEMANTIC_ACTION_IDS.__qualname__,
            },
            "validate_infoset_encoder_manifest": {
                "module": _VALIDATE_INFOSET_ENCODER_MANIFEST.__module__,
                "qualname": _VALIDATE_INFOSET_ENCODER_MANIFEST.__qualname__,
            },
            "load_distillation_dataset": {
                "module": _LOAD_DISTILLATION_DATASET.__module__,
                "qualname": _LOAD_DISTILLATION_DATASET.__qualname__,
            },
            "verify_distillation_dataset": {
                "module": _VERIFY_DISTILLATION_DATASET.__module__,
                "qualname": _VERIFY_DISTILLATION_DATASET.__qualname__,
            },
            "assert_transitive_runtime_anchors": {
                "module": _ASSERT_TRANSITIVE_RUNTIME_ANCHORS.__module__,
                "qualname": _ASSERT_TRANSITIVE_RUNTIME_ANCHORS.__qualname__,
            },
        },
    }
    bindings["live_semantic_bindings_sha256"] = _self_hash(
        bindings, "live_semantic_bindings_sha256"
    )
    return bindings


def _environment_contract() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "device": "cpu",
        "dtype": "float32",
        "reproducibility_scope": (
            "same_source_hashes_config_seed_versions_cpu_and_thread_contract"
        ),
    }


def _checkpoint_bytes(
    model: T3T4PolicyValueQNet,
    *,
    model_config_payload: Mapping[str, Any],
    source_dataset_manifest_sha256: str,
    training_config_sha256: str,
    selected_epoch: int,
) -> tuple[bytes, dict[str, Any]]:
    tensor_payloads: list[bytes] = []
    tensor_entries: list[dict[str, Any]] = []
    offset = 0
    state = model.state_dict()
    valid_name_characters = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_."
    )
    for name, tensor in sorted(state.items()):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise DistillationTrainingError("checkpoint tensors must be CPU float32")
        if not name or any(
            character not in valid_name_characters for character in name
        ):
            raise DistillationTrainingError("checkpoint tensor name is not canonical")
        array = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
        raw = array.tobytes(order="C")
        tensor_entries.append(
            {
                "name": name,
                "dtype": "<f4",
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
                "tensor_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        tensor_payloads.append(raw)
        offset += len(raw)
    header: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "promotion_eligible": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "model_config": dict(model_config_payload),
        "model_config_sha256": _CANONICAL_SHA256(model_config_payload),
        "source_dataset_manifest_sha256": _sha(
            source_dataset_manifest_sha256,
            label="checkpoint source_dataset_manifest_sha256",
        ),
        "training_config_sha256": _sha(
            training_config_sha256,
            label="checkpoint training_config_sha256",
        ),
        "selected_epoch": _strict_int(
            selected_epoch,
            label="checkpoint selected_epoch",
            minimum=1,
            maximum=10_000,
        ),
        "model_state_sha256": _model_state_sha256(model),
        "tensor_count": len(tensor_entries),
        "tensor_payload_nbytes": offset,
        "tensors": tensor_entries,
    }
    header["header_sha256"] = _self_hash(header, "header_sha256")
    header_bytes = _CANONICAL_JSON(header).encode("utf-8")
    if len(header_bytes) > MAX_CHECKPOINT_HEADER_BYTES:
        raise DistillationTrainingError("checkpoint header exceeds bound")
    payload = (
        CHECKPOINT_MAGIC
        + len(header_bytes).to_bytes(8, "big")
        + header_bytes
        + b"".join(tensor_payloads)
    )
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise DistillationTrainingError("checkpoint exceeds size bound")
    return payload, header


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DistillationTrainingError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_checkpoint(
    payload: bytes,
    *,
    expected_model_config_sha256: str,
    expected_source_dataset_manifest_sha256: str,
    expected_training_config_sha256: str,
    expected_selected_epoch: int,
) -> tuple[T3T4PolicyValueQNet, dict[str, Any]]:
    if not isinstance(payload, bytes) or not payload.startswith(CHECKPOINT_MAGIC):
        raise DistillationTrainingError("checkpoint magic mismatch")
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise DistillationTrainingError("checkpoint exceeds size bound")
    cursor = len(CHECKPOINT_MAGIC)
    if len(payload) < cursor + 8:
        raise DistillationTrainingError("checkpoint header length missing")
    header_size = int.from_bytes(payload[cursor : cursor + 8], "big")
    cursor += 8
    if not 1 <= header_size <= MAX_CHECKPOINT_HEADER_BYTES:
        raise DistillationTrainingError("checkpoint header length invalid")
    if len(payload) < cursor + header_size:
        raise DistillationTrainingError("checkpoint header truncated")
    raw_header = payload[cursor : cursor + header_size]
    cursor += header_size
    try:
        text = raw_header.decode("utf-8", errors="strict")
        header = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DistillationTrainingError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistillationTrainingError("checkpoint header JSON invalid") from exc
    if not isinstance(header, dict) or _CANONICAL_JSON(header) != text:
        raise DistillationTrainingError("checkpoint header is not canonical JSON")
    expected_keys = {
        "schema",
        "candidate_only",
        "promotion_eligible",
        "runtime_allowed",
        "serving_changed",
        "model_config",
        "model_config_sha256",
        "source_dataset_manifest_sha256",
        "training_config_sha256",
        "selected_epoch",
        "model_state_sha256",
        "tensor_count",
        "tensor_payload_nbytes",
        "tensors",
        "header_sha256",
    }
    _exact_keys(header, expected_keys, label="checkpoint header")
    if header["schema"] != CHECKPOINT_SCHEMA:
        raise DistillationTrainingError("checkpoint schema mismatch")
    if (
        header["candidate_only"] is not True
        or header["promotion_eligible"] is not False
        or header["runtime_allowed"] is not False
        or header["serving_changed"] is not False
    ):
        raise DistillationTrainingError("checkpoint candidate/runtime flags mismatch")
    if header["header_sha256"] != _self_hash(header, "header_sha256"):
        raise DistillationTrainingError("checkpoint header self-hash mismatch")
    model_config_payload = header["model_config"]
    if not isinstance(model_config_payload, dict):
        raise DistillationTrainingError("checkpoint model config must be object")
    model_config = _model_config_from_payload(model_config_payload)
    if (
        header["model_config_sha256"]
        != _sha(expected_model_config_sha256, label="expected model config SHA")
        or header["model_config_sha256"] != _CANONICAL_SHA256(model_config_payload)
    ):
        raise DistillationTrainingError("checkpoint model config binding mismatch")
    fixed = {
        "source_dataset_manifest_sha256": _sha(
            expected_source_dataset_manifest_sha256,
            label="expected source dataset manifest SHA",
        ),
        "training_config_sha256": _sha(
            expected_training_config_sha256,
            label="expected training config SHA",
        ),
        "selected_epoch": expected_selected_epoch,
    }
    for field, expected in fixed.items():
        if header[field] != expected:
            raise DistillationTrainingError(f"checkpoint {field} binding mismatch")
    raw_entries = header["tensors"]
    tensor_count = _strict_int(
        header["tensor_count"],
        label="checkpoint tensor_count",
        minimum=1,
        maximum=100_000,
    )
    if not isinstance(raw_entries, list) or len(raw_entries) != tensor_count:
        raise DistillationTrainingError("checkpoint tensor list/count mismatch")
    payload_size = _strict_int(
        header["tensor_payload_nbytes"],
        label="checkpoint tensor payload bytes",
        minimum=1,
        maximum=MAX_CHECKPOINT_BYTES,
    )
    tensor_blob = payload[cursor:]
    if len(tensor_blob) != payload_size:
        raise DistillationTrainingError("checkpoint tensor payload length mismatch")
    model = T3T4PolicyValueQNet(model_config)
    expected_state = model.state_dict()
    names: list[str] = []
    state: dict[str, torch.Tensor] = {}
    expected_offset = 0
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise DistillationTrainingError("checkpoint tensor entry must be object")
        _exact_keys(
            raw_entry,
            {"name", "dtype", "shape", "offset", "nbytes", "tensor_sha256"},
            label=f"checkpoint tensor {index}",
        )
        name = raw_entry["name"]
        if not isinstance(name, str) or name not in expected_state:
            raise DistillationTrainingError("checkpoint tensor name unknown")
        if raw_entry["dtype"] != "<f4":
            raise DistillationTrainingError("checkpoint tensor dtype mismatch")
        shape = raw_entry["shape"]
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise DistillationTrainingError("checkpoint tensor shape invalid")
        if tuple(shape) != tuple(expected_state[name].shape):
            raise DistillationTrainingError("checkpoint tensor shape mismatch")
        nbytes = int(np.prod(shape, dtype=np.int64)) * 4
        if raw_entry["offset"] != expected_offset or raw_entry["nbytes"] != nbytes:
            raise DistillationTrainingError("checkpoint tensor gap/overlap/size mismatch")
        raw = tensor_blob[expected_offset : expected_offset + nbytes]
        if hashlib.sha256(raw).hexdigest() != _sha(
            raw_entry["tensor_sha256"], label="checkpoint tensor SHA"
        ):
            raise DistillationTrainingError("checkpoint tensor SHA mismatch")
        array = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        state[name] = torch.from_numpy(array)
        names.append(name)
        expected_offset += nbytes
    if names != sorted(expected_state) or expected_offset != len(tensor_blob):
        raise DistillationTrainingError("checkpoint tensor order or terminal offset mismatch")
    model.load_state_dict(state, strict=True)
    model.eval()
    if _model_state_sha256(model) != _sha(
        header["model_state_sha256"], label="checkpoint model state SHA"
    ):
        raise DistillationTrainingError("checkpoint model state hash mismatch")
    return model, header


def _prepare_artifact_root(output_dir: str | Path) -> Path:
    supplied = Path(output_dir).absolute()
    if supplied.is_symlink():
        raise DistillationTrainingError("artifact root symlink forbidden")
    if supplied.exists():
        if not supplied.is_dir():
            raise DistillationTrainingError("artifact root must be a directory")
        if any(supplied.iterdir()):
            raise DistillationTrainingError("artifact output directory must be empty")
    else:
        supplied.mkdir(parents=True)
    return supplied.resolve(strict=True)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hardlink publishes the already-complete staged
            # inode atomically and, unlike replace(), has no overwrite mode.
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise DistillationTrainingError(
                    f"content-addressed artifact collision: {path.name}"
                )
        except OSError as exc:
            raise DistillationTrainingError(
                "atomic no-replace hardlink publication failed"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
        raise DistillationTrainingError("artifact atomic write verification failed")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (_CANONICAL_JSON(value) + "\n").encode("utf-8")


def _write_json_artifact(
    root: Path,
    *,
    prefix: str,
    payload: Mapping[str, Any],
    self_hash_field: str,
) -> dict[str, str]:
    value = dict(payload)
    value[self_hash_field] = _self_hash(value, self_hash_field)
    relative_path = f"{prefix}-{value[self_hash_field]}.json"
    encoded = _json_bytes(value)
    path = root / relative_path
    _atomic_write_bytes(path, encoded)
    return {
        "relative_path": relative_path,
        "payload_sha256": value[self_hash_field],
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _event(
    events: list[dict[str, Any]],
    *,
    kind: str,
    split: str | None,
    epoch: int,
    binding_sha256: str,
) -> None:
    event: dict[str, Any] = {
        "schema": EVENT_SCHEMA,
        "sequence": len(events),
        "kind": kind,
        "split": split,
        "epoch": epoch,
        "binding_sha256": _sha(binding_sha256, label="event binding SHA"),
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else "0" * 64
        ),
    }
    event["event_sha256"] = _self_hash(event, "event_sha256")
    events.append(event)


def _build_history(
    *,
    dataset_manifest_sha256: str,
    training_config_sha256: str,
    seed_contract_sha256: str,
    checkpoint_sha256: str,
    records: Sequence[Mapping[str, Any]],
    selected_epoch: int,
    selected_key: Sequence[float],
    final_evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for record in records:
        _event(
            events,
            kind="dev_checkpoint_candidate_evaluation",
            split="dev",
            epoch=int(record["epoch"]),
            binding_sha256=str(record["dev_evaluation_sha256"]),
        )
    _event(
        events,
        kind="dev_checkpoint_selected",
        split="dev",
        epoch=selected_epoch,
        binding_sha256=checkpoint_sha256,
    )
    for split in ("fit", "dev", "test"):
        _event(
            events,
            kind=f"selected_checkpoint_{split}_evaluation",
            split=split,
            epoch=selected_epoch,
            binding_sha256=str(final_evaluations[split]["evaluation_sha256"]),
        )
    history: dict[str, Any] = {
        "schema": TRAINING_HISTORY_SCHEMA,
        "artifact_kind": "deterministic_fit_train_dev_select_test_once_history",
        "source_dataset_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": training_config_sha256,
        "seed_contract_sha256": seed_contract_sha256,
        "fit_gradient_source_only": True,
        "dev_checkpoint_selection_only": True,
        "test_evaluated_after_checkpoint_selection_only": True,
        "training_pipeline_test_evaluation_count": 1,
        "selection_metric_order": [
            "dev.policy_ev_regret_mean",
            "dev.policy_cross_entropy_mean",
            "dev.value_rmse",
            "dev.q_se_weighted_rmse",
            "epoch_ascending",
        ],
        "epoch_count": len(records),
        "epoch_records": [dict(record) for record in records],
        "selection": {
            "selected_epoch": selected_epoch,
            "selected_key": list(selected_key),
            "selected_model_state_sha256": records[selected_epoch - 1][
                "model_state_sha256"
            ],
            "checkpoint_sha256": checkpoint_sha256,
        },
        "evaluation_events": events,
        "terminal_event_sha256": events[-1]["event_sha256"],
    }
    return history


def _build_metrics(
    *,
    dataset_manifest_sha256: str,
    checkpoint_sha256: str,
    selected_epoch: int,
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": TRAINING_METRICS_SCHEMA,
        "artifact_kind": "selected_checkpoint_locked_split_metrics",
        "source_dataset_manifest_sha256": dataset_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": selected_epoch,
        "evaluation_order": ["fit", "dev", "test"],
        "training_pipeline_test_evaluation_count": 1,
        "root_family_phase_actor_joker_metrics_included": True,
        "splits": {name: dict(evaluations[name]) for name in ("fit", "dev", "test")},
    }


def _manifest_payload(
    *,
    dataset: VerifiedDistillationDataset,
    config_payload: Mapping[str, Any],
    model_config_payload: Mapping[str, Any],
    seed_contract: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    live_bindings: Mapping[str, Any],
    environment: Mapping[str, Any],
    selected_epoch: int,
    selected_key: Sequence[float],
    checkpoint_ref: Mapping[str, Any],
    history_ref: Mapping[str, Any],
    metrics_ref: Mapping[str, Any],
) -> dict[str, Any]:
    training_config_sha256 = _CANONICAL_SHA256(config_payload)
    model_config_sha256 = _CANONICAL_SHA256(model_config_payload)
    return {
        "schema": TRAINING_ARTIFACT_SCHEMA,
        "artifact_kind": "content_addressed_t3_t4_policy_value_q_candidate",
        "opt_in_only": True,
        "candidate_only": True,
        "training_performed": True,
        "promotion_eligible": False,
        "promotion_gate_evaluated": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "global_policy_claimed": False,
        "exact_exploitability_computed": False,
        "deterministic_cpu_training": True,
        "opponent_hidden_cards_in_model_input": False,
        "audit_commitments_exposed_as_model_features": False,
        "model_feature_keys": ["state"],
        "source": {
            "teacher_bundle_manifest_sha256": dataset.source_teacher_manifest_sha256,
            "dataset_schema": dataset.manifest["schema"],
            "dataset_manifest_sha256": dataset.manifest[
                "dataset_manifest_sha256"
            ],
            "dataset_teacher_row_content_set_sha256": dataset.manifest[
                "source_contract"
            ]["teacher_row_content_set_sha256"],
        },
        "split_usage": {
            "fit": "gradient_updates_only",
            "dev": "checkpoint_selection_only",
            "test": "selected_checkpoint_single_training_pipeline_evaluation_only",
            "random_resplit_performed": False,
            "training_pipeline_test_evaluation_count": 1,
            "fresh_verifier_test_replay_is_integrity_only": True,
            "fresh_verifier_cannot_change_checkpoint_selection": True,
        },
        "training_config": dict(config_payload),
        "training_config_sha256": training_config_sha256,
        "loss_contract": _loss_contract(_config_from_payload(config_payload)),
        "model_config": dict(model_config_payload),
        "model_config_sha256": model_config_sha256,
        "seed_contract": dict(seed_contract),
        "source_file_sha256": dict(source_hashes),
        "live_semantic_bindings": dict(live_bindings),
        "environment": dict(environment),
        "selection": {
            "selected_epoch": selected_epoch,
            "selected_key": list(selected_key),
            "selection_source_split": "dev",
        },
        "checkpoint": dict(checkpoint_ref),
        "history": dict(history_ref),
        "metrics": dict(metrics_ref),
        "manifest_written_last": True,
    }


def train_distillation_candidate(
    teacher_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_teacher_manifest_sha256: str,
    config: DistillationTrainingConfig = DistillationTrainingConfig(),
) -> VerifiedDistillationTrainingArtifact:
    """Train one deterministic candidate without changing runtime or serving."""

    config_payload = _config_payload(config)
    root = _prepare_artifact_root(output_dir)
    source_hashes_before = _source_hashes()
    live_bindings_before = _live_semantic_bindings()
    try:
        dataset = _LOAD_DISTILLATION_DATASET(
            teacher_bundle_dir,
            expected_teacher_manifest_sha256=expected_teacher_manifest_sha256,
            required_splits=("fit", "dev", "test"),
        )
        _VERIFY_DISTILLATION_DATASET(dataset)
    except (DistillationDatasetError, TypeError, ValueError) as exc:
        raise DistillationTrainingError(f"fresh dataset verification failed: {exc}") from exc
    dataset_manifest_sha256 = dataset.manifest["dataset_manifest_sha256"]
    training_config_sha256 = _CANONICAL_SHA256(config_payload)
    model_config_payload = _model_config_payload(_model_config(config))
    seed_contract = _seed_contract(config)
    with _deterministic_cpu(config.seed):
        model, records, selected_epoch, selected_key = _train_and_select(
            dataset.for_split("fit"),
            dataset.for_split("dev"),
            config,
        )
        # The test split is not passed to the trainer/selector above.  This is
        # the sole training-pipeline test evaluation, after state restoration.
        evaluations = {
            "fit": _evaluate_split(
                model, dataset.for_split("fit"), config, include_groups=True
            ),
            "dev": _evaluate_split(
                model, dataset.for_split("dev"), config, include_groups=True
            ),
            "test": _evaluate_split(
                model, dataset.for_split("test"), config, include_groups=True
            ),
        }
    if _source_hashes() != source_hashes_before:
        raise DistillationTrainingError("source files changed during training")
    if _live_semantic_bindings() != live_bindings_before:
        raise DistillationTrainingError("live semantic bindings changed during training")

    checkpoint_payload, checkpoint_header = _checkpoint_bytes(
        model,
        model_config_payload=model_config_payload,
        source_dataset_manifest_sha256=dataset_manifest_sha256,
        training_config_sha256=training_config_sha256,
        selected_epoch=selected_epoch,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    checkpoint_relative_path = f"checkpoint-{checkpoint_sha256}.bin"
    _atomic_write_bytes(root / checkpoint_relative_path, checkpoint_payload)
    checkpoint_ref = {
        "relative_path": checkpoint_relative_path,
        "file_sha256": checkpoint_sha256,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "header_sha256": checkpoint_header["header_sha256"],
        "model_state_sha256": checkpoint_header["model_state_sha256"],
        "selected_epoch": selected_epoch,
    }

    history_payload = _build_history(
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_config_sha256=training_config_sha256,
        seed_contract_sha256=seed_contract["seed_contract_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        records=records,
        selected_epoch=selected_epoch,
        selected_key=selected_key,
        final_evaluations=evaluations,
    )
    history_ref = _write_json_artifact(
        root,
        prefix="training-history",
        payload=history_payload,
        self_hash_field="history_sha256",
    )
    metrics_payload = _build_metrics(
        dataset_manifest_sha256=dataset_manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        selected_epoch=selected_epoch,
        evaluations=evaluations,
    )
    metrics_ref = _write_json_artifact(
        root,
        prefix="selected-metrics",
        payload=metrics_payload,
        self_hash_field="metrics_sha256",
    )
    manifest = _manifest_payload(
        dataset=dataset,
        config_payload=config_payload,
        model_config_payload=model_config_payload,
        seed_contract=seed_contract,
        source_hashes=source_hashes_before,
        live_bindings=live_bindings_before,
        environment=_environment_contract(),
        selected_epoch=selected_epoch,
        selected_key=selected_key,
        checkpoint_ref=checkpoint_ref,
        history_ref=history_ref,
        metrics_ref=metrics_ref,
    )
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    manifest_relative_path = (
        f"{MANIFEST_PREFIX}{manifest['manifest_sha256']}.json"
    )
    _atomic_write_bytes(root / manifest_relative_path, _json_bytes(manifest))
    return verify_training_artifact(
        root,
        teacher_bundle_dir=teacher_bundle_dir,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )


def _safe_artifact_file(root: Path, relative_path: Any, *, label: str) -> Path:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).name != relative_path
        or "/" in relative_path
        or "\\" in relative_path
        or relative_path in {".", ".."}
    ):
        raise DistillationTrainingError(f"{label}: flat safe filename required")
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise DistillationTrainingError(f"{label}: real regular file required")
    if path.resolve(strict=True).parent != root:
        raise DistillationTrainingError(f"{label}: path escapes artifact root")
    return path


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.stat().st_size > 512 * 1024 * 1024:
        raise DistillationTrainingError(f"{label}: JSON artifact exceeds bound")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DistillationTrainingError(f"{label}: invalid UTF-8") from exc
    if not text.endswith("\n") or "\r" in text:
        raise DistillationTrainingError(f"{label}: canonical single-LF termination required")
    try:
        value = json.loads(
            text[:-1],
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DistillationTrainingError(f"{label}: non-finite JSON {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise DistillationTrainingError(f"{label}: invalid JSON") from exc
    if not isinstance(value, dict) or _CANONICAL_JSON(value) != text[:-1]:
        raise DistillationTrainingError(f"{label}: noncanonical JSON")
    return value


def _verify_json_ref(
    root: Path,
    reference: Mapping[str, Any],
    *,
    label: str,
    prefix: str,
    self_hash_field: str,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(reference, Mapping):
        raise DistillationTrainingError(f"{label} reference must be object")
    _exact_keys(
        reference,
        {"relative_path", "payload_sha256", "file_sha256"},
        label=f"{label} reference",
    )
    payload_sha = _sha(reference["payload_sha256"], label=f"{label} payload SHA")
    expected_name = f"{prefix}-{payload_sha}.json"
    if reference["relative_path"] != expected_name:
        raise DistillationTrainingError(f"{label}: content-addressed path mismatch")
    path = _safe_artifact_file(root, expected_name, label=label)
    if _file_sha256(path) != _sha(
        reference["file_sha256"], label=f"{label} file SHA"
    ):
        raise DistillationTrainingError(f"{label}: file SHA mismatch")
    payload = _read_canonical_json(path, label=label)
    if payload.get(self_hash_field) != payload_sha or payload_sha != _self_hash(
        payload, self_hash_field
    ):
        raise DistillationTrainingError(f"{label}: payload self-hash mismatch")
    return payload, path


def _verify_history(
    history: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    training_config: DistillationTrainingConfig,
    training_config_sha256: str,
    seed_contract_sha256: str,
    checkpoint_sha256: str,
    checkpoint_header: Mapping[str, Any],
    expected_fit_rows: int,
) -> None:
    expected_keys = {
        "schema",
        "artifact_kind",
        "source_dataset_manifest_sha256",
        "training_config_sha256",
        "seed_contract_sha256",
        "fit_gradient_source_only",
        "dev_checkpoint_selection_only",
        "test_evaluated_after_checkpoint_selection_only",
        "training_pipeline_test_evaluation_count",
        "selection_metric_order",
        "epoch_count",
        "epoch_records",
        "selection",
        "evaluation_events",
        "terminal_event_sha256",
        "history_sha256",
    }
    _exact_keys(history, expected_keys, label="training history")
    fixed = {
        "schema": TRAINING_HISTORY_SCHEMA,
        "artifact_kind": "deterministic_fit_train_dev_select_test_once_history",
        "source_dataset_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": training_config_sha256,
        "seed_contract_sha256": seed_contract_sha256,
        "fit_gradient_source_only": True,
        "dev_checkpoint_selection_only": True,
        "test_evaluated_after_checkpoint_selection_only": True,
        "training_pipeline_test_evaluation_count": 1,
    }
    for field, expected in fixed.items():
        if history.get(field) != expected:
            raise DistillationTrainingError(f"training history {field} mismatch")
    if history.get("selection_metric_order") != [
        "dev.policy_ev_regret_mean",
        "dev.policy_cross_entropy_mean",
        "dev.value_rmse",
        "dev.q_se_weighted_rmse",
        "epoch_ascending",
    ]:
        raise DistillationTrainingError("training history selection metric order mismatch")
    records = history.get("epoch_records")
    config_payload = _config_payload(training_config)
    if (
        not isinstance(records, list)
        or len(records) != config_payload["epochs"]
        or history.get("epoch_count") != len(records)
    ):
        raise DistillationTrainingError("training history epoch count mismatch")
    record_keys = {
        "epoch",
        "fit_rows_consumed",
        "fit_loss_mean",
        "dev_evaluation_sha256",
        "dev_overall",
        "selection_key",
        "model_state_sha256",
    }
    keys: list[tuple[float, ...]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise DistillationTrainingError("epoch record must be object")
        _exact_keys(record, record_keys, label=f"epoch record {index}")
        if (
            record["epoch"] != index
            or record["fit_rows_consumed"] != expected_fit_rows
        ):
            raise DistillationTrainingError("epoch record order/fit consumption invalid")
        fit_losses = record["fit_loss_mean"]
        if not isinstance(fit_losses, dict) or set(fit_losses) != {
            "policy",
            "q",
            "total",
            "value",
        } or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in fit_losses.values()
        ):
            raise DistillationTrainingError("epoch fit losses invalid")
        _sha(record["dev_evaluation_sha256"], label="epoch dev evaluation SHA")
        _sha(record["model_state_sha256"], label="epoch model state SHA")
        selection_key = record["selection_key"]
        if (
            not isinstance(selection_key, list)
            or len(selection_key) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in selection_key
            )
            or float(selection_key[-1]) != float(index)
        ):
            raise DistillationTrainingError("epoch selection key invalid")
        overall = record["dev_overall"]
        if not isinstance(overall, dict):
            raise DistillationTrainingError("epoch dev overall metrics missing")
        expected_key = (
            float(overall["policy_ev_regret_mean"]),
            float(overall["policy_cross_entropy_mean"]),
            float(overall["value_rmse"]),
            float(overall["q_se_weighted_rmse"]),
            float(index),
        )
        if tuple(float(value) for value in selection_key) != expected_key:
            raise DistillationTrainingError("epoch dev metric/selection key mismatch")
        keys.append(expected_key)
    selected_index = min(range(len(keys)), key=lambda index: keys[index])
    selected_epoch = selected_index + 1
    selection = history.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "selected_epoch",
        "selected_key",
        "selected_model_state_sha256",
        "checkpoint_sha256",
    }:
        raise DistillationTrainingError("training history selection object invalid")
    if (
        selection["selected_epoch"] != selected_epoch
        or selection["selected_key"] != list(keys[selected_index])
        or selection["selected_model_state_sha256"]
        != records[selected_index]["model_state_sha256"]
        or selection["checkpoint_sha256"] != checkpoint_sha256
        or checkpoint_header["selected_epoch"] != selected_epoch
        or checkpoint_header["model_state_sha256"]
        != selection["selected_model_state_sha256"]
    ):
        raise DistillationTrainingError("dev checkpoint selection binding mismatch")
    events = history.get("evaluation_events")
    expected_kinds = ["dev_checkpoint_candidate_evaluation"] * len(records) + [
        "dev_checkpoint_selected",
        "selected_checkpoint_fit_evaluation",
        "selected_checkpoint_dev_evaluation",
        "selected_checkpoint_test_evaluation",
    ]
    if not isinstance(events, list) or len(events) != len(expected_kinds):
        raise DistillationTrainingError("training event ledger length mismatch")
    previous = "0" * 64
    for sequence, (event, expected_kind) in enumerate(zip(events, expected_kinds)):
        if not isinstance(event, dict):
            raise DistillationTrainingError("training event must be object")
        _exact_keys(
            event,
            {
                "schema",
                "sequence",
                "kind",
                "split",
                "epoch",
                "binding_sha256",
                "previous_event_sha256",
                "event_sha256",
            },
            label=f"training event {sequence}",
        )
        if (
            event["schema"] != EVENT_SCHEMA
            or event["sequence"] != sequence
            or event["kind"] != expected_kind
            or event["previous_event_sha256"] != previous
            or event["event_sha256"] != _self_hash(event, "event_sha256")
        ):
            raise DistillationTrainingError("training event chain mismatch")
        if sequence < len(records):
            expected_split = "dev"
            expected_epoch = sequence + 1
            expected_binding = records[sequence]["dev_evaluation_sha256"]
        elif sequence == len(records):
            expected_split = "dev"
            expected_epoch = selected_epoch
            expected_binding = checkpoint_sha256
        else:
            final_offset = sequence - len(records) - 1
            expected_split = ("fit", "dev", "test")[final_offset]
            expected_epoch = selected_epoch
            expected_binding = event["binding_sha256"]
        if (
            event["split"] != expected_split
            or event["epoch"] != expected_epoch
            or event["binding_sha256"] != expected_binding
        ):
            raise DistillationTrainingError("training event semantic binding mismatch")
        previous = event["event_sha256"]
    if events[-1]["split"] != "test" or sum(
        event["split"] == "test" for event in events
    ) != 1:
        raise DistillationTrainingError("test must occur once at ledger terminal")
    if history.get("terminal_event_sha256") != previous:
        raise DistillationTrainingError("training terminal event binding mismatch")


def _verify_metrics_payload(
    metrics: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    checkpoint_sha256: str,
    selected_epoch: int,
) -> None:
    _exact_keys(
        metrics,
        {
            "schema",
            "artifact_kind",
            "source_dataset_manifest_sha256",
            "checkpoint_sha256",
            "selected_epoch",
            "evaluation_order",
            "training_pipeline_test_evaluation_count",
            "root_family_phase_actor_joker_metrics_included",
            "splits",
            "metrics_sha256",
        },
        label="training metrics",
    )
    fixed = {
        "schema": TRAINING_METRICS_SCHEMA,
        "artifact_kind": "selected_checkpoint_locked_split_metrics",
        "source_dataset_manifest_sha256": dataset_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": selected_epoch,
        "evaluation_order": ["fit", "dev", "test"],
        "training_pipeline_test_evaluation_count": 1,
        "root_family_phase_actor_joker_metrics_included": True,
    }
    for field, expected in fixed.items():
        if metrics.get(field) != expected:
            raise DistillationTrainingError(f"training metrics {field} mismatch")
    splits = metrics.get("splits")
    if not isinstance(splits, dict) or list(splits) != ["dev", "fit", "test"]:
        # canonical JSON sorts keys, so readback order is alphabetical.
        if not isinstance(splits, dict) or set(splits) != {"fit", "dev", "test"}:
            raise DistillationTrainingError("training metrics split set mismatch")
    for name in ("fit", "dev", "test"):
        evaluation = splits[name]
        if not isinstance(evaluation, dict) or evaluation.get("split") != name:
            raise DistillationTrainingError("training metrics split identity mismatch")
        if evaluation.get("groups_included") is not True:
            raise DistillationTrainingError("final stratified metrics are missing")
        for field in (
            "by_public_root_family",
            "by_phase",
            "by_actor",
            "by_joker",
            "by_phase_actor_joker",
        ):
            if not isinstance(evaluation.get(field), dict) or not evaluation[field]:
                raise DistillationTrainingError(f"training metrics {field} missing")


def verify_training_artifact(
    artifact_dir: str | Path,
    *,
    teacher_bundle_dir: str | Path,
    expected_manifest_sha256: str,
) -> VerifiedDistillationTrainingArtifact:
    """Freshly reload source data/model and recompute every final metric."""

    expected_manifest = _sha(
        expected_manifest_sha256, label="expected training manifest SHA"
    )
    supplied_root = Path(artifact_dir).absolute()
    if supplied_root.is_symlink():
        raise DistillationTrainingError("artifact root symlink forbidden")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise DistillationTrainingError("artifact root must be a real directory")
    manifest_name = f"{MANIFEST_PREFIX}{expected_manifest}.json"
    manifest_path = _safe_artifact_file(root, manifest_name, label="training manifest")
    manifest = _read_canonical_json(manifest_path, label="training manifest")
    manifest_keys = {
        "schema",
        "artifact_kind",
        "opt_in_only",
        "candidate_only",
        "training_performed",
        "promotion_eligible",
        "promotion_gate_evaluated",
        "runtime_allowed",
        "serving_changed",
        "global_policy_claimed",
        "exact_exploitability_computed",
        "deterministic_cpu_training",
        "opponent_hidden_cards_in_model_input",
        "audit_commitments_exposed_as_model_features",
        "model_feature_keys",
        "source",
        "split_usage",
        "training_config",
        "training_config_sha256",
        "loss_contract",
        "model_config",
        "model_config_sha256",
        "seed_contract",
        "source_file_sha256",
        "live_semantic_bindings",
        "environment",
        "selection",
        "checkpoint",
        "history",
        "metrics",
        "manifest_written_last",
        "manifest_sha256",
    }
    _exact_keys(manifest, manifest_keys, label="training manifest")
    fixed = {
        "schema": TRAINING_ARTIFACT_SCHEMA,
        "artifact_kind": "content_addressed_t3_t4_policy_value_q_candidate",
        "opt_in_only": True,
        "candidate_only": True,
        "training_performed": True,
        "promotion_eligible": False,
        "promotion_gate_evaluated": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "global_policy_claimed": False,
        "exact_exploitability_computed": False,
        "deterministic_cpu_training": True,
        "opponent_hidden_cards_in_model_input": False,
        "audit_commitments_exposed_as_model_features": False,
        "model_feature_keys": ["state"],
        "manifest_written_last": True,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise DistillationTrainingError(f"training manifest {field} mismatch")
    if (
        manifest.get("manifest_sha256") != expected_manifest
        or expected_manifest != _self_hash(manifest, "manifest_sha256")
    ):
        raise DistillationTrainingError("training manifest self-hash mismatch")
    if manifest.get("source_file_sha256") != _source_hashes():
        raise DistillationTrainingError("fresh source-file SHA verification failed")
    if manifest.get("live_semantic_bindings") != _live_semantic_bindings():
        raise DistillationTrainingError("fresh live semantic binding verification failed")
    if manifest.get("environment") != _environment_contract():
        raise DistillationTrainingError("fresh deterministic environment mismatch")

    config_payload = manifest.get("training_config")
    if not isinstance(config_payload, dict):
        raise DistillationTrainingError("training config object missing")
    config = _config_from_payload(config_payload)
    training_config_sha256 = _CANONICAL_SHA256(config_payload)
    if manifest.get("training_config_sha256") != training_config_sha256:
        raise DistillationTrainingError("training config SHA mismatch")
    if manifest.get("loss_contract") != _loss_contract(config):
        raise DistillationTrainingError("masked policy/value/Q loss contract mismatch")
    model_config_payload = manifest.get("model_config")
    if not isinstance(model_config_payload, dict):
        raise DistillationTrainingError("model config object missing")
    _model_config_from_payload(model_config_payload)
    model_config_sha256 = _CANONICAL_SHA256(model_config_payload)
    if manifest.get("model_config_sha256") != model_config_sha256:
        raise DistillationTrainingError("model config SHA mismatch")
    if model_config_payload != _model_config_payload(_model_config(config)):
        raise DistillationTrainingError("model/training config architecture mismatch")
    seed_contract = manifest.get("seed_contract")
    if not isinstance(seed_contract, dict) or seed_contract != _seed_contract(config):
        raise DistillationTrainingError("seed contract mismatch")

    source = manifest.get("source")
    if not isinstance(source, dict) or set(source) != {
        "teacher_bundle_manifest_sha256",
        "dataset_schema",
        "dataset_manifest_sha256",
        "dataset_teacher_row_content_set_sha256",
    }:
        raise DistillationTrainingError("training source contract invalid")
    try:
        dataset = _LOAD_DISTILLATION_DATASET(
            teacher_bundle_dir,
            expected_teacher_manifest_sha256=source[
                "teacher_bundle_manifest_sha256"
            ],
            required_splits=("fit", "dev", "test"),
        )
        _VERIFY_DISTILLATION_DATASET(dataset)
    except (DistillationDatasetError, TypeError, ValueError) as exc:
        raise DistillationTrainingError(
            f"fresh artifact dataset verification failed: {exc}"
        ) from exc
    if (
        source["dataset_schema"] != dataset.manifest["schema"]
        or source["dataset_manifest_sha256"]
        != dataset.manifest["dataset_manifest_sha256"]
        or source["dataset_teacher_row_content_set_sha256"]
        != dataset.manifest["source_contract"]["teacher_row_content_set_sha256"]
    ):
        raise DistillationTrainingError("fresh dataset provenance mismatch")

    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "selected_epoch",
        "selected_key",
        "selection_source_split",
    } or selection.get("selection_source_split") != "dev":
        raise DistillationTrainingError("manifest selection contract invalid")
    selected_epoch = _strict_int(
        selection["selected_epoch"],
        label="selected_epoch",
        minimum=1,
        maximum=config.epochs,
    )
    checkpoint_ref = manifest.get("checkpoint")
    if not isinstance(checkpoint_ref, dict):
        raise DistillationTrainingError("checkpoint reference missing")
    _exact_keys(
        checkpoint_ref,
        {
            "relative_path",
            "file_sha256",
            "checkpoint_schema",
            "header_sha256",
            "model_state_sha256",
            "selected_epoch",
        },
        label="checkpoint reference",
    )
    checkpoint_sha256 = _sha(
        checkpoint_ref["file_sha256"], label="checkpoint file SHA"
    )
    expected_checkpoint_name = f"checkpoint-{checkpoint_sha256}.bin"
    if (
        checkpoint_ref["relative_path"] != expected_checkpoint_name
        or checkpoint_ref["checkpoint_schema"] != CHECKPOINT_SCHEMA
        or checkpoint_ref["selected_epoch"] != selected_epoch
    ):
        raise DistillationTrainingError("checkpoint reference binding mismatch")
    checkpoint_path = _safe_artifact_file(
        root, expected_checkpoint_name, label="checkpoint"
    )
    if not 1 <= checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise DistillationTrainingError("checkpoint file size outside bound")
    checkpoint_payload = checkpoint_path.read_bytes()
    if hashlib.sha256(checkpoint_payload).hexdigest() != checkpoint_sha256:
        raise DistillationTrainingError("checkpoint file content SHA mismatch")
    model, checkpoint_header = _decode_checkpoint(
        checkpoint_payload,
        expected_model_config_sha256=model_config_sha256,
        expected_source_dataset_manifest_sha256=source["dataset_manifest_sha256"],
        expected_training_config_sha256=training_config_sha256,
        expected_selected_epoch=selected_epoch,
    )
    if (
        checkpoint_ref["header_sha256"] != checkpoint_header["header_sha256"]
        or checkpoint_ref["model_state_sha256"]
        != checkpoint_header["model_state_sha256"]
    ):
        raise DistillationTrainingError("checkpoint header reference mismatch")

    history, history_path = _verify_json_ref(
        root,
        manifest.get("history"),
        label="training history",
        prefix="training-history",
        self_hash_field="history_sha256",
    )
    _verify_history(
        history,
        dataset_manifest_sha256=source["dataset_manifest_sha256"],
        training_config=config,
        training_config_sha256=training_config_sha256,
        seed_contract_sha256=seed_contract["seed_contract_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_header=checkpoint_header,
        expected_fit_rows=len(dataset.for_split("fit")),
    )
    if selection["selected_key"] != history["selection"]["selected_key"]:
        raise DistillationTrainingError("manifest/history selection key mismatch")
    metrics, metrics_path = _verify_json_ref(
        root,
        manifest.get("metrics"),
        label="selected metrics",
        prefix="selected-metrics",
        self_hash_field="metrics_sha256",
    )
    _verify_metrics_payload(
        metrics,
        dataset_manifest_sha256=source["dataset_manifest_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        selected_epoch=selected_epoch,
    )
    final_events = history["evaluation_events"][-3:]
    for event, split_name in zip(final_events, ("fit", "dev", "test")):
        if event["binding_sha256"] != metrics["splits"][split_name][
            "evaluation_sha256"
        ]:
            raise DistillationTrainingError(
                "history/final metric evaluation binding mismatch"
            )
    split_usage = manifest.get("split_usage")
    if split_usage != {
        "fit": "gradient_updates_only",
        "dev": "checkpoint_selection_only",
        "test": "selected_checkpoint_single_training_pipeline_evaluation_only",
        "random_resplit_performed": False,
        "training_pipeline_test_evaluation_count": 1,
        "fresh_verifier_test_replay_is_integrity_only": True,
        "fresh_verifier_cannot_change_checkpoint_selection": True,
    }:
        raise DistillationTrainingError("manifest locked split usage mismatch")

    with _deterministic_cpu(config.seed):
        fresh_evaluations = {
            name: _evaluate_split(
                model,
                dataset.for_split(name),
                config,
                include_groups=True,
            )
            for name in ("fit", "dev", "test")
        }
    if metrics["splits"] != fresh_evaluations:
        raise DistillationTrainingError("fresh model/dataset metric replay mismatch")
    expected_files = {
        manifest_name,
        expected_checkpoint_name,
        history_path.name,
        metrics_path.name,
    }
    actual_files: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise DistillationTrainingError("orphan directory/symlink in artifact root")
        actual_files.add(path.name)
    if actual_files != expected_files:
        raise DistillationTrainingError(
            f"artifact file set mismatch: missing={sorted(expected_files-actual_files)}, "
            f"orphan={sorted(actual_files-expected_files)}"
        )
    return VerifiedDistillationTrainingArtifact(
        root=root,
        manifest=manifest,
        history=history,
        metrics=metrics,
        dataset=dataset,
        model=model,
    )


def _stabilize_torch_optimizer_runtime() -> None:
    """Materialize PyTorch's one-time optimizer wrappers before pinning them.

    ``Optimizer.__init__`` lazily installs the Dynamo-disabled ``manual_seed``
    and optimizer-step wrappers.  Pinning before that supported transition
    would make a valid train-then-verify process appear to drift.  An empty CPU
    parameter performs no random initialization or optimization.
    """

    parameter = nn.Parameter(torch.empty(0, dtype=torch.float32, device="cpu"))
    torch.optim.AdamW((parameter,), foreach=False)


_stabilize_torch_optimizer_runtime()


_RUNTIME_SEMANTIC_PATHS = (
    (
        "dataset._assert_transitive_runtime_anchors",
        "_dataset_module",
        ("_assert_transitive_runtime_anchors",),
    ),
    (
        "teacher.verify_teacher_bundle",
        "_teacher_module",
        ("verify_teacher_bundle",),
    ),
    (
        "teacher.verify_teacher_row",
        "_teacher_module",
        ("verify_teacher_row",),
    ),
    (
        "teacher.resolve_t4_second_btn_exact",
        "_teacher_module",
        ("resolve_t4_second_btn_exact",),
    ),
    (
        "teacher.verify_t4_second_btn_exact",
        "_teacher_module",
        ("verify_t4_second_btn_exact",),
    ),
    (
        "encoder.decode_infoset_key",
        "_encoder_module",
        ("decode_infoset_key",),
    ),
    (
        "encoder.encode_infoset_key",
        "_encoder_module",
        ("encode_infoset_key",),
    ),
    (
        "encoder.legal_action_mask",
        "_encoder_module",
        ("legal_action_mask",),
    ),
    (
        "encoder.semantic_action_ids",
        "_encoder_module",
        ("semantic_action_ids",),
    ),
    (
        "encoder.validate_infoset_encoder_manifest",
        "_encoder_module",
        ("validate_infoset_encoder_manifest",),
    ),
    (
        "DistillationTrainingConfig.__init__",
        "DistillationTrainingConfig",
        ("__init__",),
    ),
    ("ModelConfig.__init__", "ModelConfig", ("__init__",)),
    ("_ResidualBlock.__init__", "_ResidualBlock", ("__init__",)),
    ("_ResidualBlock.forward", "_ResidualBlock", ("forward",)),
    (
        "T3T4PolicyValueQNet.__init__",
        "T3T4PolicyValueQNet",
        ("__init__",),
    ),
    (
        "T3T4PolicyValueQNet.forward",
        "T3T4PolicyValueQNet",
        ("forward",),
    ),
    (
        "T3T4PolicyValueQNet.state_dict",
        "T3T4PolicyValueQNet",
        ("state_dict",),
    ),
    (
        "T3T4PolicyValueQNet.load_state_dict",
        "T3T4PolicyValueQNet",
        ("load_state_dict",),
    ),
    ("nn.Module.__call__", "nn", ("Module", "__call__")),
    ("nn.Module._call_impl", "nn", ("Module", "_call_impl")),
    ("nn.Module.parameters", "nn", ("Module", "parameters")),
    ("nn.Module.train", "nn", ("Module", "train")),
    ("nn.Module.eval", "nn", ("Module", "eval")),
    ("nn.Linear", "nn", ("Linear",)),
    ("nn.Linear.forward", "nn", ("Linear", "forward")),
    ("nn.Linear.reset_parameters", "nn", ("Linear", "reset_parameters")),
    ("nn.LayerNorm", "nn", ("LayerNorm",)),
    ("nn.LayerNorm.forward", "nn", ("LayerNorm", "forward")),
    (
        "nn.LayerNorm.reset_parameters",
        "nn",
        ("LayerNorm", "reset_parameters"),
    ),
    ("nn.ModuleList", "nn", ("ModuleList",)),
    ("nn.Parameter", "nn", ("Parameter",)),
    ("F.gelu", "F", ("gelu",)),
    ("F.linear", "F", ("linear",)),
    ("F.layer_norm", "F", ("layer_norm",)),
    ("F.log_softmax", "F", ("log_softmax",)),
    ("F.smooth_l1_loss", "F", ("smooth_l1_loss",)),
    ("torch.optim.AdamW", "torch", ("optim", "AdamW")),
    (
        "torch.optim.AdamW.__init__",
        "torch",
        ("optim", "AdamW", "__init__"),
    ),
    (
        "torch.optim.AdamW.zero_grad",
        "torch",
        ("optim", "AdamW", "zero_grad"),
    ),
    (
        "torch.optim.AdamW.step",
        "torch",
        ("optim", "AdamW", "step"),
    ),
    ("torch.Generator", "torch", ("Generator",)),
    (
        "torch.Generator.manual_seed",
        "torch",
        ("Generator", "manual_seed"),
    ),
    ("torch.randperm", "torch", ("randperm",)),
    ("torch.manual_seed", "torch", ("manual_seed",)),
    ("torch.as_tensor", "torch", ("as_tensor",)),
    ("torch.from_numpy", "torch", ("from_numpy",)),
    ("torch.isfinite", "torch", ("isfinite",)),
    ("torch.equal", "torch", ("equal",)),
    ("torch.allclose", "torch", ("allclose",)),
    ("torch.where", "torch", ("where",)),
    ("torch.zeros_like", "torch", ("zeros_like",)),
    ("torch.ones_like", "torch", ("ones_like",)),
    ("torch.clamp", "torch", ("clamp",)),
    ("torch.finfo", "torch", ("finfo",)),
    ("torch.no_grad", "torch", ("no_grad",)),
    (
        "torch.nn.utils.clip_grad_norm_",
        "torch",
        ("nn", "utils", "clip_grad_norm_"),
    ),
    ("torch.Tensor.backward", "torch", ("Tensor", "backward")),
    ("torch.Tensor.detach", "torch", ("Tensor", "detach")),
    ("torch.Tensor.item", "torch", ("Tensor", "item")),
    ("torch.Tensor.numpy", "torch", ("Tensor", "numpy")),
    ("torch.Tensor.clone", "torch", ("Tensor", "clone")),
    ("torch.random.get_rng_state", "torch", ("random", "get_rng_state")),
    ("torch.random.set_rng_state", "torch", ("random", "set_rng_state")),
    ("torch.get_num_threads", "torch", ("get_num_threads",)),
    ("torch.set_num_threads", "torch", ("set_num_threads",)),
    (
        "torch.are_deterministic_algorithms_enabled",
        "torch",
        ("are_deterministic_algorithms_enabled",),
    ),
    (
        "torch.use_deterministic_algorithms",
        "torch",
        ("use_deterministic_algorithms",),
    ),
    (
        "torch.get_float32_matmul_precision",
        "torch",
        ("get_float32_matmul_precision",),
    ),
    (
        "torch.set_float32_matmul_precision",
        "torch",
        ("set_float32_matmul_precision",),
    ),
    ("np.ascontiguousarray", "np", ("ascontiguousarray",)),
    ("np.asarray", "np", ("asarray",)),
    ("np.array", "np", ("array",)),
    ("np.concatenate", "np", ("concatenate",)),
    ("np.frombuffer", "np", ("frombuffer",)),
    ("np.isfinite", "np", ("isfinite",)),
    ("np.all", "np", ("all",)),
    ("np.max", "np", ("max",)),
    ("np.exp", "np", ("exp",)),
    ("np.sum", "np", ("sum",)),
    ("np.log", "np", ("log",)),
    ("np.clip", "np", ("clip",)),
    ("np.where", "np", ("where",)),
    ("np.argmax", "np", ("argmax",)),
    ("np.arange", "np", ("arange",)),
    ("np.maximum", "np", ("maximum",)),
    ("np.mean", "np", ("mean",)),
    ("np.abs", "np", ("abs",)),
    ("np.sqrt", "np", ("sqrt",)),
    ("np.square", "np", ("square",)),
    ("np.minimum", "np", ("minimum",)),
    ("np.prod", "np", ("prod",)),
    ("hashlib.sha256", "hashlib", ("sha256",)),
    ("json.loads", "json", ("loads",)),
    ("tempfile.mkstemp", "tempfile", ("mkstemp",)),
    ("os.link", "os", ("link",)),
    ("os.fsync", "os", ("fsync",)),
)


def _capture_runtime_semantic_bindings() -> tuple[
    tuple[str, str, tuple[str, ...], Any, tuple[Any, ...]], ...
]:
    bindings: list[
        tuple[str, str, tuple[str, ...], Any, tuple[Any, ...]]
    ] = []
    labels: set[str] = set()
    for label, root_name, attributes in _RUNTIME_SEMANTIC_PATHS:
        if label in labels:
            raise AssertionError(f"duplicate runtime semantic label {label}")
        labels.add(label)
        target = _resolve_runtime_semantic_target(root_name, attributes)
        descriptor_state = _freeze_semantic_descriptor(
            _callable_semantic_descriptor(target)
        )
        bindings.append(
            (label, root_name, attributes, target, descriptor_state)
        )
    return tuple(bindings)


_CANONICAL_RUNTIME_SEMANTIC_BINDINGS = _capture_runtime_semantic_bindings()


_CANONICAL_TRAINER_MODULES = (
    ("_dataset_module", _dataset_module),
    ("_teacher_module", _teacher_module),
    ("_encoder_module", _encoder_module),
    ("_runtime_anchor_module", _runtime_anchor_module),
    ("hashlib", hashlib),
    ("json", json),
    ("math", math),
    ("os", os),
    ("platform", platform),
    ("tempfile", tempfile),
    ("types", types),
    ("torch", torch),
    ("np", np),
    ("nn", nn),
    ("F", F),
)
_CANONICAL_TRAINER_CALLABLES = (
    ("load_distillation_dataset", load_distillation_dataset),
    ("verify_distillation_dataset", verify_distillation_dataset),
    ("_LOAD_DISTILLATION_DATASET", _LOAD_DISTILLATION_DATASET),
    ("_VERIFY_DISTILLATION_DATASET", _VERIFY_DISTILLATION_DATASET),
    (
        "_ASSERT_TRANSITIVE_RUNTIME_ANCHORS",
        _ASSERT_TRANSITIVE_RUNTIME_ANCHORS,
    ),
    ("infoset_encoder_manifest", infoset_encoder_manifest),
    ("legal_action_mask", legal_action_mask),
    ("semantic_action_ids", semantic_action_ids),
    ("validate_infoset_encoder_manifest", validate_infoset_encoder_manifest),
    ("_INFOSET_ENCODER_MANIFEST", _INFOSET_ENCODER_MANIFEST),
    ("_LEGAL_ACTION_MASK", _LEGAL_ACTION_MASK),
    ("_SEMANTIC_ACTION_IDS", _SEMANTIC_ACTION_IDS),
    (
        "_VALIDATE_INFOSET_ENCODER_MANIFEST",
        _VALIDATE_INFOSET_ENCODER_MANIFEST,
    ),
    ("canonical_json", canonical_json),
    ("canonical_sha256", canonical_sha256),
    ("_CANONICAL_JSON", _CANONICAL_JSON),
    ("_CANONICAL_SHA256", _CANONICAL_SHA256),
    ("DistillationTrainingConfig", DistillationTrainingConfig),
    ("ModelConfig", ModelConfig),
    ("_ResidualBlock", _ResidualBlock),
    ("T3T4PolicyValueQNet", T3T4PolicyValueQNet),
    ("_strict_int", _strict_int),
    ("_strict_float", _strict_float),
    ("_sha", _sha),
    ("_self_hash", _self_hash),
    ("_exact_keys", _exact_keys),
    ("_config_payload", _config_payload),
    ("_config_from_payload", _config_from_payload),
    ("_model_config", _model_config),
    ("_model_config_payload", _model_config_payload),
    ("_model_config_from_payload", _model_config_from_payload),
    ("_loss_batch", _loss_batch),
    ("masked_policy_value_q_loss", masked_policy_value_q_loss),
    ("_deterministic_cpu", _deterministic_cpu),
    ("_array_sha256", _array_sha256),
    ("_model_state_sha256", _model_state_sha256),
    ("_predict_split", _predict_split),
    ("_softmax_masked", _softmax_masked),
    ("_metric_payload", _metric_payload),
    ("_group_metrics", _group_metrics),
    ("_evaluate_split", _evaluate_split),
    ("_selection_key", _selection_key),
    ("_clone_state_dict", _clone_state_dict),
    ("_train_and_select", _train_and_select),
    ("_seed_contract", _seed_contract),
    ("_loss_contract", _loss_contract),
    ("_source_file_paths", _source_file_paths),
    ("_file_sha256", _file_sha256),
    ("_source_hashes", _source_hashes),
    ("_semantic_value_descriptor", _semantic_value_descriptor),
    ("_callable_semantic_descriptor", _callable_semantic_descriptor),
    ("_freeze_semantic_descriptor", _freeze_semantic_descriptor),
    ("_resolve_runtime_semantic_target", _resolve_runtime_semantic_target),
    ("_assert_runtime_semantic_graph", _assert_runtime_semantic_graph),
    ("_runtime_semantic_graph", _runtime_semantic_graph),
    ("_live_semantic_bindings", _live_semantic_bindings),
    ("_environment_contract", _environment_contract),
    ("_checkpoint_bytes", _checkpoint_bytes),
    ("_reject_duplicate_keys", _reject_duplicate_keys),
    ("_decode_checkpoint", _decode_checkpoint),
    ("_prepare_artifact_root", _prepare_artifact_root),
    ("_atomic_write_bytes", _atomic_write_bytes),
    ("_json_bytes", _json_bytes),
    ("_write_json_artifact", _write_json_artifact),
    ("_event", _event),
    ("_build_history", _build_history),
    ("_build_metrics", _build_metrics),
    ("_manifest_payload", _manifest_payload),
    ("_safe_artifact_file", _safe_artifact_file),
    ("_read_canonical_json", _read_canonical_json),
    ("_verify_json_ref", _verify_json_ref),
    ("_verify_history", _verify_history),
    ("_verify_metrics_payload", _verify_metrics_payload),
    ("_stabilize_torch_optimizer_runtime", _stabilize_torch_optimizer_runtime),
    ("_capture_runtime_semantic_bindings", _capture_runtime_semantic_bindings),
)
_CANONICAL_TRAINER_SEMANTIC_STATE = tuple(
    (
        name,
        _freeze_semantic_descriptor(_callable_semantic_descriptor(canonical)),
    )
    for name, canonical in _CANONICAL_TRAINER_CALLABLES
)
_CANONICAL_TRAINER_VALUES = (
    ("INFOSET_VECTOR_DIM", INFOSET_VECTOR_DIM),
    ("ACTION_COUNT", ACTION_COUNT),
    ("INFOSET_ENCODER_MANIFEST_SHA256", INFOSET_ENCODER_MANIFEST_SHA256),
    ("ACTION_SEMANTICS_SHA256", ACTION_SEMANTICS_SHA256),
    ("DISTILLATION_DATASET_SCHEMA", DISTILLATION_DATASET_SCHEMA),
    ("TRAINING_CONFIG_SCHEMA", TRAINING_CONFIG_SCHEMA),
    ("MODEL_CONFIG_SCHEMA", MODEL_CONFIG_SCHEMA),
    ("TRAINING_HISTORY_SCHEMA", TRAINING_HISTORY_SCHEMA),
    ("TRAINING_METRICS_SCHEMA", TRAINING_METRICS_SCHEMA),
    ("TRAINING_ARTIFACT_SCHEMA", TRAINING_ARTIFACT_SCHEMA),
    ("CHECKPOINT_SCHEMA", CHECKPOINT_SCHEMA),
    ("EVALUATION_SCHEMA", EVALUATION_SCHEMA),
    ("EVENT_SCHEMA", EVENT_SCHEMA),
    ("CHECKPOINT_MAGIC", CHECKPOINT_MAGIC),
    ("MAX_CHECKPOINT_HEADER_BYTES", MAX_CHECKPOINT_HEADER_BYTES),
    ("MAX_CHECKPOINT_BYTES", MAX_CHECKPOINT_BYTES),
    ("_PINNED_SOURCE_PATHS", _PINNED_SOURCE_PATHS),
    ("_PINNED_SOURCE_SHA256", _PINNED_SOURCE_SHA256),
    ("_RUNTIME_SEMANTIC_PATHS", _RUNTIME_SEMANTIC_PATHS),
    (
        "_CANONICAL_RUNTIME_SEMANTIC_BINDINGS",
        _CANONICAL_RUNTIME_SEMANTIC_BINDINGS,
    ),
    (
        "_CANONICAL_TRAINER_SEMANTIC_STATE",
        _CANONICAL_TRAINER_SEMANTIC_STATE,
    ),
)
_CANONICAL_EXTERNAL_CALLABLES = (
    ("os.link", os, "link", os.link),
    ("os.fsync", os, "fsync", os.fsync),
)


def _guard_trainer_runtime(function: Any) -> Any:
    modules = _CANONICAL_TRAINER_MODULES
    callables = _CANONICAL_TRAINER_CALLABLES
    callable_semantics = _CANONICAL_TRAINER_SEMANTIC_STATE
    values = _CANONICAL_TRAINER_VALUES
    external = _CANONICAL_EXTERNAL_CALLABLES
    runtime_semantics = _CANONICAL_RUNTIME_SEMANTIC_BINDINGS
    describe_callable = _callable_semantic_descriptor
    freeze_descriptor = _freeze_semantic_descriptor
    assert_runtime_graph = _assert_runtime_semantic_graph
    assert_transitive_runtime_anchors = _ASSERT_TRANSITIVE_RUNTIME_ANCHORS

    def guarded(*args: Any, **kwargs: Any) -> Any:
        for name, canonical in modules:
            if globals().get(name) is not canonical:
                raise DistillationTrainingError(
                    f"trainer runtime module alias drift: {name}"
                )
        for name, canonical in callables:
            if globals().get(name) is not canonical:
                raise DistillationTrainingError(
                    f"trainer runtime callable alias drift: {name}"
                )
        for name, expected_state in callable_semantics:
            current = globals().get(name)
            current_state = freeze_descriptor(describe_callable(current))
            if current_state != expected_state:
                raise DistillationTrainingError(
                    f"trainer runtime callable semantics drift: {name}"
                )
        for name, canonical in values:
            current = globals().get(name)
            if type(current) is not type(canonical) or current != canonical:
                raise DistillationTrainingError(
                    f"trainer runtime value binding drift: {name}"
                )
        for label, owner, attribute, canonical in external:
            if getattr(owner, attribute, None) is not canonical:
                raise DistillationTrainingError(
                    f"trainer external callable alias drift: {label}"
                )
        assert_runtime_graph(runtime_semantics)
        try:
            assert_transitive_runtime_anchors()
        except (DistillationDatasetError, RuntimeError) as exc:
            raise DistillationTrainingError(
                f"trainer transitive runtime anchor drift: {exc}"
            ) from exc
        return function(*args, **kwargs)

    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    guarded.__doc__ = function.__doc__
    guarded.__module__ = function.__module__
    return guarded


train_distillation_candidate = _guard_trainer_runtime(train_distillation_candidate)
verify_training_artifact = _guard_trainer_runtime(verify_training_artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an opt-in deterministic T3/T4 policy/value/Q candidate"
    )
    parser.add_argument("--teacher-bundle", required=True)
    parser.add_argument("--expected-teacher-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=DistillationTrainingConfig.seed)
    parser.add_argument("--epochs", type=int, default=DistillationTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=DistillationTrainingConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=DistillationTrainingConfig.hidden_dim)
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=DistillationTrainingConfig.residual_blocks,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DistillationTrainingConfig.learning_rate,
    )
    arguments = parser.parse_args(argv)
    config = DistillationTrainingConfig(
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        hidden_dim=arguments.hidden_dim,
        residual_blocks=arguments.residual_blocks,
        learning_rate=arguments.learning_rate,
    )
    artifact = train_distillation_candidate(
        arguments.teacher_bundle,
        arguments.output_dir,
        expected_teacher_manifest_sha256=arguments.expected_teacher_manifest_sha256,
        config=config,
    )
    print(
        _CANONICAL_JSON(
            {
                "artifact_root": str(artifact.root),
                "manifest_sha256": artifact.manifest["manifest_sha256"],
                "runtime_allowed": artifact.manifest["runtime_allowed"],
                "promotion_eligible": artifact.manifest["promotion_eligible"],
            }
        )
    )
    return 0


__all__ = [
    "DistillationTrainingConfig",
    "DistillationTrainingError",
    "ModelConfig",
    "T3T4PolicyValueQNet",
    "VerifiedDistillationTrainingArtifact",
    "main",
    "masked_policy_value_q_loss",
    "train_distillation_candidate",
    "verify_training_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
