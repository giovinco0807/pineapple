"""Fail-closed M3.1 ``StreetPolicyNetV1`` training and calibration pipeline.

This module is intentionally offline-only.  It binds the immutable M3.1
teacher dataset to four non-overlapping consumers:

* ``train`` updates only the shared policy/value/Q core;
* ``safety-fit`` updates only the detached uncertainty/safe heads;
* ``threshold-lock`` selects first/second-seat thresholds without gradients;
* ``diagnostic-holdout`` reports locked behavior without changing either
  weights or thresholds.

The module does not register an AI profile, make a runtime decision, invoke
cloud APIs, or treat search-teacher values as realized match EV.  Runtime
integration remains a later, separately promoted step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import hu_m31_t3_dataset_contract_v1 as dataset_contract
from .action_key import ActionKey
from .hu_infoset import ActorObservation
from .street_policy_net_v1 import (
    FEATURE_SCHEMA_HASH,
    LOSS_SCHEMA_HASH,
    MAX_LEGAL_ACTIONS,
    StreetPolicyNetV1Config,
    build_authorized_optimizer,
    build_street_policy_net_v1,
    encode_street_policy_batch,
    load_street_policy_checkpoint,
    model_state_sha256,
    parameter_names_for_update,
    save_street_policy_checkpoint,
    street_policy_training_loss,
)


VERIFIED_DATASET_SCHEMA = "hu_m31_t3_verified_dataset_v1"
TRAINING_VIEW_SCHEMA = "hu_m31_t3_street_policy_training_view_v1"
TRAINING_CONFIG_SCHEMA = "hu_m31_t3_street_policy_training_config_v1"
FIT_RECEIPT_SCHEMA = "hu_m31_t3_street_policy_fit_receipt_v1"
CHECKPOINT_BUNDLE_SCHEMA = "hu_m31_t3_street_policy_checkpoint_bundle_v1"
PREDICTION_SCHEMA = "hu_m31_t3_street_policy_gate_prediction_v1"
THRESHOLD_LOCK_SCHEMA = "hu_m31_t3_street_policy_threshold_lock_v1"
DIAGNOSTIC_REPORT_SCHEMA = "hu_m31_t3_street_policy_diagnostic_report_v1"

SPLIT_ROLES = (
    "train",
    "safety-fit",
    "threshold-lock",
    "diagnostic-holdout",
)
SEATS = ("first", "second")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_value(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value, raw


def _read_dataset_canonical_json(
    path: Path, label: str
) -> tuple[dict[str, Any], bytes]:
    """Read the M3.1 artifact dialect, whose canonical JSON ends in LF."""

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical M3.1 JSON") from exc
    if (
        not isinstance(value, dict)
        or raw != dataset_contract.canonical_bytes(value)
    ):
        raise ValueError(f"{label} is not canonical M3.1 JSON")
    return value, raw


def _write_once_canonical(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class VerifiedM31Dataset:
    """A source-replayed binding to the frozen 9,000-pair dataset."""

    plan: Mapping[str, Any]
    merge_manifest: Mapping[str, Any]
    shard_directories: Mapping[str, Path]
    receipt: Mapping[str, Any]

    @property
    def identity_sha256(self) -> str:
        return str(self.receipt["dataset_identity_sha256"])


def verify_immutable_dataset_manifest(
    *,
    plan_path: str | Path,
    merge_manifest_path: str | Path,
    shard_directories: Mapping[str, str | Path],
    expected_merge_file_sha256: str,
) -> VerifiedM31Dataset:
    """Replay the frozen plan, merge, every shard DONE, and every pair hash.

    ``expected_merge_file_sha256`` is mandatory.  Without an independently
    frozen digest, an attacker could replace a manifest and its referenced
    artifacts together.
    """

    if not _is_sha256(expected_merge_file_sha256):
        raise ValueError("expected merge SHA-256 must be a lowercase digest")
    plan_file = Path(plan_path)
    merge_file = Path(merge_manifest_path)
    plan, plan_raw = _read_dataset_canonical_json(
        plan_file, "M3.1 dataset plan"
    )
    merge, merge_raw = _read_dataset_canonical_json(
        merge_file, "M3.1 dataset merge manifest"
    )
    if _sha256_bytes(merge_raw) != expected_merge_file_sha256:
        raise ValueError("M3.1 dataset merge file SHA-256 changed")
    validated_plan = dataset_contract.validate_dataset_plan(plan)
    expected_shards = {
        str(row["shard_id"]) for row in validated_plan["shards"]
    }
    if set(shard_directories) != expected_shards:
        raise ValueError("M3.1 dataset shard-directory grid has gaps or extras")
    resolved: dict[str, Path] = {}
    for shard_id, raw_path in shard_directories.items():
        path = Path(raw_path)
        if not path.is_dir() or path.is_symlink():
            raise ValueError(
                f"M3.1 shard {shard_id} must be a non-symlink directory"
            )
        resolved[shard_id] = path.resolve()
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("M3.1 shard directories must be distinct")
    validated_merge = dataset_contract.validate_merge_manifest(
        merge,
        plan=validated_plan,
        shard_directories=resolved,
    )
    shard_done_hashes = {
        str(row["shard_id"]): str(row["done_sha256"])
        for row in validated_merge["shard_records"]
    }
    identity_payload = {
        "schema": VERIFIED_DATASET_SCHEMA,
        "run_id": dataset_contract.RUN_ID,
        "plan_sha256": dataset_contract.canonical_sha256(validated_plan),
        "plan_file_sha256": _sha256_bytes(plan_raw),
        "merge_manifest_sha256": dataset_contract.canonical_sha256(
            validated_merge
        ),
        "merge_file_sha256": expected_merge_file_sha256,
        "pair_record_aggregate_sha256": validated_merge[
            "pair_record_aggregate_sha256"
        ],
        "shard_done_hashes": shard_done_hashes,
        "split_counts": dict(validated_merge["split_counts"]),
        "paired_hand_count": validated_merge["paired_hand_count"],
        "root_count": validated_merge["root_count"],
        "source_replayed": True,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    receipt = dict(identity_payload)
    receipt["dataset_identity_sha256"] = _canonical_sha256(identity_payload)
    return VerifiedM31Dataset(
        plan=validated_plan,
        merge_manifest=validated_merge,
        shard_directories=resolved,
        receipt=receipt,
    )


@dataclass(frozen=True)
class PolicyTrainingExample:
    """One information-set-safe state plus dense legal-action targets."""

    identity: str
    split_role: str
    seat: str
    observation: Mapping[str, Any]
    legal_action_keys: tuple[str, ...]
    baseline_action_key: str
    action_q: tuple[float, ...]
    baseline_delta: tuple[float, ...]
    teacher_policy: tuple[float, ...]
    state_value: float
    downside_p95: tuple[float, ...] | None = None
    safe: tuple[float, ...] | None = None
    confirmation_delta: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("training example identity cannot be empty")
        if self.split_role not in SPLIT_ROLES:
            raise ValueError("unknown StreetPolicyNetV1 split role")
        if self.seat not in SEATS:
            raise ValueError("training example seat must be first or second")
        observation = ActorObservation.from_dict(dict(self.observation))
        if observation.to_dict() != dict(self.observation):
            raise ValueError("training observation is not canonical")
        if observation.seat != self.seat or observation.street != "T3":
            raise ValueError("M3.1 training examples must be matching-seat T3 states")
        count = len(self.legal_action_keys)
        if count <= 0 or count > MAX_LEGAL_ACTIONS:
            raise ValueError("training example needs 1..232 legal actions")
        if len(set(self.legal_action_keys)) != count:
            raise ValueError("training example contains duplicate ActionKeys")
        canonical_keys = tuple(
            key.to_token()
            for key in sorted(
                (ActionKey.from_token(token) for token in self.legal_action_keys),
                key=ActionKey.sort_key,
            )
        )
        if canonical_keys != self.legal_action_keys:
            raise ValueError("training ActionKeys are not in canonical order")
        baseline = ActionKey.from_token(self.baseline_action_key).to_token()
        if baseline not in self.legal_action_keys:
            raise ValueError("training baseline ActionKey is not legal")
        for name, values in (
            ("action_q", self.action_q),
            ("baseline_delta", self.baseline_delta),
            ("teacher_policy", self.teacher_policy),
        ):
            if len(values) != count or any(
                not math.isfinite(float(value)) for value in values
            ):
                raise ValueError(f"{name} target shape/value changed")
        baseline_index = self.legal_action_keys.index(baseline)
        if not math.isclose(
            self.baseline_delta[baseline_index], 0.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("baseline delta target must be zero")
        if any(value < 0 for value in self.teacher_policy) or not math.isclose(
            sum(self.teacher_policy), 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError("teacher policy must be a probability distribution")
        if not math.isfinite(float(self.state_value)):
            raise ValueError("state value target must be finite")
        optional = (
            self.downside_p95,
            self.safe,
            self.confirmation_delta,
        )
        if any(value is None for value in optional) and not all(
            value is None for value in optional
        ):
            raise ValueError("confirmation risk targets must be all present or absent")
        if self.downside_p95 is not None:
            assert self.safe is not None
            assert self.confirmation_delta is not None
            if not (
                len(self.downside_p95)
                == len(self.safe)
                == len(self.confirmation_delta)
                == count
            ):
                raise ValueError("confirmation risk target shape changed")
            if any(
                not math.isfinite(float(value)) or value < 0
                for value in self.downside_p95
            ):
                raise ValueError("downside p95 targets must be finite and non-negative")
            if any(value not in (0.0, 1.0) for value in self.safe):
                raise ValueError("safe labels must be binary")
            if any(
                not math.isfinite(float(value))
                for value in self.confirmation_delta
            ):
                raise ValueError("confirmation deltas must be finite")
        # This also reconstructs and checks the complete legal ActionKey set.
        encoded = encode_street_policy_batch(
            [observation],
            [self.legal_action_keys],
            [baseline],
        )
        observed = tuple(
            token
            for token in encoded.action_key_tokens[0]
            if token is not None
        )
        if observed != self.legal_action_keys:
            raise ValueError("training ActionKey/index mapping drifted")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "split_role": self.split_role,
            "seat": self.seat,
            "observation": dict(self.observation),
            "legal_action_keys": list(self.legal_action_keys),
            "baseline_action_key": self.baseline_action_key,
            "action_q": list(self.action_q),
            "baseline_delta": list(self.baseline_delta),
            "teacher_policy": list(self.teacher_policy),
            "state_value": self.state_value,
            "downside_p95": (
                list(self.downside_p95)
                if self.downside_p95 is not None
                else None
            ),
            "safe": list(self.safe) if self.safe is not None else None,
            "confirmation_delta": (
                list(self.confirmation_delta)
                if self.confirmation_delta is not None
                else None
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())


def _teacher_policy(values: Sequence[float], temperature: float) -> tuple[float, ...]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("teacher policy temperature must be positive")
    array = np.asarray(values, dtype=np.float64) / temperature
    array -= np.max(array)
    probabilities = np.exp(array)
    probabilities /= probabilities.sum()
    return tuple(float(value) for value in probabilities)


def _example_from_row(
    *,
    row: Mapping[str, Any],
    split_role: str,
    pair_identity: str,
    teacher_policy_temperature: float,
    confirmation_required: bool,
) -> PolicyTrainingExample:
    teacher = dict(row["teacher"])
    targets = list(teacher["action_targets"])
    use_confirmation = confirmation_required
    q_name = "confirmation_q" if use_confirmation else "primary_q"
    delta_name = "confirmation_delta" if use_confirmation else "primary_delta"
    action_q = tuple(float(target[q_name]) for target in targets)
    deltas = tuple(float(target[delta_name]) for target in targets)
    confirmation_delta: tuple[float, ...] | None = None
    downside: tuple[float, ...] | None = None
    safe: tuple[float, ...] | None = None
    if confirmation_required:
        primary_delta = tuple(float(target["primary_delta"]) for target in targets)
        confirmation_delta = tuple(
            float(target["confirmation_delta"]) for target in targets
        )
        downside = tuple(
            max(0.0, primary - confirmed)
            for primary, confirmed in zip(
                primary_delta, confirmation_delta, strict=True
            )
        )
        safe = tuple(
            1.0 if confirmed > 0.0 else 0.0
            for confirmed in confirmation_delta
        )
    return PolicyTrainingExample(
        identity=f"{pair_identity}:{row['seat']}:{row['root_index']}",
        split_role=split_role,
        seat=str(row["seat"]),
        observation=dict(row["observation"]),
        legal_action_keys=tuple(str(token) for token in row["legal_action_keys"]),
        baseline_action_key=str(teacher["baseline_action_key"]),
        action_q=action_q,
        baseline_delta=deltas,
        teacher_policy=_teacher_policy(action_q, teacher_policy_temperature),
        state_value=float(
            teacher["confirmation_state_value"]
            if use_confirmation
            else teacher["state_value"]
        ),
        downside_p95=downside,
        safe=safe,
        confirmation_delta=confirmation_delta,
    )


@dataclass(frozen=True)
class PolicyTrainingDataset:
    """A hash-bound in-memory view consumed by the four pipeline stages."""

    examples: tuple[PolicyTrainingExample, ...]
    manifest: Mapping[str, Any]

    @property
    def identity_sha256(self) -> str:
        return str(self.manifest["training_view_identity_sha256"])

    def for_split(self, split_role: str) -> tuple[PolicyTrainingExample, ...]:
        if split_role not in SPLIT_ROLES:
            raise ValueError("unknown training split role")
        self.validate()
        return tuple(
            example for example in self.examples if example.split_role == split_role
        )

    def validate(self) -> None:
        expected = _build_training_view_manifest(
            self.examples,
            source_dataset_identity_sha256=str(
                self.manifest["source_dataset_identity_sha256"]
            ),
            source_type=str(self.manifest["source_type"]),
        )
        if dict(self.manifest) != expected:
            raise ValueError("StreetPolicyNetV1 training view changed after binding")


def _build_training_view_manifest(
    examples: Sequence[PolicyTrainingExample],
    *,
    source_dataset_identity_sha256: str,
    source_type: str,
) -> dict[str, Any]:
    if not _is_sha256(source_dataset_identity_sha256):
        raise ValueError("source dataset identity must be a SHA-256 digest")
    if source_type not in {"verified_m31_dataset", "synthetic_cpu_smoke"}:
        raise ValueError("unknown StreetPolicyNetV1 source type")
    identities = [example.identity for example in examples]
    if len(set(identities)) != len(identities):
        raise ValueError("training view contains duplicate example identities")
    records = [
        {
            "identity": example.identity,
            "split_role": example.split_role,
            "seat": example.seat,
            "sha256": example.sha256,
        }
        for example in examples
    ]
    split_counts = {
        split: sum(record["split_role"] == split for record in records)
        for split in SPLIT_ROLES
    }
    seat_counts = {
        seat: sum(record["seat"] == seat for record in records) for seat in SEATS
    }
    identity_payload = {
        "schema": TRAINING_VIEW_SCHEMA,
        "source_type": source_type,
        "source_dataset_identity_sha256": source_dataset_identity_sha256,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "loss_schema_hash": LOSS_SCHEMA_HASH,
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "split_counts": split_counts,
        "seat_counts": seat_counts,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    manifest = dict(identity_payload)
    manifest["training_view_identity_sha256"] = _canonical_sha256(identity_payload)
    return manifest


def build_policy_training_dataset(
    examples: Iterable[PolicyTrainingExample],
    *,
    source_dataset_identity_sha256: str | None = None,
    verified_dataset: VerifiedM31Dataset | None = None,
    synthetic_cpu_smoke: bool = False,
) -> PolicyTrainingDataset:
    if synthetic_cpu_smoke:
        if verified_dataset is not None or source_dataset_identity_sha256 is None:
            raise ValueError(
                "synthetic smoke requires only its explicit synthetic source digest"
            )
        source_identity = source_dataset_identity_sha256
    else:
        if (
            verified_dataset is None
            or source_dataset_identity_sha256 is not None
            or verified_dataset.receipt.get("source_replayed") is not True
        ):
            raise ValueError(
                "production training views require a source-replayed "
                "VerifiedM31Dataset"
            )
        source_identity = verified_dataset.identity_sha256
    ordered = tuple(sorted(examples, key=lambda example: example.identity))
    source_type = (
        "synthetic_cpu_smoke" if synthetic_cpu_smoke else "verified_m31_dataset"
    )
    manifest = _build_training_view_manifest(
        ordered,
        source_dataset_identity_sha256=source_identity,
        source_type=source_type,
    )
    result = PolicyTrainingDataset(examples=ordered, manifest=manifest)
    result.validate()
    return result


def load_verified_policy_training_dataset(
    verified: VerifiedM31Dataset,
    *,
    teacher_policy_temperature: float = 1.0,
) -> PolicyTrainingDataset:
    """Load and independently hash every pair referenced by a verified merge."""

    if verified.receipt.get("source_replayed") is not True:
        raise ValueError("training requires a source-replayed M3.1 dataset")
    examples: list[PolicyTrainingExample] = []
    for record in verified.merge_manifest["pair_records"]:
        shard_id = str(record["shard_id"])
        source = verified.shard_directories[shard_id] / str(record["path"])
        pair, raw = _read_dataset_canonical_json(
            source, f"M3.1 pair {record['global_pair_index']}"
        )
        if len(raw) != int(record["bytes"]) or _sha256_bytes(raw) != record["sha256"]:
            raise ValueError("M3.1 pair bytes differ from the immutable merge")
        validated = dataset_contract.validate_pair_result(
            pair,
            plan=verified.plan,
            expected_split=str(record["split"]),
            expected_local_pair_index=int(record["local_pair_index"]),
        )
        confirmation = bool(validated["confirmation_required"])
        split = str(validated["split"])
        # Core consumes all train rows.  Risk fitting, calibration, and the
        # diagnostic report consume only independently confirmed rows.
        if split != "train" and not confirmation:
            continue
        pair_identity = f"pair:{validated['global_pair_index']:06d}"
        examples.extend(
            _example_from_row(
                row=row,
                split_role=split,
                pair_identity=pair_identity,
                teacher_policy_temperature=teacher_policy_temperature,
                confirmation_required=confirmation,
            )
            for row in validated["rows"]
        )
    result = build_policy_training_dataset(
        examples,
        verified_dataset=verified,
        synthetic_cpu_smoke=False,
    )
    _validate_production_split_view(result.examples)
    return result


def _validate_production_split_view(
    examples: Sequence[PolicyTrainingExample],
) -> None:
    """Require the exact pair/seat/confirmation consumers frozen by M3.1."""

    expected_pairs = {
        str(spec["split"]): int(spec["paired_hand_count"])
        for spec in dataset_contract.SPLIT_SPECS
    }
    if set(expected_pairs) != set(SPLIT_ROLES):
        raise RuntimeError("M3.1 dataset split specification changed")
    for split in SPLIT_ROLES:
        pair_count = expected_pairs[split]
        if pair_count % dataset_contract.CONFIRMATION_MODULUS:
            raise RuntimeError("M3.1 confirmation ratio no longer divides a split")
        confirmation_pairs = pair_count // dataset_contract.CONFIRMATION_MODULUS
        expected_examples = (
            pair_count * len(SEATS)
            if split == "train"
            else confirmation_pairs * len(SEATS)
        )
        expected_confirmed = confirmation_pairs * len(SEATS)
        rows = [example for example in examples if example.split_role == split]
        confirmed = [
            example
            for example in rows
            if example.confirmation_delta is not None
        ]
        seat_counts = {
            seat: sum(example.seat == seat for example in rows)
            for seat in SEATS
        }
        confirmed_seat_counts = {
            seat: sum(example.seat == seat for example in confirmed)
            for seat in SEATS
        }
        if (
            len(rows) != expected_examples
            or len(confirmed) != expected_confirmed
            or seat_counts
            != {seat: expected_examples // len(SEATS) for seat in SEATS}
            or confirmed_seat_counts
            != {seat: confirmation_pairs for seat in SEATS}
        ):
            raise ValueError(
                f"{split} training-view pair/seat/confirmation coverage changed"
            )


@dataclass(frozen=True)
class StreetPolicyTrainingConfig:
    seed: int = 31_000_001
    ensemble_size: int = 3
    batch_size: int = 64
    core_epochs: int = 2
    risk_epochs: int = 2
    core_learning_rate: float = 3e-4
    risk_learning_rate: float = 3e-4
    teacher_policy_temperature: float = 1.0
    disagreement_multiplier: float = 1.0
    downside_multiplier: float = 1.0
    maximum_false_positive_rate: float = 0.30
    minimum_lock_fires_per_seat: int = 25

    def __post_init__(self) -> None:
        integer_positive = (
            self.ensemble_size,
            self.batch_size,
            self.core_epochs,
            self.risk_epochs,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in integer_positive
        ):
            raise ValueError("ensemble/batch/epoch values must be positive integers")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("training seed must be a non-negative integer")
        if (
            not isinstance(self.minimum_lock_fires_per_seat, int)
            or isinstance(self.minimum_lock_fires_per_seat, bool)
            or self.minimum_lock_fires_per_seat <= 0
        ):
            raise ValueError("minimum lock fires must be a positive integer")
        for name in (
            "core_learning_rate",
            "risk_learning_rate",
            "teacher_policy_temperature",
            "disagreement_multiplier",
            "downside_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.maximum_false_positive_rate)
            or not 0 <= self.maximum_false_positive_rate <= 1
        ):
            raise ValueError("maximum false-positive rate must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_CONFIG_SCHEMA,
            **asdict(self),
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def create_deterministic_ensemble(
    torch: Any,
    *,
    training_config: StreetPolicyTrainingConfig,
    model_config: StreetPolicyNetV1Config | None = None,
    device: str | Any = "cpu",
) -> list[Any]:
    resolved = model_config or StreetPolicyNetV1Config()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    cuda_devices: list[int] = []
    if target.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target.index is None
            else int(target.index)
        ]
    models: list[Any] = []
    with torch.random.fork_rng(devices=cuda_devices):
        for index in range(training_config.ensemble_size):
            torch.manual_seed(training_config.seed + index * 1_000_003)
            if target.type == "cuda":
                torch.cuda.manual_seed_all(
                    training_config.seed + index * 1_000_003
                )
            models.append(build_street_policy_net_v1(torch, resolved).to(target))
    return models


def _ordered_examples(
    examples: Sequence[PolicyTrainingExample],
    *,
    seed: int,
    stage: str,
    model_index: int,
    epoch: int,
) -> list[PolicyTrainingExample]:
    def key(example: PolicyTrainingExample) -> bytes:
        return hashlib.sha256(
            (
                f"{seed}:{stage}:{model_index}:{epoch}:{example.identity}"
            ).encode("utf-8")
        ).digest()

    return sorted(examples, key=lambda example: (key(example), example.identity))


def _batches(
    values: Sequence[PolicyTrainingExample], batch_size: int
) -> Iterable[Sequence[PolicyTrainingExample]]:
    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]


def _encoded_examples(
    examples: Sequence[PolicyTrainingExample],
    torch: Any,
    *,
    device: str | Any = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = encode_street_policy_batch(
        [example.observation for example in examples],
        [example.legal_action_keys for example in examples],
        [example.baseline_action_key for example in examples],
    )
    target_device = torch.device(device)
    inputs = encoded.to_torch(torch, device=target_device)
    batch = len(examples)
    action_q = torch.zeros(
        (batch, MAX_LEGAL_ACTIONS),
        dtype=torch.float32,
        device=target_device,
    )
    delta = torch.zeros_like(action_q)
    teacher_policy = torch.zeros_like(action_q)
    downside = torch.zeros_like(action_q)
    safe = torch.zeros_like(action_q)
    for row, example in enumerate(examples):
        count = len(example.legal_action_keys)
        action_q[row, :count] = torch.tensor(
            example.action_q, dtype=torch.float32, device=target_device
        )
        delta[row, :count] = torch.tensor(
            example.baseline_delta,
            dtype=torch.float32,
            device=target_device,
        )
        teacher_policy[row, :count] = torch.tensor(
            example.teacher_policy,
            dtype=torch.float32,
            device=target_device,
        )
        if example.downside_p95 is not None:
            assert example.safe is not None
            downside[row, :count] = torch.tensor(
                example.downside_p95,
                dtype=torch.float32,
                device=target_device,
            )
            safe[row, :count] = torch.tensor(
                example.safe, dtype=torch.float32, device=target_device
            )
    targets = {
        "action_q": action_q,
        "baseline_delta": delta,
        "teacher_policy": teacher_policy,
        "state_value": torch.tensor(
            [example.state_value for example in examples],
            dtype=torch.float32,
            device=target_device,
        ),
        "downside_p95": downside,
        "safe": safe,
    }
    return inputs, targets


def _ensemble_device(torch: Any, models: Sequence[Any]) -> Any:
    devices = {
        str(next(model.parameters()).device)
        for model in models
    }
    if len(devices) != 1:
        raise ValueError("StreetPolicyNetV1 ensemble spans multiple devices")
    device = torch.device(next(iter(devices)))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("StreetPolicyNetV1 model is on unavailable CUDA")
    return device


def _tensor_group_hash(model: Any, names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    state = dict(model.named_parameters())
    for name in sorted(names):
        tensor = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fit_scope(
    torch: Any,
    models: Sequence[Any],
    dataset: PolicyTrainingDataset,
    *,
    training_config: StreetPolicyTrainingConfig,
    split_role: str,
    update_scope: str,
    start_epoch: int,
    end_epoch: int,
) -> dict[str, Any]:
    dataset.validate()
    if len(models) != training_config.ensemble_size:
        raise ValueError("fit ensemble size differs from training config")
    expected_scope = "core" if split_role == "train" else "risk"
    if update_scope != expected_scope or split_role not in {"train", "safety-fit"}:
        raise PermissionError("fit split/update scope is not authorized")
    examples = dataset.for_split(split_role)
    if not examples:
        raise ValueError(f"{split_role} contains no training examples")
    if split_role == "safety-fit" and any(
        example.downside_p95 is None for example in examples
    ):
        raise ValueError("safety-fit requires independent confirmation targets")
    configured_end = (
        training_config.core_epochs
        if update_scope == "core"
        else training_config.risk_epochs
    )
    if not (0 <= start_epoch < end_epoch <= configured_end):
        raise ValueError("fit epoch range is outside the preregistered config")
    before = [model_state_sha256(model) for model in models]
    device = _ensemble_device(torch, models)
    losses: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        authorized = parameter_names_for_update(model, update_scope)
        other_scope = "risk" if update_scope == "core" else "core"
        unauthorized = parameter_names_for_update(model, other_scope)
        unauthorized_before = _tensor_group_hash(model, unauthorized)
        model.train()
        for epoch in range(start_epoch, end_epoch):
            # AdamW is deliberately reset at each epoch boundary.  Therefore a
            # deterministic model checkpoint plus completed_epoch is sufficient
            # to resume exactly; opaque optimizer/RNG state is not required.
            optimizer = build_authorized_optimizer(
                torch,
                model,
                split_role=split_role,
                update_scope=update_scope,
                learning_rate=(
                    training_config.core_learning_rate
                    if update_scope == "core"
                    else training_config.risk_learning_rate
                ),
            )
            ordered = _ordered_examples(
                examples,
                seed=training_config.seed,
                stage=split_role,
                model_index=model_index,
                epoch=epoch,
            )
            epoch_losses: list[float] = []
            for batch_index, batch in enumerate(
                _batches(ordered, training_config.batch_size)
            ):
                torch.manual_seed(
                    training_config.seed
                    + model_index * 1_000_003
                    + epoch * 10_007
                    + batch_index
                )
                inputs, targets = _encoded_examples(
                    batch, torch, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                output = model(**inputs)
                loss = street_policy_training_loss(
                    torch,
                    output,
                    targets,
                    split_role=split_role,
                    update_scope=update_scope,
                )
                loss["total"].backward()
                optimizer.step()
                epoch_losses.append(float(loss["total"].detach().cpu().item()))
            losses.append(
                {
                    "model_index": model_index,
                    "epoch": epoch,
                    "batch_count": len(epoch_losses),
                    "mean_total_loss": float(np.mean(epoch_losses)),
                }
            )
        if _tensor_group_hash(model, unauthorized) != unauthorized_before:
            raise RuntimeError(
                f"{split_role} modified parameters outside {update_scope}"
            )
        if not authorized:
            raise RuntimeError("authorized parameter group is unexpectedly empty")
        model.eval()
    after = [model_state_sha256(model) for model in models]
    identity_payload = {
        "schema": FIT_RECEIPT_SCHEMA,
        "split_role": split_role,
        "update_scope": update_scope,
        "training_view_identity_sha256": dataset.identity_sha256,
        "training_config_sha256": training_config.identity_sha256,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "example_count": len(examples),
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "losses": losses,
        "unauthorized_parameter_change_count": 0,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    receipt = dict(identity_payload)
    receipt["receipt_sha256"] = _canonical_sha256(identity_payload)
    return receipt


def fit_core_from_train(
    torch: Any,
    models: Sequence[Any],
    dataset: PolicyTrainingDataset,
    *,
    training_config: StreetPolicyTrainingConfig,
    start_epoch: int = 0,
    end_epoch: int | None = None,
) -> dict[str, Any]:
    return _fit_scope(
        torch,
        models,
        dataset,
        training_config=training_config,
        split_role="train",
        update_scope="core",
        start_epoch=start_epoch,
        end_epoch=(
            training_config.core_epochs if end_epoch is None else end_epoch
        ),
    )


def fit_risk_from_safety(
    torch: Any,
    models: Sequence[Any],
    dataset: PolicyTrainingDataset,
    *,
    training_config: StreetPolicyTrainingConfig,
    start_epoch: int = 0,
    end_epoch: int | None = None,
) -> dict[str, Any]:
    return _fit_scope(
        torch,
        models,
        dataset,
        training_config=training_config,
        split_role="safety-fit",
        update_scope="risk",
        start_epoch=start_epoch,
        end_epoch=(
            training_config.risk_epochs if end_epoch is None else end_epoch
        ),
    )


def write_ensemble_checkpoint_bundle(
    path: str | Path,
    models: Sequence[Any],
    *,
    dataset: PolicyTrainingDataset,
    training_config: StreetPolicyTrainingConfig,
    stage: str,
    completed_epoch: int,
) -> dict[str, Any]:
    """Write a byte-deterministic, write-once ensemble checkpoint directory."""

    if stage not in {"core", "risk"}:
        raise ValueError("checkpoint stage must be core or risk")
    if len(models) != training_config.ensemble_size:
        raise ValueError("checkpoint ensemble size differs from training config")
    expected_epoch = (
        training_config.core_epochs
        if stage == "core"
        else training_config.risk_epochs
    )
    if not 0 <= completed_epoch <= expected_epoch:
        raise ValueError("completed epoch is outside the training config")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"checkpoint bundle already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"checkpoint temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for model_index, model in enumerate(models):
            filename = f"model_{model_index:02d}.zip"
            checkpoint = temporary / filename
            checkpoint_manifest = save_street_policy_checkpoint(
                checkpoint,
                model,
                provenance={
                    "pipeline": "hu_m31_t3_street_policy_training_v1",
                    "training_view_identity_sha256": dataset.identity_sha256,
                    "training_config_sha256": training_config.identity_sha256,
                    "stage": stage,
                    "completed_epoch": completed_epoch,
                    "model_index": model_index,
                    "epoch_optimizer": "fresh_adamw_per_epoch",
                },
            )
            raw = checkpoint.read_bytes()
            records.append(
                {
                    "model_index": model_index,
                    "path": filename,
                    "bytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                    "model_state_sha256": checkpoint_manifest[
                        "model_state_sha256"
                    ],
                    "checkpoint_identity_sha256": checkpoint_manifest[
                        "checkpoint_identity_sha256"
                    ],
                }
            )
        identity_payload = {
            "schema": CHECKPOINT_BUNDLE_SCHEMA,
            "training_view_identity_sha256": dataset.identity_sha256,
            "training_config": training_config.to_dict(),
            "training_config_sha256": training_config.identity_sha256,
            "stage": stage,
            "completed_epoch": completed_epoch,
            "ensemble_size": len(models),
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "loss_schema_hash": LOSS_SCHEMA_HASH,
            "epoch_optimizer": "fresh_adamw_per_epoch",
            "models": records,
            "teacher_values_are_realized_match_ev": False,
            "current_profile_changed": False,
        }
        manifest = dict(identity_payload)
        manifest["bundle_identity_sha256"] = _canonical_sha256(identity_payload)
        _write_once_canonical(temporary / "manifest.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_ensemble_checkpoint_bundle(
    path: str | Path,
    *,
    torch: Any,
    expected_dataset_identity_sha256: str,
    expected_training_config: StreetPolicyTrainingConfig,
    expected_stage: str,
    expected_bundle_identity_sha256: str,
    map_location: str | Any = "cpu",
) -> tuple[list[Any], dict[str, Any]]:
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("checkpoint bundle must be a non-symlink directory")
    manifest, _raw = _read_canonical_json(
        source / "manifest.json", "checkpoint bundle manifest"
    )
    required = {
        "schema",
        "training_view_identity_sha256",
        "training_config",
        "training_config_sha256",
        "stage",
        "completed_epoch",
        "ensemble_size",
        "feature_schema_hash",
        "loss_schema_hash",
        "epoch_optimizer",
        "models",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
        "bundle_identity_sha256",
    }
    if set(manifest) != required:
        raise ValueError("checkpoint bundle manifest field set changed")
    identity = dict(manifest)
    declared = identity.pop("bundle_identity_sha256")
    if (
        not _is_sha256(expected_bundle_identity_sha256)
        or declared != expected_bundle_identity_sha256
        or declared != _canonical_sha256(identity)
    ):
        raise ValueError("checkpoint bundle identity changed")
    configured_epoch = (
        expected_training_config.core_epochs
        if expected_stage == "core"
        else expected_training_config.risk_epochs
    )
    completed_epoch = manifest["completed_epoch"]
    if (
        manifest["schema"] != CHECKPOINT_BUNDLE_SCHEMA
        or manifest["training_view_identity_sha256"]
        != expected_dataset_identity_sha256
        or manifest["training_config"] != expected_training_config.to_dict()
        or manifest["training_config_sha256"]
        != expected_training_config.identity_sha256
        or manifest["stage"] != expected_stage
        or isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or not 0 <= completed_epoch <= configured_epoch
        or manifest["ensemble_size"] != expected_training_config.ensemble_size
        or manifest["feature_schema_hash"] != FEATURE_SCHEMA_HASH
        or manifest["loss_schema_hash"] != LOSS_SCHEMA_HASH
        or manifest["epoch_optimizer"] != "fresh_adamw_per_epoch"
        or manifest["teacher_values_are_realized_match_ev"] is not False
        or manifest["current_profile_changed"] is not False
    ):
        raise ValueError("checkpoint bundle contract changed")
    records = manifest["models"]
    if not isinstance(records, list) or len(records) != manifest["ensemble_size"]:
        raise ValueError("checkpoint bundle model grid changed")
    expected_files = {"manifest.json"}
    models: list[Any] = []
    for model_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "model_index",
            "path",
            "bytes",
            "sha256",
            "model_state_sha256",
            "checkpoint_identity_sha256",
        }:
            raise ValueError("checkpoint bundle model record changed")
        filename = f"model_{model_index:02d}.zip"
        if record["model_index"] != model_index or record["path"] != filename:
            raise ValueError("checkpoint bundle model order changed")
        checkpoint = source / filename
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError("checkpoint model is missing or a symlink")
        raw = checkpoint.read_bytes()
        if (
            len(raw) != record["bytes"]
            or _sha256_bytes(raw) != record["sha256"]
        ):
            raise ValueError("checkpoint model bytes changed")
        model, checkpoint_manifest = load_street_policy_checkpoint(
            checkpoint, torch=torch, map_location=map_location
        )
        expected_provenance = {
            "pipeline": "hu_m31_t3_street_policy_training_v1",
            "training_view_identity_sha256": expected_dataset_identity_sha256,
            "training_config_sha256": expected_training_config.identity_sha256,
            "stage": expected_stage,
            "completed_epoch": completed_epoch,
            "model_index": model_index,
            "epoch_optimizer": "fresh_adamw_per_epoch",
        }
        if (
            checkpoint_manifest["model_state_sha256"]
            != record["model_state_sha256"]
            or checkpoint_manifest["checkpoint_identity_sha256"]
            != record["checkpoint_identity_sha256"]
            or checkpoint_manifest["provenance"] != expected_provenance
        ):
            raise ValueError("checkpoint model provenance changed")
        models.append(model)
        expected_files.add(filename)
    children = list(source.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("checkpoint bundle contains an unsafe entry")
    observed_files = {child.name for child in children}
    if observed_files != expected_files:
        raise ValueError("checkpoint bundle has missing or unknown files")
    return models, manifest


@dataclass(frozen=True)
class GatePrediction:
    example_identity: str
    seat: str
    action_key: str
    action_index: int
    baseline_action_key: str
    baseline_index: int
    predicted_delta: float
    downside_p95: float
    ensemble_disagreement: float
    safe_probability: float
    lower_bound: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PREDICTION_SCHEMA,
            **asdict(self),
        }


def _validate_gate_predictions(
    examples: Sequence[PolicyTrainingExample],
    predictions: Sequence[GatePrediction],
    training_config: StreetPolicyTrainingConfig,
) -> None:
    by_identity = {example.identity: example for example in examples}
    if (
        len(by_identity) != len(examples)
        or len(predictions) != len(examples)
        or {prediction.example_identity for prediction in predictions}
        != set(by_identity)
    ):
        raise ValueError("StreetPolicyNetV1 prediction identity grid changed")
    for prediction in predictions:
        example = by_identity[prediction.example_identity]
        if (
            prediction.seat != example.seat
            or prediction.action_index < 0
            or prediction.action_index >= len(example.legal_action_keys)
            or prediction.action_key
            != example.legal_action_keys[prediction.action_index]
            or prediction.baseline_index
            != example.legal_action_keys.index(example.baseline_action_key)
            or prediction.baseline_action_key != example.baseline_action_key
            or prediction.downside_p95 < 0
            or prediction.ensemble_disagreement < 0
            or not 0 <= prediction.safe_probability <= 1
        ):
            raise ValueError("StreetPolicyNetV1 gate prediction contract changed")
        values = (
            prediction.predicted_delta,
            prediction.downside_p95,
            prediction.ensemble_disagreement,
            prediction.safe_probability,
            prediction.lower_bound,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("StreetPolicyNetV1 gate prediction is non-finite")
        expected_lower_bound = (
            prediction.predicted_delta
            - training_config.downside_multiplier * prediction.downside_p95
            - training_config.disagreement_multiplier
            * prediction.ensemble_disagreement
        )
        if not math.isclose(
            prediction.lower_bound,
            expected_lower_bound,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("StreetPolicyNetV1 gate lower-bound arithmetic changed")


def predict_safe_gate(
    torch: Any,
    models: Sequence[Any],
    examples: Sequence[PolicyTrainingExample],
    *,
    training_config: StreetPolicyTrainingConfig,
    batch_size: int | None = None,
) -> list[GatePrediction]:
    if len(models) != training_config.ensemble_size:
        raise ValueError("prediction ensemble size differs from training config")
    device = _ensemble_device(torch, models)
    ordered = sorted(examples, key=lambda example: example.identity)
    predictions: list[GatePrediction] = []
    resolved_batch = batch_size or training_config.batch_size
    for batch in _batches(ordered, resolved_batch):
        inputs, _targets = _encoded_examples(batch, torch, device=device)
        outputs = []
        for model in models:
            model.eval()
            with torch.inference_mode():
                outputs.append(model(**inputs))
        legal = inputs["legal_action_mask"].cpu().numpy()
        deltas = np.stack(
            [
                output["baseline_delta"].detach().cpu().numpy()
                for output in outputs
            ],
            axis=0,
        )
        downsides = np.stack(
            [
                output["uncertainty_p95"].detach().cpu().numpy()
                for output in outputs
            ],
            axis=0,
        )
        safe_probabilities = np.stack(
            [
                output["safe_probability"].detach().cpu().numpy()
                for output in outputs
            ],
            axis=0,
        )
        mean_delta = deltas.mean(axis=0)
        disagreement = deltas.std(axis=0, ddof=0)
        mean_downside = downsides.mean(axis=0)
        mean_safe = safe_probabilities.mean(axis=0)
        for row, example in enumerate(batch):
            legal_count = int(legal[row].sum())
            # np.argmax returns the first index, matching canonical ActionKey
            # tie-breaking because legal actions occupy a canonical prefix.
            candidate_index = int(np.argmax(mean_delta[row, :legal_count]))
            baseline_index = example.legal_action_keys.index(
                example.baseline_action_key
            )
            predicted_delta = float(mean_delta[row, candidate_index])
            downside = float(mean_downside[row, candidate_index])
            disagreement_value = float(disagreement[row, candidate_index])
            safe_probability = float(mean_safe[row, candidate_index])
            lower_bound = (
                predicted_delta
                - training_config.downside_multiplier * downside
                - training_config.disagreement_multiplier
                * disagreement_value
            )
            values = (
                predicted_delta,
                downside,
                disagreement_value,
                safe_probability,
                lower_bound,
            )
            if any(not math.isfinite(value) for value in values):
                raise ValueError("non-finite StreetPolicyNetV1 gate prediction")
            predictions.append(
                GatePrediction(
                    example_identity=example.identity,
                    seat=example.seat,
                    action_key=example.legal_action_keys[candidate_index],
                    action_index=candidate_index,
                    baseline_action_key=example.baseline_action_key,
                    baseline_index=baseline_index,
                    predicted_delta=predicted_delta,
                    downside_p95=downside,
                    ensemble_disagreement=disagreement_value,
                    safe_probability=safe_probability,
                    lower_bound=lower_bound,
                )
            )
    return predictions


def _gain_metrics(gains: Sequence[float]) -> dict[str, Any]:
    if not gains:
        return {
            "fire_count": 0,
            "gain_sum": 0.0,
            "gain_mean": None,
            "false_positive_count": 0,
            "false_positive_rate": None,
            "loss_p95": None,
            "loss_p99": None,
            "loss_max": None,
        }
    array = np.asarray(gains, dtype=np.float64)
    losses = np.maximum(0.0, -array)
    false_positive_count = int(np.sum(array <= 0.0))
    return {
        "fire_count": len(gains),
        "gain_sum": float(array.sum()),
        "gain_mean": float(array.mean()),
        "false_positive_count": false_positive_count,
        "false_positive_rate": false_positive_count / len(gains),
        "loss_p95": float(np.quantile(losses, 0.95, method="higher")),
        "loss_p99": float(np.quantile(losses, 0.99, method="higher")),
        "loss_max": float(losses.max()),
    }


def lock_seat_thresholds(
    torch: Any,
    models: Sequence[Any],
    dataset: PolicyTrainingDataset,
    *,
    training_config: StreetPolicyTrainingConfig,
) -> dict[str, Any]:
    """Select thresholds on threshold-lock only, with no weight update."""

    examples = dataset.for_split("threshold-lock")
    if not examples or any(
        example.confirmation_delta is None for example in examples
    ):
        raise ValueError("threshold-lock requires confirmed examples")
    model_hashes_before = [model_state_sha256(model) for model in models]
    predictions = predict_safe_gate(
        torch, models, examples, training_config=training_config
    )
    _validate_gate_predictions(examples, predictions, training_config)
    by_identity = {example.identity: example for example in examples}
    seats: dict[str, Any] = {}
    for seat in SEATS:
        seat_predictions = [
            prediction for prediction in predictions if prediction.seat == seat
        ]
        eligible = [
            prediction
            for prediction in seat_predictions
            if prediction.action_index != prediction.baseline_index
            and prediction.lower_bound > 0.0
        ]
        thresholds = sorted(
            {0.0, 1.0, *(prediction.safe_probability for prediction in eligible)}
        )
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for threshold in thresholds:
            fired = [
                prediction
                for prediction in eligible
                if prediction.safe_probability >= threshold
            ]
            gains = [
                float(
                    by_identity[prediction.example_identity].confirmation_delta[
                        prediction.action_index
                    ]
                )
                for prediction in fired
            ]
            metrics = _gain_metrics(gains)
            if (
                metrics["fire_count"]
                >= training_config.minimum_lock_fires_per_seat
                and metrics["false_positive_rate"]
                <= training_config.maximum_false_positive_rate
                and metrics["gain_mean"] > 0.0
            ):
                score = (
                    metrics["gain_sum"],
                    metrics["gain_mean"],
                    -metrics["false_positive_rate"],
                    metrics["fire_count"],
                    threshold,
                )
                candidates.append(
                    (
                        score,
                        {
                            "enabled": True,
                            "safe_probability_threshold": threshold,
                            "eligible_count": len(eligible),
                            "metrics": metrics,
                        },
                    )
                )
        if candidates:
            selected = max(candidates, key=lambda item: item[0])[1]
        else:
            selected = {
                "enabled": False,
                "safe_probability_threshold": 1.0,
                "eligible_count": len(eligible),
                "metrics": _gain_metrics([]),
            }
        seats[seat] = selected
    model_hashes_after = [model_state_sha256(model) for model in models]
    if model_hashes_after != model_hashes_before:
        raise RuntimeError("threshold-lock changed model weights")
    identity_payload = {
        "schema": THRESHOLD_LOCK_SCHEMA,
        "source_split": "threshold-lock",
        "training_view_identity_sha256": dataset.identity_sha256,
        "training_config_sha256": training_config.identity_sha256,
        "model_state_sha256": model_hashes_before,
        "model_state_sha256_after_selection": model_hashes_after,
        "weight_update_count": 0,
        "seat_thresholds": seats,
        "selection_objective": (
            "max_gain_sum_then_mean_then_lower_false_positive_then_count_then_threshold"
        ),
        "gate_formula": (
            "predicted_delta-downside_multiplier*downside_p95-"
            "disagreement_multiplier*ensemble_disagreement>0"
        ),
        "candidate_must_differ_from_baseline": True,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    result = dict(identity_payload)
    result["threshold_lock_sha256"] = _canonical_sha256(identity_payload)
    _validate_threshold_lock(
        result,
        training_config=training_config,
        expected_dataset_identity_sha256=dataset.identity_sha256,
        expected_model_hashes=model_hashes_before,
    )
    return result


def _validate_threshold_lock(
    threshold_lock: Mapping[str, Any],
    *,
    training_config: StreetPolicyTrainingConfig,
    expected_dataset_identity_sha256: str | None = None,
    expected_model_hashes: Sequence[str] | None = None,
) -> dict[str, Any]:
    value = dict(threshold_lock)
    required = {
        "schema",
        "source_split",
        "training_view_identity_sha256",
        "training_config_sha256",
        "model_state_sha256",
        "model_state_sha256_after_selection",
        "weight_update_count",
        "seat_thresholds",
        "selection_objective",
        "gate_formula",
        "candidate_must_differ_from_baseline",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
        "threshold_lock_sha256",
    }
    if set(value) != required:
        raise ValueError("StreetPolicyNetV1 threshold-lock fields changed")
    identity = dict(value)
    declared = identity.pop("threshold_lock_sha256")
    model_hashes = value["model_state_sha256"]
    after_hashes = value["model_state_sha256_after_selection"]
    if (
        value["schema"] != THRESHOLD_LOCK_SCHEMA
        or value["source_split"] != "threshold-lock"
        or value["training_config_sha256"] != training_config.identity_sha256
        or not _is_sha256(declared)
        or declared != _canonical_sha256(identity)
        or not isinstance(model_hashes, list)
        or len(model_hashes) != training_config.ensemble_size
        or any(not _is_sha256(digest) for digest in model_hashes)
        or after_hashes != model_hashes
        or value["weight_update_count"] != 0
        or value["candidate_must_differ_from_baseline"] is not True
        or value["teacher_values_are_realized_match_ev"] is not False
        or value["current_profile_changed"] is not False
    ):
        raise ValueError("StreetPolicyNetV1 threshold-lock contract changed")
    if (
        expected_dataset_identity_sha256 is not None
        and value["training_view_identity_sha256"]
        != expected_dataset_identity_sha256
    ):
        raise ValueError("StreetPolicyNetV1 threshold dataset binding changed")
    if (
        expected_model_hashes is not None
        and model_hashes != list(expected_model_hashes)
    ):
        raise ValueError("StreetPolicyNetV1 threshold model binding changed")
    seats = value["seat_thresholds"]
    if not isinstance(seats, dict) or set(seats) != set(SEATS):
        raise ValueError("StreetPolicyNetV1 seat-threshold grid changed")
    metric_fields = {
        "fire_count",
        "gain_sum",
        "gain_mean",
        "false_positive_count",
        "false_positive_rate",
        "loss_p95",
        "loss_p99",
        "loss_max",
    }
    for seat in SEATS:
        gate = seats[seat]
        if not isinstance(gate, dict) or set(gate) != {
            "enabled",
            "safe_probability_threshold",
            "eligible_count",
            "metrics",
        }:
            raise ValueError("StreetPolicyNetV1 seat-threshold fields changed")
        threshold = gate["safe_probability_threshold"]
        eligible_count = gate["eligible_count"]
        metrics = gate["metrics"]
        if (
            not isinstance(gate["enabled"], bool)
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0 <= float(threshold) <= 1
            or isinstance(eligible_count, bool)
            or not isinstance(eligible_count, int)
            or eligible_count < 0
            or not isinstance(metrics, dict)
            or set(metrics) != metric_fields
        ):
            raise ValueError("StreetPolicyNetV1 seat threshold changed")
        fire_count = metrics["fire_count"]
        false_count = metrics["false_positive_count"]
        if (
            isinstance(fire_count, bool)
            or not isinstance(fire_count, int)
            or fire_count < 0
            or isinstance(false_count, bool)
            or not isinstance(false_count, int)
            or false_count < 0
            or false_count > fire_count
            or isinstance(metrics["gain_sum"], bool)
            or not isinstance(metrics["gain_sum"], (int, float))
            or not math.isfinite(float(metrics["gain_sum"]))
            or fire_count > eligible_count
        ):
            raise ValueError("StreetPolicyNetV1 threshold metrics changed")
        optional_metrics = (
            "gain_mean",
            "false_positive_rate",
            "loss_p95",
            "loss_p99",
            "loss_max",
        )
        if fire_count == 0:
            if any(metrics[name] is not None for name in optional_metrics):
                raise ValueError("empty threshold metrics must use null values")
        elif any(
            isinstance(metrics[name], bool)
            or not isinstance(metrics[name], (int, float))
            or not math.isfinite(float(metrics[name]))
            for name in optional_metrics
        ):
            raise ValueError("nonempty threshold metrics must be finite")
        if gate["enabled"]:
            if (
                fire_count < training_config.minimum_lock_fires_per_seat
                or float(metrics["false_positive_rate"])
                > training_config.maximum_false_positive_rate
                or float(metrics["gain_mean"]) <= 0
                or min(
                    float(metrics["loss_p95"]),
                    float(metrics["loss_p99"]),
                    float(metrics["loss_max"]),
                )
                < 0
                or not (
                    float(metrics["loss_p95"])
                    <= float(metrics["loss_p99"])
                    <= float(metrics["loss_max"])
                )
            ):
                raise ValueError("enabled StreetPolicyNetV1 threshold is unsafe")
        elif fire_count != 0:
            raise ValueError("disabled StreetPolicyNetV1 threshold must not fire")
    return value


def safe_override_decision(
    prediction: GatePrediction,
    threshold_lock: Mapping[str, Any],
    *,
    training_config: StreetPolicyTrainingConfig,
) -> bool:
    validated_lock = _validate_threshold_lock(
        threshold_lock, training_config=training_config
    )
    expected_lower_bound = (
        prediction.predicted_delta
        - training_config.downside_multiplier * prediction.downside_p95
        - training_config.disagreement_multiplier
        * prediction.ensemble_disagreement
    )
    if not math.isclose(
        prediction.lower_bound,
        expected_lower_bound,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("StreetPolicyNetV1 gate lower-bound arithmetic changed")
    seat_gate = validated_lock["seat_thresholds"][prediction.seat]
    return bool(
        seat_gate["enabled"]
        and prediction.action_index != prediction.baseline_index
        and expected_lower_bound > 0.0
        and prediction.safe_probability
        >= float(seat_gate["safe_probability_threshold"])
    )


def report_diagnostic_holdout(
    torch: Any,
    models: Sequence[Any],
    dataset: PolicyTrainingDataset,
    *,
    training_config: StreetPolicyTrainingConfig,
    threshold_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen weights/thresholds to holdout and return diagnostics only."""

    examples = dataset.for_split("diagnostic-holdout")
    if not examples or any(
        example.confirmation_delta is None for example in examples
    ):
        raise ValueError("diagnostic holdout requires confirmed examples")
    if (
        threshold_lock.get("training_view_identity_sha256")
        != dataset.identity_sha256
        or threshold_lock.get("training_config_sha256")
        != training_config.identity_sha256
    ):
        raise ValueError("holdout threshold lock has different provenance")
    hashes_before = [model_state_sha256(model) for model in models]
    _validate_threshold_lock(
        threshold_lock,
        training_config=training_config,
        expected_dataset_identity_sha256=dataset.identity_sha256,
        expected_model_hashes=hashes_before,
    )
    lock_hash_before = _canonical_sha256(threshold_lock)
    predictions = predict_safe_gate(
        torch, models, examples, training_config=training_config
    )
    _validate_gate_predictions(examples, predictions, training_config)
    by_identity = {example.identity: example for example in examples}
    fired_rows: list[tuple[str, float]] = []
    for prediction in predictions:
        if safe_override_decision(
            prediction,
            threshold_lock,
            training_config=training_config,
        ):
            example = by_identity[prediction.example_identity]
            assert example.confirmation_delta is not None
            fired_rows.append(
                (
                    prediction.seat,
                    float(example.confirmation_delta[prediction.action_index]),
                )
            )
    metrics = {
        seat: _gain_metrics(
            [gain for observed_seat, gain in fired_rows if observed_seat == seat]
        )
        for seat in SEATS
    }
    metrics["overall"] = _gain_metrics([gain for _seat, gain in fired_rows])
    hashes_after = [model_state_sha256(model) for model in models]
    if hashes_after != hashes_before:
        raise RuntimeError("diagnostic holdout changed model weights")
    if _canonical_sha256(threshold_lock) != lock_hash_before:
        raise RuntimeError("diagnostic holdout changed locked thresholds")
    identity_payload = {
        "schema": DIAGNOSTIC_REPORT_SCHEMA,
        "source_split": "diagnostic-holdout",
        "diagnostic_only": True,
        "promotion_authorized": False,
        "threshold_research_performed": False,
        "training_view_identity_sha256": dataset.identity_sha256,
        "training_config_sha256": training_config.identity_sha256,
        "threshold_lock_sha256": threshold_lock["threshold_lock_sha256"],
        "model_state_sha256_before": hashes_before,
        "model_state_sha256_after": hashes_after,
        "weight_update_count": 0,
        "example_count": len(examples),
        "metrics": metrics,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    result = dict(identity_payload)
    result["report_sha256"] = _canonical_sha256(identity_payload)
    return result


__all__ = [
    "CHECKPOINT_BUNDLE_SCHEMA",
    "DIAGNOSTIC_REPORT_SCHEMA",
    "FIT_RECEIPT_SCHEMA",
    "GatePrediction",
    "PolicyTrainingDataset",
    "PolicyTrainingExample",
    "StreetPolicyTrainingConfig",
    "THRESHOLD_LOCK_SCHEMA",
    "TRAINING_VIEW_SCHEMA",
    "VERIFIED_DATASET_SCHEMA",
    "VerifiedM31Dataset",
    "build_policy_training_dataset",
    "create_deterministic_ensemble",
    "fit_core_from_train",
    "fit_risk_from_safety",
    "load_ensemble_checkpoint_bundle",
    "load_verified_policy_training_dataset",
    "lock_seat_thresholds",
    "predict_safe_gate",
    "report_diagnostic_holdout",
    "safe_override_decision",
    "verify_immutable_dataset_manifest",
    "write_ensemble_checkpoint_bundle",
]
