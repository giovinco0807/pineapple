"""Artifact-bound approximate-best-response policies for M3.1.

This module is an offline, search-teacher distillation subsystem.  It never
uses locked-population or locked-ABR seeds for fitting.  A development dataset
contains only public ``ActorObservation`` objects, canonical legal ActionKeys,
and preregistered per-action teacher values for the three response families
already frozen by the M3.1 promotion contract.

Each policy preserves a fully validated accepted StreetPolicyNetV1 checkpoint
and learns a small deterministic public-feature residual ranker on top of its
policy logits for T3 only.  T0-T2 are delegated to the pinned legacy
stage19_p0 -> stage18_p1 -> stage9f_p2 chain and T4 stays exact.  The greedy
HU-score family is an approximate, not exact, best response; the foul-pressure
and royalty-denial families are exploitative stress policies, not direct
best-response or Nash-bound claims.  Every checkpoint binds the accepted
candidate, development dataset, training configuration, learned residual,
diagnostics, and public T3 semantic probes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_street_policy_training_v1 as policy_training
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import run_hu_m31_t3_step6d_performance as step6d
from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_actions
from .cards import card_rank, card_suit
from .hu_infoset import ActorObservation
from .hu_m31_t3_promotion_runtime_closure_v1 import ABR_FACTORY_IDS
from .hu_m3_t4_runtime import HuM3T4ExactPolicy
from .street_policy_net_v1 import (
    FEATURE_SCHEMA_HASH,
    encode_street_policy_batch,
    load_street_policy_checkpoint,
    model_state_sha256,
    save_street_policy_checkpoint,
)


ABR_DEVELOPMENT_DATASET_SCHEMA = "hu_m31_t3_abr_development_dataset_v1"
ABR_DEVELOPMENT_EXAMPLE_SCHEMA = "hu_m31_t3_abr_development_example_v1"
ABR_TRAINING_CONFIG_SCHEMA = "hu_m31_t3_abr_training_config_v1"
ABR_CHECKPOINT_SCHEMA = "hu_m31_t3_abr_residual_checkpoint_v1"
ABR_BUNDLE_SCHEMA = "hu_m31_t3_abr_policy_bundle_v1"
ABR_SEMANTIC_PROBE_SCHEMA = "hu_m31_t3_abr_semantic_probe_v1"
ABR_POLICY_MANIFEST_SCHEMA = "hu_m31_t3_abr_policy_manifest_v1"
ABR_FEATURE_SCHEMA = "hu_m31_t3_abr_public_action_features_v1"

RESPONSE_IDS = tuple(
    str(value["response_id"]) for value in promotion.ABR_DESCRIPTORS
)
ABR_LEARNED_STREET = "T3"
ABR_LEGACY_PROFILE_BY_STREET = {
    "T0": "stage19_p0",
    "T1": "stage18_p1",
    "T2": "stage9f_p2",
}
ABR_RUNTIME_STREET_COMPOSITION = {
    **ABR_LEGACY_PROFILE_BY_STREET,
    ABR_LEARNED_STREET: "artifact_bound_learned_abr",
    "T4": "hu_m3_t4_exact_both_seats_v1",
}
ABR_RUNTIME_STREET_COMPOSITION_SHA256 = promotion.canonical_sha256(
    ABR_RUNTIME_STREET_COMPOSITION
)
ABR_RESPONSE_SCIENTIFIC_ROLE = {
    "greedy_search_response": "direct_hu_score_approximate_best_response",
    "foul_pressure_response": (
        "exploitative_stress_family_not_direct_best_response"
    ),
    "royalty_denial_response": (
        "exploitative_stress_family_not_direct_best_response"
    ),
}
_RESPONSE_BY_ID = {
    str(value["response_id"]): dict(value)
    for value in promotion.ABR_DESCRIPTORS
}

ABR_FEATURE_NAMES = (
    "bias",
    "seat_second",
    "street_t0",
    "street_t1",
    "street_t2",
    "street_t3",
    "street_t4",
    "hero_top_fill",
    "hero_middle_fill",
    "hero_bottom_fill",
    "opponent_top_fill",
    "opponent_middle_fill",
    "opponent_bottom_fill",
    "hero_discard_count",
    "opponent_discard_count",
    "action_top_count",
    "action_middle_count",
    "action_bottom_count",
    "action_discard_count",
    "action_top_rank_sum",
    "action_middle_rank_sum",
    "action_bottom_rank_sum",
    "action_discard_rank_sum",
    "action_internal_rank_pair",
    "action_top_rank_match",
    "action_middle_rank_match",
    "action_bottom_rank_match",
    "action_top_suit_match",
    "action_middle_suit_match",
    "action_bottom_suit_match",
)


def _canonical_bytes(value: Any) -> bytes:
    return promotion.canonical_bytes(value)


def _canonical_sha256(value: Any) -> str:
    return promotion.canonical_sha256(value)


ABR_FEATURE_SCHEMA_SHA256 = _canonical_sha256(
    {
        "schema": ABR_FEATURE_SCHEMA,
        "feature_names": list(ABR_FEATURE_NAMES),
        "observation_schema": "regular_ofc_actor_observation_v1",
        "action_key_schema": ACTION_KEY_SCHEMA,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
    }
)

_SHA_CHARS = frozenset("0123456789abcdef")
_DATASET_KEYS = frozenset(
    {
        "schema",
        "status",
        "candidate_plan_sha256",
        "candidate_model_manifest_sha256",
        "candidate_checkpoint_bundle_identity_sha256",
        "candidate_training_view_identity_sha256",
        "source_schedule",
        "locked_evaluation_schedule",
        "locked_seed_training_allowed",
        "feature_schema_sha256",
        "response_ids",
        "example_count",
        "examples",
        "example_aggregate_sha256",
        "teacher_value_perspective",
        "teacher_values_are_realized_locked_match_ev",
        "opponent_private_discards_used",
        "realized_deck_tail_used",
        "current_profile_resolved",
        "dataset_identity_sha256",
    }
)
_EXAMPLE_KEYS = frozenset(
    {
        "schema",
        "example_id",
        "source_seed",
        "observation",
        "observation_fingerprint",
        "action_key_schema",
        "legal_action_keys",
        "legal_action_set_sha256",
        "legal_action_mapping_sha256",
        "family_action_values",
        "family_teacher_action_keys",
    }
)
_RAW_EXAMPLE_KEYS = frozenset(
    {"example_id", "source_seed", "observation", "family_action_values"}
)
_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "status",
        "response_id",
        "family",
        "objective",
        "candidate_plan_sha256",
        "candidate_model_manifest_sha256",
        "candidate_checkpoint_bundle_identity_sha256",
        "candidate_source_model_index",
        "candidate_source_model_file_sha256",
        "candidate_source_model_state_sha256",
        "development_dataset_file_sha256",
        "development_dataset_identity_sha256",
        "source_schedule",
        "locked_evaluation_schedule",
        "locked_seed_training_allowed",
        "feature_schema_sha256",
        "street_policy_feature_schema_sha256",
        "feature_names",
        "score_combiner",
        "policy_factory_id",
        "training_config",
        "weights",
        "development_metrics",
        "semantic_probes",
        "semantic_probe_aggregate_sha256",
        "teacher_value_perspective",
        "teacher_values_are_realized_locked_match_ev",
        "opponent_private_discards_used",
        "realized_deck_tail_used",
        "current_profile_resolved",
        "frozen_before_locked_evaluation",
        "checkpoint_identity_sha256",
    }
)
_PROBE_KEYS = frozenset(
    {
        "schema",
        "example_id",
        "observation",
        "observation_fingerprint",
        "legal_action_set_sha256",
        "legal_action_mapping_sha256",
        "selected_action_key",
        "score_vector_sha256",
    }
)
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "status",
        "candidate_plan_sha256",
        "candidate_model_manifest_sha256",
        "candidate_checkpoint_bundle_identity_sha256",
        "development_dataset_file_sha256",
        "development_dataset_identity_sha256",
        "training_config",
        "families",
        "family_count",
        "family_aggregate_sha256",
        "locked_seed_training_allowed",
        "opponent_private_discards_used",
        "current_profile_resolved",
        "frozen_before_locked_evaluation",
        "named_profile_added",
        "current_profile_changed",
        "runtime_activated",
        "bundle_identity_sha256",
    }
)
_POLICY_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "response_id",
        "family",
        "objective",
        "candidate_plan_sha256",
        "policy_checkpoint_filename",
        "policy_checkpoint_sha256",
        "policy_checkpoint_bytes",
        "policy_checkpoint_format",
        "policy_factory_id",
        "development_schedule",
        "locked_evaluation_schedule",
        "locked_seed_training_allowed",
        "opponent_private_discards_used",
        "current_profile_resolved",
        "frozen_before_locked_evaluation",
        "manifest_identity_sha256",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _read_canonical(
    path: str | Path, label: str
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical object")
    return value, raw


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    raw = _canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise FileExistsError(
                f"immutable ABR artifact changed: {destination}"
            ) from None
    return destination.resolve()


@dataclass(frozen=True)
class AbrTrainingConfig:
    algorithm: str = "deterministic_pairwise_linear_ranker_v1"
    epochs: int = 6
    learning_rate: float = 0.05
    margin: float = 0.10
    l2: float = 0.0001
    target_gap_cap: float = 10.0
    weight_clip: float = 100.0
    weight_round_digits: int = 12
    semantic_probe_count: int = 4

    def __post_init__(self) -> None:
        if self.algorithm != "deterministic_pairwise_linear_ranker_v1":
            raise ValueError("ABR training algorithm changed")
        for name in ("epochs", "weight_round_digits", "semantic_probe_count"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "margin",
            "l2",
            "target_gap_cap",
            "weight_clip",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ABR_TRAINING_CONFIG_SCHEMA, **asdict(self)}


FROZEN_TRAINING_CONFIG = AbrTrainingConfig()


def _rank_sum(cards: Sequence[str]) -> float:
    return sum(card_rank(card) for card in cards) / (14.0 * 5.0)


def _match_count(
    cards: Sequence[str], existing: Sequence[str], *, rank: bool
) -> float:
    if not cards or not existing:
        return 0.0
    if rank:
        existing_values = {card_rank(card) for card in existing}
        count = sum(card_rank(card) in existing_values for card in cards)
    else:
        existing_values = {card_suit(card) for card in existing}
        count = sum(card_suit(card) in existing_values for card in cards)
    return count / max(1, len(cards))


def encode_public_action_features(
    observation: ActorObservation, action: Action
) -> tuple[float, ...]:
    """Encode one public information-set action without hidden fields."""

    if not isinstance(observation, ActorObservation):
        raise TypeError("ABR features require ActorObservation")
    placements = {"top": [], "middle": [], "bottom": []}
    for card, row in action.placements:
        if row not in placements:
            raise ValueError("ABR action row changed")
        placements[row].append(card)
    discard_cards = list(action.discards)
    placed_cards = [
        card for row_cards in placements.values() for card in row_cards
    ]
    ranks = [card_rank(card) for card in placed_cards]
    internal_pair = (
        (len(ranks) - len(set(ranks))) / max(1, len(ranks))
        if ranks
        else 0.0
    )
    street_values = {
        street: float(observation.street == street)
        for street in ("T0", "T1", "T2", "T3", "T4")
    }
    values = (
        1.0,
        float(observation.seat == "second"),
        street_values["T0"],
        street_values["T1"],
        street_values["T2"],
        street_values["T3"],
        street_values["T4"],
        len(observation.hero_board.top) / 3.0,
        len(observation.hero_board.middle) / 5.0,
        len(observation.hero_board.bottom) / 5.0,
        len(observation.opponent_public_board.top) / 3.0,
        len(observation.opponent_public_board.middle) / 5.0,
        len(observation.opponent_public_board.bottom) / 5.0,
        len(observation.hero_private_discards) / 4.0,
        observation.opponent_discard_count / 4.0,
        len(placements["top"]) / 3.0,
        len(placements["middle"]) / 5.0,
        len(placements["bottom"]) / 5.0,
        len(discard_cards) / 1.0,
        _rank_sum(placements["top"]),
        _rank_sum(placements["middle"]),
        _rank_sum(placements["bottom"]),
        _rank_sum(discard_cards),
        internal_pair,
        _match_count(
            placements["top"], observation.hero_board.top, rank=True
        ),
        _match_count(
            placements["middle"], observation.hero_board.middle, rank=True
        ),
        _match_count(
            placements["bottom"], observation.hero_board.bottom, rank=True
        ),
        _match_count(
            placements["top"], observation.hero_board.top, rank=False
        ),
        _match_count(
            placements["middle"], observation.hero_board.middle, rank=False
        ),
        _match_count(
            placements["bottom"], observation.hero_board.bottom, rank=False
        ),
    )
    if len(values) != len(ABR_FEATURE_NAMES) or any(
        not math.isfinite(value) for value in values
    ):
        raise AssertionError("ABR feature schema changed")
    return values


def _legal_actions(
    observation: ActorObservation,
) -> tuple[Action, ...]:
    actions = tuple(
        canonicalize_actions(
            generate_actions(
                observation.hero_board, observation.dealt_cards
            )
        )
    )
    if not actions:
        raise ValueError("ABR observation has no legal actions")
    return actions


def _argmax_index(
    values: Sequence[float], action_keys: Sequence[str]
) -> int:
    if len(values) != len(action_keys) or not values:
        raise ValueError("ABR argmax inputs changed")
    return min(
        range(len(values)),
        key=lambda index: (-float(values[index]), action_keys[index]),
    )


@lru_cache(maxsize=1)
def _locked_seed_values() -> frozenset[int]:
    values = set()
    for schedule, count in (
        (promotion.LOCKED_POPULATION, promotion.POPULATION_SEED_COUNT),
        (promotion.LOCKED_ABR, promotion.ABR_SEED_COUNT),
    ):
        for index in range(count):
            values.update(promotion.seed_values(schedule, index).values())
    return frozenset(values)


def validate_development_dataset(
    value: Mapping[str, Any],
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    dataset = deepcopy(dict(value))
    _exact_keys(dataset, _DATASET_KEYS, "ABR development dataset")
    examples = dataset.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("ABR development examples are missing")
    raw_examples = []
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("ABR development example is missing")
        _exact_keys(example, _EXAMPLE_KEYS, "ABR development example")
        raw_examples.append(
            {
                "example_id": example["example_id"],
                "source_seed": example["source_seed"],
                "observation": example["observation"],
                "family_action_values": example["family_action_values"],
            }
        )
    # Rebuild examples without recursively calling this validator.
    rebuilt = _build_dataset_unvalidated(
        promotion_plan=plan,
        candidate_bundle_manifest=candidate_bundle_manifest,
        raw_examples=raw_examples,
    )
    if dataset != rebuilt:
        raise ValueError("ABR development dataset differs from source replay")
    step6d._reject_hidden(dataset, "abr_development_dataset")
    return dataset


def _build_dataset_unvalidated(
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_manifest: Mapping[str, Any],
    raw_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Internal non-recursive dataset builder used by validation."""

    if not raw_examples:
        raise ValueError("ABR development dataset has no examples")
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    manifest = deepcopy(dict(candidate_bundle_manifest))
    model_sha = plan["artifact_binding"]["model"]["sha256"]
    compact_lock = plan["artifact_binding"]["threshold_lock_content"]
    if (
        manifest.get("schema") != policy_training.CHECKPOINT_BUNDLE_SCHEMA
        or manifest.get("stage") != "risk"
        or not _is_sha256(manifest.get("bundle_identity_sha256"))
        or not _is_sha256(manifest.get("training_view_identity_sha256"))
        or compact_lock["source_checkpoint_bundle_identity_sha256"]
        != manifest["bundle_identity_sha256"]
    ):
        raise ValueError("candidate bundle differs from promotion plan")
    normalized_examples = []
    for raw_value in raw_examples:
        raw = deepcopy(dict(raw_value))
        _exact_keys(raw, _RAW_EXAMPLE_KEYS, "raw ABR example")
        if (
            not isinstance(raw.get("example_id"), str)
            or not raw["example_id"]
            or isinstance(raw.get("source_seed"), bool)
            or not isinstance(raw["source_seed"], int)
            or raw["source_seed"] < 0
            or raw["source_seed"] in _locked_seed_values()
            or not isinstance(raw.get("observation"), Mapping)
        ):
            raise ValueError("ABR example id/source seed/observation changed")
        observation = ActorObservation.from_dict(raw["observation"])
        if observation.street != ABR_LEARNED_STREET:
            raise ValueError("ABR development examples must be T3 only")
        actions = _legal_actions(observation)
        keys = [action_key(action).to_token() for action in actions]
        family_values = raw["family_action_values"]
        if (
            not isinstance(family_values, Mapping)
            or set(family_values) != set(RESPONSE_IDS)
        ):
            raise ValueError("ABR example family grid changed")
        normalized_values = {}
        teacher_keys = {}
        for response_id in RESPONSE_IDS:
            source_values = family_values[response_id]
            if (
                not isinstance(source_values, Sequence)
                or isinstance(source_values, (str, bytes))
                or len(source_values) != len(actions)
            ):
                raise ValueError("ABR teacher target count changed")
            values = [
                _finite(value, f"{response_id} teacher value")
                for value in source_values
            ]
            if any(abs(value) > 200.0 for value in values):
                raise ValueError("ABR teacher target is outside HU score bounds")
            normalized_values[response_id] = values
            teacher_keys[response_id] = keys[
                _argmax_index(values, keys)
            ]
        example = {
            "schema": ABR_DEVELOPMENT_EXAMPLE_SCHEMA,
            "example_id": raw["example_id"],
            "source_seed": raw["source_seed"],
            "observation": observation.to_dict(),
            "observation_fingerprint": observation.fingerprint(),
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_keys": keys,
            "legal_action_set_sha256": legal_action_set_digest(actions),
            "legal_action_mapping_sha256": (
                ordered_action_mapping_digest(actions)
            ),
            "family_action_values": normalized_values,
            "family_teacher_action_keys": teacher_keys,
        }
        step6d._reject_hidden(example, f"abr_example.{raw['example_id']}")
        normalized_examples.append(example)
    normalized_examples.sort(key=lambda value: value["example_id"])
    source_seed_seats = {
        (
            value["source_seed"],
            ActorObservation.from_dict(value["observation"]).seat,
        )
        for value in normalized_examples
    }
    source_seed_counts: dict[int, int] = {}
    for value in normalized_examples:
        seed = int(value["source_seed"])
        source_seed_counts[seed] = source_seed_counts.get(seed, 0) + 1
    if (
        len({value["example_id"] for value in normalized_examples})
        != len(normalized_examples)
        or len(source_seed_seats) != len(normalized_examples)
        or any(count > 2 for count in source_seed_counts.values())
        or len(
            {
                value["observation_fingerprint"]
                for value in normalized_examples
            }
        )
        != len(normalized_examples)
    ):
        raise ValueError("ABR development examples are duplicated")
    identity = {
        "schema": ABR_DEVELOPMENT_DATASET_SCHEMA,
        "status": "frozen_search_teacher_development_only",
        "candidate_plan_sha256": promotion.canonical_sha256(plan),
        "candidate_model_manifest_sha256": model_sha,
        "candidate_checkpoint_bundle_identity_sha256": manifest[
            "bundle_identity_sha256"
        ],
        "candidate_training_view_identity_sha256": manifest[
            "training_view_identity_sha256"
        ],
        "source_schedule": "abr_development",
        "locked_evaluation_schedule": promotion.LOCKED_ABR,
        "locked_seed_training_allowed": False,
        "feature_schema_sha256": ABR_FEATURE_SCHEMA_SHA256,
        "response_ids": list(RESPONSE_IDS),
        "example_count": len(normalized_examples),
        "examples": normalized_examples,
        "example_aggregate_sha256": _canonical_sha256(
            normalized_examples
        ),
        "teacher_value_perspective": "abr_actor_higher_is_better",
        "teacher_values_are_realized_locked_match_ev": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "current_profile_resolved": False,
    }
    return {
        **identity,
        "dataset_identity_sha256": _canonical_sha256(identity),
    }


def build_development_dataset(
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_manifest: Mapping[str, Any],
    raw_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dataset = _build_dataset_unvalidated(
        promotion_plan=promotion_plan,
        candidate_bundle_manifest=candidate_bundle_manifest,
        raw_examples=raw_examples,
    )
    return validate_development_dataset(
        dataset,
        promotion_plan=promotion_plan,
        candidate_bundle_manifest=candidate_bundle_manifest,
    )


def write_development_dataset(
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_manifest: Mapping[str, Any],
    raw_examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    dataset = build_development_dataset(
        promotion_plan=promotion_plan,
        candidate_bundle_manifest=candidate_bundle_manifest,
        raw_examples=raw_examples,
    )
    _write_once(output_path, dataset)
    return dataset


def _load_candidate_bundle(
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_directory: str | Path,
    torch: Any,
) -> tuple[
    dict[str, Any],
    policy_training.StreetPolicyTrainingConfig,
    list[Any],
]:
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    directory = Path(candidate_bundle_directory)
    manifest, raw = _read_canonical(
        directory / "manifest.json", "candidate checkpoint bundle manifest"
    )
    if hashlib.sha256(raw).hexdigest() != plan["artifact_binding"]["model"][
        "sha256"
    ]:
        raise ValueError(
            "candidate checkpoint manifest differs from promotion plan"
        )
    raw_config = manifest.get("training_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("candidate training config is missing")
    config = dict(raw_config)
    if config.pop("schema", None) != policy_training.TRAINING_CONFIG_SCHEMA:
        raise ValueError("candidate training config schema changed")
    training_config = policy_training.StreetPolicyTrainingConfig(**config)
    expected_bundle = plan["artifact_binding"]["threshold_lock_content"][
        "source_checkpoint_bundle_identity_sha256"
    ]
    models, replayed = policy_training.load_ensemble_checkpoint_bundle(
        directory,
        torch=torch,
        expected_dataset_identity_sha256=str(
            manifest["training_view_identity_sha256"]
        ),
        expected_training_config=training_config,
        expected_stage="risk",
        expected_bundle_identity_sha256=expected_bundle,
    )
    if replayed != manifest:
        raise ValueError("candidate checkpoint bundle replay changed")
    return manifest, training_config, models


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return math.fsum(
        float(weight) * float(feature)
        for weight, feature in zip(weights, features, strict=True)
    )


def _example_actions_features(
    example: Mapping[str, Any],
) -> tuple[ActorObservation, tuple[Action, ...], list[tuple[float, ...]]]:
    observation = ActorObservation.from_dict(example["observation"])
    actions = _legal_actions(observation)
    keys = [action_key(action).to_token() for action in actions]
    if (
        keys != example["legal_action_keys"]
        or observation.fingerprint()
        != example["observation_fingerprint"]
    ):
        raise ValueError("ABR example ActionKey mapping changed")
    features = [
        encode_public_action_features(observation, action)
        for action in actions
    ]
    return observation, actions, features


def _street_policy_logits(
    *,
    torch: Any,
    model: Any,
    observation: ActorObservation,
    actions: Sequence[Action],
) -> list[float]:
    """Return accepted StreetPolicyNetV1 logits in exact ActionKey order."""

    keys = [action_key(action).to_token() for action in actions]
    if not keys:
        raise ValueError("ABR StreetPolicyNetV1 scoring needs legal actions")
    encoded = encode_street_policy_batch(
        [observation],
        [keys],
        [keys[0]],
    )
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise TypeError("ABR checkpoint did not load a policy network") from exc
    model.eval()
    with torch.inference_mode():
        output = model(**encoded.to_torch(torch, device=device))
    raw = output.get("policy_logits")
    legal = output.get("legal_action_mask")
    if (
        raw is None
        or legal is None
        or tuple(raw.shape) != (1, 232)
        or tuple(legal.shape) != (1, 232)
        or int(legal[0].sum().item()) != len(actions)
    ):
        raise ValueError("ABR StreetPolicyNetV1 output contract changed")
    values = [
        float(raw[0, index].detach().cpu().item())
        for index in range(len(actions))
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("ABR StreetPolicyNetV1 emitted non-finite logits")
    return values


def _combined_scores(
    *,
    torch: Any,
    model: Any,
    observation: ActorObservation,
    actions: Sequence[Action],
    features: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    base = _street_policy_logits(
        torch=torch,
        model=model,
        observation=observation,
        actions=actions,
    )
    return [
        float(base_score) + _dot(weights, action_features)
        for base_score, action_features in zip(
            base, features, strict=True
        )
    ]


def _fit_weights(
    *,
    torch: Any,
    model: Any,
    dataset: Mapping[str, Any],
    response_id: str,
    config: AbrTrainingConfig,
) -> list[float]:
    weights = [0.0] * len(ABR_FEATURE_NAMES)
    for _epoch in range(config.epochs):
        for example in dataset["examples"]:
            _observation, actions, features = _example_actions_features(
                example
            )
            keys = [action_key(action).to_token() for action in actions]
            base_scores = _street_policy_logits(
                torch=torch,
                model=model,
                observation=_observation,
                actions=actions,
            )
            targets = [
                float(value)
                for value in example["family_action_values"][response_id]
            ]
            best = _argmax_index(targets, keys)
            best_features = features[best]
            best_score = base_scores[best] + _dot(
                weights, best_features
            )
            for index, action_features in enumerate(features):
                if index == best or targets[index] >= targets[best]:
                    continue
                score = base_scores[index] + _dot(
                    weights, action_features
                )
                if best_score - score >= config.margin:
                    continue
                gap_scale = min(
                    1.0,
                    max(
                        0.1,
                        (targets[best] - targets[index])
                        / config.target_gap_cap,
                    ),
                )
                rate = config.learning_rate * gap_scale
                for feature_index in range(len(weights)):
                    update = rate * (
                        best_features[feature_index]
                        - action_features[feature_index]
                    )
                    weights[feature_index] = max(
                        -config.weight_clip,
                        min(
                            config.weight_clip,
                            weights[feature_index] + update,
                        ),
                    )
                best_score = base_scores[best] + _dot(
                    weights, best_features
                )
            shrink = 1.0 - config.learning_rate * config.l2
            weights = [weight * shrink for weight in weights]
        weights = [
            round(weight, config.weight_round_digits)
            for weight in weights
        ]
    return weights


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _select_index(
    *,
    weights: Sequence[float],
    actions: Sequence[Action],
    features: Sequence[Sequence[float]],
) -> tuple[int, list[float]]:
    scores = [_dot(weights, row) for row in features]
    keys = [action_key(action).to_token() for action in actions]
    return _argmax_index(scores, keys), scores


def _development_metrics(
    *,
    torch: Any,
    model: Any,
    dataset: Mapping[str, Any],
    response_id: str,
    weights: Sequence[float],
) -> dict[str, Any]:
    rows = []
    for example in dataset["examples"]:
        observation, actions, features = _example_actions_features(example)
        scores = _combined_scores(
            torch=torch,
            model=model,
            observation=observation,
            actions=actions,
            features=features,
            weights=weights,
        )
        keys = [action_key(action).to_token() for action in actions]
        selected = _argmax_index(scores, keys)
        targets = [
            float(value)
            for value in example["family_action_values"][response_id]
        ]
        best = _argmax_index(targets, keys)
        rows.append(
            {
                "seat": observation.seat,
                "street": observation.street,
                "top1": selected == best,
                "regret": max(0.0, targets[best] - targets[selected]),
            }
        )

    def summary(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        regrets = [float(row["regret"]) for row in subset]
        return {
            "examples": len(subset),
            "top1_rate": (
                math.fsum(bool(row["top1"]) for row in subset)
                / len(subset)
                if subset
                else 0.0
            ),
            "regret_mean": (
                math.fsum(regrets) / len(regrets) if regrets else 0.0
            ),
            "regret_p95": _quantile(regrets, 0.95),
            "regret_max": max(regrets, default=0.0),
        }

    return {
        "score_combiner": (
            "accepted_street_policy_logits_plus_public_linear_residual_v1"
        ),
        "overall": summary(rows),
        "by_seat": {
            seat: summary([row for row in rows if row["seat"] == seat])
            for seat in promotion.SEATS
        },
        "by_street": {
            ABR_LEARNED_STREET: summary(rows)
        },
        "development_only_not_locked_promotion_evidence": True,
    }


def _probe_examples(
    dataset: Mapping[str, Any], count: int
) -> list[Mapping[str, Any]]:
    selected = []
    seen_seats: set[str] = set()
    for example in dataset["examples"]:
        observation = ActorObservation.from_dict(example["observation"])
        if (
            observation.street != ABR_LEARNED_STREET
            or observation.seat in seen_seats
        ):
            continue
        selected.append(example)
        seen_seats.add(observation.seat)
        if len(selected) == count:
            break
    for example in dataset["examples"]:
        if example in selected:
            continue
        observation = ActorObservation.from_dict(example["observation"])
        if observation.street != ABR_LEARNED_STREET:
            raise ValueError("ABR semantic probes must be T3 only")
        selected.append(example)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            "ABR dataset lacks preregistered semantic probe coverage"
        )
    observations = [
        ActorObservation.from_dict(example["observation"])
        for example in selected
    ]
    if {observation.seat for observation in observations} != set(
        promotion.SEATS
    ):
        raise ValueError("ABR semantic probes require both seats")
    if {observation.street for observation in observations} != {
        ABR_LEARNED_STREET
    }:
        raise ValueError("ABR semantic probes require T3 only")
    return selected


def _semantic_probes(
    *,
    torch: Any,
    model: Any,
    dataset: Mapping[str, Any],
    weights: Sequence[float],
    config: AbrTrainingConfig,
) -> list[dict[str, Any]]:
    probes = []
    for example in _probe_examples(
        dataset, config.semantic_probe_count
    ):
        observation, actions, features = _example_actions_features(example)
        scores = _combined_scores(
            torch=torch,
            model=model,
            observation=observation,
            actions=actions,
            features=features,
            weights=weights,
        )
        keys = [action_key(action).to_token() for action in actions]
        selected = _argmax_index(scores, keys)
        probes.append(
            {
                "schema": ABR_SEMANTIC_PROBE_SCHEMA,
                "example_id": example["example_id"],
                "observation": observation.to_dict(),
                "observation_fingerprint": observation.fingerprint(),
                "legal_action_set_sha256": legal_action_set_digest(actions),
                "legal_action_mapping_sha256": (
                    ordered_action_mapping_digest(actions)
                ),
                "selected_action_key": action_key(
                    actions[selected]
                ).to_token(),
                "score_vector_sha256": _canonical_sha256(scores),
            }
        )
    return probes


def _checkpoint(
    *,
    torch: Any,
    model: Any,
    source_model_record: Mapping[str, Any],
    response_id: str,
    plan: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_file_sha256: str,
    config: AbrTrainingConfig,
) -> dict[str, Any]:
    descriptor = _RESPONSE_BY_ID[response_id]
    weights = _fit_weights(
        torch=torch,
        model=model,
        dataset=dataset,
        response_id=response_id,
        config=config,
    )
    probes = _semantic_probes(
        torch=torch,
        model=model,
        dataset=dataset,
        weights=weights,
        config=config,
    )
    identity = {
        "schema": ABR_CHECKPOINT_SCHEMA,
        "status": "frozen_independent_search_teacher_abr",
        "response_id": response_id,
        "family": descriptor["family"],
        "objective": descriptor["objective"],
        "candidate_plan_sha256": promotion.canonical_sha256(plan),
        "candidate_model_manifest_sha256": plan["artifact_binding"][
            "model"
        ]["sha256"],
        "candidate_checkpoint_bundle_identity_sha256": candidate_manifest[
            "bundle_identity_sha256"
        ],
        "candidate_source_model_index": 0,
        "candidate_source_model_file_sha256": source_model_record["sha256"],
        "candidate_source_model_state_sha256": source_model_record[
            "model_state_sha256"
        ],
        "development_dataset_file_sha256": dataset_file_sha256,
        "development_dataset_identity_sha256": dataset[
            "dataset_identity_sha256"
        ],
        "source_schedule": "abr_development",
        "locked_evaluation_schedule": promotion.LOCKED_ABR,
        "locked_seed_training_allowed": False,
        "feature_schema_sha256": ABR_FEATURE_SCHEMA_SHA256,
        "street_policy_feature_schema_sha256": FEATURE_SCHEMA_HASH,
        "feature_names": list(ABR_FEATURE_NAMES),
        "score_combiner": (
            "accepted_street_policy_logits_plus_public_linear_residual_v1"
        ),
        "policy_factory_id": ABR_FACTORY_IDS[response_id],
        "training_config": config.to_dict(),
        "weights": weights,
        "development_metrics": _development_metrics(
            torch=torch,
            model=model,
            dataset=dataset,
            response_id=response_id,
            weights=weights,
        ),
        "semantic_probes": probes,
        "semantic_probe_aggregate_sha256": _canonical_sha256(probes),
        "teacher_value_perspective": "abr_actor_higher_is_better",
        "teacher_values_are_realized_locked_match_ev": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "current_profile_resolved": False,
        "frozen_before_locked_evaluation": True,
    }
    return {
        **identity,
        "checkpoint_identity_sha256": _canonical_sha256(identity),
    }


def _policy_manifest(
    *,
    response_id: str,
    plan: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    descriptor = _RESPONSE_BY_ID[response_id]
    identity = {
        "schema": ABR_POLICY_MANIFEST_SCHEMA,
        "status": "frozen_independent_abr_policy",
        "response_id": response_id,
        "family": descriptor["family"],
        "objective": descriptor["objective"],
        "candidate_plan_sha256": promotion.canonical_sha256(plan),
        "policy_checkpoint_filename": checkpoint_path.name,
        "policy_checkpoint_sha256": _sha256_file(checkpoint_path),
        "policy_checkpoint_bytes": checkpoint_path.stat().st_size,
        "policy_checkpoint_format": "street_policy_net_v1_checkpoint_zip",
        "policy_factory_id": ABR_FACTORY_IDS[response_id],
        "development_schedule": "abr_development",
        "locked_evaluation_schedule": promotion.LOCKED_ABR,
        "locked_seed_training_allowed": False,
        "opponent_private_discards_used": False,
        "current_profile_resolved": False,
        "frozen_before_locked_evaluation": True,
    }
    return {
        **identity,
        "manifest_identity_sha256": _canonical_sha256(identity),
    }


def write_policy_bundle(
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_directory: str | Path,
    development_dataset_path: str | Path,
    output_directory: str | Path,
    torch: Any,
    training_config: AbrTrainingConfig = FROZEN_TRAINING_CONFIG,
) -> dict[str, Any]:
    """Train and freeze all three response artifacts from development data."""

    if training_config != FROZEN_TRAINING_CONFIG:
        raise ValueError("production ABR training config is preregistered")
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    candidate_manifest, _candidate_config, candidate_models = (
        _load_candidate_bundle(
            promotion_plan=plan,
            candidate_bundle_directory=candidate_bundle_directory,
            torch=torch,
        )
    )
    if not candidate_models or not isinstance(
        candidate_manifest.get("models"), list
    ):
        raise ValueError("candidate StreetPolicyNetV1 ensemble is empty")
    source_model = candidate_models[0]
    source_model_record = candidate_manifest["models"][0]
    if (
        source_model_record.get("model_index") != 0
        or source_model_record.get("model_state_sha256")
        != model_state_sha256(source_model)
    ):
        raise ValueError("candidate source checkpoint model changed")
    dataset, dataset_raw = _read_canonical(
        development_dataset_path, "ABR development dataset"
    )
    dataset = validate_development_dataset(
        dataset,
        promotion_plan=plan,
        candidate_bundle_manifest=candidate_manifest,
    )
    dataset_file_sha = hashlib.sha256(dataset_raw).hexdigest()
    destination = Path(output_directory)
    if destination.is_symlink():
        raise ValueError("ABR bundle directory is unsafe")
    destination.mkdir(parents=True, exist_ok=True)
    family_records = []
    for response_id in RESPONSE_IDS:
        checkpoint_path = destination / f"{response_id}.checkpoint.zip"
        checkpoint = _checkpoint(
            torch=torch,
            model=source_model,
            source_model_record=source_model_record,
            response_id=response_id,
            plan=plan,
            candidate_manifest=candidate_manifest,
            dataset=dataset,
            dataset_file_sha256=dataset_file_sha,
            config=training_config,
        )
        checkpoint_manifest = save_street_policy_checkpoint(
            checkpoint_path,
            source_model,
            provenance=checkpoint,
        )
        if (
            checkpoint_manifest.get("provenance") != checkpoint
            or checkpoint_manifest.get("model_state_sha256")
            != source_model_record["model_state_sha256"]
        ):
            raise ValueError("ABR StreetPolicyNetV1 checkpoint replay changed")
        manifest_path = destination / f"{response_id}.manifest.json"
        manifest = _policy_manifest(
            response_id=response_id,
            plan=plan,
            checkpoint_path=checkpoint_path,
        )
        _write_once(manifest_path, manifest)
        family_records.append(
            {
                "response_id": response_id,
                "manifest_filename": manifest_path.name,
                "manifest_sha256": _sha256_file(manifest_path),
                "manifest_bytes": manifest_path.stat().st_size,
                "checkpoint_filename": checkpoint_path.name,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_identity_sha256": checkpoint_manifest[
                    "checkpoint_identity_sha256"
                ],
                "abr_checkpoint_identity_sha256": checkpoint[
                    "checkpoint_identity_sha256"
                ],
                "semantic_probe_aggregate_sha256": checkpoint[
                    "semantic_probe_aggregate_sha256"
                ],
            }
        )
    identity = {
        "schema": ABR_BUNDLE_SCHEMA,
        "status": "frozen_three_family_abr_development_bundle",
        "candidate_plan_sha256": promotion.canonical_sha256(plan),
        "candidate_model_manifest_sha256": plan["artifact_binding"][
            "model"
        ]["sha256"],
        "candidate_checkpoint_bundle_identity_sha256": candidate_manifest[
            "bundle_identity_sha256"
        ],
        "development_dataset_file_sha256": dataset_file_sha,
        "development_dataset_identity_sha256": dataset[
            "dataset_identity_sha256"
        ],
        "training_config": training_config.to_dict(),
        "families": family_records,
        "family_count": len(family_records),
        "family_aggregate_sha256": _canonical_sha256(family_records),
        "locked_seed_training_allowed": False,
        "opponent_private_discards_used": False,
        "current_profile_resolved": False,
        "frozen_before_locked_evaluation": True,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    bundle = {
        **identity,
        "bundle_identity_sha256": _canonical_sha256(identity),
    }
    _write_once(destination / "bundle.json", bundle)
    return validate_policy_bundle(
        bundle,
        directory=destination,
        promotion_plan=plan,
        candidate_bundle_directory=candidate_bundle_directory,
        torch=torch,
    )


def _validate_probe_behavior(
    checkpoint: Mapping[str, Any],
    *,
    torch: Any,
    model: Any,
) -> None:
    weights = [float(value) for value in checkpoint["weights"]]
    probes = checkpoint["semantic_probes"]
    if (
        not isinstance(probes, list)
        or len(probes)
        != FROZEN_TRAINING_CONFIG.semantic_probe_count
    ):
        raise ValueError("ABR semantic probe coverage changed")
    seen_seats = set()
    seen_streets = set()
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise ValueError("ABR semantic probe is missing")
        _exact_keys(probe, _PROBE_KEYS, "ABR semantic probe")
        observation = ActorObservation.from_dict(probe["observation"])
        actions = _legal_actions(observation)
        features = [
            encode_public_action_features(observation, action)
            for action in actions
        ]
        scores = _combined_scores(
            torch=torch,
            model=model,
            observation=observation,
            actions=actions,
            features=features,
            weights=weights,
        )
        keys = [action_key(action).to_token() for action in actions]
        selected = _argmax_index(scores, keys)
        if (
            probe["schema"] != ABR_SEMANTIC_PROBE_SCHEMA
            or probe["observation_fingerprint"]
            != observation.fingerprint()
            or probe["legal_action_set_sha256"]
            != legal_action_set_digest(actions)
            or probe["legal_action_mapping_sha256"]
            != ordered_action_mapping_digest(actions)
            or probe["selected_action_key"]
            != action_key(actions[selected]).to_token()
            or probe["score_vector_sha256"]
            != _canonical_sha256(scores)
        ):
            raise ValueError("ABR semantic probe behavior changed")
        seen_seats.add(observation.seat)
        seen_streets.add(observation.street)
    if seen_seats != set(promotion.SEATS) or seen_streets != {
        ABR_LEARNED_STREET
    }:
        raise ValueError("ABR semantic probe grid changed")


def validate_checkpoint(
    value: Mapping[str, Any],
    *,
    response_id: str,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_manifest: Mapping[str, Any],
    torch: Any,
    model: Any,
) -> dict[str, Any]:
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    checkpoint = deepcopy(dict(value))
    _exact_keys(checkpoint, _CHECKPOINT_KEYS, "ABR checkpoint")
    descriptor = _RESPONSE_BY_ID.get(response_id)
    if descriptor is None:
        raise ValueError("ABR response id is outside the frozen grid")
    weights = checkpoint.get("weights")
    probes = checkpoint.get("semantic_probes")
    source_models = candidate_bundle_manifest.get("models")
    source_record = (
        source_models[0]
        if isinstance(source_models, list) and source_models
        else None
    )
    if (
        not isinstance(weights, list)
        or len(weights) != len(ABR_FEATURE_NAMES)
        or any(not math.isfinite(_finite(value, "ABR weight")) for value in weights)
        or not isinstance(probes, list)
    ):
        raise ValueError("ABR checkpoint weights/probes changed")
    identity = dict(checkpoint)
    declared_identity = identity.pop("checkpoint_identity_sha256")
    if (
        checkpoint["schema"] != ABR_CHECKPOINT_SCHEMA
        or checkpoint["status"] != "frozen_independent_search_teacher_abr"
        or checkpoint["response_id"] != response_id
        or checkpoint["family"] != descriptor["family"]
        or checkpoint["objective"] != descriptor["objective"]
        or checkpoint["candidate_plan_sha256"]
        != promotion.canonical_sha256(plan)
        or checkpoint["candidate_model_manifest_sha256"]
        != plan["artifact_binding"]["model"]["sha256"]
        or checkpoint["candidate_checkpoint_bundle_identity_sha256"]
        != plan["artifact_binding"]["threshold_lock_content"][
            "source_checkpoint_bundle_identity_sha256"
        ]
        or candidate_bundle_manifest.get("bundle_identity_sha256")
        != checkpoint["candidate_checkpoint_bundle_identity_sha256"]
        or checkpoint["candidate_source_model_index"] != 0
        or not isinstance(source_record, Mapping)
        or checkpoint["candidate_source_model_file_sha256"]
        != source_record.get("sha256")
        or checkpoint["candidate_source_model_state_sha256"]
        != source_record.get("model_state_sha256")
        or checkpoint["candidate_source_model_state_sha256"]
        != model_state_sha256(model)
        or not _is_sha256(
            checkpoint["development_dataset_file_sha256"]
        )
        or not _is_sha256(
            checkpoint["development_dataset_identity_sha256"]
        )
        or checkpoint["source_schedule"] != "abr_development"
        or checkpoint["locked_evaluation_schedule"]
        != promotion.LOCKED_ABR
        or checkpoint["locked_seed_training_allowed"] is not False
        or checkpoint["feature_schema_sha256"]
        != ABR_FEATURE_SCHEMA_SHA256
        or checkpoint["street_policy_feature_schema_sha256"]
        != FEATURE_SCHEMA_HASH
        or checkpoint["feature_names"] != list(ABR_FEATURE_NAMES)
        or checkpoint["score_combiner"]
        != "accepted_street_policy_logits_plus_public_linear_residual_v1"
        or checkpoint["policy_factory_id"]
        != ABR_FACTORY_IDS[response_id]
        or checkpoint["training_config"]
        != FROZEN_TRAINING_CONFIG.to_dict()
        or checkpoint["semantic_probe_aggregate_sha256"]
        != _canonical_sha256(probes)
        or checkpoint["teacher_value_perspective"]
        != "abr_actor_higher_is_better"
        or checkpoint[
            "teacher_values_are_realized_locked_match_ev"
        ]
        is not False
        or checkpoint["opponent_private_discards_used"] is not False
        or checkpoint["realized_deck_tail_used"] is not False
        or checkpoint["current_profile_resolved"] is not False
        or checkpoint["frozen_before_locked_evaluation"] is not True
        or not _is_sha256(declared_identity)
        or declared_identity != _canonical_sha256(identity)
    ):
        raise ValueError("ABR checkpoint boundary changed")
    _validate_probe_behavior(
        checkpoint,
        torch=torch,
        model=model,
    )
    step6d._reject_hidden(checkpoint, f"abr_checkpoint.{response_id}")
    return checkpoint


def validate_policy_bundle(
    value: Mapping[str, Any],
    *,
    directory: str | Path,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_directory: str | Path,
    torch: Any,
) -> dict[str, Any]:
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    candidate_manifest, _training_config, _candidate_models = (
        _load_candidate_bundle(
            promotion_plan=plan,
            candidate_bundle_directory=candidate_bundle_directory,
            torch=torch,
        )
    )
    bundle = deepcopy(dict(value))
    _exact_keys(bundle, _BUNDLE_KEYS, "ABR policy bundle")
    root = Path(directory)
    families = bundle.get("families")
    if (
        root.is_symlink()
        or not root.is_dir()
        or not isinstance(families, list)
        or len(families) != len(RESPONSE_IDS)
    ):
        raise ValueError("ABR policy bundle directory/families changed")
    records = []
    for expected_id, raw_record in zip(
        RESPONSE_IDS, families, strict=True
    ):
        if not isinstance(raw_record, Mapping):
            raise ValueError("ABR policy bundle family is missing")
        record = dict(raw_record)
        if record.get("response_id") != expected_id:
            raise ValueError("ABR policy bundle family order changed")
        manifest_path = root / str(record.get("manifest_filename"))
        checkpoint_path = root / str(record.get("checkpoint_filename"))
        if (
            manifest_path.name != record.get("manifest_filename")
            or checkpoint_path.name != record.get("checkpoint_filename")
            or manifest_path.is_symlink()
            or checkpoint_path.is_symlink()
            or not manifest_path.is_file()
            or not checkpoint_path.is_file()
            or _sha256_file(manifest_path)
            != record.get("manifest_sha256")
            or _sha256_file(checkpoint_path)
            != record.get("checkpoint_sha256")
            or manifest_path.stat().st_size
            != record.get("manifest_bytes")
            or checkpoint_path.stat().st_size
            != record.get("checkpoint_bytes")
        ):
            raise ValueError("ABR policy bundle artifact changed")
        checkpoint_model, checkpoint_manifest = (
            load_street_policy_checkpoint(
                checkpoint_path,
                torch=torch,
                map_location="cpu",
            )
        )
        checkpoint = checkpoint_manifest.get("provenance")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("ABR checkpoint provenance is missing")
        checkpoint = validate_checkpoint(
            checkpoint,
            response_id=expected_id,
            promotion_plan=plan,
            candidate_bundle_manifest=candidate_manifest,
            torch=torch,
            model=checkpoint_model,
        )
        manifest, _manifest_raw = _read_canonical(
            manifest_path, "ABR policy manifest"
        )
        _exact_keys(
            manifest, _POLICY_MANIFEST_FIELDS, "ABR policy manifest"
        )
        manifest_identity = dict(manifest)
        declared_manifest_identity = manifest_identity.pop(
            "manifest_identity_sha256"
        )
        if (
            manifest["schema"] != ABR_POLICY_MANIFEST_SCHEMA
            or manifest["status"] != "frozen_independent_abr_policy"
            or manifest["response_id"] != expected_id
            or manifest["candidate_plan_sha256"]
            != promotion.canonical_sha256(plan)
            or manifest["policy_checkpoint_filename"]
            != checkpoint_path.name
            or manifest["policy_checkpoint_sha256"]
            != record["checkpoint_sha256"]
            or manifest["policy_checkpoint_bytes"]
            != record["checkpoint_bytes"]
            or manifest["policy_checkpoint_format"]
            != "street_policy_net_v1_checkpoint_zip"
            or manifest["policy_factory_id"]
            != ABR_FACTORY_IDS[expected_id]
            or declared_manifest_identity
            != _canonical_sha256(manifest_identity)
            or record["checkpoint_identity_sha256"]
            != checkpoint_manifest["checkpoint_identity_sha256"]
            or record["abr_checkpoint_identity_sha256"]
            != checkpoint["checkpoint_identity_sha256"]
            or record["semantic_probe_aggregate_sha256"]
            != checkpoint["semantic_probe_aggregate_sha256"]
        ):
            raise ValueError("ABR policy manifest binding changed")
        records.append(record)
    identity = dict(bundle)
    declared = identity.pop("bundle_identity_sha256")
    if (
        bundle["schema"] != ABR_BUNDLE_SCHEMA
        or bundle["status"]
        != "frozen_three_family_abr_development_bundle"
        or bundle["candidate_plan_sha256"]
        != promotion.canonical_sha256(plan)
        or bundle["candidate_model_manifest_sha256"]
        != plan["artifact_binding"]["model"]["sha256"]
        or bundle["candidate_checkpoint_bundle_identity_sha256"]
        != plan["artifact_binding"]["threshold_lock_content"][
            "source_checkpoint_bundle_identity_sha256"
        ]
        or bundle["training_config"]
        != FROZEN_TRAINING_CONFIG.to_dict()
        or bundle["families"] != records
        or bundle["family_count"] != len(RESPONSE_IDS)
        or bundle["family_aggregate_sha256"]
        != _canonical_sha256(records)
        or bundle["locked_seed_training_allowed"] is not False
        or bundle["opponent_private_discards_used"] is not False
        or bundle["current_profile_resolved"] is not False
        or bundle["frozen_before_locked_evaluation"] is not True
        or bundle["named_profile_added"] is not False
        or bundle["current_profile_changed"] is not False
        or bundle["runtime_activated"] is not False
        or not _is_sha256(declared)
        or declared != _canonical_sha256(identity)
    ):
        raise ValueError("ABR policy bundle boundary changed")
    return bundle


class ArtifactBoundAbrPolicy:
    """Pinned legacy T0-T2, learned T3 policy; T4 must be wrapped exact."""

    def __init__(
        self,
        *,
        seat: str,
        policy_seed: int,
        torch: Any,
        model: Any,
        legacy_policy: object,
        checkpoint: Mapping[str, Any],
        checkpoint_file_sha256: str,
        manifest_file_sha256: str,
    ) -> None:
        if seat not in promotion.SEATS:
            raise ValueError("ABR policy seat changed")
        if (
            isinstance(policy_seed, bool)
            or not isinstance(policy_seed, int)
            or policy_seed < 0
        ):
            raise ValueError("ABR policy seed changed")
        self.seat = seat
        self.policy_seed = policy_seed
        self._torch = torch
        self._model = model
        legacy_context = getattr(legacy_policy, "decision_context", None)
        if (
            getattr(legacy_policy, "seat", None) != seat
            or not callable(
                getattr(legacy_policy, "choose_action_observation", None)
            )
            or not isinstance(legacy_context, Mapping)
            or legacy_context.get("runtime_profile") != "stage19_p0"
            or legacy_context.get("t1_continuation") != "stage18_p1"
            or legacy_context.get("t2_continuation") != "stage9f_p2"
        ):
            raise ValueError(
                "ABR legacy policy seat/interface/street chain changed"
            )
        self._legacy_policy = legacy_policy
        self.response_id = str(checkpoint["response_id"])
        self.abr_response_id = self.response_id
        self.abr_policy_factory_id = ABR_FACTORY_IDS[self.response_id]
        self.abr_policy_artifact_sha256 = checkpoint_file_sha256
        self.abr_policy_manifest_sha256 = manifest_file_sha256
        self.checkpoint_identity_sha256 = str(
            checkpoint["checkpoint_identity_sha256"]
        )
        self.weights = tuple(float(value) for value in checkpoint["weights"])
        self.opponent_private_discards_used = False
        self.current_profile_resolved = False
        self.abr_learned_streets = (ABR_LEARNED_STREET,)
        self.abr_legacy_profile_by_street = dict(
            ABR_LEGACY_PROFILE_BY_STREET
        )
        self.abr_exact_streets = ("T4",)
        self.abr_runtime_street_composition_sha256 = (
            ABR_RUNTIME_STREET_COMPOSITION_SHA256
        )
        self.abr_scientific_role = ABR_RESPONSE_SCIENTIFIC_ROLE[
            self.response_id
        ]
        self.abr_exploitability_claim = (
            "empirical_response_stress_only_not_nash_or_nashconv_bound"
        )

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        if not isinstance(observation, ActorObservation):
            raise TypeError("ABR policy requires ActorObservation")
        if observation.seat != self.seat:
            raise ValueError("ABR policy observation seat changed")
        if observation.street in ABR_LEGACY_PROFILE_BY_STREET:
            return self._legacy_policy.choose_action_observation(
                observation,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
            )
        if observation.street != ABR_LEARNED_STREET:
            raise ValueError(
                "ABR base policy accepts learned T3 only; T4 requires exact wrapper"
            )
        actions = _legal_actions(observation)
        features = [
            encode_public_action_features(observation, action)
            for action in actions
        ]
        scores = _combined_scores(
            torch=self._torch,
            model=self._model,
            observation=observation,
            actions=actions,
            features=features,
            weights=self.weights,
        )
        keys = [action_key(action).to_token() for action in actions]
        selected = _argmax_index(scores, keys)
        return actions[selected]


class ArtifactBoundAbrPolicyFactory:
    """Callable factory whose loaded behavior is pinned by two file hashes."""

    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        legacy_policy_factory: Any,
        checkpoint: Mapping[str, Any],
        checkpoint_path: str | Path,
        checkpoint_file_sha256: str,
        manifest_file_sha256: str,
        exact_t4_solver: Any | None = None,
    ) -> None:
        self.torch = torch
        self.model = model
        if not callable(legacy_policy_factory):
            raise TypeError("ABR legacy policy factory must be callable")
        self.legacy_policy_factory = legacy_policy_factory
        self.checkpoint = deepcopy(dict(checkpoint))
        self.response_id = str(checkpoint["response_id"])
        self.factory_id = ABR_FACTORY_IDS[self.response_id]
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_sha256 = checkpoint_file_sha256
        self.checkpoint_file_sha256 = checkpoint_file_sha256
        self.manifest_file_sha256 = manifest_file_sha256
        self.exact_t4_solver = exact_t4_solver
        self.legacy_profile_by_street = dict(ABR_LEGACY_PROFILE_BY_STREET)
        self.learned_streets = (ABR_LEARNED_STREET,)
        self.exact_streets = ("T4",)
        self.runtime_street_composition_sha256 = (
            ABR_RUNTIME_STREET_COMPOSITION_SHA256
        )
        self.scientific_role = ABR_RESPONSE_SCIENTIFIC_ROLE[
            self.response_id
        ]

    def __call__(self, *, policy_seed: int, seat: str) -> object:
        legacy = self.legacy_policy_factory(
            policy_seed=policy_seed,
            seat=seat,
        )
        policy = ArtifactBoundAbrPolicy(
            seat=seat,
            policy_seed=policy_seed,
            torch=self.torch,
            model=self.model,
            legacy_policy=legacy,
            checkpoint=self.checkpoint,
            checkpoint_file_sha256=self.checkpoint_file_sha256,
            manifest_file_sha256=self.manifest_file_sha256,
        )
        if self.exact_t4_solver is None:
            return policy
        return HuM3T4ExactPolicy(policy, self.exact_t4_solver)


def load_artifact_bound_policy_factory(
    *,
    response_id: str,
    manifest_path: str | Path,
    expected_manifest_file_sha256: str,
    checkpoint_path: str | Path,
    expected_checkpoint_file_sha256: str,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_directory: str | Path,
    torch: Any,
    legacy_policy_factory: Any,
    exact_t4_solver: Any | None = None,
) -> ArtifactBoundAbrPolicyFactory:
    """Load, hash-check, and semantically replay one ABR policy factory."""

    if (
        not _is_sha256(expected_manifest_file_sha256)
        or not _is_sha256(expected_checkpoint_file_sha256)
    ):
        raise ValueError("ABR factory artifact hashes must be pinned")
    manifest, manifest_raw = _read_canonical(
        manifest_path, "ABR policy manifest"
    )
    checkpoint_source = Path(checkpoint_path)
    if checkpoint_source.is_symlink() or not checkpoint_source.is_file():
        raise ValueError("ABR checkpoint is missing or unsafe")
    checkpoint_raw = checkpoint_source.read_bytes()
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != expected_manifest_file_sha256
        or hashlib.sha256(checkpoint_raw).hexdigest()
        != expected_checkpoint_file_sha256
    ):
        raise ValueError("ABR factory pinned artifact changed")
    candidate_manifest, _training_config, _candidate_models = (
        _load_candidate_bundle(
            promotion_plan=promotion_plan,
            candidate_bundle_directory=candidate_bundle_directory,
            torch=torch,
        )
    )
    model, checkpoint_manifest = load_street_policy_checkpoint(
        checkpoint_source,
        torch=torch,
        map_location="cpu",
    )
    checkpoint = checkpoint_manifest.get("provenance")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("ABR checkpoint provenance is missing")
    validated = validate_checkpoint(
        checkpoint,
        response_id=response_id,
        promotion_plan=promotion_plan,
        candidate_bundle_manifest=candidate_manifest,
        torch=torch,
        model=model,
    )
    _exact_keys(manifest, _POLICY_MANIFEST_FIELDS, "ABR policy manifest")
    identity = dict(manifest)
    declared = identity.pop("manifest_identity_sha256")
    descriptor = _RESPONSE_BY_ID[response_id]
    if (
        manifest["schema"] != ABR_POLICY_MANIFEST_SCHEMA
        or manifest["status"] != "frozen_independent_abr_policy"
        or manifest["response_id"] != response_id
        or manifest["family"] != descriptor["family"]
        or manifest["objective"] != descriptor["objective"]
        or manifest["candidate_plan_sha256"]
        != promotion.canonical_sha256(promotion_plan)
        or manifest["policy_checkpoint_filename"]
        != checkpoint_source.name
        or manifest["policy_checkpoint_sha256"]
        != expected_checkpoint_file_sha256
        or manifest["policy_checkpoint_bytes"] != len(checkpoint_raw)
        or manifest["policy_checkpoint_format"]
        != "street_policy_net_v1_checkpoint_zip"
        or manifest["policy_factory_id"]
        != ABR_FACTORY_IDS[response_id]
        or manifest["development_schedule"] != "abr_development"
        or manifest["locked_evaluation_schedule"] != promotion.LOCKED_ABR
        or manifest["locked_seed_training_allowed"] is not False
        or manifest["opponent_private_discards_used"] is not False
        or manifest["current_profile_resolved"] is not False
        or manifest["frozen_before_locked_evaluation"] is not True
        or declared != _canonical_sha256(identity)
    ):
        raise ValueError("ABR factory manifest binding changed")
    return ArtifactBoundAbrPolicyFactory(
        torch=torch,
        model=model,
        legacy_policy_factory=legacy_policy_factory,
        checkpoint=validated,
        checkpoint_path=checkpoint_source,
        checkpoint_file_sha256=expected_checkpoint_file_sha256,
        manifest_file_sha256=expected_manifest_file_sha256,
        exact_t4_solver=exact_t4_solver,
    )


__all__ = [
    "ABR_BUNDLE_SCHEMA",
    "ABR_CHECKPOINT_SCHEMA",
    "ABR_DEVELOPMENT_DATASET_SCHEMA",
    "ABR_FEATURE_NAMES",
    "ABR_FEATURE_SCHEMA_SHA256",
    "ABR_LEARNED_STREET",
    "ABR_LEGACY_PROFILE_BY_STREET",
    "ABR_RESPONSE_SCIENTIFIC_ROLE",
    "ABR_RUNTIME_STREET_COMPOSITION",
    "ABR_RUNTIME_STREET_COMPOSITION_SHA256",
    "ABR_POLICY_MANIFEST_SCHEMA",
    "ABR_SEMANTIC_PROBE_SCHEMA",
    "ArtifactBoundAbrPolicy",
    "ArtifactBoundAbrPolicyFactory",
    "AbrTrainingConfig",
    "FROZEN_TRAINING_CONFIG",
    "RESPONSE_IDS",
    "build_development_dataset",
    "encode_public_action_features",
    "load_artifact_bound_policy_factory",
    "validate_checkpoint",
    "validate_development_dataset",
    "validate_policy_bundle",
    "write_development_dataset",
    "write_policy_bundle",
]
