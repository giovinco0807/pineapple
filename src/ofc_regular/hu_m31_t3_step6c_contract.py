"""Frozen schedule and seed contract for the M3.1 T3 Step 6c pilot.

This module contains no search, cloud, training, profile-resolution, or
activation side effects.  It is the single deterministic source of truth used
by the Step 6c packager, shard runner, and validators.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


STEP6C_CONTRACT_SCHEMA = "hu_joint_policy_m31_t3_step6c_contract_v1"
STEP6C_VALIDATION_SCHEMA = "hu_joint_policy_m31_t3_step6c_contract_validation_v1"
STEP6C_BEHAVIOR_SCHEDULE_SCHEMA = "hu_m31_t3_equal_quota_seeded_block_shuffle_v2"
STEP6C_SCHEDULE_ROW_SCHEMA = "hu_m31_t3_step6c_teacher_schedule_row_v1"
STEP6C_RUN_ID = "hu-m31-step6c-production-label-pilot-v1"
STEP6C_SPLIT = "train"

STEP5_CONTRACT_BYTE_SHA256 = (
    "5f9fab4d844f1a7411a99f9dada2fe289314d4dea578d0c0f6a6d14918263c93"
)
STEP5_CONTRACT_CANONICAL_SHA256 = (
    "04c4298feaed78f327f7fb6601f70a6821c3a91dcd2996cc34becbc4bb38e0b4"
)
STEP5_VALIDATION_SHA256 = (
    "055b900e06db174f69d3ecb25fd1466f9e812d272222529121a40b71dd003bcc"
)
STEP6B_STATUS_SHA256 = (
    "76e327525bb9165732e36fe2f47dd70cf07752f93de7c68e2cdd9e59a99c16f8"
)
STEP6B_VALIDATION_SHA256 = (
    "f95445cc71b402d97eb120cde2cfdbd1235ee57ca7c83b9227bcf5f9c3a238da"
)
POLICY_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
ACCEPTED_NATIVE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
ACCEPTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)

# Filled only after the checked-in JSON contract is frozen.  These constants
# deliberately make every consumer reject a locally edited contract.
EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256 = (
    "0a745fb746766423424f71a00292dac86e6102d15187ef763cc0e7d30ed49123"
)
EXPECTED_STEP6C_CONTRACT_BYTE_SHA256 = (
    "6d9cc9f0bb79423ead58eea24b84474085071f9dd0dbda8cfc5acf89648fb8ce"
)

SEED_STRIDE = 1_000_003
TRAIN_HAND_SEED_BASE = 310_108_071_901
TRAIN_BEHAVIOR_SEED_BASE = 317_108_071_901
TRAIN_CANDIDATE_SEED_BASE = 324_108_071_901
TRAIN_EVALUATION_SEED_BASE = 331_108_071_901
TRAIN_CHILD_SEED_BASE = 338_108_071_901
TRAIN_CONFIRMATION_SEED_BASE = 345_108_071_901

# Compatibility aliases keep downstream code explicit while matching the key
# names already used by the accepted Step 6a/6b artifacts.
HAND_SEED_BASE = TRAIN_HAND_SEED_BASE
BEHAVIOR_SEED_BASE = TRAIN_BEHAVIOR_SEED_BASE
CANDIDATE_SEED_BASE = TRAIN_CANDIDATE_SEED_BASE
EVALUATION_SEED_BASE = TRAIN_EVALUATION_SEED_BASE
CHILD_SEED_BASE = TRAIN_CHILD_SEED_BASE
CONFIRMATION_SEED_BASE = TRAIN_CONFIRMATION_SEED_BASE

TRAIN_HAND_COUNT = 6_000
TRAIN_ROOT_COUNT = 12_000
BEHAVIOR_BLOCK_HANDS = 5
TRAIN_HANDS_PER_PROFILE = 1_200

PILOT_HAND_INDICES = tuple(range(50))
PILOT_ROOT_INDICES = tuple(range(100))
PILOT_HAND_COUNT = len(PILOT_HAND_INDICES)
PILOT_ROOT_COUNT = len(PILOT_ROOT_INDICES)
PILOT_SHARD_COUNT = 2
PILOT_HANDS_PER_SHARD = 25
PILOT_ROOTS_PER_SHARD = 50
PILOT_HANDS_PER_PROFILE = 10
PILOT_ROOTS_PER_PROFILE = 20

CONFIRMATION_HAND_INDICES = (5, 16, 29, 39, 45)
CONFIRMATION_ROOT_INDICES = (10, 11, 32, 33, 58, 59, 78, 79, 90, 91)
CONFIRMATION_HAND_COUNT = len(CONFIRMATION_HAND_INDICES)
CONFIRMATION_ROOT_COUNT = len(CONFIRMATION_ROOT_INDICES)
CONFIRMATION_FRACTION = 0.1
PERCENTILE_METHOD = "nearest_rank_ceil_n_times_q_v1"

CONFIRMATION_REGRET_MEAN_MAX = 0.75
CONFIRMATION_REGRET_P95_MAX = 3.0
CONFIRMATION_REGRET_P99_MAX = 6.0
CONFIRMATION_REGRET_MAX = 15.0

MAX_FIRST_P95_SECONDS = 180.0
MAX_SECOND_P95_SECONDS = 6.0
MAX_PEAK_RSS_BYTES = 1_073_741_824


@dataclass(frozen=True)
class Step6CSearchBudget:
    """One immutable T3 search budget."""

    label: str
    candidate_samples: int
    evaluation_samples: int
    downstream_t3_samples: int
    downstream_t4_samples: int = 0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Step 6c budget label must not be empty")
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.downstream_t4_samples != 0:
            raise ValueError("Step 6c requires exact downstream T4")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": self.downstream_t4_samples,
        }


PRODUCTION_LABEL_BUDGET = Step6CSearchBudget(
    "production_label_8_32_4_0",
    candidate_samples=8,
    evaluation_samples=32,
    downstream_t3_samples=4,
)
CONFIRMATION_BUDGET = Step6CSearchBudget(
    "independent_confirmation_8_128_4_0",
    candidate_samples=8,
    evaluation_samples=128,
    downstream_t3_samples=4,
)
PRODUCTION_BUDGET = PRODUCTION_LABEL_BUDGET


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical JSON byte representation used by all Step 6c hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _checked_train_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("train hand index must be an integer")
    if not 0 <= index < TRAIN_HAND_COUNT:
        raise ValueError(f"train hand index must be in 0..{TRAIN_HAND_COUNT - 1}")
    return index


def _profile_sort_digest(block: int, profile: str) -> bytes:
    payload = (
        f"{STEP6C_BEHAVIOR_SCHEDULE_SCHEMA}\0{STEP6C_SPLIT}\0"
        f"{TRAIN_BEHAVIOR_SEED_BASE}\0{block}\0{profile}"
    ).encode("ascii")
    return hashlib.sha256(payload).digest()


def behavior_profile_for_train_index(index: int) -> str:
    """Return the content-independent equal-quota profile for a train hand."""

    index = _checked_train_index(index)
    block, offset = divmod(index, BEHAVIOR_BLOCK_HANDS)
    ordered = sorted(
        M31_T3_BEHAVIOR_PROFILES,
        key=lambda profile: (_profile_sort_digest(block, profile), profile),
    )
    return ordered[offset]


def train_seed_values(index: int) -> dict[str, int]:
    """Return all six disjoint train-namespace seed values for one paired hand."""

    index = _checked_train_index(index)
    offset = SEED_STRIDE * index
    return {
        "hand": TRAIN_HAND_SEED_BASE + offset,
        "behavior": TRAIN_BEHAVIOR_SEED_BASE + offset,
        "candidate": TRAIN_CANDIDATE_SEED_BASE + offset,
        "evaluation": TRAIN_EVALUATION_SEED_BASE + offset,
        "child": TRAIN_CHILD_SEED_BASE + offset,
        "confirmation": TRAIN_CONFIRMATION_SEED_BASE + offset,
    }


def pilot_shard_for_hand_index(index: int) -> int:
    index = _checked_train_index(index)
    if index not in PILOT_HAND_INDICES:
        raise ValueError("hand index is outside the frozen Step 6c pilot")
    return index // PILOT_HANDS_PER_SHARD


def schedule_row(index: int) -> dict[str, Any]:
    index = _checked_train_index(index)
    pilot = index in PILOT_HAND_INDICES
    return {
        "schema": STEP6C_SCHEDULE_ROW_SCHEMA,
        "schedule_schema": STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
        "split": STEP6C_SPLIT,
        "train_hand_index": index,
        "root_indices": [index * 2, index * 2 + 1],
        "profile": behavior_profile_for_train_index(index),
        "seeds": train_seed_values(index),
        "pilot": pilot,
        "pilot_shard": pilot_shard_for_hand_index(index) if pilot else None,
        "confirmation": index in CONFIRMATION_HAND_INDICES,
    }


def schedule_rows(
    indices: Iterable[int] = range(TRAIN_HAND_COUNT),
) -> list[dict[str, Any]]:
    checked = [_checked_train_index(index) for index in indices]
    if len(set(checked)) != len(checked):
        raise ValueError("Step 6c schedule indices must be unique")
    return [schedule_row(index) for index in checked]


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    """Frozen nearest-rank percentile: sorted[ceil(n*q)-1]."""

    if not 0.0 <= fraction <= 1.0 or not math.isfinite(fraction):
        raise ValueError("percentile fraction must be finite and in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile values must not be empty")
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def profile_counts(indices: Iterable[int]) -> dict[str, int]:
    values = [behavior_profile_for_train_index(index) for index in indices]
    return {profile: values.count(profile) for profile in M31_T3_BEHAVIOR_PROFILES}


def seed_set(indices: Iterable[int]) -> frozenset[int]:
    values = [value for index in indices for value in train_seed_values(index).values()]
    if len(values) != len(set(values)):
        raise ValueError("Step 6c seed schedule contains an overlap")
    return frozenset(values)


def validate_frozen_schedule() -> Mapping[str, Any]:
    """Recompute every schedule invariant without reading any game content."""

    full_counts = profile_counts(range(TRAIN_HAND_COUNT))
    pilot_counts = profile_counts(PILOT_HAND_INDICES)
    shard_counts = {
        str(shard): profile_counts(
            range(
                shard * PILOT_HANDS_PER_SHARD,
                (shard + 1) * PILOT_HANDS_PER_SHARD,
            )
        )
        for shard in range(PILOT_SHARD_COUNT)
    }
    confirmation_counts = profile_counts(CONFIRMATION_HAND_INDICES)
    recomputed_confirmation_roots = tuple(
        root for hand in CONFIRMATION_HAND_INDICES for root in (hand * 2, hand * 2 + 1)
    )
    pilot_seeds = seed_set(PILOT_HAND_INDICES)
    gates = {
        "full_train_exact_profile_quota": all(
            count == TRAIN_HANDS_PER_PROFILE for count in full_counts.values()
        ),
        "pilot_exact_profile_quota": all(
            count == PILOT_HANDS_PER_PROFILE for count in pilot_counts.values()
        ),
        "each_pilot_shard_exact_profile_quota": all(
            all(count == 5 for count in counts.values())
            for counts in shard_counts.values()
        ),
        "confirmation_one_hand_per_profile": all(
            count == 1 for count in confirmation_counts.values()
        ),
        "confirmation_root_grid_frozen": (
            recomputed_confirmation_roots == CONFIRMATION_ROOT_INDICES
        ),
        "confirmation_fraction_exact": (
            CONFIRMATION_ROOT_COUNT / PILOT_ROOT_COUNT == CONFIRMATION_FRACTION
        ),
        "pilot_seed_values_unique": len(pilot_seeds) == PILOT_HAND_COUNT * 6,
    }
    if not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise ValueError(f"Step 6c frozen schedule failed: {failed}")
    return {
        "full_profile_counts": full_counts,
        "pilot_profile_counts": pilot_counts,
        "pilot_shard_profile_counts": shard_counts,
        "confirmation_profile_counts": confirmation_counts,
        "pilot_seed_count": len(pilot_seeds),
        "pilot_schedule_sha256": canonical_sha256(schedule_rows(PILOT_HAND_INDICES)),
        "gates": gates,
    }


__all__ = [
    "ACCEPTED_FEATURE_ENCODER_SHA256",
    "ACCEPTED_NATIVE_LIBRARY_SHA256",
    "BEHAVIOR_BLOCK_HANDS",
    "BEHAVIOR_SEED_BASE",
    "CANDIDATE_SEED_BASE",
    "CHILD_SEED_BASE",
    "CONFIRMATION_BUDGET",
    "CONFIRMATION_FRACTION",
    "CONFIRMATION_HAND_COUNT",
    "CONFIRMATION_HAND_INDICES",
    "CONFIRMATION_REGRET_MAX",
    "CONFIRMATION_REGRET_MEAN_MAX",
    "CONFIRMATION_REGRET_P95_MAX",
    "CONFIRMATION_REGRET_P99_MAX",
    "CONFIRMATION_ROOT_COUNT",
    "CONFIRMATION_ROOT_INDICES",
    "CONFIRMATION_SEED_BASE",
    "EVALUATION_SEED_BASE",
    "EXPECTED_STEP6C_CONTRACT_BYTE_SHA256",
    "EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256",
    "HAND_SEED_BASE",
    "MAX_FIRST_P95_SECONDS",
    "MAX_PEAK_RSS_BYTES",
    "MAX_SECOND_P95_SECONDS",
    "PERCENTILE_METHOD",
    "PILOT_HAND_COUNT",
    "PILOT_HAND_INDICES",
    "PILOT_HANDS_PER_PROFILE",
    "PILOT_HANDS_PER_SHARD",
    "PILOT_ROOT_COUNT",
    "PILOT_ROOT_INDICES",
    "PILOT_ROOTS_PER_PROFILE",
    "PILOT_ROOTS_PER_SHARD",
    "PILOT_SHARD_COUNT",
    "POLICY_REGISTRY_SHA256",
    "PRODUCTION_BUDGET",
    "PRODUCTION_LABEL_BUDGET",
    "SEED_STRIDE",
    "STEP5_CONTRACT_BYTE_SHA256",
    "STEP5_CONTRACT_CANONICAL_SHA256",
    "STEP5_VALIDATION_SHA256",
    "STEP6B_STATUS_SHA256",
    "STEP6B_VALIDATION_SHA256",
    "STEP6C_BEHAVIOR_SCHEDULE_SCHEMA",
    "STEP6C_CONTRACT_SCHEMA",
    "STEP6C_RUN_ID",
    "STEP6C_SCHEDULE_ROW_SCHEMA",
    "STEP6C_SPLIT",
    "STEP6C_VALIDATION_SCHEMA",
    "Step6CSearchBudget",
    "TRAIN_BEHAVIOR_SEED_BASE",
    "TRAIN_CANDIDATE_SEED_BASE",
    "TRAIN_CHILD_SEED_BASE",
    "TRAIN_CONFIRMATION_SEED_BASE",
    "TRAIN_EVALUATION_SEED_BASE",
    "TRAIN_HAND_COUNT",
    "TRAIN_HAND_SEED_BASE",
    "TRAIN_HANDS_PER_PROFILE",
    "TRAIN_ROOT_COUNT",
    "behavior_profile_for_train_index",
    "canonical_bytes",
    "canonical_sha256",
    "nearest_rank_percentile",
    "pilot_shard_for_hand_index",
    "profile_counts",
    "schedule_row",
    "schedule_rows",
    "seed_set",
    "train_seed_values",
    "validate_frozen_schedule",
]
