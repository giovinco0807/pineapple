"""Frozen scientific constants for M3.1 T3 Step 6d attempt 1.

Step 6d is a new experiment family.  It does not reopen or supersede the
completed Step 6c No-Go.  This module has no search, training, profile,
``current``, cloud, or artifact-generation side effects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


STEP6D_CONTRACT_SCHEMA = "hu_joint_policy_m31_t3_step6d_contract_v1"
STEP6D_VALIDATION_SCHEMA = "hu_joint_policy_m31_t3_step6d_contract_validation_v1"
STEP6D_RUN_ID = "hu-m31-step6d-performance-repair-attempt01-v1"
STEP6D_SCHEDULE_SCHEMA = "hu_m31_t3_step6d_disjoint_seed_schedule_v1"

SEED_STRIDE = 1_000_003
HISTORICAL_CONFIG_SEED_MAX = 470_607_073_398
PLANNED_SEED_COUNT = 67_500
PLANNED_SEED_MIN = 480_108_071_901
PLANNED_SEED_MAX = 680_607_073_398
PLANNED_SEED_SET_SHA256 = (
    "34e94c132a77de70409e684ddfa64f2d77c6cba38ab90a5868f0ace6c46ef392"
)

# Every consumer rejects a local edit even when the edited payload is otherwise
# semantically plausible.
EXPECTED_STEP6D_CONTRACT_BYTE_SHA256 = (
    "1924295b18070432cf3126159311102d9285dba37c498a3ea7666b0d5b777775"
)
EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256 = (
    "116a44cbd53ba6f3329b43ac8268a782f68582b7f832b062c2b0c158ba9627e3"
)

TEACHER_NAMESPACE_KEYS = (
    "hand",
    "behavior",
    "candidate",
    "evaluation",
    "child",
    "confirmation",
)
POPULATION_NAMESPACE_KEYS = (
    "hand",
    "actor_policy",
    "opponent_policy",
    "evaluation",
    "child",
    "confirmation",
)


@dataclass(frozen=True)
class Step6DSeedSchedule:
    name: str
    role: str
    index_count: int
    unit: str
    namespace_kind: str
    namespace_bases: tuple[int, int, int, int, int, int]
    training_eligible: bool
    locked_before_content_read: bool

    def __post_init__(self) -> None:
        if not self.name or not self.role or self.index_count <= 0:
            raise ValueError("Step 6d seed schedule identity changed")
        if self.namespace_kind not in {"teacher", "population"}:
            raise ValueError("Step 6d seed namespace kind changed")
        if len(self.namespace_bases) != 6:
            raise ValueError("Step 6d requires six seed namespaces")

    @property
    def namespace_keys(self) -> tuple[str, ...]:
        return (
            TEACHER_NAMESPACE_KEYS
            if self.namespace_kind == "teacher"
            else POPULATION_NAMESPACE_KEYS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "index_count": self.index_count,
            "unit": self.unit,
            "namespace_kind": self.namespace_kind,
            "namespace_bases": dict(
                zip(self.namespace_keys, self.namespace_bases, strict=True)
            ),
            "training_eligible": self.training_eligible,
            "locked_before_content_read": self.locked_before_content_read,
        }

    def values(self) -> tuple[int, ...]:
        return tuple(
            base + SEED_STRIDE * index
            for base in self.namespace_bases
            for index in range(self.index_count)
        )


SEED_SCHEDULES = (
    Step6DSeedSchedule(
        "performance_development",
        "repeatable_performance_engineering_only",
        100,
        "paired_hand",
        "teacher",
        (
            480_108_071_901,
            481_108_071_901,
            482_108_071_901,
            483_108_071_901,
            484_108_071_901,
            485_108_071_901,
        ),
        False,
        False,
    ),
    Step6DSeedSchedule(
        "performance_lock",
        "one_shot_performance_qualification_only",
        100,
        "paired_hand",
        "teacher",
        (
            490_108_071_901,
            491_108_071_901,
            492_108_071_901,
            493_108_071_901,
            494_108_071_901,
            495_108_071_901,
        ),
        False,
        True,
    ),
    Step6DSeedSchedule(
        "quality_pilot",
        "one_shot_production_label_quality_authorization_only",
        50,
        "paired_hand",
        "teacher",
        (
            500_108_071_901,
            501_108_071_901,
            502_108_071_901,
            503_108_071_901,
            504_108_071_901,
            505_108_071_901,
        ),
        False,
        True,
    ),
    Step6DSeedSchedule(
        "train",
        "teacher_training",
        6_000,
        "paired_hand",
        "teacher",
        (
            520_108_071_901,
            527_108_071_901,
            534_108_071_901,
            541_108_071_901,
            548_108_071_901,
            555_108_071_901,
        ),
        True,
        False,
    ),
    Step6DSeedSchedule(
        "safety_fit",
        "teacher_safety_estimator_fit",
        1_000,
        "paired_hand",
        "teacher",
        (
            570_108_071_901,
            572_108_071_901,
            574_108_071_901,
            576_108_071_901,
            578_108_071_901,
            580_108_071_901,
        ),
        True,
        False,
    ),
    Step6DSeedSchedule(
        "threshold_lock",
        "teacher_threshold_selection_only",
        1_000,
        "paired_hand",
        "teacher",
        (
            590_108_071_901,
            592_108_071_901,
            594_108_071_901,
            596_108_071_901,
            598_108_071_901,
            600_108_071_901,
        ),
        False,
        True,
    ),
    Step6DSeedSchedule(
        "diagnostic_teacher_holdout",
        "teacher_diagnostic_only_after_model_and_threshold_freeze",
        1_000,
        "paired_hand",
        "teacher",
        (
            610_108_071_901,
            612_108_071_901,
            614_108_071_901,
            616_108_071_901,
            618_108_071_901,
            620_108_071_901,
        ),
        False,
        True,
    ),
    Step6DSeedSchedule(
        "development_population",
        "realized_match_development_only",
        250,
        "paired_seed_per_opponent",
        "population",
        (
            630_108_071_901,
            631_108_071_901,
            632_108_071_901,
            633_108_071_901,
            634_108_071_901,
            635_108_071_901,
        ),
        False,
        False,
    ),
    Step6DSeedSchedule(
        "locked_population",
        "realized_match_promotion_holdout",
        1_000,
        "paired_seed_per_opponent",
        "population",
        (
            640_108_071_901,
            642_108_071_901,
            644_108_071_901,
            646_108_071_901,
            648_108_071_901,
            650_108_071_901,
        ),
        False,
        True,
    ),
    Step6DSeedSchedule(
        "abr_development",
        "response_policy_training_and_development_only",
        250,
        "paired_seed_per_response",
        "population",
        (
            660_108_071_901,
            661_108_071_901,
            662_108_071_901,
            663_108_071_901,
            664_108_071_901,
            665_108_071_901,
        ),
        False,
        False,
    ),
    Step6DSeedSchedule(
        "locked_abr",
        "approximate_best_response_promotion_holdout",
        500,
        "paired_seed_per_response",
        "population",
        (
            670_108_071_901,
            672_108_071_901,
            674_108_071_901,
            676_108_071_901,
            678_108_071_901,
            680_108_071_901,
        ),
        False,
        True,
    ),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seed_schedule_payload() -> list[dict[str, Any]]:
    return [schedule.to_dict() for schedule in SEED_SCHEDULES]


def planned_seed_values() -> tuple[int, ...]:
    return tuple(value for schedule in SEED_SCHEDULES for value in schedule.values())


def validate_seed_schedule() -> Mapping[str, Any]:
    values = planned_seed_values()
    unique = set(values)
    digest = hashlib.sha256(
        json.dumps(sorted(unique), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if (
        len(values) != PLANNED_SEED_COUNT
        or len(unique) != PLANNED_SEED_COUNT
        or min(unique) != PLANNED_SEED_MIN
        or max(unique) != PLANNED_SEED_MAX
        or min(unique) <= HISTORICAL_CONFIG_SEED_MAX
        or digest != PLANNED_SEED_SET_SHA256
    ):
        raise ValueError("Step 6d seed schedule changed")
    return {
        "planned_seed_count": len(unique),
        "planned_seed_min": min(unique),
        "planned_seed_max": max(unique),
        "planned_seed_set_sha256": digest,
        "schedule_seed_counts": {
            schedule.name: len(schedule.values()) for schedule in SEED_SCHEDULES
        },
    }


def schedule_by_name(name: str) -> Step6DSeedSchedule:
    matches = [schedule for schedule in SEED_SCHEDULES if schedule.name == name]
    if len(matches) != 1:
        raise KeyError(f"unknown Step 6d schedule: {name}")
    return matches[0]


def require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise ValueError(
            f"{label} fields changed: missing={sorted(wanted-observed)}, "
            f"unknown={sorted(observed-wanted)}"
        )


__all__ = [
    "EXPECTED_STEP6D_CONTRACT_BYTE_SHA256",
    "EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256",
    "HISTORICAL_CONFIG_SEED_MAX",
    "PLANNED_SEED_COUNT",
    "PLANNED_SEED_MAX",
    "PLANNED_SEED_MIN",
    "PLANNED_SEED_SET_SHA256",
    "POPULATION_NAMESPACE_KEYS",
    "SEED_SCHEDULES",
    "SEED_STRIDE",
    "STEP6D_CONTRACT_SCHEMA",
    "STEP6D_RUN_ID",
    "STEP6D_SCHEDULE_SCHEMA",
    "STEP6D_VALIDATION_SCHEMA",
    "Step6DSeedSchedule",
    "TEACHER_NAMESPACE_KEYS",
    "canonical_bytes",
    "canonical_sha256",
    "planned_seed_values",
    "require_exact_keys",
    "schedule_by_name",
    "seed_schedule_payload",
    "validate_seed_schedule",
]
