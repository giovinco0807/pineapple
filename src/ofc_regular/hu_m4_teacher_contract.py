"""Frozen contracts for the M4 second-seat T1 search teacher.

This module deliberately contains no model or replay-world adapter.  It fixes
the live deal order and the RNG/cache domains which an implementation must use
when the M3 native search is extended through T2 and T1.  In particular, a
child decision is identified only by its policy version and
``ActorObservation`` fingerprint; an outer determinization, action index, or
future index must never participate in that identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .counter_rng import CounterRngKey
from .hu_infoset import ActorObservation


HU_M4_T1_SECOND_TEACHER_SCHEMA = "hu_m4_t1_second_teacher_v1"
HU_M4_CHILD_POLICY_SCHEMA = "hu_m4_infoset_child_policy_v1"
T1_SECOND_FUTURE_CARD_COUNT = 18

RootPhase = Literal["candidate_selection", "locked_evaluation"]


@dataclass(frozen=True)
class LiveDecision:
    """One post-root decision in actual HU Pineapple order."""

    seat: Literal["first", "second"]
    street: Literal["T2", "T3", "T4"]
    draw_offset: int

    @property
    def draw_slice(self) -> slice:
        return slice(self.draw_offset, self.draw_offset + 3)


T1_SECOND_LIVE_SCHEDULE: tuple[LiveDecision, ...] = (
    LiveDecision("first", "T2", 0),
    LiveDecision("second", "T2", 3),
    LiveDecision("first", "T3", 6),
    LiveDecision("second", "T3", 9),
    LiveDecision("first", "T4", 12),
    LiveDecision("second", "T4", 15),
)


@dataclass(frozen=True)
class M4T1SecondSearchConfig:
    """Root sampling and fixed child-policy identity for an M4 label."""

    candidate_samples: int
    evaluation_samples: int
    candidate_seed: int
    evaluation_seed: int
    run_id: str
    child_policy_id: str
    child_policy_seed: int

    def __post_init__(self) -> None:
        if self.candidate_samples <= 0 or self.evaluation_samples <= 0:
            raise ValueError("candidate/evaluation sample counts must be positive")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.child_policy_id:
            raise ValueError("child_policy_id must not be empty")

    def root_run_id(self, phase: RootPhase) -> str:
        if phase not in {"candidate_selection", "locked_evaluation"}:
            raise ValueError(f"unsupported M4 root phase: {phase!r}")
        return f"{self.run_id}:{phase}"


def require_t1_second_root(observation: ActorObservation) -> None:
    """Fail closed unless *observation* is exactly a second-seat T1 root."""

    if (
        observation.street != "T1"
        or observation.seat != "second"
        or observation.to_act_order != "second"
    ):
        raise ValueError("M4 T1-second teacher requires a T1/second ActorObservation")
    # ActorObservation itself enforces the 5/7/3/0 public geometry and card
    # uniqueness.  Calling fingerprint also makes this contract explicit to
    # callers which use duck-typed values in test harnesses.
    observation.fingerprint()


def child_policy_cache_key(
    policy_id: str, observation: ActorObservation
) -> tuple[str, str, str]:
    """Return the only legal semantic key for a downstream policy action."""

    if not policy_id:
        raise ValueError("policy_id must not be empty")
    if observation.street not in {"T2", "T3", "T4"}:
        raise ValueError("M4 child policy supports only T2-T4 observations")
    return (HU_M4_CHILD_POLICY_SCHEMA, policy_id, observation.fingerprint())


def child_policy_decision_seed(
    *, base_seed: int, policy_id: str, observation: ActorObservation
) -> int:
    """Derive action randomness from the child information set only.

    ``sample_index`` is intentionally fixed at zero.  The function accepts no
    outer sample/future/action coordinate, preventing two determinizations that
    reach the same information set from selecting different child actions.
    """

    child_policy_cache_key(policy_id, observation)
    actor = 0 if observation.seat == "first" else 1
    return CounterRngKey(
        base_seed=base_seed,
        run_id=f"m4-child-policy:{policy_id}",
        phase="infoset_child_policy",
        sample_index=0,
        actor=actor,
        street=observation.street,
        stream="locked_child_action",
        counter=0,
        root_fingerprint=observation.fingerprint(),
    ).seed()


def split_t1_second_future(cards: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Split a particle tail according to :data:`T1_SECOND_LIVE_SCHEDULE`."""

    future = tuple(cards)
    if len(future) < T1_SECOND_FUTURE_CARD_COUNT:
        raise ValueError(
            "T1-second rollout needs at least "
            f"{T1_SECOND_FUTURE_CARD_COUNT} future cards"
        )
    return tuple(tuple(future[step.draw_slice]) for step in T1_SECOND_LIVE_SCHEDULE)


def require_disjoint_root_rng_keys(
    candidate_keys: Iterable[str], evaluation_keys: Iterable[str]
) -> None:
    """Reject accidental candidate/evaluation sample reuse."""

    candidate = set(candidate_keys)
    evaluation = set(evaluation_keys)
    if not candidate or not evaluation:
        raise ValueError("candidate/evaluation RNG key sets must be non-empty")
    overlap = candidate & evaluation
    if overlap:
        raise ValueError(
            "candidate-selection and locked-evaluation RNG keys overlap: "
            f"{len(overlap)}"
        )


__all__ = [
    "HU_M4_CHILD_POLICY_SCHEMA",
    "HU_M4_T1_SECOND_TEACHER_SCHEMA",
    "LiveDecision",
    "M4T1SecondSearchConfig",
    "T1_SECOND_FUTURE_CARD_COUNT",
    "T1_SECOND_LIVE_SCHEDULE",
    "child_policy_cache_key",
    "child_policy_decision_seed",
    "require_disjoint_root_rng_keys",
    "require_t1_second_root",
    "split_t1_second_future",
]
