"""Sequential-belief T3 teacher for heads-up regular OFC Pineapple.

The historical module with this filename dealt the hero directly into T4 and
scored an incomplete opponent as a standalone board.  This M2 implementation
instead follows the public game order for both seats and constructs a fresh
``ActorObservation`` at every child decision.  Root chance worlds are common
across all candidate actions, while candidate selection and the reported
locked-action evaluation use disjoint hidden-card particle batches.

Despite the compatibility filename, ordinary T3 values are Monte Carlo values
under the explicitly recorded local-belief continuation policy.  Only T4
second-seat placement and, when configured with zero samples, the T4
first-seat exchangeable marginal are exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .cards import ALL_CARDS, validate_cards
from .evaluator import BoardScore, score_board
from .hu_belief import (
    HIDDEN_CARD_BELIEF_SCHEMA,
    HiddenCardParticle,
    HiddenCardParticleBatch,
    sample_hidden_card_particles,
)
from .hu_infoset import ActorObservation, InformationSetError
from .hu_late_street_teacher import T4SearchConfig, evaluate_t4_sequential_actions
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score


T3_SEQUENTIAL_TEACHER_SCHEMA = "hu_turn3_sequential_belief_v2"
T3_EXPLICIT_SUPPORT_SCHEMA = "hu_turn3_exact_explicit_support_v1"
T3_CONTINUATION_POLICY_ID = "local_infoset_response_t3_second_t4_v1"
T3_SOLVER_ID = "python_crn_sequential_t3_reference_v1"
_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
HAND_CATEGORY_NAMES = {
    0: "high",
    1: "pair",
    2: "two_pair",
    3: "trips",
    4: "straight",
    5: "flush",
    6: "full_house",
    7: "quads",
    8: "straight_flush",
}


class Turn3TeacherError(ValueError):
    """Raised on an unsafe or inconsistent T3 teacher request."""


@dataclass(frozen=True)
class JointExactConfig:
    """Compatibility-named configuration for the M2 sequential teacher.

    ``future_samples`` is a deprecated construction alias.  When supplied it
    sets both root sample counts, but the two batches still use disjoint RNG
    domains.  New callers should use the explicit candidate/evaluation fields.
    """

    candidate_samples: int = 4
    evaluation_samples: int = 8
    downstream_t3_samples: int = 2
    downstream_t4_samples: int = 8
    seed: int = 42
    candidate_seed: int | None = None
    evaluation_seed: int | None = None
    run_id: str = "hu-m2-t3"
    opponent_t3_policy: str = "local_belief_response"
    future_samples: int | None = None
    # Opt in to the learned first-seat T4 leaf inside the native engine. Leaving
    # both unset keeps every T4 child exact and keeps the emitted request and
    # result byte-identical to a run from before this existed. A path without a
    # digest is refused by the engine rather than trusted, because weights that
    # quietly changed still produce plausible numbers.
    learned_t4_model_path: str | None = None
    learned_t4_model_sha256: str | None = None
    # Opt in to the learned T3 second-seat evaluator for the nested response the
    # first seat's rollouts play against. Same contract as the T4 fields: unset
    # keeps the sampled nested search and byte-identical output; a path without
    # a digest is refused by the engine.
    learned_t3_second_model_path: str | None = None
    learned_t3_second_model_sha256: str | None = None
    # Required by the engine's T2 evaluator, which plays the opponent's T3
    # first-seat reply through this model and refuses to run without it.
    learned_t3_first_model_path: str | None = None
    learned_t3_first_model_sha256: str | None = None
    # Consumed by the engine's `decide` request for T2 second-seat play.
    learned_t2_second_model_path: str | None = None
    learned_t2_second_model_sha256: str | None = None
    # Consumed by the engine's `decide` request for T2 first-seat play. Acting
    # first the opponent has not answered T2 yet, so the seat has its own model
    # rather than sharing the second seat's.
    learned_t2_first_model_path: str | None = None
    learned_t2_first_model_sha256: str | None = None
    # Consumed by the engine's `decide` request for T1 second-seat play, and
    # required by its T1 first-seat evaluator, whose every rollout opens with
    # the opponent's T1 second-seat reply. Acting second at T1 the opponent's
    # board carries seven cards where the first seat sees five, so the seat has
    # its own model rather than sharing the first seat's.
    learned_t1_second_model_path: str | None = None
    learned_t1_second_model_sha256: str | None = None
    # Consumed by the engine's `decide` request for T1 first-seat play, and
    # required by its T0 evaluator, whose rollouts play a T1 first-seat reply on
    # both seats. Acting first at T1 both boards carry five cards where the
    # second seat sees seven, so the seat has its own model.
    learned_t1_first_model_path: str | None = None
    learned_t1_first_model_sha256: str | None = None
    # Consumed by the engine's `decide` request for T0 second-seat play, and
    # required by its T0 first-seat evaluator, whose every rollout opens with
    # the opponent's T0 second-seat reply.
    learned_t0_second_model_path: str | None = None
    learned_t0_second_model_sha256: str | None = None
    # Consumed by the engine's `decide` request for T0 FIRST-seat play, and by
    # nothing else. Every other learned evaluator here is named twice over --
    # once by the decision that plays it and once by some rollout that has it
    # still ahead -- because every other decision sits below at least one
    # street's root. The opening street acting first sits below nothing, so this
    # one is a root policy rather than a continuation.
    #
    # Its image is not interchangeable with the seven above. Acting first the
    # opponent's board is empty, which the engine's free-slot outlook refuses,
    # so the opponent-outlook and head-to-head blocks are zero and the weights
    # were fitted that way.
    learned_t0_first_model_path: str | None = None
    learned_t0_first_model_sha256: str | None = None
    # The coarse distilled T2 pair, on the same terms as the T1 pair below: set
    # ALONGSIDE learned_t2_*, reported by the engine as "learned_fast",
    # byte-identical request when unset, path without a digest refused. Reached
    # by more shapes than the T0 second-seat pin is -- every rollout that starts
    # at T2 first or earlier still has both T2 replies ahead of it.
    fast_t2_second_model_path: str | None = None
    fast_t2_second_model_sha256: str | None = None
    # Separate from the second seat's for the same reason the full-precision
    # pair is two models: acting first at T2 the opponent's board carries seven
    # cards where the second seat sees nine.
    fast_t2_first_model_path: str | None = None
    fast_t2_first_model_sha256: str | None = None
    # The coarse distilled T1 pair. Set ALONGSIDE the learned_t1_* pair rather
    # than instead of it: the engine loads both and reaches its T1 replies
    # through these, which is why it reports those two evaluators as
    # "learned_fast" rather than "learned". Leaving them unset keeps the emitted
    # request byte-identical to one built before they existed, and a path
    # without a digest is refused by the engine exactly as the learned pair's is.
    fast_t1_second_model_path: str | None = None
    fast_t1_second_model_sha256: str | None = None
    # Separate from the second seat's for the same reason the full-precision
    # pair is two models: acting first at T1 both boards carry five cards where
    # the second seat sees seven.
    fast_t1_first_model_path: str | None = None
    fast_t1_first_model_sha256: str | None = None
    # The coarse distilled T0 second-seat reply, on the same terms as the T1
    # pair above: set ALONGSIDE learned_t0_second_*, reported by the engine as
    # "learned_fast", byte-identical request when unset, path without a digest
    # refused. Only a T0 FIRST-seat evaluation reaches it -- acting second, the
    # opening it would answer is already on the board -- and it is the reply
    # worth coarsening most, being 232 candidate boards where a turn reply is
    # twenty-seven.
    fast_t0_second_model_path: str | None = None
    fast_t0_second_model_sha256: str | None = None
    # Two-stage root schedule for the T0 evaluator, which faces roughly 232 root
    # actions where every other street faces tens. Both zero means single-stage,
    # which is the exact-comparison path the pruning validation uses; both
    # nonzero scores every action with `prefilter_samples` particles and then
    # rescores only the best `prefilter_keep` with the full evaluation set.
    prefilter_samples: int = 0
    prefilter_keep: int = 0
    # Widen that keep boundary wherever the coarse stage cannot separate the
    # actions across it. Zero, the default, leaves the boundary at the fixed
    # rank it has always been drawn at and keeps the emitted request
    # byte-identical to one built before this existed. The number is a score
    # difference measured from the schedule's own stage-one spread; the engine
    # only honours whatever was measured, and refuses a margin with no
    # two-stage boundary to widen.
    prefilter_margin: float = 0.0
    # Keep only this many T1/T2/T3 candidates, chosen by the street's learned
    # evaluator, before any particle is spent. Zero -- the default -- disables
    # it and leaves the emitted request byte-identical to one from before it
    # existed, so label plans that do not set it are unaffected.
    #
    # Distinct from prefilter_samples/prefilter_keep above, which are T0's and
    # cut with a coarse sampled pass; this cut is deterministic. Sized from the
    # top-K survival measurement in docs/trainer_ranking_quality_20260808.md:
    # ten keeps 91-96% of the rollout's own best actions, five keeps 84-86%.
    learned_prefilter_keep: int = 0
    # Evaluate one position in every N single-stage as well and report the
    # comparison, so a fleet running on a distribution the pruning validation
    # never covered carries its own measurement of what pruning cost it. Zero,
    # the default, disables it and keeps the request byte-identical.
    audit_full_every: int = 0
    # Racing (sequential halving) inside stage two. Empty, the default, spends
    # the full evaluation set on every survivor, which is the uniform schedule
    # the pruning validation measured and keeps the emitted request
    # byte-identical to one built before this existed.
    #
    # A non-empty schedule is a list of CUMULATIVE particle checkpoints into the
    # evaluation batch -- [32, 64, 128, 256] means "score every survivor on the
    # first 32, drop the ones the leader has decisively beaten, take the rest to
    # 64, and so on". Particles accumulate rather than being redrawn, so a
    # candidate that reaches the last checkpoint has consumed exactly the batch
    # a uniform run would have given it, and every comparison is paired: at any
    # checkpoint the live candidates have all consumed the SAME particles.
    #
    # The last checkpoint must equal `evaluation_samples`, because a schedule
    # that stopped short would be a cheaper evaluation dressed as the full one.
    race_schedule: tuple[int, ...] = ()
    # How decisive the leader has to be before a candidate is dropped, in units
    # of the paired difference's own standard error. Larger is more
    # conservative. Refused without a schedule to apply it to, on the same
    # grounds as a margin with no boundary to widen.
    race_lcb_z: float = 0.0
    # Legacy fields are retained as inert serialization compatibility only.
    actor: str = "hero"
    seat: str = "first"
    to_act_order: str = "first"
    use_final_turn_cache: bool = True

    def __post_init__(self) -> None:
        if self.future_samples is not None:
            if isinstance(self.future_samples, bool) or not isinstance(
                self.future_samples, int
            ):
                raise TypeError("future_samples must be an integer")
            if self.future_samples <= 0:
                raise ValueError(
                    "T3 full exact enumeration is not tractable; future_samples must be positive"
                )
            object.__setattr__(self, "candidate_samples", self.future_samples)
            object.__setattr__(self, "evaluation_samples", self.future_samples)
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.downstream_t4_samples, bool) or not isinstance(
            self.downstream_t4_samples, int
        ) or self.downstream_t4_samples < 0:
            raise ValueError("downstream_t4_samples must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in ("candidate_seed", "evaluation_seed"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        for name in ("prefilter_samples", "prefilter_keep"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        # Half a schedule is not a schedule: one field set without the other
        # would silently run single-stage and look like it had pruned.
        if bool(self.prefilter_samples) != bool(self.prefilter_keep):
            raise ValueError(
                "prefilter_samples and prefilter_keep must be set together; "
                "leave both zero for the single-stage path"
            )
        # A margin is a score difference, so an int is as acceptable as a float
        # and a bool is not either. Refused here as well as in the engine so a
        # plan is rejected before the shard opens rather than on its first
        # position.
        if isinstance(self.prefilter_margin, bool) or not isinstance(
            self.prefilter_margin, (int, float)
        ):
            raise TypeError("prefilter_margin must be a real number")
        if not math.isfinite(self.prefilter_margin) or self.prefilter_margin < 0:
            raise ValueError(
                "prefilter_margin must be a finite non-negative score "
                "difference; leave it at zero for the fixed-rank keep boundary"
            )
        # The same reasoning as half a schedule: a margin with no boundary to
        # widen reads as pruning safety and does nothing.
        if self.prefilter_margin and not self.prefilter_samples:
            raise ValueError(
                "prefilter_margin widens the two-stage prefilter's keep "
                "boundary and has no boundary to widen on the single-stage "
                "path; set prefilter_samples and prefilter_keep, or leave the "
                "margin at zero"
            )
        if isinstance(self.audit_full_every, bool) or not isinstance(
            self.audit_full_every, int
        ) or self.audit_full_every < 0:
            raise ValueError("audit_full_every must be a non-negative integer")
        # Normalised to a tuple so the config stays hashable and a caller's list
        # cannot be mutated out from under a running shard.
        if isinstance(self.race_schedule, (str, bytes)) or not isinstance(
            self.race_schedule, (list, tuple)
        ):
            raise TypeError("race_schedule must be a sequence of particle counts")
        object.__setattr__(self, "race_schedule", tuple(self.race_schedule))
        for checkpoint in self.race_schedule:
            if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
                raise TypeError("race_schedule checkpoints must be integers")
        # Cumulative and strictly increasing: a checkpoint that did not advance
        # would spend no particles and eliminate on the same evidence twice, and
        # one that went backwards has no meaning at all.
        if any(
            later <= earlier
            for earlier, later in zip(self.race_schedule, self.race_schedule[1:])
        ):
            raise ValueError(
                "race_schedule must be strictly increasing cumulative particle "
                "counts"
            )
        if self.race_schedule and self.race_schedule[0] <= 0:
            raise ValueError("race_schedule checkpoints must be positive")
        # A schedule that stopped short of the evaluation set would be a cheaper
        # evaluation wearing the full one's name, and the action it selected
        # would not have been measured at the resolution the label claims.
        if self.race_schedule and self.race_schedule[-1] != self.evaluation_samples:
            raise ValueError(
                "the last race_schedule checkpoint must equal "
                f"evaluation_samples ({self.evaluation_samples}); the schedule "
                f"ends at {self.race_schedule[-1]}"
            )
        if isinstance(self.race_lcb_z, bool) or not isinstance(
            self.race_lcb_z, (int, float)
        ):
            raise TypeError("race_lcb_z must be a real number")
        if not math.isfinite(self.race_lcb_z) or self.race_lcb_z < 0:
            raise ValueError(
                "race_lcb_z must be a finite non-negative number of standard "
                "errors"
            )
        # The same reasoning as a margin with no boundary: a z with no schedule
        # to apply it to reads as a tuned elimination rule and does nothing.
        if self.race_lcb_z and not self.race_schedule:
            raise ValueError(
                "race_lcb_z is the elimination threshold for the stage-two "
                "race and has no race to threshold; set race_schedule, or "
                "leave the z at zero"
            )
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.opponent_t3_policy != "local_belief_response":
            raise ValueError("unsupported opponent_t3_policy")


@dataclass
class _SearchContext:
    config: JointExactConfig
    fl_ev: dict[int, float]
    t4_action_cache: dict[str, Action]
    t3_second_action_cache: dict[str, Action]
    child_observation_fingerprints: set[str]
    explicit_t4_selector: Callable[[ActorObservation], Action] | None = None
    explicit_t3_second_selector: Callable[[ActorObservation], Action] | None = None


@dataclass(frozen=True)
class T3ExplicitWorld:
    """One weighted world in a declared finite T3 chance support.

    This type exists for toy games, regression oracles, and genuinely reduced
    supports.  It does not claim to enumerate the full 52-card T3 game.
    """

    opponent_private_discards: tuple[str, ...]
    future_cards: tuple[str, ...]
    weight: float
    world_id: str

    def __post_init__(self) -> None:
        opponent_discards = tuple(
            sorted(self.opponent_private_discards, key=_CARD_INDEX.__getitem__)
        )
        future = tuple(self.future_cards)
        canonical_future: list[str] = []
        for start in range(0, len(future), 3):
            canonical_future.extend(
                sorted(future[start : start + 3], key=_CARD_INDEX.__getitem__)
            )
        object.__setattr__(self, "opponent_private_discards", opponent_discards)
        object.__setattr__(self, "future_cards", tuple(canonical_future))

    def draw(self, count: int, *, offset: int = 0) -> tuple[str, ...]:
        end = offset + count
        if count < 0 or offset < 0 or end > len(self.future_cards):
            raise ValueError("explicit T3 world draw is out of bounds")
        return self.future_cards[offset:end]


def evaluate_t3_exact_explicit_support_actions(
    *,
    observation: ActorObservation,
    worlds: Sequence[T3ExplicitWorld],
    t4_selector: Callable[[ActorObservation], Action],
    t3_second_selector: Callable[[ActorObservation], Action] | None = None,
    continuation_policy_id: str,
    fl_ev: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Exactly enumerate a caller-declared finite T3 support under fixed policy.

    Every downstream selector receives only its actor observation.  The value
    is exact over ``worlds`` and the supplied deterministic continuation, not
    over the full regular-OFC chance tree unless the caller proves that its
    support is complete.
    """

    observation = _require_t3_observation(observation)
    if not worlds:
        raise Turn3TeacherError("explicit T3 support must not be empty")
    if not continuation_policy_id:
        raise Turn3TeacherError("continuation_policy_id must not be empty")
    if observation.to_act_order == "first" and t3_second_selector is None:
        raise Turn3TeacherError(
            "T3-first explicit support requires a fixed T3-second selector"
        )
    _validate_explicit_worlds(observation, worlds)
    fl_ev = fl_ev or dict(observation.scoring.fl_ev) or DEFAULT_FL_EV
    context = _SearchContext(
        config=JointExactConfig(
            candidate_samples=1,
            evaluation_samples=1,
            downstream_t3_samples=1,
            downstream_t4_samples=1,
            run_id="explicit-support-no-rng",
        ),
        fl_ev=fl_ev,
        t4_action_cache={},
        t3_second_action_cache={},
        child_observation_fingerprints=set(),
        explicit_t4_selector=t4_selector,
        explicit_t3_second_selector=t3_second_selector,
    )
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    values: list[float] = []
    rollout_counts: list[int] = []
    for action in actions:
        value = 0.0
        count = 0
        for world in worlds:
            terminal = (
                _rollout_t3_first(observation, action, world, context)
                if observation.to_act_order == "first"
                else _rollout_t3_second(observation, action, world, context)
            )
            value += world.weight * float(terminal["hero_score"])
            count += 1
        values.append(value)
        rollout_counts.append(count)
    ranked = canonical_descending_indices(values, actions)
    rows: list[dict[str, Any]] = []
    for sorted_index, original_index in enumerate(ranked):
        rows.append(
            {
                **_action_payload(actions[original_index], original_index),
                "sorted_index": sorted_index,
                "score": values[original_index],
                "joint_ev": values[original_index],
                "future_count": rollout_counts[original_index],
                "regret_vs_best": values[ranked[0]] - values[original_index],
            }
        )
    support_payload = [
        {
            "world_id": world.world_id,
            "weight": world.weight,
            "opponent_private_discards": list(world.opponent_private_discards),
            "future_cards": list(world.future_cards),
        }
        for world in worlds
    ]
    return {
        "schema": T3_EXPLICIT_SUPPORT_SCHEMA,
        "mode": "exact_over_declared_finite_support",
        "full_52_card_tree_claimed": False,
        "observation_fingerprint": observation.fingerprint(),
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "continuation_policy_id": continuation_policy_id,
        "continuation_policy_fingerprint": _digest(
            {"id": continuation_policy_id, "selectors": "caller_supplied_infoset_only"}
        ),
        "support_count": len(worlds),
        "support_weight_sum": sum(world.weight for world in worlds),
        "support_digest": _digest(support_payload),
        "legal_action_count": len(actions),
        "selected_action_original_index": ranked[0],
        "selected_action_key": action_key(actions[ranked[0]]).to_token(),
        "best_score": values[ranked[0]],
        "actions": rows,
        "teacher_notes": {
            "exactness": "exact_only_over_declared_support_and_fixed_continuation",
            "strategy_fusion_guard": "selectors_receive_only_ActorObservation",
        },
    }


def evaluate_t3_joint_exact_actions(
    *,
    observation: ActorObservation,
    config: JointExactConfig | None = None,
    candidate_belief_batch: HiddenCardParticleBatch | None = None,
    evaluation_belief_batch: HiddenCardParticleBatch | None = None,
    fl_ev: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Evaluate every legal T3 action through the actual HU action sequence."""

    observation = _require_t3_observation(observation)
    config = config or JointExactConfig(
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    fl_ev = fl_ev or dict(observation.scoring.fl_ev) or DEFAULT_FL_EV
    candidate_batch = candidate_belief_batch or sample_hidden_card_particles(
        observation,
        base_seed=(config.seed if config.candidate_seed is None else config.candidate_seed),
        run_id=f"{config.run_id}:candidate_selection",
        sample_count=config.candidate_samples,
    )
    evaluation_batch = evaluation_belief_batch or sample_hidden_card_particles(
        observation,
        base_seed=(
            config.seed if config.evaluation_seed is None else config.evaluation_seed
        ),
        run_id=f"{config.run_id}:locked_evaluation",
        sample_count=config.evaluation_samples,
    )
    candidate_batch.validate_against(observation)
    evaluation_batch.validate_against(observation)
    _assert_disjoint_batches(candidate_batch, evaluation_batch)

    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise Turn3TeacherError("T3 observation has no legal actions")
    context = _SearchContext(
        config=config,
        fl_ev=fl_ev,
        t4_action_cache={},
        t3_second_action_cache={},
        child_observation_fingerprints=set(),
    )
    candidate_rows = _score_t3_actions(
        observation, actions, candidate_batch.particles, context
    )
    evaluation_rows = _score_t3_actions(
        observation, actions, evaluation_batch.particles, context
    )
    candidate_values = [float(row["score"]) for row in candidate_rows]
    evaluation_values = [float(row["score"]) for row in evaluation_rows]
    ranked = canonical_descending_indices(candidate_values, actions)
    selected_index = ranked[0]
    candidate_second = candidate_values[ranked[1]] if len(ranked) > 1 else candidate_values[selected_index]
    evaluation_sample_best = max(evaluation_values)

    rows: list[dict[str, Any]] = []
    for sorted_index, original_index in enumerate(ranked):
        row = _action_payload(actions[original_index], original_index)
        row.update(evaluation_rows[original_index])
        row.update(
            {
                "sorted_index": sorted_index,
                "selection_score": candidate_values[original_index],
                "selection_future_count": len(candidate_batch.particles),
                "evaluation_future_count": len(evaluation_batch.particles),
                "selected_by_candidate_plan": original_index == selected_index,
                "evaluation_regret_vs_sample_best": evaluation_sample_best
                - evaluation_values[original_index],
            }
        )
        rows.append(row)

    selected_eval = evaluation_values[selected_index]
    return {
        "schema": T3_SEQUENTIAL_TEACHER_SCHEMA,
        "solver_id": T3_SOLVER_ID,
        "phase": "hu_turn3_9card",
        "street": "T3",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "policy_observation": observation.to_dict(),
        "board": _board_payload(observation.hero_board),
        "opponent_board": _board_payload(observation.opponent_public_board),
        "dealt": list(observation.dealt_cards),
        "hero_private_discards": list(observation.hero_private_discards),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "hidden_card_belief_schema": HIDDEN_CARD_BELIEF_SCHEMA,
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "best_action_original_index": selected_index,
        "selected_action_original_index": selected_index,
        "selected_action_key": action_key(actions[selected_index]).to_token(),
        "selected_action_evaluation_score": selected_eval,
        "best_score": selected_eval,
        "score_gap": candidate_values[selected_index] - candidate_second,
        "selection_score_gap": candidate_values[selected_index] - candidate_second,
        "evaluation_sample_best_score": evaluation_sample_best,
        "evaluation_sample_regret_of_locked_selection": evaluation_sample_best
        - selected_eval,
        "future_count": len(evaluation_batch.particles),
        "candidate_belief": candidate_batch.to_dict(),
        "evaluation_belief": evaluation_batch.to_dict(),
        "sample_independence": "disjoint_particle_rng_keys",
        "continuation_policy": {
            "id": T3_CONTINUATION_POLICY_ID,
            "opponent_t3_mode": config.opponent_t3_policy,
            "downstream_t3_samples": config.downstream_t3_samples,
            "downstream_t4_samples": config.downstream_t4_samples,
            "continuation_seed": config.seed,
            "continuation_run_id": config.run_id,
            "use_t4_action_cache": config.use_final_turn_cache,
            "t4_first_mode": (
                "exact_uniform_marginal"
                if config.downstream_t4_samples == 0
                else "counter_mc_belief"
            ),
            "t4_second_mode": "terminal_exhaustive",
            "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
            "fingerprint": _continuation_policy_fingerprint(config),
        },
        "child_information_set_count": len(context.child_observation_fingerprints),
        "actions": rows,
        "fl_ev": {str(key): float(value) for key, value in fl_ev.items()},
        "teacher_notes": {
            "primary_metric": "locked_action_independent_evaluation_ev",
            "t3_value_scope": "Q_pi_under_recorded_local_belief_continuation",
            "exact_components": ["T4_second_legal_actions", "terminal_HU_scoring"],
            "mc_components": ["T3_root_worlds", "T3_child_local_belief"],
            "reported_value_warning": "teacher estimate_not_realized_match_EV",
        },
    }


def evaluate_t3_joint_batch(
    observations: Sequence[ActorObservation],
    *,
    config: JointExactConfig | None = None,
    fl_ev: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Reference batch API with scalar-identical semantics."""

    return [
        evaluate_t3_joint_exact_actions(
            observation=observation,
            config=config,
            fl_ev=fl_ev,
        )
        for observation in observations
    ]


def _score_t3_actions(
    observation: ActorObservation,
    actions: Sequence[Action],
    particles: Sequence[HiddenCardParticle],
    context: _SearchContext,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actions:
        scores: list[float] = []
        final_boards: list[Board] = []
        downstream_action_counts: Counter[str] = Counter()
        for particle in particles:
            if observation.to_act_order == "first":
                terminal = _rollout_t3_first(
                    observation, action, particle, context
                )
            else:
                terminal = _rollout_t3_second(
                    observation, action, particle, context
                )
            scores.append(float(terminal["hero_score"]))
            final_boards.append(terminal["hero_final_board"])
            downstream_action_counts.update(terminal["downstream_action_keys"])
        stats = _aggregate_terminal_stats(scores, final_boards)
        stats["downstream_action_counts"] = dict(downstream_action_counts)
        rows.append(stats)
    return rows


def _rollout_t3_second(
    observation: ActorObservation,
    root_action: Action,
    particle: HiddenCardParticle,
    context: _SearchContext,
) -> dict[str, Any]:
    """T3 second -> opponent T4 first -> hero T4 second."""

    after_root = observation.hero_board.place(root_action.placements)
    opponent_t4_deal = particle.draw(3, offset=0)
    hero_t4_deal = particle.draw(3, offset=3)
    opponent_t4_observation = ActorObservation(
        hero_board=observation.opponent_public_board,
        opponent_public_board=after_root,
        dealt_cards=opponent_t4_deal,
        hero_private_discards=particle.opponent_private_discards,
        seat="first",
        street="T4",
        to_act_order="first",
        scoring=observation.scoring,
    )
    opponent_t4_action = _locked_t4_action(opponent_t4_observation, context)
    opponent_final = observation.opponent_public_board.place(
        opponent_t4_action.placements
    )
    hero_t4_observation = ActorObservation(
        hero_board=after_root,
        opponent_public_board=opponent_final,
        dealt_cards=hero_t4_deal,
        hero_private_discards=(
            *observation.hero_private_discards,
            *root_action.discards,
        ),
        seat="second",
        street="T4",
        to_act_order="second",
        scoring=observation.scoring,
    )
    hero_t4_action = _locked_t4_action(hero_t4_observation, context)
    hero_final = after_root.place(hero_t4_action.placements)
    score, _ = terminal_score(hero_final, opponent_final, context.fl_ev)
    return {
        "hero_score": float(score),
        "hero_final_board": hero_final,
        "opponent_final_board": opponent_final,
        "downstream_action_keys": (
            action_key(opponent_t4_action).to_token(),
            action_key(hero_t4_action).to_token(),
        ),
    }


def _rollout_t3_first(
    observation: ActorObservation,
    root_action: Action,
    particle: HiddenCardParticle,
    context: _SearchContext,
) -> dict[str, Any]:
    """T3 first -> opponent T3 second -> hero T4 first -> opponent T4 second."""

    after_root = observation.hero_board.place(root_action.placements)
    opponent_t3_deal = particle.draw(3, offset=0)
    hero_t4_deal = particle.draw(3, offset=3)
    opponent_t4_deal = particle.draw(3, offset=6)
    opponent_t3_observation = ActorObservation(
        hero_board=observation.opponent_public_board,
        opponent_public_board=after_root,
        dealt_cards=opponent_t3_deal,
        hero_private_discards=particle.opponent_private_discards,
        seat="second",
        street="T3",
        to_act_order="second",
        scoring=observation.scoring,
    )
    opponent_t3_action = _locked_t3_second_action(
        opponent_t3_observation, context
    )
    opponent_after_t3 = observation.opponent_public_board.place(
        opponent_t3_action.placements
    )
    hero_t4_observation = ActorObservation(
        hero_board=after_root,
        opponent_public_board=opponent_after_t3,
        dealt_cards=hero_t4_deal,
        hero_private_discards=(
            *observation.hero_private_discards,
            *root_action.discards,
        ),
        seat="first",
        street="T4",
        to_act_order="first",
        scoring=observation.scoring,
    )
    hero_t4_action = _locked_t4_action(hero_t4_observation, context)
    hero_final = after_root.place(hero_t4_action.placements)
    opponent_t4_observation = ActorObservation(
        hero_board=opponent_after_t3,
        opponent_public_board=hero_final,
        dealt_cards=opponent_t4_deal,
        hero_private_discards=(
            *particle.opponent_private_discards,
            *opponent_t3_action.discards,
        ),
        seat="second",
        street="T4",
        to_act_order="second",
        scoring=observation.scoring,
    )
    opponent_t4_action = _locked_t4_action(opponent_t4_observation, context)
    opponent_final = opponent_after_t3.place(opponent_t4_action.placements)
    score, _ = terminal_score(hero_final, opponent_final, context.fl_ev)
    return {
        "hero_score": float(score),
        "hero_final_board": hero_final,
        "opponent_final_board": opponent_final,
        "downstream_action_keys": (
            action_key(opponent_t3_action).to_token(),
            action_key(hero_t4_action).to_token(),
            action_key(opponent_t4_action).to_token(),
        ),
    }


def _locked_t4_action(
    observation: ActorObservation,
    context: _SearchContext,
) -> Action:
    """Select from the child actor's observation, never the outer deck truth."""

    fingerprint = observation.fingerprint()
    context.child_observation_fingerprints.add(fingerprint)
    cached = (
        context.t4_action_cache.get(fingerprint)
        if context.config.use_final_turn_cache
        else None
    )
    if cached is not None:
        return cached
    if context.explicit_t4_selector is not None:
        selected = context.explicit_t4_selector(observation)
        _require_legal_selected_action(observation, selected)
        if context.config.use_final_turn_cache:
            context.t4_action_cache[fingerprint] = selected
        return selected
    result = evaluate_t4_sequential_actions(
        observation,
        config=T4SearchConfig(
            candidate_samples=context.config.downstream_t4_samples,
            evaluation_samples=context.config.downstream_t4_samples,
            seed=context.config.seed,
            run_id=f"{context.config.run_id}:child-t4",
        ),
        fl_ev=context.fl_ev,
    )
    selected_key = result["selected_action_key"]
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    selected = _action_by_token(actions, selected_key)
    if context.config.use_final_turn_cache:
        context.t4_action_cache[fingerprint] = selected
    return selected


def _locked_t3_second_action(
    observation: ActorObservation,
    context: _SearchContext,
) -> Action:
    """Local belief response at a T3-second information set.

    The nested particles are sampled from the opponent actor's observation,
    not from the outer determinization.  Thus two outer worlds that reach the
    same information set obtain the same action.
    """

    observation = _require_t3_observation(observation)
    if observation.to_act_order != "second":
        raise Turn3TeacherError("nested T3 response must be second to act")
    fingerprint = observation.fingerprint()
    context.child_observation_fingerprints.add(fingerprint)
    cached = context.t3_second_action_cache.get(fingerprint)
    if cached is not None:
        return cached
    if context.explicit_t3_second_selector is not None:
        selected = context.explicit_t3_second_selector(observation)
        _require_legal_selected_action(observation, selected)
        context.t3_second_action_cache[fingerprint] = selected
        return selected
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    batch = sample_hidden_card_particles(
        observation,
        base_seed=context.config.seed,
        run_id=f"{context.config.run_id}:child-t3:{fingerprint}",
        sample_count=context.config.downstream_t3_samples,
    )
    rows = _score_t3_actions(observation, actions, batch.particles, context)
    values = [float(row["score"]) for row in rows]
    selected_index = canonical_descending_indices(values, actions)[0]
    selected = actions[selected_index]
    context.t3_second_action_cache[fingerprint] = selected
    return selected


def _aggregate_terminal_stats(
    scores: Sequence[float],
    final_boards: Sequence[Board],
) -> dict[str, Any]:
    if not scores or len(scores) != len(final_boards):
        raise Turn3TeacherError("terminal rollout result is empty or inconsistent")
    board_scores = [score_board(board.top, board.middle, board.bottom) for board in final_boards]
    busted = sum(int(value.busted) for value in board_scores)
    fl_entry = sum(int(value.fl_entry.qualifies) for value in board_scores)
    count = len(scores)
    top_categories = Counter(_category_name(value.top_value) for value in board_scores)
    middle_categories = Counter(_category_name(value.middle_value) for value in board_scores)
    bottom_categories = Counter(_category_name(value.bottom_value) for value in board_scores)
    mean = sum(float(score) for score in scores) / count
    return {
        "score": mean,
        "joint_ev": mean,
        "future_count": count,
        "non_bust_future_count": count - busted,
        "bust_count": busted,
        "bust_rate": busted / count,
        "fl_entry_count": fl_entry,
        "fl_entry_rate": fl_entry / count,
        "royalty_mean": sum(value.total_royalty for value in board_scores) / count,
        "top_royalty_mean": sum(value.top_royalty for value in board_scores) / count,
        "middle_royalty_mean": sum(value.middle_royalty for value in board_scores) / count,
        "bottom_royalty_mean": sum(value.bottom_royalty for value in board_scores) / count,
        "top_category_counts": dict(top_categories),
        "middle_category_counts": dict(middle_categories),
        "bottom_category_counts": dict(bottom_categories),
        "score_min": min(scores),
        "score_max": max(scores),
    }


def _require_t3_observation(value: object) -> ActorObservation:
    if not isinstance(value, ActorObservation):
        raise TypeError(
            "T3 teacher requires ActorObservation; raw dead_cards, WorldState, and replay truth are forbidden"
        )
    if value.street != "T3":
        raise Turn3TeacherError("T3 teacher requires a T3 observation")
    expected_opponent = 9 if value.to_act_order == "first" else 11
    if value.hero_board.card_count() != 9 or value.opponent_public_board.card_count() != expected_opponent:
        raise Turn3TeacherError(
            "invalid live T3 geometry for the requested action order"
        )
    return value


def _assert_disjoint_batches(
    candidate: HiddenCardParticleBatch,
    evaluation: HiddenCardParticleBatch,
) -> None:
    candidate_keys = {particle.rng_key_digest for particle in candidate.particles}
    evaluation_keys = {particle.rng_key_digest for particle in evaluation.particles}
    if candidate_keys & evaluation_keys:
        raise Turn3TeacherError(
            "candidate-selection and evaluation particle RNG keys overlap"
        )


def _validate_explicit_worlds(
    observation: ActorObservation,
    worlds: Sequence[T3ExplicitWorld],
) -> None:
    known = set(observation.known_unavailable_cards())
    minimum_future = 9 if observation.to_act_order == "first" else 6
    weight_sum = 0.0
    world_ids: set[str] = set()
    world_contents: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for world in worlds:
        if not world.world_id or world.world_id in world_ids:
            raise Turn3TeacherError("explicit T3 world IDs must be non-empty and unique")
        world_ids.add(world.world_id)
        if not math.isfinite(world.weight) or world.weight <= 0.0:
            raise Turn3TeacherError("explicit T3 world weights must be finite and positive")
        if len(world.opponent_private_discards) != observation.opponent_discard_count:
            raise Turn3TeacherError("explicit T3 opponent discard count is inconsistent")
        if len(world.future_cards) != minimum_future:
            raise Turn3TeacherError(
                "explicit T3 world must contain exactly the future cards consumed by this root"
            )
        hidden = (
            *world.opponent_private_discards,
            *world.future_cards,
        )
        try:
            validate_cards(hidden)
        except ValueError as exc:
            raise Turn3TeacherError(str(exc)) from exc
        if known.intersection(hidden):
            raise Turn3TeacherError("explicit T3 world overlaps actor-visible cards")
        content = (world.opponent_private_discards, world.future_cards)
        if content in world_contents:
            raise Turn3TeacherError("explicit T3 support contains a duplicate world")
        world_contents.add(content)
        weight_sum += world.weight
    if abs(weight_sum - 1.0) > 1e-12:
        raise Turn3TeacherError("explicit T3 support weights must sum to one")


def _require_legal_selected_action(
    observation: ActorObservation,
    selected: Action,
) -> None:
    selected_key = action_key(selected)
    if all(
        action_key(action) != selected_key
        for action in generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
    ):
        raise Turn3TeacherError("explicit continuation selector returned an illegal action")


def _action_by_token(actions: Sequence[Action], token: str) -> Action:
    for action in actions:
        if action_key(action).to_token() == token:
            return action
    raise RuntimeError("selected child ActionKey is not legal")


def _continuation_policy_fingerprint(config: JointExactConfig) -> str:
    payload = {
        "id": T3_CONTINUATION_POLICY_ID,
        "opponent_t3_policy": config.opponent_t3_policy,
        "downstream_t3_samples": config.downstream_t3_samples,
        "downstream_t4_samples": config.downstream_t4_samples,
        "seed": config.seed,
        "run_id": config.run_id,
    }
    return _digest(payload)


def _category_name(value: tuple[int, tuple[int, ...]]) -> str:
    return HAND_CATEGORY_NAMES.get(int(value[0]), f"category_{value[0]}")


def _action_payload(action: Action, original_index: int) -> dict[str, Any]:
    return {
        "original_index": original_index,
        "action_key": action_key(action).to_token(),
        "placements": [list(item) for item in action.placements],
        "discards": list(action.discards),
    }


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def future_digest(futures: Sequence[Sequence[str]]) -> str:
    """Compatibility utility for deterministic explicit-support tests."""

    return _digest([sorted(future) for future in futures])


def read_state_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_state_records(
    input_paths: Sequence[Path], max_states: int | None
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for input_path in input_paths:
        for state in read_state_jsonl(input_path):
            record = dict(state)
            record["source_input_path"] = str(input_path)
            states.append(record)
            if max_states is not None and len(states) >= max_states:
                return states
    return states


def evaluate_state_record(
    state: Mapping[str, Any],
    *,
    sample_id: int,
    config: JointExactConfig,
) -> dict[str, Any]:
    """Evaluate a versioned observation row; legacy raw-card rows fail closed."""

    raw_observation = state.get("policy_observation")
    if not isinstance(raw_observation, Mapping):
        raise InformationSetError(
            "M2 T3 teacher input requires versioned policy_observation; legacy board/dead_cards rows are forbidden"
        )
    observation = ActorObservation.from_dict(raw_observation)
    per_state_config = JointExactConfig(
        candidate_samples=config.candidate_samples,
        evaluation_samples=config.evaluation_samples,
        downstream_t3_samples=config.downstream_t3_samples,
        downstream_t4_samples=config.downstream_t4_samples,
        seed=config.seed,
        candidate_seed=config.candidate_seed,
        evaluation_seed=config.evaluation_seed,
        # Address chance by the semantic root, not row/chunk position.  This
        # keeps whole-file, reordered, and sharded generation identical.
        run_id=f"{config.run_id}:root:{observation.fingerprint()}",
        opponent_t3_policy=config.opponent_t3_policy,
        actor=config.actor,
        seat=observation.seat,
        to_act_order=observation.to_act_order,
        use_final_turn_cache=config.use_final_turn_cache,
    )
    sample = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=per_state_config,
    )
    sample["sample_id"] = sample_id
    sample["source"] = state.get("source", "unknown")
    sample["source_input_path"] = state.get("source_input_path")
    sample["source_state"] = {
        "state_id": state.get("state_id"),
        "hand_seed": state.get("hand_seed"),
        "hand_index": state.get("hand_index"),
        "source_input_path": state.get("source_input_path"),
        "visibility_model": state.get("visibility_model", "actor_observation_v1"),
        "discard_visibility": state.get("discard_visibility", "own_private_only"),
    }
    if "selection" in state:
        sample["selection"] = state["selection"]
    return sample


def write_summary_csv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for action in sample.get("actions", ()):
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "seat": sample.get("seat"),
                    "original_index": action.get("original_index"),
                    "sorted_index": action.get("sorted_index"),
                    "action_key": action.get("action_key"),
                    "selection_score": action.get("selection_score"),
                    "score": action.get("score"),
                    "regret_vs_best": action.get("evaluation_regret_vs_sample_best"),
                    "future_count": action.get("future_count"),
                    "bust_rate": action.get("bust_rate"),
                    "fl_entry_rate": action.get("fl_entry_rate"),
                    "royalty_mean": action.get("royalty_mean"),
                    "placements": json.dumps(
                        action.get("placements", ()), separators=(",", ":")
                    ),
                    "discards": json.dumps(
                        action.get("discards", ()), separators=(",", ":")
                    ),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--candidate-samples", type=int, default=4)
    parser.add_argument("--evaluation-samples", type=int, default=8)
    parser.add_argument("--downstream-t3-samples", type=int, default=2)
    parser.add_argument("--downstream-t4-samples", type=int, default=8)
    parser.add_argument(
        "--future-samples",
        type=int,
        help="Deprecated alias setting both candidate/evaluation counts; streams remain disjoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--run-id", default="hu-m2-t3-cli")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--disable-final-turn-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_samples = args.candidate_samples
    evaluation_samples = args.evaluation_samples
    if args.future_samples is not None:
        candidate_samples = evaluation_samples = args.future_samples
    config = JointExactConfig(
        candidate_samples=candidate_samples,
        evaluation_samples=evaluation_samples,
        downstream_t3_samples=args.downstream_t3_samples,
        downstream_t4_samples=args.downstream_t4_samples,
        seed=args.seed,
        candidate_seed=args.candidate_seed,
        evaluation_seed=args.evaluation_seed,
        run_id=args.run_id,
        use_final_turn_cache=not args.disable_final_turn_cache,
    )
    states = collect_state_records(args.input, args.max_states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    samples: list[dict[str, Any]] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for index, state in enumerate(states):
            sample = evaluate_state_record(state, sample_id=index, config=config)
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            samples.append(sample)
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {"event": "progress", "states": index + 1},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    temporary.replace(args.output)
    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, samples)
    print(
        json.dumps(
            {
                "schema": T3_SEQUENTIAL_TEACHER_SCHEMA,
                "samples": len(samples),
                "output": str(args.output),
                "summary_csv": str(args.summary_csv) if args.summary_csv else None,
                "candidate_samples": candidate_samples,
                "evaluation_samples": evaluation_samples,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
