"""Hidden-discard-safe M4 teacher for a second-seat T1 decision.

The root uses disjoint common-random particle batches for candidate selection
and locked evaluation.  T2 actions come from an explicitly supplied fixed
policy through ``choose_action_observation``.  T3 and T4 actions come from the
M3 native search.  Every downstream selector receives only a freshly built
``ActorObservation`` and is cached by its information-set fingerprint.

The resulting scores are search-teacher diagnostics.  They are not realized
match EV and must not be used as a runtime confidence gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_belief import HiddenCardParticle, HiddenCardParticleBatch, sample_hidden_card_particles
from .hu_infoset import ActorObservation, WorldState
from .hu_late_street_teacher import T4SearchConfig
from .hu_m3_rust import (
    evaluate_batch,
    evaluate_t3,
    evaluate_t4,
    t3_request,
    t4_request,
)
from .hu_m4_teacher_contract import (
    HU_M4_T1_SECOND_TEACHER_SCHEMA,
    T1_SECOND_LIVE_SCHEDULE,
    child_policy_cache_key,
    child_policy_decision_seed,
    require_disjoint_root_rng_keys,
    require_t1_second_root,
)
from .hu_turn3_joint_exact_teacher import JointExactConfig
from .state import Board
from .teacher import terminal_score
from .turn3_model import sample_to_matrix as self_board_sample_to_matrix


M4_T1_SECOND_SOLVER_ID = "python_orchestrated_m3_t3_t4_t1_second_v1"
M4_PAIRED_DELTA_SUMMARY_SCHEMA = "hu_m4_paired_delta_summary_v1"


@dataclass(frozen=True)
class M4T1TeacherConfig:
    candidate_samples: int = 4
    evaluation_samples: int = 8
    candidate_seed: int = 2026071401
    evaluation_seed: int = 2026071402
    run_id: str = "hu-m4-t1-second"
    t2_policy_id: str = "explicit-fixed-t2-policy"
    child_policy_seed: int = 2026071403
    t3_candidate_samples: int = 1
    t3_evaluation_samples: int = 1
    t3_downstream_samples: int = 1
    t4_candidate_samples: int = 1
    t4_evaluation_samples: int = 1
    batch_child_selectors: bool = False

    def __post_init__(self) -> None:
        positive = (
            "candidate_samples",
            "evaluation_samples",
            "t3_candidate_samples",
            "t3_evaluation_samples",
            "t3_downstream_samples",
            "t4_candidate_samples",
            "t4_evaluation_samples",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "candidate_seed",
            "evaluation_seed",
            "child_policy_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if not self.run_id or not self.t2_policy_id:
            raise ValueError("run_id and t2_policy_id must not be empty")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")


@dataclass(frozen=True)
class _ActionScores:
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def standard_error(self) -> float:
        if len(self.values) <= 1:
            return 0.0
        mean = self.mean
        variance = sum((value - mean) ** 2 for value in self.values) / (
            len(self.values) - 1
        )
        return math.sqrt(variance / len(self.values))


def _paired_delta_summary(
    candidate: _ActionScores, baseline: _ActionScores
) -> dict[str, Any]:
    """Summarize common-future candidate-minus-baseline outcomes.

    The two score vectors must have been produced from the same ordered
    particle batch.  Keeping the subtraction paired preserves the covariance
    benefit which is lost when two marginal standard errors are combined.
    """

    if len(candidate.values) != len(baseline.values) or not candidate.values:
        raise ValueError("paired action scores require equal non-empty vectors")
    deltas = np.asarray(
        [
            float(candidate_value - baseline_value)
            for candidate_value, baseline_value in zip(
                candidate.values, baseline.values, strict=True
            )
        ],
        dtype=np.float64,
    )
    if not np.isfinite(deltas).all():
        raise ValueError("paired action deltas must be finite")
    count = int(deltas.size)
    mean = float(np.mean(deltas))
    standard_deviation = float(np.std(deltas, ddof=1)) if count > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(count)
    return {
        "schema": M4_PAIRED_DELTA_SUMMARY_SCHEMA,
        "count": count,
        "mean": mean,
        "standard_error": float(standard_error),
        "std": standard_deviation,
        "min": float(np.min(deltas)),
        "p01": float(np.quantile(deltas, 0.01)),
        "p05": float(np.quantile(deltas, 0.05)),
        "p25": float(np.quantile(deltas, 0.25)),
        "p50": float(np.quantile(deltas, 0.50)),
        "p75": float(np.quantile(deltas, 0.75)),
        "p95": float(np.quantile(deltas, 0.95)),
        "p99": float(np.quantile(deltas, 0.99)),
        "max": float(np.max(deltas)),
        "lt0_rate": float(np.mean(deltas < 0.0)),
        "le_neg6_rate": float(np.mean(deltas <= -6.0)),
        "le_neg12_rate": float(np.mean(deltas <= -12.0)),
        "le_neg20_rate": float(np.mean(deltas <= -20.0)),
    }


class _ChildSelector:
    def __init__(
        self,
        *,
        t2_policies: Mapping[str, object],
        config: M4T1TeacherConfig,
        library: Any | None,
    ) -> None:
        if set(t2_policies) != {"first", "second"}:
            raise ValueError("t2_policies must contain exactly first and second")
        self.t2_policies = dict(t2_policies)
        self.config = config
        self.library = library
        self.cache: dict[tuple[str, str, str], Action] = {}

    def choose(self, observation: ActorObservation) -> Action:
        if observation.street == "T2":
            policy_id = f"{self.config.t2_policy_id}:seat={observation.seat}"
        elif observation.street == "T3":
            policy_id = self._t3_policy_id()
        elif observation.street == "T4":
            policy_id = self._t4_policy_id()
        else:
            raise ValueError(f"unsupported child street: {observation.street!r}")
        cache_key = child_policy_cache_key(policy_id, observation)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if observation.street == "T2":
            selected = self._choose_t2(observation, policy_id)
        elif observation.street == "T3":
            selected = self._choose_t3(observation)
        else:
            selected = self._choose_t4(observation)
        selected = _remap_legal_action(observation, action_key(selected).to_token())
        self.cache[cache_key] = selected
        return selected

    def choose_many(
        self, observations: Sequence[ActorObservation]
    ) -> tuple[Action, ...]:
        """Choose one semantic action per infoset with batched model/native calls."""

        if not observations:
            return ()
        streets = {observation.street for observation in observations}
        seats = {observation.seat for observation in observations}
        if len(streets) != 1 or len(seats) != 1:
            raise ValueError("batched child selection requires one street and seat")
        street = observations[0].street
        if street == "T2":
            policy_id = f"{self.config.t2_policy_id}:seat={observations[0].seat}"
        elif street == "T3":
            policy_id = self._t3_policy_id()
        elif street == "T4":
            policy_id = self._t4_policy_id()
        else:
            raise ValueError(f"unsupported child street: {street!r}")

        keyed = [
            (child_policy_cache_key(policy_id, observation), observation)
            for observation in observations
        ]
        missing: dict[tuple[str, str, str], ActorObservation] = {}
        for cache_key, observation in keyed:
            if cache_key not in self.cache:
                missing.setdefault(cache_key, observation)
        if missing:
            missing_keys = tuple(missing)
            missing_observations = tuple(missing.values())
            if street == "T2":
                selected = self._choose_t2_many(missing_observations, policy_id)
            elif street == "T3":
                selected = self._choose_t3_many(missing_observations)
            else:
                selected = self._choose_t4_many(missing_observations)
            if len(selected) != len(missing_keys):
                raise ValueError("batched child selector returned the wrong action count")
            for cache_key, observation, action in zip(
                missing_keys, missing_observations, selected, strict=True
            ):
                remapped = _remap_legal_action(
                    observation, action_key(action).to_token()
                )
                self.cache[cache_key] = remapped
        return tuple(self.cache[cache_key] for cache_key, _observation in keyed)

    def _choose_t2(self, observation: ActorObservation, policy_id: str) -> Action:
        policy = self.t2_policies[observation.seat]
        chooser = getattr(policy, "choose_action_observation", None)
        if not callable(chooser):
            raise TypeError(
                "M4 fixed T2 continuation must implement choose_action_observation"
            )
        seed = child_policy_decision_seed(
            base_seed=self.config.child_policy_seed,
            policy_id=policy_id,
            observation=observation,
        )
        return chooser(
            observation,
            hand_id=None,
            game_id=None,
            decision_seed=seed,
        )

    def _choose_t2_many(
        self,
        observations: Sequence[ActorObservation],
        policy_id: str,
    ) -> tuple[Action, ...]:
        policy = self.t2_policies[observations[0].seat]
        # Exact fast path for the fixed self-board RegularAiPolicy used by the
        # M4 teacher. Subclasses may alter T2 semantics and therefore fall back
        # to the observation-safe scalar API.
        from .policy import RegularAiPolicy, policy_sample

        if type(policy) is not RegularAiPolicy or getattr(policy, "turn2_model", None) is None:
            return tuple(self._choose_t2(observation, policy_id) for observation in observations)
        model = policy.turn2_model
        action_lists: list[list[Action]] = []
        feature_blocks: list[np.ndarray] = []
        try:
            for observation in observations:
                if observation.street != "T2" or observation.seat != policy.seat:
                    raise ValueError("T2 batch observation disagrees with fixed policy")
                actions = generate_turn_actions(
                    observation.hero_board, observation.dealt_cards
                )
                if not actions:
                    raise ValueError("T2 batch observation has no legal actions")
                sample = policy_sample(
                    observation.hero_board, observation.dealt_cards, actions
                )
                features, _targets = self_board_sample_to_matrix(sample)
                action_lists.append(actions)
                feature_blocks.append(features)
            predictions = np.asarray(
                model.predict_matrix(np.vstack(feature_blocks)), dtype=np.float64
            ).reshape(-1)
            if not np.isfinite(predictions).all():
                raise ValueError("T2 batch model returned non-finite predictions")
            selected: list[Action] = []
            cursor = 0
            for actions, features in zip(action_lists, feature_blocks, strict=True):
                block = predictions[cursor : cursor + features.shape[0]]
                cursor += features.shape[0]
                if block.shape != (len(actions),):
                    raise ValueError("T2 batch prediction length mismatch")
                best = float(np.max(block))
                tied = [
                    index for index, value in enumerate(block) if float(value) == best
                ]
                best_index = min(
                    tied, key=lambda index: action_key(actions[index]).sort_key()
                )
                selected.append(actions[best_index])
            if cursor != predictions.shape[0]:
                raise ValueError("T2 batch returned surplus predictions")
            return tuple(selected)
        except Exception:
            return tuple(self._choose_t2(observation, policy_id) for observation in observations)

    def _choose_t3(self, observation: ActorObservation) -> Action:
        config = self._t3_config(observation)
        result = evaluate_t3(observation, config=config, library=self.library)
        return _remap_legal_action(observation, _selected_token(result))

    def _t3_config(self, observation: ActorObservation) -> JointExactConfig:
        return JointExactConfig(
            candidate_samples=self.config.t3_candidate_samples,
            evaluation_samples=self.config.t3_evaluation_samples,
            downstream_t3_samples=self.config.t3_downstream_samples,
            downstream_t4_samples=self.config.t3_downstream_samples,
            seed=self.config.child_policy_seed,
            candidate_seed=self.config.child_policy_seed + 101,
            evaluation_seed=self.config.child_policy_seed + 102,
            run_id=f"m4-child-t3:{self._t3_policy_id()}",
            seat=observation.seat,
            to_act_order=observation.to_act_order,
        )

    def _choose_t3_many(
        self, observations: Sequence[ActorObservation]
    ) -> tuple[Action, ...]:
        requests = [
            t3_request(observation, config=self._t3_config(observation))
            for observation in observations
        ]
        results = evaluate_batch(requests, library=self.library)
        return tuple(
            _remap_legal_action(observation, _selected_token(result))
            for observation, result in zip(observations, results, strict=True)
        )

    def _choose_t4(self, observation: ActorObservation) -> Action:
        config = self._t4_config()
        result = evaluate_t4(observation, config=config, library=self.library)
        return _remap_legal_action(observation, _selected_token(result))

    def _t4_config(self) -> T4SearchConfig:
        return T4SearchConfig(
            candidate_samples=self.config.t4_candidate_samples,
            evaluation_samples=self.config.t4_evaluation_samples,
            seed=self.config.child_policy_seed,
            candidate_seed=self.config.child_policy_seed + 201,
            evaluation_seed=self.config.child_policy_seed + 202,
            run_id=f"m4-child-t4:{self._t4_policy_id()}",
        )

    def _choose_t4_many(
        self, observations: Sequence[ActorObservation]
    ) -> tuple[Action, ...]:
        config = self._t4_config()
        results = evaluate_batch(
            [t4_request(observation, config=config) for observation in observations],
            library=self.library,
        )
        return tuple(
            _remap_legal_action(observation, _selected_token(result))
            for observation, result in zip(observations, results, strict=True)
        )

    def _t3_policy_id(self) -> str:
        return (
            "m3-t3"
            f":c={self.config.t3_candidate_samples}"
            f":e={self.config.t3_evaluation_samples}"
            f":d={self.config.t3_downstream_samples}"
            f":seed={self.config.child_policy_seed}"
        )

    def _t4_policy_id(self) -> str:
        return (
            "m3-t4"
            f":c={self.config.t4_candidate_samples}"
            f":e={self.config.t4_evaluation_samples}"
            f":seed={self.config.child_policy_seed}"
        )


def evaluate_t1_second_actions(
    observation: ActorObservation,
    *,
    t2_policies: Mapping[str, object],
    baseline_action: Action | None = None,
    config: M4T1TeacherConfig | None = None,
    library: Any | None = None,
) -> dict[str, Any]:
    """Evaluate every legal T1-second action and lock before evaluation."""

    require_t1_second_root(observation)
    config = config or M4T1TeacherConfig()
    candidate = sample_hidden_card_particles(
        observation,
        base_seed=config.candidate_seed,
        run_id=f"{config.run_id}:candidate_selection",
        sample_count=config.candidate_samples,
    )
    evaluation = sample_hidden_card_particles(
        observation,
        base_seed=config.evaluation_seed,
        run_id=f"{config.run_id}:locked_evaluation",
        sample_count=config.evaluation_samples,
    )
    candidate.validate_against(observation)
    evaluation.validate_against(observation)
    expected_candidate_run_id = f"{config.run_id}:candidate_selection"
    expected_evaluation_run_id = f"{config.run_id}:locked_evaluation"
    if (
        candidate.base_seed != config.candidate_seed
        or candidate.run_id != expected_candidate_run_id
        or candidate.start_index != 0
    ):
        raise ValueError("candidate belief provenance disagrees with config")
    if (
        evaluation.base_seed != config.evaluation_seed
        or evaluation.run_id != expected_evaluation_run_id
        or evaluation.start_index != 0
    ):
        raise ValueError("evaluation belief provenance disagrees with config")
    if len(candidate.particles) != config.candidate_samples:
        raise ValueError("candidate belief count disagrees with config")
    if len(evaluation.particles) != config.evaluation_samples:
        raise ValueError("evaluation belief count disagrees with config")
    candidate_keys = tuple(row.rng_key_digest for row in candidate.particles)
    evaluation_keys = tuple(row.rng_key_digest for row in evaluation.particles)
    require_disjoint_root_rng_keys(candidate_keys, evaluation_keys)

    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise ValueError("T1-second observation has no legal actions")
    baseline_index: int | None = None
    baseline_token: str | None = None
    if baseline_action is not None:
        baseline_token = action_key(baseline_action).to_token()
        baseline_matches = [
            index
            for index, action in enumerate(actions)
            if action_key(action).to_token() == baseline_token
        ]
        if len(baseline_matches) != 1:
            raise ValueError("baseline action is not uniquely legal at the M4 root")
        baseline_index = baseline_matches[0]
    selector = _ChildSelector(
        t2_policies=t2_policies,
        config=config,
        library=library,
    )
    score_actions = (
        _score_actions_batched if config.batch_child_selectors else _score_actions
    )
    candidate_scores = score_actions(observation, actions, candidate, selector)
    candidate_values = tuple(row.mean for row in candidate_scores)
    ranking = canonical_descending_indices(candidate_values, actions)
    selected_index = ranking[0]
    selected_token = action_key(actions[selected_index]).to_token()

    # The selected semantic action is frozen before this independent pass.
    evaluation_scores = score_actions(observation, actions, evaluation, selector)
    evaluation_values = tuple(row.mean for row in evaluation_scores)
    evaluation_best = max(evaluation_values)
    candidate_second = (
        candidate_values[ranking[1]] if len(ranking) > 1 else candidate_values[selected_index]
    )
    rows = []
    for sorted_index, original_index in enumerate(ranking):
        action = actions[original_index]
        row = {
            "original_index": original_index,
            "sorted_index": sorted_index,
            "action_key": action_key(action).to_token(),
            "placements": [list(value) for value in action.placements],
            "discards": list(action.discards),
            "selection_score": candidate_values[original_index],
            "selection_standard_error": candidate_scores[
                original_index
            ].standard_error,
            "evaluation_score": evaluation_values[original_index],
            "evaluation_standard_error": evaluation_scores[
                original_index
            ].standard_error,
            "evaluation_regret_vs_sample_best": (
                evaluation_best - evaluation_values[original_index]
            ),
            "selected_by_candidate_plan": original_index == selected_index,
            "selection_future_count": len(candidate.particles),
            "evaluation_future_count": len(evaluation.particles),
        }
        if baseline_index is not None:
            row["evaluation_delta_vs_baseline"] = _paired_delta_summary(
                evaluation_scores[original_index], evaluation_scores[baseline_index]
            )
        rows.append(row)

    result = {
        "status": "ok",
        "schema": HU_M4_T1_SECOND_TEACHER_SCHEMA,
        "solver_id": M4_T1_SECOND_SOLVER_ID,
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "observation_fingerprint": observation.fingerprint(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "selected_action_original_index": selected_index,
        "selected_action_key": selected_token,
        "selection_score_gap": candidate_values[selected_index] - candidate_second,
        "selected_action_evaluation_score": evaluation_values[selected_index],
        "selected_action_evaluation_standard_error": evaluation_scores[
            selected_index
        ].standard_error,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection": (
            evaluation_best - evaluation_values[selected_index]
        ),
        "candidate_belief_digest": candidate.digest(),
        "evaluation_belief_digest": evaluation.digest(),
        "candidate_rng_key_digests": list(candidate_keys),
        "evaluation_rng_key_digests": list(evaluation_keys),
        "sample_independence": "disjoint_particle_rng_keys",
        "root_selection_lock": "candidate_action_key_locked_before_evaluation",
        "search_config": {
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "candidate_seed": config.candidate_seed,
            "evaluation_seed": config.evaluation_seed,
            "run_id": config.run_id,
            "child_policy_seed": config.child_policy_seed,
            "t3_candidate_samples": config.t3_candidate_samples,
            "t3_evaluation_samples": config.t3_evaluation_samples,
            "t3_downstream_samples": config.t3_downstream_samples,
            "t4_candidate_samples": config.t4_candidate_samples,
            "t4_evaluation_samples": config.t4_evaluation_samples,
            "batch_child_selectors": config.batch_child_selectors,
        },
        "continuation_policy": {
            "t2_policy_id": config.t2_policy_id,
            "t3_selector": "m3_rust_evaluate_t3",
            "t4_selector": "m3_rust_evaluate_t4",
            "child_selector_execution": (
                "batched_infoset_locked_v1"
                if config.batch_child_selectors
                else "scalar_infoset_locked_v1"
            ),
            "strategy_fusion_guard": (
                "child_action_cache_key_is_policy_id_plus_actor_observation_fingerprint"
            ),
            "outer_future_index_in_child_seed": False,
            "outer_action_index_in_child_seed": False,
        },
        "live_schedule": [
            {
                "seat": step.seat,
                "street": step.street,
                "draw_offset": step.draw_offset,
            }
            for step in T1_SECOND_LIVE_SCHEDULE
        ],
        "child_information_set_count": len(selector.cache),
        "actions": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    }
    if baseline_token is not None:
        result["paired_delta_baseline_action_key"] = baseline_token
        result["paired_delta_common_futures"] = True
    return result


def _score_actions(
    observation: ActorObservation,
    actions: Sequence[Action],
    batch: HiddenCardParticleBatch,
    selector: _ChildSelector,
) -> tuple[_ActionScores, ...]:
    scored = []
    for action in actions:
        values = tuple(
            _rollout_t1_second(observation, action, particle, selector)
            for particle in batch.particles
        )
        scored.append(_ActionScores(values))
    return tuple(scored)


@dataclass
class _PendingRollout:
    action_index: int
    particle_index: int
    particle: HiddenCardParticle
    boards: list[Board]
    private_discards: list[list[str]]


def _score_actions_batched(
    observation: ActorObservation,
    actions: Sequence[Action],
    batch: HiddenCardParticleBatch,
    selector: _ChildSelector,
) -> tuple[_ActionScores, ...]:
    """Advance every root-action/future path one live infoset layer at a time."""

    pending: list[_PendingRollout] = []
    for action_index, root_action in enumerate(actions):
        for particle_index, particle in enumerate(batch.particles):
            particle.validate_against(observation)
            boards = [observation.opponent_public_board, observation.hero_board]
            private_discards = [list(particle.opponent_private_discards), []]
            boards[1] = boards[1].place(root_action.placements)
            private_discards[1].extend(root_action.discards)
            pending.append(
                _PendingRollout(
                    action_index=action_index,
                    particle_index=particle_index,
                    particle=particle,
                    boards=boards,
                    private_discards=private_discards,
                )
            )

    for step in T1_SECOND_LIVE_SCHEDULE:
        player = 0 if step.seat == "first" else 1
        children: list[ActorObservation] = []
        for rollout in pending:
            dealt = rollout.particle.draw(3, offset=step.draw_offset)
            world = WorldState(
                boards=(rollout.boards[0], rollout.boards[1]),
                private_discards=(
                    tuple(rollout.private_discards[0]),
                    tuple(rollout.private_discards[1]),
                ),
                street=step.street,
                next_player=player,
                scoring=observation.scoring,
            )
            children.append(world.observe(player, dealt))
        selected = selector.choose_many(children)
        for rollout, chosen in zip(pending, selected, strict=True):
            rollout.boards[player] = rollout.boards[player].place(chosen.placements)
            rollout.private_discards[player].extend(chosen.discards)

    values: list[list[float | None]] = [
        [None] * len(batch.particles) for _action in actions
    ]
    fl_ev = {cards: value for cards, value in observation.scoring.fl_ev}
    for rollout in pending:
        if not rollout.boards[0].is_complete() or not rollout.boards[1].is_complete():
            raise ValueError("M4 batched rollout did not reach two complete boards")
        score, _ = terminal_score(
            rollout.boards[1], rollout.boards[0], fl_ev=fl_ev
        )
        values[rollout.action_index][rollout.particle_index] = float(score)
    return tuple(
        _ActionScores(tuple(float(value) for value in action_values))
        for action_values in values
    )


def _rollout_t1_second(
    observation: ActorObservation,
    root_action: Action,
    particle: HiddenCardParticle,
    selector: _ChildSelector,
) -> float:
    particle.validate_against(observation)
    boards = [observation.opponent_public_board, observation.hero_board]
    private_discards = [list(particle.opponent_private_discards), []]
    boards[1] = boards[1].place(root_action.placements)
    private_discards[1].extend(root_action.discards)

    for step in T1_SECOND_LIVE_SCHEDULE:
        player = 0 if step.seat == "first" else 1
        dealt = particle.draw(3, offset=step.draw_offset)
        world = WorldState(
            boards=(boards[0], boards[1]),
            private_discards=(
                tuple(private_discards[0]),
                tuple(private_discards[1]),
            ),
            street=step.street,
            next_player=player,
            scoring=observation.scoring,
        )
        child = world.observe(player, dealt)
        chosen = selector.choose(child)
        boards[player] = boards[player].place(chosen.placements)
        private_discards[player].extend(chosen.discards)

    if not boards[0].is_complete() or not boards[1].is_complete():
        raise ValueError("M4 T1-second rollout did not reach two complete boards")
    score, _ = terminal_score(
        boards[1],
        boards[0],
        fl_ev={cards: value for cards, value in observation.scoring.fl_ev},
    )
    return float(score)


def _selected_token(result: Mapping[str, Any]) -> str:
    token = result.get("selected_action_key")
    if not isinstance(token, str) or not token:
        raise ValueError("M3 child result has no selected_action_key")
    return token


def _remap_legal_action(observation: ActorObservation, token: str) -> Action:
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    matches = [action for action in actions if action_key(action).to_token() == token]
    if len(matches) != 1:
        raise ValueError("child selector ActionKey is not uniquely legal")
    return matches[0]


__all__ = [
    "M4_PAIRED_DELTA_SUMMARY_SCHEMA",
    "M4_T1_SECOND_SOLVER_ID",
    "M4T1TeacherConfig",
    "evaluate_t1_second_actions",
]
