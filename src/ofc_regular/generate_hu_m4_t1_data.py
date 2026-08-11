"""Generate restart-safe, hidden-discard-safe M4 T1-second teacher shards.

The generator never resolves ``current``.  Root, baseline, and continuation
policies are named explicitly, and the only card-bearing value handed to the
teacher is :class:`ActorObservation`.  Candidate-selection and locked-
evaluation futures are sampled inside :mod:`hu_m4_t1_teacher`.

Teacher scores are diagnostic labels for imitation/search training.  They are
not realized match EV and are never emitted as a runtime confidence bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import (
    action_key,
    index_actions_by_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .cards import create_deck
from .hu_infoset import ActorObservation, WorldState
from .hu_m4_t1_teacher import (
    M4_PAIRED_DELTA_SUMMARY_SCHEMA,
    M4T1TeacherConfig,
    evaluate_t1_second_actions,
)
from .hu_m4_teacher_contract import HU_M4_T1_SECOND_TEACHER_SCHEMA
from .play_ai import _choose_from_observation, _hand_decision_seed
from .policy import action_to_json, board_to_json
from .state import Board


M4_T1_DATA_SCHEMA = "hu_m4_t1_second_training_sample_v2"
M4_T1_SHARD_SCHEMA = "hu_m4_t1_second_shard_v1"
M4_T1_CHECKPOINT_SCHEMA = "hu_m4_t1_second_checkpoint_v1"
M4_T1_HEARTBEAT_SCHEMA = "hu_m4_t1_second_heartbeat_v1"
ROOT_GENERATION_POLICY = "explicit_root_population_live_t0_t1_first_v2"
ROOT_POPULATION_SCHEDULE = "weighted_quota_seeded_shuffle_v1"


_ROOT_POLICY_FAMILIES = {
    "stage19_p0": "selective_opening_fixed_chain",
    "stage9f_p2": "selective_t2_fixed_chain",
    "stage7_m5_r10": "conservative_t3_margin",
    "stage3_baseline": "baseline",
    "random_exact_final": "random_off_policy_exact_final",
}


@dataclass(frozen=True)
class M4T1DataConfig:
    roots: int = 100
    seed_start: int = 2026071401
    seed_stride: int = 1_000_003
    candidate_samples: int = 1
    evaluation_samples: int = 1
    child_policy_seed: int = 2026071403
    candidate_seed: int = 2026071401
    evaluation_seed: int = 2026071402
    root_profile: str = "stage3_baseline"
    root_profiles: tuple[str, ...] = ()
    root_profile_weights: tuple[float, ...] = ()
    baseline_profile: str = "stage18_p1"
    t2_profile: str = "stage3_baseline"
    batch_child_selectors: bool = False
    native_batch_threads: int = 4
    opening_lookahead_samples: int = 0
    split: str = "pilot"
    run_id: str = "hu-m4-t1-second"

    def __post_init__(self) -> None:
        for name in (
            "roots",
            "seed_start",
            "seed_stride",
            "candidate_samples",
            "evaluation_samples",
            "child_policy_seed",
            "candidate_seed",
            "evaluation_seed",
            "opening_lookahead_samples",
            "native_batch_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.roots <= 0 or self.seed_stride <= 0:
            raise ValueError("roots and seed_stride must be positive")
        if not 1 <= self.native_batch_threads <= 64:
            raise ValueError("native_batch_threads must be between 1 and 64")
        if self.candidate_samples <= 0 or self.evaluation_samples <= 0:
            raise ValueError("candidate/evaluation samples must be positive")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate/evaluation seeds must be distinct")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")
        if not isinstance(self.root_profiles, tuple) or not isinstance(
            self.root_profile_weights, tuple
        ):
            raise TypeError("root_profiles and root_profile_weights must be tuples")
        if self.root_profiles:
            if len(set(self.root_profiles)) != len(self.root_profiles):
                raise ValueError("root_profiles must not contain duplicates")
            if not all(isinstance(profile, str) and profile for profile in self.root_profiles):
                raise ValueError("root_profiles must contain non-empty strings")
            if self.root_profile_weights and len(self.root_profile_weights) != len(
                self.root_profiles
            ):
                raise ValueError("root profile weights must match root_profiles")
        elif self.root_profile_weights:
            raise ValueError("root profile weights require root_profiles")
        weights = self.effective_root_profile_weights()
        if not all(
            not isinstance(weight, bool)
            and isinstance(weight, (int, float))
            and math.isfinite(float(weight))
            and float(weight) > 0.0
            for weight in weights
        ):
            raise ValueError("root profile weights must be finite and positive")
        if "current" in self.effective_root_profiles():
            raise ValueError("M4 root population must not resolve current")
        if self.root_profile == "current" or self.baseline_profile == "current":
            raise ValueError("M4 data generation must not resolve current")
        if self.t2_profile == "current":
            raise ValueError("M4 continuation must not resolve current")
        if not self.split or not self.run_id:
            raise ValueError("split and run_id must not be empty")

    def hand_seed(self, root_index: int) -> int:
        if not 0 <= root_index < self.roots:
            raise IndexError("root index outside configured shard")
        return self.seed_start + root_index * self.seed_stride

    def effective_root_profiles(self) -> tuple[str, ...]:
        """Return the explicit root population, preserving the v1 default."""

        return self.root_profiles or (self.root_profile,)

    def effective_root_profile_weights(self) -> tuple[float, ...]:
        profiles = self.effective_root_profiles()
        if not self.root_profile_weights:
            return tuple(1.0 for _ in profiles)
        return tuple(float(weight) for weight in self.root_profile_weights)

    def root_profile_schedule(self) -> tuple[str, ...]:
        """Return a deterministic, quota-balanced schedule for this shard."""

        profiles = self.effective_root_profiles()
        weights = self.effective_root_profile_weights()
        total_weight = sum(weights)
        raw_counts = [self.roots * weight / total_weight for weight in weights]
        counts = [int(math.floor(value)) for value in raw_counts]
        remaining = self.roots - sum(counts)
        # Stable largest-remainder allocation makes observed shard ratios match
        # requested weights more closely than independent random draws.
        remainder_order = sorted(
            range(len(profiles)),
            key=lambda index: (-(raw_counts[index] - counts[index]), index),
        )
        for index in remainder_order[:remaining]:
            counts[index] += 1
        schedule = [
            profile
            for profile, count in zip(profiles, counts, strict=True)
            for _ in range(count)
        ]
        rng = random.Random(
            _stable_schedule_seed(
                seed_start=self.seed_start,
                seed_stride=self.seed_stride,
                profiles=profiles,
                weights=weights,
            )
        )
        rng.shuffle(schedule)
        return tuple(schedule)

    def root_profile_for(self, root_index: int) -> str:
        if not 0 <= root_index < self.roots:
            raise IndexError("root index outside configured shard")
        return self.root_profile_schedule()[root_index]


def generate_t1_second_root(
    hand_seed: int,
    *,
    root_policies: Mapping[str, object],
) -> ActorObservation:
    """Play the live prefix and return the second player's T1 observation."""

    if set(root_policies) != {"first", "second"}:
        raise ValueError("root_policies must contain exactly first and second")
    deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    private_discards: list[list[str]] = [[], []]

    def deal(count: int) -> tuple[str, ...]:
        nonlocal cursor
        cards = tuple(deck[cursor : cursor + count])
        cursor += count
        if len(cards) != count:
            raise RuntimeError("M4 root deck exhausted")
        return cards

    def choose_and_advance(player: int, dealt: tuple[str, ...], street: str) -> None:
        world = WorldState(
            boards=(boards[0], boards[1]),
            private_discards=(
                tuple(private_discards[0]),
                tuple(private_discards[1]),
            ),
            street=street,  # type: ignore[arg-type]
            next_player=player,
        )
        observation = world.observe(player, dealt)
        action = _choose_from_observation(
            root_policies[observation.seat],
            observation,
            hand_id=hand_seed,
            game_id=hand_seed,
            decision_seed=_hand_decision_seed(
                base_seed=hand_seed, observation=observation
            ),
        )
        boards[player] = boards[player].place(action.placements)
        private_discards[player].extend(action.discards)

    choose_and_advance(0, deal(5), "T0")
    choose_and_advance(1, deal(5), "T0")
    choose_and_advance(0, deal(3), "T1")
    world = WorldState(
        boards=(boards[0], boards[1]),
        private_discards=(
            tuple(private_discards[0]),
            tuple(private_discards[1]),
        ),
        street="T1",
        next_player=1,
    )
    return world.observe(1, deal(3))


def _stable_schedule_seed(
    *,
    seed_start: int,
    seed_stride: int,
    profiles: Sequence[str],
    weights: Sequence[float],
) -> int:
    payload = {
        "schema": ROOT_POPULATION_SCHEDULE,
        "seed_start": seed_start,
        "seed_stride": seed_stride,
        "profiles": list(profiles),
        "weights": [float(weight) for weight in weights],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _root_policy_family(profile: str) -> str:
    return _ROOT_POLICY_FAMILIES.get(profile, f"explicit_profile:{profile}")


def _root_population_manifest(
    config: M4T1DataConfig, *, completed_roots: int | None = None
) -> dict[str, Any]:
    schedule = config.root_profile_schedule()
    if completed_roots is None:
        completed_roots = len(schedule)
    if not 0 <= completed_roots <= len(schedule):
        raise ValueError("completed_roots is outside the root schedule")
    requested = dict(
        zip(
            config.effective_root_profiles(),
            config.effective_root_profile_weights(),
            strict=True,
        )
    )
    target_counts = Counter(schedule)
    completed_counts = Counter(schedule[:completed_roots])
    schedule_digest = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": ROOT_POPULATION_SCHEDULE,
        "selection_inputs": "root_index_seed_start_seed_stride",
        "schedule_sha256": schedule_digest,
        "profiles": [
            {
                "profile": profile,
                "policy_family": _root_policy_family(profile),
                "requested_weight": float(requested[profile]),
                "normalized_weight": float(requested[profile])
                / sum(requested.values()),
                "target_count": int(target_counts[profile]),
                "completed_count": int(completed_counts[profile]),
            }
            for profile in config.effective_root_profiles()
        ],
    }


def _profile_policy_seed(base_seed: int, profile: str, seat: str) -> int:
    digest = hashlib.sha256(
        f"m4-root-policy-v1:{base_seed}:{profile}:{seat}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def teacher_result_to_sample(
    observation: ActorObservation,
    result: Mapping[str, Any],
    *,
    hand_seed: int,
    root_index: int,
    split: str,
    baseline_action: Action,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a locked-evaluation result into the standard action-row form."""

    if result.get("status") != "ok":
        raise ValueError("M4 teacher result is not ok")
    actions = _legal_actions(observation)
    if result.get("schema") != HU_M4_T1_SECOND_TEACHER_SCHEMA:
        raise ValueError("M4 teacher result schema mismatch")
    if (
        result.get("observation_fingerprint") != observation.fingerprint()
        or result.get("street") != observation.street
        or result.get("seat") != observation.seat
        or result.get("to_act_order") != observation.to_act_order
    ):
        raise ValueError("M4 teacher result belongs to a different observation")
    if result.get("legal_action_set_digest") != legal_action_set_digest(actions):
        raise ValueError("M4 teacher legal action set digest mismatch")
    if result.get("legal_action_order_digest") != ordered_action_mapping_digest(actions):
        raise ValueError("M4 teacher legal action order digest mismatch")
    legal_actions = index_actions_by_key(actions)
    baseline_key = action_key(baseline_action)
    if baseline_key not in legal_actions:
        raise ValueError("baseline action is not legal at the M4 root")
    baseline_token = baseline_key.to_token()
    if result.get("paired_delta_baseline_action_key") != baseline_token:
        raise ValueError("M4 teacher paired-delta baseline ActionKey mismatch")
    if result.get("paired_delta_common_futures") is not True:
        raise ValueError("M4 teacher did not verify paired common futures")
    search_config = result.get("search_config")
    if not isinstance(search_config, Mapping):
        raise ValueError("M4 teacher search_config is missing")
    evaluation_samples = search_config.get("evaluation_samples")
    if (
        isinstance(evaluation_samples, bool)
        or not isinstance(evaluation_samples, int)
        or evaluation_samples <= 0
    ):
        raise ValueError("M4 teacher evaluation sample count is invalid")

    converted: list[dict[str, Any]] = []
    for teacher_row in result.get("actions", ()):
        token = str(teacher_row.get("action_key", ""))
        matches = [
            actions[index]
            for key, index in legal_actions.items()
            if key.to_token() == token
        ]
        if len(matches) != 1:
            raise ValueError("teacher ActionKey does not map uniquely to legal action")
        paired_summary = _validated_paired_delta_summary(
            teacher_row.get("evaluation_delta_vs_baseline"),
            expected_count=evaluation_samples,
        )
        payload = action_to_json(observation.hero_board, matches[0])
        payload.update(
            {
                "action_key": token,
                "score": float(teacher_row["evaluation_score"]),
                "score_se": float(teacher_row["evaluation_standard_error"]),
                "selection_score": float(teacher_row["selection_score"]),
                "selection_score_se": float(
                    teacher_row["selection_standard_error"]
                ),
                "selected_by_candidate_plan": bool(
                    teacher_row["selected_by_candidate_plan"]
                ),
                "teacher_original_index": int(teacher_row["original_index"]),
                "delta_vs_baseline": float(paired_summary["mean"]),
                "delta_se_vs_baseline": float(paired_summary["standard_error"]),
                "paired_delta_vs_baseline": paired_summary,
            }
        )
        converted.append(payload)
    if len(converted) != int(result["legal_action_count"]):
        raise ValueError("teacher did not evaluate every legal action")

    # Training order is defined by the independent evaluation pass, never by
    # candidate selection. ActionKey is the persistent runtime identity.
    converted.sort(key=lambda row: (-float(row["score"]), str(row["action_key"])))
    baseline_row_index = next(
        index
        for index, row in enumerate(converted)
        if row["action_key"] == baseline_token
    )
    best_score = float(converted[0]["score"])
    second_score = float(converted[min(1, len(converted) - 1)]["score"])
    baseline_score = float(converted[baseline_row_index]["score"])
    for row in converted:
        marginal_delta = float(row["score"]) - baseline_score
        if not math.isclose(
            float(row["delta_vs_baseline"]),
            marginal_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("paired delta mean disagrees with evaluation means")
    baseline_summary = converted[baseline_row_index]["paired_delta_vs_baseline"]
    if any(
        float(baseline_summary[key]) != 0.0
        for key in (
            "mean",
            "standard_error",
            "std",
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
            "lt0_rate",
            "le_neg6_rate",
            "le_neg12_rate",
            "le_neg20_rate",
        )
    ):
        raise ValueError("baseline paired-delta summary must be exactly zero")

    observation_payload = observation.to_dict()
    if "opponent_private_discards" in observation_payload:
        raise AssertionError("policy observation exposed opponent private discards")
    return {
        "schema": M4_T1_DATA_SCHEMA,
        "rule_set": "heads_up_regular_ofc_pineapple",
        "phase": "hu_turn1_5card",
        "teacher_value_status": "diagnostic_not_match_EV",
        "root_id": f"m4-t1-second-{root_index:08d}",
        "root_index": root_index,
        "hand_seed": hand_seed,
        "split": split,
        "seat": "second",
        "to_act_order": "second",
        "street": "T1",
        "policy_observation": observation_payload,
        "observation_fingerprint": observation.fingerprint(),
        "board": board_to_json(observation.hero_board),
        "opponent_board": board_to_json(observation.opponent_public_board),
        "dealt": list(observation.dealt_cards),
        "dead_cards": list(observation.legacy_dead_cards()),
        "hero_private_discards": list(observation.hero_private_discards),
        "best_action": 0,
        "score_gap": best_score - second_score,
        "baseline_action_key": baseline_token,
        "baseline_action_row_index": baseline_row_index,
        "baseline_teacher_score": baseline_score,
        "paired_delta_contract": {
            "schema": M4_PAIRED_DELTA_SUMMARY_SCHEMA,
            "baseline_action_key": baseline_token,
            "common_evaluation_futures": True,
            "evaluation_samples": evaluation_samples,
        },
        "selected_action_key": result["selected_action_key"],
        "selected_action_evaluation_score": result[
            "selected_action_evaluation_score"
        ],
        "selected_action_evaluation_standard_error": result[
            "selected_action_evaluation_standard_error"
        ],
        "evaluation_sample_regret_of_locked_selection": result[
            "evaluation_sample_regret_of_locked_selection"
        ],
        "action_key_schema": result["action_key_schema"],
        "legal_action_count": result["legal_action_count"],
        "legal_action_set_digest": result["legal_action_set_digest"],
        "legal_action_order_digest": result["legal_action_order_digest"],
        "candidate_belief_digest": result["candidate_belief_digest"],
        "evaluation_belief_digest": result["evaluation_belief_digest"],
        "candidate_rng_key_digests": result["candidate_rng_key_digests"],
        "evaluation_rng_key_digests": result["evaluation_rng_key_digests"],
        "sample_independence": result["sample_independence"],
        "root_selection_lock": result["root_selection_lock"],
        "continuation_policy": result["continuation_policy"],
        "search_config": result["search_config"],
        "provenance": dict(provenance),
        "actions": converted,
    }


def _validated_paired_delta_summary(
    value: Any, *, expected_count: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("M4 teacher action is missing paired-delta summary")
    if value.get("schema") != M4_PAIRED_DELTA_SUMMARY_SCHEMA:
        raise ValueError("M4 teacher paired-delta summary schema mismatch")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
        raise ValueError("M4 teacher paired-delta count mismatch")
    numeric_keys = (
        "mean",
        "standard_error",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "max",
        "lt0_rate",
        "le_neg6_rate",
        "le_neg12_rate",
        "le_neg20_rate",
    )
    normalized: dict[str, Any] = {
        "schema": M4_PAIRED_DELTA_SUMMARY_SCHEMA,
        "count": count,
    }
    for key in numeric_keys:
        number = float(value.get(key, float("nan")))
        if not math.isfinite(number):
            raise ValueError(f"M4 teacher paired-delta {key} is non-finite")
        normalized[key] = number
    if normalized["standard_error"] < 0.0 or normalized["std"] < 0.0:
        raise ValueError("M4 teacher paired-delta dispersion is negative")
    for key in ("lt0_rate", "le_neg6_rate", "le_neg12_rate", "le_neg20_rate"):
        if not 0.0 <= normalized[key] <= 1.0:
            raise ValueError(f"M4 teacher paired-delta {key} is outside [0, 1]")
    ordered = [
        normalized[key]
        for key in (
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
        )
    ]
    if ordered != sorted(ordered):
        raise ValueError("M4 teacher paired-delta quantiles are not monotone")
    return normalized


def run_m4_t1_shard(
    config: M4T1DataConfig,
    *,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    paths: ModelPaths | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one resumable local/Spot-compatible shard."""

    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    checkpoint_path = Path(checkpoint)
    heartbeat_path = Path(heartbeat)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"M4 shard already complete: {output_path}")
    if overwrite:
        for path in (output_path, partial_path, checkpoint_path, heartbeat_path):
            if path.exists():
                path.unlink()

    model_paths = paths or ModelPaths()
    if config.batch_child_selectors:
        os.environ["OFC_HU_M3_BATCH_THREADS"] = str(config.native_batch_threads)
    root_profiles = config.effective_root_profiles()
    root_schedule = config.root_profile_schedule()
    explicit_profiles = {
        *root_profiles,
        config.baseline_profile,
        config.t2_profile,
    }
    bundle = load_model_bundle(model_paths, profiles=explicit_profiles)
    root_policy_population = {
        profile: {
            seat: build_policy(
                profile,
                bundle,
                seed=_profile_policy_seed(config.seed_start, profile, seat),
                seat=seat,
                opening_lookahead_samples=config.opening_lookahead_samples,
            )
            for seat in ("first", "second")
        }
        for profile in root_profiles
    }
    baseline_policies = {
        seat: build_policy(
            config.baseline_profile,
            bundle,
            seed=config.seed_start + 100 + (0 if seat == "first" else 1),
            seat=seat,
            opening_lookahead_samples=config.opening_lookahead_samples,
        )
        for seat in ("first", "second")
    }
    t2_policies = {
        seat: build_policy(
            config.t2_profile,
            bundle,
            seed=config.child_policy_seed + (0 if seat == "first" else 1),
            seat=seat,
            opening_lookahead_samples=config.opening_lookahead_samples,
        )
        for seat in ("first", "second")
    }

    config_sha256 = _config_sha256(config)
    completed = _resume_completed_roots(
        partial_path,
        checkpoint_path,
        config=config,
        config_sha256=config_sha256,
    )
    if completed > config.roots:
        raise ValueError("partial shard contains more roots than configured")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings: list[float] = []
    mode = "a" if partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for root_index in range(completed, config.roots):
            root_started = time.perf_counter()
            hand_seed = config.hand_seed(root_index)
            selected_root_profile = root_schedule[root_index]
            observation = generate_t1_second_root(
                hand_seed,
                root_policies=root_policy_population[selected_root_profile],
            )
            baseline_action = _choose_from_observation(
                baseline_policies["second"],
                observation,
                hand_id=hand_seed,
                game_id=hand_seed,
                decision_seed=_hand_decision_seed(
                    base_seed=hand_seed, observation=observation
                ),
            )
            root_run_id = (
                f"{config.run_id}:split={config.split}:root={root_index}:"
                f"seed={hand_seed}:obs={observation.fingerprint()}"
            )
            teacher_config = M4T1TeacherConfig(
                candidate_samples=config.candidate_samples,
                evaluation_samples=config.evaluation_samples,
                candidate_seed=config.candidate_seed,
                evaluation_seed=config.evaluation_seed,
                run_id=root_run_id,
                t2_policy_id=config.t2_profile,
                child_policy_seed=config.child_policy_seed,
                batch_child_selectors=config.batch_child_selectors,
            )
            result = evaluate_t1_second_actions(
                observation,
                t2_policies=t2_policies,
                baseline_action=baseline_action,
                config=teacher_config,
            )
            sample = teacher_result_to_sample(
                observation,
                result,
                hand_seed=hand_seed,
                root_index=root_index,
                split=config.split,
                baseline_action=baseline_action,
                provenance={
                    "schema": M4_T1_SHARD_SCHEMA,
                    "root_generation_policy": ROOT_GENERATION_POLICY,
                    "root_population_schedule": ROOT_POPULATION_SCHEDULE,
                    "root_population_schedule_sha256": _root_population_manifest(
                        config
                    )["schedule_sha256"],
                    "root_profile": selected_root_profile,
                    "root_policy_family": _root_policy_family(selected_root_profile),
                    "baseline_profile": config.baseline_profile,
                    "t2_profile": config.t2_profile,
                    "native_batch_threads": config.native_batch_threads,
                    "current_profile_resolved": False,
                },
            )
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            timings.append(time.perf_counter() - root_started)
            completed = root_index + 1
            common = {
                "config_sha256": config_sha256,
                "completed_roots": completed,
                "target_roots": config.roots,
                "last_root_index": root_index,
                "last_hand_seed": hand_seed,
                "partial_sha256": _sha256(partial_path),
                "root_population_manifest": _root_population_manifest(
                    config, completed_roots=completed
                ),
                "updated_unix_seconds": time.time(),
            }
            _atomic_json(
                checkpoint_path,
                {"schema": M4_T1_CHECKPOINT_SCHEMA, **common},
            )
            _atomic_json(
                heartbeat_path,
                {"schema": M4_T1_HEARTBEAT_SCHEMA, **common},
            )

    os.replace(partial_path, output_path)
    digest = _sha256(output_path)
    summary = {
        "schema": M4_T1_SHARD_SCHEMA,
        "status": "complete",
        "config": asdict(config),
        "config_sha256": config_sha256,
        "output": str(output_path),
        "output_sha256": digest,
        "roots": completed,
        "seconds": time.perf_counter() - started,
        "seconds_per_new_root": sum(timings) / max(1, len(timings)),
        "root_population_manifest": _root_population_manifest(
            config, completed_roots=completed
        ),
        "current_profile_resolved": False,
        "teacher_value_status": "diagnostic_not_match_EV",
    }
    _atomic_json(
        checkpoint_path,
        {"schema": M4_T1_CHECKPOINT_SCHEMA, **summary},
    )
    _atomic_json(
        heartbeat_path,
        {"schema": M4_T1_HEARTBEAT_SCHEMA, **summary},
    )
    return summary


def _legal_actions(observation: ActorObservation) -> list[Action]:
    from .action_space import generate_turn_actions

    return generate_turn_actions(observation.hero_board, observation.dealt_cards)


def _config_sha256(config: M4T1DataConfig) -> str:
    encoded = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_completed_roots(
    partial_path: Path,
    checkpoint_path: Path,
    *,
    config: M4T1DataConfig,
    config_sha256: str,
) -> int:
    """Validate a checkpoint boundary and discard only uncheckpointed bytes."""

    if not partial_path.exists():
        if checkpoint_path.exists():
            raise ValueError("checkpoint exists without its partial shard")
        return 0
    raw = partial_path.read_bytes()
    if not raw:
        if checkpoint_path.exists():
            raise ValueError("checkpoint exists for an empty partial shard")
        return 0
    if not checkpoint_path.exists():
        raise ValueError("non-empty partial shard has no checkpoint")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
    if checkpoint.get("schema") != M4_T1_CHECKPOINT_SCHEMA:
        raise ValueError("partial checkpoint schema mismatch")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("partial checkpoint configuration mismatch")
    completed = checkpoint.get("completed_roots")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ValueError("partial checkpoint completed_roots is invalid")
    if completed > config.roots:
        raise ValueError("partial checkpoint exceeds configured roots")

    newline_offsets = [index + 1 for index, byte in enumerate(raw) if byte == 10]
    if len(newline_offsets) < completed:
        raise ValueError("partial shard has fewer complete rows than checkpoint")
    checkpoint_boundary = newline_offsets[completed - 1] if completed else 0
    # Bytes after the checkpoint may be a complete-but-uncheckpointed row or a
    # torn final write. Both are deterministically regenerated.
    if len(raw) != checkpoint_boundary:
        with partial_path.open("r+b") as handle:
            handle.truncate(checkpoint_boundary)
        raw = raw[:checkpoint_boundary]
    expected_partial_sha = checkpoint.get("partial_sha256")
    actual_partial_sha = hashlib.sha256(raw).hexdigest()
    if expected_partial_sha != actual_partial_sha:
        raise ValueError("partial shard hash disagrees with checkpoint")

    for expected_index, line in enumerate(raw.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("checkpointed partial row is invalid JSON") from exc
        _validate_resumed_row(row, expected_index=expected_index, config=config)
    return completed


def _validate_resumed_row(
    row: Mapping[str, Any], *, expected_index: int, config: M4T1DataConfig
) -> None:
    if row.get("schema") != M4_T1_DATA_SCHEMA:
        raise ValueError("resumed row schema mismatch")
    if row.get("split") != config.split:
        raise ValueError("resumed row split mismatch")
    if row.get("root_index") != expected_index:
        raise ValueError("resumed row root_index is not contiguous")
    if row.get("hand_seed") != config.hand_seed(expected_index):
        raise ValueError("resumed row hand_seed disagrees with config")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("resumed row provenance missing")
    expected_root_profile = config.root_profile_for(expected_index)
    expected_profiles = {
        "root_profile": expected_root_profile,
        "root_policy_family": _root_policy_family(expected_root_profile),
        "root_population_schedule": ROOT_POPULATION_SCHEDULE,
        "root_population_schedule_sha256": _root_population_manifest(config)[
            "schedule_sha256"
        ],
        "baseline_profile": config.baseline_profile,
        "t2_profile": config.t2_profile,
        "native_batch_threads": config.native_batch_threads,
        "current_profile_resolved": False,
    }
    if any(provenance.get(key) != value for key, value in expected_profiles.items()):
        raise ValueError("resumed row policy provenance disagrees with config")
    search_config = row.get("search_config")
    if not isinstance(search_config, Mapping):
        raise ValueError("resumed row search_config missing")
    expected_search = {
        "candidate_samples": config.candidate_samples,
        "evaluation_samples": config.evaluation_samples,
        "candidate_seed": config.candidate_seed,
        "evaluation_seed": config.evaluation_seed,
        "child_policy_seed": config.child_policy_seed,
        "batch_child_selectors": config.batch_child_selectors,
    }
    if any(search_config.get(key) != value for key, value in expected_search.items()):
        raise ValueError("resumed row search configuration disagrees with config")
    expected_run_id = (
        f"{config.run_id}:split={config.split}:root={expected_index}:"
        f"seed={config.hand_seed(expected_index)}:obs={row.get('observation_fingerprint')}"
    )
    if search_config.get("run_id") != expected_run_id:
        raise ValueError("resumed row run_id disagrees with config")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2026071401)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--candidate-samples", type=int, default=1)
    parser.add_argument("--evaluation-samples", type=int, default=1)
    parser.add_argument("--candidate-seed", type=int, default=2026071401)
    parser.add_argument("--evaluation-seed", type=int, default=2026071402)
    parser.add_argument("--child-policy-seed", type=int, default=2026071403)
    parser.add_argument("--root-profile", default="stage3_baseline")
    parser.add_argument(
        "--root-profiles",
        nargs="+",
        help=(
            "Explicit root-policy population. Values may be space- or "
            "comma-separated; never include current."
        ),
    )
    parser.add_argument(
        "--root-profile-weights",
        "--root-weights",
        dest="root_profile_weights",
        nargs="+",
        help="Positive weights corresponding one-to-one with --root-profiles.",
    )
    parser.add_argument("--baseline-profile", default="stage18_p1")
    parser.add_argument("--t2-profile", default="stage3_baseline")
    parser.add_argument("--batch-child-selectors", action="store_true")
    parser.add_argument("--native-batch-threads", type=int, default=4)
    parser.add_argument("--opening-lookahead-samples", type=int, default=0)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--run-id", default="hu-m4-t1-second")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _split_cli_values(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root_profiles = _split_cli_values(args.root_profiles)
    raw_weights = _split_cli_values(args.root_profile_weights)
    try:
        root_profile_weights = tuple(float(value) for value in raw_weights)
    except ValueError as exc:
        raise SystemExit("--root-profile-weights must contain numbers") from exc
    config = M4T1DataConfig(
        roots=args.roots,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        candidate_samples=args.candidate_samples,
        evaluation_samples=args.evaluation_samples,
        child_policy_seed=args.child_policy_seed,
        candidate_seed=args.candidate_seed,
        evaluation_seed=args.evaluation_seed,
        root_profile=args.root_profile,
        root_profiles=root_profiles,
        root_profile_weights=root_profile_weights,
        baseline_profile=args.baseline_profile,
        t2_profile=args.t2_profile,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
        opening_lookahead_samples=args.opening_lookahead_samples,
        split=args.split,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            run_m4_t1_shard(
                config,
                output=args.output,
                checkpoint=args.checkpoint,
                heartbeat=args.heartbeat,
                overwrite=args.overwrite,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "M4T1DataConfig",
    "generate_t1_second_root",
    "run_m4_t1_shard",
    "teacher_result_to_sample",
]
