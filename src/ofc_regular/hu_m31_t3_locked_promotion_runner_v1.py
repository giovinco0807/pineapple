"""Real-hand locked population/ABR runner for the M3.1 T3 candidate.

The runner is cloud-neutral and performs no training or profile registration.
It consumes the evaluation-only StreetPolicyNetV1 runtime, an explicit
``stage7_m5_r10`` baseline factory, the frozen five-opponent population, and
three separately frozen ABR policy artifacts.

Candidate and baseline traces reuse the same shuffled hand, physical seat,
actor/opponent policy seeds, and decision RNG namespace.  A non-fire is valid
only when the semantic T3 action, complete gameplay trajectory, and terminal
score cancel exactly.  Output rows are validated by
``hu_m31_t3_step6d_locked_promotion_v1`` before an immutable shard can be
written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from .hu_m31_t3_promotion_runtime_closure_v1 import ABR_FACTORY_IDS
from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_actions
from .cards import create_deck
from .evaluate_hu_m4_population import gameplay_digest
from .evaluate_matchups import board_score_to_json, board_to_json
from .hu_infoset import ActorObservation, WorldState
from .hu_m31_t3_street_policy_runtime_v1 import (
    BASELINE_PROFILE,
    EVALUATION_ONLY_CANDIDATE,
    EVALUATION_ONLY_WATERMARK,
    HU_M31_T3_DECISION_SCHEMA,
    HU_M31_T3_EVALUATION_RECEIPT_SCHEMA,
    HU_M31_T3_RUNTIME_SCHEMA,
    RUNTIME_SCOPE_EVALUATION_ONLY,
)
from .play_ai import _choose_from_observation, _hand_decision_seed
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score


LOCKED_PROMOTION_RUNNER_SCHEMA = "hu_m31_t3_locked_promotion_runner_v1"
ABR_POLICY_MANIFEST_SCHEMA = "hu_m31_t3_abr_policy_manifest_v1"
ABR_POLICY_BINDING_SCHEMA = "hu_m31_t3_abr_policy_binding_v1"
COUNTERFACTUAL_TRACE_SCHEMA = "hu_m31_t3_real_hand_trace_v1"

PolicyFactory = Callable[..., object]
_SHA_CHARS = frozenset("0123456789abcdef")
_POPULATION_IDS = tuple(
    str(row["opponent_id"]) for row in promotion.OPPONENT_DESCRIPTORS
)
_ABR_IDS = tuple(
    str(row["response_id"]) for row in promotion.ABR_DESCRIPTORS
)
_ABR_BY_ID = {
    str(row["response_id"]): dict(row)
    for row in promotion.ABR_DESCRIPTORS
}
_ABR_MANIFEST_FIELDS = frozenset(
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


class LockedPromotionRunnerError(RuntimeError):
    """Raised when a runtime, trace, or artifact binding changes."""


@dataclass(frozen=True)
class AbrPolicyBinding:
    """Callable policy plus independently pinned frozen ABR artifacts."""

    response_id: str
    policy_factory: PolicyFactory
    manifest_path: Path
    expected_manifest_file_sha256: str
    checkpoint_path: Path
    expected_checkpoint_file_sha256: str


@dataclass(frozen=True)
class _ValidatedAbrBinding:
    response_id: str
    policy_factory: PolicyFactory
    manifest: Mapping[str, Any]

    def make_policy(self, *, policy_seed: int, seat: str) -> object:
        policy = _make_policy(
            self.policy_factory,
            policy_seed=policy_seed,
            seat=seat,
        )
        if (
            getattr(policy, "abr_response_id", None) != self.response_id
            or getattr(policy, "abr_policy_factory_id", None)
            != ABR_FACTORY_IDS[self.response_id]
            or getattr(policy, "abr_policy_artifact_sha256", None)
            != self.manifest["policy_checkpoint_sha256"]
            or getattr(policy, "opponent_private_discards_used", None)
            is not False
            or getattr(policy, "current_profile_resolved", None) is not False
        ):
            raise LockedPromotionRunnerError(
                "ABR policy instance is not bound to its frozen factory/checkpoint"
            )
        return _BoundAbrPolicy(
            policy,
            response_id=self.response_id,
            checkpoint_sha256=str(
                self.manifest["policy_checkpoint_sha256"]
            ),
        )


class _BoundAbrPolicy:
    """Metadata-only wrapper; gameplay remains owned by the frozen policy."""

    def __init__(
        self,
        policy: object,
        *,
        response_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self._policy = policy
        self.abr_response_id = response_id
        self.abr_policy_artifact_sha256 = checkpoint_sha256
        self.opponent_private_discards_used = False
        self.current_profile_resolved = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        return _choose_from_observation(
            self._policy,
            observation,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
        )


@dataclass(frozen=True)
class _HandTrace:
    score: float
    trajectory_sha256: str
    t3_action_key: str
    t3_observation_fingerprint: str
    t3_legal_action_set_sha256: str
    t3_canonical_mapping_sha256: str
    t3_decision: Mapping[str, Any] | None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_object(
    path: Path, label: str
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise LockedPromotionRunnerError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedPromotionRunnerError(
            f"{label} is not canonical JSON"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise LockedPromotionRunnerError(
            f"{label} is not a canonical object"
        )
    return value, raw


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    raw = promotion.canonical_bytes(value)
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
                f"immutable locked-promotion shard changed: {destination}"
            ) from None


def validate_abr_policy_binding(
    binding: AbrPolicyBinding,
    *,
    plan: Mapping[str, Any],
) -> _ValidatedAbrBinding:
    validated_plan = promotion.validate_locked_promotion_plan(plan)
    if binding.response_id not in _ABR_BY_ID:
        raise ValueError("ABR response id is outside the frozen family grid")
    if not callable(binding.policy_factory):
        raise TypeError("ABR policy factory must be callable")
    if (
        not _is_sha256(binding.expected_manifest_file_sha256)
        or not _is_sha256(binding.expected_checkpoint_file_sha256)
    ):
        raise ValueError("ABR manifest/checkpoint SHA-256 must be pinned")
    manifest, manifest_raw = _read_canonical_object(
        Path(binding.manifest_path), "ABR policy manifest"
    )
    checkpoint = Path(binding.checkpoint_path)
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.suffix.casefold() != ".zip"
    ):
        raise LockedPromotionRunnerError(
            "ABR policy checkpoint is missing, unsafe, or not a checkpoint zip"
        )
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != binding.expected_manifest_file_sha256
        or _sha256_file(checkpoint)
        != binding.expected_checkpoint_file_sha256
    ):
        raise LockedPromotionRunnerError(
            "ABR policy manifest/checkpoint pinned hash changed"
        )
    expected_factory_id = ABR_FACTORY_IDS[binding.response_id]
    if (
        getattr(binding.policy_factory, "factory_id", None)
        != expected_factory_id
        or getattr(binding.policy_factory, "response_id", None)
        != binding.response_id
        or getattr(binding.policy_factory, "checkpoint_sha256", None)
        != binding.expected_checkpoint_file_sha256
        or Path(
            getattr(binding.policy_factory, "checkpoint_path", "")
        ).resolve()
        != checkpoint.resolve()
    ):
        raise LockedPromotionRunnerError(
            "ABR callable is not the pinned checkpoint-aware factory"
        )
    if set(manifest) != _ABR_MANIFEST_FIELDS:
        raise ValueError("ABR policy manifest field set changed")
    identity = dict(manifest)
    declared_identity = identity.pop("manifest_identity_sha256")
    descriptor = _ABR_BY_ID[binding.response_id]
    if (
        manifest["schema"] != ABR_POLICY_MANIFEST_SCHEMA
        or manifest["status"] != "frozen_independent_abr_policy"
        or manifest["response_id"] != binding.response_id
        or manifest["family"] != descriptor["family"]
        or manifest["objective"] != descriptor["objective"]
        or manifest["candidate_plan_sha256"]
        != promotion.canonical_sha256(validated_plan)
        or manifest["policy_checkpoint_filename"] != checkpoint.name
        or manifest["policy_checkpoint_sha256"]
        != binding.expected_checkpoint_file_sha256
        or manifest["policy_checkpoint_bytes"] != checkpoint.stat().st_size
        or manifest["policy_checkpoint_format"]
        != "street_policy_net_v1_checkpoint_zip"
        or manifest["policy_factory_id"] != expected_factory_id
        or manifest["development_schedule"] != "abr_development"
        or manifest["locked_evaluation_schedule"]
        != promotion.LOCKED_ABR
        or manifest["locked_seed_training_allowed"] is not False
        or manifest["opponent_private_discards_used"] is not False
        or manifest["current_profile_resolved"] is not False
        or manifest["frozen_before_locked_evaluation"] is not True
        or not _is_sha256(declared_identity)
        or declared_identity != _canonical_sha256(identity)
    ):
        raise ValueError("ABR policy manifest boundary changed")
    return _ValidatedAbrBinding(
        response_id=binding.response_id,
        policy_factory=binding.policy_factory,
        manifest=manifest,
    )


def _make_policy(
    factory: PolicyFactory,
    *,
    policy_seed: int,
    seat: str,
) -> object:
    if not callable(factory):
        raise TypeError("policy factory must be callable")
    policy = factory(policy_seed=policy_seed, seat=seat)
    if getattr(policy, "seat", None) != seat:
        raise LockedPromotionRunnerError(
            "policy factory returned another physical seat"
        )
    if not callable(getattr(policy, "choose_action_observation", None)):
        raise TypeError(
            "locked promotion policies require ActorObservation support"
        )
    return policy


def _runtime_receipt(runtime: object) -> Mapping[str, Any]:
    receipt = getattr(runtime, "authorization_receipt", None)
    if receipt is None:
        raise LockedPromotionRunnerError(
            "candidate lacks an evaluation-only authorization receipt"
        )
    to_dict = getattr(receipt, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        raise LockedPromotionRunnerError(
            "candidate authorization receipt is not serializable"
        )
    if not isinstance(value, Mapping):
        raise LockedPromotionRunnerError(
            "candidate authorization receipt changed"
        )
    return value


def _validate_evaluation_runtime(
    runtime: object,
    *,
    plan: Mapping[str, Any],
    seat: str,
) -> None:
    validated_plan = promotion.validate_locked_promotion_plan(plan)
    receipt = _runtime_receipt(runtime)
    binding = validated_plan["artifact_binding"]
    if (
        getattr(runtime, "runtime_id", None) != HU_M31_T3_RUNTIME_SCHEMA
        or getattr(runtime, "runtime_scope", None)
        != RUNTIME_SCOPE_EVALUATION_ONLY
        or getattr(runtime, "evaluation_only", None) is not True
        or getattr(runtime, "profile_candidate", None)
        != EVALUATION_ONLY_CANDIDATE
        or getattr(runtime, "seat", None) != seat
        or receipt.get("schema") != HU_M31_T3_EVALUATION_RECEIPT_SCHEMA
        or receipt.get("runtime_scope")
        != RUNTIME_SCOPE_EVALUATION_ONLY
        or receipt.get("candidate_id") != EVALUATION_ONLY_CANDIDATE
        or receipt.get("audit_watermark") != EVALUATION_ONLY_WATERMARK
        or receipt.get("plan_sha256")
        != promotion.canonical_sha256(validated_plan)
        or receipt.get("model_manifest_file_sha256")
        != binding["model"]["sha256"]
        or receipt.get("compatibility_threshold_lock_file_sha256")
        != binding["threshold_lock"]["sha256"]
        or receipt.get("policy_registry_file_sha256")
        != binding["policy_registry"]["sha256"]
        or receipt.get("evaluation_runtime_closure_file_sha256")
        != binding["evaluation_runtime_closure"]["sha256"]
        or receipt.get("plan_and_artifacts_source_replayed") is not True
        or receipt.get("evaluation_only") is not True
        or receipt.get("scientific_promotion_passed") is not False
        or receipt.get(
            "separate_opt_in_profile_candidate_authorized"
        )
        is not False
        or receipt.get("named_profile_added") is not False
        or receipt.get("current_profile_changed") is not False
        or receipt.get("runtime_activated") is not False
        or receipt.get("full_replacement_enabled") is not False
    ):
        raise LockedPromotionRunnerError(
            "candidate is not the plan-bound evaluation-only runtime"
        )


def _decision_payload(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise LockedPromotionRunnerError(
            "T3 runtime decision is not serializable"
        )
    if not isinstance(payload, dict):
        raise LockedPromotionRunnerError(
            "T3 runtime decision payload changed"
        )
    return payload


def _validate_t3_decision(
    decision: Mapping[str, Any],
    *,
    observation: ActorObservation,
    action: Action,
    legal_actions: Sequence[Action],
) -> None:
    canonical = canonicalize_actions(legal_actions)
    final_key = action_key(action).to_token()
    baseline_key = decision.get("baseline_action_key")
    candidate_key = decision.get("candidate_action_key")
    fired = decision.get("override_fired")
    if (
        decision.get("schema") != HU_M31_T3_DECISION_SCHEMA
        or decision.get("runtime_schema") != HU_M31_T3_RUNTIME_SCHEMA
        or decision.get("runtime_scope")
        != RUNTIME_SCOPE_EVALUATION_ONLY
        or decision.get("evaluation_only") is not True
        or decision.get("audit_watermark") != EVALUATION_ONLY_WATERMARK
        or decision.get("profile_candidate")
        != EVALUATION_ONLY_CANDIDATE
        or decision.get("baseline_profile") != BASELINE_PROFILE
        or decision.get("observation_fingerprint")
        != observation.fingerprint()
        or decision.get("seat") != observation.seat
        or decision.get("action_key_schema") != ACTION_KEY_SCHEMA
        or decision.get("legal_action_count") != len(canonical)
        or decision.get("legal_action_set_sha256")
        != legal_action_set_digest(canonical)
        or decision.get("canonical_action_mapping_sha256")
        != ordered_action_mapping_digest(canonical)
        or decision.get("final_action_key") != final_key
        or not isinstance(fired, bool)
        or decision.get("promotion_gate_file_sha256") is not None
        or decision.get("teacher_values_used") is not False
        or decision.get("opponent_private_discards_used") is not False
        or decision.get("current_profile_resolved") is not False
        or decision.get("runtime_activated") is not False
    ):
        raise LockedPromotionRunnerError(
            "evaluation-only T3 decision audit changed"
        )
    if fired:
        if final_key != candidate_key or final_key == baseline_key:
            raise LockedPromotionRunnerError(
                "fired T3 decision ActionKey relation changed"
            )
    elif final_key != baseline_key:
        raise LockedPromotionRunnerError(
            "non-fire T3 decision did not return its baseline ActionKey"
        )


def _play_real_hand(
    *,
    hand_seed: int,
    evaluation_seed: int,
    hero_seat: str,
    hero_policy: object,
    opponent_policy: object,
    capture_candidate_t3: bool,
    hero_label: str,
    opponent_label: str,
) -> _HandTrace:
    if hero_seat not in promotion.SEATS:
        raise ValueError("hero seat must be first or second")
    hero_player = 0 if hero_seat == "first" else 1
    opponent_player = 1 - hero_player
    policies = [None, None]
    labels = ["", ""]
    policies[hero_player] = hero_policy
    policies[opponent_player] = opponent_policy
    labels[hero_player] = hero_label
    labels[opponent_player] = opponent_label

    deck_rng = random.Random(hand_seed)
    deck = create_deck(shuffle=True, rng=deck_rng)
    cursor = 0
    boards = [Board(), Board()]
    private_discards: list[list[str]] = [[], []]
    turns: list[dict[str, Any]] = []
    hero_t3_key: str | None = None
    hero_t3_fingerprint: str | None = None
    hero_t3_set_digest: str | None = None
    hero_t3_mapping_digest: str | None = None
    t3_decision: dict[str, Any] | None = None

    def act(player: int, dealt: Sequence[str], street: str) -> None:
        nonlocal hero_t3_key
        nonlocal hero_t3_fingerprint
        nonlocal hero_t3_set_digest
        nonlocal hero_t3_mapping_digest
        nonlocal t3_decision
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
        legal = generate_actions(
            observation.hero_board, observation.dealt_cards
        )
        if not legal:
            raise LockedPromotionRunnerError(
                "real-hand trace has no legal action"
            )
        decision_seed = _hand_decision_seed(
            base_seed=evaluation_seed,
            observation=observation,
        )
        policy = policies[player]
        assert policy is not None
        if (
            capture_candidate_t3
            and player == hero_player
            and street == "T3"
        ):
            chooser = getattr(
                policy, "choose_action_observation_with_audit", None
            )
            if not callable(chooser):
                raise LockedPromotionRunnerError(
                    "candidate runtime lacks the audited T3 chooser"
                )
            action, raw_decision = chooser(
                observation,
                hand_id=hand_seed,
                game_id=hand_seed,
                decision_seed=decision_seed,
            )
            if raw_decision is None:
                raise LockedPromotionRunnerError(
                    "candidate runtime omitted its T3 decision"
                )
            t3_decision = _decision_payload(raw_decision)
            _validate_t3_decision(
                t3_decision,
                observation=observation,
                action=action,
                legal_actions=legal,
            )
        else:
            action = _choose_from_observation(
                policy,
                observation,
                hand_id=hand_seed,
                game_id=hand_seed,
                decision_seed=decision_seed,
            )
        action_token = action_key(action).to_token()
        legal_by_key = {
            action_key(value).to_token(): value for value in legal
        }
        if action_token not in legal_by_key:
            raise LockedPromotionRunnerError(
                "policy returned an illegal real-hand ActionKey"
            )
        if player == hero_player and street == "T3":
            canonical = canonicalize_actions(legal)
            hero_t3_key = action_token
            hero_t3_fingerprint = observation.fingerprint()
            hero_t3_set_digest = legal_action_set_digest(canonical)
            hero_t3_mapping_digest = ordered_action_mapping_digest(
                canonical
            )
        boards[player] = boards[player].place(action.placements)
        private_discards[player].extend(action.discards)
        turns.append(
            {
                "turn": street,
                "player": player,
                "profile": labels[player],
                "dealt": list(dealt),
                "placements": [
                    list(placement) for placement in action.placements
                ],
                "discards": list(action.discards),
                "action_key_schema": ACTION_KEY_SCHEMA,
                "action_key": action_token,
                "board": board_to_json(boards[player]),
            }
        )

    for player in (0, 1):
        dealt = deck[cursor : cursor + 5]
        cursor += 5
        act(player, dealt, "T0")
    for round_index in range(1, 5):
        for player in (0, 1):
            dealt = deck[cursor : cursor + 3]
            cursor += 3
            act(player, dealt, f"T{round_index}")
    if cursor != 34:
        raise AssertionError("regular HU trace consumed another deck geometry")
    if any(board.card_count() != 13 for board in boards):
        raise LockedPromotionRunnerError(
            "real-hand trace did not complete both boards"
        )
    score_p0, score0 = terminal_score(
        boards[0], boards[1], fl_ev=DEFAULT_FL_EV
    )
    _reverse, score1 = terminal_score(
        boards[1], boards[0], fl_ev=DEFAULT_FL_EV
    )
    hero_score = float(score_p0 if hero_player == 0 else -score_p0)
    if not math.isfinite(hero_score):
        raise LockedPromotionRunnerError(
            "real-hand trace returned a non-finite HU score"
        )
    if (
        hero_t3_key is None
        or hero_t3_fingerprint is None
        or hero_t3_set_digest is None
        or hero_t3_mapping_digest is None
        or (capture_candidate_t3 and t3_decision is None)
        or (not capture_candidate_t3 and t3_decision is not None)
    ):
        raise LockedPromotionRunnerError(
            "real-hand trace T3 audit coverage changed"
        )
    hand = {
        "schema": COUNTERFACTUAL_TRACE_SCHEMA,
        "seed": hand_seed,
        "profiles": {"p0": labels[0], "p1": labels[1]},
        "score_p0": float(score_p0),
        "final": {
            "p0": board_to_json(boards[0]),
            "p1": board_to_json(boards[1]),
        },
        "board_scores": {
            "p0": board_score_to_json(score0),
            "p1": board_score_to_json(score1),
        },
        "turns": turns,
    }
    return _HandTrace(
        score=hero_score,
        trajectory_sha256=gameplay_digest(hand),
        t3_action_key=hero_t3_key,
        t3_observation_fingerprint=hero_t3_fingerprint,
        t3_legal_action_set_sha256=hero_t3_set_digest,
        t3_canonical_mapping_sha256=hero_t3_mapping_digest,
        t3_decision=t3_decision,
    )


class LockedPromotionRunner:
    """Generate source-replayable realized rows for one locked plan."""

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        candidate_policy_factory: PolicyFactory,
        baseline_policy_factory: PolicyFactory,
        opponent_policy_factories: Mapping[str, PolicyFactory],
        abr_policy_bindings: Mapping[str, AbrPolicyBinding],
    ) -> None:
        self.plan = promotion.validate_locked_promotion_plan(plan)
        if not callable(candidate_policy_factory):
            raise TypeError("candidate policy factory must be callable")
        if not callable(baseline_policy_factory):
            raise TypeError("baseline policy factory must be callable")
        if tuple(opponent_policy_factories) != _POPULATION_IDS:
            raise ValueError(
                "population factories must match the frozen five-opponent order"
            )
        if any(
            not callable(factory)
            for factory in opponent_policy_factories.values()
        ):
            raise TypeError("population policy factory must be callable")
        if tuple(abr_policy_bindings) != _ABR_IDS:
            raise ValueError(
                "ABR bindings must match the frozen three-family order"
            )
        self.candidate_policy_factory = candidate_policy_factory
        self.baseline_policy_factory = baseline_policy_factory
        self.opponent_policy_factories = dict(
            opponent_policy_factories
        )
        self.abr_policy_bindings = {
            response_id: validate_abr_policy_binding(
                binding, plan=self.plan
            )
            for response_id, binding in abr_policy_bindings.items()
        }

    def _opponent_pair(
        self,
        *,
        schedule: str,
        entity_id: str,
        policy_seed: int,
        seat: str,
    ) -> tuple[object, object]:
        if schedule == promotion.LOCKED_POPULATION:
            factory = self.opponent_policy_factories[entity_id]
            return (
                _make_policy(
                    factory, policy_seed=policy_seed, seat=seat
                ),
                _make_policy(
                    factory, policy_seed=policy_seed, seat=seat
                ),
            )
        binding = self.abr_policy_bindings[entity_id]
        return (
            binding.make_policy(
                policy_seed=policy_seed, seat=seat
            ),
            binding.make_policy(
                policy_seed=policy_seed, seat=seat
            ),
        )

    def generate_hand_row(
        self,
        *,
        schedule: str,
        entity_id: str,
        seed_index: int,
        seat: str,
    ) -> dict[str, Any]:
        allowed = (
            _POPULATION_IDS
            if schedule == promotion.LOCKED_POPULATION
            else _ABR_IDS
            if schedule == promotion.LOCKED_ABR
            else ()
        )
        if entity_id not in allowed:
            raise ValueError(
                "locked promotion entity is outside its frozen schedule"
            )
        if seat not in promotion.SEATS:
            raise ValueError("locked promotion seat changed")
        seeds = promotion.seed_values(schedule, seed_index)
        candidate = _make_policy(
            self.candidate_policy_factory,
            policy_seed=seeds["actor_policy"],
            seat=seat,
        )
        _validate_evaluation_runtime(
            candidate, plan=self.plan, seat=seat
        )
        baseline = _make_policy(
            self.baseline_policy_factory,
            policy_seed=seeds["actor_policy"],
            seat=seat,
        )
        opponent_seat = "second" if seat == "first" else "first"
        candidate_opponent, baseline_opponent = self._opponent_pair(
            schedule=schedule,
            entity_id=entity_id,
            policy_seed=seeds["opponent_policy"],
            seat=opponent_seat,
        )
        candidate_trace = _play_real_hand(
            hand_seed=seeds["hand"],
            evaluation_seed=seeds["evaluation"],
            hero_seat=seat,
            hero_policy=candidate,
            opponent_policy=candidate_opponent,
            capture_candidate_t3=True,
            hero_label=EVALUATION_ONLY_CANDIDATE,
            opponent_label=entity_id,
        )
        baseline_trace = _play_real_hand(
            hand_seed=seeds["hand"],
            evaluation_seed=seeds["evaluation"],
            hero_seat=seat,
            hero_policy=baseline,
            opponent_policy=baseline_opponent,
            capture_candidate_t3=False,
            hero_label=BASELINE_PROFILE,
            opponent_label=entity_id,
        )
        decision = candidate_trace.t3_decision
        assert decision is not None
        if (
            candidate_trace.t3_observation_fingerprint
            != baseline_trace.t3_observation_fingerprint
            or candidate_trace.t3_legal_action_set_sha256
            != baseline_trace.t3_legal_action_set_sha256
            or candidate_trace.t3_canonical_mapping_sha256
            != baseline_trace.t3_canonical_mapping_sha256
            or decision["baseline_action_key"]
            != baseline_trace.t3_action_key
            or decision["final_action_key"]
            != candidate_trace.t3_action_key
        ):
            raise LockedPromotionRunnerError(
                "candidate/baseline T3 ActionKey mapping did not replay"
            )
        fired = bool(decision["override_fired"])
        delta = candidate_trace.score - baseline_trace.score
        nonfire_action_identical = (
            candidate_trace.t3_action_key
            == baseline_trace.t3_action_key
            if not fired
            else None
        )
        nonfire_trajectory_identical = (
            candidate_trace.trajectory_sha256
            == baseline_trace.trajectory_sha256
            if not fired
            else None
        )
        nonfire_cancellation = (
            bool(
                nonfire_action_identical
                and nonfire_trajectory_identical
                and math.isclose(
                    delta, 0.0, rel_tol=0.0, abs_tol=1e-12
                )
            )
            if not fired
            else None
        )
        binding = self.plan["artifact_binding"]
        row = {
            "schema": promotion.ROW_SCHEMA,
            "schedule": schedule,
            "entity_id": entity_id,
            "seed_index": seed_index,
            "hand_seed": seeds["hand"],
            "actor_policy_seed": seeds["actor_policy"],
            "opponent_policy_seed": seeds["opponent_policy"],
            "evaluation_seed": seeds["evaluation"],
            "child_seed": seeds["child"],
            "confirmation_seed": seeds["confirmation"],
            "seat": seat,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "candidate_action_key": candidate_trace.t3_action_key,
            "baseline_action_key": baseline_trace.t3_action_key,
            "legal_action_mapping_sha256": (
                candidate_trace.t3_canonical_mapping_sha256
            ),
            "candidate_trajectory_sha256": (
                candidate_trace.trajectory_sha256
            ),
            "baseline_trajectory_sha256": (
                baseline_trace.trajectory_sha256
            ),
            "score_perspective": "candidate_hero_hu_score",
            "candidate_score": candidate_trace.score,
            "baseline_score": baseline_trace.score,
            "delta": delta,
            "override_log_valid": True,
            "override_fired": fired,
            "nonfire_action_key_identical": (
                nonfire_action_identical
            ),
            "nonfire_trajectory_identical": (
                nonfire_trajectory_identical
            ),
            "nonfire_cancellation_valid": nonfire_cancellation,
            "model_sha256": binding["model"]["sha256"],
            "threshold_lock_sha256": binding["threshold_lock"][
                "sha256"
            ],
            "policy_registry_sha256": binding["policy_registry"][
                "sha256"
            ],
            "evaluation_runtime_closure_sha256": binding[
                "evaluation_runtime_closure"
            ]["sha256"],
            "counterfactual_basis": (
                "same_hand_role_policy_seeds_physical_seat_v1"
            ),
            "teacher_values_used": False,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
            "current_profile_changed": False,
        }
        return promotion.validate_hand_row(row, plan=self.plan)

    def run_shard(
        self,
        *,
        schedule: str,
        entity_id: str,
        seed_indices: Sequence[int],
        shard_id: str,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        indices = tuple(seed_indices)
        if (
            not indices
            or indices != tuple(sorted(indices))
            or len(indices) != len(set(indices))
        ):
            raise ValueError(
                "locked promotion seed indices must be nonempty, unique, "
                "and sorted"
            )
        rows = [
            self.generate_hand_row(
                schedule=schedule,
                entity_id=entity_id,
                seed_index=index,
                seat=seat,
            )
            for index in indices
            for seat in promotion.SEATS
        ]
        rows.sort(key=promotion._row_key)
        shard = promotion.build_evaluation_shard(
            plan=self.plan,
            shard_id=shard_id,
            rows=rows,
        )
        if output_path is not None:
            _write_once(Path(output_path), shard)
        return shard


__all__ = [
    "ABR_POLICY_BINDING_SCHEMA",
    "ABR_POLICY_MANIFEST_SCHEMA",
    "COUNTERFACTUAL_TRACE_SCHEMA",
    "LOCKED_PROMOTION_RUNNER_SCHEMA",
    "AbrPolicyBinding",
    "LockedPromotionRunner",
    "LockedPromotionRunnerError",
    "validate_abr_policy_binding",
]
