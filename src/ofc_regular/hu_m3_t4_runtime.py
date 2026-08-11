"""Strict production-facing runtime for the M3.0 exact T4 solver.

The native engine already contains the T4 game tree.  This module is the
policy boundary: it accepts only :class:`ActorObservation`, fixes search to
exhaustive mode, validates every ActionKey/value returned by Rust, and maps the
selected key back onto Python's freshly enumerated legal actions.

T4-first values are exact under the declared uniform exchangeable restart
belief.  T4-second values are exact terminal heads-up scores.  Neither value is
reported as realized match EV or as a full-game Nash guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    action_key_from_payload,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .cards import ALL_CARDS
from .hu_infoset import ActorObservation
from .hu_late_street_teacher import T4SearchConfig
from .hu_m3_rust import (
    HU_M3_REQUEST_SCHEMA,
    engine_version,
    evaluate_batch,
    evaluate_t4,
    load_native_engine,
    native_library_path,
    t4_request,
)


HU_M30_T4_RUNTIME_SCHEMA = "hu_m30_t4_runtime_decision_v1"
HU_M30_T4_RUNTIME_ID = "hu_m30_t4_exact_both_seats_v1"
HU_M30_T4_BELIEF_ID = "uniform_exchangeable_hidden_cards_v1"
HU_M30_T4_RUN_ID = "hu-m30-t4-runtime-v1"
HU_M30_T4_ENGINE_VERSION = "ofc_hu_m3_engine/0.1.0"
_FIRST_SOLVER_ID = "rust_t4_exchangeable_expectimax_exact_response_v1"
_SECOND_SOLVER_ID = "t4_second_terminal_exhaustive_v1"
_EXPECTED_FIRST_FUTURES = 2024


class HuM3T4RuntimeError(RuntimeError):
    """Raised when native T4 execution or its result contract is invalid."""


@dataclass(frozen=True)
class HuM3T4RuntimeConfig:
    """Startup contract for one native T4 runtime instance."""

    expected_library_sha256: str
    library_path: Path | None = None
    expected_engine_version: str = HU_M30_T4_ENGINE_VERSION
    require_release_library: bool = True
    run_id: str = HU_M30_T4_RUN_ID

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("T4 runtime run_id must not be empty")
        if not self.expected_engine_version:
            raise ValueError("expected_engine_version must not be empty")
        digest = self.expected_library_sha256.casefold()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("expected_library_sha256 must be 64 lowercase hex digits")
        object.__setattr__(self, "expected_library_sha256", digest)
        if self.library_path is not None:
            object.__setattr__(self, "library_path", Path(self.library_path))


@dataclass(frozen=True)
class T4ExactActionValue:
    original_index: int
    rank: int
    action_key: str
    ev: float
    action: Action

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_index": self.original_index,
            "rank": self.rank,
            "action_key": self.action_key,
            "ev": self.ev,
            "placements": [list(item) for item in self.action.placements],
            "discards": list(self.action.discards),
        }


@dataclass(frozen=True)
class T4ExactDecision:
    action: Action
    selected_action_key: str
    selected_ev: float
    selection_gap: float
    action_values: tuple[T4ExactActionValue, ...]
    seat: str
    value_scope: str
    observation_fingerprint: str
    legal_action_set_digest: str
    legal_action_order_digest: str
    belief_id: str | None
    belief_digest: str
    scoring_digest: str
    solver_id: str
    engine_version: str
    native_library_sha256: str
    future_count: int
    native_latency_ms: float
    validation_latency_ms: float
    total_latency_ms: float
    execution_mode: str
    batch_size: int
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HU_M30_T4_RUNTIME_SCHEMA,
            "runtime_id": HU_M30_T4_RUNTIME_ID,
            "seat": self.seat,
            "value_scope": self.value_scope,
            "observation_fingerprint": self.observation_fingerprint,
            "selected_action_key": self.selected_action_key,
            "selected_ev": self.selected_ev,
            "selection_gap": self.selection_gap,
            "selected_action": {
                "placements": [list(item) for item in self.action.placements],
                "discards": list(self.action.discards),
            },
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": self.legal_action_set_digest,
            "legal_action_order_digest": self.legal_action_order_digest,
            "action_values": [row.to_dict() for row in self.action_values],
            "belief_id": self.belief_id,
            "belief_digest": self.belief_digest,
            "scoring_digest": self.scoring_digest,
            "solver_id": self.solver_id,
            "engine_version": self.engine_version,
            "native_library_sha256": self.native_library_sha256,
            "search_mode": "exhaustive",
            "future_count": self.future_count,
            "teacher_value_status": "diagnostic_not_match_EV",
            "native_latency_ms": self.native_latency_ms,
            "validation_latency_ms": self.validation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "execution_mode": self.execution_mode,
            "batch_size": self.batch_size,
            "result_digest": self.result_digest,
        }


class HuM3T4ExactSolver:
    """Eagerly validated, no-fallback native exact T4 solver."""

    def __init__(self, config: HuM3T4RuntimeConfig) -> None:
        self.config = config
        path = self.config.library_path or native_library_path(release=True)
        self.library_path = Path(path).resolve()
        if not self.library_path.is_file():
            raise HuM3T4RuntimeError(
                f"native T4 library does not exist: {self.library_path}"
            )
        if (
            self.config.require_release_library
            and "release" not in {part.casefold() for part in self.library_path.parts}
        ):
            raise HuM3T4RuntimeError(
                f"native T4 runtime requires a release library: {self.library_path}"
            )
        self.library_sha256 = _sha256_file(self.library_path)
        if self.library_sha256 != self.config.expected_library_sha256:
            raise HuM3T4RuntimeError(
                "native T4 library SHA-256 mismatch: "
                f"expected {self.config.expected_library_sha256}, got {self.library_sha256}"
            )
        self.library = load_native_engine(path=self.library_path, build_if_missing=False)
        self.engine_version = engine_version(library=self.library)
        if self.engine_version != self.config.expected_engine_version:
            raise HuM3T4RuntimeError(
                "native T4 engine version mismatch: "
                f"expected {self.config.expected_engine_version!r}, got {self.engine_version!r}"
            )
        self.search_config = T4SearchConfig(
            candidate_samples=0,
            evaluation_samples=0,
            seed=0,
            candidate_seed=0,
            evaluation_seed=0,
            run_id=self.config.run_id,
        )

    def solve(self, observation: ActorObservation) -> T4ExactDecision:
        _require_t4_observation(observation)
        started = time.perf_counter()
        native_started = time.perf_counter()
        result = evaluate_t4(
            observation,
            config=self.search_config,
            library=self.library,
        )
        native_ms = (time.perf_counter() - native_started) * 1000.0
        validation_started = time.perf_counter()
        decision = self._validate_result(
            observation,
            result,
            native_latency_ms=native_ms,
            total_started=started,
            validation_started=validation_started,
            execution_mode="scalar",
            batch_size=1,
        )
        return decision

    def solve_many(
        self, observations: Sequence[ActorObservation]
    ) -> list[T4ExactDecision]:
        if not observations:
            return []
        for observation in observations:
            _require_t4_observation(observation)
        started = time.perf_counter()
        native_started = time.perf_counter()
        results = evaluate_batch(
            [t4_request(observation, config=self.search_config) for observation in observations],
            library=self.library,
        )
        native_total_ms = (time.perf_counter() - native_started) * 1000.0
        if len(results) != len(observations):
            raise HuM3T4RuntimeError("native T4 batch result length mismatch")
        amortized_native_ms = native_total_ms / len(observations)
        decisions: list[T4ExactDecision] = []
        for observation, result in zip(observations, results, strict=True):
            validation_started = time.perf_counter()
            decisions.append(
                self._validate_result(
                    observation,
                    result,
                    native_latency_ms=amortized_native_ms,
                    total_started=started,
                    validation_started=validation_started,
                    execution_mode="batch_amortized",
                    batch_size=len(observations),
                )
            )
        return decisions

    def _validate_result(
        self,
        observation: ActorObservation,
        result: Mapping[str, Any],
        *,
        native_latency_ms: float,
        total_started: float,
        validation_started: float,
        execution_mode: str,
        batch_size: int,
    ) -> T4ExactDecision:
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        if not actions:
            raise HuM3T4RuntimeError("T4 observation has no legal actions")
        fingerprint = observation.fingerprint()
        expected_set_digest = legal_action_set_digest(actions)
        expected_order_digest = ordered_action_mapping_digest(actions)
        _require_equal(result, "status", "ok")
        _require_equal(result, "schema", "hu_m3_engine_result_v1")
        _require_equal(result, "kind", "t4")
        _require_equal(result, "street", "T4")
        _require_equal(result, "seat", observation.seat)
        _require_equal(result, "to_act_order", observation.to_act_order)
        _require_equal(result, "observation_fingerprint", fingerprint)
        _require_equal(result, "legal_action_count", len(actions))
        _require_equal(result, "legal_action_set_digest", expected_set_digest)
        _require_equal(result, "legal_action_order_digest", expected_order_digest)
        _require_equal(result, "engine_version", self.engine_version)
        _require_equal(result, "teacher_value_status", "diagnostic_not_match_EV")

        raw_rows = result.get("actions")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(actions):
            raise HuM3T4RuntimeError("native T4 result does not cover every legal action")
        by_original_index: dict[int, T4ExactActionValue] = {}
        seen_keys: set[str] = set()
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise HuM3T4RuntimeError("native T4 action row must be an object")
            original_index = raw.get("original_index")
            rank = raw.get("sorted_index")
            token = raw.get("action_key")
            if (
                not isinstance(original_index, int)
                or isinstance(original_index, bool)
                or not 0 <= original_index < len(actions)
            ):
                raise HuM3T4RuntimeError("native T4 action row has invalid original_index")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or not 0 <= rank < len(actions)
            ):
                raise HuM3T4RuntimeError("native T4 action row has invalid sorted_index")
            if original_index in by_original_index:
                raise HuM3T4RuntimeError("native T4 result duplicates an original_index")
            if not isinstance(token, str):
                raise HuM3T4RuntimeError("native T4 action row lacks ActionKey")
            ActionKey.from_token(token)
            expected_token = action_key(actions[original_index]).to_token()
            if token != expected_token:
                raise HuM3T4RuntimeError(
                    "native T4 ActionKey does not match its original_index"
                )
            if token in seen_keys:
                raise HuM3T4RuntimeError("native T4 result duplicates an ActionKey")
            seen_keys.add(token)
            try:
                payload_token = action_key_from_payload(raw).to_token()
            except (TypeError, ValueError) as exc:
                raise HuM3T4RuntimeError("native T4 action payload is invalid") from exc
            if payload_token != token:
                raise HuM3T4RuntimeError(
                    "native T4 action payload and ActionKey disagree"
                )
            ev = _finite_float(raw.get("score"), "native T4 action EV")
            selection_ev = _finite_float(
                raw.get("selection_score"), "native T4 selection EV"
            )
            if selection_ev != ev:
                raise HuM3T4RuntimeError(
                    "exact T4 selection and evaluation EV must be identical"
                )
            by_original_index[original_index] = T4ExactActionValue(
                original_index=original_index,
                rank=rank,
                action_key=token,
                ev=ev,
                action=actions[original_index],
            )

        if set(by_original_index) != set(range(len(actions))):
            raise HuM3T4RuntimeError("native T4 original-index coverage is incomplete")
        if {row.rank for row in by_original_index.values()} != set(range(len(actions))):
            raise HuM3T4RuntimeError("native T4 rank coverage is incomplete")
        values = [by_original_index[index].ev for index in range(len(actions))]
        expected_ranking = canonical_descending_indices(values, actions)
        for rank, original_index in enumerate(expected_ranking):
            if by_original_index[original_index].rank != rank:
                raise HuM3T4RuntimeError("native T4 ranking violates ActionKey tie-break")

        selected_index = result.get("selected_action_original_index")
        selected_token = result.get("selected_action_key")
        if selected_index != expected_ranking[0]:
            raise HuM3T4RuntimeError("native T4 selected index is not the canonical argmax")
        if selected_token != action_key(actions[selected_index]).to_token():
            raise HuM3T4RuntimeError("native T4 selected ActionKey/index mismatch")
        selected_ev = _finite_float(
            result.get("selected_action_evaluation_score"),
            "native T4 selected EV",
        )
        if selected_ev != values[selected_index]:
            raise HuM3T4RuntimeError("native T4 selected EV disagrees with its action row")
        if _finite_float(result.get("best_score"), "native T4 best score") != selected_ev:
            raise HuM3T4RuntimeError("native T4 best score disagrees with selected EV")
        expected_gap = selected_ev - (
            values[expected_ranking[1]] if len(actions) > 1 else selected_ev
        )
        selection_gap = _finite_float(
            result.get("selection_score_gap"), "native T4 selection gap"
        )
        if selection_gap != expected_gap:
            raise HuM3T4RuntimeError("native T4 selection gap is inconsistent")

        if observation.to_act_order == "first":
            _require_equal(result, "solver_id", _FIRST_SOLVER_ID)
            candidate_plan = _exact_plan(result.get("candidate_plan"), "candidate")
            evaluation_plan = _exact_plan(result.get("evaluation_plan"), "evaluation")
            if candidate_plan["future_count"] != evaluation_plan["future_count"]:
                raise HuM3T4RuntimeError("exact T4 candidate/evaluation support mismatch")
            _require_equal(
                result,
                "sample_independence",
                "not_applicable_exact_enumeration",
            )
            solver_id = _FIRST_SOLVER_ID
            future_count = _EXPECTED_FIRST_FUTURES
            belief_id: str | None = HU_M30_T4_BELIEF_ID
            value_scope = "exact_terminal_hu_ev_under_uniform_exchangeable_restart_belief"
        else:
            _require_equal(result, "solver_id", _SECOND_SOLVER_ID)
            if result.get("candidate_plan") is not None or result.get("evaluation_plan") is not None:
                raise HuM3T4RuntimeError("T4-second terminal exact must not have a chance plan")
            _require_equal(
                result,
                "sample_independence",
                "not_applicable_no_future_chance",
            )
            solver_id = _SECOND_SOLVER_ID
            future_count = 1
            belief_id = None
            value_scope = "exact_terminal_hu_ev_against_complete_opponent_board"

        scoring_digest = _digest(observation.scoring.to_dict())
        belief_digest = _belief_digest(observation)
        action_values = tuple(
            sorted(
                by_original_index.values(),
                key=lambda row: ActionKey.from_token(row.action_key).sort_key(),
            )
        )
        semantic_payload = {
            "runtime_id": HU_M30_T4_RUNTIME_ID,
            "request_schema": HU_M3_REQUEST_SCHEMA,
            "observation_fingerprint": fingerprint,
            "selected_action_key": selected_token,
            "selected_ev": selected_ev,
            "selection_gap": selection_gap,
            "legal_action_set_digest": expected_set_digest,
            "legal_action_order_digest": expected_order_digest,
            "action_values": [
                [row.action_key, row.original_index, row.rank, row.ev]
                for row in action_values
            ],
            "belief_digest": belief_digest,
            "scoring_digest": scoring_digest,
            "solver_id": solver_id,
            "engine_version": self.engine_version,
            "native_library_sha256": self.library_sha256,
            "future_count": future_count,
        }
        validation_ms = (time.perf_counter() - validation_started) * 1000.0
        total_ms = (
            native_latency_ms + validation_ms
            if execution_mode == "batch_amortized"
            else (time.perf_counter() - total_started) * 1000.0
        )
        return T4ExactDecision(
            action=actions[selected_index],
            selected_action_key=str(selected_token),
            selected_ev=selected_ev,
            selection_gap=selection_gap,
            action_values=action_values,
            seat=observation.seat,
            value_scope=value_scope,
            observation_fingerprint=fingerprint,
            legal_action_set_digest=expected_set_digest,
            legal_action_order_digest=expected_order_digest,
            belief_id=belief_id,
            belief_digest=belief_digest,
            scoring_digest=scoring_digest,
            solver_id=solver_id,
            engine_version=self.engine_version,
            native_library_sha256=self.library_sha256,
            future_count=future_count,
            native_latency_ms=native_latency_ms,
            validation_latency_ms=validation_ms,
            total_latency_ms=total_ms,
            execution_mode=execution_mode,
            batch_size=batch_size,
            result_digest=_digest(semantic_payload),
        )


class HuM3T4ExactPolicy:
    """T4-only wrapper; every earlier street is delegated byte-for-semantics."""

    def __init__(
        self,
        base_policy: object,
        solver: HuM3T4ExactSolver,
        *,
        decision_log: list[dict[str, Any]] | None = None,
    ) -> None:
        self._base_policy = base_policy
        self.hu_t4_exact_solver = solver
        self.hu_t4_decision_log = decision_log
        self.last_hu_t4_decision: T4ExactDecision | None = None
        self.hu_t4_runtime_id = HU_M30_T4_RUNTIME_ID

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_policy, name)

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        if observation.street != "T4":
            return self._base_policy.choose_action_observation(
                observation,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
            )
        expected_seat = getattr(self._base_policy, "seat", None)
        if observation.seat != expected_seat:
            raise HuM3T4RuntimeError(
                f"observation seat {observation.seat!r} does not match base policy seat {expected_seat!r}"
            )
        baseline_action: Action | None = None
        if self.hu_t4_decision_log is not None:
            baseline_action = self._base_policy.choose_action_observation(
                observation,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
            )
        decision = self.hu_t4_exact_solver.solve(observation)
        self.last_hu_t4_decision = decision
        if self.hu_t4_decision_log is not None:
            record = decision.to_dict()
            assert baseline_action is not None
            try:
                baseline_key = action_key(baseline_action).to_token()
            except (AttributeError, TypeError, ValueError) as exc:
                raise HuM3T4RuntimeError(
                    "legacy T4 baseline returned an invalid Action"
                ) from exc
            action_evs = {row.action_key: row.ev for row in decision.action_values}
            if baseline_key not in action_evs:
                raise HuM3T4RuntimeError(
                    "legacy T4 baseline action is absent from exact legal values"
                )
            record.update(
                {
                    "hand_id": hand_id,
                    "game_id": game_id,
                    "decision_seed": decision_seed,
                    "baseline_action_key": baseline_key,
                    "baseline_ev_under_exact_value": action_evs[baseline_key],
                    "final_action_key": decision.selected_action_key,
                    "override_fired": baseline_key != decision.selected_action_key,
                    "exact_value_gain_vs_baseline": (
                        decision.selected_ev - action_evs[baseline_key]
                    ),
                    "fallback_policy": None,
                    "fallback_used": False,
                }
            )
            self.hu_t4_decision_log.append(record)
        return decision.action

    def choose_action(self, board: Any, dealt_cards: Iterable[str], **kwargs: Any) -> Action:
        if getattr(board, "card_count", lambda: -1)() == 11:
            raise HuM3T4RuntimeError(
                "M3.0 exact T4 requires ActorObservation via choose_action_observation; "
                "legacy dead_cards input is forbidden"
            )
        return self._base_policy.choose_action(board, dealt_cards, **kwargs)


def _require_t4_observation(observation: ActorObservation) -> None:
    if not isinstance(observation, ActorObservation):
        raise TypeError(
            "M3.0 exact T4 requires ActorObservation; WorldState and replay truth are forbidden"
        )
    if observation.street != "T4":
        raise ValueError(f"M3.0 exact T4 requires street T4, got {observation.street}")


def _exact_plan(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HuM3T4RuntimeError(f"exact T4 {name} plan must be an object")
    if value.get("mode") != "exact_uniform_marginal":
        raise HuM3T4RuntimeError(f"exact T4 {name} plan is not exhaustive")
    if value.get("future_count") != _EXPECTED_FIRST_FUTURES:
        raise HuM3T4RuntimeError(
            f"exact T4 {name} plan must contain {_EXPECTED_FIRST_FUTURES} futures"
        )
    if value.get("rng_key_digests") != [] or value.get("sample_indices") != []:
        raise HuM3T4RuntimeError(f"exact T4 {name} plan unexpectedly used RNG")
    return value


def _require_equal(payload: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise HuM3T4RuntimeError(
            f"native T4 result {key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise HuM3T4RuntimeError(f"{context} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HuM3T4RuntimeError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise HuM3T4RuntimeError(f"{context} must be finite")
    return result


def _belief_digest(observation: ActorObservation) -> str:
    known = set(observation.known_unavailable_cards())
    unknown = [card for card in ALL_CARDS if card not in known]
    if observation.to_act_order == "first" and len(unknown) != 24:
        raise HuM3T4RuntimeError(
            f"T4-first exact belief requires 24 unknown cards, got {len(unknown)}"
        )
    payload = {
        "belief_id": (
            HU_M30_T4_BELIEF_ID if observation.to_act_order == "first" else None
        ),
        "unknown_cards": unknown if observation.to_act_order == "first" else [],
        "future_count": (
            _EXPECTED_FIRST_FUTURES if observation.to_act_order == "first" else 1
        ),
        "opponent_response": "terminal_exhaustive",
    }
    return _digest(payload)


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
