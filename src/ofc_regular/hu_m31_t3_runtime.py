"""Strict M3.1 runtime boundary for T3 CRN search with exact T4 children.

This module deliberately exposes a solver component, not a named policy or a
profile hook.  It pins the already validated M3.0 native engine by SHA-256 and
version, accepts only :class:`ActorObservation`, forces every downstream T4
decision into exact mode, and validates the complete semantic result returned
by Rust before an action can be consumed by later pilot code.

Root T3 chance remains common-random Monte Carlo.  Candidate selection and
locked evaluation use disjoint counter-RNG domains.  Returned values are
teacher/search estimates, never realized match EV or a full-game optimality
claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .counter_rng import COUNTER_RNG_SCHEMA
from .hu_belief import (
    HIDDEN_CARD_BATCH_SCHEMA,
    HIDDEN_CARD_BELIEF_SCHEMA,
    HIDDEN_CARD_PRIOR,
    sample_hidden_card_particles,
)
from .hu_infoset import ActorObservation
from .hu_m3_rust import (
    HU_M3_BATCH_REQUEST_SCHEMA,
    HU_M3_REQUEST_SCHEMA,
    engine_version,
    evaluate_request,
    evaluate_t3,
    load_native_engine,
    native_library_path,
    t3_request,
)
from .hu_m3_t4_runtime import (
    HU_M30_T4_ENGINE_VERSION,
)
from .hu_turn3_joint_exact_teacher import JointExactConfig


HU_M31_T3_RUNTIME_SCHEMA = "hu_m31_t3_runtime_decision_v2"
HU_M31_T3_RUNTIME_ID = "hu_m31_t3_crn_exact_t4_v1"
HU_M31_T3_RUN_ID = "hu-m31-t3-runtime-v1"
HU_M31_T3_ENGINE_VERSION = HU_M30_T4_ENGINE_VERSION
HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA = (
    "hu_m31_t3_semantic_result_digest_v1"
)

_RESULT_SCHEMA = "hu_m3_engine_result_v1"
_BATCH_RESULT_SCHEMA = "hu_m3_engine_batch_result_v1"
_T3_SOLVER_ID = "rust_crn_sequential_t3_v1"
_CONTINUATION_POLICY_ID = "local_infoset_response_t3_second_t4_v1"
_STRATEGY_FUSION_GUARD = "child_actions_keyed_only_by_actor_observation"
_EXACT_T4_NATIVE_SEMANTICS_ID = "m30_exact_t4_native_kernel_semantics_v1"
_VALUE_SCOPE = "q_pi_uniform_exchangeable_t3_crn_with_exact_t4_children"
_TEACHER_VALUE_STATUS = "diagnostic_not_match_EV"
_RESULT_DIGEST_SCOPE = "ordered_action_mapping_bound"
_SEMANTIC_RESULT_DIGEST_SCOPE = "dealt_order_independent_action_value_result"
_EXPECTED_RESULT_KEYS = frozenset(
    {
        "status",
        "schema",
        "engine_version",
        "solver_id",
        "kind",
        "street",
        "seat",
        "to_act_order",
        "observation_fingerprint",
        "legal_action_count",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "selected_action_original_index",
        "best_action_original_index",
        "selected_action_key",
        "selected_action_evaluation_score",
        "best_score",
        "selection_score_gap",
        "score_gap",
        "evaluation_sample_best_score",
        "evaluation_sample_regret_of_locked_selection",
        "candidate_belief",
        "evaluation_belief",
        "candidate_rng_key_digests",
        "evaluation_rng_key_digests",
        "sample_independence",
        "continuation_policy",
        "child_information_set_count",
        "actions",
        "teacher_value_status",
    }
)
_EXPECTED_ACTION_ROW_KEYS = frozenset(
    {
        "original_index",
        "sorted_index",
        "action_key",
        "placements",
        "discards",
        "score",
        "joint_ev",
        "selection_score",
        "selected_by_candidate_plan",
        "evaluation_regret_vs_sample_best",
        "selection_future_count",
        "evaluation_future_count",
    }
)
_EXPECTED_BELIEF_KEYS = frozenset(
    {
        "schema",
        "belief_schema",
        "prior",
        "counter_rng_schema",
        "observation_fingerprint",
        "street",
        "base_seed",
        "run_id",
        "start_index",
        "sample_count",
        "particle_digests",
    }
)
_EXPECTED_CONTINUATION_KEYS = frozenset(
    {
        "id",
        "downstream_t3_samples",
        "downstream_t4_samples",
        "strategy_fusion_guard",
    }
)


class HuM31T3RuntimeError(RuntimeError):
    """Raised when strict T3 startup, execution, or result validation fails."""


@dataclass(frozen=True)
class HuM31T3RuntimeConfig:
    """Immutable startup and search contract for one T3 native solver."""

    expected_library_sha256: str
    library_path: Path | None = None
    expected_engine_version: str = HU_M31_T3_ENGINE_VERSION
    require_release_library: bool = True
    run_id: str = HU_M31_T3_RUN_ID
    candidate_samples: int = 1
    evaluation_samples: int = 1
    downstream_t3_samples: int = 1
    seed: int = 2026073100
    candidate_seed: int = 2026073101
    evaluation_seed: int = 2026073102

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("T3 runtime run_id must not be empty")
        if not self.expected_engine_version:
            raise ValueError("expected_engine_version must not be empty")
        digest = self.expected_library_sha256.casefold()
        if not _is_sha256(digest):
            raise ValueError("expected_library_sha256 must be 64 hexadecimal digits")
        object.__setattr__(self, "expected_library_sha256", digest)
        if self.library_path is not None:
            object.__setattr__(self, "library_path", Path(self.library_path))
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "candidate_seed", "evaluation_seed"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -(1 << 63) <= value < (1 << 63)
            ):
                raise ValueError(f"{name} must fit a signed 64-bit integer")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError(
                "candidate_seed and evaluation_seed must be distinct for the M3.1 runtime"
            )


@dataclass(frozen=True)
class T3SearchActionValue:
    original_index: int
    rank: int
    action_key: str
    selection_ev: float
    evaluation_ev: float
    evaluation_regret: float
    action: Action

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_index": self.original_index,
            "rank": self.rank,
            "action_key": self.action_key,
            "selection_ev": self.selection_ev,
            "evaluation_ev": self.evaluation_ev,
            "evaluation_regret": self.evaluation_regret,
            "placements": [list(item) for item in self.action.placements],
            "discards": list(self.action.discards),
        }


@dataclass(frozen=True)
class T3SearchDecision:
    action: Action
    selected_action_key: str
    selected_selection_ev: float
    selected_evaluation_ev: float
    selection_gap: float
    evaluation_sample_regret: float
    action_values: tuple[T3SearchActionValue, ...]
    seat: str
    value_scope: str
    observation_fingerprint: str
    legal_action_set_digest: str
    legal_action_order_digest: str
    candidate_belief_digest: str
    evaluation_belief_digest: str
    candidate_rng_digest: str
    evaluation_rng_digest: str
    child_information_set_count: int
    candidate_samples: int
    evaluation_samples: int
    downstream_t3_samples: int
    run_id: str
    continuation_seed: int
    candidate_seed: int
    evaluation_seed: int
    use_t4_action_cache: bool
    search_contract_digest: str
    solver_id: str
    engine_version: str
    native_library_sha256: str
    native_latency_ms: float
    validation_latency_ms: float
    total_latency_ms: float
    execution_mode: str
    batch_size: int
    semantic_result_digest: str
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HU_M31_T3_RUNTIME_SCHEMA,
            "runtime_id": HU_M31_T3_RUNTIME_ID,
            "seat": self.seat,
            "value_scope": self.value_scope,
            "observation_fingerprint": self.observation_fingerprint,
            "selected_action_key": self.selected_action_key,
            "selected_selection_ev": self.selected_selection_ev,
            "selected_evaluation_ev": self.selected_evaluation_ev,
            "selection_gap": self.selection_gap,
            "evaluation_sample_regret": self.evaluation_sample_regret,
            "selected_action": {
                "placements": [list(item) for item in self.action.placements],
                "discards": list(self.action.discards),
            },
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": self.legal_action_set_digest,
            "legal_action_order_digest": self.legal_action_order_digest,
            "action_values": [row.to_dict() for row in self.action_values],
            "belief_prior": HIDDEN_CARD_PRIOR,
            "candidate_belief_digest": self.candidate_belief_digest,
            "evaluation_belief_digest": self.evaluation_belief_digest,
            "candidate_rng_digest": self.candidate_rng_digest,
            "evaluation_rng_digest": self.evaluation_rng_digest,
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": 0,
            "run_id": self.run_id,
            "continuation_seed": self.continuation_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "use_t4_action_cache": self.use_t4_action_cache,
            "continuation_policy_id": _CONTINUATION_POLICY_ID,
            "strategy_fusion_guard": _STRATEGY_FUSION_GUARD,
            "search_contract_digest": self.search_contract_digest,
            "downstream_t4_native_semantics_id": _EXACT_T4_NATIVE_SEMANTICS_ID,
            "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
            "downstream_t4_mode": "exact",
            "child_information_set_count": self.child_information_set_count,
            "solver_id": self.solver_id,
            "engine_version": self.engine_version,
            "native_library_sha256": self.native_library_sha256,
            "teacher_value_status": _TEACHER_VALUE_STATUS,
            "native_latency_ms": self.native_latency_ms,
            "validation_latency_ms": self.validation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "execution_mode": self.execution_mode,
            "batch_size": self.batch_size,
            "semantic_result_digest_schema": (
                HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA
            ),
            "semantic_result_digest_scope": _SEMANTIC_RESULT_DIGEST_SCOPE,
            "semantic_result_digest": self.semantic_result_digest,
            "result_digest_scope": _RESULT_DIGEST_SCOPE,
            "result_digest": self.result_digest,
        }


class HuM31T3SearchSolver:
    """Eagerly anchored, exact-T4, no-build and no-fallback T3 solver."""

    def __init__(self, config: HuM31T3RuntimeConfig) -> None:
        self.config = config
        path = self.config.library_path or native_library_path(release=True)
        self.library_path = Path(path).resolve()
        if not self.library_path.is_file():
            raise HuM31T3RuntimeError(
                f"native T3 library does not exist: {self.library_path}"
            )
        if (
            self.config.require_release_library
            and "release" not in {part.casefold() for part in self.library_path.parts}
        ):
            raise HuM31T3RuntimeError(
                f"native T3 runtime requires a release library: {self.library_path}"
            )
        self.library_sha256 = _sha256_file(self.library_path)
        if self.library_sha256 != self.config.expected_library_sha256:
            raise HuM31T3RuntimeError(
                "native T3 library SHA-256 mismatch: "
                f"expected {self.config.expected_library_sha256}, got {self.library_sha256}"
            )
        self.library = load_native_engine(
            path=self.library_path,
            build_if_missing=False,
        )
        self.engine_version = engine_version(library=self.library)
        if self.engine_version != self.config.expected_engine_version:
            raise HuM31T3RuntimeError(
                "native T3 engine version mismatch: "
                f"expected {self.config.expected_engine_version!r}, got {self.engine_version!r}"
            )
        self.search_config = JointExactConfig(
            candidate_samples=self.config.candidate_samples,
            evaluation_samples=self.config.evaluation_samples,
            downstream_t3_samples=self.config.downstream_t3_samples,
            downstream_t4_samples=0,
            seed=self.config.seed,
            candidate_seed=self.config.candidate_seed,
            evaluation_seed=self.config.evaluation_seed,
            run_id=self.config.run_id,
            use_final_turn_cache=True,
        )

    def solve(self, observation: ActorObservation) -> T3SearchDecision:
        _require_t3_observation(observation)
        started = time.perf_counter()
        native_started = time.perf_counter()
        result = evaluate_t3(
            observation,
            config=self.search_config,
            library=self.library,
        )
        native_ms = (time.perf_counter() - native_started) * 1000.0
        validation_started = time.perf_counter()
        return self._validate_result(
            observation,
            result,
            native_latency_ms=native_ms,
            total_started=started,
            validation_started=validation_started,
            execution_mode="scalar",
            batch_size=1,
        )

    def solve_many(
        self, observations: Sequence[ActorObservation]
    ) -> list[T3SearchDecision]:
        if not observations:
            return []
        for observation in observations:
            _require_t3_observation(observation)
        started = time.perf_counter()
        native_started = time.perf_counter()
        response = evaluate_request(
            {
                "schema": HU_M3_BATCH_REQUEST_SCHEMA,
                "requests": [
                    t3_request(observation, config=self.search_config)
                    for observation in observations
                ],
            },
            library=self.library,
        )
        native_total_ms = (time.perf_counter() - native_started) * 1000.0
        _require_exact_keys(
            response,
            frozenset({"status", "schema", "engine_version", "results"}),
            "native T3 batch envelope",
        )
        _require_equal(response, "status", "ok")
        _require_equal(response, "schema", _BATCH_RESULT_SCHEMA)
        _require_equal(response, "engine_version", self.engine_version)
        results = response.get("results")
        if not isinstance(results, list) or not all(
            isinstance(row, Mapping) for row in results
        ):
            raise HuM31T3RuntimeError("native T3 batch results must be objects")
        if len(results) != len(observations):
            raise HuM31T3RuntimeError("native T3 batch result length mismatch")
        amortized_native_ms = native_total_ms / len(observations)
        decisions: list[T3SearchDecision] = []
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
    ) -> T3SearchDecision:
        _require_exact_keys(result, _EXPECTED_RESULT_KEYS, "native T3 result")
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        if not actions:
            raise HuM31T3RuntimeError("T3 observation has no legal actions")
        fingerprint = observation.fingerprint()
        expected_set_digest = legal_action_set_digest(actions)
        expected_order_digest = ordered_action_mapping_digest(actions)
        _require_equal(result, "status", "ok")
        _require_equal(result, "schema", _RESULT_SCHEMA)
        _require_equal(result, "engine_version", self.engine_version)
        _require_equal(result, "solver_id", _T3_SOLVER_ID)
        _require_equal(result, "kind", "t3")
        _require_equal(result, "street", "T3")
        _require_equal(result, "seat", observation.seat)
        _require_equal(result, "to_act_order", observation.to_act_order)
        _require_equal(result, "observation_fingerprint", fingerprint)
        _require_int_equal(result, "legal_action_count", len(actions))
        _require_equal(result, "legal_action_set_digest", expected_set_digest)
        _require_equal(result, "legal_action_order_digest", expected_order_digest)
        _require_equal(result, "sample_independence", "disjoint_particle_rng_keys")
        _require_equal(result, "teacher_value_status", _TEACHER_VALUE_STATUS)

        continuation = result.get("continuation_policy")
        if not isinstance(continuation, Mapping):
            raise HuM31T3RuntimeError("native T3 continuation_policy must be an object")
        _require_exact_keys(
            continuation,
            _EXPECTED_CONTINUATION_KEYS,
            "native T3 continuation_policy",
        )
        _require_equal(continuation, "id", _CONTINUATION_POLICY_ID)
        _require_int_equal(
            continuation,
            "downstream_t3_samples",
            self.config.downstream_t3_samples,
        )
        _require_int_equal(continuation, "downstream_t4_samples", 0)
        _require_equal(
            continuation,
            "strategy_fusion_guard",
            _STRATEGY_FUSION_GUARD,
        )

        candidate_belief_digest, expected_candidate_rng_keys = _validate_belief(
            result.get("candidate_belief"),
            role="candidate",
            observation=observation,
            base_seed=self.config.candidate_seed,
            run_id=f"{self.config.run_id}:candidate_selection",
            sample_count=self.config.candidate_samples,
        )
        evaluation_belief_digest, expected_evaluation_rng_keys = _validate_belief(
            result.get("evaluation_belief"),
            role="evaluation",
            observation=observation,
            base_seed=self.config.evaluation_seed,
            run_id=f"{self.config.run_id}:locked_evaluation",
            sample_count=self.config.evaluation_samples,
        )
        candidate_rng_keys = _sha256_list(
            result.get("candidate_rng_key_digests"),
            "candidate RNG key digests",
            expected_count=self.config.candidate_samples,
        )
        evaluation_rng_keys = _sha256_list(
            result.get("evaluation_rng_key_digests"),
            "evaluation RNG key digests",
            expected_count=self.config.evaluation_samples,
        )
        if candidate_rng_keys != expected_candidate_rng_keys:
            raise HuM31T3RuntimeError(
                "candidate RNG key digests disagree with deterministic ActorObservation belief"
            )
        if evaluation_rng_keys != expected_evaluation_rng_keys:
            raise HuM31T3RuntimeError(
                "evaluation RNG key digests disagree with deterministic ActorObservation belief"
            )
        if set(candidate_rng_keys) & set(evaluation_rng_keys):
            raise HuM31T3RuntimeError(
                "candidate-selection and evaluation RNG key digests overlap"
            )

        raw_rows = result.get("actions")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(actions):
            raise HuM31T3RuntimeError("native T3 result does not cover every legal action")
        by_original_index: dict[int, T3SearchActionValue] = {}
        seen_keys: set[str] = set()
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise HuM31T3RuntimeError("native T3 action row must be an object")
            _require_exact_keys(raw, _EXPECTED_ACTION_ROW_KEYS, "native T3 action row")
            original_index = _bounded_index(
                raw.get("original_index"), len(actions), "original_index"
            )
            rank = _bounded_index(raw.get("sorted_index"), len(actions), "sorted_index")
            if original_index in by_original_index:
                raise HuM31T3RuntimeError("native T3 result duplicates an original_index")
            token = raw.get("action_key")
            if not isinstance(token, str):
                raise HuM31T3RuntimeError("native T3 action row lacks ActionKey")
            try:
                ActionKey.from_token(token)
            except (TypeError, ValueError) as exc:
                raise HuM31T3RuntimeError("native T3 action row has invalid ActionKey") from exc
            expected_token = action_key(actions[original_index]).to_token()
            if token != expected_token:
                raise HuM31T3RuntimeError(
                    "native T3 ActionKey does not match its original_index"
                )
            if token in seen_keys:
                raise HuM31T3RuntimeError("native T3 result duplicates an ActionKey")
            seen_keys.add(token)
            try:
                payload_token = action_key_from_payload(raw).to_token()
            except (TypeError, ValueError) as exc:
                raise HuM31T3RuntimeError("native T3 action payload is invalid") from exc
            if payload_token != token:
                raise HuM31T3RuntimeError(
                    "native T3 action payload and ActionKey disagree"
                )
            selection_ev = _finite_float(
                raw.get("selection_score"), "native T3 selection EV"
            )
            evaluation_ev = _finite_float(raw.get("score"), "native T3 evaluation EV")
            _require_float_equal(
                raw.get("joint_ev"), evaluation_ev, "native T3 joint EV"
            )
            evaluation_regret = _finite_float(
                raw.get("evaluation_regret_vs_sample_best"),
                "native T3 evaluation regret",
            )
            _require_int_equal(
                raw, "selection_future_count", self.config.candidate_samples
            )
            _require_int_equal(
                raw, "evaluation_future_count", self.config.evaluation_samples
            )
            if not isinstance(raw.get("selected_by_candidate_plan"), bool):
                raise HuM31T3RuntimeError(
                    "native T3 selected_by_candidate_plan must be boolean"
                )
            by_original_index[original_index] = T3SearchActionValue(
                original_index=original_index,
                rank=rank,
                action_key=token,
                selection_ev=selection_ev,
                evaluation_ev=evaluation_ev,
                evaluation_regret=evaluation_regret,
                action=actions[original_index],
            )

        if set(by_original_index) != set(range(len(actions))):
            raise HuM31T3RuntimeError("native T3 original-index coverage is incomplete")
        if {row.rank for row in by_original_index.values()} != set(range(len(actions))):
            raise HuM31T3RuntimeError("native T3 rank coverage is incomplete")
        selection_values = [
            by_original_index[index].selection_ev for index in range(len(actions))
        ]
        evaluation_values = [
            by_original_index[index].evaluation_ev for index in range(len(actions))
        ]
        expected_ranking = canonical_descending_indices(selection_values, actions)
        for rank, original_index in enumerate(expected_ranking):
            if by_original_index[original_index].rank != rank:
                raise HuM31T3RuntimeError(
                    "native T3 ranking violates selection EV and ActionKey tie-break"
                )

        selected_index = _bounded_index(
            result.get("selected_action_original_index"),
            len(actions),
            "selected_action_original_index",
        )
        if selected_index != expected_ranking[0]:
            raise HuM31T3RuntimeError("native T3 selected index is not canonical argmax")
        best_index = _bounded_index(
            result.get("best_action_original_index"),
            len(actions),
            "best_action_original_index",
        )
        if best_index != selected_index:
            raise HuM31T3RuntimeError(
                "native T3 best action index disagrees with locked selection"
            )
        selected_token = result.get("selected_action_key")
        if selected_token != action_key(actions[selected_index]).to_token():
            raise HuM31T3RuntimeError("native T3 selected ActionKey/index mismatch")
        selected_flags = [
            bool(raw["selected_by_candidate_plan"]) for raw in raw_rows
        ]
        for raw, selected in zip(raw_rows, selected_flags, strict=True):
            should_be_selected = raw["original_index"] == selected_index
            if selected != should_be_selected:
                raise HuM31T3RuntimeError(
                    "native T3 selected_by_candidate_plan flags are inconsistent"
                )

        selected_selection_ev = selection_values[selected_index]
        selected_evaluation_ev = evaluation_values[selected_index]
        _require_float_equal(
            result.get("selected_action_evaluation_score"),
            selected_evaluation_ev,
            "native T3 selected evaluation EV",
        )
        _require_float_equal(
            result.get("best_score"),
            selected_evaluation_ev,
            "native T3 locked best score",
        )
        second_selection_ev = (
            selection_values[expected_ranking[1]]
            if len(actions) > 1
            else selected_selection_ev
        )
        selection_gap = selected_selection_ev - second_selection_ev
        _require_float_equal(
            result.get("selection_score_gap"),
            selection_gap,
            "native T3 selection gap",
        )
        _require_float_equal(
            result.get("score_gap"), selection_gap, "native T3 score gap"
        )
        evaluation_best = max(evaluation_values)
        _require_float_equal(
            result.get("evaluation_sample_best_score"),
            evaluation_best,
            "native T3 evaluation sample best score",
        )
        evaluation_sample_regret = evaluation_best - selected_evaluation_ev
        _require_float_equal(
            result.get("evaluation_sample_regret_of_locked_selection"),
            evaluation_sample_regret,
            "native T3 locked-selection evaluation regret",
        )
        for row in by_original_index.values():
            _require_float_equal(
                row.evaluation_regret,
                evaluation_best - row.evaluation_ev,
                "native T3 per-action evaluation regret",
            )

        child_count = result.get("child_information_set_count")
        if isinstance(child_count, bool) or not isinstance(child_count, int) or child_count < 0:
            raise HuM31T3RuntimeError(
                "native T3 child_information_set_count must be non-negative"
            )
        action_values = tuple(
            sorted(
                by_original_index.values(),
                key=lambda row: ActionKey.from_token(row.action_key).sort_key(),
            )
        )
        candidate_rng_digest = _digest(candidate_rng_keys)
        evaluation_rng_digest = _digest(evaluation_rng_keys)
        search_contract = _search_contract(self.config)
        search_contract_digest = _digest(search_contract)
        mapping_payload = {
            "runtime_id": HU_M31_T3_RUNTIME_ID,
            "request_schema": HU_M3_REQUEST_SCHEMA,
            "observation_fingerprint": fingerprint,
            "selected_action_key": selected_token,
            "selected_selection_ev": selected_selection_ev,
            "selected_evaluation_ev": selected_evaluation_ev,
            "selection_gap": selection_gap,
            "evaluation_sample_regret": evaluation_sample_regret,
            "legal_action_set_digest": expected_set_digest,
            "legal_action_order_digest": expected_order_digest,
            "action_values": [
                [
                    row.action_key,
                    row.original_index,
                    row.rank,
                    row.selection_ev,
                    row.evaluation_ev,
                ]
                for row in action_values
            ],
            "candidate_belief_digest": candidate_belief_digest,
            "evaluation_belief_digest": evaluation_belief_digest,
            "candidate_rng_digest": candidate_rng_digest,
            "evaluation_rng_digest": evaluation_rng_digest,
            "search_contract": search_contract,
            "search_contract_digest": search_contract_digest,
            "downstream_t4_native_semantics_id": _EXACT_T4_NATIVE_SEMANTICS_ID,
            "child_information_set_count": child_count,
            "solver_id": _T3_SOLVER_ID,
            "engine_version": self.engine_version,
            "native_library_sha256": self.library_sha256,
        }
        semantic_result_payload = {
            "schema": HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
            "runtime_id": HU_M31_T3_RUNTIME_ID,
            "request_schema": HU_M3_REQUEST_SCHEMA,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "seat": observation.seat,
            "value_scope": _VALUE_SCOPE,
            "observation_fingerprint": fingerprint,
            "selected_action_key": selected_token,
            "selected_selection_ev": _canonical_digest_float(
                selected_selection_ev
            ),
            "selected_evaluation_ev": _canonical_digest_float(
                selected_evaluation_ev
            ),
            "selection_gap": _canonical_digest_float(selection_gap),
            "evaluation_sample_regret": _canonical_digest_float(
                evaluation_sample_regret
            ),
            "legal_action_set_digest": expected_set_digest,
            "action_values": [
                [
                    row.action_key,
                    row.rank,
                    _canonical_digest_float(row.selection_ev),
                    _canonical_digest_float(row.evaluation_ev),
                    _canonical_digest_float(row.evaluation_regret),
                ]
                for row in action_values
            ],
            "candidate_belief_digest": candidate_belief_digest,
            "evaluation_belief_digest": evaluation_belief_digest,
            "candidate_rng_digest": candidate_rng_digest,
            "evaluation_rng_digest": evaluation_rng_digest,
            "search_contract": search_contract,
            "search_contract_digest": search_contract_digest,
            "sample_independence": "disjoint_particle_rng_keys",
            "downstream_t4_native_semantics_id": _EXACT_T4_NATIVE_SEMANTICS_ID,
            "teacher_value_status": _TEACHER_VALUE_STATUS,
            "child_information_set_count": child_count,
            "solver_id": _T3_SOLVER_ID,
            "engine_version": self.engine_version,
            "native_library_sha256": self.library_sha256,
        }
        validation_ms = (time.perf_counter() - validation_started) * 1000.0
        total_ms = (
            native_latency_ms + validation_ms
            if execution_mode == "batch_amortized"
            else (time.perf_counter() - total_started) * 1000.0
        )
        return T3SearchDecision(
            action=actions[selected_index],
            selected_action_key=str(selected_token),
            selected_selection_ev=selected_selection_ev,
            selected_evaluation_ev=selected_evaluation_ev,
            selection_gap=selection_gap,
            evaluation_sample_regret=evaluation_sample_regret,
            action_values=action_values,
            seat=observation.seat,
            value_scope=_VALUE_SCOPE,
            observation_fingerprint=fingerprint,
            legal_action_set_digest=expected_set_digest,
            legal_action_order_digest=expected_order_digest,
            candidate_belief_digest=candidate_belief_digest,
            evaluation_belief_digest=evaluation_belief_digest,
            candidate_rng_digest=candidate_rng_digest,
            evaluation_rng_digest=evaluation_rng_digest,
            child_information_set_count=child_count,
            candidate_samples=self.config.candidate_samples,
            evaluation_samples=self.config.evaluation_samples,
            downstream_t3_samples=self.config.downstream_t3_samples,
            run_id=self.config.run_id,
            continuation_seed=self.config.seed,
            candidate_seed=self.config.candidate_seed,
            evaluation_seed=self.config.evaluation_seed,
            use_t4_action_cache=True,
            search_contract_digest=search_contract_digest,
            solver_id=_T3_SOLVER_ID,
            engine_version=self.engine_version,
            native_library_sha256=self.library_sha256,
            native_latency_ms=native_latency_ms,
            validation_latency_ms=validation_ms,
            total_latency_ms=total_ms,
            execution_mode=execution_mode,
            batch_size=batch_size,
            semantic_result_digest=_digest(semantic_result_payload),
            result_digest=_digest(mapping_payload),
        )


def _require_t3_observation(observation: ActorObservation) -> None:
    if not isinstance(observation, ActorObservation):
        raise TypeError(
            "M3.1 T3 runtime requires ActorObservation; WorldState and replay truth are forbidden"
        )
    if observation.street != "T3":
        raise ValueError(
            f"M3.1 T3 runtime requires street T3, got {observation.street}"
        )


def _validate_belief(
    value: Any,
    *,
    role: str,
    observation: ActorObservation,
    base_seed: int,
    run_id: str,
    sample_count: int,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise HuM31T3RuntimeError(f"native T3 {role} belief must be an object")
    _require_exact_keys(value, _EXPECTED_BELIEF_KEYS, f"native T3 {role} belief")
    _require_equal(value, "schema", HIDDEN_CARD_BATCH_SCHEMA)
    _require_equal(value, "belief_schema", HIDDEN_CARD_BELIEF_SCHEMA)
    _require_equal(value, "prior", HIDDEN_CARD_PRIOR)
    _require_equal(value, "counter_rng_schema", COUNTER_RNG_SCHEMA)
    _require_equal(value, "observation_fingerprint", observation.fingerprint())
    _require_equal(value, "street", "T3")
    _require_int_equal(value, "base_seed", base_seed)
    _require_equal(value, "run_id", run_id)
    _require_int_equal(value, "start_index", 0)
    _require_int_equal(value, "sample_count", sample_count)
    particle_digests = _sha256_list(
        value.get("particle_digests"),
        f"native T3 {role} particle digests",
        expected_count=sample_count,
    )
    expected_batch = sample_hidden_card_particles(
        observation,
        base_seed=base_seed,
        run_id=run_id,
        sample_count=sample_count,
    )
    expected_payload = expected_batch.to_dict(include_particles=False)
    expected_particle_digests = tuple(expected_payload["particle_digests"])
    if particle_digests != expected_particle_digests:
        raise HuM31T3RuntimeError(
            f"native T3 {role} particle digests disagree with deterministic belief"
        )
    if _digest(value) != _digest(expected_payload):
        raise HuM31T3RuntimeError(
            f"native T3 {role} belief payload disagrees with deterministic belief"
        )
    expected_rng_keys = tuple(
        particle.rng_key_digest for particle in expected_batch.particles
    )
    return _digest(expected_payload), expected_rng_keys


def _require_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    keys = frozenset(str(key) for key in payload)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise HuM31T3RuntimeError(
            f"{context} field mismatch: missing={missing}, unknown={unknown}"
        )


def _require_equal(payload: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise HuM31T3RuntimeError(
            f"native T3 result {key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_int_equal(
    payload: Mapping[str, Any], key: str, expected: int
) -> None:
    actual = payload.get(key)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
        raise HuM31T3RuntimeError(
            f"native T3 result {key} mismatch: expected integer {expected!r}, got {actual!r}"
        )


def _bounded_index(value: Any, upper: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < upper
    ):
        raise HuM31T3RuntimeError(f"native T3 action row has invalid {name}")
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HuM31T3RuntimeError(f"{context} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise HuM31T3RuntimeError(f"{context} must be finite")
    return result


def _require_float_equal(value: Any, expected: float, context: str) -> None:
    actual = _finite_float(value, context)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise HuM31T3RuntimeError(
            f"{context} mismatch: expected {expected!r}, got {actual!r}"
        )


def _sha256_list(value: Any, context: str, *, expected_count: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise HuM31T3RuntimeError(
            f"{context} must contain exactly {expected_count} entries"
        )
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _is_sha256(item):
            raise HuM31T3RuntimeError(f"{context} contains an invalid SHA-256")
        rows.append(item)
    if len(set(rows)) != len(rows):
        raise HuM31T3RuntimeError(f"{context} contains duplicate entries")
    return tuple(rows)


def _search_contract(config: HuM31T3RuntimeConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "continuation_seed": config.seed,
        "candidate_seed": config.candidate_seed,
        "evaluation_seed": config.evaluation_seed,
        "candidate_samples": config.candidate_samples,
        "evaluation_samples": config.evaluation_samples,
        "downstream_t3_samples": config.downstream_t3_samples,
        "downstream_t4_samples": 0,
        "use_t4_action_cache": True,
        "continuation_policy_id": _CONTINUATION_POLICY_ID,
        "strategy_fusion_guard": _STRATEGY_FUSION_GUARD,
        "downstream_t4_native_semantics_id": _EXACT_T4_NATIVE_SEMANTICS_ID,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.casefold()
    )


def _canonical_digest_float(value: float) -> float:
    """Remove signed-zero noise from the cross-mode semantic certificate."""

    return 0.0 if value == 0.0 else value


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
