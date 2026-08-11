"""Production ABR-development labels from the accepted M3.1 T3 teacher.

The accepted Candidate02 library remains the authority for every legacy
action Q and selected ActionKey.  A separately hash-pinned diagnostic build
must reproduce those values bit-for-bit on every generated root before its
aggregate terminal components may be used.  The components are linear
terminal-reward terms, not synthetic labels:

* greedy: regular HU terminal score;
* foul pressure: HU score plus terminal opponent-foul and hero-scoop bonuses;
* royalty denial: HU score minus terminal opponent royalty and FL value.

Only public ``ActorObservation`` values are persisted.  Sampled opponent
discards and realized future cards remain native world state and are never
serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import run_hu_m31_t3_step6d_performance as step6d
from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .ai_profiles import ModelBundle, ModelPaths, load_model_bundle
from .hu_infoset import ActorObservation
from .hu_m3_rust import (
    evaluate_batch,
    load_native_engine,
    t3_abr_component_request,
)
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_dataset_contract_v1 import (
    ACCEPTED_CANDIDATE_LIBRARY_SHA256,
    PRIMARY_BUDGET,
)
from .hu_m31_t3_runtime import (
    HU_M31_T3_ENGINE_VERSION,
    HU_M31_T3_RUNTIME_ID,
    HU_M31_T3_RUNTIME_SCHEMA,
    HuM31T3RuntimeConfig,
    HuM31T3SearchSolver,
)
from .hu_m31_t3_step6c_contract import STEP6C_RUN_ID
from .hu_m31_t3_step6d_contract import (
    SEED_STRIDE,
    canonical_bytes,
    canonical_sha256,
    schedule_by_name,
    validate_seed_schedule,
)


ABR_TEACHER_PLAN_SCHEMA = "hu_m31_t3_abr_teacher_plan_v1"
ABR_TEACHER_PAIR_SCHEMA = "hu_m31_t3_abr_teacher_pair_evidence_v1"
ABR_TEACHER_DONE_SCHEMA = "hu_m31_t3_abr_teacher_done_v1"
ABR_TEACHER_RECEIPT_SCHEMA = "hu_m31_t3_abr_teacher_receipt_v1"
ABR_TERMINAL_COMPONENT_SCHEMA = "hu_m3_t3_terminal_component_means_v1"
ABR_DIAGNOSTIC_RESULT_SCHEMA = "hu_m3_t3_abr_component_result_v1"
ABR_FAMILY_REWARD_SCHEMA = "hu_m31_t3_abr_family_terminal_rewards_v1"

SCHEDULE_NAME = "abr_development"
MIN_PILOT_PAIRS = 50
MAX_DEVELOPMENT_PAIRS = 250
PAIR_DIRECTORY = "pairs"
PLAN_FILE = "PLAN.json"
DONE_FILE = "DONE.json"
RAW_EXAMPLES_FILE = "raw_examples.json"
TEACHER_RECEIPT_FILE = "teacher_receipt.json"

FAMILY_REWARD_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "response_id": "greedy_search_response",
        "utility": "hu_score",
        "formula": {
            "hu_score_mean": 1.0,
        },
    },
    {
        "response_id": "foul_pressure_response",
        "utility": "hu_score_plus_opponent_foul_and_hero_scoop_pressure",
        "formula": {
            "hu_score_mean": 1.0,
            "opponent_bust_rate": 3.0,
            "hero_scoop_rate": 3.0,
        },
    },
    {
        "response_id": "royalty_denial_response",
        "utility": "hu_score_minus_opponent_royalty_and_fl_value",
        "formula": {
            "hu_score_mean": 1.0,
            "opponent_royalty_mean": -0.5,
            "opponent_fl_value_mean": -0.5,
        },
    },
)
FAMILY_REWARD_SHA256 = canonical_sha256(
    {
        "schema": ABR_FAMILY_REWARD_SCHEMA,
        "perspective": "response_actor_hu_score",
        "definitions": list(FAMILY_REWARD_DEFINITIONS),
        "linear_expectation_of_terminal_rewards": True,
        "synthetic_values_allowed": False,
    }
)
RESPONSE_IDS = tuple(row["response_id"] for row in FAMILY_REWARD_DEFINITIONS)

_COMPONENT_KEYS = frozenset(
    {
        "hu_score_mean",
        "hero_bust_rate",
        "opponent_bust_rate",
        "hero_scoop_rate",
        "opponent_scoop_rate",
        "hero_royalty_mean",
        "opponent_royalty_mean",
        "hero_fl_value_mean",
        "opponent_fl_value_mean",
        "future_count",
    }
)
_ACCEPTED_DECISION_KEYS = frozenset(
    {
        "schema",
        "runtime_id",
        "seat",
        "value_scope",
        "observation_fingerprint",
        "selected_action_key",
        "selected_selection_ev",
        "selected_evaluation_ev",
        "selection_gap",
        "evaluation_sample_regret",
        "selected_action",
        "action_key_schema",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "action_values",
        "belief_prior",
        "candidate_belief_digest",
        "evaluation_belief_digest",
        "candidate_rng_digest",
        "evaluation_rng_digest",
        "candidate_samples",
        "evaluation_samples",
        "downstream_t3_samples",
        "downstream_t4_samples",
        "run_id",
        "continuation_seed",
        "candidate_seed",
        "evaluation_seed",
        "use_t4_action_cache",
        "continuation_policy_id",
        "strategy_fusion_guard",
        "search_contract_digest",
        "downstream_t4_native_semantics_id",
        "downstream_t4_native_anchor",
        "downstream_t4_mode",
        "child_information_set_count",
        "solver_id",
        "engine_version",
        "native_library_sha256",
        "teacher_value_status",
        "native_latency_ms",
        "validation_latency_ms",
        "total_latency_ms",
        "execution_mode",
        "batch_size",
        "semantic_result_digest_schema",
        "semantic_result_digest_scope",
        "semantic_result_digest",
        "result_digest_scope",
        "result_digest",
    }
)
_ACCEPTED_ACTION_KEYS = frozenset(
    {
        "original_index",
        "rank",
        "action_key",
        "selection_ev",
        "evaluation_ev",
        "evaluation_regret",
        "placements",
        "discards",
    }
)
_DIAGNOSTIC_DECISION_KEYS = frozenset(
    {
        "status",
        "schema",
        "engine_version",
        "solver_id",
        "legacy_solver_id",
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
        "terminal_component_schema",
        "terminal_component_visibility",
        "actions",
        "teacher_value_status",
    }
)
_DIAGNOSTIC_ACTION_KEYS = frozenset(
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
        "terminal_components",
    }
)
_BELIEF_KEYS = frozenset(
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
_CONTINUATION_KEYS = frozenset(
    {
        "id",
        "downstream_t3_samples",
        "downstream_t4_samples",
        "strategy_fusion_guard",
    }
)
_SHA_CHARS = frozenset("0123456789abcdef")

RootGenerator = Callable[
    [Mapping[str, Any], ModelBundle | None],
    Sequence[ActorObservation],
]
PairSearchAdapter = Callable[
    [
        Sequence[ActorObservation],
        Mapping[str, Any],
        Path | None,
        Path | None,
    ],
    tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


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


def _canonical_read(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical object")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    raw = canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != raw
            ):
                raise FileExistsError(
                    f"immutable ABR teacher artifact changed: {destination}"
                ) from None
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _source_sha256(relative: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return _sha256_file(root / relative)


def development_seed_values(index: int) -> dict[str, int]:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < MAX_DEVELOPMENT_PAIRS
    ):
        raise ValueError("ABR development pair index must be in 0..249")
    schedule = schedule_by_name(SCHEDULE_NAME)
    return {
        key: int(base) + SEED_STRIDE * index
        for key, base in zip(
            schedule.namespace_keys, schedule.namespace_bases, strict=True
        )
    }


def pair_contract(index: int) -> dict[str, Any]:
    seeds = development_seed_values(index)
    return {
        "schedule": SCHEDULE_NAME,
        "pair_index": index,
        "root_indices": [index * 2, index * 2 + 1],
        "profile": behavior_profile_for_index(index),
        "seeds": {
            "hand": seeds["hand"],
            "behavior": seeds["actor_policy"],
            "candidate": seeds["opponent_policy"],
            "evaluation": seeds["evaluation"],
            "child": seeds["child"],
            "confirmation_reserved": seeds["confirmation"],
        },
        "budget": dict(PRIMARY_BUDGET),
        "seat_order": ["first", "second"],
        "response_ids": list(RESPONSE_IDS),
        "training_eligible": False,
        "abr_response_fit_allowed": True,
        "locked_before_content_read": False,
    }


def build_plan(
    *,
    pair_count: int,
    diagnostic_library_sha256: str,
) -> dict[str, Any]:
    validate_seed_schedule()
    if (
        isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or not MIN_PILOT_PAIRS <= pair_count <= MAX_DEVELOPMENT_PAIRS
    ):
        raise ValueError(
            "ABR pilot must contain 50..250 paired roots (100..500 states)"
        )
    if not _is_sha256(diagnostic_library_sha256):
        raise ValueError("diagnostic library SHA-256 must be lowercase hex")
    schedule = schedule_by_name(SCHEDULE_NAME)
    pairs = [pair_contract(index) for index in range(pair_count)]
    identity = {
        "schema": ABR_TEACHER_PLAN_SCHEMA,
        "status": "prepared_bounded_development_only",
        "schedule": schedule.to_dict(),
        "seed_stride": SEED_STRIDE,
        "pair_count": pair_count,
        "state_count": pair_count * 2,
        "production_pair_count_required": MAX_DEVELOPMENT_PAIRS,
        "production_state_count_required": MAX_DEVELOPMENT_PAIRS * 2,
        "production_coverage_complete": pair_count
        == MAX_DEVELOPMENT_PAIRS,
        "paired_count_per_response": {
            response_id: pair_count for response_id in RESPONSE_IDS
        },
        "root_count_per_response": {
            response_id: pair_count * 2 for response_id in RESPONSE_IDS
        },
        "pair_indices": list(range(pair_count)),
        "pairs": pairs,
        "pair_aggregate_sha256": canonical_sha256(pairs),
        "behavior_profiles": list(M31_T3_BEHAVIOR_PROFILES),
        "behavior_profile_rotation_sha256": canonical_sha256(
            [row["profile"] for row in pairs]
        ),
        "budget": dict(PRIMARY_BUDGET),
        "accepted_library_sha256": ACCEPTED_CANDIDATE_LIBRARY_SHA256,
        "diagnostic_library_sha256": diagnostic_library_sha256,
        "accepted_solver_id": "rust_crn_sequential_t3_v1",
        "diagnostic_solver_id": "rust_crn_sequential_t3_abr_components_v1",
        "legacy_q_bit_exact_required_every_root": True,
        "family_reward_schema": ABR_FAMILY_REWARD_SCHEMA,
        "family_reward_definitions": list(FAMILY_REWARD_DEFINITIONS),
        "family_reward_sha256": FAMILY_REWARD_SHA256,
        "teacher_source_sha256": _source_sha256(
            "src/ofc_regular/hu_m31_t3_abr_teacher_v1.py"
        ),
        "engine_search_source_sha256": _source_sha256(
            "rust/hu_m3_engine/src/search.rs"
        ),
        "actor_observation_only": True,
        "candidate_evaluation_rng_disjoint": True,
        "synthetic_values_allowed": False,
        "teacher_values_are_realized_locked_match_ev": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "cloud_launch_authorized": False,
    }
    return {
        **identity,
        "plan_identity_sha256": canonical_sha256(identity),
    }


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    if plan.get("schema") != ABR_TEACHER_PLAN_SCHEMA:
        raise ValueError("ABR teacher plan schema changed")
    pair_count = plan.get("pair_count")
    diagnostic_sha = plan.get("diagnostic_library_sha256")
    expected = build_plan(
        pair_count=pair_count,
        diagnostic_library_sha256=diagnostic_sha,
    )
    if plan != expected:
        raise ValueError("ABR teacher plan changed")
    return plan


def prepare_run(
    *,
    run_directory: str | Path,
    pair_count: int,
    accepted_library_path: str | Path,
    diagnostic_library_path: str | Path,
    expected_diagnostic_library_sha256: str,
) -> dict[str, Any]:
    accepted = Path(accepted_library_path).resolve()
    diagnostic = Path(diagnostic_library_path).resolve()
    if (
        accepted.is_symlink()
        or diagnostic.is_symlink()
        or not accepted.is_file()
        or not diagnostic.is_file()
    ):
        raise ValueError("ABR teacher libraries must be regular files")
    if _sha256_file(accepted) != ACCEPTED_CANDIDATE_LIBRARY_SHA256:
        raise ValueError("accepted Candidate02 library SHA-256 changed")
    if (
        not _is_sha256(expected_diagnostic_library_sha256)
        or _sha256_file(diagnostic) != expected_diagnostic_library_sha256
    ):
        raise ValueError("diagnostic library SHA-256 changed")
    plan = build_plan(
        pair_count=pair_count,
        diagnostic_library_sha256=expected_diagnostic_library_sha256,
    )
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)
    _write_once(root / PLAN_FILE, plan)
    (root / PAIR_DIRECTORY).mkdir(exist_ok=True)
    validate_run_layout(root, plan=plan, allow_outputs=False)
    return plan


def _default_bundle() -> ModelBundle:
    return load_model_bundle(
        ModelPaths(),
        profiles=set(M31_T3_BEHAVIOR_PROFILES),
    )


def _default_root_generator(
    pair: Mapping[str, Any],
    bundle: ModelBundle | None,
) -> Sequence[ActorObservation]:
    if bundle is None:
        raise ValueError("ABR root generation requires the behavior model bundle")
    return generate_behavior_t3_roots(
        hand_seed=int(pair["seeds"]["hand"]),
        behavior_seed=int(pair["seeds"]["behavior"]),
        profile=str(pair["profile"]),
        bundle=bundle,
    )


def _default_pair_search(
    observations: Sequence[ActorObservation],
    pair: Mapping[str, Any],
    accepted_library_path: Path | None,
    diagnostic_library_path: Path | None,
) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
    if accepted_library_path is None or diagnostic_library_path is None:
        raise ValueError("ABR production search requires both pinned libraries")
    seeds = pair["seeds"]
    budget = pair["budget"]
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=ACCEPTED_CANDIDATE_LIBRARY_SHA256,
            library_path=accepted_library_path,
            run_id=STEP6C_RUN_ID,
            candidate_samples=int(budget["candidate_samples"]),
            evaluation_samples=int(budget["evaluation_samples"]),
            downstream_t3_samples=int(budget["downstream_t3_samples"]),
            seed=int(seeds["child"]),
            candidate_seed=int(seeds["candidate"]),
            evaluation_seed=int(seeds["evaluation"]),
        )
    )
    accepted = [row.to_dict() for row in solver.solve_many(observations)]
    diagnostic_library = load_native_engine(
        path=diagnostic_library_path,
        build_if_missing=False,
    )
    from .hu_turn3_joint_exact_teacher import JointExactConfig

    config = JointExactConfig(
        candidate_samples=int(budget["candidate_samples"]),
        evaluation_samples=int(budget["evaluation_samples"]),
        downstream_t3_samples=int(budget["downstream_t3_samples"]),
        downstream_t4_samples=0,
        seed=int(seeds["child"]),
        candidate_seed=int(seeds["candidate"]),
        evaluation_seed=int(seeds["evaluation"]),
        run_id=STEP6C_RUN_ID,
        use_final_turn_cache=True,
    )
    diagnostic = evaluate_batch(
        [
            t3_abr_component_request(observation, config=config)
            for observation in observations
        ],
        library=diagnostic_library,
    )
    return accepted, diagnostic


def _validate_observations(
    values: Sequence[ActorObservation],
) -> tuple[ActorObservation, ActorObservation]:
    if len(values) != 2:
        raise ValueError("ABR root generator must return exactly two roots")
    checked: list[ActorObservation] = []
    for value, seat in zip(values, ("first", "second"), strict=True):
        if (
            not isinstance(value, ActorObservation)
            or value.street != "T3"
            or value.seat != seat
            or value.to_act_order != seat
        ):
            raise ValueError("ABR root generator returned an invalid seat root")
        checked.append(ActorObservation.from_dict(value.to_dict()))
    if checked[0].fingerprint() == checked[1].fingerprint():
        raise ValueError("ABR paired roots share one observation fingerprint")
    return checked[0], checked[1]


def _legal_contract(
    observation: ActorObservation,
) -> tuple[list[str], str, str, str]:
    native_actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    actions = canonicalize_actions(native_actions)
    return (
        [action_key(action).to_token() for action in actions],
        legal_action_set_digest(actions),
        ordered_action_mapping_digest(native_actions),
        ordered_action_mapping_digest(actions),
    )


def _component_values(value: Mapping[str, Any]) -> dict[str, float | int]:
    if set(value) != _COMPONENT_KEYS:
        raise ValueError("ABR terminal component fields changed")
    result: dict[str, float | int] = {}
    for key in sorted(_COMPONENT_KEYS - {"future_count"}):
        result[key] = _finite(value[key], f"ABR terminal component {key}")
    future_count = value["future_count"]
    if (
        isinstance(future_count, bool)
        or not isinstance(future_count, int)
        or future_count <= 0
    ):
        raise ValueError("ABR terminal component future_count changed")
    result["future_count"] = future_count
    for key in (
        "hero_bust_rate",
        "opponent_bust_rate",
        "hero_scoop_rate",
        "opponent_scoop_rate",
    ):
        if not 0.0 <= float(result[key]) <= 1.0:
            raise ValueError(f"ABR terminal rate is out of range: {key}")
    return result


def family_values_from_components(
    components: Mapping[str, Any],
) -> dict[str, float]:
    normalized = _component_values(components)
    values: dict[str, float] = {}
    for definition in FAMILY_REWARD_DEFINITIONS:
        value = sum(
            float(normalized[name]) * float(weight)
            for name, weight in definition["formula"].items()
        )
        if not math.isfinite(value):
            raise ValueError("ABR family terminal utility is nonfinite")
        values[str(definition["response_id"])] = value
    return values


def _parity_and_label_row(
    *,
    observation: ActorObservation,
    accepted: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    pair: Mapping[str, Any],
    seat: str,
    root_index: int,
) -> dict[str, Any]:
    step6d._reject_hidden(accepted, "accepted_decision")
    step6d._reject_hidden(diagnostic, "diagnostic_decision")
    if set(accepted) != _ACCEPTED_DECISION_KEYS:
        raise ValueError("accepted ABR teacher decision fields changed")
    if set(diagnostic) != _DIAGNOSTIC_DECISION_KEYS:
        raise ValueError("diagnostic ABR teacher decision fields changed")
    (
        legal_tokens,
        set_digest,
        native_order_digest,
        canonical_order_digest,
    ) = _legal_contract(observation)
    seeds = pair["seeds"]
    budget = pair["budget"]
    if (
        accepted.get("schema") != HU_M31_T3_RUNTIME_SCHEMA
        or accepted.get("runtime_id") != HU_M31_T3_RUNTIME_ID
        or accepted.get("value_scope")
        != "q_pi_uniform_exchangeable_t3_crn_with_exact_t4_children"
        or accepted.get("solver_id") != "rust_crn_sequential_t3_v1"
        or accepted.get("engine_version") != HU_M31_T3_ENGINE_VERSION
        or accepted.get("observation_fingerprint") != observation.fingerprint()
        or accepted.get("seat") != seat
        or accepted.get("legal_action_set_digest") != set_digest
        or accepted.get("legal_action_order_digest") != native_order_digest
        or accepted.get("native_library_sha256")
        != ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or accepted.get("teacher_value_status")
        != "diagnostic_not_match_EV"
        or accepted.get("run_id") != STEP6C_RUN_ID
        or accepted.get("candidate_samples")
        != budget["candidate_samples"]
        or accepted.get("evaluation_samples")
        != budget["evaluation_samples"]
        or accepted.get("downstream_t3_samples")
        != budget["downstream_t3_samples"]
        or accepted.get("downstream_t4_samples") != 0
        or accepted.get("continuation_seed") != seeds["child"]
        or accepted.get("candidate_seed") != seeds["candidate"]
        or accepted.get("evaluation_seed") != seeds["evaluation"]
        or not _is_sha256(accepted.get("candidate_rng_digest"))
        or not _is_sha256(accepted.get("evaluation_rng_digest"))
        or accepted.get("candidate_rng_digest")
        == accepted.get("evaluation_rng_digest")
    ):
        raise ValueError("accepted ABR teacher decision contract changed")
    if (
        diagnostic.get("schema") != ABR_DIAGNOSTIC_RESULT_SCHEMA
        or diagnostic.get("status") != "ok"
        or diagnostic.get("solver_id")
        != "rust_crn_sequential_t3_abr_components_v1"
        or diagnostic.get("legacy_solver_id")
        != "rust_crn_sequential_t3_v1"
        or diagnostic.get("kind") != "t3_abr_components"
        or diagnostic.get("street") != "T3"
        or diagnostic.get("seat") != seat
        or diagnostic.get("to_act_order") != seat
        or diagnostic.get("observation_fingerprint")
        != observation.fingerprint()
        or diagnostic.get("legal_action_count") != len(legal_tokens)
        or diagnostic.get("legal_action_set_digest") != set_digest
        or diagnostic.get("legal_action_order_digest")
        != native_order_digest
        or diagnostic.get("terminal_component_schema")
        != ABR_TERMINAL_COMPONENT_SCHEMA
        or diagnostic.get("sample_independence")
        != "disjoint_particle_rng_keys"
        or diagnostic.get("teacher_value_status")
        != "diagnostic_not_match_EV"
    ):
        raise ValueError("diagnostic ABR teacher decision contract changed")
    candidate_belief = diagnostic.get("candidate_belief")
    evaluation_belief = diagnostic.get("evaluation_belief")
    continuation = diagnostic.get("continuation_policy")
    if (
        not isinstance(candidate_belief, Mapping)
        or not isinstance(evaluation_belief, Mapping)
        or set(candidate_belief) != _BELIEF_KEYS
        or set(evaluation_belief) != _BELIEF_KEYS
        or candidate_belief.get("base_seed") != seeds["candidate"]
        or evaluation_belief.get("base_seed") != seeds["evaluation"]
        or candidate_belief.get("run_id")
        != f"{STEP6C_RUN_ID}:candidate_selection"
        or evaluation_belief.get("run_id")
        != f"{STEP6C_RUN_ID}:locked_evaluation"
        or candidate_belief.get("sample_count")
        != budget["candidate_samples"]
        or evaluation_belief.get("sample_count")
        != budget["evaluation_samples"]
        or candidate_belief.get("observation_fingerprint")
        != observation.fingerprint()
        or evaluation_belief.get("observation_fingerprint")
        != observation.fingerprint()
        or not isinstance(continuation, Mapping)
        or set(continuation) != _CONTINUATION_KEYS
        or continuation.get("downstream_t3_samples")
        != budget["downstream_t3_samples"]
        or continuation.get("downstream_t4_samples") != 0
    ):
        raise ValueError("diagnostic ABR belief/continuation contract changed")
    candidate_rng = diagnostic.get("candidate_rng_key_digests")
    evaluation_rng = diagnostic.get("evaluation_rng_key_digests")
    if (
        not isinstance(candidate_rng, list)
        or not isinstance(evaluation_rng, list)
        or len(candidate_rng) != budget["candidate_samples"]
        or len(evaluation_rng) != budget["evaluation_samples"]
        or not all(_is_sha256(value) for value in candidate_rng + evaluation_rng)
        or set(candidate_rng) & set(evaluation_rng)
    ):
        raise ValueError("ABR candidate/evaluation RNG streams overlap or changed")
    accepted_rows = accepted.get("action_values")
    diagnostic_rows = diagnostic.get("actions")
    if (
        not isinstance(accepted_rows, list)
        or not isinstance(diagnostic_rows, list)
        or len(accepted_rows) != len(legal_tokens)
        or len(diagnostic_rows) != len(legal_tokens)
    ):
        raise ValueError("ABR teacher does not cover every legal action")
    accepted_by_key = {str(row.get("action_key")): row for row in accepted_rows}
    diagnostic_by_key = {
        str(row.get("action_key")): row for row in diagnostic_rows
    }
    if (
        set(accepted_by_key) != set(legal_tokens)
        or set(diagnostic_by_key) != set(legal_tokens)
        or len(accepted_by_key) != len(legal_tokens)
        or len(diagnostic_by_key) != len(legal_tokens)
    ):
        raise ValueError("ABR ActionKey mapping drifted")
    if any(set(row) != _ACCEPTED_ACTION_KEYS for row in accepted_rows):
        raise ValueError("accepted ABR action row fields changed")
    if any(set(row) != _DIAGNOSTIC_ACTION_KEYS for row in diagnostic_rows):
        raise ValueError("diagnostic ABR action row fields changed")
    native_actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    native_contract = {
        action_key(action).to_token(): {
            "original_index": index,
            "placements": [list(item) for item in action.placements],
            "discards": list(action.discards),
        }
        for index, action in enumerate(native_actions)
    }
    family_action_values = {response_id: [] for response_id in RESPONSE_IDS}
    action_components: list[dict[str, Any]] = []
    for token in legal_tokens:
        ActionKey.from_token(token)
        accepted_row = accepted_by_key[token]
        diagnostic_row = diagnostic_by_key[token]
        expected_action = native_contract[token]
        if (
            accepted_row.get("original_index")
            != expected_action["original_index"]
            or diagnostic_row.get("original_index")
            != expected_action["original_index"]
            or accepted_row.get("placements")
            != expected_action["placements"]
            or diagnostic_row.get("placements")
            != expected_action["placements"]
            or accepted_row.get("discards") != expected_action["discards"]
            or diagnostic_row.get("discards") != expected_action["discards"]
        ):
            raise ValueError("ABR native action index/payload mapping drifted")
        accepted_selection = _finite(
            accepted_row.get("selection_ev"), "accepted selection Q"
        )
        accepted_evaluation = _finite(
            accepted_row.get("evaluation_ev"), "accepted evaluation Q"
        )
        diagnostic_selection = _finite(
            diagnostic_row.get("selection_score"),
            "diagnostic selection Q",
        )
        diagnostic_evaluation = _finite(
            diagnostic_row.get("score"), "diagnostic evaluation Q"
        )
        if (
            accepted_selection != diagnostic_selection
            or accepted_evaluation != diagnostic_evaluation
            or diagnostic_row.get("joint_ev") != diagnostic_evaluation
            or diagnostic_row.get("selection_future_count")
            != budget["candidate_samples"]
            or diagnostic_row.get("evaluation_future_count")
            != budget["evaluation_samples"]
        ):
            raise ValueError(
                "diagnostic component engine is not bit-exact with accepted Q"
            )
        components = _component_values(
            diagnostic_row.get("terminal_components", {})
        )
        if float(components["hu_score_mean"]) != accepted_evaluation:
            raise ValueError("terminal HU component does not equal accepted Q")
        if components["future_count"] != budget["evaluation_samples"]:
            raise ValueError("terminal component evaluation count changed")
        family_values = family_values_from_components(components)
        for response_id in RESPONSE_IDS:
            family_action_values[response_id].append(
                family_values[response_id]
            )
        action_components.append(
            {
                "action_key": token,
                "terminal_components": components,
                "family_values": family_values,
            }
        )
    ranked_tokens = sorted(
        legal_tokens,
        key=lambda token: (
            -float(accepted_by_key[token]["selection_ev"]),
            ActionKey.from_token(token).sort_key(),
        ),
    )
    for rank, token in enumerate(ranked_tokens):
        if (
            accepted_by_key[token].get("rank") != rank
            or diagnostic_by_key[token].get("sorted_index") != rank
        ):
            raise ValueError("ABR action rank mapping drifted")
    expected_selected = ranked_tokens[0]
    second_selection = float(
        accepted_by_key[
            ranked_tokens[1] if len(ranked_tokens) > 1 else expected_selected
        ]["selection_ev"]
    )
    selected_selection = float(
        accepted_by_key[expected_selected]["selection_ev"]
    )
    evaluation_best = max(
        float(accepted_by_key[token]["evaluation_ev"])
        for token in legal_tokens
    )
    selected_evaluation = float(
        accepted_by_key[expected_selected]["evaluation_ev"]
    )
    for token in legal_tokens:
        expected_regret = (
            evaluation_best
            - float(accepted_by_key[token]["evaluation_ev"])
        )
        if (
            accepted_by_key[token].get("evaluation_regret")
            != expected_regret
            or diagnostic_by_key[token].get(
                "evaluation_regret_vs_sample_best"
            )
            != expected_regret
            or diagnostic_by_key[token].get(
                "selected_by_candidate_plan"
            )
            is not (token == expected_selected)
        ):
            raise ValueError("ABR action regret/selection mapping drifted")
    if (
        accepted.get("selected_action_key")
        != diagnostic.get("selected_action_key")
        or accepted.get("selected_action_key") != expected_selected
        or accepted.get("selected_selection_ev")
        != selected_selection
        or accepted.get("selected_evaluation_ev")
        != selected_evaluation
        or accepted.get("selected_action")
        != {
            "placements": native_contract[expected_selected]["placements"],
            "discards": native_contract[expected_selected]["discards"],
        }
        or diagnostic.get("selected_action_original_index")
        != native_contract[expected_selected]["original_index"]
        or diagnostic.get("best_action_original_index")
        != native_contract[expected_selected]["original_index"]
        or diagnostic.get("selected_action_evaluation_score")
        != selected_evaluation
        or diagnostic.get("best_score") != selected_evaluation
        or accepted.get("selection_gap")
        != selected_selection - second_selection
        or diagnostic.get("selection_score_gap")
        != selected_selection - second_selection
        or diagnostic.get("score_gap")
        != selected_selection - second_selection
        or accepted.get("evaluation_sample_regret")
        != evaluation_best - selected_evaluation
        or diagnostic.get(
            "evaluation_sample_regret_of_locked_selection"
        )
        != evaluation_best - selected_evaluation
    ):
        raise ValueError("diagnostic selected ActionKey/Q parity failed")
    observation_payload = observation.to_dict()
    return {
        "root_index": root_index,
        "seat": seat,
        "example_id": f"abr-dev-{root_index // 2:04d}-{seat}",
        "observation": observation_payload,
        "observation_fingerprint": observation.fingerprint(),
        "observation_sha256": canonical_sha256(observation_payload),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_keys": legal_tokens,
        "legal_action_set_digest": set_digest,
        "native_legal_action_order_digest": native_order_digest,
        "legal_action_order_digest": canonical_order_digest,
        "family_action_values": family_action_values,
        "action_terminal_components": action_components,
        "accepted_decision": deepcopy(dict(accepted)),
        "diagnostic_decision": deepcopy(dict(diagnostic)),
        "legacy_q_bit_exact": True,
        "candidate_evaluation_rng_disjoint": True,
    }


def build_pair_evidence(
    *,
    plan: Mapping[str, Any],
    pair: Mapping[str, Any],
    observations: Sequence[ActorObservation],
    accepted_decisions: Sequence[Mapping[str, Any]],
    diagnostic_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated_plan = validate_plan(plan)
    expected_pair = validated_plan["pairs"][pair["pair_index"]]
    if dict(pair) != expected_pair:
        raise ValueError("ABR pair contract changed")
    roots = _validate_observations(observations)
    if len(accepted_decisions) != 2 or len(diagnostic_decisions) != 2:
        raise ValueError("ABR pair search must return two decisions per engine")
    rows = [
        _parity_and_label_row(
            observation=observation,
            accepted=dict(accepted),
            diagnostic=dict(diagnostic),
            pair=pair,
            seat=seat,
            root_index=root_index,
        )
        for observation, accepted, diagnostic, seat, root_index in zip(
            roots,
            accepted_decisions,
            diagnostic_decisions,
            ("first", "second"),
            pair["root_indices"],
            strict=True,
        )
    ]
    identity = {
        "schema": ABR_TEACHER_PAIR_SCHEMA,
        "status": "complete_real_terminal_component_labels",
        "plan_sha256": canonical_sha256(validated_plan),
        "pair_contract": deepcopy(dict(pair)),
        "pair_contract_sha256": canonical_sha256(pair),
        "pair_index": pair["pair_index"],
        "root_indices": list(pair["root_indices"]),
        "profile": pair["profile"],
        "seeds": deepcopy(dict(pair["seeds"])),
        "rows": rows,
        "seat_counts": {"first": 1, "second": 1},
        "legacy_q_bit_exact_root_count": 2,
        "family_reward_sha256": FAMILY_REWARD_SHA256,
        "synthetic_values_used": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "teacher_values_are_realized_locked_match_ev": False,
        "current_profile_changed": False,
    }
    value = {
        **identity,
        "pair_evidence_identity_sha256": canonical_sha256(identity),
    }
    step6d._reject_hidden(value, "abr_pair_evidence")
    return value


def validate_pair_evidence(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    pair_index: int,
) -> dict[str, Any]:
    evidence = deepcopy(dict(value))
    validated_plan = validate_plan(plan)
    if not 0 <= pair_index < validated_plan["pair_count"]:
        raise ValueError("ABR pair evidence index is out of plan")
    pair = validated_plan["pairs"][pair_index]
    if (
        evidence.get("schema") != ABR_TEACHER_PAIR_SCHEMA
        or evidence.get("pair_index") != pair_index
        or evidence.get("pair_contract") != pair
    ):
        raise ValueError("ABR pair evidence identity changed")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("ABR pair evidence rows changed")
    observations = [
        ActorObservation.from_dict(row["observation"]) for row in rows
    ]
    rebuilt = build_pair_evidence(
        plan=validated_plan,
        pair=pair,
        observations=observations,
        accepted_decisions=[row["accepted_decision"] for row in rows],
        diagnostic_decisions=[row["diagnostic_decision"] for row in rows],
    )
    if evidence != rebuilt:
        raise ValueError("ABR pair evidence changed")
    return evidence


def pair_evidence_path(run_directory: str | Path, pair_index: int) -> Path:
    return Path(run_directory) / PAIR_DIRECTORY / f"pair-{pair_index:06d}.json"


def _expected_layout_names(plan: Mapping[str, Any]) -> set[str]:
    return {
        f"pair-{index:06d}.json" for index in plan["pair_indices"]
    }


def _scan_pair_files(
    run_directory: str | Path,
    *,
    plan: Mapping[str, Any],
) -> dict[int, Path]:
    directory = Path(run_directory) / PAIR_DIRECTORY
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("ABR pair evidence directory is unsafe")
    expected = _expected_layout_names(plan)
    observed: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ValueError(f"unknown or unsafe ABR pair artifact: {path.name}")
        index = int(path.stem.split("-")[1])
        if index in observed:
            raise ValueError("duplicate ABR pair evidence")
        observed[index] = path
    return observed


def _validate_evidence_collection(
    evidence: Mapping[int, Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> None:
    if any(index not in plan["pair_indices"] for index in evidence):
        raise ValueError("ABR evidence collection contains an extra pair")
    fingerprints: set[str] = set()
    example_ids: set[str] = set()
    for index, pair in sorted(evidence.items()):
        if pair["pair_index"] != index:
            raise ValueError("ABR evidence collection index changed")
        for row, seat in zip(pair["rows"], ("first", "second"), strict=True):
            fingerprint = str(row["observation_fingerprint"])
            example_id = str(row["example_id"])
            if (
                row["seat"] != seat
                or fingerprint in fingerprints
                or example_id in example_ids
            ):
                raise ValueError(
                    "ABR evidence collection has duplicate/unbalanced roots"
                )
            fingerprints.add(fingerprint)
            example_ids.add(example_id)
    if len(fingerprints) != len(evidence) * 2:
        raise ValueError("ABR evidence collection root coverage changed")


def validate_run_layout(
    run_directory: str | Path,
    *,
    plan: Mapping[str, Any],
    allow_outputs: bool = True,
) -> dict[int, Path]:
    root = Path(run_directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("ABR teacher run directory is unsafe")
    known = {
        PLAN_FILE,
        PAIR_DIRECTORY,
        DONE_FILE,
        RAW_EXAMPLES_FILE,
        TEACHER_RECEIPT_FILE,
    }
    if not allow_outputs:
        known = {PLAN_FILE, PAIR_DIRECTORY}
    unknown = {path.name for path in root.iterdir()} - known
    if unknown:
        raise ValueError(f"unknown ABR teacher run artifacts: {sorted(unknown)}")
    return _scan_pair_files(root, plan=plan)


def _done_value(
    *,
    plan: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    digests = [
        str(row["pair_evidence_identity_sha256"]) for row in evidence
    ]
    identity = {
        "schema": ABR_TEACHER_DONE_SCHEMA,
        "status": "complete_all_planned_pairs_source_replayed",
        "plan_sha256": canonical_sha256(plan),
        "pair_count": plan["pair_count"],
        "state_count": plan["state_count"],
        "pair_indices": list(plan["pair_indices"]),
        "pair_evidence_identity_sha256s": digests,
        "pair_evidence_aggregate_sha256": canonical_sha256(digests),
        "seat_counts": {
            "first": plan["pair_count"],
            "second": plan["pair_count"],
        },
        "legacy_q_bit_exact_root_count": plan["state_count"],
        "paired_count_per_response": {
            response_id: plan["pair_count"] for response_id in RESPONSE_IDS
        },
        "root_count_per_response": {
            response_id: plan["state_count"] for response_id in RESPONSE_IDS
        },
        "family_reward_sha256": FAMILY_REWARD_SHA256,
        "synthetic_values_used": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "current_profile_changed": False,
    }
    return {
        **identity,
        "done_identity_sha256": canonical_sha256(identity),
    }


def validate_done(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    done = deepcopy(dict(value))
    if done != _done_value(plan=validate_plan(plan), evidence=evidence):
        raise ValueError("ABR teacher DONE receipt changed")
    return done


def run_pairs(
    *,
    run_directory: str | Path,
    accepted_library_path: str | Path | None = None,
    diagnostic_library_path: str | Path | None = None,
    max_new_pairs: int | None = None,
    pair_indices: Sequence[int] | None = None,
    root_generator: RootGenerator | None = None,
    pair_search_adapter: PairSearchAdapter | None = None,
    bundle: ModelBundle | None = None,
) -> dict[str, Any]:
    root = Path(run_directory)
    plan = validate_plan(_canonical_read(root / PLAN_FILE, "ABR teacher plan"))
    if (
        isinstance(max_new_pairs, bool)
        or (
            max_new_pairs is not None
            and (not isinstance(max_new_pairs, int) or max_new_pairs < 0)
        )
    ):
        raise ValueError("max_new_pairs must be a nonnegative integer or None")
    selected_indices = (
        list(plan["pair_indices"])
        if pair_indices is None
        else list(pair_indices)
    )
    if (
        any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in plan["pair_indices"]
            for index in selected_indices
        )
        or len(set(selected_indices)) != len(selected_indices)
    ):
        raise ValueError(
            "pair_indices must be unique integer indices from the frozen plan"
        )
    selected_set = set(selected_indices)
    accepted_path = (
        None
        if accepted_library_path is None
        else Path(accepted_library_path).resolve()
    )
    diagnostic_path = (
        None
        if diagnostic_library_path is None
        else Path(diagnostic_library_path).resolve()
    )
    if pair_search_adapter is None:
        if (
            accepted_path is None
            or diagnostic_path is None
            or accepted_path.is_symlink()
            or diagnostic_path.is_symlink()
            or _sha256_file(accepted_path)
            != plan["accepted_library_sha256"]
            or _sha256_file(diagnostic_path)
            != plan["diagnostic_library_sha256"]
        ):
            raise ValueError("ABR teacher library provenance changed")
    generator = root_generator or _default_root_generator
    search = pair_search_adapter or _default_pair_search
    if bundle is None and root_generator is None:
        bundle = _default_bundle()
    files = validate_run_layout(root, plan=plan)
    validated: dict[int, dict[str, Any]] = {}
    for index, path in sorted(files.items()):
        validated[index] = validate_pair_evidence(
            _canonical_read(path, "ABR pair evidence"),
            plan=plan,
            pair_index=index,
        )
    _validate_evidence_collection(validated, plan=plan)
    if (root / DONE_FILE).exists():
        if set(validated) != set(plan["pair_indices"]):
            raise ValueError("ABR DONE exists before all pair evidence")
        done = validate_done(
            _canonical_read(root / DONE_FILE, "ABR teacher DONE"),
            plan=plan,
            evidence=[validated[index] for index in plan["pair_indices"]],
        )
        return {
            "status": "resume_complete",
            "completed_pairs": len(validated),
            "remaining_pairs": 0,
            "new_pairs": 0,
            "done_identity_sha256": done["done_identity_sha256"],
        }
    allowance = len(selected_indices) if max_new_pairs is None else max_new_pairs
    new_pairs = 0
    for pair in plan["pairs"]:
        index = int(pair["pair_index"])
        if (
            index not in selected_set
            or index in validated
            or new_pairs >= allowance
        ):
            continue
        observations = _validate_observations(generator(pair, bundle))
        accepted, diagnostic = search(
            observations,
            pair,
            accepted_path,
            diagnostic_path,
        )
        evidence = build_pair_evidence(
            plan=plan,
            pair=pair,
            observations=observations,
            accepted_decisions=accepted,
            diagnostic_decisions=diagnostic,
        )
        path = pair_evidence_path(root, index)
        _write_once(path, evidence)
        stored = validate_pair_evidence(
            _canonical_read(path, "stored ABR pair evidence"),
            plan=plan,
            pair_index=index,
        )
        validated[index] = stored
        _validate_evidence_collection(validated, plan=plan)
        new_pairs += 1
    remaining = plan["pair_count"] - len(validated)
    done_identity = None
    if remaining == 0:
        ordered = [validated[index] for index in plan["pair_indices"]]
        done = _done_value(plan=plan, evidence=ordered)
        _write_once(root / DONE_FILE, done)
        validate_done(
            _canonical_read(root / DONE_FILE, "stored ABR teacher DONE"),
            plan=plan,
            evidence=ordered,
        )
        done_identity = done["done_identity_sha256"]
    return {
        "status": "complete" if remaining == 0 else "bounded_pause",
        "completed_pairs": len(validated),
        "remaining_pairs": remaining,
        "new_pairs": new_pairs,
        "selected_pair_indices": selected_indices,
        "done_identity_sha256": done_identity,
    }


def merge_runs(
    *,
    run_directory: str | Path,
    source_run_directories: Sequence[str | Path],
) -> dict[str, Any]:
    """Merge immutable, independently generated shards into one frozen run."""

    if not source_run_directories:
        raise ValueError("at least one ABR teacher source run is required")
    root = Path(run_directory)
    plan = validate_plan(_canonical_read(root / PLAN_FILE, "ABR teacher plan"))
    destination_files = validate_run_layout(root, plan=plan)
    validated: dict[int, dict[str, Any]] = {
        index: validate_pair_evidence(
            _canonical_read(path, "ABR destination pair evidence"),
            plan=plan,
            pair_index=index,
        )
        for index, path in sorted(destination_files.items())
    }
    _validate_evidence_collection(validated, plan=plan)
    if (root / DONE_FILE).exists():
        if set(validated) != set(plan["pair_indices"]):
            raise ValueError("ABR DONE exists before all pair evidence")
        done = validate_done(
            _canonical_read(root / DONE_FILE, "ABR teacher DONE"),
            plan=plan,
            evidence=[validated[index] for index in plan["pair_indices"]],
        )
        return {
            "status": "resume_complete",
            "source_run_count": len(source_run_directories),
            "completed_pairs": len(validated),
            "remaining_pairs": 0,
            "new_pairs": 0,
            "done_identity_sha256": done["done_identity_sha256"],
        }

    merged = 0
    seen_sources: set[Path] = set()
    for source_value in source_run_directories:
        source = Path(source_value).resolve()
        if source == root.resolve() or source in seen_sources:
            raise ValueError("ABR teacher source run is duplicate or destination")
        seen_sources.add(source)
        source_plan = validate_plan(
            _canonical_read(source / PLAN_FILE, "ABR source teacher plan")
        )
        if source_plan != plan:
            raise ValueError("ABR teacher source plan changed")
        source_files = validate_run_layout(source, plan=source_plan)
        for index, path in sorted(source_files.items()):
            evidence = validate_pair_evidence(
                _canonical_read(path, "ABR source pair evidence"),
                plan=plan,
                pair_index=index,
            )
            destination = pair_evidence_path(root, index)
            existed = destination.exists()
            _write_once(destination, evidence)
            stored = validate_pair_evidence(
                _canonical_read(destination, "merged ABR pair evidence"),
                plan=plan,
                pair_index=index,
            )
            if index in validated and validated[index] != stored:
                raise ValueError("merged ABR pair evidence changed")
            validated[index] = stored
            if not existed:
                merged += 1
        _validate_evidence_collection(validated, plan=plan)

    remaining = plan["pair_count"] - len(validated)
    done_identity = None
    if remaining == 0:
        ordered = [validated[index] for index in plan["pair_indices"]]
        done = _done_value(plan=plan, evidence=ordered)
        _write_once(root / DONE_FILE, done)
        validate_done(
            _canonical_read(root / DONE_FILE, "stored ABR teacher DONE"),
            plan=plan,
            evidence=ordered,
        )
        done_identity = done["done_identity_sha256"]
    return {
        "status": "complete" if remaining == 0 else "bounded_pause",
        "source_run_count": len(source_run_directories),
        "completed_pairs": len(validated),
        "remaining_pairs": remaining,
        "new_pairs": merged,
        "done_identity_sha256": done_identity,
    }


def status(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory)
    plan = validate_plan(_canonical_read(root / PLAN_FILE, "ABR teacher plan"))
    files = validate_run_layout(root, plan=plan)
    evidence = {
        index: validate_pair_evidence(
            _canonical_read(path, "ABR pair evidence"),
            plan=plan,
            pair_index=index,
        )
        for index, path in sorted(files.items())
    }
    _validate_evidence_collection(evidence, plan=plan)
    done = None
    if (root / DONE_FILE).exists():
        if set(evidence) != set(plan["pair_indices"]):
            raise ValueError("ABR DONE exists before all pair evidence")
        done = validate_done(
            _canonical_read(root / DONE_FILE, "ABR teacher DONE"),
            plan=plan,
            evidence=[evidence[index] for index in plan["pair_indices"]],
        )
    return {
        "schema": "hu_m31_t3_abr_teacher_status_v1",
        "status": "complete" if done is not None else "in_progress",
        "pair_count": plan["pair_count"],
        "state_count": plan["state_count"],
        "completed_pairs": len(evidence),
        "completed_states": len(evidence) * 2,
        "remaining_pairs": plan["pair_count"] - len(evidence),
        "first_seat_roots": len(evidence),
        "second_seat_roots": len(evidence),
        "legacy_q_bit_exact_roots": len(evidence) * 2,
        "done_identity_sha256": (
            None if done is None else done["done_identity_sha256"]
        ),
        "cloud_resources_started": False,
        "current_profile_changed": False,
    }


def _raw_examples_from_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for pair in evidence:
        source_seed = int(pair["seeds"]["hand"])
        for row in pair["rows"]:
            examples.append(
                {
                    "example_id": row["example_id"],
                    "source_seed": source_seed,
                    "observation": deepcopy(row["observation"]),
                    "family_action_values": deepcopy(
                        row["family_action_values"]
                    ),
                }
            )
    return examples


def finalize_run(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory)
    plan = validate_plan(_canonical_read(root / PLAN_FILE, "ABR teacher plan"))
    files = validate_run_layout(root, plan=plan)
    if set(files) != set(plan["pair_indices"]):
        raise ValueError("ABR teacher cannot finalize with missing pairs")
    evidence = [
        validate_pair_evidence(
            _canonical_read(files[index], "ABR pair evidence"),
            plan=plan,
            pair_index=index,
        )
        for index in plan["pair_indices"]
    ]
    _validate_evidence_collection(
        {int(row["pair_index"]): row for row in evidence},
        plan=plan,
    )
    done = validate_done(
        _canonical_read(root / DONE_FILE, "ABR teacher DONE"),
        plan=plan,
        evidence=evidence,
    )
    from .hu_m31_t3_abr_cli_v1 import build_raw_examples_document

    raw_document = build_raw_examples_document(
        _raw_examples_from_evidence(evidence)
    )
    _write_once(root / RAW_EXAMPLES_FILE, raw_document)
    raw_sha = _sha256_file(root / RAW_EXAMPLES_FILE)
    pair_digests = [
        row["pair_evidence_identity_sha256"] for row in evidence
    ]
    identity = {
        "schema": ABR_TEACHER_RECEIPT_SCHEMA,
        "status": (
            "complete_real_teacher_production_coverage"
            if plan["production_coverage_complete"]
            else "complete_real_teacher_pilot_only"
        ),
        "plan_file_sha256": _sha256_file(root / PLAN_FILE),
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "done_file_sha256": _sha256_file(root / DONE_FILE),
        "done_identity_sha256": done["done_identity_sha256"],
        "raw_examples_filename": RAW_EXAMPLES_FILE,
        "raw_examples_file_sha256": raw_sha,
        "raw_examples_identity_sha256": raw_document[
            "raw_examples_identity_sha256"
        ],
        "pair_count": plan["pair_count"],
        "example_count": plan["state_count"],
        "seat_counts": {
            "first": plan["pair_count"],
            "second": plan["pair_count"],
        },
        "production_pair_count_required": MAX_DEVELOPMENT_PAIRS,
        "production_state_count_required": MAX_DEVELOPMENT_PAIRS * 2,
        "production_training_authorized": plan[
            "production_coverage_complete"
        ],
        "paired_count_per_response": {
            response_id: plan["pair_count"] for response_id in RESPONSE_IDS
        },
        "root_count_per_response": {
            response_id: plan["state_count"] for response_id in RESPONSE_IDS
        },
        "pair_evidence_identity_sha256s": pair_digests,
        "pair_evidence_aggregate_sha256": canonical_sha256(pair_digests),
        "accepted_library_sha256": plan["accepted_library_sha256"],
        "diagnostic_library_sha256": plan["diagnostic_library_sha256"],
        "legacy_q_bit_exact_root_count": plan["state_count"],
        "family_reward_sha256": FAMILY_REWARD_SHA256,
        "linear_expectation_of_terminal_rewards": True,
        "synthetic_values_used": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "teacher_values_are_realized_locked_match_ev": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "cloud_resources_started": False,
    }
    receipt = {
        **identity,
        "teacher_receipt_identity_sha256": canonical_sha256(identity),
    }
    _write_once(root / TEACHER_RECEIPT_FILE, receipt)
    validate_teacher_receipt_against_raw(
        receipt,
        raw_document=raw_document,
        raw_file_sha256=raw_sha,
    )
    return receipt


def validate_teacher_receipt_against_raw(
    value: Mapping[str, Any],
    *,
    raw_document: Mapping[str, Any],
    raw_file_sha256: str,
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if (
        receipt.get("schema") != ABR_TEACHER_RECEIPT_SCHEMA
        or receipt.get("status")
        not in {
            "complete_real_teacher_production_coverage",
            "complete_real_teacher_pilot_only",
        }
        or receipt.get("raw_examples_file_sha256") != raw_file_sha256
        or receipt.get("raw_examples_identity_sha256")
        != raw_document.get("raw_examples_identity_sha256")
        or receipt.get("example_count") != raw_document.get("example_count")
        or isinstance(receipt.get("pair_count"), bool)
        or not isinstance(receipt.get("pair_count"), int)
        or receipt.get("pair_count") * 2 != receipt.get("example_count")
        or receipt.get("seat_counts")
        != {
            "first": receipt.get("pair_count"),
            "second": receipt.get("pair_count"),
        }
        or receipt.get("paired_count_per_response")
        != {
            response_id: receipt.get("pair_count")
            for response_id in RESPONSE_IDS
        }
        or receipt.get("root_count_per_response")
        != {
            response_id: receipt.get("example_count")
            for response_id in RESPONSE_IDS
        }
        or receipt.get("legacy_q_bit_exact_root_count")
        != receipt.get("example_count")
        or receipt.get("family_reward_sha256") != FAMILY_REWARD_SHA256
        or receipt.get("linear_expectation_of_terminal_rewards") is not True
        or receipt.get("synthetic_values_used") is not False
        or receipt.get("opponent_private_discards_used") is not False
        or receipt.get("realized_deck_tail_used") is not False
        or receipt.get("current_profile_resolved") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("ABR production teacher receipt changed")
    identity = dict(receipt)
    declared = identity.pop("teacher_receipt_identity_sha256", None)
    if declared != canonical_sha256(identity):
        raise ValueError("ABR production teacher receipt identity changed")
    return receipt


def require_production_teacher_coverage(
    value: Mapping[str, Any],
    *,
    raw_document: Mapping[str, Any],
    raw_file_sha256: str,
) -> dict[str, Any]:
    """Reject pilots and every synthetic/incomplete production input."""

    receipt = validate_teacher_receipt_against_raw(
        value,
        raw_document=raw_document,
        raw_file_sha256=raw_file_sha256,
    )
    examples = raw_document.get("examples")
    if (
        receipt.get("status")
        != "complete_real_teacher_production_coverage"
        or receipt.get("production_training_authorized") is not True
        or receipt.get("pair_count") != MAX_DEVELOPMENT_PAIRS
        or receipt.get("example_count") != MAX_DEVELOPMENT_PAIRS * 2
        or receipt.get("production_pair_count_required")
        != MAX_DEVELOPMENT_PAIRS
        or receipt.get("production_state_count_required")
        != MAX_DEVELOPMENT_PAIRS * 2
        or receipt.get("seat_counts")
        != {
            "first": MAX_DEVELOPMENT_PAIRS,
            "second": MAX_DEVELOPMENT_PAIRS,
        }
        or receipt.get("paired_count_per_response")
        != {
            response_id: MAX_DEVELOPMENT_PAIRS
            for response_id in RESPONSE_IDS
        }
        or receipt.get("root_count_per_response")
        != {
            response_id: MAX_DEVELOPMENT_PAIRS * 2
            for response_id in RESPONSE_IDS
        }
        or not isinstance(examples, list)
        or len(examples) != MAX_DEVELOPMENT_PAIRS * 2
    ):
        raise ValueError(
            "ABR production requires exact 250-pair/500-root real-teacher coverage"
        )
    expected_seeds = {
        development_seed_values(index)["hand"]
        for index in range(MAX_DEVELOPMENT_PAIRS)
    }
    seed_seats: set[tuple[int, str]] = set()
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("ABR production example changed")
        observation = ActorObservation.from_dict(example["observation"])
        seed = example.get("source_seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in expected_seeds
            or observation.seat not in {"first", "second"}
            or set(example.get("family_action_values", {}))
            != set(RESPONSE_IDS)
        ):
            raise ValueError("ABR production root/family coverage changed")
        seed_seats.add((seed, observation.seat))
    expected_seed_seats = {
        (seed, seat)
        for seed in expected_seeds
        for seat in ("first", "second")
    }
    if seed_seats != expected_seed_seats:
        raise ValueError("ABR production paired-seat coverage changed")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate bounded real-terminal M3.1 ABR teacher labels"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-directory", required=True)
    prepare.add_argument("--pair-count", type=int, required=True)
    prepare.add_argument("--accepted-library", required=True)
    prepare.add_argument("--diagnostic-library", required=True)
    prepare.add_argument(
        "--expected-diagnostic-library-sha256", required=True
    )
    run = commands.add_parser("run")
    run.add_argument("--run-directory", required=True)
    run.add_argument("--accepted-library", required=True)
    run.add_argument("--diagnostic-library", required=True)
    run.add_argument("--max-new-pairs", type=int)
    run.add_argument("--pair-index", action="append", type=int)
    merge = commands.add_parser("merge")
    merge.add_argument("--run-directory", required=True)
    merge.add_argument(
        "--source-run-directory",
        action="append",
        required=True,
    )
    check = commands.add_parser("status")
    check.add_argument("--run-directory", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--run-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare_run(
            run_directory=args.run_directory,
            pair_count=args.pair_count,
            accepted_library_path=args.accepted_library,
            diagnostic_library_path=args.diagnostic_library,
            expected_diagnostic_library_sha256=(
                args.expected_diagnostic_library_sha256
            ),
        )
    elif args.command == "run":
        value = run_pairs(
            run_directory=args.run_directory,
            accepted_library_path=args.accepted_library,
            diagnostic_library_path=args.diagnostic_library,
            max_new_pairs=args.max_new_pairs,
            pair_indices=args.pair_index,
        )
    elif args.command == "merge":
        value = merge_runs(
            run_directory=args.run_directory,
            source_run_directories=args.source_run_directory,
        )
    elif args.command == "status":
        value = status(args.run_directory)
    else:
        value = finalize_run(args.run_directory)
    print(canonical_bytes(value).decode("ascii"), end="")
    return 0


__all__ = [
    "ABR_DIAGNOSTIC_RESULT_SCHEMA",
    "ABR_FAMILY_REWARD_SCHEMA",
    "ABR_TEACHER_DONE_SCHEMA",
    "ABR_TEACHER_PAIR_SCHEMA",
    "ABR_TEACHER_PLAN_SCHEMA",
    "ABR_TEACHER_RECEIPT_SCHEMA",
    "FAMILY_REWARD_DEFINITIONS",
    "FAMILY_REWARD_SHA256",
    "MAX_DEVELOPMENT_PAIRS",
    "MIN_PILOT_PAIRS",
    "build_pair_evidence",
    "build_plan",
    "development_seed_values",
    "family_values_from_components",
    "finalize_run",
    "main",
    "merge_runs",
    "pair_contract",
    "prepare_run",
    "run_pairs",
    "status",
    "validate_done",
    "validate_pair_evidence",
    "validate_plan",
    "validate_teacher_receipt_against_raw",
    "require_production_teacher_coverage",
]
