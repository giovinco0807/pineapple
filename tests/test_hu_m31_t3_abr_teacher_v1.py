from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ofc_regular import hu_m31_t3_abr_cli_v1 as abr_cli
from ofc_regular import hu_m31_t3_abr_teacher_v1 as subject
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_runtime import (
    HU_M31_T3_ENGINE_VERSION,
    HU_M31_T3_RUNTIME_ID,
    HU_M31_T3_RUNTIME_SCHEMA,
)
from ofc_regular.hu_m31_t3_step6d_contract import canonical_bytes
from ofc_regular.state import Board


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _board(cards: tuple[str, ...]) -> Board:
    return Board.from_rows(
        top=cards[: min(3, len(cards))],
        middle=cards[3 : min(8, len(cards))],
        bottom=cards[8:],
    )


def _observation(seat: str, offset: int) -> ActorObservation:
    cards = tuple(ALL_CARDS[offset:] + ALL_CARDS[:offset])
    opponent_count = 9 if seat == "first" else 11
    cursor = 0
    hero = cards[cursor : cursor + 9]
    cursor += 9
    opponent = cards[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = cards[cursor : cursor + 3]
    cursor += 3
    discards = cards[cursor : cursor + 2]
    return ActorObservation(
        hero_board=_board(hero),
        opponent_public_board=_board(opponent),
        dealt_cards=dealt,
        hero_private_discards=discards,
        seat=seat,  # type: ignore[arg-type]
        street="T3",
        to_act_order=seat,  # type: ignore[arg-type]
    )


def _roots(
    pair: Mapping[str, Any], bundle: object | None
) -> tuple[ActorObservation, ActorObservation]:
    del bundle
    offset = int(pair["pair_index"]) * 3
    return _observation("first", offset), _observation("second", offset + 1)


def _belief(
    observation: ActorObservation,
    *,
    seed: int,
    run_id: str,
    count: int,
    label: str,
) -> dict[str, Any]:
    return {
        "schema": "hu_hidden_card_particle_batch_v1",
        "belief_schema": "hu_hidden_card_belief_uniform_v1",
        "prior": "uniform_unknown_cards_without_replacement",
        "counter_rng_schema": "hu_counter_rng_v1",
        "observation_fingerprint": observation.fingerprint(),
        "street": "T3",
        "base_seed": seed,
        "run_id": run_id,
        "start_index": 0,
        "sample_count": count,
        "particle_digests": [
            _digest(f"{label}:particle:{index}") for index in range(count)
        ],
    }


def _decisions(
    observation: ActorObservation,
    pair: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    native_actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    canonical_actions = canonicalize_actions(native_actions)
    selection_by_key = {
        action_key(action).to_token(): float(len(canonical_actions) - index)
        for index, action in enumerate(canonical_actions)
    }
    evaluation_by_key = {
        token: value - 0.25 for token, value in selection_by_key.items()
    }
    ranked_native = sorted(
        range(len(native_actions)),
        key=lambda index: (
            -selection_by_key[action_key(native_actions[index]).to_token()],
            action_key(native_actions[index]).sort_key(),
        ),
    )
    selected_native = ranked_native[0]
    selected = action_key(native_actions[selected_native]).to_token()
    best_evaluation = max(evaluation_by_key.values())
    accepted_rows = []
    diagnostic_rows = []
    for rank, original_index in enumerate(ranked_native):
        action = native_actions[original_index]
        token = action_key(action).to_token()
        selection = selection_by_key[token]
        evaluation = evaluation_by_key[token]
        placements = [list(item) for item in action.placements]
        discards = list(action.discards)
        accepted_rows.append(
            {
                "original_index": original_index,
                "rank": rank,
                "action_key": token,
                "selection_ev": selection,
                "evaluation_ev": evaluation,
                "evaluation_regret": best_evaluation - evaluation,
                "placements": placements,
                "discards": discards,
            }
        )
        diagnostic_rows.append(
            {
                "original_index": original_index,
                "sorted_index": rank,
                "action_key": token,
                "placements": placements,
                "discards": discards,
                "score": evaluation,
                "joint_ev": evaluation,
                "selection_score": selection,
                "selected_by_candidate_plan": token == selected,
                "evaluation_regret_vs_sample_best": (
                    best_evaluation - evaluation
                ),
                "selection_future_count": pair["budget"][
                    "candidate_samples"
                ],
                "evaluation_future_count": pair["budget"][
                    "evaluation_samples"
                ],
                "terminal_components": {
                    "hu_score_mean": evaluation,
                    "hero_bust_rate": 0.0,
                    "opponent_bust_rate": (rank % 3) / 4.0,
                    "hero_scoop_rate": (rank % 2) / 4.0,
                    "opponent_scoop_rate": 0.0,
                    "hero_royalty_mean": float(rank % 5),
                    "opponent_royalty_mean": float(rank % 4),
                    "hero_fl_value_mean": 0.0,
                    "opponent_fl_value_mean": (
                        1.0 if rank % 7 == 0 else 0.0
                    ),
                    "future_count": pair["budget"]["evaluation_samples"],
                },
            }
        )
    seeds = pair["seeds"]
    budget = pair["budget"]
    native_set = legal_action_set_digest(native_actions)
    native_order = ordered_action_mapping_digest(native_actions)
    accepted = {
        "schema": HU_M31_T3_RUNTIME_SCHEMA,
        "runtime_id": HU_M31_T3_RUNTIME_ID,
        "seat": observation.seat,
        "value_scope": (
            "q_pi_uniform_exchangeable_t3_crn_with_exact_t4_children"
        ),
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": selected,
        "selected_selection_ev": selection_by_key[selected],
        "selected_evaluation_ev": evaluation_by_key[selected],
        "selection_gap": 1.0,
        "evaluation_sample_regret": 0.0,
        "selected_action": {
            "placements": [
                list(item)
                for item in native_actions[selected_native].placements
            ],
            "discards": list(native_actions[selected_native].discards),
        },
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": native_set,
        "legal_action_order_digest": native_order,
        "action_values": accepted_rows,
        "belief_prior": "uniform_unknown_cards_without_replacement",
        "candidate_belief_digest": _digest("candidate-belief"),
        "evaluation_belief_digest": _digest("evaluation-belief"),
        "candidate_rng_digest": _digest("candidate-rng"),
        "evaluation_rng_digest": _digest("evaluation-rng"),
        "candidate_samples": budget["candidate_samples"],
        "evaluation_samples": budget["evaluation_samples"],
        "downstream_t3_samples": budget["downstream_t3_samples"],
        "downstream_t4_samples": 0,
        "run_id": "hu-m31-step6c-production-label-pilot-v1",
        "continuation_seed": seeds["child"],
        "candidate_seed": seeds["candidate"],
        "evaluation_seed": seeds["evaluation"],
        "use_t4_action_cache": True,
        "continuation_policy_id": (
            "local_infoset_response_t3_second_t4_v1"
        ),
        "strategy_fusion_guard": (
            "child_actions_keyed_only_by_actor_observation"
        ),
        "search_contract_digest": _digest("search-contract"),
        "downstream_t4_native_semantics_id": "exact",
        "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
        "downstream_t4_mode": "exact",
        "child_information_set_count": 1,
        "solver_id": "rust_crn_sequential_t3_v1",
        "engine_version": HU_M31_T3_ENGINE_VERSION,
        "native_library_sha256": (
            "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
        ),
        "teacher_value_status": "diagnostic_not_match_EV",
        "native_latency_ms": 1.0,
        "validation_latency_ms": 1.0,
        "total_latency_ms": 2.0,
        "execution_mode": "batch_amortized",
        "batch_size": 2,
        "semantic_result_digest_schema": "semantic-v1",
        "semantic_result_digest_scope": "semantic",
        "semantic_result_digest": _digest("semantic"),
        "result_digest_scope": "result",
        "result_digest": _digest("result"),
    }
    candidate_rng = [
        _digest(f"candidate:{index}")
        for index in range(budget["candidate_samples"])
    ]
    evaluation_rng = [
        _digest(f"evaluation:{index}")
        for index in range(budget["evaluation_samples"])
    ]
    diagnostic = {
        "status": "ok",
        "schema": subject.ABR_DIAGNOSTIC_RESULT_SCHEMA,
        "engine_version": HU_M31_T3_ENGINE_VERSION,
        "solver_id": "rust_crn_sequential_t3_abr_components_v1",
        "legacy_solver_id": "rust_crn_sequential_t3_v1",
        "kind": "t3_abr_components",
        "street": "T3",
        "seat": observation.seat,
        "to_act_order": observation.seat,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": len(native_actions),
        "legal_action_set_digest": native_set,
        "legal_action_order_digest": native_order,
        "selected_action_original_index": selected_native,
        "best_action_original_index": selected_native,
        "selected_action_key": selected,
        "selected_action_evaluation_score": evaluation_by_key[selected],
        "best_score": evaluation_by_key[selected],
        "selection_score_gap": 1.0,
        "score_gap": 1.0,
        "evaluation_sample_best_score": best_evaluation,
        "evaluation_sample_regret_of_locked_selection": 0.0,
        "candidate_belief": _belief(
            observation,
            seed=seeds["candidate"],
            run_id=(
                "hu-m31-step6c-production-label-pilot-v1:"
                "candidate_selection"
            ),
            count=budget["candidate_samples"],
            label="candidate",
        ),
        "evaluation_belief": _belief(
            observation,
            seed=seeds["evaluation"],
            run_id=(
                "hu-m31-step6c-production-label-pilot-v1:"
                "locked_evaluation"
            ),
            count=budget["evaluation_samples"],
            label="evaluation",
        ),
        "candidate_rng_key_digests": candidate_rng,
        "evaluation_rng_key_digests": evaluation_rng,
        "sample_independence": "disjoint_particle_rng_keys",
        "continuation_policy": {
            "id": "local_infoset_response_t3_second_t4_v1",
            "downstream_t3_samples": budget["downstream_t3_samples"],
            "downstream_t4_samples": 0,
            "strategy_fusion_guard": (
                "child_actions_keyed_only_by_actor_observation"
            ),
        },
        "child_information_set_count": 1,
        "terminal_component_schema": subject.ABR_TERMINAL_COMPONENT_SCHEMA,
        "terminal_component_visibility": (
            "aggregate_only_no_sampled_cards_no_opponent_private_discards"
        ),
        "actions": diagnostic_rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    }
    return accepted, diagnostic


def _search(
    observations: Sequence[ActorObservation],
    pair: Mapping[str, Any],
    accepted_path: Path | None,
    diagnostic_path: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del accepted_path, diagnostic_path
    rows = [_decisions(observation, pair) for observation in observations]
    return [row[0] for row in rows], [row[1] for row in rows]


def _prepare_unit_run(root: Path) -> dict[str, Any]:
    plan = subject.build_plan(
        pair_count=subject.MIN_PILOT_PAIRS,
        diagnostic_library_sha256="d" * 64,
    )
    root.mkdir()
    (root / subject.PAIR_DIRECTORY).mkdir()
    (root / subject.PLAN_FILE).write_bytes(canonical_bytes(plan))
    return plan


def test_family_values_are_real_terminal_linear_utilities() -> None:
    components = {
        "hu_score_mean": 2.0,
        "hero_bust_rate": 0.0,
        "opponent_bust_rate": 0.25,
        "hero_scoop_rate": 0.5,
        "opponent_scoop_rate": 0.0,
        "hero_royalty_mean": 4.0,
        "opponent_royalty_mean": 3.0,
        "hero_fl_value_mean": 1.0,
        "opponent_fl_value_mean": 2.0,
        "future_count": 32,
    }
    values = subject.family_values_from_components(components)
    assert values == {
        "greedy_search_response": 2.0,
        "foul_pressure_response": 4.25,
        "royalty_denial_response": -0.5,
    }


def test_bounded_run_is_balanced_resumable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _prepare_unit_run(run)
    first = subject.run_pairs(
        run_directory=run,
        max_new_pairs=1,
        root_generator=_roots,
        pair_search_adapter=_search,
    )
    assert first["status"] == "bounded_pause"
    assert first["completed_pairs"] == 1
    observed = subject.status(run)
    assert observed["completed_states"] == 2
    assert observed["first_seat_roots"] == 1
    assert observed["second_seat_roots"] == 1
    assert observed["legacy_q_bit_exact_roots"] == 2
    paused = subject.run_pairs(
        run_directory=run,
        max_new_pairs=0,
        root_generator=_roots,
        pair_search_adapter=_search,
    )
    assert paused["new_pairs"] == 0
    pair_path = subject.pair_evidence_path(run, 0)
    pair = json.loads(pair_path.read_bytes())
    pair["rows"][0]["diagnostic_decision"]["actions"][0]["score"] += 1.0
    pair_path.write_bytes(canonical_bytes(pair))
    with pytest.raises(ValueError, match="bit-exact|changed"):
        subject.status(run)


def test_selected_pair_shards_merge_with_exact_plan_identity(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    for run in (destination, shard_a, shard_b):
        _prepare_unit_run(run)
    result_a = subject.run_pairs(
        run_directory=shard_a,
        pair_indices=[0, 2],
        root_generator=_roots,
        pair_search_adapter=_search,
    )
    result_b = subject.run_pairs(
        run_directory=shard_b,
        pair_indices=[1],
        root_generator=_roots,
        pair_search_adapter=_search,
    )
    assert result_a["new_pairs"] == 2
    assert result_a["selected_pair_indices"] == [0, 2]
    assert result_b["new_pairs"] == 1
    merged = subject.merge_runs(
        run_directory=destination,
        source_run_directories=[shard_a, shard_b],
    )
    assert merged["status"] == "bounded_pause"
    assert merged["completed_pairs"] == 3
    assert merged["new_pairs"] == 3
    assert subject.status(destination)["completed_states"] == 6
    resumed = subject.merge_runs(
        run_directory=destination,
        source_run_directories=[shard_a, shard_b],
    )
    assert resumed["new_pairs"] == 0


def test_candidate_and_evaluation_rng_overlap_fails_closed(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    plan = _prepare_unit_run(run)
    pair = plan["pairs"][0]
    observations = _roots(pair, None)
    accepted, diagnostic = _search(observations, pair, None, None)
    changed = deepcopy(diagnostic)
    changed[0]["evaluation_rng_key_digests"][0] = changed[0][
        "candidate_rng_key_digests"
    ][0]
    with pytest.raises(ValueError, match="RNG"):
        subject.build_pair_evidence(
            plan=plan,
            pair=pair,
            observations=observations,
            accepted_decisions=accepted,
            diagnostic_decisions=changed,
        )


def test_diagnostic_q_drift_and_hidden_truth_fail_closed(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    plan = _prepare_unit_run(run)
    pair = plan["pairs"][0]
    observations = _roots(pair, None)
    accepted, diagnostic = _search(observations, pair, None, None)
    q_drift = deepcopy(diagnostic)
    q_drift[1]["actions"][0]["score"] += 0.5
    with pytest.raises(ValueError, match="bit-exact"):
        subject.build_pair_evidence(
            plan=plan,
            pair=pair,
            observations=observations,
            accepted_decisions=accepted,
            diagnostic_decisions=q_drift,
        )
    hidden = deepcopy(diagnostic)
    hidden[0]["opponent_private_discards"] = ["As"]
    with pytest.raises(ValueError, match="forbidden hidden-information"):
        subject.build_pair_evidence(
            plan=plan,
            pair=pair,
            observations=observations,
            accepted_decisions=accepted,
            diagnostic_decisions=hidden,
        )


def test_production_normalization_rejects_pilot_receipt() -> None:
    observations = [_observation("first", 0), _observation("second", 1)]
    raw = abr_cli.build_raw_examples_document(
        [
            {
                "example_id": f"pilot-{observation.seat}",
                "source_seed": subject.development_seed_values(0)["hand"],
                "observation": observation.to_dict(),
                "family_action_values": {
                    response_id: [0.0]
                    * len(
                        canonicalize_actions(
                            generate_turn_actions(
                                observation.hero_board,
                                observation.dealt_cards,
                            )
                        )
                    )
                    for response_id in subject.RESPONSE_IDS
                },
            }
            for observation in observations
        ]
    )
    identity = {
        "schema": subject.ABR_TEACHER_RECEIPT_SCHEMA,
        "status": "complete_real_teacher_pilot_only",
        "raw_examples_file_sha256": _digest("raw"),
        "raw_examples_identity_sha256": raw[
            "raw_examples_identity_sha256"
        ],
        "pair_count": 1,
        "example_count": 2,
        "seat_counts": {"first": 1, "second": 1},
        "paired_count_per_response": {
            response_id: 1 for response_id in subject.RESPONSE_IDS
        },
        "root_count_per_response": {
            response_id: 2 for response_id in subject.RESPONSE_IDS
        },
        "legacy_q_bit_exact_root_count": 2,
        "family_reward_sha256": subject.FAMILY_REWARD_SHA256,
        "linear_expectation_of_terminal_rewards": True,
        "synthetic_values_used": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "production_training_authorized": False,
    }
    receipt = {
        **identity,
        "teacher_receipt_identity_sha256": subject.canonical_sha256(
            identity
        ),
    }
    with pytest.raises(ValueError, match="250-pair"):
        subject.require_production_teacher_coverage(
            receipt,
            raw_document=raw,
            raw_file_sha256=_digest("raw"),
        )
