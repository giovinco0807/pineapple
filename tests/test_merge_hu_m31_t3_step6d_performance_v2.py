from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

import pytest

from ofc_regular import merge_hu_m31_t3_step6d_performance_v2 as subject
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_performance as candidate02_subject,
)
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_tail_v2 as candidate02_tail_v2_subject,
)
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_full100 as candidate02_full100_subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_candidate02_full100_plan as candidate02_full100_plan,
)
from ofc_regular import run_hu_m31_t3_step6d_performance as v1
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as v2
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_runtime import (
    HU_M31_T3_ENGINE_VERSION,
    HU_M31_T3_RUNTIME_ID,
    HU_M31_T3_RUNTIME_SCHEMA,
    HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
)
from ofc_regular.state import Board


_CANDIDATE_SHA = "c" * 64
_REFERENCE_SHA = v1.REFERENCE_NATIVE_LIBRARY_SHA256


def _write_v2(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v2.canonical_bytes(value))


def _write_root(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v1._canonical_bytes(value))


def _memory(peak: int) -> dict[str, Any]:
    return {
        "supported": True,
        "source": "test",
        "rss_bytes": peak,
        "peak_rss_bytes": peak,
        "private_bytes": peak,
    }


def _observations(hand_index: int) -> tuple[ActorObservation, ActorObservation]:
    cards = create_deck(shuffle=False)
    random.Random(900_000 + hand_index).shuffle(cards)
    cursor = 0

    def take(count: int) -> list[str]:
        nonlocal cursor
        result = cards[cursor : cursor + count]
        cursor += count
        return result

    first = ActorObservation(
        hero_board=Board.from_rows(take(2), take(4), take(3)),
        opponent_public_board=Board.from_rows(take(2), take(4), take(3)),
        dealt_cards=tuple(take(3)),
        hero_private_discards=tuple(take(2)),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    second = ActorObservation(
        hero_board=Board.from_rows(take(2), take(4), take(3)),
        opponent_public_board=Board.from_rows(take(3), take(5), take(3)),
        dealt_cards=tuple(take(3)),
        hero_private_discards=tuple(take(2)),
        seat="second",
        street="T3",
        to_act_order="second",
    )
    return first, second


def _decision(
    observation: ActorObservation,
    *,
    seeds: dict[str, int],
    native_sha256: str,
    seconds: float,
    certificate: str,
    run_id: str = v1.STEP6D_RUN_ID,
) -> dict[str, Any]:
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    keyed = sorted(enumerate(legal), key=lambda item: action_key(item[1]).sort_key())
    rows = []
    for rank, (original_index, action) in enumerate(keyed):
        rows.append(
            {
                "original_index": original_index,
                "rank": rank,
                "action_key": action_key(action).to_token(),
                "selection_ev": float(len(keyed) - rank),
                "evaluation_ev": float(len(keyed) - rank) - 0.25,
                "evaluation_regret": float(rank),
                "placements": [list(item) for item in action.placements],
                "discards": list(action.discards),
            }
        )
    selected = rows[0]
    selected_action = keyed[0][1]
    return {
        "schema": HU_M31_T3_RUNTIME_SCHEMA,
        "runtime_id": HU_M31_T3_RUNTIME_ID,
        "seat": observation.seat,
        "value_scope": "q_pi_uniform_exchangeable_t3_crn_with_exact_t4_children",
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": selected["action_key"],
        "selected_selection_ev": selected["selection_ev"],
        "selected_evaluation_ev": selected["evaluation_ev"],
        "selection_gap": 1.0,
        "evaluation_sample_regret": 0.0,
        "selected_action": {
            "placements": [list(item) for item in selected_action.placements],
            "discards": list(selected_action.discards),
        },
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(legal),
        "legal_action_order_digest": ordered_action_mapping_digest(legal),
        "action_values": rows,
        "belief_prior": "uniform_exchangeable_v1",
        "candidate_belief_digest": "1" * 64,
        "evaluation_belief_digest": "2" * 64,
        "candidate_rng_digest": "3" * 64,
        "evaluation_rng_digest": "4" * 64,
        "candidate_samples": 8,
        "evaluation_samples": 32,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
        "run_id": run_id,
        "continuation_seed": seeds["child"],
        "candidate_seed": seeds["candidate"],
        "evaluation_seed": seeds["evaluation"],
        "use_t4_action_cache": True,
        "continuation_policy_id": "local_infoset_response_t3_second_t4_v1",
        "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
        "search_contract_digest": "5" * 64,
        "downstream_t4_native_semantics_id": "m30_exact_t4_native_kernel_semantics_v1",
        "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
        "downstream_t4_mode": "exact",
        "child_information_set_count": len(legal) * 80,
        "solver_id": "rust_crn_sequential_t3_v1",
        "engine_version": HU_M31_T3_ENGINE_VERSION,
        "native_library_sha256": native_sha256,
        "teacher_value_status": "diagnostic_not_match_EV",
        "native_latency_ms": seconds * 1000.0,
        "validation_latency_ms": 0.0,
        "total_latency_ms": seconds * 1000.0,
        "execution_mode": "scalar",
        "batch_size": 1,
        "semantic_result_digest_schema": HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
        "semantic_result_digest_scope": "dealt_order_independent_action_value_result",
        "semantic_result_digest": certificate * 64,
        "result_digest_scope": "ordered_action_mapping_bound",
        "result_digest": certificate * 64,
    }


def _root(
    hand_index: int, *, variant: str = v2.CANDIDATE01_VARIANT
) -> tuple[dict[str, Any], tuple[ActorObservation, ...]]:
    candidate02 = variant != v2.CANDIDATE01_VARIANT
    schedule = (
        v2.candidate02_schedule_row(hand_index)
        if candidate02
        else v1.performance_schedule_row(hand_index)
    )
    observations = _observations(hand_index)
    value = {
        "schema": (
            v2.CANDIDATE02_ROOT_SCHEMA
            if candidate02
            else v1.STEP6D_PERFORMANCE_ROOT_SCHEMA
        ),
        "contract_canonical_sha256": (
            v2._candidate02_contract_anchor_sha256()
            if candidate02
            else v1.EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        ),
        "schedule": (
            v2.CANDIDATE02_SCHEDULE if candidate02 else v1.STEP6D_PERFORMANCE_SCHEDULE
        ),
        "schedule_row_sha256": v1._digest(schedule),
        "hand_index": hand_index,
        "root_indices": schedule["root_indices"],
        "profile": schedule["profile"],
        "seeds": schedule["seeds"],
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "observations": [
            {
                "root_index": hand_index * 2 + offset,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation": observation.to_dict(),
            }
            for offset, observation in enumerate(observations)
        ],
        "current_profile_resolved": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
    }
    return value, observations


def _source_hand(
    *,
    role: str,
    root: dict[str, Any],
    observations: tuple[ActorObservation, ...],
    contract: dict[str, Any],
    manifest: dict[str, Any],
    first_seconds: float,
    second_seconds: float,
    peak: int,
) -> dict[str, Any]:
    native_sha = contract[f"{role}_library_sha256"]
    rows = []
    for offset, (observation, seconds) in enumerate(
        zip(observations, (first_seconds, second_seconds), strict=True)
    ):
        decision = _decision(
            observation,
            seeds=root["seeds"],
            native_sha256=native_sha,
            seconds=seconds,
            certificate="a" if role == "reference" else "b",
            run_id=contract["step6d_run_id"],
        )
        result = {
            "source": role,
            "native_library_sha256": native_sha,
            "solve_wall_seconds": seconds,
            "native_seconds": seconds,
            "validation_seconds": 0.0,
            "runtime_total_seconds": seconds,
            "rss_after": _memory(peak),
            "decision": decision,
        }
        portable = v1.portable_parity_payload(decision)
        rows.append(
            {
                "root_index": root["root_indices"][offset],
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation_sha256": v2.canonical_sha256(observation.to_dict()),
                "geometry": v2._source_geometry(observation, portable),
                "source_result": result,
                "portable_decision": portable,
                "portable_decision_sha256": v2.canonical_sha256(portable),
                "wall_seconds": seconds,
            }
        )
    memory = {
        "before_solver_load": _memory(peak),
        "after_solver_load": _memory(peak),
        "after_hand": _memory(peak),
        "peak_rss_bytes": peak,
    }
    schedule = v2._schedule_row(contract, root["hand_index"])
    value = {
        "schema": v2._source_hand_schema(contract),
        "contract_canonical_sha256": contract["contract_canonical_sha256"],
        "schedule": contract["schedule"],
        "run_contract_digest": manifest["run_contract_digest"],
        "shard_manifest_sha256": v2.canonical_sha256(manifest),
        "source_role": role,
        "hand_index": root["hand_index"],
        "root_indices": root["root_indices"],
        "schedule_row_sha256": v2.canonical_sha256(schedule),
        "profile": root["profile"],
        "seeds": root["seeds"],
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "root_artifact_sha256": v2.canonical_sha256(root),
        "allocation": dict(v2.ALLOCATION),
        "reference_library_sha256": contract["reference_library_sha256"],
        "candidate_library_sha256": contract["candidate_library_sha256"],
        "native_library_sha256": native_sha,
        "engine_version": HU_M31_T3_ENGINE_VERSION,
        "queue_seconds": 0.0,
        "worker_wall_seconds": first_seconds + second_seconds,
        "process_id": 123,
        "memory": memory,
        "rows": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "cloud_started": False,
    }
    return v2._validate_source_hand(
        value,
        root=root,
        run_contract=contract,
        source_role=role,
        library_sha256=native_sha,
        run_contract_digest=manifest["run_contract_digest"],
        shard_manifest_sha256=v2.canonical_sha256(manifest),
        reference_library_sha256=contract["reference_library_sha256"],
        candidate_library_sha256=contract["candidate_library_sha256"],
    )


def _fixture(
    directory: Path,
    *,
    indices: tuple[int, ...] = subject.TAIL_DIAGNOSTIC_HAND_INDICES,
    candidate_first: Callable[[int], float] = lambda _index: 100.0,
    reference_first: Callable[[int], float] = lambda _index: 160.0,
    candidate_second: Callable[[int], float] = lambda _index: 4.0,
    reference_second: Callable[[int], float] = lambda _index: 4.0,
    peak: int = 100,
    candidate_sha: str = _CANDIDATE_SHA,
    variant: str = v2.CANDIDATE01_VARIANT,
) -> tuple[Path, Path]:
    contract = v2.build_run_contract(
        candidate_library_sha256=candidate_sha,
        reference_library_sha256=_REFERENCE_SHA,
        variant=variant,
    )
    done_paths: dict[str, Path] = {}
    for role in ("candidate", "reference"):
        output = directory / role
        manifest = v2.build_shard_manifest(
            run_contract=contract,
            source_role=role,
            work_hand_indices=indices,
        )
        _write_v2(output / "run_contract.json", contract)
        _write_v2(output / "shard_manifest.json", manifest)
        for index in indices:
            root, observations = _root(index, variant=variant)
            hand = _source_hand(
                role=role,
                root=root,
                observations=observations,
                contract=contract,
                manifest=manifest,
                first_seconds=(
                    candidate_first(index)
                    if role == "candidate"
                    else reference_first(index)
                ),
                second_seconds=(
                    candidate_second(index)
                    if role == "candidate"
                    else reference_second(index)
                ),
                peak=peak,
            )
            _write_root(output / "roots" / f"hand_{index:03d}.json", root)
            _write_v2(output / "hands" / role / f"hand_{index:03d}.json", hand)
        done = v2._build_done(output_dir=output, shard_manifest=manifest)
        _write_v2(output / "DONE.json", done)
        done_paths[role] = output / "DONE.json"
    return done_paths["candidate"], done_paths["reference"]


def _rebuild_done(output: Path) -> None:
    manifest = json.loads((output / "shard_manifest.json").read_text())
    done_path = output / "DONE.json"
    done_path.unlink()
    _write_v2(done_path, v2._build_done(output_dir=output, shard_manifest=manifest))


def _mutate_candidate_hand(
    candidate_done: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    hand_path = candidate_done.parent / "hands/candidate/hand_002.json"
    hand = json.loads(hand_path.read_text())
    mutation(hand)
    _write_v2(hand_path, hand)
    _rebuild_done(candidate_done.parent)


def _mutate_decision_and_refresh_portable(
    hand: dict[str, Any], mutation: Callable[[dict[str, Any]], None]
) -> None:
    row = hand["rows"][0]
    decision = row["source_result"]["decision"]
    mutation(decision)
    row["portable_decision"] = v1.portable_parity_payload(decision)
    row["portable_decision_sha256"] = v2.canonical_sha256(row["portable_decision"])


def _mutate_child_and_refresh(hand: dict[str, Any]) -> None:
    def mutation(decision: dict[str, Any]) -> None:
        decision["child_information_set_count"] += 1

    _mutate_decision_and_refresh_portable(hand, mutation)
    hand["rows"][0]["geometry"]["child_information_set_count"] += 1


def test_tail_merge_passes_exact_precommitted_candidate01_gates(
    tmp_path: Path,
) -> None:
    candidate, reference = _fixture(tmp_path)
    result = subject.merge_performance_v2(
        candidate_done_paths=[candidate], reference_done_paths=[reference]
    )

    assert result["status"] == "pass"
    assert result["scope"] == subject.TAIL_DIAGNOSTIC_SCOPE
    assert result["hand_indices"] == list(subject.TAIL_DIAGNOSTIC_HAND_INDICES)
    assert result["integrity"]["paired_hand_parity_count"] == 10
    assert result["integrity"]["paired_root_parity_count"] == 20
    assert result["performance"]["tail_heavy_first"][
        "geometric_mean_speedup"
    ] == pytest.approx(1.6)
    assert result["candidate01_tail_qualified"] is True
    assert result["performance_candidate_frozen"] is False
    assert result["performance_lock_authorized"] is False


def test_repeated_one_hand_done_inputs_merge_in_spot_order(tmp_path: Path) -> None:
    candidate_paths: list[Path] = []
    reference_paths: list[Path] = []
    for index in subject.TAIL_DIAGNOSTIC_HAND_INDICES:
        candidate, reference = _fixture(
            tmp_path / f"hand-{index:03d}", indices=(index,)
        )
        candidate_paths.append(candidate)
        reference_paths.append(reference)
    result = subject.merge_performance_v2(
        candidate_done_paths=list(reversed(candidate_paths)),
        reference_done_paths=list(reversed(reference_paths)),
    )
    assert result["status"] == "pass"
    assert result["hand_indices"] == list(subject.TAIL_DIAGNOSTIC_HAND_INDICES)
    assert len(result["source_done_inputs"]["candidate"]) == 10


@pytest.mark.parametrize(
    ("name", "kwargs", "gate"),
    [
        (
            "median",
            {
                "candidate_first": lambda _index: 136.0,
                "reference_first": lambda _index: 220.0,
            },
            "heavy_candidate_first_median_within_135_seconds",
        ),
        (
            "max",
            {
                "candidate_first": lambda index: 146.0 if index == 2 else 100.0,
                "reference_first": lambda _index: 200.0,
            },
            "heavy_candidate_first_max_within_145_seconds",
        ),
        (
            "speedup",
            {
                "candidate_first": lambda _index: 100.0,
                "reference_first": lambda _index: 150.0,
            },
            "heavy_geometric_mean_speedup_at_least_1_55",
        ),
        (
            "second",
            {"candidate_second": lambda _index: 5.001},
            "candidate_second_max_within_5_seconds",
        ),
        (
            "rss",
            {"peak": subject.TAIL_MAX_PEAK_RSS_BYTES + 1},
            "peak_rss_within_858993459_bytes",
        ),
    ],
)
def test_tail_operational_gate_failures_are_no_go(
    tmp_path: Path, name: str, kwargs: dict[str, Any], gate: str
) -> None:
    candidate, reference = _fixture(tmp_path / name, **kwargs)
    result = subject.merge_performance_v2(
        candidate_done_paths=[candidate], reference_done_paths=[reference]
    )
    assert result["status"] == "no_go"
    assert result["gates"][gate] is False
    assert result["candidate01_tail_qualified"] is False


def test_rehashed_rng_mismatch_and_canonical_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    candidate, reference = _fixture(tmp_path / "rng")
    hand_path = candidate.parent / "hands/candidate/hand_002.json"
    hand = json.loads(hand_path.read_text())
    row = hand["rows"][0]
    row["source_result"]["decision"]["candidate_rng_digest"] = "9" * 64
    row["portable_decision"]["rng"]["candidate_rng_digest"] = "9" * 64
    row["portable_decision_sha256"] = v2.canonical_sha256(row["portable_decision"])
    _write_v2(hand_path, hand)
    _rebuild_done(candidate.parent)
    with pytest.raises(ValueError, match="RNG parity"):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate], reference_done_paths=[reference]
        )

    candidate, reference = _fixture(tmp_path / "canonical")
    candidate.write_bytes(candidate.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical"):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate], reference_done_paths=[reference]
        )


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        (
            "all_q",
            lambda hand: _mutate_decision_and_refresh_portable(
                hand,
                lambda decision: decision["action_values"][1].__setitem__(
                    "selection_ev", decision["action_values"][1]["selection_ev"] + 1.0
                ),
            ),
            "all-Q parity",
        ),
        (
            "selected",
            lambda hand: hand["rows"][0]["source_result"]["decision"].__setitem__(
                "selection_gap",
                hand["rows"][0]["source_result"]["decision"]["selection_gap"] + 1.0,
            ),
            "selected action/Q parity",
        ),
        (
            "child",
            lambda hand: _mutate_child_and_refresh(hand),
            "child-count parity",
        ),
    ],
)
def test_rehashed_q_selected_and_child_mismatches_fail_closed(
    tmp_path: Path,
    name: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    candidate, reference = _fixture(tmp_path / name)
    _mutate_candidate_hand(candidate, mutation)
    with pytest.raises(ValueError, match=message):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate], reference_done_paths=[reference]
        )


def test_role_digest_coverage_and_scope_misuse_fail_closed(tmp_path: Path) -> None:
    candidate, reference = _fixture(tmp_path / "base")
    with pytest.raises(ValueError, match="role mix"):
        subject.merge_performance_v2(
            candidate_done_paths=[reference], reference_done_paths=[reference]
        )
    with pytest.raises(ValueError, match="duplicate candidate DONE"):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate, candidate],
            reference_done_paths=[reference],
        )
    with pytest.raises(ValueError, match="cannot be applied"):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate],
            reference_done_paths=[reference],
            scope=subject.FULL_PERFORMANCE_SCOPE,
        )

    wrong = tuple([*subject.TAIL_DIAGNOSTIC_HAND_INDICES[:-1], 51])
    wrong_candidate, wrong_reference = _fixture(tmp_path / "wrong", indices=wrong)
    with pytest.raises(ValueError, match="frozen tail or hands 0..99"):
        subject.merge_performance_v2(
            candidate_done_paths=[wrong_candidate],
            reference_done_paths=[wrong_reference],
        )

    other_candidate, other_reference = _fixture(
        tmp_path / "other", candidate_sha="d" * 64
    )
    with pytest.raises(ValueError, match="run-contract digest mismatch"):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate],
            reference_done_paths=[other_reference],
        )
    assert other_candidate.is_file()


def test_missing_duplicate_and_root_or_hand_tamper_fail_closed(tmp_path: Path) -> None:
    candidate, reference = _fixture(tmp_path / "missing")
    (candidate.parent / "hands/candidate/hand_002.json").unlink()
    with pytest.raises(ValueError):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate], reference_done_paths=[reference]
        )

    candidate, reference = _fixture(tmp_path / "root")
    root = candidate.parent / "roots/hand_002.json"
    root.write_bytes(root.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical|DONE|artifact"):
        subject.merge_performance_v2(
            candidate_done_paths=[candidate], reference_done_paths=[reference]
        )


def test_summary_and_validation_are_atomic_write_once_and_recheck_sources(
    tmp_path: Path,
) -> None:
    candidate, reference = _fixture(tmp_path / "inputs")
    summary_path = tmp_path / "merge.json"
    validation_path = tmp_path / "validation.json"
    summary, validation = subject.merge_and_validate_performance_v2(
        candidate_done_paths=[candidate],
        reference_done_paths=[reference],
        summary_output_path=summary_path,
        validation_output_path=validation_path,
    )
    assert summary["status"] == validation["status"] == "pass"
    assert (
        subject.validate_performance_merge_v2(summary_path=summary_path) == validation
    )
    with pytest.raises(FileExistsError, match="write-once"):
        subject.merge_and_validate_performance_v2(
            candidate_done_paths=[candidate],
            reference_done_paths=[reference],
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )

    summary_value = json.loads(summary_path.read_text())
    summary_value["all_gates_passed"] = False
    summary_path.write_bytes(v2.canonical_bytes(summary_value))
    with pytest.raises(ValueError, match="aggregate/tamper"):
        subject.validate_performance_merge_v2(summary_path=summary_path)


def test_full_gate_is_distinct_and_uses_only_frozen_full_limits() -> None:
    assert (
        subject._scope_for_indices(
            subject.FULL_HAND_INDICES, subject.FULL_PERFORMANCE_SCOPE
        )
        == subject.FULL_PERFORMANCE_SCOPE
    )
    paired_artifacts = [
        {
            "hand_index": index,
            "profile": v1.performance_schedule_row(index)["profile"],
            "paired_seat_parity_count": 2,
        }
        for index in subject.FULL_HAND_INDICES
    ]
    paired_rows = [
        {
            "seat": seat,
            "portable_parity": {"portable_payload_exact": True},
        }
        for _index in subject.FULL_HAND_INDICES
        for seat in ("first", "second")
    ]
    performance = {
        "peak_source_process_rss_bytes": subject.TAIL_MAX_PEAK_RSS_BYTES,
        "candidate_by_seat": {
            "first": {
                "p95_seconds": 150.0,
                "p99_seconds": 240.0,
                "max_seconds": 240.0,
            },
            "second": {"p95_seconds": 5.0},
        },
    }
    mode, gates, passed = subject._build_gates(
        scope=subject.FULL_PERFORMANCE_SCOPE,
        hand_indices=subject.FULL_HAND_INDICES,
        paired_artifacts=paired_artifacts,
        paired_rows=paired_rows,
        performance=performance,
    )
    assert mode == "frozen_full_100_hand_gate"
    assert passed is True
    assert "heavy_geometric_mean_speedup_at_least_1_55" not in gates


def test_nearest_rank_and_geometric_mean_are_precommitted() -> None:
    assert subject.nearest_rank_percentile(list(range(1, 101)), 0.95) == 95.0
    assert subject.geometric_mean([1.5, 1.5, 1.5]) == pytest.approx(1.5)


def test_candidate01_merger_rejects_candidate02_contract_schema() -> None:
    contract = v2.build_run_contract(
        candidate_library_sha256=_CANDIDATE_SHA,
        reference_library_sha256=_REFERENCE_SHA,
        variant=v2.CANDIDATE02_VARIANT,
    )
    with pytest.raises(ValueError, match="contract changed"):
        subject._validate_shared_contract(contract)


def test_candidate02_tail_merger_passes_and_revalidates_sources(
    tmp_path: Path,
) -> None:
    candidate, reference = _fixture(
        tmp_path / "inputs",
        variant=v2.CANDIDATE02_VARIANT,
    )
    summary_path = tmp_path / "summary.json"
    validation_path = tmp_path / "validation.json"
    summary, validation = (
        candidate02_subject.merge_and_validate_candidate02_performance(
            candidate_done_paths=[candidate],
            reference_done_paths=[reference],
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )
    )
    assert summary["schema"] == candidate02_subject.MERGE_SCHEMA
    assert summary["scope"] == candidate02_subject.TAIL_DIAGNOSTIC_SCOPE
    assert summary["candidate_variant"] == v2.CANDIDATE02_VARIANT
    assert summary["candidate02_tail_qualified"] is True
    assert summary["full_performance_development_authorized"] is True
    assert summary["performance_lock_authorized"] is False
    assert summary["training_authorized"] is False
    assert validation["schema"] == candidate02_subject.VALIDATION_SCHEMA
    assert (
        candidate02_subject.validate_candidate02_performance_merge(
            summary_path=summary_path
        )
        == validation
    )

    tampered = json.loads(summary_path.read_text(encoding="utf-8"))
    tampered["candidate02_tail_qualified"] = False
    summary_path.write_bytes(v2.canonical_bytes(tampered))
    with pytest.raises(ValueError, match="aggregate/tamper"):
        candidate02_subject.validate_candidate02_performance_merge(
            summary_path=summary_path
        )


def test_candidate02_tail_no_go_does_not_authorize_full100(tmp_path: Path) -> None:
    candidate, reference = _fixture(
        tmp_path / "inputs",
        variant=v2.CANDIDATE02_VARIANT,
        candidate_first=lambda _index: 100.0,
        reference_first=lambda _index: 150.0,
    )
    summary = candidate02_subject.merge_candidate02_performance(
        candidate_done_paths=[candidate],
        reference_done_paths=[reference],
    )
    assert summary["status"] == "no_go"
    assert summary["candidate02_tail_qualified"] is False
    assert summary["full_performance_development_authorized"] is False
    assert summary["gates"]["heavy_geometric_mean_speedup_at_least_1_55"] is False


def test_candidate02_tail_merger_rejects_mixed_candidate01_reference(
    tmp_path: Path,
) -> None:
    candidate02, _reference02 = _fixture(
        tmp_path / "candidate02",
        variant=v2.CANDIDATE02_VARIANT,
    )
    _candidate01, reference01 = _fixture(tmp_path / "candidate01")
    with pytest.raises(ValueError, match="contract changed"):
        candidate02_subject.merge_candidate02_performance(
            candidate_done_paths=[candidate02],
            reference_done_paths=[reference01],
        )


def test_candidate02_tail_v2_passes_fresh_nonoverlapping_gate_and_revalidates(
    tmp_path: Path,
) -> None:
    candidate, reference = _fixture(
        tmp_path / "inputs",
        indices=candidate02_tail_v2_subject.TAIL_HAND_INDICES,
        variant=v2.CANDIDATE02_TAIL_V2_VARIANT,
    )
    summary_path = tmp_path / "summary.json"
    validation_path = tmp_path / "validation.json"
    summary, validation = (
        candidate02_tail_v2_subject.merge_and_validate_candidate02_tail_v2(
            candidate_done_paths=[candidate],
            reference_done_paths=[reference],
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )
    )
    assert summary["schema"] == candidate02_tail_v2_subject.MERGE_SCHEMA
    assert summary["status"] == "pass"
    assert summary["candidate_variant"] == v2.CANDIDATE02_TAIL_V2_VARIANT
    assert summary["hand_indices"] == list(
        candidate02_tail_v2_subject.TAIL_HAND_INDICES
    )
    assert summary["heavy_hand_indices"] == list(
        candidate02_tail_v2_subject.TAIL_HEAVY_HAND_INDICES
    )
    assert summary["random_hand_indices"] == list(
        candidate02_tail_v2_subject.TAIL_RANDOM_HAND_INDICES
    )
    assert not (
        set(summary["hand_indices"])
        & set(summary["excluded_candidate02_v1_hand_indices"])
    )
    assert summary["integrity"]["heavy_21x21_hand_count"] == 8
    assert summary["candidate02_tail_v2_qualified"] is True
    assert summary["full_performance_development_authorized"] is True
    assert summary["performance_lock_authorized"] is False
    assert summary["training_authorized"] is False
    assert validation["schema"] == candidate02_tail_v2_subject.VALIDATION_SCHEMA
    assert (
        candidate02_tail_v2_subject.validate_candidate02_tail_v2_merge(
            summary_path=summary_path
        )
        == validation
    )

    tampered = json.loads(summary_path.read_text(encoding="utf-8"))
    tampered["heavy_hand_indices"] = list(
        candidate02_tail_v2_subject.TAIL_RANDOM_HAND_INDICES
    )
    summary_path.write_bytes(v2.canonical_bytes(tampered))
    with pytest.raises(ValueError, match="selection changed"):
        candidate02_tail_v2_subject.validate_candidate02_tail_v2_merge(
            summary_path=summary_path
        )


def test_candidate02_tail_v2_speed_no_go_cannot_authorize_full100(
    tmp_path: Path,
) -> None:
    candidate, reference = _fixture(
        tmp_path / "inputs",
        indices=candidate02_tail_v2_subject.TAIL_HAND_INDICES,
        variant=v2.CANDIDATE02_TAIL_V2_VARIANT,
        candidate_first=lambda _index: 136.0,
        reference_first=lambda _index: 160.0,
    )
    summary = candidate02_tail_v2_subject.merge_candidate02_tail_v2(
        candidate_done_paths=[candidate],
        reference_done_paths=[reference],
    )
    assert summary["status"] == "no_go"
    assert summary["candidate02_tail_v2_qualified"] is False
    assert summary["full_performance_development_authorized"] is False
    assert summary["gates"]["heavy_candidate_first_median_within_135_seconds"] is False
    assert summary["gates"]["heavy_geometric_mean_speedup_at_least_1_55"] is False


def test_candidate02_tail_v2_rejects_selection_manifest_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(
        candidate02_tail_v2_subject.SELECTION_MANIFEST_PATH.read_text("utf-8")
    )
    manifest["heavy_hand_indices"] = list(
        candidate02_tail_v2_subject.TAIL_RANDOM_HAND_INDICES
    )
    tampered_path = tmp_path / "selection.json"
    _write_v2(tampered_path, manifest)
    monkeypatch.setattr(
        candidate02_tail_v2_subject, "SELECTION_MANIFEST_PATH", tampered_path
    )
    with pytest.raises(ValueError, match="frozen selection changed"):
        candidate02_tail_v2_subject._assert_frozen_selection()


def test_candidate02_tail_v2_rejects_wrong_tail(tmp_path: Path) -> None:
    wrong_tail = candidate02_tail_v2_subject.TAIL_HAND_INDICES[:-1]
    candidate, reference = _fixture(
        tmp_path / "inputs",
        indices=wrong_tail,
        variant=v2.CANDIDATE02_TAIL_V2_VARIANT,
    )
    with pytest.raises(ValueError, match="exact frozen ten-hand tail"):
        candidate02_tail_v2_subject.merge_candidate02_tail_v2(
            candidate_done_paths=[candidate],
            reference_done_paths=[reference],
        )


@pytest.mark.parametrize(
    "reference_variant", [v2.CANDIDATE01_VARIANT, v2.CANDIDATE02_VARIANT]
)
def test_candidate02_tail_v2_rejects_mixed_v1_contracts(
    tmp_path: Path, reference_variant: str
) -> None:
    candidate_v2, _reference_v2 = _fixture(
        tmp_path / "tail-v2",
        indices=candidate02_tail_v2_subject.TAIL_HAND_INDICES,
        variant=v2.CANDIDATE02_TAIL_V2_VARIANT,
    )
    _candidate_old, reference_old = _fixture(
        tmp_path / "old",
        variant=reference_variant,
    )
    with pytest.raises(ValueError, match="contract changed"):
        candidate02_tail_v2_subject.merge_candidate02_tail_v2(
            candidate_done_paths=[candidate_v2],
            reference_done_paths=[reference_old],
        )


def _candidate02_full100_fixture(
    directory: Path,
) -> tuple[list[Path], list[Path]]:
    plan = json.loads(
        candidate02_full100_subject.DEFAULT_PLAN_PATH.read_text(encoding="utf-8")
    )
    contract = v2.build_run_contract(
        candidate_library_sha256=candidate02_full100_plan.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=_REFERENCE_SHA,
        variant=v2.CANDIDATE02_VARIANT,
    )
    candidate_paths: list[Path] = []
    reference_paths: list[Path] = []
    for shard in plan["shards"]:
        indices = tuple(shard["work_hand_indices"])
        for role, paths in (
            ("candidate", candidate_paths),
            ("reference", reference_paths),
        ):
            output = directory / f"{role}-shard-{shard['shard_index']:02d}"
            manifest = v2.build_shard_manifest(
                run_contract=contract,
                source_role=role,
                work_hand_indices=indices,
            )
            _write_v2(output / "run_contract.json", contract)
            _write_v2(output / "shard_manifest.json", manifest)
            for index in indices:
                root_path = (
                    candidate02_full100_plan.DEFAULT_ROOT_DIR / f"hand_{index:03d}.json"
                )
                root = json.loads(root_path.read_text(encoding="utf-8"))
                observations = tuple(
                    ActorObservation.from_dict(raw["observation"])
                    for raw in root["observations"]
                )
                hand = _source_hand(
                    role=role,
                    root=root,
                    observations=observations,
                    contract=contract,
                    manifest=manifest,
                    first_seconds=(100.0 if role == "candidate" else 160.0),
                    second_seconds=4.0,
                    peak=100,
                )
                _write_root(
                    output / "roots" / f"hand_{index:03d}.json",
                    root,
                )
                _write_v2(
                    output / "hands" / role / f"hand_{index:03d}.json",
                    hand,
                )
            done = v2._build_done(output_dir=output, shard_manifest=manifest)
            _write_v2(output / "DONE.json", done)
            paths.append(output / "DONE.json")
    return candidate_paths, reference_paths


def test_candidate02_full100_merger_binds_plan_and_opens_only_lock(
    tmp_path: Path,
) -> None:
    candidate_paths, reference_paths = _candidate02_full100_fixture(tmp_path / "inputs")
    summary_path = tmp_path / "summary.json"
    validation_path = tmp_path / "validation.json"
    summary, validation = (
        candidate02_full100_subject.merge_and_validate_candidate02_full100(
            candidate_done_paths=list(reversed(candidate_paths)),
            reference_done_paths=list(reversed(reference_paths)),
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )
    )

    assert summary["schema"] == candidate02_full100_subject.MERGE_SCHEMA
    assert summary["status"] == "pass"
    assert summary["scope"] == subject.FULL_PERFORMANCE_SCOPE
    assert summary["candidate_variant"] == v2.CANDIDATE02_VARIANT
    assert (
        summary["full100_plan_sha256"] == candidate02_full100_plan.FULL100_PLAN_SHA256
    )
    assert summary["paired_hand_count"] == 100
    assert summary["root_count"] == 200
    assert len(summary["source_done_inputs"]["candidate"]) == 10
    assert len(summary["source_done_inputs"]["reference"]) == 10
    assert summary["all_gates_passed"] is True
    assert summary["performance_candidate_frozen"] is True
    assert summary["performance_lock_authorized"] is True
    assert summary["quality_pilot_authorized"] is False
    assert summary["training_authorized"] is False
    assert summary["current_profile_changed"] is False
    assert validation["source_shard_count"] == 20
    assert (
        candidate02_full100_subject.validate_candidate02_full100_merge(
            summary_path=summary_path
        )
        == validation
    )

    tampered = json.loads(summary_path.read_text(encoding="utf-8"))
    tampered["full100_plan"]["spot_package_authorized"] = True
    summary_path.write_bytes(v2.canonical_bytes(tampered))
    with pytest.raises(ValueError, match="plan contract|frozen plan"):
        candidate02_full100_subject.validate_candidate02_full100_merge(
            summary_path=summary_path
        )


def test_candidate02_full100_merger_rejects_missing_shard(
    tmp_path: Path,
) -> None:
    candidate_paths, reference_paths = _candidate02_full100_fixture(tmp_path / "inputs")
    with pytest.raises(ValueError, match="work coverage mismatch"):
        candidate02_full100_subject.merge_candidate02_full100(
            candidate_done_paths=candidate_paths[:-1],
            reference_done_paths=reference_paths,
        )
