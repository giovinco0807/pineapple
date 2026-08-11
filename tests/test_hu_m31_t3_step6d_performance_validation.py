from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

import pytest

from ofc_regular import run_hu_m31_t3_step6d_performance as runner
from ofc_regular import validate_hu_m31_t3_step6d_performance as subject
from ofc_regular.action_key import action_key, legal_action_set_digest, ordered_action_mapping_digest
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_runtime import (
    HU_M31_T3_ENGINE_VERSION,
    HU_M31_T3_RUNTIME_ID,
    HU_M31_T3_RUNTIME_SCHEMA,
    HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
)
from ofc_regular.hu_m31_t3_step6d_contract import (
    EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
    STEP6D_RUN_ID,
)
from ofc_regular.state import Board


_CANDIDATE_SHA = "c" * 64


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject._artifact_bytes(value))


def _memory(peak: int) -> dict[str, Any]:
    return {
        "supported": True,
        "source": "test",
        "rss_bytes": peak,
        "peak_rss_bytes": peak,
        "private_bytes": peak,
    }


def _observation(hand_index: int, seat: str) -> ActorObservation:
    cards = create_deck(shuffle=False)
    random.Random(100_000 + hand_index * 2 + (seat == "second")).shuffle(cards)
    cursor = 0

    def take(count: int) -> list[str]:
        nonlocal cursor
        result = cards[cursor : cursor + count]
        cursor += count
        return result

    hero = Board.from_rows(take(3), take(5), take(1))
    if seat == "first":
        opponent = Board.from_rows(take(3), take(5), take(1))
    else:
        opponent = Board.from_rows(take(3), take(5), take(3))
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=tuple(take(3)),
        hero_private_discards=tuple(take(2)),
        seat=seat,
        street="T3",
        to_act_order=seat,
    )


def _decision(
    observation: ActorObservation,
    *,
    seeds: dict[str, int],
    native_sha256: str,
    seconds: float,
    certificate: str,
) -> dict[str, Any]:
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    keyed = sorted(
        enumerate(legal), key=lambda item: action_key(item[1]).sort_key()
    )
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
        "action_key_schema": "regular_ofc_action_key_v1",
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
        "run_id": STEP6D_RUN_ID,
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
        "child_information_set_count": len(legal) * 40,
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


def _source(
    source: str,
    decision: dict[str, Any],
    *,
    seconds: float,
    peak: int,
) -> dict[str, Any]:
    return {
        "source": source,
        "native_library_sha256": decision["native_library_sha256"],
        "solve_wall_seconds": seconds,
        "native_seconds": seconds,
        "validation_seconds": 0.0,
        "runtime_total_seconds": seconds,
        "rss_after": _memory(peak),
        "decision": decision,
    }


def _root_and_hand(
    hand_index: int,
    *,
    run_contract_digest: str,
    first_seconds: float,
    second_seconds: float,
    peak: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule = subject.expected_schedule_row(hand_index)
    observations = (
        _observation(hand_index, "first"),
        _observation(hand_index, "second"),
    )
    root = {
        "schema": subject.STEP6D_PERFORMANCE_ROOT_SCHEMA,
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": subject.PERFORMANCE_DEVELOPMENT_SCHEDULE,
        "schedule_row_sha256": subject._artifact_digest(schedule),
        "hand_index": hand_index,
        "root_indices": schedule["root_indices"],
        "profile": schedule["profile"],
        "seeds": schedule["seeds"],
        "budget": dict(subject.SEARCH_BUDGET),
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
    rows = []
    for offset, (observation, seconds) in enumerate(
        zip(observations, (first_seconds, second_seconds), strict=True)
    ):
        reference_decision = _decision(
            observation,
            seeds=schedule["seeds"],
            native_sha256=subject.REFERENCE_NATIVE_LIBRARY_SHA256,
            seconds=1.0,
            certificate="a",
        )
        candidate_decision = copy.deepcopy(reference_decision)
        candidate_decision["native_library_sha256"] = _CANDIDATE_SHA
        candidate_decision["native_latency_ms"] = seconds * 1000.0
        candidate_decision["total_latency_ms"] = seconds * 1000.0
        candidate_decision["semantic_result_digest"] = "b" * 64
        candidate_decision["result_digest"] = "b" * 64
        reference = _source(
            "reference", reference_decision, seconds=1.0, peak=peak
        )
        candidate = _source(
            "candidate", candidate_decision, seconds=seconds, peak=peak
        )
        parity = runner.compare_portable_decisions(
            reference_decision, candidate_decision
        )
        rows.append(
            {
                "root_index": hand_index * 2 + offset,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "solve_order": (
                    ["reference", "candidate"]
                    if offset == 0
                    else ["candidate", "reference"]
                ),
                "geometry": runner._geometry(observation, candidate_decision),
                "reference": reference,
                "candidate": candidate,
                "parity": parity,
                "wall_seconds": seconds + 1.0,
            }
        )
    memory = {
        "before_solver_load": _memory(peak),
        "after_solver_load": _memory(peak),
        "after_hand": _memory(peak),
        "peak_rss_bytes": peak,
    }
    hand = {
        "schema": subject.STEP6D_PERFORMANCE_HAND_SCHEMA,
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": subject.PERFORMANCE_DEVELOPMENT_SCHEDULE,
        "hand_index": hand_index,
        "profile": schedule["profile"],
        "seeds": schedule["seeds"],
        "budget": dict(subject.SEARCH_BUDGET),
        "run_contract_digest": run_contract_digest,
        "root_artifact_sha256": subject._artifact_digest(root),
        "allocation": {"workers": 1, "rayon_threads_per_worker": 16},
        "reference_library_sha256": subject.REFERENCE_NATIVE_LIBRARY_SHA256,
        "candidate_library_sha256": _CANDIDATE_SHA,
        "queue_seconds": 0.0,
        "worker_wall_seconds": first_seconds + second_seconds + 2.0,
        "process_id": 123,
        "memory": memory,
        "rows": rows,
        "portable_parity_exact": True,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "cloud_started": False,
    }
    return root, hand


def _fixture(
    directory: Path,
    *,
    indices: list[int],
    first: Callable[[int], float] = lambda _index: 1.0,
    second: Callable[[int], float] = lambda _index: 1.0,
    peak: int = 100,
) -> Path:
    run_contract = runner._run_contract(
        indices=indices,
        reference_sha256=subject.REFERENCE_NATIVE_LIBRARY_SHA256,
        candidate_sha256=_CANDIDATE_SHA,
        workers=1,
        rayon_threads=16,
    )
    digest = subject._artifact_digest(run_contract)
    reports = []
    for index in indices:
        root, hand = _root_and_hand(
            index,
            run_contract_digest=digest,
            first_seconds=first(index),
            second_seconds=second(index),
            peak=peak,
        )
        _write(directory / "roots" / f"hand_{index:03d}.json", root)
        _write(directory / "hands" / f"hand_{index:03d}.json", hand)
        reports.append(hand)
    summary = runner._build_summary(
        reports=reports,
        indices=indices,
        output_dir=directory,
        reference_sha256=subject.REFERENCE_NATIVE_LIBRARY_SHA256,
        candidate_sha256=_CANDIDATE_SHA,
        workers=1,
        rayon_threads=16,
        resumed_hand_count=0,
    )
    path = directory / "summary.json"
    _write(path, summary)
    return path


def _rebuild_summary(directory: Path) -> None:
    old = json.loads((directory / "summary.json").read_text(encoding="ascii"))
    indices = old["hand_indices"]
    reports = [
        json.loads(
            (directory / "hands" / f"hand_{index:03d}.json").read_text(
                encoding="ascii"
            )
        )
        for index in indices
    ]
    summary = runner._build_summary(
        reports=reports,
        indices=indices,
        output_dir=directory,
        reference_sha256=subject.REFERENCE_NATIVE_LIBRARY_SHA256,
        candidate_sha256=_CANDIDATE_SHA,
        workers=1,
        rayon_threads=16,
        resumed_hand_count=old["resumed_hand_count"],
    )
    _write(directory / "summary.json", summary)


def _mutate_hand(directory: Path, index: int, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = directory / "hands" / f"hand_{index:03d}.json"
    value = json.loads(path.read_text(encoding="ascii"))
    mutation(value)
    _write(path, value)
    _rebuild_summary(directory)


def test_full_gate_passes_inclusive_limits_and_exact_grid(tmp_path: Path) -> None:
    summary = _fixture(
        tmp_path,
        indices=list(range(100)),
        first=lambda index: 1.0 if index < 94 else (150.0 if index < 98 else 240.0),
        second=lambda index: 1.0 if index < 94 else 5.0,
        peak=subject.MAX_PEAK_RSS_BYTES,
    )
    result = subject.validate_performance(summary_path=summary)
    assert result["status"] == "pass"
    assert result["integrity"]["paired_hand_count"] == 100
    assert result["integrity"]["root_count"] == 200
    assert result["integrity"]["seat_root_counts"] == {"first": 100, "second": 100}
    assert set(result["integrity"]["profile_hand_counts"].values()) == {20}
    assert result["performance"]["candidate_latency_by_seat"]["first"]["p95_seconds"] == 150.0
    assert result["performance"]["candidate_latency_by_seat"]["first"]["p99_seconds"] == 240.0
    assert result["performance"]["candidate_latency_by_seat"]["second"]["p95_seconds"] == 5.0
    assert result["gates"]["missing_or_censored_roots_zero"] is True


def test_full_gate_latency_failure_is_no_go(tmp_path: Path) -> None:
    summary = _fixture(
        tmp_path,
        indices=list(range(100)),
        first=lambda index: 150.001 if index >= 94 else 1.0,
    )
    result = subject.validate_performance(summary_path=summary)
    assert result["status"] == "no_go"
    assert result["gates"]["first_p95_within_150_seconds"] is False
    assert result["performance_lock_authorized"] is False


def test_subset_smoke_never_applies_performance_gate(tmp_path: Path) -> None:
    summary = _fixture(
        tmp_path,
        indices=[7],
        first=lambda _index: 999.0,
        second=lambda _index: 99.0,
        peak=subject.MAX_PEAK_RSS_BYTES + 1,
    )
    result = subject.validate_performance(summary_path=summary)
    assert result["status"] == "not_applicable"
    assert result["gate_applicable"] is False
    assert set(result["gates"].values()) == {None}
    assert result["performance_candidate_frozen"] is False


def test_nearest_rank_is_precommitted() -> None:
    values = list(range(1, 101))
    assert subject.nearest_rank_percentile(values, 0.50) == 50.0
    assert subject.nearest_rank_percentile(values, 0.95) == 95.0
    assert subject.nearest_rank_percentile(values, 0.99) == 99.0


def test_unknown_and_unrehash_tamper_fail_closed(tmp_path: Path) -> None:
    summary = _fixture(tmp_path, indices=[0])
    value = json.loads(summary.read_text(encoding="ascii"))
    value["unknown"] = True
    _write(summary, value)
    with pytest.raises(ValueError, match="unknown"):
        subject.validate_performance(summary_path=summary)

    summary = _fixture(tmp_path / "tamper", indices=[0])
    hand = summary.parent / "hands/hand_000.json"
    hand.write_bytes(hand.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash/size"):
        subject.validate_performance(summary_path=summary)


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        (
            "seed",
            lambda value: value["seeds"].__setitem__("candidate", 1),
            "seed/profile/provenance",
        ),
        (
            "profile",
            lambda value: value.__setitem__("profile", "stage7_m5_r10"),
            "seed/profile/provenance",
        ),
        (
            "root_index",
            lambda value: value["rows"][0].__setitem__("root_index", 1),
            "root/seat/index",
        ),
        (
            "action_index",
            lambda value: value["rows"][0]["candidate"]["decision"][
                "action_values"
            ][0].__setitem__("original_index", 999),
            "action index",
        ),
        (
            "q",
            lambda value: value["rows"][0]["candidate"]["decision"][
                "action_values"
            ][0].__setitem__("selection_ev", 999.0),
            "Q mismatch",
        ),
    ],
)
def test_seed_profile_root_action_index_and_q_mismatch_fail_closed(
    tmp_path: Path,
    name: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    directory = tmp_path / name
    summary = _fixture(directory, indices=[0])
    _mutate_hand(directory, 0, mutation)
    with pytest.raises(ValueError, match=message):
        subject.validate_performance(summary_path=summary)


def test_duplicate_manifest_and_summary_metric_tamper_fail_closed(tmp_path: Path) -> None:
    summary = _fixture(tmp_path, indices=[0, 1])
    value = json.loads(summary.read_text(encoding="ascii"))
    value["hand_manifest"][1] = copy.deepcopy(value["hand_manifest"][0])
    _write(summary, value)
    with pytest.raises(ValueError, match="index/path"):
        subject.validate_performance(summary_path=summary)

    summary = _fixture(tmp_path / "metric", indices=[0])
    value = json.loads(summary.read_text(encoding="ascii"))
    value["performance"]["candidate_by_seat"]["first"]["solve_wall_seconds"][
        "p95_seconds"
    ] = 0.0
    _write(summary, value)
    with pytest.raises(ValueError, match="aggregate/tamper"):
        subject.validate_performance(summary_path=summary)


def test_output_is_canonical_write_once(tmp_path: Path) -> None:
    summary = _fixture(tmp_path / "input", indices=[0])
    output = tmp_path / "validation.json"
    result = subject.validate_performance(summary_path=summary, output_path=output)
    assert json.loads(output.read_text(encoding="ascii")) == result
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.validate_performance(summary_path=summary, output_path=output)
