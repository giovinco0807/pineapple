from __future__ import annotations

import copy

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_scientific_bridge_v2 as scientific_bridge,
)
from ofc_regular import hu_m31_t3_step6d_performance_lock_v4_gate as subject
from ofc_regular import merge_hu_m31_t3_step6d_performance_v2 as performance_v2
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular.hu_m31_t3_behavior_roots import behavior_profile_for_index


def _contract() -> dict:
    return runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
    )


def _sha(value: int) -> str:
    return f"{value:064x}"[-64:]


def _parity(value: int) -> dict:
    digest = _sha(10_000 + value)
    return {
        "schema": "hu_m31_t3_step6d_portable_parity_v1",
        "candidate_portable_sha256": digest,
        "reference_portable_sha256": digest,
        "action_keys_exact": True,
        "child_information_set_count_exact": True,
        "evaluation_q_exact": True,
        "portable_payload_exact": True,
        "rng_exact": True,
        "selected_action_exact": True,
        "selection_q_exact": True,
    }


def _generic(
    *,
    first_seconds: list[float] | None = None,
    second_seconds: list[float] | None = None,
    speedup: float = 2.0,
    peak_rss: int = 700_000_000,
) -> tuple[dict, dict]:
    contract = _contract()
    first_values = first_seconds or [60.0] * 100
    second_values = second_seconds or [1.0] * 100
    paired = []
    first_speedups = []
    for hand in runner.CONTRACT_HAND_INDICES:
        rows = []
        for offset, (seat, candidate_seconds, ratio) in enumerate(
            (
                ("first", first_values[hand], speedup),
                ("second", second_values[hand], 1.1),
            )
        ):
            parity = _parity(2 * hand + offset)
            rows.append(
                {
                    "root_index": 2 * hand + offset,
                    "seat": seat,
                    "observation_fingerprint": _sha(20_000 + 2 * hand + offset),
                    "observation_sha256": _sha(30_000 + 2 * hand + offset),
                    "portable_decision_sha256": parity[
                        "candidate_portable_sha256"
                    ],
                    "portable_parity": parity,
                    "reference_solve_wall_seconds": candidate_seconds * ratio,
                    "candidate_solve_wall_seconds": candidate_seconds,
                    "paired_speedup": ratio,
                }
            )
        first_speedups.append(speedup)
        paired.append(
            {
                "hand_index": hand,
                "profile": behavior_profile_for_index(hand),
                "paired_seat_parity_count": 2,
                "rows": rows,
            }
        )
    first_summary = performance_v2._latencies(first_values)
    second_summary = performance_v2._latencies(second_values)
    generic = {
        "schema": performance_v2.MERGE_SCHEMA,
        "scope": performance_v2.FULL_PERFORMANCE_SCOPE,
        "run_contract": contract,
        "run_contract_digest": runner.canonical_sha256(contract),
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "paired_hand_count": 100,
        "root_count": 200,
        "budget": contract["budget"],
        "allocation": contract["allocation"],
        "reference_library_sha256": contract["reference_library_sha256"],
        "candidate_library_sha256": contract["candidate_library_sha256"],
        "source_done_inputs": {
            "candidate": [{} for _ in range(10)],
            "reference": [{} for _ in range(10)],
        },
        "paired_artifacts": paired,
        "integrity": {
            "candidate_hand_count": 100,
            "reference_hand_count": 100,
            "paired_hand_count": 100,
            "paired_root_count": 200,
            "unique_observation_fingerprint_count": 200,
            "paired_hand_parity_count": 100,
            "paired_root_parity_count": 200,
            "missing_hand_indices": [],
            "duplicate_hand_indices": [],
            "out_of_contract_hand_indices": [],
        },
        "performance": {
            "candidate_by_seat": {
                "first": first_summary,
                "second": second_summary,
            },
            "first_paired_speedups": first_speedups,
            "first_geometric_mean_speedup_diagnostic": (
                performance_v2.geometric_mean(first_speedups)
            ),
            "peak_source_process_rss_bytes": peak_rss,
        },
    }
    return generic, contract


def test_metrics_pass_stays_closed_until_transport_lineage() -> None:
    generic, contract = _generic()
    gate = subject.build_performance_lock_v4_gate(
        generic, run_contract=contract
    )
    assert gate["all_gates_passed"] is True
    assert all(gate["gates"].values())
    assert gate["scientific_performance_gate_passed"] is True
    assert gate["transport_lineage_required"] is True
    assert gate["transport_lineage_validated"] is False
    assert gate["performance_lock_finalized"] is False
    assert gate["performance_lock_qualified"] is False
    assert gate["one_shot_lock_consumed"] is False
    assert gate["quality_pilot_authorized"] is False
    assert gate["rerun_authorized"] is False
    assert gate["reseed_authorized"] is False
    assert gate["training_authorized"] is False
    assert gate["runtime_policy_activated"] is False


@pytest.mark.parametrize(
    ("kwargs", "gate_name"),
    [
        ({"first_seconds": [150.01] * 100}, "first_p95_within_150_seconds"),
        ({"first_seconds": [60.0] * 98 + [240.01, 240.01]},
         "first_p99_within_240_seconds"),
        ({"first_seconds": [60.0] * 99 + [240.01]},
         "first_max_within_240_seconds"),
        ({"second_seconds": [5.01] * 100}, "second_p95_within_5_seconds"),
        ({"peak_rss": 858_993_460}, "peak_rss_within_858993459_bytes"),
        ({"speedup": 1.549999},
         "first_geometric_mean_speedup_at_least_1_55"),
    ],
)
def test_each_numeric_boundary_fails_closed(
    kwargs: dict, gate_name: str
) -> None:
    generic, contract = _generic(**kwargs)
    gate = subject.build_performance_lock_v4_gate(
        generic, run_contract=contract
    )
    assert gate["all_gates_passed"] is False
    assert gate["gates"][gate_name] is False
    assert gate["candidate_finalized_no_go"] is False
    assert gate["quality_pilot_authorized"] is False
    assert gate["rerun_authorized"] is False
    assert gate["reseed_authorized"] is False


def test_rejects_stored_speedup_not_derived_from_source_times() -> None:
    generic, contract = _generic()
    generic["paired_artifacts"][0]["rows"][0]["paired_speedup"] = 99.0
    with pytest.raises(ValueError, match="differs from source times"):
        subject.build_performance_lock_v4_gate(generic, run_contract=contract)


def test_rejects_stored_latency_not_derived_from_source_rows() -> None:
    generic, contract = _generic()
    generic["performance"]["candidate_by_seat"]["first"]["p95_seconds"] = 1.0
    with pytest.raises(ValueError, match="latency differs from source rows"):
        subject.build_performance_lock_v4_gate(generic, run_contract=contract)


def test_rejects_duplicate_hand_even_if_claimed_counters_are_clean() -> None:
    generic, contract = _generic()
    generic["paired_artifacts"][-1]["hand_index"] = 0
    with pytest.raises(ValueError, match="ordered unique 0..99"):
        subject.build_performance_lock_v4_gate(generic, run_contract=contract)


def test_rejects_profile_relabel_without_an_unrelated_seat_failure() -> None:
    generic, contract = _generic()
    generic["paired_artifacts"][0]["profile"] = behavior_profile_for_index(1)
    with pytest.raises(ValueError, match="profile differs"):
        subject.build_performance_lock_v4_gate(generic, run_contract=contract)


def test_rejects_non_v4_contract_and_generic_binding() -> None:
    generic, contract = _generic()
    development = runner.build_run_contract(
        candidate_library_sha256=contract["candidate_library_sha256"],
        reference_library_sha256=contract["reference_library_sha256"],
        variant=runner.CANDIDATE02_VARIANT,
    )
    with pytest.raises(ValueError, match="requires the v4 run contract"):
        subject.build_performance_lock_v4_gate(
            generic, run_contract=development
        )


def test_rejects_tampered_stored_gate_and_bool_as_float() -> None:
    generic, contract = _generic()
    gate = subject.build_performance_lock_v4_gate(
        generic, run_contract=contract
    )
    tampered = copy.deepcopy(gate)
    tampered["thresholds"]["portable_semantic_parity_fraction"] = True
    tampered["observed"]["portable_semantic_parity_fraction"] = True
    with pytest.raises(ValueError, match="finite float"):
        subject.validate_performance_lock_v4_gate_value(
            tampered, generic=generic, run_contract=contract
        )


@pytest.mark.parametrize(
    "field",
    [
        "scientific_performance_gate_passed",
        "transport_lineage_required",
        "transport_lineage_validated",
        "performance_lock_finalized",
        "performance_lock_qualified",
        "candidate_finalized_no_go",
        "one_shot_lock_consumed",
        "rerun_authorized",
        "reseed_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    ],
)
def test_rejects_integer_alias_for_every_boundary_boolean(field: str) -> None:
    generic, contract = _generic()
    gate = subject.build_performance_lock_v4_gate(
        generic, run_contract=contract
    )
    tampered = copy.deepcopy(gate)
    tampered[field] = int(tampered[field])
    with pytest.raises(ValueError, match="boundary flags must be booleans"):
        subject.validate_performance_lock_v4_gate_value(
            tampered, generic=generic, run_contract=contract
        )


def test_rejects_bool_alias_for_root_indices() -> None:
    generic, contract = _generic()
    generic["paired_artifacts"][0]["rows"][0]["root_index"] = False
    generic["paired_artifacts"][0]["rows"][1]["root_index"] = True
    with pytest.raises(ValueError, match="paired root index"):
        subject.build_performance_lock_v4_gate(generic, run_contract=contract)


def test_scientific_rng_audit_accepts_the_plan_bound_v4_namespaces() -> None:
    seed = runner.candidate02_performance_lock_v4_seed_contract()
    plan = {
        "seed_contract": copy.deepcopy(seed),
        "full100_plan": {"run_contract": {"seed_contract": copy.deepcopy(seed)}},
    }
    audit = scientific_bridge._rng_namespace_audit(plan)
    assert audit["seed_count"] == 600
    assert audit["unique_seed_count"] == 600
    assert audit["seed_set_sha256"] == seed["seed_set_sha256"]

    forged = copy.deepcopy(plan)
    forged["seed_contract"]["namespace_bases"]["hand"] += 1
    with pytest.raises(ValueError, match="RNG namespace"):
        scientific_bridge._rng_namespace_audit(forged)
