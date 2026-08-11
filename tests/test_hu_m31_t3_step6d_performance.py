from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular.action_key import action_key, legal_action_set_digest
from ofc_regular.action_space import Action
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from ofc_regular.hu_m31_t3_step6d_contract import SEED_STRIDE, schedule_by_name
from ofc_regular import run_hu_m31_t3_step6d_performance as subject
from ofc_regular.run_hu_m31_t3_step6d_performance import (
    PERFORMANCE_BUDGET,
    PERFORMANCE_HAND_INDICES,
    STEP6D_PERFORMANCE_SCHEDULE,
    compare_portable_decisions,
    normalize_hand_indices,
    performance_schedule_row,
    performance_seed_values,
    portable_parity_payload,
    validate_performance_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs/hu_joint_policy_m31_t3_step6d_contract.json"


def _actions() -> tuple[Action, Action]:
    return (
        Action(
            placements=(("2c", "top"), ("3c", "middle")),
            discards=("4c",),
        ),
        Action(
            placements=(("2c", "bottom"), ("4c", "middle")),
            discards=("3c",),
        ),
    )


def _decision() -> dict[str, Any]:
    actions = _actions()
    rows = []
    for index, (action, selection, evaluation) in enumerate(
        zip(actions, (1.0, 0.5), (0.8, 0.6), strict=True)
    ):
        rows.append(
            {
                "original_index": index,
                "rank": index,
                "action_key": action_key(action).to_token(),
                "selection_ev": selection,
                "evaluation_ev": evaluation,
                "evaluation_regret": 0.8 - evaluation,
                "placements": [list(item) for item in action.placements],
                "discards": list(action.discards),
            }
        )
    selected = actions[0]
    return {
        "seat": "first",
        "observation_fingerprint": "1" * 64,
        "action_key_schema": "regular_ofc_action_key_v1",
        "legal_action_set_digest": legal_action_set_digest(actions),
        "action_values": rows,
        "selected_action_key": action_key(selected).to_token(),
        "selected_action": {
            "placements": [list(item) for item in selected.placements],
            "discards": list(selected.discards),
        },
        "selected_selection_ev": 1.0,
        "selected_evaluation_ev": 0.8,
        "candidate_belief_digest": "2" * 64,
        "evaluation_belief_digest": "3" * 64,
        "candidate_rng_digest": "4" * 64,
        "evaluation_rng_digest": "5" * 64,
        "search_contract_digest": "6" * 64,
        "run_id": "hu-m31-step6d-performance-repair-attempt01-v1",
        "continuation_seed": 484_108_071_901,
        "candidate_seed": 482_108_071_901,
        "evaluation_seed": 483_108_071_901,
        "candidate_samples": 8,
        "evaluation_samples": 32,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
        "child_information_set_count": 1234,
        # These fields are deliberately source-specific and must be ignored.
        "native_library_sha256": "7" * 64,
        "native_latency_ms": 123.0,
        "validation_latency_ms": 2.0,
        "total_latency_ms": 125.0,
        "semantic_result_digest": "8" * 64,
        "result_digest": "9" * 64,
    }


def test_performance_schedule_is_exact_contract_cycle_and_disjoint_seed_set() -> None:
    audit = validate_performance_contract(CONTRACT_PATH)
    schedule = schedule_by_name(STEP6D_PERFORMANCE_SCHEDULE)
    rows = [performance_schedule_row(index) for index in PERFORMANCE_HAND_INDICES]
    assert len(rows) == 100
    assert audit["performance_seed_count"] == 600
    assert audit["profile_counts"] == {
        profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES
    }
    assert [row["profile"] for row in rows[:5]] == list(M31_T3_BEHAVIOR_PROFILES)
    assert all(row["budget"] == PERFORMANCE_BUDGET for row in rows)
    assert all(row["training_eligible"] is False for row in rows)
    seeds = [value for row in rows for value in row["seeds"].values()]
    assert len(seeds) == len(set(seeds)) == 600
    assert performance_seed_values(0) == dict(
        zip(schedule.namespace_keys, schedule.namespace_bases, strict=True)
    )
    assert performance_seed_values(99) == {
        key: base + 99 * SEED_STRIDE
        for key, base in zip(
            schedule.namespace_keys, schedule.namespace_bases, strict=True
        )
    }


def test_subset_indices_are_sorted_unique_and_bounded() -> None:
    assert normalize_hand_indices(None) == tuple(range(100))
    assert normalize_hand_indices([9, 1, 4]) == (1, 4, 9)
    with pytest.raises(ValueError, match="unique"):
        normalize_hand_indices([1, 1])
    with pytest.raises(ValueError, match="0..99"):
        normalize_hand_indices([100])
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_hand_indices([])


def test_portable_payload_is_source_and_enumeration_independent() -> None:
    reference = _decision()
    candidate = copy.deepcopy(reference)
    candidate["native_library_sha256"] = "a" * 64
    candidate["native_latency_ms"] = 1.0
    candidate["validation_latency_ms"] = 0.1
    candidate["total_latency_ms"] = 1.1
    candidate["semantic_result_digest"] = "b" * 64
    candidate["result_digest"] = "c" * 64
    candidate["action_values"].reverse()
    candidate["action_values"][0]["original_index"] = 0
    candidate["action_values"][1]["original_index"] = 1
    candidate["selected_action"]["placements"].reverse()
    assert portable_parity_payload(reference) == portable_parity_payload(candidate)
    parity = compare_portable_decisions(reference, candidate)
    assert set(parity) == subject._PARITY_KEYS
    assert all(
        parity[key] is True for key in subject._PARITY_KEYS if key.endswith("_exact")
    )
    assert parity["reference_portable_sha256"] == parity["candidate_portable_sha256"]


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    [
        (lambda value: value["action_values"].pop(), "action_keys_exact"),
        (
            lambda value: value["action_values"][1].__setitem__("selection_ev", 0.51),
            "selection_q_exact",
        ),
        (
            lambda value: value["action_values"][1].__setitem__("evaluation_ev", 0.61),
            "evaluation_q_exact",
        ),
        (
            lambda value: value.__setitem__("candidate_rng_digest", "d" * 64),
            "rng_exact",
        ),
        (
            lambda value: value.__setitem__("child_information_set_count", 1235),
            "child_information_set_count_exact",
        ),
    ],
)
def test_portable_parity_fails_each_required_semantic_family(
    mutation: Any, failed_gate: str
) -> None:
    reference = _decision()
    candidate = copy.deepcopy(reference)
    mutation(candidate)
    if failed_gate == "action_keys_exact":
        # Keep the candidate internally self-consistent while removing one action.
        candidate["legal_action_set_digest"] = hashlib.sha256(
            candidate["action_values"][0]["action_key"].encode("ascii")
        ).hexdigest()
        candidate["action_values"][0]["rank"] = 0
    parity = compare_portable_decisions(reference, candidate)
    assert parity[failed_gate] is False
    assert parity["portable_payload_exact"] is False


def test_selected_action_parity_is_explicit() -> None:
    reference = _decision()
    candidate = copy.deepcopy(reference)
    replacement = _actions()[1]
    candidate["selected_action_key"] = action_key(replacement).to_token()
    candidate["selected_action"] = {
        "placements": [list(item) for item in replacement.placements],
        "discards": list(replacement.discards),
    }
    candidate["selected_selection_ev"] = 0.5
    candidate["selected_evaluation_ev"] = 0.6
    parity = compare_portable_decisions(reference, candidate)
    assert parity["selected_action_exact"] is False
    assert parity["portable_payload_exact"] is False


def test_write_once_is_canonical_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "hand.json"
    subject._write_once(path, {"b": 2, "a": 1})
    assert path.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject._write_once(path, {"a": 1, "b": 2})


def test_only_contract_allocations_are_accepted() -> None:
    for workers, rayon in ((1, 16), (2, 8), (4, 4)):
        subject._validate_allocation(workers, rayon)
    with pytest.raises(ValueError, match="1x16"):
        subject._validate_allocation(2, 16)


def test_subset_smoke_interrupt_and_resume_are_per_hand_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference" / "release" / "engine.dll"
    candidate = tmp_path / "candidate" / "release" / "engine.dll"
    reference.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()

    monkeypatch.setattr(subject, "validate_performance_contract", lambda path: {})
    monkeypatch.setattr(subject, "REFERENCE_NATIVE_LIBRARY_SHA256", reference_sha)
    monkeypatch.setattr(
        subject,
        "_materialize_roots",
        lambda **kwargs: [{"hand_index": index} for index in kwargs["indices"]],
    )

    calls: list[int] = []

    def fake_worker(payload: dict[str, Any]) -> dict[str, Any]:
        index = payload["root"]["hand_index"]
        calls.append(index)
        return {
            "hand_index": index,
            "resumed_hand_count": 0,
            "run_contract_digest": payload["run_contract_digest"],
        }

    monkeypatch.setattr(subject, "_run_hand_worker", fake_worker)
    monkeypatch.setattr(
        subject,
        "_validate_hand_artifact",
        lambda value, **kwargs: dict(value),
    )

    def fake_summary(**kwargs: Any) -> dict[str, Any]:
        return {
            "schema": subject.STEP6D_PERFORMANCE_SUMMARY_SCHEMA,
            "status": "pass",
            "resumed_hand_count": kwargs["resumed_hand_count"],
            "hand_indices": list(kwargs["indices"]),
        }

    monkeypatch.setattr(subject, "_build_summary", fake_summary)
    output = tmp_path / "output"
    interrupted = subject.run_performance_development(
        repository_root=REPO_ROOT,
        output_dir=output,
        reference_library=reference,
        reference_sha256=reference_sha,
        candidate_library=candidate,
        candidate_sha256=candidate_sha,
        workers=1,
        rayon_threads=16,
        indices=[0, 1],
        stop_after_hands=1,
    )
    assert interrupted["status"] == "interrupted_for_resume"
    assert interrupted["completed_hand_count"] == 1
    assert interrupted["pending_hand_count"] == 1
    assert (output / "hands/hand_000.json").is_file()
    assert not (output / "hands/hand_001.json").exists()

    complete = subject.run_performance_development(
        repository_root=REPO_ROOT,
        output_dir=output,
        reference_library=reference,
        reference_sha256=reference_sha,
        candidate_library=candidate,
        candidate_sha256=candidate_sha,
        workers=1,
        rayon_threads=16,
        indices=[0, 1],
    )
    assert complete["status"] == "pass"
    assert complete["resumed_hand_count"] == 1
    assert calls == [0, 1]
    assert (output / "hands/hand_001.json").is_file()
    assert (output / "summary.json").is_file()

    repeated = subject.run_performance_development(
        repository_root=REPO_ROOT,
        output_dir=output,
        reference_library=reference,
        reference_sha256=reference_sha,
        candidate_library=candidate,
        candidate_sha256=candidate_sha,
        workers=1,
        rayon_threads=16,
        indices=[0, 1],
    )
    assert repeated == complete
    assert calls == [0, 1]


def test_runner_source_has_no_historical_raw_label_or_mutation_entrypoint() -> None:
    source = (
        REPO_ROOT / "src/ofc_regular/run_hu_m31_t3_step6d_performance.py"
    ).read_text(encoding="utf-8")
    assert "quality100_validation" not in source
    assert "run_hu_m31_t3_step6c" not in source
    assert "set_current" not in source
    assert "gcloud" not in source.casefold()
    assert ".solve_many(" not in source
    assert '"training_eligible": True' not in source


def test_contract_file_is_not_modified_by_validation() -> None:
    before = CONTRACT_PATH.read_bytes()
    report = validate_performance_contract(CONTRACT_PATH)
    assert report["contract_byte_sha256"] == hashlib.sha256(before).hexdigest()
    assert CONTRACT_PATH.read_bytes() == before


def test_summary_json_fixture_is_strict_json_serializable() -> None:
    payload = portable_parity_payload(_decision())
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert json.loads(encoded) == payload
