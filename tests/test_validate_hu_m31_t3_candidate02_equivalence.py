from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import validate_hu_m31_t3_candidate02_equivalence as subject
from ofc_regular.action_key import (
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board


def _t3_first_observation() -> ActorObservation:
    # 2/4/3 leaves 1/1/2 slots and therefore the full 21-action T3 geometry.
    hero = ALL_CARDS[:9]
    opponent = ALL_CARDS[9:18]
    return ActorObservation(
        hero_board=Board.from_rows(top=hero[:2], middle=hero[2:6], bottom=hero[6:]),
        opponent_public_board=Board.from_rows(
            top=opponent[:2], middle=opponent[2:6], bottom=opponent[6:]
        ),
        dealt_cards=ALL_CARDS[18:21],
        hero_private_discards=ALL_CARDS[21:23],
        seat="first",
        street="T3",
        to_act_order="first",
    )


def _t4_second_observation() -> ActorObservation:
    hero = ALL_CARDS[:11]
    opponent = ALL_CARDS[11:24]
    return ActorObservation(
        hero_board=Board.from_rows(top=hero[:2], middle=hero[2:6], bottom=hero[6:]),
        opponent_public_board=Board.from_rows(
            top=opponent[:3], middle=opponent[3:8], bottom=opponent[8:]
        ),
        dealt_cards=ALL_CARDS[24:27],
        hero_private_discards=ALL_CARDS[27:30],
        seat="second",
        street="T4",
        to_act_order="second",
    )


def _bare_root(path: Path, observation: ActorObservation | None = None) -> Path:
    observation = observation or _t3_first_observation()
    path.write_text(
        json.dumps(observation.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _step6d_root(path: Path, observation: ActorObservation | None = None) -> Path:
    observation = observation or _t3_first_observation()
    payload = {
        "budget": {
            "candidate_samples": 1,
            "evaluation_samples": 1,
            "downstream_t3_samples": 1,
            "downstream_t4_samples": 0,
        },
        "contract_canonical_sha256": "a" * 64,
        "current_profile_resolved": False,
        "hand_index": 2,
        "observations": [
            {
                "observation": observation.to_dict(),
                "observation_fingerprint": observation.fingerprint(),
                "root_index": 4,
                "seat": observation.seat,
            }
        ],
        "opponent_private_discards_used": False,
        "profile": "stage7_m5_r10",
        "root_indices": [4],
        "schedule": "performance_development",
        "schedule_row_sha256": "b" * 64,
        "schema": subject.STEP6D_ROOT_SCHEMA,
        "seeds": {
            "behavior": 101,
            "candidate": 102,
            "child": 103,
            "confirmation": 104,
            "evaluation": 105,
            "hand": 106,
        },
        "training_eligible": False,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _native_result(request: dict[str, Any]) -> dict[str, Any]:
    observation = ActorObservation.from_dict(request["observation"])
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    rows = []
    for index, action in enumerate(actions):
        value = -float(index)
        rows.append(
            {
                "original_index": index,
                "sorted_index": index,
                "action_key": action_key(action).to_token(),
                "placements": [list(placement) for placement in action.placements],
                "discards": list(action.discards),
                "selection_score": value,
                "score": value,
                "joint_ev": value,
                "selected_by_candidate_plan": index == 0,
            }
        )
    result: dict[str, Any] = {
        "status": "ok",
        "schema": "hu_m3_engine_result_v1",
        "engine_version": "test-m3/1",
        "solver_id": "test-solver",
        "kind": observation.street.lower(),
        "street": observation.street,
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "selected_action_original_index": 0,
        "selected_action_key": action_key(actions[0]).to_token(),
        "actions": rows,
    }
    if observation.street == "T3":
        result["child_information_set_count"] = 777
    return result


def _fake_native(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate_new: Any | None = None,
) -> None:
    monkeypatch.setattr(subject, "load_native_engine", lambda *, path: Path(path).stem)
    monkeypatch.setattr(subject, "engine_version", lambda *, library: "test-m3/1")

    def evaluate(request: dict[str, Any], *, library: str) -> dict[str, Any]:
        result = _native_result(request)
        if library == "new" and mutate_new is not None:
            mutate_new(result, request)
        return result

    monkeypatch.setattr(subject, "evaluate_request", evaluate)


def test_f64_projection_is_type_safe_and_one_ulp_exact() -> None:
    left = {"q": 1.0, "integer": 1, "negative_zero": -0.0}
    same = {"negative_zero": -0.0, "integer": 1, "q": 1.0}
    one_ulp = {"q": math.nextafter(1.0, math.inf), "integer": 1, "negative_zero": -0.0}
    positive_zero = {"q": 1.0, "integer": 1, "negative_zero": 0.0}

    assert subject.compare_raw_results(left, same)["exact"] is True
    assert subject.compare_raw_results(left, one_ulp)["exact"] is False
    assert "f64_bits" in subject.compare_raw_results(left, one_ulp)["first_difference"]
    assert subject.compare_raw_results(left, positive_zero)["exact"] is False
    assert subject.f64_bit_projection(1) != subject.f64_bit_projection(1.0)


def test_received_step6d_root_schema_is_accepted_and_hidden_truth_fails_closed(
    tmp_path: Path,
) -> None:
    path = _step6d_root(tmp_path / "root.json")
    cases = subject.load_root_cases(path)

    assert len(cases) == 1
    assert cases[0].observation.fingerprint() == _t3_first_observation().fingerprint()
    assert cases[0].budget == {
        "candidate_samples": 1,
        "evaluation_samples": 1,
        "downstream_t3_samples": 1,
        "downstream_t4_samples": 0,
    }
    assert cases[0].seeds is not None and cases[0].seeds["child"] == 103

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["opponent_private_discards_used"] = True
    bad = tmp_path / "hidden.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="opponent private discards"):
        subject.load_root_cases(bad)

    payload["opponent_private_discards_used"] = False
    payload["opponent_private_discards"] = ["As"]
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        subject.load_root_cases(unknown)


def test_full_raw_old_new_and_cache_differential_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old.dll"
    new = tmp_path / "new.dll"
    old.write_bytes(b"candidate01")
    new.write_bytes(b"candidate02")
    root = _bare_root(tmp_path / "observation.json")
    _fake_native(monkeypatch)

    artifact = subject.run_differential(
        candidate01_library_path=old,
        candidate02_library_path=new,
        root_paths=[root],
    )

    assert artifact["status"] == "go"
    assert artifact["candidate01"]["sha256"] == subject.sha256_file(old)
    assert artifact["candidate02"]["sha256"] == subject.sha256_file(new)
    assert len(artifact["cases"]) == 1
    case = artifact["cases"][0]
    assert [mode["mode"] for mode in case["modes"]] == ["cache_on", "cache_off"]
    assert all(
        mode["candidate01_vs_candidate02"]["exact"] is True for mode in case["modes"]
    )
    assert case["cache_invariance"]["candidate01"]["exact"] is True
    assert case["cache_invariance"]["candidate02"]["exact"] is True
    assert case["modes"][0]["candidate01"]["legal_action_count"] == 21
    assert case["modes"][0]["candidate01"]["child_information_set_count"] == 777


def test_bare_t4_root_runs_one_exact_full_raw_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old.dll"
    new = tmp_path / "new.dll"
    old.write_bytes(b"candidate01")
    new.write_bytes(b"candidate02")
    root = _bare_root(tmp_path / "t4.json", _t4_second_observation())
    _fake_native(monkeypatch)

    artifact = subject.run_differential(
        candidate01_library_path=old,
        candidate02_library_path=new,
        root_paths=[root],
    )

    assert artifact["status"] == "go"
    assert len(artifact["cases"]) == 1
    case = artifact["cases"][0]
    assert case["street"] == "T4"
    assert case["cache_invariance"] is None
    assert [row["mode"] for row in case["modes"]] == ["exact"]
    assert case["modes"][0]["candidate01_vs_candidate02"]["exact"] is True
    assert case["modes"][0]["candidate01"]["legal_action_count"] == 6


def test_one_ulp_new_q_change_is_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old.dll"
    new = tmp_path / "new.dll"
    old.write_bytes(b"candidate01")
    new.write_bytes(b"candidate02")
    root = _bare_root(tmp_path / "observation.json")

    def mutate(result: dict[str, Any], _request: dict[str, Any]) -> None:
        result["actions"][0]["selection_score"] = math.nextafter(0.0, math.inf)

    _fake_native(monkeypatch, mutate_new=mutate)
    artifact = subject.run_differential(
        candidate01_library_path=old,
        candidate02_library_path=new,
        root_paths=[root],
    )

    assert artifact["status"] == "no_go"
    assert "candidate01/candidate02 mismatch" in artifact["failure"]["message"]
    assert "f64_bits" in artifact["failure"]["message"]


def test_candidate02_cache_only_change_is_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old.dll"
    new = tmp_path / "new.dll"
    old.write_bytes(b"candidate01")
    new.write_bytes(b"candidate02")
    root = _bare_root(tmp_path / "observation.json")

    def mutate(result: dict[str, Any], request: dict[str, Any]) -> None:
        if request["config"]["use_t4_action_cache"] is False:
            # Mutate both libraries below so old/new is exact per mode and only
            # the candidate02 cache-invariance comparison is exercised.
            result["child_information_set_count"] += 1

    monkeypatch.setattr(subject, "load_native_engine", lambda *, path: Path(path).stem)
    monkeypatch.setattr(subject, "engine_version", lambda *, library: "test-m3/1")

    def evaluate(request: dict[str, Any], *, library: str) -> dict[str, Any]:
        result = _native_result(request)
        if library == "new":
            mutate(result, request)
        return result

    monkeypatch.setattr(subject, "evaluate_request", evaluate)
    artifact = subject.run_differential(
        candidate01_library_path=old,
        candidate02_library_path=new,
        root_paths=[root],
    )

    # The cache-off old/new comparison itself already catches this, which is
    # stricter than waiting for the later cache-invariance comparison.
    assert artifact["status"] == "no_go"
    assert "child_information_set_count" in artifact["failure"]["message"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__("legal_action_order_digest", "0" * 64),
            "order digest",
        ),
        (lambda value: value["actions"][0].__setitem__("original_index", 1), "mapping"),
        (
            lambda value: value.__setitem__(
                "selected_action_key",
                "rak1:0000000000000:0000000000000:0000000000000:0000000000000",
            ),
            "selected",
        ),
        (
            lambda value: value.__setitem__("child_information_set_count", -1),
            "child_information_set_count",
        ),
    ],
)
def test_result_structure_fails_closed_on_mapping_selection_and_child_drift(
    mutation: Any, message: str
) -> None:
    observation = _t3_first_observation()
    request = {
        "observation": observation.to_dict(),
    }
    result = _native_result(request)
    corrupted = copy.deepcopy(result)
    mutation(corrupted)
    with pytest.raises((ValueError, TypeError), match=message):
        subject.validate_result_structure(corrupted, observation)


def test_root_and_library_hash_pins_and_immutable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old.dll"
    new = tmp_path / "new.dll"
    old.write_bytes(b"candidate01")
    new.write_bytes(b"candidate02")
    root = _bare_root(tmp_path / "observation.json")
    _fake_native(monkeypatch)

    with pytest.raises(ValueError, match="pinned"):
        subject.run_differential(
            candidate01_library_path=old,
            candidate02_library_path=new,
            root_paths=[root],
            expected_candidate01_sha256="0" * 64,
        )

    artifact = subject.run_differential(
        candidate01_library_path=old,
        candidate02_library_path=new,
        root_paths=[root],
        expected_candidate01_sha256=subject.sha256_file(old),
        expected_candidate02_sha256=subject.sha256_file(new),
    )
    output = tmp_path / "validation.json"
    subject.write_json_once(output, artifact)
    original = output.read_bytes()
    assert json.loads(original)["root_artifacts"][0]["sha256"] == subject.sha256_file(
        root
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.write_json_once(output, {"status": "tampered"})
    assert output.read_bytes() == original
