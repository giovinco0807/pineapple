from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

import ofc_regular.select_hu_m43_attempt09_development as selector
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.generate_hu_m4_t1_data import _profile_policy_seed
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt09_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT09_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT09_PLAN_SHA256,
    M43_ATTEMPT09_PROFILES,
    enumerate_attempt09_seed_schedules,
    load_and_validate_attempt09_plan,
)
from ofc_regular.run_hu_m43_attempt09 import (
    ATTEMPT09_AUTHORIZATION_SCHEMA,
    ATTEMPT09_BASELINE_PROFILE,
    ATTEMPT09_CONTINUATION_PROFILE,
    ATTEMPT09_ROW_SCHEMA,
    _canonical_json_bytes,
    _canonical_sha256,
)
from ofc_regular.state import Board


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt09.json"
RUN_NAME = "attempt09-selector-test"
AUTH_SHA = "a" * 64
PACKAGE_SHA = "b" * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _observation(root: int) -> ActorObservation:
    cards = random.Random(9_000_000 + root).sample(list(ALL_CARDS), 15)
    return ActorObservation(
        hero_board=Board.from_rows(top=(cards[0],), middle=(cards[1], cards[2]), bottom=(cards[3], cards[4])),
        opponent_public_board=Board.from_rows(top=(cards[5],), middle=(cards[6], cards[7], cards[8]), bottom=(cards[9], cards[10], cards[11])),
        dealt_cards=(cards[12], cards[13], cards[14]), hero_private_discards=(),
        seat="second", street="T1", to_act_order="second",
    )


@pytest.fixture(scope="module")
def development_rows() -> list[dict]:
    plan = load_and_validate_attempt09_plan(PLAN)
    schedules = enumerate_attempt09_seed_schedules(plan, population="development")
    rows = []
    for root in range(200):
        obs = _observation(root)
        actions = generate_turn_actions(obs.hero_board, obs.dealt_cards)
        baseline = action_key(actions[-1]).to_token()
        fired = root % 4 == 0
        selected = action_key(actions[0]).to_token() if fired else baseline
        profile = M43_ATTEMPT09_PROFILES[root % 5]
        seeds = {name: values[root] for name, values in schedules.items()}
        run_id = f"{RUN_NAME}:development:root={root}:search"
        fixed = {
            "schema": "hu_m43_attempt09_root_contract_v1", "mode": "development",
            "root_index": root, "root_profile": profile, "run_id": run_id, "seeds": seeds,
            "plan_sha256": M43_ATTEMPT09_PLAN_SHA256, "ai_profiles_sha256": AI_PROFILES_SHA256,
            "model_sha256": ATTEMPT09_LAMBDA_MODEL_SHA256, "authorization_sha256": AUTH_SHA,
            "source_package_sha256": PACKAGE_SHA, "batch_child_selectors": True,
            "native_batch_threads": 4,
        }
        teacher_run_id = f"{run_id}:root={root}:seed={seeds['hand']}:obs={obs.fingerprint()}"
        search_config = {
            "learned_nonbaseline_top_k": 8, "baseline_added_exactly_once": True,
            "rerank_samples": 128, "rerank_top_k": 3, "risk_reserve_count": 1, "k4_size": 4,
            "veto_samples": 256, "stress_samples": 512, "confirmation_samples": 256,
            "evaluation_samples": 256, "hand_seed": seeds["hand"], "rerank_seed": seeds["rerank"],
            "veto_seed": seeds["veto"], "stress_seed": seeds["stress"],
            "confirmation_seed": seeds["confirmation"], "evaluation_seed": seeds["evaluation"],
            "child_policy_seed": seeds["child"], "run_id": teacher_run_id,
            "batch_child_selectors": True,
        }
        rows.append({
            "schema": ATTEMPT09_ROW_SCHEMA, "root_index": root, "hand_seed": seeds["hand"],
            "root_profile": profile, "policy_observation": obs.to_dict(),
            "baseline_action_key": baseline,
            "provenance": {
                **fixed, "config_sha256": _canonical_sha256(fixed),
                "root_policy_profiles": {"first": profile, "second": profile},
                "root_policy_seeds": {seat: _profile_policy_seed(seeds["hand"], profile, seat) for seat in ("first", "second")},
                "baseline_profile": ATTEMPT09_BASELINE_PROFILE,
                "baseline_policy_seed": _profile_policy_seed(seeds["hand"], ATTEMPT09_BASELINE_PROFILE, "second"),
                "continuation_profile": ATTEMPT09_CONTINUATION_PROFILE,
                "continuation_policy_seeds": {"first": seeds["child"], "second": seeds["child"] + 1},
                "explicit_loaded_profiles": sorted({profile, ATTEMPT09_BASELINE_PROFILE, ATTEMPT09_CONTINUATION_PROFILE}),
                "opponent_private_discard_input_allowed": False,
                "opponent_profile_runtime_feature_allowed": False,
                "teacher_values_are_realized_match_ev": False,
                "teacher_ev_or_lcb_runtime_gate_allowed": False,
                "development_only": True, "fit_allowed": False,
                "threshold_selection_allowed": False, "runtime_activation_allowed": False,
                "current_profile_resolved": False, "elapsed_seconds": 1.0, "peak_rss_bytes": 1024,
            },
            "teacher": {
                "fake_selected_action_key": selected, "fake_override_fired": fired,
                "fake_exact_baseline_fallback": not fired,
                "fake_observation_fingerprint": obs.fingerprint(),
                "search_config": search_config,
                "seed_domain_provenance": {
                    "domain_order": ["hand_external", "rerank_r128", "veto_v256", "stress_x512", "confirmation_c256", "evaluation_e256", "child_policy"],
                    "hand_external": seeds["hand"], "rerank_r128": seeds["rerank"],
                    "veto_v256": seeds["veto"], "stress_x512": seeds["stress"],
                    "confirmation_c256": seeds["confirmation"], "evaluation_e256": seeds["evaluation"],
                    "child_policy": seeds["child"], "all_seven_base_seeds_pairwise_distinct": True,
                    "hand_sampled_inside_teacher": False,
                },
                "evaluation": ({"opened": True, "actions": [
                    {"action_key": selected, "raw_paired_deltas_vs_baseline": [2.0] * 256},
                    {"action_key": baseline, "raw_paired_deltas_vs_baseline": [0.0] * 256},
                ]} if fired else {"opened": False, "actions": []}),
                "rng_key_digests": {"rerank_r128": [_digest(f"rng:{root}:0"), _digest(f"rng:{root}:1")]},
                "belief_digests": {"rerank_r128": _digest(f"belief:{root}")},
            },
        })
    return rows


def _fake_validator(observation, *, baseline_action_key, payload, config):
    assert payload["fake_observation_fingerprint"] == observation.fingerprint()
    assert payload["search_config"]["run_id"] == config.run_id
    return {
        "schema": "hu_m43_attempt09_teacher_validation_v1",
        "selected_action_key": payload["fake_selected_action_key"],
        "override_fired": payload["fake_override_fired"],
        "exact_baseline_fallback": payload["fake_exact_baseline_fallback"],
        "opened_phases": list(payload["rng_key_digests"]),
    }


def _aggregate(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> dict:
    monkeypatch.setattr(selector, "validate_attempt09_teacher_output", _fake_validator)
    return selector.aggregate_attempt09_development_rows(rows,
        plan=load_and_validate_attempt09_plan(PLAN), source_input_sha256="c" * 64,
        source_plan_sha256=M43_ATTEMPT09_PLAN_SHA256, authorization_sha256=AUTH_SHA,
        source_package_sha256=PACKAGE_SHA, run_name=RUN_NAME)


def test_clean_development200_is_go(development_rows, monkeypatch):
    report = _aggregate(monkeypatch, development_rows)
    assert report["decision"] == "go"
    assert report["search_freeze_authorized"] is True
    assert report["metrics"]["overall"]["fires"] == 50
    assert report["metrics"]["overall"]["mean_delta_per_state"] == pytest.approx(.5)
    assert all(gate["passed"] for gate in report["gates"])
    assert report["science_boundary"]["future_audit_authorized"] is False
    assert report["science_boundary"]["runtime_policy_activated"] is False


def test_root72_like_tail_is_no_go(development_rows, monkeypatch):
    rows = copy.deepcopy(development_rows)
    rows[72]["teacher"]["evaluation"]["actions"][0]["raw_paired_deltas_vs_baseline"][0] = -52.45404122936691
    report = _aggregate(monkeypatch, rows)
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert report["decision"] == "no_go"
    assert report["search_freeze_authorized"] is False
    assert gates["maximum_per_fired_root_max_loss"]["passed"] is False


@pytest.mark.parametrize("tamper", ["profile", "provenance", "hidden", "rng"])
def test_identity_and_information_tampering_fails_closed(development_rows, monkeypatch, tamper):
    rows = copy.deepcopy(development_rows)
    if tamper == "profile":
        rows[0]["root_profile"] = M43_ATTEMPT09_PROFILES[1]
    elif tamper == "provenance":
        rows[0]["provenance"]["continuation_policy_seeds"]["second"] += 1
    elif tamper == "hidden":
        rows[0]["teacher"]["opponent_private_discards"] = ["As"]
    else:
        rows[1]["teacher"]["rng_key_digests"] = copy.deepcopy(rows[0]["teacher"]["rng_key_digests"])
    with pytest.raises((ValueError, AssertionError)):
        _aggregate(monkeypatch, rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_exact_ordered_200_root_set_required(development_rows, monkeypatch, mutation):
    rows = list(development_rows)
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        _aggregate(monkeypatch, rows)


def _authorization(tmp_path: Path) -> Path:
    gate = tmp_path / "preceding_gate.json"
    gate.write_bytes(b"gate\n")
    payload = {
        "schema": ATTEMPT09_AUTHORIZATION_SCHEMA, "status": "authorized", "mode": "development",
        "plan_sha256": M43_ATTEMPT09_PLAN_SHA256, "source_package_sha256": PACKAGE_SHA,
        "preceding_gate_artifact": str(gate),
        "preceding_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "root_index_first": 0, "root_index_last": 199,
        "current_profile_mutated": False, "runtime_policy_activated": False,
    }
    path = tmp_path / "authorization.json"
    path.write_bytes(_canonical_json_bytes(payload))
    return path


def _with_authorization(rows: list[dict], authorization: Path) -> list[dict]:
    rebound = copy.deepcopy(rows)
    auth_sha = hashlib.sha256(authorization.read_bytes()).hexdigest()
    fixed_keys = (
        "schema", "mode", "root_index", "root_profile", "run_id", "seeds",
        "plan_sha256", "ai_profiles_sha256", "model_sha256",
        "authorization_sha256", "source_package_sha256",
        "batch_child_selectors", "native_batch_threads",
    )
    for row in rebound:
        provenance = row["provenance"]
        provenance["authorization_sha256"] = auth_sha
        provenance["config_sha256"] = _canonical_sha256(
            {key: provenance[key] for key in fixed_keys}
        )
    return rebound


def test_canonical_input_atomic_pair_and_single_execution(development_rows, monkeypatch, tmp_path):
    monkeypatch.setattr(selector, "validate_attempt09_teacher_output", _fake_validator)
    auth = _authorization(tmp_path)
    rows = _with_authorization(development_rows, auth)
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_bytes(b"".join(_canonical_json_bytes(row) for row in rows))
    output = tmp_path / "selector"
    receipt = selector.execute_attempt09_development_selector(input_path=input_path, plan_path=PLAN,
        authorization_path=auth, source_package_sha256=PACKAGE_SHA, run_name=RUN_NAME, output_dir=output)
    decision_bytes = (output / "decision.json").read_bytes()
    assert receipt["decision_sha256"] == hashlib.sha256(decision_bytes).hexdigest()
    assert (output / "decision_receipt.json").is_file()
    assert json.loads(decision_bytes)["decision"] == "go"
    with pytest.raises(FileExistsError):
        selector.execute_attempt09_development_selector(input_path=input_path, plan_path=PLAN,
            authorization_path=auth, source_package_sha256=PACKAGE_SHA, run_name=RUN_NAME, output_dir=output)


def test_noncanonical_jsonl_is_rejected(development_rows, monkeypatch, tmp_path):
    monkeypatch.setattr(selector, "validate_attempt09_teacher_output", _fake_validator)
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in development_rows))
    with pytest.raises(ValueError, match="canonical"):
        selector.select_attempt09_development(input_path=input_path, plan_path=PLAN,
            authorization_path=_authorization(tmp_path), source_package_sha256=PACKAGE_SHA,
            run_name=RUN_NAME)
