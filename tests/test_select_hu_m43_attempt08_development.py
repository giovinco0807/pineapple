from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

import ofc_regular.run_hu_m43_attempt08_development as runner
import ofc_regular.select_hu_m43_attempt08_development as selector
import ofc_regular.finalize_hu_m43_attempt08_preflight as finalizer
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
)
from ofc_regular.state import Board
from ofc_regular.run_hu_m43_attempt08_preflight import ATTEMPT08_PREFLIGHT_SLOTS


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt08.json"
PREFLIGHT_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt08_preflight.json"
RUN_NAME = "attempt08-development-selector-test"
_PROOF_HASHES = {
    slot: hashlib.sha256(f"selector-proof:{slot}".encode("utf-8")).hexdigest()
    for slot in ATTEMPT08_PREFLIGHT_SLOTS
}
_AUTH_PAYLOAD = finalizer._authorization_payload(
    aggregate={
        "proof_file_sha256": _PROOF_HASHES,
        "proof_evidence_sha256": finalizer._sha256_value(_PROOF_HASHES),
        "proof_gates": {"all": True},
        "operational_gates": {"all": True},
        "spot_operational_evidence_sha256": "f" * 64,
        "runtime_fingerprint_sha256": (
            finalizer.ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        ),
    },
    aggregate_sha256="d" * 64,
)
_AUTH_BYTES = finalizer.canonical_json_bytes(_AUTH_PAYLOAD)
_AUTH_BINDINGS = {
    "development_open_authorization_sha256": hashlib.sha256(_AUTH_BYTES).hexdigest(),
    "preflight_plan_sha256": _AUTH_PAYLOAD["preflight_plan"]["sha256"],
    "preflight_result_sha256": _AUTH_PAYLOAD["preflight_result"]["sha256"],
    "preflight_proof_evidence_sha256": _AUTH_PAYLOAD["evidence"][
        "proof_evidence_sha256"
    ],
    "preflight_proof_gates_sha256": _AUTH_PAYLOAD["evidence"][
        "proof_gates_sha256"
    ],
    "preflight_operational_gates_sha256": _AUTH_PAYLOAD["evidence"][
        "operational_gates_sha256"
    ],
    "preflight_execution_evidence_sha256": _AUTH_PAYLOAD["evidence"][
        "preflight_execution_evidence_sha256"
    ],
    "runtime_semantic_anchor_sha256": _AUTH_PAYLOAD["evidence"][
        "runtime_semantic_anchor_sha256"
    ],
    "runtime_source_closure_sha256": _AUTH_PAYLOAD["evidence"][
        "runtime_source_closure_sha256"
    ],
    "runtime_fingerprint_sha256": _AUTH_PAYLOAD["evidence"][
        "runtime_fingerprint_sha256"
    ],
    "runtime_requirements_sha256": _AUTH_PAYLOAD["evidence"][
        "runtime_requirements_sha256"
    ],
    "gcp_image_name": _AUTH_PAYLOAD["evidence"]["gcp_image_name"],
    "gcp_image_id": _AUTH_PAYLOAD["evidence"]["gcp_image_id"],
}


def _root(root_index: int) -> ActorObservation:
    cards = random.Random(800_000 + root_index).sample(list(ALL_CARDS), 15)
    return ActorObservation(
        hero_board=Board.from_rows(
            top=(cards[0],),
            middle=(cards[1], cards[2]),
            bottom=(cards[3], cards[4]),
        ),
        opponent_public_board=Board.from_rows(
            top=(cards[5],),
            middle=(cards[6], cards[7], cards[8]),
            bottom=(cards[9], cards[10], cards[11]),
        ),
        dealt_cards=(cards[12], cards[13], cards[14]),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def development_rows() -> list[dict]:
    plan = load_and_validate_attempt08_plan(PLAN)
    schedules = enumerate_attempt08_seed_schedules(plan, population="development")
    rows: list[dict] = []
    for root_index in range(200):
        observation = _root(root_index)
        actions = generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
        baseline_token = action_key(actions[-1]).to_token()
        fired = root_index % 4 == 0
        selected_token = (
            action_key(actions[0]).to_token() if fired else baseline_token
        )
        delta = 2.0 if fired else 0.0
        seeds = {domain: values[root_index] for domain, values in schedules.items()}
        profile = M43_ATTEMPT08_PROFILES[root_index % 5]
        run_id = f"{RUN_NAME}:shard={root_index}"
        fixed = runner._fixed_contract(
            root_index=root_index,
            root_profile=profile,
            seeds=seeds,
            run_id=run_id,
            plan_sha256=M43_ATTEMPT08_PLAN_SHA256,
            ai_profiles_sha256=AI_PROFILES_SHA256,
            model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
            authorization_bindings=_AUTH_BINDINGS,
            batch_child_selectors=True,
            native_batch_threads=4,
        )
        config_sha256 = runner._canonical_sha256(fixed)
        root_input_sha256 = runner._root_input_sha256(
            root_index=root_index,
            hand_seed=seeds["hand"],
            root_profile=profile,
            observation=observation,
            baseline_action_key=baseline_token,
        )
        provenance = runner._expected_provenance(
            fixed_contract=fixed,
            config_sha256=config_sha256,
            root_input_sha256=root_input_sha256,
        )
        rows.append(
            {
                "schema": runner.ATTEMPT08_SHARD_ROW_SCHEMA,
                "root_index": root_index,
                "hand_seed": seeds["hand"],
                "root_profile": profile,
                "policy_observation": observation.to_dict(),
                "baseline_action_key": baseline_token,
                "provenance": provenance,
                "teacher": {
                    "fake_selected_action_key": selected_token,
                    "fake_override_fired": fired,
                    "fake_exact_baseline_fallback": not fired,
                    "fake_assessment_raw": [delta] * 256 if fired else [],
                    "fake_rng": {
                        "rerank_r128": [
                            _digest(f"rng:{root_index}:rerank:{index}")
                            for index in range(2)
                        ]
                    },
                    "fake_belief": {
                        "rerank_r128": _digest(f"belief:{root_index}:rerank")
                    },
                    "fake_seeds": seeds,
                    "fake_observation_fingerprint": observation.fingerprint(),
                    "fake_baseline_action_key": baseline_token,
                },
            }
        )
    return rows


def _fake_teacher_validator(
    observation,
    *,
    baseline_action_key,
    payload,
    config,
):
    assert payload["fake_observation_fingerprint"] == observation.fingerprint()
    assert payload["fake_baseline_action_key"] == baseline_action_key
    assert payload["fake_seeds"] == {
        "hand": config.hand_seed,
        "rerank": config.rerank_seed,
        "veto": config.veto_seed,
        "stress": config.stress_seed,
        "assessment": config.assessment_seed,
        "child": config.child_policy_seed,
    }
    return {
        "selected_action_key": payload["fake_selected_action_key"],
        "override_fired": payload["fake_override_fired"],
        "exact_baseline_fallback": payload["fake_exact_baseline_fallback"],
        "assessment_raw_paired_deltas_vs_baseline": payload[
            "fake_assessment_raw"
        ],
        "rng_key_digests": payload["fake_rng"],
        "belief_digests": payload["fake_belief"],
    }


def _aggregate(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> dict:
    monkeypatch.setattr(
        selector, "validate_attempt08_teacher_output", _fake_teacher_validator
    )
    return selector.aggregate_attempt08_development_rows(
        rows,
        plan=load_and_validate_attempt08_plan(PLAN),
        source_input_sha256="a" * 64,
        source_plan_sha256=M43_ATTEMPT08_PLAN_SHA256,
        authorization_bindings=_AUTH_BINDINGS,
        run_name=RUN_NAME,
    )


def test_development200_go_applies_frozen_a256_gates_once(
    development_rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _aggregate(monkeypatch, development_rows)
    observed = {gate["name"]: gate for gate in report["gates"]}

    assert report["schema"] == "hu_m43_attempt08_development_go_no_go_v1"
    assert report["decision"] == "go"
    assert report["search_freeze_authorized"] is True
    assert report["selected_arm"] is None
    assert report["selected_threshold"] is None
    assert report["decision_contract"] == {
        "single_frozen_search_architecture": True,
        "arm_selection_performed": False,
        "threshold_selection_performed": False,
        "gate_evaluation_count": 1,
        "all_gates_required": True,
    }
    assert report["development_population"]["profile_counts"] == {
        profile: 40 for profile in M43_ATTEMPT08_PROFILES
    }
    assert report["metrics"]["overall"]["fires"] == 50
    assert report["metrics"]["overall"]["mean_delta_per_state"] == pytest.approx(
        0.5
    )
    assert report["metrics"]["overall"]["mean_delta_per_fire"] == pytest.approx(
        2.0
    )
    assert all(gate["passed"] for gate in observed.values())
    assert observed["fires_total"]["requirement"] == ">= 40"
    assert observed["fires_each_profile"]["requirement"] == "each >= 3"


def test_go_authorizes_only_separate_search_freeze(
    development_rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _aggregate(monkeypatch, development_rows)
    boundary = report["science_boundary"]

    assert report["decision_scope"].startswith(
        "write_separate_immutable_search_freeze"
    )
    assert boundary["future_audit_directly_authorized"] is False
    assert boundary["fit_performed"] is False
    assert boundary["threshold_selected"] is False
    assert boundary["runtime_policy_activated"] is False
    assert boundary["current_profile_mutated"] is False
    assert boundary["teacher_values_are_realized_match_ev"] is False


def test_any_frozen_tail_gate_failure_is_no_go(
    development_rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = copy.deepcopy(development_rows)
    raw = [2.0] * 256
    raw[0] = -60.0
    rows[0]["teacher"]["fake_assessment_raw"] = raw
    report = _aggregate(monkeypatch, rows)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["decision"] == "no_go"
    assert report["search_freeze_authorized"] is False
    assert gates["maximum_per_fired_root_max_loss"]["passed"] is False
    assert gates["maximum_per_fired_root_max_loss"]["observed"] == 60.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("profile", "profile assignment"),
        ("seed", "provenance seeds|hand seed"),
        ("hidden", "hidden-information"),
        ("nonfire", "nonfire lacks exact baseline"),
        ("rng", "globally unique"),
    ),
)
def test_selector_rejects_tampered_contract_semantics(
    development_rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    rows = copy.deepcopy(development_rows)
    if mutation == "profile":
        rows[0]["root_profile"] = "stage3_baseline"
    elif mutation == "seed":
        rows[0]["provenance"]["seeds"]["assessment"] += 1
    elif mutation == "hidden":
        rows[0]["teacher"]["opponent_private_discards"] = ["As"]
    elif mutation == "nonfire":
        observation = ActorObservation.from_dict(rows[1]["policy_observation"])
        rows[1]["teacher"]["fake_selected_action_key"] = action_key(
            generate_turn_actions(
                observation.hero_board, observation.dealt_cards
            )[0]
        ).to_token()
    elif mutation == "rng":
        rows[1]["teacher"]["fake_rng"] = copy.deepcopy(
            rows[0]["teacher"]["fake_rng"]
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises((ValueError, AssertionError), match=match):
        _aggregate(monkeypatch, rows)


def test_selector_rejects_duplicate_or_incomplete_root_set(
    development_rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = copy.deepcopy(development_rows)
    duplicate[-1]["root_index"] = 0
    with pytest.raises(ValueError, match="duplicate root_index"):
        _aggregate(monkeypatch, duplicate)
    with pytest.raises(ValueError, match="exactly 200 rows"):
        _aggregate(monkeypatch, development_rows[:-1])


def test_canonical_jsonl_selector_and_no_clobber_writer(
    development_rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        selector, "validate_attempt08_teacher_output", _fake_teacher_validator
    )
    input_path = tmp_path / "development.jsonl"
    input_path.write_bytes(
        b"".join(selector._canonical_json_bytes(row) for row in development_rows)
    )
    authorization_path = tmp_path / "development-open.json"
    authorization_path.write_bytes(_AUTH_BYTES)
    report = selector.select_attempt08_development(
        input_path=input_path,
        plan_path=PLAN,
        preflight_plan_path=PREFLIGHT_PLAN,
        development_open_authorization_path=authorization_path,
        run_name=RUN_NAME,
    )
    output = tmp_path / "decision.json"

    selector.write_attempt08_development_decision(
        output,
        report,
        _lifecycle_token=selector._SELECTOR_WRITE_LIFECYCLE_TOKEN,
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        selector.write_attempt08_development_decision(
            output,
            report,
            _lifecycle_token=selector._SELECTOR_WRITE_LIFECYCLE_TOKEN,
        )
    assert output.read_bytes() == before


def test_selector_rejects_noncanonical_jsonl(
    development_rows: list[dict], tmp_path: Path
) -> None:
    path = tmp_path / "development.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, indent=2) for row in development_rows) + "\n",
        encoding="utf-8",
    )
    authorization_path = tmp_path / "development-open.json"
    authorization_path.write_bytes(_AUTH_BYTES)
    with pytest.raises(
        ValueError, match="exactly 200 rows|canonical JSONL|canonical LF JSONL"
    ):
        selector.select_attempt08_development(
            input_path=path,
            plan_path=PLAN,
            preflight_plan_path=PREFLIGHT_PLAN,
            development_open_authorization_path=authorization_path,
            run_name=RUN_NAME,
        )


def test_direct_decision_write_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="run-global Spot selector lifecycle"):
        selector.write_attempt08_development_decision(
            tmp_path / "alternate-decision.json", {"decision": "go"}
        )
