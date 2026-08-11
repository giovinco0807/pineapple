from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_abr_v1 as subject
from ofc_regular import (
    hu_m31_t3_locked_promotion_provider_v1 as provider,
)
from ofc_regular import hu_m31_t3_locked_promotion_runner_v1 as runner_module
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from ofc_regular import hu_m31_t3_street_policy_training_v1 as training
from ofc_regular.action_key import ACTION_KEY_SCHEMA, action_key
from ofc_regular.action_space import generate_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.street_policy_net_v1 import load_street_policy_checkpoint


def _training_fixture() -> Any:
    path = Path(__file__).with_name(
        "test_hu_m31_t3_street_policy_training_v1.py"
    )
    name = "_m31_abr_training_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("training fixture cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _threshold_lock(
    model_id: str, model_sha: str, bundle_identity: str
) -> dict[str, Any]:
    return {
        "schema": promotion.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": model_id,
        "model_sha256": model_sha,
        "state_action_input_schema_sha256": "b" * 64,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {"first": 0.8, "second": 0.82},
        "seat_enabled": {"first": True, "second": True},
        "source_training_threshold_lock_sha256": "c" * 64,
        "source_checkpoint_bundle_identity_sha256": bundle_identity,
        "locked_on_threshold_holdout_only": True,
        "model_change_requires_new_lock": True,
        "teacher_ev_lcb_is_runtime_gate": False,
        "top1_accuracy_is_promotion_gate": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _candidate_and_plan(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    torch = pytest.importorskip("torch")
    fixture = _training_fixture()
    config = fixture._config()
    models = training.create_deterministic_ensemble(
        torch,
        training_config=config,
        model_config=fixture._model_config(),
    )
    bundle = tmp_path / "candidate-bundle"
    manifest = training.write_ensemble_checkpoint_bundle(
        bundle,
        models,
        dataset=fixture._dataset(),
        training_config=config,
        stage="risk",
        completed_epoch=config.risk_epochs,
    )
    model_id = "street-policy-net-v1-abr-test"
    model_path = bundle / "manifest.json"
    model_sha = promotion.sha256_file(model_path)
    compact_lock = tmp_path / "compatibility-lock.json"
    compact_lock.write_bytes(
        promotion.canonical_bytes(
            _threshold_lock(
                model_id,
                model_sha,
                manifest["bundle_identity_sha256"],
            )
        )
    )
    registry = tmp_path / "provider-registry.json"
    registry.write_bytes(b"synthetic immutable provider registry")
    runtime = tmp_path / "runtime.py"
    runtime.write_bytes(b"synthetic immutable evaluation runtime")
    plan = promotion.build_locked_promotion_plan(
        plan_id="m31-t3-abr-test-plan",
        model_artifact_id=model_id,
        model_path=model_path,
        expected_model_sha256=model_sha,
        threshold_lock_path=compact_lock,
        expected_threshold_lock_sha256=promotion.sha256_file(
            compact_lock
        ),
        policy_registry_path=registry,
        expected_policy_registry_sha256=promotion.sha256_file(registry),
        evaluation_runtime_closure_path=runtime,
        expected_evaluation_runtime_closure_sha256=(
            promotion.sha256_file(runtime)
        ),
    )
    return plan, bundle, manifest


def _board(cards: list[str], counts: tuple[int, int, int]) -> Board:
    top_count, middle_count, bottom_count = counts
    return Board.from_rows(
        top=cards[:top_count],
        middle=cards[top_count : top_count + middle_count],
        bottom=cards[
            top_count + middle_count :
            top_count + middle_count + bottom_count
        ],
    )


def _observations() -> list[ActorObservation]:
    observations = []
    for seat, offset in (
        ("first", 0),
        ("second", 7),
        ("first", 14),
        ("second", 21),
    ):
        cards = list(ALL_CARDS[offset:] + ALL_CARDS[:offset])
        opponent_count = 9 if seat == "first" else 11
        hero = cards[:9]
        opponent = cards[9 : 9 + opponent_count]
        dealt_start = 9 + opponent_count
        dealt = cards[dealt_start : dealt_start + 3]
        discards = cards[dealt_start + 3 : dealt_start + 5]
        observations.append(
            ActorObservation(
                hero_board=_board(hero, (2, 3, 4)),
                opponent_public_board=_board(
                    opponent,
                    (2, 3, 4) if seat == "first" else (3, 4, 4),
                ),
                dealt_cards=tuple(dealt),
                hero_private_discards=tuple(discards),
                seat=seat,  # type: ignore[arg-type]
                street="T3",
                to_act_order=seat,  # type: ignore[arg-type]
            )
        )
    return observations


class _LegacyPolicy:
    def __init__(self, *, seat: str, policy_seed: int) -> None:
        self.seat = seat
        self.policy_seed = policy_seed
        self.last_action = None
        self.decision_context = {
            "runtime_profile": "stage19_p0",
            "t1_continuation": "stage18_p1",
            "t2_continuation": "stage9f_p2",
        }

    def choose_action_observation(
        self,
        observation: ActorObservation,
        **kwargs: Any,
    ):
        del kwargs
        self.last_action = subject._legal_actions(observation)[0]
        return self.last_action


class _LegacyStage19Factory:
    profile_id = "stage19_p0"
    factory_id = "ai_profiles.build_policy.exact_t4_v1:stage19_p0"

    def __call__(self, *, policy_seed: int, seat: str) -> _LegacyPolicy:
        return _LegacyPolicy(seat=seat, policy_seed=policy_seed)


LEGACY_STAGE19_FACTORY = _LegacyStage19Factory()


def _raw_examples() -> list[dict[str, Any]]:
    rows = []
    for index, observation in enumerate(_observations()):
        actions = subject._legal_actions(observation)
        count = len(actions)
        rows.append(
            {
                "example_id": f"example-{index:02d}",
                "source_seed": 910_000_000_000 + index * 1_000_003,
                "observation": observation.to_dict(),
                "family_action_values": {
                    "greedy_search_response": [
                        action_index / max(1, count - 1)
                        for action_index in range(count)
                    ],
                    "foul_pressure_response": [
                        (count - action_index) / max(1, count)
                        for action_index in range(count)
                    ],
                    "royalty_denial_response": [
                        -abs(action_index - count // 2) / max(1, count)
                        for action_index in range(count)
                    ],
                },
            }
        )
    return rows


@pytest.fixture(scope="module")
def abr_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("m31-abr-v1")
    plan, candidate_bundle, candidate_manifest = _candidate_and_plan(root)
    dataset_path = root / "abr-development.json"
    dataset = subject.write_development_dataset(
        promotion_plan=plan,
        candidate_bundle_manifest=candidate_manifest,
        raw_examples=_raw_examples(),
        output_path=dataset_path,
    )
    first_root = root / "abr-bundle-a"
    first = subject.write_policy_bundle(
        promotion_plan=plan,
        candidate_bundle_directory=candidate_bundle,
        development_dataset_path=dataset_path,
        output_directory=first_root,
        torch=torch,
    )
    second_root = root / "abr-bundle-b"
    second = subject.write_policy_bundle(
        promotion_plan=plan,
        candidate_bundle_directory=candidate_bundle,
        development_dataset_path=dataset_path,
        output_directory=second_root,
        torch=torch,
    )
    return {
        "root": root,
        "plan": plan,
        "candidate_bundle": candidate_bundle,
        "candidate_manifest": candidate_manifest,
        "dataset_path": dataset_path,
        "dataset": dataset,
        "bundle_a_root": first_root,
        "bundle_a": first,
        "bundle_b_root": second_root,
        "bundle_b": second,
    }


def test_development_dataset_is_public_disjoint_and_source_replayable(
    abr_artifacts: dict[str, Any],
) -> None:
    dataset = abr_artifacts["dataset"]
    assert dataset["source_schedule"] == "abr_development"
    assert dataset["locked_evaluation_schedule"] == promotion.LOCKED_ABR
    assert dataset["locked_seed_training_allowed"] is False
    assert dataset["opponent_private_discards_used"] is False
    assert dataset["realized_deck_tail_used"] is False
    assert dataset["current_profile_resolved"] is False
    assert dataset["example_count"] == 4
    assert dataset["teacher_value_perspective"] == (
        "abr_actor_higher_is_better"
    )
    assert tuple(dataset["response_ids"]) == subject.RESPONSE_IDS
    assert subject.validate_development_dataset(
        dataset,
        promotion_plan=abr_artifacts["plan"],
        candidate_bundle_manifest=abr_artifacts["candidate_manifest"],
    ) == dataset

    locked_seed = promotion.seed_values(
        promotion.LOCKED_ABR, 0
    )["hand"]
    bad = deepcopy(_raw_examples())
    bad[0]["source_seed"] = locked_seed
    with pytest.raises(ValueError, match="source seed"):
        subject.build_development_dataset(
            promotion_plan=abr_artifacts["plan"],
            candidate_bundle_manifest=abr_artifacts[
                "candidate_manifest"
            ],
            raw_examples=bad,
        )


def test_three_family_training_is_byte_deterministic_and_diagnostic_only(
    abr_artifacts: dict[str, Any],
) -> None:
    bundle_a = abr_artifacts["bundle_a"]
    bundle_b = abr_artifacts["bundle_b"]
    assert bundle_a == bundle_b
    assert bundle_a["family_count"] == 3
    assert [
        record["response_id"] for record in bundle_a["families"]
    ] == list(subject.RESPONSE_IDS)
    assert bundle_a["locked_seed_training_allowed"] is False
    assert bundle_a["named_profile_added"] is False
    assert bundle_a["current_profile_changed"] is False
    assert bundle_a["runtime_activated"] is False
    for record_a, record_b in zip(
        bundle_a["families"], bundle_b["families"], strict=True
    ):
        assert record_a == record_b
        assert (
            abr_artifacts["bundle_a_root"]
            / record_a["checkpoint_filename"]
        ).read_bytes() == (
            abr_artifacts["bundle_b_root"]
            / record_b["checkpoint_filename"]
        ).read_bytes()
        _model, checkpoint_manifest = load_street_policy_checkpoint(
            abr_artifacts["bundle_a_root"]
            / record_a["checkpoint_filename"],
            torch=pytest.importorskip("torch"),
        )
        checkpoint = checkpoint_manifest["provenance"]
        assert checkpoint_manifest["model_state_sha256"] == checkpoint[
            "candidate_source_model_state_sha256"
        ]
        assert checkpoint["policy_factory_id"] == (
            runner_module.ABR_FACTORY_IDS[record_a["response_id"]]
        )
        assert checkpoint["development_metrics"][
            "development_only_not_locked_promotion_evidence"
        ] is True


@pytest.mark.parametrize("response_id", subject.RESPONSE_IDS)
def test_checkpoint_loaded_factory_replays_semantic_behavior_and_runner_binding(
    abr_artifacts: dict[str, Any], response_id: str
) -> None:
    record = next(
        row
        for row in abr_artifacts["bundle_a"]["families"]
        if row["response_id"] == response_id
    )
    manifest_path = (
        abr_artifacts["bundle_a_root"] / record["manifest_filename"]
    )
    checkpoint_path = (
        abr_artifacts["bundle_a_root"] / record["checkpoint_filename"]
    )
    factory = subject.load_artifact_bound_policy_factory(
        response_id=response_id,
        manifest_path=manifest_path,
        expected_manifest_file_sha256=record["manifest_sha256"],
        checkpoint_path=checkpoint_path,
        expected_checkpoint_file_sha256=record["checkpoint_sha256"],
        promotion_plan=abr_artifacts["plan"],
        candidate_bundle_directory=abr_artifacts["candidate_bundle"],
        torch=pytest.importorskip("torch"),
        legacy_policy_factory=LEGACY_STAGE19_FACTORY,
    )
    _model, checkpoint_manifest = load_street_policy_checkpoint(
        checkpoint_path,
        torch=pytest.importorskip("torch"),
    )
    checkpoint = checkpoint_manifest["provenance"]
    probe = checkpoint["semantic_probes"][0]
    observation = ActorObservation.from_dict(probe["observation"])
    policy = factory(policy_seed=12345, seat=observation.seat)
    selected = policy.choose_action_observation(observation)
    assert action_key(selected).to_token() == probe["selected_action_key"]
    assert policy.opponent_private_discards_used is False
    assert policy.current_profile_resolved is False
    assert policy.abr_policy_artifact_sha256 == record[
        "checkpoint_sha256"
    ]
    assert policy.abr_learned_streets == ("T3",)
    assert policy.abr_legacy_profile_by_street == {
        "T0": "stage19_p0",
        "T1": "stage18_p1",
        "T2": "stage9f_p2",
    }
    assert policy.abr_exact_streets == ("T4",)
    opening_cards = list(ALL_CARDS)
    opponent = (
        Board()
        if observation.seat == "first"
        else _board(opening_cards[:5], (2, 1, 2))
    )
    dealt_start = 0 if observation.seat == "first" else 5
    opening = ActorObservation(
        hero_board=Board(),
        opponent_public_board=opponent,
        dealt_cards=tuple(
            opening_cards[dealt_start : dealt_start + 5]
        ),
        hero_private_discards=(),
        seat=observation.seat,
        street="T0",
        to_act_order=observation.seat,
    )
    delegated = policy.choose_action_observation(opening)
    assert delegated is policy._legacy_policy.last_action

    binding = runner_module.AbrPolicyBinding(
        response_id=response_id,
        policy_factory=factory,
        manifest_path=manifest_path,
        expected_manifest_file_sha256=record["manifest_sha256"],
        checkpoint_path=checkpoint_path,
        expected_checkpoint_file_sha256=record["checkpoint_sha256"],
    )
    validated = runner_module.validate_abr_policy_binding(
        binding, plan=abr_artifacts["plan"]
    )
    assert validated.response_id == response_id


def test_checkpoint_or_semantic_probe_tampering_fails_closed(
    abr_artifacts: dict[str, Any], tmp_path: Path
) -> None:
    record = abr_artifacts["bundle_a"]["families"][0]
    source_checkpoint = (
        abr_artifacts["bundle_a_root"] / record["checkpoint_filename"]
    )
    torch = pytest.importorskip("torch")
    model, checkpoint_manifest = load_street_policy_checkpoint(
        source_checkpoint,
        torch=torch,
    )
    checkpoint = deepcopy(checkpoint_manifest["provenance"])
    checkpoint["weights"][0] += 1.0
    with pytest.raises(ValueError, match="checkpoint boundary"):
        subject.validate_checkpoint(
            checkpoint,
            response_id=record["response_id"],
            promotion_plan=abr_artifacts["plan"],
            candidate_bundle_manifest=abr_artifacts[
                "candidate_manifest"
            ],
            torch=torch,
            model=model,
        )

    checkpoint = deepcopy(checkpoint_manifest["provenance"])
    checkpoint["semantic_probes"][0]["selected_action_key"] = (
        checkpoint["semantic_probes"][1]["selected_action_key"]
    )
    checkpoint["semantic_probe_aggregate_sha256"] = (
        promotion.canonical_sha256(checkpoint["semantic_probes"])
    )
    identity = dict(checkpoint)
    identity.pop("checkpoint_identity_sha256")
    checkpoint["checkpoint_identity_sha256"] = (
        promotion.canonical_sha256(identity)
    )
    with pytest.raises(ValueError, match="semantic probe behavior"):
        subject.validate_checkpoint(
            checkpoint,
            response_id=record["response_id"],
            promotion_plan=abr_artifacts["plan"],
            candidate_bundle_manifest=abr_artifacts[
                "candidate_manifest"
            ],
            torch=torch,
            model=model,
        )


def test_production_provider_loads_exact_three_pinned_bindings(
    abr_artifacts: dict[str, Any],
) -> None:
    bundle_path = abr_artifacts["bundle_a_root"] / "bundle.json"
    exact_t4_solver = object()
    bindings = provider.load_abr_policy_bindings(
        promotion_plan=abr_artifacts["plan"],
        candidate_bundle_directory=abr_artifacts["candidate_bundle"],
        abr_bundle_directory=abr_artifacts["bundle_a_root"],
        expected_abr_bundle_file_sha256=promotion.sha256_file(
            bundle_path
        ),
        torch=pytest.importorskip("torch"),
        legacy_policy_factory=LEGACY_STAGE19_FACTORY,
        exact_t4_solver=exact_t4_solver,
    )
    assert tuple(bindings) == subject.RESPONSE_IDS
    for response_id, binding in bindings.items():
        assert binding.response_id == response_id
        assert binding.policy_factory.factory_id == (
            runner_module.ABR_FACTORY_IDS[response_id]
        )
        assert (
            binding.policy_factory.checkpoint_sha256
            == binding.expected_checkpoint_file_sha256
        )
        runner_module.validate_abr_policy_binding(
            binding,
            plan=abr_artifacts["plan"],
        )
        policy = binding.policy_factory(
            policy_seed=1,
            seat="first",
        )
        assert policy.hu_t4_exact_solver is exact_t4_solver
        assert policy.abr_policy_factory_id == (
            runner_module.ABR_FACTORY_IDS[response_id]
        )

    with pytest.raises(
        provider.LockedPromotionProviderError,
        match="bundle.json SHA-256",
    ):
        provider.load_abr_policy_bindings(
            promotion_plan=abr_artifacts["plan"],
            candidate_bundle_directory=(
                abr_artifacts["candidate_bundle"]
            ),
            abr_bundle_directory=abr_artifacts["bundle_a_root"],
            expected_abr_bundle_file_sha256="0" * 64,
            torch=pytest.importorskip("torch"),
            legacy_policy_factory=LEGACY_STAGE19_FACTORY,
            exact_t4_solver=exact_t4_solver,
        )


def test_provider_cli_is_explicit_and_dormant_until_called(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt = {
        "schema": provider.PROVIDER_RECEIPT_SCHEMA,
        "status": "synthetic-provider-test",
        "current_profile_changed": False,
    }
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return receipt

    monkeypatch.setattr(provider, "run_provider_work_item", fake_run)
    output = tmp_path / "provider-receipt.json"
    code = provider.main(
        [
            "--plan",
            "plan.json",
            "--closure-package",
            "closure.zip",
            "--expected-closure-sha256",
            "1" * 64,
            "--extraction-root",
            "extract",
            "--source-replay-root",
            "repo",
            "--compatibility-threshold-lock",
            "lock.json",
            "--policy-registry",
            "ai_profiles.py",
            "--abr-bundle-directory",
            "abr",
            "--expected-abr-bundle-sha256",
            "2" * 64,
            "--expected-abr-production-build-receipt-sha256",
            "3" * 64,
            "--execution-plan",
            "execution.json",
            "--work-id",
            "population-stage7-m5-r10-0000-0024",
            "--shard-directory",
            "shards",
            "--output-receipt",
            str(output),
        ]
    )
    assert code == 0
    assert captured["work_id"] == (
        "population-stage7-m5-r10-0000-0024"
    )
    assert captured["require_host_target"] is True
    assert (
        captured["expected_abr_production_build_receipt_sha256"]
        == "3" * 64
    )
    assert json.loads(output.read_bytes()) == receipt
    assert json.loads(capsys.readouterr().out) == receipt
