from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ofc_regular.train_hu_m43_attempt08_distilled as distilled
import ofc_regular.hu_m43_attempt08_teacher as teacher
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput, HuM43Attempt05Model
from ofc_regular.hu_m43_attempt08_distilled_model import (
    HU_M43_ATTEMPT08_DISTILLED_FEATURE_DIM,
)
from ofc_regular.hu_m43_attempt08_teacher import (
    ATTEMPT08_FROZEN_MODEL_ID,
    ATTEMPT08_FROZEN_MODEL_SHA256,
    FrozenAttempt08LambdaRanker,
)
from ofc_regular.train_hu_m43_attempt08_distilled import (
    ATTEMPT08_PROFILES,
    Attempt08DistillationState,
    assign_attempt08_identity_folds,
    fit_attempt08_distilled_model,
    prepare_attempt08_distillation_states,
    train_attempt08_distilled_artifact,
)
from ofc_regular.hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from ofc_regular.hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
)
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ROOT = (
    ROOT
    / "outputs/gcp_runs"
    / "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228"
    / "package_src"
)


class _CandidateFold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        count = len(runtime_sample["actions"])
        rank = np.arange(count, dtype=np.float64)
        return Attempt05FoldOutput(
            rank_score=rank,
            gain_probability=np.full(count, 0.5),
            downside_p95=np.full(count, 2.0),
            downside_p99=np.full(count, 4.0),
            downside_max=np.full(count, 6.0),
        )


def _ranker() -> FrozenAttempt08LambdaRanker:
    model = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT08_FROZEN_MODEL_ID,
    )
    return FrozenAttempt08LambdaRanker(
        model=model, artifact_sha256=ATTEMPT08_FROZEN_MODEL_SHA256
    )


def _states(per_profile: int = 2) -> tuple[Attempt08DistillationState, ...]:
    rng = np.random.default_rng(4308)
    states = []
    root = 0
    for profile in ATTEMPT08_PROFILES:
        for local in range(per_profile):
            features = rng.normal(
                size=(9, HU_M43_ATTEMPT08_DISTILLED_FEATURE_DIM)
            ).astype(np.float32)
            relevance = np.zeros(9, dtype=np.int8)
            relevance[(root + local) % 8] = 4
            delta = np.linspace(-2.0, 3.0, 9)
            delta[-1] = 0.0
            safe = np.zeros(9, dtype=np.int8)
            safe[int(np.argmax(relevance))] = 1
            tails = np.column_stack(
                (
                    np.linspace(2.0, 8.0, 9),
                    np.linspace(5.0, 12.0, 9),
                    np.linspace(9.0, 18.0, 9),
                )
            )
            tails[-1] = 0.0
            states.append(
                Attempt08DistillationState(
                    root_index=root,
                    root_profile=profile,
                    observation_fingerprint=(f"{root + 1:064x}"),
                    sample={},
                    baseline_index=8,
                    proposal_indices=tuple(range(9)),
                    features=features,
                    relevance=relevance,
                    delta=delta,
                    safe=safe,
                    tails=tails,
                    selected_action_key=f"selected-{root}",
                    teacher_override_fired=True,
                )
            )
            root += 1
    return tuple(states)


def test_identity_folds_are_deterministic_and_profile_stratified() -> None:
    states = _states(per_profile=5)
    first = assign_attempt08_identity_folds(states)
    second = assign_attempt08_identity_folds(states)
    np.testing.assert_array_equal(first, second)
    for profile in ATTEMPT08_PROFILES:
        assert sorted(
            first[index]
            for index, state in enumerate(states)
            if state.root_profile == profile
        ) == list(range(5))


def test_small_distilled_fit_freezes_oof_cushions_but_not_runtime() -> None:
    config = json.loads(
        (ROOT / "configs" / "hu_joint_policy_m43_attempt08_distillation.json").read_text(
            encoding="utf-8"
        )
    )
    model, report = fit_attempt08_distilled_model(
        _states(), candidate_ranker=_ranker(), config=config, smoke=True
    )
    assert len(model.fold_predictors) == 5
    assert model.safety_enabled is False
    assert model.winner_frozen is False
    assert model.safety_threshold == 0.5
    assert model.minimum_fold_votes == 4
    assert report["states"] == 10
    assert report["audit50_fit_rows"] == 0
    assert report["threshold_sweep_performed"] is False
    assert len(report["conformal_cushions"]) == 3
    assert all(value >= 0.0 for value in report["conformal_cushions"])
    assert report["oof_policy_top1_accuracy_diagnostic_only"] >= 0.0


def test_training_manifest_binds_frozen_source_and_external_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "development.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema": distilled.ATTEMPT08_DEVELOPMENT_DECISION_SCHEMA,
                "decision": "go",
                "search_freeze_authorized": True,
                "source": {"input_jsonl_sha256": input_sha},
            }
        ),
        encoding="utf-8",
    )
    decision_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "selector_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": distilled.ATTEMPT08_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA,
                "status": "single_frozen_gate_evaluation_complete",
                "decision_sha256": decision_sha,
                "merged_sha256": input_sha,
                "decision": "go",
                "search_freeze_authorized": True,
                "gate_evaluation_count": 1,
                "selector_executed": True,
                "fit_performed": False,
                "threshold_selected": False,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        ),
        encoding="utf-8",
    )
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    pass_freeze_path = tmp_path / "development_pass_freeze.json"
    pass_freeze_path.write_text(
        json.dumps(
            {
                "schema": distilled.ATTEMPT08_DEVELOPMENT_PASS_FREEZE_SCHEMA,
                "status": "development_go_frozen_without_future_audit_authorization",
                "development_decision_sha256": decision_sha,
                "selector_receipt_sha256": receipt_sha,
                "search_freeze_authorized": True,
                "future_audit_authorized": False,
                "fit_performed": False,
                "threshold_selected": False,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
                "package_manifest_sha256": "1" * 64,
                "source_zip_sha256": "2" * 64,
                "source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
                "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            }
        ),
        encoding="utf-8",
    )
    fitted_model, diagnostics = fit_attempt08_distilled_model(
        _states(),
        candidate_ranker=_ranker(),
        config=json.loads(
            (ROOT / "configs/hu_joint_policy_m43_attempt08_distillation.json").read_text(
                encoding="utf-8"
            )
        ),
        smoke=True,
    )

    class _Loader:
        @staticmethod
        def load(*_args, **_kwargs):
            return _ranker()

    monkeypatch.setattr(distilled, "FrozenAttempt08LambdaRanker", _Loader)
    monkeypatch.setattr(distilled, "read_attempt08_jsonl", lambda _path: [{}])
    monkeypatch.setattr(
        distilled,
        "prepare_attempt08_distillation_states",
        lambda *_args, **_kwargs: _states(),
    )
    monkeypatch.setattr(
        distilled,
        "fit_attempt08_distilled_model",
        lambda *_args, **_kwargs: (fitted_model, diagnostics),
    )
    runtime_archive = tmp_path / "runtime_source.zip"
    runtime_manifest = tmp_path / "runtime_source.json"
    manifest = train_attempt08_distilled_artifact(
        input_path=input_path,
        candidate_model_path=tmp_path / "candidate.pkl",
        config_path=ROOT / "configs/hu_joint_policy_m43_attempt08_distillation.json",
        development_decision_path=decision_path,
        development_selector_receipt_path=receipt_path,
        development_pass_freeze_path=pass_freeze_path,
        output_model_path=tmp_path / "model.pkl",
        output_manifest_path=tmp_path / "training.json",
        runtime_source_root=ROOT,
        runtime_dependency_root=DEPENDENCY_ROOT,
        output_runtime_source_archive_path=runtime_archive,
        output_runtime_source_manifest_path=runtime_manifest,
        smoke=True,
    )
    runtime = json.loads(runtime_manifest.read_text(encoding="ascii"))
    assert manifest["source"]["runtime_source_archive_sha256"] == runtime["archive"]["sha256"]
    assert manifest["source"]["runtime_source_manifest_sha256"] == hashlib.sha256(
        runtime_manifest.read_bytes()
    ).hexdigest()
    assert manifest["source"]["runtime_source_closure_sha256"] == runtime["file_set"]["sha256"]
    assert manifest["source"]["runtime_semantic_closure_sha256"] == runtime["semantic_closure"]["sha256"]
    assert manifest["runtime"]["runtime_requirements_sha256"] == ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
    assert manifest["runtime"]["runtime_fingerprint_sha256"] == ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
    assert manifest["fit_contract"] == {
        "fit_mode": "smoke",
        "effective_iterations": 24,
        "states": 10,
        "folds": 5,
    }


def test_real_attempt08_teacher_row_projects_to_top8_plus_baseline(monkeypatch) -> None:
    observation = ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    baseline = action_key(legal[len(legal) // 2]).to_token()
    ranker = _ranker()

    def score(_observation, actions, batch, _selector):
        count = len(batch.particles)
        values = [float(len(actions) - index - 1) for index in range(len(actions))]
        return tuple(_ActionScores(tuple(value for _ in range(count))) for value in values)

    monkeypatch.setattr(teacher, "_score_actions", score)
    payload = teacher.evaluate_attempt08_t1_second(
        observation,
        baseline_action_key=baseline,
        ranker=ranker,
        t2_policies={
            "first": type(
                "P",
                (),
                {
                    "seat": "first",
                    "topk_context": {
                        "runtime_profile": "stage9f_p2",
                        "runtime_status": "p2_fixed",
                    },
                },
            )(),
            "second": type(
                "P",
                (),
                {
                    "seat": "second",
                    "topk_context": {
                        "runtime_profile": "stage9f_p2",
                        "runtime_status": "p2_fixed",
                    },
                },
            )(),
        },
        config=teacher.Attempt08TeacherConfig(
            frozen_model_sha256=ATTEMPT08_FROZEN_MODEL_SHA256,
            hand_seed=801,
            rerank_seed=802,
            veto_seed=803,
            stress_seed=804,
            assessment_seed=805,
            child_policy_seed=806,
            run_id="attempt08-distillation-test",
        ),
    )
    states = prepare_attempt08_distillation_states(
        [
            {
                "root_index": 0,
                "root_profile": ATTEMPT08_PROFILES[0],
                "observation_fingerprint": observation.fingerprint(),
                "policy_observation": observation.to_dict(),
                "baseline_action_key": baseline,
                "teacher": payload,
            }
        ],
        ranker=ranker,
        expected_roots=None,
    )
    assert len(states) == 1
    state = states[0]
    assert state.features.shape == (9, HU_M43_ATTEMPT08_DISTILLED_FEATURE_DIM)
    assert state.proposal_indices[-1] == state.baseline_index
    assert int(np.sum(state.relevance == 4)) == 1
    assert int(np.sum(state.safe)) == 1
    assert state.teacher_override_fired is True
