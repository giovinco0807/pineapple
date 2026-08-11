from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import ofc_regular.train_hu_m43_attempt10_distilled as distilled
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput, HuM43Attempt05Model
from ofc_regular.hu_m43_attempt10_distilled_model import (
    HU_M43_ATTEMPT10_DISTILLED_FEATURE_DIM,
    build_attempt10_distilled_features,
)
from ofc_regular.hu_m43_attempt10_teacher import (
    ATTEMPT10_FROZEN_MODEL_ID,
    ATTEMPT10_FROZEN_MODEL_SHA256,
    ATTEMPT10_TEACHER_SCHEMA,
    FrozenAttempt10LambdaRanker,
)
from ofc_regular.state import Board
from ofc_regular.train_hu_m43_attempt10_distilled import (
    ATTEMPT10_PROFILES,
    Attempt10DistillationState,
    assign_attempt10_identity_folds,
    fit_attempt10_distilled_model,
    prepare_attempt10_distillation_states,
)
from ofc_regular.hu_turn3_model import hu_policy_sample


ROOT = Path(__file__).resolve().parents[1]


class _CandidateFold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        count = len(runtime_sample["actions"])
        return Attempt05FoldOutput(
            rank_score=np.arange(count, dtype=np.float64),
            gain_probability=np.full(count, 0.5),
            downside_p95=np.full(count, 2.0),
            downside_p99=np.full(count, 4.0),
            downside_max=np.full(count, 6.0),
        )


def _ranker() -> FrozenAttempt10LambdaRanker:
    model = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT10_FROZEN_MODEL_ID,
    )
    return FrozenAttempt10LambdaRanker(
        model=model, artifact_sha256=ATTEMPT10_FROZEN_MODEL_SHA256
    )


def _observation() -> ActorObservation:
    return ActorObservation(
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


def _summary(raw: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(raw)),
        "standard_error": float(np.std(raw, ddof=1) / np.sqrt(len(raw))),
        "p05": float(np.quantile(raw, 0.05, method="linear")),
        "p01": float(np.quantile(raw, 0.01, method="linear")),
        "min": float(np.min(raw)),
        "max": float(np.max(raw)),
    }


def _development_row(*, e256_mean: float = 1.0) -> tuple[dict, str]:
    observation = _observation()
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    baseline_index = len(legal) // 2
    baseline_token = action_key(legal[baseline_index]).to_token()
    sample = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        legal,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat="second",
        to_act_order="second",
    )
    sample["policy_observation"] = observation.to_dict()
    sample["baseline_action_row_index"] = baseline_index
    sample["baseline_action_key"] = baseline_token
    built = build_attempt10_distilled_features(
        sample,
        candidate_generator=_ranker().model,
        source_candidate_sha256=ATTEMPT10_FROZEN_MODEL_SHA256,
        baseline_index=baseline_index,
    )
    tokens = [built.action_keys[index].to_token() for index in built.top12_indices]
    proposal_tokens = [*tokens, baseline_token]
    selected = tokens[0]
    rerank_rows = []
    for index, token in enumerate(proposal_tokens):
        raw = np.zeros(128) if index == 12 else np.full(128, 2.0 - index / 10.0)
        rerank_rows.append(
            {
                "action_key": token,
                "raw_paired_deltas_vs_baseline": raw.tolist(),
                "paired_delta_vs_baseline": _summary(raw),
            }
        )
    evaluation_raw = np.full(256, e256_mean)
    teacher = {
        "schema": ATTEMPT10_TEACHER_SCHEMA,
        "policy_observation": observation.to_dict(),
        "baseline_action_key": baseline_token,
        "runtime_gate_allowed": False,
        "learned_top12_action_keys": tokens,
        "search_config": {
            "hand_seed": 1001,
            "rerank_seed": 1002,
            "veto_seed": 1003,
            "stress_seed": 1004,
            "confirmation_seed": 1005,
            "evaluation_seed": 1006,
            "child_policy_seed": 1007,
            "run_id": "attempt10-distillation-unit",
            "batch_child_selectors": False,
        },
        "rerank": {"actions": rerank_rows},
        "decision": {
            "final_selected_action_key": selected,
            "override_fired": True,
            "exact_baseline_fallback": False,
        },
        "evaluation": {
            "opened": True,
            "actions": [
                {
                    "action_key": selected,
                    "raw_paired_deltas_vs_baseline": evaluation_raw.tolist(),
                },
                {
                    "action_key": baseline_token,
                    "raw_paired_deltas_vs_baseline": np.zeros(256).tolist(),
                },
            ],
        },
    }
    row = {
        "schema": distilled.ATTEMPT10_DEVELOPMENT_ROW_SCHEMA,
        "root_index": 0,
        "root_profile": ATTEMPT10_PROFILES[0],
        "policy_observation": observation.to_dict(),
        "baseline_action_key": baseline_token,
        "provenance": {
            "mode": "development",
            "development_only": True,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
        },
        "teacher": teacher,
    }
    return row, selected


def _states(per_profile: int = 2) -> tuple[Attempt10DistillationState, ...]:
    rng = np.random.default_rng(4310)
    states = []
    root = 0
    for profile in ATTEMPT10_PROFILES:
        for local in range(per_profile):
            features = rng.normal(
                size=(13, HU_M43_ATTEMPT10_DISTILLED_FEATURE_DIM)
            ).astype(np.float32)
            relevance = np.zeros(13, dtype=np.int8)
            relevance[(root + local) % 12] = 4
            delta = np.linspace(-2.0, 3.0, 13)
            delta[-1] = 0.0
            safe = np.zeros(13, dtype=np.int8)
            safe[int(np.argmax(relevance))] = 1
            tails = np.column_stack(
                (
                    np.linspace(2.0, 8.0, 13),
                    np.linspace(5.0, 12.0, 13),
                    np.linspace(9.0, 18.0, 13),
                )
            )
            tails[-1] = 0.0
            states.append(
                Attempt10DistillationState(
                    root_index=root,
                    root_profile=profile,
                    observation_fingerprint=f"{root + 1:064x}",
                    sample={},
                    baseline_index=12,
                    proposal_indices=tuple(range(13)),
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


def test_config_freezes_top12_development_only_and_e256_label() -> None:
    path = ROOT / "configs" / "hu_joint_policy_m43_attempt10_distillation.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["candidate_generation"]["nonbaseline_top_k"] == 12
    assert config["data"]["development_roots"] == 200
    assert config["data"]["audit50_may_fit_or_calibrate"] is False
    assert "E256" in config["labels"]["safe_positive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        distilled.ATTEMPT10_DISTILLATION_CONFIG_SHA256
    )


def test_prepare_uses_public_validator_r128_and_independent_e256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, selected = _development_row(e256_mean=1.0)
    calls = []

    def validate(observation, *, baseline_action_key, payload, config):
        calls.append((observation, baseline_action_key, config.evaluation_seed))
        return {
            "selected_action_key": selected,
            "override_fired": True,
        }

    monkeypatch.setattr(distilled, "validate_attempt10_teacher_output", validate)
    state = prepare_attempt10_distillation_states(
        [row], ranker=_ranker(), expected_roots=None
    )[0]
    assert calls and calls[0][2] == 1006
    assert state.features.shape == (13, HU_M43_ATTEMPT10_DISTILLED_FEATURE_DIM)
    assert state.selected_action_key == selected
    assert state.delta[0] == pytest.approx(2.0)
    assert state.delta[-1] == 0.0
    assert state.safe[np.argmax(state.relevance)] == 1

    negative = deepcopy(row)
    raw = np.full(256, -1.0).tolist()
    negative["teacher"]["evaluation"]["actions"][0][
        "raw_paired_deltas_vs_baseline"
    ] = raw
    state = prepare_attempt10_distillation_states(
        [negative], ranker=_ranker(), expected_roots=None
    )[0]
    assert not np.any(state.safe)


def test_prepare_rejects_future_audit_rows_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, _selected = _development_row()
    row["provenance"]["mode"] = "future_audit"
    monkeypatch.setattr(
        distilled,
        "validate_attempt10_teacher_output",
        lambda *_args, **_kwargs: pytest.fail("validator must not open audit rows"),
    )
    with pytest.raises(ValueError, match="Development200 rows only"):
        prepare_attempt10_distillation_states(
            [row], ranker=_ranker(), expected_roots=None
        )


def test_identity_folds_and_smoke_fit_keep_runtime_disabled() -> None:
    states = _states(per_profile=5)
    first = assign_attempt10_identity_folds(states)
    second = assign_attempt10_identity_folds(states)
    np.testing.assert_array_equal(first, second)
    for profile in ATTEMPT10_PROFILES:
        assert sorted(
            first[index]
            for index, state in enumerate(states)
            if state.root_profile == profile
        ) == list(range(5))

    config = json.loads(
        (ROOT / "configs/hu_joint_policy_m43_attempt10_distillation.json").read_text(
            encoding="utf-8"
        )
    )
    model, report = fit_attempt10_distilled_model(
        _states(), candidate_ranker=_ranker(), config=config, smoke=True
    )
    assert len(model.fold_predictors) == 5
    assert model.safety_enabled is False
    assert model.winner_frozen is False
    assert report["states"] == 10
    assert report["rows"] == 130
    assert report["audit50_fit_rows"] == 0
    assert report["threshold_sweep_performed"] is False

