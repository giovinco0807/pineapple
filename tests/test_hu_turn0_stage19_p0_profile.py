from pathlib import Path

import pytest

from ofc_regular import ai_profiles
from ofc_regular.ai_profiles import (
    DEFAULT_HU_TURN0_STAGE19_P0_ALLOWED_SEATS,
    DEFAULT_HU_TURN0_STAGE19_P0_MIN_MARGIN_BY_SEAT,
    DEFAULT_HU_TURN0_STAGE19_P0_MODEL,
    DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL,
    DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_THRESHOLD_BY_SEAT,
    DEFAULT_HU_TURN0_STAGE19_P0_TOPK,
    ModelBundle,
    ModelPaths,
    build_policy,
    load_model_bundle,
    required_profiles,
)
from ofc_regular.evaluate_matchups import PROFILE_CHOICES, parse_args


def _bundle(*, candidate=object(), selector=object()) -> ModelBundle:
    return ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn0_stage19_p0=candidate,
        hu_turn0_stage19_p0_safe_selector=selector,
        hu_turn1_stage18_p1=object(),
        hu_turn1_stage18_p1_safe_selector=object(),
        hu_turn2_stage8b=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )


def test_stage19_p0_loads_dedicated_models_and_fixed_chain(monkeypatch) -> None:
    turn0_calls: list[Path] = []
    turn0_selector_calls: list[Path] = []
    monkeypatch.setattr(ai_profiles, "load_action_value_model", lambda path: f"base:{path}")
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", lambda path: f"hu:{path}")
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn1_safe_selector_model",
        lambda path: f"t1-safe:{path}",
    )
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn2_stage8_model",
        lambda path: f"t2:{path}",
    )
    monkeypatch.setattr(
        ai_profiles,
        "load_turn0_candidate_model",
        lambda path: turn0_calls.append(path) or f"t0:{path}",
    )
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn0_safe_selector_model",
        lambda path: turn0_selector_calls.append(path) or f"t0-safe:{path}",
    )
    paths = ModelPaths(
        hu_turn0_stage19_p0=Path("t0.pkl"),
        hu_turn0_stage19_p0_safe_selector=Path("t0_safe.pkl"),
    )

    bundle = load_model_bundle(paths, {"stage19_p0"})

    assert turn0_calls == [paths.hu_turn0_stage19_p0]
    assert turn0_selector_calls == [paths.hu_turn0_stage19_p0_safe_selector]
    assert bundle.hu_turn0_stage19_p0 == "t0:t0.pkl"
    assert bundle.hu_turn0_stage19_p0_safe_selector == "t0-safe:t0_safe.pkl"
    assert bundle.hu_turn1_stage18_p1 is not None
    assert bundle.hu_turn2_stage8b is not None
    assert bundle.hu_turn3_stage7 is not None


def test_stage19_p0_profile_locks_first_seat_selective_override() -> None:
    bundle = _bundle()

    policy = build_policy(
        "stage19_p0",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=1,
    )

    assert policy.hu_turn0_model is bundle.hu_turn0_stage19_p0
    assert policy.hu_turn0_safe_selector_model is bundle.hu_turn0_stage19_p0_safe_selector
    assert policy.hu_turn0_safe_selector_enabled is True
    assert policy.hu_turn0_candidate_topk == DEFAULT_HU_TURN0_STAGE19_P0_TOPK
    assert policy.hu_turn0_min_margin_by_seat == DEFAULT_HU_TURN0_STAGE19_P0_MIN_MARGIN_BY_SEAT
    assert (
        policy.hu_turn0_safe_selector_threshold_by_seat
        == DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_THRESHOLD_BY_SEAT
    )
    assert policy.hu_turn0_allowed_seats == DEFAULT_HU_TURN0_STAGE19_P0_ALLOWED_SEATS
    assert policy.hu_turn1_topk_confirm_config.allowed_seats == ("first",)
    assert policy.t3_continuation == "stage7_m5_r10"
    assert policy.decision_context == {
        "runtime_profile": "stage19_p0",
        "runtime_status": "p0_fixed",
        "selective_override_only": True,
        "full_replacement_enabled": False,
        "fallback_policy": "stage18_p1",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
        "t1_continuation": "stage18_p1",
    }


def test_stage19_p0_second_seat_is_disabled() -> None:
    policy = build_policy(
        "stage19_p0",
        _bundle(),
        seed=123,
        seat="second",
        opening_lookahead_samples=1,
    )

    assert policy.hu_turn0_allowed_seats == ("first",)
    assert policy.hu_turn0_min_margin_by_seat["second"] == 999.0
    assert policy.hu_turn0_safe_selector_threshold_by_seat["second"] == 1.0


@pytest.mark.parametrize("missing", ["candidate", "selector"])
def test_stage19_p0_missing_model_falls_back_to_stage18_p1(missing: str) -> None:
    candidate = None if missing == "candidate" else object()
    selector = None if missing == "selector" else object()
    bundle = _bundle(candidate=candidate, selector=selector)

    fallback = build_policy(
        "stage18_p1",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=1,
    )
    policy = build_policy(
        "stage19_p0",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=1,
    )

    assert policy.hu_turn0_model is None
    assert policy.hu_turn0_safe_selector_enabled is False
    assert policy.hu_turn1_topk_confirm_config == fallback.hu_turn1_topk_confirm_config
    assert policy.decision_context == fallback.decision_context
    assert policy.decision_context["runtime_profile"] == "stage18_p1"


def test_stage19_p0_is_explicit_and_cli_paths_are_overridable() -> None:
    assert "stage19_p0" in PROFILE_CHOICES
    assert required_profiles("stage19_p0", "stage18_p1") == {
        "stage19_p0",
        "stage18_p1",
    }
    parsed = parse_args(
        [
            "--profile-a",
            "stage19_p0",
            "--profile-b",
            "stage18_p1",
            "--hu-turn0-stage19-p0-model",
            "models/t0.pkl",
            "--hu-turn0-stage19-p0-safe-selector-model",
            "models/t0_safe.pkl",
            "--hu-turn0-decision-output",
            "outputs/t0.jsonl",
        ]
    )
    assert parsed.hu_turn0_stage19_p0_model == Path("models/t0.pkl")
    assert parsed.hu_turn0_stage19_p0_safe_selector_model == Path("models/t0_safe.pkl")
    assert parsed.hu_turn0_decision_output == Path("outputs/t0.jsonl")
    assert DEFAULT_HU_TURN0_STAGE19_P0_MODEL.name.endswith("aug3.pkl")
    assert DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL.name.endswith(
        "delta_plus_meta.pkl"
    )
