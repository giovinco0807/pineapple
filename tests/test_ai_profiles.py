from pathlib import Path

from ofc_regular import ai_profiles
from ofc_regular.ai_profiles import (
    DEFAULT_HU_TURN3_CANDIDATE_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
    ModelBundle,
    ModelPaths,
    build_policy,
    load_model_bundle,
    required_profiles,
)


def test_hu_t3_candidate_profile_loads_current_models_and_hu_candidate(monkeypatch):
    action_calls = []
    hu_calls = []

    def fake_action_loader(path):
        action_calls.append(path)
        return f"action:{path}"

    def fake_hu_loader(path):
        hu_calls.append(path)
        return f"hu:{path}"

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)

    paths = ModelPaths(
        opening=Path("opening.pt"),
        old_opening=Path("old.pkl"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn3_candidate=Path("hu.pkl"),
    )
    bundle = load_model_bundle(paths, {"hu_t3_candidate"})

    assert action_calls == [paths.opening, paths.turn1, paths.turn2, paths.turn3]
    assert hu_calls == [paths.hu_turn3_candidate]
    assert bundle.opening == "action:opening.pt"
    assert bundle.turn3 == "action:turn3.pkl"
    assert bundle.hu_turn3_candidate == "hu:hu.pkl"


def test_hu_t3_candidate_profile_builds_margin_gated_policy_with_seat():
    opening = object()
    turn1 = object()
    turn2 = object()
    turn3 = object()
    hu_turn3 = object()
    bundle = ModelBundle(
        opening=opening,
        turn1=turn1,
        turn2=turn2,
        turn3=turn3,
        hu_turn3_candidate=hu_turn3,
    )

    policy = build_policy(
        "hu_t3_candidate",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
    )

    assert policy.opening_model is opening
    assert policy.turn1_model is turn1
    assert policy.turn2_model is turn2
    assert policy.turn3_model is turn3
    assert policy.hu_turn3_model is hu_turn3
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_CANDIDATE_MIN_MARGIN
    assert policy.seat == "second"
    assert policy.opening_lookahead_samples == 17


def test_current_profile_builds_stage7_m5_r10_selective_override():
    opening = object()
    turn1 = object()
    turn2 = object()
    turn3 = object()
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=opening,
        turn1=turn1,
        turn2=turn2,
        turn3=turn3,
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "current",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn3_stage7_enabled is True
    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN


def test_stage7_off_profile_keeps_stage3_hu_reference_policy():
    reference = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "stage7_off",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn3_stage7_enabled is True
    assert policy.hu_turn3_model is reference
    assert policy.hu_turn3_reference_model is None
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN


def test_stage7_optional_model_load_failure_falls_back_to_none(monkeypatch):
    def fake_action_loader(path):
        return f"action:{path}"

    def fake_hu_loader(path):
        raise OSError(f"missing {path}")

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn3_stage7=Path("missing_stage7.pt"),
        hu_turn3_stage7_reference=Path("missing_stage3.pt"),
    )
    bundle = load_model_bundle(paths, {"current"})

    assert bundle.turn3 == "action:turn3.pkl"
    assert bundle.hu_turn3_stage7 is None
    assert bundle.hu_turn3_stage7_reference is None


def test_required_profiles_keeps_hu_t3_candidate():
    assert required_profiles("hu_t3_candidate", "current") == {"hu_t3_candidate", "current"}
