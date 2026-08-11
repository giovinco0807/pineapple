from pathlib import Path

from ofc_regular import ai_profiles
from ofc_regular.ai_profiles import (
    DEFAULT_HU_TURN1_SAFE_SELECTOR_MODEL,
    DEFAULT_HU_TURN1_STAGE1_MODEL,
    DEFAULT_HU_TURN1_STAGE1_MIN_MARGIN,
    DEFAULT_HU_TURN1_STAGE18_P1_CONFIG,
    DEFAULT_HU_TURN1_STAGE18_P1_MODEL,
    DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL,
    DEFAULT_HU_TURN1_TOPK_CONFIRM_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CSEMAX2P5_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_RANK1_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2P5_CONFIG,
    DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_GATE,
    DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY,
    DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE,
    DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN,
    DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN,
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


def test_stage7_m5_r10_profile_builds_rollback_selective_override():
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
        "stage7_m5_r10",
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


def test_current_profile_builds_stage9d_accept_gated_policy():
    opening = object()
    turn1 = object()
    turn2 = object()
    turn3 = object()
    stage9d = object()
    reference = object()
    support = object()
    gate = object()
    bundle = ModelBundle(
        opening=opening,
        turn1=turn1,
        turn2=turn2,
        turn3=turn3,
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=reference,
        hu_turn3_stage9d=stage9d,
        hu_turn3_stage9d_support=support,
        hu_turn3_stage9d_gate=gate,
    )

    policy = build_policy(
        "current",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn3_model is stage9d
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_support_model is support
    assert policy.hu_turn3_gate_model is gate
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN
    assert policy.hu_turn3_min_support_margin == DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN
    assert policy.hu_turn3_min_gate_probability == DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY


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
    bundle = load_model_bundle(paths, {"stage7_m5_r10"})

    assert bundle.turn3 == "action:turn3.pkl"
    assert bundle.hu_turn3_stage7 is None
    assert bundle.hu_turn3_stage7_reference is None


def test_stage9d_profile_loads_gate_support_and_reference_models(monkeypatch):
    action_calls = []
    hu_calls = []
    gate_calls = []

    def fake_action_loader(path):
        action_calls.append(path)
        return f"action:{path}"

    def fake_hu_loader(path):
        hu_calls.append(path)
        return f"hu:{path}"

    def fake_gate_loader(path):
        gate_calls.append(path)
        return f"gate:{path}"

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_turn3_gate_model", fake_gate_loader)

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn3_stage7=Path("stage7.pt"),
        hu_turn3_stage7_reference=Path("reference.pt"),
        hu_turn3_stage9d=Path("stage9d.pkl"),
        hu_turn3_stage9d_support=Path("support.pkl"),
        hu_turn3_stage9d_gate=Path("gate.pkl"),
    )
    bundle = load_model_bundle(paths, {"stage9d_p07_relaxed_both"})

    assert action_calls == [paths.opening, paths.turn1, paths.turn2, paths.turn3]
    assert hu_calls == [
        paths.hu_turn3_stage7,
        paths.hu_turn3_stage7_reference,
        paths.hu_turn3_stage9d,
        paths.hu_turn3_stage9d_support,
    ]
    assert gate_calls == [paths.hu_turn3_stage9d_gate]
    assert bundle.hu_turn3_stage9d == "hu:stage9d.pkl"
    assert bundle.hu_turn3_stage9d_support == "hu:support.pkl"
    assert bundle.hu_turn3_stage9d_gate == "gate:gate.pkl"


def test_stage9f_profile_loads_topk_candidate_and_stage7_continuation_models(monkeypatch):
    action_calls = []
    hu_calls = []
    turn2_hu_calls = []

    def fake_action_loader(path):
        action_calls.append(path)
        return f"action:{path}"

    def fake_hu_loader(path):
        hu_calls.append(path)
        return f"hu:{path}"

    def fake_turn2_hu_loader(path):
        turn2_hu_calls.append(path)
        return f"hu_t2:{path}"

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_turn2_stage8_model", fake_turn2_hu_loader)

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn2_stage8b=Path("stage8b.pt"),
        hu_turn3_stage7=Path("stage7.pt"),
        hu_turn3_stage7_reference=Path("reference.pt"),
    )
    bundle = load_model_bundle(paths, {"stage9f_cse1p5_csemax2p5_firstseat"})

    assert action_calls == [paths.opening, paths.turn1, paths.turn2, paths.turn3]
    assert hu_calls == [paths.hu_turn3_stage7, paths.hu_turn3_stage7_reference]
    assert turn2_hu_calls == [paths.hu_turn2_stage8b]
    assert bundle.hu_turn2_stage8b == "hu_t2:stage8b.pt"
    assert bundle.hu_turn3_stage7 == "hu:stage7.pt"
    assert bundle.hu_turn3_stage7_reference == "hu:reference.pt"


def test_stage9f_p2_hu_t1_profile_loads_hu_turn1_candidate(monkeypatch):
    action_calls = []
    hu_calls = []
    turn2_hu_calls = []

    def fake_action_loader(path):
        action_calls.append(path)
        return f"action:{path}"

    def fake_hu_loader(path):
        hu_calls.append(path)
        return f"hu:{path}"

    def fake_turn2_hu_loader(path):
        turn2_hu_calls.append(path)
        return f"hu_t2:{path}"

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_turn2_stage8_model", fake_turn2_hu_loader)

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn1_stage1=Path("hu_t1.pkl"),
        hu_turn2_stage8b=Path("stage8b.pt"),
        hu_turn3_stage7=Path("stage7.pt"),
        hu_turn3_stage7_reference=Path("reference.pt"),
    )
    bundle = load_model_bundle(paths, {"stage9f_p2_hu_t1_stage1"})

    assert action_calls == [paths.opening, paths.turn1, paths.turn2, paths.turn3]
    assert hu_calls == [
        paths.hu_turn1_stage1,
        paths.hu_turn3_stage7,
        paths.hu_turn3_stage7_reference,
    ]
    assert turn2_hu_calls == [paths.hu_turn2_stage8b]
    assert bundle.hu_turn1_stage1 == "hu:hu_t1.pkl"
    assert bundle.hu_turn2_stage8b == "hu_t2:stage8b.pt"


def test_stage9f_p2_hu_t1_topk_confirm_loads_safe_selector(monkeypatch):
    action_calls = []
    hu_calls = []
    turn2_hu_calls = []
    safe_selector_calls = []

    def fake_action_loader(path):
        action_calls.append(path)
        return f"action:{path}"

    def fake_hu_loader(path):
        hu_calls.append(path)
        return f"hu:{path}"

    def fake_turn2_hu_loader(path):
        turn2_hu_calls.append(path)
        return f"hu_t2:{path}"

    def fake_safe_selector_loader(path):
        safe_selector_calls.append(path)
        return f"safe:{path}"

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_turn2_stage8_model", fake_turn2_hu_loader)
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn1_safe_selector_model",
        fake_safe_selector_loader,
    )

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn1_stage1=Path("hu_t1.pkl"),
        hu_turn1_safe_selector=Path("safe_selector.pkl"),
        hu_turn2_stage8b=Path("stage8b.pt"),
        hu_turn3_stage7=Path("stage7.pt"),
        hu_turn3_stage7_reference=Path("reference.pt"),
    )
    bundle = load_model_bundle(paths, {"stage9f_p2_hu_t1_topk_confirm"})

    assert action_calls == [paths.opening, paths.turn1, paths.turn2, paths.turn3]
    assert hu_calls == [
        paths.hu_turn1_stage1,
        paths.hu_turn3_stage7,
        paths.hu_turn3_stage7_reference,
    ]
    assert turn2_hu_calls == [paths.hu_turn2_stage8b]
    assert safe_selector_calls == [paths.hu_turn1_safe_selector]
    assert bundle.hu_turn1_safe_selector == "safe:safe_selector.pkl"


def test_stage18_p1_loads_dedicated_models_with_safe_fallback(monkeypatch):
    action_calls = []
    hu_calls = []
    selector_calls = []

    monkeypatch.setattr(
        ai_profiles,
        "load_action_value_model",
        lambda path: action_calls.append(path) or f"action:{path}",
    )
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_action_value_model",
        lambda path: hu_calls.append(path) or f"hu:{path}",
    )
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn1_safe_selector_model",
        lambda path: selector_calls.append(path) or f"safe:{path}",
    )
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn2_stage8_model",
        lambda path: f"hu_t2:{path}",
    )

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn1_stage18_p1=Path("stage18.pkl"),
        hu_turn1_stage18_p1_safe_selector=Path("stage18_safe.pkl"),
        hu_turn2_stage8b=Path("stage9f.pt"),
        hu_turn3_stage7=Path("stage7.pt"),
        hu_turn3_stage7_reference=Path("reference.pt"),
    )
    bundle = load_model_bundle(paths, {"stage18_p1"})

    assert action_calls == [paths.opening, paths.turn1, paths.turn2, paths.turn3]
    assert hu_calls == [
        paths.hu_turn1_stage18_p1,
        paths.hu_turn3_stage7,
        paths.hu_turn3_stage7_reference,
    ]
    assert selector_calls == [paths.hu_turn1_stage18_p1_safe_selector]
    assert bundle.hu_turn1_stage18_p1 == "hu:stage18.pkl"
    assert bundle.hu_turn1_stage18_p1_safe_selector == "safe:stage18_safe.pkl"


def test_stage18_p1_model_load_failure_falls_back_to_stage9f_p2(monkeypatch):
    monkeypatch.setattr(ai_profiles, "load_action_value_model", lambda path: object())
    monkeypatch.setattr(ai_profiles, "load_hu_turn2_stage8_model", lambda path: object())
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", lambda path: (_ for _ in ()).throw(OSError(path)))
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn1_safe_selector_model",
        lambda path: (_ for _ in ()).throw(OSError(path)),
    )

    bundle = load_model_bundle(ModelPaths(), {"stage18_p1"})
    policy = build_policy(
        "stage18_p1",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=1,
    )

    assert bundle.hu_turn1_stage18_p1 is None
    assert bundle.hu_turn1_stage18_p1_safe_selector is None
    assert policy.hu_turn1_candidate_models == ()
    assert policy.decision_context["fallback_policy"] == "stage9f_p2"


def test_stage9f_p2_hu_t1_topk_confirm_loads_candidate_model_pool(monkeypatch):
    hu_calls = []

    def fake_action_loader(path):
        return f"action:{path}"

    def fake_hu_loader(path):
        hu_calls.append(path)
        return f"hu:{path}"

    def fake_turn2_hu_loader(path):
        return f"hu_t2:{path}"

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_turn2_stage8_model", fake_turn2_hu_loader)
    monkeypatch.setattr(
        ai_profiles,
        "load_hu_turn1_safe_selector_model",
        lambda path: f"safe:{path}",
    )

    paths = ModelPaths(
        opening=Path("opening.pt"),
        turn1=Path("turn1.pt"),
        turn2=Path("turn2.pkl"),
        turn3=Path("turn3.pkl"),
        hu_turn1_stage1=Path("hu_t1_single.pkl"),
        hu_turn1_stage1_models=(Path("hu_t1_a.pkl"), Path("hu_t1_b.pkl")),
        hu_turn2_stage8b=Path("stage8b.pt"),
        hu_turn3_stage7=Path("stage7.pt"),
        hu_turn3_stage7_reference=Path("reference.pt"),
    )
    bundle = load_model_bundle(paths, {"stage9f_p2_hu_t1_topk_confirm"})

    assert hu_calls[:2] == list(paths.hu_turn1_stage1_models)
    assert paths.hu_turn1_stage1 not in hu_calls
    assert bundle.hu_turn1_stage1 == "hu:hu_t1_a.pkl"
    assert bundle.hu_turn1_stage1_models == ("hu:hu_t1_a.pkl", "hu:hu_t1_b.pkl")


def test_stage9f_profile_missing_candidate_model_is_not_silent(monkeypatch):
    def fake_action_loader(path):
        return f"action:{path}"

    def fake_hu_loader(path):
        return f"hu:{path}"

    def fake_turn2_hu_loader(path):
        raise OSError(f"missing {path}")

    monkeypatch.setattr(ai_profiles, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(ai_profiles, "load_hu_turn2_stage8_model", fake_turn2_hu_loader)

    paths = ModelPaths(hu_turn2_stage8b=Path("missing_stage8b.pt"))
    try:
        load_model_bundle(paths, {"stage9f_cse1p5_firstseat"})
    except OSError as exc:
        assert "missing missing_stage8b.pt" in str(exc)
    else:
        raise AssertionError("Stage9f profile should fail closed when the candidate model is missing")


def test_stage9f_profile_builds_validation_only_topk_policy():
    opening = object()
    turn1 = object()
    turn2 = object()
    turn3 = object()
    stage8b = object()
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=opening,
        turn1=turn1,
        turn2=turn2,
        turn3=turn3,
        hu_turn2_stage8b=stage8b,
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "stage9f_cse1p5_firstseat",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.opening_model is opening
    assert policy.turn1_model is turn1
    assert policy.turn2_model is turn2
    assert policy.turn3_model is turn3
    assert policy.hu_turn2_stage8b_model is stage8b
    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
    assert policy.hu_turn3_stage7_enabled is True
    assert policy.t3_continuation == "stage7_m5_r10"
    assert policy.topk_rerank_config.allowed_seats == ("first",)
    assert policy.topk_rerank_config.confirm_se_multiplier == 1.5
    assert policy.topk_context["runtime_status"] == "validation_only"
    assert DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CONFIG == (
        "k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta"
    )


def test_stage9f_csemax2p5_profile_builds_experiment_only_tail_guard_policy():
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )

    policy = build_policy(
        "stage9f_cse1p5_csemax2p5_firstseat",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.topk_rerank_config.allowed_seats == ("first",)
    assert policy.topk_rerank_config.confirm_se_multiplier == 1.5
    assert policy.topk_rerank_config.max_confirm_se == 2.5
    assert policy.topk_context["runtime_profile"] == "stage9f_cse1p5_csemax2p5_firstseat"
    assert policy.topk_context["runtime_status"] == "validation_only"
    assert DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CSEMAX2P5_CONFIG == (
        "k3/mc16/d0/se0/confirm32/cse1.5/csemax2.5/pd0/seat=first/bygate_delta"
    )


def test_stage9f_cse2_csemax2p5_profile_builds_experiment_only_tail_guard_policy():
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )

    policy = build_policy(
        "stage9f_cse2_csemax2p5_firstseat",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.topk_rerank_config.allowed_seats == ("first",)
    assert policy.topk_rerank_config.confirm_se_multiplier == 2.0
    assert policy.topk_rerank_config.max_confirm_se == 2.5
    assert policy.topk_context["runtime_profile"] == "stage9f_cse2_csemax2p5_firstseat"
    assert policy.topk_context["runtime_status"] == "validation_only"
    assert DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2P5_CONFIG == (
        "k3/mc16/d0/se0/confirm32/cse2/csemax2.5/pd0/seat=first/bygate_delta"
    )


def test_stage9f_cse2_csemax2_profile_builds_experiment_only_tail_guard_policy():
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )

    policy = build_policy(
        "stage9f_cse2_csemax2_firstseat",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.topk_rerank_config.allowed_seats == ("first",)
    assert policy.topk_rerank_config.confirm_se_multiplier == 2.0
    assert policy.topk_rerank_config.max_confirm_se == 2.0
    assert policy.topk_context["runtime_profile"] == "stage9f_cse2_csemax2_firstseat"
    assert policy.topk_context["runtime_status"] == "validation_only"
    assert DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_CONFIG == (
        "k3/mc16/d0/se0/confirm32/cse2/csemax2/pd0/seat=first/bygate_delta"
    )


def test_stage9f_cse2_csemax2_bothseat_profile_builds_validation_only_policy():
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )

    policy = build_policy(
        "stage9f_cse2_csemax2_bothseat",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
    )

    assert policy.topk_rerank_config.allowed_seats == ("first", "second")
    assert policy.topk_rerank_config.confirm_se_multiplier == 2.0
    assert policy.topk_rerank_config.max_confirm_se == 2.0
    assert policy.topk_context["runtime_profile"] == "stage9f_cse2_csemax2_bothseat"
    assert policy.topk_context["runtime_status"] == "validation_only"
    assert policy.t3_continuation == "stage7_m5_r10"
    assert DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG == (
        "k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta"
    )


def test_stage9f_p2_profile_builds_fixed_bothseat_policy():
    stage8b = object()
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=stage8b,
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "stage9f_p2",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn2_stage8b_model is stage8b
    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
    assert policy.hu_turn3_stage7_enabled is True
    assert policy.t3_continuation == "stage7_m5_r10"
    assert policy.topk_rerank_config.allowed_seats == ("first", "second")
    assert policy.topk_rerank_config.confirm_se_multiplier == 2.0
    assert policy.topk_rerank_config.max_confirm_se == 2.0
    assert policy.topk_context["runtime_profile"] == "stage9f_p2"
    assert policy.topk_context["runtime_status"] == "p2_fixed"


def test_stage9f_p2_hu_t1_profile_builds_p2_with_hu_turn1_candidate():
    stage8b = object()
    hu_turn1 = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn1_stage1=hu_turn1,
        hu_turn2_stage8b=stage8b,
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )

    policy = build_policy(
        "stage9f_p2_hu_t1_stage1",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn1_model is hu_turn1
    assert policy.hu_turn1_min_margin == DEFAULT_HU_TURN1_STAGE1_MIN_MARGIN
    assert policy.hu_turn2_stage8b_model is stage8b
    assert policy.topk_rerank_config.allowed_seats == ("first", "second")
    assert policy.t3_continuation == "stage7_m5_r10"
    assert policy.topk_context["runtime_profile"] == "stage9f_p2_hu_t1_stage1"
    assert policy.topk_context["runtime_status"] == "t1_stage1_validation"

    override_policy = build_policy(
        "stage9f_p2_hu_t1_stage1",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
        hu_turn1_min_margin=2.5,
    )
    assert override_policy.hu_turn1_min_margin == 2.5


def test_stage9f_p2_hu_t1_topk_confirm_profile_builds_validation_policy():
    stage8b = object()
    hu_turn1 = object()
    safe_selector = object()
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn1_stage1=hu_turn1,
        hu_turn1_safe_selector=safe_selector,
        hu_turn2_stage8b=stage8b,
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "stage9f_p2_hu_t1_topk_confirm",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
        hu_turn1_topk_config="k2/mc4/d0/confirm8/cse1.5/pd0/safe0.5/seat=first+second",
    )

    assert policy.hu_turn1_candidate_model is hu_turn1
    assert policy.hu_turn1_safe_selector_model is safe_selector
    assert policy.hu_turn1_topk_confirm_config.top_k == 2
    assert policy.hu_turn1_topk_confirm_config.mc_samples == 4
    assert policy.hu_turn1_topk_confirm_config.confirm_mc_samples == 8
    assert policy.hu_turn1_topk_confirm_config.confirm_se_multiplier == 1.5
    assert policy.hu_turn1_topk_confirm_config.safe_selector_threshold == 0.5
    assert policy.hu_turn1_topk_confirm_config.allowed_seats == ("first", "second")
    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.decision_context["runtime_profile"] == "stage9f_p2_hu_t1_topk_confirm"
    assert policy.decision_context["runtime_status"] == "t1_topk_confirm_validation"
    assert DEFAULT_HU_TURN1_TOPK_CONFIRM_CONFIG.startswith("k3/")


def test_stage18_p1_profile_builds_locked_first_seat_selective_override():
    candidate = object()
    selector = object()
    stage8b = object()
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn1_stage18_p1=candidate,
        hu_turn1_stage18_p1_safe_selector=selector,
        hu_turn2_stage8b=stage8b,
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "stage18_p1",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
        hu_turn1_topk_config="k1/mc1/d0/seat=first+second",
    )
    config = policy.hu_turn1_topk_confirm_config

    assert policy.hu_turn1_candidate_model is candidate
    assert policy.hu_turn1_safe_selector_model is selector
    assert config.top_k == 5
    assert config.mc_samples == 8
    assert config.min_delta == 3.0
    assert config.confirm_mc_samples == 32
    assert config.confirm_se_multiplier == 1.5
    assert config.min_predicted_delta == 1.5
    assert config.safe_selector_threshold == 0.7
    assert config.allowed_seats == ("first",)
    assert config.config_id == "k5_mc8_d3_confirm32_cse1.5_pd1.5_safe0.7_seat_first"
    assert policy.hu_turn2_stage8b_model is stage8b
    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.t3_continuation == "stage7_m5_r10"
    assert policy.decision_context == {
        "runtime_profile": "stage18_p1",
        "runtime_status": "p1_fixed",
        "selective_override_only": True,
        "full_replacement_enabled": False,
        "fallback_policy": "stage9f_p2",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
    }
    assert policy.topk_context["runtime_profile"] == "stage18_p1_t2"
    assert DEFAULT_HU_TURN1_STAGE18_P1_CONFIG == (
        "k5/mc8/d3/confirm32/cse1.5/pd1.5/seat=first/safe0.7"
    )


def test_stage9f_fast_t2_t1_teacher_profile_builds_non_production_selective_policy():
    stage8b = object()
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=stage8b,
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
    )

    policy = build_policy(
        "stage9f_fast_t2_t1_teacher",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn2_stage8_model is stage8b
    assert policy.hu_turn2_stage8_config.min_margin == DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_MIN_MARGIN
    assert policy.hu_turn2_stage8_config.reference_min_margin == 0.0
    assert policy.hu_turn2_stage8_config.gate_threshold == DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_GATE
    assert policy.hu_turn2_stage8_config.allowed_seats == ("first", "second")
    assert policy.hu_turn2_context["runtime_profile"] == "stage9f_fast_t2_t1_teacher"
    assert policy.hu_turn2_context["runtime_status"] == "t1_teacher_fast_continuation"
    assert policy.hu_turn2_context["t3_continuation"] == "stage7_m5_r10"
    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
    assert policy.hu_turn3_stage7_enabled is True


def test_stage9f_cse2_csemax2_rank1_profile_builds_experiment_only_tail_guard_policy():
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn2_stage8b=object(),
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=object(),
    )

    policy = build_policy(
        "stage9f_cse2_csemax2_rank1_firstseat",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.topk_rerank_config.allowed_seats == ("first",)
    assert policy.topk_rerank_config.confirm_se_multiplier == 2.0
    assert policy.topk_rerank_config.max_confirm_se == 2.0
    assert policy.topk_rerank_config.candidate_ev_rank_max == 1
    assert policy.topk_context["runtime_profile"] == "stage9f_cse2_csemax2_rank1_firstseat"
    assert policy.topk_context["runtime_status"] == "validation_only"
    assert DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_RANK1_CONFIG == (
        "k3/mc16/d0/se0/confirm32/cse2/csemax2/pd0/rank1/seat=first/bygate_delta"
    )


def test_stage9d_profile_builds_accept_gate_policy():
    opening = object()
    turn1 = object()
    turn2 = object()
    turn3 = object()
    stage9d = object()
    reference = object()
    support = object()
    gate = object()
    bundle = ModelBundle(
        opening=opening,
        turn1=turn1,
        turn2=turn2,
        turn3=turn3,
        hu_turn3_stage7=object(),
        hu_turn3_stage7_reference=reference,
        hu_turn3_stage9d=stage9d,
        hu_turn3_stage9d_support=support,
        hu_turn3_stage9d_gate=gate,
    )

    policy = build_policy(
        "stage9d_p07_relaxed_both",
        bundle,
        seed=123,
        seat="second",
        opening_lookahead_samples=17,
    )

    assert policy.opening_model is opening
    assert policy.turn1_model is turn1
    assert policy.turn2_model is turn2
    assert policy.turn3_model is turn3
    assert policy.hu_turn3_model is stage9d
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_support_model is support
    assert policy.hu_turn3_gate_model is gate
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN
    assert policy.hu_turn3_min_support_margin == DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN
    assert policy.hu_turn3_min_model_score == DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE
    assert policy.hu_turn3_min_gate_probability == DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY
    assert policy.hu_turn3_allowed_seats is None
    assert policy.seat == "second"


def test_stage9d_profile_missing_gate_falls_back_to_stage7_m5_r10():
    stage7 = object()
    reference = object()
    bundle = ModelBundle(
        opening=object(),
        turn1=object(),
        turn2=object(),
        turn3=object(),
        hu_turn3_stage7=stage7,
        hu_turn3_stage7_reference=reference,
        hu_turn3_stage9d=object(),
        hu_turn3_stage9d_support=object(),
        hu_turn3_stage9d_gate=None,
    )

    policy = build_policy(
        "stage9d_p07_relaxed_both",
        bundle,
        seed=123,
        seat="first",
        opening_lookahead_samples=17,
    )

    assert policy.hu_turn3_model is stage7
    assert policy.hu_turn3_reference_model is reference
    assert policy.hu_turn3_min_margin == DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN
    assert policy.hu_turn3_reference_min_margin == DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
    assert policy.hu_turn3_gate_model is None


def test_required_profiles_keeps_hu_t3_candidate_stage9d_and_stage9f():
    assert required_profiles("hu_t3_candidate", "current") == {"hu_t3_candidate", "current"}
    assert required_profiles("stage9d_p07_relaxed_both", "current") == {
        "stage9d_p07_relaxed_both",
        "current",
    }
    assert required_profiles("stage9f_cse1p5_firstseat", "current") == {
        "stage9f_cse1p5_firstseat",
        "current",
    }
    assert required_profiles("stage9f_cse1p5_csemax2p5_firstseat", "current") == {
        "stage9f_cse1p5_csemax2p5_firstseat",
        "current",
    }
    assert required_profiles("stage9f_cse2_csemax2p5_firstseat", "current") == {
        "stage9f_cse2_csemax2p5_firstseat",
        "current",
    }
    assert required_profiles("stage9f_cse2_csemax2_firstseat", "current") == {
        "stage9f_cse2_csemax2_firstseat",
        "current",
    }
    assert required_profiles("stage9f_cse2_csemax2_bothseat", "current") == {
        "stage9f_cse2_csemax2_bothseat",
        "current",
    }
    assert required_profiles("stage9f_p2", "current") == {"stage9f_p2", "current"}
    assert required_profiles("stage9f_p2_hu_t1_stage1", "stage9f_p2") == {
        "stage9f_p2_hu_t1_stage1",
        "stage9f_p2",
    }
    assert required_profiles("stage9f_p2_hu_t1_topk_confirm", "stage9f_p2") == {
        "stage9f_p2_hu_t1_topk_confirm",
        "stage9f_p2",
    }
    assert required_profiles("stage18_p1", "stage9f_p2") == {
        "stage18_p1",
        "stage9f_p2",
    }
    assert required_profiles("stage9f_fast_t2_t1_teacher", "current") == {
        "stage9f_fast_t2_t1_teacher",
        "current",
    }


def test_stage9f_profile_is_available_from_matchup_cli():
    from ofc_regular.evaluate_matchups import PROFILE_CHOICES, parse_args

    assert "stage9f_cse1p5_firstseat" in PROFILE_CHOICES
    assert "stage9f_cse1p5_csemax2p5_firstseat" in PROFILE_CHOICES
    assert "stage9f_cse2_csemax2p5_firstseat" in PROFILE_CHOICES
    assert "stage9f_cse2_csemax2_firstseat" in PROFILE_CHOICES
    assert "stage9f_cse2_csemax2_bothseat" in PROFILE_CHOICES
    assert "stage9f_p2" in PROFILE_CHOICES
    assert "stage9f_p2_hu_t1_stage1" in PROFILE_CHOICES
    assert "stage9f_p2_hu_t1_topk_confirm" in PROFILE_CHOICES
    assert "stage18_p1" in PROFILE_CHOICES
    assert "stage9f_fast_t2_t1_teacher" in PROFILE_CHOICES
    parsed = parse_args(
        [
            "--profile-a",
            "stage9f_cse1p5_firstseat",
            "--profile-b",
            "stage7_m5_r10",
            "--hu-turn2-stage8b-model",
            "models/stage9f.pt",
            "--hu-turn1-stage1-model",
            "models/hu_t1.pkl",
            "--hu-turn1-stage1-models",
            "models/hu_t1_a.pkl",
            "models/hu_t1_b.pkl",
            "--hu-turn1-safe-selector-model",
            "models/safe_selector.pkl",
            "--hu-turn1-stage18-p1-model",
            "models/stage18.pkl",
            "--hu-turn1-stage18-p1-safe-selector-model",
            "models/stage18_safe.pkl",
            "--hu-turn1-min-margin",
            "2.5",
            "--hu-turn1-topk-config",
            "k2/mc4/d0/confirm8/cse1/pd0/seat=first",
            "--seed-stride",
            "1009",
            "--topk-decision-output",
            "outputs/topk.jsonl",
            "--hu-turn1-decision-output",
            "outputs/hu_t1.jsonl",
        ]
    )
    assert parsed.profile_a == "stage9f_cse1p5_firstseat"
    assert parsed.hu_turn2_stage8b_model == Path("models/stage9f.pt")
    assert parsed.hu_turn1_stage1_model == Path("models/hu_t1.pkl")
    assert parsed.hu_turn1_stage1_models == [
        Path("models/hu_t1_a.pkl"),
        Path("models/hu_t1_b.pkl"),
    ]
    assert parsed.hu_turn1_safe_selector_model == Path("models/safe_selector.pkl")
    assert parsed.hu_turn1_stage18_p1_model == Path("models/stage18.pkl")
    assert parsed.hu_turn1_stage18_p1_safe_selector_model == Path(
        "models/stage18_safe.pkl"
    )
    assert parsed.hu_turn1_min_margin == 2.5
    assert parsed.hu_turn1_topk_config == "k2/mc4/d0/confirm8/cse1/pd0/seat=first"
    assert parsed.seed_stride == 1009
    assert parsed.topk_decision_output == Path("outputs/topk.jsonl")
    assert parsed.hu_turn1_decision_output == Path("outputs/hu_t1.jsonl")
    assert DEFAULT_HU_TURN1_STAGE1_MODEL == Path(
        "models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl"
    )
    assert DEFAULT_HU_TURN1_SAFE_SELECTOR_MODEL == Path(
        "models/hu_turn1_safe_override_selector_teacher_regret_combined296_accept0p25_gray1_delta_plus_meta.pkl"
    )
    assert DEFAULT_HU_TURN1_STAGE18_P1_MODEL == Path(
        "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl"
    )
    assert DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL == Path(
        "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl"
    )
