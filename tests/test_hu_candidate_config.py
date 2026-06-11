import argparse
import json
from pathlib import Path

from ofc_regular.evaluate_hu_candidate_config import namespace_from_config


def test_candidate_config_builds_matchup_namespace():
    config = {
        "name": "candidate",
        "model": {"path": "models/hu.pkl"},
        "runtime": {"hu_turn3_min_margin": 8.0, "hu_turn3_max_self_regret": None},
        "baseline_models": {
            "opening": "models/opening.pt",
            "turn1": "models/turn1.pt",
            "turn2": "models/turn2.pkl",
            "turn3": "models/turn3.pkl",
        },
    }
    args = argparse.Namespace(
        games=10,
        seed=7,
        name_b="baseline",
        prediction_threads=1,
        progress_every=0,
        trace_output=None,
        trace_limit=0,
        output=None,
    )

    namespace = namespace_from_config(config, args)

    assert namespace.name_a == "candidate"
    assert namespace.hu_turn3_a == Path("models/hu.pkl")
    assert namespace.hu_turn3_support_a is None
    assert namespace.hu_turn3_min_margin_a == 8.0
    assert namespace.opening_a == Path("models/opening.pt")
    assert namespace.hu_turn3_b is None


def test_support_experiment_config_builds_matchup_namespace():
    config = {
        "name": "experiment",
        "primary_model": {"path": "models/primary.pkl"},
        "reference_model": {"path": "models/reference.pt"},
        "support_model": {"path": "models/support.pkl"},
        "runtime": {
            "hu_turn3_min_margin": 8.0,
            "hu_turn3_reference_min_margin": 10.0,
            "hu_turn3_min_support_margin": 4.0,
        },
    }
    args = argparse.Namespace(
        games=10,
        seed=7,
        name_b="baseline",
        prediction_threads=1,
        progress_every=0,
        trace_output=None,
        trace_limit=0,
        output=None,
    )

    namespace = namespace_from_config(config, args)

    assert namespace.hu_turn3_a == Path("models/primary.pkl")
    assert namespace.hu_turn3_reference_a == Path("models/reference.pt")
    assert namespace.hu_turn3_support_a == Path("models/support.pkl")
    assert namespace.hu_turn3_min_margin_a == 8.0
    assert namespace.hu_turn3_reference_min_margin_a == 10.0
    assert namespace.hu_turn3_min_support_margin_a == 4.0


def test_stage7_m5_r10_production_config_builds_enabled_namespace():
    config = json.loads(Path("configs/hu_turn3_stage7_m5_r10_production.json").read_text())
    args = argparse.Namespace(
        games=10,
        seed=7,
        name_b="baseline",
        prediction_threads=1,
        progress_every=0,
        trace_output=None,
        trace_limit=0,
        output=None,
    )

    namespace = namespace_from_config(config, args)

    assert namespace.name_a == "hu_turn3_stage7_m5_r10_production"
    assert namespace.hu_turn3_a == Path("models/hu_turn3_stage7_reference_override_cached_rank_wide.pt")
    assert namespace.hu_turn3_reference_a == Path(
        "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt"
    )
    assert namespace.hu_turn3_min_margin_a == 5.0
    assert namespace.hu_turn3_reference_min_margin_a == 10.0
    assert namespace.disable_hu_turn3_stage7_a is False


def test_stage7_canary_presets_do_not_make_margin025_production_default():
    config = json.loads(Path("configs/hu_turn3_stage7_canary_presets.json").read_text())
    production_presets = [
        preset for preset in config["presets"] if preset.get("production_default") is True
    ]

    assert [preset["name"] for preset in production_presets] == ["stage7_m5_r10"]
    assert all(
        preset["runtime"]["hu_turn3_min_margin"] != 0.25
        for preset in production_presets
    )
    assert any(
        excluded["runtime"]["hu_turn3_min_margin"] == 0.25
        for excluded in config["excluded_from_production"]
    )
