import hashlib
import json
from pathlib import Path


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def test_stage19_p0_acceptance_config_is_locked_and_selective() -> None:
    config = _read("configs/hu_turn0_stage19_p0_selective_override.json")

    assert config["ai_profile"] == "stage19_p0"
    assert config["p0_fixed"] is True
    assert config["current_profile_changed"] is False
    assert config["full_replacement_enabled"] is False
    assert config["selective_override_only"] is True
    assert config["allowed_seats"] == ["first"]
    assert config["runtime"] == {
        "candidate_topk": 60,
        "min_margin_by_seat": {"first": 0.5, "second": 999.0},
        "safe_selector_threshold_by_seat": {"first": 0.6, "second": 1.0},
        "fallback_policy": "stage18_p1",
    }
    assert config["continuations"]["t1"]["profile"] == "stage18_p1"
    assert config["continuations"]["t2"]["profile"] == "stage9f_p2"
    assert config["continuations"]["t3"]["profile"] == "stage7_m5_r10"
    assert config["continuations"]["fl_ev"] == 10.227020614683454
    assert config["acceptance_evidence"]["acceptance_passed"] is True
    assert config["independent_tail_audit"]["acceptance_passed"] is True


def test_stage19_p0_model_hashes_match_locked_artifacts() -> None:
    config = _read("configs/hu_turn0_stage19_p0_selective_override.json")
    candidate = config["models"]["candidate"]
    selector = config["models"]["safe_selector"]

    assert _sha256(candidate) == config["model_integrity"]["candidate_sha256"]
    assert _sha256(selector) == config["model_integrity"]["safe_selector_sha256"]


def test_stage19_p0_presets_keep_explicit_off_rollback() -> None:
    document = _read("configs/hu_turn0_stage19_p0_presets.json")
    presets = {row["name"]: row for row in document["presets"]}

    assert document["current_profile_changed"] is False
    assert presets["stage19_p0_off"] == {
        "name": "stage19_p0_off",
        "ai_profile": "stage18_p1",
        "enabled": False,
        "production_default": False,
        "fallback_policy": "stage18_p1",
    }
    assert presets["stage19_p0"]["ai_profile"] == "stage19_p0"
    assert presets["stage19_p0"]["allowed_seats"] == ["first"]
    assert presets["stage19_p0"]["full_replacement_enabled"] is False
    assert presets["stage19_p0"]["production_default"] is False
