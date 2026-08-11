import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs" / "hu_joint_policy_m1_quarantine.json"


def test_m1_quarantine_keeps_baselines_explicit_and_blocks_promotion():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert payload["schema"] == "hu_joint_policy_m1_quarantine_v1"
    assert payload["current_profile_changed"] is False
    assert payload["large_scale_generation_allowed"] is False
    assert payload["policy_promotion_allowed"] is False
    assert payload["fixed_legacy_baseline_chain"] == [
        "stage19_p0",
        "stage18_p1",
        "stage9f_p2",
        "stage7_m5_r10",
    ]
    assert "current" not in payload["fixed_legacy_baseline_chain"]
    assert payload["legacy_evidence_status"].startswith("diagnostic_only")

    registered = [
        *payload["legacy_artifacts"],
        *payload["quarantined_generation_or_training_modules"],
    ]
    assert registered
    for entry in registered:
        assert entry["reason"]
        assert (REPO_ROOT / entry["path"]).is_file(), entry["path"]


def test_m1_safe_boundaries_are_real_source_files_and_exclude_quarantine():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    safe = set(payload["safe_m1_boundaries"])
    quarantined = {
        entry["path"]
        for entry in payload["quarantined_generation_or_training_modules"]
    }

    assert safe.isdisjoint(quarantined)
    assert "src/ofc_regular/hu_infoset.py" in safe
    assert "src/ofc_regular/hu_belief.py" in safe
    for path in safe:
        assert (REPO_ROOT / path).is_file(), path
