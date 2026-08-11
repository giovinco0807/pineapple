from __future__ import annotations

from pathlib import Path

from ofc_regular.freeze_hu_joint_policy_r0 import (
    audit_snapshot,
    load_snapshot_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "configs/hu_joint_policy_r0_snapshot_v2.json"
MISSING_R0B_ROLE = "m30_windows_exact_runtime"
ROADMAP = "docs/hu_joint_policy_full_hand_rl_milestones_20260718.md"
OLD_V1_ROOT_PREFIX = (
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-20260717-001/roots-open/"
)
REARM1_ROOT_PREFIX = (
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-rearm1-20260717-001/roots-open/"
)
DEVELOPMENT_ROOT_PREFIX = (
    "outputs/hu_joint_policy/m31_t3_step6d/"
    "candidate02_development/tail_reselection_v2/roots/"
)


def _spec():
    return load_snapshot_spec(SPEC_PATH, repo_root=REPO_ROOT)


def test_v2_pins_the_actual_rearm2_package_source_closure() -> None:
    spec = _spec()
    sources = {entry.source for entry in spec.entries}
    packaged_python = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src/ofc_regular").rglob("*.py")
    }

    assert spec.payload["snapshot_id"] == "hu-joint-policy-r0-snapshot-v2"
    assert packaged_python <= sources
    assert ROADMAP not in sources
    assert {
        "scripts/startup_hu_m31_t3_step6d_full100_v1.sh",
        "scripts/verify_hu_m31_t3_feature_encoder_platform_parity.py",
        "rust/ofc_stage3_feature_encoder/src/lib.rs",
        "configs/hu_m43_attempt08_runtime_requirements.txt",
        (
            "outputs/hu_joint_policy/m31_t3_step6d/"
            "performance_lock_rearm2/"
            "performance_lock_rearm1_startup_failure_closeout.json"
        ),
        (
            "outputs/hu_joint_policy/m31_t3_step6d/"
            "performance_lock_rearm2/precontent_plan_v1.json"
        ),
        (
            "tests/"
            "test_hu_m31_t3_step6d_performance_lock_"
            "rearm2_startup_integration.py"
        ),
    } <= sources


def test_v2_pins_all_prior_root_trees_and_controls() -> None:
    sources = {entry.source for entry in _spec().entries}
    development = {
        source for source in sources if source.startswith(DEVELOPMENT_ROOT_PREFIX)
    }
    old_v1 = {
        source for source in sources if source.startswith(OLD_V1_ROOT_PREFIX)
    }
    rearm1 = {
        source for source in sources if source.startswith(REARM1_ROOT_PREFIX)
    }

    assert len(development) == 100
    assert len(old_v1) == 105
    assert len(rearm1) == 105
    for prior in (old_v1, rearm1):
        assert any(source.endswith("/materialization.json") for source in prior)
        assert any(source.endswith("/seal.json") for source in prior)
        assert sum("/roots/hand_" in source for source in prior) == 100


def test_v2_r0a_is_go_while_unaccepted_windows_t4_remains_deferred() -> None:
    spec = _spec()
    gates = {gate.gate_id: gate for gate in spec.gates}
    assert spec.default_gate_id == "r0a_t3_research_snapshot_v2"
    assert gates["r0a_t3_research_snapshot_v2"].excluded_roles == (
        MISSING_R0B_ROLE,
    )
    assert gates["r0b_exact_t4_release_runtime"].excluded_roles == ()

    r0a = audit_snapshot(spec)
    assert r0a["status"] == "go"
    assert r0a["current_profile_changed"] is False
    assert {issue["role"] for issue in r0a["deferred_entry_issues"]} == {
        MISSING_R0B_ROLE
    }

    r0b = audit_snapshot(spec, gate_id="r0b_exact_t4_release_runtime")
    assert r0b["status"] == "no_go"
    assert {
        (issue["code"], issue.get("role")) for issue in r0b["issues"]
    } >= {("missing_required_entry", MISSING_R0B_ROLE)}
