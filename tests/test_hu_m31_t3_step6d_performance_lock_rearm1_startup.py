from __future__ import annotations

import subprocess
from pathlib import Path


STARTUP = Path("scripts/startup_hu_m31_t3_step6d_full100_v1.sh")


def test_rearm1_startup_identity_is_fully_pinned() -> None:
    source = STARTUP.read_text(encoding="utf-8")
    for value in (
        "lock_rearm1",
        "hu_m31_t3_step6d_candidate02_performance_lock_rearm1_precontent_plan_v1",
        "performance_lock_rearm1_fresh_roots_only",
        "3b8a4230531f0d81c5320b2b0d051878113ac57a38ad97fdb0b1d89e0f17a886",
        "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5",
        "hu_m31_t3_step6d_performance_lock_rearm1_global_claim_v2",
        "global_rearm1_claim_persisted_before_new_root_touch_",
        "hu_m31_t3_step6d_performance_lock_rearm1_root_materialization_v2",
        "all_100_rearm1_roots_materialized_exact_claimed_identity",
        "hu_m31_t3_step6d_performance_lock_rearm1_root_seal_v2",
        "sealed_100_fresh_disjoint_hidden_safe_performance_lock_rearm1_roots",
        "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v2",
        "candidate02_compact_scorer_performance_lock_recovery_v2",
        "performance_lock_recovery_v2",
        "performance_lock_recovery_source_shard_done_v2",
    ):
        assert value in source


def test_rearm1_does_not_replace_v1_or_development_startup_identity() -> None:
    source = STARTUP.read_text(encoding="utf-8")
    for value in (
        "hu_m31_t3_step6d_candidate02_performance_lock_precontent_plan_v1",
        "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b",
        "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4",
        "hu_m31_t3_step6d_candidate02_full100_plan_v1",
        "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758",
        "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd",
    ):
        assert value in source


def test_rearm2_phase_spec_pins_v3_jobs_done_and_smoke_receipt() -> None:
    source = STARTUP.read_text(encoding="utf-8")
    for value in (
        "lock_rearm2",
        "hu_m31_t3_step6d_candidate02_performance_lock_rearm2_precontent_plan_v1",
        "performance_lock_rearm2_fresh_roots_only",
        "8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5",
        "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5",
        "candidate02_compact_scorer_performance_lock_recovery_v3",
        "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v3",
        "performance_lock_recovery_shard_manifest_v3",
        "performance_lock_recovery_source_shard_done_v3",
        "performance_lock_recovery_v3",
        "hu_m31_t3_step6d_performance_lock_rearm2_global_spot_claim_v1",
        "global_rearm2_one_shot_spot_identity_claimed_after_exhaustive_",
        "preauthorize_smoke_receipt_sha256",
    ):
        assert value in source
    assert 'lock_mode = package_phase in ("lock", "lock_rearm1", "lock_rearm2")' in source
    assert '[[ "$RUNNER_DONE_SCHEMA" == "$PRECONTENT_DONE_SCHEMA" ]]' in source


def test_rearm1_startup_shell_is_syntactically_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", STARTUP.as_posix()],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
