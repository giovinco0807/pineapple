from pathlib import Path

from ofc_regular import hu_m31_t3_locked_promotion_cli_v1 as cli
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from ofc_regular.action_key import ACTION_KEY_SCHEMA


def test_build_plan_cli_requires_external_hashes_and_writes_once(
    tmp_path: Path,
) -> None:
    model_id = "m31-cli-fixture"
    model = tmp_path / "manifest.json"
    model.write_bytes(b"model")
    model_sha = promotion.sha256_file(model)
    lock = {
        "schema": promotion.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": model_id,
        "model_sha256": model_sha,
        "state_action_input_schema_sha256": "1" * 64,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {"first": 0.5, "second": 0.5},
        "seat_enabled": {"first": True, "second": True},
        "source_training_threshold_lock_sha256": "2" * 64,
        "source_checkpoint_bundle_identity_sha256": "3" * 64,
        "locked_on_threshold_holdout_only": True,
        "model_change_requires_new_lock": True,
        "teacher_ev_lcb_is_runtime_gate": False,
        "top1_accuracy_is_promotion_gate": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    lock_path = tmp_path / "threshold.json"
    lock_path.write_bytes(promotion.canonical_bytes(lock))
    registry = tmp_path / "ai_profiles.py"
    registry.write_bytes(b"registry")
    runtime = tmp_path / "closure.zip"
    runtime.write_bytes(b"closure")
    output = tmp_path / "plan.json"
    argv = [
        "build-plan",
        "--plan-id",
        "m31-cli-plan",
        "--model-artifact-id",
        model_id,
        "--model",
        str(model),
        "--expected-model-sha256",
        model_sha,
        "--threshold-lock",
        str(lock_path),
        "--expected-threshold-lock-sha256",
        promotion.sha256_file(lock_path),
        "--policy-registry",
        str(registry),
        "--expected-policy-registry-sha256",
        promotion.sha256_file(registry),
        "--runtime-closure",
        str(runtime),
        "--expected-runtime-closure-sha256",
        promotion.sha256_file(runtime),
        "--output",
        str(output),
    ]
    assert cli.main(argv) == 0
    assert promotion.validate_locked_promotion_plan(
        promotion._read_canonical(output, "plan")
    )
    assert cli.main(argv) == 0

    wrong = list(argv)
    wrong[wrong.index("--expected-model-sha256") + 1] = "0" * 64
    assert cli.main(wrong) == 2
