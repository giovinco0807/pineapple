from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_start_script_has_four_separate_phases_and_spot_guards() -> None:
    text = _text("Start-GcpHuM43Attempt08Audit50Run.ps1")
    for phase in ("PackageOnly", "AuthorizeAudit", "AuthorizeLaunch", "CreateInstances"):
        assert phase in text
    assert "provisioning-model','SPOT" in text
    assert "1449487925682397051" in text
    assert "c4-highmem-4" in text
    assert "0-24" in text
    assert "DONE already exists" in text
    assert "current" not in text.lower() or "current_profile_mutated" in text


def test_status_is_done_only_and_receive_claims_before_content() -> None:
    status = _text("Get-GcpHuM43Attempt08Audit50RunStatus.ps1")
    receive = _text("Receive-GcpHuM43Attempt08Audit50Run.ps1")
    assert "DONE.json" in status
    assert "teacher.jsonl" not in status
    assert receive.index("audit50_output_consumption.json") < receive.index(
        "$names = @('teacher.jsonl'"
    )
    assert "audit50_selector.json" in receive
    assert "select-once" in receive
    assert "automatic gate reevaluation is forbidden" in receive
    assert "MaxParallel" not in receive
    assert "$resumePreSelector" in receive
    assert "Assert-M43A8Audit50SafePreSelectorState" in receive
    assert "Crash recovery after the atomic Move" in receive
    assert "Remove-Item -LiteralPath $staging -Recurse -Force" in receive
    assert receive.index("validate-done-set") < receive.index("@('claim'")
    assert receive.index("@('claim'") < receive.index("$names = @('teacher.jsonl'")
    assert receive.index("revalidate audit50 merge during pre-selector recovery") < receive.index(
        "@('select-once'"
    )


def test_startup_reconstructs_separate_frozen_base_and_publishes_done_last() -> None:
    text = _text("startup_hu_m43_attempt08_audit50.sh")
    assert "AUDIT=\"$BASE/audit_run\"" in text
    assert "DEV=\"$BASE/development_run\"" in text
    assert "validate-launch" in text
    assert '"$RESUME_URI/objects/$digest/$name"' in text
    assert "validate-resume-commit" in text
    assert text.index("validate-resume-commit") < text.index(
        'gcloud storage cp "$RESUME_URI/objects/$digest/$name"'
    )
    assert text.index(
        'upload_once_or_verify "$RESULT/$name" "$RESUME_URI/objects/$digest/$name"'
    ) < text.index(
        'upload_once_or_verify "$RESULT/resume_commit.json" "$RESUME_URI/resume_commit.json"'
    )
    assert text.index('for name in "${FILES[@]}"; do upload_once_or_verify') < text.index(
        'upload_once_or_verify "$RESULT/DONE.json"'
    )
    assert text.index("validate-completed") < text.index(
        'for name in "${FILES[@]}"; do upload_once_or_verify'
    )
    assert "run_hu_m43_attempt08_future_audit" not in text
    assert "current" not in text.lower()


def test_claim_language_records_truthful_preclaim_operations() -> None:
    source = (
        ROOT / "src/ofc_regular/hu_m43_attempt08_audit50_spot.py"
    ).read_text(encoding="utf-8")
    assert "candidate_model_artifact_sha256_validated" in source
    assert "candidate_model_deserialized" in source
    assert "candidate_model_inference_performed" in source
    assert "root_spec_derived_from_frozen_schedule" in source
    assert "claimed_before_any_audit50_seed_model_observation_or_teacher" not in source
    assert "root_claimed_before_seed_model_observation_or_teacher" not in source
