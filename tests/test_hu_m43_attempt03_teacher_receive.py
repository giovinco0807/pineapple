from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from ofc_regular import validate_hu_m43_attempt03_teacher_receive as receive


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt03.json"
RUN_NAME = "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228"
RUN_DIR = ROOT / "outputs" / "gcp_runs" / RUN_NAME
MANIFEST = RUN_DIR / "manifest.json"
SCHEDULE = RUN_DIR / "shards_manifest.jsonl"
SCRIPT = ROOT / "scripts" / "Receive-GcpHuM43Attempt03TeacherRun.ps1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def test_attempt03_schedule_is_exactly_fit_0_49_and_precal_50_69() -> None:
    _, specs = receive.load_and_validate_schedule(
        plan_path=PLAN, schedule_path=SCHEDULE
    )
    assert [row["shard"] for row in specs[:50]] == list(range(50))
    assert {row["logical_split"] for row in specs[:50]} == {"train.fit"}
    assert [row["shard"] for row in specs[50:]] == list(range(50, 70))
    assert {row["logical_split"] for row in specs[50:]} == {
        "train.precal_holdout"
    }
    assert all(row["candidate_samples"] == 2 for row in specs)
    assert all(row["evaluation_samples"] == 64 for row in specs)
    assert all(set(row["profile_quota_per_shard"].values()) == {2} for row in specs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_seed", 1, "candidate_seed"),
        ("evaluation_samples", 16, "evaluation_samples"),
        ("logical_split", "train.fit", "logical_split"),
    ],
)
def test_schedule_rejects_seed_budget_and_role_mutation(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    rows = [json.loads(line) for line in SCHEDULE.read_text().splitlines()]
    rows[50][field] = value
    path = tmp_path / "schedule.jsonl"
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        receive.load_and_validate_schedule(plan_path=PLAN, schedule_path=path)


def test_local_closure_matches_the_immutable_attempt03_run() -> None:
    audit = receive.validate_closure(
        plan_path=PLAN,
        manifest_path=MANIFEST,
        schedule_path=SCHEDULE,
        source_path=RUN_DIR / "ofc_regular_hu_m43_attempt03_teacher_source.zip",
        startup_path=RUN_DIR / "startup_hu_m43_attempt03_teacher.sh",
        model_manifest_path=RUN_DIR / "source_model_manifest.json",
        native_manifest_path=RUN_DIR / "source_native_manifest.json",
        run_name=RUN_NAME,
        project_id="ofc-solver-485418",
        bucket="pokerhu-ofc-solver-485418-training",
    )
    assert audit["status"] == "pass"
    assert audit["manifest_sha256"] == (
        "1e9cc09b3968322bb5b2eeb137a9ddc7aa067e9b8f3f8a2efd424daea2441aea"
    )
    assert audit["precal_result_content_opened"] is False
    assert audit["current_profile_resolved"] is False


def _write_synthetic_shard(tmp_path: Path) -> dict[str, Path]:
    spec = json.loads(SCHEDULE.read_text(encoding="utf-8").splitlines()[0])
    profiles = (
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    )
    rows = []
    for index in range(10):
        rows.append(
            {
                "schema": "hu_m4_t1_second_training_sample_v2",
                "split": "train",
                "seat": "second",
                "street": "T1",
                "to_act_order": "second",
                "hand_seed": spec["seed_start"] + index * spec["seed_stride"],
                "observation_fingerprint": f"{index + 1:064x}",
                "provenance": {
                    "current_profile_resolved": False,
                    "baseline_profile": "stage18_p1",
                    "t2_profile": "stage9f_p2",
                    "native_batch_threads": 4,
                    "root_profile": profiles[index // 2],
                },
                "search_config": {
                    "candidate_samples": 2,
                    "evaluation_samples": 64,
                    "candidate_seed": spec["candidate_seed"],
                    "evaluation_seed": spec["evaluation_seed"],
                    "child_policy_seed": spec["child_policy_seed"],
                },
            }
        )
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "hu_m4_t1_second_shard_v1",
        "status": "complete",
        "current_profile_resolved": False,
        "teacher_value_status": "diagnostic_not_match_EV",
        "roots": 10,
        "output_sha256": _sha(teacher),
        "config": {
            "baseline_profile": "stage18_p1",
            "t2_profile": "stage9f_p2",
            "batch_child_selectors": True,
            "native_batch_threads": 4,
            "opening_lookahead_samples": 0,
            "roots": 10,
            "seed_start": spec["seed_start"],
            "seed_stride": spec["seed_stride"],
            "candidate_seed": spec["candidate_seed"],
            "evaluation_seed": spec["evaluation_seed"],
            "child_policy_seed": spec["child_policy_seed"],
            "candidate_samples": 2,
            "evaluation_samples": 64,
            "split": "train",
            "run_id": f"{RUN_NAME}:shard=0",
            "root_profile": "stage3_baseline",
            "root_profiles": list(profiles),
            "root_profile_weights": [1.0] * 5,
        },
        "root_population_manifest": {
            "schema": "weighted_quota_seeded_shuffle_v1",
            "profiles": [
                {"profile": profile, "target_count": 2, "completed_count": 2}
                for profile in profiles
            ],
        },
    }
    paths: dict[str, Path] = {"teacher": teacher}
    for name in ("checkpoint", "heartbeat", "generator_summary"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        paths[name] = path
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    done = {
        "schema": "hu_m43_attempt02_teacher_done_v1",
        "status": "complete",
        "run_name": RUN_NAME,
        "shard": 0,
        "split": "train",
        "roots": 10,
        "output_prefix": spec["output_prefix"],
        "manifest_sha256": _sha(MANIFEST),
        "shards_manifest_sha256": _sha(SCHEDULE),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "model_manifest_sha256": manifest["model_manifest_sha256"],
        "native_manifest_sha256": manifest["native_manifest_sha256"],
        "output_sha256": _sha(teacher),
        "checkpoint_sha256": _sha(paths["checkpoint"]),
        "heartbeat_sha256": _sha(paths["heartbeat"]),
    }
    done_path = tmp_path / "DONE.json"
    done_path.write_text(json.dumps(done, sort_keys=True) + "\n", encoding="utf-8")
    paths["done"] = done_path
    return paths


def test_shard_validator_checks_done_summaries_schedule_and_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_synthetic_shard(tmp_path)
    monkeypatch.setattr(
        receive,
        "read_and_audit_shard",
        lambda *args, **kwargs: {"records": 10, "paired_delta_records": 10},
    )
    audit = receive.validate_shard(
        plan_path=PLAN,
        manifest_path=MANIFEST,
        schedule_path=SCHEDULE,
        shard=0,
        teacher_path=paths["teacher"],
        done_path=paths["done"],
        checkpoint_path=paths["checkpoint"],
        heartbeat_path=paths["heartbeat"],
        generator_summary_path=paths["generator_summary"],
        run_name=RUN_NAME,
    )
    assert audit["status"] == "pass"
    assert audit["roots"] == 10
    assert set(audit["profile_counts"].values()) == {2}
    assert audit["candidate_samples"] == 2
    assert audit["evaluation_samples"] == 64
    assert audit["current_profile_resolved"] is False


def test_shard_validator_rejects_completion_summary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_synthetic_shard(tmp_path)
    heartbeat = json.loads(paths["heartbeat"].read_text(encoding="utf-8"))
    heartbeat["config"]["evaluation_samples"] = 16
    paths["heartbeat"].write_text(json.dumps(heartbeat), encoding="utf-8")
    done = json.loads(paths["done"].read_text(encoding="utf-8"))
    done["heartbeat_sha256"] = _sha(paths["heartbeat"])
    paths["done"].write_text(json.dumps(done), encoding="utf-8")
    monkeypatch.setattr(
        receive,
        "read_and_audit_shard",
        lambda *args, **kwargs: {"records": 10, "paired_delta_records": 10},
    )
    with pytest.raises(ValueError, match="evaluation_samples"):
        receive.validate_shard(
            plan_path=PLAN,
            manifest_path=MANIFEST,
            schedule_path=SCHEDULE,
            shard=0,
            teacher_path=paths["teacher"],
            done_path=paths["done"],
            checkpoint_path=paths["checkpoint"],
            heartbeat_path=paths["heartbeat"],
            generator_summary_path=paths["generator_summary"],
            run_name=RUN_NAME,
        )


def test_receiver_keeps_fit_and_precal_cloud_boundaries_separate() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "gcloud storage ls" not in text
    assert '"$script:prefix/results/**' not in text
    assert "Wildcard GCS URI is forbidden" in text
    assert "$fitSpecs.Count -ne 50" in text
    assert "((0..49) -join ',')" in text
    assert "$precalSpecs.Count -ne 20" in text
    assert "((50..69) -join ',')" in text
    canary = text.index("Receive-VerifiedShard -Spec $fitSpecs[0]")
    fan_in = text.index("$fitSpecs | Select-Object -Skip 1", canary)
    assert canary < fan_in
    claim = text.index("Write-Utf8NoBomCreateNew -Path $openClaimPath")
    first_precal = text.index("Receive-VerifiedShard -Spec $precalSpecs[0]", claim)
    finalize = text.index('"ofc_regular.hu_m43_attempt03_contract", "finalize-fresh"')
    assert claim < first_precal < finalize
    assert "claimed_before_any_precal_result_download_or_parse" in text
    assert (
        '"8aed10b143172c9d3b2a98fca199e4fbe4324f5956406fea51c2d17fcb614b9d"'
        in text
    )
    assert "OpenPrecalibration requires PrecalOpenAuthorizationPath" in text
    authorization = text.index('"validate-precal-open"')
    assert authorization < claim
    assert "precal_open_authorization_file_sha256" in text
    assert "original_model_freeze_file_sha256" in text
    assert "model_freeze_lineage_file_sha256" in text
    assert "model_freeze_file_sha256" in text
    assert "pre-cal open claim changed after exclusive creation" in text
    receipt_write = text.index(
        'Write-Utf8NoBom -Path $oneShotReceiptPath'
    )
    receipt_hash = text.index(
        'Add-CanonicalSelfHash -Path $oneShotReceiptPath -Field "receipt_sha256"'
    )
    receipt_verify = text.index(
        'Assert-CanonicalSelfHash -Path $oneShotReceiptPath -Field "receipt_sha256"'
    )
    assert receipt_write < receipt_hash < receipt_verify
    assert "global_consumption_marker_must_be_created_exclusively_before_parse" in text
    assert "attempt03_contract_finalized = $false" in text
    assert "precal_result_prefix_listed = $false" in text
    assert "precal_result_downloaded = $false" in text
    assert "precal_result_opened = $false" in text
    assert "current_profile_mutated = $false" in text


def test_attempt03_freeze_lineage_validator_recomputes_pre_row_boundary() -> None:
    audit = receive.validate_attempt03_model_freeze_lineage(
        repo_root=ROOT,
        model_freeze_path=(
            ROOT / "configs" / "hu_joint_policy_m43_attempt03_model_freeze.json"
        ),
        original_freeze_path=(
            ROOT
            / "configs"
            / "hu_joint_policy_m43_attempt03_model_freeze_original_ce47.json"
        ),
        freeze_lineage_path=(
            ROOT
            / "configs"
            / "hu_joint_policy_m43_attempt03_model_freeze_lineage.json"
        ),
    )
    assert audit["amended_freeze_file_sha256"].startswith("8aed10b1")
    assert audit["original_freeze_file_sha256"].startswith("ce47b111")
    assert audit["freeze_lineage_file_sha256"].startswith("b93999e2")
    assert audit["amendment_precedes_first_row_seconds"] == pytest.approx(
        97.4080822
    )


def test_precal_open_authorization_rehashes_all_fit_artifacts() -> None:
    scratch_root = ROOT / "outputs" / "test_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as temporary:
        work = Path(temporary)
        candidate = work / "candidate.pkl"
        bundle = work / "fit_bundle.pkl"
        candidate.write_bytes(b"candidate")
        bundle.write_bytes(b"fit-bundle")
        amended = (
            ROOT / "configs" / "hu_joint_policy_m43_attempt03_model_freeze.json"
        )
        training_freeze = {
            "schema": "hu_m43_attempt03_training_pipeline_freeze_v1",
            "status": (
                "frozen_after_one_structural_train_fit_canary_before_"
                "row_valued_design_or_any_holdout_open"
            ),
            "parent_model_freeze": {
                "path": (
                    "configs/hu_joint_policy_m43_attempt03_model_freeze.json"
                ),
                "file_sha256": _sha(amended),
            },
            "decision_boundary": {
                "cloud_teacher_rows_may_exist": True,
                "attempt03_fit_rows_received_locally": 1,
                "attempt03_fit_rows_structurally_inspected": 1,
                "attempt03_fit_jsonl_content_parse_count": 1,
                "attempt03_fit_row_valued_labels_or_metrics_used_for_design": 0,
                "precalibration_rows_opened": 0,
                "sealed_calibration_rows_opened": 0,
                "inherited_locked_rows_opened": 0,
                "source_or_config_selected_from_row_values": False,
            },
        }
        training_freeze["freeze_sha256"] = receive._canonical_sha256(
            training_freeze
        )
        training_freeze_path = work / "training_freeze.json"
        training_freeze_path.write_text(
            json.dumps(training_freeze, sort_keys=True), encoding="utf-8"
        )
        training_freeze_file_sha = _sha(training_freeze_path)
        training_config_sha = "a" * 64
        contract = {
            "schema": "hu_m43_attempt03_v5_fold_cloud_contract_v1",
            "status": "frozen_fit700_only",
            "training_config_sha256": training_config_sha,
            "model_freeze": {"file_sha256": _sha(amended)},
            "training_freeze": {"file_sha256": training_freeze_file_sha},
        }
        contract["contract_sha256"] = receive._canonical_sha256(contract)
        contract_path = work / "fold_contract.json"
        contract_path.write_text(
            json.dumps(contract, sort_keys=True), encoding="utf-8"
        )
        fit_manifest = {
            "schema": "hu_m43_attempt03_v5_fit_manifest_v1",
            "status": "fit_candidate_precalibration_unopened",
            "model_sha256": _sha(candidate),
            "fit_bundle_sha256": _sha(bundle),
            "fold_cloud_contract_sha256": contract["contract_sha256"],
            "training_config_sha256": training_config_sha,
            "model_freeze_file_sha256": _sha(amended),
            "training_freeze_file_sha256": training_freeze_file_sha,
        }
        fit_manifest["manifest_sha256"] = receive._canonical_sha256(
            fit_manifest
        )
        fit_manifest_path = work / "fit_manifest.json"
        fit_manifest_path.write_text(
            json.dumps(fit_manifest, sort_keys=True), encoding="utf-8"
        )
        authorization = {
            "schema": "hu_m43_attempt03_precal_open_authorization_v1",
            "status": "authorized_to_create_receiver_open_claim",
            "decision_basis": "frozen_schema_and_hash_chain_only",
            "row_valued_metric_used_for_authorization": False,
            "precalibration_path_received": False,
            "precalibration_content_read": False,
            "candidate_model_path": str(candidate.resolve()),
            "candidate_model_sha256": _sha(candidate),
            "fit_bundle_path": str(bundle.resolve()),
            "fit_bundle_sha256": _sha(bundle),
            "fit_manifest_path": str(fit_manifest_path.resolve()),
            "fit_manifest_file_sha256": _sha(fit_manifest_path),
            "fit_manifest_canonical_sha256": fit_manifest["manifest_sha256"],
            "fold_cloud_contract_path": str(contract_path.resolve()),
            "fold_cloud_contract_file_sha256": _sha(contract_path),
            "fold_cloud_contract_canonical_sha256": contract["contract_sha256"],
            "training_config_sha256": training_config_sha,
            "model_freeze_path": str(amended.resolve()),
            "model_freeze_file_sha256": _sha(amended),
            "training_freeze_path": str(training_freeze_path.resolve()),
            "training_freeze_file_sha256": training_freeze_file_sha,
            "v5_implementation_sha256": "b" * 64,
            "stage18_model_sha256": "c" * 64,
            "fit_manifest_status_required": (
                "fit_candidate_precalibration_unopened"
            ),
            "candidate_safety_enabled": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
        }
        authorization["authorization_sha256"] = receive._canonical_sha256(
            authorization
        )
        authorization_path = work / "authorization.json"
        authorization_path.write_text(
            json.dumps(authorization, sort_keys=True), encoding="utf-8"
        )
        kwargs = {
            "repo_root": ROOT,
            "authorization_path": authorization_path,
            "model_freeze_path": amended,
            "original_freeze_path": (
                ROOT
                / "configs"
                / "hu_joint_policy_m43_attempt03_model_freeze_original_ce47.json"
            ),
            "freeze_lineage_path": (
                ROOT
                / "configs"
                / "hu_joint_policy_m43_attempt03_model_freeze_lineage.json"
            ),
        }
        audit = receive.validate_precal_open_authorization(**kwargs)
        assert audit["status"] == "pass_without_precalibration_access"
        assert audit["precalibration_content_read"] is False

        candidate.write_bytes(b"tampered-after-authorization")
        with pytest.raises(ValueError, match="actual hash changed"):
            receive.validate_precal_open_authorization(**kwargs)


def test_receiver_canary_only_addresses_shard_zero_without_fit_publication() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    branch = text.index("if ($CanaryOnly)")
    canary = text.index(
        "Receive-VerifiedShard -Spec $fitSpecs[0]", branch
    )
    canary_return = text.index("return", canary)
    full_fan_in = text.index("$fitSpecs | Select-Object -Skip 1", canary_return)
    assert branch < canary < canary_return < full_fan_in
    segment = text[branch:canary_return]
    assert "Receive-VerifiedShard -Spec $fitSpecs[1]" not in segment
    assert "shards_1_through_69_addressed = $false" in segment
    assert "fit_output_published = $false" in segment
    assert "CanaryOnly does not publish an OutputDir" in segment


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt03_receiver_parses_as_powershell() -> None:
    shell = _powershell()
    assert shell is not None
    command = (
        f"$t=$null;$e=$null;"
        f"[Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$t,[ref]$e)|Out-Null;"
        "$e|%{$_.Message};if($e.Count){exit 1}"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
