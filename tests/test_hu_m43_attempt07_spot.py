from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt07_spot as spot
from ofc_regular.hu_m43_attempt07_contract import (
    M43_ATTEMPT07_PROFILES,
    M43_ATTEMPT07_SEED_STRIDE,
    load_and_validate_attempt07_plan,
)
from ofc_regular.hu_m43_attempt06_spot import PINNED_TEMPLATE_RUN


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
START = ROOT / "scripts" / "Start-GcpHuM43Attempt07DevelopmentRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt07DevelopmentRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt07DevelopmentRun.ps1"
STARTUP = ROOT / "scripts" / "startup_hu_m43_attempt07_development.sh"

SEED_BASES = {
    "hand": 60_106_071_901,
    "screen": 61_106_071_901,
    "rerank": 62_106_071_901,
    "veto": 63_106_071_901,
    "assessment": 64_106_071_901,
    "child": 65_106_071_901,
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def test_schedule_is_exact_100_mod5_one_root_shards_with_six_fixed_seeds() -> None:
    plan = load_and_validate_attempt07_plan(PLAN)
    rows = spot.build_attempt07_spot_schedule(plan)

    assert spot.EXPECTED_SHARDS == 100
    assert spot.ROOTS_PER_SHARD == 1
    assert spot.NATIVE_BATCH_THREADS == 4
    assert spot.MAX_WAVE_SHARDS == 50
    assert len(rows) == 100
    assert M43_ATTEMPT07_PROFILES == (
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    )

    expected_keys = {
        "schema",
        "shard",
        "root_index",
        "roots",
        "root_profile",
        "seeds",
        "baseline_profile",
        "continuation_profile",
        "batch_child_selectors",
        "native_batch_threads",
        "output_prefix",
    }
    all_seeds: list[int] = []
    for index, row in enumerate(rows):
        assert set(row) == expected_keys
        assert row["schema"] == spot.SHARD_SCHEMA
        assert row["shard"] == row["root_index"] == index
        assert row["roots"] == 1
        assert row["root_profile"] == M43_ATTEMPT07_PROFILES[index % 5]
        assert row["baseline_profile"] == "stage18_p1"
        assert row["continuation_profile"] == "stage9f_p2"
        assert row["batch_child_selectors"] is True
        assert row["native_batch_threads"] == 4
        assert row["output_prefix"] == f"shard_{index:03d}"
        assert set(row["seeds"]) == set(SEED_BASES)
        assert row["seeds"] == {
            domain: base + M43_ATTEMPT07_SEED_STRIDE * index
            for domain, base in SEED_BASES.items()
        }
        assert len(set(row["seeds"].values())) == 6
        all_seeds.extend(row["seeds"].values())

    assert len(set(all_seeds)) == 600
    assert {
        profile: sum(row["root_profile"] == profile for row in rows)
        for profile in M43_ATTEMPT07_PROFILES
    } == {profile: 20 for profile in M43_ATTEMPT07_PROFILES}


def test_package_source_cannot_generate_or_run_a_development_root() -> None:
    source = inspect.getsource(spot.package_attempt07_spot)

    for forbidden in (
        "generate_t1_second_root(",
        "build_attempt07_root",
        "run_hu_m43_attempt07_development",
        "teacher.jsonl",
        "root.jsonl",
    ):
        assert forbidden not in source
    for frozen_boundary in (
        '"fresh_root_opened": False',
        '"teacher_executed": False',
        '"gcloud_invoked": False',
        '"instances_created": False',
        '"current_profile_mutated": False',
        '"runtime_policy_activated": False',
        '"recommended_machine_type": "c4-standard-4"',
        '"recommended_wave_shards": MAX_WAVE_SHARDS',
    ):
        assert frozen_boundary in source


def test_package_only_function_freezes_exact_preflight_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = runpy.run_path(
        str(ROOT / "tests/test_finalize_hu_m43_attempt07_preflight.py")
    )
    fixture = namespace["_fixture"](tmp_path / "preflight")
    repo = tmp_path / "repo"
    source_dir = repo / "src/ofc_regular"
    source_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "src/ofc_regular/ai_profiles.py", source_dir / "ai_profiles.py")
    template = repo / "outputs/gcp_runs" / PINNED_TEMPLATE_RUN / "package_src"
    template.mkdir(parents=True)
    (template / "source_model_manifest.json").write_text("{}\n", encoding="utf-8")
    (template / "source_native_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        spot,
        "_verify_pinned_runtime_closure",
        lambda _root: ({"models": []}, {"binaries": []}),
    )
    run_name = "attempt07-development-package-test"
    run_dir = repo / "outputs/gcp_runs" / run_name
    result = spot.package_attempt07_spot(
        repo_root=repo,
        run_dir=run_dir,
        run_name=run_name,
        plan_path=PLAN,
        status_path=ROOT / "configs/hu_joint_policy_m43_attempt07_status.json",
        model_path=ROOT
        / "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl",
        startup_path=STARTUP,
        preflight_aggregate_path=fixture["aggregate"],
        preflight_plan_path=ROOT
        / "configs/hu_joint_policy_m43_attempt07_preflight.json",
        preflight_manifest_path=fixture["manifest"],
        preflight_schedule_path=fixture["schedule"],
        preflight_launch_authorization_path=fixture["authorization"],
        preflight_done_paths=fixture["done"],
    )
    assert result["status"] == "packaged_without_root_or_gcloud"
    assert result["fresh_root_opened"] is False
    assert result["teacher_executed"] is False
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    assert set(manifest) == spot._PACKAGE_MANIFEST_KEYS
    assert spot._resolve_attempt07_authorization_layout(run_dir)["name"] == (
        "full_package_run"
    )
    assert manifest["preflight_done_sha256"] == {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in fixture["done"].items()
    }
    assert manifest["source_zip_sha256"] == hashlib.sha256(
        (run_dir / "ofc_regular_hu_m43_attempt07_development_source.zip").read_bytes()
    ).hexdigest()
    for label in spot._PREFLIGHT_SLOTS:
        assert (
            run_dir / "preflight_closure" / f"{label}.DONE.json"
        ).read_bytes() == fixture["done"][label].read_bytes()
    resumed = spot.package_attempt07_spot(
        repo_root=repo,
        run_dir=run_dir,
        run_name=run_name,
        plan_path=PLAN,
        status_path=ROOT / "configs/hu_joint_policy_m43_attempt07_status.json",
        model_path=ROOT
        / "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl",
        startup_path=STARTUP,
        preflight_aggregate_path=fixture["aggregate"],
        preflight_plan_path=ROOT
        / "configs/hu_joint_policy_m43_attempt07_preflight.json",
        preflight_manifest_path=fixture["manifest"],
        preflight_schedule_path=fixture["schedule"],
        preflight_launch_authorization_path=fixture["authorization"],
        preflight_done_paths=fixture["done"],
        resume_existing=True,
    )
    assert resumed["resumed_existing_package"] is True


def _authorization_fixture(tmp_path: Path) -> dict[str, Path]:
    namespace = runpy.run_path(
        str(ROOT / "tests/test_finalize_hu_m43_attempt07_preflight.py")
    )
    fixture = namespace["_fixture"](tmp_path / "source")
    work = tmp_path / "worker"
    (work / "configs").mkdir(parents=True)
    (work / "frozen/preflight/done").mkdir(parents=True)
    shutil.copy2(PLAN, work / "configs/hu_joint_policy_m43_attempt07.json")
    shutil.copy2(
        ROOT / "configs/hu_joint_policy_m43_attempt07_preflight.json",
        work / "configs/hu_joint_policy_m43_attempt07_preflight.json",
    )
    status = ROOT / "configs/hu_joint_policy_m43_attempt07_status.json"
    shutil.copy2(status, work / "configs/hu_joint_policy_m43_attempt07_status.json")
    shutil.copy2(fixture["development_schedule"], work / "shards_manifest.jsonl")
    shutil.copy2(fixture["aggregate"], work / "frozen/attempt07_preflight_aggregate.json")
    shutil.copy2(fixture["manifest"], work / "frozen/preflight/manifest.json")
    shutil.copy2(fixture["schedule"], work / "frozen/preflight/shards_manifest.jsonl")
    shutil.copy2(
        fixture["authorization"], work / "frozen/preflight/launch_authorization.json"
    )
    for label, path in fixture["done"].items():
        shutil.copy2(path, work / f"frozen/preflight/done/{label}.json")
    template = ROOT / "outputs/gcp_runs" / PINNED_TEMPLATE_RUN / "package_src"
    shutil.copy2(template / "source_model_manifest.json", work / "source_model_manifest.json")
    shutil.copy2(template / "source_native_manifest.json", work / "source_native_manifest.json")
    (work / "source_closure_manifest.json").write_bytes(b"frozen-test-closure\n")

    manifest = json.loads(fixture["development_manifest"].read_bytes())
    manifest["status_sha256"] = hashlib.sha256(status.read_bytes()).hexdigest()
    manifest["source_closure_sha256"] = hashlib.sha256(
        (work / "source_closure_manifest.json").read_bytes()
    ).hexdigest()
    manifest["startup_sha256"] = hashlib.sha256(STARTUP.read_bytes()).hexdigest()
    (work / "manifest.json").write_bytes(spot.canonical_json_bytes(manifest))
    auth_path = work / "spot_authorization.json"
    namespace["_run"](
        tmp_path / "source",
        {
            **fixture,
            "development_manifest": work / "manifest.json",
            "development_schedule": work / "shards_manifest.jsonl",
        },
        output=auth_path,
    )
    return {
        "work": work,
        "manifest": work / "manifest.json",
        "aggregate": work / "frozen/attempt07_preflight_aggregate.json",
        "authorization": auth_path,
    }


def test_authorization_recomputes_metrics_from_packaged_raw_done(tmp_path: Path) -> None:
    fixture = _authorization_fixture(tmp_path)
    audit = spot.validate_attempt07_spot_authorization(
        authorization_path=fixture["authorization"],
        manifest_path=fixture["manifest"],
        preflight_aggregate_path=fixture["aggregate"],
    )
    assert audit["status"] == "pass"

    aggregate = json.loads(fixture["aggregate"].read_bytes())
    aggregate["operational_diagnostics"]["jobs"]["root0_batch_a"][
        "elapsed_seconds"
    ] = 101.0
    aggregate["operational_diagnostics"]["elapsed_seconds_sum"] += 1.0
    fixture["aggregate"].write_bytes(spot.canonical_json_bytes(aggregate))
    manifest = json.loads(fixture["manifest"].read_bytes())
    manifest["preflight_aggregate_sha256"] = hashlib.sha256(
        fixture["aggregate"].read_bytes()
    ).hexdigest()
    fixture["manifest"].write_bytes(spot.canonical_json_bytes(manifest))
    authorization = json.loads(fixture["authorization"].read_bytes())
    authorization["preflight_aggregate_sha256"] = manifest[
        "preflight_aggregate_sha256"
    ]
    authorization["development_manifest_sha256"] = hashlib.sha256(
        fixture["manifest"].read_bytes()
    ).hexdigest()
    metrics = authorization["operational_metrics"]
    metrics["elapsed_seconds"]["root0_batch_a"] = 101.0
    metrics["root0_batch_median_elapsed_seconds"] = 105.5
    metrics["root0_batch_replicate_elapsed_ratio"] = 110.0 / 101.0
    metrics["scalar_to_root0_batch_median_speedup"] = 300.0 / 105.5
    gates = authorization["operational_gates"]
    gates["root0_batch_replicate_elapsed_ratio"]["observed"] = 110.0 / 101.0
    gates["scalar_to_root0_batch_median_speedup"]["observed"] = 300.0 / 105.5
    fixture["authorization"].write_bytes(spot.canonical_json_bytes(authorization))

    with pytest.raises(ValueError, match="operational metrics changed"):
        spot.validate_attempt07_spot_authorization(
            authorization_path=fixture["authorization"],
            manifest_path=fixture["manifest"],
            preflight_aggregate_path=fixture["aggregate"],
        )


def test_authorization_rejects_self_consistent_weakened_preflight_plan(
    tmp_path: Path,
) -> None:
    fixture = _authorization_fixture(tmp_path)
    plan_path = (
        fixture["work"] / "configs/hu_joint_policy_m43_attempt07_preflight.json"
    )
    plan = json.loads(plan_path.read_bytes())
    semantic_rewrite = tmp_path / "same-plan-different-bytes.json"
    semantic_rewrite.write_bytes(spot.canonical_json_bytes(plan))
    assert semantic_rewrite.read_bytes() != plan_path.read_bytes()
    with pytest.raises(ValueError, match="Attempt07 preflight plan changed"):
        spot.load_preflight_plan(semantic_rewrite)

    plan["operational_go_no_go"]["batch_elapsed_seconds_per_root_max"] = 999999.0
    plan_path.write_bytes(spot.canonical_json_bytes(plan))
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    aggregate = json.loads(fixture["aggregate"].read_bytes())
    aggregate["contract"]["preflight_plan_sha256"] = plan_sha256
    fixture["aggregate"].write_bytes(spot.canonical_json_bytes(aggregate))
    aggregate_sha256 = hashlib.sha256(fixture["aggregate"].read_bytes()).hexdigest()

    manifest = json.loads(fixture["manifest"].read_bytes())
    manifest["preflight_plan_sha256"] = plan_sha256
    manifest["preflight_aggregate_sha256"] = aggregate_sha256
    fixture["manifest"].write_bytes(spot.canonical_json_bytes(manifest))

    authorization = json.loads(fixture["authorization"].read_bytes())
    authorization["preflight_plan_sha256"] = plan_sha256
    authorization["preflight_aggregate_sha256"] = aggregate_sha256
    authorization["development_manifest_sha256"] = hashlib.sha256(
        fixture["manifest"].read_bytes()
    ).hexdigest()
    authorization["operational_gates"]["batch_elapsed_seconds_per_root"][
        "requirement"
    ] = "<= 999999.0"
    fixture["authorization"].write_bytes(spot.canonical_json_bytes(authorization))

    with pytest.raises(ValueError, match="Attempt07 preflight plan changed"):
        spot.validate_attempt07_spot_authorization(
            authorization_path=fixture["authorization"],
            manifest_path=fixture["manifest"],
            preflight_aggregate_path=fixture["aggregate"],
        )


def test_startup_source_closure_rejects_unlisted_extra_file(tmp_path: Path) -> None:
    text = _read(STARTUP)
    marker = "python3 - <<'PY'\n"
    start = text.index(marker) + len(marker)
    validator = text[start : text.index("\nPY\n", start)]
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"closed payload\n")
    closure = {
        "schema": spot.SOURCE_CLOSURE_SCHEMA,
        "status": "closed_before_any_development_root",
        "run_name": "attempt07-source-closure-test",
        "plan_sha256": "0" * 64,
        "status_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "ai_profiles_sha256": "3" * 64,
        "preflight_aggregate_sha256": "4" * 64,
        "preflight_plan_sha256": "5" * 64,
        "preflight_manifest_sha256": "6" * 64,
        "preflight_schedule_sha256": "7" * 64,
        "preflight_launch_authorization_sha256": "8" * 64,
        "preflight_done_sha256": {},
        "files": [
            {
                "path": "payload.bin",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
        "fresh_root_opened": False,
        "teacher_executed": False,
        "current_profile_mutated": False,
    }
    (tmp_path / "source_closure_manifest.json").write_bytes(
        spot.canonical_json_bytes(closure)
    )
    valid = subprocess.run(
        [sys.executable, "-c", validator], cwd=tmp_path, check=False
    )
    assert valid.returncode == 0

    (tmp_path / "unlisted-extra.py").write_text("raise SystemExit\n", encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, "-c", validator], cwd=tmp_path, check=False
    )
    assert rejected.returncode != 0


def _write_complete_done_set(fixture: dict[str, Path], done_root: Path) -> None:
    manifest_path = fixture["manifest"]
    manifest = json.loads(manifest_path.read_bytes())
    schedule = spot.load_schedule(fixture["work"] / "shards_manifest.jsonl")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    authorization_sha256 = hashlib.sha256(
        fixture["authorization"].read_bytes()
    ).hexdigest()
    done_root.mkdir()
    artifact_sha256 = "a" * 64
    for spec in schedule:
        shard = spec["shard"]
        done = {
            "schema": spot.DONE_SCHEMA,
            "status": "complete",
            "run_name": manifest["run_name"],
            "run_id": f"{manifest['run_name']}:shard={shard}",
            "shard": shard,
            "root_index": shard,
            "root_profile": spec["root_profile"],
            "seeds": spec["seeds"],
            "output_prefix": spec["output_prefix"],
            "manifest_sha256": manifest_sha256,
            "source_sha256": manifest["source_zip_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "schedule_sha256": manifest["schedule_sha256"],
            "plan_sha256": manifest["plan_sha256"],
            "status_sha256": manifest["status_sha256"],
            "model_sha256": manifest["model_sha256"],
            "ai_profiles_sha256": manifest["ai_profiles_sha256"],
            "preflight_plan_sha256": manifest["preflight_plan_sha256"],
            "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
            "source_closure_sha256": manifest["source_closure_sha256"],
            "source_model_manifest_sha256": manifest[
                "source_model_manifest_sha256"
            ],
            "source_native_manifest_sha256": manifest[
                "source_native_manifest_sha256"
            ],
            "authorization_sha256": authorization_sha256,
            "global_claim_sha256": artifact_sha256,
            "root_claim_sha256": artifact_sha256,
            "output_sha256": artifact_sha256,
            "checkpoint_sha256": artifact_sha256,
            "heartbeat_sha256": artifact_sha256,
            "generator_summary_sha256": artifact_sha256,
            "run_log_sha256": artifact_sha256,
            "config_sha256": artifact_sha256,
            "elapsed_seconds": 1.0,
            "peak_rss_bytes": 1,
            "native_batch_threads": spot.NATIVE_BATCH_THREADS,
            "teacher_values_are_realized_match_ev": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        assert set(done) == spot._DONE_KEYS
        (done_root / f"DONE-{shard:03d}.json").write_bytes(
            spot.canonical_json_bytes(done)
        )


def test_consumption_claim_checks_full_done_closure_before_claim(tmp_path: Path) -> None:
    fixture = _authorization_fixture(tmp_path)
    done_root = tmp_path / "done"
    _write_complete_done_set(fixture, done_root)
    arguments = {
        "done_root": done_root,
        "schedule_path": fixture["work"] / "shards_manifest.jsonl",
        "manifest_path": fixture["manifest"],
        "authorization_path": fixture["authorization"],
        "preflight_aggregate_path": fixture["aggregate"],
    }
    claim = spot.create_attempt07_output_consumption_claim(
        **arguments, output=tmp_path / "valid-claim.json"
    )
    assert claim["all_done_markers_verified"] is True

    done_path = done_root / "DONE-000.json"
    original = done_path.read_bytes()
    adversarial_values = {
        "source_sha256": "b" * 64,
        "startup_sha256": "b" * 64,
        "status_sha256": "b" * 64,
        "source_closure_sha256": "b" * 64,
        "source_model_manifest_sha256": "b" * 64,
        "source_native_manifest_sha256": "b" * 64,
        "native_batch_threads": spot.NATIVE_BATCH_THREADS + 1,
        "teacher_values_are_realized_match_ev": True,
    }
    for field, replacement in adversarial_values.items():
        changed = json.loads(original)
        assert changed[field] != replacement
        changed[field] = replacement
        done_path.write_bytes(spot.canonical_json_bytes(changed))
        with pytest.raises(ValueError, match="DONE-only claim identity changed"):
            spot.create_attempt07_output_consumption_claim(
                **arguments, output=tmp_path / f"rejected-{field}.json"
            )
        done_path.write_bytes(original)

    changed = json.loads(original)
    changed["unlisted_identity"] = "injected"
    done_path.write_bytes(spot.canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="DONE fields changed"):
        spot.create_attempt07_output_consumption_claim(
            **arguments, output=tmp_path / "rejected-extra-field.json"
        )


def _received_closure(fixture: dict[str, Path], destination: Path) -> Path:
    work = fixture["work"]
    destination.mkdir()
    copies = {
        fixture["manifest"]: "manifest.json",
        work / "shards_manifest.jsonl": "shards_manifest.jsonl",
        work / "source_closure_manifest.json": "source_closure_manifest.json",
        work / "configs/hu_joint_policy_m43_attempt07.json": (
            "hu_joint_policy_m43_attempt07.json"
        ),
        work / "configs/hu_joint_policy_m43_attempt07_status.json": (
            "hu_joint_policy_m43_attempt07_status.json"
        ),
        work / "configs/hu_joint_policy_m43_attempt07_preflight.json": (
            "hu_joint_policy_m43_attempt07_preflight.json"
        ),
        work / "frozen/attempt07_preflight_aggregate.json": (
            "attempt07_preflight_aggregate.json"
        ),
        work / "source_model_manifest.json": "source_model_manifest.json",
        work / "source_native_manifest.json": "source_native_manifest.json",
        STARTUP: "startup_hu_m43_attempt07_development.sh",
        fixture["authorization"]: "spot_authorization.json",
    }
    for source, relative in copies.items():
        shutil.copy2(source, destination / relative)
    preflight = destination / "preflight_closure"
    preflight.mkdir()
    for name in ("manifest.json", "shards_manifest.jsonl", "launch_authorization.json"):
        shutil.copy2(work / "frozen/preflight" / name, preflight / name)
    for label in spot._PREFLIGHT_SLOTS:
        shutil.copy2(
            work / "frozen/preflight/done" / f"{label}.json",
            preflight / f"{label}.DONE.json",
        )
    return destination


def _valid_attempt07_row(root_index: int) -> dict:
    namespace = runpy.run_path(
        str(ROOT / "tests/test_select_hu_m43_attempt07_development_arm.py")
    )
    plan = load_and_validate_attempt07_plan(PLAN)
    schedules = namespace["enumerate_attempt07_seed_schedules"](
        plan, population="development"
    )
    seeds = {domain: values[root_index] for domain, values in schedules.items()}
    observation = namespace["_root"](root_index)
    profile = M43_ATTEMPT07_PROFILES[root_index % len(M43_ATTEMPT07_PROFILES)]
    actions = namespace["generate_turn_actions"](
        observation.hero_board, observation.dealt_cards
    )
    baseline = namespace["action_key"](actions[-1]).to_token()
    teacher_module = namespace["teacher"]
    original_scalar = teacher_module._score_actions
    original_batch = teacher_module._score_actions_batched
    teacher_module._score_actions = namespace["_scorer"]
    teacher_module._score_actions_batched = namespace["_scorer"]
    try:
        teacher_row = teacher_module.evaluate_attempt07_t1_second(
            observation,
            baseline_action_key=baseline,
            ranker=namespace["_Ranker"](),
            t2_policies={
                seat: namespace["_Stage9fPolicy"](seat)
                for seat in ("first", "second")
            },
            config=teacher_module.Attempt07TeacherConfig(
                frozen_model_sha256=namespace["MODEL_HASH"],
                screen_seed=seeds["screen"],
                rerank_seed=seeds["rerank"],
                veto_seed=seeds["veto"],
                assessment_seed=seeds["assessment"],
                child_policy_seed=seeds["child"],
                run_id=(
                    f"attempt07-development-test:shard={root_index}:root={root_index}:"
                    f"seed={seeds['hand']}:obs={observation.fingerprint()}"
                ),
                batch_child_selectors=True,
            ),
        )
    finally:
        teacher_module._score_actions = original_scalar
        teacher_module._score_actions_batched = original_batch
    provenance = namespace["_provenance"](
        root_index=root_index,
        profile=profile,
        seeds=seeds,
        observation=observation,
        baseline_token=baseline,
    )
    provenance["run_id"] = f"attempt07-development-test:shard={root_index}"
    fixed = namespace["runner"]._fixed_contract(
        root_index=root_index,
        root_profile=profile,
        seeds=seeds,
        run_id=provenance["run_id"],
        plan_sha256=provenance["plan_sha256"],
        ai_profiles_sha256=provenance["ai_profiles_sha256"],
        model_sha256=provenance["model_sha256"],
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    provenance["config_sha256"] = namespace["runner"]._canonical_sha256(fixed)
    return {
        "schema": namespace["runner"].ATTEMPT07_SHARD_ROW_SCHEMA,
        "root_index": root_index,
        "hand_seed": seeds["hand"],
        "root_profile": profile,
        "policy_observation": observation.to_dict(),
        "baseline_action_key": baseline,
        "provenance": provenance,
        "teacher": teacher_row,
    }


def test_received_closure_layout_authorizes_and_audits_shard(tmp_path: Path) -> None:
    fixture = _authorization_fixture(tmp_path / "fixture")
    closure = _received_closure(fixture, tmp_path / "closure")
    authorization_audit = spot.validate_attempt07_spot_authorization(
        authorization_path=fixture["authorization"],
        manifest_path=closure / "manifest.json",
        preflight_aggregate_path=closure / "attempt07_preflight_aggregate.json",
    )
    assert authorization_audit["status"] == "pass"

    manifest = json.loads((closure / "manifest.json").read_bytes())
    schedule = spot.load_schedule(closure / "shards_manifest.jsonl")
    spec = schedule[0]
    shard_dir = tmp_path / "shard_000"
    shard_dir.mkdir()
    row = _valid_attempt07_row(0)
    teacher_path = shard_dir / "teacher.jsonl"
    teacher_path.write_bytes(spot.canonical_json_bytes(row))
    config_sha256 = row["provenance"]["config_sha256"]
    output_sha256 = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    checkpoint = {
        "schema": spot.ATTEMPT07_CHECKPOINT_SCHEMA,
        "completed_roots": 1,
        "root_index": 0,
        "config_sha256": config_sha256,
    }
    heartbeat = {
        "schema": spot.ATTEMPT07_HEARTBEAT_SCHEMA,
        "status": "complete",
        "root_index": 0,
        "output_sha256": output_sha256,
    }
    generator_summary = {
        "schema": spot.ATTEMPT07_SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "root_index": 0,
        "config_sha256": config_sha256,
        "output_sha256": output_sha256,
    }
    (shard_dir / "checkpoint.json").write_bytes(spot.canonical_json_bytes(checkpoint))
    (shard_dir / "heartbeat.json").write_bytes(spot.canonical_json_bytes(heartbeat))
    (shard_dir / "generator_summary.json").write_bytes(
        spot.canonical_json_bytes(generator_summary)
    )
    (shard_dir / "run.log").write_bytes(b"attempt07 test run\n")
    shutil.copy2(fixture["authorization"], shard_dir / "authorization.json")
    manifest_sha256 = hashlib.sha256((closure / "manifest.json").read_bytes()).hexdigest()
    authorization_sha256 = hashlib.sha256(
        fixture["authorization"].read_bytes()
    ).hexdigest()
    common_claim = {
        "run_name": manifest["run_name"],
        "manifest_sha256": manifest_sha256,
        "source_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "status_sha256": manifest["status_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "authorization_sha256": authorization_sha256,
        "native_batch_threads": 4,
    }
    global_claim = {
        **common_claim,
        "schema": spot.GLOBAL_CLAIM_SCHEMA,
        "status": "claimed_before_any_development_root",
        "roots": 100,
        "shards": 100,
        "roots_per_shard": 1,
        "development_started_when_claimed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    root_claim = {
        **common_claim,
        "schema": spot.ROOT_CLAIM_SCHEMA,
        "status": "claimed_before_root_generation",
        "run_id": f"{manifest['run_name']}:shard=0",
        "root_index": 0,
        "root_profile": spec["root_profile"],
        "seeds": spec["seeds"],
        "output_prefix": spec["output_prefix"],
        "alternate_seed_or_result_retry_allowed": False,
        "deterministic_recompute_allowed": True,
        "deterministic_recompute_mode": spot.ROOT_REMATERIALIZATION_MODE,
    }
    (shard_dir / "global_claim.json").write_bytes(
        spot.canonical_json_bytes(global_claim)
    )
    (shard_dir / "root_claim.json").write_bytes(spot.canonical_json_bytes(root_claim))
    artifact_hashes = {
        "output_sha256": output_sha256,
        "checkpoint_sha256": hashlib.sha256(
            (shard_dir / "checkpoint.json").read_bytes()
        ).hexdigest(),
        "heartbeat_sha256": hashlib.sha256(
            (shard_dir / "heartbeat.json").read_bytes()
        ).hexdigest(),
        "generator_summary_sha256": hashlib.sha256(
            (shard_dir / "generator_summary.json").read_bytes()
        ).hexdigest(),
        "run_log_sha256": hashlib.sha256((shard_dir / "run.log").read_bytes()).hexdigest(),
        "global_claim_sha256": hashlib.sha256(
            (shard_dir / "global_claim.json").read_bytes()
        ).hexdigest(),
        "root_claim_sha256": hashlib.sha256(
            (shard_dir / "root_claim.json").read_bytes()
        ).hexdigest(),
    }
    done = {
        "schema": spot.DONE_SCHEMA,
        "status": "complete",
        "run_name": manifest["run_name"],
        "run_id": f"{manifest['run_name']}:shard=0",
        "shard": 0,
        "root_index": 0,
        "root_profile": spec["root_profile"],
        "seeds": spec["seeds"],
        "output_prefix": spec["output_prefix"],
        "manifest_sha256": manifest_sha256,
        "source_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "status_sha256": manifest["status_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "authorization_sha256": authorization_sha256,
        **artifact_hashes,
        "config_sha256": config_sha256,
        "elapsed_seconds": 1.0,
        "peak_rss_bytes": 1,
        "native_batch_threads": 4,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    assert set(done) == spot._DONE_KEYS
    (shard_dir / "DONE.json").write_bytes(spot.canonical_json_bytes(done))
    received = spot.validate_received_attempt07_shard(
        shard_dir=shard_dir,
        schedule_path=closure / "shards_manifest.jsonl",
        manifest_path=closure / "manifest.json",
        output=shard_dir / "received_audit.json",
    )
    assert received["status"] == "pass"
    assert received["root_index"] == 0

    (closure / "configs").mkdir()
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        spot.validate_attempt07_spot_authorization(
            authorization_path=fixture["authorization"],
            manifest_path=closure / "manifest.json",
            preflight_aggregate_path=closure / "attempt07_preflight_aggregate.json",
        )
    (closure / "configs").rmdir()
    for name in (
        "hu_joint_policy_m43_attempt07_preflight.json",
        "source_model_manifest.json",
        "source_native_manifest.json",
    ):
        path = closure / name
        held = path.with_suffix(path.suffix + ".held")
        path.rename(held)
        with pytest.raises(ValueError, match="unknown or ambiguous"):
            spot.validate_attempt07_spot_authorization(
                authorization_path=fixture["authorization"],
                manifest_path=closure / "manifest.json",
                preflight_aggregate_path=closure
                / "attempt07_preflight_aggregate.json",
            )
        held.rename(path)


def test_start_package_only_returns_before_every_remote_or_instance_call() -> None:
    text = _read(START)
    package_comment = text.index("# PackageOnly terminates")
    package_branch = text.index("if ($PackageOnly) {", package_comment)
    package_return = text.index("return", package_branch)
    first_remote_helper = text.index("function Publish-M43A7ImmutableObject")
    first_gcloud_process = text.index("Invoke-M43A4GcloudProcess", package_return)
    first_remote_probe = text.index("Test-M43A4GcsObject", package_return)

    assert package_branch < package_return < first_remote_helper
    assert package_return < first_gcloud_process
    assert package_return < first_remote_probe
    prefix = text[:package_return]
    assert "Invoke-M43A4GcloudProcess" not in prefix
    assert "Invoke-M43A4Gcloud `" not in prefix
    assert "Test-M43A4GcsObject" not in prefix
    for token in (
        "packaged_without_root_or_gcloud",
        "fresh_root_opened -ne $false",
        "teacher_executed -ne $false",
        "gcloud_invoked -ne $false",
        "instances_created -ne $false",
        "current_profile_mutated -ne $false",
    ):
        assert token in text[:package_return]


def test_start_freezes_bounded_spot_launch_and_exact_authorization() -> None:
    text = _read(START)
    expected_authorization_keys = frozenset(
        {
            "schema",
            "status",
            "spot_authorized",
            "attempt07_plan_sha256",
            "preflight_plan_sha256",
            "preflight_aggregate_sha256",
            "preflight_manifest_sha256",
            "preflight_schedule_sha256",
            "preflight_launch_authorization_sha256",
            "done_sha256",
            "development_run_name",
            "development_manifest_sha256",
            "development_schedule_sha256",
            "development_source_closure_sha256",
            "development_source_zip_sha256",
            "development_startup_sha256",
            "development_status_sha256",
            "development_source_model_manifest_sha256",
            "development_source_native_manifest_sha256",
            "development_total_roots",
            "development_total_shards",
            "development_roots_per_shard",
            "development_native_batch_threads",
            "development_max_wave_shards",
            "development_machine_type",
            "development_root_profile_assignment",
            "development_batch_child_selectors",
            "development_package_frozen_before_authorization",
            "operational_metrics",
            "operational_gates",
            "all_gates_passed",
            "development_started",
            "fresh_development_root_opened",
            "current_profile_mutated",
            "runtime_policy_activated",
        }
    )
    assert spot.AUTHORIZATION_SCHEMA == (
        "hu_m43_attempt07_development_spot_authorization_v1"
    )
    assert spot._AUTHORIZATION_KEYS == expected_authorization_keys

    for token in (
        "$TotalShards = 100",
        "$MaxWaveShards = 50",
        "$NativeBatchThreads = 4",
        "One launch wave may contain at most 50 shards",
        "Shard index outside 0..99",
        "$selectedShards = @(Expand-M43A7ShardSelection -Values $StartShards)",
        "if ($MachineType -ne 'c4-standard-4')",
        "--machine-type',$MachineType,'--provisioning-model=SPOT'",
        "--instance-termination-action=DELETE",
        "NATIVE_BATCH_THREADS=$NativeBatchThreads",
        "validate-authorization",
        "Compare-Object $authorizationProperties $expectedAuthorizationProperties",
        "hu_m43_attempt07_development_spot_authorization_v1",
        "authorized_after_attempt07_preflight",
        "authorization.preflight_aggregate_sha256 -ne [string]$manifest.preflight_aggregate_sha256",
        "authorization does not bind exactly five preflight DONE files",
        "--if-generation-match=0",
    ):
        assert token in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_start_accepts_arbitrary_subset_but_rejects_more_than_fifty() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(START)
    function_start = text.index("function Expand-M43A7ShardSelection {")
    function_end = text.index("function ConvertTo-M43A7VmPrefix {", function_start)
    function_source = text[function_start:function_end]
    command = (
        function_source
        + ";$TotalShards=100;$MaxWaveShards=50;"
        + "$picked=@(Expand-M43A7ShardSelection -Values @('99,0,17-19','42'));"
        + "if(($picked -join ',') -ne '0,17,18,19,42,99'){"
        + "throw 'arbitrary shard subset changed'};"
        + "$rejected=$false;try{"
        + "[void](Expand-M43A7ShardSelection -Values @('0-50'))"
        + "}catch{$rejected=$true};"
        + "if(-not $rejected){throw '51-shard wave was accepted'}"
    )
    result = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_startup_checks_done_then_claims_then_runs_and_commits_resume() -> None:
    text = _read(STARTUP)
    done_check = text.index('if [[ "$(gcs_state "$DONE_URI")" == present ]]; then')
    global_claim = text.index(
        'claim_once /tmp/global_claim.json "$GLOBAL_CLAIM_URI"'
    )
    root_claim = text.index('claim_once /tmp/root_claim.json "$ROOT_CLAIM_URI"')
    commit_check = text.index(
        'if [[ "$(gcs_state "$COMMIT_URI")" == present ]]; then'
    )
    runner = text.index(
        "python -B -m ofc_regular.run_hu_m43_attempt07_development"
    )
    assert done_check < global_claim < root_claim < commit_check < runner

    resume_output_read = text.index(
        'gcloud storage cp "$RESUME_URI/objects/$RESUME_OUTPUT_SHA/teacher.jsonl"'
    )
    resume_checkpoint_read = text.index(
        'gcloud storage cp "$RESUME_URI/objects/$RESUME_CHECKPOINT_SHA/checkpoint.json"'
    )
    resume_output_publish = text.index(
        'upload_once_or_verify "$RESULT/teacher.jsonl" '
        '"$RESUME_URI/objects/$OUTPUT_SHA256/teacher.jsonl"'
    )
    resume_checkpoint_publish = text.index(
        'upload_once_or_verify "$RESULT/checkpoint.json" '
        '"$RESUME_URI/objects/$CHECKPOINT_SHA256/checkpoint.json"'
    )
    resume_commit_publish = text.index(
        'upload_once_or_verify /tmp/resume_COMMIT_new.json "$COMMIT_URI"'
    )
    assert commit_check < resume_output_read < runner
    assert commit_check < resume_checkpoint_read < runner
    assert runner < resume_output_publish < resume_commit_publish
    assert runner < resume_checkpoint_publish < resume_commit_publish

    for token in (
        "--if-generation-match=0",
        "cmp -s \"$marker\" \"$existing\"",
        "same_root_index_same_six_seeds_same_frozen_closure_after_matching_claim",
        "alternate_seed_or_result_retry_allowed",
        "--root-index \"$ROOT_INDEX\"",
        "--batch-child-selectors --native-batch-threads 4",
        "trap cleanup EXIT",
        'if [[ "$SELF_DELETE" == 1 ]]',
        'gcloud compute instances delete "$INSTANCE_NAME"',
    ):
        assert token in text


def test_startup_publishes_done_last_after_every_result_artifact() -> None:
    text = _read(STARTUP)
    done_publish_token = 'upload_once_or_verify "$RESULT/DONE.json" "$DONE_URI"'
    done_publish = text.rindex(done_publish_token)
    artifact_tokens = (
        'upload_once_or_verify "$RESULT/teacher.jsonl" "$RESULT_URI/teacher.jsonl"',
        'upload_once_or_verify "$RESULT/checkpoint.json" "$RESULT_URI/checkpoint.json"',
        'upload_once_or_verify "$RESULT/heartbeat.json" "$RESULT_URI/heartbeat.json"',
        'upload_once_or_verify "$RESULT/generator_summary.json" "$RESULT_URI/generator_summary.json"',
        'upload_once_or_verify "$RESULT/run.log" "$RESULT_URI/run.log"',
        'upload_once_or_verify "$RESULT/authorization.json" "$RESULT_URI/authorization.json"',
        'upload_once_or_verify "$RESULT/global_claim.json" "$RESULT_URI/global_claim.json"',
        'upload_once_or_verify "$RESULT/root_claim.json" "$RESULT_URI/root_claim.json"',
    )
    assert all(text.index(token) < done_publish for token in artifact_tokens)
    assert text.count(done_publish_token) == 1
    assert "upload_once_or_verify" not in text[done_publish + len(done_publish_token) :]
    assert "DONE is the final create-only publication" in text


def test_status_is_strictly_done_only_and_never_addresses_teacher_content() -> None:
    text = _read(STATUS)

    assert '"$prefix/results/$outputPrefix/DONE.json"' in text
    assert "teacher.jsonl" not in text
    assert "storage', 'ls'" not in text
    assert "/results/**" not in text
    assert "$selected.Count -eq 100" in text
    assert "$doneRows.Count -eq 100" in text
    assert "all_100_done" in text
    assert "result_content_addressed = $false" in text
    assert "selector_executed = $false" in text


def test_receive_claims_all_100_done_before_first_teacher_and_keeps_selector_separate() -> None:
    text = _read(RECEIVE)
    schedule_100 = text.index("if ($specs.Count -ne 100)")
    phase_one = text.index("# Phase 1 is deliberately DONE-only")
    done_uri = text.index('$doneUri = "$prefix/results/', phase_one)
    local_claim = text.index("'claim-complete-output'", done_uri)
    remote_claim = text.index("Claim-M43A7RemoteOutputOnce", local_claim)
    first_teacher = text.index("teacher.jsonl")

    assert schedule_100 < phase_one < done_uri < local_claim < remote_claim
    assert remote_claim < first_teacher
    assert "all_done_markers_verified -ne $true" in text[local_claim:first_teacher]
    assert "[int]$claim.expected_shards -ne 100" in text[local_claim:first_teacher]
    assert "--if-generation-match=0" in text
    assert "result_objects_addressed_when_claimed -ne $false" in text

    selector = "ofc_regular.select_hu_m43_attempt07_development_arm"
    selector_positions = []
    offset = 0
    while True:
        position = text.find(selector, offset)
        if position < 0:
            break
        selector_positions.append(position)
        offset = position + len(selector)
    merge = text.index("$mergeRaw = Invoke-M43A7ReceivePython")
    assert len(selector_positions) == 2
    assert all(position > merge for position in selector_positions)
    assert "'-B','-m','ofc_regular.select_hu_m43_attempt07_development_arm'" not in text
    assert "complete_no_selection" in text
    assert "selector_executed = $true" not in text
    assert "selector_executed = $false" in text
    assert "selector_command_required" in text
    assert "selector_command =" in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt07_development_powershell_scripts_parse() -> None:
    shell = _powershell()
    assert shell is not None
    paths = ",".join(
        f"'{str(path).replace(chr(39), chr(39) * 2)}'"
        for path in (START, STATUS, RECEIVE)
    )
    command = (
        f"$files=@({paths});"
        "foreach($f in $files){$tokens=$null;$errors=$null;"
        "[void][Management.Automation.Language.Parser]::ParseFile("
        "$f,[ref]$tokens,[ref]$errors);"
        "if(@($errors).Count){throw (($errors|ForEach-Object{$_.ToString()})"
        "-join [Environment]::NewLine)}}"
    )
    result = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="native POSIX bash unavailable",
)
def test_attempt07_development_startup_shell_parses() -> None:
    result = subprocess.run(
        [shutil.which("bash") or "bash", "-n", str(STARTUP)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
