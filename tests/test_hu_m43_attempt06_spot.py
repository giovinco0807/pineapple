from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

import ofc_regular.ai_profiles as ai_profiles
import ofc_regular.generate_hu_m4_t1_data as root_generation
import ofc_regular.hu_m43_attempt06_spot as spot
import ofc_regular.play_ai as play_ai
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt06_contract import load_and_validate_attempt06_plan
from ofc_regular.state import Board


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt06TeacherRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt06TeacherRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt06TeacherRun.ps1"
STARTUP = ROOT / "scripts" / "startup_hu_m43_attempt06_teacher.sh"
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt06.json"
TEST_SOURCE_SHA256 = "c" * 64
TEST_STARTUP_SHA256 = "d" * 64
TEST_STATUS_SHA256 = "e" * 64
TEST_SOURCE_CLOSURE_SHA256 = "f" * 64


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def _fixture_root() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _claims(tmp_path: Path, *, root_index: int = 0) -> tuple[Path, Path, dict]:
    schedule = spot.build_attempt06_spot_schedule(
        load_and_validate_attempt06_plan(PLAN)
    )
    spec = schedule[root_index]
    common = {
        "run_name": "attempt06-test-run",
        "manifest_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "plan_sha256": spot.M43_ATTEMPT06_PLAN_SHA256,
        "model_sha256": spot.M43_ATTEMPT06_LAMBDA_SHA256,
        "source_sha256": TEST_SOURCE_SHA256,
        "startup_sha256": TEST_STARTUP_SHA256,
        "status_sha256": TEST_STATUS_SHA256,
        "source_closure_sha256": TEST_SOURCE_CLOSURE_SHA256,
        "source_model_manifest_sha256": spot.PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": spot.PINNED_NATIVE_MANIFEST_SHA256,
        "native_batch_threads": spot.ATTEMPT06_NATIVE_BATCH_THREADS,
    }
    global_marker = tmp_path / "global.json"
    global_marker.write_text(
        json.dumps(
            {
                "schema": "hu_m43_attempt06_global_consumption_marker_v1",
                "status": "consumed_before_any_fresh_root_content_read",
                **common,
                "roots": 50,
                "shards": 50,
                "roots_per_shard": 1,
            }
        ),
        encoding="utf-8",
    )
    root_claim = tmp_path / "claim.json"
    root_claim.write_text(
        json.dumps(
            {
                "schema": "hu_m43_attempt06_root_consumption_claim_v1",
                "status": "claimed_before_materializing_policy_observation",
                **common,
                "run_id": f"attempt06-test-run:shard={root_index}",
                "root_index": root_index,
                "hand_seed": spec["hand_seed"],
                "root_profile": spec["root_profile"],
                "retry_same_seed_after_open_allowed": False,
                "fresh_audit_retry_or_alternate_sample_allowed": False,
                "deterministic_claim_recovery_allowed": True,
                "deterministic_claim_recovery_mode": (
                    spot.ROOT_REMATERIALIZATION_MODE
                ),
            }
        ),
        encoding="utf-8",
    )
    return global_marker, root_claim, spec


def _label_sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_canonical(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot.canonical_json_bytes(payload))


def _received_shard_fixture(
    tmp_path: Path,
    *,
    local_run: str = "attempt06-synthetic",
    global_run: str | None = None,
    claim_run: str | None = None,
    done_run: str | None = None,
    row_run: str | None = None,
    schedule_seed_field: str | None = None,
    config_sha_component: str | None = None,
) -> tuple[Path, Path, Path]:
    """Build a card-static canonical receive fixture; no audit seed is dealt."""

    global_run = global_run or local_run
    claim_run = claim_run or local_run
    done_run = done_run or local_run
    row_run = row_run or local_run
    plan = load_and_validate_attempt06_plan(PLAN)
    schedule = spot.build_attempt06_spot_schedule(plan)
    if schedule_seed_field is not None:
        schedule[0][schedule_seed_field] = 999
    schedule_path = tmp_path / "shards_manifest.jsonl"
    schedule_path.write_bytes(
        b"".join(spot.canonical_json_bytes(row) for row in schedule)
    )
    schedule_sha = spot.sha256_file(schedule_path)
    spec = schedule[0]
    manifest_sha = "a" * 64
    source_sha = TEST_SOURCE_SHA256
    startup_sha = TEST_STARTUP_SHA256
    status_sha = TEST_STATUS_SHA256
    closure_sha = TEST_SOURCE_CLOSURE_SHA256
    closure = {
        "manifest_sha256": manifest_sha,
        "source_sha256": source_sha,
        "startup_sha256": startup_sha,
        "schedule_sha256": schedule_sha,
        "plan_sha256": spot.M43_ATTEMPT06_PLAN_SHA256,
        "status_sha256": status_sha,
        "model_sha256": spot.M43_ATTEMPT06_LAMBDA_SHA256,
        "source_closure_sha256": closure_sha,
        "source_model_manifest_sha256": spot.PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": spot.PINNED_NATIVE_MANIFEST_SHA256,
        "native_batch_threads": spot.ATTEMPT06_NATIVE_BATCH_THREADS,
    }
    shard_dir = tmp_path / "shard_000"
    global_path = shard_dir / "global_consumption_marker.json"
    _write_canonical(
        global_path,
        {
            "schema": "hu_m43_attempt06_global_consumption_marker_v1",
            "status": "consumed_before_any_fresh_root_content_read",
            "run_name": global_run,
            **closure,
            "roots": 50,
            "shards": 50,
            "roots_per_shard": 1,
            "fresh_200_fit_authorized": False,
            "runtime_policy_activated": False,
        },
    )
    global_sha = spot.sha256_file(global_path)
    claim_path = shard_dir / "root_claim.json"
    _write_canonical(
        claim_path,
        {
            "schema": "hu_m43_attempt06_root_consumption_claim_v1",
            "status": "claimed_before_materializing_policy_observation",
            "run_name": claim_run,
            "run_id": f"{claim_run}:shard=0",
            **closure,
            "root_index": 0,
            "hand_seed": spec["hand_seed"],
            "root_profile": spec["root_profile"],
            "retry_same_seed_after_open_allowed": False,
            "fresh_audit_retry_or_alternate_sample_allowed": False,
            "deterministic_claim_recovery_allowed": True,
            "deterministic_claim_recovery_mode": spot.ROOT_REMATERIALIZATION_MODE,
        },
    )
    claim_sha = spot.sha256_file(claim_path)

    namespace = runpy.run_path(
        str(ROOT / "tests" / "test_audit_hu_m43_attempt06_search_quality.py")
    )
    row = namespace["_row"](0, fired=True, mean=2.0)
    provenance = dict(row["provenance"])
    provenance.update(
        {
            "run_name": row_run,
            "run_id": f"{row_run}:shard=0",
            "schedule_sha256": schedule_sha,
            "package_manifest_sha256": manifest_sha,
            "global_consumption_marker_sha256": global_sha,
            "root_consumption_claim_sha256": claim_sha,
            "source_sha256": source_sha,
            "startup_sha256": startup_sha,
            "status_sha256": status_sha,
            "source_closure_sha256": closure_sha,
            "native_batch_threads": spot.ATTEMPT06_NATIVE_BATCH_THREADS,
        }
    )
    for key in ("input_sha256", "config_sha256", "model_sha256"):
        provenance.pop(key, None)
    root_path = shard_dir / "root.jsonl"
    _write_canonical(
        root_path,
        {
            "schema": spot.ATTEMPT06_ROOT_SCHEMA,
            "root_index": 0,
            "hand_seed": spot.ATTEMPT06_HAND_SEED_START,
            "root_profile": spot.M43_ATTEMPT06_PROFILES[0],
            "policy_observation": row["policy_observation"],
            "baseline_action_key": row["baseline_action_key"],
            "provenance": provenance,
        },
    )
    input_sha = spot.sha256_file(root_path)
    config_sha = spot.attempt06_fixed_contract_sha256(
        root_index=0,
        input_sha256=input_sha,
        model_sha256=spot.M43_ATTEMPT06_LAMBDA_SHA256,
        source_model_manifest_sha256=spot.PINNED_MODEL_MANIFEST_SHA256,
        source_native_manifest_sha256=spot.PINNED_NATIVE_MANIFEST_SHA256,
        run_id=f"{local_run}:shard=0",
        batch_child_selectors=True,
        native_batch_threads=spot.ATTEMPT06_NATIVE_BATCH_THREADS,
    )

    def component_config_sha(component: str) -> str:
        if config_sha_component == component:
            return _label_sha(f"wrong-{component}-config")
        return config_sha

    row["provenance"] = {
        **provenance,
        "input_sha256": input_sha,
        "config_sha256": component_config_sha("row"),
        "model_sha256": spot.M43_ATTEMPT06_LAMBDA_SHA256,
    }
    observation = ActorObservation.from_dict(row["policy_observation"])
    row["teacher"]["search_config"]["run_id"] = (
        f"{row_run}:shard=0:root=0:seed={spot.ATTEMPT06_HAND_SEED_START}:"
        f"obs={observation.fingerprint()}"
    )
    teacher_path = shard_dir / "teacher.jsonl"
    _write_canonical(teacher_path, row)
    output_sha = spot.sha256_file(teacher_path)

    checkpoint_path = shard_dir / "checkpoint.json"
    _write_canonical(
        checkpoint_path,
        {
            "schema": spot.ATTEMPT06_CHECKPOINT_SCHEMA,
            "config_sha256": component_config_sha("checkpoint"),
            "input_sha256": input_sha,
            "model_sha256": spot.M43_ATTEMPT06_LAMBDA_SHA256,
            "completed_roots": 1,
            "target_roots": 1,
            "root_index": 0,
            "source_model_manifest_sha256": spot.PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": spot.PINNED_NATIVE_MANIFEST_SHA256,
        },
    )
    heartbeat_path = shard_dir / "heartbeat.json"
    _write_canonical(
        heartbeat_path,
        {
            "schema": spot.ATTEMPT06_HEARTBEAT_SCHEMA,
            "status": "complete",
            "config_sha256": component_config_sha("heartbeat"),
            "input_sha256": input_sha,
            "model_sha256": spot.M43_ATTEMPT06_LAMBDA_SHA256,
            "root_index": 0,
            "output_sha256": output_sha,
            "source_model_manifest_sha256": spot.PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": spot.PINNED_NATIVE_MANIFEST_SHA256,
        },
    )
    summary_path = shard_dir / "generator_summary.json"
    _write_canonical(
        summary_path,
        {
            "schema": spot.ATTEMPT06_SHARD_SUMMARY_SCHEMA,
            "status": "complete",
            "config_sha256": component_config_sha("summary"),
            "root_index": 0,
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "model_sha256": spot.M43_ATTEMPT06_LAMBDA_SHA256,
            "source_model_manifest_sha256": spot.PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": spot.PINNED_NATIVE_MANIFEST_SHA256,
        },
    )
    run_log_path = shard_dir / "run.log"
    run_log_path.write_text("synthetic receive fixture\n", encoding="utf-8")
    done_path = shard_dir / "DONE.json"
    _write_canonical(
        done_path,
        {
            "schema": spot.DONE_SCHEMA,
            "status": "complete",
            "run_name": done_run,
            "run_id": f"{done_run}:shard=0",
            "shard": 0,
            "root_index": 0,
            "hand_seed": spec["hand_seed"],
            "root_profile": spec["root_profile"],
            "roots": 1,
            "output_prefix": spec["output_prefix"],
            **closure,
            "config_sha256": component_config_sha("done"),
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "checkpoint_sha256": spot.sha256_file(checkpoint_path),
            "heartbeat_sha256": spot.sha256_file(heartbeat_path),
            "generator_summary_sha256": spot.sha256_file(summary_path),
            "run_log_sha256": spot.sha256_file(run_log_path),
            "global_consumption_marker_sha256": global_sha,
            "root_consumption_claim_sha256": claim_sha,
            "teacher_values_are_realized_match_ev": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    marker_path = tmp_path / "M43_ATTEMPT06_AUDIT_CONSUMED.json"
    _write_canonical(
        marker_path,
        {
            "schema": spot.AUDIT_OUTPUT_CONSUMPTION_SCHEMA,
            "status": "consumed_before_any_result_teacher_or_root_read",
            "run_name": local_run,
            "authorization_file_sha256": _label_sha("authorization"),
            "manifest_sha256": manifest_sha,
            "schedule_sha256": schedule_sha,
            "global_consumption_marker_sha256": global_sha,
            "expected_shards": 50,
            "expected_roots": 50,
            "result_objects_addressed_when_claimed": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    return shard_dir, schedule_path, marker_path


def test_schedule_is_exactly_fifty_one_root_shards_and_mod5_profiles() -> None:
    plan = load_and_validate_attempt06_plan(PLAN)
    rows = spot.build_attempt06_spot_schedule(plan)
    assert len(rows) == 50
    for index, row in enumerate(rows):
        assert row["schema"] == spot.SHARD_SCHEMA
        assert row["shard"] == index == row["root_index"]
        assert row["roots"] == 1
        assert row["root_profile"] == spot.M43_ATTEMPT06_PROFILES[index % 5]
        assert row["hand_seed"] == 17_306_071_901 + 1_000_003 * index
        assert row["candidate_seed"] == 23_306_071_901 + 1_000_003 * index
        assert row["evaluation_seed"] == 24_306_071_901 + 1_000_003 * index
        assert row["child_policy_seed"] == 25_306_071_901 + 1_000_003 * index
        assert (row["candidate_samples"], row["evaluation_samples"]) == (8, 128)
        assert row["native_batch_threads"] == 4
        assert row["learned_nonbaseline_top_k"] == 8
    assert {profile: sum(row["root_profile"] == profile for row in rows) for profile in spot.M43_ATTEMPT06_PROFILES} == {
        profile: 10 for profile in spot.M43_ATTEMPT06_PROFILES
    }


def test_receive_accepts_only_fully_bound_canonical_v2_fixture(tmp_path: Path) -> None:
    shard_dir, schedule, marker = _received_shard_fixture(tmp_path)
    output = tmp_path / "received_audit.json"
    result = spot.validate_received_attempt06_shard(
        shard_dir=shard_dir,
        schedule_path=schedule,
        shard=0,
        consumption_marker=marker,
        output=output,
    )
    assert result["status"] == "pass_after_local_consumption_claim"
    assert output.exists()


@pytest.mark.parametrize(
    "component", ("done", "row", "checkpoint", "heartbeat", "summary")
)
def test_receive_rejects_config_sha_mismatch_in_every_bound_artifact(
    tmp_path: Path, component: str
) -> None:
    shard_dir, schedule, marker = _received_shard_fixture(
        tmp_path, config_sha_component=component
    )
    with pytest.raises(ValueError, match="config|provenance|checkpoint|heartbeat|summary"):
        spot.validate_received_attempt06_shard(
            shard_dir=shard_dir,
            schedule_path=schedule,
            shard=0,
            consumption_marker=marker,
            output=tmp_path / "received_audit.json",
        )


@pytest.mark.parametrize("component", ("global", "claim", "done", "row"))
def test_receive_rejects_cross_run_identity_even_when_each_object_is_self_consistent(
    tmp_path: Path, component: str
) -> None:
    kwargs = {f"{component}_run": f"attempt06-{component}-other"}
    shard_dir, schedule, marker = _received_shard_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        spot.validate_received_attempt06_shard(
            shard_dir=shard_dir,
            schedule_path=schedule,
            shard=0,
            consumption_marker=marker,
            output=tmp_path / "received_audit.json",
        )


@pytest.mark.parametrize(
    "seed_field",
    ("hand_seed", "candidate_seed", "evaluation_seed", "child_policy_seed"),
)
def test_receive_rejects_schedule_seed_999_before_accepting_shard(
    tmp_path: Path, seed_field: str
) -> None:
    shard_dir, schedule, marker = _received_shard_fixture(
        tmp_path, schedule_seed_field=seed_field
    )
    with pytest.raises(ValueError, match="schedule identity changed"):
        spot.validate_received_attempt06_shard(
            shard_dir=shard_dir,
            schedule_path=schedule,
            shard=0,
            consumption_marker=marker,
            output=tmp_path / "received_audit.json",
        )


def test_package_path_cannot_call_fresh_root_builder() -> None:
    source = inspect.getsource(spot.package_attempt06_spot)
    assert "build_attempt06_root_input(" not in source
    assert "generate_t1_second_root(" not in source
    assert '"fresh_seed_content_opened": False' in source
    assert '"teacher_executed": False' in source
    assert '"native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS' in source


def test_package_rejects_recursive_source_destination_before_any_copy(
    tmp_path: Path,
) -> None:
    del tmp_path
    run_name = "attempt06-recursive-self-copy-test"
    unsafe = ROOT / "src" / "ofc_regular" / run_name
    with pytest.raises(ValueError, match="exactly"):
        spot.package_attempt06_spot(
            repo_root=ROOT,
            run_dir=unsafe,
            run_name=run_name,
            plan_path=ROOT / "missing-plan.json",
            status_path=ROOT / "missing-status.json",
            model_path=ROOT / "missing-model.pkl",
            startup_path=ROOT / "missing-startup.sh",
        )
    assert not unsafe.exists()
    assert not unsafe.with_name(unsafe.name + ".building").exists()


def test_root_builder_fails_before_deal_without_claim_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("fresh audit seed was dealt before claim validation")

    monkeypatch.setattr(root_generation, "generate_t1_second_root", forbidden)
    with pytest.raises(FileNotFoundError):
        spot.build_attempt06_root_input(
            output=tmp_path / "root.jsonl",
            root_index=0,
            hand_seed=17_306_071_901,
            root_profile="stage19_p0",
            plan_sha256=spot.M43_ATTEMPT06_PLAN_SHA256,
            schedule_sha256="b" * 64,
            model_sha256=spot.M43_ATTEMPT06_LAMBDA_SHA256,
            manifest_sha256="a" * 64,
            source_sha256=TEST_SOURCE_SHA256,
            startup_sha256=TEST_STARTUP_SHA256,
            status_sha256=TEST_STATUS_SHA256,
            source_closure_sha256=TEST_SOURCE_CLOSURE_SHA256,
            global_marker=tmp_path / "missing-global.json",
            root_claim=tmp_path / "missing-claim.json",
            run_name="attempt06-test-run",
            run_id="attempt06-test-run:shard=0",
        )
    assert opened is False
    assert not (tmp_path / "root.jsonl").exists()


def test_root_builder_uses_claimed_fixture_without_dealing_audit_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_marker, root_claim, spec = _claims(tmp_path)
    observation = _fixture_root()
    calls: list[tuple[int, object]] = []
    monkeypatch.setattr(ai_profiles, "load_model_bundle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        ai_profiles,
        "build_policy",
        lambda profile, _bundle, **kwargs: {"profile": profile, **kwargs},
    )

    def fixture_only(seed, *, root_policies):
        # The frozen audit seed is never dealt.  This is an in-memory synthetic
        # infoset used only to verify claim/order/serialization wiring.
        calls.append((seed, root_policies))
        return observation

    monkeypatch.setattr(root_generation, "generate_t1_second_root", fixture_only)
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    monkeypatch.setattr(
        play_ai, "_choose_from_observation", lambda *_args, **_kwargs: actions[-1]
    )
    output = tmp_path / "root.jsonl"
    result = spot.build_attempt06_root_input(
        output=output,
        root_index=0,
        hand_seed=spec["hand_seed"],
        root_profile=spec["root_profile"],
        plan_sha256=spot.M43_ATTEMPT06_PLAN_SHA256,
        schedule_sha256="b" * 64,
        model_sha256=spot.M43_ATTEMPT06_LAMBDA_SHA256,
        manifest_sha256="a" * 64,
        source_sha256=TEST_SOURCE_SHA256,
        startup_sha256=TEST_STARTUP_SHA256,
        status_sha256=TEST_STATUS_SHA256,
        source_closure_sha256=TEST_SOURCE_CLOSURE_SHA256,
        global_marker=global_marker,
        root_claim=root_claim,
        run_name="attempt06-test-run",
        run_id="attempt06-test-run:shard=0",
    )
    assert len(calls) == 1
    assert result["status"] == "fresh_root_materialized_after_external_claim"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "root_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
    }
    assert payload["provenance"]["root_profile"] == "stage19_p0"
    assert payload["provenance"]["baseline_profile"] == "stage18_p1"
    assert payload["provenance"]["current_profile_resolved"] is False
    assert payload["provenance"]["deterministic_claim_recovery_allowed"] is True
    assert (
        payload["provenance"]["fresh_audit_retry_or_alternate_sample_allowed"]
        is False
    )
    assert "opponent_private_discards" not in json.dumps(payload, sort_keys=True)


def test_matching_existing_claim_can_only_rematerialize_identical_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic crash recovery never deals the frozen audit seed in this test."""

    global_marker, root_claim, spec = _claims(tmp_path)
    observation = _fixture_root()
    calls: list[int] = []
    monkeypatch.setattr(ai_profiles, "load_model_bundle", lambda *_a, **_k: object())
    monkeypatch.setattr(
        ai_profiles,
        "build_policy",
        lambda profile, _bundle, **kwargs: {"profile": profile, **kwargs},
    )

    def fixture_only(seed, *, root_policies):
        del root_policies
        calls.append(seed)
        return observation

    monkeypatch.setattr(root_generation, "generate_t1_second_root", fixture_only)
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    monkeypatch.setattr(
        play_ai, "_choose_from_observation", lambda *_args, **_kwargs: actions[-1]
    )

    def materialize(name: str) -> Path:
        output = tmp_path / name
        spot.build_attempt06_root_input(
            output=output,
            root_index=0,
            hand_seed=spec["hand_seed"],
            root_profile=spec["root_profile"],
            plan_sha256=spot.M43_ATTEMPT06_PLAN_SHA256,
            schedule_sha256="b" * 64,
            model_sha256=spot.M43_ATTEMPT06_LAMBDA_SHA256,
            manifest_sha256="a" * 64,
            source_sha256=TEST_SOURCE_SHA256,
            startup_sha256=TEST_STARTUP_SHA256,
            status_sha256=TEST_STATUS_SHA256,
            source_closure_sha256=TEST_SOURCE_CLOSURE_SHA256,
            global_marker=global_marker,
            root_claim=root_claim,
            run_name="attempt06-test-run",
            run_id="attempt06-test-run:shard=0",
        )
        return output

    first = materialize("first.jsonl")
    recovered = materialize("recovered.jsonl")
    assert first.read_bytes() == recovered.read_bytes()
    assert calls == [spec["hand_seed"], spec["hand_seed"]]


def test_claim_closure_mismatch_fails_before_synthetic_root_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_marker, root_claim, spec = _claims(tmp_path)
    claim = json.loads(root_claim.read_text(encoding="utf-8"))
    claim["source_closure_sha256"] = "0" * 64
    root_claim.write_text(json.dumps(claim), encoding="utf-8")
    opened = False

    def forbidden(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("claim mismatch reached root generation")

    monkeypatch.setattr(root_generation, "generate_t1_second_root", forbidden)
    with pytest.raises(ValueError, match="per-root claim proof changed"):
        spot.build_attempt06_root_input(
            output=tmp_path / "root.jsonl",
            root_index=0,
            hand_seed=spec["hand_seed"],
            root_profile=spec["root_profile"],
            plan_sha256=spot.M43_ATTEMPT06_PLAN_SHA256,
            schedule_sha256="b" * 64,
            model_sha256=spot.M43_ATTEMPT06_LAMBDA_SHA256,
            manifest_sha256="a" * 64,
            source_sha256=TEST_SOURCE_SHA256,
            startup_sha256=TEST_STARTUP_SHA256,
            status_sha256=TEST_STATUS_SHA256,
            source_closure_sha256=TEST_SOURCE_CLOSURE_SHA256,
            global_marker=global_marker,
            root_claim=root_claim,
            run_name="attempt06-test-run",
            run_id="attempt06-test-run:shard=0",
        )
    assert opened is False


def test_root_index_run_id_mismatch_fails_before_synthetic_root_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_marker, root_claim, spec = _claims(tmp_path)
    claim = json.loads(root_claim.read_text(encoding="utf-8"))
    claim["run_id"] = "attempt06-test-run:shard=49"
    root_claim.write_text(json.dumps(claim), encoding="utf-8")
    opened = False

    def forbidden(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("run_id mismatch reached root generation")

    monkeypatch.setattr(root_generation, "generate_t1_second_root", forbidden)
    with pytest.raises(ValueError, match="run_name/run_id binding changed"):
        spot.build_attempt06_root_input(
            output=tmp_path / "root.jsonl",
            root_index=0,
            hand_seed=spec["hand_seed"],
            root_profile=spec["root_profile"],
            plan_sha256=spot.M43_ATTEMPT06_PLAN_SHA256,
            schedule_sha256="b" * 64,
            model_sha256=spot.M43_ATTEMPT06_LAMBDA_SHA256,
            manifest_sha256="a" * 64,
            source_sha256=TEST_SOURCE_SHA256,
            startup_sha256=TEST_STARTUP_SHA256,
            status_sha256=TEST_STATUS_SHA256,
            source_closure_sha256=TEST_SOURCE_CLOSURE_SHA256,
            global_marker=global_marker,
            root_claim=root_claim,
            run_name="attempt06-test-run",
            run_id="attempt06-test-run:shard=49",
        )
    assert opened is False


def test_scripts_preserve_profile_registry_and_never_register_runtime() -> None:
    assert hashlib.sha256(
        (ROOT / "src/ofc_regular/ai_profiles.py").read_bytes()
    ).hexdigest() == "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    for path in (START, STATUS, RECEIVE, STARTUP):
        text = _read(path)
        assert "build_policy_profile" not in text
        assert "runtime_policy_activated" in text
        assert "current_profile_mutated" in text or path == STARTUP


def test_start_package_only_precedes_every_gcloud_and_opens_no_seed() -> None:
    text = _read(START)
    package_branch = text.index("if ($PackageOnly) {")
    remote_prefix = text.index('$prefix = "gs://$Bucket/runs/$RunName"')
    assert package_branch < remote_prefix
    branch = text[package_branch:remote_prefix]
    assert "gcloud" in branch  # explicit false result fields only
    assert "Invoke-M43A4Gcloud" not in branch
    assert "Test-M43A4GcsObject" not in branch
    for token in (
        "pass_no_fresh_content_no_gcloud",
        "fresh_seed_content_opened = $false",
        "teacher_executed = $false",
        "instances_created = $false",
        "$TotalShards = 50",
        "$RootsPerShard = 1",
        "SpotAuthorizationPath",
        "RunDir must be exactly outputs/gcp_runs/<RunName>",
        "local_correctness",
        "scalar_batch_exact_parity",
        "latency_profile",
    ):
        assert token in text


def test_start_splits_local_package_resume_from_remote_run_resume() -> None:
    text = _read(START)
    assert "[switch]$ResumePackage" in text
    assert "[switch]$ResumeExisting" in text
    package_resume = text.index("if ($ResumePackage -or $ResumeExisting) {")
    package_call = text.index("Invoke-M43A6Python", package_resume)
    remote_prefix = text.index('$prefix = "gs://$Bucket/runs/$RunName"')
    remote_resume = text.index("if ($ResumeExisting) {", remote_prefix)
    first_publish = text.index("Publish-M43A6ImmutableObject", remote_resume)
    assert package_resume < package_call < remote_prefix < remote_resume < first_publish
    assert "--resume-existing" in text[package_resume:package_call]
    assert "$ResumePackage" not in text[remote_prefix:first_publish]
    assert "Test-M43A4GcsObject" in text[remote_resume:first_publish]


def test_start_always_array_wraps_single_shard_selection() -> None:
    text = _read(START)
    assert (
        "$selected = @(Expand-M43A6ShardSelection "
        "-Values $StartShards -Count $TotalShards)"
    ) in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_start_single_shard_18_has_count_one_in_powershell() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(START)
    function_start = text.index("function Expand-M43A6ShardSelection {")
    function_end = text.index("function ConvertTo-M43A6VmPrefix {", function_start)
    function_source = text[function_start:function_end]
    command = (
        function_source
        + ";$StartShards=@('18');$TotalShards=50;"
        + "$selected=@(Expand-M43A6ShardSelection "
        + "-Values $StartShards -Count $TotalShards);"
        + "if($selected.Count -ne 1 -or [int]$selected[0] -ne 18){"
        + "throw 'single-shard selection was not a one-element array'}"
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


def test_status_always_array_wraps_single_shard_selection() -> None:
    text = _read(STATUS)
    assert "$selected = @(Expand-M43A6StatusSelection -Values $Shards)" in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_status_single_shard_18_has_count_one_in_powershell() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(STATUS)
    function_start = text.index("function Expand-M43A6StatusSelection {")
    function_end = text.index("function Read-M43A6RemoteJsonExact {", function_start)
    function_source = text[function_start:function_end]
    command = (
        function_source
        + ";$Shards=@('18');"
        + "$selected=@(Expand-M43A6StatusSelection -Values $Shards);"
        + "if($selected.Count -ne 1 -or [int]$selected[0] -ne 18){"
        + "throw 'single status shard selection was not a one-element array'}"
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


def test_startup_claims_before_root_deal_and_teacher_and_is_restart_safe() -> None:
    text = _read(STARTUP)
    global_claim = text.index('claim_once "$RESULT/global_consumption_marker.json"')
    root_claim = text.index('ROOT_CLAIM_STATE="$(claim_once')
    build_root = text.index("ofc_regular.hu_m43_attempt06_spot build-root-input")
    teacher = text.index("ofc_regular.hu_m43_attempt06_teacher shard")
    assert global_claim < root_claim < build_root < teacher
    for token in (
        '--global-marker "$RESULT/global_consumption_marker.json"',
        '--root-claim "$RESULT/root_claim.json"',
        '--manifest-sha256 "$MANIFEST_SHA256"',
        '--source-sha256 "$SOURCE_SHA256"',
        '--startup-sha256 "$STARTUP_SHA256"',
        '--status-sha256 "$STATUS_SHA256"',
        '--source-closure-sha256 "$CLOSURE_SHA256"',
        '--source-model-manifest-sha256 "$SOURCE_MODEL_MANIFEST_SHA256"',
        '--source-native-manifest-sha256 "$SOURCE_NATIVE_MANIFEST_SHA256"',
        '"$ROOT_CLAIM_STATE" == owner || "$ROOT_CLAIM_STATE" == existing',
        "same_seed_same_frozen_closure_after_matching_claim_and_absent_root_only",
        "fresh_audit_retry_or_alternate_sample_allowed",
        "It is not a fresh-audit retry or an alternate sample",
        'gcloud storage objects describe "$uri"',
        '"$RESUME_URI/checkpoint.json"',
        '"$RESUME_URI/teacher.jsonl.partial"',
        "worker_heartbeat running",
        "deterministic_full_root_recompute_same_frozen_closure",
        'if [[ "$RESUME_COMPLETED" == 0 ]]',
        "checkpoint0 unexpectedly has a remote partial; refusing ambiguous resume",
        "checkpoint1 exists without partial output",
        '--if-generation-match=0',
    ):
        assert token in text
    assert "gcloud storage ls " not in text
    assert '"$SYNC/"*' not in text


def test_status_reads_only_exact_done_metadata() -> None:
    text = _read(STATUS)
    assert '"$prefix/results/$($spec.output_prefix)/DONE.json"' in text
    assert "teacher_payload_downloaded = $false" in text
    assert "root_payload_downloaded = $false" in text
    assert "storage', 'ls'" not in text
    assert "/results/**" not in text
    assert "teacher.jsonl" not in text


def test_receive_claims_before_any_result_uri_and_uses_exact_artifacts() -> None:
    text = _read(RECEIVE)
    claim = text.index("Write-M43A4Utf8CreateNew -Path $claimPath")
    remote_claim = text.index("Claim-M43A6RemoteAuditOutputOnce", claim)
    result = text.index('$resultPrefix = "$prefix/results/', claim)
    assert claim < remote_claim < result
    assert "/results/" not in text[:claim]
    assert (
        "outputs/hu_joint_policy/m43_attempt06_search_quality/"
        "M43_ATTEMPT06_AUDIT_CONSUMED.json"
    ) in text
    assert "must equal the single canonical Attempt06 marker path" in text
    assert "[IO.Path]::GetFileName($claimPath)" not in text
    assert '"$prefix/audit_output_boundary/CONSUMED.json"' in text
    assert "--if-generation-match=0" in text
    assert "already consumed on another host" in text
    assert "[int]$manifest.native_batch_threads -ne 4" in text
    assert (
        "[int]$globalMarker.native_batch_threads -ne "
        "[int]$manifest.native_batch_threads"
    ) in text
    for name in (
        "root.jsonl",
        "teacher.jsonl",
        "checkpoint.json",
        "heartbeat.json",
        "generator_summary.json",
        "run.log",
        "global_consumption_marker.json",
        "root_claim.json",
    ):
        assert f'"$resultPrefix/{name}"' in text
    assert "[ValidateRange(1, 8)][int]$MaxParallel = 8" in text
    assert "Invoke-M43A4ParallelExactGcsCopies" in text
    assert "merged_fifty_fresh_rows_without_fit_or_threshold_selection" in text
    assert "go_no_go_computed = $false" in text
    assert "storage', 'ls'" not in text
    assert "/results/**" not in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt06_powershell_scripts_parse() -> None:
    shell = _powershell()
    assert shell is not None
    paths = ",".join(
        f"'{str(path).replace(chr(39), chr(39) * 2)}'"
        for path in (START, STATUS, RECEIVE)
    )
    command = (
        f"$files=@({paths});"
        "foreach($f in $files){$tokens=$null;$errors=$null;"
        "[void][Management.Automation.Language.Parser]::ParseFile($f,[ref]$tokens,[ref]$errors);"
        "if(@($errors).Count){throw (($errors|ForEach-Object{$_.ToString()})-join [Environment]::NewLine)}}"
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
def test_attempt06_startup_shell_parses() -> None:
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
