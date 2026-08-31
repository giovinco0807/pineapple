import copy
import json
import multiprocessing
import threading
from pathlib import Path

import pytest

import ai.tutor.run_m3_behavior_calibration_pipeline as pipeline_module
from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_temperature_calibration import (
    build_temperature_gate_config,
    verify_temperature_gate_config,
)
from ai.tutor.behavior_temperature_calibration_shards import (
    build_sharded_behavior_temperature_calibration,
)
from ai.tutor.collect_hu_behavior_trace_shards import (
    _Aggregate,
    _build_top_manifest,
    collect_sharded_behavior_traces,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
)
from ai.tutor.evaluate_hu_behavior_trace_shards import (
    EXPECTED_INPUT_POLICY_ID,
    evaluate_hu_behavior_trace_shards,
)
from ai.tutor.frozen_behavior_torch import TurnActorBehaviorDispatch
from ai.tutor.run_m3_behavior_calibration_pipeline import (
    PIPELINE_RESULT_SCHEMA,
    _publish_calibration_no_replace,
    _read_existing_evaluation,
    run_m3_behavior_calibration_pipeline,
)
from test_behavior_temperature_calibration_shards import (
    CHALLENGE_ID,
    CHALLENGE_PREFIX,
    _evaluators,
)


def _hold_pipeline_writer_lock_in_child(run_root, ready, release):
    with pipeline_module._pipeline_writer_lock(Path(run_root)):
        ready.set()
        release.wait(30)


def _small_gate():
    return build_temperature_gate_config(
        gate_id="test-m3-pipeline-gate-v1",
        min_fit_decisions_per_role=1,
        min_dev_decisions_per_role=1,
        min_test_decisions_per_role=1,
        min_challenge_decisions_per_role_joker=1,
        min_roots_per_split_role=1,
        min_challenge_roots_per_role_joker=1,
        challenge_root_namespace_prefix=CHALLENGE_PREFIX,
        bootstrap_replicates=20,
        bootstrap_seed="test-m3-pipeline-bootstrap-v1",
        ece_bins=5,
    )


def _write_canonical(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


def _plan_section(collection):
    manifest = collection.manifest
    return {
        "collection_config": manifest["collection_config"],
        "range": {
            "root_index_start": manifest["root_index_start"],
            "root_index_stop_exclusive": manifest["root_index_stop_exclusive"],
            "root_count": collection.root_count,
            "shard_size_roots": manifest["shard_size"],
            "shard_count": collection.shard_count,
            "shards": [
                {
                    "shard_index": entry["index"],
                    "root_index_start": entry["root_index_start"],
                    "root_index_stop_exclusive": entry[
                        "root_index_stop_exclusive"
                    ],
                    "root_count": entry["root_count"],
                    "root_id_order_sha256": entry["root_id_order_sha256"],
                }
                for entry in manifest["shards"]
            ],
        },
    }


def _fixture(tmp_path: Path):
    run_dir = tmp_path / "run"
    evaluators = _evaluators()
    dispatch = TurnActorBehaviorDispatch(
        evaluators, model_id=EXPECTED_INPUT_POLICY_ID
    )
    natural = collect_sharded_behavior_traces(
        run_dir / "natural",
        BehaviorTraceCollectionConfig(
            seed_namespace="test-m3-pipeline-natural-v1",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
        ),
        dispatch,
        total_root_target=32,
        shard_size=16,
    )
    challenge = collect_sharded_behavior_traces(
        run_dir / "challenge",
        BehaviorTraceCollectionConfig(
            seed_namespace="test-m3-pipeline-challenge-v1",
            root_sampling_mode=TARGETED_JOKER_CHALLENGE,
            challenge_id=CHALLENGE_ID,
            challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
        ),
        dispatch,
        total_root_target=12,
        shard_size=5,
    )
    gate = _small_gate()
    plan = {
        "schema": "ofc_behavior_collection_plan/v1",
        "plan_sha256": canonical_sha256({"fixture": str(tmp_path)}),
        "temperature_gate": {
            "gate_config_sha256": gate["gate_config_sha256"],
            "config": gate,
        },
        "source_contract": {
            "calibration_contract_source_sha256": natural.manifest[
                "source_hashes"
            ]["behavior_contract_source_sha256"],
            "collector_source_sha256": natural.manifest["source_hashes"][
                "core_collector_source_sha256"
            ],
        },
        "natural": _plan_section(natural),
        "joker_challenge": _plan_section(challenge),
    }
    gate_path = tmp_path / "gate.json"
    plan_path = tmp_path / "plan.json"
    _write_canonical(gate_path, gate)
    _write_canonical(plan_path, plan)
    return run_dir, evaluators, gate, gate_path, plan, plan_path


def _install_test_plan_verifier(monkeypatch, gate, plan):
    monkeypatch.setattr(pipeline_module, "build_temperature_gate_config", lambda: gate)

    def verify(raw, *, gate_config, workspace_root):
        assert raw == plan
        assert gate_config == gate
        assert Path(workspace_root).is_absolute()
        return raw

    monkeypatch.setattr(pipeline_module, "verify_behavior_collection_plan", verify)


def _evaluation_fixture(tmp_path: Path, *, roots=3):
    evaluators = _evaluators()
    dispatch = TurnActorBehaviorDispatch(
        evaluators, model_id=EXPECTED_INPUT_POLICY_ID
    )
    collection_dir = tmp_path / "collection"
    evaluation_dir = tmp_path / "evaluation"
    collection = collect_sharded_behavior_traces(
        collection_dir,
        BehaviorTraceCollectionConfig(
            seed_namespace=f"test-m3-recovery-{tmp_path.name}-v1",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
        ),
        dispatch,
        total_root_target=roots,
        shard_size=1,
    )
    return collection, collection_dir, evaluation_dir, evaluators


@pytest.fixture(scope="module")
def publication_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("m3-no-replace-publication")
    run_dir, evaluators, gate, _gate_path, _plan, _plan_path = _fixture(root)
    evaluate_hu_behavior_trace_shards(
        run_dir / "natural", run_dir / "natural_evaluation", evaluators
    )
    evaluate_hu_behavior_trace_shards(
        run_dir / "challenge", run_dir / "challenge_evaluation", evaluators
    )
    artifact = build_sharded_behavior_temperature_calibration(
        run_dir / "natural",
        run_dir / "natural_evaluation",
        run_dir / "challenge",
        run_dir / "challenge_evaluation",
        evaluators,
        gate_config=gate,
        scratch_dir=root / "build-scratch",
    )
    return root, run_dir, evaluators, artifact


def _publish_fixture_artifact(publication_fixture, output: Path, scratch: Path):
    _root, run_dir, evaluators, artifact = publication_fixture
    return _publish_calibration_no_replace(
        artifact=artifact,
        output=output,
        natural_collection_dir=run_dir / "natural",
        natural_evaluation_dir=run_dir / "natural_evaluation",
        challenge_collection_dir=run_dir / "challenge",
        challenge_evaluation_dir=run_dir / "challenge_evaluation",
        evaluators=evaluators,
        scratch_dir=scratch,
    )


def test_checked_in_gate_is_canonical_content_addressed_default():
    root = Path(__file__).resolve().parents[1]
    path = root / pipeline_module.DEFAULT_GATE_CONFIG
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    gate = json.loads(raw)
    assert raw == (canonical_json(gate) + "\n").encode("utf-8")
    assert verify_temperature_gate_config(gate) == build_temperature_gate_config()
    assert gate["gate_config_sha256"] == (
        "4f90642b47740785ea767aa80ce88055b429769756b657441299449c1e708569"
    )


def test_recovers_one_exact_next_orphan_from_authenticated_prefix(tmp_path):
    _collection, collection_dir, evaluation_dir, evaluators = _evaluation_fixture(
        tmp_path
    )
    prefix = evaluate_hu_behavior_trace_shards(
        collection_dir,
        evaluation_dir,
        evaluators,
        max_new_shards=1,
    )
    prefix_bytes = (evaluation_dir / "manifest.json").read_bytes()
    two = evaluate_hu_behavior_trace_shards(
        collection_dir,
        evaluation_dir,
        evaluators,
        resume=True,
        max_new_shards=1,
    )
    assert two.shard_count == 2
    (evaluation_dir / "manifest.json").write_bytes(prefix_bytes)

    recovered = _read_existing_evaluation(
        collection_dir=collection_dir,
        evaluation_dir=evaluation_dir,
        evaluators=evaluators,
    )
    assert recovered is not None
    assert recovered.manifest == two.manifest
    assert recovered.shard_count == prefix.shard_count + 1


def test_recovers_published_shard_zero_when_top_was_never_written(tmp_path):
    _collection, collection_dir, evaluation_dir, evaluators = _evaluation_fixture(
        tmp_path
    )
    first = evaluate_hu_behavior_trace_shards(
        collection_dir,
        evaluation_dir,
        evaluators,
        max_new_shards=1,
    )
    (evaluation_dir / "manifest.json").unlink()

    recovered = _read_existing_evaluation(
        collection_dir=collection_dir,
        evaluation_dir=evaluation_dir,
        evaluators=evaluators,
    )
    assert recovered is not None
    assert recovered.manifest == first.manifest
    assert (evaluation_dir / "manifest.json").is_file()


def test_run_wide_writer_lock_prevents_delayed_stale_runner_rollback(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    top = run_dir / "simulated-top"
    top.write_text("0", encoding="ascii")
    entered = threading.Event()
    release = threading.Event()
    observed_prefixes = []
    first_results = []
    first_errors = []

    def simulated_locked_pipeline(**_kwargs):
        prefix = int(top.read_text(encoding="ascii"))
        observed_prefixes.append(prefix)
        if prefix == 0:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release first pipeline writer")
        top.write_text(str(prefix + 1), encoding="ascii")
        return {"published_prefix": prefix + 1}

    monkeypatch.setattr(
        pipeline_module,
        "_run_m3_behavior_calibration_pipeline_locked",
        simulated_locked_pipeline,
    )
    kwargs = {
        "workspace_root": tmp_path,
        "run_dir": run_dir,
        "plan_path": tmp_path / "unused-plan.json",
        "gate_config_path": tmp_path / "unused-gate.json",
    }

    def first_runner():
        try:
            first_results.append(run_m3_behavior_calibration_pipeline(**kwargs))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            first_errors.append(exc)

    thread = threading.Thread(target=first_runner, daemon=True)
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(RuntimeError, match="another M3 calibration pipeline"):
            run_m3_behavior_calibration_pipeline(**kwargs)
        assert top.read_text(encoding="ascii") == "0"
        assert observed_prefixes == [0]
    finally:
        release.set()
        thread.join(10)

    assert not thread.is_alive()
    assert first_errors == []
    assert first_results == [{"published_prefix": 1}]
    assert top.read_text(encoding="ascii") == "1"

    retry = run_m3_behavior_calibration_pipeline(**kwargs)
    assert retry == {"published_prefix": 2}
    assert observed_prefixes == [0, 1]
    assert top.read_text(encoding="ascii") == "2"


def test_writer_lock_is_cross_process_and_released_after_process_exit(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_pipeline_writer_lock_in_child,
        args=(str(run_dir), ready, release),
    )
    process.start()
    try:
        assert ready.wait(20), f"lock holder failed with exit code {process.exitcode}"
        with pytest.raises(RuntimeError, match="another M3 calibration pipeline"):
            with pipeline_module._pipeline_writer_lock(run_dir):
                pytest.fail("concurrent process acquired the writer lock")
    finally:
        if process.is_alive():
            process.terminate()
        process.join(20)
        if process.is_alive():  # pragma: no cover - hard cleanup on platform failure
            process.kill()
            process.join(20)

    assert not process.is_alive()
    lock_path = run_dir / pipeline_module.PIPELINE_WRITER_LOCK_NAME
    assert lock_path.is_file() and lock_path.stat().st_size >= 1
    with pipeline_module._pipeline_writer_lock(run_dir):
        pass


@pytest.mark.parametrize("failure", ["multiple", "invalid", "staging"])
def test_recovery_rejects_multiple_invalid_and_staging_orphans(
    tmp_path, failure
):
    _collection, collection_dir, evaluation_dir, evaluators = _evaluation_fixture(
        tmp_path
    )
    first = evaluate_hu_behavior_trace_shards(
        collection_dir,
        evaluation_dir,
        evaluators,
        max_new_shards=1,
    )
    prefix_bytes = (evaluation_dir / "manifest.json").read_bytes()
    if failure == "staging":
        staging = evaluation_dir / ".evaluation-shard-000001.crash.partial"
        staging.mkdir()
        (staging / "evaluations.jsonl").write_text("partial", encoding="utf-8")
        expected = "unpublished staging"
    else:
        evaluate_hu_behavior_trace_shards(
            collection_dir,
            evaluation_dir,
            evaluators,
            resume=True,
            max_new_shards=2 if failure == "multiple" else 1,
        )
        (evaluation_dir / "manifest.json").write_bytes(prefix_bytes)
        if failure == "multiple":
            expected = "exactly one next orphan"
        else:
            path = (
                evaluation_dir
                / "evaluation-shard-000001"
                / "evaluations.jsonl"
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["visible_joker_count"] = (
                rows[0]["visible_joker_count"] + 1
            ) % 3
            path.write_bytes(
                ("\n".join(canonical_json(row) for row in rows) + "\n").encode()
            )
            expected = "evaluation row SHA-256 mismatch"

    with pytest.raises(ValueError, match=expected):
        _read_existing_evaluation(
            collection_dir=collection_dir,
            evaluation_dir=evaluation_dir,
            evaluators=evaluators,
        )
    assert (evaluation_dir / "manifest.json").read_bytes() == prefix_bytes
    assert first.shard_count == 1


def test_runner_resumes_authenticated_prefix_then_publishes_once(
    tmp_path, monkeypatch
):
    run_dir, evaluators, gate, gate_path, plan, plan_path = _fixture(tmp_path)
    _install_test_plan_verifier(monkeypatch, gate, plan)
    # Match production's already-complete challenge side.
    evaluate_hu_behavior_trace_shards(
        run_dir / "challenge", run_dir / "challenge_evaluation", evaluators
    )

    partial = run_m3_behavior_calibration_pipeline(
        workspace_root=tmp_path,
        run_dir=run_dir,
        plan_path=plan_path,
        gate_config_path=gate_path,
        max_new_evaluation_shards=1,
        evaluators=evaluators,
    )
    assert partial["schema"] == PIPELINE_RESULT_SCHEMA
    assert partial["stage"] == "natural_evaluation_incomplete"
    assert partial["pipeline_complete"] is False
    assert partial["evaluations"]["natural"]["shard_count"] == 1
    assert partial["calibration"] is None
    assert not (run_dir / "calibration.json").exists()

    completed = run_m3_behavior_calibration_pipeline(
        workspace_root=tmp_path,
        run_dir=run_dir,
        plan_path=plan_path,
        gate_config_path=gate_path,
        max_new_evaluation_shards=1,
        evaluators=evaluators,
    )
    assert completed["stage"] == "calibration_complete"
    assert completed["pipeline_complete"] is True
    assert completed["evaluations"]["natural"]["evaluation_complete"] is True
    assert completed["calibration"]["created"] is True
    before = (run_dir / "calibration.json").read_bytes()

    replay = run_m3_behavior_calibration_pipeline(
        workspace_root=tmp_path,
        run_dir=run_dir,
        plan_path=plan_path,
        gate_config_path=gate_path,
        evaluators=evaluators,
    )
    assert replay["calibration"]["created"] is False
    assert replay["calibration"]["artifact_sha256"] == completed["calibration"][
        "artifact_sha256"
    ]
    assert (run_dir / "calibration.json").read_bytes() == before


def test_calibration_writer_targets_private_complete_stage_not_final(
    publication_fixture, tmp_path, monkeypatch
):
    output = tmp_path / "calibration.json"
    original_writer = pipeline_module.write_sharded_behavior_temperature_calibration
    targets = []

    def recording_writer(artifact, path):
        targets.append(Path(path).resolve())
        return original_writer(artifact, path)

    monkeypatch.setattr(
        pipeline_module,
        "write_sharded_behavior_temperature_calibration",
        recording_writer,
    )
    artifact, created = _publish_fixture_artifact(
        publication_fixture, output, tmp_path / "scratch"
    )
    assert created is True
    assert artifact["artifact_sha256"]
    assert output.is_file()
    assert len(targets) == 1 and targets[0] != output.resolve()
    assert targets[0].parent == output.parent.resolve()
    assert targets[0].name.startswith(f".{output.name}.")
    assert not targets[0].exists()


def test_equal_concurrent_calibration_publish_is_freshly_accepted(
    publication_fixture, tmp_path, monkeypatch
):
    output = tmp_path / "calibration.json"
    original_link = pipeline_module.os.link

    def winning_competitor(source, destination):
        original_link(source, destination)
        raise FileExistsError("simulated equal concurrent winner")

    monkeypatch.setattr(pipeline_module.os, "link", winning_competitor)
    artifact, created = _publish_fixture_artifact(
        publication_fixture, output, tmp_path / "scratch"
    )
    assert created is False
    assert artifact["artifact_sha256"]
    assert output.is_file()
    assert not list(tmp_path.glob(f".{output.name}.*.complete"))


def test_different_concurrent_calibration_is_not_overwritten(
    publication_fixture, tmp_path, monkeypatch
):
    output = tmp_path / "calibration.json"
    competitor = tmp_path / "competitor.complete"
    competitor.write_bytes(b"{}\n")
    original_link = pipeline_module.os.link

    def different_competitor(_source, destination):
        original_link(competitor, destination)
        raise FileExistsError("simulated different concurrent winner")

    monkeypatch.setattr(pipeline_module.os, "link", different_competitor)
    with pytest.raises(ValueError):
        _publish_fixture_artifact(
            publication_fixture, output, tmp_path / "scratch"
        )
    assert output.read_bytes() == b"{}\n"
    assert not list(tmp_path.glob(f".{output.name}.*.complete"))


def test_staging_failure_never_leaves_partial_final_name(
    publication_fixture, tmp_path, monkeypatch
):
    output = tmp_path / "calibration.json"

    def failing_writer(_artifact, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated staged writer crash")

    monkeypatch.setattr(
        pipeline_module,
        "write_sharded_behavior_temperature_calibration",
        failing_writer,
    )
    with pytest.raises(RuntimeError, match="staged writer crash"):
        _publish_fixture_artifact(
            publication_fixture, output, tmp_path / "scratch"
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.complete"))


def test_gate_drift_fails_before_any_evaluation_output(tmp_path, monkeypatch):
    run_dir, evaluators, gate, gate_path, plan, plan_path = _fixture(tmp_path)
    _install_test_plan_verifier(monkeypatch, gate, plan)
    changed = build_temperature_gate_config(
        gate_id="different-valid-gate-v1",
        min_fit_decisions_per_role=1,
        min_dev_decisions_per_role=1,
        min_test_decisions_per_role=1,
        min_challenge_decisions_per_role_joker=1,
        min_roots_per_split_role=1,
        min_challenge_roots_per_role_joker=1,
        challenge_root_namespace_prefix=CHALLENGE_PREFIX,
        bootstrap_replicates=20,
        bootstrap_seed="different-valid-gate-bootstrap-v1",
        ece_bins=5,
    )
    _write_canonical(gate_path, changed)

    with pytest.raises(ValueError, match="current locked production default"):
        run_m3_behavior_calibration_pipeline(
            workspace_root=tmp_path,
            run_dir=run_dir,
            plan_path=plan_path,
            gate_config_path=gate_path,
            evaluators=evaluators,
        )
    assert not (run_dir / "natural_evaluation").exists()
    assert not (run_dir / "challenge_evaluation").exists()


def test_preregistered_root_commitment_drift_fails_before_evaluation(
    tmp_path, monkeypatch
):
    run_dir, evaluators, gate, gate_path, plan, plan_path = _fixture(tmp_path)
    changed = copy.deepcopy(plan)
    changed["natural"]["range"]["shards"][0]["root_id_order_sha256"] = "f" * 64
    _write_canonical(plan_path, changed)
    _install_test_plan_verifier(monkeypatch, gate, changed)

    with pytest.raises(ValueError, match="shard 0 differs from preregistration"):
        run_m3_behavior_calibration_pipeline(
            workspace_root=tmp_path,
            run_dir=run_dir,
            plan_path=plan_path,
            gate_config_path=gate_path,
            evaluators=evaluators,
        )
    assert not (run_dir / "natural_evaluation").exists()
    assert not (run_dir / "challenge_evaluation").exists()


def test_growing_natural_collection_fails_fast_before_evaluation(
    tmp_path, monkeypatch
):
    run_dir, evaluators, gate, gate_path, plan, plan_path = _fixture(tmp_path)
    _install_test_plan_verifier(monkeypatch, gate, plan)
    top_path = run_dir / "natural" / "manifest.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    incomplete = _build_top_manifest(
        config=BehaviorTraceCollectionConfig.from_canonical_dict(
            top["collection_config"]
        ),
        policy=top["policy"],
        shard_size=top["shard_size"],
        requested_total_root_target=top["requested_total_root_target"] + 1,
        shards=top["shards"],
        aggregate=_Aggregate.from_manifest(top),
    )
    _write_canonical(top_path, incomplete)

    with pytest.raises(ValueError, match="natural collection is not complete"):
        run_m3_behavior_calibration_pipeline(
            workspace_root=tmp_path,
            run_dir=run_dir,
            plan_path=plan_path,
            gate_config_path=gate_path,
            evaluators=evaluators,
        )
    assert not (run_dir / "natural_evaluation").exists()
    assert not (run_dir / "challenge_evaluation").exists()


def test_cli_requires_explicit_gate_config():
    with pytest.raises(SystemExit) as exc:
        pipeline_module.main(
            [
                "--run-dir",
                "run",
                "--plan",
                "plan.json",
            ]
        )
    assert exc.value.code == 2
