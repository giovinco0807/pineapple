from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import ofc_regular.hu_rl_scalar_parity as scalar_parity_module
from ofc_regular.action_key import ActionKey
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_rl_contract import HuRlActorViewV1
from ofc_regular.hu_rl_scalar_parity import (
    MAX_PARITY_WORKERS,
    PRIVILEGED_AUDIT_ROLE,
    SCALAR_PARITY_SUMMARY_SCHEMA,
    SCALAR_PARITY_RUN_CONTRACT_SCHEMA,
    SCALAR_TRACE_REQUEST_SCHEMA,
    SCALAR_TRACE_RESULT_SCHEMA,
    HuRlScalarBinaryError,
    HuRlScalarParityError,
    ScalarTraceRequestV1,
    build_scalar_parity_run_contract,
    build_python_scalar_trace,
    compare_rust_scalar_trace,
    generate_seeded_scalar_trace_requests,
    invoke_prebuilt_scalar_binary,
    load_scalar_parity_summary,
    pin_scalar_parity_binary,
    run_scalar_parity_suite,
    runtime_source_relative_paths,
    scalar_parity_run_contract_digest,
    validate_runtime_source_identity,
    write_scalar_parity_summary,
)


def test_seeded_requests_are_deterministic_unique_and_replay_valid() -> None:
    first = generate_seeded_scalar_trace_requests(hands=8, seed=20260722)
    second = generate_seeded_scalar_trace_requests(hands=8, seed=20260722)

    assert first == second
    assert len({request.explicit_deck for request in first}) == 8
    assert len({request.digest() for request in first}) == 8
    for request in first:
        assert request.to_dict()["schema"] == SCALAR_TRACE_REQUEST_SCHEMA
        assert len(request.explicit_deck) == 52
        assert set(request.explicit_deck) == set(ALL_CARDS)
        assert len(request.selected_indices) == 10
        assert "explicit_deck=<redacted>" in repr(request)
        assert ScalarTraceRequestV1.from_dict(request.to_dict()) == request


def test_python_trace_has_closed_ten_decision_result_contract() -> None:
    request = generate_seeded_scalar_trace_requests(hands=1, seed=31)[0]
    result = build_python_scalar_trace(request)

    assert result["schema"] == SCALAR_TRACE_RESULT_SCHEMA
    assert result["artifact_role"] == "privileged_correctness_audit_only"
    assert result["policy_input_eligible"] is False
    assert result["replay_eligible"] is False
    assert result["training_eligible"] is False
    assert result["contains_cross_actor_private_information"] is True
    assert len(result["decisions"]) == 10
    assert "explicit_deck" not in _all_mapping_keys(result)
    for ordinal, decision in enumerate(result["decisions"]):
        assert set(decision) == {
            "ordinal",
            "actor",
            "street",
            "actor_view",
            "actor_view_digest",
            "legal_action_mapping",
            "selected_index",
            "selected_action_key",
            "step",
        }
        assert decision["ordinal"] == ordinal
        assert decision["actor"] == ordinal % 2
        assert decision["street"] == f"T{ordinal // 2}"
        assert set(decision["legal_action_mapping"]) == {
            "action_count",
            "action_set_digest",
            "action_order_digest",
        }
        assert set(decision["step"]) == {"public_event", "done", "rewards"}
        assert decision["step"]["done"] is (ordinal == 9)
    assert result["terminal"]["rewards"] == result["decisions"][-1]["step"][
        "rewards"
    ]
    proof = compare_rust_scalar_trace(request, result)
    assert proof["exact"] is True
    assert proof["decision_count"] == 10


def test_full_trace_is_privileged_because_it_reconstructs_both_seats_discards() -> None:
    request = generate_seeded_scalar_trace_requests(hands=1, seed=47)[0]
    result = build_python_scalar_trace(request)
    discard_sequence_by_actor: dict[int, list[str]] = {0: [], 1: []}
    for decision in result["decisions"]:
        selected = ActionKey.from_token(decision["selected_action_key"])
        discard_sequence_by_actor[decision["actor"]].extend(
            selected.cards("discards")
        )

    discards_by_actor = {
        actor: set(cards) for actor, cards in discard_sequence_by_actor.items()
    }
    assert {actor: len(cards) for actor, cards in discards_by_actor.items()} == {0: 4, 1: 4}
    assert discards_by_actor[0].isdisjoint(discards_by_actor[1])
    # At T4 each actor sees its three prior discards and its current dealt
    # cards. The aggregate selected-key trace additionally reconstructs the
    # final discard choice and all four private discards of the other actor.
    for actor in (0, 1):
        t4 = result["decisions"][8 + actor]["actor_view"]
        assert set(t4["observation"]["hero_private_discards"]) == set(
            discard_sequence_by_actor[actor][:3]
        )
        parsed_view = HuRlActorViewV1.from_dict(t4)
        assert set(parsed_view.observation.known_unavailable_cards()).isdisjoint(
            discards_by_actor[1 - actor]
        )

    assert result["artifact_role"] == "privileged_correctness_audit_only"
    assert result["contains_cross_actor_private_information"] is True
    assert result["policy_input_eligible"] is False
    assert result["replay_eligible"] is False
    assert result["training_eligible"] is False


def test_request_contract_rejects_unknown_invalid_and_illegal_indices() -> None:
    request = generate_seeded_scalar_trace_requests(hands=1, seed=5)[0].to_dict()
    unknown = {**request, "deck_tail": []}
    with pytest.raises(HuRlScalarParityError, match="unknown fields: deck_tail"):
        ScalarTraceRequestV1.from_dict(unknown)

    duplicate = deepcopy(request)
    duplicate["explicit_deck"][-1] = duplicate["explicit_deck"][0]
    with pytest.raises(HuRlScalarParityError, match="not a valid complete regular deck"):
        ScalarTraceRequestV1.from_dict(duplicate)

    illegal = deepcopy(request)
    illegal["selected_indices"][0] = 232
    with pytest.raises(HuRlScalarParityError, match="outside its legal mapping"):
        ScalarTraceRequestV1.from_dict(illegal)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("actor_view_digest", "actor view digest mismatch"),
        ("action_count", "action count mismatch"),
        ("selected_key", "selected ActionKey mismatch"),
        ("public_event", "public event/action mismatch|invalid public event"),
        ("reward", "nonterminal result rewards are nonzero"),
        ("terminal", "duplicate cards|invalid result terminal board"),
        ("classification", "privileged audit classification mismatch"),
        ("unknown", "unknown fields: audit_truth"),
    ),
)
def test_result_tampering_is_rejected_fail_closed(mutation: str, message: str) -> None:
    request = generate_seeded_scalar_trace_requests(hands=1, seed=71)[0]
    result = build_python_scalar_trace(request)
    tampered = deepcopy(result)
    if mutation == "actor_view_digest":
        tampered["decisions"][0]["actor_view_digest"] = "0" * 64
    elif mutation == "action_count":
        tampered["decisions"][0]["legal_action_mapping"]["action_count"] -= 1
    elif mutation == "selected_key":
        tampered["decisions"][0]["selected_action_key"] = tampered["decisions"][
            0
        ]["actor_view"]["legal_action_mapping"]["action_keys"][0]
    elif mutation == "public_event":
        event = tampered["decisions"][2]["step"]["public_event"]
        event["top_placement_mask"], event["middle_placement_mask"] = (
            event["middle_placement_mask"],
            event["top_placement_mask"],
        )
    elif mutation == "reward":
        tampered["decisions"][0]["step"]["rewards"] = [1.0, -1.0]
    elif mutation == "terminal":
        tampered["terminal"]["boards"][0]["top"][0] = tampered["terminal"][
            "boards"
        ][1]["top"][0]
    elif mutation == "classification":
        tampered["policy_input_eligible"] = 0
    elif mutation == "unknown":
        tampered["decisions"][0]["step"]["audit_truth"] = {}

    with pytest.raises(HuRlScalarParityError, match=message):
        compare_rust_scalar_trace(request, tampered)


def test_result_with_valid_shape_but_different_terminal_value_reports_path_only() -> None:
    request = generate_seeded_scalar_trace_requests(hands=1, seed=99)[0]
    result = build_python_scalar_trace(request)
    tampered = deepcopy(result)
    left, right = tampered["terminal"]["boards"]
    left["top"], right["top"] = right["top"], left["top"]
    # Keep this structurally valid only when the swapped top rows have equal sizes.
    with pytest.raises(HuRlScalarParityError) as caught:
        compare_rust_scalar_trace(request, tampered)
    assert "explicit_deck" not in str(caught.value)
    assert not any(card in str(caught.value) for card in ALL_CARDS)


def test_real_prebuilt_binary_four_hand_smoke_if_available() -> None:
    binary = _prebuilt_binary()
    if binary is None:
        pytest.skip("prebuilt hu_rl_scalar_trace binary is not available")
    requests = generate_seeded_scalar_trace_requests(hands=4, seed=20260722)
    for request in requests:
        result = invoke_prebuilt_scalar_binary(binary, request, timeout_seconds=30.0)
        assert compare_rust_scalar_trace(request, result)["exact"] is True


def test_parallel_suite_is_byte_identical_to_scalar_suite_if_available(
    tmp_path: Path,
) -> None:
    binary = _prebuilt_binary()
    if binary is None:
        pytest.skip("prebuilt hu_rl_scalar_trace binary is not available")
    pinned = pin_scalar_parity_binary(binary, tmp_path / "pinned")
    kwargs = _suite_contract_kwargs(global_hands=2)
    scalar = run_scalar_parity_suite(
        pinned,
        hands=2,
        seed=20260724,
        workers=1,
        timeout_seconds=30.0,
        **kwargs,
    )
    parallel = run_scalar_parity_suite(
        pinned,
        hands=2,
        seed=20260724,
        workers=2,
        timeout_seconds=30.0,
        **kwargs,
    )
    assert scalar["proofs"] == parallel["proofs"]
    assert scalar["run_contract"] == parallel["run_contract"]
    assert scalar["workers"] == 1
    assert parallel["workers"] == 2


def test_sharded_suite_uses_the_same_global_hand_contract_if_available(
    tmp_path: Path,
) -> None:
    binary = _prebuilt_binary()
    if binary is None:
        pytest.skip("prebuilt hu_rl_scalar_trace binary is not available")
    pinned = pin_scalar_parity_binary(binary, tmp_path / "pinned")
    kwargs = _suite_contract_kwargs(global_hands=3)
    full = run_scalar_parity_suite(
        pinned,
        hands=3,
        seed=20260728,
        workers=2,
        timeout_seconds=30.0,
        **kwargs,
    )
    tail = run_scalar_parity_suite(
        pinned,
        hands=2,
        hand_start=1,
        seed=20260728,
        workers=2,
        timeout_seconds=30.0,
        **kwargs,
    )
    assert tail["hand_start"] == 1
    assert tail["hand_end_exclusive"] == 3
    assert tail["proofs"] == full["proofs"][1:]


@pytest.mark.parametrize("workers", (0, 17, True))
def test_suite_rejects_invalid_worker_count(workers: object) -> None:
    with pytest.raises(HuRlScalarParityError, match="workers must be an integer"):
        run_scalar_parity_suite(
            Path("missing"),
            hands=1,
            seed=1,
            workers=workers,  # type: ignore[arg-type]
            **_suite_contract_kwargs(global_hands=1),
        )


@pytest.mark.parametrize(("hand_start", "hands"), ((-1, 1), (10_000, 1)))
def test_suite_rejects_invalid_hand_range(hand_start: int, hands: int) -> None:
    with pytest.raises(HuRlScalarParityError, match="hand_start"):
        run_scalar_parity_suite(
            Path("missing"),
            hands=hands,
            hand_start=hand_start,
            seed=1,
            workers=1,
            **_suite_contract_kwargs(global_hands=10_000),
        )


@pytest.mark.parametrize("hands", (0, -1, True))
def test_suite_rejects_invalid_hands_directly(hands: object) -> None:
    with pytest.raises(HuRlScalarParityError, match="hands must be an integer"):
        run_scalar_parity_suite(
            Path("missing"),
            hands=hands,  # type: ignore[arg-type]
            hand_start=5,
            seed=1,
            workers=1,
            **_suite_contract_kwargs(global_hands=10),
        )


def test_v3_run_contract_is_content_addressed_privileged_and_exhaustive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "engine.exe"
    source.write_bytes(b"frozen scalar engine")
    pinned = pin_scalar_parity_binary(source, tmp_path / "pins")
    assert pin_scalar_parity_binary(pinned, tmp_path / "pins") == pinned
    root = _repo_root()
    contract = build_scalar_parity_run_contract(
        pinned,
        profile_path=root / "src/ofc_regular/ai_profiles.py",
        source_root=root,
        seed=17,
        global_hands=4,
    )

    assert contract["schema"] == SCALAR_PARITY_RUN_CONTRACT_SCHEMA
    assert contract["native_binary"]["content_addressed_name"] == pinned.name
    assert contract["native_binary"]["sha256"] in pinned.name
    assert contract["native_binary"]["read_only"] is True
    assert contract["artifact_role"] == PRIVILEGED_AUDIT_ROLE
    assert contract["contains_raw_cross_actor_private_information"] is False
    assert contract["contains_reconstructable_hidden_oracle_state"] is True
    assert contract["policy_input_eligible"] is False
    paths = set(runtime_source_relative_paths(root))
    assert paths == set(contract["validator_sources"])
    for required in (
        "configs/fl_ev_regular_2k.json",
        "configs/fl_ev_regular_v3_direct2.json",
        "src/ofc_regular/__init__.py",
        "src/ofc_regular/action_space.py",
        "src/ofc_regular/hu_infoset.py",
        "src/ofc_regular/policy.py",
        "src/ofc_regular/teacher.py",
        "rust/hu_rl_engine/src/trace.rs",
        "rust/hu_m3_engine/src/scoring.rs",
    ):
        assert required in paths
    assert len(scalar_parity_run_contract_digest(contract)) == 64


def test_runtime_manifest_covers_transitive_local_python_and_all_rust_sources() -> None:
    root = _repo_root()
    manifest = set(runtime_source_relative_paths(root))
    module_paths = {
        path.stem: path for path in (root / "src/ofc_regular").glob("*.py")
    }
    # Importing any package submodule executes ``ofc_regular.__init__`` first.
    # Include both roots so the manifest also closes over those eager imports.
    pending = ["__init__", "hu_rl_scalar_parity"]
    closure: set[str] = set()
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        closure.add(module)
        tree = ast.parse(module_paths[module].read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                dependency = node.module.split(".")[0]
                if dependency in module_paths and dependency not in closure:
                    pending.append(dependency)
    assert {
        module_paths[module].relative_to(root).as_posix() for module in closure
    } <= manifest
    rust_sources = {
        path.relative_to(root).as_posix()
        for crate in ("hu_m3_engine", "hu_rl_engine")
        for path in (root / "rust" / crate / "src").rglob("*.rs")
        if "bin" not in path.relative_to(root / "rust" / crate / "src").parts
    }
    rust_sources.add("rust/hu_rl_engine/src/bin/hu_rl_scalar_trace.rs")
    assert rust_sources <= manifest


def test_runtime_manifest_hashes_default_fl_ev_config_and_effective_scoring(
    tmp_path: Path,
) -> None:
    root = _repo_root()
    source = tmp_path / "engine.exe"
    source.write_bytes(b"frozen scalar engine")
    pinned = pin_scalar_parity_binary(source, tmp_path / "pins")
    contract = build_scalar_parity_run_contract(
        pinned,
        profile_path=root / "src/ofc_regular/ai_profiles.py",
        source_root=root,
        seed=19,
        global_hands=1,
    )

    for relpath in (
        "configs/fl_ev_regular_2k.json",
        "configs/fl_ev_regular_v3_direct2.json",
    ):
        config = root / relpath
        assert contract["validator_sources"][relpath] == {
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "size_bytes": config.stat().st_size,
        }
    effective = contract["effective_scoring"]
    assert effective["config_relpath"] == "configs/fl_ev_regular_v4_selfplay.json"
    assert effective["fl_ev"] == [{"cards": 14, "value_hex": float(9.6).hex()}]
    assert len(effective["canonical_sha256"]) == 64


def test_copied_source_root_and_mismatched_fl_config_fail_before_execution(
    tmp_path: Path,
) -> None:
    root = _repo_root()
    source = tmp_path / "engine.exe"
    source.write_bytes(b"frozen scalar engine")
    pinned = pin_scalar_parity_binary(source, tmp_path / "pins")
    copied_root = tmp_path / "source"
    for relpath in runtime_source_relative_paths(root):
        destination = copied_root / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relpath).read_bytes())
    profile = copied_root / "src/ofc_regular/ai_profiles.py"
    profile.write_bytes((root / "src/ofc_regular/ai_profiles.py").read_bytes())
    fl_config = copied_root / "configs/fl_ev_regular_v3_direct2.json"
    payload = json.loads(fl_config.read_text(encoding="utf-8"))
    payload["fl_ev"]["14"] = 999.0
    fl_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HuRlScalarParityError, match="frozen source_root"):
        run_scalar_parity_suite(
            pinned,
            hands=1,
            seed=19,
            global_hands=1,
            profile_path=profile,
            source_root=copied_root,
            workers=1,
        )


def test_effective_scoring_and_runtime_dependency_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo_root()
    source = tmp_path / "engine.exe"
    source.write_bytes(b"frozen scalar engine")
    pinned = pin_scalar_parity_binary(source, tmp_path / "pins")
    contract = build_scalar_parity_run_contract(
        pinned,
        profile_path=root / "src/ofc_regular/ai_profiles.py",
        source_root=root,
        seed=19,
        global_hands=1,
    )
    runtime = contract["python_runtime"]
    assert runtime["executable_binary"]["size_bytes"] > 0
    assert len(runtime["executable_binary"]["sha256"]) == 64
    assert runtime["numpy"]["version"]
    assert runtime["numpy"]["distribution_name"].lower() == "numpy"
    assert runtime["numpy"]["module_file"]["size_bytes"] > 0
    assert len(runtime["numpy"]["record_sha256"]) == 64

    monkeypatch.setattr(scalar_parity_module._teacher_module, "DEFAULT_FL_EV", {14: 999.0})
    with pytest.raises(HuRlScalarParityError, match="effective Python scoring"):
        build_scalar_parity_run_contract(
            pinned,
            profile_path=root / "src/ofc_regular/ai_profiles.py",
            source_root=root,
            seed=19,
            global_hands=1,
        )


def test_runtime_entrypoint_source_identity_rejects_copied_path(tmp_path: Path) -> None:
    root = _repo_root()
    copied = tmp_path / "run_hu_rl_scalar_parity.py"
    copied.write_bytes((root / "scripts/run_hu_rl_scalar_parity.py").read_bytes())
    with pytest.raises(HuRlScalarParityError, match="frozen source_root"):
        validate_runtime_source_identity(
            root,
            copied,
            "scripts/run_hu_rl_scalar_parity.py",
        )


def test_atomic_shard_write_is_byte_identical_resume_and_partial_fail_closed(
    tmp_path: Path,
) -> None:
    summary = _fake_summary(tmp_path)
    target = tmp_path / "shard_00000_00000.json"
    first = write_scalar_parity_summary(target, summary)
    frozen = target.read_bytes()
    second = write_scalar_parity_summary(target, summary)

    assert first == second
    assert target.read_bytes() == frozen
    loaded, encoded = load_scalar_parity_summary(
        target,
        expected_hand_start=0,
        expected_hands=1,
        expected_run_contract_sha256=summary["run_contract_sha256"],
    )
    assert loaded == first
    assert encoded == frozen

    partial = tmp_path / "shard_00001_00001.json"
    partial.write_bytes(b'{"schema":')
    other = {**summary, "hand_start": 1, "hand_end_exclusive": 2}
    other["proofs"] = [{**summary["proofs"][0], "hand_ordinal": 1}]
    with pytest.raises(HuRlScalarParityError, match="existing artifact"):
        write_scalar_parity_summary(partial, other)
    assert partial.read_bytes() == b'{"schema":'


def test_persisted_summary_rejects_float_identity_and_contract_drift(
    tmp_path: Path,
) -> None:
    summary = write_scalar_parity_summary(
        tmp_path / "valid.json", _fake_summary(tmp_path)
    )
    tampered = deepcopy(summary)
    tampered["seed"] = float(tampered["seed"])
    path = tmp_path / "float.json"
    path.write_text(
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    with pytest.raises(HuRlScalarParityError, match="integer field"):
        load_scalar_parity_summary(path)

    drifted = deepcopy(summary)
    drifted["run_contract_sha256"] = "0" * 64
    drift = tmp_path / "drift.json"
    drift.write_text(
        json.dumps(
            drifted,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    with pytest.raises(HuRlScalarParityError, match="run contract changed"):
        load_scalar_parity_summary(drift)


def test_run_script_redacts_runtime_failure(tmp_path: Path) -> None:
    root = _repo_root()
    missing = tmp_path / "private-card-name-AS.exe"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_hu_rl_scalar_parity.py"),
            "--binary",
            str(missing),
            "--pin-dir",
            str(tmp_path / "pins"),
            "--profile",
            str(root / "src/ofc_regular/ai_profiles.py"),
            "--source-root",
            str(root),
            "--hands",
            "1",
            "--global-hands",
            "1",
        ],
        cwd=root,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        '{"error":"scalar parity run failed","status":"error"}\n'
    )
    assert str(missing) not in completed.stderr


def test_run_script_persists_then_validates_resume_if_binary_available(
    tmp_path: Path,
) -> None:
    binary = _prebuilt_binary()
    if binary is None:
        pytest.skip("prebuilt hu_rl_scalar_trace binary is not available")
    root = _repo_root()
    output = tmp_path / "shard_00000_00000.json"
    command = [
        sys.executable,
        str(root / "scripts/run_hu_rl_scalar_parity.py"),
        "--binary",
        str(binary),
        "--pin-dir",
        str(tmp_path / "pins"),
        "--profile",
        str(root / "src/ofc_regular/ai_profiles.py"),
        "--source-root",
        str(root),
        "--hands",
        "1",
        "--global-hands",
        "1",
        "--seed",
        "20260731",
        "--workers",
        "1",
        "--output",
        str(output),
    ]
    first = subprocess.run(
        command,
        cwd=root,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )
    assert first.returncode == 0, first.stderr
    frozen = output.read_bytes()
    second = subprocess.run(
        command,
        cwd=root,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )
    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout
    stdout_receipt = json.loads(first.stdout)
    assert "proofs" not in stdout_receipt
    assert "seed" not in stdout_receipt
    assert stdout_receipt["contains_reconstructable_hidden_oracle_state"] is True
    assert output.read_bytes() == frozen
    loaded, _ = load_scalar_parity_summary(output)
    assert loaded["artifact_written"] is True
    assert loaded["contains_reconstructable_hidden_oracle_state"] is True


def test_suite_detects_binary_or_manifest_drift_during_one_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "engine.exe"
    source.write_bytes(b"frozen engine")
    pinned = pin_scalar_parity_binary(source, tmp_path / "pins")
    kwargs = _suite_contract_kwargs(global_hands=1)

    def rebuild_binary(job: tuple[object, ...]) -> dict[str, object]:
        executable = Path(str(job[1]))
        executable.chmod(executable.stat().st_mode | 0o200)
        executable.write_bytes(b"changed during shard")
        return {
            "hand_ordinal": 0,
            "request_sha256": "1" * 64,
            "result_sha256": "2" * 64,
            "decision_count": 10,
            "exact": True,
        }

    monkeypatch.setattr(
        scalar_parity_module, "_run_seeded_scalar_parity_job", rebuild_binary
    )
    with pytest.raises((HuRlScalarBinaryError, HuRlScalarParityError)):
        run_scalar_parity_suite(
            pinned,
            hands=1,
            seed=19,
            workers=1,
            **kwargs,
        )

    pinned = pin_scalar_parity_binary(source, tmp_path / "second-pins")
    original_builder = scalar_parity_module.build_scalar_parity_run_contract
    calls = 0

    def drifting_builder(*args: object, **builder_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        contract = original_builder(*args, **builder_kwargs)  # type: ignore[arg-type]
        if calls >= 2:
            contract = deepcopy(contract)
            contract["python_runtime"]["version"] = "drifted"  # type: ignore[index]
        return contract

    monkeypatch.setattr(
        scalar_parity_module, "build_scalar_parity_run_contract", drifting_builder
    )
    monkeypatch.setattr(
        scalar_parity_module,
        "_run_seeded_scalar_parity_job",
        lambda _job: {
            "hand_ordinal": 0,
            "request_sha256": "3" * 64,
            "result_sha256": "4" * 64,
            "decision_count": 10,
            "exact": True,
        },
    )
    with pytest.raises(HuRlScalarParityError, match="inputs changed"):
        run_scalar_parity_suite(
            pinned,
            hands=1,
            seed=23,
            workers=1,
            **kwargs,
        )


def _prebuilt_binary() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    names = ("hu_rl_scalar_trace.exe", "hu_rl_scalar_trace")
    directories = (
        root / "rust" / "hu_rl_engine" / "target" / "debug",
        root / "rust" / "hu_rl_engine" / "target" / "release",
        root / "target" / "debug",
        root / "target" / "release",
    )
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _suite_contract_kwargs(*, global_hands: int) -> dict[str, object]:
    root = _repo_root()
    return {
        "global_hands": global_hands,
        "profile_path": root / "src/ofc_regular/ai_profiles.py",
        "source_root": root,
    }


def _fake_summary(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "fake_engine.exe"
    source.write_bytes(b"fake engine bytes")
    pinned = pin_scalar_parity_binary(source, tmp_path / "fake_pins")
    root = _repo_root()
    contract = build_scalar_parity_run_contract(
        pinned,
        profile_path=root / "src/ofc_regular/ai_profiles.py",
        source_root=root,
        seed=17,
        global_hands=2,
    )
    return {
        "schema": SCALAR_PARITY_SUMMARY_SCHEMA,
        "status": "exact_python_rust_scalar_parity",
        "hands": 1,
        "hand_start": 0,
        "hand_end_exclusive": 1,
        "global_hands": 2,
        "seed": 17,
        "workers": MAX_PARITY_WORKERS,
        "total_decisions": 10,
        "unique_request_count": 1,
        "proofs": [
            {
                "hand_ordinal": 0,
                "request_sha256": "1" * 64,
                "result_sha256": "2" * 64,
                "decision_count": 10,
                "exact": True,
            }
        ],
        "run_contract": contract,
        "run_contract_sha256": scalar_parity_run_contract_digest(contract),
        "artifact_role": PRIVILEGED_AUDIT_ROLE,
        "contains_raw_cross_actor_private_information": False,
        "contains_reconstructable_hidden_oracle_state": True,
        "full_trace_persisted": False,
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "artifact_written": False,
    }


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for nested in value.values():
            keys.update(_all_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_mapping_keys(nested))
    return keys
