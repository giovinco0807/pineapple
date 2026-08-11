import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from ofc_regular.artifact_manifest import collect_git_state
from ofc_regular.freeze_hu_joint_policy_r0 import (
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_NAME,
    ARCHIVE_SPEC_NAME,
    SPEC_SCHEMA,
    SnapshotContractError,
    SnapshotNoGoError,
    SnapshotVerificationError,
    audit_snapshot,
    create_snapshot,
    load_snapshot_spec,
    verify_snapshot,
)


AUTHORIZATION = {
    "cloud_authorized": False,
    "current_profile_changed": False,
    "promotion_authorized": False,
    "quality_authorized": False,
    "runtime_activation_authorized": False,
    "training_authorized": False,
}


def test_repository_spec_splits_r0a_research_from_r0b_release_gate():
    repo = Path(__file__).resolve().parents[1]
    spec = load_snapshot_spec(
        repo / "configs/hu_joint_policy_r0_snapshot_v1.json",
        repo_root=repo,
    )
    gates = {gate.gate_id: gate for gate in spec.gates}
    assert spec.default_gate_id == "r0a_t3_research_snapshot"
    assert gates["r0a_t3_research_snapshot"].excluded_roles == (
        "m30_windows_exact_runtime",
    )
    assert gates["r0b_exact_t4_release_runtime"].excluded_roles == ()
    release_entry = next(
        entry
        for entry in spec.entries
        if entry.role == "m30_windows_exact_runtime"
    )
    assert (
        release_entry.sha256
        == "03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69"
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "R0 Test")
    _git(repo, "config", "user.email", "r0@example.invalid")
    _write(repo / ".gitignore", b"models/\noutputs/\nsnapshots/\n")
    _write(repo / "src/ai_profiles.py", b'CURRENT = "unchanged"\n')
    _write(repo / "src/runtime.py", b"VALUE = 1\n")
    _write(repo / "configs/runtime.json", b'{"mode":"explicit"}\n')
    _write(repo / "bin/engine.dll", b"native-engine-v1")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    # Preserve a real dirty tree while keeping ignored payloads explicit.
    _write(repo / "src/runtime.py", b"VALUE = 2\n")
    _write(repo / "scratch.txt", b"unrelated dirty file\n")
    _write(repo / "models/model.bin", b"model-v1")
    _write(repo / "outputs/evidence.json", b'{"passed":true}\n')

    state = collect_git_state(repo)
    entries = [
        {
            "role": "policy_registry",
            "kind": "source",
            "source_scope": "repo",
            "source": "src/ai_profiles.py",
            "archive_path": "source/src/ai_profiles.py",
            "sha256": _sha(repo / "src/ai_profiles.py"),
        },
        {
            "role": "runtime_source",
            "kind": "source",
            "source_scope": "repo",
            "source": "src/runtime.py",
            "archive_path": "source/src/runtime.py",
            "sha256": _sha(repo / "src/runtime.py"),
        },
        {
            "role": "runtime_config",
            "kind": "config",
            "source_scope": "repo",
            "source": "configs/runtime.json",
            "archive_path": "config/runtime.json",
            "sha256": _sha(repo / "configs/runtime.json"),
        },
        {
            "role": "runtime_model",
            "kind": "model",
            "source_scope": "repo",
            "source": "models/model.bin",
            "archive_path": "model/model.bin",
            "sha256": _sha(repo / "models/model.bin"),
        },
        {
            "role": "runtime_binary",
            "kind": "binary",
            "source_scope": "repo",
            "source": "bin/engine.dll",
            "archive_path": "binary/engine.dll",
            "sha256": _sha(repo / "bin/engine.dll"),
        },
        {
            "role": "validation_artifact",
            "kind": "artifact",
            "source_scope": "repo",
            "source": "outputs/evidence.json",
            "archive_path": "artifact/evidence.json",
            "sha256": _sha(repo / "outputs/evidence.json"),
        },
    ]
    spec_payload = {
        "schema": SPEC_SCHEMA,
        "snapshot_id": "r0-test-snapshot",
        "repository": {
            "head_commit": state["head_commit"],
            "branch": state["branch"],
            "upstream": state["upstream"],
            "upstream_commit": state["upstream_commit"],
            "require_dirty": True,
        },
        "policy_registry": {
            "role": "policy_registry",
            "sha256": entries[0]["sha256"],
            "current_profile_changed": False,
        },
        "profiles": [
            "stage19_p0",
            "stage18_p1",
            "stage9f_p2",
            "stage7_m5_r10",
        ],
        "authorization": AUTHORIZATION,
        "gates": {
            "default_gate_id": "r0a_t3_research_snapshot",
            "definitions": [
                {
                    "gate_id": "r0a_t3_research_snapshot",
                    "purpose": "reproducible_t3_research_snapshot",
                    "excluded_roles": [],
                },
                {
                    "gate_id": "r0b_exact_t4_release_runtime",
                    "purpose": "exact_t4_release_runtime_recovery",
                    "excluded_roles": [],
                },
            ],
        },
        "entries": entries,
        "notes": ["test-only R0 fixture"],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(spec_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "state": state,
        "spec_path": spec_path,
        "spec_payload": spec_payload,
    }


def _load_fixture_spec(fixture: dict[str, object]):
    return load_snapshot_spec(
        fixture["spec_path"],
        repo_root=fixture["repo"],
    )


def _rewrite_spec(fixture: dict[str, object], mutate) -> Path:
    payload = json.loads(json.dumps(fixture["spec_payload"]))
    mutate(payload)
    path = Path(fixture["spec_path"]).with_name(
        f"mutated-{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]}.json"
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_audit_and_create_are_deterministic_and_preserve_dirty_tree_and_current(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    repo = Path(fixture["repo"])
    spec = _load_fixture_spec(fixture)
    state_before = collect_git_state(repo)
    registry_before = _sha(repo / "src/ai_profiles.py")

    audit = audit_snapshot(spec)
    assert audit["status"] == "go"
    assert audit["verified_entry_count"] == 6
    assert audit["current_profile_changed"] is False

    first = tmp_path / "snapshot-a"
    second = tmp_path / "snapshot-b"
    first_result = create_snapshot(spec, first)
    second_result = create_snapshot(spec, second)

    assert first_result["status"] == second_result["status"] == "pass"
    assert first_result["payload_archive_sha256"] == second_result[
        "payload_archive_sha256"
    ]
    assert _sha(first / ARCHIVE_NAME) == _sha(second / ARCHIVE_NAME)
    assert verify_snapshot(first)["status"] == "pass"
    assert collect_git_state(repo) == state_before
    assert _sha(repo / "src/ai_profiles.py") == registry_before

    with zipfile.ZipFile(first / ARCHIVE_NAME) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert names[:2] != [ARCHIVE_SPEC_NAME, ARCHIVE_MANIFEST_NAME]
        assert ARCHIVE_SPEC_NAME in names
        assert ARCHIVE_MANIFEST_NAME in names
        assert len(names) == len(set(names)) == 8
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_missing_or_pin_mismatch_is_no_go_and_create_writes_nothing(tmp_path: Path):
    fixture = _fixture(tmp_path)
    repo = Path(fixture["repo"])
    binary = repo / "bin/engine.dll"
    binary.unlink()
    spec = _load_fixture_spec(fixture)

    missing = audit_snapshot(spec)
    assert missing["status"] == "no_go"
    assert {
        issue["code"] for issue in missing["issues"]
    } >= {"missing_required_entry"}
    output = tmp_path / "missing-output"
    with pytest.raises(SnapshotNoGoError):
        create_snapshot(spec, output)
    assert not output.exists()

    _write(binary, b"wrong-native-engine")
    mismatch = audit_snapshot(spec)
    assert mismatch["status"] == "no_go"
    assert {issue["code"] for issue in mismatch["issues"]} >= {
        "entry_pin_mismatch"
    }


def test_r0a_can_go_while_r0b_fails_closed_on_missing_release_binary(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)

    def mutate(payload):
        payload["entries"].append(
            {
                "role": "m30_windows_exact_runtime",
                "kind": "binary",
                "source_scope": "repo",
                "source": "bin/missing-m30.dll",
                "archive_path": "binary/windows/missing-m30.dll",
                "sha256": "a" * 64,
            }
        )
        payload["gates"]["definitions"][0]["excluded_roles"] = [
            "m30_windows_exact_runtime"
        ]

    path = _rewrite_spec(fixture, mutate)
    spec = load_snapshot_spec(path, repo_root=fixture["repo"])

    r0a = audit_snapshot(spec)
    assert r0a["selected_gate_id"] == "r0a_t3_research_snapshot"
    assert r0a["status"] == "go"
    assert r0a["entry_count"] == 6
    assert {
        issue["role"] for issue in r0a["deferred_entry_issues"]
    } == {"m30_windows_exact_runtime"}
    assert {
        result["gate_id"]: result["status"]
        for result in r0a["gate_results"]
    } == {
        "r0a_t3_research_snapshot": "go",
        "r0b_exact_t4_release_runtime": "no_go",
    }

    r0b = audit_snapshot(spec, gate_id="r0b_exact_t4_release_runtime")
    assert r0b["status"] == "no_go"
    assert {
        (issue["code"], issue.get("role")) for issue in r0b["issues"]
    } >= {("missing_required_entry", "m30_windows_exact_runtime")}

    r0a_output = tmp_path / "r0a-snapshot"
    result = create_snapshot(spec, r0a_output)
    assert result["status"] == "pass"
    assert result["selected_gate_id"] == "r0a_t3_research_snapshot"
    assert verify_snapshot(r0a_output)["entry_count"] == 6

    r0b_output = tmp_path / "r0b-snapshot"
    with pytest.raises(SnapshotNoGoError):
        create_snapshot(
            spec,
            r0b_output,
            gate_id="r0b_exact_t4_release_runtime",
        )
    assert not r0b_output.exists()


@pytest.mark.parametrize("duplicate_field", ["role", "source", "archive_path"])
def test_duplicate_contract_fields_fail_closed(tmp_path: Path, duplicate_field: str):
    fixture = _fixture(tmp_path)

    def mutate(payload):
        payload["entries"][1][duplicate_field] = payload["entries"][0][
            duplicate_field
        ]

    path = _rewrite_spec(fixture, mutate)
    with pytest.raises(SnapshotContractError, match="duplicate"):
        load_snapshot_spec(path, repo_root=fixture["repo"])


def test_unknown_or_policy_registry_gate_exclusion_fails_closed(tmp_path: Path):
    fixture = _fixture(tmp_path)

    unknown = _rewrite_spec(
        fixture,
        lambda payload: payload["gates"]["definitions"][0][
            "excluded_roles"
        ].append("unknown-role"),
    )
    with pytest.raises(SnapshotContractError, match="unknown roles"):
        load_snapshot_spec(unknown, repo_root=fixture["repo"])

    policy = _rewrite_spec(
        fixture,
        lambda payload: payload["gates"]["definitions"][0][
            "excluded_roles"
        ].append("policy_registry"),
    )
    with pytest.raises(SnapshotContractError, match="policy registry"):
        load_snapshot_spec(policy, repo_root=fixture["repo"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "../outside.py"),
        ("archive_path", "../escape.bin"),
        ("archive_path", "/absolute.bin"),
        ("archive_path", "payload\\windows.bin"),
    ],
)
def test_traversal_and_nonportable_paths_fail_closed(
    tmp_path: Path, field: str, value: str
):
    fixture = _fixture(tmp_path)

    def mutate(payload):
        payload["entries"][1][field] = value

    path = _rewrite_spec(fixture, mutate)
    with pytest.raises(SnapshotContractError):
        load_snapshot_spec(path, repo_root=fixture["repo"])


def test_symlinked_input_is_no_go(tmp_path: Path):
    fixture = _fixture(tmp_path)
    repo = Path(fixture["repo"])
    target = repo / "src/runtime-real.py"
    _write(target, (repo / "src/runtime.py").read_bytes())
    link = repo / "src/runtime-link.py"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    def mutate(payload):
        payload["entries"][1]["source"] = "src/runtime-link.py"

    path = _rewrite_spec(fixture, mutate)
    spec = load_snapshot_spec(path, repo_root=repo)
    audit = audit_snapshot(spec)
    assert audit["status"] == "no_go"
    assert {issue["code"] for issue in audit["issues"]} >= {"symlink_forbidden"}


def test_existing_output_is_never_overwritten(tmp_path: Path):
    fixture = _fixture(tmp_path)
    spec = _load_fixture_spec(fixture)
    output = tmp_path / "snapshot"
    create_snapshot(spec, output)
    marker = (output / "SNAPSHOT_READY.json").read_bytes()

    with pytest.raises(FileExistsError):
        create_snapshot(spec, output)
    assert (output / "SNAPSHOT_READY.json").read_bytes() == marker


@pytest.mark.parametrize("target", [ARCHIVE_NAME, "payload_manifest.json"])
def test_snapshot_tamper_is_detected(tmp_path: Path, target: str):
    fixture = _fixture(tmp_path)
    spec = _load_fixture_spec(fixture)
    output = tmp_path / "snapshot"
    create_snapshot(spec, output)
    path = output / target
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(SnapshotVerificationError):
        verify_snapshot(output)


def test_unignored_output_inside_repo_is_rejected_without_dirtying_repo(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    repo = Path(fixture["repo"])
    spec = _load_fixture_spec(fixture)
    state_before = collect_git_state(repo)
    output = repo / "not-ignored/snapshot"

    with pytest.raises(SnapshotContractError, match="Git-ignored"):
        create_snapshot(spec, output)
    assert not output.exists()
    assert collect_git_state(repo) == state_before
