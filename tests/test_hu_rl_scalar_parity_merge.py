from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ofc_regular.hu_rl_scalar_parity import (
    PRIVILEGED_AUDIT_ROLE,
    SCALAR_PARITY_SUMMARY_SCHEMA,
    build_scalar_parity_run_contract,
    pin_scalar_parity_binary,
    scalar_parity_run_contract_digest,
)
from ofc_regular.hu_rl_scalar_parity_merge import (
    HuRlScalarParityMergeError,
    SCALAR_PARITY_MERGE_SCHEMA,
    merge_scalar_parity_shards,
    validate_merge_runtime_entrypoint,
    write_merge_receipt,
)


def _digest(value: str | bytes) -> str:
    encoded = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _payload(
    start: int,
    hands: int,
    *,
    contract: dict[str, object],
) -> dict[str, object]:
    proofs = [
        {
            "hand_ordinal": ordinal,
            "request_sha256": _digest(f"request-{ordinal}"),
            "result_sha256": _digest(f"result-{ordinal}"),
            "decision_count": 10,
            "exact": True,
        }
        for ordinal in range(start, start + hands)
    ]
    return {
        "schema": SCALAR_PARITY_SUMMARY_SCHEMA,
        "status": "exact_python_rust_scalar_parity",
        "hands": hands,
        "hand_start": start,
        "hand_end_exclusive": start + hands,
        "global_hands": contract["global_hands"],
        "seed": contract["seed"],
        "workers": 2,
        "total_decisions": hands * 10,
        "unique_request_count": hands,
        "proofs": proofs,
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
        "artifact_written": True,
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    source_binary = tmp_path / "engine.exe"
    source_binary.write_bytes(b"pinned binary")
    binary = pin_scalar_parity_binary(source_binary, tmp_path / "pins")
    profile = tmp_path / "ai_profiles.py"
    profile.write_bytes(b"pinned profile")
    contract = build_scalar_parity_run_contract(
        binary,
        profile_path=profile,
        source_root=root,
        seed=17,
        global_hands=4,
    )
    shard0 = tmp_path / "shard_00000_00001.json"
    shard1 = tmp_path / "shard_00002_00003.json"
    _write(shard0, _payload(0, 2, contract=contract))
    _write(shard1, _payload(2, 2, contract=contract))
    return {
        "root": root,
        "shards": [shard1, shard0],
        "binary": binary,
        "profile": profile,
        "contract": contract,
        "contract_sha": scalar_parity_run_contract_digest(contract),
    }


def _merge(fixture: dict[str, object], **overrides: object) -> dict[str, object]:
    kwargs = {
        "binary_path": fixture["binary"],
        "profile_path": fixture["profile"],
        "expected_seed": 17,
        "expected_hands": 4,
        "expected_run_contract_sha256": fixture["contract_sha"],
        "source_root": fixture["root"],
        **overrides,
    }
    return merge_scalar_parity_shards(fixture["shards"], **kwargs)  # type: ignore[arg-type]


def test_merge_binds_contiguous_proofs_frozen_contract_and_privileged_class(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _merge(fixture)

    assert result["schema"] == SCALAR_PARITY_MERGE_SCHEMA
    assert result["status"] == "exact_python_rust_scalar_parity_merged"
    assert result["hands"] == 4
    assert result["total_decisions"] == 40
    assert result["shard_count"] == 2
    assert result["unique_request_count"] == 4
    assert result["unique_result_count"] == 4
    assert result["run_contract_sha256"] == fixture["contract_sha"]
    assert result["native_binary"] == fixture["contract"]["native_binary"]  # type: ignore[index]
    assert result["profile_registry"] == fixture["contract"][  # type: ignore[index]
        "profile_registry"
    ]
    assert set(result["merger_sources"]) == {
        "scripts/merge_hu_rl_scalar_parity.py",
        "src/ofc_regular/hu_rl_scalar_parity_merge.py",
    }
    assert result["artifact_role"] == PRIVILEGED_AUDIT_ROLE
    assert result["contains_raw_cross_actor_private_information"] is False
    assert result["contains_reconstructable_hidden_oracle_state"] is True
    assert result["policy_input_eligible"] is False
    assert result["replay_eligible"] is False
    assert result["training_eligible"] is False
    assert result["current_profile_changed"] is False
    assert [row["hand_start"] for row in result["shards"]] == [0, 2]
    assert len(result["self_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    (
        "gap",
        "unsafe",
        "duplicate",
        "duplicate_field",
        "float",
        "nan",
        "contract",
        "binary",
        "profile",
        "source_root",
    ),
)
def test_merge_rejects_tamper_rebuild_drift_and_non_strict_types(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    overrides: dict[str, object] = {}
    if mutation == "gap":
        _write(
            fixture["shards"][0],  # type: ignore[index]
            _payload(3, 1, contract=fixture["contract"]),  # type: ignore[arg-type]
        )
    elif mutation == "unsafe":
        path = fixture["shards"][1]  # type: ignore[index]
        payload = _payload(0, 2, contract=fixture["contract"])  # type: ignore[arg-type]
        payload["training_eligible"] = True
        _write(path, payload)
    elif mutation == "duplicate":
        path = fixture["shards"][0]  # type: ignore[index]
        payload = _payload(2, 2, contract=fixture["contract"])  # type: ignore[arg-type]
        payload["proofs"][0]["request_sha256"] = _digest("request-0")  # type: ignore[index]
        _write(path, payload)
    elif mutation == "duplicate_field":
        path = fixture["shards"][1]  # type: ignore[index]
        path.write_text('{"schema":"x","schema":"y"}\n', encoding="ascii")
    elif mutation == "float":
        path = fixture["shards"][1]  # type: ignore[index]
        payload = _payload(0, 2, contract=fixture["contract"])  # type: ignore[arg-type]
        payload["seed"] = 17.0
        _write(path, payload)
    elif mutation == "nan":
        path = fixture["shards"][1]  # type: ignore[index]
        path.write_text('{"seed":NaN}\n', encoding="ascii")
    elif mutation == "contract":
        path = fixture["shards"][0]  # type: ignore[index]
        payload = _payload(2, 2, contract=fixture["contract"])  # type: ignore[arg-type]
        payload["run_contract_sha256"] = "0" * 64
        _write(path, payload)
    elif mutation == "binary":
        binary = fixture["binary"]  # type: ignore[assignment]
        binary.chmod(binary.stat().st_mode | stat.S_IWUSR)
        binary.write_bytes(b"rebuilt during evidence run")
    elif mutation == "profile":
        fixture["profile"].write_bytes(b"changed profile")  # type: ignore[union-attr]
    elif mutation == "source_root":
        overrides["source_root"] = tmp_path / "wrong-source-root"

    with pytest.raises(HuRlScalarParityMergeError):
        _merge(fixture, **overrides)


def test_merge_rejects_rebinding_old_shards_to_a_new_current_binary(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    replacement = tmp_path / "replacement.exe"
    replacement.write_bytes(b"different executable")
    replacement_pinned = pin_scalar_parity_binary(replacement, tmp_path / "replacement-pins")

    with pytest.raises(HuRlScalarParityMergeError, match="drifted"):
        _merge(fixture, binary_path=replacement_pinned)


@pytest.mark.parametrize("legacy_version", ("v1", "v2"))
def test_v1_and_v2_shards_are_ineligible_for_v3_merge(
    tmp_path: Path, legacy_version: str
) -> None:
    fixture = _fixture(tmp_path)
    legacy = {
        "schema": f"regular_ofc_hu_rl_scalar_parity_summary_{legacy_version}",
        "status": "exact_python_rust_scalar_parity",
    }
    _write(fixture["shards"][1], legacy)  # type: ignore[index]
    with pytest.raises(HuRlScalarParityMergeError, match="v3 validation"):
        _merge(fixture)


def test_merge_rejects_copied_source_root_and_entrypoint(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    copied_root = tmp_path / "copied-source"
    for relpath in (
        "scripts/merge_hu_rl_scalar_parity.py",
        "src/ofc_regular/hu_rl_scalar_parity_merge.py",
    ):
        destination = copied_root / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relpath).read_bytes())  # type: ignore[operator]

    with pytest.raises(HuRlScalarParityMergeError, match="executing merger source"):
        _merge(fixture, source_root=copied_root)
    with pytest.raises(HuRlScalarParityMergeError, match="merge CLI"):
        validate_merge_runtime_entrypoint(
            root,
            copied_root / "scripts/merge_hu_rl_scalar_parity.py",
        )


def test_write_merge_receipt_is_atomic_resume_and_mismatch_safe(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = _merge(fixture)
    target = tmp_path / "receipt.json"
    write_merge_receipt(target, receipt)
    frozen = target.read_bytes()
    write_merge_receipt(target, receipt)
    assert target.read_bytes() == frozen

    second_path = fixture["shards"][0]  # type: ignore[index]
    second_payload = _payload(2, 2, contract=fixture["contract"])  # type: ignore[arg-type]
    second_payload["proofs"][0]["result_sha256"] = "f" * 64  # type: ignore[index]
    _write(second_path, second_payload)
    changed = _merge(fixture)
    with pytest.raises(HuRlScalarParityMergeError):
        write_merge_receipt(target, changed)
    assert target.read_bytes() == frozen


def test_equal_result_digest_is_allowed_for_different_hidden_tails(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    second = fixture["shards"][0]  # type: ignore[index]
    payload = _payload(2, 2, contract=fixture["contract"])  # type: ignore[arg-type]
    payload["proofs"][0]["result_sha256"] = _digest("result-0")  # type: ignore[index]
    _write(second, payload)

    receipt = _merge(fixture)
    assert receipt["unique_request_count"] == 4
    assert receipt["unique_result_count"] == 3


def test_merge_script_redacts_runtime_failure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    secret = tmp_path / "opponent-private-discard-AS"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/merge_hu_rl_scalar_parity.py"),
            "--shard-dir",
            str(secret),
            "--binary",
            str(secret / "binary.exe"),
            "--profile",
            str(secret / "profile.py"),
            "--source-root",
            str(root),
            "--expected-seed",
            "17",
            "--expected-hands",
            "4",
            "--expected-run-contract-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "receipt.json"),
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
        '{"error":"scalar parity merge failed","status":"error"}\n'
    )
    assert str(secret) not in completed.stderr


def test_merge_script_persists_then_byte_validates_resume(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    output = tmp_path / "merged.json"
    command = [
        sys.executable,
        str(root / "scripts/merge_hu_rl_scalar_parity.py"),
        "--shard-dir",
        str(tmp_path),
        "--binary",
        str(fixture["binary"]),
        "--profile",
        str(fixture["profile"]),
        "--source-root",
        str(root),
        "--expected-seed",
        "17",
        "--expected-hands",
        "4",
        "--expected-run-contract-sha256",
        str(fixture["contract_sha"]),
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
    assert "run_contract" not in stdout_receipt
    assert "seed" not in stdout_receipt
    assert stdout_receipt["contains_reconstructable_hidden_oracle_state"] is True
    assert output.read_bytes() == frozen
