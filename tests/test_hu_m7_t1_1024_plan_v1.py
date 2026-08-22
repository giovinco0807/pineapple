from __future__ import annotations

import hashlib
import io
import json
import pathlib
import tarfile
import zipfile

import pytest

from ofc_regular import hu_m7_t1_1024_plan_v1 as subject
from ofc_regular.hu_m31_label_gen_worker_v1 import load_plan as load_worker_plan


MEMBERS = {
    "engine_library_sha256": "runtime/native/libofc_hu_m3_engine.so",
    "feature_encoder_library_sha256": (
        "runtime/native/libofc_stage3_feature_encoder.so"
    ),
    "fast_t0_second_model_sha256": "runtime/weights/fast_t0_second_v1.bin",
    "fast_t1_first_model_sha256": "runtime/weights/fast_t1_first_v1.bin",
    "fast_t1_second_model_sha256": "runtime/weights/fast_t1_second_v1.bin",
    "fast_t2_first_model_sha256": "runtime/weights/fast_t2_first_v1.bin",
    "fast_t2_second_model_sha256": "runtime/weights/fast_t2_second_v1.bin",
    "t0_first_model_sha256": "runtime/weights/t0first_model_v1.bin",
    "t0_second_model_sha256": "runtime/weights/t0_model_v1.bin",
    "t1_first_model_sha256": "runtime/weights/t1first_model_v1.bin",
    "t1_second_model_sha256": "runtime/weights/t1_model_v1.bin",
    "t2_first_model_sha256": "runtime/weights/t2first_model_v1.bin",
    "t2_first_model_v2_sha256": "runtime/weights/t2first_model_v2.bin",
    "t2_second_model_sha256": "runtime/weights/t2_model_v1.bin",
    "t2_second_model_v2_sha256": "runtime/weights/t2_model_v2.bin",
    "t3_first_model_sha256": "runtime/weights/t3first_model_v1.bin",
    "t3_first_model_v2_sha256": "runtime/weights/t3first_model_v2.bin",
    "t3_second_model_sha256": "runtime/weights/t3_model_v2.bin",
    "t3_second_model_v3_sha256": "runtime/weights/t3_model_v3.bin",
    "t4_model_sha256": "runtime/weights/t4_model_v5.bin",
    "t4_model_v6_sha256": "runtime/weights/t4_model_v6.bin",
    "fl_ev.config_sha256": "runtime/configs/fl_ev_regular_v4_selfplay.json",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _put_dotted(root: dict, dotted: str, value: str) -> None:
    current = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _fixture_package(
    root: pathlib.Path,
) -> tuple[pathlib.Path, subject.ExpectedPackageIdentity]:
    package = root / "package"
    package.mkdir()
    fl = {
        "rule_set": "regular",
        "include_jokers": False,
        "deck_cards": 52,
        "fl_entry_cards": {"qq": 14, "kk": 14, "aa": 14, "trips": 14},
        "fl_stay_cards": 14,
        "fl_ev": {"14": 9.6},
    }
    payloads = {
        name: (
            json.dumps(fl, sort_keys=True).encode()
            if name.endswith("fl_ev_regular_v4_selfplay.json")
            else f"fixture:{name}".encode()
        )
        for name in MEMBERS.values()
    }
    runtime = package / "runtime.tar.gz"
    with tarfile.open(runtime, "w:gz") as archive:
        for name, data in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
        model = b"legacy-profile-model"
        info = tarfile.TarInfo("runtime/models/legacy.pkl")
        info.size = len(model)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(model))
        python = b"pass\n"
        info = tarfile.TarInfo("runtime/src/ofc_regular/module.py")
        info.size = len(python)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(python))

    wheelhouse = package / "wheelhouse.zip"
    with zipfile.ZipFile(wheelhouse, "w") as archive:
        archive.writestr("fixture.whl", b"wheel")

    runtime_raw = runtime.read_bytes()
    wheelhouse_raw = wheelhouse.read_bytes()
    ledger: dict = {
        "schema": subject.LEDGER_SCHEMA,
        "runtime_archive": {
            "file": "runtime.tar.gz",
            "bytes": len(runtime_raw),
            "sha256": _sha(runtime_raw),
        },
        "wheelhouse_archive": {
            "file": "wheelhouse.zip",
            "bytes": len(wheelhouse_raw),
            "sha256": _sha(wheelhouse_raw),
        },
        "fl_ev": {
            "cards": 14,
            "value": 9.6,
            "config": "configs/fl_ev_regular_v4_selfplay.json",
        },
        "python_files": 1,
        "model_files": 1,
        "weight_files": 19,
        "wheel_files": 1,
    }
    for dotted, name in MEMBERS.items():
        _put_dotted(ledger, dotted, _sha(payloads[name]))
    ledger_raw = json.dumps(ledger, sort_keys=True).encode()
    (package / "ledger.json").write_bytes(ledger_raw)
    identity = subject.ExpectedPackageIdentity(
        ledger_sha256=_sha(ledger_raw),
        runtime_sha256=_sha(runtime_raw),
        wheelhouse_sha256=_sha(wheelhouse_raw),
    )
    return package, identity


def test_reviewed_m7v6_identity_is_pinned() -> None:
    assert subject.DEFAULT_PACKAGE_IDENTITY == subject.ExpectedPackageIdentity(
        ledger_sha256="66389e97e2bec5169db1225ea988ba3730055767cfbbf0c589a21fe8d7b0fabf",
        runtime_sha256="4ce40384c7fcdb64a507c45b36273584d1c57f1930a0ffd0ed0fc1569aea2466",
        wheelhouse_sha256="338ca072984775dc886e7c9c88d1d7e10b5363ae4aff5a7ba936eebf57d25053",
    )


def test_builds_paired_25k_1024_plans_with_expected_asymmetry(
    tmp_path: pathlib.Path,
) -> None:
    package, identity = _fixture_package(tmp_path)
    plans, audit = subject.build_plan_pair(package, expected_identity=identity)
    first, second = plans["first"], plans["second"]

    assert audit.ledger_sha256 == identity.ledger_sha256
    assert len(audit.declared_hashes) == 24
    assert len(audit.member_hashes) == 22
    for seat, plan in plans.items():
        assert plan["street"] == "T1"
        assert plan["seat"] == seat
        assert plan["samples"] == 1024
        assert plan["seeds_per_position"] == 1
        assert plan["hand_seed_base"] == 948_000_000
        assert plan["eval_seed_base"] == 11_000_000
        assert plan["fl_ev_value"] == 9.6
        assert len(plan["shards"]) == 300
        assert sum(row["count"] for row in plan["shards"]) == 25_000
        assert "probe" not in plan
        assert plan["t3_first_model"] == "weights/t3first_model_v2.bin"
        assert plan["t3_second_model"] == "weights/t3_model_v3.bin"
        assert plan["t4_model"] == "weights/t4_model_v6.bin"
        # The continuation this generation exists to promote: T2 v2, both
        # seats, on both plans.
        assert plan["t2_first_model"] == "weights/t2first_model_v2.bin"
        assert plan["t2_second_model"] == "weights/t2_model_v2.bin"
        assert plan["provenance"]["package_audit"][
            "all_declared_hashes_recomputed"
        ]

    assert first["shards"] == second["shards"]
    assert first["t1_second_model"] == "weights/t1_model_v1.bin"
    assert "t1_second_model" not in second
    assert "t1_second_model_sha256" not in second


def test_shards_are_contiguous_unique_and_larger_shards_are_last() -> None:
    shards = subject.make_shards()
    assert [row["shard_id"] for row in shards] == [
        f"{index:03d}" for index in range(300)
    ]
    assert [row["count"] for row in shards[:200]] == [83] * 200
    assert [row["count"] for row in shards[200:]] == [84] * 100
    start = 0
    for row in shards:
        assert row["start"] == start
        start += row["count"]
    assert start == 25_000


def test_sharding_records_six_bounded_cloud_waves(
    tmp_path: pathlib.Path,
) -> None:
    package, identity = _fixture_package(tmp_path)
    plans, _ = subject.build_plan_pair(package, expected_identity=identity)
    sharding = plans["first"]["provenance"]["sharding"]
    assert sharding["shards"] == 300
    assert sharding["recommended_max_live_c4_standard_8"] == 58
    assert sharding["intended_workers_per_vm"] == 6
    assert sharding["waves_at_recommended_max_live"] == 6
    assert "860" in sharding["sizing_reason"]


def test_audit_rejects_a_changed_runtime_archive(tmp_path: pathlib.Path) -> None:
    package, identity = _fixture_package(tmp_path)
    with (package / "runtime.tar.gz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(subject.PlanValidationError, match="runtime digest"):
        subject.audit_package(package, expected_identity=identity)


def test_audit_rejects_member_hash_drift_even_with_reissued_identity(
    tmp_path: pathlib.Path,
) -> None:
    package, identity = _fixture_package(tmp_path)
    ledger = json.loads((package / "ledger.json").read_text())
    ledger["t2_second_model_v2_sha256"] = "0" * 64
    raw = json.dumps(ledger, sort_keys=True).encode()
    (package / "ledger.json").write_bytes(raw)
    changed_identity = subject.ExpectedPackageIdentity(
        ledger_sha256=_sha(raw),
        runtime_sha256=identity.runtime_sha256,
        wheelhouse_sha256=identity.wheelhouse_sha256,
    )
    with pytest.raises(subject.PlanValidationError, match="t2_model_v2.bin digest"):
        subject.audit_package(package, expected_identity=changed_identity)


def test_audit_rejects_an_unhandled_new_hash_field(tmp_path: pathlib.Path) -> None:
    package, identity = _fixture_package(tmp_path)
    ledger = json.loads((package / "ledger.json").read_text())
    ledger["new_model_sha256"] = "1" * 64
    raw = json.dumps(ledger, sort_keys=True).encode()
    (package / "ledger.json").write_bytes(raw)
    changed_identity = subject.ExpectedPackageIdentity(
        ledger_sha256=_sha(raw),
        runtime_sha256=identity.runtime_sha256,
        wheelhouse_sha256=identity.wheelhouse_sha256,
    )
    with pytest.raises(subject.PlanValidationError, match="field set drifted"):
        subject.audit_package(package, expected_identity=changed_identity)


def test_audit_rejects_fl_semantic_drift(tmp_path: pathlib.Path) -> None:
    package, identity = _fixture_package(tmp_path)
    ledger = json.loads((package / "ledger.json").read_text())
    ledger["fl_ev"]["value"] = 10.227020614683454
    raw = json.dumps(ledger, sort_keys=True).encode()
    (package / "ledger.json").write_bytes(raw)
    changed_identity = subject.ExpectedPackageIdentity(
        ledger_sha256=_sha(raw),
        runtime_sha256=identity.runtime_sha256,
        wheelhouse_sha256=identity.wheelhouse_sha256,
    )
    with pytest.raises(subject.PlanValidationError, match="ledger FL EV is not 9.6"):
        subject.audit_package(package, expected_identity=changed_identity)


def test_write_is_deterministic_and_write_once(tmp_path: pathlib.Path) -> None:
    package, identity = _fixture_package(tmp_path)
    output = tmp_path / "plans"
    manifest = subject.write_plan_pair_once(
        package, output, expected_identity=identity
    )
    first_raw = (output / subject.FIRST_FILENAME).read_bytes()
    second_raw = (output / subject.SECOND_FILENAME).read_bytes()
    stored = json.loads((output / subject.MANIFEST_FILENAME).read_text())

    assert stored == manifest
    assert stored["plans"]["first"]["sha256"] == _sha(first_raw)
    assert stored["plans"]["second"]["sha256"] == _sha(second_raw)
    assert stored["paired_contract"]["current_profile_changed"] is False
    with pytest.raises(subject.PlanValidationError, match="already exists"):
        subject.write_plan_pair_once(package, output, expected_identity=identity)
    assert (output / subject.FIRST_FILENAME).read_bytes() == first_raw
    assert (output / subject.SECOND_FILENAME).read_bytes() == second_raw


def test_written_plans_pass_the_actual_label_worker_contract(
    tmp_path: pathlib.Path,
) -> None:
    package, identity = _fixture_package(tmp_path)
    output = tmp_path / "plans"
    subject.write_plan_pair_once(package, output, expected_identity=identity)

    first = load_worker_plan(output / subject.FIRST_FILENAME)
    second = load_worker_plan(output / subject.SECOND_FILENAME)
    assert (first["street"], first["seat"], first["samples"]) == (
        "T1",
        "first",
        1024,
    )
    assert (second["street"], second["seat"], second["samples"]) == (
        "T1",
        "second",
        1024,
    )


def test_pair_validator_rejects_second_seat_t1_reply_pin(
    tmp_path: pathlib.Path,
) -> None:
    package, identity = _fixture_package(tmp_path)
    plans, _ = subject.build_plan_pair(package, expected_identity=identity)
    plans["second"]["t1_second_model"] = "weights/t1_model_v1.bin"
    plans["second"]["t1_second_model_sha256"] = plans["first"][
        "t1_second_model_sha256"
    ]
    with pytest.raises(subject.PlanValidationError, match="unreachable"):
        subject.validate_plan_pair(plans)


def test_pair_validator_rejects_fast_reply_pins(tmp_path: pathlib.Path) -> None:
    package, identity = _fixture_package(tmp_path)
    plans, _ = subject.build_plan_pair(package, expected_identity=identity)
    plans["first"]["fast_t2_first_model"] = "weights/fast_t2_first_v1.bin"
    with pytest.raises(subject.PlanValidationError, match="full-precision"):
        subject.validate_plan_pair(plans)
