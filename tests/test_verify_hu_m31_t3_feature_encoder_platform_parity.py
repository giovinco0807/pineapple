from __future__ import annotations

import copy
import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts import verify_hu_m31_t3_feature_encoder_platform_parity as subject


REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_SOURCE = REPO_ROOT / "rust/ofc_stage3_feature_encoder/src/lib.rs"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"
AI_PROFILES = REPO_ROOT / "src/ofc_regular/ai_profiles.py"


class _FakeFunction:
    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        self.argtypes: list[Any] = []
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        return self.function(*args)


class _FakeLibrary:
    def __init__(self, marker: float = 1.25) -> None:
        self.marker = marker
        self.ofc_stage3_feature_dim = _FakeFunction(lambda: subject.FEATURE_DIM)
        self.ofc_stage3_encode = _FakeFunction(self._encode)

    def _encode(self, *args: Any) -> int:
        features = args[14]
        row_to_state = args[15]
        row_to_action = args[16]
        profile_pointer = args[17]
        features[0] = self.marker
        features[-1] = -self.marker
        for row_index in range(subject.ROW_COUNT):
            state_index, action_index = divmod(row_index, subject.ACTION_COUNT)
            row_to_state[row_index] = state_index
            row_to_action[row_index] = action_index
        profile = ctypes.cast(
            profile_pointer, ctypes.POINTER(subject._RustProfile)
        ).contents
        profile.rows = subject.ROW_COUNT
        return 0


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(subject.canonical_bytes(value))


def _probe(platform: str, *, output_sha: str = "a" * 64) -> dict[str, Any]:
    suffix = ".dll" if platform == "windows" else ".so"
    library_sha = (
        "b" * 64 if platform == "windows" else subject.EXPECTED_LINUX_LIBRARY_SHA256
    )
    library_bytes = 199_680 if platform == "windows" else 489_144
    return {
        "schema": subject.PROBE_SCHEMA,
        "status": "fixed_512_row_feature_encoder_probe_complete",
        "platform_label": platform,
        "generator": subject._generator_record(),
        "build_inputs": subject._default_build_inputs(),
        "library": {
            "path": f"/frozen/ofc_stage3_feature_encoder{suffix}",
            "bytes": library_bytes,
            "sha256": library_sha,
        },
        "input_bytes": subject.EXPECTED_INPUT_BYTES,
        "input_sha256": subject.EXPECTED_INPUT_SHA256,
        "encoder_status": 0,
        "feature_dim": subject.FEATURE_DIM,
        "row_count": subject.ROW_COUNT,
        "output_encoding": subject.OUTPUT_ENCODING,
        "output_bytes": subject.EXPECTED_OUTPUT_BYTES,
        "output_sha256": output_sha,
    }


def _lock_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ai_profiles = subject._file_record(AI_PROFILES, "ai_profiles")
    claim = {
        "schema": subject.CLAIM_SCHEMA,
        "status": (
            "global_one_shot_claim_persisted_before_lock_root_touch_"
            "crash_consumes_claim"
        ),
        "scope": "candidate02_performance_lock_roots_only",
        "opened_unix_ns": 1,
        "global_claim_path": "/frozen/GLOBAL_PERFORMANCE_LOCK_CLAIM.json",
        "lock_output_directory": "/frozen/roots-open",
        "precontent_plan": {},
        "development_go": {},
        "step6d_contract": {},
        "lock_run_contract": {},
        "lock_run_contract_digest": "2" * 64,
        "runner_source": {},
        "ai_profiles_current": ai_profiles,
        "accepted_binaries": {
            "candidate": {},
            "reference": {},
            "feature_encoder": {
                "path": "/frozen/libofc_stage3_feature_encoder.so",
                "bytes": 489_144,
                "sha256": subject.EXPECTED_LINUX_LIBRARY_SHA256,
            },
        },
        "model_inputs": {},
        "root_generator_inputs": {},
        "image": {},
        "allocation": {},
        "seed_contract": {},
        "startup_source": {},
        "restrictions": {
            "alternate_seed_allowed": False,
            "reseed_allowed": False,
            "current_profile_resolution_allowed": False,
            "opponent_private_discards_allowed": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_activation_allowed": False,
        },
    }
    root_hashes = [
        hashlib.sha256(f"root-{index}".encode("ascii")).hexdigest()
        for index in range(100)
    ]
    materialization = {
        "schema": subject.MATERIALIZATION_SCHEMA,
        "status": "all_100_lock_roots_materialized_same_identity",
        "global_claim_sha256": subject.canonical_sha256(claim),
        "plan_sha256": "1" * 64,
        "run_contract_digest": "2" * 64,
        "hand_indices": list(range(100)),
        "root_count": 100,
        "root_artifact_sha256": root_hashes,
        "aggregate_root_sha256": subject.canonical_sha256(root_hashes),
        "same_identity_resume_only": True,
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    seal = {
        "schema": subject.SEAL_SCHEMA,
        "status": "sealed_100_disjoint_hidden_safe_performance_lock_roots",
        "global_claim_sha256": subject.canonical_sha256(claim),
        "materialization_sha256": subject.canonical_sha256(materialization),
        "plan_sha256": materialization["plan_sha256"],
        "run_contract_digest": materialization["run_contract_digest"],
        "hand_indices": list(range(100)),
        "root_count": 100,
        "observation_count": 200,
        "profile_counts": {},
        "seat_counts": {"first": 100, "second": 100},
        "root_artifact_sha256": root_hashes,
        "aggregate_root_sha256": materialization["aggregate_root_sha256"],
        "root_topology_sha256": "3" * 64,
        "observation_fingerprint_sha256": "4" * 64,
        "root_artifact_unique": True,
        "observation_fingerprint_unique": True,
        "development_comparison": {},
        "visibility": {
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        },
        "selection_inputs": {
            "all_100_preregistered_hands_used": True,
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
        },
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    return claim, materialization, seal


def _fixture_paths(
    tmp_path: Path,
    *,
    mutate: (
        Callable[
            [
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ],
            None,
        ]
        | None
    ) = None,
) -> dict[str, Path]:
    windows = _probe("windows")
    linux = _probe("linux")
    claim, materialization, seal = _lock_artifacts()
    if mutate is not None:
        mutate(windows, linux, claim, materialization, seal)
    paths = {
        "windows_probe": tmp_path / subject.WINDOWS_PROBE_NAME,
        "linux_probe": tmp_path / subject.LINUX_PROBE_NAME,
        "global_claim": tmp_path / "GLOBAL_PERFORMANCE_LOCK_CLAIM.json",
        "materialization": tmp_path / "materialization.json",
        "seal": tmp_path / "seal.json",
        "output": tmp_path / subject.PARITY_RECEIPT_NAME,
    }
    for key, value in (
        ("windows_probe", windows),
        ("linux_probe", linux),
        ("global_claim", claim),
        ("materialization", materialization),
        ("seal", seal),
    ):
        _write(paths[key], value)
    return paths


def _compare(paths: dict[str, Path]) -> dict[str, Any]:
    return subject.write_comparison_receipt(
        windows_probe_path=paths["windows_probe"],
        linux_probe_path=paths["linux_probe"],
        global_claim_path=paths["global_claim"],
        materialization_path=paths["materialization"],
        seal_path=paths["seal"],
        rust_source_path=RUST_SOURCE,
        cargo_lock_path=CARGO_LOCK,
        ai_profiles_path=AI_PROFILES,
        output_path=paths["output"],
    )


def test_fixed_generator_contract_is_64_by_8_and_stable() -> None:
    assert subject.STATE_COUNT == 64
    assert subject.ACTION_COUNT == 8
    assert subject.ROW_COUNT == 512
    assert subject.EXPECTED_INPUT_BYTES == 8661
    assert (
        subject.EXPECTED_INPUT_SHA256
        == "58d74968abf6a9ec9bb097fac00b7b3ecf0f3f58c554c973284d74edf460a535"
    )
    assert subject.EXPECTED_OUTPUT_BYTES == 2_206_720
    assert subject.PROBE_GENERATOR_CONTRACT["opponent_private_discards_used"] is False
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "from ofc_regular" not in source


def test_probe_is_canonical_write_once_and_checks_row_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = tmp_path / "encoder.dll"
    library.write_bytes(b"fake-dll")
    monkeypatch.setattr(subject, "_load_library", lambda _path: _FakeLibrary())
    output = tmp_path / subject.WINDOWS_PROBE_NAME

    probe = subject.write_probe(library, "windows", output)

    assert probe["row_count"] == 512
    assert probe["output_bytes"] == 2_206_720
    assert output.read_bytes() == subject.canonical_bytes(probe)
    with pytest.raises(FileExistsError):
        subject.write_probe(library, "windows", output)


def test_actual_windows_dll_probe_when_built(tmp_path: Path) -> None:
    library = REPO_ROOT / "target/release/ofc_stage3_feature_encoder.dll"
    if not library.is_file():
        pytest.skip("Windows Stage3 feature encoder DLL is not built")

    probe = subject.write_probe(
        library, "windows", tmp_path / subject.WINDOWS_PROBE_NAME
    )

    assert probe["encoder_status"] == 0
    assert probe["row_count"] == 512
    assert probe["output_sha256"] == (
        "a1cea3acc7b6fc5cf79dff6e5a61d3970f3ac2589f79623b4a2fd22ab83ea5e3"
    )


def test_compare_binds_probe_files_script_and_lock_chain_write_once(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)

    receipt = _compare(paths)

    assert receipt["platform_parity"]["input_bit_exact"] is True
    assert receipt["platform_parity"]["output_bit_exact"] is True
    assert receipt["lock_chain"]["root_count"] == 100
    assert receipt["lock_chain"]["reseeded"] is False
    assert receipt["mutation_scope"]["current_profile_changed"] is False
    assert receipt["generator"]["script"]["path"].endswith(
        "verify_hu_m31_t3_feature_encoder_platform_parity.py"
    )
    for key in (
        "windows_probe",
        "linux_probe",
        "global_claim",
        "materialization",
        "seal",
    ):
        assert receipt["artifacts"][key]["sha256"] == subject.sha256_file(paths[key])
    assert paths["output"].read_bytes() == subject.canonical_bytes(receipt)
    with pytest.raises(FileExistsError):
        _compare(paths)


def test_compare_rejects_output_difference_before_writing(tmp_path: Path) -> None:
    def mutate(
        _windows: dict[str, Any],
        linux: dict[str, Any],
        _claim: dict[str, Any],
        _materialization: dict[str, Any],
        _seal: dict[str, Any],
    ) -> None:
        linux["output_sha256"] = "c" * 64

    paths = _fixture_paths(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="output is not bit-exact"):
        _compare(paths)
    assert not paths["output"].exists()


@pytest.mark.parametrize("tamper", ("reseed", "current", "chain", "linux_hash"))
def test_compare_rejects_lock_or_binary_tamper(tmp_path: Path, tamper: str) -> None:
    def mutate(
        _windows: dict[str, Any],
        linux: dict[str, Any],
        claim: dict[str, Any],
        materialization: dict[str, Any],
        seal: dict[str, Any],
    ) -> None:
        if tamper == "reseed":
            materialization["reseeded"] = True
        elif tamper == "current":
            seal["current_profile_changed"] = True
        elif tamper == "chain":
            seal["materialization_sha256"] = "0" * 64
        else:
            linux["library"]["sha256"] = "d" * 64
            claim["accepted_binaries"]["feature_encoder"]["sha256"] = "d" * 64

    paths = _fixture_paths(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        _compare(paths)
    assert not paths["output"].exists()


def test_compare_rejects_probe_script_tamper(tmp_path: Path) -> None:
    def mutate(
        windows: dict[str, Any],
        _linux: dict[str, Any],
        _claim: dict[str, Any],
        _materialization: dict[str, Any],
        _seal: dict[str, Any],
    ) -> None:
        windows["generator"]["script"]["sha256"] = "e" * 64

    paths = _fixture_paths(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="probe script content changed"):
        _compare(paths)


def test_compare_rejects_noncanonical_probe(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    value = json.loads(paths["windows_probe"].read_text(encoding="utf-8"))
    paths["windows_probe"].write_text(
        json.dumps(value, indent=2), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="not canonical JSON"):
        _compare(paths)
