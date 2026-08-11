from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular import hu_m31_t3_step6d_fresh_quality_local_staging_v1 as subject
from ofc_regular import hu_m31_t3_step6d_fresh_quality_transport_v1 as transport
from ofc_regular import hu_m31_t3_step6d_fresh_quality_v1 as quality


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001/"
    "package_src/native/candidate/release/libofc_hu_m3_engine.so"
)
FEATURE = (
    ROOT
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001/"
    "package_src/target/release/libofc_stage3_feature_encoder.so"
)
STARTUP = ROOT / "scripts/startup_hu_m31_t3_step6d_fresh_quality_v1.sh"
PROFILE = ROOT / "src/ofc_regular/ai_profiles.py"

_AUTHORIZATION = {
    "schema": quality.PERFORMANCE_RECEIPT_SCHEMA,
    "status": "qualified",
    "decision": quality.QUALIFIED_DECISION,
    "receipt_sha256": "a" * 64,
    "performance_lock_qualified": True,
    "quality_pilot_authorized": True,
    "performance_lock_finalized": True,
    "one_shot_lock_consumed": True,
    "current_profile_changed": False,
}


def _observations(pair_index: int, phase: str) -> tuple[ActorObservation, ...]:
    deck = create_deck(shuffle=False)
    if phase == quality.CONFIRMATION_PHASE:
        deck = list(reversed(deck))
    offset = pair_index % len(deck)
    deck = deck[offset:] + deck[:offset]
    first_board = Board.from_rows(
        top=deck[0:2], middle=deck[2:6], bottom=deck[6:9]
    )
    second_board = Board.from_rows(
        top=deck[9:11], middle=deck[11:15], bottom=deck[15:18]
    )
    first = ActorObservation(
        hero_board=first_board,
        opponent_public_board=second_board,
        dealt_cards=tuple(deck[20:23]),
        hero_private_discards=tuple(deck[18:20]),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    first_after = Board.from_rows(
        top=first_board.top,
        middle=(*first_board.middle, deck[20]),
        bottom=(*first_board.bottom, deck[21]),
    )
    second = ActorObservation(
        hero_board=second_board,
        opponent_public_board=first_after,
        dealt_cards=tuple(deck[25:28]),
        hero_private_discards=tuple(deck[23:25]),
        seat="second",
        street="T3",
        to_act_order="second",
    )
    return first, second


def _wheelhouse(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    payload = b"offline-local-staging-smoke-wheel"
    filename = "fresh_quality_smoke-1.0-py3-none-any.whl"
    archive = directory / "wheelhouse.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.writestr(filename, payload)
    entries = [
        {
            "filename": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "distribution": "fresh-quality-smoke",
            "version": "1.0",
            "tags": ["py3-none-any"],
        }
    ]
    manifest = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1",
        "status": "complete_hash_pinned_offline_wheelhouse",
        "requirements_sha256": "0" * 64,
        "python_abi": "cp311",
        "target_os": "linux",
        "target_architecture": "x86_64",
        "network_install_allowed": False,
        "entries": entries,
        "entry_count": 1,
        "entries_sha256": hashlib.sha256(
            quality.canonical_bytes(entries) + b"\n"
        ).hexdigest(),
    }
    manifest_path = directory / "wheelhouse_manifest.json"
    manifest_path.write_bytes(quality.canonical_bytes(manifest) + b"\n")
    return archive, manifest_path


def _authorized_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(
        quality,
        "_load_performance_authorization",
        lambda _path: dict(_AUTHORIZATION),
    )
    monkeypatch.setattr(quality.step6d_v1, "load_model_bundle", lambda *_a, **_k: object())
    monkeypatch.setattr(quality.step6d_v1, "_absolute_model_paths", lambda *_a: object())
    monkeypatch.setattr(
        quality,
        "_generate_observations",
        lambda *, repository_root, row, bundle: _observations(
            row["pair_index"], row["phase"]
        ),
    )
    receipt = tmp_path / "qualified-v4-receipt.json"
    receipt.write_bytes(quality.canonical_bytes({"test_receipt": True}))
    wheel, wheel_manifest = _wheelhouse(tmp_path / "accepted-wheelhouse")
    return receipt, wheel, wheel_manifest


def _build(
    *,
    performance: Path,
    wheel: Path,
    wheel_manifest: Path,
    output: Path,
) -> dict:
    return subject.build_local_staging(
        repository_root=ROOT,
        performance_receipt_path=performance,
        run_name="regular-hu-m31-t3-fqv1-local-smoke-001",
        candidate_library_path=CANDIDATE,
        feature_encoder_path=FEATURE,
        wheelhouse_archive_path=wheel,
        wheelhouse_manifest_path=wheel_manifest,
        startup_script_path=STARTUP,
        profile_registry_path=PROFILE,
        output_directory=output,
    )


def test_local_staging_builds_replays_and_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    performance, wheel, wheel_manifest = _authorized_inputs(monkeypatch, tmp_path)
    output = tmp_path / "local-staging"
    receipt = _build(
        performance=performance,
        wheel=wheel,
        wheel_manifest=wheel_manifest,
        output=output,
    )

    assert receipt["paired_hand_count"] == 55
    assert receipt["root_count"] == 110
    assert receipt["job_count"] == 15
    assert receipt["wave_job_counts"] == [8, 7]
    assert receipt["opponent_private_discards_used"] is False
    assert receipt["hidden_information_field_count"] == 0
    assert receipt["cloud_called"] is False
    assert receipt["current_profile_changed"] is False
    assert (output / subject.READY_NAME).is_file()
    assert not (output / subject.FAILURE_NAME).exists()
    assert subject.validate_local_staging_receipt(
        output / subject.READY_NAME
    ) == receipt
    launch = transport.validate_local_launch_manifest(
        subject._read_canonical(output / "launch.json", "launch"),
        staging_directory=output,
    )
    assert launch["job_count"] == 15
    assert launch["wave_job_counts"] == [8, 7]
    assert b"opponent_private_discards" not in (
        output / "control/materialization.json"
    ).read_bytes()

    with pytest.raises(FileExistsError, match="create-only"):
        _build(
            performance=performance,
            wheel=wheel,
            wheel_manifest=wheel_manifest,
            output=output,
        )

    launch_path = output / "launch.json"
    launch_path.write_bytes(launch_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="launch_manifest hash/size/path changed"):
        subject.validate_local_staging_receipt(output / subject.READY_NAME)


def test_local_staging_preserves_partial_failure_without_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    performance, wheel, wheel_manifest = _authorized_inputs(monkeypatch, tmp_path)
    output = tmp_path / "failed-local-staging"

    def fail_materialization(**_kwargs):
        raise RuntimeError("synthetic materialization failure")

    monkeypatch.setattr(
        quality, "materialize_fresh_quality_roots", fail_materialization
    )
    with pytest.raises(RuntimeError, match="synthetic materialization failure"):
        _build(
            performance=performance,
            wheel=wheel,
            wheel_manifest=wheel_manifest,
            output=output,
        )

    assert output.is_dir()
    assert not (output / subject.READY_NAME).exists()
    failure = subject._read_canonical(
        output / subject.FAILURE_NAME, "failure marker"
    )
    assert failure["status"] == "partial_local_staging_preserved_not_authorized"
    assert failure["failed_stage"] == "materialize_55_paired_roots"
    assert failure["cloud_called"] is False
    assert failure["current_profile_changed"] is False


def test_module_cli_guard_exposes_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ofc_regular.hu_m31_t3_step6d_fresh_quality_local_staging_v1",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "--performance-receipt" in completed.stdout
    assert "--output-directory" in completed.stdout
