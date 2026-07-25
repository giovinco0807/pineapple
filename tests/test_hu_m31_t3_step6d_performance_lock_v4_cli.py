from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_performance_lock_v4_cli as subject


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.write_bytes(subject.pure.canonical_bytes(value))
    return path.resolve()


def test_merge_cli_routes_canonical_explicit_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = _write(tmp_path / "candidate.json", {"role": "candidate"})
    reference = _write(tmp_path / "reference.json", {"role": "reference"})
    plan = _write(tmp_path / "plan.json", {"kind": "plan"})
    material = _write(tmp_path / "material.json", {"kind": "material"})
    seal = _write(tmp_path / "seal.json", {"kind": "seal"})
    output = (tmp_path / "pure.json").resolve()
    captured: dict[str, Any] = {}

    def fake_merge(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema": "synthetic-pure"}

    monkeypatch.setattr(
        subject.pure,
        "merge_and_write_candidate02_performance_lock_v4",
        fake_merge,
    )
    assert (
        subject.main(
            [
                "merge",
                "--candidate-done",
                str(candidate),
                "--reference-done",
                str(reference),
                "--plan",
                str(plan),
                "--materialization-receipt",
                str(material),
                "--root-seal",
                str(seal),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured["candidate_done_paths"] == [candidate]
    assert captured["reference_done_paths"] == [reference]
    assert captured["plan_value"] == {"kind": "plan"}
    assert captured["materialization_value"] == {"kind": "material"}
    assert captured["root_seal_value"] == {"kind": "seal"}
    assert captured["output_path"] == output
    assert json.loads(capsys.readouterr().out) == {"schema": "synthetic-pure"}


def test_merge_rejects_candidate_reference_path_alias(tmp_path: Path) -> None:
    done = _write(tmp_path / "done.json", {"role": "both"})
    plan = _write(tmp_path / "plan.json", {"kind": "plan"})
    material = _write(tmp_path / "material.json", {"kind": "material"})
    seal = _write(tmp_path / "seal.json", {"kind": "seal"})
    with pytest.raises(ValueError, match="disjoint"):
        subject.merge_from_paths(
            candidate_done_paths=[done],
            reference_done_paths=[done],
            plan_path=plan,
            materialization_receipt_path=material,
            root_seal_path=seal,
            output_path=tmp_path / "output.json",
        )


def test_finalize_cli_binds_path_and_embedded_merge_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    files = {
        name: _write(tmp_path / f"{name}.json", {"kind": name})
        for name in (
            "pure",
            "plan",
            "material",
            "seal",
            "merge_view",
            "profile",
            "wave_plan",
            "attempt_ledger",
            "accepted_snapshot",
            "lifecycle",
        )
    }
    outer = tmp_path / "outer"
    outer.mkdir()
    output = (tmp_path / "receipt.json").resolve()
    captured: dict[str, Any] = {}

    def fake_finalize(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema": "synthetic-final"}

    monkeypatch.setattr(
        subject.production,
        "write_performance_lock_v4_production_receipt",
        fake_finalize,
    )
    assert (
        subject.main(
            [
                "finalize",
                "--pure-merge",
                str(files["pure"]),
                "--plan",
                str(files["plan"]),
                "--materialization-receipt",
                str(files["material"]),
                "--root-seal",
                str(files["seal"]),
                "--outer-package",
                str(outer.resolve()),
                "--merge-view-manifest",
                str(files["merge_view"]),
                "--current-profile-registry",
                str(files["profile"]),
                "--expected-profile-sha256",
                "a" * 64,
                "--wave-plan",
                str(files["wave_plan"]),
                "--attempt-ledger",
                str(files["attempt_ledger"]),
                "--accepted-results-snapshot",
                str(files["accepted_snapshot"]),
                "--validated-lifecycle-chain",
                str(files["lifecycle"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured["output_path"] == output
    assert captured["pure_merge_path"] == files["pure"]
    assert captured["merge_view_manifest_path"] == files["merge_view"]
    assert captured["merge_view_manifest"] == {"kind": "merge_view"}
    assert captured["wave_plan"] == {"kind": "wave_plan"}
    assert captured["attempt_ledger"] == {"kind": "attempt_ledger"}
    assert captured["accepted_results_snapshot"] == {
        "kind": "accepted_snapshot"
    }
    assert captured["validated_lifecycle_chain"] == {"kind": "lifecycle"}
    assert json.loads(capsys.readouterr().out) == {"schema": "synthetic-final"}


def test_json_inputs_must_be_canonical_lf(tmp_path: Path) -> None:
    path = tmp_path / "not-canonical.json"
    path.write_text('{"z":1, "a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        subject._canonical_mapping(path.resolve(), "input")
