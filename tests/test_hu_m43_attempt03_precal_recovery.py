from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ofc_regular import recover_hu_m43_attempt03_precal_finalize as recovery


ROOT = Path(__file__).resolve().parents[1]
RECEIVER = ROOT / "scripts" / "Receive-GcpHuM43Attempt03TeacherRun.ps1"
RESUME = ROOT / "scripts" / "Resume-GcpHuM43Attempt03TeacherRun.ps1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role(root: Path, paths: list[Path]) -> dict[str, object]:
    rows = []
    for index, path in enumerate(paths):
        rows.append(
            {
                "index": index,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "records": 10,
                "file_sha256": _sha(path),
                "canonical_rows_sha256": f"{index + 1:064x}",
                "identity_sha256": f"{index + 11:064x}",
            }
        )
    unsigned = {
        "ordered_shards": rows,
        "records": len(paths) * 10,
        "canonical_rows_sha256": "f" * 64,
    }
    return {**unsigned, "ordered_shards_sha256": recovery._digest(unsigned)}


def test_row_blind_role_binding_rejects_byte_tamper(tmp_path: Path) -> None:
    # Deliberately not valid JSON: recovery must use byte/identity bindings and
    # must never inspect teacher row values.
    paths = []
    for shard in range(2):
        path = tmp_path / f"shard-{shard}.jsonl"
        path.write_bytes(b"not-json\n" * 10)
        paths.append(path)
    role = _role(tmp_path, paths)
    bound, rows = recovery._validate_role(
        root=tmp_path.resolve(), role=role, expected_shards=2
    )
    assert bound == [path.resolve() for path in paths]
    assert len(rows) == 2

    paths[1].write_bytes(b"tampered\n" * 10)
    with pytest.raises(ValueError, match="bytes changed|file hash changed"):
        recovery._validate_role(
            root=tmp_path.resolve(), role=role, expected_shards=2
        )


def test_merge_digest_exactly_matches_receiver_crlf_stream(tmp_path: Path) -> None:
    shards = []
    for index in range(2):
        path = tmp_path / f"{index}.jsonl"
        path.write_bytes(f"a-{index}\nb-{index}\n".encode())
        shards.append(path)
    merged = b"a-0\r\nb-0\r\na-1\r\nb-1\r\n"
    digest, rows = recovery._merge_digest(shards)
    assert rows == 4
    assert digest == hashlib.sha256(merged).hexdigest()


def test_preflight_binding_rejects_resigned_lifecycle_tamper(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    preflight = {
        "schema": "hu_m43_attempt03_preflight_receipt_v1",
        "status": "pass_frozen_before_fresh_generation",
    }
    preflight["receipt_sha256"] = recovery._digest(preflight)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "preflight_file_sha256": _sha(preflight_path),
                "plan_file_sha256": _sha(plan),
            }
        ),
        encoding="utf-8",
    )
    assert recovery._validate_preflight_binding(
        preflight_path=preflight_path,
        plan_path=plan,
        manifest_path=manifest_path,
    )["receipt_sha256"] == preflight["receipt_sha256"]

    preflight["status"] = "tampered-but-resigned"
    preflight.pop("receipt_sha256")
    preflight["receipt_sha256"] = recovery._digest(preflight)
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preflight_file_sha256"] = _sha(preflight_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="preflight/manifest binding changed"):
        recovery._validate_preflight_binding(
            preflight_path=preflight_path,
            plan_path=plan,
            manifest_path=manifest_path,
        )


def test_fit_receipt_requires_one_to_one_contract_shards(tmp_path: Path) -> None:
    paths = []
    for shard in range(2):
        path = tmp_path / f"fit-{shard}.jsonl"
        path.write_bytes(b"opaque\n" * 10)
        paths.append(path.resolve())
    contract_rows = [
        {"file_sha256": _sha(path)} for path in paths
    ]
    receipt = {
        "fresh_train_fit": {
            "shards": [
                {
                    "shard": index,
                    "logical_split": "train.fit",
                    "split_shard": index,
                    "rows": 10,
                    "sha256": _sha(path),
                    "path": str(path),
                }
                for index, path in enumerate(paths)
            ]
        }
    }
    recovery._validate_fit_receipt_shards(
        root=tmp_path.resolve(),
        fit_receipt=receipt,
        contract_paths=paths,
        contract_rows=contract_rows,
    )
    receipt["fresh_train_fit"]["shards"][0]["path"] = str(paths[1])
    with pytest.raises(ValueError, match="receipt/contract shard mismatch"):
        recovery._validate_fit_receipt_shards(
            root=tmp_path.resolve(),
            fit_receipt=receipt,
            contract_paths=paths,
            contract_rows=contract_rows,
        )


def test_publish_stage_rejects_destination_race_without_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    final = tmp_path / "final"
    stage.mkdir()
    (stage / "receipt.json").write_text("stage", encoding="utf-8")
    real_rename = recovery.os.rename

    def racing_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text("keep", encoding="utf-8")
        real_rename(source, destination)

    monkeypatch.setattr(recovery.os, "rename", racing_rename)
    with pytest.raises(FileExistsError):
        recovery._publish_stage_no_clobber(stage, final)
    assert (stage / "receipt.json").read_text(encoding="utf-8") == "stage"
    assert (final / "sentinel.txt").read_text(encoding="utf-8") == "keep"


def test_forward_receiver_rebases_closure_after_atomic_move() -> None:
    text = RECEIVER.read_text(encoding="utf-8")
    captured = text.index("$preMoveClosureHashes = [ordered]@{}")
    moved = text.index(
        "Move-Item -LiteralPath $precalDownloadStage -Destination $PrecalDownloadDir",
        captured,
    )
    reassigned = text.index("$Closure = Rebase-FrozenClosureAfterMove", moved)
    receipt_hash = text.index(
        "manifest_sha256 = Get-Sha256 $Closure.paths.manifest", reassigned
    )
    assert captured < moved < reassigned < receipt_hash
    function = text[
        text.index("function Rebase-FrozenClosureAfterMove") : text.index(
            "$repoRoot =", text.index("function Rebase-FrozenClosureAfterMove")
        )
    ]
    assert "foreach ($name in $rebased.Keys)" in function
    assert "$rebased.psobject.Properties" not in function
    assert "immutable closure changed while rebasing after atomic move" in text


@pytest.mark.skipif(
    shutil.which("powershell") is None,
    reason="Windows PowerShell 5 unavailable",
)
def test_forward_rebase_runs_with_exactly_six_keys_in_powershell5(
    tmp_path: Path,
) -> None:
    source = tmp_path / "final"
    (source / "source").mkdir(parents=True)
    names = {
        "manifest": "manifest.json",
        "schedule": "source/shards_manifest.jsonl",
        "model_manifest": "source/source_model_manifest.json",
        "native_manifest": "source/source_native_manifest.json",
        "source": "source/ofc_regular_hu_m43_attempt03_teacher_source.zip",
        "startup": "source/startup_hu_m43_attempt03_teacher.sh",
    }
    for index, relative in enumerate(names.values()):
        (source / relative).write_bytes(f"file-{index}".encode())
    text = RECEIVER.read_text(encoding="utf-8")
    start = text.index("function Rebase-FrozenClosureAfterMove")
    function = text[start : text.index("$repoRoot =", start)]
    quoted = str(source).replace("'", "''")
    command = r'''
$ErrorActionPreference='Stop'
function Get-Sha256([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
''' + function + f"\n$root='{quoted}'\n" + r'''
$hashes=[ordered]@{
 manifest=Get-Sha256 (Join-Path $root 'manifest.json')
 schedule=Get-Sha256 (Join-Path $root 'source/shards_manifest.jsonl')
 model_manifest=Get-Sha256 (Join-Path $root 'source/source_model_manifest.json')
 native_manifest=Get-Sha256 (Join-Path $root 'source/source_native_manifest.json')
 source=Get-Sha256 (Join-Path $root 'source/ofc_regular_hu_m43_attempt03_teacher_source.zip')
 startup=Get-Sha256 (Join-Path $root 'source/startup_hu_m43_attempt03_teacher.sh')
}
$closure=[pscustomobject]@{audit='preserved'}
$result=Rebase-FrozenClosureAfterMove -Closure $closure -FinalDownloadDir $root -PreMoveClosureHashes $hashes
[pscustomobject]@{
 major=$PSVersionTable.PSVersion.Major
 count=@($result.paths.psobject.Properties).Count
 names=@($result.paths.psobject.Properties.Name)
 audit=$result.audit
}|ConvertTo-Json -Compress
'''
    result = subprocess.run(
        [
            shutil.which("powershell"),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "major": 5,
        "count": 6,
        "names": list(names),
        "audit": "preserved",
    }


def test_resume_wrapper_requires_exact_incident_witnesses() -> None:
    text = RESUME.read_text(encoding="utf-8")
    for token in (
        "ResumeClaimedPrecalFinalize must be explicitly selected",
        "ExpectedStageName",
        "ExpectedOpenClaimFileSha256",
        "ExpectedDataContractFileSha256",
        "ExpectedMergedPrecalFileSha256",
        "AuditOnly",
        "recover_hu_m43_attempt03_precal_finalize",
    ):
        assert token in text


@pytest.mark.skipif(
    shutil.which("pwsh") is None and shutil.which("powershell") is None,
    reason="PowerShell unavailable",
)
def test_receiver_and_resume_scripts_parse_in_powershell() -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    paths = ",".join(f"'{path}'" for path in (RECEIVER, RESUME))
    command = (
        f"$bad=@();foreach($f in @({paths})){{"
        "$t=$null;$e=$null;"
        "[Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$bad+=$e}};"
        "if($bad.Count){$bad|ForEach-Object{$_.Message};exit 1}"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
