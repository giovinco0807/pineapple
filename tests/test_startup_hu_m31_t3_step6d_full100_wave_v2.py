from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2


ROOT = Path(__file__).resolve().parents[1]
STARTUP = ROOT / "scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh"


def _source() -> str:
    return STARTUP.read_text(encoding="utf-8")


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _heredocs() -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY", _source(), flags=re.DOTALL)


def _bootstrap_block() -> str:
    return next(block for block in _heredocs() if "bootstrap metadata fields changed" in block)


def _archive_block() -> str:
    return next(block for block in _heredocs() if "scientific archive missing/extra member" in block)


def _shell_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}$",
        _source(),
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    return match.group(0)


def _binding(prefix: str, suffix: str, token: str) -> dict[str, object]:
    return {
        "object_name": f"{prefix}/{suffix}",
        "sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "bytes": 100 + len(token),
    }


def _bootstrap() -> dict[str, object]:
    run = "regular-hu-m31-c02-f100wv2-20260722-003"
    job = "candidate-shard-00"
    attempt = "a00"
    prefix = "hu-m31-t3/full100-wave-v2/content-digest"
    instance = "f100-test-w00-j00-a00"
    value: dict[str, object] = {
        "schema": package_v2.JOB_BOOTSTRAP_SCHEMA,
        "status": "single_job_bootstrap_bound_to_prelaunch_authorization",
        "run_name": run,
        "execution_identity_sha256": "1" * 64,
        "wave_plan_sha256": "2" * 64,
        "attempt_ledger_sha256": "3" * 64,
        "resume_sha256": "4" * 64,
        "observed_transition_digest": "5" * 64,
        "wave_index": 0,
        "job_id": job,
        "source_role": "candidate",
        "attempt_id": attempt,
        "instance_name": instance,
        "artifact_prefix": (
            f"runs/{run}-tag/full100-wave-v2/results/jobs/"
            f"{job}/attempts/{attempt}"
        ),
        "bucket": "ofc-test-bucket",
        "content_prefix": prefix,
        "outer_manifest_sha256": "6" * 64,
        "content_payload_sha256": "7" * 64,
        "scientific_source": _binding(prefix, "content/science/source.zip", "science"),
        "scientific_manifest": _binding(
            prefix, "content/science/manifest.json", "science-manifest"
        ),
        "wheelhouse": _binding(
            prefix, "content/wheelhouse/wheelhouse.zip", "wheels"
        ),
        "wheelhouse_manifest": _binding(
            prefix,
            "content/wheelhouse/wheelhouse_manifest.json",
            "wheel-manifest",
        ),
        "startup": _binding(
            prefix,
            "content/startup/startup_hu_m31_t3_step6d_full100_wave_v2.sh",
            "startup",
        ),
        "wave_plan": _binding(prefix, "control/wave_plan.json", "wave"),
        "job_manifest": _binding(
            prefix, f"content/jobs/{job}.json", "job"
        ),
        "prelaunch_authorization_sha256": "8" * 64,
        "worker_principal": "ofc-worker@ofc-project-123.iam.gserviceaccount.com",
        "one_vm_one_job_one_role": True,
        "additional_create_authorized": False,
        "hidden_truth_exposed": False,
    }
    value["startup"]["sha256"] = "9" * 64  # type: ignore[index]
    value["startup"]["bytes"] = 12345  # type: ignore[index]
    value["bootstrap_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def test_shell_and_inline_python_syntax_are_valid() -> None:
    source = _source()
    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    bash = shutil.which("bash")
    if bash:
        result = subprocess.run(
            [bash, "-n", STARTUP.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    blocks = _heredocs()
    assert len(blocks) == 5
    for index, block in enumerate(blocks):
        compile(block, f"{STARTUP.name}#heredoc-{index}", "exec")


def test_bootstrap_contract_matches_outer_package_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    value = _bootstrap()
    assert set(value) == package_v2._BOOTSTRAP_KEYS
    raw = _canonical(value)
    encoded = base64.b64encode(raw).decode("ascii")
    path = tmp_path / "bootstrap.json"
    path.write_bytes(raw)
    good = subprocess.run(
        [
            sys.executable,
            "-c",
            _bootstrap_block(),
            str(path),
            encoded,
            "9" * 64,
            str(value["instance_name"]),
            str(value["worker_principal"]),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert good.returncode == 0, good.stderr
    assert len(good.stdout.splitlines()) == 39

    extra = deepcopy(value)
    extra["extra_job"] = "reference-shard-00"
    extra["bootstrap_sha256"] = hashlib.sha256(
        _canonical({k: v for k, v in extra.items() if k != "bootstrap_sha256"})
    ).hexdigest()
    extra_path = tmp_path / "extra.json"
    extra_raw = _canonical(extra)
    extra_path.write_bytes(extra_raw)
    rejected = subprocess.run(
        [
            sys.executable,
            "-c",
            _bootstrap_block(),
            str(extra_path),
            base64.b64encode(extra_raw).decode("ascii"),
            "9" * 64,
            str(value["instance_name"]),
            str(value["worker_principal"]),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert rejected.returncode != 0
    assert "fields changed" in rejected.stderr

    hidden = deepcopy(value)
    hidden["hidden_truth_exposed"] = True
    unsigned = {k: v for k, v in hidden.items() if k != "bootstrap_sha256"}
    hidden["bootstrap_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    hidden_path = tmp_path / "hidden.json"
    hidden_raw = _canonical(hidden)
    hidden_path.write_bytes(hidden_raw)
    rejected = subprocess.run(
        [
            sys.executable,
            "-c",
            _bootstrap_block(),
            str(hidden_path),
            base64.b64encode(hidden_raw).decode("ascii"),
            "9" * 64,
            str(value["instance_name"]),
            str(value["worker_principal"]),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert rejected.returncode != 0
    assert "trust boundary changed" in rejected.stderr


def _build_fake_archives(
    root: Path, *, extra_science_member: bool = False, wrong_role: bool = False
) -> list[str]:
    content = root / "content"
    (root / "science").mkdir(parents=True)
    (root / "wheelhouse").mkdir()
    content.mkdir()
    work = list(range(10))
    allocation = {"workers": 1, "rayon_threads_per_worker": 16}
    run_contract = {"allocation": allocation, "test": "frozen"}
    run_digest = _sha(_canonical(run_contract, newline=True))
    job = {
        "schema": "hu_m31_t3_step6d_performance_shard_manifest_v2",
        "run_contract": run_contract,
        "run_contract_digest": run_digest,
        "source_role": "reference" if wrong_role else "candidate",
        "work_hand_indices": work,
    }
    job_raw = _canonical(job, newline=True)
    job_path = content / "job_manifest.json"
    job_path.write_bytes(job_raw)
    full_plan = {
        "jobs": [
            {
                "job_id": "candidate-shard-00",
                "source_role": "candidate",
                "work_hand_indices": work,
                "shard_manifest_sha256": _sha(job_raw),
            }
        ],
        "run_contract": run_contract,
    }
    full_plan_raw = _canonical(full_plan, newline=True)
    requirements_raw = b"dummy-package==1.0\n"
    library_raw = b"fake-native-library"
    members: dict[str, bytes] = {
        "configs/hu_m43_attempt08_runtime_requirements.txt": requirements_raw,
        "frozen/full100_plan.json": full_plan_raw,
        "native/candidate/release/libofc_hu_m3_engine.so": library_raw,
    }
    for index in range(100):
        members[f"frozen/full100_roots/hand_{index:03d}.json"] = _canonical(
            {"hand_index": index, "opponent_private_discards_used": False},
            newline=True,
        )
    source = content / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
        if extra_science_member:
            archive.writestr("rogue/extra_job.json", b"{}\n")
    source_raw = source.read_bytes()
    source_entries = {
        name: {"sha256": _sha(raw), "bytes": len(raw)}
        for name, raw in sorted(members.items())
    }
    science_manifest = {
        "schema": "hu_m31_t3_step6d_full100_spot_package_v1",
        "source_sha256": _sha(source_raw),
        "source_bytes": len(source_raw),
        "source_entries": source_entries,
        "source_entry_count": len(source_entries),
        "plan_sha256": _sha(full_plan_raw),
        "run_contract_digest": run_digest,
        "accepted_candidate": {
            "package_path": "native/candidate/release/libofc_hu_m3_engine.so",
            "sha256": _sha(library_raw),
        },
    }
    science_manifest_path = content / "scientific_manifest.json"
    science_manifest_path.write_bytes(_canonical(science_manifest, newline=True))

    wheel_name = "dummy_package-1.0-py3-none-any.whl"
    wheel_raw = b"fake-wheel"
    wheel_archive = content / "wheelhouse.zip"
    with zipfile.ZipFile(wheel_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(wheel_name, wheel_raw)
    wheel_entries = [
        {
            "filename": wheel_name,
            "sha256": _sha(wheel_raw),
            "bytes": len(wheel_raw),
            "distribution": "dummy-package",
            "version": "1.0",
            "tags": ["py3-none-any"],
        }
    ]
    wheel_manifest = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1",
        "status": "complete_hash_pinned_offline_wheelhouse",
        "requirements_sha256": _sha(requirements_raw),
        "python_abi": "cp311",
        "target_os": "linux",
        "target_architecture": "x86_64",
        "network_install_allowed": False,
        "entries": wheel_entries,
        "entry_count": 1,
        "entries_sha256": _sha(_canonical(wheel_entries, newline=True)),
    }
    wheel_manifest_path = content / "wheelhouse_manifest.json"
    wheel_manifest_path.write_bytes(_canonical(wheel_manifest))
    wave = {
        "schema": "hu_m31_t3_step6d_full100_wave_plan_v2",
        "schedule_sha256": "2" * 64,
        "run_name": "regular-hu-m31-c02-f100wv2-20260722-003",
        "execution_identity_sha256": "1" * 64,
        "current_profile_changed": False,
        "cloud_launch_authorized": False,
        "full100_plan": full_plan,
        "full100_plan_sha256": _sha(full_plan_raw),
        "run_contract_digest": run_digest,
        "runtime_binding": {
            "package_sha256": _sha(source_raw),
            "image_digest": "sha256:" + "a" * 64,
            "binary_sha256_by_role": {"candidate": _sha(library_raw)},
            "allocation_digest": wave_v2.canonical_sha256(allocation),
        },
    }
    wave_path = content / "wave_plan.json"
    # The real outer package uses wave_v2.canonical_bytes, including one LF.
    wave_path.write_bytes(_canonical(wave, newline=True))
    bootstrap = {
        "run_name": wave["run_name"],
        "execution_identity_sha256": wave["execution_identity_sha256"],
        "job_id": "candidate-shard-00",
        "source_role": "candidate",
    }
    bootstrap_path = root / "bootstrap.json"
    bootstrap_path.write_bytes(_canonical(bootstrap))
    return [
        str(root),
        str(bootstrap_path),
        _sha(source_raw),
        str(len(source_raw)),
        _sha(science_manifest_path.read_bytes()),
        "2" * 64,
        _sha(wave_path.read_bytes()),
        _sha(job_raw),
        _sha(wheel_archive.read_bytes()),
        _sha(wheel_manifest_path.read_bytes()),
    ]


def test_fake_closed_world_archive_smoke_and_extra_job_rejection(
    tmp_path: Path,
) -> None:
    good_args = _build_fake_archives(tmp_path / "good")
    good = subprocess.run(
        [sys.executable, "-c", _archive_block(), *good_args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert good.returncode == 0, good.stderr
    assert len(good.stdout.splitlines()) == 8

    bad_args = _build_fake_archives(
        tmp_path / "extra", extra_science_member=True
    )
    bad = subprocess.run(
        [sys.executable, "-c", _archive_block(), *bad_args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert bad.returncode != 0
    assert "missing/extra member" in bad.stderr

    wrong_args = _build_fake_archives(tmp_path / "wrong-role", wrong_role=True)
    wrong = subprocess.run(
        [sys.executable, "-c", _archive_block(), *wrong_args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert wrong.returncode != 0
    assert "selected runner job changed" in wrong.stderr


def test_transport_is_atomic_create_only_offline_and_done_last() -> None:
    source = _source()
    assert 'OWNED_ROOT_REAL="$(realpath -e "$OWNED_ROOT")"' in source
    assert "os.fsync(stream.fileno())" in source
    assert "os.replace(temporary, destination)" in source
    assert "os.fsync(fd)" in source
    assert "ifGenerationMatch=0" in source
    assert "--no-index" in source
    assert '--find-links "$WHEELS"' in source
    assert "network_install_allowed" in source
    assert "gcloud compute instances create" not in source
    assert "gcloud " not in source
    assert "gsutil" not in source
    assert "set -x" not in source
    assert "export RAYON_NUM_THREADS=16" in source
    assert "export OFC_HU_M3_BATCH_THREADS=1" in source
    assert source.count(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2"
    ) == 2
    checkpoint = source.index("checkpoint_hand()")
    runner = source.index(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2",
        checkpoint,
    )
    readback = source.rindex("# Read back every checkpoint before")
    done_upload = source.rindex(
        'upload_create_only "$TRANSPORT_DONE" "$ARTIFACT_PREFIX/DONE.json"'
    )
    assert checkpoint < runner < readback < done_upload
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    assert "DONE.json" not in cleanup
    for token in (
        "opponent_private_discards",
        "metadata_hidden_truth_exposed",
        'value["hidden_truth_exposed"] is not False',
        "additional_create_authorized",
        "extra/unplanned job",
    ):
        assert token in source


def test_transport_done_identity_uses_wave_v2_canonical_newline() -> None:
    source = _source()
    assert 'hashlib.sha256(canonical(roots) + b"\\n").hexdigest()' in source
    assert (
        'hashlib.sha256(canonical(identity_value) + b"\\n").hexdigest()'
        in source
    )


def test_fake_gcs_rest_publish_uses_generation_zero(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    trace = tmp_path / "curl.trace"
    fake = fake_bin / "curl"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >\"$TRACE\"\n"
        "printf '%s\\n' '{\"generation\":\"1\"}'\n",
        encoding="utf-8",
        newline="\n",
    )
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "export PATH=\"$(pwd)/bin:$PATH\"\n"
        "export TRACE=\"$(pwd)/curl.trace\"\n"
        "PROJECT_ID=ofc-test\nBUCKET=ofc-bucket\n"
        "token() { printf '%s' fake-token; }\n"
        "fsync_file_parent() { :; }\n"
        + _shell_function("encoded_object")
        + "\n"
        + _shell_function("upload_create_only")
        + "\nprintf x >payload.bin\n"
        "upload_create_only payload.bin safe/attempt/path.json\n",
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [bash, "./harness.sh"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0 and "No such file" in result.stderr:
        pytest.skip("the available bash cannot access the pytest temp directory")
    assert result.returncode == 0, result.stderr
    invocation = trace.read_text(encoding="utf-8")
    assert invocation.count("ifGenerationMatch=0") == 1
    assert "-X POST" in invocation
    assert "--data-binary @payload.bin" in invocation
    assert "name=safe%2Fattempt%2Fpath.json" in invocation
    assert "gs://" not in invocation
