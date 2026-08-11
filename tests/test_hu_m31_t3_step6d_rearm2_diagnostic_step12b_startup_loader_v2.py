from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_startup_loader_v2
    as subject,
)


def test_loader_is_metadata_sized_and_source_download_is_after_trust_gate() -> None:
    source = subject.build_startup_loader_source()
    assert len(source.encode("utf-8")) < 262_144
    assert len(bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS) == 12
    assert repr(
        tuple(sorted(bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS))
    ) in source
    main = source[source.index("def main():") :]
    assert main.index(
        "checked_manifest = validate_deployment_and_authorization("
    ) < main.index("ensure_host_prerequisites()")
    assert main.index("ensure_host_prerequisites()") < main.index(
        "token = worker_access_token()"
    )
    assert main.index("token = worker_access_token()") < main.index(
        "runtime_raw = generation_pinned_get("
    )
    assert "generation_pinned_bootstrap_source_download" in source
    assert "bounded_host_prerequisite_install" in source
    assert "requests" not in source
    assert "google.cloud" not in source


def test_loader_runs_stdlib_only_under_isolated_python(
    tmp_path: Path,
) -> None:
    loader = tmp_path / "startup_loader.py"
    loader.write_text(
        subject.build_startup_loader_source(), encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(loader), "--self-test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    receipt = json.loads(completed.stdout)
    assert receipt == {
        "network_called": False,
        "repo_imported": False,
        "required_source_file_count": 12,
        "schema": subject.STARTUP_LOADER_SCHEMA,
        "site_packages_imported": False,
        "status": "stdlib_isolated_self_test_passed",
    }


def test_loader_source_is_deterministic() -> None:
    assert (
        subject.build_startup_loader_source()
        == subject.build_startup_loader_source()
    )
    assert len(subject.startup_loader_sha256()) == 64


def test_loader_failure_marker_precedes_bounded_shutdown() -> None:
    source = subject.build_startup_loader_source()
    failure = source[source.index("except BaseException as error:") :]
    assert subject.STARTUP_LOADER_SCHEMA in source
    assert "FAILURE_MARKER_PREFIX" in failure
    assert "OFC_STEP12N_WORKER_FAILURE_V1 " in source
    assert failure.index("print(") < failure.index(
        "time.sleep(FAILURE_HOLD_SECONDS)"
    )
    assert failure.index("time.sleep(FAILURE_HOLD_SECONDS)") < failure.index(
        '["/sbin/shutdown", "-h", "now"]'
    )


def test_missing_venv_runs_bounded_prerequisite_install() -> None:
    namespace = {"__name__": "startup_loader_test"}
    exec(
        compile(
            subject.build_startup_loader_source(),
            "<startup-loader-test>",
            "exec",
        ),
        namespace,
    )
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    namespace["subprocess"] = SimpleNamespace(run=fake_run)
    namespace["ensure_host_prerequisites"]()
    assert len(calls) == 4
    assert calls[1][:5] == [
        "/usr/bin/timeout",
        "300",
        "/usr/bin/env",
        "DEBIAN_FRONTEND=noninteractive",
        "/usr/bin/apt-get",
    ]
    assert "update" in calls[1]
    assert "install" in calls[2]
    assert "python3-venv" in calls[2]
    assert calls[3][1:] == ["-c", "import ensurepip,venv"]


def test_materialized_exact_source_bundle_imports_vm_entrypoint_isolated(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sources = {
        path: (repo_root / "src" / Path(path)).read_text(
            encoding="utf-8"
        )
        for path in bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    }
    bundle = bootstrap_source.build_runtime_source_bundle(sources)
    destination = tmp_path / "runtime"
    bootstrap_source.materialize_runtime_source_bundle(
        bundle, destination=destination
    )
    module_name = (
        "ofc_regular.hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_vm_prebootstrap_v2"
    )
    program = r"""
import importlib, pathlib, sys
runtime = pathlib.Path(sys.argv[1]).resolve()
forbidden = pathlib.Path(sys.argv[2]).resolve()
assert sys.flags.isolated
assert "site" not in sys.modules
sys.path.insert(0, str(runtime))
for entry in sys.path[1:]:
    if not entry:
        continue
    lowered = entry.lower()
    assert "site-packages" not in lowered
    resolved = pathlib.Path(entry).resolve()
    assert resolved != forbidden and forbidden not in resolved.parents
module = importlib.import_module(sys.argv[3])
assert callable(module.run_downloaded_entrypoint)
print(module.DOWNLOAD_RECEIPT_SCHEMA)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(destination),
            str(repo_root),
            module_name,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.strip().endswith(
        "step12b_bootstrap_download_receipt_v2"
    )
