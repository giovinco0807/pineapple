from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1 as transport,
)


_SHUTDOWN_COMMAND = (
    "  if ! /usr/bin/timeout 120 /sbin/shutdown -h now >/dev/null 2>&1; then\n"
    '    if [[ -f "$SHUTDOWN_MARKER" && ! -L "$SHUTDOWN_MARKER" ]] \\\n'
    '      && [[ "$(<"$SHUTDOWN_MARKER")" == "shutdown-requested" ]]; then\n'
    '      rm -f -- "$SHUTDOWN_MARKER"\n'
    "    fi\n"
    "    return 1\n"
    "  fi"
)
_SHUTDOWN_MARKER = "SHUTDOWN_MARKER=/run/ofc-m31-step11-shutdown-requested"
_AUTHORIZED_ARGS = [
    "authorized-worker",
    "contract.json",
    "authorization.json",
    "claim.json",
    "controller-public-key.json",
    "fresh-root",
]


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    assert len(drive) == 1
    relative = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{relative}"


def _instrumented_bootstrap(tmp_path: Path, module_source: str) -> Path:
    source_path = transport._REPO_ROOT / transport.BOOTSTRAP_RELATIVE
    source = source_path.read_text(encoding="utf-8")
    assert source.count(_SHUTDOWN_COMMAND) == 1
    assert source.count(_SHUTDOWN_MARKER) == 1
    source = source.replace(
        _SHUTDOWN_COMMAND,
        (
            "  printf 'shutdown\\n' >> "
            '"${OFC_DIAGNOSTIC_10C2_TEST_SHUTDOWN_LOG}"'
        ),
    )
    source = source.replace(
        _SHUTDOWN_MARKER,
        f"SHUTDOWN_MARKER={_wsl_path(tmp_path / 'shutdown.marker')}",
    )

    root = tmp_path / "outer"
    script = root / transport.BOOTSTRAP_RELATIVE
    script.parent.mkdir(parents=True)
    script.write_text(source, encoding="utf-8", newline="\n")
    script.chmod(0o755)

    package = root / "src" / "ofc_regular"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"{transport.__name__.rsplit('.', 1)[-1]}.py").write_text(
        module_source,
        encoding="utf-8",
        newline="\n",
    )
    return script


def _run(
    tmp_path: Path,
    *,
    args: list[str],
    module_source: str = "raise SystemExit(0)\n",
    environment: Mapping[str, str] | None = None,
    preseed_shutdown_marker: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    script = _instrumented_bootstrap(tmp_path, module_source)
    shutdown_log = tmp_path / "shutdown.log"
    if preseed_shutdown_marker:
        (tmp_path / "shutdown.marker").write_text(
            "outer-wrapper-already-requested\n", encoding="utf-8"
        )
    env = dict(os.environ)
    env.pop("OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER", None)
    env.pop("OFC_DIAGNOSTIC_10C2_LOCAL_PREFLIGHT", None)
    env.update(
        {
            "OFC_DIAGNOSTIC_10C2_PYTHON": "python3",
            "OFC_DIAGNOSTIC_10C2_TEST_SHUTDOWN_LOG": _wsl_path(shutdown_log),
            "PYTHONNOUSERSITE": "1",
        }
    )
    if environment is not None:
        env.update(environment)
    propagated = [
        "OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER",
        "OFC_DIAGNOSTIC_10C2_LOCAL_PREFLIGHT",
        "OFC_DIAGNOSTIC_10C2_PYTHON",
        "OFC_DIAGNOSTIC_10C2_TEST_SHUTDOWN_LOG",
        "PYTHONNOUSERSITE",
    ]
    inherited_wslenv = env.get("WSLENV", "")
    env["WSLENV"] = ":".join(
        [value for value in [inherited_wslenv, *propagated] if value]
    )
    completed = subprocess.run(
        ["bash", _wsl_path(script), *args],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = (
        shutdown_log.read_text(encoding="utf-8").splitlines()
        if shutdown_log.is_file()
        else []
    )
    return completed, lines


@pytest.mark.parametrize(
    ("args", "module_source", "environment", "expected_returncode"),
    [
        (
            [],
            "raise SystemExit(0)\n",
            {"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
            64,
        ),
        (
            ["unknown-mode"],
            "raise SystemExit(0)\n",
            {"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
            64,
        ),
        (_AUTHORIZED_ARGS, "raise SystemExit(0)\n", {}, 78),
        (
            ["authorized-worker", "too-few"],
            "raise SystemExit(0)\n",
            {"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
            78,
        ),
        (
            _AUTHORIZED_ARGS,
            "raise ImportError('injected import failure')\n",
            {"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
            1,
        ),
        (
            _AUTHORIZED_ARGS,
            (
                "import argparse\n"
                "argparse.ArgumentParser().error('injected argparse failure')\n"
            ),
            {"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
            2,
        ),
    ],
)
def test_authorized_preentry_failures_request_exactly_one_bounded_shutdown(
    tmp_path: Path,
    args: list[str],
    module_source: str,
    environment: Mapping[str, str],
    expected_returncode: int,
) -> None:
    completed, shutdowns = _run(
        tmp_path,
        args=args,
        module_source=module_source,
        environment=environment,
    )
    assert completed.returncode == expected_returncode
    assert shutdowns == ["shutdown"]


@pytest.mark.parametrize(
    ("args", "module_source", "expected_returncode"),
    [
        (
            ["local-preflight", "contract.json", "package", "fresh-root"],
            "raise ImportError('injected local import failure')\n",
            1,
        ),
        (
            ["local-preflight", "contract.json", "package", "fresh-root"],
            (
                "import argparse\n"
                "argparse.ArgumentParser().error('injected local argparse failure')\n"
            ),
            2,
        ),
        (["local-preflight", "wrong-arity"], "raise SystemExit(0)\n", 78),
    ],
)
def test_local_preflight_failure_never_requests_shutdown(
    tmp_path: Path,
    args: list[str],
    module_source: str,
    expected_returncode: int,
) -> None:
    completed, shutdowns = _run(
        tmp_path,
        args=args,
        module_source=module_source,
        environment={
            "OFC_DIAGNOSTIC_10C2_LOCAL_PREFLIGHT": "1",
            # Even a stray authorized flag must not turn the explicit local
            # development mode into a host shutdown path.
            "OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1",
        },
    )
    assert completed.returncode == expected_returncode
    assert shutdowns == []


def test_python_shutdown_attempt_exit_code_is_not_repeated_by_shell(
    tmp_path: Path,
) -> None:
    source = (
        transport._REPO_ROOT / transport.BOOTSTRAP_RELATIVE
    ).read_text(encoding="utf-8")
    assert (
        "PYTHON_SHUTDOWN_ATTEMPTED_STATUS="
        f"{transport.AUTHORIZED_WORKER_SHUTDOWN_ATTEMPTED_EXIT_CODE}"
    ) in source
    completed, shutdowns = _run(
        tmp_path,
        args=_AUTHORIZED_ARGS,
        module_source="raise SystemExit(86)\n",
        environment={"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
    )
    assert (
        completed.returncode
        == transport.AUTHORIZED_WORKER_SHUTDOWN_ATTEMPTED_EXIT_CODE
    )
    assert shutdowns == []


def test_preexisting_outer_wrapper_marker_prevents_cross_process_duplicate(
    tmp_path: Path,
) -> None:
    completed, shutdowns = _run(
        tmp_path,
        args=_AUTHORIZED_ARGS,
        module_source="raise ImportError('injected import failure')\n",
        environment={"OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER": "1"},
        preseed_shutdown_marker=True,
    )
    assert completed.returncode == 1
    assert shutdowns == []
