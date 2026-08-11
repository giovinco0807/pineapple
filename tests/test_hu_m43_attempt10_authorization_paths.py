from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.select_hu_m43_attempt09_development as attempt09_selector
from ofc_regular.hu_m43_attempt10_contract import M43_ATTEMPT10_PLAN_SHA256
from ofc_regular.run_hu_m43_attempt10 import ATTEMPT10_AUTHORIZATION_SCHEMA
from ofc_regular.select_hu_m43_attempt10_development import (
    _attempt10_authorization,
    _attempt10_selector_bindings,
    select_attempt10_development,
)


GATE_RELATIVE = Path("artifacts/attempt10/preceding_gate.json")
ROOT = Path(__file__).resolve().parents[1]


def _canonical(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _authorization(run_dir: Path, *, mode: str = "development") -> tuple[Path, str]:
    source_sha = "a" * 64
    gate = run_dir / "package_src" / GATE_RELATIVE
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_bytes(b'{"decision":"authorize"}\n')
    first, last = ((0, 199) if mode == "development" else (200, 249))
    path = run_dir / "execution_authorization.json"
    _canonical(
        path,
        {
            "schema": ATTEMPT10_AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "mode": mode,
            "plan_sha256": M43_ATTEMPT10_PLAN_SHA256,
            "source_package_sha256": source_sha,
            "preceding_gate_artifact": GATE_RELATIVE.as_posix(),
            "preceding_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            "root_index_first": first,
            "root_index_last": last,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    return path, source_sha


@pytest.mark.parametrize("mode", ["development", "future_audit"])
def test_authorization_resolves_package_gate_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    path, source_sha = _authorization(tmp_path / "run", mode=mode)
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    payload, digest = _attempt10_authorization(
        path, mode=mode, source_package_sha256=source_sha
    )
    assert payload is not None and payload["mode"] == mode
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_authorization_accepts_vm_sibling_gate(tmp_path: Path) -> None:
    path, source_sha = _authorization(tmp_path / "run")
    packaged = path.parent / "package_src" / GATE_RELATIVE
    sibling = path.parent / GATE_RELATIVE
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(packaged.read_bytes())
    packaged.unlink()
    payload, _digest = _attempt10_authorization(
        path, mode="development", source_package_sha256=source_sha
    )
    assert payload is not None


def test_authorization_never_trusts_cwd_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, source_sha = _authorization(tmp_path / "run")
    packaged = path.parent / "package_src" / GATE_RELATIVE
    decoy_root = tmp_path / "decoy"
    decoy = decoy_root / GATE_RELATIVE
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(packaged.read_bytes())
    packaged.unlink()
    monkeypatch.chdir(decoy_root)
    with pytest.raises(ValueError, match="missing or changed"):
        _attempt10_authorization(
            path, mode="development", source_package_sha256=source_sha
        )


def test_selector_binding_restores_attempt09_authorization() -> None:
    original = attempt09_selector._authorization
    with _attempt10_selector_bindings():
        assert attempt09_selector._authorization is _attempt10_authorization
    assert attempt09_selector._authorization is original


def test_repo_root_selector_path_uses_package_owned_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, source_sha = _authorization(tmp_path / "run")
    rows = tmp_path / "development.jsonl"
    rows.write_bytes(b"{}\n" * 200)
    captured = {}

    def aggregate(_rows, **kwargs):
        captured.update(kwargs)
        return {"status": "authorization-path-ok"}

    monkeypatch.setattr(
        attempt09_selector, "aggregate_attempt09_development_rows", aggregate
    )
    monkeypatch.chdir(ROOT)
    result = select_attempt10_development(
        input_path=rows,
        plan_path=ROOT / "configs/hu_joint_policy_m43_attempt10.json",
        authorization_path=path,
        source_package_sha256=source_sha,
        run_name="regular-hu-m43-attempt10-development200-unit",
    )
    assert result == {"status": "authorization-path-ok"}
    assert captured["authorization_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
