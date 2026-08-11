from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import ofc_regular.validate_hu_m43_attempt07_closeout as closeout


ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m43_attempt07_development"
    / "regular-hu-m43-attempt07-development100-20260714-150046"
    / "merged"
    / "development_arm_selection.json"
)


def _write_canonical(path: Path, payload: dict) -> None:
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def test_actual_attempt07_no_go_closeout_validates_exact_chain() -> None:
    report = closeout.validate_attempt07_no_go_closeout(ROOT)

    assert report == {
        "schema": "hu_m43_attempt07_no_go_closeout_validation_v1",
        "status": "validated_complete_no_go_development",
        "run_name": "regular-hu-m43-attempt07-development100-20260714-150046",
        "decision": "no_go",
        "selected_arm": None,
        "winner": None,
        "validated_artifacts": 14,
        "selection_sha256": (
            "53b797756747cbb71f89251020e7fdc86dabce75cd298507dc59c0764ed6373a"
        ),
        "merged_input_sha256": (
            "c15e018fee5e86cbcba8fadcfdaa5a13b25866efa1092c7b74177814049761a5"
        ),
        "audit50_authorized": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_mutation_allowed": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("decision", "go"),
        ("selected_arm", "r32_v64"),
        ("winner", {"arm_name": "r32_v64"}),
    ),
)
def test_selection_validator_rejects_any_go_or_winner(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    payload[field] = value
    candidate = tmp_path / "selection.json"
    _write_canonical(candidate, payload)

    with pytest.raises(ValueError, match=field.replace("_", " ") + "|selection"):
        closeout._validate_selection(candidate)


@pytest.mark.parametrize(
    "field",
    (
        "fit_performed",
        "threshold_selected",
        "future_audit_authorized",
        "future_audit_opened",
        "runtime_policy_activated",
        "current_profile_mutated",
    ),
)
def test_selection_validator_rejects_open_science_boundary(
    tmp_path: Path, field: str
) -> None:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    payload["science_boundary"][field] = True
    candidate = tmp_path / "selection.json"
    _write_canonical(candidate, payload)

    with pytest.raises(ValueError, match="science boundary"):
        closeout._validate_selection(candidate)


def test_selection_validator_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    candidate = tmp_path / "selection.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical selector encoding"):
        closeout._validate_selection(candidate)


def test_selection_validator_rejects_nonzero_integrity_count(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(json.loads(SELECTION.read_text(encoding="utf-8")))
    payload["integrity"]["rng_domain_violation_count"] = 1
    candidate = tmp_path / "selection.json"
    _write_canonical(candidate, payload)

    with pytest.raises(ValueError, match="selection integrity"):
        closeout._validate_selection(candidate)


def test_cli_is_read_only_and_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    closeout_path = ROOT / "configs" / "hu_joint_policy_m43_attempt07_closeout.json"
    before = closeout_path.read_bytes()

    assert closeout.main(["--repo-root", str(ROOT)]) == 0
    first = capsys.readouterr().out
    assert closeout.main(["--repo-root", str(ROOT)]) == 0
    second = capsys.readouterr().out

    assert first == second
    assert json.loads(first)["decision"] == "no_go"
    assert closeout_path.read_bytes() == before
