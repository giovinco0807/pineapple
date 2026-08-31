import hashlib
import json
from pathlib import Path

import pytest

from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_temperature_calibration import (
    CALIBRATION_SCHEMA,
    verify_behavior_temperature_calibration,
)
from ai.tutor.collect_hu_behavior_traces import read_behavior_trace_dataset
from ai.tutor.run_behavior_calibration_smoke import (
    REPORT_SCHEMA,
    _read_canonical_json,
    _read_canonical_jsonl,
    run_behavior_calibration_smoke,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_saved_real_checkpoint_smoke_is_self_bound_and_nonpromoting():
    root = Path("ai/reports/m3_behavior_calibration_smoke_20260713")
    result_path = root / "result.json"
    if not result_path.is_file():
        pytest.skip("saved behavior calibration smoke is not present")
    raw = result_path.read_text(encoding="utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 1
    result = json.loads(raw)
    assert canonical_json(result) == raw[:-1]
    unsigned = dict(result)
    recorded = unsigned.pop("report_sha256")
    assert recorded == canonical_sha256(unsigned)
    assert result["schema"] == REPORT_SCHEMA
    assert result["readback_verified"] is True
    assert result["strategic_strength_evaluated"] is False
    assert result["strategic_strength_claimed"] is False
    assert result["production_gate_passed"] is False
    assert result["promotion_eligible"] is False
    assert result["joker_challenge"]["target_cell_count"] == 12

    for population, directory in (("natural", "natural"), ("joker_challenge", "challenge")):
        section = result[population]
        assert section["decision_file_sha256"] == _sha256(
            root / directory / "decisions.jsonl"
        )
        assert section["hidden_roots_file_sha256"] == _sha256(
            root / directory / "roots.jsonl"
        )
        assert section["evaluation_file_sha256"] == _sha256(
            root / directory / "model_evaluations.jsonl"
        )

    natural = read_behavior_trace_dataset(root / "natural" / "decisions.jsonl")
    challenge = read_behavior_trace_dataset(root / "challenge" / "decisions.jsonl")
    natural_rows = _read_canonical_jsonl(
        root / "natural" / "model_evaluations.jsonl"
    )
    challenge_rows = _read_canonical_jsonl(
        root / "challenge" / "model_evaluations.jsonl"
    )
    calibration = _read_canonical_json(root / "calibration.json")
    assert calibration["schema"] == CALIBRATION_SCHEMA
    assert verify_behavior_temperature_calibration(
        natural.records,
        natural_rows,
        calibration,
        challenge_records=challenge.records,
        challenge_evaluation_rows=challenge_rows,
    ) == calibration


@pytest.mark.parametrize(
    ("natural_roots", "challenge_roots", "message"),
    ((11, 12, "at least 12"), (12, 13, "multiple of 12")),
)
def test_smoke_rejects_incomplete_preregistered_root_grids(
    tmp_path, natural_roots, challenge_roots, message
):
    with pytest.raises(ValueError, match=message):
        run_behavior_calibration_smoke(
            workspace_root=Path("."),
            output_dir=tmp_path,
            natural_root_count=natural_roots,
            challenge_root_count=challenge_roots,
        )
