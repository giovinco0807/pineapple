"""Run the real-checkpoint behavior trace/calibration wiring smoke.

This is deliberately a non-promoting bootstrap.  Actions are sampled from the
same frozen T1/T2 prior whose raw logits are then read directly, so the run
proves the physical trace, exact sampler, logit extractor, split, Joker
challenge, fitter, and readback wiring.  It does *not* prove strategic strength
or satisfy the production sample-count gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_logit_evaluator_torch import (
    TorchPolicyValueLegalLogitEvaluator,
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.behavior_temperature_calibration import (
    build_behavior_temperature_calibration,
    build_model_evaluation_row,
    build_temperature_gate_config,
    verify_behavior_temperature_calibration,
    verify_model_evaluation_row,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
    collect_hu_behavior_traces,
    read_behavior_trace_dataset,
    write_behavior_trace_dataset,
)
from ai.tutor.frozen_behavior_torch import TurnActorBehaviorDispatch


REPORT_SCHEMA = "ofc_behavior_calibration_smoke_report/v2"
NATURAL_SEED_NAMESPACE = "m3-behavior-calibration-smoke-20260713-v2"
CHALLENGE_SEED_NAMESPACE = "m3-joker-grid-smoke-20260713-v1"
CHALLENGE_ID = "m3-joker-challenge-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


def _write_canonical_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        raise ValueError("canonical JSONL output requires at least one row")
    payload = "\n".join(canonical_json(row) for row in rows) + "\n"
    _atomic_write(path, payload.encode("utf-8"))


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise ValueError(f"{path.name} must contain one canonical JSON object")
    value = json.loads(raw[:-1])
    if not isinstance(value, dict) or canonical_json(value) != raw[:-1]:
        raise ValueError(f"{path.name} is not canonical JSON")
    return value


def _read_canonical_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise ValueError(f"{path.name} must end with a newline")
    rows: list[dict[str, Any]] = []
    for line in raw[:-1].split("\n"):
        value = json.loads(line)
        if not isinstance(value, dict) or canonical_json(value) != line:
            raise ValueError(f"{path.name} contains non-canonical JSONL")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return tuple(rows)


def _evaluation_rows(
    records: Sequence[Mapping[str, Any]],
    evaluators: Mapping[tuple[int, str], TorchPolicyValueLegalLogitEvaluator],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for record in records:
        key = (int(record["turn"]), str(record["actor"]))
        evaluator = evaluators.get(key)
        if evaluator is None:
            raise KeyError(f"no raw-logit evaluator for T{key[0]} {key[1]}")
        rows.append(build_model_evaluation_row(record, evaluator))
    return tuple(rows)


def run_behavior_calibration_smoke(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    natural_root_count: int = 12,
    challenge_root_count: int = 12,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    if natural_root_count < 12:
        raise ValueError("smoke requires at least 12 natural roots")
    if challenge_root_count < 12 or challenge_root_count % 12:
        raise ValueError("challenge roots must be a positive multiple of 12")

    evaluators = build_known_hu_policy_value_logit_evaluators(root)
    behavior = TurnActorBehaviorDispatch(
        evaluators,
        model_id="known_hu_t1_t2_policyvalue_prior_dispatch_v1",
    )
    natural_config = BehaviorTraceCollectionConfig(
        seed_namespace=NATURAL_SEED_NAMESPACE,
        root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
    )
    challenge_config = BehaviorTraceCollectionConfig(
        seed_namespace=CHALLENGE_SEED_NAMESPACE,
        root_sampling_mode=TARGETED_JOKER_CHALLENGE,
        challenge_id=CHALLENGE_ID,
        challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
    )
    natural = collect_hu_behavior_traces(
        natural_config, behavior, root_count=natural_root_count
    )
    challenge = collect_hu_behavior_traces(
        challenge_config, behavior, root_count=challenge_root_count
    )
    natural = write_behavior_trace_dataset(
        natural, output / "natural" / "decisions.jsonl"
    )
    challenge = write_behavior_trace_dataset(
        challenge, output / "challenge" / "decisions.jsonl"
    )

    natural_rows = _evaluation_rows(natural.records, evaluators)
    challenge_rows = _evaluation_rows(challenge.records, evaluators)
    natural_rows_path = output / "natural" / "model_evaluations.jsonl"
    challenge_rows_path = output / "challenge" / "model_evaluations.jsonl"
    _write_canonical_jsonl(natural_rows_path, natural_rows)
    _write_canonical_jsonl(challenge_rows_path, challenge_rows)

    gate_config = build_temperature_gate_config()
    calibration = build_behavior_temperature_calibration(
        natural.records,
        natural_rows,
        challenge_records=challenge.records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=gate_config,
    )
    if calibration["promotion_eligible"] is not False:
        raise AssertionError("small self-sampled smoke must never become promotable")
    config_path = output / "temperature_gate_config.json"
    calibration_path = output / "calibration.json"
    _write_canonical_json(config_path, gate_config)
    _write_canonical_json(calibration_path, calibration)

    # Independent disk readback and complete raw-derived rebuild.
    natural_readback = read_behavior_trace_dataset(
        output / "natural" / "decisions.jsonl"
    )
    challenge_readback = read_behavior_trace_dataset(
        output / "challenge" / "decisions.jsonl"
    )
    natural_rows_readback = _read_canonical_jsonl(natural_rows_path)
    challenge_rows_readback = _read_canonical_jsonl(challenge_rows_path)
    for row, record in zip(natural_rows_readback, natural_readback.records):
        verify_model_evaluation_row(row, record)
    for row, record in zip(challenge_rows_readback, challenge_readback.records):
        verify_model_evaluation_row(row, record)
    calibration_readback = _read_canonical_json(calibration_path)
    rebuilt = verify_behavior_temperature_calibration(
        natural_readback.records,
        natural_rows_readback,
        calibration_readback,
        challenge_records=challenge_readback.records,
        challenge_evaluation_rows=challenge_rows_readback,
    )
    if rebuilt != calibration:
        raise AssertionError("calibration readback changed the artifact")

    split_counts = natural.manifest["behavior_decision_manifest"][
        "decision_counts_by_split"
    ]
    if any(int(split_counts[name]) <= 0 for name in ("fit", "dev", "test")):
        raise AssertionError("natural smoke did not cover fit/dev/test")
    targeted_cells = {
        (
            int(trace["challenge_target"]["turn"]),
            str(trace["challenge_target"]["actor"]),
            int(trace["challenge_target"]["visible_joker_count"]),
        )
        for trace in challenge.hidden_roots
    }
    expected_cells = {
        (turn, actor, joker)
        for turn in (1, 2)
        for actor in ("bb", "btn")
        for joker in (0, 1, 2)
    }
    if targeted_cells != expected_cells:
        raise AssertionError("Joker challenge did not cover all 12 target cells")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "bootstrap_kind": "self_sampled_frozen_prior_likelihood_wiring_smoke",
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "promotion_eligible": False,
        "production_gate_applied": True,
        "production_gate_passed": calibration["gate_result"][
            "all_required_gates_passed"
        ],
        "behavior_model_id": behavior.model_id,
        "behavior_model_sha256": behavior.model_sha256,
        "natural": {
            "root_count": natural.root_count,
            "decision_count": len(natural.records),
            "split_decision_counts": split_counts,
            "collection_content_sha256": natural.manifest[
                "collection_content_sha256"
            ],
            "decision_file_sha256": _sha256_file(
                output / "natural" / "decisions.jsonl"
            ),
            "hidden_roots_file_sha256": _sha256_file(
                output / "natural" / "roots.jsonl"
            ),
            "evaluation_file_sha256": _sha256_file(natural_rows_path),
        },
        "joker_challenge": {
            "root_count": challenge.root_count,
            "decision_count": len(challenge.records),
            "target_cell_count": len(targeted_cells),
            "target_cells_sha256": canonical_sha256(
                [list(cell) for cell in sorted(targeted_cells)]
            ),
            "root_namespace_prefix": f"{CHALLENGE_ID}/",
            "collection_content_sha256": challenge.manifest[
                "collection_content_sha256"
            ],
            "decision_file_sha256": _sha256_file(
                output / "challenge" / "decisions.jsonl"
            ),
            "hidden_roots_file_sha256": _sha256_file(
                output / "challenge" / "roots.jsonl"
            ),
            "evaluation_file_sha256": _sha256_file(challenge_rows_path),
        },
        "gate_config_sha256": gate_config["gate_config_sha256"],
        "gate_result_sha256": calibration["gate_result"]["gate_result_sha256"],
        "calibration_artifact_sha256": calibration["artifact_sha256"],
        "gate_failures": list(calibration["gate_result"]["failures"]),
        "readback_verified": True,
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = output / "result.json"
    _write_canonical_json(report_path, report)
    if _read_canonical_json(report_path) != report:
        raise AssertionError("report readback mismatch")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai/reports/m3_behavior_calibration_smoke_20260713"),
    )
    parser.add_argument("--natural-roots", type=int, default=12)
    parser.add_argument("--challenge-roots", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_behavior_calibration_smoke(
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        natural_root_count=args.natural_roots,
        challenge_root_count=args.challenge_roots,
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
