"""Strict-OOF top-4 search diagnostic for frozen No-Go Attempt05 artifacts.

For each old-dev900 state this diagnostic uses only the fold predictor whose
fold index equals that state's held-out identity fold.  It takes the model's
nonbaseline top four, adds the explicit baseline, locks one action by the
existing independent c2 ``selection_score`` with ActionKey tie-breaking, and
evaluates that lock with the independent e64 score/paired-delta labels.

This is Attempt06 design evidence only.  It cannot promote Attempt05, authorize
a new audit, tune a threshold, fit a model, or mutate a runtime profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import action_key_from_payload
from .hu_m43_attempt05_model import HuM43Attempt05Model
from .hu_m43_attempt05_training import (
    ATTEMPT05_DEV_SCHEMA,
    ATTEMPT05_FAMILIES,
    ATTEMPT05_PROFILES,
    DEFAULT_ATTEMPT02_TRAIN,
    DEFAULT_ATTEMPT03_CONSUMED_PRECAL,
    DEFAULT_ATTEMPT03_FIT,
    load_attempt05_dev900,
)
from .train_hu_m4_joint_model import read_teacher_jsonl


ATTEMPT05_TOP4_DIAGNOSTIC_SCHEMA = "hu_m43_attempt05_strict_oof_top4_c2_e64_diagnostic_v1"
_TAIL_NAMES = ("p95", "p99", "max")
_TAIL_LIMITS = (25.0, 40.0, 50.0)


def diagnose_top4_search(architecture_dir: str | Path) -> dict[str, Any]:
    root = Path(architecture_dir)
    architecture_bytes = (root / "architecture_comparison.json").read_bytes()
    architecture = json.loads(architecture_bytes)
    if architecture.get("schema") != ATTEMPT05_DEV_SCHEMA:
        raise ValueError("Attempt05 diagnostic architecture schema mismatch")
    if architecture.get("selected_family") is not None or "no_go" not in str(
        architecture.get("status", "")
    ):
        raise ValueError("Attempt05 top4 diagnostic requires the frozen No-Go result")

    data = load_attempt05_dev900()
    raw_rows = [
        *read_teacher_jsonl(DEFAULT_ATTEMPT02_TRAIN),
        *read_teacher_jsonl(DEFAULT_ATTEMPT03_FIT),
        *read_teacher_jsonl(DEFAULT_ATTEMPT03_CONSUMED_PRECAL),
    ]
    raw_by_fingerprint = {
        str(row["observation_fingerprint"]): row for row in raw_rows
    }
    if len(raw_by_fingerprint) != 900:
        raise ValueError("Attempt05 diagnostic raw identities changed")
    _validate_c2_e64_contract(raw_rows)

    family_reports: dict[str, Any] = {}
    for family in ATTEMPT05_FAMILIES:
        architecture_family = architecture["families"][family]
        artifact = root / f"{family}_candidate.pkl"
        artifact_sha256 = _file_sha256(artifact)
        if artifact_sha256 != architecture_family["candidate_artifact_sha256"]:
            raise ValueError("Attempt05 diagnostic artifact hash mismatch")
        model = HuM43Attempt05Model.load(
            artifact, expected_sha256=artifact_sha256
        )
        folds = {predictor.fold_index: predictor for predictor in model.fold_predictors}
        architecture_oof = {
            row["fingerprint"]: row for row in architecture_family["rows"]
        }
        state_rows = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            for index, sample in enumerate(data.samples):
                fold = int(data.fold_ids[index])
                output = folds[fold].predict(
                    sample.policy_sample, baseline_index=sample.baseline_index
                )
                rank = np.asarray(output.rank_score, dtype=np.float64)
                actions = sample.policy_sample["actions"]
                action_keys = tuple(action_key_from_payload(action) for action in actions)
                nonbaseline = [
                    action_index
                    for action_index in range(len(actions))
                    if action_index != sample.baseline_index
                ]
                top4 = sorted(
                    nonbaseline,
                    key=lambda action_index: (
                        -float(rank[action_index]),
                        action_keys[action_index].sort_key(),
                    ),
                )[:4]
                if len(top4) != 4:
                    raise ValueError("Attempt05 diagnostic state has fewer than four candidates")
                oof_top1_key = action_keys[top4[0]].to_token()
                expected_top1_key = architecture_oof[
                    sample.observation_fingerprint
                ]["candidate_action_key"]
                if oof_top1_key != expected_top1_key:
                    raise ValueError("Attempt05 strict-OOF top1 reconstruction mismatch")

                raw = raw_by_fingerprint[sample.observation_fingerprint]
                raw_actions = {
                    action_key_from_payload(action): action
                    for action in raw["actions"]
                }
                if set(raw_actions) != set(action_keys):
                    raise ValueError("Attempt05 diagnostic raw/legal ActionKey set mismatch")
                selection_score = np.asarray(
                    [float(raw_actions[key]["selection_score"]) for key in action_keys]
                )
                independent_score = np.asarray(
                    [float(raw_actions[key]["score"]) for key in action_keys]
                )
                independent_delta = np.asarray(
                    [
                        float(raw_actions[key]["paired_delta_vs_baseline"]["mean"])
                        for key in action_keys
                    ]
                )
                np.testing.assert_allclose(
                    independent_score, sample.teacher_scores, rtol=0.0, atol=0.0
                )
                np.testing.assert_allclose(
                    independent_delta,
                    sample.teacher_paired_delta_mean,
                    rtol=0.0,
                    atol=0.0,
                )
                oracle = min(
                    top4,
                    key=lambda action_index: (
                        -float(independent_delta[action_index]),
                        action_keys[action_index].sort_key(),
                    ),
                )
                search_pool = [*top4, sample.baseline_index]
                selected = min(
                    search_pool,
                    key=lambda action_index: (
                        -float(selection_score[action_index]),
                        action_keys[action_index].sort_key(),
                    ),
                )
                tail_arrays = (
                    np.asarray(sample.downside_loss_p95),
                    np.asarray(sample.downside_loss_p99),
                    np.asarray(sample.downside_loss_max),
                )
                selected_tails = [float(values[selected]) for values in tail_arrays]
                oracle_tails = [float(values[oracle]) for values in tail_arrays]
                state_rows.append(
                    {
                        "fingerprint": sample.observation_fingerprint,
                        "profile": data.profiles[index],
                        "held_out_fold": fold,
                        "top4_action_keys": [action_keys[value].to_token() for value in top4],
                        "top4_has_positive_e64_delta": bool(
                            np.any(independent_delta[top4] > 0.0)
                        ),
                        "oracle_top4_action_key": action_keys[oracle].to_token(),
                        "oracle_top4_e64_score": float(independent_score[oracle]),
                        "oracle_top4_e64_delta": float(independent_delta[oracle]),
                        "oracle_top4_downside": oracle_tails,
                        "c2_selected_action_key": action_keys[selected].to_token(),
                        "c2_selected_baseline": selected == sample.baseline_index,
                        "c2_selection_score": float(selection_score[selected]),
                        "c2_selected_e64_score": float(independent_score[selected]),
                        "c2_selected_e64_delta": float(independent_delta[selected]),
                        "c2_selected_downside": selected_tails,
                    }
                )
        family_reports[family] = {
            "artifact": artifact.as_posix(),
            "artifact_sha256": artifact_sha256,
            "rank_source": "one_strictly_held_out_fold_predictor_per_state",
            "identity_leakage": 0,
            "top1_reconstruction_matches_architecture_report": True,
            "overall": _metrics(state_rows),
            "profile": {
                profile: _metrics(
                    [row for row in state_rows if row["profile"] == profile]
                )
                for profile in ATTEMPT05_PROFILES
            },
            "row_digest_sha256": _canonical_sha256(state_rows),
            "rows": state_rows,
        }

    return {
        "schema": ATTEMPT05_TOP4_DIAGNOSTIC_SCHEMA,
        "status": "attempt06_design_diagnostic_only_attempt05_remains_no_go",
        "architecture_report_sha256": hashlib.sha256(architecture_bytes).hexdigest(),
        "architecture_status": architecture["status"],
        "architecture_selected_family": architecture["selected_family"],
        "candidate_generation": "strict_oof_model_nonbaseline_top4",
        "candidate_lock": (
            "top4_plus_baseline_argmax_existing_c2_selection_score_"
            "then_ActionKey"
        ),
        "independent_evaluation": "existing_e64_score_and_paired_delta",
        "c2_e64_independence": {
            "candidate_samples": 2,
            "evaluation_samples": 64,
            "separate_rng_contract_verified": True,
        },
        "strict_oof": {
            "folds": 5,
            "state_fold_source": "Attempt05 deterministic data.fold_ids",
            "predictor_fold_equals_state_held_out_fold": True,
            "identity_leakage": 0,
            "five_fold_ensemble_applied_to_dev900": False,
        },
        "families": family_reports,
        "claim_boundary": {
            "attempt05_promotion_authorized": False,
            "new_audit_authorized": False,
            "threshold_selected": False,
            "model_fitted": False,
            "new_seed_used": False,
            "fresh_or_locked_data_opened": False,
            "teacher_values_reported_as_realized_ev": False,
            "current_profile_mutated": False,
        },
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Attempt05 diagnostic metric group is empty")
    oracle_delta = np.asarray([row["oracle_top4_e64_delta"] for row in rows])
    selected_delta = np.asarray([row["c2_selected_e64_delta"] for row in rows])
    selected_score = np.asarray([row["c2_selected_e64_score"] for row in rows])
    fires = np.asarray([not row["c2_selected_baseline"] for row in rows], dtype=bool)
    fired_delta = selected_delta[fires]
    top4_coverage = np.asarray(
        [row["top4_has_positive_e64_delta"] for row in rows], dtype=bool
    )
    selected_tails = np.asarray(
        [row["c2_selected_downside"] for row in rows if not row["c2_selected_baseline"]],
        dtype=np.float64,
    )
    oracle_tails = np.asarray(
        [row["oracle_top4_downside"] for row in rows], dtype=np.float64
    )
    return {
        "states": len(rows),
        "top4_positive_coverage": float(np.mean(top4_coverage)),
        "oracle_top4": {
            "mean_e64_delta": float(np.mean(oracle_delta)),
            "positive_rate": float(np.mean(oracle_delta > 0.0)),
            "mean_e64_score": float(
                np.mean([row["oracle_top4_e64_score"] for row in rows])
            ),
            "selected_downside_target_maxima": dict(
                zip(_TAIL_NAMES, np.max(oracle_tails, axis=0).tolist(), strict=True)
            ),
        },
        "c2_search_selected_e64": {
            "fires": int(np.sum(fires)),
            "fire_rate": float(np.mean(fires)),
            "positive_fires": int(np.sum(fired_delta > 0.0)),
            "false_positive_fires": int(np.sum(fired_delta <= 0.0)),
            "false_positive_rate_per_fire": (
                float(np.mean(fired_delta <= 0.0)) if fired_delta.size else None
            ),
            "mean_delta_per_state": float(np.mean(selected_delta)),
            "mean_delta_per_fire": (
                float(np.mean(fired_delta)) if fired_delta.size else None
            ),
            "positive_rate_per_state": float(np.mean(selected_delta > 0.0)),
            "positive_rate_per_fire": (
                float(np.mean(fired_delta > 0.0)) if fired_delta.size else None
            ),
            "mean_absolute_e64_score": float(np.mean(selected_score)),
            "selected_downside_target_maxima": (
                dict(
                    zip(
                        _TAIL_NAMES,
                        np.max(selected_tails, axis=0).tolist(),
                        strict=True,
                    )
                )
                if selected_tails.size
                else dict.fromkeys(_TAIL_NAMES)
            ),
            "selected_downside_target_means": (
                dict(
                    zip(
                        _TAIL_NAMES,
                        np.mean(selected_tails, axis=0).tolist(),
                        strict=True,
                    )
                )
                if selected_tails.size
                else dict.fromkeys(_TAIL_NAMES)
            ),
            "tail_limit_violation_rate_per_fire": (
                {
                    name: float(np.mean(selected_tails[:, index] > limit))
                    for index, (name, limit) in enumerate(
                        zip(_TAIL_NAMES, _TAIL_LIMITS, strict=True)
                    )
                }
                if selected_tails.size
                else dict.fromkeys(_TAIL_NAMES)
            ),
        },
    }


def _validate_c2_e64_contract(rows: Sequence[Mapping[str, Any]]) -> None:
    candidate_seeds = set()
    evaluation_seeds = set()
    for row in rows:
        search = row.get("search_config")
        if not isinstance(search, Mapping):
            raise ValueError("Attempt05 diagnostic row lacks search_config")
        if int(search.get("candidate_samples", -1)) != 2 or int(
            search.get("evaluation_samples", -1)
        ) != 64:
            raise ValueError("Attempt05 diagnostic c2/e64 contract changed")
        candidate_seeds.add(str(search.get("candidate_seed")))
        evaluation_seeds.add(str(search.get("evaluation_seed")))
        for action in row["actions"]:
            paired = action.get("paired_delta_vs_baseline")
            if not isinstance(paired, Mapping) or int(paired.get("count", -1)) != 64:
                raise ValueError("Attempt05 diagnostic paired e64 count changed")
            for field in ("selection_score", "score"):
                if not math.isfinite(float(action[field])):
                    raise ValueError("Attempt05 diagnostic score is non-finite")
    if candidate_seeds & evaluation_seeds:
        raise ValueError("Attempt05 diagnostic candidate/evaluation RNG seeds overlap")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = diagnose_top4_search(args.architecture_dir)
    _write_no_clobber(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ATTEMPT05_TOP4_DIAGNOSTIC_SCHEMA", "diagnose_top4_search"]
