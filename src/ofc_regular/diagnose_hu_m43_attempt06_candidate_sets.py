"""Final multiple-use old-dev candidate-set diagnostic for Attempt06 design.

No model is fitted and no threshold is selected.  Each state is scored only by
the LambdaRank fold predictor that held out that state's observation identity.
The six user-fixed candidate-set rules are compared with existing c2 selection
scores and independent e64 labels.  Results are development-only because the
same old dev900 has already been inspected multiple times.
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

from .action_key import ActionKey, action_key_from_payload
from .hu_m43_attempt05_model import HuM43Attempt05Model
from .hu_m43_attempt05_training import (
    ATTEMPT05_DEV_SCHEMA,
    ATTEMPT05_PROFILES,
    DEFAULT_ATTEMPT02_TRAIN,
    DEFAULT_ATTEMPT03_CONSUMED_PRECAL,
    DEFAULT_ATTEMPT03_FIT,
    load_attempt05_dev900,
)
from .train_hu_m4_joint_model import read_teacher_jsonl


ATTEMPT06_CANDIDATE_SET_DIAGNOSTIC_SCHEMA = (
    "hu_m43_attempt06_lambda_strict_oof_candidate_sets_dev900_v1"
)
_TAIL_NAMES = ("p95", "p99", "max")
_TAIL_LIMITS = (25.0, 40.0, 50.0)
_COVERAGE_OVERALL_MIN = 0.70
_COVERAGE_EACH_PROFILE_MIN = 0.60
_SET_ORDER = (
    "rank_top4",
    "gain_top4",
    "rank_top6",
    "rank_top8",
    "union_rank4_gain4",
    "rank_top12",
)
_NOMINAL_MAX = {
    "rank_top4": 4,
    "gain_top4": 4,
    "rank_top6": 6,
    "rank_top8": 8,
    "union_rank4_gain4": 8,
    "rank_top12": 12,
}


def diagnose_candidate_sets(architecture_dir: str | Path) -> dict[str, Any]:
    root = Path(architecture_dir)
    architecture_bytes = (root / "architecture_comparison.json").read_bytes()
    architecture = json.loads(architecture_bytes)
    if architecture.get("schema") != ATTEMPT05_DEV_SCHEMA:
        raise ValueError("Attempt06 diagnostic architecture schema mismatch")
    if architecture.get("selected_family") is not None or "no_go" not in str(
        architecture.get("status", "")
    ):
        raise ValueError("Attempt06 diagnostic requires frozen Attempt05 No-Go")

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
        raise ValueError("Attempt06 diagnostic raw identities changed")
    _validate_c2_e64(raw_rows)

    family_report = architecture["families"]["lambda_rank"]
    artifact = root / "lambda_rank_candidate.pkl"
    artifact_sha256 = _file_sha256(artifact)
    if artifact_sha256 != family_report["candidate_artifact_sha256"]:
        raise ValueError("Attempt06 Lambda artifact digest mismatch")
    model = HuM43Attempt05Model.load(artifact, expected_sha256=artifact_sha256)
    if model.family != "lambda_rank":
        raise ValueError("Attempt06 candidate diagnostic requires LambdaRank")
    predictors = {fold.fold_index: fold for fold in model.fold_predictors}
    expected_oof_top1 = {
        row["fingerprint"]: row["candidate_action_key"]
        for row in family_report["rows"]
    }
    rows_by_set: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _SET_ORDER
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        for state_index, sample in enumerate(data.samples):
            fold_index = int(data.fold_ids[state_index])
            output = predictors[fold_index].predict(
                sample.policy_sample, baseline_index=sample.baseline_index
            )
            rank = np.asarray(output.rank_score, dtype=np.float64)
            gain = np.asarray(output.gain_probability, dtype=np.float64)
            actions = sample.policy_sample["actions"]
            keys = tuple(action_key_from_payload(action) for action in actions)
            nonbaseline = [
                index
                for index in range(len(actions))
                if index != sample.baseline_index
            ]
            rank_order = sorted(
                nonbaseline,
                key=lambda index: (-float(rank[index]), keys[index].sort_key()),
            )
            gain_order = sorted(
                nonbaseline,
                key=lambda index: (-float(gain[index]), keys[index].sort_key()),
            )
            if len(rank_order) < 4 or len(gain_order) < 4:
                raise ValueError("Attempt06 state has fewer than four legal nonbaseline actions")
            if keys[rank_order[0]].to_token() != expected_oof_top1[
                sample.observation_fingerprint
            ]:
                raise ValueError("Attempt06 strict-OOF top1 reconstruction mismatch")
            rank4 = tuple(rank_order[:4])
            gain4 = tuple(gain_order[:4])
            union = tuple(
                sorted(
                    set((*rank4, *gain4)),
                    key=lambda index: keys[index].sort_key(),
                )
            )
            candidate_sets = {
                "rank_top4": rank4,
                "gain_top4": gain4,
                "rank_top6": tuple(rank_order[:6]),
                "rank_top8": tuple(rank_order[:8]),
                "union_rank4_gain4": union,
                "rank_top12": tuple(rank_order[:12]),
            }

            raw = raw_by_fingerprint[sample.observation_fingerprint]
            raw_by_key = {
                action_key_from_payload(action): action for action in raw["actions"]
            }
            if set(raw_by_key) != set(keys):
                raise ValueError("Attempt06 raw/legal ActionKey set mismatch")
            c2 = np.asarray(
                [float(raw_by_key[key]["selection_score"]) for key in keys]
            )
            e64_score = np.asarray(
                [float(raw_by_key[key]["score"]) for key in keys]
            )
            e64_delta = np.asarray(
                [
                    float(raw_by_key[key]["paired_delta_vs_baseline"]["mean"])
                    for key in keys
                ]
            )
            np.testing.assert_array_equal(e64_score, sample.teacher_scores)
            np.testing.assert_array_equal(
                e64_delta, sample.teacher_paired_delta_mean
            )
            tail_arrays = (
                np.asarray(sample.downside_loss_p95),
                np.asarray(sample.downside_loss_p99),
                np.asarray(sample.downside_loss_max),
            )

            for set_name, candidates in candidate_sets.items():
                if sample.baseline_index in candidates or len(set(candidates)) != len(
                    candidates
                ):
                    raise ValueError("Attempt06 candidate set is not unique nonbaseline")
                oracle = _argmax_with_action_key(e64_delta, candidates, keys)
                selected = _argmax_with_action_key(
                    c2, (*candidates, sample.baseline_index), keys
                )
                rows_by_set[set_name].append(
                    {
                        "fingerprint": sample.observation_fingerprint,
                        "profile": data.profiles[state_index],
                        "held_out_fold": fold_index,
                        "candidate_count": len(candidates),
                        "candidate_action_keys": [
                            keys[index].to_token()
                            for index in sorted(
                                candidates, key=lambda index: keys[index].sort_key()
                            )
                        ],
                        "positive_coverage": bool(np.any(e64_delta[list(candidates)] > 0.0)),
                        "oracle_e64_delta": float(e64_delta[oracle]),
                        "oracle_e64_score": float(e64_score[oracle]),
                        "oracle_downside": [
                            float(values[oracle]) for values in tail_arrays
                        ],
                        "c2_selected_action_key": keys[selected].to_token(),
                        "c2_selected_baseline": selected == sample.baseline_index,
                        "c2_selection_score": float(c2[selected]),
                        "c2_selected_e64_delta": float(e64_delta[selected]),
                        "c2_selected_e64_score": float(e64_score[selected]),
                        "c2_selected_downside": [
                            float(values[selected]) for values in tail_arrays
                        ],
                    }
                )

    set_reports = {}
    for set_name in _SET_ORDER:
        rows = rows_by_set[set_name]
        overall = _metrics(rows)
        profiles = {
            profile: _metrics([row for row in rows if row["profile"] == profile])
            for profile in ATTEMPT05_PROFILES
        }
        gate = bool(
            overall["oracle"]["positive_coverage"] >= _COVERAGE_OVERALL_MIN
            and all(
                report["oracle"]["positive_coverage"]
                >= _COVERAGE_EACH_PROFILE_MIN
                for report in profiles.values()
            )
        )
        set_reports[set_name] = {
            "nominal_max_candidates": _NOMINAL_MAX[set_name],
            "overall": overall,
            "profile": profiles,
            "coverage_gate_pass": gate,
            "row_digest_sha256": _canonical_sha256(rows),
            "rows": rows,
        }

    passing = [name for name in _SET_ORDER if set_reports[name]["coverage_gate_pass"]]
    passing.sort(
        key=lambda name: (
            _NOMINAL_MAX[name],
            float(set_reports[name]["overall"]["candidate_count_mean"]),
            _SET_ORDER.index(name),
        )
    )
    recommendation = passing[0] if passing else None
    return {
        "schema": ATTEMPT06_CANDIDATE_SET_DIAGNOSTIC_SCHEMA,
        "status": (
            "development_candidate_set_proposed_no_promotion_authority"
            if recommendation is not None
            else "development_no_go_no_candidate_set_meets_coverage_gate"
        ),
        "data_classification": "multiple_use_development_only_not_fresh",
        "states": 900,
        "family": "lambda_rank",
        "artifact": artifact.as_posix(),
        "artifact_sha256": artifact_sha256,
        "candidate_sets_fixed_before_execution": list(_SET_ORDER),
        "strict_oof": {
            "predictor_fold_equals_state_held_out_fold": True,
            "identity_leakage": 0,
            "ensemble_in_sample_leakage_used": False,
        },
        "selection": {
            "candidate_ties": "ActionKey",
            "oracle_ties": "ActionKey",
            "c2_ties": "ActionKey",
            "baseline_added_after_candidate_set": True,
            "independent_evaluation": "e64 score and paired delta",
        },
        "coverage_gate": {
            "overall_min": _COVERAGE_OVERALL_MIN,
            "each_profile_min": _COVERAGE_EACH_PROFILE_MIN,
            "minimum_set_order": (
                "nominal_max_candidates_then_mean_actual_count_then_fixed_set_order"
            ),
        },
        "recommended_minimum_set": recommendation,
        "sets": set_reports,
        "claim_boundary": {
            "new_training_performed": False,
            "threshold_selected": False,
            "new_seed_used": False,
            "fresh_or_locked_data_opened": False,
            "attempt05_promotion_authorized": False,
            "attempt06_runtime_authorized": False,
            "new_audit_authorized": False,
            "teacher_values_reported_as_realized_ev": False,
            "current_profile_mutated": False,
        },
    }


def _argmax_with_action_key(
    values: np.ndarray, candidates: Sequence[int], keys: Sequence[ActionKey]
) -> int:
    return min(
        (int(index) for index in candidates),
        key=lambda index: (-float(values[index]), keys[index].sort_key()),
    )


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Attempt06 metric group is empty")
    counts = np.asarray([row["candidate_count"] for row in rows], dtype=np.int32)
    coverage = np.asarray([row["positive_coverage"] for row in rows], dtype=bool)
    oracle_delta = np.asarray([row["oracle_e64_delta"] for row in rows])
    selected_delta = np.asarray([row["c2_selected_e64_delta"] for row in rows])
    fires = np.asarray([not row["c2_selected_baseline"] for row in rows], dtype=bool)
    fired_delta = selected_delta[fires]
    selected_tails = np.asarray(
        [row["c2_selected_downside"] for row in rows if not row["c2_selected_baseline"]],
        dtype=np.float64,
    )
    oracle_tails = np.asarray([row["oracle_downside"] for row in rows])
    return {
        "states": len(rows),
        "candidate_count_min": int(np.min(counts)),
        "candidate_count_mean": float(np.mean(counts)),
        "candidate_count_max": int(np.max(counts)),
        "oracle": {
            "positive_coverage": float(np.mean(coverage)),
            "mean_e64_delta": float(np.mean(oracle_delta)),
            "positive_rate": float(np.mean(oracle_delta > 0.0)),
            "mean_e64_score": float(
                np.mean([row["oracle_e64_score"] for row in rows])
            ),
            "downside_target_maxima": dict(
                zip(_TAIL_NAMES, np.max(oracle_tails, axis=0).tolist(), strict=True)
            ),
        },
        "c2_selection_e64": {
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
            "mean_e64_score": float(
                np.mean([row["c2_selected_e64_score"] for row in rows])
            ),
            "downside_target_maxima": (
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
            "downside_target_means": (
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


def _validate_c2_e64(rows: Sequence[Mapping[str, Any]]) -> None:
    candidate_seeds = set()
    evaluation_seeds = set()
    for row in rows:
        search = row.get("search_config")
        if not isinstance(search, Mapping):
            raise ValueError("Attempt06 row lacks search_config")
        if int(search.get("candidate_samples", -1)) != 2 or int(
            search.get("evaluation_samples", -1)
        ) != 64:
            raise ValueError("Attempt06 c2/e64 contract changed")
        candidate_seeds.add(str(search.get("candidate_seed")))
        evaluation_seeds.add(str(search.get("evaluation_seed")))
        for action in row["actions"]:
            paired = action.get("paired_delta_vs_baseline")
            if not isinstance(paired, Mapping) or int(paired.get("count", -1)) != 64:
                raise ValueError("Attempt06 e64 paired count changed")
            if not all(
                math.isfinite(float(action[field]))
                for field in ("selection_score", "score")
            ):
                raise ValueError("Attempt06 score is non-finite")
    if candidate_seeds & evaluation_seeds:
        raise ValueError("Attempt06 candidate/evaluation seeds overlap")


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
    report = diagnose_candidate_sets(args.architecture_dir)
    _write_no_clobber(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "recommended_minimum_set": report["recommended_minimum_set"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT06_CANDIDATE_SET_DIAGNOSTIC_SCHEMA",
    "diagnose_candidate_sets",
]
