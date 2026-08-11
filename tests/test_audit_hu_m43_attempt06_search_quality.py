from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ofc_regular.action_key import ActionKey, action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_belief import sample_hidden_card_particles
from ofc_regular.audit_hu_m43_attempt06_search_quality import (
    ATTEMPT06_SEARCH_QUALITY_AUDIT_SCHEMA,
    audit_attempt06_search_quality,
    validate_attempt06_teacher_row,
    write_attempt06_search_quality_report,
)
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt06_contract import (
    M43_ATTEMPT06_LAMBDA_SHA256,
    M43_ATTEMPT06_PLAN_SHA256,
    M43_ATTEMPT06_PROFILES,
    M43_ATTEMPT06_SEED_STRIDE,
)
from ofc_regular.hu_m43_attempt06_spot import (
    AUDIT_OUTPUT_CONSUMPTION_SCHEMA,
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
    RECEIVE_MERGE_SCHEMA,
    ROOT_REMATERIALIZATION_MODE,
)
from ofc_regular.hu_m43_attempt06_teacher import (
    ATTEMPT06_CANDIDATE_SEED_START,
    ATTEMPT06_CHILD_SEED_START,
    ATTEMPT06_EVALUATION_SEED_START,
    ATTEMPT06_HAND_SEED_START,
    ATTEMPT06_ROOT_GENERATION_POLICY,
    ATTEMPT06_ROOT_PROVENANCE_SCHEMA,
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_SOLVER_ID,
    ATTEMPT06_TEACHER_SCHEMA,
)
from ofc_regular.hu_m4_t1_teacher import M4_PAIRED_DELTA_SUMMARY_SCHEMA
from ofc_regular.hu_m4_teacher_contract import T1_SECOND_LIVE_SCHEDULE
from ofc_regular.state import Board


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "configs/hu_joint_policy_m43_attempt06.json"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_digest(tokens: list[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()


def _set_digest(tokens: list[str]) -> str:
    keys = sorted((ActionKey.from_token(token) for token in tokens), key=ActionKey.sort_key)
    return _ordered_digest([key.to_token() for key in keys])


def _observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _raw_deltas(
    *,
    mean: float,
    minimum: float = -20.0,
    p01: float = -15.0,
    p05: float = -10.0,
) -> list[float]:
    if mean == 0.0 and minimum == -20.0 and p01 == -15.0 and p05 == -10.0:
        return [0.0] * 128
    prefix = [minimum, p01, p01, p05, p05, p05, p05, p05]
    remainder = (mean * 128 - sum(prefix)) / 120
    if remainder < p05:
        raise ValueError("synthetic paired remainder would break quantile order")
    return [*prefix, *([remainder] * 120)]


def _pair(raw: list[float]) -> dict[str, object]:
    array = np.asarray(raw, dtype=np.float64)
    standard_deviation = float(np.std(array, ddof=1))
    return {
        "schema": M4_PAIRED_DELTA_SUMMARY_SCHEMA,
        "count": 128,
        "mean": float(np.mean(array)),
        "standard_error": standard_deviation / math.sqrt(128),
        "std": standard_deviation,
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01, method="linear")),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "p25": float(np.quantile(array, 0.25, method="linear")),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p75": float(np.quantile(array, 0.75, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "p99": float(np.quantile(array, 0.99, method="linear")),
        "max": float(np.max(array)),
        "lt0_rate": float(np.mean(array < 0.0)),
        "le_neg6_rate": float(np.mean(array <= -6.0)),
        "le_neg12_rate": float(np.mean(array <= -12.0)),
        "le_neg20_rate": float(np.mean(array <= -20.0)),
    }


def _loss(pair: dict[str, object]) -> dict[str, float]:
    return {
        "p95": max(0.0, -float(pair["p05"])),
        "p99": max(0.0, -float(pair["p01"])),
        "max": max(0.0, -float(pair["min"])),
    }


def _teacher(
    root_index: int,
    *,
    fired: bool,
    mean: float,
    minimum: float = -20.0,
    p01: float = -15.0,
    p05: float = -10.0,
) -> tuple[dict[str, object], str]:
    observation = _observation()
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    legal_tokens = [action_key(action).to_token() for action in legal]
    baseline_index = len(legal) - 1
    baseline = legal_tokens[baseline_index]
    scores = [float(len(legal) - index) for index in range(len(legal))]
    ranked_nonbaseline = sorted(
        (index for index in range(len(legal)) if index != baseline_index),
        key=lambda index: (
            -scores[index],
            action_key(legal[index]).sort_key(),
        ),
    )
    top_indices = ranked_nonbaseline[:8]
    top8 = [legal_tokens[index] for index in top_indices]
    candidate_indices = [*top_indices, baseline_index]
    candidate_tokens = [legal_tokens[index] for index in candidate_indices]
    selected_position = 0 if fired else 8
    selected_index = candidate_indices[selected_position]
    selected = candidate_tokens[selected_position]
    selected_raw = _raw_deltas(
        mean=mean, minimum=minimum, p01=p01, p05=p05
    )
    selected_pair = _pair(selected_raw)
    selected_loss = _loss(selected_pair)
    zero_pair = _pair([0.0] * 128)
    zero_loss = _loss(zero_pair)
    evaluated_positions = list(dict.fromkeys((selected_position, 8)))
    candidate_rows = []
    for position, original_index in enumerate(candidate_indices):
        is_selected = position == selected_position
        is_evaluated = position in evaluated_positions
        pair = (
            copy.deepcopy(selected_pair if is_selected else zero_pair)
            if is_evaluated
            else None
        )
        loss = (
            copy.deepcopy(selected_loss if is_selected else zero_loss)
            if is_evaluated
            else None
        )
        selection_mean = (
            20.0
            if is_selected
            else 10.0 - position
        )
        evaluation_mean = (
            10.0 + mean
            if is_selected
            else (10.0 if is_evaluated else None)
        )
        action = legal[original_index]
        candidate_rows.append(
            {
                "candidate_position": position,
                "original_legal_index": original_index,
                "learned_nonbaseline_rank": position + 1 if position < 8 else None,
                "action_key": candidate_tokens[position],
                "placements": [list(value) for value in action.placements],
                "discards": list(action.discards),
                "model_rank_score": scores[original_index],
                "model_rank_disagreement": 0.25,
                "candidate_mean": selection_mean,
                "candidate_standard_error": 0.1,
                "evaluation_mean": evaluation_mean,
                "evaluation_standard_error": 0.2 if is_evaluated else None,
                "paired_evaluation_delta_vs_baseline": pair,
                "paired_evaluation_override_loss": loss,
                "paired_evaluation_state_mean_loss_diagnostic": (
                    max(0.0, -float(pair["mean"]))
                    if pair is not None
                    else None
                ),
                "evaluation_scope": (
                    "locked_action_and_explicit_baseline"
                    if is_evaluated
                    else "not_evaluated_after_c8_lock"
                ),
                "is_explicit_baseline": position == 8,
                "selected_by_c8": is_selected,
                "evaluation_sample_best": is_selected if is_evaluated else None,
            }
        )
    legal_rows = [
        {
            "original_legal_index": index,
            "action_key": token,
            "model_rank_score": scores[index],
            "model_rank_disagreement": 0.25,
            "in_learned_top8": index in set(top_indices),
            "is_explicit_baseline": index == baseline_index,
        }
        for index, token in enumerate(legal_tokens)
    ]
    run_id = f"attempt06-synthetic:shard={root_index}"
    root_run_id = (
        f"{run_id}:root={root_index}:seed="
        f"{ATTEMPT06_HAND_SEED_START + M43_ATTEMPT06_SEED_STRIDE * root_index}:"
        f"obs={observation.fingerprint()}"
    )
    candidate_batch = sample_hidden_card_particles(
        observation,
        base_seed=ATTEMPT06_CANDIDATE_SEED_START
        + M43_ATTEMPT06_SEED_STRIDE * root_index,
        run_id=f"{root_run_id}:candidate_selection",
        sample_count=8,
    )
    evaluation_batch = sample_hidden_card_particles(
        observation,
        base_seed=ATTEMPT06_EVALUATION_SEED_START
        + M43_ATTEMPT06_SEED_STRIDE * root_index,
        run_id=f"{root_run_id}:locked_evaluation",
        sample_count=128,
    )
    teacher = {
        "status": "ok",
        "schema": ATTEMPT06_TEACHER_SCHEMA,
        "solver_id": ATTEMPT06_SOLVER_ID,
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "observation_fingerprint": observation.fingerprint(),
        "action_key_schema": "regular_ofc_action_key_v1",
        "frozen_candidate_generator": {
            "family": "lambda_rank",
            "model_id": "hu-m43-attempt05-lambda_rank-dev900-oof",
            "artifact_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
            "purpose": "candidate_generation_only",
            "runtime_authorized": False,
            "profile_runtime_feature": False,
        },
        "legal_action_count": len(legal),
        "legal_action_set_digest": _set_digest(legal_tokens),
        "legal_action_order_digest": _ordered_digest(legal_tokens),
        "legal_action_keys": legal_tokens,
        "legal_actions": legal_rows,
        "learned_top8_action_keys": top8,
        "learned_top8_order_digest": _ordered_digest(top8),
        "candidate_action_count": 9,
        "candidate_action_set_digest": _set_digest(candidate_tokens),
        "candidate_action_order_digest": _ordered_digest(candidate_tokens),
        "evaluation_action_count": len(evaluated_positions),
        "evaluation_action_keys": [
            candidate_tokens[position] for position in evaluated_positions
        ],
        "evaluation_action_order_digest": _ordered_digest(
            [candidate_tokens[position] for position in evaluated_positions]
        ),
        "evaluation_scope": "c8_locked_action_plus_explicit_baseline_only",
        "baseline_action_key": baseline,
        "baseline_original_legal_index": baseline_index,
        "selected_action_key": selected,
        "selected_action_original_legal_index": selected_index,
        "selected_action_candidate_position": selected_position,
        "override_fired": fired,
        "selection_score_gap": (
            sorted(
                (float(row["candidate_mean"]) for row in candidate_rows),
                reverse=True,
            )[0]
            - sorted(
                (float(row["candidate_mean"]) for row in candidate_rows),
                reverse=True,
            )[1]
        ),
        "selected_action_evaluation_mean": candidate_rows[selected_position][
            "evaluation_mean"
        ],
        "selected_action_evaluation_standard_error": candidate_rows[
            selected_position
        ]["evaluation_standard_error"],
        "selected_action_paired_evaluation_delta_vs_baseline": copy.deepcopy(
            selected_pair
        ),
        "selected_action_paired_evaluation_deltas_vs_baseline": list(
            selected_raw
        ),
        "selected_action_paired_evaluation_deltas_sha256": hashlib.sha256(
            json.dumps(
                selected_raw,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "selected_action_paired_evaluation_override_loss": copy.deepcopy(
            selected_loss
        ),
        "selected_action_state_mean_loss_diagnostic": max(0.0, -mean),
        "audit_aggregation_contract": {
            "fire_definition": "selected_action_key_differs_from_explicit_baseline",
            "per_root_tail_source": "selected_action_paired_evaluation_override_loss",
            "aggregate_over_fires": {
                "p95": "maximum_of_per_root_e128_p95",
                "p99": "maximum_of_per_root_e128_p99",
                "max": "maximum_of_per_root_e128_max",
            },
            "quantile_method": "numpy_linear",
            "state_mean_loss_is_non_gate_diagnostic": True,
        },
        "evaluation_sample_best_action_key": selected,
        "candidate_belief_digest": candidate_batch.digest(),
        "evaluation_belief_digest": evaluation_batch.digest(),
        "candidate_rng_key_digests": [
            particle.rng_key_digest for particle in candidate_batch.particles
        ],
        "evaluation_rng_key_digests": [
            particle.rng_key_digest for particle in evaluation_batch.particles
        ],
        "sample_independence": "disjoint_particle_rng_keys",
        "root_selection_lock": "top8_fixed_before_sampling_then_c8_locked_before_e128",
        "search_config": {
            "learned_nonbaseline_top_k": 8,
            "baseline_added_exactly_once": True,
            "candidate_samples": 8,
            "evaluation_samples": 128,
            "evaluation_action_scope": (
                "locked_action_plus_explicit_baseline_only"
            ),
            "candidate_seed": ATTEMPT06_CANDIDATE_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            "evaluation_seed": ATTEMPT06_EVALUATION_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            "child_policy_seed": ATTEMPT06_CHILD_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            "run_id": root_run_id,
            "batch_child_selectors": True,
            "candidate_tie_break": "ActionKey",
            "search_tie_break": "ActionKey",
        },
        "continuation_policy": {
            "t2_policy_id": "stage9f_p2",
            "t2_resolution": "explicit_profile_never_current",
            "t2_runtime_status": "p2_fixed",
            "t3_selector": "m3_rust_evaluate_t3",
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "hypothetical_t4_selector": "m3_rust_evaluate_t4",
            "hypothetical_t4_candidate_samples": 1,
            "hypothetical_t4_evaluation_samples": 1,
            "real_live_t4_exact_unchanged": True,
        },
        "live_schedule": [
            {
                "seat": step.seat,
                "street": step.street,
                "draw_offset": step.draw_offset,
            }
            for step in T1_SECOND_LIVE_SCHEDULE
        ],
        "actions": candidate_rows,
        "child_information_set_count": 0,
        "teacher_value_status": "diagnostic_not_match_EV",
        "runtime_gate_allowed": False,
    }
    return teacher, baseline


def _row(
    root_index: int,
    *,
    fired: bool,
    mean: float,
    minimum: float = -20.0,
    p01: float = -15.0,
    p05: float = -10.0,
) -> dict[str, object]:
    teacher, baseline = _teacher(
        root_index,
        fired=fired,
        mean=mean,
        minimum=minimum,
        p01=p01,
        p05=p05,
    )
    run_name = "attempt06-synthetic"
    return {
        "schema": ATTEMPT06_SHARD_ROW_SCHEMA,
        "root_index": root_index,
        "hand_seed": ATTEMPT06_HAND_SEED_START
        + M43_ATTEMPT06_SEED_STRIDE * root_index,
        "root_profile": M43_ATTEMPT06_PROFILES[root_index % 5],
        "policy_observation": _observation().to_dict(),
        "baseline_action_key": baseline,
        "provenance": {
            "schema": ATTEMPT06_ROOT_PROVENANCE_SCHEMA,
            "root_index": root_index,
            "root_profile": M43_ATTEMPT06_PROFILES[root_index % 5],
            "root_policy_seed_base": ATTEMPT06_HAND_SEED_START,
            "root_generation_policy": ATTEMPT06_ROOT_GENERATION_POLICY,
            "baseline_profile": "stage18_p1",
            "plan_sha256": M43_ATTEMPT06_PLAN_SHA256,
            "candidate_model_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
            "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
            "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "current_profile_resolved": False,
            "opponent_private_discard_input_allowed": False,
            "teacher_value_status": "diagnostic_not_match_EV",
            "run_name": run_name,
            "run_id": f"{run_name}:shard={root_index}",
            "schedule_sha256": _sha("schedule"),
            "package_manifest_sha256": _sha("package"),
            "global_consumption_marker_sha256": _sha("global-marker"),
            "root_consumption_claim_sha256": _sha(f"root-claim-{root_index}"),
            "source_sha256": _sha("source"),
            "startup_sha256": _sha("startup"),
            "status_sha256": _sha("status"),
            "source_closure_sha256": _sha("source-closure"),
            "native_batch_threads": 4,
            "input_sha256": _sha(f"input-{root_index}"),
            "config_sha256": _sha(f"config-{root_index}"),
            "model_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
            "fresh_audit_retry_or_alternate_sample_allowed": False,
            "deterministic_claim_recovery_allowed": True,
            "deterministic_claim_recovery_mode": ROOT_REMATERIALIZATION_MODE,
        },
        "teacher": teacher,
    }


def _write_fixture(
    root: Path,
    *,
    tail_failure_root: int | None = None,
) -> dict[str, object]:
    marker = root / "consumption_marker.json"
    marker.write_text(
        json.dumps(
            {
                "schema": AUDIT_OUTPUT_CONSUMPTION_SCHEMA,
                "status": "consumed_before_any_result_teacher_or_root_read",
                "run_name": "attempt06-synthetic",
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fired = set(range(10))
    rows = []
    for root_index in range(50):
        kwargs: dict[str, float] = {}
        if root_index == tail_failure_root:
            kwargs = {"minimum": -45.0, "p01": -30.0, "p05": -26.0}
        rows.append(
            _row(
                root_index,
                fired=root_index in fired,
                mean=2.0 if root_index in fired else 0.0,
                **kwargs,
            )
        )
    merged = root / "merged50.jsonl"
    merged.write_bytes(
        b"".join(
            (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            for row in rows
        )
    )
    receipt = root / "merge_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": RECEIVE_MERGE_SCHEMA,
                "status": "merged_fifty_fresh_rows_without_fit_or_threshold_selection",
                "roots": 50,
                "shards": 50,
                "roots_per_shard": 1,
                "profile_counts": {profile: 10 for profile in M43_ATTEMPT06_PROFILES},
                "merged_teacher_sha256": _file_sha(merged),
                "consumption_marker_file_sha256": _file_sha(marker),
                "consumption_marker_run_name": "attempt06-synthetic",
                "shard_audit_sha256": [
                    _sha(f"received-audit-{index}") for index in range(50)
                ],
                "teacher_values_are_realized_match_ev": False,
                "fit_performed": False,
                "threshold_selected": False,
                "go_no_go_computed": False,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "marker": marker,
        "merged": merged,
        "receipt": receipt,
        "rows": rows,
    }


def _audit(paths: dict[str, object]) -> dict[str, object]:
    return audit_attempt06_search_quality(
        merged_path=paths["merged"],
        merge_receipt_path=paths["receipt"],
        consumption_marker_path=paths["marker"],
        plan_path=PLAN,
    )


def _rewrite_merged_and_receipt(paths: dict[str, object]) -> None:
    rows = paths["rows"]
    assert isinstance(rows, list)
    merged = paths["merged"]
    receipt = paths["receipt"]
    assert isinstance(merged, Path)
    assert isinstance(receipt, Path)
    merged.write_bytes(
        b"".join(
            (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            for row in rows
        )
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["merged_teacher_sha256"] = _file_sha(merged)
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_go_uses_metric_b_and_only_authorizes_separate_freeze(tmp_path: Path) -> None:
    report = _audit(_write_fixture(tmp_path))

    assert report["schema"] == ATTEMPT06_SEARCH_QUALITY_AUDIT_SCHEMA
    assert report["status"] == "go_create_separate_freeze_only"
    assert report["decision"] == "go"
    assert report["all_gates_passed"] is True
    assert report["metrics"]["overall"]["fires"] == 10
    assert report["metrics"]["overall"]["mean_delta_per_state"] == pytest.approx(
        0.4
    )
    assert report["metrics"]["overall"][
        "metric_b_maximum_per_fired_root_loss"
    ] == {"p95": 10.0, "p99": 15.0, "max": 20.0}
    assert report["metrics"]["paired_mean_cross_fire_loss"] == {
        "classification": "diagnostic_only_not_gate",
        "source": "max(0,-paired_e128_mean)_per_fired_root",
        "quantile_method": "numpy_linear",
        "p95": 0.0,
        "p99": 0.0,
        "max": 0.0,
    }
    assert report["science_boundary"][
        "fresh_200_root_fit_or_distillation_authorized"
    ] is False
    assert report["science_boundary"]["fit_performed"] is False
    assert report["science_boundary"]["threshold_selected"] is False
    assert report["science_boundary"]["current_profile_mutated"] is False


def test_metric_b_tail_can_fail_while_paired_mean_diagnostic_is_zero(
    tmp_path: Path,
) -> None:
    report = _audit(_write_fixture(tmp_path, tail_failure_root=0))

    assert report["decision"] == "no_go"
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert gates["metric_b_p95_loss"]["observed"] == 26.0
    assert gates["metric_b_p95_loss"]["passed"] is False
    assert report["metrics"]["paired_mean_cross_fire_loss"]["p99"] == 0.0


def test_marker_is_validated_before_any_missing_result_path(tmp_path: Path) -> None:
    marker = tmp_path / "bad-marker.json"
    marker.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="consumption marker changed"):
        audit_attempt06_search_quality(
            merged_path=tmp_path / "does-not-exist-merged.jsonl",
            merge_receipt_path=tmp_path / "does-not-exist-receipt.json",
            consumption_marker_path=marker,
            plan_path=tmp_path / "does-not-exist-plan.json",
        )


def test_strict_receive_validator_accepts_v2_and_rejects_unlocked_e128() -> None:
    row = _row(0, fired=True, mean=2.0)
    provenance = row["provenance"]
    validate_attempt06_teacher_row(
        row,
        root_index=0,
        expected_run_name="attempt06-synthetic",
        expected_schedule_sha256=provenance["schedule_sha256"],
        expected_package_manifest_sha256=provenance[
            "package_manifest_sha256"
        ],
        expected_global_consumption_marker_sha256=provenance[
            "global_consumption_marker_sha256"
        ],
        expected_root_consumption_claim_sha256=provenance[
            "root_consumption_claim_sha256"
        ],
        expected_input_sha256=provenance["input_sha256"],
        expected_source_sha256=provenance["source_sha256"],
        expected_startup_sha256=provenance["startup_sha256"],
        expected_status_sha256=provenance["status_sha256"],
        expected_source_closure_sha256=provenance["source_closure_sha256"],
    )

    row["teacher"]["actions"][1]["evaluation_mean"] = 999.0
    with pytest.raises(ValueError, match="strict integrity"):
        validate_attempt06_teacher_row(row, root_index=0)


@pytest.mark.parametrize(
    "mutation",
    (
        "run_name",
        "candidate_seed",
        "evaluation_seed",
        "child_policy_seed",
        "evaluation_action_scope",
        "batch_child_selectors",
        "rng_self_report",
        "live_schedule",
        "unknown_teacher_field",
        "unknown_provenance_field",
        "search_config_cards",
        "frozen_model_id",
        "loss_extra_field",
        "selection_gap_card",
        "child_count_card",
        "raw_missing",
        "raw_modified",
        "raw_summary_consistent_modified",
        "t2_policy",
        "missing_legal_action_keys",
        "duplicate_top8",
        "selected_by_c8",
        "paired_count",
        "paired_loss_formula",
    ),
)
def test_strict_receive_validator_rejects_adversarial_teacher_mutations(
    mutation: str,
) -> None:
    row = _row(0, fired=True, mean=2.0)
    teacher = row["teacher"]
    assert isinstance(teacher, dict)
    kwargs: dict[str, object] = {}
    if mutation == "run_name":
        kwargs["expected_run_name"] = "attempt06-different-run"
    elif mutation in {"candidate_seed", "evaluation_seed", "child_policy_seed"}:
        teacher["search_config"][mutation] = 999
    elif mutation == "evaluation_action_scope":
        teacher["search_config"][mutation] = "WRONG"
    elif mutation == "batch_child_selectors":
        teacher["search_config"][mutation] = False
    elif mutation == "rng_self_report":
        teacher["candidate_rng_key_digests"] = [
            _sha(f"forged-candidate-{index}") for index in range(8)
        ]
        teacher["evaluation_rng_key_digests"] = [
            _sha(f"forged-evaluation-{index}") for index in range(128)
        ]
        teacher["candidate_belief_digest"] = _sha("forged-candidate-belief")
        teacher["evaluation_belief_digest"] = _sha("forged-evaluation-belief")
    elif mutation == "live_schedule":
        teacher["live_schedule"] = []
    elif mutation == "unknown_teacher_field":
        teacher["sneaky"] = {"cards": ["As"]}
    elif mutation == "unknown_provenance_field":
        row["provenance"]["sneaky_hidden_hash"] = "3" * 64
    elif mutation == "search_config_cards":
        teacher["search_config"]["cards"] = ["As"]
    elif mutation == "frozen_model_id":
        teacher["frozen_candidate_generator"]["model_id"] = "As"
    elif mutation == "loss_extra_field":
        teacher["selected_action_paired_evaluation_override_loss"][
            "cards"
        ] = ["As"]
    elif mutation == "selection_gap_card":
        teacher["selection_score_gap"] = "As"
    elif mutation == "child_count_card":
        teacher["child_information_set_count"] = "As"
    elif mutation == "raw_missing":
        del teacher["selected_action_paired_evaluation_deltas_vs_baseline"]
    elif mutation == "raw_modified":
        teacher["selected_action_paired_evaluation_deltas_vs_baseline"][0] -= 1.0
    elif mutation == "raw_summary_consistent_modified":
        raw = list(
            teacher["selected_action_paired_evaluation_deltas_vs_baseline"]
        )
        raw[0] -= 1.0
        summary = _pair(raw)
        loss = _loss(summary)
        teacher["selected_action_paired_evaluation_deltas_vs_baseline"] = raw
        teacher["selected_action_paired_evaluation_delta_vs_baseline"] = copy.deepcopy(
            summary
        )
        teacher["selected_action_paired_evaluation_override_loss"] = copy.deepcopy(
            loss
        )
        selected_position = teacher["selected_action_candidate_position"]
        teacher["actions"][selected_position][
            "paired_evaluation_delta_vs_baseline"
        ] = copy.deepcopy(summary)
        teacher["actions"][selected_position][
            "paired_evaluation_override_loss"
        ] = copy.deepcopy(loss)
        diagnostic = max(0.0, -float(summary["mean"]))
        teacher["selected_action_state_mean_loss_diagnostic"] = diagnostic
        teacher["actions"][selected_position][
            "paired_evaluation_state_mean_loss_diagnostic"
        ] = diagnostic
    elif mutation == "t2_policy":
        teacher["continuation_policy"]["t2_policy_id"] = "WRONG"
    elif mutation == "missing_legal_action_keys":
        del teacher["legal_action_keys"]
    elif mutation == "duplicate_top8":
        teacher["learned_top8_action_keys"][1] = teacher[
            "learned_top8_action_keys"
        ][0]
    elif mutation == "selected_by_c8":
        teacher["actions"][1]["selected_by_c8"] = True
    elif mutation == "paired_count":
        teacher["selected_action_paired_evaluation_delta_vs_baseline"][
            "count"
        ] = 127
    elif mutation == "paired_loss_formula":
        teacher["selected_action_paired_evaluation_override_loss"]["p95"] += 1.0
    else:  # pragma: no cover - parametrization owns this domain.
        raise AssertionError(mutation)
    with pytest.raises(ValueError):
        validate_attempt06_teacher_row(row, root_index=0, **kwargs)


def test_nonfire_must_have_complete_counterfactual_cancellation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    rows = paths["rows"]
    assert isinstance(rows, list)
    teacher = rows[11]["teacher"]
    for pair in (
        teacher["selected_action_paired_evaluation_delta_vs_baseline"],
        teacher["actions"][8]["paired_evaluation_delta_vs_baseline"],
    ):
        for name in (
            "mean",
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
        ):
            pair[name] = 0.01
    teacher["selected_action_state_mean_loss_diagnostic"] = 0.0
    teacher["actions"][8]["paired_evaluation_state_mean_loss_diagnostic"] = 0.0
    _rewrite_merged_and_receipt(paths)

    with pytest.raises(
        ValueError,
        match="counterfactual cancellation|paired raw/summary|selected action/e128",
    ):
        _audit(paths)


def test_mapping_rng_and_hidden_violations_are_no_go_gates(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rows = paths["rows"]
    assert isinstance(rows, list)
    rows[0]["teacher"]["legal_action_order_digest"] = "0" * 64
    rows[1]["teacher"]["evaluation_rng_key_digests"][0] = rows[1]["teacher"][
        "candidate_rng_key_digests"
    ][0]
    rows[2]["teacher"]["frozen_candidate_generator"]["model_id"] = "As"
    _rewrite_merged_and_receipt(paths)

    report = _audit(paths)
    assert report["decision"] == "no_go"
    assert report["integrity"]["action_mapping_violation_count"] == 1
    assert report["integrity"]["rng_domain_violation_count"] >= 1
    assert report["integrity"]["hidden_information_violation_count"] == 1
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert gates["action_mapping_violation_count"]["passed"] is False
    assert gates["rng_domain_violation_count"]["passed"] is False
    assert gates["hidden_information_violation_count"]["passed"] is False


def test_receipt_hash_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    receipt = paths["receipt"]
    assert isinstance(receipt, Path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["merged_teacher_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from receipt"):
        _audit(paths)


def test_report_writer_is_one_shot_no_clobber(tmp_path: Path) -> None:
    report = _audit(_write_fixture(tmp_path))
    output = tmp_path / "search_quality_report.json"

    write_attempt06_search_quality_report(output, report)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_attempt06_search_quality_report(output, {"tampered": True})
    assert output.read_bytes() == original
