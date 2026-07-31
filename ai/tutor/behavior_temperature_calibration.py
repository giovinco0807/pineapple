"""Content-bound scalar-temperature calibration for M3 behavior likelihoods.

This module is deliberately separate from the frozen behavior adapters.  It
records *pre-temperature* legal-action logits directly from a checkpoint,
binds those logits to a verified behavior-decision record, and evaluates a
preregistered fit/dev/locked-test protocol.  It never reconstructs logits from
Q32 probabilities.

The protocol is intentionally fail closed:

* T1-BB, T1-BTN, T2-BB, and T2-BTN have independent temperatures.
* fit creates one scalar-temperature candidate in ``[0.05, 20]``;
* dev selects that candidate or the identity temperature without seeing test;
* either frozen dev-selected candidate is promotion-valid by default; requiring
  a non-identity fit temperature is an explicit opt-in policy only;
* locked test is consumed by one metrics pass after selection is frozen;
* all published metrics and paired-root bootstrap intervals are regenerated
  from raw decision records and content-bound model-evaluation rows;
* promotion is true only when every configured count and quality check passes.

JSON artifacts contain no binary JSON numbers.  Model logits and derived
floating-point metrics use canonical ``float.hex`` strings, while thresholds
use canonical rational strings.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ai.tutor.behavior_calibration_contract import (
    SPLIT_NAMES,
    canonical_sha256,
    canonical_snapshot,
    verify_behavior_decision_dataset,
    verify_behavior_decision_log,
)
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet


MODEL_EVALUATION_SCHEMA = "ofc_behavior_model_evaluation/v1"
CALIBRATION_SCHEMA = "ofc_behavior_temperature_calibration/v2"
GATE_CONFIG_SCHEMA = "ofc_behavior_temperature_gate_config/v2"
GATE_RESULT_SCHEMA = "ofc_behavior_temperature_gate_result/v2"

ALLOW_DEV_SELECTED_CANDIDATE = "allow_dev_selected_fit_or_identity"
REQUIRE_NONIDENTITY_FIT = "require_dev_selected_nonidentity_fit"
DEV_SELECTION_POLICIES = frozenset(
    {ALLOW_DEV_SELECTED_CANDIDATE, REQUIRE_NONIDENTITY_FIT}
)

ROLE_KEYS = ("t1_bb", "t1_btn", "t2_bb", "t2_btn")
JOKER_KEYS = ("joker_0", "joker_1", "joker_2")
TEMPERATURE_DENOMINATOR = 1_000_000
TEMPERATURE_MIN_NUMERATOR = 50_000
TEMPERATURE_MAX_NUMERATOR = 20_000_000
IDENTITY_TEMPERATURE_NUMERATOR = TEMPERATURE_DENOMINATOR

_SHA256_LENGTH = 64
_EVALUATION_KEYS = frozenset(
    {
        "schema",
        "promotion_eligible",
        "logit_contract",
        "record_sha256",
        "decision_id",
        "root_id",
        "split",
        "turn",
        "actor",
        "visible_joker_count",
        "legal_action_ids",
        "observed_action_key",
        "logits_f64_hex",
        "checkpoint_sha256",
        "model_sha256",
        "row_extractor_sha256",
        "adapter_source_sha256",
        "evaluation_row_sha256",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "preregistered_before_locked_test",
        "minimum_counts",
        "quality_thresholds",
        "bootstrap",
        "metric_contract",
        "selection_policy",
        "challenge_contract",
        "gate_config_sha256",
    }
)


@runtime_checkable
class PreTemperatureLegalLogitEvaluator(Protocol):
    """Strict checkpoint evaluator used to construct one evaluation row.

    Implementations must return the selected legal logits from the frozen
    checkpoint before temperature scaling or probability quantization.  The
    four hashes bind checkpoint bytes, model/architecture identity, the row
    extractor implementation, and the adapter implementation respectively.
    """

    checkpoint_sha256: str
    model_sha256: str
    row_extractor_sha256: str
    adapter_source_sha256: str

    def pre_temperature_legal_logits(
        self,
        information: BehaviorInfoSet,
        legal_action_ids: tuple[str, ...],
    ) -> Sequence[float]:
        """Return one finite float64-compatible logit per legal action."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _role_key(turn: int, actor: str) -> str:
    key = f"t{turn}_{actor}"
    if key not in ROLE_KEYS:
        raise ValueError(f"unsupported calibration role: {key}")
    return key


def _joker_key(count: int) -> str:
    key = f"joker_{count}"
    if key not in JOKER_KEYS:
        raise ValueError(f"unsupported visible Joker count: {count}")
    return key


def _canonical_float_hex(value: float, *, label: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite float64")
    if number == 0.0:
        number = 0.0
    return number.hex()


def _parse_float_hex(value: Any, *, label: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a canonical float64 hex string")
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a float64 hex string") from exc
    if not math.isfinite(number) or _canonical_float_hex(number, label=label) != value:
        raise ValueError(f"{label} is not a canonical finite float64 hex string")
    return number


def _fraction_text(value: Fraction | int | str, *, label: str) -> str:
    if isinstance(value, (bool, float)):
        raise TypeError(f"{label} must be an exact Fraction/int/str")
    try:
        result = value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} is not an exact rational") from exc
    return f"{result.numerator}/{result.denominator}"


def _parse_fraction(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a canonical rational string")
    result = Fraction(value)
    if value != f"{result.numerator}/{result.denominator}":
        raise ValueError(f"{label} is not a canonical rational string")
    return result


def _temperature_payload(numerator: int) -> dict[str, int]:
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError("temperature numerator must be an integer")
    if not TEMPERATURE_MIN_NUMERATOR <= numerator <= TEMPERATURE_MAX_NUMERATOR:
        raise ValueError("temperature must be in [0.05,20]")
    return {"numerator": numerator, "denominator": TEMPERATURE_DENOMINATOR}


def _temperature_value(payload: Mapping[str, Any], *, label: str) -> float:
    raw = _require_mapping(payload, label=label)
    _require_exact_keys(
        raw, frozenset({"numerator", "denominator"}), label=label
    )
    numerator = raw["numerator"]
    denominator = raw["denominator"]
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError(f"{label}.numerator must be an integer")
    if denominator != TEMPERATURE_DENOMINATOR:
        raise ValueError(f"{label} denominator must be 1000000")
    _temperature_payload(numerator)
    return numerator / TEMPERATURE_DENOMINATOR


def _unsigned_hash(value: Mapping[str, Any], hash_key: str, *, label: str) -> str:
    recorded = _require_sha256(value.get(hash_key), label=hash_key)
    unsigned = dict(value)
    unsigned.pop(hash_key, None)
    if canonical_sha256(unsigned) != recorded:
        raise ValueError(f"{label} SHA-256 mismatch")
    return recorded


def build_model_evaluation_row(
    record: Mapping[str, Any],
    evaluator: PreTemperatureLegalLogitEvaluator,
) -> dict[str, Any]:
    """Evaluate and bind pre-temperature checkpoint logits for one raw record."""
    verified = verify_behavior_decision_log(record)
    for attribute in (
        "checkpoint_sha256",
        "model_sha256",
        "row_extractor_sha256",
        "adapter_source_sha256",
    ):
        _require_sha256(getattr(evaluator, attribute, None), label=attribute)
    method = getattr(evaluator, "pre_temperature_legal_logits", None)
    if not callable(method):
        raise TypeError("evaluator must implement pre_temperature_legal_logits")

    legal_action_ids = tuple(record["legal_action_ids"])
    logits_raw = method(verified.information, legal_action_ids)
    if isinstance(logits_raw, (str, bytes)) or not isinstance(logits_raw, Sequence):
        raise TypeError("pre-temperature logits must be a sequence")
    if len(logits_raw) != len(legal_action_ids):
        raise ValueError("pre-temperature logits must match legal-action count")
    logits_hex = [
        _canonical_float_hex(value, label=f"logits[{index}]")
        for index, value in enumerate(logits_raw)
    ]

    row: dict[str, Any] = {
        "schema": MODEL_EVALUATION_SCHEMA,
        "promotion_eligible": False,
        "logit_contract": (
            "frozen_checkpoint_pre_temperature_selected_legal_logits_float64"
        ),
        "record_sha256": verified.record_sha256,
        "decision_id": verified.decision_id,
        "root_id": verified.root_id,
        "split": verified.split,
        "turn": verified.turn,
        "actor": verified.actor,
        "visible_joker_count": verified.visible_joker_count,
        "legal_action_ids": list(legal_action_ids),
        "observed_action_key": record["observed_action_key"],
        "logits_f64_hex": logits_hex,
        "checkpoint_sha256": evaluator.checkpoint_sha256,
        "model_sha256": evaluator.model_sha256,
        "row_extractor_sha256": evaluator.row_extractor_sha256,
        "adapter_source_sha256": evaluator.adapter_source_sha256,
    }
    row["evaluation_row_sha256"] = canonical_sha256(row)
    verify_model_evaluation_row(row, record)
    return row


@dataclass(frozen=True)
class VerifiedModelEvaluation:
    evaluation_row_sha256: str
    record_sha256: str
    decision_id: str
    root_id: str
    split: str
    role: str
    visible_joker_count: int
    legal_action_ids: tuple[str, ...]
    observed_index: int
    logits: tuple[float, ...]
    checkpoint_sha256: str
    model_sha256: str
    row_extractor_sha256: str
    adapter_source_sha256: str


def verify_model_evaluation_row(
    row: Mapping[str, Any], record: Mapping[str, Any]
) -> VerifiedModelEvaluation:
    """Verify row integrity and bind every duplicated field to a raw record."""
    raw = _require_mapping(row, label="model evaluation row")
    _require_exact_keys(raw, _EVALUATION_KEYS, label="model evaluation row")
    if raw["schema"] != MODEL_EVALUATION_SCHEMA:
        raise ValueError("unsupported model evaluation schema")
    if raw["promotion_eligible"] is not False:
        raise ValueError("individual model evaluation rows are never promotion eligible")
    if raw["logit_contract"] != (
        "frozen_checkpoint_pre_temperature_selected_legal_logits_float64"
    ):
        raise ValueError("model evaluation logit contract mismatch")
    row_sha = _unsigned_hash(
        raw,
        "evaluation_row_sha256",
        label="model evaluation row",
    )

    verified = verify_behavior_decision_log(record)
    duplicated = {
        "record_sha256": verified.record_sha256,
        "decision_id": verified.decision_id,
        "root_id": verified.root_id,
        "split": verified.split,
        "turn": verified.turn,
        "actor": verified.actor,
        "visible_joker_count": verified.visible_joker_count,
        "legal_action_ids": list(record["legal_action_ids"]),
        "observed_action_key": record["observed_action_key"],
    }
    for key, expected in duplicated.items():
        if raw[key] != expected:
            raise ValueError(f"model evaluation {key} does not match raw record")

    for hash_key in (
        "checkpoint_sha256",
        "model_sha256",
        "row_extractor_sha256",
        "adapter_source_sha256",
    ):
        _require_sha256(raw[hash_key], label=hash_key)
    logits_raw = raw["logits_f64_hex"]
    if not isinstance(logits_raw, list) or len(logits_raw) != len(
        record["legal_action_ids"]
    ):
        raise ValueError("model evaluation logits must match legal-action count")
    logits = tuple(
        _parse_float_hex(value, label=f"logits_f64_hex[{index}]")
        for index, value in enumerate(logits_raw)
    )
    try:
        observed_index = record["legal_action_ids"].index(
            record["observed_action_key"]
        )
    except ValueError as exc:  # raw verifier should already make this impossible
        raise AssertionError("verified observed action is absent from legal IDs") from exc

    return VerifiedModelEvaluation(
        evaluation_row_sha256=row_sha,
        record_sha256=verified.record_sha256,
        decision_id=verified.decision_id,
        root_id=verified.root_id,
        split=verified.split,
        role=_role_key(verified.turn, verified.actor),
        visible_joker_count=verified.visible_joker_count,
        legal_action_ids=tuple(record["legal_action_ids"]),
        observed_index=observed_index,
        logits=logits,
        checkpoint_sha256=raw["checkpoint_sha256"],
        model_sha256=raw["model_sha256"],
        row_extractor_sha256=raw["row_extractor_sha256"],
        adapter_source_sha256=raw["adapter_source_sha256"],
    )


def build_temperature_gate_config(
    *,
    gate_id: str = "m3_behavior_temperature_locked_test_v2",
    min_fit_decisions_per_role: int = 50_000,
    min_dev_decisions_per_role: int = 10_000,
    min_test_decisions_per_role: int = 20_000,
    min_challenge_decisions_per_role_joker: int = 2_000,
    min_roots_per_split_role: int = 2,
    min_challenge_roots_per_role_joker: int = 2,
    require_joker_challenge: bool = True,
    challenge_root_namespace_prefix: str = "m3-joker-challenge-v1/",
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: str = "m3-behavior-temperature-bootstrap-v1",
    ece_bins: int = 15,
    max_test_nll_delta_vs_t1: Fraction | int | str = Fraction(0),
    max_test_nll_delta_vs_t1_ucb: Fraction | int | str = Fraction(5, 1000),
    min_test_uniform_improvement_role_lcb: Fraction | int | str = Fraction(2, 100),
    min_test_uniform_improvement_joker_lcb: Fraction | int | str = Fraction(1, 100),
    max_test_role_marginal_ece: Fraction | int | str = Fraction(3, 100),
    max_test_joker_marginal_ece: Fraction | int | str = Fraction(5, 100),
    max_test_brier_delta_vs_t1: Fraction | int | str = Fraction(2, 1000),
    min_test_residual_temperature: Fraction | int | str = Fraction(4, 5),
    max_test_residual_temperature: Fraction | int | str = Fraction(5, 4),
    require_nonidentity_temperature: bool = False,
) -> dict[str, Any]:
    """Build the preregistered, self-hashed locked-test gate configuration."""
    if not isinstance(gate_id, str) or not gate_id:
        raise ValueError("gate_id must be a non-empty string")
    integer_values = {
        "min_fit_decisions_per_role": min_fit_decisions_per_role,
        "min_dev_decisions_per_role": min_dev_decisions_per_role,
        "min_test_decisions_per_role": min_test_decisions_per_role,
        "min_challenge_decisions_per_role_joker": min_challenge_decisions_per_role_joker,
        "min_roots_per_split_role": min_roots_per_split_role,
        "min_challenge_roots_per_role_joker": min_challenge_roots_per_role_joker,
        "bootstrap_replicates": bootstrap_replicates,
        "ece_bins": ece_bins,
    }
    for label, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if bootstrap_replicates < 20:
        raise ValueError("bootstrap_replicates must be at least 20")
    if not isinstance(bootstrap_seed, str) or not bootstrap_seed:
        raise ValueError("bootstrap_seed must be a non-empty string")
    if type(require_joker_challenge) is not bool:
        raise TypeError("require_joker_challenge must be bool")
    if type(require_nonidentity_temperature) is not bool:
        raise TypeError("require_nonidentity_temperature must be bool")
    if (
        not isinstance(challenge_root_namespace_prefix, str)
        or not challenge_root_namespace_prefix
    ):
        raise ValueError("challenge root namespace prefix must be non-empty")

    config: dict[str, Any] = {
        "schema": GATE_CONFIG_SCHEMA,
        "gate_id": gate_id,
        "preregistered_before_locked_test": True,
        "minimum_counts": {
            "decisions_per_role_by_split": {
                "fit": min_fit_decisions_per_role,
                "dev": min_dev_decisions_per_role,
                "test": min_test_decisions_per_role,
            },
            "challenge_decisions_per_role_joker": (
                min_challenge_decisions_per_role_joker
            ),
            "roots_per_split_role": min_roots_per_split_role,
            "challenge_roots_per_role_joker": min_challenge_roots_per_role_joker,
        },
        "quality_thresholds": {
            "max_test_nll_delta_vs_t1": _fraction_text(
                max_test_nll_delta_vs_t1, label="max_test_nll_delta_vs_t1"
            ),
            "max_test_nll_delta_vs_t1_ucb": _fraction_text(
                max_test_nll_delta_vs_t1_ucb,
                label="max_test_nll_delta_vs_t1_ucb",
            ),
            "min_test_uniform_improvement_role_lcb": _fraction_text(
                min_test_uniform_improvement_role_lcb,
                label="min_test_uniform_improvement_role_lcb",
            ),
            "min_test_uniform_improvement_joker_lcb": _fraction_text(
                min_test_uniform_improvement_joker_lcb,
                label="min_test_uniform_improvement_joker_lcb",
            ),
            "max_test_role_marginal_ece": _fraction_text(
                max_test_role_marginal_ece,
                label="max_test_role_marginal_ece",
            ),
            "max_test_joker_marginal_ece": _fraction_text(
                max_test_joker_marginal_ece,
                label="max_test_joker_marginal_ece",
            ),
            "max_test_brier_delta_vs_t1": _fraction_text(
                max_test_brier_delta_vs_t1,
                label="max_test_brier_delta_vs_t1",
            ),
            "min_test_residual_temperature": _fraction_text(
                min_test_residual_temperature,
                label="min_test_residual_temperature",
            ),
            "max_test_residual_temperature": _fraction_text(
                max_test_residual_temperature,
                label="max_test_residual_temperature",
            ),
        },
        "bootstrap": {
            "algorithm": "sha256_cluster_root_percentile_v1",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence": "95/100",
            "quantiles": ["1/40", "39/40"],
        },
        "metric_contract": {
            "nll": "stable_logsumexp_observed_action_mean_nats",
            "temperature_bounds": ["1/20", "20/1"],
            "temperature_quantization_denominator": TEMPERATURE_DENOMINATOR,
            "dev_selection": "lower_nll_fit_candidate_vs_identity_tie_identity",
            "dev_candidate_set": ["fit_temperature", "identity_temperature"],
            "selection_frozen_before_locked_test": True,
            "locked_test_selection_input": False,
            "locked_test_evaluation_passes": 1,
            "locked_test_temperature_source": "frozen_dev_selected_candidate",
            "metric_temperature_binding": "applied_temperature_exact_payload",
            "brier": "multiclass_sum_squared_error_per_decision",
            "marginal_ece": "pooled_legal_class_instances_equal_width",
            "ece_bins": ece_bins,
            "bootstrap_unit": "root_id",
            "float_storage": "canonical_float64_hex",
        },
        "selection_policy": (
            REQUIRE_NONIDENTITY_FIT
            if require_nonidentity_temperature
            else ALLOW_DEV_SELECTED_CANDIDATE
        ),
        "challenge_contract": {
            "required_for_promotion": require_joker_challenge,
            "root_namespace_prefix": challenge_root_namespace_prefix,
            "separate_from_main_root_set": True,
            "used_for_temperature_fit": False,
            "used_for_dev_selection": False,
            "used_for_locked_test_role_metrics": False,
            "purpose": "targeted_role_by_visible_joker_diagnostics_only",
        },
    }
    config["gate_config_sha256"] = canonical_sha256(config)
    verify_temperature_gate_config(config)
    return config


def verify_temperature_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_mapping(config, label="temperature gate config")
    _require_exact_keys(raw, _CONFIG_KEYS, label="temperature gate config")
    if raw["schema"] != GATE_CONFIG_SCHEMA:
        raise ValueError("unsupported temperature gate config schema")
    if not isinstance(raw["gate_id"], str) or not raw["gate_id"]:
        raise ValueError("temperature gate_id must be non-empty")
    if raw["preregistered_before_locked_test"] is not True:
        raise ValueError("temperature gate must be preregistered before locked test")
    _unsigned_hash(raw, "gate_config_sha256", label="temperature gate config")

    minimum = _require_mapping(raw["minimum_counts"], label="minimum_counts")
    _require_exact_keys(
        minimum,
        frozenset(
            {
                "decisions_per_role_by_split",
                "challenge_decisions_per_role_joker",
                "roots_per_split_role",
                "challenge_roots_per_role_joker",
            }
        ),
        label="minimum_counts",
    )
    per_split = _require_mapping(
        minimum["decisions_per_role_by_split"],
        label="decisions_per_role_by_split",
    )
    _require_exact_keys(per_split, frozenset(SPLIT_NAMES), label="split minimums")
    for key, value in list(per_split.items()) + [
        (
            "challenge_decisions_per_role_joker",
            minimum["challenge_decisions_per_role_joker"],
        ),
        ("roots_per_split_role", minimum["roots_per_split_role"]),
        (
            "challenge_roots_per_role_joker",
            minimum["challenge_roots_per_role_joker"],
        ),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"minimum count {key} must be a positive integer")

    thresholds = _require_mapping(
        raw["quality_thresholds"], label="quality_thresholds"
    )
    threshold_keys = frozenset(
        {
            "max_test_nll_delta_vs_t1",
            "max_test_nll_delta_vs_t1_ucb",
            "min_test_uniform_improvement_role_lcb",
            "min_test_uniform_improvement_joker_lcb",
            "max_test_role_marginal_ece",
            "max_test_joker_marginal_ece",
            "max_test_brier_delta_vs_t1",
            "min_test_residual_temperature",
            "max_test_residual_temperature",
        }
    )
    _require_exact_keys(thresholds, threshold_keys, label="quality_thresholds")
    parsed_thresholds = {
        key: _parse_fraction(value, label=key)
        for key, value in thresholds.items()
    }
    if parsed_thresholds["min_test_residual_temperature"] <= 0:
        raise ValueError("minimum residual temperature must be positive")
    if (
        parsed_thresholds["min_test_residual_temperature"]
        > parsed_thresholds["max_test_residual_temperature"]
    ):
        raise ValueError("residual temperature bounds are inverted")

    bootstrap = _require_mapping(raw["bootstrap"], label="bootstrap")
    if bootstrap != {
        "algorithm": "sha256_cluster_root_percentile_v1",
        "replicates": bootstrap.get("replicates"),
        "seed": bootstrap.get("seed"),
        "confidence": "95/100",
        "quantiles": ["1/40", "39/40"],
    }:
        raise ValueError("bootstrap contract mismatch")
    if (
        isinstance(bootstrap["replicates"], bool)
        or not isinstance(bootstrap["replicates"], int)
        or bootstrap["replicates"] < 20
    ):
        raise ValueError("bootstrap replicates must be at least 20")
    if not isinstance(bootstrap["seed"], str) or not bootstrap["seed"]:
        raise ValueError("bootstrap seed must be non-empty")

    metric = _require_mapping(raw["metric_contract"], label="metric_contract")
    expected_metric = {
        "nll": "stable_logsumexp_observed_action_mean_nats",
        "temperature_bounds": ["1/20", "20/1"],
        "temperature_quantization_denominator": TEMPERATURE_DENOMINATOR,
        "dev_selection": "lower_nll_fit_candidate_vs_identity_tie_identity",
        "dev_candidate_set": ["fit_temperature", "identity_temperature"],
        "selection_frozen_before_locked_test": True,
        "locked_test_selection_input": False,
        "locked_test_evaluation_passes": 1,
        "locked_test_temperature_source": "frozen_dev_selected_candidate",
        "metric_temperature_binding": "applied_temperature_exact_payload",
        "brier": "multiclass_sum_squared_error_per_decision",
        "marginal_ece": "pooled_legal_class_instances_equal_width",
        "ece_bins": metric.get("ece_bins"),
        "bootstrap_unit": "root_id",
        "float_storage": "canonical_float64_hex",
    }
    if metric != expected_metric:
        raise ValueError("metric contract mismatch")
    if (
        isinstance(metric["ece_bins"], bool)
        or not isinstance(metric["ece_bins"], int)
        or metric["ece_bins"] <= 0
    ):
        raise ValueError("ECE bins must be a positive integer")
    if (
        not isinstance(raw["selection_policy"], str)
        or raw["selection_policy"] not in DEV_SELECTION_POLICIES
    ):
        raise ValueError("unsupported temperature dev-selection promotion policy")
    challenge = _require_mapping(raw["challenge_contract"], label="challenge_contract")
    expected_challenge = {
        "required_for_promotion": challenge.get("required_for_promotion"),
        "root_namespace_prefix": challenge.get("root_namespace_prefix"),
        "separate_from_main_root_set": True,
        "used_for_temperature_fit": False,
        "used_for_dev_selection": False,
        "used_for_locked_test_role_metrics": False,
        "purpose": "targeted_role_by_visible_joker_diagnostics_only",
    }
    if challenge != expected_challenge:
        raise ValueError("Joker challenge contract mismatch")
    if type(challenge["required_for_promotion"]) is not bool:
        raise TypeError("challenge required flag must be bool")
    if (
        not isinstance(challenge["root_namespace_prefix"], str)
        or not challenge["root_namespace_prefix"]
    ):
        raise ValueError("challenge root namespace prefix must be non-empty")
    return canonical_snapshot(raw)


@dataclass(frozen=True)
class _BoundRow:
    evaluation: VerifiedModelEvaluation

    @property
    def root_id(self) -> str:
        return self.evaluation.root_id

    @property
    def split(self) -> str:
        return self.evaluation.split

    @property
    def role(self) -> str:
        return self.evaluation.role

    @property
    def joker_key(self) -> str:
        return _joker_key(self.evaluation.visible_joker_count)


def _stable_probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def _row_values(row: _BoundRow, temperature: float) -> tuple[float, float, float, float]:
    logits = row.evaluation.logits
    observed = row.evaluation.observed_index
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    logsumexp = maximum + math.log(math.fsum(math.exp(x - maximum) for x in scaled))
    nll = logsumexp - scaled[observed]
    probabilities = _stable_probabilities(logits, temperature)
    brier = math.fsum(
        (probability - (1.0 if index == observed else 0.0)) ** 2
        for index, probability in enumerate(probabilities)
    )
    uniform_nll = math.log(len(logits))
    uniform_brier = 1.0 - 1.0 / len(logits)
    return nll, brier, uniform_nll, uniform_brier


def _mean_nll(rows: Sequence[_BoundRow], temperature: float) -> float | None:
    if not rows:
        return None
    return math.fsum(_row_values(row, temperature)[0] for row in rows) / len(rows)


def _temperature_gradient(rows: Sequence[_BoundRow], beta: float) -> float:
    contributions: list[float] = []
    for row in rows:
        logits = row.evaluation.logits
        scaled = [beta * value for value in logits]
        maximum = max(scaled)
        weights = [math.exp(value - maximum) for value in scaled]
        denominator = math.fsum(weights)
        expected_logit = math.fsum(
            weight * logit for weight, logit in zip(weights, logits)
        ) / denominator
        contributions.append(expected_logit - logits[row.evaluation.observed_index])
    return math.fsum(contributions) / len(contributions)


def _fit_temperature(rows: Sequence[_BoundRow]) -> tuple[int, str]:
    """Minimize observed-action NLL in inverse-temperature space."""
    if not rows:
        return IDENTITY_TEMPERATURE_NUMERATOR, "insufficient_rows"
    beta_low = 0.05
    beta_high = 20.0
    gradient_low = _temperature_gradient(rows, beta_low)
    gradient_high = _temperature_gradient(rows, beta_high)
    if gradient_low == 0.0 and gradient_high == 0.0:
        return IDENTITY_TEMPERATURE_NUMERATOR, "flat_identity"
    if gradient_low >= 0.0:
        optimum_beta = beta_low
        status = "lower_beta_boundary"
    elif gradient_high <= 0.0:
        optimum_beta = beta_high
        status = "upper_beta_boundary"
    else:
        low = beta_low
        high = beta_high
        for _ in range(96):
            midpoint = (low + high) / 2.0
            gradient = _temperature_gradient(rows, midpoint)
            if gradient < 0.0:
                low = midpoint
            else:
                high = midpoint
        optimum_beta = (low + high) / 2.0
        status = "interior_bisection"
    temperature = 1.0 / optimum_beta
    numerator = int(math.floor(temperature * TEMPERATURE_DENOMINATOR + 0.5))
    numerator = min(
        TEMPERATURE_MAX_NUMERATOR,
        max(TEMPERATURE_MIN_NUMERATOR, numerator),
    )
    return numerator, status


def _metric_summary(
    rows: Sequence[_BoundRow],
    temperature_payload: Mapping[str, Any],
    *,
    ece_bins: int,
) -> dict[str, Any]:
    applied_temperature = canonical_snapshot(
        _require_mapping(temperature_payload, label="applied_temperature")
    )
    temperature = _temperature_value(
        applied_temperature, label="applied_temperature"
    )
    if not rows:
        return {
            "applied_temperature": applied_temperature,
            "decision_count": 0,
            "root_count": 0,
            "nll_f64_hex": None,
            "nll_t1_f64_hex": None,
            "nll_uniform_f64_hex": None,
            "nll_delta_vs_t1_f64_hex": None,
            "nll_improvement_vs_uniform_f64_hex": None,
            "brier_f64_hex": None,
            "brier_t1_f64_hex": None,
            "brier_uniform_f64_hex": None,
            "brier_delta_vs_t1_f64_hex": None,
            "marginal_ece_f64_hex": None,
        }
    nll_values: list[float] = []
    nll_t1_values: list[float] = []
    nll_uniform_values: list[float] = []
    brier_values: list[float] = []
    brier_t1_values: list[float] = []
    brier_uniform_values: list[float] = []
    bin_counts = [0] * ece_bins
    bin_probability_sums = [0.0] * ece_bins
    bin_label_sums = [0] * ece_bins
    for row in rows:
        nll, brier, uniform_nll, uniform_brier = _row_values(row, temperature)
        nll_t1, brier_t1, _unused_nll, _unused_brier = _row_values(row, 1.0)
        nll_values.append(nll)
        nll_t1_values.append(nll_t1)
        nll_uniform_values.append(uniform_nll)
        brier_values.append(brier)
        brier_t1_values.append(brier_t1)
        brier_uniform_values.append(uniform_brier)
        probabilities = _stable_probabilities(row.evaluation.logits, temperature)
        for index, probability in enumerate(probabilities):
            bin_index = min(int(probability * ece_bins), ece_bins - 1)
            bin_counts[bin_index] += 1
            bin_probability_sums[bin_index] += probability
            bin_label_sums[bin_index] += int(index == row.evaluation.observed_index)

    count = len(rows)
    class_instance_count = sum(bin_counts)
    nll_mean = math.fsum(nll_values) / count
    nll_t1_mean = math.fsum(nll_t1_values) / count
    nll_uniform_mean = math.fsum(nll_uniform_values) / count
    brier_mean = math.fsum(brier_values) / count
    brier_t1_mean = math.fsum(brier_t1_values) / count
    brier_uniform_mean = math.fsum(brier_uniform_values) / count
    ece = math.fsum(
        (bin_count / class_instance_count)
        * abs(
            bin_probability_sums[index] / bin_count
            - bin_label_sums[index] / bin_count
        )
        for index, bin_count in enumerate(bin_counts)
        if bin_count
    )
    return {
        "applied_temperature": applied_temperature,
        "decision_count": count,
        "root_count": len({row.root_id for row in rows}),
        "nll_f64_hex": _canonical_float_hex(nll_mean, label="nll"),
        "nll_t1_f64_hex": _canonical_float_hex(nll_t1_mean, label="nll_t1"),
        "nll_uniform_f64_hex": _canonical_float_hex(
            nll_uniform_mean, label="nll_uniform"
        ),
        "nll_delta_vs_t1_f64_hex": _canonical_float_hex(
            nll_mean - nll_t1_mean, label="nll_delta_vs_t1"
        ),
        "nll_improvement_vs_uniform_f64_hex": _canonical_float_hex(
            nll_uniform_mean - nll_mean, label="nll_improvement_vs_uniform"
        ),
        "brier_f64_hex": _canonical_float_hex(brier_mean, label="brier"),
        "brier_t1_f64_hex": _canonical_float_hex(
            brier_t1_mean, label="brier_t1"
        ),
        "brier_uniform_f64_hex": _canonical_float_hex(
            brier_uniform_mean, label="brier_uniform"
        ),
        "brier_delta_vs_t1_f64_hex": _canonical_float_hex(
            brier_mean - brier_t1_mean, label="brier_delta_vs_t1"
        ),
        "marginal_ece_f64_hex": _canonical_float_hex(ece, label="marginal_ece"),
    }


def _bootstrap_metric_ci(
    rows: Sequence[_BoundRow],
    *,
    temperature: float,
    metric: str,
    replicates: int,
    seed: str,
    context: str,
) -> dict[str, Any]:
    if not rows:
        return {"lcb_f64_hex": None, "ucb_f64_hex": None}
    by_root: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        nll, _brier, uniform_nll, _uniform_brier = _row_values(row, temperature)
        nll_t1, _brier_t1, _u1, _u2 = _row_values(row, 1.0)
        if metric == "nll_delta_vs_t1":
            value = nll - nll_t1
        elif metric == "nll_improvement_vs_uniform":
            value = uniform_nll - nll
        else:  # pragma: no cover - private call sites are exhaustive
            raise ValueError(f"unsupported bootstrap metric: {metric}")
        by_root[row.root_id].append(value)
    roots = sorted(by_root)
    estimates: list[float] = []
    for replicate in range(replicates):
        sampled_values: list[float] = []
        for draw in range(len(roots)):
            digest = hashlib.sha256(
                (
                    f"{seed}\0{context}\0{metric}\0{replicate}\0{draw}"
                ).encode("utf-8")
            ).digest()
            root = roots[int.from_bytes(digest[:8], "big") % len(roots)]
            sampled_values.extend(by_root[root])
        estimates.append(math.fsum(sampled_values) / len(sampled_values))
    estimates.sort()
    lower_index = int(math.floor((replicates - 1) * 0.025))
    upper_index = int(math.ceil((replicates - 1) * 0.975))
    return {
        "lcb_f64_hex": _canonical_float_hex(
            estimates[lower_index], label="bootstrap_lcb"
        ),
        "ucb_f64_hex": _canonical_float_hex(
            estimates[upper_index], label="bootstrap_ucb"
        ),
    }


def _with_bootstrap(
    summary: dict[str, Any],
    rows: Sequence[_BoundRow],
    *,
    temperature: float,
    replicates: int,
    seed: str,
    context: str,
) -> dict[str, Any]:
    result = dict(summary)
    result["paired_root_bootstrap_95"] = {
        "nll_delta_vs_t1": _bootstrap_metric_ci(
            rows,
            temperature=temperature,
            metric="nll_delta_vs_t1",
            replicates=replicates,
            seed=seed,
            context=context,
        ),
        "nll_improvement_vs_uniform": _bootstrap_metric_ci(
            rows,
            temperature=temperature,
            metric="nll_improvement_vs_uniform",
            replicates=replicates,
            seed=seed,
            context=context,
        ),
    }
    return result


def _metric_float(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary[key]
    return None if value is None else _parse_float_hex(value, label=key)


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    observed: Any,
    required: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": canonical_snapshot(observed),
            "required": canonical_snapshot(required),
        }
    )


def _build_gate_result(
    *,
    config: Mapping[str, Any],
    temperatures: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    minimum = config["minimum_counts"]
    thresholds = {
        key: _parse_fraction(value, label=key)
        for key, value in config["quality_thresholds"].items()
    }
    checks: list[dict[str, Any]] = []

    for split in SPLIT_NAMES:
        required_decisions = minimum["decisions_per_role_by_split"][split]
        for role in ROLE_KEYS:
            summary = metrics[split]["by_role"][role]
            _check(
                checks,
                name=f"count.{split}.{role}.decisions",
                passed=summary["decision_count"] >= required_decisions,
                observed=summary["decision_count"],
                required={"gte": required_decisions},
            )
            _check(
                checks,
                name=f"count.{split}.{role}.roots",
                passed=summary["root_count"] >= minimum["roots_per_split_role"],
                observed=summary["root_count"],
                required={"gte": minimum["roots_per_split_role"]},
            )

    for role in ROLE_KEYS:
        selected = temperatures[role]["dev_selected_candidate"]
        fit_temperature = canonical_snapshot(
            _require_mapping(
                temperatures[role]["fit_temperature"],
                label=f"temperatures.{role}.fit_temperature",
            )
        )
        final_temperature = canonical_snapshot(
            _require_mapping(
                temperatures[role]["final_temperature"],
                label=f"temperatures.{role}.final_temperature",
            )
        )
        identity_temperature = _temperature_payload(
            IDENTITY_TEMPERATURE_NUMERATOR
        )
        expected_temperature = {
            "fit_temperature": fit_temperature,
            "identity_temperature": identity_temperature,
        }.get(selected)
        metric_temperature_bindings = [
            metrics[split]["by_role"][role].get("applied_temperature")
            for split in SPLIT_NAMES
        ]
        metric_temperature_bindings.extend(
            metrics[split]["by_role_joker"][role][joker].get(
                "applied_temperature"
            )
            for split in SPLIT_NAMES
            for joker in JOKER_KEYS
        )
        metric_temperature_bindings.extend(
            metrics["challenge"]["by_role_joker"][role][joker].get(
                "applied_temperature"
            )
            for joker in JOKER_KEYS
        )
        all_metrics_use_final_temperature = all(
            binding == final_temperature for binding in metric_temperature_bindings
        )
        selection_is_frozen_and_applied = (
            expected_temperature is not None
            and temperatures[role].get("dev_selection_frozen_before_locked_test")
            is True
            and final_temperature == expected_temperature
            and all_metrics_use_final_temperature
        )
        _check(
            checks,
            name=f"selection.{role}.dev_candidate_frozen_and_applied",
            passed=selection_is_frozen_and_applied,
            observed={
                "dev_selected_candidate": selected,
                "selection_frozen_before_locked_test": temperatures[role].get(
                    "dev_selection_frozen_before_locked_test"
                ),
                "final_temperature": final_temperature,
                "metric_temperature_binding_count": len(
                    metric_temperature_bindings
                ),
                "all_metrics_use_final_temperature": (
                    all_metrics_use_final_temperature
                ),
                "metric_temperature_bindings_sha256": canonical_sha256(
                    metric_temperature_bindings
                ),
            },
            required={
                "candidate": "fit_temperature_or_identity_temperature",
                "selection_frozen_before_locked_test": True,
                "final_and_locked_test_temperature": "exact_selected_candidate",
            },
        )
        strict_nonidentity = config["selection_policy"] == REQUIRE_NONIDENTITY_FIT
        selected_nonidentity_fit = (
            selected == "fit_temperature"
            and fit_temperature != identity_temperature
            and final_temperature == fit_temperature
        )
        _check(
            checks,
            name=f"selection.{role}.promotion_policy",
            passed=(not strict_nonidentity) or selected_nonidentity_fit,
            observed={
                "policy": config["selection_policy"],
                "dev_selected_candidate": selected,
                "final_temperature": final_temperature,
            },
            required=(
                "dev_selected_nonidentity_fit_temperature"
                if strict_nonidentity
                else "dev_selected_fit_or_identity_temperature"
            ),
        )
        role_test = metrics["test"]["by_role"][role]
        nll_delta = _metric_float(role_test, "nll_delta_vs_t1_f64_hex")
        nll_ucb_raw = role_test["paired_root_bootstrap_95"][
            "nll_delta_vs_t1"
        ]["ucb_f64_hex"]
        nll_ucb = (
            None
            if nll_ucb_raw is None
            else _parse_float_hex(nll_ucb_raw, label="nll_delta_ucb")
        )
        uniform_lcb_raw = role_test["paired_root_bootstrap_95"][
            "nll_improvement_vs_uniform"
        ]["lcb_f64_hex"]
        uniform_lcb = (
            None
            if uniform_lcb_raw is None
            else _parse_float_hex(uniform_lcb_raw, label="uniform_improvement_lcb")
        )
        brier_delta = _metric_float(role_test, "brier_delta_vs_t1_f64_hex")
        ece = _metric_float(role_test, "marginal_ece_f64_hex")
        residual = _temperature_value(
            temperatures[role]["locked_test_residual_temperature"],
            label="locked_test_residual_temperature",
        )
        quality_checks = (
            (
                "nll_delta_vs_t1",
                nll_delta,
                nll_delta is not None
                and nll_delta <= float(thresholds["max_test_nll_delta_vs_t1"]),
                {"lte": config["quality_thresholds"]["max_test_nll_delta_vs_t1"]},
            ),
            (
                "nll_delta_vs_t1_ucb",
                nll_ucb,
                nll_ucb is not None
                and nll_ucb
                <= float(thresholds["max_test_nll_delta_vs_t1_ucb"]),
                {
                    "lte": config["quality_thresholds"][
                        "max_test_nll_delta_vs_t1_ucb"
                    ]
                },
            ),
            (
                "uniform_improvement_lcb",
                uniform_lcb,
                uniform_lcb is not None
                and uniform_lcb
                >= float(
                    thresholds["min_test_uniform_improvement_role_lcb"]
                ),
                {
                    "gte": config["quality_thresholds"][
                        "min_test_uniform_improvement_role_lcb"
                    ]
                },
            ),
            (
                "marginal_ece",
                ece,
                ece is not None
                and ece <= float(thresholds["max_test_role_marginal_ece"]),
                {
                    "lte": config["quality_thresholds"][
                        "max_test_role_marginal_ece"
                    ]
                },
            ),
            (
                "brier_delta_vs_t1",
                brier_delta,
                brier_delta is not None
                and brier_delta
                <= float(thresholds["max_test_brier_delta_vs_t1"]),
                {
                    "lte": config["quality_thresholds"][
                        "max_test_brier_delta_vs_t1"
                    ]
                },
            ),
            (
                "residual_temperature",
                residual,
                float(thresholds["min_test_residual_temperature"])
                <= residual
                <= float(thresholds["max_test_residual_temperature"]),
                {
                    "gte": config["quality_thresholds"][
                        "min_test_residual_temperature"
                    ],
                    "lte": config["quality_thresholds"][
                        "max_test_residual_temperature"
                    ],
                },
            ),
        )
        for suffix, observed_float, passed, required in quality_checks:
            _check(
                checks,
                name=f"quality.test.{role}.{suffix}",
                passed=passed,
                observed=(
                    None
                    if observed_float is None
                    else _canonical_float_hex(observed_float, label=suffix)
                ),
                required=required,
            )

        if not config["challenge_contract"]["required_for_promotion"]:
            continue
        for joker in JOKER_KEYS:
            cell = metrics["challenge"]["by_role_joker"][role][joker]
            _check(
                checks,
                name=f"count.challenge.{role}.{joker}.decisions",
                passed=cell["decision_count"]
                >= minimum["challenge_decisions_per_role_joker"],
                observed=cell["decision_count"],
                required={"gte": minimum["challenge_decisions_per_role_joker"]},
            )
            _check(
                checks,
                name=f"count.challenge.{role}.{joker}.roots",
                passed=cell["root_count"]
                >= minimum["challenge_roots_per_role_joker"],
                observed=cell["root_count"],
                required={"gte": minimum["challenge_roots_per_role_joker"]},
            )
            cell_lcb_raw = cell["paired_root_bootstrap_95"][
                "nll_improvement_vs_uniform"
            ]["lcb_f64_hex"]
            cell_lcb = (
                None
                if cell_lcb_raw is None
                else _parse_float_hex(cell_lcb_raw, label="cell_uniform_lcb")
            )
            cell_ece = _metric_float(cell, "marginal_ece_f64_hex")
            _check(
                checks,
                name=f"quality.challenge.{role}.{joker}.uniform_improvement_lcb",
                passed=cell_lcb is not None
                and cell_lcb
                >= float(
                    thresholds["min_test_uniform_improvement_joker_lcb"]
                ),
                observed=(
                    None
                    if cell_lcb is None
                    else _canonical_float_hex(cell_lcb, label="cell_uniform_lcb")
                ),
                required={
                    "gte": config["quality_thresholds"][
                        "min_test_uniform_improvement_joker_lcb"
                    ]
                },
            )
            _check(
                checks,
                name=f"quality.challenge.{role}.{joker}.marginal_ece",
                passed=cell_ece is not None
                and cell_ece
                <= float(thresholds["max_test_joker_marginal_ece"]),
                observed=(
                    None
                    if cell_ece is None
                    else _canonical_float_hex(cell_ece, label="cell_ece")
                ),
                required={
                    "lte": config["quality_thresholds"][
                        "max_test_joker_marginal_ece"
                    ]
                },
            )

    passed = all(item["passed"] for item in checks)
    result: dict[str, Any] = {
        "schema": GATE_RESULT_SCHEMA,
        "gate_id": config["gate_id"],
        "gate_config_sha256": config["gate_config_sha256"],
        "locked_test_evaluation_passes": 1,
        "all_required_gates_passed": passed,
        "promotion_eligible": passed,
        "checks": checks,
        "failures": [item["name"] for item in checks if not item["passed"]],
    }
    result["gate_result_sha256"] = canonical_sha256(result)
    return result


def build_behavior_temperature_calibration(
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    challenge_records: Sequence[Mapping[str, Any]] = (),
    challenge_evaluation_rows: Sequence[Mapping[str, Any]] = (),
    gate_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic calibration artifact from raw, verified inputs."""
    config = verify_temperature_gate_config(
        gate_config if gate_config is not None else build_temperature_gate_config()
    )
    raw_manifest = verify_behavior_decision_dataset(records)
    records_by_sha: dict[str, Mapping[str, Any]] = {}
    for record in records:
        verified = verify_behavior_decision_log(record)
        if verified.record_sha256 in records_by_sha:
            raise ValueError("duplicate raw record SHA-256")
        records_by_sha[verified.record_sha256] = record

    evaluations_by_record: dict[str, VerifiedModelEvaluation] = {}
    evaluation_hashes: list[str] = []
    for row in evaluation_rows:
        record_sha = row.get("record_sha256") if isinstance(row, Mapping) else None
        if record_sha not in records_by_sha:
            raise ValueError("model evaluation row references an unknown raw record")
        verified_row = verify_model_evaluation_row(row, records_by_sha[record_sha])
        if verified_row.record_sha256 in evaluations_by_record:
            raise ValueError("duplicate model evaluation for one raw record")
        evaluations_by_record[verified_row.record_sha256] = verified_row
        evaluation_hashes.append(verified_row.evaluation_row_sha256)
    missing = sorted(set(records_by_sha) - set(evaluations_by_record))
    if missing:
        raise ValueError(f"missing model evaluation rows for {len(missing)} records")

    if bool(challenge_records) != bool(challenge_evaluation_rows):
        raise ValueError(
            "Joker challenge records and evaluation rows must be supplied together"
        )
    challenge_manifest: dict[str, Any] | None = None
    challenge_records_by_sha: dict[str, Mapping[str, Any]] = {}
    challenge_evaluations_by_record: dict[str, VerifiedModelEvaluation] = {}
    challenge_evaluation_hashes: list[str] = []
    if challenge_records:
        challenge_manifest = verify_behavior_decision_dataset(challenge_records)
        prefix = config["challenge_contract"]["root_namespace_prefix"]
        for record in challenge_records:
            verified = verify_behavior_decision_log(record)
            if not verified.root_id.startswith(prefix):
                raise ValueError(
                    "Joker challenge root is outside the configured namespace"
                )
            if verified.record_sha256 in challenge_records_by_sha:
                raise ValueError("duplicate Joker challenge raw record SHA-256")
            challenge_records_by_sha[verified.record_sha256] = record
        for row in challenge_evaluation_rows:
            record_sha = row.get("record_sha256") if isinstance(row, Mapping) else None
            if record_sha not in challenge_records_by_sha:
                raise ValueError(
                    "Joker challenge evaluation references an unknown raw record"
                )
            verified_row = verify_model_evaluation_row(
                row, challenge_records_by_sha[record_sha]
            )
            if verified_row.record_sha256 in challenge_evaluations_by_record:
                raise ValueError("duplicate Joker challenge model evaluation")
            challenge_evaluations_by_record[verified_row.record_sha256] = verified_row
            challenge_evaluation_hashes.append(
                verified_row.evaluation_row_sha256
            )
        challenge_missing = sorted(
            set(challenge_records_by_sha) - set(challenge_evaluations_by_record)
        )
        if challenge_missing:
            raise ValueError(
                "missing Joker challenge evaluations for "
                f"{len(challenge_missing)} records"
            )

    bound_rows = [
        _BoundRow(evaluations_by_record[record_sha])
        for record_sha in sorted(evaluations_by_record)
    ]
    challenge_bound_rows = [
        _BoundRow(challenge_evaluations_by_record[record_sha])
        for record_sha in sorted(challenge_evaluations_by_record)
    ]
    role_bindings: dict[str, Any] = {}
    for role in ROLE_KEYS:
        bindings = {
            (
                row.evaluation.checkpoint_sha256,
                row.evaluation.model_sha256,
                row.evaluation.row_extractor_sha256,
                row.evaluation.adapter_source_sha256,
            )
            for row in bound_rows
            if row.role == role
        }
        if len(bindings) > 1:
            raise ValueError(f"role {role} mixes model or extractor bindings")
        if not bindings:
            role_bindings[role] = None
        else:
            checkpoint, model, extractor, adapter = next(iter(bindings))
            role_bindings[role] = {
                "checkpoint_sha256": checkpoint,
                "model_sha256": model,
                "row_extractor_sha256": extractor,
                "adapter_source_sha256": adapter,
            }
    for row in challenge_bound_rows:
        main_binding = role_bindings[row.role]
        if main_binding is None:
            raise ValueError(
                f"Joker challenge role {row.role} has no main model binding"
            )
        challenge_binding = {
            "checkpoint_sha256": row.evaluation.checkpoint_sha256,
            "model_sha256": row.evaluation.model_sha256,
            "row_extractor_sha256": row.evaluation.row_extractor_sha256,
            "adapter_source_sha256": row.evaluation.adapter_source_sha256,
        }
        if challenge_binding != main_binding:
            raise ValueError(
                f"Joker challenge role {row.role} does not match main model binding"
            )

    grouped: dict[str, dict[str, list[_BoundRow]]] = {
        split: {role: [] for role in ROLE_KEYS} for split in SPLIT_NAMES
    }
    grouped_joker: dict[str, dict[str, dict[str, list[_BoundRow]]]] = {
        split: {
            role: {joker: [] for joker in JOKER_KEYS} for role in ROLE_KEYS
        }
        for split in SPLIT_NAMES
    }
    challenge_grouped_joker: dict[str, dict[str, list[_BoundRow]]] = {
        role: {joker: [] for joker in JOKER_KEYS} for role in ROLE_KEYS
    }
    roots_by_split = {split: set() for split in SPLIT_NAMES}
    for row in bound_rows:
        grouped[row.split][row.role].append(row)
        grouped_joker[row.split][row.role][row.joker_key].append(row)
        roots_by_split[row.split].add(row.root_id)
    challenge_roots: set[str] = set()
    for row in challenge_bound_rows:
        challenge_grouped_joker[row.role][row.joker_key].append(row)
        challenge_roots.add(row.root_id)
    intersections = {
        "fit_dev": len(roots_by_split["fit"] & roots_by_split["dev"]),
        "fit_test": len(roots_by_split["fit"] & roots_by_split["test"]),
        "dev_test": len(roots_by_split["dev"] & roots_by_split["test"]),
    }
    if any(intersections.values()):
        raise ValueError("root overlap across calibration splits")
    main_roots = set().union(*roots_by_split.values())
    challenge_prefix = config["challenge_contract"]["root_namespace_prefix"]
    if any(root.startswith(challenge_prefix) for root in main_roots):
        raise ValueError("main calibration root uses the Joker challenge namespace")
    main_challenge_overlap = len(main_roots & challenge_roots)
    if main_challenge_overlap:
        raise ValueError("main and Joker challenge root sets overlap")

    temperatures: dict[str, Any] = {}
    for role in ROLE_KEYS:
        fit_rows = grouped["fit"][role]
        dev_rows = grouped["dev"][role]
        fit_numerator, optimizer_status = _fit_temperature(fit_rows)
        fit_temperature = fit_numerator / TEMPERATURE_DENOMINATOR
        fit_dev_nll = _mean_nll(dev_rows, fit_temperature)
        identity_dev_nll = _mean_nll(dev_rows, 1.0)
        if (
            fit_dev_nll is not None
            and identity_dev_nll is not None
            and fit_numerator != IDENTITY_TEMPERATURE_NUMERATOR
            and fit_dev_nll < identity_dev_nll
        ):
            selected = "fit_temperature"
            final_numerator = fit_numerator
        else:
            selected = "identity_temperature"
            final_numerator = IDENTITY_TEMPERATURE_NUMERATOR
        final_temperature = final_numerator / TEMPERATURE_DENOMINATOR

        test_scaled_rows: list[_BoundRow] = []
        for row in grouped["test"][role]:
            scaled_evaluation = VerifiedModelEvaluation(
                **{
                    **row.evaluation.__dict__,
                    "logits": tuple(
                        value / final_temperature for value in row.evaluation.logits
                    ),
                }
            )
            test_scaled_rows.append(_BoundRow(scaled_evaluation))
        residual_numerator, residual_status = _fit_temperature(test_scaled_rows)
        temperatures[role] = {
            "optimizer": {
                "objective": "fit_observed_action_nll",
                "parameterization": "inverse_temperature_beta_bisection_96",
                "status": optimizer_status,
            },
            "fit_temperature": _temperature_payload(fit_numerator),
            "dev_candidate_nll": {
                "fit_temperature_f64_hex": (
                    None
                    if fit_dev_nll is None
                    else _canonical_float_hex(fit_dev_nll, label="fit_dev_nll")
                ),
                "identity_temperature_f64_hex": (
                    None
                    if identity_dev_nll is None
                    else _canonical_float_hex(identity_dev_nll, label="identity_dev_nll")
                ),
            },
            "dev_selected_candidate": selected,
            "dev_selection_frozen_before_locked_test": True,
            "final_temperature": _temperature_payload(final_numerator),
            "locked_test_residual_temperature": _temperature_payload(
                residual_numerator
            ),
            "locked_test_residual_optimizer_status": residual_status,
        }

    ece_bins = config["metric_contract"]["ece_bins"]
    bootstrap_replicates = config["bootstrap"]["replicates"]
    bootstrap_seed = config["bootstrap"]["seed"]
    metrics: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        by_role: dict[str, Any] = {}
        by_role_joker: dict[str, Any] = {}
        for role in ROLE_KEYS:
            temperature_payload = temperatures[role]["final_temperature"]
            temperature = _temperature_value(
                temperature_payload, label=f"temperatures.{role}.final_temperature"
            )
            summary = _metric_summary(
                grouped[split][role], temperature_payload, ece_bins=ece_bins
            )
            if split == "test":
                summary = _with_bootstrap(
                    summary,
                    grouped[split][role],
                    temperature=temperature,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                    context=f"{split}.{role}",
                )
            by_role[role] = summary
            by_role_joker[role] = {}
            for joker in JOKER_KEYS:
                cell_summary = _metric_summary(
                    grouped_joker[split][role][joker],
                    temperature_payload,
                    ece_bins=ece_bins,
                )
                if split == "test":
                    cell_summary = _with_bootstrap(
                        cell_summary,
                        grouped_joker[split][role][joker],
                        temperature=temperature,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed,
                        context=f"{split}.{role}.{joker}",
                    )
                by_role_joker[role][joker] = cell_summary
        metrics[split] = {
            "by_role": by_role,
            "by_role_joker": by_role_joker,
        }
    challenge_metrics: dict[str, Any] = {"by_role_joker": {}}
    for role in ROLE_KEYS:
        temperature_payload = temperatures[role]["final_temperature"]
        temperature = _temperature_value(
            temperature_payload, label=f"temperatures.{role}.final_temperature"
        )
        challenge_metrics["by_role_joker"][role] = {}
        for joker in JOKER_KEYS:
            rows = challenge_grouped_joker[role][joker]
            cell_summary = _metric_summary(
                rows,
                temperature_payload,
                ece_bins=ece_bins,
            )
            cell_summary = _with_bootstrap(
                cell_summary,
                rows,
                temperature=temperature,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                context=f"challenge.{role}.{joker}",
            )
            challenge_metrics["by_role_joker"][role][joker] = cell_summary
    metrics["challenge"] = challenge_metrics

    gate_result = _build_gate_result(
        config=config,
        temperatures=temperatures,
        metrics=metrics,
    )
    fit_dev_record_hashes = sorted(
        row.evaluation.record_sha256
        for row in bound_rows
        if row.split in ("fit", "dev")
    )
    fit_dev_evaluation_hashes = sorted(
        row.evaluation.evaluation_row_sha256
        for row in bound_rows
        if row.split in ("fit", "dev")
    )
    test_record_hashes = sorted(
        row.evaluation.record_sha256 for row in bound_rows if row.split == "test"
    )
    test_evaluation_hashes = sorted(
        row.evaluation.evaluation_row_sha256
        for row in bound_rows
        if row.split == "test"
    )
    artifact: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "promotion_eligible": gate_result["promotion_eligible"],
        "raw_decision_manifest": raw_manifest,
        "raw_record_set_sha256": canonical_sha256(sorted(records_by_sha)),
        "evaluation_row_set_sha256": canonical_sha256(sorted(evaluation_hashes)),
        "record_evaluation_binding_complete": True,
        "root_overlap_audit": {
            "checked": True,
            "pairwise_overlap_counts": intersections,
            "overlap_count": sum(intersections.values()),
            "main_challenge_overlap_count": main_challenge_overlap,
            "root_list_commitments_by_split": {
                split: canonical_sha256(sorted(roots_by_split[split]))
                for split in SPLIT_NAMES
            },
            "challenge_root_list_commitment": canonical_sha256(
                sorted(challenge_roots)
            ),
        },
        "role_model_bindings": role_bindings,
        "selection_contract": {
            "fit_dev_only_record_set_sha256": canonical_sha256(
                fit_dev_record_hashes
            ),
            "fit_dev_only_evaluation_set_sha256": canonical_sha256(
                fit_dev_evaluation_hashes
            ),
            "candidate_set": ["fit_temperature", "identity_temperature"],
            "promotion_policy": config["selection_policy"],
            "locked_test_was_selection_input": False,
            "selection_frozen_before_locked_test": True,
            "selected_candidate_by_role": {
                role: temperatures[role]["dev_selected_candidate"]
                for role in ROLE_KEYS
            },
            "selected_temperature_set_sha256": canonical_sha256(
                {
                    role: temperatures[role]["final_temperature"]
                    for role in ROLE_KEYS
                }
            ),
            "published_metrics_temperature_bound": True,
        },
        "locked_test_contract": {
            "record_set_sha256": canonical_sha256(test_record_hashes),
            "evaluation_set_sha256": canonical_sha256(test_evaluation_hashes),
            "evaluation_passes": 1,
            "used_for_selection": False,
        },
        "joker_challenge": {
            "present": bool(challenge_bound_rows),
            "required_for_promotion": config["challenge_contract"][
                "required_for_promotion"
            ],
            "root_namespace_prefix": config["challenge_contract"][
                "root_namespace_prefix"
            ],
            "raw_decision_manifest": challenge_manifest,
            "raw_record_set_sha256": canonical_sha256(
                sorted(challenge_records_by_sha)
            ),
            "evaluation_row_set_sha256": canonical_sha256(
                sorted(challenge_evaluation_hashes)
            ),
            "root_count": len(challenge_roots),
            "used_for_temperature_fit": False,
            "used_for_dev_selection": False,
            "used_for_locked_test_role_metrics": False,
        },
        "temperatures": temperatures,
        "metrics": metrics,
        "gate_config": config,
        "gate_result": gate_result,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def verify_behavior_temperature_calibration(
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    challenge_records: Sequence[Mapping[str, Any]] = (),
    challenge_evaluation_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild every metric/gate result and require exact artifact equality."""
    raw = _require_mapping(artifact, label="temperature calibration artifact")
    if raw.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported temperature calibration artifact schema")
    _unsigned_hash(raw, "artifact_sha256", label="temperature calibration artifact")
    gate_config = _require_mapping(raw.get("gate_config"), label="gate_config")
    verify_temperature_gate_config(gate_config)
    gate_result = _require_mapping(raw.get("gate_result"), label="gate_result")
    _unsigned_hash(gate_result, "gate_result_sha256", label="temperature gate result")
    rebuilt = build_behavior_temperature_calibration(
        records,
        evaluation_rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_evaluation_rows,
        gate_config=gate_config,
    )
    if canonical_snapshot(raw) != rebuilt:
        raise ValueError("temperature calibration artifact does not match raw inputs")
    return rebuilt


__all__ = [
    "ALLOW_DEV_SELECTED_CANDIDATE",
    "CALIBRATION_SCHEMA",
    "DEV_SELECTION_POLICIES",
    "GATE_CONFIG_SCHEMA",
    "GATE_RESULT_SCHEMA",
    "MODEL_EVALUATION_SCHEMA",
    "PreTemperatureLegalLogitEvaluator",
    "REQUIRE_NONIDENTITY_FIT",
    "ROLE_KEYS",
    "build_behavior_temperature_calibration",
    "build_model_evaluation_row",
    "build_temperature_gate_config",
    "verify_behavior_temperature_calibration",
    "verify_model_evaluation_row",
    "verify_temperature_gate_config",
]
