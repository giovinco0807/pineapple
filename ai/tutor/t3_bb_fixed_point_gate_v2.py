"""Production-only proof gate for the endogenous T3-BB fixed point.

Version one of the gate accepted self-hashed distributions and permissive
sample/threshold settings.  That was useful as a wiring diagnostic, but it was
not a promotion proof.  This module deliberately has no diagnostic pass mode:
promotion always means at least three independent seeds, one hundred locked
roots in each BB/BTN x Joker0/1/2 stratum, three trailing converged
transitions, and exact TV limits no weaker than 1/100.

The verifier reconstructs every public ``InfoSetKey`` and every T3-BB
``BehaviorInfoSet`` query, fresh-reads MCCFR checkpoint bundles, reconstructs
their complete average strategies, replays restricted physical-card ranges,
and re-verifies exact candidate-policy tables.  Candidate-query bundles
``Q_r`` and six-stratum evaluation bundles ``B_r`` are deliberately separate:
``Q_r -> C_r`` derives the exact policy table, then ``R_r(C_r) -> B_r`` derives
the posterior and evaluation policy.  This acyclic split makes the physical
artifacts constructible and rejects the former ``C_r/B_r/R_r`` content-hash
cycle.  Row distributions are merely redundant claims: policy values must
equal checkpoint strategy content and BTN posterior values must equal
independently replayed range artifacts.  A compact M3 likelihood binding can
only be emitted after repeating that full audit.  Convergence is a wiring and
fixed-point claim only; strategic strength remains the responsibility of the
independent M3 strength gate.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import action_key
from ai.tutor.promotion_gate_m3_full_card_strength import (
    INFORMATION_MODEL,
    POSITION_CONTRACT_VERSION,
    REQUIRED_EXCLUDED_PARTITIONS,
    REQUIRED_STRATA,
    RULESET,
    SOLVER_ADAPTER,
    SOLVER_METHOD,
    SOURCE_SCHEMA,
    SOLVER_SCHEMA,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    T3_FULL_CARD_RANGE_SOURCE_PATH,
    root_identity_commitment_sha256 as common_root_identity_commitment_sha256,
    verify_t3_bb_likelihood_binding,
)
from ai.tutor.t3_bb_candidate_queries import behavior_t3_bb_to_t3_first_key
from ai.tutor.calibrated_behavior_bootstrap import BOOTSTRAP_MODEL_TYPE
from ai.tutor.t3_bb_checkpoint_bundle import (
    T3BBCheckpointKey,
    load_verified_t3_bb_checkpoint_strategy_profiles,
    verify_t3_bb_checkpoint_bundle,
)
from ai.tutor.t3_bb_fixed_point_runtime import (
    ITERATION_MODEL_TYPE,
    MODEL_TYPE as CANDIDATE_MODEL_TYPE,
    verify_t3_bb_candidate_policy_artifact,
)
from ai.tutor.t3_bb_range_evidence import verify_restricted_range_evidence
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall


CONFIG_SCHEMA = "ofc_m3_t3_bb_fixed_point_gate_config/v2"
EVIDENCE_SCHEMA = "ofc_m3_t3_bb_fixed_point_evidence/v2"
RESULT_SCHEMA = "ofc_m3_t3_bb_fixed_point_gate_result/v2"
ROOT_MANIFEST_SCHEMA = "ofc_m3_t3_bb_fixed_point_roots/v2"
ROOT_COMMITMENT_SCHEMA = "ofc_m3_t3_bb_fixed_point_root_commitment/v2"
PARTITION_SCHEMA = "ofc_m3_t3_bb_fixed_point_excluded_partition/v2"
QUERY_MANIFEST_SCHEMA = "ofc_m3_t3_bb_candidate_query_manifest/v1"
QUERY_COMMITMENT_SCHEMA = "ofc_m3_t3_bb_candidate_query_commitment/v1"
GATE_ID = "promotion_gate_m3_t3_bb_fixed_point_v2"
SCOPE = "production_t3_bb_candidate_policy_btn_posterior_fixed_point"
EVIDENCE_KIND = "fresh_checkpoint_and_restricted_range_replay"
PASS_STATUS = "t3_bb_fixed_point_production_ready"
FAIL_STATUS = "t3_bb_fixed_point_production_blocked"

MIN_INDEPENDENT_SEEDS = 3
MIN_ROOTS_PER_STRATUM = 100
MIN_CONSECUTIVE_CONVERGED_ROUNDS = 3
MAX_POLICY_TV = Fraction(1, 100)
MAX_BTN_POSTERIOR_TV = Fraction(1, 100)
MAX_CROSS_SEED_POLICY_TV = Fraction(1, 100)
MAX_CROSS_SEED_BTN_POSTERIOR_TV = Fraction(1, 100)
Q32_DENOMINATOR = 1 << 32

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_KEYS = frozenset(
    {
        "root_id",
        "stratum",
        "actor",
        "phase",
        "visible_joker_count",
        "observation",
        "observation_digest",
        "root_identity_commitment_sha256",
        "root_commitment_sha256",
    }
)
_ROOT_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "locked_before_evaluation",
        "ruleset",
        "position_contract_version",
        "evaluation_seeds",
        "roots",
        "manifest_sha256",
    }
)
_PARTITION_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "root_identity_commitments",
        "solver_seeds",
        "manifest_sha256",
    }
)
_QUERY_KEYS = frozenset(
    {
        "query_id",
        "behavior_information",
        "behavior_information_digest",
        "converted_observation",
        "converted_observation_digest",
        "source_root_commitments",
        "query_commitment_sha256",
    }
)
_QUERY_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "evaluation_root_manifest_sha256",
        "queries",
        "manifest_sha256",
    }
)
_ROW_KEYS = frozenset(
    {
        "round_index",
        "solver_seed",
        "root_id",
        "root_identity_commitment_sha256",
        "root_commitment_sha256",
        "stratum",
        "actor",
        "phase",
        "visible_joker_count",
        "observation_digest",
        "previous_candidate_query_checkpoint_bundle_sha256",
        "current_candidate_query_checkpoint_bundle_sha256",
        "previous_evaluation_checkpoint_bundle_sha256",
        "current_evaluation_checkpoint_bundle_sha256",
        "previous_candidate_policy_artifact_sha256",
        "current_candidate_policy_artifact_sha256",
        "previous_range_evidence_artifact_sha256",
        "current_range_evidence_artifact_sha256",
        "previous_policy_distribution",
        "current_policy_distribution",
        "previous_btn_posterior_weights",
        "current_btn_posterior_weights",
        "exact_exploitability_computed",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "scope",
        "locked_before_evaluation",
        "ruleset",
        "position_contract_version",
        "information_model",
        "exact_exploitability_computed",
        "strategic_strength_evaluated",
        "requires_independent_strength_gate",
        "cold_start_zero_drift_is_strength_evidence",
        "candidate_policy_method",
        "selected_promotion_seed",
        "approved_source_manifest_sha256",
        "approved_solver_manifest_sha256",
        "approved_range_builder_source_sha256",
        "approved_root_manifest_sha256",
        "approved_candidate_query_manifest_sha256",
        "approved_excluded_partition_sha256",
        "thresholds",
        "gate_config_sha256",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "min_independent_seeds",
        "min_roots_per_stratum",
        "min_consecutive_converged_rounds",
        "max_policy_tv",
        "max_btn_posterior_weight_tv",
        "max_cross_seed_policy_tv",
        "max_cross_seed_btn_posterior_weight_tv",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "scope",
        "evidence_kind",
        "production_promotion_claim",
        "gate_config_sha256",
        "exact_exploitability_computed",
        "strategic_strength_evaluated",
        "requires_independent_strength_gate",
        "cold_start_zero_drift_is_strength_evidence",
        "candidate_policy_method",
        "source_manifest",
        "source_manifest_sha256",
        "solver_manifest",
        "solver_manifest_sha256",
        "root_manifest",
        "candidate_query_manifest",
        "excluded_partitions",
        "candidate_query_checkpoint_bundles",
        "evaluation_checkpoint_bundles",
        "candidate_policy_artifacts",
        "restricted_range_artifacts",
        "raw_iteration_rows",
        "published_summary",
        "artifact_sha256",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return canonical_sha256(unsigned)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label}: exact fields required; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label}: lowercase SHA256 required")
    return str(value)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label}: object required")
    return value


def _fraction(value: Any, *, label: str, nonnegative: bool = True) -> Fraction:
    if not isinstance(value, str) or value.strip() != value or "/" not in value:
        raise TypeError(f"{label}: canonical rational string required")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label}: invalid rational") from exc
    if value != f"{parsed.numerator}/{parsed.denominator}":
        raise ValueError(f"{label}: reduced canonical rational required")
    if nonnegative and parsed < 0:
        raise ValueError(f"{label}: nonnegative rational required")
    return parsed


def _encode_fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _distribution(
    value: Any, *, label: str, normalize: bool = False
) -> dict[str, Fraction]:
    raw = _require_mapping(value, label=label)
    if not raw:
        raise ValueError(f"{label}: non-empty distribution required")
    result: dict[str, Fraction] = {}
    for key, encoded in sorted(raw.items()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label}: non-empty string keys required")
        result[key] = _fraction(encoded, label=f"{label}.{key}")
    total = sum(result.values(), Fraction(0, 1))
    if total <= 0:
        raise ValueError(f"{label}: positive total mass required")
    if normalize:
        return {key: value / total for key, value in result.items()}
    if total != 1:
        raise ValueError(f"{label}: probabilities must sum exactly to one")
    return result


def encode_distribution(value: Mapping[str, Fraction]) -> dict[str, str]:
    total = sum(value.values(), Fraction(0, 1))
    if not value or total <= 0:
        raise ValueError("distribution must have positive mass")
    return {
        key: _encode_fraction(Fraction(raw) / total)
        for key, raw in sorted(value.items())
    }


def _tv(left: Mapping[str, Fraction], right: Mapping[str, Fraction]) -> Fraction:
    keys = set(left) | set(right)
    return sum(
        (abs(left.get(key, Fraction(0, 1)) - right.get(key, Fraction(0, 1))) for key in keys),
        Fraction(0, 1),
    ) / 2


def quantize_exact_distribution_q32(
    distribution: Mapping[str, Fraction],
) -> dict[str, Fraction]:
    """Largest-remainder Q32 quantization over exact rational input."""

    if not distribution:
        raise ValueError("Q32 distribution must not be empty")
    values = {key: Fraction(value) for key, value in distribution.items()}
    if any(not isinstance(key, str) or not key for key in values):
        raise TypeError("Q32 action IDs must be non-empty strings")
    if any(value < 0 for value in values.values()):
        raise ValueError("Q32 probabilities must be nonnegative")
    total = sum(values.values(), Fraction(0, 1))
    if total <= 0:
        raise ValueError("Q32 distribution must carry positive mass")
    floors: dict[str, int] = {}
    remainders: list[tuple[Fraction, str]] = []
    for action_id in sorted(values):
        quota = values[action_id] * Q32_DENOMINATOR / total
        units = quota.numerator // quota.denominator
        floors[action_id] = units
        remainders.append((quota - units, action_id))
    remaining = Q32_DENOMINATOR - sum(floors.values())
    if not 0 <= remaining <= len(floors):
        raise RuntimeError("Q32 largest remainder is invalid")
    for _remainder, action_id in sorted(
        remainders, key=lambda item: (-item[0], item[1])
    )[:remaining]:
        floors[action_id] += 1
    result = {
        action_id: Fraction(units, Q32_DENOMINATOR)
        for action_id, units in sorted(floors.items())
    }
    if sum(result.values(), Fraction(0, 1)) != 1:
        raise RuntimeError("Q32 result does not sum to one")
    return result


def _parse_recall(value: Any, *, label: str) -> PrivateRecall:
    raw = _require_mapping(value, label=label)
    _exact_keys(raw, frozenset({"dealt_by_turn", "discards_by_turn"}), label=label)
    dealt_raw = raw["dealt_by_turn"]
    discard_raw = raw["discards_by_turn"]
    if not isinstance(dealt_raw, list) or not isinstance(discard_raw, list):
        raise TypeError(f"{label}: recall arrays required")
    dealt: list[tuple[int, tuple[str, ...]]] = []
    for index, item in enumerate(dealt_raw):
        row = _require_mapping(item, label=f"{label}.dealt_by_turn[{index}]")
        if set(row) != {"turn", "cards"} or not isinstance(row["cards"], list):
            raise ValueError(f"{label}.dealt_by_turn[{index}]: schema mismatch")
        dealt.append((row["turn"], tuple(row["cards"])))
    discards: list[tuple[int, str]] = []
    for index, item in enumerate(discard_raw):
        row = _require_mapping(item, label=f"{label}.discards_by_turn[{index}]")
        if set(row) != {"turn", "card"}:
            raise ValueError(f"{label}.discards_by_turn[{index}]: schema mismatch")
        discards.append((row["turn"], row["card"]))
    recall = PrivateRecall(tuple(dealt), tuple(discards))
    if recall.to_canonical_dict() != dict(raw):
        raise ValueError(f"{label}: non-canonical recall")
    return recall


def reconstruct_infoset_key(value: Any, *, label: str = "observation") -> InfoSetKey:
    raw = _require_mapping(value, label=label)
    expected = frozenset(
        {
            "contract_version",
            "actor",
            "turn",
            "phase",
            "board_bb",
            "board_btn",
            "public_action_history",
            "own_recall",
            "current_draw",
            "fantasy_state",
        }
    )
    _exact_keys(raw, expected, label=label)
    boards: dict[str, tuple[tuple[str, ...], ...]] = {}
    for actor in ("bb", "btn"):
        board = _require_mapping(raw[f"board_{actor}"], label=f"{label}.board_{actor}")
        if set(board) != {"top", "middle", "bottom"}:
            raise ValueError(f"{label}.board_{actor}: exact row fields required")
        boards[actor] = tuple(tuple(board[row]) for row in ("top", "middle", "bottom"))
    history_raw = raw["public_action_history"]
    if not isinstance(history_raw, list):
        raise TypeError(f"{label}.public_action_history: list required")
    history = []
    for index, item in enumerate(history_raw):
        row = _require_mapping(item, label=f"{label}.public_action_history[{index}]")
        if set(row) != {"turn", "actor", "placements"} or not isinstance(row["placements"], list):
            raise ValueError(f"{label}.public_action_history[{index}]: schema mismatch")
        placements = []
        for placement in row["placements"]:
            if not isinstance(placement, list) or len(placement) != 2:
                raise ValueError(f"{label}: placements must be [card,row]")
            placements.append((placement[0], placement[1]))
        history.append((row["turn"], row["actor"], tuple(placements)))
    key = InfoSetKey(
        contract_version=raw["contract_version"],
        actor=raw["actor"],
        turn=raw["turn"],
        phase=raw["phase"],
        board_bb=boards["bb"],
        board_btn=boards["btn"],
        public_action_history=tuple(history),
        own_recall=_parse_recall(raw["own_recall"], label=f"{label}.own_recall"),
        current_draw=tuple(raw["current_draw"]),
        fantasy_state=raw["fantasy_state"],
    )
    if key.to_canonical_dict() != dict(raw):
        raise ValueError(f"{label}: payload is not canonical")
    key.canonical_json()
    return key


def reconstruct_behavior_information(
    value: Any, *, label: str = "behavior_information"
) -> BehaviorInfoSet:
    raw = _require_mapping(value, label=label)
    expected = frozenset(
        {
            "position_contract_version",
            "actor",
            "turn",
            "board_bb",
            "board_btn",
            "public_action_history",
            "own_recall_before",
            "current_draw",
            "legal_action_ids",
            "fantasy_state",
        }
    )
    _exact_keys(raw, expected, label=label)
    if raw["position_contract_version"] != POSITION_CONTRACT_VERSION:
        raise ValueError(f"{label}: position contract mismatch")
    # Reuse the strict InfoSet parser by supplying the T3-first phase and the
    # corresponding own-recall field, then independently validate legal support
    # through behavior_t3_bb_to_t3_first_key below.
    observation_payload = {
        "contract_version": raw["position_contract_version"],
        "actor": raw["actor"],
        "turn": raw["turn"],
        "phase": "t3_first",
        "board_bb": raw["board_bb"],
        "board_btn": raw["board_btn"],
        "public_action_history": raw["public_action_history"],
        "own_recall": raw["own_recall_before"],
        "current_draw": raw["current_draw"],
        "fantasy_state": raw["fantasy_state"],
    }
    key = reconstruct_infoset_key(observation_payload, label=f"{label}.as_infoset")
    legal = raw["legal_action_ids"]
    if not isinstance(legal, list):
        raise TypeError(f"{label}.legal_action_ids: list required")
    information = BehaviorInfoSet(
        actor=key.actor,
        turn=key.turn,
        board_bb=key.board_bb,
        board_btn=key.board_btn,
        public_action_history=key.public_action_history,
        own_recall_before=key.own_recall,
        current_draw=key.current_draw,
        legal_action_ids=tuple(legal),
        fantasy_state=key.fantasy_state,
    )
    if information.to_canonical_dict() != dict(raw):
        raise ValueError(f"{label}: payload is not canonical")
    behavior_t3_bb_to_t3_first_key(information)
    return information


def visible_joker_count(key: InfoSetKey) -> int:
    cards: set[str] = set(key.current_draw)
    for board in (key.board_bb, key.board_btn):
        for row in board:
            cards.update(row)
    for _turn, dealt in key.own_recall.dealt_by_turn:
        cards.update(dealt)
    cards.update(card for _turn, card in key.own_recall.discards_by_turn)
    return len(cards & {"X1", "X2"})


def _legal_action_ids_for_infoset(key: InfoSetKey) -> tuple[str, ...]:
    rows = key.board_bb if key.actor == "bb" else key.board_btn
    board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
    actions = get_turn_actions(list(key.current_draw), board)
    result = tuple(sorted(action_key(action) for action in actions))
    if not result or len(result) != len(set(result)):
        raise ValueError("information set has invalid legal action support")
    return result


def root_identity_commitment_sha256(root_id: str, observation_digest: str) -> str:
    if not isinstance(root_id, str) or not root_id.strip():
        raise ValueError("root_id must be non-empty")
    _require_sha256(observation_digest, label="observation_digest")
    # Keep the established split-independent identity domain so a source root
    # reused by training/calibration/smoke cannot evade overlap checks by
    # changing its observation.  The separate full commitment below binds this
    # identity to the freshly reconstructed observation digest and payload.
    return common_root_identity_commitment_sha256(root_id)


def root_commitment_sha256(root: Mapping[str, Any]) -> str:
    content = dict(root)
    content.pop("root_commitment_sha256", None)
    return canonical_sha256({"schema": ROOT_COMMITMENT_SCHEMA, "root": content})


def query_commitment_sha256(query: Mapping[str, Any]) -> str:
    content = dict(query)
    content.pop("query_commitment_sha256", None)
    return canonical_sha256({"schema": QUERY_COMMITMENT_SCHEMA, "query": content})


def build_root_record(root_id: str, observation: InfoSetKey) -> dict[str, Any]:
    if not isinstance(observation, InfoSetKey):
        raise TypeError("observation must be InfoSetKey")
    if observation.turn != 3 or observation.actor not in ("bb", "btn"):
        raise ValueError("root observation must be a T3 BB/BTN decision")
    phase = "t3_first" if observation.actor == "bb" else "t3_second"
    if observation.phase != phase:
        raise ValueError("root actor/phase mismatch")
    joker_count = visible_joker_count(observation)
    digest = observation.digest()
    root: dict[str, Any] = {
        "root_id": root_id,
        "stratum": f"{observation.actor}_joker{joker_count}",
        "actor": observation.actor,
        "phase": observation.phase,
        "visible_joker_count": joker_count,
        "observation": observation.to_canonical_dict(),
        "observation_digest": digest,
        "root_identity_commitment_sha256": root_identity_commitment_sha256(root_id, digest),
    }
    root["root_commitment_sha256"] = root_commitment_sha256(root)
    return root


def build_root_manifest(
    roots: Sequence[Mapping[str, Any]], *, evaluation_seeds: Sequence[int]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_production_fixed_point_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "evaluation_seeds": list(evaluation_seeds),
        "roots": sorted(
            (copy.deepcopy(dict(root)) for root in roots),
            key=lambda root: (str(root.get("stratum")), str(root.get("root_id"))),
        ),
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    _validate_root_manifest(manifest, enforce_production_minima=False)
    return manifest


def build_candidate_query_record(
    query_id: str,
    information: BehaviorInfoSet,
    *,
    source_root_commitments: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("query_id must be non-empty")
    converted = behavior_t3_bb_to_t3_first_key(information)
    sources = sorted(set(source_root_commitments))
    if not sources or any(not _is_sha256(value) for value in sources):
        raise ValueError("candidate query requires source root commitments")
    query: dict[str, Any] = {
        "query_id": query_id,
        "behavior_information": information.to_canonical_dict(),
        "behavior_information_digest": information.digest(),
        "converted_observation": converted.to_canonical_dict(),
        "converted_observation_digest": converted.digest(),
        "source_root_commitments": sources,
    }
    query["query_commitment_sha256"] = query_commitment_sha256(query)
    return query


def build_candidate_query_manifest(
    queries: Sequence[Mapping[str, Any]], *, evaluation_root_manifest_sha256: str
) -> dict[str, Any]:
    _require_sha256(
        evaluation_root_manifest_sha256, label="evaluation_root_manifest_sha256"
    )
    manifest: dict[str, Any] = {
        "schema": QUERY_MANIFEST_SCHEMA,
        "purpose": "candidate_behavior_query_to_public_solver_infoset_coverage",
        "evaluation_root_manifest_sha256": evaluation_root_manifest_sha256,
        "queries": sorted(
            (copy.deepcopy(dict(query)) for query in queries),
            key=lambda query: str(query.get("query_id")),
        ),
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    return manifest


def build_excluded_partition(
    purpose: str,
    *,
    root_identity_commitments: Sequence[str],
    solver_seeds: Sequence[int],
) -> dict[str, Any]:
    if purpose not in REQUIRED_EXCLUDED_PARTITIONS:
        raise ValueError("unsupported excluded partition purpose")
    partition: dict[str, Any] = {
        "schema": PARTITION_SCHEMA,
        "purpose": purpose,
        "root_identity_commitments": sorted(set(root_identity_commitments)),
        "solver_seeds": sorted(set(solver_seeds)),
    }
    partition["manifest_sha256"] = self_hash(partition, "manifest_sha256")
    return partition


def _production_thresholds() -> dict[str, Any]:
    return {
        "min_independent_seeds": MIN_INDEPENDENT_SEEDS,
        "min_roots_per_stratum": MIN_ROOTS_PER_STRATUM,
        "min_consecutive_converged_rounds": MIN_CONSECUTIVE_CONVERGED_ROUNDS,
        "max_policy_tv": _encode_fraction(MAX_POLICY_TV),
        "max_btn_posterior_weight_tv": _encode_fraction(MAX_BTN_POSTERIOR_TV),
        "max_cross_seed_policy_tv": _encode_fraction(MAX_CROSS_SEED_POLICY_TV),
        "max_cross_seed_btn_posterior_weight_tv": _encode_fraction(
            MAX_CROSS_SEED_BTN_POSTERIOR_TV
        ),
    }


def build_locked_t3_bb_fixed_point_gate_config(
    *,
    selected_promotion_seed: int,
    approved_source_manifest_sha256: str,
    approved_solver_manifest_sha256: str,
    approved_range_builder_source_sha256: str,
    approved_root_manifest_sha256: str,
    approved_candidate_query_manifest_sha256: str,
    approved_excluded_partition_sha256: Mapping[str, str],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "information_model": INFORMATION_MODEL,
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
        "candidate_policy_method": SOLVER_METHOD,
        "selected_promotion_seed": selected_promotion_seed,
        "approved_source_manifest_sha256": approved_source_manifest_sha256,
        "approved_solver_manifest_sha256": approved_solver_manifest_sha256,
        "approved_range_builder_source_sha256": approved_range_builder_source_sha256,
        "approved_root_manifest_sha256": approved_root_manifest_sha256,
        "approved_candidate_query_manifest_sha256": approved_candidate_query_manifest_sha256,
        "approved_excluded_partition_sha256": dict(approved_excluded_partition_sha256),
        "thresholds": _production_thresholds(),
    }
    config["gate_config_sha256"] = self_hash(config, "gate_config_sha256")
    return verify_locked_t3_bb_fixed_point_gate_config(config)


def verify_locked_t3_bb_fixed_point_gate_config(config: Any) -> dict[str, Any]:
    raw = _require_mapping(config, label="config")
    _exact_keys(raw, _CONFIG_KEYS, label="config")
    expected = {
        "schema": CONFIG_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "information_model": INFORMATION_MODEL,
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
        "candidate_policy_method": SOLVER_METHOD,
        "thresholds": _production_thresholds(),
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            raise ValueError(f"config.{field}: production constant mismatch")
    if not _is_int(raw.get("selected_promotion_seed")):
        raise ValueError("config.selected_promotion_seed: integer required")
    for field in (
        "approved_source_manifest_sha256",
        "approved_solver_manifest_sha256",
        "approved_range_builder_source_sha256",
        "approved_root_manifest_sha256",
        "approved_candidate_query_manifest_sha256",
        "gate_config_sha256",
    ):
        _require_sha256(raw.get(field), label=f"config.{field}")
    partitions = _require_mapping(
        raw.get("approved_excluded_partition_sha256"),
        label="config.approved_excluded_partition_sha256",
    )
    if set(partitions) != set(REQUIRED_EXCLUDED_PARTITIONS):
        raise ValueError("config: exact training/calibration/smoke partition set required")
    for purpose, digest in partitions.items():
        _require_sha256(digest, label=f"config.partition.{purpose}")
    if self_hash(raw, "gate_config_sha256") != raw["gate_config_sha256"]:
        raise ValueError("config.gate_config_sha256: self-hash mismatch")
    return copy.deepcopy(dict(raw))


def _validate_root_manifest(
    value: Any, *, enforce_production_minima: bool = True
) -> dict[str, Any]:
    raw = _require_mapping(value, label="root_manifest")
    _exact_keys(raw, _ROOT_MANIFEST_KEYS, label="root_manifest")
    expected = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_production_fixed_point_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            raise ValueError(f"root_manifest.{field}: mismatch")
    if self_hash(raw, "manifest_sha256") != raw.get("manifest_sha256"):
        raise ValueError("root_manifest.manifest_sha256: self-hash mismatch")
    seeds = raw.get("evaluation_seeds")
    if not isinstance(seeds, list) or any(not _is_int(seed) for seed in seeds):
        raise TypeError("root_manifest.evaluation_seeds: integer list required")
    if not seeds or seeds != sorted(set(seeds)):
        raise ValueError("root_manifest: canonical non-empty independent seeds required")
    if enforce_production_minima and len(seeds) < MIN_INDEPENDENT_SEEDS:
        raise ValueError("root_manifest: at least 3 canonical independent seeds required")
    roots = raw.get("roots")
    if not isinstance(roots, list):
        raise TypeError("root_manifest.roots: list required")
    by_id: dict[str, dict[str, Any]] = {}
    counts = {stratum: 0 for stratum in REQUIRED_STRATA}
    identities: set[str] = set()
    commitments: set[str] = set()
    legal_by_id: dict[str, tuple[str, ...]] = {}
    prior_sort_key: tuple[str, str] | None = None
    for index, item in enumerate(roots):
        root = _require_mapping(item, label=f"root_manifest.roots[{index}]")
        _exact_keys(root, _ROOT_KEYS, label=f"root_manifest.roots[{index}]")
        root_id = root.get("root_id")
        if not isinstance(root_id, str) or not root_id.strip() or root_id in by_id:
            raise ValueError("root_manifest: unique non-empty root IDs required")
        key = reconstruct_infoset_key(root.get("observation"), label=f"root[{root_id}].observation")
        digest = key.digest()
        actor = key.actor
        phase = "t3_first" if actor == "bb" else "t3_second"
        joker_count = visible_joker_count(key)
        stratum = f"{actor}_joker{joker_count}"
        comparisons = {
            "observation_digest": digest,
            "actor": actor,
            "phase": phase,
            "visible_joker_count": joker_count,
            "stratum": stratum,
            "root_identity_commitment_sha256": root_identity_commitment_sha256(root_id, digest),
            "root_commitment_sha256": root_commitment_sha256(root),
        }
        if key.turn != 3 or key.phase != phase:
            raise ValueError(f"root[{root_id}]: actor/phase must be a canonical T3 decision")
        for field, wanted in comparisons.items():
            if root.get(field) != wanted:
                raise ValueError(f"root[{root_id}].{field}: reconstructed value mismatch")
        if stratum not in REQUIRED_STRATA:
            raise ValueError(f"root[{root_id}]: unsupported stratum")
        sort_key = (stratum, root_id)
        if prior_sort_key is not None and sort_key <= prior_sort_key:
            raise ValueError("root_manifest.roots: canonical stratum/root_id order required")
        prior_sort_key = sort_key
        identity = str(root["root_identity_commitment_sha256"])
        commitment = str(root["root_commitment_sha256"])
        if identity in identities or commitment in commitments:
            raise ValueError("root_manifest: duplicate identity/full commitment")
        identities.add(identity)
        commitments.add(commitment)
        counts[stratum] += 1
        by_id[root_id] = copy.deepcopy(dict(root))
        legal_by_id[root_id] = _legal_action_ids_for_infoset(key)
    if set(counts) != set(REQUIRED_STRATA) or any(count <= 0 for count in counts.values()):
        raise ValueError("root_manifest: every exact stratum requires at least one root")
    if enforce_production_minima and any(
        count < MIN_ROOTS_PER_STRATUM for count in counts.values()
    ):
        raise ValueError("root_manifest: every exact stratum requires at least 100 roots")
    return {
        "raw": copy.deepcopy(dict(raw)),
        "by_id": by_id,
        "seeds": tuple(seeds),
        "identities": identities,
        "commitments": commitments,
        "counts": counts,
        "legal_by_id": legal_by_id,
    }


def _validate_candidate_query_manifest(
    value: Any, *, root_state: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _require_mapping(value, label="candidate_query_manifest")
    _exact_keys(raw, _QUERY_MANIFEST_KEYS, label="candidate_query_manifest")
    if raw.get("schema") != QUERY_MANIFEST_SCHEMA or raw.get("purpose") != (
        "candidate_behavior_query_to_public_solver_infoset_coverage"
    ):
        raise ValueError("candidate_query_manifest: contract mismatch")
    if raw.get("evaluation_root_manifest_sha256") != root_state["raw"]["manifest_sha256"]:
        raise ValueError("candidate_query_manifest: evaluation root binding mismatch")
    if self_hash(raw, "manifest_sha256") != raw.get("manifest_sha256"):
        raise ValueError("candidate_query_manifest.manifest_sha256: self-hash mismatch")
    queries = raw.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("candidate_query_manifest.queries: non-empty list required")
    by_behavior_digest: dict[str, dict[str, Any]] = {}
    query_ids: set[str] = set()
    covered_roots: set[str] = set()
    prior: str | None = None
    for index, item in enumerate(queries):
        query = _require_mapping(item, label=f"candidate queries[{index}]")
        _exact_keys(query, _QUERY_KEYS, label=f"candidate queries[{index}]")
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in query_ids:
            raise ValueError("candidate queries: unique non-empty query IDs required")
        if prior is not None and query_id <= prior:
            raise ValueError("candidate queries: canonical query_id order required")
        prior = query_id
        query_ids.add(query_id)
        information = reconstruct_behavior_information(
            query.get("behavior_information"), label=f"candidate query {query_id}"
        )
        converted = behavior_t3_bb_to_t3_first_key(information)
        converted_from_payload = reconstruct_infoset_key(
            query.get("converted_observation"),
            label=f"candidate query {query_id}.converted_observation",
        )
        if converted.to_canonical_dict() != converted_from_payload.to_canonical_dict():
            raise ValueError(f"candidate query {query_id}: conversion payload mismatch")
        behavior_digest = information.digest()
        if behavior_digest in by_behavior_digest:
            raise ValueError("candidate queries: duplicate behavior information digest")
        comparisons = {
            "behavior_information_digest": behavior_digest,
            "converted_observation_digest": converted.digest(),
            "query_commitment_sha256": query_commitment_sha256(query),
        }
        for field, wanted in comparisons.items():
            if query.get(field) != wanted:
                raise ValueError(f"candidate query {query_id}.{field}: mismatch")
        sources = query.get("source_root_commitments")
        if (
            not isinstance(sources, list)
            or not sources
            or sources != sorted(set(sources))
            or any(source not in root_state["commitments"] for source in sources)
        ):
            raise ValueError(f"candidate query {query_id}: invalid source root coverage")
        covered_roots.update(sources)
        by_behavior_digest[behavior_digest] = copy.deepcopy(dict(query))
    expected_source_roots = {
        root["root_commitment_sha256"]
        for root in root_state["by_id"].values()
        if root["actor"] == "btn"
    }
    if covered_roots != expected_source_roots:
        raise ValueError(
            "candidate queries: exact BTN t3_second source-root coverage required"
        )
    return {"raw": copy.deepcopy(dict(raw)), "by_behavior_digest": by_behavior_digest}


def _validate_partitions(
    value: Any, *, root_state: Mapping[str, Any], evaluation_seeds: Sequence[int]
) -> dict[str, str]:
    raw = _require_mapping(value, label="excluded_partitions")
    if set(raw) != set(REQUIRED_EXCLUDED_PARTITIONS):
        raise ValueError("excluded_partitions: exact training/calibration/smoke set required")
    roots_by_partition: dict[str, set[str]] = {}
    seeds_by_partition: dict[str, set[int]] = {}
    hashes: dict[str, str] = {}
    for purpose in REQUIRED_EXCLUDED_PARTITIONS:
        partition = _require_mapping(raw[purpose], label=f"partition.{purpose}")
        _exact_keys(partition, _PARTITION_KEYS, label=f"partition.{purpose}")
        if partition.get("schema") != PARTITION_SCHEMA or partition.get("purpose") != purpose:
            raise ValueError(f"partition.{purpose}: contract mismatch")
        if self_hash(partition, "manifest_sha256") != partition.get("manifest_sha256"):
            raise ValueError(f"partition.{purpose}: self-hash mismatch")
        identities = partition.get("root_identity_commitments")
        seeds = partition.get("solver_seeds")
        if (
            not isinstance(identities, list)
            or not identities
            or identities != sorted(set(identities))
            or any(not _is_sha256(item) for item in identities)
        ):
            raise ValueError(f"partition.{purpose}: canonical non-empty root identities required")
        if (
            not isinstance(seeds, list)
            or not seeds
            or seeds != sorted(set(seeds))
            or any(not _is_int(seed) for seed in seeds)
        ):
            raise ValueError(f"partition.{purpose}: canonical non-empty seeds required")
        roots_by_partition[purpose] = set(identities)
        seeds_by_partition[purpose] = set(seeds)
        hashes[purpose] = str(partition["manifest_sha256"])
        if roots_by_partition[purpose] & root_state["identities"]:
            raise ValueError(f"partition.{purpose}: evaluation root overlap")
        if seeds_by_partition[purpose] & set(evaluation_seeds):
            raise ValueError(f"partition.{purpose}: evaluation seed overlap")
    for left, right in itertools.combinations(REQUIRED_EXCLUDED_PARTITIONS, 2):
        if roots_by_partition[left] & roots_by_partition[right]:
            raise ValueError(f"excluded partitions {left}/{right}: root overlap")
        if seeds_by_partition[left] & seeds_by_partition[right]:
            raise ValueError(f"excluded partitions {left}/{right}: seed overlap")
    return hashes


def _validate_source_and_solver(
    evidence: Mapping[str, Any], *, config: Mapping[str, Any], workspace_root: Path
) -> tuple[str, str, str]:
    source = _require_mapping(evidence.get("source_manifest"), label="source_manifest")
    if set(source) != {"schema", "files"} or source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source_manifest: exact schema/files contract required")
    source_sha = canonical_sha256(source)
    if source_sha != evidence.get("source_manifest_sha256") or source_sha != config[
        "approved_source_manifest_sha256"
    ]:
        raise ValueError("source_manifest: hash/lock mismatch")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source_manifest.files: non-empty list required")
    seen: set[str] = set()
    range_source_sha: str | None = None
    for item in files:
        row = _require_mapping(item, label="source_manifest.files[]")
        if set(row) != {"path", "sha256"}:
            raise ValueError("source_manifest file: exact path/sha256 required")
        path = row.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path in seen
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("source_manifest file: safe unique repository path required")
        seen.add(path)
        digest = _require_sha256(row.get("sha256"), label=f"source file {path}")
        resolved_root = workspace_root.resolve()
        resolved = (resolved_root / path).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("source manifest path escapes workspace") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(f"source file {path}: regular non-symlink file required")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source file {path}: fresh bytes hash mismatch")
        if path == T3_FULL_CARD_RANGE_SOURCE_PATH:
            range_source_sha = digest
    if [item["path"] for item in files] != sorted(seen):
        raise ValueError("source_manifest.files: canonical path order required")
    if range_source_sha is None or range_source_sha != config[
        "approved_range_builder_source_sha256"
    ]:
        raise ValueError("source_manifest: required range builder source mismatch")

    solver = _require_mapping(evidence.get("solver_manifest"), label="solver_manifest")
    expected_solver = {
        "schema": SOLVER_SCHEMA,
        "method": SOLVER_METHOD,
        "adapter": SOLVER_ADAPTER,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "turns": [3, 4],
        "actors": ["bb", "btn"],
        "physical_joker_ids": ["X1", "X2"],
        "information_model": INFORMATION_MODEL,
        "strategy_fusion": False,
        "full_card": True,
        "hu_exact": False,
        "exact_exploitability_computed": False,
        "source_manifest_sha256": source_sha,
    }
    if dict(solver) != expected_solver:
        raise ValueError("solver_manifest: exact production solver contract required")
    solver_sha = canonical_sha256(solver)
    if solver_sha != evidence.get("solver_manifest_sha256") or solver_sha != config[
        "approved_solver_manifest_sha256"
    ]:
        raise ValueError("solver_manifest: hash/lock mismatch")
    return source_sha, solver_sha, range_source_sha


def _candidate_artifacts(
    value: Any, *, solver_sha: str, range_source_sha: str
) -> dict[str, dict[str, Any]]:
    raw = _require_mapping(value, label="candidate_policy_artifacts")
    if not raw:
        raise ValueError("candidate_policy_artifacts: non-empty map required")
    result: dict[str, dict[str, Any]] = {}
    for key, artifact in raw.items():
        digest = _require_sha256(key, label="candidate artifact map key")
        verified = verify_t3_bb_candidate_policy_artifact(artifact)
        if verified["artifact_sha256"] != digest:
            raise ValueError("candidate artifact map key/content mismatch")
        manifest = verified["model_manifest"]
        if manifest["solver_manifest_sha256"] != solver_sha:
            raise ValueError("candidate artifact solver binding mismatch")
        if manifest["range_builder_source_sha256"] != range_source_sha:
            raise ValueError("candidate artifact range source binding mismatch")
        result[digest] = verified
    return result


def _expected_bundle_keys(
    *, round_index: int, root_state: Mapping[str, Any]
) -> tuple[T3BBCheckpointKey, ...]:
    return tuple(
        sorted(
            T3BBCheckpointKey(
                round_index=round_index,
                root_id=root_id,
                root_commitment_sha256=root["root_commitment_sha256"],
                solver_seed=seed,
            )
            for seed in root_state["seeds"]
            for root_id, root in root_state["by_id"].items()
        )
    )


def _expected_query_bundle_keys(
    *,
    round_index: int,
    query_state: Mapping[str, Any],
    evaluation_seeds: Sequence[int],
) -> tuple[T3BBCheckpointKey, ...]:
    return tuple(
        sorted(
            T3BBCheckpointKey(
                round_index=round_index,
                root_id=query["query_id"],
                root_commitment_sha256=query["query_commitment_sha256"],
                solver_seed=seed,
            )
            for seed in evaluation_seeds
            for query in query_state["by_behavior_digest"].values()
        )
    )


def _evaluation_checkpoint_bundles(
    value: Any,
    *,
    bundle_root: Path,
    round_indices: set[int],
    root_state: Mapping[str, Any],
    source_sha: str,
    solver_sha: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[T3BBCheckpointKey, dict[str, dict[str, Fraction]]]],
]:
    raw = _require_mapping(value, label="evaluation_checkpoint_bundles")
    manifests: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[T3BBCheckpointKey, dict[str, dict[str, Fraction]]]] = {}
    seen_rounds: set[int] = set()
    for map_key, manifest in raw.items():
        digest = _require_sha256(map_key, label="checkpoint bundle map key")
        if not isinstance(manifest, Mapping):
            raise TypeError("checkpoint bundle manifest object required")
        round_index = manifest.get("round_index")
        if not _is_int(round_index) or round_index not in round_indices:
            raise ValueError("checkpoint bundle: unexpected round")
        if round_index in seen_rounds:
            raise ValueError("checkpoint bundle: exactly one bundle per round required")
        seen_rounds.add(round_index)
        expected_keys = _expected_bundle_keys(round_index=round_index, root_state=root_state)
        verified = verify_t3_bb_checkpoint_bundle(
            manifest, bundle_root=bundle_root, expected_keys=expected_keys
        )
        if verified["bundle_checkpoint_sha256"] != digest:
            raise ValueError("checkpoint bundle map key/content mismatch")
        if verified.get("promotion_eligible") is not False or verified.get(
            "exact_exploitability_computed"
        ) is not False:
            raise ValueError("checkpoint bundle must remain non-promoting/non-exact")
        for entry in verified["entries"]:
            if entry["solver_manifest_sha256"] != solver_sha:
                raise ValueError("checkpoint bundle solver binding mismatch")
            if entry["source_manifest_sha256"] != source_sha:
                raise ValueError("checkpoint bundle source binding mismatch")
            root = root_state["by_id"][entry["root_id"]]
            if entry["observation_digest"] != root["observation_digest"]:
                raise ValueError("checkpoint bundle observation binding mismatch")
        strategy_profiles = load_verified_t3_bb_checkpoint_strategy_profiles(
            verified, bundle_root=bundle_root, expected_keys=expected_keys
        )
        manifests[digest] = verified
        profiles[digest] = strategy_profiles
    if seen_rounds != round_indices:
        raise ValueError(
            "evaluation checkpoint bundles: exact state-round coverage required"
        )
    return manifests, profiles


def _normalized_checkpoint_distribution(
    profile: Mapping[str, Mapping[str, Fraction]], digest: str, *, label: str
) -> dict[str, Fraction]:
    if digest not in profile:
        raise ValueError(f"{label}: checkpoint strategy lacks information digest {digest}")
    raw = profile[digest]
    if not raw or any(value < 0 for value in raw.values()):
        raise ValueError(f"{label}: invalid checkpoint strategy distribution")
    total = sum(raw.values(), Fraction(0, 1))
    if total <= 0:
        raise ValueError(f"{label}: checkpoint strategy has no positive mass")
    return {key: Fraction(value) / total for key, value in sorted(raw.items())}


def _verify_candidate_query_tables(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[T3BBCheckpointKey, Mapping[str, Mapping[str, Fraction]]]],
    query_state: Mapping[str, Any],
    root_state: Mapping[str, Any],
) -> None:
    query_map = query_state["by_behavior_digest"]
    expected_behavior_digests = set(query_map)
    for artifact_sha, artifact in artifacts.items():
        manifest = artifact["model_manifest"]
        bundle_sha = manifest["checkpoint_sha256"]
        if bundle_sha not in bundles:
            raise ValueError(
                "candidate artifact checkpoint is not an embedded verified "
                "candidate-query bundle"
            )
        table = manifest["probabilities"]
        if set(table) != expected_behavior_digests:
            raise ValueError("candidate artifact: exact candidate-query coverage mismatch")
        round_profiles = profiles[bundle_sha]
        for behavior_digest, query in query_map.items():
            converted_digest = query["converted_observation_digest"]
            per_seed: list[dict[str, Fraction]] = []
            for seed in root_state["seeds"]:
                key = T3BBCheckpointKey(
                    round_index=bundles[bundle_sha]["round_index"],
                    root_id=query["query_id"],
                    root_commitment_sha256=query["query_commitment_sha256"],
                    solver_seed=seed,
                )
                if key not in round_profiles:
                    raise ValueError(
                        f"candidate query {behavior_digest}: missing query-bundle "
                        f"profile for seed {seed}"
                    )
                per_seed.append(
                    _normalized_checkpoint_distribution(
                        round_profiles[key],
                        converted_digest,
                        label=f"query {behavior_digest} seed {seed}",
                    )
                )
            support = set(per_seed[0])
            if any(set(item) != support for item in per_seed[1:]):
                raise ValueError("candidate query: cross-seed legal support mismatch")
            averaged = {
                action_id: sum(
                    (item[action_id] for item in per_seed), Fraction(0, 1)
                )
                / len(per_seed)
                for action_id in sorted(support)
            }
            expected_q32 = quantize_exact_distribution_q32(averaged)
            actual = _distribution(
                table[behavior_digest],
                label=f"candidate artifact {artifact_sha}.{behavior_digest}",
            )
            legal_support = set(
                query["behavior_information"]["legal_action_ids"]
            )
            if set(actual) != legal_support:
                raise ValueError(
                    f"candidate artifact {behavior_digest}: legal action support mismatch"
                )
            if actual != expected_q32:
                raise ValueError(
                    "candidate artifact table does not equal seed-aggregated Q32 "
                    f"checkpoint strategy for {behavior_digest}"
                )


def _range_artifacts(value: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    raw = _require_mapping(value, label="restricted_range_artifacts")
    artifacts: dict[str, dict[str, Any]] = {}
    audits: dict[str, Mapping[str, Any]] = {}
    for map_key, artifact in raw.items():
        digest = _require_sha256(map_key, label="range artifact map key")
        verified, audit = verify_restricted_range_evidence(artifact)
        if verified["artifact_sha256"] != digest:
            raise ValueError("range artifact map key/content mismatch")
        artifacts[digest] = verified
        audits[digest] = audit
    if not artifacts:
        raise ValueError("restricted_range_artifacts: non-empty map required")
    return artifacts, audits


def _query_range_is_bootstrap_only(artifact: Mapping[str, Any]) -> None:
    manifest = artifact.get("behavior_model_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("candidate-query range behavior manifest is missing")
    expected = {
        "model_type": BOOTSTRAP_MODEL_TYPE,
        "promotion_eligible": False,
        "fixed_point_bootstrap_only": True,
        "no_fallback": True,
        "t3_bb_route_included": False,
        "t3_bb_likelihood_binding_included": False,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted:
            raise ValueError(
                f"candidate-query range bootstrap manifest {field} mismatch"
            )


def _candidate_query_checkpoint_bundles(
    value: Any,
    *,
    bundle_root: Path,
    round_indices: set[int],
    query_state: Mapping[str, Any],
    root_state: Mapping[str, Any],
    source_sha: str,
    solver_sha: str,
    ranges: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[T3BBCheckpointKey, dict[str, dict[str, Fraction]]]],
]:
    raw = _require_mapping(value, label="candidate_query_checkpoint_bundles")
    manifests: dict[str, dict[str, Any]] = {}
    profiles: dict[
        str, dict[T3BBCheckpointKey, dict[str, dict[str, Fraction]]]
    ] = {}
    range_index: dict[
        tuple[str, str, int, int, str, str, str], Mapping[str, Any]
    ] = {}
    for artifact in ranges.values():
        key = (
            artifact["root_id"],
            artifact["root_commitment_sha256"],
            artifact["round_index"],
            artifact["solver_seed"],
            artifact["observation_digest"],
            artifact["range_content_sha256"],
            artifact["range_build_sha256"],
        )
        if key in range_index:
            raise ValueError("restricted range artifacts contain duplicate content binding")
        range_index[key] = artifact
    query_by_id = {
        query["query_id"]: query
        for query in query_state["by_behavior_digest"].values()
    }
    seen_rounds: set[int] = set()
    for map_key, manifest in raw.items():
        digest = _require_sha256(map_key, label="candidate-query bundle map key")
        if not isinstance(manifest, Mapping):
            raise TypeError("candidate-query bundle manifest object required")
        round_index = manifest.get("round_index")
        if not _is_int(round_index) or round_index not in round_indices:
            raise ValueError("candidate-query checkpoint bundle: unexpected round")
        if round_index in seen_rounds:
            raise ValueError(
                "candidate-query checkpoint bundle: exactly one bundle per round required"
            )
        seen_rounds.add(round_index)
        expected_keys = _expected_query_bundle_keys(
            round_index=round_index,
            query_state=query_state,
            evaluation_seeds=root_state["seeds"],
        )
        verified = verify_t3_bb_checkpoint_bundle(
            manifest, bundle_root=bundle_root, expected_keys=expected_keys
        )
        if verified["bundle_checkpoint_sha256"] != digest:
            raise ValueError("candidate-query bundle map key/content mismatch")
        if verified.get("promotion_eligible") is not False or verified.get(
            "exact_exploitability_computed"
        ) is not False:
            raise ValueError("candidate-query bundle must remain non-promoting/non-exact")
        for entry in verified["entries"]:
            if entry["solver_manifest_sha256"] != solver_sha:
                raise ValueError("candidate-query bundle solver binding mismatch")
            if entry["source_manifest_sha256"] != source_sha:
                raise ValueError("candidate-query bundle source binding mismatch")
            query = query_by_id.get(entry["root_id"])
            if query is None:
                raise ValueError("candidate-query bundle has an unknown query root")
            if (
                entry["root_commitment_sha256"]
                != query["query_commitment_sha256"]
                or entry["observation_digest"]
                != query["converted_observation_digest"]
            ):
                raise ValueError("candidate-query bundle query identity mismatch")
            range_key = (
                entry["root_id"],
                entry["root_commitment_sha256"],
                entry["round_index"],
                entry["solver_seed"],
                entry["observation_digest"],
                entry["range_content_sha256"],
                entry["range_build_sha256"],
            )
            range_artifact = range_index.get(range_key)
            if range_artifact is None:
                raise ValueError(
                    "candidate-query bundle entry lacks matching replayed range evidence"
                )
            _query_range_is_bootstrap_only(range_artifact)
        strategy_profiles = load_verified_t3_bb_checkpoint_strategy_profiles(
            verified, bundle_root=bundle_root, expected_keys=expected_keys
        )
        manifests[digest] = verified
        profiles[digest] = strategy_profiles
    if seen_rounds != round_indices:
        raise ValueError(
            "candidate-query checkpoint bundles: exact state-round coverage required"
        )
    return manifests, profiles


def _iteration_behavior_candidate_binding(
    artifact: Mapping[str, Any], *, candidate_artifact_sha: str
) -> None:
    manifest = artifact["behavior_model_manifest"]
    if not isinstance(manifest, Mapping) or manifest.get("no_fallback") is not True:
        raise ValueError("range behavior manifest must be a no-fallback model")
    if manifest.get("model_type") == ITERATION_MODEL_TYPE:
        if (
            manifest.get("fixed_point_iteration_only") is not True
            or manifest.get("fixed_point_converged") is not False
        ):
            raise ValueError("range behavior iteration flags are invalid")
        bound = manifest.get("t3_bb_candidate_policy_artifact_sha256")
    elif manifest.get("model_type") == CANDIDATE_MODEL_TYPE:
        bound = artifact.get("behavior_model_sha256")
    else:
        raise ValueError("range behavior model is not a fixed-point iteration candidate")
    if bound != candidate_artifact_sha:
        raise ValueError("range behavior candidate artifact binding mismatch")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("iteration range behavior must remain non-promoting")


def _validate_rows(
    rows_value: Any,
    *,
    root_state: Mapping[str, Any],
    candidate_artifacts: Mapping[str, Mapping[str, Any]],
    candidate_query_bundles: Mapping[str, Mapping[str, Any]],
    evaluation_bundles: Mapping[str, Mapping[str, Any]],
    evaluation_profiles: Mapping[
        str, Mapping[T3BBCheckpointKey, Mapping[str, Mapping[str, Fraction]]]
    ],
    ranges: Mapping[str, Mapping[str, Any]],
    range_audits: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("raw_iteration_rows: non-empty list required")
    rows: list[dict[str, Any]] = []
    bundle_entries_by_key: dict[
        str, dict[T3BBCheckpointKey, Mapping[str, Any]]
    ] = {}
    for bundle_sha, bundle in evaluation_bundles.items():
        bundle_entries_by_key[bundle_sha] = {
            T3BBCheckpointKey(
                round_index=entry["round_index"],
                root_id=entry["root_id"],
                root_commitment_sha256=entry["root_commitment_sha256"],
                solver_seed=entry["solver_seed"],
            ): entry
            for entry in bundle["entries"]
        }
    rounds: set[int] = set()
    seen: set[tuple[int, int, str]] = set()
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    prior_identity: tuple[int, int, str] | None = None
    for index, item in enumerate(rows_value):
        row = _require_mapping(item, label=f"raw_iteration_rows[{index}]")
        _exact_keys(row, _ROW_KEYS, label=f"raw_iteration_rows[{index}]")
        round_index = row.get("round_index")
        seed = row.get("solver_seed")
        root_id = row.get("root_id")
        if not _is_int(round_index) or round_index < 2:
            raise ValueError("row.round_index: transition round >=2 required")
        if seed not in root_state["seeds"] or root_id not in root_state["by_id"]:
            raise ValueError("row: root/seed outside locked evaluation manifest")
        identity = (round_index, seed, root_id)
        if identity in seen:
            raise ValueError("rows: duplicate round/seed/root")
        if prior_identity is not None and identity <= prior_identity:
            raise ValueError("rows: canonical round/seed/root order required")
        prior_identity = identity
        seen.add(identity)
        rounds.add(round_index)
        root = root_state["by_id"][root_id]
        for field in (
            "root_identity_commitment_sha256",
            "root_commitment_sha256",
            "stratum",
            "actor",
            "phase",
            "visible_joker_count",
            "observation_digest",
        ):
            if row.get(field) != root[field]:
                raise ValueError(f"row {identity}.{field}: root manifest mismatch")
        if row.get("exact_exploitability_computed") is not False:
            raise ValueError("row exact exploitability must remain false")
        previous_query_bundle_sha = _require_sha256(
            row.get("previous_candidate_query_checkpoint_bundle_sha256"),
            label="previous candidate-query bundle",
        )
        current_query_bundle_sha = _require_sha256(
            row.get("current_candidate_query_checkpoint_bundle_sha256"),
            label="current candidate-query bundle",
        )
        previous_evaluation_bundle_sha = _require_sha256(
            row.get("previous_evaluation_checkpoint_bundle_sha256"),
            label="previous evaluation bundle",
        )
        current_evaluation_bundle_sha = _require_sha256(
            row.get("current_evaluation_checkpoint_bundle_sha256"),
            label="current evaluation bundle",
        )
        if (
            previous_query_bundle_sha not in candidate_query_bundles
            or current_query_bundle_sha not in candidate_query_bundles
            or previous_evaluation_bundle_sha not in evaluation_bundles
            or current_evaluation_bundle_sha not in evaluation_bundles
        ):
            raise ValueError("row references an unverified query/evaluation bundle")
        for bundle, wanted_round, label in (
            (
                candidate_query_bundles[previous_query_bundle_sha],
                round_index - 1,
                "previous candidate-query",
            ),
            (
                candidate_query_bundles[current_query_bundle_sha],
                round_index,
                "current candidate-query",
            ),
            (
                evaluation_bundles[previous_evaluation_bundle_sha],
                round_index - 1,
                "previous evaluation",
            ),
            (
                evaluation_bundles[current_evaluation_bundle_sha],
                round_index,
                "current evaluation",
            ),
        ):
            if bundle["round_index"] != wanted_round:
                raise ValueError(f"row {label} bundle round chain mismatch")
        # This explicit inequality rejects the old impossible
        # C_r -> B_r -> R_r(C_r) -> C_r content-hash cycle.  Candidate policy
        # tables come from Q_r; evaluation policies/ranges come from B_r.
        if (
            previous_query_bundle_sha == previous_evaluation_bundle_sha
            or current_query_bundle_sha == current_evaluation_bundle_sha
        ):
            raise ValueError(
                "candidate-query and evaluation bundles must be causally separate"
            )
        previous_candidate_sha = _require_sha256(
            row.get("previous_candidate_policy_artifact_sha256"), label="previous candidate"
        )
        current_candidate_sha = _require_sha256(
            row.get("current_candidate_policy_artifact_sha256"), label="current candidate"
        )
        if previous_candidate_sha not in candidate_artifacts or current_candidate_sha not in candidate_artifacts:
            raise ValueError("row references unverified candidate artifact")
        if candidate_artifacts[previous_candidate_sha]["model_manifest"]["checkpoint_sha256"] != previous_query_bundle_sha:
            raise ValueError("row previous candidate/query-checkpoint chain mismatch")
        if candidate_artifacts[current_candidate_sha]["model_manifest"]["checkpoint_sha256"] != current_query_bundle_sha:
            raise ValueError("row current candidate/query-checkpoint chain mismatch")
        key_previous = T3BBCheckpointKey(
            round_index=round_index - 1,
            root_id=root_id,
            root_commitment_sha256=root["root_commitment_sha256"],
            solver_seed=seed,
        )
        key_current = T3BBCheckpointKey(
            round_index=round_index,
            root_id=root_id,
            root_commitment_sha256=root["root_commitment_sha256"],
            solver_seed=seed,
        )
        previous_policy = _normalized_checkpoint_distribution(
            evaluation_profiles[previous_evaluation_bundle_sha][key_previous],
            root["observation_digest"],
            label=f"row {identity} previous root policy",
        )
        current_policy = _normalized_checkpoint_distribution(
            evaluation_profiles[current_evaluation_bundle_sha][key_current],
            root["observation_digest"],
            label=f"row {identity} current root policy",
        )
        claimed_previous = _distribution(
            row.get("previous_policy_distribution"), label=f"row {identity}.previous_policy"
        )
        claimed_current = _distribution(
            row.get("current_policy_distribution"), label=f"row {identity}.current_policy"
        )
        if claimed_previous != previous_policy or claimed_current != current_policy:
            raise ValueError("row policy claim does not match fresh checkpoint strategy")
        expected_legal_support = set(root_state["legal_by_id"][root_id])
        if set(previous_policy) != expected_legal_support or set(
            current_policy
        ) != expected_legal_support:
            raise ValueError("row checkpoint policy legal action support mismatch")

        previous_range_sha = _require_sha256(
            row.get("previous_range_evidence_artifact_sha256"), label="previous range"
        )
        current_range_sha = _require_sha256(
            row.get("current_range_evidence_artifact_sha256"), label="current range"
        )
        if previous_range_sha not in ranges or current_range_sha not in ranges:
            raise ValueError("row references unverified restricted range artifact")
        previous_range = ranges[previous_range_sha]
        current_range = ranges[current_range_sha]
        for range_artifact, expected_round, candidate_sha, bundle_sha, key in (
            (
                previous_range,
                round_index - 1,
                previous_candidate_sha,
                previous_evaluation_bundle_sha,
                key_previous,
            ),
            (
                current_range,
                round_index,
                current_candidate_sha,
                current_evaluation_bundle_sha,
                key_current,
            ),
        ):
            if (
                range_artifact["root_id"] != root_id
                or range_artifact["root_commitment_sha256"] != root["root_commitment_sha256"]
                or range_artifact["round_index"] != expected_round
                or range_artifact["solver_seed"] != seed
                or range_artifact["observation_digest"] != root["observation_digest"]
            ):
                raise ValueError("restricted range root/round/seed/observation binding mismatch")
            _iteration_behavior_candidate_binding(
                range_artifact, candidate_artifact_sha=candidate_sha
            )
            bundle_entry = bundle_entries_by_key[bundle_sha][key]
            if (
                bundle_entry["range_content_sha256"] != range_artifact["range_content_sha256"]
                or bundle_entry["range_build_sha256"] != range_artifact["range_build_sha256"]
            ):
                raise ValueError("checkpoint bundle/range content binding mismatch")

        if root["actor"] == "btn":
            previous_posterior = _distribution(
                row.get("previous_btn_posterior_weights"),
                label=f"row {identity}.previous_posterior",
            )
            current_posterior = _distribution(
                row.get("current_btn_posterior_weights"),
                label=f"row {identity}.current_posterior",
            )
            audited_previous = _distribution(
                range_audits[previous_range_sha]["posterior_weights"],
                label=f"range {previous_range_sha}.posterior",
            )
            audited_current = _distribution(
                range_audits[current_range_sha]["posterior_weights"],
                label=f"range {current_range_sha}.posterior",
            )
            if previous_posterior != audited_previous or current_posterior != audited_current:
                raise ValueError("BTN posterior claim does not match replayed restricted range")
        else:
            if row.get("previous_btn_posterior_weights") is not None or row.get(
                "current_btn_posterior_weights"
            ) is not None:
                raise ValueError("BB row must not claim a BTN posterior")
            previous_posterior = None
            current_posterior = None
        normalized = copy.deepcopy(dict(row))
        normalized["_previous_policy"] = previous_policy
        normalized["_current_policy"] = current_policy
        normalized["_previous_posterior"] = previous_posterior
        normalized["_current_posterior"] = current_posterior
        rows.append(normalized)
        by_key[identity] = normalized

    ordered_rounds = tuple(sorted(rounds))
    if not ordered_rounds or ordered_rounds != tuple(range(ordered_rounds[0], ordered_rounds[-1] + 1)):
        raise ValueError("rows: contiguous transition rounds required")
    expected = {
        (round_index, seed, root_id)
        for round_index in ordered_rounds
        for seed in root_state["seeds"]
        for root_id in root_state["by_id"]
    }
    if seen != expected:
        raise ValueError("rows: exact round x seed x root Cartesian coverage required")
    # A state has one Q/C pair and one separate R/B pair shared across all
    # seeds/roots.
    for round_index in ordered_rounds:
        current_pairs = {
            (
                row["current_candidate_query_checkpoint_bundle_sha256"],
                row["current_evaluation_checkpoint_bundle_sha256"],
                row["current_candidate_policy_artifact_sha256"],
            )
            for row in rows
            if row["round_index"] == round_index
        }
        previous_pairs = {
            (
                row["previous_candidate_query_checkpoint_bundle_sha256"],
                row["previous_evaluation_checkpoint_bundle_sha256"],
                row["previous_candidate_policy_artifact_sha256"],
            )
            for row in rows
            if row["round_index"] == round_index
        }
        if len(current_pairs) != 1 or len(previous_pairs) != 1:
            raise ValueError("rows: one round-wide Q/C and R/B state chain required")
    for round_index in ordered_rounds[1:]:
        for seed in root_state["seeds"]:
            for root_id in root_state["by_id"]:
                prior = by_key[(round_index - 1, seed, root_id)]
                current = by_key[(round_index, seed, root_id)]
                if (
                    current["previous_candidate_query_checkpoint_bundle_sha256"]
                    != prior["current_candidate_query_checkpoint_bundle_sha256"]
                    or current["previous_evaluation_checkpoint_bundle_sha256"]
                    != prior["current_evaluation_checkpoint_bundle_sha256"]
                    or current["previous_candidate_policy_artifact_sha256"]
                    != prior["current_candidate_policy_artifact_sha256"]
                    or current["previous_range_evidence_artifact_sha256"]
                    != prior["current_range_evidence_artifact_sha256"]
                    or current["_previous_policy"] != prior["_current_policy"]
                    or current["_previous_posterior"] != prior["_current_posterior"]
                ):
                    raise ValueError("rows: previous-to-current round chain mismatch")
    return rows, ordered_rounds


def _derive_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    rounds: Sequence[int],
    root_state: Mapping[str, Any],
) -> dict[str, Any]:
    by_round_root: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    round_metrics: list[dict[str, Any]] = []
    for row in rows:
        by_round_root[(row["round_index"], row["root_id"])].append(row)
    converged_flags: list[bool] = []
    for round_index in rounds:
        round_rows = [row for row in rows if row["round_index"] == round_index]
        within_policy = max(
            (_tv(row["_previous_policy"], row["_current_policy"]) for row in round_rows),
            default=Fraction(0, 1),
        )
        btn_rows = [row for row in round_rows if row["actor"] == "btn"]
        within_posterior = max(
            (
                _tv(row["_previous_posterior"], row["_current_posterior"])
                for row in btn_rows
            ),
            default=Fraction(0, 1),
        )
        cross_policy = Fraction(0, 1)
        cross_posterior = Fraction(0, 1)
        pair_count = 0
        posterior_pair_count = 0
        for root_id, root in root_state["by_id"].items():
            root_rows = sorted(
                by_round_root[(round_index, root_id)], key=lambda row: row["solver_seed"]
            )
            for left, right in itertools.combinations(root_rows, 2):
                pair_count += 1
                cross_policy = max(
                    cross_policy,
                    _tv(left["_current_policy"], right["_current_policy"]),
                )
                if root["actor"] == "btn":
                    posterior_pair_count += 1
                    cross_posterior = max(
                        cross_posterior,
                        _tv(left["_current_posterior"], right["_current_posterior"]),
                    )
        converged = (
            within_policy <= MAX_POLICY_TV
            and within_posterior <= MAX_BTN_POSTERIOR_TV
            and cross_policy <= MAX_CROSS_SEED_POLICY_TV
            and cross_posterior <= MAX_CROSS_SEED_BTN_POSTERIOR_TV
        )
        converged_flags.append(converged)
        round_metrics.append(
            {
                "round_index": round_index,
                "max_previous_to_current_policy_tv": _encode_fraction(within_policy),
                "max_previous_to_current_btn_posterior_tv": _encode_fraction(within_posterior),
                "max_same_root_cross_seed_policy_tv": _encode_fraction(cross_policy),
                "max_same_root_cross_seed_btn_posterior_tv": _encode_fraction(cross_posterior),
                "cross_seed_policy_pair_count": pair_count,
                "cross_seed_btn_posterior_pair_count": posterior_pair_count,
                "converged": converged,
            }
        )
    trailing = 0
    for converged in reversed(converged_flags):
        if not converged:
            break
        trailing += 1
    return {
        "production_minima": _production_thresholds(),
        "independent_seeds": list(root_state["seeds"]),
        "root_counts_by_stratum": dict(root_state["counts"]),
        "transition_rounds": list(rounds),
        "round_metrics": round_metrics,
        "final_consecutive_converged_rounds": trailing,
        "all_six_strata_exactly_covered": set(root_state["counts"]) == set(REQUIRED_STRATA),
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
    }


def _production_minima_failures(
    *,
    root_state: Mapping[str, Any],
    rounds: Sequence[int],
    trailing_converged_rounds: int,
    production_claim: bool,
) -> list[str]:
    failures: list[str] = []
    if len(root_state["seeds"]) < MIN_INDEPENDENT_SEEDS:
        failures.append(
            "insufficient independent seeds: "
            f"{len(root_state['seeds'])} < {MIN_INDEPENDENT_SEEDS}"
        )
    deficient_strata = {
        stratum: count
        for stratum, count in root_state["counts"].items()
        if count < MIN_ROOTS_PER_STRATUM
    }
    if deficient_strata:
        failures.append(
            "insufficient roots per stratum: "
            f"required={MIN_ROOTS_PER_STRATUM}, actual={deficient_strata}"
        )
    if len(rounds) < MIN_CONSECUTIVE_CONVERGED_ROUNDS:
        failures.append(
            "insufficient transition rounds: "
            f"{len(rounds)} < {MIN_CONSECUTIVE_CONVERGED_ROUNDS}"
        )
    if trailing_converged_rounds < MIN_CONSECUTIVE_CONVERGED_ROUNDS:
        failures.append(
            "insufficient trailing production-converged rounds: "
            f"{trailing_converged_rounds} < {MIN_CONSECUTIVE_CONVERGED_ROUNDS}"
        )
    if production_claim is False:
        failures.append("production promotion claim is false")
    return failures


def _clean_metrics_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: copy.deepcopy(value) for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _failure_result(
    *,
    evidence: Any,
    config: Any,
    failures: Sequence[str],
    derived_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def hash_claim(value: Any) -> str | None:
        return value if _is_sha256(value) else None

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "status": FAIL_STATUS,
        "passed": False,
        "promotion_eligible": False,
        "production_minima_enforced": True,
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
        "evidence_sha256": hash_claim(evidence.get("artifact_sha256")) if isinstance(evidence, Mapping) else None,
        "gate_config_sha256": hash_claim(config.get("gate_config_sha256")) if isinstance(config, Mapping) else None,
        "candidate_policy_artifact_sha256": None,
        "candidate_policy_checkpoint_sha256": None,
        "final_evaluation_checkpoint_bundle_sha256": None,
        "solver_manifest_sha256": hash_claim(evidence.get("solver_manifest_sha256")) if isinstance(evidence, Mapping) else None,
        "range_builder_source_sha256": hash_claim(config.get("approved_range_builder_source_sha256")) if isinstance(config, Mapping) else None,
        "derived_metrics": (
            copy.deepcopy(dict(derived_metrics))
            if derived_metrics is not None
            else None
        ),
        "failures": list(failures),
    }
    result["gate_result_sha256"] = self_hash(result, "gate_result_sha256")
    return result


def validate_t3_bb_fixed_point_evidence(
    evidence: Any,
    *,
    config: Any,
    checkpoint_bundle_root: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
    _check_published_summary: bool = True,
) -> dict[str, Any]:
    """Fresh-replay production evidence and return a deterministic gate result."""

    try:
        locked = verify_locked_t3_bb_fixed_point_gate_config(config)
        raw = _require_mapping(evidence, label="evidence")
        _exact_keys(raw, _EVIDENCE_KEYS, label="evidence")
        expected = {
            "schema": EVIDENCE_SCHEMA,
            "gate_id": GATE_ID,
            "scope": SCOPE,
            "evidence_kind": EVIDENCE_KIND,
            "gate_config_sha256": locked["gate_config_sha256"],
            "exact_exploitability_computed": False,
            "strategic_strength_evaluated": False,
            "requires_independent_strength_gate": True,
            "cold_start_zero_drift_is_strength_evidence": False,
            "candidate_policy_method": SOLVER_METHOD,
        }
        for field, wanted in expected.items():
            if raw.get(field) != wanted:
                raise ValueError(f"evidence.{field}: contract mismatch")
        production_claim = raw.get("production_promotion_claim")
        if not isinstance(production_claim, bool):
            raise TypeError("evidence.production_promotion_claim: boolean required")
        if self_hash(raw, "artifact_sha256") != raw.get("artifact_sha256"):
            raise ValueError("evidence.artifact_sha256: self-hash mismatch")
        source_sha, solver_sha, range_source_sha = _validate_source_and_solver(
            raw, config=locked, workspace_root=Path(workspace_root)
        )
        root_state = _validate_root_manifest(
            raw.get("root_manifest"), enforce_production_minima=False
        )
        if root_state["raw"]["manifest_sha256"] != locked["approved_root_manifest_sha256"]:
            raise ValueError("root_manifest: locked hash mismatch")
        if locked["selected_promotion_seed"] not in root_state["seeds"]:
            raise ValueError("selected promotion seed is outside locked evaluation seeds")
        query_state = _validate_candidate_query_manifest(
            raw.get("candidate_query_manifest"), root_state=root_state
        )
        if query_state["raw"]["manifest_sha256"] != locked[
            "approved_candidate_query_manifest_sha256"
        ]:
            raise ValueError("candidate_query_manifest: locked hash mismatch")
        partition_hashes = _validate_partitions(
            raw.get("excluded_partitions"),
            root_state=root_state,
            evaluation_seeds=root_state["seeds"],
        )
        if partition_hashes != locked["approved_excluded_partition_sha256"]:
            raise ValueError("excluded partitions: locked hashes mismatch")
        candidate_artifacts = _candidate_artifacts(
            raw.get("candidate_policy_artifacts"),
            solver_sha=solver_sha,
            range_source_sha=range_source_sha,
        )
        raw_rows = raw.get("raw_iteration_rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("raw_iteration_rows required")
        transition_rounds = {
            row.get("round_index")
            for row in raw_rows
            if isinstance(row, Mapping) and _is_int(row.get("round_index"))
        }
        if not transition_rounds:
            raise ValueError("raw_iteration_rows have no valid rounds")
        bundle_rounds = set(transition_rounds) | {min(transition_rounds) - 1}
        range_artifacts, range_audits = _range_artifacts(
            raw.get("restricted_range_artifacts")
        )
        evaluation_bundles, evaluation_profiles = _evaluation_checkpoint_bundles(
            raw.get("evaluation_checkpoint_bundles"),
            bundle_root=Path(checkpoint_bundle_root),
            round_indices=bundle_rounds,
            root_state=root_state,
            source_sha=source_sha,
            solver_sha=solver_sha,
        )
        query_bundles, query_profiles = _candidate_query_checkpoint_bundles(
            raw.get("candidate_query_checkpoint_bundles"),
            bundle_root=Path(checkpoint_bundle_root),
            round_indices=bundle_rounds,
            query_state=query_state,
            root_state=root_state,
            source_sha=source_sha,
            solver_sha=solver_sha,
            ranges=range_artifacts,
        )
        _verify_candidate_query_tables(
            artifacts=candidate_artifacts,
            bundles=query_bundles,
            profiles=query_profiles,
            query_state=query_state,
            root_state=root_state,
        )
        rows, rounds = _validate_rows(
            raw_rows,
            root_state=root_state,
            candidate_artifacts=candidate_artifacts,
            candidate_query_bundles=query_bundles,
            evaluation_bundles=evaluation_bundles,
            evaluation_profiles=evaluation_profiles,
            ranges=range_artifacts,
            range_audits=range_audits,
        )
        metrics = _derive_metrics(rows, rounds=rounds, root_state=root_state)
        if _check_published_summary and raw.get("published_summary") != metrics:
            raise ValueError("published_summary does not match exact re-derived metrics")
        trailing = metrics["final_consecutive_converged_rounds"]
        minima_failures = _production_minima_failures(
            root_state=root_state,
            rounds=rounds,
            trailing_converged_rounds=trailing,
            production_claim=production_claim,
        )
        if minima_failures:
            return _failure_result(
                evidence=raw,
                config=locked,
                failures=minima_failures,
                derived_metrics=metrics,
            )
        final_round = rounds[-1]
        final_rows = [row for row in rows if row["round_index"] == final_round]
        candidate_shas = {row["current_candidate_policy_artifact_sha256"] for row in final_rows}
        checkpoint_shas = {
            row["current_candidate_query_checkpoint_bundle_sha256"]
            for row in final_rows
        }
        evaluation_shas = {
            row["current_evaluation_checkpoint_bundle_sha256"]
            for row in final_rows
        }
        if (
            len(candidate_shas) != 1
            or len(checkpoint_shas) != 1
            or len(evaluation_shas) != 1
        ):
            raise ValueError("final round has no unique Q/C and R/B state")
        candidate_sha = next(iter(candidate_shas))
        checkpoint_sha = next(iter(checkpoint_shas))
        # selected_promotion_seed is a pre-locked audit identity.  The promoted
        # table itself is the Q32 aggregate of every independent seed.
        if locked["selected_promotion_seed"] not in {
            row["solver_seed"] for row in final_rows
        }:
            raise ValueError("selected promotion seed missing from final coverage")
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "gate_id": GATE_ID,
            "scope": SCOPE,
            "status": PASS_STATUS,
            "passed": True,
            "promotion_eligible": True,
            "production_minima_enforced": True,
            "exact_exploitability_computed": False,
            "strategic_strength_evaluated": False,
            "requires_independent_strength_gate": True,
            "cold_start_zero_drift_is_strength_evidence": False,
            "evidence_sha256": raw["artifact_sha256"],
            "gate_config_sha256": locked["gate_config_sha256"],
            "candidate_policy_artifact_sha256": candidate_sha,
            "candidate_policy_checkpoint_sha256": checkpoint_sha,
            "final_evaluation_checkpoint_bundle_sha256": next(
                iter(evaluation_shas)
            ),
            "solver_manifest_sha256": solver_sha,
            "range_builder_source_sha256": range_source_sha,
            "derived_metrics": metrics,
            "failures": [],
        }
        result["gate_result_sha256"] = self_hash(result, "gate_result_sha256")
        return result
    except (OSError, TypeError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        return _failure_result(evidence=evidence, config=config, failures=[str(exc)])


def derive_t3_bb_fixed_point_metrics(
    evidence: Any,
    *,
    config: Any,
    checkpoint_bundle_root: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return metrics only after the same full replay used by promotion."""

    result = validate_t3_bb_fixed_point_evidence(
        evidence,
        config=config,
        checkpoint_bundle_root=checkpoint_bundle_root,
        workspace_root=workspace_root,
        _check_published_summary=False,
    )
    if result["derived_metrics"] is None:
        raise ValueError("cannot derive metrics: " + "; ".join(result["failures"]))
    return copy.deepcopy(result["derived_metrics"])


def build_t3_bb_fixed_point_evidence(
    *,
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    solver_manifest: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    candidate_query_manifest: Mapping[str, Any],
    excluded_partitions: Mapping[str, Mapping[str, Any]],
    candidate_query_checkpoint_bundles: Mapping[str, Mapping[str, Any]],
    evaluation_checkpoint_bundles: Mapping[str, Mapping[str, Any]],
    candidate_policy_artifacts: Mapping[str, Mapping[str, Any]],
    restricted_range_artifacts: Mapping[str, Mapping[str, Any]],
    raw_iteration_rows: Sequence[Mapping[str, Any]],
    checkpoint_bundle_root: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a production artifact, derive its summary, then replay it again.

    There is intentionally no ``diagnostic`` or threshold argument.  If the
    supplied assets do not meet the hard production contract this function
    raises and cannot manufacture a promotion-shaped result.
    """

    locked = verify_locked_t3_bb_fixed_point_gate_config(config)
    source = copy.deepcopy(dict(source_manifest))
    solver = copy.deepcopy(dict(solver_manifest))
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "evidence_kind": EVIDENCE_KIND,
        "production_promotion_claim": True,
        "gate_config_sha256": locked["gate_config_sha256"],
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
        "candidate_policy_method": SOLVER_METHOD,
        "source_manifest": source,
        "source_manifest_sha256": canonical_sha256(source),
        "solver_manifest": solver,
        "solver_manifest_sha256": canonical_sha256(solver),
        "root_manifest": copy.deepcopy(dict(root_manifest)),
        "candidate_query_manifest": copy.deepcopy(dict(candidate_query_manifest)),
        "excluded_partitions": {
            key: copy.deepcopy(dict(value))
            for key, value in excluded_partitions.items()
        },
        "candidate_query_checkpoint_bundles": {
            key: copy.deepcopy(dict(value))
            for key, value in candidate_query_checkpoint_bundles.items()
        },
        "evaluation_checkpoint_bundles": {
            key: copy.deepcopy(dict(value))
            for key, value in evaluation_checkpoint_bundles.items()
        },
        "candidate_policy_artifacts": {
            key: copy.deepcopy(dict(value))
            for key, value in candidate_policy_artifacts.items()
        },
        "restricted_range_artifacts": {
            key: copy.deepcopy(dict(value))
            for key, value in restricted_range_artifacts.items()
        },
        "raw_iteration_rows": [copy.deepcopy(dict(row)) for row in raw_iteration_rows],
        "published_summary": {},
    }
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")
    evidence["published_summary"] = derive_t3_bb_fixed_point_metrics(
        evidence,
        config=locked,
        checkpoint_bundle_root=checkpoint_bundle_root,
        workspace_root=workspace_root,
    )
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")
    result = validate_t3_bb_fixed_point_evidence(
        evidence,
        config=locked,
        checkpoint_bundle_root=checkpoint_bundle_root,
        workspace_root=workspace_root,
    )
    if result.get("passed") is not True or result.get("promotion_eligible") is not True:
        raise ValueError("production evidence did not pass: " + "; ".join(result["failures"]))
    return copy.deepcopy(evidence), result


def verify_t3_bb_fixed_point_gate_result(
    evidence: Any,
    *,
    config: Any,
    gate_result: Any,
    checkpoint_bundle_root: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
) -> dict[str, Any]:
    raw = _require_mapping(gate_result, label="gate_result")
    if raw.get("gate_result_sha256") != self_hash(raw, "gate_result_sha256"):
        raise ValueError("gate_result: self-hash mismatch")
    expected = validate_t3_bb_fixed_point_evidence(
        evidence,
        config=config,
        checkpoint_bundle_root=checkpoint_bundle_root,
        workspace_root=workspace_root,
    )
    if dict(raw) != expected:
        raise ValueError("gate_result: does not match fresh replay")
    return copy.deepcopy(dict(raw))


def build_t3_bb_likelihood_binding(
    evidence: Any,
    *,
    config: Any,
    gate_result: Any,
    checkpoint_bundle_root: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Emit the existing M3 binding only from a fully replayed v2 pass."""

    verified = verify_t3_bb_fixed_point_gate_result(
        evidence,
        config=config,
        gate_result=gate_result,
        checkpoint_bundle_root=checkpoint_bundle_root,
        workspace_root=workspace_root,
    )
    if (
        verified.get("passed") is not True
        or verified.get("promotion_eligible") is not True
        or verified.get("production_minima_enforced") is not True
        or verified.get("exact_exploitability_computed") is not False
        or verified.get("strategic_strength_evaluated") is not False
        or verified.get("requires_independent_strength_gate") is not True
        or verified.get("cold_start_zero_drift_is_strength_evidence") is not False
    ):
        raise ValueError("v2 fixed-point gate did not pass production promotion")
    binding: dict[str, Any] = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": verified[
            "candidate_policy_artifact_sha256"
        ],
        "candidate_policy_checkpoint_sha256": verified[
            "candidate_policy_checkpoint_sha256"
        ],
        "fixed_point_evidence_sha256": verified["evidence_sha256"],
        "fixed_point_gate_result_sha256": verified["gate_result_sha256"],
        "fixed_point_config_sha256": verified["gate_config_sha256"],
        "range_builder_source_sha256": verified["range_builder_source_sha256"],
        "solver_manifest_sha256": verified["solver_manifest_sha256"],
    }
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")
    return verify_t3_bb_likelihood_binding(
        binding,
        solver_manifest_sha256=verified["solver_manifest_sha256"],
        range_builder_source_sha256=verified["range_builder_source_sha256"],
    )


__all__ = [
    "CONFIG_SCHEMA",
    "EVIDENCE_KIND",
    "EVIDENCE_SCHEMA",
    "GATE_ID",
    "MAX_BTN_POSTERIOR_TV",
    "MAX_CROSS_SEED_BTN_POSTERIOR_TV",
    "MAX_CROSS_SEED_POLICY_TV",
    "MAX_POLICY_TV",
    "MIN_CONSECUTIVE_CONVERGED_ROUNDS",
    "MIN_INDEPENDENT_SEEDS",
    "MIN_ROOTS_PER_STRATUM",
    "PARTITION_SCHEMA",
    "QUERY_MANIFEST_SCHEMA",
    "RESULT_SCHEMA",
    "ROOT_MANIFEST_SCHEMA",
    "SCOPE",
    "build_candidate_query_manifest",
    "build_candidate_query_record",
    "build_excluded_partition",
    "build_locked_t3_bb_fixed_point_gate_config",
    "build_root_manifest",
    "build_root_record",
    "build_t3_bb_fixed_point_evidence",
    "build_t3_bb_likelihood_binding",
    "canonical_json",
    "canonical_sha256",
    "derive_t3_bb_fixed_point_metrics",
    "encode_distribution",
    "quantize_exact_distribution_q32",
    "query_commitment_sha256",
    "reconstruct_behavior_information",
    "reconstruct_infoset_key",
    "root_commitment_sha256",
    "root_identity_commitment_sha256",
    "self_hash",
    "validate_t3_bb_fixed_point_evidence",
    "verify_locked_t3_bb_fixed_point_gate_config",
    "verify_t3_bb_fixed_point_gate_result",
    "visible_joker_count",
]
