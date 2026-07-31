"""Restricted, independently replayable full-card posterior range evidence.

The public fixed-point gate must not accept posterior weights that are merely
typed into a row.  This module serializes the hidden physical particles, exact
weights, public observation, and range manifests into a restricted artifact.
Fresh verification reconstructs :class:`InfoSetKey`, :class:`JointParticle`,
and :class:`FullCardRange`, then delegates to ``verify_full_card_range`` so the
deck partition, private recall, commitments, posterior mass, ESS, behavior
audit, and content/build hashes are all re-derived.

Artifacts contain opponent private recall and undealt cards.  They belong in a
restricted evidence directory and must never be supplied to a policy input or
public runtime response.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from ai.tutor.promotion_gate_m3_full_card_strength import (
    canonical_json,
    canonical_sha256,
)
from ai.tutor.t3_hu_full_card_range import (
    FullCardRange,
    verify_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall


EVIDENCE_SCHEMA = "ofc_t3_bb_restricted_range_evidence/v2"
_SHA256_LENGTH = 64
_TOP_KEYS = frozenset(
    {
        "schema",
        "restricted_hidden_information",
        "root_id",
        "root_commitment_sha256",
        "round_index",
        "solver_seed",
        "observation",
        "observation_digest",
        "behavior_model_id",
        "behavior_model_sha256",
        "epsilon",
        "opponent_public_evidence_normalizer_exact",
        "effective_sample_size_exact",
        "range_sha256",
        "range_content_sha256",
        "range_build_sha256",
        "particle_commitments",
        "particles",
        "content_manifest",
        "build_manifest",
        "behavior_model_manifest",
        "artifact_sha256",
    }
)
_PARTICLE_KEYS = frozenset(
    {"commitment", "bb_recall", "btn_recall", "undealt_cards", "weight"}
)
_RECALL_KEYS = frozenset({"dealt_by_turn", "discards_by_turn"})


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} exact fields mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _canonical_fraction(value: Any, *, label: str, positive: bool) -> Fraction:
    if not isinstance(value, str) or value.strip() != value or "/" not in value:
        raise TypeError(f"{label} must be a canonical rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} is not a rational") from exc
    if value != f"{result.numerator}/{result.denominator}":
        raise ValueError(f"{label} is not reduced canonical rational data")
    if result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _recall_payload(recall: PrivateRecall) -> dict[str, Any]:
    return recall.to_canonical_dict()


def _parse_recall(value: Any, *, label: str) -> PrivateRecall:
    raw = _require_mapping(value, label=label)
    _exact_keys(raw, _RECALL_KEYS, label=label)
    dealt_raw = raw["dealt_by_turn"]
    discards_raw = raw["discards_by_turn"]
    if not isinstance(dealt_raw, list) or not isinstance(discards_raw, list):
        raise TypeError(f"{label} recall lists are required")
    dealt: list[tuple[int, tuple[str, ...]]] = []
    for index, item in enumerate(dealt_raw):
        row = _require_mapping(item, label=f"{label}.dealt_by_turn[{index}]")
        if set(row) != {"turn", "cards"} or not isinstance(row["cards"], list):
            raise ValueError(f"{label}.dealt_by_turn[{index}] schema mismatch")
        dealt.append((row["turn"], tuple(row["cards"])))
    discards: list[tuple[int, str]] = []
    for index, item in enumerate(discards_raw):
        row = _require_mapping(item, label=f"{label}.discards_by_turn[{index}]")
        if set(row) != {"turn", "card"}:
            raise ValueError(f"{label}.discards_by_turn[{index}] schema mismatch")
        discards.append((row["turn"], row["card"]))
    recall = PrivateRecall(tuple(dealt), tuple(discards))
    if recall.to_canonical_dict() != dict(raw):
        raise ValueError(f"{label} is not canonical")
    return recall


def _parse_observation(value: Any) -> InfoSetKey:
    raw = _require_mapping(value, label="observation")
    expected = {
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
    if set(raw) != expected:
        raise ValueError("observation exact fields mismatch")
    boards: dict[str, tuple[tuple[str, ...], ...]] = {}
    for actor in ("bb", "btn"):
        board = _require_mapping(raw[f"board_{actor}"], label=f"board_{actor}")
        if set(board) != {"top", "middle", "bottom"}:
            raise ValueError(f"board_{actor} row fields mismatch")
        boards[actor] = tuple(
            tuple(board[row]) for row in ("top", "middle", "bottom")
        )
    history_raw = raw["public_action_history"]
    if not isinstance(history_raw, list):
        raise TypeError("observation public_action_history must be a list")
    history = []
    for index, item in enumerate(history_raw):
        row = _require_mapping(item, label=f"public_action_history[{index}]")
        if set(row) != {"turn", "actor", "placements"} or not isinstance(
            row["placements"], list
        ):
            raise ValueError(f"public_action_history[{index}] schema mismatch")
        placements = []
        for placement in row["placements"]:
            if not isinstance(placement, list) or len(placement) != 2:
                raise ValueError("public placement must be [card,row]")
            placements.append((placement[0], placement[1]))
        history.append((row["turn"], row["actor"], tuple(placements)))
    observation = InfoSetKey(
        contract_version=raw["contract_version"],
        actor=raw["actor"],
        turn=raw["turn"],
        phase=raw["phase"],
        board_bb=boards["bb"],
        board_btn=boards["btn"],
        public_action_history=tuple(history),
        own_recall=_parse_recall(raw["own_recall"], label="observation.own_recall"),
        current_draw=tuple(raw["current_draw"]),
        fantasy_state=raw["fantasy_state"],
    )
    if observation.to_canonical_dict() != dict(raw):
        raise ValueError("observation is not canonical")
    return observation


def build_restricted_range_evidence(
    observation: InfoSetKey,
    full_range: FullCardRange,
    *,
    root_id: str,
    root_commitment_sha256: str,
    round_index: int,
    solver_seed: int,
) -> dict[str, Any]:
    """Serialize one already-built range after independent in-memory replay."""

    if not isinstance(root_id, str) or not root_id:
        raise ValueError("root_id must be non-empty")
    root_commitment = _require_sha256(
        root_commitment_sha256, label="root_commitment_sha256"
    )
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
    ):
        raise ValueError("round_index must be a nonnegative integer")
    if isinstance(solver_seed, bool) or not isinstance(solver_seed, int):
        raise TypeError("solver_seed must be an integer")
    verify_full_card_range(observation, full_range)
    particles = []
    for commitment, particle in zip(
        full_range.particle_commitments, full_range.particles
    ):
        particles.append(
            {
                "commitment": commitment,
                "bb_recall": _recall_payload(particle.bb_recall),
                "btn_recall": _recall_payload(particle.btn_recall),
                "undealt_cards": list(particle.undealt_cards),
                "weight": f"{particle.weight.numerator}/{particle.weight.denominator}",
            }
        )
    artifact: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "restricted_hidden_information": True,
        "root_id": root_id,
        "root_commitment_sha256": root_commitment,
        "round_index": round_index,
        "solver_seed": solver_seed,
        "observation": observation.to_canonical_dict(),
        "observation_digest": observation.digest(),
        "behavior_model_id": full_range.behavior_model_id,
        "behavior_model_sha256": full_range.behavior_model_sha256,
        "epsilon": f"{full_range.epsilon.numerator}/{full_range.epsilon.denominator}",
        "opponent_public_evidence_normalizer_exact": (
            f"{full_range.evidence_normalizer.numerator}/"
            f"{full_range.evidence_normalizer.denominator}"
        ),
        "effective_sample_size_exact": (
            f"{full_range.effective_sample_size.numerator}/"
            f"{full_range.effective_sample_size.denominator}"
        ),
        "range_sha256": full_range.range_sha256,
        "range_content_sha256": full_range.range_content_sha256,
        "range_build_sha256": full_range.range_build_sha256,
        "particle_commitments": list(full_range.particle_commitments),
        "particles": particles,
        "content_manifest": copy.deepcopy(
            dict(full_range.metadata["content_manifest"])
        ),
        "build_manifest": copy.deepcopy(
            dict(full_range.metadata["build_manifest"])
        ),
        "behavior_model_manifest": copy.deepcopy(
            dict(full_range.metadata["behavior_model_manifest"])
        ),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_restricted_range_evidence(artifact)[0]


def verify_restricted_range_evidence(
    artifact: Any,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Reconstruct and fully verify one restricted posterior artifact."""

    raw = _require_mapping(artifact, label="range evidence")
    _exact_keys(raw, _TOP_KEYS, label="range evidence")
    if raw.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("range evidence schema mismatch")
    if raw.get("restricted_hidden_information") is not True:
        raise ValueError("range evidence must be marked restricted")
    if not isinstance(raw.get("root_id"), str) or not raw["root_id"]:
        raise ValueError("range evidence root_id must be non-empty")
    _require_sha256(raw.get("root_commitment_sha256"), label="root commitment")
    if (
        isinstance(raw.get("round_index"), bool)
        or not isinstance(raw.get("round_index"), int)
        or raw["round_index"] < 0
    ):
        raise ValueError("range evidence round_index is invalid")
    if isinstance(raw.get("solver_seed"), bool) or not isinstance(
        raw.get("solver_seed"), int
    ):
        raise TypeError("range evidence solver_seed is invalid")
    supplied_hash = _require_sha256(raw.get("artifact_sha256"), label="artifact hash")
    unsigned = dict(raw)
    unsigned.pop("artifact_sha256")
    if canonical_sha256(unsigned) != supplied_hash:
        raise ValueError("range evidence artifact SHA-256 mismatch")
    observation = _parse_observation(raw["observation"])
    if raw.get("observation_digest") != observation.digest():
        raise ValueError("range evidence observation digest mismatch")
    for field in (
        "behavior_model_sha256",
        "range_sha256",
        "range_content_sha256",
        "range_build_sha256",
    ):
        _require_sha256(raw.get(field), label=field)
    if not isinstance(raw.get("behavior_model_id"), str) or not raw[
        "behavior_model_id"
    ]:
        raise ValueError("behavior_model_id must be non-empty")
    epsilon = _canonical_fraction(raw.get("epsilon"), label="epsilon", positive=False)
    evidence_normalizer = _canonical_fraction(
        raw.get("opponent_public_evidence_normalizer_exact"),
        label="opponent_public_evidence_normalizer_exact",
        positive=True,
    )
    ess = _canonical_fraction(
        raw.get("effective_sample_size_exact"),
        label="effective_sample_size_exact",
        positive=True,
    )
    raw_particles = raw.get("particles")
    commitments = raw.get("particle_commitments")
    if not isinstance(raw_particles, list) or not raw_particles:
        raise ValueError("range evidence particles must be non-empty")
    if not isinstance(commitments, list) or len(commitments) != len(raw_particles):
        raise ValueError("range evidence particle commitment count mismatch")
    particles: list[JointParticle] = []
    parsed_commitments: list[str] = []
    for index, item in enumerate(raw_particles):
        row = _require_mapping(item, label=f"particles[{index}]")
        _exact_keys(row, _PARTICLE_KEYS, label=f"particles[{index}]")
        commitment = _require_sha256(
            row.get("commitment"), label=f"particles[{index}].commitment"
        )
        undealt = row.get("undealt_cards")
        if not isinstance(undealt, list):
            raise TypeError(f"particles[{index}].undealt_cards must be a list")
        particle = JointParticle(
            bb_recall=_parse_recall(
                row["bb_recall"], label=f"particles[{index}].bb_recall"
            ),
            btn_recall=_parse_recall(
                row["btn_recall"], label=f"particles[{index}].btn_recall"
            ),
            undealt_cards=tuple(undealt),
            weight=_canonical_fraction(
                row.get("weight"), label=f"particles[{index}].weight", positive=True
            ),
        )
        particles.append(particle)
        parsed_commitments.append(commitment)
    if list(commitments) != parsed_commitments:
        raise ValueError("range evidence commitment order mismatch")
    metadata = {
        "content_manifest": copy.deepcopy(dict(_require_mapping(
            raw["content_manifest"], label="content_manifest"
        ))),
        "build_manifest": copy.deepcopy(dict(_require_mapping(
            raw["build_manifest"], label="build_manifest"
        ))),
        "behavior_model_manifest": copy.deepcopy(dict(_require_mapping(
            raw["behavior_model_manifest"], label="behavior_model_manifest"
        ))),
        "opponent_public_evidence_normalizer_exact": raw[
            "opponent_public_evidence_normalizer_exact"
        ],
        "opponent_public_evidence_scope": raw["build_manifest"].get(
            "opponent_public_evidence_scope"
        ),
    }
    reconstructed = FullCardRange(
        observation_digest=observation.digest(),
        particles=tuple(particles),
        particle_commitments=tuple(parsed_commitments),
        behavior_model_id=raw["behavior_model_id"],
        behavior_model_sha256=raw["behavior_model_sha256"],
        epsilon=epsilon,
        evidence_normalizer=evidence_normalizer,
        effective_sample_size=ess,
        range_sha256=raw["range_sha256"],
        range_content_sha256=raw["range_content_sha256"],
        range_build_sha256=raw["range_build_sha256"],
        metadata=metadata,
    )
    audit = verify_full_card_range(observation, reconstructed)
    posterior = {
        commitment: f"{particle.weight.numerator}/{particle.weight.denominator}"
        for commitment, particle in zip(parsed_commitments, particles)
    }
    result = {
        **dict(audit),
        "root_id": raw["root_id"],
        "root_commitment_sha256": raw["root_commitment_sha256"],
        "round_index": raw["round_index"],
        "solver_seed": raw["solver_seed"],
        "observation_digest": observation.digest(),
        "artifact_sha256": supplied_hash,
        "posterior_weights": posterior,
    }
    return copy.deepcopy(dict(raw)), result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"range evidence JSON has duplicate key {key!r}")
        result[key] = value
    return result


def write_restricted_range_evidence(
    path: str | Path, artifact: Mapping[str, Any]
) -> Path:
    verified, _audit = verify_restricted_range_evidence(artifact)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(verified))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)
    read_restricted_range_evidence(output)
    return output


def read_restricted_range_evidence(
    path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        if not text.endswith("\n") or text.count("\n") != 1:
            raise ValueError("range evidence must be one canonical JSON line")
        raw = json.loads(text[:-1], object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read restricted range evidence: {exc}") from exc
    if canonical_json(raw) != text[:-1]:
        raise ValueError("range evidence file is not canonical JSON")
    return verify_restricted_range_evidence(raw)


__all__ = [
    "EVIDENCE_SCHEMA",
    "build_restricted_range_evidence",
    "read_restricted_range_evidence",
    "verify_restricted_range_evidence",
    "write_restricted_range_evidence",
]
