"""Run a real-checkpoint, permanently non-promoting T3-BB fixed-point smoke.

This producer is intentionally much smaller than the production promotion
gate.  It proves that the endogenous T3-BB likelihood loop can be executed
from raw-verified T1/T2 calibration evidence without using an unverified T3
ranking prior:

* the saved behavior-calibration smoke is rebuilt from every raw decision and
  direct-logit row before its four exact T1/T2 checkpoints are loaded;
* a uniform T3 distribution is used exactly once, to discover the complete
  set of T3-BB behavior queries on a fixed physical-particle support;
* every discovered query is converted to the public ``t3_first`` key and
  solved by the real full-card external-sampling MCCFR implementation;
* independent-seed policies are averaged and quantized to an exact Q32 table;
* the table is attached only through the non-promoted fixed-point iteration
  dispatch, which fails closed for every missing query;
* every canonical BB/BTN x Joker0/1/2 evaluation job writes an actual solver
  checkpoint, average-strategy file, restricted posterior artifact, and a
  fresh-verified round bundle; and
* two or more transitions retain the candidate/checkpoint chain plus raw
  policy and BTN-posterior drift.

The default fixture is two seeds, one root in each of six strata, two
transitions, one MCCFR iteration, and one posterior particle.  Those settings
are useful for deterministic end-to-end diagnostics and deliberately cannot
satisfy the production fixed-point gate.  No artifact emitted here is a
strategic-strength, convergence, exact-exploitability, or promotion claim.
Restricted files contain opponent private recall and undealt physical cards;
they must never be passed to a public policy input or response.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.behavior_temperature_calibration import (
    verify_behavior_temperature_calibration,
    verify_model_evaluation_row,
)
from ai.tutor.calibrated_behavior_bootstrap import (
    FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
    build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch,
)
from ai.tutor.collect_hu_behavior_traces import (
    CollectedBehaviorTraces,
    read_behavior_trace_dataset,
)
from ai.tutor.promotion_gate_m3_full_card_strength import (
    INFORMATION_MODEL,
    REQUIRED_STRATA,
    RULESET,
    SOLVER_ADAPTER,
    SOLVER_METHOD,
    SOLVER_SCHEMA,
    SOURCE_SCHEMA,
    canonical_json,
    canonical_sha256,
    root_identity_commitment_sha256,
)
from ai.tutor.t3_bb_candidate_queries import (
    Q32_DENOMINATOR,
    behavior_t3_bb_to_t3_first_key,
    quantize_mccfr_distribution_q32,
)
from ai.tutor.t3_bb_checkpoint_bundle import (
    T3BBCheckpointBundleEntrySource,
    T3BBCheckpointKey,
    build_t3_bb_checkpoint_bundle,
    load_t3_bb_checkpoint_bundle_root_policies,
    read_t3_bb_checkpoint_bundle,
    write_t3_bb_checkpoint_bundle_manifest,
)
from ai.tutor import t3_bb_fixed_point_gate_v2 as fixed_point_gate_v2
from ai.tutor.t3_bb_fixed_point_runtime import (
    build_fixed_point_iteration_behavior_dispatch,
    build_t3_bb_candidate_policy_artifact,
    read_t3_bb_candidate_policy_artifact,
    verify_t3_bb_candidate_policy_artifact,
)
from ai.tutor.t3_bb_range_evidence import (
    build_restricted_range_evidence,
    read_restricted_range_evidence,
    write_restricted_range_evidence,
)
from ai.tutor.t3_hu_full_card_mccfr import (
    FullCardGenerativeAdapter,
    solve_full_card_external_sampling_mccfr,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    BehaviorInfoSet,
    FrozenBehaviorModel,
    FullCardRange,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture


REPORT_SCHEMA = "ofc_t3_bb_fixed_point_smoke_report/v1"
ROOT_OBSERVATION_SCHEMA = "ofc_t3_bb_fixed_point_smoke_roots/v1"
DISCOVERY_SCHEMA = "ofc_t3_bb_fixed_point_query_discovery/v1"
ROUND_SCHEMA = "ofc_t3_bb_fixed_point_smoke_round/v1"
FIXTURE_ID = "t3-bb-fixed-point-real-mccfr-smoke-20260713-v1"
DEFAULT_CALIBRATION_REPORT = Path(
    "ai/reports/m3_behavior_calibration_smoke_20260713"
)
DEFAULT_OUTPUT_DIR = Path("ai/reports/t3_bb_fixed_point_smoke_20260713")
DEFAULT_SOLVER_SEEDS = (20260731, 20260732)
DEFAULT_TRANSITIONS = 2
DEFAULT_ITERATIONS = 1
DEFAULT_MAX_PARTICLES = 1
DEFAULT_MAX_INFOSETS = 100_000
DEFAULT_EPSILON = Fraction(1, 1000)

_SOURCE_FILES = (
    "ai/tutor/t3_hu_full_card_range.py",
    "ai/tutor/t3_hu_full_card_mccfr.py",
    "ai/tutor/t3_bb_candidate_queries.py",
    "ai/tutor/t3_bb_fixed_point_runtime.py",
    "ai/tutor/t3_bb_range_evidence.py",
    "ai/tutor/t3_bb_checkpoint_bundle.py",
    "ai/tutor/t3_bb_fixed_point_gate_v2.py",
    "ai/tutor/run_t3_bb_fixed_point_smoke.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
    if _read_canonical_json(path) != dict(value):
        raise AssertionError(f"canonical JSON readback changed {path.name}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise ValueError(f"{path} must be one canonical JSON line")
    value = json.loads(raw[:-1], object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict) or canonical_json(value) != raw[:-1]:
        raise ValueError(f"{path} is not canonical JSON")
    return value


def _read_canonical_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise ValueError(f"{path} must end in one newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw[:-1].split("\n")):
        value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(value, dict) or canonical_json(value) != line:
            raise ValueError(f"{path} row {index} is not canonical JSON")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return tuple(rows)


@dataclass(frozen=True)
class _CalibrationRaw:
    natural: CollectedBehaviorTraces
    natural_evaluations: tuple[dict[str, Any], ...]
    challenge: CollectedBehaviorTraces
    challenge_evaluations: tuple[dict[str, Any], ...]
    calibration: dict[str, Any]
    file_sha256: Mapping[str, str]


def _load_calibration_raw(report_dir: Path) -> _CalibrationRaw:
    natural_decisions = report_dir / "natural" / "decisions.jsonl"
    challenge_decisions = report_dir / "challenge" / "decisions.jsonl"
    natural_evaluations_path = report_dir / "natural" / "model_evaluations.jsonl"
    challenge_evaluations_path = (
        report_dir / "challenge" / "model_evaluations.jsonl"
    )
    calibration_path = report_dir / "calibration.json"
    natural = read_behavior_trace_dataset(natural_decisions)
    challenge = read_behavior_trace_dataset(challenge_decisions)
    natural_evaluations = _read_canonical_jsonl(natural_evaluations_path)
    challenge_evaluations = _read_canonical_jsonl(challenge_evaluations_path)
    if len(natural.records) != len(natural_evaluations):
        raise ValueError("natural decision/evaluation row count mismatch")
    if len(challenge.records) != len(challenge_evaluations):
        raise ValueError("challenge decision/evaluation row count mismatch")
    for record, row in zip(natural.records, natural_evaluations):
        verify_model_evaluation_row(row, record)
    for record, row in zip(challenge.records, challenge_evaluations):
        verify_model_evaluation_row(row, record)
    calibration = _read_canonical_json(calibration_path)
    rebuilt = verify_behavior_temperature_calibration(
        natural.records,
        natural_evaluations,
        calibration,
        challenge_records=challenge.records,
        challenge_evaluation_rows=challenge_evaluations,
    )
    if rebuilt != calibration:
        raise AssertionError("raw calibration rebuild changed the saved artifact")
    if calibration.get("promotion_eligible") is not False:
        raise ValueError(
            "this diagnostic requires the saved non-promoting calibration smoke"
        )
    paths = {
        "natural_decisions": natural_decisions,
        "natural_roots": report_dir / "natural" / "roots.jsonl",
        "natural_manifest": report_dir / "natural" / "manifest.json",
        "natural_evaluations": natural_evaluations_path,
        "challenge_decisions": challenge_decisions,
        "challenge_roots": report_dir / "challenge" / "roots.jsonl",
        "challenge_manifest": report_dir / "challenge" / "manifest.json",
        "challenge_evaluations": challenge_evaluations_path,
        "calibration": calibration_path,
        "temperature_gate_config": report_dir / "temperature_gate_config.json",
    }
    return _CalibrationRaw(
        natural=natural,
        natural_evaluations=natural_evaluations,
        challenge=challenge,
        challenge_evaluations=challenge_evaluations,
        calibration=calibration,
        file_sha256=MappingProxyType(
            {name: _sha256_file(path) for name, path in sorted(paths.items())}
        ),
    )


def _load_t1_t2_bootstrap(
    workspace_root: Path, calibration_report_dir: Path
) -> tuple[FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch, _CalibrationRaw]:
    raw = _load_calibration_raw(calibration_report_dir)
    dispatch = build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
        raw.natural.records,
        raw.natural_evaluations,
        raw.calibration,
        challenge_records=raw.challenge.records,
        challenge_evaluation_rows=raw.challenge_evaluations,
        workspace_root=workspace_root,
        require_promoted_source=False,
        model_id="t3_bb_fixed_point_smoke_raw_verified_t1_t2_bootstrap_v1",
    )
    manifest = dict(dispatch.model_manifest)
    required_false = (
        "source_calibration_promotion_eligible",
        "source_calibration_all_required_gates_passed",
        "promotion_eligible",
        "strategic_strength_evaluated",
        "strategic_strength_claimed",
    )
    if any(manifest.get(field) is not False for field in required_false):
        raise AssertionError("diagnostic T1/T2 bootstrap made a promotion claim")
    if manifest.get("fixed_point_bootstrap_only") is not True:
        raise AssertionError("T1/T2 runtime is not fixed-point-bootstrap-only")
    return dispatch, raw


class _T3BBQueryDiscoveryDispatch(FrozenBehaviorModel):
    """Capture T3-BB queries while using no T3 strategic prior."""

    def __init__(
        self, bootstrap: FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch
    ) -> None:
        self._bootstrap = bootstrap
        self._queries: dict[str, tuple[BehaviorInfoSet, InfoSetKey]] = {}
        self._manifest = {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": "t3_bb_query_discovery_only_uniform_support_v1",
            "model_type": "t1_t2_bootstrap_plus_t3_bb_query_discovery_only",
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "promotion_eligible": False,
            "strategic_strength_evaluated": False,
            "strategic_strength_claimed": False,
            "query_discovery_only": True,
            "t3_policy_prior_used": False,
            "t3_discovery_distribution": "uniform_legal_not_a_candidate_policy",
            "no_fallback": True,
            "t1_t2_bootstrap_model_sha256": bootstrap.model_sha256,
        }

    @property
    def model_id(self) -> str:
        return str(self._manifest["model_id"])

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._manifest))

    @property
    def model_sha256(self) -> str:
        return canonical_sha256(self._manifest)

    @property
    def queries(self) -> Mapping[str, tuple[BehaviorInfoSet, InfoSetKey]]:
        return MappingProxyType(dict(self._queries))

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if information.turn in (1, 2) and information.actor in ("bb", "btn"):
            return self._bootstrap.action_distribution(information)
        if information.turn != 3 or information.actor != "bb":
            raise KeyError(
                f"query discovery has no route for T{information.turn} "
                f"{information.actor}"
            )
        key = behavior_t3_bb_to_t3_first_key(information)
        digest = information.digest()
        prior = self._queries.setdefault(digest, (information, key))
        if prior != (information, key):
            raise AssertionError("T3-BB behavior query digest collision")
        probability = Fraction(1, information.legal_action_count)
        return BehaviorDistribution(
            information_digest=digest,
            probabilities=MappingProxyType(
                {
                    action_id: probability
                    for action_id in information.legal_action_ids
                }
            ),
            source="uniform_model",
            used_fallback=False,
        )


@dataclass(frozen=True)
class _Root:
    root_id: str
    stratum: str
    actor: str
    visible_joker_count: int
    observation: InfoSetKey
    fixture_manifest_sha256: str
    root_identity_commitment_sha256: str
    root_commitment_sha256: str
    gate_record: Mapping[str, Any]

    def gate_payload(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.gate_record))


def _build_roots(
    *, rust_solver_path: Path | None, rust_timeout_s: float
) -> tuple[_Root, ...]:
    roots: list[_Root] = []
    for actor in ("bb", "btn"):
        for joker_count in (0, 1, 2):
            stratum = f"{actor}_joker{joker_count}"
            fixture = compile_canonical_reduced_fixture(
                actor,
                joker_count,
                rust_solver_path=rust_solver_path,
                rust_timeout_s=rust_timeout_s,
            )
            observation = fixture.root.branches[0].child.infoset_key
            expected_phase = "t3_first" if actor == "bb" else "t3_second"
            if (
                observation.actor != actor
                or observation.turn != 3
                or observation.phase != expected_phase
                or sum(card in ("X1", "X2") for card in observation.current_draw)
                != joker_count
            ):
                raise AssertionError(f"canonical {stratum} fixture identity drift")
            root_id = f"{FIXTURE_ID}/{stratum}"
            payload = fixed_point_gate_v2.build_root_record(root_id, observation)
            roots.append(
                _Root(
                    root_id=root_id,
                    stratum=stratum,
                    actor=actor,
                    visible_joker_count=joker_count,
                    observation=observation,
                    fixture_manifest_sha256=fixture.fixture_manifest_sha256,
                    root_identity_commitment_sha256=payload[
                        "root_identity_commitment_sha256"
                    ],
                    root_commitment_sha256=payload["root_commitment_sha256"],
                    gate_record=MappingProxyType(copy.deepcopy(payload)),
                )
            )
    if {root.stratum for root in roots} != set(REQUIRED_STRATA):
        raise AssertionError("canonical roots do not cover the six required strata")
    return tuple(sorted(roots, key=lambda root: root.stratum))


def _root_observation_manifest(
    roots: Sequence[_Root], seeds: Sequence[int]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": ROOT_OBSERVATION_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "promotion_eligible": False,
        "restricted_hidden_information": False,
        "fixed_particle_support": True,
        "solver_seeds": list(seeds),
        "roots": [
            {
                **root.gate_payload(),
                "fixture_manifest_sha256": root.fixture_manifest_sha256,
                "observation_digest": root.observation.digest(),
                "observation": root.observation.to_canonical_dict(),
            }
            for root in roots
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _discover_queries(
    roots: Sequence[_Root],
    seeds: Sequence[int],
    discovery: _T3BBQueryDiscoveryDispatch,
    *,
    max_particles: int,
    epsilon: Fraction,
) -> tuple[dict[str, tuple[BehaviorInfoSet, InfoSetKey]], dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    sources_by_query: dict[str, set[str]] = {}
    for root in roots:
        if root.actor != "btn":
            continue
        for seed in seeds:
            result = build_history_weighted_full_card_range(
                root.observation,
                discovery,
                epsilon=epsilon,
                max_particles=max_particles,
                seed=seed,
            )
            audited_digests = {
                str(row["information_digest"])
                for row in result.metadata["behavior_query_audit"]
            }
            t3_query_digests = sorted(audited_digests & set(discovery.queries))
            if not t3_query_digests:
                raise AssertionError("BTN discovery range contained no T3-BB query")
            for digest in t3_query_digests:
                sources_by_query.setdefault(digest, set()).add(
                    root.root_commitment_sha256
                )
            ranges.append(
                {
                    "root_id": root.root_id,
                    "stratum": root.stratum,
                    "solver_seed": seed,
                    "observation_digest": root.observation.digest(),
                    "particle_commitments": list(result.particle_commitments),
                    "range_content_sha256": result.range_content_sha256,
                    "range_build_sha256": result.range_build_sha256,
                    "behavior_query_count": result.metadata[
                        "behavior_query_count"
                    ],
                    "t3_bb_query_digests": t3_query_digests,
                    "uniform_fallback_count": result.metadata[
                        "behavior_uniform_fallback_count"
                    ],
                }
            )
    queries = dict(discovery.queries)
    if not queries:
        raise AssertionError("BTN range discovery captured no T3-BB query")
    if set(sources_by_query) != set(queries):
        raise AssertionError("query discovery source-root coverage is incomplete")
    manifest: dict[str, Any] = {
        "schema": DISCOVERY_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "promotion_eligible": False,
        "strategic_strength_evaluated": False,
        "query_discovery_only": True,
        "t3_policy_prior_used": False,
        "uniform_distribution_used_only_for_query_discovery": True,
        "restricted_hidden_information": True,
        "fixed_particle_support": True,
        "max_particles": max_particles,
        "epsilon": f"{epsilon.numerator}/{epsilon.denominator}",
        "solver_seeds": list(seeds),
        "discovery_behavior_model_sha256": discovery.model_sha256,
        "range_audit": ranges,
        "queries": [
            {
                "behavior_information_digest": digest,
                "behavior_information": information.to_canonical_dict(),
                "solver_information_digest": key.digest(),
                "solver_information": key.to_canonical_dict(),
                "source_root_commitments": sorted(sources_by_query[digest]),
            }
            for digest, (information, key) in sorted(queries.items())
        ],
    }
    manifest["artifact_sha256"] = canonical_sha256(manifest)
    return queries, manifest


def _source_manifests(
    workspace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    files = []
    for relative in sorted(_SOURCE_FILES):
        path = workspace_root / relative
        files.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
            }
        )
    source: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "files": files,
    }
    source_sha256 = canonical_sha256(source)
    range_source_sha256 = next(
        row["sha256"]
        for row in files
        if row["path"] == "ai/tutor/t3_hu_full_card_range.py"
    )
    solver: dict[str, Any] = {
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
        "source_manifest_sha256": source_sha256,
    }
    solver_sha256 = canonical_sha256(solver)
    return source, solver, range_source_sha256, source_sha256, solver_sha256


def _query_root_id(digest: str) -> str:
    return f"{FIXTURE_ID}/query/{digest}"


def _build_candidate_query_manifest(
    queries: Mapping[str, tuple[BehaviorInfoSet, InfoSetKey]],
    discovery_artifact: Mapping[str, Any],
    *,
    evaluation_root_manifest_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    discovery_rows = {
        str(row["behavior_information_digest"]): row
        for row in discovery_artifact["queries"]
    }
    if set(discovery_rows) != set(queries):
        raise AssertionError("discovery/query manifest coverage mismatch")
    records: dict[str, Mapping[str, Any]] = {}
    for digest, (information, _key) in sorted(queries.items()):
        record = fixed_point_gate_v2.build_candidate_query_record(
            _query_root_id(digest),
            information,
            source_root_commitments=discovery_rows[digest][
                "source_root_commitments"
            ],
        )
        if record["behavior_information_digest"] != digest:
            raise AssertionError("v2 query record changed the behavior digest")
        records[digest] = MappingProxyType(copy.deepcopy(record))
    manifest = fixed_point_gate_v2.build_candidate_query_manifest(
        list(records.values()),
        evaluation_root_manifest_sha256=evaluation_root_manifest_sha256,
    )
    return records, manifest


@dataclass(frozen=True)
class _SolvedJob:
    key: T3BBCheckpointKey
    source: T3BBCheckpointBundleEntrySource
    root_policy: Mapping[str, float]
    posterior_weights: Mapping[str, Fraction]
    range_evidence_relative_path: str
    range_evidence_sha256: str
    range_content_sha256: str
    range_build_sha256: str
    behavior_model_sha256: str


def _solve_job(
    *,
    bundle_root: Path,
    relative_job_dir: Path,
    bundle_round_index: int,
    evidence_round_index: int,
    root_id: str,
    root_commitment: str,
    solver_seed: int,
    observation: InfoSetKey,
    root_range: FullCardRange,
    iterations: int,
    max_infosets: int,
    solver_manifest_sha256: str,
    source_manifest_sha256: str,
) -> _SolvedJob:
    job_dir = bundle_root / relative_job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    restricted_path = job_dir / "restricted-range.json"
    range_artifact = build_restricted_range_evidence(
        observation,
        root_range,
        root_id=root_id,
        root_commitment_sha256=root_commitment,
        round_index=evidence_round_index,
        solver_seed=solver_seed,
    )
    write_restricted_range_evidence(restricted_path, range_artifact)
    range_readback, range_audit = read_restricted_range_evidence(restricted_path)
    if range_readback != range_artifact:
        raise AssertionError("restricted range evidence readback changed")

    checkpoint_path = job_dir / "checkpoint.json"
    strategy_path = job_dir / "average-strategy.json"
    adapter = FullCardGenerativeAdapter(observation, root_range)
    result = solve_full_card_external_sampling_mccfr(
        adapter,
        iterations=iterations,
        seed=solver_seed,
        max_infosets=max_infosets,
        linear_averaging=True,
        checkpoint_path=checkpoint_path,
    )
    if result.metadata.get("full_card") is not True:
        raise AssertionError("solver did not report a full-card traversal")
    if result.metadata.get("exact_exploitability_computed") is not False:
        raise AssertionError("smoke solver made an exact exploitability claim")
    if result.metadata.get("full_card_policy_promoted") is not False:
        raise AssertionError("smoke solver promoted a policy")
    _atomic_write(strategy_path, result.average_strategy_json.encode("utf-8"))
    if strategy_path.read_text(encoding="utf-8") != result.average_strategy_json:
        raise AssertionError("average strategy readback changed solver bytes")

    matching = [
        distribution
        for key, distribution in result.average_strategy.items()
        if key.digest() == observation.digest()
    ]
    if len(matching) != 1:
        raise AssertionError("solver result lacks one exact root policy")
    root_policy = {
        action_id: float(probability)
        for action_id, probability in sorted(matching[0].items())
    }
    if not math.isclose(
        math.fsum(root_policy.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError("root strategy does not sum to one")
    posterior = {
        commitment: Fraction(weight)
        for commitment, weight in sorted(
            range_audit["posterior_weights"].items()
        )
    }
    key = T3BBCheckpointKey(
        round_index=bundle_round_index,
        root_id=root_id,
        root_commitment_sha256=root_commitment,
        solver_seed=solver_seed,
    )
    source = T3BBCheckpointBundleEntrySource(
        **key.as_dict(),
        checkpoint_path=checkpoint_path.relative_to(bundle_root).as_posix(),
        average_strategy_json_path=strategy_path.relative_to(bundle_root).as_posix(),
        range_content_sha256=root_range.range_content_sha256,
        range_build_sha256=root_range.range_build_sha256,
        observation_digest=observation.digest(),
        solver_manifest_sha256=solver_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    return _SolvedJob(
        key=key,
        source=source,
        root_policy=MappingProxyType(root_policy),
        posterior_weights=MappingProxyType(posterior),
        range_evidence_relative_path=restricted_path.relative_to(
            bundle_root
        ).as_posix(),
        range_evidence_sha256=range_artifact["artifact_sha256"],
        range_content_sha256=root_range.range_content_sha256,
        range_build_sha256=root_range.range_build_sha256,
        behavior_model_sha256=root_range.behavior_model_sha256,
    )


def _write_candidate_artifact(
    path: Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    verified = verify_t3_bb_candidate_policy_artifact(artifact)
    _write_canonical_json(path, verified)
    readback = read_t3_bb_candidate_policy_artifact(path)
    if readback != verified:
        raise AssertionError("candidate policy artifact readback changed")
    return readback


def _assert_candidate_query_coverage(
    artifact: Mapping[str, Any],
    queries: Mapping[str, tuple[BehaviorInfoSet, InfoSetKey]],
) -> None:
    verified = verify_t3_bb_candidate_policy_artifact(artifact)
    probabilities = verified["model_manifest"]["probabilities"]
    expected = set(queries)
    actual = set(probabilities)
    if actual != expected:
        raise ValueError(
            "candidate query coverage mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for digest, (information, _key) in queries.items():
        support = set(probabilities[digest])
        legal = set(information.legal_action_ids)
        if support != legal:
            raise ValueError(
                f"candidate query {digest} support mismatch: "
                f"missing={sorted(legal - support)}, extra={sorted(support - legal)}"
            )


@dataclass(frozen=True)
class _CandidateRound:
    candidate_index: int
    artifact: Mapping[str, Any]
    query_bundle: Mapping[str, Any]
    query_jobs: Mapping[tuple[str, int], _SolvedJob]
    evaluation_bundle: Mapping[str, Any]
    evaluation_jobs: Mapping[tuple[str, int], _SolvedJob]
    round_manifest: Mapping[str, Any]


def _candidate_round(
    *,
    output_dir: Path,
    candidate_index: int,
    queries: Mapping[str, tuple[BehaviorInfoSet, InfoSetKey]],
    query_records: Mapping[str, Mapping[str, Any]],
    roots: Sequence[_Root],
    seeds: Sequence[int],
    bootstrap: FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
    epsilon: Fraction,
    max_particles: int,
    iterations: int,
    max_infosets: int,
    solver_manifest_sha256: str,
    source_manifest_sha256: str,
    range_builder_source_sha256: str,
) -> _CandidateRound:
    bundle_round = candidate_index + 1
    round_relative = Path(f"candidate-{candidate_index:03d}")
    round_dir = output_dir / round_relative
    query_relative = round_relative / "query-bundle"
    query_root = output_dir / query_relative
    query_root.mkdir(parents=True, exist_ok=True)
    query_jobs: dict[tuple[str, int], _SolvedJob] = {}
    for digest, (_information, observation) in sorted(queries.items()):
        query_record = query_records[digest]
        query_id = str(query_record["query_id"])
        query_commitment = str(query_record["query_commitment_sha256"])
        for seed in seeds:
            root_range = build_history_weighted_full_card_range(
                observation,
                bootstrap,
                epsilon=epsilon,
                max_particles=max_particles,
                seed=seed,
            )
            job = _solve_job(
                bundle_root=output_dir,
                relative_job_dir=query_relative / "jobs" / digest / str(seed),
                bundle_round_index=bundle_round,
                evidence_round_index=bundle_round,
                root_id=query_id,
                root_commitment=query_commitment,
                solver_seed=seed,
                observation=observation,
                root_range=root_range,
                iterations=iterations,
                max_infosets=max_infosets,
                solver_manifest_sha256=solver_manifest_sha256,
                source_manifest_sha256=source_manifest_sha256,
            )
            query_jobs[(digest, seed)] = job
    query_keys = [job.key for job in query_jobs.values()]
    query_bundle = build_t3_bb_checkpoint_bundle(
        output_dir,
        [job.source for job in query_jobs.values()],
        expected_keys=query_keys,
    )
    query_bundle = write_t3_bb_checkpoint_bundle_manifest(
        query_root / "manifest.json",
        query_bundle,
        bundle_root=output_dir,
        expected_keys=query_keys,
    )
    # Fresh loading proves each aggregate distribution comes from the actual
    # checkpoint-derived strategy bytes rather than a copied summary.
    query_policies = load_t3_bb_checkpoint_bundle_root_policies(
        query_bundle, bundle_root=output_dir, expected_keys=query_keys
    )
    q32_rows: dict[str, Mapping[str, Fraction]] = {}
    for digest, (information, _observation) in sorted(queries.items()):
        rows = [
            query_policies[query_jobs[(digest, seed)].key] for seed in seeds
        ]
        averaged = {
            action_id: math.fsum(row[action_id] for row in rows) / len(rows)
            for action_id in information.legal_action_ids
        }
        total = math.fsum(averaged.values())
        normalized = {
            action_id: float(value / total)
            for action_id, value in averaged.items()
        }
        q32_rows[digest] = quantize_mccfr_distribution_q32(
            normalized, legal_action_ids=information.legal_action_ids
        )
    artifact = build_t3_bb_candidate_policy_artifact(
        q32_rows,
        checkpoint_sha256=query_bundle["bundle_checkpoint_sha256"],
        solver_manifest_sha256=solver_manifest_sha256,
        range_builder_source_sha256=range_builder_source_sha256,
        model_id=(
            f"t3_bb_fixed_point_smoke_candidate_{candidate_index:03d}_q32_v1"
        ),
    )
    candidate_path = round_dir / "candidate-policy.json"
    artifact = _write_candidate_artifact(candidate_path, artifact)
    _assert_candidate_query_coverage(artifact, queries)

    iteration_dispatch = build_fixed_point_iteration_behavior_dispatch(
        bootstrap,
        artifact,
        model_id=(
            f"t3_bb_fixed_point_smoke_iteration_{candidate_index:03d}_v1"
        ),
    )
    dispatch_manifest = dict(iteration_dispatch.model_manifest)
    if (
        dispatch_manifest.get("promotion_eligible") is not False
        or dispatch_manifest.get("fixed_point_iteration_only") is not True
        or dispatch_manifest.get("no_fallback") is not True
    ):
        raise AssertionError("candidate iteration dispatch claim drift")

    evaluation_relative = round_relative / "evaluation-bundle"
    evaluation_root = output_dir / evaluation_relative
    evaluation_root.mkdir(parents=True, exist_ok=True)
    evaluation_jobs: dict[tuple[str, int], _SolvedJob] = {}
    for root in roots:
        for seed in seeds:
            # The same root/seed/max-particle tuple is reused at every candidate
            # index.  Candidate probabilities may change weights, never support.
            root_range = build_history_weighted_full_card_range(
                root.observation,
                iteration_dispatch,
                epsilon=epsilon,
                max_particles=max_particles,
                seed=seed,
            )
            if root_range.metadata["behavior_uniform_fallback_count"] != 0:
                raise AssertionError("fixed-point evaluation used behavior fallback")
            job = _solve_job(
                bundle_root=output_dir,
                relative_job_dir=(
                    evaluation_relative / "jobs" / root.stratum / str(seed)
                ),
                bundle_round_index=bundle_round,
                evidence_round_index=bundle_round,
                root_id=root.root_id,
                root_commitment=root.root_commitment_sha256,
                solver_seed=seed,
                observation=root.observation,
                root_range=root_range,
                iterations=iterations,
                max_infosets=max_infosets,
                solver_manifest_sha256=solver_manifest_sha256,
                source_manifest_sha256=source_manifest_sha256,
            )
            evaluation_jobs[(root.root_id, seed)] = job
    evaluation_keys = [job.key for job in evaluation_jobs.values()]
    evaluation_bundle = build_t3_bb_checkpoint_bundle(
        output_dir,
        [job.source for job in evaluation_jobs.values()],
        expected_keys=evaluation_keys,
    )
    evaluation_bundle = write_t3_bb_checkpoint_bundle_manifest(
        evaluation_root / "manifest.json",
        evaluation_bundle,
        bundle_root=output_dir,
        expected_keys=evaluation_keys,
    )
    loaded_evaluation = load_t3_bb_checkpoint_bundle_root_policies(
        evaluation_bundle,
        bundle_root=output_dir,
        expected_keys=evaluation_keys,
    )
    for job in evaluation_jobs.values():
        if loaded_evaluation[job.key] != dict(job.root_policy):
            raise AssertionError("evaluation bundle policy differs from solver result")

    restricted_assets = [
        {
            "kind": kind,
            "root_id": job.key.root_id,
            "solver_seed": job.key.solver_seed,
            "relative_path": job.range_evidence_relative_path,
            "artifact_sha256": job.range_evidence_sha256,
            "range_content_sha256": job.range_content_sha256,
            "range_build_sha256": job.range_build_sha256,
        }
        for kind, jobs in (
            ("candidate_query", query_jobs),
            ("evaluation_root", evaluation_jobs),
        )
        for job in jobs.values()
    ]
    round_manifest: dict[str, Any] = {
        "schema": ROUND_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "candidate_index": candidate_index,
        "bundle_round_index": bundle_round,
        "promotion_eligible": False,
        "strategic_strength_evaluated": False,
        "exact_exploitability_computed": False,
        "candidate_policy_artifact_path": "candidate-policy.json",
        "candidate_policy_artifact_sha256": artifact["artifact_sha256"],
        "candidate_policy_checkpoint_sha256": artifact["model_manifest"][
            "checkpoint_sha256"
        ],
        "candidate_semantic_table_sha256": canonical_sha256(
            artifact["model_manifest"]["probabilities"]
        ),
        "query_bundle_path": "query-bundle/manifest.json",
        "query_bundle_manifest_sha256": query_bundle["manifest_sha256"],
        "query_bundle_checkpoint_sha256": query_bundle[
            "bundle_checkpoint_sha256"
        ],
        "query_job_count": len(query_jobs),
        "evaluation_bundle_path": "evaluation-bundle/manifest.json",
        "evaluation_bundle_manifest_sha256": evaluation_bundle[
            "manifest_sha256"
        ],
        "evaluation_bundle_checkpoint_sha256": evaluation_bundle[
            "bundle_checkpoint_sha256"
        ],
        "evaluation_job_count": len(evaluation_jobs),
        "restricted_hidden_information_assets": sorted(
            restricted_assets,
            key=lambda row: (
                row["kind"],
                row["root_id"],
                row["solver_seed"],
            ),
        ),
        "iteration_dispatch_model_sha256": iteration_dispatch.model_sha256,
    }
    round_manifest["manifest_sha256"] = canonical_sha256(round_manifest)
    _write_canonical_json(round_dir / "round.json", round_manifest)
    return _CandidateRound(
        candidate_index=candidate_index,
        artifact=MappingProxyType(copy.deepcopy(artifact)),
        query_bundle=MappingProxyType(copy.deepcopy(query_bundle)),
        query_jobs=MappingProxyType(dict(query_jobs)),
        evaluation_bundle=MappingProxyType(copy.deepcopy(evaluation_bundle)),
        evaluation_jobs=MappingProxyType(dict(evaluation_jobs)),
        round_manifest=MappingProxyType(copy.deepcopy(round_manifest)),
    )


def _excluded_partitions_v2(
    raw: _CalibrationRaw,
    *,
    evaluation_seeds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    natural_ids = sorted(
        {
            str(record["root_id"])
            for record in raw.natural.records
            if isinstance(record.get("root_id"), str)
        }
    )
    challenge_ids = sorted(
        {
            str(record["root_id"])
            for record in raw.challenge.records
            if isinstance(record.get("root_id"), str)
        }
    )
    if not natural_ids or not challenge_ids:
        raise AssertionError("behavior calibration partitions are empty")
    reserved = set(evaluation_seeds)
    excluded_seeds = (-20260701, -20260702, -20260703)
    if reserved & set(excluded_seeds):
        raise AssertionError("diagnostic evaluation/excluded seed overlap")
    return {
        "training": fixed_point_gate_v2.build_excluded_partition(
            "training",
            root_identity_commitments=[
                root_identity_commitment_sha256(root_id)
                for root_id in natural_ids
            ],
            solver_seeds=[excluded_seeds[0]],
        ),
        "calibration": fixed_point_gate_v2.build_excluded_partition(
            "calibration",
            root_identity_commitments=[
                root_identity_commitment_sha256(root_id)
                for root_id in challenge_ids
            ],
            solver_seeds=[excluded_seeds[1]],
        ),
        "smoke": fixed_point_gate_v2.build_excluded_partition(
            "smoke",
            root_identity_commitments=[
                root_identity_commitment_sha256(
                    f"{FIXTURE_ID}/excluded-production-smoke"
                )
            ],
            solver_seeds=[excluded_seeds[2]],
        ),
    }


def _encoded_policy(value: Mapping[str, float]) -> dict[str, str]:
    return fixed_point_gate_v2.encode_distribution(
        {
            action_id: Fraction(str(probability))
            for action_id, probability in value.items()
        }
    )


def _encoded_posterior(value: Mapping[str, Fraction]) -> dict[str, str]:
    return fixed_point_gate_v2.encode_distribution(
        {commitment: Fraction(weight) for commitment, weight in value.items()}
    )


def _raw_iteration_rows_v2(
    roots: Sequence[_Root],
    seeds: Sequence[int],
    rounds: Sequence[_CandidateRound],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for current_index in range(1, len(rounds)):
        previous = rounds[current_index - 1]
        current = rounds[current_index]
        round_index = current.candidate_index + 1
        if round_index != previous.candidate_index + 2:
            raise AssertionError("candidate state round chain is not contiguous")
        for seed in seeds:
            for root in sorted(roots, key=lambda item: item.root_id):
                previous_job = previous.evaluation_jobs[(root.root_id, seed)]
                current_job = current.evaluation_jobs[(root.root_id, seed)]
                gate_root = root.gate_record
                if set(previous_job.root_policy) != set(current_job.root_policy):
                    raise AssertionError("evaluation policy support drifted")
                if set(previous_job.posterior_weights) != set(
                    current_job.posterior_weights
                ):
                    raise AssertionError("fixed physical posterior support drifted")
                rows.append(
                    {
                        "round_index": round_index,
                        "solver_seed": seed,
                        "root_id": root.root_id,
                        "root_identity_commitment_sha256": gate_root[
                            "root_identity_commitment_sha256"
                        ],
                        "root_commitment_sha256": gate_root[
                            "root_commitment_sha256"
                        ],
                        "stratum": gate_root["stratum"],
                        "actor": gate_root["actor"],
                        "phase": gate_root["phase"],
                        "visible_joker_count": gate_root[
                            "visible_joker_count"
                        ],
                        "observation_digest": gate_root["observation_digest"],
                        "previous_candidate_query_checkpoint_bundle_sha256": (
                            previous.query_bundle["bundle_checkpoint_sha256"]
                        ),
                        "current_candidate_query_checkpoint_bundle_sha256": (
                            current.query_bundle["bundle_checkpoint_sha256"]
                        ),
                        "previous_evaluation_checkpoint_bundle_sha256": (
                            previous.evaluation_bundle["bundle_checkpoint_sha256"]
                        ),
                        "current_evaluation_checkpoint_bundle_sha256": (
                            current.evaluation_bundle["bundle_checkpoint_sha256"]
                        ),
                        "previous_candidate_policy_artifact_sha256": (
                            previous.artifact["artifact_sha256"]
                        ),
                        "current_candidate_policy_artifact_sha256": current.artifact[
                            "artifact_sha256"
                        ],
                        "previous_range_evidence_artifact_sha256": (
                            previous_job.range_evidence_sha256
                        ),
                        "current_range_evidence_artifact_sha256": (
                            current_job.range_evidence_sha256
                        ),
                        "previous_policy_distribution": _encoded_policy(
                            previous_job.root_policy
                        ),
                        "current_policy_distribution": _encoded_policy(
                            current_job.root_policy
                        ),
                        "previous_btn_posterior_weights": (
                            _encoded_posterior(previous_job.posterior_weights)
                            if root.actor == "btn"
                            else None
                        ),
                        "current_btn_posterior_weights": (
                            _encoded_posterior(current_job.posterior_weights)
                            if root.actor == "btn"
                            else None
                        ),
                        "exact_exploitability_computed": False,
                    }
                )
    return rows


def _restricted_artifact_map(
    output_dir: Path, rounds: Sequence[_CandidateRound]
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for round_ in rounds:
        for job in (*round_.query_jobs.values(), *round_.evaluation_jobs.values()):
            artifact, _audit = read_restricted_range_evidence(
                output_dir / job.range_evidence_relative_path
            )
            digest = artifact["artifact_sha256"]
            if digest in artifacts and artifacts[digest] != artifact:
                raise AssertionError("restricted range SHA-256 collision")
            artifacts[digest] = artifact
    return dict(sorted(artifacts.items()))


def _build_gate_artifacts_v2(
    *,
    output_dir: Path,
    workspace_root: Path,
    roots: Sequence[_Root],
    seeds: Sequence[int],
    rounds: Sequence[_CandidateRound],
    raw: _CalibrationRaw,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    solver_manifest: Mapping[str, Any],
    solver_manifest_sha256: str,
    range_builder_source_sha256: str,
    root_manifest: Mapping[str, Any],
    candidate_query_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    partitions = _excluded_partitions_v2(raw, evaluation_seeds=seeds)
    config = fixed_point_gate_v2.build_locked_t3_bb_fixed_point_gate_config(
        selected_promotion_seed=seeds[0],
        approved_source_manifest_sha256=source_manifest_sha256,
        approved_solver_manifest_sha256=solver_manifest_sha256,
        approved_range_builder_source_sha256=range_builder_source_sha256,
        approved_root_manifest_sha256=root_manifest["manifest_sha256"],
        approved_candidate_query_manifest_sha256=candidate_query_manifest[
            "manifest_sha256"
        ],
        approved_excluded_partition_sha256={
            purpose: partition["manifest_sha256"]
            for purpose, partition in partitions.items()
        },
    )
    query_bundles = {
        round_.query_bundle["bundle_checkpoint_sha256"]: copy.deepcopy(
            dict(round_.query_bundle)
        )
        for round_ in rounds
    }
    evaluation_bundles = {
        round_.evaluation_bundle["bundle_checkpoint_sha256"]: copy.deepcopy(
            dict(round_.evaluation_bundle)
        )
        for round_ in rounds
    }
    candidates = {
        round_.artifact["artifact_sha256"]: copy.deepcopy(dict(round_.artifact))
        for round_ in rounds
    }
    evidence: dict[str, Any] = {
        "schema": fixed_point_gate_v2.EVIDENCE_SCHEMA,
        "gate_id": fixed_point_gate_v2.GATE_ID,
        "scope": fixed_point_gate_v2.SCOPE,
        "evidence_kind": fixed_point_gate_v2.EVIDENCE_KIND,
        "production_promotion_claim": False,
        "gate_config_sha256": config["gate_config_sha256"],
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
        "candidate_policy_method": SOLVER_METHOD,
        "source_manifest": copy.deepcopy(dict(source_manifest)),
        "source_manifest_sha256": source_manifest_sha256,
        "solver_manifest": copy.deepcopy(dict(solver_manifest)),
        "solver_manifest_sha256": solver_manifest_sha256,
        "root_manifest": copy.deepcopy(dict(root_manifest)),
        "candidate_query_manifest": copy.deepcopy(
            dict(candidate_query_manifest)
        ),
        "excluded_partitions": partitions,
        "candidate_query_checkpoint_bundles": query_bundles,
        "evaluation_checkpoint_bundles": evaluation_bundles,
        "candidate_policy_artifacts": candidates,
        "restricted_range_artifacts": _restricted_artifact_map(
            output_dir, rounds
        ),
        "raw_iteration_rows": _raw_iteration_rows_v2(roots, seeds, rounds),
        "published_summary": {},
    }
    evidence["artifact_sha256"] = fixed_point_gate_v2.self_hash(
        evidence, "artifact_sha256"
    )
    metrics = fixed_point_gate_v2.derive_t3_bb_fixed_point_metrics(
        evidence,
        config=config,
        checkpoint_bundle_root=output_dir,
        workspace_root=workspace_root,
    )
    evidence["published_summary"] = metrics
    evidence["artifact_sha256"] = fixed_point_gate_v2.self_hash(
        evidence, "artifact_sha256"
    )
    result = fixed_point_gate_v2.validate_t3_bb_fixed_point_evidence(
        evidence,
        config=config,
        checkpoint_bundle_root=output_dir,
        workspace_root=workspace_root,
    )
    if result.get("passed") is not False or result.get("promotion_eligible") is not False:
        raise AssertionError("small v2 smoke unexpectedly passed production promotion")
    failures = "\n".join(result.get("failures", ()))
    for required in (
        "insufficient independent seeds",
        "insufficient roots per stratum",
        "insufficient transition rounds",
        "production promotion claim is false",
    ):
        if required not in failures:
            raise AssertionError(f"v2 smoke did not fail closed for {required}")
    try:
        fixed_point_gate_v2.build_t3_bb_likelihood_binding(
            evidence,
            config=config,
            gate_result=result,
            checkpoint_bundle_root=output_dir,
            workspace_root=workspace_root,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("small v2 smoke emitted a promotion binding")
    return evidence, config, result


def _write_gate_artifacts_v2(
    output_dir: Path,
    workspace_root: Path,
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    config_path = output_dir / "fixed-point-config.json"
    evidence_path = output_dir / "fixed-point-evidence.json"
    result_path = output_dir / "fixed-point-result.json"
    _write_canonical_json(config_path, config)
    _write_canonical_json(evidence_path, evidence)
    _write_canonical_json(result_path, result)
    verified = fixed_point_gate_v2.verify_t3_bb_fixed_point_gate_result(
        _read_canonical_json(evidence_path),
        config=_read_canonical_json(config_path),
        gate_result=_read_canonical_json(result_path),
        checkpoint_bundle_root=output_dir,
        workspace_root=workspace_root,
    )
    if verified != dict(result):
        raise AssertionError("v2 gate result readback changed")


def _validate_inputs(
    *,
    seeds: Sequence[int],
    transitions: int,
    iterations: int,
    max_particles: int,
    max_infosets: int,
    epsilon: Fraction,
) -> tuple[int, ...]:
    if (
        isinstance(seeds, (str, bytes))
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise TypeError("solver_seeds must be a sequence of integers")
    normalized = tuple(sorted(set(seeds)))
    if len(normalized) < 2:
        raise ValueError("fixed-point smoke requires at least two independent seeds")
    if transitions < 2:
        raise ValueError("fixed-point smoke requires at least two transitions")
    if iterations <= 0 or max_particles <= 0 or max_infosets <= 0:
        raise ValueError("iterations/max_particles/max_infosets must be positive")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    return normalized


def run_t3_bb_fixed_point_smoke(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    calibration_report_dir: str | Path = DEFAULT_CALIBRATION_REPORT,
    solver_seeds: Sequence[int] = DEFAULT_SOLVER_SEEDS,
    transitions: int = DEFAULT_TRANSITIONS,
    iterations: int = DEFAULT_ITERATIONS,
    max_particles: int = DEFAULT_MAX_PARTICLES,
    max_infosets: int = DEFAULT_MAX_INFOSETS,
    epsilon: Fraction | int | str = DEFAULT_EPSILON,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Execute, persist, and fresh-read the non-promoting smoke tree."""

    root = Path(workspace_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    calibration_dir = Path(calibration_report_dir)
    if not calibration_dir.is_absolute():
        calibration_dir = root / calibration_dir
    calibration_dir = calibration_dir.resolve()
    exact_epsilon = Fraction(epsilon)
    seeds = _validate_inputs(
        seeds=solver_seeds,
        transitions=transitions,
        iterations=iterations,
        max_particles=max_particles,
        max_infosets=max_infosets,
        epsilon=exact_epsilon,
    )
    if isinstance(rust_timeout_s, bool) or rust_timeout_s <= 0:
        raise ValueError("rust_timeout_s must be positive")

    output.mkdir(parents=True, exist_ok=True)
    bootstrap, raw = _load_t1_t2_bootstrap(root, calibration_dir)
    (
        source_manifest,
        solver_manifest,
        range_builder_sha,
        source_manifest_sha256,
        solver_manifest_sha256,
    ) = _source_manifests(root)
    _write_canonical_json(output / "source-manifest.json", source_manifest)
    _write_canonical_json(output / "solver-manifest.json", solver_manifest)
    bootstrap_manifest = dict(bootstrap.model_manifest)
    bootstrap_artifact = {
        "schema": "ofc_t3_bb_fixed_point_smoke_bootstrap/v1",
        "promotion_eligible": False,
        "raw_calibration_fully_reverified": True,
        "source_calibration_report_relative_path": DEFAULT_CALIBRATION_REPORT.as_posix(),
        "source_calibration_files_sha256": dict(raw.file_sha256),
        "calibration_artifact_sha256": raw.calibration["artifact_sha256"],
        "bootstrap_model_sha256": bootstrap.model_sha256,
        "bootstrap_model_manifest": bootstrap_manifest,
    }
    bootstrap_artifact["artifact_sha256"] = canonical_sha256(bootstrap_artifact)
    _write_canonical_json(output / "bootstrap.json", bootstrap_artifact)

    roots = _build_roots(
        rust_solver_path=(
            Path(rust_solver_path).resolve() if rust_solver_path is not None else None
        ),
        rust_timeout_s=float(rust_timeout_s),
    )
    root_observations = _root_observation_manifest(roots, seeds)
    _write_canonical_json(output / "roots.json", root_observations)
    production_root_manifest = fixed_point_gate_v2.build_root_manifest(
        [root.gate_payload() for root in roots], evaluation_seeds=seeds
    )
    _write_canonical_json(
        output / "fixed-point-root-manifest-v2.json", production_root_manifest
    )
    discovery = _T3BBQueryDiscoveryDispatch(bootstrap)
    queries, discovery_artifact = _discover_queries(
        roots,
        seeds,
        discovery,
        max_particles=max_particles,
        epsilon=exact_epsilon,
    )
    restricted_discovery = output / "restricted" / "query-discovery.json"
    _write_canonical_json(restricted_discovery, discovery_artifact)
    query_records, candidate_query_manifest = _build_candidate_query_manifest(
        queries,
        discovery_artifact,
        evaluation_root_manifest_sha256=production_root_manifest[
            "manifest_sha256"
        ],
    )
    _write_canonical_json(
        output / "candidate-query-manifest-v2.json", candidate_query_manifest
    )

    rounds: list[_CandidateRound] = []
    for candidate_index in range(transitions + 1):
        rounds.append(
            _candidate_round(
                output_dir=output,
                candidate_index=candidate_index,
                queries=queries,
                query_records=query_records,
                roots=roots,
                seeds=seeds,
                bootstrap=bootstrap,
                epsilon=exact_epsilon,
                max_particles=max_particles,
                iterations=iterations,
                max_infosets=max_infosets,
                solver_manifest_sha256=solver_manifest_sha256,
                source_manifest_sha256=source_manifest_sha256,
                range_builder_source_sha256=range_builder_sha,
            )
        )
    evidence, config, gate_result = _build_gate_artifacts_v2(
        output_dir=output,
        workspace_root=root,
        roots=roots,
        seeds=seeds,
        rounds=rounds,
        raw=raw,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        solver_manifest=solver_manifest,
        solver_manifest_sha256=solver_manifest_sha256,
        range_builder_source_sha256=range_builder_sha,
        root_manifest=production_root_manifest,
        candidate_query_manifest=candidate_query_manifest,
    )
    _write_gate_artifacts_v2(output, root, evidence, config, gate_result)

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "artifact_kind": "real_full_card_mccfr_fixed_point_wiring_smoke",
        "promotion_eligible": False,
        "fixed_point_promotion_passed": False,
        "fixed_point_converged_claimed": False,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "independent_strength_gate_required": True,
        "diagnostic_zero_drift_is_strength_evidence": False,
        "exact_exploitability_computed": False,
        "all_turn_ai_complete": False,
        "production_gate_applied": True,
        "production_gate_passed": False,
        "v2_fresh_asset_replay_completed": gate_result["derived_metrics"]
        is not None,
        "t3_bb_likelihood_binding_emitted": False,
        "source_calibration_promotion_eligible": False,
        "raw_calibration_fully_reverified": True,
        "initial_t3_policy_prior_used": False,
        "initial_t3_uniform_use_scope": "query_discovery_only",
        "missing_query_fallback_allowed": False,
        "fixed_particle_support": True,
        "solver_seeds": list(seeds),
        "transition_count": transitions,
        "candidate_count": len(rounds),
        "iterations_per_job": iterations,
        "max_particles": max_particles,
        "max_infosets": max_infosets,
        "epsilon": f"{exact_epsilon.numerator}/{exact_epsilon.denominator}",
        "strata": sorted(root.stratum for root in roots),
        "query_count": len(queries),
        "source_manifest_sha256": source_manifest_sha256,
        "solver_manifest_sha256": solver_manifest_sha256,
        "range_builder_source_sha256": range_builder_sha,
        "bootstrap_artifact_sha256": bootstrap_artifact["artifact_sha256"],
        "bootstrap_model_sha256": bootstrap.model_sha256,
        "root_observation_manifest_sha256": root_observations[
            "manifest_sha256"
        ],
        "fixed_point_root_manifest_v2_sha256": production_root_manifest[
            "manifest_sha256"
        ],
        "candidate_query_manifest_v2_sha256": candidate_query_manifest[
            "manifest_sha256"
        ],
        "query_discovery_artifact_sha256": discovery_artifact[
            "artifact_sha256"
        ],
        "candidate_chain": [
            {
                "candidate_index": round_.candidate_index,
                "candidate_policy_artifact_sha256": round_.artifact[
                    "artifact_sha256"
                ],
                "candidate_policy_checkpoint_sha256": round_.artifact[
                    "model_manifest"
                ]["checkpoint_sha256"],
                "round_manifest_sha256": round_.round_manifest[
                    "manifest_sha256"
                ],
                "query_bundle_manifest_sha256": round_.query_bundle[
                    "manifest_sha256"
                ],
                "evaluation_bundle_manifest_sha256": round_.evaluation_bundle[
                    "manifest_sha256"
                ],
            }
            for round_ in rounds
        ],
        "candidate_semantic_tables_identical": len(
            {
                round_.round_manifest["candidate_semantic_table_sha256"]
                for round_ in rounds
            }
        )
        == 1,
        "fixed_point_gate_schema": fixed_point_gate_v2.EVIDENCE_SCHEMA,
        "fixed_point_config_sha256": config["gate_config_sha256"],
        "fixed_point_evidence_sha256": evidence["artifact_sha256"],
        "fixed_point_gate_result_sha256": gate_result["gate_result_sha256"],
        "fixed_point_gate_failures": list(gate_result["failures"]),
        "production_minima": copy.deepcopy(
            gate_result["derived_metrics"]["production_minima"]
        ),
        "diagnostic_round_metrics": copy.deepcopy(
            gate_result["derived_metrics"]["round_metrics"]
        ),
        "raw_iteration_row_count": len(evidence["raw_iteration_rows"]),
        "restricted_hidden_information_present": True,
        "restricted_hidden_information_public_policy_input": False,
        "atomic_write_readback_verified": True,
    }
    report["report_sha256"] = canonical_sha256(report)
    _write_canonical_json(output / "result.json", report)
    verified = verify_t3_bb_fixed_point_smoke_output(
        output, calibration_report_dir=calibration_dir, workspace_root=root
    )
    if verified != report:
        raise AssertionError("final smoke report readback changed")
    return report


def verify_t3_bb_fixed_point_smoke_output(
    output_dir: str | Path,
    *,
    calibration_report_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fresh-read every published boundary and reject any promotion claim."""

    output = Path(output_dir).resolve()
    workspace = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    report = _read_canonical_json(output / "result.json")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("fixed-point smoke report schema mismatch")
    claimed = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("fixed-point smoke report hash mismatch")
    required_false = (
        "promotion_eligible",
        "fixed_point_promotion_passed",
        "fixed_point_converged_claimed",
        "strategic_strength_evaluated",
        "strategic_strength_claimed",
        "diagnostic_zero_drift_is_strength_evidence",
        "exact_exploitability_computed",
        "all_turn_ai_complete",
        "production_gate_passed",
        "t3_bb_likelihood_binding_emitted",
        "source_calibration_promotion_eligible",
        "initial_t3_policy_prior_used",
        "missing_query_fallback_allowed",
        "restricted_hidden_information_public_policy_input",
    )
    if any(report.get(field) is not False for field in required_false):
        raise ValueError("fixed-point smoke contains a forbidden positive claim")
    if report.get("raw_calibration_fully_reverified") is not True:
        raise ValueError("fixed-point smoke did not reverify raw calibration")
    if report.get("independent_strength_gate_required") is not True:
        raise ValueError("fixed-point smoke weakened the independent strength gate")
    if report.get("v2_fresh_asset_replay_completed") is not True:
        raise ValueError("fixed-point smoke did not complete the v2 asset replay")
    if report.get("initial_t3_uniform_use_scope") != "query_discovery_only":
        raise ValueError("initial T3 uniform scope drift")
    if report.get("atomic_write_readback_verified") is not True:
        raise ValueError("fixed-point smoke lacks readback proof")

    source = _read_canonical_json(output / "source-manifest.json")
    solver = _read_canonical_json(output / "solver-manifest.json")
    bootstrap = _read_canonical_json(output / "bootstrap.json")
    roots = _read_canonical_json(output / "roots.json")
    production_roots = _read_canonical_json(
        output / "fixed-point-root-manifest-v2.json"
    )
    discovery = _read_canonical_json(output / "restricted" / "query-discovery.json")
    candidate_query_manifest = _read_canonical_json(
        output / "candidate-query-manifest-v2.json"
    )
    if canonical_sha256(source) != report["source_manifest_sha256"]:
        raise ValueError("published source manifest hash mismatch")
    if canonical_sha256(solver) != report["solver_manifest_sha256"]:
        raise ValueError("published solver manifest hash mismatch")
    for value, field, expected in (
        (bootstrap, "artifact_sha256", report["bootstrap_artifact_sha256"]),
        (roots, "manifest_sha256", report["root_observation_manifest_sha256"]),
        (
            production_roots,
            "manifest_sha256",
            report["fixed_point_root_manifest_v2_sha256"],
        ),
        (
            candidate_query_manifest,
            "manifest_sha256",
            report["candidate_query_manifest_v2_sha256"],
        ),
        (
            discovery,
            "artifact_sha256",
            report["query_discovery_artifact_sha256"],
        ),
    ):
        if value.get(field) != expected:
            raise ValueError(f"published {field} does not match report")
        content = dict(value)
        content.pop(field, None)
        if canonical_sha256(content) != expected:
            raise ValueError(f"published {field} self hash mismatch")

    if calibration_report_dir is not None:
        raw = _load_calibration_raw(Path(calibration_report_dir).resolve())
        if raw.calibration["artifact_sha256"] != bootstrap[
            "calibration_artifact_sha256"
        ]:
            raise ValueError("bootstrap calibration artifact drift")
        if dict(raw.file_sha256) != bootstrap["source_calibration_files_sha256"]:
            raise ValueError("bootstrap raw calibration file drift")

    seeds = tuple(report["solver_seeds"])
    root_rows = roots.get("roots")
    if not isinstance(root_rows, list) or len(root_rows) != 6:
        raise ValueError("root observation manifest must contain six roots")
    root_by_id = {row["root_id"]: row for row in root_rows}
    discovery_query_rows = discovery.get("queries")
    if (
        not isinstance(discovery_query_rows, list)
        or len(discovery_query_rows) != report["query_count"]
    ):
        raise ValueError("query discovery count mismatch")
    query_rows = candidate_query_manifest.get("queries")
    if not isinstance(query_rows, list) or len(query_rows) != report["query_count"]:
        raise ValueError("candidate query manifest count mismatch")
    query_ids = {
        row["behavior_information_digest"]: row for row in query_rows
    }
    if len(query_ids) != len(query_rows):
        raise ValueError("query discovery contains duplicate digests")

    chain = report.get("candidate_chain")
    if not isinstance(chain, list) or len(chain) != report["candidate_count"]:
        raise ValueError("candidate chain count mismatch")
    for expected_index, link in enumerate(chain):
        if link.get("candidate_index") != expected_index:
            raise ValueError("candidate chain index gap")
        round_dir = output / f"candidate-{expected_index:03d}"
        round_manifest = _read_canonical_json(round_dir / "round.json")
        if round_manifest.get("manifest_sha256") != link[
            "round_manifest_sha256"
        ]:
            raise ValueError("round manifest hash mismatch")
        round_unsigned = dict(round_manifest)
        round_unsigned.pop("manifest_sha256", None)
        if canonical_sha256(round_unsigned) != round_manifest["manifest_sha256"]:
            raise ValueError("round manifest self hash mismatch")
        candidate = read_t3_bb_candidate_policy_artifact(
            round_dir / round_manifest["candidate_policy_artifact_path"]
        )
        if candidate["artifact_sha256"] != link[
            "candidate_policy_artifact_sha256"
        ]:
            raise ValueError("candidate policy hash mismatch")
        if set(candidate["model_manifest"]["probabilities"]) != set(query_ids):
            raise ValueError("candidate policy query coverage gap")

        query_keys = [
            T3BBCheckpointKey(
                round_index=expected_index + 1,
                root_id=row["query_id"],
                root_commitment_sha256=row["query_commitment_sha256"],
                solver_seed=seed,
            )
            for digest, row in sorted(query_ids.items())
            for seed in seeds
        ]
        query_bundle = read_t3_bb_checkpoint_bundle(
            round_dir / round_manifest["query_bundle_path"],
            bundle_root=output,
            expected_keys=query_keys,
        )
        if query_bundle["manifest_sha256"] != link[
            "query_bundle_manifest_sha256"
        ]:
            raise ValueError("query bundle hash mismatch")
        if candidate["model_manifest"]["checkpoint_sha256"] != query_bundle[
            "bundle_checkpoint_sha256"
        ]:
            raise ValueError("candidate policy is not bound to query bundle")
        evaluation_keys = [
            T3BBCheckpointKey(
                round_index=expected_index + 1,
                root_id=root_id,
                root_commitment_sha256=row["root_commitment_sha256"],
                solver_seed=seed,
            )
            for root_id, row in sorted(root_by_id.items())
            for seed in seeds
        ]
        evaluation_bundle = read_t3_bb_checkpoint_bundle(
            round_dir / round_manifest["evaluation_bundle_path"],
            bundle_root=output,
            expected_keys=evaluation_keys,
        )
        if evaluation_bundle["manifest_sha256"] != link[
            "evaluation_bundle_manifest_sha256"
        ]:
            raise ValueError("evaluation bundle hash mismatch")
        for asset in round_manifest["restricted_hidden_information_assets"]:
            restricted, _audit = read_restricted_range_evidence(
                output / asset["relative_path"]
            )
            if restricted["artifact_sha256"] != asset["artifact_sha256"]:
                raise ValueError("restricted range artifact hash mismatch")

    evidence = _read_canonical_json(output / "fixed-point-evidence.json")
    config = _read_canonical_json(output / "fixed-point-config.json")
    gate_result = _read_canonical_json(output / "fixed-point-result.json")
    verified_gate = fixed_point_gate_v2.verify_t3_bb_fixed_point_gate_result(
        evidence,
        config=config,
        gate_result=gate_result,
        checkpoint_bundle_root=output,
        workspace_root=workspace,
    )
    if (
        verified_gate.get("passed") is not False
        or verified_gate.get("promotion_eligible") is not False
    ):
        raise ValueError("diagnostic fixed-point gate unexpectedly passed")
    if evidence["artifact_sha256"] != report["fixed_point_evidence_sha256"]:
        raise ValueError("fixed-point evidence hash mismatch")
    if config["gate_config_sha256"] != report["fixed_point_config_sha256"]:
        raise ValueError("fixed-point config hash mismatch")
    if gate_result["gate_result_sha256"] != report[
        "fixed_point_gate_result_sha256"
    ]:
        raise ValueError("fixed-point gate result hash mismatch")
    try:
        fixed_point_gate_v2.build_t3_bb_likelihood_binding(
            evidence,
            config=config,
            gate_result=gate_result,
            checkpoint_bundle_root=output,
            workspace_root=workspace,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise ValueError("diagnostic fixed-point output can emit a binding")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--calibration-report-dir", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument(
        "--solver-seed", type=int, action="append", dest="solver_seeds"
    )
    parser.add_argument("--transitions", type=int, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--max-particles", type=int, default=DEFAULT_MAX_PARTICLES)
    parser.add_argument("--max-infosets", type=int, default=DEFAULT_MAX_INFOSETS)
    parser.add_argument("--epsilon", default="1/1000")
    parser.add_argument("--rust-solver-path", type=Path, default=None)
    parser.add_argument("--rust-timeout-s", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_t3_bb_fixed_point_smoke(
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        calibration_report_dir=args.calibration_report_dir,
        solver_seeds=(
            tuple(args.solver_seeds)
            if args.solver_seeds is not None
            else DEFAULT_SOLVER_SEEDS
        ),
        transitions=args.transitions,
        iterations=args.iterations,
        max_particles=args.max_particles,
        max_infosets=args.max_infosets,
        epsilon=Fraction(args.epsilon),
        rust_solver_path=args.rust_solver_path,
        rust_timeout_s=args.rust_timeout_s,
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CALIBRATION_REPORT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOLVER_SEEDS",
    "REPORT_SCHEMA",
    "run_t3_bb_fixed_point_smoke",
    "verify_t3_bb_fixed_point_smoke_output",
]
