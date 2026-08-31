from __future__ import annotations

import copy
import hashlib
from fractions import Fraction
from functools import lru_cache

import pytest

import ai.tutor.t3_bb_fixed_point_gate_v2 as gate
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import action_key
from ai.tutor.promotion_gate_m3_full_card_strength import (
    REQUIRED_STRATA,
    SOURCE_SCHEMA,
    SOLVER_ADAPTER,
    SOLVER_METHOD,
    SOLVER_SCHEMA,
)
from ai.tutor.t3_bb_checkpoint_bundle import T3BBCheckpointKey
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture


def _sha(label: str) -> str:
    return gate.canonical_sha256({"fixture": label})


@lru_cache(maxsize=None)
def _observation(actor: str, joker: int):
    compiled = compile_canonical_reduced_fixture(actor, joker)
    return compiled.root.branches[0].child.state.infoset_key


def _behavior(key) -> BehaviorInfoSet:
    rows = key.board_bb
    board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
    legal = tuple(
        sorted(action_key(action) for action in get_turn_actions(list(key.current_draw), board))
    )
    return BehaviorInfoSet(
        actor="bb",
        turn=3,
        board_bb=key.board_bb,
        board_btn=key.board_btn,
        public_action_history=key.public_action_history,
        own_recall_before=key.own_recall,
        current_draw=key.current_draw,
        legal_action_ids=legal,
        fantasy_state=key.fantasy_state,
    )


@lru_cache(maxsize=1)
def _production_roots():
    roots = []
    for stratum in REQUIRED_STRATA:
        actor, joker_text = stratum.split("_joker")
        observation = _observation(actor, int(joker_text))
        for index in range(gate.MIN_ROOTS_PER_STRATUM):
            roots.append(
                gate.build_root_record(
                    f"v2-production-{stratum}-{index:03d}", observation
                )
            )
    return tuple(roots)


@lru_cache(maxsize=1)
def _root_manifest():
    return gate.build_root_manifest(
        _production_roots(), evaluation_seeds=(101, 202, 303)
    )


@lru_cache(maxsize=1)
def _root_state():
    return gate._validate_root_manifest(_root_manifest())


@lru_cache(maxsize=1)
def _query_manifest():
    roots = _production_roots()
    queries = []
    for joker in range(3):
        sources = [
            root["root_commitment_sha256"]
            for root in roots
            if root["actor"] == "btn" and root["visible_joker_count"] == joker
        ]
        queries.append(
            gate.build_candidate_query_record(
                f"candidate-query-joker{joker}",
                _behavior(_observation("bb", joker)),
                source_root_commitments=sources,
            )
        )
    return gate.build_candidate_query_manifest(
        queries,
        evaluation_root_manifest_sha256=_root_manifest()["manifest_sha256"],
    )


def _partition(purpose: str, seed: int):
    return gate.build_excluded_partition(
        purpose,
        root_identity_commitments=[_sha(f"excluded-root-{purpose}")],
        solver_seeds=[seed],
    )


def _config():
    partitions = {
        purpose: _partition(purpose, index + 1)
        for index, purpose in enumerate(("training", "calibration", "smoke"))
    }
    config = gate.build_locked_t3_bb_fixed_point_gate_config(
        selected_promotion_seed=101,
        approved_source_manifest_sha256=_sha("source-manifest"),
        approved_solver_manifest_sha256=_sha("solver-manifest"),
        approved_range_builder_source_sha256=_sha("range-source"),
        approved_root_manifest_sha256=_root_manifest()["manifest_sha256"],
        approved_candidate_query_manifest_sha256=_query_manifest()["manifest_sha256"],
        approved_excluded_partition_sha256={
            purpose: partition["manifest_sha256"]
            for purpose, partition in partitions.items()
        },
    )
    return config, partitions


def test_production_config_has_unweakenable_exact_rational_minima():
    config, _partitions = _config()

    assert config["thresholds"] == {
        "min_independent_seeds": 3,
        "min_roots_per_stratum": 100,
        "min_consecutive_converged_rounds": 3,
        "max_policy_tv": "1/100",
        "max_btn_posterior_weight_tv": "1/100",
        "max_cross_seed_policy_tv": "1/100",
        "max_cross_seed_btn_posterior_weight_tv": "1/100",
    }
    assert config["strategic_strength_evaluated"] is False
    assert config["requires_independent_strength_gate"] is True
    assert config["cold_start_zero_drift_is_strength_evidence"] is False

    weakened = copy.deepcopy(config)
    weakened["thresholds"]["max_policy_tv"] = "1/1"
    weakened["gate_config_sha256"] = gate.self_hash(
        weakened, "gate_config_sha256"
    )
    with pytest.raises(ValueError, match="production constant mismatch"):
        gate.verify_locked_t3_bb_fixed_point_gate_config(weakened)

    smoke = copy.deepcopy(config)
    smoke["thresholds"]["min_independent_seeds"] = 2
    smoke["gate_config_sha256"] = gate.self_hash(smoke, "gate_config_sha256")
    with pytest.raises(ValueError, match="production constant mismatch"):
        gate.verify_locked_t3_bb_fixed_point_gate_config(smoke)


def test_root_manifest_reconstructs_infosets_actor_phase_joker_and_both_commitments():
    state = _root_state()

    assert state["counts"] == {stratum: 100 for stratum in REQUIRED_STRATA}
    assert len(state["identities"]) == 600
    assert len(state["commitments"]) == 600
    for root in state["by_id"].values():
        key = gate.reconstruct_infoset_key(root["observation"])
        assert key.digest() == root["observation_digest"]
        assert root["phase"] == ("t3_first" if root["actor"] == "bb" else "t3_second")
        assert gate.visible_joker_count(key) == root["visible_joker_count"]

    tampered = copy.deepcopy(_root_manifest())
    tampered["roots"][0]["actor"] = "btn"
    tampered["roots"][0]["root_commitment_sha256"] = gate.root_commitment_sha256(
        tampered["roots"][0]
    )
    tampered["manifest_sha256"] = gate.self_hash(tampered, "manifest_sha256")
    with pytest.raises(ValueError, match="reconstructed value mismatch"):
        gate._validate_root_manifest(tampered)


def test_candidate_query_mapping_is_reconstructed_and_table_is_seed_aggregate_q32():
    root_state = _root_state()
    query_state = gate._validate_candidate_query_manifest(
        _query_manifest(), root_state=root_state
    )
    bundle_sha = _sha("bundle-round-4")
    profiles = {}
    for seed in root_state["seeds"]:
        for query in query_state["by_behavior_digest"].values():
            key = T3BBCheckpointKey(
                round_index=4,
                root_id=query["query_id"],
                root_commitment_sha256=query["query_commitment_sha256"],
                solver_seed=seed,
            )
            legal = query["behavior_information"]["legal_action_ids"]
            # Distinct exact decimal-rational seed profiles are averaged before
            # deterministic Q32 quantization.
            numerator = {101: 49, 202: 50, 303: 51}[seed]
            if len(legal) == 1:
                distribution = {legal[0]: Fraction(1, 1)}
            else:
                first = Fraction(numerator, 100)
                tail = (1 - first) / (len(legal) - 1)
                distribution = {legal[0]: first, **{item: tail for item in legal[1:]}}
            profiles.setdefault(key, {})[
                query["converted_observation_digest"]
            ] = distribution

    table = {}
    for digest, query in query_state["by_behavior_digest"].items():
        legal = query["behavior_information"]["legal_action_ids"]
        if len(legal) == 1:
            average = {legal[0]: Fraction(1, 1)}
        else:
            average = {
                legal[0]: Fraction(1, 2),
                **{
                    item: Fraction(1, 2 * (len(legal) - 1))
                    for item in legal[1:]
                },
            }
        table[digest] = gate.encode_distribution(
            gate.quantize_exact_distribution_q32(average)
        )
    artifact_sha = _sha("candidate-round-4")
    artifacts = {
        artifact_sha: {
            "artifact_sha256": artifact_sha,
            "model_manifest": {
                "checkpoint_sha256": bundle_sha,
                "probabilities": table,
            },
        }
    }
    gate._verify_candidate_query_tables(
        artifacts=artifacts,
        bundles={bundle_sha: {"round_index": 4}},
        profiles={bundle_sha: profiles},
        query_state=query_state,
        root_state=root_state,
    )

    tampered = copy.deepcopy(artifacts)
    digest = next(iter(table))
    actions = list(tampered[artifact_sha]["model_manifest"]["probabilities"][digest])
    tampered[artifact_sha]["model_manifest"]["probabilities"][digest] = {
        action: ("1/1" if index == 0 else "0/1")
        for index, action in enumerate(actions)
    }
    with pytest.raises(ValueError, match="seed-aggregated Q32"):
        gate._verify_candidate_query_tables(
            artifacts=tampered,
            bundles={bundle_sha: {"round_index": 4}},
            profiles={bundle_sha: profiles},
            query_state=query_state,
            root_state=root_state,
        )


def test_candidate_artifact_map_calls_runtime_content_verifier(monkeypatch):
    candidate_sha = _sha("candidate")
    solver_sha = _sha("solver")
    source_sha = _sha("range")
    artifact = {
        "artifact_sha256": candidate_sha,
        "model_manifest": {
            "solver_manifest_sha256": solver_sha,
            "range_builder_source_sha256": source_sha,
        },
    }
    calls = []

    def fake_verify(value):
        calls.append(value)
        return copy.deepcopy(value)

    monkeypatch.setattr(gate, "verify_t3_bb_candidate_policy_artifact", fake_verify)
    assert gate._candidate_artifacts(
        {candidate_sha: artifact},
        solver_sha=solver_sha,
        range_source_sha=source_sha,
    ) == {candidate_sha: artifact}
    assert calls == [artifact]


def test_query_bundle_uses_exact_query_keys_and_bootstrap_ranges(monkeypatch, tmp_path):
    root_state = _root_state()
    query_state = gate._validate_candidate_query_manifest(
        _query_manifest(), root_state=root_state
    )
    round_index = 1
    bundle_sha = _sha("query-bundle-content")
    solver_sha = _sha("solver-manifest")
    source_sha = _sha("source-manifest")
    entries = []
    ranges = {}
    profiles = {}
    for seed in root_state["seeds"]:
        for query in query_state["by_behavior_digest"].values():
            content_sha = _sha(f"query-range-content-{seed}-{query['query_id']}")
            build_sha = _sha(f"query-range-build-{seed}-{query['query_id']}")
            entries.append(
                {
                    "round_index": round_index,
                    "root_id": query["query_id"],
                    "root_commitment_sha256": query[
                        "query_commitment_sha256"
                    ],
                    "solver_seed": seed,
                    "observation_digest": query[
                        "converted_observation_digest"
                    ],
                    "range_content_sha256": content_sha,
                    "range_build_sha256": build_sha,
                    "solver_manifest_sha256": solver_sha,
                    "source_manifest_sha256": source_sha,
                }
            )
            ranges[_sha(f"query-range-artifact-{seed}-{query['query_id']}")] = {
                "root_id": query["query_id"],
                "root_commitment_sha256": query["query_commitment_sha256"],
                "round_index": round_index,
                "solver_seed": seed,
                "observation_digest": query["converted_observation_digest"],
                "range_content_sha256": content_sha,
                "range_build_sha256": build_sha,
                "behavior_model_manifest": {
                    "model_type": gate.BOOTSTRAP_MODEL_TYPE,
                    "promotion_eligible": False,
                    "fixed_point_bootstrap_only": True,
                    "no_fallback": True,
                    "t3_bb_route_included": False,
                    "t3_bb_likelihood_binding_included": False,
                    "strategic_strength_evaluated": False,
                    "strategic_strength_claimed": False,
                },
            }
            key = T3BBCheckpointKey(
                round_index=round_index,
                root_id=query["query_id"],
                root_commitment_sha256=query["query_commitment_sha256"],
                solver_seed=seed,
            )
            legal = query["behavior_information"]["legal_action_ids"]
            profiles[key] = {
                query["converted_observation_digest"]: {
                    action: Fraction(1, len(legal)) for action in legal
                }
            }
    manifest = {
        "round_index": round_index,
        "bundle_checkpoint_sha256": bundle_sha,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "entries": entries,
    }
    monkeypatch.setattr(
        gate, "verify_t3_bb_checkpoint_bundle", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(
        gate,
        "load_verified_t3_bb_checkpoint_strategy_profiles",
        lambda *args, **kwargs: profiles,
    )

    verified, loaded = gate._candidate_query_checkpoint_bundles(
        {bundle_sha: manifest},
        bundle_root=tmp_path,
        round_indices={1},
        query_state=query_state,
        root_state=root_state,
        source_sha=source_sha,
        solver_sha=solver_sha,
        ranges=ranges,
    )
    assert verified == {bundle_sha: manifest}
    assert loaded == {bundle_sha: profiles}

    tampered = copy.deepcopy(ranges)
    next(iter(tampered.values()))["behavior_model_manifest"][
        "model_type"
    ] = gate.ITERATION_MODEL_TYPE
    with pytest.raises(ValueError, match="bootstrap manifest model_type"):
        gate._candidate_query_checkpoint_bundles(
            {bundle_sha: manifest},
            bundle_root=tmp_path,
            round_indices={1},
            query_state=query_state,
            root_state=root_state,
            source_sha=source_sha,
            solver_sha=solver_sha,
            ranges=tampered,
        )


def test_metrics_recompute_every_same_root_seed_pair_for_policy_and_btn_posterior():
    root_state = {
        "seeds": (101, 202, 303),
        "by_id": {
            "bb": {"actor": "bb"},
            "btn": {"actor": "btn"},
        },
        "counts": {stratum: 100 for stratum in REQUIRED_STRATA},
    }
    rows = []
    policies = {
        101: {"a": Fraction(1, 2), "b": Fraction(1, 2)},
        202: {"a": Fraction(51, 100), "b": Fraction(49, 100)},
        303: {"a": Fraction(52, 100), "b": Fraction(48, 100)},
    }
    for root_id in ("bb", "btn"):
        for seed in root_state["seeds"]:
            posterior = policies[seed] if root_id == "btn" else None
            rows.append(
                {
                    "round_index": 2,
                    "root_id": root_id,
                    "solver_seed": seed,
                    "actor": root_id,
                    "_previous_policy": policies[seed],
                    "_current_policy": policies[seed],
                    "_previous_posterior": posterior,
                    "_current_posterior": posterior,
                }
            )
    metrics = gate._derive_metrics(rows, rounds=(2,), root_state=root_state)
    round_metrics = metrics["round_metrics"][0]

    assert round_metrics["cross_seed_policy_pair_count"] == 6
    assert round_metrics["cross_seed_btn_posterior_pair_count"] == 3
    assert round_metrics["max_same_root_cross_seed_policy_tv"] == "1/50"
    assert round_metrics["max_same_root_cross_seed_btn_posterior_tv"] == "1/50"
    assert round_metrics["converged"] is False


def test_insufficient_manifest_and_failure_result_can_never_build_binding(monkeypatch, tmp_path):
    roots = [
        gate.build_root_record(
            f"small-{stratum}",
            _observation(*(
                (stratum.split("_joker")[0], int(stratum.split("_joker")[1]))
            )),
        )
        for stratum in REQUIRED_STRATA
    ]
    diagnostic_manifest = gate.build_root_manifest(
        roots, evaluation_seeds=(1, 2, 3)
    )
    with pytest.raises(ValueError, match="at least 100 roots"):
        gate._validate_root_manifest(diagnostic_manifest)

    monkeypatch.setattr(
        gate,
        "verify_t3_bb_fixed_point_gate_result",
        lambda *args, **kwargs: {
            "passed": False,
            "promotion_eligible": False,
            "production_minima_enforced": True,
            "exact_exploitability_computed": False,
        },
    )
    with pytest.raises(ValueError, match="did not pass production"):
        gate.build_t3_bb_likelihood_binding(
            {},
            config={},
            gate_result={},
            checkpoint_bundle_root=tmp_path,
            workspace_root=tmp_path,
        )


def test_source_manifest_fresh_hash_is_required(tmp_path):
    workspace = tmp_path / "workspace"
    source_file = workspace / gate.T3_FULL_CARD_RANGE_SOURCE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"range-source-v1\n")
    source_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    source_manifest = {
        "schema": SOURCE_SCHEMA,
        "files": [
            {"path": gate.T3_FULL_CARD_RANGE_SOURCE_PATH, "sha256": source_digest}
        ],
    }
    source_sha = gate.canonical_sha256(source_manifest)
    solver_manifest = {
        "schema": SOLVER_SCHEMA,
        "method": SOLVER_METHOD,
        "adapter": SOLVER_ADAPTER,
        "ruleset": gate.RULESET,
        "position_contract_version": gate.POSITION_CONTRACT_VERSION,
        "turns": [3, 4],
        "actors": ["bb", "btn"],
        "physical_joker_ids": ["X1", "X2"],
        "information_model": gate.INFORMATION_MODEL,
        "strategy_fusion": False,
        "full_card": True,
        "hu_exact": False,
        "exact_exploitability_computed": False,
        "source_manifest_sha256": source_sha,
    }
    solver_sha = gate.canonical_sha256(solver_manifest)
    evidence = {
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_sha,
        "solver_manifest": solver_manifest,
        "solver_manifest_sha256": solver_sha,
    }
    config = {
        "approved_source_manifest_sha256": source_sha,
        "approved_solver_manifest_sha256": solver_sha,
        "approved_range_builder_source_sha256": source_digest,
    }

    assert gate._validate_source_and_solver(
        evidence, config=config, workspace_root=workspace
    ) == (source_sha, solver_sha, source_digest)
    source_file.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="fresh bytes hash mismatch"):
        gate._validate_source_and_solver(
            evidence, config=config, workspace_root=workspace
        )


@lru_cache(maxsize=1)
def _mock_production_payload():
    """Large enough to exercise the real production minima, with I/O mocked later."""

    config, partitions = _config()
    root_state = _root_state()
    query_state = gate._validate_candidate_query_manifest(
        _query_manifest(), root_state=root_state
    )
    evaluation_bundles = {}
    evaluation_profiles_by_bundle = {}
    query_bundles = {}
    query_profiles_by_bundle = {}
    candidates = {}
    ranges = {}
    audits = {}
    evaluation_bundle_sha_by_round = {
        round_index: _sha(f"evaluation-bundle-{round_index}")
        for round_index in range(1, 5)
    }
    query_bundle_sha_by_round = {
        round_index: _sha(f"query-bundle-{round_index}")
        for round_index in range(1, 5)
    }
    candidate_sha_by_round = {
        round_index: _sha(f"candidate-{round_index}") for round_index in range(1, 5)
    }
    query_strategy_rows = {}
    candidate_table = {}
    for behavior_digest, query in query_state["by_behavior_digest"].items():
        legal = query["behavior_information"]["legal_action_ids"]
        uniform = {action: Fraction(1, len(legal)) for action in legal}
        query_strategy_rows[query["converted_observation_digest"]] = uniform
        candidate_table[behavior_digest] = gate.encode_distribution(
            gate.quantize_exact_distribution_q32(uniform)
        )

    range_sha_by_key = {}
    encoded_policy_by_root = {}
    exact_policy_by_root = {}
    for root_id, root in root_state["by_id"].items():
        legal = root_state["legal_by_id"][root_id]
        exact = {action: Fraction(1, len(legal)) for action in legal}
        exact_policy_by_root[root_id] = exact
        encoded_policy_by_root[root_id] = gate.encode_distribution(exact)

    for round_index in range(1, 5):
        evaluation_bundle_sha = evaluation_bundle_sha_by_round[round_index]
        query_bundle_sha = query_bundle_sha_by_round[round_index]
        candidate_sha = candidate_sha_by_round[round_index]
        candidates[candidate_sha] = {
            "artifact_sha256": candidate_sha,
            "model_manifest": {
                "checkpoint_sha256": query_bundle_sha,
                "probabilities": copy.deepcopy(candidate_table),
            },
        }
        evaluation_bundle_entries = []
        evaluation_profiles = {}
        query_bundle_entries = []
        round_query_profiles = {}
        for seed in root_state["seeds"]:
            for query in query_state["by_behavior_digest"].values():
                query_key = T3BBCheckpointKey(
                    round_index=round_index,
                    root_id=query["query_id"],
                    root_commitment_sha256=query["query_commitment_sha256"],
                    solver_seed=seed,
                )
                round_query_profiles[query_key] = {
                    query["converted_observation_digest"]: copy.deepcopy(
                        query_strategy_rows[query["converted_observation_digest"]]
                    )
                }
                query_bundle_entries.append(
                    {
                        "round_index": round_index,
                        "root_id": query["query_id"],
                        "root_commitment_sha256": query[
                            "query_commitment_sha256"
                        ],
                        "solver_seed": seed,
                    }
                )
        for seed in root_state["seeds"]:
            for root_id, root in sorted(root_state["by_id"].items()):
                key = T3BBCheckpointKey(
                    round_index=round_index,
                    root_id=root_id,
                    root_commitment_sha256=root["root_commitment_sha256"],
                    solver_seed=seed,
                )
                profile = {}
                profile[root["observation_digest"]] = exact_policy_by_root[root_id]
                evaluation_profiles[key] = profile
                range_sha = _sha(f"range-{round_index}-{seed}-{root_id}")
                range_content = _sha(f"range-content-{round_index}-{seed}-{root_id}")
                range_build = _sha(f"range-build-{round_index}-{seed}-{root_id}")
                range_sha_by_key[(round_index, seed, root_id)] = range_sha
                ranges[range_sha] = {
                    "root_id": root_id,
                    "root_commitment_sha256": root["root_commitment_sha256"],
                    "round_index": round_index,
                    "solver_seed": seed,
                    "observation_digest": root["observation_digest"],
                    "range_content_sha256": range_content,
                    "range_build_sha256": range_build,
                    "behavior_model_sha256": _sha(
                        f"iteration-model-{round_index}"
                    ),
                    "behavior_model_manifest": {
                        "model_type": gate.ITERATION_MODEL_TYPE,
                        "promotion_eligible": False,
                        "no_fallback": True,
                        "fixed_point_iteration_only": True,
                        "fixed_point_converged": False,
                        "t3_bb_candidate_policy_artifact_sha256": candidate_sha,
                    },
                }
                audits[range_sha] = {"posterior_weights": {"particle": "1/1"}}
                evaluation_bundle_entries.append(
                    {
                        "round_index": round_index,
                        "root_id": root_id,
                        "root_commitment_sha256": root["root_commitment_sha256"],
                        "solver_seed": seed,
                        "range_content_sha256": range_content,
                        "range_build_sha256": range_build,
                    }
                )
        query_bundles[query_bundle_sha] = {
            "round_index": round_index,
            "entries": query_bundle_entries,
        }
        query_profiles_by_bundle[query_bundle_sha] = round_query_profiles
        evaluation_bundles[evaluation_bundle_sha] = {
            "round_index": round_index,
            "entries": evaluation_bundle_entries,
        }
        evaluation_profiles_by_bundle[evaluation_bundle_sha] = evaluation_profiles

    rows = []
    for round_index in (2, 3, 4):
        for seed in root_state["seeds"]:
            for root_id, root in sorted(root_state["by_id"].items()):
                is_btn = root["actor"] == "btn"
                rows.append(
                    {
                        "round_index": round_index,
                        "solver_seed": seed,
                        "root_id": root_id,
                        "root_identity_commitment_sha256": root[
                            "root_identity_commitment_sha256"
                        ],
                        "root_commitment_sha256": root[
                            "root_commitment_sha256"
                        ],
                        "stratum": root["stratum"],
                        "actor": root["actor"],
                        "phase": root["phase"],
                        "visible_joker_count": root["visible_joker_count"],
                        "observation_digest": root["observation_digest"],
                        "previous_candidate_query_checkpoint_bundle_sha256": query_bundle_sha_by_round[
                            round_index - 1
                        ],
                        "current_candidate_query_checkpoint_bundle_sha256": query_bundle_sha_by_round[
                            round_index
                        ],
                        "previous_evaluation_checkpoint_bundle_sha256": evaluation_bundle_sha_by_round[
                            round_index - 1
                        ],
                        "current_evaluation_checkpoint_bundle_sha256": evaluation_bundle_sha_by_round[
                            round_index
                        ],
                        "previous_candidate_policy_artifact_sha256": candidate_sha_by_round[
                            round_index - 1
                        ],
                        "current_candidate_policy_artifact_sha256": candidate_sha_by_round[
                            round_index
                        ],
                        "previous_range_evidence_artifact_sha256": range_sha_by_key[
                            (round_index - 1, seed, root_id)
                        ],
                        "current_range_evidence_artifact_sha256": range_sha_by_key[
                            (round_index, seed, root_id)
                        ],
                        "previous_policy_distribution": encoded_policy_by_root[root_id],
                        "current_policy_distribution": encoded_policy_by_root[root_id],
                        "previous_btn_posterior_weights": (
                            {"particle": "1/1"} if is_btn else None
                        ),
                        "current_btn_posterior_weights": (
                            {"particle": "1/1"} if is_btn else None
                        ),
                        "exact_exploitability_computed": False,
                    }
                )
    zero = "0/1"
    summary = {
        "production_minima": gate._production_thresholds(),
        "independent_seeds": list(root_state["seeds"]),
        "root_counts_by_stratum": dict(root_state["counts"]),
        "transition_rounds": [2, 3, 4],
        "round_metrics": [
            {
                "round_index": round_index,
                "max_previous_to_current_policy_tv": zero,
                "max_previous_to_current_btn_posterior_tv": zero,
                "max_same_root_cross_seed_policy_tv": zero,
                "max_same_root_cross_seed_btn_posterior_tv": zero,
                "cross_seed_policy_pair_count": 1800,
                "cross_seed_btn_posterior_pair_count": 900,
                "converged": True,
            }
            for round_index in (2, 3, 4)
        ],
        "final_consecutive_converged_rounds": 3,
        "all_six_strata_exactly_covered": True,
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
    }
    evidence = {
        "schema": gate.EVIDENCE_SCHEMA,
        "gate_id": gate.GATE_ID,
        "scope": gate.SCOPE,
        "evidence_kind": gate.EVIDENCE_KIND,
        "production_promotion_claim": True,
        "gate_config_sha256": config["gate_config_sha256"],
        "exact_exploitability_computed": False,
        "strategic_strength_evaluated": False,
        "requires_independent_strength_gate": True,
        "cold_start_zero_drift_is_strength_evidence": False,
        "candidate_policy_method": SOLVER_METHOD,
        "source_manifest": {},
        "source_manifest_sha256": config["approved_source_manifest_sha256"],
        "solver_manifest": {},
        "solver_manifest_sha256": config["approved_solver_manifest_sha256"],
        "root_manifest": _root_manifest(),
        "candidate_query_manifest": _query_manifest(),
        "excluded_partitions": partitions,
        "candidate_query_checkpoint_bundles": {
            key: {"round_index": value["round_index"]}
            for key, value in query_bundles.items()
        },
        "evaluation_checkpoint_bundles": {
            key: {"round_index": value["round_index"]}
            for key, value in evaluation_bundles.items()
        },
        "candidate_policy_artifacts": {
            key: {"fixture": key} for key in candidates
        },
        "restricted_range_artifacts": {},
        "raw_iteration_rows": rows,
        "published_summary": summary,
    }
    evidence["artifact_sha256"] = gate.self_hash(evidence, "artifact_sha256")
    return {
        "evidence": evidence,
        "config": config,
        "candidates": candidates,
        "query_bundles": query_bundles,
        "query_profiles": query_profiles_by_bundle,
        "evaluation_bundles": evaluation_bundles,
        "evaluation_profiles": evaluation_profiles_by_bundle,
        "ranges": ranges,
        "audits": audits,
    }


def _install_mock_production_io(monkeypatch, payload):
    monkeypatch.setattr(
        gate,
        "_validate_source_and_solver",
        lambda *args, **kwargs: (
            _sha("source-manifest"),
            _sha("solver-manifest"),
            _sha("range-source"),
        ),
    )
    monkeypatch.setattr(
        gate,
        "_candidate_artifacts",
        lambda *args, **kwargs: payload["candidates"],
    )
    monkeypatch.setattr(
        gate,
        "_evaluation_checkpoint_bundles",
        lambda *args, **kwargs: (
            payload["evaluation_bundles"],
            payload["evaluation_profiles"],
        ),
    )
    monkeypatch.setattr(
        gate,
        "_candidate_query_checkpoint_bundles",
        lambda *args, **kwargs: (
            payload["query_bundles"],
            payload["query_profiles"],
        ),
    )
    monkeypatch.setattr(
        gate,
        "_range_artifacts",
        lambda *args, **kwargs: (payload["ranges"], payload["audits"]),
    )


def test_production_cartesian_evidence_passes_and_binding_replays_gate(
    monkeypatch, tmp_path
):
    payload = _mock_production_payload()
    _install_mock_production_io(monkeypatch, payload)

    result = gate.validate_t3_bb_fixed_point_evidence(
        payload["evidence"],
        config=payload["config"],
        checkpoint_bundle_root=tmp_path,
        workspace_root=tmp_path,
    )

    assert result["passed"] is True, result["failures"]
    assert result["promotion_eligible"] is True
    assert result["production_minima_enforced"] is True
    assert result["strategic_strength_evaluated"] is False
    assert result["requires_independent_strength_gate"] is True
    assert result["cold_start_zero_drift_is_strength_evidence"] is False
    assert result["derived_metrics"]["final_consecutive_converged_rounds"] == 3
    assert result["derived_metrics"]["round_metrics"][-1][
        "cross_seed_policy_pair_count"
    ] == 1800
    binding = gate.build_t3_bb_likelihood_binding(
        payload["evidence"],
        config=payload["config"],
        gate_result=result,
        checkpoint_bundle_root=tmp_path,
        workspace_root=tmp_path,
    )
    assert binding["promotion_eligible"] is True
    assert binding["fixed_point_converged"] is True
    assert binding["candidate_policy_artifact_sha256"] == result[
        "candidate_policy_artifact_sha256"
    ]
    assert binding["fixed_point_evidence_sha256"] == payload["evidence"][
        "artifact_sha256"
    ]


def test_rehashed_policy_copy_tamper_fails_against_checkpoint_content(
    monkeypatch, tmp_path
):
    payload = _mock_production_payload()
    _install_mock_production_io(monkeypatch, payload)
    evidence = copy.deepcopy(payload["evidence"])
    row = evidence["raw_iteration_rows"][0]
    actions = list(row["current_policy_distribution"])
    row["current_policy_distribution"] = {
        action: ("1/1" if index == 0 else "0/1")
        for index, action in enumerate(actions)
    }
    evidence["artifact_sha256"] = gate.self_hash(evidence, "artifact_sha256")

    result = gate.validate_t3_bb_fixed_point_evidence(
        evidence,
        config=payload["config"],
        checkpoint_bundle_root=tmp_path,
        workspace_root=tmp_path,
    )
    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert any("fresh checkpoint strategy" in failure for failure in result["failures"])


def test_false_claim_is_fully_replayed_but_can_never_promote(monkeypatch, tmp_path):
    payload = _mock_production_payload()
    _install_mock_production_io(monkeypatch, payload)
    evidence = copy.deepcopy(payload["evidence"])
    evidence["production_promotion_claim"] = False
    evidence["artifact_sha256"] = gate.self_hash(evidence, "artifact_sha256")

    result = gate.validate_t3_bb_fixed_point_evidence(
        evidence,
        config=payload["config"],
        checkpoint_bundle_root=tmp_path,
        workspace_root=tmp_path,
    )

    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["derived_metrics"] == evidence["published_summary"]
    assert result["failures"] == ["production promotion claim is false"]


def test_diagnostic_minima_audit_lists_seeds_roots_rounds_and_false_claim():
    failures = gate._production_minima_failures(
        root_state={
            "seeds": (11, 22),
            "counts": {stratum: 1 for stratum in REQUIRED_STRATA},
        },
        rounds=(2, 3),
        trailing_converged_rounds=2,
        production_claim=False,
    )

    assert any("independent seeds" in failure for failure in failures)
    assert any("roots per stratum" in failure for failure in failures)
    assert any("transition rounds" in failure for failure in failures)
    assert any("trailing production-converged" in failure for failure in failures)
    assert "production promotion claim is false" in failures


def test_old_single_bundle_content_hash_cycle_is_explicitly_rejected(
    monkeypatch, tmp_path
):
    base = _mock_production_payload()
    payload = dict(base)
    payload["candidates"] = copy.deepcopy(base["candidates"])
    payload["query_bundles"] = dict(base["query_bundles"])
    payload["query_profiles"] = dict(base["query_profiles"])
    evidence = copy.deepcopy(base["evidence"])

    round_two_rows = [
        row for row in evidence["raw_iteration_rows"] if row["round_index"] == 2
    ]
    query_sha = round_two_rows[0][
        "current_candidate_query_checkpoint_bundle_sha256"
    ]
    evaluation_sha = round_two_rows[0][
        "current_evaluation_checkpoint_bundle_sha256"
    ]
    candidate_sha = round_two_rows[0][
        "current_candidate_policy_artifact_sha256"
    ]
    # Model the old impossible contract by making C2 point at B2 and exposing
    # B2 under both verifier maps.  The gate must reject the shared identity
    # before accepting any self-hashed row claims.
    payload["candidates"][candidate_sha]["model_manifest"][
        "checkpoint_sha256"
    ] = evaluation_sha
    payload["query_bundles"][evaluation_sha] = base["evaluation_bundles"][
        evaluation_sha
    ]
    payload["query_profiles"][evaluation_sha] = base["query_profiles"][query_sha]
    for row in evidence["raw_iteration_rows"]:
        if row["round_index"] == 2:
            row["current_candidate_query_checkpoint_bundle_sha256"] = evaluation_sha
        if row["round_index"] == 3:
            row["previous_candidate_query_checkpoint_bundle_sha256"] = evaluation_sha
    evidence["candidate_query_checkpoint_bundles"][evaluation_sha] = {
        "round_index": 2
    }
    evidence["artifact_sha256"] = gate.self_hash(evidence, "artifact_sha256")
    payload["evidence"] = evidence
    _install_mock_production_io(monkeypatch, payload)

    result = gate.validate_t3_bb_fixed_point_evidence(
        evidence,
        config=payload["config"],
        checkpoint_bundle_root=tmp_path,
        workspace_root=tmp_path,
    )
    assert result["passed"] is False
    assert any("causally separate" in failure for failure in result["failures"])
