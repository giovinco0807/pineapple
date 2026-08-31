from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import pytest

import ai.tutor.promotion_gate_m3_full_card_smoke as smoke
import ai.tutor.t3_hu_full_card_range as range_module
from ai.engine.encoding import ALL_CARDS
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.promotion_gate_m3_full_card_smoke import (
    FullCardSmokeStratumInput,
    REQUIRED_STRATA,
    canonical_sha256,
    run_six_strata_full_card_smoke,
    run_validate_and_write_six_strata_smoke,
    validate_m3_full_card_smoke_evidence,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    build_history_weighted_full_card_range,
    verify_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall


@dataclass(frozen=True)
class _Observation:
    actor: str
    phase: str
    current_draw: tuple[str, ...]
    digest_value: str
    turn: int = 3

    def digest(self) -> str:
        return self.digest_value


class _Adapter:
    def __init__(self, observation, root_range):
        self.observation = observation
        self.root_range = root_range


def _model_manifest(*, promotion_eligible: bool) -> dict:
    return {
        "schema": "ofc_frozen_behavior_model/v1",
        "model_id": "fixture_policy_value_prior_v1",
        "model_type": "torch_policy_value_teacher_boltzmann_v1",
        "position_contract_version": "bb_first_v1",
        "promotion_eligible": promotion_eligible,
        "calibration_status": "uncalibrated_policy_ranking_logits",
    }


def _range_metadata(
    actor: str,
    joker_count: int,
    observation_digest: str,
    *,
    promotion_eligible: bool,
) -> dict:
    model = _model_manifest(promotion_eligible=promotion_eligible)
    model_sha = canonical_sha256(model)
    content = {
        "schema": "ofc_full_card_range_content/v1",
        "range_model": "history_weighted_full_card_discard_particles_v1",
        "position_contract_version": "bb_first_v1",
        "deck_size": 54,
        "physical_joker_ids": ["X1", "X2"],
        "observation_digest": observation_digest,
        "actor": actor,
        "turn": 3,
        "phase": "t3_first" if actor == "bb" else "t3_second",
        "visible_joker_count": joker_count,
        "behavior_model_id": model["model_id"],
        "behavior_model_sha256": model_sha,
        "epsilon": "0/1",
        "particle_count": 1,
        "expected_undealt_card_count": 29 if actor == "bb" else 26,
        "posterior_probability_mass_exact": "1/1",
        "particles": [
            {
                "commitment": canonical_sha256(
                    {"actor": actor, "joker_count": joker_count, "particle": 0}
                ),
                "weight": "1/1",
            }
        ],
    }
    content_sha = canonical_sha256(content)
    information_digest = canonical_sha256(
        {"observation": observation_digest, "behavior_query": 0}
    )
    build = {
        "schema": "ofc_full_card_range_build/v1",
        "range_model": content["range_model"],
        "range_content_sha256": content_sha,
        "sampler": "synthetic_test_fixture_v1",
        "seed": 17,
        "behavior_query_count": 1,
        "behavior_unique_query_count": 1,
        "behavior_model_evaluation_count": 1,
        "behavior_distribution_source_counts": {"model": 1},
        "behavior_uniform_fallback_count": 0,
        "behavior_uniform_fallback_unique_count": 0,
        "behavior_model_hit_rate_exact": "1/1",
        "behavior_model_hit_rate": 1.0,
        "behavior_distribution_validation_failures": 0,
        "behavior_query_audit": [
            {
                "information_digest": information_digest,
                "distribution_sha256": canonical_sha256(
                    {"information": information_digest, "distribution": "fixture"}
                ),
                "source": "model",
                "used_fallback": False,
                "evaluation_count": 1,
            }
        ],
    }
    build_sha = canonical_sha256(build)
    return {
        "behavior_model_id": model["model_id"],
        "behavior_model_sha256": model_sha,
        "range_sha256": content_sha,
        "range_content_sha256": content_sha,
        "range_build_sha256": build_sha,
        "content_manifest": content,
        "build_manifest": build,
        "behavior_model_manifest": model,
    }


def _draw(joker_count: int, index: int) -> tuple[str, ...]:
    non_jokers = (
        ("Ac", "Kd", "Qh"),
        ("2c", "3d", "4h"),
        ("5c", "6d", "7h"),
        ("8c", "9d", "Th"),
        ("Jc", "Qd", "Kh"),
        ("As", "Ks", "Qs"),
    )[index]
    if joker_count == 0:
        return non_jokers
    if joker_count == 1:
        return ("X1", non_jokers[0], non_jokers[1])
    return ("X1", "X2", non_jokers[0])


def _inputs(*, promotion_eligible: bool = False) -> dict:
    result = {}
    for index, name in enumerate(REQUIRED_STRATA):
        actor, joker_text = name.split("_joker")
        joker_count = int(joker_text)
        digest = canonical_sha256(
            {"fixture": "six_strata", "actor": actor, "joker": joker_count}
        )
        observation = _Observation(
            actor=actor,
            phase="t3_first" if actor == "bb" else "t3_second",
            current_draw=_draw(joker_count, index),
            digest_value=digest,
        )
        metadata = _range_metadata(
            actor,
            joker_count,
            digest,
            promotion_eligible=promotion_eligible,
        )
        root_range = SimpleNamespace(
            metadata=metadata,
            range_sha256=metadata["range_sha256"],
            range_content_sha256=metadata["range_content_sha256"],
            range_build_sha256=metadata["range_build_sha256"],
            behavior_model_id=metadata["behavior_model_id"],
            behavior_model_sha256=metadata["behavior_model_sha256"],
        )
        result[name] = FullCardSmokeStratumInput(observation, root_range)
    return result


def _fake_verify(observation, root_range):
    return {
        "verified": True,
        "particle_count": 1,
        "effective_sample_size_exact": "1/1",
        "behavior_query_count": 1,
        "behavior_unique_query_count": 1,
        "behavior_uniform_fallback_count": 0,
        "range_content_sha256": root_range.range_content_sha256,
        "range_build_sha256": root_range.range_build_sha256,
    }


def _fake_solve(
    adapter,
    *,
    iterations: int,
    seed: int,
    max_infosets: int,
    linear_averaging: bool,
):
    observation = adapter.observation
    root_range = adapter.root_range
    strategy = {observation: {"fixture_action": 1.0}}
    regret = {observation: {"fixture_action": 0.0}}
    average_json = f'{{"kind":"average","root":"{observation.digest()}"}}'
    current_json = f'{{"kind":"current","root":"{observation.digest()}"}}'
    average_sha = hashlib.sha256(average_json.encode()).hexdigest()
    current_sha = hashlib.sha256(current_json.encode()).hexdigest()
    metadata = {
        "method": "full_card_dynamic_external_sampling_mccfr_plus_v1",
        "adapter": "full_card_generative_t3_t4_v1",
        "sampling_scheme": "external_sampling",
        "traverser_schedule": "bb_then_btn_each_iteration",
        "alternating_updates": True,
        "regret_matching_plus": True,
        "encountered_infoset_tables": True,
        "opponent_sample_cached_per_infoset": True,
        "future_chance_sampled_per_physical_state": True,
        "root_posterior_sampled_once_per_traversal": True,
        "posterior_probability_multiplied_after_sampling": False,
        "chance_probability_multiplied_after_sampling": False,
        "joint_particle_weight_used_after_sampling": False,
        "table_key_contains_particle_commitment": False,
        "table_key_contains_remaining_cards": False,
        "table_key_contains_particle_weight": False,
        "artifact_contains_raw_particle_world": False,
        "full_card": True,
        "full_card_policy_promoted": False,
        "hu_exact": False,
        "runtime_integrated": False,
        "exact_exploitability_computed": False,
        "average_strategy_sha256": average_sha,
        "current_strategy_sha256": current_sha,
        "range_content_sha256": root_range.range_content_sha256,
        "range_build_sha256": root_range.range_build_sha256,
        "behavior_model_id": root_range.behavior_model_id,
        "behavior_model_sha256": root_range.behavior_model_sha256,
        "position_contract_version": "bb_first_v1",
        "linear_averaging": linear_averaging,
        "max_infosets": max_infosets,
    }
    stats = {
        "traversals": 2 * iterations,
        "traversals_by_actor": {"bb": iterations, "btn": iterations},
        "root_posterior_samples": 2 * iterations,
        "future_draw_samples": 6 * iterations,
        "future_draw_samples_by_next_phase": {"t3_second": iterations},
        "decision_visits": 8 * iterations,
        "terminal_visits": 2 * iterations,
        "traverser_actions_expanded": 4 * iterations,
        "opponent_action_samples": 2 * iterations,
        "opponent_action_cache_hits": 0,
        "strategy_sum_updates": 2 * iterations,
        "infosets_created": 1,
    }
    return SimpleNamespace(
        iterations=iterations,
        traversals=2 * iterations,
        seed=seed,
        encountered_infosets=1,
        average_strategy=strategy,
        current_strategy=strategy,
        cumulative_regret_plus=regret,
        average_strategy_json=average_json,
        current_strategy_json=current_json,
        average_strategy_sha256=average_sha,
        current_strategy_sha256=current_sha,
        sampling_stats=stats,
        metadata=metadata,
    )


@pytest.fixture
def patched_runner(monkeypatch):
    monkeypatch.setattr(smoke, "verify_full_card_range", _fake_verify)
    monkeypatch.setattr(smoke, "FullCardGenerativeAdapter", _Adapter)
    monkeypatch.setattr(smoke, "solve_full_card_external_sampling_mccfr", _fake_solve)


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = canonical_sha256(value)


def _failure_text(result: dict) -> str:
    return "\n".join(result["failures"])


_BB_BOARD = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc",),
)
_BTN_BOARD = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As",),
)
_PUBLIC_HISTORY = (
    (
        0,
        "bb",
        (
            ("4h", "top"),
            ("2c", "top"),
            ("3d", "top"),
            ("7c", "middle"),
            ("Qc", "bottom"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("8c", "top"),
            ("5c", "top"),
            ("6d", "top"),
            ("9c", "middle"),
            ("As", "bottom"),
        ),
    ),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)
_BB_T3_PLACEMENTS = (("Qd", "bottom"), ("Kh", "bottom"))
_PUBLIC_HISTORY_AFTER_BB_T3 = _PUBLIC_HISTORY + (
    (3, "bb", _BB_T3_PLACEMENTS),
)
_BB_BOARD_AFTER_T3 = (
    _BB_BOARD[0],
    _BB_BOARD[1],
    ("Qc", "Qd", "Kh"),
)


def _private_recall(actor: str) -> PrivateRecall:
    placements = {
        "bb": (("7d", "8h"), ("9s", "Tc")),
        "btn": (("9d", "Jh"), ("Qs", "Kc")),
    }[actor]
    discards = {"bb": ("6c", "6s"), "btn": ("4s", "7h")}[actor]
    return PrivateRecall(
        dealt_by_turn=(
            (1, (*placements[0], discards[0])),
            (2, (*placements[1], discards[1])),
        ),
        discards_by_turn=((1, discards[0]), (2, discards[1])),
    )


def _append_bb_t3_recall(recall: PrivateRecall) -> PrivateRecall:
    return PrivateRecall(
        dealt_by_turn=recall.dealt_by_turn + ((3, ("Qd", "Kh", "5d")),),
        discards_by_turn=recall.discards_by_turn + ((3, "5d"),),
    )


def _physical_observation(actor: str, joker_count: int) -> InfoSetKey:
    current_draw = {
        0: ("Ad", "Kd", "Jd"),
        1: ("X1", "Ad", "Kd"),
        2: ("X1", "X2", "Ad"),
    }[joker_count]
    undealt = {
        0: ("X1", "X2", "Ac"),
        1: ("X2", "Ac", "2s"),
        2: ("Ac", "2s", "3s"),
    }[joker_count]
    bb_recall = _private_recall("bb")
    btn_recall = _private_recall("btn")
    if actor == "btn":
        bb_recall = _append_bb_t3_recall(bb_recall)
    particle = JointParticle(
        bb_recall=bb_recall,
        btn_recall=btn_recall,
        undealt_cards=undealt,
    )
    return InfoSetKey.for_particle(
        particle,
        contract_version=POSITION_CONTRACT_VERSION,
        actor=actor,
        turn=3,
        phase="t3_first" if actor == "bb" else "t3_second",
        board_bb=_BB_BOARD if actor == "bb" else _BB_BOARD_AFTER_T3,
        board_btn=_BTN_BOARD,
        public_action_history=(
            _PUBLIC_HISTORY if actor == "bb" else _PUBLIC_HISTORY_AFTER_BB_T3
        ),
        current_draw=current_draw,
    )


class _ExactModelSourceUniformBehavior:
    model_id = "canonical_smoke_exact_model_source_uniform_v1"

    @property
    def model_manifest(self):
        return {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": self.model_id,
            "model_type": "canonical_smoke_exact_model_source_uniform",
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "promotion_eligible": False,
        }

    @property
    def model_sha256(self):
        return canonical_sha256(self.model_manifest)

    def action_distribution(self, information):
        probability = Fraction(1, information.legal_action_count)
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=MappingProxyType(
                {
                    action_id: probability
                    for action_id in information.legal_action_ids
                }
            ),
            source="model",
            used_fallback=False,
        )


def _physical_six_strata_inputs(monkeypatch) -> dict:
    def select_nonjoker_assignment(
        pool, count, *, max_particles, seed, observation_digest
    ):
        assignment = tuple(card for card in pool if card not in ("X1", "X2"))[
            :count
        ]
        assert len(assignment) == count
        return (assignment,), math.perm(len(pool), count), False

    monkeypatch.setattr(
        range_module, "_select_assignments", select_nonjoker_assignment
    )
    behavior = _ExactModelSourceUniformBehavior()
    inputs = {}
    for name in REQUIRED_STRATA:
        actor, joker_text = name.split("_joker")
        observation = _physical_observation(actor, int(joker_text))
        root_range = build_history_weighted_full_card_range(
            observation,
            behavior,
            epsilon=0,
            max_particles=1,
            seed=31,
        )
        assert verify_full_card_range(observation, root_range)["verified"] is True
        inputs[name] = FullCardSmokeStratumInput(observation, root_range)
    return inputs


def test_six_strata_runner_passes_execution_but_cannot_promote_policy(patched_runner):
    evidence = run_six_strata_full_card_smoke(
        _inputs(promotion_eligible=False),
        iterations=1,
        base_seed=9000,
        max_infosets=100,
    )
    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is True, result["failures"]
    assert result["execution_smoke_passed"] is True
    assert result["status"] == "m3_full_card_smoke_ready_nonpromoted"
    assert result["behavior_prior_promotion_eligible"] is False
    assert result["m3_promotion_passed"] is False
    assert result["full_card_policy_promoted"] is False
    assert "behavior_prior_not_promotion_eligible" in result["promotion_blockers"]
    assert set(result["strata"]) == set(REQUIRED_STRATA)
    assert all(row["execution_passed"] for row in result["strata"].values())
    assert evidence["summary"]["execution_smoke_passed"] is True
    assert evidence["summary"]["m3_promotion_passed"] is False
    assert canonical_sha256({
        key: value for key, value in evidence.items() if key != "artifact_sha256"
    }) == evidence["artifact_sha256"]


def test_real_full_card_dynamic_solver_runs_all_six_physical_strata(monkeypatch):
    inputs = _physical_six_strata_inputs(monkeypatch)
    card_value = {card: index + 1 for index, card in enumerate(ALL_CARDS)}

    def light_terminal(self, terminal):
        def board_value(rows):
            return sum(
                (row_index + 1) * card_value[card]
                for row_index, row in enumerate(rows)
                for card in row
            )

        return float(board_value(terminal.board_bb) - board_value(terminal.board_btn))

    monkeypatch.setattr(
        smoke.FullCardGenerativeAdapter, "terminal_utility_bb", light_terminal
    )
    evidence = run_six_strata_full_card_smoke(
        inputs,
        iterations=1,
        base_seed=17000,
        max_infosets=10_000,
    )
    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is True, result["failures"]
    assert result["execution_smoke_passed"] is True
    assert result["behavior_prior_promotion_eligible"] is False
    assert result["m3_promotion_passed"] is False
    assert all(
        evidence["strata"][name]["solver"]["metadata"]["full_card"] is True
        for name in REQUIRED_STRATA
    )


def test_even_six_eligible_behavior_priors_do_not_turn_smoke_into_promotion(
    patched_runner,
):
    evidence = run_six_strata_full_card_smoke(
        _inputs(promotion_eligible=True),
        iterations=1,
    )
    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is True
    assert result["behavior_prior_promotion_eligible"] is True
    assert result["m3_promotion_passed"] is False
    assert result["promotion_blockers"] == [
        "smoke_only_no_strength_or_exploitability_gate",
        "solver_contract_explicitly_nonpromoted",
    ]


def test_role_and_visible_joker_are_independently_bound_to_stratum(patched_runner):
    evidence = run_six_strata_full_card_smoke(_inputs(), iterations=1)
    evidence["strata"]["bb_joker0"]["range"]["metadata"]["content_manifest"][
        "visible_joker_count"
    ] = 2
    _rehash(evidence["strata"]["bb_joker0"]["range"], "range_manifest_sha256")
    _rehash(evidence, "artifact_sha256")

    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is False
    assert "content_manifest.visible_joker_count: must equal 0" in _failure_text(result)


def test_solver_cannot_claim_promoted_even_with_rehashed_manifests(patched_runner):
    evidence = run_six_strata_full_card_smoke(_inputs(), iterations=1)
    solver = evidence["strata"]["btn_joker2"]["solver"]
    solver["metadata"]["full_card_policy_promoted"] = True
    _rehash(solver, "solver_manifest_sha256")
    _rehash(evidence, "artifact_sha256")

    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is False
    assert "metadata.full_card_policy_promoted: must equal False" in _failure_text(
        result
    )
    assert result["m3_promotion_passed"] is False


def test_range_model_hash_and_behavior_audit_are_rederived(patched_runner):
    evidence = run_six_strata_full_card_smoke(_inputs(), iterations=1)
    range_manifest = evidence["strata"]["bb_joker1"]["range"]
    range_manifest["metadata"]["build_manifest"]["behavior_query_count"] = 99
    _rehash(range_manifest, "range_manifest_sha256")
    _rehash(evidence, "artifact_sha256")

    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is False
    text = _failure_text(result)
    assert "range_build_sha256: manifest hash binding mismatch" in text
    assert "behavior_query_count: expected raw-derived 1" in text


def test_solver_exception_is_captured_and_fails_closed(patched_runner, monkeypatch):
    def sometimes_fails(adapter, **kwargs):
        if adapter.observation.actor == "btn" and "X2" in adapter.observation.current_draw:
            raise RuntimeError("synthetic solver failure")
        return _fake_solve(adapter, **kwargs)

    monkeypatch.setattr(smoke, "solve_full_card_external_sampling_mccfr", sometimes_fails)
    evidence = run_six_strata_full_card_smoke(_inputs(), iterations=1)
    result = validate_m3_full_card_smoke_evidence(evidence)

    assert evidence["strata"]["btn_joker2"]["solver"]["completed"] is False
    assert result["passed"] is False
    assert result["execution_smoke_passed"] is False
    assert "solver.completed: must be true" in _failure_text(result)
    assert result["m3_promotion_passed"] is False


def test_missing_stratum_never_yields_partial_success(patched_runner):
    inputs = _inputs()
    inputs.pop("btn_joker2")
    evidence = run_six_strata_full_card_smoke(inputs, iterations=1)
    result = validate_m3_full_card_smoke_evidence(evidence)

    assert evidence["summary"]["execution_smoke_passed"] is False
    assert result["passed"] is False
    assert "exact six-stratum input required" in _failure_text(result)


def test_runner_writes_content_addressed_evidence_and_result_atomically(
    patched_runner, tmp_path
):
    evidence_path = tmp_path / "evidence.json"
    result_path = tmp_path / "result.json"
    evidence, result = run_validate_and_write_six_strata_smoke(
        _inputs(),
        evidence_path=evidence_path,
        result_path=result_path,
        iterations=1,
    )

    loaded_evidence = smoke.json.loads(evidence_path.read_text(encoding="utf-8"))
    loaded_result = smoke.json.loads(result_path.read_text(encoding="utf-8"))
    assert loaded_evidence == evidence
    assert loaded_result == result
    assert validate_m3_full_card_smoke_evidence(loaded_evidence)["passed"] is True
    assert not list(tmp_path.glob("*.tmp"))


def test_non_finite_or_rehashed_summary_claim_fails_closed(patched_runner):
    evidence = run_six_strata_full_card_smoke(_inputs(), iterations=1)
    evidence["summary"]["m3_promotion_passed"] = True
    evidence["not_finite"] = float("nan")
    evidence["artifact_sha256"] = "0" * 64

    result = validate_m3_full_card_smoke_evidence(evidence)

    assert result["passed"] is False
    text = _failure_text(result)
    assert "finite canonical JSON" in text
    assert "not equal to independently derived summary" in text
    assert result["m3_promotion_passed"] is False
