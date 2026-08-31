import math
from pathlib import Path

import pytest

from ai.tutor.exact_late import default_rust_t3_exact_solver_path
from ai.tutor.t3_hu_public_tree_cfr import (
    PublicTreeChanceNode,
    PublicTreeDecisionNode,
    solve_recursive_public_tree_cfr_plus,
)
from ai.tutor.t3_hu_reduced_fixtures import (
    FIXTURE_ID_BB_JOKER0,
    compile_bb_joker0_reduced_fixture,
    compile_canonical_reduced_fixture,
)


@pytest.fixture(scope="module")
def all_compiled_fixtures():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")
    return {
        (actor, joker): compile_canonical_reduced_fixture(
            actor,
            joker,
            rust_solver_path=solver,
            rust_timeout_s=30.0,
        )
        for actor in ("bb", "btn")
        for joker in (0, 1, 2)
    }


@pytest.fixture(scope="module")
def compiled_fixture(all_compiled_fixtures):
    return all_compiled_fixtures[("bb", 0)]


@pytest.mark.parametrize(
    ("actor", "joker", "leaf_positions", "shared_t4"),
    [
        ("bb", 0, 18, 9),
        ("bb", 1, 18, 9),
        ("bb", 2, 18, 9),
        ("btn", 0, 6, 0),
        ("btn", 1, 6, 0),
        ("btn", 2, 6, 0),
    ],
)
def test_all_required_actor_joker_strata_compile_real_all_action_trees(
    all_compiled_fixtures,
    actor,
    joker,
    leaf_positions,
    shared_t4,
):
    fixture = all_compiled_fixtures[(actor, joker)]
    assert fixture.actor == actor
    assert fixture.visible_joker_count == joker
    assert fixture.metadata["root_legal_actions"] == 3
    assert fixture.metadata["physical_t4_leaf_positions"] == leaf_positions
    assert fixture.shared_t4_infoset_pairs == shared_t4
    assert fixture.metadata["selection_performed_in_physical_world"] is False
    assert len(fixture.fixture_manifest_sha256) == 64
    root_nodes = [branch.child for branch in fixture.root.branches]
    assert root_nodes[0].infoset_key == root_nodes[1].infoset_key
    assert sum(
        card.startswith("X") for card in root_nodes[0].infoset_key.current_draw
    ) == joker
    assert all(
        result["selection_performed"] is False
        and result["candidate_count"] == 3
        for result in fixture.physical_leaf_results
    )
    assert len(
        {item.fixture_manifest_sha256 for item in all_compiled_fixtures.values()}
    ) == 6


def test_bb_joker0_fixture_keeps_every_real_action_and_shared_public_infoset(
    compiled_fixture,
):
    fixture = compiled_fixture
    assert fixture.fixture_id == FIXTURE_ID_BB_JOKER0
    assert fixture.actor == "bb"
    assert fixture.visible_joker_count == 0
    assert isinstance(fixture.root, PublicTreeChanceNode)
    assert [branch.outcome_id for branch in fixture.root.branches] == [
        "world-a",
        "world-b",
    ]
    root_nodes = [branch.child for branch in fixture.root.branches]
    assert all(isinstance(node, PublicTreeDecisionNode) for node in root_nodes)
    assert root_nodes[0].infoset_key == root_nodes[1].infoset_key
    assert len(root_nodes[0].action_ids) == len(root_nodes[1].action_ids) == 3
    assert root_nodes[0].action_ids == root_nodes[1].action_ids

    assert len(fixture.physical_leaf_results) == 18
    assert fixture.shared_t4_infoset_pairs == 9
    commitments = {
        result["physical_state_commitment"]
        for result in fixture.physical_leaf_results
    }
    assert len(commitments) == 18
    for result in fixture.physical_leaf_results:
        assert result["selection_performed"] is False
        assert result["particle_aggregation_performed"] is False
        assert result["requires_infoset_aggregation"] is True
        assert result["source"] == "rust_exact_physical_t4_action_vector"
        assert result["legal_actions"] == result["evaluated_actions"] == 3
        assert len(result["action_keys"]) == 3
        assert "best" not in result and "chosen_action" not in result

    assert fixture.metadata == {
        "method": "canonical_reduced_real_card_fixture",
        "fixture_id": FIXTURE_ID_BB_JOKER0,
        "actor": "bb",
        "visible_joker_count": 0,
        "position_contract_version": "bb_first_v1",
        "physical_world_count": 2,
        "root_legal_actions": 3,
        "btn_decision_nodes": 6,
        "physical_t4_leaf_positions": 18,
        "shared_t4_infoset_pairs": 9,
        "selection_performed_in_physical_world": False,
        "strategy_fusion": False,
        "rust_leaf_integrated": True,
        "full_card": False,
        "hu_exact": False,
    }


def test_bb_joker0_real_tree_converges_with_infoset_aware_best_responses(
    compiled_fixture,
):
    solved = solve_recursive_public_tree_cfr_plus(
        compiled_fixture.root,
        iterations=200,
        checkpoints=(1, 50, 200),
        max_pure_profiles=100_000,
    )

    assert len(solved.average_strategy) == 16
    assert solved.metrics.btn_best_response <= solved.metrics.value_bb
    assert solved.metrics.value_bb <= solved.metrics.bb_best_response
    assert math.isfinite(solved.metrics.exploitability)
    assert solved.metrics.exploitability < 0.01
    assert solved.exploitability_trace[-1][1] < solved.exploitability_trace[0][1]
    assert solved.metadata["strategy_fusion"] is False
    assert solved.metadata["rust_leaf_integrated"] is True
    assert solved.metadata["terminal_utility_sources"] == [
        "rust_exact_physical_t4_action_vector"
    ]


def test_bb_joker0_fixture_manifest_is_stable_under_world_input_reordering(
    compiled_fixture,
):
    solver = default_rust_t3_exact_solver_path()
    reversed_fixture = compile_bb_joker0_reduced_fixture(
        rust_solver_path=solver,
        rust_timeout_s=30.0,
        reverse_world_order=True,
    )

    assert reversed_fixture.fixture_manifest_sha256 == (
        compiled_fixture.fixture_manifest_sha256
    )
    assert reversed_fixture.root_infoset_digest == compiled_fixture.root_infoset_digest
    assert reversed_fixture.shared_t4_infoset_pairs == 9


def test_btn_joker0_real_tree_converges_with_one_shared_btn_root_infoset(
    all_compiled_fixtures,
):
    fixture = all_compiled_fixtures[("btn", 0)]
    solved = solve_recursive_public_tree_cfr_plus(
        fixture.root,
        iterations=200,
        checkpoints=(1, 50, 200),
        max_pure_profiles=100_000,
    )

    assert len(solved.average_strategy) == 7
    assert solved.metrics.btn_best_response <= solved.metrics.value_bb
    assert solved.metrics.value_bb <= solved.metrics.bb_best_response
    assert solved.metrics.exploitability < 0.01
    assert solved.exploitability_trace[-1][1] < solved.exploitability_trace[0][1]
    assert solved.metadata["rust_leaf_integrated"] is True
