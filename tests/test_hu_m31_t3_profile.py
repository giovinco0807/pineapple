from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ofc_regular.action_key import (
    action_key,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_m31_t3_runtime import (
    T3SearchActionValue,
    T3SearchDecision,
)
from ofc_regular.validate_hu_m31_t3_profile import (
    PROFILE_SCHEMA,
    T3ProfileConfig,
    _write_json_atomic,
    generate_general_t3_roots,
    process_memory_snapshot,
    run_profile,
)


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class _FakeSolver:
    def __init__(self, config: T3ProfileConfig, *, confirmation: bool) -> None:
        self.config = SimpleNamespace(
            run_id=config.run_id,
            candidate_seed=config.candidate_seed,
            evaluation_seed=config.evaluation_seed,
            candidate_samples=(
                config.confirmation_candidate_samples
                if confirmation
                else config.candidate_samples
            ),
            evaluation_samples=(
                config.confirmation_evaluation_samples
                if confirmation
                else config.evaluation_samples
            ),
            downstream_t3_samples=(
                config.confirmation_downstream_t3_samples
                if confirmation
                else config.downstream_t3_samples
            ),
        )
        self.search_config = SimpleNamespace(downstream_t4_samples=0)
        self.engine_version = "fake-m31-profile-engine"
        self.library_path = Path("fake-release-engine.dll")
        self.library_sha256 = "a" * 64
        self._confirmation = confirmation

    def solve(self, observation):
        return self._decision(observation, execution_mode="scalar", batch_size=1)

    def solve_many(self, observations):
        return [
            self._decision(
                observation,
                execution_mode="batch_amortized",
                batch_size=len(observations),
            )
            for observation in observations
        ]

    def _decision(self, observation, *, execution_mode, batch_size):
        actions = generate_turn_actions(
            observation.hero_board,
            observation.dealt_cards,
        )
        factor = 1.25 if self._confirmation else 1.0
        selection = [
            factor * float(sum(action_key(action).masks) % 19)
            for action in actions
        ]
        evaluation = [
            factor * float(sum(action_key(action).masks) % 23)
            for action in actions
        ]
        ranking = canonical_descending_indices(selection, actions)
        rank_by_index = {index: rank for rank, index in enumerate(ranking)}
        best_evaluation = max(evaluation)
        values = tuple(
            sorted(
                (
                    T3SearchActionValue(
                        original_index=index,
                        rank=rank_by_index[index],
                        action_key=action_key(action).to_token(),
                        selection_ev=selection[index],
                        evaluation_ev=evaluation[index],
                        evaluation_regret=best_evaluation - evaluation[index],
                        action=action,
                    )
                    for index, action in enumerate(actions)
                ),
                key=lambda row: row.action_key,
            )
        )
        selected_index = ranking[0]
        second = selection[ranking[1]] if len(ranking) > 1 else selection[selected_index]
        fingerprint = observation.fingerprint()
        semantic = _digest(
            [
                fingerprint,
                [[row.action_key, row.rank, row.selection_ev, row.evaluation_ev] for row in values],
            ]
        )
        mapping = _digest(
            [fingerprint, [action_key(action).to_token() for action in actions]]
        )
        return T3SearchDecision(
            action=actions[selected_index],
            selected_action_key=action_key(actions[selected_index]).to_token(),
            selected_selection_ev=selection[selected_index],
            selected_evaluation_ev=evaluation[selected_index],
            selection_gap=selection[selected_index] - second,
            evaluation_sample_regret=(
                best_evaluation - evaluation[selected_index]
            ),
            action_values=values,
            seat=observation.seat,
            value_scope="fake-profile-values",
            observation_fingerprint=fingerprint,
            legal_action_set_digest=legal_action_set_digest(actions),
            legal_action_order_digest=ordered_action_mapping_digest(actions),
            candidate_belief_digest=_digest([fingerprint, "candidate-belief"]),
            evaluation_belief_digest=_digest([fingerprint, "evaluation-belief"]),
            candidate_rng_digest=_digest([fingerprint, "candidate-rng"]),
            evaluation_rng_digest=_digest([fingerprint, "evaluation-rng"]),
            child_information_set_count=len(actions) * 2,
            candidate_samples=self.config.candidate_samples,
            evaluation_samples=self.config.evaluation_samples,
            downstream_t3_samples=self.config.downstream_t3_samples,
            run_id=self.config.run_id,
            continuation_seed=7,
            candidate_seed=self.config.candidate_seed,
            evaluation_seed=self.config.evaluation_seed,
            use_t4_action_cache=True,
            search_contract_digest=_digest(["fake-contract", factor]),
            solver_id="fake-solver",
            engine_version=self.engine_version,
            native_library_sha256=self.library_sha256,
            native_latency_ms=1.0,
            validation_latency_ms=0.1,
            total_latency_ms=1.1,
            execution_mode=execution_mode,
            batch_size=batch_size,
            semantic_result_digest=semantic,
            result_digest=mapping,
        )


def test_general_root_generation_is_balanced_deterministic_and_diverse():
    config = T3ProfileConfig(
        root_count=4,
        deterministic_roots=2,
        permutation_roots=2,
        ladder_roots=2,
    )
    first = generate_general_t3_roots(config)
    repeated = generate_general_t3_roots(config)

    assert [row.observation.seat for row in first] == [
        "first",
        "second",
        "first",
        "second",
    ]
    assert [row.hand_seed for row in first] == [
        config.seed_start,
        config.seed_start,
        config.seed_start + config.seed_stride,
        config.seed_start + config.seed_stride,
    ]
    assert [row.observation.fingerprint() for row in first] == [
        row.observation.fingerprint() for row in repeated
    ]
    action_counts = {
        len(
            generate_turn_actions(
                row.observation.hero_board,
                row.observation.dealt_cards,
            )
        )
        for row in first
    }
    assert action_counts == {9, 21}
    assert all(
        "opponent_private_discards" not in row.observation.to_dict()
        for row in first
    )


def test_fake_profile_exercises_all_integrity_and_projection_gates():
    config = T3ProfileConfig(
        root_count=4,
        deterministic_roots=2,
        permutation_roots=2,
        ladder_roots=2,
    )
    memory = lambda: {
        "supported": True,
        "source": "test",
        "rss_bytes": 100,
        "peak_rss_bytes": 200,
        "private_bytes": 80,
    }

    report = run_profile(
        config=config,
        primary_solver=_FakeSolver(config, confirmation=False),
        confirmation_solver=_FakeSolver(config, confirmation=True),
        memory_sampler=memory,
    )

    assert report["schema"] == PROFILE_SCHEMA
    assert report["status"] == "pass"
    assert report["all_gates_passed"] is True
    assert all(report["gates"].values())
    assert report["performance"]["by_seat"]["first"]["count"] == 2
    assert report["performance"]["by_seat"]["second"]["count"] == 2
    assert report["permutation_probe"]["root_count"] == 2
    assert report["sample_budget_ladder"]["root_count"] == 2
    assert report["memory"]["peak_rss_bytes"] == 200
    assert report["local100_projection"]["projected_total_seconds"] < 3600.0
    assert report["root_generation"]["current_profile_read"] is False
    assert report["spot_vm_started"] is False


def test_memory_snapshot_and_write_once_artifact(tmp_path):
    snapshot = process_memory_snapshot()
    assert snapshot["supported"] is True
    assert snapshot["peak_rss_bytes"] > 0

    path = tmp_path / "profile.json"
    _write_json_atomic(path, {"status": "pass"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "pass"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(path, {"status": "changed"})


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"root_count": 3}, "positive even"),
        ({"seed_stride": 0}, "positive"),
        ({"candidate_seed": 7, "evaluation_seed": 7}, "distinct"),
        ({"permutation_roots": 1}, "must be even"),
        (
            {"candidate_samples": 3, "confirmation_candidate_samples": 2},
            "must not be smaller",
        ),
    ),
)
def test_profile_config_fails_closed(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        T3ProfileConfig(**kwargs)
