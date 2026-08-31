import json
from argparse import Namespace
from pathlib import Path

import pytest

import ai.tutor.run_t3_hu_sampling_gate as gate
from ai.tutor.run_t3_hu_sampling_gate import (
    _run_seed,
    compare_seed_runs,
    refine_positions,
    sanitize_position,
)


def _result(
    decision_id: str,
    position: str,
    scores: tuple[float, float],
    *,
    best: str,
) -> dict:
    return {
        "pilot": {
            "decision_id": decision_id,
            "cluster_id": f"root:{decision_id}",
            "stratum": f"{position}_joker0",
            "visible_joker_count": 0,
        },
        "position": position,
        "best": {"action_key": best},
        "candidates": [
            {"action_key": "a", "metrics": {"score": scores[0], "standard_error": 0.1}},
            {"action_key": "b", "metrics": {"score": scores[1], "standard_error": 0.1}},
        ],
        "elapsed_ms": 10.0,
        "legal_actions": 2,
    }


def test_compare_seed_runs_reports_each_position_separately() -> None:
    run_a = [
        _result("bb", "bb", (2.0, 1.0), best="a"),
        _result("btn", "btn", (2.0, 1.0), best="a"),
    ]
    run_b = [
        _result("bb", "bb", (1.0, 2.0), best="b"),
        _result("btn", "btn", (2.1, 1.0), best="a"),
    ]

    comparisons, summary = compare_seed_runs(run_a, run_b)

    assert len(comparisons) == 2
    assert summary["overall"]["strict_top1_agreement"] == 0.5
    assert summary["by_position"]["bb"]["strict_top1_agreement"] == 0.0
    assert summary["by_position"]["btn"]["strict_top1_agreement"] == 1.0


def _position(decision_id: str, position: str = "btn") -> dict:
    return {
        "schema": gate.SCHEMA,
        "turn": 3,
        "actor": position,
        "position": position,
        "is_btn": position == "btn",
        "first_actor": "bb",
        "board": {},
        "opponent_board": {},
        "dealt": [],
        "known_discards_self": [],
        "public_exclude": [],
        "public_action_history": [],
        "pilot": {
            "decision_id": decision_id,
            "cluster_id": f"root:{decision_id}",
            "stratum": f"{position}_joker0",
            "visible_joker_count": 0,
        },
    }


def test_sanitize_position_allows_only_public_action_history_fields() -> None:
    record = {
        **_position("source"),
        "trace": [
            {
                "turn": 1,
                "seat": "bb",
                "dealt": ["PRIVATE_DEALT"],
                "model_score": 99.0,
                "branch_role": "PRIVATE_BRANCH",
                "action": {
                    "placements": [["Ah", "top"], ["Kd", "mid"]],
                    "discard": "PRIVATE_DISCARD",
                },
            }
        ],
    }
    record.pop("public_action_history")

    sanitized = sanitize_position(record, source_index=0, root=7)

    assert sanitized["position_contract_version"] == "bb_first_v1"
    assert sanitized["public_action_history"] == [
        {"turn": 1, "actor": "bb", "placements": [["Ah", "top"], ["Kd", "middle"]]}
    ]
    serialized = json.dumps(sanitized["public_action_history"])
    assert "PRIVATE_DEALT" not in serialized
    assert "PRIVATE_DISCARD" not in serialized
    assert "PRIVATE_BRANCH" not in serialized
    assert "model_score" not in serialized

    record["public_action_history"] = [
        {
            "turn": 2,
            "actor": "btn",
            "placements": [["Qs", "bottom"]],
            "discard": "PRIVATE_EXISTING_DISCARD",
            "metadata": {"private": True},
        }
    ]
    sanitized_existing = sanitize_position(record, source_index=0, root=7)
    assert sanitized_existing["public_action_history"] == [
        {"turn": 2, "actor": "btn", "placements": [["Qs", "bottom"]]}
    ]


def test_sanitize_position_rejects_ambiguous_private_exclude() -> None:
    record = _position("source")
    record["exclude"] = ["Ah"]

    with pytest.raises(ValueError, match="opponent private discards"):
        sanitize_position(record, source_index=0, root=7)


def test_refinement_filters_disagreements_then_ranks_and_limits() -> None:
    positions = [_position("z", "bb"), _position("b"), _position("a"), _position("ok")]
    comparisons = [
        {"decision_id": "z", "strict_top1_agreement": False,
         "cross_regret_a_to_b": 1.0, "cross_regret_b_to_a": 0.0},
        {"decision_id": "b", "strict_top1_agreement": False,
         "cross_regret_a_to_b": 2.0, "cross_regret_b_to_a": 0.0},
        {"decision_id": "a", "strict_top1_agreement": False,
         "cross_regret_a_to_b": 2.0, "cross_regret_b_to_a": 1.0},
        {"decision_id": "ok", "strict_top1_agreement": True,
         "cross_regret_a_to_b": 10.0, "cross_regret_b_to_a": 10.0},
    ]

    selected, details = refine_positions(
        positions,
        position_filter="all",
        limit=2,
        comparisons=comparisons,
        disagreements_only=True,
    )

    assert [row["pilot"]["decision_id"] for row in selected] == ["a", "b"]
    assert details["after_position_filter"] == 4
    assert details["before_limit"] == 3
    assert details["requested_positions"] == 2
    assert details["ranked_by"] == "max_cross_regret_desc_then_decision_id"


def test_position_filter_is_applied_before_limit() -> None:
    positions = [_position("bb1", "bb"), _position("btn1"), _position("btn2")]

    selected, details = refine_positions(positions, position_filter="btn", limit=1)

    assert [row["pilot"]["decision_id"] for row in selected] == ["btn1"]
    assert details["after_position_filter"] == 2


def test_run_seed_resumes_only_an_exact_prefix_with_matching_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_evaluate(position, *, seed, outer_samples, response_inner_samples, **_kwargs):
        calls.append(position["pilot"]["decision_id"])
        actor = position["position"]
        return {
            "schema": gate.SCHEMA,
            "method": gate.METHOD,
            "information_model": gate.INFORMATION_MODEL,
            "hu_exact": False,
            "inner_t4_exact": True,
            "position": actor,
            "seed": seed,
            "outer_samples": outer_samples,
            "response_inner_samples": response_inner_samples if actor == "bb" else 0,
            "legal_actions": 1,
            "evaluated_actions": 1,
            "candidates": [{"action_key": "only"}],
            "elapsed_ms": 1.0,
        }

    monkeypatch.setattr(gate, "evaluate_t3_hu_sampled", fake_evaluate)
    output = tmp_path / "seed.jsonl"
    positions = [_position("a"), _position("b"), _position("c")]
    kwargs = {
        "seed": 11,
        "btn_outer": 4,
        "bb_outer": 2,
        "bb_inner": 1,
        "rust_solver_path": None,
        "rust_timeout_s": 30.0,
        "output_path": output,
    }

    first, first_resume = _run_seed(positions[:2], **kwargs)
    resumed, resume = _run_seed(positions, **kwargs)

    assert len(first) == 2
    assert len(resumed) == 3
    assert calls == ["a", "b", "c"]
    assert first_resume["existing_rows"] == 0
    assert resume["existing_rows"] == 2
    assert resume["new_rows"] == 1
    assert resume["resumed"] is True

    with pytest.raises(ValueError, match="not a requested-position prefix"):
        _run_seed([positions[1], positions[0], positions[2]], **kwargs)
    with pytest.raises(ValueError, match="run config mismatch"):
        _run_seed(positions, **{**kwargs, "btn_outer": 8})


def test_run_reuses_positions_and_records_refinement_and_resume_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions_path = tmp_path / "positions_input.jsonl"
    comparisons_path = tmp_path / "comparisons_input.jsonl"
    positions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [_position("low"), _position("high"), _position("bb", "bb")]),
        encoding="utf-8",
    )
    comparisons_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"decision_id": "low", "strict_top1_agreement": False,
                 "cross_regret_a_to_b": 1.0, "cross_regret_b_to_a": 0.0},
                {"decision_id": "high", "strict_top1_agreement": False,
                 "cross_regret_a_to_b": 3.0, "cross_regret_b_to_a": 0.0},
                {"decision_id": "bb", "strict_top1_agreement": False,
                 "cross_regret_a_to_b": 9.0, "cross_regret_b_to_a": 0.0},
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_seed(selected, *, seed, output_path, **_kwargs):
        rows = [_result(row["pilot"]["decision_id"], row["position"], (2.0, 1.0), best="a") for row in selected]
        return rows, {
            "seed": seed,
            "output": str(output_path),
            "requested_rows": len(selected),
            "existing_rows": 1,
            "new_rows": len(selected) - 1,
            "resumed": True,
            "complete_before_run": False,
            "run_config": {},
        }

    monkeypatch.setattr(gate, "_run_seed", fake_run_seed)
    output_dir = tmp_path / "out"
    summary = gate.run(
        Namespace(
            input="",
            positions_input=str(positions_path),
            comparisons_input=str(comparisons_path),
            disagreements_only=True,
            position_filter="btn",
            limit=1,
            output_dir=str(output_dir),
            selection_seed=1,
            max_root=99,
            eval_seeds="11,12",
            btn_outer=4,
            bb_outer=2,
            bb_inner=1,
            rust_solver="",
            rust_timeout_s=30.0,
        )
    )

    written_positions = [json.loads(line) for line in (output_dir / "positions.jsonl").read_text().splitlines()]
    assert [row["pilot"]["decision_id"] for row in written_positions] == ["high"]
    assert summary["selection"]["mode"] == "positions_input"
    assert summary["refinement"]["position_filter"] == "btn"
    assert summary["refinement"]["requested_positions"] == 1
    assert summary["resume"]["total_existing_rows"] == 2
    assert summary["resume"]["total_new_rows"] == 0
