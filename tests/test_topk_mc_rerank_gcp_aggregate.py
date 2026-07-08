"""GCP shard aggregation tests for the Stage8b TopK + MC rerank run."""

from __future__ import annotations

import csv
from pathlib import Path

from ofc_regular.aggregate_hu_turn2_stage8b_topk_mc_rerank import collect_seed_rows
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import aggregate_topk_seed_rows


def _seed_row(config_id: str, seed: int, ev: float, overrides: int) -> dict:
    return {
        "config_id": config_id,
        "top_k": 3,
        "mc_samples": 64,
        "confirm_mc_samples": 64,
        "min_rerank_delta": 0.5,
        "se_multiplier": 0.0,
        "allowed_seats": "first",
        "candidate_ev_rank_max": "",
        "min_gate_probability": "",
        "topk_score": "delta",
        "seed": seed,
        "paired_seeds": 300,
        "hands": 600,
        "ev_per_hand": ev,
        "paired_score_std_error": 0.05,
        "paired_seed_wins": 150,
        "paired_seed_losses": 120,
        "paired_seed_ties": 30,
        "decision_count": 500,
        "override_count": overrides,
        "avg_rerank_delta_on_override": 2.0,
        "avg_confirm_delta_on_override": 0.8,
        "confirmed_override_count": overrides,
    }


def test_collect_and_aggregate_shard_seed_rows(tmp_path: Path):
    for shard, seed in enumerate((11, 12, 13)):
        shard_dir = tmp_path / "results" / f"shard_{shard:03d}_k3_seed{seed}"
        shard_dir.mkdir(parents=True)
        row = _seed_row("k3_mc64_d0.5_se0_seatfirst", seed, ev=0.05 * (shard + 1), overrides=40)
        with (shard_dir / "seed_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    rows = collect_seed_rows(tmp_path)
    assert len(rows) == 3

    grid = aggregate_topk_seed_rows(rows)
    assert len(grid) == 1
    entry = grid[0]
    assert entry["paired_seeds"] == 900
    assert entry["seed_count"] == 3
    assert entry["override_count"] == 120
    assert entry["confirmed_override_count"] == 120
    assert abs(float(entry["aggregate_ev_per_hand"]) - 0.10) < 1e-9
    # Unbiased confirm gain propagates, distinct from the selection delta.
    assert abs(float(entry["avg_confirm_gain_on_override"]) - 0.8) < 1e-9
    assert abs(float(entry["avg_gain_on_override"]) - 2.0) < 1e-9
