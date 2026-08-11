import json

from ofc_regular.prepare_hu_turn0_safe_selector_dataset import prepare_dataset


def _row(target_id, label, delta, samples):
    return {
        "target_id": target_id,
        "seat": "first",
        "source_config_id": "main",
        "predicted_margin": 1.0,
        "candidate_delta_vs_baseline": delta,
        "safe_override_label": label,
        "future_samples": samples,
        "action_mapping_verified": True,
        "common_random_futures_verified": True,
    }


def test_prepare_dataset_replaces_refined_rows_and_reports_stability(tmp_path):
    mc32 = [
        _row("a", "positive", 3.0, 32),
        _row("b", "negative", -2.0, 32),
        _row("c", "gray", 0.1, 32),
    ]
    mc512 = [
        _row("a", "gray", 0.3, 512),
        _row("b", "negative", -1.5, 512),
    ]

    summary = prepare_dataset(mc32_rows=mc32, mc512_rows=mc512, output_dir=tmp_path)

    assert summary["mc32_rows"] == 3
    assert summary["mc512_rows"] == 2
    assert summary["mc32_positive_survival_rate"] == 0.0
    assert summary["action_mapping_verified"] is True
    broad = [
        json.loads(line)
        for line in (tmp_path / "selector_broad_with_mc512_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["target_id"] for row in broad] == ["a", "b", "c"]
    assert broad[0]["future_samples"] == 512
    assert broad[0]["selector_label_source"] == "mc512_refinement"
    assert broad[2]["future_samples"] == 32
    assert broad[2]["selector_label_source"] == "mc32_broad"
