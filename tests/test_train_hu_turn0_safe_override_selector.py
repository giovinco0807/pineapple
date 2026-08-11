import pickle

import numpy as np

from ofc_regular.hu_turn0_safe_selector import (
    load_hu_turn0_safe_selector_model,
    score_hu_turn0_safe_selector,
)
from ofc_regular.train_hu_turn0_safe_override_selector import (
    build_selector_data,
    row_to_feature_vector,
    stable_fold_assignments,
)


class FixedEstimator:
    def predict_proba(self, features):
        return np.asarray([[0.2, 0.8] for _ in range(len(features))])


def _action(top=(), middle=(), bottom=()):
    placements = [
        *[(card, "top") for card in top],
        *[(card, "middle") for card in middle],
        *[(card, "bottom") for card in bottom],
    ]
    return {
        "placements": placements,
        "discards": [],
        "score": 0.0,
        "next_board": {
            "top": list(top),
            "middle": list(middle),
            "bottom": list(bottom),
        },
    }


def _row(label="positive", target_id="row-1"):
    return {
        "target_id": target_id,
        "seat": "first",
        "hero_board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "cards_to_place": ["As", "Kh", "Qd", "Jc", "Ts"],
        "dead_cards": [],
        "candidate_action": _action(("As",), ("Kh", "Qd"), ("Jc", "Ts")),
        "baseline_action": _action(("Ts",), ("Qd", "Jc"), ("As", "Kh")),
        "candidate_action_index": 10,
        "baseline_action_index": 20,
        "candidate_topk": 60,
        "predicted_margin": 1.5,
        "candidate_delta_vs_baseline": 2.0 if label == "positive" else -2.0,
        "safe_override_label": label,
        "common_random_futures_verified": True,
        "action_mapping_verified": True,
    }


def test_feature_vector_modes_are_finite_and_action_sensitive():
    row = _row()
    meta = row_to_feature_vector(row, feature_mode="meta_only")
    compact = row_to_feature_vector(row, feature_mode="compact_candidate_delta_plus_meta")
    delta = row_to_feature_vector(row, feature_mode="delta_plus_meta")
    candidate_delta = row_to_feature_vector(row, feature_mode="candidate_delta_plus_meta")

    assert meta.shape == (4,)
    assert compact.shape == (92,)
    assert delta.shape[0] > meta.shape[0]
    assert candidate_delta.shape[0] > delta.shape[0]
    assert np.isfinite(candidate_delta).all()
    assert np.any(delta[:-4] != 0.0)


def test_build_data_uses_mc_labels_and_low_weight_gray():
    rows = [_row("positive", "p"), _row("negative", "n"), _row("gray", "g")]
    data = build_selector_data(rows, feature_mode="meta_only", gray_weight=0.05)

    assert data.labels.tolist() == [1, 0, 0]
    assert data.label_names == ("positive", "negative", "gray")
    assert data.weights[0] == 1.0
    assert data.weights[1] >= 2.0
    assert data.weights[2] == 0.05


def test_build_data_filters_seat_and_upweights_mc512():
    first = _row("positive", "first")
    first["future_samples"] = 512
    first_negative = _row("negative", "first-negative")
    second = _row("negative", "second")
    second["seat"] = "second"

    data = build_selector_data(
        [first, first_negative, second],
        feature_mode="meta_only",
        high_mc_weight=3.0,
        allowed_seats=("first",),
    )

    assert len(data.rows) == 2
    assert data.weights[0] == 3.0


def test_build_data_accepts_native_t0_numeric_label_ids():
    rows = []
    for label_id, target_id in ((1, "positive"), (0, "negative"), (-1, "gray")):
        row = _row("gray", target_id)
        row.pop("safe_override_label")
        row["safe_override_label_id"] = label_id
        rows.append(row)

    data = build_selector_data(rows, feature_mode="meta_only")

    assert data.label_names == ("positive", "negative", "gray")


def test_fold_assignments_are_stable_and_stratified():
    rows = [
        _row(label, f"{label}-{index}")
        for label in ("positive", "negative", "gray")
        for index in range(10)
    ]
    data = build_selector_data(rows, feature_mode="meta_only")

    first = stable_fold_assignments(data, folds=5)
    second = stable_fold_assignments(data, folds=5)

    assert first.tolist() == second.tolist()
    for label in ("positive", "negative", "gray"):
        folds = {
            int(first[index])
            for index, value in enumerate(data.label_names)
            if value == label
        }
        assert folds == {0, 1, 2, 3, 4}


def test_runtime_selector_loads_and_scores(tmp_path):
    path = tmp_path / "selector.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model_kind": "hu_turn0_safe_override_selector_sklearn",
                "feature_mode": "meta_only",
                "estimator": FixedEstimator(),
            },
            handle,
        )

    model = load_hu_turn0_safe_selector_model(path)
    assert score_hu_turn0_safe_selector(model, _row()) == 0.8
