from types import SimpleNamespace

import json

import numpy as np

from ai.training.sample_action_value_data import ARRAY_SPECS, build
from ai.training.train_action_value_reranker import ActionValueDataset


def _write_source(path, *, label_note="") -> None:
    path.mkdir(parents=True)
    n = 2
    arrays = {
        "states": np.ones((n, 3), dtype=np.float32),
        "scores": np.array([1.0, 0.0], dtype=np.float32),
        "bust": np.array([0.0, 1.0], dtype=np.float32),
        "fl": np.array([1.0, 0.0], dtype=np.float32),
        "fl_types": np.ones((n, 4), dtype=np.float32),
        "turns": np.array([3, 3], dtype=np.int16),
        "action_indices": np.array([1, 2], dtype=np.int16),
        "candidate_ranks": np.array([1, 2], dtype=np.int16),
        "route_tags": np.array([8, 0], dtype=np.int16),
        "base_scores": np.array([0.5, -0.5], dtype=np.float32),
        "sample_weights": np.ones(n, dtype=np.float32),
        "teacher_gaps": np.array([0.0, 1.0], dtype=np.float32),
    }
    for name in ARRAY_SPECS:
        np.save(path / f"{name}.npy", arrays[name])
    np.save(path / "group_ids.npy", np.array([0, 0], dtype=np.int64))
    metadata = {"n_samples": n, "n_records": 1}
    if label_note:
        metadata["label_note"] = label_note
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_mixed_data_marks_ev_only_sources_as_missing_aux_labels(tmp_path) -> None:
    broad = tmp_path / "broad"
    hard = tmp_path / "hard"
    output = tmp_path / "mixed"
    _write_source(broad, label_note="FL/bust labels are unavailable and set to zero.")
    _write_source(hard)

    build(
        SimpleNamespace(
            base_data=str(broad),
            base_samples=10,
            base_aux_labels="auto",
            extra_data=[str(hard)],
            extra_samples=0,
            extra_repeat=1,
            extra_aux_labels="auto",
            hard_group_jsonl=[],
            hard_repeat=1,
            hard_aux_labels="auto",
            output=str(output),
            seed=7,
        )
    )

    mask = np.load(output / "aux_label_mask.npy")
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["samples_by_aux_label"] == {"available": 2, "missing": 2}
    assert metadata["sources"][0]["aux_labels"] is False
    assert metadata["sources"][1]["aux_labels"] is True

    dataset = ActionValueDataset(output, np.arange(4), score_mean=0.0, score_std=1.0)
    assert float(dataset[0]["aux_label_mask"]) == 0.0
    assert float(dataset[2]["aux_label_mask"]) == 1.0
