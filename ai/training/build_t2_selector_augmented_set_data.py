"""Build selector-augmented set-reranker data for T2.

The base action-value set reranker consumes one feature vector per legal
candidate.  This converter appends runtime-only selector agreement features
from ``train_t2_selector_feature_ranker`` to the existing state/action feature
matrix so the neural set model can see the same fast model votes that the
sklearn diagnostics use.

Teacher EV stays only in ``scores.npy``; selector features are built from
runtime predictions, ranks, gaps, z-scores, and votes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.train_t2_selector_feature_ranker import (  # noqa: E402
    load_selector_scores,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
)


OPTIONAL_ARRAYS = [
    "sample_weights.npy",
    "group_sample_weights.npy",
    "positions.npy",
    "turns.npy",
    "route_tags.npy",
    "teacher_gaps.npy",
    "candidate_ranks.npy",
]


def copy_array_if_exists(src_dir: Path, dst_dir: Path, name: str) -> bool:
    src = src_dir / name
    if not src.exists():
        return False
    shutil.copy2(src, dst_dir / name)
    return True


def run(args: argparse.Namespace) -> None:
    started = time.time()
    data_spec = parse_named_path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {spec.name: spec for spec in [parse_model(value) for value in args.model]}
    selectors = [parse_selector(value) for value in args.selector]
    lgbm_selectors = [parse_lgbm_selector(value) for value in args.lgbm_selector]
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}

    selector_data = load_selector_scores(
        data_spec,
        model_specs,
        selectors,
        lgbm_selectors,
        loaded_models,
        loaded_lgbm,
        include_base=not args.no_base,
    )
    augmented = np.concatenate(
        [
            selector_data.x.astype(np.float32, copy=False),
            selector_data.selector_features.astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    np.save(out_dir / "states.npy", augmented)
    np.save(out_dir / "scores.npy", selector_data.scores.astype(np.float32, copy=False))
    np.save(out_dir / "base_scores.npy", selector_data.base_scores.astype(np.float32, copy=False))

    copied_arrays: list[str] = []
    for required in ("group_ids.npy",):
        shutil.copy2(data_spec.path / required, out_dir / required)
        copied_arrays.append(required)
    for name in OPTIONAL_ARRAYS:
        if copy_array_if_exists(data_spec.path, out_dir, name):
            copied_arrays.append(name)

    source_meta = {}
    source_meta_path = data_spec.path / "metadata.json"
    if source_meta_path.exists():
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    metadata = {
        "source_data": str(data_spec.path),
        "n_samples": int(augmented.shape[0]),
        "n_groups": int(len(selector_data.bounds)),
        "state_dim": int(augmented.shape[1]),
        "base_state_dim": int(selector_data.x.shape[1]),
        "selector_feature_dim": int(selector_data.selector_features.shape[1]),
        "selector_feature_names": selector_data.feature_names,
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": [
            {"name": spec.name, "model_names": list(spec.model_names), "gamma": float(spec.gamma)}
            for spec in selectors
        ],
        "lgbm_selectors": {name: str(path) for name, path in lgbm_selectors},
        "include_base_selector": not args.no_base,
        "copied_arrays": copied_arrays,
        "source_metadata": source_meta,
        "elapsed_seconds": time.time() - started,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "n_samples": int(augmented.shape[0]),
                "n_groups": int(len(selector_data.bounds)),
                "state_dim": int(augmented.shape[1]),
                "elapsed_seconds": metadata["elapsed_seconds"],
            },
            indent=2,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="name=dataset_dir")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selector and not args.lgbm_selector and args.no_base:
        raise ValueError("At least one selector source is required")
    run(args)


if __name__ == "__main__":
    main()
