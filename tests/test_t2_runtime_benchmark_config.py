from ai.tutor.benchmark_t2_t3_union_runtime import merged_runtime_models


def test_merged_runtime_models_allows_mode_ensemble_override():
    runtime_payload = {
        "models": {
            "action_value": "base.pt",
            "turn_model_ensembles": {
                "2": {
                    "checkpoints": ["old_a.pt", "old_b.pt"],
                    "weights": [0.5, 0.5],
                }
            },
        }
    }
    mode_payload = {
        "models": {
            "turn_model_ensembles": {
                "2": {
                    "checkpoints": ["new_a.pt", "new_b.pt"],
                    "weights": [0.25, 0.75],
                }
            }
        }
    }

    models = merged_runtime_models(runtime_payload, mode_payload)

    assert models["action_value"] == "base.pt"
    assert models["turn_model_ensembles"]["2"]["checkpoints"] == ["new_a.pt", "new_b.pt"]
    assert models["turn_model_ensembles"]["2"]["weights"] == [0.25, 0.75]
