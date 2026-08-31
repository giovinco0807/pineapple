from types import SimpleNamespace

import torch

from ai.models.action_value_reranker import ActionValueReranker
from ai.training.train_action_value_reranker import load_compatible_state_dict, select_best_metric


def test_load_compatible_state_dict_partially_expands_input_projection() -> None:
    source = ActionValueReranker(input_dim=520, hidden=32, n_blocks=1, dropout=0.0)
    target = ActionValueReranker(input_dim=617, hidden=32, n_blocks=1, dropout=0.0)

    with torch.no_grad():
        source.input_proj[0].weight.fill_(0.125)
        source.input_proj[0].bias.fill_(0.25)
        source.score_head.weight.fill_(0.5)

    target_before = target.input_proj[0].weight.detach().clone()

    missing, unexpected, partial, shape_mismatched = load_compatible_state_dict(
        target,
        source.state_dict(),
    )

    assert unexpected == []
    assert shape_mismatched == []
    assert missing == []
    assert partial == ["input_proj.0.weight:(32, 520)->(32, 617)"]
    assert torch.allclose(target.input_proj[0].weight[:, :520], source.input_proj[0].weight)
    assert torch.allclose(target.input_proj[0].weight[:, 520:], target_before[:, 520:])
    assert torch.allclose(target.input_proj[0].bias, source.input_proj[0].bias)
    assert torch.allclose(target.score_head.weight, source.score_head.weight)


def test_select_best_metric_uses_requested_topk_bucket() -> None:
    metrics = {
        "score_mae": 10.0,
        "group_top3": 0.5,
        "group_top5": 0.6,
        "group_top10": 0.7,
        "group_top15": 0.75,
        "group_top20": 0.8,
        "group_regret": 99.0,
        "group_top3_rerank_regret": 3.0,
        "group_top5_rerank_regret": 5.0,
        "group_top10_rerank_regret": 10.0,
        "group_top15_rerank_regret": 15.0,
        "group_top20_rerank_regret": 20.0,
    }

    def selected(target_topk: int) -> float:
        return select_best_metric(
            metrics,
            SimpleNamespace(selection_metric="topk", target_topk=target_topk),
        )

    assert selected(3) == 3.0 + (1.0 - 0.5) * 10.0 + 0.1
    assert selected(5) == 5.0 + (1.0 - 0.6) * 10.0 + 0.1
    assert selected(10) == 10.0 + (1.0 - 0.7) * 10.0 + 0.1
    assert selected(15) == 15.0 + (1.0 - 0.75) * 10.0 + 0.1
    assert selected(20) == 20.0 + (1.0 - 0.8) * 10.0 + 0.1
