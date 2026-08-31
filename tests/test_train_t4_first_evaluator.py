import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import ai.tutor.train_t4_first_evaluator as trainer
from ai.tutor.train_t4_first_evaluator import (
    HIDDEN,
    best_action_ranking_loss,
    parse_hidden,
    run,
    soft_regret_loss,
    validate_fit_dev,
)


def test_parse_hidden_accepts_explicit_architecture_and_rejects_empty_layers():
    assert parse_hidden("512,256,128") == (512, 256, 128)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_hidden("512,0,128")
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        parse_hidden("512,nope,128")


@pytest.mark.parametrize(
    ("extra", "expected_hidden", "expected_weight_decay", "expected_dev_dir"),
    [
        ([], HIDDEN, 0.01, None),
        (["--hidden", "512,256,128", "--weight-decay", "0.001"],
         (512, 256, 128), 0.001, None),
        (["--dev-data-dir", "external-dev"], HIDDEN, 0.01, Path("external-dev")),
    ],
)
def test_cli_preserves_defaults_and_wires_training_overrides(
    monkeypatch, extra, expected_hidden, expected_weight_decay, expected_dev_dir
):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "parameters": 0,
            "selected_epoch": 0,
            "test": None,
            "test_by_visible_jokers": None,
        }

    monkeypatch.setattr(trainer, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--data-dir", "unused", "--out-dir", "unused", *extra],
    )
    trainer.main()

    assert captured["hidden"] == expected_hidden
    assert captured["weight_decay"] == expected_weight_decay
    assert captured["dev_data_dir"] == expected_dev_dir


def test_soft_regret_rewards_putting_probability_on_the_best_action():
    truth = torch.tensor([3.0, 2.0, -1.0, 5.0, 4.5])
    correct = torch.tensor([4.0, 0.0, -3.0, 3.0, 0.0])
    reversed_order = torch.tensor([-3.0, 0.0, 4.0, 0.0, 3.0])

    good = soft_regret_loss(correct, truth, [3, 2], temperature=0.5)
    bad = soft_regret_loss(reversed_order, truth, [3, 2], temperature=0.5)

    assert good.item() < bad.item()


def test_soft_regret_is_shift_invariant_and_differentiable():
    truth = torch.tensor([1.0, 0.0, -2.0])
    prediction = torch.tensor([0.2, 0.1, -0.3], requires_grad=True)
    loss = soft_regret_loss(prediction, truth, [3], temperature=1.0)
    shifted = soft_regret_loss(prediction + 100.0, truth, [3], temperature=1.0)

    assert shifted.item() == pytest.approx(loss.item(), abs=1e-6)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0


def test_soft_regret_rejects_an_invalid_partition():
    with pytest.raises(ValueError, match="partition"):
        soft_regret_loss(torch.zeros(3), torch.zeros(3), [2], temperature=1.0)


def test_best_action_ranking_loss_rewards_the_correct_order():
    truth = torch.tensor([4.0, 1.0, -2.0, 3.0, 2.0])
    correct = torch.tensor([2.0, 1.0, -1.0, 5.0, 0.0])
    reversed_order = torch.tensor([-1.0, 1.0, 2.0, 0.0, 5.0])
    good = best_action_ranking_loss(correct, truth, [3, 2], temperature=0.5)
    bad = best_action_ranking_loss(reversed_order, truth, [3, 2], temperature=0.5)
    assert good < bad


def test_best_action_ranking_loss_is_shift_invariant_and_differentiable():
    prediction = torch.tensor([0.0, 0.5, -1.0], requires_grad=True)
    truth = torch.tensor([3.0, 1.0, 0.0])
    loss = best_action_ranking_loss(prediction, truth, [3], temperature=1.0)
    shifted = best_action_ranking_loss(
        prediction + 100.0, truth, [3], temperature=1.0
    )
    assert torch.allclose(loss, shifted, atol=1e-6)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_run_supports_soft_rank_without_best_rank_and_records_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    x = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    y = np.asarray([1.0, 0.0, 0.5, 1.5], dtype=np.float32)
    fit_roots = np.asarray([10, 10, 20, 20], dtype=np.int64)
    dev_roots = np.asarray([30, 30, 40, 40], dtype=np.int64)
    jokers = np.zeros(4, dtype=np.int8)
    np.savez(data_dir / "fit.npz", x=x, y=y, roots=fit_roots, jokers=jokers)
    np.savez(data_dir / "dev.npz", x=x, y=y, roots=dev_roots, jokers=jokers)

    report = run(
        data_dir=data_dir,
        out_dir=out_dir,
        epochs=1,
        batch_size=48,
        learning_rate=1e-3,
        device_name="cpu",
        select_on="regret",
        rank_weight=0.01,
        best_rank_weight=0.0,
        seed=7,
        evaluate_test=False,
        hidden=(4,),
        weight_decay=0.01,
    )

    checkpoint = torch.load(
        out_dir / "evaluator_best.pt", map_location="cpu", weights_only=False
    )
    assert report["test_touched_once"] is False
    assert checkpoint["rank_weight"] == pytest.approx(0.01)
    assert checkpoint["best_rank_weight"] == pytest.approx(0.0)


def test_external_dev_keeps_fit_only_scaler(tmp_path):
    data_dir = tmp_path / "data"
    dev_dir = tmp_path / "external_dev"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    dev_dir.mkdir()
    fit_x = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]], dtype=np.float32)
    dev_x = fit_x + 100.0
    y = np.asarray([1.0, 0.0, 0.5, 1.5], dtype=np.float32)
    jokers = np.zeros(4, dtype=np.int8)
    np.savez(
        data_dir / "fit.npz",
        x=fit_x,
        y=y,
        roots=np.asarray([10, 10, 20, 20], dtype=np.int64),
        jokers=jokers,
    )
    np.savez(
        dev_dir / "dev.npz",
        x=dev_x,
        y=y,
        roots=np.asarray([30, 30, 40, 40], dtype=np.int64),
        jokers=jokers,
    )

    report = run(
        data_dir=data_dir,
        dev_data_dir=dev_dir,
        out_dir=out_dir,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        device_name="cpu",
        select_on="regret",
        seed=11,
        evaluate_test=False,
        hidden=(4,),
    )

    checkpoint = torch.load(
        out_dir / "evaluator_best.pt", map_location="cpu", weights_only=False
    )
    assert torch.equal(checkpoint["input_mean"], torch.tensor([3.0, 5.0]))
    assert checkpoint["dev_data_dir"] == str(dev_dir)
    assert report["dev_data_dir"] == str(dev_dir)


def test_validate_fit_dev_rejects_width_drift_and_root_overlap():
    with pytest.raises(ValueError, match="feature width"):
        validate_fit_dev(
            torch.zeros((2, 3)),
            torch.zeros((2, 4)),
            np.asarray([1, 1]),
            np.asarray([2, 2]),
        )
    with pytest.raises(ValueError, match="root overlap"):
        validate_fit_dev(
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            np.asarray([1, 1]),
            np.asarray([1, 1]),
        )


def test_validate_fit_dev_preserves_legacy_no_roots_but_requires_them_externally():
    features = torch.zeros((2, 3))
    validate_fit_dev(features, features, None, None)

    with pytest.raises(ValueError, match="both need roots"):
        validate_fit_dev(features, features, None, None, require_roots=True)
    with pytest.raises(ValueError, match="both need roots"):
        validate_fit_dev(
            features,
            features,
            np.asarray([1, 1]),
            None,
        )
