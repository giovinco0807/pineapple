from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_street_policy_training_cli_v1 as subject
from ofc_regular import hu_m31_t3_street_policy_training_v1 as training
from ofc_regular.action_key import action_key, canonicalize_actions
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.street_policy_net_v1 import StreetPolicyNetV1Config


def _observation(index: int, seat: str) -> ActorObservation:
    cards = list(ALL_CARDS)
    random.Random(91_000 + index).shuffle(cards)
    cursor = 0

    def take(count: int) -> tuple[str, ...]:
        nonlocal cursor
        result = tuple(cards[cursor : cursor + count])
        cursor += count
        return result

    return ActorObservation(
        hero_board=Board.from_rows(take(2), take(3), take(4)),
        opponent_public_board=(
            Board.from_rows(take(2), take(3), take(4))
            if seat == "first"
            else Board.from_rows(take(2), take(4), take(5))
        ),
        dealt_cards=take(3),
        hero_private_discards=take(2),
        seat=seat,  # type: ignore[arg-type]
        street="T3",
        to_act_order=seat,  # type: ignore[arg-type]
    )


def _example(split: str, index: int, seat: str) -> training.PolicyTrainingExample:
    observation = _observation(index, seat)
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    keys = tuple(action_key(action).to_token() for action in actions)
    count = len(keys)
    baseline_index = count - 1
    q = tuple(1.0 - action_index / max(1, count - 1) for action_index in range(count))
    baseline = q[baseline_index]
    delta = tuple(value - baseline for value in q)
    logits = tuple(2.0**value for value in q)
    mass = sum(logits)
    teacher_policy = tuple(value / mass for value in logits)
    confirmed = tuple(delta)
    return training.PolicyTrainingExample(
        identity=f"{split}:{seat}:{index:03d}",
        split_role=split,
        seat=seat,
        observation=observation.to_dict(),
        legal_action_keys=keys,
        baseline_action_key=keys[baseline_index],
        action_q=q,
        baseline_delta=delta,
        teacher_policy=teacher_policy,
        state_value=max(q),
        downside_p95=tuple(0.0 for _ in keys),
        safe=tuple(1.0 if value > 0 else 0.0 for value in confirmed),
        confirmation_delta=confirmed,
    )


def _dataset() -> training.PolicyTrainingDataset:
    examples = []
    index = 0
    for split in training.SPLIT_ROLES:
        for seat in training.SEATS:
            examples.append(_example(split, index, seat))
            index += 1
    return training.build_policy_training_dataset(
        examples,
        source_dataset_identity_sha256="a" * 64,
        synthetic_cpu_smoke=True,
    )


def _run_config() -> subject.StreetPolicyRunConfig:
    return subject.StreetPolicyRunConfig(
        training_config=training.StreetPolicyTrainingConfig(
            seed=777,
            ensemble_size=2,
            batch_size=2,
            core_epochs=2,
            risk_epochs=1,
            core_learning_rate=1e-3,
            risk_learning_rate=1e-3,
            disagreement_multiplier=0.01,
            downside_multiplier=0.01,
            maximum_false_positive_rate=1.0,
            minimum_lock_fires_per_seat=1,
        ),
        model_config=StreetPolicyNetV1Config(
            card_embedding_dim=4,
            zone_embedding_dim=2,
            token_hidden_dim=6,
            context_hidden_dim=4,
            seat_embedding_dim=2,
            street_embedding_dim=2,
            state_hidden_dim=8,
            action_hidden_dim=8,
        ),
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_default_config_is_write_once_externally_hash_pinned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    receipt = subject.write_default_run_config(path)
    loaded = subject.load_run_config(path, expected_file_sha256=receipt["file_sha256"])
    assert loaded.to_dict() == subject.default_run_config().to_dict()
    with pytest.raises(FileExistsError, match="write-once"):
        subject.write_default_run_config(path)
    value = json.loads(path.read_text(encoding="ascii"))
    value["training_config"]["core_epochs"] = 99
    path.write_bytes(subject._canonical_bytes(value))  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="file SHA-256 changed"):
        subject.load_run_config(path, expected_file_sha256=receipt["file_sha256"])


def test_atomic_epoch_resume_matches_clean_run_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset()
    run_config = _run_config()
    source_receipt = {"dataset_identity_sha256": "a" * 64}
    clean_root = tmp_path / "clean"
    resumed_root = tmp_path / "resumed"
    clean = subject.execute_training_run(
        torch=torch,
        dataset=dataset,
        run_config=run_config,
        output_root=clean_root,
        requested_device="cpu",
        source_dataset_receipt=source_receipt,
        allow_synthetic_cpu_smoke=True,
    )

    original = subject._write_epoch  # type: ignore[attr-defined]
    interrupted = {"raised": False}

    def fail_after_first(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if (
            kwargs["stage"] == "core"
            and kwargs["epoch"] == 1
            and not interrupted["raised"]
        ):
            interrupted["raised"] = True
            raise RuntimeError("simulated process interruption")
        return result

    monkeypatch.setattr(subject, "_write_epoch", fail_after_first)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        subject.execute_training_run(
            torch=torch,
            dataset=dataset,
            run_config=run_config,
            output_root=resumed_root,
            requested_device="cpu",
            source_dataset_receipt=source_receipt,
            allow_synthetic_cpu_smoke=True,
        )
    assert (resumed_root / "core_epoch_001" / "epoch_receipt.json").is_file()
    assert not (resumed_root / "core_epoch_002").exists()

    monkeypatch.setattr(subject, "_write_epoch", original)
    resumed = subject.execute_training_run(
        torch=torch,
        dataset=dataset,
        run_config=run_config,
        output_root=resumed_root,
        requested_device="cpu",
        source_dataset_receipt=source_receipt,
        allow_synthetic_cpu_smoke=True,
    )
    assert resumed == clean
    assert _files(resumed_root) == _files(clean_root)

    repeated = subject.execute_training_run(
        torch=torch,
        dataset=dataset,
        run_config=run_config,
        output_root=resumed_root,
        requested_device="cpu",
        source_dataset_receipt=source_receipt,
        allow_synthetic_cpu_smoke=True,
    )
    assert repeated == resumed
    assert _files(resumed_root) == _files(clean_root)


def test_resume_fails_closed_on_epoch_tamper(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset()
    root = tmp_path / "run"
    receipt = {"dataset_identity_sha256": "a" * 64}
    subject.execute_training_run(
        torch=torch,
        dataset=dataset,
        run_config=_run_config(),
        output_root=root,
        requested_device="cpu",
        source_dataset_receipt=receipt,
        allow_synthetic_cpu_smoke=True,
    )
    path = root / "core_epoch_001" / "fit_receipt.json"
    value = json.loads(path.read_text(encoding="ascii"))
    value["losses"][0]["mean_total_loss"] += 0.1
    path.write_bytes(subject._canonical_bytes(value))  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="fit receipt"):
        subject.execute_training_run(
            torch=torch,
            dataset=dataset,
            run_config=_run_config(),
            output_root=root,
            requested_device="cpu",
            source_dataset_receipt=receipt,
            allow_synthetic_cpu_smoke=True,
        )


def test_synthetic_dataset_is_never_accepted_by_production_mode(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(PermissionError, match="source-replayed"):
        subject.execute_training_run(
            torch=torch,
            dataset=_dataset(),
            run_config=_run_config(),
            output_root=tmp_path / "run",
            requested_device="cpu",
            source_dataset_receipt={"dataset_identity_sha256": "a" * 64},
        )


def test_training_output_root_must_be_absolute() -> None:
    torch = pytest.importorskip("torch")
    relative_root = Path("relative-street-policy-output")
    assert not relative_root.exists()
    with pytest.raises(ValueError, match="output root must be absolute"):
        subject.execute_training_run(
            torch=torch,
            dataset=_dataset(),
            run_config=_run_config(),
            output_root=relative_root,
            requested_device="cpu",
            source_dataset_receipt={"dataset_identity_sha256": "a" * 64},
            allow_synthetic_cpu_smoke=True,
        )
    assert not relative_root.exists()


def test_training_output_root_rejects_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "junction-root"
    root.mkdir()
    original = subject._is_link_or_junction  # type: ignore[attr-defined]

    def mark_root_as_junction(path: Path) -> bool:
        return path == root or original(path)

    monkeypatch.setattr(subject, "_is_link_or_junction", mark_root_as_junction)
    with pytest.raises(ValueError, match="non-link directory"):
        subject.execute_training_run(
            torch=torch,
            dataset=_dataset(),
            run_config=_run_config(),
            output_root=root,
            requested_device="cpu",
            source_dataset_receipt={"dataset_identity_sha256": "a" * 64},
            allow_synthetic_cpu_smoke=True,
        )
    assert list(root.iterdir()) == []


def test_training_output_root_rejects_junction_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "junction-ancestor" / "run"
    ancestor = root.parent
    ancestor.mkdir()
    original = subject._is_link_or_junction  # type: ignore[attr-defined]

    def mark_ancestor_as_junction(path: Path) -> bool:
        return path == ancestor or original(path)

    monkeypatch.setattr(subject, "_is_link_or_junction", mark_ancestor_as_junction)
    with pytest.raises(ValueError, match="ancestry contains a link or junction"):
        subject.execute_training_run(
            torch=torch,
            dataset=_dataset(),
            run_config=_run_config(),
            output_root=root,
            requested_device="cpu",
            source_dataset_receipt={"dataset_identity_sha256": "a" * 64},
            allow_synthetic_cpu_smoke=True,
        )
    assert not root.exists()


def test_training_cli_consumes_the_same_exact_shard_map_as_merge_dataset(
    tmp_path: Path,
) -> None:
    plan = training.dataset_contract.build_dataset_plan()
    shard_map = {
        str(row["shard_id"]): str((tmp_path / str(row["shard_id"])).resolve())
        for row in plan["shards"]
    }
    path = tmp_path / "shard-map.json"
    path.write_bytes(subject._canonical_bytes(shard_map))

    loaded = subject._load_shard_map(path)
    assert loaded == {key: Path(value) for key, value in shard_map.items()}

    wrapped = tmp_path / "wrapped-shard-map.json"
    wrapped.write_bytes(
        subject._canonical_bytes(
            {
                "schema": "hu_m31_t3_dataset_shard_map_v1",
                "shard_directories": shard_map,
            }
        )
    )
    with pytest.raises(ValueError, match="exact absolute 360-shard map"):
        subject._load_shard_map(wrapped)
