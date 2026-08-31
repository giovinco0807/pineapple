import hashlib
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

import ai.tutor.t3_t4_distillation_dataset as dataset_module
import ai.tutor.t3_t4_distillation_teacher as teacher_module
import ai.tutor.t3_t4_distillation_training as training_module
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_distillation_dataset import (
    DistillationDatasetError,
    TARGET_METHOD_CODES,
    TRAINING_SAMPLE_KEYS,
    load_distillation_dataset,
    verify_distillation_dataset,
)
from ai.tutor.t3_t4_distillation_teacher import (
    MCCFR_SOLVER_METHOD,
    Q32_DENOMINATOR,
    build_split_assignment,
    build_t4_btn_exact_teacher_row,
    build_teacher_row,
    write_teacher_bundle,
)
from ai.tutor.t3_t4_infoset_encoder import INFOSET_VECTOR_DIM, semantic_action_ids
from ai.tutor.t3_t4_distillation_training import (
    DistillationTrainingConfig,
    DistillationTrainingError,
    masked_policy_value_q_loss,
    train_distillation_candidate,
    verify_training_artifact,
)
from ai.tutor.t4_btn_exact_resolver import (
    T4_BTN_EXACT_METHOD,
    resolve_t4_second_btn_exact,
)


BB_T0 = (
    ("4h", "top"),
    ("2c", "top"),
    ("3d", "top"),
    ("7c", "middle"),
    ("Qc", "bottom"),
)
BTN_T0 = (
    ("8c", "top"),
    ("5c", "top"),
    ("6d", "top"),
    ("9c", "middle"),
    ("As", "bottom"),
)
BB_T1 = (("7d", "middle"), ("8h", "middle"))
BTN_T1 = (("9d", "middle"), ("Jh", "middle"))
BB_T2 = (("9s", "middle"), ("Tc", "middle"))
BTN_T2 = (("Qs", "middle"), ("Kc", "middle"))
BB_T3 = (("Qd", "bottom"), ("X1", "bottom"))
BTN_T3 = (("2d", "bottom"), ("3h", "bottom"))
BB_T4 = (("Ad", "bottom"), ("Kd", "bottom"))

HISTORY_T3_FIRST = (
    (0, "bb", BB_T0),
    (0, "btn", BTN_T0),
    (1, "bb", BB_T1),
    (1, "btn", BTN_T1),
    (2, "bb", BB_T2),
    (2, "btn", BTN_T2),
)
HISTORY_T3_SECOND = HISTORY_T3_FIRST + ((3, "bb", BB_T3),)
HISTORY_T4_FIRST = HISTORY_T3_SECOND + ((3, "btn", BTN_T3),)
HISTORY_T4_SECOND = HISTORY_T4_FIRST + ((4, "bb", BB_T4),)

BB_BOARD_9 = (
    ("2c", "3d", "4h"),
    ("7c", "7d", "8h", "9s", "Tc"),
    ("Qc",),
)
BTN_BOARD_9 = (
    ("5c", "6d", "8c"),
    ("9c", "9d", "Jh", "Kc", "Qs"),
    ("As",),
)
BB_BOARD_11 = (BB_BOARD_9[0], BB_BOARD_9[1], ("Qc", "Qd", "X1"))
BTN_BOARD_11 = (BTN_BOARD_9[0], BTN_BOARD_9[1], ("2d", "3h", "As"))
BB_BOARD_13 = (
    BB_BOARD_9[0],
    BB_BOARD_9[1],
    ("Ad", "Kd", "Qc", "Qd", "X1"),
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _recall(actor: str, through_turn: int) -> PrivateRecall:
    public_cards = {
        "bb": {
            1: ("7d", "8h"),
            2: ("9s", "Tc"),
            3: ("Qd", "X1"),
        },
        "btn": {
            1: ("9d", "Jh"),
            2: ("Qs", "Kc"),
            3: ("2d", "3h"),
        },
    }[actor]
    discards = {
        "bb": {1: "6c", 2: "6s", 3: "X2"},
        "btn": {1: "4s", 2: "7h", 3: "4d"},
    }[actor]
    return PrivateRecall(
        dealt_by_turn=tuple(
            (turn, (*public_cards[turn], discards[turn]))
            for turn in range(1, through_turn + 1)
        ),
        discards_by_turn=tuple(
            (turn, discards[turn]) for turn in range(1, through_turn + 1)
        ),
    )


def _key(phase: str) -> InfoSetKey:
    spec = {
        "t3_first": (
            "bb",
            3,
            BB_BOARD_9,
            BTN_BOARD_9,
            HISTORY_T3_FIRST,
            _recall("bb", 2),
            ("Qd", "X1", "X2"),
        ),
        "t3_second": (
            "btn",
            3,
            BB_BOARD_11,
            BTN_BOARD_9,
            HISTORY_T3_SECOND,
            _recall("btn", 2),
            ("2d", "3h", "4d"),
        ),
        "t4_first": (
            "bb",
            4,
            BB_BOARD_11,
            BTN_BOARD_11,
            HISTORY_T4_FIRST,
            _recall("bb", 3),
            ("Ad", "Jd", "Kd"),
        ),
        "t4_second": (
            "btn",
            4,
            BB_BOARD_13,
            BTN_BOARD_11,
            HISTORY_T4_SECOND,
            _recall("btn", 3),
            ("Ah", "Qh", "Th"),
        ),
    }[phase]
    actor, turn, board_bb, board_btn, history, recall, draw = spec
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor=actor,
        turn=turn,
        phase=phase,
        board_bb=board_bb,
        board_btn=board_btn,
        public_action_history=history,
        own_recall=recall,
        current_draw=draw,
        fantasy_state=None,
    )


@lru_cache(maxsize=None)
def _assignment_for_split(split: str, tag: str) -> dict:
    for index in range(100_000):
        assignment = build_split_assignment(
            full_deal_commitment_sha256=_hash(f"{tag}:full:{index}"),
            public_root_family_commitment_sha256=_hash(f"{tag}:family:{index}"),
        )
        if assignment["split"] == split:
            return assignment
    raise AssertionError(f"could not find deterministic {split} assignment")


def _bindings(tag: str, *, exact: bool = False) -> dict[str, str]:
    fields = [
        "public_root_commitment_sha256",
        "public_root_mixture_sha256",
        "range_content_sha256",
        "range_build_sha256",
        "behavior_model_sha256",
        "source_manifest_sha256",
    ]
    if not exact:
        fields.extend(
            [
                "solver_source_sha256",
                "solver_config_sha256",
                "solver_checkpoint_sha256",
            ]
        )
    return {field: _hash(f"{tag}:{field}") for field in fields}


def _lineage(tag: str) -> dict:
    return {
        "descendant_public_path_sha256": _hash(f"{tag}:descendant"),
        "restricted_variant_root_commitment_sha256": _hash(f"{tag}:variant"),
        "restricted_private_type_commitment_sha256": _hash(f"{tag}:private"),
        "seat_swap_index": 0,
        "suit_augmentation_index": 0,
    }


def _labels(key: InfoSetKey) -> tuple[dict[str, str], dict[str, dict[str, float | int]]]:
    legal = [action_id for action_id in semantic_action_ids(key) if action_id]
    strategy = {
        action_id: f"{index + 1}/{len(legal)}"
        for index, action_id in enumerate(legal)
    }
    moments: dict[str, dict[str, float | int]] = {}
    for index, action_id in enumerate(legal):
        mean = float(index - 1)
        values = (mean - 1.0, mean, mean, mean + 1.0)
        moments[action_id] = {
            "count": len(values),
            "sum": sum(values),
            "sum_squares": sum(value * value for value in values),
        }
    return strategy, moments


def _mccfr_row(
    phase: str,
    split: str,
    tag: str,
    *,
    visits: int = 1234,
) -> dict:
    key = _key(phase)
    strategy, moments = _labels(key)
    return build_teacher_row(
        key,
        split_assignment=_assignment_for_split(split, tag),
        lineage=_lineage(tag),
        bindings=_bindings(tag),
        solver_method=MCCFR_SOLVER_METHOD,
        solver_iterations_completed=10_000,
        solver_seed_index=0,
        payoff_seed_index=0,
        average_strategy_by_action_id=strategy,
        action_payoff_moments_by_action_id=moments,
        infoset_visit_count=visits,
        infoset_reach_probability=Fraction(1, 100),
    )


def _exact_row(split: str, tag: str) -> dict:
    key = _key("t4_second")
    return build_t4_btn_exact_teacher_row(
        key,
        resolution=resolve_t4_second_btn_exact(key),
        split_assignment=_assignment_for_split(split, tag),
        lineage=_lineage(tag),
        bindings=_bindings(tag, exact=True),
    )


@pytest.fixture()
def complete_bundle(tmp_path):
    rows = [
        _mccfr_row("t3_first", "fit", "fit-t3"),
        _exact_row("fit", "fit-t4-exact"),
        _mccfr_row("t3_second", "dev", "dev-t3"),
        _mccfr_row("t4_first", "test", "test-t4"),
    ]
    return write_teacher_bundle(tmp_path / "teacher", rows, shard_size=2)


def test_loader_materializes_locked_policy_value_q_targets(complete_bundle):
    dataset = load_distillation_dataset(
        complete_bundle.root,
        expected_teacher_manifest_sha256=complete_bundle.manifest["manifest_sha256"],
    )
    assert {name: len(dataset.for_split(name)) for name in ("fit", "dev", "test")} == {
        "fit": 2,
        "dev": 1,
        "test": 1,
    }
    assert dataset.manifest["opt_in_only"] is True
    assert dataset.manifest["training_performed"] is False
    assert dataset.manifest["promotion_eligible"] is False
    assert dataset.manifest["serving_changed"] is False
    audit = dataset.manifest["provenance_overlap_audit"]
    assert audit["public_root_family_disjoint"] is True
    assert audit["cross_split_overlap_count"] == 0
    assert audit["random_resplit_performed"] is False

    fit = dataset.for_split("fit")
    assert fit.states.shape == (2, INFOSET_VECTOR_DIM)
    assert fit.policy_targets.shape == (2, 27)
    assert fit.q_targets.shape == (2, 27)
    assert np.all(fit.policy_q32.sum(axis=1) == Q32_DENOMINATOR)
    assert np.array_equal(fit.q_target_masks, fit.legal_action_masks)
    assert np.allclose(
        fit.value_targets,
        np.sum(
            fit.policy_targets * np.where(fit.q_target_masks, fit.q_targets, 0.0),
            axis=1,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    assert set(int(value) for value in fit.target_method_codes) == {
        TARGET_METHOD_CODES[MCCFR_SOLVER_METHOD],
        TARGET_METHOD_CODES[T4_BTN_EXACT_METHOD],
    }
    exact_index = int(
        np.flatnonzero(
            fit.target_method_codes == TARGET_METHOD_CODES[T4_BTN_EXACT_METHOD]
        )[0]
    )
    assert np.count_nonzero(fit.policy_q32[exact_index]) == 1
    assert np.all(
        fit.q_standard_errors[exact_index][fit.q_target_masks[exact_index]] == 0.0
    )
    assert verify_distillation_dataset(dataset) is dataset


def test_training_interface_is_hidden_information_free_and_dataloader_compatible(
    complete_bundle,
):
    torch = pytest.importorskip("torch")
    dataset = load_distillation_dataset(
        complete_bundle.root,
        expected_teacher_manifest_sha256=complete_bundle.manifest["manifest_sha256"],
    )
    fit = dataset.for_split("fit")
    sample = fit[0]
    assert tuple(sample) == TRAINING_SAMPLE_KEYS
    assert dataset.manifest["model_feature_keys"] == ["state"]
    forbidden_tokens = ("commitment", "private", "opponent", "full_deal", "root_family")
    assert not any(
        token in key.lower() for key in sample for token in forbidden_tokens
    )
    assert not any(
        token in key.lower()
        for key in fit.training_batch()
        for token in forbidden_tokens
    )
    assert fit.provenance.full_deal_commitment_sha256
    assert fit.provenance.public_root_family_commitment_sha256

    batch = next(iter(torch.utils.data.DataLoader(fit, batch_size=2, shuffle=False)))
    assert batch["state"].shape == (2, INFOSET_VECTOR_DIM)
    assert batch["policy_target"].shape == (2, 27)
    assert batch["q_target_mask"].dtype == torch.bool


def test_loader_requires_pinned_manifest_and_nonempty_requested_splits(tmp_path):
    row = _mccfr_row("t3_first", "fit", "fit-only")
    bundle = write_teacher_bundle(tmp_path / "teacher", [row], shard_size=1)
    with pytest.raises(DistillationDatasetError, match="pinned SHA-256"):
        load_distillation_dataset(
            bundle.root,
            expected_teacher_manifest_sha256="0" * 64,
            required_splits=("fit",),
        )
    with pytest.raises(DistillationDatasetError, match="required locked splits are empty"):
        load_distillation_dataset(
            bundle.root,
            expected_teacher_manifest_sha256=bundle.manifest["manifest_sha256"],
        )
    loaded = load_distillation_dataset(
        bundle.root,
        expected_teacher_manifest_sha256=bundle.manifest["manifest_sha256"],
        required_splits=("fit",),
    )
    assert len(loaded.for_split("fit")) == 1
    assert len(loaded.for_split("dev")) == 0
    assert len(loaded.for_split("test")) == 0


def test_structural_fail_rows_never_enter_training_dataset(tmp_path):
    row = _mccfr_row("t3_first", "fit", "low-quality", visits=0)
    assert row["quality"]["structural_quality_passed"] is False
    bundle = write_teacher_bundle(tmp_path / "teacher", [row], shard_size=1)
    with pytest.raises(DistillationDatasetError, match="structural-fail"):
        load_distillation_dataset(
            bundle.root,
            expected_teacher_manifest_sha256=bundle.manifest["manifest_sha256"],
            required_splits=("fit",),
        )


def test_verifier_detects_tensor_mutation_and_cross_split_root_family_forgery(
    complete_bundle,
):
    dataset = load_distillation_dataset(
        complete_bundle.root,
        expected_teacher_manifest_sha256=complete_bundle.manifest["manifest_sha256"],
    )
    fit = dataset.for_split("fit")
    with pytest.raises(ValueError):
        fit.states[0, 0] = 1.0 - fit.states[0, 0]

    dev = dataset.for_split("dev")
    forged_provenance = replace(
        dev.provenance,
        public_root_family_commitment_sha256=(
            fit.provenance.public_root_family_commitment_sha256[0],
        ),
    )
    forged_dev = replace(dev, provenance=forged_provenance)
    forged_splits = MappingProxyType(
        {"fit": fit, "dev": forged_dev, "test": dataset.for_split("test")}
    )
    forged_dataset = replace(dataset, splits=forged_splits)
    with pytest.raises(
        DistillationDatasetError,
        match="cross-split provenance overlap|public-root family",
    ):
        verify_distillation_dataset(forged_dataset)


def test_public_dataset_boundaries_reject_paired_verifier_alias_substitution(
    complete_bundle,
    monkeypatch,
):
    loaded = dataset_module.load_distillation_dataset(
        complete_bundle.root,
        expected_teacher_manifest_sha256=complete_bundle.manifest["manifest_sha256"],
    )
    calls = []

    def forged_verifier(_path):
        calls.append("forged")
        return complete_bundle

    monkeypatch.setattr(dataset_module, "verify_teacher_bundle", forged_verifier)
    monkeypatch.setattr(dataset_module, "_VERIFY_TEACHER_BUNDLE", forged_verifier)
    with pytest.raises(DistillationDatasetError, match="runtime callable alias drift"):
        dataset_module.load_distillation_dataset(
            complete_bundle.root,
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
        )
    with pytest.raises(DistillationDatasetError, match="runtime callable alias drift"):
        dataset_module.verify_distillation_dataset(loaded)
    assert calls == []


def test_teacher_verifier_same_object_code_drift_is_rejected_transitively(
    complete_bundle,
    tmp_path,
    monkeypatch,
):
    loaded = dataset_module.load_distillation_dataset(
        complete_bundle.root,
        expected_teacher_manifest_sha256=complete_bundle.manifest[
            "manifest_sha256"
        ],
    )

    def forged(*_args, **_kwargs):
        raise AssertionError("forged teacher verifier must not execute")

    verifier = teacher_module.verify_teacher_bundle
    assert verifier.__closure__ is None and forged.__closure__ is None
    monkeypatch.setattr(verifier, "__code__", forged.__code__)

    with pytest.raises(
        DistillationDatasetError,
        match="runtime callable semantics drift: verify_teacher_bundle",
    ):
        dataset_module.load_distillation_dataset(
            complete_bundle.root,
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
        )
    with pytest.raises(
        DistillationDatasetError,
        match="runtime callable semantics drift: verify_teacher_bundle",
    ):
        dataset_module.verify_distillation_dataset(loaded)
    with pytest.raises(
        DistillationTrainingError,
        match="runtime callable semantics drift: load_distillation_dataset",
    ):
        training_module.train_distillation_candidate(
            complete_bundle.root,
            tmp_path / "candidate",
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
            config=_tiny_training_config(),
        )
    with pytest.raises(
        DistillationTrainingError,
        match="runtime callable semantics drift: load_distillation_dataset",
    ):
        training_module.verify_training_artifact(
            tmp_path / "missing",
            teacher_bundle_dir=complete_bundle.root,
            expected_manifest_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("mutated_module", "mutated_name", "consumer_module"),
    (
        (
            "ai.tutor.t3_t4_distillation_teacher",
            "verify_teacher_bundle",
            "ai.tutor.t3_t4_distillation_dataset",
        ),
        (
            "ai.tutor.t3_t4_distillation_teacher",
            "verify_teacher_bundle",
            "ai.tutor.t3_t4_distillation_training",
        ),
        (
            "ai.tutor.t3_t4_distillation_teacher",
            "_read_canonical",
            "ai.tutor.t3_t4_distillation_dataset",
        ),
        (
            "ai.tutor.t3_t4_infoset_encoder",
            "decode_infoset_key",
            "ai.tutor.t3_t4_distillation_dataset",
        ),
    ),
)
def test_pre_consumer_import_function_drift_fails_closed_in_fresh_process(
    mutated_module,
    mutated_name,
    consumer_module,
):
    script = f"""
import importlib

module = importlib.import_module({mutated_module!r})
module._IMPORT_ORDER_SENTINEL = 0

def forged(*_args, **_kwargs):
    global _IMPORT_ORDER_SENTINEL
    _IMPORT_ORDER_SENTINEL += 1
    raise RuntimeError("forged callable executed")

target = getattr(module, {mutated_name!r})
assert target.__closure__ is None
target.__code__ = forged.__code__
try:
    importlib.import_module({consumer_module!r})
except Exception as exc:
    assert "pre-consumer-import runtime" in str(exc), repr(exc)
else:
    raise AssertionError("consumer import accepted forged runtime semantics")
assert module._IMPORT_ORDER_SENTINEL == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("mutated_module", "mutated_name", "primary_name", "mirror_name", "consumer"),
    (
        (
            "ai.tutor.t3_t4_distillation_teacher",
            "verify_teacher_bundle",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
            "ai.tutor.t3_t4_distillation_dataset",
        ),
        (
            "ai.tutor.t3_t4_distillation_teacher",
            "verify_teacher_bundle",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
            "ai.tutor.t3_t4_distillation_training",
        ),
        (
            "ai.tutor.t3_t4_infoset_encoder",
            "decode_infoset_key",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
            "ai.tutor.t3_t4_distillation_dataset",
        ),
        (
            "ai.tutor.t3_t4_infoset_encoder",
            "decode_infoset_key",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
            "ai.tutor.t3_t4_distillation_training",
        ),
    ),
)
def test_pre_consumer_import_paired_anchor_rewrite_fails_closed(
    mutated_module,
    mutated_name,
    primary_name,
    mirror_name,
    consumer,
):
    script = f"""
import importlib
from ai.tutor.runtime_semantic_anchor import build_module_function_anchor

module = importlib.import_module({mutated_module!r})
module._PAIRED_REWRITE_SENTINEL = 0

def forged(*_args, **_kwargs):
    global _PAIRED_REWRITE_SENTINEL
    _PAIRED_REWRITE_SENTINEL += 1
    raise RuntimeError("forged callable executed")

target = getattr(module, {mutated_name!r})
assert target.__closure__ is None
target.__code__ = forged.__code__
replacement = build_module_function_anchor(vars(module))
setattr(module, {primary_name!r}, replacement)
setattr(module, {mirror_name!r}, replacement)
try:
    importlib.import_module({consumer!r})
except Exception as exc:
    assert "registry drift" in str(exc), repr(exc)
else:
    raise AssertionError("consumer import accepted a rebuilt paired anchor")
assert module._PAIRED_REWRITE_SENTINEL == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_module_anchor_registry_rejects_duplicate_and_namespace_rebind():
    script = """
from ai.tutor.runtime_semantic_anchor import register_module_function_anchor

namespace = {"__name__": "tests.synthetic_anchor_a"}
exec("def target(): return 1", namespace)
register_module_function_anchor(namespace["__name__"], namespace)
try:
    register_module_function_anchor(namespace["__name__"], namespace)
except RuntimeError as exc:
    assert "duplicate registration" in str(exc), repr(exc)
else:
    raise AssertionError("duplicate registry registration was accepted")
namespace["__name__"] = "tests.synthetic_anchor_b"
try:
    register_module_function_anchor(namespace["__name__"], namespace)
except RuntimeError as exc:
    assert "namespace rebind" in str(exc), repr(exc)
else:
    raise AssertionError("registry namespace rebind was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_module_owned_anchor_alias_drift_fails_closed(
    complete_bundle,
    monkeypatch,
):
    monkeypatch.setattr(
        teacher_module,
        "_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        (),
    )
    with pytest.raises(
        DistillationDatasetError,
        match="transitive runtime anchor alias drift: teacher",
    ):
        dataset_module.load_distillation_dataset(
            complete_bundle.root,
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
        )


@pytest.mark.parametrize(
    ("producer", "label", "primary_name", "mirror_name"),
    (
        (
            teacher_module,
            "teacher",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        ),
        (
            dataset_module._encoder_module,
            "encoder",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        ),
    ),
)
@pytest.mark.parametrize("drift", ("primary", "mirror", "both"))
def test_trainer_rejects_post_import_producer_anchor_drift_before_io(
    producer,
    label,
    primary_name,
    mirror_name,
    drift,
    tmp_path,
    monkeypatch,
):
    if drift in {"primary", "both"}:
        monkeypatch.setattr(producer, primary_name, ())
    if drift in {"mirror", "both"}:
        monkeypatch.setattr(producer, mirror_name, ())

    expected = rf"trainer transitive runtime anchor drift:.*{label}"
    with pytest.raises(DistillationTrainingError, match=expected):
        training_module.train_distillation_candidate(
            tmp_path / "missing-teacher",
            tmp_path / "candidate",
            expected_teacher_manifest_sha256="0" * 64,
            config=_tiny_training_config(),
        )
    with pytest.raises(DistillationTrainingError, match=expected):
        training_module.verify_training_artifact(
            tmp_path / "missing-artifact",
            teacher_bundle_dir=tmp_path / "missing-teacher",
            expected_manifest_sha256="0" * 64,
        )


def _tiny_training_config() -> DistillationTrainingConfig:
    return DistillationTrainingConfig(
        seed=314159,
        epochs=2,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_dim=8,
        residual_blocks=1,
        policy_loss_weight=1.0,
        value_loss_weight=0.2,
        q_loss_weight=0.5,
        q_standard_error_floor=0.1,
        q_precision_weight_cap=50.0,
        huber_delta=1.0,
        evaluation_batch_size=2,
    )


def test_deterministic_fit_dev_test_once_training_and_fresh_verifier(
    complete_bundle,
    tmp_path,
):
    first = train_distillation_candidate(
        complete_bundle.root,
        tmp_path / "candidate-a",
        expected_teacher_manifest_sha256=complete_bundle.manifest["manifest_sha256"],
        config=_tiny_training_config(),
    )
    second = train_distillation_candidate(
        complete_bundle.root,
        tmp_path / "candidate-b",
        expected_teacher_manifest_sha256=complete_bundle.manifest["manifest_sha256"],
        config=_tiny_training_config(),
    )
    assert first.manifest["manifest_sha256"] == second.manifest["manifest_sha256"]
    first_files = {path.name: path.read_bytes() for path in first.root.iterdir()}
    second_files = {path.name: path.read_bytes() for path in second.root.iterdir()}
    assert first_files == second_files

    manifest = first.manifest
    assert manifest["promotion_eligible"] is False
    assert manifest["promotion_gate_evaluated"] is False
    assert manifest["runtime_allowed"] is False
    assert manifest["serving_changed"] is False
    assert manifest["split_usage"] == {
        "fit": "gradient_updates_only",
        "dev": "checkpoint_selection_only",
        "test": "selected_checkpoint_single_training_pipeline_evaluation_only",
        "random_resplit_performed": False,
        "training_pipeline_test_evaluation_count": 1,
        "fresh_verifier_test_replay_is_integrity_only": True,
        "fresh_verifier_cannot_change_checkpoint_selection": True,
    }
    assert set(manifest["source_file_sha256"]) == {
        "dataset",
        "encoder",
        "model",
        "runtime_anchor",
        "teacher",
        "trainer",
    }
    assert manifest["source_file_sha256"]["model"] == manifest[
        "source_file_sha256"
    ]["trainer"]
    assert manifest["live_semantic_bindings"]["action_semantics_sha256"]
    assert manifest["loss_contract"]["illegal_policy_logits_excluded"] is True
    assert manifest["loss_contract"]["illegal_q_predictions_excluded"] is True
    assert "se_squared" in manifest["loss_contract"]["q_standard_error_weighting"]

    history = first.history
    assert history["epoch_count"] == 2
    assert all(record["fit_rows_consumed"] == 2 for record in history["epoch_records"])
    assert history["training_pipeline_test_evaluation_count"] == 1
    event_kinds = [event["kind"] for event in history["evaluation_events"]]
    assert event_kinds == [
        "dev_checkpoint_candidate_evaluation",
        "dev_checkpoint_candidate_evaluation",
        "dev_checkpoint_selected",
        "selected_checkpoint_fit_evaluation",
        "selected_checkpoint_dev_evaluation",
        "selected_checkpoint_test_evaluation",
    ]
    assert sum(event["split"] == "test" for event in history["evaluation_events"]) == 1
    for split in ("fit", "dev", "test"):
        metrics = first.metrics["splits"][split]
        assert metrics["by_public_root_family"]
        assert metrics["by_phase"]
        assert metrics["by_actor"]
        assert metrics["by_joker"]
        assert metrics["by_phase_actor_joker"]

    verified = verify_training_artifact(
        first.root,
        teacher_bundle_dir=complete_bundle.root,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    assert verified.manifest == manifest

    checkpoint = first.root / manifest["checkpoint"]["relative_path"]
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    with pytest.raises(DistillationTrainingError, match="checkpoint file content SHA"):
        verify_training_artifact(
            first.root,
            teacher_bundle_dir=complete_bundle.root,
            expected_manifest_sha256=manifest["manifest_sha256"],
        )


def test_masked_policy_q_loss_ignores_illegal_outputs_and_uses_q_uncertainty():
    config = _tiny_training_config()
    mask = np.zeros((1, 27), dtype=bool)
    mask[0, :2] = True
    policy = np.zeros((1, 27), dtype=np.float32)
    policy[0, :2] = 0.5
    q_target = np.full((1, 27), np.nan, dtype=np.float32)
    q_target[0, :2] = 0.0
    q_se = np.full((1, 27), np.nan, dtype=np.float32)
    q_se[0, :2] = (0.1, 10.0)
    batch = {
        "state": np.zeros((1, INFOSET_VECTOR_DIM), dtype=np.float32),
        "legal_action_mask": mask,
        "policy_target": policy,
        "value_target": np.zeros(1, dtype=np.float32),
        "q_target": q_target,
        "q_target_mask": mask,
        "q_standard_error": q_se,
    }
    base = {
        "policy_logits": torch.zeros((1, 27), dtype=torch.float32),
        "value": torch.zeros(1, dtype=torch.float32),
        "q_values": torch.zeros((1, 27), dtype=torch.float32),
    }
    illegal_changed = {key: value.clone() for key, value in base.items()}
    illegal_changed["policy_logits"][0, 2:] = 1e6
    illegal_changed["q_values"][0, 2:] = -1e6
    base_loss = masked_policy_value_q_loss(base, batch, config)
    changed_loss = masked_policy_value_q_loss(illegal_changed, batch, config)
    assert torch.equal(base_loss["policy"], changed_loss["policy"])
    assert torch.equal(base_loss["q"], changed_loss["q"])

    low_se_error = {key: value.clone() for key, value in base.items()}
    low_se_error["q_values"][0, 0] = 1.0
    high_se_error = {key: value.clone() for key, value in base.items()}
    high_se_error["q_values"][0, 1] = 1.0
    assert masked_policy_value_q_loss(low_se_error, batch, config)["q"] > (
        masked_policy_value_q_loss(high_se_error, batch, config)["q"]
    )


def test_content_addressed_writer_never_overwrites_racing_or_existing_bytes(
    tmp_path,
    monkeypatch,
):
    stable = tmp_path / "stable.bin"
    training_module._atomic_write_bytes(stable, b"same")
    training_module._atomic_write_bytes(stable, b"same")
    with pytest.raises(DistillationTrainingError, match="artifact collision"):
        training_module._atomic_write_bytes(stable, b"different")
    assert stable.read_bytes() == b"same"

    racing = tmp_path / "racing.bin"

    def racing_link(_source, destination):
        destination_path = type(racing)(destination)
        destination_path.write_bytes(b"racer-won")
        raise FileExistsError(destination)

    monkeypatch.setattr(training_module.os, "link", racing_link)
    with pytest.raises(DistillationTrainingError, match="artifact collision"):
        training_module._atomic_write_bytes(racing, b"candidate")
    assert racing.read_bytes() == b"racer-won"


def test_trainer_public_boundary_rejects_dataset_alias_substitution(
    complete_bundle,
    tmp_path,
    monkeypatch,
):
    calls = []

    def forged_dataset_boundary(*_args, **_kwargs):
        calls.append("forged")
        raise AssertionError("must not execute")

    monkeypatch.setattr(
        training_module, "_LOAD_DISTILLATION_DATASET", forged_dataset_boundary
    )
    monkeypatch.setattr(
        training_module, "_VERIFY_DISTILLATION_DATASET", forged_dataset_boundary
    )
    with pytest.raises(DistillationTrainingError, match="runtime callable alias drift"):
        training_module.train_distillation_candidate(
            complete_bundle.root,
            tmp_path / "candidate",
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
            config=_tiny_training_config(),
        )
    with pytest.raises(DistillationTrainingError, match="runtime callable alias drift"):
        training_module.verify_training_artifact(
            tmp_path / "missing",
            teacher_bundle_dir=complete_bundle.root,
            expected_manifest_sha256="0" * 64,
        )
    assert calls == []


def test_trainer_public_boundary_rejects_live_action_alias_substitution(
    complete_bundle,
    tmp_path,
    monkeypatch,
):
    calls = []

    def forged_mask(_key):
        calls.append("forged")
        return np.ones(27, dtype=bool)

    monkeypatch.setattr(training_module, "legal_action_mask", forged_mask)
    monkeypatch.setattr(training_module, "_LEGAL_ACTION_MASK", forged_mask)
    with pytest.raises(DistillationTrainingError, match="runtime callable alias drift"):
        training_module.train_distillation_candidate(
            complete_bundle.root,
            tmp_path / "candidate",
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
            config=_tiny_training_config(),
        )
    assert calls == []


def test_trainer_public_boundary_rejects_in_place_model_method_code_drift(
    complete_bundle,
    tmp_path,
    monkeypatch,
):
    def forged_forward(_self, _states):
        raise AssertionError("forged model forward must not execute")

    canonical_forward = training_module.T3T4PolicyValueQNet.forward
    monkeypatch.setattr(canonical_forward, "__code__", forged_forward.__code__)

    with pytest.raises(
        DistillationTrainingError,
        match=(
            "runtime semantic descriptor drift: "
            "T3T4PolicyValueQNet.forward"
        ),
    ):
        training_module.train_distillation_candidate(
            complete_bundle.root,
            tmp_path / "candidate",
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
            config=_tiny_training_config(),
        )
    with pytest.raises(
        DistillationTrainingError,
        match=(
            "runtime semantic descriptor drift: "
            "T3T4PolicyValueQNet.forward"
        ),
    ):
        training_module.verify_training_artifact(
            tmp_path / "missing",
            teacher_bundle_dir=complete_bundle.root,
            expected_manifest_sha256="0" * 64,
        )


def test_trainer_public_boundary_rejects_functional_attribute_substitution(
    complete_bundle,
    tmp_path,
    monkeypatch,
):
    calls = []

    def forged_smooth_l1_loss(*_args, **_kwargs):
        calls.append("forged")
        raise AssertionError("forged loss must not execute")

    monkeypatch.setattr(
        training_module.F,
        "smooth_l1_loss",
        forged_smooth_l1_loss,
    )
    with pytest.raises(
        DistillationTrainingError,
        match="runtime semantic target drift: F.smooth_l1_loss",
    ):
        training_module.train_distillation_candidate(
            complete_bundle.root,
            tmp_path / "candidate",
            expected_teacher_manifest_sha256=complete_bundle.manifest[
                "manifest_sha256"
            ],
            config=_tiny_training_config(),
        )
    with pytest.raises(
        DistillationTrainingError,
        match="runtime semantic target drift: F.smooth_l1_loss",
    ):
        training_module.verify_training_artifact(
            tmp_path / "missing",
            teacher_bundle_dir=complete_bundle.root,
            expected_manifest_sha256="0" * 64,
        )
    assert calls == []
