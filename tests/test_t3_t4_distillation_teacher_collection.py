import copy
import hashlib
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

import ai.tutor.t3_t4_distillation_teacher_collection as collection_module
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_distillation_teacher import (
    MCCFR_SOLVER_METHOD,
    build_split_assignment,
    build_teacher_row,
    canonical_json,
)
from ai.tutor.t3_t4_distillation_teacher_collection import (
    PHASE_ACTOR_JOKER_CELLS,
    TeacherCollectionError,
    TeacherCollectionLockError,
    append_teacher_chunk,
    build_generation_plan,
    finalize_teacher_collection,
    initialize_teacher_collection,
    resume_teacher_collection,
    verify_teacher_collection,
)
from ai.tutor.t3_t4_infoset_encoder import semantic_action_ids


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
        "bb": {1: ("7d", "8h"), 2: ("9s", "Tc"), 3: ("Qd", "X1")},
        "btn": {1: ("9d", "Jh"), 2: ("Qs", "Kc"), 3: ("2d", "3h")},
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


def _base_key(phase: str) -> InfoSetKey:
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


def _key_with_jokers(phase: str, joker_count: int) -> InfoSetKey:
    key = _base_key(phase)
    substitutions = {"X1": "Ac", "X2": "2h"}

    def card(value: str) -> str:
        return substitutions.get(value, value)

    board_bb = tuple(tuple(card(value) for value in row) for row in key.board_bb)
    board_btn = tuple(tuple(card(value) for value in row) for row in key.board_btn)
    history = tuple(
        (
            turn,
            actor,
            tuple((card(value), row) for value, row in action),
        )
        for turn, actor, action in key.public_action_history
    )
    recall = PrivateRecall(
        dealt_by_turn=tuple(
            (turn, tuple(card(value) for value in values))
            for turn, values in key.own_recall.dealt_by_turn
        ),
        discards_by_turn=tuple(
            (turn, card(value))
            for turn, value in key.own_recall.discards_by_turn
        ),
    )
    draw = [card(value) for value in key.current_draw]
    for index, joker in enumerate(("X1", "X2")[:joker_count]):
        draw[index] = joker
    return replace(
        key,
        board_bb=board_bb,
        board_btn=board_btn,
        public_action_history=history,
        own_recall=recall,
        current_draw=tuple(draw),
    )


def _assignment_for_split(
    split: str,
    tag: str,
    *,
    root_family_sha256: str | None = None,
) -> dict:
    for index in range(100_000):
        assignment = build_split_assignment(
            full_deal_commitment_sha256=_hash(f"{tag}:full:{index}"),
            public_root_family_commitment_sha256=(
                root_family_sha256 or _hash(f"{tag}:family:{index}")
            ),
        )
        if assignment["split"] == split:
            return assignment
    raise AssertionError(f"could not find deterministic {split} assignment")


def _bindings(tag: str) -> dict[str, str]:
    return {
        field: _hash(f"{tag}:{field}")
        for field in (
            "public_root_commitment_sha256",
            "public_root_mixture_sha256",
            "range_content_sha256",
            "range_build_sha256",
            "behavior_model_sha256",
            "source_manifest_sha256",
            "solver_source_sha256",
            "solver_config_sha256",
            "solver_checkpoint_sha256",
        )
    }


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


def _row(
    phase: str,
    joker_count: int,
    split: str,
    tag: str,
    *,
    assignment: dict | None = None,
) -> dict:
    key = _key_with_jokers(phase, joker_count)
    strategy, moments = _labels(key)
    return build_teacher_row(
        key,
        split_assignment=assignment or _assignment_for_split(split, tag),
        lineage=_lineage(tag),
        bindings=_bindings(tag),
        solver_method=MCCFR_SOLVER_METHOD,
        solver_iterations_completed=10_000,
        solver_seed_index=0,
        payoff_seed_index=0,
        average_strategy_by_action_id=strategy,
        action_payoff_moments_by_action_id=moments,
        infoset_visit_count=1234,
        infoset_reach_probability=Fraction(1, 100),
    )


@lru_cache(maxsize=1)
def _coverage_rows() -> tuple[dict, ...]:
    phases = (
        "t3_first",
        "t3_second",
        "t4_first",
        "t4_second",
    )
    splits = ("fit", "dev", "test")
    rows: list[dict] = []
    index = 0
    for phase in phases:
        for joker_count in range(3):
            split = splits[index % len(splits)]
            rows.append(
                _row(
                    phase,
                    joker_count,
                    split,
                    f"coverage-{phase}-{joker_count}-{split}",
                )
            )
            index += 1
    return tuple(rows)


def _plan(*, expected_rows: int = 12, chunk_limit: int = 4) -> dict:
    return build_generation_plan(
        generation_source_sha256=_hash("collection-generation-source"),
        expected_total_row_count=expected_rows,
        chunk_row_limit=chunk_limit,
        bundle_shard_size=2,
        phase_actor_joker_minimum_counts={
            cell: 1 for cell in PHASE_ACTOR_JOKER_CELLS
        },
    )


def _run_complete(root: Path) -> tuple[dict, object]:
    plan = _plan()
    current = initialize_teacher_collection(root, plan)
    assert current.top_manifest["sequence"] == 0
    rows = _coverage_rows()
    for start in range(0, len(rows), 4):
        current = append_teacher_chunk(
            root,
            rows[start : start + 4],
            expected_generation_plan_sha256=plan["generation_plan_sha256"],
        )
    finalized = finalize_teacher_collection(
        root,
        expected_generation_plan_sha256=plan["generation_plan_sha256"],
    )
    return plan, finalized


def _artifact_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_deterministic_streaming_finalize_and_fresh_verification(tmp_path):
    first_plan, first = _run_complete(tmp_path / "first")
    second_plan, second = _run_complete(tmp_path / "second")
    assert first_plan == second_plan
    assert first.finalized is True
    assert second.finalized is True
    assert first.top_manifest["chunk_count"] == 3
    assert first.top_manifest["sequence"] == 4
    assert first.top_manifest["statistics"]["row_count"] == 12
    assert all(
        count == 1
        for count in first.top_manifest["statistics"][
            "phase_actor_joker_counts"
        ].values()
    )
    assert all(
        count > 0
        for count in first.top_manifest["statistics"]["split_row_counts"].values()
    )
    verified = verify_teacher_collection(
        first.root,
        expected_generation_plan_sha256=first_plan["generation_plan_sha256"],
        require_finalized=True,
    )
    assert verified.top_manifest == first.top_manifest
    assert not hasattr(verified, "rows")
    assert _artifact_snapshot(first.root) == _artifact_snapshot(second.root)


@pytest.mark.parametrize("crash_point", ("after_marker", "before_marker"))
def test_crash_safe_resume_adopts_exactly_one_next_orphan(
    tmp_path,
    monkeypatch,
    crash_point,
):
    plan = _plan()
    root = tmp_path / crash_point
    initialize_teacher_collection(root, plan)
    if crash_point == "after_marker":
        original = collection_module._publish_top_manifest

        def crash_top(path, manifest):
            if manifest["sequence"] == 1:
                raise RuntimeError("simulated crash after complete chunk")
            return original(path, manifest)

        target_name = "_publish_top_manifest"
        replacement = crash_top
    else:
        original = collection_module._publish_json_no_replace

        def crash_marker(path, value):
            if path.name.endswith(".complete.json"):
                raise RuntimeError("simulated crash before completion marker")
            return original(path, value)

        target_name = "_publish_json_no_replace"
        replacement = crash_marker
    with monkeypatch.context() as scoped:
        scoped.setattr(collection_module, target_name, replacement)
        with pytest.raises(RuntimeError, match="simulated crash"):
            append_teacher_chunk(
                root,
                _coverage_rows()[:4],
                expected_generation_plan_sha256=plan["generation_plan_sha256"],
            )
    resumed = resume_teacher_collection(
        root,
        expected_generation_plan_sha256=plan["generation_plan_sha256"],
    )
    assert resumed.top_manifest["chunk_count"] == 1
    assert resumed.top_manifest["statistics"]["row_count"] == 4
    assert resumed.top_manifest["sequence"] == 1


def test_multiple_gap_and_invalid_orphans_fail_closed(tmp_path):
    rows = _coverage_rows()

    multiple_plan = _plan()
    multiple_root = tmp_path / "multiple"
    initialize_teacher_collection(multiple_root, multiple_plan)
    with collection_module._collection_lock(multiple_root.resolve()):
        collection_module._publish_chunk_locked(
            root=multiple_root.resolve(),
            plan=multiple_plan,
            index=0,
            row_start=0,
            rows=rows[:2],
        )
        collection_module._publish_chunk_locked(
            root=multiple_root.resolve(),
            plan=multiple_plan,
            index=0,
            row_start=0,
            rows=rows[2:4],
        )
    with pytest.raises(TeacherCollectionError, match="multiple next chunk orphans"):
        resume_teacher_collection(
            multiple_root,
            expected_generation_plan_sha256=multiple_plan[
                "generation_plan_sha256"
            ],
        )

    gap_plan = _plan()
    gap_root = tmp_path / "gap"
    initialize_teacher_collection(gap_root, gap_plan)
    with collection_module._collection_lock(gap_root.resolve()):
        collection_module._publish_chunk_locked(
            root=gap_root.resolve(),
            plan=gap_plan,
            index=1,
            row_start=0,
            rows=rows[:2],
        )
    with pytest.raises(TeacherCollectionError, match="index gap or overlap"):
        resume_teacher_collection(
            gap_root,
            expected_generation_plan_sha256=gap_plan["generation_plan_sha256"],
        )

    invalid_plan = _plan()
    invalid_root = tmp_path / "invalid"
    initialize_teacher_collection(invalid_root, invalid_plan)
    with collection_module._collection_lock(invalid_root.resolve()):
        entry = collection_module._publish_chunk_locked(
            root=invalid_root.resolve(),
            plan=invalid_plan,
            index=0,
            row_start=0,
            rows=rows[:2],
        )
    marker = invalid_root / entry["completion_marker_relative_path"]
    payload = copy.deepcopy(entry)
    payload["row_start"] = 1
    marker.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(TeacherCollectionError, match="invalid orphan completion marker"):
        resume_teacher_collection(
            invalid_root,
            expected_generation_plan_sha256=invalid_plan[
                "generation_plan_sha256"
            ],
        )


def test_duplicate_and_cross_split_observation_overlap_fail_across_chunks(tmp_path):
    duplicate_plan = _plan()
    duplicate_root = tmp_path / "duplicate"
    initialize_teacher_collection(duplicate_root, duplicate_plan)
    append_teacher_chunk(
        duplicate_root,
        [_coverage_rows()[0]],
        expected_generation_plan_sha256=duplicate_plan["generation_plan_sha256"],
    )
    with pytest.raises(TeacherCollectionError, match="duplicate row identity"):
        append_teacher_chunk(
            duplicate_root,
            [_coverage_rows()[0]],
            expected_generation_plan_sha256=duplicate_plan[
                "generation_plan_sha256"
            ],
        )

    overlap_plan = _plan()
    overlap_root = tmp_path / "information-overlap"
    initialize_teacher_collection(overlap_root, overlap_plan)
    first = _row("t3_first", 0, "fit", "same-observation-fit")
    second = _row("t3_first", 0, "test", "same-observation-test")
    append_teacher_chunk(
        overlap_root,
        [first],
        expected_generation_plan_sha256=overlap_plan["generation_plan_sha256"],
    )
    with pytest.raises(
        TeacherCollectionError,
        match="cross-split information_digest overlap",
    ):
        append_teacher_chunk(
            overlap_root,
            [second],
            expected_generation_plan_sha256=overlap_plan[
                "generation_plan_sha256"
            ],
        )

    family_plan = _plan()
    family_root = tmp_path / "family-overlap"
    initialize_teacher_collection(family_root, family_plan)
    shared_family = _hash("shared-root-family")
    fit_assignment = _assignment_for_split(
        "fit", "family-fit", root_family_sha256=shared_family
    )
    test_assignment = _assignment_for_split(
        "test", "family-test", root_family_sha256=shared_family
    )
    append_teacher_chunk(
        family_root,
        [
            _row(
                "t3_first",
                0,
                "fit",
                "family-fit-row",
                assignment=fit_assignment,
            )
        ],
        expected_generation_plan_sha256=family_plan["generation_plan_sha256"],
    )
    with pytest.raises(TeacherCollectionError, match="multiple full deals"):
        append_teacher_chunk(
            family_root,
            [
                _row(
                    "t3_second",
                    0,
                    "test",
                    "family-test-row",
                    assignment=test_assignment,
                )
            ],
            expected_generation_plan_sha256=family_plan[
                "generation_plan_sha256"
            ],
        )


def test_finalization_requires_plan_total_nonempty_splits_and_coverage(tmp_path):
    partial_plan = _plan()
    partial_root = tmp_path / "partial"
    initialize_teacher_collection(partial_root, partial_plan)
    append_teacher_chunk(
        partial_root,
        _coverage_rows()[:4],
        expected_generation_plan_sha256=partial_plan["generation_plan_sha256"],
    )
    with pytest.raises(TeacherCollectionError, match="final row count"):
        finalize_teacher_collection(
            partial_root,
            expected_generation_plan_sha256=partial_plan[
                "generation_plan_sha256"
            ],
        )

    split_plan = _plan()
    split_root = tmp_path / "split"
    initialize_teacher_collection(split_root, split_plan)
    fit_rows = tuple(
        _row(
            phase,
            joker,
            "fit",
            f"all-fit-{phase}-{joker}",
        )
        for phase in ("t3_first", "t3_second", "t4_first", "t4_second")
        for joker in range(3)
    )
    for start in range(0, len(fit_rows), 4):
        append_teacher_chunk(
            split_root,
            fit_rows[start : start + 4],
            expected_generation_plan_sha256=split_plan[
                "generation_plan_sha256"
            ],
        )
    with pytest.raises(TeacherCollectionError, match="fit/dev/test"):
        finalize_teacher_collection(
            split_root,
            expected_generation_plan_sha256=split_plan[
                "generation_plan_sha256"
            ],
        )

    coverage_plan = _plan()
    coverage_root = tmp_path / "coverage"
    initialize_teacher_collection(coverage_root, coverage_plan)
    deficient = list(_coverage_rows()[:-1])
    deficient.append(_row("t3_first", 0, "fit", "coverage-extra"))
    for start in range(0, len(deficient), 4):
        append_teacher_chunk(
            coverage_root,
            deficient[start : start + 4],
            expected_generation_plan_sha256=coverage_plan[
                "generation_plan_sha256"
            ],
        )
    with pytest.raises(TeacherCollectionError, match="coverage deficits"):
        finalize_teacher_collection(
            coverage_root,
            expected_generation_plan_sha256=coverage_plan[
                "generation_plan_sha256"
            ],
        )


def test_plan_is_immutable_lock_is_exclusive_and_tamper_fails(tmp_path):
    plan = _plan()
    root = tmp_path / "collection"
    initialized = initialize_teacher_collection(root, plan)
    changed = build_generation_plan(
        generation_source_sha256=_hash("different-generation-source"),
        expected_total_row_count=12,
        chunk_row_limit=4,
        bundle_shard_size=2,
        phase_actor_joker_minimum_counts={
            cell: 1 for cell in PHASE_ACTOR_JOKER_CELLS
        },
    )
    with pytest.raises(TeacherCollectionError, match="immutable"):
        initialize_teacher_collection(root, changed)
    assert len(list(root.glob("generation-plan-*.json"))) == 1

    with collection_module._collection_lock(root.resolve()):
        with pytest.raises(TeacherCollectionLockError, match="OS lock is busy"):
            resume_teacher_collection(
                root,
                expected_generation_plan_sha256=plan["generation_plan_sha256"],
            )
    advanced = append_teacher_chunk(
        root,
        _coverage_rows()[:4],
        expected_generation_plan_sha256=plan["generation_plan_sha256"],
    )
    assert advanced.top_manifest["sequence"] > initialized.top_manifest["sequence"]
    assert len(list((root / "manifests").glob("top-*.json"))) == 2

    marker = root / advanced.top_manifest["chunks"][0][
        "completion_marker_relative_path"
    ]
    marker_payload = json_load(marker)
    marker_payload["serving_changed"] = True
    marker.write_text(canonical_json(marker_payload) + "\n", encoding="utf-8")
    with pytest.raises(TeacherCollectionError):
        verify_teacher_collection(
            root,
            expected_generation_plan_sha256=plan["generation_plan_sha256"],
        )


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
