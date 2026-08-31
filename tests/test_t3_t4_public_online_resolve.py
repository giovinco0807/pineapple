import hashlib
import inspect
import json
import math
from dataclasses import replace
from fractions import Fraction

import pytest

import ai.tutor.t3_hu_full_card_range as range_module
import ai.tutor.t3_hu_multi_root_mccfr as multi_root_module
import ai.tutor.t3_t4_public_online_resolve as online_module
import ai.tutor.t3_t4_solve_root_teacher_adapter as solve_root_module
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_full_card_range import (
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.runtime_semantic_anchor import build_module_function_anchor
from ai.tutor.t3_t4_public_online_resolve import (
    ONLINE_RESOLVE_REMAINING_LIMITATIONS,
    PublicOnlineResolveError,
    resolve_compiled_public_root_mixture,
    select_scoped_policy_for_actual_infoset,
    verify_online_public_resolve,
    verify_scoped_policy_selection,
)
from ai.tutor.t3_t4_public_root_mixture import (
    ExplicitHypotheticalPrivateType,
    PublicRootBindings,
    PublicRootContext,
    compile_explicit_public_root_mixture,
)
from ai.tutor.t3_t4_solve_root_teacher_adapter import (
    EVIDENCE_MANIFEST_FILENAME,
    SolveRootTeacherAdapterError,
    build_prelabel_solve_root_teacher_plan,
    build_solve_root_teacher_evidence,
    resolve_prelabel_solve_root_plan,
    write_prelabel_solve_root_teacher_plan,
)


BB_BOARD = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc",),
)
BTN_BOARD = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As",),
)
PUBLIC_HISTORY = (
    (
        0,
        "bb",
        (
            ("4h", "top"),
            ("2c", "top"),
            ("3d", "top"),
            ("7c", "middle"),
            ("Qc", "bottom"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("8c", "top"),
            ("5c", "top"),
            ("6d", "top"),
            ("9c", "middle"),
            ("As", "bottom"),
        ),
    ),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)
BB_T3_PLACEMENTS = (("Qd", "bottom"), ("Kh", "bottom"))
BTN_T3_PLACEMENTS = (("2h", "bottom"), ("3h", "bottom"))
BB_BOARD_11 = (BB_BOARD[0], BB_BOARD[1], ("Qc", "Qd", "Kh"))
BTN_BOARD_11 = (BTN_BOARD[0], BTN_BOARD[1], ("As", "2h", "3h"))
PUBLIC_HISTORY_AFTER_BB_T3 = PUBLIC_HISTORY + ((3, "bb", BB_T3_PLACEMENTS),)
PUBLIC_HISTORY_AFTER_T3 = PUBLIC_HISTORY_AFTER_BB_T3 + (
    (3, "btn", BTN_T3_PLACEMENTS),
)


def _recall(actor: str, first_discard: str, second_discard: str) -> PrivateRecall:
    placements = {
        "bb": (("7d", "8h"), ("9s", "Tc")),
        "btn": (("9d", "Jh"), ("Qs", "Kc")),
    }[actor]
    return PrivateRecall(
        dealt_by_turn=(
            (1, (*placements[0], first_discard)),
            (2, (*placements[1], second_discard)),
        ),
        discards_by_turn=((1, first_discard), (2, second_discard)),
    )


def _append_recall(
    recall: PrivateRecall,
    *,
    turn: int,
    placements: tuple[tuple[str, str], ...],
    discard: str,
) -> PrivateRecall:
    placed = tuple(card for card, _row in placements)
    return PrivateRecall(
        dealt_by_turn=recall.dealt_by_turn + ((turn, (*placed, discard)),),
        discards_by_turn=recall.discards_by_turn + ((turn, discard),),
    )


def _public_payload(phase: str) -> dict:
    if phase == "t3_first":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "bb",
            "turn": 3,
            "phase": phase,
            "board_bb": BB_BOARD,
            "board_btn": BTN_BOARD,
            "public_action_history": PUBLIC_HISTORY,
            "fantasy_state": None,
        }
    if phase == "t3_second":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "btn",
            "turn": 3,
            "phase": phase,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD,
            "public_action_history": PUBLIC_HISTORY_AFTER_BB_T3,
            "fantasy_state": None,
        }
    if phase == "t4_first":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "bb",
            "turn": 4,
            "phase": phase,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD_11,
            "public_action_history": PUBLIC_HISTORY_AFTER_T3,
            "fantasy_state": None,
        }
    raise ValueError(f"unsupported phase {phase}")


def _own_recall(phase: str) -> PrivateRecall:
    if phase == "t3_first":
        return _recall("bb", "6c", "6s")
    if phase == "t3_second":
        return _recall("btn", "4s", "7h")
    if phase == "t4_first":
        return _append_recall(
            _recall("bb", "6c", "6s"),
            turn=3,
            placements=BB_T3_PLACEMENTS,
            discard="5d",
        )
    raise ValueError(f"unsupported phase {phase}")


def _draw(phase: str, joker: str) -> tuple[str, str, str]:
    return ("Ad", "Kd", joker) if phase == "t3_first" else ("Ad", "Jd", joker)


def _one_assignment(pool, count, *, max_particles, seed, observation_digest):
    del max_particles, seed, observation_digest
    assignment = tuple(card for card in pool if card not in ("X1", "X2"))[:count]
    return (assignment,), math.perm(len(pool), count), False


def _build_case(phase: str):
    model = UniformLegalBehaviorModel()
    bindings = PublicRootBindings.capture(
        behavior_model_id=model.model_id,
        behavior_model_sha256=model.model_sha256,
    )
    context = PublicRootContext.from_public_mapping(
        _public_payload(phase),
        bindings=bindings,
    )
    recall = _own_recall(phase)
    support = []
    for index, (joker, mass) in enumerate(
        (("X1", Fraction(1, 3)), ("X2", Fraction(2, 3)))
    ):
        current_draw = _draw(phase, joker)
        observation = InfoSetKey(
            contract_version=context.contract_version,
            actor=context.actor,
            turn=context.turn,
            phase=context.phase,
            board_bb=context.board_bb,
            board_btn=context.board_btn,
            public_action_history=context.public_action_history,
            own_recall=recall,
            current_draw=current_draw,
            fantasy_state=None,
        )
        root_range = build_history_weighted_full_card_range(
            observation,
            model,
            epsilon=0,
            max_particles=1,
            seed=700 + index,
        )
        support.append(
            ExplicitHypotheticalPrivateType(
                type_id=f"online-type-{index}",
                hypothetical_own_recall=recall,
                hypothetical_current_draw=current_draw,
                prior_mass=mass,
                root_range=root_range,
            )
        )
    mixture = compile_explicit_public_root_mixture(context, tuple(support))
    return context, tuple(support), mixture


@pytest.fixture(scope="module")
def online_cases():
    patcher = pytest.MonkeyPatch()
    patcher.setattr(range_module, "_select_assignments", _one_assignment)
    try:
        yield {phase: _build_case(phase) for phase in ("t3_first", "t3_second", "t4_first")}
    finally:
        patcher.undo()


@pytest.mark.parametrize(
    ("phase", "actor"),
    (("t3_first", "bb"), ("t3_second", "btn"), ("t4_first", "bb")),
)
def test_real_full_card_online_resolve_vertical_smoke(online_cases, phase, actor):
    context, _support, mixture = online_cases[phase]
    resolved = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=2,
        max_infosets=100_000,
        linear_averaging=False,
    )
    verification = verify_online_public_resolve(context, mixture, resolved)
    manifest = resolved.manifest

    assert verification["verified"] is True
    assert resolved.solver_result.iterations == 1
    assert resolved.solver_result.root_count == 2
    assert all(key.actor in ("bb", "btn") for key in resolved.solver_result.average_strategy)
    assert any(key.actor == actor for key in resolved.solver_result.average_strategy)
    assert manifest["algorithm_validation_only"] is True
    assert manifest["finite_explicit_support_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["runtime_integrated"] is False
    assert manifest["global_unseen_state_policy_claim"] is False
    assert manifest["actual_infoset_consumed_by_solver"] is False
    assert manifest["source_binding"]["binding_sha256"]
    assert manifest["solver_result_binding"]["binding_sha256"]
    assert manifest["checkpoint_binding"]["checkpoint_enabled"] is False

    actual = next(
        entry.adapter.observation
        for entry in mixture.entries
        if entry.adapter.observation in resolved.solver_result.average_strategy
    )
    before = tuple(dict(entry.adapter.sampling_audit()) for entry in mixture.entries)
    selection = select_scoped_policy_for_actual_infoset(
        context,
        mixture,
        resolved,
        actual,
    )
    after = tuple(dict(entry.adapter.sampling_audit()) for entry in mixture.entries)
    assert before == after
    assert verify_scoped_policy_selection(
        context,
        mixture,
        resolved,
        selection,
    )["verified"] is True
    assert math.isclose(sum(selection.action_probabilities.values()), 1.0)
    assert selection.selection_manifest["actual_infoset_used_for_solver_execution"] is False


def test_online_solve_api_has_no_actual_infoset_input_and_replay_is_deterministic(
    monkeypatch,
):
    patch = pytest.MonkeyPatch()
    patch.setattr(range_module, "_select_assignments", _one_assignment)
    try:
        first_context, _first_support, first_mixture = _build_case("t3_first")
        replay_context, _replay_support, replay_mixture = _build_case("t3_first")
    finally:
        patch.undo()
    parameters = inspect.signature(resolve_compiled_public_root_mixture).parameters
    assert "actual_infoset" not in parameters
    assert "actual_private_cards" not in parameters
    assert "actual_infoset" in inspect.signature(
        select_scoped_policy_for_actual_infoset
    ).parameters

    first = resolve_compiled_public_root_mixture(
        first_context,
        first_mixture,
        iterations=2,
        seed=20260713,
        max_infosets=100_000,
        linear_averaging=True,
    )
    replay = resolve_compiled_public_root_mixture(
        replay_context,
        replay_mixture,
        iterations=2,
        seed=20260713,
        max_infosets=100_000,
        linear_averaging=True,
    )
    assert first.solver_result.average_strategy_json == replay.solver_result.average_strategy_json
    assert first.solver_result.current_strategy_json == replay.solver_result.current_strategy_json
    assert first.manifest_json == replay.manifest_json
    assert first.manifest_sha256 == replay.manifest_sha256


def test_selector_rejects_missing_nonmember_duplicate_and_public_context_mismatch(
    online_cases,
):
    context, _support, mixture = online_cases["t3_first"]
    resolved = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=0,
        max_infosets=100_000,
        linear_averaging=False,
    )
    missing = next(
        entry.adapter.observation
        for entry in mixture.entries
        if entry.adapter.observation not in resolved.solver_result.average_strategy
    )
    with pytest.raises(PublicOnlineResolveError, match="missing from the encountered"):
        select_scoped_policy_for_actual_infoset(
            context, mixture, resolved, missing
        )

    nonmember = InfoSetKey(
        contract_version=context.contract_version,
        actor=context.actor,
        turn=context.turn,
        phase=context.phase,
        board_bb=context.board_bb,
        board_btn=context.board_btn,
        public_action_history=context.public_action_history,
        own_recall=_own_recall("t3_first"),
        current_draw=("Ad", "Kd", "Jd"),
        fantasy_state=None,
    )
    with pytest.raises(PublicOnlineResolveError, match="not a member"):
        select_scoped_policy_for_actual_infoset(
            context, mixture, resolved, nonmember
        )

    other_actual = online_cases["t3_second"][2].entries[0].adapter.observation
    with pytest.raises(PublicOnlineResolveError, match="public context does not match"):
        select_scoped_policy_for_actual_infoset(
            context, mixture, resolved, other_actual
        )

    duplicate = replace(mixture, entries=(mixture.entries[0], mixture.entries[0]))
    with pytest.raises(ValueError, match="not unique|duplicate"):
        select_scoped_policy_for_actual_infoset(
            context, duplicate, resolved, mixture.entries[0].adapter.observation
        )


def test_checkpoint_create_and_resume_hashes_are_bound(online_cases, tmp_path):
    context, support, mixture = online_cases["t4_first"]
    checkpoint = tmp_path / "online-resolve-checkpoint.json"
    first = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=17,
        max_infosets=10_000,
        checkpoint_path=checkpoint,
    )
    first_checkpoint = first.manifest["checkpoint_binding"]
    assert first_checkpoint["checkpoint_enabled"] is True
    assert first_checkpoint["checkpoint_output_persisted"] is True
    assert first_checkpoint["checkpoint_output_file_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    expected_checkpoint_sha256 = first.solver_result.metadata["checkpoint_sha256"]

    resumed_mixture = compile_explicit_public_root_mixture(context, support)
    resumed = resolve_compiled_public_root_mixture(
        context,
        resumed_mixture,
        iterations=1,
        seed=17,
        max_infosets=10_000,
        resume_from=checkpoint,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    checkpoint_binding = resumed.manifest["checkpoint_binding"]
    assert resumed.solver_result.iterations == 2
    assert checkpoint_binding["resume_used"] is True
    assert checkpoint_binding["completed_iterations_before"] == 1
    assert checkpoint_binding["resume_checkpoint_payload_sha256"] == (
        expected_checkpoint_sha256
    )
    assert checkpoint_binding["final_checkpoint_payload_sha256"] == (
        resumed.solver_result.metadata["checkpoint_sha256"]
    )
    assert verify_online_public_resolve(
        context,
        resumed_mixture,
        resumed,
        expected_manifest_sha256=resumed.manifest_sha256,
    )["verified"] is True

    fresh_mixture = compile_explicit_public_root_mixture(context, support)
    with pytest.raises(PublicOnlineResolveError, match="requires externally trusted"):
        resolve_compiled_public_root_mixture(
            context,
            fresh_mixture,
            iterations=1,
            seed=17,
            max_infosets=10_000,
            resume_from=checkpoint,
        )
    with pytest.raises(PublicOnlineResolveError, match="valid only with resume_from"):
        resolve_compiled_public_root_mixture(
            context,
            fresh_mixture,
            iterations=1,
            seed=17,
            max_infosets=10_000,
            expected_checkpoint_sha256="0" * 64,
        )


def test_sampled_solve_root_plan_persists_checkpoint_and_rejects_sparse_policy(
    online_cases,
    tmp_path,
):
    context, _support, mixture = online_cases["t4_first"]
    selected = max(mixture.entries, key=lambda entry: entry.prior_mass)
    plan = build_prelabel_solve_root_teacher_plan(
        context,
        mixture,
        root_id_sha256=selected.root_id_sha256,
        full_deal_commitment_sha256=hashlib.sha256(b"sampled-deal").hexdigest(),
        public_root_family_commitment_sha256=hashlib.sha256(
            b"sampled-family"
        ).hexdigest(),
        solver_seed_index=3,
        payoff_seed_index=4,
        solver_iterations=20,
        max_infosets=100_000,
        linear_averaging=True,
        samples_per_action=2,
    )
    manifest = plan.manifest
    assert manifest["assignment_before_labels"] is True
    assert manifest["label_fields_present"] is False
    assert manifest["label_branch"] == "sampled_mccfr"
    assert manifest["sampled_plan"]["derived_solver_seed"] % 2 == 0
    assert manifest["sampled_plan"]["derived_payoff_seed"] % 2 == 1

    plan_path = write_prelabel_solve_root_teacher_plan(
        tmp_path / "sampled-plan.json",
        context,
        mixture,
        plan,
    )
    checkpoint_path = tmp_path / "sampled-checkpoint.json"
    resolved = resolve_prelabel_solve_root_plan(
        context,
        mixture,
        plan,
        checkpoint_path=checkpoint_path,
    )
    assert checkpoint_path.is_file()
    assert resolved.solver_result.metadata["checkpoint_sha256"]
    assert resolved.manifest["checkpoint_binding"][
        "checkpoint_output_persisted"
    ] is True

    evidence_dir = tmp_path / "sparse-evidence"
    with pytest.raises(
        SolveRootTeacherAdapterError,
        match="missing an exact continuation InfoSetKey",
    ):
        build_solve_root_teacher_evidence(
            context,
            mixture,
            prelabel_plan_path=plan_path,
            output_dir=evidence_dir,
            resolved=resolved,
            checkpoint_path=checkpoint_path,
        )
    assert not (evidence_dir / EVIDENCE_MANIFEST_FILENAME).exists()


def test_solve_root_runtime_guard_rejects_paired_function_and_anchor_rebuild(
    monkeypatch,
):
    original = solve_root_module._select_root

    def substituted(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(solve_root_module, "_select_root", substituted)
    forged = build_module_function_anchor(vars(solve_root_module))
    monkeypatch.setattr(solve_root_module, "_MODULE_RUNTIME_ANCHOR", forged)
    monkeypatch.setattr(
        solve_root_module,
        "_MODULE_RUNTIME_ANCHOR_MIRROR",
        forged,
    )
    with pytest.raises(SolveRootTeacherAdapterError, match="registry drift"):
        solve_root_module._require_runtime_contract()


def test_result_and_selection_tamper_wrong_strategy_and_member_fail_closed(online_cases):
    context, _support, mixture = online_cases["t3_second"]
    resolved = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=2,
        max_infosets=100_000,
    )
    actual = next(
        entry.adapter.observation
        for entry in mixture.entries
        if entry.adapter.observation in resolved.solver_result.average_strategy
    )
    selection = select_scoped_policy_for_actual_infoset(
        context, mixture, resolved, actual
    )

    manifest = json.loads(resolved.manifest_json)
    manifest["promotion_eligible"] = True
    tampered_json = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    tampered = replace(
        resolved,
        manifest_json=tampered_json,
        manifest_sha256=hashlib.sha256(tampered_json.encode()).hexdigest(),
    )
    with pytest.raises(
        PublicOnlineResolveError,
        match="manifest requires promotion_eligible=false|manifest content mismatch",
    ):
        verify_online_public_resolve(context, mixture, tampered)

    checkpoint_extra = json.loads(resolved.manifest_json)
    checkpoint = checkpoint_extra["checkpoint_binding"]
    checkpoint.pop("binding_sha256")
    checkpoint["uncommitted_extra"] = "forbidden"
    checkpoint["binding_sha256"] = hashlib.sha256(
        json.dumps(
            checkpoint,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    checkpoint_extra_json = json.dumps(
        checkpoint_extra,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    checkpoint_extra_result = replace(
        resolved,
        manifest_json=checkpoint_extra_json,
        manifest_sha256=hashlib.sha256(checkpoint_extra_json.encode()).hexdigest(),
    )
    with pytest.raises(PublicOnlineResolveError, match="checkpoint binding schema"):
        verify_online_public_resolve(
            context,
            mixture,
            checkpoint_extra_result,
        )

    solver_tampered = replace(
        resolved,
        solver_result=replace(
            resolved.solver_result,
            average_strategy_sha256="0" * 64,
        ),
    )
    with pytest.raises(PublicOnlineResolveError, match="average strategy"):
        verify_online_public_resolve(context, mixture, solver_tampered)

    policy = dict(selection.action_probabilities)
    first_action = next(iter(policy))
    policy[first_action] += 0.01
    policy_tampered = replace(selection, action_probabilities=policy)
    with pytest.raises(PublicOnlineResolveError, match="policy row mismatch"):
        verify_scoped_policy_selection(
            context, mixture, resolved, policy_tampered
        )
    wrong_strategy = replace(selection, strategy_kind="current")
    with pytest.raises(PublicOnlineResolveError, match="policy row mismatch|manifest"):
        verify_scoped_policy_selection(
            context, mixture, resolved, wrong_strategy
        )
    wrong_member = replace(
        selection,
        actual_infoset=online_cases["t3_first"][2].entries[0].adapter.observation,
    )
    with pytest.raises(PublicOnlineResolveError, match="public context does not match"):
        verify_scoped_policy_selection(context, mixture, resolved, wrong_member)
    selection_manifest = json.loads(selection.selection_manifest_json)
    selection_manifest["runtime_integrated"] = True
    selection_json = json.dumps(
        selection_manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_tampered = replace(
        selection,
        selection_manifest_json=selection_json,
        selection_manifest_sha256=hashlib.sha256(selection_json.encode()).hexdigest(),
    )
    with pytest.raises(
        PublicOnlineResolveError,
        match="selection manifest requires runtime_integrated=False|manifest content mismatch",
    ):
        verify_scoped_policy_selection(
            context, mixture, resolved, manifest_tampered
        )
    assert verify_scoped_policy_selection(
        context,
        mixture,
        resolved,
        selection,
        expected_manifest_sha256=selection.selection_manifest_sha256,
    )["verified"] is True


def test_solver_callable_replay_is_rejected_and_t4_btn_bypass_is_documented(
    online_cases, monkeypatch
):
    context, _support, mixture = online_cases["t4_first"]
    solver = multi_root_module.solve_multi_root_external_sampling_mccfr
    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "solve_multi_root_external_sampling_mccfr",
            lambda roots, **kwargs: solver(roots, **kwargs),
        )
        with pytest.raises(ValueError, match="overridden|drifted"):
            resolve_compiled_public_root_mixture(
                context,
                mixture,
                iterations=1,
                seed=2,
                max_infosets=10_000,
            )
    assert "t4_second_btn_exact_bypass_is_not_integrated_by_this_wrapper" in (
        ONLINE_RESOLVE_REMAINING_LIMITATIONS
    )


def test_wrapper_helpers_and_limitations_are_runtime_attested(online_cases, monkeypatch):
    context, _support, mixture = online_cases["t3_first"]
    resolved = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=3,
        max_infosets=100_000,
    )
    actual = next(
        entry.adapter.observation
        for entry in mixture.entries
        if entry.adapter.observation in resolved.solver_result.average_strategy
    )

    targets = (
        "_manifest",
        "_selection_manifest",
        "_solver_result_binding",
        "_scoped_policy",
    )
    for target in targets:
        with monkeypatch.context() as patch:
            patch.setattr(online_module, target, lambda *args, **kwargs: {})
            with pytest.raises(PublicOnlineResolveError, match="runtime callables"):
                resolve_compiled_public_root_mixture(
                    context,
                    mixture,
                    iterations=1,
                    seed=3,
                    max_infosets=100_000,
                )
            with pytest.raises(PublicOnlineResolveError, match="runtime callables"):
                verify_online_public_resolve(context, mixture, resolved)
            with pytest.raises(PublicOnlineResolveError, match="runtime callables"):
                select_scoped_policy_for_actual_infoset(
                    context,
                    mixture,
                    resolved,
                    actual,
                )

    transitive_targets = (
        "_profile_table_sha256",
        "_checkpoint_envelope",
        "_self_hashed",
    )
    for target in transitive_targets:
        with monkeypatch.context() as patch:
            patch.setattr(online_module, target, lambda *args, **kwargs: {})
            with pytest.raises(PublicOnlineResolveError, match="runtime semantic binding"):
                resolve_compiled_public_root_mixture(
                    context,
                    mixture,
                    iterations=1,
                    seed=3,
                    max_infosets=100_000,
                )
            with pytest.raises(PublicOnlineResolveError, match="runtime semantic binding"):
                verify_online_public_resolve(context, mixture, resolved)
            with pytest.raises(PublicOnlineResolveError, match="runtime semantic binding"):
                select_scoped_policy_for_actual_infoset(
                    context,
                    mixture,
                    resolved,
                    actual,
                )

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "_runtime_semantic_graph",
            lambda *args, **kwargs: {},
        )
        with pytest.raises(PublicOnlineResolveError, match="runtime_semantic_graph"):
            verify_online_public_resolve(context, mixture, resolved)

    with monkeypatch.context() as patch:
        patch.setattr(
            online_module,
            "ONLINE_RESOLVE_REMAINING_LIMITATIONS",
            (*ONLINE_RESOLVE_REMAINING_LIMITATIONS, "tampered-limitation"),
        )
        with pytest.raises(PublicOnlineResolveError, match="limitations"):
            resolve_compiled_public_root_mixture(
                context,
                mixture,
                iterations=1,
                seed=3,
                max_infosets=100_000,
            )
        with pytest.raises(PublicOnlineResolveError, match="limitations"):
            verify_online_public_resolve(context, mixture, resolved)
        with pytest.raises(PublicOnlineResolveError, match="limitations"):
            select_scoped_policy_for_actual_infoset(
                context,
                mixture,
                resolved,
                actual,
            )


@pytest.mark.parametrize(
    "target",
    (
        "_profile_table_sha256",
        "_checkpoint_envelope",
        "_self_hashed",
    ),
)
def test_transitive_wrapper_helper_monkeypatch_is_rejected(monkeypatch, target):
    with monkeypatch.context() as patch:
        patch.setattr(online_module, target, lambda *args, **kwargs: {})
        with pytest.raises(PublicOnlineResolveError, match="runtime semantic binding"):
            online_module._require_wrapper_runtime_contract()


def test_runtime_semantic_graph_builder_identity_is_rejected(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "_runtime_semantic_graph",
            lambda *args, **kwargs: {},
        )
        with pytest.raises(PublicOnlineResolveError, match="runtime_semantic_graph"):
            online_module._require_wrapper_runtime_contract()


def test_solver_result_cannot_be_spliced_into_same_count_different_prior_mixture(
    online_cases,
):
    context, support, mixture = online_cases["t3_first"]
    resolved = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=5,
        max_infosets=100_000,
    )
    alternate_support = tuple(
        replace(item, prior_mass=Fraction(1, 2)) for item in support
    )
    alternate_mixture = compile_explicit_public_root_mixture(
        context,
        alternate_support,
    )
    assert len(alternate_mixture.entries) == len(mixture.entries)
    assert {
        entry.adapter.observation for entry in alternate_mixture.entries
    } == {entry.adapter.observation for entry in mixture.entries}

    rebound_manifest = json.loads(resolved.manifest_json)
    rebound_manifest["compiled_support_sha256"] = alternate_mixture.support_sha256
    rebound_manifest["compiled_mixture_audit_manifest_sha256"] = (
        alternate_mixture.audit_manifest_sha256
    )
    rebound_json = json.dumps(
        rebound_manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rebound = replace(
        resolved,
        manifest_json=rebound_json,
        manifest_sha256=hashlib.sha256(rebound_json.encode()).hexdigest(),
    )
    with pytest.raises(
        PublicOnlineResolveError,
        match="root-prior manifest does not match compiled support",
    ):
        verify_online_public_resolve(context, alternate_mixture, rebound)


def test_fully_rehashed_illegal_solver_action_set_is_rejected(online_cases):
    context, _support, mixture = online_cases["t4_first"]
    resolved = resolve_compiled_public_root_mixture(
        context,
        mixture,
        iterations=1,
        seed=7,
        max_infosets=100_000,
    )
    key = next(iter(resolved.solver_result.average_strategy))
    average = {
        infoset: dict(row)
        for infoset, row in resolved.solver_result.average_strategy.items()
    }
    current = {
        infoset: dict(row)
        for infoset, row in resolved.solver_result.current_strategy.items()
    }
    regrets = {
        infoset: dict(row)
        for infoset, row in resolved.solver_result.cumulative_regret_plus.items()
    }
    average[key] = {"illegal-action": 1.0}
    current[key] = {"illegal-action": 1.0}
    regrets[key] = {"illegal-action": 0.0}
    average_json = multi_root_module.serialize_strategy_profile(average)
    current_json = multi_root_module.serialize_strategy_profile(current)
    average_sha256 = hashlib.sha256(average_json.encode()).hexdigest()
    current_sha256 = hashlib.sha256(current_json.encode()).hexdigest()
    metadata = dict(resolved.solver_result.metadata)
    metadata["average_strategy_sha256"] = average_sha256
    metadata["current_strategy_sha256"] = current_sha256
    tampered_solver = replace(
        resolved.solver_result,
        average_strategy=average,
        current_strategy=current,
        cumulative_regret_plus=regrets,
        average_strategy_json=average_json,
        current_strategy_json=current_json,
        average_strategy_sha256=average_sha256,
        current_strategy_sha256=current_sha256,
        metadata=metadata,
    )
    tampered = replace(resolved, solver_result=tampered_solver)
    with pytest.raises(
        PublicOnlineResolveError,
        match="action set does not exactly match canonical legal actions",
    ):
        verify_online_public_resolve(context, mixture, tampered)
