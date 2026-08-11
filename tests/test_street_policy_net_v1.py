from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ofc_regular.action_key import ActionKey, action_key, canonicalize_actions
from ofc_regular.action_space import generate_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.street_policy_net_v1 import (
    FEATURE_SCHEMA_HASH,
    LOSS_SCHEMA_HASH,
    LOSS_WEIGHTS,
    MAX_LEGAL_ACTIONS,
    StreetPolicyNetV1Config,
    authorize_weight_update,
    build_authorized_optimizer,
    build_street_policy_net_v1,
    encode_street_policy_batch,
    feature_schema_payload,
    load_street_policy_checkpoint,
    model_state_sha256,
    parameter_names_for_update,
    save_street_policy_checkpoint,
    street_policy_training_loss,
)


def _t0_first(*, reverse_dealt: bool = False) -> ActorObservation:
    dealt = ("Ah", "Kd", "Qc", "Js", "Th")
    if reverse_dealt:
        dealt = tuple(reversed(dealt))
    return ActorObservation(
        hero_board=Board(),
        opponent_public_board=Board(),
        dealt_cards=dealt,
        hero_private_discards=(),
        seat="first",
        street="T0",
        to_act_order="first",
    )


def _t0_second(*, reverse_public_rows: bool = False) -> ActorObservation:
    top = ("2h", "3d")
    middle = ("4c", "5s")
    bottom = ("6h",)
    if reverse_public_rows:
        top = tuple(reversed(top))
        middle = tuple(reversed(middle))
    return ActorObservation(
        hero_board=Board(),
        opponent_public_board=Board.from_rows(top, middle, bottom),
        dealt_cards=("7d", "8c", "9s", "Td", "Jc"),
        hero_private_discards=(),
        seat="second",
        street="T0",
        to_act_order="second",
    )


def _keys(observation: ActorObservation) -> list[ActionKey]:
    actions = canonicalize_actions(
        generate_actions(observation.hero_board, observation.dealt_cards)
    )
    return [action_key(action) for action in actions]


def _encoded_pair():
    first = _t0_first()
    second = _t0_second()
    first_keys = _keys(first)
    second_keys = _keys(second)
    assert len(first_keys) == len(second_keys) == MAX_LEGAL_ACTIONS
    return encode_street_policy_batch(
        [first, second],
        [first_keys, second_keys],
        [first_keys[17], second_keys[29]],
    )


def test_feature_and_loss_contract_hashes_are_stable() -> None:
    assert FEATURE_SCHEMA_HASH == "d71fd9a9b2e6be4ea535fd9763d154dd30b26a62ed160a3b8b27a6b323a8bdf3"
    assert LOSS_SCHEMA_HASH == "db1bb6291ee669e5268a2f13c5c8ae76118bdf3538a60d7157cf4084f3013277"
    payload = feature_schema_payload()
    assert payload["belief"]["source"] == "actor_observation_only"
    assert "opponent_private_discards" in payload["belief"]["forbidden"]
    assert "realized_deck_tail" in payload["belief"]["forbidden"]
    assert LOSS_WEIGHTS == {
        "action_q_huber": 1.0,
        "baseline_delta_huber": 1.0,
        "teacher_policy_kl": 0.25,
        "state_value_huber": 0.25,
        "ranking": 0.5,
        "uncertainty_quantile": 0.2,
        "safe_bce": 0.5,
    }


def test_encoder_has_232_mask_and_canonical_action_mapping() -> None:
    observation = _t0_first()
    keys = _keys(observation)
    encoded = encode_street_policy_batch(
        [observation],
        [list(reversed(keys))],
        [keys[91].to_token()],
    )
    assert encoded.legal_action_mask.shape == (1, 232)
    assert encoded.legal_action_mask.all()
    assert encoded.baseline_indices.tolist() == [91]
    assert encoded.action_key_tokens[0][0] == keys[0].to_token()
    assert encoded.action_key_tokens[0][-1] == keys[-1].to_token()


def test_state_and_action_encoding_are_order_invariant() -> None:
    first = _t0_first()
    permuted = _t0_first(reverse_dealt=True)
    keys = _keys(first)
    encoded_a = encode_street_policy_batch([first], [keys], [keys[10]])
    encoded_b = encode_street_policy_batch(
        [permuted], [list(reversed(keys))], [keys[10]]
    )
    for name in (
        "state_card_ids",
        "state_zone_ids",
        "state_card_mask",
        "action_card_ids",
        "action_zone_ids",
        "action_card_mask",
        "legal_action_mask",
        "scalar_context",
    ):
        np.testing.assert_array_equal(getattr(encoded_a, name), getattr(encoded_b, name))
    assert encoded_a.action_key_tokens == encoded_b.action_key_tokens


@pytest.mark.parametrize(
    "field,value",
    [
        ("opponent_private_discards", ["2h"]),
        ("deck_tail", ["2h"]),
        ("world_state", {}),
        ("replay_truth", {}),
    ],
)
def test_encoder_rejects_hidden_truth_and_tail_fields(field: str, value: object) -> None:
    observation = _t0_first()
    payload = observation.to_dict()
    payload[field] = value
    keys = _keys(observation)
    with pytest.raises(ValueError, match="unknown fields"):
        encode_street_policy_batch([payload], [keys], [keys[0]])


def test_encoder_rejects_positional_or_malformed_action_identity() -> None:
    observation = _t0_first()
    keys = _keys(observation)
    with pytest.raises(TypeError, match="semantic ActionKey"):
        encode_street_policy_batch([observation], [[0]], [keys[0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="baseline ActionKey"):
        impossible = ActionKey(discard_mask=1)
        encode_street_policy_batch([observation], [keys], [impossible])
    with pytest.raises(ValueError, match="every dealt card"):
        encode_street_policy_batch(
            [observation],
            [[ActionKey(top_mask=keys[0].top_mask)]],
            [ActionKey(top_mask=keys[0].top_mask)],
        )


def test_encoder_requires_complete_legal_set_by_default_and_defers_fl() -> None:
    observation = _t0_first()
    keys = _keys(observation)
    with pytest.raises(ValueError, match="complete legal action set"):
        encode_street_policy_batch([observation], [keys[:8]], [keys[0]])
    fl = ActorObservation(
        hero_board=Board(),
        opponent_public_board=Board(),
        dealt_cards=tuple(f"{rank}h" for rank in "23456789TJQKA") + ("2d",),
        hero_private_discards=(),
        seat="first",
        street="FL",
        to_act_order="first",
        hero_in_fantasyland=True,
    )
    with pytest.raises(ValueError, match="FL requires"):
        encode_street_policy_batch(
            [fl],
            [[ActionKey()]],
            [ActionKey()],
            require_complete_legal_set=False,
        )


def test_cpu_forward_all_heads_and_illegal_mask() -> None:
    torch = pytest.importorskip("torch")
    encoded = _encoded_pair()
    # Exercise padding/masking with a smaller legal set in the first row.
    first = _t0_first()
    first_keys = _keys(first)[:7]
    short = encode_street_policy_batch(
        [first],
        [first_keys],
        [first_keys[2]],
        require_complete_legal_set=False,
    )
    config = StreetPolicyNetV1Config(
        card_embedding_dim=8,
        zone_embedding_dim=4,
        token_hidden_dim=12,
        context_hidden_dim=8,
        seat_embedding_dim=4,
        street_embedding_dim=4,
        state_hidden_dim=16,
        action_hidden_dim=16,
    )
    torch.manual_seed(7)
    model = build_street_policy_net_v1(torch, config).eval()
    with torch.inference_mode():
        output = model(**short.to_torch(torch))
        paired = model(**encoded.to_torch(torch))
    assert output["policy_logits"].shape == (1, 232)
    assert output["state_value"].shape == (1,)
    for name in (
        "action_q",
        "baseline_delta",
        "uncertainty_p95",
        "safe_logits",
        "safe_probability",
    ):
        assert output[name].shape == (1, 232)
    assert torch.isneginf(output["policy_logits"][0, 7:]).all()
    assert torch.equal(output["action_q"][0, 7:], torch.zeros(225))
    assert output["baseline_delta"][0, 2].item() == 0.0
    assert torch.all(output["uncertainty_p95"][0, :7] > 0)
    assert paired["policy_logits"].shape == (2, 232)
    assert model.seat_embedding.num_embeddings == 2
    assert hasattr(model, "policy_seat_calibration")
    assert sum(name == "state_backbone" for name, _ in model.named_modules()) == 1


def test_model_output_is_invariant_to_card_order() -> None:
    torch = pytest.importorskip("torch")
    original = _t0_first()
    permuted = _t0_first(reverse_dealt=True)
    keys = _keys(original)
    encoded = encode_street_policy_batch(
        [original, permuted],
        [keys, list(reversed(keys))],
        [keys[0], keys[0]],
    )
    torch.manual_seed(11)
    model = build_street_policy_net_v1(
        torch,
        StreetPolicyNetV1Config(
            card_embedding_dim=8,
            zone_embedding_dim=4,
            token_hidden_dim=8,
            context_hidden_dim=8,
            seat_embedding_dim=4,
            street_embedding_dim=4,
            state_hidden_dim=12,
            action_hidden_dim=12,
        ),
    ).eval()
    with torch.inference_mode():
        output = model(**encoded.to_torch(torch))
    for name in (
        "policy_logits",
        "state_value",
        "action_q",
        "baseline_delta",
        "uncertainty_p95",
        "safe_logits",
    ):
        torch.testing.assert_close(output[name][0], output[name][1], rtol=0, atol=0)


def test_network_rejects_undeclared_hidden_tensor_field() -> None:
    torch = pytest.importorskip("torch")
    first = _t0_first()
    keys = _keys(first)
    batch = encode_street_policy_batch(
        [first], [keys], [keys[0]]
    ).to_torch(torch)
    batch["opponent_private_discards"] = torch.zeros((1, 4), dtype=torch.long)
    model = build_street_policy_net_v1(torch, StreetPolicyNetV1Config())
    with pytest.raises(ValueError, match="hidden truth"):
        model(**batch)


def test_parameter_ownership_and_split_role_guards() -> None:
    torch = pytest.importorskip("torch")
    model = build_street_policy_net_v1(torch, StreetPolicyNetV1Config())
    core = set(parameter_names_for_update(model, "core"))
    risk = set(parameter_names_for_update(model, "risk"))
    all_names = {name for name, _ in model.named_parameters()}
    assert core
    assert risk
    assert not core & risk
    assert core | risk == all_names
    assert all(
        name.startswith(("uncertainty_head.", "safe_head.", "uncertainty_seat_calibration.", "safe_seat_calibration."))
        for name in risk
    )
    authorize_weight_update(split_role="train", update_scope="core")
    authorize_weight_update(split_role="safety-fit", update_scope="risk")
    authorize_weight_update(split_role="threshold-lock", update_scope=None)
    with pytest.raises(PermissionError):
        authorize_weight_update(split_role="threshold-lock", update_scope="core")
    with pytest.raises(PermissionError):
        authorize_weight_update(split_role="safety-fit", update_scope="core")
    with pytest.raises(PermissionError):
        authorize_weight_update(split_role="train", update_scope="risk")
    optimizer = build_authorized_optimizer(
        torch,
        model,
        split_role="safety-fit",
        update_scope="risk",
        learning_rate=1e-3,
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    named = dict(model.named_parameters())
    assert optimizer_ids == {id(named[name]) for name in risk}


def test_core_and_risk_losses_follow_split_and_gradient_boundaries() -> None:
    torch = pytest.importorskip("torch")
    first = _t0_first()
    keys = _keys(first)[:6]
    encoded = encode_street_policy_batch(
        [first], [keys], [keys[0]], require_complete_legal_set=False
    )
    torch.manual_seed(17)
    model = build_street_policy_net_v1(
        torch,
        StreetPolicyNetV1Config(
            card_embedding_dim=8,
            zone_embedding_dim=4,
            token_hidden_dim=8,
            context_hidden_dim=8,
            seat_embedding_dim=4,
            street_embedding_dim=4,
            state_hidden_dim=12,
            action_hidden_dim=12,
        ),
    )
    batch = encoded.to_torch(torch)
    legal = batch["legal_action_mask"]
    teacher = torch.zeros((1, 232))
    teacher[0, :6] = torch.tensor([0.35, 0.25, 0.15, 0.1, 0.1, 0.05])
    q = torch.zeros((1, 232))
    q[0, :6] = torch.linspace(-1, 1, 6)
    core_loss = street_policy_training_loss(
        torch,
        model(**batch),
        {
            "action_q": q,
            "baseline_delta": q - q[:, :1],
            "teacher_policy": teacher,
            "state_value": torch.tensor([0.25]),
        },
        split_role="train",
        update_scope="core",
    )
    assert set(core_loss["components"]) == {
        "action_q_huber",
        "baseline_delta_huber",
        "teacher_policy_kl",
        "state_value_huber",
        "ranking",
    }
    core_loss["total"].backward()
    named = dict(model.named_parameters())
    assert any(named[name].grad is not None for name in parameter_names_for_update(model, "core"))
    assert all(named[name].grad is None for name in parameter_names_for_update(model, "risk"))

    model.zero_grad(set_to_none=True)
    risk_loss = street_policy_training_loss(
        torch,
        model(**batch),
        {
            "downside_p95": torch.where(
                legal, torch.full((1, 232), 2.0), torch.zeros((1, 232))
            ),
            "safe": torch.where(
                legal, torch.ones((1, 232)), torch.zeros((1, 232))
            ),
        },
        split_role="safety-fit",
        update_scope="risk",
    )
    assert set(risk_loss["components"]) == {
        "uncertainty_quantile",
        "safe_bce",
    }
    risk_loss["total"].backward()
    assert all(named[name].grad is None for name in parameter_names_for_update(model, "core"))
    assert any(named[name].grad is not None for name in parameter_names_for_update(model, "risk"))


def test_checkpoint_is_byte_deterministic_write_once_and_round_trips(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    config = StreetPolicyNetV1Config(
        card_embedding_dim=8,
        zone_embedding_dim=4,
        token_hidden_dim=8,
        context_hidden_dim=8,
        seat_embedding_dim=4,
        street_embedding_dim=4,
        state_hidden_dim=12,
        action_hidden_dim=12,
    )
    torch.manual_seed(23)
    model = build_street_policy_net_v1(torch, config).eval()
    one = tmp_path / "one.spn1"
    two = tmp_path / "two.spn1"
    provenance = {
        "split_role": "train",
        "dataset_identity": "pilot-only",
        "optimizer_steps": 1,
    }
    manifest_one = save_street_policy_checkpoint(
        one, model, provenance=provenance
    )
    manifest_two = save_street_policy_checkpoint(
        two, model, provenance=provenance
    )
    assert one.read_bytes() == two.read_bytes()
    assert (
        manifest_one["checkpoint_identity_sha256"]
        == manifest_two["checkpoint_identity_sha256"]
    )
    assert manifest_one["model_state_sha256"] == model_state_sha256(model)
    with pytest.raises(FileExistsError):
        save_street_policy_checkpoint(one, model, provenance=provenance)

    loaded, loaded_manifest = load_street_policy_checkpoint(one, torch=torch)
    assert loaded_manifest == manifest_one
    assert model_state_sha256(loaded) == model_state_sha256(model)
    encoded = _encoded_pair().to_torch(torch)
    with torch.inference_mode():
        expected = model(**encoded)
        actual = loaded.eval()(**encoded)
    for name in (
        "policy_logits",
        "state_value",
        "action_q",
        "baseline_delta",
        "uncertainty_p95",
        "safe_logits",
    ):
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def test_checkpoint_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = build_street_policy_net_v1(torch, StreetPolicyNetV1Config())
    original = tmp_path / "original.spn1"
    tampered = tmp_path / "tampered.spn1"
    save_street_policy_checkpoint(
        original,
        model,
        provenance={"split_role": "train", "dataset_identity": "pilot"},
    )
    with zipfile.ZipFile(original, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["feature_schema_hash"] = "0" * 64
    entries["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as output:
        for name in sorted(entries):
            output.writestr(name, entries[name])
    with pytest.raises(ValueError, match="feature schema hash"):
        load_street_policy_checkpoint(tampered, torch=torch)


def test_cuda_forward_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is optional")
    first = _t0_first()
    keys = _keys(first)[:4]
    encoded = encode_street_policy_batch(
        [first], [keys], [keys[0]], require_complete_legal_set=False
    )
    model = build_street_policy_net_v1(
        torch,
        StreetPolicyNetV1Config(
            card_embedding_dim=8,
            zone_embedding_dim=4,
            token_hidden_dim=8,
            context_hidden_dim=8,
            seat_embedding_dim=4,
            street_embedding_dim=4,
            state_hidden_dim=12,
            action_hidden_dim=12,
        ),
    ).to("cuda")
    with torch.inference_mode():
        output = model(**encoded.to_torch(torch, device="cuda"))
    assert output["policy_logits"].is_cuda
    assert output["policy_logits"].shape == (1, 232)
