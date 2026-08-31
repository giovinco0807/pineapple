from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from ai.tutor.t3_bb_checkpoint_bundle import (
    BUNDLE_SCHEMA,
    T3BBCheckpointBundleEntrySource,
    T3BBCheckpointBundleError,
    T3BBCheckpointKey,
    build_t3_bb_checkpoint_bundle,
    canonical_json,
    canonical_sha256,
    load_t3_bb_checkpoint_bundle_root_policies,
    load_verified_t3_bb_checkpoint_strategy_profiles,
    read_t3_bb_checkpoint_bundle,
    verify_t3_bb_checkpoint_bundle,
    write_t3_bb_checkpoint_bundle,
    write_t3_bb_checkpoint_bundle_manifest,
)
from ai.tutor.t3_hu_full_card_mccfr import (
    FullCardGenerativeAdapter,
    solve_full_card_external_sampling_mccfr,
)


def _sha(value: object) -> str:
    return canonical_sha256({"fixture": value})


def _checkpoint_and_strategy(
    *,
    root_id: str,
    solver_seed: int,
    range_content_sha256: str,
    range_build_sha256: str,
) -> tuple[bytes, bytes, str]:
    infoset = {
        "actor": "bb",
        "fixture_root": root_id,
        "fixture_seed": solver_seed,
        "phase": "t3_first",
    }
    infoset_json = canonical_json(infoset)
    observation_digest = hashlib.sha256(infoset_json.encode("utf-8")).hexdigest()
    range_binding = {
        "observation_digest": observation_digest,
        "particle_count": 2,
        "particle_commitments_sha256": _sha((root_id, "particles")),
        "root_distribution_sha256": _sha((root_id, "distribution")),
        "range_content_sha256": range_content_sha256,
        "range_build_sha256": range_build_sha256,
        "behavior_model_id": "fixture",
        "behavior_model_sha256": _sha("behavior"),
        "behavior_model_manifest": {"fixture": True},
    }
    adapter_unsigned = {
        "format": "full_card_dynamic_mccfr_adapter_manifest_v1",
        "adapter": "full_card_generative_t3_t4_v1",
        "root_infoset_canonical_json": infoset_json,
        "root_infoset_sha256": observation_digest,
        "remaining_phase_sequence": [
            "t3_first",
            "t3_second",
            "t4_first",
            "t4_second",
        ],
        "root_sampling_contract": "exact_fraction_sequential_conditional_v1",
        "draw_sampling_contract": "uniform_unordered_three_card_combinadic_v1",
        "policy_identity_contract": "infoset_key_plus_lexical_action_key_v1",
        "position_contract_version": "bb_first_v1",
        "physical_joker_ids": ["X1", "X2"],
        "range_behavior_binding": range_binding,
        "terminal_utility_binding": {"fixture": "terminal"},
    }
    adapter = dict(adapter_unsigned)
    adapter["manifest_sha256"] = canonical_sha256(adapter_unsigned)
    table = {
        "infoset_canonical_json": infoset_json,
        "infoset_sha256": observation_digest,
        "actor": "bb",
        "stable_action_ids": ["action-a", "action-b"],
        "regret_plus_hex": [float(0.0).hex(), float(2.0).hex()],
        "strategy_sum_hex": [float(1.0).hex(), float(3.0).hex()],
    }
    payload = {
        "format": "full_card_dynamic_mccfr_state_v1",
        "completed_iterations": 10,
        "seed": solver_seed,
        "linear_averaging": True,
        "solver_config": {"method": "fixture-mccfr"},
        "adapter_manifest": adapter,
        "adapter_manifest_sha256": canonical_sha256(adapter),
        "tables": [table],
        "rng_state": {"fixture": "rng"},
        "sampling_stats": {"traversals": 20},
    }
    envelope = {
        "format": "full_card_dynamic_mccfr_checkpoint_v1",
        "checkpoint_sha256": canonical_sha256(payload),
        "payload": payload,
    }
    checkpoint_bytes = canonical_json(envelope).encode("utf-8") + b"\n"
    strategy = {
        "schema": "ofc_full_card_public_strategy/v1",
        "policy_identity_contract": "infoset_key_plus_lexical_action_key_v1",
        "records": [
            {
                "infoset_digest": observation_digest,
                "infoset": infoset,
                "actions": [
                    {"action_id": "action-a", "probability": 0.25},
                    {"action_id": "action-b", "probability": 0.75},
                ],
            }
        ],
    }
    strategy_bytes = canonical_json(strategy).encode("utf-8")
    return checkpoint_bytes, strategy_bytes, observation_digest


def _write_job(
    root: Path, *, round_index: int, root_id: str, solver_seed: int
) -> tuple[T3BBCheckpointKey, T3BBCheckpointBundleEntrySource]:
    commitment = _sha(("root-commitment", root_id))
    range_content = _sha(("range-content", root_id, solver_seed))
    range_build = _sha(("range-build", root_id, solver_seed))
    checkpoint, strategy, observation = _checkpoint_and_strategy(
        root_id=root_id,
        solver_seed=solver_seed,
        range_content_sha256=range_content,
        range_build_sha256=range_build,
    )
    directory = root / "jobs" / f"{root_id}-{solver_seed}"
    directory.mkdir(parents=True)
    checkpoint_path = directory / "checkpoint.json"
    strategy_path = directory / "average-strategy.json"
    checkpoint_path.write_bytes(checkpoint)
    strategy_path.write_bytes(strategy)
    relative_checkpoint = checkpoint_path.relative_to(root).as_posix()
    relative_strategy = strategy_path.relative_to(root).as_posix()
    key = T3BBCheckpointKey(
        round_index=round_index,
        root_id=root_id,
        root_commitment_sha256=commitment,
        solver_seed=solver_seed,
    )
    source = T3BBCheckpointBundleEntrySource(
        **key.as_dict(),
        checkpoint_path=relative_checkpoint,
        average_strategy_json_path=relative_strategy,
        range_content_sha256=range_content,
        range_build_sha256=range_build,
        observation_digest=observation,
        solver_manifest_sha256=_sha(("solver-manifest", root_id, solver_seed)),
        source_manifest_sha256=_sha("source-manifest"),
    )
    return key, source


def _fixture_bundle(
    tmp_path: Path, *, round_index: int = 3
) -> tuple[Path, list[T3BBCheckpointKey], list[T3BBCheckpointBundleEntrySource]]:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    pairs = [
        _write_job(root, round_index=round_index, root_id=root_id, solver_seed=seed)
        for root_id in ("root-a", "root-b")
        for seed in (17, 29)
    ]
    return root, [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _resign_entry_and_manifest(manifest: dict, index: int) -> None:
    entry = manifest["entries"][index]
    entry.pop("entry_sha256", None)
    entry["entry_sha256"] = canonical_sha256(entry)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)


def test_round_bundle_is_deterministic_complete_nonpromoting_and_fresh_verified(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    first = build_t3_bb_checkpoint_bundle(
        root, list(reversed(sources)), expected_keys=list(reversed(keys))
    )
    second = build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)

    assert first == second
    assert first["schema"] == BUNDLE_SCHEMA
    assert first["round_index"] == 3
    assert first["entry_count"] == 4
    assert first["promotion_eligible"] is False
    assert first["exact_exploitability_computed"] is False
    assert len(first["bundle_checkpoint_sha256"]) == 64
    assert [
        (entry["root_id"], entry["solver_seed"]) for entry in first["entries"]
    ] == [("root-a", 17), ("root-a", 29), ("root-b", 17), ("root-b", 29)]
    for entry in first["entries"]:
        actual = (root / entry["checkpoint_path"]).read_bytes()
        assert entry["checkpoint_file_bytes_sha256"] == hashlib.sha256(actual).hexdigest()
        assert len(entry["average_strategy_json_bytes_sha256"]) == 64
        assert len(entry["average_strategy_json_content_sha256"]) == 64

    assert verify_t3_bb_checkpoint_bundle(
        first, bundle_root=root, expected_keys=keys
    ) == first


def test_atomic_write_and_canonical_readback(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    written = write_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)
    path = root / "manifest.json"

    assert path.read_bytes() == canonical_json(written).encode("utf-8") + b"\n"
    assert read_t3_bb_checkpoint_bundle(path, expected_keys=keys) == written
    assert not list(root.glob(".manifest.json.*.tmp"))

    second_path = root / "published.json"
    assert write_t3_bb_checkpoint_bundle_manifest(
        second_path, written, bundle_root=root, expected_keys=keys
    ) == written


def test_profiles_and_root_policies_are_loaded_from_verified_strategy_bytes(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    manifest = build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)

    profiles = load_verified_t3_bb_checkpoint_strategy_profiles(
        manifest, bundle_root=root, expected_keys=keys
    )
    roots = load_t3_bb_checkpoint_bundle_root_policies(
        manifest, bundle_root=root, expected_keys=keys
    )
    assert set(profiles) == set(keys) == set(roots)
    for key in keys:
        assert len(profiles[key]) == 1
        distribution = next(iter(profiles[key].values()))
        assert distribution == {
            "action-a": Fraction(1, 4),
            "action-b": Fraction(3, 4),
        }
        assert roots[key] == {"action-a": 0.25, "action-b": 0.75}


def test_real_full_card_solver_checkpoint_and_result_strategy_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise actual solver serialization, not a hand-built envelope only."""

    helper_path = Path(__file__).with_name("test_t3_hu_full_card_mccfr.py")
    spec = importlib.util.spec_from_file_location(
        "t3_bundle_real_mccfr_helpers", helper_path
    )
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    observation, root_range = helpers._one_world_natural_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    helpers._install_light_terminal(monkeypatch, adapter)

    root = tmp_path / "real-bundle"
    root.mkdir()
    checkpoint = root / "checkpoint.json"
    strategy = root / "average-strategy.json"
    result = solve_full_card_external_sampling_mccfr(
        adapter,
        iterations=1,
        seed=313,
        max_infosets=50_000,
        checkpoint_path=checkpoint,
    )
    strategy.write_text(result.average_strategy_json, encoding="utf-8")
    key = T3BBCheckpointKey(
        round_index=1,
        root_id="real-full-card-e2e",
        root_commitment_sha256=_sha("real-full-card-e2e"),
        solver_seed=313,
    )
    source = T3BBCheckpointBundleEntrySource(
        **key.as_dict(),
        checkpoint_path="checkpoint.json",
        average_strategy_json_path="average-strategy.json",
        range_content_sha256=root_range.range_content_sha256,
        range_build_sha256=root_range.range_build_sha256,
        observation_digest=observation.digest(),
        solver_manifest_sha256=_sha("real-solver-manifest"),
        source_manifest_sha256=_sha("real-source-manifest"),
    )

    manifest = build_t3_bb_checkpoint_bundle(root, [source], expected_keys=[key])
    profiles = load_verified_t3_bb_checkpoint_strategy_profiles(
        manifest, bundle_root=root, expected_keys=[key]
    )
    entry = manifest["entries"][0]
    assert entry["average_strategy_json_bytes_sha256"] == result.average_strategy_sha256
    assert entry["checkpoint_content_sha256"] == result.metadata["checkpoint_sha256"]
    assert observation.digest() in profiles[key]
    assert len(profiles[key][observation.digest()]) == 3


@pytest.mark.parametrize("asset", ["checkpoint", "strategy"])
def test_post_build_asset_tampering_fails_fresh_readback(
    tmp_path: Path, asset: str
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    manifest = build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)
    entry = manifest["entries"][0]
    field = "checkpoint_path" if asset == "checkpoint" else "average_strategy_json_path"
    path = root / entry[field]
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(T3BBCheckpointBundleError):
        verify_t3_bb_checkpoint_bundle(manifest, bundle_root=root, expected_keys=keys)


def test_strategy_must_be_the_checkpoint_derived_average_policy(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    strategy_path = root / os.fspath(sources[0].average_strategy_json_path)
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    strategy["records"][0]["actions"][0]["probability"] = 0.5
    strategy["records"][0]["actions"][1]["probability"] = 0.5
    strategy_path.write_text(canonical_json(strategy), encoding="utf-8")

    with pytest.raises(T3BBCheckpointBundleError, match="strategy sums"):
        build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)


def test_checkpoint_seed_range_and_observation_are_bound_to_entry(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    changes = (
        {"solver_seed": 999},
        {"range_content_sha256": _sha("wrong-range")},
        {"range_build_sha256": _sha("wrong-build")},
        {"observation_digest": _sha("wrong-observation")},
    )
    for change in changes:
        changed = [copy.copy(source) for source in sources]
        values = dict(changed[0].__dict__)
        values.update(change)
        changed[0] = T3BBCheckpointBundleEntrySource(**values)
        with pytest.raises(T3BBCheckpointBundleError, match="mismatch"):
            build_t3_bb_checkpoint_bundle(root, changed, expected_keys=keys)


def test_missing_or_unexpected_key_is_a_coverage_gap(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    with pytest.raises(T3BBCheckpointBundleError, match="exact coverage mismatch"):
        build_t3_bb_checkpoint_bundle(root, sources[:-1], expected_keys=keys)

    extra_key, extra_source = _write_job(
        root, round_index=3, root_id="root-c", solver_seed=17
    )
    with pytest.raises(T3BBCheckpointBundleError, match="exact coverage mismatch"):
        build_t3_bb_checkpoint_bundle(
            root, [*sources, extra_source], expected_keys=keys
        )
    assert extra_key not in keys


def test_duplicate_keys_expected_keys_and_physical_paths_are_rejected(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    with pytest.raises(T3BBCheckpointBundleError, match="duplicate key"):
        build_t3_bb_checkpoint_bundle(root, sources, expected_keys=[*keys, keys[0]])
    with pytest.raises(T3BBCheckpointBundleError, match="duplicate coverage key"):
        build_t3_bb_checkpoint_bundle(
            root, [*sources, sources[0]], expected_keys=keys
        )

    alias_values = dict(sources[1].__dict__)
    alias_values["checkpoint_path"] = sources[0].checkpoint_path
    alias_values["average_strategy_json_path"] = sources[0].average_strategy_json_path
    alias_values["solver_seed"] = sources[0].solver_seed
    alias_values["root_id"] = "root-alias"
    alias_values["root_commitment_sha256"] = _sha("root-alias")
    alias = T3BBCheckpointBundleEntrySource(**alias_values)
    alias_key = alias.key
    with pytest.raises(T3BBCheckpointBundleError):
        build_t3_bb_checkpoint_bundle(
            root, [sources[0], alias], expected_keys=[keys[0], alias_key]
        )


@pytest.mark.parametrize(
    "bad_path",
    ["../outside.json", "/absolute/checkpoint.json", "C:/escape.json", "."],
)
def test_path_traversal_and_absolute_assets_are_rejected(
    tmp_path: Path, bad_path: str
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    values = dict(sources[0].__dict__)
    values["checkpoint_path"] = bad_path
    changed = [T3BBCheckpointBundleEntrySource(**values), *sources[1:]]
    with pytest.raises(T3BBCheckpointBundleError, match="path|asset"):
        build_t3_bb_checkpoint_bundle(root, changed, expected_keys=keys)


def test_symlink_assets_are_rejected_even_when_the_target_is_inside_root(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    target = root / os.fspath(sources[0].checkpoint_path)
    link = root / "checkpoint-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    values = dict(sources[0].__dict__)
    values["checkpoint_path"] = link.relative_to(root).as_posix()
    changed = [T3BBCheckpointBundleEntrySource(**values), *sources[1:]]
    with pytest.raises(T3BBCheckpointBundleError, match="symlink"):
        build_t3_bb_checkpoint_bundle(root, changed, expected_keys=keys)


def test_hash_string_is_not_checkpoint_bytes_or_an_asset_reference(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    checkpoint = root / os.fspath(sources[0].checkpoint_path)
    checkpoint.write_bytes((_sha("pretend-checkpoint") + "\n").encode("ascii"))
    with pytest.raises(T3BBCheckpointBundleError, match="not checkpoint bytes"):
        build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)

    root2, keys2, sources2 = _fixture_bundle(tmp_path / "other")
    values = dict(sources2[0].__dict__)
    values["checkpoint_path"] = _sha("only-a-reference")
    changed = [T3BBCheckpointBundleEntrySource(**values), *sources2[1:]]
    with pytest.raises(T3BBCheckpointBundleError, match="asset"):
        build_t3_bb_checkpoint_bundle(root2, changed, expected_keys=keys2)


def test_resigned_hash_claim_cannot_replace_fresh_file_derivation(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    manifest = build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)
    forged = copy.deepcopy(manifest)
    forged["entries"][0]["checkpoint_file_bytes_sha256"] = _sha("forged")
    _resign_entry_and_manifest(forged, 0)

    with pytest.raises(T3BBCheckpointBundleError, match="fresh file-derived"):
        verify_t3_bb_checkpoint_bundle(forged, bundle_root=root, expected_keys=keys)


def test_checkpoint_internal_payload_hash_is_verified(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    checkpoint = root / os.fspath(sources[0].checkpoint_path)
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["payload"]["completed_iterations"] = 11
    checkpoint.write_bytes(canonical_json(envelope).encode("utf-8") + b"\n")

    with pytest.raises(T3BBCheckpointBundleError, match="content hash mismatch"):
        build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)


def test_manifest_self_hash_and_locked_expected_set_are_both_required(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    manifest = build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)
    tampered = copy.deepcopy(manifest)
    tampered["scope"] = "forged"
    with pytest.raises(T3BBCheckpointBundleError, match="scope"):
        verify_t3_bb_checkpoint_bundle(tampered, bundle_root=root, expected_keys=keys)

    with pytest.raises(T3BBCheckpointBundleError, match="coverage"):
        verify_t3_bb_checkpoint_bundle(
            manifest, bundle_root=root, expected_keys=keys[:-1]
        )


def test_one_bundle_cannot_mix_rounds_and_hash_bindings_must_be_lowercase_sha256(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    mixed = [*keys[:-1], T3BBCheckpointKey(**{**keys[-1].as_dict(), "round_index": 4})]
    with pytest.raises(T3BBCheckpointBundleError, match="exactly one round"):
        build_t3_bb_checkpoint_bundle(root, sources, expected_keys=mixed)

    values = dict(sources[0].__dict__)
    values["solver_manifest_sha256"] = "A" * 64
    changed = [T3BBCheckpointBundleEntrySource(**values), *sources[1:]]
    with pytest.raises(T3BBCheckpointBundleError, match="lowercase SHA256"):
        build_t3_bb_checkpoint_bundle(root, changed, expected_keys=keys)


def test_one_root_id_has_one_commitment_and_native_path_objects_are_supported(
    tmp_path: Path,
) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    changed_key = T3BBCheckpointKey(
        **{**keys[1].as_dict(), "root_commitment_sha256": _sha("drift")}
    )
    with pytest.raises(T3BBCheckpointBundleError, match="multiple commitments"):
        build_t3_bb_checkpoint_bundle(
            root, sources, expected_keys=[keys[0], changed_key, *keys[2:]]
        )

    values = dict(sources[0].__dict__)
    values["checkpoint_path"] = Path(os.fspath(sources[0].checkpoint_path))
    values["average_strategy_json_path"] = Path(
        os.fspath(sources[0].average_strategy_json_path)
    )
    native_source = T3BBCheckpointBundleEntrySource(**values)
    manifest = build_t3_bb_checkpoint_bundle(
        root, [native_source, *sources[1:]], expected_keys=keys
    )
    assert manifest["entry_count"] == 4


def test_read_rejects_noncanonical_or_duplicate_key_manifest_json(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    manifest = build_t3_bb_checkpoint_bundle(root, sources, expected_keys=keys)
    path = root / "bad.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(T3BBCheckpointBundleError, match="non-canonical"):
        read_t3_bb_checkpoint_bundle(path, expected_keys=keys)

    path.write_text('{"schema":1,"schema":2}\n', encoding="utf-8")
    with pytest.raises(T3BBCheckpointBundleError, match="duplicate JSON key"):
        read_t3_bb_checkpoint_bundle(path, expected_keys=keys)


def test_manifest_name_cannot_escape_bundle_root(tmp_path: Path) -> None:
    root, keys, sources = _fixture_bundle(tmp_path)
    with pytest.raises(T3BBCheckpointBundleError, match="plain file name"):
        write_t3_bb_checkpoint_bundle(
            root, sources, expected_keys=keys, manifest_name="../manifest.json"
        )
