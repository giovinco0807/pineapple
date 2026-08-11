from __future__ import annotations

import hashlib
import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_opt_in_registration_v1 as subject
from ofc_regular import hu_m31_t3_street_policy_runtime_v1 as runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_REGISTRY = REPOSITORY_ROOT / "src/ofc_regular/ai_profiles.py"


def _write_canonical(path: Path, value: dict[str, Any]) -> Path:
    path.write_bytes(subject.canonical_bytes(value))
    return path


def _file_record(path: Path, schema: str = "fixture") -> dict[str, Any]:
    return {
        "absolute_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": subject.sha256_file(path),
        "schema": schema,
        "internal_identity_sha256": "a" * 64,
    }


def _resolver_manifest(tmp_path: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "plan",
        "merge",
        "gate",
        "rich",
        "compact",
        "closure",
        "registry",
        "config",
    ):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_bytes(name.encode("ascii"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest_path = bundle / "manifest.json"
    manifest_path.write_bytes(b"manifest")
    return {
        "schema": subject.REGISTRATION_MANIFEST_SCHEMA,
        "status": subject.REGISTRATION_STATUS,
        "profile_id": subject.PROFILE_ID,
        "baseline_profile_id": subject.BASELINE_PROFILE_ID,
        "manifest_identity_sha256": "b" * 64,
        "artifacts": {
            "promotion_plan": _file_record(files["plan"]),
            "promotion_merge": _file_record(files["merge"]),
            "promotion_gate": _file_record(files["gate"]),
            "checkpoint_bundle": {
                "absolute_path": str(bundle.resolve()),
                "manifest_absolute_path": str(manifest_path.resolve()),
                "manifest_file_sha256": subject.sha256_file(manifest_path),
                "bundle_identity_sha256": "c" * 64,
                "training_view_identity_sha256": "d" * 64,
                "members": [],
            },
            "training_run_config": _file_record(files["config"]),
            "training_threshold_lock": _file_record(files["rich"]),
            "compatibility_threshold_lock": _file_record(files["compact"]),
            "evaluation_runtime_closure": _file_record(files["closure"]),
            "policy_registry_before": _file_record(files["registry"]),
        },
    }


def _patch_manifest(tmp_path: Path) -> dict[str, Any]:
    frozen = tmp_path / "frozen-ai_profiles.py"
    if not frozen.exists():
        frozen.write_bytes(POLICY_REGISTRY.read_bytes())
    return {
        "manifest_identity_sha256": "a" * 64,
        "artifacts": {
            "policy_registry_before": {
                "absolute_path": str(frozen.resolve()),
                "file_sha256": subject.PINNED_POLICY_REGISTRY_SHA256,
            }
        },
    }


def _write_patch_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, Path]:
    manifest = _patch_manifest(tmp_path)
    patch_target = tmp_path / "live-ai_profiles.py"
    if not patch_target.exists():
        patch_target.write_bytes(POLICY_REGISTRY.read_bytes())
    monkeypatch.setattr(
        subject,
        "validate_registration_manifest",
        lambda *args, **kwargs: manifest,
    )
    spec = subject.build_ai_profiles_patch_spec(
        registration_manifest_path=tmp_path / "manifest.json",
        expected_registration_manifest_file_sha256="1" * 64,
        policy_registry_patch_target_path=patch_target.resolve(),
        torch=object(),
        source_replay_root=REPOSITORY_ROOT,
    )
    path = _write_canonical(tmp_path / "patch-spec.json", spec)
    return spec, path, patch_target


def _passing_gate() -> dict[str, Any]:
    return {
        "status": "pass",
        "all_gates_passed": True,
        "scientific_promotion_passed": True,
        "separate_opt_in_profile_candidate_authorized": True,
        "gates": {"all": True},
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }


def test_policy_registry_is_unchanged_and_pinned() -> None:
    assert subject.sha256_file(POLICY_REGISTRY) == (
        "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    )
    assert subject.PROFILE_ID not in POLICY_REGISTRY.read_text(
        encoding="utf-8"
    )


def test_manifest_and_resolver_have_no_default_or_environment_fallback() -> None:
    signature = inspect.signature(subject.resolve_explicit_opt_in_t3_policy)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    source = inspect.getsource(subject.resolve_explicit_opt_in_t3_policy)
    assert "getenv" not in source
    assert "environ" not in source
    parser_source = inspect.getsource(subject._add_manifest_arguments)
    assert "required=True" in parser_source
    assert "default=" not in parser_source


def test_file_reader_rejects_relative_missing_and_tampered(
    tmp_path: Path,
) -> None:
    value = {"value": 1}
    path = _write_canonical(tmp_path / "value.json", value)
    expected = subject.sha256_file(path)
    assert subject._read_canonical_file(
        path.resolve(), expected_sha256=expected, label="fixture"
    )[0] == value
    with pytest.raises(ValueError, match="absolute"):
        subject._read_canonical_file(
            Path("value.json"), expected_sha256=expected, label="fixture"
        )
    with pytest.raises(subject.OptInRegistrationError, match="SHA-256"):
        subject._read_canonical_file(
            path.resolve(), expected_sha256="0" * 64, label="fixture"
        )
    path.write_bytes(b"{}")
    with pytest.raises(subject.OptInRegistrationError, match="SHA-256"):
        subject._read_canonical_file(
            path.resolve(), expected_sha256=expected, label="fixture"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "no_go"),
        ("all_gates_passed", False),
        ("scientific_promotion_passed", False),
        ("separate_opt_in_profile_candidate_authorized", False),
        ("current_profile_changed", True),
        ("runtime_activated", True),
        ("full_replacement_enabled", True),
    ],
)
def test_registration_rejects_every_nonpassing_gate_boundary(
    field: str, value: object
) -> None:
    gate = _passing_gate()
    gate[field] = value
    with pytest.raises(PermissionError, match="fully passing"):
        subject._require_passing_gate(gate)


def test_registration_rejects_partial_gate_map() -> None:
    gate = _passing_gate()
    gate["gates"] = {"one": True, "two": False}
    with pytest.raises(PermissionError, match="fully passing"):
        subject._require_passing_gate(gate)


def test_authorization_must_equal_full_source_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = {"authorization_sha256": "0" * 64}
    monkeypatch.setattr(
        subject.post_dataset,
        "build_opt_in_registration_authorization",
        lambda **kwargs: {"different": True},
    )
    with pytest.raises(
        subject.OptInRegistrationError, match="differs from source replay"
    ):
        subject._replay_authorization(
            authorization=authorization,
            promotion_plan={},
            execution_plan={},
            shard_directory=tmp_path,
            merge={},
            gate={},
        )


def test_create_only_writer_refuses_replay(tmp_path: Path) -> None:
    output = (tmp_path / "receipt.json").resolve()
    subject._write_once(output, {"value": 1})
    with pytest.raises(FileExistsError, match="create-only"):
        subject._write_once(output, {"value": 1})


@pytest.mark.parametrize("seat", ["first", "second"])
def test_resolver_forwards_exact_baseline_and_nonfire_rng_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seat: str,
) -> None:
    manifest = _resolver_manifest(tmp_path)
    monkeypatch.setattr(
        subject,
        "validate_registration_manifest",
        lambda *args, **kwargs: manifest,
    )
    config = SimpleNamespace(training_config=object())
    monkeypatch.setattr(
        subject.training_cli, "load_run_config", lambda *args, **kwargs: config
    )

    class Baseline:
        def __init__(self) -> None:
            self.seat = seat
            self.action = object()
            self.rng = random.Random(771 + (seat == "second"))
            self.calls = 0

        def choose_action_observation(self, observation: object) -> object:
            self.calls += 1
            self.rng.random()
            return self.action

    class Candidate:
        def __init__(self, baseline: Baseline) -> None:
            self._baseline_policy = baseline
            self.runtime_scope = runtime.RUNTIME_SCOPE_QUALIFIED_OPT_IN

        def choose_action_observation(self, observation: object) -> object:
            # This represents the existing runtime's locked non-fire path.
            return self._baseline_policy.choose_action_observation(observation)

    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Candidate:
        captured.update(kwargs)
        return Candidate(kwargs["baseline_policy"])

    monkeypatch.setattr(
        subject.policy_runtime,
        "build_opt_in_t3_policy_candidate",
        factory,
    )
    baseline = Baseline()
    control = random.Random()
    control.setstate(baseline.rng.getstate())
    control.random()
    candidate = subject.resolve_explicit_opt_in_t3_policy(
        registration_manifest_path=(tmp_path / "registration.json").resolve(),
        expected_registration_manifest_file_sha256="e" * 64,
        baseline_policy=baseline,
        baseline_profile_id=subject.BASELINE_PROFILE_ID,
        torch=object(),
        source_replay_root=REPOSITORY_ROOT,
    )
    assert captured["baseline_policy"] is baseline
    assert captured["baseline_profile_id"] == "stage7_m5_r10"
    assert baseline.calls == 0
    returned = candidate.choose_action_observation(object())
    assert returned is baseline.action
    assert baseline.calls == 1
    assert baseline.rng.getstate() == control.getstate()


def test_resolver_cannot_resolve_current_before_any_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subject,
        "validate_registration_manifest",
        lambda *args, **kwargs: pytest.fail("manifest must not be read"),
    )
    with pytest.raises(ValueError, match="current is forbidden"):
        subject.resolve_explicit_opt_in_t3_policy(
            registration_manifest_path=(tmp_path / "missing.json").resolve(),
            expected_registration_manifest_file_sha256="f" * 64,
            baseline_policy=object(),
            baseline_profile_id="current",
            torch=object(),
            source_replay_root=REPOSITORY_ROOT,
        )


def test_manifest_validator_rejects_tamper_before_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _resolver_manifest(tmp_path)
    payload = dict(manifest)
    payload.pop("manifest_identity_sha256")
    manifest["manifest_identity_sha256"] = subject.canonical_sha256(payload)
    path = _write_canonical(tmp_path / "registration.json", manifest)
    file_sha = subject.sha256_file(path)
    monkeypatch.setattr(
        subject,
        "build_registration_manifest",
        lambda **kwargs: manifest,
    )
    # The shape is intentionally incomplete for a replay manifest, so the
    # validator must fail closed instead of inventing a default artifact path.
    with pytest.raises(subject.OptInRegistrationError, match="source replay"):
        subject.validate_registration_manifest(
            path.resolve(),
            expected_manifest_file_sha256=file_sha,
            torch=object(),
            source_replay_root=REPOSITORY_ROOT,
        )
    manifest["status"] = "activated"
    path.write_bytes(subject.canonical_bytes(manifest))
    with pytest.raises(
        subject.OptInRegistrationError, match="identity or boundary"
    ):
        subject.validate_registration_manifest(
            path.resolve(),
            expected_manifest_file_sha256=subject.sha256_file(path),
            torch=object(),
            source_replay_root=REPOSITORY_ROOT,
        )


def test_patch_spec_and_before_receipt_do_not_apply_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_sha = subject.sha256_file(POLICY_REGISTRY)
    spec, spec_path, _patch_target = _write_patch_spec(
        tmp_path, monkeypatch
    )
    assert spec["patch_applied"] is False
    assert spec["named_profile_added"] is False
    assert spec["current_profile_changed"] is False
    receipt = subject.build_before_registration_receipt(
        registration_manifest_path=(tmp_path / "manifest.json").resolve(),
        expected_registration_manifest_file_sha256="1" * 64,
        patch_spec_path=spec_path.resolve(),
        expected_patch_spec_file_sha256=subject.sha256_file(spec_path),
        torch=object(),
        source_replay_root=REPOSITORY_ROOT,
    )
    assert receipt["phase"] == "before"
    assert receipt["registration_applied"] is False
    assert receipt["named_profile_added"] is False
    assert receipt["current_profile_changed"] is False
    assert subject.sha256_file(POLICY_REGISTRY) == before_sha


def test_patch_spec_refuses_to_mutate_frozen_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _patch_manifest(tmp_path)
    monkeypatch.setattr(
        subject,
        "validate_registration_manifest",
        lambda *args, **kwargs: manifest,
    )
    with pytest.raises(
        subject.OptInRegistrationError, match="must be distinct"
    ):
        subject.build_ai_profiles_patch_spec(
            registration_manifest_path=(tmp_path / "manifest.json").resolve(),
            expected_registration_manifest_file_sha256="1" * 64,
            policy_registry_patch_target_path=manifest["artifacts"][
                "policy_registry_before"
            ]["absolute_path"],
            torch=object(),
            source_replay_root=REPOSITORY_ROOT,
        )


def _synthetic_after_source() -> str:
    source = POLICY_REGISTRY.read_text(encoding="utf-8")
    source = source.replace(
        'ProfileName = Literal[\n    "current",',
        (
            'ProfileName = Literal[\n'
            '    "current",\n'
            f'    "{subject.PROFILE_ID}",'
        ),
        1,
    )
    source = source.replace(
        "    hu_turn1_topk_config: str | None = None,\n"
        ") -> RegularAiPolicy:",
        (
            "    hu_turn1_topk_config: str | None = None,\n"
            "    m31_registration_manifest_path: str | None = None,\n"
            "    m31_registration_manifest_file_sha256: str | None = None,\n"
            "    m31_torch: Any | None = None,\n"
            "    m31_source_replay_root: str | None = None,\n"
            ") -> RegularAiPolicy:"
        ),
        1,
    )
    branch = (
        f'    if profile == "{subject.PROFILE_ID}":\n'
        "        if any(value is None for value in (\n"
        "            m31_registration_manifest_path,\n"
        "            m31_registration_manifest_file_sha256,\n"
        "            m31_torch,\n"
        "            m31_source_replay_root,\n"
        "        )):\n"
        "            raise ValueError(\"explicit M3.1 registration inputs required\")\n"
        "        baseline = _build_stage7_policy(\n"
        "            bundle,\n"
        "            seed=seed,\n"
        "            seat=seat,\n"
        "            opening_lookahead_samples=opening_lookahead_samples,\n"
        "            enabled=True,\n"
        "            hu_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,\n"
        "            reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,\n"
        "        )\n"
        "        return resolve_explicit_opt_in_t3_policy(\n"
        "            registration_manifest_path=m31_registration_manifest_path,\n"
        "            expected_registration_manifest_file_sha256=m31_registration_manifest_file_sha256,\n"
        "            baseline_policy=baseline,\n"
        '            baseline_profile_id="stage7_m5_r10",\n'
        "            torch=m31_torch,\n"
        "            source_replay_root=m31_source_replay_root,\n"
        "        )\n"
    )
    source = source.replace(
        '    if profile == "current":\n',
        branch + '    if profile == "current":\n',
        1,
    )
    return source


def test_after_receipt_accepts_only_minimal_explicit_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _patch_manifest(tmp_path)
    patch_target = tmp_path / "live-ai_profiles.py"
    patch_target.write_bytes(POLICY_REGISTRY.read_bytes())
    monkeypatch.setattr(
        subject,
        "validate_registration_manifest",
        lambda *args, **kwargs: manifest,
    )
    spec = subject.build_ai_profiles_patch_spec(
        registration_manifest_path=(tmp_path / "manifest.json").resolve(),
        expected_registration_manifest_file_sha256="1" * 64,
        policy_registry_patch_target_path=patch_target.resolve(),
        torch=object(),
        source_replay_root=REPOSITORY_ROOT,
    )
    spec_path = _write_canonical(tmp_path / "patch.json", spec)
    before = subject.build_before_registration_receipt(
        registration_manifest_path=(tmp_path / "manifest.json").resolve(),
        expected_registration_manifest_file_sha256="1" * 64,
        patch_spec_path=spec_path.resolve(),
        expected_patch_spec_file_sha256=subject.sha256_file(spec_path),
        torch=object(),
        source_replay_root=REPOSITORY_ROOT,
    )
    before_path = _write_canonical(tmp_path / "before.json", before)
    patch_target.write_text(_synthetic_after_source(), encoding="utf-8")
    after_sha = subject.sha256_file(patch_target)
    receipt = subject.build_after_registration_receipt(
        registration_manifest_path=(tmp_path / "manifest.json").resolve(),
        expected_registration_manifest_file_sha256="1" * 64,
        patch_spec_path=spec_path.resolve(),
        expected_patch_spec_file_sha256=subject.sha256_file(spec_path),
        before_receipt_path=before_path.resolve(),
        expected_before_receipt_file_sha256=subject.sha256_file(before_path),
        policy_registry_after_path=patch_target.resolve(),
        expected_policy_registry_after_file_sha256=after_sha,
        torch=object(),
        source_replay_root=REPOSITORY_ROOT,
    )
    assert receipt["registration_applied"] is True
    assert receipt["named_profile_added"] is True
    assert receipt["current_profile_changed"] is False
    assert receipt["runtime_activated"] is False
    assert subject.sha256_file(POLICY_REGISTRY) == (
        subject.PINNED_POLICY_REGISTRY_SHA256
    )


def test_after_receipt_rejects_current_branch_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, _ = _write_patch_spec(tmp_path, monkeypatch)
    source = _synthetic_after_source().replace(
        '    if profile == "current":\n',
        '    if profile == "current":\n        seed += 1\n',
        1,
    )
    with pytest.raises(
        subject.OptInRegistrationError, match="current profile branch changed"
    ):
        subject._validate_after_registry(source, spec=spec)


def test_cli_requires_all_manifest_paths() -> None:
    parser = subject._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["create-manifest"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "validate-manifest",
                "--manifest",
                "x",
                "--manifest-sha256",
                "0" * 64,
            ]
        )
