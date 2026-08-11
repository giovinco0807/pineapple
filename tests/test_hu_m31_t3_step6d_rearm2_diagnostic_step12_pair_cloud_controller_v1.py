from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1 as controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_cloud_controller_v1
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1
    as pair_contract_module,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
STEP11_BOOTSTRAP = (
    REPO_ROOT
    / "scripts"
    / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
)
STEP11_PREBOOTSTRAP = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
)
STEP12_PREBOOTSTRAP = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_v1.py"
)
INSTANCE_NAMES = [
    "r2d-10c2-s2-candidate-01-a0-29e3c6f8",
    "r2d-10c2-s2-reference-01-a0-29e3c6f8",
]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_prebootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_step12_prebootstrap",
        STEP12_PREBOOTSTRAP,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pair_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    stage1 = _read(STEP11_ROOT / "transport_contract.json")
    done = _read(STEP11_ROOT / "late_done_envelope.json")
    recovery = _read(STEP11_ROOT / "late_done_recovery_receipt.json")
    public_key = _read(STEP11_ROOT / "controller_public_key.json")
    receive = adapter.build_receive(
        stage1["adapter_preview"],
        done_records=[done],
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    contracts = [
        transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public_key,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=receive,
        )
        for job_id in plan.STAGE2_JOB_IDS
    ]
    pair = pair_contract_module.build_pair_contract(
        contracts[0],
        contracts[1],
        recovery,
    )
    return contracts[0], contracts[1], pair, recovery


def test_insert_body_binds_c4_8_but_preserves_frozen_inner_contract(
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, reference, _pair, _recovery = pair_inputs
    public_key = _read(STEP11_ROOT / "controller_public_key.json")
    for contract in (candidate, reference):
        body = subject.build_insert_body(
            transport_contract=contract,
            authorization={"schema": "test-authorization"},
            public_key_record=public_key,
            startup_path=STEP11_BOOTSTRAP,
            prebootstrap_path=STEP12_PREBOOTSTRAP,
            base_prebootstrap_path=STEP11_PREBOOTSTRAP,
        )
        assert body["name"] == contract["metadata_binding"]["instance_name"]
        assert body["machineType"].endswith("/machineTypes/c4-standard-8")
        assert body["networkInterfaces"][0]["accessConfigs"] == []
        assert body["scheduling"]["provisioningModel"] == "SPOT"
        metadata = {
            row["key"]: row["value"] for row in body["metadata"]["items"]
        }
        embedded = json.loads(metadata[subject.TRANSPORT_METADATA_KEY])
        assert (
            embedded["direct_stage_identity"]["inputs"]["machine_type"]
            == "c4-standard-16"
        )
        assert (
            embedded["metadata_binding"]["worker_invocation"][
                "direct_runner_environment"
            ]["RAYON_NUM_THREADS"]
            == "16"
        )
        assert hashlib.sha256(
            metadata[subject.BASE_PREBOOTSTRAP_METADATA_KEY].encode("utf-8")
        ).hexdigest() == _load_prebootstrap().BASE_SHA256


def test_prebootstrap_pins_base_and_only_accepts_exact_stage2_pair(
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, reference, _pair, _recovery = pair_inputs
    prebootstrap = _load_prebootstrap()
    base = STEP11_PREBOOTSTRAP.read_bytes()
    assert hashlib.sha256(base).hexdigest() == prebootstrap.BASE_SHA256
    assert prebootstrap._job_from_contract(candidate) == "candidate-shard-01"
    assert prebootstrap._job_from_contract(reference) == "reference-shard-01"

    mutations: list[tuple[dict[str, Any], str]] = []
    wrong_role = copy.deepcopy(candidate)
    wrong_role["metadata_binding"]["source_role"] = "reference"
    mutations.append((wrong_role, "source role"))
    attempt1 = copy.deepcopy(candidate)
    attempt1["metadata_binding"]["attempt_index"] = 1
    mutations.append((attempt1, "attempt"))
    wrong_jobs = copy.deepcopy(candidate)
    wrong_jobs["adapter_preview"]["selected_job_ids"] = [
        "candidate-shard-01"
    ]
    mutations.append((wrong_jobs, "job set"))
    legacy_done = copy.deepcopy(candidate)
    legacy_done["remote_layout"]["jobs"][0]["done_uri"] = (
        candidate["adapter_preview"]["jobs"][0]["done_uri"]
    )
    mutations.append((legacy_done, "legacy namespace"))
    for changed, _label in mutations:
        with pytest.raises(ValueError, match="frozen Stage2"):
            prebootstrap._job_from_contract(changed)


def test_prebootstrap_rejects_any_base_source_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prebootstrap = _load_prebootstrap()
    original = STEP11_PREBOOTSTRAP.read_bytes()
    monkeypatch.setattr(prebootstrap, "_metadata_get", lambda _key: original)
    frozen = prebootstrap._load_frozen_base()
    assert frozen.STAGE_ID == "stage1_lifecycle_one_candidate_vm"
    assert frozen.JOB_ID == "candidate-shard-00"

    monkeypatch.setattr(
        prebootstrap,
        "_metadata_get",
        lambda _key: original + b"\n",
    )
    with pytest.raises(ValueError, match="source changed"):
        prebootstrap._load_frozen_base()


def test_done_monitor_uses_direct_v1_and_accepts_terminal_late_publication(
    monkeypatch: pytest.MonkeyPatch,
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, _reference, _pair, _recovery = pair_inputs
    direct_jobs = candidate["remote_layout"]["jobs"]
    preview = candidate["adapter_preview"]
    subject._bind_instance_names(INSTANCE_NAMES)
    reads: dict[str, int] = {}

    def fake_read_done_once(*, client: Any, done_uri: str) -> Any:
        del client
        reads[done_uri] = reads.get(done_uri, 0) + 1
        job_id = next(
            row["job_id"] for row in direct_jobs if row["done_uri"] == done_uri
        )
        if job_id == "candidate-shard-01" and reads[done_uri] == 1:
            return None
        return (
            {"uri": done_uri, "generation": 1},
            {"job_id": job_id},
        )

    provider_calls: list[str] = []

    def fake_provider_get(_client: Any, name: str) -> Any:
        provider_calls.append(name)
        return controller.HttpResponse(404, {}, b""), None

    monkeypatch.setattr(subject, "_read_done_once", fake_read_done_once)
    monkeypatch.setattr(subject, "_provider_get", fake_provider_get)
    monkeypatch.setattr(
        subject.adapter,
        "build_receive",
        lambda _preview, *, done_records: {
            "job_ids": [row["job_id"] for row in done_records]
        },
    )
    records, receive = subject.wait_for_pair_done(
        collector_client=object(),
        instance_client=object(),
        preview=preview,
        direct_jobs=direct_jobs,
        instance_names=INSTANCE_NAMES,
        timeout_seconds=5,
        now=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert [row["uri"] for row in records] == [
        row["done_uri"] for row in direct_jobs
    ]
    assert receive["job_ids"] == list(plan.STAGE2_JOB_IDS)
    assert provider_calls == [INSTANCE_NAMES[0]]
    assert reads[direct_jobs[0]["done_uri"]] == 2
    assert all(
        subject.DIRECT_RESULT_MARKER in row["done_uri"] for row in direct_jobs
    )


def test_done_monitor_rejects_legacy_or_duplicate_result_namespace(
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, _reference, _pair, _recovery = pair_inputs
    subject._bind_instance_names(INSTANCE_NAMES)
    direct_jobs = copy.deepcopy(candidate["remote_layout"]["jobs"])
    direct_jobs[0]["done_uri"] = candidate["adapter_preview"]["jobs"][0][
        "done_uri"
    ]
    with pytest.raises(ValueError, match="escaped exact pair"):
        subject.wait_for_pair_done(
            collector_client=object(),
            instance_client=object(),
            preview=candidate["adapter_preview"],
            direct_jobs=direct_jobs,
            instance_names=INSTANCE_NAMES,
            timeout_seconds=1,
        )
    duplicate = copy.deepcopy(candidate["remote_layout"]["jobs"])
    duplicate[1]["done_uri"] = duplicate[0]["done_uri"]
    with pytest.raises(ValueError, match="escaped exact pair"):
        subject.wait_for_pair_done(
            collector_client=object(),
            instance_client=object(),
            preview=candidate["adapter_preview"],
            direct_jobs=duplicate,
            instance_names=INSTANCE_NAMES,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("instances", 0, "done_uri"),
            "gs://example/hu-m31-r2diag-worker-v1/wrong/DONE.envelope.json",
        ),
        (
            ("result_namespace", "authoritative_source"),
            "adapter_preview_worker_v1",
        ),
        (("authorization_boundary", "attempt1_authorized"), True),
        (("current_profile_changed",), True),
    ],
)
def test_execution_revalidates_pair_contract_runtime_bindings(
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
    path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    candidate, reference, pair, _recovery = pair_inputs
    changed = copy.deepcopy(pair)
    cursor: Any = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    unsigned = dict(changed)
    unsigned.pop("pair_contract_sha256")
    changed["pair_contract_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="execution contract changed"):
        subject._validate_pair_execution_contract(
            changed,
            transports=(candidate, reference),
        )


class _PostClient:
    def __init__(self, events: list[str], *, fail_second_insert: bool = False):
        self.events = events
        self.fail_second_insert = fail_second_insert
        self.insert_count = 0

    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> controller.HttpResponse:
        del content_type, timeout_seconds
        assert method == "POST"
        assert body is not None
        if "/setMetadata" in url:
            name = next(name for name in INSTANCE_NAMES if name in url)
            self.events.append(f"claim_cas:{name}")
            return controller.HttpResponse(202, {}, b'{"name":"claim-op"}')
        payload = json.loads(body)
        name = payload["name"]
        self.insert_count += 1
        self.events.append(f"insert:{name}")
        status = (
            500
            if self.fail_second_insert and self.insert_count == 2
            else 202
        )
        return controller.HttpResponse(status, {}, b'{"name":"insert-op"}')


class _DummySigner:
    public_record: dict[str, Any] = {}


def _patch_execution_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    monkeypatch.setattr(
        subject.controller,
        "build_controller_authorization",
        lambda *, contract, **_kwargs: {
            "job_id": contract["metadata_binding"]["job_id"]
        },
    )
    monkeypatch.setattr(
        subject,
        "build_insert_body",
        lambda *, transport_contract, **_kwargs: {
            "name": transport_contract["metadata_binding"]["instance_name"],
            "metadata": {"items": []},
        },
    )

    def fake_wait_zone_operation(
        *,
        initial: dict[str, Any],
        expected_instance_url: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del initial
        name = expected_instance_url.rsplit("/", 1)[-1]
        events.append(f"operation:{name}")
        return {"target": expected_instance_url}

    monkeypatch.setattr(
        subject.step11_cloud,
        "wait_zone_operation",
        fake_wait_zone_operation,
    )
    provider_counts: dict[str, int] = {}

    def fake_provider_get(_client: Any, name: str) -> Any:
        provider_counts[name] = provider_counts.get(name, 0) + 1
        events.append(f"provider_get:{name}:{provider_counts[name]}")
        return controller.HttpResponse(200, {}, b"{}"), {
            "name": name,
            "id": "1" if name == INSTANCE_NAMES[0] else "2",
        }

    monkeypatch.setattr(subject, "_provider_get", fake_provider_get)

    def fake_validate_provider(
        _value: Any, *, expected_name: str, **_kwargs: Any
    ) -> dict[str, Any]:
        events.append(f"provider_validate:{expected_name}")
        return {
            "name": expected_name,
            "instance_id": (
                "1" if expected_name == INSTANCE_NAMES[0] else "2"
            ),
        }

    monkeypatch.setattr(subject, "_validate_provider", fake_validate_provider)

    def fake_claim(*, contract: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        job_id = contract["metadata_binding"]["job_id"]
        events.append(f"claim_build:{job_id}")
        return {"job_id": job_id}

    monkeypatch.setattr(subject.controller, "build_worker_claim", fake_claim)
    monkeypatch.setattr(
        subject.step11_cloud,
        "build_claim_cas_body",
        lambda **_kwargs: {"fingerprint": "test"},
    )

    def fake_validate_claimed(
        _value: Any, *, expected_name: str, **_kwargs: Any
    ) -> dict[str, Any]:
        events.append(f"claim_readback:{expected_name}")
        return {"name": expected_name, "claimed": True}

    monkeypatch.setattr(
        subject,
        "_validate_claimed_provider",
        fake_validate_claimed,
    )


def test_execute_pair_orders_both_inserts_and_claims_before_single_revoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, reference, pair, _recovery = pair_inputs
    events: list[str] = []
    _patch_execution_dependencies(monkeypatch, events)
    monkeypatch.setattr(
        subject,
        "wait_for_pair_done",
        lambda **_kwargs: (
            [{"generation": 1}, {"generation": 2}],
            {"selected_job_ids": list(plan.STAGE2_JOB_IDS)},
        ),
    )
    monkeypatch.setattr(
        subject.step11_cloud,
        "GenerationPinnedGcsBackend",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        subject.receiver,
        "materialize_and_validate_received_stage",
        lambda *_args, **_kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        subject.step11_cloud,
        "_validate_cloud_materialization",
        lambda *, value, **_kwargs: value,
    )
    monkeypatch.setattr(
        subject.step11_cloud,
        "wait_for_instance_absence",
        lambda *, instance_name, **_kwargs: {
            "instance_name": instance_name,
            "provider_get_status": 404,
        },
    )

    callback_count = 0

    def callback(value: dict[str, Any]) -> dict[str, Any]:
        nonlocal callback_count
        callback_count += 1
        events.append("post_pair_revoke")
        assert len(value["provider_instance_ids"]) == 2
        assert len(value["claim_sha256s"]) == 2
        body = {
            "schema": subject.POST_PAIR_CLAIM_RECEIPT_SCHEMA,
            "status": "launch_and_worker_actas_removed_after_both_claims",
            "instance_names": value["instance_names"],
            "claim_sha256s": value["claim_sha256s"],
            "launch_binding_removed": True,
            "worker_actas_binding_removed": True,
            "readback_verified": True,
        }
        return {**body, "receipt_sha256": subject.canonical_sha256(body)}

    package_generations = {
        row["uri"]: index + 1
        for index, row in enumerate(
            candidate["remote_layout"]["package_inventory"]["records"]
        )
    }
    receipt = subject.execute_pair_attempt0(
        execute=True,
        execution_confirmation=subject.EXECUTION_CONFIRMATION,
        client=_PostClient(events),
        collector_client=object(),
        pair_contract=pair,
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        signer=_DummySigner(),
        package_generations=package_generations,
        external_preflight_receipt_sha256="1" * 64,
        issued_unix_seconds=1,
        expires_unix_seconds=2,
        post_pair_claim_callback=callback,
        startup_path=STEP11_BOOTSTRAP,
        prebootstrap_path=STEP12_PREBOOTSTRAP,
        base_prebootstrap_path=STEP11_PREBOOTSTRAP,
        destination_root=tmp_path / "received",
        request_ids=[str(uuid.uuid4()) for _ in range(6)],
    )
    assert callback_count == 1
    first_claim = min(
        index
        for index, event in enumerate(events)
        if event.startswith("claim_build:")
    )
    assert max(
        index
        for index, event in enumerate(events)
        if event.startswith("insert:")
    ) < first_claim
    assert max(
        index
        for index, event in enumerate(events)
        if event.startswith("provider_get:") and event.endswith(":1")
    ) < first_claim
    revoke_index = events.index("post_pair_revoke")
    assert sum(event.startswith("claim_readback:") for event in events) == 2
    assert max(
        index
        for index, event in enumerate(events)
        if event.startswith("claim_readback:")
    ) < revoke_index
    assert receipt["provider_get_404_count"] == 2
    assert receipt["actual_machine_type"] == "c4-standard-8"
    assert receipt["inner_machine_type"] == "c4-standard-16"
    assert receipt["inner_rayon_threads"] == 16
    assert receipt["performance_lock_evidence"] is False
    assert receipt["training_eligible"] is False


def test_execute_pair_failure_attempts_cleanup_for_both_exact_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pair_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    candidate, reference, pair, _recovery = pair_inputs
    events: list[str] = []
    _patch_execution_dependencies(monkeypatch, events)
    cleanup_names: list[str] = []

    def fake_delete(*, instance_name: str, **_kwargs: Any) -> dict[str, Any]:
        cleanup_names.append(instance_name)
        if instance_name == INSTANCE_NAMES[0]:
            raise RuntimeError("first cleanup failed")
        return {"instance_name": instance_name, "provider_get_status": 404}

    monkeypatch.setattr(
        subject.step11_cloud,
        "delete_exact_instance",
        fake_delete,
    )
    package_generations = {
        row["uri"]: index + 1
        for index, row in enumerate(
            candidate["remote_layout"]["package_inventory"]["records"]
        )
    }
    with pytest.raises(subject.Step12PairExecutionError) as captured:
        subject.execute_pair_attempt0(
            execute=True,
            execution_confirmation=subject.EXECUTION_CONFIRMATION,
            client=_PostClient(events, fail_second_insert=True),
            collector_client=object(),
            pair_contract=pair,
            candidate_transport_contract=candidate,
            reference_transport_contract=reference,
            signer=_DummySigner(),
            package_generations=package_generations,
            external_preflight_receipt_sha256="1" * 64,
            issued_unix_seconds=1,
            expires_unix_seconds=2,
            post_pair_claim_callback=lambda _value: {},
            startup_path=STEP11_BOOTSTRAP,
            prebootstrap_path=STEP12_PREBOOTSTRAP,
            base_prebootstrap_path=STEP11_PREBOOTSTRAP,
            destination_root=tmp_path / "received",
            request_ids=[str(uuid.uuid4()) for _ in range(6)],
        )
    assert cleanup_names == INSTANCE_NAMES
    assert [row["instance_name"] for row in captured.value.cleanup] == (
        INSTANCE_NAMES
    )
    assert captured.value.cleanup[0]["cleanup_failed"] is True
