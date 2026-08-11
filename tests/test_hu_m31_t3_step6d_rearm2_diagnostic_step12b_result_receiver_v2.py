from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as worker_adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as payload_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as step11_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_alias_bridge_v2
    as alias_bridge,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_result_receiver_v2
    as subject,
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
RUN_NONCE = "a5" * 32


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def context() -> dict[str, Any]:
    signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    public = dict(signer.public_record)
    stage1 = _read(STEP11_ROOT / "transport_contract.json")
    done = _read(STEP11_ROOT / "late_done_envelope.json")
    receive = worker_adapter.build_receive(
        stage1["adapter_preview"], done_records=[done]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public,
        run_nonce=RUN_NONCE,
    )
    return {
        "public": public,
        "payloads": payloads,
        "deployment": deployment,
    }


class _Writer:
    def __init__(self) -> None:
        self.values: dict[str, tuple[dict[str, Any], bytes]] = {}

    def conditional_create(
        self, *, uri: str, content: bytes
    ) -> dict[str, Any]:
        generation = 1_900_400_000_000_000 + len(self.values)
        readback = {
            "uri": uri,
            "generation": generation,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "created": True,
        }
        self.values[uri] = (readback, content)
        return readback


class _Store:
    def __init__(self, writer: _Writer) -> None:
        self.writer = writer

    @staticmethod
    def _record(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "uri": value["uri"],
            "generation": value["generation"],
            "metageneration": 1,
            "bytes": value["bytes"],
            "sha256": value["sha256"],
            "crc32c": "AAAAAA==",
            "etag": f"etag-{value['generation']}",
        }

    def list_prefix(self, *, prefix: str) -> list[dict[str, Any]]:
        return [
            self._record(readback)
            for uri, (readback, _) in self.writer.values.items()
            if uri.startswith(prefix.rstrip("/") + "/")
        ]

    def read_bytes(self, *, uri: str, generation: int) -> bytes:
        readback, raw = self.writer.values[uri]
        assert readback["generation"] == generation
        return raw

    def read_current(
        self, *, uri: str, allow_missing: bool = False
    ) -> tuple[dict[str, Any], bytes] | None:
        value = self.writer.values.get(uri)
        if value is None:
            assert allow_missing
            return None
        readback, raw = value
        return self._record(readback), raw


def _published(
    context: dict[str, Any], tmp_path: Path
) -> tuple[_Writer, str]:
    deployment = context["deployment"]
    job_id = deployment["selected_job_ids"][0]
    binding = deployment["payload_binding"]["job_bindings"][0]
    output = tmp_path / "worker-output"
    for relative in binding["tree_paths"]:
        path = output.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                {"path": relative},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    writer = _Writer()
    receipt = alias_bridge.publish_direct_v2_output(
        deployment_contract=deployment,
        execution_job_id=job_id,
        payload_contract=context["payloads"][0],
        output_dir=output,
        writer=writer,
    )
    assert receipt["result_object_count"] == 44
    return writer, job_id


def test_receive_role_verifies_exact_44_and_materializes(
    context: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, job_id = _published(context, tmp_path)
    validated: list[Path] = []
    monkeypatch.setattr(
        subject.runner,
        "validate_completed_output",
        lambda path: validated.append(Path(path)),
    )
    destination = tmp_path / "received"
    receipt = subject.receive_role(
        deployment_contract=context["deployment"],
        candidate_payload_contract=context["payloads"][0],
        reference_payload_contract=context["payloads"][1],
        controller_public_key_record=context["public"],
        run_nonce=RUN_NONCE,
        execution_job_id=job_id,
        store=_Store(writer),
        destination=destination,
    )
    assert receipt["result_object_count"] == 44
    assert receipt["generation_pinned_read_count"] == 44
    assert receipt["runner_validate_completed_output_performed"] is True
    assert validated == [destination.resolve()]
    assert len([path for path in destination.rglob("*") if path.is_file()]) == 23


def test_done_tamper_fails_before_materialization(
    context: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, job_id = _published(context, tmp_path)
    done_uri = context["deployment"]["remote_layout"]["jobs"][0][
        "done_uri"
    ]
    readback, raw = writer.values[done_uri]
    changed = json.loads(raw)
    changed["old_result_write_count"] = 1
    changed_raw = alias_bridge.canonical_bytes(changed)
    writer.values[done_uri] = (
        {
            **readback,
            "bytes": len(changed_raw),
            "sha256": hashlib.sha256(changed_raw).hexdigest(),
        },
        changed_raw,
    )
    monkeypatch.setattr(
        subject.runner, "validate_completed_output", lambda _: None
    )
    destination = tmp_path / "tampered-received"
    with pytest.raises(ValueError, match="digest|semantics"):
        subject.receive_role(
            deployment_contract=context["deployment"],
            candidate_payload_contract=context["payloads"][0],
            reference_payload_contract=context["payloads"][1],
            controller_public_key_record=context["public"],
            run_nonce=RUN_NONCE,
            execution_job_id=job_id,
            store=_Store(writer),
            destination=destination,
        )
    assert not destination.exists()


class _DiagnosticObserver:
    def __init__(self, *, instance_name: str, marker: dict[str, Any]) -> None:
        self.instance_name = instance_name
        self.contents = (
            "guest booted\n"
            + subject.WORKER_FAILURE_MARKER_PREFIX
            + json.dumps(marker, sort_keys=True, separators=(",", ":"))
            + "\n"
        )

    def get_instance(self, *, instance_name: str) -> dict[str, Any]:
        assert instance_name == self.instance_name
        return {"name": instance_name, "status": "RUNNING"}

    def get_serial_port_output(
        self, *, instance_name: str, start: int = -65_536
    ) -> dict[str, Any]:
        assert instance_name == self.instance_name
        assert start == -65_536
        return {
            "instance_name": instance_name,
            "port": 1,
            "start": 0,
            "next": len(self.contents.encode("utf-8")),
            "contents": self.contents,
        }


def test_pair_receiver_recovers_sanitized_worker_failure_without_waiting(
    context: dict[str, Any], tmp_path: Path
) -> None:
    deployment = context["deployment"]
    instance_name = deployment["instances"][0]["instance_name"]
    marker = {
        "schema": subject.WORKER_FAILURE_MARKER_SCHEMA,
        "status": "worker_failed_before_done",
        "stage": "host_prerequisite_install",
        "exception_type": "CalledProcessError",
    }
    observer = _DiagnosticObserver(
        instance_name=instance_name, marker=marker
    )
    sleeps: list[float] = []
    with pytest.raises(subject.WorkerDiagnosticFailure) as raised:
        subject.wait_for_pair_and_receive(
            deployment_contract=deployment,
            candidate_payload_contract=context["payloads"][0],
            reference_payload_contract=context["payloads"][1],
            controller_public_key_record=context["public"],
            run_nonce=RUN_NONCE,
            store=_Store(_Writer()),
            instance_observer=observer,
            destination_root=tmp_path / "pair-received",
            sleep=sleeps.append,
        )
    receipt = raised.value.receipt
    assert receipt["status"] == (
        "sanitized_worker_failure_recovered_before_done"
    )
    assert receipt["failure_stage"] == "host_prerequisite_install"
    assert receipt["exception_type"] == "CalledProcessError"
    assert receipt["serial_contents_stored"] is False
    assert receipt["access_token_stored"] is False
    assert "guest booted" not in repr(receipt)
    assert sleeps == []
