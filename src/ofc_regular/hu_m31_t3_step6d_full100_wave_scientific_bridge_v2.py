"""Bridge immutable full100 wave-v2 results into the scientific merger.

The receiver is deliberately injected through a tiny protocol.  This module
does not import a cloud client or the concrete receiver.  It revalidates the
complete 440-object accepted inventory, builds a separate write-once merge
view, and delegates scientific replay to the existing Candidate02 merger.

The accepted receiver tree is never modified.  A passing receipt opens only
the one-shot performance-lock stage; quality, training, profiles, and
``current`` remain closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as launch_bundle_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import merge_hu_m31_t3_step6d_candidate02_full100 as scientific
from . import merge_hu_m31_t3_step6d_full100_received_v1 as received_v1
from . import merge_hu_m31_t3_step6d_performance_v2 as performance_v2
from . import run_hu_m31_t3_step6d_performance_v2 as runner


ACCEPTED_RESULTS_SNAPSHOT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_accepted_results_snapshot_v2"
)
MERGE_VIEW_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_scientific_merge_view_v2"
)
GATE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_scientific_gate_receipt_v2"
)
PAIR_ATOMIC_ACCEPTANCE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_pair_atomic_acceptance_v2"
)
PAIR_LAUNCH_LINEAGE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_pair_launch_lineage_audit_v2"
)
VALIDATED_LIFECYCLE_CHAIN_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_validated_lifecycle_chain_v2"
)

EXPECTED_STARTUP_SHA256 = (
    "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
)
RUN004_PRELAUNCH_TARGET = {
    "startup_sha256": (
        "e59aae26e715a1615a0d4f488e1f0c8a579bda9405d9bd1a491ff360959777ae"
    ),
    "content_payload_sha256": (
        "f16f4b3369ce8972c0154a74561cbe6a98a148616713349a773c887bbd0d634f"
    ),
    "outer_manifest_sha256": (
        "553a673bf6ecdb4b11691a4553f40c53033500d160c438b221604c56a5b78b39"
    ),
    "wave_plan_sha256": (
        "65b22f5068cb2ba5309453ae433731c489533d0e3dc25a10cc62b6943acacf91"
    ),
    "execution_identity_sha256": (
        "000ea2c657279ae18f86374ac1e4033c75ab1c2335c551966cc6254f494ec311"
    ),
}
# Frozen compatibility guard for the one authorized run005 identity.  The
# bridge remains generic for other run identities; callers cannot replace
# this pin with an input-supplied target.
RUN005_PRELAUNCH_TARGET = {
    "run_name": "regular-hu-m31-c02-f100wv2-20260722-005",
    "startup_sha256": EXPECTED_STARTUP_SHA256,
    "content_payload_sha256": (
        "0c7bf6f1d7a77ae0999f71b3382982fd1d8420e0cde9b808798236c862f45f67"
    ),
    "outer_manifest_sha256": (
        "36a83ed9fe4d836f58c6371e33ac64da456f8cc5cccd7236ce9085a62961a02b"
    ),
    "wave_plan_sha256": (
        "e4846f54eaba01953b4c90c0e1b2cb27baf0640ca578fac489378f90a0536bf5"
    ),
    "execution_identity_sha256": (
        "d81c12713084aaffbdeaeedeea65c7d263d923fd3cb5b6009698d3fa79365de6"
    ),
    "content_prefix": (
        "hu-m31-t3/full100-wave-v2/content/"
        "0c7bf6f1d7a77ae0999f71b3382982fd1d8420e0cde9b808798236c862f45f67"
    ),
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC_WHOLE_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TRANSPORT_DONE_SCHEMA = "hu_m31_t3_step6d_full100_wave_attempt_done_v2"
_SNAPSHOT_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "attempt_ledger_sha256",
        "expected_startup_sha256", "content_payload_sha256",
        "outer_manifest_sha256", "accepted_root", "receiver_receipt_path",
        "receiver_receipt_sha256", "expected_inventory",
        "expected_inventory_sha256", "observed_inventory",
        "observed_inventory_sha256", "accepted_job_count",
        "accepted_object_count", "pair_atomic_acceptance",
        "validated_lifecycle_chain", "validated_lifecycle_chain_sha256",
        "local_path_layout",
        "immutable_local_copy", "receiver_validation_complete",
        "performance_lock_authorized", "quality_pilot_authorized",
        "training_eligible", "current_profile_changed", "snapshot_sha256",
    }
)
_TRANSPORT_DONE_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "attempt_ledger_sha256", "resume_sha256",
        "observed_transition_digest", "wave_index", "job_id",
        "source_role", "attempt_id", "package_sha256", "image_digest",
        "binary_sha256", "allocation_digest", "run_contract_digest",
        "root_digest", "done_identity_sha256", "content_payload_sha256",
        "outer_manifest_sha256", "prelaunch_authorization_sha256",
        "worker_principal", "work_hand_indices", "artifact_count",
        "artifacts", "runner_done_sha256", "metadata_hidden_truth_exposed",
        "opponent_private_discards_used", "training_eligible",
        "quality_evidence", "promotion_evidence", "current_profile_changed",
    }
)
_TRANSPORT_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_MERGE_JOB_KEYS = frozenset(
    {
        "job_id", "source_role", "shard_index", "work_hand_indices",
        "accepted_attempt_id", "accepted_instance_id", "accepted_done_path",
        "accepted_done_sha256", "merge_done_path", "merge_done_sha256",
        "runner_done_sha256", "shard_manifest_sha256",
        "run_contract_digest", "package_sha256", "image_digest",
        "binary_sha256", "allocation_digest", "root_digest",
        "candidate_reference_process_isolated",
    }
)
_MERGE_VIEW_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "attempt_ledger_sha256",
        "accepted_snapshot_sha256", "accepted_root", "merge_view_root",
        "jobs", "job_count", "candidate_job_count", "reference_job_count",
        "paired_hand_count", "root_count", "run_contract_digest",
        "rng_namespace_audit", "pair_launch_lineage_audit",
        "all_jobs_distinct_instances",
        "all_candidate_reference_pairs_distinct_processes",
        "accepted_tree_modified", "write_once_materialization",
        "current_profile_changed", "manifest_sha256",
    }
)
_PAIR_LINEAGE_KEYS = frozenset(
    {
        "schema", "status", "pairs", "pair_count", "wave_lineage_count",
        "paired_hand_count", "same_attempt_lane_for_every_pair",
        "same_launch_lineage_for_every_pair", "all_pair_processes_isolated",
        "cross_lane_pair_count", "lineage_drift_pair_count",
        "current_profile_changed", "pair_launch_lineage_sha256",
    }
)
_PAIR_LINEAGE_ROW_KEYS = frozenset(
    {
        "pair_id", "wave_index", "shard_index", "work_hand_indices",
        "lifecycle_transition_index",
        "attempt_id", "candidate_job_id", "reference_job_id",
        "candidate_instance_id", "reference_instance_id",
        "candidate_launch_receipt_sha256", "reference_launch_receipt_sha256",
        "attempt_ledger_sha256", "resume_sha256",
        "observed_transition_digest", "prelaunch_authorization_sha256",
        "candidate_worker_principal", "reference_worker_principal",
        "content_payload_sha256", "outer_manifest_sha256",
        "package_sha256", "image_digest", "allocation_digest",
        "run_contract_digest", "pair_mapping_sha256",
        "wave_launch_lineage_sha256", "common_lineage_sha256",
        "validated_lifecycle_proof_sha256",
        "controller_lifecycle_receipt_sha256",
        "wave_launch_receipt_sha256",
        "actual_launch_receipt_sha256",
        "actual_launch_receipt_present",
        "receiver_transition_lifecycle_binding_sha256",
        "gce_absence_receipt_sha256", "worker_iam_cleanup_receipt_sha256",
        "candidate_runner_done_sha256",
        "reference_runner_done_sha256", "candidate_root_digest",
        "reference_root_digest", "same_attempt_lane",
        "same_launch_mapping_lineage", "same_prelaunch_authorization",
        "separate_instances_and_processes",
    }
)
_LIFECYCLE_CHAIN_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "final_attempt_ledger_sha256", "wave_proofs",
        "wave_proof_count", "wave_indices", "execution_transition_count",
        "execution_attempt_count", "execution_launch_receipt_count",
        "accepted_job_count", "accepted_launch_receipt_count",
        "all_actual_launch_receipts_revalidated",
        "all_gce_absence_receipts_revalidated",
        "all_worker_iam_cleanup_receipts_revalidated",
        "all_selected_instances_absent", "all_selected_boot_disks_absent",
        "all_worker_iam_bindings_absent", "running_instance_count",
        "current_profile_changed", "chain_sha256",
    }
)
_LIFECYCLE_PROOF_KEYS = frozenset(
    {
        "transition_index", "wave_index", "observed_transition_digest",
        "selected_attempts", "selected_attempt_count",
        "accepted_attempt_count", "failed_attempt_count",
        "exact_created_attempt_count", "exact_uncreated_attempt_count",
        "receiver_transition_lifecycle_binding_sha256",
        "receiver_gce_absence_receipt_sha256",
        "launch_attempt_ledger_sha256", "launch_resume_sha256",
        "launch_observed_transition_digest",
        "controller_validated_proof_sha256",
        "controller_lifecycle_receipt_sha256", "gce_create_receipt_sha256",
        "wave_launch_receipt_sha256", "actual_launch_receipt_sha256",
        "actual_launch_receipt_present",
        "launch_mapping_sha256", "prelaunch_authorization_sha256",
        "gce_absence_receipt_sha256", "worker_iam_cleanup_receipt_sha256",
        "gce_create_receipt_revalidated", "actual_launch_receipt_revalidated",
        "gce_absence_receipt_revalidated",
        "worker_iam_cleanup_receipt_revalidated", "all_selected_instances_absent",
        "all_selected_boot_disks_absent", "worker_iam_bindings_absent",
        "additional_create_authorized", "running_instance_count",
        "proof_sha256",
    }
)
_LIFECYCLE_ATTEMPT_KEYS = frozenset(
    {
        "job_id", "source_role", "pair_id", "peer_job_id", "attempt_id",
        "instance_id", "launch_receipt_sha256", "worker_principal",
        "exact_instance_created",
        "terminal_status", "pair_atomic_outcome", "valid_done_observed",
    }
)
_CONTROLLER_LIFECYCLE_PROOF_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_controller_lifecycle_validated_proof_v2"
)
_CONTROLLER_PROOF_KEYS = frozenset(
    {
        "schema", "status", "controller_context_sha256", "run_name",
        "execution_identity_sha256", "wave_plan_sha256", "attempt_ledger_sha256",
        "resume_plan_sha256", "wave_index", "lifecycle_event_sha256",
        "lifecycle_receipt_sha256", "launch_event_sha256", "delete_event_sha256",
        "absence_event_sha256", "worker_iam_cleanup_event_sha256",
        "launch_bundle_sha256", "gce_create_receipt", "actual_launch_receipt",
        "gce_delete_receipt", "gce_absence_receipt",
        "worker_iam_cleanup_receipt", "selected_instance_mapping",
        "gce_create_rows", "actual_launch_rows", "selected_instance_count",
        "exact_created_instance_count", "exact_uncreated_instance_count",
        "create_classification", "actual_launch_receipt_present",
        "lifecycle_attested_at_utc",
        "journal_event_count", "journal_hash_chain_valid",
        "all_producer_receipts_valid", "all_owned_instances_absent",
        "all_owned_boot_disks_absent", "worker_iam_bindings_absent",
        "additional_create_authorized", "current_profile_changed", "proof_sha256",
    }
)
_CONTROLLER_MAPPING_KEYS = frozenset(
    {
        "job_id", "source_role", "attempt_id", "instance_id",
        "artifact_prefix", "launch_receipt_sha256", "exact_instance_created",
        "provider_instance_id", "provider_boot_disk_id", "gce_spec_sha256",
        "gce_operation_id", "actual_launch_operation_id",
        "actual_launch_instance_status", "ownership_label",
        "final_instance_absent", "final_boot_disk_absent",
    }
)
_GATE_KEYS = frozenset(
    {
        "exactly_100_paired_hands", "exactly_200_roots",
        "exactly_100_first_and_100_second", "exactly_20_hands_per_profile",
        "portable_semantic_parity_fraction_one",
        "missing_or_censored_roots_zero", "first_p95_within_150_seconds",
        "first_p99_within_240_seconds", "first_max_within_240_seconds",
        "second_p95_within_5_seconds", "peak_rss_within_858993459_bytes",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "decision", "run_name",
        "execution_identity_sha256", "wave_plan_sha256",
        "attempt_ledger_sha256", "wave_plan", "attempt_ledger",
        "validated_lifecycle_chain", "prelaunch_binding",
        "validated_lifecycle_chain_sha256",
        "accepted_snapshot_sha256", "receiver_receipt_sha256",
        "expected_inventory_sha256", "observed_inventory_sha256",
        "merge_view_manifest", "merge_view_manifest_sha256",
        "scientific_merge", "scientific_merge_sha256", "performance_gate",
        "performance_gate_sha256", "run_contract_digest",
        "rng_namespace_contract_sha256", "pair_launch_lineage_sha256",
        "job_count", "paired_hand_count",
        "root_count", "candidate_reference_separate_processes",
        "one_shot_immutable", "all_gates_passed",
        "performance_candidate_frozen", "performance_lock_authorized",
        "quality_pilot_authorized", "artifact_fanout_authorized",
        "training_authorized", "current_profile_changed",
        "named_profile_added", "runtime_policy_activated", "m31_complete",
        "receipt_sha256",
    }
)


class AcceptedResultsAdapterV2(Protocol):
    """Receiver-neutral provider for a complete immutable accepted snapshot."""

    def load_accepted_results(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        validated_lifecycle_chain: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class ValidatedLifecycleChainAdapterV2(Protocol):
    """Producer-revalidated controller lifecycle evidence for all three waves."""

    def load_validated_lifecycle_chain(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class CallbackValidatedLifecycleChainAdapterV2:
    """Invoke a controller-owned replay validator; no self-seal fallback exists."""

    def __init__(
        self,
        *,
        lifecycle_sources: Sequence[Mapping[str, Any]],
        validator: Callable[
            [Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]],
            Mapping[str, Any],
        ],
    ) -> None:
        if validator is None:
            raise TypeError("controller lifecycle replay validator is required")
        self._sources = tuple(deepcopy(dict(row)) for row in lifecycle_sources)
        self._validator = validator

    def load_validated_lifecycle_chain(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        value = self._validator(
            deepcopy(dict(wave_plan)),
            deepcopy(dict(attempt_ledger)),
            deepcopy(self._sources),
        )
        if not isinstance(value, Mapping):
            raise ValueError("controller lifecycle validator returned no chain")
        return deepcopy(dict(value))


class ControllerReceiverLifecycleChainAdapterV2:
    """Replay controller producers and receiver receipts, then normalize them."""

    def __init__(
        self,
        *,
        controller_replay_callbacks: Sequence[
            Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        ],
        receiver_receipts: Sequence[Mapping[str, Any]],
        receiver_validator: Callable[
            [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
        ],
    ) -> None:
        if (
            not controller_replay_callbacks
            or len(controller_replay_callbacks) != len(receiver_receipts)
            or not callable(receiver_validator)
            or any(not callable(callback) for callback in controller_replay_callbacks)
        ):
            raise ValueError(
                "one controller replay and receiver receipt per execution is required"
            )
        self._controller_callbacks = tuple(controller_replay_callbacks)
        self._receiver_receipts = tuple(
            deepcopy(dict(receipt)) for receipt in receiver_receipts
        )
        self._receiver_validator = receiver_validator

    def load_validated_lifecycle_chain(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        controller_proofs: list[dict[str, Any]] = []
        receiver_receipts: list[dict[str, Any]] = []
        for callback, raw_receipt in zip(
            self._controller_callbacks, self._receiver_receipts, strict=True
        ):
            proof = callback(
                deepcopy(dict(wave_plan)), deepcopy(dict(attempt_ledger))
            )
            receipt = self._receiver_validator(
                deepcopy(dict(wave_plan)), deepcopy(raw_receipt)
            )
            if not isinstance(proof, Mapping) or not isinstance(receipt, Mapping):
                raise ValueError("lifecycle producer validator returned no evidence")
            controller_proofs.append(deepcopy(dict(proof)))
            receiver_receipts.append(deepcopy(dict(receipt)))
        return normalize_controller_receiver_lifecycle_chain(
            wave_plan=wave_plan,
            attempt_ledger=attempt_ledger,
            controller_proofs=controller_proofs,
            receiver_receipts=receiver_receipts,
        )


class _PinnedValidatedLifecycleChainAdapterV2:
    """Internal replay of a chain already validated in this same call."""

    def __init__(self, chain: Mapping[str, Any]) -> None:
        self._chain = deepcopy(dict(chain))

    def load_validated_lifecycle_chain(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del wave_plan, attempt_ledger
        return deepcopy(self._chain)


class StaticAcceptedResultsAdapterV2:
    """Small adapter useful for an offline receiver handoff and tests."""

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self._snapshot = deepcopy(dict(snapshot))

    def load_accepted_results(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        validated_lifecycle_chain: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del wave_plan, attempt_ledger
        if self._snapshot.get("validated_lifecycle_chain") != dict(
            validated_lifecycle_chain
        ):
            raise ValueError("static accepted snapshot lifecycle chain changed")
        return deepcopy(self._snapshot)


class ReceiverReceiptAcceptedResultsAdapterV2:
    """Thin adapter for the concrete receiver's final immutable receipt.

    ``validator`` is injectable so this bridge remains independently testable.
    When omitted, the concrete receiver validator is imported only at call
    time; the bridge itself still depends solely on ``AcceptedResultsAdapterV2``.
    """

    def __init__(
        self,
        *,
        receiver_receipt: Mapping[str, Any],
        receiver_receipt_path: str | Path,
        expected_startup_sha256: str,
        content_payload_sha256: str,
        outer_manifest_sha256: str,
        validator: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self._receipt = deepcopy(dict(receiver_receipt))
        self._receipt_path = Path(receiver_receipt_path)
        self._startup_sha = expected_startup_sha256
        self._content_sha = content_payload_sha256
        self._outer_sha = outer_manifest_sha256
        self._validator = validator

    def load_accepted_results(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        validated_lifecycle_chain: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._validator is None:
            from . import (  # pylint: disable=import-outside-toplevel
                hu_m31_t3_step6d_full100_wave_result_receiver_v2 as receiver_v2,
            )

            validated = receiver_v2.validate_receiver_receipt(
                wave_plan, self._receipt
            )
        else:
            validated = self._validator(wave_plan, self._receipt)
        if not isinstance(validated, Mapping):
            raise ValueError("receiver validator did not return a receipt")
        receipt = deepcopy(dict(validated))
        path = self._receipt_path
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != canonical_bytes(receipt)
        ):
            raise ValueError("receiver receipt file is not exact canonical evidence")
        ledger = wave_v2.validate_attempt_ledger(wave_plan, attempt_ledger)
        lifecycle_chain = validate_validated_lifecycle_chain(
            wave_plan=wave_plan,
            attempt_ledger=ledger,
            value=validated_lifecycle_chain,
        )
        expected = receipt.get("expected_artifact_inventory")
        observed = receipt.get("observed_artifact_inventory")
        materialized = receipt.get("materialized_paths")
        next_resume = receipt.get("next_resume_plan")
        wave_index = receipt.get("wave_index")
        transition_lifecycle = receipt.get("transition_lifecycle_binding")
        attempt_results = receipt.get("attempt_results")
        embedded_controller_proof = receipt.get("validated_lifecycle_proof")
        if (
            isinstance(wave_index, bool)
            or not isinstance(wave_index, int)
            or wave_index not in range(3)
            or not isinstance(transition_lifecycle, Mapping)
            or not isinstance(attempt_results, list)
            or not isinstance(embedded_controller_proof, Mapping)
        ):
            raise ValueError("receiver lifecycle evidence is missing")
        embedded_controller_body = deepcopy(dict(embedded_controller_proof))
        embedded_controller_digest = embedded_controller_body.pop(
            "proof_sha256", None
        )
        binding = deepcopy(dict(transition_lifecycle))
        binding_digest = binding.pop("binding_sha256", None)
        matching_proofs = [
            row for row in lifecycle_chain["wave_proofs"]
            if row["receiver_transition_lifecycle_binding_sha256"]
            == binding_digest
        ]
        if len(matching_proofs) != 1:
            raise ValueError("receiver final lifecycle execution is missing or duplicate")
        proof = matching_proofs[0]
        accepted_attempts = [
            row for row in attempt_results
            if isinstance(row, Mapping) and row.get("terminal_status") == "accepted"
        ]
        proof_attempts = proof["selected_attempts"]
        if (
            binding_digest != canonical_sha256(binding)
            or embedded_controller_digest
            != controller_v2.canonical_sha256(embedded_controller_body)
            or embedded_controller_digest
            != proof["controller_validated_proof_sha256"]
            or binding_digest
            != proof["receiver_transition_lifecycle_binding_sha256"]
            or binding.get("observed_transition_digest")
            != proof["observed_transition_digest"]
            or binding.get("wave_index") != wave_index
            or binding.get("wave_launch_receipt_sha256")
            != proof["wave_launch_receipt_sha256"]
            or binding.get("launch_mapping_sha256")
            != proof["launch_mapping_sha256"]
            or binding.get("accepted_launch_receipt_sha256s")
            != [row["launch_receipt_sha256"] for row in proof_attempts]
            or binding.get("lifecycle_proof_sha256")
            != proof["controller_validated_proof_sha256"]
            or binding.get("controller_lifecycle_receipt_sha256")
            != proof["controller_lifecycle_receipt_sha256"]
            or binding.get("gce_absence_receipt_sha256")
            != proof["receiver_gce_absence_receipt_sha256"]
            or binding.get("worker_iam_cleanup_receipt_sha256")
            != proof["worker_iam_cleanup_receipt_sha256"]
            or binding.get("all_selected_instances_absent") is not True
            or binding.get("all_selected_boot_disks_absent") is not True
            or binding.get("worker_iam_bindings_absent") is not True
            or [row.get("job_id") for row in accepted_attempts]
            != [row["job_id"] for row in proof_attempts]
            or [row.get("launch_receipt_sha256") for row in accepted_attempts]
            != [row["launch_receipt_sha256"] for row in proof_attempts]
        ):
            raise ValueError("receiver and controller lifecycle chains differ")
        if (
            receipt.get("status") != "all_jobs_complete_exact_inventory_accepted"
            or receipt.get("attempt_ledger") != ledger
            or not isinstance(expected, Mapping)
            or not isinstance(observed, Mapping)
            or not isinstance(materialized, list)
            or len(materialized) != 440
            or len(set(materialized)) != 440
            or set(materialized)
            != {str(row["path"]) for row in observed.get("objects", [])}
            or not isinstance(next_resume, Mapping)
            or next_resume.get("all_jobs_complete") is not True
            or receipt.get("done_is_only_worker_commit_marker") is not True
            or receipt.get("acceptance_create_only") is not True
            or receipt.get("pair_atomicity_enforced") is not True
            or receipt.get("lifecycle_proof_sha256")
            != proof["controller_validated_proof_sha256"]
            or receipt.get("gce_absence_receipt_sha256")
            != proof["receiver_gce_absence_receipt_sha256"]
            or receipt.get("worker_iam_cleanup_receipt_sha256")
            != proof["worker_iam_cleanup_receipt_sha256"]
            or receipt.get("worker_iam_bindings_absent") is not True
            or receipt.get("vm_lifecycle_mutation_performed") is not False
            or receipt.get("current_profile_changed") is not False
        ):
            raise ValueError("receiver final accepted-results handoff changed")
        return build_accepted_results_snapshot(
            wave_plan=wave_plan,
            attempt_ledger=ledger,
            accepted_root=str(receipt.get("local_destination")),
            receiver_receipt_path=path,
            expected_inventory=expected,
            observed_inventory=observed,
            validated_lifecycle_chain=validated_lifecycle_chain,
            expected_startup_sha256=self._startup_sha,
            content_payload_sha256=self._content_sha,
            outer_manifest_sha256=self._outer_sha,
        )


@dataclass(frozen=True)
class _AcceptedEvidence:
    snapshot: dict[str, Any]
    accepted_root: Path
    expected: dict[str, Any]
    observed: dict[str, Any]
    jobs: tuple[dict[str, Any], ...]
    pair_launch_lineage_audit: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonzero_sha(value: Any, label: str) -> str:
    digest = _require_sha(value, label)
    if digest == "0" * 64:
        raise ValueError(f"{label} must be nonzero")
    return digest


def _require_utc_whole_seconds(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC_WHOLE_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{label} must be RFC3339 UTC whole seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid UTC timestamp") from exc
    return value


def _exact_keys(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed")


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical LF JSON")
    return value


def _read_transport_done_canonical(path: Path, label: str) -> dict[str, Any]:
    """Read the startup-owned DONE contract, whose JSON has no trailing LF."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if (
        not isinstance(value, dict)
        or raw != controller_v2.canonical_bytes(value)
    ):
        raise ValueError(f"{label} is not canonical newline-free JSON")
    return value


def _write_file_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable merge-view file exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable merge-view file exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _safe_local_path(root: Path, object_path: str) -> Path:
    fragment = PurePosixPath(object_path)
    if (
        fragment.is_absolute()
        or not fragment.parts
        or any(part in {"", ".", ".."} for part in fragment.parts)
    ):
        raise ValueError("accepted object path is unsafe")
    target = root.joinpath(*fragment.parts)
    cursor = root
    for part in fragment.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("accepted object path contains a symlink")
    resolved = target.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("accepted object path escapes accepted root") from exc
    if not resolved.is_file():
        raise ValueError(f"accepted object is missing: {object_path}")
    return resolved


def _plan_jobs(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scientific_jobs = {
        row["job_id"]: deepcopy(dict(row))
        for row in plan["full100_plan"]["jobs"]
    }
    result: dict[str, dict[str, Any]] = {}
    for wave in plan["waves"]:
        for pair in wave["candidate_reference_pairs"]:
            for role in wave_v2.SOURCE_ROLES:
                job_id = pair[f"{role}_job_id"]
                job = scientific_jobs[job_id]
                result[job_id] = {
                    **job,
                    "wave_index": wave["wave_index"],
                    "instance_ids": deepcopy(
                        pair[f"{role}_attempt_instance_ids"]
                    ),
                }
    if set(result) != set(plan["coverage"]["job_ids"]):
        raise ValueError("wave job metadata is incomplete")
    return result


def _accepted_instances(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, tuple[str, str]]:
    latest = ledger["transitions"][-1]
    result: dict[str, tuple[str, str]] = {}
    for row in latest["attempt_history"]:
        attempts = row["attempts"]
        if not attempts or attempts[-1]["terminal_status"] != "accepted":
            raise ValueError("scientific bridge requires every job accepted")
        accepted = attempts[-1]
        result[row["job_id"]] = (
            accepted["attempt_id"], accepted["instance_id"]
        )
    if set(result) != set(plan["coverage"]["job_ids"]):
        raise ValueError("accepted attempt coverage is incomplete")
    if len({instance for _, instance in result.values()}) != 20:
        raise ValueError("accepted attempts reused a VM/process identity")
    return result


def _pair_atomic_acceptance(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind each source pair to one common retry lane and its exact mapping."""

    latest = ledger["transitions"][-1]
    histories = {
        row["job_id"]: row["attempts"]
        for row in latest["attempt_history"]
    }
    accepted = _accepted_instances(plan, ledger)
    rows: list[dict[str, Any]] = []
    covered_hands: list[int] = []
    for wave in plan["waves"]:
        for pair in wave["candidate_reference_pairs"]:
            candidate_job = pair["candidate_job_id"]
            reference_job = pair["reference_job_id"]
            candidate_attempt, candidate_instance = accepted[candidate_job]
            reference_attempt, reference_instance = accepted[reference_job]
            if candidate_attempt != reference_attempt:
                raise ValueError(
                    "candidate/reference pair crossed accepted attempt lanes"
                )
            attempt_id = candidate_attempt
            candidate_terminal = histories[candidate_job][-1]
            reference_terminal = histories[reference_job][-1]
            candidate_launch = _require_sha(
                candidate_terminal["launch_receipt_sha256"],
                "candidate launch receipt",
            )
            reference_launch = _require_sha(
                reference_terminal["launch_receipt_sha256"],
                "reference launch receipt",
            )
            if (
                candidate_instance
                != pair["candidate_attempt_instance_ids"][attempt_id]
                or reference_instance
                != pair["reference_attempt_instance_ids"][attempt_id]
                or candidate_instance == reference_instance
                or candidate_launch == reference_launch
            ):
                raise ValueError(
                    "candidate/reference pair escaped its launch mapping lineage"
                )
            work = list(pair["work_hand_indices"])
            mapping = {
                "wave_index": wave["wave_index"],
                "pair_id": pair["pair_id"],
                "shard_index": pair["shard_index"],
                "work_hand_indices": work,
                "attempt_id": attempt_id,
                "candidate_job_id": candidate_job,
                "reference_job_id": reference_job,
                "candidate_instance_id": candidate_instance,
                "reference_instance_id": reference_instance,
            }
            rows.append(
                {
                    **mapping,
                    "candidate_launch_receipt_sha256": candidate_launch,
                    "reference_launch_receipt_sha256": reference_launch,
                    "pair_mapping_sha256": canonical_sha256(mapping),
                    "same_attempt_lane": True,
                    "distinct_instances": True,
                    "distinct_launch_receipts": True,
                }
            )
            covered_hands.extend(work)
    if (
        len(rows) != 10
        or sorted(covered_hands) != list(range(100))
        or len({row["candidate_instance_id"] for row in rows}
               | {row["reference_instance_id"] for row in rows}) != 20
        or len({row["candidate_launch_receipt_sha256"] for row in rows}
               | {row["reference_launch_receipt_sha256"] for row in rows}) != 20
    ):
        raise ValueError("pair-atomic accepted coverage changed")
    body = {
        "schema": PAIR_ATOMIC_ACCEPTANCE_SCHEMA,
        "status": "all_candidate_reference_pairs_accepted_atomically",
        "pairs": rows,
        "pair_count": 10,
        "paired_hand_count": 100,
        "accepted_job_count": 20,
        "same_attempt_lane_for_every_pair": True,
        "all_instances_distinct": True,
        "all_launch_receipts_distinct": True,
        "current_profile_changed": False,
    }
    return {**body, "pair_atomic_acceptance_sha256": canonical_sha256(body)}


def _execution_transition_contexts(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Rebuild every receiver execution from the immutable ledger in order."""

    transitions = ledger["transitions"]
    contexts: list[dict[str, Any]] = []
    for transition_index in range(1, len(transitions)):
        pre_transitions = transitions[:transition_index]
        pre_ledger = wave_v2.build_attempt_ledger(
            plan,
            transitions=pre_transitions,
            consumed_transition_digests=[
                row["transition_digest"] for row in pre_transitions[:-1]
            ],
        )
        resume = wave_v2.build_resume_plan(plan, attempt_ledger=pre_ledger)
        selected = resume.get("selected_attempts")
        if (
            resume.get("all_jobs_complete") is True
            or not isinstance(selected, list)
            or not selected
        ):
            raise ValueError("ledger contains a transition after execution completed")
        post_transitions = transitions[: transition_index + 1]
        post_ledger = wave_v2.build_attempt_ledger(
            plan,
            transitions=post_transitions,
            consumed_transition_digests=[
                row["transition_digest"] for row in post_transitions[:-1]
            ],
        )
        before = {
            row["job_id"]: row["attempts"]
            for row in pre_transitions[-1]["attempt_history"]
        }
        after = {
            row["job_id"]: row["attempts"]
            for row in post_transitions[-1]["attempt_history"]
        }
        selected_by_job = {row["job_id"]: row for row in selected}
        if len(selected_by_job) != len(selected):
            raise ValueError("resume selected duplicate execution jobs")
        terminal_by_job: dict[str, dict[str, Any]] = {}
        for job_id in plan["coverage"]["job_ids"]:
            prior = before[job_id]
            current = after[job_id]
            if job_id not in selected_by_job:
                if current != prior:
                    raise ValueError("ledger transition changed an unselected job")
                continue
            expected = selected_by_job[job_id]
            if len(current) != len(prior) + 1:
                raise ValueError("ledger transition did not commit one selected attempt")
            terminal = current[-1]
            if (
                terminal.get("attempt_id") != expected["attempt_id"]
                or terminal.get("instance_id") != expected["instance_id"]
                or terminal.get("terminal_status") not in {"accepted", "failed"}
            ):
                raise ValueError("ledger terminal attempt differs from selected resume")
            _require_nonzero_sha(
                terminal.get("launch_receipt_sha256"),
                "execution attempt launch receipt",
            )
            terminal_by_job[job_id] = deepcopy(dict(terminal))
        contexts.append(
            {
                "transition_index": transition_index,
                "wave_index": resume["resume_wave_index"],
                "selected_attempts": deepcopy(selected),
                "terminal_by_job": terminal_by_job,
                "pre_attempt_ledger": pre_ledger,
                "resume_plan": resume,
                "post_attempt_ledger": post_ledger,
                "observed_transition_digest": post_transitions[-1][
                    "transition_digest"
                ],
            }
        )
    if not contexts or contexts[-1]["post_attempt_ledger"] != ledger:
        raise ValueError("lifecycle execution chain does not reach the final ledger")
    return tuple(contexts)


def _validate_controller_lifecycle_proof(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("controller lifecycle proof is missing")
    proof = deepcopy(dict(value))
    _exact_keys(proof, _CONTROLLER_PROOF_KEYS, "controller lifecycle proof")
    digest = proof.pop("proof_sha256", None)
    if digest != controller_v2.canonical_sha256(proof):
        raise ValueError("controller lifecycle proof digest changed")
    proof["proof_sha256"] = _require_nonzero_sha(
        digest, "controller lifecycle proof"
    )
    _require_utc_whole_seconds(
        proof.get("lifecycle_attested_at_utc"),
        "controller lifecycle attested time",
    )
    wave_index = execution["wave_index"]
    expected_selected = execution["selected_attempts"]
    expected_jobs = [row["job_id"] for row in expected_selected]
    selected = proof.get("selected_instance_mapping")
    actual = proof.get("actual_launch_receipt")
    create_receipt = proof.get("gce_create_receipt")
    delete_receipt = proof.get("gce_delete_receipt")
    absence_receipt = proof.get("gce_absence_receipt")
    iam_receipt = proof.get("worker_iam_cleanup_receipt")
    if (
        not isinstance(selected, list)
        or len(selected) != len(expected_jobs)
        or not all(isinstance(item, Mapping) for item in (
            create_receipt, delete_receipt, absence_receipt, iam_receipt,
        ))
        or (actual is not None and not isinstance(actual, Mapping))
    ):
        raise ValueError("controller lifecycle producer evidence is incomplete")
    create_rows = proof.get("gce_create_rows")
    if not isinstance(create_rows, list) or any(
        not isinstance(row, Mapping) for row in create_rows
    ):
        raise ValueError("controller exact-create rows are incomplete")
    create_by_job = {row.get("job_id"): row for row in create_rows}
    if (
        len(create_by_job) != len(create_rows)
        or not set(create_by_job).issubset(expected_jobs)
    ):
        raise ValueError("controller exact-create job coverage changed")
    created_count = len(create_rows)
    classification = (
        "all_selected_created"
        if created_count == len(expected_jobs)
        else "no_selected_created"
        if created_count == 0
        else "partial_selected_created"
    )
    actual_rows = proof.get("actual_launch_rows")
    if not isinstance(actual_rows, list) or any(
        not isinstance(row, Mapping) for row in actual_rows
    ):
        raise ValueError("controller actual-launch rows are malformed")
    actual_by_job = {row.get("job_id"): row for row in actual_rows}
    if (
        len(actual_by_job) != len(actual_rows)
        or (
            actual is None
            and (actual_rows or proof.get("actual_launch_receipt_present") is not False)
        )
        or (
            actual is not None
            and (
                set(actual_by_job) != set(expected_jobs)
                or created_count != len(expected_jobs)
                or proof.get("actual_launch_receipt_present") is not True
            )
        )
    ):
        raise ValueError("controller actual-launch coverage changed")
    context = {
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": execution["pre_attempt_ledger"][
            "ledger_sha256"
        ],
        "resume_plan_sha256": execution["resume_plan"]["resume_sha256"],
        "wave_index": wave_index,
    }
    if (
        proof.get("schema") != _CONTROLLER_LIFECYCLE_PROOF_SCHEMA
        or proof.get("status")
        != "controller_journal_and_all_producer_receipts_revalidated"
        or proof.get("controller_context_sha256")
        != controller_v2.canonical_sha256(context)
        or any(proof.get(key) != item for key, item in context.items())
        or proof.get("selected_instance_count") != len(expected_jobs)
        or proof.get("exact_created_instance_count") != created_count
        or proof.get("exact_uncreated_instance_count")
        != len(expected_jobs) - created_count
        or proof.get("create_classification") != classification
        or not isinstance(proof.get("journal_event_count"), int)
        or isinstance(proof.get("journal_event_count"), bool)
        or proof["journal_event_count"] <= 0
        or proof.get("journal_hash_chain_valid") is not True
        or proof.get("all_producer_receipts_valid") is not True
        or proof.get("all_owned_instances_absent") is not True
        or proof.get("all_owned_boot_disks_absent") is not True
        or proof.get("worker_iam_bindings_absent") is not True
        or proof.get("additional_create_authorized") is not False
        or proof.get("current_profile_changed") is not False
        or proof.get("gce_create_rows") != create_receipt.get("rows")
        or proof.get("actual_launch_rows")
        != ([] if actual is None else actual.get("rows"))
    ):
        raise ValueError("controller lifecycle proof authority or coverage changed")
    for field in (
        "attempt_ledger_sha256", "resume_plan_sha256", "lifecycle_event_sha256",
        "lifecycle_receipt_sha256", "launch_event_sha256", "delete_event_sha256",
        "absence_event_sha256", "worker_iam_cleanup_event_sha256",
        "launch_bundle_sha256",
    ):
        _require_nonzero_sha(proof.get(field), f"controller proof {field}")
    for label, receipt in (
        ("GCE create", create_receipt),
        ("GCE delete", delete_receipt),
        ("GCE absence", absence_receipt),
        ("worker IAM cleanup", iam_receipt),
    ):
        _require_nonzero_sha(receipt.get("receipt_sha256"), f"{label} receipt")
    if actual is not None:
        _require_nonzero_sha(actual.get("receipt_sha256"), "actual launch receipt")
        _require_nonzero_sha(
            actual.get("prelaunch_authorization_sha256"),
            "actual launch prelaunch authorization",
        )
    clean_selected: list[dict[str, Any]] = []
    provider_instances: set[str] = set()
    provider_disks: set[str] = set()
    launch_receipts: set[str] = set()
    worker_principals: set[str] = set()
    for expected, raw in zip(expected_selected, selected, strict=True):
        expected_job = expected["job_id"]
        if not isinstance(raw, Mapping):
            raise ValueError("controller selected mapping is not an object")
        row = deepcopy(dict(raw))
        create_row = create_by_job.get(expected_job)
        exact_created = create_row is not None
        worker_principal = (
            None if create_row is None else create_row.get("service_account")
        )
        _exact_keys(row, _CONTROLLER_MAPPING_KEYS, "controller selected mapping")
        if (
            row.get("job_id") != expected_job
            or any(
                row.get(field) != expected[field]
                for field in (
                    "source_role", "attempt_id", "instance_id", "artifact_prefix"
                )
            )
            or row.get("exact_instance_created") is not exact_created
            or row.get("final_instance_absent") is not True
            or row.get("final_boot_disk_absent") is not True
        ):
            raise ValueError("controller selected exact-create classification changed")
        _require_nonzero_sha(row.get("launch_receipt_sha256"), "attempt launch receipt")
        launch_receipts.add(row["launch_receipt_sha256"])
        provider_fields = (
            "provider_instance_id", "provider_boot_disk_id", "gce_spec_sha256",
            "gce_operation_id",
        )
        actual_fields = (
            "actual_launch_operation_id", "actual_launch_instance_status",
        )
        if exact_created:
            if (
                not isinstance(worker_principal, str)
                or not worker_principal
                or not isinstance(row.get("provider_instance_id"), str)
                or not row["provider_instance_id"]
                or not isinstance(row.get("provider_boot_disk_id"), str)
                or not row["provider_boot_disk_id"]
                or (
                    row.get("gce_operation_id") is not None
                    and (
                        not isinstance(row["gce_operation_id"], str)
                        or not row["gce_operation_id"]
                    )
                )
            ):
                raise ValueError("exact-created lifecycle provider lineage is missing")
            _require_nonzero_sha(row.get("gce_spec_sha256"), "GCE instance spec")
            if actual is None:
                if any(row.get(field) is not None for field in actual_fields):
                    raise ValueError("non-actual launch carries actual provider lineage")
            elif any(
                not isinstance(row.get(field), str) or not row[field]
                for field in actual_fields
            ):
                raise ValueError("actual launch provider lineage is missing")
            provider_instances.add(row["provider_instance_id"])
            provider_disks.add(row["provider_boot_disk_id"])
            worker_principals.add(worker_principal)
        elif (
            worker_principal is not None
            or any(row.get(field) is not None for field in (*provider_fields, *actual_fields))
        ):
            raise ValueError("uncreated lifecycle attempt carries provider identity")
        clean_selected.append(row)
    if (
        len(provider_instances) != created_count
        or len(provider_disks) != created_count
        or len(launch_receipts) != len(expected_jobs)
        or len(worker_principals) != created_count
    ):
        raise ValueError("controller selected provider or worker identity was reused")
    proof["selected_instance_mapping"] = clean_selected
    return proof


def normalize_controller_receiver_lifecycle_chain(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    controller_proofs: Sequence[Mapping[str, Any]],
    receiver_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-bind every producer replay to its exact receiver transition."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    executions = _execution_transition_contexts(plan, ledger)
    if (
        len(controller_proofs) != len(executions)
        or len(receiver_receipts) != len(executions)
    ):
        raise ValueError(
            "lifecycle normalization requires every observed execution receipt"
        )
    pair_by_job: dict[str, Mapping[str, Any]] = {}
    for wave in plan["waves"]:
        for pair in wave["candidate_reference_pairs"]:
            pair_by_job[pair["candidate_job_id"]] = pair
            pair_by_job[pair["reference_job_id"]] = pair
    normalized: list[dict[str, Any]] = []
    seen_controller_proofs: set[str] = set()
    seen_receiver_bindings: set[str] = set()
    for execution, proof_raw, receipt_raw in zip(
        executions, controller_proofs, receiver_receipts, strict=True
    ):
        wave_index = execution["wave_index"]
        proof = _validate_controller_lifecycle_proof(
            proof_raw, plan=plan, execution=execution
        )
        if not isinstance(receipt_raw, Mapping):
            raise ValueError("validated receiver receipt is missing")
        receipt = deepcopy(dict(receipt_raw))
        attempts = receipt.get("attempt_results")
        lifecycle_raw = receipt.get("transition_lifecycle_binding")
        if not isinstance(attempts, list) or not isinstance(lifecycle_raw, Mapping):
            raise ValueError("receiver lifecycle commit is missing")
        lifecycle = deepcopy(dict(lifecycle_raw))
        lifecycle_digest = lifecycle.pop("binding_sha256", None)
        if lifecycle_digest != canonical_sha256(lifecycle):
            raise ValueError("receiver transition lifecycle digest changed")
        lifecycle["binding_sha256"] = _require_nonzero_sha(
            lifecycle_digest, "receiver transition lifecycle binding"
        )
        expected_selected = execution["selected_attempts"]
        expected_jobs = [row["job_id"] for row in expected_selected]
        terminal_by_job = execution["terminal_by_job"]
        controller_rows = proof["selected_instance_mapping"]
        if len(attempts) != len(expected_jobs):
            raise ValueError("receiver selected attempt cardinality changed")
        create_by_job = {
            row["job_id"]: row for row in proof["gce_create_rows"]
        }
        selected_attempts: list[dict[str, Any]] = []
        for expected, controller_row, attempt_raw in zip(
            expected_selected, controller_rows, attempts, strict=True
        ):
            expected_job = expected["job_id"]
            if not isinstance(attempt_raw, Mapping):
                raise ValueError("receiver attempt result is not an object")
            attempt = dict(attempt_raw)
            terminal = terminal_by_job[expected_job]
            pair = pair_by_job[expected_job]
            role = "candidate" if expected_job == pair["candidate_job_id"] else "reference"
            peer_role = "reference" if role == "candidate" else "candidate"
            create_row = create_by_job.get(expected_job)
            exact_created = controller_row["exact_instance_created"]
            worker_principal = (
                None if create_row is None else create_row["service_account"]
            )
            if (
                attempt.get("job_id") != expected_job
                or attempt.get("source_role") != role
                or any(
                    attempt.get(field) != expected[field]
                    for field in ("attempt_id", "instance_id")
                )
                or attempt.get("attempt_id") != controller_row["attempt_id"]
                or attempt.get("instance_id") != controller_row["instance_id"]
                or attempt.get("launch_receipt_sha256")
                != controller_row["launch_receipt_sha256"]
                or attempt.get("launch_receipt_sha256")
                != terminal["launch_receipt_sha256"]
                or attempt.get("lifecycle_proof_sha256")
                != proof["proof_sha256"]
                or attempt.get("gce_absence_receipt_sha256")
                != proof["gce_absence_receipt"]["receipt_sha256"]
                or attempt.get("terminal_status") != terminal["terminal_status"]
                or attempt.get("pair_id") != pair["pair_id"]
                or attempt.get("peer_job_id") != pair[f"{peer_role}_job_id"]
                or not isinstance(attempt.get("pair_atomic_outcome"), str)
                or not attempt["pair_atomic_outcome"]
                or attempt.get("exact_instance_created") is not exact_created
                or not isinstance(attempt.get("valid_done_observed"), bool)
                or (
                    terminal["terminal_status"] == "accepted"
                    and (
                        exact_created is not True
                        or not isinstance(worker_principal, str)
                        or not worker_principal
                        or attempt.get("valid_done_observed") is not True
                        or attempt.get("pair_atomic_outcome")
                        != "accepted_both_exact_created_valid_done"
                    )
                )
                or (
                    exact_created is False
                    and (
                        worker_principal is not None
                        or terminal["terminal_status"] != "failed"
                        or attempt.get("valid_done_observed") is not False
                    )
                )
                or (
                    exact_created is True
                    and (
                        not isinstance(worker_principal, str)
                        or not worker_principal
                    )
                )
            ):
                raise ValueError("receiver attempt differs from execution/controller proof")
            selected_attempts.append(
                {
                    "job_id": expected_job,
                    "source_role": role,
                    "pair_id": pair["pair_id"],
                    "peer_job_id": pair[f"{peer_role}_job_id"],
                    "attempt_id": controller_row["attempt_id"],
                    "instance_id": controller_row["instance_id"],
                    "launch_receipt_sha256": controller_row[
                        "launch_receipt_sha256"
                    ],
                    "worker_principal": worker_principal,
                    "exact_instance_created": exact_created,
                    "terminal_status": terminal["terminal_status"],
                    "pair_atomic_outcome": attempt["pair_atomic_outcome"],
                    "valid_done_observed": attempt["valid_done_observed"],
                }
            )
        pair_groups: dict[str, list[dict[str, Any]]] = {}
        for attempt in selected_attempts:
            pair_groups.setdefault(attempt["pair_id"], []).append(attempt)
        if any(
            len(rows) != 2
            or len({row["terminal_status"] for row in rows}) != 1
            or len({row["pair_atomic_outcome"] for row in rows}) != 1
            for rows in pair_groups.values()
        ):
            raise ValueError("receiver execution broke pair-atomic outcome binding")
        accepted_attempts = [
            row for row in selected_attempts
            if row["terminal_status"] == "accepted"
        ]
        failed_attempts = [
            row for row in selected_attempts
            if row["terminal_status"] == "failed"
        ]
        actual = proof["actual_launch_receipt"]
        create = proof["gce_create_receipt"]
        wave_launch_receipt = create if actual is None else actual
        absence = proof["gce_absence_receipt"]
        iam = proof["worker_iam_cleanup_receipt"]
        receiver_attempt_ledger = receipt.get("attempt_ledger")
        embedded_proof = receipt.get("validated_lifecycle_proof")
        if not isinstance(receiver_attempt_ledger, Mapping):
            raise ValueError("receiver wave attempt ledger is missing")
        receiver_ledger = wave_v2.validate_attempt_ledger(
            plan, receiver_attempt_ledger
        )
        if (
            receipt.get("wave_index") != wave_index
            or receipt.get("run_name") != plan["run_name"]
            or receipt.get("execution_identity_sha256")
            != plan["execution_identity_sha256"]
            or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
            or receipt.get("previous_attempt_ledger_sha256")
            != execution["pre_attempt_ledger"]["ledger_sha256"]
            or receipt.get("input_resume_plan_sha256")
            != execution["resume_plan"]["resume_sha256"]
            or receiver_ledger != execution["post_attempt_ledger"]
            or receipt.get("pair_atomicity_enforced") is not True
            or embedded_proof != proof
            or receipt.get("lifecycle_proof_sha256") != proof["proof_sha256"]
            or receipt.get("failed_job_ids")
            != [row["job_id"] for row in failed_attempts]
            or receipt.get("accepted_job_ids")
            != [row["job_id"] for row in accepted_attempts]
            or receipt.get("worker_iam_bindings_absent") is not True
            or lifecycle.get("wave_index") != wave_index
            or lifecycle.get("observed_transition_digest")
            != execution["observed_transition_digest"]
            or lifecycle.get("accepted_launch_receipt_sha256s")
            != [row["launch_receipt_sha256"] for row in accepted_attempts]
            or lifecycle.get("lifecycle_proof_sha256")
            != proof["proof_sha256"]
            or lifecycle.get("controller_lifecycle_receipt_sha256")
            != proof["lifecycle_receipt_sha256"]
            or lifecycle.get("wave_launch_receipt_sha256")
            != wave_launch_receipt.get("receipt_sha256")
            or lifecycle.get("launch_mapping_sha256")
            != controller_v2.canonical_sha256(
                proof["selected_instance_mapping"]
            )
            or lifecycle.get("gce_absence_receipt_sha256")
            != receipt.get("gce_absence_receipt_sha256")
            or receipt.get("gce_absence_receipt_sha256")
            != absence.get("receipt_sha256")
            or lifecycle.get("worker_iam_cleanup_receipt_sha256")
            != iam.get("receipt_sha256")
            or receipt.get("worker_iam_cleanup_receipt_sha256")
            != iam.get("receipt_sha256")
            or lifecycle.get("all_selected_instances_absent") is not True
            or lifecycle.get("all_selected_boot_disks_absent") is not True
            or lifecycle.get("worker_iam_bindings_absent") is not True
        ):
            raise ValueError("receiver lifecycle commit differs from controller replay")
        if (
            proof["proof_sha256"] in seen_controller_proofs
            or lifecycle_digest in seen_receiver_bindings
        ):
            raise ValueError("duplicate lifecycle proof or receiver receipt detected")
        seen_controller_proofs.add(proof["proof_sha256"])
        seen_receiver_bindings.add(lifecycle_digest)
        proof_body = {
            "transition_index": execution["transition_index"],
            "wave_index": wave_index,
            "observed_transition_digest": lifecycle[
                "observed_transition_digest"
            ],
            "selected_attempts": selected_attempts,
            "selected_attempt_count": len(selected_attempts),
            "accepted_attempt_count": len(accepted_attempts),
            "failed_attempt_count": len(failed_attempts),
            "exact_created_attempt_count": sum(
                row["exact_instance_created"] for row in selected_attempts
            ),
            "exact_uncreated_attempt_count": sum(
                not row["exact_instance_created"] for row in selected_attempts
            ),
            "receiver_transition_lifecycle_binding_sha256": lifecycle_digest,
            "receiver_gce_absence_receipt_sha256": receipt[
                "gce_absence_receipt_sha256"
            ],
            "launch_attempt_ledger_sha256": proof["attempt_ledger_sha256"],
            "launch_resume_sha256": proof["resume_plan_sha256"],
            "launch_observed_transition_digest": execution[
                "pre_attempt_ledger"
            ]["latest_transition_digest"],
            "controller_validated_proof_sha256": proof["proof_sha256"],
            "controller_lifecycle_receipt_sha256": proof[
                "lifecycle_receipt_sha256"
            ],
            "gce_create_receipt_sha256": create["receipt_sha256"],
            "wave_launch_receipt_sha256": wave_launch_receipt[
                "receipt_sha256"
            ],
            "actual_launch_receipt_sha256": (
                None if actual is None else actual["receipt_sha256"]
            ),
            "actual_launch_receipt_present": actual is not None,
            "launch_mapping_sha256": controller_v2.canonical_sha256(
                proof["selected_instance_mapping"]
            ),
            "prelaunch_authorization_sha256": (
                None
                if actual is None
                else actual["prelaunch_authorization_sha256"]
            ),
            "gce_absence_receipt_sha256": absence["receipt_sha256"],
            "worker_iam_cleanup_receipt_sha256": iam["receipt_sha256"],
            "gce_create_receipt_revalidated": True,
            "actual_launch_receipt_revalidated": actual is not None,
            "gce_absence_receipt_revalidated": True,
            "worker_iam_cleanup_receipt_revalidated": True,
            "all_selected_instances_absent": True,
            "all_selected_boot_disks_absent": True,
            "worker_iam_bindings_absent": True,
            "additional_create_authorized": False,
            "running_instance_count": 0,
        }
        normalized.append(
            {**proof_body, "proof_sha256": canonical_sha256(proof_body)}
        )
    body = {
        "schema": VALIDATED_LIFECYCLE_CHAIN_SCHEMA,
        "status": "all_execution_transitions_controller_lifecycle_replay_validated",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "final_attempt_ledger_sha256": ledger["ledger_sha256"],
        "wave_proofs": normalized,
        "wave_proof_count": len(normalized),
        "wave_indices": [row["wave_index"] for row in normalized],
        "execution_transition_count": len(normalized),
        "execution_attempt_count": sum(
            row["selected_attempt_count"] for row in normalized
        ),
        "execution_launch_receipt_count": len(
            {
                attempt["launch_receipt_sha256"]
                for row in normalized
                for attempt in row["selected_attempts"]
            }
        ),
        "accepted_job_count": 20,
        "accepted_launch_receipt_count": 20,
        "all_actual_launch_receipts_revalidated": True,
        "all_gce_absence_receipts_revalidated": True,
        "all_worker_iam_cleanup_receipts_revalidated": True,
        "all_selected_instances_absent": True,
        "all_selected_boot_disks_absent": True,
        "all_worker_iam_bindings_absent": True,
        "running_instance_count": 0,
        "current_profile_changed": False,
    }
    return validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value={**body, "chain_sha256": canonical_sha256(body)},
    )


def validate_validated_lifecycle_chain(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify normalized output from the controller-owned replay validator."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    if not isinstance(value, Mapping):
        raise ValueError("validated lifecycle chain is missing")
    chain = deepcopy(dict(value))
    _exact_keys(chain, _LIFECYCLE_CHAIN_KEYS, "validated lifecycle chain")
    digest = chain.pop("chain_sha256", None)
    if digest != canonical_sha256(chain):
        raise ValueError("validated lifecycle chain digest changed")
    chain["chain_sha256"] = _require_nonzero_sha(
        digest, "validated lifecycle chain"
    )
    proofs = chain.get("wave_proofs")
    executions = _execution_transition_contexts(plan, ledger)
    if not isinstance(proofs, list) or len(proofs) != len(executions):
        raise ValueError(
            "validated lifecycle chain must cover every execution transition"
        )

    pair_atomic = _pair_atomic_acceptance(plan, ledger)
    pair_by_job: dict[str, Mapping[str, Any]] = {}
    for pair in pair_atomic["pairs"]:
        pair_by_job[pair["candidate_job_id"]] = pair
        pair_by_job[pair["reference_job_id"]] = pair
    final_history = {
        row["job_id"]: row["attempts"][-1]
        for row in ledger["transitions"][-1]["attempt_history"]
    }
    accepted_launches: set[str] = set()
    accepted_instances: set[str] = set()
    accepted_jobs: set[str] = set()
    execution_launches: set[str] = set()
    execution_instances: set[str] = set()
    execution_attempt_count = 0
    proof_hashes: set[str] = set()
    lifecycle_hashes: set[str] = set()
    for execution, raw in zip(executions, proofs, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("validated lifecycle execution proof is not an object")
        proof = deepcopy(dict(raw))
        _exact_keys(proof, _LIFECYCLE_PROOF_KEYS, "lifecycle execution proof")
        proof_digest = proof.pop("proof_sha256", None)
        if proof_digest != canonical_sha256(proof):
            raise ValueError("validated lifecycle execution proof digest changed")
        proof["proof_sha256"] = _require_nonzero_sha(
            proof_digest, "validated lifecycle execution proof"
        )
        wave_index = execution["wave_index"]
        expected_selected = execution["selected_attempts"]
        attempts = proof.get("selected_attempts")
        if (
            proof.get("transition_index") != execution["transition_index"]
            or proof.get("wave_index") != wave_index
            or proof.get("observed_transition_digest")
            != execution["observed_transition_digest"]
            or proof.get("launch_observed_transition_digest")
            != execution["pre_attempt_ledger"]["latest_transition_digest"]
            or proof.get("launch_attempt_ledger_sha256")
            != execution["pre_attempt_ledger"]["ledger_sha256"]
            or proof.get("launch_resume_sha256")
            != execution["resume_plan"]["resume_sha256"]
            or not isinstance(attempts, list)
            or len(attempts) != len(expected_selected)
            or proof.get("selected_attempt_count") != len(attempts)
        ):
            raise ValueError("validated lifecycle execution binding changed")
        clean_attempts: list[dict[str, Any]] = []
        accepted_count = 0
        failed_count = 0
        exact_created_count = 0
        pair_groups: dict[str, list[dict[str, Any]]] = {}
        for expected, attempt_raw in zip(
            expected_selected, attempts, strict=True
        ):
            expected_job = expected["job_id"]
            if not isinstance(attempt_raw, Mapping):
                raise ValueError("lifecycle selected attempt is not an object")
            attempt = deepcopy(dict(attempt_raw))
            _exact_keys(
                attempt, _LIFECYCLE_ATTEMPT_KEYS, "lifecycle selected attempt"
            )
            pair = pair_by_job[expected_job]
            role = "candidate" if expected_job == pair["candidate_job_id"] else "reference"
            peer_role = "reference" if role == "candidate" else "candidate"
            terminal = execution["terminal_by_job"][expected_job]
            if (
                attempt.get("job_id") != expected_job
                or attempt.get("source_role") != role
                or attempt.get("pair_id") != pair["pair_id"]
                or attempt.get("peer_job_id") != pair[f"{peer_role}_job_id"]
                or attempt.get("attempt_id") != expected["attempt_id"]
                or attempt.get("instance_id") != expected["instance_id"]
                or attempt.get("launch_receipt_sha256")
                != terminal["launch_receipt_sha256"]
                or not isinstance(attempt.get("exact_instance_created"), bool)
                or attempt.get("terminal_status") != terminal["terminal_status"]
                or not isinstance(attempt.get("pair_atomic_outcome"), str)
                or not attempt["pair_atomic_outcome"]
                or not isinstance(attempt.get("valid_done_observed"), bool)
                or (
                    terminal["terminal_status"] == "accepted"
                    and (
                        attempt["exact_instance_created"] is not True
                        or not isinstance(attempt.get("worker_principal"), str)
                        or not attempt["worker_principal"]
                        or attempt["valid_done_observed"] is not True
                        or attempt["pair_atomic_outcome"]
                        != "accepted_both_exact_created_valid_done"
                    )
                )
                or (
                    attempt["exact_instance_created"] is False
                    and (
                        attempt.get("worker_principal") is not None
                        or terminal["terminal_status"] != "failed"
                        or attempt["valid_done_observed"] is not False
                    )
                )
                or (
                    attempt["exact_instance_created"] is True
                    and (
                        not isinstance(attempt.get("worker_principal"), str)
                        or not attempt["worker_principal"]
                    )
                )
            ):
                raise ValueError(
                    "lifecycle selected attempt differs from execution ledger mapping"
                )
            _require_nonzero_sha(
                attempt["launch_receipt_sha256"], "accepted launch receipt"
            )
            execution_launches.add(attempt["launch_receipt_sha256"])
            execution_instances.add(attempt["instance_id"])
            if terminal["terminal_status"] == "accepted":
                accepted_count += 1
                if expected_job in accepted_jobs:
                    raise ValueError("lifecycle accepted a job more than once")
                accepted_jobs.add(expected_job)
                accepted_launches.add(attempt["launch_receipt_sha256"])
                accepted_instances.add(attempt["instance_id"])
            else:
                failed_count += 1
            if attempt["exact_instance_created"]:
                exact_created_count += 1
            pair_groups.setdefault(attempt["pair_id"], []).append(attempt)
            clean_attempts.append(attempt)
        proof["selected_attempts"] = clean_attempts
        execution_attempt_count += len(clean_attempts)
        if (
            proof.get("accepted_attempt_count") != accepted_count
            or proof.get("failed_attempt_count") != failed_count
            or proof.get("exact_created_attempt_count") != exact_created_count
            or proof.get("exact_uncreated_attempt_count")
            != len(clean_attempts) - exact_created_count
            or accepted_count + failed_count != len(clean_attempts)
            or len(
                {
                    row["worker_principal"]
                    for row in clean_attempts
                    if row["exact_instance_created"]
                }
            )
            != exact_created_count
            or any(
                len(rows) != 2
                or len({row["terminal_status"] for row in rows}) != 1
                or len({row["pair_atomic_outcome"] for row in rows}) != 1
                for rows in pair_groups.values()
            )
        ):
            raise ValueError("lifecycle execution pair outcome counts changed")
        for field in (
            "receiver_transition_lifecycle_binding_sha256",
            "receiver_gce_absence_receipt_sha256",
            "launch_attempt_ledger_sha256", "launch_resume_sha256",
            "launch_observed_transition_digest",
            "controller_validated_proof_sha256",
            "controller_lifecycle_receipt_sha256",
            "gce_create_receipt_sha256", "wave_launch_receipt_sha256",
            "launch_mapping_sha256",
            "gce_absence_receipt_sha256",
            "worker_iam_cleanup_receipt_sha256",
        ):
            _require_nonzero_sha(proof.get(field), f"lifecycle {field}")
        if (
            not isinstance(proof.get("actual_launch_receipt_present"), bool)
            or (
                (proof["actual_launch_receipt_sha256"] is None)
                is not (proof["actual_launch_receipt_present"] is False)
            )
            or (
                (proof["prelaunch_authorization_sha256"] is None)
                is not (proof["actual_launch_receipt_present"] is False)
            )
            or (
                proof["actual_launch_receipt_present"]
                and (
                    _require_nonzero_sha(
                        proof["actual_launch_receipt_sha256"],
                        "lifecycle actual launch receipt",
                    )
                    != proof["actual_launch_receipt_sha256"]
                    or _require_nonzero_sha(
                        proof["prelaunch_authorization_sha256"],
                        "lifecycle prelaunch authorization",
                    )
                    != proof["prelaunch_authorization_sha256"]
                )
            )
            or proof.get("gce_create_receipt_revalidated") is not True
            or proof.get("actual_launch_receipt_revalidated")
            is not proof["actual_launch_receipt_present"]
            or proof.get("gce_absence_receipt_revalidated") is not True
            or proof.get("worker_iam_cleanup_receipt_revalidated") is not True
            or proof.get("all_selected_instances_absent") is not True
            or proof.get("all_selected_boot_disks_absent") is not True
            or proof.get("worker_iam_bindings_absent") is not True
            or proof.get("additional_create_authorized") is not False
            or proof.get("running_instance_count") != 0
        ):
            raise ValueError("lifecycle proof is not fully quiescent and revalidated")
        if (
            proof_digest in proof_hashes
            or proof["controller_lifecycle_receipt_sha256"] in lifecycle_hashes
        ):
            raise ValueError("lifecycle execution proof was duplicated")
        proof_hashes.add(proof_digest)
        lifecycle_hashes.add(proof["controller_lifecycle_receipt_sha256"])

    if (
        len(proof_hashes) != len(executions)
        or len(lifecycle_hashes) != len(executions)
        or len(execution_launches) != execution_attempt_count
        or len(execution_instances) != execution_attempt_count
        or accepted_jobs != set(plan["coverage"]["job_ids"])
        or len(accepted_launches) != 20
        or len(accepted_instances) != 20
        or any(
            final_history[job_id]["terminal_status"] != "accepted"
            for job_id in accepted_jobs
        )
        or chain.get("schema") != VALIDATED_LIFECYCLE_CHAIN_SCHEMA
        or chain.get("status")
        != "all_execution_transitions_controller_lifecycle_replay_validated"
        or chain.get("run_name") != plan["run_name"]
        or chain.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or chain.get("wave_plan_sha256") != plan["schedule_sha256"]
        or chain.get("final_attempt_ledger_sha256") != ledger["ledger_sha256"]
        or chain.get("wave_proof_count") != len(executions)
        or chain.get("wave_indices")
        != [row["wave_index"] for row in executions]
        or chain.get("execution_transition_count") != len(executions)
        or chain.get("execution_attempt_count") != execution_attempt_count
        or chain.get("execution_launch_receipt_count")
        != len(execution_launches)
        or chain.get("accepted_job_count") != 20
        or chain.get("accepted_launch_receipt_count") != 20
        or chain.get("all_actual_launch_receipts_revalidated") is not True
        or chain.get("all_gce_absence_receipts_revalidated") is not True
        or chain.get("all_worker_iam_cleanup_receipts_revalidated") is not True
        or chain.get("all_selected_instances_absent") is not True
        or chain.get("all_selected_boot_disks_absent") is not True
        or chain.get("all_worker_iam_bindings_absent") is not True
        or chain.get("running_instance_count") != 0
        or chain.get("current_profile_changed") is not False
    ):
        raise ValueError("validated lifecycle chain coverage or authority changed")
    return chain


def _enforce_prelaunch_guard(
    plan: Mapping[str, Any],
    *,
    startup_sha256: str,
    content_payload_sha256: str,
    outer_manifest_sha256: str,
) -> None:
    _require_sha(startup_sha256, "startup SHA")
    _require_sha(content_payload_sha256, "content payload SHA")
    _require_sha(outer_manifest_sha256, "outer manifest SHA")
    descriptor = science_registry.descriptor_for_wave_plan(plan)
    science_registry.resolve_startup_sha256(plan, startup_sha256)
    if descriptor.legacy_development_identity and (
        launch_bundle_v2.EXPECTED_STARTUP_SHA256 != EXPECTED_STARTUP_SHA256
        or startup_sha256 != EXPECTED_STARTUP_SHA256
    ):
        raise ValueError("accepted results startup SHA changed")
    if not descriptor.legacy_development_identity:
        return
    run004_identity_touched = (
        plan["schedule_sha256"] == RUN004_PRELAUNCH_TARGET["wave_plan_sha256"]
        or plan["execution_identity_sha256"]
        == RUN004_PRELAUNCH_TARGET["execution_identity_sha256"]
    )
    if run004_identity_touched:
        raise ValueError("superseded run004 prelaunch target cannot be merged")
    run005_identity_touched = (
        plan["run_name"] == RUN005_PRELAUNCH_TARGET["run_name"]
        or plan["schedule_sha256"]
        == RUN005_PRELAUNCH_TARGET["wave_plan_sha256"]
        or plan["execution_identity_sha256"]
        == RUN005_PRELAUNCH_TARGET["execution_identity_sha256"]
    )
    if run005_identity_touched:
        actual = {
            "run_name": plan["run_name"],
            "startup_sha256": startup_sha256,
            "content_payload_sha256": content_payload_sha256,
            "outer_manifest_sha256": outer_manifest_sha256,
            "wave_plan_sha256": plan["schedule_sha256"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "content_prefix": RUN005_PRELAUNCH_TARGET["content_prefix"],
        }
        if actual != RUN005_PRELAUNCH_TARGET:
            raise ValueError("run005 prelaunch target was partially mixed or drifted")


def build_accepted_results_snapshot(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    accepted_root: str | Path,
    receiver_receipt_path: str | Path,
    expected_inventory: Mapping[str, Any],
    observed_inventory: Mapping[str, Any],
    validated_lifecycle_chain: Mapping[str, Any],
    expected_startup_sha256: str,
    content_payload_sha256: str,
    outer_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the exact receiver-to-bridge adapter snapshot."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    expected = wave_v2.validate_artifact_inventory(
        plan, ledger, expected_inventory
    )
    observed = wave_v2.validate_observed_artifact_inventory(
        plan, ledger, observed_inventory
    )
    pair_atomic = _pair_atomic_acceptance(plan, ledger)
    lifecycle = validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value=validated_lifecycle_chain,
    )
    root = Path(accepted_root)
    receipt = Path(receiver_receipt_path)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("accepted root must be an absolute non-symlink directory")
    if not receipt.is_absolute() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("receiver receipt must be an absolute safe file")
    body = {
        "schema": ACCEPTED_RESULTS_SNAPSHOT_SCHEMA,
        "status": "complete_immutable_receiver_inventory_ready_for_science",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "expected_startup_sha256": expected_startup_sha256,
        "content_payload_sha256": content_payload_sha256,
        "outer_manifest_sha256": outer_manifest_sha256,
        "accepted_root": str(root.resolve()),
        "receiver_receipt_path": str(receipt.resolve()),
        "receiver_receipt_sha256": _sha256_file(receipt.resolve()),
        "expected_inventory": deepcopy(expected),
        "expected_inventory_sha256": expected["inventory_sha256"],
        "observed_inventory": deepcopy(observed),
        "observed_inventory_sha256": observed["inventory_sha256"],
        "accepted_job_count": 20,
        "accepted_object_count": 440,
        "pair_atomic_acceptance": pair_atomic,
        "validated_lifecycle_chain": lifecycle,
        "validated_lifecycle_chain_sha256": lifecycle["chain_sha256"],
        "local_path_layout": "accepted_root_join_object_path",
        "immutable_local_copy": True,
        "receiver_validation_complete": True,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    body["snapshot_sha256"] = canonical_sha256(body)
    return body


def _rng_namespace_audit(plan: Mapping[str, Any]) -> dict[str, Any]:
    seed = deepcopy(dict(plan["seed_contract"]))
    run_contract = plan.get("full100_plan", {}).get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("scientific run contract is missing")
    expected_seed = run_contract.get("seed_contract")
    bases = seed.get("namespace_bases")
    stride = seed.get("seed_stride")
    if (
        not isinstance(expected_seed, Mapping)
        or seed != expected_seed
        or not isinstance(bases, Mapping)
        or set(bases) != {
            "hand", "behavior", "candidate", "evaluation", "child",
            "confirmation",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in bases.values()
        )
        or isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride <= 0
    ):
        raise ValueError("RNG namespace contract drifted")
    rows = [
        base + stride * hand
        for hand in runner.CONTRACT_HAND_INDICES
        for base in bases.values()
    ]
    seed_schema = seed.get("schema")
    if seed_schema == "hu_m31_t3_step6d_candidate02_disjoint_seed_schedule_v1":
        expected_seed_set_sha256 = canonical_sha256(sorted(rows))
    elif seed_schema == (
        "hu_m31_t3_step6d_candidate02_performance_lock_seed_schedule_v4"
    ):
        expected_seed_set_sha256 = runner.contract_canonical_sha256(sorted(rows))
    else:
        raise ValueError("RNG namespace canonicalization domain is not allowlisted")
    if (
        len(rows) != 600
        or len(set(rows)) != 600
        or seed.get("seed_count") != 600
        or seed.get("namespace_count") != 6
        or seed.get("hand_count") != 100
        or seed.get("seed_set_sha256") != expected_seed_set_sha256
        or seed.get("all_values_unique") is not True
    ):
        raise ValueError("RNG namespace contains duplicate, missing, or drifted seeds")
    return {
        "contract_sha256": canonical_sha256(seed),
        "seed_set_sha256": seed["seed_set_sha256"],
        "namespace_names": list(bases),
        "namespace_count": 6,
        "hand_count": 100,
        "seed_count": 600,
        "unique_seed_count": 600,
        "duplicate_seed_count": 0,
        "missing_seed_count": 0,
        "drifted_seed_count": 0,
        "all_values_unique": True,
    }


def _validate_transport_done(
    *,
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    meta: Mapping[str, Any],
    expected_record: Mapping[str, Any],
    observed_by_path: Mapping[str, Mapping[str, Any]],
    accepted_attempt_id: str,
    content_payload_sha256: str,
    outer_manifest_sha256: str,
) -> dict[str, Any]:
    done = deepcopy(dict(value))
    _exact_keys(done, _TRANSPORT_DONE_KEYS, "transport DONE")
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("transport DONE artifacts are missing")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ValueError("transport DONE artifact is not an object")
        _exact_keys(row, _TRANSPORT_ARTIFACT_KEYS, "transport DONE artifact")
    work = list(meta["work_hand_indices"])
    expected_artifacts: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for hand, root_path, hand_path in zip(
        work,
        expected_record["root_paths"],
        expected_record["source_hand_paths"],
        strict=True,
    ):
        root = observed_by_path[root_path]
        source_hand = observed_by_path[hand_path]
        roots.append({"hand_index": hand, "sha256": root["sha256"]})
        expected_artifacts.extend(
            (
                {
                    "path": f"roots/hand_{hand:03d}.json",
                    "sha256": root["sha256"],
                    "bytes": root["bytes"],
                },
                {
                    "path": f"hands/{meta['source_role']}/hand_{hand:03d}.json",
                    "sha256": source_hand["sha256"],
                    "bytes": source_hand["bytes"],
                },
            )
        )
    root_digest = canonical_sha256(roots)
    binding = plan["runtime_binding"]
    if (
        done.get("schema") != _TRANSPORT_DONE_SCHEMA
        or done.get("status") != "complete_validated_single_job_attempt"
        or done.get("run_name") != plan["run_name"]
        or done.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or done.get("wave_plan_sha256") != plan["schedule_sha256"]
        or done.get("wave_index") != meta["wave_index"]
        or done.get("job_id") != meta["job_id"]
        or done.get("source_role") != meta["source_role"]
        or done.get("attempt_id") != accepted_attempt_id
        or done.get("package_sha256") != binding["package_sha256"]
        or done.get("image_digest") != binding["image_digest"]
        or done.get("binary_sha256")
        != binding["binary_sha256_by_role"][meta["source_role"]]
        or done.get("allocation_digest") != binding["allocation_digest"]
        or done.get("run_contract_digest") != plan["run_contract_digest"]
        or done.get("root_digest") != root_digest
        or done.get("done_identity_sha256")
        != wave_v2.expected_done_identity_sha256(
            plan,
            job_id=meta["job_id"],
            attempt_id=accepted_attempt_id,
            root_digest=root_digest,
        )
        or done.get("content_payload_sha256") != content_payload_sha256
        or done.get("outer_manifest_sha256") != outer_manifest_sha256
        or done.get("work_hand_indices") != work
        or done.get("artifact_count") != 20
        or done.get("artifacts") != expected_artifacts
        or any(
            _SHA.fullmatch(str(done.get(field))) is None
            for field in (
                "attempt_ledger_sha256", "resume_sha256",
                "observed_transition_digest", "prelaunch_authorization_sha256",
                "runner_done_sha256",
            )
        )
        or not isinstance(done.get("worker_principal"), str)
        or not done["worker_principal"]
        or done.get("metadata_hidden_truth_exposed") is not False
        or done.get("opponent_private_discards_used") is not False
        or any(
            done.get(field) is not False
            for field in (
                "training_eligible", "quality_evidence", "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("transport DONE job/attempt/runtime/RNG binding drifted")
    return done


def _pair_launch_lineage_audit(
    *,
    plan: Mapping[str, Any],
    pair_atomic: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    validated_lifecycle_chain: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove each accepted pair came from one common launch lineage."""

    by_job = {str(row["meta"]["job_id"]): row for row in jobs}
    lifecycle_by_job: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for proof in validated_lifecycle_chain["wave_proofs"]:
        for attempt in proof["selected_attempts"]:
            if attempt["terminal_status"] != "accepted":
                continue
            job_id = attempt["job_id"]
            if job_id in lifecycle_by_job:
                raise ValueError("accepted lifecycle job appeared in multiple executions")
            lifecycle_by_job[job_id] = (proof, attempt)
    if set(lifecycle_by_job) != set(plan["coverage"]["job_ids"]):
        raise ValueError("accepted lifecycle job coverage changed")
    rows: list[dict[str, Any]] = []
    common_done_fields = (
        "attempt_ledger_sha256",
        "resume_sha256",
        "observed_transition_digest",
        "prelaunch_authorization_sha256",
        "content_payload_sha256",
        "outer_manifest_sha256",
        "package_sha256",
        "image_digest",
        "allocation_digest",
        "run_contract_digest",
    )
    for pair in pair_atomic["pairs"]:
        candidate = by_job[pair["candidate_job_id"]]
        reference = by_job[pair["reference_job_id"]]
        candidate_done = candidate["transport_done"]
        reference_done = reference["transport_done"]
        lifecycle, candidate_lifecycle_attempt = lifecycle_by_job[
            pair["candidate_job_id"]
        ]
        reference_lifecycle, reference_lifecycle_attempt = lifecycle_by_job[
            pair["reference_job_id"]
        ]
        if (
            candidate["attempt_id"] != pair["attempt_id"]
            or reference["attempt_id"] != pair["attempt_id"]
            or candidate["instance_id"] != pair["candidate_instance_id"]
            or reference["instance_id"] != pair["reference_instance_id"]
            or candidate_done["wave_index"] != pair["wave_index"]
            or reference_done["wave_index"] != pair["wave_index"]
            or reference_lifecycle["proof_sha256"] != lifecycle["proof_sha256"]
            or candidate_lifecycle_attempt["attempt_id"] != pair["attempt_id"]
            or reference_lifecycle_attempt["attempt_id"] != pair["attempt_id"]
            or candidate_done["worker_principal"]
            != candidate_lifecycle_attempt["worker_principal"]
            or reference_done["worker_principal"]
            != reference_lifecycle_attempt["worker_principal"]
            or candidate_done["worker_principal"]
            == reference_done["worker_principal"]
            or any(
                candidate_done[field] != reference_done[field]
                for field in common_done_fields
            )
            or candidate_done["attempt_ledger_sha256"]
            != lifecycle["launch_attempt_ledger_sha256"]
            or candidate_done["resume_sha256"]
            != lifecycle["launch_resume_sha256"]
            or candidate_done["observed_transition_digest"]
            != lifecycle["launch_observed_transition_digest"]
            or (
                lifecycle["prelaunch_authorization_sha256"] is not None
                and candidate_done["prelaunch_authorization_sha256"]
                != lifecycle["prelaunch_authorization_sha256"]
            )
        ):
            raise ValueError(
                "candidate/reference pair launch lineage or attempt lane drifted"
            )
        wave_lineage = {
            field: candidate_done[field] for field in common_done_fields
        }
        wave_lineage.update(
            {
                "lifecycle_transition_index": lifecycle["transition_index"],
                "wave_index": pair["wave_index"],
                "attempt_id": pair["attempt_id"],
            }
        )
        common_lineage = {
            **wave_lineage,
            "pair_mapping_sha256": pair["pair_mapping_sha256"],
        }
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "lifecycle_transition_index": lifecycle["transition_index"],
                "shard_index": pair["shard_index"],
                "work_hand_indices": list(pair["work_hand_indices"]),
                "attempt_id": pair["attempt_id"],
                "candidate_job_id": pair["candidate_job_id"],
                "reference_job_id": pair["reference_job_id"],
                "candidate_instance_id": pair["candidate_instance_id"],
                "reference_instance_id": pair["reference_instance_id"],
                "candidate_launch_receipt_sha256": pair[
                    "candidate_launch_receipt_sha256"
                ],
                "reference_launch_receipt_sha256": pair[
                    "reference_launch_receipt_sha256"
                ],
                "candidate_worker_principal": candidate_done[
                    "worker_principal"
                ],
                "reference_worker_principal": reference_done[
                    "worker_principal"
                ],
                **common_lineage,
                "wave_launch_lineage_sha256": canonical_sha256(wave_lineage),
                "common_lineage_sha256": canonical_sha256(common_lineage),
                "validated_lifecycle_proof_sha256": lifecycle["proof_sha256"],
                "controller_lifecycle_receipt_sha256": lifecycle[
                    "controller_lifecycle_receipt_sha256"
                ],
                "wave_launch_receipt_sha256": lifecycle[
                    "wave_launch_receipt_sha256"
                ],
                "actual_launch_receipt_sha256": lifecycle[
                    "actual_launch_receipt_sha256"
                ],
                "actual_launch_receipt_present": lifecycle[
                    "actual_launch_receipt_present"
                ],
                "receiver_transition_lifecycle_binding_sha256": lifecycle[
                    "receiver_transition_lifecycle_binding_sha256"
                ],
                "gce_absence_receipt_sha256": lifecycle[
                    "gce_absence_receipt_sha256"
                ],
                "worker_iam_cleanup_receipt_sha256": lifecycle[
                    "worker_iam_cleanup_receipt_sha256"
                ],
                "candidate_runner_done_sha256": candidate_done[
                    "runner_done_sha256"
                ],
                "reference_runner_done_sha256": reference_done[
                    "runner_done_sha256"
                ],
                "candidate_root_digest": candidate_done["root_digest"],
                "reference_root_digest": reference_done["root_digest"],
                "same_attempt_lane": True,
                "same_launch_mapping_lineage": True,
                "same_prelaunch_authorization": True,
                "separate_instances_and_processes": True,
            }
        )
    wave_lineages: dict[int, set[str]] = {}
    wave_lifecycle_bindings: dict[int, set[tuple[str, ...]]] = {}
    for row in rows:
        wave_lineages.setdefault(row["lifecycle_transition_index"], set()).add(
            row["wave_launch_lineage_sha256"]
        )
        wave_lifecycle_bindings.setdefault(
            row["lifecycle_transition_index"], set()
        ).add(
            tuple(
                row[field]
                for field in (
                    "validated_lifecycle_proof_sha256",
                    "controller_lifecycle_receipt_sha256",
                    "wave_launch_receipt_sha256",
                    "receiver_transition_lifecycle_binding_sha256",
                    "gce_absence_receipt_sha256",
                    "worker_iam_cleanup_receipt_sha256",
                )
            )
        )
    if (
        len(rows) != 10
        or any(len(values) != 1 for values in wave_lineages.values())
        or len({next(iter(values)) for values in wave_lineages.values()})
        != len(wave_lineages)
        or set(wave_lifecycle_bindings) != set(wave_lineages)
        or any(len(values) != 1 for values in wave_lifecycle_bindings.values())
        or len({row["candidate_runner_done_sha256"] for row in rows}
               | {row["reference_runner_done_sha256"] for row in rows}) != 20
        or any(
            row["candidate_instance_id"] == row["reference_instance_id"]
            or row["candidate_launch_receipt_sha256"]
            == row["reference_launch_receipt_sha256"]
            or row["candidate_worker_principal"]
            == row["reference_worker_principal"]
            for row in rows
        )
    ):
        raise ValueError("pair launch lineage coverage or isolation changed")
    body = {
        "schema": PAIR_LAUNCH_LINEAGE_SCHEMA,
        "status": "all_pairs_share_exact_wave_launch_lineage",
        "pairs": rows,
        "pair_count": 10,
        "wave_lineage_count": len(wave_lineages),
        "paired_hand_count": 100,
        "same_attempt_lane_for_every_pair": True,
        "same_launch_lineage_for_every_pair": True,
        "all_pair_processes_isolated": True,
        "cross_lane_pair_count": 0,
        "lineage_drift_pair_count": 0,
        "current_profile_changed": False,
    }
    return {**body, "pair_launch_lineage_sha256": canonical_sha256(body)}


def validate_accepted_results_snapshot(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    validated_lifecycle_chain: Mapping[str, Any],
    value: Mapping[str, Any],
) -> _AcceptedEvidence:
    """Revalidate adapter evidence and every immutable local object byte."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    if not isinstance(value, Mapping):
        raise ValueError("accepted-results snapshot must be an object")
    snapshot = deepcopy(dict(value))
    _exact_keys(snapshot, _SNAPSHOT_KEYS, "accepted-results snapshot")
    digest = snapshot.pop("snapshot_sha256", None)
    if digest != canonical_sha256(snapshot):
        raise ValueError("accepted-results snapshot digest changed")
    snapshot["snapshot_sha256"] = _require_sha(digest, "snapshot digest")
    expected = wave_v2.validate_artifact_inventory(
        plan, ledger, snapshot.get("expected_inventory")
    )
    observed = wave_v2.validate_observed_artifact_inventory(
        plan, ledger, snapshot.get("observed_inventory")
    )
    if expected != wave_v2.expected_artifact_inventory(
        plan, attempt_ledger=ledger
    ):
        raise ValueError("receiver expected inventory is not deterministic")
    pair_atomic = _pair_atomic_acceptance(plan, ledger)
    expected_lifecycle = validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value=validated_lifecycle_chain,
    )
    embedded_lifecycle = validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value=snapshot.get("validated_lifecycle_chain"),
    )
    root = Path(str(snapshot.get("accepted_root")))
    receipt_path = Path(str(snapshot.get("receiver_receipt_path")))
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or str(root.resolve()) != snapshot.get("accepted_root")
        or not receipt_path.is_absolute()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or str(receipt_path.resolve()) != snapshot.get("receiver_receipt_path")
    ):
        raise ValueError("accepted root or receiver receipt is unsafe")
    _enforce_prelaunch_guard(
        plan,
        startup_sha256=str(snapshot.get("expected_startup_sha256")),
        content_payload_sha256=str(snapshot.get("content_payload_sha256")),
        outer_manifest_sha256=str(snapshot.get("outer_manifest_sha256")),
    )
    if (
        snapshot.get("schema") != ACCEPTED_RESULTS_SNAPSHOT_SCHEMA
        or snapshot.get("status")
        != "complete_immutable_receiver_inventory_ready_for_science"
        or snapshot.get("run_name") != plan["run_name"]
        or snapshot.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or snapshot.get("wave_plan_sha256") != plan["schedule_sha256"]
        or snapshot.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or snapshot.get("expected_inventory_sha256")
        != expected["inventory_sha256"]
        or snapshot.get("observed_inventory_sha256")
        != observed["inventory_sha256"]
        or snapshot.get("receiver_receipt_sha256")
        != _sha256_file(receipt_path.resolve())
        or snapshot.get("accepted_job_count") != 20
        or snapshot.get("accepted_object_count") != 440
        or snapshot.get("pair_atomic_acceptance") != pair_atomic
        or embedded_lifecycle != expected_lifecycle
        or snapshot.get("validated_lifecycle_chain_sha256")
        != expected_lifecycle["chain_sha256"]
        or snapshot.get("local_path_layout")
        != "accepted_root_join_object_path"
        or snapshot.get("immutable_local_copy") is not True
        or snapshot.get("receiver_validation_complete") is not True
        or any(
            snapshot.get(field) is not False
            for field in (
                "performance_lock_authorized", "quality_pilot_authorized",
                "training_eligible", "current_profile_changed",
            )
        )
    ):
        raise ValueError("accepted-results snapshot boundary changed")

    observed_by_path = {row["path"]: row for row in observed["objects"]}
    expected_paths = set(observed_by_path)
    artifact_root = root.joinpath(
        *PurePosixPath(plan["artifact_contract"]["prefix"]).parts
    )
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("accepted artifact root is missing or unsafe")
    actual_paths: set[str] = set()
    for candidate in artifact_root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("accepted artifact tree contains a symlink")
        if candidate.is_file():
            actual_paths.add(candidate.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("accepted local inventory has missing or extra paths")
    for object_path, row in observed_by_path.items():
        local = _safe_local_path(root.resolve(), object_path)
        if local.stat().st_size != row["bytes"] or _sha256_file(local) != row["sha256"]:
            raise ValueError("accepted local object bytes or SHA changed")

    meta_by_job = _plan_jobs(plan)
    accepted = _accepted_instances(plan, ledger)
    records = {row["job_id"]: row for row in expected["records"]}
    jobs: list[dict[str, Any]] = []
    runner_hashes: set[str] = set()
    for job_id in plan["coverage"]["job_ids"]:
        meta = meta_by_job[job_id]
        attempt_id, instance_id = accepted[job_id]
        record = records[job_id]
        if (
            attempt_id != record["accepted_attempt_id"]
            or instance_id != meta["instance_ids"][attempt_id]
        ):
            raise ValueError("accepted job/attempt/instance binding drifted")
        done_path = _safe_local_path(root.resolve(), record["done_path"])
        done = _validate_transport_done(
            value=_read_transport_done_canonical(
                done_path, "accepted transport DONE"
            ),
            plan=plan,
            meta=meta,
            expected_record=record,
            observed_by_path=observed_by_path,
            accepted_attempt_id=attempt_id,
            content_payload_sha256=snapshot["content_payload_sha256"],
            outer_manifest_sha256=snapshot["outer_manifest_sha256"],
        )
        if done["runner_done_sha256"] in runner_hashes:
            raise ValueError("accepted jobs duplicate a runner DONE identity")
        runner_hashes.add(done["runner_done_sha256"])
        jobs.append(
            {
                "meta": meta,
                "record": record,
                "attempt_id": attempt_id,
                "instance_id": instance_id,
                "transport_done": done,
            }
        )
    _rng_namespace_audit(plan)
    pair_lineage = _pair_launch_lineage_audit(
        plan=plan,
        pair_atomic=pair_atomic,
        jobs=jobs,
        validated_lifecycle_chain=embedded_lifecycle,
    )
    return _AcceptedEvidence(
        snapshot=snapshot,
        accepted_root=root.resolve(),
        expected=expected,
        observed=observed,
        jobs=tuple(jobs),
        pair_launch_lineage_audit=pair_lineage,
    )


def _materialize_merge_view(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    evidence: _AcceptedEvidence,
    merge_view_root: Path,
) -> tuple[dict[str, Any], tuple[Path, ...], tuple[Path, ...]]:
    target = Path(merge_view_root)
    if not target.is_absolute():
        raise ValueError("merge-view root must be absolute")
    target = target.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("scientific merge-view is write-once")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    if staging.exists():
        raise FileExistsError("scientific merge-view staging path exists")
    staging.mkdir()
    contract = runner.validate_run_contract(plan["full100_plan"]["run_contract"])
    observed = {row["path"]: row for row in evidence.observed["objects"]}
    rows: list[dict[str, Any]] = []
    candidate: list[Path] = []
    reference: list[Path] = []
    try:
        for item in evidence.jobs:
            meta = item["meta"]
            record = item["record"]
            role = meta["source_role"]
            job_dir = staging / "jobs" / meta["job_id"]
            manifest = runner.build_shard_manifest(
                run_contract=contract,
                source_role=role,
                work_hand_indices=meta["work_hand_indices"],
            )
            manifest_sha = canonical_sha256(manifest)
            if manifest_sha != meta["shard_manifest_sha256"]:
                raise ValueError("merge-view shard manifest digest drifted")
            _write_file_once(job_dir / "run_contract.json", canonical_bytes(contract))
            _write_file_once(job_dir / "shard_manifest.json", canonical_bytes(manifest))
            for object_path in [*record["root_paths"], *record["source_hand_paths"]]:
                relative = PurePosixPath(object_path).relative_to(
                    PurePosixPath(record["attempt_prefix"])
                )
                source = _safe_local_path(evidence.accepted_root, object_path)
                if _sha256_file(source) != observed[object_path]["sha256"]:
                    raise ValueError("merge-view source object changed after snapshot")
                _write_file_once(job_dir.joinpath(*relative.parts), source.read_bytes())
            runner_done = runner._build_done(
                output_dir=job_dir, shard_manifest=manifest
            )
            runner_done_sha = canonical_sha256(runner_done)
            if runner_done_sha != item["transport_done"]["runner_done_sha256"]:
                raise ValueError("reconstructed runner DONE does not match transport DONE")
            _write_file_once(job_dir / "DONE.json", canonical_bytes(runner_done))
            runner.validate_done(
                runner_done, output_dir=job_dir, shard_manifest=manifest
            )
            final_done = target / "jobs" / meta["job_id"] / "DONE.json"
            rows.append(
                {
                    "job_id": meta["job_id"],
                    "source_role": role,
                    "shard_index": meta["shard_index"],
                    "work_hand_indices": list(meta["work_hand_indices"]),
                    "accepted_attempt_id": item["attempt_id"],
                    "accepted_instance_id": item["instance_id"],
                    "accepted_done_path": record["done_path"],
                    "accepted_done_sha256": record["done_sha256"],
                    "merge_done_path": str(final_done),
                    "merge_done_sha256": runner_done_sha,
                    "runner_done_sha256": item["transport_done"]["runner_done_sha256"],
                    "shard_manifest_sha256": manifest_sha,
                    "run_contract_digest": plan["run_contract_digest"],
                    "package_sha256": record["package_sha256"],
                    "image_digest": record["image_digest"],
                    "binary_sha256": record["binary_sha256"],
                    "allocation_digest": record["allocation_digest"],
                    "root_digest": record["root_digest"],
                    "candidate_reference_process_isolated": True,
                }
            )
            (candidate if role == "candidate" else reference).append(final_done)
        for row in rows:
            _exact_keys(row, _MERGE_JOB_KEYS, "merge-view job")
        rng = _rng_namespace_audit(plan)
        manifest_body = {
            "schema": MERGE_VIEW_MANIFEST_SCHEMA,
            "status": "immutable_scientific_merge_view_complete",
            "run_name": plan["run_name"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "wave_plan_sha256": plan["schedule_sha256"],
            "attempt_ledger_sha256": ledger["ledger_sha256"],
            "accepted_snapshot_sha256": evidence.snapshot["snapshot_sha256"],
            "accepted_root": str(evidence.accepted_root),
            "merge_view_root": str(target),
            "jobs": rows,
            "job_count": 20,
            "candidate_job_count": 10,
            "reference_job_count": 10,
            "paired_hand_count": 100,
            "root_count": 200,
            "run_contract_digest": plan["run_contract_digest"],
            "rng_namespace_audit": rng,
            "pair_launch_lineage_audit": deepcopy(
                evidence.pair_launch_lineage_audit
            ),
            "all_jobs_distinct_instances": True,
            "all_candidate_reference_pairs_distinct_processes": True,
            "accepted_tree_modified": False,
            "write_once_materialization": True,
            "current_profile_changed": False,
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": canonical_sha256(manifest_body),
        }
        _exact_keys(manifest, _MERGE_VIEW_KEYS, "merge-view manifest")
        _write_file_once(staging / "MERGE_VIEW.json", canonical_bytes(manifest))
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest, tuple(candidate), tuple(reference)


def _performance_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    performance = summary.get("performance")
    integrity = summary.get("integrity")
    gates = summary.get("gates")
    if (
        not isinstance(performance, Mapping)
        or not isinstance(integrity, Mapping)
        or not isinstance(gates, Mapping)
        or set(gates) != _GATE_KEYS
    ):
        raise ValueError("scientific performance/gate schema changed")
    candidate = performance.get("candidate_by_seat")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate seat performance is missing")
    first = candidate.get("first")
    second = candidate.get("second")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ValueError("candidate seat latency summary is missing")
    roots = summary.get("root_count")
    parity_count = integrity.get("paired_root_parity_count")
    if (
        isinstance(roots, bool)
        or not isinstance(roots, int)
        or roots <= 0
        or isinstance(parity_count, bool)
        or not isinstance(parity_count, int)
    ):
        raise ValueError("portable parity counts are invalid")
    observed = {
        "portable_semantic_parity_fraction": parity_count / roots,
        "first_p95_seconds": float(first["p95_seconds"]),
        "first_p99_seconds": float(first["p99_seconds"]),
        "first_max_seconds": float(first["max_seconds"]),
        "second_p95_seconds": float(second["p95_seconds"]),
        "peak_rss_bytes": int(performance["peak_source_process_rss_bytes"]),
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in observed.values()
    ):
        raise ValueError("performance gate contains non-finite values")
    expected_performance = {
        "portable_semantic_parity_fraction_one": (
            observed["portable_semantic_parity_fraction"] == 1.0
        ),
        "first_p95_within_150_seconds": observed["first_p95_seconds"] <= 150.0,
        "first_p99_within_240_seconds": observed["first_p99_seconds"] <= 240.0,
        "first_max_within_240_seconds": observed["first_max_seconds"] <= 240.0,
        "second_p95_within_5_seconds": observed["second_p95_seconds"] <= 5.0,
        "peak_rss_within_858993459_bytes": (
            observed["peak_rss_bytes"] <= 858_993_459
        ),
    }
    if any(gates[key] is not value for key, value in expected_performance.items()):
        raise ValueError("scientific performance gate threshold drifted")
    structural = _GATE_KEYS - set(expected_performance)
    if any(not isinstance(gates[key], bool) for key in structural):
        raise ValueError("scientific structural gate is not boolean")
    if any(gates[key] is not True for key in structural):
        raise ValueError("scientific merge lost exact 100-hand/200-root structure")
    all_gates = all(gates.values())
    if summary.get("all_gates_passed") is not all_gates:
        raise ValueError("scientific all-gates aggregate drifted")
    return {
        "mode": "frozen_full_100_hand_gate",
        "thresholds": {
            "portable_semantic_parity_fraction": 1.0,
            "first_p95_seconds_max": 150.0,
            "first_p99_seconds_max": 240.0,
            "first_max_seconds_max": 240.0,
            "second_p95_seconds_max": 5.0,
            "peak_rss_bytes_max": 858_993_459,
        },
        "observed": observed,
        "gates": dict(gates),
        "all_gates_passed": all_gates,
    }


def validate_scientific_gate_receipt_value(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("scientific gate receipt must be an object")
    receipt = deepcopy(dict(value))
    # accepted_results_snapshot is a later addition: the production bridge
    # replays the snapshot by value and cannot rebuild it from the digest.
    # Receipts frozen before it exists stay readable, and when it is present it
    # must agree with the digest beside it.
    snapshot_value = receipt.get("accepted_results_snapshot")
    if snapshot_value is not None:
        if (
            not isinstance(snapshot_value, Mapping)
            or snapshot_value.get("snapshot_sha256")
            != receipt.get("accepted_snapshot_sha256")
        ):
            raise ValueError("scientific gate accepted snapshot changed")
        _exact_keys(
            {k: v for k, v in receipt.items() if k != "accepted_results_snapshot"},
            _RECEIPT_KEYS,
            "scientific gate receipt",
        )
    else:
        _exact_keys(receipt, _RECEIPT_KEYS, "scientific gate receipt")
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("scientific gate receipt digest changed")
    plan_value = receipt.get("wave_plan")
    ledger_value = receipt.get("attempt_ledger")
    lifecycle_value = receipt.get("validated_lifecycle_chain")
    if not all(
        isinstance(item, Mapping)
        for item in (plan_value, ledger_value, lifecycle_value)
    ):
        raise ValueError("scientific gate lifecycle replay evidence is missing")
    plan = wave_v2.validate_wave_plan(plan_value)
    ledger = wave_v2.validate_attempt_ledger(plan, ledger_value)
    lifecycle = validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value=lifecycle_value,
    )
    all_gates = receipt.get("all_gates_passed")
    manifest = receipt.get("merge_view_manifest")
    pair_lineage = (
        None
        if not isinstance(manifest, Mapping)
        else manifest.get("pair_launch_lineage_audit")
    )
    if (
        not isinstance(all_gates, bool)
        or not isinstance(pair_lineage, Mapping)
        or receipt.get("schema") != GATE_RECEIPT_SCHEMA
        or receipt.get("status") != ("pass" if all_gates else "no_go")
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
        or receipt.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or receipt.get("validated_lifecycle_chain_sha256")
        != lifecycle["chain_sha256"]
        or receipt.get("run_contract_digest") != plan["run_contract_digest"]
        or receipt.get("scientific_merge_sha256")
        != canonical_sha256(receipt.get("scientific_merge"))
        or receipt.get("performance_gate_sha256")
        != canonical_sha256(receipt.get("performance_gate"))
        or receipt.get("merge_view_manifest_sha256")
        != canonical_sha256(receipt.get("merge_view_manifest"))
        or receipt.get("pair_launch_lineage_sha256")
        != pair_lineage.get("pair_launch_lineage_sha256")
        or receipt.get("job_count") != 20
        or receipt.get("paired_hand_count") != 100
        or receipt.get("root_count") != 200
        or receipt.get("candidate_reference_separate_processes") is not True
        or receipt.get("one_shot_immutable") is not True
        or receipt.get("performance_candidate_frozen") is not all_gates
        or receipt.get("performance_lock_authorized") is not all_gates
        or any(
            receipt.get(field) is not False
            for field in (
                "quality_pilot_authorized", "artifact_fanout_authorized",
                "training_authorized", "current_profile_changed",
                "named_profile_added", "runtime_policy_activated", "m31_complete",
            )
        )
    ):
        raise ValueError("scientific gate receipt boundary changed")
    receipt["receipt_sha256"] = _require_sha(digest, "gate receipt digest")
    return receipt


def _validate_pair_launch_lineage_audit(
    value: Any, *, merge_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("pair launch lineage audit is missing")
    audit = deepcopy(dict(value))
    _exact_keys(audit, _PAIR_LINEAGE_KEYS, "pair launch lineage audit")
    digest = audit.pop("pair_launch_lineage_sha256", None)
    if digest != canonical_sha256(audit):
        raise ValueError("pair launch lineage audit digest changed")
    audit["pair_launch_lineage_sha256"] = _require_sha(
        digest, "pair launch lineage audit"
    )
    pairs = audit.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 10:
        raise ValueError("pair launch lineage audit cardinality changed")
    by_job = {str(row["job_id"]): row for row in merge_rows}
    seen_shards: set[int] = set()
    covered_hands: list[int] = []
    wave_lineages: dict[int, set[str]] = {}
    wave_lifecycle_bindings: dict[int, set[tuple[str, ...]]] = {}
    launch_receipts: set[str] = set()
    instances: set[str] = set()
    runner_done_hashes: set[str] = set()
    for raw in pairs:
        if not isinstance(raw, Mapping):
            raise ValueError("pair launch lineage row is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _PAIR_LINEAGE_ROW_KEYS, "pair launch lineage row")
        shard = row.get("shard_index")
        wave_index = row.get("wave_index")
        lifecycle_transition_index = row.get("lifecycle_transition_index")
        attempt = row.get("attempt_id")
        work = row.get("work_hand_indices")
        if (
            isinstance(shard, bool)
            or not isinstance(shard, int)
            or shard not in range(10)
            or shard in seen_shards
            or isinstance(wave_index, bool)
            or not isinstance(wave_index, int)
            or wave_index not in range(3)
            or isinstance(lifecycle_transition_index, bool)
            or not isinstance(lifecycle_transition_index, int)
            or lifecycle_transition_index < 1
            or attempt not in wave_v2.ATTEMPT_IDS
            or not isinstance(work, list)
            or len(work) != 10
            or len(set(work)) != 10
            or any(isinstance(hand, bool) or not isinstance(hand, int) for hand in work)
        ):
            raise ValueError("pair launch lineage partition changed")
        candidate_job = f"candidate-shard-{shard:02d}"
        reference_job = f"reference-shard-{shard:02d}"
        candidate = by_job.get(candidate_job)
        reference = by_job.get(reference_job)
        if (
            row.get("pair_id") != f"paired-shard-{shard:02d}"
            or row.get("candidate_job_id") != candidate_job
            or row.get("reference_job_id") != reference_job
            or candidate is None
            or reference is None
            or candidate.get("source_role") != "candidate"
            or reference.get("source_role") != "reference"
            or candidate.get("shard_index") != shard
            or reference.get("shard_index") != shard
            or candidate.get("work_hand_indices") != work
            or reference.get("work_hand_indices") != work
            or candidate.get("accepted_attempt_id") != attempt
            or reference.get("accepted_attempt_id") != attempt
            or candidate.get("accepted_instance_id")
            != row.get("candidate_instance_id")
            or reference.get("accepted_instance_id")
            != row.get("reference_instance_id")
            or candidate.get("runner_done_sha256")
            != row.get("candidate_runner_done_sha256")
            or reference.get("runner_done_sha256")
            != row.get("reference_runner_done_sha256")
            or candidate.get("root_digest") != row.get("candidate_root_digest")
            or reference.get("root_digest") != row.get("reference_root_digest")
        ):
            raise ValueError("pair launch lineage differs from merge-view jobs")
        mapping = {
            "wave_index": wave_index,
            "pair_id": row["pair_id"],
            "shard_index": shard,
            "work_hand_indices": work,
            "attempt_id": attempt,
            "candidate_job_id": candidate_job,
            "reference_job_id": reference_job,
            "candidate_instance_id": row["candidate_instance_id"],
            "reference_instance_id": row["reference_instance_id"],
        }
        if row.get("pair_mapping_sha256") != canonical_sha256(mapping):
            raise ValueError("pair launch mapping digest changed")
        wave_lineage = {
            field: row[field]
            for field in (
                "attempt_ledger_sha256", "resume_sha256",
                "observed_transition_digest", "prelaunch_authorization_sha256",
                "content_payload_sha256",
                "outer_manifest_sha256", "package_sha256", "image_digest",
                "allocation_digest", "run_contract_digest", "wave_index",
                "attempt_id", "lifecycle_transition_index",
            )
        }
        common = {**wave_lineage, "pair_mapping_sha256": row["pair_mapping_sha256"]}
        if row.get("wave_launch_lineage_sha256") != canonical_sha256(
            wave_lineage
        ):
            raise ValueError("wave launch lineage digest changed")
        if row.get("common_lineage_sha256") != canonical_sha256(common):
            raise ValueError("pair common launch lineage digest changed")
        for field in (
            "candidate_launch_receipt_sha256", "reference_launch_receipt_sha256",
            "attempt_ledger_sha256", "resume_sha256",
            "observed_transition_digest", "prelaunch_authorization_sha256",
            "content_payload_sha256", "outer_manifest_sha256", "package_sha256",
            "allocation_digest", "run_contract_digest", "pair_mapping_sha256",
            "wave_launch_lineage_sha256", "common_lineage_sha256",
            "validated_lifecycle_proof_sha256",
            "controller_lifecycle_receipt_sha256",
            "wave_launch_receipt_sha256",
            "receiver_transition_lifecycle_binding_sha256",
            "gce_absence_receipt_sha256", "worker_iam_cleanup_receipt_sha256",
            "candidate_runner_done_sha256",
            "reference_runner_done_sha256", "candidate_root_digest",
            "reference_root_digest",
        ):
            _require_sha(row.get(field), f"pair lineage {field}")
        if (
            not isinstance(row.get("actual_launch_receipt_present"), bool)
            or (
                (row.get("actual_launch_receipt_sha256") is None)
                is not (row["actual_launch_receipt_present"] is False)
            )
            or (
                row["actual_launch_receipt_present"]
                and _require_sha(
                    row["actual_launch_receipt_sha256"],
                    "pair lineage actual launch receipt",
                )
                != row["actual_launch_receipt_sha256"]
            )
            or not isinstance(row.get("candidate_worker_principal"), str)
            or not row["candidate_worker_principal"]
            or not isinstance(row.get("reference_worker_principal"), str)
            or not row["reference_worker_principal"]
            or row["candidate_worker_principal"]
            == row["reference_worker_principal"]
            or row.get("same_attempt_lane") is not True
            or row.get("same_launch_mapping_lineage") is not True
            or row.get("same_prelaunch_authorization") is not True
            or row.get("separate_instances_and_processes") is not True
            or row["candidate_instance_id"] == row["reference_instance_id"]
            or row["candidate_launch_receipt_sha256"]
            == row["reference_launch_receipt_sha256"]
        ):
            raise ValueError("pair launch lineage isolation changed")
        seen_shards.add(shard)
        covered_hands.extend(work)
        wave_lineages.setdefault(lifecycle_transition_index, set()).add(
            row["wave_launch_lineage_sha256"]
        )
        wave_lifecycle_bindings.setdefault(
            lifecycle_transition_index, set()
        ).add(
            tuple(
                row[field]
                for field in (
                    "validated_lifecycle_proof_sha256",
                    "controller_lifecycle_receipt_sha256",
                    "wave_launch_receipt_sha256",
                    "receiver_transition_lifecycle_binding_sha256",
                    "gce_absence_receipt_sha256",
                    "worker_iam_cleanup_receipt_sha256",
                )
            )
        )
        launch_receipts.update(
            (row["candidate_launch_receipt_sha256"],
             row["reference_launch_receipt_sha256"])
        )
        instances.update(
            (row["candidate_instance_id"], row["reference_instance_id"])
        )
        runner_done_hashes.update(
            (row["candidate_runner_done_sha256"],
             row["reference_runner_done_sha256"])
        )
    if (
        seen_shards != set(range(10))
        or sorted(covered_hands) != list(range(100))
        or any(len(values) != 1 for values in wave_lineages.values())
        or len({next(iter(values)) for values in wave_lineages.values()})
        != len(wave_lineages)
        or set(wave_lifecycle_bindings) != set(wave_lineages)
        or any(len(values) != 1 for values in wave_lifecycle_bindings.values())
        or len(launch_receipts) != 20
        or len(instances) != 20
        or len(runner_done_hashes) != 20
        or audit.get("schema") != PAIR_LAUNCH_LINEAGE_SCHEMA
        or audit.get("status") != "all_pairs_share_exact_wave_launch_lineage"
        or audit.get("pair_count") != 10
        or audit.get("wave_lineage_count") != len(wave_lineages)
        or audit.get("paired_hand_count") != 100
        or audit.get("same_attempt_lane_for_every_pair") is not True
        or audit.get("same_launch_lineage_for_every_pair") is not True
        or audit.get("all_pair_processes_isolated") is not True
        or audit.get("cross_lane_pair_count") != 0
        or audit.get("lineage_drift_pair_count") != 0
        or audit.get("current_profile_changed") is not False
    ):
        raise ValueError("pair launch lineage coverage changed")
    return audit


def _validate_merge_view_manifest_files(
    manifest_value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[Path, ...], tuple[Path, ...]]:
    if not isinstance(manifest_value, Mapping):
        raise ValueError("merge-view manifest must be an object")
    manifest = deepcopy(dict(manifest_value))
    _exact_keys(manifest, _MERGE_VIEW_KEYS, "merge-view manifest")
    digest = manifest.pop("manifest_sha256", None)
    if digest != canonical_sha256(manifest):
        raise ValueError("merge-view manifest digest changed")
    manifest["manifest_sha256"] = _require_sha(digest, "merge-view digest")
    root = Path(str(manifest.get("merge_view_root")))
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or str(root.resolve()) != manifest.get("merge_view_root")
    ):
        raise ValueError("merge-view root is missing or unsafe")
    stored = _read_canonical(root / "MERGE_VIEW.json", "stored merge-view manifest")
    if stored != manifest:
        raise ValueError("stored merge-view manifest differs from receipt")
    rows = manifest.get("jobs")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("merge-view jobs do not exactly cover twenty jobs")
    by_role: dict[str, list[Path]] = {role: [] for role in wave_v2.SOURCE_ROLES}
    hands: dict[str, list[int]] = {role: [] for role in wave_v2.SOURCE_ROLES}
    jobs: set[str] = set()
    instances: set[str] = set()
    clean_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("merge-view job is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _MERGE_JOB_KEYS, "merge-view job")
        role = row.get("source_role")
        job_id = row.get("job_id")
        instance = row.get("accepted_instance_id")
        done_path = Path(str(row.get("merge_done_path")))
        if (
            role not in by_role
            or not isinstance(job_id, str)
            or job_id in jobs
            or not isinstance(instance, str)
            or instance in instances
            or not done_path.is_absolute()
            or done_path.is_symlink()
            or not done_path.is_file()
            or _sha256_file(done_path) != row.get("merge_done_sha256")
            or row.get("merge_done_sha256") != row.get("runner_done_sha256")
            or row.get("candidate_reference_process_isolated") is not True
        ):
            raise ValueError("merge-view job/path/process binding changed")
        try:
            done_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("merge-view DONE path escapes its immutable root") from exc
        work = row.get("work_hand_indices")
        if (
            not isinstance(work, list)
            or len(work) != 10
            or len(set(work)) != 10
            or any(isinstance(hand, bool) or not isinstance(hand, int) for hand in work)
        ):
            raise ValueError("merge-view work partition changed")
        jobs.add(job_id)
        instances.add(instance)
        hands[role].extend(work)
        by_role[role].append(done_path.resolve())
        clean_rows.append(row)
    pair_lineage = _validate_pair_launch_lineage_audit(
        manifest.get("pair_launch_lineage_audit"), merge_rows=clean_rows
    )
    if (
        len(jobs) != 20
        or len(instances) != 20
        or any(len(by_role[role]) != 10 for role in by_role)
        or any(sorted(hands[role]) != list(range(100)) for role in hands)
        or manifest.get("job_count") != 20
        or manifest.get("candidate_job_count") != 10
        or manifest.get("reference_job_count") != 10
        or manifest.get("paired_hand_count") != 100
        or manifest.get("root_count") != 200
        or manifest.get("all_jobs_distinct_instances") is not True
        or manifest.get("all_candidate_reference_pairs_distinct_processes")
        is not True
        or manifest.get("accepted_tree_modified") is not False
        or manifest.get("write_once_materialization") is not True
        or manifest.get("current_profile_changed") is not False
        or manifest.get("pair_launch_lineage_audit") != pair_lineage
    ):
        raise ValueError("merge-view coverage or authority changed")
    return manifest, tuple(by_role["candidate"]), tuple(by_role["reference"])


def _validate_pair_lineage_lifecycle_binding(
    *,
    manifest: Mapping[str, Any],
    validated_lifecycle_chain: Mapping[str, Any],
) -> None:
    """Cross-bind transport lineage rows to the normalized producer proof."""

    audit = manifest.get("pair_launch_lineage_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("pair launch lineage audit is missing")
    proofs_raw = validated_lifecycle_chain.get("wave_proofs")
    if not isinstance(proofs_raw, list):
        raise ValueError("validated lifecycle wave proofs are missing")
    proofs = {
        proof.get("transition_index"): proof
        for proof in proofs_raw
        if isinstance(proof, Mapping)
    }
    if len(proofs) != len(proofs_raw):
        raise ValueError("validated lifecycle execution proof coverage changed")
    seen_pairs: set[str] = set()
    for raw in audit.get("pairs", []):
        if not isinstance(raw, Mapping):
            raise ValueError("pair lifecycle lineage row is not an object")
        row = dict(raw)
        transition_index = row.get("lifecycle_transition_index")
        proof = proofs.get(transition_index)
        if proof is None:
            raise ValueError("pair lifecycle lineage references an unknown execution")
        expected_hashes = {
            "validated_lifecycle_proof_sha256": proof["proof_sha256"],
            "controller_lifecycle_receipt_sha256": proof[
                "controller_lifecycle_receipt_sha256"
            ],
            "wave_launch_receipt_sha256": proof[
                "wave_launch_receipt_sha256"
            ],
            "actual_launch_receipt_sha256": proof[
                "actual_launch_receipt_sha256"
            ],
            "actual_launch_receipt_present": proof[
                "actual_launch_receipt_present"
            ],
            "receiver_transition_lifecycle_binding_sha256": proof[
                "receiver_transition_lifecycle_binding_sha256"
            ],
            "gce_absence_receipt_sha256": proof[
                "gce_absence_receipt_sha256"
            ],
            "worker_iam_cleanup_receipt_sha256": proof[
                "worker_iam_cleanup_receipt_sha256"
            ],
            "attempt_ledger_sha256": proof["launch_attempt_ledger_sha256"],
            "resume_sha256": proof["launch_resume_sha256"],
            "observed_transition_digest": proof[
                "launch_observed_transition_digest"
            ],
        }
        if (
            any(row.get(field) != value for field, value in expected_hashes.items())
            or (
                proof["prelaunch_authorization_sha256"] is not None
                and row.get("prelaunch_authorization_sha256")
                != proof["prelaunch_authorization_sha256"]
            )
        ):
            raise ValueError(
                "pair transport lineage differs from validated lifecycle proof"
            )
        selected = {
            attempt["job_id"]: attempt
            for attempt in proof["selected_attempts"]
        }
        candidate = selected.get(row.get("candidate_job_id"))
        reference = selected.get(row.get("reference_job_id"))
        pair_id = row.get("pair_id")
        if (
            not isinstance(pair_id, str)
            or pair_id in seen_pairs
            or candidate is None
            or reference is None
            or candidate.get("pair_id") != pair_id
            or reference.get("pair_id") != pair_id
            or candidate.get("peer_job_id") != row.get("reference_job_id")
            or reference.get("peer_job_id") != row.get("candidate_job_id")
            or candidate.get("attempt_id") != row.get("attempt_id")
            or reference.get("attempt_id") != row.get("attempt_id")
            or candidate.get("instance_id") != row.get("candidate_instance_id")
            or reference.get("instance_id") != row.get("reference_instance_id")
            or candidate.get("launch_receipt_sha256")
            != row.get("candidate_launch_receipt_sha256")
            or reference.get("launch_receipt_sha256")
            != row.get("reference_launch_receipt_sha256")
            or candidate.get("worker_principal")
            != row.get("candidate_worker_principal")
            or reference.get("worker_principal")
            != row.get("reference_worker_principal")
            or candidate.get("worker_principal")
            == reference.get("worker_principal")
            or candidate.get("terminal_status") != "accepted"
            or reference.get("terminal_status") != "accepted"
        ):
            raise ValueError(
                "pair accepted mapping differs from validated lifecycle proof"
            )
        seen_pairs.add(pair_id)
    if len(seen_pairs) != 10:
        raise ValueError("pair lifecycle lineage coverage changed")


def validate_scientific_gate_receipt(
    *, receipt_path: str | Path
) -> dict[str, Any]:
    """Replay the scientific merge from a stored write-once receipt/view."""

    target = Path(receipt_path).resolve()
    receipt = validate_scientific_gate_receipt_value(
        _read_canonical(target, "scientific gate receipt")
    )
    manifest, candidate, reference = _validate_merge_view_manifest_files(
        receipt["merge_view_manifest"]
    )
    lifecycle_chain = validate_validated_lifecycle_chain(
        wave_plan=receipt["wave_plan"],
        attempt_ledger=receipt["attempt_ledger"],
        value=receipt["validated_lifecycle_chain"],
    )
    _validate_pair_lineage_lifecycle_binding(
        manifest=manifest,
        validated_lifecycle_chain=lifecycle_chain,
    )
    if receipt["merge_view_manifest_sha256"] != canonical_sha256(manifest):
        raise ValueError("receipt merge-view hash changed")
    stored_summary = receipt["scientific_merge"]
    if not isinstance(stored_summary, Mapping):
        raise ValueError("stored scientific summary is missing")
    stored_plan = receipt["wave_plan"]
    if not isinstance(stored_plan, Mapping):
        raise ValueError("stored scientific wave plan is missing")
    # The development summary carries its own plan; the v4 summary is the inner
    # performance merge and does not, so replay from the receipt's wave plan.
    plan_value = stored_summary.get("full100_plan")
    if not isinstance(plan_value, Mapping):
        plan_value = stored_plan.get("full100_plan")
    if not isinstance(plan_value, Mapping):
        raise ValueError("stored scientific full100 plan is missing")
    descriptor = science_registry.descriptor_for_plan(plan_value)
    recomputed = _validate_merged_summary(
        _merge_for_lineage(
            descriptor=descriptor,
            plan={"full100_plan": plan_value},
            candidate_paths=candidate,
            reference_paths=reference,
        ),
        descriptor=descriptor,
        candidate_paths=candidate,
        reference_paths=reference,
    )
    if recomputed != stored_summary:
        raise ValueError("stored scientific summary differs from source replay")
    gate = _performance_gate(recomputed)
    if (
        gate != receipt["performance_gate"]
        or canonical_sha256(gate) != receipt["performance_gate_sha256"]
        or receipt["rng_namespace_contract_sha256"]
        != manifest["rng_namespace_audit"]["contract_sha256"]
        or receipt["pair_launch_lineage_sha256"]
        != manifest["pair_launch_lineage_audit"][
            "pair_launch_lineage_sha256"
        ]
        or receipt["all_gates_passed"] is not gate["all_gates_passed"]
    ):
        raise ValueError("stored gate differs from scientific source replay")
    return receipt


def _validate_merged_summary(
    merged: Mapping[str, Any],
    *,
    descriptor: Any,
    candidate_paths: Sequence[Path],
    reference_paths: Sequence[Path],
) -> dict[str, Any]:
    """Validate a merged summary and return the development-shaped one.

    The development merger emits that shape directly. The performance-lock-v4
    merger wraps it: the wrapper carries frozen lineage the development shape
    has no field for, and nests the comparable summary under its generic key.
    Each lineage validates its own wrapper with its own validator, and the
    inner summary then goes through the shared gate checks unchanged.
    """

    if descriptor.merge_generic_key is None:
        return received_v1._validate_scientific_summary(
            merged,
            candidate_done_paths=candidate_paths,
            reference_done_paths=reference_paths,
        )
    module = descriptor.merge_module()
    validator = getattr(module, descriptor.merge_validator_name, None)
    if not callable(validator):
        raise ValueError(
            f"scientific merge validator changed for {descriptor.science_kind}"
        )
    validator(deepcopy(dict(merged)))
    inner = merged.get(descriptor.merge_generic_key)
    if not isinstance(inner, Mapping):
        raise ValueError(
            f"scientific merge is missing {descriptor.merge_generic_key}"
        )
    # The wrapper validator above already bound this inner summary by digest,
    # and it came straight out of merge_performance_v2, so re-deriving it here
    # would only repeat that work. Check its identity and its source inputs.
    summary = deepcopy(dict(inner))
    if (
        set(summary) != performance_v2._SUMMARY_KEYS
        or summary.get("schema") != performance_v2.MERGE_SCHEMA
    ):
        raise ValueError("scientific inner merge summary schema changed")
    expected = {
        "candidate": {Path(path).resolve() for path in candidate_paths},
        "reference": {Path(path).resolve() for path in reference_paths},
    }
    if any(
        received_v1._summary_done_paths(summary, role) != expected[role]
        for role in ("candidate", "reference")
    ):
        raise ValueError("scientific merge source DONE inputs changed")
    return summary


def _merge_for_lineage(
    *,
    descriptor: Any,
    plan: Mapping[str, Any],
    candidate_paths: Sequence[Path],
    reference_paths: Sequence[Path],
) -> dict[str, Any]:
    """Run the scientific merger this lineage registers.

    The development merger takes the plan alone. The performance-lock-v4 merger
    also replays frozen materialization and root-seal evidence that the wave
    plan does not carry, so those are loaded from the lineage's own frozen
    paths and pinned against the digests the plan already binds.
    """

    module = descriptor.merge_module()
    merge = getattr(module, descriptor.merge_function_name)
    arguments: dict[str, Any] = {
        "candidate_done_paths": candidate_paths,
        "reference_done_paths": reference_paths,
        "plan_value": plan["full100_plan"],
    }
    if descriptor.merge_lineage_path_names:
        plan_module = descriptor.plan_module()
        for name in descriptor.merge_lineage_path_names:
            path = getattr(plan_module, name, None)
            if path is None:
                raise ValueError(
                    f"frozen lineage path {name} is unavailable for "
                    f"{descriptor.science_kind}"
                )
            source = Path(path)
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"frozen lineage evidence is missing: {source}")
            keyword = _MERGE_LINEAGE_KEYWORDS[name]
            arguments[keyword] = json.loads(source.read_text(encoding="utf-8"))
    return merge(**arguments)


_MERGE_LINEAGE_KEYWORDS = {
    "DEFAULT_MATERIALIZATION_RECEIPT_PATH": "materialization_value",
    "DEFAULT_ROOT_SEAL_PATH": "root_seal_value",
}


def merge_accepted_results_to_scientific_gate(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    accepted_results_adapter: AcceptedResultsAdapterV2,
    lifecycle_chain_adapter: ValidatedLifecycleChainAdapterV2,
    merge_view_root: str | Path,
) -> dict[str, Any]:
    """Materialize a scientific view and return its one-shot gate receipt."""

    if not hasattr(accepted_results_adapter, "load_accepted_results"):
        raise TypeError("accepted-results adapter protocol is not implemented")
    if not hasattr(lifecycle_chain_adapter, "load_validated_lifecycle_chain"):
        raise TypeError("validated lifecycle-chain adapter is required")
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    lifecycle_chain = validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value=lifecycle_chain_adapter.load_validated_lifecycle_chain(
            wave_plan=deepcopy(plan), attempt_ledger=deepcopy(ledger)
        ),
    )
    raw_snapshot = accepted_results_adapter.load_accepted_results(
        wave_plan=deepcopy(plan),
        attempt_ledger=deepcopy(ledger),
        validated_lifecycle_chain=deepcopy(lifecycle_chain),
    )
    evidence = validate_accepted_results_snapshot(
        wave_plan=plan,
        attempt_ledger=ledger,
        validated_lifecycle_chain=lifecycle_chain,
        value=raw_snapshot,
    )
    manifest, candidate_paths, reference_paths = _materialize_merge_view(
        plan=plan,
        ledger=ledger,
        evidence=evidence,
        merge_view_root=Path(merge_view_root),
    )
    descriptor = science_registry.descriptor_for_plan(plan["full100_plan"])
    merged = _merge_for_lineage(
        descriptor=descriptor,
        plan=plan,
        candidate_paths=candidate_paths,
        reference_paths=reference_paths,
    )
    scientific_summary = _validate_merged_summary(
        merged,
        descriptor=descriptor,
        candidate_paths=candidate_paths,
        reference_paths=reference_paths,
    )
    if (
        scientific_summary.get("run_contract") != plan["full100_plan"]["run_contract"]
        or scientific_summary.get("run_contract_digest")
        != plan["run_contract_digest"]
        or scientific_summary.get("allocation") != runner.ALLOCATION
        or scientific_summary.get("candidate_library_sha256")
        != plan["runtime_binding"]["binary_sha256_by_role"]["candidate"]
        or scientific_summary.get("reference_library_sha256")
        != plan["runtime_binding"]["binary_sha256_by_role"]["reference"]
    ):
        raise ValueError("scientific merge and accepted runtime binding differ")
    gate = _performance_gate(scientific_summary)
    all_gates = gate["all_gates_passed"] is True
    prelaunch = {
        "expected_startup_sha256": evidence.snapshot["expected_startup_sha256"],
        "content_payload_sha256": evidence.snapshot["content_payload_sha256"],
        "outer_manifest_sha256": evidence.snapshot["outer_manifest_sha256"],
        "superseded_run004_guard": deepcopy(RUN004_PRELAUNCH_TARGET),
        "superseded_run004_rejected": True,
        "run005_guard": deepcopy(RUN005_PRELAUNCH_TARGET),
        "run005_guard_applied": (
            plan["run_name"] == RUN005_PRELAUNCH_TARGET["run_name"]
        ),
    }
    body = {
        "schema": GATE_RECEIPT_SCHEMA,
        "status": "pass" if all_gates else "no_go",
        "decision": (
            "full100_wave_v2_go_open_one_shot_performance_lock_only"
            if all_gates
            else "full100_wave_v2_no_go_performance_lock_closed"
        ),
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "wave_plan": deepcopy(plan),
        "attempt_ledger": deepcopy(ledger),
        "validated_lifecycle_chain": deepcopy(lifecycle_chain),
        "validated_lifecycle_chain_sha256": lifecycle_chain["chain_sha256"],
        "prelaunch_binding": prelaunch,
        "accepted_snapshot_sha256": evidence.snapshot["snapshot_sha256"],
        # The production bridge replays this snapshot by value, and it cannot
        # be rebuilt from the digests alone, so carry the body it was hashed
        # from rather than forcing a downstream reconstruction.
        "accepted_results_snapshot": deepcopy(evidence.snapshot),
        "receiver_receipt_sha256": evidence.snapshot["receiver_receipt_sha256"],
        "expected_inventory_sha256": evidence.expected["inventory_sha256"],
        "observed_inventory_sha256": evidence.observed["inventory_sha256"],
        "merge_view_manifest": manifest,
        "merge_view_manifest_sha256": canonical_sha256(manifest),
        "scientific_merge": scientific_summary,
        "scientific_merge_sha256": canonical_sha256(scientific_summary),
        "performance_gate": gate,
        "performance_gate_sha256": canonical_sha256(gate),
        "run_contract_digest": plan["run_contract_digest"],
        "rng_namespace_contract_sha256": manifest["rng_namespace_audit"][
            "contract_sha256"
        ],
        "pair_launch_lineage_sha256": manifest[
            "pair_launch_lineage_audit"
        ]["pair_launch_lineage_sha256"],
        "job_count": 20,
        "paired_hand_count": 100,
        "root_count": 200,
        "candidate_reference_separate_processes": True,
        "one_shot_immutable": True,
        "all_gates_passed": all_gates,
        "performance_candidate_frozen": all_gates,
        "performance_lock_authorized": all_gates,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    return validate_scientific_gate_receipt_value(
        {**body, "receipt_sha256": canonical_sha256(body)}
    )


def merge_and_write_scientific_gate_receipt(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    accepted_results_adapter: AcceptedResultsAdapterV2,
    lifecycle_chain_adapter: ValidatedLifecycleChainAdapterV2,
    merge_view_root: str | Path,
    receipt_output_path: str | Path,
) -> dict[str, Any]:
    """Run the bridge once and publish an immutable canonical receipt."""

    output = Path(receipt_output_path)
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    lifecycle_chain = validate_validated_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        value=lifecycle_chain_adapter.load_validated_lifecycle_chain(
            wave_plan=deepcopy(plan), attempt_ledger=deepcopy(ledger)
        ),
    )
    accepted_snapshot = accepted_results_adapter.load_accepted_results(
        wave_plan=deepcopy(dict(wave_plan)),
        attempt_ledger=deepcopy(dict(attempt_ledger)),
        validated_lifecycle_chain=deepcopy(lifecycle_chain),
    )
    accepted_root = Path(str(accepted_snapshot.get("accepted_root", ""))).resolve()
    view = Path(merge_view_root).resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("scientific gate receipt is write-once")
    if output == accepted_root or accepted_root in output.parents:
        raise ValueError("gate receipt must be outside accepted receiver tree")
    if output == view or view in output.parents:
        raise ValueError("gate receipt must be outside scientific merge-view")
    receipt = merge_accepted_results_to_scientific_gate(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        accepted_results_adapter=StaticAcceptedResultsAdapterV2(accepted_snapshot),
        lifecycle_chain_adapter=_PinnedValidatedLifecycleChainAdapterV2(
            lifecycle_chain
        ),
        merge_view_root=view,
    )
    _write_file_once(output, canonical_bytes(receipt))
    return receipt


__all__ = [
    "ACCEPTED_RESULTS_SNAPSHOT_SCHEMA",
    "AcceptedResultsAdapterV2",
    "CallbackValidatedLifecycleChainAdapterV2",
    "ControllerReceiverLifecycleChainAdapterV2",
    "EXPECTED_STARTUP_SHA256",
    "GATE_RECEIPT_SCHEMA",
    "MERGE_VIEW_MANIFEST_SCHEMA",
    "RUN004_PRELAUNCH_TARGET",
    "RUN005_PRELAUNCH_TARGET",
    "ReceiverReceiptAcceptedResultsAdapterV2",
    "StaticAcceptedResultsAdapterV2",
    "VALIDATED_LIFECYCLE_CHAIN_SCHEMA",
    "ValidatedLifecycleChainAdapterV2",
    "build_accepted_results_snapshot",
    "canonical_bytes",
    "canonical_sha256",
    "merge_accepted_results_to_scientific_gate",
    "merge_and_write_scientific_gate_receipt",
    "normalize_controller_receiver_lifecycle_chain",
    "validate_accepted_results_snapshot",
    "validate_scientific_gate_receipt",
    "validate_scientific_gate_receipt_value",
    "validate_validated_lifecycle_chain",
]
