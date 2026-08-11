"""Post-run trust audit for the one-shot Attempt05 old-dev900 comparison.

This command does not fit, rank, select, or rewrite the architecture report.
It revalidates all 900 inputs with the current hardened loader and verifies
independent artifact load/prediction parity after a source change that landed
while the one-shot process was already running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import action_key_from_payload
from .hu_m43_attempt05_model import HuM43Attempt05Model
from .hu_m43_attempt05_training import (
    ATTEMPT05_DEV_SCHEMA,
    ATTEMPT05_FAMILIES,
    ATTEMPT05_PROFILES,
    load_attempt05_dev900,
)


ATTEMPT05_POSTRUN_AUDIT_SCHEMA = "hu_m43_attempt05_postrun_trust_audit_v1"


def audit_attempt05_postrun(
    architecture_dir: str | Path,
    *,
    command: str,
    wall_seconds: float,
    parity_states_per_profile: int = 2,
) -> dict[str, Any]:
    root = Path(architecture_dir)
    report_path = root / "architecture_comparison.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema") != ATTEMPT05_DEV_SCHEMA:
        raise ValueError("Attempt05 architecture report schema mismatch")
    if report.get("selected_family") is not None:
        raise ValueError("Attempt05 postrun audit expected the frozen No-Go report")

    # This call is the full current trust-boundary validation: canonical
    # observation, complete legal ActionKey set, baseline key/index, T1-second,
    # paired targets, unique identities, and profile balance.
    data = load_attempt05_dev900()
    if data.source_manifest != report.get("source_manifest"):
        raise ValueError("Attempt05 current source manifest differs from one-shot report")
    fingerprints = [sample.observation_fingerprint for sample in data.samples]
    selected_indices = []
    for profile in ATTEMPT05_PROFILES:
        selected_indices.extend(
            [
                index
                for index, value in enumerate(data.profiles)
                if value == profile
            ][:parity_states_per_profile]
        )

    artifact_audits: dict[str, Any] = {}
    for family in ATTEMPT05_FAMILIES:
        family_report = report["families"][family]
        artifact = root / f"{family}_candidate.pkl"
        artifact_sha256 = _file_sha256(artifact)
        if artifact_sha256 != family_report["candidate_artifact_sha256"]:
            raise ValueError(f"Attempt05 {family} artifact digest differs from report")
        first = HuM43Attempt05Model.load(
            artifact, expected_sha256=artifact_sha256
        )
        second = HuM43Attempt05Model.load(
            artifact, expected_sha256=artifact_sha256
        )
        if first.runtime_enabled or first.winner_frozen:
            raise ValueError("Attempt05 candidate artifact is unexpectedly active")
        parity_rows = []
        for index in selected_indices:
            sample = data.samples[index]
            left = first.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            )
            right = second.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            )
            for name in (
                "action_score",
                "rank_score",
                "gain_probability",
                "downside_p95",
                "downside_p99",
                "downside_max",
                "upper_downside_p95",
                "upper_downside_p99",
                "upper_downside_max",
                "rank_disagreement",
                "gate_eligible_mask",
            ):
                if not np.array_equal(getattr(left, name), getattr(right, name)):
                    raise ValueError(
                        f"Attempt05 {family} independent-load {name} parity failed"
                    )
            if left.proposal_index != right.proposal_index:
                raise ValueError("Attempt05 independent-load proposal parity failed")
            action_key = action_key_from_payload(
                sample.policy_sample["actions"][left.proposal_index]
            ).to_token()
            parity_rows.append(
                {
                    "fingerprint": sample.observation_fingerprint,
                    "profile": data.profiles[index],
                    "proposal_action_key": action_key,
                    "rank": float(left.rank_score[left.proposal_index]),
                    "gain": float(left.gain_probability[left.proposal_index]),
                    "tails": [
                        float(left.downside_p95[left.proposal_index]),
                        float(left.downside_p99[left.proposal_index]),
                        float(left.downside_max[left.proposal_index]),
                    ],
                }
            )
        artifact_audits[family] = {
            "artifact": artifact.as_posix(),
            "artifact_sha256": artifact_sha256,
            "two_independent_loads_equal": True,
            "prediction_parity_states": len(parity_rows),
            "prediction_parity_sha256": _canonical_sha256(parity_rows),
            "runtime_enabled": False,
            "winner_frozen": False,
        }

    return {
        "schema": ATTEMPT05_POSTRUN_AUDIT_SCHEMA,
        "status": "pass_current_trust_boundary_no_reselection",
        "one_shot_command": command,
        "one_shot_wall_seconds": float(wall_seconds),
        "one_shot_report": report_path.as_posix(),
        "one_shot_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "one_shot_status": report["status"],
        "one_shot_selected_family": report["selected_family"],
        "source_timing": {
            "comparison_process_started_before_final_trust_hardening_landed": True,
            "model_weights_recomputed_after_hardening": False,
            "architecture_selection_recomputed_after_hardening": False,
            "comparison_reexecuted": False,
            "reason_no_rerun": (
                "one-shot contract; old dev records were already infoset-safe and "
                "current hardened loader revalidated the identical source hashes"
            ),
        },
        "current_loader_revalidation": {
            "states": len(data.samples),
            "unique_observation_identities": len(set(fingerprints)),
            "identity_sha256": _canonical_sha256(sorted(fingerprints)),
            "profile_counts": {
                profile: data.profiles.count(profile) for profile in ATTEMPT05_PROFILES
            },
            "source_manifest_equal_to_one_shot": True,
            "canonical_policy_observation": True,
            "complete_legal_action_key_set": True,
            "baseline_action_key_index_binding": True,
            "t1_second_only": True,
        },
        "artifact_audits": artifact_audits,
        "current_source_sha256": {
            "model": _file_sha256(Path(__file__).with_name("hu_m43_attempt05_model.py")),
            "training": _file_sha256(Path(__file__).with_name("hu_m43_attempt05_training.py")),
            "postrun_audit": _file_sha256(Path(__file__)),
        },
        "fresh_or_locked_data_opened": False,
        "spot_started": False,
        "current_profile_mutated": False,
        "new_audit_authorized": False,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--one-shot-command", required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--parity-states-per-profile", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_attempt05_postrun(
        args.architecture_dir,
        command=args.one_shot_command,
        wall_seconds=args.wall_seconds,
        parity_states_per_profile=args.parity_states_per_profile,
    )
    _write_no_clobber(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "one_shot_status": report["one_shot_status"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT05_POSTRUN_AUDIT_SCHEMA",
    "audit_attempt05_postrun",
]
