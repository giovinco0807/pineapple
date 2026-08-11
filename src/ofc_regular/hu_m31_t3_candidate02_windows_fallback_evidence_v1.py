"""Seal one path-free Windows/Linux Candidate02 fallback parity receipt.

This is diagnostic platform evidence only.  It neither replaces the accepted
Linux binaries nor authorizes cloud execution, training, promotion, or a
profile change.  The accepted performance-lock-v4 receipt remains the source
of the Linux portable decision digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import verify_hu_m31_t3_feature_encoder_platform_parity as feature

from .hu_infoset import ActorObservation
from .hu_m31_t3_performance_lock_v4_portable_receipt_v1 import (
    PINNED_CANDIDATE_LIBRARY_SHA256,
    PINNED_PROFILE_REGISTRY_SHA256,
    PINNED_PURE_V4_MERGE_SHA256,
    PINNED_RECEIPT_FILE_SHA256,
    PINNED_RECEIPT_SHA256,
    canonical_bytes,
    canonical_sha256,
    validate_pinned_portable_receipt,
)
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .run_hu_m31_t3_step6d_performance import portable_parity_payload


SCHEMA = "hu_m31_t3_candidate02_windows_fallback_evidence_v1"
STATUS = "diagnostic_cross_platform_semantic_parity_sealed_not_authorized"
DECISION = "windows_candidate02_semantic_fallback_supported_primary_linux_unchanged"

WINDOWS_CANDIDATE_SHA256 = (
    "4d07d13eadc243567f7fb73e567a73e5b3f955842ddf4e5b836f7df6b43a5245"
)
WINDOWS_FEATURE_SHA256 = (
    "24d156f455867ddb5915b07d440f72c790f893e310ef95d273b74d7b4d017b6b"
)
LINUX_FEATURE_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
FEATURE_INPUT_SHA256 = (
    "58d74968abf6a9ec9bb097fac00b7b3ecf0f3f58c554c973284d74edf460a535"
)
FEATURE_OUTPUT_SHA256 = (
    "a1cea3acc7b6fc5cf79dff6e5a61d3970f3ac2589f79623b4a2fd22ab83ea5e3"
)
ROOT_FILE_SHA256 = (
    "cda07629b32399b64204570fcea52f78655d758599afc204bb7a75fab6dbece5"
)
PRIOR_EQUIVALENCE_FILE_SHA256 = (
    "92bf5495805740b426bd9ef02be13ef3db4a538d67f309981af92f1102d74ff1"
)
PRIOR_CANDIDATE01_SHA256 = (
    "cc1728641ece247d3dd2d1977a5f4e29b745da7acd06c6c60298917c87ac50d4"
)
RUN_ID = "hu_m31_t3_step6d_candidate02_performance_lock_v4"
EXPECTED_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 32,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
EXPECTED_SEEDS = {
    "behavior": 721108071901,
    "candidate": 722108071901,
    "child": 724108071901,
    "confirmation": 725108071901,
    "evaluation": 723108071901,
    "hand": 720108071901,
}
EXPECTED_ROWS = (
    {
        "root_index": 0,
        "seat": "first",
        "observation_fingerprint": (
            "bc98a059a3ff4391157a036de5c2cc47a6b9efdff82f9f24cbb3888c6d024310"
        ),
        "portable_decision_sha256": (
            "2652920aee40e41932f4b9e4c702871fc19adbce15e0c28b1dbdc6bad00adc68"
        ),
        "action_count": 9,
        "child_information_set_count": 61_560,
    },
    {
        "root_index": 1,
        "seat": "second",
        "observation_fingerprint": (
            "b6192ce48bc09cd1d2fa266e1aca9e7945dbd20a24c615169fc4ffa4e67b1a16"
        ),
        "portable_decision_sha256": (
            "2c73d58fb614901bac4a1f6c1e2178e75cf28c1bb5f7209eef973cc9bf4a5dbf"
        ),
        "action_count": 21,
        "child_information_set_count": 1_680,
    },
)

_TOP_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "accepted_scientific_identity",
        "candidate_engine_parity",
        "feature_encoder_parity",
        "prior_windows_equivalence",
        "fallback_boundary",
        "receipt_sha256",
    }
)
_SEMANTIC_KEYS = frozenset(
    {
        "portable_payload_exact",
        "action_keys_exact",
        "selection_q_exact",
        "evaluation_q_exact",
        "selected_action_exact",
        "rng_exact",
        "child_information_set_count_exact",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return source


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _plain_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw not in (
        canonical_bytes(value),
        canonical_bytes(value) + b"\n",
    ):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _file_record(path: str | Path, label: str) -> dict[str, Any]:
    source = _plain_file(path, label)
    return {"sha256": sha256_file(source), "bytes": source.stat().st_size}


def _validate_feature_probe(
    value: Mapping[str, Any],
    *,
    label: str,
    library_sha256: str,
) -> dict[str, Any]:
    probe = deepcopy(dict(value))
    library = probe.get("library")
    generator = probe.get("generator")
    contract = generator.get("contract") if isinstance(generator, Mapping) else None
    if (
        probe.get("schema") != feature.PROBE_SCHEMA
        or probe.get("status") != "fixed_512_row_feature_encoder_probe_complete"
        or probe.get("platform_label") != label
        or not isinstance(library, Mapping)
        or library.get("sha256") != library_sha256
        or not isinstance(contract, Mapping)
        or generator.get("contract_sha256")
        != feature.PROBE_GENERATOR_CONTRACT_SHA256
        or probe.get("input_sha256") != FEATURE_INPUT_SHA256
        or probe.get("output_sha256") != FEATURE_OUTPUT_SHA256
        or probe.get("input_bytes") != 8661
        or probe.get("output_bytes") != 2_206_720
        or probe.get("row_count") != 512
        or contract.get("opponent_private_discards_used") is not False
        or contract.get("root_or_seed_input_used") is not False
    ):
        raise ValueError(f"{label} feature probe changed")
    return probe


def _linux_rows(
    receipt: Mapping[str, Any], *, hand_index: int
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    pure = receipt.get("pure_v4_merge")
    generic = pure.get("generic_merge") if isinstance(pure, Mapping) else None
    contract = generic.get("run_contract") if isinstance(generic, Mapping) else None
    paired = generic.get("paired_artifacts") if isinstance(generic, Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or contract.get("candidate_library_sha256")
        != PINNED_CANDIDATE_LIBRARY_SHA256
        or contract.get("step6d_run_id") != RUN_ID
        or contract.get("budget") != EXPECTED_BUDGET
        or not isinstance(paired, list)
        or len(paired) != 100
    ):
        raise ValueError("accepted Linux Candidate02 merge changed")
    hand = paired[hand_index]
    rows = hand.get("rows") if isinstance(hand, Mapping) else None
    if (
        not isinstance(hand, Mapping)
        or hand.get("hand_index") != hand_index
        or not isinstance(rows, list)
        or len(rows) != 2
    ):
        raise ValueError("accepted Linux paired hand changed")
    for row in rows:
        parity = row.get("portable_parity") if isinstance(row, Mapping) else None
        if (
            not isinstance(parity, Mapping)
            or set(parity)
            != _SEMANTIC_KEYS
            | {
                "schema",
                "candidate_portable_sha256",
                "reference_portable_sha256",
            }
            or any(parity.get(field) is not True for field in _SEMANTIC_KEYS)
            or parity.get("candidate_portable_sha256")
            != row.get("portable_decision_sha256")
            or parity.get("reference_portable_sha256")
            != row.get("portable_decision_sha256")
        ):
            raise ValueError("accepted Linux semantic parity changed")
    return contract, rows


def build_evidence(
    *,
    performance_receipt_path: str | Path,
    root_path: str | Path,
    windows_candidate_path: str | Path,
    linux_candidate_path: str | Path,
    windows_feature_probe_path: str | Path,
    linux_feature_probe_path: str | Path,
    prior_equivalence_path: str | Path,
) -> dict[str, Any]:
    performance = validate_pinned_portable_receipt(
        performance_receipt_path,
        expected_profile_sha256=PINNED_PROFILE_REGISTRY_SHA256,
    )
    contract, linux_rows = _linux_rows(performance, hand_index=0)
    windows_candidate = _file_record(
        windows_candidate_path, "Windows Candidate02"
    )
    linux_candidate = _file_record(linux_candidate_path, "Linux Candidate02")
    if (
        windows_candidate["sha256"] != WINDOWS_CANDIDATE_SHA256
        or linux_candidate["sha256"] != PINNED_CANDIDATE_LIBRARY_SHA256
    ):
        raise ValueError("Candidate02 platform binary pin changed")

    root_file = _plain_file(root_path, "performance-lock-v4 paired root")
    root = _read_canonical(root_file, "performance-lock-v4 paired root")
    if (
        sha256_file(root_file) != ROOT_FILE_SHA256
        or root.get("hand_index") != 0
        or root.get("budget") != EXPECTED_BUDGET
        or root.get("seeds") != EXPECTED_SEEDS
        or root.get("opponent_private_discards_used") is not False
        or root.get("current_profile_resolved") is not False
    ):
        raise ValueError("performance-lock-v4 replay root changed")

    seeds = root["seeds"]
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=WINDOWS_CANDIDATE_SHA256,
            library_path=Path(windows_candidate_path),
            require_release_library=False,
            run_id=RUN_ID,
            candidate_samples=8,
            evaluation_samples=32,
            downstream_t3_samples=4,
            seed=seeds["child"],
            candidate_seed=seeds["candidate"],
            evaluation_seed=seeds["evaluation"],
        )
    )
    candidate_rows: list[dict[str, Any]] = []
    observations = root.get("observations")
    if not isinstance(observations, list) or len(observations) != 2:
        raise ValueError("paired replay observations changed")
    for source, expected in zip(observations, linux_rows, strict=True):
        observation = ActorObservation.from_dict(source["observation"])
        portable = portable_parity_payload(solver.solve(observation).to_dict())
        digest = canonical_sha256(portable)
        if (
            source.get("root_index") != expected.get("root_index")
            or observation.seat != expected.get("seat")
            or observation.fingerprint()
            != expected.get("observation_fingerprint")
            or digest != expected.get("portable_decision_sha256")
        ):
            raise ValueError("Windows/Linux Candidate02 portable result changed")
        candidate_rows.append(
            {
                "root_index": source["root_index"],
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "portable_decision_sha256": digest,
                "portable_payload": portable,
                "semantic_families": {
                    field: True for field in sorted(_SEMANTIC_KEYS)
                },
            }
        )

    windows_probe = _validate_feature_probe(
        _read_canonical(windows_feature_probe_path, "Windows feature probe"),
        label="windows",
        library_sha256=WINDOWS_FEATURE_SHA256,
    )
    linux_probe = _validate_feature_probe(
        _read_canonical(linux_feature_probe_path, "Linux feature probe"),
        label="linux",
        library_sha256=LINUX_FEATURE_SHA256,
    )
    if (
        windows_probe["input_sha256"] != linux_probe["input_sha256"]
        or windows_probe["output_sha256"] != linux_probe["output_sha256"]
    ):
        raise ValueError("feature encoder cross-platform parity changed")

    prior_path = _plain_file(prior_equivalence_path, "prior Windows equivalence")
    prior = _read_canonical(prior_path, "prior Windows equivalence")
    if (
        sha256_file(prior_path) != PRIOR_EQUIVALENCE_FILE_SHA256
        or prior.get("status") != "go"
        or prior.get("decision") != "candidate02_bit_exact_equivalence_passed"
        or prior.get("candidate02", {}).get("sha256")
        != WINDOWS_CANDIDATE_SHA256
        or any(
            mode.get("candidate01_vs_candidate02", {}).get("exact") is not True
            for case in prior.get("cases", [])
            for mode in case.get("modes", [])
        )
    ):
        raise ValueError("prior Windows Candidate02 equivalence changed")

    core = {
        "schema": SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "accepted_scientific_identity": {
            "performance_receipt_file_sha256": PINNED_RECEIPT_FILE_SHA256,
            "performance_receipt_sha256": PINNED_RECEIPT_SHA256,
            "pure_v4_merge_sha256": PINNED_PURE_V4_MERGE_SHA256,
            "profile_registry_sha256": PINNED_PROFILE_REGISTRY_SHA256,
            "linux_candidate": linux_candidate,
        },
        "candidate_engine_parity": {
            "windows_candidate": windows_candidate,
            "paired_root_file_sha256": ROOT_FILE_SHA256,
            "hand_index": 0,
            "run_id": RUN_ID,
            "budget": dict(EXPECTED_BUDGET),
            "rows": candidate_rows,
            "paired_root_count": 2,
            "cross_platform_portable_payload_exact": True,
        },
        "feature_encoder_parity": {
            "windows_library": {
                "sha256": WINDOWS_FEATURE_SHA256,
                "bytes": windows_probe["library"]["bytes"],
            },
            "linux_library": {
                "sha256": LINUX_FEATURE_SHA256,
                "bytes": linux_probe["library"]["bytes"],
            },
            "row_count": 512,
            "input_bytes": 8661,
            "input_sha256": FEATURE_INPUT_SHA256,
            "output_bytes": 2_206_720,
            "output_sha256": FEATURE_OUTPUT_SHA256,
            "input_bit_exact": True,
            "output_bit_exact": True,
        },
        "prior_windows_equivalence": {
            "file_sha256": PRIOR_EQUIVALENCE_FILE_SHA256,
            "candidate01_sha256": prior["candidate01"]["sha256"],
            "candidate02_sha256": prior["candidate02"]["sha256"],
            "case_count": len(prior["cases"]),
            "cache_on_off_exact": True,
            "raw_json_ieee_f64_bit_exact": True,
        },
        "fallback_boundary": {
            "primary_platform": "accepted_linux_or_wsl",
            "windows_role": "slow_local_fallback_only",
            "windows_candidate_must_be_repackaged_under_release_path": True,
            "estimated_windows_slowdown_vs_accepted_linux": "about_6.2x_on_hand_000",
            "cloud_called": False,
            "training_eligible": False,
            "quality_evidence": False,
            "promotion_evidence": False,
            "replacement_linux_binary_authorized": False,
            "current_profile_changed": False,
            "opponent_private_discards_used": False,
        },
    }
    return validate_evidence_value(
        {**core, "receipt_sha256": canonical_sha256(core)}
    )


def validate_evidence_value(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if set(receipt) != _TOP_KEYS:
        raise ValueError("Windows fallback evidence fields changed")
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("Windows fallback evidence digest changed")
    receipt["receipt_sha256"] = digest
    accepted = receipt.get("accepted_scientific_identity")
    candidate = receipt.get("candidate_engine_parity")
    feature_parity = receipt.get("feature_encoder_parity")
    prior = receipt.get("prior_windows_equivalence")
    boundary = receipt.get("fallback_boundary")
    rows = candidate.get("rows") if isinstance(candidate, Mapping) else None
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or receipt.get("decision") != DECISION
        or accepted
        != {
            "performance_receipt_file_sha256": PINNED_RECEIPT_FILE_SHA256,
            "performance_receipt_sha256": PINNED_RECEIPT_SHA256,
            "pure_v4_merge_sha256": PINNED_PURE_V4_MERGE_SHA256,
            "profile_registry_sha256": PINNED_PROFILE_REGISTRY_SHA256,
            "linux_candidate": {
                "sha256": PINNED_CANDIDATE_LIBRARY_SHA256,
                "bytes": 1_230_800,
            },
        }
        or not isinstance(candidate, Mapping)
        or candidate.get("windows_candidate")
        != {"sha256": WINDOWS_CANDIDATE_SHA256, "bytes": 954_368}
        or candidate.get("paired_root_file_sha256") != ROOT_FILE_SHA256
        or candidate.get("hand_index") != 0
        or candidate.get("run_id") != RUN_ID
        or candidate.get("budget") != EXPECTED_BUDGET
        or candidate.get("paired_root_count") != 2
        or candidate.get("cross_platform_portable_payload_exact") is not True
        or not isinstance(rows, list)
        or len(rows) != 2
        or any(
            row.get("root_index") != expected["root_index"]
            or row.get("seat") != expected["seat"]
            or row.get("observation_fingerprint")
            != expected["observation_fingerprint"]
            or row.get("portable_decision_sha256")
            != expected["portable_decision_sha256"]
            or row.get("portable_decision_sha256")
            != canonical_sha256(row.get("portable_payload"))
            or len(row.get("portable_payload", {}).get("action_values", []))
            != expected["action_count"]
            or row.get("portable_payload", {}).get(
                "child_information_set_count"
            )
            != expected["child_information_set_count"]
            or set(row.get("semantic_families", {})) != _SEMANTIC_KEYS
            or any(
                row["semantic_families"].get(field) is not True
                for field in _SEMANTIC_KEYS
            )
            for row, expected in zip(rows, EXPECTED_ROWS, strict=True)
        )
        or not isinstance(feature_parity, Mapping)
        or feature_parity.get("windows_library")
        != {"sha256": WINDOWS_FEATURE_SHA256, "bytes": 199_680}
        or feature_parity.get("linux_library")
        != {"sha256": LINUX_FEATURE_SHA256, "bytes": 489_144}
        or feature_parity.get("row_count") != 512
        or feature_parity.get("input_bytes") != 8661
        or feature_parity.get("output_bytes") != 2_206_720
        or feature_parity.get("input_sha256") != FEATURE_INPUT_SHA256
        or feature_parity.get("output_sha256") != FEATURE_OUTPUT_SHA256
        or feature_parity.get("input_bit_exact") is not True
        or feature_parity.get("output_bit_exact") is not True
        or prior
        != {
            "file_sha256": PRIOR_EQUIVALENCE_FILE_SHA256,
            "candidate01_sha256": PRIOR_CANDIDATE01_SHA256,
            "candidate02_sha256": WINDOWS_CANDIDATE_SHA256,
            "case_count": 2,
            "cache_on_off_exact": True,
            "raw_json_ieee_f64_bit_exact": True,
        }
        or not isinstance(boundary, Mapping)
        or boundary.get("primary_platform") != "accepted_linux_or_wsl"
        or any(
            boundary.get(field) is not False
            for field in (
                "cloud_called",
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "replacement_linux_binary_authorized",
                "current_profile_changed",
                "opponent_private_discards_used",
            )
        )
    ):
        raise ValueError("Windows fallback evidence safety boundary changed")
    return receipt


def write_evidence(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(canonical_bytes(dict(value)))
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--windows-candidate", type=Path, required=True)
    parser.add_argument("--linux-candidate", type=Path, required=True)
    parser.add_argument("--windows-feature-probe", type=Path, required=True)
    parser.add_argument("--linux-feature-probe", type=Path, required=True)
    parser.add_argument("--prior-equivalence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = build_evidence(
        performance_receipt_path=args.performance_receipt,
        root_path=args.root,
        windows_candidate_path=args.windows_candidate,
        linux_candidate_path=args.linux_candidate,
        windows_feature_probe_path=args.windows_feature_probe,
        linux_feature_probe_path=args.linux_feature_probe,
        prior_equivalence_path=args.prior_equivalence,
    )
    write_evidence(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "decision": value["decision"],
                "receipt_sha256": value["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
