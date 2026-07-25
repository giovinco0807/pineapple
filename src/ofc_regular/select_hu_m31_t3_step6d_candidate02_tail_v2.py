"""Select the fresh Candidate02 tail probe from root topology only.

The first Candidate02 tail probe exposed ten logical hands to runtime timing.
This selector therefore starts from the already-frozen all-100 Candidate02
root set, excludes those ten hands, and uses only public root topology:

* for each non-random behavior profile, take the first two remaining roots
  whose first-seat action matrix is exactly 21 by 21;
* for the random behavior profile, take the first two remaining roots as an
  explicit low-cost calibration stratum.

No candidate/reference result, elapsed time, memory value, EV, Q value, or
teacher output is accepted as selector input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_space import generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


SELECTION_MANIFEST_SCHEMA = "hu_m31_t3_step6d_candidate02_tail_reselection_manifest_v2"
ROOT_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_root_v1"
CONTRACT_HAND_INDICES = tuple(range(100))
PRIOR_RUNTIME_EXPOSED_HAND_INDICES = (2, 6, 7, 9, 13, 20, 21, 29, 33, 50)
HEAVY_HAND_INDICES = (0, 5, 12, 16, 17, 23, 41, 43)
RANDOM_HAND_INDICES = (4, 14)
TAIL_HAND_INDICES = (0, 4, 5, 12, 14, 16, 17, 23, 41, 43)
ALL100_ROOT_SHA256 = "0aacb1b7f9b3c45a7ca51d9e58218e55ef3cc29159e2d431757806be076e6796"
TOPOLOGY_SHA256 = "779e6a6d2ce6d84ccb31df6e02c48efdb7833ef06dab0625aeeab551d3d81633"
SELECTION_MANIFEST_SHA256 = (
    "62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1"
)
RANDOM_PROFILE = "random_exact_final"
NON_RANDOM_PROFILES = tuple(
    profile for profile in M31_T3_BEHAVIOR_PROFILES if profile != RANDOM_PROFILE
)

_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "root_schema",
        "contract_hand_indices",
        "prior_runtime_exposed_hand_indices",
        "all100_root_sha256",
        "topology_sha256",
        "selection_rule",
        "heavy_hand_indices_by_profile",
        "heavy_hand_indices",
        "random_hand_indices",
        "tail_hand_indices",
        "runtime_results_used",
        "timing_used",
        "memory_used",
        "teacher_values_used",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_SELECTION_RULE = {
    "ordering": "ascending_hand_index",
    "heavy_profiles": list(NON_RANDOM_PROFILES),
    "heavy_per_profile": 2,
    "heavy_predicate": (
        "first_hero_legal_actions_eq_21_and_"
        "first_opponent_response_legal_actions_eq_21"
    ),
    "random_profile": RANDOM_PROFILE,
    "random_count": 2,
    "random_predicate": "profile_only",
    "exclude_prior_runtime_exposed": True,
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _read_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"Candidate02 tail-v2 root is not canonical: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite Candidate02 tail-v2 manifest: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite Candidate02 tail-v2 manifest: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_frozen_roots(root_dir: str | Path) -> list[dict[str, Any]]:
    """Load and independently validate the exact all-100 Candidate02 roots."""

    # The producer owns the complete root provenance contract.  Importing here
    # avoids a module cycle when the runner consumes the frozen manifest hash.
    from . import run_hu_m31_t3_step6d_performance_v2 as runner

    directory = Path(root_dir).resolve()
    expected_names = [f"hand_{index:03d}.json" for index in CONTRACT_HAND_INDICES]
    observed_names = sorted(path.name for path in directory.glob("hand_*.json"))
    if observed_names != expected_names:
        raise ValueError("Candidate02 tail-v2 requires exactly roots 000..099")
    roots: list[dict[str, Any]] = []
    for index, name in zip(CONTRACT_HAND_INDICES, expected_names, strict=True):
        root = _read_canonical(directory / name)
        runner._validate_candidate02_root_artifact(
            root,
            expected_row=runner.candidate02_schedule_row(index),
        )
        roots.append(root)
    digest = canonical_sha256([canonical_sha256(root) for root in roots])
    if digest != ALL100_ROOT_SHA256:
        raise ValueError(
            "Candidate02 all-100 frozen root digest changed: "
            f"expected {ALL100_ROOT_SHA256}, got {digest}"
        )
    return roots


def topology_rows(roots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract the only root data allowed to influence the selection."""

    if len(roots) != len(CONTRACT_HAND_INDICES):
        raise ValueError("Candidate02 tail-v2 topology requires exactly 100 roots")
    rows: list[dict[str, Any]] = []
    for index, root in enumerate(roots):
        if root.get("hand_index") != index or root.get("schema") != ROOT_SCHEMA:
            raise ValueError("Candidate02 tail-v2 root order/schema changed")
        observations = root.get("observations")
        if not isinstance(observations, list) or len(observations) != 2:
            raise ValueError("Candidate02 tail-v2 root observations changed")
        first_raw = observations[0]
        if not isinstance(first_raw, Mapping):
            raise ValueError("Candidate02 tail-v2 first observation is missing")
        first = ActorObservation.from_dict(first_raw["observation"])
        if (
            first.seat != "first"
            or first.to_act_order != "first"
            or first.street != "T3"
            or first_raw.get("observation_fingerprint") != first.fingerprint()
        ):
            raise ValueError("Candidate02 tail-v2 first observation changed")
        rows.append(
            {
                "hand_index": index,
                "profile": root["profile"],
                "first_fingerprint": first_raw["observation_fingerprint"],
                "hero_legal_actions": len(
                    generate_turn_actions(first.hero_board, first.dealt_cards)
                ),
                "opponent_response_legal_actions": len(
                    generate_turn_actions(
                        first.opponent_public_board,
                        first.dealt_cards,
                    )
                ),
            }
        )
    digest = canonical_sha256(rows)
    if digest != TOPOLOGY_SHA256:
        raise ValueError(
            "Candidate02 all-100 topology digest changed: "
            f"expected {TOPOLOGY_SHA256}, got {digest}"
        )
    return rows


def build_selection_manifest(
    roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = topology_rows(roots)
    exposed = set(PRIOR_RUNTIME_EXPOSED_HAND_INDICES)
    heavy_by_profile: dict[str, list[int]] = {}
    for profile in NON_RANDOM_PROFILES:
        eligible = [
            int(row["hand_index"])
            for row in rows
            if row["profile"] == profile
            and row["hand_index"] not in exposed
            and row["hero_legal_actions"] == 21
            and row["opponent_response_legal_actions"] == 21
        ]
        if len(eligible) < 2:
            raise ValueError(f"Candidate02 tail-v2 lacks two heavy roots for {profile}")
        heavy_by_profile[profile] = eligible[:2]
    random = [
        int(row["hand_index"])
        for row in rows
        if row["profile"] == RANDOM_PROFILE and row["hand_index"] not in exposed
    ][:2]
    if len(random) != 2:
        raise ValueError("Candidate02 tail-v2 lacks two random calibration roots")
    heavy = sorted(index for values in heavy_by_profile.values() for index in values)
    tail = sorted([*heavy, *random])
    value = {
        "schema": SELECTION_MANIFEST_SCHEMA,
        "root_schema": ROOT_SCHEMA,
        "contract_hand_indices": list(CONTRACT_HAND_INDICES),
        "prior_runtime_exposed_hand_indices": list(PRIOR_RUNTIME_EXPOSED_HAND_INDICES),
        "all100_root_sha256": ALL100_ROOT_SHA256,
        "topology_sha256": TOPOLOGY_SHA256,
        "selection_rule": dict(_SELECTION_RULE),
        "heavy_hand_indices_by_profile": heavy_by_profile,
        "heavy_hand_indices": heavy,
        "random_hand_indices": random,
        "tail_hand_indices": tail,
        "runtime_results_used": False,
        "timing_used": False,
        "memory_used": False,
        "teacher_values_used": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    return validate_selection_manifest(value)


def validate_selection_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("Candidate02 tail-v2 selection manifest fields changed")
    expected_by_profile = {
        "stage19_p0": [0, 5],
        "stage9f_p2": [16, 41],
        "stage7_m5_r10": [12, 17],
        "stage3_baseline": [23, 43],
    }
    if (
        payload.get("schema") != SELECTION_MANIFEST_SCHEMA
        or payload.get("root_schema") != ROOT_SCHEMA
        or payload.get("contract_hand_indices") != list(CONTRACT_HAND_INDICES)
        or payload.get("prior_runtime_exposed_hand_indices")
        != list(PRIOR_RUNTIME_EXPOSED_HAND_INDICES)
        or payload.get("all100_root_sha256") != ALL100_ROOT_SHA256
        or payload.get("topology_sha256") != TOPOLOGY_SHA256
        or payload.get("selection_rule") != _SELECTION_RULE
        or payload.get("heavy_hand_indices_by_profile") != expected_by_profile
        or payload.get("heavy_hand_indices") != list(HEAVY_HAND_INDICES)
        or payload.get("random_hand_indices") != list(RANDOM_HAND_INDICES)
        or payload.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or set(payload["tail_hand_indices"]) & set(PRIOR_RUNTIME_EXPOSED_HAND_INDICES)
        or any(
            payload.get(field) is not False
            for field in (
                "runtime_results_used",
                "timing_used",
                "memory_used",
                "teacher_values_used",
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("Candidate02 tail-v2 selection manifest changed")
    if canonical_sha256(payload) != SELECTION_MANIFEST_SHA256:
        raise ValueError("Candidate02 tail-v2 selection manifest digest changed")
    return payload


def select_candidate02_tail_v2(root_dir: str | Path) -> dict[str, Any]:
    return build_selection_manifest(load_frozen_roots(root_dir))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = select_candidate02_tail_v2(args.root_dir)
    if args.output is not None:
        _write_once(args.output.resolve(), manifest)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL100_ROOT_SHA256",
    "CONTRACT_HAND_INDICES",
    "HEAVY_HAND_INDICES",
    "PRIOR_RUNTIME_EXPOSED_HAND_INDICES",
    "RANDOM_HAND_INDICES",
    "SELECTION_MANIFEST_SHA256",
    "SELECTION_MANIFEST_SCHEMA",
    "TAIL_HAND_INDICES",
    "TOPOLOGY_SHA256",
    "build_selection_manifest",
    "canonical_bytes",
    "canonical_sha256",
    "load_frozen_roots",
    "main",
    "select_candidate02_tail_v2",
    "topology_rows",
    "validate_selection_manifest",
]
