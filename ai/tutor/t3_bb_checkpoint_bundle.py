"""Content-addressed round bundles for T3-BB fixed-point MCCFR output.

One bundle represents exactly one fixed-point round across every locked
``(root, solver_seed)`` job.  It turns the many physical MCCFR checkpoints
into one path-independent checkpoint identity while retaining an independently
verifiable, path-binding manifest.

This is a transport and completeness artifact only.  It deliberately carries
``promotion_eligible == False`` and does not claim convergence, strength, or
exact exploitability.  Those claims belong to the fixed-point and M3 strength
gates.

The verifier never trusts caller-supplied checkpoint or strategy digests.  It
fresh-reads both files, validates the full-card MCCFR envelope, reconstructs
the average public strategy from the checkpoint's strategy sums, and then
re-derives every entry, coverage, bundle, and manifest hash.  Asset paths are
strict relative POSIX paths below the bundle root; traversal, symlink aliases,
duplicate keys, duplicate files, and missing coverage fail closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_SCHEMA = "ofc_t3_bb_round_checkpoint_bundle/v1"
ENTRY_SCHEMA = "ofc_t3_bb_round_checkpoint_entry/v1"
COVERAGE_SCHEMA = "ofc_t3_bb_round_checkpoint_coverage/v1"
BUNDLE_CONTENT_SCHEMA = "ofc_t3_bb_round_checkpoint_content/v1"
ARTIFACT_KIND = "t3_bb_fixed_point_round_mccfr_checkpoint_bundle"
SCOPE = "one_fixed_point_round_all_locked_roots_and_solver_seeds"

FULL_CARD_CHECKPOINT_FORMAT = "full_card_dynamic_mccfr_checkpoint_v1"
FULL_CARD_STATE_FORMAT = "full_card_dynamic_mccfr_state_v1"
FULL_CARD_ADAPTER_FORMAT = "full_card_dynamic_mccfr_adapter_manifest_v1"
PUBLIC_STRATEGY_SCHEMA = "ofc_full_card_public_strategy/v1"
POLICY_IDENTITY_CONTRACT = "infoset_key_plus_lexical_action_key_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_ONLY_RE = re.compile(rb"^[0-9a-f]{64}$")

_KEY_FIELDS = frozenset(
    {"round_index", "root_id", "root_commitment_sha256", "solver_seed"}
)
_SOURCE_FIELDS = frozenset(
    {
        *_KEY_FIELDS,
        "checkpoint_path",
        "average_strategy_json_path",
        "range_content_sha256",
        "range_build_sha256",
        "observation_digest",
        "solver_manifest_sha256",
        "source_manifest_sha256",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "schema",
        *_KEY_FIELDS,
        "checkpoint_path",
        "checkpoint_file_bytes_sha256",
        "checkpoint_content_sha256",
        "average_strategy_json_path",
        "average_strategy_json_bytes_sha256",
        "average_strategy_json_content_sha256",
        "range_content_sha256",
        "range_build_sha256",
        "observation_digest",
        "solver_manifest_sha256",
        "source_manifest_sha256",
        "entry_sha256",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "artifact_kind",
        "scope",
        "promotion_eligible",
        "exact_exploitability_computed",
        "round_index",
        "expected_key_set_sha256",
        "entry_count",
        "entries",
        "bundle_checkpoint_sha256",
        "manifest_sha256",
    }
)
_CHECKPOINT_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "completed_iterations",
        "seed",
        "linear_averaging",
        "solver_config",
        "adapter_manifest",
        "adapter_manifest_sha256",
        "tables",
        "rng_state",
        "sampling_stats",
    }
)
_CHECKPOINT_TABLE_FIELDS = frozenset(
    {
        "infoset_canonical_json",
        "infoset_sha256",
        "actor",
        "stable_action_ids",
        "regret_plus_hex",
        "strategy_sum_hex",
    }
)


class T3BBCheckpointBundleError(ValueError):
    """A T3-BB checkpoint bundle failed its fail-closed contract."""


def canonical_json(value: Any) -> str:
    """Serialize finite JSON with the bundle's canonical encoding."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise T3BBCheckpointBundleError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise T3BBCheckpointBundleError(f"{label}: lowercase SHA256 required")
    return value


def _require_int(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise T3BBCheckpointBundleError(f"{label}: integer required")
    if positive and value <= 0:
        raise T3BBCheckpointBundleError(f"{label}: positive integer required")
    return value


def _require_root_id(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise T3BBCheckpointBundleError(
            f"{label}: non-empty, trimmed root identifier required"
        )
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise T3BBCheckpointBundleError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise T3BBCheckpointBundleError(f"non-finite JSON constant: {value}")


def _parse_json_bytes(value: bytes, *, label: str) -> Any:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise T3BBCheckpointBundleError(f"{label}: valid UTF-8 JSON required") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except T3BBCheckpointBundleError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise T3BBCheckpointBundleError(f"{label}: valid JSON required") from exc


@dataclass(frozen=True, order=True)
class T3BBCheckpointKey:
    """Coverage identity for one physical solver job."""

    round_index: int
    root_id: str
    root_commitment_sha256: str
    solver_seed: int

    def validated(self, *, label: str = "key") -> "T3BBCheckpointKey":
        _require_int(self.round_index, label=f"{label}.round_index", positive=True)
        _require_root_id(self.root_id, label=f"{label}.root_id")
        _require_sha256(
            self.root_commitment_sha256,
            label=f"{label}.root_commitment_sha256",
        )
        _require_int(self.solver_seed, label=f"{label}.solver_seed")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "root_id": self.root_id,
            "root_commitment_sha256": self.root_commitment_sha256,
            "solver_seed": self.solver_seed,
        }


@dataclass(frozen=True)
class T3BBCheckpointBundleEntrySource:
    """Physical files and locked bindings used to materialize one entry."""

    round_index: int
    root_id: str
    root_commitment_sha256: str
    solver_seed: int
    checkpoint_path: str | os.PathLike[str]
    average_strategy_json_path: str | os.PathLike[str]
    range_content_sha256: str
    range_build_sha256: str
    observation_digest: str
    solver_manifest_sha256: str
    source_manifest_sha256: str

    @property
    def key(self) -> T3BBCheckpointKey:
        return T3BBCheckpointKey(
            round_index=self.round_index,
            root_id=self.root_id,
            root_commitment_sha256=self.root_commitment_sha256,
            solver_seed=self.solver_seed,
        )


def _coerce_key(value: Any, *, label: str) -> T3BBCheckpointKey:
    if isinstance(value, T3BBCheckpointKey):
        return value.validated(label=label)
    if not isinstance(value, Mapping):
        raise T3BBCheckpointBundleError(f"{label}: key object required")
    if set(value) != _KEY_FIELDS:
        raise T3BBCheckpointBundleError(f"{label}: exact key fields required")
    return T3BBCheckpointKey(
        round_index=value.get("round_index"),
        root_id=value.get("root_id"),
        root_commitment_sha256=value.get("root_commitment_sha256"),
        solver_seed=value.get("solver_seed"),
    ).validated(label=label)


def _coerce_source(value: Any, *, label: str) -> T3BBCheckpointBundleEntrySource:
    if isinstance(value, T3BBCheckpointBundleEntrySource):
        source = value
    elif isinstance(value, Mapping):
        if set(value) != _SOURCE_FIELDS:
            raise T3BBCheckpointBundleError(f"{label}: exact source fields required")
        source = T3BBCheckpointBundleEntrySource(**dict(value))
    else:
        raise T3BBCheckpointBundleError(f"{label}: entry source object required")
    source.key.validated(label=label)
    for field in (
        "range_content_sha256",
        "range_build_sha256",
        "observation_digest",
        "solver_manifest_sha256",
        "source_manifest_sha256",
    ):
        _require_sha256(getattr(source, field), label=f"{label}.{field}")
    return source


def _normalize_expected_keys(
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> tuple[T3BBCheckpointKey, ...]:
    if isinstance(expected_keys, (str, bytes, Mapping)):
        raise T3BBCheckpointBundleError("expected_keys: sequence of key objects required")
    try:
        values = tuple(expected_keys)
    except TypeError as exc:
        raise T3BBCheckpointBundleError("expected_keys: iterable required") from exc
    if not values:
        raise T3BBCheckpointBundleError("expected_keys: at least one key required")
    keys = tuple(
        _coerce_key(value, label=f"expected_keys[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(keys)) != len(keys):
        raise T3BBCheckpointBundleError("expected_keys: duplicate key")
    rounds = {key.round_index for key in keys}
    if len(rounds) != 1:
        raise T3BBCheckpointBundleError(
            "expected_keys: one bundle must contain exactly one round"
        )
    root_commitments: dict[str, str] = {}
    for key in keys:
        prior = root_commitments.setdefault(
            key.root_id, key.root_commitment_sha256
        )
        if prior != key.root_commitment_sha256:
            raise T3BBCheckpointBundleError(
                "expected_keys: one root_id cannot have multiple commitments"
            )
    return tuple(sorted(keys))


def _coverage_sha256(keys: Sequence[T3BBCheckpointKey]) -> str:
    return canonical_sha256(
        {"schema": COVERAGE_SCHEMA, "keys": [key.as_dict() for key in keys]}
    )


def _safe_asset(
    bundle_root: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[str, Path]:
    root = Path(bundle_root)
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise T3BBCheckpointBundleError(f"bundle_root: cannot resolve: {exc}") from exc
    if not root_resolved.is_dir():
        raise T3BBCheckpointBundleError("bundle_root: directory required")

    try:
        raw = (
            relative_path.as_posix()
            if isinstance(relative_path, Path)
            else os.fspath(relative_path)
        )
    except TypeError as exc:
        raise T3BBCheckpointBundleError(f"{label}: path-like value required") from exc
    if isinstance(raw, bytes):
        raise T3BBCheckpointBundleError(f"{label}: text path required")
    if not raw or "\\" in raw or "\x00" in raw or ":" in raw:
        raise T3BBCheckpointBundleError(
            f"{label}: strict relative POSIX path required"
        )
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise T3BBCheckpointBundleError(
            f"{label}: path traversal or absolute path rejected"
        )
    candidate = root_resolved.joinpath(*pure.parts)
    current = root_resolved
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise T3BBCheckpointBundleError(f"{label}: symlink assets are rejected")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise T3BBCheckpointBundleError(
            f"{label}: asset must resolve below bundle_root"
        ) from exc
    if not resolved.is_file():
        raise T3BBCheckpointBundleError(f"{label}: regular file required")
    return pure.as_posix(), resolved


def _normalized_average(values: Sequence[float]) -> list[float]:
    if not values:
        raise T3BBCheckpointBundleError("checkpoint table has no strategy actions")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise T3BBCheckpointBundleError(
            "checkpoint strategy sums must be finite and non-negative"
        )
    total = math.fsum(values)
    if total <= 0.0:
        probability = 1.0 / len(values)
        return [probability for _ in values]
    return [value / total for value in values]


def _validate_checkpoint_and_reconstruct_strategy(
    checkpoint_bytes: bytes,
    *,
    solver_seed: int,
    observation_digest: str,
    range_content_sha256: str,
    range_build_sha256: str,
) -> tuple[str, dict[str, Any]]:
    if not checkpoint_bytes:
        raise T3BBCheckpointBundleError("checkpoint: non-empty bytes required")
    if _DIGEST_ONLY_RE.fullmatch(checkpoint_bytes.strip()) is not None:
        raise T3BBCheckpointBundleError(
            "checkpoint: a SHA256 string is not checkpoint bytes"
        )
    envelope = _parse_json_bytes(checkpoint_bytes, label="checkpoint")
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "format",
        "checkpoint_sha256",
        "payload",
    }:
        raise T3BBCheckpointBundleError("checkpoint: envelope schema mismatch")
    if envelope.get("format") != FULL_CARD_CHECKPOINT_FORMAT:
        raise T3BBCheckpointBundleError("checkpoint: full-card format required")
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_PAYLOAD_FIELDS:
        raise T3BBCheckpointBundleError("checkpoint: payload schema mismatch")
    content_sha = _require_sha256(
        envelope.get("checkpoint_sha256"), label="checkpoint.checkpoint_sha256"
    )
    if canonical_sha256(payload) != content_sha:
        raise T3BBCheckpointBundleError("checkpoint: payload content hash mismatch")
    if payload.get("format") != FULL_CARD_STATE_FORMAT:
        raise T3BBCheckpointBundleError("checkpoint: solver-state format mismatch")
    if _require_int(payload.get("seed"), label="checkpoint.payload.seed") != solver_seed:
        raise T3BBCheckpointBundleError("checkpoint: solver seed binding mismatch")
    _require_int(
        payload.get("completed_iterations"),
        label="checkpoint.payload.completed_iterations",
        positive=True,
    )
    if not isinstance(payload.get("linear_averaging"), bool):
        raise T3BBCheckpointBundleError(
            "checkpoint.payload.linear_averaging: boolean required"
        )

    adapter = payload.get("adapter_manifest")
    if not isinstance(adapter, Mapping):
        raise T3BBCheckpointBundleError("checkpoint: adapter manifest required")
    if adapter.get("format") != FULL_CARD_ADAPTER_FORMAT:
        raise T3BBCheckpointBundleError("checkpoint: adapter format mismatch")
    adapter_hash = _require_sha256(
        payload.get("adapter_manifest_sha256"),
        label="checkpoint.payload.adapter_manifest_sha256",
    )
    if canonical_sha256(adapter) != adapter_hash:
        raise T3BBCheckpointBundleError("checkpoint: adapter manifest hash mismatch")
    adapter_self_hash = _require_sha256(
        adapter.get("manifest_sha256"),
        label="checkpoint.adapter_manifest.manifest_sha256",
    )
    adapter_unsigned = dict(adapter)
    adapter_unsigned.pop("manifest_sha256", None)
    if canonical_sha256(adapter_unsigned) != adapter_self_hash:
        raise T3BBCheckpointBundleError("checkpoint: adapter self-hash mismatch")
    if adapter.get("root_infoset_sha256") != observation_digest:
        raise T3BBCheckpointBundleError("checkpoint: observation binding mismatch")
    range_binding = adapter.get("range_behavior_binding")
    if not isinstance(range_binding, Mapping):
        raise T3BBCheckpointBundleError("checkpoint: range binding required")
    for field, wanted in (
        ("observation_digest", observation_digest),
        ("range_content_sha256", range_content_sha256),
        ("range_build_sha256", range_build_sha256),
    ):
        if range_binding.get(field) != wanted:
            raise T3BBCheckpointBundleError(
                f"checkpoint: range binding {field} mismatch"
            )

    tables = payload.get("tables")
    if not isinstance(tables, list) or not tables:
        raise T3BBCheckpointBundleError("checkpoint: non-empty table list required")
    records: list[tuple[tuple[str, str], dict[str, Any]]] = []
    seen_infosets: set[tuple[str, str]] = set()
    root_record_count = 0
    for index, table in enumerate(tables):
        label = f"checkpoint.payload.tables[{index}]"
        if not isinstance(table, Mapping) or set(table) != _CHECKPOINT_TABLE_FIELDS:
            raise T3BBCheckpointBundleError(f"{label}: exact table schema required")
        infoset_json = table.get("infoset_canonical_json")
        if not isinstance(infoset_json, str) or not infoset_json:
            raise T3BBCheckpointBundleError(f"{label}: infoset JSON string required")
        infoset_bytes = infoset_json.encode("utf-8")
        infoset = _parse_json_bytes(infoset_bytes, label=f"{label}.infoset")
        if canonical_json(infoset) != infoset_json:
            raise T3BBCheckpointBundleError(f"{label}: infoset JSON is not canonical")
        digest = _require_sha256(
            table.get("infoset_sha256"), label=f"{label}.infoset_sha256"
        )
        if hashlib.sha256(infoset_bytes).hexdigest() != digest:
            raise T3BBCheckpointBundleError(f"{label}: infoset digest mismatch")
        if digest == observation_digest:
            root_record_count += 1
        identity = (digest, infoset_json)
        if identity in seen_infosets:
            raise T3BBCheckpointBundleError(f"{label}: duplicate infoset")
        seen_infosets.add(identity)
        action_ids = table.get("stable_action_ids")
        if (
            not isinstance(action_ids, list)
            or not action_ids
            or any(not isinstance(action, str) or not action for action in action_ids)
            or action_ids != sorted(set(action_ids))
        ):
            raise T3BBCheckpointBundleError(
                f"{label}: sorted unique action identifiers required"
            )
        raw_sums = table.get("strategy_sum_hex")
        raw_regrets = table.get("regret_plus_hex")
        if (
            not isinstance(raw_sums, list)
            or not isinstance(raw_regrets, list)
            or len(raw_sums) != len(action_ids)
            or len(raw_regrets) != len(action_ids)
        ):
            raise T3BBCheckpointBundleError(f"{label}: action-vector length mismatch")
        try:
            sums = [float.fromhex(value) for value in raw_sums]
            regrets = [float.fromhex(value) for value in raw_regrets]
        except (TypeError, ValueError) as exc:
            raise T3BBCheckpointBundleError(
                f"{label}: hexadecimal float vectors required"
            ) from exc
        _normalized_average(regrets)  # validates finite/non-negative regret+ values
        probabilities = _normalized_average(sums)
        records.append(
            (
                identity,
                {
                    "infoset_digest": digest,
                    "infoset": infoset,
                    "actions": [
                        {"action_id": action_id, "probability": probabilities[offset]}
                        for offset, action_id in enumerate(action_ids)
                    ],
                },
            )
        )
    if records != sorted(records, key=lambda item: item[0]):
        raise T3BBCheckpointBundleError("checkpoint: table order is not canonical")
    if root_record_count != 1:
        raise T3BBCheckpointBundleError(
            "checkpoint: exactly one root-observation table required"
        )
    strategy = {
        "schema": PUBLIC_STRATEGY_SCHEMA,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "records": [record for _identity, record in records],
    }
    return content_sha, strategy


def _validate_strategy(
    strategy_bytes: bytes,
    *,
    expected_strategy: Mapping[str, Any],
) -> tuple[str, str]:
    if not strategy_bytes:
        raise T3BBCheckpointBundleError("average_strategy_json: non-empty bytes required")
    parsed = _parse_json_bytes(strategy_bytes, label="average_strategy_json")
    if not isinstance(parsed, Mapping):
        raise T3BBCheckpointBundleError(
            "average_strategy_json: strategy object required, not a hash string"
        )
    if parsed != expected_strategy:
        raise T3BBCheckpointBundleError(
            "average_strategy_json: does not match checkpoint strategy sums"
        )
    return _file_bytes_sha256(strategy_bytes), canonical_sha256(parsed)


def _materialize_entry(
    bundle_root: str | os.PathLike[str],
    source_value: T3BBCheckpointBundleEntrySource | Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], tuple[str, str]]:
    source = _coerce_source(source_value, label=label)
    checkpoint_relative, checkpoint_file = _safe_asset(
        bundle_root, source.checkpoint_path, label=f"{label}.checkpoint_path"
    )
    strategy_relative, strategy_file = _safe_asset(
        bundle_root,
        source.average_strategy_json_path,
        label=f"{label}.average_strategy_json_path",
    )
    checkpoint_resolved = os.path.normcase(str(checkpoint_file))
    strategy_resolved = os.path.normcase(str(strategy_file))
    if checkpoint_resolved == strategy_resolved:
        raise T3BBCheckpointBundleError(
            f"{label}: checkpoint and strategy must be distinct files"
        )

    try:
        checkpoint_bytes = checkpoint_file.read_bytes()
        strategy_bytes = strategy_file.read_bytes()
    except OSError as exc:
        raise T3BBCheckpointBundleError(f"{label}: asset read failed: {exc}") from exc
    checkpoint_content_sha, expected_strategy = (
        _validate_checkpoint_and_reconstruct_strategy(
            checkpoint_bytes,
            solver_seed=source.solver_seed,
            observation_digest=source.observation_digest,
            range_content_sha256=source.range_content_sha256,
            range_build_sha256=source.range_build_sha256,
        )
    )
    strategy_bytes_sha, strategy_content_sha = _validate_strategy(
        strategy_bytes, expected_strategy=expected_strategy
    )
    entry: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        **source.key.as_dict(),
        "checkpoint_path": checkpoint_relative,
        "checkpoint_file_bytes_sha256": _file_bytes_sha256(checkpoint_bytes),
        "checkpoint_content_sha256": checkpoint_content_sha,
        "average_strategy_json_path": strategy_relative,
        "average_strategy_json_bytes_sha256": strategy_bytes_sha,
        "average_strategy_json_content_sha256": strategy_content_sha,
        "range_content_sha256": source.range_content_sha256,
        "range_build_sha256": source.range_build_sha256,
        "observation_digest": source.observation_digest,
        "solver_manifest_sha256": source.solver_manifest_sha256,
        "source_manifest_sha256": source.source_manifest_sha256,
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry, (checkpoint_resolved, strategy_resolved)


def _entry_key(entry: Mapping[str, Any], *, label: str) -> T3BBCheckpointKey:
    return _coerce_key(
        {field: entry.get(field) for field in _KEY_FIELDS}, label=label
    )


def _content_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: entry[field]
        for field in (
            "round_index",
            "root_id",
            "root_commitment_sha256",
            "solver_seed",
            "checkpoint_file_bytes_sha256",
            "checkpoint_content_sha256",
            "average_strategy_json_bytes_sha256",
            "average_strategy_json_content_sha256",
            "range_content_sha256",
            "range_build_sha256",
            "observation_digest",
            "solver_manifest_sha256",
            "source_manifest_sha256",
        )
    }


def _bundle_checkpoint_sha256(
    *,
    round_index: int,
    expected_key_set_sha256: str,
    entries: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_sha256(
        {
            "schema": BUNDLE_CONTENT_SCHEMA,
            "round_index": round_index,
            "expected_key_set_sha256": expected_key_set_sha256,
            "entries": [_content_entry(entry) for entry in entries],
        }
    )


def build_t3_bb_checkpoint_bundle(
    bundle_root: str | os.PathLike[str],
    entries: Iterable[T3BBCheckpointBundleEntrySource | Mapping[str, Any]],
    *,
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a bundle manifest from fresh physical file reads."""

    locked_keys = _normalize_expected_keys(expected_keys)
    if isinstance(entries, (str, bytes, Mapping)):
        raise T3BBCheckpointBundleError("entries: sequence of entry sources required")
    try:
        source_values = tuple(entries)
    except TypeError as exc:
        raise T3BBCheckpointBundleError("entries: iterable required") from exc
    source_keys = [
        _coerce_source(source, label=f"entries[{index}]").key
        for index, source in enumerate(source_values)
    ]
    if len(set(source_keys)) != len(source_keys):
        raise T3BBCheckpointBundleError("entries: duplicate coverage key")
    materialized: list[dict[str, Any]] = []
    resolved_assets: set[str] = set()
    for index, source in enumerate(source_values):
        entry, paths = _materialize_entry(
            bundle_root, source, label=f"entries[{index}]"
        )
        for path in paths:
            if path in resolved_assets:
                raise T3BBCheckpointBundleError(
                    "entries: duplicate physical asset path"
                )
            resolved_assets.add(path)
        materialized.append(entry)

    keys = [_entry_key(entry, label=f"entries[{index}]") for index, entry in enumerate(materialized)]
    if len(set(keys)) != len(keys):
        raise T3BBCheckpointBundleError("entries: duplicate coverage key")
    actual_set = set(keys)
    expected_set = set(locked_keys)
    if actual_set != expected_set:
        missing = [key.as_dict() for key in sorted(expected_set - actual_set)]
        unexpected = [key.as_dict() for key in sorted(actual_set - expected_set)]
        raise T3BBCheckpointBundleError(
            f"entries: exact coverage mismatch; missing={missing}, unexpected={unexpected}"
        )
    materialized.sort(key=lambda entry: _entry_key(entry, label="entry"))
    coverage_sha = _coverage_sha256(locked_keys)
    round_index = locked_keys[0].round_index
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "scope": SCOPE,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "round_index": round_index,
        "expected_key_set_sha256": coverage_sha,
        "entry_count": len(materialized),
        "entries": materialized,
        "bundle_checkpoint_sha256": _bundle_checkpoint_sha256(
            round_index=round_index,
            expected_key_set_sha256=coverage_sha,
            entries=materialized,
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return verify_t3_bb_checkpoint_bundle(
        manifest, bundle_root=bundle_root, expected_keys=locked_keys
    )


def verify_t3_bb_checkpoint_bundle(
    manifest: Any,
    *,
    bundle_root: str | os.PathLike[str],
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> dict[str, Any]:
    """Fresh-read every asset and exactly re-derive a bundle manifest."""

    locked_keys = _normalize_expected_keys(expected_keys)
    if not isinstance(manifest, Mapping) or set(manifest) != _BUNDLE_FIELDS:
        raise T3BBCheckpointBundleError("manifest: exact bundle fields required")
    expected_constants = {
        "schema": BUNDLE_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "scope": SCOPE,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "round_index": locked_keys[0].round_index,
    }
    for field, wanted in expected_constants.items():
        if manifest.get(field) != wanted:
            raise T3BBCheckpointBundleError(f"manifest.{field}: contract mismatch")
    coverage_sha = _coverage_sha256(locked_keys)
    if manifest.get("expected_key_set_sha256") != coverage_sha:
        raise T3BBCheckpointBundleError(
            "manifest.expected_key_set_sha256: locked coverage mismatch"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise T3BBCheckpointBundleError("manifest.entries: list required")
    if _require_int(manifest.get("entry_count"), label="manifest.entry_count") != len(entries):
        raise T3BBCheckpointBundleError("manifest.entry_count: mismatch")

    rebuilt: list[dict[str, Any]] = []
    seen_keys: set[T3BBCheckpointKey] = set()
    seen_assets: set[str] = set()
    prior_key: T3BBCheckpointKey | None = None
    for index, raw_entry in enumerate(entries):
        label = f"manifest.entries[{index}]"
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != _ENTRY_FIELDS:
            raise T3BBCheckpointBundleError(f"{label}: exact entry fields required")
        if raw_entry.get("schema") != ENTRY_SCHEMA:
            raise T3BBCheckpointBundleError(f"{label}.schema: contract mismatch")
        for field in (
            "checkpoint_file_bytes_sha256",
            "checkpoint_content_sha256",
            "average_strategy_json_bytes_sha256",
            "average_strategy_json_content_sha256",
            "range_content_sha256",
            "range_build_sha256",
            "observation_digest",
            "solver_manifest_sha256",
            "source_manifest_sha256",
            "entry_sha256",
        ):
            _require_sha256(raw_entry.get(field), label=f"{label}.{field}")
        key = _entry_key(raw_entry, label=label)
        if key in seen_keys:
            raise T3BBCheckpointBundleError(f"{label}: duplicate coverage key")
        seen_keys.add(key)
        if prior_key is not None and key <= prior_key:
            raise T3BBCheckpointBundleError(
                "manifest.entries: canonical key order required"
            )
        prior_key = key
        source = T3BBCheckpointBundleEntrySource(
            **key.as_dict(),
            checkpoint_path=raw_entry.get("checkpoint_path"),
            average_strategy_json_path=raw_entry.get("average_strategy_json_path"),
            range_content_sha256=raw_entry.get("range_content_sha256"),
            range_build_sha256=raw_entry.get("range_build_sha256"),
            observation_digest=raw_entry.get("observation_digest"),
            solver_manifest_sha256=raw_entry.get("solver_manifest_sha256"),
            source_manifest_sha256=raw_entry.get("source_manifest_sha256"),
        )
        rebuilt_entry, paths = _materialize_entry(
            bundle_root, source, label=label
        )
        for path in paths:
            if path in seen_assets:
                raise T3BBCheckpointBundleError(
                    "manifest.entries: duplicate physical asset path"
                )
            seen_assets.add(path)
        if dict(raw_entry) != rebuilt_entry:
            raise T3BBCheckpointBundleError(
                f"{label}: fresh file-derived entry mismatch"
            )
        rebuilt.append(rebuilt_entry)

    actual_keys = tuple(_entry_key(entry, label="entry") for entry in rebuilt)
    if actual_keys != locked_keys:
        missing = [key.as_dict() for key in sorted(set(locked_keys) - set(actual_keys))]
        unexpected = [key.as_dict() for key in sorted(set(actual_keys) - set(locked_keys))]
        raise T3BBCheckpointBundleError(
            "manifest.entries: exact expected coverage missing or has a gap; "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_bundle_sha = _bundle_checkpoint_sha256(
        round_index=locked_keys[0].round_index,
        expected_key_set_sha256=coverage_sha,
        entries=rebuilt,
    )
    if manifest.get("bundle_checkpoint_sha256") != expected_bundle_sha:
        raise T3BBCheckpointBundleError(
            "manifest.bundle_checkpoint_sha256: fresh content mismatch"
        )
    unsigned = dict(manifest)
    claimed_manifest_sha = _require_sha256(
        unsigned.pop("manifest_sha256", None), label="manifest.manifest_sha256"
    )
    if canonical_sha256(unsigned) != claimed_manifest_sha:
        raise T3BBCheckpointBundleError("manifest.manifest_sha256: self-hash mismatch")
    return json.loads(canonical_json(manifest))


def _atomic_write(path: Path, payload: bytes) -> None:
    if not path.name:
        raise T3BBCheckpointBundleError("manifest_path: file name required")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except (AttributeError, OSError):
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_t3_bb_checkpoint_bundle_manifest(
    manifest_path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    bundle_root: str | os.PathLike[str],
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically publish a prebuilt manifest and verify disk readback."""

    locked_keys = _normalize_expected_keys(expected_keys)
    verified = verify_t3_bb_checkpoint_bundle(
        manifest, bundle_root=bundle_root, expected_keys=locked_keys
    )
    target = Path(manifest_path)
    payload = canonical_json(verified).encode("utf-8") + b"\n"
    _atomic_write(target, payload)
    readback = read_t3_bb_checkpoint_bundle(
        target, bundle_root=bundle_root, expected_keys=locked_keys
    )
    if readback != verified:
        raise T3BBCheckpointBundleError("manifest readback differs from written value")
    return readback


def write_t3_bb_checkpoint_bundle(
    bundle_root: str | os.PathLike[str],
    entries: Iterable[T3BBCheckpointBundleEntrySource | Mapping[str, Any]],
    *,
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
    manifest_name: str = "manifest.json",
) -> dict[str, Any]:
    """Build and atomically publish a bundle manifest under ``bundle_root``."""

    if (
        not isinstance(manifest_name, str)
        or not manifest_name
        or PurePosixPath(manifest_name).name != manifest_name
        or "\\" in manifest_name
        or ":" in manifest_name
    ):
        raise T3BBCheckpointBundleError("manifest_name: plain file name required")
    locked_keys = _normalize_expected_keys(expected_keys)
    manifest = build_t3_bb_checkpoint_bundle(
        bundle_root, entries, expected_keys=locked_keys
    )
    return write_t3_bb_checkpoint_bundle_manifest(
        Path(bundle_root) / manifest_name,
        manifest,
        bundle_root=bundle_root,
        expected_keys=locked_keys,
    )


def read_t3_bb_checkpoint_bundle(
    manifest_path: str | os.PathLike[str],
    *,
    bundle_root: str | os.PathLike[str] | None = None,
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> dict[str, Any]:
    """Read a canonical manifest and fresh-verify all physical assets."""

    path = Path(manifest_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise T3BBCheckpointBundleError(f"manifest read failed: {exc}") from exc
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise T3BBCheckpointBundleError(
            "manifest: canonical JSON plus one trailing newline required"
        )
    parsed = _parse_json_bytes(payload, label="manifest")
    if canonical_json(parsed).encode("utf-8") + b"\n" != payload:
        raise T3BBCheckpointBundleError("manifest: non-canonical disk encoding")
    return verify_t3_bb_checkpoint_bundle(
        parsed,
        bundle_root=path.parent if bundle_root is None else bundle_root,
        expected_keys=expected_keys,
    )


def load_t3_bb_checkpoint_bundle_root_policies(
    manifest: Any,
    *,
    bundle_root: str | os.PathLike[str],
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> dict[T3BBCheckpointKey, dict[str, float]]:
    """Return fresh-verified root action distributions for fixed-point rows.

    This deliberately verifies the complete bundle before exposing any
    distribution.  It then fresh-reads each strategy asset again and selects
    the unique record whose information-set digest equals the entry's locked
    observation digest.  Consumers can therefore bind a fixed-point row's
    ``current_policy_distribution`` to actual bundle bytes, not to a copied
    hash or summary.
    """

    locked_keys = _normalize_expected_keys(expected_keys)
    verified = verify_t3_bb_checkpoint_bundle(
        manifest, bundle_root=bundle_root, expected_keys=locked_keys
    )
    policies: dict[T3BBCheckpointKey, dict[str, float]] = {}
    for index, entry in enumerate(verified["entries"]):
        label = f"manifest.entries[{index}].average_strategy_json_path"
        _relative, strategy_file = _safe_asset(
            bundle_root, entry["average_strategy_json_path"], label=label
        )
        try:
            strategy_bytes = strategy_file.read_bytes()
        except OSError as exc:
            raise T3BBCheckpointBundleError(f"{label}: asset read failed: {exc}") from exc
        if _file_bytes_sha256(strategy_bytes) != entry[
            "average_strategy_json_bytes_sha256"
        ]:
            raise T3BBCheckpointBundleError(f"{label}: fresh bytes hash mismatch")
        strategy = _parse_json_bytes(strategy_bytes, label=label)
        if not isinstance(strategy, Mapping) or canonical_sha256(strategy) != entry[
            "average_strategy_json_content_sha256"
        ]:
            raise T3BBCheckpointBundleError(f"{label}: fresh content hash mismatch")
        matching = [
            record
            for record in strategy.get("records", [])
            if isinstance(record, Mapping)
            and record.get("infoset_digest") == entry["observation_digest"]
        ]
        if len(matching) != 1:
            raise T3BBCheckpointBundleError(
                f"{label}: unique root-observation strategy record required"
            )
        actions = matching[0].get("actions")
        if not isinstance(actions, list) or not actions:
            raise T3BBCheckpointBundleError(f"{label}: root actions required")
        distribution: dict[str, float] = {}
        for action in actions:
            if not isinstance(action, Mapping) or set(action) != {
                "action_id",
                "probability",
            }:
                raise T3BBCheckpointBundleError(
                    f"{label}: exact root action fields required"
                )
            action_id = action.get("action_id")
            probability = action.get("probability")
            if (
                not isinstance(action_id, str)
                or not action_id
                or action_id in distribution
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or float(probability) < 0.0
            ):
                raise T3BBCheckpointBundleError(
                    f"{label}: invalid root action distribution"
                )
            distribution[action_id] = float(probability)
        if not math.isclose(
            math.fsum(distribution.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise T3BBCheckpointBundleError(
                f"{label}: root action probabilities must sum to one"
            )
        key = _entry_key(entry, label=f"manifest.entries[{index}]")
        policies[key] = distribution
    if tuple(sorted(policies)) != locked_keys:
        raise T3BBCheckpointBundleError("root policy coverage mismatch")
    return policies


def load_verified_t3_bb_checkpoint_strategy_profiles(
    manifest: Any,
    *,
    bundle_root: str | os.PathLike[str],
    expected_keys: Iterable[T3BBCheckpointKey | Mapping[str, Any]],
) -> dict[T3BBCheckpointKey, dict[str, dict[str, Fraction]]]:
    """Load every fresh-verified strategy table as exact decimal rationals.

    The returned outer key is the bundle coverage identity.  The next key is
    the information-set digest, followed by the stable action identifier.
    ``Fraction(str(probability))`` preserves the exact decimal number committed
    by canonical JSON instead of introducing a second binary-float rounding.
    The round-wide identity that a runtime candidate artifact must bind is
    ``manifest["bundle_checkpoint_sha256"]``.
    """

    locked_keys = _normalize_expected_keys(expected_keys)
    verified = verify_t3_bb_checkpoint_bundle(
        manifest, bundle_root=bundle_root, expected_keys=locked_keys
    )
    profiles: dict[T3BBCheckpointKey, dict[str, dict[str, Fraction]]] = {}
    for index, entry in enumerate(verified["entries"]):
        label = f"manifest.entries[{index}].average_strategy_json_path"
        _relative, strategy_file = _safe_asset(
            bundle_root, entry["average_strategy_json_path"], label=label
        )
        try:
            strategy_bytes = strategy_file.read_bytes()
        except OSError as exc:
            raise T3BBCheckpointBundleError(f"{label}: asset read failed: {exc}") from exc
        if _file_bytes_sha256(strategy_bytes) != entry[
            "average_strategy_json_bytes_sha256"
        ]:
            raise T3BBCheckpointBundleError(f"{label}: fresh bytes hash mismatch")
        strategy = _parse_json_bytes(strategy_bytes, label=label)
        if not isinstance(strategy, Mapping) or canonical_sha256(strategy) != entry[
            "average_strategy_json_content_sha256"
        ]:
            raise T3BBCheckpointBundleError(f"{label}: fresh content hash mismatch")
        records = strategy.get("records")
        if not isinstance(records, list) or not records:
            raise T3BBCheckpointBundleError(f"{label}: strategy records required")
        profile: dict[str, dict[str, Fraction]] = {}
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "infoset_digest",
                "infoset",
                "actions",
            }:
                raise T3BBCheckpointBundleError(
                    f"{label}: exact strategy record fields required"
                )
            digest = _require_sha256(
                record.get("infoset_digest"), label=f"{label}.infoset_digest"
            )
            if digest in profile:
                raise T3BBCheckpointBundleError(f"{label}: duplicate infoset digest")
            actions = record.get("actions")
            if not isinstance(actions, list) or not actions:
                raise T3BBCheckpointBundleError(f"{label}: strategy actions required")
            distribution: dict[str, Fraction] = {}
            for action in actions:
                if not isinstance(action, Mapping) or set(action) != {
                    "action_id",
                    "probability",
                }:
                    raise T3BBCheckpointBundleError(
                        f"{label}: exact strategy action fields required"
                    )
                action_id = action.get("action_id")
                probability = action.get("probability")
                if (
                    not isinstance(action_id, str)
                    or not action_id
                    or action_id in distribution
                    or isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(float(probability))
                    or float(probability) < 0.0
                ):
                    raise T3BBCheckpointBundleError(
                        f"{label}: invalid strategy action distribution"
                    )
                distribution[action_id] = Fraction(str(probability))
            if not math.isclose(
                math.fsum(float(value) for value in distribution.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise T3BBCheckpointBundleError(
                    f"{label}: strategy probabilities must sum to one"
                )
            profile[digest] = distribution
        key = _entry_key(entry, label=f"manifest.entries[{index}]")
        profiles[key] = profile
    if tuple(sorted(profiles)) != locked_keys:
        raise T3BBCheckpointBundleError("strategy profile coverage mismatch")
    return profiles


__all__ = [
    "ARTIFACT_KIND",
    "BUNDLE_CONTENT_SCHEMA",
    "BUNDLE_SCHEMA",
    "COVERAGE_SCHEMA",
    "ENTRY_SCHEMA",
    "SCOPE",
    "T3BBCheckpointBundleEntrySource",
    "T3BBCheckpointBundleError",
    "T3BBCheckpointKey",
    "build_t3_bb_checkpoint_bundle",
    "canonical_json",
    "canonical_sha256",
    "load_t3_bb_checkpoint_bundle_root_policies",
    "load_verified_t3_bb_checkpoint_strategy_profiles",
    "read_t3_bb_checkpoint_bundle",
    "verify_t3_bb_checkpoint_bundle",
    "write_t3_bb_checkpoint_bundle",
    "write_t3_bb_checkpoint_bundle_manifest",
]
