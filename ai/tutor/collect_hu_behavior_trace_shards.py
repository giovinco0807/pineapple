"""Append-only sharded HU behavior trace collection.

This module is deliberately a storage/orchestration layer over
``collect_hu_behavior_traces``.  Every published ``shard-XXXXXX`` directory is
one independently readable and verifiable core dataset.  The top-level
manifest is the commit marker and binds the ordered shard layout while also
carrying shard-boundary-independent streaming commitments to every decision
and hidden root.

Raw trace collections are never promotion evidence.  This module does not
merge shard rows into a monolithic JSONL file and never mutates a published
shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import ai.tutor.behavior_calibration_contract as behavior_contract_module
import ai.tutor.collect_hu_behavior_traces as core_collector_module
from ai.tutor.behavior_calibration_contract import (
    SPLIT_NAMES,
    canonical_json,
    canonical_sha256,
    commit_hidden_root_trace,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
    CollectedBehaviorTraces,
    _model_identity,
    collect_hu_behavior_traces,
    read_behavior_trace_dataset,
    write_behavior_trace_dataset,
)
from ai.tutor.t3_hu_full_card_range import FrozenBehaviorModel


SHARDED_COLLECTION_SCHEMA = "ofc_behavior_trace_sharded_collection/v1"
SHARD_ENTRY_SCHEMA = "ofc_behavior_trace_shard_entry/v1"
STREAM_CHAIN_SCHEMA = "ofc_behavior_trace_stream_chain/v1"
CLI_RESULT_SCHEMA = "ofc_behavior_trace_sharded_cli_result/v1"
TOP_MANIFEST_NAME = "manifest.json"
DECISIONS_NAME = "decisions.jsonl"
ROOTS_NAME = "roots.jsonl"
SHARD_MANIFEST_NAME = "manifest.json"
_SHARD_RE = re.compile(r"^shard-([0-9]{6})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CELL_KEYS = tuple(
    f"t{turn}_{actor}_joker_{joker}"
    for turn in (1, 2)
    for actor in ("bb", "btn")
    for joker in (0, 1, 2)
)
_CHALLENGE_CELL_KEYS = tuple(
    f"t{turn}_{actor}_joker_{joker}"
    for turn in (1, 2)
    for actor in ("bb", "btn")
    for joker in (0, 1, 2)
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "shard_orchestrator_source_sha256": _sha256_file(__file__),
        "core_collector_source_sha256": _sha256_file(core_collector_module.__file__),
        "behavior_contract_source_sha256": _sha256_file(
            behavior_contract_module.__file__
        ),
    }


def _chain_genesis(stream: str) -> str:
    return hashlib.sha256(
        canonical_json(
            {"schema": STREAM_CHAIN_SCHEMA, "stream": stream, "genesis": True}
        ).encode("utf-8")
    ).hexdigest()


def _chain_step(previous: str, value: Any) -> str:
    if not _SHA256_RE.fullmatch(previous):
        raise ValueError("stream chain state is not a SHA-256")
    return hashlib.sha256(
        bytes.fromhex(previous) + canonical_json(value).encode("utf-8")
    ).hexdigest()


def _empty_census() -> dict[str, Any]:
    return {
        "decision_counts_by_cell": {key: 0 for key in _CELL_KEYS},
        "decision_counts_by_split": {split: 0 for split in SPLIT_NAMES},
        "root_counts_by_split": {split: 0 for split in SPLIT_NAMES},
        "btn_seat_root_counts": {"0": 0, "1": 0},
        "challenge_target_root_counts": {
            key: 0 for key in _CHALLENGE_CELL_KEYS
        },
    }


def _empty_commitments() -> dict[str, str]:
    return {
        "decision_record_order_chain_sha256": _chain_genesis("decision_record_sha256"),
        "root_id_order_chain_sha256": _chain_genesis("root_id"),
        "hidden_root_commitment_order_chain_sha256": _chain_genesis(
            "hidden_root_commitment"
        ),
        "root_index_order_chain_sha256": _chain_genesis("root_index"),
    }


@dataclass
class _Aggregate:
    commitments: dict[str, str]
    census: dict[str, Any]
    root_count: int = 0
    decision_count: int = 0
    policy_query_count: int = 0
    model_evaluation_count: int = 0
    elapsed_collection_runtime_ns: int = 0
    shard_artifact_bytes: int = 0

    @classmethod
    def empty(cls) -> "_Aggregate":
        return cls(commitments=_empty_commitments(), census=_empty_census())

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "_Aggregate":
        counters = manifest["counters"]
        return cls(
            commitments=dict(manifest["global_content_commitments"]),
            census=json.loads(canonical_json(manifest["census"])),
            root_count=counters["root_count"],
            decision_count=counters["decision_count"],
            policy_query_count=counters["policy_query_count"],
            model_evaluation_count=counters["model_evaluation_count"],
            elapsed_collection_runtime_ns=counters[
                "elapsed_collection_runtime_ns"
            ],
            shard_artifact_bytes=counters["shard_artifact_bytes"],
        )

    def add_dataset(
        self, dataset: CollectedBehaviorTraces, *, artifact_bytes: int
    ) -> None:
        seen_root_splits: dict[str, str] = {}
        for record in dataset.records:
            turn = record["turn"]
            actor = record["actor"]
            joker = record["visible_joker_count"]
            cell = f"t{turn}_{actor}_joker_{joker}"
            if cell not in self.census["decision_counts_by_cell"]:
                raise ValueError("decision lies outside the T1/T2 role/Joker census")
            split = record["split"]
            if split not in SPLIT_NAMES:
                raise ValueError("decision has an unknown locked split")
            root_id = record["root_id"]
            prior = seen_root_splits.setdefault(root_id, split)
            if prior != split:
                raise ValueError("one root crosses locked splits")
            self.census["decision_counts_by_cell"][cell] += 1
            self.census["decision_counts_by_split"][split] += 1
            self.commitments["decision_record_order_chain_sha256"] = _chain_step(
                self.commitments["decision_record_order_chain_sha256"],
                record["record_sha256"],
            )

        for trace in dataset.hidden_roots:
            root_id = trace["root_id"]
            split = seen_root_splits.get(root_id)
            if split is None:
                raise ValueError("hidden root has no decision rows")
            self.census["root_counts_by_split"][split] += 1
            btn_seat = str(trace["seat_assignment"]["btn"])
            if btn_seat not in ("0", "1"):
                raise ValueError("hidden root BTN seat is outside HU seats")
            self.census["btn_seat_root_counts"][btn_seat] += 1
            target = trace["challenge_target"]
            if target is not None:
                target_cell = (
                    f"t{target['turn']}_{target['actor']}_"
                    f"joker_{target['visible_joker_count']}"
                )
                if target_cell not in self.census["challenge_target_root_counts"]:
                    raise ValueError("hidden root challenge target is outside the census")
                self.census["challenge_target_root_counts"][target_cell] += 1
            self.commitments["root_id_order_chain_sha256"] = _chain_step(
                self.commitments["root_id_order_chain_sha256"], root_id
            )
            self.commitments[
                "hidden_root_commitment_order_chain_sha256"
            ] = _chain_step(
                self.commitments[
                    "hidden_root_commitment_order_chain_sha256"
                ],
                commit_hidden_root_trace(trace),
            )
            self.commitments["root_index_order_chain_sha256"] = _chain_step(
                self.commitments["root_index_order_chain_sha256"],
                trace["root_index"],
            )

        self.root_count += dataset.root_count
        self.decision_count += len(dataset.records)
        self.policy_query_count += dataset.manifest["policy_query_count"]
        self.model_evaluation_count += dataset.manifest["model_evaluation_count"]
        self.elapsed_collection_runtime_ns += dataset.elapsed_runtime_ns
        self.shard_artifact_bytes += artifact_bytes


@dataclass(frozen=True)
class ShardedBehaviorTraceCollection:
    output_dir: Path
    manifest: dict[str, Any]
    config: BehaviorTraceCollectionConfig
    verification_runtime_ns: int

    @property
    def root_count(self) -> int:
        return self.manifest["counters"]["root_count"]

    @property
    def decision_count(self) -> int:
        return self.manifest["counters"]["decision_count"]

    @property
    def shard_count(self) -> int:
        return len(self.manifest["shards"])

    @property
    def collection_complete(self) -> bool:
        return self.manifest["collection_complete"]


def _read_top_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise ValueError("top manifest must be one canonical JSON object with trailing newline")
    value = json.loads(raw[:-1])
    if not isinstance(value, dict) or canonical_json(value) != raw[:-1]:
        raise ValueError("top manifest is not canonical JSON")
    return value


def _atomic_write_top_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(manifest)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # This top-level manifest is always the final publication marker.
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _shard_name(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 999999:
        raise ValueError("shard index must fit the six-digit shard namespace")
    return f"shard-{index:06d}"


def _artifact_sizes(shard_dir: Path) -> dict[str, int]:
    sizes = {
        "decisions_bytes": (shard_dir / DECISIONS_NAME).stat().st_size,
        "hidden_roots_bytes": (shard_dir / ROOTS_NAME).stat().st_size,
        "shard_manifest_bytes": (shard_dir / SHARD_MANIFEST_NAME).stat().st_size,
    }
    sizes["total_bytes"] = sum(sizes.values())
    return sizes


def _build_shard_entry(
    *, index: int, shard_dir: Path, dataset: CollectedBehaviorTraces
) -> dict[str, Any]:
    sizes = _artifact_sizes(shard_dir)
    shard_aggregate = _Aggregate.empty()
    shard_aggregate.add_dataset(dataset, artifact_bytes=sizes["total_bytes"])
    entry = {
        "schema": SHARD_ENTRY_SCHEMA,
        "index": index,
        "name": _shard_name(index),
        "root_index_start": dataset.root_index_start,
        "root_index_stop_exclusive": dataset.root_index_stop_exclusive,
        "root_count": dataset.root_count,
        "decision_count": len(dataset.records),
        "policy_query_count": dataset.manifest["policy_query_count"],
        "model_evaluation_count": dataset.manifest["model_evaluation_count"],
        "elapsed_collection_runtime_ns": dataset.elapsed_runtime_ns,
        "collection_content_sha256": dataset.manifest[
            "collection_content_sha256"
        ],
        "shard_manifest_sha256": dataset.manifest["manifest_sha256"],
        "record_order_sha256": dataset.manifest["record_order_sha256"],
        "root_id_order_sha256": dataset.manifest["root_id_order_sha256"],
        "root_commitment_order_sha256": dataset.manifest["hidden_root_artifact"][
            "root_commitment_order_sha256"
        ],
        "census": shard_aggregate.census,
        "decisions_file_sha256": _sha256_file(shard_dir / DECISIONS_NAME),
        "hidden_roots_file_sha256": _sha256_file(shard_dir / ROOTS_NAME),
        "shard_manifest_file_sha256": _sha256_file(
            shard_dir / SHARD_MANIFEST_NAME
        ),
        "artifact_sizes": sizes,
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry


def _collection_content_payload(
    *,
    config: BehaviorTraceCollectionConfig,
    policy: Mapping[str, str],
    aggregate: _Aggregate,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": SHARDED_COLLECTION_SCHEMA,
        "promotion_eligible": False,
        "collection_config": config.to_canonical_dict(),
        "policy": dict(policy),
        "root_index_start": 0,
        "root_index_stop_exclusive": aggregate.root_count,
        "root_count": aggregate.root_count,
        "decision_count": aggregate.decision_count,
        "global_content_commitments": aggregate.commitments,
        "census": aggregate.census,
        "source_hashes": dict(source_hashes),
    }


def _build_top_manifest(
    *,
    config: BehaviorTraceCollectionConfig,
    policy: Mapping[str, str],
    shard_size: int,
    requested_total_root_target: int,
    shards: Sequence[Mapping[str, Any]],
    aggregate: _Aggregate,
) -> dict[str, Any]:
    source_hashes = _source_hashes()
    content_payload = _collection_content_payload(
        config=config,
        policy=policy,
        aggregate=aggregate,
        source_hashes=source_hashes,
    )
    counters = {
        "root_count": aggregate.root_count,
        "decision_count": aggregate.decision_count,
        "policy_query_count": aggregate.policy_query_count,
        "model_evaluation_count": aggregate.model_evaluation_count,
        "elapsed_collection_runtime_ns": aggregate.elapsed_collection_runtime_ns,
        "shard_artifact_bytes": aggregate.shard_artifact_bytes,
    }
    manifest: dict[str, Any] = {
        "schema": SHARDED_COLLECTION_SCHEMA,
        "promotion_eligible": False,
        "raw_collection_only": True,
        "collection_config": config.to_canonical_dict(),
        "policy": dict(policy),
        "shard_size": shard_size,
        "requested_total_root_target": requested_total_root_target,
        "collection_complete": aggregate.root_count == requested_total_root_target,
        "root_index_start": 0,
        "root_index_stop_exclusive": aggregate.root_count,
        "counters": counters,
        "census": aggregate.census,
        "global_content_commitments": aggregate.commitments,
        "source_hashes": source_hashes,
        "shards": [dict(entry) for entry in shards],
        "collection_content_sha256": canonical_sha256(content_payload),
    }
    manifest["ordered_shard_entry_chain_sha256"] = _ordered_shard_chain(shards)
    manifest["layout_sha256"] = canonical_sha256(
        {
            "shard_size": shard_size,
            "requested_total_root_target": requested_total_root_target,
            "collection_complete": manifest["collection_complete"],
            "shards": manifest["shards"],
            "counters": counters,
            "collection_content_sha256": manifest["collection_content_sha256"],
        }
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _ordered_shard_chain(shards: Sequence[Mapping[str, Any]]) -> str:
    chain = _chain_genesis("ordered_shard_entry_sha256")
    for entry in shards:
        chain = _chain_step(chain, entry["entry_sha256"])
    return chain


def _verify_manifest_envelope(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "promotion_eligible",
        "raw_collection_only",
        "collection_config",
        "policy",
        "shard_size",
        "requested_total_root_target",
        "collection_complete",
        "root_index_start",
        "root_index_stop_exclusive",
        "counters",
        "census",
        "global_content_commitments",
        "source_hashes",
        "shards",
        "collection_content_sha256",
        "ordered_shard_entry_chain_sha256",
        "layout_sha256",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("top manifest keys mismatch")
    if (
        manifest["schema"] != SHARDED_COLLECTION_SCHEMA
        or manifest["promotion_eligible"] is not False
        or manifest["raw_collection_only"] is not True
    ):
        raise ValueError("unsupported or promotable sharded raw collection")
    recorded = manifest["manifest_sha256"]
    if not isinstance(recorded, str) or not _SHA256_RE.fullmatch(recorded):
        raise ValueError("top manifest requires a lowercase SHA-256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    if canonical_sha256(unsigned) != recorded:
        raise ValueError("top manifest SHA-256 mismatch")


def _finalized_shard_directories(output_dir: Path) -> set[str]:
    names: set[str] = set()
    if not output_dir.exists():
        return names
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = _SHARD_RE.fullmatch(child.name)
        if match:
            names.add(child.name)
        elif child.name.startswith("shard-"):
            raise ValueError(f"unexpected malformed finalized shard directory: {child.name}")
        # Dot-prefixed *.partial staging directories are intentionally ignored.
    return names


def read_sharded_behavior_trace_collection(
    output_dir: str | Path,
    *,
    behavior_model: FrozenBehaviorModel | None = None,
) -> ShardedBehaviorTraceCollection:
    """Verify the top marker and every referenced immutable shard."""
    started = time.perf_counter_ns()
    root = Path(output_dir).resolve()
    manifest_path = root / TOP_MANIFEST_NAME
    manifest = _read_top_manifest(manifest_path)
    _verify_manifest_envelope(manifest)

    config = BehaviorTraceCollectionConfig.from_canonical_dict(
        manifest["collection_config"]
    )
    shard_size = manifest["shard_size"]
    target = manifest["requested_total_root_target"]
    if (
        isinstance(shard_size, bool)
        or not isinstance(shard_size, int)
        or shard_size <= 0
    ):
        raise ValueError("top manifest shard_size must be positive")
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("top manifest total-root target must be positive")
    if manifest["source_hashes"] != _source_hashes():
        raise ValueError("collector source hash binding mismatch")
    policy = manifest["policy"]
    if set(policy) != {"model_id", "model_sha256"}:
        raise ValueError("top manifest policy keys mismatch")
    if behavior_model is not None:
        model_id, model_sha = _model_identity(behavior_model)
        if policy != {"model_id": model_id, "model_sha256": model_sha}:
            raise ValueError("resume behavior policy does not match sharded collection")

    raw_shards = manifest["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("sharded collection must contain at least one shard")
    expected_names: set[str] = set()
    expected_start = 0
    aggregate = _Aggregate.empty()
    verified_entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_shards):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("shard entry must be an object")
        entry = dict(raw_entry)
        recorded_entry_hash = entry.get("entry_sha256")
        unsigned_entry = dict(entry)
        unsigned_entry.pop("entry_sha256", None)
        if canonical_sha256(unsigned_entry) != recorded_entry_hash:
            raise ValueError("shard entry SHA-256 mismatch")
        name = _shard_name(index)
        if entry.get("schema") != SHARD_ENTRY_SCHEMA:
            raise ValueError("unsupported shard entry schema")
        if entry.get("index") != index or entry.get("name") != name:
            raise ValueError("duplicate/out-of-order shard index or name")
        start = entry.get("root_index_start")
        stop = entry.get("root_index_stop_exclusive")
        if (
            start != expected_start
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or stop <= start
            or stop - start > shard_size
        ):
            raise ValueError("shard ranges contain a gap, overlap, or invalid size")
        expected_names.add(name)
        shard_dir = root / name
        dataset = read_behavior_trace_dataset(shard_dir / DECISIONS_NAME)
        if dataset.config != config:
            raise ValueError("shard collection config does not match top manifest")
        if dataset.manifest["policy"] != policy:
            raise ValueError("shard policy does not match top manifest")
        if (
            dataset.root_index_start != start
            or dataset.root_index_stop_exclusive != stop
        ):
            raise ValueError("shard dataset range does not match shard entry")
        rebuilt_entry = _build_shard_entry(
            index=index, shard_dir=shard_dir, dataset=dataset
        )
        if rebuilt_entry != entry:
            raise ValueError("shard entry does not match its immutable artifacts")
        aggregate.add_dataset(
            dataset, artifact_bytes=entry["artifact_sizes"]["total_bytes"]
        )
        verified_entries.append(entry)
        expected_start = stop

    actual_names = _finalized_shard_directories(root)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        orphan = sorted(actual_names - expected_names)
        raise ValueError(
            f"finalized shard directory set mismatch; missing={missing}, orphan={orphan}"
        )
    if expected_start != aggregate.root_count:
        raise ValueError("aggregate root count does not equal contiguous shard range")
    rebuilt = _build_top_manifest(
        config=config,
        policy=policy,
        shard_size=shard_size,
        requested_total_root_target=target,
        shards=verified_entries,
        aggregate=aggregate,
    )
    if rebuilt != manifest:
        raise ValueError("top manifest does not match verified shard artifacts")
    if manifest["root_index_start"] != 0 or manifest[
        "root_index_stop_exclusive"
    ] != aggregate.root_count:
        raise ValueError("top manifest range does not match shards")
    if target < aggregate.root_count:
        raise ValueError("top manifest target is below committed root count")
    if manifest["collection_complete"] is not (target == aggregate.root_count):
        raise ValueError("top manifest completion flag mismatch")
    return ShardedBehaviorTraceCollection(
        output_dir=root,
        manifest=dict(manifest),
        config=config,
        verification_runtime_ns=time.perf_counter_ns() - started,
    )


def _publish_one_shard(
    *,
    output_dir: Path,
    shard_index: int,
    dataset: CollectedBehaviorTraces,
) -> tuple[dict[str, Any], int]:
    name = _shard_name(shard_index)
    final_dir = output_dir / name
    if final_dir.exists():
        raise FileExistsError(f"published shard is immutable: {final_dir}")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{name}.", suffix=".partial", dir=output_dir)
    )
    try:
        write_behavior_trace_dataset(dataset, staging_dir / DECISIONS_NAME)
        # The directory appears under its finalized name only after the shard's
        # own manifest-last write and readback verification both succeeded.
        os.replace(staging_dir, final_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    entry = _build_shard_entry(
        index=shard_index, shard_dir=final_dir, dataset=dataset
    )
    return entry, entry["artifact_sizes"]["total_bytes"]


def collect_sharded_behavior_traces(
    output_dir: str | Path,
    config: BehaviorTraceCollectionConfig,
    behavior_model: FrozenBehaviorModel,
    *,
    total_root_target: int,
    shard_size: int,
    resume: bool = False,
    forced_decks: Mapping[int, Sequence[str]] | None = None,
) -> ShardedBehaviorTraceCollection:
    """Create or append exact contiguous shards up to ``total_root_target``."""
    if (
        isinstance(total_root_target, bool)
        or not isinstance(total_root_target, int)
        or total_root_target <= 0
    ):
        raise ValueError("total_root_target must be a positive integer")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / TOP_MANIFEST_NAME
    model_id, model_sha = _model_identity(behavior_model)
    policy = {"model_id": model_id, "model_sha256": model_sha}

    if resume:
        existing = read_sharded_behavior_trace_collection(
            root, behavior_model=behavior_model
        )
        if existing.config != config:
            raise ValueError("resume config does not match sharded collection")
        if existing.manifest["shard_size"] != shard_size:
            raise ValueError("resume shard size does not match sharded collection")
        prior_target = existing.manifest["requested_total_root_target"]
        current = existing.root_count
        if total_root_target < current or (
            not existing.collection_complete and total_root_target < prior_target
        ):
            raise ValueError("resume cannot shrink committed or outstanding root target")
        shards = [dict(entry) for entry in existing.manifest["shards"]]
        aggregate = _Aggregate.from_manifest(existing.manifest)
    else:
        if manifest_path.exists():
            raise FileExistsError("top manifest already exists; use resume")
        finalized = _finalized_shard_directories(root)
        if finalized:
            raise ValueError(f"output contains orphan finalized shards: {sorted(finalized)}")
        current = 0
        shards = []
        aggregate = _Aggregate.empty()

    forced = dict(forced_decks or {})
    expected_forced = set(range(current, total_root_target))
    if config.challenge_deck_source == core_collector_module.CALLER_SUPPLIED_DECKS:
        if set(forced) != expected_forced:
            raise ValueError("caller-supplied sharded collection needs every new root deck")
    elif forced:
        raise ValueError("forced decks are only allowed by caller-supplied challenge config")

    while current < total_root_target:
        count = min(shard_size, total_root_target - current)
        shard_forced = (
            {index: forced[index] for index in range(current, current + count)}
            if forced
            else None
        )
        dataset = collect_hu_behavior_traces(
            config,
            behavior_model,
            root_count=count,
            root_index_start=current,
            forced_decks=shard_forced,
        )
        entry, artifact_bytes = _publish_one_shard(
            output_dir=root,
            shard_index=len(shards),
            dataset=dataset,
        )
        shards.append(entry)
        aggregate.add_dataset(dataset, artifact_bytes=artifact_bytes)
        current += count
        top = _build_top_manifest(
            config=config,
            policy=policy,
            shard_size=shard_size,
            requested_total_root_target=total_root_target,
            shards=shards,
            aggregate=aggregate,
        )
        _atomic_write_top_manifest(manifest_path, top)

    if not manifest_path.exists():
        # A verified no-op resume can update an old outstanding target only by
        # explicitly republishing the top marker; published shards stay untouched.
        raise RuntimeError("no top manifest was published")
    if resume and current == total_root_target:
        prior = _read_top_manifest(manifest_path)
        if prior["requested_total_root_target"] != total_root_target:
            top = _build_top_manifest(
                config=config,
                policy=policy,
                shard_size=shard_size,
                requested_total_root_target=total_root_target,
                shards=shards,
                aggregate=aggregate,
            )
            _atomic_write_top_manifest(manifest_path, top)
    return read_sharded_behavior_trace_collection(root, behavior_model=behavior_model)


def resume_sharded_behavior_trace_collection(
    output_dir: str | Path,
    behavior_model: FrozenBehaviorModel,
    *,
    total_root_target: int,
    shard_size: int,
    expected_config: BehaviorTraceCollectionConfig | None = None,
    forced_decks: Mapping[int, Sequence[str]] | None = None,
) -> ShardedBehaviorTraceCollection:
    root = Path(output_dir).resolve()
    stored = _read_top_manifest(root / TOP_MANIFEST_NAME)
    stored_config = BehaviorTraceCollectionConfig.from_canonical_dict(
        stored["collection_config"]
    )
    config = expected_config if expected_config is not None else stored_config
    return collect_sharded_behavior_traces(
        root,
        config,
        behavior_model,
        total_root_target=total_root_target,
        shard_size=shard_size,
        resume=True,
        forced_decks=forced_decks,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-root-target", type=int, required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--seed-namespace")
    parser.add_argument(
        "--root-sampling-mode",
        choices=(NATURAL_UNIFORM_SHUFFLE, TARGETED_JOKER_CHALLENGE),
    )
    parser.add_argument("--challenge-id")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    from ai.tutor.frozen_behavior_torch import (
        build_known_hu_policy_value_prior_dispatch,
    )

    output = Path(args.output_dir).resolve()
    behavior_model = build_known_hu_policy_value_prior_dispatch(args.workspace_root)
    if args.resume:
        stored = _read_top_manifest(output / TOP_MANIFEST_NAME)
        stored_config = BehaviorTraceCollectionConfig.from_canonical_dict(
            stored["collection_config"]
        )
        mode = args.root_sampling_mode or stored_config.root_sampling_mode
        seed = args.seed_namespace or stored_config.seed_namespace
        challenge_id = (
            args.challenge_id
            if args.challenge_id is not None
            else stored_config.challenge_id
        )
        if mode == NATURAL_UNIFORM_SHUFFLE:
            if args.challenge_id is not None:
                parser.error("--challenge-id is valid only for targeted collection")
            config = BehaviorTraceCollectionConfig(
                seed_namespace=seed,
                root_sampling_mode=mode,
            )
        else:
            config = BehaviorTraceCollectionConfig(
                seed_namespace=seed,
                root_sampling_mode=mode,
                challenge_id=challenge_id,
                challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
            )
        result = resume_sharded_behavior_trace_collection(
            output,
            behavior_model,
            total_root_target=args.total_root_target,
            shard_size=args.shard_size,
            expected_config=config,
        )
    else:
        if not args.seed_namespace:
            parser.error("--seed-namespace is required for a new collection")
        mode = args.root_sampling_mode or NATURAL_UNIFORM_SHUFFLE
        if mode == NATURAL_UNIFORM_SHUFFLE:
            if args.challenge_id is not None:
                parser.error("--challenge-id is valid only for targeted collection")
            config = BehaviorTraceCollectionConfig(
                seed_namespace=args.seed_namespace,
                root_sampling_mode=mode,
            )
        else:
            config = BehaviorTraceCollectionConfig(
                seed_namespace=args.seed_namespace,
                root_sampling_mode=mode,
                challenge_id=args.challenge_id or "m3-joker-challenge-v1",
                challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
            )
        result = collect_sharded_behavior_traces(
            output,
            config,
            behavior_model,
            total_root_target=args.total_root_target,
            shard_size=args.shard_size,
        )

    print(
        canonical_json(
            {
                "schema": CLI_RESULT_SCHEMA,
                "output_dir": str(result.output_dir),
                "top_manifest_path": str(result.output_dir / TOP_MANIFEST_NAME),
                "root_sampling_mode": result.config.root_sampling_mode,
                "root_count": result.root_count,
                "decision_count": result.decision_count,
                "shard_count": result.shard_count,
                "collection_complete": result.collection_complete,
                "collection_content_sha256": result.manifest[
                    "collection_content_sha256"
                ],
                "layout_sha256": result.manifest["layout_sha256"],
                "manifest_sha256": result.manifest["manifest_sha256"],
                "elapsed_collection_runtime_ns": result.manifest["counters"][
                    "elapsed_collection_runtime_ns"
                ],
                "verification_runtime_ns": result.verification_runtime_ns,
                "shard_artifact_bytes": result.manifest["counters"][
                    "shard_artifact_bytes"
                ],
                "top_manifest_bytes": (
                    result.output_dir / TOP_MANIFEST_NAME
                ).stat().st_size,
                "promotion_eligible": False,
            }
        )
    )
    return 0


__all__ = [
    "CLI_RESULT_SCHEMA",
    "SHARDED_COLLECTION_SCHEMA",
    "SHARD_ENTRY_SCHEMA",
    "ShardedBehaviorTraceCollection",
    "collect_sharded_behavior_traces",
    "main",
    "read_sharded_behavior_trace_collection",
    "resume_sharded_behavior_trace_collection",
]


if __name__ == "__main__":
    raise SystemExit(main())
