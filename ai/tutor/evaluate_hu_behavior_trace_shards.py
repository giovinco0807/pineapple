"""Append-only direct-logit evaluation for sharded HU behavior traces.

The input is a fully verified, immutable
``ofc_behavior_trace_sharded_collection/v1`` directory.  One input shard is
mapped to one ``evaluation-shard-XXXXXX`` directory; rows are never combined
into a process-wide list or a monolithic JSONL file.  Each output shard is
written to a private staging directory, writes its manifest last, is verified,
and is then atomically published.  The top-level manifest is the final commit
marker after every appended shard.

This is raw calibration input only.  Neither rows, shard manifests, nor the
top manifest make a promotion claim.  A separate temperature-calibration gate
must rebuild and judge the resulting evidence.
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
from typing import Any, Iterator, Mapping, Sequence

import ai.tutor.behavior_logit_evaluator_torch as logit_evaluator_module
import ai.tutor.behavior_temperature_calibration as calibration_module
import ai.tutor.collect_hu_behavior_trace_shards as input_shards_module
from ai.tutor.behavior_calibration_contract import (
    SPLIT_NAMES,
    canonical_json,
    canonical_sha256,
)
from ai.tutor.behavior_logit_evaluator_torch import EVALUATOR_VERSION
from ai.tutor.behavior_temperature_calibration import (
    MODEL_EVALUATION_SCHEMA,
    PreTemperatureLegalLogitEvaluator,
    VerifiedModelEvaluation,
    build_model_evaluation_row,
    verify_model_evaluation_row,
)
from ai.tutor.collect_hu_behavior_trace_shards import (
    SHARDED_COLLECTION_SCHEMA,
    ShardedBehaviorTraceCollection,
    read_sharded_behavior_trace_collection,
)
from ai.tutor.collect_hu_behavior_traces import (
    CollectedBehaviorTraces,
    read_behavior_trace_dataset,
)
from ai.tutor.frozen_behavior_torch import TurnActorBehaviorDispatch


SHARDED_EVALUATION_SCHEMA = "ofc_behavior_trace_sharded_evaluation/v1"
EVALUATION_SHARD_MANIFEST_SCHEMA = (
    "ofc_behavior_trace_evaluation_shard_manifest/v1"
)
EVALUATION_SHARD_ENTRY_SCHEMA = "ofc_behavior_trace_evaluation_shard_entry/v1"
EVALUATION_STREAM_CHAIN_SCHEMA = "ofc_behavior_evaluation_stream_chain/v1"
EVALUATOR_SET_SCHEMA = "ofc_behavior_logit_evaluator_set/v1"
CLI_RESULT_SCHEMA = "ofc_behavior_trace_sharded_evaluation_cli_result/v1"

EXPECTED_INPUT_POLICY_ID = "known_hu_t1_t2_policyvalue_prior_dispatch_v1"
TOP_MANIFEST_NAME = "manifest.json"
EVALUATIONS_NAME = "evaluations.jsonl"
SHARD_MANIFEST_NAME = "manifest.json"

_EVALUATION_SHARD_RE = re.compile(r"^evaluation-shard-([0-9]{6})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTES = ((1, "bb"), (1, "btn"), (2, "bb"), (2, "btn"))
_ROLE_KEYS = tuple(f"t{turn}_{actor}" for turn, actor in _ROUTES)
_JOKER_KEYS = ("joker_0", "joker_1", "joker_2")
_EVALUATOR_MANIFEST_KEYS = {
    "schema",
    "evaluator_version",
    "checkpoint_sha256",
    "model_sha256",
    "row_extractor_sha256",
    "adapter_source_sha256",
    "temperature",
    "device",
    "logit_contract",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _source_hashes() -> dict[str, str]:
    return {
        "sharded_evaluation_source_sha256": _sha256_file(__file__),
        "logit_evaluator_source_sha256": _sha256_file(
            logit_evaluator_module.__file__
        ),
        "temperature_calibration_source_sha256": _sha256_file(
            calibration_module.__file__
        ),
        "sharded_collection_source_sha256": _sha256_file(
            input_shards_module.__file__
        ),
    }


def _chain_genesis(stream: str) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema": EVALUATION_STREAM_CHAIN_SCHEMA,
                "stream": stream,
                "genesis": True,
            }
        ).encode("utf-8")
    ).hexdigest()


def _chain_step(previous: str, value: Any) -> str:
    _require_sha256(previous, label="stream-chain state")
    return hashlib.sha256(
        bytes.fromhex(previous) + canonical_json(value).encode("utf-8")
    ).hexdigest()


def _empty_census() -> dict[str, dict[str, int]]:
    return {
        "row_counts_by_role": {key: 0 for key in _ROLE_KEYS},
        "row_counts_by_split": {key: 0 for key in SPLIT_NAMES},
        "row_counts_by_joker": {key: 0 for key in _JOKER_KEYS},
        "row_counts_by_role_split": {
            f"{role}_{split}": 0
            for role in _ROLE_KEYS
            for split in SPLIT_NAMES
        },
        "row_counts_by_role_joker": {
            f"{role}_{joker}": 0
            for role in _ROLE_KEYS
            for joker in _JOKER_KEYS
        },
        "row_counts_by_role_split_joker": {
            f"{role}_{split}_{joker}": 0
            for role in _ROLE_KEYS
            for split in SPLIT_NAMES
            for joker in _JOKER_KEYS
        },
    }


def _empty_commitments() -> dict[str, str]:
    return {
        "evaluation_row_order_chain_sha256": _chain_genesis(
            "evaluation_row_sha256"
        ),
        "record_order_chain_sha256": _chain_genesis("record_sha256"),
        "decision_id_order_chain_sha256": _chain_genesis("decision_id"),
        "root_id_order_chain_sha256": _chain_genesis("root_id_per_decision"),
    }


@dataclass
class _StreamAggregate:
    census: dict[str, dict[str, int]]
    commitments: dict[str, str]
    row_count: int = 0

    @classmethod
    def empty(cls) -> "_StreamAggregate":
        return cls(census=_empty_census(), commitments=_empty_commitments())

    @classmethod
    def from_top_manifest(cls, manifest: Mapping[str, Any]) -> "_StreamAggregate":
        return cls(
            census=json.loads(canonical_json(manifest["census"])),
            commitments=dict(manifest["global_content_commitments"]),
            row_count=manifest["counters"]["row_count"],
        )

    def add(self, row: Mapping[str, Any], verified: VerifiedModelEvaluation) -> None:
        role = verified.role
        split = verified.split
        joker = f"joker_{verified.visible_joker_count}"
        if role not in _ROLE_KEYS or split not in SPLIT_NAMES or joker not in _JOKER_KEYS:
            raise ValueError("evaluation row lies outside the T1/T2 split/Joker census")
        self.census["row_counts_by_role"][role] += 1
        self.census["row_counts_by_split"][split] += 1
        self.census["row_counts_by_joker"][joker] += 1
        self.census["row_counts_by_role_split"][f"{role}_{split}"] += 1
        self.census["row_counts_by_role_joker"][f"{role}_{joker}"] += 1
        self.census["row_counts_by_role_split_joker"][
            f"{role}_{split}_{joker}"
        ] += 1
        for key, value in (
            ("evaluation_row_order_chain_sha256", verified.evaluation_row_sha256),
            ("record_order_chain_sha256", verified.record_sha256),
            ("decision_id_order_chain_sha256", verified.decision_id),
            ("root_id_order_chain_sha256", verified.root_id),
        ):
            self.commitments[key] = _chain_step(self.commitments[key], value)
        self.row_count += 1


@dataclass(frozen=True)
class ShardedBehaviorTraceEvaluation:
    input_dir: Path
    output_dir: Path
    manifest: dict[str, Any]
    verification_runtime_ns: int

    @property
    def row_count(self) -> int:
        return self.manifest["counters"]["row_count"]

    @property
    def shard_count(self) -> int:
        return self.manifest["counters"]["evaluation_shard_count"]

    @property
    def evaluation_complete(self) -> bool:
        return self.manifest["evaluation_complete"]


def _role_key(turn: int, actor: str) -> str:
    key = f"t{turn}_{actor}"
    if key not in _ROLE_KEYS:
        raise ValueError(f"unsupported evaluator route: {key}")
    return key


def _evaluator_bindings(
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
) -> tuple[list[dict[str, Any]], str, dict[str, Mapping[str, Any]]]:
    if not isinstance(evaluators, Mapping) or set(evaluators) != set(_ROUTES):
        raise ValueError("evaluators must contain exactly T1/T2 x BB/BTN routes")
    bindings: list[dict[str, Any]] = []
    by_role: dict[str, Mapping[str, Any]] = {}
    for turn, actor in _ROUTES:
        evaluator = evaluators[(turn, actor)]
        method = getattr(evaluator, "pre_temperature_legal_logits", None)
        if not callable(method):
            raise TypeError("every route must provide direct pre-temperature logits")
        raw_manifest = getattr(evaluator, "evaluator_manifest", None)
        if not isinstance(raw_manifest, Mapping):
            raise TypeError("every route must expose an evaluator_manifest")
        manifest = dict(raw_manifest)
        if set(manifest) != _EVALUATOR_MANIFEST_KEYS:
            raise ValueError("evaluator manifest keys mismatch")
        if (
            manifest["schema"] != "ofc_behavior_logit_evaluator/v1"
            or manifest["evaluator_version"] != EVALUATOR_VERSION
            or manifest["temperature"] != "1/1"
            or manifest["device"] != "cpu"
            or manifest["logit_contract"]
            != "checkpoint_policy_head_pre_temperature_pre_quantization"
        ):
            raise ValueError("evaluator is not the direct CPU identity-temperature reader")
        hashes: dict[str, str] = {}
        for key in (
            "checkpoint_sha256",
            "model_sha256",
            "row_extractor_sha256",
            "adapter_source_sha256",
        ):
            value = _require_sha256(getattr(evaluator, key, None), label=key)
            if manifest[key] != value:
                raise ValueError(f"evaluator manifest {key} mismatch")
            hashes[key] = value
        model_id = getattr(evaluator, "model_id", None)
        model_manifest = getattr(evaluator, "model_manifest", None)
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("evaluator behavior model_id must be non-empty")
        if not isinstance(model_manifest, Mapping):
            raise TypeError("evaluator must expose its frozen behavior model manifest")
        # The legacy frozen-model manifest contains checkpoint metrics as JSON
        # floats and therefore uses that module's canonical hasher rather than
        # the calibration contract's float-rejecting helper.  Do not re-hash it
        # here with a different contract.  ``_expected_input_policy`` below
        # constructs TurnActorBehaviorDispatch, whose native constructor
        # authenticates every child manifest against exactly ``model_sha256``.
        binding: dict[str, Any] = {
            "turn": turn,
            "actor": actor,
            "role": _role_key(turn, actor),
            "behavior_model_id": model_id,
            **hashes,
            "evaluator_manifest_sha256": canonical_sha256(manifest),
        }
        binding["evaluator_binding_sha256"] = canonical_sha256(binding)
        bindings.append(binding)
        by_role[binding["role"]] = binding
    set_sha = canonical_sha256(
        {"schema": EVALUATOR_SET_SCHEMA, "routes": bindings}
    )
    return bindings, set_sha, by_role


def _expected_input_policy(
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
) -> dict[str, str]:
    # TurnActorBehaviorDispatch authenticates every child model manifest before
    # deriving the exact policy identity used by the trace collector.
    dispatch = TurnActorBehaviorDispatch(
        evaluators,  # type: ignore[arg-type]
        model_id=EXPECTED_INPUT_POLICY_ID,
    )
    return {"model_id": dispatch.model_id, "model_sha256": dispatch.model_sha256}


def _input_collection_binding(
    collection: ShardedBehaviorTraceCollection,
) -> dict[str, Any]:
    manifest = collection.manifest
    return {
        "schema": SHARDED_COLLECTION_SCHEMA,
        "collection_complete": manifest["collection_complete"],
        "collection_content_sha256": manifest["collection_content_sha256"],
        "layout_sha256": manifest["layout_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "ordered_shard_entry_chain_sha256": manifest[
            "ordered_shard_entry_chain_sha256"
        ],
        "policy": dict(manifest["policy"]),
        "root_count": manifest["counters"]["root_count"],
        "decision_count": manifest["counters"]["decision_count"],
        "shard_count": len(manifest["shards"]),
        "shards": [
            {
                "index": entry["index"],
                "name": entry["name"],
                "entry_sha256": entry["entry_sha256"],
            }
            for entry in manifest["shards"]
        ],
    }


def _input_shard_binding(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": entry["index"],
        "name": entry["name"],
        "entry_sha256": entry["entry_sha256"],
        "collection_content_sha256": entry["collection_content_sha256"],
        "shard_manifest_sha256": entry["shard_manifest_sha256"],
        "decisions_file_sha256": entry["decisions_file_sha256"],
        "decision_count": entry["decision_count"],
    }


def _evaluation_contract() -> dict[str, Any]:
    return {
        "model_evaluation_schema": MODEL_EVALUATION_SCHEMA,
        "input_to_output_shard_mapping": "one_to_one_same_index",
        "logit_contract": (
            "frozen_checkpoint_pre_temperature_selected_legal_logits_float64"
        ),
        "route_contract": "exact_t1_t2_bb_btn_no_fallback",
        "publication_contract": "manifest_last_atomic_directory_then_top_marker",
        "combined_jsonl_present": False,
    }


def _shard_name(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 999999:
        raise ValueError("evaluation shard index must fit six digits")
    return f"evaluation-shard-{index:06d}"


def _read_canonical_manifest(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{label} must be one canonical JSON object with trailing newline")
    try:
        text = raw[:-1].decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != text:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_manifest_last(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(manifest)) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_top(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(manifest)) + "\n").encode("utf-8")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _iter_canonical_jsonl(path: Path) -> Iterator[tuple[dict[str, Any], bytes]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise ValueError(
                    f"evaluation JSONL line {line_number} lacks trailing newline"
                )
            payload = raw_line[:-1]
            if not payload:
                raise ValueError(f"evaluation JSONL line {line_number} is empty")
            try:
                text = payload.decode("utf-8")
                row = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"evaluation JSONL line {line_number} is invalid UTF-8 JSON"
                ) from exc
            if not isinstance(row, dict) or canonical_json(row) != text:
                raise ValueError(
                    f"evaluation JSONL line {line_number} is not canonical JSON"
                )
            yield row, raw_line


def _fresh_input_shard(
    collection: ShardedBehaviorTraceCollection,
    index: int,
) -> tuple[CollectedBehaviorTraces, Mapping[str, Any]]:
    entry = collection.manifest["shards"][index]
    shard_dir = collection.output_dir / entry["name"]
    dataset = read_behavior_trace_dataset(shard_dir / "decisions.jsonl")
    rebuilt = input_shards_module._build_shard_entry(  # type: ignore[attr-defined]
        index=index, shard_dir=shard_dir, dataset=dataset
    )
    if rebuilt != entry:
        raise ValueError("input shard changed after collection verification")
    return dataset, entry


def _check_row_evaluator_binding(
    row: Mapping[str, Any],
    verified: VerifiedModelEvaluation,
    bindings_by_role: Mapping[str, Mapping[str, Any]],
) -> None:
    binding = bindings_by_role.get(verified.role)
    if binding is None:
        raise ValueError("evaluation row has no exact evaluator route")
    for key in (
        "checkpoint_sha256",
        "model_sha256",
        "row_extractor_sha256",
        "adapter_source_sha256",
    ):
        if row[key] != binding[key]:
            raise ValueError(f"evaluation row {key} does not match route binding")


def _shard_content_payload(
    *,
    index: int,
    input_collection_content_sha256: str,
    input_entry: Mapping[str, Any],
    evaluator_set_sha256: str,
    source_hashes: Mapping[str, str],
    aggregate: _StreamAggregate,
    evaluations_file_sha256: str,
    evaluations_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_SHARD_MANIFEST_SCHEMA,
        "promotion_eligible": False,
        "raw_evaluation_only": True,
        "index": index,
        "name": _shard_name(index),
        "input_collection_content_sha256": input_collection_content_sha256,
        "input_shard": _input_shard_binding(input_entry),
        "evaluator_set_sha256": evaluator_set_sha256,
        "source_hashes": dict(source_hashes),
        "row_count": aggregate.row_count,
        "census": aggregate.census,
        "content_commitments": aggregate.commitments,
        "evaluations_artifact": {
            "name": EVALUATIONS_NAME,
            "file_sha256": evaluations_file_sha256,
            "bytes": evaluations_bytes,
            "row_count": aggregate.row_count,
        },
    }


def _build_shard_manifest(**kwargs: Any) -> dict[str, Any]:
    payload = _shard_content_payload(**kwargs)
    manifest = dict(payload)
    manifest["shard_content_sha256"] = canonical_sha256(payload)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _build_shard_entry(shard_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_bytes = (shard_dir / EVALUATIONS_NAME).stat().st_size
    manifest_bytes = (shard_dir / SHARD_MANIFEST_NAME).stat().st_size
    entry: dict[str, Any] = {
        "schema": EVALUATION_SHARD_ENTRY_SCHEMA,
        "index": manifest["index"],
        "name": manifest["name"],
        "input_shard": dict(manifest["input_shard"]),
        "row_count": manifest["row_count"],
        "census": json.loads(canonical_json(manifest["census"])),
        "content_commitments": dict(manifest["content_commitments"]),
        "shard_content_sha256": manifest["shard_content_sha256"],
        "shard_manifest_sha256": manifest["manifest_sha256"],
        "evaluations_file_sha256": _sha256_file(shard_dir / EVALUATIONS_NAME),
        "shard_manifest_file_sha256": _sha256_file(
            shard_dir / SHARD_MANIFEST_NAME
        ),
        "artifact_sizes": {
            "evaluations_bytes": evaluation_bytes,
            "shard_manifest_bytes": manifest_bytes,
            "total_bytes": evaluation_bytes + manifest_bytes,
        },
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry


def _verify_manifest_hash(
    manifest: Mapping[str, Any], hash_key: str, *, label: str
) -> None:
    recorded = _require_sha256(manifest.get(hash_key), label=f"{label}.{hash_key}")
    unsigned = dict(manifest)
    unsigned.pop(hash_key, None)
    if canonical_sha256(unsigned) != recorded:
        raise ValueError(f"{label} SHA-256 mismatch")


def _verify_one_evaluation_shard(
    *,
    shard_dir: Path,
    index: int,
    dataset: CollectedBehaviorTraces,
    input_entry: Mapping[str, Any],
    input_collection_content_sha256: str,
    evaluator_set_sha256: str,
    source_hashes: Mapping[str, str],
    bindings_by_role: Mapping[str, Mapping[str, Any]],
    global_aggregate: _StreamAggregate,
) -> tuple[dict[str, Any], _StreamAggregate]:
    manifest = _read_canonical_manifest(
        shard_dir / SHARD_MANIFEST_NAME, label="evaluation shard manifest"
    )
    _verify_manifest_hash(manifest, "manifest_sha256", label="evaluation shard manifest")
    if manifest.get("schema") != EVALUATION_SHARD_MANIFEST_SCHEMA:
        raise ValueError("unsupported evaluation shard manifest schema")
    if (
        manifest.get("promotion_eligible") is not False
        or manifest.get("raw_evaluation_only") is not True
    ):
        raise ValueError("raw evaluation shard cannot be promotion eligible")

    iterator = _iter_canonical_jsonl(shard_dir / EVALUATIONS_NAME)
    local = _StreamAggregate.empty()
    file_digest = hashlib.sha256()
    byte_count = 0
    for record in dataset.records:
        try:
            row, raw_line = next(iterator)
        except StopIteration as exc:
            raise ValueError("evaluation shard has fewer rows than input decisions") from exc
        verified = verify_model_evaluation_row(row, record)
        _check_row_evaluator_binding(row, verified, bindings_by_role)
        local.add(row, verified)
        global_aggregate.add(row, verified)
        file_digest.update(raw_line)
        byte_count += len(raw_line)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("evaluation shard has more rows than input decisions")

    rebuilt = _build_shard_manifest(
        index=index,
        input_collection_content_sha256=input_collection_content_sha256,
        input_entry=input_entry,
        evaluator_set_sha256=evaluator_set_sha256,
        source_hashes=source_hashes,
        aggregate=local,
        evaluations_file_sha256=file_digest.hexdigest(),
        evaluations_bytes=byte_count,
    )
    if rebuilt != manifest:
        raise ValueError("evaluation shard manifest does not match raw rows/input binding")
    if manifest["shard_content_sha256"] != canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"shard_content_sha256", "manifest_sha256"}
        }
    ):
        raise ValueError("evaluation shard content SHA-256 mismatch")
    entry = _build_shard_entry(shard_dir, manifest)
    return entry, local


def _ordered_entry_chain(entries: Sequence[Mapping[str, Any]]) -> str:
    chain = _chain_genesis("evaluation_shard_entry_sha256")
    for entry in entries:
        chain = _chain_step(chain, entry["entry_sha256"])
    return chain


def _build_top_manifest(
    *,
    input_binding: Mapping[str, Any],
    evaluator_bindings: Sequence[Mapping[str, Any]],
    evaluator_set_sha256: str,
    source_hashes: Mapping[str, str],
    entries: Sequence[Mapping[str, Any]],
    aggregate: _StreamAggregate,
) -> dict[str, Any]:
    artifact_bytes = sum(entry["artifact_sizes"]["total_bytes"] for entry in entries)
    counters = {
        "input_shards_processed": len(entries),
        "evaluation_shard_count": len(entries),
        "row_count": aggregate.row_count,
        "evaluation_artifact_bytes": artifact_bytes,
    }
    complete = len(entries) == input_binding["shard_count"]
    content_payload = {
        "schema": SHARDED_EVALUATION_SCHEMA,
        "promotion_eligible": False,
        "raw_evaluation_only": True,
        "input_collection": dict(input_binding),
        "evaluation_contract": _evaluation_contract(),
        "evaluator_routes": [dict(binding) for binding in evaluator_bindings],
        "evaluator_set_sha256": evaluator_set_sha256,
        "source_hashes": dict(source_hashes),
        "row_count": aggregate.row_count,
        "census": aggregate.census,
        "global_content_commitments": aggregate.commitments,
    }
    manifest: dict[str, Any] = {
        **content_payload,
        "evaluation_complete": complete,
        "counters": counters,
        "shards": [dict(entry) for entry in entries],
        "evaluation_content_sha256": canonical_sha256(content_payload),
        "ordered_evaluation_shard_entry_chain_sha256": _ordered_entry_chain(entries),
    }
    manifest["layout_sha256"] = canonical_sha256(
        {
            "input_collection_layout_sha256": input_binding["layout_sha256"],
            "evaluation_complete": complete,
            "counters": counters,
            "shards": manifest["shards"],
            "evaluation_content_sha256": manifest["evaluation_content_sha256"],
        }
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _finalized_evaluation_directories(output_dir: Path) -> set[str]:
    names: set[str] = set()
    if not output_dir.exists():
        return names
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if _EVALUATION_SHARD_RE.fullmatch(child.name):
            names.add(child.name)
        elif child.name.startswith("evaluation-shard-"):
            raise ValueError(
                f"malformed finalized evaluation shard directory: {child.name}"
            )
        # Private dot-prefixed staging directories are never published state.
    return names


def _verify_top_envelope(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "promotion_eligible",
        "raw_evaluation_only",
        "input_collection",
        "evaluation_contract",
        "evaluator_routes",
        "evaluator_set_sha256",
        "source_hashes",
        "row_count",
        "census",
        "global_content_commitments",
        "evaluation_complete",
        "counters",
        "shards",
        "evaluation_content_sha256",
        "ordered_evaluation_shard_entry_chain_sha256",
        "layout_sha256",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("evaluation top manifest keys mismatch")
    if (
        manifest["schema"] != SHARDED_EVALUATION_SCHEMA
        or manifest["promotion_eligible"] is not False
        or manifest["raw_evaluation_only"] is not True
    ):
        raise ValueError("unsupported or promotable sharded evaluation")
    _verify_manifest_hash(manifest, "manifest_sha256", label="evaluation top manifest")


def _validate_distinct_roots(input_dir: Path, output_dir: Path) -> None:
    if (
        input_dir == output_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError("input and evaluation output directories must be disjoint")


def _verified_input_and_bindings(
    input_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
) -> tuple[
    ShardedBehaviorTraceCollection,
    dict[str, Any],
    list[dict[str, Any]],
    str,
    dict[str, Mapping[str, Any]],
]:
    collection = read_sharded_behavior_trace_collection(input_dir)
    if not collection.collection_complete:
        raise ValueError("input sharded behavior collection must be complete and frozen")
    evaluator_bindings, evaluator_set_sha, bindings_by_role = _evaluator_bindings(
        evaluators
    )
    expected_policy = _expected_input_policy(evaluators)
    if collection.manifest["policy"] != expected_policy:
        raise ValueError("input behavior policy does not match the four evaluator routes")
    input_binding = _input_collection_binding(collection)
    return (
        collection,
        input_binding,
        evaluator_bindings,
        evaluator_set_sha,
        bindings_by_role,
    )


def read_sharded_behavior_trace_evaluation(
    input_dir: str | Path,
    output_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
) -> ShardedBehaviorTraceEvaluation:
    """Freshly verify the input, every output shard, and the top marker."""
    started = time.perf_counter_ns()
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    _validate_distinct_roots(input_root, output_root)
    (
        collection,
        input_binding,
        evaluator_bindings,
        evaluator_set_sha,
        bindings_by_role,
    ) = _verified_input_and_bindings(input_root, evaluators)
    source_hashes = _source_hashes()
    manifest = _read_canonical_manifest(
        output_root / TOP_MANIFEST_NAME, label="evaluation top manifest"
    )
    _verify_top_envelope(manifest)
    if manifest["source_hashes"] != source_hashes:
        raise ValueError("evaluation source hash binding mismatch")
    if manifest["input_collection"] != input_binding:
        raise ValueError("evaluation input collection binding drift")
    if (
        manifest["evaluator_routes"] != evaluator_bindings
        or manifest["evaluator_set_sha256"] != evaluator_set_sha
    ):
        raise ValueError("evaluation evaluator/checkpoint binding drift")
    if manifest["evaluation_contract"] != _evaluation_contract():
        raise ValueError("evaluation contract drift")

    raw_entries = manifest["shards"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("evaluation top manifest must reference at least one shard")
    if len(raw_entries) > len(collection.manifest["shards"]):
        raise ValueError("evaluation has more shards than its input collection")
    global_aggregate = _StreamAggregate.empty()
    verified_entries: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("evaluation shard entry must be an object")
        entry = dict(raw_entry)
        recorded_hash = entry.get("entry_sha256")
        unsigned = dict(entry)
        unsigned.pop("entry_sha256", None)
        if canonical_sha256(unsigned) != recorded_hash:
            raise ValueError("evaluation shard entry SHA-256 mismatch")
        name = _shard_name(index)
        if (
            entry.get("schema") != EVALUATION_SHARD_ENTRY_SCHEMA
            or entry.get("index") != index
            or entry.get("name") != name
        ):
            raise ValueError("evaluation shard gap, duplicate, or out-of-order index")
        dataset, input_entry = _fresh_input_shard(collection, index)
        if entry.get("input_shard") != _input_shard_binding(input_entry):
            raise ValueError("evaluation shard points to the wrong input shard")
        shard_dir = output_root / name
        rebuilt_entry, _local = _verify_one_evaluation_shard(
            shard_dir=shard_dir,
            index=index,
            dataset=dataset,
            input_entry=input_entry,
            input_collection_content_sha256=input_binding[
                "collection_content_sha256"
            ],
            evaluator_set_sha256=evaluator_set_sha,
            source_hashes=source_hashes,
            bindings_by_role=bindings_by_role,
            global_aggregate=global_aggregate,
        )
        if rebuilt_entry != entry:
            raise ValueError("evaluation shard entry does not match immutable artifacts")
        verified_entries.append(entry)
        expected_names.add(name)

    actual_names = _finalized_evaluation_directories(output_root)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        orphan = sorted(actual_names - expected_names)
        raise ValueError(
            f"evaluation shard directory set mismatch; missing={missing}, orphan={orphan}"
        )
    rebuilt = _build_top_manifest(
        input_binding=input_binding,
        evaluator_bindings=evaluator_bindings,
        evaluator_set_sha256=evaluator_set_sha,
        source_hashes=source_hashes,
        entries=verified_entries,
        aggregate=global_aggregate,
    )
    if rebuilt != manifest:
        raise ValueError("evaluation top manifest does not match verified shard artifacts")
    return ShardedBehaviorTraceEvaluation(
        input_dir=input_root,
        output_dir=output_root,
        manifest=dict(manifest),
        verification_runtime_ns=time.perf_counter_ns() - started,
    )


def _write_evaluation_rows(
    *,
    path: Path,
    dataset: CollectedBehaviorTraces,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    bindings_by_role: Mapping[str, Mapping[str, Any]],
    global_aggregate: _StreamAggregate,
) -> tuple[_StreamAggregate, str, int]:
    local = _StreamAggregate.empty()
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("xb") as handle:
        for record in dataset.records:
            route = (record["turn"], record["actor"])
            evaluator = evaluators.get(route)
            if evaluator is None:
                raise ValueError("raw decision has no exact direct-logit evaluator route")
            row = build_model_evaluation_row(record, evaluator)
            verified = verify_model_evaluation_row(row, record)
            _check_row_evaluator_binding(row, verified, bindings_by_role)
            payload = (canonical_json(row) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            byte_count += len(payload)
            local.add(row, verified)
            global_aggregate.add(row, verified)
        handle.flush()
        os.fsync(handle.fileno())
    return local, digest.hexdigest(), byte_count


def _publish_one_evaluation_shard(
    *,
    output_dir: Path,
    index: int,
    collection: ShardedBehaviorTraceCollection,
    input_binding: Mapping[str, Any],
    evaluator_set_sha256: str,
    source_hashes: Mapping[str, str],
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    bindings_by_role: Mapping[str, Mapping[str, Any]],
    global_aggregate: _StreamAggregate,
) -> dict[str, Any]:
    name = _shard_name(index)
    final_dir = output_dir / name
    if final_dir.exists():
        raise FileExistsError(f"published evaluation shard is immutable: {final_dir}")
    dataset, input_entry = _fresh_input_shard(collection, index)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{name}.", suffix=".partial", dir=output_dir)
    )
    try:
        local, file_sha, byte_count = _write_evaluation_rows(
            path=staging_dir / EVALUATIONS_NAME,
            dataset=dataset,
            evaluators=evaluators,
            bindings_by_role=bindings_by_role,
            global_aggregate=global_aggregate,
        )
        manifest = _build_shard_manifest(
            index=index,
            input_collection_content_sha256=input_binding[
                "collection_content_sha256"
            ],
            input_entry=input_entry,
            evaluator_set_sha256=evaluator_set_sha256,
            source_hashes=source_hashes,
            aggregate=local,
            evaluations_file_sha256=file_sha,
            evaluations_bytes=byte_count,
        )
        # Manifest is deliberately the final file created in the staging dir.
        _write_manifest_last(staging_dir / SHARD_MANIFEST_NAME, manifest)
        check_global = _StreamAggregate.empty()
        staged_entry, _ = _verify_one_evaluation_shard(
            shard_dir=staging_dir,
            index=index,
            dataset=dataset,
            input_entry=input_entry,
            input_collection_content_sha256=input_binding[
                "collection_content_sha256"
            ],
            evaluator_set_sha256=evaluator_set_sha256,
            source_hashes=source_hashes,
            bindings_by_role=bindings_by_role,
            global_aggregate=check_global,
        )
        if check_global.row_count != local.row_count:
            raise AssertionError("staged evaluation shard readback row count mismatch")
        os.replace(staging_dir, final_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    final_entry = _build_shard_entry(
        final_dir,
        _read_canonical_manifest(
            final_dir / SHARD_MANIFEST_NAME, label="evaluation shard manifest"
        ),
    )
    if final_entry != staged_entry:
        raise ValueError("atomic evaluation shard publication changed its bytes")
    return final_entry


def evaluate_hu_behavior_trace_shards(
    input_dir: str | Path,
    output_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    *,
    resume: bool = False,
    max_new_shards: int | None = None,
) -> ShardedBehaviorTraceEvaluation:
    """Evaluate a fixed collection, appending only its next input-shard prefix.

    The input collection must already be complete; a collection that is still
    growing is rejected rather than silently rebinding existing output.  A new
    run evaluates all shards by default.  A resume without an explicit
    ``max_new_shards`` appends exactly one next input shard, after freshly
    verifying every published prefix shard.  Larger bounded resume batches can
    be requested explicitly.  Every iteration commits exactly one next input
    shard and republishes the top marker before considering the following one.
    """
    if max_new_shards is not None and (
        isinstance(max_new_shards, bool)
        or not isinstance(max_new_shards, int)
        or max_new_shards <= 0
    ):
        raise ValueError("max_new_shards must be a positive integer or None")
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    _validate_distinct_roots(input_root, output_root)
    (
        collection,
        input_binding,
        evaluator_bindings,
        evaluator_set_sha,
        bindings_by_role,
    ) = _verified_input_and_bindings(input_root, evaluators)
    source_hashes = _source_hashes()
    output_root.mkdir(parents=True, exist_ok=True)
    top_path = output_root / TOP_MANIFEST_NAME

    if resume:
        existing = read_sharded_behavior_trace_evaluation(
            input_root, output_root, evaluators
        )
        entries = [dict(entry) for entry in existing.manifest["shards"]]
        aggregate = _StreamAggregate.from_top_manifest(existing.manifest)
    else:
        if top_path.exists():
            raise FileExistsError("evaluation top manifest exists; use resume")
        finalized = _finalized_evaluation_directories(output_root)
        if finalized:
            raise ValueError(
                f"evaluation output contains orphan finalized shards: {sorted(finalized)}"
            )
        entries = []
        aggregate = _StreamAggregate.empty()

    start = len(entries)
    total = len(collection.manifest["shards"])
    effective_limit = 1 if resume and max_new_shards is None else max_new_shards
    stop = total if effective_limit is None else min(total, start + effective_limit)
    for index in range(start, stop):
        entry = _publish_one_evaluation_shard(
            output_dir=output_root,
            index=index,
            collection=collection,
            input_binding=input_binding,
            evaluator_set_sha256=evaluator_set_sha,
            source_hashes=source_hashes,
            evaluators=evaluators,
            bindings_by_role=bindings_by_role,
            global_aggregate=aggregate,
        )
        entries.append(entry)
        top = _build_top_manifest(
            input_binding=input_binding,
            evaluator_bindings=evaluator_bindings,
            evaluator_set_sha256=evaluator_set_sha,
            source_hashes=source_hashes,
            entries=entries,
            aggregate=aggregate,
        )
        _atomic_write_top(top_path, top)

    if not top_path.exists():
        raise RuntimeError("no evaluation shard/top marker was published")
    return read_sharded_behavior_trace_evaluation(
        input_root, output_root, evaluators
    )


def resume_hu_behavior_trace_shard_evaluation(
    input_dir: str | Path,
    output_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    *,
    max_new_shards: int | None = None,
) -> ShardedBehaviorTraceEvaluation:
    return evaluate_hu_behavior_trace_shards(
        input_dir,
        output_dir,
        evaluators,
        resume=True,
        max_new_shards=max_new_shards,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-shards", type=int)
    args = parser.parse_args(argv)

    evaluators = logit_evaluator_module.build_known_hu_policy_value_logit_evaluators(
        args.workspace_root
    )
    result = evaluate_hu_behavior_trace_shards(
        args.input_dir,
        args.output_dir,
        evaluators,
        resume=args.resume,
        max_new_shards=args.max_new_shards,
    )
    print(
        canonical_json(
            {
                "schema": CLI_RESULT_SCHEMA,
                "input_dir": str(result.input_dir),
                "output_dir": str(result.output_dir),
                "top_manifest_path": str(result.output_dir / TOP_MANIFEST_NAME),
                "row_count": result.row_count,
                "evaluation_shard_count": result.shard_count,
                "evaluation_complete": result.evaluation_complete,
                "evaluation_content_sha256": result.manifest[
                    "evaluation_content_sha256"
                ],
                "layout_sha256": result.manifest["layout_sha256"],
                "manifest_sha256": result.manifest["manifest_sha256"],
                "verification_runtime_ns": result.verification_runtime_ns,
                "promotion_eligible": False,
            }
        )
    )
    return 0


__all__ = [
    "CLI_RESULT_SCHEMA",
    "EVALUATIONS_NAME",
    "EVALUATION_SHARD_ENTRY_SCHEMA",
    "EVALUATION_SHARD_MANIFEST_SCHEMA",
    "EXPECTED_INPUT_POLICY_ID",
    "SHARDED_EVALUATION_SCHEMA",
    "ShardedBehaviorTraceEvaluation",
    "evaluate_hu_behavior_trace_shards",
    "main",
    "read_sharded_behavior_trace_evaluation",
    "resume_hu_behavior_trace_shard_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
