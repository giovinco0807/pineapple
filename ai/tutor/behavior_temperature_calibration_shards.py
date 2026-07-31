"""Bounded-memory temperature calibration over immutable trace shards.

This module is the production-scale counterpart of
``behavior_temperature_calibration.py``.  It does not concatenate raw trace or
evaluation JSONL files and it never retains the full JSON corpus in memory.
Instead, each immutable input/evaluation shard is freshly verified and joined
one-to-one, after which only the metric fields are stored in an ephemeral
SQLite database.

The numerical protocol is deliberately *not* a new calibration algorithm.
Rows are read back in the same ``record_sha256`` order used by the v2 in-memory
builder and are passed to the exact v2 temperature, metric, paired-root
bootstrap, and gate implementations.  Consequently a small sharded corpus has
exactly the same role temperatures, metrics, and gate result as the v2
in-memory builder while large corpora remain bounded by one input shard plus
one metric group.

The published artifact has a distinct schema.  Its promotion bit is derived
only from the rebuilt v2 gate result, and it binds both natural and Joker
challenge collection/evaluation content, layouts, ordered entries, evaluator
routes, checkpoints, and source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import ai.tutor.behavior_temperature_calibration as legacy_calibration
import ai.tutor.collect_hu_behavior_trace_shards as collection_shards
import ai.tutor.evaluate_hu_behavior_trace_shards as evaluation_shards
from ai.tutor.behavior_calibration_contract import (
    SPLIT_NAMES,
    canonical_json,
    canonical_sha256,
    canonical_snapshot,
)
from ai.tutor.behavior_logit_evaluator_torch import (
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.behavior_temperature_calibration import (
    CALIBRATION_SCHEMA,
    IDENTITY_TEMPERATURE_NUMERATOR,
    JOKER_KEYS,
    ROLE_KEYS,
    TEMPERATURE_DENOMINATOR,
    PreTemperatureLegalLogitEvaluator,
    VerifiedModelEvaluation,
    _BoundRow,
    _build_gate_result,
    _canonical_float_hex,
    _fit_temperature,
    _mean_nll,
    _metric_summary,
    _temperature_payload,
    _temperature_value,
    _with_bootstrap,
    build_temperature_gate_config,
    verify_model_evaluation_row,
    verify_temperature_gate_config,
)
from ai.tutor.collect_hu_behavior_trace_shards import (
    ShardedBehaviorTraceCollection,
    read_sharded_behavior_trace_collection,
)
from ai.tutor.collect_hu_behavior_traces import (
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
)
from ai.tutor.evaluate_hu_behavior_trace_shards import (
    ShardedBehaviorTraceEvaluation,
    read_sharded_behavior_trace_evaluation,
)


SHARDED_CALIBRATION_SCHEMA = "ofc_behavior_temperature_calibration_sharded/v1"
CLI_RESULT_SCHEMA = "ofc_behavior_temperature_calibration_sharded_cli_result/v1"
ARTIFACT_NAME = "calibration.json"

_DATASET_NATURAL = 0
_DATASET_CHALLENGE = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KEYS = frozenset(
    {
        "schema",
        "promotion_eligible",
        "promotion_derivation",
        "computation_contract",
        "source_hashes",
        "input_bindings",
        "record_evaluation_binding_complete",
        "root_overlap_audit",
        "role_model_bindings",
        "selection_contract",
        "locked_test_contract",
        "joker_challenge",
        "temperatures",
        "metrics",
        "gate_config",
        "gate_result",
        "artifact_sha256",
    }
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "sharded_calibration_source_sha256": _sha256_file(__file__),
        "legacy_v2_calibration_source_sha256": _sha256_file(
            legacy_calibration.__file__
        ),
        "sharded_collection_source_sha256": _sha256_file(
            collection_shards.__file__
        ),
        "sharded_evaluation_source_sha256": _sha256_file(
            evaluation_shards.__file__
        ),
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_canonical_object(path: Path, *, label: str) -> dict[str, Any]:
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


def _collection_binding(
    collection: ShardedBehaviorTraceCollection,
) -> dict[str, Any]:
    manifest = collection.manifest
    return {
        "schema": manifest["schema"],
        "collection_complete": manifest["collection_complete"],
        "collection_content_sha256": manifest["collection_content_sha256"],
        "layout_sha256": manifest["layout_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "ordered_shard_entry_chain_sha256": manifest[
            "ordered_shard_entry_chain_sha256"
        ],
        "source_hashes": dict(manifest["source_hashes"]),
        "policy": dict(manifest["policy"]),
        "collection_config": canonical_snapshot(manifest["collection_config"]),
        "root_count": manifest["counters"]["root_count"],
        "decision_count": manifest["counters"]["decision_count"],
        "shard_count": len(manifest["shards"]),
        "shards": [
            {
                "index": entry["index"],
                "name": entry["name"],
                "root_index_start": entry["root_index_start"],
                "root_index_stop_exclusive": entry[
                    "root_index_stop_exclusive"
                ],
                "entry_sha256": entry["entry_sha256"],
                "collection_content_sha256": entry[
                    "collection_content_sha256"
                ],
                "shard_manifest_sha256": entry["shard_manifest_sha256"],
                "decisions_file_sha256": entry["decisions_file_sha256"],
                "hidden_roots_file_sha256": entry[
                    "hidden_roots_file_sha256"
                ],
                "shard_manifest_file_sha256": entry[
                    "shard_manifest_file_sha256"
                ],
                "decision_count": entry["decision_count"],
            }
            for entry in manifest["shards"]
        ],
    }


def _evaluation_binding(
    evaluation: ShardedBehaviorTraceEvaluation,
) -> dict[str, Any]:
    manifest = evaluation.manifest
    return {
        "schema": manifest["schema"],
        "evaluation_complete": manifest["evaluation_complete"],
        "evaluation_content_sha256": manifest["evaluation_content_sha256"],
        "layout_sha256": manifest["layout_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "ordered_evaluation_shard_entry_chain_sha256": manifest[
            "ordered_evaluation_shard_entry_chain_sha256"
        ],
        "source_hashes": dict(manifest["source_hashes"]),
        "evaluator_set_sha256": manifest["evaluator_set_sha256"],
        "evaluator_routes": canonical_snapshot(manifest["evaluator_routes"]),
        "input_collection": canonical_snapshot(manifest["input_collection"]),
        "row_count": manifest["counters"]["row_count"],
        "shard_count": manifest["counters"]["evaluation_shard_count"],
        "shards": [
            {
                "index": entry["index"],
                "name": entry["name"],
                "entry_sha256": entry["entry_sha256"],
                "shard_content_sha256": entry["shard_content_sha256"],
                "shard_manifest_sha256": entry["shard_manifest_sha256"],
                "evaluations_file_sha256": entry[
                    "evaluations_file_sha256"
                ],
                "shard_manifest_file_sha256": entry[
                    "shard_manifest_file_sha256"
                ],
                "input_shard": canonical_snapshot(entry["input_shard"]),
                "row_count": entry["row_count"],
            }
            for entry in manifest["shards"]
        ],
    }


@dataclass(frozen=True)
class _VerifiedInputs:
    natural_collection: ShardedBehaviorTraceCollection
    natural_evaluation: ShardedBehaviorTraceEvaluation
    challenge_collection: ShardedBehaviorTraceCollection
    challenge_evaluation: ShardedBehaviorTraceEvaluation


def _verify_inputs(
    *,
    natural_collection_dir: str | Path,
    natural_evaluation_dir: str | Path,
    challenge_collection_dir: str | Path,
    challenge_evaluation_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    challenge_prefix: str,
) -> _VerifiedInputs:
    roots = [
        Path(natural_collection_dir).resolve(),
        Path(natural_evaluation_dir).resolve(),
        Path(challenge_collection_dir).resolve(),
        Path(challenge_evaluation_dir).resolve(),
    ]
    if len(set(roots)) != len(roots):
        raise ValueError("natural/challenge collection/evaluation roots must be distinct")

    natural_evaluation = read_sharded_behavior_trace_evaluation(
        roots[0], roots[1], evaluators
    )
    challenge_evaluation = read_sharded_behavior_trace_evaluation(
        roots[2], roots[3], evaluators
    )
    natural_collection = read_sharded_behavior_trace_collection(roots[0])
    challenge_collection = read_sharded_behavior_trace_collection(roots[2])

    for label, collection, evaluation in (
        ("natural", natural_collection, natural_evaluation),
        ("challenge", challenge_collection, challenge_evaluation),
    ):
        if not collection.collection_complete:
            raise ValueError(f"{label} collection must be complete and frozen")
        if not evaluation.evaluation_complete:
            raise ValueError(f"{label} evaluation must cover the complete collection")
        if evaluation.row_count != collection.decision_count:
            raise ValueError(f"{label} evaluation/collection decision count mismatch")
        if evaluation.shard_count != collection.shard_count:
            raise ValueError(f"{label} evaluation/collection shard count mismatch")

    if natural_collection.config.root_sampling_mode != NATURAL_UNIFORM_SHUFFLE:
        raise ValueError("natural calibration collection must use uniform shuffle")
    if challenge_collection.config.root_sampling_mode != TARGETED_JOKER_CHALLENGE:
        raise ValueError("Joker challenge collection must use targeted challenge sampling")
    expected_challenge_id = challenge_prefix[:-1] if challenge_prefix.endswith("/") else None
    if (
        expected_challenge_id is None
        or challenge_collection.config.challenge_id != expected_challenge_id
    ):
        raise ValueError(
            "Joker challenge collection id must exactly match the gate namespace prefix"
        )
    if (
        natural_evaluation.manifest["evaluator_set_sha256"]
        != challenge_evaluation.manifest["evaluator_set_sha256"]
        or natural_evaluation.manifest["evaluator_routes"]
        != challenge_evaluation.manifest["evaluator_routes"]
    ):
        raise ValueError("natural and challenge evaluator/checkpoint routes differ")
    return _VerifiedInputs(
        natural_collection=natural_collection,
        natural_evaluation=natural_evaluation,
        challenge_collection=challenge_collection,
        challenge_evaluation=challenge_evaluation,
    )


def _open_metric_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.executescript(
        """
        CREATE TABLE metric_rows (
            dataset INTEGER NOT NULL,
            record_sha256 TEXT NOT NULL,
            evaluation_row_sha256 TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            root_id TEXT NOT NULL,
            split TEXT NOT NULL,
            role TEXT NOT NULL,
            joker_count INTEGER NOT NULL,
            observed_index INTEGER NOT NULL,
            logits_f64 BLOB NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            row_extractor_sha256 TEXT NOT NULL,
            adapter_source_sha256 TEXT NOT NULL,
            PRIMARY KEY (dataset, record_sha256),
            UNIQUE (dataset, decision_id),
            UNIQUE (dataset, root_id, role)
        ) WITHOUT ROWID;
        CREATE TABLE roots (
            dataset INTEGER NOT NULL,
            root_id TEXT NOT NULL,
            split TEXT NOT NULL,
            root_commitment TEXT NOT NULL,
            seed_namespace TEXT NOT NULL,
            PRIMARY KEY (dataset, root_id)
        ) WITHOUT ROWID;
        """
    )
    connection.commit()
    return connection


def _pack_logits(values: Sequence[float]) -> bytes:
    if not values:
        raise ValueError("verified evaluation row unexpectedly has no logits")
    return struct.pack(f"<{len(values)}d", *values)


def _unpack_logits(value: bytes) -> tuple[float, ...]:
    if not value or len(value) % 8:
        raise ValueError("metric-store logits blob is malformed")
    return struct.unpack(f"<{len(value) // 8}d", value)


def _bindings_by_role(
    evaluation: ShardedBehaviorTraceEvaluation,
) -> dict[str, Mapping[str, Any]]:
    return {row["role"]: row for row in evaluation.manifest["evaluator_routes"]}


def _ingest_verified_join(
    connection: sqlite3.Connection,
    *,
    dataset_kind: int,
    collection: ShardedBehaviorTraceCollection,
    evaluation: ShardedBehaviorTraceEvaluation,
    challenge_prefix: str,
) -> None:
    bindings = _bindings_by_role(evaluation)
    if set(bindings) != set(ROLE_KEYS):
        raise ValueError("evaluation routes do not cover exactly the four roles")
    inserted_rows = 0
    inserted_roots = 0
    for index, (input_entry, output_entry) in enumerate(
        zip(collection.manifest["shards"], evaluation.manifest["shards"])
    ):
        dataset, rebuilt_input_entry = (  # type: ignore[attr-defined]
            evaluation_shards._fresh_input_shard(collection, index)
        )
        if rebuilt_input_entry != input_entry:
            raise ValueError("input shard changed during calibration join")
        expected_input_binding = evaluation_shards._input_shard_binding(  # type: ignore[attr-defined]
            input_entry
        )
        if output_entry["input_shard"] != expected_input_binding:
            raise ValueError("evaluation shard no longer binds its input shard")
        shard_dir = evaluation.output_dir / output_entry["name"]
        if _sha256_file(shard_dir / evaluation_shards.SHARD_MANIFEST_NAME) != output_entry[
            "shard_manifest_file_sha256"
        ]:
            raise ValueError("evaluation shard manifest changed during calibration join")

        iterator = evaluation_shards._iter_canonical_jsonl(  # type: ignore[attr-defined]
            shard_dir / evaluation_shards.EVALUATIONS_NAME
        )
        digest = hashlib.sha256()
        byte_count = 0
        shard_rows = 0
        local_roots: set[str] = set()
        connection.execute("BEGIN IMMEDIATE")
        try:
            for record in dataset.records:
                try:
                    row, raw_line = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "evaluation shard has fewer rows than its input shard"
                    ) from exc
                verified = verify_model_evaluation_row(row, record)
                binding = bindings.get(verified.role)
                if binding is None:
                    raise ValueError("joined row has no exact evaluator route")
                for key in (
                    "checkpoint_sha256",
                    "model_sha256",
                    "row_extractor_sha256",
                    "adapter_source_sha256",
                ):
                    if getattr(verified, key) != binding[key]:
                        raise ValueError(f"joined row {key} differs from evaluator route")

                root_id = verified.root_id
                if dataset_kind == _DATASET_NATURAL:
                    if root_id.startswith(challenge_prefix):
                        raise ValueError("natural root uses the Joker challenge namespace")
                elif not root_id.startswith(challenge_prefix):
                    raise ValueError("Joker challenge root is outside the gate namespace")

                if root_id not in local_roots:
                    local_roots.add(root_id)
                    try:
                        connection.execute(
                            "INSERT INTO roots VALUES (?, ?, ?, ?, ?)",
                            (
                                dataset_kind,
                                root_id,
                                verified.split,
                                record["root_commitment"],
                                record["seed_namespace"],
                            ),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            "one root appears in more than one immutable input shard"
                        ) from exc
                    inserted_roots += 1
                try:
                    connection.execute(
                        """
                        INSERT INTO metric_rows VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            dataset_kind,
                            verified.record_sha256,
                            verified.evaluation_row_sha256,
                            verified.decision_id,
                            verified.root_id,
                            verified.split,
                            verified.role,
                            verified.visible_joker_count,
                            verified.observed_index,
                            sqlite3.Binary(_pack_logits(verified.logits)),
                            verified.checkpoint_sha256,
                            verified.model_sha256,
                            verified.row_extractor_sha256,
                            verified.adapter_source_sha256,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "duplicate record, decision, or root/role in sharded calibration input"
                    ) from exc
                digest.update(raw_line)
                byte_count += len(raw_line)
                shard_rows += 1
                inserted_rows += 1
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("evaluation shard has more rows than its input shard")
            if shard_rows != output_entry["row_count"]:
                raise ValueError("evaluation shard row count drifted during join")
            if digest.hexdigest() != output_entry["evaluations_file_sha256"]:
                raise ValueError("evaluation shard bytes changed during calibration join")
            if byte_count != (shard_dir / evaluation_shards.EVALUATIONS_NAME).stat().st_size:
                raise ValueError("evaluation shard byte count changed during join")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    if inserted_rows != collection.decision_count or inserted_rows != evaluation.row_count:
        raise ValueError("joined decision total does not match completed manifests")
    if inserted_roots != collection.root_count:
        raise ValueError("joined root total does not match completed collection")


def _finish_metric_store(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX metric_group_order
            ON metric_rows(dataset, role, split, record_sha256);
        CREATE INDEX metric_group_joker_order
            ON metric_rows(dataset, role, split, joker_count, record_sha256);
        CREATE INDEX metric_challenge_order
            ON metric_rows(dataset, role, joker_count, record_sha256);
        CREATE INDEX root_split_order
            ON roots(dataset, split, root_id);
        ANALYZE;
        """
    )
    connection.commit()


class _SqliteBoundRows:
    """Re-iterable, record-SHA-ordered row view over the metric store."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        dataset: int,
        role: str,
        split: str | None = None,
        joker_count: int | None = None,
        logit_divisor: float = 1.0,
    ) -> None:
        if role not in ROLE_KEYS:
            raise ValueError("unknown metric role")
        clauses = ["dataset = ?", "role = ?"]
        parameters: list[Any] = [dataset, role]
        if split is not None:
            if split not in SPLIT_NAMES:
                raise ValueError("unknown metric split")
            clauses.append("split = ?")
            parameters.append(split)
        if joker_count is not None:
            if joker_count not in (0, 1, 2):
                raise ValueError("unknown Joker stratum")
            clauses.append("joker_count = ?")
            parameters.append(joker_count)
        self._connection = connection
        self._where = " AND ".join(clauses)
        self._parameters = tuple(parameters)
        self._logit_divisor = float(logit_divisor)
        if not self._logit_divisor > 0.0:
            raise ValueError("logit divisor must be positive")
        self._count = int(
            connection.execute(
                f"SELECT COUNT(*) FROM metric_rows WHERE {self._where}",
                self._parameters,
            ).fetchone()[0]
        )

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[_BoundRow]:
        query = f"""
            SELECT evaluation_row_sha256, record_sha256, decision_id, root_id,
                   split, role, joker_count, observed_index, logits_f64,
                   checkpoint_sha256, model_sha256, row_extractor_sha256,
                   adapter_source_sha256
              FROM metric_rows
             WHERE {self._where}
             ORDER BY record_sha256
        """
        for raw in self._connection.execute(query, self._parameters):
            logits = _unpack_logits(raw[8])
            if self._logit_divisor != 1.0:
                logits = tuple(value / self._logit_divisor for value in logits)
            evaluation = VerifiedModelEvaluation(
                evaluation_row_sha256=raw[0],
                record_sha256=raw[1],
                decision_id=raw[2],
                root_id=raw[3],
                split=raw[4],
                role=raw[5],
                visible_joker_count=raw[6],
                # The v2 numerical functions use logits/observed_index only;
                # action identifiers were already freshly verified at ingest.
                legal_action_ids=tuple("" for _ in logits),
                observed_index=raw[7],
                logits=logits,
                checkpoint_sha256=raw[9],
                model_sha256=raw[10],
                row_extractor_sha256=raw[11],
                adapter_source_sha256=raw[12],
            )
            yield _BoundRow(evaluation)


def _streaming_list_sha256(
    connection: sqlite3.Connection,
    query: str,
    parameters: Sequence[Any] = (),
) -> str:
    """Hash a one-column ordered query exactly like canonical_sha256(list)."""
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for (value,) in connection.execute(query, tuple(parameters)):
        if not isinstance(value, str):
            raise TypeError("streaming list commitment accepts text values only")
        if not first:
            digest.update(b",")
        digest.update(canonical_json(value).encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _count(
    connection: sqlite3.Connection, query: str, parameters: Sequence[Any] = ()
) -> int:
    return int(connection.execute(query, tuple(parameters)).fetchone()[0])


def _role_model_bindings(connection: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ROLE_KEYS:
        natural = connection.execute(
            """
            SELECT DISTINCT checkpoint_sha256, model_sha256,
                            row_extractor_sha256, adapter_source_sha256
              FROM metric_rows WHERE dataset = ? AND role = ?
            """,
            (_DATASET_NATURAL, role),
        ).fetchall()
        if len(natural) > 1:
            raise ValueError(f"role {role} mixes natural model/extractor bindings")
        if not natural:
            result[role] = None
            continue
        checkpoint, model, extractor, adapter = natural[0]
        binding = {
            "checkpoint_sha256": checkpoint,
            "model_sha256": model,
            "row_extractor_sha256": extractor,
            "adapter_source_sha256": adapter,
        }
        challenge = connection.execute(
            """
            SELECT DISTINCT checkpoint_sha256, model_sha256,
                            row_extractor_sha256, adapter_source_sha256
              FROM metric_rows WHERE dataset = ? AND role = ?
            """,
            (_DATASET_CHALLENGE, role),
        ).fetchall()
        if challenge != [natural[0]]:
            raise ValueError(
                f"Joker challenge role {role} does not match its natural model binding"
            )
        result[role] = binding
    return result


def _build_numerical_result(
    connection: sqlite3.Connection, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    temperatures: dict[str, Any] = {}
    for role in ROLE_KEYS:
        fit_rows = _SqliteBoundRows(
            connection, dataset=_DATASET_NATURAL, split="fit", role=role
        )
        dev_rows = _SqliteBoundRows(
            connection, dataset=_DATASET_NATURAL, split="dev", role=role
        )
        fit_numerator, optimizer_status = _fit_temperature(fit_rows)  # type: ignore[arg-type]
        fit_temperature = fit_numerator / TEMPERATURE_DENOMINATOR
        fit_dev_nll = _mean_nll(dev_rows, fit_temperature)  # type: ignore[arg-type]
        identity_dev_nll = _mean_nll(dev_rows, 1.0)  # type: ignore[arg-type]
        if (
            fit_dev_nll is not None
            and identity_dev_nll is not None
            and fit_numerator != IDENTITY_TEMPERATURE_NUMERATOR
            and fit_dev_nll < identity_dev_nll
        ):
            selected = "fit_temperature"
            final_numerator = fit_numerator
        else:
            selected = "identity_temperature"
            final_numerator = IDENTITY_TEMPERATURE_NUMERATOR
        final_temperature = final_numerator / TEMPERATURE_DENOMINATOR
        test_scaled_rows = _SqliteBoundRows(
            connection,
            dataset=_DATASET_NATURAL,
            split="test",
            role=role,
            logit_divisor=final_temperature,
        )
        residual_numerator, residual_status = _fit_temperature(  # type: ignore[arg-type]
            test_scaled_rows
        )
        temperatures[role] = {
            "optimizer": {
                "objective": "fit_observed_action_nll",
                "parameterization": "inverse_temperature_beta_bisection_96",
                "status": optimizer_status,
            },
            "fit_temperature": _temperature_payload(fit_numerator),
            "dev_candidate_nll": {
                "fit_temperature_f64_hex": (
                    None
                    if fit_dev_nll is None
                    else _canonical_float_hex(fit_dev_nll, label="fit_dev_nll")
                ),
                "identity_temperature_f64_hex": (
                    None
                    if identity_dev_nll is None
                    else _canonical_float_hex(
                        identity_dev_nll, label="identity_dev_nll"
                    )
                ),
            },
            "dev_selected_candidate": selected,
            "dev_selection_frozen_before_locked_test": True,
            "final_temperature": _temperature_payload(final_numerator),
            "locked_test_residual_temperature": _temperature_payload(
                residual_numerator
            ),
            "locked_test_residual_optimizer_status": residual_status,
        }

    ece_bins = config["metric_contract"]["ece_bins"]
    replicates = config["bootstrap"]["replicates"]
    seed = config["bootstrap"]["seed"]
    metrics: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        by_role: dict[str, Any] = {}
        by_role_joker: dict[str, Any] = {}
        for role in ROLE_KEYS:
            payload = temperatures[role]["final_temperature"]
            temperature = _temperature_value(
                payload, label=f"temperatures.{role}.final_temperature"
            )
            rows = _SqliteBoundRows(
                connection,
                dataset=_DATASET_NATURAL,
                split=split,
                role=role,
            )
            summary = _metric_summary(rows, payload, ece_bins=ece_bins)  # type: ignore[arg-type]
            if split == "test":
                summary = _with_bootstrap(
                    summary,
                    rows,  # type: ignore[arg-type]
                    temperature=temperature,
                    replicates=replicates,
                    seed=seed,
                    context=f"{split}.{role}",
                )
            by_role[role] = summary
            by_role_joker[role] = {}
            for joker_count, joker in enumerate(JOKER_KEYS):
                cell = _SqliteBoundRows(
                    connection,
                    dataset=_DATASET_NATURAL,
                    split=split,
                    role=role,
                    joker_count=joker_count,
                )
                cell_summary = _metric_summary(  # type: ignore[arg-type]
                    cell, payload, ece_bins=ece_bins
                )
                if split == "test":
                    cell_summary = _with_bootstrap(
                        cell_summary,
                        cell,  # type: ignore[arg-type]
                        temperature=temperature,
                        replicates=replicates,
                        seed=seed,
                        context=f"{split}.{role}.{joker}",
                    )
                by_role_joker[role][joker] = cell_summary
        metrics[split] = {"by_role": by_role, "by_role_joker": by_role_joker}

    challenge_metrics: dict[str, Any] = {"by_role_joker": {}}
    for role in ROLE_KEYS:
        payload = temperatures[role]["final_temperature"]
        temperature = _temperature_value(
            payload, label=f"temperatures.{role}.final_temperature"
        )
        challenge_metrics["by_role_joker"][role] = {}
        for joker_count, joker in enumerate(JOKER_KEYS):
            rows = _SqliteBoundRows(
                connection,
                dataset=_DATASET_CHALLENGE,
                role=role,
                joker_count=joker_count,
            )
            summary = _metric_summary(rows, payload, ece_bins=ece_bins)  # type: ignore[arg-type]
            summary = _with_bootstrap(
                summary,
                rows,  # type: ignore[arg-type]
                temperature=temperature,
                replicates=replicates,
                seed=seed,
                context=f"challenge.{role}.{joker}",
            )
            challenge_metrics["by_role_joker"][role][joker] = summary
    metrics["challenge"] = challenge_metrics
    gate_result = _build_gate_result(
        config=config, temperatures=temperatures, metrics=metrics
    )
    return temperatures, metrics, gate_result


def _build_artifact(
    connection: sqlite3.Connection,
    *,
    inputs: _VerifiedInputs,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    role_bindings = _role_model_bindings(connection)
    overlap = _count(
        connection,
        """
        SELECT COUNT(*) FROM roots AS natural
        INNER JOIN roots AS challenge USING(root_id)
        WHERE natural.dataset = ? AND challenge.dataset = ?
        """,
        (_DATASET_NATURAL, _DATASET_CHALLENGE),
    )
    if overlap:
        raise ValueError("natural and Joker challenge root sets overlap")
    split_intersections: dict[str, int] = {}
    for left, right, key in (
        ("fit", "dev", "fit_dev"),
        ("fit", "test", "fit_test"),
        ("dev", "test", "dev_test"),
    ):
        split_intersections[key] = _count(
            connection,
            """
            SELECT COUNT(*) FROM roots AS a INNER JOIN roots AS b USING(root_id)
             WHERE a.dataset = ? AND b.dataset = ?
               AND a.split = ? AND b.split = ?
            """,
            (_DATASET_NATURAL, _DATASET_NATURAL, left, right),
        )
    if any(split_intersections.values()):
        raise ValueError("root overlap across natural calibration splits")

    temperatures, metrics, gate_result = _build_numerical_result(
        connection, config
    )
    fit_dev_record_sha = _streaming_list_sha256(
        connection,
        """
        SELECT record_sha256 FROM metric_rows
         WHERE dataset = ? AND split IN ('fit','dev') ORDER BY record_sha256
        """,
        (_DATASET_NATURAL,),
    )
    fit_dev_evaluation_sha = _streaming_list_sha256(
        connection,
        """
        SELECT evaluation_row_sha256 FROM metric_rows
         WHERE dataset = ? AND split IN ('fit','dev') ORDER BY record_sha256
        """,
        (_DATASET_NATURAL,),
    )
    test_record_sha = _streaming_list_sha256(
        connection,
        """
        SELECT record_sha256 FROM metric_rows
         WHERE dataset = ? AND split = 'test' ORDER BY record_sha256
        """,
        (_DATASET_NATURAL,),
    )
    test_evaluation_sha = _streaming_list_sha256(
        connection,
        """
        SELECT evaluation_row_sha256 FROM metric_rows
         WHERE dataset = ? AND split = 'test' ORDER BY record_sha256
        """,
        (_DATASET_NATURAL,),
    )
    selected_temperature_sha = canonical_sha256(
        {
            role: temperatures[role]["final_temperature"]
            for role in ROLE_KEYS
        }
    )
    root_commitments = {
        split: _streaming_list_sha256(
            connection,
            "SELECT root_id FROM roots WHERE dataset = ? AND split = ? ORDER BY root_id",
            (_DATASET_NATURAL, split),
        )
        for split in SPLIT_NAMES
    }
    challenge_root_commitment = _streaming_list_sha256(
        connection,
        "SELECT root_id FROM roots WHERE dataset = ? ORDER BY root_id",
        (_DATASET_CHALLENGE,),
    )
    natural_record_set_sha = _streaming_list_sha256(
        connection,
        "SELECT record_sha256 FROM metric_rows "
        "WHERE dataset = ? ORDER BY record_sha256",
        (_DATASET_NATURAL,),
    )
    natural_evaluation_set_sha = _streaming_list_sha256(
        connection,
        "SELECT evaluation_row_sha256 FROM metric_rows "
        "WHERE dataset = ? ORDER BY evaluation_row_sha256",
        (_DATASET_NATURAL,),
    )
    challenge_record_set_sha = _streaming_list_sha256(
        connection,
        "SELECT record_sha256 FROM metric_rows "
        "WHERE dataset = ? ORDER BY record_sha256",
        (_DATASET_CHALLENGE,),
    )
    challenge_evaluation_set_sha = _streaming_list_sha256(
        connection,
        "SELECT evaluation_row_sha256 FROM metric_rows "
        "WHERE dataset = ? ORDER BY evaluation_row_sha256",
        (_DATASET_CHALLENGE,),
    )

    artifact: dict[str, Any] = {
        "schema": SHARDED_CALIBRATION_SCHEMA,
        "promotion_eligible": gate_result["promotion_eligible"],
        "promotion_derivation": {
            "sole_authority": "gate_result.promotion_eligible",
            "gate_result_sha256": gate_result["gate_result_sha256"],
            "all_required_gates_passed": gate_result[
                "all_required_gates_passed"
            ],
            "manual_override_supported": False,
        },
        "computation_contract": {
            "legacy_v2_schema": CALIBRATION_SCHEMA,
            "temperature_metric_gate_semantics": "exact_legacy_v2_implementation",
            "row_order": "record_sha256_ascending_binary",
            "join": "fresh_verified_one_input_row_to_one_evaluation_row_per_shard",
            "storage": "ephemeral_sqlite_metric_fields_only",
            "raw_json_corpus_retained_in_memory": False,
            "bootstrap_algorithm_changed": False,
            "combined_jsonl_present": False,
        },
        "source_hashes": _source_hashes(),
        "input_bindings": {
            "natural": {
                "collection": _collection_binding(inputs.natural_collection),
                "evaluation": _evaluation_binding(inputs.natural_evaluation),
                "raw_record_set_sha256": natural_record_set_sha,
                "evaluation_row_set_sha256": natural_evaluation_set_sha,
            },
            "challenge": {
                "collection": _collection_binding(inputs.challenge_collection),
                "evaluation": _evaluation_binding(inputs.challenge_evaluation),
                "raw_record_set_sha256": challenge_record_set_sha,
                "evaluation_row_set_sha256": challenge_evaluation_set_sha,
            },
        },
        "record_evaluation_binding_complete": True,
        "root_overlap_audit": {
            "checked": True,
            "pairwise_overlap_counts": split_intersections,
            "overlap_count": sum(split_intersections.values()),
            "main_challenge_overlap_count": overlap,
            "root_list_commitments_by_split": root_commitments,
            "challenge_root_list_commitment": challenge_root_commitment,
        },
        "role_model_bindings": role_bindings,
        "selection_contract": {
            "fit_dev_only_record_set_sha256": fit_dev_record_sha,
            "fit_dev_only_evaluation_set_sha256": fit_dev_evaluation_sha,
            "candidate_set": ["fit_temperature", "identity_temperature"],
            "promotion_policy": config["selection_policy"],
            "locked_test_was_selection_input": False,
            "selection_frozen_before_locked_test": True,
            "selected_candidate_by_role": {
                role: temperatures[role]["dev_selected_candidate"]
                for role in ROLE_KEYS
            },
            "selected_temperature_set_sha256": selected_temperature_sha,
            "published_metrics_temperature_bound": True,
        },
        "locked_test_contract": {
            "record_set_sha256": test_record_sha,
            "evaluation_set_sha256": test_evaluation_sha,
            "evaluation_passes": 1,
            "used_for_selection": False,
        },
        "joker_challenge": {
            "present": True,
            "required_for_promotion": config["challenge_contract"][
                "required_for_promotion"
            ],
            "root_namespace_prefix": config["challenge_contract"][
                "root_namespace_prefix"
            ],
            "root_count": _count(
                connection,
                "SELECT COUNT(*) FROM roots WHERE dataset = ?",
                (_DATASET_CHALLENGE,),
            ),
            "used_for_temperature_fit": False,
            "used_for_dev_selection": False,
            "used_for_locked_test_role_metrics": False,
        },
        "temperatures": temperatures,
        "metrics": metrics,
        "gate_config": canonical_snapshot(config),
        "gate_result": gate_result,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    _verify_artifact_envelope(artifact)
    return artifact


def _verify_artifact_envelope(artifact: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping):
        raise TypeError("sharded calibration artifact must be an object")
    if frozenset(artifact) != _ARTIFACT_KEYS:
        raise ValueError("sharded calibration artifact keys mismatch")
    if artifact.get("schema") != SHARDED_CALIBRATION_SCHEMA:
        raise ValueError("unsupported sharded calibration schema")
    recorded = _require_sha256(
        artifact.get("artifact_sha256"), label="artifact_sha256"
    )
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    if canonical_sha256(unsigned) != recorded:
        raise ValueError("sharded calibration artifact SHA-256 mismatch")
    gate = artifact.get("gate_result")
    if not isinstance(gate, Mapping):
        raise TypeError("sharded calibration gate_result must be an object")
    gate_sha = _require_sha256(
        gate.get("gate_result_sha256"), label="gate_result_sha256"
    )
    unsigned_gate = dict(gate)
    unsigned_gate.pop("gate_result_sha256", None)
    if canonical_sha256(unsigned_gate) != gate_sha:
        raise ValueError("sharded calibration gate result SHA-256 mismatch")
    promotion = gate.get("promotion_eligible")
    if type(promotion) is not bool:
        raise TypeError("gate promotion_eligible must be bool")
    expected_derivation = {
        "sole_authority": "gate_result.promotion_eligible",
        "gate_result_sha256": gate_sha,
        "all_required_gates_passed": gate.get("all_required_gates_passed"),
        "manual_override_supported": False,
    }
    if (
        artifact.get("promotion_eligible") is not promotion
        or artifact.get("promotion_derivation") != expected_derivation
        or gate.get("all_required_gates_passed") is not promotion
    ):
        raise ValueError("promotion eligibility is not derived solely from the gate")


def build_sharded_behavior_temperature_calibration(
    natural_collection_dir: str | Path,
    natural_evaluation_dir: str | Path,
    challenge_collection_dir: str | Path,
    challenge_evaluation_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    *,
    gate_config: Mapping[str, Any] | None = None,
    scratch_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build one content-bound, bounded-memory sharded calibration artifact."""
    config = verify_temperature_gate_config(
        gate_config if gate_config is not None else build_temperature_gate_config()
    )
    inputs = _verify_inputs(
        natural_collection_dir=natural_collection_dir,
        natural_evaluation_dir=natural_evaluation_dir,
        challenge_collection_dir=challenge_collection_dir,
        challenge_evaluation_dir=challenge_evaluation_dir,
        evaluators=evaluators,
        challenge_prefix=config["challenge_contract"]["root_namespace_prefix"],
    )
    scratch_root = None if scratch_dir is None else Path(scratch_dir).resolve()
    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ofc-sharded-temperature-", dir=scratch_root
    ) as raw_temp:
        database_path = Path(raw_temp) / "metric-store.sqlite3"
        connection = _open_metric_store(database_path)
        try:
            _ingest_verified_join(
                connection,
                dataset_kind=_DATASET_NATURAL,
                collection=inputs.natural_collection,
                evaluation=inputs.natural_evaluation,
                challenge_prefix=config["challenge_contract"][
                    "root_namespace_prefix"
                ],
            )
            _ingest_verified_join(
                connection,
                dataset_kind=_DATASET_CHALLENGE,
                collection=inputs.challenge_collection,
                evaluation=inputs.challenge_evaluation,
                challenge_prefix=config["challenge_contract"][
                    "root_namespace_prefix"
                ],
            )
            _finish_metric_store(connection)
            return _build_artifact(connection, inputs=inputs, config=config)
        finally:
            connection.close()


def verify_sharded_behavior_temperature_calibration(
    natural_collection_dir: str | Path,
    natural_evaluation_dir: str | Path,
    challenge_collection_dir: str | Path,
    challenge_evaluation_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    artifact: Mapping[str, Any],
    *,
    scratch_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild from immutable shards and require exact artifact equality."""
    _verify_artifact_envelope(artifact)
    gate_config = artifact.get("gate_config")
    if not isinstance(gate_config, Mapping):
        raise TypeError("sharded calibration gate_config must be an object")
    rebuilt = build_sharded_behavior_temperature_calibration(
        natural_collection_dir,
        natural_evaluation_dir,
        challenge_collection_dir,
        challenge_evaluation_dir,
        evaluators,
        gate_config=gate_config,
        scratch_dir=scratch_dir,
    )
    if canonical_snapshot(artifact) != rebuilt:
        raise ValueError("sharded calibration artifact does not match fresh inputs")
    return rebuilt


def write_sharded_behavior_temperature_calibration(
    artifact: Mapping[str, Any], path: str | Path
) -> Path:
    """Atomically publish one canonical artifact; existing paths are immutable."""
    _verify_artifact_envelope(artifact)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(dict(artifact)) + "\n").encode("utf-8")
    if target.exists():
        existing = _read_canonical_object(target, label="existing calibration artifact")
        if existing != canonical_snapshot(artifact):
            raise FileExistsError("published calibration artifact path is immutable")
        return target
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    if _read_canonical_object(target, label="written calibration artifact") != canonical_snapshot(
        artifact
    ):
        raise IOError("sharded calibration artifact readback mismatch")
    return target


def read_sharded_behavior_temperature_calibration(
    path: str | Path,
    natural_collection_dir: str | Path,
    natural_evaluation_dir: str | Path,
    challenge_collection_dir: str | Path,
    challenge_evaluation_dir: str | Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    *,
    scratch_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read a canonical artifact and freshly verify every bound shard."""
    artifact = _read_canonical_object(
        Path(path).resolve(), label="sharded calibration artifact"
    )
    return verify_sharded_behavior_temperature_calibration(
        natural_collection_dir,
        natural_evaluation_dir,
        challenge_collection_dir,
        challenge_evaluation_dir,
        evaluators,
        artifact,
        scratch_dir=scratch_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build bounded-memory temperature calibration from trace shards"
    )
    parser.add_argument("--natural-collection-dir", required=True, type=Path)
    parser.add_argument("--natural-evaluation-dir", required=True, type=Path)
    parser.add_argument("--challenge-collection-dir", required=True, type=Path)
    parser.add_argument("--challenge-evaluation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--gate-config", type=Path)
    args = parser.parse_args(argv)

    gate_config = None
    if args.gate_config is not None:
        gate_config = _read_canonical_object(
            args.gate_config.resolve(), label="temperature gate config"
        )
    evaluators = build_known_hu_policy_value_logit_evaluators(
        args.workspace_root.resolve()
    )
    artifact = build_sharded_behavior_temperature_calibration(
        args.natural_collection_dir,
        args.natural_evaluation_dir,
        args.challenge_collection_dir,
        args.challenge_evaluation_dir,
        evaluators,
        gate_config=gate_config,
        scratch_dir=args.scratch_dir,
    )
    path = write_sharded_behavior_temperature_calibration(artifact, args.output)
    print(
        canonical_json(
            {
                "schema": CLI_RESULT_SCHEMA,
                "artifact": str(path),
                "artifact_sha256": artifact["artifact_sha256"],
                "gate_result_sha256": artifact["gate_result"][
                    "gate_result_sha256"
                ],
                "promotion_eligible": artifact["promotion_eligible"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_NAME",
    "CLI_RESULT_SCHEMA",
    "SHARDED_CALIBRATION_SCHEMA",
    "build_sharded_behavior_temperature_calibration",
    "read_sharded_behavior_temperature_calibration",
    "verify_sharded_behavior_temperature_calibration",
    "write_sharded_behavior_temperature_calibration",
]
