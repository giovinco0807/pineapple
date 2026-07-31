"""Nonproduction online resolve over a compiled public T3/T4 root mixture.

This module deliberately separates two operations:

1. :func:`resolve_compiled_public_root_mixture` solves the complete declared
   ex-ante mixture and has no actual-hand input.
2. :func:`select_scoped_policy_for_actual_infoset` accepts an actual
   :class:`InfoSetKey` only after the solve, proves exact membership in the
   compiled support, and returns that one scoped policy row.

The result is algorithm-validation evidence over finite explicit support.  It
is never promotion, runtime integration, a global unseen-state policy, or an
exact exploitability certificate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import ai.engine.action_space as _action_space_module
import ai.tutor.exact_late as _exact_late_module
import ai.tutor.t3_hu_multi_root_mccfr as _multi_root_module
import ai.tutor.t3_t4_public_root_mixture as _mixture_module
from ai.engine.encoding import Board
from ai.tutor.t3_hu_multi_root_mccfr import MultiRootExternalSamplingMccfrResult
from ai.tutor.t3_hu_public_cfr import InfoSetKey
from ai.tutor.t3_t4_public_root_mixture import (
    CompiledPublicRootMixture,
    PublicRootContext,
    verify_compiled_public_root_mixture,
)


ONLINE_RESOLVE_SCHEMA = "ofc_t3_t4_public_online_resolve/v1"
SCOPED_POLICY_SELECTION_SCHEMA = "ofc_t3_t4_scoped_policy_selection/v1"
ONLINE_RESOLVE_METHOD = "finite_public_root_shared_mccfr_online_resolve_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_SOLVER = _multi_root_module.solve_multi_root_external_sampling_mccfr
_CANONICAL_GET_TURN_ACTIONS = _action_space_module.get_turn_actions
_CANONICAL_ACTION_KEY = _exact_late_module.action_key
_CANONICAL_VERIFY_COMPILED_MIXTURE = verify_compiled_public_root_mixture
_CANONICAL_PYTHON_FUNCTION_BINDER = _multi_root_module._python_function_binding
_CANONICAL_CONTENT_SHA256 = _multi_root_module._checkpoint_content_sha256
_CANONICAL_RUNTIME_SEMANTIC_GRAPH = _multi_root_module._runtime_semantic_graph
_CANONICAL_SERIALIZE_STRATEGY_PROFILE = (
    _multi_root_module.serialize_strategy_profile
)

ONLINE_RESOLVE_REMAINING_LIMITATIONS = (
    "finite_explicit_actor_private_type_support_is_not_full_deck_coverage",
    "sampled_mccfr_has_no_exact_exploitability_certificate",
    "independent_strength_evaluation_requires_separate_payoff_seeds",
    "global_serving_requires_online_root_generation_or_a_distilled_generalizing_policy",
    "t4_second_btn_exact_bypass_is_not_integrated_by_this_wrapper",
)
_CANONICAL_ONLINE_RESOLVE_REMAINING_LIMITATIONS = (
    "finite_explicit_actor_private_type_support_is_not_full_deck_coverage",
    "sampled_mccfr_has_no_exact_exploitability_certificate",
    "independent_strength_evaluation_requires_separate_payoff_seeds",
    "global_serving_requires_online_root_generation_or_a_distilled_generalizing_policy",
    "t4_second_btn_exact_bypass_is_not_integrated_by_this_wrapper",
)
_CANONICAL_WRAPPER_CALLABLES: Mapping[str, Any] | None = None
_CANONICAL_WRAPPER_RUNTIME_ROOTS: Mapping[str, Any] | None = None
_CANONICAL_WRAPPER_RUNTIME_SEMANTIC_BINDING_SHA256: str | None = None


class PublicOnlineResolveError(ValueError):
    """The online resolve, its artifact binding, or post-solve selection failed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PublicOnlineResolveError(f"{label} must be a lowercase SHA256 digest")
    return value


def _json_snapshot(value: Any, *, label: str) -> Any:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{label} contains a non-string mapping key")
                result[key] = convert(nested)
            return result
        if isinstance(item, (tuple, list)):
            return [convert(nested) for nested in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a non-finite float")
            return item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise TypeError(f"{label} contains non-JSON value {type(item).__name__}")

    try:
        return json.loads(_canonical_json(convert(value)))
    except (TypeError, ValueError) as exc:
        raise PublicOnlineResolveError(f"{label} is not canonical JSON data") from exc


def _self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_snapshot(dict(payload), label="self-hashed payload")
    result["binding_sha256"] = _canonical_sha256(result)
    return result


def _verify_self_hash(value: Any, *, label: str) -> dict[str, Any]:
    snapshot = _json_snapshot(value, label=label)
    if not isinstance(snapshot, dict):
        raise PublicOnlineResolveError(f"{label} must be a JSON object")
    declared = snapshot.get("binding_sha256")
    _require_sha256(declared, label=f"{label}.binding_sha256")
    content = dict(snapshot)
    del content["binding_sha256"]
    if _canonical_sha256(content) != declared:
        raise PublicOnlineResolveError(f"{label} self-hash mismatch")
    return snapshot


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.resolve(strict=True).read_bytes()).hexdigest()
    except OSError as exc:
        raise PublicOnlineResolveError(f"cannot bind file {path}") from exc


def _path_commitment(path: str | os.PathLike[str]) -> str:
    normalized = str(Path(path).expanduser().resolve())
    return hashlib.sha256(
        f"ofc-online-resolve-path-v1\x00{normalized}".encode("utf-8")
    ).hexdigest()


def _require_solver_identity() -> None:
    if _multi_root_module.solve_multi_root_external_sampling_mccfr is not _CANONICAL_SOLVER:
        raise PublicOnlineResolveError("canonical multi-root solver callable was overridden")


def _wrapper_callable_binding(callables: Mapping[str, Any]) -> dict[str, Any]:
    return _CANONICAL_RUNTIME_SEMANTIC_GRAPH(
        callables,
        data_roots={
            "ONLINE_RESOLVE_REMAINING_LIMITATIONS": (
                _CANONICAL_ONLINE_RESOLVE_REMAINING_LIMITATIONS
            ),
            "ONLINE_RESOLVE_SCHEMA": ONLINE_RESOLVE_SCHEMA,
            "SCOPED_POLICY_SELECTION_SCHEMA": SCOPED_POLICY_SELECTION_SCHEMA,
            "ONLINE_RESOLVE_METHOD": ONLINE_RESOLVE_METHOD,
        },
    )


def _require_wrapper_runtime_contract() -> str:
    callables = _CANONICAL_WRAPPER_CALLABLES
    runtime_roots = _CANONICAL_WRAPPER_RUNTIME_ROOTS
    expected_sha256 = _CANONICAL_WRAPPER_RUNTIME_SEMANTIC_BINDING_SHA256
    if (
        callables is None
        or runtime_roots is None
        or expected_sha256 is None
    ):  # pragma: no cover - import defense
        raise PublicOnlineResolveError("wrapper runtime contract is not initialized")
    drift = sorted(
        name for name, canonical in callables.items() if globals().get(name) is not canonical
    )
    if drift:
        raise PublicOnlineResolveError(
            f"online wrapper runtime callables were overridden: {drift}"
        )
    if tuple(ONLINE_RESOLVE_REMAINING_LIMITATIONS) != (
        _CANONICAL_ONLINE_RESOLVE_REMAINING_LIMITATIONS
    ):
        raise PublicOnlineResolveError("online wrapper limitations were overridden")
    external_drift = []
    if _mixture_module.verify_compiled_public_root_mixture is not (
        _CANONICAL_VERIFY_COMPILED_MIXTURE
    ) or verify_compiled_public_root_mixture is not _CANONICAL_VERIFY_COMPILED_MIXTURE:
        external_drift.append("verify_compiled_public_root_mixture")
    if _action_space_module.get_turn_actions is not _CANONICAL_GET_TURN_ACTIONS:
        external_drift.append("get_turn_actions")
    if _exact_late_module.action_key is not _CANONICAL_ACTION_KEY:
        external_drift.append("action_key")
    if _multi_root_module._python_function_binding is not _CANONICAL_PYTHON_FUNCTION_BINDER:
        external_drift.append("python_function_binding")
    if _multi_root_module._checkpoint_content_sha256 is not _CANONICAL_CONTENT_SHA256:
        external_drift.append("checkpoint_content_sha256")
    if _multi_root_module._runtime_semantic_graph is not (
        _CANONICAL_RUNTIME_SEMANTIC_GRAPH
    ):
        external_drift.append("runtime_semantic_graph")
    if _multi_root_module.serialize_strategy_profile is not (
        _CANONICAL_SERIALIZE_STRATEGY_PROFILE
    ):
        external_drift.append("serialize_strategy_profile")
    if external_drift:
        raise PublicOnlineResolveError(
            f"online wrapper external runtime callables were overridden: {external_drift}"
        )
    live = _wrapper_callable_binding(runtime_roots)
    if live["binding_sha256"] != expected_sha256:
        raise PublicOnlineResolveError("online wrapper runtime semantic binding drifted")
    return expected_sha256


def _live_source_binding(
    context: PublicRootContext,
    *,
    _runtime_contract_attestor: Any = _require_wrapper_runtime_contract,
) -> dict[str, Any]:
    wrapper_runtime_sha256 = _runtime_contract_attestor()
    _require_solver_identity()
    context.bindings.verify_live()
    context_binding = context.bindings.to_canonical_dict()
    source_rows = context_binding["source_sha256s"]
    wrapper_path = Path(__file__)
    payload = {
        "schema": "ofc_public_online_resolve_source_binding/v1",
        "canonical_solver_callable_identity_verified": True,
        "canonical_solver_source_binding_sha256": context.bindings.canonical_solver_source_binding_sha256,
        "canonical_solver_runtime_semantic_binding_sha256": context.bindings.canonical_solver_runtime_semantic_binding_sha256,
        "compiled_context_source_set_sha256": context_binding["source_set_sha256"],
        "compiled_mixture_source_sha256": source_rows["public_root_mixture_compiler"],
        "multi_root_solver_source_sha256": source_rows["multi_root_mccfr"],
        "online_resolve_wrapper_source_sha256": _file_sha256(wrapper_path),
        "online_resolve_wrapper_runtime_semantic_binding_sha256": (
            wrapper_runtime_sha256
        ),
    }
    return _self_hashed(payload)


def _profile_table_sha256(
    table: Mapping[InfoSetKey, Mapping[str, float]],
    *,
    label: str,
) -> str:
    if not isinstance(table, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, InfoSetKey) for key in table):
        raise TypeError(f"{label} keys must be InfoSetKey")
    rows: list[dict[str, Any]] = []
    for key in sorted(table, key=lambda item: (item.digest(), item.canonical_json())):
        actions = table[key]
        if not isinstance(actions, Mapping) or not actions:
            raise PublicOnlineResolveError(f"{label} rows must be non-empty mappings")
        action_rows: list[dict[str, str]] = []
        for action_id in sorted(actions):
            if not isinstance(action_id, str) or not action_id:
                raise PublicOnlineResolveError(f"{label} action IDs must be non-empty strings")
            value = actions[action_id]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} values must be finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise PublicOnlineResolveError(f"{label} values must be finite")
            action_rows.append({"action_id": action_id, "value_hex": number.hex()})
        rows.append(
            {
                "infoset_canonical_json": key.canonical_json(),
                "infoset_sha256": key.digest(),
                "actions": action_rows,
            }
        )
    return _canonical_sha256(rows)


def _support_table_sha256(
    table: Mapping[InfoSetKey, int],
) -> str:
    if not isinstance(table, Mapping):
        raise TypeError("infoset_root_support_count must be a mapping")
    if any(not isinstance(key, InfoSetKey) for key in table):
        raise TypeError("infoset support keys must be InfoSetKey")
    rows = []
    for key in sorted(table, key=lambda item: (item.digest(), item.canonical_json())):
        count = table[key]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PublicOnlineResolveError("infoset support counts must be positive ints")
        rows.append({"infoset_sha256": key.digest(), "root_support_count": count})
    return _canonical_sha256(rows)


def _shared_infosets_sha256(keys: tuple[InfoSetKey, ...]) -> str:
    if not isinstance(keys, tuple):
        raise TypeError("shared_across_roots_infosets must be a tuple")
    digests = []
    for key in keys:
        if not isinstance(key, InfoSetKey):
            raise TypeError("shared infosets must be InfoSetKey")
        digests.append({"canonical_json": key.canonical_json(), "sha256": key.digest()})
    stable = sorted(digests, key=lambda row: (row["sha256"], row["canonical_json"]))
    if len({row["canonical_json"] for row in stable}) != len(stable):
        raise PublicOnlineResolveError("shared infosets contain duplicates")
    return _canonical_sha256(stable)


def _legal_action_ids_for_key(key: InfoSetKey) -> tuple[str, ...]:
    if not isinstance(key, InfoSetKey):
        raise TypeError("legal-action key must be InfoSetKey")
    actor_rows = key.board_bb if key.actor == "bb" else key.board_btn
    board = Board(
        top=list(actor_rows[0]),
        middle=list(actor_rows[1]),
        bottom=list(actor_rows[2]),
    )
    action_ids = tuple(
        sorted(
            _CANONICAL_ACTION_KEY(action)
            for action in _CANONICAL_GET_TURN_ACTIONS(
                list(key.current_draw),
                board,
            )
        )
    )
    if not action_ids:
        raise PublicOnlineResolveError("solver infoset has no canonical legal actions")
    if len(set(action_ids)) != len(action_ids):
        raise PublicOnlineResolveError(
            "canonical legal action IDs are not unique for solver infoset"
        )
    return action_ids


def _validate_solver_policy_tables(
    result: MultiRootExternalSamplingMccfrResult,
) -> None:
    average_keys = set(result.average_strategy)
    current_keys = set(result.current_strategy)
    regret_keys = set(result.cumulative_regret_plus)
    support_keys = set(result.infoset_root_support_count)
    if not average_keys or not (
        average_keys == current_keys == regret_keys == support_keys
    ):
        raise PublicOnlineResolveError(
            "solver policy, regret, and support infoset key sets must match exactly"
        )

    for key in average_keys:
        legal_action_ids = _legal_action_ids_for_key(key)
        legal_action_set = set(legal_action_ids)
        rows = (
            ("average strategy", result.average_strategy[key], True),
            ("current strategy", result.current_strategy[key], True),
            ("cumulative regret plus", result.cumulative_regret_plus[key], False),
        )
        for label, row, is_strategy in rows:
            if not isinstance(row, Mapping) or set(row) != legal_action_set:
                raise PublicOnlineResolveError(
                    f"{label} action set does not exactly match canonical legal actions"
                )
            values: list[float] = []
            for action_id in legal_action_ids:
                value = row[action_id]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise PublicOnlineResolveError(f"{label} contains a non-numeric value")
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise PublicOnlineResolveError(
                        f"{label} contains a negative or non-finite value"
                    )
                values.append(number)
            if is_strategy and not math.isclose(
                math.fsum(values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise PublicOnlineResolveError(f"{label} probabilities do not sum to one")


def _compiled_support_solver_binding(
    mixture: CompiledPublicRootMixture,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support_authorization = mixture.audit_manifest.get("support_authorization")
    expected_authorization = {
        "authorization_class": (
            _mixture_module.EXPLICIT_SUPPORT_AUTHORIZATION_CLASS
        ),
        "proposal_origin_commitments_verified": False,
        "promoted_behavior_likelihood_verified": False,
        "conditional_range_posterior_verified": False,
        "posterior_join_normalization_verified": False,
        "authorized_as_production_mccfr_prior": False,
        "required_production_api": (
            "compile_posterior_joined_public_root_mixture"
        ),
    }
    if not isinstance(support_authorization, Mapping) or dict(
        support_authorization
    ) != expected_authorization:
        raise PublicOnlineResolveError(
            "compiled explicit support authorization is missing or overclaims production"
        )
    entries = tuple(sorted(mixture.entries, key=lambda entry: entry.root_id))
    root_prior_manifest = {
        "schema": "ofc_multi_root_exact_prior/v1",
        "sampling_contract": _multi_root_module.SUPER_ROOT_SAMPLING_CONTRACT,
        "roots": [
            {
                "root_id_sha256": entry.root_id_sha256,
                "prior_mass_exact": (
                    f"{entry.prior_mass.numerator}/{entry.prior_mass.denominator}"
                ),
                "observation_sha256": entry.adapter.observation.digest(),
                "conditional_particle_count": len(entry.adapter.root_distribution),
                "range_content_sha256": entry.adapter.root_range.range_content_sha256,
                "range_build_sha256": entry.adapter.root_range.range_build_sha256,
            }
            for entry in entries
        ],
    }
    audit_rows = mixture.audit_manifest.get("support_rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != len(entries):
        raise PublicOnlineResolveError("compiled mixture support audit rows are invalid")
    adapter_by_root: dict[str, str] = {}
    for row in audit_rows:
        if not isinstance(row, Mapping):
            raise PublicOnlineResolveError("compiled support audit row is invalid")
        root_sha256 = row.get("root_id_sha256")
        adapter_sha256 = row.get("canonical_full_card_adapter_binding_sha256")
        _require_sha256(root_sha256, label="compiled support root ID SHA256")
        _require_sha256(
            adapter_sha256,
            label="compiled support adapter binding SHA256",
        )
        if root_sha256 in adapter_by_root:
            raise PublicOnlineResolveError("compiled support audit has duplicate roots")
        adapter_by_root[root_sha256] = adapter_sha256
    expected_root_ids = {entry.root_id_sha256 for entry in entries}
    if set(adapter_by_root) != expected_root_ids:
        raise PublicOnlineResolveError("compiled support adapter roots do not match entries")
    support_binding = _self_hashed(
        {
            "schema": "ofc_public_online_compiled_support_solver_binding/v1",
            "compiled_support_sha256": mixture.support_sha256,
            "compiled_mixture_audit_manifest_sha256": mixture.audit_manifest_sha256,
            "support_authorization": expected_authorization,
            "root_prior_manifest_sha256": _canonical_sha256(root_prior_manifest),
            "adapter_binding_sha256s": [
                {
                    "root_id_sha256": entry.root_id_sha256,
                    "adapter_binding_sha256": adapter_by_root[entry.root_id_sha256],
                }
                for entry in entries
            ],
        }
    )
    return root_prior_manifest, support_binding


def _solver_result_binding(
    result: MultiRootExternalSamplingMccfrResult,
    *,
    mixture: CompiledPublicRootMixture,
) -> dict[str, Any]:
    if type(result) is not MultiRootExternalSamplingMccfrResult:
        raise TypeError("solver result must be MultiRootExternalSamplingMccfrResult")
    _validate_solver_policy_tables(result)
    average_json = _multi_root_module.serialize_strategy_profile(result.average_strategy)
    current_json = _multi_root_module.serialize_strategy_profile(result.current_strategy)
    average_sha256 = hashlib.sha256(average_json.encode("utf-8")).hexdigest()
    current_sha256 = hashlib.sha256(current_json.encode("utf-8")).hexdigest()
    if average_json != result.average_strategy_json or average_sha256 != result.average_strategy_sha256:
        raise PublicOnlineResolveError("average strategy serialization/hash mismatch")
    if current_json != result.current_strategy_json or current_sha256 != result.current_strategy_sha256:
        raise PublicOnlineResolveError("current strategy serialization/hash mismatch")
    expected_root_count = len(mixture.entries)
    if result.root_count != expected_root_count:
        raise PublicOnlineResolveError("solver result root count does not match mixture")
    metadata = _json_snapshot(result.metadata, label="solver metadata")
    stats = _json_snapshot(result.sampling_stats, label="solver sampling stats")
    required_false = (
        "promotion_eligible",
        "runtime_integrated",
        "global_unseen_state_policy_claim",
        "exact_exploitability_computed",
    )
    for key in required_false:
        if metadata.get(key) is not False:
            raise PublicOnlineResolveError(f"solver metadata requires {key}=false")
    if metadata.get("policy_scope") != "supplied_root_set_online_tabular_solve":
        raise PublicOnlineResolveError("solver policy scope is not supplied-root online solve")
    if metadata.get("average_strategy_sha256") != average_sha256:
        raise PublicOnlineResolveError("solver metadata average strategy hash mismatch")
    if metadata.get("current_strategy_sha256") != current_sha256:
        raise PublicOnlineResolveError("solver metadata current strategy hash mismatch")
    if metadata.get("root_count") != expected_root_count:
        raise PublicOnlineResolveError("solver metadata root count mismatch")
    root_prior_manifest, compiled_support_binding = _compiled_support_solver_binding(
        mixture
    )
    if metadata.get("root_prior_manifest") != root_prior_manifest:
        raise PublicOnlineResolveError(
            "solver root-prior manifest does not match compiled support"
        )
    root_prior_sha256 = _canonical_sha256(root_prior_manifest)
    if metadata.get("root_prior_manifest_sha256") != root_prior_sha256:
        raise PublicOnlineResolveError("solver root-prior manifest SHA256 mismatch")
    if result.encountered_infosets != len(result.average_strategy):
        raise PublicOnlineResolveError("solver encountered infoset count mismatch")
    payload = {
        "schema": "ofc_public_online_solver_result_binding/v1",
        "iterations": result.iterations,
        "traversals": result.traversals,
        "seed": result.seed,
        "root_count": result.root_count,
        "encountered_infosets": result.encountered_infosets,
        "average_strategy_sha256": average_sha256,
        "current_strategy_sha256": current_sha256,
        "cumulative_regret_plus_sha256": _profile_table_sha256(
            result.cumulative_regret_plus,
            label="cumulative_regret_plus",
        ),
        "infoset_root_support_count_sha256": _support_table_sha256(
            result.infoset_root_support_count
        ),
        "shared_across_roots_infosets_sha256": _shared_infosets_sha256(
            result.shared_across_roots_infosets
        ),
        "sampling_stats_sha256": _canonical_sha256(stats),
        "metadata_sha256": _canonical_sha256(metadata),
        "solver_method": metadata.get("method"),
        "compiled_support_solver_binding_sha256": compiled_support_binding[
            "binding_sha256"
        ],
        "root_prior_manifest_sha256": root_prior_sha256,
        "solver_checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "solver_checkpoint_source_binding_sha256": metadata.get(
            "checkpoint_source_binding_sha256"
        ),
    }
    _require_sha256(
        payload["solver_checkpoint_sha256"],
        label="solver checkpoint SHA256",
        allow_none=True,
    )
    _require_sha256(
        payload["solver_checkpoint_source_binding_sha256"],
        label="solver checkpoint source binding SHA256",
        allow_none=True,
    )
    return _self_hashed(payload)


def _checkpoint_envelope(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.resolve(strict=True).read_bytes()
    except OSError as exc:
        raise PublicOnlineResolveError(f"{label} checkpoint file is unavailable") from exc
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PublicOnlineResolveError(f"{label} checkpoint is not UTF-8 JSON") from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise PublicOnlineResolveError(f"{label} checkpoint envelope is invalid")
    declared = envelope.get("checkpoint_sha256")
    _require_sha256(declared, label=f"{label} checkpoint payload SHA256")
    computed = _canonical_sha256(envelope["payload"])
    if declared != computed:
        raise PublicOnlineResolveError(f"{label} checkpoint payload hash mismatch")
    return envelope, hashlib.sha256(raw).hexdigest()


def _prepare_checkpoint_binding(
    *,
    resume_from: str | os.PathLike[str] | None,
    checkpoint_path: str | os.PathLike[str] | None,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    if resume_from is None and expected_checkpoint_sha256 is not None:
        raise PublicOnlineResolveError(
            "expected_checkpoint_sha256 is valid only with resume_from"
        )
    if resume_from is not None and expected_checkpoint_sha256 is None:
        raise PublicOnlineResolveError(
            "resume_from requires externally trusted expected_checkpoint_sha256"
        )
    completed_before = 0
    resume_file_sha256 = None
    resume_path_sha256 = None
    resume_payload_sha256 = None
    if resume_from is not None:
        expected = _require_sha256(
            expected_checkpoint_sha256,
            label="expected_checkpoint_sha256",
        )
        resume_path = Path(resume_from)
        envelope, resume_file_sha256 = _checkpoint_envelope(
            resume_path,
            label="resume input",
        )
        resume_payload_sha256 = envelope["checkpoint_sha256"]
        if resume_payload_sha256 != expected:
            raise PublicOnlineResolveError(
                "resume checkpoint does not match externally trusted SHA256"
            )
        completed_before = envelope["payload"].get("completed_iterations")
        if (
            isinstance(completed_before, bool)
            or not isinstance(completed_before, int)
            or completed_before <= 0
        ):
            raise PublicOnlineResolveError(
                "resume checkpoint completed_iterations must be positive"
            )
        resume_path_sha256 = _path_commitment(resume_path)
    output_path_sha256 = (
        _path_commitment(checkpoint_path) if checkpoint_path is not None else None
    )
    return {
        "schema": "ofc_public_online_checkpoint_binding/v1",
        "checkpoint_enabled": resume_from is not None or checkpoint_path is not None,
        "resume_used": resume_from is not None,
        "resume_path_sha256": resume_path_sha256,
        "resume_file_sha256": resume_file_sha256,
        "resume_checkpoint_payload_sha256": resume_payload_sha256,
        "externally_trusted_resume_checkpoint_sha256": expected_checkpoint_sha256,
        "completed_iterations_before": completed_before,
        "checkpoint_write_requested": checkpoint_path is not None,
        "checkpoint_output_path_sha256": output_path_sha256,
    }


def _finalize_checkpoint_binding(
    prepared: Mapping[str, Any],
    *,
    result: MultiRootExternalSamplingMccfrResult,
    checkpoint_path: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    payload = dict(prepared)
    metadata = result.metadata
    final_payload_sha256 = metadata.get("checkpoint_sha256")
    final_source_sha256 = metadata.get("checkpoint_source_binding_sha256")
    enabled = bool(payload["checkpoint_enabled"])
    if bool(metadata.get("checkpoint_enabled")) != enabled:
        raise PublicOnlineResolveError("solver checkpoint-enabled flag mismatch")
    if enabled:
        _require_sha256(final_payload_sha256, label="final checkpoint payload SHA256")
        _require_sha256(
            final_source_sha256,
            label="final checkpoint source binding SHA256",
        )
    elif final_payload_sha256 is not None or final_source_sha256 is not None:
        raise PublicOnlineResolveError("disabled checkpoint solve emitted checkpoint hashes")
    output_file_sha256 = None
    output_persisted = checkpoint_path is not None
    if checkpoint_path is not None:
        envelope, output_file_sha256 = _checkpoint_envelope(
            Path(checkpoint_path),
            label="checkpoint output",
        )
        if envelope["checkpoint_sha256"] != final_payload_sha256:
            raise PublicOnlineResolveError(
                "checkpoint output payload hash does not match solver result"
            )
    payload.update(
        {
            "final_checkpoint_payload_sha256": final_payload_sha256,
            "final_checkpoint_source_binding_sha256": final_source_sha256,
            "checkpoint_output_persisted": output_persisted,
            "checkpoint_output_file_sha256": output_file_sha256,
        }
    )
    return _self_hashed(payload)


@dataclass(frozen=True, slots=True)
class OnlinePublicResolveResult:
    """Shared solver result plus a canonical immutable nonproduction manifest."""

    solver_result: MultiRootExternalSamplingMccfrResult = field(repr=False)
    manifest_json: str = field(repr=False)
    manifest_sha256: str

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.manifest_json))


@dataclass(frozen=True, slots=True)
class ScopedPolicySelection:
    """One exact compiled-root policy selected only after the shared solve."""

    actual_infoset: InfoSetKey = field(repr=False)
    strategy_kind: Literal["average", "current"]
    action_probabilities: Mapping[str, float]
    policy_sha256: str
    selection_manifest_json: str = field(repr=False)
    selection_manifest_sha256: str

    @property
    def selection_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.selection_manifest_json))


def _solve_request(
    *,
    iterations: int,
    seed: int,
    max_infosets: int,
    linear_averaging: bool,
    completed_before: int,
) -> dict[str, Any]:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise PublicOnlineResolveError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(max_infosets, bool)
        or not isinstance(max_infosets, int)
        or max_infosets <= 0
    ):
        raise PublicOnlineResolveError("max_infosets must be a positive integer")
    if not isinstance(linear_averaging, bool):
        raise TypeError("linear_averaging must be bool")
    return {
        "schema": "ofc_public_online_solve_request/v1",
        "additional_iterations": iterations,
        "completed_iterations_before": completed_before,
        "expected_completed_iterations_total": completed_before + iterations,
        "seed": seed,
        "max_infosets": max_infosets,
        "linear_averaging": linear_averaging,
        "actual_infoset_parameter_present": False,
        "actual_infoset_consumed": False,
    }


def _manifest(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    *,
    solve_request: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    solver_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ONLINE_RESOLVE_SCHEMA,
        "method": ONLINE_RESOLVE_METHOD,
        "public_context_digest": context.digest(),
        "compiled_support_sha256": mixture.support_sha256,
        "compiled_mixture_audit_manifest_sha256": mixture.audit_manifest_sha256,
        "support_authorization": dict(
            mixture.audit_manifest["support_authorization"]
        ),
        "authorized_as_production_mccfr_prior": False,
        "solve_request": dict(solve_request),
        "source_binding": dict(source_binding),
        "solver_result_binding": dict(solver_binding),
        "checkpoint_binding": dict(checkpoint_binding),
        "algorithm_validation_only": True,
        "finite_explicit_support_only": True,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "global_unseen_state_policy_claim": False,
        "actual_infoset_consumed_by_mixture_compilation": False,
        "actual_infoset_consumed_by_solver": False,
        "post_solve_exact_membership_selector_only": True,
        "serving_default_changed": False,
        "remaining_limitations": list(
            _CANONICAL_ONLINE_RESOLVE_REMAINING_LIMITATIONS
        ),
    }


def resolve_compiled_public_root_mixture(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    *,
    iterations: int,
    seed: int,
    max_infosets: int,
    linear_averaging: bool = True,
    resume_from: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> OnlinePublicResolveResult:
    """Freshly verify and solve the whole mixture without an actual-hand input."""

    _require_wrapper_runtime_contract()
    verify_compiled_public_root_mixture(context, mixture)
    _require_solver_identity()
    checkpoint_pre = _prepare_checkpoint_binding(
        resume_from=resume_from,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    solve_request = _solve_request(
        iterations=iterations,
        seed=seed,
        max_infosets=max_infosets,
        linear_averaging=linear_averaging,
        completed_before=checkpoint_pre["completed_iterations_before"],
    )
    source_binding = _live_source_binding(context)
    solver_result = _CANONICAL_SOLVER(
        mixture.entries,
        iterations=iterations,
        seed=seed,
        max_infosets=max_infosets,
        linear_averaging=linear_averaging,
        resume_from=resume_from,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if solver_result.iterations != solve_request["expected_completed_iterations_total"]:
        raise PublicOnlineResolveError("solver completed iteration count mismatch")
    if solver_result.seed != seed:
        raise PublicOnlineResolveError("solver seed mismatch")
    if solver_result.metadata.get("max_infosets") != max_infosets:
        raise PublicOnlineResolveError("solver max_infosets mismatch")
    if solver_result.metadata.get("linear_averaging") is not linear_averaging:
        raise PublicOnlineResolveError("solver linear_averaging mismatch")
    solver_binding = _solver_result_binding(
        solver_result,
        mixture=mixture,
    )
    checkpoint_binding = _finalize_checkpoint_binding(
        checkpoint_pre,
        result=solver_result,
        checkpoint_path=checkpoint_path,
    )
    manifest = _manifest(
        context,
        mixture,
        solve_request=solve_request,
        source_binding=source_binding,
        solver_binding=solver_binding,
        checkpoint_binding=checkpoint_binding,
    )
    manifest_json = _canonical_json(manifest)
    resolved = OnlinePublicResolveResult(
        solver_result=solver_result,
        manifest_json=manifest_json,
        manifest_sha256=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )
    verify_online_public_resolve(context, mixture, resolved)
    return resolved


def _validate_solve_and_checkpoint_bindings(
    solve_request: Any,
    checkpoint_binding: Any,
    solver_result: MultiRootExternalSamplingMccfrResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = _json_snapshot(solve_request, label="solve request")
    checkpoint = _verify_self_hash(checkpoint_binding, label="checkpoint binding")
    required_request_keys = {
        "schema",
        "additional_iterations",
        "completed_iterations_before",
        "expected_completed_iterations_total",
        "seed",
        "max_infosets",
        "linear_averaging",
        "actual_infoset_parameter_present",
        "actual_infoset_consumed",
    }
    if not isinstance(request, dict) or set(request) != required_request_keys:
        raise PublicOnlineResolveError("solve request schema mismatch")
    required_checkpoint_keys = {
        "schema",
        "checkpoint_enabled",
        "resume_used",
        "resume_path_sha256",
        "resume_file_sha256",
        "resume_checkpoint_payload_sha256",
        "externally_trusted_resume_checkpoint_sha256",
        "completed_iterations_before",
        "checkpoint_write_requested",
        "checkpoint_output_path_sha256",
        "final_checkpoint_payload_sha256",
        "final_checkpoint_source_binding_sha256",
        "checkpoint_output_persisted",
        "checkpoint_output_file_sha256",
        "binding_sha256",
    }
    if set(checkpoint) != required_checkpoint_keys:
        raise PublicOnlineResolveError("checkpoint binding schema mismatch")
    if checkpoint["schema"] != "ofc_public_online_checkpoint_binding/v1":
        raise PublicOnlineResolveError("checkpoint binding version mismatch")
    if request["schema"] != "ofc_public_online_solve_request/v1":
        raise PublicOnlineResolveError("solve request version mismatch")
    additional = request["additional_iterations"]
    before = request["completed_iterations_before"]
    total = request["expected_completed_iterations_total"]
    if (
        isinstance(additional, bool)
        or not isinstance(additional, int)
        or additional <= 0
        or isinstance(before, bool)
        or not isinstance(before, int)
        or before < 0
        or total != before + additional
        or solver_result.iterations != total
    ):
        raise PublicOnlineResolveError("solve request iteration accounting mismatch")
    if request["seed"] != solver_result.seed:
        raise PublicOnlineResolveError("solve request seed mismatch")
    if request["max_infosets"] != solver_result.metadata.get("max_infosets"):
        raise PublicOnlineResolveError("solve request max_infosets mismatch")
    if request["linear_averaging"] is not solver_result.metadata.get(
        "linear_averaging"
    ):
        raise PublicOnlineResolveError("solve request linear_averaging mismatch")
    if request["actual_infoset_parameter_present"] is not False or request[
        "actual_infoset_consumed"
    ] is not False:
        raise PublicOnlineResolveError("solve request consumed an actual infoset")
    if checkpoint.get("completed_iterations_before") != before:
        raise PublicOnlineResolveError("checkpoint/solve prior iteration mismatch")
    enabled = checkpoint.get("checkpoint_enabled")
    resume_used = checkpoint["resume_used"]
    write_requested = checkpoint["checkpoint_write_requested"]
    output_persisted = checkpoint["checkpoint_output_persisted"]
    if not all(
        isinstance(value, bool)
        for value in (enabled, resume_used, write_requested, output_persisted)
    ):
        raise PublicOnlineResolveError("checkpoint state flags must be booleans")
    if enabled is not (resume_used or write_requested):
        raise PublicOnlineResolveError("checkpoint enabled state is not derivable")
    if output_persisted is not write_requested:
        raise PublicOnlineResolveError("checkpoint persistence state is not derivable")
    if enabled is not solver_result.metadata.get(
        "checkpoint_enabled"
    ):
        raise PublicOnlineResolveError("checkpoint binding enabled flag mismatch")
    optional_hash_fields = (
        "resume_path_sha256",
        "resume_file_sha256",
        "resume_checkpoint_payload_sha256",
        "externally_trusted_resume_checkpoint_sha256",
        "checkpoint_output_path_sha256",
        "final_checkpoint_payload_sha256",
        "final_checkpoint_source_binding_sha256",
        "checkpoint_output_file_sha256",
    )
    for field_name in optional_hash_fields:
        _require_sha256(
            checkpoint[field_name],
            label=f"checkpoint binding {field_name}",
            allow_none=True,
        )
    resume_fields = (
        "resume_path_sha256",
        "resume_file_sha256",
        "resume_checkpoint_payload_sha256",
        "externally_trusted_resume_checkpoint_sha256",
    )
    if resume_used:
        if any(checkpoint[field_name] is None for field_name in resume_fields):
            raise PublicOnlineResolveError("resume checkpoint hashes are incomplete")
        if checkpoint["resume_checkpoint_payload_sha256"] != checkpoint[
            "externally_trusted_resume_checkpoint_sha256"
        ]:
            raise PublicOnlineResolveError("resume checkpoint trust hash mismatch")
        if before <= 0:
            raise PublicOnlineResolveError("resume checkpoint has no prior iterations")
    elif any(checkpoint[field_name] is not None for field_name in resume_fields) or before != 0:
        raise PublicOnlineResolveError("non-resume solve contains resume state")
    output_fields = (
        "checkpoint_output_path_sha256",
        "checkpoint_output_file_sha256",
    )
    if write_requested:
        if any(checkpoint[field_name] is None for field_name in output_fields):
            raise PublicOnlineResolveError("checkpoint output hashes are incomplete")
    elif any(checkpoint[field_name] is not None for field_name in output_fields):
        raise PublicOnlineResolveError("non-writing solve contains output hashes")
    final_fields = (
        "final_checkpoint_payload_sha256",
        "final_checkpoint_source_binding_sha256",
    )
    if enabled:
        if any(checkpoint[field_name] is None for field_name in final_fields):
            raise PublicOnlineResolveError("enabled checkpoint hashes are incomplete")
    elif any(checkpoint[field_name] is not None for field_name in final_fields):
        raise PublicOnlineResolveError("disabled checkpoint contains final hashes")
    if checkpoint.get("final_checkpoint_payload_sha256") != solver_result.metadata.get(
        "checkpoint_sha256"
    ):
        raise PublicOnlineResolveError("checkpoint final payload hash mismatch")
    if checkpoint.get(
        "final_checkpoint_source_binding_sha256"
    ) != solver_result.metadata.get("checkpoint_source_binding_sha256"):
        raise PublicOnlineResolveError("checkpoint source binding hash mismatch")
    return request, checkpoint


def verify_online_public_resolve(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    result: OnlinePublicResolveResult,
    *,
    expected_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Freshly verify live inputs and every committed solver/result hash."""

    _require_wrapper_runtime_contract()
    verify_compiled_public_root_mixture(context, mixture)
    _require_solver_identity()
    if not isinstance(result, OnlinePublicResolveResult):
        raise TypeError("result must be OnlinePublicResolveResult")
    try:
        manifest = json.loads(result.manifest_json)
    except json.JSONDecodeError as exc:
        raise PublicOnlineResolveError("online resolve manifest is not JSON") from exc
    if _canonical_json(manifest) != result.manifest_json:
        raise PublicOnlineResolveError("online resolve manifest is not canonical JSON")
    computed_manifest_sha256 = hashlib.sha256(
        result.manifest_json.encode("utf-8")
    ).hexdigest()
    if computed_manifest_sha256 != result.manifest_sha256:
        raise PublicOnlineResolveError("online resolve manifest SHA256 mismatch")
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, label="expected_manifest_sha256")
        if result.manifest_sha256 != expected_manifest_sha256:
            raise PublicOnlineResolveError(
                "online resolve manifest does not match externally trusted SHA256"
            )
    required_true_flags = (
        "algorithm_validation_only",
        "finite_explicit_support_only",
        "post_solve_exact_membership_selector_only",
    )
    required_false_flags = (
        "promotion_eligible",
        "runtime_integrated",
        "global_unseen_state_policy_claim",
        "actual_infoset_consumed_by_mixture_compilation",
        "actual_infoset_consumed_by_solver",
        "authorized_as_production_mccfr_prior",
        "serving_default_changed",
    )
    for flag in required_true_flags:
        if manifest.get(flag) is not True:
            raise PublicOnlineResolveError(f"online resolve manifest requires {flag}=true")
    for flag in required_false_flags:
        if manifest.get(flag) is not False:
            raise PublicOnlineResolveError(f"online resolve manifest requires {flag}=false")
    if manifest.get("remaining_limitations") != list(
        _CANONICAL_ONLINE_RESOLVE_REMAINING_LIMITATIONS
    ):
        raise PublicOnlineResolveError(
            "online resolve manifest limitations do not match the canonical limitations"
        )
    source_binding = _live_source_binding(context)
    solver_binding = _solver_result_binding(
        result.solver_result,
        mixture=mixture,
    )
    request, checkpoint = _validate_solve_and_checkpoint_bindings(
        manifest.get("solve_request"),
        manifest.get("checkpoint_binding"),
        result.solver_result,
    )
    expected = _manifest(
        context,
        mixture,
        solve_request=request,
        source_binding=source_binding,
        solver_binding=solver_binding,
        checkpoint_binding=checkpoint,
    )
    if manifest != expected:
        raise PublicOnlineResolveError("online resolve manifest content mismatch")
    return MappingProxyType(
        {
            "verified": True,
            "manifest_sha256": result.manifest_sha256,
            "public_context_digest": context.digest(),
            "compiled_support_sha256": mixture.support_sha256,
            "solver_result_binding_sha256": solver_binding["binding_sha256"],
            "algorithm_validation_only": True,
            "authorized_as_production_mccfr_prior": False,
            "promotion_eligible": False,
            "runtime_integrated": False,
            "global_unseen_state_policy_claim": False,
        }
    )


def _public_context_tuple(value: PublicRootContext | InfoSetKey) -> tuple[Any, ...]:
    return (
        value.contract_version,
        value.actor,
        value.turn,
        value.phase,
        value.board_bb,
        value.board_btn,
        value.public_action_history,
        value.fantasy_state,
    )


def _exact_compiled_member(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    actual_infoset: InfoSetKey,
):
    if not isinstance(actual_infoset, InfoSetKey):
        raise TypeError("actual_infoset must be InfoSetKey")
    actual_infoset.canonical_json()
    if _public_context_tuple(actual_infoset) != _public_context_tuple(context):
        raise PublicOnlineResolveError(
            "actual infoset public context does not match compiled mixture"
        )
    matches = tuple(
        entry
        for entry in mixture.entries
        if entry.adapter.observation == actual_infoset
    )
    if not matches:
        raise PublicOnlineResolveError(
            "actual infoset is not a member of the compiled private-type support"
        )
    if len(matches) != 1:
        raise PublicOnlineResolveError(
            "actual infoset matches duplicate compiled observations"
        )
    return matches[0]


def _scoped_policy(
    resolved: OnlinePublicResolveResult,
    actual_infoset: InfoSetKey,
    strategy_kind: str,
    matched_entry: Any,
) -> tuple[dict[str, float], str]:
    if strategy_kind not in ("average", "current"):
        raise PublicOnlineResolveError("strategy_kind must be average or current")
    table = (
        resolved.solver_result.average_strategy
        if strategy_kind == "average"
        else resolved.solver_result.current_strategy
    )
    raw_policy = table.get(actual_infoset)
    if raw_policy is None:
        raise PublicOnlineResolveError(
            "compiled member is missing from the encountered solver policy"
        )
    if matched_entry.adapter.observation != actual_infoset:
        raise PublicOnlineResolveError(
            "selected compiled support entry does not match the actual infoset"
        )
    legal_action_ids = _legal_action_ids_for_key(actual_infoset)
    if set(raw_policy) != set(legal_action_ids):
        raise PublicOnlineResolveError(
            "selected policy action set does not exactly match canonical legal actions"
        )
    policy: dict[str, float] = {}
    for action_id in legal_action_ids:
        probability = float(raw_policy[action_id])
        if not math.isfinite(probability) or probability < 0:
            raise PublicOnlineResolveError("selected policy has invalid probabilities")
        policy[action_id] = probability
    if not policy or not math.isclose(math.fsum(policy.values()), 1.0, abs_tol=1e-9):
        raise PublicOnlineResolveError("selected policy probabilities do not sum to one")
    policy_binding = [
        {"action_id": action_id, "probability_hex": value.hex()}
        for action_id, value in policy.items()
    ]
    return policy, _canonical_sha256(policy_binding)


def _selection_manifest(
    *,
    resolved: OnlinePublicResolveResult,
    actual_infoset: InfoSetKey,
    root_id_sha256: str,
    strategy_kind: str,
    policy_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SCOPED_POLICY_SELECTION_SCHEMA,
        "online_resolve_manifest_sha256": resolved.manifest_sha256,
        "actual_infoset_sha256": actual_infoset.digest(),
        "exact_compiled_observation_match_count": 1,
        "matched_root_id_sha256": root_id_sha256,
        "strategy_kind": strategy_kind,
        "policy_sha256": policy_sha256,
        "actual_infoset_used_after_solve_only": True,
        "actual_infoset_used_for_mixture_compilation": False,
        "actual_infoset_used_for_solver_execution": False,
        "policy_scope": "one_exact_compiled_infoset_member",
        "promotion_eligible": False,
        "runtime_integrated": False,
        "global_unseen_state_policy_claim": False,
    }


def select_scoped_policy_for_actual_infoset(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    resolved: OnlinePublicResolveResult,
    actual_infoset: InfoSetKey,
    *,
    strategy_kind: Literal["average", "current"] = "average",
) -> ScopedPolicySelection:
    """Select one exact member policy after solve; never rerun or alter search."""

    _require_wrapper_runtime_contract()
    verify_online_public_resolve(context, mixture, resolved)
    matched = _exact_compiled_member(context, mixture, actual_infoset)
    policy, policy_sha256 = _scoped_policy(
        resolved,
        actual_infoset,
        strategy_kind,
        matched,
    )
    selection_manifest = _selection_manifest(
        resolved=resolved,
        actual_infoset=actual_infoset,
        root_id_sha256=matched.root_id_sha256,
        strategy_kind=strategy_kind,
        policy_sha256=policy_sha256,
    )
    selection_json = _canonical_json(selection_manifest)
    selection = ScopedPolicySelection(
        actual_infoset=actual_infoset,
        strategy_kind=strategy_kind,
        action_probabilities=MappingProxyType(policy),
        policy_sha256=policy_sha256,
        selection_manifest_json=selection_json,
        selection_manifest_sha256=hashlib.sha256(
            selection_json.encode("utf-8")
        ).hexdigest(),
    )
    verify_scoped_policy_selection(context, mixture, resolved, selection)
    return selection


def verify_scoped_policy_selection(
    context: PublicRootContext,
    mixture: CompiledPublicRootMixture,
    resolved: OnlinePublicResolveResult,
    selection: ScopedPolicySelection,
    *,
    expected_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Freshly verify exact membership, selected row, hashes, and scope flags."""

    _require_wrapper_runtime_contract()
    verify_online_public_resolve(context, mixture, resolved)
    if not isinstance(selection, ScopedPolicySelection):
        raise TypeError("selection must be ScopedPolicySelection")
    matched = _exact_compiled_member(context, mixture, selection.actual_infoset)
    expected_policy, expected_policy_sha256 = _scoped_policy(
        resolved,
        selection.actual_infoset,
        selection.strategy_kind,
        matched,
    )
    actual_policy = dict(selection.action_probabilities)
    if actual_policy != expected_policy:
        raise PublicOnlineResolveError("scoped selection policy row mismatch")
    if selection.policy_sha256 != expected_policy_sha256:
        raise PublicOnlineResolveError("scoped selection policy SHA256 mismatch")
    try:
        manifest = json.loads(selection.selection_manifest_json)
    except json.JSONDecodeError as exc:
        raise PublicOnlineResolveError("selection manifest is not JSON") from exc
    if _canonical_json(manifest) != selection.selection_manifest_json:
        raise PublicOnlineResolveError("selection manifest is not canonical JSON")
    computed_sha256 = hashlib.sha256(
        selection.selection_manifest_json.encode("utf-8")
    ).hexdigest()
    if computed_sha256 != selection.selection_manifest_sha256:
        raise PublicOnlineResolveError("selection manifest SHA256 mismatch")
    if expected_manifest_sha256 is not None:
        _require_sha256(
            expected_manifest_sha256,
            label="expected selection manifest SHA256",
        )
        if selection.selection_manifest_sha256 != expected_manifest_sha256:
            raise PublicOnlineResolveError(
                "selection manifest does not match externally trusted SHA256"
            )
    required_selection_values = {
        "actual_infoset_used_after_solve_only": True,
        "actual_infoset_used_for_mixture_compilation": False,
        "actual_infoset_used_for_solver_execution": False,
        "exact_compiled_observation_match_count": 1,
        "policy_scope": "one_exact_compiled_infoset_member",
        "promotion_eligible": False,
        "runtime_integrated": False,
        "global_unseen_state_policy_claim": False,
    }
    for field_name, required_value in required_selection_values.items():
        if manifest.get(field_name) != required_value or type(
            manifest.get(field_name)
        ) is not type(required_value):
            raise PublicOnlineResolveError(
                f"selection manifest requires {field_name}={required_value!r}"
            )
    expected_manifest = _selection_manifest(
        resolved=resolved,
        actual_infoset=selection.actual_infoset,
        root_id_sha256=matched.root_id_sha256,
        strategy_kind=selection.strategy_kind,
        policy_sha256=expected_policy_sha256,
    )
    if manifest != expected_manifest:
        raise PublicOnlineResolveError("selection manifest content mismatch")
    return MappingProxyType(
        {
            "verified": True,
            "selection_manifest_sha256": selection.selection_manifest_sha256,
            "actual_infoset_sha256": selection.actual_infoset.digest(),
            "strategy_kind": selection.strategy_kind,
            "policy_sha256": selection.policy_sha256,
            "promotion_eligible": False,
            "runtime_integrated": False,
            "global_unseen_state_policy_claim": False,
        }
    )


_CANONICAL_WRAPPER_CALLABLES = MappingProxyType(
    {
        "_compiled_support_solver_binding": _compiled_support_solver_binding,
        "_exact_compiled_member": _exact_compiled_member,
        "_finalize_checkpoint_binding": _finalize_checkpoint_binding,
        "_legal_action_ids_for_key": _legal_action_ids_for_key,
        "_live_source_binding": _live_source_binding,
        "_manifest": _manifest,
        "_prepare_checkpoint_binding": _prepare_checkpoint_binding,
        "_public_context_tuple": _public_context_tuple,
        "_require_solver_identity": _require_solver_identity,
        "_require_wrapper_runtime_contract": _require_wrapper_runtime_contract,
        "_scoped_policy": _scoped_policy,
        "_selection_manifest": _selection_manifest,
        "_solve_request": _solve_request,
        "_solver_result_binding": _solver_result_binding,
        "_validate_solve_and_checkpoint_bindings": (
            _validate_solve_and_checkpoint_bindings
        ),
        "_validate_solver_policy_tables": _validate_solver_policy_tables,
        "_wrapper_callable_binding": _wrapper_callable_binding,
    }
)
_CANONICAL_WRAPPER_RUNTIME_ROOTS = MappingProxyType(
    {
        name: function
        for name, function in _CANONICAL_WRAPPER_CALLABLES.items()
        if name != "_require_wrapper_runtime_contract"
    }
)
_CANONICAL_WRAPPER_RUNTIME_SEMANTIC_BINDING_SHA256 = _wrapper_callable_binding(
    _CANONICAL_WRAPPER_RUNTIME_ROOTS
)["binding_sha256"]


__all__ = [
    "ONLINE_RESOLVE_METHOD",
    "ONLINE_RESOLVE_REMAINING_LIMITATIONS",
    "ONLINE_RESOLVE_SCHEMA",
    "OnlinePublicResolveResult",
    "PublicOnlineResolveError",
    "SCOPED_POLICY_SELECTION_SCHEMA",
    "ScopedPolicySelection",
    "resolve_compiled_public_root_mixture",
    "select_scoped_policy_for_actual_infoset",
    "verify_online_public_resolve",
    "verify_scoped_policy_selection",
]
