"""Run the v3 M3.0 exact-T4 pilot with the second-seat gate on native latency.

Pre-registered in configs/hu_joint_policy_m30_t4_runtime_v3_candidate.json.
The v1 validator (validate_hu_m30_t4_runtime.py) stays frozen; this module
reuses its observation generation and pilot loop, then applies the v3 gate
vector: semantic and first-seat gates unchanged, the second-seat 2.0 ms p99
gate evaluated on native engine latency, plus a 10.0 ms second-seat total
latency sanity bound. The v1 gate vector is kept in the artifact as
``v1_reference_gates`` and is not binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .hu_m3_t4_runtime import HuM3T4ExactSolver, HuM3T4RuntimeConfig
from .validate_hu_m30_t4_runtime import (
    FIRST_P95_GATE_MS,
    FIRST_P99_GATE_MS,
    _percentile,
    _write_json_atomic,
    run_pilot,
)

PILOT_SCHEMA_V3 = "hu_m30_t4_runtime_pilot_v3"
SECOND_NATIVE_P99_GATE_MS = 2.0
SECOND_TOTAL_P99_SANITY_GATE_MS = 10.0
_V1_SECOND_TOTAL_GATE = "second_p99_latency"


def compute_v3_result(v1_result: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a v1 pilot payload with the pre-registered v3 gate vector."""

    rows = v1_result["rows"]
    second_rows = [row for row in rows if row["seat"] == "second"]
    if not second_rows:
        raise ValueError("pilot payload contains no second-seat rows")
    native = sorted(float(row["native_latency_ms"]) for row in second_rows)
    total = sorted(float(row["total_latency_ms"]) for row in second_rows)

    v1_gates = dict(v1_result["gates"])
    if _V1_SECOND_TOTAL_GATE not in v1_gates:
        raise ValueError("pilot payload is missing the v1 second-seat gate")
    gates = {
        name: value
        for name, value in v1_gates.items()
        if name != _V1_SECOND_TOTAL_GATE
    }
    second_native_p99 = _percentile(native, 0.99)
    second_total_p99 = _percentile(total, 0.99)
    gates["second_native_p99_latency"] = (
        second_native_p99 <= SECOND_NATIVE_P99_GATE_MS
    )
    gates["second_total_p99_latency_sanity"] = (
        second_total_p99 <= SECOND_TOTAL_P99_SANITY_GATE_MS
    )

    result = dict(v1_result)
    result["schema"] = PILOT_SCHEMA_V3
    result["v1_schema"] = v1_result["schema"]
    result["v1_reference_gates"] = v1_gates
    result["gates"] = gates
    result["all_gates_passed"] = all(gates.values())
    result["status"] = "pass" if result["all_gates_passed"] else "fail"
    result["second_seat_latency_decomposition"] = {
        "native_p50_ms": _percentile(native, 0.50),
        "native_p95_ms": _percentile(native, 0.95),
        "native_p99_ms": second_native_p99,
        "native_max_ms": native[-1],
        "total_p50_ms": _percentile(total, 0.50),
        "total_p95_ms": _percentile(total, 0.95),
        "total_p99_ms": second_total_p99,
        "total_max_ms": total[-1],
        "native_gate_ms": SECOND_NATIVE_P99_GATE_MS,
        "total_sanity_gate_ms": SECOND_TOTAL_P99_SANITY_GATE_MS,
        "first_total_p95_gate_ms": FIRST_P95_GATE_MS,
        "first_total_p99_gate_ms": FIRST_P99_GATE_MS,
    }
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--states", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-stride", type=int, required=True)
    parser.add_argument("--python-parity-roots", type=int, default=4)
    parser.add_argument("--determinism-roots", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    solver = HuM3T4ExactSolver(
        HuM3T4RuntimeConfig(
            library_path=args.library,
            expected_library_sha256=args.library_sha256,
        )
    )
    v1_result = run_pilot(
        solver=solver,
        states=args.states,
        seed=args.seed,
        seed_stride=args.seed_stride,
        python_parity_roots=args.python_parity_roots,
        determinism_roots=args.determinism_roots,
    )
    result = compute_v3_result(v1_result)
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
        )
    )
    if not result["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
