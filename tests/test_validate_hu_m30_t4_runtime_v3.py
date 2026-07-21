from __future__ import annotations

import pytest

from ofc_regular.validate_hu_m30_t4_runtime_v3 import (
    PILOT_SCHEMA_V3,
    SECOND_NATIVE_P99_GATE_MS,
    SECOND_TOTAL_P99_SANITY_GATE_MS,
    compute_v3_result,
)


def _make_v1_result(
    *,
    second_native_ms: list[float],
    second_total_ms: list[float],
    semantic_ok: bool = True,
    first_latency_ok: bool = True,
    v1_second_gate: bool = False,
) -> dict:
    rows = [
        {
            "seat": "second",
            "native_latency_ms": native,
            "total_latency_ms": total,
        }
        for native, total in zip(second_native_ms, second_total_ms, strict=True)
    ]
    rows.extend(
        {"seat": "first", "native_latency_ms": 5.0, "total_latency_ms": 7.0}
        for _ in second_native_ms
    )
    return {
        "schema": "hu_m30_t4_runtime_pilot_v1",
        "status": "fail",
        "gates": {
            "python_reference_parity": semantic_ok,
            "scalar_batch_parity": semantic_ok,
            "deterministic_rerun": semantic_ok,
            "first_p95_latency": first_latency_ok,
            "first_p99_latency": first_latency_ok,
            "second_p99_latency": v1_second_gate,
        },
        "all_gates_passed": False,
        "rows": rows,
    }


def test_native_within_gate_passes_even_when_total_spikes():
    result = compute_v3_result(
        _make_v1_result(
            second_native_ms=[0.6] * 49 + [1.3],
            second_total_ms=[1.3] * 49 + [4.5],
        )
    )
    assert result["schema"] == PILOT_SCHEMA_V3
    assert result["status"] == "pass"
    assert result["all_gates_passed"] is True
    assert result["gates"]["second_native_p99_latency"] is True
    assert result["gates"]["second_total_p99_latency_sanity"] is True
    assert "second_p99_latency" not in result["gates"]
    assert result["v1_reference_gates"]["second_p99_latency"] is False


def test_native_p99_over_gate_fails():
    result = compute_v3_result(
        _make_v1_result(
            second_native_ms=[SECOND_NATIVE_P99_GATE_MS + 0.5] * 50,
            second_total_ms=[3.0] * 50,
        )
    )
    assert result["status"] == "fail"
    assert result["gates"]["second_native_p99_latency"] is False


def test_total_p99_over_sanity_bound_fails():
    result = compute_v3_result(
        _make_v1_result(
            second_native_ms=[0.6] * 50,
            second_total_ms=[SECOND_TOTAL_P99_SANITY_GATE_MS + 1.0] * 50,
        )
    )
    assert result["status"] == "fail"
    assert result["gates"]["second_total_p99_latency_sanity"] is False


def test_semantic_gate_failure_propagates():
    result = compute_v3_result(
        _make_v1_result(
            second_native_ms=[0.6] * 50,
            second_total_ms=[1.3] * 50,
            semantic_ok=False,
        )
    )
    assert result["status"] == "fail"
    assert result["gates"]["python_reference_parity"] is False


def test_first_seat_latency_gates_remain_binding():
    result = compute_v3_result(
        _make_v1_result(
            second_native_ms=[0.6] * 50,
            second_total_ms=[1.3] * 50,
            first_latency_ok=False,
        )
    )
    assert result["status"] == "fail"
    assert result["gates"]["first_p95_latency"] is False


def test_percentile_uses_p99_not_max():
    # 100 second-seat rows: p99 is the 99th ordered value, so a single
    # extreme outlier beyond it must not fail the native gate.
    result = compute_v3_result(
        _make_v1_result(
            second_native_ms=[0.6] * 98 + [1.9, 50.0],
            second_total_ms=[1.3] * 100,
        )
    )
    assert result["gates"]["second_native_p99_latency"] is True
    decomposition = result["second_seat_latency_decomposition"]
    assert decomposition["native_p99_ms"] == 1.9
    assert decomposition["native_max_ms"] == 50.0


def test_missing_second_rows_is_rejected():
    payload = _make_v1_result(
        second_native_ms=[0.6],
        second_total_ms=[1.3],
    )
    payload["rows"] = [row for row in payload["rows"] if row["seat"] != "second"]
    with pytest.raises(ValueError):
        compute_v3_result(payload)


def test_missing_v1_second_gate_is_rejected():
    payload = _make_v1_result(
        second_native_ms=[0.6],
        second_total_ms=[1.3],
    )
    del payload["gates"]["second_p99_latency"]
    with pytest.raises(ValueError):
        compute_v3_result(payload)
