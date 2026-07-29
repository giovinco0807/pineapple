"""Python/Rust parity harness for the exact T4 first-seat solver.

The Python resolver ``ai/tutor/t4_bb_exact_resolver.py`` is the semantic
oracle; the Rust crate ``ai/rust_solver/t4_first_exact`` must reproduce every
legal action's EV.  Positions are drawn from the probe's random generator so
Joker 0/1/2 strata and unfiltered board shapes are all covered.

Usage:
    python -m ai.tutor.t4_first_exact_parity --roots 12 --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board

SCHEMA = "ofc_t4_first_exact_parity/v1"


def _solver_path(workspace_root: Path) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return (
        workspace_root
        / "ai"
        / "rust_solver"
        / "target"
        / "release"
        / f"t4_first_exact{suffix}"
    )


def python_action_evs(root: dict) -> dict[str, float]:
    """Exact uniform-deal EV per legal action, via the canonical Python path."""
    return probe.exact_action_table(root)


def rust_action_evs(
    roots: list[dict],
    *,
    workspace_root: Path,
    fl_ev_config: Path,
) -> tuple[list[dict[str, float]], float]:
    solver = _solver_path(workspace_root)
    if not solver.is_file():
        raise FileNotFoundError(
            f"build the crate first: cargo build --release -p t4_first_exact ({solver})"
        )
    with tempfile.TemporaryDirectory() as scratch:
        in_path = Path(scratch) / "in.jsonl"
        out_path = Path(scratch) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for index, root in enumerate(roots):
                handle.write(
                    json.dumps(
                        {
                            "id": f"root-{index}",
                            "bb": {
                                "top": list(root["bb_board"][0]),
                                "middle": list(root["bb_board"][1]),
                                "bottom": list(root["bb_board"][2]),
                            },
                            "btn": {
                                "top": list(root["btn_board"][0]),
                                "middle": list(root["btn_board"][1]),
                                "bottom": list(root["btn_board"][2]),
                            },
                            "draw": list(root["draw"]),
                            "dead": list(root["bb_discards"]),
                        }
                    )
                    + "\n"
                )
        started = time.time()
        subprocess.run(
            [
                str(solver),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--fl-ev-config",
                str(fl_ev_config),
            ],
            check=True,
            capture_output=True,
        )
        elapsed = time.time() - started
        tables = []
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                tables.append(
                    {row["action_key"]: float(row["ev"]) for row in payload["actions"]}
                )
    return tables, elapsed


def run(*, roots: int, seed: int, workspace_root: Path, out_path: Path) -> dict:
    generated = [probe.sample_random_root(seed + index) for index in range(roots)]
    fl_ev_config = workspace_root / "ai" / "config" / "fl_ev.json"

    rust_tables, rust_seconds = rust_action_evs(
        generated,
        workspace_root=workspace_root,
        fl_ev_config=fl_ev_config,
    )

    python_started = time.time()
    python_tables = [python_action_evs(root) for root in generated]
    python_seconds = time.time() - python_started

    rows = []
    max_abs_delta = 0.0
    mismatched_keys = 0
    for index, root in enumerate(generated):
        py = python_tables[index]
        rs = rust_tables[index]
        key_match = set(py) == set(rs)
        if not key_match:
            mismatched_keys += 1
        deltas = [abs(py[key] - rs[key]) for key in set(py) & set(rs)]
        worst = max(deltas) if deltas else float("inf")
        max_abs_delta = max(max_abs_delta, worst)
        all_cards = (
            [card for row in root["bb_board"] for card in row]
            + [card for row in root["btn_board"] for card in row]
            + list(root["draw"])
        )
        rows.append(
            {
                "seed": root["seed"],
                "visible_joker_count": sum(
                    1 for card in all_cards if card in ("X1", "X2")
                ),
                "python_action_count": len(py),
                "rust_action_count": len(rs),
                "action_keys_match": key_match,
                "max_abs_ev_delta": worst,
            }
        )

    report = {
        "schema": SCHEMA,
        "roots": len(generated),
        "seed": seed,
        "action_key_set_mismatches": mismatched_keys,
        "max_abs_ev_delta": max_abs_delta,
        "python_seconds": python_seconds,
        "rust_seconds": rust_seconds,
        "speedup": python_seconds / rust_seconds if rust_seconds > 0 else None,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        roots=args.roots,
        seed=args.seed,
        workspace_root=args.workspace_root.resolve(strict=True),
        out_path=args.out,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "roots",
                    "action_key_set_mismatches",
                    "max_abs_ev_delta",
                    "python_seconds",
                    "rust_seconds",
                    "speedup",
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
