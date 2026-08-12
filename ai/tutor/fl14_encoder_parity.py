"""Python/Rust parity for the 104-dim FL14 teacher vector.

`ai/tutor/encode_fl14_teacher.py` is the truth: what it writes into the
teacher arrays is what the FL14 evaluators were fitted on, so the playout's
`encode_for` has to reproduce it dim for dim.  A width that merely *loads* is
the dangerous case -- the net answers confidently about a position it was
never shown -- which is why this compares numbers rather than shapes.

Only two of the four blocks cross the language boundary.  The rowwise 41 and
the joint 8 come from the same Rust binary on both sides, since the Python
encoder shells out to `--joint-outlook` for them; what those columns test is
that `encode_for`'s memoized rowwise path and its sample-count rule reproduce
what that mode returns.  The actor 48 and the deck 7 are computed
independently in each language, and that is where drift would live -- a
disagreement in the actor block would mean the Python and Rust encoders had
already parted company, which reaches far past the FL14 teachers.

Boards come from the shipped label files rather than a generator, and are
parsed with the encoder's own `JokerNamer`, so the positions compared are
literally rows the teacher encoded.

Usage:
    python -m ai.tutor.fl14_encoder_parity \
        --model D:/ofc_data/fl14_t2_model_v1/evaluator.bin \
        --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ai.engine.encoding import ALL_CARDS
from ai.tutor.encode_fl14_teacher import (
    ACTOR_SIZE,
    CONTEXT_SIZE,
    FEATURE_SIZE,
    JOINT_SIZE,
    ROWWISE_SIZE,
    JokerNamer,
    context_block,
    fetch_blocks,
)
from ai.tutor.solver_paths import _solver_path
from ai.tutor.t3_second_features import actor_block

SCHEMA = "ofc_fl14_encoder_parity/v1"

BLOCKS = (
    ("actor", 0, ACTOR_SIZE),
    ("rowwise", ACTOR_SIZE, ACTOR_SIZE + ROWWISE_SIZE),
    ("joint", ACTOR_SIZE + ROWWISE_SIZE, ACTOR_SIZE + ROWWISE_SIZE + JOINT_SIZE),
    ("context", FEATURE_SIZE - CONTEXT_SIZE, FEATURE_SIZE),
)

# The rule `encode_for` uses to recover `--joint-samples` from the board, and
# the value the FL14 teachers passed at each street: exact enumeration once a
# placement leaves two open slots, 400 sampled completions above that.
JOINT_SAMPLES_ABOVE_EXACT = 400


def joint_samples_for(rows: list[list[str]]) -> int:
    open_slots = sum(capacity - len(row) for capacity, row in zip((3, 5, 5), rows))
    return 0 if open_slots <= 2 else JOINT_SAMPLES_ABOVE_EXACT


def pool_for(rows: list[list[str]], dead: list[str]) -> list[str]:
    seen = {card for row in rows for card in row} | set(dead)
    if len(seen) != sum(len(row) for row in rows) + len(dead):
        raise AssertionError(f"a card repeats in {rows} + {dead}")
    return [card for card in ALL_CARDS if card not in seen]


def cases_from_labels(path: Path, street: str, scan: int) -> list[dict]:
    """Every (width, board joker count) combination this file can supply.

    A T2 record carries both widths this street sees: `board` is the 7-card
    position the root acts from, and each `action_key` is the 9-card board one
    action reaches.  A T3 record's actions are 11 cards.
    """
    found: dict[tuple[int, int], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= scan or not line.strip():
                break
            record = json.loads(line)
            root = int(record["root"])
            variants = []

            namer = JokerNamer()
            rows_before = namer.rows(record["board"])
            dead_before = namer(record["dead"]) + namer(record["draw"])
            variants.append((rows_before, dead_before))

            for action in record["actions"][:1]:
                namer = JokerNamer()
                rows_text, discard = action["action_key"].rsplit("|", 1)
                rows_after = namer.rows(rows_text)
                variants.append((rows_after, namer(discard) + namer(record["dead"])))

            for rows, dead in variants:
                width = sum(len(row) for row in rows)
                jokers = sum(
                    1 for row in rows for card in row if card in ("X1", "X2")
                )
                key = (width, jokers)
                if key in found:
                    continue
                found[key] = {
                    "id": f"{street}-{width}c-{jokers}j-root{root}",
                    "rows": rows,
                    "dead": dead,
                    "pool": pool_for(rows, dead),
                    # The teacher shares one completion sample across a root's
                    # actions, and its seed is the root; reusing that shape
                    # keeps this comparing what the trainer saw.
                    "seed": f"root/{root}",
                    "width": width,
                    "jokers": jokers,
                }
    return [found[key] for key in sorted(found)]


def python_vectors(cases: list[dict], workspace_root: Path, solver: str) -> list[list[float]]:
    """The encoder's own assembly, block for block, on these boards."""
    blocks: dict[str, tuple] = {}
    by_samples: dict[int, list[dict]] = {}
    for case in cases:
        by_samples.setdefault(joint_samples_for(case["rows"]), []).append(case)
    for samples, group in by_samples.items():
        requests = [
            {
                "id": case["id"],
                "seed": case["seed"],
                "board": {
                    "top": case["rows"][0],
                    "middle": case["rows"][1],
                    "bottom": case["rows"][2],
                },
                "pool": case["pool"],
            }
            for case in group
        ]
        blocks.update(fetch_blocks(requests, workspace_root, samples, solver))

    vectors = []
    for case in cases:
        rowwise, joint = blocks[case["id"]]
        actor, _categories = actor_block(case["rows"], case["pool"])
        vector = (
            actor
            + [float(value) for value in rowwise]
            + [float(value) for value in joint]
            + context_block(case["pool"])
        )
        if len(vector) != FEATURE_SIZE:
            raise AssertionError(f"feature size drifted: {len(vector)}")
        vectors.append(vector)
    return vectors


def rust_vectors(
    cases: list[dict], model: Path, fl_ev_config: Path, solver: str
) -> list[list[float]]:
    """`playout::encode_for` on the same boards, through --encode-features."""
    with tempfile.TemporaryDirectory() as scratch:
        in_path = Path(scratch) / "in.jsonl"
        out_path = Path(scratch) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(
                    json.dumps(
                        {
                            "id": case["id"],
                            "seed": case["seed"],
                            "board": {
                                "top": case["rows"][0],
                                "middle": case["rows"][1],
                                "bottom": case["rows"][2],
                            },
                            "pool": case["pool"],
                        }
                    )
                    + "\n"
                )
        subprocess.run(
            [
                solver,
                "--encode-features",
                "--t1-t2-model", str(model),
                "--fl-ev-config", str(fl_ev_config),
                "--input", str(in_path),
                "--output", str(out_path),
            ],
            check=True,
        )
        emitted = {}
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    emitted[payload["id"]] = payload["features"]
    return [emitted[case["id"]] for case in cases]


def run(
    *,
    t2_labels: Path,
    t3_labels: Path,
    scan: int,
    model: Path,
    workspace_root: Path,
    solver: str,
    tolerance: float,
) -> dict:
    cases = cases_from_labels(t2_labels, "t2", scan) + cases_from_labels(
        t3_labels, "t3", scan
    )
    cases = [case for case in cases if case["width"] in (7, 9, 11)]
    if not cases:
        raise SystemExit("no boards collected")

    fl_ev_config = workspace_root / "ai" / "config" / "fl_ev.json"
    py = np.asarray(python_vectors(cases, workspace_root, solver), dtype=np.float32)
    rs = np.asarray(
        rust_vectors(cases, model, fl_ev_config, solver), dtype=np.float32
    )
    if py.shape != rs.shape:
        raise AssertionError(f"shape mismatch: python {py.shape} rust {rs.shape}")

    delta = np.abs(py - rs)
    per_block = {
        name: float(delta[:, start:stop].max()) for name, start, stop in BLOCKS
    }
    rows = [
        {
            "id": case["id"],
            "width": case["width"],
            "board_jokers": case["jokers"],
            "pool_jokers": sum(1 for card in case["pool"] if card in ("X1", "X2")),
            "pool_size": len(case["pool"]),
            "joint_samples": joint_samples_for(case["rows"]),
            "max_abs_delta": float(delta[index].max()),
            "worst_dim": int(delta[index].argmax()),
        }
        for index, case in enumerate(cases)
    ]
    return {
        "schema": SCHEMA,
        "boards": len(cases),
        "feature_size": FEATURE_SIZE,
        "tolerance": tolerance,
        "max_abs_delta": float(delta.max()),
        "max_abs_delta_by_block": per_block,
        "actor_block_agrees": per_block["actor"] <= tolerance,
        "passed": float(delta.max()) <= tolerance,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t2-labels",
        type=Path,
        default=Path("D:/ofc_data/fl14_t2_teacher_v1/t2_labels.jsonl"),
    )
    parser.add_argument(
        "--t3-labels",
        type=Path,
        default=Path("D:/ofc_data/fl14_teacher_v1/t3_labels.jsonl"),
    )
    parser.add_argument("--scan", type=int, default=400, help="label lines read")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("D:/ofc_data/fl14_t2_model_v1/evaluator.bin"),
        help="a 104-dim evaluator image; --encode-features reads its width",
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--solver", default=None, help="override the solver binary")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve(strict=True)
    solver = args.solver or str(_solver_path(workspace_root))
    if not Path(solver).exists():
        raise SystemExit(f"no solver binary at {solver}")
    report = run(
        t2_labels=args.t2_labels,
        t3_labels=args.t3_labels,
        scan=args.scan,
        model=args.model,
        workspace_root=workspace_root,
        solver=solver,
        tolerance=args.tolerance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "boards",
                    "max_abs_delta",
                    "max_abs_delta_by_block",
                    "actor_block_agrees",
                    "passed",
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    for row in report["rows"]:
        print(
            f"  {row['id']:<28} width={row['width']:>2} "
            f"board_jokers={row['board_jokers']} pool={row['pool_size']} "
            f"samples={row['joint_samples']:>3} "
            f"max|delta|={row['max_abs_delta']:.3e} @dim {row['worst_dim']}"
        )
    if not report["actor_block_agrees"]:
        print(
            "ACTOR BLOCK MISMATCH: the Python and Rust actor encoders have "
            "drifted apart, which affects every width that uses them, not "
            "just 104.",
            file=sys.stderr,
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
