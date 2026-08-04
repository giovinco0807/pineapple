"""Batched sampled joint-outlook blocks from the Rust solver.

Shared by the regret gates and any encoder that needs the 8-dim block for
boards with more than two open slots.  Sampling is deterministic per request
id, so a gate's features are reproducible.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from ai.tutor.solver_paths import _solver_path


def fetch_joint_blocks(
    requests: list[dict],
    workspace_root: Path,
    samples: int = 150,
    max_arrangements: int = 32,
) -> dict[str, list[float]]:
    """requests: [{id, board:{top,middle,bottom}, pool:[cards]}] -> id -> block."""
    solver = str(_solver_path(workspace_root)).replace(
        "t4_first_exact.exe", "t4_first_exact_jo.exe"
    )
    if not Path(solver).exists():
        solver = str(_solver_path(workspace_root))
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.jsonl"
        out_path = Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                payload = dict(request)
                payload["samples"] = samples
                payload["max_arrangements"] = max_arrangements
                handle.write(json.dumps(payload) + "\n")
        subprocess.run(
            [
                solver,
                "--input", str(in_path),
                "--output", str(out_path),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--joint-outlook", "--chunk-size", "256",
            ],
            check=True,
        )
        blocks: dict[str, list[float]] = {}
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    blocks[payload["id"]] = payload["joint_block"]
    return blocks
