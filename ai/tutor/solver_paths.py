"""Where the Rust solver binaries live.

Its own module so label generators can find the binary without importing
`generate_t4_first_teacher`, whose import chain reaches the MCTS stack and
through it torch -- 800 MB that fleet workers would have to install to
compute one path string.
"""
from __future__ import annotations

import sys
from pathlib import Path


def solver_path(workspace_root: Path, name: str = "t4_first_exact") -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return workspace_root / "ai" / "rust_solver" / "target" / "release" / f"{name}{suffix}"


# The generators were written against this name.
_solver_path = solver_path


def count_library_args() -> list[str]:
    """Per-count FL library flags, gated behind JOKER_COUNT_LIBS=1.

    The gate exists because flipping this changes labels for every
    non-14-count root (the referee starts using the true 15/16/17 board
    distributions), and in-flight runs against the older binary would
    reject the flags.  Pass-2 relabeling turns it on.
    """
    import os

    if os.environ.get("JOKER_COUNT_LIBS") != "1":
        return []
    root = os.environ.get("JOKER_LIBS_ROOT", "D:/ofc_data")
    out: list[str] = []
    for count in (15, 16, 17):
        out += [f"--fl-library-{count}", f"{root}/fl_library_{count}_v3"]
    return out
