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
