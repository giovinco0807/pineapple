"""The canonical Fantasyland EV table, read straight from the config.

Reproduces `RolloutEvaluator._load_fl_ev` (verified equal) without importing
it.  That module pulls in the whole MCTS stack and, through it, torch -- an
800 MB dependency for four constants, which every fleet label worker would
otherwise have to install before it could generate a single row.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "fl_ev.json"


def load_fl_ev(path: Path = CONFIG_PATH) -> dict[int, float]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("reward_mode") == "direct" and "fl_ev_direct" in config:
        return {int(key): value for key, value in config["fl_ev_direct"].items()}
    opponent = config["opponent_avg_royalty"]
    return {
        int(cards): (stats["R"] - opponent) / (1 - stats["stay_rate"])
        for cards, stats in config["fl_stats"].items()
    }


FL_EV = load_fl_ev()
