"""Fantasyland EV constant for the trainer, read from the repo config.

The constant is versioned on disk (`configs/fl_ev_regular_*.json`) and has moved
twice; hard-coding it here is how the trainer drifted to the June value while
the engine had already moved on.  Read the reader-default file instead, and
fall back to its value only if the file is missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("trainer.fl_ev")

_ROOT = Path(__file__).resolve().parent.parent

# configs/fl_ev_regular_v4_selfplay.json is the current reader default; the v3
# and 2k files stay on disk as historical records and must not be read here.
FL_EV_CONFIG = _ROOT / "configs" / "fl_ev_regular_v4_selfplay.json"
FL_EV_FALLBACK = 9.6


def _load() -> float:
    try:
        payload = json.loads(FL_EV_CONFIG.read_text(encoding="utf-8"))
        value = float(payload["fl_ev"]["14"])
    except Exception:
        logger.warning("FL EV config unreadable (%s); using %s", FL_EV_CONFIG, FL_EV_FALLBACK)
        return FL_EV_FALLBACK
    logger.info("FL EV(14) = %s from %s", value, FL_EV_CONFIG.name)
    return value


FL_EV_14 = _load()
