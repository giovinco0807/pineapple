"""Aggregate played hands into a strength read.

The trainer already stores everything a result needs in each hand's summary;
this turns a pile of hands into the numbers you would actually judge a player
by. The headline is points per hand with an interval, because a running total
answers "how did today go" and not "am I better than this bot".

Every rate here uses the same denominators as the rest of the project: FL entry
is counted over normal-state hands with stays excluded, which is what the owner
ruled and what the fl_ev work measures against. Fantasyland hands are not
playable in the trainer yet, so at present every hand is a normal-state hand
and the denominator is simply the hand count -- when FL play lands, this
function is where the exclusion has to be applied rather than assumed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

# Normal approximation. OFC hand scores are bounded and roughly symmetric once
# a few hundred hands are in, so a t/z interval is honest enough here; below
# ~30 hands the interval is reported but flagged as unreliable rather than
# silently trusted.
Z95 = 1.959963984540054
MIN_HANDS_FOR_INTERVAL = 30


def _mean_ci(values: List[float]) -> Dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "ci95": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None, "ci95": None}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": [mean - Z95 * se, mean + Z95 * se],
    }


def _rate(count: int, total: int) -> float | None:
    return (count / total) if total else None


def summarise(hands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn stored hand rows into a strength read.

    `hands` is what `TrainerStore.list_hands` returns: rows carrying `position`,
    `score` and the parsed `summary` written when the hand finished.
    """
    scores: List[float] = []
    by_seat: Dict[str, List[float]] = {"first": [], "second": []}
    hero_bust = opp_bust = hero_fl = opp_fl = scoops = 0
    hero_royalty_total = opp_royalty_total = 0.0
    ev_loss_total = 0.0
    decisions = mistakes = 0

    for row in hands:
        summary = row.get("summary") or {}
        score = row.get("score")
        if score is None:
            score = summary.get("score")
        if score is None:
            continue
        score = float(score)
        scores.append(score)
        seat = row.get("position")
        if seat in by_seat:
            by_seat[seat].append(score)

        hero_bust += 1 if summary.get("hero_busted") else 0
        opp_bust += 1 if summary.get("opp_busted") else 0
        hero_fl += 1 if (summary.get("hero_fl") or {}).get("entry") else 0
        opp_fl += 1 if (summary.get("opp_fl") or {}).get("entry") else 0
        scoops += 1 if summary.get("scoop") else 0
        hero_royalty_total += float(summary.get("hero_royalty") or 0.0)
        opp_royalty_total += float(summary.get("opp_royalty") or 0.0)
        ev_loss_total += float(row.get("total_ev_loss") or 0.0)
        decisions += int(row.get("decisions") or 0)
        mistakes += int(row.get("mistakes") or 0)

    overall = _mean_ci(scores)
    n = overall["n"]
    return {
        "overall": overall,
        "reliable": n >= MIN_HANDS_FOR_INTERVAL,
        "min_hands_for_interval": MIN_HANDS_FOR_INTERVAL,
        "by_seat": {seat: _mean_ci(vals) for seat, vals in by_seat.items()},
        "total_score": sum(scores) if scores else 0.0,
        "rates": {
            "hero_bust": _rate(hero_bust, n),
            "opp_bust": _rate(opp_bust, n),
            "hero_fl_entry": _rate(hero_fl, n),
            "opp_fl_entry": _rate(opp_fl, n),
            "scoop": _rate(scoops, n),
        },
        "royalty": {
            "hero_per_hand": (hero_royalty_total / n) if n else None,
            "opp_per_hand": (opp_royalty_total / n) if n else None,
        },
        "play": {
            "decisions": decisions,
            "mistakes": mistakes,
            "ev_loss_total": ev_loss_total,
            "ev_loss_per_hand": (ev_loss_total / n) if n else None,
            "mistake_rate": _rate(mistakes, decisions),
        },
    }
