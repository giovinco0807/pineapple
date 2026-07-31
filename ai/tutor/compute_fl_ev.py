"""The M-C fixed point: Fantasyland EV under the joker-rule count carryover.

State space (hero perspective; both-normal is the zero baseline by symmetry):

    F(n)    = value of being in FL with n cards vs a normal opponent
    G(n,m)  = value of FL(n) vs FL(m); G(m,n) = -G(n,m), G(n,n) = 0

Transitions carry the entry count (commit 373ee02): a stay redeals the SAME n,
so chains compound at their own count.  From (FL n, normal):

    F(n) = e_f(n) + P(stay, no entry) F(n) + sum_m P(stay, entry m) G(n,m)
                  - sum_m P(no stay, entry m) F(m)

and from (FL n, FL m):

    G(n,m) = e_ff(n,m) + P(both stay) G(n,m) + P(hero only) F(n)
                       - P(opp only) F(m)

All probabilities are measured inputs (the solver assumes nothing about
independence: the FL-vs-FL convolution showed joint stay is measurably below
the product because the hands share a deck).  The solved F table is what the
terminal scorer reads as FL_EV, closing the loop the rules contract requires.

The measured inputs come from the normal-vs-FL chain (e_f, entry rates) and
the board libraries (stay rates, e_ff); this module only solves the system,
so it is testable now against closed forms and drives iteration later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

COUNTS = (14, 15, 16, 17)


@dataclass
class FlEvInputs:
    """Measured quantities feeding the fixed point.

    entry_given_stay / entry_given_no_stay are conditional distributions of
    the NORMAL opponent's FL entry count while the hero sits in FL(n); the
    key 0 means no entry.  Joint stay probabilities for FL-vs-FL come in as
    stay_both[(n, m)] with the marginals implied by stay[n] and stay[m].
    """

    immediate_vs_normal: dict[int, float]
    stay: dict[int, float]
    entry_given_stay: dict[int, dict[int, float]] = field(default_factory=dict)
    entry_given_no_stay: dict[int, dict[int, float]] = field(default_factory=dict)
    immediate_ff: dict[tuple[int, int], float] = field(default_factory=dict)
    stay_both: dict[tuple[int, int], float] = field(default_factory=dict)

    def entry_distribution(self, n: int, stayed: bool) -> dict[int, float]:
        table = self.entry_given_stay if stayed else self.entry_given_no_stay
        return table.get(n, {0: 1.0})

    def ff_immediate(self, n: int, m: int) -> float:
        if n == m:
            return 0.0
        if (n, m) in self.immediate_ff:
            return self.immediate_ff[(n, m)]
        if (m, n) in self.immediate_ff:
            return -self.immediate_ff[(m, n)]
        return 0.0

    def ff_stay_both(self, n: int, m: int) -> float:
        key = (n, m) if (n, m) in self.stay_both else (m, n)
        if key in self.stay_both:
            return self.stay_both[key]
        return self.stay[n] * self.stay[m]  # independence fallback, flagged


def solve(
    inputs: FlEvInputs,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> dict:
    """Value-iterate the linear system to its unique fixed point."""
    f = {n: 0.0 for n in COUNTS}
    g = {(n, m): 0.0 for n in COUNTS for m in COUNTS}

    for iteration in range(max_iterations):
        delta = 0.0

        new_g = {}
        for n in COUNTS:
            for m in COUNTS:
                if n == m:
                    new_g[(n, m)] = 0.0
                    continue
                both = inputs.ff_stay_both(n, m)
                hero_only = inputs.stay[n] - both
                opp_only = inputs.stay[m] - both
                value = (
                    inputs.ff_immediate(n, m)
                    + both * g[(n, m)]
                    + hero_only * f[n]
                    - opp_only * f[m]
                )
                new_g[(n, m)] = value

        new_f = {}
        for n in COUNTS:
            stay_n = inputs.stay[n]
            value = inputs.immediate_vs_normal[n]
            for entry, probability in inputs.entry_distribution(n, True).items():
                weight = stay_n * probability
                if entry == 0:
                    value += weight * f[n]
                else:
                    value += weight * new_g[(n, entry)]
            for entry, probability in inputs.entry_distribution(n, False).items():
                weight = (1.0 - stay_n) * probability
                if entry != 0:
                    value -= weight * f[entry]
            new_f[n] = value

        for n in COUNTS:
            delta = max(delta, abs(new_f[n] - f[n]))
        for key in new_g:
            delta = max(delta, abs(new_g[key] - g[key]))
        f, g = new_f, new_g
        if delta < tolerance:
            break
    else:
        raise RuntimeError("FL EV fixed point did not converge")

    return {
        "schema": "ofc_fl_ev_fixed_point/v1",
        "iterations": iteration + 1,
        "fl_ev": {str(n): f[n] for n in COUNTS},
        "fl_vs_fl": {f"{n}v{m}": g[(n, m)] for n in COUNTS for m in COUNTS if n < m},
        "count_carryover": True,
    }


def write_fl_ev_config(result: dict, template_path: Path, out_path: Path) -> None:
    """Emit an fl_ev.json-shaped file with the solved table, preserving shape."""
    template = json.loads(template_path.read_text(encoding="utf-8"))
    table = {key: round(value, 2) for key, value in result["fl_ev"].items()}
    template["fl_ev"] = table
    template["fl_ev_direct"] = table
    template["source"] = "fixed_point_" + result["schema"]
    out_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )
