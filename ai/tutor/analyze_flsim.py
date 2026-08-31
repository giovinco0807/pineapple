"""Fantasyland EV from the pinned-hero simulator (`t4_first_exact --fl-sim`).

The measurement is the owner's frozen one (2026-08-15) and this file does not
restate it -- `analyze_selfplay_flev.py` is the canonical statement, and the
simulator reproduces its game.  What is different here is only that the hero's
entry width is chosen instead of observed, so every chain in the file is a
sample of one width and `fl_ev(w)` is a single mean rather than a bucket.

Two readings come out of the same file:

    fl_ev_all    every chain except each block's burn-in.  This is the number.
                 The opponent is wherever its own state machine left it, which
                 is what the reference chain totals also averaged over.
    fl_ev_solo   chains whose every hand faced a normal opponent.  Burn-in
                 chains count here: conditioning on "the opponent stayed normal
                 throughout" gives the same distribution whether the opponent
                 started there by force or by its own draws, because staying
                 normal is a function of the opponent's cards alone.

Nothing is corrected, blended or floored.  The structural gaps between this
world and the reference one (no stacks, no session truncation, an opponent that
is always blind) are named in `fl_sim.rs` and priced in the calibration gate;
they are not patched here.

A third reading arrives with `--ref-shares` (v2):

    fl_ev_strat  the same chains, post-stratified on the opponent's state at
                 the chain's first hand and reweighted to the reference's
                 chain-start shares.  This is the table's number.

The reason it is the number and `fl_ev_all` is not: the table is read where a
qualification is *entered*, and what the entering player faces is the chain-start
distribution, not the distribution of states a chain wanders through once it is
running.  The simulator's own start shares are the latter (its hero is pinned in
Fantasyland forever, so its opponent is more often in Fantasyland too), and the
gap is large -- five points of share at width 16, worth three and a half points
of `fl_ev`.  Stratifying makes the composition an argument instead of an
accident, and the stratum means it prints are the durable product: any future
composition can be priced from them without simulating again.

Burn-in chains count toward the strata.  Chain 0 is a chain whose opponent was
forced to start normal, which is exactly a sample of the normal stratum, by the
same argument that lets `fl_ev_solo` keep them.

The invariant checks are not optional extras.  `chain_integrity` re-derives the
opponent's state machine from the recorded entries and stays and refuses the
file if the replay disagrees, and `hero_foul_check` refuses a file in which the
hero's settlement is not consistent with a legal Fantasyland board.  Both raise
rather than report: a number computed from a file that failed them would look
exactly like a number computed from one that passed.  The empty-stratum guard is
the same kind of refusal: a stratum the reference gives weight to and this run
never visited would silently drop that weight, and a mean is not the place to
find out.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# `frontier::scoop_aware_line` maps three signs to a line score: three of a
# kind scoops to +/-6, everything else is the sum.
LINE_VALUES = frozenset({-6, -2, -1, 0, 1, 2, 6})

SCHEMA = "ofc_flsim/v1"
REPORT_SCHEMA = "ofc_flsim_report/v2"
REF_SHARES_SCHEMA = "ofc_flsim_ref_shares/v1"

# The opponent's state at a chain's first hand: the stratifying variable.  Fixed
# order so that two runs of this file over the same input agree byte for byte.
LABELS = ("normal", "fl14", "fl15", "fl16", "fl17")


def mean_se(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "se": None}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"n": n, "mean": mean, "se": math.sqrt(var / n)}


def load_ref_shares(path: Path, width: int) -> dict:
    """The reference composition for this width, from the file that derived it.

    The shares are never written into this file as constants.  They are a
    property of a particular self-play run and will be re-derived from a newer
    one; a copy here would be a second source of truth that no longer matches
    the first.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != REF_SHARES_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {REF_SHARES_SCHEMA}, found "
            f"{document.get('schema')!r}"
        )
    row = document.get("widths", {}).get(str(width))
    if row is None:
        raise ValueError(f"{path}: no reference shares for width {width}")
    shares = row["shares"]
    unknown = sorted(set(shares) - set(LABELS))
    if unknown:
        raise ValueError(f"{path}: unknown stratum label(s) {unknown}")
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{path}: width {width} shares sum to {total}, not 1")
    return {
        "path": str(path),
        "source": document.get("source"),
        "caveats": document.get("caveats", []),
        "pi": {label: float(shares.get(label, 0.0)) for label in LABELS},
        "n_ref_chains": int(row["n_chains"]),
        "row": row,
    }


def stratified(
    settles: dict[str, list[float]], reference: dict
) -> dict:
    """Post-stratified chain mean at the reference's chain-start shares.

    `se_sim` is the sampling error of the stratum means at fixed weights;
    `se_share` is the multinomial delta-method error of the weights themselves,
    which the reference's chain count caps and no amount of simulation buys
    down.  They are reported apart because only one of them can be paid for.
    """
    pi = reference["pi"]
    strata = {label: mean_se(settles.get(label, [])) for label in LABELS}
    missing = [
        label for label in LABELS if pi[label] > 0.0 and strata[label]["n"] == 0
    ]
    if missing:
        raise ValueError(
            f"empty stratum/strata {missing} carry reference weight "
            f"{[pi[label] for label in missing]}; the stratified mean would "
            f"silently drop that weight.  Run more blocks, or use a reference "
            f"whose composition this run reaches"
        )
    mean = sum(pi[label] * strata[label]["mean"] for label in LABELS if pi[label] > 0)
    # A stratum of one chain has a mean but no standard error.  Its variance
    # contribution is dropped rather than guessed, which understates se_sim, so
    # the fact is recorded next to the number.
    thin = [
        label for label in LABELS if pi[label] > 0 and strata[label]["se"] is None
    ]
    var_sim = sum(
        (pi[label] ** 2) * (strata[label]["se"] ** 2)
        for label in LABELS
        if pi[label] > 0 and strata[label]["se"] is not None
    )
    var_share = sum(
        pi[label] * (strata[label]["mean"] - mean) ** 2
        for label in LABELS
        if pi[label] > 0
    ) / reference["n_ref_chains"]
    base = strata["normal"]["mean"]
    return {
        "mean": mean,
        "se_sim": math.sqrt(var_sim),
        "se_share": math.sqrt(var_share),
        "shares_source": {
            "path": reference["path"],
            "source": reference["source"],
            "caveats": reference["caveats"],
            "n_ref_chains": reference["n_ref_chains"],
        },
        "pi": {label: pi[label] for label in LABELS},
        "strata": {label: strata[label] for label in LABELS},
        # One share point moved from normal into stratum s costs 0.01 times
        # this.  At width 15 the normal-to-fl16 spread is about 27 points, so
        # five points of share is 1.4 points of fl_ev -- larger than the
        # measurement error this file can otherwise reach.
        "sensitivity": {
            label: (
                strata[label]["mean"] - base
                if strata[label]["n"] and base is not None
                else None
            )
            for label in LABELS
        },
        "includes_burn_in": True,
        "strata_without_se": thin,
    }


def next_opp(hand: dict) -> str:
    """The opponent's state after this hand, from what the hand recorded.

    This is `self_play.rs`'s transition (524-541) rewritten over the JSON, so
    that replaying it against the next hand's `opp` is an independent check on
    the simulator rather than a restatement of it.
    """
    opp = hand["opp"]
    if opp == "normal":
        entry = hand["opp_entry"]
        return f"fl{entry}" if entry >= 14 else "normal"
    return opp if hand["opp_stay"] else "normal"


def check_hand_settlement(hand: dict, where: str) -> None:
    """Refuse any settlement a legal hero board could not have produced.

    A Fantasyland board never fouls (`play_fl` always returns `busted: false`,
    and a Fantasyland hand always has a legal arrangement), so the hero's foul
    is checkable even though no field records it:

    * against a fouled opponent the hero must collect exactly `6 + hero_roy`;
      had the hero also fouled the settlement would be 0;
    * against a live opponent the settlement minus the royalty difference must
      be one of the seven line scores; had the hero fouled it would be
      `-(6 + opp_roy)`, which leaves `-6 - hero_roy` and escapes the set for
      every non-zero royalty.
    """
    if hand["opp_foul"]:
        expected = 6 + hand["hero_roy"]
        if hand["settle"] != expected:
            raise ValueError(
                f"{where}: opponent fouled so the hero must collect "
                f"6 + hero_roy = {expected}, but the settlement is "
                f"{hand['settle']}; the hero fouled too, or the settlement is "
                f"not the hero's side of the table"
            )
        return
    lines = hand["settle"] - hand["hero_roy"] + hand["opp_roy"]
    if lines not in LINE_VALUES:
        raise ValueError(
            f"{where}: settlement {hand['settle']} with royalties "
            f"{hand['hero_roy']}/{hand['opp_roy']} implies a line score of "
            f"{lines}, which is not one of {sorted(LINE_VALUES)}; the hero "
            f"fouled, or the royalties do not belong to this settlement"
        )


def check_chain(chain: dict, expected_start: str | None, expected_h: int | None) -> None:
    """Every internal consistency the schema allows, as a refusal."""
    where = f"block {chain['block']} chain {chain['chain']}"
    hands = chain["hands"]
    if not hands:
        raise ValueError(f"{where}: a chain has at least the entry hand")
    if chain["n_hands"] != len(hands):
        raise ValueError(
            f"{where}: n_hands {chain['n_hands']} but {len(hands)} hand rows"
        )
    total = sum(hand["settle"] for hand in hands)
    if chain["sum_settle"] != total:
        raise ValueError(
            f"{where}: sum_settle {chain['sum_settle']} but the hands add to {total}"
        )
    if chain["burn_in"] != (chain["chain"] == 0):
        raise ValueError(f"{where}: burn_in must mark exactly chain 0")
    if chain["chain"] == 0 and chain["opp_at_start"] != "normal":
        raise ValueError(
            f"{where}: chain 0 forces the opponent to normal, found "
            f"{chain['opp_at_start']}"
        )
    if chain["opp_at_start"] != hands[0]["opp"]:
        raise ValueError(
            f"{where}: opp_at_start {chain['opp_at_start']} but the first hand "
            f"was played against {hands[0]['opp']}"
        )
    if expected_start is not None and chain["opp_at_start"] != expected_start:
        raise ValueError(
            f"{where}: the previous chain left the opponent at {expected_start} "
            f"but this chain starts at {chain['opp_at_start']}"
        )
    solo = all(hand["opp"] == "normal" for hand in hands)
    if chain["solo"] != solo:
        raise ValueError(f"{where}: solo is {chain['solo']} but the hands say {solo}")

    for index, hand in enumerate(hands):
        last = index == len(hands) - 1
        seat = f"{where} hand {index} (h={hand['h']})"
        # A stay re-enters at the same width, so a chain ends when and only
        # when the hero does not stay.
        if hand["hero_stay"] == last:
            raise ValueError(
                f"{seat}: hero_stay is {hand['hero_stay']} but this is "
                f"{'the last' if last else 'not the last'} hand of the chain"
            )
        if expected_h is not None and hand["h"] != expected_h:
            raise ValueError(
                f"{seat}: deals in a block are consecutive, expected h={expected_h}"
            )
        expected_h = hand["h"] + 1
        if hand["opp"] == "normal":
            if hand["opp_stay"]:
                raise ValueError(f"{seat}: a normal hand cannot stay")
        else:
            if hand["opp_entry"] != 0:
                raise ValueError(
                    f"{seat}: a Fantasyland hand does not requalify, "
                    f"opp_entry must be 0"
                )
            if hand["opp"] not in {"fl14", "fl15", "fl16", "fl17"}:
                raise ValueError(f"{seat}: unknown opponent state {hand['opp']}")
        check_hand_settlement(hand, seat)
        if not last and hands[index + 1]["opp"] != next_opp(hand):
            raise ValueError(
                f"{seat}: entry {hand['opp_entry']} / stay {hand['opp_stay']} "
                f"leads to {next_opp(hand)} but the next hand was played "
                f"against {hands[index + 1]['opp']}"
            )


def analyze(run: Path, ref_shares: Path | None = None) -> dict:
    """Read one simulator run and return its report.  Raises on a bad file.

    With `ref_shares` the report also carries `fl_ev_strat`, the chain mean
    reweighted to that file's chain-start composition, and
    `per_hand_oppfl_reweighted`, the opponent-Fantasyland hand rate the same
    reweighting implies -- the check that conditioning on the start state was
    enough to reproduce the reference's within-chain mix.
    """
    meta: dict | None = None
    chain_settles: list[float] = []
    solo_settles: list[float] = []
    # Stratification runs over every chain, burn-in included: chain 0 is a
    # normal-start chain by construction, which is what the normal stratum is.
    strat_settles: dict[str, list[float]] = {label: [] for label in LABELS}
    strat_hands: dict[str, int] = {label: 0 for label in LABELS}
    strat_fl_hands: dict[str, int] = {label: 0 for label in LABELS}
    per_hand_all: list[float] = []
    per_hand_headline: list[float] = []
    vs_normal: list[float] = []
    vs_fl: dict[str, list[float]] = defaultdict(list)
    stays_all: list[int] = []
    stays_headline: list[int] = []
    lengths: Counter[int] = Counter()
    opp_normal_hands = 0
    opp_entries: Counter[int] = Counter()
    headline_chains = 0
    headline_opp_fl_at_start = 0
    burn_in_chains = 0
    solo_chains = 0
    hands_total = 0
    hands_headline = 0
    hero_foul_checked = 0

    # Per block: where the previous chain left the opponent, and the next deal
    # index.  Chains arrive in (block, chain) order, which is itself checked.
    block_tail: dict[int, tuple[str, int]] = {}
    seen: set[tuple[int, int]] = set()
    previous_key: tuple[int, int] | None = None

    with run.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if meta is None:
                if row.get("schema") != SCHEMA:
                    raise ValueError(
                        f"{run}: first line must be the {SCHEMA} metadata, found "
                        f"schema {row.get('schema')!r}"
                    )
                if not 14 <= row.get("width", 0) <= 17:
                    raise ValueError(f"{run}: metadata width {row.get('width')!r}")
                meta = row
                continue

            key = (row["block"], row["chain"])
            if key in seen:
                raise ValueError(f"{run}:{number}: block/chain {key} appears twice")
            seen.add(key)
            if previous_key is not None and key < previous_key:
                raise ValueError(
                    f"{run}:{number}: chains must be written in (block, chain) "
                    f"order, {key} follows {previous_key}"
                )
            previous_key = key

            tail = block_tail.get(row["block"])
            expected_start = tail[0] if tail else None
            expected_h = tail[1] if tail else 0
            check_chain(row, expected_start, expected_h)

            hands = row["hands"]
            hero_foul_checked += len(hands)
            block_tail[row["block"]] = (next_opp(hands[-1]), hands[-1]["h"] + 1)

            hands_total += len(hands)
            start = row["opp_at_start"]
            if start not in strat_settles:
                raise ValueError(
                    f"{run}:{number}: opp_at_start {start!r} is not one of "
                    f"{list(LABELS)}"
                )
            strat_settles[start].append(row["sum_settle"])
            strat_hands[start] += len(hands)
            for hand in hands:
                stays_all.append(1 if hand["hero_stay"] else 0)
                per_hand_all.append(hand["settle"])
                if hand["opp"] == "normal":
                    opp_normal_hands += 1
                    if hand["opp_entry"] >= 14:
                        opp_entries[hand["opp_entry"]] += 1
                else:
                    strat_fl_hands[start] += 1

            if row["burn_in"]:
                burn_in_chains += 1
            else:
                headline_chains += 1
                hands_headline += len(hands)
                chain_settles.append(row["sum_settle"])
                lengths[row["n_hands"]] += 1
                if row["opp_at_start"] != "normal":
                    headline_opp_fl_at_start += 1
                for hand in hands:
                    stays_headline.append(1 if hand["hero_stay"] else 0)
                    per_hand_headline.append(hand["settle"])
                    if hand["opp"] == "normal":
                        vs_normal.append(hand["settle"])
                    else:
                        vs_fl[hand["opp"][2:]].append(hand["settle"])

            if row["solo"]:
                solo_chains += 1
                solo_settles.append(row["sum_settle"])

    if meta is None:
        raise ValueError(f"{run}: empty file, no metadata line")
    if headline_chains == 0:
        raise ValueError(f"{run}: no non-burn-in chains, nothing to report")

    stay_rate = sum(stays_all) / max(len(stays_all), 1)
    stay_rate_headline = sum(stays_headline) / max(len(stays_headline), 1)

    strat: dict | None = None
    reweighted: float | None = None
    if ref_shares is not None:
        reference = load_ref_shares(ref_shares, meta["width"])
        strat = stratified(strat_settles, reference)
        all_chains = len(seen)
        sim_shares = {
            label: len(strat_settles[label]) / all_chains for label in LABELS
        }
        strat["sim_shares"] = sim_shares
        # Every chain in stratum s carries the same weight, so the per-hand
        # rate can be formed from the stratum totals directly.  It is a ratio of
        # weighted totals, not a mean of per-chain rates: long chains should
        # count for the hands they contain.
        pi = reference["pi"]
        weighted_fl = weighted_hands = 0.0
        for label in LABELS:
            if pi[label] <= 0.0 or sim_shares[label] == 0.0:
                continue
            weight = pi[label] / sim_shares[label]
            weighted_fl += weight * strat_fl_hands[label]
            weighted_hands += weight * strat_hands[label]
        reweighted = weighted_fl / weighted_hands if weighted_hands else None

    def geometric(per_hand: list[float], stay: float) -> float | None:
        if not per_hand or stay >= 1.0:
            return None
        return (sum(per_hand) / len(per_hand)) / (1.0 - stay)

    report = {
        "schema": REPORT_SCHEMA,
        "run": {
            "path": str(run),
            "width": meta["width"],
            "seed": meta["seed"],
            "blocks": meta["blocks"],
            "chains_per_block": meta["chains_per_block"],
            "fl_ev": meta["fl_ev"],
            "fl_ev_config_sha256": meta["fl_ev_config_sha256"],
            "models": meta.get("models"),
            "deal": meta.get("deal"),
        },
        "counts": {
            "chains": len(seen),
            "headline_chains": headline_chains,
            "burn_in_chains": burn_in_chains,
            "solo_chains": solo_chains,
            "hands": hands_total,
            "headline_hands": hands_headline,
            "blocks_seen": len(block_tail),
        },
        # Chain totals, raw points, no baseline subtracted.  `fl_ev_all` is the
        # natural reading -- this run's own chain-start composition -- and is
        # kept for comparison; `fl_ev_strat` is the table's number when a
        # reference composition is supplied.
        "fl_ev_all": mean_se(chain_settles),
        "fl_ev_solo": mean_se(solo_settles),
        "fl_ev_strat": strat,
        "per_hand_oppfl_reweighted": reweighted,
        "per_hand": {
            "fl_vs_normal": mean_se(vs_normal),
            "fl_vs_fl": {
                width: mean_se(values) for width, values in sorted(vs_fl.items())
            },
            "all": mean_se(per_hand_headline),
        },
        "stay_rate": stay_rate,
        "stay_rate_headline": stay_rate_headline,
        "mean_chain_length": (
            sum(length * count for length, count in lengths.items())
            / max(sum(lengths.values()), 1)
        ),
        "chain_length_hist": {
            str(length): lengths[length] for length in sorted(lengths)
        },
        # The geometric form the reference report carries.  Read it as a
        # closure check, not a memorylessness test: when every chain ends in a
        # non-stay hand the count of non-stay hands *is* the count of chains,
        # so per_hand_mean / (1 - stay_rate) collapses to total / chains
        # exactly.  It agrees to the last digit in the reference file for the
        # same reason.  What it would catch is a truncated or unclosed chain --
        # which `chain_integrity` already refuses -- so its value here is that
        # `geometric_check_headline` must equal `fl_ev_all` bit for bit, and
        # `geometric_check` must equal the mean over all chains including the
        # burn-ins -- in both cases to floating-point rounding, which is why the
        # reference file's own pair differs in the last digit.  A gap between
        # the two beyond that is the burn-in chains, not a defect.
        "geometric_check": geometric(per_hand_all, stay_rate),
        "geometric_check_headline": geometric(per_hand_headline, stay_rate_headline),
        "opp_fl_at_start_rate": headline_opp_fl_at_start / headline_chains,
        "opp_entry_rate": sum(opp_entries.values()) / max(opp_normal_hands, 1),
        "opp_entry_by_width": {
            str(width): opp_entries[width] / max(opp_normal_hands, 1)
            for width in sorted(opp_entries)
        },
        "opp_normal_hands": opp_normal_hands,
        "hero_foul_check": {
            "hands_checked": hero_foul_checked,
            "ok": True,
            "note": (
                "every hand's settlement is consistent with a hero board that "
                "did not foul; a violation raises rather than reporting"
            ),
        },
        "chain_integrity": {
            "chains_checked": len(seen),
            "ok": True,
            "note": (
                "hero_stay false on the last hand of a chain and true on every "
                "other; the opponent's replayed state machine agrees with the "
                "recorded entries and stays across hands and across chain "
                "boundaries within a block; deals are consecutive per block"
            ),
        },
        "notes": {
            "geometric_check": (
                "over every hand including burn-in chains, matching "
                "analyze_selfplay_flev, so it equals the mean over all chains "
                "burn-ins included; geometric_check_headline restricts to the "
                "chains fl_ev_all is measured on and therefore reproduces "
                "fl_ev_all to floating-point rounding.  Both are closure "
                "checks: every chain here ends in a non-stay hand, which is "
                "what makes the identity hold"
            ),
            "per_hand": "non-burn-in chains only, so it pairs with fl_ev_all",
            "fl_ev_strat": (
                "every chain including the burn-ins, post-stratified on "
                "opp_at_start and reweighted to the reference's chain-start "
                "shares.  se_sim is buyable with blocks; se_share is capped by "
                "the reference's chain count and is not.  geometric_check stays "
                "on the natural reading, which is the one it closes over"
            ),
            "per_hand_oppfl_reweighted": (
                "the opponent-Fantasyland share of hero hands after the same "
                "reweighting.  Agreement with the reference's per-hand rate is "
                "the evidence that conditioning on the start state was "
                "sufficient -- that the opponent's process *inside* a chain is "
                "the same process in both worlds"
            ),
            "raw_vs_paid": (
                "raw settlements only: this world has no stacks, so there is no "
                "paid reading to give.  In the reference data paid minus raw is "
                "+0.02/-0.02/-0.30/-2.45 at widths 14/15/16/17, and a zero-stack "
                "session end truncates a further +0.005/+0.04/+0.64/+7.98 that "
                "raw does not show at all"
            ),
        },
    }
    # Without a reference composition the report is v1's, key for key: the
    # stratified reading is not a null, it is a question that was not asked.
    if strat is None:
        del report["fl_ev_strat"]
        del report["per_hand_oppfl_reweighted"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--ref-shares", type=Path,
        help="ofc_flsim_ref_shares/v1 file (fl_sim_driver.py "
             "--extract-ref-shares writes it).  Without it the report omits "
             "fl_ev_strat and per_hand_oppfl_reweighted and is otherwise "
             "unchanged",
    )
    args = parser.parse_args()

    report = analyze(args.run, args.ref_shares)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
