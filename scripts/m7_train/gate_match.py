"""Mirrored duplicate match between two pinned chains.

A held-out number says how well a model copies its teacher. It does not say
whether the model PLAYS better -- M7's own T3 street improved its held-out
regret by 13.3% and moved head-to-head EV by nothing measurable. So a model is
promoted on a match, not on a metric.

Duplicate, because OFC variance dwarfs the effect being measured: every deal is
played TWICE from the same deck, with the chains swapping seats. Luck cancels
and only the decisions remain.

    game 1:  candidate = seat first,  incumbent = seat second
    game 2:  incumbent = seat first,  candidate = seat second
    pair     = (candidate's points in g1) + (candidate's points in g2)
             = s(g1) - s(g2)          because the game is zero sum

The self-test falls out of that algebra: if the two chains choose identically
everywhere, the two games ARE the same game and the pair is exactly 0.0. Any
nonzero pair on a deal where no decision differed is a harness bug, not an edge,
and the harness refuses to report a verdict when it sees one.

For that to hold, a decision must be a pure function of (observation, chain).
The engine's sampling seed is therefore derived from the observation
fingerprint rather than from a counter -- the same information set reached in
either game draws the same particles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time

REPO = ("/mnt/c/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/"
        "regular-ofc-pineapple")
sys.path.insert(0, REPO + "/src")

STREETS = (("T0", 5), ("T1", 3), ("T2", 3), ("T3", 3), ("T4", 3))

# Every learned weight the engine needs, by the JointExactConfig field stem.
WEIGHT_STEMS = (
    "t4", "t3_second", "t3_first", "t2_second", "t2_first",
    "t1_second", "t1_first", "t0_second", "t0_first",
)
FAST_STEMS = ("t0_second", "t1_second", "t1_first", "t2_second", "t2_first")

RUNTIME_FILES = {
    "t4": "weights/t4_model_v6.bin",
    "t3_second": "weights/t3_model_v3.bin",
    "t3_first": "weights/t3first_model_v2.bin",
    "t2_second": "weights/t2_model_v1.bin",
    "t2_first": "weights/t2first_model_v1.bin",
    "t1_second": "weights/t1_model_v1.bin",
    "t1_first": "weights/t1first_model_v1.bin",
    "t0_second": "weights/t0_model_v1.bin",
    "t0_first": "weights/t0first_model_v1.bin",
}
FAST_FILES = {
    "t0_second": "weights/fast_t0_second_v1.bin",
    "t1_second": "weights/fast_t1_second_v1.bin",
    "t1_first": "weights/fast_t1_first_v1.bin",
    "t2_second": "weights/fast_t2_second_v1.bin",
    "t2_first": "weights/fast_t2_first_v1.bin",
}

# Sampling budget per street. `decide` runs each street's learned evaluator
# once rather than searching, so these only shape whatever downstream sampling
# a street's evaluator still does. T4 decides by exact enumeration and ignores
# them entirely.
SAMPLES = {
    "T0": (1, 1, 1),
    "T1": (1, 1, 1),
    "T2": (1, 1, 1),
    "T3": (1, 1, 1),
    # T4 decides by exact enumeration and reads none of these, but
    # JointExactConfig still validates them as positive.
    "T4": (1, 1, 1),
}


def sha256_of(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def build_chain(runtime: pathlib.Path, override: dict[str, str] | None):
    """A chain is the full pin set, optionally with one weight replaced."""
    override = override or {}
    weights = {}
    for stem, rel in RUNTIME_FILES.items():
        path = pathlib.Path(override.get(stem, runtime / rel)).resolve()
        if not path.is_file():
            raise SystemExit(f"missing weight {stem}: {path}")
        weights[stem] = (str(path), sha256_of(path))
    for stem, rel in FAST_FILES.items():
        path = (runtime / rel).resolve()
        weights["fast_" + stem] = (str(path), sha256_of(path))
    return weights


def make_config_factory(weights):
    from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig

    def factory(observation):
        cand, evals, ds3 = SAMPLES[observation.street]
        seed = int(observation.fingerprint()[:16], 16) & ((1 << 63) - 1)
        kwargs = {}
        for stem in WEIGHT_STEMS:
            path, digest = weights[stem]
            kwargs[f"learned_{stem}_model_path"] = path
            kwargs[f"learned_{stem}_model_sha256"] = digest
        for stem in FAST_STEMS:
            path, digest = weights["fast_" + stem]
            kwargs[f"fast_{stem}_model_path"] = path
            kwargs[f"fast_{stem}_model_sha256"] = digest
        return JointExactConfig(
            candidate_samples=cand, evaluation_samples=evals,
            downstream_t3_samples=ds3, downstream_t4_samples=0,
            seed=seed, candidate_seed=seed, evaluation_seed=seed,
            run_id=f"gate:{observation.fingerprint()[:16]}",
            seat=observation.seat, to_act_order=observation.to_act_order,
            **kwargs,
        )

    return factory


def decide(observation, config_factory, library):
    """The chain's move, through the engine's own `decide` request.

    This is the production decision path -- the same request the webapp runtime
    issues -- and it is what "pure engine at all ten slots" means. It returns
    the chosen placement, not a ranked list, because it runs the street's
    learned evaluator once instead of scoring every candidate through a search.

    The difference is not a detail: scoring all 232 T0 candidates the analysis
    way costs about 20 seconds a decision, which is 94% of a game and makes a
    20,000-deal match impossible. `decide` is what the earlier gates measured
    3,241 games/h with.
    """
    from ofc_regular.hu_m3_rust import (
        HU_M3_REQUEST_SCHEMA, _joint_config_payload, evaluate_request,
    )

    config = config_factory(observation)
    response = evaluate_request(
        {
            "schema": HU_M3_REQUEST_SCHEMA,
            "kind": "decide",
            "observation": observation.to_dict(),
            "observation_fingerprint": observation.fingerprint(),
            "config": _joint_config_payload(config),
        },
        library=library,
    )
    if "placements" not in response:
        raise SystemExit(
            f"engine refused a {observation.street} decision: {response!r}")
    placements = tuple(
        (str(row[0]), str(row[1])) if isinstance(row, (list, tuple))
        else (str(row["row"]), str(row["card"]))
        for row in response["placements"]
    )
    discards = tuple(str(card) for card in response.get("discards", ()))
    # The decision identity for divergence detection: what was placed where,
    # and what was thrown. Sorted so an ordering difference inside one
    # placement list cannot masquerade as a different decision.
    identity = (tuple(sorted(placements)), tuple(sorted(discards)))
    return placements, discards, identity


def play_hand(hand_seed, factories, library, fl_ev):
    """One hand. factories[0] plays the first seat, factories[1] the second."""
    from ofc_regular.cards import create_deck
    from ofc_regular.hu_infoset import ActorObservation
    from ofc_regular.state import Board
    from ofc_regular.teacher import terminal_score

    deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
    boards = [Board.from_rows(), Board.from_rows()]
    discards: list[list[str]] = [[], []]
    keys: list[str] = []
    cursor = 0
    for street, count in STREETS:
        for player in (0, 1):
            dealt = tuple(deck[cursor:cursor + count])
            cursor += count
            if len(dealt) != count:
                raise RuntimeError("deck exhausted")
            seat = "first" if player == 0 else "second"
            observation = ActorObservation(
                hero_board=boards[player],
                opponent_public_board=boards[1 - player],
                dealt_cards=dealt,
                hero_private_discards=tuple(discards[player]),
                seat=seat, street=street, to_act_order=seat)
            placements, thrown, identity = decide(
                observation, factories[player], library)
            boards[player] = boards[player].place(placements)
            discards[player].extend(thrown)
            keys.append(identity)
    score_first, _ = terminal_score(boards[0], boards[1], fl_ev)
    return float(score_first), tuple(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="/home/wner/ofc-labelgen-m7v5/build/runtime")
    ap.add_argument("--swap-stem", default=None,
                    help="which weight the candidate chain replaces, e.g. t2_second")
    ap.add_argument("--swap-path", default=None)
    ap.add_argument("--deals", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--self-play", action="store_true",
                    help="candidate vs ITSELF: every pair must be exactly 0.0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ofc_regular import hu_m3_rust
    from ofc_regular.hu_infoset import load_default_fl_ev

    runtime = pathlib.Path(a.runtime).resolve()
    library = hu_m3_rust.load_native_engine(
        path=str(runtime / "native/libofc_hu_m3_engine.so"), build_if_missing=False)
    fl_ev = load_default_fl_ev()

    override = ({a.swap_stem: a.swap_path}
                if a.swap_stem and a.swap_path else None)
    candidate = build_chain(runtime, override)
    incumbent = build_chain(runtime, None if not a.self_play else override)
    if a.self_play:
        incumbent = candidate

    cand_factory = make_config_factory(candidate)
    inc_factory = make_config_factory(incumbent)

    pairs, divergent = [], []
    began = time.time()
    for index in range(a.start, a.start + a.deals):
        hand_seed = a.seed_base + index
        s1, k1 = play_hand(hand_seed, (cand_factory, inc_factory), library, fl_ev)
        s2, k2 = play_hand(hand_seed, (inc_factory, cand_factory), library, fl_ev)
        pairs.append(s1 - s2)
        divergent.append(k1 != k2)

    seconds = time.time() - began
    nonzero_on_identical = [
        p for p, d in zip(pairs, divergent) if not d and p != 0.0]
    mean = statistics.mean(pairs)
    stderr = (statistics.stdev(pairs) / math.sqrt(len(pairs))
              if len(pairs) > 1 else 0.0)
    div = [p for p, d in zip(pairs, divergent) if d]
    record = {
        "schema": "ofc_gate_match_shard_v1",
        "deals": len(pairs), "games": 2 * len(pairs),
        "start": a.start, "seed_base": a.seed_base,
        "self_play": a.self_play,
        "swap_stem": a.swap_stem, "swap_path": a.swap_path,
        "divergence_rate": sum(divergent) / len(divergent),
        "paired_mean": mean, "paired_stderr": stderr,
        "ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr],
        "paired_mean_on_divergent": statistics.mean(div) if div else 0.0,
        "nonzero_pairs": sum(1 for p in pairs if p != 0.0),
        "max_abs_pair": max(abs(p) for p in pairs) if pairs else 0.0,
        "nonzero_on_identical": len(nonzero_on_identical),
        "max_abs_on_identical": max((abs(p) for p in nonzero_on_identical),
                                    default=0.0),
        "seconds": seconds,
        "games_per_hour": 2 * len(pairs) / seconds * 3600.0 if seconds else 0.0,
        "pairs": pairs, "divergent": divergent,
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    print(f"deals {len(pairs)}  divergence {record['divergence_rate']:.3f}  "
          f"mean {mean:+.4f}  ci95 [{record['ci95'][0]:+.4f}, "
          f"{record['ci95'][1]:+.4f}]  {record['games_per_hour']:.0f} games/h")
    if nonzero_on_identical:
        print(f"HARNESS BUG: {len(nonzero_on_identical)} deals had NO decision "
              f"difference but a nonzero pair (max "
              f"{record['max_abs_on_identical']:.6f}) -- the mirror does not "
              f"cancel and no verdict may be read from this run")
        return 1
    if a.self_play and record["nonzero_pairs"]:
        print(f"SELF-TEST FAILED: {record['nonzero_pairs']} nonzero pairs "
              f"playing a chain against itself")
        return 1
    print("mirror clean: every identical-decision deal cancelled to exactly 0.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
