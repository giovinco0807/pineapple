"""T0-BTN policy net: hero's five plus the opponent's placed street-0 board
-> a distribution over all 232 openings.

The Button analogue of `train_t0_policy.py` (read that first: the two-stage
recipe, the action index, the v1 trap).  Everything that differs is in the
decision state:

  * the state is (hero five, opponent board).  Suit symmetry is now a JOINT
    property -- a global suit permutation applied to both sides leaves the
    value unchanged, one applied to hero alone does not -- so the four suits
    are canonicalised from the union of both sides' natural cards by a rule
    that is a function of the card SETS, never of the order anything arrived
    in (`canonical_btn`; the selftest shuffles inputs to prove it);
  * jokers are numbered across the whole deal (hero may hold X2 while the
    opponent holds X1) and carry no identity.  Hero jokers are renumbered
    X1..Xk by sorted original name and become positions; opponent jokers are
    only counted per row;
  * features are 213 wide: hero 54 (exactly as BB) + 3 x 52 opponent rows +
    3 per-row joker counts.  No standardisation.

The action index and the legal mask are IMPORTED from the BB script -- one
decode for both seats -- so a Rust transcription that already agrees with BB
on the decode only has to agree with this file on the encode.

Data spelling (matters for every loader here): the label files, the referee
material and the pairs_meta anchor all spell hero jokers by DEAL POSITION
(Rust `normalise_hero`): a hero holding only X2 is written X1 in every
action key.  The loaders apply that rename (`rename_hero_jokers`) before
decoding; `canonical_btn`'s own renumbering is then the identity on hero
names, and only the opponent side ever shows an X2 without an X1.

Stage B substitution (vs BB): the dense landscape anchor is the SERVED
EVALUATOR's score for all 232 openings of each referee root
(`t0_btn_evalfix/pairs_meta.jsonl`, `own_score`), not corrected-chain
values -- the Button material's roots are not t0r1 roots, so no chain label
exists for them.  The anchor's job is unchanged (pin the ~214 unrefereed
logits so the referee term cannot float a hallucinated argmax to the top),
but its values come from the very instrument whose picks the referee
corrects, so the anchor is exactly as biased as serving and the referee term
is the only source of correction.

    python -m ai.tutor.train_t0_btn_policy --selftest
    python -m ai.tutor.train_t0_btn_policy --torch-seed 1 --out D:/ofc_data/hu/t0_btn_policy_s1
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.tutor.train_t0_policy import MASK, RANKS, RV, SUITS, action_index, legal_mask

D = Path("D:/ofc_data/hu")
FEATURE_SIZE = 213
ACTION_SIZE = 243
ROW_CAPACITY = (3, 5, 5)
HERO_JOKER_BASE = 52          # hero joker i -> slot 52 + i
OPP_BASE = 54                 # opponent row r natural -> 54 + 52*r + slot
OPP_JOKER_BASE = 210          # opponent row r joker count -> 210 + r

STAGE_A = (
    ("lap1", "t0r1_labels/t0_btn_l1.jsonl", "onpol_requests/t0_btn_onpol.jsonl"),
    ("lap2", "t0r1_labels/t0_btn_l2.jsonl", "onpol2_requests/t0_btn_lap2.jsonl"),
)
MATERIAL = "t0btn_mine/material_btnmine3.jsonl"
PAIRS_META = "t0_btn_evalfix/pairs_meta.jsonl"

__all__ = ["canonical_btn", "feats_btn", "action_index", "legal_mask", "MASK",
           "rename_hero_jokers", "parse_board_spec", "board_spec", "openings",
           "encode_state", "decode_key", "PolicyBtn"]


def is_joker(card: str) -> bool:
    return card.startswith("X")


def rename_hero_jokers(cards):
    """Rust `normalise_hero`: hero jokers renamed X1..Xk in deal order.

    The data spells hero jokers this way (see the module docstring), and so
    do the keys the Rust deep path writes; every loader and the parity
    harness call this before decoding anything.
    """
    n = 0
    out = []
    for c in cards:
        if is_joker(c):
            n += 1
            out.append(f"X{n}")
        else:
            out.append(c)
    return out


def parse_board_spec(spec: str):
    """'top|mid|bot' with comma-separated cards -> three lists (empty rows ok)."""
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"board spec wants three rows, got {spec!r}")
    return [[c for c in part.split(",") if c] for part in parts]


def board_spec(rows) -> str:
    return "|".join(",".join(row) for row in rows)


def suit_keys(hero, opp_rows):
    """Per suit: (hero_mask, opp_top_mask, opp_mid_mask, opp_bot_mask), 13-bit
    ints with rank 2 = bit 0 ... A = bit 12, from the natural cards only."""
    key = {s: [0, 0, 0, 0] for s in SUITS}
    for c in hero:
        if not is_joker(c):
            key[c[1]][0] |= 1 << (RV[c[0]] - 2)
    for r, row in enumerate(opp_rows):
        for c in row:
            if not is_joker(c):
                key[c[1]][r + 1] |= 1 << (RV[c[0]] - 2)
    return {s: tuple(v) for s, v in key.items()}


def canonical_btn(hero, opp_rows):
    """Joint suit canonicalisation of (hero five, opponent board).

    Returns (canon, mapping, opp_canon, opp_jokers):
      canon      hero's canonical names in POSITION ORDER for the action
                 index: naturals (rank desc, canonical suit index asc), then
                 hero jokers X1..Xk (renumbered by sorted original name);
      mapping    original hero name -> canonical name;
      opp_canon  three lists of the opponent's canonical natural names;
      opp_jokers three ints, the opponent's joker count per row.

    Suits sort by key DESCENDING (tuple comparison); a residual tie -- all
    four masks equal, possible only when the suit is absent from both sides
    -- falls to the original suit's index in "cdhs" ASCENDING.  The result
    depends on the card sets alone.
    """
    if len(hero) != 5:
        raise ValueError(f"hero wants five cards, got {hero}")
    if len(opp_rows) != 3 or sum(len(r) for r in opp_rows) != 5:
        raise ValueError(f"opponent board wants three rows holding five cards, got {opp_rows}")
    for r, row in enumerate(opp_rows):
        if len(row) > ROW_CAPACITY[r]:
            raise ValueError(f"opponent row {r} holds {len(row)} of {ROW_CAPACITY[r]}")
    keys = suit_keys(hero, opp_rows)
    # Negating each component turns lexicographic ascending into lexicographic
    # descending exactly; the index is the residual tie-break.
    order = sorted(SUITS, key=lambda s: (tuple(-m for m in keys[s]), SUITS.index(s)))
    relabel = {s: SUITS[i] for i, s in enumerate(order)}
    plain = [c for c in hero if not is_joker(c)]
    jokers = sorted(c for c in hero if is_joker(c))
    canon = sorted((c[0] + relabel[c[1]] for c in plain),
                   key=lambda c: (-RV[c[0]], SUITS.index(c[1])))
    canon += [f"X{i + 1}" for i in range(len(jokers))]
    mapping = {c: c[0] + relabel[c[1]] for c in plain}
    for i, j in enumerate(jokers):
        mapping[j] = f"X{i + 1}"
    opp_canon = [sorted(c[0] + relabel[c[1]] for c in row if not is_joker(c)) for row in opp_rows]
    opp_jokers = [sum(1 for c in row if is_joker(c)) for row in opp_rows]
    return canon, mapping, opp_canon, opp_jokers


def natural_slot(card: str) -> int:
    return (RV[card[0]] - 2) * 4 + SUITS.index(card[1])


def feats_btn(canon, opp_canon, opp_jokers) -> np.ndarray:
    """213 dims: hero 54 | opp top 52 | opp mid 52 | opp bot 52 | joker counts 3."""
    x = np.zeros(FEATURE_SIZE, np.float32)
    for c in canon:
        if is_joker(c):
            x[HERO_JOKER_BASE + int(c[1:]) - 1] = 1.0
        else:
            x[natural_slot(c)] = 1.0
    for r, row in enumerate(opp_canon):
        for c in row:
            x[OPP_BASE + 52 * r + natural_slot(c)] = 1.0
    for r, n in enumerate(opp_jokers):
        x[OPP_JOKER_BASE + r] = float(n)
    return x


def encode_state(hero, opp_rows):
    """(features, canon, mapping) for one decision state, hero names as given."""
    canon, mapping, opp_canon, opp_jokers = canonical_btn(hero, opp_rows)
    return feats_btn(canon, opp_canon, opp_jokers), canon, mapping


def decode_key(key, canon, mapping):
    """The shared decode, tolerant of a key naming a card hero does not hold
    (returns None instead of raising, so loaders can count the failure)."""
    try:
        return action_index(key, canon, mapping)
    except KeyError:
        return None


def openings(draw):
    """Every distinct arrangement of the five, keyed as Rust keys them."""
    seen = {}
    for mask in range(3 ** 5):
        rows = [[], [], []]
        code = mask
        legal = True
        for card in draw:
            row = code % 3
            code //= 3
            rows[row].append(card)
            if len(rows[row]) > ROW_CAPACITY[row]:
                legal = False
                break
        if not legal:
            continue
        key = "|".join(",".join(sorted(row)) for row in rows)
        seen.setdefault(key, rows)
    return seen


class PolicyBtn(nn.Module):
    def __init__(self, input_dim: int = FEATURE_SIZE, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACTION_SIZE))

    def forward(self, x):
        out = self.net(x)
        return out.masked_fill(~torch.as_tensor(MASK, device=out.device), -1e9)


def policy_from_state(state) -> "PolicyBtn":
    """Rebuild the net with the widths the checkpoint was trained at."""
    first = state["net.0.weight"]
    return PolicyBtn(input_dim=first.shape[1], hidden=first.shape[0])


def soft_target(vals: np.ndarray, tau: float) -> np.ndarray:
    t = np.zeros(ACTION_SIZE, np.float32)
    fin = np.isfinite(vals)
    e = np.exp((vals[fin] - vals[fin].max()) / tau)
    t[fin] = e / e.sum()
    return t


def load_stage_a(tau):
    """Dense corrected-chain labels joined per lap; returns (X, T, ids, stats)."""
    xs, ts, ids = [], [], []
    stats = dict(label_rows=0, joined=0, unmapped_keys=0, illegal_keys=0,
                 rows_dropped=0, rows_with_232=0, per_lap={})
    for lap, f, reqf in STAGE_A:
        # Lap ids collide (1,685 ids appear in both laps with different
        # draws): a label file joins ONLY its own lap's request file.
        reqs = {}
        for l in open(D / reqf, encoding="utf-8"):
            r = json.loads(l)
            reqs[r["id"]] = r
        lap_stats = dict(label_rows=0, joined=0, kept=0)
        for l in open(D / f, encoding="utf-8"):
            row = json.loads(l)
            stats["label_rows"] += 1
            lap_stats["label_rows"] += 1
            r = reqs.get(row["id"])
            if r is None:
                continue
            stats["joined"] += 1
            lap_stats["joined"] += 1
            hero = rename_hero_jokers(r["draw"])
            x, canon, mp = encode_state(hero, r["opp_board"])
            vals = np.full(ACTION_SIZE, -np.inf, np.float32)
            bad = 0
            for a in row["actions"]:
                idx = decode_key(a["action_key"], canon, mp)
                if idx is None:
                    bad += 1
                    continue
                if not MASK[idx]:
                    stats["illegal_keys"] += 1
                    bad += 1
                    continue
                vals[idx] = max(vals[idx], a["value"])
            stats["unmapped_keys"] += bad
            if int(np.isfinite(vals).sum()) == 232:
                stats["rows_with_232"] += 1
            if bad:
                stats["rows_dropped"] += 1
                continue
            xs.append(x)
            ts.append(soft_target(vals, tau))
            ids.append(f"{lap}:{row['id']}")
            lap_stats["kept"] += 1
        stats["per_lap"][lap] = lap_stats
    return np.stack(xs), np.stack(ts), ids, stats


def label_fit(net, X, T, dev):
    """How well the net reproduces its own dense labels: top-1 / top-3 of the
    label argmax, and the soft cross-entropy.  A broken encoder shows here
    long before it shows on the referee holdout."""
    top1 = top3 = 0
    ce = 0.0
    with torch.no_grad():
        for s in range(0, len(X), 4096):
            logits = net(X[s:s + 4096])
            lab = T[s:s + 4096].argmax(-1)
            top1 += int((logits.argmax(-1) == lab).sum())
            top3 += int((logits.topk(3, -1).indices == lab[:, None]).any(-1).sum())
            ce += float(-(T[s:s + 4096] * torch.log_softmax(logits, -1)).sum(-1).sum())
    n = len(X)
    return dict(n=n, top1=top1 / n, top3=top3 / n, soft_ce=ce / n)


def load_stage_b(tau_a):
    """Referee material + the served evaluator's dense anchor per root."""
    rows = [json.loads(l) for l in open(D / MATERIAL, encoding="utf-8")]
    own = {}
    for l in open(D / PAIRS_META, encoding="utf-8"):
        p = json.loads(l)
        own.setdefault(p["root"], []).append((p["key"], p["own_score"]))
    out = []
    stats = dict(material_rows=len(rows), kept=0, sel_unmapped=0, dense_unmapped=0,
                 dense_rows_short=0, pick_outside_sel=0)
    for r in rows:
        hero = rename_hero_jokers(r["cards"].split(","))
        opp_rows = parse_board_spec(r["opp_board"])
        x, canon, mp = encode_state(hero, opp_rows)
        idxs, means = [], []
        skip = False
        for k, v in r["sel_means"].items():
            idx = decode_key(k, canon, mp)
            if idx is None or not MASK[idx]:
                stats["sel_unmapped"] += 1
                skip = True
                break
            idxs.append(idx)
            means.append(v)
        serving = decode_key(r["model_pick"], canon, mp)
        ref = decode_key(r["ref_pick"], canon, mp)
        if skip or serving is None or ref is None:
            continue
        if serving not in idxs or ref not in idxs:
            stats["pick_outside_sel"] += 1
        vals = np.full(ACTION_SIZE, -np.inf, np.float32)
        for k, s in own.get(r["id"], ()):
            idx = decode_key(k, canon, mp)
            if idx is None or not MASK[idx] or s is None:
                stats["dense_unmapped"] += 1
                continue
            vals[idx] = max(vals[idx], s)
        n_dense = int(np.isfinite(vals).sum())
        if n_dense != 232:
            stats["dense_rows_short"] += 1
        if n_dense == 0:
            continue
        # Depth weighting as in BB: escalated re-audits over scored pairs over
        # select-only agreements.  This material has no escalations; the
        # "ci" rows are the scored pairs.
        depth_w = 3.0 if r.get("escalated") else (2.0 if "ci" in r else 1.0)
        out.append(dict(id=r["id"], x=x, idxs=np.array(idxs),
                        means=np.array(means, np.float32),
                        verdict=r["verdict"], margin=float(r.get("margin", 0.0)),
                        weight=depth_w, serving=serving, ref=ref,
                        dense=soft_target(vals, tau_a)))
        stats["kept"] += 1
    return out, stats


def holdout_split(mat, n_hold):
    """Deterministic: the n_hold roots with the smallest sha256(id) are held out."""
    order = sorted(mat, key=lambda h: hashlib.sha256(h["id"].encode("utf-8")).hexdigest())
    return order[:n_hold], order[n_hold:]


def eval_hold(net, hold, dev):
    """Referee agreement / serving agreement / argmax-in-candidates / recovery,
    plus the v1-trap counter: how often the whole-232 argmax is a move nobody
    refereed (the failure that sank BB v1 -- novel picks confirmed at -14)."""
    agree_ref = agree_serve = in_cands = trap = 0
    ref_top4 = ref_top8 = serve_top8 = 0
    pick_val = serve_val = best_val = 0.0
    n = 0
    with torch.no_grad():
        for h in hold:
            cand = h["idxs"].tolist()
            if h["serving"] not in cand:
                continue
            n += 1
            logits = net(torch.tensor(h["x"], device=dev).unsqueeze(0))[0]
            sub = logits[torch.tensor(h["idxs"], device=dev)]
            pick_i = int(sub.argmax())
            g = int(logits.argmax())
            in_cands += g in set(cand)
            trap += g not in set(cand)
            agree_ref += int(h["idxs"][pick_i] == h["ref"])
            agree_serve += int(h["idxs"][pick_i] == h["serving"])
            pick_val += float(h["means"][pick_i])
            serve_val += float(h["means"][cand.index(h["serving"])])
            best_val += float(h["means"].max())
            # fence recall: the policy serves as a top-K shortlist for the
            # evaluator (pgate: argmax direct serving lost by 0.72/hand), so
            # what matters is whether the good openings are inside its top-K.
            top8 = logits.topk(8).indices.tolist()
            ref_top4 += h["ref"] in top8[:4]
            ref_top8 += h["ref"] in top8
            serve_top8 += h["serving"] in top8
    m = dict(n=n, agree_ref=agree_ref / n, agree_serve=agree_serve / n,
             argmax_in_cands=in_cands / n, v1_trap_count=trap, v1_trap_rate=trap / n,
             pick_val=pick_val / n, serve_val=serve_val / n, best_val=best_val / n,
             recovery=(pick_val - serve_val) / (best_val - serve_val + 1e-9),
             ref_in_top4=ref_top4 / n, ref_in_top8=ref_top8 / n, serve_in_top8=serve_top8 / n)
    print(f"  holdout(n={n}): 審判一致 {m['agree_ref']:.1%}  配信一致 {m['agree_serve']:.1%}  "
          f"global-argmaxが候補内 {m['argmax_in_cands']:.1%}  "
          f"v1-trap(未審判の手がargmax) {trap}/{n} = {m['v1_trap_rate']:.1%}", flush=True)
    print(f"  候補内価値: policy {m['pick_val']:+.3f}  serving {m['serve_val']:+.3f}  "
          f"best {m['best_val']:+.3f}  -> 回収率 {m['recovery']:.1%}", flush=True)
    print(f"  柵の再現率: 審判最善がtop-4 {m['ref_in_top4']:.1%} / top-8 {m['ref_in_top8']:.1%}  "
          f"配信手がtop-8 {m['serve_in_top8']:.1%}", flush=True)
    return m


# ----------------------------------------------------------------------------
# property tests
# ----------------------------------------------------------------------------

def _permute_cards(cards, perm):
    return [c if is_joker(c) else c[0] + perm[c[1]] for c in cards]


def _permute_key(key, perm):
    return "|".join(",".join(sorted(_permute_cards([c for c in part.split(",") if c], perm)))
                    for part in key.split("|"))


def _random_state(rng):
    deck = [r + s for r in RANKS for s in SUITS] + ["X1", "X2"]
    rng.shuffle(deck)
    hero, opp = deck[:5], deck[5:10]
    while True:
        rows = [[], [], []]
        for c in opp:
            rows[rng.randrange(3)].append(c)
        if len(rows[0]) <= 3:
            return hero, rows


def tie_symmetries(hero, opp_rows):
    """Suit maps that are symmetries of the state: every way of permuting
    suits within a group of PRESENT suits whose four masks are identical
    (e.g. hero Ac,Ad with the opponent's Kc,Kd both in the middle).

    Such a group is where the index tie-break fires on suits that matter.
    The features do not see the difference, and Python and Rust break the
    tie identically from the same original names, but a global permutation
    that swaps two tied suits maps each opening to its mirror image under
    the swap -- so across permutations the 232 indices agree only up to
    these maps.  Absent suits tie too, but nothing carries their label.
    """
    keys = suit_keys(hero, opp_rows)
    groups = {}
    for s in SUITS:
        if any(keys[s]):
            groups.setdefault(keys[s], []).append(s)
    tied = [g for g in groups.values() if len(g) > 1]
    maps = [dict(zip(SUITS, SUITS))]
    for group in tied:
        maps = [dict(m, **dict(zip(group, p))) for m in maps for p in itertools.permutations(group)]
    return maps


def has_present_tie(hero, opp_rows) -> bool:
    return len(tie_symmetries(hero, opp_rows)) > 1


def selftest():
    rng = random.Random(20260903)
    states = [
        (["Ah", "Kd", "7c", "7s", "2h"], [["Qs"], ["Jd", "Jh", "3h", "3d"], []]),
        (["As", "Kd", "X2", "7c", "2h"], [["X1"], ["4s", "3d"], ["Qc", "9c"]]),
        (["7d", "As", "X1", "Kd", "X2"], [[], ["5s", "6s"], ["2c", "Qc", "Qs"]]),
        (["2s", "Ac", "8h", "7c", "7h"], [["X1"], ["6s", "X2"], ["5h", "Qh"]]),
        # two suits absent from both sides: the residual index tie-break
        (["Ah", "Kh", "7c", "5c", "2h"], [["Qc"], ["Jh", "3h"], ["9c", "8c"]]),
        # hero suits tie, the opponent board breaks it
        (["Ac", "Ad", "7h", "5s", "2h"], [["Kd"], ["9s", "8s"], ["4c", "3h"]]),
        (["Ac", "Ad", "Ah", "As", "X1"], [["X2"], ["Kc", "Kd"], ["Kh", "Ks"]]),
        # present-suit ties: swapping c<->d is a symmetry of these states
        (["Ac", "Ad", "7h", "5s", "2h"], [["Ks"], ["Kc", "Kd"], ["4s", "3h"]]),
        (["Ac", "Ad", "Ah", "X1", "X2"], [["Kc", "Kd", "Kh"], ["2s"], ["3s"]]),
        (["9c", "9d", "9h", "9s", "2c"], [["5s"], [], ["Qc", "Qd", "Qh", "Qs"]]),
    ] + [_random_state(rng) for _ in range(120)]
    legal = {i for i in range(ACTION_SIZE) if MASK[i]}
    assert len(legal) == 232
    checked = tied_states = strict_states = 0
    for hero, opp in states:
        x0, canon, mp = encode_state(hero, opp)
        assert x0.shape == (FEATURE_SIZE,) and x0[:54].sum() == 5 and x0[54:210].sum() + x0[210:].sum() == 5
        keys = list(openings(hero))
        assert len(keys) == 232, (hero, len(keys))
        idx0 = {k: decode_key(k, canon, mp) for k in keys}
        assert set(idx0.values()) == legal, hero
        syms = tie_symmetries(hero, opp)
        tied_states += len(syms) > 1
        strict_states += len(syms) == 1
        # (i) every global suit permutation, (ii) with hero order and row-
        # internal order shuffled on top: same bytes; and the same index per
        # opening -- strictly when no two present suits tie, otherwise up to
        # the tie symmetry (the ONE state map g under which the permuted
        # state's relabelling equals the original's must exist, and then every
        # opening's index must be that of its g-image).
        for perm in itertools.permutations(SUITS):
            pm = dict(zip(SUITS, perm))
            h2 = _permute_cards(hero, pm)
            rng.shuffle(h2)
            o2 = [_permute_cards(row, pm) for row in opp]
            for row in o2:
                rng.shuffle(row)
            x2, c2, m2 = encode_state(h2, o2)
            assert np.array_equal(x0, x2), (hero, opp, perm)
            assert c2 == canon, (hero, opp, perm)
            fits = [g for g in syms
                    if all(m2[c if is_joker(c) else c[0] + pm[c[1]]] == mp[c if is_joker(c) else c[0] + g[c[1]]]
                           for c in hero)]
            assert len(fits) >= 1, (hero, opp, perm, mp, m2)
            g = fits[0]
            for k in keys:
                assert decode_key(_permute_key(k, pm), c2, m2) == idx0[_permute_key(k, g)], (hero, opp, perm, k)
            checked += 1
    assert strict_states > 0 and tied_states > 0, (strict_states, tied_states)
    # (iii) hero X2 + opp X1 renumbers to hero X1; the swapped spelling is the
    # same state, byte for byte and index for index.
    hero, opp = ["As", "Kd", "X2", "7c", "2h"], [["X1"], ["4s", "3d"], ["Qc", "9c"]]
    x, canon, mp = encode_state(hero, opp)
    assert canon[-1] == "X1" and mp["X2"] == "X1" and x[52] == 1.0 and x[53] == 0.0
    assert list(x[210:213]) == [1.0, 0.0, 0.0]
    hero_s, opp_s = ["As", "Kd", "X1", "7c", "2h"], [["X2"], ["4s", "3d"], ["Qc", "9c"]]
    xs, cs, ms = encode_state(hero_s, opp_s)
    assert np.array_equal(x, xs)
    for k in openings(hero):
        assert decode_key(k, canon, mp) == decode_key(k.replace("X2", "X1"), cs, ms)
    # the tie cases have the documented relabelling
    _, mp, _, _ = canonical_btn(["Ah", "Kh", "7c", "5c", "2h"], [["Qc"], ["Jh", "3h"], ["9c", "8c"]])
    assert mp["Ah"] == "Ac" and mp["7c"] == "7d", mp
    _, mp, opp_c, _ = canonical_btn(["Ac", "Ad", "7h", "5s", "2h"], [["Kd"], ["9s", "8s"], ["4c", "3h"]])
    assert mp["Ad"] == "Ac" and mp["Ac"] == "Ad" and opp_c[0] == ["Kc"], (mp, opp_c)
    # the data's deal-order joker spelling
    assert rename_hero_jokers(["X2", "8d", "X1", "4d", "8s"]) == ["X1", "8d", "X2", "4d", "8s"]
    assert rename_hero_jokers(["2c", "X2", "9h"]) == ["2c", "X1", "9h"]
    assert parse_board_spec("Kd|4c,4s|Qc,Qs") == [["Kd"], ["4c", "4s"], ["Qc", "Qs"]]
    assert parse_board_spec("|5s,6s|2c,Qc,Qs") == [[], ["5s", "6s"], ["2c", "Qc", "Qs"]]
    print(f"selftest ok: {len(states)} states x 24 suit permutations x 232 openings "
          f"({checked} permuted states; {strict_states} strict, {tied_states} with a present-suit tie "
          f"checked up to the tie symmetry), joker renumbering, tie-breaks")


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tau-a", type=float, default=2.0)
    ap.add_argument("--tau-b", type=float, default=1.5)
    ap.add_argument("--epochs-a", type=int, default=30)
    ap.add_argument("--epochs-b", type=int, default=200)
    ap.add_argument("--lr-a", type=float, default=1e-3)
    ap.add_argument("--lr-b", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="diagnostic only: hold this fraction of stage-A roots out of A "
                         "(sha256 on lap:id) and report the label fit on them; 0 = the BB recipe")
    ap.add_argument("--ablate-opp", action="store_true",
                    help="diagnostic only: zero the opponent features (dims 54..212) everywhere, "
                         "to show what the opponent board adds to the label fit")
    ap.add_argument("--no-depth-weights", action="store_true")
    ap.add_argument("--no-margin-hinge", action="store_true")
    ap.add_argument("--torch-seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    # Weight init and batch order decide 21-52% of BB holdout recovery on an
    # identical recipe; a run without a pinned seed is not a measurement.
    if args.torch_seed is None:
        ap.error("--torch-seed is required (recipe comparisons need it pinned)")
    if args.out is None:
        args.out = D / f"t0_btn_policy_s{args.torch_seed}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.torch_seed)
    rng = random.Random(20260828)
    report = dict(args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  device=dev)

    print("loading stage A ...", flush=True)
    xa, ta, ids_a, stats_a = load_stage_a(args.tau_a)
    report["join"] = stats_a
    print(f"A: {len(xa)} roots  (labels {stats_a['label_rows']}, joined {stats_a['joined']}, "
          f"rows with 232 mapped {stats_a['rows_with_232']}, unmapped keys {stats_a['unmapped_keys']}, "
          f"illegal {stats_a['illegal_keys']}, dropped {stats_a['rows_dropped']})", flush=True)
    xv = tv = None
    if args.val_frac > 0:
        digest = np.array([int(hashlib.sha256(i.encode("utf-8")).hexdigest()[:8], 16) for i in ids_a])
        cut = np.sort(digest)[int(len(digest) * args.val_frac)]
        val = digest < cut
        xv, tv = xa[val], ta[val]
        xa, ta = xa[~val], ta[~val]
        print(f"A: diagnostic split, {len(xa)} train / {len(xv)} val", flush=True)
    mat, stats_b = load_stage_b(args.tau_a)
    if args.ablate_opp:
        xa[:, OPP_BASE:] = 0.0
        if xv is not None:
            xv[:, OPP_BASE:] = 0.0
        for h in mat:
            h["x"][OPP_BASE:] = 0.0
        print("DIAGNOSTIC: opponent features zeroed everywhere", flush=True)
    hold, fit_b = holdout_split(mat, args.holdout)
    rng.shuffle(fit_b)
    if args.no_depth_weights:
        for h in fit_b:
            h["weight"] = 1.0
    # Relative authority, not a global LR inflation: normalise to mean 1.
    wm = sum(h["weight"] for h in fit_b) / len(fit_b)
    for h in fit_b:
        h["weight"] /= wm
    stats_b.update(fit=len(fit_b), holdout=len(hold), weight_mean_raw=wm,
                   holdout_ids=[h["id"] for h in hold])
    report["stage_b"] = stats_b
    print(f"B: {len(fit_b)} fit + {len(hold)} holdout (sha256 split; weight mean {wm:.2f} normalized; "
          f"sel unmapped {stats_b['sel_unmapped']}, dense unmapped {stats_b['dense_unmapped']}, "
          f"dense rows short {stats_b['dense_rows_short']})", flush=True)

    net = PolicyBtn(hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr_a)
    XA = torch.tensor(xa, device=dev)
    TA = torch.tensor(ta, device=dev)
    for ep in range(args.epochs_a):
        perm = torch.randperm(len(XA), device=dev)
        tot = 0.0
        for s in range(0, len(XA), 4096):
            b = perm[s:s + 4096]
            loss = -(TA[b] * torch.log_softmax(net(XA[b]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 9:
            print(f"A ep{ep+1} loss {tot/len(XA):.4f}", flush=True)

    report["stage_a_fit_train"] = label_fit(net, XA, TA, dev)
    print(f"A fit (train): top-1 {report['stage_a_fit_train']['top1']:.1%}  "
          f"top-3 {report['stage_a_fit_train']['top3']:.1%}  "
          f"soft-CE {report['stage_a_fit_train']['soft_ce']:.4f}", flush=True)
    if xv is not None:
        report["stage_a_fit_val"] = label_fit(net, torch.tensor(xv, device=dev), torch.tensor(tv, device=dev), dev)
        print(f"A fit (val):   top-1 {report['stage_a_fit_val']['top1']:.1%}  "
              f"top-3 {report['stage_a_fit_val']['top3']:.1%}  "
              f"soft-CE {report['stage_a_fit_val']['soft_ce']:.4f}", flush=True)
    print("after stage A:", flush=True)
    report["after_A"] = eval_hold(net, hold, dev)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=net.state_dict(), stage="A"), args.out / "policy_stage_a.pt")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr_b)
    XB = torch.tensor(np.stack([h["x"] for h in fit_b]), device=dev)
    TB = torch.tensor(np.stack([h["dense"] for h in fit_b]), device=dev)
    replay = torch.randperm(len(XA))[:len(XA) // 10]
    trace = []
    for ep in range(args.epochs_b):
        order = list(range(len(fit_b)))
        rng.shuffle(order)
        tot = 0.0
        opt.zero_grad()
        for step, i in enumerate(order):
            h = fit_b[i]
            logits = net(XB[i].unsqueeze(0))[0]
            # anchor: the served evaluator's full 232-opening landscape
            loss = -(TB[i] * torch.log_softmax(logits, -1)).sum()
            # correction: the referee's ordering over its candidates
            idxs = torch.tensor(h["idxs"], device=dev)
            sub = logits[idxs]
            m = torch.tensor(h["means"], device=dev)
            target = torch.softmax(m / args.tau_b, -1)
            loss = loss + 2.0 * -(target * torch.log_softmax(sub, -1)).sum()
            if h["verdict"] == "error":
                li = logits[h["ref"]] - logits[h["serving"]]
                hinge_w = 1.0 if args.no_margin_hinge else min(1.0 + h["margin"] / 4.0, 3.0)
                loss = loss + hinge_w * torch.relu(1.0 - li)
            loss = loss * h.get("weight", 1.0)
            tot += loss.item()
            loss.backward()
            if step % 64 == 63:
                opt.step()
                opt.zero_grad()
        b = replay[torch.randperm(len(replay))[:2048]].to(dev)
        loss = -(TA[b] * torch.log_softmax(net(XA[b]), -1)).sum(-1).mean() * 3.0
        loss.backward()
        opt.step()
        opt.zero_grad()
        if ep % 50 == 49:
            print(f"B ep{ep+1} loss {tot/len(fit_b):.4f}", flush=True)
            trace.append(dict(epoch=ep + 1, loss=tot / len(fit_b), **eval_hold(net, hold, dev)))
    report["B_trace"] = trace

    torch.save(dict(model_state_dict=net.state_dict(), stage="AB"), args.out / "policy_best.pt")
    print("saved ->", args.out / "policy_best.pt", flush=True)
    print("final:", flush=True)
    report["after_AB"] = eval_hold(net, hold, dev)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("report ->", args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
