"""
Convert MC teacher JSONL to NPZ training data (822-dim: state + histogram).

Reads JSONL from generate_mc_teacher.py, creates:
  states.npy    (N, 822)  - 522 board encoding + 300 histogram
  actions.npy   (N,)      - best action index
  action_evs.npy (N, 250) - all candidates' EVs (padded)
  valid_masks.npy (N, 250) - which actions valid
  rewards.npy   (N,)      - best EV
  busted.npy    (N,)      - best candidate bust rate
  fl_entry.npy  (N,)      - best candidate FL rate
  metadata.json           - statistics

Usage:
    python ai/training/convert_mc_teacher.py data/mc_teacher/merged.jsonl \
        --output data/mc_teacher_npz/
"""
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM
from ai.engine.action_space import Action, get_semantic_action_index

MAX_ACTIONS = 250
HIST_DIM = 300
FULL_DIM = STATE_DIM + HIST_DIM  # 522 + 300 = 822


def row_name_to_pos(name):
    """Convert row name to position index: top=0, mid=1, bot=2."""
    return {"top": 0, "middle": 1, "bottom": 2}.get(name, -1)


def normalize_joker_refs(placements, discard, dealt_cards):
    """Map Rust/prob-engine joker refs back to concrete dealt joker ids.

    The prob engine may emit a generic Xj while Python action indexing uses the
    concrete X1/X2 ids from dealt_cards. Assign unresolved joker refs to the
    dealt jokers not already used by the candidate.
    """
    dealt_jokers = [c for c in dealt_cards if str(c).startswith("X")]
    used_jokers = []
    unresolved = []

    raw = []
    for i, (card, pos) in enumerate(placements):
        raw.append(["placement", i, card, pos])
        if str(card).startswith("X"):
            if card in dealt_jokers and card not in used_jokers:
                used_jokers.append(card)
            else:
                unresolved.append(("placement", i))

    raw_discard = discard
    if discard is not None and str(discard).startswith("X"):
        if discard in dealt_jokers and discard not in used_jokers:
            used_jokers.append(discard)
        else:
            unresolved.append(("discard", None))

    available = [c for c in dealt_jokers if c not in used_jokers]

    def next_joker():
        if available:
            return available.pop(0)
        if dealt_jokers:
            return dealt_jokers[0]
        return "Xj"

    replacements = {}
    for where, idx in unresolved:
        replacements[(where, idx)] = next_joker()

    norm_placements = []
    for i, (card, pos) in enumerate(placements):
        if str(card).startswith("X") and (card not in dealt_jokers or card == "Xj"):
            card = replacements.get(("placement", i), card)
        norm_placements.append((card, pos))

    norm_discard = raw_discard
    if raw_discard is not None and str(raw_discard).startswith("X") and (
        raw_discard not in dealt_jokers or raw_discard == "Xj"
    ):
        norm_discard = replacements.get(("discard", None), raw_discard)
    return norm_placements, norm_discard


def build_observation(record):
    """Build Observation from a teacher data record."""
    board = record["board"]
    dealt = record["dealt"]
    exclude = record.get("exclude", [])

    bs = Board()
    for c in board.get("top", []):
        bs.top.append(c)
    for c in board.get("mid", []) + board.get("middle", []):
        bs.middle.append(c)
    for c in board.get("bot", []) + board.get("bottom", []):
        bs.bottom.append(c)

    # Reconstruct opponent board from exclude (approximate)
    bo = Board()
    # Exclude contains opponent board + self discards. We don't separate them
    # for encoding, but the card matrix just needs them in LOC_UNSEEN or discard.
    # For simplicity, put self discards in known_discards.

    obs = Observation(
        board_self=bs,
        board_opponent=bo,
        dealt_cards=tuple(dealt),
        known_discards_self=tuple(),
        turn=record["turn"],
        is_btn=True,
        is_fl=False,
        opp_is_fl=False,
        chips_self=100.0,
        chips_opponent=100.0,
    )
    return obs


def action_to_index_t0(placements, dealt_cards):
    """Map T0 placements to action index via enumeration."""
    from ai.engine.action_space import get_initial_actions

    placements, _ = normalize_joker_refs(placements, None, dealt_cards)

    # Build target key
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card, pos in placements:
        by_pos[pos].append(card)
    target = (
        tuple(sorted(by_pos["top"])),
        tuple(sorted(by_pos["middle"])),
        tuple(sorted(by_pos["bottom"])),
    )

    b = Board()
    all_actions = get_initial_actions(dealt_cards, b)
    for idx, a in enumerate(all_actions):
        a_pos = {"top": [], "middle": [], "bottom": []}
        for card, pos in a.placements:
            a_pos[pos].append(card)
        a_key = (
            tuple(sorted(a_pos["top"])),
            tuple(sorted(a_pos["middle"])),
            tuple(sorted(a_pos["bottom"])),
        )
        if a_key == target:
            return min(idx, MAX_ACTIONS - 1)
    return 0


def action_to_index_t1plus(candidate, dealt_cards):
    """Map T1+ candidate to fixed 27-slot semantic index."""
    placements = candidate.get("placements", [])
    if len(placements) != 2:
        return 0

    placements, discard = normalize_joker_refs(
        placements,
        candidate.get("discard"),
        dealt_cards,
    )
    placed_cards = {card for card, _pos in placements}
    if discard is None:
        discards = [card for card in dealt_cards if card not in placed_cards]
        if len(discards) != 1:
            return 0
        discard = discards[0]

    if any(row_name_to_pos(pos) < 0 for _card, pos in placements):
        return 0

    return get_semantic_action_index(
        Action(placements=list(placements), discard=discard),
        dealt_cards,
    )


def get_candidate_ev(candidate, eval_mode):
    """Extract EV from candidate depending on eval mode."""
    if eval_mode.startswith("mc") or eval_mode == "t0_ladder":
        mc = candidate.get("mc", {})
        return mc.get("avg_score", 0.0)
    else:
        return candidate.get("ev", 0.0)


def get_candidate_bust(candidate, eval_mode):
    """Extract bust rate from candidate."""
    if eval_mode.startswith("mc") or eval_mode == "t0_ladder":
        mc = candidate.get("mc", {})
        return mc.get("bust_rate", 0.0)
    else:
        return candidate.get("bust_prob", 0.0)


def get_candidate_fl(candidate, eval_mode):
    """Extract FL rate from candidate."""
    if eval_mode.startswith("mc") or eval_mode == "t0_ladder":
        mc = candidate.get("mc", {})
        return mc.get("fl_rate", 0.0)
    else:
        return candidate.get("fl_rate", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Convert MC teacher JSONL to NPZ")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--turns", type=str, default="0,1,2,3,4",
                        help="Which turns to include (default: all)")
    args = parser.parse_args()

    include_turns = set(int(t) for t in args.turns.split(","))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.input}")
    print(f"Output:  {out_dir}")
    print(f"Turns:   {sorted(include_turns)}")
    print(f"Dims:    {FULL_DIM} (state={STATE_DIM} + hist={HIST_DIM})")

    # First pass: count records
    n_total = 0
    with open(args.input) as f:
        for line in f:
            d = json.loads(line)
            turn = d.get("turn", -1)
            if turn >= 0 and turn in include_turns and d.get("candidates"):
                n_total += 1

    print(f"Records: {n_total}")

    if n_total == 0:
        print("No records to convert!")
        return

    # Allocate arrays
    states = np.zeros((n_total, FULL_DIM), dtype=np.float32)
    actions = np.zeros(n_total, dtype=np.int32)
    action_evs = np.zeros((n_total, MAX_ACTIONS), dtype=np.float32)
    valid_masks = np.zeros((n_total, MAX_ACTIONS), dtype=np.bool_)
    rewards = np.zeros(n_total, dtype=np.float32)
    busted = np.zeros(n_total, dtype=np.float32)
    fl_entry = np.zeros(n_total, dtype=np.float32)

    # Second pass: convert
    idx = 0
    stats = {"turns": {}, "skipped": 0}
    t_start = time.time()

    with open(args.input) as f:
        for line in f:
            d = json.loads(line)
            turn = d.get("turn", -1)
            if turn < 0 or turn not in include_turns or not d.get("candidates"):
                continue

            candidates = d["candidates"]
            eval_mode = d.get("eval_mode", "exact")
            hist_after = d.get("hist_after", [0.0] * HIST_DIM)

            # Build observation and encode state
            try:
                obs = build_observation(d)
                state_522 = encode_state(obs)
            except Exception as e:
                stats["skipped"] += 1
                continue

            # Histogram features (300 dims)
            hist_arr = np.array(hist_after[:HIST_DIM], dtype=np.float32)
            if len(hist_arr) < HIST_DIM:
                hist_arr = np.pad(hist_arr, (0, HIST_DIM - len(hist_arr)))

            # Full 822-dim state
            states[idx] = np.concatenate([state_522, hist_arr])

            # Best action index
            best = candidates[0]
            if turn == 0:
                actions[idx] = action_to_index_t0(best["placements"], d["dealt"])
            else:
                actions[idx] = action_to_index_t1plus(best, d["dealt"])

            # All candidates' EVs
            for cand in candidates[:MAX_ACTIONS]:
                ev = get_candidate_ev(cand, eval_mode)
                if turn == 0:
                    action_idx = action_to_index_t0(cand["placements"], d["dealt"])
                else:
                    action_idx = action_to_index_t1plus(cand, d["dealt"])
                action_evs[idx, action_idx] = ev
                valid_masks[idx, action_idx] = True

            # Labels
            rewards[idx] = get_candidate_ev(best, eval_mode)
            busted[idx] = get_candidate_bust(best, eval_mode)
            fl_entry[idx] = get_candidate_fl(best, eval_mode)

            # Stats
            stats["turns"][turn] = stats["turns"].get(turn, 0) + 1

            idx += 1
            if idx % 1000 == 0:
                print(f"  {idx}/{n_total} records converted...")

    # Trim if skipped
    if idx < n_total:
        states = states[:idx]
        actions = actions[:idx]
        action_evs = action_evs[:idx]
        valid_masks = valid_masks[:idx]
        rewards = rewards[:idx]
        busted = busted[:idx]
        fl_entry = fl_entry[:idx]

    # Save
    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "action_evs.npy", action_evs)
    np.save(out_dir / "valid_masks.npy", valid_masks)
    np.save(out_dir / "rewards.npy", rewards)
    np.save(out_dir / "busted.npy", busted)
    np.save(out_dir / "fl_entry.npy", fl_entry)

    # Metadata
    metadata = {
        "n_samples": idx,
        "state_dim": FULL_DIM,
        "turns": stats["turns"],
        "skipped": stats["skipped"],
        "avg_reward": float(rewards.mean()),
        "avg_bust": float(busted.mean()),
        "avg_fl": float(fl_entry.mean()),
        "source": args.input,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n=== Conversion complete ({elapsed:.1f}s) ===")
    print(f"  Samples:  {idx}")
    print(f"  State dim: {FULL_DIM}")
    print(f"  Turns:    {stats['turns']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Avg reward: {rewards.mean():+.2f}")
    print(f"  Avg bust:   {busted.mean():.1%}")
    print(f"  Avg FL:     {fl_entry.mean():.1%}")
    print(f"  Saved to:   {out_dir}")


if __name__ == "__main__":
    main()
