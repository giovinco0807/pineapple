"""
Generate Off-Policy VN Training Data from Expectimax Results

For each decision point, applies suboptimal actions (bottom N% by EV) to create
"bad state" training samples. The VN learns to evaluate dangerous board
configurations that the BC policy might reach through mistakes.

Data volume estimate (bottom_frac=0.5, 329 files):
  T0:  ~38K   (232 * 50% * 329)
  T1:  ~231K  (27 * 50% * 50 CRs * 329)
  T2:  ~658K  (~8 * 50% * 500 CRs * 329)
  T3:  ~1.48M (~5 * 50% * 1500 CRs * 329)
  T4:  ~2.96M (~6 * 50% * 3000 CRs * 329)
  Total: ~5.4M samples

Labels:
  score:    Expectimax EV (already computed)
  busted:   T4 exact, T3 MC (optional), others 0
  fl_entry: T4 exact, others 0

Output: value_train.npz compatible with train_value.py

Usage:
    python -m ai.training.generate_offpolicy_data data/expectimax_results_v3/ \\
        --output data/offpolicy_train --bottom-frac 0.5
    python -m ai.training.generate_offpolicy_data data/expectimax_results_v3/ \\
        --output data/offpolicy_train --bottom-frac 0.5 --t3-mc 5
"""
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM, ALL_CARDS
from ai.engine.game_engine import (
    evaluate_hand,
    check_fl_entry,
    evaluate_board_with_joker_constraint,
)
from ai.engine.action_space import get_turn_actions

ARROW = "\u2192"
ROW_MAP = {"T": "top", "M": "middle", "B": "bottom"}


# ── Parsing ──────────────────────────────────────────────────

def parse_t0_placements(desc):
    """Parse T0 action desc like '4s→M 4h→M 6s→B Qs→B Ks→T'."""
    placements = []
    for part in desc.split():
        card, row = part.split(ARROW)
        placements.append((card, ROW_MAP[row]))
    return placements


def parse_turn_placements(desc):
    """Parse T1-T4 action desc like 'd:2c 6c→B Qc→B'."""
    parts = desc.split()
    discard = parts[0][2:]
    placements = []
    for part in parts[1:]:
        card, row = part.split(ARROW)
        placements.append((card, ROW_MAP[row]))
    return discard, placements


# ── Board helpers ────────────────────────────────────────────

def apply_to_board(board, placements):
    """Return new Board with placements applied."""
    b = Board(top=list(board.top), middle=list(board.middle),
              bottom=list(board.bottom))
    for card, pos in placements:
        getattr(b, pos).append(card)
    return b


def check_bust(top, mid, bot):
    """True if completed board violates row ordering."""
    return bool(evaluate_board_with_joker_constraint(top, mid, bot)["busted"])


def terminal_bust_fl(board):
    """Return exact terminal bust/FL labels from one canonical evaluation."""
    evaluated = evaluate_board_with_joker_constraint(
        board.top, board.middle, board.bottom
    )
    return (
        1.0 if evaluated["busted"] else 0.0,
        1.0 if evaluated["fl_entry"] else 0.0,
    )


def bust_fl_mc(board, remaining, n_mc=5):
    """MC bust/FL probability for 11-card board (post-T3 action).

    Samples random T4 deals from remaining cards, checks if any
    valid T4 placement avoids bust. Also checks FL entry.
    Returns (bust_prob, fl_prob).
    """
    if len(remaining) < 3:
        return 0.0, 0.0
    n_bust = 0
    n_fl = 0
    for _ in range(n_mc):
        deal = random.sample(remaining, 3)
        actions = get_turn_actions(deal, board)
        best_board = None
        for a in actions:
            b = apply_to_board(board, a.placements)
            if not check_bust(b.top, b.middle, b.bottom):
                best_board = b
                break
        if best_board is None:
            n_bust += 1
        else:
            evaluated = evaluate_board_with_joker_constraint(
                best_board.top, best_board.middle, best_board.bottom
            )
            if evaluated["fl_entry"]:
                n_fl += 1
    return n_bust / n_mc, n_fl / n_mc


def bust_fl_mc_deep(board, remaining, turns_left, n_mc=10):
    """MC bust/FL probability for boards with multiple turns remaining.

    Plays out `turns_left` turns using greedy placement (pick first
    non-busting action, or random if all bust), then checks final bust/FL.
    Returns (bust_prob, fl_prob).

    Args:
        board: Current board state (post-action)
        remaining: List of unseen cards
        turns_left: Number of turns to play out (T2→2, T1→3)
        n_mc: Number of MC samples
    """
    if len(remaining) < 3 * turns_left:
        return 0.0, 0.0

    n_bust = 0
    n_fl = 0
    for _ in range(n_mc):
        pool = list(remaining)
        random.shuffle(pool)
        b = Board(top=list(board.top), middle=list(board.middle),
                  bottom=list(board.bottom))
        idx = 0
        busted = False

        for t in range(turns_left):
            if idx + 3 > len(pool):
                break
            deal = pool[idx:idx + 3]
            idx += 3
            actions = get_turn_actions(deal, b)
            if not actions:
                break

            if t == turns_left - 1:
                # Last turn: check if any placement avoids bust
                best_board = None
                for a in actions:
                    ab = apply_to_board(b, a.placements)
                    if not check_bust(ab.top, ab.middle, ab.bottom):
                        best_board = ab
                        break
                if best_board is None:
                    n_bust += 1
                    busted = True
                else:
                    b = best_board
            else:
                # Intermediate turn: pick first valid placement
                placed = False
                for a in actions:
                    ab = apply_to_board(b, a.placements)
                    b = ab
                    placed = True
                    break
                if not placed:
                    break
        else:
            # All turns completed
            if not busted and b.is_complete():
                if check_bust(b.top, b.middle, b.bottom):
                    n_bust += 1
                    busted = True

        if not busted and b.is_complete():
            evaluated = evaluate_board_with_joker_constraint(
                b.top, b.middle, b.bottom
            )
            if evaluated["fl_entry"]:
                n_fl += 1

    return n_bust / n_mc, n_fl / n_mc


def encode_post_action(board, turn, discards=None):
    """Encode a post-action board state (no dealt cards in hand)."""
    obs = Observation(
        board_self=board,
        board_opponent=Board(),
        dealt_cards=[],
        known_discards_self=discards or [],
        turn=turn,
        is_btn=True,
    )
    return encode_state(obs)


# ── File processing ──────────────────────────────────────────

def process_file(filepath, bottom_frac, t3_mc, buf, t2_mc=0, t1_mc=0):
    """Process one Expectimax JSON → append off-policy records to buf."""
    try:
        with open(filepath, encoding="utf-8") as f:
            result = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0

    if "all_actions" not in result or not result["all_actions"]:
        return 0

    n_added = 0

    # ---- T0 off-policy ----
    actions = result["all_actions"]
    n = len(actions)
    cutoff = n - int(n * bottom_frac)
    cutoff = max(cutoff, 1)  # always skip best action

    for ar in actions[cutoff:]:
        placements = parse_t0_placements(ar["action_desc"])
        board = apply_to_board(Board(), placements)
        state = encode_post_action(board, turn=0)
        buf.add(state, ar["ev"], 0, 0.0, 0.0)
        n_added += 1

    # ---- T1-T4 off-policy ----
    for cr in result.get("choice_records", []):
        turn = cr["turn"]
        board = Board(top=list(cr["top"]), middle=list(cr["mid"]),
                      bottom=list(cr["bot"]))
        cr_actions = cr["actions"]
        n_act = len(cr_actions)
        act_cutoff = n_act - int(n_act * bottom_frac)
        act_cutoff = max(act_cutoff, 1)

        for ar in cr_actions[act_cutoff:]:
            discard, placements = parse_turn_placements(ar["desc"])
            board_after = apply_to_board(board, placements)
            ncards = (len(board_after.top) + len(board_after.middle)
                      + len(board_after.bottom))

            # Bust / FL labels
            if ncards == 13:  # T4: exact
                bust, fl = terminal_bust_fl(board_after)
            elif ncards == 11 and t3_mc > 0:  # T3: 1-turn MC
                known = set(board_after.top + board_after.middle
                            + board_after.bottom)
                known.add(discard)
                remaining = [c for c in ALL_CARDS if c not in known]
                bust, fl = bust_fl_mc(board_after, remaining, n_mc=t3_mc)
            elif ncards == 9 and t2_mc > 0:  # T2: 2-turn deep MC
                known = set(board_after.top + board_after.middle
                            + board_after.bottom)
                known.add(discard)
                remaining = [c for c in ALL_CARDS if c not in known]
                bust, fl = bust_fl_mc_deep(
                    board_after, remaining, turns_left=2, n_mc=t2_mc)
            elif ncards == 7 and t1_mc > 0:  # T1: 3-turn deep MC
                known = set(board_after.top + board_after.middle
                            + board_after.bottom)
                known.add(discard)
                remaining = [c for c in ALL_CARDS if c not in known]
                bust, fl = bust_fl_mc_deep(
                    board_after, remaining, turns_left=3, n_mc=t1_mc)
            else:
                bust = 0.0
                fl = 0.0

            state = encode_post_action(
                board_after, turn=turn, discards=[discard])
            buf.add(state, ar["ev"], turn, bust, fl)
            n_added += 1

    return n_added


# ── Chunked buffer (memory-efficient) ────────────────────────

CHUNK_SIZE = 50_000


class RecordBuffer:
    """Accumulates records in chunks to avoid massive Python lists."""

    def __init__(self):
        self._buf_s = []   # state arrays
        self._buf_sc = []  # scores
        self._buf_t = []   # turns
        self._buf_b = []   # busted
        self._buf_f = []   # fl_entry
        self._chunks = []  # list of (states, scores, turns, busted, fl) arrays
        self._flushed = 0

    @property
    def total(self):
        return self._flushed + len(self._buf_s)

    def add(self, state, score, turn, bust, fl):
        self._buf_s.append(state.astype(np.float16))
        self._buf_sc.append(score)
        self._buf_t.append(turn)
        self._buf_b.append(bust)
        self._buf_f.append(fl)
        if len(self._buf_s) >= CHUNK_SIZE:
            self._flush()

    def _flush(self):
        if not self._buf_s:
            return
        n = len(self._buf_s)
        self._chunks.append((
            np.stack(self._buf_s),
            np.array(self._buf_sc, dtype=np.float32),
            np.array(self._buf_t, dtype=np.int32),
            np.array(self._buf_b, dtype=np.float32),
            np.array(self._buf_f, dtype=np.float32),
        ))
        self._flushed += n
        self._buf_s.clear()
        self._buf_sc.clear()
        self._buf_t.clear()
        self._buf_b.clear()
        self._buf_f.clear()

    def save_to_npz(self, output_path):
        """Save directly to NPZ using memmap to avoid OOM on large datasets."""
        import tempfile
        self._flush()
        if not self._chunks:
            print("No chunks to save!")
            return 0

        N = self._flushed
        print(f"  Total: {N:,} samples, {len(self._chunks)} chunks")

        # Create memmap temp files for each array (prefer D: if available)
        tmp_base = "D:/tmp" if Path("D:/").exists() else None
        if tmp_base:
            Path(tmp_base).mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=tmp_base)
        obs_mm = np.lib.format.open_memmap(
            f"{tmpdir}/obs.npy", mode='w+', dtype=np.float16, shape=(N, STATE_DIM))
        score_mm = np.lib.format.open_memmap(
            f"{tmpdir}/score.npy", mode='w+', dtype=np.float32, shape=(N,))
        turn_mm = np.lib.format.open_memmap(
            f"{tmpdir}/turn.npy", mode='w+', dtype=np.int32, shape=(N,))
        busted_mm = np.lib.format.open_memmap(
            f"{tmpdir}/busted.npy", mode='w+', dtype=np.float32, shape=(N,))
        fl_mm = np.lib.format.open_memmap(
            f"{tmpdir}/fl.npy", mode='w+', dtype=np.float32, shape=(N,))

        # Fill memmap chunk by chunk
        offset = 0
        for i, (s, sc, t, b, f) in enumerate(self._chunks):
            n = len(s)
            obs_mm[offset:offset+n] = s
            score_mm[offset:offset+n] = sc
            turn_mm[offset:offset+n] = t
            busted_mm[offset:offset+n] = b
            fl_mm[offset:offset+n] = f
            offset += n
            if (i + 1) % 50 == 0:
                print(f"    chunk {i+1}/{len(self._chunks)}")

        # Print stats from memmap (no extra memory)
        turn_counts = Counter(turn_mm.tolist())
        print(f"\n{'='*60}")
        print(f"  Off-policy training data: {N:,} samples")
        print(f"{'='*60}")
        for t in sorted(turn_counts):
            mask = turn_mm == t
            avg_ev = float(np.mean(score_mm[mask]))
            bust_r = float(np.mean(busted_mm[mask]))
            fl_r = float(np.mean(fl_mm[mask]))
            print(f"  T{t}: {turn_counts[t]:>8,}  "
                  f"EV={avg_ev:>+6.1f}  "
                  f"bust={bust_r:.1%}  FL={fl_r:.1%}")
        print(f"  {'─'*54}")
        print(f"  Total: {N:>8,}  "
              f"EV={float(np.mean(score_mm)):>+6.1f}  "
              f"bust={float(np.mean(busted_mm)):.1%}  "
              f"FL={float(np.mean(fl_mm)):.1%}")

        # Save compressed NPZ
        print(f"\n  Saving to {output_path} ...")
        np.savez_compressed(
            output_path,
            obs=obs_mm,
            score=score_mm,
            turn=turn_mm,
            busted=busted_mm,
            fl_entry=fl_mm,
        )

        # Cleanup memmap files
        del obs_mm, score_mm, turn_mm, busted_mm, fl_mm
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        self._chunks.clear()
        return N


# ── Also generate on-policy bust/FL labels ───────────────────

def process_file_onpolicy_labels(filepath, buf, t3_mc=10, t2_mc=15, t1_mc=20):
    """Generate bust/FL labels for on-policy (best action) ALL turns.

    For each decision point, applies the BEST action (rank 0) and estimates
    bust/FL probability via MC rollout.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            result = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0

    n_added = 0

    # T0: best action
    if "all_actions" in result and result["all_actions"]:
        best = result["all_actions"][0]  # rank 0
        placements = parse_t0_placements(best["action_desc"])
        board = apply_to_board(Board(), placements)
        # T0 board (5 cards) - no MC, label as 0 (too early to predict)
        state = encode_post_action(board, turn=0)
        buf.add(state, best["ev"], 0, 0.0, 0.0)
        n_added += 1

    # T1-T4: best action per choice record
    for cr in result.get("choice_records", []):
        turn = cr["turn"]
        board = Board(top=list(cr["top"]), middle=list(cr["mid"]),
                      bottom=list(cr["bot"]))

        if not cr["actions"]:
            continue

        best = cr["actions"][0]  # rank 0
        discard, placements = parse_turn_placements(best["desc"])
        board_after = apply_to_board(board, placements)
        ncards = board_after.card_count()

        # Bust/FL labels
        if ncards == 13:  # T4: exact
            bust, fl = terminal_bust_fl(board_after)
        elif ncards == 11 and t3_mc > 0:  # T3: 1-turn MC
            known = set(board_after.top + board_after.middle
                        + board_after.bottom)
            known.add(discard)
            remaining = [c for c in ALL_CARDS if c not in known]
            bust, fl = bust_fl_mc(board_after, remaining, n_mc=t3_mc)
        elif ncards == 9 and t2_mc > 0:  # T2: 2-turn deep MC
            known = set(board_after.top + board_after.middle
                        + board_after.bottom)
            known.add(discard)
            remaining = [c for c in ALL_CARDS if c not in known]
            bust, fl = bust_fl_mc_deep(
                board_after, remaining, turns_left=2, n_mc=t2_mc)
        elif ncards == 7 and t1_mc > 0:  # T1: 3-turn deep MC
            known = set(board_after.top + board_after.middle
                        + board_after.bottom)
            known.add(discard)
            remaining = [c for c in ALL_CARDS if c not in known]
            bust, fl = bust_fl_mc_deep(
                board_after, remaining, turns_left=3, n_mc=t1_mc)
        else:
            bust = 0.0
            fl = 0.0

        state = encode_post_action(board_after, turn=turn, discards=[discard])
        buf.add(state, best["ev"], turn, bust, fl)
        n_added += 1

    return n_added


# ── Parallel worker functions ─────────────────────────────────

def _worker_onpolicy(args):
    """Worker: process one file for on-policy labels. Returns numpy arrays."""
    filepath, t3_mc, t2_mc, t1_mc, seed = args
    random.seed(seed)

    try:
        with open(filepath, encoding="utf-8") as f:
            result = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    records = []  # (state, score, turn, bust, fl)

    # T0: best action
    if "all_actions" in result and result["all_actions"]:
        best = result["all_actions"][0]
        placements = parse_t0_placements(best["action_desc"])
        board = apply_to_board(Board(), placements)
        state = encode_post_action(board, turn=0)
        records.append((state, best["ev"], 0, 0.0, 0.0))

    # T1-T4: best action per choice record
    for cr in result.get("choice_records", []):
        turn = cr["turn"]
        board = Board(top=list(cr["top"]), middle=list(cr["mid"]),
                      bottom=list(cr["bot"]))
        if not cr["actions"]:
            continue

        best = cr["actions"][0]
        discard, placements = parse_turn_placements(best["desc"])
        board_after = apply_to_board(board, placements)
        ncards = board_after.card_count()

        if ncards == 13:
            bust, fl = terminal_bust_fl(board_after)
        elif ncards == 11 and t3_mc > 0:
            known = set(board_after.top + board_after.middle
                        + board_after.bottom)
            known.add(discard)
            remaining = [c for c in ALL_CARDS if c not in known]
            bust, fl = bust_fl_mc(board_after, remaining, n_mc=t3_mc)
        elif ncards == 9 and t2_mc > 0:
            known = set(board_after.top + board_after.middle
                        + board_after.bottom)
            known.add(discard)
            remaining = [c for c in ALL_CARDS if c not in known]
            bust, fl = bust_fl_mc_deep(
                board_after, remaining, turns_left=2, n_mc=t2_mc)
        elif ncards == 7 and t1_mc > 0:
            known = set(board_after.top + board_after.middle
                        + board_after.bottom)
            known.add(discard)
            remaining = [c for c in ALL_CARDS if c not in known]
            bust, fl = bust_fl_mc_deep(
                board_after, remaining, turns_left=3, n_mc=t1_mc)
        else:
            bust = 0.0
            fl = 0.0

        state = encode_post_action(board_after, turn=turn, discards=[discard])
        records.append((state, best["ev"], turn, bust, fl))

    if not records:
        return None

    states = np.stack([r[0] for r in records]).astype(np.float16)
    scores = np.array([r[1] for r in records], dtype=np.float32)
    turns = np.array([r[2] for r in records], dtype=np.int32)
    busted = np.array([r[3] for r in records], dtype=np.float32)
    fl_entry = np.array([r[4] for r in records], dtype=np.float32)
    return states, scores, turns, busted, fl_entry


def _worker_offpolicy(args):
    """Worker: process one file for off-policy data. Returns numpy arrays."""
    filepath, bottom_frac, t3_mc, t2_mc, t1_mc, seed = args
    random.seed(seed)

    try:
        with open(filepath, encoding="utf-8") as f:
            result = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if "all_actions" not in result or not result["all_actions"]:
        return None

    records = []

    # T0 off-policy
    actions = result["all_actions"]
    n = len(actions)
    cutoff = n - int(n * bottom_frac)
    cutoff = max(cutoff, 1)

    for ar in actions[cutoff:]:
        placements = parse_t0_placements(ar["action_desc"])
        board = apply_to_board(Board(), placements)
        state = encode_post_action(board, turn=0)
        records.append((state, ar["ev"], 0, 0.0, 0.0))

    # T1-T4 off-policy
    for cr in result.get("choice_records", []):
        turn = cr["turn"]
        board = Board(top=list(cr["top"]), middle=list(cr["mid"]),
                      bottom=list(cr["bot"]))
        cr_actions = cr["actions"]
        n_act = len(cr_actions)
        act_cutoff = n_act - int(n_act * bottom_frac)
        act_cutoff = max(act_cutoff, 1)

        for ar in cr_actions[act_cutoff:]:
            discard, placements = parse_turn_placements(ar["desc"])
            board_after = apply_to_board(board, placements)
            ncards = (len(board_after.top) + len(board_after.middle)
                      + len(board_after.bottom))

            if ncards == 13:
                bust, fl = terminal_bust_fl(board_after)
            elif ncards == 11 and t3_mc > 0:
                known = set(board_after.top + board_after.middle
                            + board_after.bottom)
                known.add(discard)
                remaining = [c for c in ALL_CARDS if c not in known]
                bust, fl = bust_fl_mc(board_after, remaining, n_mc=t3_mc)
            elif ncards == 9 and t2_mc > 0:
                known = set(board_after.top + board_after.middle
                            + board_after.bottom)
                known.add(discard)
                remaining = [c for c in ALL_CARDS if c not in known]
                bust, fl = bust_fl_mc_deep(
                    board_after, remaining, turns_left=2, n_mc=t2_mc)
            elif ncards == 7 and t1_mc > 0:
                known = set(board_after.top + board_after.middle
                            + board_after.bottom)
                known.add(discard)
                remaining = [c for c in ALL_CARDS if c not in known]
                bust, fl = bust_fl_mc_deep(
                    board_after, remaining, turns_left=3, n_mc=t1_mc)
            else:
                bust = 0.0
                fl = 0.0

            state = encode_post_action(
                board_after, turn=turn, discards=[discard])
            records.append((state, ar["ev"], turn, bust, fl))

    if not records:
        return None

    states = np.stack([r[0] for r in records]).astype(np.float16)
    scores = np.array([r[1] for r in records], dtype=np.float32)
    turns = np.array([r[2] for r in records], dtype=np.int32)
    busted = np.array([r[3] for r in records], dtype=np.float32)
    fl_entry = np.array([r[4] for r in records], dtype=np.float32)
    return states, scores, turns, busted, fl_entry


# ── Main ─────────────────────────────────────────────────────

def _collect_results(results_iter, buf, total_files, start, label=""):
    """Collect worker results into buffer with progress reporting."""
    done = 0
    for result in results_iter:
        done += 1
        if result is not None:
            states, scores, turns, busted, fl_entry = result
            for j in range(len(states)):
                buf.add(states[j], scores[j], turns[j], busted[j], fl_entry[j])

        if done % 50 == 0 or done == total_files:
            elapsed = time.time() - start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total_files - done) / speed if speed > 0 else 0
            print(f"  {label}[{done:>3}/{total_files}] "
                  f"{buf.total:>8,} samples  "
                  f"{elapsed:>6.1f}s  "
                  f"ETA {eta:>5.0f}s  "
                  f"({speed:.1f} files/s)")
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Generate off-policy VN training data from Expectimax")
    parser.add_argument("input", help="Directory with Expectimax JSON files")
    parser.add_argument("--output", default="data/offpolicy_train",
                        help="Output directory")
    parser.add_argument("--bottom-frac", type=float, default=0.5,
                        help="Fraction of suboptimal actions (default: 0.5)")
    parser.add_argument("--t3-mc", type=int, default=0,
                        help="MC samples for T3 bust prob (0=skip, 5=recommended)")
    parser.add_argument("--t2-mc", type=int, default=0,
                        help="MC samples for T2 bust prob (0=skip, 10=recommended)")
    parser.add_argument("--t1-mc", type=int, default=0,
                        help="MC samples for T1 bust prob (0=skip, 20=recommended)")
    parser.add_argument("--add-onpolicy-t4", action="store_true",
                        help="Also add on-policy T4 bust/FL labels (legacy)")
    parser.add_argument("--add-onpolicy", action="store_true",
                        help="Also add on-policy bust/FL labels for ALL turns")
    parser.add_argument("--onpolicy-only", action="store_true",
                        help="Only generate on-policy labels (skip off-policy)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max JSON files to process (0=all)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    n_workers = args.workers or min(cpu_count(), 12)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    if args.max_files > 0:
        json_files = json_files[:args.max_files]

    print(f"Input: {input_path} ({len(json_files)} files)")
    print(f"Workers: {n_workers}")
    if not args.onpolicy_only:
        print(f"Bottom fraction: {args.bottom_frac}")
    print(f"MC samples: T1={args.t1_mc}, T2={args.t2_mc}, T3={args.t3_mc}")
    if args.onpolicy_only:
        print(f"Mode: on-policy only")
    elif args.add_onpolicy:
        print(f"On-policy labels: True (all turns)")
    else:
        print(f"On-policy T4 labels: {args.add_onpolicy_t4}")
    print()

    buf = RecordBuffer()
    start = time.time()

    # --- Parallel on-policy ---
    if args.add_onpolicy or args.onpolicy_only:
        t3 = args.t3_mc or 10
        t2 = args.t2_mc or 15
        t1 = args.t1_mc or 20
        tasks = [(str(jf), t3, t2, t1, args.seed + i)
                 for i, jf in enumerate(json_files)]

        print(f"=== On-policy labeling ({len(tasks)} files, {n_workers} workers) ===")
        with Pool(n_workers) as pool:
            results = pool.imap_unordered(_worker_onpolicy, tasks, chunksize=4)
            _collect_results(results, buf, len(tasks), start, "ON ")

        elapsed = time.time() - start
        print(f"  On-policy done: {buf.total:,} samples in {elapsed:.1f}s\n")

    # --- Parallel off-policy ---
    if not args.onpolicy_only:
        offpolicy_start = time.time()
        tasks = [(str(jf), args.bottom_frac, args.t3_mc, args.t2_mc,
                  args.t1_mc, args.seed + 10000 + i)
                 for i, jf in enumerate(json_files)]

        print(f"=== Off-policy generation ({len(tasks)} files, {n_workers} workers) ===")
        with Pool(n_workers) as pool:
            results = pool.imap_unordered(_worker_offpolicy, tasks, chunksize=4)
            _collect_results(results, buf, len(tasks), offpolicy_start, "OFF ")

        elapsed = time.time() - offpolicy_start
        print(f"  Off-policy done in {elapsed:.1f}s\n")

    # --- Legacy sequential on-policy T4 ---
    if args.add_onpolicy_t4 and not args.add_onpolicy and not args.onpolicy_only:
        t4_tasks = [(str(jf), 0, 0, 0, args.seed + 20000 + i)
                    for i, jf in enumerate(json_files)]
        print(f"=== On-policy T4 labels ({len(t4_tasks)} files, {n_workers} workers) ===")
        with Pool(n_workers) as pool:
            results = pool.imap_unordered(_worker_onpolicy, t4_tasks, chunksize=8)
            _collect_results(results, buf, len(t4_tasks), time.time(), "T4 ")

    if buf.total == 0:
        print("No records generated!")
        return

    elapsed = time.time() - start
    print(f"\nGenerated {buf.total:,} samples in {elapsed:.1f}s")
    print("Saving via memmap (memory-safe)...")

    npz_path = output_path / "value_train.npz"
    N = buf.save_to_npz(npz_path)

    if N > 0:
        file_size = npz_path.stat().st_size
        print(f"\n  File: {file_size / 1e9:.2f} GB")
        print(f"  Time: {time.time() - start:.1f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
