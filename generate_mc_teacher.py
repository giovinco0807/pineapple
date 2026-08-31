"""
Generate MC teacher data for VN/BC training.

Plays hands using prob_engine MC at each turn, records all candidates'
MC statistics in JSONL format.

T0: MC simulation (all ~232 candidates)
T1: MC simulation (all ~27 candidates)
T2: MC simulation (all ~15 candidates)
T3: prob_engine exact EV (remaining slots small, independence assumption OK)
T4: exhaustive evaluation (all placements scored directly)

Usage:
    python generate_mc_teacher.py --n-hands 10 --seed 42 --sims 50
    python generate_mc_teacher.py --n-hands 1000 --seed 42 --sims 50 \
        --hand-start 0 --hand-end 100 --output data/mc_teacher/shard_0.jsonl
"""
import sys
import json
import random
import argparse
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state
from ai.engine.game_engine import Hand, GameEngine
from ai.engine.action_space import get_initial_actions, get_turn_actions
from ai.prob_engine_wrapper import (
    evaluate_mc_t0, evaluate_mc_candidates, evaluate_candidates,
    evaluate_board, get_feature_vector,
)

# Global VN for --vn-policy mode
_vn_model = None
_vn_device = "cpu"
_vn_norm = {"mean": 0.0, "std": 1.0}


def match_action(placements_from_rust, actions):
    """Find matching Action object for Rust placements. Returns (action, matched)."""
    best_set = set((c, p) for c, p in placements_from_rust)

    # Direct match
    for a in actions:
        a_set = set((c, p) for c, p in a.placements)
        if a_set == best_set:
            return a, True

    # Joker fallback: Xj -> X1/X2
    joker_map = {}
    for c, p in placements_from_rust:
        if c == "Xj":
            for a in actions:
                for ac, ap in a.placements:
                    if ac.startswith("X") and ac != "Xj" and ap == p:
                        if ac not in joker_map.values():
                            joker_map[c] = ac
                            break

    if joker_map:
        mapped_set = set()
        for c, p in placements_from_rust:
            mapped_set.add((joker_map.get(c, c), p))
        for a in actions:
            a_set = set((c, p) for c, p in a.placements)
            if a_set == mapped_set:
                return a, True

    return actions[0] if actions else None, False


def get_histogram_features(top, mid, bot, exclude, turn):
    """Get 300-dim histogram features for a board state."""
    try:
        board_eval = evaluate_board(
            top=top, mid=mid, bot=bot,
            exclude=exclude, turn=turn,
        )
        feats = get_feature_vector(board_eval)
        return feats[:300]  # fine_hist only (3 x 100)
    except Exception:
        return [0.0] * 300


def vn_pick_best(obs, candidates, actions):
    """Use VN to select best action index among candidates.

    Returns the index into candidates list (not actions list).
    Falls back to MC-best (idx=0) if VN evaluation fails (e.g., Joker cards X1/X2).
    """
    import torch
    if _vn_model is None:
        return 0  # fallback to MC-best

    try:
        # Build resulting states for each candidate
        states = []
        turns = []
        for cand in candidates:
            placements = cand["placements"]
            new_top = list(obs.board_self.top)
            new_mid = list(obs.board_self.middle)
            new_bot = list(obs.board_self.bottom)
            new_discards = list(obs.known_discards_self)

            for card, pos in placements:
                if pos == "top":
                    new_top.append(card)
                elif pos in ("mid", "middle"):
                    new_mid.append(card)
                elif pos in ("bot", "bottom"):
                    new_bot.append(card)
            if "discard" in cand and cand["discard"]:
                new_discards.append(cand["discard"])

            new_board = Board(top=new_top, middle=new_mid, bottom=new_bot)
            new_obs = Observation(
                board_self=new_board,
                board_opponent=obs.board_opponent,
                dealt_cards=[],
                known_discards_self=new_discards,
                turn=obs.turn,
                is_btn=obs.is_btn,
            )
            state = encode_state(new_obs)
            states.append(state)
            turns.append(obs.turn)

        with torch.no_grad():
            state_t = torch.tensor(np.array(states), dtype=torch.float32).to(_vn_device)
            turn_t = torch.tensor(turns, dtype=torch.long).to(_vn_device)
            out = _vn_model(state_t, turn_t)
            values = out["value"].squeeze(-1).cpu().numpy()
            bust_probs = out["bust_prob"].squeeze(-1).cpu().numpy()
            fl_probs = out["fl_prob"].squeeze(-1).cpu().numpy()

        # Denormalize and compute adjusted score
        scores = values * _vn_norm["std"] + _vn_norm["mean"]
        adjusted = scores * (1 - bust_probs) + (-8.0) * bust_probs + 10.0 * fl_probs
        return int(np.argmax(adjusted))
    except Exception:
        return 0  # fallback to MC-best (e.g., Joker X1/X2 not encodable)



def generate_hand(deck, hand_id, mc_sims=50, mc_turns=(0, 1, 2)):
    """Play one hand, record all candidates at each turn.

    Args:
        deck: Shuffled 53-card deck
        hand_id: Unique hand identifier
        mc_sims: MC simulations per candidate for MC turns
        mc_turns: Which turns use MC (default: T0, T1, T2). Others use exact.

    Returns:
        list of turn records (dicts)
    """
    hand = Hand(deck=list(deck), btn=0)
    hero = 0
    opp = 1
    records = []

    # ---- T0 ----
    hero_obs = hand.get_observation(hero)
    dealt_hero = list(hero_obs.dealt_cards)
    opp_obs = hand.get_observation(opp)

    exclude = []  # BTN acts first, no opponent info yet

    t0_start = time.time()
    if 0 in mc_turns:
        result = evaluate_mc_t0(dealt=dealt_hero, exclude=exclude, sims=mc_sims)
        eval_mode = f"mc{mc_sims}"
    else:
        # Fallback: not typically used since T0 should always be MC
        result = {"candidates": []}
        eval_mode = "none"

    candidates = result.get("candidates", [])
    t0_elapsed = time.time() - t0_start

    if not candidates:
        return records

    # Record T0
    records.append({
        "hand_id": hand_id,
        "turn": 0,
        "board": {"top": [], "mid": [], "bot": []},
        "dealt": dealt_hero,
        "exclude": exclude,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": 0,
        "eval_mode": eval_mode,
        "elapsed_s": round(t0_elapsed, 2),
    })

    # Apply T0
    hero_actions = get_initial_actions(hero_obs.dealt_cards, hero_obs.board_self)
    opp_actions = get_initial_actions(opp_obs.dealt_cards, opp_obs.board_self)

    # VN-guided: let VN pick the best action
    if _vn_model is not None:
        vn_best_idx = vn_pick_best(hero_obs, candidates, hero_actions)
        records[-1]["vn_best_idx"] = vn_best_idx
    else:
        vn_best_idx = 0  # MC-best

    best_t0 = candidates[vn_best_idx]
    hero_action, matched = match_action(best_t0["placements"], hero_actions)
    if not matched:
        records[-1]["match_warning"] = True

    hand.apply_action(hero, hero_action)
    hand.apply_action(opp, opp_actions[0])

    # Record histogram after T0 placement
    h_obs = hand.get_observation(hero)
    top_after = list(h_obs.board_self.top)
    mid_after = list(h_obs.board_self.middle)
    bot_after = list(h_obs.board_self.bottom)
    exclude_after = list(h_obs.board_opponent.top) + list(h_obs.board_opponent.middle) + list(h_obs.board_opponent.bottom)
    records[-1]["board_after"] = {"top": top_after, "mid": mid_after, "bot": bot_after}
    records[-1]["hist_after"] = get_histogram_features(top_after, mid_after, bot_after, exclude_after, 1)

    # ---- T1-T4 ----
    for turn_num in range(1, 5):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()

        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue

            if seat != hero:
                obs = hand.get_observation(seat)
                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                if actions:
                    hand.apply_action(seat, actions[0])
                continue

            # Hero turn
            obs = hand.get_observation(seat)
            top = list(obs.board_self.top)
            mid = list(obs.board_self.middle)
            bot = list(obs.board_self.bottom)
            dealt = list(obs.dealt_cards)
            cc = obs.board_self.card_count()

            exclude = []
            exclude.extend(obs.board_opponent.top)
            exclude.extend(obs.board_opponent.middle)
            exclude.extend(obs.board_opponent.bottom)
            exclude.extend(obs.known_discards_self)

            t_start = time.time()

            if turn_num in mc_turns and cc < 11:
                # MC evaluation
                result = evaluate_mc_candidates(
                    top=top, mid=mid, bot=bot, dealt=dealt,
                    exclude=exclude, turn=turn_num, sims=mc_sims,
                )
                eval_mode = f"mc{mc_sims}"
            else:
                # Exact evaluation (T3/T4 or cc==11)
                result = evaluate_candidates(
                    top=top, mid=mid, bot=bot, dealt=dealt,
                    exclude=exclude, turn=turn_num, position="bb",
                )
                eval_mode = "exact"

            candidates = result.get("candidates", [])
            t_elapsed = time.time() - t_start

            if not candidates:
                continue

            # Record this turn
            record = {
                "hand_id": hand_id,
                "turn": turn_num,
                "board": {"top": top, "mid": mid, "bot": bot},
                "dealt": dealt,
                "exclude": exclude,
                "n_candidates": len(candidates),
                "candidates": candidates,
                "best_idx": 0,
                "eval_mode": eval_mode,
                "elapsed_s": round(t_elapsed, 2),
            }

            # Apply best action (VN-guided or MC-best)
            if _vn_model is not None:
                vn_idx = vn_pick_best(obs, candidates, get_turn_actions(obs.dealt_cards, obs.board_self))
                record["vn_best_idx"] = vn_idx
            else:
                vn_idx = 0
            best = candidates[vn_idx]
            best_placements = [(c, p) for c, p in best["placements"]]
            actions = get_turn_actions(obs.dealt_cards, obs.board_self)
            action, matched = match_action(best_placements, actions)
            if not matched:
                record["match_warning"] = True
            if action:
                hand.apply_action(hero, action)

            # Record histogram after placement
            h_obs = hand.get_observation(hero)
            top_after = list(h_obs.board_self.top)
            mid_after = list(h_obs.board_self.middle)
            bot_after = list(h_obs.board_self.bottom)
            exc_after = list(h_obs.board_opponent.top) + list(h_obs.board_opponent.middle) + list(h_obs.board_opponent.bottom)
            exc_after.extend(h_obs.known_discards_self)
            record["board_after"] = {"top": top_after, "mid": mid_after, "bot": bot_after}
            if cc < 11:  # Don't compute histogram for completed boards
                record["hist_after"] = get_histogram_features(
                    top_after, mid_after, bot_after, exc_after, turn_num + 1)

            records.append(record)

    # Final result
    game_result = GameEngine.compute_result(hand)
    hero_board = hand.boards[hero]
    final = {
        "hand_id": hand_id,
        "turn": -1,  # sentinel for final result
        "final_board": {
            "top": list(hero_board.top),
            "mid": list(hero_board.middle),
            "bot": list(hero_board.bottom),
        },
        "busted": game_result.busted[hero],
        "fl_entry": game_result.fl_entry[hero],
        "royalty": game_result.royalties[hero]["total"],
        "score": game_result.raw_score[hero],
    }
    records.append(final)

    return records


def main():
    global _vn_model, _vn_device, _vn_norm

    parser = argparse.ArgumentParser(description="Generate MC teacher data")
    parser.add_argument("--n-hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sims", type=int, default=50, help="MC sims per candidate")
    parser.add_argument("--hand-start", type=int, default=None,
                        help="Start index for sharding (inclusive)")
    parser.add_argument("--hand-end", type=int, default=None,
                        help="End index for sharding (exclusive)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: stdout summary)")
    parser.add_argument("--mc-turns", type=str, default="0,1,2",
                        help="Comma-separated turns to use MC (default: 0,1,2)")
    parser.add_argument("--vn-policy", type=str, default=None,
                        help="Path to VN model for action selection (hybrid mode)")
    parser.add_argument("--vn-norm", type=str, default=None,
                        help="Path to VN norm_stats.json")
    args = parser.parse_args()

    mc_turns = set(int(t) for t in args.mc_turns.split(","))

    # Load VN for hybrid mode
    if args.vn_policy:
        import torch
        from ai.models.networks import ValueNetworkV3
        _vn_device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(args.vn_policy, map_location=_vn_device, weights_only=False)
        sd = ck.get("model_state_dict", ck)
        # Detect input dim from checkpoint
        in_dim = sd["shared.0.weight"].shape[1]
        _vn_model = ValueNetworkV3(input_dim=in_dim).to(_vn_device)
        _vn_model.load_state_dict(sd)
        _vn_model.eval()
        if args.vn_norm and Path(args.vn_norm).exists():
            ns = json.load(open(args.vn_norm))
            _vn_norm = {"mean": ns.get("mean", ns.get("score_mean", 0)), "std": ns.get("std", ns.get("score_std", 1))}
        print(f"  VN policy: {args.vn_policy} (dim={in_dim}, {_vn_device})", file=sys.stderr)

    # Generate all decks for reproducibility (same as eval_mcts.py sharding)
    rng = random.Random(args.seed)
    all_decks = []
    for _ in range(args.n_hands):
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        all_decks.append(deck)

    # Apply sharding
    start = args.hand_start if args.hand_start is not None else 0
    end = args.hand_end if args.hand_end is not None else args.n_hands
    decks = all_decks[start:end]

    outfile = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        outfile = open(args.output, "w")

    mode_str = "HYBRID (VN play + MC label)" if _vn_model else "MC Teacher"
    print(f"=== {mode_str} ===", file=sys.stderr)
    print(f"  Hands: {len(decks)} (global {start}-{end} of {args.n_hands})", file=sys.stderr)
    print(f"  Seed: {args.seed}, Sims: {args.sims}", file=sys.stderr)
    print(f"  MC turns: {sorted(mc_turns)}", file=sys.stderr)
    print(f"  Output: {args.output or 'stdout'}", file=sys.stderr)

    total_start = time.time()
    stats = {"hands": 0, "busted": 0, "fl": 0, "scores": [], "turns_recorded": 0}

    for i, deck in enumerate(decks):
        hand_id = start + i
        t_hand = time.time()

        records = generate_hand(deck, hand_id, mc_sims=args.sims, mc_turns=mc_turns)

        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            if outfile:
                outfile.write(line + "\n")

        # Stats from final record
        final = records[-1] if records and records[-1].get("turn") == -1 else None
        if final:
            stats["hands"] += 1
            if final["busted"]:
                stats["busted"] += 1
            if final["fl_entry"]:
                stats["fl"] += 1
            stats["scores"].append(final["score"])
            stats["turns_recorded"] += len(records) - 1  # exclude final

        hand_time = time.time() - t_hand
        if (i + 1) % max(1, len(decks) // 10) == 0 or i == 0:
            n = stats["hands"]
            avg_score = sum(stats["scores"]) / n if n else 0
            bust_pct = stats["busted"] / n * 100 if n else 0
            fl_pct = stats["fl"] / n * 100 if n else 0
            elapsed = time.time() - total_start
            print(f"  [{i+1:4d}/{len(decks)}]  {elapsed:6.0f}s  "
                  f"Score={avg_score:+.2f}  Bust={bust_pct:.1f}%  FL={fl_pct:.1f}%  "
                  f"({hand_time:.1f}s/hand)", file=sys.stderr)

    total_time = time.time() - total_start
    n = stats["hands"]

    print(f"\n=== Done ({total_time:.0f}s) ===", file=sys.stderr)
    if n > 0:
        import numpy as np
        s = np.array(stats["scores"])
        print(f"  Hands: {n}", file=sys.stderr)
        print(f"  Score: {s.mean():+.2f} +/- {s.std():.2f}", file=sys.stderr)
        print(f"  Bust:  {stats['busted']/n*100:.1f}%", file=sys.stderr)
        print(f"  FL:    {stats['fl']/n*100:.1f}%", file=sys.stderr)
        print(f"  Turns recorded: {stats['turns_recorded']}", file=sys.stderr)
        print(f"  Speed: {total_time/n:.1f}s/hand", file=sys.stderr)

    if outfile:
        outfile.close()


if __name__ == "__main__":
    main()
