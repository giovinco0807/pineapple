import sys
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.engine.encoding import Board, Observation, encode_state, RANK_VALUES
from ai.engine.scoring import _evaluate_with_joker_constraint
from ai.engine.game_engine import evaluate_hand

def is_busted(board_dict):
    """Determine if a completed board is busted according to OFC rules."""
    bot_val = evaluate_hand(board_dict["bottom"], 5)
    mid_val = _evaluate_with_joker_constraint(board_dict["middle"], 5, max_value=bot_val)
    top_val = _evaluate_with_joker_constraint(board_dict["top"], 3, max_value=mid_val)
    return top_val > mid_val or mid_val > bot_val

def check_fl_potential(top_cards):
    """Check if top row cards have FL entry potential (QQ+)."""
    if not top_cards:
        return False
    from collections import Counter
    ranks = []
    jokers = 0
    for c in top_cards:
        if c in ('X1', 'X2'):
            jokers += 1
        elif len(c) >= 2:
            r = c[:-1]
            if r in RANK_VALUES:
                ranks.append(RANK_VALUES[r])

    rank_counts = Counter(ranks)
    for r, cnt in rank_counts.items():
        if cnt + jokers >= 2 and r >= RANK_VALUES['Q']:
            return True
    for r, cnt in rank_counts.items():
        if cnt + jokers >= 3:
            return True
    return False

def load_game_states(jsonl_path):
    """Load btn boards and discards from t4_game_states.jsonl keyed by sample_id."""
    states = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            states[data['sample_id']] = data
    return states

def convert_t4_solutions(solutions_path, states_data, top_k=0, min_ev=None, max_samples=0):
    all_obs = []
    all_scores = []
    all_fl = []
    all_busted = []
    
    n_hands = 0
    with open(solutions_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            sol = json.loads(line)
            sample_id = sol['sample_id']
            
            if sample_id not in states_data:
                print(f"WARN: Sample ID {sample_id} not found in game states!")
                continue
            
            if max_samples > 0 and n_hands >= max_samples:
                break

            state = states_data[sample_id]
            btn_dict = state['btn']
            btn_board = Board(
                top=btn_dict.get('top', []),
                middle=btn_dict.get('middle', []),
                bottom=btn_dict.get('bottom', [])
            )
            
            # Known discards before this turn
            bb_discards = state['bb'].get('discards', [])
            
            placements = sol['placements']
            # Sort by EV descending just in case
            placements.sort(key=lambda x: x['ev'], reverse=True)
            
            if top_k > 0:
                placements = placements[:top_k]
            
            if min_ev is not None:
                placements = [p for p in placements if p['ev'] >= min_ev]
            
            for p in placements:
                cards_placed = p['cards_placed']
                slots = p['slots']
                ev = p['ev']
                
                # Construct the new board
                bb_top = list(sol['bb_top'])
                bb_middle = list(sol['bb_middle'])
                bb_bottom = list(sol['bb_bottom'])
                
                for c, slot in zip(cards_placed, slots):
                    if slot == "top": bb_top.append(c)
                    elif slot == "mid": bb_middle.append(c)
                    elif slot == "bot": bb_bottom.append(c)
                
                board_self = Board(top=bb_top, middle=bb_middle, bottom=bb_bottom)
                
                # Figure out the single discarded card for this turn
                drawn_set = set(sol['bb_drawn'])
                placed_set = set(cards_placed)
                discarded_this_turn = list(drawn_set - placed_set)
                
                all_discards = list(set(bb_discards) | set(discarded_this_turn))
                
                obs = Observation(
                    board_self=board_self,
                    board_opponent=btn_board,
                    dealt_cards=[],  # Already placed
                    known_discards_self=all_discards,
                    turn=4,
                    is_btn=False,
                    is_fl=False,
                    opp_is_fl=False,
                )
                
                obs_vec = encode_state(obs)
                
                fl_entry = check_fl_potential(bb_top)
                busted = is_busted({"top": bb_top, "middle": bb_middle, "bottom": bb_bottom})
                
                all_obs.append(obs_vec)
                all_scores.append(ev)
                all_fl.append(fl_entry)
                all_busted.append(busted)
            
            n_hands += 1

    return all_obs, all_scores, all_fl, all_busted, n_hands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', required=True, help='t4_game_states.jsonl path')
    parser.add_argument('--solutions', required=True, help='t4_exact_solutions.jsonl path')
    parser.add_argument('--output', required=True, help='Output NPZ path')
    parser.add_argument('--top-k', type=int, default=0, help='Top K placements per state')
    parser.add_argument('--min-ev', type=float, default=None)
    parser.add_argument('--max-hands', type=int, default=0)
    args = parser.parse_args()

    print(f"Loading game states from {args.states}...")
    states_data = load_game_states(args.states)
    print(f"Loaded {len(states_data)} states.")

    print(f"Converting solutions from {args.solutions}...")
    all_obs, all_scores, all_fl, all_busted, n_hands = convert_t4_solutions(
        args.solutions, states_data, top_k=args.top_k, min_ev=args.min_ev, max_samples=args.max_hands
    )

    if not all_obs:
        print("ERROR: No data collected!")
        sys.exit(1)

    obs_array = np.array(all_obs, dtype=np.float32)
    score_array = np.array(all_scores, dtype=np.float32)
    turn_array = np.full(len(all_obs), 4, dtype=np.int8)  # Turn 4
    busted_array = np.array(all_busted, dtype=np.bool_)
    fl_array = np.array(all_fl, dtype=np.bool_)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=obs_array,
        score=score_array,
        turn=turn_array,
        busted=busted_array,
        fl_entry=fl_array,
    )

    print(f"Conversion Complete!")
    print(f"Hands processed: {n_hands}")
    print(f"Total samples: {len(obs_array)}")
    print(f"Avg EV: {score_array.mean():.2f}")
    print(f"Busted: {busted_array.sum()} ({busted_array.mean()*100:.1f}%)")
    print(f"FL Entry: {fl_array.sum()} ({fl_array.mean()*100:.1f}%)")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
