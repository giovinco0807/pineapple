"""
Analyze P99 worst-case samples to identify root causes of prediction failures.
"""
import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

def has_joker(cards):
    return any(c.startswith("X") for c in cards)

def count_jokers_in_board(board):
    all_cards = board.get("top", []) + board.get("middle", []) + board.get("bottom", [])
    return sum(1 for c in all_cards if c.startswith("X"))

def count_jokers_in_dealt(dealt):
    return sum(1 for c in dealt if c.startswith("X"))

def board_card_count(board):
    return len(board.get("top", [])) + len(board.get("middle", [])) + len(board.get("bottom", []))

def row_completeness(board):
    top = len(board.get("top", []))
    mid = len(board.get("middle", []))
    bot = len(board.get("bottom", []))
    return {"top": f"{top}/3", "mid": f"{mid}/5", "bot": f"{bot}/5", "total": top + mid + bot}

def identify_hand_type(cards, row_name):
    """Rough hand type from cards."""
    if not cards:
        return "empty"
    real = [c for c in cards if not c.startswith("X")]
    if not real:
        return "joker_only"
    ranks = [c[0] for c in real]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    
    if row_name == "top":
        if len(counts) >= 1 and counts[0] >= 3:
            return "trips"
        elif len(counts) >= 1 and counts[0] >= 2:
            return "pair"
        else:
            return "high_card"
    else:
        if len(counts) >= 1 and counts[0] >= 4:
            return "quads"
        elif len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 2:
            return "full_house"
        elif len(counts) >= 1 and counts[0] >= 3:
            return "trips"
        elif len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
            return "two_pair"
        elif len(counts) >= 1 and counts[0] >= 2:
            return "pair"
        else:
            return "high_card"

def describe_action_semantics(idx):
    """Decode semantic action index [0,26]."""
    discard_slot = idx // 9
    rem = idx % 9
    pos0 = rem // 3
    pos1 = rem % 3
    positions = ["T", "M", "B"]
    return f"d:{discard_slot} -> ({positions[pos0]},{positions[pos1]})"

def main():
    parser = argparse.ArgumentParser(description="Analyze T3 P99 worst-case samples")
    parser.add_argument(
        "p99_file",
        nargs="?",
        default=str(Path(__file__).parent / "data" / "t3_oracle" / "t3_p99_samples.jsonl"),
    )
    args = parser.parse_args()
    p99_file = Path(args.p99_file)
    
    samples = []
    with open(p99_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"=== P99 Analysis: {len(samples)} samples ===\n")
    
    # ---------- Basic stats ----------
    regrets = [s["regret"] for s in samples]
    best_evs = [s["best_ev"] for s in samples]
    pred_evs = [s["pred_ev"] for s in samples]
    value_preds = [s["value_pred"] for s in samples]
    
    print(f"Regret:    min={min(regrets):.1f}  max={max(regrets):.1f}  mean={sum(regrets)/len(regrets):.1f}")
    print(f"Best EV:   min={min(best_evs):.1f}  max={max(best_evs):.1f}  mean={sum(best_evs)/len(best_evs):.1f}")
    print(f"Pred EV:   min={min(pred_evs):.1f}  max={max(pred_evs):.1f}  mean={sum(pred_evs)/len(pred_evs):.1f}")
    print(f"Value pred: min={min(value_preds):.1f}  max={max(value_preds):.1f}  mean={sum(value_preds)/len(value_preds):.1f}")
    print()
    
    # ---------- Joker Analysis ----------
    joker_board_count = 0
    joker_dealt_count = 0
    joker_any_count = 0
    for s in samples:
        board = s["state"]["board"]
        dealt = s["state"]["dealt"]
        jb = count_jokers_in_board(board)
        jd = count_jokers_in_dealt(dealt)
        if jb > 0:
            joker_board_count += 1
        if jd > 0:
            joker_dealt_count += 1
        if jb > 0 or jd > 0:
            joker_any_count += 1
    
    print(f"--- Joker Prevalence ---")
    print(f"  Joker in board:  {joker_board_count}/{len(samples)} ({100*joker_board_count/len(samples):.1f}%)")
    print(f"  Joker in dealt:  {joker_dealt_count}/{len(samples)} ({100*joker_dealt_count/len(samples):.1f}%)")
    print(f"  Joker anywhere:  {joker_any_count}/{len(samples)} ({100*joker_any_count/len(samples):.1f}%)")
    print()
    
    # ---------- FL Status ----------
    fl_self = sum(1 for s in samples if s["state"]["meta"]["is_fl"] == 1.0)
    fl_opp = sum(1 for s in samples if s["state"]["meta"]["opp_is_fl"] == 1.0)
    both_fl = sum(1 for s in samples if s["state"]["meta"]["is_fl"] == 1.0 and s["state"]["meta"]["opp_is_fl"] == 1.0)
    print(f"--- Fantasyland Status ---")
    print(f"  Self in FL:  {fl_self}/{len(samples)} ({100*fl_self/len(samples):.1f}%)")
    print(f"  Opp in FL:   {fl_opp}/{len(samples)} ({100*fl_opp/len(samples):.1f}%)")
    print(f"  Both FL:     {both_fl}/{len(samples)} ({100*both_fl/len(samples):.1f}%)")
    print()
    
    # ---------- Action Index Analysis ----------
    true_actions = Counter(s["true_action_idx"] for s in samples)
    pred_actions = Counter(s["pred_action_idx"] for s in samples)
    print(f"--- True Action Distribution (top 10) ---")
    for idx, cnt in true_actions.most_common(10):
        print(f"  action {idx:2d} [{describe_action_semantics(idx)}]: {cnt}")
    print()
    print(f"--- Pred Action Distribution (top 10) ---")
    for idx, cnt in pred_actions.most_common(10):
        print(f"  action {idx:2d} [{describe_action_semantics(idx)}]: {cnt}")
    print()
    
    # ---------- Confusion: where does the model go wrong? ----------
    # True action is in pred top-3?
    true_in_top1 = 0
    true_in_top2 = 0
    true_in_top3 = 0
    for s in samples:
        pred_top3_idxs = [p["action_idx"] for p in s["pred_top3_by_logit"]]
        true_idx = s["true_action_idx"]
        if pred_top3_idxs[0] == true_idx:
            true_in_top1 += 1
        if true_idx in pred_top3_idxs[:2]:
            true_in_top2 += 1
        if true_idx in pred_top3_idxs:
            true_in_top3 += 1
    
    print(f"--- True Action Recovery ---")
    print(f"  True in pred top-1:  {true_in_top1}/{len(samples)} ({100*true_in_top1/len(samples):.1f}%)")
    print(f"  True in pred top-2:  {true_in_top2}/{len(samples)} ({100*true_in_top2/len(samples):.1f}%)")
    print(f"  True in pred top-3:  {true_in_top3}/{len(samples)} ({100*true_in_top3/len(samples):.1f}%)")
    print()
    
    # ---------- Logit gap analysis ----------
    # When wrong: how close was correct action in logit space?
    logit_gaps = []
    for s in samples:
        if s["true_action_idx"] == s["pred_action_idx"]:
            continue
        pred_top3 = s["pred_top3_by_logit"]
        best_logit = pred_top3[0]["logit"]
        true_logit = None
        for p in pred_top3:
            if p["action_idx"] == s["true_action_idx"]:
                true_logit = p["logit"]
                break
        if true_logit is not None:
            logit_gaps.append(best_logit - true_logit)
    
    if logit_gaps:
        print(f"--- Logit Gap (best_pred - true, when wrong & true in top-3) ---")
        print(f"  count: {len(logit_gaps)}")
        print(f"  mean gap: {sum(logit_gaps)/len(logit_gaps):.3f}")
        print(f"  min gap:  {min(logit_gaps):.3f}")
        print(f"  max gap:  {max(logit_gaps):.3f}")
        # Histogram
        bins = [0, 0.5, 1.0, 2.0, 5.0, 100.0]
        for i in range(len(bins)-1):
            cnt = sum(1 for g in logit_gaps if bins[i] <= g < bins[i+1])
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {cnt}")
        print()
    
    # ---------- Board shape analysis ----------
    total_cards_dist = Counter()
    for s in samples:
        board = s["state"]["board"]
        tc = board_card_count(board)
        total_cards_dist[tc] += 1
    print(f"--- Board Card Count Distribution ---")
    for tc in sorted(total_cards_dist.keys()):
        print(f"  {tc} cards: {total_cards_dist[tc]}")
    print()
    
    # ---------- Row hand types ----------
    top_types = Counter()
    mid_types = Counter()
    bot_types = Counter()
    for s in samples:
        board = s["state"]["board"]
        top_types[identify_hand_type(board["top"], "top")] += 1
        mid_types[identify_hand_type(board["middle"], "mid")] += 1
        bot_types[identify_hand_type(board["bottom"], "bot")] += 1
    
    print(f"--- Row Hand Types ---")
    print(f"  Top:    {dict(top_types.most_common())}")
    print(f"  Middle: {dict(mid_types.most_common())}")
    print(f"  Bottom: {dict(bot_types.most_common())}")
    print()
    
    # ---------- Valid action count ----------
    vac = Counter(s["valid_action_count"] for s in samples)
    print(f"--- Valid Action Count ---")
    for v in sorted(vac.keys()):
        print(f"  {v} valid: {vac[v]}")
    print()
    
    # ---------- Confusion pattern: pred vs true action semantics ----------
    confusion = Counter()
    for s in samples:
        if s["true_action_idx"] != s["pred_action_idx"]:
            true_sem = describe_action_semantics(s["true_action_idx"])
            pred_sem = describe_action_semantics(s["pred_action_idx"])
            # Extract discard and placement patterns
            true_d = s["true_action_idx"] // 9
            pred_d = s["pred_action_idx"] // 9
            if true_d == pred_d:
                confusion["same_discard_wrong_placement"] += 1
            else:
                confusion["wrong_discard"] += 1
    
    print(f"--- Confusion Patterns ---")
    for k, v in confusion.most_common():
        print(f"  {k}: {v}")
    print()
    
    # ---------- EV spread analysis ----------
    # How many samples have extremely concentrated EV (all actions ≈ same EV)?
    ev_spread_narrow = 0  # best - second < 10
    ev_spread_wide = 0    # best - second > 50
    for s in samples:
        top3 = s["true_top3_by_ev"]
        if len(top3) >= 2:
            gap = top3[0]["ev"] - top3[1]["ev"]
            if gap < 10:
                ev_spread_narrow += 1
            elif gap > 50:
                ev_spread_wide += 1
    print(f"--- EV Gap (best - 2nd best) ---")
    print(f"  Narrow (<10): {ev_spread_narrow}")
    print(f"  Wide (>50):   {ev_spread_wide}")
    print()
    
    # ---------- Worst 10 samples detailed ----------
    print(f"=== Top 10 Worst Samples ===\n")
    for s in samples[:10]:
        board = s["state"]["board"]
        dealt = s["state"]["dealt"]
        meta = s["state"]["meta"]
        
        joker_str = ""
        board_jokers = count_jokers_in_board(board)
        dealt_jokers = count_jokers_in_dealt(dealt)
        if board_jokers or dealt_jokers:
            joker_str = f" [JOKER: board={board_jokers}, dealt={dealt_jokers}]"
        
        rc = row_completeness(board)
        print(f"--- Rank {s['tail_rank']} | regret={s['regret']:.1f} | best_ev={s['best_ev']:.1f} | pred_ev={s['pred_ev']:.1f}{joker_str} ---")
        print(f"  Board: top={board['top']}({rc['top']}) mid={board['middle']}({rc['mid']}) bot={board['bottom']}({rc['bot']})")
        print(f"  Dealt: {dealt}")
        print(f"  FL: self={meta['is_fl']} opp={meta['opp_is_fl']}")
        print(f"  True action idx={s['true_action_idx']} [{describe_action_semantics(s['true_action_idx'])}]")
        print(f"  Pred action idx={s['pred_action_idx']} [{describe_action_semantics(s['pred_action_idx'])}]")
        true_top3_str = [(t['action_idx'], round(t['ev'], 1)) for t in s['true_top3_by_ev']]
        pred_top3_str = [(t['action_idx'], round(t['logit'], 2), round(t['ev'], 1)) for t in s['pred_top3_by_logit']]
        print(f"  True top3 EVs:  {true_top3_str}")
        print(f"  Pred top3:      {pred_top3_str}")
        print(f"  Valid actions: {s['valid_action_count']}")
        print()


if __name__ == "__main__":
    main()
