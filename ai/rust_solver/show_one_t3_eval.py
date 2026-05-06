import os
import sys
import json
import subprocess
from pathlib import Path

# Add paths for encoding and generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ai.heuristic_bot.t3_generator import generate_random_t3_state

def main():
    exe_path = str(Path(__file__).resolve().parent / "target" / "release" / "t3_exact.exe")
    if not os.path.exists(exe_path):
        print(f"Error: Could not find t3_exact.exe at {exe_path}")
        sys.exit(1)
        
    print("Using hardcoded T3 state from previous run...")
    state = {
        "board_top": ["Kh", "X2"],
        "board_mid": ["7c", "2d"],
        "board_bot": ["Qc", "8h", "Th", "Ts", "8c"],
        "discards": ["9h", "9s"],
        "dealt": ["Qd", "8s", "3h"]
    }
            
    print("\n" + "="*50)
    print(" [ GENERATED T3 STATE ]")
    print("="*50)
    print(f" Top ({len(state['board_top'])}): {' '.join(state['board_top'])}")
    print(f" Mid ({len(state['board_mid'])}): {' '.join(state['board_mid'])}")
    print(f" Bot ({len(state['board_bot'])}): {' '.join(state['board_bot'])}")
    print("-" * 50)
    print(f" Dealt Cards: {' '.join(state['dealt'])}")
    print(f" Discards   : {' '.join(state['discards'])}")
    print("="*50 + "\n")
    
    print("Calculating Exact Expectimax EV for all actions (1.6 Million nodes)...")
    
    # Run Rust solver
    proc = subprocess.Popen(
        [exe_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    req = {
        "board_top": state["board_top"],
        "board_mid": state["board_mid"],
        "board_bot": state["board_bot"],
        "discards": state["discards"],
        "dealt": state["dealt"],
        "bust_penalty": -4.0,
        "fl_ev": {"14": 24.92, "15": 31.43, "16": 39.77, "17": 47.74}
    }
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.close()
    
    output = proc.stdout.readline().strip()
    proc.wait()
    
    if not output:
        print("Error: No output from solver.")
        sys.exit(1)
        
    res = json.loads(output)
    if "error" in res:
        print(f"Solver Error: {res['error']}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print(f" [ EXACT EXPECTIMAX RESULTS ] (Took {res['elapsed_ms']} ms)")
    print("="*50)
    
    actions = res["actions"]
    print(f" Evaluated {len(actions)} valid placements.")
    print(f" Best Action: {actions[0]['action_desc']} => EV: {actions[0]['ev']:.3f}\n")
    
    print(" [Rank] | [Action Description]                | [Exact EV]")
    print("--------|-------------------------------------|------------")
    for idx, a in enumerate(actions):
        rank = str(idx + 1).rjust(5)
        desc = a['action_desc'].ljust(35)
        ev = f"{a['ev']:.4f}".rjust(10)
        
        # Highlight top 3
        if idx < 3:
            print(f" {rank}  | {desc} | {ev}  <--")
        else:
            print(f" {rank}  | {desc} | {ev}")
            
    print("="*50)

if __name__ == "__main__":
    main()
