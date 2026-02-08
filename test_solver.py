import subprocess, json

# Test: cards that include both wheel (A2345) and K-high straight (KQJT9)
# + joker + 66 for top
cards = [
    {"rank": 14, "suit": 0},  # Ah
    {"rank": 5, "suit": 1},   # 5d
    {"rank": 4, "suit": 3},   # 4s
    {"rank": 2, "suit": 1},   # 2d
    {"rank": 3, "suit": 3},   # 3s
    {"rank": 13, "suit": 1},  # Kd
    {"rank": 12, "suit": 0},  # Qh
    {"rank": 11, "suit": 2},  # Jc
    {"rank": 10, "suit": 0},  # Th
    {"rank": 9, "suit": 1},   # 9d
    {"rank": 6, "suit": 1},   # 6d
    {"rank": 6, "suit": 2},   # 6c
    {"rank": 0, "suit": 4},   # Joker (X1)
    {"rank": 8, "suit": 1},   # 8d
]

request = json.dumps({"cards": cards, "version": 2})
result = subprocess.run(
    [r"ai\rust_solver\target\release\fl_solver.exe"],
    input=request, capture_output=True, text=True, timeout=10
)
for line in result.stdout.split('\n'):
    if line.startswith('{'):
        resp = json.loads(line)
        if resp.get("success"):
            p = resp["placement"]
            rank_rev = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A',0:'JK'}
            suit_rev = {0:'h',1:'d',2:'c',3:'s',4:'*'}
            def fmt(cards_list):
                return [f"{rank_rev.get(c['rank'],'?')}{suit_rev.get(c['suit'],'?')}" for c in cards_list]
            print(f"Top:    {fmt(p['top'])}")
            print(f"Middle: {fmt(p['middle'])}")
            print(f"Bottom: {fmt(p['bottom'])}")
            print(f"Score:  {p.get('score')}")
            print(f"FL Stay: {p.get('can_stay')}")
            print(f"Royalties: {p.get('royalties')}")
        else:
            print(f"Error: {resp.get('error')}")
    elif line.strip():
        print(line)
