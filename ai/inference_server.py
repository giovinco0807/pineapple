import socket
import json
import threading
import traceback
import torch
import numpy as np
import os

from ai.models.t1_network import T1PlacementNet, CARD_DIM, encode_card_str, MAX_CARDS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models (currently just t1_bb, but extensible)
models = {}

def load_models():
    print(f"Loading models on {DEVICE}...")
    import glob
    model_paths = glob.glob('ai/models/t1_placement_net_*.pt')
    for model_path in model_paths:
        # Extract model name (e.g. 't1_bb' from 'ai/models/t1_placement_net_t1_bb.pt')
        # Wait, the old file was 't1_placement_net_bb.pt'.
        # Let's standardize: we will save as 't1_placement_net_t1_bb.pt' in the new script.
        # But for backward compatibility we also check the old name.
        basename = os.path.basename(model_path)
        if basename == 't1_placement_net_bb.pt':
            model_name = 't1_bb'
        else:
            model_name = basename.replace('t1_placement_net_', '').replace('.pt', '')
            
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model = T1PlacementNet(128, 4, 4, 256, 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            models[model_name] = model
            print(f"Loaded {model_name} model (Epoch {checkpoint.get('epoch', 0)})")
        except Exception as e:
            print(f"Warning: Could not load {model_name} from {model_path}: {e}")

def parse_board(board_str):
    import re
    top_match = re.search(r'Top\[(.*?)\]', board_str)
    mid_match = re.search(r'Mid\[(.*?)\]', board_str)
    bot_match = re.search(r'Bot\[(.*?)\]', board_str)
    top = top_match.group(1).split() if top_match and top_match.group(1) else []
    mid = mid_match.group(1).split() if mid_match and mid_match.group(1) else []
    bot = bot_match.group(1).split() if bot_match and bot_match.group(1) else []
    return top, mid, bot

def build_features(data):
    top, mid, bot = parse_board(data.get('board', 'Top[] Mid[] Bot[]'))
    hand_str = data.get('hand', '')
    hand = hand_str.split() if hand_str else []
    
    opp_top = data.get('opp_top', "").split()
    opp_mid = data.get('opp_mid', "").split()
    opp_bot = data.get('opp_bot', "").split()
    dead = data.get('dead_cards', "").split()
    
    all_cards = []
    for c in top: all_cards.append((c, 1))
    for c in mid: all_cards.append((c, 2))
    for c in bot: all_cards.append((c, 3))
    
    for c in opp_top: all_cards.append((c, 4))
    for c in opp_mid: all_cards.append((c, 5))
    for c in opp_bot: all_cards.append((c, 6))
    
    for c in dead: all_cards.append((c, 7))
    
    raw_features = [encode_card_str(c, r) for c, r in all_cards]
    
    features = np.zeros((MAX_CARDS, CARD_DIM), dtype=np.float32)
    features[:len(raw_features)] = np.stack(raw_features) if raw_features else np.zeros((0, CARD_DIM), dtype=np.float32)
    
    if hand:
        features[-len(hand):] = np.stack([encode_card_str(c, 0) for c in hand])
        
    return torch.from_numpy(features).unsqueeze(0).to(DEVICE)

def handle_client(conn, addr):
    #print(f"Connected by {addr}")
    try:
        buffer = ""
        while True:
            data = conn.recv(8192)
            if not data:
                break
            buffer += data.decode('utf-8')
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not line.strip():
                    continue
                
                req = json.loads(line)
                model_name = req.get('model', 't1_bb')
                
                if model_name not in models:
                    print(f"Model {model_name} not found. Initializing random model for testing...")
                    model = T1PlacementNet(128, 4, 4, 256, 0.0).to(DEVICE)
                    model.eval()
                    models[model_name] = model
                
                n_hand = 5 if model_name.startswith('t0') else 3
                
                features = build_features(req)
                with torch.no_grad():
                    logits, ev_pred = models[model_name](features, n_hand=n_hand)
                    
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
                ev = ev_pred.item()
                
                resp = {
                    "probs": probs,
                    "ev": ev
                }
                
                conn.sendall((json.dumps(resp) + "\n").encode('utf-8'))
    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        conn.close()

def start_server(host='127.0.0.1', port=5555):
    load_models()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(10)
    
    print(f"Inference server listening on {host}:{port}")
    
    try:
        while True:
            conn, addr = server.accept()
            # Handle each connection in a new thread
            t = threading.Thread(target=handle_client, args=(conn, addr))
            t.daemon = True
            t.start()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.close()

if __name__ == '__main__':
    start_server()
