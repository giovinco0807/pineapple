import torch
import numpy as np
from itertools import product
from ai.models.t1_network import T1PlacementNet, encode_card_str, CARD_DIM

class T1Agent:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        d_model = config.get('d_model', 128)
        num_layers = config.get('num_layers', 4)
        dropout = config.get('dropout', 0.2)
        
        self.model = T1PlacementNet(d_model=d_model, nhead=4, num_layers=num_layers, dim_ff=d_model*2, dropout=dropout)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_action(self, top, mid, bot, hand):
        """
        top, mid, bot: list of card strings e.g. ['Ad', 'Ah']
        hand: list of 3 card strings e.g. ['X2', '8d', '5c']
        Returns a dictionary indicating the placement.
        """
        assert len(hand) == 3, "T1 hand must have exactly 3 cards"
        
        all_cards = []
        for c in top: all_cards.append((c, 1))
        for c in mid: all_cards.append((c, 2))
        for c in bot: all_cards.append((c, 3))
        for c in hand: all_cards.append((c, 0))
        
        features = np.stack([encode_card_str(c, r) for c, r in all_cards])
        features = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            row_logits, ev_pred = self.model(features)
        
        # row_logits is [1, 3, 4]
        logits = row_logits[0].cpu().numpy()  # shape (3, 4)
        
        best_score = -float('inf')
        best_config = None
        
        # 4 classes: 0=Top, 1=Mid, 2=Bot, 3=Discard
        for config in product(range(4), repeat=3):
            # Constraint 1: Exactly 1 discard
            if config.count(3) != 1:
                continue
                
            # Count added cards to each row
            added = {0: 0, 1: 0, 2: 0}
            for p in config:
                if p != 3:
                    added[p] += 1
            
            # Constraint 2: Row limits
            if len(top) + added[0] > 3: continue
            if len(mid) + added[1] > 5: continue
            if len(bot) + added[2] > 5: continue
            
            # Valid configuration! Compute score
            score = logits[0, config[0]] + logits[1, config[1]] + logits[2, config[2]]
            if score > best_score:
                best_score = score
                best_config = config
                
        # Format the result
        target_map_rev = {0: 'Top', 1: 'Mid', 2: 'Bot', 3: 'Discard'}
        result = []
        for i, c in enumerate(hand):
            result.append((c, target_map_rev[best_config[i]]))
            
        return result, ev_pred.item()
