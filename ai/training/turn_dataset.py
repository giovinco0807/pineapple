import json
import torch
from torch.utils.data import Dataset

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']

class TurnValueDataset(Dataset):
    """
    Dataset for Turn (T1-T4) Value Network.
    Input: Board state (after placement)
    Target: EV (Expected Value)
    """
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                placements = data.get('placements', [])
                for placement in placements:
                    p_str = placement.get('p', '')
                    ev = placement.get('ev', 0.0)
                    board = self._parse_placement_string(p_str)
                    if board:
                        self.samples.append({
                            'board': board,
                            'ev': ev
                        })
        print(f"Loaded {len(self.samples)} board states from {jsonl_path}")
    
    def _parse_placement_string(self, p_str):
        import re
        parsed = {'top': [], 'mid': [], 'bot': []}
        
        m_top = re.search(r'Top\[(.*?)\]', p_str)
        m_mid = re.search(r'Mid\[(.*?)\]', p_str)
        m_bot = re.search(r'Bot\[(.*?)\]', p_str)
        
        def parse_row(row_str):
            cards = []
            if not row_str: return cards
            tokens = row_str.split()
            for t in tokens:
                if t == '-' or not t: continue
                if t.lower() in ('jo', 'jk', 'joker'):
                    cards.append({'rank': 'Joker', 'suit': 'joker'})
                else:
                    if len(t) == 2:
                        cards.append({'rank': t[0].upper(), 'suit': t[1].lower()})
            return cards

        if m_top: parsed['top'] = parse_row(m_top.group(1))
        if m_mid: parsed['mid'] = parse_row(m_mid.group(1))
        if m_bot: parsed['bot'] = parse_row(m_bot.group(1))
        
        return parsed

    def __len__(self):
        return len(self.samples)
    
    def encode_card(self, card):
        # 13(rank) + 4(suit) + 1(joker) + 1(valid)
        features = [0.0] * 19
        if card['rank'] == 'Joker':
            features[17] = 1.0
        else:
            if card['rank'] in RANKS:
                features[RANKS.index(card['rank'])] = 1.0
            if card['suit'] in SUITS:
                features[13 + SUITS.index(card['suit'])] = 1.0
        features[18] = 1.0
        return features

    def __getitem__(self, idx):
        sample = self.samples[idx]
        board = sample['board']
        
        # Encode board as 13 cards (3 top, 5 mid, 5 bot)
        encoded_board = []
        labels_row = []
        
        for r_idx, row in enumerate(['top', 'mid', 'bot']):
            for card in board[row]:
                encoded_board.append(self.encode_card(card))
                labels_row.append(r_idx)
                
        # pad to 13 cards
        while len(encoded_board) < 13:
            encoded_board.append([0.0] * 19)
            labels_row.append(-1)
            
        return {
            'board': torch.tensor(encoded_board, dtype=torch.float32),
            'rows': torch.tensor(labels_row, dtype=torch.long),
            'ev': torch.tensor(sample['ev'], dtype=torch.float32)
        }

def collate_fn(batch):
    return {
        'board': torch.stack([b['board'] for b in batch]),
        'rows': torch.stack([b['rows'] for b in batch]),
        'ev': torch.stack([b['ev'] for b in batch])
    }
