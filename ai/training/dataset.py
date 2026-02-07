"""
Dataset for Fantasyland supervised learning.
"""

import json
import torch
from torch.utils.data import Dataset

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']


class FantasylandDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 配置先のマッピングを作成
        placement_map = {}
        for card in sample['solution']['top']:
            key = self._card_key(card)
            placement_map[key] = 0  # top
        for card in sample['solution']['middle']:
            key = self._card_key(card)
            placement_map[key] = 1  # middle
        for card in sample['solution']['bottom']:
            key = self._card_key(card)
            placement_map[key] = 2  # bottom
        for card in sample['solution']['discards']:
            key = self._card_key(card)
            placement_map[key] = 3  # discard
        
        # カードをエンコード
        hand_encoded = []
        labels = []
        
        for card in sample['hand']:
            features = self.encode_card(card)
            hand_encoded.append(features)
            
            key = self._card_key(card)
            label = placement_map.get(key, 3)
            labels.append(label)
        
        n_cards = len(sample['hand'])
        
        # パディング（17枚まで）
        while len(hand_encoded) < 17:
            hand_encoded.append([0] * 19)
            labels.append(-1)  # 無視するラベル
        
        return {
            'hand': torch.tensor(hand_encoded, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'n_cards': torch.tensor(n_cards, dtype=torch.long),
            'reward': torch.tensor(sample['reward'], dtype=torch.float32),
            'royalties': torch.tensor(sample['royalties'], dtype=torch.float32),
            'can_stay': torch.tensor(sample['can_stay'], dtype=torch.bool),
        }
    
    def _card_key(self, card):
        """カードのユニークキーを生成"""
        if card['is_joker']:
            return ('joker', card.get('suit', 'j'))
        return (card['rank'], card['suit'])
    
    def encode_card(self, card):
        """カードを19次元ベクトルにエンコード"""
        # 13(rank) + 4(suit) + 1(joker) + 1(valid)
        features = [0.0] * 19
        
        if card['is_joker']:
            features[17] = 1.0  # joker flag
        else:
            # rank one-hot
            if card['rank'] in RANKS:
                rank_idx = RANKS.index(card['rank'])
                features[rank_idx] = 1.0
            elif card['rank'] == 'T':
                features[8] = 1.0  # 10
            
            # suit one-hot
            if card['suit'] in SUITS:
                suit_idx = SUITS.index(card['suit'])
                features[13 + suit_idx] = 1.0
        
        features[18] = 1.0  # valid card flag
        
        return features


def collate_fn(batch):
    """カスタムcollate関数"""
    return {
        'hand': torch.stack([b['hand'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'n_cards': torch.stack([b['n_cards'] for b in batch]),
        'reward': torch.stack([b['reward'] for b in batch]),
        'royalties': torch.stack([b['royalties'] for b in batch]),
        'can_stay': torch.stack([b['can_stay'] for b in batch]),
    }
