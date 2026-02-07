"""
Behavior Cloning for FL RL Policy

Pre-trains the MaskablePPO policy on expert demonstrations from exhaustive solver.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ai.fl_rl_env import FantasylandEnv, MAX_CARDS, CARD_FEATURES, POSITIONS


class ExpertDataset(Dataset):
    """Dataset of expert demonstrations for behavior cloning."""
    
    def __init__(self, jsonl_path: str, num_cards: int = 14):
        self.samples = []
        self.num_cards = num_cards
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    # Filter by card count
                    if len(sample['hand']) == num_cards:
                        self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples with {num_cards} cards")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build observation (same as fl_rl_env)
        cards_obs = np.zeros((MAX_CARDS, CARD_FEATURES + POSITIONS), dtype=np.float32)
        
        for i, card in enumerate(sample['hand']):
            cards_obs[i, :CARD_FEATURES] = self._encode_card(card)
        
        obs = cards_obs.flatten()
        
        # Build expert actions (card_idx, position) pairs
        actions = []
        for card_idx, card in enumerate(sample['hand']):
            key = self._card_key(card)
            
            # Find position in solution
            position = None
            for pos_idx, pos_name in enumerate(['top', 'middle', 'bottom', 'discards']):
                for sol_card in sample['solution'][pos_name]:
                    if self._card_key(sol_card) == key:
                        position = pos_idx
                        break
                if position is not None:
                    break
            
            if position is not None:
                action = card_idx * POSITIONS + position
                actions.append(action)
        
        return {
            'obs': torch.tensor(obs, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long),
            'reward': sample['reward'],
            'can_stay': sample['can_stay'],
        }
    
    def _card_key(self, card):
        if card['is_joker']:
            return ('joker', card.get('suit', 'j'))
        return (card['rank'], card['suit'])
    
    def _encode_card(self, card) -> np.ndarray:
        RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        SUITS = ['s', 'h', 'd', 'c']
        
        # CARD_FEATURES = 18: 13(rank) + 4(suit) + 1(joker)
        features = np.zeros(CARD_FEATURES, dtype=np.float32)
        
        if card['is_joker']:
            features[17] = 1.0  # joker flag at index 17
        else:
            if card['rank'] in RANKS:
                features[RANKS.index(card['rank'])] = 1.0
            if card['suit'] in SUITS:
                features[13 + SUITS.index(card['suit'])] = 1.0
        
        return features


def pretrain_policy(args):
    """Pretrain policy with behavior cloning."""
    print(f"Behavior Cloning Pretraining")
    print(f"  Data: {args.data_path}")
    print(f"  Num cards: {args.num_cards}")
    print(f"  Epochs: {args.epochs}")
    print()
    
    # Load dataset
    dataset = ExpertDataset(args.data_path, num_cards=args.num_cards)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create environment and model
    def mask_fn(env):
        return env.action_masks()
    
    env = FantasylandEnv(num_cards=args.num_cards)
    env = ActionMasker(env, mask_fn)
    
    # Create or load model
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            verbose=0,
        )
    
    # Get policy network
    policy = model.policy
    device = policy.device
    
    # Optimizer for behavior cloning
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(dataset)} samples...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            obs_batch = batch['obs'].to(device)
            actions_batch = batch['actions']  # List of tensors
            
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            
            for i in range(len(obs_batch)):
                sample_obs = obs_batch[i:i+1]
                sample_actions = actions_batch[i].to(device)
                
                if len(sample_actions) == 0:
                    continue
                
                # Get action logits from policy
                features = policy.extract_features(sample_obs)
                if hasattr(policy, 'mlp_extractor'):
                    latent_pi = policy.mlp_extractor.forward_actor(features)
                    logits = policy.action_net(latent_pi)
                else:
                    logits = policy.action_net(features)
                
                # Use first action as target (simplified BC)
                target = sample_actions[0:1]
                loss = criterion(logits, target)
                batch_loss = batch_loss + loss
                
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += 1
            
            if batch_loss.requires_grad:
                batch_loss.backward()
                optimizer.step()
            
            total_loss += batch_loss.item()
        
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Accuracy={accuracy*100:.2f}%")
    
    # Save pretrained model
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "fl_bc_pretrained")
    model.save(output_path)
    print(f"\nPretrained model saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning for FL RL")
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--num-cards", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None, help="Resume from model")
    parser.add_argument("--output-dir", type=str, default="ai/training/fl_bc")
    
    args = parser.parse_args()
    pretrain_policy(args)


if __name__ == "__main__":
    main()
