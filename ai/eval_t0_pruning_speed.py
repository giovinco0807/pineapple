import time
import torch
import torch.nn.functional as F
import itertools
import numpy as np

# Import the model and preprocessing from the training script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from train_t0_placement import T0PlacementNet, CARD_DIM, NUM_ROWS, MAX_CARDS

def generate_valid_t0_placements():
    """Generate all valid T0 placements (Top <= 3 cards). Returns shape (232, 5)."""
    valid = []
    # 0=Top, 1=Mid, 2=Bot
    for p in itertools.product([0, 1, 2], repeat=5):
        if p.count(0) <= 3:
            valid.append(p)
    return torch.tensor(valid, dtype=torch.long)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'ai/models/t0_placement_net_v4.pt'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    d_model = config.get('d_model', 128)
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.2)
    
    model = T0PlacementNet(
        d_model=d_model, nhead=4, num_layers=num_layers,
        dim_ff=d_model * 2, dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    V_tensor = generate_valid_t0_placements().to(device)
    num_valid = V_tensor.shape[0]
    print(f"Total valid T0 placements: {num_valid}")
    
    # Warmup and benchmarking
    def benchmark(batch_size, num_iterations=100):
        # Create dummy batch of features (B, 5, CARD_DIM)
        # Random normal is fine for speed test
        dummy_input = torch.randn(batch_size, MAX_CARDS, CARD_DIM, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                logits, ev_pred = model(dummy_input)
                log_p = F.log_softmax(logits, dim=-1)
                
                scores = torch.zeros(batch_size, num_valid, device=device)
                b_idx = torch.arange(batch_size, device=device).view(-1, 1)
                for i in range(MAX_CARDS):
                    v_idx = V_tensor[:, i].view(1, -1)
                    scores += log_p[b_idx, i, v_idx]
                
                top_scores, top_indices = torch.topk(scores, 50, dim=1)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                logits, ev_pred = model(dummy_input)
                log_p = F.log_softmax(logits, dim=-1)
                
                scores = torch.zeros(batch_size, num_valid, device=device)
                b_idx = torch.arange(batch_size, device=device).view(-1, 1)
                for i in range(MAX_CARDS):
                    v_idx = V_tensor[:, i].view(1, -1)
                    scores += log_p[b_idx, i, v_idx]
                
                top_scores, top_indices = torch.topk(scores, 50, dim=1)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        time_per_batch_ms = (total_time / num_iterations) * 1000
        time_per_hand_ms = time_per_batch_ms / batch_size
        hands_per_sec = (batch_size * num_iterations) / total_time
        
        print(f"Batch Size {batch_size:4d} | Latency: {time_per_batch_ms:.2f} ms/batch ({time_per_hand_ms:.4f} ms/hand) | Throughput: {hands_per_sec:,.0f} hands/sec")
        
    print("\n--- Pruning to Top 50 Benchmark ---")
    benchmark(batch_size=1, num_iterations=1000)
    benchmark(batch_size=32, num_iterations=100)
    benchmark(batch_size=128, num_iterations=100)
    benchmark(batch_size=1024, num_iterations=50)

if __name__ == '__main__':
    main()
