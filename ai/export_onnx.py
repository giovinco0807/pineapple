"""
Export PolicyNetSmall model to ONNX format for inference.

Usage:
    python ai/export_onnx.py --model ai/models/t0_ranking_v2/bc_policy_best.pt --output ai/models/t0_ranking_v2/policy.onnx
"""
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.training.train_ranking import PolicyNetSmall
from ai.engine.encoding import STATE_DIM
from ai.engine.action_space import MAX_ACTIONS


def export_onnx(model_path: str, output_path: str, input_dim: int = STATE_DIM):
    """Export PolicyNetSmall to ONNX."""
    model = PolicyNetSmall(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy inputs
    dummy_state = torch.randn(1, input_dim)
    dummy_mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_state, dummy_mask),
        str(output_path),
        input_names=["state", "valid_mask"],
        output_names=["probs"],
        dynamic_axes={
            "state": {0: "batch"},
            "valid_mask": {0: "batch"},
            "probs": {0: "batch"},
        },
        opset_version=17,
    )
    
    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    
    test_state = np.random.randn(1, input_dim).astype(np.float32)
    test_mask = np.ones((1, MAX_ACTIONS), dtype=bool)
    
    result = sess.run(None, {"state": test_state, "valid_mask": test_mask})
    print(f"ONNX export successful: {output_path}")
    print(f"  Model size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Output shape: {result[0].shape}")
    print(f"  Sum of probs: {result[0].sum():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai/models/t0_ranking_v2/bc_policy_best.pt")
    parser.add_argument("--output", default="ai/models/t0_ranking_v2/policy.onnx")
    parser.add_argument("--input-dim", type=int, default=STATE_DIM)
    args = parser.parse_args()
    
    export_onnx(args.model, args.output, args.input_dim)
