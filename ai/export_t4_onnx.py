import sys
from pathlib import Path
import torch

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))

from training.train_t4_oracle import T4PolicyValueNet

STATE_DIM = 522
ACTION_DIM = 27

def main():
    device = torch.device("cpu")
    model = T4PolicyValueNet(state_dim=STATE_DIM, n_actions=ACTION_DIM).to(device)
    pt_path = AI_DIR / "data/t4_oracle_v2/t4_policyvalue_best.pt"
    onnx_path = AI_DIR / "data/t4_oracle_v2/t4_oracle.onnx"
    checkpoint = torch.load(pt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_state = torch.randn(1, STATE_DIM).to(device)
    dummy_mask = torch.ones(1, ACTION_DIM, dtype=torch.bool).to(device)

    torch.onnx.export(
        model,
        (dummy_state, dummy_mask),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["state", "mask"],
        output_names=["logits", "value"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "value": {0: "batch_size"}
        }
    )
    print(f"Exported T4 model to {onnx_path}")

if __name__ == "__main__":
    main()
