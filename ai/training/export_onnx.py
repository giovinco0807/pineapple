"""
Export PyTorch models to ONNX format for Rust benchmark.

Usage:
    python -m ai.training.export_onnx
    python -m ai.training.export_onnx --bc ai/models/expectimax_bc_v4/bc_policy_best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np

from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.engine.encoding import STATE_DIM
from ai.engine.action_space import MAX_ACTIONS


class PolicyNetworkONNX(nn.Module):
    """Wrapper that outputs raw logits (no mask, no softmax) for ONNX export."""

    def __init__(self, policy: PolicyNetwork):
        super().__init__()
        self.net = policy.net

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ValueNetworkONNX(nn.Module):
    """Wrapper that outputs (value, bust_prob, fl_prob, royalty_ev) as a single tensor."""

    def __init__(self, vn: ValueNetwork):
        super().__init__()
        self.shared = vn.shared
        self.value_head = vn.value_head
        self.bust_head = vn.bust_head
        self.fl_head = vn.fl_head
        self.royalty_head = vn.royalty_head

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.shared(state)
        value = self.value_head(x)
        bust_prob = torch.sigmoid(self.bust_head(x))
        fl_prob = torch.sigmoid(self.fl_head(x))
        royalty_ev = self.royalty_head(x)
        # Concatenate: [value, bust_prob, fl_prob, royalty_ev] each (batch, 1)
        return torch.cat([value, bust_prob, fl_prob, royalty_ev], dim=-1)


def load_model(cls, path: str):
    model = cls()
    ck = torch.load(path, map_location='cpu', weights_only=True)
    state_dict = ck.get('model_state_dict', ck)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ck


def export_bc(model_path: str, output_path: str):
    policy, _ = load_model(PolicyNetwork, model_path)
    wrapper = PolicyNetworkONNX(policy)
    wrapper.eval()

    dummy = torch.randn(1, STATE_DIM)
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    test_input = np.random.randn(2, STATE_DIM).astype(np.float32)
    onnx_out = sess.run(None, {"state": test_input})[0]
    torch_out = wrapper(torch.from_numpy(test_input)).detach().numpy()
    max_diff = np.max(np.abs(onnx_out - torch_out))
    print(f"  BC exported: {output_path} (max diff: {max_diff:.2e})")
    assert max_diff < 1e-3, f"ONNX/PyTorch mismatch: {max_diff}"


def export_vn(model_path: str, output_path: str):
    vn, ck = load_model(ValueNetwork, model_path)
    wrapper = ValueNetworkONNX(vn)
    wrapper.eval()

    dummy = torch.randn(1, STATE_DIM)
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["state"],
        output_names=["output"],  # [value, bust_prob, fl_prob, royalty_ev]
        dynamic_axes={"state": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    # Save norm_stats alongside
    ns = ck.get('norm_stats', None)
    if ns:
        import json
        ns_path = output_path.replace('.onnx', '_norm_stats.json')
        json_ns = {k: float(v) if isinstance(v, (int, float)) else v for k, v in ns.items()}
        with open(ns_path, 'w') as f:
            json.dump(json_ns, f)
        print(f"  Norm stats saved: {ns_path}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    test_input = np.random.randn(2, STATE_DIM).astype(np.float32)
    onnx_out = sess.run(None, {"state": test_input})[0]
    torch_out = wrapper(torch.from_numpy(test_input)).detach().numpy()
    max_diff = np.max(np.abs(onnx_out - torch_out))
    print(f"  VN exported: {output_path} (max diff: {max_diff:.2e})")
    assert max_diff < 1e-3, f"ONNX/PyTorch mismatch: {max_diff}"


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--bc", default="ai/models/selfplay_iter17/bc_policy_best.pt")
    parser.add_argument("--vn", default="ai/models/value_v1/value_best.pt")
    parser.add_argument("--bc-t3", default="ai/models/bc_t3/bc_policy_best.pt")
    parser.add_argument("--bc-t4", default="ai/models/bc_t4/bc_policy_best.pt")
    parser.add_argument("--out-dir", default="ai/rust_benchmark/models")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Exporting models to ONNX...")

    # Common BC
    export_bc(args.bc, str(out / "bc.onnx"))

    # VN
    export_vn(args.vn, str(out / "vn.onnx"))

    # Per-turn BC
    for turn, path in [(3, args.bc_t3), (4, args.bc_t4)]:
        if Path(path).exists():
            export_bc(path, str(out / f"bc_t{turn}.onnx"))
        else:
            print(f"  Skipping T{turn} BC: {path} not found")

    print("\nDone! All models exported to", out)


if __name__ == "__main__":
    main()
