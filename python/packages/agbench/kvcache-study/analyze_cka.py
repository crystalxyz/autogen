#!/usr/bin/env python3
"""CKA similarity analysis on saved KV tensors."""

import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def linear_cka_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear CKA using kernel formulation (Kornblith et al., 2019)."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = (K * L).sum()
    norm_k = (K * K).sum().sqrt()
    norm_l = (L * L).sum().sqrt()
    if norm_k == 0 or norm_l == 0:
        return torch.tensor(0.0)
    return hsic / (norm_k * norm_l)


def main():
    parser = argparse.ArgumentParser(description="CKA similarity on saved KV tensors")
    parser.add_argument("data_dir", help="Path to saved KV tensors directory")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)

    num_samples = meta["num_samples"]
    shared_layers = meta["shared_layers"]
    model1 = meta["model1"]
    model2 = meta["model2"]

    print(f"CKA: {model1} vs {model2}, {num_samples} samples, {shared_layers} layers")

    results = {}
    for mode in ["Key", "Value"]:
        prefix = "k" if mode == "Key" else "v"
        cka_per_layer = np.zeros(shared_layers)

        for layer_idx in range(shared_layers):
            t1 = torch.load(os.path.join(args.data_dir, f"{prefix}_model1_layer{layer_idx:02d}.pt"),
                            weights_only=True)  # [N, 1, seq, d]
            t2 = torch.load(os.path.join(args.data_dir, f"{prefix}_model2_layer{layer_idx:02d}.pt"),
                            weights_only=True)

            cka_values = []
            for s in range(num_samples):
                rep1 = t1[s].squeeze(0).float()  # [seq, d]
                rep2 = t2[s].squeeze(0).float()
                cka_values.append(linear_cka_kernel(rep1, rep2).item())

            cka_per_layer[layer_idx] = np.mean(cka_values)
            print(f"[{mode}] Layer {layer_idx:2d}: CKA={cka_per_layer[layer_idx]:.4f}")

        results[mode] = cka_per_layer
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(shared_layers), results["Key"], marker="o", ms=4, lw=1.5, label="Key-Cache")
    ax.plot(range(shared_layers), results["Value"], marker="s", ms=4, lw=1.5, label="Value-Cache")
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("CKA Similarity", fontsize=14)
    ax.set_title(f"Cross-Model CKA: {model1} vs {model2}", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or os.path.join(args.data_dir, "cka.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
