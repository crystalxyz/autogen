#!/usr/bin/env python3
"""Cosine similarity analysis on saved KV tensors."""

import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity on saved KV tensors")
    parser.add_argument("data_dir", help="Path to saved KV tensors directory")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)

    num_samples = meta["num_samples"]
    shared_layers = meta["shared_layers"]
    model1 = meta["model1"]
    model2 = meta["model2"]

    print(f"Cosine: {model1} vs {model2}, {num_samples} samples, {shared_layers} layers")

    results = {}
    for mode in ["Key", "Value"]:
        prefix = "k" if mode == "Key" else "v"
        cos_per_layer = np.zeros(shared_layers)

        for layer_idx in range(shared_layers):
            t1 = torch.load(os.path.join(args.data_dir, f"{prefix}_model1_layer{layer_idx:02d}.pt"),
                            weights_only=True)
            t2 = torch.load(os.path.join(args.data_dir, f"{prefix}_model2_layer{layer_idx:02d}.pt"),
                            weights_only=True)

            cos_values = []
            for s in range(num_samples):
                rep1 = t1[s].squeeze(0).float().flatten()
                rep2 = t2[s].squeeze(0).float().flatten()
                cos_values.append(
                    torch.nn.functional.cosine_similarity(
                        rep1.unsqueeze(0), rep2.unsqueeze(0)
                    ).item()
                )

            cos_per_layer[layer_idx] = np.mean(cos_values)
            print(f"[{mode}] Layer {layer_idx:2d}: cos={cos_per_layer[layer_idx]:.4f}")

        results[mode] = cos_per_layer
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(shared_layers), results["Key"], marker="o", ms=4, lw=1.5, label="Key-Cache")
    ax.plot(range(shared_layers), results["Value"], marker="s", ms=4, lw=1.5, label="Value-Cache")
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title(f"Cross-Model Cosine Similarity: {model1} vs {model2}", fontsize=15)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or os.path.join(args.data_dir, "cosine.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
