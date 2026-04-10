#!/usr/bin/env python3
"""Procrustes alignment analysis on saved KV tensors."""

import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Procrustes alignment on saved KV tensors")
    parser.add_argument("data_dir", help="Path to saved KV tensors directory")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)

    num_samples = meta["num_samples"]
    shared_layers = meta["shared_layers"]
    model1 = meta["model1"]
    model2 = meta["model2"]

    print(f"Procrustes: {model1} ← {model2}, {num_samples} samples, {shared_layers} layers")

    results = {}
    for mode in ["Key", "Value"]:
        prefix = "k" if mode == "Key" else "v"
        nmse_per_layer = np.zeros(shared_layers)
        cos_per_layer = np.zeros(shared_layers)

        for layer_idx in range(shared_layers):
            t1 = torch.load(os.path.join(args.data_dir, f"{prefix}_model1_layer{layer_idx:02d}.pt"),
                            weights_only=True)
            t2 = torch.load(os.path.join(args.data_dir, f"{prefix}_model2_layer{layer_idx:02d}.pt"),
                            weights_only=True)

            nmse_samples = []
            cos_samples = []

            for s in range(num_samples):
                M1 = t1[s].squeeze(0).float()
                M2 = t2[s].squeeze(0).float()

                # Procrustes: SVD of M1^T @ M2
                cross = M1.T @ M2
                U, S, Vh = torch.linalg.svd(cross)
                R = Vh.T @ U.T

                M1_hat = M2 @ R

                energy = (M1 ** 2).sum().item()
                error = ((M1 - M1_hat) ** 2).sum().item()
                nmse_samples.append(error / energy if energy > 0 else float('inf'))
                # Per-token cosine similarity averaged across tokens
                cos_samples.append(
                    torch.nn.functional.cosine_similarity(M1, M1_hat, dim=-1).mean().item()
                )

            nmse_per_layer[layer_idx] = np.mean(nmse_samples)
            cos_per_layer[layer_idx] = np.mean(cos_samples)
            print(f"[{mode}] Layer {layer_idx:2d}: NMSE={nmse_per_layer[layer_idx]:.4f}  "
                  f"cos={cos_per_layer[layer_idx]:.4f}")

        results[mode] = {"nmse": nmse_per_layer, "cos": cos_per_layer}
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for mode in ["Key", "Value"]:
        ax.plot(range(shared_layers), results[mode]["nmse"],
                marker="o" if mode == "Key" else "s", ms=4, lw=1.5, label=f"{mode}-Cache")
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("NMSE", fontsize=14)
    ax.set_title("Procrustes: NMSE per layer", fontsize=15)
    ax.set_yscale("log")
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mode in ["Key", "Value"]:
        ax.plot(range(shared_layers), results[mode]["cos"],
                marker="o" if mode == "Key" else "s", ms=4, lw=1.5, label=f"{mode}-Cache")
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title("Procrustes: Cosine Similarity per layer", fontsize=15)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Procrustes: {model1} ← {model2}", fontsize=15, y=1.02)
    plt.tight_layout()

    out = args.output or os.path.join(args.data_dir, "procrustes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
