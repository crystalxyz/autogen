#!/usr/bin/env python3
"""Least-squares projection analysis with leave-one-out CV on saved KV tensors."""

import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Least-squares projection on saved KV tensors")
    parser.add_argument("data_dir", help="Path to saved KV tensors directory")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("--num-folds", type=int, default=None,
                        help="Number of LOO folds (default: all samples)")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)

    num_samples = meta["num_samples"]
    shared_layers = meta["shared_layers"]
    model1 = meta["model1"]
    model2 = meta["model2"]
    num_folds = args.num_folds or num_samples

    print(f"Least-squares: {model1} ← {model2}, {num_samples} samples, "
          f"{shared_layers} layers, {num_folds} LOO folds")

    results = {}
    for mode in ["Key", "Value"]:
        prefix = "k" if mode == "Key" else "v"
        train_mse_per_layer = np.zeros(shared_layers)
        test_nmse_per_layer = np.zeros(shared_layers)
        test_cos_per_layer = np.zeros(shared_layers)

        for layer_idx in range(shared_layers):
            t1 = torch.load(os.path.join(args.data_dir, f"{prefix}_model1_layer{layer_idx:02d}.pt"),
                            weights_only=True)
            t2 = torch.load(os.path.join(args.data_dir, f"{prefix}_model2_layer{layer_idx:02d}.pt"),
                            weights_only=True)

            # Collect all samples
            all_m1 = [t1[s].squeeze(0).float() for s in range(num_samples)]
            all_m2 = [t2[s].squeeze(0).float() for s in range(num_samples)]

            train_mse_folds = []
            test_nmse_folds = []
            test_cos_folds = []

            for held_out in range(num_folds):
                # Train: all except held_out
                train_m1 = torch.cat([all_m1[i] for i in range(num_samples) if i != held_out], dim=0)
                train_m2 = torch.cat([all_m2[i] for i in range(num_samples) if i != held_out], dim=0)

                W_star = torch.linalg.lstsq(train_m2, train_m1).solution

                # Train error
                train_pred = train_m2 @ W_star
                train_mse_folds.append(((train_m1 - train_pred) ** 2).mean().item())

                # Test error
                test_m1 = all_m1[held_out]
                test_m2 = all_m2[held_out]
                test_pred = test_m2 @ W_star

                test_error = ((test_m1 - test_pred) ** 2).sum().item()
                test_energy = (test_m1 ** 2).sum().item()
                test_nmse_folds.append(test_error / test_energy if test_energy > 0 else float('inf'))
                # Per-token cosine similarity averaged across tokens
                test_cos_folds.append(
                    torch.nn.functional.cosine_similarity(test_m1, test_pred, dim=-1).mean().item()
                )

            train_mse_per_layer[layer_idx] = np.mean(train_mse_folds)
            test_nmse_per_layer[layer_idx] = np.mean(test_nmse_folds)
            test_cos_per_layer[layer_idx] = np.mean(test_cos_folds)
            print(f"[{mode}] Layer {layer_idx:2d}: train_MSE={train_mse_per_layer[layer_idx]:.6f}  "
                  f"test_NMSE={test_nmse_per_layer[layer_idx]:.4f}  "
                  f"test_cos={test_cos_per_layer[layer_idx]:.4f}")

        results[mode] = {
            "train_mse": train_mse_per_layer,
            "test_nmse": test_nmse_per_layer,
            "test_cos": test_cos_per_layer,
        }
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for mode in ["Key", "Value"]:
        marker = "o" if mode == "Key" else "s"
        ax.plot(range(shared_layers), results[mode]["test_nmse"],
                marker=marker, ms=4, lw=1.5, label=f"{mode} test")
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("NMSE", fontsize=14)
    ax.set_title("Least-Squares (LOO): NMSE per layer", fontsize=15)
    ax.set_yscale("log")
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mode in ["Key", "Value"]:
        marker = "o" if mode == "Key" else "s"
        ax.plot(range(shared_layers), results[mode]["test_cos"],
                marker=marker, ms=4, lw=1.5, label=f"{mode} test")
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title("Least-Squares (LOO): Cosine per layer", fontsize=15)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Least-Squares: {model1} ← {model2}", fontsize=15, y=1.02)
    plt.tight_layout()

    out = args.output or os.path.join(args.data_dir, "lstsq.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")

    # Save numerical results
    results_path = os.path.join(args.data_dir, "lstsq_results.json")
    save_results = {}
    for mode in results:
        save_results[mode] = {k: v.tolist() for k, v in results[mode].items()}
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
