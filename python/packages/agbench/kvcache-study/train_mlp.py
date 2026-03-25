"""
Phase 3: Train per-layer MLPs to map KV tensors from model A (small) to model B (large).

Usage:
    python train_mlp.py --model_a data/Qwen_Qwen3-1.7B --model_b data/Qwen_Qwen3-4B --output_dir results/mlp/
    python train_mlp.py --model_a data/Qwen_Qwen3-1.7B --model_b data/Qwen_Qwen3-4B --tensor_type keys --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LayerMLP(nn.Module):
    """
    MLP that maps a flattened KV vector from model A to model B for one layer.
    Uses skip connections and LayerNorm for stability.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_hidden: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_kv_dataset(
    dir_a: Path,
    dir_b: Path,
    tensor_type: str,
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (input, target) pairs for a specific layer.
    Returns (X_a, X_b) each of shape [n_problems, flattened_dim].
    """
    files_a = sorted(dir_a.glob("*.pt"))
    files_b = sorted(dir_b.glob("*.pt"))

    # Match by filename (task_id)
    names_a = {f.stem: f for f in files_a}
    names_b = {f.stem: f for f in files_b}
    common = sorted(names_a.keys() & names_b.keys())

    if not common:
        raise ValueError(f"No overlapping task IDs between {dir_a} and {dir_b}")

    X_a, X_b = [], []
    for name in common:
        data_a = torch.load(names_a[name], weights_only=True)
        data_b = torch.load(names_b[name], weights_only=True)
        t_a = data_a[tensor_type][layer_idx]  # [num_heads, seq_len, head_dim]
        t_b = data_b[tensor_type][layer_idx]
        # Mean-pool over seq_len then flatten
        X_a.append(t_a.mean(dim=1).numpy().flatten())
        X_b.append(t_b.mean(dim=1).numpy().flatten())

    return np.stack(X_a, axis=0), np.stack(X_b, axis=0)


def make_splits(n: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_layer(
    X_a: np.ndarray,
    X_b: np.ndarray,
    layer_idx: int,
    tensor_type: str,
    out_dir: Path,
    hidden_dim: int = 512,
    num_hidden: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Train one MLP for one layer. Returns metrics dict."""
    n = len(X_a)
    train_idx, val_idx, test_idx = make_splits(n)

    def to_tensor(arr: np.ndarray, indices) -> torch.Tensor:
        return torch.tensor(arr[indices], dtype=torch.float32).to(device)

    X_tr, Y_tr = to_tensor(X_a, train_idx), to_tensor(X_b, train_idx)
    X_val, Y_val = to_tensor(X_a, val_idx), to_tensor(X_b, val_idx)
    X_te, Y_te = to_tensor(X_a, test_idx), to_tensor(X_b, test_idx)

    in_dim = X_a.shape[1]
    out_dim = X_b.shape[1]

    model = LayerMLP(in_dim, out_dim, hidden_dim=hidden_dim, num_hidden=num_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    Layer {layer_idx:02d} [{tensor_type}] epoch {epoch:3d}/{epochs} "
                  f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_te), Y_te).item()
        # Cosine similarity
        pred = model(X_te)
        cos_sim = nn.functional.cosine_similarity(pred, Y_te, dim=-1).mean().item()

    # Save model
    ckpt_path = out_dir / f"layer_{layer_idx:02d}_{tensor_type}.pt"
    torch.save({"state_dict": best_state, "in_dim": in_dim, "out_dim": out_dim,
                "hidden_dim": hidden_dim, "num_hidden": num_hidden}, ckpt_path)

    return {
        "layer": layer_idx,
        "tensor_type": tensor_type,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_cos_sim": cos_sim,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(metrics_list: list[dict], out_dir: Path, tensor_type: str) -> None:
    n = len(metrics_list)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]

    for i, m in enumerate(metrics_list):
        ax = axes[i]
        ax.plot(m["train_losses"], label="train", linewidth=1.2)
        ax.plot(m["val_losses"], label="val", linewidth=1.2)
        ax.set_title(f"Layer {m['layer']} | test_loss={m['test_loss']:.4f}")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"MLP training curves ({tensor_type})", fontsize=12)
    plt.tight_layout()
    path = out_dir / f"loss_curves_{tensor_type}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved loss curves to {path}")


def plot_test_summary(metrics_list: list[dict], out_dir: Path, tensor_type: str) -> None:
    layers = [m["layer"] for m in metrics_list]
    test_losses = [m["test_loss"] for m in metrics_list]
    cos_sims = [m["test_cos_sim"] for m in metrics_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(layers, test_losses)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("MSE (test)")
    axes[0].set_title(f"Per-layer test MSE ({tensor_type})")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(layers, cos_sims, color="orange")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cosine similarity (test)")
    axes[1].set_title(f"Per-layer cosine similarity ({tensor_type})")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / f"test_summary_{tensor_type}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved test summary to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train per-layer MLPs to map KV tensors between models.")
    parser.add_argument("--model_a", type=str, default="data/Qwen_Qwen3-1.7B")
    parser.add_argument("--model_b", type=str, default="data/Qwen_Qwen3-4B")
    parser.add_argument("--tensor_type", type=str, choices=["keys", "values", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default="results/mlp")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_hidden", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer indices to train (default: all layers)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dir_a = Path(args.model_a)
    dir_b = Path(args.model_b)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of layers from first file
    sample_file = next(dir_a.glob("*.pt"))
    sample = torch.load(sample_file, weights_only=True)
    num_layers = len(sample["keys"])
    layer_indices = args.layers if args.layers is not None else list(range(num_layers))

    print(f"Model A: {dir_a.name} | Model B: {dir_b.name}")
    print(f"Total layers: {num_layers}, training layers: {layer_indices}")
    print(f"Device: {args.device}")

    tensor_types = ["keys", "values"] if args.tensor_type == "both" else [args.tensor_type]

    all_metrics = {}
    for ttype in tensor_types:
        print(f"\n{'='*60}")
        print(f"Training MLPs for: {ttype}")
        print(f"{'='*60}")
        metrics_list = []
        t0 = time.time()

        for layer_idx in layer_indices:
            print(f"\n  Layer {layer_idx}/{max(layer_indices)} ...")
            X_a, X_b = load_kv_dataset(dir_a, dir_b, ttype, layer_idx)
            m = train_layer(
                X_a, X_b, layer_idx, ttype, out_dir,
                hidden_dim=args.hidden_dim,
                num_hidden=args.num_hidden,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=args.device,
            )
            metrics_list.append(m)
            print(f"  -> test_loss={m['test_loss']:.4f}, cos_sim={m['test_cos_sim']:.4f}")

        elapsed = time.time() - t0
        print(f"\nFinished {ttype} in {elapsed:.1f}s")

        # Save metrics (without large loss arrays for brevity)
        summary = [{k: v for k, v in m.items() if k not in ("train_losses", "val_losses")}
                   for m in metrics_list]
        with open(out_dir / f"metrics_{ttype}.json", "w") as f:
            json.dump(summary, f, indent=2)

        plot_loss_curves(metrics_list, out_dir, ttype)
        plot_test_summary(metrics_list, out_dir, ttype)
        all_metrics[ttype] = summary

    # Print overall summary
    print("\n=== Overall Summary ===")
    for ttype, summary in all_metrics.items():
        avg_cos = np.mean([m["test_cos_sim"] for m in summary])
        avg_mse = np.mean([m["test_loss"] for m in summary])
        print(f"  {ttype}: avg_cos_sim={avg_cos:.4f}, avg_mse={avg_mse:.4f}")


if __name__ == "__main__":
    main()
