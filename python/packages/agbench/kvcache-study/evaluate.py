"""
Phase 4: Evaluate MLP-transformed KV cache by injecting it into model B at inference time.

Instead of running the real prefill on model B, we:
1. Run prefill on model A (small) to get KV tensors
2. Apply trained MLPs to transform them to model B's space
3. Inject into model B and run generation
4. Compare outputs vs. baseline (real model B prefill)

Usage:
    python evaluate.py \\
        --model_a Qwen/Qwen3-1.7B \\
        --model_b Qwen/Qwen3-4B \\
        --mlp_dir results/mlp \\
        --output_dir results/eval \\
        --max_problems 20
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_mlp import LayerMLP


# ---------------------------------------------------------------------------
# Load trained MLPs
# ---------------------------------------------------------------------------

def load_mlps(mlp_dir: Path, tensor_type: str, device: str) -> dict[int, nn.Module]:
    """Load all per-layer MLPs of a given tensor_type. Returns {layer_idx: model}."""
    mlps = {}
    for ckpt_path in sorted(mlp_dir.glob(f"layer_*_{tensor_type}.pt")):
        # filename: layer_02_keys.pt
        layer_idx = int(ckpt_path.stem.split("_")[1])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        mlp = LayerMLP(
            ckpt["in_dim"], ckpt["out_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_hidden=ckpt["num_hidden"],
        ).to(device)
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.eval()
        mlps[layer_idx] = mlp
    if not mlps:
        raise FileNotFoundError(f"No MLP checkpoints found in {mlp_dir} for type '{tensor_type}'")
    return mlps


# ---------------------------------------------------------------------------
# KV cache injection
# ---------------------------------------------------------------------------

def transform_kv_cache(
    past_kv: tuple,
    key_mlps: dict[int, nn.Module],
    val_mlps: dict[int, nn.Module],
    device: str,
) -> tuple:
    """
    Apply trained MLPs to transform past_key_values from model A space to model B space.

    past_kv: tuple of (key, value) per layer, each [1, num_heads_a, seq_len, head_dim_a]
    Returns: tuple of (key, value) per layer in model B dimensions.
    """
    new_kv = []
    for layer_idx, (k, v) in enumerate(past_kv):
        # k, v: [1, num_heads_a, seq_len, head_dim_a]
        batch, num_heads_a, seq_len, head_dim_a = k.shape

        def transform_tensor(t: torch.Tensor, mlps: dict) -> torch.Tensor:
            if layer_idx not in mlps:
                return t  # no MLP for this layer, pass through (may cause shape mismatch)
            mlp = mlps[layer_idx]
            # Mean-pool over seq_len, flatten, transform, then broadcast back
            # Shape: [1, num_heads_a, seq_len, head_dim_a] -> [1, d_a]
            pooled = t.mean(dim=2).reshape(batch, -1).to(device, dtype=torch.float32)
            out = mlp(pooled)  # [1, d_b]
            # Reshape to model B's expected [1, num_heads_b, seq_len, head_dim_b]
            # We need to know num_heads_b and head_dim_b from the MLP's out_dim
            out_dim = out.shape[-1]
            # Assume head_dim_b matches (num_heads_b * head_dim_b = out_dim)
            # We broadcast the single pooled vector across all seq positions
            # out: [1, out_dim] -> [1, num_heads_b, head_dim_b] -> [1, num_heads_b, seq_len, head_dim_b]
            # Determine num_heads_b: assume same head_dim as model A for now
            # (exact factoring depends on model config — users can override)
            num_heads_b = out_dim // head_dim_a if out_dim % head_dim_a == 0 else 1
            head_dim_b = out_dim // num_heads_b
            out = out.reshape(batch, num_heads_b, head_dim_b)
            out = out.unsqueeze(2).expand(batch, num_heads_b, seq_len, head_dim_b)
            return out.contiguous()

        new_k = transform_tensor(k, key_mlps)
        new_v = transform_tensor(v, val_mlps)
        new_kv.append((new_k, new_v))

    return tuple(new_kv)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_real_prefill(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    device: str = "cuda",
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def generate_with_transformed_kv(
    model_a,
    model_b,
    tokenizer_a,
    tokenizer_b,
    prompt: str,
    key_mlps: dict,
    val_mlps: dict,
    max_new_tokens: int = 200,
    device: str = "cuda",
) -> str:
    """
    Run prefill on model_a, transform KV, inject into model_b, then generate.
    """
    inputs_a = tokenizer_a(prompt, return_tensors="pt").to(device)
    out_a = model_a(**inputs_a, use_cache=True)
    transformed_kv = transform_kv_cache(out_a.past_key_values, key_mlps, val_mlps, device)

    # Re-tokenize for model_b (may differ slightly)
    inputs_b = tokenizer_b(prompt, return_tensors="pt").to(device)
    input_ids = inputs_b["input_ids"]

    # Generate using the injected KV cache as the prefix
    # We pass past_key_values so the model treats the sequence as already prefilled
    out = model_b.generate(
        input_ids=input_ids[:, -1:],  # feed only the last token; prefix is in cache
        past_key_values=transformed_kv,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer_b.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_token_overlap(pred: str, ref: str) -> float:
    pred_toks = set(pred.split())
    ref_toks = set(ref.split())
    if not ref_toks:
        return 0.0
    return len(pred_toks & ref_toks) / len(ref_toks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP-transformed KV cache injection.")
    parser.add_argument("--model_a", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_b", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--mlp_dir", type=str, default="results/mlp")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--max_problems", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    mlp_dir = Path(args.mlp_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    problems_file = Path(args.data_dir) / "problems.json"
    with open(problems_file) as f:
        problems = json.load(f)
    problems = problems[:args.max_problems]
    print(f"Evaluating on {len(problems)} problems.")

    # Load MLPs
    print("Loading MLPs ...")
    key_mlps = load_mlps(mlp_dir, "keys", args.device)
    val_mlps = load_mlps(mlp_dir, "values", args.device)
    print(f"  Loaded {len(key_mlps)} key MLPs, {len(val_mlps)} value MLPs.")

    # Load models
    print(f"Loading {args.model_a} ...")
    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True
    ).eval()

    print(f"Loading {args.model_b} ...")
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True
    ).eval()

    results = []
    for i, problem in enumerate(problems):
        prompt = problem["prompt"]
        task_id = problem["task_id"]
        print(f"\n[{i+1}/{len(problems)}] {task_id}")

        # Baseline: real model_b prefill
        baseline = generate_with_real_prefill(
            model_b, tok_b, prompt,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )

        # Transformed: model_a prefill -> MLP -> inject into model_b
        try:
            transformed = generate_with_transformed_kv(
                model_a, model_b, tok_a, tok_b, prompt,
                key_mlps, val_mlps,
                max_new_tokens=args.max_new_tokens, device=args.device,
            )
        except Exception as e:
            print(f"  ERROR during transformed generation: {e}")
            transformed = ""

        overlap = compute_token_overlap(transformed, baseline)
        print(f"  token_overlap={overlap:.3f}")
        print(f"  baseline[:100]:    {baseline[:100]!r}")
        print(f"  transformed[:100]: {transformed[:100]!r}")

        results.append({
            "task_id": task_id,
            "baseline": baseline,
            "transformed": transformed,
            "token_overlap": overlap,
        })

    # Save results
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    avg_overlap = sum(r["token_overlap"] for r in results) / len(results)
    print(f"\nAverage token overlap (transformed vs baseline): {avg_overlap:.4f}")


if __name__ == "__main__":
    main()
