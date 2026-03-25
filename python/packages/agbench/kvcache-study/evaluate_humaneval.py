"""
Phase 4 (HumanEval): Evaluate hybrid KV cache on HumanEval pass@1.

Hybrid KV construction for each prompt:
  - Layers 0-27  : 1.7B prefill KV tensors, transformed per-token via trained MLPs
  - Layers 28-35 : 4B prefill KV tensors (real, from a partial 4B forward pass)

Generation is done entirely by Qwen3-4B with thinking disabled.
Three conditions are compared:
  1. baseline_4b  : full 4B real prefill (upper bound)
  2. baseline_mlp : transformed 1.7B KV for layers 0-27 only, layers 28-35 zeroed / skipped
  3. hybrid       : our method — transformed 1.7B for 0-27, real 4B for 28-35

Usage:
    python evaluate_humaneval.py \\
        --model_small Qwen/Qwen3-1.7B \\
        --model_large Qwen/Qwen3-4B \\
        --mlp_dir results/mlp \\
        --output_dir results/humaneval \\
        --max_problems 20
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_mlp import LayerMLP


# ---------------------------------------------------------------------------
# MLP loading
# ---------------------------------------------------------------------------


def load_mlps(mlp_dir: Path, tensor_type: str, device: str) -> dict[int, nn.Module]:
    mlps = {}
    for ckpt_path in sorted(mlp_dir.glob(f"layer_*_{tensor_type}.pt")):
        layer_idx = int(ckpt_path.stem.split("_")[1])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        mlp = LayerMLP(
            ckpt["in_dim"],
            ckpt["out_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_hidden=ckpt["num_hidden"],
        ).to(device)
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.eval()
        mlps[layer_idx] = mlp
    return mlps


# ---------------------------------------------------------------------------
# Per-token KV transformation
# ---------------------------------------------------------------------------


def transform_kv_per_token(
    past_kv_small: list,
    key_mlps: dict[int, nn.Module],
    val_mlps: dict[int, nn.Module],
    num_heads_large: int,
    head_dim_large: int,
    device: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply trained MLPs per token to transform small model KV to large model space.

    The MLPs were trained on mean-pooled (over seq) vectors of shape [num_heads * head_dim].
    We apply them per-token: for each token position, flatten heads → MLP → reshape.

    Returns list of (key, value) per layer, each [1, num_heads_large, seq_len, head_dim_large].
    """
    transformed = []
    for layer_idx, (k, v) in enumerate(past_kv_small):
        # k, v: [1, num_heads_small, seq_len, head_dim_small]
        if layer_idx not in key_mlps:
            transformed.append((k, v))
            continue

        batch, num_heads_s, seq_len, head_dim_s = k.shape
        in_dim = num_heads_s * head_dim_s

        def apply_per_token(t: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
            # t: [1, num_heads_s, seq_len, head_dim_s]
            # Rearrange to [seq_len, num_heads_s * head_dim_s]
            t_flat = t.squeeze(0).permute(1, 0, 2).reshape(seq_len, in_dim)
            t_flat = t_flat.to(device, dtype=torch.float32)
            with torch.no_grad():
                out = mlp(t_flat)  # [seq_len, num_heads_large * head_dim_large]
            # Reshape to [1, num_heads_large, seq_len, head_dim_large]
            out = out.reshape(seq_len, num_heads_large, head_dim_large)
            out = out.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads_large, seq_len, head_dim_large]
            return out.to(dtype=torch.float16)

        new_k = apply_per_token(k, key_mlps[layer_idx])
        new_v = apply_per_token(v, val_mlps[layer_idx])
        transformed.append((new_k, new_v))

    return transformed


# ---------------------------------------------------------------------------
# Partial 4B forward pass to get real KV for layers 28-35
# ---------------------------------------------------------------------------


def kv_to_list(past_kv) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert DynamicCache or tuple-of-tuples to a plain list of (k, v) tuples."""
    from transformers import DynamicCache

    if isinstance(past_kv, DynamicCache):
        legacy = past_kv.to_legacy_cache()
        return list(legacy)
    return [(past_kv[i][0], past_kv[i][1]) for i in range(len(past_kv))]


def list_to_cache(kv_list: list[tuple[torch.Tensor, torch.Tensor]]):
    """Convert a list of (k, v) tuples to a DynamicCache for generation."""
    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(tuple(kv_list))


# ---------------------------------------------------------------------------
# Build hybrid KV cache
# ---------------------------------------------------------------------------


def build_hybrid_kv(
    model_small,
    model_large,
    tokenizer_small,
    tokenizer_large,
    prompt: str,
    key_mlps: dict,
    val_mlps: dict,
    device: str,
) -> tuple:
    """
    Returns a tuple of (key, value) pairs for all large-model layers:
      - Layers 0 .. num_small_layers-1 : MLP-transformed small model KV
      - Layers num_small_layers .. end  : real large model KV
    """
    # --- small model prefill (all tokens except last, so generate() sees one new token) ---
    inputs_small = tokenizer_small(prompt, return_tensors="pt").to(device)
    inputs_small["input_ids"] = inputs_small["input_ids"][:, :-1]
    inputs_small["attention_mask"] = inputs_small["attention_mask"][:, :-1]
    with torch.no_grad():
        out_small = model_small(**inputs_small, use_cache=True)

    small_kv = kv_to_list(out_small.past_key_values)
    num_small_layers = len(small_kv)

    # Infer large model head config from first layer of large model KV
    inputs_large = tokenizer_large(prompt, return_tensors="pt").to(device)
    inputs_large["input_ids"] = inputs_large["input_ids"][:, :-1]
    inputs_large["attention_mask"] = inputs_large["attention_mask"][:, :-1]
    with torch.no_grad():
        out_large = model_large(**inputs_large, use_cache=True)

    large_kv = kv_to_list(out_large.past_key_values)
    num_heads_large = large_kv[0][0].shape[1]
    head_dim_large = large_kv[0][0].shape[3]

    # Transform small KV per-token for layers 0..num_small_layers-1
    transformed_shallow = transform_kv_per_token(
        small_kv,
        key_mlps,
        val_mlps,
        num_heads_large,
        head_dim_large,
        device,
    )

    # Combine: transformed small for layers 0-27, real large for layers 28-35
    hybrid_list = transformed_shallow + large_kv[num_small_layers:]

    return list_to_cache(hybrid_list), list_to_cache(large_kv), num_small_layers


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_prompt(tokenizer, prompt_text: str) -> str:
    """Wrap HumanEval prompt in Qwen3 chat template with thinking disabled."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding assistant. "
                "Complete the given Python function. "
                "Output only the completed function body, no explanation or markdown."
            ),
        },
        {
            "role": "user",
            "content": f"Complete this Python function:\n\n{prompt_text}",
        },
    ]
    # enable_thinking=False disables Qwen3's chain-of-thought <think> block
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text


@torch.no_grad()
def generate_from_kv(
    model_large,
    tokenizer_large,
    prompt_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int,
    device: str,
) -> str:
    """Generate text from model_large using a pre-filled KV cache (manual loop)."""
    eos_token_id = tokenizer_large.eos_token_id
    cache_len = past_key_values[0][0].shape[2]  # seq_len in cache
    # print(f"Prompt ids: {prompt_ids.shape}, KV cache seq_len: {cache_len}")

    # Start with the last prompt token (not yet in cache)
    input_id = prompt_ids[:, -1:]  # [1, 1]
    generated_ids = []

    for step in range(max_new_tokens):
        # Build position_ids and cache_position for this single token
        pos = cache_len + step
        position_ids = torch.tensor([[pos]], device=device)
        cache_position = torch.tensor([pos], device=device)

        outputs = model_large(
            input_ids=input_id,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1)  # greedy

        if next_token.item() == eos_token_id:
            break

        generated_ids.append(next_token.item())
        input_id = next_token.unsqueeze(0)  # [1, 1]

    return tokenizer_large.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_baseline(
    model_large,
    tokenizer_large,
    prompt_ids: torch.Tensor,
    past_key_values: tuple,
    max_new_tokens: int,
    device: str,
) -> str:
    """Same as generate_from_kv but uses real 4B KV (baseline)."""
    return generate_from_kv(model_large, tokenizer_large, prompt_ids, past_key_values, max_new_tokens, device)


# ---------------------------------------------------------------------------
# Code extraction and HumanEval execution
# ---------------------------------------------------------------------------


def extract_code(text: str, prompt: str) -> str:
    """
    Extract the Python function body from the model's output.
    Strips markdown fences if present, then prepends the original prompt.
    """
    # Strip markdown code fences
    text = re.sub(r"```python\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # If the model re-stated the function signature, prepend any imports from prompt
    if text.startswith("def "):
        # Extract import lines from the original prompt
        import_lines = [line for line in prompt.splitlines() if line.startswith(("import ", "from "))]
        if import_lines:
            return "\n".join(import_lines) + "\n\n" + text
        return text

    # Otherwise prepend original prompt so we have a complete function
    return prompt + text


def run_humaneval_test(task_id: str, completion: str, test: str, entry_point: str) -> bool:
    """
    Execute the completion against HumanEval test cases in a subprocess.
    Returns True if all tests pass.
    """
    code = completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=10,
            capture_output=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="HumanEval evaluation with hybrid KV cache.")
    parser.add_argument("--model_small", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_large", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--mlp_dir", type=str, default="results/mlp")
    parser.add_argument("--output_dir", type=str, default="results/humaneval")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HumanEval
    dataset = load_dataset("openai_humaneval", split="test")
    problems = list(dataset)
    if args.max_problems:
        problems = problems[: args.max_problems]
    print(f"Evaluating on {len(problems)} HumanEval problems.")

    # Load MLPs
    mlp_dir = Path(args.mlp_dir)
    print("Loading MLPs ...")
    key_mlps = load_mlps(mlp_dir, "keys", args.device)
    val_mlps = load_mlps(mlp_dir, "values", args.device)
    num_small_layers = len(key_mlps)
    print(f"  {num_small_layers} key MLPs, {len(val_mlps)} value MLPs loaded.")

    # Load models
    print(f"Loading {args.model_small} ...")
    tok_small = AutoTokenizer.from_pretrained(args.model_small, trust_remote_code=True)
    model_small = AutoModelForCausalLM.from_pretrained(
        args.model_small,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    print(f"Loading {args.model_large} ...")
    tok_large = AutoTokenizer.from_pretrained(args.model_large, trust_remote_code=True)
    model_large = AutoModelForCausalLM.from_pretrained(
        args.model_large,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    results = []
    pass_counts = {"baseline_4b": 0, "hybrid": 0}

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt_text = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]
        print(f"\n[{i + 1}/{len(problems)}] {task_id}")

        # Build chat prompt
        chat_prompt = build_prompt(tok_large, prompt_text)
        print(f"Chat prompt length: {len(chat_prompt)}")
        prompt_ids = tok_large(chat_prompt, return_tensors="pt").input_ids.to(args.device)
        print(f"Prompt ids: {prompt_ids.shape}")

        # Build hybrid + baseline KV caches
        try:
            hybrid_kv, real_kv, n_small = build_hybrid_kv(
                model_small,
                model_large,
                tok_small,
                tok_large,
                chat_prompt,
                key_mlps,
                val_mlps,
                args.device,
            )
        except Exception as e:
            print(f"  KV build error: {e}")
            results.append({"task_id": task_id, "error": str(e)})
            continue

        # --- Baseline: full 4B real prefill ---
        try:
            print(f"Prompt length: {len(prompt_ids)}")
            baseline_text = generate_from_kv(
                model_large,
                tok_large,
                prompt_ids,
                real_kv,
                args.max_new_tokens,
                args.device,
            )
            print("successfully generated kv")
            baseline_code = extract_code(baseline_text, prompt_text)
            print("successfully extract code")
            baseline_pass = run_humaneval_test(task_id, baseline_code, test_code, entry_point)
        except Exception as e:
            print(f"  baseline error: {e}")
            traceback.print_exc()
            baseline_text, baseline_code, baseline_pass = "", "", False

        # --- Hybrid: transformed 1.7B (layers 0-27) + real 4B (layers 28-35) ---
        try:
            hybrid_text = generate_from_kv(
                model_large,
                tok_large,
                prompt_ids,
                hybrid_kv,
                args.max_new_tokens,
                args.device,
            )
            hybrid_code = extract_code(hybrid_text, prompt_text)
            hybrid_pass = run_humaneval_test(task_id, hybrid_code, test_code, entry_point)
        except Exception as e:
            print(f"  hybrid error: {e}")
            traceback.print_exc()
            hybrid_text, hybrid_code, hybrid_pass = "", "", False

        pass_counts["baseline_4b"] += int(baseline_pass)
        pass_counts["hybrid"] += int(hybrid_pass)

        print(f"  baseline_4b: {'PASS' if baseline_pass else 'FAIL'} | hybrid: {'PASS' if hybrid_pass else 'FAIL'}")

        results.append(
            {
                "task_id": task_id,
                "baseline_pass": baseline_pass,
                "hybrid_pass": hybrid_pass,
                "baseline_code": baseline_code,
                "hybrid_code": hybrid_code,
                "num_small_layers": n_small,
            }
        )

    # Save results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    n = len([r for r in results if "error" not in r])
    print(f"\n{'=' * 50}")
    print(f"Results over {n} problems:")
    print(
        f"  baseline_4b  pass@1 = {pass_counts['baseline_4b']}/{n} "
        f"({100 * pass_counts['baseline_4b'] / max(n, 1):.1f}%)"
    )
    print(f"  hybrid       pass@1 = {pass_counts['hybrid']}/{n} ({100 * pass_counts['hybrid'] / max(n, 1):.1f}%)")
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
