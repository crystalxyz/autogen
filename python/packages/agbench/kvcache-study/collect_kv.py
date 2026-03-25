"""
Phase 1: Collect prefill KV tensors from qwen3-1.7b and qwen3-4b for each HumanEval prompt.

Usage:
    python collect_kv.py --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B --output_dir data/
    python collect_kv.py --models Qwen/Qwen3-1.7B --output_dir data/ --max_problems 20
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_humaneval_prompts(max_problems: Optional[int] = None) -> list[dict]:
    """Load HumanEval prompts (prompt text only, no solutions)."""
    dataset = load_dataset("openai_humaneval", split="test")
    problems = [{"task_id": ex["task_id"], "prompt": ex["prompt"]} for ex in dataset]
    if max_problems is not None:
        problems = problems[:max_problems]
    return problems


def collect_kv_for_model(
    model_name: str,
    prompts: list[dict],
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 1,
) -> None:
    """
    Run prefill for each prompt and save KV tensors.

    Saves one file per problem: data/<model_slug>/<task_id>.pt
    Each file is a dict: {"keys": [layer_tensor, ...], "values": [layer_tensor, ...]}
    where each layer_tensor has shape [num_heads, seq_len, head_dim].
    """
    model_slug = model_name.replace("/", "_")
    save_dir = output_dir / model_slug
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # When device_map="auto", inputs must go to the model's first device, not "auto"
    input_device = device if device != "auto" else next(model.parameters()).device

    already_done = {p.stem for p in save_dir.glob("*.pt")}

    for i, problem in enumerate(prompts):
        task_id = problem["task_id"].replace("/", "_")
        if task_id in already_done:
            print(f"  [{i+1}/{len(prompts)}] {task_id} already exists, skipping.")
            continue

        prompt_text = problem["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )

        past_kv = outputs.past_key_values  # tuple of (key, value) per layer
        # Each key/value tensor shape: [batch, num_heads, seq_len, head_dim]
        keys = [layer[0].squeeze(0).cpu().to(torch.float32) for layer in past_kv]
        values = [layer[1].squeeze(0).cpu().to(torch.float32) for layer in past_kv]

        save_path = save_dir / f"{task_id}.pt"
        torch.save({"keys": keys, "values": values}, save_path)
        print(f"  [{i+1}/{len(prompts)}] Saved {save_path} — {len(keys)} layers, seq_len={keys[0].shape[1]}")

    print(f"Done with {model_name}. Files in {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect prefill KV tensors for HumanEval prompts.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
        help="HuggingFace model IDs to collect from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/kvcache_data",
        help="Directory to save KV tensors.",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Limit to first N HumanEval problems (default: all 164).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_humaneval_prompts(max_problems=args.max_problems)
    print(f"Loaded {len(prompts)} HumanEval prompts.")

    # Save problem list for reference
    with open(output_dir / "problems.json", "w") as f:
        json.dump(prompts, f, indent=2)

    for model_name in args.models:
        collect_kv_for_model(
            model_name=model_name,
            prompts=prompts,
            output_dir=output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
