#!/usr/bin/env python3
"""
Collect KV-cache tensors from two models on a calibration dataset and save to disk.

Usage:
    python collect_kv_cross_model.py \
        --model1 qwen/qwen3-0.6b \
        --model2 qwen/qwen3-1.7b \
        --num-samples 100 \
        --output-dir /share/gupta/xz957/kvcache_tensors

export DATA=/share/gupta/xz957/kvcache_tensors/qwen_qwen3-0.6b_vs_qwen_qwen3-1.7b_n100

Output structure:
    <output_dir>/<model1>_vs_<model2>_n<num_samples>/
        metadata.json
        k_model1_layer00.pt   # [num_samples, 1, seq_len, d]
        v_model1_layer00.pt
        k_model2_layer00.pt
        v_model2_layer00.pt
        ...
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name, device="cuda", dtype=torch.float16, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def with_kv_hook(model, run_inference):
    """Run inference with hooks on k_proj and v_proj, collecting outputs."""
    collected_k = []
    collected_v = []

    def k_hook(module, input, output):
        collected_k.append(output.detach().cpu())

    def v_hook(module, input, output):
        collected_v.append(output.detach().cpu())

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(v_hook))

    run_inference()

    for h in hooks:
        h.remove()

    return collected_k, collected_v


def build_prompt(sample, context_char_limit=10000):
    """Build a TriviaQA prompt with long context."""
    context = str(sample["search_results"]["search_context"])[:context_char_limit]
    return f"Context: {context}\n Question: {sample['question']}\n Answer:"


def collect_kv(model, tokenizer, dataset, num_samples):
    """Collect KV outputs for num_samples from dataset."""
    def run():
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i == num_samples:
                    break
                print(f"  Processing sample {i}/{num_samples}", end="\r")
                prompt = build_prompt(sample)
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                model(inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda())
        print()

    return with_kv_hook(model, run)


def save_kv_tensors(k_outputs, v_outputs, num_samples, num_layers, save_path, model_tag):
    """Save per-layer KV tensors as list of tensors (variable seq_len per sample)."""
    for layer_idx in range(num_layers):
        k = [k_outputs[s * num_layers + layer_idx] for s in range(num_samples)]
        v = [v_outputs[s * num_layers + layer_idx] for s in range(num_samples)]
        torch.save(k, os.path.join(save_path, f"k_{model_tag}_layer{layer_idx:02d}.pt"))
        torch.save(v, os.path.join(save_path, f"v_{model_tag}_layer{layer_idx:02d}.pt"))
        print(f"  Saved {model_tag} layer {layer_idx:02d}: {num_samples} samples, e.g. k={k[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="Collect KV tensors from two models")
    parser.add_argument("--model1", required=True, help="First model name (e.g. qwen/qwen3-0.6b)")
    parser.add_argument("--model2", required=True, help="Second model name (e.g. qwen/qwen3-1.7b)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--output-dir", default="/share/gupta/xz957/kvcache_tensors",
                        help="Base output directory")
    parser.add_argument("--cache-dir", default="/share/gupta/xz957/huggingface",
                        help="HuggingFace cache directory")
    parser.add_argument("--context-limit", type=int, default=10000,
                        help="Character limit for context in prompts")
    parser.add_argument("--dataset", default="trivia_qa", help="Dataset name")
    parser.add_argument("--dataset-config", default="rc", help="Dataset config")
    parser.add_argument("--dataset-split", default="validation", help="Dataset split")
    args = parser.parse_args()

    # Output path
    tag = f"{args.model1.replace('/', '_')}_vs_{args.model2.replace('/', '_')}_n{args.num_samples}"
    save_path = os.path.join(args.output_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}/{args.dataset_config} ({args.dataset_split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split,
                           cache_dir=args.cache_dir)

    # Model 1
    print(f"\nLoading model 1: {args.model1}")
    model1, tok1 = load_model_and_tokenizer(args.model1, cache_dir=args.cache_dir)
    model1 = model1.eval().cuda()

    print(f"Collecting KV from {args.model1}...")
    k1, v1 = collect_kv(model1, tok1, dataset, args.num_samples)
    num_layers1 = len(k1) // args.num_samples
    print(f"  {num_layers1} layers, {args.num_samples} samples")

    print(f"Saving model1 tensors...")
    save_kv_tensors(k1, v1, args.num_samples, num_layers1, save_path, "model1")

    del model1, tok1, k1, v1
    torch.cuda.empty_cache()

    # Model 2
    print(f"\nLoading model 2: {args.model2}")
    model2, tok2 = load_model_and_tokenizer(args.model2, cache_dir=args.cache_dir)
    model2 = model2.eval().cuda()

    print(f"Collecting KV from {args.model2}...")
    k2, v2 = collect_kv(model2, tok2, dataset, args.num_samples)
    num_layers2 = len(k2) // args.num_samples
    print(f"  {num_layers2} layers, {args.num_samples} samples")

    print(f"Saving model2 tensors...")
    save_kv_tensors(k2, v2, args.num_samples, num_layers2, save_path, "model2")

    del model2, tok2, k2, v2
    torch.cuda.empty_cache()

    # Save metadata
    meta = {
        "model1": args.model1,
        "model2": args.model2,
        "num_samples": args.num_samples,
        "num_layers1": num_layers1,
        "num_layers2": num_layers2,
        "shared_layers": min(num_layers1, num_layers2),
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "context_char_limit": args.context_limit,
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved to {save_path}")
    print(f"Metadata: {json.dumps(meta, indent=2)}")
    print(f"Files: {len(os.listdir(save_path))} total")


if __name__ == "__main__":
    main()
