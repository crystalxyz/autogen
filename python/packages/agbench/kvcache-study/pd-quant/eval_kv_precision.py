"""
Eval harness: measure quality impact of FP8 KV cache vs BF16 KV cache.

Setup:
  - Weights are BF16 throughout (no weight quantization).
  - KV cache is either kept in BF16 ("baseline") or simulated FP8 e4m3
    via quantize-dequantize at every cache write ("fp8_kv").
  - Prefill computes K/V in BF16, writes to cache (optionally Q-DQ'd),
    decode reads from cache and runs attention.

This isolates the effect of KV cache precision on generation quality
without the confounds of weight quantization or kernel differences.

Tasks:
  - GSM8K (math reasoning, long decode)
  - MMLU subset (short answer, prefill-heavy)
  - HumanEval (code, structured decode)

Run:
  python eval_kv_precision.py --model Qwen/Qwen3-4B --task gsm8k --n 50 \
      --kv-mode baseline
  python eval_kv_precision.py --model Qwen/Qwen3-4B --task gsm8k --n 50 \
      --kv-mode fp8_kv

Then diff the two output JSON files.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# FP8 quant/dequant simulation
# ---------------------------------------------------------------------------
# We simulate FP8 e4m3 by casting BF16 -> float8_e4m3fn -> BF16. This matches
# what real FP8 KV cache does at the storage boundary: the values that come
# back out of the cache have been through one round-trip of FP8 quantization.
#
# Per-tensor scaling is used (single scalar per (layer, K-or-V) tensor),
# matching SGLang's default `fp8_e4m3` KV mode. If you want per-head or
# per-token scaling, change `compute_scale` accordingly.

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # ~448


def compute_scale(t: torch.Tensor) -> torch.Tensor:
    # Per-tensor amax scaling. Add small eps to avoid div-by-zero on empty
    # caches.
    amax = t.abs().amax().clamp(min=1e-12)
    return (FP8_MAX / amax).to(torch.float32)


def fp8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """Simulate writing `t` to an FP8 cache and reading it back."""
    if t.numel() == 0:
        return t
    scale = compute_scale(t)
    q = (t.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dq = q.to(torch.float32) / scale
    return dq.to(t.dtype)


# ---------------------------------------------------------------------------
# Cache wrapper that applies the round-trip on every update
# ---------------------------------------------------------------------------
# HF >=4.36 uses DynamicCache. We subclass it so that every time a layer
# writes new K/V to the cache, those new entries get the FP8 round-trip.
# Existing entries are left alone (they were already round-tripped when first
# written), so we don't accumulate quantization noise on prefill tokens.

from transformers.cache_utils import DynamicCache


class FP8DynamicCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = fp8_roundtrip(key_states)
        value_states = fp8_roundtrip(value_states)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def make_cache(kv_mode: str):
    if kv_mode == "baseline":
        return DynamicCache()
    elif kv_mode == "fp8_kv":
        return FP8DynamicCache()
    else:
        raise ValueError(f"unknown kv_mode {kv_mode}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(model, tok, prompt, kv_mode, max_new_tokens=512, device="cuda"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    cache = make_cache(kv_mode)
    out = model.generate(
        ids,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy for reproducibility
        temperature=1.0,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    gen = out[0, ids.shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def load_task(task: str, n: int):
    if task == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        ds = ds.select(range(min(n, len(ds))))
        items = []
        for ex in ds:
            prompt = (
                f"Solve the following problem. End with 'The answer is X.'\n\nProblem: {ex['question']}\n\nSolution:"
            )
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"prompt": prompt, "gold": gold, "raw": ex})
        return items

    if task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))
        items = []
        letters = ["A", "B", "C", "D"]
        for ex in ds:
            choices = "\n".join(f"{l}. {c}" for l, c in zip(letters, ex["choices"]))
            prompt = f"Question: {ex['question']}\n{choices}\nAnswer with a single letter A, B, C, or D.\nAnswer:"
            items.append({"prompt": prompt, "gold": letters[ex["answer"]], "raw": ex})
        return items

    if task == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        ds = ds.select(range(min(n, len(ds))))
        items = []
        for ex in ds:
            prompt = ex["prompt"]
            items.append({"prompt": prompt, "gold": ex["test"], "raw": ex})
        return items

    raise ValueError(f"unknown task {task}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def extract_gsm8k_answer(text: str):
    # Look for "The answer is X." then fall back to last number.
    m = re.search(r"answer is\s*\$?(-?[0-9.,]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def extract_mmlu_answer(text: str):
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else ""


def score_item(task, gold, output):
    if task == "gsm8k":
        pred = extract_gsm8k_answer(output)
        try:
            return float(pred) == float(gold)
        except ValueError:
            return False
    if task == "mmlu":
        return extract_mmlu_answer(output) == gold
    if task == "humaneval":
        # Real pass@1 needs sandbox execution. For a quick proxy, just
        # record outputs and score offline. Return None here.
        return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--task", choices=["gsm8k", "mmlu", "humaneval"], required=True)
    p.add_argument("--kv-mode", choices=["baseline", "fp8_kv"], required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"Loading {args.model} in BF16 ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",  # works on H100; flash_attention_2 also fine
    )
    model.eval()

    items = load_task(args.task, args.n)
    print(f"Task={args.task}  n={len(items)}  kv_mode={args.kv_mode}")

    results = []
    correct = 0
    counted = 0
    t0 = time.time()
    for i, item in enumerate(items):
        out = generate(
            model,
            tok,
            item["prompt"],
            kv_mode=args.kv_mode,
            max_new_tokens=args.max_new_tokens,
        )
        ok = score_item(args.task, item["gold"], out)
        if ok is not None:
            counted += 1
            correct += int(ok)
        results.append(
            {
                "idx": i,
                "prompt": item["prompt"],
                "gold": item["gold"],
                "output": out,
                "correct": ok,
            }
        )
        if (i + 1) % 10 == 0:
            acc = correct / counted if counted else float("nan")
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(items)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    acc = correct / counted if counted else None
    summary = {
        "model": args.model,
        "task": args.task,
        "kv_mode": args.kv_mode,
        "n": len(items),
        "n_scored": counted,
        "accuracy": acc,
        "elapsed_sec": elapsed,
    }
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    out_path = args.out or f"results_{args.task}_{args.kv_mode}.json"
    Path(out_path).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
