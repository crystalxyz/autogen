"""
PD Disaggregation KV Cache Quantization Study

Simulates prefill-decode disaggregation with KV cache quantization:
  Prefill:   Qwen3-0.6B (fp32) -> KV cache
  Transfer:  quantize KV to fp8 (simulating network transfer compression)
  Decode:    Qwen3-0.6B-FP8 consumes the quantized KV cache

Compares decode quality across: no quantization (fp32), fp16, fp8_e4m3, fp8_e5m2.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

from transformers.cache_utils import DynamicCache

def quantize_kv_cache(kv_cache, target_dtype, decode_dtype=None) -> DynamicCache:
    """Quantize KV cache to target_dtype, optionally cast to decode_dtype for model consumption.

    For fp8, attention kernels can't consume fp8 directly, so we quantize (to measure error)
    then cast to decode_dtype (e.g. fp16) for actual decode.
    """
    if decode_dtype is None:
        decode_dtype = target_dtype
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k, v = kv_cache[layer_idx]
        k_q = k.to(target_dtype).to(decode_dtype)
        v_q = v.to(target_dtype).to(decode_dtype)
        new_cache.update(k_q, v_q, layer_idx)
    return new_cache

@dataclass
class QuantizationMetrics:
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    cosine_sim: float
    snr_db: float  # signal-to-noise ratio


def compute_metrics(original: torch.Tensor, quantized: torch.Tensor) -> QuantizationMetrics:
    """Compute quantization error metrics between original and quantized tensors."""
    diff = (original.float() - quantized.float())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    orig_flat = original.float().flatten()
    quant_flat = quantized.float().flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
    ).item()

    signal_power = original.float().pow(2).mean().item()
    noise_power = diff.pow(2).mean().item()
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return QuantizationMetrics(
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        rmse=rmse,
        cosine_sim=cosine_sim,
        snr_db=snr_db,
    )



def run_prefill(model, tokenizer, prompt: str, device: str):
    """Run prefill and return logits + KV cache in original precision."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.logits, outputs.past_key_values, inputs["input_ids"]


def run_decode_step(model, input_id: torch.Tensor, kv_cache: Tuple, device: str):
    """Run a single decode step given a token and KV cache."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_id.to(device),
            past_key_values=kv_cache,
            use_cache=True,
        )
    return outputs.logits, outputs.past_key_values


def run_decode_loop(model, tokenizer, prefill_logits, kv_cache, max_new_tokens: int, device: str):
    """Run autoregressive decode for max_new_tokens steps."""
    next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token.item()]
    current_kv = kv_cache

    for _ in range(max_new_tokens - 1):
        logits, current_kv = run_decode_step(model, next_token, current_kv, device)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated_ids, current_kv


def analyze_kv_cache_stats(kv_cache, label: str):
    print(f"\n{'='*60}")
    print(f"KV Cache Stats: {label}")
    print(f"{'='*60}")
    for i in range(len(kv_cache)):
        k, v = kv_cache[i]
        print(f"  Layer {i:2d} | K: dtype={k.dtype}, shape={list(k.shape)}, "
              f"range=[{k.float().min().item():.4f}, {k.float().max().item():.4f}], "
              f"std={k.float().std().item():.4f}")
        print(f"           | V: dtype={v.dtype}, shape={list(v.shape)}, "
              f"range=[{v.float().min().item():.4f}, {v.float().max().item():.4f}], "
              f"std={v.float().std().item():.4f}")


def compute_kv_cache_size(kv_cache) -> Dict[str, float]:
    total_bytes = 0
    for i in range(len(kv_cache)):
        k, v = kv_cache[i]
        total_bytes += k.nelement() * k.element_size()
        total_bytes += v.nelement() * v.element_size()
    return {"total_mb": total_bytes / (1024 ** 2), "total_bytes": total_bytes}


def main():
    prefill_model_name = "Qwen/Qwen3-0.6B"
    decode_model_name = "Qwen/Qwen3-0.6B-FP8"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 50

    prompts = [
        "The future of artificial intelligence is",
        "In a distant galaxy, a civilization discovered that",
        "The key principles of distributed systems include",
    ]

    print(f"Prefill model: {prefill_model_name} (fp32)")
    print(f"Decode model:  {decode_model_name} (fp8 weights)")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(prefill_model_name)

    print("Loading prefill model (fp32)...")
    prefill_model = AutoModelForCausalLM.from_pretrained(
        prefill_model_name, dtype=torch.float32, device_map=device,
    )
    prefill_model.eval()

    print("Loading decode model (fp8)...")
    decode_model = AutoModelForCausalLM.from_pretrained(
        decode_model_name, device_map=device,
    )
    decode_model.eval()

    # Quantization configs: (label, quantize_dtype, decode_dtype)
    # fp8 must be upcast for attention; fp16 is sufficient since the FP8 model accepts it
    quant_configs = [
        ("fp32 (none)", None,                   None),
        # ("fp16",        torch.float16,           None),
        ("fp8_e4m3",    torch.float8_e4m3fn,     torch.float16),
        ("fp8_e5m2",    torch.float8_e5m2,       torch.float16),
    ]

    for prompt in prompts:
        print(f"\n{'#'*70}")
        print(f"Prompt: {prompt!r}")
        print(f"{'#'*70}")

        # === Prefill with fp32 model ===
        prefill_logits, kv_orig, input_ids = run_prefill(prefill_model, tokenizer, prompt, device)
        n_prefill_layers = len(kv_orig)
        size_orig = compute_kv_cache_size(kv_orig)
        print(f"Prefill tokens: {input_ids.shape[1]}, KV: {size_orig['total_mb']:.2f} MB, {n_prefill_layers} layers")

        # Snapshot prefill KV before any decode mutates it
        prefill_kv_snapshot = []
        for i in range(n_prefill_layers):
            k, v = kv_orig[i]
            prefill_kv_snapshot.append((k.clone(), v.clone()))

        # === Baseline: prefill model decodes its own fp32 KV ===
        t0 = time.perf_counter()
        ids_baseline, _ = run_decode_loop(prefill_model, tokenizer, prefill_logits, kv_orig, max_new_tokens, device)
        t_baseline = time.perf_counter() - t0
        text_baseline = tokenizer.decode(ids_baseline, skip_special_tokens=True)

        # === Summary table ===
        print(f"\n  {'Config':<14} {'Compress':<10} {'K SNR(dB)':<12} {'V SNR(dB)':<12} {'CosSim(K)':<12} {'CosSim(V)':<12} {'Match':<16} {'Time':<8}")
        print(f"  {'-'*98}")
        print(f"  {'baseline':<14} {'—':<10} {'—':<12} {'—':<12} {'—':<12} {'—':<12} {'—':<16} {t_baseline:<8.3f}")
        print(f"  (prefill fp32 model decodes its own KV)")
        print(f"  > {text_baseline}")

        for label, qdtype, ddtype in quant_configs:
            # Build quantized KV cache from snapshot
            kv_q = DynamicCache()
            for i, (k, v) in enumerate(prefill_kv_snapshot):
                if qdtype is not None:
                    actual_ddtype = ddtype if ddtype is not None else qdtype
                    k_q = k.to(qdtype).to(actual_ddtype)
                    v_q = v.to(qdtype).to(actual_ddtype)
                else:
                    k_q, v_q = k.clone(), v.clone()
                kv_q.update(k_q, v_q, i)

            # Compression ratio
            if qdtype is not None:
                elem_size = torch.tensor([], dtype=qdtype).element_size()
                eff_bytes = sum((k.nelement() + v.nelement()) * elem_size for k, v in prefill_kv_snapshot)
                ratio_str = f"{size_orig['total_bytes'] / eff_bytes:.1f}x"
            else:
                ratio_str = "1.0x"

            # Aggregate quantization error
            all_k_snr, all_v_snr, all_k_cos, all_v_cos = [], [], [], []
            for i in range(n_prefill_layers):
                k_orig_snap, v_orig_snap = prefill_kv_snapshot[i]
                k_q_i, v_q_i = kv_q[i]
                mk = compute_metrics(k_orig_snap, k_q_i)
                mv = compute_metrics(v_orig_snap, v_q_i)
                all_k_snr.append(mk.snr_db)
                all_v_snr.append(mv.snr_db)
                all_k_cos.append(mk.cosine_sim)
                all_v_cos.append(mv.cosine_sim)

            # Decode with FP8 model
            t0 = time.perf_counter()
            ids_q, _ = run_decode_loop(decode_model, tokenizer, prefill_logits, kv_q, max_new_tokens, device)
            t_q = time.perf_counter() - t0
            text_q = tokenizer.decode(ids_q, skip_special_tokens=True)

            # Compare against baseline
            if ids_baseline == ids_q:
                match_str = f"EXACT ({len(ids_baseline)}t)"
            else:
                min_len = min(len(ids_baseline), len(ids_q))
                d = next((j for j in range(min_len) if ids_baseline[j] != ids_q[j]), min_len)
                match_str = f"div@{d}/{len(ids_baseline)}"

            k_snr = np.mean(all_k_snr)
            v_snr = np.mean(all_v_snr)
            k_cos = np.mean(all_k_cos)
            v_cos = np.mean(all_v_cos)
            snr_k_str = f"{k_snr:.1f}" if not np.isnan(k_snr) else "NaN"
            snr_v_str = f"{v_snr:.1f}" if not np.isnan(v_snr) else "NaN"
            cos_k_str = f"{k_cos:.8f}" if not np.isnan(k_cos) else "NaN"
            cos_v_str = f"{v_cos:.8f}" if not np.isnan(v_cos) else "NaN"

            print(f"  {label:<14} {ratio_str:<10} {snr_k_str:<12} {snr_v_str:<12} "
                  f"{cos_k_str:<12} {cos_v_str:<12} {match_str:<16} {t_q:<8.3f}")
            print(f"  > {text_q}")


if __name__ == "__main__":
    main()