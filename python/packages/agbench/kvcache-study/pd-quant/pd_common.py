"""
Shared helpers for the PD-disagg quantization studies.

Two sibling scripts use this module:
  * eval.py          - weight quantization only (fp32 vs fp8 model weights)
  * eval_kvcache.py  - KV cache quantization only (single model, varying KV precision)

The split keeps each axis isolated so accuracy effects don't get conflated.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Evaluation passages (held fixed across both studies for comparability).
# Each passage is split into a prefix (prefilled) and a continuation
# (teacher-forced for perplexity).
# ---------------------------------------------------------------------------

PASSAGES = [
    "The future of artificial intelligence is shaped less by raw compute "
    "than by the design choices that determine which information a model "
    "can attend to, how that information is compressed, and how cheaply "
    "it can be moved between machines. As context windows grow, the key "
    "value cache becomes the dominant cost of serving large language "
    "models, and quantizing it is one of the most direct ways to reduce "
    "memory bandwidth and inter-node transfer.",

    "In a distant galaxy, a civilization discovered that their most "
    "precious resource was not energy or matter, but coherent memory. "
    "They built vast archives whose contents could be summoned in an "
    "instant, but only if each fragment was stored with just enough "
    "precision to remain faithful when reassembled by a distant reader.",

    "The key principles of distributed systems include partial failure, "
    "asynchronous communication, and the impossibility of a perfect "
    "global clock. Designers must therefore choose, on every code path, "
    "what to replicate, what to recompute, and what to throw away when "
    "a node disappears without warning.",
]


# ---------------------------------------------------------------------------
# KV-tensor diagnostic metrics (sanity signal for quantization error).
# ---------------------------------------------------------------------------

@dataclass
class QuantizationMetrics:
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    cosine_sim: float
    snr_db: float  # signal-to-noise ratio


def compute_metrics(original: torch.Tensor, quantized: torch.Tensor) -> QuantizationMetrics:
    """Compute quantization-error metrics between original and quantized tensors."""
    diff = original.float() - quantized.float()
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


def compute_kv_cache_size(kv_cache) -> Dict[str, float]:
    total_bytes = 0
    for i in range(len(kv_cache)):
        k, v = kv_cache[i]
        total_bytes += k.nelement() * k.element_size()
        total_bytes += v.nelement() * v.element_size()
    return {"total_mb": total_bytes / (1024 ** 2), "total_bytes": total_bytes}


# ---------------------------------------------------------------------------
# Prefill + perplexity scoring.
# ---------------------------------------------------------------------------

def run_prefill(model, input_ids: torch.Tensor, device: str):
    """Run prefill on `input_ids` and return (last_logits, kv_cache)."""
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    return outputs.logits, outputs.past_key_values


def score_continuation_perplexity(
    model,
    prefill_last_logits: torch.Tensor,
    kv_cache,
    target_ids: torch.Tensor,
    device: str,
) -> Tuple[float, float]:
    """Teacher-forced perplexity of `target_ids` under `model`, given a KV cache.

    The first target token is scored against the last prefill logit (which
    predicts the token immediately following the prefix). The remaining
    target tokens are scored in a single forward pass that consumes the
    (possibly-quantized) KV cache.

    Returns (mean_nll, perplexity).
    """
    target_ids = target_ids.to(device)
    assert target_ids.dim() == 2 and target_ids.size(0) == 1
    T = target_ids.size(1)
    if T == 0:
        return 0.0, 1.0

    log_softmax = torch.nn.functional.log_softmax
    nlls = []

    # Token 0: predicted by the last prefill logit, no extra forward needed.
    first_logits = prefill_last_logits[:, -1, :]  # [1, V]
    first_logp = log_softmax(first_logits.float(), dim=-1)
    nlls.append(-first_logp[0, target_ids[0, 0]].item())

    if T > 1:
        # Feed target_ids[:, :-1] through the decoder; its logits at each
        # position predict target_ids[:, 1:]. KV cache grows along the way.
        feed_ids = target_ids[:, :-1]
        with torch.no_grad():
            outputs = model(
                input_ids=feed_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
        logits = outputs.logits  # [1, T-1, V]
        logp = log_softmax(logits.float(), dim=-1)
        labels = target_ids[:, 1:]  # [1, T-1]
        token_nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        nlls.extend(token_nll[0].tolist())

    mean_nll = float(np.mean(nlls))
    ppl = float(np.exp(mean_nll))
    return mean_nll, ppl


def snapshot_kv(kv_cache):
    """Detach + clone KV cache layers into a list of (k, v) tuples.

    Useful so a quantized cache can be reconstructed multiple times without
    being mutated by intervening forward passes.
    """
    snapshot = []
    for i in range(len(kv_cache)):
        k, v = kv_cache[i]
        snapshot.append((k.clone(), v.clone()))
    return snapshot


def build_quantized_kv(snapshot, qdtype, ddtype) -> DynamicCache:
    """Build a fresh DynamicCache from a snapshot, optionally quantized.

    `qdtype=None` clones unchanged. Otherwise tensors are cast to `qdtype`
    (to apply quantization rounding) and then to `ddtype` (the dtype the
    attention kernel can actually consume — fp8 isn't directly supported,
    so fp16 is the typical choice).
    """
    cache = DynamicCache()
    for i, (k, v) in enumerate(snapshot):
        if qdtype is None:
            k_q, v_q = k.clone(), v.clone()
        else:
            actual_ddtype = ddtype if ddtype is not None else qdtype
            k_q = k.to(qdtype).to(actual_ddtype)
            v_q = v.to(qdtype).to(actual_ddtype)
        cache.update(k_q, v_q, i)
    return cache
