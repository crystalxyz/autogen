"""
Weight Quantization Study (Perplexity, SGLang server backend)

Same experiment as eval.py, but the two models live in **separate SGLang
servers** instead of being loaded in-process. This sidesteps HuggingFace
Transformers' fine-grained-FP8 / deep-gemm path (currently broken on
Blackwell aarch64) — SGLang has its own FP8 kernels.

Usage:

  1. Launch two SGLang servers (in separate terminals):

       python -m sglang.launch_server \\
           --model-path Qwen/Qwen3-0.6B --port 30001

       python -m sglang.launch_server \\
           --model-path Qwen/Qwen3-0.6B-FP8 --port 30002

  2. Run this script:

       python eval_two_servers.py \
           --baseline-url http://localhost:30000/v1 \
           --fp8-url      http://localhost:30001/v1

For each passage, we tokenize locally (HF), split into prefix + target,
then send the full passage to each server with echo=true, max_tokens=0,
logprobs=1. The server returns per-token logprobs of the prompt tokens;
we sum the target-region logprobs to get NLL, then exp(mean NLL) = ppl.
"""

import argparse
import time
from typing import Dict, List

import numpy as np
import requests
from transformers import AutoTokenizer

from pd_common import PASSAGES


def server_token_logprobs(
    base_url: str, model: str, prompt: str, timeout: float = 120.0
) -> List[float]:
    """Hit /v1/completions with echo=true, max_tokens=0, logprobs=1.

    Returns the per-token logprob list for the prompt tokens. The first
    token's logprob is typically None (no preceding context); we replace
    it with 0.0, but we never include it in the perplexity sum anyway
    because the prefix tokens are excluded by the caller.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "echo": True,
        "logprobs": 1,
        "temperature": 0.0,
    }
    r = requests.post(f"{base_url}/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    lp = data["choices"][0]["logprobs"]["token_logprobs"]
    # SGLang/OpenAI convention: first token has logprob == None
    return [0.0 if x is None else float(x) for x in lp]


def perplexity_from_logprobs(token_logprobs: List[float], n_prefix: int) -> tuple:
    """Compute (mean_nll, ppl) over the target region only.

    `token_logprobs[i]` is the logprob of token i given tokens 0..i-1.
    The target region is tokens [n_prefix, len), so we sum those.
    """
    target_lps = token_logprobs[n_prefix:]
    if len(target_lps) == 0:
        return 0.0, 1.0
    mean_nll = float(-np.mean(target_lps))
    ppl = float(np.exp(mean_nll))
    return mean_nll, ppl


def discover_model_id(base_url: str) -> str:
    """Ask the SGLang server for its served model id."""
    r = requests.get(f"{base_url}/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", default="http://localhost:30000/v1",
                        help="OpenAI-compatible base URL of the baseline (fp32/bf16) SGLang server")
    parser.add_argument("--fp8-url", default="http://localhost:30001/v1",
                        help="OpenAI-compatible base URL of the fp8-weight SGLang server")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                        help="HF tokenizer used to split passages into prefix/target token spans")
    parser.add_argument("--prefix-frac", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Baseline server: {args.baseline_url}")
    print(f"FP8 server:      {args.fp8_url}")
    print(f"Tokenizer:       {args.tokenizer}")

    base_model = discover_model_id(args.baseline_url)
    fp8_model = discover_model_id(args.fp8_url)
    print(f"Baseline model id: {base_model}")
    print(f"FP8 model id:      {fp8_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    sum_nll: Dict[str, float] = {"baseline": 0.0, "fp8": 0.0}
    sum_tok: Dict[str, int] = {"baseline": 0, "fp8": 0}

    for passage in PASSAGES:
        print(f"\n{'#'*70}")
        print(f"Passage: {passage[:60]!r}...")
        print(f"{'#'*70}")

        # Locally tokenize to find the prefix/target split. We assume the
        # server uses the same tokenizer (true when --tokenizer matches the
        # served model). We don't *send* token ids — we send the text and
        # let the server retokenize, but the count must match for slicing.
        full_ids = tokenizer(passage, add_special_tokens=False).input_ids
        n_total = len(full_ids)
        n_prefix = max(1, int(n_total * args.prefix_frac))
        n_target = n_total - n_prefix
        if n_target == 0:
            print("  (passage too short to score; skipping)")
            continue
        print(f"Total tokens: {n_total}, prefix: {n_prefix}, target: {n_target}")

        results = {}
        for label, url, model in (
            ("baseline", args.baseline_url, base_model),
            ("fp8",      args.fp8_url,      fp8_model),
        ):
            t0 = time.perf_counter()
            lps = server_token_logprobs(url, model, passage)
            elapsed = time.perf_counter() - t0

            if len(lps) != n_total:
                print(f"  WARN: {label} returned {len(lps)} token logprobs, "
                      f"local tokenizer counted {n_total}. Tokenizer mismatch?")

            nll, ppl = perplexity_from_logprobs(lps, n_prefix)
            results[label] = (nll, ppl, elapsed)
            sum_nll[label] += nll * n_target
            sum_tok[label] += n_target

        nll_b, ppl_b, t_b = results["baseline"]
        nll_q, ppl_q, t_q = results["fp8"]
        dppl = (ppl_q - ppl_b) / ppl_b * 100.0

        print(f"\n  {'Weights':<10} {'NLL':<12} {'PPL':<12} {'dPPL%':<10} {'Time(s)':<10}")
        print(f"  {'-'*54}")
        print(f"  {'baseline':<10} {nll_b:<12.4f} {ppl_b:<12.4f} {'—':<10} {t_b:<10.3f}")
        print(f"  {'fp8':<10} {nll_q:<12.4f} {ppl_q:<12.4f} {dppl:<+10.2f} {t_q:<10.3f}")

    # === Overall (token-weighted) summary ===
    print(f"\n{'='*70}")
    print("Overall perplexity (token-weighted across passages)")
    print(f"{'='*70}")
    overall_b = float(np.exp(sum_nll["baseline"] / max(1, sum_tok["baseline"])))
    overall_q = float(np.exp(sum_nll["fp8"] / max(1, sum_tok["fp8"])))
    delta = (overall_q - overall_b) / overall_b * 100.0
    print(f"  {'Weights':<10} {'PPL':<14} {'dPPL vs baseline':<18}")
    print(f"  {'-'*44}")
    print(f"  {'baseline':<10} {overall_b:<14.4f} {'—':<18}")
    print(f"  {'fp8':<10} {overall_q:<14.4f} {delta:<+18.2f}")


if __name__ == "__main__":
    main()
