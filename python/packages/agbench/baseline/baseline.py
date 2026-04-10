"""
Evaluate Qwen3 on HumanEval Benchmark

Runs a single model on the full HumanEval dataset with greedy decoding,
extracts code from model output, and evaluates correctness.
"""

import os
import re
import signal
import json
from contextlib import contextmanager
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
NUM_SAMPLES = None  # None = all 164
TIMEOUT = 5
DTYPE = "auto"  # "auto", "float16", "bfloat16", "float32"
VERBOSE = True


# ── Sandboxed Execution ───────────────────────────────────────────────────────

class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager to limit execution time (Unix only)."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def check_correctness(problem: dict, code: str, timeout: int = 5) -> dict:
    """Run the extracted code against the HumanEval test cases."""
    task_id = problem["task_id"]
    full_code = code + "\n\n" + problem["test"] + f"\ncheck({problem['entry_point']})\n"

    try:
        exec_globals = {}
        with time_limit(timeout):
            exec(full_code, exec_globals)
        return {"task_id": task_id, "passed": True, "error": None}
    except TimeoutError:
        return {"task_id": task_id, "passed": False, "error": "Timed out"}
    except Exception as e:
        return {"task_id": task_id, "passed": False, "error": str(e)}


# ── Completion Extraction & Prompt Formatting ─────────────────────────────────

def extract_code(generated_text: str) -> str:
    """Extract python code from model output. Looks for ```python blocks, falls back to raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def format_prompt(problem_prompt: str) -> str:
    """Wrap the HumanEval prompt with a chat instruction."""
    instruction = (
        "Complete the following Python function. "
        "Return ONLY the complete function implementation in a python code block.\n\n"
    )
    return instruction + problem_prompt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[DTYPE]

    print(f"Model: {MODEL_NAME} | Device: {device} | Dtype: {DTYPE}")
    print(f"Max tokens: {MAX_NEW_TOKENS} | Temperature: {TEMPERATURE} | Timeout: {TIMEOUT}s")

    # Load model & tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load dataset
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    if NUM_SAMPLES is not None:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    print(f"{len(dataset)} problems loaded")

    # Run evaluation
    results = []
    passed = 0

    for i, problem in enumerate(dataset):
        task_id = problem["task_id"]
        formatted_prompt = format_prompt(problem["prompt"])

        messages = [{"role": "user", "content": formatted_prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": TEMPERATURE > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if TEMPERATURE > 0:
            gen_kwargs["temperature"] = TEMPERATURE

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        code = extract_code(generated_text)

        result = check_correctness(problem, code, timeout=TIMEOUT)
        results.append(result)

        if result["passed"]:
            passed += 1

        status = "PASS" if result["passed"] else "FAIL"
        acc_so_far = passed / (i + 1) * 100

        if VERBOSE:
            print(f"[{i+1}/{len(dataset)}] {task_id}: {status}")
            if result["error"]:
                print(f"  Error: {result['error']}")
        else:
            print(f"[{i+1}/{len(dataset)}] {task_id}: {status}  (running pass@1: {acc_so_far:.1f}%)")

    # Results summary
    total = len(results)
    accuracy = passed / total * 100 if total > 0 else 0.0

    print(f"\nModel: {MODEL_NAME}")
    print(f"Total: {total} | Passed: {passed} | Failed: {total - passed}")
    print(f"Pass@1: {accuracy:.2f}%")

    errors = [r for r in results if not r["passed"]]
    if errors:
        timeout_count = sum(1 for r in errors if r["error"] == "Timed out")
        runtime_count = len(errors) - timeout_count
        print(f"\nError breakdown: Timeouts={timeout_count}, Runtime errors={runtime_count}")


if __name__ == "__main__":
    main()
