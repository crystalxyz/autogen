"""LMEval scenario runner.

Invokes lm-evaluation-harness programmatically against the SGLang server
whose base_url is injected into config.yaml by run_bench.py. One lm-eval
task per agbench scenario instance. Strictly serial: num_concurrent=1 and
batch_size=1 (the latter is also enforced by local-chat-completions itself).

Prints agbench's standard success/time/token markers so the default scorer
and tabulator pick up results without a custom scorer.
"""

import json
import os
import sys
import time
import traceback

# Force line-buffered stdout/stderr so every print() flushes to
# console_log.txt the moment it is emitted, not in 4-8 KB chunks.
# Critical for long scenarios where the process can be killed by an
# external timeout before block-buffered prints would otherwise reach
# disk. Independent of and complementary to PYTHONUNBUFFERED.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except AttributeError:
    pass

import yaml


# --- per-request latency capture -----------------------------------------
# lm-eval has no built-in latency tracking. We monkey-patch the
# TemplateAPI.model_call method (used by local-chat-completions for both
# generation and loglikelihood) and record one entry per HTTP call.
# With num_concurrent=1 and batch_size=1 this order matches the task's
# request order, so the i-th entry corresponds to the i-th dataset row
# in generation tasks.
_LATENCIES: list = []


def _install_latency_hook() -> None:
    from lm_eval.models.api_models import TemplateAPI

    orig = TemplateAPI.model_call

    def timed_model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        t0 = time.perf_counter()
        status = "ok"
        err: "str | None" = None
        try:
            out = orig(self, messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs)
            return out
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            t1 = time.perf_counter()
            _LATENCIES.append({
                "index": len(_LATENCIES),
                "start": t0,
                "end": t1,
                "elapsed": t1 - t0,
                "generate": generate,
                "status": status,
                "error": err,
                "num_messages": len(messages) if messages is not None else 0,
            })

    TemplateAPI.model_call = timed_model_call


def _latency_summary(latencies: list) -> dict:
    if not latencies:
        return {}
    ok = [e["elapsed"] for e in latencies if e["status"] == "ok"]
    if not ok:
        return {"n": 0, "n_error": len(latencies)}
    ok_sorted = sorted(ok)
    n = len(ok_sorted)

    def pct(q: float) -> float:
        return ok_sorted[min(n - 1, int(q * n))]

    return {
        "n": n,
        "n_error": len(latencies) - n,
        "mean": sum(ok_sorted) / n,
        "min": ok_sorted[0],
        "max": ok_sorted[-1],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "total": sum(ok_sorted),
    }
# -------------------------------------------------------------------------


def read_placeholder(path: str, default: str) -> str:
    """Read a substitution file; fall back to `default` if the placeholder
    was never substituted (shouldn't happen via agbench but keeps
    standalone runs workable)."""
    with open(path, "rt") as fh:
        value = fh.read().strip()
    if value.startswith("__") and value.endswith("__"):
        return default
    return value


def main() -> None:
    # --- 1. Load the backend config injected by run_bench.py ---------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model_config"]["config"]
    model = mc["model"]
    base_url = mc.get("base_url")
    api_key = mc.get("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))

    if not base_url:
        raise RuntimeError(
            "config.yaml has no model_config.config.base_url. "
            "run_bench.py should have injected one pointing at the SGLang server."
        )

    # lm-eval's local-chat-completions adapter reads OPENAI_API_KEY from env.
    os.environ["OPENAI_API_KEY"] = api_key

    # --- 2. Load the per-task substitutions --------------------------------
    task_name = read_placeholder("task_name.txt", "gsm8k")
    num_fewshot_str = read_placeholder("fewshot.txt", "0")
    limit_str = read_placeholder("limit.txt", "0")

    try:
        num_fewshot = int(num_fewshot_str)
    except ValueError:
        num_fewshot = 0
    try:
        limit_val: "int | None" = int(limit_str)
        if limit_val <= 0:
            limit_val = None
    except ValueError:
        limit_val = None

    print(f"[lmeval_config] model={model} base_url={base_url} task={task_name}"
          f" num_fewshot={num_fewshot} limit={limit_val}")

    # --- 3. Run lm-eval ----------------------------------------------------
    # Build model_args. SGLang's chat endpoint lives at /v1/chat/completions.
    # base_url from run_bench.py is http://host:port/v1, so append the rest.
    chat_url = base_url.rstrip("/") + "/chat/completions"
    model_args = ",".join([
        f"model={model}",
        f"base_url={chat_url}",
        "num_concurrent=1",       # serial: one in-flight request
        "max_retries=3",
        "tokenized_requests=False",
        # Raise generation ceiling well above lm-eval task defaults
        # (gsm8k defaults to ~256) so thinking-mode models have room to
        # finish reasoning and emit the final answer line.
        "max_gen_toks=8192",
    ])

    start_time = time.time()
    success = False
    results: dict = {}
    try:
        # Import here so import errors surface in console_log.txt after we
        # already printed the config line above.
        from lm_eval import simple_evaluate  # type: ignore

        _install_latency_hook()

        results = simple_evaluate(
            model="local-chat-completions",
            model_args=model_args,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=1,         # serial: no client-side batching
            limit=limit_val,
            apply_chat_template=True,
            log_samples=True,     # capture per-sample prompts + responses
        ) or {}

        task_results = results.get("results", {}).get(task_name)
        if task_results is None:
            print(f"[lmeval_error] no results returned for task={task_name}")
        else:
            # Emit a grep-friendly metric line per non-stderr metric.
            for metric, value in task_results.items():
                if metric in ("alias",) or metric.endswith("_stderr,none"):
                    continue
                stderr_key = metric.replace(",", "_stderr,", 1) if "," in metric else f"{metric}_stderr"
                stderr = task_results.get(stderr_key, "")
                print(f"[lmeval_metric] task={task_name} metric={metric}"
                      f" value={value} stderr={stderr}")
            success = True
    except Exception:
        print("[lmeval_error] simple_evaluate raised an exception:")
        traceback.print_exc()
        success = False

    end_time = time.time()
    elapsed = end_time - start_time

    # --- 4. Attach per-request latencies & summary to the results ---------
    latency_summary = _latency_summary(_LATENCIES)
    if isinstance(results, dict):
        results["latencies"] = _LATENCIES
        results["latency_stats"] = latency_summary

    if latency_summary:
        print(
            f"[lmeval_latency] n={latency_summary['n']} "
            f"mean={latency_summary['mean']:.3f}s "
            f"p50={latency_summary['p50']:.3f}s "
            f"p90={latency_summary['p90']:.3f}s "
            f"p99={latency_summary['p99']:.3f}s "
            f"min={latency_summary['min']:.3f}s "
            f"max={latency_summary['max']:.3f}s "
            f"total={latency_summary['total']:.3f}s"
        )

    # --- 5. Persist full lm-eval results -----------------------------------
    try:
        with open("lm_eval_result.json", "w") as fh:
            # results contains non-JSON-serializable bits (e.g. numpy) in
            # some versions — default=str keeps it writable.
            json.dump(results, fh, indent=2, default=str)
    except Exception:
        traceback.print_exc()

    # --- 5. Emit agbench-standard markers ----------------------------------
    # Timing marker (picked up by default_timer regex: '(\w+) execution time: (\d+\.\d+)')
    print(f"LMEval execution time: {elapsed:.2f} seconds")

    # Token usage marker — reuse the same format as HumanEval/AgentChat so
    # existing token_usage_stat.py scripts keep working. lm-eval exposes
    # accumulated counts on the model adapter.
    try:
        from lm_eval.api.registry import get_model  # noqa: F401
        # Not a reliable way to fetch back the model object post-evaluate.
        # Instead, inspect results['config'] / results.get('n-shot') if
        # present, or fall back silently.
        usage = results.get("samples") or {}
        # No-op by default; presence of usage fields depends on lm-eval version.
        if isinstance(usage, dict) and usage:
            pass
    except Exception:
        pass

    # Success / failure marker (picked up by default_scorer).
    if success:
        print("ALL TESTS PASSED !#!#")
    else:
        print("SOME TESTS FAILED !#!#")

    # Final hard flush before exit, in case anything is still pending.
    sys.stdout.flush()
    sys.stderr.flush()

    # Non-success exit so the outer runner also sees a failure for tabulation.
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
