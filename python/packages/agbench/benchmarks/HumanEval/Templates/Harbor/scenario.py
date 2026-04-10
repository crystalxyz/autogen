"""
agbench scenario shim that runs a Harbor agent against a HumanEval problem.

Flow per task:
  1. Read prompt.txt / test.txt / custom_code_executor.py (already substituted
     by agbench's expand_scenario).
  2. Synthesize a Harbor task directory at ./harbor_task/ with:
       - task.toml         (copied from this template dir)
       - instruction.md    (the HumanEval prompt + a "write to /logs/agent/solution.py" hint)
       - environment/      (placeholder; ApptainerEnvironment ignores its contents)
       - tests/test.sh     (verifier — runs the candidate against the unit tests
                            and writes 1|0 to /logs/verifier/reward.txt)
       - tests/_check.py   (the HumanEval `def check(candidate):` block)
  3. Invoke `harbor run` (from harbor-env) with --environment-import-path
     pointed at apptainer_env:ApptainerEnvironment, passing the SGLang server
     base_url / api_key as agent env vars.
  4. Read the resulting reward.txt from the harbor jobs/ directory.
  5. Emit the agbench markers ("ALL TESTS PASSED !#!#",
     "AgentChat execution time: ... !#!#", "SCENARIO.PY COMPLETE !#!#") so the
     existing default_scorer/default_timer in tabulate_cmd.py work unchanged.

Configuration (in agbench config.yaml under a top-level `harbor:` key):
    harbor:
      cli: /home/xz957/.conda/envs/harbor-env/bin/harbor   # optional
      agent: mini-swe-agent
      image: docker://python:3.12-slim
      extra_args: []           # raw passthrough to `harbor run`
      env_passthrough: []      # extra agent env vars (KEY=VALUE strings)
"""

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


HARBOR_CLI_DEFAULT = "/home/xz957/.conda/envs/harbor-env/bin/harbor"
TEMPLATE_DIR = Path(__file__).resolve().parent


def _read(path: str) -> str:
    with open(path, "rt") as fh:
        return fh.read()


def _extract_entry_point() -> str:
    # custom_code_executor.py contains a single line `ENTRY_POINT = "<name>"`
    src = _read("custom_code_executor.py")
    ns: dict = {}
    exec(src, ns)
    return ns["ENTRY_POINT"]


def _extract_model_endpoint(model_config: dict) -> tuple[str, str, str]:
    """Pull (model_name, base_url, api_key) from an autogen ChatCompletionClient
    component config. Both flat and nested-config layouts are accepted."""
    cfg = model_config.get("config", model_config)
    model = cfg.get("model") or cfg.get("model_name") or "unknown"
    base_url = cfg.get("base_url") or cfg.get("api_base") or ""
    api_key = cfg.get("api_key") or "EMPTY"
    return model, base_url, api_key


def _build_harbor_task(harbor_task_dir: Path, prompt: str, entry_point: str, test_code: str) -> None:
    if harbor_task_dir.exists():
        shutil.rmtree(harbor_task_dir)
    (harbor_task_dir / "environment").mkdir(parents=True)
    (harbor_task_dir / "tests").mkdir(parents=True)

    shutil.copy(TEMPLATE_DIR / "task.toml", harbor_task_dir / "task.toml")
    # placeholder so the env dir is not empty (some Harbor codepaths expect a file)
    (harbor_task_dir / "environment" / ".gitkeep").write_text("")

    instruction = f"""\
# HumanEval task

Implement the Python function `{entry_point}` so it satisfies the
specification below. Write your **complete** solution (including any imports
and helper functions) to the file `/logs/agent/solution.py` inside the
container. Do not modify the function signature.

The grading harness will execute:

```python
exec(open("/logs/agent/solution.py").read())
check({entry_point})  # raises AssertionError on failure
```

Function specification:

```python
{prompt}
```

When you are done, the file `/logs/agent/solution.py` must define
`{entry_point}`. Then exit.
"""
    (harbor_task_dir / "instruction.md").write_text(instruction)

    # The HumanEval `check(candidate)` function lives here. test.sh imports it
    # together with the agent's solution.py and runs check(<entry_point>).
    (harbor_task_dir / "tests" / "_check.py").write_text(test_code)

    test_sh = f"""#!/bin/bash
set -u
mkdir -p /logs/verifier

SOL=/logs/agent/solution.py
if [ ! -f "$SOL" ]; then
  echo "FAIL: $SOL not found" > /logs/verifier/test-stdout.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

cat > /tmp/_run_check.py <<'PYEOF'
import sys, traceback
ns = {{}}
with open("/logs/agent/solution.py") as fh:
    exec(fh.read(), ns)
with open("/tests/_check.py") as fh:
    exec(fh.read(), ns)
try:
    ns["check"](ns[{entry_point!r}])
    print("OK")
    sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(1)
PYEOF

if python3 /tmp/_run_check.py > /logs/verifier/test-stdout.txt 2> /logs/verifier/test-stderr.txt; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
    test_sh_path = harbor_task_dir / "tests" / "test.sh"
    test_sh_path.write_text(test_sh)
    test_sh_path.chmod(0o755)


def _find_reward(jobs_dir: Path) -> float | None:
    matches = glob.glob(str(jobs_dir / "**" / "verifier" / "reward.txt"), recursive=True)
    if not matches:
        return None
    try:
        return float(Path(matches[0]).read_text().strip() or "0")
    except ValueError:
        return None


def main() -> int:
    config = yaml.safe_load(_read("config.yaml"))
    harbor_cfg = config.get("harbor", {}) or {}

    cli = harbor_cfg.get("cli") or os.environ.get("HARBOR_CLI", HARBOR_CLI_DEFAULT)
    agent = harbor_cfg.get("agent")  # one of harbor's built-ins; mutually exclusive with agent_import_path
    agent_import_path = harbor_cfg.get("agent_import_path")  # e.g. "example_agent:ExampleAgent"
    if not agent and not agent_import_path:
        agent = "mini-swe-agent"
    image = harbor_cfg.get("image", "docker://python:3.12-slim")
    extra_args = list(harbor_cfg.get("extra_args") or [])
    env_passthrough = list(harbor_cfg.get("env_passthrough") or [])
    # Some Harbor agents (mini-swe-agent in particular) require a litellm
    # provider prefix on the model name. For an OpenAI-compatible local server
    # like SGLang, the conventional prefix is "openai/". Custom external
    # agents that talk to the OpenAI client directly should set "" here.
    model_prefix = harbor_cfg.get("model_prefix", "openai/")

    raw_model, base_url, api_key = _extract_model_endpoint(config["model_config"])
    model = f"{model_prefix}{raw_model}" if model_prefix else raw_model
    if not base_url:
        print("WARNING: model_config has no base_url; agent will use provider default", file=sys.stderr)

    prompt = _read("prompt.txt")
    test_code = _read("test.txt")
    entry_point = _extract_entry_point()

    cwd = Path.cwd()
    harbor_task_dir = cwd / "harbor_task"
    jobs_dir = cwd / "harbor_jobs"
    if jobs_dir.exists():
        shutil.rmtree(jobs_dir)

    _build_harbor_task(harbor_task_dir, prompt, entry_point, test_code)

    cmd = [
        cli, "run",
        "-p", str(harbor_task_dir),
        "-m", model,
        "--environment-import-path", "apptainer_env:ApptainerEnvironment",
        "--ek", f"image={image}",
        "-o", str(jobs_dir),
        "-n", "1",
        "-k", "1",
        "--no-export-traces",
    ]
    if agent_import_path:
        cmd += ["--agent-import-path", agent_import_path]
    if agent:
        cmd += ["-a", agent]
    if base_url:
        cmd += ["--ae", f"OPENAI_BASE_URL={base_url}"]
        cmd += ["--ae", f"OPENAI_API_BASE={base_url}"]  # litellm alias
    cmd += ["--ae", f"OPENAI_API_KEY={api_key}"]
    for kv in env_passthrough:
        cmd += ["--ae", kv]
    cmd += list(extra_args)

    # Make ApptainerEnvironment importable to harbor's Python.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TEMPLATE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # Several built-in Harbor agents (mini-swe-agent, claude-code, ...) read
    # model endpoint config from os.environ rather than the --ae values, so
    # we set the standard vars here as well.
    if base_url:
        env["OPENAI_API_BASE"] = base_url
        env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"] = api_key
    env["MSWEA_API_KEY"] = api_key  # mini-swe-agent fallback

    print(f"[harbor-shim] launching: {' '.join(cmd)}", flush=True)
    start = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    print(f"[harbor-shim] harbor exit code: {proc.returncode}", flush=True)
    print(f"AgentChat execution time: {elapsed:.2f} seconds !#!#", flush=True)

    reward = _find_reward(jobs_dir)
    print(f"[harbor-shim] reward = {reward}", flush=True)
    if reward is not None and reward >= 1.0:
        print("ALL TESTS PASSED !#!#", flush=True)

    print("SCENARIO.PY COMPLETE !#!#", flush=True)
    # Always exit 0 — agbench's scorer reads the markers from the log, not the
    # exit code, and a non-zero exit would mark the task as a hard failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
