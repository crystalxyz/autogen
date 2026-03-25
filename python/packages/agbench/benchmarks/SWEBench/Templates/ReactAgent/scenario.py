import asyncio
import json
import os
import re
import subprocess
import sys
import time

import yaml

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, ModelFamily
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext

# Task parameters (substituted by agbench)
INSTANCE_ID = "__INSTANCE_ID__"
REPO = "__REPO__"
BASE_COMMIT = "__BASE_COMMIT__"
FAIL_TO_PASS = json.loads('__FAIL_TO_PASS__')
PASS_TO_PASS = json.loads('__PASS_TO_PASS__')
VERSION = "__VERSION__"


def setup_repo(repo: str, base_commit: str, repo_dir: str) -> None:
    """Clone the repository and check out the base commit."""
    repo_url = f"https://github.com/{repo}.git"

    if not os.path.isdir(repo_dir):
        print(f"Cloning {repo_url} ...")
        subprocess.run(["git", "clone", "--depth=50", repo_url, repo_dir], check=True)

    print(f"Checking out base commit {base_commit} ...")
    subprocess.run(["git", "fetch", "origin", base_commit], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "checkout", base_commit], cwd=repo_dir, check=True)


def install_repo(repo_dir: str) -> None:
    """Install the repository in development mode."""
    print("Installing repository ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=repo_dir,
        check=True,
    )


def run_tests(repo_dir: str, test_ids: list[str]) -> tuple[int, int]:
    """Run the given test IDs with pytest. Returns (passed, total)."""
    if not test_ids:
        return 0, 0

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-header", "-rN", "-x", "--tb=no", "-q"] + test_ids,
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    print(output)

    # Parse pytest summary line: "X passed, Y failed ..."
    passed = 0
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))

    return passed, len(test_ids)


async def main() -> None:
    # Load model configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_client = ChatCompletionClient.load_component(config["model_config"])

    # Model context
    model_context: ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    # Set up the repository
    repo_dir = os.path.join(os.getcwd(), "repo")
    setup_repo(REPO, BASE_COMMIT, repo_dir)
    install_repo(repo_dir)

    # Build the test command for the executor
    test_ids_str = " ".join(FAIL_TO_PASS)

    start_time = time.time()

    # Coder
    coder_agent = MagenticOneCoderAgent(
        name="coder",
        model_client=model_client,
    )
    coder_agent._model_context = model_context  # type: ignore

    # Executor
    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir=repo_dir),
        sources=["coder"],
        repo_dir=repo_dir,
        fail_to_pass=FAIL_TO_PASS,
    )

    # Termination condition
    termination = TextMentionTermination(text="TERMINATE", sources=["executor"])

    # Define a team
    agent_team = RoundRobinGroupChat([coder_agent, executor], max_turns=12, termination_condition=termination)

    task = f"""Fix the following GitHub issue in the repository {REPO}.
The repository is checked out at: {repo_dir}

## Issue Description

"""
    with open("prompt.txt", "rt") as fh:
        task += fh.read().strip()

    task += f"""

## Instructions

1. Explore the repository structure to understand the codebase.
2. Identify the root cause of the issue.
3. Make the necessary code changes to fix the issue.
4. Do NOT modify any test files.
5. Provide your fix as a shell script in a ```sh code block that applies the changes using sed, patch, or python scripts.
   The script will be executed with the working directory set to: {repo_dir}
6. After the script runs, the following tests will be executed automatically to verify your fix:
   {test_ids_str}
"""

    # Run the team
    stream = agent_team.run_stream(task=task)
    await Console(stream)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Final evaluation: run FAIL_TO_PASS and PASS_TO_PASS tests
    print("\n--- Running FAIL_TO_PASS tests ---")
    fail_passed, fail_total = run_tests(repo_dir, FAIL_TO_PASS)

    print("\n--- Running PASS_TO_PASS tests ---")
    pass_passed, pass_total = run_tests(repo_dir, PASS_TO_PASS)

    # Report results
    print(f"\nFAIL_TO_PASS: {fail_passed}/{fail_total} passed")
    print(f"PASS_TO_PASS: {pass_passed}/{pass_total} passed")

    fail_ok = fail_total == 0 or fail_passed == fail_total
    pass_ok = pass_total == 0 or pass_passed == pass_total

    if fail_ok and pass_ok:
        print("ALL TESTS PASSED !#!#")
    else:
        print("SOME TESTS FAILED !#!#")

    print("SCENARIO.PY COMPLETE !#!#")


if __name__ == "__main__":
    asyncio.run(main())
