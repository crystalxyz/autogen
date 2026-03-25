import asyncio
import json
import os
import subprocess
import sys

import yaml
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

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
    total = len(test_ids)
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))

    return passed, total


async def main() -> None:
    import re

    # Load model configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    file_surfer_client = ChatCompletionClient.load_component(config["file_surfer_client"])

    # Read the problem statement
    with open("prompt.txt", "rt") as fh:
        problem_statement = fh.read().strip()

    # Set up the repository
    repo_dir = os.path.join(os.getcwd(), "repo")
    setup_repo(REPO, BASE_COMMIT, repo_dir)
    install_repo(repo_dir)

    # Build task description for the agents
    task = f"""You are a software engineer working on a GitHub repository. Your task is to fix a bug or implement a feature described in the following GitHub issue.

Repository: {REPO}
Repository location on disk: {repo_dir}

## Issue Description

{problem_statement}

## Instructions

1. Explore the repository structure to understand the codebase.
2. Identify the root cause of the issue based on the problem statement.
3. Make the necessary code changes to fix the issue.
4. Ensure your changes are minimal and focused on the issue.
5. Do NOT modify any test files.
6. After making your changes, print "CHANGES COMPLETE" to signal completion.

Work directly with the files in the repository at: {repo_dir}
"""

    # Set up agents
    coder = MagenticOneCoderAgent(
        name="Coder",
        model_client=coder_client,
    )

    executor = CodeExecutorAgent(
        name="Executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir=repo_dir),
    )

    file_surfer = FileSurfer(
        name="FileSurfer",
        model_client=file_surfer_client,
    )

    team = MagenticOneGroupChat(
        [coder, executor, file_surfer],
        model_client=orchestrator_client,
        max_turns=30,
    )

    # Run the agent team
    stream = team.run_stream(task=task)
    await Console(stream)

    # Evaluate the fix by running the tests
    print("\n--- Running FAIL_TO_PASS tests ---")
    fail_passed, fail_total = run_tests(repo_dir, FAIL_TO_PASS)

    print("\n--- Running PASS_TO_PASS tests ---")
    pass_passed, pass_total = run_tests(repo_dir, PASS_TO_PASS)

    # Report results
    print(f"\nFAIL_TO_PASS: {fail_passed}/{fail_total} passed")
    print(f"PASS_TO_PASS: {pass_passed}/{pass_total} passed")

    # Success: all FAIL_TO_PASS tests now pass AND no PASS_TO_PASS regressions
    fail_ok = fail_total == 0 or fail_passed == fail_total
    pass_ok = pass_total == 0 or pass_passed == pass_total

    if fail_ok and pass_ok:
        print("ALL TESTS PASSED !#!#")
    else:
        print("SOME TESTS FAILED !#!#")

    print("SCENARIO.PY COMPLETE !#!#")


if __name__ == "__main__":
    asyncio.run(main())
