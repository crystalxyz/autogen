"""
Controlled workload executor with concurrency and rate limiting for agbench.
Implements Poisson process-based rate limiting similar to sglang bench_serving.py.
"""

import glob
import json
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .apptainer_instance import ApptainerInstance
from .execution import run_scenario_in_apptainer, run_scenario_natively
from .hooks import Event, EventType, get_hook_manager
from .scenario_utils import expand_scenario, get_scenario_env


# Get a random number generator for subsampling (shared with run_cmd.py)
import random

subsample_rng = random.Random(425)

# Success string patterns for checking scenario completion
SUCCESS_STRINGS = [
    "ALL TESTS PASSED !#!#",
]

COMPLETED_STRINGS = [
    "SCENARIO.PY COMPLETE !#!#",
]


def _cleanup_logs(results_dir: str) -> None:
    """
    Remove unnecessary files from results directory, keeping only console_log.txt.

    Files to keep:
    - console_log.txt: The main execution log

    All other files (scenario.py, config.yaml, requirements.txt, etc.) are removed
    as they are just copies of templates and not needed for analysis.

    Args:
        results_dir: Path to the results directory to clean up
    """
    # Files to keep
    keep_files = {"console_log.txt"}

    try:
        # Get all files in the directory
        all_files = glob.glob(os.path.join(results_dir, "*"))

        # Remove files that are not in the keep list
        for file_path in all_files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                if filename not in keep_files:
                    os.remove(file_path)
    except Exception as e:
        # Don't fail the run if cleanup fails, just print a warning
        print(f"Warning: Failed to clean up logs in {results_dir}: {e}")


def _check_scenario_success(results_dir: str) -> Optional[bool]:
    """
    Check if a scenario completed successfully by reading console_log.txt.

    This is the default scorer that checks for SUCCESS_STRINGS and COMPLETED_STRINGS.
    For benchmark-specific scoring (e.g. GAIA's FINAL ANSWER matching), use a custom
    scorer from the benchmark's custom_tabulate.py instead.

    Args:
        results_dir: Path to the repetition results directory

    Returns:
        True if successful, False if completed but failed, None if not completed
    """
    console_log = os.path.join(results_dir, "console_log.txt")
    if not os.path.isfile(console_log):
        return None

    try:
        with open(console_log, "rt") as fh:
            content = fh.read()

            # Check for explicit success strings
            for s in SUCCESS_STRINGS:
                if s in content:
                    return True

            # Check for completion without success
            for s in COMPLETED_STRINGS:
                if s in content:
                    return False

            # Not completed
            return None
    except OSError:
        return None


def _get_scenario_elapsed_time(results_dir: str) -> Optional[float]:
    """
    Get the elapsed time for a scenario from console_log.txt.

    Args:
        results_dir: Path to the repetition results directory

    Returns:
        Elapsed time in seconds, or None if not available
    """
    console_log = os.path.join(results_dir, "console_log.txt")
    if not os.path.isfile(console_log):
        return None

    try:
        with open(console_log, "rt") as fh:
            content = fh.read()
            # Look for timing pattern
            match = re.search(r"AgentChat execution time:\s*([\d.]+)", content)
            if match:
                return float(match.group(1))
            return None
    except (OSError, ValueError):
        return None


def _get_scenario_turns(results_dir: str) -> Optional[int]:
    """
    Get the number of conversation turns from console_log.txt.

    Counts occurrences of "TextMessage" in the log.

    Args:
        results_dir: Path to the repetition results directory

    Returns:
        Number of turns, or None if not available
    """
    console_log = os.path.join(results_dir, "console_log.txt")
    if not os.path.isfile(console_log):
        return None

    try:
        with open(console_log, "rt") as fh:
            content = fh.read()
            count = content.count("TextMessage")
            if count == 0:
                return None
            return max(count - 1, 0)
    except OSError:
        return None


def _extract_task_info(results_repetition: str) -> tuple[Optional[str], Optional[int]]:
    """
    Extract task_id and repetition_id from a results path.

    Args:
        results_repetition: Path like "Results/scenario_timestamp/task_id/rep_id"

    Returns:
        Tuple of (task_id, repetition_id)
    """
    try:
        parts = results_repetition.rstrip(os.sep).split(os.sep)
        if len(parts) >= 2:
            rep_id = int(parts[-1])
            task_id = parts[-2]
            return task_id, rep_id
    except (ValueError, IndexError):
        pass
    return None, None


def prepare_task_list(
    lines: List[str],
    scenario_name: str,
    scenario_dir: str,
    results_dir: str,
    n_repeats: int,
    config_file: Optional[str],
    subsample: Optional[float] = None,
    skip_scenario_subdir: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare a list of tasks from scenario lines.

    Args:
        lines: Lines from the scenario file
        scenario_name: Name of the scenario
        scenario_dir: Directory containing the scenario
        results_dir: Base results directory
        n_repeats: Number of repetitions per scenario
        config_file: Path to config file
        subsample: Subsample proportion or count

    Returns:
        List of task dictionaries ready for execution
    """
    # Subsample if needed
    if subsample is not None:
        n = 0
        if 0 <= subsample < 1:
            n = int(len(lines) * subsample + 0.5)
        else:
            n = int(subsample)
        n = max(0, min(n, len(lines)))
        lines = subsample_rng.sample(lines, n)

    tasks = []
    for line in lines:
        instance = json.loads(line)

        # Create result directories
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        # Results for the scenario (skip if already included in results_dir)
        if skip_scenario_subdir:
            results_scenario = results_dir
        else:
            results_scenario = os.path.join(results_dir, scenario_name)
            if not os.path.isdir(results_scenario):
                os.mkdir(results_scenario)

        results_instance = os.path.join(results_scenario, instance["id"])
        if not os.path.isdir(results_instance):
            os.mkdir(results_instance)

        # Create tasks for each repetition
        for i in range(n_repeats):
            results_repetition = os.path.join(results_instance, str(i))

            # Skip if already exists
            if os.path.isdir(results_repetition):
                print(f"Found folder {results_repetition} ... Skipping.")
                continue

            tasks.append(
                {
                    "scenario_dir": scenario_dir,
                    "instance": instance,
                    "results_repetition": results_repetition,
                    "config_file": config_file,
                }
            )

    return tasks


def create_task_runner(
    is_native: bool,
    apptainer_instance: Optional[ApptainerInstance],
    token_provider: Optional[Callable[[], str]],
    env_file: Optional[str],
    emit_events: bool = False,
    minimal_logs: bool = False,
    scorer: Optional[Callable[[str], Optional[bool]]] = None,
) -> Callable[[Dict[str, Any]], None]:
    """
    Create a task runner function with the specified execution environment.

    Args:
        is_native: Whether to run natively
        apptainer_instance: Apptainer instance to use (if not native)
        token_provider: Token provider function
        env_file: Environment file path
        emit_events: Whether to emit hook events for streaming
        minimal_logs: If True, clean up unnecessary files after execution, keeping only console_log.txt
        scorer: Optional custom scorer function (from benchmark's custom_tabulate.py).
                Takes a results directory path and returns True/False/None.
                If None, falls back to _check_scenario_success.

    Returns:
        A function that executes a single task
    """

    def run_task(task_info: Dict[str, Any]) -> None:
        """Run a single scenario task."""
        results_repetition = task_info["results_repetition"]

        # Extract task_id and repetition_id for events
        task_id, repetition_id = _extract_task_info(results_repetition)

        # Emit REPETITION_START event
        if emit_events:
            hook_manager = get_hook_manager()
            hook_manager.emit(
                Event(
                    event_type=EventType.REPETITION_START,
                    task_id=task_id,
                    repetition_id=repetition_id,
                )
            )

        start_time = time.time()

        try:
            print(f"Expanding scenario at {results_repetition}")
            # Expand the scenario
            expand_scenario(
                task_info["scenario_dir"], task_info["instance"], results_repetition, task_info["config_file"]
            )
            print(f"Finish expanding scenario at {results_repetition}")

            # Prepare the environment
            task_env = get_scenario_env(token_provider=token_provider, env_file=env_file)

            # Run the scenario
            if is_native:
                print(f"Will run natively for {results_repetition}")
                run_scenario_natively(results_repetition, task_env)
            elif apptainer_instance is not None:
                print(f"Will run in apptainer for {results_repetition}")
                run_scenario_in_apptainer(
                    results_repetition,
                    apptainer_instance=apptainer_instance,
                )

            # Check success, elapsed time, and turns
            elapsed_time = time.time() - start_time
            if scorer is not None:
                success = scorer(results_repetition)
            else:
                success = _check_scenario_success(results_repetition)
            logged_elapsed = _get_scenario_elapsed_time(results_repetition)
            if logged_elapsed is not None:
                elapsed_time = logged_elapsed
            turns = _get_scenario_turns(results_repetition)

            # Clean up unnecessary files if minimal_logs is enabled
            if minimal_logs:
                _cleanup_logs(results_repetition)

            # Emit REPETITION_END event
            if emit_events:
                hook_manager = get_hook_manager()
                hook_manager.emit(
                    Event(
                        event_type=EventType.REPETITION_END,
                        task_id=task_id,
                        repetition_id=repetition_id,
                        success=success,
                        elapsed_time=elapsed_time,
                        turns=turns,
                    )
                )

        except Exception as e:
            # On error, emit end event with failure
            elapsed_time = time.time() - start_time
            if emit_events:
                hook_manager = get_hook_manager()
                hook_manager.emit(
                    Event(
                        event_type=EventType.REPETITION_END,
                        task_id=task_id,
                        repetition_id=repetition_id,
                        success=False,
                        elapsed_time=elapsed_time,
                        turns=None,
                        metadata={"error": str(e)},
                    )
                )
            raise

    return run_task


def execute_with_poisson_rate(
    tasks: List[Dict[str, Any]],
    run_task: Callable[[Dict[str, Any]], None],
    num_concurrent: int,
    request_rate: Optional[float] = None,
) -> None:
    """
    Execute tasks with controlled concurrency and Poisson-distributed rate limiting.

    Uses exponential distribution for inter-arrival times, which models a Poisson process.

    Args:
        tasks: List of task dictionaries to execute
        run_task: Function to execute a single task
        num_concurrent: Maximum number of concurrent tasks
        request_rate: Target request rate (requests per second). If None, run as fast as possible.
    """
    print(f"Running {len(tasks)} scenarios with concurrency={num_concurrent}, rate={request_rate}")

    start_time = time.time()
    completed = 0
    pending_futures: List[Future] = []

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        if request_rate is None:
            # Burst mode: submit all tasks immediately
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in futures:
                future.result()
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    print(f"Progress: {completed}/{len(tasks)} completed, rate: {rate:.2f} req/s")
        else:
            # Poisson-distributed arrival times using exponential distribution
            for task in tasks:
                # Submit the task
                future = executor.submit(run_task, task)
                pending_futures.append(future)

                # Sample next inter-arrival time from exponential distribution
                # Mean inter-arrival time is 1/request_rate
                interval = np.random.exponential(1.0 / request_rate)

                # Wait for the sampled interval before submitting next request
                time.sleep(interval)

                # Check and report on completed tasks
                pending_futures = [f for f in pending_futures if not f.done()]
                completed = (
                    len(tasks) - len(pending_futures) - (len([t for t in tasks if tasks.index(t) > tasks.index(task)]))
                )

                if (tasks.index(task) + 1) % 10 == 0:
                    total_elapsed = time.time() - start_time
                    actual_rate = (len([t for t in tasks if tasks.index(t) <= tasks.index(task)])) / total_elapsed
                    print(
                        f"Submitted: {len([t for t in tasks if tasks.index(t) <= tasks.index(task)])}/{len(tasks)}, "
                        f"target rate: {request_rate:.2f}, actual: {actual_rate:.2f} req/s"
                    )

            # Wait for all remaining tasks to complete
            for future in pending_futures:
                future.result()

    total_time = time.time() - start_time
    final_rate = len(tasks) / total_time if total_time > 0 else 0
    print(f"Completed {len(tasks)} scenarios in {total_time:.2f}s, avg rate: {final_rate:.2f} req/s")
