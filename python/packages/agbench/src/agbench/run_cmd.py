import argparse
import errno
import json
import logging
import os
import pathlib
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import yaml
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from .version import __version__
from .apptainer_instance import ApptainerInstance
from .scenario_utils import (
    discover_scenario_files,
    expand_scenario,
    find_autogen_repo,
    get_scenario_env,
    get_scenario_name_and_dir,
    mkdir_p,
    prepare_apptainer_binds,
    split_jsonl,
    ScenarioInstance,
)
from .execution import run_scenario_natively, run_scenario_in_apptainer
from .workload_executor import (
    create_task_runner,
    execute_with_poisson_rate,
    prepare_task_list,
)
from .hooks import Event, EventType, get_hook_manager
from .load_module import load_module
from .tabulate_cmd import find_tabulate_module

DOCKER_AVAILABLE = False
docker = None  # type: ignore
APIError = Exception  # type: ignore
DockerException = Exception  # type: ignore
ImageNotFound = Exception  # type: ignore

# Figure out where everything is
SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

TASK_TIMEOUT = 60 * 120  # 120 minutes

RESOURCES_PATH = os.path.join(SCRIPT_DIR, "res")

# What platform are we running?
IS_WIN32 = sys.platform == "win32"

# This is the tag given to the image that is *built* when no other image is provided.
# Do not use this field to specify the name of an existing image (e.g., on Dockerhub)
DEFAULT_DOCKER_IMAGE_TAG = "agbench"

# Get a random number generator for subsampling
subsample_rng = random.Random(425)

# Success string patterns for checking scenario completion
SUCCESS_STRINGS = [
    "ALL TESTS PASSED !#!#",
]

COMPLETED_STRINGS = [
    "SCENARIO.PY COMPLETE !#!#",
]


def discover_scorer(scenario_dir: str) -> Optional[Callable[[str], Optional[bool]]]:
    """
    Discover a custom scorer function from the benchmark's custom_tabulate.py.

    Searches scenario_dir and walks up to its parent looking for custom_tabulate.py
    (or Scripts/custom_tabulate.py). This handles the case where scenario_dir points
    to a subdirectory like Tasks/ while custom_tabulate.py lives in a sibling
    Scripts/ directory.

    Args:
        scenario_dir: Directory containing the scenario (e.g. benchmarks/GAIA/Tasks/)

    Returns:
        The scorer function if found, None otherwise
    """
    # Allow walking up one level so that e.g. benchmarks/GAIA/Tasks/ can find
    # benchmarks/GAIA/Scripts/custom_tabulate.py
    parent_dir = os.path.dirname(os.path.abspath(scenario_dir))
    module_path = find_tabulate_module(scenario_dir, stop_dir=parent_dir)
    if module_path is None:
        return None

    try:
        module = load_module(module_path)
        scorer_fn = getattr(module, "scorer", None)
        if scorer_fn is not None and callable(scorer_fn):
            print(f"Using custom scorer from '{module_path}'")
            return scorer_fn
    except Exception as e:
        print(f"Warning: Failed to load custom scorer from '{module_path}': {e}")

    return None


def check_scenario_success(results_dir: str) -> Optional[bool]:
    """
    Check if a scenario completed successfully by reading console_log.txt.

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

            # Check for success
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


def get_scenario_elapsed_time(results_dir: str) -> Optional[float]:
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


def get_scenario_turns(results_dir: str) -> Optional[int]:
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


def get_timestamped_results_dir(scenario_name: str, base_dir: str = "Results") -> str:
    """
    Generate a results directory path with scenario name and timestamp suffix to avoid conflicts.

    Args:
        scenario_name: Name of the scenario being run
        base_dir: Base results directory (default: "Results")

    Returns:
        Directory path in format: {base_dir}/{scenario_name}_{timestamp}
        Example: Results/human_eval_20260106_143025
    """
    # Resume hook: AGBENCH_RESULTS_DIR_OVERRIDE forces a specific results dir
    # (used by the rerun workflow to fill in missing reps in an existing run).
    override = os.environ.get("AGBENCH_RESULTS_DIR_OVERRIDE")
    if override:
        return override
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash = random.randint(0, 999)
    return os.path.join(base_dir, f"{scenario_name}_{timestamp}_{hash}")


def run_scenarios(
    scenario: str,
    n_repeats: int,
    is_native: bool,
    config_file: Union[None, str],
    token_provider: Optional[Callable[[], str]],
    docker_image: Optional[str] = None,
    results_dir: str = "Results",
    subsample: Union[None, int, float] = None,
    env_file: Union[None, str] = None,
    apptainer_image: Optional[str] = None,
    use_apptainer: bool = False,
    skip_scenario_subdir: bool = False,
    streaming: bool = False,
    api_port: Optional[int] = None,
    no_progress_ui: bool = False,
    minimal_logs: bool = False,
) -> None:
    """
    Run a set agbench scenarios a given number of times.

    Args:
        scenario (path):    The file or folder containing the scenario JSONL instances. If given a folder, then
                            all JSONL files in the folder will be loaded and run.
        n_repeats (int):    The number of times each scenario instance will be repeated
        is_native (bool):   True if the scenario should be run locally rather than in Docker (proceed with caution!)
        results_dir (path): The folder were results will be saved.
        streaming: Whether to enable streaming results
        api_port: Port for API server (if streaming enabled)
        no_progress_ui: Whether to disable progress UI
        minimal_logs: If True, clean up unnecessary files after execution, keeping only console_log.txt
    """

    files: List[str] = []

    # Figure out which files or folders we are working with
    if scenario == "-" or os.path.isfile(scenario):
        files.append(scenario)
    elif os.path.isdir(scenario):
        for f in os.listdir(scenario):
            scenario_file = os.path.join(scenario, f)

            if not os.path.isfile(scenario_file):
                continue

            if not scenario_file.lower().endswith(".jsonl"):
                continue

            files.append(scenario_file)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), scenario)

    # Create persistent apptainer instance if using apptainer
    apptainer_instance = None
    if use_apptainer:
        # Determine the image to use
        image_to_use = apptainer_image
        if image_to_use is None:
            raise FileNotFoundError(f"Apptainer image not provided!")

        # Get bind mounts and environment
        env = get_scenario_env(token_provider=token_provider, env_file=env_file)
        binds = prepare_apptainer_binds(env)

        # Create and start the persistent instance
        apptainer_instance = ApptainerInstance(image_to_use, binds, env)
        apptainer_instance.start()

    # Initialize streaming components
    result_tracker = None
    progress_ui = None
    api_server = None

    if streaming:
        from .live_results import LiveResultTracker
        from .progress_ui import ProgressUI

        # Ensure results directory exists for result.json
        os.makedirs(results_dir, exist_ok=True)

        # Get scenario name for tracking
        scenario_name_for_tracking = "benchmark"
        if files:
            if files[0] == "-":
                scenario_name_for_tracking = "stdin"
            else:
                parts = os.path.basename(files[0]).split(".")
                parts.pop()
                scenario_name_for_tracking = ".".join(parts)

        # Initialize result tracker
        result_tracker = LiveResultTracker(
            scenario_name=scenario_name_for_tracking,
            results_dir=results_dir,
            auto_register=True,
        )

        # Initialize API server if requested
        if api_port is not None:
            from .api_server import APIServer

            api_server = APIServer(
                results_dir=results_dir,
                port=api_port,
                result_tracker=result_tracker,
            )
            api_server.start()

    try:
        # Emit RUN_START event
        if streaming:
            hook_manager = get_hook_manager()
            hook_manager.emit(Event(event_type=EventType.RUN_START))

        # Run all the scenario files
        for scenario_file in files:
            scenario_name: Optional[str] = None
            scenario_dir: Optional[str] = None
            file_handle = None

            # stdin
            if scenario_file == "-":
                scenario_name = "stdin"
                scenario_dir = "."
                file_handle = sys.stdin
            else:
                scenario_name_parts = os.path.basename(scenario_file).split(".")
                scenario_name_parts.pop()
                scenario_name = ".".join(scenario_name_parts)
                scenario_dir = os.path.dirname(os.path.realpath(scenario_file))
                file_handle = open(scenario_file, "rt")

            # Discover custom scorer for this benchmark
            custom_scorer = discover_scorer(scenario_dir) if scenario_dir != "." else None

            # Read all the lines, then subsample if needed
            lines = [line for line in file_handle]
            if subsample is not None:
                # How many lines are we sampling
                n = 0
                # It's a proportion
                if 0 <= subsample < 1:
                    n = int(len(lines) * subsample + 0.5)
                # It's a raw count
                else:
                    n = int(subsample)
                n = max(0, min(n, len(lines)))
                lines = subsample_rng.sample(lines, n)

            # Calculate total tasks for progress UI
            total_tasks = len(lines) * n_repeats

            # Initialize progress UI if streaming enabled
            if streaming and not no_progress_ui:
                from .progress_ui import ProgressUI

                progress_ui = ProgressUI(
                    scenario_name=scenario_name or "benchmark",
                    total_tasks=total_tasks,
                )
                progress_ui.start()

            instances = [json.loads(line) for line in lines]

            for i in range(n_repeats):
                for instance in instances:
                    # Create a folder to store the results
                    # Results base
                    if not os.path.isdir(results_dir):
                        os.makedirs(results_dir, exist_ok=True)
                    # Results for the scenario
                    results_scenario = os.path.join(results_dir, scenario_name)
                    if not os.path.isdir(results_scenario):
                        os.mkdir(results_scenario)

                    # Results for the instance
                    results_instance = os.path.join(results_scenario, instance["id"])
                    if not os.path.isdir(results_instance):
                        os.mkdir(results_instance)

                    # Results for the repeats
                    results_repetition = os.path.join(results_instance, str(i))

                    # Emit REPETITION_START event
                    if streaming:
                        hook_manager = get_hook_manager()
                        hook_manager.emit(
                            Event(
                                event_type=EventType.REPETITION_START,
                                task_id=instance["id"],
                                repetition_id=i,
                            )
                        )

                    start_time = time.time()

                    # Expand the scenario
                    expand_scenario(scenario_dir, instance, results_repetition, config_file)

                    # Prepare the environment (keys/values that need to be added)
                    env = get_scenario_env(token_provider=token_provider, env_file=env_file)

                    # Run the scenario
                    if is_native:
                        run_scenario_natively(results_repetition, env)
                    elif use_apptainer:
                        run_scenario_in_apptainer(
                            results_repetition,
                            apptainer_instance=apptainer_instance,
                        )

                    # Clean up unnecessary files if minimal_logs is enabled
                    if minimal_logs:
                        from .workload_executor import _cleanup_logs
                        _cleanup_logs(results_repetition)

                    # Emit REPETITION_END event
                    if streaming:
                        elapsed_time = time.time() - start_time
                        if custom_scorer is not None:
                            success = custom_scorer(results_repetition)
                        else:
                            success = check_scenario_success(results_repetition)
                        logged_elapsed = get_scenario_elapsed_time(results_repetition)
                        if logged_elapsed is not None:
                            elapsed_time = logged_elapsed
                        turns = get_scenario_turns(results_repetition)

                        hook_manager = get_hook_manager()
                        hook_manager.emit(
                            Event(
                                event_type=EventType.REPETITION_END,
                                task_id=instance["id"],
                                repetition_id=i,
                                success=success,
                                elapsed_time=elapsed_time,
                                turns=turns,
                            )
                        )

            # Close regular files
            if scenario_file != "-":
                file_handle.close()

            # Stop progress UI after this scenario file
            if progress_ui is not None:
                progress_ui.stop()
                progress_ui = None

        # Emit RUN_END event
        if streaming:
            hook_manager = get_hook_manager()
            hook_manager.emit(Event(event_type=EventType.RUN_END))

    finally:
        # Clean up streaming components
        if progress_ui is not None:
            progress_ui.stop()
        if api_server is not None:
            api_server.stop()
        if result_tracker is not None:
            result_tracker.unregister()

        # Clean up the persistent apptainer instance
        if apptainer_instance is not None:
            apptainer_instance.stop()


def run_scenarios_subset(
    scenario_name: str,
    scenarios: List[Dict[str, Any]],
    n_repeats: int,
    is_native: bool,
    config_file: Union[None, str],
    docker_image: Optional[str] = None,
    results_dir: str = "Results",
    subsample: Union[None, int, float] = None,
    env_file: Union[None, str] = None,
    apptainer_image: Optional[str] = None,
    use_apptainer: bool = False,
    apptainer_instance_name: Optional[str] = None,
    skip_scenario_subdir: bool = False,
) -> None:
    """
    Run a subset of agbench scenarios a given number of times.
    In parallel mode, all workers share the same apptainer instance.

    Args:
        apptainer_instance_name: Name of a shared apptainer instance (for parallel mode)
    """
    # If we have an instance name, create a lightweight wrapper to use it
    apptainer_instance = None
    if use_apptainer and apptainer_instance_name:
        # Create a minimal instance object that references the shared instance
        # We don't call start() since it's already running in the parent process
        image_to_use = apptainer_image if apptainer_image else f"{DEFAULT_DOCKER_IMAGE_TAG}.sif"
        env = get_scenario_env(env_file=env_file)
        binds = prepare_apptainer_binds(env)
        apptainer_instance = ApptainerInstance(image_to_use, binds, env)
        apptainer_instance.instance_name = apptainer_instance_name  # Use the shared instance

    if apptainer_instance is None:
        raise RuntimeError(f"Apptainer image not set up correctly!")

    for instance in scenarios:
        # Create a folder to store the results
        # Results base

        mkdir_p(results_dir)

        # Results for the scenario (skip if already included in results_dir)
        if skip_scenario_subdir:
            results_scenario = results_dir
        else:
            results_scenario = os.path.join(results_dir, scenario_name)
            mkdir_p(results_scenario)

        # Results for the instance
        results_instance = os.path.join(results_scenario, instance["id"])
        mkdir_p(results_instance)

        # Results for the repeats
        for i in range(0, n_repeats):
            results_repetition = os.path.join(results_instance, str(i))

            # Skip it if it already exists
            if os.path.isdir(results_repetition):
                print(f"Found folder {results_repetition} ... Skipping.")
                continue
            # print(f"Running scenario {results_repetition}")

            # Expand the scenario
            expand_scenario(".", instance, results_repetition, config_file)  # type: ignore

            # Prepare the environment (keys/values that need to be added)
            env = get_scenario_env(env_file=env_file)

            # Run the scenario
            if is_native:
                run_scenario_natively(results_repetition, env)
            elif use_apptainer:
                run_scenario_in_apptainer(
                    results_repetition,
                    apptainer_instance=apptainer_instance,
                )


def run_parallel(args: argparse.Namespace, results_dir: str = "Results", skip_scenario_subdir: bool = False) -> None:
    """
    Run scenarios in parallel.
    Creates a single shared apptainer instance that all workers use concurrently.

    Args:
        args: Command-line arguments
        results_dir: Directory to save results (default: "Results")
        skip_scenario_subdir: Whether to skip creating scenario subdirectory (default: False)
    """
    # Read and split the JSONL file
    scenarios = split_jsonl(args.scenario, args.parallel)
    scenario_name_parts = os.path.basename(args.scenario).split(".")
    scenario_name_parts.pop()
    scenario_name = ".".join(scenario_name_parts)

    # Create a shared persistent apptainer instance if needed
    apptainer_instance: Optional[ApptainerInstance] = None
    use_apptainer = getattr(args, "apptainer", False)

    if use_apptainer:
        # Determine the image to use
        apptainer_image = getattr(args, "apptainer_image", None)
        image_to_use = apptainer_image
        if image_to_use is None:
            raise FileNotFoundError(f"Apptainer image not provided!")

        # Get bind mounts and environment
        env = get_scenario_env(env_file=args.env)
        binds = prepare_apptainer_binds(env)

        # Create and start the shared persistent instance
        apptainer_instance = ApptainerInstance(image_to_use, binds, env)
        apptainer_instance.start()
        print(f"Shared apptainer instance created for {args.parallel} parallel workers")

    try:
        # Create a pool of worker processes
        with Pool(processes=args.parallel) as pool:
            # Prepare arguments for each worker, passing the instance name
            worker_args = [
                (
                    scenario_name,
                    scenario_subset,
                    args.repeat,
                    args.native,
                    args.config,
                    args.docker_image,
                    results_dir,
                    args.subsample,
                    args.env,
                    getattr(args, "apptainer_image", None),
                    getattr(args, "apptainer", False),
                    apptainer_instance.instance_name if apptainer_instance else None,
                    skip_scenario_subdir,
                )
                for scenario_subset in scenarios
            ]

            # Run scenarios in parallel
            pool.starmap(run_scenarios_subset, worker_args)

    finally:
        # Clean up the shared apptainer instance
        if apptainer_instance is not None:
            apptainer_instance.stop()


def run_scenarios_with_rate_control(
    scenario: str,
    n_repeats: int,
    is_native: bool,
    config_file: Union[None, str],
    token_provider: Optional[Callable[[], str]],
    results_dir: str = "Results",
    subsample: Union[None, int, float] = None,
    env_file: Union[None, str] = None,
    apptainer_image: Optional[str] = None,
    use_apptainer: bool = False,
    num_concurrent: int = 1,
    request_rate: Optional[float] = None,
    skip_scenario_subdir: bool = False,
    streaming: bool = False,
    api_port: Optional[int] = None,
    no_progress_ui: bool = False,
    minimal_logs: bool = False,
) -> None:
    """
    Run scenarios with controlled concurrency and Poisson-distributed rate limiting.

    Implements rate limiting similar to sglang's bench_serving.py using exponential
    distribution for inter-arrival times (Poisson process).

    Args:
        scenario: Path to scenario file or directory
        n_repeats: Number of repetitions per scenario
        is_native: Whether to run natively
        config_file: Path to config file
        token_provider: Token provider function
        results_dir: Directory for results
        subsample: Subsample proportion or count
        env_file: Environment file path
        apptainer_image: Path to apptainer image
        use_apptainer: Whether to use apptainer
        num_concurrent: Maximum number of concurrent requests
        request_rate: Target request rate (requests per second). If None, run as fast as possible.
        streaming: Whether to enable streaming results
        api_port: Port for API server (if streaming enabled)
        no_progress_ui: Whether to disable progress UI
        minimal_logs: If True, clean up unnecessary files after execution, keeping only console_log.txt
    """
    # Discover scenario files
    files = discover_scenario_files(scenario)

    # Create persistent apptainer instance if using apptainer
    apptainer_instance = None
    if use_apptainer:
        image_to_use = apptainer_image
        if image_to_use is None:
            raise FileNotFoundError("Apptainer image not provided!")

        env = get_scenario_env(token_provider=token_provider, env_file=env_file)
        binds = prepare_apptainer_binds(env)
        apptainer_instance = ApptainerInstance(image_to_use, binds, env)
        apptainer_instance.start()

    # Initialize streaming components
    result_tracker = None
    progress_ui = None
    api_server = None

    if streaming:
        from .live_results import LiveResultTracker
        from .progress_ui import ProgressUI

        # Ensure results directory exists for result.json
        os.makedirs(results_dir, exist_ok=True)

        # Get scenario name for tracking
        scenario_name = "benchmark"
        if files:
            scenario_name, _ = get_scenario_name_and_dir(files[0])

        # Initialize result tracker
        result_tracker = LiveResultTracker(
            scenario_name=scenario_name,
            results_dir=results_dir,
            auto_register=True,
        )

        # Initialize API server if requested
        if api_port is not None:
            from .api_server import APIServer

            api_server = APIServer(
                results_dir=results_dir,
                port=api_port,
                result_tracker=result_tracker,
            )
            api_server.start()

    try:
        # Emit RUN_START event
        if streaming:
            hook_manager = get_hook_manager()
            hook_manager.emit(Event(event_type=EventType.RUN_START))

        # Process each scenario file
        for scenario_file in files:
            # Get scenario name and directory
            scenario_name, scenario_dir = get_scenario_name_and_dir(scenario_file)

            # Discover custom scorer for this benchmark
            custom_scorer = discover_scorer(scenario_dir) if scenario_dir != "." else None

            # Read scenario lines
            if scenario_file == "-":
                file_handle = sys.stdin
            else:
                file_handle = open(scenario_file, "rt")

            lines = [line for line in file_handle]

            if scenario_file != "-":
                file_handle.close()

            # Prepare task list
            tasks = prepare_task_list(
                lines=lines,
                scenario_name=scenario_name,
                scenario_dir=scenario_dir,
                results_dir=results_dir,
                n_repeats=n_repeats,
                config_file=config_file,
                subsample=subsample,
                skip_scenario_subdir=skip_scenario_subdir,
            )

            # Initialize progress UI if streaming enabled
            if streaming and not no_progress_ui:
                progress_ui = ProgressUI(
                    scenario_name=scenario_name,
                    total_tasks=len(tasks),
                )
                progress_ui.start()

            # Create task runner with streaming support
            run_task = create_task_runner(
                is_native=is_native,
                apptainer_instance=apptainer_instance,
                token_provider=token_provider,
                env_file=env_file,
                emit_events=streaming,
                minimal_logs=minimal_logs,
                scorer=custom_scorer,
            )

            # Execute with Poisson-distributed rate limiting
            execute_with_poisson_rate(
                tasks=tasks,
                run_task=run_task,
                num_concurrent=num_concurrent,
                request_rate=request_rate,
            )

            # Stop progress UI after this scenario
            if progress_ui is not None:
                progress_ui.stop()
                progress_ui = None

        # Emit RUN_END event
        if streaming:
            hook_manager = get_hook_manager()
            hook_manager.emit(Event(event_type=EventType.RUN_END))

    finally:
        # Clean up streaming components
        if progress_ui is not None:
            progress_ui.stop()
        if api_server is not None:
            api_server.stop()
        if result_tracker is not None:
            result_tracker.unregister()

        # Clean up the persistent apptainer instance
        if apptainer_instance is not None:
            apptainer_instance.stop()


def get_azure_token_provider() -> Optional[Callable[[], str]]:
    """
    Get the Azure bearer token generator if a token wasn't provided and there's any evidence of using Azure.
    """
    if not os.environ.get("AZURE_OPENAI_AD_TOKEN") and os.path.isdir(pathlib.Path("~/.azure").expanduser()):
        logging.disable(logging.CRITICAL)
        try:
            azure_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            azure_token_provider()  # Call it once to warm it up, and make sure it doesn't throw an error
            print("Found Azure token provider.")
            return azure_token_provider
        except ClientAuthenticationError:
            error_message = traceback.format_exc()
            print(
                f"Azure token provider failed loading. Try using 'az login --use-device-code'\n\nError details:\n{error_message}\n\nContinuing without Azure token provider..."
            )
        logging.disable(logging.NOTSET)
    return None


def run_cli(args: Sequence[str]) -> None:
    invocation_cmd = args[0]
    args = args[1:]

    # Prepare the argument parser
    parser = argparse.ArgumentParser(
        prog=invocation_cmd,
        description=f"{invocation_cmd} will run the specified AutoGen scenarios for a given number of repetitions and record all logs and trace information. When running in a Docker environment (default), each run will begin from a common, tightly controlled, environment. The resultant logs can then be further processed by other scripts to produce metrics.".strip(),
    )

    parser.add_argument(
        "scenario",
        help="The JSONL scenario file to run. If a directory is specified, then all JSONL scenarios in the directory are run. If set to '-', then read from stdin.",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        help="The number of repetitions to run for each scenario (default: 1).",
        default=1,
    )
    parser.add_argument(
        "-s",
        "--subsample",
        type=str,
        help='Run on a subsample of the tasks in the JSONL file(s). If a decimal value is specified, then run on the given proportion of tasks in each file. For example "0.7" would run on 70%% of tasks, and "1.0" would run on 100%% of tasks. If an integer value is specified, then randomly select *that* number of tasks from each specified JSONL file. For example "7" would run tasks, while "1" would run only 1 task from each specified JSONL file. (default: 1.0; which is 100%%)',
        default=None,
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        help="The number of parallel processes to run (default: 1).",
        default=1,
    )
    parser.add_argument(
        "-a",
        "--azure",
        action="store_true",
        help="Use Azure identity to pass an AZURE_OPENAI_AD_TOKEN to the task environment. This is necessary when using Azure-hosted OpenAI models rather than those hosted by OpenAI.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        help="The environment file to load into Docker, or into the native task context",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="The config file to copy into the Task.",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--docker-image",
        type=str,
        help="The Docker image to use when running scenarios. Can not be used together with --native or --apptainer. (default: '"
        + DEFAULT_DOCKER_IMAGE_TAG
        + "', which will be created if not present)",
        default=None,
    )
    parser.add_argument(
        "--apptainer",
        action="store_true",
        help="Run the scenarios in Apptainer containers instead of Docker. Can not be used together with --native or --docker-image.",
    )
    parser.add_argument(
        "--apptainer-image",
        type=str,
        help="The Apptainer image (.sif file) to use when running scenarios with --apptainer. (default: '"
        + DEFAULT_DOCKER_IMAGE_TAG
        + ".sif')",
        default=None,
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Run the scenarios natively rather than in docker. NOTE: This is not advisable, and should be done with great caution.",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        help="Maximum number of concurrent scenario executions. Enables controlled execution mode with Poisson-distributed rate limiting (similar to sglang bench_serving.py). Cannot be used with --parallel.",
        default=None,
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Target request rate (requests per second) for Poisson-distributed scenario arrivals. Requires --num-concurrent. If not specified, scenarios run as fast as possible with the given concurrency limit.",
        default=None,
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming results with live progress tracking. Creates result.json in the results directory with real-time updates.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        help="Start a REST API server on the specified port for external monitoring. Requires --streaming. Provides endpoints: /api/status, /api/logs/{task_id}/{rep_id}, /api/health",
        default=None,
    )
    parser.add_argument(
        "--no-progress-ui",
        action="store_true",
        help="Disable the terminal progress UI when --streaming is enabled. Useful for non-interactive environments.",
    )
    parser.add_argument(
        "--minimal-logs",
        action="store_true",
        help="Keep only console_log.txt in results directories, removing template files (scenario.py, config.yaml, etc.) after execution. Reduces storage usage significantly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Base directory for results (default: 'Results'). A timestamped subdirectory will be created inside.",
        default="Results",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.config is not None:
        # Make sure the config file is readable, so that we fail early
        with open(parsed_args.config, "r"):
            pass

    # don't support parallel and subsample together
    if parsed_args.parallel > 1 and parsed_args.subsample is not None:
        sys.exit("The options --parallel and --subsample can not be used together currently. Exiting.")

    # Validate controlled execution parameters
    if parsed_args.num_concurrent is not None and parsed_args.parallel > 1:
        sys.exit("The options --num-concurrent and --parallel can not be used together. Exiting.")

    if parsed_args.request_rate is not None and parsed_args.num_concurrent is None:
        sys.exit("The option --request-rate requires --num-concurrent to be specified. Exiting.")

    # Validate streaming parameters
    if parsed_args.api_port is not None and not parsed_args.streaming:
        sys.exit("The option --api-port requires --streaming to be specified. Exiting.")

    if parsed_args.no_progress_ui and not parsed_args.streaming:
        sys.exit("The option --no-progress-ui requires --streaming to be specified. Exiting.")

    # Don't allow both --docker-image and --native on the same command
    if parsed_args.docker_image is not None and parsed_args.native:
        sys.exit("The options --native and --docker-image can not be used together. Exiting.")

    # Don't allow both --apptainer and --native on the same command
    if parsed_args.apptainer and parsed_args.native:
        sys.exit("The options --native and --apptainer can not be used together. Exiting.")

    # Don't allow both --apptainer and --docker-image on the same command
    if parsed_args.apptainer and parsed_args.docker_image is not None:
        sys.exit("The options --apptainer and --docker-image can not be used together. Exiting.")

    # --apptainer-image requires --apptainer
    if parsed_args.apptainer_image is not None and not parsed_args.apptainer:
        sys.exit("The option --apptainer-image requires --apptainer to be specified. Exiting.")

    # Check Docker availability when Docker mode is requested (neither --native nor --apptainer)
    if not parsed_args.native and not parsed_args.apptainer:
        if not DOCKER_AVAILABLE:
            sys.exit(
                "Docker is not available on this system. Please either:\n"
                "  1. Install Docker and the Python docker library: pip install docker\n"
                "  2. Use Apptainer mode: add --apptainer flag\n"
                "  3. Use native mode: add --native flag (proceed with caution)\n"
                "Exiting."
            )

    # Warn if running natively
    if parsed_args.native:
        if IS_WIN32:
            sys.exit("Running scenarios with --native is not supported in Windows. Exiting.")

        # Crystal: Comment out the native execution warning prompt for smoother runs

        # sys.stderr.write(
        #     "WARNING: Running natively, without Docker, not only poses the usual risks of executing arbitrary AI generated code on your machine, it also makes it impossible to ensure that each test starts from a known and consistent set of initial conditions. For example, if the agents spend time debugging and installing Python libraries to solve the task, then those libraries will be available to all other runs. In other words, earlier runs can influence later runs, leading to many confounds in testing.\n\n"
        # )

        # # Does an environment variable override the prompt?
        # allow_native = os.environ.get("AGBENCH_ALLOW_NATIVE")
        # if allow_native is None or allow_native == "":
        #     choice = input(
        #         'Are you absolutely sure you want to continue with native execution? Type "Yes" exactly, and in full, to proceed: '
        #     )
        #     if choice.strip().lower() != "yes":
        #         sys.exit("Received '" + choice + "'. Exiting.")
        # elif allow_native.strip().lower() != "yes":
        #     sys.exit(f"Exiting because AGBENCH_ALLOW_NATIVE is '{allow_native}'\n")
        # else:
        #     sys.stderr.write(f"Continuing because AGBENCH_ALLOW_NATIVE is '{allow_native}'\n")
        #     time.sleep(0.75)  # Pause very briefly so the message isn't lost in the noise

    # Parse the subsample
    subsample = None
    if parsed_args.subsample is not None:
        subsample = float(parsed_args.subsample)
        if "." in parsed_args.subsample:  # Intention is to run on a proportion
            if subsample == 1.0:  # Intention is to run 100%, which is the default
                subsample = None  # None means 100% ... which use None to differentiate from the integer 1
            elif subsample < 0 or subsample > 1.0:
                raise (
                    ValueError(
                        "Subsample must either be an integer (specified without a decimal), or a Real number between 0.0 and 1.0"
                    )
                )

    # Get the Azure bearer token generator if a token wasn't provided and there's any evidence of using Azure
    azure_token_provider = None
    if parsed_args.azure:
        azure_token_provider = get_azure_token_provider()

    # Extract scenario name from the scenario file path
    scenario_file = parsed_args.scenario
    if scenario_file == "-":
        scenario_name = "stdin"
    else:
        scenario_name_parts = os.path.basename(scenario_file).split(".")
        if scenario_name_parts[-1].lower() == "jsonl":
            scenario_name_parts.pop()
        scenario_name = ".".join(scenario_name_parts)

    # Generate timestamped results directory
    timestamped_results_dir = get_timestamped_results_dir(scenario_name, base_dir=parsed_args.results_dir)
    print(f"Results will be saved to: {timestamped_results_dir}")

    # Run the scenario
    if parsed_args.parallel > 1:
        run_parallel(parsed_args, timestamped_results_dir, skip_scenario_subdir=True)
    elif parsed_args.num_concurrent is not None:
        # Controlled execution mode with Poisson-distributed rate limiting
        run_scenarios_with_rate_control(
            scenario=parsed_args.scenario,
            n_repeats=parsed_args.repeat,
            is_native=True if parsed_args.native else False,
            config_file=parsed_args.config,
            token_provider=azure_token_provider,
            results_dir=timestamped_results_dir,
            subsample=subsample,
            env_file=parsed_args.env,
            apptainer_image=parsed_args.apptainer_image,
            use_apptainer=parsed_args.apptainer,
            num_concurrent=parsed_args.num_concurrent,
            request_rate=parsed_args.request_rate,
            skip_scenario_subdir=True,
            streaming=parsed_args.streaming,
            api_port=parsed_args.api_port,
            no_progress_ui=parsed_args.no_progress_ui,
            minimal_logs=parsed_args.minimal_logs,
        )
    else:
        run_scenarios(
            scenario=parsed_args.scenario,
            n_repeats=parsed_args.repeat,
            is_native=True if parsed_args.native else False,
            config_file=parsed_args.config,
            token_provider=azure_token_provider,
            docker_image=parsed_args.docker_image,
            results_dir=timestamped_results_dir,
            subsample=subsample,
            env_file=parsed_args.env,
            apptainer_image=parsed_args.apptainer_image,
            use_apptainer=parsed_args.apptainer,
            skip_scenario_subdir=True,
            streaming=parsed_args.streaming,
            api_port=parsed_args.api_port,
            no_progress_ui=parsed_args.no_progress_ui,
            minimal_logs=parsed_args.minimal_logs,
        )
