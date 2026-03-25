import argparse
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import tabulate as tb

from .load_module import load_module
from .live_results import load_live_results

# Figure out where everything is
SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

TABULATE_FILE = "custom_tabulate.py"

SUCCESS_STRINGS = [
    "ALL TESTS PASSED !#!#",
]

COMPLETED_STRINGS = [
    "SCENARIO.PY COMPLETE !#!#",
]

EXCLUDE_DIR_NAMES = ["__pycache__"]

TIMER_REGEX = r"AgentChat execution time:\s*([\d.]+)(?:\s*seconds)?(?:\s*!#!#)?"


def find_tabulate_module(search_dir: str, stop_dir: Optional[str] = None) -> Optional[str]:
    """Hunt for the tabulate script."""

    search_dir = os.path.abspath(search_dir)
    if not os.path.isdir(search_dir):
        raise ValueError(f"'{search_dir}' is not a directory.")

    stop_dir = None if stop_dir is None else os.path.abspath(stop_dir)

    while True:
        path = os.path.join(search_dir, TABULATE_FILE)
        if os.path.isfile(path):
            return path

        path = os.path.join(search_dir, "Scripts", TABULATE_FILE)
        if os.path.isfile(path):
            return path

        path = os.path.join(search_dir, "scripts", TABULATE_FILE)
        if os.path.isfile(path):
            return path

        # Stop if we hit the stop_dir
        if search_dir == stop_dir:
            break

        # Stop if we hit the root
        parent_dir = os.path.abspath(os.path.join(search_dir, os.pardir))
        if parent_dir == search_dir:
            break

        search_dir = parent_dir

    return None


def default_scorer(instance_dir: str, success_strings: List[str] = SUCCESS_STRINGS) -> Optional[bool]:
    console_log = os.path.join(instance_dir, "console_log.txt")
    if os.path.isfile(console_log):
        with open(console_log, "rt") as fh:
            content = fh.read()

            # It succeeded
            for s in success_strings:
                if s in content:
                    return True

            # It completed without succeeding
            for s in COMPLETED_STRINGS:
                if s in content:
                    return False

            # Has not, or did not, complete
            return None
    else:
        return None


def default_timer(instance_dir: str, timer_regex: str = TIMER_REGEX) -> Optional[float]:
    console_log = os.path.join(instance_dir, "console_log.txt")
    if os.path.isfile(console_log):
        with open(console_log, "rt") as fh:
            content = fh.read()

            # It succeeded
            m = re.search(timer_regex, content)
            if m:
                return float(m.group(1))
            else:
                return None
    else:
        return None


def default_turns(instance_dir: str) -> Optional[int]:
    console_log = os.path.join(instance_dir, "console_log.txt")
    if os.path.isfile(console_log):
        with open(console_log, "rt") as fh:
            content = fh.read()
            count = content.count("TextMessage")
            if count == 0 or count == 1:
                return None
            return count - 1
    else:
        return None


def default_runtimes(instance_dir: str) -> Dict[str, List[float]]:
    console_log = os.path.join(instance_dir, "console_log.txt")
    if not os.path.isfile(console_log):
        return {}
    with open(console_log, "rt") as fh:
        content = fh.read()
    matches = re.findall(r"\[runtime\]\s*name=([^\s]+)\s+seconds=([\d.]+)", content)
    runtimes: Dict[str, List[float]] = {}
    for name, seconds in matches:
        runtimes.setdefault(name, []).append(float(seconds))
    return runtimes


ScorerFunc = Callable[[str], Optional[bool]]
TimerFunc = Callable[[str], Optional[float]]
TurnsFunc = Callable[[str], Optional[int]]
RuntimesFunc = Callable[[str], Dict[str, List[float]]]


def show_live_results(runlogs: str) -> bool:
    """
    Show live results from result.json if available.

    Args:
        runlogs: Path to the results directory

    Returns:
        True if live results were shown, False otherwise
    """
    live_results = load_live_results(runlogs)
    if live_results is None:
        return False

    print("\n" + "=" * 60)
    print("LIVE RESULTS (from result.json)")
    print("=" * 60)

    # Show run status
    status = live_results.get("status", "unknown")
    scenario_name = live_results.get("scenario_name", "unknown")
    print(f"\nScenario: {scenario_name}")
    print(f"Status: {status}")

    # Show timing
    start_time = live_results.get("start_time")
    end_time = live_results.get("end_time")
    if start_time:
        print(f"Start time: {start_time}")
    if end_time:
        print(f"End time: {end_time}")

    # Show stats
    stats = live_results.get("stats", {})
    if stats:
        print("\nProgress:")
        print(f"  Total repetitions:      {stats.get('total_repetitions', 0)}")
        print(f"  Completed repetitions:  {stats.get('completed_repetitions', 0)}")
        print(f"  Successful repetitions: {stats.get('successful_repetitions', 0)}")
        print(f"  Failed repetitions:     {stats.get('failed_repetitions', 0)}")
        print(f"  Running repetitions:    {stats.get('running_repetitions', 0)}")
        print(f"  Success rate:           {stats.get('success_rate', 0):.1%}")
        print(f"  Total elapsed time:     {stats.get('total_elapsed_time', 0):.1f}s")

    # Show per-task summary if available
    tasks = live_results.get("tasks", {})
    if tasks:
        print(f"\nTasks: {len(tasks)}")

        # Build a summary table
        task_summaries: List[Dict[str, Any]] = []
        for task_id, task_data in sorted(tasks.items()):
            repetitions = task_data.get("repetitions", {})
            completed = sum(1 for r in repetitions.values() if r.get("status") == "completed")
            successful = sum(1 for r in repetitions.values() if r.get("success") is True)
            running = sum(1 for r in repetitions.values() if r.get("status") == "running")

            task_summaries.append({
                "Task ID": task_id,
                "Repetitions": len(repetitions),
                "Completed": completed,
                "Successful": successful,
                "Running": running,
            })

        if task_summaries:
            df = pd.DataFrame(task_summaries)
            print("\n" + tb.tabulate(df, headers="keys", tablefmt="simple", showindex=False))  # type: ignore

    print("\n" + "=" * 60 + "\n")
    return True


def default_tabulate(
    args: List[str],
    scorer: ScorerFunc = default_scorer,
    timer: TimerFunc = default_timer,
    turns: TurnsFunc = default_turns,
    runtimes: RuntimesFunc = default_runtimes,
    exclude_dir_names: List[str] = EXCLUDE_DIR_NAMES,
) -> None:
    invocation_cmd = args[0]
    args = args[1:]

    warning = f"CAUTION: '{invocation_cmd}' is in early preview and is not thoroughly tested.\nPlease do not cite values from these calculations in academic work without first inspecting and verifying the results in the run logs yourself."

    # Prepare the argument parser
    parser = argparse.ArgumentParser(
        prog=invocation_cmd,
        description=f"{invocation_cmd} will tabulate the results of a previous run.",
    )

    parser.add_argument(
        "runlogs",
        help="The path where the run's logs are stored.",
    )
    parser.add_argument(
        "-c",
        "--csv",
        action="store_true",
        help="Output the results in CSV format.",
    )

    parser.add_argument(
        "-e", "--excel", help="Output the results in Excel format. Please specify a path for the Excel file.", type=str
    )

    parsed_args = parser.parse_args(args)
    runlogs: str = parsed_args.runlogs

    # Check for live results first (shows partial progress for in-progress runs)
    has_live_results = show_live_results(runlogs)

    all_results: List[Dict[str, Any]] = list()
    max_instances: Optional[int] = None
    runtime_names: set[str] = set()

    # Skip non-directory entries and result.json
    exclude_files = ["result.json", "result.json.tmp"]

    for task_id in sorted(
        os.listdir(runlogs),
        key=lambda s: os.path.getmtime(os.path.join(runlogs, s)),
    ):
        if task_id in exclude_dir_names or task_id in exclude_files:
            continue

        task_path = os.path.join(runlogs, task_id)

        if not os.path.isdir(task_path):
            continue

        # Collect the results vector
        results: Dict[str, Any] = {"Task Id": task_id}

        # Collect the results for each instance.
        instance_dirs = sorted(
            os.listdir(task_path),
            key=lambda s: os.path.getmtime(os.path.join(task_path, s)),
        )
        instances = [int(d) for d in instance_dirs if d.isdigit()]
        if not instances:
            continue

        for instance in instances:
            instance_dir = os.path.join(task_path, str(instance))
            results[f"Trial {instance} Success"] = scorer(instance_dir)
            results[f"Trial {instance} Time"] = timer(instance_dir)
            results[f"Trial {instance} Turns"] = turns(instance_dir)
            instance_runtimes = runtimes(instance_dir)
            for name, values in instance_runtimes.items():
                runtime_names.add(name)
                results[f"Trial {instance} [{name}] runtime"] = ", ".join(f"{v:.6f}" for v in values)

        max_instances = max(instances) if max_instances is None else max(max_instances, max(instances))

        # Buffer the results
        all_results.append(results)

    if not all_results or max_instances is None:
        sys.stderr.write("No completed instances found to tabulate.\n\n")
        return

    num_instances = max_instances + 1

    # Pad the results to max_instances
    for result in all_results:
        for i in range(num_instances):
            if f"Trial {i} Success" not in result:
                result[f"Trial {i} Success"] = None
            if f"Trial {i} Time" not in result:
                result[f"Trial {i} Time"] = None
            if f"Trial {i} Turns" not in result:
                result[f"Trial {i} Turns"] = None
            for name in runtime_names:
                key = f"Trial {i} [{name}] runtime"
                if key not in result:
                    result[key] = None

    # Create dataframe from results.
    df = pd.DataFrame(all_results)

    if parsed_args.csv:
        # Print out the dataframe in CSV format
        print(df.to_csv(index=False))
        # Print out alpha-version warning
        sys.stderr.write("\n" + warning + "\n\n")
    else:
        # Tabulate the results.
        print(tb.tabulate(df, headers="keys", tablefmt="simple"))  # type: ignore

        def _check_true(x: Any) -> Any:
            if isinstance(x, pd.Series):
                return x.apply(lambda y: y is True)  # type: ignore
            else:
                return x is True

        def _check_false(x: Any) -> Any:
            if isinstance(x, pd.Series):
                return x.apply(lambda y: y is False)  # type: ignore
            else:
                return x is False

        # Aggregate statistics for all tasks for each trials.
        print("\nSummary Statistics\n")
        score_columns = ["Trial " + str(i) + " Success" for i in range(num_instances)]
        # Count the number of successes when the value is True.
        successes = df[score_columns].apply(_check_true).sum(axis=0)  # type: ignore
        # Count the number of failures when the value is False.
        failures: pd.Series = df[score_columns].apply(_check_false).sum(axis=0)  # type: ignore
        # Count the number of missing
        missings = df[score_columns].isna().sum(axis=0)  # type: ignore
        # Count the total number of instances
        totals = successes + failures + missings  # type: ignore
        # Calculate the average success rates
        avg_success_rates = successes / (successes + failures)  # type: ignore
        time_columns = ["Trial " + str(i) + " Time" for i in range(num_instances)]  # type: ignore
        # Count the total time of non-null values
        total_times = df[time_columns].sum(axis=0, skipna=True)  # type: ignore
        # Calculate the average time of non-null values
        avg_times = df[time_columns].mean(axis=0, skipna=True)  # type: ignore
        turns_columns = ["Trial " + str(i) + " Turns" for i in range(num_instances)]  # type: ignore
        total_turns = df[turns_columns].sum(axis=0, skipna=True)  # type: ignore
        avg_turns = df[turns_columns].mean(axis=0, skipna=True)  # type: ignore

        def _list(series: Any) -> List[Any]:
            # If iteraable, convert to list
            if hasattr(series, "__iter__") and not isinstance(series, str):
                return list(series)
            else:
                # If not iterable, return the series
                return [series]

        # Create a per-trial summary dataframe
        trial_df = pd.DataFrame(
            {
                "Successes": _list(successes),  # type: ignore
                "Failures": _list(failures),  # type: ignore
                "Missing": _list(missings),  # type: ignore
                "Total": _list(totals),  # type: ignore
                "Average Success Rate": _list(avg_success_rates),  # type: ignore
                "Average Time": _list(avg_times),  # type: ignore
                "Total Time": _list(total_times),  # type: ignore
                "Average Turns": _list(avg_turns),  # type: ignore
                "Total Turns": _list(total_turns),  # type: ignore
            },
            index=[f"Trial {i}" for i in range(num_instances)],
        )
        # Print out the per-trial summary dataframe.
        print(tb.tabulate(trial_df, headers="keys", tablefmt="simple"))  # type: ignore

        # Aggregate statistics across tasks for all trials.
        # At least one success for each trial, averaged across tasks.
        average_at_least_one_success = df[score_columns].any(axis=1).mean(skipna=True)  # type: ignore
        # All successes for each trial
        average_all_successes = df[score_columns].all(axis=1).mean(skipna=True)  # type: ignore

        # Create a dataframe
        trial_aggregated_df = pd.DataFrame(
            {
                "At Least One Success": [average_at_least_one_success],  # type: ignore
                "All Successes": [average_all_successes],  # type: ignore
            },
            index=["Trial Aggregated"],
        )
        # Print out the trial-aggregated dataframe.
        print(tb.tabulate(trial_aggregated_df, headers="keys", tablefmt="simple"))  # type: ignore

        # Print out alpha-version warning
        sys.stderr.write("\n" + warning + "\n\n")


def tabulate_cli(args: Sequence[str]) -> None:
    invocation_cmd = args[0]
    args = args[1:]

    # We won't assume much about the arguments, letting the dynamically-loaded
    # tabulate modules parse the arguments however they want. But, we will use
    # bare arguments (not starting a "-"), to help us find what module to load.
    module_path = find_tabulate_module(os.getcwd(), stop_dir=os.getcwd())
    for arg in reversed(args):
        if module_path is not None:
            break
        if arg.startswith("-"):
            continue
        module_path = find_tabulate_module(arg)

    # Load the module and hand over control
    if module_path is None:
        sys.stderr.write("Using default tabulation method.\n\n")
        default_tabulate([invocation_cmd] + list(args))
    else:
        sys.stderr.write(f"Using tabulation method defined in '{module_path}'\n\n")
        load_module(module_path).main([invocation_cmd] + list(args))
