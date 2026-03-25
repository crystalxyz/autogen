"""
Scenario execution functions for agbench.
"""

import os
import pathlib
import subprocess
import sys
from typing import Dict, Optional

from . import __version__
from .apptainer_instance import ApptainerInstance
from .hooks import Event, EventType, get_hook_manager

# Get the path to the template directory
BASE_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template")

# Default task timeout in seconds
TASK_TIMEOUT = 1200


def _extract_task_info(work_dir: str) -> tuple[Optional[str], Optional[int]]:
    """
    Extract task_id and repetition_id from a work directory path.

    Args:
        work_dir: Path like "Results/scenario_timestamp/task_id/rep_id"

    Returns:
        Tuple of (task_id, repetition_id)
    """
    try:
        parts = work_dir.rstrip(os.sep).split(os.sep)
        if len(parts) >= 2:
            rep_id = int(parts[-1])
            task_id = parts[-2]
            return task_id, rep_id
    except (ValueError, IndexError):
        pass
    return None, None


def run_scenario_natively(
    work_dir: str,
    env: Dict[str, str],
    timeout: int = TASK_TIMEOUT,
    emit_events: bool = False,
) -> None:
    """
    Run a scenario in the native environment.

    Args:
        work_dir (path): the path to the working directory previously created to house this sceario instance
        env: Environment variables dictionary
        timeout: Timeout in seconds
        emit_events: Whether to emit execution events for streaming
    """
    # Convert work_dir to absolute path to avoid race conditions with os.chdir()
    work_dir_abs = os.path.abspath(work_dir)

    # Extract task info for events
    task_id, repetition_id = _extract_task_info(work_dir)

    # Emit EXECUTION_START event
    if emit_events:
        hook_manager = get_hook_manager()
        hook_manager.emit(
            Event(
                event_type=EventType.EXECUTION_START,
                task_id=task_id,
                repetition_id=repetition_id,
            )
        )

    # print(f"Running scenario natively in work dir: {os.path.relpath(work_dir_abs)}")

    # Prepare the environment
    full_env = os.environ.copy()
    full_env.update(env)

    print(f"\n\n{work_dir_abs}\n===================================================================")

    # Read the run.sh template and replace parameters
    template_path = os.path.join(BASE_TEMPLATE_PATH, "run.sh")
    with open(template_path, "r") as f:
        template_content = f.read()

    # Replace template parameters using string replacement
    run_script = template_content.replace("{__version__}", __version__)
    run_script = run_script.replace("{timeout + 30}", str(timeout + 30))
    run_script = run_script.replace("{timeout}", str(timeout))

    # Write the run script to the work directory
    with open(os.path.join(work_dir_abs, "run.sh"), "wt", newline="\n") as f:
        f.write(run_script)

    # Run the script and log the output using absolute paths
    console_log_path = os.path.join(work_dir_abs, "console_log.txt")
    run_script_path = os.path.join(work_dir_abs, "run.sh")

    with open(console_log_path, "wb") as f:
        process = subprocess.Popen(
            ["sh", run_script_path],
            cwd=work_dir_abs,  # Set working directory for subprocess instead of using os.chdir()
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for c in iter(lambda: process.stdout.read(1), b""):  # type: ignore
            f.write(c)
            # os.write(sys.stdout.fileno(), c)  # Write binary to stdout

    # Emit EXECUTION_END event
    if emit_events:
        hook_manager = get_hook_manager()
        hook_manager.emit(
            Event(
                event_type=EventType.EXECUTION_END,
                task_id=task_id,
                repetition_id=repetition_id,
            )
        )

    return


def run_scenario_in_apptainer(
    work_dir: str,
    apptainer_instance: ApptainerInstance,
    timeout: int = TASK_TIMEOUT,
    emit_events: bool = False,
) -> None:
    """
    Run a scenario in an Apptainer container.

    Args:
        work_dir (path): the path to the working directory previously created to house this scenario instance
        apptainer_instance: a persistent apptainer instance to use
        timeout (Optional, int): the number of seconds to allow a container to run before timing out
        emit_events: Whether to emit execution events for streaming
    """
    # Extract task info for events
    task_id, repetition_id = _extract_task_info(work_dir)

    # Emit EXECUTION_START event
    if emit_events:
        hook_manager = get_hook_manager()
        hook_manager.emit(
            Event(
                event_type=EventType.EXECUTION_START,
                task_id=task_id,
                repetition_id=repetition_id,
            )
        )

    # Read the run.sh template and replace parameters
    template_path = os.path.join(BASE_TEMPLATE_PATH, "run.sh")
    with open(template_path, "r") as f:
        template_content = f.read()

    # Replace template parameters using string replacement
    run_script = template_content.replace("{__version__}", __version__)
    run_script = run_script.replace("{timeout + 30}", str(timeout + 30))
    run_script = run_script.replace("{timeout}", str(timeout))

    # Write the run script to the work directory
    with open(os.path.join(work_dir, "run.sh"), "wt", newline="\n") as f:
        f.write(run_script)

    work_dir_abs = str(pathlib.Path(work_dir).absolute())
    log_file_path = os.path.join(work_dir, "console_log.txt")

    # print(f"Running scenario in work dir: {os.path.relpath(work_dir)}")
    # print("===================================================================")

    # Add this work_dir as a bind mount by creating a temporary bind
    # We need to add it to the instance's existing binds
    instance_binds = apptainer_instance.binds.copy()
    instance_binds.append(f"{work_dir_abs}:/workspace")

    # For persistent instance, we need to stop and restart with new bind
    # Or use apptainer exec with instance:// URI which supports additional binds
    # The exec command in the instance will handle the workspace bind

    # Execute using the exec method with the work directory bound to the host path
    # We execute run.sh using its absolute path in the bind mount
    apptainer_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{work_dir_abs}:{work_dir_abs}",
        f"instance://{apptainer_instance.instance_name}",
        "sh",
        "-c",
        f"cd {work_dir_abs} && sh run.sh",
    ]

    process = None

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        try:
            process = subprocess.Popen(
                apptainer_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Stream output
            if process.stdout:
                for line in iter(process.stdout.readline, b""):
                    line_str = line.decode("utf-8")
                    log_file.write(line_str)
                    log_file.flush()
                    sys.stdout.write(line_str)
                    sys.stdout.flush()

            # Wait for completion
            return_code = process.wait()

            if return_code != 0:
                print(f"\nScenario exited with code: {return_code}")

        except KeyboardInterrupt:
            log_file.write("\nKeyboard interrupt (Ctrl-C). Attempting to exit gracefully.\n")
            log_file.flush()
            sys.stdout.write("\nKeyboard interrupt (Ctrl-C). Attempting to exit gracefully.\n")
            sys.stdout.flush()
            if process is not None:
                process.terminate()
                process.wait()
            sys.exit(1)

    # Emit EXECUTION_END event
    if emit_events:
        hook_manager = get_hook_manager()
        hook_manager.emit(
            Event(
                event_type=EventType.EXECUTION_END,
                task_id=task_id,
                repetition_id=repetition_id,
            )
        )

    return
