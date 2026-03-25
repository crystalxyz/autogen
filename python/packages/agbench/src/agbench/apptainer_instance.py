"""
Apptainer instance management for agbench.
"""

import atexit
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional


class ApptainerInstance:
    """
    Manages a persistent Apptainer instance that can be reused across multiple scenario runs.
    """

    def __init__(self, apptainer_image: str, binds: List[str], env: Dict[str, str]):
        self.apptainer_image = apptainer_image
        self.binds = binds
        self.env = env
        self.instance_name: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """Start the persistent apptainer instance."""
        if self.instance_name is not None:
            return  # Already started

        # Generate a unique instance name
        self.instance_name = f"agbench_instance_{os.getpid()}_{int(time.time())}"

        # Build the apptainer instance start command
        apptainer_cmd = [
            "apptainer", "instance", "start",
            "--writable-tmpfs",
            "--containall",
        ]

        # Add bind mounts
        for bind in self.binds:
            apptainer_cmd.extend(["--bind", bind])

        # Add environment variables
        for key, value in self.env.items():
            apptainer_cmd.extend(["--env", f"{key}={value}"])

        # Add the image and instance name
        apptainer_cmd.append(self.apptainer_image)
        apptainer_cmd.append(self.instance_name)

        print(f"Starting persistent apptainer instance: {self.instance_name}")
        print(f"Running: {' '.join(apptainer_cmd[:6])} ... {apptainer_cmd[-2:]}")

        # Start the instance
        result = subprocess.run(apptainer_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start apptainer instance: {result.stderr}")

        print(f"Apptainer instance {self.instance_name} started successfully")

        # Register cleanup on exit
        atexit.register(self.stop)

    def exec(self, command: List[str], work_dir: str, log_file_path: str) -> int:
        """Execute a command in the running apptainer instance."""
        if self.instance_name is None:
            raise RuntimeError("Apptainer instance not started")

        # Build the exec command
        apptainer_cmd = [
            "apptainer", "exec",
            "--pwd", work_dir,
            f"instance://{self.instance_name}",
        ] + command

        print(f"Executing in instance {self.instance_name}: {' '.join(command)}")

        # Run the command and stream output
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
                    print(f"\nCommand exited with code: {return_code}")

                return return_code

            except KeyboardInterrupt:
                log_file.write("\nKeyboard interrupt (Ctrl-C). Attempting to exit gracefully.\n")
                log_file.flush()
                sys.stdout.write("\nKeyboard interrupt (Ctrl-C). Attempting to exit gracefully.\n")
                sys.stdout.flush()
                process.terminate()
                process.wait()
                sys.exit(1)

    def stop(self) -> None:
        """Stop the persistent apptainer instance."""
        if self.instance_name is None:
            return  # Not started

        print(f"Stopping apptainer instance: {self.instance_name}")

        # Stop the instance
        result = subprocess.run(
            ["apptainer", "instance", "stop", self.instance_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Failed to stop apptainer instance: {result.stderr}")
        else:
            print(f"Apptainer instance {self.instance_name} stopped")

        self.instance_name = None
