#!/usr/bin/env python3
"""
Concurrent SGLang Model Server Launcher and AgBench Runner

This script:
1. Launches multiple sglang model servers on different ports
2. Waits for all servers to be ready
3. Runs multiple agbench benchmarks concurrently, each pointing to a different server
4. Handles graceful shutdown of all servers

Configuration can be provided via:
- Command line arguments
- YAML configuration file
- Environment variables
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class SGLangServerConfig:
    """Configuration for a single sglang server instance."""

    model: str  # Model name/path (e.g., "Qwen/Qwen2.5-7B-Instruct")
    port: int  # Port to run the server on
    host: str = "0.0.0.0"  # Host to bind to
    tp: int = 1  # Tensor parallelism
    dp: int = 1  # Data parallelism
    mem_fraction: float = 0.9  # GPU memory fraction
    device: int | str = "auto"  # GPU device index or "auto"
    log_requests: bool = False  # Enable request/response logging for trajectory analysis
    max_tokens: int | None = None  # Maximum tokens to generate in completion
    min_tokens: int | None = None  # Minimum tokens to generate in completion
    extra_body: dict[str, Any] = field(default_factory=dict)  # Extra body params for OpenAI API (e.g., chat_template_kwargs)
    extra_args: list[str] = field(default_factory=list)  # Additional sglang arguments
    env: dict[str, str] = field(default_factory=dict)  # Environment variables

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str  # Identifier for this benchmark run
    scenario_file: str  # Path to the scenario JSONL file
    server_port: int  # Port of the sglang server to use (for single-model scenarios)
    config_template: str | None = None  # Path to config template (will be modified with correct port)
    server_port_map: dict[str, int] = field(default_factory=dict)  # Map config section names to server ports (for multi-model scenarios, e.g. {"model_config_turn1": 30013, "model_config_turn2": 30014})
    env_file: str | None = None  # Path to ENV.yaml file
    num_concurrent: int = 1  # Number of concurrent tasks
    request_rate: float | None = None  # Poisson request rate (None = burst mode)
    repeat: int = 1  # Number of repetitions per task
    native: bool = True  # Run in native mode (not Docker)
    minimal_logs: bool = True  # Keep only console_log.txt in results (default: True)
    extra_args: list[str] = field(default_factory=list)  # Additional agbench arguments


@dataclass
class RunConfig:
    """Top-level configuration for the entire run."""

    servers: list[SGLangServerConfig]
    benchmarks: list[BenchmarkConfig]
    server_startup_timeout: int = 600  # seconds to wait for servers to start
    server_health_check_interval: int = 5  # seconds between health checks
    shutdown_timeout: int = 30  # seconds to wait for graceful shutdown


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = """
# Example configuration for run_bench.py
# Modify this file to match your setup

servers:
  - model: "Qwen/Qwen2.5-7B-Instruct"
    port: 30000
    tp: 1
    mem_fraction: 0.45
    log_requests: false  # Enable for trajectory analysis (logs full request/response)
    max_tokens: 1000  # Maximum tokens to generate (optional)
    min_tokens: 10  # Minimum tokens to generate (optional)

  - model: "Qwen/Qwen2.5-14B-Instruct"
    port: 30001
    tp: 1
    mem_fraction: 0.45
    log_requests: false
    max_tokens: 1000  # Maximum tokens to generate (optional)
    min_tokens: 10  # Minimum tokens to generate (optional)

benchmarks:
  - name: "humaneval_qwen7b"
    scenario_file: "benchmarks/HumanEval/Tasks/human_eval_AgentChat.jsonl"
    server_port: 30000
    num_concurrent: 4
    repeat: 1

  - name: "humaneval_qwen14b"
    scenario_file: "benchmarks/HumanEval/Tasks/human_eval_AgentChat.jsonl"
    server_port: 30001
    num_concurrent: 4
    repeat: 1

server_startup_timeout: 600
server_health_check_interval: 5
shutdown_timeout: 30
"""


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: str | None) -> RunConfig:
    """Load configuration from a YAML file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(DEFAULT_CONFIG)

    servers = [SGLangServerConfig(**s) for s in data.get("servers", [])]
    benchmarks = [BenchmarkConfig(**b) for b in data.get("benchmarks", [])]

    return RunConfig(
        servers=servers,
        benchmarks=benchmarks,
        server_startup_timeout=data.get("server_startup_timeout", 600),
        server_health_check_interval=data.get("server_health_check_interval", 5),
        shutdown_timeout=data.get("shutdown_timeout", 30),
    )


def _update_config_section(config_section: dict[str, Any], server: SGLangServerConfig) -> None:
    """Update a single model config section with the server's base_url, max_tokens, min_tokens, and extra_body."""
    if "config" in config_section:
        config_section["config"]["base_url"] = server.base_url
        if server.max_tokens is not None:
            config_section["config"]["max_tokens"] = server.max_tokens
        if server.min_tokens is not None:
            config_section["config"]["min_tokens"] = server.min_tokens
        if server.extra_body:
            config_section["config"]["extra_body"] = server.extra_body


def generate_agbench_config(
    server: SGLangServerConfig,
    template_path: str | None,
    output_path: str,
    all_servers: list[SGLangServerConfig] | None = None,
    server_port_map: dict[str, int] | None = None,
) -> None:
    """Generate an agbench config.yaml file for a specific server.

    For multi-model scenarios (e.g., two-turn benchmarks), use server_port_map
    to specify which config sections map to which server ports.
    E.g., {"model_config_turn1": 30013, "model_config_turn2": 30014}
    """
    if template_path and Path(template_path).exists():
        with open(template_path) as f:
            config = yaml.safe_load(f)

        # Update the primary model_config section
        if "model_config" in config and "config" in config["model_config"]:
            _update_config_section(config["model_config"], server)

        # Update additional config sections specified in server_port_map
        # Supports dot-notation for nested keys, e.g. "model_configs.0.6b"
        if server_port_map and all_servers:
            servers_by_port = {s.port: s for s in all_servers}
            for section_name, port in server_port_map.items():
                if port not in servers_by_port:
                    continue
                # Navigate dot-separated keys
                parts = section_name.split(".")
                target = config
                for part in parts:
                    if isinstance(target, dict) and part in target:
                        target = target[part]
                    else:
                        target = None
                        break
                if isinstance(target, dict):
                    _update_config_section(target, servers_by_port[port])
    else:
        # Generate a default config
        config_dict = {
            "model": server.model.split("/")[-1],
            "base_url": server.base_url,
            "api_key": "EMPTY",
            "stream_options": {"include_usage": True},
            "model_info": {
                "family": server.model.split("/")[0],
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
            },
        }
        if server.max_tokens is not None:
            config_dict["max_tokens"] = server.max_tokens
        if server.min_tokens is not None:
            config_dict["min_tokens"] = server.min_tokens
        if server.extra_body:
            config_dict["extra_body"] = server.extra_body

        config = {
            "model_config": {
                "provider": "autogen_ext.models.openai.OpenAIChatCompletionClient",
                "config": config_dict,
            }
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# =============================================================================
# Server Management
# =============================================================================


class SGLangServer:
    """Manages a single sglang server process."""

    def __init__(self, config: SGLangServerConfig):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None

    def start(self, log_dir: Path) -> None:
        """Start the sglang server process."""
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"sglang_server_{self.config.port}.log"

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model,
            "--port",
            str(self.config.port),
            "--host",
            self.config.host,
            "--tp",
            str(self.config.tp),
            "--dp",
            str(self.config.dp),
            "--mem-fraction-static",
            str(self.config.mem_fraction),
        ]

        # Enable request logging for trajectory analysis
        if self.config.log_requests:
            cmd.extend(["--log-requests", "--log-requests-level", "2"])

        cmd.extend(self.config.extra_args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.device)
        env.update(self.config.env)

        self._log_file = open(log_path, "w")
        print(f"Starting sglang server for {self.config.model} on port {self.config.port}")
        print(f"  Log file: {log_path}")
        print(f"  Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    async def wait_until_ready(self, timeout: int, check_interval: int) -> bool:
        """Wait for the server to be ready to accept requests."""
        health_url = f"http://{self.config.host}:{self.config.port}/health"
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                if self.process and self.process.poll() is not None:
                    print(f"Server on port {self.config.port} exited unexpectedly with code {self.process.returncode}")
                    return False

                try:
                    response = await client.get(health_url, timeout=5.0)
                    if response.status_code == 200:
                        print(f"Server on port {self.config.port} is ready")
                        return True
                except (httpx.RequestError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(check_interval)

        print(f"Server on port {self.config.port} failed to start within {timeout} seconds")
        return False

    def stop(self, timeout: int = 30) -> None:
        """Stop the server process gracefully."""
        if self.process is None:
            return

        print(f"Stopping server on port {self.config.port}")

        # Try graceful shutdown first
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Server on port {self.config.port} did not terminate gracefully, killing...")
            self.process.kill()
            self.process.wait()

        if self._log_file:
            self._log_file.close()
            self._log_file = None

        print(f"Server on port {self.config.port} stopped")


# =============================================================================
# Benchmark Execution
# =============================================================================


async def run_benchmark(benchmark: BenchmarkConfig, config: RunConfig, work_dir: Path) -> tuple[str, bool]:
    """Run a single benchmark and return (name, success)."""
    print(f"\nStarting benchmark: {benchmark.name}")

    # Find the server config for this benchmark
    server = next((s for s in config.servers if s.port == benchmark.server_port), None)
    if server is None:
        print(f"  Error: No server configured for port {benchmark.server_port}")
        return (benchmark.name, False)

    # Generate config file for this benchmark
    config_path = work_dir / f"config_{benchmark.name}.yaml"
    generate_agbench_config(
        server,
        benchmark.config_template,
        str(config_path),
        all_servers=config.servers,
        server_port_map=benchmark.server_port_map if benchmark.server_port_map else None,
    )

    # Build the agbench command
    results_dir = work_dir / "Results"
    cmd = [
        "agbench",
        "run",
        benchmark.scenario_file,
        "-c",
        str(config_path),
        "--repeat",
        str(benchmark.repeat),
        # "--subsample",
        # "50",
        "--streaming",  # to allow streaming the updates to results.json
        "--results-dir",
        str(results_dir),
    ]

    if benchmark.request_rate is not None:
        cmd.extend(["--request-rate", str(benchmark.request_rate)])

    if benchmark.env_file:
        cmd.extend(["-e", benchmark.env_file])

    if benchmark.native:
        cmd.append("--native")

    if benchmark.minimal_logs:
        cmd.append("--minimal-logs")

    cmd.extend(benchmark.extra_args)

    print(f"  Command: {' '.join(cmd)}")

    # Run the benchmark
    log_path = work_dir / f"benchmark_{benchmark.name}.log"
    with open(log_path, "w") as log_file:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent),  # Run from agbench package root
        )
        await process.wait()

    success = process.returncode == 0
    status = "completed successfully" if success else f"failed with code {process.returncode}"
    print(f"  Benchmark {benchmark.name} {status}")
    print(f"  Log file: {log_path}")

    return (benchmark.name, success)


async def run_all_benchmarks(config: RunConfig, work_dir: Path) -> dict[str, bool]:
    """Run all benchmarks concurrently."""
    tasks = [run_benchmark(b, config, work_dir) for b in config.benchmarks]
    results = await asyncio.gather(*tasks)
    return dict(results)


# =============================================================================
# Main Orchestration
# =============================================================================


async def main_async(config: RunConfig, work_dir: Path) -> int:
    """Main async entry point."""
    servers: list[SGLangServer] = []
    exit_code = 0

    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame: Any) -> None:
        print(f"\nReceived signal {sig}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start all servers
        print("=" * 60)
        print("Starting SGLang Servers")
        print("=" * 60)

        for server_config in config.servers:
            server = SGLangServer(server_config)
            server.start(work_dir / "logs")
            servers.append(server)

        # Wait for all servers to be ready
        print("\nWaiting for servers to be ready...")
        ready_tasks = [
            s.wait_until_ready(config.server_startup_timeout, config.server_health_check_interval) for s in servers
        ]
        ready_results = await asyncio.gather(*ready_tasks)

        if not all(ready_results):
            print("\nError: Not all servers started successfully")
            exit_code = 1
            return exit_code

        print("\n" + "=" * 60)
        print("All servers ready, starting benchmarks")
        print("=" * 60)

        # Run benchmarks
        benchmark_results = await run_all_benchmarks(config, work_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("Benchmark Results Summary")
        print("=" * 60)

        for name, success in benchmark_results.items():
            status = "PASSED" if success else "FAILED"
            print(f"  {name}: {status}")

        if not all(benchmark_results.values()):
            exit_code = 1

    except Exception as e:
        print(f"\nError during execution: {e}")
        exit_code = 1

    finally:
        # Stop all servers
        print("\n" + "=" * 60)
        print("Shutting down servers")
        print("=" * 60)

        for server in servers:
            server.stop(config.shutdown_timeout)

    return exit_code


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Concurrent SGLang server launcher and agbench runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a config file
  python run_bench.py --config my_config.yaml

  # Generate a sample config file
  python run_bench.py --generate-config > my_config.yaml

  # Quick run with command line arguments (single server + benchmark)
  python run_bench.py \\
    --model "Qwen/Qwen2.5-7B-Instruct" \\
    --port 30000 \\
    --scenario benchmarks/HumanEval/Tasks/human_eval_AgentChat.jsonl \\
    --num-concurrent 4

Configuration File Format:
  See the generated sample config for the full YAML schema.
  The config file supports multiple servers and benchmarks.
""",
    )

    # Config file options
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Print a sample configuration file and exit",
    )

    # Quick run options (alternative to config file)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to serve (for quick single-server run)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=30000,
        help="Port for the server (default: 30000)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        help="Scenario JSONL file to run",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=4,
        help="Number of concurrent benchmark tasks (default: 4)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Poisson request rate (default: burst mode)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions per task (default: 1)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.9,
        help="GPU memory fraction (default: 0.9)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="GPU device index or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run benchmark in Docker mode (default: native mode)",
    )
    parser.add_argument(
        "--log-requests",
        action="store_true",
        help="Enable sglang request/response logging for trajectory analysis",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate in completion (default: None)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        help="Minimum tokens to generate in completion (default: None)",
    )

    # Output options
    parser.add_argument(
        "--work-dir",
        type=str,
        default="run_bench_output",
        help="Output directory for all results, logs, and configs (default: run_bench_output)",
    )

    args = parser.parse_args()

    # Generate sample config if requested
    if args.generate_config:
        print(DEFAULT_CONFIG)
        return

    # Build configuration
    if args.config:
        config = load_config(args.config)
    elif args.model and args.scenario:
        # Quick run mode
        config = RunConfig(
            servers=[
                SGLangServerConfig(
                    model=args.model,
                    port=args.port,
                    tp=args.tp,
                    mem_fraction=args.mem_fraction,
                    device=args.device,
                    log_requests=args.log_requests,
                    max_tokens=args.max_tokens,
                    min_tokens=args.min_tokens,
                )
            ],
            benchmarks=[
                BenchmarkConfig(
                    name="benchmark",
                    scenario_file=args.scenario,
                    server_port=args.port,
                    num_concurrent=args.num_concurrent,
                    request_rate=args.request_rate,
                    repeat=args.repeat,
                    native=not args.docker,
                )
            ],
        )
    else:
        parser.error("Either --config or both --model and --scenario are required")

    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run
    exit_code = asyncio.run(main_async(config, work_dir))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
