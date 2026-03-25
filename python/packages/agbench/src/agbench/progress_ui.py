"""
Terminal progress UI for agbench streaming results using Rich library.

Provides real-time visual feedback during benchmark runs.
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .hooks import Event, EventType, get_hook_manager

# Try to import Rich, gracefully degrade if not available
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressUI:
    """
    Terminal progress UI using Rich library.

    Displays a live-updating table with benchmark progress statistics.
    Falls back to simple text output if Rich is not installed.
    """

    def __init__(self, scenario_name: str, total_tasks: int = 0) -> None:
        """
        Initialize the progress UI.

        Args:
            scenario_name: Name of the scenario being run
            total_tasks: Expected total number of task repetitions
        """
        self.scenario_name = scenario_name
        self.total_tasks = total_tasks
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.running = 0
        self.start_time: Optional[datetime] = None
        self._lock = threading.Lock()
        self._registered = False
        self._live: Any = None
        self._running = False
        self._update_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the progress UI."""
        if self._running:
            return

        self._running = True
        self.start_time = datetime.now()

        # Register for events
        self.register()

        if RICH_AVAILABLE:
            self._start_rich_ui()
        else:
            self._start_simple_ui()

    def stop(self) -> None:
        """Stop the progress UI."""
        if not self._running:
            return

        self._running = False

        # Unregister from events
        self.unregister()

        if RICH_AVAILABLE and self._live is not None:
            self._live.stop()
            self._live = None

        # Wait for update thread to finish
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None

        # Print final summary
        self._print_summary()

    def register(self) -> None:
        """Register with the HookManager to receive events."""
        if self._registered:
            return

        hook_manager = get_hook_manager()
        hook_manager.register(EventType.REPETITION_START, self._on_repetition_start)
        hook_manager.register(EventType.REPETITION_END, self._on_repetition_end)
        self._registered = True

    def unregister(self) -> None:
        """Unregister from the HookManager."""
        if not self._registered:
            return

        hook_manager = get_hook_manager()
        hook_manager.unregister(EventType.REPETITION_START, self._on_repetition_start)
        hook_manager.unregister(EventType.REPETITION_END, self._on_repetition_end)
        self._registered = False

    def _on_repetition_start(self, event: Event) -> None:
        """Handle REPETITION_START event."""
        with self._lock:
            self.running += 1

    def _on_repetition_end(self, event: Event) -> None:
        """Handle REPETITION_END event."""
        with self._lock:
            self.running = max(0, self.running - 1)
            self.completed += 1
            if event.success is True:
                self.successful += 1
            elif event.success is False:
                self.failed += 1

    def _get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            elapsed = 0.0
            rate = 0.0
            eta = None

            if self.start_time is not None:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed > 0 and self.completed > 0:
                    rate = self.completed / elapsed
                    remaining = self.total_tasks - self.completed
                    if rate > 0:
                        eta = remaining / rate

            success_rate = (
                self.successful / self.completed if self.completed > 0 else 0.0
            )

            return {
                "total": self.total_tasks,
                "completed": self.completed,
                "successful": self.successful,
                "failed": self.failed,
                "running": self.running,
                "success_rate": success_rate,
                "elapsed": elapsed,
                "rate": rate,
                "eta": eta,
            }

    def _start_rich_ui(self) -> None:
        """Start Rich-based UI with live updates."""
        console = Console()

        def generate_table() -> Table:
            stats = self._get_stats()

            table = Table(title=f"Benchmark: {self.scenario_name}")

            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")

            table.add_row("Total Tasks", str(stats["total"]))
            table.add_row("Completed", str(stats["completed"]))
            table.add_row("Successful", f"{stats['successful']} ({stats['success_rate']:.1%})")
            table.add_row("Failed", str(stats["failed"]))
            table.add_row("Running", str(stats["running"]))
            table.add_row("Rate", f"{stats['rate']:.2f} tasks/sec")
            table.add_row("Elapsed", f"{stats['elapsed']:.1f}s")

            if stats["eta"] is not None:
                table.add_row("ETA", f"{stats['eta']:.1f}s")
            else:
                table.add_row("ETA", "calculating...")

            return table

        self._live = Live(generate_table(), console=console, refresh_per_second=10)
        self._live.start()

        # Start background update thread
        def update_loop() -> None:
            while self._running:
                if self._live is not None:
                    self._live.update(generate_table())
                time.sleep(0.1)  # 10 Hz refresh

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()

    def _start_simple_ui(self) -> None:
        """Start simple text-based UI (fallback when Rich is not available)."""
        print(f"\nStarting benchmark: {self.scenario_name}")
        print(f"Total tasks: {self.total_tasks}")
        print("-" * 40)

        # Start background update thread for periodic status
        def update_loop() -> None:
            last_completed = 0
            while self._running:
                stats = self._get_stats()
                if stats["completed"] != last_completed or stats["completed"] % 10 == 0:
                    print(
                        f"Progress: {stats['completed']}/{stats['total']} "
                        f"({stats['success_rate']:.1%} success, "
                        f"{stats['rate']:.2f} tasks/sec)"
                    )
                    last_completed = stats["completed"]
                time.sleep(5.0)  # Update every 5 seconds

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()

    def _print_summary(self) -> None:
        """Print final summary."""
        stats = self._get_stats()

        print("\n" + "=" * 50)
        print(f"Benchmark Complete: {self.scenario_name}")
        print("=" * 50)
        print(f"  Total:      {stats['total']}")
        print(f"  Completed:  {stats['completed']}")
        print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1%})")
        print(f"  Failed:     {stats['failed']}")
        print(f"  Time:       {stats['elapsed']:.1f}s")
        print(f"  Rate:       {stats['rate']:.2f} tasks/sec")
        print("=" * 50 + "\n")

    def update_total(self, total: int) -> None:
        """Update the total number of tasks."""
        with self._lock:
            self.total_tasks = total


def is_rich_available() -> bool:
    """Check if Rich library is available."""
    return RICH_AVAILABLE
