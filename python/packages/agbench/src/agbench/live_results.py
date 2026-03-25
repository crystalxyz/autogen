"""
Live result tracking and persistence for agbench streaming results.

Provides real-time tracking of benchmark progress and writes result.json files.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .hooks import Event, EventType, get_hook_manager


@dataclass
class RepetitionResult:
    """Result of a single repetition of a task."""

    status: str = "pending"  # pending, running, completed
    success: Optional[bool] = None
    elapsed_time: Optional[float] = None
    turns: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "turns": self.turns,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class TaskResult:
    """Result of a task across all its repetitions."""

    task_id: str
    repetitions: Dict[int, RepetitionResult] = field(default_factory=dict)

    def get_stats(self) -> Dict[str, Any]:
        """Calculate per-task statistics."""
        completed = 0
        successful = 0
        failed = 0
        running = 0
        elapsed_times = []
        turn_counts = []

        for rep in self.repetitions.values():
            if rep.status == "completed":
                completed += 1
                if rep.success is True:
                    successful += 1
                elif rep.success is False:
                    failed += 1
                if rep.elapsed_time is not None:
                    elapsed_times.append(rep.elapsed_time)
                if rep.turns is not None:
                    turn_counts.append(rep.turns)
            elif rep.status == "running":
                running += 1

        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
        min_time = min(elapsed_times) if elapsed_times else None
        max_time = max(elapsed_times) if elapsed_times else None
        total_time = sum(elapsed_times) if elapsed_times else None

        avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else None
        min_turns = min(turn_counts) if turn_counts else None
        max_turns = max(turn_counts) if turn_counts else None
        total_turns = sum(turn_counts) if turn_counts else None

        success_rate = successful / completed if completed > 0 else None

        return {
            "total_repetitions": len(self.repetitions),
            "completed": completed,
            "successful": successful,
            "failed": failed,
            "running": running,
            "success_rate": round(success_rate, 4) if success_rate is not None else None,
            "avg_time": round(avg_time, 2) if avg_time is not None else None,
            "min_time": round(min_time, 2) if min_time is not None else None,
            "max_time": round(max_time, 2) if max_time is not None else None,
            "total_time": round(total_time, 2) if total_time is not None else None,
            "avg_turns": round(avg_turns, 2) if avg_turns is not None else None,
            "min_turns": min_turns,
            "max_turns": max_turns,
            "total_turns": total_turns,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "stats": self.get_stats(),
            "repetitions": {str(k): v.to_dict() for k, v in self.repetitions.items()},
        }


@dataclass
class RunResult:
    """Overall result of a benchmark run."""

    scenario_name: str
    status: str = "pending"  # pending, running, completed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    results_dir: Optional[str] = None
    tasks: Dict[str, TaskResult] = field(default_factory=dict)

    def get_stats(self) -> Dict[str, Any]:
        """Calculate aggregate statistics."""
        total_tasks = len(self.tasks)
        total_repetitions = 0
        completed_repetitions = 0
        successful_repetitions = 0
        failed_repetitions = 0
        running_repetitions = 0
        pending_repetitions = 0
        all_elapsed_times = []
        all_turn_counts = []
        task_avg_times = []
        task_avg_turns = []

        for task in self.tasks.values():
            task_times = []
            task_turns = []
            for rep in task.repetitions.values():
                total_repetitions += 1
                if rep.status == "completed":
                    completed_repetitions += 1
                    if rep.success is True:
                        successful_repetitions += 1
                    elif rep.success is False:
                        failed_repetitions += 1
                    if rep.elapsed_time is not None:
                        all_elapsed_times.append(rep.elapsed_time)
                        task_times.append(rep.elapsed_time)
                    if rep.turns is not None:
                        all_turn_counts.append(rep.turns)
                        task_turns.append(rep.turns)
                elif rep.status == "running":
                    running_repetitions += 1
                else:
                    pending_repetitions += 1

            # Calculate per-task averages
            if task_times:
                task_avg_times.append(sum(task_times) / len(task_times))
            if task_turns:
                task_avg_turns.append(sum(task_turns) / len(task_turns))

        # Calculate overall metrics
        success_rate = (
            successful_repetitions / completed_repetitions
            if completed_repetitions > 0
            else 0.0
        )

        total_elapsed_time = sum(all_elapsed_times) if all_elapsed_times else 0.0
        avg_time_per_rep = (
            total_elapsed_time / len(all_elapsed_times)
            if all_elapsed_times
            else None
        )
        min_time = min(all_elapsed_times) if all_elapsed_times else None
        max_time = max(all_elapsed_times) if all_elapsed_times else None
        avg_time_per_task = (
            sum(task_avg_times) / len(task_avg_times)
            if task_avg_times
            else None
        )

        # Turn statistics
        total_turns = sum(all_turn_counts) if all_turn_counts else None
        avg_turns_per_rep = (
            total_turns / len(all_turn_counts)
            if all_turn_counts
            else None
        )
        min_turns = min(all_turn_counts) if all_turn_counts else None
        max_turns = max(all_turn_counts) if all_turn_counts else None
        avg_turns_per_task = (
            sum(task_avg_turns) / len(task_avg_turns)
            if task_avg_turns
            else None
        )

        return {
            "total_tasks": total_tasks,
            "total_repetitions": total_repetitions,
            "completed_repetitions": completed_repetitions,
            "successful_repetitions": successful_repetitions,
            "failed_repetitions": failed_repetitions,
            "running_repetitions": running_repetitions,
            "pending_repetitions": pending_repetitions,
            "success_rate": round(success_rate, 4),
            "total_elapsed_time": round(total_elapsed_time, 2),
            "avg_time_per_repetition": round(avg_time_per_rep, 2) if avg_time_per_rep is not None else None,
            "avg_time_per_task": round(avg_time_per_task, 2) if avg_time_per_task is not None else None,
            "min_time": round(min_time, 2) if min_time is not None else None,
            "max_time": round(max_time, 2) if max_time is not None else None,
            "total_turns": total_turns,
            "avg_turns_per_repetition": round(avg_turns_per_rep, 2) if avg_turns_per_rep is not None else None,
            "avg_turns_per_task": round(avg_turns_per_task, 2) if avg_turns_per_task is not None else None,
            "min_turns": min_turns,
            "max_turns": max_turns,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stats": self.get_stats(),
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }


class LiveResultTracker:
    """
    Tracks live results and writes them to result.json.

    Automatically registers with the HookManager to receive events.
    """

    def __init__(
        self,
        scenario_name: str,
        results_dir: str,
        auto_register: bool = True,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            scenario_name: Name of the scenario being run
            results_dir: Directory where result.json will be written
            auto_register: Whether to automatically register with HookManager
        """
        self.results_dir = results_dir
        self.result_file = os.path.join(results_dir, "result.json")
        self.run_result = RunResult(
            scenario_name=scenario_name,
            results_dir=results_dir,
        )
        self._lock = threading.Lock()
        self._registered = False

        if auto_register:
            self.register()

    def register(self) -> None:
        """Register with the HookManager to receive events."""
        if self._registered:
            return

        hook_manager = get_hook_manager()
        hook_manager.register(None, self._handle_event)
        self._registered = True

    def unregister(self) -> None:
        """Unregister from the HookManager."""
        if not self._registered:
            return

        hook_manager = get_hook_manager()
        hook_manager.unregister(None, self._handle_event)
        self._registered = False

    def _handle_event(self, event: Event) -> None:
        """Handle an incoming event."""
        with self._lock:
            if event.event_type == EventType.RUN_START:
                self._on_run_start(event)
            elif event.event_type == EventType.RUN_END:
                self._on_run_end(event)
            elif event.event_type == EventType.SCENARIO_START:
                self._on_scenario_start(event)
            elif event.event_type == EventType.SCENARIO_END:
                self._on_scenario_end(event)
            elif event.event_type == EventType.REPETITION_START:
                self._on_repetition_start(event)
            elif event.event_type == EventType.REPETITION_END:
                self._on_repetition_end(event)
            elif event.event_type == EventType.EXECUTION_START:
                self._on_execution_start(event)
            elif event.event_type == EventType.EXECUTION_END:
                self._on_execution_end(event)
            elif event.event_type == EventType.EXECUTION_PROGRESS:
                self._on_execution_progress(event)

            # Write updated results after each event
            self._write_results()

    def _on_run_start(self, event: Event) -> None:
        """Handle RUN_START event."""
        self.run_result.status = "running"
        self.run_result.start_time = event.timestamp.isoformat()

    def _on_run_end(self, event: Event) -> None:
        """Handle RUN_END event."""
        self.run_result.status = "completed"
        self.run_result.end_time = event.timestamp.isoformat()

    def _on_scenario_start(self, event: Event) -> None:
        """Handle SCENARIO_START event."""
        # Scenario start doesn't need special handling for now
        pass

    def _on_scenario_end(self, event: Event) -> None:
        """Handle SCENARIO_END event."""
        # Scenario end doesn't need special handling for now
        pass

    def _on_repetition_start(self, event: Event) -> None:
        """Handle REPETITION_START event."""
        if event.task_id is None or event.repetition_id is None:
            return

        # Ensure task exists
        if event.task_id not in self.run_result.tasks:
            self.run_result.tasks[event.task_id] = TaskResult(task_id=event.task_id)

        task = self.run_result.tasks[event.task_id]

        # Create or update repetition
        if event.repetition_id not in task.repetitions:
            task.repetitions[event.repetition_id] = RepetitionResult()

        rep = task.repetitions[event.repetition_id]
        rep.status = "running"
        rep.start_time = event.timestamp.isoformat()

    def _on_repetition_end(self, event: Event) -> None:
        """Handle REPETITION_END event."""
        if event.task_id is None or event.repetition_id is None:
            return

        # Ensure task and repetition exist
        if event.task_id not in self.run_result.tasks:
            self.run_result.tasks[event.task_id] = TaskResult(task_id=event.task_id)

        task = self.run_result.tasks[event.task_id]

        if event.repetition_id not in task.repetitions:
            task.repetitions[event.repetition_id] = RepetitionResult()

        rep = task.repetitions[event.repetition_id]
        rep.status = "completed"
        rep.success = event.success
        rep.elapsed_time = event.elapsed_time
        rep.turns = event.turns
        rep.end_time = event.timestamp.isoformat()

    def _on_execution_start(self, event: Event) -> None:
        """Handle EXECUTION_START event."""
        # For now, this is handled the same as repetition start
        pass

    def _on_execution_end(self, event: Event) -> None:
        """Handle EXECUTION_END event."""
        # For now, this is handled the same as repetition end
        pass

    def _on_execution_progress(self, event: Event) -> None:
        """Handle EXECUTION_PROGRESS event."""
        # Progress events can be used for real-time updates
        # Currently just triggers a write
        pass

    def _write_results(self) -> None:
        """Write current results to result.json."""
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Write atomically using temp file
        temp_file = self.result_file + ".tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.run_result.to_dict(), f, indent=2)
                # Explicitly flush and sync to ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, self.result_file)
        except Exception as e:
            print(f"Warning: Failed to write result.json: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def get_result(self) -> RunResult:
        """Get the current run result."""
        return self.run_result

    def get_result_dict(self) -> Dict[str, Any]:
        """Get the current run result as a dictionary."""
        with self._lock:
            return self.run_result.to_dict()


def load_live_results(results_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load live results from a result.json file.

    Args:
        results_dir: Directory containing result.json

    Returns:
        Dictionary of results, or None if not found
    """
    result_file = os.path.join(results_dir, "result.json")
    if not os.path.exists(result_file):
        return None

    try:
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load result.json: {e}")
        return None
