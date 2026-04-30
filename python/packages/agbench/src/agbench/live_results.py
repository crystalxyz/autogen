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
    rounds: Optional[int] = None
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_reasoning_tokens: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "rounds": self.rounds,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
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
        round_counts = []
        prompt_tokens_list = []
        completion_tokens_list = []
        reasoning_tokens_list = []

        for rep in self.repetitions.values():
            if rep.status == "completed":
                completed += 1
                if rep.success is True:
                    successful += 1
                elif rep.success is False:
                    failed += 1
                if rep.elapsed_time is not None:
                    elapsed_times.append(rep.elapsed_time)
                if rep.rounds is not None:
                    round_counts.append(rep.rounds)
                if rep.total_prompt_tokens is not None:
                    prompt_tokens_list.append(rep.total_prompt_tokens)
                if rep.total_completion_tokens is not None:
                    completion_tokens_list.append(rep.total_completion_tokens)
                if rep.total_reasoning_tokens is not None:
                    reasoning_tokens_list.append(rep.total_reasoning_tokens)
            elif rep.status == "running":
                running += 1

        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
        min_time = min(elapsed_times) if elapsed_times else None
        max_time = max(elapsed_times) if elapsed_times else None
        total_time = sum(elapsed_times) if elapsed_times else None

        avg_rounds = sum(round_counts) / len(round_counts) if round_counts else None
        min_rounds = min(round_counts) if round_counts else None
        max_rounds = max(round_counts) if round_counts else None
        total_rounds = sum(round_counts) if round_counts else None

        total_prompt_tokens = sum(prompt_tokens_list) if prompt_tokens_list else None
        total_completion_tokens = (
            sum(completion_tokens_list) if completion_tokens_list else None
        )
        total_reasoning_tokens = (
            sum(reasoning_tokens_list) if reasoning_tokens_list else None
        )

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
            "avg_rounds": round(avg_rounds, 2) if avg_rounds is not None else None,
            "min_rounds": min_rounds,
            "max_rounds": max_rounds,
            "total_rounds": total_rounds,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
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
        task_avg_times = []

        all_round_counts: list[int] = []
        all_prompt_tokens: list[int] = []
        all_completion_tokens: list[int] = []
        all_reasoning_tokens: list[int] = []
        task_avg_rounds: list[float] = []

        for task in self.tasks.values():
            task_times = []
            task_rounds: list[int] = []
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
                    if rep.rounds is not None:
                        all_round_counts.append(rep.rounds)
                        task_rounds.append(rep.rounds)
                    if rep.total_prompt_tokens is not None:
                        all_prompt_tokens.append(rep.total_prompt_tokens)
                    if rep.total_completion_tokens is not None:
                        all_completion_tokens.append(rep.total_completion_tokens)
                    if rep.total_reasoning_tokens is not None:
                        all_reasoning_tokens.append(rep.total_reasoning_tokens)
                elif rep.status == "running":
                    running_repetitions += 1
                else:
                    pending_repetitions += 1

            # Calculate per-task averages
            if task_times:
                task_avg_times.append(sum(task_times) / len(task_times))
            if task_rounds:
                task_avg_rounds.append(sum(task_rounds) / len(task_rounds))

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

        # Round statistics (scenario-defined; null when not provided)
        total_rounds = sum(all_round_counts) if all_round_counts else None
        avg_rounds_per_rep = (
            total_rounds / len(all_round_counts) if all_round_counts else None
        )
        min_rounds = min(all_round_counts) if all_round_counts else None
        max_rounds = max(all_round_counts) if all_round_counts else None
        avg_rounds_per_task = (
            sum(task_avg_rounds) / len(task_avg_rounds) if task_avg_rounds else None
        )

        # Token totals across all repetitions
        total_prompt_tokens = sum(all_prompt_tokens) if all_prompt_tokens else None
        total_completion_tokens = (
            sum(all_completion_tokens) if all_completion_tokens else None
        )
        total_reasoning_tokens = (
            sum(all_reasoning_tokens) if all_reasoning_tokens else None
        )
        avg_prompt_tokens_per_rep = (
            total_prompt_tokens / len(all_prompt_tokens) if all_prompt_tokens else None
        )
        avg_completion_tokens_per_rep = (
            total_completion_tokens / len(all_completion_tokens)
            if all_completion_tokens
            else None
        )
        avg_reasoning_tokens_per_rep = (
            total_reasoning_tokens / len(all_reasoning_tokens)
            if all_reasoning_tokens
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
            "total_rounds": total_rounds,
            "avg_rounds_per_repetition": round(avg_rounds_per_rep, 2) if avg_rounds_per_rep is not None else None,
            "avg_rounds_per_task": round(avg_rounds_per_task, 2) if avg_rounds_per_task is not None else None,
            "min_rounds": min_rounds,
            "max_rounds": max_rounds,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "avg_prompt_tokens_per_repetition": (
                round(avg_prompt_tokens_per_rep, 2)
                if avg_prompt_tokens_per_rep is not None
                else None
            ),
            "avg_completion_tokens_per_repetition": (
                round(avg_completion_tokens_per_rep, 2)
                if avg_completion_tokens_per_rep is not None
                else None
            ),
            "avg_reasoning_tokens_per_repetition": (
                round(avg_reasoning_tokens_per_rep, 2)
                if avg_reasoning_tokens_per_rep is not None
                else None
            ),
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
        rep.rounds = event.rounds
        rep.total_prompt_tokens = event.total_prompt_tokens
        rep.total_completion_tokens = event.total_completion_tokens
        rep.total_reasoning_tokens = event.total_reasoning_tokens
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
