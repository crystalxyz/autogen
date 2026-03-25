"""
Event hook system for agbench streaming results.

Provides a publish-subscribe mechanism for lifecycle events during benchmark runs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading


class EventType(Enum):
    """Types of events emitted during benchmark execution."""

    RUN_START = "run_start"
    SCENARIO_START = "scenario_start"
    REPETITION_START = "repetition_start"
    EXECUTION_START = "execution_start"
    EXECUTION_PROGRESS = "execution_progress"
    EXECUTION_END = "execution_end"
    REPETITION_END = "repetition_end"
    SCENARIO_END = "scenario_end"
    RUN_END = "run_end"


@dataclass
class Event:
    """
    Represents an event in the benchmark lifecycle.

    Attributes:
        event_type: The type of event
        timestamp: When the event occurred
        task_id: The task/scenario instance ID (e.g., "HumanEval_0")
        repetition_id: The repetition number (e.g., 0, 1, 2)
        success: Whether the task succeeded (for end events)
        elapsed_time: Time taken in seconds (for end events)
        turns: Number of conversation turns (for end events)
        metadata: Additional event-specific data
    """

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    task_id: Optional[str] = None
    repetition_id: Optional[int] = None
    success: Optional[bool] = None
    elapsed_time: Optional[float] = None
    turns: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "repetition_id": self.repetition_id,
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "turns": self.turns,
            "metadata": self.metadata,
        }


# Type alias for hook callback functions
HookCallback = Callable[[Event], None]


class HookManager:
    """
    Manages event hooks for the benchmark system.

    Thread-safe implementation supporting multiple callbacks per event type.
    """

    def __init__(self) -> None:
        self._hooks: Dict[EventType, List[HookCallback]] = {et: [] for et in EventType}
        self._global_hooks: List[HookCallback] = []
        self._lock = threading.Lock()

    def register(
        self,
        event_type: Optional[EventType],
        callback: HookCallback,
    ) -> None:
        """
        Register a callback for an event type.

        Args:
            event_type: The event type to listen for, or None for all events
            callback: The function to call when the event occurs
        """
        with self._lock:
            if event_type is None:
                self._global_hooks.append(callback)
            else:
                self._hooks[event_type].append(callback)

    def unregister(
        self,
        event_type: Optional[EventType],
        callback: HookCallback,
    ) -> bool:
        """
        Unregister a callback for an event type.

        Args:
            event_type: The event type the callback was registered for
            callback: The callback to remove

        Returns:
            True if the callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type is None:
                if callback in self._global_hooks:
                    self._global_hooks.remove(callback)
                    return True
            else:
                if callback in self._hooks[event_type]:
                    self._hooks[event_type].remove(callback)
                    return True
            return False

    def emit(self, event: Event) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event: The event to emit
        """
        with self._lock:
            callbacks = list(self._hooks[event.event_type]) + list(self._global_hooks)

        # Call callbacks outside the lock to prevent deadlocks
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                # Log but don't crash on callback errors
                print(f"Warning: Hook callback error for {event.event_type.value}: {e}")

    def clear(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear all callbacks for an event type, or all callbacks if event_type is None.

        Args:
            event_type: The event type to clear, or None to clear all
        """
        with self._lock:
            if event_type is None:
                for et in EventType:
                    self._hooks[et] = []
                self._global_hooks = []
            else:
                self._hooks[event_type] = []


# Singleton instance
_hook_manager: Optional[HookManager] = None
_singleton_lock = threading.Lock()


def get_hook_manager() -> HookManager:
    """
    Get the singleton HookManager instance.

    Returns:
        The global HookManager instance
    """
    global _hook_manager
    if _hook_manager is None:
        with _singleton_lock:
            if _hook_manager is None:
                _hook_manager = HookManager()
    return _hook_manager


def reset_hook_manager() -> None:
    """
    Reset the singleton HookManager instance.

    Primarily useful for testing.
    """
    global _hook_manager
    with _singleton_lock:
        _hook_manager = None
