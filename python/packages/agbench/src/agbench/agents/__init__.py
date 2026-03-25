"""
Reusable agent implementations for AutoGenBench.

This module provides reusable agent components that can be imported
and used across different benchmark templates.

Available Agents:
    debating_agent: Multi-agent debate framework for collaborative problem solving
"""

from . import debating_agent

__all__ = ["debating_agent"]
