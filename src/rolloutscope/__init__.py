"""rolloutscope — a linter for RL rollouts."""

from __future__ import annotations

from .adapters import load, sniff
from .diagnostics import Config, analyze
from .schema import SCHEMA_VERSION, AgentTrace, Step, TokenUsage, ToolCall, TraceBatch

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AgentTrace",
    "Config",
    "Step",
    "TokenUsage",
    "ToolCall",
    "TraceBatch",
    "__version__",
    "analyze",
    "load",
    "sniff",
]
