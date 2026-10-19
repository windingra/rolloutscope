"""Framework adapters. All of them return a :class:`TraceBatch`."""

from .base import load, read_jsonl, sniff

__all__ = ["load", "read_jsonl", "sniff"]
