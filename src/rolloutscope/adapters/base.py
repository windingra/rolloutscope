"""Adapter layer: read whatever a trainer already wrote to disk.

The whole point is *zero instrumentation* — no re-run, no SDK, no config change.
Every adapter returns a :class:`~rolloutscope.schema.TraceBatch` and degrades
gracefully when a column is missing (a missing column should skip a rule, never
crash the report).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from ..schema import AgentTrace, TraceBatch

Adapter = Callable[..., TraceBatch]


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def list_files(path: str) -> list[str]:
    if os.path.isdir(path):
        out: list[str] = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith((".parquet", ".jsonl", ".json", ".csv")):
                    out.append(os.path.join(root, f))
        return sorted(out)
    return [path]


def sniff(path: str) -> str:
    """Guess the source format from the file layout and the first record."""
    files = list_files(path)
    if not files:
        raise FileNotFoundError(f"no .parquet/.jsonl/.json/.csv found under {path!r}")

    parquet = [f for f in files if f.endswith(".parquet")]
    if parquet:
        try:
            import pyarrow.parquet as pq

            cols = set(pq.ParquetFile(parquet[0]).schema_arrow.names)
        except Exception:
            cols = set()
        if {"responses", "input_ids"} & cols:
            return "verl"
        if {"completion", "reward"} <= cols or {"prompt", "completion"} <= cols:
            return "trl"
        return "parquet"

    first = files[0]
    if first.endswith(".json"):
        try:
            with open(first, encoding="utf-8") as fh:
                obj = json.load(fh)
        except json.JSONDecodeError:
            return "generic"
        if isinstance(obj, dict):
            if "resourceSpans" in obj:
                return "otel"
            if "log_history" in obj:
                return "trl-state"
            if "traces" in obj:
                return "generic"
        return "generic"

    if first.endswith(".csv"):
        return "trl"

    rows = read_jsonl(first)
    if not rows:
        return "generic"
    keys = set(rows[0].keys())
    if "steps" in keys:
        return "generic"
    if {"responses", "input_ids"} & keys:
        return "verl"
    if {"completion", "prompt"} <= keys or "completion" in keys:
        return "trl"
    return "generic"


def load(path: str, adapter: str = "auto", **kwargs: Any) -> TraceBatch:
    """Load a path with an explicit adapter or auto-detection."""
    from . import generic, otel_adapter, trl, verl

    name = adapter or "auto"
    if name == "auto":
        name = sniff(path)

    if name == "verl":
        batch = verl.load(path, **kwargs)
    elif name in ("trl", "trl-state"):
        batch = trl.load(path, **kwargs)
    elif name == "otel":
        batch = otel_adapter.load(path, **kwargs)
    elif name in ("generic", "jsonl") or name == "parquet":
        batch = generic.load(path, **kwargs)
    else:
        raise ValueError(f"unknown adapter {name!r}; try auto|verl|trl|generic|otel")
    batch.source = batch.source or path
    return batch


__all__ = ["Adapter", "AgentTrace", "list_files", "load", "read_jsonl", "sniff"]
