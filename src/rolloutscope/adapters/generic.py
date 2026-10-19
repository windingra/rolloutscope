"""Generic adapter: the native ``AgentTrace`` jsonl (or a JSON list of traces)."""

from __future__ import annotations

import json
import os
from typing import Any

from ..schema import AgentTrace, TraceBatch
from .base import read_jsonl


def load(path: str, **_: Any) -> TraceBatch:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = payload.get("traces", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("expected a JSON list of traces or {'traces': [...]}")
    else:
        rows = read_jsonl(path)

    traces = [AgentTrace.from_dict(r, i) for i, r in enumerate(rows) if isinstance(r, dict)]
    if not traces:
        raise ValueError(f"no traces found in {path!r}")
    framework = traces[0].framework or "generic"
    run_ids = {t.run_id for t in traces if t.run_id}
    name = run_ids.pop() if len(run_ids) == 1 else os.path.splitext(os.path.basename(path))[0]
    return TraceBatch(
        traces=traces,
        name=name or "traces",
        framework=framework,
        source=path,
    )
