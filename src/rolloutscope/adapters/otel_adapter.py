"""OTel adapter: ingest spans exported by any GenAI-instrumented agent.

Accepts OTLP/JSON (``{"resourceSpans": [...]}``) or a flat list of span dicts,
so data from OpenInference, Langfuse, Traceloop or a collector file-drop can be
diagnosed without touching the agent code.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..otel import from_otel
from ..schema import TraceBatch


def load(path: str, **_: Any) -> TraceBatch:
    payloads: list[Any] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                payloads.append(_read(os.path.join(path, name)))
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    payloads.append(json.loads(line))
    else:
        payloads.append(_read(path))

    batches = [from_otel(p, name=os.path.basename(os.path.normpath(path))) for p in payloads if p]
    if not batches:
        raise ValueError(f"no spans found in {path!r}")
    merged = batches[0]
    for b in batches[1:]:
        merged.extend(b.traces)
    merged.source = path
    merged.framework = "otel"
    return merged


def _read(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
