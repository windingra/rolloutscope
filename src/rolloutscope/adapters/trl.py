"""TRL adapter.

Two artifacts are supported, independently or together:

1. ``trainer_state.json`` — ``log_history`` already contains ``entropy``,
   ``completions/mean_length``, ``frac_reward_zero_std`` and ``reward``. That is
   enough for the entropy / length / zero-advantage rules to run even without a
   single stored completion.
2. A completions table written by ``log_completions`` (jsonl / csv / parquet)
   with ``prompt`` + ``completion`` + ``reward`` columns. GRPO groups are
   reconstructed by hashing the prompt text, which is exactly the grouping the
   trainer used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import pandas as pd

from ..schema import (
    INVALID_EMPTY,
    INVALID_TRUNCATED,
    STATUS_INVALID,
    STATUS_OK,
    AgentTrace,
    Step,
    TokenUsage,
    TraceBatch,
)
from .base import list_files, read_jsonl

_METRIC_KEYS = {
    "entropy": ["entropy"],
    "completions/mean_length": ["completions/mean_length", "completions_mean_length"],
    "frac_reward_zero_std": ["frac_reward_zero_std"],
    "reward": ["reward", "rewards/mean"],
    "reward_std": ["reward_std", "rewards/std"],
    "kl": ["kl"],
    "loss": ["loss"],
}


def load(path: str, **_: Any) -> TraceBatch:
    files = list_files(path)
    state_files = [f for f in files if f.endswith(".json") and not f.endswith(".jsonl")]
    table_files = [f for f in files if f.endswith((".jsonl", ".csv", ".parquet"))]

    metrics: dict[str, dict[int, float]] = {}
    metadata: dict[str, Any] = {}
    for f in state_files:
        m, meta = _read_trainer_state(f)
        if m:
            metrics.update(m)
            metadata.update(meta)
            break

    traces: list[AgentTrace] = []
    for f in table_files:
        traces.extend(_read_completions(f))

    if not traces and not metrics:
        raise ValueError(
            f"no TRL artifacts found under {path!r} (expected trainer_state.json or a completions table)"
        )

    return TraceBatch(
        traces=traces,
        name=os.path.basename(os.path.normpath(path)) or "trl-run",
        framework="trl",
        source=path,
        metrics=metrics,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# trainer_state.json
# ---------------------------------------------------------------------------
def _read_trainer_state(path: str) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}, {}
    history = state.get("log_history") or []
    metrics: dict[str, dict[int, float]] = {}
    for row in history:
        step = row.get("step")
        if step is None:
            continue
        for canonical, aliases in _METRIC_KEYS.items():
            for alias in aliases:
                if alias in row and isinstance(row[alias], (int, float)):
                    metrics.setdefault(canonical, {})[int(step)] = float(row[alias])
                    break
    meta = {
        "source": "trl.trainer_state",
        "epoch": state.get("epoch"),
        "global_step": state.get("global_step"),
    }
    return metrics, meta


# ---------------------------------------------------------------------------
# completions table
# ---------------------------------------------------------------------------
def _read_completions(path: str) -> list[AgentTrace]:
    if path.endswith(".jsonl"):
        rows = read_jsonl(path)
    elif path.endswith(".csv"):
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        rows = pd.read_parquet(path).to_dict(orient="records")
    return _rows_to_traces(rows)


def _rows_to_traces(rows: list[dict[str, Any]]) -> list[AgentTrace]:
    traces: list[AgentTrace] = []
    seen: dict[tuple[Any, str], int] = {}
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        if isinstance(prompt, list):
            prompt = _last_user(prompt)
        prompt = None if prompt is None else str(prompt)
        completion = row.get("completion")
        if isinstance(completion, list):
            completion = "\n".join(str(c) for c in completion)
        completion = "" if completion is None else str(completion)

        reward = _reward(row)
        step = _int(row.get("step", row.get("global_step", 0)), 0)
        key = (step, _digest(prompt or ""))
        idx = seen.get(key, 0)
        seen[key] = idx + 1

        tokens = _int(row.get("completion_tokens"), None)
        if tokens is None:
            ids = row.get("completion_ids")
            if isinstance(ids, list):
                tokens = len(ids)
        estimated = tokens is None
        if estimated:
            tokens = max(1, math.ceil(len(completion) / 4)) if completion else 0

        finish = row.get("finish_reason")
        truncated = isinstance(finish, str) and finish.lower() in {"length", "max_tokens"}
        status, reason = STATUS_OK, None
        if tokens == 0:
            status, reason = STATUS_INVALID, INVALID_EMPTY
        elif truncated:
            status, reason = STATUS_INVALID, INVALID_TRUNCATED

        traces.append(
            AgentTrace(
                trace_id=str(row.get("trace_id") or f"trl-{i}"),
                framework="trl",
                step=step,
                group_id=_digest(prompt or f"row-{i}"),
                rollout_index=idx,
                prompt_id=_digest(prompt or f"row-{i}"),
                prompt=prompt,
                steps=[
                    Step(
                        index=0,
                        text=completion,
                        finish_reason=finish if isinstance(finish, str) else None,
                        token_usage=TokenUsage(output_tokens=tokens),
                    )
                ],
                reward=reward,
                status=status,
                invalid_reason=reason,
                metadata={"source": "trl.completions", "token_estimate": estimated},
            )
        )
    return traces


def _reward(row: dict[str, Any]) -> float | None:
    for key in ("reward", "rewards", "score", "reward_total"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and val:
            nums = [float(v) for v in val.values() if isinstance(v, (int, float))]
            if nums:
                return float(sum(nums))
    per_func = [k for k in row if str(k).startswith("rewards/")]
    if per_func:
        nums = [float(row[k]) for k in per_func if isinstance(row[k], (int, float))]
        if nums:
            return float(sum(nums))
    return None


def _last_user(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() in {"user", "human"}:
            return str(msg.get("content", ""))
    return None


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
