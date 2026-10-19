"""Vectorized flattening of traces into pandas frames.

Every rule reads from these two frames instead of walking Python objects, so a
100k-trace scan stays fast:

* ``trace_frame``  — one row per rollout
* ``group_frame``  — one row per (step, group) with reward statistics
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    INVALID_TOOL_FORMAT,
    INVALID_TOOL_LOOP,
    INVALID_TRUNCATED,
    STATUS_OK,
    AgentTrace,
    TraceBatch,
)

#: Tool-call syntax that leaked into plain text instead of structured
#: ``tool_calls`` — the classic chat-template / parser mismatch in agent RL.
_TEXT_TOOL_PATTERNS = re.compile(
    r"(<tool_call>|<\|python_tag\|>|<\|tool_call\|>|\"tool_calls\"\s*:|"
    r"<function[=\s]|\bfunction_call\b)",
    re.IGNORECASE,
)

TRACE_COLUMNS = [
    "trace_id",
    "framework",
    "run_id",
    "step",
    "group_key",
    "rollout_index",
    "prompt",
    "reward",
    "advantage",
    "status",
    "invalid_reason",
    "output_tokens",
    "input_tokens",
    "total_tokens",
    "num_tool_calls",
    "num_turns",
    "tool_format_errors",
    "tool_errors",
    "tool_loop",
    "truncated",
    "mean_nll",
    "text_len",
    "last_finish_reason",
    "text_tool_syntax",
]


def _trace_row(t: AgentTrace) -> dict[str, Any]:
    calls = list(t.all_tool_calls)
    format_errors = sum(1 for c in calls if c.is_format_error)
    text_tool_syntax = False
    if not calls and t.text:
        text_tool_syntax = bool(_TEXT_TOOL_PATTERNS.search(t.text))
    if text_tool_syntax:
        format_errors += 1
    return {
        "trace_id": t.trace_id,
        "framework": t.framework,
        "run_id": t.run_id,
        "step": t.step if t.step is not None else 0,
        "group_key": t.group_key,
        "rollout_index": t.rollout_index,
        "prompt": t.prompt,
        "reward": t.reward,
        "advantage": t.advantage,
        "status": t.status,
        "invalid_reason": t.invalid_reason,
        "output_tokens": t.output_tokens,
        "input_tokens": t.input_tokens,
        "total_tokens": t.total_tokens,
        "num_tool_calls": len(calls),
        "num_turns": t.num_turns,
        "tool_format_errors": format_errors,
        "tool_errors": sum(1 for c in calls if c.error),
        "tool_loop": _has_tool_loop(calls),
        "truncated": t.is_truncated,
        "mean_nll": t.mean_token_nll,
        "text_len": len(t.text),
        "last_finish_reason": t.steps[-1].finish_reason if t.steps else None,
        "text_tool_syntax": text_tool_syntax,
    }


def _has_tool_loop(calls: list, repeat: int = 3) -> bool:
    if len(calls) < repeat:
        return False
    run = 1
    for prev, cur in zip(calls, calls[1:]):
        run = run + 1 if prev.signature == cur.signature else 1
        if run >= repeat:
            return True
    return False


def trace_frame(batch: TraceBatch) -> pd.DataFrame:
    rows = [_trace_row(t) for t in batch.traces]
    df = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    for col in ("reward", "advantage", "mean_nll", "rollout_index"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("output_tokens", "input_tokens", "total_tokens", "num_tool_calls", "num_turns"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype("int64")
    return df


def infer_pass_threshold(rewards: pd.Series) -> float:
    """Guess the reward value that means "solved".

    Correctness rewards are almost always in ``[0, 1]`` with 1 = solved, so that
    is the default; otherwise the midpoint of the observed range is used. Users
    can always override with ``--pass-threshold``.
    """
    vals = rewards.dropna()
    if vals.empty:
        return 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return hi
    if hi >= 0.999 and lo >= 0:
        return 1.0
    return (lo + hi) / 2.0


def group_frame(df: pd.DataFrame, pass_threshold: float, zero_std: float = 1e-6) -> pd.DataFrame:
    """One row per (step, group) with the statistics GRPO actually cares about."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "group_key",
                "n",
                "reward_mean",
                "reward_std",
                "pass_rate",
                "tokens",
                "zero_adv",
                "saturated",
                "collapsed",
                "informative",
            ]
        )
    g = df.groupby(["step", "group_key"], dropna=False)
    out = g.agg(
        n=("trace_id", "size"),
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        pass_rate=("reward", lambda s: float((s >= pass_threshold).mean()) if s.notna().any() else np.nan),
        tokens=("total_tokens", "sum"),
    ).reset_index()
    out["reward_std"] = out["reward_std"].fillna(0.0)
    out["zero_adv"] = out["reward_std"] < zero_std
    out["saturated"] = out["zero_adv"] & (out["pass_rate"] >= 1.0)
    out["collapsed"] = out["zero_adv"] & (out["pass_rate"] <= 0.0)
    out["informative"] = ~out["zero_adv"]
    return out


def step_frame(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Per-training-step aggregates used by the report charts."""
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("step").agg(
        traces=("trace_id", "size"),
        mean_reward=("reward", "mean"),
        mean_length=("output_tokens", "mean"),
        median_length=("output_tokens", "median"),
        p10_length=("output_tokens", lambda s: s.quantile(0.10)),
        p90_length=("output_tokens", lambda s: s.quantile(0.90)),
        mean_nll=("mean_nll", "mean"),
        mean_tool_calls=("num_tool_calls", "mean"),
        invalid=("status", lambda s: float((s != STATUS_OK).mean())),
    )
    gagg = groups.groupby("step").agg(
        groups=("group_key", "size"),
        zero_adv=("zero_adv", "mean"),
        saturated=("saturated", "mean"),
        collapsed=("collapsed", "mean"),
        informative=("informative", "mean"),
        tokens=("tokens", "sum"),
    )
    out = agg.join(gagg, how="outer").reset_index().sort_values("step")
    out["length_cv"] = df.groupby("step")["output_tokens"].apply(
        lambda s: float(s.std() / s.mean()) if s.mean() else 0.0
    ).reindex(out["step"]).to_numpy()
    return out


def advantage_zero_ratio(groups: pd.DataFrame) -> float:
    return float(groups["zero_adv"].mean()) if len(groups) else 0.0


def invalid_mask(df: pd.DataFrame) -> pd.Series:
    """Rollouts that cannot contribute a useful gradient for structural reasons."""
    return (
        (df["status"] != STATUS_OK)
        | df["truncated"]
        | (df["tool_format_errors"] > 0)
        | (df["output_tokens"] == 0)
        | (df["tool_loop"])
    )


def invalid_reason_series(df: pd.DataFrame) -> pd.Series:
    """Normalized invalid-reason label per rollout (``""`` when valid)."""
    reason = pd.Series([""] * len(df), index=df.index, dtype="object")
    reason[df["status"] != STATUS_OK] = df.loc[df["status"] != STATUS_OK, "invalid_reason"].fillna(
        "unknown"
    )
    reason[df["truncated"]] = INVALID_TRUNCATED
    reason[df["tool_format_errors"] > 0] = INVALID_TOOL_FORMAT
    reason[df["tool_loop"]] = INVALID_TOOL_LOOP
    reason[(df["output_tokens"] == 0) & (reason == "")] = "empty"
    return reason
