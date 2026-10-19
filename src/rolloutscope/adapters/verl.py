"""verl adapter.

Reads what ``verl`` already writes: a ``DataProto.save_to_disk`` directory
(parquet) or a jsonl dump of the same columns.

Columns used, all optional except one of ``responses`` / ``rm_scores``:

    uid / index        -> group id (GRPO groups on uid) and rollout index
    responses          -> generated token ids  -> output_tokens, truncation
    response_mask      -> valid token mask
    rollout_log_probs  -> mean -log p(token)   -> entropy proxy
    rm_scores          -> scalar reward (value at the last valid position)
    advantages         -> used directly when present
    raw_prompt         -> prompt text
    step / global_step -> training step

verl has no text column, so the report shows token counts unless you pass a
tokenizer. That is deliberate: the diagnostics do not need decoded text.
"""

from __future__ import annotations

import ast
import os
from typing import Any

import numpy as np

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


def load(path: str, tokenizer: Any = None, max_traces: int | None = None, **_: Any) -> TraceBatch:
    rows = _read_rows(path)
    if max_traces:
        rows = rows[:max_traces]
    tok = _load_tokenizer(tokenizer)
    traces = [_row_to_trace(r, i, tok) for i, r in enumerate(rows)]
    return TraceBatch(traces=traces, name=_run_name(path), framework="verl", source=path)


def _read_rows(path: str) -> list[dict[str, Any]]:
    files = list_files(path)
    parquet = [f for f in files if f.endswith(".parquet")]
    if parquet:
        import pandas as pd

        frames = [pd.read_parquet(f) for f in parquet]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        return df.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    for f in files:
        if f.endswith(".jsonl"):
            rows.extend(read_jsonl(f))
    return rows


def _run_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or "verl-run"


# ---------------------------------------------------------------------------
# column helpers — verl dumps come back as numpy arrays, lists, bytes or str
# ---------------------------------------------------------------------------
def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        try:
            return np.frombuffer(value, dtype=np.int64).tolist()
        except ValueError:
            return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def _numeric(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=object)
    elif isinstance(value, (bytes, bytearray)):
        arr = np.frombuffer(value, dtype=np.float32)
    elif isinstance(value, str):
        try:
            arr = np.asarray(ast.literal_eval(value), dtype=object)
        except (ValueError, SyntaxError):
            return np.array([])
    else:
        arr = np.asarray([value])
    try:
        return arr.astype(float).ravel()
    except (TypeError, ValueError):
        return np.array([])


def _last_valid_score(scores: Any, mask: np.ndarray) -> float | None:
    arr = _numeric(scores)
    if arr.size == 0:
        return None
    if mask.size and mask.size == arr.size:
        valid = np.flatnonzero(mask > 0)
        if valid.size:
            return float(arr[valid[-1]])
    nonzero = np.flatnonzero(arr != 0)
    if nonzero.size:
        return float(arr[nonzero[-1]])
    return float(arr[-1])


def _get(row: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _prompt_text(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    seq = _to_list(raw) if isinstance(raw, (list, tuple, np.ndarray)) else []
    if not seq:
        return None
    for msg in reversed(seq):
        if isinstance(msg, dict):
            content = msg.get("content")
            if content is None and isinstance(msg.get("parts"), list):
                content = "".join(str(p.get("content", "")) for p in msg["parts"] if isinstance(p, dict))
            if content:
                return str(content)
    return None


def _row_to_trace(row: dict[str, Any], index: int, tokenizer: Any) -> AgentTrace:
    responses = _to_list(_get(row, "responses", "completion_ids", "response"))
    mask = _numeric(_get(row, "response_mask", "loss_mask", "completion_mask"))
    if mask.size == 0:
        mask = np.ones(len(responses), dtype=float)
    valid = int(mask.sum()) if mask.size else len(responses)

    logprobs = _numeric(_get(row, "rollout_log_probs", "log_probs", "old_log_probs"))
    lp_list = None
    if logprobs.size:
        if logprobs.size == mask.size:
            lp_list = logprobs[mask > 0].tolist()
        else:
            lp_list = logprobs[:valid].tolist() if valid else logprobs.tolist()

    reward = _last_valid_score(_get(row, "rm_scores", "token_level_scores", "scores"), mask)
    if reward is None:
        raw_reward = _get(row, "reward", "score", "rewards")
        nums = _numeric(raw_reward)
        reward = float(nums.ravel()[0]) if nums.size else None

    advantage = _numeric(_get(row, "advantages", "advantage"))
    advantage_val = float(advantage.ravel()[0]) if advantage.size else None

    group_raw = _get(row, "uid", "group_id", "index", "data_source")
    group_id = None if group_raw is None else str(group_raw)

    step_raw = _get(row, "step", "global_step", "global_steps")
    try:
        step = int(step_raw) if step_raw is not None else 0
    except (TypeError, ValueError):
        step = 0

    text = ""
    if tokenizer is not None and responses:
        try:
            text = tokenizer.decode(responses[:valid] or responses, skip_special_tokens=True)
        except Exception:
            text = ""

    finish = _get(row, "finish_reason", "stop_reason")
    truncated = bool(mask.size and mask.size == len(mask) and valid >= len(mask) and len(mask) >= 32)
    if isinstance(finish, str) and finish.lower() in {"length", "max_tokens", "max_length"}:
        truncated = True

    status, reason = STATUS_OK, None
    if valid == 0:
        status, reason = STATUS_INVALID, INVALID_EMPTY
    elif truncated:
        status, reason = STATUS_INVALID, INVALID_TRUNCATED

    step_obj = Step(
        index=0,
        text=text,
        finish_reason=finish if isinstance(finish, str) else ("length" if truncated else None),
        model=_get(row, "model", "model_name"),
        token_usage=TokenUsage(input_tokens=0, output_tokens=valid),
        logprobs=lp_list,
    )

    return AgentTrace(
        trace_id=str(_get(row, "trace_id", "uid", "request_id") or f"verl-{index}"),
        framework="verl",
        run_id=_run_name_from_row(row),
        step=step,
        group_id=group_id,
        rollout_index=index,
        prompt_id=group_id,
        prompt=_prompt_text(_get(row, "raw_prompt", "prompt", "prompts")),
        steps=[step_obj],
        reward=reward,
        advantage=advantage_val,
        status=status,
        invalid_reason=reason,
        metadata={"source": "verl", "has_text": bool(text)},
    )


def _run_name_from_row(row: dict[str, Any]) -> str | None:
    v = _get(row, "experiment_name", "project_name", "run_id")
    return None if v is None else str(v)


def _load_tokenizer(tokenizer: Any):
    if tokenizer is None or not isinstance(tokenizer, str):
        return tokenizer
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    except Exception:
        return None
