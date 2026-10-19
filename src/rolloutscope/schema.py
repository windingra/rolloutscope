"""Core data model for rolloutscope.

The model is intentionally a *superset* of the OpenTelemetry GenAI semantic
conventions: every trace-level concept maps 1:1 onto a ``gen_ai.*`` span or
attribute, and the RL-specific dimensions that OTel does not define yet are
carried in ``rolloutscope.*`` attributes (see :mod:`rolloutscope.otel`).

    TraceBatch
    └── AgentTrace            one rollout == one sample inside a GRPO group
        └── Step              one LLM call == one ``chat`` span
            └── ToolCall      one tool execution == one ``execute_tool`` span
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "0.1"

#: ``status`` values for a rollout.
STATUS_OK = "ok"
STATUS_INVALID = "invalid"
STATUS_ERROR = "error"

#: ``invalid_reason`` vocabulary (kept stable so rules can match on it).
INVALID_TRUNCATED = "truncated"
INVALID_EMPTY = "empty"
INVALID_TOOL_FORMAT = "tool_format"
INVALID_TOOL_ERROR = "tool_error"
INVALID_TOOL_LOOP = "tool_loop"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TokenUsage:
    """Token accounting, mirrors ``gen_ai.usage.*``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> TokenUsage:
        if raw is None:
            return cls()
        if isinstance(raw, TokenUsage):
            return raw
        if isinstance(raw, (int, float)):
            return cls(output_tokens=int(raw))
        return cls(
            input_tokens=_as_int(raw.get("input_tokens")) or 0,
            output_tokens=_as_int(raw.get("output_tokens")) or 0,
            cached_input_tokens=_as_int(raw.get("cached_input_tokens")) or 0,
        )


@dataclass
class ToolCall:
    """One tool execution, mirrors an ``execute_tool`` span."""

    name: str = ""
    call_id: str = ""
    arguments: str = ""
    result: str | None = None
    error: str | None = None
    duration_ms: float | None = None

    @property
    def args_dict(self) -> dict[str, Any] | None:
        """Parsed arguments, or ``None`` when the model emitted invalid JSON.

        A ``None`` here is a first-class diagnostic signal (``tool_format``),
        not an exception: broken tool syntax is one of the most common ways a
        rollout becomes useless.
        """
        if isinstance(self.arguments, dict):  # tolerated for convenience
            return self.arguments
        text = (self.arguments or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @property
    def is_format_error(self) -> bool:
        return self.args_dict is None

    @property
    def signature(self) -> tuple[str, str]:
        return (self.name, (self.arguments or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> ToolCall:
        if isinstance(raw, ToolCall):
            return raw
        args = raw.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        return cls(
            name=str(raw.get("name") or raw.get("tool_name") or ""),
            call_id=str(raw.get("call_id") or raw.get("id") or ""),
            arguments=args or "",
            result=raw.get("result"),
            error=raw.get("error"),
            duration_ms=_as_float(raw.get("duration_ms")),
        )


@dataclass
class Step:
    """One assistant turn, i.e. one ``chat`` span."""

    index: int = 0
    text: str = ""
    finish_reason: str | None = None
    model: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    logprobs: list[float] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float | None = None
    #: Aggregate ``-log p`` when per-token logprobs are unavailable (e.g. after a
    #: round-trip through OTel, which has no logprob convention).
    mean_nll_hint: float | None = None

    @property
    def output_tokens(self) -> int:
        return self.token_usage.output_tokens

    @property
    def mean_token_nll(self) -> float | None:
        """Mean ``-log p(token)`` over generated tokens.

        NOTE: this is the negative log-likelihood of the *sampled* tokens, not
        the entropy of the policy. E[NLL] >= H(pi), so it is a usable upper
        bound proxy for tracking collapse, but the report labels it ``proxy``
        to avoid over-claiming.
        """
        if not self.logprobs:
            return self.mean_nll_hint
        vals = [-lp for lp in self.logprobs if lp is not None]
        if not vals:
            return self.mean_nll_hint
        return sum(vals) / len(vals)

    @property
    def is_truncated(self) -> bool:
        return (self.finish_reason or "").lower() in {"length", "max_tokens", "max_length"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "token_usage": self.token_usage.to_dict(),
            "logprobs": self.logprobs,
            "mean_nll": self.mean_token_nll,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, raw: Any, index: int = 0) -> Step:
        if isinstance(raw, Step):
            return raw
        logprobs = raw.get("logprobs")
        return cls(
            index=_as_int(raw.get("index")) if raw.get("index") is not None else index,
            text=raw.get("text") or raw.get("content") or "",
            finish_reason=raw.get("finish_reason"),
            model=raw.get("model"),
            token_usage=TokenUsage.from_dict(raw.get("token_usage") or raw.get("usage")),
            logprobs=logprobs,
            tool_calls=[ToolCall.from_dict(t) for t in (raw.get("tool_calls") or [])],
            latency_ms=_as_float(raw.get("latency_ms")),
            mean_nll_hint=_as_float(raw.get("mean_nll")) if not logprobs else None,
        )


@dataclass
class AgentTrace:
    """One rollout: a full multi-turn agent run for a single sample."""

    trace_id: str
    framework: str = "generic"
    # --- RL linkage -------------------------------------------------------
    run_id: str | None = None
    step: int | None = None
    group_id: str | None = None
    rollout_index: int | None = None
    prompt_id: str | None = None
    # --- content ----------------------------------------------------------
    prompt: str | None = None
    steps: list[Step] = field(default_factory=list)
    # --- outcome ----------------------------------------------------------
    reward: float | None = None
    advantage: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    status: str = STATUS_OK
    invalid_reason: str | None = None
    # --- bookkeeping ------------------------------------------------------
    agent_name: str = "agent"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def input_tokens(self) -> int:
        return sum(s.token_usage.input_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def num_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def num_turns(self) -> int:
        return len(self.steps)

    @property
    def all_tool_calls(self) -> Iterator[ToolCall]:
        for s in self.steps:
            yield from s.tool_calls

    @property
    def mean_token_nll(self) -> float | None:
        vals = [s.mean_token_nll for s in self.steps if s.mean_token_nll is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def is_truncated(self) -> bool:
        return any(s.is_truncated for s in self.steps)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.steps if s.text)

    @property
    def group_key(self) -> str:
        return self.group_id or self.prompt_id or self.trace_id

    @property
    def tool_error_count(self) -> int:
        return sum(1 for t in self.all_tool_calls if t.error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "framework": self.framework,
            "run_id": self.run_id,
            "step": self.step,
            "group_id": self.group_id,
            "rollout_index": self.rollout_index,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "steps": [s.to_dict() for s in self.steps],
            "reward": self.reward,
            "advantage": self.advantage,
            "reward_components": self.reward_components,
            "status": self.status,
            "invalid_reason": self.invalid_reason,
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int = 0) -> AgentTrace:
        def _dt(v: Any) -> datetime | None:
            if not v:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except ValueError:
                return None

        return cls(
            trace_id=str(raw.get("trace_id") or raw.get("id") or f"trace-{index}"),
            framework=str(raw.get("framework") or "generic"),
            run_id=raw.get("run_id"),
            step=_as_int(raw.get("step")),
            group_id=None if raw.get("group_id") is None else str(raw.get("group_id")),
            rollout_index=_as_int(raw.get("rollout_index")),
            prompt_id=None if raw.get("prompt_id") is None else str(raw.get("prompt_id")),
            prompt=raw.get("prompt"),
            steps=[Step.from_dict(s, i) for i, s in enumerate(raw.get("steps") or [])],
            reward=_as_float(raw.get("reward")),
            advantage=_as_float(raw.get("advantage")),
            reward_components={
                str(k): float(v)
                for k, v in (raw.get("reward_components") or {}).items()
                if _as_float(v) is not None
            },
            status=str(raw.get("status") or STATUS_OK),
            invalid_reason=raw.get("invalid_reason"),
            agent_name=str(raw.get("agent_name") or "agent"),
            started_at=_dt(raw.get("started_at")),
            ended_at=_dt(raw.get("ended_at")),
            metadata=raw.get("metadata") or {},
        )


@dataclass
class TraceBatch:
    """A collection of traces from one analysis scope."""

    traces: list[AgentTrace] = field(default_factory=list)
    name: str = "run"
    framework: str = "generic"
    source: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    #: Scalar metrics logged by the trainer, keyed by metric name then step,
    #: e.g. ``{"entropy": {0: 1.9, 1: 1.8}, "train_loss": {...}}``.
    metrics: dict[str, dict[int, float]] = field(default_factory=dict)
    #: Free-form run metadata (model name, dataset, num_rollouts, ...).
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self) -> Iterator[AgentTrace]:
        return iter(self.traces)

    def __getitem__(self, item):
        return self.traces[item]

    def extend(self, traces: Iterable[AgentTrace]) -> None:
        self.traces.extend(traces)

    @property
    def steps(self) -> list[int]:
        return sorted({t.step for t in self.traces if t.step is not None})

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(t.to_dict(), ensure_ascii=False) + "\n" for t in self.traces)

    @classmethod
    def from_jsonl(cls, path: str, **kwargs: Any) -> TraceBatch:
        traces: list[AgentTrace] = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    traces.append(AgentTrace.from_dict(json.loads(line), i))
        return cls(traces=traces, **kwargs)

    def summary(self) -> dict[str, Any]:
        n = len(self.traces)
        return {
            "name": self.name,
            "framework": self.framework,
            "traces": n,
            "steps": len(self.steps),
            "groups": len({t.group_key for t in self.traces}),
            "tool_calls": sum(t.num_tool_calls for t in self.traces),
            "total_tokens": sum(t.total_tokens for t in self.traces),
        }
