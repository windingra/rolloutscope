"""OpenTelemetry GenAI mapping.

Design rule: **do not invent a format**. A rollout is emitted as a normal OTel
trace that any GenAI-aware backend already understands:

    invoke_agent {agent}                     gen_ai.operation.name=invoke_agent
    ├── chat {model}                         gen_ai.operation.name=chat
    │   └── execute_tool {tool}              gen_ai.operation.name=execute_tool
    └── chat {model}

The RL dimensions OTel does not define yet (reward, advantage, group id,
training step, rollout index) ride along as ``rolloutscope.*`` attributes, so
nothing pollutes the ``gen_ai.*`` namespace. The mapping is bidirectional:
traces instrumented with OpenInference / Langfuse / Traceloop can be fed back
into the diagnostic engine through :func:`from_otel`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .schema import (
    SCHEMA_VERSION,
    STATUS_INVALID,
    AgentTrace,
    Step,
    TokenUsage,
    ToolCall,
    TraceBatch,
)

#: ``rolloutscope.*`` extension attributes. Documented in docs/PLAN.md §3.3 and
#: intended as a proposal for the OTel GenAI semantic conventions.
RL_ATTRS = {
    "schema": "rolloutscope.schema.version",
    "framework": "rolloutscope.framework",
    "step": "rolloutscope.train.step",
    "group": "rolloutscope.group.id",
    "rollout_index": "rolloutscope.rollout.index",
    "reward": "rolloutscope.reward",
    "advantage": "rolloutscope.advantage",
    "reward_components": "rolloutscope.reward.components",
    "status": "rolloutscope.status",
    "invalid_reason": "rolloutscope.invalid.reason",
}

SCOPE_NAME = "rolloutscope"


# ---------------------------------------------------------------------------
# attribute encoding (OTLP JSON)
# ---------------------------------------------------------------------------
def _any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_any_value(v) for v in value]}}
    return {"stringValue": "" if value is None else str(value)}


def _encode_attrs(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _any_value(v)} for k, v in attrs.items() if v is not None]


def _decode_value(node: dict[str, Any]) -> Any:
    if "stringValue" in node:
        return node["stringValue"]
    if "intValue" in node:
        try:
            return int(node["intValue"])
        except (TypeError, ValueError):
            return None
    if "doubleValue" in node:
        return float(node["doubleValue"])
    if "boolValue" in node:
        return bool(node["boolValue"])
    if "arrayValue" in node:
        return [_decode_value(v) for v in node["arrayValue"].get("values", [])]
    if "kvlistValue" in node:
        return {kv["key"]: _decode_value(kv["value"]) for kv in node["kvlistValue"].get("values", [])}
    return None


def _decode_attrs(raw: Any) -> dict[str, Any]:
    """Accept both OTLP ``[{"key","value"}]`` and plain ``{k: v}``."""
    if isinstance(raw, dict):
        return dict(raw)
    out: dict[str, Any] = {}
    for item in raw or []:
        if not isinstance(item, dict) or "key" not in item:
            continue
        out[item["key"]] = _decode_value(item.get("value") or {})
    return out


def _messages_to_text(messages: Any) -> str:
    """Best-effort flattening of OTel ``gen_ai.*.messages`` payloads."""
    if messages is None:
        return ""
    if isinstance(messages, str):
        text = messages.strip()
        if text.startswith(("[", "{")):
            try:
                return _messages_to_text(json.loads(text))
            except json.JSONDecodeError:
                pass
        return messages
    if isinstance(messages, dict):
        messages = [messages]
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        content = msg.get("content")
        if content is None and isinstance(msg.get("parts"), list):
            content = "".join(
                str(p.get("content", "")) for p in msg["parts"] if isinstance(p, dict)
            )
        if content is not None:
            parts.append(str(content))
    return "\n".join(p for p in parts if p)


def _first_user_text(messages: Any) -> str | None:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages or None
    if isinstance(messages, dict):
        messages = [messages]
    for msg in messages or []:
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() in {"user", "human"}:
            text = _messages_to_text([msg])
            if text:
                return text
    return None


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def trace_to_spans(trace: AgentTrace, base_time_ns: int | None = None) -> list[dict[str, Any]]:
    """Flatten one :class:`AgentTrace` into OTel span dicts."""
    root_id = f"{trace.trace_id}-root"[:32]
    now = base_time_ns if base_time_ns is not None else 1_700_000_000_000_000_000
    cursor = now
    spans: list[dict[str, Any]] = []

    root_attrs: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": trace.agent_name,
        "gen_ai.conversation.id": trace.trace_id,
        RL_ATTRS["schema"]: SCHEMA_VERSION,
        RL_ATTRS["framework"]: trace.framework,
        RL_ATTRS["step"]: trace.step,
        RL_ATTRS["group"]: trace.group_key,
        RL_ATTRS["rollout_index"]: trace.rollout_index,
        RL_ATTRS["reward"]: trace.reward,
        RL_ATTRS["advantage"]: trace.advantage,
        RL_ATTRS["status"]: trace.status,
        RL_ATTRS["invalid_reason"]: trace.invalid_reason,
    }
    if trace.reward_components:
        root_attrs[RL_ATTRS["reward_components"]] = json.dumps(
            trace.reward_components, ensure_ascii=False
        )
    if trace.prompt:
        root_attrs["gen_ai.input.messages"] = json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": trace.prompt}]}],
            ensure_ascii=False,
        )

    duration = 1_000_000  # 1ms default tick
    total = max(len(trace.steps), 1)
    root_end = cursor + duration * total * 2
    spans.append(
        {
            "traceId": trace.trace_id,
            "spanId": root_id,
            "parentSpanId": "",
            "name": f"invoke_agent {trace.agent_name}",
            "kind": "INTERNAL",
            "startTimeUnixNano": cursor,
            "endTimeUnixNano": root_end,
            "attributes": root_attrs,
            "status": {"code": "ERROR" if trace.status != "ok" else "OK"},
        }
    )

    for step in trace.steps:
        chat_id = f"{trace.trace_id}-s{step.index}"[:32]
        start = cursor + duration
        end = start + int((step.latency_ms or 1.0) * 1_000_000)
        cursor = end
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": step.model or trace.metadata.get("model") or "unknown",
            "gen_ai.response.finish_reasons": [step.finish_reason] if step.finish_reason else [],
            "gen_ai.usage.input_tokens": step.token_usage.input_tokens,
            "gen_ai.usage.output_tokens": step.token_usage.output_tokens,
        }
        if step.token_usage.cached_input_tokens:
            attrs["gen_ai.usage.cache_read.input_tokens"] = step.token_usage.cached_input_tokens
        # OTel has no convention for logprobs; keep the aggregate so the entropy
        # rule still works after a round-trip through a tracing backend.
        if step.mean_token_nll is not None:
            attrs["rolloutscope.logprob.mean_nll"] = round(float(step.mean_token_nll), 6)
        if step.text:
            attrs["gen_ai.output.messages"] = json.dumps(
                [{"role": "assistant", "parts": [{"type": "text", "content": step.text}]}],
                ensure_ascii=False,
            )
        spans.append(
            {
                "traceId": trace.trace_id,
                "spanId": chat_id,
                "parentSpanId": root_id,
                "name": f"chat {attrs['gen_ai.request.model']}",
                "kind": "CLIENT",
                "startTimeUnixNano": start,
                "endTimeUnixNano": end,
                "attributes": attrs,
                "status": {"code": "OK"},
            }
        )
        for j, call in enumerate(step.tool_calls):
            spans.append(
                {
                    "traceId": trace.trace_id,
                    "spanId": f"{trace.trace_id}-s{step.index}-t{j}"[:32],
                    "parentSpanId": chat_id,
                    "name": f"execute_tool {call.name}",
                    "kind": "INTERNAL",
                    "startTimeUnixNano": start,
                    "endTimeUnixNano": end,
                    "attributes": {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": call.name,
                        "gen_ai.tool.call.id": call.call_id,
                        "gen_ai.tool.call.arguments": call.arguments,
                        "gen_ai.tool.call.result": call.result,
                        "error.type": call.error,
                    },
                    "status": {"code": "ERROR" if call.error else "OK"},
                }
            )
    return spans


def to_otlp(batch: TraceBatch, service_name: str = "rolloutscope") -> dict[str, Any]:
    """Serialize a batch into an OTLP/JSON ``ExportTraceServiceRequest``."""
    spans: list[dict[str, Any]] = []
    for i, trace in enumerate(batch.traces):
        spans.extend(trace_to_spans(trace, base_time_ns=1_700_000_000_000_000_000 + i * 10**9))
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _encode_attrs(
                        {"service.name": service_name, "rolloutscope.framework": batch.framework}
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {"name": SCOPE_NAME, "version": SCHEMA_VERSION},
                        "spans": [
                            {
                                "traceId": s["traceId"],
                                "spanId": s["spanId"],
                                "parentSpanId": s["parentSpanId"],
                                "name": s["name"],
                                "kind": s["kind"],
                                "startTimeUnixNano": str(s["startTimeUnixNano"]),
                                "endTimeUnixNano": str(s["endTimeUnixNano"]),
                                "attributes": _encode_attrs(s["attributes"]),
                                "status": s["status"],
                            }
                            for s in spans
                        ],
                    }
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------
def _iter_spans(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict) and "resourceSpans" in payload:
        for rs in payload.get("resourceSpans") or []:
            for ss in rs.get("scopeSpans") or []:
                yield from ss.get("spans") or []
        return
    if isinstance(payload, dict) and "spans" in payload:
        yield from payload["spans"]
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "resourceSpans" in item:
                yield from _iter_spans(item)
            else:
                yield item


def from_otel(payload: Any, name: str = "otel-import") -> TraceBatch:
    """Rebuild a :class:`TraceBatch` from OTel spans (OTLP JSON or flat list)."""
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in _iter_spans(payload):
        tid = span.get("traceId") or span.get("trace_id")
        if not tid:
            continue
        by_trace.setdefault(str(tid), []).append(span)

    traces: list[AgentTrace] = []
    for tid, spans in by_trace.items():
        traces.append(_spans_to_trace(tid, spans))
    framework = "otel"
    for t in traces:
        if t.framework != "generic":
            framework = t.framework
            break
    return TraceBatch(traces=traces, name=name, framework=framework, source="otel")


def _spans_to_trace(tid: str, spans: list[dict[str, Any]]) -> AgentTrace:
    decoded: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for s in spans:
        attrs = _decode_attrs(s.get("attributes"))
        start = int(s.get("startTimeUnixNano") or 0)
        decoded.append((s, attrs, start))
    decoded.sort(key=lambda x: (x[2], str(x[0].get("spanId"))))

    root = next(
        (
            (s, a)
            for s, a, _ in decoded
            if a.get("gen_ai.operation.name") == "invoke_agent" or not s.get("parentSpanId")
        ),
        decoded[0][:2],
    )
    root_attrs = root[1]

    reward = root_attrs.get(RL_ATTRS["reward"])
    components_raw = root_attrs.get(RL_ATTRS["reward_components"])
    components: dict[str, float] = {}
    if isinstance(components_raw, str) and components_raw:
        try:
            components = {str(k): float(v) for k, v in json.loads(components_raw).items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            components = {}

    chats = [(s, a, t) for s, a, t in decoded if a.get("gen_ai.operation.name") == "chat"]
    chat_ids = {str(s.get("spanId")) for s, _, _ in chats}
    tool_children: dict[str, list[dict[str, Any]]] = {}
    for s, a, _ in decoded:
        if a.get("gen_ai.operation.name") == "execute_tool":
            parent = str(s.get("parentSpanId") or "")
            tool_children.setdefault(parent, []).append(a)

    steps: list[Step] = []
    for i, (s, a, _) in enumerate(chats or [(None, root_attrs, 0)]):
        tools = [
            ToolCall(
                name=str(t.get("gen_ai.tool.name") or ""),
                call_id=str(t.get("gen_ai.tool.call.id") or ""),
                arguments=t.get("gen_ai.tool.call.arguments") or "",
                result=t.get("gen_ai.tool.call.result"),
                error=t.get("error.type"),
            )
            for t in tool_children.get(str(s.get("spanId")), [])
        ] if s is not None else []
        if s is not None and not tools and a.get("gen_ai.tool.name"):
            tools = [
                ToolCall(
                    name=str(a.get("gen_ai.tool.name") or ""),
                    call_id=str(a.get("gen_ai.tool.call.id") or ""),
                    arguments=a.get("gen_ai.tool.call.arguments") or "",
                    result=a.get("gen_ai.tool.call.result"),
                )
            ]
        finish = a.get("gen_ai.response.finish_reasons")
        if isinstance(finish, list):
            finish = finish[0] if finish else None
        steps.append(
            Step(
                index=i,
                text=_messages_to_text(a.get("gen_ai.output.messages")),
                finish_reason=finish,
                model=a.get("gen_ai.request.model"),
                token_usage=TokenUsage(
                    input_tokens=int(a.get("gen_ai.usage.input_tokens") or 0),
                    output_tokens=int(a.get("gen_ai.usage.output_tokens") or 0),
                    cached_input_tokens=int(a.get("gen_ai.usage.cache_read.input_tokens") or 0),
                ),
                tool_calls=tools,
                mean_nll_hint=_maybe_float(a.get("rolloutscope.logprob.mean_nll")),
            )
        )
    if not chats:
        steps = []

    return AgentTrace(
        trace_id=tid,
        framework=str(root_attrs.get(RL_ATTRS["framework"]) or "otel"),
        step=_maybe_int(root_attrs.get(RL_ATTRS["step"])),
        group_id=(
            None if root_attrs.get(RL_ATTRS["group"]) is None else str(root_attrs.get(RL_ATTRS["group"]))
        ),
        rollout_index=_maybe_int(root_attrs.get(RL_ATTRS["rollout_index"])),
        prompt=_first_user_text(root_attrs.get("gen_ai.input.messages")),
        steps=steps,
        reward=_maybe_float(reward),
        advantage=_maybe_float(root_attrs.get(RL_ATTRS["advantage"])),
        reward_components=components,
        status=str(root_attrs.get(RL_ATTRS["status"]) or "ok"),
        invalid_reason=root_attrs.get(RL_ATTRS["invalid_reason"]),
        agent_name=str(root_attrs.get("gen_ai.agent.name") or "agent"),
        metadata={"chat_spans": len(chat_ids)},
    )


def _maybe_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


__all__ = ["RL_ATTRS", "STATUS_INVALID", "from_otel", "to_otlp", "trace_to_spans"]
