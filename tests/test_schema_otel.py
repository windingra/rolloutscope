"""Schema and OTel round-trip tests."""

from __future__ import annotations

import json

from rolloutscope.otel import from_otel, to_otlp, trace_to_spans
from rolloutscope.schema import AgentTrace, Step, TokenUsage, ToolCall, TraceBatch


def _trace() -> AgentTrace:
    return AgentTrace(
        trace_id="t1",
        framework="verl",
        step=3,
        group_id="g1",
        rollout_index=2,
        prompt="what is 2+2?",
        steps=[
            Step(
                index=0,
                text="I will use the calculator.",
                finish_reason="tool_calls",
                model="qwen",
                token_usage=TokenUsage(input_tokens=10, output_tokens=20, cached_input_tokens=4),
                logprobs=[-0.5] * 20,
                tool_calls=[ToolCall(name="calc", call_id="c1", arguments='{"a": 2}', result="4")],
            ),
            Step(index=1, text="The answer is 4.", finish_reason="stop", token_usage=TokenUsage(output_tokens=5)),
        ],
        reward=1.0,
        advantage=0.8,
        reward_components={"accuracy": 1.0},
    )


def test_dict_round_trip_keeps_everything():
    t = _trace()
    back = AgentTrace.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back.trace_id == t.trace_id
    assert back.step == 3 and back.group_id == "g1" and back.rollout_index == 2
    assert back.reward == 1.0 and back.advantage == 0.8
    assert back.reward_components == {"accuracy": 1.0}
    assert len(back.steps) == 2
    assert back.steps[0].tool_calls[0].args_dict == {"a": 2}
    assert back.output_tokens == 25
    assert back.num_tool_calls == 1


def test_derived_fields():
    t = _trace()
    assert t.total_tokens == 10 + 20 + 5
    assert t.mean_token_nll is not None and abs(t.mean_token_nll - 0.5) < 1e-9
    assert not t.is_truncated


def test_tool_call_format_error_is_detected():
    assert ToolCall(name="x", arguments='{"a": ').is_format_error
    assert not ToolCall(name="x", arguments='{"a": 1}').is_format_error
    assert not ToolCall(name="x", arguments="").is_format_error


def test_otel_round_trip():
    batch = TraceBatch(traces=[_trace()], name="rt", framework="verl")
    payload = to_otlp(batch)
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

    back = from_otel(json.loads(json.dumps(payload)))
    assert len(back) == 1
    t = back[0]
    assert t.trace_id == "t1"
    assert t.reward == 1.0
    assert t.advantage == 0.8
    assert t.step == 3
    assert t.group_id == "g1"
    assert t.prompt == "what is 2+2?"
    assert len(t.steps) == 2
    assert t.steps[0].tool_calls[0].name == "calc"
    assert t.steps[0].tool_calls[0].result == "4"
    assert t.steps[0].text == "I will use the calculator."
    assert t.steps[1].finish_reason == "stop"
    assert t.output_tokens == 25
    # logprobs cannot cross an OTel boundary, but the aggregate must survive
    assert t.mean_token_nll is not None and abs(t.mean_token_nll - 0.5) < 1e-6


def test_otel_span_names_follow_semconv():
    spans = trace_to_spans(_trace())
    names = [s["name"] for s in spans]
    assert names[0].startswith("invoke_agent ")
    assert "chat qwen" in names
    assert "execute_tool calc" in names
    root = spans[0]["attributes"]
    assert root["gen_ai.operation.name"] == "invoke_agent"
    chat = next(s for s in spans if s["name"] == "chat qwen")
    assert chat["attributes"]["gen_ai.usage.output_tokens"] == 20
    assert chat["attributes"]["gen_ai.response.finish_reasons"] == ["tool_calls"]
