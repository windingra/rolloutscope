"""Shared fixtures: tiny hand-built rollout batches with known pathologies."""

from __future__ import annotations

import pytest

from rolloutscope.schema import (
    INVALID_TOOL_FORMAT,
    INVALID_TRUNCATED,
    STATUS_INVALID,
    STATUS_OK,
    AgentTrace,
    Step,
    TokenUsage,
    ToolCall,
    TraceBatch,
)


def make_trace(
    step: int,
    group: str,
    index: int,
    reward: float,
    tokens: int = 100,
    tool_calls: int = 2,
    nll: float = 3.0,
    text: str = "reasoning ... the answer is 42",
    invalid: str | None = None,
) -> AgentTrace:
    calls = []
    for i in range(tool_calls):
        args = '{"expression": "1+1"}' if invalid != INVALID_TOOL_FORMAT else '{"expression": '
        calls.append(ToolCall(name="calculator", call_id=f"c{i}", arguments=args, result="2"))
    finish = "length" if invalid == INVALID_TRUNCATED else ("tool_calls" if calls else "stop")
    return AgentTrace(
        trace_id=f"s{step}-{group}-n{index}",
        framework="verl",
        step=step,
        group_id=group,
        rollout_index=index,
        prompt=f"prompt {group}",
        steps=[
            Step(
                index=0,
                text=text,
                finish_reason=finish,
                token_usage=TokenUsage(input_tokens=50, output_tokens=tokens),
                logprobs=[-nll] * min(tokens, 16),
                tool_calls=calls,
            )
        ],
        reward=reward,
        status=STATUS_INVALID if invalid else STATUS_OK,
        invalid_reason=invalid,
    )


def group_trace(
    step: int, group: str, rewards: list[float], tokens: int = 100, token_step: int = 0, **kwargs
) -> list[AgentTrace]:
    return [
        make_trace(step, group, i, r, tokens=tokens + i * token_step, **kwargs)
        for i, r in enumerate(rewards)
    ]


@pytest.fixture
def healthy_batch() -> TraceBatch:
    """Every group is informative: half the rollouts solve the prompt."""
    traces = []
    for step in range(4):
        for g in range(6):
            traces += group_trace(step, f"g{g}", [1.0, 1.0, 0.0, 0.0], tokens=300, token_step=20, nll=3.0)
    return TraceBatch(traces=traces, name="healthy", framework="verl")


@pytest.fixture
def broken_batch() -> TraceBatch:
    """Per step: 2 saturated, 2 collapsed, 4 informative groups + 2 invalid rollouts."""
    traces = []
    for step in range(4):
        nll = 3.0 - 0.5 * step
        # 2 saturated groups (all correct) and 2 collapsed (all wrong)
        traces += group_trace(step, "sat-a", [1.0] * 4, tokens=300, token_step=25, nll=nll)
        traces += group_trace(step, "sat-b", [1.0] * 4, tokens=300, token_step=25, nll=nll)
        traces += group_trace(step, "col-a", [0.0] * 4, tokens=300, token_step=25, nll=nll)
        traces += group_trace(step, "col-b", [0.0] * 4, tokens=300, token_step=25, nll=nll)
        # 4 informative groups; the first one hides a short high-reward rollout
        mixed = group_trace(step, "mix-a", [1.0, 0.0, 1.0, 0.0], tokens=300, token_step=25, nll=nll)
        mixed[0] = make_trace(step, "mix-a", 0, 1.0, tokens=40, tool_calls=0, nll=nll)
        traces += mixed
        traces += group_trace(step, "mix-b", [1.0, 1.0, 1.0, 0.0], tokens=300, token_step=25, nll=nll)
        traces += group_trace(step, "mix-c", [1.0, 0.0, 1.0, 0.0], tokens=300, token_step=25, nll=nll)
        traces += group_trace(step, "mix-d", [0.0, 1.0, 0.0, 1.0], tokens=300, token_step=25, nll=nll)
        # invalid rollouts live inside a real group, like they do in a real run
        traces.append(make_trace(step, "mix-a", 4, 0.0, invalid=INVALID_TOOL_FORMAT))
        traces.append(make_trace(step, "mix-b", 4, 0.0, invalid=INVALID_TRUNCATED))
    return TraceBatch(traces=traces, name="broken", framework="verl")
