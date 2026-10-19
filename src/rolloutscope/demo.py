"""Synthetic demo run.

A 16-step GRPO run on a tool-using math task, with a story baked in:

* steps 0–5   learning: lengths grow, ERR ~0.75, entropy stable
* steps 6–11  saturation: the policy farms the easy prompts, ERR falls,
              lengths and entropy start to shrink
* steps 12–15 reward hacking: short answers that skip the tool still score 1.0,
              length and entropy collapse, reward keeps *rising*

The reward curve looks healthy the whole time. That is the point: the numbers
that get plotted by default are the ones that lie.

Everything is generated from a fixed seed, so the demo report is reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import (
    INVALID_TOOL_FORMAT,
    INVALID_TOOL_LOOP,
    INVALID_TRUNCATED,
    STATUS_INVALID,
    STATUS_OK,
    AgentTrace,
    Step,
    TokenUsage,
    ToolCall,
    TraceBatch,
)

REASONING = [
    "First I need to parse the quantities in the problem.",
    "Let me set up the equation for the total cost.",
    "The discount applies before tax, so I should be careful about ordering.",
    "I will verify the arithmetic with the calculator.",
    "That matches my estimate, so the intermediate value looks right.",
    "One more check: the units are consistent across terms.",
    "The remaining step is to substitute back into the original expression.",
]

PROMPTS = [
    "A shop sells notebooks at 12 yuan each. If I buy 7 and get a 15% discount, how much do I pay?",
    "A train travels 240 km in 3 hours, then 180 km in 2 hours. What is the average speed?",
    "A tank fills at 4 L/min and drains at 1.5 L/min. How long to fill 250 L?",
    "If 3x + 7 = 28, what is 2x - 1?",
    "A rectangle has perimeter 46 and width 8. What is its area?",
    "A worker earns 85 yuan/hour for 6 hours and 120 yuan/hour for overtime. Total for 8 hours?",
    "Two numbers sum to 47 and differ by 9. What is their product?",
    "A recipe for 4 people needs 300 g rice. How much for 10 people?",
    "A phone costs 2400 yuan with 20% off, then 5% tax. Final price?",
    "A pool is 2/5 full. After adding 180 L it is 3/4 full. Capacity?",
    "A car uses 6.5 L per 100 km. How much fuel for a 420 km trip?",
    "A class has 32 students, 5/8 are girls. How many boys?",
    "A sum doubles in 8 years at simple interest. What is the annual rate?",
    "A box holds 24 cans. How many boxes for 300 cans, and how many spare?",
    "A jacket is discounted 30% to 560 yuan. What was the original price?",
    "A pipe fills a tank in 6 h, another in 4 h. Together, how long?",
    "A sequence starts 3, 7, 11, ... What is the 20th term?",
    "A cyclist rides 15 km at 20 km/h, then 10 km at 12 km/h. Total time?",
    "A savings account earns 3.5% yearly on 8000 yuan. Balance after 2 years?",
    "A photograph is 12 cm by 18 cm. Scaled to width 30 cm, what is the height?",
]


@dataclass
class DemoConfig:
    steps: int = 16
    prompts: int = 20
    rollouts: int = 8
    seed: int = 7
    max_tokens: int = 640
    hack_start: int = 12
    saturation_start: int = 6


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _curves(step: int, cfg: DemoConfig) -> dict[str, float]:
    """The story, as numbers: sat/collapsed up, informative pass rate up, length and NLL down."""
    s = cfg.steps - 1
    if step < cfg.saturation_start:
        t = step / max(1, cfg.saturation_start - 1)
        sat, col = _lerp(0.10, 0.15, t), _lerp(0.05, 0.07, t)
        length, nll, p_info = _lerp(330, 470, t), _lerp(2.95, 3.05, t), _lerp(0.52, 0.58, t)
    elif step < cfg.hack_start:
        t = (step - cfg.saturation_start) / max(1, cfg.hack_start - cfg.saturation_start - 1)
        sat, col = _lerp(0.15, 0.33, t), _lerp(0.07, 0.15, t)
        length, nll, p_info = _lerp(470, 260, t), _lerp(3.05, 2.25, t), _lerp(0.58, 0.72, t)
    else:
        t = (step - cfg.hack_start) / max(1, s - cfg.hack_start)
        sat, col = _lerp(0.33, 0.45, t), _lerp(0.15, 0.26, t)
        length, nll, p_info = _lerp(260, 140, t), _lerp(2.25, 1.35, t), _lerp(0.72, 0.80, t)
    return {
        "saturated": sat,
        "collapsed": col,
        "length": length,
        "nll": nll,
        "p_informative": p_info,
    }


def generate(cfg: DemoConfig | None = None) -> TraceBatch:
    cfg = cfg or DemoConfig()
    rng = np.random.default_rng(cfg.seed)
    traces: list[AgentTrace] = []

    for step in range(cfg.steps):
        c = _curves(step, cfg)
        span = max(1, cfg.steps - cfg.hack_start)
        hack_rate = 0.0 if step < cfg.hack_start else _lerp(0.10, 0.55, (step - cfg.hack_start) / span)

        # Exact counts per step (then shuffled) so the curves are smooth and the
        # prompt ordering stays random.
        n_sat = int(round(c["saturated"] * cfg.prompts))
        n_col = int(round(c["collapsed"] * cfg.prompts))
        kinds = ["saturated"] * n_sat + ["collapsed"] * n_col
        kinds += ["informative"] * max(0, cfg.prompts - len(kinds))
        rng.shuffle(kinds)

        for p in range(cfg.prompts):
            prompt = PROMPTS[p % len(PROMPTS)]
            group_id = f"step{step}-p{p:03d}"
            kind = kinds[p]

            for n in range(cfg.rollouts):
                reward = _reward(kind, rng, c["p_informative"])
                length = int(min(cfg.max_tokens, max(24, rng.lognormal(np.log(c["length"]), 0.22))))
                hacked = bool(reward >= 0.99 and rng.random() < hack_rate)
                if hacked:
                    length = int(max(18, length * rng.uniform(0.14, 0.24)))

                n_tools = 0 if hacked else int(rng.integers(1, 4))
                invalid = _invalid_kind(rng, hacked)
                if invalid in (INVALID_TRUNCATED, INVALID_TOOL_FORMAT):
                    n_tools = max(n_tools, 1)
                if invalid == INVALID_TOOL_LOOP:
                    n_tools = 4

                answer = str(int(abs(rng.normal(120, 60))))
                turns, _n_turns = _build_trajectory(
                    length, answer, hacked, n_tools, invalid, rng, prompt
                )
                nll = float(np.clip(rng.normal(c["nll"], 0.28), 0.05, 8.0))
                logprobs = [float(-np.clip(rng.normal(nll, 0.45), 0.02, 12.0)) for _ in range(min(length, 48))]

                steps = _build_steps(turns, nll, logprobs, length, invalid, rng)
                status = STATUS_INVALID if invalid else STATUS_OK
                traces.append(
                    AgentTrace(
                        trace_id=f"s{step:02d}-p{p:03d}-n{n}",
                        framework="verl",
                        run_id="grpo-math-tool-2026-10",
                        step=step,
                        group_id=group_id,
                        rollout_index=n,
                        prompt_id=group_id,
                        prompt=prompt,
                        steps=steps,
                        reward=round(reward, 4),
                        status=status,
                        invalid_reason=invalid,
                        agent_name="math_agent",
                        metadata={
                            "model": "qwen2.5-7b-instruct",
                            "hacked": hacked,
                            "logprobs_sampled": True,
                            "num_rollouts": cfg.rollouts,
                        },
                    )
                )

    return TraceBatch(
        traces=traces,
        name="grpo-math-tool-2026-10",
        framework="verl",
        metadata={
            "model": "qwen2.5-7b-instruct",
            "dataset": "math-tool-synth",
            "num_rollouts": cfg.rollouts,
            "synthetic": True,
        },
    )


def _reward(kind: str, rng: np.random.Generator, p_informative: float = 0.62) -> float:
    if kind == "saturated":
        return 1.0
    if kind == "collapsed":
        return 0.0
    return 1.0 if rng.random() < p_informative else 0.0


def _invalid_kind(rng: np.random.Generator, hacked: bool) -> str | None:
    roll = rng.random()
    if hacked:
        return None if roll > 0.02 else INVALID_TOOL_FORMAT
    if roll < 0.045:
        return INVALID_TOOL_FORMAT
    if roll < 0.085:
        return INVALID_TRUNCATED
    if roll < 0.098:
        return INVALID_TOOL_LOOP
    return None


def _build_trajectory(length: int, answer: str, hacked: bool, n_tools: int, invalid, rng, prompt):
    """Return (turns, n) where turns is a list of (text, tool_calls, finish)."""
    if hacked:
        text = f"The answer is {answer}."
        return [(text, [], "stop")], 1

    turns = []
    if n_tools:
        budget = max(1, length // max(1, n_tools + 1))
        for i in range(n_tools):
            reasoning = _filler(budget, rng)
            expr = f"{int(rng.integers(10, 900))} * {int(rng.integers(2, 19))} + {int(rng.integers(1, 90))}"
            if invalid == INVALID_TOOL_FORMAT and i == 0:
                args = '{"expression": ' + expr.replace(" ", "")  # deliberately broken JSON
            else:
                args = json.dumps({"expression": expr})
            if invalid == INVALID_TOOL_LOOP:
                args = json.dumps({"expression": "42 * 2"})
            turns.append(
                (
                    reasoning,
                    [ToolCall(name="calculator", call_id=f"call_{i}", arguments=args, result=str(int(rng.integers(100, 9000))))],
                    "tool_calls",
                )
            )
    final = f"Using the computed value, the answer is {answer}."
    if invalid == INVALID_TRUNCATED:
        final = "Using the computed value, the answer is"
        turns.append((final, [], "length"))
    else:
        turns.append((final, [], "stop"))
    return turns, len(turns)


def _filler(target_tokens: int, rng: np.random.Generator) -> str:
    chars = max(40, int(target_tokens * 3.6))
    order = rng.permutation(len(REASONING))
    out: list[str] = []
    i = 0
    while sum(len(x) + 1 for x in out) < chars:
        out.append(REASONING[int(order[i % len(order)])])
        i += 1
        if i % len(order) == 0:
            order = rng.permutation(len(REASONING))
    return " ".join(out)


def _build_steps(turns, nll: float, logprobs, total_tokens: int, invalid, rng) -> list[Step]:
    steps: list[Step] = []
    remaining = total_tokens
    for i, (text, tools, finish) in enumerate(turns):
        est = max(8, int(len(text) / 3.6))
        if i == len(turns) - 1:
            est = max(8, remaining)
        est = min(est, max(8, remaining))
        remaining = max(0, remaining - est)
        steps.append(
            Step(
                index=i,
                text=text,
                finish_reason=finish,
                model="qwen2.5-7b-instruct",
                token_usage=TokenUsage(input_tokens=int(rng.integers(180, 320)), output_tokens=est),
                logprobs=logprobs[: min(len(logprobs), est)] if i == 0 else None,
                tool_calls=tools,
                latency_ms=float(rng.integers(400, 2400)),
            )
        )
    return steps


# ---------------------------------------------------------------------------
# writers: native jsonl + verl-shaped parquet + TRL artifacts + OTel spans
# ---------------------------------------------------------------------------
def write_demo(out_dir: str = "demo/out", cfg: DemoConfig | None = None) -> dict[str, str]:
    from . import report as report_mod
    from .adapters import verl as verl_adapter
    from .diagnostics import analyze
    from .otel import to_otlp

    cfg = cfg or DemoConfig()
    os.makedirs(out_dir, exist_ok=True)
    batch = generate(cfg)
    paths: dict[str, str] = {}

    native = os.path.join(out_dir, "traces.jsonl")
    batch.to_jsonl(native)
    paths["traces"] = native

    paths["verl_dump"] = _write_verl_dump(batch, os.path.join(out_dir, "verl_dump"), cfg.max_tokens)
    paths["trl"] = _write_trl(batch, os.path.join(out_dir, "trl_run"))

    spans_path = os.path.join(out_dir, "spans.json")
    # Export one complete GRPO group per step: a partial group would look like a
    # zero-variance group and produce a misleading diagnostic.
    per_step: dict[int | None, dict[str, list[AgentTrace]]] = {}
    for t in batch.traces:
        per_step.setdefault(t.step, {}).setdefault(t.group_key, []).append(t)
    picked: list[AgentTrace] = []
    for _step, groups in sorted(per_step.items(), key=lambda kv: (kv[0] is None, kv[0])):
        first_key = min(groups)
        picked.extend(groups[first_key])
    subset = TraceBatch(traces=picked, name=batch.name, framework=batch.framework)
    with open(spans_path, "w", encoding="utf-8") as fh:
        json.dump(to_otlp(subset), fh)
    paths["spans"] = spans_path

    result = analyze(batch)
    html_path = report_mod.write_html(result, os.path.join(out_dir, "report.html"))
    paths["report"] = html_path
    json_path = report_mod.write_json(result, os.path.join(out_dir, "report.json"))
    paths["json"] = json_path

    # Prove the verl adapter round-trips: re-load the parquet and re-analyze.
    try:
        reloaded = verl_adapter.load(paths["verl_dump"])
        paths["verl_traces"] = str(len(reloaded))
    except Exception as exc:  # pragma: no cover - optional dependency
        paths["verl_traces"] = f"skipped ({type(exc).__name__})"

    return paths


def _write_verl_dump(batch: TraceBatch, out_dir: str, max_len: int = 640) -> str:
    """Write a verl-shaped parquet dump (token ids only, exactly like verl)."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return out_dir

    rows: list[dict[str, Any]] = []
    for i, t in enumerate(batch.traces):
        # A truncated response occupies the full budget, exactly like verl pads it.
        n = max_len if t.invalid_reason == INVALID_TRUNCATED else min(t.output_tokens, max_len)
        responses = [(i * 7 + j) % 15000 for j in range(max_len)]
        mask = [1] * n + [0] * (max_len - n)
        logprobs = [0.0] * max_len
        for j in range(n):
            lp = t.steps[0].logprobs[j] if t.steps and t.steps[0].logprobs and j < len(t.steps[0].logprobs) else -1.0
            logprobs[j] = lp
        scores = [0.0] * max_len
        if n:
            scores[n - 1] = t.reward if t.reward is not None else 0.0
        rows.append(
            {
                "uid": t.group_id,
                "index": i,
                "step": t.step,
                "responses": responses,
                "response_mask": mask,
                "rollout_log_probs": logprobs,
                "rm_scores": scores,
                "raw_prompt": [{"role": "user", "content": t.prompt or ""}],
            }
        )
    path = os.path.join(out_dir, "rollout_dump.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return out_dir


def _write_trl(batch: TraceBatch, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    history = []
    for step in sorted({t.step for t in batch.traces if t.step is not None}):
        subset = [t for t in batch.traces if t.step == step]
        rewards = np.array([t.reward for t in subset if t.reward is not None], dtype=float)
        groups: dict[str, list[float]] = {}
        for t in subset:
            groups.setdefault(t.group_key, []).append(t.reward or 0.0)
        zero_std = np.mean([np.std(v) < 1e-6 for v in groups.values()]) if groups else 0.0
        nlls = [s.mean_token_nll for t in subset for s in t.steps if s.mean_token_nll is not None]
        history.append(
            {
                "step": step,
                "loss": round(float(0.9 - 0.02 * step + 0.01 * np.sin(step)), 4),
                "reward": round(float(rewards.mean()), 4) if rewards.size else None,
                "reward_std": round(float(rewards.std()), 4) if rewards.size else None,
                "frac_reward_zero_std": round(float(zero_std), 4),
                "entropy": round(float(np.mean(nlls)), 4) if nlls else None,
                "completions/mean_length": round(float(np.mean([t.output_tokens for t in subset])), 1),
                "kl": round(float(0.02 + 0.004 * step), 4),
            }
        )
    state_path = os.path.join(out_dir, "trainer_state.json")
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump({"log_history": history, "global_step": len(history), "epoch": 1.0}, fh, indent=2)

    comp_path = os.path.join(out_dir, "completions.jsonl")
    with open(comp_path, "w", encoding="utf-8") as fh:
        for t in batch.traces:
            fh.write(
                json.dumps(
                    {
                        "step": t.step,
                        "prompt": t.prompt,
                        "completion": t.text,
                        "reward": t.reward,
                        "completion_tokens": t.output_tokens,
                        "finish_reason": t.steps[-1].finish_reason if t.steps else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_dir


__all__ = ["DemoConfig", "generate", "write_demo"]
