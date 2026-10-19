"""R4 — suspected reward hacking: high reward, suspiciously cheap rollout."""

from __future__ import annotations

import numpy as np

from ..base import AnalysisContext, Evidence, Finding, Rule, Severity, pct


class RewardHackingRule(Rule):
    """Flag rollouts that scored high while doing much less work than their peers.

    Definition, per training step ``s`` with reward quantile ``q`` and medians
    ``L_s`` (output tokens) / ``T_s`` (tool calls):

        flag_i = [r_i >= q90] and ( [len_i < 0.35 * L_s] or [T_s >= 2 and calls_i <= 0.2 * T_s] )

    plus a ``lift`` check: ``P(high reward | short) / P(high reward | not short)``.
    A high lift means "being short" and "being scored high" are associated, which
    is the statistical fingerprint of a shortcut reward, not of an efficient
    solution. The rule stays quiet on text-only tasks (no tool dimension) and on
    steps with too few samples.
    """

    id = "reward-hacking"
    title = "Suspected reward hacking"
    category = "reward"

    def run(self, ctx: AnalysisContext) -> Finding:
        if not ctx.has_rewards:
            return self.skipped("no reward column found", self.title, self.id)
        df = ctx.df.dropna(subset=["reward"])
        if len(df) < ctx.config.hack_min_samples:
            return self.skipped(
                f"only {len(df)} rewarded rollouts; need {ctx.config.hack_min_samples}",
                self.title,
                self.id,
            )

        flags = []
        lifts = []
        for step, chunk in df.groupby("step"):
            if len(chunk) < 10:
                continue
            q = float(chunk["reward"].quantile(ctx.config.hack_reward_quantile))
            high = chunk["reward"] >= q
            if high.all() or (~high).all():
                continue  # no discrimination possible in this step
            len_med = float(chunk["output_tokens"].median())
            tool_med = float(chunk["num_tool_calls"].median())
            short = chunk["output_tokens"] < ctx.config.hack_length_ratio * len_med if len_med else False
            starved = (
                chunk["num_tool_calls"] <= ctx.config.hack_tool_ratio * tool_med if tool_med >= 2 else False
            )
            step_flag = high & (short | starved)
            flags.append(step_flag)
            lift = _lift(high, short if not isinstance(short, bool) else None)
            if lift is not None:
                lifts.append(lift)

        if not flags:
            return self.skipped("no step had enough spread in rewards", self.title, self.id)

        flag_series = _concat(flags, df.index)
        flagged = int(flag_series.sum())
        rate = flagged / len(df) if len(df) else 0.0
        lift = float(np.median(lifts)) if lifts else None

        if rate >= 0.08 and (lift or 0) >= 2.0:
            severity = Severity.CRITICAL
        elif rate >= 0.03:
            severity = Severity.WARN
        elif rate > 0:
            severity = Severity.INFO
        else:
            severity = Severity.OK

        if severity is Severity.OK:
            headline = "No high-reward / low-effort rollouts found — reward looks aligned with real work"
        else:
            headline = (
                f"{pct(rate)} of rollouts scored in the top {int((1 - ctx.config.hack_reward_quantile) * 100)}% "
                f"of reward while doing <{ctx.config.hack_length_ratio:.0%} of the median work"
                + (f" (lift {lift:.1f}×)" if lift else "")
            )

        actions = []
        if severity is not Severity.OK:
            actions.append(
                "Open the evidence rollouts and check whether they answer without using the tool / without "
                "showing reasoning. If yes, the reward is gameable by a shorter policy."
            )
            actions.append(
                "Harden the reward: require the final answer to be grounded in a tool result, add a "
                "process/format check, or normalise reward by verified steps rather than by output alone."
            )
        if lift and lift >= 2.0:
            actions.append(
                f"Being short is {lift:.1f}× more likely to earn a top reward than being long — this "
                "association is the strongest single signal of a shortcut."
            )

        return Finding(
            rule_id=self.id,
            title=self.title,
            severity=severity,
            headline=headline,
            detail=(
                "Reward hacking in agent RL rarely looks like a wrong answer; it looks like a *cheap* answer "
                "that still passes the checker. Because the checker usually sees only the final text, a policy "
                "can learn to skip the tool call and guess, or to emit the shortest string that matches the "
                "answer regex. This rule compares each rollout against its own cohort, so it needs no "
                "ground-truth labels."
            ),
            metrics={
                "flagged": flagged,
                "flag_rate": round(rate, 4),
                "lift": None if lift is None else round(lift, 3),
                "reward_quantile": ctx.config.hack_reward_quantile,
                "length_ratio": ctx.config.hack_length_ratio,
                "tool_ratio": ctx.config.hack_tool_ratio,
            },
            evidence=_evidence(ctx, flag_series, df),
            actions=actions,
            penalty=ctx.config.weight_reward_hack * min(1.0, rate / ctx.config.hack_rate_cap),
            extra={"flagged_ids": df.loc[flag_series, "trace_id"].astype(str).head(300).tolist()},
        )


def _concat(flags, index) -> np.ndarray:
    out = np.zeros(len(index), dtype=bool)
    positions = {idx: i for i, idx in enumerate(index)}
    for series in flags:
        for idx, val in series.items():
            out[positions[idx]] = bool(val)
    return out


def _lift(high, short) -> float | None:
    if short is None or isinstance(short, bool):
        return None
    if int(short.sum()) == 0 or int((~short).sum()) == 0:
        return None
    p_high_given_short = float(high[short].mean())
    p_high_given_long = float(high[~short].mean())
    if p_high_given_long <= 0:
        return None
    return p_high_given_short / p_high_given_long


def _evidence(ctx: AnalysisContext, flag_series: np.ndarray, df) -> list[Evidence]:
    if not flag_series.any():
        return []
    subset = df.loc[flag_series]
    subset = subset.sort_values("reward", ascending=False).head(3)
    out: list[Evidence] = []
    for _, row in subset.iterrows():
        step_med = float(df.loc[df["step"] == row["step"], "output_tokens"].median())
        out.append(
            Evidence(
                trace_id=str(row["trace_id"]),
                label="high reward, unusually cheap rollout",
                detail=(
                    f"step {int(row['step'])}: reward={row['reward']}, "
                    f"output_tokens={int(row['output_tokens'])} (step median {step_med:.0f}), "
                    f"tool_calls={int(row['num_tool_calls'])}, turns={int(row['num_turns'])}"
                ),
            )
        )
    return out
