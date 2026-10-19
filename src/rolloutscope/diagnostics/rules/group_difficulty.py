"""R6 — difficulty structure and the Effective Rollout Ratio (ERR).

This is the headline metric of the report. Instead of one scalar ("reward went
up"), it decomposes every group into three buckets and answers the question the
user actually has: *how much of this step was worth computing?*
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisContext, Finding, Rule, Severity, pct


class GroupDifficultyRule(Rule):
    """Bucket groups into saturated / informative / collapsed and track ERR."""

    id = "group-difficulty"
    title = "Difficulty structure (ERR)"
    category = "gradient"

    def run(self, ctx: AnalysisContext) -> Finding:
        if not ctx.has_rewards:
            return self.skipped("no reward column found", self.title, self.id)
        groups = ctx.groups
        if groups.empty:
            return self.skipped("no groups", self.title, self.id)

        sat = float(groups["saturated"].mean())
        col = float(groups["collapsed"].mean())
        info = float(groups["informative"].mean())
        err = info

        density = _gradient_density(ctx)

        per_step = (
            groups.groupby("step")
            .agg(
                saturated=("saturated", "mean"),
                collapsed=("collapsed", "mean"),
                informative=("informative", "mean"),
            )
            .reset_index()
        )
        drift = None
        if len(per_step) >= 4:
            head = per_step.head(2)[["saturated", "collapsed"]].mean()
            tail = per_step.tail(2)[["saturated", "collapsed"]].mean()
            drift = {
                "saturated_delta": round(float(tail["saturated"] - head["saturated"]), 4),
                "collapsed_delta": round(float(tail["collapsed"] - head["collapsed"]), 4),
            }
        recent_err = float(per_step.tail(3)["informative"].mean()) if len(per_step) else err

        # Severity follows the *recent* structure: a run that started healthy and
        # ended with two thirds of its groups dead is in trouble, even if the
        # whole-run average still looks acceptable.
        judged = min(err, recent_err)
        if judged < 0.30:
            severity = Severity.CRITICAL
        elif judged < 0.50:
            severity = Severity.WARN
        elif judged < 0.70:
            severity = Severity.INFO
        else:
            severity = Severity.OK

        if severity is Severity.OK:
            headline = f"ERR {err:.0%} — most groups still produce gradient"
        elif abs(recent_err - err) > 0.08:
            headline = (
                f"ERR {recent_err:.0%} in the last 3 steps ({err:.0%} over the whole run) — "
                f"{pct(sat)} of groups are saturated, {pct(col)} collapsed"
            )
        else:
            headline = (
                f"ERR {err:.0%} — only {int(info * len(groups))}/{len(groups)} groups can move the policy "
                f"({pct(sat)} saturated, {pct(col)} collapsed)"
            )

        actions = []
        if drift and drift["saturated_delta"] > 0.05:
            actions.append(
                f"Saturated groups grew by {drift['saturated_delta']:+.0%} between the first and last steps: "
                "the policy is getting better at the easy prompts while ERR drops. That is the classic "
                "'reward up, capability flat' curve — add harder prompts or re-weight by pass rate."
            )
        if drift and drift["collapsed_delta"] > 0.05:
            actions.append(
                f"Collapsed groups grew by {drift['collapsed_delta']:+.0%}: prompts are getting too hard for the "
                "current policy (or the agent loop broke). Consider curriculum ordering or hint injection."
            )
        if severity is not Severity.OK:
            actions.append(
                "Report ERR next to reward in your experiment tracker. Reward can rise while ERR falls, and "
                "ERR is the one that predicts whether the next 100 steps will actually improve the model."
            )

        return Finding(
            rule_id=self.id,
            title=self.title,
            severity=severity,
            headline=headline,
            detail=(
                "ERR = informative groups / all groups, where a group is informative when 0 < pass_rate < 1. "
                "Saturated groups are already solved (no gradient, pure cost); collapsed groups are never "
                "solved (zero advantage under standard GRPO). Gradient density is sum|A_i| per 1k tokens: "
                "how much learning signal each unit of compute buys. This rule intentionally carries no "
                "score penalty — its signal is already captured by the zero-advantage rule; it is here to "
                "explain *why* the ratio is what it is."
            ),
            metrics={
                "err": round(err, 4),
                "err_recent": round(recent_err, 4),
                "saturated_ratio": round(sat, 4),
                "collapsed_ratio": round(col, 4),
                "informative_groups": int(round(info * len(groups))),
                "groups": len(groups),
                "gradient_density": None if density is None else round(density, 4),
                "drift": drift,
                "per_step": per_step.round(4).to_dict(orient="records"),
            },
            actions=actions,
            penalty=0.0,
        )


def _gradient_density(ctx: AnalysisContext) -> float | None:
    """sum|A| per 1k tokens, computing GRPO advantages when they are absent."""
    df = ctx.df
    if df["reward"].notna().sum() == 0:
        return None
    work = df.dropna(subset=["reward"]).copy()
    if work["advantage"].notna().any():
        work["a"] = work["advantage"].abs()
    else:
        g = work.groupby(["step", "group_key"])["reward"]
        mean = g.transform("mean")
        std = g.transform("std").fillna(0.0)
        work["a"] = ((work["reward"] - mean) / (std + 1e-4)).abs()
        work.loc[std < 1e-6, "a"] = 0.0
    tokens = float(work["total_tokens"].sum())
    if tokens <= 0:
        return None
    return float(work["a"].sum()) / (tokens / 1000.0)


__all__ = ["GroupDifficultyRule", "np", "pd"]
