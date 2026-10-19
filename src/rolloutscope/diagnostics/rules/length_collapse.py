"""R3 — output-length collapse: the model learns to stop talking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisContext, Evidence, Finding, Rule, Severity, pct


class LengthCollapseRule(Rule):
    """Flag a sustained drop in output length, or a degenerate constant length.

    Length is the cheapest observable proxy for 'the model gave up': when a
    reasoning policy collapses it typically first shortens its chains, then
    starts emitting the same short template. Because this is also the signature
    of a *successful* length penalty, the rule only fires when length drops
    faster than reward improves.
    """

    id = "length-collapse"
    title = "Output length collapse"
    category = "exploration"

    def run(self, ctx: AnalysisContext) -> Finding:
        steps = ctx.steps
        if steps.empty or len(steps) < 3:
            logged = ctx.batch.metrics.get("completions/mean_length") or {}
            if logged:
                return _from_logged_metric(ctx, logged)
            return self.skipped("need at least 3 training steps", self.title, self.id)

        series = steps.set_index("step")["median_length"].astype(float)
        base_n = min(ctx.config.baseline_steps, max(1, len(series) // 3))
        baseline = float(np.median(series.iloc[:base_n].to_numpy()))
        if baseline <= 0:
            return self.skipped("baseline output length is zero", self.title, self.id)

        ratio = series / baseline
        breach = ratio < ctx.config.length_drop_ratio
        first_breach = int(ratio[breach].index.min()) if breach.any() else None
        sustained = _max_consecutive(breach) >= 2
        min_ratio = float(ratio.min())
        depth = max(0.0, 1.0 - min_ratio) if breach.any() else 0.0

        cvs = steps["length_cv"].astype(float).dropna()
        degenerate = bool((cvs < ctx.config.length_cv_min).any()) if len(cvs) else False
        degenerate_step = int(steps.loc[cvs.idxmin(), "step"]) if degenerate else None

        # A deliberate length penalty looks like: length down, reward up.
        reward_delta = None
        if "mean_reward" in steps and steps["mean_reward"].notna().any():
            r = steps["mean_reward"].astype(float).dropna()
            if len(r) >= 3:
                reward_delta = float(r.iloc[-1] - r.iloc[0])
        # A deliberate length penalty looks like: a moderate trim plus a real
        # reward gain. A 60%+ collapse is not a penalty, it is the policy
        # finding a cheap exit — even if reward happens to drift up.
        reward_improved = reward_delta is not None and reward_delta > 0.02
        intended = reward_improved and min_ratio >= 0.5

        if sustained and min_ratio < 0.4:
            severity = Severity.CRITICAL
        elif sustained and not intended:
            severity = Severity.WARN
        elif sustained and intended:
            severity = Severity.INFO
        elif degenerate:
            severity = Severity.WARN
        elif breach.any():
            severity = Severity.INFO
        else:
            severity = Severity.OK

        if severity is Severity.OK:
            headline = (
                f"Output length held at {min_ratio:.0%}–{float(ratio.max()):.0%} of baseline "
                f"({baseline:.0f} tokens) — no collapse"
            )
        elif degenerate and not sustained:
            headline = (
                f"Output length is nearly constant at step {degenerate_step} "
                f"(CV={float(cvs.min()):.3f}) — rollouts are degenerating into one template"
            )
        else:
            headline = (
                f"Median output length fell to {min_ratio:.0%} of the early baseline "
                f"({baseline:.0f} → {series.min():.0f} tokens)"
                + (f", first at step {first_breach}" if first_breach is not None else "")
            )
            if intended:
                headline += " (reward also improved — may be an intended length penalty)"
            elif reward_improved:
                headline += " (reward also rose, but not because harder prompts were solved)"
        actions = []
        if severity in (Severity.CRITICAL, Severity.WARN):
            if reward_improved:
                actions.append(
                    "Reward rose while length fell: check whether a length-based reward term is dominating "
                    "the task reward, and whether accuracy on held-out tasks is still improving."
                )
            else:
                actions.append(
                    "Length fell without a reward gain — inspect the shortest rollouts (evidence below). "
                    "Common causes: premature stop tokens, an aggressive length penalty, or truncation."
                )
            actions.append(
                "Compare against the entropy rule: length + entropy collapsing together is the classic "
                "mode-collapse pair and usually needs a temperature/KL change, not a prompt change."
            )

        return Finding(
            rule_id=self.id,
            title=self.title,
            severity=severity,
            headline=headline,
            detail=(
                "A reasoning/agent policy that is still learning normally grows or holds its output length. "
                "A sustained drop means the policy found a cheap exit: emit a short answer and stop. "
                "This rule deliberately separates 'intended length penalty' (length down, reward up) from "
                "'collapse' (length down, reward flat)."
            ),
            metrics={
                "baseline_median_length": round(baseline, 1),
                "min_ratio": round(min_ratio, 4),
                "first_breach_step": first_breach,
                "sustained": bool(sustained),
                "degenerate_length": degenerate,
                "degenerate_step": degenerate_step,
                "reward_delta": None if reward_delta is None else round(reward_delta, 4),
                "per_step": {int(k): round(float(v), 1) for k, v in series.items()},
            },
            evidence=_evidence(ctx, series, first_breach),
            actions=actions,
            penalty=ctx.config.weight_length * depth * (1.0 if sustained else 0.5)
            + (5.0 if degenerate else 0.0),
        )


def _max_consecutive(mask) -> int:
    best = run = 0
    for flag in list(mask):
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _evidence(ctx: AnalysisContext, series, first_breach: int | None) -> list[Evidence]:
    step = first_breach if first_breach is not None else int(series.idxmin())
    subset = ctx.df[ctx.df["step"] == step]
    if subset.empty:
        return []
    row = subset.sort_values("output_tokens").iloc[0]
    out = [
        Evidence(
            trace_id=str(row["trace_id"]),
            label="shortest rollout at the collapse step",
            detail=(
                f"step {step}: output_tokens={int(row['output_tokens'])}, "
                f"reward={row['reward']}, turns={int(row['num_turns'])}, "
                f"tools={int(row['num_tool_calls'])}"
            ),
        )
    ]
    longest = subset.sort_values("output_tokens", ascending=False).iloc[0]
    if str(longest["trace_id"]) != str(row["trace_id"]):
        out.append(
            Evidence(
                trace_id=str(longest["trace_id"]),
                label="longest rollout at the same step (for contrast)",
                detail=f"step {step}: output_tokens={int(longest['output_tokens'])}, reward={longest['reward']}",
            )
        )
    return out


def _from_logged_metric(ctx: AnalysisContext, logged: dict) -> Finding:
    """Fallback for metrics-only runs (TRL logs ``completions/mean_length``)."""
    series = pd.Series({int(k): float(v) for k, v in logged.items()}).sort_index()
    series = series[series.notna()]
    if len(series) < 3:
        return Rule.skipped("not enough length history", "Output length collapse", "length-collapse")
    base_n = min(ctx.config.baseline_steps, max(1, len(series) // 3))
    baseline = float(np.median(series.iloc[:base_n].to_numpy()))
    if baseline <= 0:
        return Rule.skipped("baseline length is zero", "Output length collapse", "length-collapse")
    ratio = series / baseline
    min_ratio = float(ratio.min())
    breach = ratio < ctx.config.length_drop_ratio
    first_breach = int(ratio[breach].index.min()) if breach.any() else None
    sustained = _max_consecutive(breach) >= 2
    depth = max(0.0, 1.0 - min_ratio) if breach.any() else 0.0
    severity = (
        Severity.CRITICAL
        if sustained and min_ratio < 0.4
        else Severity.WARN
        if sustained
        else Severity.INFO
        if breach.any()
        else Severity.OK
    )
    return Finding(
        rule_id="length-collapse",
        title="Output length collapse",
        severity=severity,
        headline=(
            f"Logged mean length fell to {min_ratio:.0%} of baseline ({baseline:.0f} tokens)"
            if severity is not Severity.OK
            else f"Logged mean length stayed at {min_ratio:.0%}+ of baseline ({baseline:.0f} tokens)"
        ),
        detail="Read from the trainer's own `completions/mean_length` metric; no per-sample lengths available.",
        metrics={
            "source": "trainer_metric",
            "baseline_median_length": round(baseline, 1),
            "min_ratio": round(min_ratio, 4),
            "first_breach_step": first_breach,
            "sustained": bool(sustained),
            "per_step": {int(k): round(float(v), 1) for k, v in series.items()},
        },
        actions=["Export completions to see which rollouts got short."] if severity is not Severity.OK else [],
        penalty=ctx.config.weight_length * depth * (1.0 if sustained else 0.5),
    )


__all__ = ["LengthCollapseRule", "pct"]
